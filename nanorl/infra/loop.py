import time
import traceback
from copy import copy
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import dm_env
import tqdm

from nanorl import agent, replay, specs
from nanorl.infra import Experiment, utils

EnvFn = Callable[[], dm_env.Environment]
AgentFn = Callable[[dm_env.Environment], agent.Agent]
ReplayFn = Callable[[dm_env.Environment], replay.ReplayBuffer]
LoggerFn = Callable[[], Any]


def environment_worker(env_fn: EnvFn, env_name: str, pipe: Connection):
    env = env_fn(environment_name=env_name)
    timestep = env.reset()
    pipe.send(timestep)
    while True:
        action = pipe.recv()
        timestep = env.step(action)
        pipe.send(timestep)
        if timestep.last():
            stats = env.get_statistics()
            timestep = env.reset()
            pipe.send((stats, timestep))


def train_loop(
    experiment: Experiment,
    env_names: Sequence[str],
    env_fn: EnvFn,
    agent_fn: AgentFn,
    replay_fn: ReplayFn,
    max_steps: int,
    warmstart_steps: int,
    log_interval: int,
    checkpoint_interval: int,
    resets: bool,
    reset_interval: int,
    tqdm_bar: bool,
    num_workers: int,
) -> None:
    env: dm_env.Environment = env_fn(environment_name=env_names[0])
    agent = agent_fn(env)
    spec = specs.EnvironmentSpec.make(env)

    num_workers = num_workers * len(env_names)
    replay_buffers = [replay_fn(env) for _ in range(num_workers)]
    pipes, child_pipes = zip(*[Pipe() for _ in range(num_workers)])
    procs = [
        Process(
            target=environment_worker,
            args=(env_fn, env_names[i % len(env_names)], pipe)
        ) for i, pipe in enumerate(child_pipes)
    ]
    for p in procs:
        p.start()

    timesteps = [pipe.recv() for pipe in pipes]
    for replay_buffer, ts in zip(replay_buffers, timesteps):
        replay_buffer.insert(ts, None)

    start_time = time.time()
    for step in tqdm.tqdm(range(0, max_steps), disable=not tqdm_bar):
        if step % len(timesteps) == 0 or step < warmstart_steps:
            if step < warmstart_steps:
                actions = [spec.sample_action(random_state=env.random_state) for _ in timesteps]
            else:
                agent, actions = agent.sample_actions(replay.stack_obs([ts.observation for ts in timesteps], axis=0))

            for action, pipe in zip(actions, pipes, strict=True):
                pipe.send(action)

        if step >= warmstart_steps and (replay_buffer := replay_buffers[step % len(replay_buffers)]).is_ready():
            transitions = replay_buffer.sample()
            agent, metrics = agent.update(transitions)

            if step % log_interval == 0:
                experiment.log(utils.prefix_dict(f"train/{env_names[step % len(env_names)]}", metrics), step=step)

            if checkpoint_interval >= 0 and step % checkpoint_interval == 0:
                print("Checkpointing!")
                experiment.save_checkpoint(agent, step=step)

            if step % log_interval == 0:
                experiment.log({"train/fps": int(step / (time.time() - start_time))}, step=step)

        if resets and step % reset_interval == 0:
            agent = agent_fn(env)

        if (step + 1) % len(timesteps) == 0 or step + 1 < warmstart_steps:
            timesteps = [pipe.recv() for pipe in pipes]
            for i in range(len(timesteps)):
                timestep, action = timesteps[i], actions[i]
                replay_buffers[i].insert(timestep, action)

                if timestep.last():
                    stats, timestep = pipes[i].recv()
                    experiment.log(utils.prefix_dict(f"train/{env_names[i % len(env_names)]}", stats), step=step)
                    replay_buffers[i].insert(timestep, None)
                    timesteps[i] = timestep

    for proc in procs:
        proc.terminate()
    # Save final checkpoint and replay buffer.
    experiment.save_checkpoint(agent, step=max_steps, overwrite=True)
    utils.atomic_save(experiment.data_dir / "replay_buffer.pkl", replay_buffer.data)


def eval(
    experiment: Experiment,
    checkpoint: str,
    agent: agent.Agent,
    envs: Sequence[dm_env.Environment],
    num_episodes: int,
    env_names: Sequence[str],
    ckpt_exp: Optional[Experiment] = None,
):
    # Restore checkpoint.
    if ckpt_exp is None:
        ckpt_exp = experiment
    agent = ckpt_exp.restore_checkpoint(agent)
    i = int(Path(checkpoint).stem.split("_")[-1])
    print(f"Evaluating checkpoint at iteration {i}")

    # Eval!
    for _ in range(num_episodes):
        envs_copy = copy(envs)
        timesteps = [env.reset() for env in envs_copy]
        while envs_copy:
            actions = agent.eval_actions(replay.stack_obs([ts.observation for ts in timesteps], axis=0))
            new_timesteps = []
            new_envs = []
            for env, timestep, action in zip(envs_copy, timesteps, actions, strict=True):
                timestep = env.step(action)
                if not timestep.last():
                    new_timesteps.append(timestep)
                    new_envs.append(env)
            timesteps = new_timesteps
            envs_copy = new_envs

    for env, env_name in zip(envs, env_names, strict=True):
        # Log statistics.
        log_dict = utils.prefix_dict(f"eval/{env_name}", env.get_statistics())
        log_dict.update(utils.prefix_dict(f"eval_music/{env_name}", env.get_musical_metrics()))
        experiment.log(log_dict, step=i)

        # Maybe log video.
        if hasattr(env, 'latest_filename'):
            experiment.log_video(env.latest_filename, step=i)
    return i


def eval_loop(
    experiment: Experiment,
    env_fn: Callable[[str], dm_env.Environment],
    agent_fn: AgentFn,
    num_episodes: int,
    max_steps: int,
    env_names: Sequence[str],
) -> None:
    try:
        envs = [env_fn(env_name) for env_name in env_names]
        agent = agent_fn(envs[0])

        last_checkpoint = None
        while True:
            # Wait for new checkpoint.
            checkpoint = experiment.latest_checkpoint()
            if checkpoint == last_checkpoint or checkpoint is None:
                time.sleep(10.0)
            else:
                i = eval(
                    experiment=experiment,
                    checkpoint=checkpoint,
                    agent=agent,
                    envs=envs,
                    num_episodes=num_episodes,
                    env_names=env_names,
                )
                print(f"Done evaluating checkpoint {i}")
                last_checkpoint = checkpoint

                # Exit if we've evaluated the last checkpoint.
                if i >= max_steps:
                    print(f"Last checkpoint (iteration {i}) evaluated, exiting")
                    break
    except Exception:
        traceback.print_exc()
        raise
