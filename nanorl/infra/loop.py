from multiprocessing import Pipe, Process, Queue
from queue import Empty
import time
from pathlib import Path
from typing import Any, Callable
from multiprocessing.connection import Connection

from jax import numpy as jnp
import dm_env
from robopianist import suite
import tqdm
from nanorl import agent, replay, specs

from nanorl.infra import Experiment, utils


EnvFn = Callable[[], dm_env.Environment]
AgentFn = Callable[[dm_env.Environment], agent.Agent]
ReplayFn = Callable[[dm_env.Environment], replay.ReplayBuffer]
LoggerFn = Callable[[], Any]


def environment_worker(env_fn: EnvFn, pipe: Connection, relabel: bool = False):
    env = env_fn()
    stuff = []
    timestep = env.reset()
    stuff.append((timestep, None))
    pipe.send(timestep)
    orig_rewards = 0
    while True:
        action = pipe.recv()
        timestep = env.step(action)
        orig_rewards += timestep.reward
        pipe.send(timestep)
        stuff.append((timestep, action))
        if timestep.last():
            stats = env.get_statistics()
            stats["total_reward"] = orig_rewards
            stats.update(env.get_musical_metrics())
            orig_rewards = 0
            actual_keys_played = env.task.actual_keys_played
            actual_sustain = env.task.actual_sustain
            actual_fingers = env.task.actual_fingers_used
            timestep = env.reset()
            pipe.send((stats, timestep))

            if relabel:
                replay_env = env_fn(replay_keys=(actual_keys_played, actual_sustain, actual_fingers))
                replay_timestep = replay_env.reset()
                replay_buffer = []
                relabel_rewards = 0
                # this one doesn't work for some reason
                replay_buffer.append((replay_timestep, None))
                # print(f"Expecting {len(stuff)} relabelled")
                # print(replay_timestep)
                count_relabelled = 0
                for replayed_timestep, action in stuff:
                    if action is None:
                        # print("Skipping none action")
                        continue
                    new_timestep = replay_env.step(action)
                    count_relabelled += 1
                    relabel_rewards += new_timestep.reward
                    replay_buffer.append((new_timestep, action))
                    # Maybe required to "reset" buffer?
                    if new_timestep.last():
                        if count_relabelled != len(stuff):
                            print(f"weird, relabelled length not matching at {count_relabelled}")
                        # replay_buffer.append((new_timestep, None))
                        break
                relabel_stats = {"total_rewards": relabel_rewards}
                try:
                    relabel_stats.update(replay_env.get_musical_metrics())
                except ValueError:
                    print("Couldn't get replay metrics?")
                pipe.send((replay_buffer, relabel_stats))
            stuff = []
            stuff.append((timestep, None))


def train_loop(
    experiment: Experiment,
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
    relabel: bool
) -> None:
    env = env_fn()
    agent = agent_fn(env)
    replay_buffers = [replay_fn(env) for _ in range(num_workers)]

    spec = specs.EnvironmentSpec.make(env)

    pipes, child_pipes = zip(*[Pipe() for _ in range(num_workers)])
    procs = [Process(target=environment_worker, args=(env_fn, pipe, relabel)) for pipe in child_pipes]
    for p in procs:
        p.start()

    timesteps = [pipe.recv() for pipe in pipes]
    for replay_buffer, ts in zip(replay_buffers, timesteps):
        replay_buffer.insert(ts, None)

    start_time = time.time()
    for step in tqdm.tqdm(range(1, max_steps + 1), disable=not tqdm_bar):
        if step < warmstart_steps // num_workers:
            actions = [spec.sample_action(random_state=env.random_state) for _ in timesteps]
        else:
            agent, actions = agent.sample_actions(jnp.stack([ts.observation for ts in timesteps]))

        for action, pipe in zip(actions, pipes):
            pipe.send(action)

        buff_idx = step % num_workers
        if step >= warmstart_steps // num_workers and replay_buffers[buff_idx].is_ready():
            transitions = replay_buffers[buff_idx].sample()
            agent, metrics = agent.update(transitions)
            if step % log_interval == 0:
                experiment.log(utils.prefix_dict("train", metrics), step=step)

            if checkpoint_interval >= 0 and step % checkpoint_interval == 0:
                print("Checkpointing!")
                experiment.save_checkpoint(agent, step=step)

            if step % log_interval == 0:
                experiment.log({"train/fps": int(step / (time.time() - start_time))}, step=step)

            if resets and step % reset_interval == 0:
                agent = agent_fn(env)

        timesteps = [pipe.recv() for pipe in pipes]
        for i in range(len(timesteps)):
            timestep, action = timesteps[i], actions[i]
            # print("timestep", timestep)
            # print("action shape 1", action.shape)
            replay_buffers[i].insert(timestep, action)

            if timestep.last():
                stats, timestep = pipes[i].recv()
                experiment.log(utils.prefix_dict("train", stats), step=step)
                # The order here is actually very important, must do this, then
                # add the timestep, None to the buffer
                if relabel:
                    replay_buffer_list, relabel_stats = pipes[i].recv()
                    assert replay_buffer_list[0][1] is None
                    for (ts, ac) in replay_buffer_list:
                        replay_buffers[i].insert(ts, ac)
                    experiment.log(utils.prefix_dict("train_relabel", relabel_stats), step=step)
                replay_buffers[i].insert(timestep, None)
                timesteps[i] = timestep
                                

    for proc in procs:
        proc.terminate()
    # Save final checkpoint and replay buffer.
    experiment.save_checkpoint(agent, step=max_steps, overwrite=True)
    utils.atomic_save(experiment.data_dir / "replay_buffer.pkl", replay_buffer.data)


def eval_loop(
    experiment: Experiment,
    env_fn: EnvFn,
    agent_fn: AgentFn,
    num_episodes: int,
    max_steps: int,
) -> None:
    env = env_fn()
    agent = agent_fn(env)

    last_checkpoint = None
    while True:
        # Wait for new checkpoint.
        checkpoint = experiment.latest_checkpoint()
        if checkpoint == last_checkpoint or checkpoint is None:
            time.sleep(10.0)
        else:
            # Restore checkpoint.
            agent = experiment.restore_checkpoint(agent)
            i = int(Path(checkpoint).stem.split("_")[-1])
            print(f"Evaluating checkpoint at iteration {i}")

            # Eval!
            for _ in range(num_episodes):
                timestep = env.reset()
                while not timestep.last():
                    timestep = env.step(agent.eval_actions(timestep.observation))

            # Log statistics.
            log_dict = utils.prefix_dict("eval", env.get_statistics())
            log_dict.update(utils.prefix_dict("eval_music", env.get_musical_metrics()))
            experiment.log(log_dict, step=i)

            # Maybe log video.
            experiment.log_video(env.latest_filename, step=i)

            print(f"Done evaluating checkpoint {i}")
            last_checkpoint = checkpoint

            # Exit if we've evaluated the last checkpoint.
            if i >= max_steps:
                print(f"Last checkpoint (iteration {i}) evaluated, exiting")
                break
