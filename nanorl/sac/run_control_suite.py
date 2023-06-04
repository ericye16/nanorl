"""Train a SAC agent on dm_control suite tasks."""

import time
from concurrent import futures
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Optional

import dm_env
import tyro

# from dm_control import suite
from robopianist import suite
from robopianist.wrappers import MidiEvaluationWrapper, PianoSoundVideoWrapper

from nanorl import SAC, SACConfig, replay, specs
from nanorl.infra import Experiment, eval_loop, seed_rngs, train_loop, wrap_env


@dataclass(frozen=True)
class Args:
    # Experiment configuration.
    root_dir: str = "/tmp/nanorl"
    """Where experiment directories are created."""
    seed: int = 42
    """RNG seed."""
    max_steps: int = 1_000_000
    """Total number of environment steps to train for."""
    warmstart_steps: int = 5_000
    """Number of steps in which to take random actions before starting training."""
    log_interval: int = 1_000
    """Number of steps between logging to wandb."""
    checkpoint_interval: int = -1
    """Number of steps between checkpoints and evaluations. Set to -1 to disable."""
    reset_interval: int = 200_000
    """Number of steps between resetting the policy."""
    eval_episodes: int = 10
    """Number of episodes to run at every evaluation."""
    batch_size: int = 256
    """Batch size for training."""
    discount: float = 0.99
    """Discount factor."""
    tqdm_bar: bool = False
    """Whether to use a tqdm progress bar in the training loop."""
    resets: bool = False
    """Whether to periodically reset the actor / critic layers."""
    init_from_checkpoint: Optional[str] = None
    """Path to a checkpoint to initialize the agent from."""
    ft: Optional[str] = None
    """Path to a checkpoint to initialize the agent from for finetuning."""
    num_workers: int = 1
    """Number of workers to use for parallel environment rollouts."""

    # Replay buffer configuration.
    replay_capacity: int = 1_000_000
    """Replay buffer capacity."""
    offline_dataset: Optional[str] = None
    """Path to a pickle file containing a list of transitions."""
    offline_pct: float = 0.5
    """Percentage of offline data to use."""

    # W&B configuration.
    use_wandb: bool = False
    project: str = "nanorl"
    entity: str = ""
    name: str = ""
    tags: str = ""
    notes: str = ""
    mode: str = "online"

    # Task configuration.
    pretrain_envs: List[str] = field(default_factory=lambda: [])
    eval_envs: List[str] = field(default_factory=lambda: [])
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = False

    # Environment wrapper configuration.
    frame_stack: int = 1
    """Number of frames to stack."""
    clip: bool = True
    """Whether to clip actions outside the canonical range."""
    record_dir: Optional[Path] = None
    """Where evaluation video renders are saved."""
    record_every: int = 1
    """How often to record videos."""
    camera_id: Optional[str | int] = 0
    """Camera to use for rendering."""
    action_reward_observation: bool = False
    """Whether to include the action and reward in the observation."""

    # SAC-specific configuration.
    agent_config: SACConfig = SACConfig()


def agent_fn(env: dm_env.Environment, *, args, training: bool = True) -> SAC:
    agent = SAC.initialize(
        spec=specs.EnvironmentSpec.make(env),
        config=args.agent_config,
        seed=args.seed,
        discount=args.discount,
    )

    if training:
        if args.init_from_checkpoint is not None:
            ckpt_exp = Experiment(Path(args.init_from_checkpoint)).assert_exists()
            agent = ckpt_exp.restore_checkpoint(agent)
        elif args.ft is not None:
            agent_copy = SAC.initialize(
                spec=specs.EnvironmentSpec.make(env),
                config=args.agent_config,
                seed=args.seed,
                discount=args.discount,
            )
            ckpt_exp = Experiment(Path(args.ft)).assert_exists()
            agent_copy = ckpt_exp.restore_checkpoint(agent_copy)
            agent = agent.replace(
                actor=agent_copy.actor,
                critic=agent_copy.critic,
                target_critic=agent.target_critic,
            )

    return agent


def replay_fn(env: dm_env.Environment, *, args) -> replay.ReplayBuffer:
    if args.offline_dataset is not None:
        offline_dataset = Path(args.offline_dataset)
        if not offline_dataset.exists():
            raise FileNotFoundError(f"Offline dataset {offline_dataset} not found.")
    else:
        offline_dataset = None

    utd_ratio = max(
        args.agent_config.critic_utd_ratio, args.agent_config.actor_utd_ratio
    )

    return replay.ReplayBuffer(
        capacity=args.replay_capacity // args.num_workers,
        batch_size=args.batch_size * utd_ratio,
        spec=specs.EnvironmentSpec.make(env),
        offline_dataset=offline_dataset,
        offline_pct=args.offline_pct,
    )


def env_fn(*, args, environment_name: str = None, record_dir: Optional[Path] = None) -> dm_env.Environment:
    env = suite.load(
        environment_name=environment_name,
        seed=args.seed,
        stretch=args.stretch_factor,
        shift=args.shift_factor,
        **dict(
            n_steps_lookahead=args.n_steps_lookahead,
            trim_silence=args.trim_silence,
            gravity_compensation=args.gravity_compensation,
            reduced_action_space=args.reduced_action_space,
            control_timestep=args.control_timestep,
            wrong_press_termination=args.wrong_press_termination,
            disable_fingering_reward=args.disable_fingering_reward,
            disable_forearm_reward=args.disable_forearm_reward,
            disable_colorization=args.disable_colorization,
            disable_hand_collisions=args.disable_hand_collisions,
            primitive_fingertip_collisions=args.primitive_fingertip_collisions,
            change_color_on_activation=True,
        ),
    )

    return wrap_env(
        env=env,
        record_dir=record_dir,
        record_every=args.record_every,
        frame_stack=args.frame_stack,
        clip=args.clip,
        camera_id=args.camera_id,
        action_reward_observation=args.action_reward_observation,
    )


def main(args: Args) -> None:
    run_name = "-".join(filter(bool, ["SAC", args.name, str(args.seed), str(time.time())]))

    # Seed RNGs.
    seed_rngs(args.seed)

    # Setup the experiment for checkpoints, videos, metadata, etc.
    experiment = Experiment(Path(args.root_dir) / run_name).assert_new()
    experiment.write_metadata("config", args)

    if args.use_wandb:
        experiment.enable_wandb(
            project=args.project,
            entity=args.entity or None,
            tags=(args.tags.split(",") if args.tags else []),
            notes=args.notes or None,
            config=asdict(args),
            mode=args.mode,
            name=run_name,
            sync_tensorboard=True,
        )

    pool = futures.ThreadPoolExecutor(1)
    # Run eval in a background thread. Eval continuously monitor for checkpoints and evaluates them
    pool.submit(
        eval_loop,
        experiment=experiment,
        env_fn=lambda env: MidiEvaluationWrapper(
            env_fn(args=args, environment_name=env),
            # PianoSoundVideoWrapper(
            #     env_fn(args=args, record_dir=experiment.data_dir / "videos"),
            #     record_every=1,
            #     camera_id="piano/back",
            #     record_dir=experiment.data_dir / "videos",
            # )
        ),
        agent_fn=partial(agent_fn, args=args, training=False),
        num_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        env_names=args.eval_envs,
    )

    train_loop(
        experiment=experiment,
        env_fns=[
            partial(env_fn, args=args, environment_name=env)
            for env in args.pretrain_envs
        ],
        agent_fn=partial(agent_fn, args=args),
        replay_fn=partial(replay_fn, args=args),
        max_steps=args.max_steps,
        warmstart_steps=args.warmstart_steps,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        resets=args.resets,
        reset_interval=args.reset_interval,
        tqdm_bar=args.tqdm_bar,
        num_workers=args.num_workers,
    )

    # Clean up.
    pool.shutdown()


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
