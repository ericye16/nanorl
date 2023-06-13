"""Train a SAC agent on dm_control suite tasks."""

import copy
from functools import partial
import time
import dataclasses
from dataclasses import asdict, dataclass
from concurrent import futures
from pathlib import Path
from typing import Optional
import dm_env
import tyro

# from dm_control import suite
from robopianist import suite

from nanorl import replay, specs
from nanorl import SAC, SACConfig
from nanorl.infra import seed_rngs, Experiment, train_loop, eval_loop, wrap_env
from robopianist.wrappers import PianoSoundVideoWrapper, MidiEvaluationWrapper


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
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
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
    print_fingers_used: bool = False
    relabel: bool = False
    use_tiered_reward: bool = False

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


def agent_fn(env: dm_env.Environment, *, args) -> SAC:
    agent = SAC.initialize(
        spec=specs.EnvironmentSpec.make(env),
        config=args.agent_config,
        seed=args.seed,
        discount=args.discount,
    )

    if args.init_from_checkpoint is not None:
        ckpt_exp = Experiment(Path(args.init_from_checkpoint)).assert_exists()
        agent = ckpt_exp.restore_checkpoint(agent)

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


def env_fn(*, args, record_dir: Optional[Path] = None, replay_keys = None) -> dm_env.Environment:
    env = suite.load(
        environment_name=args.environment_name,
        replay_keys=replay_keys,
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
            print_fingers_used=args.print_fingers_used,
            change_color_on_activation=True,
            use_tiered_reward=args.use_tiered_reward,
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
    if args.name:
        run_name = args.name
    else:
        run_name = f"SAC-{args.environment_name}-{args.seed}-key_press_change"
        if args.relabel:
            run_name += "-relabel"
        if args.use_tiered_reward:
            run_name += "-tiered_reward"
        run_name += f"-{time.time()}"

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

    # Run training in a background thread.
    pool.submit(
        train_loop,
        experiment=experiment,
        env_fn=lambda **kwargs: MidiEvaluationWrapper(env_fn(args=args, **kwargs)),
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
        relabel=args.relabel,
    )

    # Continuously monitor for checkpoints and evaluate.
    eval_loop(
        experiment=experiment,
        env_fn=lambda: MidiEvaluationWrapper(
            PianoSoundVideoWrapper(
                env_fn(args=args, record_dir=experiment.data_dir / "videos"),
                record_every=1,
                camera_id="piano/back",
                record_dir=experiment.data_dir / "videos",
            )
        ),
        agent_fn=partial(agent_fn, args=args),
        num_episodes=args.eval_episodes,
        max_steps=args.max_steps,
    )

    # Clean up.
    pool.shutdown()


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
