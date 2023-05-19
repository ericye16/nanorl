#!/bin/bash
#
# Run 1 seed of SAC on cartpole swingup.

WANDB_DIR=/data/nanorl/ MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python nanorl/sac/run_control_suite.py \
    --root-dir /data/nanorl/runs/ \
    --warmstart-steps 5000 \
    --checkpoint_interval 10000 \
    --max-steps 1000000 \
    --discount 0.99 \
    --n_seconds_lookahead 0.5 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --agent-config.activation "relu" \
    --action_reward_observation \
    --eval_episodes 1 \
    --reduced_action_space \
    --gravity_compensation \
    --tqdm-bar \
    --environment_name RoboPianist-debug-TwinkleTwinkleLittleStar-v0
    # --environment_name RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0
    # --init_from_checkpoint /data/nanorl/runs/SAC-RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0-42-1684391655.3145144/checkpoint_1000000 \
    # --offline_dataset /data/nanorl/runs/SAC-RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0-42-1684391655.3145144/replay_buffer.pkl
    # --domain-name cartpole \
    # --task-name swingup
