#!/bin/bash

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python nanorl/sac/run_control_suite.py \
    --root-dir ~/nanorl/runs/ \
    --warmstart-steps 5000 \
    --checkpoint_interval 10000 \
    --max-steps 1000000 \
    --discount 0.99 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar \
    --num_workers 1 \
    "$@"
    # --environment_name RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0
    # --init_from_checkpoint /data/nanorl/runs/SAC-RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0-42-1684391655.3145144/checkpoint_1000000 \
    # --offline_dataset /data/nanorl/runs/SAC-RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0-42-1684391655.3145144/replay_buffer.pkl
    # --domain-name cartpole \
    # --task-name swingup
