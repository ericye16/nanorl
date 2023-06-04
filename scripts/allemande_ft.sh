#!/bin/bash

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python nanorl/sac/run_control_suite.py \
    --init_from_checkpoint ~/cs224r/nanorl/runs/Allemande-Scales-Pretrain/checkpoint_1000000/ \
    --pretrain_envs RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0 \
    --eval_envs RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0 \
    --name "Allemande+Scales-Pretrain" \
    --root-dir ~/cs224r/nanorl/runs/ \
    --warmstart-steps 0 \
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
    --num_workers 10 \
    "$@"
