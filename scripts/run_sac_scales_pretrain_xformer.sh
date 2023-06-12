#!/bin/bash

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python nanorl/sac/run_control_suite.py \
    --pretrain_envs RoboPianist-debug-cmaj-v0 \
                    RoboPianist-debug-c#maj-v0 \
                    RoboPianist-debug-dmaj-v0 \
                    RoboPianist-debug-d#maj-v0 \
                    RoboPianist-debug-emaj-v0 \
                    RoboPianist-debug-fmaj-v0 \
                    RoboPianist-debug-f#maj-v0 \
                    RoboPianist-debug-gmaj-v0 \
                    RoboPianist-debug-g#maj-v0 \
                    RoboPianist-debug-amaj-v0 \
                    RoboPianist-debug-a#maj-v0 \
                    RoboPianist-debug-bmaj-v0 \
    --eval_envs RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0 \
                RoboPianist-debug-cmaj-v0 \
    --name "Xformer-Scales-Pretrain" \
    --root-dir ~/cs224r/nanorl/runs/ \
    --warmstart-steps 5000 \
    --checkpoint_interval 10000 \
    --max-steps 1000000 \
    --discount 0.99 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --agent-config.use_transformer \
    --agent-config.num_heads 4 \
    --replay_capacity 100000 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 100 \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar \
    --num_workers 1 \
    "$@"
