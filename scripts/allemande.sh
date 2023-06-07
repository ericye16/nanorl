#!/bin/bash

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python nanorl/sac/run_control_suite.py \
    --pretrain_envs RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0 \
    --eval_envs RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0 \
    --root-dir ~/cs224r/nanorl/runs/ \
    --name "Allemande" \
    --warmstart-steps 5000 \
    --max-steps 100000 \
    --discount 0.99 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 128 128 \
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
    --agent_config.num_qs 2 \
    --replay_capacity 1000000 \
    --checkpoint_interval 10000 \
    --log_interval 100 \
    --batch_size 256 \
    --agent_config.actor_lr 3e-4 \
    --agent_config.critic_lr 3e-4 \
    --agent_config.temp_lr 3e-4 \
    "$@"
