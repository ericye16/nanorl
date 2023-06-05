#!/bin/bash

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python nanorl/sac/run_control_suite.py \
    --ft /home/matt/cs224r/nanorl/runs/SAC-PIG-Pretrain-42-1685669585.6487474/checkpoint_28000 \
    --pretrain_envs RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0 \
                RoboPianist-etude-12-FrenchSuiteNo5Sarabande-v0 \
                RoboPianist-etude-12-PianoSonataD8451StMov-v0 \
                RoboPianist-etude-12-PartitaNo26-v0 \
                RoboPianist-etude-12-WaltzOp64No1-v0 \
                RoboPianist-etude-12-BagatelleOp3No4-v0 \
                RoboPianist-etude-12-KreislerianaOp16No8-v0 \
                RoboPianist-etude-12-FrenchSuiteNo5Gavotte-v0 \
                RoboPianist-etude-12-PianoSonataNo232NdMov-v0 \
                RoboPianist-etude-12-GolliwoggsCakewalk-v0 \
                RoboPianist-etude-12-PianoSonataNo21StMov-v0 \
                RoboPianist-etude-12-PianoSonataK279InCMajor1StMov-v0 \
    --eval_envs RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0 \
                RoboPianist-etude-12-FrenchSuiteNo5Sarabande-v0 \
                RoboPianist-etude-12-PianoSonataD8451StMov-v0 \
                RoboPianist-etude-12-PartitaNo26-v0 \
                RoboPianist-etude-12-WaltzOp64No1-v0 \
                RoboPianist-etude-12-BagatelleOp3No4-v0 \
                RoboPianist-etude-12-KreislerianaOp16No8-v0 \
                RoboPianist-etude-12-FrenchSuiteNo5Gavotte-v0 \
                RoboPianist-etude-12-PianoSonataNo232NdMov-v0 \
                RoboPianist-etude-12-GolliwoggsCakewalk-v0 \
                RoboPianist-etude-12-PianoSonataNo21StMov-v0 \
                RoboPianist-etude-12-PianoSonataK279InCMajor1StMov-v0 \
    --root-dir ~/cs224r/nanorl/runs/ \
    --name "Finetune-All-Etudes" \
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
    --num_workers 1 \
    --update_period 10 \
    --agent_config.num_qs 2 \
    --replay_capacity 1000000 \
    --log_interval 10 \
    --batch_size 256 \
    --agent_config.actor_lr 3e-4 \
    --agent_config.critic_lr 3e-4 \
    --agent_config.temp_lr 3e-4 \
    --eval_episodes 10
