#!/bin/bash

train () {
    MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python nanorl/sac/run_control_suite.py \
        --environment_name "$1" \
        --root-dir ~/baseline2/runs/ \
        --name "$1" \
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
        --tqdm-bar
}

# train "RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0"
# train "RoboPianist-etude-12-FrenchSuiteNo5Sarabande-v0"
# train "RoboPianist-etude-12-PianoSonataD8451StMov-v0"
# train "RoboPianist-etude-12-PartitaNo26-v0"
# train "RoboPianist-etude-12-WaltzOp64No1-v0"
# train "RoboPianist-etude-12-BagatelleOp3No4-v0"
train "RoboPianist-etude-12-KreislerianaOp16No8-v0"
train "RoboPianist-etude-12-FrenchSuiteNo5Gavotte-v0"
train "RoboPianist-etude-12-PianoSonataNo232NdMov-v0"
# train "RoboPianist-etude-12-GolliwoggsCakewalk-v0"
# train "RoboPianist-etude-12-PianoSonataNo21StMov-v0"
# train "RoboPianist-etude-12-PianoSonataK279InCMajor1StMov-v0"