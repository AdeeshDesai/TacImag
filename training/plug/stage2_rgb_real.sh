#!/usr/bin/env bash
# TacImag | task: plug | modality: rgb_real (stage 2 baseline: real tactile RGB sensor)
# usage: bash training/plug/stage2_rgb_real.sh [seed] [hydra overrides...]
cd "$(dirname "$0")/../.."
python train_stage2.py tactile=rgb_real task=plug training.seed=${1:-42} "${@:2}"
