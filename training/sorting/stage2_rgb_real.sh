#!/usr/bin/env bash
# TacImag | task: sorting | modality: rgb_real (stage 2 baseline: real tactile RGB sensor)
# usage: bash training/sorting/stage2_rgb_real.sh [seed] [hydra overrides...]
cd "$(dirname "$0")/../.."
python train_stage2.py tactile=rgb_real task=sorting training.seed=${1:-42} "${@:2}"
