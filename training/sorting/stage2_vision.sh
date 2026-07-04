#!/usr/bin/env bash
# TacImag | task: sorting | modality: vision (stage 2 baseline: vision only)
# usage: bash training/sorting/stage2_vision.sh [seed] [hydra overrides...]
cd "$(dirname "$0")/../.."
python train_stage2.py tactile=vision task=sorting training.seed=${1:-42} "${@:2}"
