#!/usr/bin/env bash
# TacImag | task: gear | modality: vision (stage 2 baseline: vision only)
# usage: bash training/gear/stage2_vision.sh [seed] [hydra overrides...]
cd "$(dirname "$0")/../.."
python train_stage2.py tactile=vision task=gear training.seed=${1:-42} "${@:2}"
