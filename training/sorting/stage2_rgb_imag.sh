#!/usr/bin/env bash
# TacImag | task: sorting | modality: rgb_imag (stage 2 OURS: imagined tactile RGB (no sensor at deploy))
# usage: bash training/sorting/stage2_rgb_imag.sh [seed] [hydra overrides...]
cd "$(dirname "$0")/../.."
python train_stage2.py tactile=rgb_imag task=sorting training.seed=${1:-42} "${@:2}"
