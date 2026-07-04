#!/usr/bin/env bash
# TacImag | task: gear | modality: rgb_imag (stage 2 OURS: imagined tactile RGB (no sensor at deploy))
# usage: bash training/gear/stage2_rgb_imag.sh [seed] [hydra overrides...]
cd "$(dirname "$0")/../.."
python train_stage2.py tactile=rgb_imag task=gear training.seed=${1:-42} "${@:2}"
