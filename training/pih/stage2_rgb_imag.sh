#!/usr/bin/env bash
# TacImag | task: pih | modality: rgb_imag (stage 2 OURS: imagined tactile RGB (no sensor at deploy))
# usage: bash training/pih/stage2_rgb_imag.sh [seed] [hydra overrides...]
cd "$(dirname "$0")/../.."
python train_stage2.py tactile=rgb_imag task=pih training.seed=${1:-42} "${@:2}"
