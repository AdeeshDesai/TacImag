#!/usr/bin/env bash
# TacImag | task: sorting | modality: ff_imag (stage 2 OURS: imagined force field (no sensor at deploy))
# usage: bash training/sorting/stage2_ff_imag.sh [seed] [hydra overrides...]
cd "$(dirname "$0")/../.."
python train_stage2.py tactile=ff_imag task=sorting training.seed=${1:-42} "${@:2}"
