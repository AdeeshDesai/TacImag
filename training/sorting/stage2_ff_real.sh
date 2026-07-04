#!/usr/bin/env bash
# TacImag | task: sorting | modality: ff_real (stage 2 baseline: real force-field sensor)
# usage: bash training/sorting/stage2_ff_real.sh [seed] [hydra overrides...]
cd "$(dirname "$0")/../.."
python train_stage2.py tactile=ff_real task=sorting training.seed=${1:-42} "${@:2}"
