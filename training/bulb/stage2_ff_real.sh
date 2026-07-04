#!/usr/bin/env bash
# TacImag | task: bulb | modality: ff_real (stage 2 baseline: real force-field sensor)
# usage: bash training/bulb/stage2_ff_real.sh [seed] [hydra overrides...]
cd "$(dirname "$0")/../.."
python train_stage2.py tactile=ff_real task=bulb training.seed=${1:-42} "${@:2}"
