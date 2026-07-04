#!/usr/bin/env bash
# TacImag | task: plug | modality: rgb_imag (stage 2 OURS: imagined tactile RGB (no sensor at deploy))
# usage: bash training/plug/stage2_rgb_imag.sh [seed] [hydra overrides...]
cd "$(dirname "$0")/../.."
python train_stage2.py tactile=rgb_imag task=plug training.seed=${1:-42} "${@:2}"
