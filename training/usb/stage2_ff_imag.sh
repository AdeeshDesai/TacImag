#!/usr/bin/env bash
# TacImag | task: usb | modality: ff_imag (stage 2 OURS: imagined force field (no sensor at deploy))
# usage: bash training/usb/stage2_ff_imag.sh [seed] [hydra overrides...]
cd "$(dirname "$0")/../.."
python train_stage2.py tactile=ff_imag task=usb training.seed=${1:-42} "${@:2}"
