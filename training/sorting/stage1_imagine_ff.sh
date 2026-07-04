#!/usr/bin/env bash
# TacImag | task: sorting | modality: tacff (stage 1: imagine the contact force field)
# usage: bash training/sorting/stage1_imagine_ff.sh [seed] [hydra overrides...]
cd "$(dirname "$0")/../.."
python train_stage1.py modality=tacff task=sorting training.seed=${1:-42} "${@:2}"
