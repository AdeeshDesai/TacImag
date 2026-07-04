#!/usr/bin/env bash
# TacImag | task: plug | modality: tacff (stage 1: imagine the contact force field)
# usage: bash training/plug/stage1_imagine_ff.sh [seed] [hydra overrides...]
cd "$(dirname "$0")/../.."
python train_stage1.py modality=tacff task=plug training.seed=${1:-42} "${@:2}"
