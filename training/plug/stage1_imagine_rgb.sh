#!/usr/bin/env bash
# TacImag | task: plug | modality: tacrgb (stage 1: imagine the tactile RGB image)
# usage: bash training/plug/stage1_imagine_rgb.sh [seed] [hydra overrides...]
cd "$(dirname "$0")/../.."
python train_stage1.py modality=tacrgb task=plug training.seed=${1:-42} "${@:2}"
