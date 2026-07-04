#!/usr/bin/env bash
# TacImag stage-2 launcher.
#   bash scripts/train_stage2.sh --modality ff_imag --task usb --seed 42 [extra hydra overrides...]
# Modalities: vision | rgb_real | rgb_imag | ff_real | ff_imag
set -euo pipefail

MODALITY=ff_imag
TASK=usb
SEED=42
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --modality) MODALITY="$2"; shift 2 ;;
        --task)     TASK="$2";     shift 2 ;;
        --seed)     SEED="$2";     shift 2 ;;
        *)          EXTRA+=("$1"); shift ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python train_stage2.py \
    tactile="${MODALITY}" \
    task="${TASK}" \
    training.seed="${SEED}" \
    "${EXTRA[@]+"${EXTRA[@]}"}"
