#!/usr/bin/env bash
# TacImag stage-1 launcher.
#   bash scripts/train_stage1.sh --modality tacff --task usb --seed 42 [extra hydra overrides...]
set -euo pipefail

MODALITY=tacff
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

python train_stage1.py \
    modality="${MODALITY}" \
    task="${TASK}" \
    training.seed="${SEED}" \
    "${EXTRA[@]+"${EXTRA[@]}"}"
