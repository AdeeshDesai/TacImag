#!/usr/bin/env bash
# Run a stage across ALL tasks (usb plug pih gear bulb sorting), sequentially.
#
#   bash scripts/run_all.sh stage1 tacff              # imagine FF, all tasks
#   bash scripts/run_all.sh stage1 tacrgb 43          # seed override
#   bash scripts/run_all.sh stage2 ff_imag            # ours, all tasks
#   bash scripts/run_all.sh stage2 all                # all 5 modalities x all tasks
#
#   stage1 modalities: tacff | tacrgb
#   stage2 modalities: vision | rgb_real | ff_real | rgb_imag | ff_imag | all
#
# Extra hydra overrides pass through, e.g.:
#   bash scripts/run_all.sh stage2 ff_imag 42 logging.mode=offline
set -euo pipefail

STAGE=${1:?usage: run_all.sh <stage1|stage2> <modality|all> [seed] [overrides...]}
MODALITY=${2:?missing modality}
SEED=${3:-42}
EXTRA=("${@:4}")

TASKS=(usb plug pih gear bulb sorting)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

run() {  # run <stage> <modality> <task>
    local stage=$1 mod=$2 task=$3
    echo "=================================================================="
    echo " ${stage} | modality=${mod} | task=${task} | seed=${SEED}"
    echo "=================================================================="
    if [ "${stage}" = "stage1" ]; then
        python train_stage1.py modality="${mod}" task="${task}" \
            training.seed="${SEED}" "${EXTRA[@]+"${EXTRA[@]}"}"
    else
        python train_stage2.py tactile="${mod}" task="${task}" \
            training.seed="${SEED}" "${EXTRA[@]+"${EXTRA[@]}"}"
    fi
}

if [ "${STAGE}" = "stage2" ] && [ "${MODALITY}" = "all" ]; then
    for mod in vision rgb_real ff_real rgb_imag ff_imag; do
        for task in "${TASKS[@]}"; do run stage2 "${mod}" "${task}"; done
    done
elif [ "${STAGE}" = "stage1" ] && [ "${MODALITY}" = "all" ]; then
    for mod in tacff tacrgb; do
        for task in "${TASKS[@]}"; do run stage1 "${mod}" "${task}"; done
    done
else
    for task in "${TASKS[@]}"; do run "${STAGE}" "${MODALITY}" "${task}"; done
fi

echo "ALL RUNS DONE (${STAGE}, ${MODALITY}, seed ${SEED})"
