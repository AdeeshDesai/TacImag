#!/usr/bin/env bash
# Download the ManiFeel episode datasets from HuggingFace into data/.
# ONE dataset per task, shared by BOTH stages (it contains the camera obs,
# the tactile RGB, and the force field).
#
#   bash scripts/download_data.sh usb            # one task
#   bash scripts/download_data.sh usb plug pih   # several
#   bash scripts/download_data.sh all            # all six
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
HF_BASE="https://huggingface.co/datasets/purdue-mars/manifeel/resolve/main/data"

declare -A DS=(
    [usb]=usb_quan_Aug05
    [plug]=plug_quan_Aug02
    [pih]=pih_quan_June06
    [gear]=gear_quan_Sep15
    [bulb]=bulb_quan_Sep19
    [sorting]=sorting_quan_Aug8
)

TASKS=("$@")
[ ${#TASKS[@]} -eq 0 ] && { echo "usage: $0 <task...|all>   tasks: ${!DS[*]}"; exit 1; }
[ "${TASKS[0]}" = "all" ] && TASKS=(usb plug pih gear bulb sorting)

mkdir -p "${DATA_DIR}"
for task in "${TASKS[@]}"; do
    name="${DS[$task]:-}"
    [ -z "${name}" ] && { echo "unknown task: ${task} (tasks: ${!DS[*]})"; exit 1; }
    if [ -d "${DATA_DIR}/${name}/data" ]; then
        echo "✓ ${name} already present"
        continue
    fi
    echo "Downloading ${name}.zip ..."
    curl -L --fail -C - "${HF_BASE}/${name}.zip" -o "${DATA_DIR}/${name}.zip"
    echo "Extracting ..."
    unzip -q -o "${DATA_DIR}/${name}.zip" -d "${DATA_DIR}"
    rm -f "${DATA_DIR}/${name}.zip"
    echo "✓ data/${name}"
done
echo "Done. Both stages read from these directories (see tacimag/config/*/task/)."
