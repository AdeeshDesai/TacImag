#!/usr/bin/env bash
# Download pretrained STAGE-1 imagination checkpoints from HuggingFace into
# data/checkpoints/stage1/ — after this, stage-2 *_imag training works with
# zero extra arguments (the configs' default checkpoint paths match).
#
#   bash scripts/download_ckpts.sh usb              # both modalities, one task
#   bash scripts/download_ckpts.sh usb plug         # several tasks
#   bash scripts/download_ckpts.sh all              # all six tasks
#
# No authentication needed (the checkpoint repo is public).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_REPO="Adeesh09/TacImag"

TASKS=("$@")
[ ${#TASKS[@]} -eq 0 ] && { echo "usage: $0 <task...|all>"; exit 1; }
[ "${TASKS[0]}" = "all" ] && TASKS=(usb plug pih gear bulb sorting)

PY="$(command -v python || command -v python3)"
"${PY}" -c 'import huggingface_hub' 2>/dev/null || {
    echo "ERROR: huggingface_hub not found — activate the TacImag conda environment first." >&2
    exit 1
}
for task in "${TASKS[@]}"; do
    for mod in tacff tacrgb; do
        dst="${REPO_ROOT}/data/checkpoints/stage1/${mod}_${task}/latest.ckpt"
        if [ -f "${dst}" ]; then
            echo "✓ ${mod}_${task} already present"
            continue
        fi
        echo "Downloading ${mod}_${task} ..."
        "${PY}" - "$task" "$mod" <<EOF
import os, sys
from huggingface_hub import hf_hub_download
task, mod = sys.argv[1], sys.argv[2]
hf_hub_download(
    repo_id="${HF_REPO}",
    repo_type="dataset",
    filename=f"checkpoints/stage1/{mod}_{task}/latest.ckpt",
    local_dir="${REPO_ROOT}/data",
    token=os.environ.get("HF_TOKEN"),
)
EOF
        echo "✓ data/checkpoints/stage1/${mod}_${task}/latest.ckpt"
    done
done
echo "Done. Stage-2 *_imag scripts will pick these up automatically."
