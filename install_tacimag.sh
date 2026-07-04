#!/usr/bin/env bash
#
# TacImag setup: clones the third-party dependencies into thirdparty/ and
# installs the environment.
#
#   bash install_tacimag.sh
#
# Options (env vars):
#   SKIP_MANIFEEL_INSTALL=true    skip conda env creation (e.g. you already have it)
#   SKIP_ISAACGYM_DOWNLOAD=true   skip IsaacGym download (stage-1 only needs no sim)
#   TACIMAG_NONINTERACTIVE=true   accept ManiFeel installer defaults (no prompts)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
THIRDPARTY_ROOT="${REPO_ROOT}/thirdparty"
MANIFEEL_ROOT="${THIRDPARTY_ROOT}/manifeel"
ISAACGYM_TAR="${THIRDPARTY_ROOT}/IsaacGym_Preview_TacSL_Package.tar.gz"
ISAACGYM_DIR="${THIRDPARTY_ROOT}/IsaacGym_Preview_TacSL_Package"

MANIFEEL_REPO_URL="${MANIFEEL_REPO_URL:-https://github.com/purdue-mars/manifeel.git}"
ISAACGYM_GDRIVE_URL="${ISAACGYM_GDRIVE_URL:-https://drive.google.com/file/d/13dFRF9EXpzIWaJF2Z6f7BsuPUGQkPE8v/view?usp=sharing}"
SKIP_ISAACGYM_DOWNLOAD="${SKIP_ISAACGYM_DOWNLOAD:-false}"
SKIP_MANIFEEL_INSTALL="${SKIP_MANIFEEL_INSTALL:-false}"
TACIMAG_NONINTERACTIVE="${TACIMAG_NONINTERACTIVE:-true}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-tacimag}"

echo "=========================================="
echo "TacImag Setup"
echo "=========================================="
echo "Repo root:       ${REPO_ROOT}"
echo "Thirdparty root: ${THIRDPARTY_ROOT}"
echo ""

ensure_gdown() {
    if command -v gdown >/dev/null 2>&1; then return; fi
    echo "gdown not found. Installing with pip..."
    python3 -m pip install --user gdown
    export PATH="${HOME}/.local/bin:${PATH}"
    command -v gdown >/dev/null 2>&1 || {
        echo "ERROR: gdown installed but not on PATH. Try: export PATH=\"\${HOME}/.local/bin:\${PATH}\"" >&2
        exit 1
    }
}

mkdir -p "${THIRDPARTY_ROOT}"

echo "=========================================="
echo "1. Clone ManiFeel"
echo "=========================================="
if [ -d "${MANIFEEL_ROOT}/.git" ]; then
    echo "ManiFeel already exists: ${MANIFEEL_ROOT}"
elif [ -d "${MANIFEEL_ROOT}" ]; then
    echo "ERROR: ${MANIFEEL_ROOT} exists but is not a git repository." >&2
    exit 1
else
    git clone "${MANIFEEL_REPO_URL}" "${MANIFEEL_ROOT}"
fi

# Use our installer copy so the conda env is named "tacimag"
# (configurable via CONDA_ENV_NAME).
cp "${REPO_ROOT}/scripts/manifeel_codepatch/install.sh" "${MANIFEEL_ROOT}/install.sh"
chmod +x "${MANIFEEL_ROOT}/install.sh"
echo ""

echo "=========================================="
echo "2. Download IsaacGym TacSL Package"
echo "=========================================="
if [ "${SKIP_ISAACGYM_DOWNLOAD}" = "true" ]; then
    echo "Skipping (SKIP_ISAACGYM_DOWNLOAD=true) — stage-1 imagination training needs no simulator."
elif [ -d "${ISAACGYM_DIR}" ]; then
    echo "IsaacGym already extracted: ${ISAACGYM_DIR}"
else
    if [ ! -f "${ISAACGYM_TAR}" ]; then
        ensure_gdown
        # older gdown needs --fuzzy to accept share URLs; gdown >= 6 dropped
        # the flag (that behavior is the default)
        if gdown --help 2>/dev/null | grep -q -- "--fuzzy"; then
            gdown --fuzzy "${ISAACGYM_GDRIVE_URL}" -O "${ISAACGYM_TAR}"
        else
            gdown "${ISAACGYM_GDRIVE_URL}" -O "${ISAACGYM_TAR}"
        fi
    fi
    tar -xzf "${ISAACGYM_TAR}" -C "${THIRDPARTY_ROOT}"
fi
echo ""

echo "=========================================="
echo "3. Install ManiFeel environment"
echo "=========================================="
if [ "${SKIP_MANIFEEL_INSTALL}" = "true" ]; then
    echo "Skipping (SKIP_MANIFEEL_INSTALL=true)"
else
    # ManiFeel's installer prompts for the conda home. In non-interactive
    # mode, feed it empty lines so every prompt takes its detected default.
    # (Its CI=true flag is NOT used here: that also skips the IsaacGym and
    # isaacgymenvs installs, which real users need.)
    if [ "${TACIMAG_NONINTERACTIVE}" = "true" ]; then
        printf '\n\n\n\n\n\n\n\n\n\n' | CONDA_ENV_NAME="${CONDA_ENV_NAME}" bash "${MANIFEEL_ROOT}/install.sh"
    else
        CONDA_ENV_NAME="${CONDA_ENV_NAME}" bash "${MANIFEEL_ROOT}/install.sh"
    fi
fi
echo ""

echo "=========================================="
echo "4. Install TacImag package (editable)"
echo "=========================================="
# install into the conda env the installer created,
# not whatever python happens to be on PATH
CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null | tail -n 1 | awk '{print $NF}')"
fi
TACIMAG_ENV="${CONDA_ENV_PATH:-${CONDA_BASE}/envs/${CONDA_ENV_NAME}}"
if [ -x "${TACIMAG_ENV}/bin/pip" ]; then
    "${TACIMAG_ENV}/bin/pip" install -e "${REPO_ROOT}"
else
    echo "WARNING: conda env not found at ${TACIMAG_ENV}; installing with python3 -m pip"
    python3 -m pip install -e "${REPO_ROOT}"
fi
echo ""

echo "=========================================="
echo "TacImag setup complete"
echo "=========================================="
echo ""
echo "Activate the environment:"
echo "  conda activate ${CONDA_ENV_NAME}"
echo ""
echo "Quickstart:"
echo "  bash scripts/download_data.sh usb"
echo "  bash scripts/download_ckpts.sh usb"
echo "  bash training/usb/stage2_ff_imag.sh"
