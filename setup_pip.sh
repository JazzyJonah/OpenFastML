#!/usr/bin/env bash

# Must be sourced:
#   source setup_pip.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Run this script with: source setup_pip.sh" >&2
    exit 1
fi

set -e

REPO_ROOT="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
    pwd
)"

VENV_DIR="${REPO_ROOT}/.venv"
UV_DIR="${REPO_ROOT}/.uv-bin"
UV_BIN="${UV_DIR}/uv"

# Keep uv's downloaded Python separate from Conda and the system Python.
export UV_PYTHON_INSTALL_DIR="${REPO_ROOT}/.uv-python"

###############################################################################
# Install uv locally if necessary
###############################################################################

if [[ ! -x "${UV_BIN}" ]]; then
    echo "uv was not found. Installing it locally..."

    mkdir -p "${UV_DIR}"

    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh |
            env UV_UNMANAGED_INSTALL="${UV_DIR}" sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh |
            env UV_UNMANAGED_INSTALL="${UV_DIR}" sh
    else
        echo "Neither curl nor wget was found." >&2
        echo "Install one of them and rerun: source setup_pip.sh" >&2
        return 1
    fi
fi

###############################################################################
# Download a managed, non-Conda Python 3.10
###############################################################################

echo "Ensuring Python 3.10 is available..."
"${UV_BIN}" python install 3.10

###############################################################################
# Detect an existing venv created from the wrong Python
###############################################################################

RECREATE_VENV=0

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    RECREATE_VENV=1
else
    VENV_VERSION="$(
        "${VENV_DIR}/bin/python" -c \
            'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
    )"

    VENV_BASE_PREFIX="$(
        "${VENV_DIR}/bin/python" -c \
            'import sys; print(sys.base_prefix)'
    )"

    if [[ "${VENV_VERSION}" != "3.10" ]]; then
        echo "Existing .venv uses Python ${VENV_VERSION}; recreating it."
        RECREATE_VENV=1
    elif [[ "${VENV_BASE_PREFIX}" != "${UV_PYTHON_INSTALL_DIR}"* ]]; then
        echo "Existing .venv was not created from the managed Python."
        echo "Current base: ${VENV_BASE_PREFIX}"
        echo "Recreating it to avoid Conda library contamination."
        RECREATE_VENV=1
    fi
fi

###############################################################################
# Create the virtual environment
###############################################################################

if [[ "${RECREATE_VENV}" -eq 1 ]]; then
    rm -rf "${VENV_DIR}"

    UV_MANAGED_PYTHON=1 "${UV_BIN}" venv \
        --python 3.10 \
        --seed \
        "${VENV_DIR}"
fi

###############################################################################
# Activate and install dependencies
###############################################################################

source "${VENV_DIR}/bin/activate"
hash -r

python -m pip install --upgrade pip setuptools wheel
python -m pip install --requirement "${REPO_ROOT}/requirements.txt"
python -m pip install --editable "${REPO_ROOT}" --no-deps

###############################################################################
# Verify the environment
###############################################################################

python - <<'PY'
import sys

print()
print("OpenFastML environment activated")
print("Python executable:", sys.executable)
print("Python base:      ", sys.base_prefix)
print("Python version:   ", sys.version.splitlines()[0])
PY