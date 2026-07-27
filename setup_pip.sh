#!/usr/bin/env bash

# This script must be sourced:
#     source setup.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Run this script with: source setup.sh" >&2
    exit 1
fi

REPO_ROOT="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
    pwd
)"

VENV_DIR="${REPO_ROOT}/.venv"
REQUIREMENTS_FILE="${REPO_ROOT}/requirements.txt"

# Users can override this, for example:
# OPENFASTML_PYTHON=/path/to/python3.10 source setup.sh
PYTHON_BIN="${OPENFASTML_PYTHON:-python3.10}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python 3.10 was not found." >&2
    echo "Install Python 3.10 or set OPENFASTML_PYTHON." >&2
    return 1
fi

PYTHON_VERSION="$(
    "${PYTHON_BIN}" -c \
        'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)"

if [[ "${PYTHON_VERSION}" != "3.10" ]]; then
    echo "OpenFastML requires Python 3.10." >&2
    echo "Found Python ${PYTHON_VERSION} at: $(command -v "${PYTHON_BIN}")" >&2
    return 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Creating virtual environment at ${VENV_DIR}..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}" || return 1
fi

source "${VENV_DIR}/bin/activate" || return 1
hash -r

python -m pip install --upgrade pip setuptools wheel || return 1
python -m pip install --requirement "${REQUIREMENTS_FILE}" || return 1

python -m pip install -e "$REPO_ROOT" --no-deps

echo
echo "OpenFastML environment activated."
echo "Python: $(command -v python)"
