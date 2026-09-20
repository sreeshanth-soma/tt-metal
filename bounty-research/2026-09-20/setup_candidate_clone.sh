#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repository="$(cd "$script_dir/../.." && pwd -P)"
cd "$repository"

if [[ $# -ne 0 || "$(uname -s)" != "Linux" ]]; then
    echo "Run on the Linux hardware host from the newly cloned candidate branch: bash $0" >&2
    exit 2
fi
if [[ "$(git rev-parse --show-toplevel)" != "$repository" ]]; then
    echo "This script must stay inside its candidate Git checkout." >&2
    exit 2
fi
if [[ -e "$repository/python_env" || -L "$repository/python_env" ]]; then
    echo "Refusing to overwrite python_env. For an already prepared checkout, activate its environment and run:" >&2
    echo "bash bounty-research/2026-09-20/run_candidate_validation.sh --build" >&2
    exit 2
fi

PYTHONPATH="$script_dir" python3 -c 'from repeat_candidate_probe import preflight; preflight()'
export TT_METAL_HOME="$repository"
export TT_METAL_RUNTIME_ROOT="$repository"
export PYTHON_ENV_DIR="$repository/python_env"
export LD_LIBRARY_PATH="$repository/build/lib"
trap 'echo "Candidate checkout and logs retained: $repository"' EXIT

echo "Initializing this clone's submodules. Your existing tt-metal installation is not modified."
git submodule update --init --recursive
echo "Building the candidate; log: $repository/candidate_build.log"
if env -u VIRTUAL_ENV -u PYTHONPATH ./build_metal.sh --build-tests >candidate_build.log 2>&1; then
    echo "Build completed. Creating this clone's Python environment."
else
    build_status=$?
    tail -n 60 candidate_build.log >&2
    exit "$build_status"
fi
if env -u VIRTUAL_ENV -u PYTHONPATH ./create_venv.sh --env-dir "$PYTHON_ENV_DIR" >candidate_venv.log 2>&1; then
    echo "Python environment ready; log: $repository/candidate_venv.log"
else
    environment_status=$?
    tail -n 60 candidate_venv.log >&2
    exit "$environment_status"
fi
source "$PYTHON_ENV_DIR/bin/activate"
bash "$script_dir/run_candidate_validation.sh"
