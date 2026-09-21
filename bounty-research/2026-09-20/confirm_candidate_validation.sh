#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repository="$(cd "$script_dir/../.." && pwd -P)"
cd "$repository"

if [[ "$(uname -s)" != "Linux" || $# -gt 1 || ( $# -eq 1 && "$1" != "--build" ) ]]; then
    echo "Run on the Linux hardware host: bash $0 [--build]" >&2
    exit 2
fi
python_executable="$repository/python_env/bin/python"
if [[ ! -x "$python_executable" || ! -f "$repository/python_env/bin/activate" ]]; then
    echo "This checkout needs its own prepared python_env; do not use another checkout's environment." >&2
    exit 2
fi

results="$(mktemp -d "${TMPDIR:-/tmp}/tt-repeat-confirm.XXXXXX")"
echo "Confirmation artifacts: $results"
trap 'echo "Confirmation artifacts retained: $results"' EXIT
directories=()
for run in 1 2 3; do
    run_parent="$results/run-$run"
    mkdir -p "$run_parent"
    validation_command=(bash "$script_dir/run_candidate_validation.sh")
    if [[ "$run" -eq 1 && $# -eq 1 ]]; then
        validation_command+=(--build)
    fi
    echo "Validation run $run/3; log: $run_parent/validation.log"
    if TMPDIR="$run_parent" "${validation_command[@]}" \
        > "$run_parent/validation.log" 2>&1; then
        matches=("$run_parent"/tt-repeat-direct.*)
        if [[ ${#matches[@]} -ne 1 || ! -d "${matches[0]}" ]]; then
            echo "Expected exactly one artifact directory for run $run" >&2
            exit 2
        fi
        directories+=("${matches[0]}")
    else
        status=$?
        tail -n 60 "$run_parent/validation.log" >&2
        echo "Run $run failed (exit $status); stopping without a passing confirmation." >&2
        exit "$status"
    fi
done

if env -u PYTHONHOME PYTHONNOUSERSITE=1 PYTHONPATH="$script_dir" \
    "$python_executable" "$script_dir/confirm_candidate_runs.py" "${directories[@]}" \
    --json-output "$results/CONFIRMATION.json" > "$results/CONFIRMATION.md"; then
    cat "$results/CONFIRMATION.md"
else
    status=$?
    cat "$results/CONFIRMATION.md"
    echo "Confirmation did not pass (exit $status). Keep all runs; do not select only the fastest." >&2
    exit "$status"
fi
