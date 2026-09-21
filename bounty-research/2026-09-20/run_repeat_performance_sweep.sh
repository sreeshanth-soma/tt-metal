#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repository="$(cd "$script_dir/../.." && pwd -P)"
cd "$repository"

if [[ "$(uname -s)" != "Linux" || $# -gt 1 || ( $# -eq 1 && "$1" != "--build" ) ]]; then
    echo "Run on the Linux hardware host: bash $0 [--build]" >&2
    exit 2
fi

unset PYTHONHOME TT_METAL_DEVICE_PROFILER TT_METAL_PROFILER_DIR TRACY_PORT
export TT_METAL_HOME="$repository"
export TT_METAL_RUNTIME_ROOT="$repository"
export PYTHON_ENV_DIR="$repository/python_env"
export LD_LIBRARY_PATH="$repository/build/lib"
export PYTHONPATH="$repository:$repository/ttnn"
export PYTHONNOUSERSITE=1
python_executable="$PYTHON_ENV_DIR/bin/python"
if [[ ! -f "$PYTHON_ENV_DIR/bin/activate" || ! -x "$python_executable" ]]; then
    echo "This checkout needs its own prepared python_env; do not reuse another checkout's environment." >&2
    exit 2
fi
source "$PYTHON_ENV_DIR/bin/activate"
results="$(mktemp -d "${TMPDIR:-/tmp}/tt-repeat-sweep.XXXXXX")"
echo "Sweep artifacts: $results"
trap 'echo "Sweep artifacts retained: $results"' EXIT

run_logged() {
    local log_file="$1"
    shift
    if "$@" > "$log_file" 2>&1; then
        return 0
    else
        local status=$?
        echo "Stage failed (exit $status). Log: $log_file" >&2
        tail -n 60 "$log_file" >&2
        return "$status"
    fi
}

PYTHONPATH="$script_dir:$PYTHONPATH" "$python_executable" -c \
    'from repeat_candidate_probe import preflight, require_clean_state, source_state; preflight(); require_clean_state(source_state())'
"$python_executable" "$script_dir/repeat_candidate_probe.py" --suite sweep --list > "$results/manifest.json"
"$python_executable" -c \
    'import json, sys; print("\n".join(case["name"] for case in json.load(open(sys.argv[1]))["cases"]))' \
    "$results/manifest.json" > "$results/cases.txt"

if [[ $# -eq 1 ]]; then
    echo "Building candidate; log: $results/build.log"
    run_logged "$results/build.log" ./build_metal.sh --build-tests
fi

echo "Running device correctness/cache/routing tests; log: $results/correctness.log"
run_logged "$results/correctness.log" "$python_executable" -m pytest -q \
    tests/ttnn/unit_tests/operations/data_movement/test_repeat_interleave.py \
    tests/ttnn/nightly/unit_tests/operations/data_movement/test_repeat_interleave_codegen_routing.py \
    -k 'codegen or non_default_tile' --junitxml "$results/correctness.xml"

echo "Benchmarking all 96 cases without profiling; log: $results/unprofiled.log"
run_logged "$results/unprofiled.log" "$python_executable" "$script_dir/repeat_candidate_probe.py" \
    --suite sweep --warmup 10 --samples 51 --output "$results/unprofiled.jsonl"

while IFS= read -r case_name; do
    case_dir="$results/$case_name"
    mkdir -p "$case_dir"
    echo "Capturing $case_name; log: $case_dir/capture.log"
    run_logged "$case_dir/capture.log" "$python_executable" -m tracy -p -r --check-exit-code -o "$case_dir" \
        "$script_dir/repeat_candidate_probe.py" --suite sweep --case "$case_name" --warmup 3 --samples 5 \
        --tracy-signposts --output "$case_dir/probe.jsonl"
done < "$results/cases.txt"

if "$python_executable" "$script_dir/report_repeat_sweep.py" "$results" \
    --json-output "$results/SWEEP.json" > "$results/SWEEP.md" 2> "$results/report.log"; then
    cat "$results/SWEEP.md"
else
    status=$?
    cat "$results/SWEEP.md"
    cat "$results/report.log" >&2
    echo "Sweep report returned $status: 1 means non-improving cases; 2 means invalid/incomplete evidence." >&2
    exit "$status"
fi
