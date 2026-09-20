#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "$script_dir/../.." && pwd)"
cd "$repository"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Run this workflow on the Linux hardware host, not the local host-test machine." >&2
    exit 2
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--build" ) ]]; then
    echo "Usage: bash $0 [--build]" >&2
    exit 2
fi

export TT_METAL_HOME="$repository"
export TT_METAL_RUNTIME_ROOT="$repository"
export PYTHONPATH="$repository:$repository/ttnn${PYTHONPATH:+:$PYTHONPATH}"
results="$(mktemp -d "${TMPDIR:-/tmp}/tt-repeat-direct.XXXXXX")"
echo "Results: $results"
trap 'echo "Artifacts retained: $results"' EXIT

run_logged() {
    local log_file="$1"
    shift
    if "$@" >"$log_file" 2>&1; then
        return 0
    else
        local status=$?
        echo "Stage failed (exit $status). Log: $log_file" >&2
        tail -n 60 "$log_file" >&2
        return "$status"
    fi
}

PYTHONPATH="$script_dir:$PYTHONPATH" python -c 'from repeat_candidate_probe import preflight; preflight()'

if [[ $# -eq 1 ]]; then
    echo "Building candidate; log: $results/build.log"
    run_logged "$results/build.log" ./build_metal.sh --build-tests
fi

echo "Running device correctness/cache/routing tests; log: $results/correctness.log"
run_logged "$results/correctness.log" \
    env -u TT_METAL_DEVICE_PROFILER -u TT_METAL_PROFILER_DIR -u TRACY_PORT \
    python -m pytest -q \
    tests/ttnn/unit_tests/operations/data_movement/test_repeat_interleave.py \
    tests/ttnn/nightly/unit_tests/operations/data_movement/test_repeat_interleave_codegen_routing.py \
    -k 'codegen or non_default_tile' --junitxml "$results/correctness.xml"

echo "Benchmarking six cases without device profiling; log: $results/unprofiled.log"
run_logged "$results/unprofiled.log" \
    env -u TT_METAL_DEVICE_PROFILER -u TT_METAL_PROFILER_DIR -u TRACY_PORT \
    python "$script_dir/repeat_candidate_probe.py" --warmup 10 --samples 51 \
    --output "$results/unprofiled.jsonl"

for case_name in aligned_h aligned_w gva_decode_h gva_prefill_h ragged_w; do
    case_dir="$results/$case_name"
    mkdir -p "$case_dir"
    echo "Capturing $case_name; log: $case_dir/capture.log"
    run_logged "$case_dir/capture.log" python -m tracy -p -r --check-exit-code -o "$case_dir" \
        "$script_dir/repeat_candidate_probe.py" --case "$case_name" --warmup 3 --samples 5 \
        --tracy-signposts --output "$case_dir/probe.jsonl"
    reports=("$case_dir"/reports/*/ops_perf_results_*.csv)
    if [[ ${#reports[@]} -ne 1 || ! -f "${reports[0]}" ]]; then
        echo "Expected exactly one operations CSV for $case_name" >&2
        exit 1
    fi
    python "$script_dir/summarize_repeat_capture.py" "${reports[0]}" --require-candidate \
        > "$case_dir/device_summary.json"
done

python "$script_dir/report_candidate_run.py" "$results" > "$results/SUMMARY.md"
cat "$results/SUMMARY.md"
echo "Validation completed. Review correctness, host distributions and device summaries before promoting routing."
