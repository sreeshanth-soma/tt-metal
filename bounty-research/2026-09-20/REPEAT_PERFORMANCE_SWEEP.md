# Repeat-interleave: broader performance screen

Prepared September 21, 2026, after the user-reported three-run confirmation in
`PACKED_ROW_CONFIRMATION.md`. **The broader screen has not been run on hardware.**
This update changes validation tools only, not the measured reader, device
tests, C++ dispatch or automatic public H/W routing.

## Fixed coverage

The manifest in `repeat_sweep_cases.py` defines **32 geometries x three dtypes =
96 cases**. It is a bounded screen, not an exhaustive cross product of all
supported parameters. Every listed geometry runs with BF16, FP32 and INT32:

- The five original H/W benchmark geometries, now in all three dtypes.
- Both axes at repeat counts 2, 3, 4, 7, 8, 16, 33 and 127, using rank-2
  ragged shapes across face/tile boundaries.
- Aligned W repeats 4 and 8, plus the original aligned W repeat 2.
- Rank-3 H and negative-axis W cases, and single-element H/W inputs.
- 71-, 72- and 73-output-tile cases, around the observed 8 x 9 core-grid
  partition boundary. These are fixed geometries, not claims about other grids.
- Larger aligned H/W cases, up to **16 MiB per padded output tensor**. This is
  not a bound on total device memory, native intermediate buffers, caches or
  profiler artifact storage.

Only the candidate's existing scope is exercised: default 32 x 32 TILE layout,
interleaved DRAM input/output, ranks 2–4, H/W repeats >= 2. No L1, sharded,
non-default-tile, unsupported-dtype or degenerate-input performance claim is
added. The existing device routing tests still check native fallback.

Preview the complete matrix without importing Torch/TTNN or opening a device:

```bash
python3 bounty-research/2026-09-20/repeat_candidate_probe.py --suite sweep --list
```

## What is measured

The existing default six-case probe and three-run workflow remain available;
`--suite sweep` explicitly opts into the new matrix. Each sweep case compares
the public operation with the private forced-codegen operation on the same
device input and checks both against `torch.repeat_interleave`, bit for bit,
before and after timing. Input transfers must also be bit-exact. INT32 inputs
span the signed 32-bit range without float conversion; FP32 inputs exercise
more mantissa bits than BF16. A native precision/support failure is not a
candidate speedup: it stops validation and retains the diagnostic evidence.

1. Run the existing focused device correctness/cache/routing tests once.
2. Run all 96 cases unprofiled: ten warmups and 51 samples per implementation.
3. Capture each case separately with Tracy: three warmups and five samples per
   implementation. Per-case captures preserve the existing bounded-profile
   workflow; they are not 96 cases in one potentially overflowing capture.
4. Re-parse raw JSONL, JUnit XML and every operations CSV. Do not trust cached
   timing summaries or ratios. Validate the fixed manifest, geometry, dtype,
   exact checks, clean sources, start/end source snapshots, reported runtime/
   device identity, distinct probe starts and complete sample indices.
5. Write `SWEEP.md` and `SWEEP.json`, retaining every case and every
   non-improvement. Host distributions remain separate from per-call sums of
   profiled device-kernel durations. Candidate captures must contain the new
   tiled-copy reader; native-versus-native comparisons cannot pass.

The benchmark alternates implementation order per measured sample. Compilation
warmups are excluded and outputs are explicitly released. Timing statistics
are recomputed from the raw samples; outliers are retained. The JSON report
includes all host ranges, device sample sums, per-case ratios/specifications
and hashes of the raw artifacts. Ratios <= 1 are non-improvements; a few wins
or an aggregate average cannot conceal a losing case.

## Hardware handoff

Use the **already prepared** `$HOME/tt-metal-repeat-direct` checkout on the
Linux TT host. Do not rerun setup, alter the original checkout, clear caches,
upgrade firmware or remove the earlier result directories.

The new files must first be committed and published to the development branch;
an old `git pull` cannot fetch uncommitted local work. Once this revision is
available there:

```bash
cd "$HOME/tt-metal-repeat-direct" &&
git pull --ff-only &&
bash bounty-research/2026-09-20/run_repeat_performance_sweep.sh
```

The already validated C++ reader is unchanged, so this workflow does not
request a rebuild by default. `--build` is available if a fresh build is
needed. The wrapper requires Linux and this checkout's own Python environment,
isolates library/import paths, rejects dirty tracked sources before building
or benchmarking, and unsets inherited profiler/Python-home settings. Do not
discard local changes to satisfy preflight; preserve them first.

This is one larger screen, not another three-run confirmation. It starts one
unprofiled probe and 96 separately profiled probes, in addition to the device
tests and any requested build. It can take substantially longer and generate
more artifacts than the six-case run; no hardware runtime estimate has been
measured locally. Avoid concurrent device workloads. Each stage logs progress
and preserves evidence in a new `/tmp/tt-repeat-sweep.*` directory.

On completion, share the printed `SWEEP.md`. Preserve `SWEEP.json`, the
manifest, logs and raw artifacts on the host. If a stage fails, share its
printed error and log path instead. Build, correctness or capture errors stop
the workflow; timing non-improvements do **not** stop collection early.

Report exit status meanings:

- **0:** complete evidence; all 96 cases improve both measured medians.
- **1:** complete evidence, but at least one case does not improve both medians.
  This is useful screening data, not a reason to drop that case and rerun only
  winners. `SWEEP.md` and `SWEEP.json` remain available.
- **2:** reporting refused invalid/incomplete evidence. Earlier stages retain
  their own failure codes, so identify the failing stage from the log rather
  than interpreting every shell exit as a performance finding.

Even status 0 is not routing approval. Expanded-matrix repeatability, an
evidence-backed narrow gate, architecture coverage, installed-wheel packaging,
project CI and model integration remain separate decisions. This screen does
not establish statistical significance, model speedup or bounty eligibility.

## Local verification

The host unit tests use synthetic probe/CSV fixtures and mocked shell/runtime
calls. They check coverage/memory bounds, dtype handling, refusal of changed or
partial evidence, retention of regressions and isolation/failure-stop behavior.
The existing guarded-reader sanitizer suite also remains applicable. These
tests do not execute the new sweep on a Tenstorrent device.

```bash
python3 -m unittest discover -s bounty-research/2026-09-20 -p 'test_*.py' -v
```

Current results and source hashes are recorded in `SWEEP_HOST_VALIDATION.json`
and `SWEEP_HOST_VALIDATION.txt`. The earlier `HOST_VALIDATION.json` retains the
previous 59-test local run and its historical tool hashes; it is not a claim
that those hashes describe this newer validation tooling.
