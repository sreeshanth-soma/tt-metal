# Repeat-interleave: implementation and acceptance package

Updated: **September 21, 2026**. Base: `9173350554b616022b3aa7c4fbad33f13cb36aee`.

## Bottom line

The submission worth pursuing is a **measured, narrowly routed, tile-preserving copy implementation of H/W repeat-interleave**. It is not a universal dense-matmul replacement and not a redo of the already-merged repeat codegen port.

The packed-row candidate completed the user's Wormhole build and validation at commit `0f7d9dfdacd347baf3f3cdd0fd7b75dc365c55a1`: **177 device tests passed**, and **all five H/W cases improved both unprofiled host latency and profiled device-kernel sums**. The reported ratios are 3.023x–4.048x for host latency and 1.821x–2.236x for device-kernel sums. See `PACKED_ROW_HARDWARE.md` for the measurements, source hash and provenance limits. This is one successful run, not a universal or production-routing claim. The validation-only follow-up keeps the measured kernel unchanged and checks three complete runs before any routing decision. The H/W path remains accessible through the existing private forced-codegen entry only; public routing stays native.

The earlier scalar reader passed 135 tests but regressed device time in four of five cases. Its separate evidence remains in `DIRECT_TILE_V1_HARDWARE.md`; do not combine those measurements with the packed-row result.

The candidate is on the development branch `perf/repeat-interleave-tile-copy` for hardware validation. No issue, assignment request, bounty claim or PR has been submitted. No changes were made to the earlier hardware installation.

## What maintainers would need

These are acceptance prerequisites and relevant review precedents, not a promise from a maintainer to accept this candidate.

| Requirement | Evidence or precedent | Current status |
| --- | --- | --- |
| An issue with appropriate scope and reviewers | `CONTRIBUTING.md:61` requires an issue and an approved PR | Not created or approved |
| Extend current code rather than duplicate a recent port | Merged PR `tenstorrent/tt-metal#50700`; current repeat codegen gate defers H/W | Reuses that operation, factory and writer |
| A support gate that matches real kernel/CB capability | PR #50700 review discussion `3702406157` / `3703441078` identified oversized CB allocation | Explicit layout, storage, padding, geometry and CB bounds |
| Positive forced-path tests, not native-versus-native evidence | PR #50700 review discussion `3702406210` / `3703442203` | New positive H/W cases and bitwise CPU reference checks |
| Rebound input/output buffers on cache hits | Same review required a second dispatch with fresh allocations | Tests retain both inputs and outputs and check cache count |
| Installed JIT source availability | PR #50700 review discussion `3702406187` / `3703439258` found a packaging omission | New reader matches existing CMake `*.cpp` kernel glob; a fresh configure is required |
| No dead or redundant kernel code | PR #50700 review discussion `3751739682` / `3756762075` | One reader, no new public API; unnecessary second scratch page removed |
| Actual performance evidence without regressions | Performance category in `CONTRIBUTING.md:160` | Packed-row run improves both medians in five BF16 cases; repeated runs and broader coverage remain pending |
| Required CI, beyond initial PR checks | `CONTRIBUTING.md:134` | Not run; maintainer-triggered coverage may be needed |

The title/description searches found no open PR explicitly implementing this generic tiled H/W path at the audit time. That is **not proof of exclusivity** or permission to claim someone else's issue. Model-specific repeat elimination, concat ports, and matmul tuning are separate overlapping areas to avoid packaging into this change.

Bounty eligibility is separate from technical acceptance. The published terms inspected on September 20 require a merged contribution addressing an appropriately tagged open bounty issue, assignment at PR submission, and the remaining program conditions. This candidate has no confirmed bounty or assignment. The repository's AI restrictions (`CONTRIBUTING.md:584`) prohibit automated/AI-generated assignment claims; any submission requires human review and responsibility. This document is an internal implementation checklist, not an assignment-request template.

The official terms were rechecked on September 21. Exhibit A lists $501–$1,999 for medium and $2,000–$3,000 for hard tasks, with performance work among the examples. Those are program categories, **not a reward assigned to this candidate**. No payment is promised, and this work could remain an unpaid contribution. Before treating more hardware or integration effort as paid work, the human contributor should establish maintainer interest, approved issue scope and actual bounty eligibility; more benchmark runs cannot establish any of those.

## Why the matmul experiment is not the solution

The user's earlier **unmodified, older hardware checkout** was `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9`, not this base. Its results motivate the design but do not measure the new code.

| Earlier profiled case | Public operation sequence | Public median kernel-sum | Selector median kernel-sum |
| --- | --- | ---: | ---: |
| `aligned_w` | transpose, untilize, concat, tilize, transpose | 20.397 µs | 14.823 µs |
| `gva_prefill_h` | untilize-with-unpadding, concat, tilize | 49.278 µs | 565.617 µs |

The width kernel-sum comparison is about **1.376×**, not the roughly 3× host-latency ratio. Prefill's selector uses two cores and is about **11.478× slower by kernel-sum**. A matmul has arithmetic/nonfinite-value semantics and selector allocation that a copy should not need. See `HARDWARE_RESULTS.md` and the two `*_profile_observed.json` files for provenance and the retained outliers.

## Implemented scope

- Rank 2–4; repeated H or W; positive or negative dimension spelling.
- BF16, FP32 and INT32, copied as raw bits. No arithmetic on tensor values.
- Default untransposed 32×32 TILE layout with canonical padding.
- Interleaved **DRAM input and output** only for the new H/W reader.
- Repeats greater than one; nonempty tensors; representable output dimensions and page count.
- Other tile shapes, L1 I/O, sharded I/O, unsupported dtypes and degenerate cases retain native handling. Existing outer-TILE and ROW_MAJOR codegen implementations are not expanded.
- The private forced-codegen call fails outside its support gate; it never silently runs native.
- Public routing checks the H/W holdback before the new geometry/L1 checks, avoiding that extra work on the unchanged native route.

The production/device-test changes modify six existing files and add one reader. Research, mocks and benchmark scripts are separate. The earlier standalone `core.patch` and archive, if present locally, are historical artifacts and do not contain the packed-row revision; use the Git branch instead.

### Dataflow and geometry

Each core owns a contiguous range of output tiles. It reads one source tile into a scratch CB, rearranges its raw 16-bit/32-bit elements into the output CB, and uses the existing batched interleaved writer. There is no compute kernel, dense selector, transpose, untilize, concat or tilize in the candidate device operation.

For tile coordinate `q`, element offset `offset` in `[0,31]`, and integer repeat count `repeats`:

```text
source_element = floor((32*q + offset) / repeats)
source_tile    = floor(q / repeats)
```

The source tile is constant across all 32 output offsets: writing `q = repeats*block + remainder` bounds `32*remainder + offset` below `32*repeats`. **A second source tile is never needed.** Source-tile boundaries expand to output boundaries divisible by 32. The source tile can also be reused while consecutive output tiles refer to it.

Face layout is explicit: four 16×16 faces in each 32×32 tile. BF16 values are packed into 32-bit stores without arithmetic conversion. Invalid output rows/columns are zero-filled rather than populated from input padding.

The packed-row revision loads an H face row once and reuses its words across consecutive repeated output rows, stopping at face boundaries and logical padding. Aligned W repeats of 2, 4 and 8 load contiguous words and replicate their integer bit fields; other W repeat counts retain an explicit column map. Face-row loads/stores are unrolled, and NoC barriers run only after an actual scratch-tile read. A masked tail prevents poisoned input padding from appearing in the output. This changes neither supported layouts/dtypes nor the CB allocation or public routing gate.

The output page count is recomputed from the **repeated logical shape**, then rounded to tiles. It is not `input_tiles * repeats`. For example, H=8 repeated four times gives H=32 and still one tile row. The same geometry helper is used by the support gate and dispatch helper; it checks canonical padding and overflow before allocation.

CB staging per core is bounded:

- Eight output tiles, preserving the existing writer's `depth >= 2*batch` protocol with batch four.
- One scratch tile.
- **18 KiB for BF16; 36 KiB for FP32/INT32**, excluding code, stack and runtime-reserved memory.
- The new gate checks available L1 for those nine pages and checks the input buffer's page pitch.

On the observed 8×9 grid, output-page splitting assigns the width case's 64 tiles to 64 cores and prefill's 256 tiles across 72 cores. These are factory-derived work assignments, not measured utilization or performance claims.

## Validation

### Completed locally

- Compile and execute the **actual reader source**, not a Python translation, for 32 compile-time variants: two element widths × two axes × eight repeat counts (2, 3, 4, 7, 8, 16, 33, 127).
- **8,032 geometry/data cases**, including face/tile boundaries, ragged shapes, multiple batches, uneven core partitions, the supplied width/prefill geometries, repeated calls with changed data, and poisoned input padding.
- AddressSanitizer and UndefinedBehaviorSanitizer, `-Wall -Wextra -Werror`.
- Guarded scratch/output allocations, current-reservation checks on every write, delayed mock NoC reads until the barrier, batched writer backpressure/ring-wrap checks, raw special-value bit patterns, and exact padded-output comparison.
- Mock access counters require exactly one store per output word, packed source loads for H and aligned W, and source-word reuse for aligned W. They are structural checks, **not device timing estimates**.
- The host suite also covers profiler signposts, alternated benchmark order, CLI preflight, old-checkout rejection, committed-descendant acceptance, simulation/debug-mode rejection, CSV aggregation, report identity checks, environment isolation, failure-stop behavior, and three-run confirmation. Confirmation fixtures reject copied or mismatched runs, insufficient samples, skips, dirty tracked sources, and any target-case median regression. These fixtures are not hardware measurements. `HOST_VALIDATION.json` and `HOST_VALIDATION.txt` record the current run's counts, hashes and formatting/syntax checks.

These are **host mocks**, not official tt-emule or Tenstorrent hardware. They do not validate actual NoC transactions, the complete TTNN C++ build, physical allocator behavior, real cache rebinding, device compiler ABI, or speed. `HOST_VALIDATION.json` records the final source hash and results.

### Device evidence and repeatability

The first 81 added device-test instances were included in the user's passing 135-test run at commit `8d601ab0ee2e7bd4383fd347b4ac6aecf820d912`. The packed-row revision adds 42 more instances: ragged/single-element W repeats, an R=16 general-path case, and special-value bit patterns for R=2/4/8. There are now 123 added instances relative to the pinned base; existing routing tests continue to check native fallback. **The user's second run passed all 177 focused tests at commit `0f7d9dfdacd347baf3f3cdd0fd7b75dc365c55a1`, with no skips/failures/errors.**

The local environment is macOS without a TT device or Torch/TTNN runtime. The second supplied summary reports the packed-row Linux build, hardware correctness and five-case performance result; its raw artifacts remain on the user's hardware host and have not been independently re-parsed locally. The three-run workflow has not yet been executed on that hardware. Broader shape/dtype performance, Blackhole, installed-wheel packaging, project CI and model-level performance remain unverified.

## Hardware handoff: repeated confirmation

For the already prepared candidate checkout, **do not clone again or rerun setup**:

```bash
cd "$HOME/tt-metal-repeat-direct" &&
git pull --ff-only &&
bash bounty-research/2026-09-20/confirm_candidate_validation.sh --build
```

The confirmation wrapper builds once, then runs three complete validations in separate child processes. It activates the prepared candidate environment through the existing runner and saves all three runs under a new `/tmp/tt-repeat-confirm.*` directory. Keep the previous `/tmp/tt-repeat-direct.5cmJXI` and `/tmp/tt-repeat-direct.xLpGzF` directories unchanged. Only the final `CONFIRMATION.md` needs to be shared; keep `CONFIRMATION.json` and all raw artifacts alongside it.

The report requires matching commit/source/runtime/device identity, a clean tracked checkout, distinct timezone-aware probe start times, the same executed test counts with zero skips, at least 51 host samples and five device samples per leg, and equal sample counts across runs. It re-parses the raw probes/CSVs and requires an improvement in both median metrics in **every run** for every target H/W case. Samples are not pooled, outliers are not discarded, and the outer control is excluded from the pass decision. A regression returns a nonzero status but preserves the report; a failed child run stops further execution and preserves its logs. Passing this gate does not enable routing or prove statistical significance or performance outside these cases.

### Initial setup for a different machine

Clone the development branch into a **new directory** on the Linux hardware machine. The branch includes the implementation and tools; no archive transfer or manual patch application is needed:

```bash
cd "$HOME"
git clone --single-branch --branch perf/repeat-interleave-tile-copy \
    https://github.com/sreeshanth-soma/tt-metal.git tt-metal-repeat-direct
cd tt-metal-repeat-direct
bash bounty-research/2026-09-20/setup_candidate_clone.sh
```

Leave the original `$HOME/tt-metal` checkout and its Python environment alone. Git refuses to clone over an existing nonempty destination. The setup script also refuses to overwrite an existing `python_env`; use a genuinely new clone for initial preparation.

The script verifies that the candidate descends from the pinned base, checks its sources/environment, initializes submodules, builds, creates a separate Python environment, and runs validation. Build and environment logs remain in the new checkout as `candidate_build.log` and `candidate_venv.log`. It does not reset the old source, overwrite its build/venv, clear global JIT caches, upgrade drivers/firmware, or post anything. A new environment/full build requires disk, network and the Linux build prerequisites in `INSTALLING.md`; missing prerequisites cause an error, not an automatic system upgrade. The user successfully completed this setup for the first candidate.

Probe records contain both the pinned base and the actual candidate commit, plus source hashes. Summaries reject measurements from different commits, sources, runtimes or devices. Do not use a shallow clone that omits the pinned base's history.

For a single diagnostic rerun in an already prepared checkout:

```bash
cd "$HOME/tt-metal-repeat-direct" &&
git pull --ff-only &&
bash bounty-research/2026-09-20/run_candidate_validation.sh --build
```

The runner activates that checkout's own `python_env`, uses its absolute Python executable and build libraries, and replaces inherited source paths. It is safe to launch from a shell still displaying the original checkout's `(python_env)` prompt. Activation stays inside the child script and does not change the parent shell. `git pull --ff-only` does not discard local edits or rewrite history. The runner creates a different results directory for every run.

The runner stops on failure, preserves logs, and performs:

- Focused device correctness/cache/routing tests.
- All six benchmark cases, 10 warmups and 51 samples, without device profiling.
- Five separate bounded Tracy captures, three warmups and five samples each.
- Automatic CSV summaries that exclude setup/warmup/unlabelled operations, sum per-operation kernel durations **within each measured call before taking the median**, and refuse incomplete/mixed-device comparisons.
- A check that each H/W `direct_codegen` region contains exactly one device operation and the **new reader's filename**, not a matmul or fallback.
- One `SUMMARY.md` with device-test counts, host medians/ranges and separate device-kernel sums. It refuses failed/incomplete probes or mismatched source/runtime/device identities across captures.

The runner prints progress and the final `SUMMARY.md`, while verbose compiler/profiler output stays in its log files. On failure it prints the log's last 60 lines. Keep the complete `/tmp/tt-repeat-direct.*` artifact directory; do not paste giant CSVs. For a compact view of any existing repeat capture, the summarizer also accepts the earlier selector captures without `--require-candidate`:

```bash
python bounty-research/2026-09-20/summarize_repeat_capture.py /path/to/ops_perf_results.csv
```

## Gates before submission

- Confirm the passing Wormhole result across independent runs and broader inputs, then test Blackhole before any claimed cross-architecture routing.
- Verify installed JIT sources, runtime source root, and that both measurements use the same source revision and execution settings.
- Review every case's distribution, not just the best ratio. Keep outliers and profiling overhead separate from device-kernel time.
- Promote **only measured-winning, correctness-validated configurations** in `is_demoted`; keep losing/unsupported configurations native. Do not turn this forced-only patch into a claimed production improvement without that step.
- Run the relevant real model path if presenting this as a model optimization; the synthetic prefill shape is not an end-to-end model benchmark.
- Complete project CI/package/license-header checks and human code review; the host harness does not replace them.
- Obtain issue/scope approval and, if seeking payment, the separate required bounty assignment/eligibility. Neither has been established by these results.

The defensible outcome now is **a packed-row candidate with 177 reported passing Wormhole tests and one run of five H/W cases winning both timing metrics, plus a locally tested repeatability workflow**, not a production-routing, acceptance or payment guarantee.
