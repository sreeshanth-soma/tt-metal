# Repeat-interleave: implementation and acceptance package

Date: **September 20, 2026**. Base: `9173350554b616022b3aa7c4fbad33f13cb36aee`.

## Bottom line

The submission worth pursuing is a **measured, narrowly routed, tile-preserving copy implementation of H/W repeat-interleave**. It is not a universal dense-matmul replacement and not a redo of the already-merged repeat codegen port.

That candidate is now implemented in the working tree, with device regression tests and host-side verification. **It is not yet ready to claim a production speedup or submit as a completed performance fix.** The new H/W path is deliberately accessible through the existing private forced-codegen entry only. Automatic H/W routing stays on native until actual hardware correctness and performance support promotion.

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
| Actual performance evidence without regressions | Performance category in `CONTRIBUTING.md:160` | Hardware measurements of this candidate are still missing |
| Required CI, beyond initial PR checks | `CONTRIBUTING.md:134` | Not run; maintainer-triggered coverage may be needed |

The title/description searches found no open PR explicitly implementing this generic tiled H/W path at the audit time. That is **not proof of exclusivity** or permission to claim someone else's issue. Model-specific repeat elimination, concat ports, and matmul tuning are separate overlapping areas to avoid packaging into this change.

Bounty eligibility is separate from technical acceptance. The published terms inspected on September 20 require a merged contribution addressing an appropriately tagged open bounty issue, assignment at PR submission, and the remaining program conditions. This candidate has no confirmed bounty or assignment. The repository's AI restrictions (`CONTRIBUTING.md:584`) prohibit automated/AI-generated assignment claims; any submission requires human review and responsibility. This document is an internal implementation checklist, not an assignment-request template.

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

The patch modifies six existing source/test files and adds one reader. Research, mocks and benchmark scripts are separate from the proposed production diff in `core.patch`.

### Dataflow and geometry

Each core owns a contiguous range of output tiles. It reads one source tile into a scratch CB, rearranges its raw 16-bit/32-bit elements into the output CB, and uses the existing batched interleaved writer. There is no compute kernel, dense selector, transpose, untilize, concat or tilize in the candidate device operation.

For tile coordinate `q`, element offset `offset` in `[0,31]`, and integer repeat count `repeats`:

```text
source_element = floor((32*q + offset) / repeats)
source_tile    = floor(q / repeats)
```

The source tile is constant across all 32 output offsets: writing `q = repeats*block + remainder` bounds `32*remainder + offset` below `32*repeats`. **A second source tile is never needed.** Source-tile boundaries expand to output boundaries divisible by 32. The source tile can also be reused while consecutive output tiles refer to it.

Face layout is explicit: four 16×16 faces in each 32×32 tile. BF16 values are packed into 32-bit stores without arithmetic conversion. Invalid output rows/columns are zero-filled rather than populated from input padding.

The output page count is recomputed from the **repeated logical shape**, then rounded to tiles. It is not `input_tiles * repeats`. For example, H=8 repeated four times gives H=32 and still one tile row. The same geometry helper is used by the support gate and dispatch helper; it checks canonical padding and overflow before allocation.

CB staging per core is bounded:

- Eight output tiles, preserving the existing writer's `depth >= 2*batch` protocol with batch four.
- One scratch tile.
- **18 KiB for BF16; 36 KiB for FP32/INT32**, excluding code, stack and runtime-reserved memory.
- The new gate checks available L1 for those nine pages and checks the input buffer's page pitch.

On the observed 8×9 grid, output-page splitting assigns the width case's 64 tiles to 64 cores and prefill's 256 tiles across 72 cores. These are factory-derived work assignments, not measured utilization or performance claims.

## Validation

### Completed locally

- Compile and execute the **actual reader source**, not a Python translation, for 24 compile-time variants: two element widths × two axes × six repeat counts (2, 3, 4, 7, 33, 127).
- **6,024 geometry/data cases**, including face/tile boundaries, ragged shapes, multiple batches, uneven core partitions, the supplied width/prefill geometries, repeated calls with changed data, and poisoned input padding.
- AddressSanitizer and UndefinedBehaviorSanitizer, `-Wall -Wextra -Werror`.
- Guarded scratch/output allocations, delayed mock NOC reads until the barrier, batched writer backpressure/ring-wrap checks, raw special-value bit patterns, and exact padded-output comparison.
- **39 passing host unittest methods**, including the actual-reader sweep, profiler signposts, alternated benchmark order, CLI preflight, old-checkout rejection, committed-descendant acceptance, simulation/debug-mode rejection, CSV aggregation and the readable report's commit/source checks.
- Pinned formatting checks pass: Black 23.10.1 on nine Python files and clang-format 19.1.4 on modified C++ lines and the new reader. Python 3.10 grammar checks pass for eleven files; both hardware scripts pass Bash syntax and refuse execution on the local non-Linux machine.
- The focused seven-file production/device-test patch was also applied to an archive of the exact base; all resulting files byte-matched the working tree. The branch already contains these changes, so do not reapply the earlier archive's `core.patch`.

These are **host mocks**, not official tt-emule or Tenstorrent hardware. They do not validate actual NoC transactions, the complete TTNN C++ build, physical allocator behavior, real cache rebinding, device compiler ABI, or speed. `HOST_VALIDATION.json` records the final source hash and results.

### Authored, not executed here

81 new parameterized device-test instances cover finite bitwise correctness, cache-hit input/output rebinding, NaN payloads/infinities/signed zero/subnormals, output-extent overflow and explicit L1-I/O refusal. Existing routing tests continue to check native fallback.

The local environment is macOS without a TT device or Torch/TTNN runtime. No connected cloud browser was available. Consequently the Linux host build, device JIT, device tests, installed-package check, Wormhole/Blackhole performance, and model integration remain unverified.

## Hardware handoff: one bounded workflow

Clone the development branch into a **new directory** on the Linux hardware machine. The branch includes the implementation and tools; no archive transfer or manual patch application is needed:

```bash
cd "$HOME"
git clone --single-branch --branch perf/repeat-interleave-tile-copy \
    https://github.com/sreeshanth-soma/tt-metal.git tt-metal-repeat-direct
cd tt-metal-repeat-direct
bash bounty-research/2026-09-20/setup_candidate_clone.sh
```

Leave the original `$HOME/tt-metal` checkout and its Python environment alone. Git refuses to clone over an existing nonempty destination. The setup script also refuses to overwrite an existing `python_env`; use a genuinely new clone for initial preparation.

The script verifies that the candidate descends from the pinned base, checks its sources/environment, initializes submodules, builds, creates a separate Python environment, and runs validation. Build and environment logs remain in the new checkout as `candidate_build.log` and `candidate_venv.log`. It does not reset the old source, overwrite its build/venv, clear global JIT caches, upgrade drivers/firmware, or post anything. A new environment/full build requires disk, network and the Linux build prerequisites in `INSTALLING.md`; missing prerequisites cause an error, not an automatic system upgrade. The complete Linux setup has not been executed here.

Probe records contain both the pinned base and the actual candidate commit, plus source hashes. Summaries reject measurements from different commits, sources, runtimes or devices. Do not use a shallow clone that omits the pinned base's history.

For an already prepared candidate checkout with its environment activated:

```bash
bash bounty-research/2026-09-20/run_candidate_validation.sh --build
```

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

- Build and run the new tests on real Wormhole, then Blackhole for any claimed cross-architecture routing.
- Verify installed JIT sources, runtime source root, and that both measurements use the same source revision and execution settings.
- Review every case's distribution, not just the best ratio. Keep outliers and profiling overhead separate from device-kernel time.
- Promote **only measured-winning, correctness-validated configurations** in `is_demoted`; keep losing/unsupported configurations native. Do not turn this forced-only patch into a claimed production improvement without that step.
- Run the relevant real model path if presenting this as a model optimization; the synthetic prefill shape is not an end-to-end model benchmark.
- Complete project CI/package/license-header checks and human code review; the host harness does not replace them.
- Obtain issue/scope approval and, if seeking payment, the separate required bounty assignment/eligibility. Neither has been established by these results.

The defensible outcome now is **a concrete, locally verified candidate and a reproducible hardware validation package**, not an acceptance or payment guarantee.
