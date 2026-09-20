# Hardware screening results — September 20, 2026

## First repeat-interleave measurement

Provenance: terminal output supplied by the user in this conversation. The original
JSONL/log files remain on the user's cloud instance; they have not been downloaded
or independently rerun locally. This is preliminary performance evidence, not a
confirmed defect, approved bounty, or measured production-kernel optimization.

| Field | Observed value |
| --- | --- |
| Run start | 2026-09-20 11:27:52 UTC |
| Cloud checkout | `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9` |
| Tracked source changes | None reported by `git diff HEAD --stat` |
| Architecture | `Arch.WORMHOLE_B0`; initialization identifies board `n150` |
| Reported compute grid | 8 × 9 |
| Python / PyTorch | 3.10.19 / 2.11.0+cpu |
| Input | BF16, TILE, DRAM, logical shape `[1, 1, 8, 64]` |
| Operation | Repeat each element four times along dimension 2 |
| Output | Logical shape `[1, 1, 32, 64]`, 2,048 elements |
| Program cache | Enabled |
| Warmups / samples | 3 / 9 per implementation |
| Input round trip | Exact |
| Both outputs before and after timing | Exact against the PyTorch reference |
| Completion | `suite_finished` and `device_closed`; no probe error |

| Implementation | Median (µs) | Minimum (µs) | Maximum (µs) |
| --- | ---: | ---: | ---: |
| Public `ttnn.repeat_interleave` | 450.819 | 399.039 | 491.678 |
| Pre-uploaded one-hot selector matmul | 229.019 | 221.949 | 268.259 |

The ratio of medians is **1.96848**, a **49.1993% lower measured latency** for the
selector diagnostic, or 221.800 µs less per call in this run. These are synchronized
host-to-completion timings, including host dispatch and synchronization, not isolated
device execution times. Even though the observed sample ranges do not overlap, one
process and nine samples do not establish reproducibility across runs or systems.

Raw samples, in microseconds:

```json
{
  "public_repeat": [491.678, 450.819, 447.848, 470.938, 436.458, 399.039, 462.399, 472.208, 442.288],
  "selector_matmul": [235.109, 221.949, 236.249, 229.019, 268.259, 225.569, 228.879, 223.789, 240.149]
}
```

The selector has 256 logical elements. Its creation and upload are excluded from
warm timing. Exactness applies only to this seeded, finite, dyadic-valued BF16 input;
it does not establish general floating-point equivalence, exceptional-value behavior,
or bitwise preservation. A dense selector is a diagnostic comparator, not the proposed
production solution. Its storage and setup costs must not be omitted from a deployment
claim.

## Revision check

GitHub commit metadata identifies the cloud SHA as **Release: (MINOR) Promote main
→ stable (2026-08-18)**, authored and committed August 18, 2026. The local research
checkout is `9173350554b616022b3aa7c4fbad33f13cb36aee`, dated September 20, 2026.
Updating the local Mac repository did not update the cloud checkout or its build.
The cloud log also reports that the newer repeat-interleave codegen support file is
missing. The checkout SHA alone does not certify which source built the loaded native
extension, so a final benchmark needs a documented, matching source/build revision.

The checked September 20 source explicitly rejects TILE H/W axes in
`repeat_interleave_codegen_supported.cpp:115`. Its native implementation still
converts to ROW_MAJOR, unsqueezes, concatenates, reshapes, and restores the layout
(`repeat_interleave.cpp:142`). The source retrieved at the cloud SHA also uses that
composition. Thus the proposed TILE-preserving H/W path remains worth investigating,
but **these measurements do not establish its performance on September 20 main**;
the routing and constituent operations can change between revisions.

Verified upstream provenance:

```text
https://github.com/tenstorrent/tt-metal/commit/9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9
https://github.com/tenstorrent/tt-metal/blob/9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9/ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/repeat_interleave.cpp
https://github.com/tenstorrent/tt-metal/blob/9173350554b616022b3aa7c4fbad33f13cb36aee/ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_supported.cpp
```

## Matrix attempt and probe correction

The user supplied the JSONL and log for the run starting at 2026-09-20 11:33:23 UTC
on the same cloud checkout. The first case, `aligned_h`, passed input round-trip and
public `repeat_interleave` correctness: all 65,536 output elements matched exactly.
The diagnostic selector matmul then failed host-side output-shape validation. No
`repeat_result` or timing samples were produced, and the remaining five cases did
not run. The device closed normally.

This is a **probe compatibility error**, not evidence that `repeat_interleave` is
incorrect. The probe supplied a rank-2 selector `[128, 64]` as the left operand and
the rank-4 tensor `[1, 4, 64, 128]` as the right operand. The cloud matmul output-shape
routine rejects the non-singleton leading dimension of that higher-rank right operand.
The earlier decode case had only singleton batch dimensions, so it did not expose
this invalid assumption in our comparator.

Probe revision `repeat-selector-batches-v2` constructs a selector for every input
batch and initializes its one-hot entries across all batches. Both operands now
have identical batch dimensions: for `aligned_h`, the selector is `[1, 4, 128, 64]`.
The same rule applies to H and W comparisons. This is materialized replication, not
a zero-cost broadcast; the `selector_setup` event and result's logical element count
include these copies, while construction/upload remain outside warm timing. Each run
now records the probe revision and script SHA-256. Rerun the original decode case with
the corrected probe before combining its results with the matrix.

Host-only regression tests check allocation shapes and one-hot indexing for the five
comparator cases, multiple non-singleton batch dimensions, and rejection of unsupported
outer-axis selectors. They do not exercise PyTorch or TTNN hardware. The following
user-run measurement verifies the corrected comparator on one hardware case.

## Corrected aligned-H hardware result

The user supplied JSONL and log contents for the run starting at **2026-09-20
11:41:36 UTC**. The reported probe revision is `repeat-selector-batches-v2`, and its
SHA-256 matches the local script exactly:

```text
d3e4967e87a48cca0b9004408fa0b0ab53237dd1422d5d712af4b4ca0c5e679b
```

The same Wormhole cloud checkout, `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9`, ran
`aligned_h`: BF16 input `[1, 4, 64, 128]`, repeats 2 along dimension 2, output
`[1, 4, 128, 128]`. Input round-trip and both implementations' pre/post-timing
correctness checks were exact; all **65,536** output values matched the reference.
The run emitted `suite_finished` and `device_closed`, with no probe error.

| Implementation | Median (µs) | Minimum (µs) | Maximum (µs) |
| --- | ---: | ---: | ---: |
| Public `ttnn.repeat_interleave` | 461.149 | 436.898 | 494.589 |
| Batched one-hot selector matmul | 196.480 | 178.129 | 212.279 |

With three warmups and nine samples per implementation, the ratio of medians is
**2.34705**. The diagnostic has **57.3934% lower measured latency**, a difference of
264.669 µs. These are synchronized host-to-completion measurements, not device-only
execution times. The selector is `[1, 4, 128, 64]` with **32,768 logical elements**;
its materialization and upload remain excluded from warm timing.

```json
{
  "public_repeat": [490.928, 455.318, 461.149, 479.498, 436.898, 446.869, 462.628, 494.589, 445.829],
  "selector_matmul": [194.319, 211.769, 210.4, 192.9, 187.06, 196.48, 203.339, 178.129, 212.279]
}
```

This resolves the observed batch-shape failure for `aligned_h` and supports continuing
the screen. It is not verification of all cases, a new production kernel, a model
speedup, or a benchmark of the September 20 source build. The earlier decode result
used the original selector construction; rerunning it within the corrected matrix
keeps the comparison method consistent.

## Completed six-case BF16 matrix

The user supplied terminal output for the run starting at **2026-09-20 11:44:03 UTC**,
using the same probe SHA and cloud checkout. All six public-operation cases and all
five selector comparisons passed exact correctness before and after timing, with
five warmups and 31 measured calls per implementation. All input round trips passed.
The run completed and closed the device. Selected metadata, correctness flags, and
all 341 timing samples are transcribed in `repeat_matrix_v2_observed.json`; the file
is explicitly not a downloaded copy of the original JSONL.

| Case | Public median (µs) | Selector median (µs) | Public / selector | Selector latency relative to public |
| --- | ---: | ---: | ---: | --- |
| `aligned_h` | 446.988 | 194.719 | 2.296× | 56.4% lower |
| `aligned_w` | 573.528 | 190.849 | 3.005× | 66.7% lower |
| `gva_decode_h` | 416.019 | 193.409 | 2.151× | 53.5% lower |
| `gva_prefill_h` | 492.168 | 719.238 | 0.684× | **46.1% higher** |
| `ragged_w` | 569.518 | 180.279 | 3.159× | 68.3% lower |
| `outer_control` | 447.199 | Not tested | Not applicable | Not applicable |

The four winning comparisons have non-overlapping observed sample ranges within this
run. The aligned-H medians are also broadly consistent with the separate preceding
process (2.347× then 2.296×), but the entire matrix has not yet been repeated across
processes. The prefill result is a genuine counterexample to a blanket selector
replacement: its public path has one **2,301.712 µs** sample, but all other public
samples are below even the fastest selector sample. Retain that outlier in the raw
data; it does not reverse the median conclusion, and its cause is not established.

Interpretation and limits:

- Prioritize investigation of a TILE-preserving H/W copy implementation, especially
  the width cases. This is a proposed direction, not an implemented optimization.
- Do not replace `repeat_interleave` globally with this matmul comparator. Besides
  the measured prefill regression, it requires pre-uploaded dense batched selectors
  and is tested only on a restricted set of finite dyadic values.
- Keep prefill as a performance regression guard. Its loss does not prove a direct
  data-movement kernel would lose; that implementation has not been written or timed.
- The outer-axis control has no alternative implementation comparison. Its cloud
  source predates the newer codegen route; it is not a measured codegen baseline.
- Host dispatch and synchronization remain included. A flat roughly 180–195 µs
  selector latency across four differently sized cases is consistent with substantial
  fixed overhead, but is not proof of the device-versus-host time breakdown.
- No model execution, device-only profiling, updated-source hardware benchmark, or
  full-value-domain correctness validation has been performed. Do not quote a funded
  bounty value or a twofold/threefold production-model speedup from these numbers.

Reproduction command for this completed run:

```bash
cd ~/tt-metal
RUN="/tmp/tt-bounty-matrix-v2-$(date +%Y%m%d-%H%M%S)"
set -o pipefail
python -u /tmp/hardware_probe.py \
  --suite repeat --dtype bf16 --warmup 5 --samples 31 \
  --output "$RUN.jsonl" 2>&1 | tee "$RUN.log"
printf '\nResults: %s.jsonl\nLog: %s.log\n' "$RUN" "$RUN"
```

## Profiling readiness

Do not keep collecting the same host-time matrix before learning what it measures.
Check the existing cloud profiler installation without opening the device or changing
the working build. The binary names and directory below are verified against
`tools/tracy/common.py` at the cloud SHA. A module specification and executable files
are prerequisites, not proof that the loaded native extension was built with profiling.
The CMake cache is supporting build metadata, not a source-to-binary provenance guarantee.

```bash
cd ~/tt-metal
python -c 'import importlib.util; print("tracy:", importlib.util.find_spec("tracy"))'
ls -l build/tools/profiler/bin/tracy-capture build/tools/profiler/bin/tracy-csvexport
find -L build -maxdepth 3 -name CMakeCache.txt -exec grep -H -E '^(ENABLE_TRACY|CMAKE_BUILD_TYPE):' {} \;
printf 'TT_METAL_HOME=%s\nTT_METAL_DPRINT_CORES=%s\nTT_METAL_WATCHER=%s\n' "${TT_METAL_HOME-}" "${TT_METAL_DPRINT_CORES-}" "${TT_METAL_WATCHER-}"
```

After reviewing readiness, profile a winning width case and the losing prefill case,
keeping profiler-perturbed host timings separate from these unprofiled measurements.
Account for every primitive making up a public composite call; do not compare one
constituent kernel to a full alternative. Then reproduce on a matching current-source
build and establish an actual workload bottleneck before proposing acceptance criteria.
Do not blindly pull/rebuild the cloud installation or reset the device.

The user's subsequent readiness output confirms that Python resolves `tracy` from
`/home/user/tt-metal/tools/tracy`, both `tracy-capture` and `tracy-csvexport` exist
and are executable, and the main CMake cache says `Release` and `ENABLE_TRACY=ON`.
`TT_METAL_HOME` points to that checkout; `TT_METAL_DPRINT_CORES` and `TT_METAL_WATCHER`
print empty. The separate WASM viewer build's `MinSizeRel` setting is not the main
Metal build type. These checks justify a first capture without reinstalling or
rebuilding source; actual device records and source-to-binary provenance remain
unverified until the capture is inspected.

## Labeled device captures

The local probe is now revision `repeat-selector-batches-v3-tracy`. Its optional
`--tracy-signposts` flag leaves selector construction and operation arguments unchanged
and brackets each warmup and measured call with the established Tracy signpost format:

```text
TT_BOUNTY/{case}/{implementation}/{warmup|sample}/{iteration}/{begin|end|aborted}
```

Input preparation, initial correctness, and post-timing correctness are outside these
regions. A measured region starts after synchronizing previous work and ends after
the operation's completion synchronization. A failed operation gets an `aborted`
marker rather than a successful `end`. JSONL explicitly flags device profiling so
its instrumented host timings cannot be mistaken for the earlier unprofiled baseline.
Use device rows within complete `sample` regions when analyzing the report, not
warmups, setup, correctness transfers, or individual cores counted repeatedly.

The flag requires a single repeat case, at most ten warmups and sixteen samples,
and an enabled device profiler; debug-print/watcher conflicts fail before device
opening. Eighteen host unit tests, Python 3.10 syntax checks, and five NumPy selector
identities pass locally. These tests do not validate device capture.

Copy the updated local `hardware_probe.py` over `/tmp/hardware_probe.py` on the cloud
before running this command. It creates a fresh temporary root, captures the winning
`aligned_w` and losing `gva_prefill_h` cases sequentially, and stops the loop if Tracy
or the workload reports failure. `TT_METAL_PROFILER_DIR` is set before Tracy imports
as well as through `-o`, keeping module-level paths and the report destination aligned.
`-p` avoids whole-script Python tracing; `-r` requests the operation report. The
standard Tracy CLI may start its bundled local WASM viewer and copy captures into
its viewer directory; opening the viewer is unnecessary for CSV analysis.

```bash
cd ~/tt-metal
ROOT="$(mktemp -d /tmp/tt-repeat-profile.XXXXXX)"
set -o pipefail

for CASE in aligned_w gva_prefill_h; do
  OUT="$ROOT/$CASE"
  if ! TT_METAL_PROFILER_DIR="$OUT" python -u -m tracy -p -r -v \
    --check-exit-code -o "$OUT" \
    /tmp/hardware_probe.py \
    --suite repeat --case "$CASE" --dtype bf16 \
    --warmup 3 --samples 5 --tracy-signposts \
    --output "$OUT/probe.jsonl" 2>&1 | tee "$ROOT/$CASE.log"; then
    printf '\nCapture failed; send %s/%s.log\n' "$ROOT" "$CASE"
    break
  fi
done

printf '\nAll results: %s\n' "$ROOT"
find "$ROOT" -type f -name 'ops_perf_results*.csv' -print
```

Collect both `ops_perf_results*.csv` files plus each case's `probe.jsonl`. If capture
or report processing fails, preserve the artifacts and send the relevant case log;
do not reset the device or launch competing capture jobs. Do not interpret the probe's
profiled `median_us` ratio as the device-only ratio. First inspect the report's actual
device fields, operation IDs, and complete labeled regions, including every primitive
in each composite call and any overlap between kernel intervals.

## Capture completion — initial status before CSV inspection

The user supplied terminal output showing both captures completed on September 20,
2026, under `/tmp/tt-repeat-profile.AJAB2B`. This is capture-status evidence from the
log, not the contents of either generated CSV. The profile revision reported for
prefill is `repeat-selector-batches-v3-tracy` with SHA-256
`0cb431b151fcd2b77178258003a8f8097ddd07d4b04f8675796c910055c835c6`.

Both cases emitted exact post-timing correctness for both implementations, followed
by `suite_finished`, `device_closed`, saved Tracy traces, and successful CSV report
generation. The displayed `#pragma message` compiler notes did not stop execution.
The output includes instrumented device-kernel compilation (`PROFILE_KERNEL=1`),
not evidence of a manual repository rebuild or a change to the source revision.

| Case | Profiled public host median (µs) | Profiled selector host median (µs) | Reported device-kernel count | Reported device-kernel duration range (ns) |
| --- | ---: | ---: | ---: | ---: |
| `aligned_w` | 654.497 | 216.119 | 60 | 1,984–14,998 |
| `gva_prefill_h` | 370.348 | 681.907 | 40 | 9,472–566,487 |

The reported kernel averages are 5,839.1 ns and 153,765.0 ns, respectively. These
are mixed run-level summaries. They do not assign durations to either implementation
or isolate the measured calls from warmups and correctness work. No per-implementation
device-only ratio has been derived. In particular, the width probe's 3.028× ratio
remains a **profiled host-to-completion** ratio, not a measured device-kernel speedup.
The prefill public host samples include a 4,134.093 µs outlier; its cause is unassigned.

Generated report paths, as printed in the supplied log:

```text
/tmp/tt-repeat-profile.AJAB2B/aligned_w/reports/2026_09_20_12_03_27/ops_perf_results_2026_09_20_12_03_27.csv
/tmp/tt-repeat-profile.AJAB2B/gva_prefill_h/reports/2026_09_20_12_03_40/ops_perf_results_2026_09_20_12_03_40.csv
```

The next step is **read-only export**, not another hardware run. Prefer the original
CSV attachments, or project these columns while preserving every row, its order,
signposts, operation IDs, and units. Missing requested headers are reported explicitly.
This compact export does not sum, discard, or reinterpret any durations.

```bash
python - <<'PY'
import csv
import sys
from pathlib import Path

root = Path("/tmp/tt-repeat-profile.AJAB2B")
columns = [
    "OP CODE", "OP TYPE", "GLOBAL CALL COUNT", "DEVICE ID", "CORE COUNT",
    "HOST START TS", "HOST DURATION [ns]",
    "DEVICE FW START CYCLE", "DEVICE FW END CYCLE",
    "DEVICE FW DURATION [ns]", "DEVICE KERNEL DURATION [ns]",
    "OP TO OP LATENCY [ns]", "PROGRAM CACHE HIT",
]
for case in ("aligned_w", "gva_prefill_h"):
    reports = sorted((root / case / "reports").rglob("ops_perf_results*.csv"))
    if len(reports) != 1:
        raise SystemExit(f"Expected one report for {case}, found {len(reports)}")
    print(f"\n=== {case}: {reports[0]} ===")
    with reports[0].open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        headers = reader.fieldnames or []
        missing = [name for name in columns if name not in headers]
        print("Missing requested columns:", missing)
        selected = [name for name in columns if name in headers]
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(selected)
        for row in reader:
            writer.writerow([row.get(name, "") for name in selected])
PY
```

Do not infer that the longest prefill kernel belongs to matmul, or that a particular
width duration belongs to transpose/concat, until the per-operation rows and complete
sample signposts establish that attribution. Preserve both captures unchanged.

## Width CSV analysis — five operations versus one

The user subsequently pasted the complete `aligned_w` CSV named above. Only the
five complete `sample/0` through `sample/4` regions for each implementation are used
here; initial correctness, three warmups, and post-timing correctness are excluded.
Selected measured rows, call IDs, region timestamps, and calculations are preserved
in `aligned_w_profile_observed.json`. This is a labeled transcription of supplied
output, not the original CSV, a new hardware run, or independent source/binary
verification.

Every measured public call contains this device-operation sequence:

```text
Transpose → UntilizeCodegen → Concat → Tilize → Transpose
```

The first transpose exchanges height and width. Untilize converts the tiled tensor
to row-major storage, concat performs the repetition, tilize restores tiled storage,
and the final transpose restores the axes. The selector comparator instead contains
one `MatmulDeviceOperation`. These are operation counts, not counts of individual
RISC/core kernels.

| Metric, median across five samples | Public repeat | Selector matmul |
| --- | ---: | ---: |
| Device operations per call | 5 | 1 |
| Sum of `DEVICE KERNEL DURATION [ns]`, converted to µs | 20.397 | 14.823 |
| Probe's profiled host-to-completion latency (µs) | 654.497 | 216.119 |

The public kernel-duration sums, in sample order, are
`[20499, 20407, 20049, 20228, 20397]` ns. The selector durations are
`[14823, 14684, 14668, 14826, 14998]` ns. Summation happens **within each call before
taking the median**; per-core or per-RISC durations are not added together. The
kernel-sum ratio is **1.376×**, with a median difference of **5.574 µs**, not the
**3.028×** ratio of profiled full-call medians. Summed kernel intervals exclude
between-operation gaps and must not be reported as full-call device elapsed time.

For each public sample, summing `OP TO OP LATENCY [ns]` only for its second through
fifth operations gives `[437235, 469014, 422023, 482453, 406081]` ns, median
**437.235 µs**. The first operation's value belongs to the transition from preceding
work and is deliberately excluded. The four internal transitions are much larger
than the kernel durations. This supports investigating time outside active kernel
intervals and the cost of the multi-operation path. It does **not** isolate those
gaps as Python time, CPU execution, allocation, synchronization, or dispatch alone.
Host timestamps and device cycles are not subtracted from each other. The probe's
host timing also excludes some signpost overhead, so the marker spans are not
substituted for its recorded samples.

The practical optimization hypothesis is therefore a direct tiled repeat operation
that avoids the intermediate conversions and multiple launches. The CSV motivates
that investigation; it does not prove a proposed kernel's speedup or justify using
the dense selector as a production replacement. The earlier unprofiled width ratio
remains 3.005× for the tested synthetic case, not a device-kernel ratio. Hardware
validation on current main, general input coverage, and production-model impact
remain outstanding. The CSV's warmed `PROGRAM CACHE HIT=False` fields alone are
not sufficient to conclude that every call recompiles.

The next requested artifact was the prefill CSV, now analyzed in the following
section. No new run or rebuild was necessary. To minimize copy/paste mistakes,
the user was given this single read-only command in the same cloud terminal:

```bash
cat /tmp/tt-repeat-profile.AJAB2B/gva_prefill_h/reports/2026_09_20_12_03_40/ops_perf_results_2026_09_20_12_03_40.csv
```

## Prefill CSV analysis — selector matmul is the slow operation

The user supplied the complete `gva_prefill_h` CSV from the same capture root.
Its selected measured rows and host samples are transcribed in
`gva_prefill_h_profile_observed.json`, with explicit provenance. Analysis uses all
five complete measured regions per implementation, excluding initial correctness,
warmups, and post-timing correctness. No host outlier is removed.

Each public call contains three device operations:

```text
UntilizeWithUnpadding → Concat → Tilize
```

Their reported core counts are respectively **64, 72, and 64**. Each selector call
contains one `MatmulDeviceOperation` reporting **2 cores**, against **72 available
worker cores**. The long approximately 566 µs device intervals can now be assigned
to matmul directly, rather than inferred from the mixed histogram.

| Metric, median across five samples | Public repeat | Selector matmul |
| --- | ---: | ---: |
| Device operations per call | 3 | 1 |
| Reported cores for each operation | 64 / 72 / 64 | 2 |
| Sum of `DEVICE KERNEL DURATION [ns]`, converted to µs | 49.278 | 565.617 |
| Probe's profiled host-to-completion latency (µs) | 370.348 | 681.907 |

The public kernel-duration sums are `[49095, 49278, 49280, 48893, 50100]` ns.
The selector durations are `[566326, 565366, 566487, 565617, 565154]` ns.
The selector/public median ratio is **11.478× for summed kernel duration** and
**1.841× for profiled full-call latency**. These are different metrics. The earlier
unprofiled prefill medians remain **492.168 µs public versus 719.238 µs selector**,
or **1.461× longer selector latency**; profiling results do not replace that baseline.

The selected matmul program is `MatmulMultiCoreReuseMultiCast1DProgramConfig` with
`fuse_batch=0`, `per_core_M=1`, and `per_core_N=1`. Both operands have batch prefix
`[1, 128]`. The grid allowance of `8-9` does not mean all 72 cores are used: the CSV
reports only two for this operation. This is evidence that the selected configuration
uses little spatial parallelism for this workload, not proof that two-core allocation
is the sole cause or that using more cores guarantees a particular speedup. The
CSV does not test alternative matmul configurations. Reported core counts must not
be reinterpreted as measured FPU utilization.

The public path's internal `OP TO OP LATENCY [ns]` sums, excluding each region's
first operation, are `[165552, 167291, 153025, 158829, 162713]` ns, median
**162.713 µs**. These gaps remain significant, but unlike the width case the selector
has a much longer kernel interval that more than offsets its lower operation count.

The public sample-0 host time of **4,134.093 µs** remains in the five-sample median.
Its three device kernel durations still sum to **49.095 µs**, so it is not evidence
of a similarly prolonged repeat kernel. In the host clock alone, the gap from the
last device operation's recorded `HOST END TS` to the region's end signpost is
**3,835.404 µs**. That host operation endpoint is not device completion; this does
not isolate the stall as scheduling, synchronization, profiling, or another cause.
No host timestamp is subtracted from a device-cycle timestamp.

## Decision after both captures

- **Reject a universal dense-matmul replacement.** It wins the width diagnostic but
  loses prefill, even with selector creation excluded. Correctness on the tested
  finite BF16 inputs does not establish full floating-point copy semantics.
- **Keep the direct tiled-repeat hypothesis.** Width executes five device operations
  and prefill three, including conversions through row-major storage. A direct copy
  kernel could avoid those conversions and launches without doing dense matrix
  multiplication, but no such kernel or its speedup has been demonstrated here.
- **Treat matmul configuration as a separate experiment.** The prefill result measures
  this particular selected program, not every possible implementation of selection
  through matmul, and is not evidence that more launches are generally faster.
- **Advance to implementation scoping, not more identical profiling runs.** Preserve
  the original capture directory. Before asserting a current-main or production
  improvement, validate the relevant baseline on the target revision and an actual
  model workload. No bounty award, acceptance, or production speedup is established.

Both requested CSVs have now been inspected. The user does not need to rerun these
captures or supply the same logs again to complete this diagnostic stage.
