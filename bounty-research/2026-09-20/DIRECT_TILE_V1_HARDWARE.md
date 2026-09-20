# Direct tiled repeat: first Wormhole hardware result

Date: September 20, 2026. Base: `9173350554b616022b3aa7c4fbad33f13cb36aee`.
Candidate commit: `8d601ab0ee2e7bd4383fd347b4ac6aecf820d912`.
Reader SHA-256: `425d81dc8450185e131a3b9a69c62899b15c16841dd2733474969051bf3e30ee`.

## Provenance and limits

The user ran `setup_candidate_clone.sh` on their Linux hardware host and supplied
the completed terminal summary. The build, separate environment creation and
validation workflow completed. These numbers are transcribed from that summary;
the underlying CSVs, JSONL samples, JUnit XML and traces have not been transferred
to or independently inspected on the local development machine.

- Checkout on the hardware host: `/home/user/tt-metal-repeat-direct`.
- Original artifacts on that host: `/tmp/tt-repeat-direct.xLpGzF`.
- Device reported: `Arch.WORMHOLE_B0`, compute grid `8 x 9`.
- Focused device tests: **135 passed, 0 skipped, no failures/errors**.
- All benchmark input, pre-timing and post-timing bitwise checks passed.
- The new reader was identified in every candidate capture.
- Automatic H/W routing remained disabled; the benchmark used the private forced entry.

This evidence applies to the commit and reader hash above, **not to later reader
optimizations**. No Blackhole run, model-level benchmark, maintainer acceptance or
bounty eligibility is established.

## Unprofiled host-to-completion latency

All times are microseconds, with 51 samples per implementation per case.

| Case | Public median [min, max] | Candidate median [min, max] | Public/candidate |
| --- | ---: | ---: | ---: |
| aligned_h | 438.719 [417.629, 479.778] | 156.270 [144.699, 176.979] | 2.807x |
| aligned_w | 494.099 [465.719, 570.208] | 125.010 [115.510, 163.849] | 3.952x |
| gva_decode_h | 358.399 [343.079, 404.449] | 119.700 [108.669, 140.990] | 2.994x |
| gva_prefill_h | 429.279 [408.759, 485.418] | 202.070 [193.080, 230.239] | 2.124x |
| ragged_w | 501.989 [477.099, 583.339] | 121.620 [110.620, 166.759] | 4.128x |
| outer_control | 121.070 [115.400, 160.629] | 115.530 [109.350, 141.790] | 1.048x |

The outer control does not exercise the new H/W reader.

## Profiled device kernel time

Median of the per-call sums of kernel durations, in microseconds. The runner
requests five measured calls per implementation, after three warmups. This is
not unprofiled host latency and must not be combined with the preceding table.

| Case | Public | Candidate | Public/candidate |
| --- | ---: | ---: | ---: |
| aligned_h | 13.800 | 25.776 | 0.535x |
| aligned_w | 20.493 | 25.703 | 0.797x |
| gva_decode_h | 9.185 | 24.611 | 0.373x |
| gva_prefill_h | 48.989 | 100.538 | 0.487x |
| ragged_w | 29.527 | 25.416 | 1.162x |

Ratios below one indicate a candidate regression. The first version improves
host latency in the five H/W cases but regresses device kernel time in four.
**Do not enable automatic H/W routing based on this run.**

## Next revision

The original reader assembles each BF16 output word from two scalar source
loads and repeats row/column lookup work for every output word. The proposed
revision uses packed word loads, reuses a loaded face row across repeated H
rows, specializes aligned small W repeats, and unrolls face-row writes. It
retains bitwise copies, padding zeroing, one scratch tile and forced-only routing.

Reducing scalar work is an optimization hypothesis, not measured hardware
improvement. Rerun the same workflow at the new commit and retain these first-run
artifacts as a baseline. Do not substitute local mock timings for device results.
