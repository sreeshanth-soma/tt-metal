# Packed-row repeat: passing Wormhole hardware result

Reported September 21, 2026. Base: `9173350554b616022b3aa7c4fbad33f13cb36aee`.
Tested commit: `0f7d9dfdacd347baf3f3cdd0fd7b75dc365c55a1`.
Reader SHA-256: `987446139978004bd4b9c148742382f8aefae8259fb27946baf8b280431676fc`.

## Provenance

The user pulled the packed-row revision into `/home/user/tt-metal-repeat-direct`
and ran `run_candidate_validation.sh --build` on their Linux hardware host.
The supplied terminal summary shows a successful build, **177 passing device
tests with zero skips/failures/errors**, passing benchmark input/pre-timing/
post-timing bitwise checks, and the new reader in every candidate capture.

- Reported device: `Arch.WORMHOLE_B0`, grid `8 x 9`.
- Original artifacts: `/tmp/tt-repeat-direct.5cmJXI` on the hardware host.
- Earlier scalar-reader artifacts: `/tmp/tt-repeat-direct.xLpGzF` on that host.
- These tables are transcribed from the supplied summary. Raw JSONL, CSV, JUnit
  XML and Tracy files are not available on the local development machine and
  have not been independently re-parsed here.
- The run covers the forced candidate, not automatic public H/W routing.

## Unprofiled host-to-completion latency

Times are microseconds. Each implementation has 51 measured samples per case.
All reported minima and maxima, including outliers, are retained.

| Case | Public median [min, max] | Candidate median [min, max] | Public/candidate |
| --- | ---: | ---: | ---: |
| aligned_h | 386.829 [374.178, 422.738] | 124.539 [115.509, 147.549] | 3.106x |
| aligned_w | 371.049 [341.539, 494.608] | 101.719 [90.769, 334.929] | 3.648x |
| gva_decode_h | 361.709 [347.888, 393.889] | 119.669 [110.550, 142.139] | 3.023x |
| gva_prefill_h | 422.308 [392.968, 722.617] | 131.510 [118.320, 202.749] | 3.211x |
| ragged_w | 494.188 [480.228, 555.118] | 122.069 [114.919, 146.079] | 4.048x |
| outer_control | 133.720 [120.319, 164.619] | 124.670 [112.100, 157.479] | 1.073x |

The outer control uses the existing outer-axis path and is not evidence for the
new reader. In particular, do not credit its small ratio to the H/W optimization.

## Profiled device kernel time

Median per-call sums of kernel durations, in microseconds. The runner requests
five measured calls after three warmups. These are separate from the unprofiled
host latency measurements.

| Case | Public | Candidate | Public/candidate |
| --- | ---: | ---: | ---: |
| aligned_h | 14.058 | 7.252 | 1.938x |
| aligned_w | 20.514 | 9.641 | 2.128x |
| gva_decode_h | 9.145 | 4.963 | 1.843x |
| gva_prefill_h | 48.910 | 21.878 | 2.236x |
| ragged_w | 29.761 | 16.339 | 1.821x |

## Conclusion and remaining gates

All five measured H/W cases improve both reported medians in this run: host
latency ratios are **3.023x–4.048x** and device-kernel-sum ratios are
**1.821x–2.236x**. The earlier scalar reader's four device-time regressions are
not present in this packed-row run.

This is one successful run of these BF16 benchmark cases on this Wormhole
configuration, not a guarantee for every supported shape, dtype, architecture
or model. The 177-test correctness result does not establish performance for
all those test inputs. Public H/W routing remains disabled.

The subsequent validation-only update kept the measured reader unchanged.
The user reported a passing three-run confirmation at commit
`472a874bde067c9cca9ba3a61fe8087706ad600a`; see `PACKED_ROW_CONFIRMATION.md` for
that separate record and its provenance limits. All five target cases improved
both metrics in every run. This does not enable routing or replace a broader
shape/performance sweep, appropriate architecture coverage, model integration,
project CI, or maintainer review.

No issue, PR, bounty assignment or eligibility has been established by this run.
