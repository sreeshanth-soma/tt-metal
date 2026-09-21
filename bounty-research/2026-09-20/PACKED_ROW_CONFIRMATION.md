# Packed-row repeat: three-run Wormhole confirmation

Reported September 21, 2026. Tested commit:
`472a874bde067c9cca9ba3a61fe8087706ad600a`.
Reader SHA-256: `987446139978004bd4b9c148742382f8aefae8259fb27946baf8b280431676fc`.

## Provenance and result

The user supplied the complete terminal output from
`confirm_candidate_validation.sh --build`: **PASS across three complete runs**
on `Arch.WORMHOLE_B0`, compute grid `8 x 9`. Each run passed **177 device tests**
(531 test executions, not 531 distinct tests). The measured reader is unchanged
from the earlier single-run result in `PACKED_ROW_HARDWARE.md`.

The report states that all probes have distinct starts, clean tracked sources,
matching commit/source/runtime/device identity, and matching sample counts:
at least 51 unprofiled host samples and five profiled device samples per leg.
These checks were performed by the workflow on the user's hardware host.
The raw JSONL, CSV, JUnit XML and Tracy files are not available locally and
have **not** been independently re-parsed on the development machine.

Artifacts remain on the hardware host:

- Root: `/tmp/tt-repeat-confirm.q95sgv`.
- Run 1: `run-1/tt-repeat-direct.qcVTXT`.
- Run 2: `run-2/tt-repeat-direct.Adikbo`.
- Run 3: `run-3/tt-repeat-direct.Iz774C`.

## Reported ratios

Ratios are public/candidate medians, in run order. Above 1 means a lower
candidate median. These rounded ratios are transcribed from the supplied
report, not reconstructed from raw samples. Samples are not pooled across runs.

| Case | Unprofiled host ratios | Profiled kernel-sum ratios |
| --- | --- | --- |
| aligned_h | 2.862x, 3.240x, 2.797x | 1.951x, 1.929x, 1.950x |
| aligned_w | 3.567x, 4.199x, 3.549x | 2.140x, 2.132x, 2.130x |
| gva_decode_h | 2.963x, 3.073x, 3.118x | 1.838x, 1.859x, 1.887x |
| gva_prefill_h | 2.941x, 3.357x, 3.300x | 2.237x, 2.247x, 2.233x |
| ragged_w | 3.616x, 4.293x, 4.075x | 1.806x, 1.799x, 1.810x |
| outer_control | 1.042x, 1.049x, 1.049x | not captured; excluded |

Every target case improved both metrics in every run. Across target cases/runs,
reported host ratios range from **2.797x to 4.293x** and device-kernel-sum
ratios from **1.799x to 2.247x**. The outer-axis control does not use the new
reader and receives no optimization credit.

## Next gate

Repeatability is established for these five BF16 benchmark inputs on the
reported Wormhole configuration. It does not establish performance for every
supported shape, repeat count or dtype, or for another architecture.

The next step is the bounded broader screen in `REPEAT_PERFORMANCE_SWEEP.md`.
It retains every case and regression instead of selecting only the five
original winners. Automatic public H/W routing remains disabled. No model-level
speedup, project CI result, maintainer acceptance or bounty eligibility follows
from this confirmation.
