---
status: infrastructure_failure
lane: gateb-v14-performance
date: 2026-08-04
type: finding
---

# V14 Stage A Historical Control Construction Failed

The repaired performance matrix used candidate `4d58091cc` and historical
control `6c9034991`. It completed three candidate workers, then stopped at
sequence 4 because the shared harness passed the new default-off
`enable_snr_direct_outputs` field into the historical `CoreSimConfig`, which
does not define that field.

This is a harness compatibility failure. The partial candidate timings are not
a performance or promotion verdict, no scientific seeds were opened, and the
remaining cells were not started. The terminal receipt is
`research/findings/raw/v14_stageA_performance_4d58091cc.json`.

The repair omits the field only when an older config type does not define it
and direct outputs are false. It fails closed if direct outputs are requested
on such a source. A focused historical-control worker completed after the
repair, so a newly sealed complete matrix is authorized; the partial receipt
must not be combined with it.
