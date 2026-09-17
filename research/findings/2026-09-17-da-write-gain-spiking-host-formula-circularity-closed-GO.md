---
type: finding
status: verified
date: 2026-09-17
mechanism: close a host-formula CIRCULARITY in the "spiking" da-write-gain — the spiking write-gain population's output
  SCALE was silently pinned to the host da_to_encoding_gain formula's own raw output at K_DA_REF=2.0; now the
  population's MEASURED rate range maps onto the CALLER's own (g_min,g_max) interface bounds instead
integration_faculty: da-gated-encoding (write-gain)
lane: reward/neuromodulation (scaffold-retirement adjacent)
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO 6/6. The spiking write-gain is now INDEPENDENT of the host formula's numeric scale — it derives its gain
  from its own measured spiking rate range onto the composer's legitimate (g_min,g_max) write-strength interface, not
  from da_to_encoding_gain(...K_DA_REF...). Verified: load-bearing all seeds, monotonic all seeds, lesion collapses the
  gain to zero all seeds (full attribution, mean span intact vs lesion-zero), determinism-seeded all seeds, and the
  spiking gain still tracks the host formula's SHAPE at high parity correlation — so it is a faithful spiking
  derivation, not a divergence. K_DA_REF/da_to_encoding_gain are kept ONLY as the reported parity comparator, never as a
  mechanism input. Additive + byte-identical-off (BRAIN_DA_ENCODING_SPIKING_GAIN=0). Removes a real host-dependence
  hiding inside a "spiking" mechanism (the scale was tracing the host formula, not just its shape).
runner: research/runners/_da_write_gain_spiking_derisk.py (--g-min 1.0 --g-max 3.0)
artifacts:
  - research/findings/raw/_da_write_gain_spiking/6seed_ginterface_calib.json
external: NO-EXTERNAL-NEEDED — a calibration-anchor correction confirmed by the runner's own load-bearing/monotonic/
  lesion/parity gate; the DA-gated encoding mechanism is already biology-grounded (Turrigiano/SNc-DA).
builds_on:
  - research/findings/2026-08-25-da-encoding-faculty-default-on-flip.md
---

# da-write-gain spiking — host-formula circularity closed (GO 6/6)

A research pass found a real circularity: the "spiking" da-write-gain population's output SCALE was pinned to the HOST
formula's own raw output (`_RAW_G_LO/_RAW_G_HI = da_to_encoding_gain(..., K_DA_REF=2.0, ...)`), so the spiking gain was
not independent of the host formula — it traced not just the formula's SHAPE but its numeric SCALE.

## The fix

In `_rate_to_gain`, the population's measured rate range now maps onto the CALLER's own `(g_min, g_max)` interface
bounds (the composer's legitimate write-strength range, an existing parameter) instead of onto the host formula's raw
output. `K_DA_REF`/`da_to_encoding_gain` remain ONLY as the reported parity comparator (`g_host`), never inside the
mechanism. Additive; byte-identical when off.

## Verify (from research/findings/raw/_da_write_gain_spiking/6seed_ginterface_calib.json)
<!--derived-->
GO 6/6: load_bearing_all_seeds, monotonic_all_seeds, lesion_collapses_all_seeds (lesion_attribution 1.0, mean span
intact ~1.47 vs lesion 0.0), determinism_seeded_all_seeds; parity corr with the host formula's shape ~0.996-0.9998
across seeds. So the spiking gain is a faithful, INDEPENDENT derivation (right shape, own scale), not a relabelled host
lookup. Committed on main (b02c4eaee). This reduces a named host residual on the da-gated-encoding ledger row (the write
gain no longer depends on the host formula's numeric output).
