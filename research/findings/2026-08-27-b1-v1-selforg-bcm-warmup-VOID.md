---
type: finding
status: void
date: 2026-08-27
mechanism: b1-v1-selforg-bcm-homeostatic-warmup
lane: b1-v1-selforg
artifacts:
  - research/findings/raw/_b1_v1_selforg_bcm_warmup_6seed.json
  - research/findings/raw/_b1_v1_selforg_bcm_warmup_6seed.json.prov.json
  - research/findings/2026-08-27-b1-v1-selforg-bcm-warmup-hardening-SCOPING.md
  - research/runners/_b1_v1_selforg_bcm_derisk.py
  - research/findings/raw/_b1_v1_bcm_dose_calibration_probe/warmup0.json
  - research/findings/raw/_b1_v1_bcm_dose_calibration_probe/warmup10.json
  - research/findings/raw/_b1_v1_bcm_dose_calibration_probe/warmup50.json
  - research/findings/raw/_b1_v1_bcm_dose_calibration_probe/warmup100.json
  - research/findings/raw/_b1_v1_bcm_dose_calibration_probe/warmup250.json
  - research/findings/raw/_b1_v1_bcm_dose_calibration_probe/warmup500.json
---

# B1 V1 self-org BCM homeostatic-scaling warm-up, 6-seed hardening eval: INSTRUMENT VOID — the staged 4000-step warm-up dose killed the pre-BCM operating point on 6/6 seeds

**One-line verdict.** The GPU 6-seed eval staged by
`research/findings/2026-08-27-b1-v1-selforg-bcm-warmup-hardening-SCOPING.md` came back **VOID**, not a
verdict on the hardening idea: the runner's own pre-BCM operating-point precondition (`op_point_ok`) reads
**`false` on 6/6 seeds**, `dev_firing_fraction_mean = 0.0` (dead-forward/silent, `< dev_active_lo = 0.005`
on every seed). BCM never got a live V1 to work with, so nothing about whether the warm-up stabilizes the
BCM common-mode break was tested. This is an **instrument/dose failure**, not a NO-GO on the mechanism.

## What ran

The exact staged command (`research/findings/2026-08-27-b1-v1-selforg-bcm-warmup-hardening-SCOPING.md`,
confirmed byte-identical against the run's own provenance sidecar
`research/findings/raw/_b1_v1_selforg_bcm_warmup_6seed.json.prov.json`, `git_sha=6fb0611e8`,
`sim_backend=cupy`, started `2026-08-27T06:54:31`):

```
SIM_BACKEND=cupy python -u -m research.runners._b1_v1_selforg_bcm_derisk \
    --seeds 42 43 44 100 101 102 --dev-steps 40000 \
    --bcm-gain 800 --bcm-pre-floor 0.002 --bcm-theta-alpha 0.001 \
    --warmup-steps 4000 \
    --out research/findings/raw/_b1_v1_selforg_bcm_warmup_6seed.json
```

`--warmup-steps 4000` (10% of the 40000-step development budget) is the ONLY experimental change from the
committed 6-seed BCM PARTIAL (`research/findings/raw/_b1_v1_selforg_bcm_6seed.json`,
`research/findings/2026-08-26-b1-v1-selforg-onbridge-BCM-sliding-threshold.md`), which ran the identical
config at `--warmup-steps 0` and reached the active-sparse operating point on all 6 seeds.

## Evidence (both `summary` and per-seed detail, `_b1_v1_selforg_bcm_warmup_6seed.json`)

<!--derived-->

| seed | `op_point_ok` | `dev_firing_fraction` | `v1_firing_rate` | `frac_cells_all_zero` | per-seed verdict |
|---|---|---|---|---|---|
| 42 | false | 0.0 | 0.0 | 0.9465 | VOID |
| 43 | false | 0.0 | 0.0 | 0.9470 | VOID |
| 44 | false | 0.0 | 0.0 | 0.9551 | VOID |
| 100 | false | 0.0 | 0.0 | 0.9463 | VOID |
| 101 | false | 0.0 | 0.0 | 0.9436 | VOID |
| 102 | false | 0.0 | 0.0 | 0.9436 | VOID |

`summary.overall_verdict = "VOID"`, `summary.n_op_point_verified = 0`, `summary.dev_firing_fraction_mean =
0.0`, `summary.dev_active_band = [0.005, 0.05]` (dev_firing_fraction sits below the LOW edge on every
seed — the runner's own printed label for this shape is `"SILENT (dead-forward -> VOID)"`, per
`research/runners/_b1_v1_selforg_onbridge_derisk.py`'s `develop()`/`run_seed()`). Every seed's own
`weight_diagnosis` also reads `"COMMON-MODE CONVERGENCE"` — but that diagnostic runs on the raw weights
regardless of firing and is not informative here; the operating-point check (added specifically to catch
dead-forward runs, `2026-08-14-b1-v1-selforg-onbridge-operating-point-BOUNDARY.md`) is the one that governs
the verdict, and it is unambiguous: V1 never fired during the 40000-step BCM development phase that
followed the warm-up.

## Root cause: the warm-up dose over-suppressed firing before BCM ever engaged

The pre-BCM `homeostatic_warmup()` phase (`research/runners/_b1_v1_selforg_onbridge_derisk.py:338-394`)
runs the bridge's Turrigiano multiplicative synaptic-scaling mechanism
(`cfg.enable_synaptic_scaling=True`, Hebbian/BCM frozen) for `--warmup-steps` simulation steps before the
main BCM development phase begins. `develop()` (same file, line 312) then counts V1 spikes over the
**entire** subsequent `--dev-steps` window and returns the mean per-step firing fraction — this is what
`dev_firing_fraction` measures, and it is 0.0 on every seed: the population that scaling handed to BCM was
already silent, and 40000 further steps of correlational BCM development (which requires post-synaptic
activity to produce ANY LTP/LTD signal) could not revive it. The dose (4000 steps) was chosen by
extrapolation in the preregistration — "10% of the main development budget... enough presentations... for
the firing-rate EMA and the multiplicative scaling to equalize" — and was never smoke-tested against the
operating-point precondition before the full 6-seed GPU run was staged.

**A same-session diagnostic (numpy, reduced scale, single-seed) locates the collapse as far steeper than
the preregistration's "10% dose" framing implied.** At a reduced architecture (`--n-orient 8 --n-freq 4
--n-pos 8`, `n_v1=2048` vs the production 8192) and a shortened `--dev-steps 4000`, single-seed (42) probes
across the warm-up dose (`research/findings/raw/_b1_v1_bcm_dose_calibration_probe/warmup{0,10,50,100,250,500}.json`)
gave:

| `--warmup-steps` | `dev_firing_fraction` | `op_point_ok` |
|---|---|---|
| 0 (control) | 0.01894 | true |
| 10 | 0.00063 | false |
| 50 | 0.00006 | false |
| 100 | 0.0 | false |
| 250 | 0.0 | false |
| 500 | 0.0 | false |

<!--derived-->
Firing collapses by ~30x between 0 and 10 warm-up steps and is fully dead by 100, at this reduced scale —
consistent with the production run's full collapse at 4000, but suggesting the true failure threshold is
far below the 4000-step (or even the 250/500-step) range, not a matter of dosing down by half or a
quarter. This is offered as a lead for the follow-up screen (below), not as a substitute for it: reduced
architecture and a 10x-shorter development window are a different regime from the production config, and
this probe used a single seed with no controls. It is not committed as an artifact and carries no
`verdict`/`preconditions` block of its own.

## Explicit framing: this is an INSTRUMENT failure, not a verdict on the hardening lever

Per the honesty boundary and this project's operating-point discipline
(`research/findings/2026-08-14-b1-v1-selforg-onbridge-operating-point-BOUNDARY.md`,
`tools/gates/operating_point.py`): a run whose own precondition fails is **UNDEFINED, never a negative**.
The pre-BCM homeostatic-scaling warm-up mechanism itself (Turrigiano & Nelson 2004; the mechanism the
preregistration motivated as the missing companion process for BCM's seed-variance) is **not refuted** by
this result — only the specific dose (4000 steps at the tested rate constants) is shown to be too high.
The 3/6-seed BCM PARTIAL this lever was hardening
(`research/findings/2026-08-26-b1-v1-selforg-onbridge-BCM-sliding-threshold.md`) is unaffected and stands
as before.

## Next action

**Follow-up (2026-08-27, done):** `research/findings/2026-08-27-b1-v1-selforg-bcm-warmup-dose-pilot-NOGO.md`
ran the "screen a lower dose" next action stated below — the mini-PC pool dose-screen this section originally
pointed to never materialized, so the follow-up ran a direct single-seed two-axis pilot instead (warm-up
duration 1-4000 steps AND scaling-rate down to the `syn_scaling_rate` recorded in
`research/findings/raw/_b1_v1_bcm_dose_pilot_2axis/r0.0002_n500.json.prov.json`) and found **no dose on either
axis** leaves V1 in the active-sparse band; a freeze-only/scaling-off
control isolates the forced-on Turrigiano synaptic-scaling flag itself (not the dose) as the cause. NO-GO on the
warm-up lever as implemented; see that finding for the full two-axis evidence and the mechanistic ablation. This
section's original text is kept for the record below.

~~Screen a **lower** warm-up dose that keeps `dev_firing_fraction_mean` clearly inside the active-sparse band
(`[0.005, 0.05]`) before re-running the 6-seed hardening eval. A CPU/numpy dose-screen (reduced scale, 3
seeds x a fine-grained low-dose grid) has been staged on the mini-PC pool this session — see the companion
session's pool dispatch (`research/findings/raw/_b1_v1_bcm_dose_screen/`, once results land) — to locate the
largest warm-up dose that still leaves a valid pre-BCM operating point, before spending GPU time on a second
6-seed run at a calibrated dose.~~

## Non-claims

- Does not retract or reweigh the 2026-08-26 BCM PARTIAL (3/6 seeds clear the margin at `--warmup-steps 0`).
- Does not claim Turrigiano homeostatic scaling is the wrong companion process for BCM's seed-variance —
  only that the tested dose (4000 steps, this rate constant set) drives it past the point where BCM has
  anything to work with.
- The reduced-scale local dose table above is a same-session diagnostic lead, not a controlled result: no
  multi-seed replication, no controls, and a different (smaller) architecture/`--dev-steps` than the
  production config. Treat it as motivation for the staged screen, not as its answer.

## Sources

Turrigiano & Nelson 2004, Nat Rev Neurosci 5:97 (homeostatic synaptic scaling — the mechanism under test).
Bienenstock, Cooper & Munro 1982, J Neurosci 2(1):32-48 (BCM). The lever being hardened:
`2026-08-26-b1-v1-selforg-onbridge-BCM-sliding-threshold.md`. The operating-point instrument that produced
this verdict: `2026-08-14-b1-v1-selforg-onbridge-operating-point-BOUNDARY.md`. The preregistration this run
executed: `2026-08-27-b1-v1-selforg-bcm-warmup-hardening-SCOPING.md`.
