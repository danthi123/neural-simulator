---
type: finding
status: no-go
date: 2026-08-27
mechanism: b1-v1-selforg-bcm-homeostatic-warmup
lane: b1-v1-selforg
seeds: [42]
seed-waiver: single-seed dose-pilot locating a viable OPERATING POINT before any multi-seed spend, per
  CLAUDE.md "pilot before N-seed spend"; not a generalization claim, and no positive verdict is asserted.
artifacts:
  - research/findings/raw/_b1_v1_bcm_dose_pilot_2axis/freeze_only_scaling_off_control.json
  - research/findings/raw/_b1_v1_bcm_dose_pilot_2axis/sanity0.json
  - research/findings/raw/_b1_v1_bcm_dose_pilot_2axis/nf_iso_false_1.json
  - research/findings/raw/_b1_v1_bcm_dose_pilot_2axis/nf_iso_true_100.json
  - research/findings/raw/_b1_v1_bcm_dose_pilot_2axis/r0.0002_n500.json
  - research/findings/raw/_b1_v1_bcm_dose_calibration_probe/warmup0.json
  - research/findings/2026-08-27-b1-v1-selforg-bcm-warmup-VOID.md
---

# B1 V1 self-org BCM homeostatic warm-up: two-axis dose pilot finds NO viable dose — the forced-on Turrigiano synaptic scaling collapses V1 near-instantly and rate-insensitively; NO-GO on the warm-up AS IMPLEMENTED

**One-line verdict.** Following up the 2026-08-27 VOID (`research/findings/2026-08-27-b1-v1-selforg-bcm-warmup-VOID.md`,
6/6 GPU seeds dead-forward at `--warmup-steps 4000`), a same-session single-seed reduced-scale dose pilot swept
**two independent axes** — warm-up duration (1 to 4000 steps) and the scaling rate constant (0.02 down to 0.0002,
a 100x range) — and found **no combination that leaves V1 in the active-sparse operating band**. A decisive
ablation isolates the cause: the IDENTICAL pre-development freeze phase with synaptic scaling left OFF is
**stable across every tested duration** (`dev_firing_fraction` 0.01262–0.01894, `op_point_ok=true` at 1/10/100/500/4000
steps). This is not a "10% dose was too high" problem — it is a **mechanism-level instability** in the forced-on
Turrigiano multiplicative synaptic-scaling phase itself, at this substrate's very sparse V1 pathway (density
0.001). **NO-GO on the warm-up hardening lever as implemented; the 6-seed hardening re-eval is NOT staged**
(no calibrated dose exists to run it at — staging one would reproduce the same VOID).

Artifact: `research/findings/raw/_b1_v1_bcm_dose_pilot_2axis/freeze_only_scaling_off_control.json`.

## What ran (all single-seed 42, reduced-scale: `--n-orient 8 --n-freq 4 --n-pos 8 --dev-steps 4000`, `--bcm-gain
800 --bcm-pre-floor 0.002 --bcm-theta-alpha 0.001` — identical config to the predecessor's own numpy probe)

**Axis 1 — warm-up duration, default rate (0.02), the COMMITTED runner's own code path**
(`research/findings/raw/_b1_v1_bcm_dose_pilot_2axis/nf_iso_false_*.json`, plus the already-committed
`research/findings/raw/_b1_v1_bcm_dose_calibration_probe/warmup*.json` for cross-check):

| `--warmup-steps` | `dev_firing_fraction` | `op_point_ok` |
|---|---|---|
| 0 (control) | 0.01894 | true |
| 1 | 0.00006 | false |
| 2 | 0.00028 | false |
| 3 | 0.00062 | false |
| 5 | 0.00081 | false |
| 10 | 0.00063 | false (matches the already-committed `warmup10.json` exactly) |
| 50 / 100 / 250 / 500 (committed probe) | ≤0.00006 / 0.0 / 0.0 / 0.0 | false |

Collapse is already >30x below the `dev_active_lo=0.005` floor at the SINGLE smallest nonzero dose (1 step) and
never recovers at any larger dose tested — not a graded dose-response with a safe low end, a near-instant
transition.

**Axis 2 — scaling rate constant, `--syn-scaling-rate` swept 100x below default**
(`research/findings/raw/_b1_v1_bcm_dose_pilot_2axis/r0.*_n*.json`, `syn_scaling_rate` recorded in each
provenance sidecar):

| `--syn-scaling-rate` | steps 100 | steps 500 | steps 4000 |
|---|---|---|---|
| 0.005 (4x gentler) | 0.0 | 0.0 | 0.0 |
| 0.001 (20x gentler) | 0.0 | 0.0 | 0.0 |
| 0.0002 (100x gentler) | 0.0 | 0.0 | (not run) |

At the gentlest tested rate, the per-tick formula (scale factor = 1 plus rate times rate-error, clipped to
[0.95, 1.05]) predicts a cumulative multiplicative drift over 500 ticks on the order of a tenth of a percent —
numerically negligible (an arithmetic estimate from the formula, not a measured value). <!--derived--> The
observed effect (exact full-population silence) is far larger than the documented mechanism accounts for,
across every step count tried, with no sensitivity to the 100x rate range tested. This rules out "the rate was
simply too aggressive" as the story.

**Decisive ablation — isolate which of (freeze Hebbian) vs (force scaling on) causes the collapse.**
`research/findings/raw/_b1_v1_bcm_dose_pilot_2axis/nf_iso_true_*.json` reruns axis 1 with the concurrent per-cell
threshold-adaptation homeostasis
(`cfg.enable_homeostasis`, section 5a of `sim/bridge.py`, normally always-on per `build_v1_bridge` line 230)
forced OFF for the warm-up window only (external monkeypatch of `homeostatic_warmup()`, no repo file touched) —
scaling (section 5b) still forced on. Result: **collapse is essentially unchanged** (n=1: 0.00009 vs 0.00006; n=10:
0.00144 vs 0.00063; n=100: 0.0 vs 0.0) — ruling out "two homeostatic mechanisms double-regulating the same
target" as the cause. `freeze_only_scaling_off_control.json` then reruns the SAME freeze protocol with scaling
left OFF (never forced on) — the only remaining difference from `homeostatic_warmup()`. Result: **stable at every
tested duration up to the full 4000-step budget** (0.01262–0.01894, `op_point_ok=true` throughout). This isolates
the forced-on `enable_synaptic_scaling` flag itself — not the Hebbian freeze, not the duration, not a
double-homeostasis interaction — as the necessary and sufficient trigger.

## Root cause: the Turrigiano multiplicative scaling implementation, not the warm-up dose

Per the CLAUDE.md wall-reframe ("what else does the real system run alongside this, that we replaced with a
constant?"), this arc's premise was that BCM's seed-variance is missing a homeostatic-scaling companion process
(Turrigiano & Nelson 2004). That premise survives — `freeze_only_scaling_off_control.json` shows the surrounding
protocol (frozen pre-development under threshold-homeostasis alone) is perfectly stable. What fails is the
SPECIFIC forced-on multiplicative-scaling implementation (`sim/bridge.py` ~line 11088–11125,
`scale_factors = 1 + cfg.synaptic_scaling_rate * rate_error`, clipped `[0.95, 1.05]` per tick) at this
substrate's very sparse V1 pathway (retina→V1 density 0.001, few afferent synapses per postsynaptic cell): its
observed effect size is far larger and far less rate-sensitive than the documented per-tick formula predicts,
consistent with a genuine implementation issue in that code path for this sparsity regime rather than a
parameter that can be dosed down. Diagnosing the exact numerical mechanism inside `sim/bridge.py` is a separate,
narrower engineering task — out of scope here (no `sim/` edit made or needed for this dose-pilot verdict).

## Why the 6-seed hardening re-eval is NOT staged

The task that motivated this pilot called for staging the 6-seed BCM-warmup hardening eval "at the sane dose."
No sane dose was found across two independent axes (duration 1–4000 steps; rate 0.02→0.0002, 100x range), all
single-seed/reduced-scale but internally cross-validated (the `--warmup-steps 10` point exactly reproduces the
already-committed calibration probe's 0.00063). Staging 6 GPU seeds at any of these known-collapsing
configurations would reproduce the identical VOID at 6x the cost — exactly the "N-seed spend before a pilot"
failure this whole arc exists to prevent. **One cheap confirmatory check remains queued**
(`bash tools/gpu_queue.sh add`, production scale `n_v1=8192` vs this pilot's reduced 2048, 1 seed, `--warmup-steps
500 --syn-scaling-rate 0.0002`, writing into `research/findings/raw/_b1_v1_bcm_dose_finegrain_probe/` once it
runs — not yet materialized, so not cited above as an artifact) to rule out a reduced-scale-architecture artifact
before treating this as final; PARTIAL on that one point, pending harvest by the queue controller. If it comes
back non-silent, that would justify a follow-up production-scale pilot (not a 6-seed commitment) — that decision
is deferred, not taken here.

## Non-claims

- Does not retract or reweigh the 2026-08-26 BCM PARTIAL (3/6 seeds clear the margin at `--warmup-steps 0`) or the
  2026-08-27 VOID (its 6/6 GPU dead-forward result at `--warmup-steps 4000` stands; this finding explains WHY no
  lower dose would have fared better).
- Does not claim Turrigiano homeostatic scaling is wrong as a companion-process HYPOTHESIS for BCM's
  seed-variance — the freeze-only control shows the surrounding protocol is fine; only THIS scaling
  implementation, at THIS sparsity, is shown unstable.
- Single-seed throughout (a labelled pilot, `seed-waiver` above) — not a generalization claim, and no positive
  verdict is asserted that would need 6-seed evidence.
- The queued production-scale confirmatory point has not returned; this finding's NO-GO rests on the
  reduced-scale two-axis pilot alone.

## Sources

Turrigiano & Nelson 2004, Nat Rev Neurosci 5:97 (homeostatic synaptic scaling). **External literature check
(this session, DR gate for lane b1-v1-selforg):** a bioRxiv preprint specifically on synaptic-scaling stability
in feedforward rate-propagation circuits — "Local homeostatic scaling supports stable rate propagation under
noise and heterogeneity" (bioRxiv 2025.11.25.689806) — reports that stable transmission in feedforward circuits
needs restrictive conditions on the scaling feedback gain (rate propagation amplifies or decays away from the
optimum; scaling is stable only when its feedback gain is small relative to the update/transport delay).
Overman & Clopath, eLife 2024;13:e88376 report that homeostatic synaptic scaling needs a STRUCTURAL-plasticity
companion process to hold firing rate stable in sparse recurrent networks after synapse loss — scaling alone is
not sufficient. Both are consistent with this finding: this substrate's V1 pathway is purely feedforward and
very sparse (retina→V1 density 0.001) with no recurrent or structural companion process to buffer the scaling
feedback loop, plausibly why even a 100x-reduced per-tick gain still drove it outside the stable regime. Also
locally relevant (project's own prior finding, not this arc):
`research/findings/2026-08-07-laneD-v1-pooler-trace-homeostatic-scaling-REGRESSES-not-the-companion-process.md`
— Turrigiano scaling REGRESSED a different V1-pathway pooler's operating point too, there via an interaction
with a hard connectivity threshold; a second, independent instance of this scaling implementation failing as a
"missing companion process" fix at a V1-adjacent wall in this codebase. The predecessor VOID:
`research/findings/2026-08-27-b1-v1-selforg-bcm-warmup-VOID.md`. The lever being hardened:
`research/findings/2026-08-26-b1-v1-selforg-onbridge-BCM-sliding-threshold.md`. The operating-point instrument:
`research/findings/2026-08-14-b1-v1-selforg-onbridge-operating-point-BOUNDARY.md`. Mechanism code:
`sim/bridge.py` (synaptic scaling, ~line 11088–11125), `research/runners/_b1_v1_selforg_onbridge_derisk.py`
(`homeostatic_warmup()`, `build_v1_bridge()` line 230).
