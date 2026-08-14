---
type: finding
status: live
lane: gap#5
date: 2026-08-14
mechanism: one-brain-merge
---

# Config-superset surprise residual: BOTH mapped levers (robust margin read, higher cross gain) fall short — the three residual axes share ONE cause, the surprise organ's seed-dependent recall (PARTIAL)

**Verdict:** PARTIAL (cell `0.5:PRX` = 1/6, unchanged from `0.5:PR`) · **Backend:** numpy (bit-exact CPU) · 6 seeds
(42,43,44,100,101,102) × 2 cells (PR, PRX).
**Refines** `research/findings/2026-08-14-per-region-homeostasis-reconciles-configsuperset-BOUNDARY.md` — tests its two
"mapped next levers" and shows each falls short, correcting its "a robust read closes surp_answer_preserved+surp_byte_id"
prediction with the real per-axis cause.

**Runner:** `_one_brain_merge_configsuperset_production_derisk.py` (new `0.5:PRX` cell = `0.5:PR` + L1) ·
**Artifact:** `research/findings/raw/_one_brain_merge_configsuperset_robustread_6seed.json` · **NO `sim/` edit** (`git diff sim/` empty).
**Reproduce:**
```
SIM_BACKEND=numpy OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m \
  research.runners._one_brain_merge_configsuperset_production_derisk \
  --seeds 42,43,44,100,101,102 --cells 0.5:PR,0.5:PRX \
  --out research/findings/raw/_one_brain_merge_configsuperset_robustread_6seed.json
```

## Result: PRX == PR == 1/6 — neither surprise-side lever lifts its axis

| cell | GO/6 | comp byte-id | comp AUC≥.80 | comp ans | surp byte-id | surp ans | cross LB | det / pool / iso |
|---|---|---|---|---|---|---|---|---|
| 0.5, PR  | 1 | 6 | 6 | 6 | 4 | 2 | 2 | 6 / 6 / 6·6 |
| 0.5, PRX | 1 | 6 | 6 | 6 | 4 | 2 | 2 | 6 / 6 / 6·6 |

Comprehension is FULLY reconciled (6/6/6) — the [HOMEO] closure holds. The surprise side does not close: the three
merge-specific axes stay at PR's values. L1's only measurable effect was one preserved fact (seed 44 surp-ans 22→23).
Per-seed PRX: cross_lb True on 42,100 (+25 Hz) / False (+0) on 43,44,101,102; surp-ans 24,23,23,21,19,24 /24.

## Why each residual axis does NOT close (each cause isolated, not guessed)

**cross_load_bearing (2/6) — L2 "raise the cross gain" is FALSIFIED.** A trained sweep (seeds 42/43, cross weight
40→250) shows: on a WORKING seed (42) raising the gain 40→60 SILENCES sel_agent under contradict (25→0 Hz — the strong
contradict surprise over-drives the NMDA-bearing sel_agent out of its spiking regime, a depolarization-block-like
effect); on a WEAK seed (43) NO weight (≤250) ever makes contradict fire sel_agent (~5-6 Hz surprise_S is simply below
its fixed `vpeak`). The anti-cheat "cap CROSS_WEIGHT_HIGH at the largest load-bearing value" therefore caps it at 40 ==
CROSS_WEIGHT (zero lift). Higher gain cannot help; the binding quantity is the surprise_S contradict rate vs a fixed
threshold, not the cross weight.

**surp_byte_id (4/6) — NOT a hair-trigger; L1's margin is orthogonal to the cause.** A symmetric-read diagnostic
(measure merged and decoupled with identical treatment) gives merged == decoupled BYTE-IDENTICAL per-fact rates on ALL
seeds incl. 101/102, so byte-id holds at EVERY margin incl. mu=0. The run's 4/6 is an artifact of the harness's
ASYMMETRIC read-isolation: the merged comp read restores the `_S` slice (`iso_mask=surp_mask`), the decoupled uses
`iso_mask=None`, so per-region homeostasis drifts the two `_S` thresholds apart during the comp battery → a ~0.49 Hz
rate split → a near-threshold fact flips on 101/102. A margin on the FINAL read cannot fix upstream threshold drift.

**surp_answer_preserved (2/6) — native false-alarms, not common-mode near-boundary flips.** The native dt=1.0
standalone has genuine false-alarm confirm facts (per-fact confirm rates up to 5.61 Hz seed100, 3.7 seed101, 3.47
seed44) — its recall is imperfect and seed-dependent. The merged (dt=0.5) is CLEANER (confirm ≈ 0 on all facts). The
axis penalizes the merged for disagreeing with a NOISIER baseline. A margin big enough to absorb a ~5 Hz native
false-alarm (mf≥0.8) also nulls genuine ~5 Hz violations → the pre-registered native-floor anti-cheat FAILS. At the
floor-safe mf=0.25 the false-alarms survive and the mismatch stands.

## The one deeper cause + the named next lever

All three residuals reduce to ONE thing: the surprise organ's recall is imperfect and seed-varying — weak/nonuniform
contradict firing (blocks the cross on 4/6 seeds), native confirm false-alarms (blocks answer-preservation), and
homeostasis-sensitive near-threshold facts (surp_byte_id). The named deeper closure is the **fully-learned CA3
all-to-all recall rung** (`2026-06-05-D-cue-recall-RESOLVED`) — uniform, strong, false-alarm-free violation firing
across seeds would drive the cross at the SAFE weight of 40 AND remove the native false-alarms — not a per-read margin.
A cheaper same-rule-both-sides sub-fix for surp_byte_id alone: symmetrize the decoupled twin's comp read to also protect
the `_S` slice (proven to give byte-identical rates), 4→6; it does not change the GO verdict (cross_lb still binds).

## Anti-cheats (verified)

- **Byte-identical-when-off:** new PR rows are byte-identical to the prior perregion artifact on all 6 seeds across
  every decision field (go, one/det/nmda, comp byte-id/AUC/ans, surp byte-id/ans, cross intact/lesion, read-iso).
  New behaviour is reachable ONLY via the `PRX` tag; `git diff sim/` empty.
- **Ablation is load-bearing-NEUTRAL, honestly:** PR (levers off) and PRX (L1 on, floor=True + cross_confirm_below_
  contradict=True on all 6 PRX seeds) both = 1/6. The levers are ACTIVE but produce no net axis gain — an honest null,
  not a cheat that loosened the gate.
- **Native-floor holds** (`native_robust_floor_ok`=True, 6/6 PRX): mf=0.25 does not trivially null the faculty.
- **Cross stays load-bearing not a floor artifact:** `cross_confirm_below_contradict`=True 6/6 (no inversion at 40);
  cross_lesion=0 all seeds. Determinism (cfg.seed, build-twice byte-id) 6/6; comprehension NOT regressed (6/6, byte-id
  to PR); one shared pool (2088) 6/6; read-isolation 6/6 both organs; same `_classify_surprise` rule both sides;
  windowed `cp_firing_states` read; `_semantic_contrast` tripwire intact.

## What stays a residual / follow-on

The [HOMEO] BOUNDARY's comprehension-side closure stands (production role read fully reconciled on one shared pool).
The surprise side does NOT reach full-cell GO via the two mapped levers; the residual is redirected from "a robust read
/ higher gain" to the CA3-recall rung (uniform violation firing). This is a DE-RISK; the production `shared=` wiring +
default flip remains the gated follow-on, unblocked on comprehension, surprise-blocked on recall uniformity.

## Files

- Runner: `_one_brain_merge_configsuperset_production_derisk.py` gains `0.5:PRX` (L1 robust margin read; L2 falsified,
  `CROSS_WEIGHT_HIGH`=40==`CROSS_WEIGHT`), a shared `_classify_surprise` rule, and the native-floor / cross-direction
  anti-cheats. Additive, default-preserving. **NO `sim/` edit.**
- Artifact: `research/findings/raw/_one_brain_merge_configsuperset_robustread_6seed.json` (+ `.prov.json` sidecar).
</content>
