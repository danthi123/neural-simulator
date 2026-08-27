---
status: live
type: finding
lane: integration
date: 2026-08-27
mechanism: the FIRST LEARNED cross-region edge on the shared merge pool — d6 working-memory referent → comprehension role-competition (WM-guided reference resolution), grown from ~0.05 by the substrate's own rate-window Hebbian
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_onebrain_integration_r1_wm_comprehension.py
artifacts:
  - research/findings/raw/_onebrain_integration_r1_wm_comprehension_6seed.json
builds_on:
  - research/findings/2026-08-27-onebrain-integration-phase-DESIGN.md
  - research/findings/2026-08-11-cross-region-synaptic-interaction-true-one-brain-6seed.md
---

# One-brain INTEGRATION R1 — a LEARNED cross-region edge: WM referent → comprehension role competition (6/6 GO)

The design (`2026-08-27-onebrain-integration-phase-DESIGN.md`) defined the phase AFTER migration byte-identity: the
FIRST LEARNED faculty→faculty cross-edge on the shared merge pool, admitted by a FUNCTIONAL gate (F1–F4) that
replaces byte-identity. This finding builds R1: **d6_multiref_wm's held-referent slot pool `w{k}` → comprehension's
`sel_agent`/`sel_patient` Wong-Wang accumulators** — WM-guided reference resolution. **6/6 GO** on every gate arm.
It advances the 2026-08-11 cross-region GO (which HAND-INJECTED a fixed-weight pathway) by making the edge FORM from
experience via the substrate's own plasticity. numpy CPU; **NO `sim/` edit**.

## Result — 6/6 GO on every arm (`research/findings/raw/_onebrain_integration_r1_wm_comprehension_6seed.json`)

<!--derived-->

Per-arm across seeds 42/43/44/100/101/102: **F1 6/6 · F2 6/6 · F3 6/6 · F4 6/6 · emergence 6/6 ·
lesion-recovers-migration 6/6.** The four earned verdict preconditions
(`tools.verdict.Verdict`) all hold: f2_lesion_removes_shift · migration_byte_identity · emergence (learned,
not hand-set) · moat_no_winner_from_silence.

## The mechanism (emergence-compliant; NO sim/ edit)

ONE shared spiking bridge holds BOTH organs' regions via `merge_organs([d6_multiref_wm, comprehension], wire=True)`
(config UNION, per-region-seamed wiring, settle-to-rest) — both organs are ORGAN-READ CLOSED on this pool. A SINGLE
plastic cross-edge set `w{0,1} → {sel_agent, sel_patient}` is injected at w0≈0.05 (near-zero) as the **SOLE plastic
synapse**: the design's whitelist inversion (`cp_plasticity_rate_gain=0` everywhere, then `set_plasticity_gate("wm_to_sel",1)`),
so every migrated edge is byte-frozen. The edge GROWS by the substrate's OWN rate-window Hebbian
(`_apply_branchless_hebbian`/the co-activity trace, `sim/bridge.py:1181`,`:9767`; bounded by `hebbian_max_weight`)
over experiential episodes where a referent is HELD in WM (a slow-NMDA persistent bump in `w{k}`) while its role
FIRES (the cue-driven sel competition). `enable_hebbian_learning` is flipped live ONLY around training, then frozen
for every read.

## F1 — the faculty still works (edge present)

<!--derived-->

Comprehension keeps its well-vs-ill separation on all 6 seeds: mean well margin 0.384–0.407 (all ≥ the
per-seed threshold 0.306–0.328 → comprehended), mean ill margin 0.062–0.075 (all < threshold → abstained); d6
`all_recovered=True`. The GROWN edge perturbs the well margin down from the isolated ~0.46 to ~0.40 (measured: with
the edge LESIONED the well margin recovers to ~0.46, so this ~0.06 is the edge's own effect, NOT the FP floor which
is only ~0.003). It is a genuine STANDING top-down bias — the WM pool carries low baseline activity that the
learned weight transmits to the sel pools even without a referent actively loaded (consistent with corticofugal
top-down feedback). The gate explicitly permits perturbation that does not cross the boundary: the well margin
stays ~0.07 above the decision threshold and the migrated faculty's OWN task decision is unchanged. (In F2 this
standing bias is COMMON-MODE across all conditions and cancels in the matched-control Δ.)

## F2 — the interaction is real (the crux: vary-then-lesion, both directions)

<!--derived-->

On an AMBIGUOUS balanced-cue item (baseline signed margin ≈ 0, a near-tie sel competition), VARYING which referent
is HELD in WM shifts the signed sel margin (sel_agent − sel_patient) toward that referent's LEARNED role, measured
against a MATCHED control-hold of a no-cross-edge slot pool (`w2`, same load/hold perturbation): **Δ(ref0)= +0.019
… +0.029 (toward agent)**, **Δ(ref1)= −0.018 … −0.021 (toward patient)**, all 6 seeds. LESIONING the cross-edge
(zero its weights; plasticity frozen during the read so the zeroed edge cannot regrow) collapses BOTH shifts to
**+0.000 exactly** — the shift is caused by the edge, not a confound (`tools.lab.attributable_to` = **1.0** for both
directions on all 6 seeds: the entire shift is attributable to the cross-edge). This is the two-sided load-bearing test the
design names the crux (the anti-hollow-integration check): the WM state demonstrably DRIVES comprehension's read,
and the drive VANISHES when the coupling is cut.

## F3 — no runaway

<!--derived-->

Per-region firing stays in a physiological band across the load→hold→comprehend burst (sel pools ~0.23–0.26, the
ambiguous cue pop ~0.24, the held bump ~0.09 spikes/neuron/step); the cross weight CONVERGES (soft-bounded by
`hebbian_max_weight=40`, reaching ~11–14 with the growth DECELERATING window-over-window) rather than diverging; the
pool stays alive (no silence, no seizure).

## F4 — the moat / honesty holds

<!--derived-->

(a) On a SILENT input (no cue evidence) with a referent held, the WM-only margin is 0.06–0.086 — a sub-decision
LEAN, only ~0.19–0.27 of a genuine-decision magnitude (the clear-item margin 0.320): **no winner from silence**.
(b) A CLEAR agent-dominant item is NOT flipped by a WRONG (patient-biasing) WM referent — the margin stays
strongly positive (agent), retaining >½ its no-WM value. The bias only reweights genuine ambiguity; it cannot
manufacture a fact or override clear evidence. The production comprehension DECISION additionally hard-resets WM
(structurally WM-independent), so the abstain gate never sees the bias.

## emergence — the edge LEARNED (not hand-set); the mapping formed from co-activity

<!--derived-->

The cross-edge grows from 0.05 to ~11–14 by the substrate's rate-window Hebbian (the mission's emergence bar: a
LEARNED weight, not a hand-set matrix — the residual the 2026-08-11 hand-injected GO left open). The referent→role
MAPPING is set by co-activity: `w0→sel_agent` and `w1→sel_patient` grow while the MISMATCHED (never-co-activated)
pairs `w0→sel_patient`/`w1→sel_agent` stay at 0.05 exactly on all 6 seeds (the negative control). The whitelist
held byte-perfectly: every NON-cross (migrated) weight is byte-unchanged after training (`frozen_weight_maxdrift=0`
on all seeds).

## lesion-recovers-migration — integration added ONLY the declared edge

<!--derived-->

With the cross-edge lesioned, (1) the pool's base connectivity is BYTE-IDENTICAL to the plain (no-cross-edge)
merged pool (exact dict compare of every non-cross edge, 6/6), and (2) comprehension makes the SAME decisions
(each battery item comprehended-vs-abstain unchanged, read against the migrated pool's own threshold). The residual
read wobble (maxerr 0.006–0.033) is the FP-layout floor from the extra zero cross-edges perturbing the matvec
summation order on near-zero ill-item margins — far below the ~0.33 decision gap. The migration safety net (the
byte-identity gate) remains a separate, still-runnable lane.

## Honest scope / residuals (declared; ride the design's burn-down)

<!--derived-->

- **NOT strict `self-organized`** (per `docs/TERMS.md`): the WEIGHT and the referent→role MAPPING are LEARNED from
  co-activity by the substrate's plasticity, but the EXPERIENCE is HOST-CURATED (the training schedules which
  referent co-fires with which role) and the TOPOLOGY (which regions connect) is host-chosen. Both are declared
  `scaffold_residual`s on the design's burn-down — the faithful end state has the pairing emerge from raw dialogue.
- **Two-factor Hebbian**; R2 is the three-factor neuromodulator-gated upgrade (a relevance/novelty modulator gating
  WHEN the edge learns), which needs the neuromod pool seam (curiosity's declared deferral).
- The ambiguous item is a **balanced-cue competition** (a substrate stand-in for a full pronoun-resolution
  discourse); this is the R1 de-risk, not the live-chat pronoun demo (the design's production-flip criterion).
- **Not a production flip**: this is the R1 organ-level GO on the merge pool. Wiring the cross-edge into
  `server.py brain_chat` fits AFTER R1–R3 hold their F-gate and the whole-pool multi-turn chat stays stable (design §4).

## Files

- `research/runners/_onebrain_integration_r1_wm_comprehension.py` — the R1 runner (F1–F4 gate + emergence +
  lesion-recovers-migration; 6-seed; numpy CPU; NO `sim/` edit).
- `research/findings/raw/_onebrain_integration_r1_wm_comprehension_6seed.json` — the 6/6 GO artifact + preconditions.
