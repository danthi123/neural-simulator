---
type: finding
status: contributing
date: 2026-08-20
mechanism: dendritic-plateau-coincidence-burst
lane: EPISODIC
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_gap5_d5_learn_through_use_derisk.py — a genuine D5 EpisodicDapMemory store encoded to
  BORDERLINE strength, recalled N times; each recall opens the step-2 self-terminating apical-plateau window and the
  SUBSTRATE's OWN BTSP (fused_btsp_update, gated by IS_post = max(cp_v_apical - v_hold, 0)) potentiates the co-active
  within-assembly recurrence, written back to the organ's real store; robustness measured before/after against a fixed
  lesion; no-cue + clamp anti-cheats; tools.verdict + attributable_to.
runner: research/runners/_gap5_d5_learn_through_use_derisk.py
external: NO-EXTERNAL-NEEDED — composes the arc's own GO pieces (step-2 self-terminating window +
  [[btsp-place-field-formation]]'s one-shot plateau-gated LTP); BTSP's plasticity trigger IS the dendritic plateau /
  Ca2+ (Bittner & Magee), banked in research/biology/dendritic-plateau-coincidence-burst.md.
artifacts:
  - research/findings/raw/_d5_learn_through_use/seed42.json
  - research/findings/raw/_d5_learn_through_use/seed43.json
  - research/findings/raw/_d5_learn_through_use/seed44.json
  - research/findings/raw/_d5_learn_through_use/seed100.json
  - research/findings/raw/_d5_learn_through_use/seed101.json
  - research/findings/raw/_d5_learn_through_use/seed102.json
  - research/findings/raw/_d5_learn_through_use/summary_6seed.json
---
# GO (6/6): USING a memory STRENGTHENS it on the real D5 organ — recall-driven, plateau-gated BTSP — ARC-1 (learn-through-use) CLOSED

Artifact: research/findings/raw/_d5_learn_through_use/summary_6seed.json (6/6 GO) · the per-seed
research/findings/raw/_d5_learn_through_use/seed42.json + seed43.json + seed44.json + seed100.json + seed101.json + seed102.json.

**One line.** The capstone of the D5 learn-through-use arc, and its actual mission capability: **a memory the brain
USES (recalls) becomes more robust** — completing from a sparser cue and surviving a within-recurrence lesion it
previously failed — via the substrate's OWN dendritic-plateau-gated BTSP, on the real production D5 organ, NO `sim/`
edit and NO host weight formula. 6/6-seed GO, adversarially verified (4 lenses CONFIRMED including two independent live
reproductions of the decisive control; no confound). Steps 1
([[2026-08-20-ecker-real-d5-store-does-NOT-reactivate-via-soma-recurrence-dendritic-latch-is-the-read]]) + 2
([[2026-08-20-d5-dendritic-latch-self-terminates-into-discrete-apical-plateau-BTSP-window-6seed-GO]]) → this closes arc-1.

## The mechanism (the substrate's own BTSP, plateau-gated; NO `sim/` edit, NO host formula)
During a recall the runner triggers the engine's OWN BTSP block (`sim/bridge.py` 4b-bis `fused_btsp_update`, guarded by
`cfg.enable_btsp`): each step `dw = eta · Etilde_pre · IS_post · (w_max − w)`, where `Etilde_pre` is the seconds-long
presynaptic eligibility (low-pass of firing) and **`IS_post = max(cp_v_apical − v_hold, 0)` is the step-2 dendritic
apical plateau itself**. The strengthened weights land in `mem.bridge.cp_connections.data`, which IS `mem.R.C.data` by
object identity — the organ's own store, the same array `recall()` reads (verified: not a scratch copy). The runner
never computes or assigns a `dw`; `git diff sim/` is empty vs main AND origin/main. Boundedness of each episode comes
from step-2's Ecker-`b` apical adaptation collapsing the plateau (IS_post→0).

## The 6-seed verdict (GO 6/6) + the decisive control
<!--derived-->
Borderline store (encode_train_events=15 vs a full-GO store's 40 — deliberate robustness headroom), 8 recalls:
- **STRENGTHENS** — after use the store survives a within-recurrence lesion it did NOT before: at the fixed headline
  lesion 0.7, seed42 held-cell completion **0.1667 → 0.6667** (1/6 → 4/6 held cells; the level the baseline fails,
  measured matched before/after with one shared W0 + snapshot, so the gain is 100% attributable to the weight change).
  Triply supported per seed: max-surviving-lesion up, min-cue-current down (160→120 pA), survives-a-new-lesion.
- **BOUNDED** — per-episode `dw_on` strictly shrinks (seed42 6.271 → 0.028), `w_dog_final` (74.6–83.2) stays below the
  soft-bound ceiling 100. No blow-up.
- **SPECIFIC** — the never-recalled 'cat' within-recurrence drifts ≤ 5% of dog's gain (cat_drift ≤ 0.566; ~0.97–1.0
  dog-vs-cat attributable); between-assembly weight flat.
- **THE DECISIVE CONTROL (no-window):** both no-cue AND clamp give `dw = 0.0` on every seed. Two independent live
  reproductions confirm the clamp changes ONLY the window: cue firing 48.0 vs 48.0 (ratio 1.000) and presynaptic
  eligibility 0.006567 vs 0.006569 (ratio 1.000) are held IDENTICAL between arms, while forcing `cp_v_apical` to rest
  zeroes the plateau (IS_post 28.23 → 0.0) and strengthening vanishes (dw 11.8 → 0.0).

## Why this is LEARN-THROUGH-USE, not re-encoding — the two legs (state both; the clamp alone is not enough)
<!--derived-->
The clamp control proves the potentiation requires the **plateau (IS_post>0), not the cue current** — but the clamp
alone does NOT separate retrieval-practice from restudy (re-encoding also needs a plateau). The learn-through-use
content rests on a SECOND leg, the **protocol**: only the ~5–13 CUE cells are externally driven; the 6 HELD cells are
DISJOINT and receive ZERO external current, so their plateaus — and hence the cue→held / held→held within-assembly LTP
the robustness readouts depend on — can arise ONLY from **completion = recall**. So the strengthening of the completion
pathway is retrieval-driven. Honest scope: this is recall-driven (use-dependent) potentiation via the substrate's OWN
plateau-gated BTSP — the SAME LTP substrate engaged by retrieval, not a distinct mechanism.

## Honest scope (the reframings the verification required — stated, not hidden)
<!--derived-->
1. **MONOTONE = one-shot-then-asymptote**, not unbounded climbing: `dw_on` is strictly decreasing every seed, the
   robustness trajectory is non-decreasing then plateaus — faithful to one-shot BTSP (it deepens cells that already
   plateau, it does not recruit new held cells).
2. **BOUNDED = interference-not-weight-blowup**: the multiplicative soft-bound `(w_max−w)` caps each weight by
   construction, so "weights don't blow up" is partly trivial. The LOAD-BEARING boundedness is the first-class
   contrast — a PERSISTENT (adapt-OFF) latch causes INTERFERENCE (unrelated drive spuriously potentiates
   non-member→dog-held, interf_dw 1.1–2.1) that step-2's self-terminating window SUPPRESSES (0.0–0.63). The window is
   what prevents interference-corruption, not weight runaway.
3. **BORDERLINE store = fragile-but-working**: it completes at full cue but is brittle to lesion — the deliberate
   headroom, not a strong store. `store('dog')` is not byte-reproducible (step-1's seam); every draw across two 6-seed
   runs gave the same qualitative GO.
4. **Specificity rests on the WEIGHT DRIFT** (cat_drift, above), NOT the vacuous completion sub-check (cat is a
   never-formed floor, ml_cat = −1.0 before and after — that leg proves nothing and is not relied on).

## Adversarial verification + next
Four lenses (mechanism-legitimacy, window-isolation [decisive], strengthening-reality/specificity, verdict/gate-
readiness) all CONFIRMED, no confound; the synthesizer independently reproduced the clamp isolation live (cue+eligibility
byte-identical, only IS_post zeroed) and confirmed sim/ zero-diff + the object-identity write-back + the fixed-lesion
matched before/after. Runner cleaned (a dead `--lesion-frac` argparse flag removed pre-bank). NEXT (step-4, the
production-integration rung = the mission spine): wire the recall→self-terminating-window→BTSP loop under
`continuous_engine.py`'s idle tick (additive, default-off) so a live-chat recall CONSOLIDATES the used memory between
turns, and prove it is LOAD-BEARING (a used memory measurably more robust in a later turn, the gain vanishing under
lesion) — then the production-default flip (owner UX). (Agent-built; parent ran the 4-lens adversarial workflow, verified
the substrate-BTSP + object-identity + sim-untouched + the decisive control from the artifacts, corrected the headline
number to the banked 0.1667→0.6667, and removed the dead flag before banking.)
