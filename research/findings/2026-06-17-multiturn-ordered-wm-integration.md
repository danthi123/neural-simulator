# Multi-turn agent with an ORDER-ENCODED discourse buffer — multi-referent disambiguation in PRODUCTION (GO)

**Date:** 2026-06-17
**Status:** **GO, 6 seeds (42 43 44 100 101 102).** The production multi-turn conversational agent now resolves a
turn-2 bare pronoun to the **foregrounded (most-recent gamma-slot) referent among SEVERAL held**, on the
project's spiking resonate-and-fire phasor substrate. This is the production version of the CYCLE-135 de-risk
(`2026-06-17-ordered-wm-position-binding-derisk.md`): the order-encoded working memory replaces the rate-attractor
discourse buffer that failed multi-referent disambiguation across three converging negatives. All four
load-bearing controls pass **6/6** (multi-referent resolution / the order-control FLIP / the no-confab moat /
single-referent un-regressed), plus the V1 production path is unbroken. **No `sim/` edit; reuse-by-import.**

## The gap this closes

The production `MultiTurnAgent` (`research/runners/multi_turn_agent.py`) holds discourse referents in the
rate-attractor `SpikingLoopContextBuffer` — a **SET with no order**, whose winner is decided by intrinsic basin
asymmetry. That substrate is production-GO for the **single-referent** case (one held antecedent → a turn-2
pronoun resolves; `2026-06-17-multiturn-anaphora-derisk-GO.md`), but it is exactly the substrate that **FAILED
multi-referent disambiguation** across three converging negatives — recency, a salience boost, and
biased-competition WTA, all 0/6 on the order-control (`2026-06-17-multireferent-disambiguation-NEGATIVE.md`,
`-biased-competition-wta-multireferent-derisk.md`). With several referents held, a bare pronoun cannot be
resolved to the foregrounded one because the buffer has no notion of *which slot / how recent*.

The validated fix (CYCLE 135): an **order-encoded WM** (theta-gamma / Lisman-Idiart) — bind each held item to a
gamma-slot **position** phasor, bundle into one composite, read item-at-slot-k by spiking `unbind(C, position_k)`.
The winner is **which slot you read**, so it flips deterministically with the discourse order. This integration
puts that order-encoded WM into the agent as the discourse buffer.

## Build (no `sim/` edit, reuse-by-import)

1. **`research/runners/ordered_position_wm.py`** — the de-risk's `OrderedPositionWM` promoted to a clean
   production module. It **subclasses the deployed composer `RFPhasorComposer`** (resonate-and-fire phasor
   neurons + complex synapses); position phasors are added to the composer's extensible `roles` dict (the same
   spiking machinery as SVO role vectors). The de-risk runner
   (`_phaseB_ordered_wm_position_binding_derisk.py`) now **imports** this module (single source of truth) via a
   thin frozen-pre-registration wrapper that pins D=256 / n_slots=7 / the frozen 0.15 threshold — its numbers are
   **byte-identical** (verified: seed-42 recall 1.000/1.000/1.000, disambiguation 1.000/1.000, moat@frozen
   119/120+59/60, moat@principled 60/60 — matching the de-risk's published table).
   - **Familiarity-threshold change vs the de-risk.** The de-risk pre-registered a *frozen* 0.15 and honestly
     reported BOUNDARY because that value sat in the bundle-cross-talk noise tail. The production module instead
     **calibrates** the threshold from the WM's own measured groundable-vs-ungroundable separation
     (`calibrate_threshold`, the principled `cleanup_separated` placement rule, Bogacz-Brown familiarity). On the
     agent's 6-referent buffer at D=128 this calibrates to **0.266–0.315** per seed (groundable real-slot match
     ~0.6, ungroundable empty/scrambled ~0.17 → a clean midpoint). This is threshold hygiene, **not**
     tune-to-pass: the calibration measures the floor from the WM's intrinsic match distributions, never from a
     downstream test. (A code-fidelity bug found + fixed during the build: the calibration must draw its probe
     items from the *cleanup-candidate set*, since a discourse slot only ever holds a referent — calibrating over
     the full vocab injected action words at slots and collapsed the groundable floor → spurious fallback.)

2. **`research/runners/multi_turn_agent_v2.py`** — `MultiTurnAgentV2`. Wraps the same
   `BrainConversationalAgent` (parser + composer + dlPFC) as V1; its discourse buffer is an `OrderedPositionWM`
   built with the **same (seed, D=128, vocab)** as the agent's composer — so the WM's concept codes are
   **byte-identical** to the composer's (both draw from `default_rng(seed)` over the sorted vocab; the position
   phasors use a disjoint `seed+1000` stream), and a word read out of a slot is a genuine composer concept the
   Q&A path uses directly (code-parity asserted in the test). The buffer cleans up slot reads against the
   **referent subset only** (so a discourse pronoun never resolves to an action word). Referents are introduced
   in surface order (subject then object) into a sliding window of the last `n_slots`; a bare pronoun resolves to
   the **most-recent occupied slot** via spiking `unbind`, familiarity-gated → abstain (None) when the discourse
   is empty. The single-referent path is preserved (one referent ⇒ the most-recent slot *is* that referent), so
   V2 is a strict superset of V1's anaphora capability.

## Results — the four load-bearing controls, 6 seeds

| seed | calibrated thr. | (1) multi-referent | (2) ORDER-control FLIP | (3) no-confab moat | (4) single-referent regression |
|---|---|---|---|---|---|
| 42  | 0.306 | PASS | PASS | PASS | PASS |
| 43  | 0.315 | PASS | PASS | PASS | PASS |
| 44  | 0.277 | PASS | PASS | PASS | PASS |
| 100 | 0.266 | PASS | PASS | PASS | PASS |
| 101 | 0.283 | PASS | PASS | PASS | PASS |
| 102 | 0.303 | PASS | PASS | PASS | PASS |
| **count** | — | **6/6** | **6/6** | **6/6** | **6/6** |

- **(1) Multi-referent resolution (the capability).** `hear("dog see cat")` holds `[dog, cat]` (cat
  most-recent); a turn-2 bare pronoun `it` resolves to **cat** and `what does it eat?` reads the cat's fact →
  **fish**. 6/6.
- **(2) ORDER-CONTROL (the wall).** Swap the last-introduced referent — `hear("cat see dog")` holds `[cat, dog]`
  (dog most-recent) — and the SAME pronoun now resolves to **dog** → `what does it eat?` → **worm**. The
  resolution **FLIPS** (cat→fish vs dog→worm) purely because the discourse order changed. This is exactly the
  order-control the three rate-buffer negatives failed at **0/6**; here it is **6/6**, because the winner is
  *which slot you read*, not an intrinsic basin.
- **(3) No-confab moat (load-bearing).** With no referent held (empty discourse), `it` grounds nothing →
  `most_recent_referent()` is **None**, `what_does("it", …)`, `reason_chain("it", …)` abstain (**None**), and
  `is_it_true("it", …)` is **"unknown"** — no confabulated antecedent. 6/6. (Per the owner's 2026-06-17
  moat-relaxation the moat is no longer a hard gate; here it is **free** from the composer's familiarity gate and
  reported clean — zero breaches across all seeds.)
- **(4) Single-referent regression.** Both the genuinely-single-referent case (introduce only `cat` → `it`=cat →
  fish) and the exact V1 production contract (`hear("dog chase cat")` → `it`=cat → fish, and a pronoun-cued
  2-hop chain → worm) still resolve. 6/6. No regression.

**Test suite:** `tests/test_multi_turn_ordered_wm.py` — **31/31 PASSED** (the 4 controls × 6 seeds = 24, plus
V1-unbroken, plus code-parity × 6). The existing V1 gate `tests/test_multi_turn_agent.py` — **3/3 PASSED** (no
regression from promoting the module / adding V2).

## Honest scope

- **Capability + moat: robustly GO.** Multi-referent disambiguation incl. the order-control flip is 6/6, and the
  familiarity moat is **clean** at the principled (calibrated) threshold across all seeds — the BOUNDARY of the
  de-risk (a frozen threshold in the noise tail) is resolved by the principled placement, as the de-risk itself
  prescribed.
- **What this is and isn't.** This is the production *integration* of an already-validated mechanism: the
  agent now disambiguates among **several** held referents by slot. "Most-recent" is the foregrounding heuristic
  (the usual antecedent of a pronoun); richer salience (e.g. binding a pronoun to a non-most-recent referent by
  syntactic role or attention) is addressable since *any* held slot is readable (`referent_at(slot)`), but the
  selection policy beyond recency is a follow-on. Validated at vocab 9 (6 referents + 3 actions), D=128, on the
  CPU/numpy backend (the spiking RF composer runs each op as a small `SimulationBridge`).
- **The substrate is the deployed one.** Binding/unbinding/bundling/cleanup are the production composer's spiking
  RF operations; the discourse buffer reuses them by import. No new mechanism, no `sim/` edit.

## Contrast (the headline)

| | rate-attractor discourse buffer (V1, `SpikingLoopContextBuffer`) | order-encoded discourse buffer (V2, spiking RF) |
|---|---|---|
| single-referent anaphora | GO | GO (strict superset) |
| multi-referent resolution | — (no notion of recency/slot) | **6/6** |
| ORDER-control FLIP | **0/6** (recency, salience-boost, WTA all fail) | **6/6** |
| why the winner is chosen | intrinsic basin asymmetry (uncontrollable) | **which slot you read** (deterministic) |
| no-confab moat | held by the agent's gate | clean familiarity separation, 6/6 (principled thr.) |

## Reproduce

```bash
# Demo transcript (multi-referent resolve + the order-control flip + the moat):
SIM_BACKEND=numpy python -m research.runners.multi_turn_ordered_wm_demo

# The production gate (4 controls × 6 seeds + V1-unbroken + code-parity):
SIM_BACKEND=numpy python -m pytest tests/test_multi_turn_ordered_wm.py -v
```

CPU (numpy backend). Deliverables: `research/runners/ordered_position_wm.py` (promoted module),
`research/runners/multi_turn_agent_v2.py` (the V2 agent), `tests/test_multi_turn_ordered_wm.py`,
`research/runners/multi_turn_ordered_wm_demo.py`. No `sim/` edit; no git commit (controller commits after
verifying).
