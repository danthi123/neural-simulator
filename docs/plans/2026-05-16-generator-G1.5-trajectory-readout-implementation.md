---
type: plan
status: live
date: 2026-05-16
---

# G1.5 — Order-Sensitive Trajectory Readout Probe — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Design: `docs/plans/2026-05-16-generative-G1-followup-branches-design.md`
> (Branch G1.5). This is the cheapest pre-staged FAIL-branch probe after
> the G1 NEGATIVE (`research/findings/2026-05-16-generator-G1-songbird-NEGATIVE.md`).

**Goal:** Test whether the order signal exists in the substrate's
*dynamics during the production* but was discarded by G1's
argmax-of-final-residual readout. Change the READOUT only (no
architecture rewrite): decode an ordered length-N concept trajectory
(the pool's expressed concept in the un-driven gap after each slot's
ignition) instead of one argmax of the final residual. Re-run the
SAME pre-registered gate (permuted-ORDER control, FIXED
`_G1_MARGIN=0.10`/`_G1_ABS_FLOOR=0.5`, regime-recalibrated
control-frozen floor). PASS ⇒ controller viable with a better judge
(→ G2 scale). FAIL ⇒ order signal genuinely absent from substrate
dynamics → G1.6/P justified by evidence.

**Architecture:** A `--readout {final,trajectory}` mode threaded
through the EXISTING `song_g1_ignite` / `song_g1_train` /
`song_g1_gate`. `trajectory` = a write-only ignite-slot → brief
un-driven gap → read-argmax → next-slot loop returning an ordered
decoded list of length = production length (so `score_order` can
reach 1.0 and reflect ORDER, vs G1's length-1 0.5-capped readout).
Isolated ckpt/sidecar namespace (`song_g1_traj.*`) so G1's and G1.5's
frozen floors/checkpoints never collide. Step-0 control-calibration
re-derived IN the trajectory regime (same AUC/control-max
methodology), frozen, never tuned.

**Anti-cheat (unchanged, non-negotiable):** pure `g1_verdict` /
`score_order` / `permuted_order_controls` UNMODIFIED. Bars never
touched. 650 never used. The trajectory-regime abstention floor is
pre-registered from a CONTROL distribution (same method that produced
G1's 72.0), frozen to the G1.5 sidecar, never recomputed at gate
time. The no-harm contract (write-only ignition, no regression of the
validated path / abstention moat) must be re-proven for the new
readout path before any G1.5 training.

**Reuse (DRY — do NOT rebuild):** `sim/song_hvc.py`,
`research/runners/song_g1_core.py` (verdict/score/permuted — bars
fixed), `song_g1_ignite.py` (write-only ignition + the existing
`_pattern_global_arrs` mapping), `song_g1_train.py` (kill-safe loop,
Step-0 calibration, smoke isolation, cross-mode refusal),
`song_g1_gate.py` (sidecar-frozen-floor gate), `song_g1_noharm_probe.py`
(the no-harm contract), `sim/train_checkpoint.py`, the validated G.20
320-sparse bridges. ASCII-only prints. Pure logic = CPU pytest;
integration validated by the re-run no-harm probe + the pre-registered
gate (project pattern).

---

### Task 1: `ignite_and_trajectory_decode` — write-only ordered trajectory readout

**Files:** Modify `research/runners/song_g1_ignite.py`; Test
`tests/test_song_g1_ignite_smoke.py` (extend import/signature smoke).

Add `ignite_and_trajectory_decode(member, concept_indices,
drive_pA=1500.0, steps_per=100, gap_steps=20, decode_steps=30) -> list`:
for each concept idx in order: (1) WRITE-ONLY drive that concept's
sparse pattern (reuse `_pattern_global_arrs`, exactly the existing
`ignite_sequence` inner drive) for `steps_per` steps; (2) zero
`cp_external_input_current`, free-run `gap_steps` (un-driven gap so
the read reflects the pool's RESPONSE/integration, NOT the driven
input — avoids the trivial "read back what you drove" circularity);
(3) over `decode_steps` (still un-driven) accumulate
`cp_firing_states[pattern_arr].sum()` per concept pattern (the SAME
validated stim-recall accumulation as `self_comprehend`), take argmax
→ that is `decoded[t]` (the concept the pool expresses after
integrating slots 0..t — order-dependent); record its accumulated
rate. Return `(decoded_list, per_slot_rates_list)` where
`len(decoded_list) == len(concept_indices)`. WRITE-ONLY invariant:
only `cp_external_input_current` writes at concept patterns + zeroing;
NO RegionPathway / commit_engram_tag / weight / plasticity mutation
(identical guarantee to `ignite_sequence`/`self_comprehend`). Lazy
heavy imports. ASCII-only.

**Step 1 (failing smoke):** extend `tests/test_song_g1_ignite_smoke.py`
to also assert `hasattr(ig, "ignite_and_trajectory_decode")`.
**Step 2:** run → FAIL. **Step 3:** implement as above. **Step 4:**
smoke PASS. **Step 5:** commit
`feat(song-g1.5): write-only ordered trajectory readout`.

---

### Task 2: No-harm re-proof for the trajectory readout path

**Files:** Modify `research/runners/song_g1_noharm_probe.py` (add a
`--readout trajectory` mode that, in PASS B, additionally constructs
nothing new but documents that the new readout fn is import-present
and write-only) OR add a thin assertion the new fn does not touch
bridge weights. Simplest sound form: re-run the EXISTING no-harm
probe unchanged (the new fn is additive + write-only by construction;
the silent-SongHVC contract is unaffected by an unused readout fn).

**Step:** Run `python -m research.runners.song_g1_noharm_probe`
ONCE. It MUST still PASS (>=8 cushioned validated-known, all KNOWN_OK
WITH silent SongHVC, abstention moat holds) — proving the additive
trajectory-readout code did not regress the validated path. If FAIL:
STOP, investigate, do not proceed (load-bearing, same rule as G1).
**Commit** the (re-run) `song_g1_noharm.json` finding.
> GATE: Task 2 PASS is REQUIRED before Task 3 training.

---

### Task 3: `--readout` mode threaded through trainer + gate (isolated namespace)

**Files:** Modify `research/runners/song_g1_train.py` and
`research/runners/song_g1_gate.py`; extend `tests/test_song_g1_gate.py`
(pure-logic only).

- Add `--readout {final,trajectory}` (default `final` — G1 behavior
  byte-unchanged when absent). When `trajectory`: the decode step uses
  `ignite_and_trajectory_decode` (ordered length-N decoded) instead of
  `_integrated_decode` (length-1); `score_order(decoded_traj,
  intended)` (unmodified) now scores true ORDER.
- **Namespace isolation:** when `--readout trajectory` and `--ckpt`
  is the default, redirect to `song_g1_traj.ckpt.npz` (+ sidecar) —
  reuse the EXACT smoke-isolation idiom (`_smoke_ckpt_path` pattern)
  so G1 (`song_g1.ckpt.npz`) and G1.5 frozen floors/weights never
  collide. Sidecar records `"readout":"trajectory"`; the gate REFUSES
  a sidecar whose `readout` != the run's `--readout` (same
  cross-mode-refusal pattern as the smoke flag) so a final-regime
  floor can never gate a trajectory run or vice-versa.
- **Step-0 re-calibration in the trajectory regime:** Step 0 measures
  encoded (intended-order) vs control (permuted-ORDER + random)
  trajectory-decode top-rate, sets `g1_abstain` = control-max (the
  SAME pre-registered operating criterion that produced G1's 72.0),
  frozen to the G1.5 sidecar, never retuned. Do NOT reuse G1's 72.0
  (different readout regime).
- Gate (`song_g1_gate.py --readout trajectory`): held-out only,
  trajectory decode, `gate_cleared = top-rate >= G1.5-sidecar-frozen
  g1_abstain`, `true_score = score_order(decoded_traj, intended)`,
  `best_perm` over permuted-ORDER controls (now meaningfully
  order-sensitive — a scrambled order yields a different trajectory),
  pure `g1_verdict` aggregate on means. Bars UNTOUCHED.

**Step 1:** pure tests for any new pure helper (e.g. readout-mode
dispatch / sidecar `readout` cross-mode-refusal `_check_sidecar_usable`
extension) in `tests/test_song_g1_gate.py`; run → cover the new
branch. **Step 2:** `--smoke --readout trajectory` (isolated path):
confirm it writes ONLY `song_g1_traj.smoke.*`, prints a finite
trajectory-regime `g1_abstain` (NOT 650, NOT G1's 72.0), kill-safe
resume works, a `--readout final` run does NOT reuse the trajectory
sidecar. **Step 3:** delete smoke artifacts. **Step 4:** commit
`feat(song-g1.5): --readout trajectory mode (isolated namespace, regime-recalibrated frozen floor)`.

---

### Task 4: Run + honest verdict + route (controller-driven)

1. Launch kill-safe `song_g1_train --readout trajectory --epochs 60
   --n-babble 8 --temperature0 0.5 --lr 0.5 --recover-steps 200
   --seed 42` (background, user can game/resume — Inc-3/G1 pattern).
2. On completion run `song_g1_gate --readout trajectory` for the
   pre-registered verdict.
3. Propagate honestly (findings doc + `capability_status` pillar,
   schema test green, push both remotes) whichever way it lands. Gate
   NOT tuned, controller NOT config-cranked.
4. **Route:** PASS ⇒ the controller works with an order-sensitive
   judge → G2 (multi-seed + held-out novel-compositional + cross-bridge).
   FAIL ⇒ order signal genuinely absent from substrate dynamics →
   execute pre-staged Branch G1.6 (developmental scaffolding), then P.

## Notes for the executor

- Anti-cheat: `g1_verdict`/`score_order`/`permuted_order_controls`
  bars (`_G1_MARGIN=0.10`, `_G1_ABS_FLOOR=0.5`) NEVER touched; the
  trajectory-regime floor is pre-registered control-max, frozen,
  never recomputed at gate time; 650 never used.
- The un-driven gap before each per-slot read is load-bearing: it
  prevents the trivial "read back the concept you're driving"
  circularity, so the trajectory reflects the pool's order-dependent
  integration. Keep `gap_steps` > 0.
- `--readout final` MUST remain byte-identical to G1 behavior
  (default; do not regress the recorded G1 negative's reproducibility).
- DRY: thread a flag; do NOT fork new train/gate files. Reuse the
  smoke-isolation + cross-mode-refusal machinery for the readout
  namespace.
- No-harm probe (Task 2) PASS is REQUIRED before Task 3 training
  (same load-bearing rule as G1).
