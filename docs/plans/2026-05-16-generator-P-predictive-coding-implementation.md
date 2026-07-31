---
type: plan
status: live
date: 2026-05-16
---

# Generator P (decisive cheap-first slice) — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> (fresh subagent per task + spec/quality review). Design:
> `docs/plans/2026-05-16-generator-P-predictive-coding-topdown-design.md`.
> This is the cheap-first decisive slice (same falsify-cheaply scoping
> that scoped G1 to the B-probe). Scale/cross-bridge/multi-turn are
> LATER increments — NOT here (YAGNI; this slice's pre-registered gate
> decides whether P is pursued).

**Goal:** Test the P thesis: a net-new top-down predictive-coding
layer that *predicts the next concept given the sequence-so-far*
(prediction error = the order-sensitive learning signal the
recognition-only substrate provably does NOT provide) lets a
self-contained active-inference rollout emit order-correct
propositions that pass the SAME pre-registered held-out
permuted-ORDER gate G1/G1.5 failed.

**Architecture:** Pure-numpy `PredictiveCoder` (`sim/predictive_coding.py`)
— recurrent `pc_state` of the prefix, learned top-down predictor
`pc_state -> next-concept logits`, Rao-Ballard prediction error,
self-supervised P-weight update (reuses the stabilized
`sim.bptt_snn` CE/softmax), active-inference next-concept selection.
Integration: P's ONLY write into concept pools is the (P-T) top-down
prediction current via the EXISTING write-only `song_g1_ignite`
surface; a `--mode p` is threaded through the proven
`song_g1_train`/`song_g1_gate` isolation+cross-mode-refusal+
sidecar-frozen-floor machinery. Validated G.20 substrate + `song_hvc`
+ `song_g1_core` (bars) are UNMODIFIED — P is purely additive.

**Tech Stack:** numpy (pure PC core, CPU-testable), the validated
G.20 320-sparse bridges, `sim.bptt_snn` (stabilized CE/softmax),
`sim.train_checkpoint` (kill-safe), pytest.

**Anti-cheat (non-negotiable):** `song_g1_core.g1_verdict`/
`score_order`/`permuted_order_controls` REUSED UNMODIFIED
(`_G1_MARGIN=0.10`/`_G1_ABS_FLOOR=0.5` never touched). 650 never
used. P-regime abstention floor is pre-registered control-max/AUC
calibrated, frozen to an isolated `song_g1.pc.*` sidecar, NEVER
recomputed at gate time. The LOAD-BEARING no-harm probe re-proof is
now CRITICAL (P adds a pathway into concept pools) and MUST PASS
before any P training. A maxed P FAIL is an honest, terminal,
decision-relevant finding — propagated, never tuned away; the
validated grounded-memory + no-confabulation asset is then the
deliverable.

**Reuse (DRY — do NOT rebuild):**
- `sim/bptt_snn.py`: `cross_entropy_loss_np(logits,target)->float`,
  `softmax_grad_np(logits,target)->ndarray` (log-sum-exp stabilized
  since the Inc-3 fix — use these, do NOT hand-roll softmax)
- `sim/train_checkpoint.py`: `save_checkpoint(path,epoch,weights,
  rng_state,loss_history)`, `load_checkpoint(path)->dict|None`,
  `resume_epoch(ckpt)->int`
- `research/runners/song_g1_ignite.py`: `load_members(seed=42)`,
  `ignite_sequence` (write-only drive idiom), `self_comprehend`,
  `_pattern_global_arrs`
- `research/runners/song_g1_core.py`: `g1_verdict`, `score_order`,
  `permuted_order_controls` (UNMODIFIED)
- `research/runners/song_g1_noharm_probe.py` (re-run as the P
  training gate; cushioned validated-known, abstention moat)
- `research/runners/song_g1_train.py` / `song_g1_gate.py`: the
  `--readout`/smoke namespace-isolation + `cross_mode_mismatch`
  refusal + sidecar-frozen-floor + Step-0 control-calibration
  machinery (ADD a P mode, same pattern; do NOT fork)
- `research/runners/g20_vocab_spec_320.py`, the 320-sparse bridges,
  `--enable-pfc-nmda` region framework
- Pattern references: `sim/song_hvc.py` + `tests/test_song_hvc.py`
  (how a pure controller was TDD'd), `research/runners/song_g1_core.py`
  + `tests/test_song_g1_core.py`

**Conventions:** ASCII-only `print()` (Windows cp1252). Pure logic =
CPU pytest, bite-sized failing-test→impl→commit. Integration
validated by the no-harm re-proof + the pre-registered gate (project
pattern; no contrived orchestration unit tests). Build on `main`,
purely additive, frequent commits, trust-but-verify each subagent.

---

## Phase A — Pure predictive-coding core (CPU TDD)

### Task 1: PredictiveCoder state init + deterministic prefix update

**Files:** Create `sim/predictive_coding.py`; Test
`tests/test_predictive_coding.py`

**Step 1: Write the failing test**

```python
import numpy as np
from sim.predictive_coding import PredictiveCoder

def test_pc_state_resets_and_updates_deterministically():
    pc = PredictiveCoder(n_concepts=8, state_dim=16, seed=42)
    pc.reset(intention=[3, 5])
    assert pc.state.shape == (16,)
    assert np.allclose(pc.state, 0.0)            # reset -> zero prefix
    pc.update_state(3)
    s1 = pc.state.copy()
    assert not np.allclose(s1, 0.0)               # prefix changed state
    pc.update_state(5)
    s2 = pc.state.copy()
    assert not np.allclose(s2, s1)                # order-dependent
    # deterministic given seed + same concept stream
    pc2 = PredictiveCoder(n_concepts=8, state_dim=16, seed=42)
    pc2.reset(intention=[3, 5]); pc2.update_state(3); pc2.update_state(5)
    assert np.allclose(pc2.state, s2)
    # ORDER matters: 3->5 != 5->3
    pc3 = PredictiveCoder(n_concepts=8, state_dim=16, seed=42)
    pc3.reset(intention=[3, 5]); pc3.update_state(5); pc3.update_state(3)
    assert not np.allclose(pc3.state, s2)
```

**Step 2:** `python -m pytest tests/test_predictive_coding.py -q` →
FAIL (`ModuleNotFoundError: No module named 'sim.predictive_coding'`)

**Step 3: Minimal implementation**

```python
"""predictive_coding: net-new top-down predictive-coding core.

Pure, deterministic, backend-agnostic (numpy). A recurrent pc_state
encodes the sequence-so-far (prefix); a learned top-down predictor
maps pc_state -> next-concept logits; the Rao-Ballard prediction
error (realized - predicted) is the order-sensitive learning signal
the recognition-only G.20 substrate provably does NOT provide
(Rao & Ballard 1999; Friston active inference; Bastos 2012).

This module ONLY computes predictions/errors/updates on its own
small weights -- it never touches a bridge and never feeds
non-specific activity into concept pools (the v12/v13/v15/G1
"first, do no harm" lesson). The substrate stays UNCHANGED.
"""
from __future__ import annotations
import numpy as np


class PredictiveCoder:
    def __init__(self, n_concepts: int, state_dim: int = 64,
                 seed: int = 42, leak: float = 0.9):
        self.n_concepts = int(n_concepts)
        self.state_dim = int(state_dim)
        self.seed = int(seed)
        self.leak = float(leak)
        rng = np.random.default_rng(seed)
        # W_in: concept one-hot -> state increment; W_pred: state ->
        # next-concept logits. Small init (the predictor is the only
        # learned machinery; substrate untouched).
        self.W_in = rng.normal(
            0.0, 0.1, (n_concepts, state_dim)).astype(np.float32)
        self.W_pred = rng.normal(
            0.0, 0.1, (state_dim, n_concepts)).astype(np.float32)
        # learnable output bias: makes the start-of-sequence ([] prefix,
        # zero pc_state) transition learnable (its CE gradient is `err`,
        # always nonzero -- textbook softmax/LM output-layer bias).
        self.b_pred = np.zeros(n_concepts, dtype=np.float32)
        self.state = np.zeros(state_dim, dtype=np.float32)
        self._intention: list = []

    def reset(self, intention: list) -> None:
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        self._intention = [int(c) for c in intention]

    def update_state(self, realized_concept_idx: int) -> None:
        c = int(realized_concept_idx)
        if not (0 <= c < self.n_concepts):
            raise IndexError(
                "concept idx %d out of [0,%d)" % (c, self.n_concepts))
        # leaky recurrent prefix accumulation (order-dependent)
        self.state = (self.leak * self.state
                      + self.W_in[c]).astype(np.float32)
```

**Step 4:** `python -m pytest tests/test_predictive_coding.py -q` →
PASS

**Step 5: Commit**

```bash
git add sim/predictive_coding.py tests/test_predictive_coding.py
git commit -m "feat(generator-P): PredictiveCoder pure recurrent prefix state"
```

---

### Task 2: Top-down next-concept prediction (forward)

**Files:** Modify `sim/predictive_coding.py`; Test
`tests/test_predictive_coding.py`

**Step 1: Failing test**

```python
def test_predict_next_logits_shape_and_determinism():
    pc = PredictiveCoder(n_concepts=8, state_dim=16, seed=1)
    pc.reset(intention=[2, 4]); pc.update_state(2)
    logits = pc.predict_next()
    assert logits.shape == (8,)
    assert np.all(np.isfinite(logits))
    assert np.allclose(pc.predict_next(), logits)   # pure/deterministic
    # different prefix -> different prediction
    pc.update_state(4)
    assert not np.allclose(pc.predict_next(), logits)
```

**Step 2:** run → FAIL (`predict_next` missing). **Step 3:** add

```python
    def predict_next(self) -> np.ndarray:
        """Top-down generative prediction: pc_state -> next-concept
        logits, with a learnable output bias so the start-of-sequence
        ([] prefix, zero state) transition is learnable. Pure/det."""
        return (self.state @ self.W_pred + self.b_pred).astype(np.float32)
```

**Step 4:** run → PASS. **Step 5:** commit
`feat(generator-P): top-down next-concept prediction forward`.

---

### Task 3: Rao-Ballard prediction error (reuse stabilized softmax)

**Files:** Modify `sim/predictive_coding.py`; Test
`tests/test_predictive_coding.py`

**Step 1: Failing test**

```python
def test_prediction_error_is_softmax_minus_onehot():
    from sim.bptt_snn import softmax_grad_np
    pc = PredictiveCoder(n_concepts=6, state_dim=12, seed=2)
    pc.reset(intention=[1, 3]); pc.update_state(1)
    logits = pc.predict_next()
    err = pc.prediction_error(realized_next_idx=3)
    assert err.shape == (6,)
    assert np.all(np.isfinite(err))
    # Rao-Ballard residual == stabilized softmax CE gradient (DRY:
    # reuses sim.bptt_snn.softmax_grad_np, the log-sum-exp-stable one)
    expected = softmax_grad_np(logits.reshape(1, -1), 3)[0]
    assert np.allclose(err, expected, atol=1e-6)
```

**Step 2:** FAIL. **Step 3:** add (DRY — reuse the Inc-3-stabilized
softmax/CE; do NOT hand-roll exp):

```python
    def prediction_error(self, realized_next_idx: int) -> np.ndarray:
        """Rao-Ballard residual = softmax(predicted) - onehot(realized)
        = the stabilized CE gradient w.r.t. logits. Reuses
        sim.bptt_snn.softmax_grad_np (log-sum-exp stable since the
        Inc-3 fix). Order-sensitive: depends on pc_state (the prefix)."""
        from sim.bptt_snn import softmax_grad_np
        logits = self.predict_next().reshape(1, -1)
        return softmax_grad_np(logits, int(realized_next_idx))[0]
```

**Step 4:** PASS. **Step 5:** commit
`feat(generator-P): Rao-Ballard prediction error (DRY stabilized softmax)`.

---

### Task 4: Self-supervised P-weight update (predictor only; substrate untouched)

**Files:** Modify `sim/predictive_coding.py`; Test
`tests/test_predictive_coding.py`

**Step 1: Failing test**

```python
def test_learn_reduces_ce_on_a_fixed_prefix_target():
    from sim.bptt_snn import cross_entropy_loss_np
    pc = PredictiveCoder(n_concepts=6, state_dim=12, seed=3)
    prefix, target = [1, 4], 2
    def ce():
        pc.reset(intention=prefix + [target])
        for c in prefix: pc.update_state(c)
        return cross_entropy_loss_np(
            pc.predict_next().reshape(1, -1), target)
    before = ce()
    for _ in range(200):
        pc.learn(prefix=prefix, target_next_idx=target, lr=0.05)
    after = ce()
    assert after < before * 0.5      # self-supervised CE drops
    # learning is confined to P weights; shapes unchanged
    assert pc.W_pred.shape == (12, 6) and pc.W_in.shape == (6, 12)
```

**Step 2:** FAIL. **Step 3:** add `learn` — rebuild pc_state from
prefix, predict, error (Task 3), gradient-descend `W_pred` (and
`W_in` via the linear chain rule) on that single (prefix→target)
example. Pure numpy; small P layer only; NEVER touches any bridge /
substrate array. (Reference `sim/song_hvc.py:reinforce` for the
in-module pure-update pattern.)

```python
    def learn(self, prefix: list, target_next_idx: int,
              lr: float) -> None:
        self.reset(self._intention or (list(prefix) + [target_next_idx]))
        # recompute prefix state, tracking the concepts for W_in grad
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        contribs = []
        for c in prefix:
            self.state = (self.leak * self.state
                          + self.W_in[int(c)]).astype(np.float32)
            contribs.append(int(c))
        err = self.prediction_error(int(target_next_idx))   # (n_concepts,)
        # dL/dW_pred = outer(state, err); dL/dstate = W_pred @ err
        gW_pred = np.outer(self.state, err).astype(np.float32)
        dstate = (self.W_pred @ err).astype(np.float32)
        self.W_pred -= lr * gW_pred
        # output-bias grad (softmax-CE): dL/db_pred == err. Always
        # nonzero, so the []->first-concept transition is learnable.
        self.b_pred -= lr * err
        # W_in grad: each prefix concept contributed leak**k * W_in[c]
        # to state; apply the same dstate to the concepts' rows (a
        # 1-step approximation -- sufficient for the cheap P probe).
        for c in set(contribs):
            self.W_in[c] -= lr * dstate
        np.clip(self.W_pred, -5.0, 5.0, out=self.W_pred)
        np.clip(self.b_pred, -5.0, 5.0, out=self.b_pred)
        np.clip(self.W_in, -5.0, 5.0, out=self.W_in)
```

**Step 4:** PASS. **Step 5:** commit
`feat(generator-P): self-supervised predictor-only weight update`.

---

### Task 5: Active-inference next-concept selection

**Files:** Modify `sim/predictive_coding.py`; Test
`tests/test_predictive_coding.py`

**Step 1: Failing test**

```python
def test_select_next_picks_the_learned_continuation():
    pc = PredictiveCoder(n_concepts=6, state_dim=12, seed=4)
    prefix, target = [0, 5], 3
    for _ in range(300):
        pc.learn(prefix=prefix, target_next_idx=target, lr=0.05)
    pc.reset(intention=prefix + [target])
    for c in prefix: pc.update_state(c)
    # active inference: emit the concept the generative model most
    # predicts given the prefix (argmax predicted prob)
    assert pc.select_next(candidates=list(range(6))) == target
    # restricting candidates still returns the best AVAILABLE one
    alt = pc.select_next(candidates=[1, 3, 4])
    assert alt == 3
```

**Step 2:** FAIL. **Step 3:** add

```python
    def select_next(self, candidates: list) -> int:
        """Active inference: emit the candidate concept the top-down
        generative model most predicts given the current prefix
        (argmax predicted logit over candidates). Pure."""
        logits = self.predict_next()
        cand = [int(c) for c in candidates
                if 0 <= int(c) < self.n_concepts]
        if not cand:
            raise ValueError("no valid candidates")
        return max(cand, key=lambda c: float(logits[c]))
```

**Step 4:** PASS. **Step 5:** commit
`feat(generator-P): active-inference next-concept selection`.

---

### Task 6: Pure full-rollout helper (prefix -> ordered production)

**Files:** Modify `sim/predictive_coding.py`; Test
`tests/test_predictive_coding.py`

**Step 1: Failing test**

```python
def test_rollout_reproduces_a_learned_two_concept_proposition():
    pc = PredictiveCoder(n_concepts=8, state_dim=24, seed=5)
    intended = [2, 6]                       # ordered proposition
    # self-supervised on each prefix->next of the intended order
    for _ in range(400):
        pc.learn(prefix=[], target_next_idx=2, lr=0.05)
        pc.learn(prefix=[2], target_next_idx=6, lr=0.05)
    produced = pc.rollout(intention=intended, length=2,
                          candidates=list(range(8)))
    assert produced == intended            # order-correct generation
```

**Step 2:** FAIL. **Step 3:** add `rollout` — reset(intention),
then for t in range(length): `c = select_next(candidates)`;
`update_state(c)`; append c. Returns the ordered produced list.
Pure (no bridge).

**Step 4:** PASS. **Step 5:** commit
`feat(generator-P): pure active-inference rollout`.

---

## Phase B — Integration (no-harm-gated + pre-registered-gate-validated)

### Task 7: P-mode threaded through ignition + no-harm RE-PROOF (LOAD-BEARING, CRITICAL)

**Files:** Modify `research/runners/song_g1_ignite.py` (add a P
top-down-prediction-current ignition variant — write-only, the ONLY
P write into concept pools); re-run `song_g1_noharm_probe.py`.

The P top-down prediction is delivered into concept pools by driving
the *predicted* concept's sparse pattern via the EXISTING write-only
`_pattern_global_arrs` / `cp_external_input_current` idiom (identical
allowed-write surface as `ignite_sequence`; NO RegionPathway, NO
weight/tag mutation, NO non-specific feedback — separation of
concerns). Add `ignite_prediction(member, concept_idx, drive_pA,
steps_per)` (a thin alias of the existing per-concept write-only
drive; reuse, do not duplicate logic).

**Step 1:** smoke (import/signature) that `ignite_prediction` exists
and is write-only by construction (mirrors `ignite_sequence`).
**Step 2:** RUN `python -m research.runners.song_g1_noharm_probe`
ONCE with the P-ignition code present (additive, P untrained/silent).
It MUST PASS (≥8 cushioned validated-known all KNOWN_OK, abstention
moat holds, band excess ≤0). **This is CRITICAL: P now adds a write
path into concept pools — if W→A binding or the abstention moat
regresses, STOP, P's separation-of-concerns is wrong, fix it before
ANY P training.** **Step 5:** commit the runner change + the re-run
`song_g1_noharm.json` finding.
> GATE: Task 7 no-harm PASS is REQUIRED before Task 9 training.

---

### Task 8: `--mode p` threaded through trainer + gate (isolated namespace, regime-recalibrated frozen floor)

**Files:** Modify `research/runners/song_g1_train.py`,
`research/runners/song_g1_gate.py`; extend `tests/test_song_g1_gate.py`
(pure-logic only).

Add `--mode {songbird,p}` (default `songbird` = G1 byte-identical).
When `p`: the per-candidate decode is the P active-inference rollout
(`PredictiveCoder.rollout` driving `ignite_prediction` per slot, then
the EXISTING `self_comprehend` integrated readout for the
gate-cleared rate); training (Task 9) self-supervises the
`PredictiveCoder` (Task 4) on the frozen TRAIN propositions. Reuse —
do NOT fork — the proven machinery:
- namespace isolation: default `--ckpt` → `song_g1.pc.ckpt.npz`
  (compose with smoke → `.pc.smoke.`), same `_smoke_ckpt_path`/
  `_traj_ckpt_path` idiom;
- sidecar records `"mode":"p"`; `_check_sidecar_usable` REFUSES a
  sidecar whose `mode` != run's `--mode` (same hard-refusal class as
  smoke/readout cross-mode);
- Step-0 control-calibration re-derived IN the P regime (same
  control-max/AUC methodology that produced G1's 72.0; NEVER 650;
  NEVER G1/traj values; frozen; never recomputed at gate time);
- gate REUSES `song_g1_core.g1_verdict`/`score_order`/
  `permuted_order_controls` UNMODIFIED; FIXED bars untouched.

**Step 1:** pure tests in `tests/test_song_g1_gate.py` for the new
`_check_sidecar_usable` `mode` cross-refusal + any pure P-mode
dispatch helper; run → cover new branch. **Step 2:** `--smoke
--mode p` (isolated path) — confirm writes ONLY `song_g1.pc.smoke.*`,
finite P-regime `g1_abstain` (NOT 650/72.0/46.0), kill-safe resume,
a `--mode songbird` run does NOT reuse the P sidecar. **Step 3:**
delete smoke artifacts. **Step 4:** commit
`feat(generator-P): --mode p (isolated namespace, regime-recalibrated frozen floor, cross-mode refusal)`.

---

### Task 9: Kill-safe self-supervised P trainer

**Files:** the `--mode p` training path in
`research/runners/song_g1_train.py` (reuse the kill-safe
checkpoint/sidecar/resume loop).

Per epoch, per frozen TRAIN proposition: self-supervise the
`PredictiveCoder` on each prefix→next of the intended order
(`PredictiveCoder.learn`); periodically eval an active-inference
rollout → `self_comprehend` integrated rate → `gate_cleared` vs the
Step-0 P-regime frozen floor → `compose_reward` (unmodified
`song_g1_core`) for the per-epoch metric. Per-epoch
`save_checkpoint` (PredictiveCoder W_in/W_pred + rng + loss_history);
auto-resume; `KeyboardInterrupt`→checkpoint+exit; held-out
propositions NEVER trained. ASCII-only prints. Kill-safe long run via
`run_in_background` (user games/resumes — Inc-3/G1 pattern).

**Step:** `--smoke --mode p` 2-epoch resume smoke (build/kill-safe
validation only, NOT a result). Commit
`feat(generator-P): kill-safe self-supervised predictive-coding trainer`.

---

### Task 10: Run + pre-registered gate + honest propagation (terminal-decision-relevant)

1. (After Task 7 no-harm PASS) launch kill-safe `song_g1_train
   --mode p --epochs 60 ...` (background; user games/resumes).
2. On completion run `song_g1_gate --mode p` — held-out novel
   propositions, P active-inference rollout, permuted-ORDER control,
   P-regime sidecar-frozen floor, UNMODIFIED `g1_verdict` (FIXED
   ≥10% margin + 0.5 floor). The verdict is whatever it is.
3. Propagate honestly: findings doc
   `research/findings/2026-05-16-generator-P-<PASS|NEGATIVE>.md` +
   `webapp/capability_status.json` pillar +
   `pytest tests/test_webapp_server.py -k capability_status -q`
   green + push BOTH remotes. Gate NOT tuned; controller/predictor
   NOT config-cranked; full protocol run.
4. **Route:** PASS ⇒ predictive-coding yields self-contained
   order-correct generation → next increment G2 (multi-seed +
   held-out novel-compositional + cross-bridge + grammaticality +
   multi-turn). FAIL ⇒ the honest, terminal, decision-relevant
   conclusion: self-contained local generative *production* is out of
   reach on this substrate/hardware under no-cheating/local
   constraints — propagated with no spin; the validated grounded
   continual memory + no-confabulation abstention stands as the
   robust deliverable (untouched, no-harm-re-proven throughout).

## Future increments (NOT in this plan — YAGNI; this slice's gate decides)

- **G2 (only if P PASSes):** multi-seed P; held-out *novel
  compositional* propositions never trained; cross-bridge
  propositions; +grammaticality; multi-turn generated conversation
  with the abstention moat intact + CLS no-forgetting.

## Notes for the executor

- Anti-cheat: `g1_verdict`/`score_order`/`permuted_order_controls`
  bars NEVER touched; P-regime floor pre-registered control-max,
  frozen, never recomputed; 650 never used; gate never tuned;
  predictor never config-cranked; full pre-registered protocol run.
- **Task 7 no-harm PASS is REQUIRED before Task 9 training** — P now
  writes top-down into concept pools, so "first do no harm" is
  CRITICAL, not incidental. A regression there = P separation-of-
  concerns bug; STOP and fix before training.
- DRY: reuse `sim.bptt_snn` stabilized CE/softmax (do NOT hand-roll
  exp — the Inc-3 overflow bug), `sim.train_checkpoint`,
  `song_g1_ignite` write-only surface, `song_g1_core` (unmodified),
  the song_g1_train/gate isolation+cross-mode-refusal+
  sidecar-frozen-floor machinery (extend, do NOT fork).
- YAGNI: pure PC core + active-inference rollout + the one
  pre-registered gate. No scale/cross-bridge/multi-turn here.
- `--mode songbird` MUST remain byte-identical (G1/G1.5 stay
  reproducible). Pure logic (Tasks 1-6) = CPU pytest; integration
  (7-10) validated by the no-harm re-proof + the pre-registered gate.
- A maxed P FAIL is a real, terminal finding — propagate honestly,
  do NOT iterate the predictor to chase the gate (the config-cranking
  the project forbids).
- Pre-data correction (2026-05-16, TDD-caught latent spec bug -- NOT
  test/spec-hacking): Task 6 TDD revealed the original PredictiveCoder
  spec had no learnable parameter affecting predict_next at zero state,
  so the start-of-sequence ([] prefix) transition was structurally
  unlearnable (learn(prefix=[]) was a no-op: outer(0,err)=0, no prefix
  concepts). Root-cause fix applied BEFORE any P result: a learnable
  output bias b_pred (predict_next += b_pred; learn: b_pred -= lr*err;
  bias CE-gradient is err, always nonzero). Textbook (every softmax/LM
  output has a bias). Preserves Tasks 1-5 (b_pred zero-init; non-empty-
  prefix only gains capacity); makes Task 6 pass legitimately (the
  test/assertion was NOT altered -- the prior subagent correctly
  refused to fake-pass). Analogous to the G1 C1/C2 pre-data integrity
  corrections; no anti-cheat bar is involved (pure core, no gate yet).
