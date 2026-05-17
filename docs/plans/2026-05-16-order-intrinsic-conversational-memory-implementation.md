# Order-Intrinsic Conversational Memory — Decisive Cheap-First Slice — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Design: `docs/plans/2026-05-16-order-intrinsic-conversational-memory-design.md`.
> Cheap-first decisive slice (scoped like G1's B-probe / P's slice).
> Full conversational primitive set / scale / multi-turn = LATER
> increments, NOT here (YAGNI; this slice's pre-registered multi-seed
> gate decides whether the line is pursued).

**Goal:** Test the core thesis: a *trained* `ec_context(position)->concept-pool`
read-back pathway over the UNCHANGED multi-seed-validated D.11/P4.1
`(word,position)->distinct` store lets a 2-3 concept proposition be
read back **in correct order via a deterministic position sweep**
(NO learned sequence model), beating a permuted-ORDER control by the
pre-registered FIXED bar **multi-seed** — the precise thing the
order-blind line (G1/G1.5/P) could not do.

**Architecture:** Reuse the validated positional store + engram +
`song_g1_core` verdict (all UNMODIFIED). Net-new = a plastic
`ec_context->concept-pool` pathway strengthened by Hebbian/STDP
co-firing *during* the existing validated encode (Tonegawa engram
binds all co-active elements; catalog D.14/D.02). Produce = sweep
positions 0..L-1, drive `ec_context(k)` alone, decode the now-strongly
-bound concept. Order is intrinsic (D.11 code) and read by a
deterministic sweep, not learned.

**Tech Stack:** numpy (pure scoring/decode core, CPU-testable),
CuPy/RTX 3090 (substrate; production backend — the numpy-only
`bridge.py:5360` IndexError is out of scope, do not use numpy for the
episodic-context arch), pytest, `sim.train_checkpoint` (kill-safe).

**Anti-cheat (non-negotiable):** `song_g1_core.g1_verdict`/
`score_order`/`permuted_order_controls` REUSED UNMODIFIED
(`_G1_MARGIN=0.10`/`_G1_ABS_FLOOR=0.5` never touched). 650 never
used. The read-back-regime abstention floor is pre-registered
control-max-calibrated, frozen to an isolated sidecar, NEVER
recomputed at gate time. **Multi-seed (>=3) only** — single-seed /
near-noise is explicitly NOT a pass (the probe proved single-seed
unreliable: 50%). The LOAD-BEARING no-harm check
(`validate_positional_binding` must stay PASS with the trained
pathway present + abstention moat unregressed) MUST pass before the
gate is trusted. A maxed FAIL is an honest, terminal, propagated
finding (this line also ends honestly; the validated grounded-memory
+ no-confabulation asset remains the deliverable; do NOT
config-crank).

**Reuse (DRY — do NOT rebuild):**
- `sim/text_embeddings.py:208 positional_drive_pattern(position,
  n_neurons=200, drive_max_pA=200.0, sparsity=0.1,
  n_max_positions=16)` (D.11; deterministic)
- `research/runners/concept_pool_demo.py build_concept_bridge(...,
  enable_positional_context=True, n_ec_context=200,
  ec_context_to_pool_density=0.30, ec_context_to_pool_weight=3.0)`
  (the ec_context->concept-pool wiring the prototype uses)
- `research/runners/test_word_order_discrimination.py` —
  `query_position(bridge, position, all_pool_names, n_ec_context,
  n_max_positions, positional_sparsity, ec_drive_pA, stim_steps)`
  returns `{pool: rate}`; the encode loop. STUDY as the weak-50%
  baseline; the net-new piece is making the ec_context->pool pathway
  *plastic + co-firing-trained* during encode.
- `research/runners/validate_positional_binding.py` (re-run UNCHANGED
  as the no-harm store-distinctness check; multi-seed 3/3 PASS
  baseline; `2026-05-11-P41-positional-multiseed.md`)
- `research/runners/song_g1_core.py`: `score_order(decoded,intended)`,
  `permuted_order_controls(intended,rng,n)`,
  `g1_verdict(true_score,best_perm_score,gate_cleared)` (UNMODIFIED)
- `sim/bridge.py` engram API (`start_engram_recording:2485`,
  `commit_engram_tag:2514`, `stimulate_tag:2599`),
  `sim/train_checkpoint.py`, the abstention moat, the validated
  decoder readout
- Pattern refs: `sim/song_hvc.py`+`tests/test_song_hvc.py`,
  `research/runners/song_g1_core.py`+`tests/test_song_g1_core.py`
  (how pure cores were TDD'd)

**Conventions:** ASCII-only `print()` (Windows cp1252). Pure logic =
CPU pytest, bite-sized failing-test->impl->commit. Integration
validated by the no-harm check + the pre-registered multi-seed gate
(project pattern; import/signature smoke only, no contrived
orchestration unit tests). Build on `main`, purely additive, frequent
commits, trust-but-verify each subagent.

---

## Phase A — Pure read-back/scoring core (CPU TDD)

New module `research/runners/order_intrinsic_core.py`. Reuses
`song_g1_core` UNMODIFIED. Tests `tests/test_order_intrinsic_core.py`.

### Task 1: position-sweep decode (argmax per position + abstention)

**Step 1: failing test**
```python
import numpy as np
from research.runners.order_intrinsic_core import decode_position_sweep

def test_decode_position_sweep_argmax_and_abstain():
    # per-position pool-firing-rate dicts (query_position output shape)
    per_pos = [
        {"A": 0.50, "B": 0.10, "C": 0.05},   # pos0 -> A (clears floor)
        {"A": 0.08, "B": 0.40, "C": 0.06},   # pos1 -> B
        {"A": 0.02, "B": 0.03, "C": 0.02},   # pos2 -> below floor 0.10
    ]
    decoded, conf, abstained = decode_position_sweep(per_pos, floor=0.10)
    assert decoded == ["A", "B", None]        # pos2 abstains, no confab
    assert abstained == [2]
    assert conf[0] == 0.50 and conf[1] == 0.40
    # deterministic + tie-break stable (first max)
    d2, _, _ = decode_position_sweep(
        [{"A": 0.2, "B": 0.2}], floor=0.0)
    assert d2 == ["A"]
    # empty -> empty
    assert decode_position_sweep([], floor=0.1) == ([], [], [])
```
**Step 2:** `python -m pytest tests/test_order_intrinsic_core.py -q` -> FAIL (module missing)
**Step 3: implement** `decode_position_sweep(per_pos_rates: list[dict], floor: float) -> tuple[list, list, list]`: for each position dict, take the max-rate concept (stable first-max on tie); if its rate `<= floor` the slot is `None` (abstain — the no-confabulation moat per position) and the index is added to `abstained`; return `(decoded, conf, abstained)`. Pure numpy/stdlib, ASCII docstring, deterministic.
**Step 4:** PASS. **Step 5:** commit `feat(order-intrinsic): pure position-sweep decode + per-slot abstention`.

### Task 2: control-max frozen-floor calibration (pure)

**Step 1: failing test**
```python
from research.runners.order_intrinsic_core import control_max_floor

def test_control_max_floor_is_control_max_operating_point():
    enc = [0.50, 0.42, 0.61]          # encoded (intended) top-rates
    ctl = [0.20, 0.31, 0.18, 0.27]    # control (permuted/random) top-rates
    f = control_max_floor(enc, ctl)
    assert f == 0.31                  # the SAME operating criterion
                                      # that produced prior floors (control-max)
    assert control_max_floor([0.9], []) == 0.0   # no controls -> 0.0
```
**Step 2:** FAIL. **Step 3: implement** `control_max_floor(encoded_toprates, control_toprates) -> float`: return `float(max(control_toprates))` if any else `0.0` (the documented control-max operating point — the exact methodology that produced prior frozen floors; encoded arg accepted for signature parity / transparency logging, not used in the bar). Pure.
**Step 4:** PASS. **Step 5:** commit `feat(order-intrinsic): pure control-max frozen-floor calibration`.

### Task 3: pure pre-registered order-intrinsic verdict (reuses UNMODIFIED song_g1_core)

**Step 1: failing test**
```python
import numpy as np
from research.runners.order_intrinsic_core import order_intrinsic_verdict

def test_order_intrinsic_verdict_reuses_g1_bars():
    # true-order decoded correct; permuted-order decoded scrambled;
    # gate cleared -> PASS via UNMODIFIED g1_verdict (>=10% + >=0.5)
    v = order_intrinsic_verdict(
        true_decoded=[1, 2, 3], intended=[1, 2, 3],
        perm_decoded=[[2, 1, 3], [3, 2, 1]], gate_cleared=True)
    assert v["GATE"] == "PASS" and v["true_score"] == 1.0
    # gate not cleared -> FAIL regardless
    assert order_intrinsic_verdict([1,2,3],[1,2,3],[[2,1,3]],
                                   gate_cleared=False)["GATE"] == "FAIL"
    # true == permuted (no order learned) -> FAIL
    assert order_intrinsic_verdict([2,1,3],[1,2,3],[[2,1,3]],
                                   gate_cleared=True)["GATE"] == "FAIL"
```
**Step 2:** FAIL. **Step 3: implement** `order_intrinsic_verdict(true_decoded, intended, perm_decoded, gate_cleared) -> dict`: `from research.runners.song_g1_core import score_order, g1_verdict`; `true_score = score_order(true_decoded, intended)`; `best_perm = max((score_order(pd, intended) for pd in perm_decoded), default=0.0)`; `return g1_verdict(true_score, best_perm, gate_cleared)` augmented with `true_score`/`best_perm_score` keys for the JSON. Do NOT reimplement the bars — reuse `g1_verdict` UNMODIFIED.
**Step 4:** PASS. **Step 5:** commit `feat(order-intrinsic): pure pre-registered verdict (reuses UNMODIFIED g1_verdict bars)`.

### Task 4: pure aggregate-over-held-out-props + multi-seed verdict

**Step 1: failing test**
```python
from research.runners.order_intrinsic_core import aggregate_multiseed

def test_aggregate_multiseed_requires_all_seeds_pass():
    # per-seed list of per-prop verdict dicts (from order_intrinsic_verdict)
    seed_ok = [{"GATE":"PASS"},{"GATE":"PASS"}]
    seed_bad = [{"GATE":"PASS"},{"GATE":"FAIL"}]
    assert aggregate_multiseed([seed_ok, seed_ok, seed_ok])["GATE"] == "PASS"
    assert aggregate_multiseed([seed_ok, seed_bad, seed_ok])["GATE"] == "FAIL"
    assert aggregate_multiseed([seed_ok, seed_ok])["GATE"] == "FAIL"  # <3 seeds
```
**Step 2:** FAIL. **Step 3: implement** `aggregate_multiseed(per_seed_prop_verdicts, min_seeds=3) -> dict`: PASS iff `len(per_seed_prop_verdicts) >= min_seeds` AND every prop in every seed has `GATE == "PASS"`; report counts. Pure. (Pre-registered: multi-seed is mandatory — single-seed is NOT a pass.)
**Step 4:** PASS. **Step 5:** commit `feat(order-intrinsic): pure multi-seed aggregate verdict (>=3 seeds mandatory)`.

---

## Phase B — Integration (no-harm + pre-registered-gate validated)

### Task 5: trained `ec_context(position)->concept-pool` encode pathway

**Files:** Create `research/runners/order_intrinsic_encode.py`;
smoke `tests/test_order_intrinsic_encode_smoke.py` (import/signature
only — NO contrived orchestration unit test; real validation = Task 6
no-harm + Task 7 gate).

Study `test_word_order_discrimination.py` (the weak-50% baseline:
`build_concept_bridge(enable_positional_context=True ...)` then encode
word+position co-drive, `query_position` reads ec_context(pos) alone).
Net-new: ensure the `ec_context->concept-pool` pathway is **plastic
and STDP/Hebbian-strengthened during the co-drive encode** (it binds
position->concept — the Tonegawa engram over all co-active elements,
D.14/D.02), so `query_position` read-back is strong, not the raw-trace
50%. Expose:
- `build_order_intrinsic_bridge(seed, ...)` -> bridge with the
  validated positional store + the plastic ec_context->pool pathway
  (reuse `build_concept_bridge(enable_positional_context=True ...)`;
  ensure that pathway's plasticity gate is OPEN during encode).
- `encode_proposition(bridge, concept_words, ...)` -> for k,word: drive
  `lang_input(word)` + `ec_context(k)` (via `positional_drive_pattern`)
  for the encode window with the pool + ec_context->pool plasticity
  ON; commit one engram over the co-active set (D.14). DRY: reuse the
  prototype's drive idiom + the engram API; the ONLY new behavior is
  the pathway being plastic+trained during the co-drive.
- `readback_sweep(bridge, length, ...)` -> for k in range(length):
  `query_position`-style drive of `ec_context(k)` ALONE -> per-pos
  pool-rate dict. Returns `list[dict]` (feeds Phase-A
  `decode_position_sweep`). Write-only into the substrate (ext current
  only); reuse the prototype's `query_position` mechanism.

Steps: smoke (import + the 3 callables exist) FAIL -> implement ->
smoke PASS -> `python -m pytest tests/test_order_intrinsic_encode_smoke.py tests/test_order_intrinsic_core.py -q` all pass -> commit
`feat(order-intrinsic): trained ec_context->concept readback encode/sweep`.

### Task 6: LOAD-BEARING no-harm check (CRITICAL — before the gate is trusted)

**Files:** Create `research/runners/order_intrinsic_noharm.py`;
output `research/findings/raw/g11_bg/order_intrinsic_noharm.json`.

Re-run the UNCHANGED `validate_positional_binding` logic **with the
new trained ec_context->pool pathway present** (import + invoke its
validator, or run it against `build_order_intrinsic_bridge`). It MUST
stay PASS — the validated `(word,position)->distinct` store
distinctness (same-word/diff-pos cos < 0.4 AND diff-word/same-pos
cos < 0.4, multi-seed 3/3 baseline) must be UNREGRESSED, proving the
additive trained pathway did not damage the validated store. ALSO
probe the no-confabulation abstention moat is unregressed (a known
proposition clears the floor; an unknown abstains). Write JSON +
ASCII verdict; exit nonzero on FAIL.

**Run it ONCE.** PASS -> commit the finding, the gate (Task 7) is
authorized. FAIL -> STOP: the trained pathway regressed the validated
store/moat; fix its separation-of-concerns (open the pool pathway
plasticity ONLY during the order-intrinsic encode, never bleeding
into the validated DG/CA3 store path) before any gate run. Do NOT
weaken `validate_positional_binding`. Commit the JSON (PASS or FAIL)
as a recorded finding. Commit `feat(order-intrinsic): LOAD-BEARING no-harm check (store distinctness + moat unregressed)`.
> GATE: Task 6 PASS REQUIRED before Task 7 is trusted.

```
+---------------------------------------------------------------+
| PRE-GATE CORRECTION (CATEGORY ERROR) -- 2026-05-16             |
+---------------------------------------------------------------+
| Task-6 no-harm OVERALL =                                       |
|   (store-distinctness-unregressed: every seed max CA3 cos<0.4) |
|   AND (no-confabulation: every seed's never-encoded control    |
|        ABSTAINS).                                              |
| Encoded-vs-control MAGNITUDE separation is the pre-registered  |
| Task 7 capability gate (proper config), NOT a Task-6 no-harm   |
| criterion. The a78815b run recorded OVERALL=FAIL ONLY because  |
| it folded that capability-magnitude bar into moat_PASS at a    |
| deliberately-minimal speed config -- under-powered-pre-empting |
| the decisive Task 7 gate (same integrity-correction class as   |
| the documented C1/C2, Inc-3-held-out, P-bias, Task-5-retarget  |
| corrections). RECOMPUTED purely from the already-recorded      |
| a78815b run data (NO GPU re-run, no pass-chasing):             |
|   store-distinctness: 3/3 PASS (seed42 0.263, seed43 0.177,    |
|                        seed44 0.166; all < 0.4)                |
|   no-confabulation (control abstains): 3/3 PASS                |
|   -> Task-6 no-harm SATISFIED.                                 |
| The capability question is NOT removed/weakened -- it IS the   |
| pre-registered Task 7 gate, run honestly at proper config      |
| regardless. order_intrinsic_noharm.py OVERALL logic is fixed   |
| for FUTURE runs; the JSON was rewritten by a pure recompute    |
| from the recorded per-seed numbers (no GPU job).               |
+---------------------------------------------------------------+
```

### Task 7: pre-registered MULTI-SEED gate + honest propagation

**Files:** Create `research/runners/order_intrinsic_gate.py`; out
`research/findings/raw/g11_bg/order_intrinsic_gate.json`; findings
`research/findings/2026-05-16-order-intrinsic-<PASS|NEGATIVE>.md`;
modify `webapp/capability_status.json`.

For each of >=3 seeds: build bridge; FROZEN train propositions vs a
DISJOINT held-out set (held-out NEVER encoded-for-training-bias —
note: this slice's "training" is the per-proposition co-drive encode;
held-out props are encoded then immediately read back to test the
*pathway*, never used to tune anything). Step-0: control-calibrated
frozen abstention floor = `control_max_floor` over permuted-ORDER +
random-order read-back top-rates (Task 2; frozen to an isolated
sidecar `order_intrinsic.<seed>.json`; NEVER 650; never recomputed at
gate time). Per held-out prop: `encode_proposition` -> `readback_sweep`
-> `decode_position_sweep` (Task 1, vs frozen floor) -> true decoded;
build `permuted_order_controls` (UNMODIFIED song_g1_core), read each
back the same way -> perm decoded; `order_intrinsic_verdict` (Task 3).
Aggregate via `aggregate_multiseed` (Task 4; >=3 seeds, every prop
every seed must PASS). Kill-safe if multi-seed run is long (reuse
`sim.train_checkpoint`; long runs via `run_in_background`,
user games/resumes). ASCII-only.

Honest propagation (EITHER outcome): findings doc (state the
pre-registered gate, the numbers, PASS or honest terminal NEGATIVE;
if FAIL: this order-intrinsic line also terminates honestly, the
validated grounded-memory + no-confabulation agent is the deliverable,
do NOT config-crank); `capability_status.json` pillar
(VALIDATED only if multi-seed PASS, else NEGATIVE/TERMINAL honestly);
`python -m pytest tests/test_webapp_server.py -k capability_status -q`
green; commit + push BOTH remotes. Gate NOT tuned; floor
sidecar-frozen never recomputed; bars never touched.

**Route:** PASS (multi-seed) => order-intrinsic structured-proposition
conversation is real -> next increment: scale + the full
conversational primitive set over it (tell/ask/compose/answer/abstain/
ordered-readback, multi-turn) — separate later plan. FAIL => honest
terminal finding; the validated grounded-memory asset stands.

## Future increments (NOT here — this slice's gate decides)

Only if the slice PASSes multi-seed: scale concepts/positions;
compose/intersect over (concept@position) memory; multi-turn
structured-proposition conversation with the abstention moat + CLS
no-forgetting. Separate design+plan after the verdict.

## Notes for the executor

- Anti-cheat: `g1_verdict`/`score_order`/`permuted_order_controls`
  bars NEVER touched; floor pre-registered control-max, frozen, never
  recomputed; 650 never used; **multi-seed (>=3) mandatory — a
  single-seed or near-noise result is NOT a pass** (the probe showed
  single-seed 50% is unreliable); gate never tuned; full pre-registered
  protocol; honest negative propagated, no config-cranking.
- **Task 6 no-harm PASS is REQUIRED before trusting Task 7** — the
  trained pathway must not regress the validated store distinctness or
  the moat (the v12/v13/v15/G1 "first do no harm" lesson; CRITICAL
  here because the pathway writes into concept pools).
- DRY: reuse `song_g1_core` (UNMODIFIED), the validated positional
  store, the prototype's drive/`query_position` idiom,
  `validate_positional_binding`, `sim.train_checkpoint`. The ONLY
  net-new mechanism is the *plastic, co-firing-trained*
  ec_context->pool pathway + the deterministic position-sweep
  producer + the pure core. No learned sequence model anywhere.
- Production backend is CuPy/RTX 3090. The numpy-only
  `bridge.py:5360` IndexError is OUT OF SCOPE (do not fix; do not use
  numpy for the episodic-context arch).
- Honest ceiling: structured-proposition conversation
  (tell/ask/compose/answer/abstain/ordered-readback), NOT free-form
  LLM generation. Never overclaim. A maxed FAIL is a real terminal
  finding for this line; the validated asset remains the deliverable.
