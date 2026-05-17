# Dendritic Credit Assignment — Implementation Plan (decisive cheap-first slice of Arch A)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute task-by-task (fresh subagent per task; two-stage spec+quality review; controller trust-but-verify each git diff with protected modules byte-empty). Tasks 1+2 additionally get a dedicated adversarial reviewer before Phase B (mirror how Generator-S/D/G/H load-bearing cores got the rigorous adversarial review that caught real holes). Task 6 is the CONTROLLER's job, NOT a subagent.

**Goal:** Build the decisive cheap slice that answers — *does biologically-local dendritic credit assignment (segregated 2-compartment spiking neuron + FIXED-RANDOM apical feedback + LOCAL Urbanczik-Senn plasticity, NO weight transport, NO autograd) genuinely approximate the gradient and solve a hidden-credit task in THIS project's spiking substrate?* — gated by a pre-registered FIXED-bar multi-seed verdict.

**Architecture:** 3 net-new pure-numpy modules + 1 thin gate runner. The decisive metric is (a) cosine alignment of the local rule's weight change vs the `sim/bptt_snn` BPTT oracle gradient on a tiny spiking net, WITHOUT weight transport, AND (b) a spiking hidden-credit task the dendritic learner solves but a no-hidden-credit floor learner provably cannot, AND (c) a permuted-label control that does NOT clear. Everything load-bearing is reused byte-UNMODIFIED. The full `bio_three_factor` W→A confirmation is an explicit LATER increment (YAGNI for the decisive slice).

**Tech Stack:** Python 3, numpy, stdlib (pure, CPU). `sim/bptt_snn` reused ONLY as the gradient correctness ORACLE in tests + the gate runner's offline measurement — NEVER imported by any shipped `sim/dendritic_*.py` learning module (self-contained + biologically-local by construction; no `torch.autograd`/`backward` in any shipped module). pytest. ASCII-only prints. @superpowers:test-driven-development every task.

---

## Context the implementer MUST know (zero-context briefing)

- **Why this exists:** every project negative traced to one root — local biological rules (STDP/three-factor/Hebbian) on POINT neurons cannot do hidden/temporal credit assignment; gradient can (2026-05-05 W→A verdict: classical-DA 1/6, graded-DA 0/6, gradient 3/3 under identical arch). A 2026-05-17 falsify-cheaply rate/numpy probe gave a CLEAN PROBE POSITIVE (bars byte-fixed; control STRENGTHENED to the canonical single-layer-delta floor): a 2-compartment learner with FIXED-RANDOM apical feedback (feedback alignment — no weight transport, no global backprop, local info) + local somato-dendritic mismatch solved XOR 1.000/5-seeds = the exact-backprop oracle, where a competent linear baseline scored 0.50 (chance). This plan tests whether that PRINCIPLE survives the SPIKING substrate. **Honest ceiling (NEVER spun): a PASS addresses ONLY the credit-assignment ROOT (#1); it does NOT solve developmental/embodiment (#3) and is NOT "conversation solved".**
- **The gradient ORACLE (`sim/bptt_snn.py`, reuse, do NOT modify):** `LIFLayer`, `forward_unroll(...) -> {"spikes":[per-layer], "v":[per-layer]}`, `cross_entropy_loss_np(logits, target_idx) -> float`, `softmax_grad_np(logits, target_idx) -> grad`, `backward_unroll(...) -> (weight_grads, input_grad)`, `make_abc_dataset(...) -> (inputs, targets)`. This is the TRUE BPTT gradient for a tiny spiking net. It is the correctness ceiling ONLY — it uses weight transport and is biologically implausible; it is imported in tests + the gate runner's offline measurement, NEVER in `sim/dendritic_*.py`.
- **Protected / DRY (byte-UNMODIFIED across the WHOLE dendritic commit range — verify empty-diff each commit):** `research/runners/abstention_gate.py` + `tests/test_abstention_gate.py` (the validated no-confab moat, gate 650 — the distinctive contribution, MUST stay byte-identical + green), `research/runners/gate_core.py`/`song_g1_core.py`/`subword_lm_gate_core.py`/`generator_g_core.py`/`generator_h_core.py` (each owns frozen bars), `sim/bptt_snn.py`/`sim/bptt_snn_gpu.py`, `sim/bridge.py`, `research/runners/bio_three_factor.py`. Dendritic work adds NO new global bar; its bars live ONLY in `research/runners/dendritic_core.py` as frozen `_DEND_*` constants.
- **Anti-cheat non-negotiables:** FIXED bars never tuned; ≥3 seeds; the 2026-05-03 permuted-label control is LOAD-BEARING (it caught a year of false positives — a result that does not beat its own permuted control is NOT real); a false-PASS or FAIL is an honest propagated finding, NOT config-cranked. An Arch-A FAIL is the decision-relevant terminus (the principle does not survive the spiking substrate at feasible local scale) — NOT a license to escalate to Arch B/C.

---

## Task 0: Falsify-cheaply grounding pin (commit now; green only after Task 4 — intentional)

**Files:** Create `tests/test_dendritic_grounding.py`

**Step 1: Write the failing test**

```python
"""Grounding pin: the dendritic_wa_gate pipeline TURNS end-to-end at a
tiny config and produces an interpretable verdict. (The PRINCIPLE was
already grounded by the 2026-05-17 rate/XOR falsify-cheaply probe;
this pin is the spiking-substrate end-to-end turn.) Green after Task 4."""
import subprocess
import sys
import json
import pytest


def test_dendritic_wa_gate_pipeline_turns(tmp_path):
    out = str(tmp_path / "d.json")
    ck = str(tmp_path / "d.ckpt")
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.dendritic_wa_gate",
         "--seeds", "42,43,44", "--tiny", "--out", out, "--ckpt", ck],
        capture_output=True, text=True, timeout=600)
    if r.returncode == 2 and "NOT RUNNABLE" in r.stdout:
        pytest.skip("dependency absent in this env")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = json.loads(open(out, encoding="utf-8").read())
    assert d["n_seeds"] == 3 and "aggregate_verdict" in d
    for s in d["per_seed"]:
        assert "verdict" in s and "grad_cosine" in s
```

**Step 2:** Run `python -m pytest tests/test_dendritic_grounding.py -q` → FAIL (no module). Intentional; green after Task 4.

**Step 3: Commit**
```bash
git add tests/test_dendritic_grounding.py
git commit -m "test(Dendritic): falsify-cheaply grounding pin (spiking pipeline turns end-to-end) -- green after Task 4"
```

---

## Phase A — pure-CPU-TDD (Tasks 1–3). Fresh subagent per task; strict failing-test→minimal-impl→run→commit; controller trust-but-verify each diff. Tasks 1+2 get a dedicated adversarial reviewer BEFORE Phase B.

### Task 1: `sim/dendritic_neuron.py` — spiking two-compartment pyramidal (LOAD-BEARING)

**Files:** Create `sim/dendritic_neuron.py`, `tests/test_dendritic_neuron.py`

**Step 1: failing tests** (`tests/test_dendritic_neuron.py`)

```python
"""Pure CPU tests. LOAD-BEARING: (A) the FIXED-RANDOM apical feedback
matrix is provably never mutated by any call (no weight transport);
(B) apical depolarization genuinely LOWERS the somatic effective
threshold (Larkum BAC); (C) no autograd/torch imported."""
import numpy as np
import sim.dendritic_neuron as dn


def test_no_autograd_imported():
    import inspect
    src = inspect.getsource(dn)
    assert "torch" not in src and "autograd" not in src


def test_fixed_apical_feedback_never_mutated():
    layer = dn.DendriticLayer(n_pre=4, n_post=3, n_teacher=2, seed=7)
    B0 = layer.B_apical.copy()
    rng = np.random.default_rng(0)
    for _ in range(50):
        layer.step(x_basal=rng.normal(size=4),
                   teacher=rng.normal(size=2))
    assert np.array_equal(layer.B_apical, B0)  # FIXED random, untouched


def test_apical_depolarization_lowers_threshold():
    layer = dn.DendriticLayer(n_pre=4, n_post=3, n_teacher=2, seed=1)
    x = np.ones(4) * 0.5
    # no apical drive vs strong apical drive, same basal input
    s_noap = layer.effective_threshold(teacher=np.zeros(2))
    s_ap = layer.effective_threshold(teacher=np.ones(2) * 5.0)
    assert np.all(s_ap <= s_noap)            # BAC: apical eases firing
    assert np.any(s_ap < s_noap)


def test_step_is_deterministic_given_state():
    a = dn.DendriticLayer(n_pre=3, n_post=2, n_teacher=2, seed=42)
    b = dn.DendriticLayer(n_pre=3, n_post=2, n_teacher=2, seed=42)
    x = np.array([0.2, -0.4, 0.7])
    t = np.array([1.0, 0.0])
    o1 = a.step(x_basal=x, teacher=t)
    o2 = b.step(x_basal=x, teacher=t)
    assert np.array_equal(o1["soma_rate"], o2["soma_rate"])
    assert np.array_equal(o1["v_basal"], o2["v_basal"])
```

**Step 3: implementation** (`sim/dendritic_neuron.py`)

```python
"""Spiking two-compartment pyramidal (Larkum BAC; Guerguiev-Lillicrap-
Richards 2017 segregated dendrites). Per-neuron compartments: basal
(bottom-up forward drive), apical (top-down feedback through a FIXED
RANDOM projection -- feedback alignment, set once from seed, NEVER
learned, NO weight transport from forward weights), soma (BAC
integration: basal alone needs high threshold; apical depolarization
LOWERS the effective threshold). Pure numpy; biologically-local by
construction; NO autograd. ASCII only. Mirrors the SHAPE of
sim/bptt_snn.LIFLayer but does NOT import or modify it."""
from __future__ import annotations
import numpy as np


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class DendriticLayer:
    def __init__(self, n_pre, n_post, n_teacher, seed=0,
                 theta_high=1.0, apical_gain=0.5, leak=0.9):
        rng = np.random.default_rng(seed)
        self.W_basal = rng.normal(0.0, 1.0, (n_pre, n_post))
        # FIXED RANDOM apical feedback -- feedback alignment. Never
        # learned, never read from W_basal (no weight transport).
        self.B_apical = rng.normal(0.0, 1.0, (n_teacher, n_post))
        self.theta_high = float(theta_high)
        self.apical_gain = float(apical_gain)
        self.leak = float(leak)
        self.v_basal = np.zeros(n_post)
        self.v_apical = np.zeros(n_post)

    def _apical_drive(self, teacher):
        return np.asarray(teacher, float) @ self.B_apical

    def effective_threshold(self, teacher):
        # Larkum BAC: apical depolarization lowers the threshold.
        ap = np.maximum(0.0, self._apical_drive(teacher))
        return self.theta_high - self.apical_gain * ap

    def step(self, x_basal, teacher):
        self.v_basal = (self.leak * self.v_basal
                        + np.asarray(x_basal, float) @ self.W_basal)
        self.v_apical = self._apical_drive(teacher)
        theta_eff = self.theta_high - self.apical_gain * np.maximum(
            0.0, self.v_apical)
        soma_rate = _sig(self.v_basal - theta_eff)
        return {"soma_rate": soma_rate, "v_basal": self.v_basal.copy(),
                "v_apical": self.v_apical.copy(),
                "theta_eff": theta_eff}
```

**Step 4:** `python -m pytest tests/test_dendritic_neuron.py -q` → PASS. **Step 5: commit** `feat(Dendritic): spiking 2-compartment pyramidal (FIXED-random apical feedback never mutated; BAC threshold; no autograd)`. **Step 6:** controller `git show --stat HEAD` ONLY the 2 files; protected byte-empty.

---

### Task 2: `sim/dendritic_plasticity.py` — LOCAL Urbanczik-Senn rule (LOAD-BEARING: the credit-assignment proof)

**Files:** Create `sim/dendritic_plasticity.py`, `tests/test_dendritic_plasticity.py`

**Step 1: failing tests** — the decisive one finite-difference-checks the local rule vs the `bptt_snn` oracle gradient WITHOUT weight transport.

```python
"""LOAD-BEARING: the LOCAL somato-dendritic rule's weight-change
direction positively aligns with the TRUE gradient (sim/bptt_snn
oracle) on a tiny net WITHOUT weight transport -- the GLR-2017
credit-assignment proof in THIS codebase. Oracle imported HERE
(test) only; never in the shipped module. No autograd in the module."""
import numpy as np
import inspect
import sim.dendritic_plasticity as dp


def test_no_autograd_in_module():
    src = inspect.getsource(dp)
    assert "torch" not in src and "autograd" not in src


def test_local_update_shapes_and_purity():
    pre = np.array([0.5, 0.2, 0.9])
    soma = np.array([0.8, 0.1])
    vbas = np.array([0.3, -0.4])
    gate = np.array([1.0, 0.0])
    dw = dp.urbanczik_senn_update(pre, soma, vbas, gate)
    assert dw.shape == (3, 2)
    # apical-gated: post-unit 1 (gate 0) gets zero weight change
    assert np.allclose(dw[:, 1], 0.0)


def test_local_rule_aligns_with_bptt_oracle_no_weight_transport():
    """Cosine(local Delta-w, true BPTT grad) must be POSITIVE and well
    above orthogonal-chance on a tiny 2-layer net, using FIXED-random
    feedback (no weight transport). Reproduces GLR-2017 in-codebase."""
    rng = np.random.default_rng(0)
    n_in, n_hid, n_out = 5, 6, 3
    X = rng.normal(size=(8, n_in))
    y = rng.integers(0, n_out, size=8)
    W1 = rng.normal(0, 0.5, (n_in, n_hid))
    W2 = rng.normal(0, 0.5, (n_hid, n_out))
    Bfix = rng.normal(0, 1.0, (n_out, n_hid))  # FIXED random feedback

    def sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    h = sig(X @ W1)
    logits = h @ W2
    # softmax error at output
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    e = p.copy()
    e[np.arange(8), y] -= 1.0
    # TRUE gradient wrt W1 (oracle: uses W2 = weight transport)
    g_true = X.T @ ((e @ W2.T) * h * (1 - h))
    # LOCAL dendritic update: apical = e @ Bfix (FIXED random, NOT W2)
    apical = e @ Bfix
    dw_local = np.zeros_like(W1)
    for i in range(8):
        dw_local += dp.urbanczik_senn_update(
            X[i], h[i], h[i], np.ones(n_hid),
            apical_signal=apical[i])
    cos = (np.sum(g_true * -dw_local)
           / (np.linalg.norm(g_true) * np.linalg.norm(dw_local) + 1e-9))
    assert cos > 0.3, cos      # genuine gradient approximation
```

**Step 3: implementation** (`sim/dendritic_plasticity.py`)

```python
"""LOCAL Urbanczik-Senn somato-dendritic mismatch plasticity, apical-
gated. Delta-w ~ apical-gated (somatic_rate - phi(v_basal)) * pre.
When an apical_signal is supplied (top-down feedback delivered via the
neuron's FIXED-RANDOM B_apical -- NO weight transport), it sets the
local dendritic target the soma is pulled toward. Pure numpy;
biologically-LOCAL by construction; NO autograd. ASCII only. Does NOT
import sim/bptt_snn (that is the test-only oracle)."""
from __future__ import annotations
import numpy as np


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def urbanczik_senn_update(pre_rate, soma_rate, v_basal,
                          apical_gate, apical_signal=None, lr=1.0):
    """pre_rate (n_pre,), soma_rate/v_basal/apical_gate (n_post,).
    apical_signal (n_post,) optional: the top-down teaching mismatch
    already projected through the FIXED-RANDOM apical feedback by the
    caller (no weight transport here). Returns dW (n_pre, n_post)."""
    pre = np.asarray(pre_rate, float)
    soma = np.asarray(soma_rate, float)
    vb = np.asarray(v_basal, float)
    gate = np.asarray(apical_gate, float)
    if apical_signal is None:
        mismatch = soma - _sig(vb)            # self-prediction error
    else:
        # apical-driven local target (GLR-2017): soma pulled toward
        # the FIXED-random-projected top-down signal.
        mismatch = np.asarray(apical_signal, float) * soma * (1.0 - soma)
    dw = np.outer(pre, lr * gate * mismatch)
    return dw
```

**Step 4:** PASS. **Step 5: commit** `feat(Dendritic): LOCAL Urbanczik-Senn plasticity -- aligns with bptt_snn oracle gradient WITHOUT weight transport (GLR-2017 in-codebase, adversarially pinned)`. **Step 6:** controller trust-but-verify + **dedicated adversarial reviewer for Tasks 1+2** (charge: try to break — can the FIXED feedback ever be mutated / read from W_basal (weight transport)? is the cosine test gameable / is 0.3 a vacuous bar given orthogonal-chance≈0? does any path import autograd? is apical-gating real? STRENGTHEN-only fixes; mirror the Generator-S/D/G/H reviews that caught real holes). Must APPROVE before Phase B.

---

### Task 3: `research/runners/dendritic_core.py` — FIXED-bar verdict (own frozen constants)

**Files:** Create `research/runners/dendritic_core.py`, `tests/test_dendritic_core.py`

**Frozen bars (justified, never tuned):** `_DEND_GRAD_COSINE_MIN = 0.30` (local-rule vs true-gradient cosine; orthogonal-chance ≈ 0, so 0.30 is a defensible floor proving genuine-but-imperfect gradient approximation — feedback alignment is known imperfect, NOT perfect backprop; never tuned), `_DEND_HIDDEN_CREDIT_MIN = 0.90` (spiking hidden-credit task accuracy the dendritic learner must reach), `_DEND_NOHIDDEN_FLOOR_MAX = 0.70` (the no-hidden-credit baseline must NOT exceed — proves the task genuinely requires hidden credit), `_DEND_PERMUTED_MAX = 0.70` (permuted-label control must NOT clear — the 2026-05-03 catcher), `_DEND_MIN_SEEDS = 3`.

Test the adversarial matrix (always-pass-without-permuted-control→FAIL-closed; missing/vacuous control→FAIL-closed; non-finite→FAIL-closed; `biologically_local=False`→FAIL-closed; results cannot move the frozen `_DEND_*`; <3 seeds→FAIL) — mirror `tests/test_generator_h_core.py` shape exactly. `dend_verdict(hidden_credit, nohidden_floor, permuted, grad_cosine, biologically_local, has_permuted_control)` PASS iff `has_permuted_control` AND `biologically_local is True` AND all metrics finite AND `hidden_credit >= _DEND_HIDDEN_CREDIT_MIN` AND `nohidden_floor <= _DEND_NOHIDDEN_FLOOR_MAX` AND `permuted <= _DEND_PERMUTED_MAX` AND `grad_cosine >= _DEND_GRAD_COSINE_MIN`. `dend_aggregate_multiseed` PASS iff ≥3 seeds, every seed has the permuted control, every seed PASS. (Full code: mirror `research/runners/generator_h_core.py` structure verbatim — `import math`; non-finite fail-closed guard FIRST; frozen module constants; strict comparators; same return-dict shape.) **Step 6:** dedicated adversarial reviewer (same charge style as Generator-H core). Commit `feat(Dendritic): pure FIXED-bar verdict (own frozen _DEND_* bars; permuted-control + biologically-local + oracle-cosine load-bearing; adversarially pinned)`.

---

## Phase B — integration (Tasks 4–5). Import/signature smoke + the gate itself (project pattern; NOT contrived orchestration tests).

### Task 4: `research/runners/dendritic_wa_gate.py` — thin gate runner (makes Task 0 green)

**Files:** Create `research/runners/dendritic_wa_gate.py`, `tests/test_dendritic_wa_gate_smoke.py`

DRY mirror of `research/runners/generator_h_gate.py` SHAPE (kill-safe `.resume.json`; `<3 seeds -> return 2`; ASCII-only banner stating the HONEST CEILING: "credit-assignment ROOT #1 only; NOT conversation-solved; NOT #3 developmental/embodiment"). The **decisive cheap slice** task = a tiny spiking hidden-credit task built from `sim.bptt_snn.make_abc_dataset` (reuse, do NOT rebuild): three learners on the SAME tiny spiking net — (a) **dendritic** (Task1 `DendriticLayer` hidden + Task2 local plasticity, FIXED-random apical feedback), (b) **no-hidden-credit floor** (output-layer delta only — provably fails hidden credit), (c) **permuted-label** dendritic (the load-bearing control). Plus the **oracle cosine** via `sim.bptt_snn.backward_unroll` (offline measurement in the runner — allowed; the *shipped* `sim/dendritic_*.py` import no oracle/autograd). Per seed record: hidden_credit acc, nohidden_floor acc, permuted acc, grad_cosine, biologically_local flag (asserted: FIXED feedback unchanged + no autograd in shipped modules), + the learned-behaviour transcript. `dend_verdict` + `dend_aggregate_multiseed` (seeds 42,43,44). Smoke test: module imports, `main` callable, `--seeds 42,43 --tiny` → exit 2 + "NOT RUNNABLE"; full `--seeds 42,43,44 --tiny` → pipeline turns (makes Task 0 green). Commit `feat(Dendritic): gate runner (spiking hidden-credit + oracle-cosine + permuted control; FIXED bars via dendritic_core; kill-safe; honest ceiling)`. **Step 6:** controller verify ONLY 2 files; `git diff 3528981..HEAD` on all protected modules EMPTY; assert no `sim/dendritic_*.py` imports `torch`/`autograd`.

### Task 5: `tests/test_dendritic_noharm.py` — LOAD-BEARING no-harm pin

Mirror `tests/test_generator_h_noharm.py`: (1) `abstention_gate` byte-contract intact (gate 650, the 5 canonical assertions) + `tests/test_abstention_gate.py` green; (2) importing `sim.dendritic_neuron` / `sim.dendritic_plasticity` / `research.runners.dendritic_core` / `research.runners.dendritic_wa_gate` does NOT pull `song_g1_core`/`subword_lm_gate_core`/`generator_g_core`/`generator_h_core`/`gate_core`; (3) `dendritic_core` owns its frozen `_DEND_*`; (4) NEW: assert no shipped `sim/dendritic_*.py` source contains `torch` or `autograd` (self-contained biologically-local by construction). Commit `test(Dendritic): LOAD-BEARING no-harm (moat byte-identical+green; no protected-core pull; no autograd in shipped modules; _DEND_* frozen)`. **Step 6:** controller whole-range protected diff `git diff 3528981..HEAD -- <all protected>` EMPTY; representative validated suite green.

---

## Task 6: Controller-only — decisive run + MANDATORY smell-test + honest propagation + (conditional) integrate

> CONTROLLER's job, NOT a subagent.

1. **Grounding-first tiny:** `python -m research.runners.dendritic_wa_gate --seeds 42,43,44 --tiny --out .../dendritic_wa_gate.tiny.json` → exit 0, pipeline turns, Task 0 pin green. Toy verdict NOT propagated.
2. **Decisive run:** `python -m research.runners.dendritic_wa_gate --seeds 42,43,44 --out research/findings/raw/g11_bg/dendritic_wa_gate.json` (FIXED pre-registered config; kill-safe).
3. **MANDATORY anti-cheat smell-test (scrutinize a PASS HARDER than a FAIL):** recompute from recorded JSON (NO re-run, NO bar-tuning). Verify per seed: the **permuted-label control genuinely does NOT clear** `_DEND_PERMUTED_MAX` (the 2026-05-03 false-positive catcher — a result not beating its own permuted control is NOT real); the **no-hidden-credit floor genuinely fails** (≤0.70 — proves the task needs hidden credit); **biologically-local genuinely holds** (FIXED apical feedback bytes unchanged post-training; no `sim/dendritic_*.py` imports autograd); the **grad_cosine ≥ 0.30** is genuine (read the actual local-Δw-vs-oracle numbers). Read the learned-behaviour transcripts. NO overclaim.
4. **Honest propagation (EITHER outcome):** write `research/findings/2026-05-17-dendritic-credit-assignment-{PASS|NEGATIVE}.md` (verbatim numbers + transcripts + FIXED bars + the explicit HONEST CEILING: credit-assignment ROOT #1 ONLY, NOT conversation-solved, NOT #3); append a `webapp/capability_status.json` pillar (honest verdict + ceiling foregrounded); `python -m pytest tests/test_webapp_server.py -k capability_status -q` MUST stay green (fix JSON if drifted, NOT the test); commit + push BOTH remotes.
5. **CONDITIONAL integration (ONLY if scrutinized genuine PASS — the "integrate" the user asked for):** wire the validated dendritic credit-assignment as an OPT-IN, DEFAULT-OFF path additively (new flag / new optional module entry point) usable by the validated W→A/`bio_three_factor` pipeline WITHOUT modifying any frozen/validated module; integration is itself gated by the SAME no-harm discipline (moat + frozen cores byte-untouched + green; protected whole-range empty-diff; representative suite green). If FAIL: NO integration — propagate the honest decision-relevant terminus (the principle does not survive the spiking substrate at feasible local scale); the deliverable stays the validated assets; NOT config-cranked into Arch B/C.
6. Bring Task 6's result back to the controller for the next autonomous decision point. **LATER increment (noted, NOT in this slice, YAGNI):** the full `bio_three_factor.run_three_factor` W→A 4/6-aligned confirmation + scaling.

---

## Remember
- Exact paths; complete code in plan; TDD (fail→impl→pass→commit); frequent commits.
- DRY: `abstention_gate`/`bptt_snn`/`bio_three_factor`/frozen cores byte-UNMODIFIED; `dendritic_core` owns frozen `_DEND_*`; protected byte-empty every commit-scoped diff.
- @superpowers:test-driven-development every task; @superpowers:subagent-driven-development drives it; Tasks 1+2 + Task 3 get the dedicated adversarial reviewer before Phase B.
- ASCII-only; ≥3 seeds; permuted-label control LOAD-BEARING; FIXED bars NEVER tuned; no autograd in any shipped `sim/dendritic_*.py`; honest ceiling stated up front and never spun (ROOT #1 only); the validated no-confab moat MUST stay byte-identical + green.
