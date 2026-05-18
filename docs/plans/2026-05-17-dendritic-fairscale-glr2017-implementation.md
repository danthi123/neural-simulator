# Dendritic Fair-Scale GLR-2017 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (fresh subagent per task; two-stage spec+quality review; controller trust-but-verify each git diff with protected modules byte-empty). Tasks 1 & 2 additionally get a DEDICATED ADVERSARIAL REVIEWER before Phase B (mirror the dendritic-Phase-A review that caught the W2-confound + sign + truthy-string holes). Task 5 is the CONTROLLER's job, NOT a subagent.

**Goal:** Run the literal Lillicrap-2016/GLR-2017 discriminating regime (MNIST deep-MLP feedback-alignment, owner-authorized Option 2) and decide, with a pre-registered THREE-STATE (VOID/PASS/FAIL) gate, whether biologically-LOCAL dendritic credit assignment genuinely learns real data where the readout confound is defeated.

**Architecture:** 3 net-new modules — `sim/dendritic_mlp.py` (deep sigmoid MLP; per-layer FIXED-RANDOM feedback; hidden learning delegated to the committed sign-correct `sim.dendritic_plasticity`; `oracle` mode = hand-derived numpy backprop fenced as measurement/validity ONLY), `research/runners/dendritic_fair_core.py` (frozen `_DFAIR_*`; VOID-first instrument-validity gate), `research/runners/dendritic_fair_gate.py` (kill-safe/pausable runner reusing `sim.train_checkpoint`). Everything load-bearing reused byte-UNMODIFIED.

**Tech Stack:** Python 3, numpy, stdlib. NO `torch.autograd`/`backward` in any shipped learning path (the `oracle` is hand-derived numpy backprop, measurement-only). pytest. ASCII-only prints. @superpowers:test-driven-development every task.

---

## Context the implementer MUST know (zero-context briefing)

- **Why this exists:** every cheap probe for "does the biologically-local dendritic rule do genuine credit assignment" was NON-DISCRIMINATING — a readout over rich hidden features rescues *any* hidden rule (even wrong-sign) at cheap scale (4 attempts, structural). The owner explicitly chose **Option 2**: run the literature's OWN discriminating regime (MNIST FA), where task structure defeats the readout confound (a linear/random-feature readout provably cannot reach high MNIST accuracy — Lillicrap 2016). The rule FORM is already correct (committed `sim.dendritic_plasticity`, weight-transport cos=+1.0).
- **HONEST CEILING (bake into every doc/banner; NEVER spin):** a scrutinized PASS = the #1 credit-assignment lever works biologically-locally at the literature's discriminating scale on real data — addresses the ROOT, explicitly **NOT** #3 (developmental/embodiment), **NOT** "conversation solved"; integration into the conversational stack is a SEPARATE later effort, only noted. A faithful FAIL at the literature's own regime = the strongest possible honest terminus. **VOID** (V1 or V2 unmet) = "instrument not soundly constructible even at fair scale" — explicitly a third state, NOT a science PASS/FAIL.
- **Committed credit-assignment core (reuse byte-UNMODIFIED, do NOT modify):** `sim/dendritic_plasticity.py::urbanczik_senn_update(pre_rate, soma_rate, v_basal, apical_gate, apical_signal=None, lr=1.0)`. Sign-correct: when `apical_signal` given, `mismatch = apical_signal * soma_rate * (1 - soma_rate)`; returns `dw = np.outer(pre_rate, lr*apical_gate*mismatch)`. Batched equivalent (gate=ones, faithfulness-asserted): `dW = pre.T @ ((e @ B_l) * a_l * (1-a_l))`. The `soma*(1-soma)` is the sigmoid-derivative → hidden activations MUST be **sigmoid** for faithful reuse (Lillicrap 2016 used sigmoid/tanh MNIST MLPs; this is correct, not a limitation).
- **Kill-safe reuse (HARD owner requirement; do NOT modify these):** `sim/train_checkpoint.py::save_checkpoint(path, epoch, weights, rng_state, loss_history)` (atomic `.tmp`+`os.replace`), `load_checkpoint(path)`→`{epoch,weights(list[np]),rng_state,loss_history}|None`, `resume_epoch(ckpt)`→`0` or `last+1`. Pattern reference (study, do NOT modify): `research/runners/scaled_subword_lm_train.py` (per-epoch checkpoint; resume from last; `KeyboardInterrupt`→flush final checkpoint→clean `exit 0`).
- **Self-contained MNIST (mirror, do NOT modify `corpus_fetch.py`):** download MNIST ONCE from a public URL → cache `data/mnist.npz` (verify shape/count) → thereafter zero external dependency; degrade/clear NOT-RUNNABLE `exit 2` if absent+offline (mirror `corpus_fetch`'s cache-once + offline-degrade discipline; record provenance in the JSON). Public dataset is owner-authorized; the *artifact* (trained weights) needs no external dep → self-contained at runtime.
- **Protected / DRY (byte-UNMODIFIED across the WHOLE fair-scale commit range — verify empty-diff EVERY commit-scoped diff):** `sim/dendritic_plasticity.py`, `sim/dendritic_neuron.py`, `research/runners/abstention_gate.py` + `tests/test_abstention_gate.py` (the distinctive no-confab moat — MUST stay 7/7 green), every frozen `*_core` (`gate_core`/`song_g1_core`/`subword_lm_gate_core`/`generator_g_core`/`generator_h_core`/`dendritic_core`), `sim/bptt_snn*`, `sim/bridge.py`, `sim/train_checkpoint.py`, `research/runners/scaled_subword_lm_train.py`, `research/runners/corpus_fetch.py`, `bio_three_factor`. `dendritic_fair_core` owns its OWN frozen `_DFAIR_*` (NEVER mutate `dendritic_core`'s `_DEND_*`). NO new global bar.
- **Anti-cheat non-negotiables:** FIXED `_DFAIR_*` never tuned; ≥3 seeds; the load-bearing controls (wrong-sign, global-scalar, permuted-label) MUST fail and the oracle MUST work or the result is **VOID**; mandatory smell-test scrutinizing a PASS HARDER than a FAIL; VOID/FAIL/false-PASS-caught is an honest propagated finding, NOT config-cranked.

---

## Task 0: Falsify-cheaply grounding pin (commit now; green after Task 3)

**Files:** Create `tests/test_dendritic_fair_grounding.py`

```python
"""Grounding pin: the dendritic_fair_gate pipeline TURNS end-to-end on
a TINY synthetic zero-network config and produces an interpretable
THREE-STATE (VOID/PASS/FAIL) verdict. Green after Task 3."""
import subprocess, sys, json, pytest


def test_dendritic_fair_gate_pipeline_turns(tmp_path):
    out = str(tmp_path / "d.json"); ck = str(tmp_path / "d.ckpt")
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.dendritic_fair_gate",
         "--seeds", "42,43,44", "--tiny-synth", "--out", out,
         "--ckpt-dir", ck],
        capture_output=True, text=True, timeout=900)
    if r.returncode == 2 and "NOT RUNNABLE" in r.stdout:
        pytest.skip("dependency/dataset absent in this env")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = json.loads(open(out, encoding="utf-8").read())
    assert d["n_seeds"] == 3 and d["GATE"] in ("VOID", "PASS", "FAIL")
    for s in d["per_seed"]:
        assert s["verdict"]["GATE"] in ("VOID", "PASS", "FAIL")
```

Run `python -m pytest tests/test_dendritic_fair_grounding.py -q` → FAIL (no module; intentional). Commit:
```bash
git add tests/test_dendritic_fair_grounding.py
git commit -m "test(Dendritic-fair): falsify-cheaply grounding pin (3-state pipeline turns) -- green after Task 3"
```

---

## Phase A — pure-CPU-TDD (Tasks 1–2). Fresh subagent per task; controller trust-but-verify each diff. Both get the dedicated adversarial reviewer BEFORE Phase B.

### Task 1: `sim/dendritic_mlp.py` — deep sigmoid MLP, feedback alignment (LOAD-BEARING)

**Files:** Create `sim/dendritic_mlp.py`, `tests/test_dendritic_mlp.py`

**Step 1: failing tests** (`tests/test_dendritic_mlp.py`)

```python
"""LOAD-BEARING: (A) per-layer FIXED-RANDOM feedback B never mutated
and never derived from forward W (no weight transport); (B) the
batched hidden update EQUALS the committed per-sample
sim.dendritic_plasticity sum (faithful reuse); (C) NO autograd in the
shipped path; (D) oracle mode is hand-derived numpy backprop that
genuinely descends loss on a tiny problem (positive control works);
(E) modes are clean + deterministic."""
import numpy as np
import inspect
import sim.dendritic_mlp as dm
import sim.dendritic_plasticity as dp


def test_no_autograd_in_module():
    src = inspect.getsource(dm)
    assert "torch" not in src and "autograd" not in src


def test_fixed_feedback_never_mutated_no_weight_transport():
    net = dm.DendriticMLP([12, 16, 16, 4], seed=7)
    B0 = [b.copy() for b in net.B]
    W0 = [w.copy() for w in net.W]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 12)); y = rng.integers(0, 4, 20)
    for _ in range(5):
        net.train_step(X, y, mode="local_correct", lr=0.1)
    for b, b0 in zip(net.B, B0):          # B never changes
        assert np.array_equal(b, b0)
    for b in net.B:                       # B never equals any W / W.T
        for w in net.W:
            assert b.shape != w.shape or not np.array_equal(b, w)
            assert (b.shape != w.T.shape) or not np.array_equal(b, w.T)


def test_batched_update_equals_committed_per_sample_sum():
    net = dm.DendriticMLP([5, 6, 3], seed=1)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(8, 5)); y = rng.integers(0, 3, 8)
    # the module exposes the layer-0 hidden dW it WOULD apply:
    dW0 = net._debug_hidden_dW0(X, y)
    acts, e = net._debug_fwd_err(X, y)
    pre, soma = acts[0], acts[1]
    ap = e @ net.B[0]
    ref = np.zeros_like(net.W[0])
    for i in range(8):
        ref += dp.urbanczik_senn_update(
            pre[i], soma[i], soma[i], np.ones(soma.shape[1]),
            apical_signal=ap[i])
    assert np.allclose(dW0, ref, atol=1e-9)


def test_oracle_mode_is_positive_control_descends_loss():
    net = dm.DendriticMLP([8, 16, 16, 3], seed=3)
    rng = np.random.default_rng(4)
    X = rng.normal(size=(64, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 0).astype(int)
    L0 = net.loss(X, y)
    for _ in range(300):
        net.train_step(X, y, mode="oracle", lr=0.2)
    assert net.loss(X, y) < 0.5 * L0       # hand-derived BP works


def test_modes_deterministic_given_seed():
    a = dm.DendriticMLP([6, 8, 3], seed=42)
    b = dm.DendriticMLP([6, 8, 3], seed=42)
    rng = np.random.default_rng(5)
    X = rng.normal(size=(10, 6)); y = rng.integers(0, 3, 10)
    a.train_step(X, y, mode="local_correct", lr=0.1)
    b.train_step(X, y, mode="local_correct", lr=0.1)
    assert all(np.array_equal(x, z) for x, z in zip(a.W, b.W))
```

**Step 3: implementation** (`sim/dendritic_mlp.py`)

```python
"""Deep sigmoid MLP for literature-faithful feedback alignment
(Lillicrap 2016; GLR-2017). Per HIDDEN layer: forward W + a FIXED
RANDOM feedback matrix B (set once from seed, NEVER learned, NEVER
derived from any forward W -- no weight transport). Hidden learning
delegates to the committed sign-correct
sim.dendritic_plasticity.urbanczik_senn_update (batched == per-sample
sum). Output layer by local delta. `oracle` mode is a HAND-DERIVED
numpy backprop used ONLY as the V1 positive-control + the emergent-
alignment measurement -- it is fenced as measurement/validity, NOT a
shipped biologically-local learning mode, and uses NO autograd. Pure
numpy; ASCII only. Does NOT import sim.bptt_snn."""
from __future__ import annotations
import numpy as np
from sim.dendritic_plasticity import urbanczik_senn_update  # committed


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class DendriticMLP:
    def __init__(self, sizes, seed=0):
        # sizes e.g. [784,512,256,128,10]; hidden activations sigmoid.
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.n_out = sizes[-1]
        self.W, self.B = [], []
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            # Glorot (sigmoid-appropriate) -> oracle trains (V1).
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            self.W.append(rng.uniform(-lim, lim, (fan_in, fan_out)))
        # one FIXED RANDOM feedback per HIDDEN layer (n_out x n_hid)
        for i in range(1, len(sizes) - 1):
            self.B.append(rng.normal(0, 1.0, (self.n_out, sizes[i])))

    # ---- forward / loss -------------------------------------------
    def _forward(self, X):
        acts = [np.asarray(X, float)]
        for li in range(len(self.W) - 1):
            acts.append(_sig(acts[-1] @ self.W[li]))
        logits = acts[-1] @ self.W[-1]
        return acts, logits

    def loss(self, X, y):
        _, lg = self._forward(X)
        p = _softmax(lg)
        n = len(y)
        return float(-np.log(p[np.arange(n), y] + 1e-12).mean())

    def accuracy(self, X, y):
        _, lg = self._forward(X)
        return float(np.mean(np.argmax(lg, 1) == y))

    # ---- internal helpers for tests -------------------------------
    def _debug_fwd_err(self, X, y):
        acts, lg = self._forward(X)
        e = _softmax(lg).copy()
        e[np.arange(len(y)), y] -= 1.0          # dL/dlogits
        return acts, e

    def _debug_hidden_dW0(self, X, y):
        acts, e = self._debug_fwd_err(X, y)
        a_prev, a_l = acts[0], acts[1]
        ap = e @ self.B[0]
        return a_prev.T @ (ap * a_l * (1.0 - a_l))   # == committed sum

    # ---- one training step ----------------------------------------
    def train_step(self, X, y, mode, lr):
        acts, e = self._debug_fwd_err(X, y)        # e = dL/dlogits
        nW = len(self.W)
        upd = [None] * nW
        # output layer: local delta (descent) for ALL local modes;
        # oracle uses the same exact output gradient (identical here).
        upd[-1] = -(acts[-1].T @ e)
        if mode == "oracle":
            # hand-derived numpy backprop (measurement/validity ONLY)
            delta = e
            for li in range(nW - 1, 0, -1):
                gW = acts[li].T @ delta
                upd[li] = -gW if li == nW - 1 else upd[li]
                if li >= 1:
                    a = acts[li]
                    delta = (delta @ self.W[li].T) * a * (1.0 - a)
                    upd[li - 1] = -(acts[li - 1].T @ delta) \
                        if li - 1 >= 1 else upd[li - 1]
            # layer 0..nW-2 hidden grads via the same backprop:
            acts2, e2 = self._debug_fwd_err(X, y)
            d = e2
            grads = [None] * nW
            grads[nW - 1] = acts2[nW - 1].T @ d
            for li in range(nW - 2, -1, -1):
                a = acts2[li + 1]
                d = (d @ self.W[li + 1].T) * a * (1.0 - a)
                grads[li] = acts2[li].T @ d
            for li in range(nW):
                upd[li] = -grads[li]
        else:
            for li in range(nW - 1):               # hidden layers
                a_prev, a_l = acts[li], acts[li + 1]
                ap = e @ self.B[li]                 # FIXED random
                if mode in ("local_correct", "permuted"):
                    dW = a_prev.T @ (ap * a_l * (1.0 - a_l))
                elif mode == "local_wrongsign":
                    dW = -(a_prev.T @ (ap * a_l * (1.0 - a_l)))
                elif mode == "global_scalar":
                    g = float(self.loss(X, y))     # single scalar
                    dW = -g * (a_prev.T @ (a_l - 0.5))
                else:
                    raise ValueError("unknown mode %r" % mode)
                upd[li] = -dW if mode != "global_scalar" else dW
        for li in range(nW):
            self.W[li] = self.W[li] + lr * upd[li]

    def hidden_grad_alignment(self, X, y):
        """cos(applied layer-0 local update, hand-derived true grad).
        Measurement-only (uses backprop); never fed to the rule."""
        acts, e = self._debug_fwd_err(X, y)
        local = -(self._debug_hidden_dW0(X, y))    # applied step dir
        d = e
        for li in range(len(self.W) - 1, 0, -1):
            a = acts[li]
            d = (d @ self.W[li].T) * a * (1.0 - a)
        gtrue = acts[0].T @ d
        desc = -gtrue
        n = float(np.sum(local * desc))
        return n / (np.linalg.norm(local) * np.linalg.norm(desc) + 1e-9)
```

> NOTE: the implementer must verify `test_batched_update_equals_committed_per_sample_sum` passes EXACTLY (1e-9) — it pins faithful reuse of the committed rule. If the committed `urbanczik_senn_update` default `lr` or arg order differs, adapt the *test's* per-sample call to the committed signature (do NOT modify `sim/dendritic_plasticity.py`); the batched form in `_debug_hidden_dW0` must equal that committed sum.

Run tests → all pass. Commit:
```bash
git add sim/dendritic_mlp.py tests/test_dendritic_mlp.py
git commit -m "feat(Dendritic-fair): deep sigmoid MLP feedback-alignment (FIXED-random B never transported; committed-rule batched==per-sample; oracle=hand-BP measurement-only; no autograd)"
```
Controller: `git show --stat HEAD` ONLY the 2 files; protected byte-empty.

### Task 2: `research/runners/dendritic_fair_core.py` — THREE-STATE FIXED-bar verdict

**Files:** Create `research/runners/dendritic_fair_core.py`, `tests/test_dendritic_fair_core.py`

**Frozen bars (justified, never tuned; MNIST 10-class chance 0.10):** `_DFAIR_ORACLE_MIN=0.95` (V1: a sound harness trivially backprops MNIST >0.95 — the cheap probes died exactly here; ≥0.95 proves the instrument's optimization works), `_DFAIR_WRONGSIGN_MAX=0.30` (V2: an inverted hidden rule on real MNIST, confound defeated, must be ≪ correct; 0.30 is generous vs chance 0.10), `_DFAIR_CORRECT_MIN=0.90` (load-bearing: GLR-2017 FA reaches ~0.95+ on MNIST; ≥0.90 = genuinely learned), `_DFAIR_GLOBALSCALAR_MAX=0.30`, `_DFAIR_PERMUTED_MAX=0.30`, `_DFAIR_ALIGN_MIN=0.30` (emergent-alignment cosine end-of-training), `_DFAIR_MIN_SEEDS=3`.

**Step 1: failing adversarial tests** (`tests/test_dendritic_fair_core.py`) — mirror `tests/test_dendritic_core.py` shape; cover: frozen bars exact; `_good()` → PASS; **VOID (not FAIL)** when `oracle<0.95` / `wrongsign>0.30` / `has_controls` not `True` / `biologically_local` not `True` / any non-finite / a numeric arg is a string (coerce→VOID, must NOT raise); truthy-string `has_controls='false'`→VOID; PASS→FAIL when (instrument valid but) `correct<0.90` or `globalscalar>0.30` or `permuted>0.30` or `end_align<0.30`; results cannot mutate `_DFAIR_*`; aggregate: `<3 seeds`→FAIL, any seed VOID→VOID, all valid+all PASS→PASS, else FAIL; VOID strictly distinct from FAIL in every path.

**Step 3: implementation** (`research/runners/dendritic_fair_core.py`)

```python
"""Pure FIXED-bar THREE-STATE (VOID/PASS/FAIL) verdict for the
owner-authorized fair-scale dendritic GLR-2017 run. Own frozen
constants; does NOT import/modify any other core or abstention_gate.
INSTRUMENT-VALIDITY FIRST: a non-discriminating or oracle-broken run
is VOID (instrument not sound) -- explicitly NOT a science PASS/FAIL.
Pure stdlib; CPU-unit-testable."""
from __future__ import annotations
import math
from typing import Dict

_DFAIR_ORACLE_MIN = 0.95
_DFAIR_WRONGSIGN_MAX = 0.30
_DFAIR_CORRECT_MIN = 0.90
_DFAIR_GLOBALSCALAR_MAX = 0.30
_DFAIR_PERMUTED_MAX = 0.30
_DFAIR_ALIGN_MIN = 0.30
_DFAIR_MIN_SEEDS = 3


def _nums_ok(*xs):
    for x in xs:
        try:
            if not math.isfinite(float(x)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def dfair_verdict(oracle_heldout, correct_heldout, wrongsign_heldout,
                  globalscalar_heldout, permuted_heldout,
                  end_align_cos, biologically_local,
                  has_controls) -> Dict:
    finite = _nums_ok(oracle_heldout, correct_heldout,
                      wrongsign_heldout, globalscalar_heldout,
                      permuted_heldout, end_align_cos)
    bio = (biologically_local is True)
    ctrl = (has_controls is True)
    # ---- INSTRUMENT VALIDITY FIRST (VOID, not FAIL) ---------------
    valid = bool(finite and bio and ctrl
                 and float(oracle_heldout) >= _DFAIR_ORACLE_MIN
                 and float(wrongsign_heldout) <= _DFAIR_WRONGSIGN_MAX)
    if not valid:
        return {"GATE": "VOID", "instrument_valid": False,
                "biologically_local": bio, "has_controls": ctrl,
                "finite": bool(finite),
                "reason": "V1/V2 instrument-validity unmet "
                          "(oracle>=%.2f and wrongsign<=%.2f and "
                          "bio-local and controls and finite "
                          "required)" % (_DFAIR_ORACLE_MIN,
                                         _DFAIR_WRONGSIGN_MAX),
                "bars": _bars()}
    learned = float(correct_heldout) >= _DFAIR_CORRECT_MIN
    gs_fail = float(globalscalar_heldout) <= _DFAIR_GLOBALSCALAR_MAX
    pm_fail = float(permuted_heldout) <= _DFAIR_PERMUTED_MAX
    aligned = float(end_align_cos) >= _DFAIR_ALIGN_MIN
    gate = bool(learned and gs_fail and pm_fail and aligned)
    return {"GATE": "PASS" if gate else "FAIL",
            "instrument_valid": True,
            "task_learned": bool(learned),
            "globalscalar_fails": bool(gs_fail),
            "permuted_fails": bool(pm_fail),
            "emergent_alignment": bool(aligned),
            "oracle_heldout": float(oracle_heldout),
            "correct_heldout": float(correct_heldout),
            "wrongsign_heldout": float(wrongsign_heldout),
            "globalscalar_heldout": float(globalscalar_heldout),
            "permuted_heldout": float(permuted_heldout),
            "end_align_cos": float(end_align_cos),
            "bars": _bars()}


def _bars():
    return {"oracle_min": _DFAIR_ORACLE_MIN,
            "wrongsign_max": _DFAIR_WRONGSIGN_MAX,
            "correct_min": _DFAIR_CORRECT_MIN,
            "globalscalar_max": _DFAIR_GLOBALSCALAR_MAX,
            "permuted_max": _DFAIR_PERMUTED_MAX,
            "align_min": _DFAIR_ALIGN_MIN}


def dfair_aggregate_multiseed(per_seed, min_seeds=_DFAIR_MIN_SEEDS):
    n = len(per_seed)
    eff = max(int(min_seeds), _DFAIR_MIN_SEEDS)
    gates = [v.get("GATE") for v in per_seed]
    if n < eff or n == 0:
        return {"GATE": "FAIL", "n_seeds": n, "min_seeds": eff,
                "reason": "fewer than %d seeds" % eff}
    if any(g == "VOID" for g in gates):
        return {"GATE": "VOID", "n_seeds": n, "min_seeds": eff,
                "n_void": sum(g == "VOID" for g in gates),
                "reason": "instrument VOID in >=1 seed"}
    n_pass = sum(g == "PASS" for g in gates)
    return {"GATE": "PASS" if n_pass == n else "FAIL",
            "n_seeds": n, "min_seeds": eff, "n_pass": n_pass}
```

Run tests → pass. Commit `feat(Dendritic-fair): THREE-STATE FIXED-bar verdict (VOID-first instrument-validity; own frozen _DFAIR_*; adversarially pinned)`. Controller trust-but-verify + **DEDICATED ADVERSARIAL REVIEWER for Tasks 1+2** (charge: can a non-discriminating run score PASS instead of VOID? can wrong-sign-rescue slip to FAIL instead of VOID? can B be weight-transported / mutated? any autograd in the shipped path? truthy-string/ non-numeric bypass? does `oracle` genuinely descend? STRENGTHEN-only fixes; frozen bar VALUES byte-unchanged). Must APPROVE before Phase B.

---

## Phase B — integration (Tasks 3–4): import/signature smoke + the gate itself.

### Task 3: `research/runners/dendritic_fair_gate.py` — kill-safe/pausable runner

**Files:** Create `research/runners/dendritic_fair_gate.py`, `tests/test_dendritic_fair_gate_smoke.py`

Requirements (study `research/runners/scaled_subword_lm_train.py` for the kill-safe pattern + `research/runners/corpus_fetch.py` for the cache discipline; reuse `sim.train_checkpoint`):
- CLI: `--seeds 42,43,44`, `--tiny-synth` (tiny synthetic dataset + 1-epoch + tiny net so the pipeline turns in seconds — makes Task 0 green; toy verdict NOT propagated), `--epochs` (pre-registered default for the real run), `--out`, `--ckpt-dir`, `--mnist-cache data/mnist.npz`.
- `<3 seeds` → print `[NOT RUNNABLE] >=3 seeds MANDATORY` → `return 2`. MNIST cache absent **and** offline → `[NOT RUNNABLE] MNIST cache absent...` → `return 2`.
- MNIST: function `_load_mnist(cache, allow_download)` — if `data/mnist.npz` exists, load+verify (shapes (N,784) float in [0,1], labels 0..9); else download once from a public URL, cache as `.npz`, verify; record provenance + `degraded`/`source` in JSON (mirror `corpus_fetch`). `--tiny-synth` bypasses MNIST entirely with a deterministic synthetic 3-class set.
- Per `seed` in seeds, per `condition` in `("oracle","local_correct","local_wrongsign","global_scalar","permuted")`: build `DendriticMLP` (FIXED pre-registered sizes, e.g. `[784,512,256,128,10]`; `[24,16,16,3]` under `--tiny-synth`); train `--epochs` epochs (mini-batch); **per-epoch atomic checkpoint** via `sim.train_checkpoint.save_checkpoint(ckpt_path(seed,condition), epoch, weights=net.W+net.B, rng_state=..., loss_history=align_history+heldout_history)`; on (re)start `load_checkpoint`+`resume_epoch` → resume that (seed,condition) from last completed epoch (resume-stable: seed-deterministic data/init); `permuted` uses a fixed per-seed label permutation; record per-epoch heldout + `net.hidden_grad_alignment` curve. `try/except KeyboardInterrupt` → flush final checkpoint for the in-flight (seed,condition,epoch) → print clean message → `return 0` (resumable).
- After all: per seed, `dfair_verdict(oracle_heldout, correct_heldout, wrongsign_heldout, globalscalar_heldout, permuted_heldout, end_align_cos=<local_correct tail-mean alignment>, biologically_local=True, has_controls=True)`; `dfair_aggregate_multiseed`. `biologically_local` is asserted True only after a runtime self-check that no shipped module imported autograd and every `net.B` is unchanged vs its seed-regenerated value (no weight transport) — else pass `biologically_local=False` (→ VOID).
- Write JSON (`n_seeds`, `per_seed`:[{seed, per-condition heldout + curves, verdict}], `aggregate_verdict`, `GATE`, `mnist_provenance`, honest-ceiling string) + ASCII verdict block printing `VOID|PASS|FAIL` explicitly + the HONEST CEILING banner (literature-scale credit-assignment ROOT only; NOT #3; NOT conversation-solved; integration a separate later effort). Honest-propagation = CONTROLLER's job.

Smoke test (`tests/test_dendritic_fair_gate_smoke.py`): module imports; `main` callable; `--seeds 42,43 --tiny-synth` → exit 2 + "NOT RUNNABLE"; `--seeds 42,43,44 --tiny-synth` → exit 0, pipeline turns, 3-state verdict in JSON (makes Task 0 green). Commit `feat(Dendritic-fair): kill-safe/pausable runner (reuse sim.train_checkpoint + corpus_fetch cache discipline; 5 conditions; 3-state verdict; honest ceiling)`. Controller: ONLY 2 files; `git diff <range> -- <all protected>` EMPTY; assert no shipped module imports `torch`/`autograd`.

### Task 4: `tests/test_dendritic_fair_noharm.py` — LOAD-BEARING no-harm

Mirror `tests/test_generator_h_noharm.py`/`test_dendritic_noharm.py`: (1) `abstention_gate` byte-contract intact + `tests/test_abstention_gate.py` green (the distinctive moat); (2) importing the 3 net-new modules does NOT pull/mutate any frozen `*_core`; (3) `dendritic_fair_core` owns its frozen `_DFAIR_*` (exact values) and `dendritic_core._DEND_*` are untouched; (4) NO shipped `sim/dendritic_mlp.py` / `research/runners/dendritic_fair_*` source contains `torch` or `autograd`. Commit. Controller: whole-range `git diff <design-base>..HEAD -- <all protected>` EMPTY; representative validated suite green.

---

## Task 5: CONTROLLER-ONLY — decisive kill-safe run + MANDATORY smell-test + honest propagation

> NOT a subagent. The owner authorized this week-scale run (Option 2).

1. **Grounding-first tiny-synth:** `python -m research.runners.dendritic_fair_gate --seeds 42,43,44 --tiny-synth --out .../dfair.tiny.json --ckpt-dir .../dfair_tiny` → exit 0, pipeline turns, Task 0 pin green. Toy verdict NOT propagated.
2. **Decisive run (pre-registered FIXED config; pausable):** `python -m research.runners.dendritic_fair_gate --seeds 42,43,44 --epochs <pre-registered> --out research/findings/raw/g11_bg/dendritic_fair_gate.json --ckpt-dir research/findings/raw/g11_bg/dfair_ckpt` (kill-safe — if interrupted for GPU/time, simply re-invoke; it resumes per (seed,condition,epoch) with ≤1 epoch lost). Do NOT tune anything between resumes.
3. **MANDATORY anti-cheat smell-test (scrutinize a PASS HARDER than a FAIL; recompute from recorded JSON; NO re-run; NO bar-tuning):** verify V1 `oracle_heldout >= 0.95` per seed (instrument sound — else the whole run is VOID, honestly); V2 `wrongsign_heldout <= 0.30` per seed (instrument discriminates — else VOID); `global_scalar` and `permuted` genuinely ≤ 0.30 (controls fail); `biologically_local` genuinely True (every `B` byte-identical to its seed-regenerated value → no weight transport; no autograd imported); read the per-epoch curves — emergent alignment for `local_correct` genuinely INCREASES over training and ends ≥ 0.30 (the Lillicrap signature, not a fluke); `correct_heldout >= 0.90` is genuine learning not memorization (held-out). A nominal PASS with any of these not genuinely holding → report VOID/FAIL honestly.
4. **Honest propagation (EVERY outcome):** write `research/findings/2026-05-17-dendritic-fairscale-glr2017-{PASS|FAIL|VOID}.md` (verbatim per-seed numbers + curves + the explicit HONEST CEILING) + append a `webapp/capability_status.json` pillar (honest 3-state verdict; ceiling foregrounded; `as_of` bump); `python -m pytest tests/test_webapp_server.py -k capability_status -q` MUST stay green (fix JSON not test); `python -m pytest tests/test_abstention_gate.py -q` MUST be 7/7 (moat not regressed); commit + push BOTH remotes.
   - **PASS:** the #1 credit-assignment lever genuinely works biologically-locally at the literature's discriminating scale on real data — major decision-relevant; state explicitly it is NOT #3 / NOT conversation-solved; **integration into the conversational stack is a SEPARATE later effort (Arch C), noted not started.**
   - **FAIL:** strongest possible honest terminus (fails at the literature's own discriminating regime, locally).
   - **VOID:** instrument not soundly constructible even at fair scale — converges with the project-wide joint-infeasibility boundary; honest, not spun.
5. Bring Task 5's result back to the controller for the next decision point. **LATER increments (noted, NOT now):** Arch B (CIFAR-conv harder confirmation) and Arch C (dendritic rule → bio W→A integration) — conditional on a scrutinized PASS.

---

## Remember
- Exact paths; complete code in plan; TDD (fail→impl→pass→commit); frequent commits.
- DRY: committed `sim.dendritic_plasticity` + `sim.train_checkpoint` + `corpus_fetch`-discipline + `abstention_gate` + frozen cores byte-UNMODIFIED; `dendritic_fair_core` owns frozen `_DFAIR_*`; protected byte-empty in every commit-scoped diff.
- @superpowers:test-driven-development each task; @superpowers:subagent-driven-development drives execution; Tasks 1+2 get the dedicated adversarial reviewer before Phase B.
- ASCII-only; ≥3 seeds; THREE-STATE VOID/PASS/FAIL with V1/V2 instrument-validity FIRST; FIXED `_DFAIR_*` NEVER tuned; KILL-SAFE/PAUSABLE (≤1 epoch lost on interrupt); NO autograd in any shipped learning path (oracle = hand-derived numpy BP, measurement-only); honest ceiling stated up front and never spun (ROOT only; integration separate); the validated no-confab moat MUST stay byte-identical + 7/7 green.
```
