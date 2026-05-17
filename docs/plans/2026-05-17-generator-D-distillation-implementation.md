# Generator-D — Soft-Target Distillation into a Spiking LM — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task (continuous autonomous arc; do NOT stop to ask between tasks or increments — user authorized a week of autonomous work 2026-05-17).

**Goal:** Test, with the SAME hardened pre-registered multi-seed gate, whether the spiking SNN can absorb a competent teacher's DENSE soft next-token distribution (KL/dark-knowledge) where it catastrophically failed on a hard one-hot target (Generator-S: held-out ppl ~10^5, 200x worse than random).

**Architecture:** Net-new = a pure smoothed back-off trigram teacher (grounded: held-out ppl 14.3 vs random 513) + a pure soft-cross-entropy loss/grad + a DRY mirror of the validated `scaled_subword_lm_train` whose ONLY change is the loss (soft-xent to teacher dist vs one-hot CE) + a thin gate runner. Everything else (LIF BPTT, surrogate grad, kill-safe checkpoint, BPE, corpus fetch, generation, the HARDENED gate_core) is reused UNMODIFIED.

**Tech Stack:** Python, numpy (CPU tests), CuPy (RTX 3090 via `sim.bptt_snn_gpu`), stdlib only for the teacher (zero new deps).

**Validated APIs reused UNMODIFIED (DRY — do NOT reimplement):**
- `sim.bpe_tokenizer.BPETokenizer` (`.train/.encode/.decode/.vocab_size`).
- `sim.bptt_snn_gpu`: `LIFLayerXP`, `forward_unroll_xp`, `backward_unroll_xp`, `_get_backend`.
- `sim.bptt_snn.cross_entropy_loss_np(logits,target_idx)->float`, `softmax_grad_np(logits,target_idx)->ndarray` — used ONLY as the equivalence ORACLE in soft_xent tests.
- `sim.char_tokenizer.make_seq_dataset(text,tok,seq_len,n_samples,rng)` -> `(X (n,T,V) one-hot float32, y (n,T) int64)`.
- `sim.train_checkpoint.{save_checkpoint(path,epoch,weights,rng_state,loss_history), load_checkpoint, resume_epoch}` (atomic).
- `research.runners.scaled_subword_lm_train.train_subword_lm` — the loop SHAPE the distill trainer mirrors (init std 2.0 first / 0.5 later; threshold 1.0; leak 0.95; logits = `state["spikes"][-1].sum(axis=0)`; per-sample loss/grad; `backward_unroll_xp`; `layer.W_in -= lr*wg[li]`; kill-safe; OOM-halving; KeyboardInterrupt clean exit).
- `research.runners.corpus_fetch.{fetch_corpus,split_corpus}`, `research.runners.subword_lm_generate.generate`.
- `research.runners.subword_lm_gate_core` — **HARDENED, FROZEN**: bars `_GS_PPL_MARGIN=0.20`, `_GS_GENERALIZATION_MAX=1.5`, `_GS_DISTINCT_MIN=0.5`, `_GS_COPY_MAX=0.20`, `_GS_MIN_SEEDS=3`, `_GS_ABS_COMPETENCE_PPL_RATIO=1.0`; `gs_verdict(...,uniform_ppl=)` fail-closed without `uniform_ppl`; `gs_aggregate_multiseed`. Generator-D introduces **NO new bar** and does **NOT** modify gate_core.
- `research.runners.subword_lm_gate` — the orchestration SHAPE `generator_d_gate` mirrors (`_heldout_nll`, `_word_shuffle`, per-seed kill-safe `.resume.json`, ASCII verdict block).

**MUST NOT touch (LOAD-BEARING no-harm):** `sim/bridge.py`, `sim/*` validated modules, `research/runners/g20_*`, `song_g1_core.py`, `subword_lm_gate_core.py` (FROZEN bars), `order_intrinsic_*`, any shipped validated runner. Generator-D is PURELY ADDITIVE new files.

**Anti-cheat (non-negotiable):** the gate is the SAME hardened gate_core (absolute-competence floor is the exact bar Generator-S's noise model failed). The gate judges the **student's held-out ppl** (teacher gone). MANDATORY post-run smell-test: a nominal PASS is scrutinized HARDER than a FAIL — student held-out ppl sanity-checked vs uniform-random (513) BEFORE any propagation (the Generator-S false-PASS was caught exactly this way). Bars NEVER tuned; a false PASS is propagated as the honest NEGATIVE recomputed from recorded data. FAIL ⇒ honest propagation + immediately the pre-staged Generator-E — NOT a config-crank, NOT a stop.

**ASCII-only prints (Windows cp1252). Commit after every task. Push both remotes (`origin`,`gitea`) after each phase.**

---

### Task 0: Falsify-cheaply grounding pin (zero network, before ANY scaled/corpus work)

**Files:** Create `tests/test_generator_d_grounding.py`

**Step 1: Write the test**

```python
"""Grounding: the distill plumbing TURNS end-to-end on local
shakespeare (zero network) -- a tiny spiking student trained by
soft-xent against the trigram teacher REDUCES loss. If this regresses,
STOP (systematic-debugging) -- do not run the scaled/corpus gate on a
broken pipeline."""
import os
import pytest


def test_distill_plumbing_reduces_loss_local_shakespeare(tmp_path):
    if not os.path.exists("data/tinyshakespeare.txt"):
        pytest.skip("local grounding corpus absent")
    from research.runners.distill_subword_lm_train import (
        train_distill_subword_lm)
    r = train_distill_subword_lm(
        seed=42, corpus_path="data/tinyshakespeare.txt",
        vocab_size=64, hidden_layers=[32], T=12, epochs=3,
        batch_size=8, n_train_samples=40,
        ckpt_path=str(tmp_path / "d.ckpt.npz"),
        bpe_path=str(tmp_path / "d.bpe.json"),
        backend="cpu", verbose=False)
    assert r["final_loss"] is not None
    assert r["final_loss"] <= r["initial_loss"], (
        "distill plumbing does not reduce loss -- STOP, root-cause "
        "before any scaled/corpus run")
    assert r["vocab_size"] > 1 and r["n_layers"] == 2
```

**Step 2:** Run `pytest tests/test_generator_d_grounding.py -q` → expect FAIL (module missing — Task 3 not built yet). This pin is committed now but goes GREEN only after Task 3; that is intentional (it is the Task-3 gate). Mark xfail-until-Task-3 by committing it and noting it in the Task 3 step.

**Step 3:** (no impl yet)

**Step 5: Commit**
```bash
git add tests/test_generator_d_grounding.py
git commit -m "test(Generator-D): falsify-cheaply grounding pin (distill plumbing reduces loss; zero network) -- green after Task 3"
```

---

## PHASE A — pure-logic CPU-TDD

### Task 1: Pure smoothed back-off trigram teacher

**Files:** Create `sim/ngram_teacher.py`; Test `tests/test_ngram_teacher.py`

**Step 1: Write the failing tests**

```python
import numpy as np
from sim.ngram_teacher import NgramTeacher

def test_soft_dist_is_a_valid_distribution():
    ids = [1,2,3,1,2,3,1,2,4,2,3,1] * 20
    t = NgramTeacher(); t.train(ids, vocab_size=8)
    q = t.soft_dist((1, 2))
    assert q.shape == (8,)
    assert abs(float(q.sum()) - 1.0) < 1e-9
    assert (q >= 0).all()

def test_beats_uniform_on_structured_corpus():
    import math
    ids = ([1,2,3] * 400)                 # perfectly predictable
    V = 6
    t = NgramTeacher(); t.train(ids, vocab_size=V)
    # held-out continues the same pattern; teacher NLL << ln V (uniform)
    held = [1,2,3] * 50
    nll = []
    for i in range(2, len(held)):
        p = float(t.soft_dist((held[i-2], held[i-1]))[held[i]])
        nll.append(-math.log(max(p, 1e-12)))
    ppl = math.exp(sum(nll)/len(nll))
    assert ppl < V                        # decisively beats uniform

def test_deterministic():
    ids = [3,1,4,1,5,9,2,6,5,3,5] * 30
    a = NgramTeacher(); a.train(ids, 12)
    b = NgramTeacher(); b.train(ids, 12)
    assert np.array_equal(a.soft_dist((1,5)), b.soft_dist((1,5)))

def test_backoff_and_short_context_safe():
    ids = [1,2,1,2,1,2] * 10
    t = NgramTeacher(); t.train(ids, vocab_size=5)
    # unseen trigram ctx -> backs off, still a valid distribution
    q = t.soft_dist((4, 4))
    assert abs(float(q.sum()) - 1.0) < 1e-9
    # short / empty context safe
    assert t.soft_dist(()).shape == (5,)
    assert t.soft_dist((1,)).shape == (5,)
    assert abs(float(t.soft_dist(()).sum()) - 1.0) < 1e-9
```

**Step 2:** `pytest tests/test_ngram_teacher.py -q` → FAIL (module missing).

**Step 3: Implement** `sim/ngram_teacher.py` (EXACT algorithm = grounded probe `ba1jyepwf`, ppl 14.3):

```python
"""Pure smoothed back-off trigram LM over BPE token ids -- the
Generator-D distillation teacher. Stdlib Counter + numpy ONLY (zero
new deps, zero external weights; a statistical model of the
user-authorized corpus -> in-constraints, training-time only).
Grounded competent: held-out ppl 14.3 vs uniform-random 513 on
TinyStories (probe ba1jyepwf). Deterministic. soft_dist returns the
dense soft target (length-V probability vector summing to 1)."""
from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np


class NgramTeacher:
    def __init__(self):
        self._uni = Counter()
        self._bi = defaultdict(Counter)
        self._trg = defaultdict(Counter)
        self._V = 0
        self._k = 0.1

    def train(self, train_ids, vocab_size: int, k: float = 0.1) -> None:
        self._V = int(vocab_size)
        self._k = float(k)
        ti = list(train_ids)
        self._uni = Counter(ti)
        self._bi = defaultdict(Counter)
        self._trg = defaultdict(Counter)
        for i in range(len(ti) - 1):
            self._bi[ti[i]][ti[i + 1]] += 1
        for i in range(len(ti) - 2):
            self._trg[(ti[i], ti[i + 1])][ti[i + 2]] += 1

    def soft_dist(self, ctx) -> np.ndarray:
        """Dense length-V soft target. Back-off: trigram if its ctx
        count >= 5, else bigram if >= 2, else unigram; add-k smoothed
        over the FULL vocab so every entry is > 0 and the vector sums
        to 1. ctx may be (), (a,) or (a,b)."""
        V = self._V
        k = self._k
        ctx = tuple(ctx)
        counts = None
        if len(ctx) >= 2:
            c3 = self._trg.get((ctx[-2], ctx[-1]))
            if c3 is not None and sum(c3.values()) >= 5:
                counts = c3
        if counts is None and len(ctx) >= 1:
            c2 = self._bi.get(ctx[-1])
            if c2 is not None and sum(c2.values()) >= 2:
                counts = c2
        if counts is None:
            counts = self._uni
        tot = sum(counts.values()) + k * V
        q = np.full(V, k / tot, dtype=np.float64)
        for w, c in counts.items():
            if 0 <= w < V:
                q[w] = (c + k) / tot
        s = q.sum()
        if s > 0:
            q = q / s                       # exact-normalize (defensive)
        return q
```

**Step 4:** `pytest tests/test_ngram_teacher.py -q` → all 4 PASS. Root-cause any failure WITHOUT weakening a test or touching another file.

**Step 5: Commit**
```bash
git add sim/ngram_teacher.py tests/test_ngram_teacher.py
git commit -m "feat(Generator-D): pure smoothed back-off trigram teacher (grounded ppl 14.3; zero-dep; in-constraints)"
```

---

### Task 2: Pure soft cross-entropy loss + gradient (LOAD-BEARING — a wrong distill grad silently invalidates the verdict)

**Files:** Create `sim/soft_xent.py`; Test `tests/test_soft_xent.py`

**Step 1: Write the failing tests** (incl. the load-bearing adversarial oracle checks)

```python
import numpy as np
from sim.soft_xent import soft_xent_loss, soft_xent_grad
from sim.bptt_snn import cross_entropy_loss_np, softmax_grad_np

def test_equals_hard_CE_when_q_is_one_hot():
    # THE load-bearing equivalence: soft-xent with a one-hot q must
    # equal the validated hard cross_entropy_loss_np / softmax_grad_np
    # (proves it is a faithful generalization, not a different metric).
    rng = np.random.default_rng(0)
    for _ in range(5):
        logits = rng.normal(0, 3, (1, 7)).astype(np.float64)
        tgt = int(rng.integers(0, 7))
        q = np.zeros(7); q[tgt] = 1.0
        assert abs(soft_xent_loss(logits, q)
                   - cross_entropy_loss_np(logits, tgt)) < 1e-6
        assert np.allclose(soft_xent_grad(logits, q),
                           softmax_grad_np(logits, tgt), atol=1e-6)

def test_grad_is_finite_difference_correct():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 1, (1, 5)).astype(np.float64)
    q = rng.random(5); q = q / q.sum()
    g = soft_xent_grad(logits, q)
    eps = 1e-5
    for j in range(5):
        lp = logits.copy(); lp[0, j] += eps
        lm = logits.copy(); lm[0, j] -= eps
        fd = (soft_xent_loss(lp, q) - soft_xent_loss(lm, q)) / (2*eps)
        assert abs(fd - g[0, j]) < 1e-4

def test_loss_nonnegative_and_minimized_at_match():
    q = np.array([0.1, 0.7, 0.2])
    # logits proportional to log q -> softmax == q -> loss == entropy(q)
    near = soft_xent_loss(np.log(q).reshape(1, 3), q)
    far = soft_xent_loss(np.array([[5.0, -5.0, 0.0]]), q)
    assert near <= far and near >= 0.0

def test_renormalizes_and_handles_nonfinite_without_crash():
    logits = np.array([[1.0, 2.0, 3.0]])
    q_bad = np.array([2.0, 2.0, 4.0])               # sums to 8, not 1
    L = soft_xent_loss(logits, q_bad)               # must renormalize
    assert np.isfinite(L)
    g = soft_xent_grad(logits, q_bad)
    assert np.isfinite(g).all() and g.shape == (1, 3)
    # garbage logits do not crash (numerically stable)
    assert np.isfinite(soft_xent_loss(np.array([[1e9, -1e9, 0.0]]),
                                      np.array([0.3,0.3,0.4])))
```

**Step 2:** `pytest tests/test_soft_xent.py -q` → FAIL (module missing).

**Step 3: Implement** `sim/soft_xent.py`:

```python
"""Pure soft cross-entropy loss + gradient -- the ONLY change vs
Generator-S's one-hot CE. soft_xent_loss = -sum_w q_w log softmax(z)_w;
soft_xent_grad = softmax(z) - q (exact d/dz of soft-xent). Numerically
stable (log-sum-exp), mirroring sim.bptt_snn.cross_entropy_loss_np /
softmax_grad_np, of which this is the faithful generalization (equal
when q is one-hot). Pure numpy; CPU-unit-testable."""
from __future__ import annotations
import numpy as np


def _log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=-1, keepdims=True)
    zs = z - m
    return zs - np.log(np.exp(zs).sum(axis=-1, keepdims=True))


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=-1, keepdims=True)


def _norm_q(q, V):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    q = np.where(np.isfinite(q), q, 0.0)
    q = np.clip(q, 0.0, None)
    s = q.sum()
    if s <= 0:
        return np.full(V, 1.0 / V)          # degenerate -> uniform
    return q / s


def soft_xent_loss(logits, q) -> float:
    """logits (1,V), q (V,). Returns mean over the (size-1) batch the
    soft cross-entropy -sum_w q_w log softmax(logits)_w (batch-mean to
    match cross_entropy_loss_np's contract)."""
    lg = np.asarray(logits, dtype=np.float64)
    V = lg.shape[-1]
    qn = _norm_q(q, V)
    ls = _log_softmax(lg)                    # (1,V)
    return float(-(qn * ls).sum(axis=-1).mean())


def soft_xent_grad(logits, q) -> np.ndarray:
    """d/dlogits of soft_xent_loss = (softmax(logits) - q) / batch.
    Shape (1,V), matching softmax_grad_np's batch-mean convention."""
    lg = np.asarray(logits, dtype=np.float64)
    B, V = lg.shape
    qn = _norm_q(q, V)
    return (_softmax(lg) - qn.reshape(1, V)) / B
```

**Step 4:** `pytest tests/test_soft_xent.py -q` → all 4 PASS (esp. the hard-CE equivalence + FD-gradient). Root-cause failures WITHOUT weakening; if a genuine spec contradiction, STOP and report (do NOT fake-pass — a wrong gradient here silently invalidates the decisive verdict).

**Step 5: Commit**
```bash
git add sim/soft_xent.py tests/test_soft_xent.py
git commit -m "feat(Generator-D): pure soft-xent loss+grad (faithful generalization of validated hard CE; FD-verified)"
```

**End of Phase A:** `git push origin HEAD && git push gitea HEAD`. Dispatch a RIGOROUS spec+quality review of Task 2 (the soft-xent grad is load-bearing — the hard-CE-equivalence + FD-correctness are the anti-cheat properties; review them adversarially) before Phase B.

---

## PHASE B — integration (import/signature smoke + the gate itself; project pattern)

### Task 3: Distillation trainer (DRY mirror of validated `scaled_subword_lm_train`)

**Files:** Create `research/runners/distill_subword_lm_train.py`; Test `tests/test_distill_subword_lm_train_smoke.py`

**Reference:** `train_distill_subword_lm(...)` is a byte-mirror of `research/runners/scaled_subword_lm_train.py:train_subword_lm` (read that file; copy it verbatim) with EXACTLY these changes and NOTHING else:
- import add: `from sim.ngram_teacher import NgramTeacher` and `from sim.soft_xent import soft_xent_loss, soft_xent_grad`; remove the `from sim.bptt_snn import cross_entropy_loss_np, softmax_grad_np` usage (keep import only if still referenced — it is NOT after this change).
- after `X_np, y_np = make_seq_dataset(...)`: add
  ```python
  X_ids = X_np.argmax(axis=2)                      # (n,T) input ids
  teacher = NgramTeacher()
  teacher.train(list(tok.encode(corpus)), vocab_size=V)
  ```
- the per-sample loss/grad block becomes (T is the unroll len; the
  next-token context is the window's last two ids):
  ```python
  og = np.zeros((T, B, V), dtype=np.float32)
  bl = 0.0
  for s in range(B):
      gi = start + s                               # global sample idx
      ctx = (int(X_ids[gi, T-2]) if T >= 2 else
             int(X_ids[gi, T-1]),
             int(X_ids[gi, T-1]))
      q = teacher.soft_dist(ctx)                    # dense soft target
      bl += soft_xent_loss(logits_np[s:s+1], q)
      og[:, s, :] = soft_xent_grad(logits_np[s:s+1], q)[0]
  ```
  (NOTE: use the SHUFFLED-permuted global index consistently — mirror
  exactly how `scaled_subword_lm_train` indexes `Xs/ys`; the teacher
  ctx must come from the SAME permuted window as `logits`. Recover ids
  from the permuted batch: compute `Xb_ids = (Xs[start:end] if not
  is_gpu else Xs[start:end].get()).argmax(axis=2)` and use
  `Xb_ids[s, T-2], Xb_ids[s, T-1]` — this guarantees ctx aligns with
  the logits regardless of backend/shuffle. Implement it this
  backend-safe way.)
- function name `train_distill_subword_lm`; returns dict adds
  `"_teacher": teacher` (so the gate can record teacher held-out ppl
  for transparency); everything else (kill-safe resume via
  `sim.train_checkpoint`, OOM-halving, KeyboardInterrupt clean exit,
  `_layers`/`_tok`, ASCII prints, init std/threshold/leak,
  forward/backward unroll, SGD) IDENTICAL to the validated mirror.
- `main()` argparse identical to `scaled_subword_lm_train`'s (same
  flags) so it is CLI-runnable.

**Smoke test** `tests/test_distill_subword_lm_train_smoke.py`:

```python
import inspect
from research.runners.distill_subword_lm_train import (
    train_distill_subword_lm)

def test_signature():
    p = inspect.signature(train_distill_subword_lm).parameters
    for k in ("seed","corpus_path","vocab_size","hidden_layers","T",
              "epochs","batch_size","lr","n_train_samples",
              "ckpt_path","bpe_path","backend"):
        assert k in p

def test_tiny_cpu_distill_reduces_loss_and_is_resumable(tmp_path):
    ck = str(tmp_path/"d.ckpt.npz"); bp = str(tmp_path/"d.bpe.json")
    r = train_distill_subword_lm(
        seed=42, corpus_path="data/tinyshakespeare.txt",
        vocab_size=64, hidden_layers=[32], T=12, epochs=3,
        batch_size=8, n_train_samples=32, ckpt_path=ck, bpe_path=bp,
        backend="cpu", verbose=False)
    assert r["final_loss"] is not None
    assert r["final_loss"] <= r["initial_loss"]
    assert r["n_layers"] == 2 and "_teacher" in r
    import os
    from sim.train_checkpoint import load_checkpoint, resume_epoch
    assert os.path.exists(ck)
    assert resume_epoch(load_checkpoint(ck)) == 3
    r2 = train_distill_subword_lm(
        seed=42, corpus_path="data/tinyshakespeare.txt",
        vocab_size=64, hidden_layers=[32], T=12, epochs=5,
        batch_size=8, n_train_samples=32, ckpt_path=ck, bpe_path=bp,
        backend="cpu", verbose=False)
    assert len(r2["loss_history"]) == 5            # resumed 3 -> 4,5
```

**Procedure:** TDD (smoke fails → mirror impl → smoke passes); then run the Task-0 grounding pin (`pytest tests/test_generator_d_grounding.py -q` → now GREEN). Verify `git status --porcelain` shows ZERO modifications to `scaled_subword_lm_train.py`/`bptt_snn*`/`train_checkpoint`/`bpe_tokenizer`/`ngram_teacher`/`soft_xent` (reused by import only). Commit:
```bash
git add research/runners/distill_subword_lm_train.py tests/test_distill_subword_lm_train_smoke.py
git commit -m "feat(Generator-D): distillation trainer (DRY mirror of validated scaled_subword_lm_train; ONLY the loss swapped to soft-xent-to-teacher; kill-safe)"
```

---

### Task 4: Generator-D gate runner (DRY mirror of `subword_lm_gate`; passes uniform_ppl=V)

**Files:** Create `research/runners/generator_d_gate.py`; Test `tests/test_generator_d_gate_smoke.py`

**Reference:** byte-mirror `research/runners/subword_lm_gate.py` (read it; copy verbatim) with EXACTLY these changes:
- import `from research.runners.distill_subword_lm_train import train_distill_subword_lm` instead of `train_subword_lm`; call it for BOTH the real model and the word-shuffle control model (the control's teacher is the trigram trained on the word-shuffled corpus inside the trainer — honest: same dense-target mechanism, only sequence order destroyed).
- the `gs_verdict(...)` call MUST pass `uniform_ppl=rtok.vocab_size` (the hardened gate_core is fail-closed without it — this is the flagged controller follow-up; `rtok` is the real student's tokenizer, vocab == V).
- add to each `per_seed_records` entry `"teacher_heldout_ppl": <perplexity(_heldout_nll(teacher-as-dist...))>` — compute the teacher's OWN held-out ppl for TRANSPARENCY only (NOT a gate input). Simplest: a local `_teacher_heldout_ppl(teacher, tok, heldout_text, eval_positions)` using `perplexity` over `-log teacher.soft_dist(ctx)[true]`. Record it; the gate verdict uses ONLY the student metrics.
- banner/JSON strings say "Generator-D distillation"; `--out`/`--ckpt` default to `generator_d_gate.json` / `generator_d_gate.ckpt` (isolated namespace). `<3 seeds -> exit 2` retained. Kill-safe `.resume.json` retained. Honest-propagation remains the CONTROLLER's post-run job (same contract).

**Smoke** `tests/test_generator_d_gate_smoke.py`:

```python
import subprocess, sys

def test_import_and_passes_uniform_ppl():
    import research.runners.generator_d_gate as g
    import inspect
    src = inspect.getsource(g)
    assert "uniform_ppl=" in src           # MUST pass the floor baseline
    assert "song_g1_core" not in src       # no g1 ref
    assert "_GS_" not in src or "subword_lm_gate_core" in src  # no bar redef

def test_fewer_than_3_seeds_exit_2():
    r = subprocess.run([sys.executable,"-m",
        "research.runners.generator_d_gate","--seeds","42,43"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "NOT RUNNABLE" in r.stdout

def test_help():
    r = subprocess.run([sys.executable,"-m",
        "research.runners.generator_d_gate","--help"],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
```

**Procedure:** TDD smoke → mirror impl → smoke passes. Verify additive + `subword_lm_gate_core.py`/`song_g1_core.py` byte-UNTOUCHED + it passes `uniform_ppl=`. Commit:
```bash
git add research/runners/generator_d_gate.py tests/test_generator_d_gate_smoke.py
git commit -m "feat(Generator-D): gate runner (DRY mirror of subword_lm_gate; distill trainer; passes uniform_ppl=V to HARDENED gate_core; kill-safe)"
```

---

### Task 5: LOAD-BEARING no-harm

**Files:** Create `tests/test_generator_d_noharm.py`

```python
"""LOAD-BEARING no-harm: Generator-D is PURELY ADDITIVE; the validated
deliverable + the FROZEN gate_core bars are byte-untouched."""
import sys

def test_gate_core_bars_frozen_and_g1_untouched():
    import research.runners.subword_lm_gate_core as g
    assert (g._GS_PPL_MARGIN, g._GS_GENERALIZATION_MAX,
            g._GS_DISTINCT_MIN, g._GS_COPY_MAX, g._GS_MIN_SEEDS,
            g._GS_ABS_COMPETENCE_PPL_RATIO) == (0.20,1.5,0.5,0.20,3,1.0)
    assert not hasattr(g, "_G1_MARGIN")

def test_generator_d_does_not_pull_song_g1_core():
    import research.runners.distill_subword_lm_train  # noqa
    import sim.ngram_teacher, sim.soft_xent           # noqa
    assert "research.runners.song_g1_core" not in sys.modules
```

Plus the controller verifies (Task 5 step): full existing suite green
(`pytest tests/ -q` representative core subset incl.
`test_subword_lm_gate_core test_order_intrinsic_core test_webapp_server -k capability_status`) and `git diff --stat <gen-D-range> -- sim/bridge.py research/runners/g20_*.py research/runners/song_g1_core.py research/runners/subword_lm_gate_core.py research/runners/scaled_subword_lm_train.py sim/bptt_snn*.py sim/train_checkpoint.py` is EMPTY.

**Commit + push both remotes.** Dispatch spec+quality review of Phase B.

---

### Task 6: Decisive multi-seed run + honest propagation (CONTROLLER, not a subagent)

1. **Grounding-first (falsify-cheaply):** run `generator_d_gate` on LOCAL shakespeare (zero network), 3 seeds, tiny config — prove the full distill→control→heldout→verdict pipeline turns end-to-end + interpretable. Toy verdict NOT propagated. If broken → systematic-debugging.
2. **Measure GPU per-epoch cost** at the decisive per-model size (pre-data feasibility check; do NOT resize toward a pass — only toward feasibility, documented, before any decisive result).
3. **Decisive run:** cached TinyStories, FIXED pre-registered config (vocab 512, hidden 256,256, T 32, 40 epochs, batch 32, 2000 samples, seeds 42,43,44), kill-safe `run_in_background` (user games/resumes; do parallel pre-staging of Generator-E while it trains).
4. **MANDATORY anti-cheat smell-test BEFORE propagating:** scrutinize a nominal PASS HARDER than a FAIL — recompute from the recorded JSON: is the STUDENT's held-out ppl actually below uniform-random (513) and meaningfully coherent, or is it (like Generator-S) astronomically bad with vacuously-satisfied relative bars? The hardened gate_core's absolute-competence floor should now catch that automatically, but VERIFY from recorded numbers (no re-run, no bar-tuning-toward-pass). A false PASS is propagated as the honest NEGATIVE.
5. **Honest propagation EITHER way:** findings doc `research/findings/2026-05-17-generator-D-distillation-<PASS|NEGATIVE>.md` (honest mechanism, no overclaim, bars echoed/untouched, corpus + degraded flag, teacher ppl for transparency, student metrics decide); `webapp/capability_status.json` pillar (VALIDATED if a scrutinized real PASS else NEGATIVE; schema `{name,status,metric}`); `pytest tests/test_webapp_server.py -k capability_status` 6/6 green; commit + push BOTH remotes.
6. **Continuous arc — no stop/ask/config-crank:** PASS ⇒ design Generator-C (distill-pretrained spiking cortex onto the validated grounded-memory arch). FAIL ⇒ immediately Generator-E (pre-staged: (i) continuous teacher-hidden-vector regression target — even denser; (ii) a non-spiking catalog-grounded sequence substrate, e.g. echo-state reservoir readout, to test whether the spiking constraint itself is the bottleneck) — new design doc → writing-plans → subagent-driven-development, same hardened gate_core.

---

## Notes
- DRY: validated BPTT core / `scaled_subword_lm_train` loop-shape / `train_checkpoint` / `make_seq_dataset` / hardened `gate_core` reused UNMODIFIED. NO new bar. `song_g1_core` UNTOUCHED.
- YAGNI: cheap decisive slice. Generator-C + open-weights-teacher variant are later increments (noted, not built).
- TDD: pure logic (Tasks 1,2) failing-test→impl→commit; integration (3,4) import/signature smoke + the gate itself.
- @superpowers:systematic-debugging if the grounding pipeline breaks (root cause first; never paper over to reach a verdict).
- @superpowers:subagent-driven-development for execution; trust-but-verify each subagent's `git diff`; protected modules byte-empty in each commit-scoped diff.
- The Generator-S lesson is mandatory here: scrutinize a nominal PASS harder than a FAIL; a noise/gate-hole artifact is the honest NEGATIVE.
