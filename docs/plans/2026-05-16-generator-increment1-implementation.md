---
type: plan
status: live
date: 2026-05-16
---

# Self-Contained Generator — Increment 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (user authorized autonomous design→plan→implement). Increment 1 only. No network. No LLM. No templates.

**Goal:** Bring the project's own Phase-2 surrogate-grad BPTT sequence-generator infra from `path-f-hybrid` onto `main`, prove it still works on current code, and reproduce a real local training loss-reduction on a local corpus (the foundation everything else builds on).

**Architecture:** Pure cross-branch port via `git checkout path-f-hybrid -- <paths>` (no rewrite — the files + their tests are the spec). Verify by running the ported tests on `main`, then a short local training run with a falsifiable anti-cheat gate (loss reduction ≫ permuted-corpus control).

**Tech Stack:** existing CuPy/numpy backend, the ported `sim/bptt_snn*.py` / `surrogate_grad.py` / `char_tokenizer.py`, `research/runners/cortex_pretraining.py`. RTX 3090 local. Zero network.

---

### Task 1: Port Phase-2 generator infra to `main`

**Files (port verbatim from `path-f-hybrid`):**
- `sim/bptt_snn.py`, `sim/bptt_snn_gpu.py`, `sim/surrogate_grad.py`, `sim/char_tokenizer.py`
- `research/runners/cortex_pretraining.py`
- `tests/test_bptt_snn.py`, `tests/test_bptt_snn_gpu.py`, `tests/test_surrogate_grad.py`, `tests/test_char_tokenizer.py`

**Step 1: Port** (exact command)
```bash
git checkout path-f-hybrid -- sim/bptt_snn.py sim/bptt_snn_gpu.py sim/surrogate_grad.py sim/char_tokenizer.py research/runners/cortex_pretraining.py tests/test_bptt_snn.py tests/test_bptt_snn_gpu.py tests/test_surrogate_grad.py tests/test_char_tokenizer.py
```

**Step 2: Run the ported tests (they ARE the spec — must pass on main)**
```bash
python -m pytest tests/test_surrogate_grad.py tests/test_char_tokenizer.py tests/test_bptt_snn.py tests/test_bptt_snn_gpu.py -q
```
Expected: all PASS. If a test fails due to a `main`-vs-`path-f-hybrid` API drift (e.g. `sim.backend` import shape), fix the *ported file's* import to match `main`'s current API (minimal, no behavior change); re-run until green. Do NOT weaken a test to pass.

**Step 3: Commit**
```bash
git add sim/bptt_snn.py sim/bptt_snn_gpu.py sim/surrogate_grad.py sim/char_tokenizer.py research/runners/cortex_pretraining.py tests/test_bptt_snn.py tests/test_bptt_snn_gpu.py tests/test_surrogate_grad.py tests/test_char_tokenizer.py
git commit -m "feat(generator): port Phase-2 surrogate-grad BPTT infra to main (Increment 1)"
```

---

### Task 2: Local corpus availability (no network)

**Files:** Create `research/runners/local_corpus.py`; Test `tests/test_local_corpus.py`

The generator needs raw local text. Tiny-Shakespeare raw text may not be committed (only the trained `.npz`). Provide a deterministic local corpus loader with a zero-download fallback to the repo's own English text.

**Step 1: Failing test**
```python
from research.runners.local_corpus import load_local_corpus
def test_returns_nonempty_text():
    txt = load_local_corpus()
    assert isinstance(txt, str) and len(txt) > 50_000  # substantial
def test_deterministic():
    assert load_local_corpus() == load_local_corpus()
def test_no_network_used(monkeypatch):
    import socket
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    load_local_corpus()  # must not touch network
```

**Step 2:** `pytest tests/test_local_corpus.py -q` → FAIL.

**Step 3: Implement** — `load_local_corpus()`: if a committed local Shakespeare/text file exists (search `research/datasets/`, `path-f-hybrid` may have shipped one — check `git show path-f-hybrid:research/datasets/` style), use it; ELSE deterministically concatenate the repo's own `research/findings/*.md` (sorted by name) into one text blob (≈296K words, fully local, zero-download). Pure file I/O, no network.

**Step 4:** `pytest -q` → PASS (3).

**Step 5: Commit** `feat(generator): local zero-download corpus loader (Increment 1)`

---

### Task 3: Reproduce a local training loss-reduction with an anti-cheat gate

**Files:** Create `research/runners/generator_baseline_smoke.py`

Short local training run of the ported generator on the local corpus + a **permuted-corpus control** (anti-cheat: shuffle the corpus characters, train identically; real corpus must beat it decisively — proves it learns real sequence structure, not memorization artifacts).

**Step 1: Build the smoke** — load corpus (Task 2) + a permuted copy; tokenize via `char_tokenizer`; train the ported BPTT net (small config, ~a few hundred steps, minutes on the 3090) on each; report start/end held-out loss + perplexity for REAL vs PERMUTED.

**Step 2: Run**
```bash
python -m research.runners.generator_baseline_smoke --steps 300 --out research/findings/raw/g11_bg/generator_baseline.json
```

**Gate (falsifiable, anti-cheat):**
- REAL corpus: end loss ≪ start loss (substantial reduction).
- REAL end-loss ≪ PERMUTED end-loss (real structure learned, not just fitting noise).
- If the gate PASSES → the sim's own generator demonstrably learns real local text on `main`. Foundation established.
- If it FAILS → honest finding (port broke something / infra regressed); document, do not paper over.

**Step 3: Commit** `feat(generator): local baseline smoke + permuted anti-cheat gate (Increment 1)`

---

### Task 4: Propagate (honest)

**Files:** Create `research/findings/2026-05-16-generator-increment1-foundation.md`; Modify `webapp/capability_status.json` (pillar: status VALIDATED iff Task-3 gate passed, else BOUNDARY/NEGATIVE honestly). Run `pytest tests/test_webapp_server.py -k capability_status -q`.

Findings doc states plainly: this is ONLY the foundation (the sim's own generator learns real local text on main, anti-cheat-gated) — NOT yet conversational, NOT yet distilled, NOT yet grounded. Honest next = Increment 2 (distillation teacher). Commit + push both remotes.

---

## Notes for the executor

- **No network, no LLM, no templates anywhere in Increment 1.** This increment is purely "does the sim's own generator infra work on main + learn real local text."
- The ported tests are the spec — if they fail, fix the *ported code's* import/API drift to match `main`, never weaken the test.
- Anti-cheat gate (Task 3 permuted control) is load-bearing — it proves real sequence learning, mirroring the project's permuted-label discipline. A failed gate is an honest finding.
- DRY: reuse `sim.backend` (`get_backend`/`fuse`) as `main` exposes it; the ported GPU file may need its import block aligned to `main`'s backend API (minimal).
- Keep increments honest and small; Increment 1 ships a *foundation*, not a conversational system. No overclaiming in the findings doc or capability_status.
