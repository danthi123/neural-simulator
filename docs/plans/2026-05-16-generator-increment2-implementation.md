---
type: plan
status: live
date: 2026-05-16
---

# Self-Contained Generator — Increment 2 (data distillation) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (user authorized autonomous design→plan→implement). Increment 2 only. Data/sequence-level distillation (Kim & Rush 2016) — NOT logit distillation. Teacher = training-time ONLY, code-asserted out of the self-contained runtime.

**Goal:** Show that training the project's own char-level spiking generator on a *teacher-generated* corpus beats the Increment-1 same-architecture baseline — a real, anti-cheat-controlled lift that lives entirely in the student's own self-contained weights.

**Architecture:** A training-time-only local Qwen2.5-0.5B-Instruct samples clean English text → cached corpus → the SAME Increment-1 student (`cortex_pretraining.train_shakespeare`) trains on it → controlled gate: distilled-student vs Inc-1-baseline-student vs permuted, identical config, **teacher absent at eval**. Reuses Increment-1 (`local_corpus`, `generator_baseline_smoke` pattern). Purely additive on `main`.

**Tech Stack:** local `transformers`+`torch` (RTX 3090, weights already cached), Increment-1 BPTT generator, pytest.

---

### Task 1: Training-time-only teacher text generator (runtime-isolated)

**Files:** Create `research/runners/distill_teacher.py`; Test `tests/test_distill_teacher.py`

**Step 1: Failing test (pure/guards — no model load):**
```python
import importlib, pathlib, re, pytest
from research.runners import distill_teacher as dt

def test_offline_enforced_constant():
    # module must force transformers offline (no re-fetch at gen time)
    assert dt.LOCAL_FILES_ONLY is True

def test_runtime_isolation_guard_exists():
    assert hasattr(dt, "assert_training_time_only")

def test_no_runtime_module_imports_teacher():
    # the self-contained runtime/honesty path must NEVER import the teacher
    root = pathlib.Path(__file__).resolve().parents[1]
    runtime = [
        root/"research/runners/grounded_generative_agent.py",
        root/"research/runners/g20_generative_agent.py",
        root/"research/runners/abstention_gate.py",
        root/"research/runners/concept_grammar.py",
    ]
    for f in runtime:
        if f.exists():
            assert "distill_teacher" not in f.read_text(encoding="utf-8"), f
```

**Step 2:** `pytest tests/test_distill_teacher.py -q` → FAIL (module missing).

**Step 3: Minimal implementation**
```python
"""TRAINING-TIME ONLY local teacher (Qwen2.5-0.5B-Instruct) for
sequence-level / data distillation (Kim & Rush 2016). Generates clean
English text the student trains on. NEVER imported by the
self-contained runtime path (test-enforced). Offline after the
one-time cached fetch."""
from __future__ import annotations
import os
LOCAL_FILES_ONLY = True
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

def assert_training_time_only() -> None:
    """Loud marker; the runtime path must never call this."""
    return None

def generate_corpus(n_passages: int = 200, max_new_tokens: int = 160,
                     seed: int = 42) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert_training_time_only()
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=LOCAL_FILES_ONLY)
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL, local_files_only=LOCAL_FILES_ONLY,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu")
    prompts = [
        "Write a short, simple paragraph of plain English prose.",
        "Tell a brief everyday story in simple sentences.",
        "Describe a common object in a few clear sentences.",
        "Explain a simple idea in plain language.",
    ]
    out = []
    for i in range(n_passages):
        p = prompts[i % len(prompts)]
        msgs = [{"role": "user", "content": p}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       return_tensors="pt").to(model.device)
        gen = model.generate(ids, do_sample=True, temperature=0.8,
                              top_p=0.95, max_new_tokens=max_new_tokens,
                              pad_token_id=tok.eos_token_id)
        txt = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
        out.append(txt.strip())
    return "\n\n".join(out)
```

**Step 4:** `pytest tests/test_distill_teacher.py -q` → PASS (3).

**Step 5: Commit** `feat(generator-Inc2): training-time-only teacher text generator (runtime-isolated)`

---

### Task 2: Build + cache the distillation corpus

**Files:** Create `research/runners/build_distill_corpus.py`; Test `tests/test_build_distill_corpus.py`

**Step 1: Failing test (pure cleaner — no model):**
```python
from research.runners.build_distill_corpus import clean_corpus, DISTILL_PATH
def test_clean_ascii_and_nonempty():
    raw = "Héllo  world!\n\n\nUnicode ✓ tail"
    c = clean_corpus(raw)
    assert all(32 <= ord(ch) < 127 or ch == "\n" for ch in c)
    assert "Hllo" not in c and len(c) > 0  # accented stripped, content kept
def test_deterministic():
    assert clean_corpus("abc abc") == clean_corpus("abc abc")
def test_distill_path_under_datasets():
    assert "research" in str(DISTILL_PATH) and str(DISTILL_PATH).endswith(".txt")
```

**Step 2:** FAIL.

**Step 3: Implement** — `clean_corpus(raw)`: keep printable ASCII + newline, collapse 3+ newlines, strip. `DISTILL_PATH = research/datasets/distill_corpus.txt`. `main()`: `distill_teacher.generate_corpus(...)` → `clean_corpus` → write `DISTILL_PATH` (skip regen if exists unless `--force`); print char count.

**Step 4:** PASS (3).

**Step 5:** Run it once (GPU, one-time): `python -m research.runners.build_distill_corpus --n-passages 200` → writes `research/datasets/distill_corpus.txt` (expect ≥150 KB ASCII English). Commit code + the cached corpus: `feat(generator-Inc2): build+cache teacher-distilled corpus (Kim&Rush data distillation)`

---

### Task 3: Controlled anti-cheat distillation gate

**Files:** Create `research/runners/distill_gate.py`

Reuses Increment-1: `local_corpus.load_local_corpus` (baseline), the new `distill_corpus.txt` (distilled), and `cortex_pretraining.train_shakespeare` via the tmpfile pattern from `generator_baseline_smoke.py` (DRY — copy that proven harness). Three identical-config student trainings (same seed, hidden, epochs, T, n_samples): **REAL-baseline** (Inc-1 corpus), **DISTILLED** (teacher corpus), **PERMUTED** (shuffled distilled chars). Teacher is **absent** here — only its previously-cached text file is read.

**Gate (falsifiable, anti-cheat):**
- DISTILLED end-loss < REAL-baseline end-loss by ≥ a real margin (e.g. ≥10% lower), AND
- DISTILLED end-loss < PERMUTED end-loss decisively (learned real structure, not data-size/noise artifact).
- PASS ⇒ teacher-distilled data gives the student's OWN weights a genuine lift, self-contained. FAIL ⇒ honest NEGATIVE (data distillation didn't help at PoC scale) — document, do not paper over, do not tune the gate to pass.

Output JSON `research/findings/raw/g11_bg/distill_gate.json` + printed verdict. Run it (GPU, minutes). Commit: `feat(generator-Inc2): controlled anti-cheat distillation gate (distilled vs baseline vs permuted)`

---

### Task 4: Propagate (honest, PoC-framed)

**Files:** Create `research/findings/2026-05-16-generator-increment2-distillation.md`; Modify `webapp/capability_status.json` (pillar VALIDATED only if Task-3 gate PASSES; else NEGATIVE/BOUNDARY honestly). Run `pytest tests/test_webapp_server.py -k capability_status -q`.

Findings doc states plainly: this is **sequence-level data distillation** (Kim & Rush 2016) at **PoC scale**; the student is still a small char-level net, **not LLM-fluent**; teacher used training-time only and is code-asserted out of the self-contained runtime; the gate is what makes the lift credible (teacher absent at eval, beats baseline AND permuted). If FAIL: honest negative — distillation didn't lift at this scale; the path/mechanism is still sound, scale/recipe is the open question. Commit + push both remotes.

---

## Notes for the executor

- **Teacher is training-time ONLY.** Task-1's runtime-isolation test is load-bearing — the self-contained runtime must never import `distill_teacher`. If a later increment's runtime needs to import it, that's a design violation, not a test to weaken.
- DRY: copy the proven REAL-vs-PERMUTED harness from `generator_baseline_smoke.py`; reuse `cortex_pretraining.train_shakespeare`, `local_corpus`. Do NOT reimplement BPTT or the trainer.
- YAGNI: no logit distillation, no KD temperature, no multi-teacher — just teacher-text → student-trains-on-it → controlled gate.
- Honesty/anti-cheat is the point: the gate must show the SELF-CONTAINED student (teacher absent) beats the Inc-1 baseline AND a permuted control. A failed gate is a real finding; never tune the threshold to force PASS, never stub the teacher.
- Offline: `HF_HUB_OFFLINE=1`/`local_files_only=True` so the teacher cannot re-fetch at generation time (one-time cached fetch already done).
- Keep ASCII in all `print()` (Windows cp1252 console — a non-ASCII char crashed an Inc-1 run).
- Pure logic (clean_corpus, guards) = CPU pytest. Teacher-gen + trainings validated by the controlled gate (project pattern).
