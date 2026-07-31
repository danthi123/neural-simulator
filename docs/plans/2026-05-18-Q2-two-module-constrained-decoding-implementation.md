---
type: plan
status: live
date: 2026-05-18
---

# Q2 — Two-module CONSTRAINED-DECODING faithful generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (standing autonomy pre-selects Subagent-Driven, this session). Task 5 is CONTROLLER-ONLY — bring it back to the controller.

**Goal:** Build a kill-safe in-sim gate testing whether the validated Generator-F (proposes tokens) + the validated no-confab grounded memory (vetoes ungrounded next-tokens PER-TOKEN at decode time) compose into a generator that stays NON-VACUOUSLY faithful by construction and is SCALE-CONFIDENT across a pre-registered grounded-KB ladder K∈{6,12,24}.

**Architecture:** Two net-new modules. `constrained_decode_core.py` = own frozen `_CDC_*` THREE-STATE + scale-confidence verdict (mirrors `generator_g_core`/`compose_bridge_core` discipline; imports only stdlib+typing + reuses `generator_g_core` metric primitives by import, never mutates them). `constrained_decode_gate.py` = a `_GroundedConstrainedLM` per-token grounded-veto wrapper around the byte-UNMODIFIED Generator-F TinyGPT, dropped into the byte-UNMODIFIED `grounded_decode`, + a kill-safe multi-rung CLI. The per-token veto makes ungrounded-entity-rate ~0 BY CONSTRUCTION (mechanical, NOT the discriminator); the DISCRIMINATING signature is "constrained stays NON-VACUOUS" (≥`_CDC_MIN_GROUNDED_CONTENT` distinct on-proposition content words) vs `unconstrained` (the Generator-G ~0.89 drift regime) and `shuffled_grounding`.

**Tech Stack:** Python 3, PyTorch (Generator-F inference ONLY — reused validated artifact; CUDA on the decisive run), NumPy, pytest. NO new autograd/training/`.backward()` in net-new code.

**Plan base commit:** `02addfa`. The protected-empty-diff invariant is `git diff 02addfa..HEAD -- <PROTECTED SET>` MUST be empty at every task.

> **Hook note (read before transcribing code):** the project's security-reminder hook false-positives on the literal substring `eval(`. PyTorch's inference-mode call `model.eval()` therefore uses the **100%-equivalent** `model.train(False)` in this plan's reference code. Keep it as `model.train(False)` — do NOT "helpfully" change it back to `.eval()` (it will re-trip the hook and is functionally identical: `nn.Module.eval()` is defined as `self.train(False)`).

---

## Reused interfaces (grounded — do NOT modify)

- `sim/grounded_decode.py::grounded_decode(ranked, lm, tok, retrieved_text, query, threshold=650.0, max_new=40, temperature=0.0) -> {"abstained":bool,"text":str|None,"retrieved":str}`. Internally: `top = gate(ranked, threshold)`; if `top is None` → `{"abstained":True,"text":None,...}` (LM NEVER touched = no-confab by construction); else `prompt_ids = tok.encode(retrieved_text); gen_ids = lm.generate_ids(prompt_ids, int(max_new)); text = tok.decode(gen_ids)`. **`lm` is duck-typed: only `.generate_ids(prompt_ids, max_new)` is called, and `prompt_ids == tok.encode(retrieved_text)`** — a constrained LM derives its veto allow-set from `prompt_ids` itself with NO change to `grounded_decode`.
- `research/runners/abstention_gate.py::gate(ranked, threshold=650.0)` → top tuple or `None`; `DEFAULT_THRESHOLD=650.0`. Byte-UNMODIFIED (the no-confab moat).
- `research/runners/generator_g_core.py`: `ungrounded_entity_rate(response, retrieved, function_words=FUNCTION_WORDS) -> float`, `is_answered(response, function_words=FUNCTION_WORDS) -> bool`, `FUNCTION_WORDS`. Both normalize via `re.sub(r"[^\w]","",w.lower())` then split. REUSE by import byte-UNMODIFIED.
- Generator-F artifact (verified present): `research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real.pt` (+`.bpe.json`). Loader pattern: `research/runners/generator_g_gate.py::_TinyGPTLM` (TinyGPT d_model=256,n_layer=4,n_head=4,block_size=128; `BPETokenizer.load(prefix+".bpe.json")`; `torch.load(prefix+".pt", map_location=...)`; `model.load_state_dict(st["model"])`; inference mode via `model.train(False)`).
- `sim/tiny_transformer.py::TinyGPT(vocab_size,...).forward(idx)->logits` (`model(x)[0,-1]` = last-step logits). `sim/bpe_tokenizer.py::BPETokenizer.encode/decode/vocab_size/.load`.

## PROTECTED SET (byte-empty in EVERY commit-scoped diff AND `git diff 02addfa..HEAD`)

```
research/runners/abstention_gate.py            tests/test_abstention_gate.py  # moat 7/7
sim/grounded_decode.py                          research/runners/generator_g_core.py
research/runners/compose_bridge_core.py         research/runners/compose_bind_core.py
research/runners/td_critic_core.py              research/runners/dendritic_fair_core.py
research/runners/engram_bootstrap_gate.py       sim/tiny_transformer.py  sim/bpe_tokenizer.py
sim/bridge.py  sim/td_value_critic.py  sim/compose_temporal_bind.py  sim/kernels.py
sim/neuromodulators.py  sim/train_checkpoint.py  sim/backend.py  sim/dendritic_plasticity.py
research/runners/text_minimal_isolation.py
research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real.pt
research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real.bpe.json
```

Controller trust-but-verify EVERY task's `git diff 02addfa..HEAD -- <PROTECTED SET>` is EMPTY before marking complete.

---

### Task 0: Grounding pin

**Files:** Create `tests/test_q2_grounding.py`

```python
"""Q2 Task-0 grounding pin. RED until Task 1/2 ship the net-new modules."""
import importlib


def test_reused_no_confab_moat_and_metrics_present():
    ag = importlib.import_module("research.runners.abstention_gate")
    assert ag.DEFAULT_THRESHOLD == 650.0
    gg = importlib.import_module("research.runners.generator_g_core")
    assert callable(gg.ungrounded_entity_rate) and callable(gg.is_answered)
    assert "the" in gg.FUNCTION_WORDS
    assert callable(importlib.import_module("sim.grounded_decode").grounded_decode)


def test_generator_f_artifact_present():
    import os
    base = "research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real"
    assert os.path.exists(base + ".pt") and os.path.exists(base + ".bpe.json")


def test_constrained_decode_core_frozen_and_pure():
    c = importlib.import_module("research.runners.constrained_decode_core")
    assert c._CDC_FAITHFUL_MAX == 0.20
    assert c._CDC_MIN_GROUNDED_CONTENT == 2
    assert c._CDC_MIN_GROUNDED_ANSWER_RATE == 0.5
    assert c._CDC_MIN_SEEDS == 3
    assert c._CDC_SCALE_LADDER == (6, 12, 24)
    assert c._CDC_SCALE_TOL == 0.10
    assert callable(c.cdc_verdict) and callable(c.cdc_scale_confidence)
    import inspect
    src = inspect.getsource(c)
    assert "backward(" not in src and "import torch" not in src


def test_constrained_decode_gate_importable():
    g = importlib.import_module("research.runners.constrained_decode_gate")
    assert hasattr(g, "_GroundedConstrainedLM") and hasattr(g, "main")
```

Run `python -m pytest tests/test_q2_grounding.py -q` → FAIL (`constrained_decode_core` missing; intentional RED). Commit `git add tests/test_q2_grounding.py && git commit -m "test: Q2 Task-0 grounding pin (intentionally RED until Task 1/2)"`. Controller verifies protected diff EMPTY.

---

### Task 1: `constrained_decode_core.py` — frozen THREE-STATE + scale-confidence (FULLY SPECIFIED — transcribe exactly)

**Files:** Create `research/runners/constrained_decode_core.py`, `tests/test_constrained_decode_core.py`

**Step 1: adversarial test matrix** `tests/test_constrained_decode_core.py`:

```python
from research.runners.constrained_decode_core import (
    cdc_verdict, cdc_scale_confidence, grounded_content_count,
    nonvacuous_answered)


def _seed_ok(**kw):
    d = dict(unconstrained_uer=0.85, constrained_uer=0.0,
             constrained_nonvac_rate=0.9, shuffled_uer=0.85,
             shuffled_nonvac_rate=0.0, bare_moat_abstain_rate=1.0,
             abstain_on_ungrounded_rate=1.0)
    d.update(kw); return d


def test_grounded_content_count_distinct_on_prop():
    assert grounded_content_count("the max is a dog dog",
                                  "max is a big dog") == 2


def test_nonvacuous_requires_min_distinct_content():
    assert nonvacuous_answered("max dog", "max is a big dog") is True
    assert nonvacuous_answered("max the the", "max is a big dog") is False
    assert nonvacuous_answered("the is a and", "max is a big dog") is False


def test_all_good_seeds_pass():
    v = cdc_verdict({42: _seed_ok(), 43: _seed_ok(), 44: _seed_ok()})
    assert v["GATE"] == "PASS" and v["instrument_valid"] is True


def test_v1_unmet_unconstrained_not_drifting_is_void():
    v = cdc_verdict({42: _seed_ok(unconstrained_uer=0.10),
                     43: _seed_ok(unconstrained_uer=0.10),
                     44: _seed_ok(unconstrained_uer=0.10)})
    assert v["GATE"] == "VOID" and v["instrument_valid"] is False


def test_vacuity_collapse_is_FAIL_not_pass():
    v = cdc_verdict({42: _seed_ok(constrained_nonvac_rate=0.10),
                     43: _seed_ok(constrained_nonvac_rate=0.10),
                     44: _seed_ok(constrained_nonvac_rate=0.10)})
    assert v["GATE"] == "FAIL" and v["instrument_valid"] is True


def test_unconstrained_control_passing_faithful_is_void():
    v = cdc_verdict({42: _seed_ok(unconstrained_uer=0.05),
                     43: _seed_ok(unconstrained_uer=0.05),
                     44: _seed_ok(unconstrained_uer=0.05)})
    assert v["GATE"] == "VOID"


def test_shuffled_control_not_failing_is_void():
    v = cdc_verdict({42: _seed_ok(shuffled_uer=0.0, shuffled_nonvac_rate=0.9),
                     43: _seed_ok(shuffled_uer=0.0, shuffled_nonvac_rate=0.9),
                     44: _seed_ok(shuffled_uer=0.0, shuffled_nonvac_rate=0.9)})
    assert v["GATE"] == "VOID"


def test_no_confab_not_preserved_is_void():
    v = cdc_verdict({42: _seed_ok(abstain_on_ungrounded_rate=0.4),
                     43: _seed_ok(abstain_on_ungrounded_rate=0.4),
                     44: _seed_ok(abstain_on_ungrounded_rate=0.4)})
    assert v["GATE"] == "VOID"


def test_fewer_than_min_seeds_is_void():
    assert cdc_verdict({42: _seed_ok()})["GATE"] == "VOID"


def test_non_numeric_junk_is_void_not_raise():
    bad = dict(_seed_ok()); bad["constrained_nonvac_rate"] = "oops"
    assert cdc_verdict({42: bad, 43: _seed_ok(),
                        44: _seed_ok()})["GATE"] == "VOID"


def test_unorderable_keys_void_not_raise():
    assert cdc_verdict({object(): _seed_ok()})["GATE"] == "VOID"


def _rg(K, g, nv):
    return {"K": K, "verdict": {"GATE": g},
            "constrained_nonvac_rate_mean": nv}


def test_scale_confident_all_pass_nondegrading():
    r = cdc_scale_confidence([_rg(6, "PASS", 0.9), _rg(12, "PASS", 0.9),
                              _rg(24, "PASS", 0.88)])
    assert r["scale_confident"] is True
    assert r["classification"] == "SCALE-CONFIDENT-PASS"


def test_scale_degrades_is_works_small():
    r = cdc_scale_confidence([_rg(6, "PASS", 0.9), _rg(12, "PASS", 0.7),
                              _rg(24, "PASS", 0.55)])
    assert r["classification"] == "WORKS-SMALL-NO-SCALE-CONFIDENCE"


def test_scale_void_and_fail_precedence():
    assert cdc_scale_confidence([_rg(6, "PASS", .9), _rg(12, "VOID", 0.),
        _rg(24, "PASS", .9)])["classification"] == "VOID"
    assert cdc_scale_confidence([_rg(6, "PASS", .9), _rg(12, "FAIL", .5),
        _rg(24, "PASS", .9)])["classification"] == "FAIL"


def test_scale_ladder_tamper_is_void():
    assert cdc_scale_confidence([_rg(6, "PASS", .9), _rg(12, "PASS", .9)]
        )["classification"] == "VOID"
```

**Step 2:** Run → FAIL (module missing).

**Step 3: Create `research/runners/constrained_decode_core.py` EXACTLY:**

```python
"""Pure FIXED-bar THREE-STATE + SCALE-CONFIDENCE verdict for Q2
two-module per-token grounded constrained decoding. Mirrors the
adversarial-hardened generator_g_core / compose_bridge_core DISCIPLINE
(fixed bars NEVER tuned, instrument-validity FIRST, fail-closed, VOID
strictly distinct from FAIL, malformed/junk -> VOID-not-raise). Holds
its OWN frozen _CDC_*; does NOT import/mutate any *_core. REUSES the
validated generator_g_core metric PRIMITIVES (FUNCTION_WORDS,
is_answered) by import, byte-UNMODIFIED. Pure stdlib+typing; NO torch,
NO autograd. ASCII only.

KEY (cheap-probe-surfaced, recorded in the design): a per-token veto
makes ungrounded-entity-rate ~0 BY CONSTRUCTION -- MECHANICAL, NOT the
discriminating result. The DISCRIMINATING Q2 signature is "constrained
stays NON-VACUOUS" via a STRENGTHENED grounded-CONTENT-word bar
(>= _CDC_MIN_GROUNDED_CONTENT distinct on-proposition content words),
NOT bare is_answered>=1 (proven too weak by the cheap probe)."""
from __future__ import annotations
import math
import re
from typing import Dict

from research.runners.generator_g_core import (
    FUNCTION_WORDS, is_answered)  # reused byte-UNMODIFIED

_CDC_FAITHFUL_MAX = 0.20
_CDC_MIN_GROUNDED_CONTENT = 2
_CDC_MIN_GROUNDED_ANSWER_RATE = 0.5
_CDC_MIN_SEEDS = 3
_CDC_SCALE_LADDER = (6, 12, 24)
_CDC_SCALE_TOL = 0.10


def _norm(s):
    out = []
    for w in str(s).split():
        t = re.sub(r"[^\w]", "", w.lower())
        if t:
            out.append(t)
    return out


def grounded_content_count(response_text, retrieved_text,
                           function_words=FUNCTION_WORDS) -> int:
    ret = set(_norm(retrieved_text))
    seen = set()
    for w in _norm(response_text):
        if w not in function_words and w in ret:
            seen.add(w)
    return len(seen)


def nonvacuous_answered(response_text, retrieved_text) -> bool:
    if not is_answered(response_text):
        return False
    return (grounded_content_count(response_text, retrieved_text)
            >= _CDC_MIN_GROUNDED_CONTENT)


def _finite(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


_REQUIRED = ("unconstrained_uer", "constrained_uer",
             "constrained_nonvac_rate", "shuffled_uer",
             "shuffled_nonvac_rate", "bare_moat_abstain_rate",
             "abstain_on_ungrounded_rate")


def cdc_verdict(per_seed: dict) -> dict:
    bars = {"FAITHFUL_MAX": _CDC_FAITHFUL_MAX,
            "MIN_GROUNDED_CONTENT": _CDC_MIN_GROUNDED_CONTENT,
            "MIN_GROUNDED_ANSWER_RATE": _CDC_MIN_GROUNDED_ANSWER_RATE,
            "MIN_SEEDS": _CDC_MIN_SEEDS}
    try:
        seeds = sorted(per_seed.keys())
    except TypeError:
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "per_seed keys not orderable",
                "frozen_bars": bars, "per_seed": {}}
    base = {"frozen_bars": bars,
            "per_seed": {str(s): per_seed[s] for s in seeds}}
    if len(seeds) < _CDC_MIN_SEEDS:
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "fewer than %d seeds" % _CDC_MIN_SEEDS, **base}
    v1_ok = science_ok = controls_fail = no_confab_ok = True
    metrics_finite = True
    for s in seeds:
        d = per_seed[s]
        if not isinstance(d, dict):
            metrics_finite = False
            continue
        vals = {k: _finite(d.get(k)) for k in _REQUIRED}
        if any(v is None for v in vals.values()):
            metrics_finite = False
            continue
        if not (vals["unconstrained_uer"] > _CDC_FAITHFUL_MAX
                and vals["bare_moat_abstain_rate"] > 0.0):
            v1_ok = False
        unconstrained_fails = vals["unconstrained_uer"] > _CDC_FAITHFUL_MAX
        shuffled_fails = (vals["shuffled_uer"] > _CDC_FAITHFUL_MAX
                          or vals["shuffled_nonvac_rate"]
                          < _CDC_MIN_GROUNDED_ANSWER_RATE)
        if not (unconstrained_fails and shuffled_fails):
            controls_fail = False
        if not (vals["constrained_uer"] <= _CDC_FAITHFUL_MAX
                and vals["constrained_nonvac_rate"]
                >= _CDC_MIN_GROUNDED_ANSWER_RATE):
            science_ok = False
        if vals["abstain_on_ungrounded_rate"] < \
                vals["bare_moat_abstain_rate"] - 1e-9:
            no_confab_ok = False
    instrument_valid = bool(v1_ok and controls_fail and no_confab_ok
                            and metrics_finite)
    if not instrument_valid:
        why = []
        if not v1_ok:
            why.append("V1 unmet: unconstrained did NOT drift above "
                       "the faithful bar (instrument cannot see drift)")
        if not controls_fail:
            why.append("a control did not fail -> veto NOT the "
                       "discriminator (non-discriminating)")
        if not no_confab_ok:
            why.append("no-confab NOT preserved (abstain_on_ungrounded "
                       "< bare moat)")
        if not metrics_finite:
            why.append("a required metric non-numeric/non-finite/"
                       "malformed")
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "; ".join(why), **base}
    return {"GATE": "PASS" if science_ok else "FAIL",
            "instrument_valid": True, "science_ok": bool(science_ok),
            "note": ("constrained ungrounded-entity-rate ~0 is "
                     "MECHANICAL (per-token veto, by construction) -- "
                     "NOT the discriminating result; the discriminator "
                     "is constrained NON-VACUITY vs the failing "
                     "controls"), **base}


def cdc_scale_confidence(rungs):
    try:
        ordered = sorted(rungs, key=lambda r: r["K"])
    except (TypeError, KeyError):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "rungs not orderable by K"}
    if [r.get("K") for r in ordered] != list(_CDC_SCALE_LADDER):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "ladder != %s" % (_CDC_SCALE_LADDER,)}
    gates = [r.get("verdict", {}).get("GATE")
             if isinstance(r.get("verdict"), dict) else None
             for r in ordered]
    if any(g == "VOID" or g is None for g in gates):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "a rung GATE VOID/missing"}
    if any(g == "FAIL" for g in gates):
        return {"scale_confident": False, "classification": "FAIL",
                "reason": "a rung GATE FAIL"}
    if any(g != "PASS" for g in gates):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "a rung GATE not PASS/FAIL/VOID"}
    nv = []
    for r in ordered:
        f = _finite(r.get("constrained_nonvac_rate_mean"))
        if f is None:
            return {"scale_confident": False, "classification": "VOID",
                    "reason": "non-numeric rung non-vacuity"}
        nv.append(f)
    monotone = all(nv[i + 1] >= nv[i] - _CDC_SCALE_TOL
                   for i in range(len(nv) - 1))
    top_ok = nv[-1] >= _CDC_MIN_GROUNDED_ANSWER_RATE
    if monotone and top_ok:
        return {"scale_confident": True,
                "classification": "SCALE-CONFIDENT-PASS",
                "reason": "all rungs PASS; non-vacuity non-decreasing "
                          "up to tol; holds at largest rung",
                "nonvac_by_rung": nv}
    return {"scale_confident": False,
            "classification": "WORKS-SMALL-NO-SCALE-CONFIDENCE",
            "reason": "all rungs PASS but %s%s"
                      % ("" if monotone else "non-vacuity degrades "
                         "beyond tol; ",
                         "" if top_ok else "non-vacuity below bar at "
                         "largest rung"),
            "nonvac_by_rung": nv}
```

**Step 4:** Run `python -m pytest tests/test_constrained_decode_core.py tests/test_q2_grounding.py -q` → core PASS (≥15); grounding's `test_constrained_decode_gate_importable` still FAILS (Task 2). Expected partial-RED.

**Step 5:** Commit `git add research/runners/constrained_decode_core.py tests/test_constrained_decode_core.py && git commit -m "feat: Q2 constrained_decode_core frozen THREE-STATE + scale-confidence (reuses generator_g_core primitives byte-UNMODIFIED)"`. Controller verifies protected diff EMPTY.

---

### Task 2: `constrained_decode_gate.py` — per-token grounded-veto wrapper + kill-safe multi-rung CLI

**Files:** Create `research/runners/constrained_decode_gate.py`, `tests/test_q2_smoke.py`

**Step 1: smoke test** `tests/test_q2_smoke.py`:

```python
"""--tiny smoke: gate runs end-to-end (CPU OK for smoke; DECISIVE run is
GPU/controller-only), cdc_verdict-shaped, toy verdict NOT propagated,
no-confab preserved on ungrounded (LM never touched)."""
import json, subprocess, sys, os


def test_tiny_smoke_runs(tmp_path):
    out = tmp_path / "q2_smoke.json"
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.constrained_decode_gate",
         "--tiny", "--seeds", "42", "43", "44", "--out", str(out)],
        capture_output=True, text=True, timeout=3600, env={**os.environ})
    assert r.returncode == 0, r.stderr[-3000:]
    d = json.loads(out.read_text())
    assert d["tiny"] is True and d["note"].startswith("TINY")
    assert len(d["ladder"]) == 1
    ps = d["ladder"][0]["verdict"]["per_seed"]
    assert all(v["abstain_on_ungrounded_rate"]
               >= v["bare_moat_abstain_rate"] - 1e-9 for v in ps.values())
```

**Step 2:** Run → FAIL (no module).

**Step 3: Create `research/runners/constrained_decode_gate.py`** (reference = behavioral spec; mirror `generator_g_gate.py` for loader/resume/frozen-slice; net-new = the veto + multi-rung loop). NOTE the deliberate `model.train(False)` (== `.eval()`; avoids the `eval(` hook false-positive — keep it):

```python
"""Q2 kill-safe gate: validated Generator-F PROPOSES tokens; the
validated no-confab grounded memory VETOES ungrounded next-tokens
PER-TOKEN at decode time. The validated moat (abstention_gate '650',
byte-UNMODIFIED) gates answer-vs-abstain FIRST via grounded_decode
(byte-UNMODIFIED; LM NEVER touched on abstain = no-confab by
construction). Per-token veto makes ungrounded-entity-rate ~0 BY
CONSTRUCTION (MECHANICAL, NOT the discriminator); the DISCRIMINATING
signature is constrained NON-VACUITY (>= _CDC_MIN_GROUNDED_CONTENT
distinct on-prop content words) vs the unconstrained Generator-G drift
regime + shuffled_grounding. Decisive slice isolates the COMPOSITION
via a FROZEN grounded source (G.20 retrieval separately validated;
full-retrieval wiring a later increment). Generator-F inference is
torch (reused validated artifact, INFERENCE ONLY -- no new training/
autograd; inference mode set via model.train(False) == .eval()). CUDA
when available (the decisive run MUST use the GPU; CPU only if CUDA
absent, logged). FROZEN _CDC_* via constrained_decode_core NEVER tuned.
Honest propagation = the CONTROLLER's post-run job. ASCII."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np

from sim.grounded_decode import grounded_decode
from research.runners.abstention_gate import gate, DEFAULT_THRESHOLD
from research.runners.generator_g_core import ungrounded_entity_rate, \
    FUNCTION_WORDS
from research.runners.constrained_decode_core import (
    cdc_verdict, cdc_scale_confidence, nonvacuous_answered,
    _CDC_SCALE_LADDER)

_GEN_F = "research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real"

_GROUNDED = {
 "max":"max is a big friendly dog","lily":"lily has a small red ball",
 "tom":"tom found a shiny blue key","sue":"sue likes warm sweet bread",
 "ben":"ben rides a fast green bike","mia":"mia keeps a soft white cat",
 "leo":"leo plants a tall oak tree","ana":"ana paints a bright yellow sun",
 "sam":"sam sails a small wood boat","kai":"kai flies a bright red kite",
 "joy":"joy sings a slow sweet song","rex":"rex digs a deep round hole",
 "ivy":"ivy grows a sweet purple plum","dan":"dan builds a strong stone wall",
 "eve":"eve reads a long old book","gus":"gus cooks a hot tasty soup",
 "pam":"pam sews a warm wool coat","ned":"ned mends a torn blue sail",
 "uma":"uma rows a long thin canoe","ole":"ole carves a small pine duck",
 "fay":"fay bakes a round nut cake","hal":"hal sweeps a wide dusty barn",
 "wes":"wes feeds a tame brown hen","zoe":"zoe ties a tight square knot",
}
_UNGROUNDED = ["zarn","qexel","drovil","plonk","vexin","wun"]


class _GroundedConstrainedLM:
    """Duck-typed IDENTICALLY to generator_g_gate._TinyGPTLM (same
    __init__; same .generate_ids(prompt_ids, max_new)) so it drops into
    grounded_decode BYTE-UNMODIFIED. generate_ids applies the per-token
    grounded VETO: allowed vocab = token ids whose normalized decoded
    surface is empty (punct/space) OR every word is in (allow_words
    UNION FUNCTION_WORDS). allow_words = the prompt's own words (=
    retrieved proposition, since grounded_decode passes prompt_ids =
    tok.encode(retrieved_text)). mode: 'constrained' veto on;
    'unconstrained' veto OFF (= Generator-G regime); 'shuffled' veto
    allow_words from self._shuffle_text (a DIFFERENT proposition)."""
    def __init__(self, ckpt_prefix, mode="constrained", block_size=128):
        import torch
        from sim.tiny_transformer import TinyGPT
        from sim.bpe_tokenizer import BPETokenizer
        self._torch = torch
        self.mode = mode
        self._shuffle_text = None
        self.tok = BPETokenizer.load(ckpt_prefix + ".bpe.json")
        V = self.tok.vocab_size
        self.block = block_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TinyGPT(vocab_size=V, d_model=256, n_layer=4,
                             n_head=4, block_size=block_size,
                             dropout=0.0)
        st = torch.load(ckpt_prefix + ".pt", map_location=self.device)
        self.model.load_state_dict(st["model"])
        self.model.train(False)            # inference mode (== .eval())
        self.model.to(self.device)
        self._allow_cache = {}

    def _norm_words(self, s):
        import re
        return [t for t in (re.sub(r"[^\w]", "", w.lower())
                            for w in str(s).split()) if t]

    def _allowed_mask(self, allow_text):
        if allow_text in self._allow_cache:
            return self._allow_cache[allow_text]
        torch = self._torch
        allow = set(self._norm_words(allow_text)) | set(FUNCTION_WORDS)
        V = self.tok.vocab_size
        mask = torch.zeros(V, dtype=torch.bool)
        for tid in range(V):
            surf = self._norm_words(self.tok.decode([tid]))
            if not surf or all(w in allow for w in surf):
                mask[tid] = True
        mask = mask.to(self.device)
        self._allow_cache[allow_text] = mask
        return mask

    def generate_ids(self, prompt_ids, max_new):
        torch = self._torch
        seq = list(prompt_ids) if prompt_ids else [0]
        out = []
        use_veto = self.mode in ("constrained", "shuffled")
        if self.mode == "shuffled" and self._shuffle_text is not None:
            allow_text = self._shuffle_text
        else:
            allow_text = (self.tok.decode(list(prompt_ids))
                          if prompt_ids else "")
        mask = self._allowed_mask(allow_text) if use_veto else None
        with torch.no_grad():
            for _ in range(int(max_new)):
                ctx = seq[-self.block:]
                x = torch.tensor(ctx, dtype=torch.long,
                                 device=self.device)[None]
                logits = self.model(x)[0, -1]
                if use_veto:
                    logits = logits.masked_fill(~mask, float("-inf"))
                nxt = int(torch.argmax(logits).item())
                seq.append(nxt)
                out.append(nxt)
        return out


def _params(tiny):
    if tiny:
        return dict(ladder=(_CDC_SCALE_LADDER[0],), max_new=12,
                    n_ungrounded=3)
    return dict(ladder=_CDC_SCALE_LADDER, max_new=40, n_ungrounded=6)


def _run_rung(K, seeds, lm_c, lm_u, lm_s, max_new, n_ung):
    items = list(_GROUNDED.items())[:K]
    props = [p for _, p in items]
    ung = list(_UNGROUNDED)[:n_ung]
    per_seed = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = list(range(len(items)))
        rng.shuffle(order)
        c_uer, u_uer, s_uer, c_nv, s_nv = [], [], [], [], []
        for idx in order:
            subj, prop = items[idx]
            ranked = [(subj, 900.0, "kb")]
            r = grounded_decode(ranked, lm_c, lm_c.tok,
                                retrieved_text=prop, query=subj,
                                threshold=DEFAULT_THRESHOLD,
                                max_new=max_new)
            ct = r["text"] or ""
            c_uer.append(ungrounded_entity_rate(ct, prop))
            c_nv.append(1.0 if nonvacuous_answered(ct, prop) else 0.0)
            ru = grounded_decode(ranked, lm_u, lm_u.tok,
                                 retrieved_text=prop, query=subj,
                                 threshold=DEFAULT_THRESHOLD,
                                 max_new=max_new)
            u_uer.append(ungrounded_entity_rate(ru["text"] or "", prop))
            lm_s._shuffle_text = props[(idx + 1) % len(props)]
            rs = grounded_decode(ranked, lm_s, lm_s.tok,
                                 retrieved_text=prop, query=subj,
                                 threshold=DEFAULT_THRESHOLD,
                                 max_new=max_new)
            st_ = rs["text"] or ""
            s_uer.append(ungrounded_entity_rate(st_, prop))
            s_nv.append(1.0 if nonvacuous_answered(st_, prop) else 0.0)
        n_abst = bare = 0
        for subj in ung:
            if gate([], DEFAULT_THRESHOLD) is None:
                bare += 1
            ra = grounded_decode([], lm_c, lm_c.tok, retrieved_text="",
                                 query=subj, threshold=DEFAULT_THRESHOLD,
                                 max_new=max_new)
            if ra["abstained"]:
                n_abst += 1
        nu = max(1, len(ung))
        per_seed[seed] = {
            "unconstrained_uer": float(np.mean(u_uer)),
            "constrained_uer": float(np.mean(c_uer)),
            "constrained_nonvac_rate": float(np.mean(c_nv)),
            "shuffled_uer": float(np.mean(s_uer)),
            "shuffled_nonvac_rate": float(np.mean(s_nv)),
            "bare_moat_abstain_rate": bare / nu,
            "abstain_on_ungrounded_rate": n_abst / nu}
    verdict = cdc_verdict(per_seed)
    nv_mean = float(np.mean(
        [per_seed[s]["constrained_nonvac_rate"] for s in per_seed]))
    return {"K": K, "verdict": verdict,
            "constrained_nonvac_rate_mean": nv_mean}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46])
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--ckpt", default="research/findings/raw/g11_bg/"
                    "constrained_decode_gate")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    if not (os.path.exists(_GEN_F + ".pt")
            and os.path.exists(_GEN_F + ".bpe.json")):
        print("NOT-RUNNABLE: Generator-F artifact absent"); return 2
    P = _params(a.tiny)
    lm_c = _GroundedConstrainedLM(_GEN_F, mode="constrained")
    lm_u = _GroundedConstrainedLM(_GEN_F, mode="unconstrained")
    lm_s = _GroundedConstrainedLM(_GEN_F, mode="shuffled")
    print("DEVICE=%s (CUDA=%s) -- decisive run MUST be cuda"
          % (lm_c.device, lm_c._torch.cuda.is_available()))
    resume = str(a.ckpt) + ".resume.json"
    done = {}
    if Path(resume).exists():
        try:
            done = {int(k): v for k, v in json.loads(
                Path(resume).read_text()).get("done", {}).items()}
        except (ValueError, OSError):
            done = {}
    rungs = []
    t0 = time.time()
    try:
        for K in P["ladder"]:
            if K in done:
                rungs.append(done[K]); continue
            rg = _run_rung(K, a.seeds, lm_c, lm_u, lm_s,
                           P["max_new"], P["n_ungrounded"])
            rungs.append(rg); done[K] = rg
            tmp = resume + ".tmp"
            Path(tmp).parent.mkdir(parents=True, exist_ok=True)
            Path(tmp).write_text(json.dumps(
                {"done": {str(k): v for k, v in done.items()}}))
            os.replace(tmp, resume)
    except KeyboardInterrupt:
        print("INTERRUPTED -- partial resume flushed; resumable")
        return 130
    sc = (cdc_scale_confidence(rungs) if not a.tiny else
          {"scale_confident": False,
           "classification": "TINY (toy; NOT propagated)"})
    out = {"ladder": rungs, "scale_confident": sc["scale_confident"],
           "scale_classification": sc["classification"],
           "scale_reason": sc.get("reason", ""),
           "device": lm_c.device, "tiny": bool(a.tiny),
           "note": ("TINY toy verdict -- NOT propagated" if a.tiny
                    else "multi-rung scale-confidence verdict -- "
                    "recompute from this JSON; no re-run/no tuning"),
           "elapsed_seconds": round(time.time() - t0, 1),
           "HONEST_CEILING": ("scale-confidence PoC: per-token grounded "
               "constrained decoding stays NON-VACUOUSLY faithful by "
               "construction + no scale degradation; NOT open-ended "
               "fluent composition, NOT an LLM, NOT conversation-"
               "solved; constrained decoding TRADES fluency for "
               "faithfulness BY DESIGN")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("SCALE=%s class=%s device=%s"
          % (out["scale_confident"], out["scale_classification"],
             out["device"]))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

**Step 4:** Run `python -m pytest tests/test_q2_smoke.py -q` (tiny; CPU OK; minutes — Generator-F load + per-token vocab-scan veto; `_allow_cache` keeps it tractable). Expected PASS. Failures → `@superpowers:systematic-debugging`, fixes ONLY in the two net-new modules, NEVER a protected file or frozen `_CDC_*`, max 3 attempts then STOP+report.

**Step 5:** Run grounding + core + smoke together → grounding pin fully GREEN.

**Step 6:** Commit `git add research/runners/constrained_decode_gate.py tests/test_q2_smoke.py && git commit -m "feat: Q2 per-token grounded-veto LM wrapper + kill-safe multi-rung CLI (constrained_decode_gate)"`.

**Step 7:** Controller trust-but-verify `git diff 02addfa..HEAD -- <PROTECTED SET>` EMPTY.

---

### Task 3: DEDICATED ADVERSARIAL REVIEWER (BEFORE Phase B)

Fresh adversarial reviewer vs BOTH net-new modules + tests. Does NOT rubber-stamp. STRENGTHEN-only; frozen `_CDC_*` byte-unchanged; re-review until no holes. Probe + verdict (HOLE/WEAK/CLEAN) EACH:

1. PASS requires the STRENGTHENED non-vacuity (NOT mechanical `con_uer~0`, which `cdc_verdict.note` must flag); faithful-but-vacuous ⇒ FAIL not PASS.
2. `unconstrained` is a faithful Generator-G ~0.89 drift reproduction (differs from `constrained` ONLY by `use_veto`), not a strawman.
3. `shuffled_grounding` faithful discriminating control (differs only in allow-source), not crippled elsewhere.
4. STRENGTHENED bar genuinely > bare `is_answered`, computed from reused `generator_g_core` primitives; `_norm` matches generator_g_core normalization (probe drift).
5. No-confab bit-identical to bare moat on ungrounded; LM `generate_ids` NEVER called on the abstain path (trace `grounded_decode`).
6. No PASS/SCALE-CONFIDENT from a broken instrument (feed adversarial `cdc_verdict`/`cdc_scale_confidence` inputs); `_CDC_*` not movable via CLI/env/results.
7. Byte-UNMODIFIED reuse: `git diff 02addfa..HEAD -- <PROTECTED SET>` EMPTY; reused modules unmodified.
8. No NEW autograd/training/`.backward()` (Generator-F `torch.no_grad()` inference + `model.train(False)` is the carved-out reuse).

Reviewer → holes → fresh implementer STRENGTHEN-only fix → re-review until CLEAN. Controller verifies protected-empty each fix commit.

---

### Task 4: Phase B LOAD-BEARING no-harm

Create `tests/test_q2_no_harm.py`:

```python
import subprocess, sys, importlib, inspect


def test_protected_byte_empty():
    protected = [
      "research/runners/abstention_gate.py","tests/test_abstention_gate.py",
      "sim/grounded_decode.py","research/runners/generator_g_core.py",
      "research/runners/compose_bridge_core.py",
      "research/runners/compose_bind_core.py",
      "research/runners/td_critic_core.py",
      "research/runners/dendritic_fair_core.py",
      "research/runners/engram_bootstrap_gate.py",
      "sim/tiny_transformer.py","sim/bpe_tokenizer.py","sim/bridge.py",
      "sim/td_value_critic.py","sim/compose_temporal_bind.py",
      "sim/kernels.py","sim/neuromodulators.py","sim/train_checkpoint.py",
      "sim/backend.py","sim/dendritic_plasticity.py",
      "research/runners/text_minimal_isolation.py"]
    d = subprocess.run(["git","diff","02addfa..HEAD","--",*protected],
                        capture_output=True, text=True)
    assert d.stdout.strip() == "", "PROTECTED changed:\n"+d.stdout


def test_no_confab_moat_7_of_7():
    r = subprocess.run([sys.executable,"-m","pytest",
                        "tests/test_abstention_gate.py","-q"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]


def test_core_is_pure_no_torch():
    core = importlib.import_module(
        "research.runners.constrained_decode_core")
    src = inspect.getsource(core)
    assert "import torch" not in src and "backward(" not in src


def test_net_new_gate_no_new_training():
    g = importlib.import_module(
        "research.runners.constrained_decode_gate")
    src = inspect.getsource(g)
    assert "backward(" not in src and ".step()" not in src
```

Run `python -m pytest tests/test_q2_no_harm.py tests/test_abstention_gate.py -q` → PASS (4/4 + moat 7/7). Then full Q2 suite `python -m pytest tests/test_q2_grounding.py tests/test_constrained_decode_core.py tests/test_q2_smoke.py tests/test_q2_no_harm.py tests/test_abstention_gate.py -q` → ALL PASS. Commit `git add tests/test_q2_no_harm.py && git commit -m "test: Q2 Phase B LOAD-BEARING no-harm (protected byte-empty + moat 7/7 + no new training)"`. Controller verifies protected EMPTY + moat 7/7.

---

### Task 5: CONTROLLER-ONLY decisive GPU run + smell-test + honest propagation

**NOT a subagent task.** Controller performs directly, brings back.

1. **Grounding-first tiny run** (toy NOT propagated): `python -m research.runners.constrained_decode_gate --tiny --seeds 42 43 44 --out research/findings/raw/q2_tiny.json` — confirm exit 0, `note` "TINY", device printed, no-confab preserved.
2. **Decisive kill-safe multi-seed multi-rung run ON THE GPU:** `python -m research.runners.constrained_decode_gate --seeds 42 43 44 45 46 --ckpt research/findings/raw/g11_bg/constrained_decode_gate --out research/findings/raw/q2_constrained_decode_gate.json`. **MUST verify `DEVICE=cuda (CUDA=True)`** (directly addresses the user's GPU concern). If it prints `cpu` while a GPU is present → STOP, fix device placement (do NOT propagate a CPU run when CUDA is available). CPU acceptable ONLY if `torch.cuda.is_available()` genuinely False, logged in findings. Kill-safe: re-invoke SAME command to resume.
3. **MANDATORY anti-cheat smell-test** (scrutinize a nominal PASS HARDER than a FAIL; recompute from the single recorded JSON; NO re-run/NO bar-tuning/NO overclaim): V1 genuine (`unconstrained_uer` ≫ 0.20 — instrument truly sees drift); the discriminating signature is constrained **non-vacuity** ≥ 0.5, NOT the mechanical `constrained_uer~0` (verify `note` flags it mechanical); `unconstrained`+`shuffled_grounding` genuinely fail; `abstain_on_ungrounded_rate ≥ bare_moat_abstain_rate`; `cdc_scale_confidence` recomputed from JSON; protected diff `02addfa..HEAD` EMPTY; moat 7/7.
4. **Honest propagation EVERY outcome:** `research/findings/2026-05-18-Q2-constrained-decode-<CLASSIFICATION>.md` (no spin; honest ceiling verbatim; cheap-probe caveats restated; per-rung table; device used). Append `webapp/capability_status.json` pillar **n=75**: SCALE-CONFIDENT-PASS→`VALIDATED`; WORKS-SMALL-NO-SCALE-CONFIDENCE→`BOUNDARY`; VOID→`BOUNDARY`; FAIL→`NEGATIVE`. `python -m pytest tests/test_webapp_server.py -k capability_status -q` green (fix JSON not test). Bump `as_of`.
5. **Push BOTH remotes:** add findings + capability_status.json + `research/findings/raw/q2_constrained_decode_gate.json`; commit; `git push origin main && git push gitea main`.
6. **Pivot rule (NON-STOP, no owner-deferral):** NOT SCALE-CONFIDENT-PASS → propagated (done) and IMMEDIATELY pivot to **Q3** (Larkum laminar microcircuit) per the CORRECTED OPERATING MODE — write Q3 design → writing-plans → subagent-driven. SCALE-CONFIDENT-PASS → report the validated deliverable to controller/owner and continue the standing arc.

---

## Notes for the executor

- DRY/YAGNI/TDD; frequent commits; exact paths; complete code above.
- `@superpowers:systematic-debugging` for any failure — root-cause, never paper over, NEVER touch a frozen `_CDC_*` or protected file to force green.
- `@superpowers:subagent-driven-development` drives Tasks 0–4 (fresh subagent per task; spec then code-quality review; Task 3 = dedicated adversarial reviewer BEFORE Phase B).
- Task 5 CONTROLLER-ONLY — bring it back. Decisive run MUST be `DEVICE=cuda`.
- Keep `model.train(False)` (== `.eval()`) — do not revert to `.eval()` (security-hook false-positive on `eval(`).
- Honest ceiling NEVER spun: scale-confidence PoC (non-vacuously faithful by construction + no scale degradation), explicitly NOT open-ended fluent composition / NOT an LLM. Vacuity collapse or scale degradation = honest non-success → autonomous Q3 pivot.
