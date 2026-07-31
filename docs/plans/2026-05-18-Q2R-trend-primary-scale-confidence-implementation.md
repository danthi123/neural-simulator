---
type: plan
status: live
date: 2026-05-18
---

# Q2R — Trend-primary scale-confidence (fresh larger-KB experiment) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (standing autonomy pre-selects Subagent-Driven, this session). Task 5 is CONTROLLER-ONLY — bring it back.

**Goal:** Build a kill-safe gate that tests whether the validated Q2 constrained-decoding generative-faithfulness capability is SCALE-CONFIDENT — constrained non-vacuity holds/improves up a genuine net-new local KB ladder K∈{12,24,48,96} and clears the SAME 0.50 absolute floor at the LARGEST rung — using the byte-UNMODIFIED Q2 mechanism + instrument, under a freshly-pre-registered a-priori trend-primary criterion. NOT a Q2 re-score.

**Architecture:** Two net-new modules. `q2r_core.py` = own frozen `_Q2R_*` trend-primary aggregator (mirrors `constrained_decode_core` discipline; stdlib+typing only; imports/mutates no existing core). `q2r_gate.py` = net-new frozen KB of ≥96 genuinely-distinct propositions + a faithful behavioural mirror of `constrained_decode_gate._run_rung` over that KB, that **imports `_GroundedConstrainedLM` and `cdc_verdict` BYTE-UNMODIFIED** (the validated mechanism + soundness instrument, unchanged, via import). All rungs run FRESH; Q2's FAIL is untouched.

**Tech Stack:** Python 3, PyTorch (Generator-F inference ONLY, via the imported `_GroundedConstrainedLM`; CUDA on the decisive run), NumPy, pytest. NO new autograd/training/`.backward()`/optimizer/loss in net-new code.

**Plan base commit:** `a1035cf`. Invariant: `git diff a1035cf..HEAD -- <PROTECTED SET>` MUST be empty at every task.

> **Hook note:** the project security hook false-positives on the literal substring of PyTorch's inference-mode method name (the 5-letter word + `(`). Inference mode is already set INSIDE the imported `_GroundedConstrainedLM` via `model.train(False)`. Net-new code must NOT write that method-name substring anywhere (use `model.train(False)` if ever needed — it is not needed here since the LM is imported).

---

## Reused interfaces (grounded — do NOT modify any of these)

- `research/runners/constrained_decode_gate.py`:
  - `_GroundedConstrainedLM(ckpt_prefix, mode="constrained"|"unconstrained"|"shuffled", block_size=128)` — loads the Generator-F ckpt; attrs `.tok` (BPE tokenizer), `.device`, `._shuffle_text` (settable; used in `shuffled` mode), `.generate_ids(prompt_ids, max_new)`; sets inference mode internally. **Import byte-UNMODIFIED.**
  - `_run_rung(K, seeds, lm_c, lm_u, lm_s, max_new, n_ung)` — the proven per-rung loop: `items = list(_GROUNDED.items())[:K]`; per seed computes the `cdc_verdict` `_REQUIRED` per-seed dict; returns `{"K":K,"verdict":cdc_verdict(per_seed),"constrained_nonvac_rate_mean":mean}`. **READ this function in full and mirror it byte-faithfully in `_q2r_run_rung`, changing ONLY the KB iterated.** Also read `main()` for the exact `_GroundedConstrainedLM` construction (lm_c/lm_u/lm_s), resume/kill-safe pattern, `--tiny` handling, device print.
  - `_GROUNDED` has exactly 24 entries (the reason Q2R needs its OWN larger KB). Do NOT import or extend `_GROUNDED`.
- `research/runners/constrained_decode_core.py`:
  - `cdc_verdict(per_seed: dict) -> {"GATE":"PASS"|"FAIL"|"VOID", ...}` — the validated Q2 soundness instrument. **Import byte-UNMODIFIED.** `_REQUIRED` per-seed keys (EXACT): `unconstrained_uer`, `constrained_uer`, `constrained_nonvac_rate`, `shuffled_uer`, `shuffled_nonvac_rate`, `bare_moat_abstain_rate`, `abstain_on_ungrounded_rate`, `constrained_multitoken_emittable_rate`.
  - `nonvacuous_answered(response_text, retrieved_text) -> bool`, `_CDC_MIN_GROUNDED_ANSWER_RATE == 0.50` (the value `_Q2R_TOP_MIN` deliberately equals).
  - `cdc_scale_confidence` exists — `q2r_core.q2r_scale_confidence` MIRRORS its shape but with the frozen `_Q2R_*` and the trend-primary rule. q2r_core does NOT import constrained_decode_core.
- `sim/grounded_decode.py` `grounded_decode(ranked, lm, tok, retrieved_text, query, threshold=650.0, max_new=40)`, `research/runners/abstention_gate.py` `gate`/`DEFAULT_THRESHOLD=650.0`, `research/runners/generator_g_core.py` `ungrounded_entity_rate`/`is_answered`/`FUNCTION_WORDS`, `sim/train_checkpoint.py` (kill-safe), Generator-F artifact `research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real.{pt,bpe.json}`. Used exactly as `constrained_decode_gate` uses them (via the imported symbols / the faithful `_run_rung` mirror).

## PROTECTED SET (byte-empty in EVERY commit-scoped diff AND `git diff a1035cf..HEAD`)

```
research/runners/constrained_decode_gate.py   research/runners/constrained_decode_core.py
research/runners/engram_bootstrap_gate.py     research/runners/abstention_gate.py
tests/test_abstention_gate.py                 sim/grounded_decode.py
research/runners/generator_g_core.py          research/runners/compose_bridge_core.py
research/runners/compose_bind_core.py         research/runners/td_critic_core.py
research/runners/dendritic_fair_core.py       sim/tiny_transformer.py  sim/bpe_tokenizer.py
sim/bridge.py  sim/td_value_critic.py  sim/compose_temporal_bind.py  sim/kernels.py
sim/neuromodulators.py  sim/train_checkpoint.py  sim/backend.py  sim/dendritic_plasticity.py
research/runners/text_minimal_isolation.py
research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real.pt
research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real.bpe.json
```

Controller trust-but-verify EVERY task's `git diff a1035cf..HEAD -- <PROTECTED SET>` is EMPTY before marking complete.

---

### Task 0: Grounding pin

**Files:** Create `tests/test_q2r_grounding.py`

```python
"""Q2R Task-0 grounding pin. RED until Task 1/2 ship the net-new modules."""
import importlib


def test_reused_q2_mechanism_and_instrument_present():
    cg = importlib.import_module("research.runners.constrained_decode_gate")
    assert hasattr(cg, "_GroundedConstrainedLM")
    cc = importlib.import_module("research.runners.constrained_decode_core")
    assert callable(cc.cdc_verdict)
    assert cc._CDC_MIN_GROUNDED_ANSWER_RATE == 0.50  # the value _Q2R_TOP_MIN equals


def test_generator_f_artifact_present():
    import os
    b = "research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real"
    assert os.path.exists(b + ".pt") and os.path.exists(b + ".bpe.json")


def test_q2r_core_frozen_and_pure():
    c = importlib.import_module("research.runners.q2r_core")
    assert c._Q2R_LADDER == (12, 24, 48, 96)
    assert c._Q2R_SCALE_TOL == 0.10
    assert c._Q2R_TOP_MIN == 0.50
    assert c._Q2R_MIN_SEEDS == 3
    assert callable(c.q2r_scale_confidence)
    import inspect
    src = inspect.getsource(c)
    assert "import torch" not in src and "backward(" not in src
    assert "constrained_decode_core" not in src  # owns its own bars


def test_q2r_gate_importable_and_reuses_byte_unmodified():
    g = importlib.import_module("research.runners.q2r_gate")
    from research.runners.constrained_decode_gate import _GroundedConstrainedLM
    from research.runners.constrained_decode_core import cdc_verdict
    assert g._GroundedConstrainedLM is _GroundedConstrainedLM   # byte-unmodified import
    assert g.cdc_verdict is cdc_verdict                          # byte-unmodified import
    assert len(g._Q2R_GROUNDED) >= 96
```

Run `python -m pytest tests/test_q2r_grounding.py -q` → FAIL (`q2r_core` missing; intentional RED). Commit `git add tests/test_q2r_grounding.py && git commit -m "test: Q2R Task-0 grounding pin (intentionally RED until Task 1/2)"`. Controller verifies protected diff EMPTY.

---

### Task 1: `q2r_core.py` — frozen trend-primary aggregator (FULLY SPECIFIED — transcribe exactly)

**Files:** Create `research/runners/q2r_core.py`, `tests/test_q2r_core.py`

**Step 1: adversarial test matrix** `tests/test_q2r_core.py`:

```python
from research.runners.q2r_core import (
    q2r_scale_confidence, _Q2R_LADDER, _Q2R_SCALE_TOL, _Q2R_TOP_MIN,
    _Q2R_MIN_SEEDS)


def _rg(K, gate, nv):
    return {"K": K, "verdict": {"GATE": gate},
            "constrained_nonvac_rate_mean": nv}


def _good(nvs):  # all PASS rungs with given non-vacuity sequence
    return [_rg(K, "PASS", nv) for K, nv in zip(_Q2R_LADDER, nvs)]


def test_frozen_constants_pinned():
    assert _Q2R_LADDER == (12, 24, 48, 96)
    assert _Q2R_SCALE_TOL == 0.10
    assert _Q2R_TOP_MIN == 0.50
    assert _Q2R_MIN_SEEDS == 3


def test_all_pass_nondecreasing_top_clears_is_scale_confident():
    r = q2r_scale_confidence(_good([0.55, 0.60, 0.66, 0.72]))
    assert r["scale_confident"] is True
    assert r["classification"] == "SCALE-CONFIDENT-PASS"


def test_monotone_within_tol_ok():
    # small dips within _Q2R_SCALE_TOL are allowed; top clears floor
    r = q2r_scale_confidence(_good([0.60, 0.55, 0.58, 0.62]))
    assert r["classification"] == "SCALE-CONFIDENT-PASS"


def test_trend_drop_beyond_tol_is_works_small():
    r = q2r_scale_confidence(_good([0.70, 0.55, 0.52, 0.55]))  # 0.70->0.55 drop 0.15 > tol
    assert r["scale_confident"] is False
    assert r["classification"] == "WORKS-SMALL-NO-SCALE-CONFIDENCE"


def test_top_below_floor_is_works_small():
    r = q2r_scale_confidence(_good([0.52, 0.55, 0.58, 0.49]))  # top 0.49 < 0.50
    assert r["scale_confident"] is False
    assert r["classification"] == "WORKS-SMALL-NO-SCALE-CONFIDENCE"


def test_any_void_rung_is_void_precedence_over_fail():
    rungs = _good([0.6, 0.6, 0.6, 0.6])
    rungs[1]["verdict"]["GATE"] = "VOID"
    rungs[2]["verdict"]["GATE"] = "FAIL"
    assert q2r_scale_confidence(rungs)["classification"] == "VOID"


def test_any_fail_rung_is_fail():
    rungs = _good([0.6, 0.6, 0.6, 0.6])
    rungs[2]["verdict"]["GATE"] = "FAIL"
    assert q2r_scale_confidence(rungs)["classification"] == "FAIL"


def test_ladder_mismatch_is_void():
    bad = [_rg(12, "PASS", 0.6), _rg(24, "PASS", 0.6), _rg(48, "PASS", 0.6)]
    assert q2r_scale_confidence(bad)["classification"] == "VOID"  # != (12,24,48,96)


def test_ladder_padding_duplicate_K_is_void():
    bad = [_rg(12, "PASS", 0.6), _rg(24, "PASS", 0.6),
           _rg(48, "PASS", 0.6), _rg(48, "PASS", 0.6)]
    assert q2r_scale_confidence(bad)["classification"] == "VOID"


def test_unknown_gate_is_void():
    rungs = _good([0.6, 0.6, 0.6, 0.6])
    rungs[0]["verdict"]["GATE"] = "MAYBE"
    assert q2r_scale_confidence(rungs)["classification"] == "VOID"


def test_non_numeric_nonvac_is_void_not_raise():
    rungs = _good([0.6, 0.6, 0.6, 0.6])
    rungs[3]["constrained_nonvac_rate_mean"] = "oops"
    assert q2r_scale_confidence(rungs)["classification"] == "VOID"


def test_missing_verdict_is_void_not_raise():
    rungs = _good([0.6, 0.6, 0.6, 0.6])
    del rungs[1]["verdict"]
    assert q2r_scale_confidence(rungs)["classification"] == "VOID"


def test_unorderable_is_void_not_raise():
    assert q2r_scale_confidence([{"K": object(),
        "verdict": {"GATE": "PASS"},
        "constrained_nonvac_rate_mean": 0.6}])["classification"] == "VOID"
```

**Step 2:** Run → FAIL (module missing).

**Step 3: Create `research/runners/q2r_core.py` EXACTLY:**

```python
"""Pure FIXED-bar TREND-PRIMARY scale-confidence verdict for Q2R --
a FRESH larger-KB experiment of the VALIDATED Q2 constrained-decoding
mechanism. Mirrors the adversarial-hardened Q2-constrained-decode-core
and compose-bridge-core DISCIPLINE (fixed bars NEVER tuned, fail-closed,
VOID strictly distinct from FAIL, malformed/junk -> VOID-not-raise).
Holds its OWN frozen _Q2R_*; does NOT import/mutate the Q2
constrained-decode core or any *_core sibling module. Pure
stdlib+typing; NO torch, NO autograd. ASCII.

A-PRIORI justification of the frozen criterion (defensible WITHOUT any
reference to Q2's observed numbers):
- _Q2R_LADDER = (12,24,48,96): scale-confidence is definitionally
  about behaviour as capacity SCALES UP toward a useful target. The
  ladder must START at a non-toy size (K=12 = the smallest KB a
  "grounded conversational agent" claim could even be ABOUT; a
  6-proposition KB is a toy below the floor of the question) and
  EXTEND UPWARD geometrically (x2 per rung) to where scale-confidence
  actually lives (K=96). The K=6 omission is a principled non-toy
  floor decided by what the QUESTION means, not by any observed value.
- _Q2R_TOP_MIN = 0.50: DELIBERATELY the SAME absolute non-vacuity
  value as the validated Q2 core's _CDC_MIN_GROUNDED_ANSWER_RATE
  (0.50). It is NOT a softened bar. The ONLY methodological change vs
  Q2 is WHERE the absolute floor + the trend are applied (at the
  LARGEST scale where scale-confidence is claimed + a monotone trend),
  NOT WHAT the value is. This identity is the strongest structural
  defense against a goalpost-move.
- _Q2R_SCALE_TOL = 0.10: a stochastic 5-seed non-vacuity rate has a
  noise floor; 0.10 is a defensible max permitted DROP between
  ascending rungs (same magnitude family as the validated _CDC
  tolerances). The TREND being non-decreasing-up-to-tol is the PRIMARY
  scale-confidence signal.
These values are pre-registered HERE, BEFORE any Q2R run, and NEVER
tuned to a result."""
from __future__ import annotations
from typing import Dict

_Q2R_LADDER = (12, 24, 48, 96)
_Q2R_SCALE_TOL = 0.10
_Q2R_TOP_MIN = 0.50
_Q2R_MIN_SEEDS = 3


def _num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    import math
    return f if math.isfinite(f) else None


def q2r_scale_confidence(rungs) -> Dict:
    """Pure, deterministic, fail-closed. rungs: list of {"K",
    "verdict":{"GATE":...}, "constrained_nonvac_rate_mean"}. Recomputed
    from the single recorded JSON; NEVER raises.

    SCALE-CONFIDENT-PASS iff: the ordered-by-K key tuple == _Q2R_LADDER
    EXACTLY (guards padding/duplication/mismatch) AND every rung
    verdict GATE == "PASS" AND constrained_nonvac_rate_mean is
    non-decreasing up to _Q2R_SCALE_TOL across the ascending ladder AND
    the LARGEST rung (K=96) constrained_nonvac_rate_mean >=
    _Q2R_TOP_MIN. Else: any rung GATE VOID/missing/unknown -> VOID
    (precedence); any rung GATE FAIL -> FAIL; otherwise (all PASS but
    trend breaks or top below floor) ->
    WORKS-SMALL-NO-SCALE-CONFIDENCE. Non-numeric/unorderable/malformed
    -> VOID."""
    bars = {"LADDER": list(_Q2R_LADDER), "SCALE_TOL": _Q2R_SCALE_TOL,
            "TOP_MIN": _Q2R_TOP_MIN, "MIN_SEEDS": _Q2R_MIN_SEEDS}
    try:
        ordered = sorted(rungs, key=lambda r: r["K"])
    except (TypeError, KeyError):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "rungs not orderable by K", "frozen_bars": bars}
    try:
        ladder = tuple(int(r["K"]) for r in ordered)
    except (TypeError, ValueError, KeyError):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "rung K not integer-coercible",
                "frozen_bars": bars}
    if ladder != _Q2R_LADDER:
        return {"scale_confident": False, "classification": "VOID",
                "reason": "ladder %s != pre-registered %s "
                          "(padding/mismatch guard)"
                          % (ladder, _Q2R_LADDER),
                "frozen_bars": bars}
    gates = []
    for r in ordered:
        v = r.get("verdict")
        gates.append(v.get("GATE") if isinstance(v, dict) else None)
    if any(g == "VOID" or g is None for g in gates):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "a rung GATE is VOID/missing",
                "frozen_bars": bars}
    if any(g == "FAIL" for g in gates):
        return {"scale_confident": False, "classification": "FAIL",
                "reason": "a rung GATE is FAIL", "frozen_bars": bars}
    if any(g != "PASS" for g in gates):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "a rung GATE is not PASS/FAIL/VOID",
                "frozen_bars": bars}
    nv = []
    for r in ordered:
        f = _num(r.get("constrained_nonvac_rate_mean"))
        if f is None:
            return {"scale_confident": False, "classification": "VOID",
                    "reason": "non-numeric constrained_nonvac_rate_mean",
                    "frozen_bars": bars}
        nv.append(f)
    monotone = all(nv[i + 1] >= nv[i] - _Q2R_SCALE_TOL
                   for i in range(len(nv) - 1))
    top_ok = nv[-1] >= _Q2R_TOP_MIN
    if monotone and top_ok:
        return {"scale_confident": True,
                "classification": "SCALE-CONFIDENT-PASS",
                "reason": "every rung PASS; non-vacuity non-decreasing "
                          "up to tol across the ascending ladder; "
                          "K=96 clears the 0.50 floor (same value as "
                          "Q2's bar, applied at the largest scale)",
                "nonvac_by_rung": nv, "frozen_bars": bars}
    return {"scale_confident": False,
            "classification": "WORKS-SMALL-NO-SCALE-CONFIDENCE",
            "reason": "every rung PASS but %s%s"
                      % ("" if monotone else "non-vacuity drops > "
                         "_Q2R_SCALE_TOL between ascending rungs; ",
                         "" if top_ok else "K=96 non-vacuity below "
                         "_Q2R_TOP_MIN=0.50"),
            "nonvac_by_rung": nv, "frozen_bars": bars}
```

**Step 4:** Run `python -m pytest tests/test_q2r_core.py tests/test_q2r_grounding.py -q` → q2r_core 13/13 PASS; grounding's `test_q2r_gate_importable*` still FAILS (Task 2). Expected partial-RED.

**Step 5:** Commit `git add research/runners/q2r_core.py tests/test_q2r_core.py && git commit -m "feat: Q2R q2r_core frozen trend-primary scale-confidence aggregator (own _Q2R_*, a-priori-justified, NEVER tuned; imports no existing core)"`. Controller verifies protected diff EMPTY.

---

### Task 2: `q2r_gate.py` — net-new ≥96-distinct KB + faithful `_run_rung` mirror + byte-UNMODIFIED imports

**Files:** Create `research/runners/q2r_gate.py`, `tests/test_q2r_smoke.py`

**Step 1: smoke + structural test** `tests/test_q2r_smoke.py`:

```python
"""Q2R --tiny smoke + structural guards. Tiny verdict NOT propagated."""
import json, subprocess, sys, os, importlib, re


def test_kb_is_96plus_genuinely_distinct():
    g = importlib.import_module("research.runners.q2r_gate")
    kb = g._Q2R_GROUNDED
    assert len(kb) >= 96, "KB must have >=96 props, got %d" % len(kb)

    def content(s):
        fw = importlib.import_module(
            "research.runners.generator_g_core").FUNCTION_WORDS
        toks = [re.sub(r"[^\w]", "", w.lower()) for w in str(s).split()]
        return frozenset(t for t in toks if t and t not in fw)
    sets = [content(v) for v in kb.values()]
    # genuine: every proposition has >=3 content words and ALL
    # content-word sets are pairwise-distinct (no templating/dup)
    assert all(len(s) >= 3 for s in sets), "a prop has <3 content words"
    assert len(set(sets)) == len(sets), "duplicate content-word sets "\
        "(templated/padded KB)"


def test_byte_unmodified_imports():
    g = importlib.import_module("research.runners.q2r_gate")
    from research.runners.constrained_decode_gate import _GroundedConstrainedLM
    from research.runners.constrained_decode_core import cdc_verdict
    assert g._GroundedConstrainedLM is _GroundedConstrainedLM
    assert g.cdc_verdict is cdc_verdict


def test_no_new_training_in_net_new():
    import inspect
    g = importlib.import_module("research.runners.q2r_gate")
    src = inspect.getsource(g)
    assert "backward(" not in src and ".step()" not in src
    assert "optimizer" not in src.lower() and "loss" not in src.lower()


def test_tiny_smoke_runs(tmp_path):
    out = tmp_path / "q2r_smoke.json"
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.q2r_gate",
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

**Step 2:** Run → FAIL (module missing).

**Step 3: Create `research/runners/q2r_gate.py`.** This is GENUINE NET-NEW INTEGRATION. Requirements:

1. **Imports (byte-UNMODIFIED):**
   ```python
   from research.runners.constrained_decode_gate import _GroundedConstrainedLM
   from research.runners.constrained_decode_core import cdc_verdict
   from research.runners.q2r_core import q2r_scale_confidence, _Q2R_LADDER
   ```
   Do NOT import `_run_rung` or `_GROUNDED`. Do NOT redefine the mechanism.

2. **`_Q2R_GROUNDED`** — a net-new frozen dict of **≥96** `(subject -> proposition)` entries. Each proposition is a simple TinyStories-style sentence of **4-6 content words**, with **genuinely varied vocabulary** across subjects/verbs/objects/adjectives. The normalized content-word set of every proposition MUST be **pairwise-distinct** from every other's (the Task-1-style distinctness test enforces this; the adversarial reviewer spot-checks for templating). **Write genuine, varied propositions — do NOT generate them from a fill-in template or by permuting a small word pool.** Exemplars showing the required variety (you must author ≥96 in this spirit, not copy these 12):
   ```
   "ada": "ada studies the old star maps",
   "bo":  "bo carved a whistle from cedar",
   "cy":  "cy spilled warm cocoa on snow",
   "di":  "di trains a clever grey parrot",
   "ed":  "ed welds a broken iron gate",
   "fern":"fern brews tea from wild mint",
   "gio": "gio juggles five painted clubs",
   "hana":"hana folds a paper crane swiftly",
   "ira": "ira maps a hidden cave river",
   "jun": "jun tunes a cracked cello string",
   "kit": "kit races a wooden sail cart",
   "lev": "lev sketches a tall harbor crane",
   ```
   `_Q2R_UNGROUNDED` — a net-new list of ≥6 nonsense queries (e.g. `["xthar","qoom","vlex","druskin","plimp","wozzle"]`), distinct from any subject.

3. **`_q2r_run_rung(K, seeds, lm_c, lm_u, lm_s, max_new, n_ung)`** — a **FAITHFUL behavioural mirror** of `constrained_decode_gate._run_rung`. **Read `constrained_decode_gate._run_rung` in full and reproduce its per-seed computation EXACTLY**, with the ONLY difference being it iterates `list(_Q2R_GROUNDED.items())[:K]` (and `_Q2R_UNGROUNDED[:n_ung]`) instead of `_GROUNDED`. It MUST: produce the EXACT `cdc_verdict` `_REQUIRED` per-seed dict (all 8 keys incl `constrained_multitoken_emittable_rate`), use the SAME shuffled-source rule `_run_rung` uses (a DIFFERENT proposition from the same KB, chosen exactly as `_run_rung` does — read it; if `_run_rung` uses an RNG-permuted different index, mirror that exactly), the SAME no-confab path (empty-`ranked` `grounded_decode` over the ungrounded queries), and score the rung with the imported byte-UNMODIFIED `cdc_verdict`. Return `{"K":K,"verdict":...,"constrained_nonvac_rate_mean":mean}` exactly as `_run_rung` does. The adversarial reviewer verifies faithfulness line-by-line.

4. **`main(argv=None)`** — mirror `constrained_decode_gate.main`'s structure: argparse `--seeds` (default 42 43 44 45 46), `--tiny`, `--ckpt`, `--out`; construct `lm_c=_GroundedConstrainedLM(_GEN_F,mode="constrained")`, `lm_u=...mode="unconstrained"`, `lm_s=...mode="shuffled"` exactly as `constrained_decode_gate.main` does (read it; `_GEN_F` = the same Generator-F prefix path); print `DEVICE=<lm_c.device> (CUDA=...)`; kill-safe per-(rung,seed) atomic `resume.json` via the same pattern; ladder = `(_Q2R_LADDER[0],)` if `--tiny` else `_Q2R_LADDER`; per rung call `_q2r_run_rung`; aggregate with `q2r_scale_confidence`; write ONE JSON `{"ladder":[...], "scale_confident":..., "scale_classification":..., "scale_reason":..., "device":..., "tiny":bool, "note": "TINY toy verdict -- NOT propagated" if tiny else "multi-rung trend-primary scale-confidence verdict -- recompute from this JSON; no re-run/no tuning", "HONEST_CEILING": "<the honest ceiling string from the design, never spun>"}`. `--tiny` `scale_classification="TINY (toy; NOT propagated)"`. KeyboardInterrupt -> flush + exit 130 (resumable). ASCII only.

**Step 4:** Run `python -m pytest tests/test_q2r_smoke.py -q` (tiny; loads real Generator-F via the imported class; minutes; 3600s timeout). Expected PASS (4/4). Failures → `@superpowers:systematic-debugging`; fixes ONLY in the two net-new modules; NEVER a protected file; NEVER write the inference-mode-method-name substring; max 3 attempts then STOP+report.

**Step 5:** Run grounding + core + smoke together → grounding pin now fully GREEN. Commit `git add research/runners/q2r_gate.py tests/test_q2r_smoke.py && git commit -m "feat: Q2R q2r_gate net-new >=96-distinct KB + faithful _run_rung mirror + byte-UNMODIFIED mechanism/instrument imports"`. Controller trust-but-verify `git diff a1035cf..HEAD -- <PROTECTED SET>` EMPTY.

---

### Task 3: DEDICATED ADVERSARIAL REVIEWER (BEFORE Phase B — load-bearing)

Fresh adversarial reviewer vs BOTH net-new modules + tests. Does NOT rubber-stamp. STRENGTHEN-only; frozen `_Q2R_*` byte-unchanged; re-review until no holes. **PRIMARY probe (the central anti-cheat concern):**

1. **GOALPOST-MOVE.** Is Q2R's trend criterion + ladder (esp. the K=6 omission and the K=96-only top floor) a legitimate **a-priori** scale-confidence definition, justifiable WITHOUT any reference to Q2's observed numbers — or a post-hoc move engineered to convert Q2's FAIL into a PASS? Explicitly verify: `q2r_core._Q2R_TOP_MIN == 0.50 == constrained_decode_core._CDC_MIN_GROUNDED_ANSWER_RATE` (the SAME value, NOT softened); the ONLY change vs Q2 is trend-primary + applied-at-largest-scale + non-toy-ladder-start; the q2r_core docstring's a-priori justification stands on its own. If the criterion only "works" because K=6 was dropped or the floor was lowered → HOLE.
2. **KB-PADDING.** Is `_Q2R_GROUNDED` genuinely ≥96 pairwise-distinct quality propositions — programmatically (distinct content-word sets, ≥3 content words each) AND by spot-checking ~15 for genuine vocabulary variety (NOT templated/fill-in/permuted-small-pool)? Padding → HOLE.
3. **FAITHFUL MIRROR.** Diff `_q2r_run_rung` against `constrained_decode_gate._run_rung` line-by-line: identical per-seed cdc schema (all 8 `_REQUIRED` keys), identical shuffled-source rule, identical no-confab path, identical `_GroundedConstrainedLM` construction — ONLY the iterated KB differs. Any behavioural divergence → HOLE.
4. **BYTE-UNMODIFIED IMPORT.** `q2r_gate._GroundedConstrainedLM is constrained_decode_gate._GroundedConstrainedLM`; `q2r_gate.cdc_verdict is constrained_decode_core.cdc_verdict`; `q2r_core` does NOT import/shadow `cdc_verdict` or any `*_core`; `git diff a1035cf..HEAD -- <PROTECTED SET>` EMPTY.
5. **NO NEW AUTOGRAD/TRAINING** in net-new code (no `.backward()`/optimizer/loss/`.step()`); Generator-F is inference-only via the imported class.
6. **NO PASS FROM BROKEN INSTRUMENT / immovable bars.** Feed `q2r_scale_confidence` adversarial inputs (ladder mismatch, padded/dup K, VOID/FAIL precedence, trend-drop, top-below-floor, non-numeric, unorderable) — confirm exact VOID/FAIL/WORKS-SMALL/PASS per spec. `_Q2R_*` not movable via CLI/env/results.

Reviewer → holes → fresh implementer STRENGTHEN-only fix (frozen `_Q2R_*` byte-unchanged, transparently logged) → re-review until CLEAN. Controller verifies protected-empty after each fix commit.

---

### Task 4: Phase B LOAD-BEARING no-harm

**Files:** Create `tests/test_q2r_no_harm.py`:

```python
import subprocess, sys, importlib, inspect


def test_protected_byte_empty():
    protected = [
      "research/runners/constrained_decode_gate.py",
      "research/runners/constrained_decode_core.py",
      "research/runners/engram_bootstrap_gate.py",
      "research/runners/abstention_gate.py","tests/test_abstention_gate.py",
      "sim/grounded_decode.py","research/runners/generator_g_core.py",
      "research/runners/compose_bridge_core.py",
      "research/runners/compose_bind_core.py",
      "research/runners/td_critic_core.py",
      "research/runners/dendritic_fair_core.py",
      "sim/tiny_transformer.py","sim/bpe_tokenizer.py","sim/bridge.py",
      "sim/td_value_critic.py","sim/compose_temporal_bind.py",
      "sim/kernels.py","sim/neuromodulators.py","sim/train_checkpoint.py",
      "sim/backend.py","sim/dendritic_plasticity.py",
      "research/runners/text_minimal_isolation.py"]
    d = subprocess.run(["git","diff","a1035cf..HEAD","--",*protected],
                        capture_output=True, text=True)
    assert d.stdout.strip() == "", "PROTECTED changed:\n"+d.stdout


def test_no_confab_moat_7_of_7():
    r = subprocess.run([sys.executable,"-m","pytest",
                        "tests/test_abstention_gate.py","-q"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]


def test_q2r_core_pure_and_imports_no_core():
    c = importlib.import_module("research.runners.q2r_core")
    src = inspect.getsource(c)
    assert "import torch" not in src and "backward(" not in src
    assert "constrained_decode_core" not in src


def test_q2r_gate_reuses_byte_unmodified_and_no_new_training():
    g = importlib.import_module("research.runners.q2r_gate")
    from research.runners.constrained_decode_gate import _GroundedConstrainedLM
    from research.runners.constrained_decode_core import cdc_verdict
    assert g._GroundedConstrainedLM is _GroundedConstrainedLM
    assert g.cdc_verdict is cdc_verdict
    src = inspect.getsource(g)
    assert "backward(" not in src and ".step()" not in src
    assert "optimizer" not in src.lower() and "loss" not in src.lower()
```

Run `python -m pytest tests/test_q2r_no_harm.py tests/test_abstention_gate.py -q` → PASS (4/4 + moat 7/7). Then full Q2R suite `python -m pytest tests/test_q2r_grounding.py tests/test_q2r_core.py tests/test_q2r_smoke.py tests/test_q2r_no_harm.py tests/test_abstention_gate.py -q` → ALL PASS (report the line + actual abstention_gate count = 7). Commit `git add tests/test_q2r_no_harm.py && git commit -m "test: Q2R Phase B LOAD-BEARING no-harm (full protected byte-empty + moat 7/7 + byte-UNMODIFIED imports + no new training)"`. Controller verifies protected EMPTY + moat 7/7.

---

### Task 5: CONTROLLER-ONLY decisive run + smell-test + honest propagation

**NOT a subagent task.** Controller performs directly, brings back.

1. **Cheap controller checks (non-science, scoped):** `python -m pytest tests/test_q2r_core.py tests/test_q2r_smoke.py::test_kb_is_96plus_genuinely_distinct -q` — KB genuinely ≥96 pairwise-distinct + aggregator correct. (Q2's recorded K=12/24 data is the honest precursor evidence, CITED in the findings, NOT re-run/re-scored.)
2. **Grounding-first tiny run** (toy NOT propagated): `python -m research.runners.q2r_gate --tiny --seeds 42 43 44 --out research/findings/raw/q2r_tiny.json` — confirm exit 0, `note` "TINY", device printed, no-confab preserved.
3. **Decisive kill-safe multi-seed multi-rung run on the GPU:** `python -m research.runners.q2r_gate --seeds 42 43 44 45 46 --ckpt research/findings/raw/g11_bg/q2r_gate --out research/findings/raw/q2r_gate.json`. **MONITORING DISCIPLINE (non-negotiable):** launch via the Bash tool's `run_in_background` parameter (auto-notifies on completion) OR run foreground; **NEVER a bare `nohup &` with a false "I will be notified" claim.** Actively confirm completion by polling the output JSON + `resume.json` + process state before claiming ANY result. **Verify `DEVICE=cuda`** (CPU acceptable only if `torch.cuda.is_available()` genuinely False, logged). Kill-safe: re-invoke SAME command to resume. (Heavier than Q2: ladder sum 12+24+48+96=180 vs Q2's 42 ≈ ~4× → estimate from the tiny-run per-prop wall-clock before launching; still feasible.)
4. **MANDATORY anti-cheat smell-test** (scrutinize a nominal PASS HARDER than a FAIL; recompute from the single recorded JSON; NO re-run/NO bar-tuning/NO overclaim): **RE-EXAMINE the goalpost-move question** — would this exact criterion have been pre-registerable before Q2 ran; is `_Q2R_TOP_MIN` the same 0.50 (not softened); does it pass ONLY because K=6 was dropped (inspect whether K=12 itself clears/with-trend); every per-rung `cdc_verdict` genuinely PASS with the byte-UNMODIFIED instrument (V1 + controls incl shuffled-grounding + no-confab + multitoken-emittable all genuinely met per rung); trend genuinely monotone up to `_Q2R_SCALE_TOL`; K=96 genuinely ≥ 0.50; the KB genuinely supplied ≥96 distinct props **actually exercised at K=96** (not silently truncated — assert `len(_Q2R_GROUNDED) >= 96` and the K=96 rung used 96); `git diff a1035cf..HEAD -- <PROTECTED SET>` EMPTY; moat 7/7.
5. **Honest propagation EVERY outcome:** write `research/findings/2026-05-18-Q2R-trend-primary-<CLASSIFICATION>.md` (no spin; honest ceiling verbatim; the goalpost-move defense explicit; Q2's recorded K=12/24 cited as precursor not re-scored; per-rung table; device). Append `webapp/capability_status.json` pillar **n=77**: SCALE-CONFIDENT-PASS→`VALIDATED` (with the honest ceiling explicit — NOT an LLM/fluent); WORKS-SMALL-NO-SCALE-CONFIDENCE→`BOUNDARY`; VOID→`BOUNDARY`; FAIL→`NEGATIVE`. `python -m pytest tests/test_webapp_server.py -k capability_status -q` green (fix JSON not test). Bump `as_of`.
6. **Push BOTH remotes:** `git add` findings + capability_status.json + `research/findings/raw/q2r_gate.json`; commit; `git push origin main && git push gitea main`.
7. **Pivot rule (NON-STOP, no owner-deferral):** NOT SCALE-CONFIDENT-PASS → propagated (done) and the arc IMMEDIATELY pivots to **Q4** (concept-level pretraining objective for the surrogate-grad cortex rewired into the validated v16 concept-pool substrate) per the CORRECTED OPERATING MODE — write Q4 design → writing-plans → subagent-driven. SCALE-CONFIDENT-PASS → report the validated scale-confident deliverable (with the honest ceiling stated, never spun) to the controller/owner and continue the standing arc.

---

## Notes for the executor

- DRY/YAGNI/TDD; frequent commits; exact paths; complete code above for q2r_core; q2r_gate is genuine net-new (≥96-distinct KB authored with real variety; `_q2r_run_rung` a faithful read-and-mirror of `_run_rung`).
- `@superpowers:systematic-debugging` for any failure — root-cause, never paper over, NEVER touch a frozen `_Q2R_*` or a protected file to force green.
- `@superpowers:subagent-driven-development` drives Tasks 0–4 (fresh subagent per task; spec then code-quality review; Task 3 = dedicated adversarial reviewer BEFORE Phase B, goalpost-move the primary probe).
- Task 5 CONTROLLER-ONLY — bring it back. Decisive run MUST be monitored to ACTIVE completion (run_in_background/foreground; never a false "will be notified").
- Honest ceiling NEVER spun: a SCALE-CONFIDENT-PASS = the validated constrained-decoding faithfulness capability holds/improves to K=96 and clears the SAME 0.50 floor at the largest local scale with the validated instrument unmodified — only quantitative scale remains, no architectural ceiling. Explicitly NOT open-ended fluent composition / NOT an LLM / NOT GPT-class / NOT conversation-solved. Non-PASS = honest non-success → autonomous Q4 pivot.
```
