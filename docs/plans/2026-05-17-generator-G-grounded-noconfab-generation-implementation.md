# Generator-G — Grounded No-Confabulation Generation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task (continuous autonomous arc; do NOT stop to ask between tasks — user authorized a week of autonomous work 2026-05-17).

**Goal:** Decide, with a pre-registered FIXED-bar multi-seed gate whose LOAD-BEARING criterion is *no-confabulation PRESERVED*, whether Generator-F's validated coherent generation can be added to the validated no-confab moat WITHOUT destroying it — the honest culmination, or the decision-relevant terminus.

**Architecture:** The validated no-confab moat (`abstention_gate.gate`, "gate 650", byte-UNMODIFIED) decides answer-vs-abstain FIRST; the trained Generator-F TinyGPT generates ONLY when grounded, decoded faithfulness-constrained (greedy) conditioned on the retrieved proposition. No-confab is preserved BY CONSTRUCTION (the LM never sees an ungrounded query); the genuinely-open question is grounding *faithfulness* at the small-LM ceiling (the probe showed naive conditioning renames Max->Bob).

**Tech Stack:** PyTorch (trained Generator-F ckpt reused, NO retrain), numpy/stdlib. CPU-fast (no training in the decisive slice). The decisive slice isolates the COMPOSITION via a frozen deterministic grounded source — the G.20 retrieval is ALREADY separately multi-seed-validated (90% multi-tag, gate-650 pinned); re-running it is not Generator-G's question. Full validated-G.20-retrieval wiring = a noted later increment (YAGNI here).

**Validated APIs reused UNMODIFIED (DRY — do NOT reimplement/modify):**
- `research.runners.abstention_gate`: `DEFAULT_THRESHOLD=650.0`, `abstain(top_conf, threshold)->bool` (True iff `top_conf<=threshold`), `gate(ranked, threshold)->top tuple|None` (None==abstain). The validated no-confab moat; pinned by `tests/test_abstention_gate.py`. **byte-UNMODIFIED.**
- `sim.tiny_transformer.TinyGPT` + the trained Generator-F checkpoint `research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real.pt` (state under key `"model"`) + `.bpe.json` (the BPE) at d_model 256 / n_layer 4 / n_head 4 / block_size 128 (the FIXED Generator-F config). `sim.bpe_tokenizer.BPETokenizer`.

**MUST NOT touch (LOAD-BEARING no-harm):** `research/runners/abstention_gate.py` (the validated moat — the project's distinctive contribution; MUST stay byte-identical and `tests/test_abstention_gate.py` green), `research/runners/song_g1_core.py`, `research/runners/subword_lm_gate_core.py`, `sim/tiny_transformer.py`, `sim/bridge.py`, `research/runners/g20_*`, any validated module. Generator-G is PURELY ADDITIVE; NO new GLOBAL bar (its own frozen `_GG_*` constants live in its own module).

**Anti-cheat (non-negotiable):** (a) the abstain path MUST never invoke the LM (no-confab by construction) — Task 1 pins it with a spy LM that raises if called. (b) The genuinely-open decisive bar is grounding faithfulness (`ungrounded_entity_rate <= 0.20` — the Max->Bob catcher). (c) MANDATORY post-run smell-test: scrutinize a PASS HARDER than a FAIL — read the actual ungrounded transcripts (every one MUST be ABSTAIN) + grounded transcripts (did it rename entities?); verify grounded-answer-rate genuinely >0.5 (not trivially always-abstain). NO overclaim (small-LM ceiling, NOT an LLM); NO bar-tuning; recompute from recorded data. Honest expectation up front: the probe indicates faithfulness is HARD at this ceiling — a FAIL is the decision-relevant terminus (the two validated assets stay SEPARATE), a PASS is the honest culmination.

**ASCII-only prints (Windows cp1252). Commit after every task. Push both remotes (`origin`,`gitea`) after each phase.**

---

### Task 0: Falsify-cheaply grounding pin (green after Task 3)

**Files:** Create `tests/test_generator_g_grounding.py`

```python
"""Grounding: the generator_g_gate pipeline TURNS end-to-end at a
tiny zero-network config and produces an interpretable verdict.
(Faithfulness-is-hard already grounded by the conditioning probe.)
Green after Task 3."""
import subprocess
import sys
import json
import pytest


def test_generator_g_gate_pipeline_turns(tmp_path):
    out = str(tmp_path / "g.json")
    ck = str(tmp_path / "g.ckpt")
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.generator_g_gate",
         "--seeds", "42,43,44", "--tiny", "--out", out,
         "--ckpt", ck],
        capture_output=True, text=True, timeout=600)
    if r.returncode == 2 and "NOT RUNNABLE" in r.stdout:
        pytest.skip("trained Generator-F ckpt absent in this env")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = json.loads(open(out, encoding="utf-8").read())
    assert d["n_seeds"] == 3 and "aggregate_verdict" in d
    for s in d["per_seed"]:
        assert "verdict" in s and "transcripts" in s
```

**Step 2:** `pytest tests/test_generator_g_grounding.py -q` -> FAIL (module missing — green after Task 3; it IS the Task-3 gate).
**Step 5: Commit** `git add tests/test_generator_g_grounding.py && git commit -m "test(Generator-G): falsify-cheaply grounding pin (pipeline turns) -- green after Task 3"`

---

## PHASE A — pure-logic CPU-TDD

### Task 1: Pure grounded-decode policy (abstain path NEVER calls the LM — LOAD-BEARING)

**Files:** Create `sim/grounded_decode.py`; Test `tests/test_grounded_decode.py`

**Step 1: Write the failing tests**

```python
import pytest
from sim.grounded_decode import grounded_decode


class _SpyLM:
    """Raises if generation is attempted -- proves the abstain path
    NEVER invokes the LM (no-confab BY CONSTRUCTION)."""
    def __init__(self):
        self.called = False
    def __call__(self, *a, **k):
        self.called = True
        raise AssertionError("LM invoked on the abstain path!")


class _EchoLM:
    """Deterministic stand-in: 'generates' by echoing the retrieved
    prompt ids back (faithful by construction) so the policy wiring
    is testable without torch."""
    def generate_ids(self, prompt_ids, max_new):
        return list(prompt_ids[:max_new])


class _Tok:
    def encode(self, s):
        return [ord(c) % 50 for c in s]
    def decode(self, ids):
        return "".join(chr(65 + (i % 26)) for i in ids)


def test_abstain_path_never_touches_the_lm():
    spy = _SpyLM()
    # ranked top confidence 100 <= 650 -> moat abstains
    r = grounded_decode([("dog", 100.0, "t1")], spy, _Tok(),
                        retrieved_text="dog is big",
                        query="what is dog", threshold=650.0)
    assert r["abstained"] is True and r["text"] is None
    assert spy.called is False              # LOAD-BEARING


def test_empty_ranked_abstains_no_lm():
    spy = _SpyLM()
    r = grounded_decode([], spy, _Tok(), retrieved_text="",
                        query="q", threshold=650.0)
    assert r["abstained"] is True and spy.called is False


def test_grounded_path_decodes_conditioned_on_retrieved():
    # top confidence 900 > 650 -> grounded -> LM IS used, conditioned
    # on the retrieved proposition text.
    r = grounded_decode([("dog", 900.0, "t1")], _EchoLM(), _Tok(),
                        retrieved_text="dog is big",
                        query="what is dog", threshold=650.0,
                        max_new=8)
    assert r["abstained"] is False
    assert isinstance(r["text"], str) and len(r["text"]) > 0
    assert r["retrieved"] == "dog is big"


def test_threshold_boundary_is_abstain():
    spy = _SpyLM()
    # exactly == threshold abstains (abstain() is <=)
    r = grounded_decode([("x", 650.0, "t")], spy, _Tok(),
                        retrieved_text="x y", query="q",
                        threshold=650.0)
    assert r["abstained"] is True and spy.called is False
```

**Step 2:** `pytest tests/test_grounded_decode.py -q` -> FAIL (module missing).

**Step 3: Implement** `sim/grounded_decode.py` (copy precisely):

```python
"""Generator-G grounded-decode policy. The validated no-confab moat
(research.runners.abstention_gate.gate, byte-UNMODIFIED, 'gate 650')
decides answer-vs-abstain FIRST; the fluent LM is invoked ONLY on the
grounded path, conditioned on the retrieved proposition. No-confab is
preserved BY CONSTRUCTION: on the abstain path the LM object is never
touched. Pure policy; the decode delegates to a duck-typed `lm`
(TinyGPT-backed in the runner; a stand-in in tests). ASCII only."""
from __future__ import annotations


def grounded_decode(ranked, lm, tok, retrieved_text, query,
                    threshold=650.0, max_new=40, temperature=0.0):
    """ranked: list[(concept, rate, tag)] desc (validated retrieval
    output). Returns {abstained, text, retrieved}. The validated moat
    decides FIRST; the LM is touched ONLY when grounded."""
    from research.runners.abstention_gate import gate
    top = gate(ranked, threshold)            # validated moat; None=abstain
    if top is None:
        return {"abstained": True, "text": None,
                "retrieved": retrieved_text}
    # GROUNDED: faithfulness-constrained decode conditioned on the
    # retrieved proposition. temperature 0.0 == greedy/argmax ==
    # maximally faithful (the probe showed temp-1 free sampling
    # renames entities).
    prompt_ids = tok.encode(retrieved_text)
    gen_ids = lm.generate_ids(prompt_ids, int(max_new))
    return {"abstained": False,
            "text": tok.decode(gen_ids),
            "retrieved": retrieved_text}
```

**Step 4:** `pytest tests/test_grounded_decode.py -q` -> all 4 PASS. The spy-LM-never-called tests are LOAD-BEARING (the no-confab-by-construction guarantee) — root-cause any failure WITHOUT weakening; STOP+report a genuine spec contradiction (do NOT fake-pass).

**Step 5: Commit**
```bash
git add sim/grounded_decode.py tests/test_grounded_decode.py
git commit -m "feat(Generator-G): pure grounded-decode policy (validated moat gates FIRST; abstain path never touches the LM -- no-confab by construction)"
```

---

### Task 2: Pure FIXED-bar no-confab-preservation + faithfulness core

**Files:** Create `research/runners/generator_g_core.py`; Test `tests/test_generator_g_core.py`

**Step 1: Write the failing tests** (adversarial; the FIXED bars are load-bearing)

```python
from research.runners.generator_g_core import (
    ungrounded_entity_rate, gg_verdict, gg_aggregate_multiseed,
    _GG_UNGROUNDED_ENTITY_MAX, _GG_MIN_GROUNDED_ANSWER_RATE,
    _GG_MIN_SEEDS, FUNCTION_WORDS,
)


def test_ungrounded_entity_rate_catches_renamed_entity():
    # retrieved "max is a big dog" ; response renames max->bob
    r = ungrounded_entity_rate("bob is a big dog",
                               "max is a big dog", FUNCTION_WORDS)
    assert r > 0.0                       # 'bob' is ungrounded
    # faithful echo -> 0 ungrounded content
    assert ungrounded_entity_rate("max is a big dog",
                                  "max is a big dog",
                                  FUNCTION_WORDS) == 0.0


def test_verdict_passes_only_when_all_three_bars_met():
    v = gg_verdict(abstain_on_ungrounded_rate=1.0,
                   bare_moat_abstain_rate=1.0,
                   grounded_answer_rate=0.9,
                   mean_ungrounded_entity_rate=0.05,
                   has_ungrounded_control=True)
    assert v["GATE"] == "PASS"


def test_no_confab_regression_is_fail():
    # fluent layer abstains LESS than the bare moat -> no-confab
    # REGRESSED -> FAIL (the load-bearing bar)
    v = gg_verdict(0.80, 1.0, 0.9, 0.0, True)
    assert v["GATE"] == "FAIL" and v["no_confab_preserved"] is False


def test_trivial_always_abstain_is_fail():
    # abstains on everything (grounded_answer_rate 0) -> FAIL
    v = gg_verdict(1.0, 1.0, 0.0, 0.0, True)
    assert v["GATE"] == "FAIL"


def test_unfaithful_generation_is_fail():
    # renames entities (ungrounded-entity-rate over bar) -> FAIL
    v = gg_verdict(1.0, 1.0, 0.9, 0.55, True)
    assert v["GATE"] == "FAIL" and v["grounded_faithful"] is False


def test_missing_ungrounded_control_is_fail_closed():
    v = gg_verdict(1.0, 1.0, 0.9, 0.0, has_ungrounded_control=False)
    assert v["GATE"] == "FAIL"


def test_results_cannot_move_fixed_bars():
    assert (_GG_UNGROUNDED_ENTITY_MAX, _GG_MIN_GROUNDED_ANSWER_RATE,
            _GG_MIN_SEEDS) == (0.20, 0.5, 3)
    b = gg_verdict(1.0, 1.0, 1.0, 0.0, True)
    assert b["bars"] == {"ungrounded_entity_max": 0.20,
                         "min_grounded_answer_rate": 0.5}


def test_multiseed_requires_3_all_pass_each_has_both_probes():
    P = {"GATE": "PASS", "n_grounded": 4, "n_ungrounded": 4}
    F = {"GATE": "FAIL", "n_grounded": 4, "n_ungrounded": 4}
    Z = {"GATE": "PASS", "n_grounded": 0, "n_ungrounded": 4}
    assert gg_aggregate_multiseed([P, P, P])["GATE"] == "PASS"
    assert gg_aggregate_multiseed([P, F, P])["GATE"] == "FAIL"
    assert gg_aggregate_multiseed([P, P])["GATE"] == "FAIL"   # <3
    assert gg_aggregate_multiseed([P, P, Z])["GATE"] == "FAIL" # no grounded probe
```

**Step 2:** `pytest tests/test_generator_g_core.py -q` -> FAIL (module missing).

**Step 3: Implement** `research/runners/generator_g_core.py`:

```python
"""Pure FIXED-bar no-confab-preservation + grounding-faithfulness
verdict for Generator-G. Mirrors the song_g1_core/subword_lm_gate_core
DISCIPLINE (fixed bars, never tuned, multi-seed, fail-closed) but
holds Generator-G's OWN frozen constants -- does NOT import/modify
song_g1_core or subword_lm_gate_core. Pure numpy/stdlib;
CPU-unit-testable."""
from __future__ import annotations
from typing import Dict, List

# PRE-REGISTERED FIXED bars (frozen; results provably cannot move
# them; NEVER tuned). The no-confab-PRESERVED bar is relational
# (>= the frozen bare-moat rate) so it has no separate constant.
_GG_UNGROUNDED_ENTITY_MAX = 0.20      # grounding faithfulness (Max->Bob catcher)
_GG_MIN_GROUNDED_ANSWER_RATE = 0.5    # anti-trivial-always-abstain
_GG_MIN_SEEDS = 3

FUNCTION_WORDS = frozenset((
    "a an the is are was were be been to of in on at and or but it "
    "its he she they them his her their this that with for as you i "
    "we my your we had has have did do does not no yes very so then "
    "one day").split())


def ungrounded_entity_rate(response_text, retrieved_text,
                           function_words=FUNCTION_WORDS) -> float:
    """Fraction of response CONTENT words (not function words) that do
    NOT appear in the retrieved proposition. High == the LM invented
    ungrounded content (renamed entities / confabulated)."""
    ret = set(w.lower() for w in str(retrieved_text).split())
    resp = [w.lower() for w in str(response_text).split()]
    content = [w for w in resp if w not in function_words]
    if not content:
        return 0.0
    ungrounded = sum(1 for w in content if w not in ret)
    return ungrounded / len(content)


def gg_verdict(abstain_on_ungrounded_rate, bare_moat_abstain_rate,
               grounded_answer_rate, mean_ungrounded_entity_rate,
               has_ungrounded_control) -> Dict:
    # LOAD-BEARING: the fluent layer must NOT abstain less than the
    # bare validated moat on ungrounded queries (no-confab PRESERVED).
    # fail-closed without the ungrounded control.
    no_confab = (bool(has_ungrounded_control)
                 and abstain_on_ungrounded_rate
                 >= bare_moat_abstain_rate - 1e-9)
    not_trivial = grounded_answer_rate >= _GG_MIN_GROUNDED_ANSWER_RATE
    faithful = mean_ungrounded_entity_rate <= _GG_UNGROUNDED_ENTITY_MAX
    gate = bool(no_confab and not_trivial and faithful)
    return {
        "GATE": "PASS" if gate else "FAIL",
        "no_confab_preserved": bool(no_confab),
        "answers_grounded_not_trivial": bool(not_trivial),
        "grounded_faithful": bool(faithful),
        "abstain_on_ungrounded_rate": float(abstain_on_ungrounded_rate),
        "bare_moat_abstain_rate": float(bare_moat_abstain_rate),
        "grounded_answer_rate": float(grounded_answer_rate),
        "mean_ungrounded_entity_rate":
            float(mean_ungrounded_entity_rate),
        "bars": {"ungrounded_entity_max": _GG_UNGROUNDED_ENTITY_MAX,
                 "min_grounded_answer_rate":
                     _GG_MIN_GROUNDED_ANSWER_RATE},
    }


def gg_aggregate_multiseed(per_seed_verdicts,
                           min_seeds: int = _GG_MIN_SEEDS) -> Dict:
    n = len(per_seed_verdicts)
    eff_min = max(int(min_seeds), _GG_MIN_SEEDS)
    n_pass = sum(1 for v in per_seed_verdicts
                 if v.get("GATE") == "PASS")
    both_probes = (n > 0 and all(
        v.get("n_grounded", 0) > 0 and v.get("n_ungrounded", 0) > 0
        for v in per_seed_verdicts))
    gate = bool(n >= eff_min and n_pass == n and n > 0
                and both_probes)
    return {"GATE": "PASS" if gate else "FAIL", "n_seeds": n,
            "min_seeds": eff_min, "n_pass": n_pass,
            "all_have_both_probes": both_probes,
            "all_pass": (n > 0 and n_pass == n)}
```

**Step 4:** `pytest tests/test_generator_g_core.py -q` -> all 8 PASS. Root-cause WITHOUT weakening; STOP+report a genuine spec contradiction.

**Step 5: Commit**
```bash
git add research/runners/generator_g_core.py tests/test_generator_g_core.py
git commit -m "feat(Generator-G): pure FIXED-bar no-confab-preservation + faithfulness core (load-bearing relational no-confab bar; adversarially pinned)"
```

**End of Phase A:** `git push origin HEAD && git push gitea HEAD`. Dispatch a RIGOROUS adversarial review of Tasks 1+2 (the spy-LM-never-called no-confab-by-construction guarantee + the relational no-confab-preserved bar + always-abstain/missing-control/bars-immutable adversarial properties) before Phase B.

---

## PHASE B — integration (import/signature smoke + the gate itself)

### Task 3: Generator-G gate runner

**Files:** Create `research/runners/generator_g_gate.py`; Test `tests/test_generator_g_gate_smoke.py`

**Reference design** (the runner; study `research/runners/abstention_gate.py` + `research/runners/subword_lm_gate.py` orchestration shape):
- `--seeds 42,43,44` (>=3 enforced; `<3 -> print NOT RUNNABLE; return 2`), `--tiny` (toy: smaller max_new + smaller frozen sets for the grounding pin), `--ckpt`, `--out` (default `research/findings/raw/g11_bg/generator_g_gate.json`).
- A `_TinyGPTLM` adapter wrapping the trained Generator-F TinyGPT: `__init__` loads `research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real.pt` (state dict under `"model"`) into `TinyGPT(vocab_size=<from .bpe.json>, d_model=256, n_layer=4, n_head=4, block_size=128)` + the `.bpe.json` tokenizer; `generate_ids(prompt_ids, max_new)` = greedy (argmax) autoregressive continuation (temperature 0 == faithful), sliding the 128 context. If the ckpt/bpe artifact is ABSENT -> the runner prints `[NOT RUNNABLE] trained Generator-F checkpoint absent` and returns 2 (the grounding pin skips on rc==2; the controller's decisive run guarantees the artifact exists).
- A FROZEN deterministic grounded source (the decisive slice isolates the COMPOSITION; G.20 retrieval is ALREADY separately multi-seed-validated — full-G.20 wiring is a documented later increment): `GROUNDED = {query: (retrieved_text, confidence>650)}` for a frozen set of stored simple propositions (e.g. TinyStories-style "max is a big dog", "lily has a red ball", ...), and a DISJOINT `UNGROUNDED` query list (never stored). For a grounded query the runner builds `ranked=[(subject, confidence, "kb")]` (confidence 900 > 650); for an ungrounded query `ranked=[]` (-> `gate()` None -> abstain). The seed permutes the frozen sets / decode order (not the bars).
- Per seed: for each grounded query call `grounded_decode(ranked, lm, tok, retrieved_text, query, 650.0)`; for each ungrounded query likewise. Compute: `bare_moat_abstain_rate` = fraction of (grounded+ungrounded) queries where `abstention_gate.gate(ranked,650.0) is None` computed DIRECTLY on the same ranked WITHOUT the LM (the frozen reference); `abstain_on_ungrounded_rate` = fraction of UNGROUNDED where `grounded_decode` abstained; `grounded_answer_rate` = fraction of GROUNDED where it answered (not abstained); `mean_ungrounded_entity_rate` = mean `generator_g_core.ungrounded_entity_rate(resp, retrieved)` over answered grounded. `v = gg_verdict(abstain_on_ungrounded_rate, <bare-moat rate on the ungrounded subset>, grounded_answer_rate, mean_ungrounded_entity_rate, has_ungrounded_control=len(UNGROUNDED)>0)`; tag `n_grounded`/`n_ungrounded`. `gg_aggregate_multiseed`.
- Record per-seed `transcripts`: every ungrounded `(query -> "ABSTAIN")` and every grounded `(query -> response[:200])` for the MANDATORY smell-test. JSON + ASCII verdict block; kill-safe `.resume.json`; honest-propagation-is-controller's-job; banner states the HONEST CEILING ("small-Transformer simple grounded responses + preserved no-confab; NOT an LLM"). NO `song_g1_core`/`subword_lm_gate_core` import; NO bar redefinition; reuse `abstention_gate` UNMODIFIED. ASCII-only. `return 0` (verdict computed) / `2` (not runnable).

**Smoke** `tests/test_generator_g_gate_smoke.py`:

```python
import subprocess, sys, inspect


def test_import_no_bar_no_g1_uses_validated_moat():
    import research.runners.generator_g_gate as g
    src = inspect.getsource(g)
    assert "abstention_gate" in src           # validated moat reused
    assert "song_g1_core" not in src
    assert "subword_lm_gate_core" not in src
    assert "_GG_UNGROUNDED_ENTITY_MAX =" not in src  # no bar redef
    assert "grounded_decode" in src
    assert "generator_g_core" in src


def test_fewer_than_3_seeds_exit_2():
    r = subprocess.run([sys.executable, "-m",
        "research.runners.generator_g_gate", "--seeds", "42,43"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "NOT RUNNABLE" in r.stdout


def test_help():
    r = subprocess.run([sys.executable, "-m",
        "research.runners.generator_g_gate", "--help"],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
```

**Procedure:** TDD smoke -> impl -> smoke passes; then run the Task-0 grounding pin (`pytest tests/test_generator_g_grounding.py -q` -> GREEN, or SKIP if the trained ckpt is absent in this env — the controller's Task 5 guarantees it exists). Verify `git status --porcelain` ZERO modifications to pre-existing files (esp. `abstention_gate.py`, `subword_lm_gate_core.py`, `song_g1_core.py`, `tiny_transformer.py`, `grounded_decode.py`, `generator_g_core.py` UNTOUCHED). Commit:
```bash
git add research/runners/generator_g_gate.py tests/test_generator_g_gate_smoke.py
git commit -m "feat(Generator-G): gate runner (validated moat gates FIRST; trained Generator-F TinyGPT faithfulness-constrained on grounded; FIXED bars via generator_g_core; kill-safe)"
```

---

### Task 4: LOAD-BEARING no-harm (the validated no-confab moat MUST stay byte-identical + green)

**Files:** Create `tests/test_generator_g_noharm.py`

```python
"""LOAD-BEARING no-harm: Generator-G is PURELY ADDITIVE; the validated
no-confab moat (abstention_gate) + frozen anti-cheat cores stay
byte-untouched and green; NO new global bar; no song_g1_core pull."""
import sys


def test_validated_moat_byte_contract_intact():
    from research.runners.abstention_gate import (
        gate, abstain, DEFAULT_THRESHOLD)
    assert DEFAULT_THRESHOLD == 650.0           # the validated gate
    assert abstain(650.0, 650.0) is True        # <= threshold abstains
    assert abstain(650.01, 650.0) is False
    assert gate([("x", 700.0, "t")], 650.0) == ("x", 700.0, "t")
    assert gate([("x", 600.0, "t")], 650.0) is None


def test_generator_g_does_not_pull_song_g1_or_subword_core():
    before = ("research.runners.song_g1_core" in sys.modules,
              "research.runners.subword_lm_gate_core" in sys.modules)
    import sim.grounded_decode  # noqa
    import research.runners.generator_g_core  # noqa
    import research.runners.generator_g_gate  # noqa
    after = ("research.runners.song_g1_core" in sys.modules,
             "research.runners.subword_lm_gate_core" in sys.modules)
    assert before == after


def test_gg_core_owns_its_frozen_bars():
    import research.runners.generator_g_core as c
    assert (c._GG_UNGROUNDED_ENTITY_MAX,
            c._GG_MIN_GROUNDED_ANSWER_RATE, c._GG_MIN_SEEDS) == (
        0.20, 0.5, 3)
```

Controller also verifies: `pytest tests/test_abstention_gate.py -q` GREEN (the validated moat untouched + still passing — the project's distinctive contribution must NOT be regressed) + representative validated suite green; and `git diff --stat <gen-G-range> -- research/runners/abstention_gate.py research/runners/song_g1_core.py research/runners/subword_lm_gate_core.py sim/tiny_transformer.py sim/bridge.py research/runners/g20_*.py` is EMPTY.

**Commit + push both remotes.** Controller spec+quality check of Phase B (proportional; the anti-cheat firewall is the UNMODIFIED validated moat + the adversarially-reviewed generator_g_core).

---

### Task 5: Decisive multi-seed run + honest propagation (CONTROLLER, not a subagent)

1. **Grounding-first:** Task-0 pin (tiny zero-network) proves the pipeline turns + is interpretable. Re-confirm; toy verdict NOT propagated. If broken -> @superpowers:systematic-debugging.
2. **Decisive run:** reuse the trained Generator-F ckpt + the frozen grounded/ungrounded sets at the FIXED pre-registered config (seeds 42,43,44; default sizes). CPU-fast (no training); kill-safe; may run foreground or `run_in_background`.
3. **MANDATORY anti-cheat smell-test BEFORE propagating (scrutinize a nominal PASS HARDER than a FAIL):** recompute from the recorded JSON (NO re-run, NO bar-tuning): **read EVERY ungrounded transcript — each MUST be "ABSTAIN"** (no-confab genuinely preserved; if a single ungrounded query produced fluent text, that is a FAIL regardless of the aggregate); verify `grounded_answer_rate` genuinely > 0.5 (NOT trivially always-abstain — the agent must actually answer grounded queries); **read the grounded transcripts — did the LM rename/confabulate entities (Max->Bob)?** characterize the TRUE faithfulness honestly; confirm `mean_ungrounded_entity_rate <= 0.20` genuinely. The bars are enforced by `gg_core`; the transcript read is the honest adjudication.
4. **Honest propagation EITHER way:** findings doc `research/findings/2026-05-17-generator-G-grounded-noconfab-<PASS|NEGATIVE>.md`. **If scrutinized-genuine PASS:** "a self-contained, local, no-cheat agent generates coherent SIMPLE grounded responses AND preserves the validated no-confabulation property a small LLM lacks — the honest culmination within the explicit small-Transformer ceiling; NOT an LLM, NOT global coherence; the validated biology-grounded memory remains the SEPARATE primary asset" — with verbatim transcripts, never spun. **If FAIL:** the decision-relevant terminal conclusion — fluency and no-confabulation do NOT compose into a single self-contained artifact at feasible local scale (likely the faithfulness bar — the LM confabulates entities under conditioning, per the probe); the two validated assets (Generator-F fluency; the grounded no-confab memory) stand as SEPARATE deliverables; the converged honest conclusion of the whole arc. `webapp/capability_status.json` pillar (VALIDATED only if scrutinized-genuine PASS at the explicit ceiling with transcripts + the no-confab-preservation result foregrounded; else NEGATIVE/BOUNDARY); `pytest tests/test_webapp_server.py -k capability_status` 6/6 green; commit + push BOTH remotes; bars NOT tuned.
5. **Continuous arc — no stop/ask/config-crank:** PASS => the honest culmination; propagate, and the unified artifact stands at the small-LM ceiling (later increments: real-G.20-retrieval wiring, multi-turn — noted not built). FAIL => propagate the terminal decision-relevant conclusion + the converged honest synthesis of the entire conversational-generation arc (10 mechanisms): the deliverables are the SEPARATE validated assets (Generator-F small-LM fluency; the grounded no-confabulation memory). Either way continue autonomously; do NOT stop to ask.

---

## Notes
- DRY: `abstention_gate` (validated moat) / `TinyGPT` / trained Generator-F ckpt / BPE reused byte-UNMODIFIED. NO new global bar. `song_g1_core`/`subword_lm_gate_core` byte-UNTOUCHED.
- YAGNI: cheap decisive slice; the COMPOSITION question only (G.20 retrieval already separately validated; full-retrieval wiring + multi-turn are later increments, noted not built).
- TDD: pure logic (Tasks 1,2) failing-test->impl->commit with a rigorous adversarial review (the spy-LM-never-called no-confab-by-construction guarantee + the relational no-confab-preserved bar are load-bearing); Task 3 import/signature smoke + the gate itself.
- @superpowers:systematic-debugging if the grounding pipeline breaks.
- @superpowers:subagent-driven-development for execution; trust-but-verify each subagent's `git diff`; protected modules byte-empty in each commit-scoped diff; `tests/test_abstention_gate.py` MUST stay green (the validated moat is the project's distinctive contribution — never regress it).
- The Generator-S/D/E/F lesson is mandatory: scrutinize a nominal PASS HARDER than a FAIL; read the ACTUAL transcripts (ungrounded must all ABSTAIN; grounded must not rename entities); never overclaim beyond the small-Transformer ceiling; the validated biology-grounded asset is the separate primary contribution and is untouched.
