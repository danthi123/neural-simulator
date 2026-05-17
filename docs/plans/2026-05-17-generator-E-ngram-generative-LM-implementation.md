# Generator-E — Self-Contained N-gram Generative LM through the SAME Hardened Gate — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task (continuous autonomous arc; do NOT stop to ask between tasks or increments — user authorized a week of autonomous work 2026-05-17).

**Goal:** Decide, with the SAME unmodified HARDENED gate_core that 9 neural attempts failed, whether a self-contained n-gram generative LM clears the rigorous anti-cheat gate (esp. the load-bearing verbatim-copy/regurgitation bar) — a decision-relevant terminus test of the project's actual goal.

**Architecture:** 3 tiny net-new pure-ish files: an n-gram sampler + an n-gram held-out-nll + a thin gate runner that mirrors `subword_lm_gate.py` but uses the UNMODIFIED `NgramTeacher` as the runtime generative model and a word-shuffled-trained n-gram as the load-bearing control. Everything load-bearing (NgramTeacher, hardened gate_core, corpus_fetch, BPE) reused byte-UNMODIFIED.

**Tech Stack:** Python, numpy/stdlib only. NO GPU (n-gram train+gen is CPU-seconds). Trivially kill-safe.

**Validated APIs reused UNMODIFIED (DRY — do NOT reimplement):**
- `sim.ngram_teacher.NgramTeacher`: `NgramTeacher()`, `.train(train_ids: list[int], vocab_size: int, k: float=0.1) -> None`, `.soft_dist(ctx) -> np.ndarray (len V, sums to 1)`; `ctx` may be `()`, `(a,)`, `(a,b)`. Grounded competent (held-out ppl ~14-15 vs random 513, probe `ba1jyepwf`). This IS the generative model now (not a teacher).
- `research.runners.subword_lm_gate_core` (HARDENED, FROZEN): `perplexity(nll_list)->float`, `distinct_ngram_ratio(ids,n=3)->float`, `verbatim_copy_fraction(gen,train,n=8)->float`, `gs_verdict(heldout_ppl,shuffled_ppl,train_ppl,distinct,copy_frac,has_shuffled_control,uniform_ppl=None)->dict` (fail-closed without `uniform_ppl`), `gs_aggregate_multiseed(per_seed,min_seeds=3)->dict`. Bars `_GS_PPL_MARGIN=0.20`, `_GS_GENERALIZATION_MAX=1.5`, `_GS_DISTINCT_MIN=0.5`, `_GS_COPY_MAX=0.20`, `_GS_MIN_SEEDS=3`, `_GS_ABS_COMPETENCE_PPL_RATIO=1.0`. **byte-UNMODIFIED; NO new bar.**
- `research.runners.corpus_fetch.{fetch_corpus,split_corpus}`, `sim.bpe_tokenizer.BPETokenizer`.
- `research.runners.subword_lm_gate` — orchestration SHAPE to mirror (`_word_shuffle`, per-seed kill-safe `.resume.json`, ASCII verdict block, `<3 seeds -> exit 2`, honest-propagation-is-controller's-job).

**MUST NOT touch (LOAD-BEARING no-harm):** `sim/ngram_teacher.py`, `research/runners/subword_lm_gate_core.py` (frozen bars), `research/runners/song_g1_core.py`, `sim/bridge.py`, `research/runners/g20_*`, any validated runner. Generator-E is PURELY ADDITIVE new files. NO new bar; the HARDENED gate_core decides.

**Anti-cheat (non-negotiable):** an n-gram's chief failure mode is REGURGITATION — the hardened `verbatim_copy_fraction<=0.20` + BPE-invariant word-shuffle control + absolute-competence floor + `>=3` sampling seeds are exactly the load-bearing adjudicators. MANDATORY post-run smell-test: scrutinize a nominal PASS HARDER than a FAIL — verify from the recorded JSON that copy_frac is genuinely <=0.20 AND held-out ppl is genuinely competent AND the real n-gram genuinely beats the word-shuffle control by >=20%. Recompute from recorded data; NO re-run; NO bar-tuning. NO overclaiming: a PASS is reported strictly at its honest ceiling (n-gram-class local coherence; self-contained/local/no-cheat; explicitly NOT an LLM; the validated grounded-memory + no-confabulation remains the separate primary deliverable). A false-PASS or maxed-FAIL is an honest propagated finding.

**ASCII-only prints (Windows cp1252). Commit after every task. Push both remotes (`origin`,`gitea`) after each phase.**

---

### Task 0: Falsify-cheaply grounding pin (green after Task 3)

**Files:** Create `tests/test_generator_e_grounding.py`

**Step 1: Write the test**

```python
"""Grounding: the generator_e_gate pipeline TURNS end-to-end on local
shakespeare (zero network) and produces an interpretable verdict.
(N-gram ppl competence is already grounded by probe ba1jyepwf ~14-15;
this pin is the END-TO-END pipeline gate.) Green after Task 3."""
import os
import subprocess
import sys
import json
import pytest


def test_generator_e_gate_pipeline_turns_local(tmp_path):
    if not os.path.exists("data/tinyshakespeare.txt"):
        pytest.skip("local grounding corpus absent")
    out = str(tmp_path / "e.json")
    ck = str(tmp_path / "e.ckpt")
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.generator_e_gate",
         "--seeds", "42,43,44", "--corpus", "data/tinyshakespeare.txt",
         "--vocab-size", "96", "--gen-tokens", "40",
         "--eval-positions", "60", "--out", out, "--ckpt", ck],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    d = json.loads(open(out, encoding="utf-8").read())
    assert d["n_seeds"] == 3 and "aggregate_verdict" in d
    assert all("verdict" in s for s in d["per_seed"])
    for s in d["per_seed"]:
        assert s["uniform_ppl"] == d["config"]["vocab_size"]
```

**Step 2:** `pytest tests/test_generator_e_grounding.py -q` → FAIL (module missing — green after Task 3; it IS the Task-3 gate).

**Step 5: Commit**
```bash
git add tests/test_generator_e_grounding.py
git commit -m "test(Generator-E): falsify-cheaply grounding pin (gate pipeline turns end-to-end; zero network) -- green after Task 3"
```

---

## PHASE A — pure-logic CPU-TDD

### Task 1: Pure n-gram sampler

**Files:** Create `sim/ngram_generate.py`; Test `tests/test_ngram_generate.py`

**Step 1: Write the failing tests**

```python
import numpy as np
from sim.ngram_generate import ngram_sample_next, ngram_generate


class _FakeTeacher:
    """Minimal stand-in with the NgramTeacher.soft_dist contract."""
    def __init__(self, V, peak=None):
        self.V = V
        self.peak = peak
    def soft_dist(self, ctx):
        q = np.full(self.V, 1.0 / self.V, dtype=np.float64)
        if self.peak is not None:
            q[:] = 0.01 / self.V
            q[self.peak] = 1.0 - 0.01 + 0.01 / self.V
            q = q / q.sum()
        return q


def test_sample_next_temp0_is_argmax_stable_first_max():
    t = _FakeTeacher(5, peak=3)
    assert ngram_sample_next(t, (1, 2), np.random.default_rng(0),
                             temperature=0.0) == 3
    # uniform -> stable FIRST max (index 0)
    tu = _FakeTeacher(5)
    assert ngram_sample_next(tu, (), np.random.default_rng(7),
                             temperature=0.0) == 0


def test_sample_next_temp_in_range_and_seed_reproducible():
    t = _FakeTeacher(8)
    a = ngram_sample_next(t, (1,), np.random.default_rng(42), 1.0)
    b = ngram_sample_next(t, (1,), np.random.default_rng(42), 1.0)
    assert a == b and 0 <= a < 8


def test_sample_next_degenerate_safe():
    t = _FakeTeacher(4)
    assert 0 <= ngram_sample_next(t, (1, 2),
                                  np.random.default_rng(1), 1.0) < 4
    assert 0 <= ngram_sample_next(t, (),
                                  np.random.default_rng(1), 0.0) < 4


def test_generate_length_and_range_and_reproducible():
    t = _FakeTeacher(6, peak=2)
    g1 = ngram_generate(t, [5, 1], 10, np.random.default_rng(3), 1.0)
    g2 = ngram_generate(t, [5, 1], 10, np.random.default_rng(3), 1.0)
    assert g1 == g2 and len(g1) == 10
    assert all(0 <= x < 6 for x in g1)
    assert g1 != [5, 1]                       # returns generated, not prompt
    # empty prompt safe; temp0 deterministic toward the peak
    g0 = ngram_generate(t, [], 5, np.random.default_rng(0), 0.0)
    assert g0 == [2, 2, 2, 2, 2]
```

**Step 2:** `pytest tests/test_ngram_generate.py -q` → FAIL (module missing).

**Step 3: Implement** `sim/ngram_generate.py`:

```python
"""Pure autoregressive sampler over a NgramTeacher's dense soft
distribution -- the Generator-E runtime generative model. The
NgramTeacher (sim.ngram_teacher) is reused UNMODIFIED; this only
samples from its `soft_dist`. Pure numpy/stdlib; CPU-unit-testable;
self-contained at runtime (count tables + BPE JSON only)."""
from __future__ import annotations
import numpy as np


def ngram_sample_next(teacher, ctx, rng, temperature: float = 1.0) -> int:
    """Sample the next token id from teacher.soft_dist(ctx).

    temperature == 0 -> deterministic argmax (stable FIRST max).
    temperature  > 0 -> sample from the temperature-reweighted
                        distribution p ~ q ** (1/T), renormalized.
    Degenerate / non-finite q is made safe (never raises; always an
    in-range int)."""
    q = np.asarray(teacher.soft_dist(ctx), dtype=np.float64).reshape(-1)
    V = q.shape[0]
    if V == 0:
        return 0
    q = np.where(np.isfinite(q), q, 0.0)
    q = np.clip(q, 0.0, None)
    if q.sum() <= 0:
        q = np.full(V, 1.0 / V)
    if temperature is None or temperature <= 0.0:
        return int(np.argmax(q))            # stable first-max
    z = np.power(q, 1.0 / float(temperature))
    s = z.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.argmax(q))
    return int(rng.choice(V, p=z / s))


def ngram_generate(teacher, prompt_ids, n_tokens, rng,
                    temperature: float = 1.0):
    """Autoregressive generation. ctx = trailing up-to-2 ids of
    (prompt + generated-so-far) -- the trigram context the
    NgramTeacher backs off over. Returns ONLY the generated id list
    (prompt excluded)."""
    seq = list(prompt_ids)
    out = []
    for _ in range(int(n_tokens)):
        ctx = tuple(seq[-2:])               # (), (a,) or (a,b)
        nxt = ngram_sample_next(teacher, ctx, rng, temperature)
        seq.append(nxt)
        out.append(nxt)
    return out
```

**Step 4:** `pytest tests/test_ngram_generate.py -q` → all 4 PASS. Root-cause any failure WITHOUT weakening a test or touching another file. If a genuine spec contradiction, STOP and report (do NOT fake-pass).

**Step 5: Commit**
```bash
git add sim/ngram_generate.py tests/test_ngram_generate.py
git commit -m "feat(Generator-E): pure n-gram sampler (temperature/argmax, seed-reproducible, degenerate-safe)"
```

---

### Task 2: Pure n-gram held-out NLL

**Files:** Create `sim/ngram_ppl.py`; Test `tests/test_ngram_ppl.py`

**Step 1: Write the failing tests**

```python
import math
import numpy as np
from sim.ngram_ppl import ngram_heldout_nll


class _T:
    def __init__(self, table):
        # table: dict ctx-tuple -> length-V prob list
        self.table = table
        self.V = len(next(iter(table.values())))
    def soft_dist(self, ctx):
        return np.asarray(self.table[tuple(ctx)], dtype=np.float64)


def test_nll_matches_hand_computed():
    # ids = [0,1,2]; only i=2 scored: ctx=(0,1) p(2)=0.5 -> nll=ln2
    t = _T({(0, 1): [0.25, 0.25, 0.5]})
    nll = ngram_heldout_nll(t, [0, 1, 2])
    assert len(nll) == 1
    assert abs(nll[0] - math.log(2.0)) < 1e-9


def test_short_input_is_empty():
    t = _T({(0, 1): [0.5, 0.5]})
    assert ngram_heldout_nll(t, []) == []
    assert ngram_heldout_nll(t, [0]) == []
    assert ngram_heldout_nll(t, [0, 1]) == []     # need >=3 to score i>=2


def test_zero_prob_is_clamped_not_inf():
    t = _T({(0, 1): [1.0, 0.0, 0.0]})              # p(true=2)=0
    nll = ngram_heldout_nll(t, [0, 1, 2])
    assert math.isfinite(nll[0]) and nll[0] == -math.log(1e-12)


def test_perplexity_roundtrip():
    from research.runners.subword_lm_gate_core import perplexity
    t = _T({(0, 1): [0.0, 0.0, 1.0], (1, 2): [0.0, 0.0, 1.0]})
    # ids [0,1,2,2]: i=2 ctx(0,1) p(2)=1 -> 0 ; i=3 ctx(1,2) p(2)=1 -> 0
    assert abs(perplexity(ngram_heldout_nll(t, [0, 1, 2, 2]))
               - 1.0) < 1e-9
```

**Step 2:** `pytest tests/test_ngram_ppl.py -q` → FAIL (module missing).

**Step 3: Implement** `sim/ngram_ppl.py`:

```python
"""Pure held-out per-token NLL for a NgramTeacher -- the EXACT formula
from the grounded probe ba1jyepwf (held-out ppl ~14-15). Combine with
subword_lm_gate_core.perplexity for ppl. Pure numpy/stdlib;
CPU-unit-testable."""
from __future__ import annotations
import math
import numpy as np


def ngram_heldout_nll(teacher, ids):
    """Per-token NLL over `ids`: for i in range(2, len(ids)),
    nll_i = -log(max(teacher.soft_dist((ids[i-2], ids[i-1]))[ids[i]],
                     1e-12)). ids shorter than 3 -> []. The 1e-12 floor
    clamps zero-prob (never +inf), matching the grounded probe."""
    ids = list(ids)
    n = len(ids)
    if n < 3:
        return []
    out = []
    for i in range(2, n):
        q = np.asarray(teacher.soft_dist((ids[i - 2], ids[i - 1])),
                       dtype=np.float64)
        p = float(q[ids[i]]) if 0 <= ids[i] < q.shape[0] else 0.0
        out.append(-math.log(max(p, 1e-12)))
    return out
```

**Step 4:** `pytest tests/test_ngram_ppl.py -q` → all 4 PASS. Root-cause any failure WITHOUT weakening; STOP+report a genuine spec contradiction (do NOT fake-pass).

**Step 5: Commit**
```bash
git add sim/ngram_ppl.py tests/test_ngram_ppl.py
git commit -m "feat(Generator-E): pure n-gram held-out NLL (exact grounded-probe formula; zero-prob clamped)"
```

**End of Phase A:** `git push origin HEAD && git push gitea HEAD`. Controller trust-but-verify (git-match the verbatim references + green + additive); these are small/pure and the load-bearing anti-cheat is the already-adversarially-reviewed UNMODIFIED hardened gate_core, so a focused controller spec+quality check is proportional (verify: ngram_sample_next temp0=argmax / temp>0 seed-reproducible / degenerate-safe; ngram_heldout_nll formula exactness + zero-prob clamp).

---

## PHASE B — integration (import/signature smoke + the gate itself)

### Task 3: Generator-E gate runner (DRY mirror of `subword_lm_gate`)

**Files:** Create `research/runners/generator_e_gate.py`; Test `tests/test_generator_e_gate_smoke.py`

**Reference:** byte-mirror `research/runners/subword_lm_gate.py` (read it) with EXACTLY these changes:
- Remove the SNN trainer / GPU / `_get_backend` / `_heldout_nll` (SNN) paths. The model is the n-gram.
- Per seed: `fetch_corpus(name=a.corpus, max_bytes=a.max_corpus_mb*1_000_000)` + `split_corpus`; `tok = BPETokenizer(); tok.train(train_text, vocab_size=a.vocab_size)`; `V = tok.vocab_size`.
  - `tr_ids = tok.encode(train_text)`; `ho_ids = tok.encode(heldout_text)`.
  - `real = NgramTeacher(); real.train(tr_ids, vocab_size=V)`.
  - control: `ctl_text = _word_shuffle(train_text, np.random.default_rng(seed*911+1))`; `ctl_ids = tok.encode(ctl_text)`; `ctl = NgramTeacher(); ctl.train(ctl_ids, vocab_size=V)` (BPE-invariant word-shuffle -> identical vocab/token distribution, trigram structure destroyed).
  - `ho_ppl = perplexity(ngram_heldout_nll(real, ho_ids))`; `ctl_ppl = perplexity(ngram_heldout_nll(ctl, ho_ids))`; `tr_ppl = perplexity(ngram_heldout_nll(real, tr_ids[:len(ho_ids)]))`.
  - generation: `prompt_ids = tok.encode(" ".join(heldout_text.split()[:8]))`; `gen_ids = ngram_generate(real, prompt_ids, a.gen_tokens, np.random.default_rng(seed*13+5), 1.0)`.
  - `distinct = distinct_ngram_ratio(gen_ids, n=3)`; `copy = verbatim_copy_fraction(gen_ids, tr_ids, n=8)`.
  - `v = gs_verdict(heldout_ppl=ho_ppl, shuffled_ppl=ctl_ppl, train_ppl=tr_ppl, distinct=distinct, copy_frac=copy, has_shuffled_control=True, uniform_ppl=V)` — MUST pass `uniform_ppl=V` (hardened gate_core fail-closed without it).
  - per_seed_records: seed, ho_ppl, shuffled_ctl_ppl, train_ppl, uniform_ppl=V, distinct_trigram, verbatim_copy_frac, gen_sample = `tok.decode(gen_ids)[:240]`, verdict.
- Keep IDENTICAL: argparse shape (add NOTHING GPU; keep `--seeds --corpus --max-corpus-mb --vocab-size --gen-tokens --eval-positions --out --ckpt`; defaults `--out research/findings/raw/g11_bg/generator_e_gate.json`, `--ckpt research/findings/raw/g11_bg/generator_e_gate.ckpt`, `--vocab-size 512`, `--gen-tokens 200`, `--eval-positions 4000`), `<3 seeds -> exit 2`, per-seed kill-safe `.resume.json`, `gs_aggregate_multiseed`, JSON write, ASCII verdict block, banner says "GENERATOR-E ... n-gram generative LM ... SAME HARDENED gate_core ... NOT an LLM (honest ceiling)", `if agg["GATE"]!="PASS": print NOTE proceed to Generator-F-or-terminal; do NOT config-crank`, honest-propagation-is-controller's-job, `return 0`. NO GPU import. ASCII only.
- `--eval-positions` is accepted for arg-compat but the n-gram nll scores every position (cheap); you MAY cap `ho_ids`/`tr_ids` scoring to `a.eval_positions*?` only if needed for speed — but n-gram is CPU-fast so score the full held-out (document if you cap; default = full).

**Smoke** `tests/test_generator_e_gate_smoke.py`:

```python
import subprocess, sys, inspect

def test_import_passes_uniform_ppl_no_bar_no_g1_no_gpu():
    import research.runners.generator_e_gate as g
    src = inspect.getsource(g)
    assert "uniform_ppl=" in src
    assert "song_g1_core" not in src
    assert "_GS_PPL_MARGIN =" not in src
    assert "_GS_ABS_COMPETENCE_PPL_RATIO =" not in src
    assert "import cupy" not in src and "_get_backend" not in src
    import numpy as np
    s = g._word_shuffle("a b c d e f g h", np.random.default_rng(1))
    assert sorted(s.split()) == list("abcdefgh")

def test_fewer_than_3_seeds_exit_2():
    r = subprocess.run([sys.executable, "-m",
        "research.runners.generator_e_gate", "--seeds", "42,43"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "NOT RUNNABLE" in r.stdout

def test_help():
    r = subprocess.run([sys.executable, "-m",
        "research.runners.generator_e_gate", "--help"],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "MULTI-SEED" in r.stdout
```

**Procedure:** TDD smoke (fails → mirror impl → passes); then run the Task-0 grounding pin (`pytest tests/test_generator_e_grounding.py -q` → now GREEN, proving the pipeline turns end-to-end on local shakespeare). Verify `git status --porcelain` shows ZERO modifications to any pre-existing file (esp. `subword_lm_gate.py`, `subword_lm_gate_core.py`, `song_g1_core.py`, `ngram_teacher.py`, `ngram_generate.py`, `ngram_ppl.py` UNTOUCHED — reused by import only). Commit:
```bash
git add research/runners/generator_e_gate.py tests/test_generator_e_gate_smoke.py
git commit -m "feat(Generator-E): n-gram generative LM gate runner (DRY mirror of subword_lm_gate; NgramTeacher model + word-shuffle-control n-gram; passes uniform_ppl=V to HARDENED gate_core; CPU/kill-safe)"
```

---

### Task 4: LOAD-BEARING no-harm

**Files:** Create `tests/test_generator_e_noharm.py`

```python
"""LOAD-BEARING no-harm: Generator-E is PURELY ADDITIVE; the validated
deliverable + the FROZEN hardened gate_core bars are byte-untouched;
NgramTeacher reused UNMODIFIED; NO new bar; no song_g1_core pull."""
import sys


def test_hardened_bars_frozen_and_no_g1():
    import research.runners.subword_lm_gate_core as g
    assert (g._GS_PPL_MARGIN, g._GS_GENERALIZATION_MAX,
            g._GS_DISTINCT_MIN, g._GS_COPY_MAX, g._GS_MIN_SEEDS,
            g._GS_ABS_COMPETENCE_PPL_RATIO) == (0.20, 1.5, 0.5, 0.20,
                                                3, 1.0)
    assert not hasattr(g, "_G1_MARGIN")


def test_generator_e_does_not_pull_song_g1_core():
    before = "research.runners.song_g1_core" in sys.modules
    import sim.ngram_generate  # noqa: F401
    import sim.ngram_ppl  # noqa: F401
    import research.runners.generator_e_gate  # noqa: F401
    after = "research.runners.song_g1_core" in sys.modules
    assert before == after


def test_ngram_teacher_reused_unmodified_contract():
    # Generator-E treats NgramTeacher as the model; pin its contract.
    import numpy as np
    from sim.ngram_teacher import NgramTeacher
    t = NgramTeacher()
    t.train([1, 2, 3, 1, 2, 3, 1, 2, 4] * 30, vocab_size=8)
    q = t.soft_dist((1, 2))
    assert q.shape == (8,) and abs(float(q.sum()) - 1.0) < 1e-9
```

Controller also verifies: full representative existing suite green
(`pytest tests/test_subword_lm_gate_core.py tests/test_ngram_teacher.py tests/test_soft_xent.py tests/test_order_intrinsic_core.py tests/test_webapp_server.py -k "capability_status or gate_core or ngram or soft_xent or order_intrinsic" -q`) and
`git diff --stat <gen-E-range> -- research/runners/subword_lm_gate_core.py research/runners/song_g1_core.py sim/bridge.py sim/ngram_teacher.py research/runners/g20_*.py research/runners/subword_lm_gate.py` is EMPTY.

**Commit + push both remotes.** Controller spec+quality check of Phase B (proportional — the anti-cheat firewall is the UNMODIFIED already-adversarially-reviewed hardened gate_core).

---

### Task 5: Decisive multi-seed run + honest propagation (CONTROLLER, not a subagent)

1. **Grounding-first (falsify-cheaply):** the Task-0 pin (local shakespeare, tiny config, zero network) already proves the pipeline turns + is interpretable. Re-confirm; toy verdict NOT propagated. If broken → @superpowers:systematic-debugging (root cause first; never paper over to reach a verdict).
2. **Decisive run:** cached TinyStories, FIXED pre-registered config (`--seeds 42,43,44 --corpus tinystories --vocab-size 512 --gen-tokens 200 --eval-positions 4000`), CPU, fast (n-gram = CPU-seconds; kill-safe `.resume.json` but unlikely needed). May run foreground (fast) or `run_in_background`; monitor.
3. **MANDATORY anti-cheat smell-test BEFORE propagating (scrutinize a nominal PASS HARDER than a FAIL):** an n-gram's chief cheat is REGURGITATION. Recompute from the recorded JSON (NO re-run, NO bar-tuning): is `verbatim_copy_frac` genuinely `<= 0.20` on every seed? is held-out ppl genuinely competent (`< uniform 512`, ideally near the grounded ~15)? does the real n-gram genuinely beat the word-shuffle control by `>= 20%`? is `distinct_trigram >= 0.5` (not degenerate looping)? The hardened gate_core enforces all of these, but VERIFY the numbers tell an honest story (a PASS that squeaks the copy bar at 0.19 with ppl just under 512 is materially weaker than a PASS at copy 0.02 / ppl 15 — report the actual numbers, do not spin).
4. **Honest propagation EITHER way:** findings doc `research/findings/2026-05-17-generator-E-ngram-generative-LM-<PASS|NEGATIVE>.md`. **If PASS:** report STRICTLY at the honest ceiling — "a self-contained, local, no-cheat n-gram-class generative LM clears the SAME rigorous hardened anti-cheat gate that 9 neural attempts failed; n-gram-class LOCAL coherence only, explicitly NOT an LLM, NOT global coherence/reasoning; the validated grounded-memory + no-confabulation remains the separate primary deliverable" — NEVER spun as more than the bars certify. **If FAIL** (likely the verbatim-copy or word-shuffle-control bar): report it as the terminal, decision-relevant conclusion — even the only competent self-contained candidate does not clear the rigorous gate at feasible local scale without cheating-by-memorization; the validated grounded-memory asset stands. `webapp/capability_status.json` pillar (`status` VALIDATED only if a scrutinized real PASS strictly ceiling-framed, else NEGATIVE; schema `{name,status,metric}`); `pytest tests/test_webapp_server.py -k capability_status` 6/6 green; commit + push BOTH remotes; bars NOT tuned.
5. **Continuous arc — no stop/ask/config-crank:** PASS ⇒ Generator-F (integrate the n-gram-class generator with the validated grounded-memory + no-confabulation arch for grounded n-gram-class conversation, and/or scale via higher-order Kneser-Ney) — new design doc → writing-plans → subagent-driven-development. FAIL ⇒ propagate the terminal decision-relevant finding (no self-contained no-cheat generator clears this gate at feasible local scale) and the converged honest conclusion of the whole conversational-generation arc; the validated grounded-memory + no-confabulation agent is the deliverable. Either way: continue autonomously, do NOT stop to ask.

---

## Notes
- DRY: NgramTeacher / hardened gate_core / corpus_fetch / BPE reused UNMODIFIED. NO new bar. `song_g1_core` UNTOUCHED.
- YAGNI: cheap decisive trigram slice. Higher-order/Kneser-Ney + Generator-F are later increments (noted, not built).
- TDD: pure logic (Tasks 1,2) failing-test→impl→commit; integration (3) import/signature smoke + the gate itself.
- @superpowers:systematic-debugging if the grounding pipeline breaks.
- @superpowers:subagent-driven-development for execution; trust-but-verify each subagent's `git diff`; protected modules byte-empty in each commit-scoped diff.
- The Generator-S/D lesson is mandatory: scrutinize a nominal PASS harder than a FAIL; the n-gram-specific cheat is REGURGITATION — the verbatim-copy bar is the load-bearing adjudicator; never overclaim beyond the honest n-gram-class ceiling.
