# Scaled Subword Spiking LM (Generator-S) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task (continuous autonomous arc; do NOT stop to ask between tasks or between increments — user authorized a week of autonomous work 2026-05-17).

**Goal:** Test, with a pre-registered FIXED-bar multi-seed gate, whether a subword spiking neural LM trained by surrogate-grad BPTT on a real public corpus generates coherent held-out text (self-contained at runtime).

**Architecture:** Net-new = a pure deterministic Sennrich-style BPE tokenizer + a pure gate-scoring core + a corpus fetcher + a scaled subword SNN trainer whose per-epoch loop is a DRY mirror of the validated `cortex_pretraining.train_shakespeare`. Everything else (LIF BPTT forward/backward, surrogate gradient, atomic kill-safe checkpoint, CE/softmax) is reused UNMODIFIED from the validated Phase-2.1/2.2 core.

**Tech Stack:** Python, numpy (CPU tests), CuPy (RTX 3090 production via `sim.bptt_snn_gpu` backend), `sim.train_checkpoint` (atomic os.replace), urllib (corpus fetch, training-time only).

**Validated APIs reused UNMODIFIED (DRY — do NOT reimplement):**
- `sim.bptt_snn_gpu`: `LIFLayerXP(W_in,n_post,threshold,leak)`, `forward_unroll_xp(inputs,layers,xp)`→`{"spikes":[...],"v":[...]}`, `backward_unroll_xp(inputs,layers,state,output_grad,xp)`→`(weight_grads,input_grad)`, `_get_backend(prefer_gpu)`→`(xp,is_gpu)`, `atan_surrogate`.
- `sim.bptt_snn`: `cross_entropy_loss_np(logits,target_idx)->float`, `softmax_grad_np(logits,target_idx)->np.ndarray`.
- `sim.char_tokenizer.make_seq_dataset(text,tokenizer,seq_len,n_samples,rng)` — uses ONLY `tokenizer.encode(text)->List[int]` + `tokenizer.vocab_size`; a BPE tokenizer exposing those is a drop-in.
- `sim.train_checkpoint`: `save_checkpoint(path,epoch,weights,rng_state,loss_history)` (atomic), `load_checkpoint(path)`→ckpt|None, `resume_epoch(ckpt)`→int.
- `research.runners.cortex_pretraining.train_shakespeare` — the loop SHAPE to mirror (init std 2.0 first layer / 0.5 later; per-sample CE→softmax_grad→backward_unroll_xp→`layer.W_in -= lr*grad`).

**MUST NOT touch (LOAD-BEARING no-harm):** `sim/bridge.py`, `sim/*` validated modules, `research/runners/g20_*`, `song_g1_core.py`, `order_intrinsic_*`, any shipped validated runner. Generator-S is PURELY ADDITIVE new files. `song_g1_core`'s bars (0.10/0.5) are NOT imported — Generator-S has its OWN frozen bars.

**Anti-cheat (non-negotiable):** FIXED bars `_GS_PPL_MARGIN=0.20`, `_GS_GENERALIZATION_MAX=1.5`, `_GS_DISTINCT_MIN=0.5`, `_GS_COPY_MAX=0.20`, `_GS_MIN_SEEDS=3`, frozen as module constants the moment Task 4 lands; NEVER recomputed/tuned at gate time. Shuffled-token control load-bearing. Multi-seed ≥3. FAIL ⇒ honest propagation + immediately proceed to pre-staged Generator-D (local-teacher distillation) — NOT a config-crank, NOT a stop.

**ASCII-only prints (Windows cp1252). Commit after every task. Push both remotes (`origin`, `gitea`) after each phase.**

---

### Task 0: Falsify-cheaply grounding smoke (zero network, before ANY scaled/corpus work)

**Files:**
- Create: `tests/test_generator_s_grounding.py`

**Step 1: Write the failing test**

```python
"""Grounding smoke: the VALIDATED BPTT spiking core trains end-to-end
on the local 1.1MB data/tinyshakespeare.txt (zero network). If this
regresses, STOP -- the validated core is broken, nothing downstream
is interpretable."""
import os, numpy as np, pytest

def test_validated_bptt_core_trains_on_local_shakespeare():
    if not os.path.exists("data/tinyshakespeare.txt"):
        pytest.skip("local grounding corpus absent")
    from research.runners.cortex_pretraining import train_shakespeare
    r = train_shakespeare(seed=42, T=24, hidden_layers=[48],
                          epochs=6, batch_size=16, n_train_samples=120,
                          corpus_path="data/tinyshakespeare.txt",
                          backend="cpu", verbose=False)
    assert r["final_loss"] < r["initial_loss"], (
        "validated BPTT core no longer reduces loss -- STOP, fix the "
        "core before any Generator-S work")
    assert r["vocab_size"] > 10 and r["n_layers"] == 2
```

**Step 2: Run — expect PASS immediately** (this pins the validated core, not new code).
Run: `pytest tests/test_generator_s_grounding.py -q`
Expected: PASS (≈ a few seconds). If FAIL → STOP, the validated core regressed; do not proceed.

**Step 3:** (no implementation — this is a pin on existing validated code)

**Step 4:** Re-run, confirm PASS.

**Step 5: Commit**
```bash
git add tests/test_generator_s_grounding.py
git commit -m "test(Generator-S): falsify-cheaply grounding pin on validated BPTT core (zero network)"
```

---

## PHASE A — pure-logic CPU-TDD (BPE tokenizer + gate core)

### Task 1: Pure deterministic BPE tokenizer

**Files:**
- Create: `sim/bpe_tokenizer.py`
- Test: `tests/test_bpe_tokenizer.py`

Sennrich-2016 word-frequency BPE: split on whitespace, each word = tuple of chars + end-marker `</w>`; iteratively merge the highest-frequency adjacent symbol pair (deterministic lexicographic tie-break) until `vocab_size` reached; encode applies learned merges greedily; JSON save/load. Interface-compatible with `make_seq_dataset` (`.encode`, `.vocab_size`).

**Step 1: Write the failing tests**

```python
import json, numpy as np
from sim.bpe_tokenizer import BPETokenizer

def test_train_encode_decode_roundtrip():
    corpus = ("the cat sat on the mat . the cat ran . " * 50)
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=60)
    assert tok.vocab_size >= 1 and tok.vocab_size <= 60
    s = "the cat sat"
    ids = tok.encode(s)
    assert all(isinstance(i, int) and 0 <= i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == s          # lossless roundtrip on seen text

def test_training_is_deterministic():
    corpus = "aa bb ab ba aa bb ab ba " * 40
    a = BPETokenizer(); a.train(corpus, vocab_size=40)
    b = BPETokenizer(); b.train(corpus, vocab_size=40)
    assert a.merges == b.merges and a.vocab == b.vocab   # deterministic

def test_save_load_roundtrip_is_byte_stable(tmp_path):
    corpus = "hello world hello there world . " * 30
    tok = BPETokenizer(); tok.train(corpus, vocab_size=50)
    p = tmp_path / "bpe.json"; tok.save(str(p))
    tok2 = BPETokenizer.load(str(p))
    assert tok2.vocab == tok.vocab and tok2.merges == tok.merges
    assert tok2.encode("hello world") == tok.encode("hello world")
    # JSON is the only artifact -> self-contained at runtime
    json.loads(p.read_text())

def test_make_seq_dataset_dropin_compat():
    from sim.char_tokenizer import make_seq_dataset
    corpus = "the quick brown fox jumps over the lazy dog . " * 60
    tok = BPETokenizer(); tok.train(corpus, vocab_size=80)
    rng = np.random.default_rng(0)
    X, y = make_seq_dataset(corpus, tok, seq_len=8, n_samples=5, rng=rng)
    assert X.shape == (5, 8, tok.vocab_size) and y.shape == (5, 8)

def test_unknown_char_does_not_crash_encode():
    tok = BPETokenizer(); tok.train("abc abc abc " * 20, vocab_size=20)
    tok.encode("abc zzz")   # unseen 'z' must not raise (skip-or-byte)
```

**Step 2: Run — expect FAIL** (`ModuleNotFoundError: sim.bpe_tokenizer`).
Run: `pytest tests/test_bpe_tokenizer.py -q`

**Step 3: Minimal implementation** (`sim/bpe_tokenizer.py`)

```python
"""Pure deterministic Sennrich-2016 word-frequency BPE. No external
tokenizer dependency -> self-contained at runtime (the artifact is a
JSON merge table). Interface-compatible with sim.char_tokenizer
(.encode/.decode/.vocab_size/.encode_one_hot) so make_seq_dataset and
the validated trainer are drop-in DRY-reusable."""
from __future__ import annotations
import json
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np

_EOW = "</w>"


class BPETokenizer:
    def __init__(self):
        self.merges: List[Tuple[str, str]] = []
        self.vocab: List[str] = []          # index -> symbol; 0 == <UNK>
        self._sym_to_id: Dict[str, int] = {}

    # ---- training -----------------------------------------------------
    def train(self, corpus: str, vocab_size: int) -> None:
        words = [w for w in corpus.split() if w]
        wfreq = Counter(words)
        # word -> list of symbols (chars + end-of-word marker)
        splits = {w: list(w) + [_EOW] for w in wfreq}
        merges: List[Tuple[str, str]] = []
        base = sorted({c for w in wfreq for c in w}) + [_EOW]
        while len(set(base) | {"".join(m) for m in merges}) < vocab_size:
            pair_freq: Counter = Counter()
            for w, f in wfreq.items():
                s = splits[w]
                for i in range(len(s) - 1):
                    pair_freq[(s[i], s[i + 1])] += f
            if not pair_freq:
                break
            best_f = max(pair_freq.values())
            # deterministic: among max-freq pairs pick lexicographically smallest
            best = min(p for p, c in pair_freq.items() if c == best_f)
            merges.append(best)
            a, b = best
            ab = a + b
            for w in splits:
                s = splits[w]
                i = 0
                out = []
                while i < len(s):
                    if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                        out.append(ab)
                        i += 2
                    else:
                        out.append(s[i])
                        i += 1
                splits[w] = out
        self.merges = merges
        symbols = ["<UNK>"] + sorted(
            set(base) | {"".join(m) for m in merges})
        self.vocab = symbols
        self._sym_to_id = {s: i for i, s in enumerate(symbols)}

    # ---- encode/decode ------------------------------------------------
    def _encode_word(self, word: str) -> List[str]:
        s = list(word) + [_EOW]
        for a, b in self.merges:
            i = 0
            out = []
            while i < len(s):
                if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            s = out
        return s

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for w in text.split():
            for sym in self._encode_word(w):
                ids.append(self._sym_to_id.get(sym, 0))   # 0 == <UNK>
        return ids

    def decode(self, ids: List[int]) -> str:
        words, cur = [], []
        for i in ids:
            if not (0 <= i < len(self.vocab)):
                continue
            sym = self.vocab[i]
            if sym == "<UNK>":
                continue
            if sym.endswith(_EOW):
                cur.append(sym[: -len(_EOW)])
                words.append("".join(cur))
                cur = []
            else:
                cur.append(sym)
        if cur:
            words.append("".join(cur))
        return " ".join(words)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode_one_hot(self, text: str) -> np.ndarray:
        ids = self.encode(text)
        oh = np.zeros((len(ids), self.vocab_size), dtype=np.float32)
        for t, i in enumerate(ids):
            oh[t, i] = 1.0
        return oh

    # ---- persistence (JSON only -> self-contained) --------------------
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": [list(m) for m in self.merges],
                       "vocab": self.vocab}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        t = cls()
        t.merges = [tuple(m) for m in d["merges"]]
        t.vocab = list(d["vocab"])
        t._sym_to_id = {s: i for i, s in enumerate(t.vocab)}
        return t
```

**Step 4: Run — expect PASS.** `pytest tests/test_bpe_tokenizer.py -q`
(Fix the roundtrip: if `decode(encode(s)) == s` fails on multi-space, the tests use single-spaced text so `" ".join(words)` is exact. Keep tests single-spaced.)

**Step 5: Commit**
```bash
git add sim/bpe_tokenizer.py tests/test_bpe_tokenizer.py
git commit -m "feat(Generator-S): pure deterministic BPE tokenizer (self-contained JSON artifact; make_seq_dataset drop-in)"
```

---

### Task 2: Pure gate-scoring core (FIXED bars, adversarial-tested)

**Files:**
- Create: `research/runners/subword_lm_gate_core.py`
- Test: `tests/test_subword_lm_gate_core.py`

**Step 1: Write the failing tests** (incl. adversarial anti-cheat)

```python
import numpy as np
from research.runners.subword_lm_gate_core import (
    perplexity, shuffled_token_control, distinct_ngram_ratio,
    verbatim_copy_fraction, gs_verdict, gs_aggregate_multiseed,
    _GS_PPL_MARGIN, _GS_GENERALIZATION_MAX, _GS_DISTINCT_MIN,
    _GS_COPY_MAX, _GS_MIN_SEEDS,
)

def test_perplexity_is_exp_mean_nll():
    assert abs(perplexity([0.0, 0.0]) - 1.0) < 1e-9
    assert abs(perplexity([np.log(4)]) - 4.0) < 1e-6
    assert perplexity([]) == float("inf")          # no tokens -> inf

def test_shuffled_control_is_a_permutation_not_identity():
    ids = list(range(50))
    out = shuffled_token_control(ids, np.random.default_rng(1))
    assert sorted(out) == sorted(ids) and out != ids   # same multiset, reordered

def test_distinct_and_copy_metrics():
    assert distinct_ngram_ratio([1,2,3,1,2,3], n=3) == 2/4   # {123,231,312,123}->2 uniq/4
    g = [1,2,3,4]; tr = [9,1,2,3,8]
    # trigram (1,2,3) of g appears verbatim in tr -> 1 of 2 g-trigrams
    assert abs(verbatim_copy_fraction(g, tr, n=3) - 0.5) < 1e-9

def test_verdict_passes_only_when_all_fixed_bars_met():
    v = gs_verdict(heldout_ppl=10.0, shuffled_ppl=20.0, train_ppl=8.0,
                   distinct=0.7, copy_frac=0.05, has_shuffled_control=True)
    assert v["GATE"] == "PASS"            # 10<=0.8*20; 10<=1.5*8; .7>=.5; .05<=.2

def test_no_shuffled_control_is_fail_even_if_perfect():
    v = gs_verdict(1.0, 1e9, 1.0, 1.0, 0.0, has_shuffled_control=False)
    assert v["GATE"] == "FAIL"

def test_memorization_is_fail():
    # held-out >> train ppl -> memorization (the Inc-3 failure mode)
    v = gs_verdict(heldout_ppl=100.0, shuffled_ppl=1e9, train_ppl=5.0,
                   distinct=0.9, copy_frac=0.0, has_shuffled_control=True)
    assert v["GATE"] == "FAIL"

def test_degenerate_or_copy_generation_is_fail():
    assert gs_verdict(5,99,5,0.10,0.0,True)["GATE"] == "FAIL"   # distinct too low
    assert gs_verdict(5,99,5,0.9,0.80,True)["GATE"] == "FAIL"   # too much verbatim copy

def test_results_cannot_move_the_fixed_bars():
    assert (_GS_PPL_MARGIN, _GS_GENERALIZATION_MAX, _GS_DISTINCT_MIN,
            _GS_COPY_MAX, _GS_MIN_SEEDS) == (0.20, 1.5, 0.5, 0.20, 3)
    b = gs_verdict(1e-9, 1e9, 1e-9, 1.0, 0.0, True)
    assert b["bars"] == {"ppl_margin":0.20,"generalization_max":1.5,
                         "distinct_min":0.5,"copy_max":0.20}

def test_multiseed_requires_3_and_all_pass():
    P = {"GATE":"PASS"}; F = {"GATE":"FAIL"}
    assert gs_aggregate_multiseed([P,P,P])["GATE"] == "PASS"
    assert gs_aggregate_multiseed([P,F,P])["GATE"] == "FAIL"
    assert gs_aggregate_multiseed([P,P])["GATE"] == "FAIL"      # <3 seeds
    assert gs_aggregate_multiseed([])["GATE"] == "FAIL"
```

**Step 2: Run — expect FAIL** (module missing). `pytest tests/test_subword_lm_gate_core.py -q`

**Step 3: Minimal implementation** (`research/runners/subword_lm_gate_core.py`)

```python
"""Pure pre-registered scoring/verdict core for Generator-S. Mirrors
the song_g1_core pure-verdict DISCIPLINE (fixed bars, never tuned;
control load-bearing; >=3 seeds) but holds Generator-S's OWN frozen
constants -- it does NOT import or modify song_g1_core. Pure
numpy/stdlib; CPU-unit-testable; no IO, no heavy import."""
from __future__ import annotations
import math
from typing import Dict, List

# ---- PRE-REGISTERED FIXED BARS (frozen the moment this lands; NEVER
#      tuned/recomputed at gate time; results provably cannot move them)
_GS_PPL_MARGIN = 0.20          # held-out ppl <= (1-0.20)*shuffled ppl
_GS_GENERALIZATION_MAX = 1.5   # held-out ppl <= 1.5 * train ppl
_GS_DISTINCT_MIN = 0.5         # distinct-trigram ratio >= 0.5
_GS_COPY_MAX = 0.20            # verbatim train n-gram copy <= 0.20
_GS_MIN_SEEDS = 3              # single-seed is NOT a pass (whole-arc rule)


def perplexity(nll_per_token: List[float]) -> float:
    if not nll_per_token:
        return float("inf")
    return float(math.exp(sum(nll_per_token) / len(nll_per_token)))


def shuffled_token_control(token_ids, rng):
    out = list(token_ids)
    rng.shuffle(out)
    if out == list(token_ids) and len(set(token_ids)) > 1:
        out.reverse()                       # guarantee non-identity
    return out


def distinct_ngram_ratio(ids: List[int], n: int = 3) -> float:
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


def verbatim_copy_fraction(gen: List[int], train: List[int],
                            n: int = 8) -> float:
    if len(gen) < n:
        return 0.0
    tr = {tuple(train[i:i + n]) for i in range(len(train) - n + 1)}
    gg = [tuple(gen[i:i + n]) for i in range(len(gen) - n + 1)]
    if not gg:
        return 0.0
    return sum(1 for g in gg if g in tr) / len(gg)


def gs_verdict(heldout_ppl, shuffled_ppl, train_ppl, distinct,
               copy_frac, has_shuffled_control) -> Dict:
    real_structure = (has_shuffled_control and shuffled_ppl > 0
                      and heldout_ppl <= (1.0 - _GS_PPL_MARGIN)
                      * shuffled_ppl)
    generalizes = (train_ppl > 0
                   and heldout_ppl <= _GS_GENERALIZATION_MAX * train_ppl)
    non_degenerate = distinct >= _GS_DISTINCT_MIN
    not_copying = copy_frac <= _GS_COPY_MAX
    gate = bool(real_structure and generalizes and non_degenerate
                and not_copying)
    return {
        "GATE": "PASS" if gate else "FAIL",
        "real_structure_vs_shuffled": bool(real_structure),
        "generalizes_not_memorizes": bool(generalizes),
        "non_degenerate_generation": bool(non_degenerate),
        "not_verbatim_copying": bool(not_copying),
        "heldout_ppl": float(heldout_ppl),
        "shuffled_ppl": float(shuffled_ppl),
        "train_ppl": float(train_ppl),
        "distinct_trigram": float(distinct),
        "verbatim_copy_frac": float(copy_frac),
        "bars": {"ppl_margin": _GS_PPL_MARGIN,
                 "generalization_max": _GS_GENERALIZATION_MAX,
                 "distinct_min": _GS_DISTINCT_MIN,
                 "copy_max": _GS_COPY_MAX},
    }


def gs_aggregate_multiseed(per_seed_verdicts, min_seeds: int = _GS_MIN_SEEDS):
    n = len(per_seed_verdicts)
    n_pass = sum(1 for v in per_seed_verdicts if v.get("GATE") == "PASS")
    gate = bool(n >= int(min_seeds) and n_pass == n and n > 0)
    return {"GATE": "PASS" if gate else "FAIL", "n_seeds": n,
            "min_seeds": int(min_seeds), "n_pass": n_pass,
            "all_pass": (n > 0 and n_pass == n)}
```

**Step 4: Run — expect PASS.** `pytest tests/test_subword_lm_gate_core.py -q`

**Step 5: Commit**
```bash
git add research/runners/subword_lm_gate_core.py tests/test_subword_lm_gate_core.py
git commit -m "feat(Generator-S): pure FIXED-bar gate core (shuffled control load-bearing; >=3 seeds; adversarially pinned)"
```

**End of Phase A:** `git push origin HEAD && git push gitea HEAD`. Dispatch spec+quality review of Tasks 1+2 (pure cores; adversarial anti-cheat properties) before Phase B.

---

## PHASE B — integration (import/signature smoke + the gate itself; NOT contrived orchestration unit tests)

### Task 3: Corpus fetch + cache (network ONLY at fetch; idempotent)

**Files:**
- Create: `research/runners/corpus_fetch.py`
- Test: `tests/test_corpus_fetch_smoke.py` (import/signature + offline split logic only)

`fetch_corpus(name, max_bytes, out_dir="data/corpus", seed=42)`:
- `name="tinystories"` → URL `https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt` (the *valid* split ≈ tens of MB — bounded, sufficient for the cheap slice; primary). `name="wikitext"` → alt URL. Stream at most `max_bytes`.
- If `out_dir/<name>.txt` already cached → DO NOT re-download (idempotent).
- On any network failure → return the LOCAL `data/tinyshakespeare.txt` path with `degraded=True` flagged (honest: a shakespeare-only result is a weaker claim; the gate JSON records which corpus was used).
- `split_corpus(text, heldout_frac=0.1)` → `(train_text, heldout_text)` deterministic contiguous tail split (pure, unit-tested).
- ASCII clean: drop non-printable; collapse whitespace runs to single spaces (so BPE word-split + decode roundtrip is well-defined).

Smoke test asserts: module imports; `split_corpus("a "*100, 0.1)` returns disjoint train/heldout with ~90/10 sizes; `fetch_corpus` has the documented signature; offline path returns the shakespeare fallback when `out_dir` is a tmp empty dir and network is blocked (monkeypatch urllib to raise) → `degraded=True`.

**Steps:** failing smoke → impl → pass → commit
`git commit -m "feat(Generator-S): authorized public-corpus fetch+cache (idempotent; offline->shakespeare degraded flag; deterministic split)"`

---

### Task 4: Scaled subword SNN trainer (DRY mirror of train_shakespeare; kill-safe)

**Files:**
- Create: `research/runners/scaled_subword_lm_train.py`
- Test: `tests/test_scaled_subword_lm_train_smoke.py` (import/signature + a 2-epoch CPU micro-train returns decreasing loss + writes a resumable checkpoint)

`train_subword_lm(seed, corpus_path, vocab_size, hidden_layers, T, epochs, batch_size, lr, n_train_samples, ckpt_path, bpe_path, backend, verbose)`:
- Train (or `BPETokenizer.load(bpe_path)` if cached — BPE trained ONCE, reused across seeds for comparability) the BPE tokenizer on the TRAIN split; save merge table JSON.
- Build dataset via the UNMODIFIED `make_seq_dataset(train_text, bpe_tok, T, n_train_samples, rng)`.
- Build `V -> hidden_layers... -> V` `LIFLayerXP` stack with the EXACT init from `train_shakespeare` (std=2.0 first layer, 0.5 later, threshold 1.0, leak 0.95).
- Per-epoch loop = byte-for-byte the SHAPE of `train_shakespeare`'s loop (forward_unroll_xp → logits=last-layer spike sum → per-sample `cross_entropy_loss_np`/`softmax_grad_np` → `backward_unroll_xp` → `layer.W_in -= lr*grad`).
- Net-new ONLY: (a) subword vocab from BPE; (b) `sim.train_checkpoint.save_checkpoint(ckpt_path, epoch, weights=[l.W_in (host) for l in layers], rng_state, loss_history)` after EVERY epoch; on start `load_checkpoint`→`resume_epoch` to skip done epochs and restore weights; (c) `try/except cupy.cuda.memory.OutOfMemoryError` → halve batch_size and retry the epoch; (d) `except KeyboardInterrupt` → final atomic checkpoint flush + clean `return` (user frees GPU to game; re-run resumes). ASCII-only prints.
- Returns dict incl. `loss_history`, `final_loss`, `vocab_size`, `bpe_path`, `_layers` (for generation), `corpus_degraded`.

Smoke (CPU, tiny): `train_subword_lm(seed=42, corpus_path="data/tinyshakespeare.txt", vocab_size=64, hidden_layers=[32], T=16, epochs=2, batch_size=8, n_train_samples=40, backend="cpu")` → `final_loss <= initial_loss` and `load_checkpoint(ckpt)` is not None and `resume_epoch>=2`.

**Steps:** failing smoke → impl → pass → commit
`git commit -m "feat(Generator-S): scaled subword SNN trainer (DRY mirror of validated train_shakespeare; kill-safe atomic resume; OOM-halving)"`

---

### Task 5: Autoregressive generation (pure sampling + DRY forward)

**Files:**
- Create: `research/runners/subword_lm_generate.py`
- Test: `tests/test_subword_lm_generate.py` (PURE sampling logic only)

Pure `sample_next(logits, rng, temperature)` → int (argmax when temp==0; else temperature-softmax categorical). Unit-test: temp==0 ⇒ argmax deterministic; temp>0 ⇒ valid index, seeded-reproducible; degenerate logits handled. `generate(layers, bpe_tok, prompt, n_tokens, T, xp, rng, temperature)` reuses `forward_unroll_xp` (DRY) to roll the context window and `sample_next` to extend; returns token-id list + decoded string. (Generation function is integration — covered by the gate, not a contrived unit test; only `sample_next` is unit-tested.)

**Steps:** failing test (`sample_next`) → impl → pass → commit
`git commit -m "feat(Generator-S): autoregressive generation (pure temperature sampling unit-tested; forward reuses validated unroll)"`

---

### Task 6: Pre-registered multi-seed gate runner

**Files:**
- Create: `research/runners/subword_lm_gate.py`
- Test: `tests/test_subword_lm_gate_smoke.py` (import/signature + `--help`; the long gate is NOT unit-tested — project pattern)

`main()` argparse: `--seeds "42,43,44"` (≥3 enforced; `<3 → print NOT RUNNABLE, return 2`), `--corpus tinystories`, `--max-corpus-mb`, `--vocab-size`, `--hidden`, `--T`, `--epochs`, `--batch`, `--n-train-samples`, `--out`, `--ckpt` (isolated Generator-S namespace), `--bpe`. Per seed (kill-safe; resume skips completed seeds via a `.resume.json` like order_intrinsic_gate):
1. Fetch+cache corpus (Task 3) once; train BPE once (shared across seeds); split train/heldout.
2. Train the subword SNN on the TRAIN split (Task 4, kill-safe).
3. Train an IDENTICAL model on the **shuffled-token control** corpus (`shuffled_token_control` of the train token-ids, re-detokenized via the same BPE) — the load-bearing anti-cheat.
4. Compute held-out token NLL (teacher-forced forward over heldout ids) → `perplexity` (real model AND shuffled-control model); train-split perplexity (for the memorization bar).
5. Generate continuations of held-out prompts (Task 5) → `distinct_ngram_ratio`, `verbatim_copy_fraction` vs train.
6. `gs_verdict(...)` per seed; collect.
7. `gs_aggregate_multiseed(per_seed)`; write the full JSON (config, which corpus + `corpus_degraded`, per-seed metrics, frozen bars echoed, aggregate). ASCII verdict block. Exit 0 for PASS or FAIL (both valid computed results); exit 2 only if not runnable. Honest propagation is the CONTROLLER's post-run job (same contract as song_g1_gate / order_intrinsic_gate).

Smoke asserts import + arg parser + `<3 seeds → returns 2`. Long gate NOT run here.

**Steps:** failing smoke → impl → pass → commit
`git commit -m "feat(Generator-S): pre-registered multi-seed gate runner (shuffled-token control; FIXED bars via gate_core; kill-safe; honest-propagation is controller's job)"`

---

### Task 7: LOAD-BEARING no-harm verification

**Files:** Test: `tests/test_generator_s_noharm.py`

**Step 1: Write the test**
```python
"""LOAD-BEARING no-harm: Generator-S is PURELY ADDITIVE. The validated
deliverable + anti-cheat machinery are byte-untouched."""
import subprocess

def test_validated_modules_untouched_by_generator_s():
    # song_g1_core bars + order_intrinsic core + bridge must NOT appear
    # in Generator-S's diff range. Verified at review time via git; this
    # test pins the invariant that Generator-S imports NOTHING that
    # mutates them and does not shadow the g1 bars.
    import research.runners.subword_lm_gate_core as g
    assert not hasattr(g, "_G1_MARGIN") and not hasattr(g, "_G1_ABS_FLOOR")
    assert g._GS_PPL_MARGIN == 0.20            # its OWN frozen bars
    # importing Generator-S must not import song_g1_core
    import sys
    before = "research.runners.song_g1_core" in sys.modules
    import research.runners.subword_lm_gate_core  # noqa
    after = "research.runners.song_g1_core" in sys.modules
    assert before == after  # Generator-S core does not pull song_g1_core
```

Plus: run the FULL existing suite to confirm green (no regression):
`pytest tests/ -q -x` (expected: all pass; Generator-S only adds files).
And `git show --stat HEAD~6..HEAD -- sim/bridge.py research/runners/g20_*.py research/runners/song_g1_core.py research/runners/order_intrinsic_*.py` → expected EMPTY (Generator-S touched none).

**Step 5: Commit**
`git commit -m "test(Generator-S): LOAD-BEARING no-harm pin (validated deliverable + g1 bars byte-untouched; full suite green)"`
Then `git push origin HEAD && git push gitea HEAD`. Spec+quality review of Phase B (Tasks 3-7) before the decisive run.

---

### Task 8: Decisive multi-seed run + honest propagation (continuous arc — do NOT stop)

**Not a TDD task — the pre-registered decision + propagation.**

1. **Grounding first (falsify-cheaply):** run the gate on the LOCAL shakespeare (zero network), 3 seeds, tiny config, to prove the full pipeline (BPE→train→shuffled-control→heldout-ppl→generate→verdict) runs end-to-end and is interpretable. If the pipeline is broken → systematic-debugging, do NOT propagate a confounded verdict.
2. **Decisive run:** fetch the authorized public corpus (TinyStories valid split, bounded MB); launch `subword_lm_gate.py --seeds 42,43,44 ...` at the pre-registered config, **kill-safe, `run_in_background`** (user games/resumes — do NOT idle-wait; do parallel pre-staging of Generator-D while it trains). Monitor the log; the detached run will not auto-notify, poll periodically.
3. **Honest propagation EITHER way** (the non-negotiable discipline):
   - Findings doc `research/findings/2026-05-17-generator-S-subword-spiking-LM-<PASS|NEGATIVE>.md` (honest mechanism, no spin, no overclaim; bars echoed, never tuned; which corpus + degraded flag stated).
   - `webapp/capability_status.json` pillar (`status` VALIDATED if PASS else NEGATIVE; schema `{name,status,metric}`); `pytest tests/test_webapp_server.py -k capability_status` 6/6 green.
   - Commit + **push BOTH remotes**.
4. **Continuous autonomous arc — do NOT stop, do NOT ask, do NOT config-crank:**
   - **PASS ⇒ Generator-C:** scale (bigger corpus/params/context) then wire the corpus-pretrained spiking language cortex onto the validated grounded-memory + no-confabulation arch (new design doc → writing-plans → subagent-driven-development).
   - **FAIL ⇒ Generator-D:** knowledge distillation from a LOCAL open-weights teacher (Phi-3/Llama-3.2/Qwen2.5; teacher at TRAINING time only, runtime = trained spiking net) — new design doc, same FIXED-bar multi-seed gate discipline, pre-registered, immediately.
   - Either way: continue the design→plan→implement→gate→propagate→next loop autonomously until the user explicitly says stop.

---

## Notes
- DRY: validated BPTT core / `train_shakespeare` loop shape / `train_checkpoint` / `make_seq_dataset` / CE+softmax reused UNMODIFIED. `song_g1_core` UNTOUCHED (Generator-S owns its bars).
- YAGNI: this is the cheap decisive slice. Bigger corpus/params/context + Generator-C are LATER increments (noted, not built here). The multi-seed gate decides.
- TDD: pure logic (Tasks 1,2,5-sampling) is failing-test→impl→commit; integration (3,4,6) is import/signature smoke + the gate itself.
- @superpowers:systematic-debugging if the grounding pipeline breaks (root cause before any fix; never paper over to reach a verdict).
- @superpowers:subagent-driven-development for execution: fresh subagent per task, spec then quality review; trust-but-verify each subagent's actual `git diff` before marking complete.
