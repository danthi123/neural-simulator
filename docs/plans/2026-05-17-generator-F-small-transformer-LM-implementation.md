---
type: plan
status: live
date: 2026-05-17
---

# Generator-F — Small From-Scratch Transformer LM — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task (continuous autonomous arc; do NOT stop to ask between tasks — user authorized a week of autonomous work 2026-05-17).

**Goal:** Decide, with the SAME unmodified HARDENED gate_core + a mandatory honest coherence read, whether a small from-scratch Transformer LM (self-contained/local/no-cheat, authorized corpus) reaches competent + coherent generation where 9 spiking/order-blind/statistical mechanisms failed — the evidence-mandated answer to "was the spiking substrate the wall?"

**Architecture:** 3 net-new files: a minimal self-contained PyTorch decoder-only GPT + a kill-safe BPTT trainer + a thin gate runner mirroring `subword_lm_gate.py`. Everything load-bearing (BPE, corpus_fetch, HARDENED gate_core, word-shuffle control) reused byte-UNMODIFIED.

**Tech Stack:** PyTorch (available, CUDA on the 3090), numpy/stdlib. Probe `bzzzmy1se` already grounded feasibility (2-layer toy: ppl 21.95 vs 513 + visibly coherent, 20s).

**Validated APIs reused UNMODIFIED (DRY — do NOT reimplement):**
- `sim.bpe_tokenizer.BPETokenizer`: `.train(text,vocab_size)`, `.encode(text)->list[int]`, `.decode(ids)->str`, `.vocab_size`, `.save/.load`.
- `research.runners.corpus_fetch.{fetch_corpus,split_corpus}`.
- `research.runners.subword_lm_gate_core` (HARDENED, FROZEN): `perplexity`, `distinct_ngram_ratio`, `verbatim_copy_fraction`, `gs_verdict(...,uniform_ppl=)` (fail-closed without `uniform_ppl`), `gs_aggregate_multiseed`. Bars `_GS_PPL_MARGIN=0.20`, `_GS_GENERALIZATION_MAX=1.5`, `_GS_DISTINCT_MIN=0.5`, `_GS_COPY_MAX=0.20`, `_GS_MIN_SEEDS=3`, `_GS_ABS_COMPETENCE_PPL_RATIO=1.0`. **byte-UNMODIFIED; NO new bar.**
- `research.runners.subword_lm_gate` — orchestration SHAPE + `_word_shuffle` to mirror (per-seed kill-safe `.resume.json`, ASCII verdict block, `<3 seeds -> exit 2`, honest-propagation-is-controller's-job).
- `sim.train_checkpoint` — the atomic `tmp + os.replace` IDIOM to mirror for torch state.

**MUST NOT touch (LOAD-BEARING no-harm):** `research/runners/subword_lm_gate_core.py` (frozen bars), `research/runners/song_g1_core.py`, `sim/bridge.py`, `research/runners/g20_*`, any validated runner. Generator-F is PURELY ADDITIVE new files. NO new bar.

**Anti-cheat (non-negotiable):** (a) the **causal mask** is load-bearing — a non-causal mask silently leaks future tokens = a perplexity cheat; Task 1's causal-correctness test pins it and gets a rigorous review. (b) a small Transformer on a bounded corpus CAN memorize/regurgitate — the hardened verbatim-copy + generalization + word-shuffle bars + mandatory smell-test are the adjudicators. (c) MANDATORY post-run smell-test: scrutinize a nominal PASS HARDER than a FAIL — verify from recorded JSON that copy_frac genuinely <=0.20, ppl genuinely competent, real beats word-shuffle >=20%, AND **read the actual generated text and characterize its TRUE coherence ceiling honestly** (small-Transformer TinyStories-class, NOT GPT-class, shown not described). NO overclaim; NO bar-tuning; recompute from recorded data only. The validated biology-grounded grounded-memory + no-confabulation remains the SEPARATE primary asset (Generator-F does not relitigate or replace it).

**ASCII-only prints (Windows cp1252). Commit after every task. Push both remotes (`origin`,`gitea`) after each phase.**

---

### Task 0: Falsify-cheaply grounding pin (green after Task 3)

**Files:** Create `tests/test_generator_f_grounding.py`

**Step 1: Write the test**

```python
"""Grounding: the generator_f_gate pipeline TURNS end-to-end on local
shakespeare (zero network) at a TINY config and produces an
interpretable verdict. Competence+coherence already grounded by probe
bzzzmy1se. Green after Task 3."""
import os
import subprocess
import sys
import json
import pytest


def test_generator_f_gate_pipeline_turns_local(tmp_path):
    if not os.path.exists("data/tinyshakespeare.txt"):
        pytest.skip("local grounding corpus absent")
    out = str(tmp_path / "f.json")
    ck = str(tmp_path / "f.ckpt")
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.generator_f_gate",
         "--seeds", "42,43,44", "--corpus", "data/tinyshakespeare.txt",
         "--vocab-size", "96", "--d-model", "32", "--n-layer", "1",
         "--n-head", "2", "--block-size", "16", "--steps", "20",
         "--batch-size", "8", "--gen-tokens", "30",
         "--eval-positions", "40", "--device", "cpu",
         "--out", out, "--ckpt", ck],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = json.loads(open(out, encoding="utf-8").read())
    assert d["n_seeds"] == 3 and "aggregate_verdict" in d
    assert all("verdict" in s for s in d["per_seed"])
    for s in d["per_seed"]:
        assert s["uniform_ppl"] == d["config"]["vocab_size"]
        assert "gen_sample" in s
```

**Step 2:** `pytest tests/test_generator_f_grounding.py -q` -> FAIL (module missing — green after Task 3; it IS the Task-3 gate).

**Step 5: Commit**
```bash
git add tests/test_generator_f_grounding.py
git commit -m "test(Generator-F): falsify-cheaply grounding pin (gate pipeline turns end-to-end; zero network) -- green after Task 3"
```

---

## PHASE A — pure/unit TDD

### Task 1: Minimal self-contained PyTorch decoder-only GPT (causal-mask correctness is LOAD-BEARING)

**Files:** Create `sim/tiny_transformer.py`; Test `tests/test_tiny_transformer.py`

**Step 1: Write the failing tests** (the causal-correctness + save/load-exact tests are load-bearing)

```python
import json
import torch
from sim.tiny_transformer import TinyGPT


def _mk(V=20, dm=16, nl=2, nh=2, bs=8):
    torch.manual_seed(0)
    m = TinyGPT(vocab_size=V, d_model=dm, n_layer=nl, n_head=nh,
                block_size=bs).eval()
    return m


def test_forward_shape_and_param_count():
    m = _mk()
    idx = torch.randint(0, 20, (3, 8))
    out = m(idx)
    assert tuple(out.shape) == (3, 8, 20)
    assert sum(p.numel() for p in m.parameters()) > 0


def test_deterministic_same_seed_same_output():
    a = _mk()
    b = _mk()
    idx = torch.randint(0, 20, (2, 8))
    assert torch.allclose(a(idx), b(idx), atol=0)


def test_causal_mask_no_future_leak():
    # LOAD-BEARING: changing a LATER input token must NOT change the
    # logits at any EARLIER position. A non-causal mask leaks the
    # future and silently cheats perplexity.
    m = _mk(V=20, bs=8)
    torch.manual_seed(1)
    idx = torch.randint(0, 20, (1, 8))
    base = m(idx).detach().clone()
    idx2 = idx.clone()
    idx2[0, 7] = (idx[0, 7].item() + 5) % 20
    alt = m(idx2).detach()
    assert torch.equal(base[:, :7, :], alt[:, :7, :])
    assert not torch.equal(base[:, 7, :], alt[:, 7, :])
    idx3 = idx.clone()
    idx3[0, 4] = (idx[0, 4].item() + 3) % 20
    alt3 = m(idx3).detach()
    assert torch.equal(base[:, :4, :], alt3[:, :4, :])


def test_save_load_roundtrip_logit_exact(tmp_path):
    m = _mk()
    idx = torch.randint(0, 20, (2, 8))
    before = m(idx).detach().clone()
    p = str(tmp_path / "tg")
    m.save(p)
    assert json.loads(open(p + ".meta.json").read())["d_model"] == 16
    m2 = TinyGPT.load(p).eval()
    assert torch.equal(m2(idx).detach(), before)
```

**Step 2:** `pytest tests/test_tiny_transformer.py -q` -> FAIL (module missing).

**Step 3: Implement** `sim/tiny_transformer.py` (copy precisely):

```python
"""Minimal self-contained decoder-only GPT (PyTorch). Generator-F
language model. Self-contained at runtime: the artifact is the
state_dict (.pt) + a sidecar hyperparam JSON; zero external
dependency, no external LLM, no runtime corpus. ASCII only."""
from __future__ import annotations
import json
import torch
import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x):
        n = x.size(1)
        # causal mask: True == NOT allowed to attend (j > i masked).
        mask = torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=x.device),
            diagonal=1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask,
                         need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layer=4, n_head=4,
                 block_size=128, dropout=0.0):
        super().__init__()
        self.cfg = {"vocab_size": int(vocab_size),
                    "d_model": int(d_model), "n_layer": int(n_layer),
                    "n_head": int(n_head),
                    "block_size": int(block_size),
                    "dropout": float(dropout)}
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_Block(d_model, n_head, dropout)
             for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        n = idx.size(1)
        pos = torch.arange(n, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos)[None, :, :])
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path + ".pt")
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(self.cfg, f)

    @classmethod
    def load(cls, path: str) -> "TinyGPT":
        with open(path + ".meta.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        m = cls(**cfg)
        m.load_state_dict(torch.load(path + ".pt",
                                     map_location="cpu"))
        return m
```

**Step 4:** `pytest tests/test_tiny_transformer.py -q` -> all 4 PASS. `test_causal_mask_no_future_leak` is LOAD-BEARING — if it fails, the mask leaks the future (a silent perplexity cheat); root-cause WITHOUT weakening (do NOT relax `torch.equal`; a wrong-diagonal/non-causal mask is a real bug to fix). STOP+report a genuine spec contradiction (do NOT fake-pass).

**Step 5: Commit**
```bash
git add sim/tiny_transformer.py tests/test_tiny_transformer.py
git commit -m "feat(Generator-F): minimal self-contained PyTorch decoder-only GPT (causal no-future-leak + save/load logit-exact pinned)"
```

**End of Phase A:** `git push origin HEAD && git push gitea HEAD`. Dispatch a RIGOROUS adversarial review of Task 1 focused on the causal mask (is `triu(diagonal=1)` the correct no-future-leak mask for `nn.MultiheadAttention`'s `attn_mask` True==masked convention? any path where position t attends to >t? dropout-off determinism; save/load bit-exactness) BEFORE Phase B — a future-token leak silently invalidates the decisive perplexity verdict.

---

## PHASE B — integration (import/signature smoke + the gate itself)

### Task 2: Kill-safe BPTT trainer

**Files:** Create `research/runners/tiny_transformer_train.py`; Test `tests/test_tiny_transformer_train_smoke.py`

`train_tiny_gpt(seed=42, corpus_path="data/tinyshakespeare.txt", vocab_size=512, d_model=256, n_layer=4, n_head=4, block_size=128, steps=12000, batch_size=64, lr=3e-4, ckpt_path=..., bpe_path=..., device="auto", print_every=500, verbose=True) -> dict`:
- Read `corpus_path`; BPE: load `bpe_path` if exists else train on the corpus + save (cached, reused across seeds for comparability).
- `device`: `"auto"` -> `"cuda" if torch.cuda.is_available() else "cpu"`; honor explicit `"cpu"`/`"cuda"`.
- Encode corpus -> 1-D `torch.long` on device. Random `(batch_size, block_size+1)` contiguous windows; x=`[:, :-1]`, y=`[:, 1:]`; loss = `torch.nn.functional.cross_entropy(logits.reshape(-1,V), y.reshape(-1))`. `AdamW(lr)`, `CosineAnnealingLR(T_max=steps)`, `clip_grad_norm_(1.0)`.
- **Kill-safe atomic resume (mirror the `sim.train_checkpoint` os.replace idiom for torch state):** every `print_every` steps AND at end, write `<ckpt>.pt.tmp` = `torch.save({"model":...,"optim":...,"sched":...,"step":...,"loss_history":...,"torch_rng":torch.get_rng_state()})` then `os.replace(tmp,<ckpt>.pt)`. On start: if `<ckpt>.pt` exists, load + restore model/optim/sched/step/rng + resume from `step`. `KeyboardInterrupt` -> flush atomic ckpt + clean `return` (resumable). `torch.cuda.OutOfMemoryError` or `RuntimeError` containing "out of memory" -> `torch.cuda.empty_cache()`, `batch_size=max(1,batch_size//2)`, retry.
- Returns dict mirroring `scaled_subword_lm_train.train_subword_lm`'s contract: `loss_history`, `initial_loss`, `final_loss`, `vocab_size`(=`tok.vocab_size`), `n_layer`, `device`, `interrupted`, `bpe_path`, `ckpt_path`, `_model`(TinyGPT, eval mode), `_tok`. ASCII-only prints.
- FIXED PRE-REGISTERED defaults baked here (frozen NOW, NOT tuned post-hoc): d_model 256, n_layer 4, n_head 4, block_size 128, steps 12000, batch_size 64, lr 3e-4, vocab_size 512. `main()` argparse mirrors them.

**Smoke** `tests/test_tiny_transformer_train_smoke.py`:

```python
import inspect
from research.runners.tiny_transformer_train import train_tiny_gpt


def test_signature():
    p = inspect.signature(train_tiny_gpt).parameters
    for k in ("seed", "corpus_path", "vocab_size", "d_model",
              "n_layer", "n_head", "block_size", "steps",
              "batch_size", "lr", "ckpt_path", "bpe_path", "device"):
        assert k in p


def test_tiny_cpu_train_reduces_loss_and_resumes(tmp_path):
    ck = str(tmp_path / "t.ckpt")
    bp = str(tmp_path / "t.bpe.json")
    r = train_tiny_gpt(seed=42,
                        corpus_path="data/tinyshakespeare.txt",
                        vocab_size=64, d_model=32, n_layer=1,
                        n_head=2, block_size=16, steps=5,
                        batch_size=8, ckpt_path=ck, bpe_path=bp,
                        device="cpu", verbose=False)
    assert r["final_loss"] is not None
    assert r["final_loss"] <= r["initial_loss"]
    assert r["vocab_size"] > 1 and "_model" in r and "_tok" in r
    import os
    assert os.path.exists(ck + ".pt")
    r2 = train_tiny_gpt(seed=42,
                        corpus_path="data/tinyshakespeare.txt",
                        vocab_size=64, d_model=32, n_layer=1,
                        n_head=2, block_size=16, steps=8,
                        batch_size=8, ckpt_path=ck, bpe_path=bp,
                        device="cpu", verbose=False)
    assert len(r2["loss_history"]) >= len(r["loss_history"])
```

**Procedure:** TDD smoke -> impl -> smoke passes. Verify `git status --porcelain` ZERO modifications to pre-existing files (`bpe_tokenizer.py`, `tiny_transformer.py`, `subword_lm_gate*.py` UNTOUCHED — reused by import only). Commit:
```bash
git add research/runners/tiny_transformer_train.py tests/test_tiny_transformer_train_smoke.py
git commit -m "feat(Generator-F): kill-safe PyTorch BPTT trainer (atomic os.replace resume; OOM-halve; FIXED pre-registered config)"
```

---

### Task 3: Generator-F gate runner (DRY mirror of `subword_lm_gate`)

**Files:** Create `research/runners/generator_f_gate.py`; Test `tests/test_generator_f_gate_smoke.py`

**Reference:** byte-mirror `research/runners/subword_lm_gate.py`'s orchestration with these changes:
- import `from research.runners.tiny_transformer_train import train_tiny_gpt`; train the REAL model on the train-split file AND an identical model on the `_word_shuffle`-d train split written to a temp file (BPE-invariant control). Copy `_word_shuffle` verbatim from `subword_lm_gate.py` (as Generator-D/E did).
- `_heldout_nll(model, tok, text, block_size, device, max_positions)`: encode text -> ids; stepped windows of length `block_size+1` (cap count by `max_positions`); `model.eval()`, `with torch.no_grad()`: `ce = torch.nn.functional.cross_entropy(model(x[:, :-1]).reshape(-1,V), y.reshape(-1))`; append `float(ce)`; return list. Combined with `subword_lm_gate_core.perplexity`. SAME logits semantics as training.
- generation: autoregressive multinomial sampling (temperature 1.0) seeded per seed, last `block_size` context window, `--gen-tokens` tokens -> `gen_ids`; `gen_sample = tok.decode(gen_ids)[:300]`.
- `v = gs_verdict(heldout_ppl=ho, shuffled_ppl=ctl, train_ppl=tr, distinct=distinct_ngram_ratio(gen_ids,3), copy_frac=verbatim_copy_fraction(gen_ids, tr_ids, 8), has_shuffled_control=True, uniform_ppl=tok.vocab_size)` — MUST pass `uniform_ppl` (hardened gate_core fail-closed without it).
- argparse defaults: `--seeds 42,43,44 --corpus tinystories --max-corpus-mb 8 --vocab-size 512 --d-model 256 --n-layer 4 --n-head 4 --block-size 128 --steps 12000 --batch-size 64 --gen-tokens 200 --eval-positions 2000 --device auto --out research/findings/raw/g11_bg/generator_f_gate.json --ckpt research/findings/raw/g11_bg/generator_f_gate.ckpt`. `<3 seeds -> exit 2`; per-seed kill-safe `.resume.json`; `gs_aggregate_multiseed`; JSON + ASCII verdict block; banner states the HONEST CEILING ("small-Transformer TinyStories-class coherent SIMPLE-STORY generation, explicitly NOT an LLM"); honest-propagation-is-controller's-job; `return 0`. ASCII-only. NO `song_g1_core`, NO `_GS_*=` bar assignment.

**Smoke** `tests/test_generator_f_gate_smoke.py`:

```python
import subprocess
import sys
import inspect


def test_import_passes_uniform_ppl_no_bar_no_g1():
    import research.runners.generator_f_gate as g
    src = inspect.getsource(g)
    assert "uniform_ppl=" in src
    assert "song_g1_core" not in src
    assert "_GS_PPL_MARGIN =" not in src
    assert "_GS_ABS_COMPETENCE_PPL_RATIO =" not in src
    import numpy as np
    s = g._word_shuffle("a b c d e f g h",
                        np.random.default_rng(1))
    assert sorted(s.split()) == list("abcdefgh")


def test_fewer_than_3_seeds_exit_2():
    r = subprocess.run([sys.executable, "-m",
        "research.runners.generator_f_gate", "--seeds", "42,43"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "NOT RUNNABLE" in r.stdout


def test_help():
    r = subprocess.run([sys.executable, "-m",
        "research.runners.generator_f_gate", "--help"],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "MULTI-SEED" in r.stdout
```

**Procedure:** TDD smoke -> mirror impl -> smoke passes; then run the Task-0 grounding pin (`pytest tests/test_generator_f_grounding.py -q` -> now GREEN, proving the pipeline turns end-to-end on local shakespeare zero-network). Verify `git status --porcelain` ZERO modifications to pre-existing files. Commit:
```bash
git add research/runners/generator_f_gate.py tests/test_generator_f_gate_smoke.py
git commit -m "feat(Generator-F): Transformer-LM gate runner (DRY mirror of subword_lm_gate; real+word-shuffle-control; passes uniform_ppl=V to HARDENED gate_core; kill-safe)"
```

---

### Task 4: LOAD-BEARING no-harm

**Files:** Create `tests/test_generator_f_noharm.py`

```python
"""LOAD-BEARING no-harm: Generator-F is PURELY ADDITIVE; the FROZEN
hardened gate_core bars + the validated assets are byte-untouched;
NO new bar; no song_g1_core pull; torch is a pre-existing dep."""
import sys


def test_hardened_bars_frozen_and_no_g1():
    import research.runners.subword_lm_gate_core as g
    assert (g._GS_PPL_MARGIN, g._GS_GENERALIZATION_MAX,
            g._GS_DISTINCT_MIN, g._GS_COPY_MAX, g._GS_MIN_SEEDS,
            g._GS_ABS_COMPETENCE_PPL_RATIO) == (0.20, 1.5, 0.5,
                                                0.20, 3, 1.0)
    assert not hasattr(g, "_G1_MARGIN")


def test_generator_f_does_not_pull_song_g1_core():
    before = "research.runners.song_g1_core" in sys.modules
    import sim.tiny_transformer  # noqa: F401
    import research.runners.tiny_transformer_train  # noqa: F401
    import research.runners.generator_f_gate  # noqa: F401
    after = "research.runners.song_g1_core" in sys.modules
    assert before == after


def test_torch_is_preexisting_dependency():
    import importlib.util
    assert importlib.util.find_spec("torch") is not None
```

Controller also verifies: representative existing suite green
(`pytest tests/test_subword_lm_gate_core.py tests/test_ngram_teacher.py tests/test_order_intrinsic_core.py tests/test_webapp_server.py -k "capability_status or gate_core or ngram or order_intrinsic" -q`) and
`git diff --stat <gen-F-range> -- research/runners/subword_lm_gate_core.py research/runners/song_g1_core.py sim/bridge.py research/runners/g20_*.py research/runners/subword_lm_gate.py sim/bpe_tokenizer.py` is EMPTY. torch is already importable — no `requirements.txt` change; if a pin is desired, FLAG for the user, do not add unilaterally.

**Commit + push both remotes.** Controller spec+quality check of Phase B (proportional; the anti-cheat firewall is the UNMODIFIED already-adversarially-reviewed hardened gate_core + the Task-1 causal-mask review).

---

### Task 5: Decisive multi-seed run + honest propagation (CONTROLLER, not a subagent)

1. **Grounding-first:** the Task-0 pin (local shakespeare, tiny config, cpu, zero network) already proves the pipeline turns + is interpretable. Re-confirm; toy verdict NOT propagated. If broken -> @superpowers:systematic-debugging.
2. **GPU per-step feasibility (pre-data, toward feasibility only, NEVER toward a pass):** time ~50 steps at the FIXED decisive config on the 3090; extrapolate 12000 steps x (real + word-shuffle-control) x 3 seeds. If genuinely infeasible as a cheap slice (>~6h), do a ONE-TIME documented pre-data sizing correction toward feasibility, frozen BEFORE any decisive result — NEVER toward a pass; bars + anti-cheat unchanged regardless of model size.
3. **Decisive run:** cached TinyStories, FIXED pre-registered config, `device auto` (CUDA), kill-safe `run_in_background` (user games/resumes; do parallel pre-staging of Generator-G while it trains; monitor via the log).
4. **MANDATORY anti-cheat smell-test BEFORE propagating (scrutinize a nominal PASS HARDER than a FAIL):** recompute from the recorded JSON (NO re-run, NO bar-tuning): `verbatim_copy_frac <= 0.20` every seed (a small Transformer CAN memorize a bounded corpus — load-bearing here)? held-out ppl genuinely competent (<< uniform 512; ideally probe-indicated ~15-25)? real genuinely beats word-shuffle control by >= 20%? distinct >= 0.5? **AND read the actual `gen_sample` text on every seed and characterize its TRUE coherence ceiling honestly** — small-Transformer TinyStories-class coherent simple-story English vs still-fragmentary; show verbatim samples; never spin. The hardened gate enforces the numeric bars; the human-judgment coherence read is the honest-ceiling adjudication.
5. **Honest propagation EITHER way:** findings doc `research/findings/2026-05-17-generator-F-small-transformer-LM-<PASS|NEGATIVE>.md`. **If scrutinized-genuine PASS:** report at the honest ceiling — "a self-contained, local, no-cheat small Transformer LM (trained on the authorized public corpus) reaches competent + coherent small-Transformer TinyStories-class generation and clears the SAME rigorous hardened gate 9 spiking/order-blind/statistical mechanisms failed; the user's north-star within the explicit small-LM ceiling; NOT GPT-class, NOT general reasoning; the validated biology-grounded grounded-memory + no-confabulation remains the SEPARATE primary asset; the spiking-substrate line is honestly terminally-negative for self-contained generation and Generator-F does not retro-justify it" — verbatim samples, never spun beyond bars + the honest coherence read. **If FAIL or honest-ceiling-too-low:** report the precise terminal decision-relevant finding; the converged conclusion stands; the validated grounded-memory asset is the deliverable. `webapp/capability_status.json` pillar (`status` VALIDATED only if scrutinized-genuine PASS with the small-LM ceiling EXPLICIT in the metric + verbatim samples, NOT spun as an LLM; BOUNDARY if genuine-gate-pass but coherence ceiling clearly sub-conversational; NEGATIVE if FAIL); schema `{name,status,metric}`; `pytest tests/test_webapp_server.py -k capability_status` 6/6 green; commit + push BOTH remotes; bars NOT tuned.
6. **Continuous arc — no stop/ask/config-crank:** scrutinized-genuine PASS => Generator-G (ground the small Transformer's generation on the validated no-confabulation memory — the honest realization of the conversational goal within the small-LM ceiling) — new design doc -> writing-plans -> subagent-driven-development. FAIL/ceiling-too-low => propagate the terminal decision-relevant finding + the converged honest conclusion of the whole arc; the validated grounded-memory + no-confab agent is the deliverable. Either way: continue autonomously, do NOT stop to ask.

---

## Notes
- DRY: BPE / corpus_fetch / HARDENED gate_core / train_checkpoint-idiom reused UNMODIFIED. NO new bar. `song_g1_core`/`subword_lm_gate_core` byte-UNTOUCHED.
- YAGNI: cheap decisive slice only. Generator-G synthesis + larger scaling are later increments (noted, not built).
- TDD: Task 1 strict failing-test->impl->commit with a RIGOROUS causal-mask adversarial review (load-bearing — a future leak is a silent perplexity cheat); Tasks 2-3 import/signature smoke + the gate itself.
- @superpowers:systematic-debugging if the grounding pipeline breaks.
- @superpowers:subagent-driven-development for execution; trust-but-verify each subagent's `git diff`; protected modules byte-empty in each commit-scoped diff.
- The Generator-S/D/E lesson is mandatory: scrutinize a nominal PASS HARDER than a FAIL; for a Transformer the chief cheats are future-token-leak (Task-1 causal pin + review) and memorization/regurgitation (hardened verbatim-copy + generalization + word-shuffle bars + the mandatory smell-test); ALWAYS read the actual generated text and report the true coherence ceiling — never overclaim beyond small-Transformer TinyStories-class, never spin as an LLM. The validated biology-grounded asset is the separate primary contribution and is untouched.
