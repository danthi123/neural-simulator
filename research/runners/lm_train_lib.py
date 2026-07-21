"""Core library for the autonomous, incremental, resumable LM-training workflow (2026-07-21).

ASSEMBLY ONLY -- every risky piece is already de-risked + committed; this wires them together:
  * MODEL / recurrence : `_lmtrain_chunked_scan.WKV` (chunked-scan multi-layer SSM; numerically == the loop,
                         gate 4.77e-07; ~30x faster; torch.compile-able). Reuse-by-import.
  * DATA cursor        : `_lmtrain_stream_cursor_derisk.TokenStream` (resumable, 10/10 restart + 11/11 rollover).
  * RESUME pattern     : `_lmtrain_resume_correctness_derisk` (bit-exact {model,opt,RNG,cursor} save/restore).
  * EVAL               : `_emerge_wkv_lm_derisk.eval_perdepth` (held-out per-depth NLL). Reuse-by-import.
  * TOKENIZER          : `sim.bpe_tokenizer.BPETokenizer` (frozen-vocab BPE) OR a byte fallback.

This module supplies: TrainConfig, corpus tokenize (memmap token files + frozen vocab), model/opt/cosine-sched
builders, ATOMIC checkpoint save/load of {model, optimizer, LR-scheduler, TokenStream cursor, torch+cuda+numpy+
python RNG, step, tokens_seen, config}, a fixed held-out benchmark (per-depth NLL + ppl + fixed-prompt samples),
and a bit-exact resume selftest. NO `sim/` edit.
"""
from __future__ import annotations
import os, json, math, random, shutil, time, copy
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn

from research.runners._lmtrain_chunked_scan import WKV as ChunkedWKV
from research.runners._lmtrain_stream_cursor_derisk import TokenStream
from sim.bpe_tokenizer import BPETokenizer


# ============================================================ config ================================================
@dataclass
class TrainConfig:
    # --- FROZEN per lineage (model + data identity; must not change across increments) ---
    corpus_path: str = "data/corpus/wikitext103.txt"
    tokenizer: str = "bpe"          # "bpe" | "byte"
    vocab_size: int = 8000          # bpe target vocab (byte -> 256)
    d_model: int = 512
    n_layers: int = 6
    chunk_c: int = 16               # chunked-scan chunk length (recurrence, not the train increment)
    seq_len: int = 256              # T
    corpus_max_chars: int = 0       # 0 = whole file; else bound the tokenized slice (test)
    bpe_train_chars: int = 5_000_000  # chars of the slice the BPE vocab is learned on
    val_frac: float = 0.02          # held-out fraction (contiguous tail of the tokenized stream)
    seed: int = 42
    # --- optimization (frozen for a clean cosine curve; changeable only via a fresh lineage) ---
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    lr_decay_steps: int = 100_000   # cosine decays lr->min over this many steps
    min_lr_ratio: float = 0.1
    batch: int = 32
    # --- benchmark ---
    eval_seq_len: int = 128
    max_eval_seqs: int = 200
    gen_tokens: int = 40

    def frozen_dict(self):
        # the identity fields a resume must not silently change
        keys = ["corpus_path", "tokenizer", "vocab_size", "d_model", "n_layers", "chunk_c", "seq_len",
                "corpus_max_chars", "bpe_train_chars", "val_frac", "seed", "lr", "weight_decay",
                "warmup_steps", "lr_decay_steps", "min_lr_ratio", "batch"]
        return {k: getattr(self, k) for k in keys}


# ============================================================ RNG capture ============================================
def set_all_rng(seed: int):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st):
    torch.set_rng_state(st["torch"]); np.random.set_state(st["numpy"]); random.setstate(st["python"])
    if torch.cuda.is_available() and st.get("cuda") is not None:
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except Exception:
            pass


# ============================================================ tokenizers ============================================
class ByteTokenizer:
    """Minimal byte tokenizer (vocab=256). Interface-compatible subset (encode/decode/vocab_size)."""
    vocab_size = 256

    def encode(self, text): return list(text.encode("utf-8", errors="ignore"))

    def decode(self, ids):
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="ignore")

    def save(self, path):
        Path(path).write_text(json.dumps({"kind": "byte"}))

    @classmethod
    def load(cls, path):
        return cls()


def _load_tokenizer(run_dir: Path, cfg: TrainConfig):
    p = run_dir / "tokenizer.json"
    if cfg.tokenizer == "byte":
        return ByteTokenizer.load(str(p)) if p.exists() else ByteTokenizer()
    return BPETokenizer.load(str(p))


# ============================================================ corpus pipeline ========================================
def _fast_bpe_encode(tok: BPETokenizer, text: str) -> np.ndarray:
    """Tokenize a whole slice with a per-unique-word memo (the standard trick -- pure-Python BPE is O(merges)/word)."""
    words = text.split()
    s2i = tok._sym_to_id
    cache = {}
    ids: List[int] = []
    for w in words:
        enc = cache.get(w)
        if enc is None:
            enc = [s2i.get(sym, 0) for sym in tok._encode_word(w)]
            cache[w] = enc
        ids.extend(enc)
    return np.asarray(ids, dtype=np.int32)


def tokenize_corpus(cfg: TrainConfig, run_dir: Path, log=print) -> int:
    """Train/freeze the tokenizer + tokenize the (bounded) corpus slice to memmap-able token files. Idempotent.
    Writes: tokenizer.json, tokens_train.npy, tokens_val.npy, config.json. Returns the frozen vocab size."""
    run_dir.mkdir(parents=True, exist_ok=True)
    tr_p, va_p, tk_p = run_dir / "tokens_train.npy", run_dir / "tokens_val.npy", run_dir / "tokenizer.json"
    cfg_p = run_dir / "config.json"

    if tr_p.exists() and va_p.exists() and tk_p.exists():
        tok = _load_tokenizer(run_dir, cfg)
        log(f"[tokenize] cached: vocab={tok.vocab_size}  train={np.load(tr_p, mmap_mode='r').shape[0]} "
            f"val={np.load(va_p, mmap_mode='r').shape[0]} tokens")
        return tok.vocab_size

    t0 = time.time()
    nbytes = None if cfg.corpus_max_chars <= 0 else cfg.corpus_max_chars
    with open(cfg.corpus_path, encoding="utf-8", errors="ignore") as f:
        text = f.read() if nbytes is None else f.read(nbytes)
    log(f"[tokenize] read {len(text):,} chars from {cfg.corpus_path}")

    if cfg.tokenizer == "byte":
        tok = ByteTokenizer()
        ids = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8).astype(np.int32)
    else:
        tok = BPETokenizer()
        tok.train(text[: cfg.bpe_train_chars], cfg.vocab_size)     # learn vocab on a (sub)slice
        log(f"[tokenize] BPE trained: vocab={tok.vocab_size} on {min(len(text), cfg.bpe_train_chars):,} chars "
            f"({time.time()-t0:.1f}s)")
        ids = _fast_bpe_encode(tok, text)                          # apply to the whole slice (memoized)
    tok.save(str(tk_p))

    n = len(ids)
    cut = int((1.0 - cfg.val_frac) * n)
    np.save(tr_p, ids[:cut]); np.save(va_p, ids[cut:])
    cfg_p.write_text(json.dumps(cfg.frozen_dict(), indent=2))
    log(f"[tokenize] {n:,} tokens -> train={cut:,} val={n-cut:,}  vocab={tok.vocab_size}  ({time.time()-t0:.1f}s)")
    return tok.vocab_size


def load_tokens(run_dir: Path, split="train"):
    return np.load(run_dir / f"tokens_{split}.npy", mmap_mode="r")


def make_held_out(run_dir: Path, cfg: TrainConfig) -> List[List[int]]:
    """A FIXED held-out shard: contiguous eval_seq_len sequences over tokens_val.npy (deterministic)."""
    val = np.asarray(load_tokens(run_dir, "val"))
    T = cfg.eval_seq_len
    n_seq = min(len(val) // T, cfg.max_eval_seqs)
    return [val[i * T:(i + 1) * T].tolist() for i in range(n_seq)]


# ============================================================ model / opt / sched ====================================
def build_model(cfg: TrainConfig, V: int, device: str) -> nn.Module:
    return ChunkedWKV(V, cfg.d_model, cfg.n_layers, block="chunked", C=cfg.chunk_c).to(device)


def _lr_lambda(cfg: TrainConfig):
    w, dec, floor = cfg.warmup_steps, cfg.lr_decay_steps, cfg.min_lr_ratio

    def fn(step):
        if w > 0 and step < w:
            return step / max(1, w)
        prog = min(1.0, (step - w) / max(1, dec - w))
        return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * prog))
    return fn


def build_opt_sched(model: nn.Module, cfg: TrainConfig, device: str):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.95), fused=(device == "cuda"))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg))
    return opt, sched


# ============================================================ train / eval ===========================================
def run_train_steps(model, opt, sched, stream: TokenStream, n_steps: int, V: int, device: str,
                    amp: bool, record=False):
    """Train n_steps from the stream; advances model/opt/sched/cursor. Returns (mean_loss, tokens_added[, losses])."""
    model.train()
    lossf = nn.CrossEntropyLoss()
    losses = []
    tot, cnt = 0.0, 0
    for _ in range(n_steps):
        batch = stream.next_batch()                                 # [B, T] numpy (resumable cursor)
        x = torch.as_tensor(np.ascontiguousarray(batch), dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(bool(amp) and device == "cuda")):
            y = model(x)[:, :-1]
            loss = lossf(y.reshape(-1, V).float(), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        lv = float(loss.detach()); tot += lv; cnt += 1
        if record:
            losses.append(lv)
    mean = tot / max(1, cnt)
    tokens = n_steps * stream.batch * stream.T
    return (mean, tokens, losses) if record else (mean, tokens)


def benchmark(model, held_ids, V, device, tokenizer, cfg: TrainConfig, prompts=None, seed=0):
    """FIXED held-out per-depth NLL (reuse eval_perdepth) + overall ppl + fixed-prompt generation samples."""
    from research.runners._emerge_wkv_lm_derisk import eval_perdepth
    model.eval()
    ce_by, cnt_by = eval_perdepth(model, None, held_ids, V, device, permute=False, memoryless=False, seed=seed)
    tot_ce = sum(ce_by[b] * cnt_by[b] for b in cnt_by); tot_n = sum(cnt_by.values())
    nll = tot_ce / max(1, tot_n)
    by_depth = {b: round(ce_by[b], 4) for b in sorted(ce_by)}
    samples = []
    for p in (prompts or []):
        samples.append({"prompt": p, "text": generate(model, tokenizer, p, cfg.gen_tokens, device, cfg, seed=seed)})
    return {"val_nll": round(nll, 4), "val_ppl": round(math.exp(min(nll, 30)), 2),
            "n_eval_tokens": int(tot_n), "by_depth": by_depth, "samples": samples}


def generate(model, tokenizer, prompt_text, n_tokens, device, cfg: TrainConfig, temp=0.8, seed=0):
    ids = tokenizer.encode(prompt_text) or [0]
    rng = np.random.default_rng(seed * 131 + 7)
    model.eval()
    with torch.no_grad():
        for _ in range(n_tokens):
            ctx = ids[-cfg.seq_len:]
            x = torch.as_tensor([ctx], dtype=torch.long, device=device)
            logits = model(x)[0, -1]
            p = torch.softmax(logits.float() / temp, -1).cpu().numpy()
            p = p / p.sum()
            ids.append(int(rng.choice(len(p), p=p)))
    return tokenizer.decode(ids)


# ============================================================ checkpointing ==========================================
def atomic_torch_save(obj, path: Path):
    tmp = str(path) + ".new"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(run_dir: Path, model, opt, sched, stream: TokenStream, step: int, tokens_seen: int,
                    cfg: TrainConfig, history_keep: int = 5):
    ckdir = run_dir / "ckpt"; (ckdir / "history").mkdir(parents=True, exist_ok=True)
    ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
          "cursor": stream.state(), "rng": capture_rng(), "step": int(step),
          "tokens_seen": int(tokens_seen), "config": cfg.frozen_dict()}
    atomic_torch_save(ck, ckdir / "latest.pt")
    hist = ckdir / "history" / f"ckpt_step{step}.pt"
    atomic_torch_save(ck, hist)
    # prune old history
    snaps = sorted((ckdir / "history").glob("ckpt_step*.pt"), key=lambda p: p.stat().st_mtime)
    for old in snaps[:-history_keep]:
        try: old.unlink()
        except OSError: pass


def mark_best(run_dir: Path):
    ck = run_dir / "ckpt" / "latest.pt"
    if ck.exists():
        atomic_torch_save(torch.load(ck, map_location="cpu", weights_only=False), run_dir / "ckpt" / "best.pt")


def load_checkpoint(run_dir: Path, model, opt, sched, stream: TokenStream, device: str):
    """Restore {model, opt, sched, cursor, RNG, step, tokens_seen} in place. Returns (step, tokens_seen).
    Loads to CPU (so the torch/cuda RNG ByteTensors stay CPU tensors, as set_rng_state requires) then moves the
    optimizer state onto `device` (model.load_state_dict copies cross-device into the already-on-device params)."""
    ck = torch.load(run_dir / "ckpt" / "latest.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])                         # copy_ handles CPU-state -> on-device params
    opt.load_state_dict(ck["opt"])
    for st in opt.state.values():                              # move optimizer moment tensors onto the device
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                st[k] = v.to(device)
    sched.load_state_dict(ck["sched"])
    stream.load_state(ck["cursor"]); restore_rng(ck["rng"])
    return int(ck["step"]), int(ck["tokens_seen"])


def has_checkpoint(run_dir: Path) -> bool:
    return (run_dir / "ckpt" / "latest.pt").exists()


# ============================================================ selftest (bit-exact resume) ===========================
def selftest(device="cpu", K=12) -> bool:
    """Bit-exact resume THROUGH THE REAL save/load machinery: uninterrupted 2K-step loss == interrupted (K + restore
    + K). Deterministic tiny config, CPU. This is `_lmtrain_resume_correctness_derisk` re-run through lm_train_lib."""
    import tempfile
    cfg = TrainConfig(d_model=32, n_layers=2, chunk_c=8, seq_len=16, batch=8, vocab_size=64,
                      lr=1e-2, warmup_steps=3, lr_decay_steps=200, min_lr_ratio=0.1, seed=42)
    V = cfg.vocab_size
    tokens = np.random.default_rng(0).integers(0, V, size=30000).astype(np.int32)

    def fresh_stream():
        return TokenStream(tokens, cfg.seq_len, cfg.batch, seed=123)

    # uninterrupted 2K
    set_all_rng(cfg.seed); m1 = build_model(cfg, V, device); o1, s1 = build_opt_sched(m1, cfg, device)
    st1 = fresh_stream()
    _, _, un = run_train_steps(m1, o1, s1, st1, 2 * K, V, device, amp=False, record=True)

    # interrupted: K -> checkpoint (real machinery) -> fresh objects (restart) -> restore -> K
    set_all_rng(cfg.seed); m2 = build_model(cfg, V, device); o2, s2 = build_opt_sched(m2, cfg, device)
    st2 = fresh_stream()
    _, tok2, first = run_train_steps(m2, o2, s2, st2, K, V, device, amp=False, record=True)
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td)
        save_checkpoint(rd, m2, o2, s2, st2, step=K, tokens_seen=tok2, cfg=cfg)
        m3 = build_model(cfg, V, device); o3, s3 = build_opt_sched(m3, cfg, device); st3 = fresh_stream()
        step, _ = load_checkpoint(rd, m3, o3, s3, st3, device)
        _, _, resumed = run_train_steps(m3, o3, s3, st3, K, V, device, amp=False, record=True)

    max_diff = max(abs(a - b) for a, b in zip(un[K:], resumed))
    cursor_ok = (step == K)
    print(f"  uninterrupted[K:2K] loss: {[round(x,4) for x in un[K:]]}")
    print(f"  resumed        [K:2K] loss: {[round(x,4) for x in resumed]}")
    print(f"  restored step = {step} (expected {K}); max|loss diff| = {max_diff:.2e}")
    ok = (max_diff < 1e-6) and cursor_ok
    print("  SELFTEST PASS (checkpoint->resume == uninterrupted, bit-exact)" if ok
          else f"  SELFTEST FAIL (diff {max_diff:.2e} / step {step})")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ok = selftest(device=a.device)
    raise SystemExit(0 if ok else 1)
