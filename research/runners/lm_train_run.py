"""Autonomous, incremental, RESUMABLE LM-training launcher (2026-07-21).

Mirrors `research/runners/develop_run.py` (the resumable, PAUSE-sentinel, per-increment-checkpoint, self-driven
pattern) -- retargeted from the artificial-life day-loop to the WKV language-cortex train-loop. Everything risky is
already de-risked (see lm_train_lib.py header); this is the orchestration:

    while not paused and budget remains:  train a chunk of N steps -> checkpoint -> benchmark -> log -> repeat

A resume loads the latest checkpoint and CONTINUES the exact trajectory (bit-exact; lm_train_lib.selftest). A PAUSE
sentinel stops cleanly at the next checkpoint boundary (zero work lost). NO `sim/` edit.

Usage:
  tokenize:  python -m research.runners.lm_train_run tokenize --root RUN [model/corpus flags]   (one-time setup)
  start   :  python -m research.runners.lm_train_run start    --root RUN [flags]                (fresh OR resume)
  pause   :  python -m research.runners.lm_train_run pause     --root RUN     (stop at next checkpoint boundary)
  resume  :  python -m research.runners.lm_train_run resume    --root RUN     (removes PAUSE, then run start again)
  status  :  python -m research.runners.lm_train_run status    --root RUN     (no GPU; step/tokens/ppl)
  selftest:  python -m research.runners.lm_train_run selftest                 (bit-exact resume proof, CPU)
  e2e     :  python -m research.runners.lm_train_run e2e       [--root RUN]    (full train->ckpt->bench->RESUME test)

The model/corpus config is FROZEN at first tokenize/start (config.json). A resume loads it and refuses a silent
architecture change; only runtime flags (--device/--amp/--compile/--max-increments/--max-tokens/--max-hours) vary.
"""
from __future__ import annotations
import os, sys, json, time, argparse
from pathlib import Path

import numpy as np
import torch

from research.runners import lm_train_lib as L
from research.runners.lm_train_lib import TrainConfig
from research.runners._lmtrain_stream_cursor_derisk import TokenStream

FIXED_PROMPTS = ["the meaning of", "in the year", "she was", "once there was a"]


# ------------------------------------------------------------------ helpers -----------------------------------------
def _paths(root: str):
    rd = Path(root)
    return rd, rd / "PAUSE", rd / "config.json", rd / "progress.jsonl", rd / "metadata.json", rd / "train.log"


def _log(logf: Path, msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(logf, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _cfg_from_args(a) -> TrainConfig:
    return TrainConfig(
        corpus_path=a.corpus_path, tokenizer=a.tokenizer, vocab_size=a.vocab_size, d_model=a.d_model,
        n_layers=a.n_layers, chunk_c=a.chunk_c, seq_len=a.seq_len, corpus_max_chars=a.corpus_max_chars,
        bpe_train_chars=a.bpe_train_chars, val_frac=a.val_frac, seed=a.seed, lr=a.lr,
        weight_decay=a.weight_decay, warmup_steps=a.warmup_steps, lr_decay_steps=a.lr_decay_steps,
        min_lr_ratio=a.min_lr_ratio, batch=a.batch, eval_seq_len=a.eval_seq_len,
        max_eval_seqs=a.max_eval_seqs, gen_tokens=a.gen_tokens)


def _load_or_freeze_config(cfg_path: Path, a) -> TrainConfig:
    """First call FREEZES the model/corpus config; a resume LOADS it (refusing a silent identity change)."""
    want = _cfg_from_args(a)
    if cfg_path.exists():
        frozen = json.loads(cfg_path.read_text())
        cfg = TrainConfig(**{**want.__dict__, **frozen})            # frozen identity wins over CLI
        # warn on attempted identity changes
        for k, v in frozen.items():
            if getattr(want, k) != v and k in ("d_model", "n_layers", "vocab_size", "seq_len", "corpus_path",
                                               "tokenizer", "chunk_c", "batch"):
                print(f"[lm_train] NOTE: --{k} {getattr(want,k)} ignored; lineage is frozen at {k}={v}.", flush=True)
        return cfg
    return want


# ------------------------------------------------------------------ status ------------------------------------------
def _status(root: str) -> int:
    rd, pause, cfg_p, prog_p, meta_p, _ = _paths(root)
    if not cfg_p.exists():
        print(f"[lm_train] no run at {rd} -- run `tokenize` then `start`.")
        return 0
    cfg = json.loads(cfg_p.read_text())
    step = tokens = 0; last = {}
    if prog_p.exists():
        lines = [l for l in prog_p.read_text().splitlines() if l.strip()]
        if lines:
            last = json.loads(lines[-1]); step = last.get("step", 0); tokens = last.get("tokens_seen", 0)
    resumable = L.has_checkpoint(rd)
    paused = os.path.exists(pause)
    print(f"[lm_train] run={rd}")
    print(f"  model: d_model={cfg['d_model']} n_layers={cfg['n_layers']} vocab={cfg['vocab_size']} T={cfg['seq_len']}")
    print(f"  step={step}  tokens_seen={tokens:,}  increments_logged={len(lines) if prog_p.exists() else 0}")
    if last:
        print(f"  latest: val_ppl={last.get('val_ppl')}  val_nll={last.get('val_nll')}  "
              f"mean_train_loss={last.get('mean_train_loss')}")
    print(f"  checkpoint present (resumable): {resumable}   paused: {paused}")
    return 0


# ------------------------------------------------------------------ the autonomous loop -----------------------------
def _start(a) -> int:
    root = a.root
    rd, pause, cfg_p, prog_p, meta_p, logf = _paths(root)
    rd.mkdir(parents=True, exist_ok=True)

    if os.path.exists(pause):
        print(f"[lm_train] PAUSE present -> not starting. remove it (or `resume`): {os.path.abspath(pause)}",
              flush=True)
        return 0

    cfg = _load_or_freeze_config(cfg_p, a)
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) corpus (tokenize if missing) -> frozen vocab + memmap token files
    V = L.tokenize_corpus(cfg, rd, log=lambda m: _log(logf, m))
    tokenizer = L._load_tokenizer(rd, cfg)
    train_tokens = L.load_tokens(rd, "train")
    held_ids = L.make_held_out(rd, cfg)

    # 2) model / opt / cosine-sched + resumable data stream
    L.set_all_rng(cfg.seed)
    model = L.build_model(cfg, V, device)
    opt, sched = L.build_opt_sched(model, cfg, device)
    stream = TokenStream(train_tokens, cfg.seq_len, cfg.batch, seed=cfg.seed)

    step, tokens_seen = 0, 0
    resumed = False
    if L.has_checkpoint(rd):
        step, tokens_seen = L.load_checkpoint(rd, model, opt, sched, stream, device)
        resumed = True
    if a.compile and device == "cuda":
        model = torch.compile(model)

    n_params = sum(p.numel() for p in (model.parameters()))
    _log(logf, f"{'RESUME' if resumed else 'START'} @ step={step} tokens={tokens_seen:,}  device={device} "
               f"params={n_params/1e6:.1f}M V={V} amp={a.amp} compile={a.compile}  "
               f"chunk_steps={a.chunk_steps} max_increments={a.max_increments}")

    best_nll = float("inf")
    if meta_p.exists():
        best_nll = json.loads(meta_p.read_text()).get("best_val_nll", float("inf"))

    t0 = time.time(); increments = 0
    while True:
        if os.path.exists(pause):
            _log(logf, f"PAUSE sentinel present -> stopping cleanly at step={step} (zero work lost)."); break
        if a.max_increments and increments >= a.max_increments:
            _log(logf, f"reached --max-increments={a.max_increments} for this invocation -> stop."); break
        if a.max_tokens and tokens_seen >= a.max_tokens:
            _log(logf, f"reached --max-tokens={a.max_tokens:,} -> stop."); break
        if a.max_hours and (time.time() - t0) / 3600.0 >= a.max_hours:
            _log(logf, f"reached --max-hours={a.max_hours} -> stop."); break

        # --- train one increment ---
        tc = time.time()
        mean_loss, toks = L.run_train_steps(model, opt, sched, stream, a.chunk_steps, V, device, amp=a.amp)
        step += a.chunk_steps; tokens_seen += toks
        train_s = time.time() - tc

        # --- checkpoint (atomic) BEFORE benchmark so a bench crash never loses training ---
        save_model = getattr(model, "_orig_mod", model)            # unwrap torch.compile for a portable state_dict
        L.save_checkpoint(rd, save_model, opt, sched, stream, step, tokens_seen, cfg, history_keep=a.history_keep)

        # --- benchmark on the FIXED held-out shard ---
        tb = time.time()
        m = L.benchmark(save_model, held_ids, V, device, tokenizer, cfg, prompts=FIXED_PROMPTS, seed=cfg.seed)
        bench_s = time.time() - tb

        is_best = m["val_nll"] < best_nll
        if is_best:
            best_nll = m["val_nll"]; L.mark_best(rd)

        rec = {"step": step, "tokens_seen": tokens_seen, "mean_train_loss": round(mean_loss, 4),
               "val_nll": m["val_nll"], "val_ppl": m["val_ppl"], "n_eval_tokens": m["n_eval_tokens"],
               "by_depth": m["by_depth"], "lr": round(sched.get_last_lr()[0], 6), "is_best": is_best,
               "train_s": round(train_s, 1), "bench_s": round(bench_s, 1), "wall_s": round(time.time() - t0, 1)}
        with open(prog_p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        meta_p.write_text(json.dumps({"config": cfg.frozen_dict(), "last_step": step,
                                      "tokens_seen": tokens_seen, "best_val_nll": best_nll}, indent=2))
        (rd / "samples.txt").write_text(f"step {step}  tokens {tokens_seen:,}  val_ppl {m['val_ppl']}\n" +
                                        "\n".join(f"  [{s['prompt']}] -> {s['text']}" for s in m["samples"]))
        _log(logf, f"step {step:>7} tok {tokens_seen:>12,} | train_loss {mean_loss:.4f} | val_ppl {m['val_ppl']:.2f}"
                   f" nll {m['val_nll']:.4f}{'  *best' if is_best else ''} | {train_s:.1f}s+{bench_s:.1f}s")
        increments += 1

    _log(logf, f"stopped after {increments} increment(s) this invocation, step={step}, wall {time.time()-t0:.1f}s. "
               f"resume: re-run `start`.  status: `status`.")
    return 0


# ------------------------------------------------------------------ end-to-end proof --------------------------------
def _e2e(a) -> int:
    """Full loop train->ckpt->benchmark->RESUME->continue on a TINY config; prove it resumes at the right step,
    continues (loss drops, cursor intact), logs accumulate, and PAUSE stops cleanly. Self-contained (CPU)."""
    import subprocess, shutil
    root = a.root or "bridges/lmtrain/e2e_test"
    shutil.rmtree(root, ignore_errors=True)
    py = [sys.executable, "-m", "research.runners.lm_train_run"]
    common = ["--root", root, "--corpus-path", "data/corpus/wikitext103.txt", "--tokenizer", "bpe",
              "--vocab-size", "400", "--corpus-max-chars", "300000", "--bpe-train-chars", "120000",
              "--d-model", "64", "--n-layers", "2", "--chunk-c", "8", "--seq-len", "48", "--batch", "16",
              "--device", "cpu", "--amp", "0", "--chunk-steps", "20", "--warmup-steps", "5",
              "--lr-decay-steps", "400", "--eval-seq-len", "48", "--max-eval-seqs", "60", "--gen-tokens", "16",
              "--history-keep", "3"]

    def run(cmd, tag):
        print(f"\n===== E2E {tag}: {' '.join(cmd[3:])} =====", flush=True)
        r = subprocess.run(py + cmd, capture_output=True, text=True)
        sys.stdout.write(r.stdout[-2500:]);
        if r.returncode != 0:
            sys.stderr.write(r.stderr[-3000:])
        return r.returncode == 0

    prog = Path(root) / "progress.jsonl"

    def records():
        return [json.loads(l) for l in prog.read_text().splitlines() if l.strip()] if prog.exists() else []

    ok = True
    # 1) tokenize once
    ok &= run(["tokenize"] + common, "tokenize")
    # 2) first START: 2 increments then stop at a checkpoint boundary (simulates being killed cleanly)
    ok &= run(["start"] + common + ["--max-increments", "2"], "start #1 (2 increments)")
    r1 = records()
    # 3) SIMULATED RESTART: a brand-new process `start` must RESUME from the latest checkpoint and CONTINUE
    ok &= run(["start"] + common + ["--max-increments", "2"], "start #2 (RESUME + 2 more)")
    r2 = records()

    # ---- checks ----
    steps = [r["step"] for r in r2]
    resumed_at = r1[-1]["step"] if r1 else None
    cont_step = (len(r2) == 4 and steps == [20, 40, 60, 80])
    # did the resumed process pick up EXACTLY where #1 stopped (no re-do, no skip)?
    resume_correct = (resumed_at == 40 and r2[2]["step"] == 60)
    # loss trend (tiny corpus, few steps -> should trend down over the 4 increments)
    losses = [r["mean_train_loss"] for r in r2]
    loss_drops = losses[-1] < losses[0]
    logs_accum = len(r2) == 4 and all("val_ppl" in r for r in r2)

    # 4) PAUSE stops cleanly (0 increments), then resume clears it
    ok &= run(["pause", "--root", root], "pause")
    paused_run = subprocess_capture(py + ["start"] + common + ["--max-increments", "5"])
    n_before = len(records())
    # (the paused start should add 0 increments)
    pause_clean = (len(records()) == n_before) and ("PAUSE present" in paused_run or "not starting" in paused_run)
    run(["resume", "--root", root], "resume (clear PAUSE)")

    print("\n" + "=" * 78)
    print("E2E RESULTS")
    print(f"  #1 stopped at step             : {resumed_at}")
    print(f"  #2 (resume) step trajectory    : {steps}   (expect [20,40,60,80])")
    print(f"  resumes at right step (no redo): {resume_correct}")
    print(f"  training continues (4 incr)    : {cont_step}")
    print(f"  benchmark logs accumulate      : {logs_accum}  (val_ppl each: {[r['val_ppl'] for r in r2]})")
    print(f"  train loss drops over run      : {loss_drops}  ({losses})")
    print(f"  PAUSE stops cleanly (0 incr)   : {pause_clean}")
    all_ok = ok and cont_step and resume_correct and loss_drops and logs_accum and pause_clean
    print(f"\n  E2E {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


def subprocess_capture(cmd):
    import subprocess
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    sys.stdout.write(out[-1500:])
    return out


# ------------------------------------------------------------------ argparse / main ---------------------------------
def _add_common(sp):
    sp.add_argument("--root", default="bridges/lmtrain/run1")
    sp.add_argument("--corpus-path", default="data/corpus/wikitext103.txt")
    sp.add_argument("--tokenizer", choices=["bpe", "byte"], default="bpe")
    sp.add_argument("--vocab-size", type=int, default=8000)
    sp.add_argument("--corpus-max-chars", type=int, default=0, help="0=whole file; else bound the tokenized slice")
    sp.add_argument("--bpe-train-chars", type=int, default=5_000_000)
    sp.add_argument("--val-frac", type=float, default=0.02)
    sp.add_argument("--d-model", type=int, default=512)
    sp.add_argument("--n-layers", type=int, default=6)
    sp.add_argument("--chunk-c", type=int, default=16, help="chunked-scan chunk length (recurrence)")
    sp.add_argument("--seq-len", type=int, default=256)
    sp.add_argument("--batch", type=int, default=32)
    sp.add_argument("--lr", type=float, default=3e-4)
    sp.add_argument("--weight-decay", type=float, default=0.1)
    sp.add_argument("--warmup-steps", type=int, default=200)
    sp.add_argument("--lr-decay-steps", type=int, default=100_000)
    sp.add_argument("--min-lr-ratio", type=float, default=0.1)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--eval-seq-len", type=int, default=128)
    sp.add_argument("--max-eval-seqs", type=int, default=200)
    sp.add_argument("--gen-tokens", type=int, default=40)
    # runtime-only (may vary per invocation)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--amp", type=int, default=1, help="bf16 autocast (GPU only)")
    sp.add_argument("--compile", type=int, default=0, help="torch.compile (GPU)")
    sp.add_argument("--chunk-steps", type=int, default=500, help="train steps per increment (checkpoint boundary)")
    sp.add_argument("--max-increments", type=int, default=0, help="stop after N increments this invocation (0=until paused/budget)")
    sp.add_argument("--max-tokens", type=int, default=0, help="budget cap (0=none)")
    sp.add_argument("--max-hours", type=float, default=0.0, help="budget cap (0=none)")
    sp.add_argument("--history-keep", type=int, default=5)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Autonomous incremental resumable LM training.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tokenize", "start"):
        _add_common(sub.add_parser(name))
    for name in ("pause", "resume", "status"):
        sp = sub.add_parser(name); sp.add_argument("--root", default="bridges/lmtrain/run1")
    st = sub.add_parser("selftest"); st.add_argument("--device", default="cpu")
    e = sub.add_parser("e2e"); e.add_argument("--root", default=None)
    a = ap.parse_args()

    if a.cmd == "status":
        return _status(a.root)
    if a.cmd == "pause":
        Path(a.root).mkdir(parents=True, exist_ok=True); Path(a.root, "PAUSE").touch()
        print(f"[lm_train] PAUSE created -> the run stops at the next checkpoint boundary: {os.path.abspath(Path(a.root,'PAUSE'))}")
        return 0
    if a.cmd == "resume":
        p = Path(a.root, "PAUSE")
        if p.exists(): p.unlink(); print(f"[lm_train] PAUSE cleared. Re-run `start --root {a.root}` to continue.")
        else: print("[lm_train] no PAUSE present; just run `start`.")
        return 0
    if a.cmd == "selftest":
        return 0 if L.selftest(device=a.device) else 1
    if a.cmd == "e2e":
        return _e2e(a)
    if a.cmd == "tokenize":
        cfg = _load_or_freeze_config(Path(a.root, "config.json"), a)
        L.tokenize_corpus(cfg, Path(a.root), log=print)
        return 0
    if a.cmd == "start":
        return _start(a)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
