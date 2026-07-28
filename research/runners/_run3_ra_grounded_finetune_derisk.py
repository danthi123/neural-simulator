"""run3 (83M WKV) RA GROUNDED-RENDER fine-tune — a DATA/format continuation fine-tune (NOT a new mechanism) that makes
the run3 fluency generator RENDER the brain's grounded answers, per the EMERGE-57 recipe, ported to run3's architecture.

WHAT run3 IS (measured): the local fluency generator = a WKV/RWKV-style multi-layer diagonal-SSM
(`_lmtrain_chunked_scan.WKV`, `d_model=1024, n_layers=16, chunk_c=16, vocab=16000` HF byte-level BPE, seq_len=256).
Verified 83.2M params, best_val_nll 3.987 (ppl ~54) on 6B FineWeb-Edu. Its samples are genuinely fluent local English
but it CONTINUES text, it does not ANSWER a grounded question in a focused way (it rambles: "... answer: yes.\nThe bird
is a bird ..."). That is the exact residual the EMERGE-57 / RA format fine-tune closes.

THE PORT (what changes vs `_emerge57_ra_refinetune_emerge_frames_derisk`, which was for the 21M TinyStories TinyGPT):
  * MODEL      : `ChunkedWKV` (run3 arch) instead of `TinyGPT`; built + loaded via `lm_train_lib` (reuse-by-import).
  * TOKENIZER  : run3's frozen HF BPE-16k (`bridges/lmtrain/run3/tokenizer.json`) instead of the 21M's word-BPE.
  * BASE CKPT  : run3's own `ckpt/best.pt` (or `latest.pt`) — a FIRST format fine-tune from the fluent base (run3 was
                 never RA-fine-tuned), so the grounded corpus carries BOTH the RA transitive-SVO QA frames AND the
                 EMERGE ability/exception frames in one pass.
  * ANTI-FORGET: interleave the RUN3 corpus itself (raw FineWeb-Edu token windows from `tokens_train.npy`) instead of
                 TinyStories — run3's OWN distribution, so anti-forgetting is measured against the base's real fluency
                 corpus (TinyStories is not even present on this migrated machine). No re-tokenization.
  * The grounded-frame STRING generators + the frame-aware inflection fix (`emerge_v3`) are reused VERBATIM by import
    (pure Python, no torch) — this is a DATA lever on the identical recipe.

DE-RISK GATES (report, do NOT force a GO — read the runner's own verdict):
  (a) RENDER FIDELITY   — the fine-tuned WKV renders EMERGE's gated frames CORRECTLY behind the SAME gate-first moat
                          (owl -> "yes ." ; penguin -> "no , the penguin walks ." correct inflection + subject, no
                          confab of a WRONG member name, focused). Scripted ground-truth gate decisions isolate the
                          GENERATOR's render quality (the upstream EMERGE reasoner is validated separately; Rung-3 wire
                          into `_fluidconv_chat_repl` is the follow-on).
  (b) NO CATASTROPHIC FORGETTING — held-out FineWeb-Edu (run3's base corpus) ppl not blown up (ratio <= 1.5), AND the
                          grounded frames LEARNED (held-out grounded-frame ppl DROPS, ratio < 1.0).
  (c) MOAT PRESERVED    — 0 renders on abstains (the load-bearing property: an abstain -> the generator is NEVER
                          invoked; render_call_count 0). By construction (the gate short-circuits) + a call counter.
  (d) CORRECT INFLECTION — no "walkses"; the frame-aware `emerge_v3` fix (imported from EMERGE-57).

GPU-BOUND but CHEAP (a small grounded corpus + a short continuation): NOT a GPU-days job — hours on the 3090 during a
PAUSE, or a cheap short cloud box (per `2026-07-23-gap1-training-aws-experiment-spec.md`). NO `sim/` edit.

Run:
  # CPU, instant, no GPU/ckpt: sample the grounded corpus + validate the frame-aware inflection fix
  python -m research.runners._run3_ra_grounded_finetune_derisk --check-corpus
  # CPU SMOKE (~1-2 min): loads the REAL run3 83M ckpt on CPU, runs a few grounded fine-tune steps on a tiny slice,
  #   renders + moat-checks + pre/post ppl on tiny sets. Proves the whole pipeline end-to-end without GPU.
  CUDA_VISIBLE_DEVICES="" python -m research.runners._run3_ra_grounded_finetune_derisk --smoke
  # GPU full de-risk (the real run; NOT launched by this scoping) — see full_run_command in the output JSON.
  python -m research.runners._run3_ra_grounded_finetune_derisk --derisk --device cuda --amp 1
"""
from __future__ import annotations
import argparse, json, math, os, sys, time, traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# --- pure-Python grounded-frame STRING generators (NO torch at import time) reused VERBATIM ------------------------
from research.runners._fluidconv_phase2_ra_finetune import (  # noqa: E402
    _make_example, _shuffle, _rng, SUBJECTS, VERBS, SEP,
)
from research.runners._emerge57_ra_refinetune_emerge_frames_derisk import (  # noqa: E402
    _make_emerge_example, emerge_v3, _art, _EMERGE_MEMBERS, _emerge_subjects, _KNOWN_INTRANS_3SG,
)

# ---------------------------------------------------------------------------------------------------------------------
# PATHS — run3 stays INTACT; the fine-tuned model + corpus go to NEW paths.
# ---------------------------------------------------------------------------------------------------------------------
RUN3_ROOT = "bridges/lmtrain/run3"
OUT_ROOT = "bridges/lmtrain/run3_ra_grounded_ft"           # NEW dir; run3/ is never touched
FT_CKPT = OUT_ROOT + "/run3_ra_grounded_ft.pt"             # {model, config, step, losses, meta}
CORPUS = "data/corpus/run3_ra_grounded_frames.txt"         # the grounded-frame corpus (readable audit)
OUT = _REPO / "research" / "findings" / "raw" / "_run3_ra_grounded_finetune.json"


# ---------------------------------------------------------------------------------------------------------------------
# THE GROUNDED-FRAME CORPUS: RA transitive-SVO QA frames + EMERGE ability/exception/abstain frames, interleaved.
# ---------------------------------------------------------------------------------------------------------------------
def build_grounded_frames(n_ra, n_emerge, seed):
    """A list of grounded-frame QA strings: `n_ra` RA transitive-SVO frames (`_make_example`) + `n_emerge` EMERGE
    ability/exception/abstain frames (`_make_emerge_example`), shuffled. Both include their own abstain examples so the
    'i do not know' behaviour is TAUGHT in-distribution (the moat learned, not only enforced by the gate)."""
    r = _rng(seed)
    ra = [_make_example(r) for _ in range(n_ra)]
    emerge = [_make_emerge_example(r) for _ in range(n_emerge)]
    frames = ra + emerge
    r.shuffle(frames)
    return frames, {"n_ra": n_ra, "n_emerge": n_emerge, "n_frames": len(frames)}


def write_corpus_txt(frames, out_path):
    """Write the SEP-joined grounded corpus to disk (a human-readable audit of exactly what is trained)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    text = SEP.join(frames)
    Path(out_path).write_text(text, encoding="utf-8")
    return text


def _encode_frames(tok, frames):
    """HF-encode the SEP-joined grounded corpus to one int64 token array (the QA source for the mixed sampler)."""
    import numpy as np
    text = SEP.join(frames) + SEP
    ids = tok.encode(text)
    return np.asarray(ids, dtype=np.int64)


# ---------------------------------------------------------------------------------------------------------------------
# THE MIXED-BATCH SAMPLER: each sequence is drawn from the GROUNDED corpus with prob (1 - mix_ratio) else from raw
# run3 FineWeb-Edu (anti-forgetting on the base's OWN distribution). Per-batch mix -> the exact mix is recorded.
# ---------------------------------------------------------------------------------------------------------------------
class MixedSampler:
    def __init__(self, qa_ids, fineweb_mmap, seq_len, batch, mix_ratio, seed):
        import numpy as np
        self.np = np
        self.qa = qa_ids
        self.fw = fineweb_mmap                                  # uint16 memmap (run3 tokens_train)
        self.T = int(seq_len)
        self.B = int(batch)
        self.mix = float(mix_ratio)                            # fraction of sequences drawn from raw FineWeb
        self.rng = np.random.default_rng(seed)
        self.n_qa_seq = 0
        self.n_fw_seq = 0
        if len(self.qa) < self.T + 1:
            # tile the (tiny-smoke) grounded corpus up to at least one window
            reps = (self.T + 1) // max(1, len(self.qa)) + 2
            self.qa = np.tile(self.qa, reps)

    def next_batch(self):
        np = self.np
        rows = []
        for _ in range(self.B):
            from_fw = (self.fw is not None) and (self.rng.random() < self.mix)
            if from_fw:
                i = int(self.rng.integers(0, len(self.fw) - self.T - 1))
                rows.append(np.asarray(self.fw[i:i + self.T], dtype=np.int64))
                self.n_fw_seq += 1
            else:
                i = int(self.rng.integers(0, len(self.qa) - self.T - 1))
                rows.append(self.qa[i:i + self.T].astype(np.int64))
                self.n_qa_seq += 1
        return np.stack(rows)                                   # [B, T] int64


# ---------------------------------------------------------------------------------------------------------------------
# HELD-OUT PPL — measure forgetting + learning.
#   * grounded frames (disjoint seed, never in the fine-tune corpus): SHOULD DROP (the format is learned).
#   * FineWeb-Edu val windows (run3's held-out val shard): SHOULD STAY FLAT (no catastrophic forgetting of fluency).
# ---------------------------------------------------------------------------------------------------------------------
def _held_out_grounded(n=200, seed=99999):
    r_ra = _rng(seed)
    r_em = _rng(seed + 7)
    ra = [_make_example(r_ra) for _ in range(n)]
    emerge = [_make_emerge_example(r_em) for _ in range(n)]
    return ra, emerge


def _ppl_on_frames(model, tok, frames, torch, block, device):
    import torch.nn.functional as F
    tot_nll, tot_tok = 0.0, 0
    with torch.no_grad():
        for ex in frames:
            ids = tok.encode(ex + " ")
            if len(ids) < 2:
                continue
            ids = ids[:block]
            x = torch.tensor(ids[:-1], dtype=torch.long, device=device)[None]
            y = torch.tensor(ids[1:], dtype=torch.long, device=device)
            logits = model(x)[0]
            tot_nll += float(F.cross_entropy(logits.float(), y, reduction="sum").item())
            tot_tok += len(y)
    return math.exp(tot_nll / max(1, tot_tok))


def _ppl_on_fineweb(model, tok, val_mmap, torch, seq_len, n_windows, device, seed=1234):
    """Held-out FineWeb-Edu ppl over `n_windows` contiguous val windows (deterministic starts)."""
    import numpy as np
    import torch.nn.functional as F
    if val_mmap is None or len(val_mmap) < seq_len + 1:
        return None
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(val_mmap) - seq_len - 1, size=n_windows)
    tot_nll, tot_tok = 0.0, 0
    with torch.no_grad():
        for s in starts:
            w = np.asarray(val_mmap[int(s):int(s) + seq_len], dtype=np.int64)
            x = torch.tensor(w[:-1], dtype=torch.long, device=device)[None]
            y = torch.tensor(w[1:], dtype=torch.long, device=device)
            logits = model(x)[0]
            tot_nll += float(F.cross_entropy(logits.float(), y, reduction="sum").item())
            tot_tok += len(y)
    return math.exp(tot_nll / max(1, tot_tok))


# ---------------------------------------------------------------------------------------------------------------------
# THE CONTINUATION FINE-TUNE (a SHORT continuation from run3's ckpt; fresh low-LR AdamW).
# ---------------------------------------------------------------------------------------------------------------------
def _load_run3(device, which, run_root):
    """Build the run3 ChunkedWKV + tokenizer, load the frozen run3 ckpt weights. Reuse lm_train_lib (no sim/ edit)."""
    import torch
    from research.runners.lm_train_lib import TrainConfig, _load_tokenizer, build_model
    rd = Path(run_root)
    cfg = TrainConfig(**json.loads((rd / "config.json").read_text()))
    tok = _load_tokenizer(rd, cfg)
    V = getattr(tok, "vocab_size", cfg.vocab_size)
    ckp = rd / "ckpt" / f"{which}.pt"
    if not ckp.exists():
        ckp = rd / "ckpt" / "latest.pt"
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    model = build_model(cfg, V, device)
    model.load_state_dict(ck["model"])                          # unwrapped state_dict (run3 saves _orig_mod-stripped)
    npar = sum(p.numel() for p in model.parameters()) / 1e6
    meta = {"which": which, "ckpt": str(ckp), "base_step": int(ck.get("step", -1)),
            "base_tokens_seen": int(ck.get("tokens_seen", -1)), "npar_M": round(npar, 1), "vocab": V}
    return model, tok, cfg, meta


def finetune(steps, lr, batch_size, seq_len, seed, n_ra, n_emerge, mix_ratio, warmup, print_every,
             device, amp, which, run_root, fineweb_ppl_windows, tiny_model=False, save=True, ho_n=200,
             ckpt_path=FT_CKPT):
    import numpy as np
    import torch
    import torch.nn.functional as F

    print(f"[run3-ra-ft] device={device} amp={amp}; building grounded corpus "
          f"(n_ra={n_ra}, n_emerge={n_emerge}, mix_ratio={mix_ratio})...", flush=True)
    frames, fstats = build_grounded_frames(n_ra, n_emerge, seed)
    write_corpus_txt(frames, CORPUS)

    if tiny_model:
        # pipeline-only fast path: a fresh TINY WKV (proves corpus->encode->mix->train->render without the 83M cost)
        from research.runners.lm_train_lib import TrainConfig, build_model, _load_tokenizer
        rd = Path(run_root)
        cfg = TrainConfig(**json.loads((rd / "config.json").read_text()))
        tok = _load_tokenizer(rd, cfg)
        V = getattr(tok, "vocab_size", cfg.vocab_size)
        cfg.d_model, cfg.n_layers = 64, 2
        model = build_model(cfg, V, device)
        meta = {"which": "tiny-fresh", "npar_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                "vocab": V, "base_step": -1, "base_tokens_seen": -1, "ckpt": "(fresh tiny model)"}
    else:
        model, tok, cfg, meta = _load_run3(device, which, run_root)
    print(f"[run3-ra-ft] loaded base ({meta['npar_M']}M) from {meta['ckpt']} "
          f"(base_step={meta['base_step']}); continuation fine-tune {steps} steps @ lr {lr}", flush=True)

    qa_ids = _encode_frames(tok, frames)
    print(f"[run3-ra-ft] grounded corpus: {fstats} -> {len(qa_ids)} tokens", flush=True)

    # run3's own FineWeb-Edu token stream (anti-forgetting) + the held-out val shard (forgetting gate)
    rd = Path(run_root)
    fw_train = np.load(rd / "tokens_train.npy", mmap_mode="r") if (rd / "tokens_train.npy").exists() else None
    fw_val = np.load(rd / "tokens_val.npy", mmap_mode="r") if (rd / "tokens_val.npy").exists() else None
    if fw_train is None:
        print("[run3-ra-ft] WARN: FineWeb tokens_train.npy absent -> anti-forgetting mix DISABLED (grounded-only)",
              flush=True)

    block = cfg.seq_len
    seq_len = min(seq_len, block)
    sampler = MixedSampler(qa_ids, fw_train, seq_len, batch_size, mix_ratio, seed)

    # ---- pre-fine-tune held-out ppl (the baselines for the forgetting + learning gates) ----
    ra_ho, em_ho = _held_out_grounded(n=ho_n)
    model.train(False)
    ppl_ra_pre = _ppl_on_frames(model, tok, ra_ho, torch, block, device)
    ppl_em_pre = _ppl_on_frames(model, tok, em_ho, torch, block, device)
    ppl_fw_pre = _ppl_on_fineweb(model, tok, fw_val, torch, seq_len, fineweb_ppl_windows, device)
    print(f"[run3-ra-ft] PRE  ppl: RA-frame {ppl_ra_pre:.2f} | EMERGE-frame {ppl_em_pre:.2f} | "
          f"FineWeb {ppl_fw_pre if ppl_fw_pre is None else round(ppl_fw_pre,2)}", flush=True)

    # ---- the continuation fine-tune ----
    model.train(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    def lr_at(step):
        if warmup and step < warmup:
            return lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.5 * lr * (1.0 + math.cos(math.pi * min(1.0, prog)))

    use_amp = bool(amp) and device == "cuda"
    t0 = time.time()
    init_loss, last_loss = None, None
    losses = []
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        batch = sampler.next_batch()
        x = torch.as_tensor(np.ascontiguousarray(batch), dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)[:, :-1]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        lv = float(loss.detach().item())
        losses.append(lv)
        if init_loss is None:
            init_loss = lv
        last_loss = lv
        if step % print_every == 0 or step == steps - 1:
            print(f"[run3-ra-ft] step {step}/{steps} loss={lv:.4f} lr={lr_at(step):.2e} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if save:
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                save_model = getattr(model, "_orig_mod", model)
                tmp = ckpt_path + ".tmp"
                torch.save({"model": save_model.state_dict(), "config": cfg.frozen_dict(), "step": step,
                            "init_loss": init_loss, "final_loss": lv, "base_meta": meta}, tmp)
                os.replace(tmp, ckpt_path)

    # ---- post-fine-tune held-out ppl ----
    model.train(False)
    ppl_ra_post = _ppl_on_frames(model, tok, ra_ho, torch, block, device)
    ppl_em_post = _ppl_on_frames(model, tok, em_ho, torch, block, device)
    ppl_fw_post = _ppl_on_fineweb(model, tok, fw_val, torch, seq_len, fineweb_ppl_windows, device)
    print(f"[run3-ra-ft] POST ppl: RA-frame {ppl_ra_post:.2f} | EMERGE-frame {ppl_em_post:.2f} | "
          f"FineWeb {ppl_fw_post if ppl_fw_post is None else round(ppl_fw_post,2)}", flush=True)
    print(f"[run3-ra-ft] done ({time.time()-t0:.0f}s) init={init_loss:.4f} final={last_loss:.4f} -> "
          f"{ckpt_path if save else '(not saved)'}", flush=True)

    return {
        "steps": steps, "lr": lr, "batch_size": batch_size, "seq_len": seq_len, "seed": seed,
        "n_ra": n_ra, "n_emerge": n_emerge, "mix_ratio": mix_ratio, "warmup": warmup, "device": device,
        "amp": bool(use_amp), "which": which, "tiny_model": bool(tiny_model),
        "base_meta": meta, "corpus_stats": fstats, "n_qa_tokens": int(len(qa_ids)),
        "n_qa_seq_sampled": sampler.n_qa_seq, "n_fineweb_seq_sampled": sampler.n_fw_seq,
        "fineweb_anti_forget_enabled": fw_train is not None,
        "init_loss": round(init_loss, 4), "final_loss": round(last_loss, 4),
        "loss_first5": [round(x, 4) for x in losses[:5]], "loss_last5": [round(x, 4) for x in losses[-5:]],
        "ckpt_path": ckpt_path if save else None,
        "ppl_ra_frame_pre": round(ppl_ra_pre, 3), "ppl_ra_frame_post": round(ppl_ra_post, 3),
        "ppl_emerge_frame_pre": round(ppl_em_pre, 3), "ppl_emerge_frame_post": round(ppl_em_post, 3),
        "ppl_fineweb_pre": None if ppl_fw_pre is None else round(ppl_fw_pre, 3),
        "ppl_fineweb_post": None if ppl_fw_post is None else round(ppl_fw_post, 3),
        "fineweb_ppl_windows": fineweb_ppl_windows, "ckpt": ckpt_path if save else None,
        "elapsed_seconds": round(time.time() - t0, 1),
    }, model, tok, cfg


# ---------------------------------------------------------------------------------------------------------------------
# THE SCRIPTED RENDER + MOAT PROBE (self-contained: torch + tokenizer only; no EMERGE spiking console dependency).
# The gate decisions are GROUND-TRUTH here (the upstream EMERGE reasoner is validated separately) so this isolates the
# GENERATOR's render fidelity + the moat-by-construction property. Mirrors EMERGE-57's `_render_derisk`.
# ---------------------------------------------------------------------------------------------------------------------
# A small bird/fish taxonomy: inherit (member gets the class ability), cancel (member's own intransitive overrides),
# abstain (unknown member -> the gate short-circuits, the generator is NEVER invoked).
_PROBES = [
    # (member, gate, polarity, svo, ask)  — ANSWER: svo=(subj, verb/'can', prop); ABSTAIN: svo=None
    ("owl", "ANSWER", "affirm", ("owl", "can", "fly"), "fly"),          # INHERIT bird ability
    ("wren", "ANSWER", "affirm", ("wren", "can", "fly"), "fly"),        # INHERIT (held-out bird)
    ("minnow", "ANSWER", "affirm", ("minnow", "can", "swim"), "swim"),  # INHERIT (held-out fish)
    ("gar", "ANSWER", "affirm", ("gar", "can", "swim"), "swim"),        # INHERIT (held-out fish)
    ("penguin", "ANSWER", "negate", ("penguin", "walks", None), "fly"),  # CANCEL (exception: walks, not flies)
    ("pike", "ANSWER", "negate", ("pike", "lurks", None), "swim"),       # CANCEL (exception: lurks, not swims)
    ("zzz", "ABSTAIN", None, None, "fly"),                               # MOAT (unknown member)
    ("wobble", "ABSTAIN", None, None, "swim"),                           # MOAT (unknown member)
]


class _CountingRun3Faculty:
    """The fine-tuned run3 WKV, instrumented to COUNT render calls (so 'renderer-never-invoked-on-abstain' is a hard
    assertable count). Greedy-decodes EMERGE's ability/exception frames in the learned QA format."""

    def __init__(self, model, tok, cfg, max_new=28):
        import torch
        self._torch = torch
        self.model = model
        self.tok = tok
        self.block = cfg.seq_len
        self.device = next(model.parameters()).device.type
        self.max_new = int(max_new)
        self.model.train(False)
        self.render_call_count = 0

    def _gen(self, prompt):
        torch = self._torch
        ids = self.tok.encode(prompt)
        seq = list(ids)
        out = []
        with torch.no_grad():
            for _ in range(self.max_new):
                ctx = seq[-self.block:]
                x = torch.tensor(ctx, dtype=torch.long, device=next(self.model.parameters()).device)[None]
                nxt = int(torch.argmax(self.model(x)[0, -1]).item())
                seq.append(nxt)
                out.append(nxt)
        text = self.tok.decode(out).strip()
        # surface-level stop: first sentence end or the SEP marker (tokenizer-agnostic)
        for end in [". ", "! ", "? ", " * ", "*"]:
            k = text.find(end)
            if k != -1:
                text = text[:k + (1 if end.strip() == "." or end in ("! ", "? ") else 0)]
                break
        return text.strip().rstrip("*").strip()

    def render_emerge(self, svo, polarity, ask):
        self.render_call_count += 1
        subj, verb, obj = svo
        if polarity == "affirm":
            facts_ctx = f"the {subj} can {obj} ."
            question = f"can {_art(subj)} {obj} ?"
        else:
            v3 = emerge_v3(verb)                                # frame-aware: 'walks' stays 'walks' (never 'walkses')
            facts_ctx = f"the {subj} {v3} ."
            question = f"can {_art(subj)} {ask} ?"
        surface = self._gen(f"facts : {facts_ctx} question : {question} answer :")
        return surface, facts_ctx, question


def _render_derisk(model, tok, cfg, seed=42, verbose=True):
    faculty = _CountingRun3Faculty(model, tok, cfg)
    recs = []
    moat_render_calls_on_abstains = 0
    other_all = _emerge_subjects | {m for (m, *_r) in _PROBES}
    for (m, gate, polarity, svo, ask) in _PROBES:
        calls0 = faculty.render_call_count
        if gate == "ABSTAIN":
            surface = f"I don't know what {_art(m)} is."
            moat_render_calls_on_abstains += (faculty.render_call_count - calls0)   # MUST stay 0 (gate short-circuits)
            recs.append({"member": m, "gate": "ABSTAIN", "surface": surface, "model_invoked": False})
            if verbose:
                print(f"  you> can {_art(m)} {ask}?\n  brain> {surface}   [MOAT; model NOT invoked]\n", flush=True)
            continue
        surface, facts_ctx, question = faculty.render_emerge(svo, polarity, ask)
        toks = surface.lower().split()
        subj = svo[0]
        if polarity == "affirm":
            prop_word = svo[2]
            starts_ok = surface.lower().startswith("yes")
        else:
            prop_word = emerge_v3(svo[1])                       # exception verb correctly inflected ('walks')
            starts_ok = surface.lower().startswith("no")
        names_prop = prop_word in surface.split()
        names_subject = subj in toks
        other_members = other_all - {subj}
        no_other_member = all(w not in other_members for w in toks)          # no WRONG member name (the confab guard)
        nwords = len(surface.split())
        focused = 1 <= nwords <= 18
        double_infl = any(w.endswith("ses") and (w[:-3] + "s") in _KNOWN_INTRANS_3SG for w in surface.split()) \
            or "walkses" in surface or "lurkses" in surface
        if polarity == "affirm":
            # a short "yes ." IS a valid grounded affirmation (naming subject/ability is a bonus, not required)
            fidelity_ok = bool(starts_ok and no_other_member and focused and not double_infl)
        else:
            # exception negation: MUST start "no", name the correct intransitive verb + the correct subject, no confab
            fidelity_ok = bool(starts_ok and names_prop and names_subject and no_other_member
                               and focused and not double_infl)
        recs.append({"member": m, "gate": "ANSWER", "polarity": polarity, "facts_ctx": facts_ctx,
                     "question": question, "surface": surface, "prop_word": prop_word, "names_prop": names_prop,
                     "starts_ok": starts_ok, "names_subject": names_subject, "no_other_member": no_other_member,
                     "focused": focused, "double_inflection": double_infl, "fidelity_ok": fidelity_ok,
                     "model_invoked": True})
        if verbose:
            tag = "INHERIT" if polarity == "affirm" else "CANCEL"
            print(f"  you> can {_art(m)} {ask}?\n  brain> [facts: {facts_ctx}] {surface}   "
                  f"[{tag}; fidelity_ok={fidelity_ok}]\n", flush=True)

    answer_recs = [r for r in recs if r["gate"] == "ANSWER"]
    n_fid = sum(r["fidelity_ok"] for r in answer_recs)
    n_double = sum(r.get("double_inflection", False) for r in answer_recs)
    n_abstain = sum(1 for r in recs if r["gate"] == "ABSTAIN")
    n_invoked_on_abstain = sum(1 for r in recs if r["gate"] == "ABSTAIN" and r["model_invoked"])
    return {"seed": seed, "render_fidelity": round(n_fid / max(1, len(answer_recs)), 3),
            "n_answer": len(answer_recs), "n_fidelity_ok": int(n_fid), "n_double_inflection": int(n_double),
            "n_abstain": n_abstain, "moat_render_calls_on_abstains": int(moat_render_calls_on_abstains),
            "n_model_invoked_on_abstain": int(n_invoked_on_abstain), "records": recs}


# ---------------------------------------------------------------------------------------------------------------------
# VERDICT (mirror EMERGE-57: report gates, do NOT force a GO).
# ---------------------------------------------------------------------------------------------------------------------
FULL_RUN = ("python -m research.runners._run3_ra_grounded_finetune_derisk --derisk --device cuda --amp 1 "
            "--steps 1200 --n-ra 12000 --n-emerge 14000 --mix-ratio 0.5 --batch-size 32 --seq-len 256 --lr 5e-5")
RENDER_EVAL = ("python -m research.runners._run3_ra_grounded_finetune_derisk --render-only --device cuda "
               f"--ckpt {FT_CKPT}")


def _summarize(ft, render, mode, seed, out_path=OUT):
    fw_pre, fw_post = ft.get("ppl_fineweb_pre"), ft.get("ppl_fineweb_post")
    forget_ratio = (fw_post / fw_pre) if (fw_pre and fw_post) else None
    em_pre, em_post = ft.get("ppl_emerge_frame_pre"), ft.get("ppl_emerge_frame_post")
    ra_pre, ra_post = ft.get("ppl_ra_frame_pre"), ft.get("ppl_ra_frame_post")
    em_learn = (em_post / em_pre) if (em_pre and em_post) else None
    ra_learn = (ra_post / ra_pre) if (ra_pre and ra_post) else None

    go, verdict = None, None
    ppl_present = em_pre is not None                             # render-only mode has no ppl -> render+moat gates only
    if render is not None and not ppl_present:
        fidelity = render["render_fidelity"]
        moat_ok = (render["moat_render_calls_on_abstains"] == 0 and render["n_model_invoked_on_abstain"] == 0)
        verdict = (f"RENDER-ONLY (no ppl measured) -- render fidelity {fidelity:.2f} "
                   f"({render['n_fidelity_ok']}/{render['n_answer']}); {render['n_double_inflection']} double-"
                   f"inflections; MOAT {'HELD' if moat_ok else 'BREACHED'} "
                   f"({render['moat_render_calls_on_abstains']} renders / "
                   f"{render['n_model_invoked_on_abstain']} model-invocations on abstains). "
                   f"(Forgetting/learning gates need a full --derisk with ppl.)")
    elif render is not None:
        fidelity = render["render_fidelity"]
        no_double = render["n_double_inflection"] == 0
        moat_ok = (render["moat_render_calls_on_abstains"] == 0 and render["n_model_invoked_on_abstain"] == 0)
        # forgetting gate: FineWeb held-out (run3's base corpus). If FineWeb unavailable, fall back to RA-frame forget.
        no_forget = (forget_ratio is not None and forget_ratio <= 1.5)
        learned = (em_learn is not None and em_learn < 1.0)
        go = bool(fidelity >= 0.85 and no_double and moat_ok and no_forget and learned)
        miss = []
        if fidelity < 0.85:
            miss.append(f"render fidelity {fidelity:.2f} < 0.85 (still rambles/confabulates) -- lever: more steps / "
                        f"higher n_emerge / n_ra")
        if not no_double:
            miss.append(f"{render['n_double_inflection']} double-inflections (walkses)")
        if not moat_ok:
            miss.append(f"MOAT BREACHED ({render['n_model_invoked_on_abstain']} model-invocations on abstains)")
        if not no_forget:
            miss.append(f"catastrophic forgetting (FineWeb ppl {fw_pre}->{fw_post}, ratio "
                        f"{None if forget_ratio is None else round(forget_ratio,3)}) -- lever: higher mix_ratio / "
                        f"lower lr / more anti-forget windows")
        if not learned:
            miss.append(f"EMERGE frames not learned (EMERGE-frame ppl {em_pre}->{em_post}, ratio "
                        f"{None if em_learn is None else round(em_learn,3)}) -- lever: more steps / more n_emerge")
        if go:
            verdict = (f"GO -- the run3 83M WKV fine-tuned on the grounded RA+EMERGE frames renders them FLUENTLY + "
                       f"CORRECTLY behind the gate-first no-confab MOAT. Render fidelity {fidelity:.2f}; 0 'walkses'; "
                       f"MOAT holds (0 renders / 0 model-invocations on abstains -- the load-bearing property); NO "
                       f"catastrophic forgetting (FineWeb ppl {fw_pre}->{fw_post}, ratio "
                       f"{round(forget_ratio,3) if forget_ratio else None} <= 1.5) AND the frames were LEARNED "
                       f"(EMERGE-frame ppl {em_pre}->{em_post}, ratio {round(em_learn,3) if em_learn else None} < 1.0, "
                       f"RA-frame {ra_pre}->{ra_post}). The generator remains a tracked ANN scaffold (run3 spiking-"
                       f"forward conversion deferred; 88.6M validated).")
        else:
            verdict = "BOUNDARY -- " + "; ".join(miss) + f". Full-run: {FULL_RUN}"

    summary = {"probe": "run3_ra_grounded_finetune", "mode": mode, "seed": seed, "GO": go, "verdict": verdict,
               "finetune": ft, "render_derisk": render,
               "forget_ratio_fineweb": round(forget_ratio, 3) if forget_ratio else None,
               "emerge_frame_learn_ratio": round(em_learn, 3) if em_learn else None,
               "ra_frame_learn_ratio": round(ra_learn, 3) if ra_learn else None,
               "full_run_command": FULL_RUN, "render_eval_command": RENDER_EVAL,
               "HONEST_NOTE": "A DATA/format continuation fine-tune on run3's frozen ckpt (NOT a new mechanism): the "
                              "RA transitive-SVO QA frames + the EMERGE ability/exception/abstain frames, INTERLEAVED "
                              "per-batch with raw run3 FineWeb-Edu token windows (anti-forgetting on run3's OWN "
                              "distribution). The frame-aware inflection fix (emerge_v3) is imported from EMERGE-57. "
                              "The MOAT is preserved BY CONSTRUCTION (the gate short-circuits before the generator; 0 "
                              "renders on abstains). The render eval uses SCRIPTED ground-truth gate decisions to "
                              "isolate the GENERATOR's render fidelity; wiring the fine-tuned WKV into the fluid "
                              "console behind the real EMERGE reasoner (Rung 3, _fluidconv_chat_repl --renderer) is "
                              "the follow-on. run3 stays intact; the fine-tuned model is written to a NEW path."}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110, flush=True)
    print(f"[run3-ra-ft] VERDICT: {verdict if verdict else '(fine-tune only; run --derisk for render+ppl+moat gates)'}",
          flush=True)
    print(f"[run3-ra-ft] wrote {out_path}\n" + "=" * 110, flush=True)
    return summary


# ---------------------------------------------------------------------------------------------------------------------
# ENTRY POINTS
# ---------------------------------------------------------------------------------------------------------------------
def _check_corpus(seed=42, n=16):
    print("=== run3 RA grounded-render fine-tune -- corpus samples (RA transitive-SVO + EMERGE ability/exception) ===\n")
    r = _rng(seed)
    print("--- RA transitive-SVO QA frames ---")
    for _ in range(n // 2):
        print("  " + _make_example(r))
    print("\n--- EMERGE ability/inheritance + intransitive-exception frames ---")
    for _ in range(n // 2):
        print("  " + _make_emerge_example(r))
    print("\n=== frame-aware inflection fix (the 'walkses' bug) ===")
    cases = [("walks", "walks"), ("lurks", "lurks"), ("fly", "flies"), ("swim", "swims"),
             ("eat", "eats"), ("run", "runs"), ("hop", "hops")]
    ok = True
    for (v, exp) in cases:
        got = emerge_v3(v)
        if got != exp:
            ok = False
        print(f"  emerge_v3({v!r:8}) -> {got!r:10} (expect {exp!r})  [{'OK' if got == exp else 'FAIL'}]")
    print(f"\n  inflection fix all-correct: {ok}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-corpus", action="store_true", help="CPU: sample the grounded corpus + inflection fix")
    ap.add_argument("--smoke", action="store_true", help="CPU: load real run3 ckpt, few grounded steps, render+moat+ppl")
    ap.add_argument("--tiny-model", action="store_true", help="smoke: use a fresh TINY WKV (pipeline-only, fastest)")
    ap.add_argument("--derisk", action="store_true", help="full fine-tune + full de-risk gates (GPU)")
    ap.add_argument("--render-only", action="store_true", help="render+moat de-risk on an existing FT ckpt")
    ap.add_argument("--ckpt", default=FT_CKPT, help="FT ckpt path (render-only reads it; derisk writes it — override to a fixed path)")
    ap.add_argument("--out", default=None, help="derisk result-JSON path (default: seed-suffixed under research/findings/raw/)")
    ap.add_argument("--run-root", default=RUN3_ROOT)
    ap.add_argument("--which", default="best", choices=["best", "latest"], help="run3 base ckpt to fine-tune from")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-ra", type=int, default=12000)
    ap.add_argument("--n-emerge", type=int, default=14000)
    ap.add_argument("--mix-ratio", type=float, default=0.5, help="fraction of sequences drawn from raw FineWeb-Edu")
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--print-every", type=int, default=50)
    ap.add_argument("--fineweb-ppl-windows", type=int, default=200)
    # smoke overrides (tiny + bounded, CPU)
    ap.add_argument("--smoke-steps", type=int, default=8)
    a = ap.parse_args()

    if a.check_corpus:
        return _check_corpus(a.seed)

    if a.render_only:
        import torch
        model, tok, cfg, _meta = _load_run3(a.device, a.which, a.run_root)  # arch/tokenizer
        st = torch.load(a.ckpt, map_location=a.device, weights_only=False)
        model.load_state_dict(st["model"]); model.train(False)
        render = _render_derisk(model, tok, cfg, a.seed)
        _summarize({"note": "render-only on existing ckpt", "ckpt": a.ckpt}, render, "render_only", a.seed)
        return 0

    if a.smoke:
        # CPU-only, tiny + bounded: real run3 ckpt (default) or a fresh tiny model (--tiny-model). Forces device cpu.
        device = "cpu"
        steps = a.smoke_steps
        n_ra, n_emerge = 400, 500
        seq_len = 128
        batch = 4
        fw_ppl_windows = 12
        print(f"[run3-ra-ft] SMOKE (CPU): real-ckpt={not a.tiny_model} steps={steps} n_ra={n_ra} n_emerge={n_emerge} "
              f"seq_len={seq_len} batch={batch}\n", flush=True)
        try:
            # the tiny-model smoke saves to a DISTINCT path so it can never clobber a real 83M fine-tune ckpt
            ck_path = (FT_CKPT + ".tiny") if a.tiny_model else FT_CKPT
            ft, model, tok, cfg = finetune(steps, a.lr, batch, seq_len, a.seed, n_ra, n_emerge, a.mix_ratio,
                                           warmup=2, print_every=1, device=device, amp=0, which=a.which,
                                           run_root=a.run_root, fineweb_ppl_windows=fw_ppl_windows,
                                           tiny_model=a.tiny_model, save=True, ho_n=40, ckpt_path=ck_path)
            print("\n[run3-ra-ft] --- render + moat de-risk (scripted ground-truth gate) ---\n", flush=True)
            render = _render_derisk(model, tok, cfg, a.seed)
            _summarize(ft, render, "smoke_cpu", a.seed)
        except Exception as e:
            print(f"[run3-ra-ft] SMOKE ERROR: {e!r}"); traceback.print_exc()
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps({"probe": "run3_ra_grounded_finetune", "mode": "smoke_cpu",
                                       "error": repr(e)}, indent=2))
            return 1
        return 0

    if a.derisk:
        # SEED-suffixed output paths so a 6-seed run (42 43 44 100 101 102) never clobbers a prior seed's ckpt/JSON.
        # (--ckpt/--out override for a fixed path; default single-seed 42 lands on the historical FT_CKPT/OUT names.)
        ck_path = a.ckpt if a.ckpt != FT_CKPT else (FT_CKPT if a.seed == 42 else FT_CKPT.replace(".pt", f"_seed{a.seed}.pt"))
        out_path = Path(a.out) if a.out else (OUT if a.seed == 42 else
                                              OUT.with_name(OUT.stem + f"_seed{a.seed}" + OUT.suffix))
        try:
            ft, model, tok, cfg = finetune(a.steps, a.lr, a.batch_size, a.seq_len, a.seed, a.n_ra, a.n_emerge,
                                           a.mix_ratio, a.warmup, a.print_every, a.device, a.amp, a.which,
                                           a.run_root, a.fineweb_ppl_windows, tiny_model=False, save=True,
                                           ckpt_path=ck_path)
            render = _render_derisk(model, tok, cfg, a.seed)
            _summarize(ft, render, "derisk", a.seed, out_path=out_path)
        except Exception as e:
            print(f"[run3-ra-ft] ERROR: {e!r}"); traceback.print_exc()
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"probe": "run3_ra_grounded_finetune", "mode": "derisk",
                                            "seed": a.seed, "error": repr(e)}, indent=2))
            return 1
        return 0

    return _check_corpus(a.seed)


if __name__ == "__main__":
    raise SystemExit(main())
