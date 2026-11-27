# Capability: Language
# Record: docs/plans/2026-11-27-spiking-cortex-route-memo.md
# Expected-fail: partly. At 0.1B a ppl gap near 0.3 nats is the cited expectation (SpikingMamba-130M); in-brain memory is
#   weak at 0.1B even for the teacher (in-context 6/12). A fluency miss is a scale result; a sleep or reinstatement miss
#   is a verdict on THIS method (L1 counts, clipped STE, KL + hidden distillation, this data mix), not on spiking cortices.
"""E1: distil RWKV-7 0.1B (rwkv7-g1d-0.1b) into a spiking twin (level L1, brain/spiking_twin.py) and test it in the brain.

Arms (prereg brain/research/2026-11-27-spiking-twin-e1-prereg.md):
  a   twin from the teacher's weights, data = fineweb_edu + teacher-written chat
  a2  as a, plus a share of self-generated prefixes (SpikeOPD-style, abstract-level: the twin writes the continuation,
      the teacher's token KL is the target on it)
  b   same twin, RWKV-LM x070-style random init (SpikingTwin.random_init), same tokens (route-B control)
  c   the teacher (spikes off, teacher weights): reference, no training
Read-outs: held-out NLL on WikiText-103 validation and on teacher-written chat (ratio to c), the spike-off lesion of the
same twin, spikes per channel and silent share, sleep_probe recall (no_sleep / sleep_true / sleep_shuffled) with the twin
as the cortex, the scale-ladder memory rows (+ a wrong-memory sham), and a side-by-side transcript (read whole, no bar).

Every subcommand that loads a model refuses to run unless POOL_NODE=1 or BRAIN_ALLOW_W3090=1. Every unit (a training
checkpoint, one eval unit) is written atomically (tmp + os.replace) and skipped / resumed when present.

  BRAIN_ALLOW_W3090=1 python -m brain.experiments.spiking_twin_e1 prep  --out-dir D [--device cuda]
  BRAIN_ALLOW_W3090=1 python -m brain.experiments.spiking_twin_e1 train --arm a --seed 7 --out-dir D [--max-steps N]
  BRAIN_ALLOW_W3090=1 python -m brain.experiments.spiking_twin_e1 eval  --arm a --seed 7 --out-dir D
  python -m brain.experiments.spiking_twin_e1 score --out-dir D [--seeds 42,43,44,100,101,102]
  python -m brain.experiments.spiking_twin_e1 estimate [--tok-per-s 25000]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import subprocess
import sys
import time

SEEDS = [42, 43, 44, 100, 101, 102]
DEV_SEED = 7
ARMS = ("a", "a2", "b", "c")
TRAINED = ("a", "a2", "b")
TEACHER = "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
FINEWEB = "/home/dant123/Projects/sim/data/corpus/fineweb_edu.txt"
WT103_VAL = ("/home/dant123/Projects/Lift-Gate-Glue/data/wikitext103/wikitext-103-raw-v1/"
             "validation-00000-of-00001.parquet")
# Registered constants (prereg section 3). Changing one after data is a new experiment.
# micro_b 16 x accum 1 (review 11-27 item 5): the per-token WKV loop costs about the same at B=4 and B=16, so one
# 16-sequence micro-batch per step is about 4x the throughput of 4 x 4 for the same tokens per step.
CFG = dict(tokens_per_arm=300_000_000, seq=512, micro_b=16, accum=1, lr_init=1e-4, lr_rand=6e-4, warmup=500,
           lr_floor=0.1, wd=0.01, clip=1.0, kl_w=1.0, hid_w=0.5, chat_share=0.10, onpolicy_share=0.25,
           prompt_len=64, regen_every=16, gen_temp=1.0, gen_top_p=0.7, max_count=4, calib_seqs=16,
           ckpt_every=500, fineweb_tokens=300_000_000, chat_train_tokens=10_000_000, chat_eval_tokens=100_000,
           wt103_eval_tokens=None, sleep_steps=240, sleep_lr=5e-4, sleep_kl=1.0)
USER_MARK = "User:"          # the chat-format marker only; every word after it is written by the teacher
SLEEP_ARMS = ("no_sleep", "sleep_true", "sleep_shuffled")
LADDER_GAINS = (1.5, 2.5, 4.0)
# Arm c units that do not depend on the seed (greedy, fixed fact banks, teacher weights): measured once and shared by
# every seed; c's sleep unit is seeded (episodes, adapter init) and runs per seed.
C_SHARED_UNITS = ("ppl", "ladder", "transcript")
REFILL_S_E = 4.0             # [E] seconds per a2 refill (64 prompts x 448 sampled tokens); the smoke measures it


# ---------------------------------------------------------------------------------------------------- host + io
def host_guard(env=None):
    env = os.environ if env is None else env
    if env.get("POOL_NODE") != "1" and env.get("BRAIN_ALLOW_W3090") != "1":
        raise SystemExit("spiking_twin_e1: refuses to load a model unless POOL_NODE=1 or BRAIN_ALLOW_W3090=1")


def provenance(device="?", precision="?") -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "brain/"], capture_output=True, text=True,
                                    cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip())
    except Exception:
        commit, dirty = None, None
    p = {"host": socket.gethostname(), "commit": commit, "brain_dirty": dirty, "device": device,
         "precision": precision, "python": sys.version.split()[0]}
    try:
        import torch
        p["torch"] = torch.__version__
        if device == "cuda" and torch.cuda.is_available():
            p["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    if os.environ.get("HOST"):
        p["host_env"] = os.environ["HOST"].strip()
    return p


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def torch_save(obj, path):
    import torch
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def unit_path(out_dir, arm, seed, unit):
    if arm == "c" and unit in C_SHARED_UNITS:
        return os.path.join(out_dir, f"eval_c_shared_{unit}.json")
    return os.path.join(out_dir, f"eval_{arm}_s{seed}_{unit}.json")


def twin_path(out_dir, arm, seed):
    return os.path.join(out_dir, f"twin_{arm}_s{seed}.pt")


# ---------------------------------------------------------------------------------------------------- schedule
def lr_at(step, total, peak, warmup=CFG["warmup"], floor=CFG["lr_floor"]):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def total_steps(cfg=CFG):
    return math.ceil(cfg["tokens_per_arm"] / (cfg["seq"] * cfg["micro_b"] * cfg["accum"]))


# ---------------------------------------------------------------------------------------------------- data
class Stream:
    """Deterministic per-seed draw of training sequences from the token pools. Arms a, a2 and b with one seed draw the
    SAME sequences in the same order; a2 then replaces slot 0 of each micro-batch by a self-generated continuation of a
    prompt drawn from a SEPARATE prompt stream (seed + 50000), so its main stream stays aligned with a and b."""

    def __init__(self, fineweb, chat, seed, cfg=CFG):
        self.fw, self.chat, self.cfg = fineweb, chat, cfg
        self.rng = random.Random(seed * 1_000_003 + 17)

    def draw(self):
        L = self.cfg["seq"]
        if self.chat is not None and len(self.chat) >= L and self.rng.random() < self.cfg["chat_share"]:
            # the chat pool is rows of exactly seq tokens, each starting at the marker: draw whole rows, so a training
            # window never straddles two dialogues (the eval windows are aligned the same way)
            o = L * self.rng.randrange(0, len(self.chat) // L)
            return [int(t) for t in self.chat[o:o + L]]
        o = self.rng.randrange(0, len(self.fw) - L - 1)
        return [int(t) for t in self.fw[o:o + L]]

    def micro(self):
        return [self.draw() for _ in range(self.cfg["micro_b"])]


def tokenize_fineweb(pipe, path, n_tokens, out, chunk_chars=1 << 20):
    """First n_tokens of fineweb_edu (newline-aligned chunks) -> uint16 .npy. Resumes from out + '.part'."""
    import numpy as np
    if os.path.exists(out):
        return
    part = out + ".part.json"
    st = read_json(part) if os.path.exists(part) else {"offset": 0, "shards": []}
    with open(path, "rb") as fh:
        fh.seek(st["offset"])
        have = sum(s[1] for s in st["shards"])
        while have < n_tokens:
            buf = fh.read(chunk_chars)
            if not buf:
                break
            nl = buf.rfind(b"\n")
            if nl > 0 and len(buf) == chunk_chars:
                fh.seek(fh.tell() - (len(buf) - nl - 1))
                buf = buf[:nl + 1]
            ids = np.asarray(pipe.encode(buf.decode("utf-8", "ignore")), dtype=np.uint16)
            sp = f"{out}.shard{len(st['shards']):05d}.npy"
            np.save(sp + ".tmp.npy", ids)
            os.replace(sp + ".tmp.npy", sp)
            st["shards"].append([sp, int(ids.size)])
            st["offset"] = fh.tell()
            have += int(ids.size)
            write_json(part, st)
    allids = np.concatenate([np.load(s) for s, _ in st["shards"]])[:n_tokens]
    np.save(out + ".tmp.npy", allids)
    os.replace(out + ".tmp.npy", out)
    for s, _ in st["shards"]:
        os.remove(s)
    os.remove(part)


def tokenize_wt103(pipe, path, out):
    import numpy as np
    import pyarrow.parquet as pq
    if os.path.exists(out):
        return
    text = "".join(pq.read_table(path).column("text").to_pylist())
    ids = np.asarray(pipe.encode(text), dtype=np.uint16)
    np.save(out + ".tmp.npy", ids)
    os.replace(out + ".tmp.npy", out)


def sample_batch(twin, prompts, n_new, temp, top_p, gen):
    """Batched sampling from a twin (no grad): prompts (B, P) list -> (B, P + n_new) token lists."""
    import torch
    with torch.no_grad():
        logits, st = twin.forward_batch(torch.tensor(prompts))
        lg = logits[:, -1].float()
        out = [list(p) for p in prompts]
        for _ in range(n_new):
            p = torch.softmax(lg / max(temp, 1e-6), -1)
            sp, si = torch.sort(p, -1, descending=True)
            keep = (sp.cumsum(-1) - sp) < top_p
            sp = sp * keep
            pick = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=gen)
            tok = si.gather(-1, pick)[:, 0]
            for b, t in enumerate(tok.tolist()):
                out[b].append(t)
            logits, st = twin.forward_batch(tok.view(-1, 1), st)
            lg = logits[:, -1].float()
    return out


def chat_pool(twin, pipe, n_tokens, seed, out, seq, batch=64, temp=1.0, top_p=0.7):
    """Teacher-written chat: batches of sequences that start at the bare 'User:' marker; the teacher writes everything
    after it (user turns and assistant turns). Saved as one uint16 array; resumes per batch."""
    import numpy as np
    import torch
    if os.path.exists(out):
        return
    part = out + ".part"
    os.makedirs(part, exist_ok=True)
    mark = pipe.encode(USER_MARK)
    nb = math.ceil(n_tokens / (batch * seq))
    for j in range(nb):
        sp = os.path.join(part, f"b{j:05d}.npy")
        if os.path.exists(sp):
            continue
        g = torch.Generator(device=twin.z["emb.weight"].device).manual_seed(seed * 100_000 + j)
        seqs = sample_batch(twin, [mark] * batch, seq - len(mark), temp, top_p, g)
        np.save(sp + ".tmp.npy", np.asarray(seqs, dtype=np.uint16).reshape(-1))
        os.replace(sp + ".tmp.npy", sp)
    allids = np.concatenate([np.load(os.path.join(part, f"b{j:05d}.npy")) for j in range(nb)])[:n_tokens]
    np.save(out + ".tmp.npy", allids)
    os.replace(out + ".tmp.npy", out)


# ---------------------------------------------------------------------------------------------------- models
def load_teacher(device, model=None):
    from brain.cortex import Cortex, MODEL_DIR
    path = model or os.path.join(MODEL_DIR, TEACHER)
    return Cortex(model_path=path, device=device, cuda_precision="fp32")


def make_twin(cortex, *, spikes=True, trainable=False, ckpt=False, blob=None):
    from brain.spiking_twin import SpikingTwin
    tw = SpikingTwin(cortex, spikes=spikes, trainable=trainable, ckpt=ckpt, max_count=CFG["max_count"])
    if blob is not None:
        tw.load_weights(blob)
    return tw


def onpolicy_k(cfg=CFG):
    """Self-generated sequences per micro-batch in arm a2."""
    return max(1, round(cfg["onpolicy_share"] * cfg["micro_b"]))


def refill_size(cfg=CFG):
    """Prompts per a2 refill: enough self-generated sequences for regen_every optimizer steps."""
    return cfg["regen_every"] * cfg["accum"] * onpolicy_k(cfg)


def n_refills(cfg=CFG):
    return math.ceil(total_steps(cfg) / cfg["regen_every"])


def distill_loss(student, teacher, toks, cfg=CFG):
    """Token KL(teacher || student) per position + hid_w x mean over layers of the relative residual MSE."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        t_logits, _, t_h = teacher.forward_batch(toks, hiddens=True)
        t_lp = F.log_softmax(t_logits.float(), -1)
    s_logits, _, s_h = student.forward_batch(toks, hiddens=True)
    s_lp = F.log_softmax(s_logits.float(), -1)
    kl = (t_lp.exp() * (t_lp - s_lp)).sum(-1).mean()
    hid = sum(((sh.float() - th.float()) ** 2).mean() / (th.float() ** 2).mean().clamp_min(1e-8)
              for sh, th in zip(s_h, t_h)) / len(s_h)
    return cfg["kl_w"] * kl + cfg["hid_w"] * hid, float(kl), float(hid)


def train(arm, seed, out_dir, device="cuda", precision="bf16", max_steps=None, cfg=CFG, cortex=None, pools=None,
          log=print):
    """Train one arm; resumes from ckpt_<arm>_s<seed>.pt. Returns the train record (also written as JSON)."""
    import numpy as np
    import torch
    assert arm in TRAINED, arm
    final, rec_path = twin_path(out_dir, arm, seed), os.path.join(out_dir, f"train_{arm}_s{seed}.json")
    if os.path.exists(final) and os.path.exists(rec_path) and max_steps is None:
        return read_json(rec_path)
    cortex = cortex or load_teacher(device)
    if pools is None:
        pools = (np.load(os.path.join(out_dir, "fineweb.npy"), mmap_mode="r"),
                 np.load(os.path.join(out_dir, "chat_train.npy"), mmap_mode="r"))
    torch.manual_seed(seed)
    teacher = make_twin(cortex, spikes=False)
    student = make_twin(cortex, spikes=True, trainable=True, ckpt=(device == "cuda"))
    # Distillation trains the weights and thetas only. The slow-store adapters stay at zero and out of the forward
    # (enabled=False): they are sleep's, and a trained twin is saved without them (weights_state).
    student.enabled = teacher.enabled = False
    for d in (student.A_k, student.B_k, student.A_v, student.B_v):
        for prm in d.values():
            prm.requires_grad_(False)
    stream = Stream(pools[0], pools[1], seed, cfg)
    prompt_stream = Stream(pools[0], pools[1], seed + 50_000, cfg)
    if arm == "b":
        student.random_init(seed)
    calib_rng = random.Random(seed + 991)
    L = cfg["seq"]
    calib = [[[int(t) for t in pools[0][o:o + L]]] for o in
             (calib_rng.randrange(0, len(pools[0]) - L - 1) for _ in range(cfg["calib_seqs"]))]
    student.calibrate([torch.tensor(c) for c in calib])
    params = [p for p in student.parameters() if p.requires_grad]
    peak = cfg["lr_rand"] if arm == "b" else cfg["lr_init"]
    opt = torch.optim.AdamW(params, lr=peak, weight_decay=cfg["wd"], betas=(0.9, 0.99))
    n_total = total_steps(cfg)
    step, losses, t_train = 0, [], 0.0
    ck = os.path.join(out_dir, f"ckpt_{arm}_s{seed}.pt")
    if os.path.exists(ck):
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        student.load_weights(blob["twin"])
        opt.load_state_dict(blob["opt"])
        step, losses, t_train = blob["step"], blob["losses"], blob["t_train"]
        stream.rng.setstate(blob["stream_rng"])
        prompt_stream.rng.setstate(blob["prompt_rng"])
        torch.set_rng_state(blob["torch_rng"])
        log(f"resumed {arm} s{seed} at step {step}")
    gen = torch.Generator(device=student.z["emb.weight"].device).manual_seed(seed * 7 + 3)
    buf = []
    if os.path.exists(ck) and "gen_state" in blob:    # a2: the sampler and the unconsumed self-generated buffer
        gen.set_state(blob["gen_state"])
        buf = [list(x) for x in blob["buf"]]
    amp = (device == "cuda" and precision == "bf16")
    stop = n_total if max_steps is None else min(n_total, step + max_steps)
    tok_done = 0
    t0 = time.time()
    while step < stop:
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step, n_total, peak, cfg["warmup"], cfg["lr_floor"])
        opt.zero_grad(set_to_none=True)
        acc = [0.0, 0.0, 0.0]
        for _ in range(cfg["accum"]):
            mb = stream.micro()
            if arm == "a2":
                if not buf:
                    student.eval()
                    prompts = [s[:cfg["prompt_len"]] for s in (prompt_stream.draw() for _ in range(refill_size(cfg)))]
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        buf = sample_batch(student, prompts, L - cfg["prompt_len"], cfg["gen_temp"], cfg["gen_top_p"], gen)
                for j in range(onpolicy_k(cfg)):
                    mb[j] = buf.pop()
            toks = torch.tensor(mb, device=student.z["emb.weight"].device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss, kl, hid = distill_loss(student, teacher, toks, cfg)
            (loss / cfg["accum"]).backward()
            acc[0] += float(loss) / cfg["accum"]; acc[1] += kl / cfg["accum"]; acc[2] += hid / cfg["accum"]
            tok_done += toks.numel()
        torch.nn.utils.clip_grad_norm_(params, cfg["clip"])
        opt.step()
        with torch.no_grad():
            for th in student.theta.values():
                th.clamp_(min=1e-4)
        step += 1
        losses.append([step] + [round(x, 5) for x in acc])
        if step % 50 == 0:
            log(f"{arm} s{seed} step {step}/{n_total} loss {acc[0]:.4f} kl {acc[1]:.4f} hid {acc[2]:.4f} "
                f"tok/s {tok_done / max(time.time() - t0, 1e-9):.0f}")
        if step % cfg["ckpt_every"] == 0 or step == stop:
            t_now = t_train + time.time() - t0
            torch_save({"twin": student.weights_state(), "opt": opt.state_dict(), "step": step, "losses": losses,
                        "t_train": t_now, "stream_rng": stream.rng.getstate(),
                        "prompt_rng": prompt_stream.rng.getstate(), "torch_rng": torch.get_rng_state(),
                        "gen_state": gen.get_state(), "buf": [list(x) for x in buf]}, ck)
    elapsed = time.time() - t0
    refill_s = None
    if max_steps is not None:
        # smoke: time one a2-shaped refill (refill_size prompts, seq - prompt_len sampled tokens) on this student.
        # It is a fixed per-refill cost, not a throughput: estimate() adds it as a constant (review 11-27 item 6).
        # Separate prompt stream and generator, so the training state above is untouched.
        tstream = Stream(pools[0], None, seed + 70_000, cfg)
        prompts = [tstream.draw()[:cfg["prompt_len"]] for _ in range(refill_size(cfg))]
        tg = torch.Generator(device=student.z["emb.weight"].device).manual_seed(0)
        student.eval()
        if device == "cuda":
            torch.cuda.synchronize()
        tr = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            sample_batch(student, prompts, L - cfg["prompt_len"], cfg["gen_temp"], cfg["gen_top_p"], tg)
        if device == "cuda":
            torch.cuda.synchronize()
        refill_s = time.time() - tr
    rec = {"arm": arm, "seed": seed, "steps": step, "steps_total": n_total, "cfg": cfg,
           "tok_per_s": tok_done / max(elapsed, 1e-9), "t_train_s": t_train + elapsed, "refill_s": refill_s,
           "final_loss": losses[-1] if losses else None, "loss_tail": losses[-20:],
           "init": "rwkv" if arm == "b" else "teacher", "init_fallback": getattr(student, "init_fallback", None),
           "provenance": provenance(device, precision)}
    if step >= n_total:
        torch_save(student.weights_state(), final)
        write_json(rec_path, rec)
        if os.path.exists(ck):
            os.remove(ck)     # the final twin + record supersede the resume checkpoint (~1.2 GB each with AdamW)
    else:
        write_json(os.path.join(out_dir, f"train_{arm}_s{seed}.partial.json"), rec)
    return rec


# ---------------------------------------------------------------------------------------------------- eval units
def eval_nll(twin, ids, seq, batch=8, stats=False):
    """Mean next-token NLL (nats) over non-overlapping seq-token windows, state reset per window."""
    import torch
    import torch.nn.functional as F
    n = (len(ids) - 1) // seq
    tot, cnt = 0.0, 0
    twin.stats = {} if stats else None
    with torch.no_grad():
        for j in range(0, n, batch):
            rows = [[int(t) for t in ids[w * seq:(w + 1) * seq + 1]] for w in range(j, min(n, j + batch))]
            x = torch.tensor([r[:-1] for r in rows], device=twin.z["emb.weight"].device)
            y = torch.tensor([r[1:] for r in rows], device=x.device)
            lg, _ = twin.forward_batch(x)
            tot += float(F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="sum"))
            cnt += y.numel()
    summ = twin.spike_summary() if stats else None
    twin.stats = None
    return tot / max(cnt, 1), cnt, summ


def unigram_nll(ids, seq):
    """In-sample unigram entropy (nats/token) of the scored targets: the best any unigram model can do on them. A
    trained arm at or above it has learned nothing beyond token frequencies."""
    import numpy as np
    n = (len(ids) - 1) // seq
    y = np.asarray(ids[1:n * seq + 1], dtype=np.int64)
    p = np.bincount(y).astype(np.float64)
    p = p[p > 0] / y.size
    return float(-(p * np.log(p)).sum())


def unit_ppl(twin, arm, wt, chat, seq):
    r = {"wt103_unigram_nll": unigram_nll(wt, seq)}
    r["wt103_nll"], r["wt103_tokens"], summ = eval_nll(twin, wt, seq, stats=(arm != "c"))
    r["chat_nll"], r["chat_tokens"], _ = eval_nll(twin, chat, seq)
    if arm != "c":
        r.update({"mean_count": summ["mean_count"], "silent_share": summ["silent_share"], "sites": summ["sites"]})
        twin.spikes = False   # lesion: the same trained weights with graded inputs
        r["wt103_nll_spikes_off"] = eval_nll(twin, wt, seq)[0]
        r["chat_nll_spikes_off"] = eval_nll(twin, chat, seq)[0]
        twin.spikes = True
    return r


def unit_sleep(build_twin, seed, cfg=CFG):
    """sleep_probe on the twin: each arm starts from fresh zero adapters (seeded as sleep_probe does)."""
    import torch
    from brain.brain import BIRTH
    from brain.experiments.facts import FACTS
    from brain.experiments.sleep_probe import GENERIC, build_arm_episodes, evaluate
    from brain.sleep import sleep
    eps = build_arm_episodes(seed)
    keys_all = [k for _, _, k in FACTS]
    out = {}
    for arm in SLEEP_ARMS:
        torch.manual_seed(seed)
        pc = build_twin()
        with torch.no_grad():
            base = [s.float() for s in pc(pc.cortex.pipe.encode(BIRTH), None)[1]]
        losses = []
        if eps[arm] is not None:
            losses = sleep(pc, base, eps[arm], GENERIC, steps=cfg["sleep_steps"], lr=cfg["sleep_lr"],
                           kl_weight=cfg["sleep_kl"], seed=seed)
        r = evaluate(pc, base, keys_all)
        r["final_replay_loss"] = losses[-1][0] if losses else None
        r["adapter_norm"] = pc.adapter_norm()
        out[arm] = r
    return out


def unit_ladder(c):
    """The 11-02 scale-ladder rows (brain/experiments/scale_ladder.py, same logic) on any Cortex-API object, plus a
    wrong-memory sham at gain 4.0 (question j gets fact j+1's delta)."""
    import torch
    from brain.brain import BIRTH
    from brain.experiments.facts import CHAT, FACTS, PARA
    _, base = c.read(BIRTH, None)
    _, conv = c.read(CHAT, base)

    def ask(st, q):
        lg, s = c.read(f"User: {q}\n\nAssistant:", st)
        return c.speak(lg, s, max_tokens=20, temperature=0)[0]
    r = {}
    hit = 0
    for say, q, k in FACTS:
        _, s = c.read(f"User: {say}\n\nAssistant: Okay!\n\n" + CHAT, base)
        hit += k in ask(s, q).lower()
    r["in_context"] = hit
    D = [c.matrix_delta(c.read(f"User: {s}\n\n", base)[1], base) for s, _, _ in FACTS]
    for alpha in LADDER_GAINS:
        r[f"reinst_{alpha}"] = sum(k in ask(c.add_delta(conv, d, alpha), q).lower() for d, (_, q, k) in zip(D, FACTS))
    r["sham_4.0"] = sum(k in ask(c.add_delta(conv, D[(j + 1) % len(D)], 4.0), q).lower()
                        for j, (_, q, k) in enumerate(FACTS))
    keys = torch.stack([c.key_of(d) for d in D])
    mu = keys.mean(0)
    kn = keys - mu
    kn = kn / kn.norm(dim=1, keepdim=True)
    for name, qs in (("same_words", [q for _, q, _ in FACTS]), ("paraphrase", PARA)):
        ok = 0
        for j, q in enumerate(qs):
            v = c.key_of(c.matrix_delta(c.read(f"User: {q}\n\n", base)[1], base)) - mu
            ok += int(torch.argmax(kn @ (v / v.norm()))) == j
        r[name] = ok
    return r


def unit_transcript(c, max_tokens=60):
    """The weekly test's turns (brain/eval/weekly.py SESSION_1 + SESSION_2) as one plain conversation, no hippocampus,
    greedy. Read whole beside the teacher; no bar."""
    from brain.brain import BIRTH
    from brain.eval.weekly import SESSION_1, SESSION_2
    _, st = c.read(BIRTH, None)
    rows = []
    for turn in SESSION_1 + SESSION_2:
        lg, st = c.read(f"User: {turn}\n\nAssistant:", st)
        reply, st = c.speak(lg, st, max_tokens=max_tokens, temperature=0)
        _, st = c.read(f" {reply}\n\n", st) if reply else (None, st)
        rows.append([turn, reply])
    return rows


def evaluate_arm(arm, seed, out_dir, device="cuda", units=("ppl", "sleep", "ladder", "transcript"), cortex=None,
                 pools=None, cfg=CFG, log=print):
    import numpy as np
    import torch
    from brain.spiking_twin import TwinCortex
    todo = [u for u in units if not os.path.exists(unit_path(out_dir, arm, seed, u))]
    if not todo:
        return
    blob = None
    if arm != "c":
        tp = twin_path(out_dir, arm, seed)
        if not os.path.exists(tp):
            raise SystemExit(f"no trained twin at {tp}")
        blob = torch.load(tp, map_location="cpu", weights_only=False)
    cortex = cortex or load_teacher(device)
    spikes = arm != "c"
    build = lambda: make_twin(cortex, spikes=spikes, blob=blob)
    prov = provenance(device, "fp32")
    for u in todo:
        t0 = time.time()
        if u == "ppl":
            wt, chat = pools if pools is not None else (np.load(os.path.join(out_dir, "wt103_val.npy")),
                                                        np.load(os.path.join(out_dir, "chat_eval.npy")))
            res = unit_ppl(build(), arm, wt, chat, cfg["seq"])
        elif u == "sleep":
            res = unit_sleep(build, seed, cfg)
        elif u == "ladder":
            res = unit_ladder(TwinCortex(build()))
        elif u == "transcript":
            res = unit_transcript(TwinCortex(build()))
        else:
            raise ValueError(u)
        write_json(unit_path(out_dir, arm, seed, u), {"arm": arm, "seed": seed, "unit": u, "result": res,
                                                      "elapsed_s": time.time() - t0, "provenance": prov})
        log(f"eval {arm} s{seed} {u} done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------------------------------- verdict
BARS = dict(ppl_ratio=1.5, init_gap_nats=1.0, sleep_frac=0.5, sleep_margin=1.0, twin_shuffled_max=1.5,
            false_recall_over_teacher=0.5, b_learned_nats=1.0, reinst=8, twin_sham_max=3.0, a2_better_nats=0.05)
CONTROLS = dict(teacher_in_context=(5, 7), teacher_reinst=(10, 12), teacher_same_words=11, teacher_sleep_min=2.0,
                shuffled_max=1.5, no_sleep_max=1.0, sham_max=3.0, lesion_min_nats=0.01, silent_max=0.99)
NEED = {"ppl": ["wt103_nll", "chat_nll"], "ladder": ["in_context", "reinst_4.0", "sham_4.0", "same_words"],
        "sleep": ["recall", "false_recall"]}


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def load_records(out_dir, seeds):
    recs = {}
    for arm in ARMS:
        for s in seeds:
            for u in ("ppl", "sleep", "ladder"):
                p = unit_path(out_dir, arm, s, u)
                if os.path.exists(p):
                    recs.setdefault(arm, {}).setdefault(s, {})[u] = read_json(p)["result"]
    return recs


def _mean(xs):
    return sum(xs) / len(xs)


def _arm_finite(arm, r):
    keys = NEED["ppl"] + (["wt103_nll_spikes_off", "mean_count", "silent_share"] if arm != "c" else ["wt103_unigram_nll"])
    return (all(_finite(r["ppl"].get(k)) for k in keys) and all(_finite(r["ladder"].get(k)) for k in NEED["ladder"])
            and all(_finite(r["sleep"].get(a, {}).get(k)) for a in SLEEP_ARMS for k in NEED["sleep"]))


def verdict(recs, seeds) -> dict:
    """The registered rule (prereg section 5). recs[arm][seed] = {"ppl": {...}, "sleep": {arm: {"recall": n,
    "false_recall": n}}, "ladder": {...}}. VOID if any control fails; else GO(a | a2), FLUENT_ONLY or NEGATIVE.
    Arm b (the route-B contrast) never voids a and a2 once present: if it is non-finite anywhere, or its WT103 NLL is
    not at least b_learned_nats below the unigram entropy, F2 is unassessable (f2_assessable False, F2 None) and b is
    left out of C3 and C5."""
    void = []
    # C6 completeness (every arm) and finiteness (a, a2, c; b's finiteness decides f2_assessable instead)
    b_bad = []
    for arm in ARMS:
        for s in seeds:
            r = recs.get(arm, {}).get(s)
            if r is None or any(u not in r for u in ("ppl", "sleep", "ladder")):
                void.append(f"C6 missing {arm} s{s}")
                continue
            if not _arm_finite(arm, r):
                if arm == "b":
                    b_bad.append(f"b non-finite s{s}")
                else:
                    void.append(f"C6 non-finite {arm} s{s}")
    if void:
        return {"verdict": "VOID", "reasons": void, "seeds": seeds, "claim": sorted(seeds) == sorted(SEEDS)}
    m = lambda arm, f: _mean([f(recs[arm][s]) for s in seeds])
    sl = lambda a, k="recall": (lambda r: r["sleep"][a][k])
    t = {"in_context": m("c", lambda r: r["ladder"]["in_context"]), "reinst": m("c", lambda r: r["ladder"]["reinst_4.0"]),
         "same_words": m("c", lambda r: r["ladder"]["same_words"]), "sleep_true": m("c", sl("sleep_true")),
         "sleep_shuffled": m("c", sl("sleep_shuffled")), "sham": m("c", lambda r: r["ladder"]["sham_4.0"]),
         "false_recall": m("c", sl("sleep_true", "false_recall")),
         "unigram_nll": m("c", lambda r: r["ppl"]["wt103_unigram_nll"])}
    if not b_bad:
        b_nll = m("b", lambda r: r["ppl"]["wt103_nll"])
        if b_nll > t["unigram_nll"] - BARS["b_learned_nats"]:
            b_bad.append(f"b not trained: WT103 NLL {b_nll:.3f} not {BARS['b_learned_nats']} nat below unigram "
                         f"{t['unigram_nll']:.3f}")
    b_ok = not b_bad
    lo, hi = CONTROLS["teacher_in_context"]
    if not lo <= t["in_context"] <= hi:
        void.append(f"C1 teacher in-context {t['in_context']:.2f} outside [{lo},{hi}]")
    lo, hi = CONTROLS["teacher_reinst"]
    if not lo <= t["reinst"] <= hi:
        void.append(f"C1 teacher reinstatement@4.0 {t['reinst']:.2f} outside [{lo},{hi}]")
    if t["same_words"] < CONTROLS["teacher_same_words"]:
        void.append(f"C1 teacher same-words {t['same_words']:.2f} < {CONTROLS['teacher_same_words']}")
    if t["sleep_true"] < CONTROLS["teacher_sleep_min"]:
        void.append(f"C2 teacher sleep recall {t['sleep_true']:.2f} < {CONTROLS['teacher_sleep_min']}")
    if t["sleep_shuffled"] > CONTROLS["shuffled_max"]:
        void.append(f"C2 teacher shuffled recall {t['sleep_shuffled']:.2f} > {CONTROLS['shuffled_max']}")
    for arm in ARMS:
        if arm == "b" and not b_ok:
            continue
        ns = m(arm, sl("no_sleep"))
        if ns > CONTROLS["no_sleep_max"]:
            void.append(f"C3 {arm} no-sleep recall {ns:.2f} > {CONTROLS['no_sleep_max']}")
    if t["sham"] > CONTROLS["sham_max"]:
        void.append(f"C4 teacher wrong-memory sham {t['sham']:.2f} > {CONTROLS['sham_max']}")
    for arm in TRAINED:
        if arm == "b" and not b_ok:
            continue
        les = m(arm, lambda r: abs(r["ppl"]["wt103_nll_spikes_off"] - r["ppl"]["wt103_nll"]))
        sil = m(arm, lambda r: r["ppl"]["silent_share"])
        cnt = m(arm, lambda r: r["ppl"]["mean_count"])
        if les < CONTROLS["lesion_min_nats"] or sil >= CONTROLS["silent_max"] or cnt <= 0:
            void.append(f"C5 {arm} spikes not engaged (lesion {les:.4f} nats, silent {sil:.3f}, count {cnt:.3f})")
    if void:
        return {"verdict": "VOID", "reasons": void, "teacher": t, "seeds": seeds,
                "claim": sorted(seeds) == sorted(SEEDS)}
    nll = lambda arm, k: m(arm, lambda r: r["ppl"][k])
    arms = {}
    for X in ("a", "a2"):
        gap_wt, gap_chat = nll(X, "wt103_nll") - nll("c", "wt103_nll"), nll(X, "chat_nll") - nll("c", "chat_nll")
        d = {"ratio_wt103": math.exp(gap_wt), "ratio_chat": math.exp(gap_chat), "gap_chat": gap_chat,
             "init_gap": (nll("b", "wt103_nll") - nll(X, "wt103_nll")) if b_ok else None,
             "sleep_true": m(X, sl("sleep_true")), "sleep_shuffled": m(X, sl("sleep_shuffled")),
             "false_recall": m(X, sl("sleep_true", "false_recall")),
             "reinst": m(X, lambda r: r["ladder"]["reinst_4.0"]), "sham": m(X, lambda r: r["ladder"]["sham_4.0"])}
        d["F1"] = d["ratio_wt103"] <= BARS["ppl_ratio"] and d["ratio_chat"] <= BARS["ppl_ratio"]
        d["F2"] = (d["init_gap"] >= BARS["init_gap_nats"]) if b_ok else None
        d["S"] = (d["sleep_true"] >= BARS["sleep_frac"] * t["sleep_true"]
                  and d["sleep_true"] - d["sleep_shuffled"] >= BARS["sleep_margin"]
                  and d["sleep_shuffled"] <= BARS["twin_shuffled_max"]
                  and d["false_recall"] <= t["false_recall"] + BARS["false_recall_over_teacher"])
        d["R"] = d["reinst"] >= BARS["reinst"] and d["sham"] <= BARS["twin_sham_max"]
        f2 = d["F2"] is not False             # unassessable F2 does not fail X; the verdict says so (f2_assessable)
        d["pass"] = d["F1"] and f2 and d["S"] and d["R"]
        arms[X] = d
    if arms["a"]["pass"]:
        carry = "a2" if arms["a2"]["pass"] and arms["a2"]["gap_chat"] <= arms["a"]["gap_chat"] - BARS["a2_better_nats"] else "a"
        v = "GO"
    elif arms["a2"]["pass"]:
        v, carry = "GO", "a2"
    elif any(arms[X]["F1"] and arms[X]["F2"] is not False for X in arms):
        v, carry = "FLUENT_ONLY", None
    else:
        v, carry = "NEGATIVE", None
    return {"verdict": v, "carry": carry, "arms": arms, "teacher": t, "seeds": seeds,
            "claim": sorted(seeds) == sorted(SEEDS),
            "f2_assessable": b_ok, "b_status": b_bad,   # False: no route-B (init) claim; b's recipe failed, not a or a2
            "stop_rule": v in ("FLUENT_ONLY", "NEGATIVE"),   # both methods (a, a2) failed: two failed methods
            "reasons": []}


# ---------------------------------------------------------------------------------------------------- estimate
def estimate(tok_per_s=25000.0, refill_s=REFILL_S_E, eval_h=0.5, usd_per_h=1.0, cfg=CFG):
    """GPU-hours. a2 adds n_refills x refill_s as a constant (sampling cost does not scale with training tok/s).
    Arm c: its sleep unit runs per seed (eval_h / 2 [E]); its ppl/ladder/transcript run once (eval_h / 2)."""
    train_h = cfg["tokens_per_arm"] / tok_per_s / 3600
    a2_extra_h = n_refills(cfg) * refill_s / 3600
    per = {"a": train_h + eval_h, "a2": train_h + a2_extra_h + eval_h, "b": train_h + eval_h, "c": eval_h / 2}
    seed_h = sum(per.values())
    total = 7 * seed_h + eval_h / 2
    return {"tok_per_s": tok_per_s, "refill_s": refill_s, "a2_extra_h": a2_extra_h, "per_arm_h": per,
            "per_seed_h": seed_h, "dev_h": seed_h + eval_h / 2, "six_h": 6 * seed_h, "total_h": total,
            "g5_usd": total * usd_per_h}


# ---------------------------------------------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fineweb", default=FINEWEB)
    p.add_argument("--wt103", default=WT103_VAL)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--arm", required=True, choices=TRAINED if name == "train" else ARMS)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--device", default="cuda")
        if name == "train":
            p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
            p.add_argument("--max-steps", type=int, default=None, help="smoke / throughput gate; never a final twin")
        else:
            p.add_argument("--units", default="ppl,sleep,ladder,transcript")
    p = sub.add_parser("score")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--out", default=None)
    p = sub.add_parser("estimate")
    p.add_argument("--tok-per-s", type=float, default=25000.0)
    p.add_argument("--refill-s", type=float, default=REFILL_S_E)
    a = ap.parse_args(argv)
    if a.cmd == "estimate":
        print(json.dumps(estimate(a.tok_per_s, a.refill_s), indent=1))
        return
    if a.cmd == "score":
        seeds = [int(x) for x in a.seeds.split(",")]
        v = verdict(load_records(a.out_dir, seeds), seeds)
        v["provenance"] = provenance()
        if a.out:
            write_json(a.out, v)
        print(json.dumps(v, indent=1))
        return
    host_guard()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.cmd == "prep":
        for path in (a.fineweb, a.wt103):
            if not os.path.exists(path):
                raise SystemExit(f"missing corpus {path} (E1 downloads nothing)")
        c = load_teacher(a.device)
        tokenize_wt103(c.pipe, a.wt103, os.path.join(a.out_dir, "wt103_val.npy"))
        tokenize_fineweb(c.pipe, a.fineweb, CFG["fineweb_tokens"], os.path.join(a.out_dir, "fineweb.npy"))
        teacher = make_twin(c, spikes=False)
        chat_pool(teacher, c.pipe, CFG["chat_eval_tokens"], 1, os.path.join(a.out_dir, "chat_eval.npy"), CFG["seq"])
        chat_pool(teacher, c.pipe, CFG["chat_train_tokens"], 0, os.path.join(a.out_dir, "chat_train.npy"), CFG["seq"])
        write_json(os.path.join(a.out_dir, "prep.json"), {"cfg": CFG, "provenance": provenance(a.device, "fp32")})
    elif a.cmd == "train":
        train(a.arm, a.seed, a.out_dir, a.device, a.precision, a.max_steps)
    elif a.cmd == "eval":
        evaluate_arm(a.arm, a.seed, a.out_dir, a.device, tuple(u for u in a.units.split(",") if u))


if __name__ == "__main__":
    main()
