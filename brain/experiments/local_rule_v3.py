# Capability: Sleep consolidation
# Record: brain/research/2026-11-27-local-rule-v3-DESIGN.md
# Expected-fail: the second-reading ("copy mode") target may store the sentence yet transfer to the question worse than
#   the gradient night; residual targets may drift many directions; rank 8 with a KP-trained A may not separate 12 facts;
#   single-form replay gives weak reworded recall at 0.1B (PREREQUISITES). A negative is a verdict on this two-phase
#   layerwise-target method at this adapter rank and scale, not on local learning.
"""Local rule v3: two-phase layerwise targets from the cortex's own in-context activity. No error travels backward.

Per replayed telling X (both phases forward-only, no_grad, the same cortex):
1. Clamped phase (teacher), adapters off: the cortex reads X twice in one stream (token ids of X concatenated with
   themselves, so positions align 1:1 by construction: no re-tokenisation at the seam); on the SECOND reading it
   records, per block i and position t, the residual after the block x*_{i+1} and the channel-mix hidden h*_i. Cached
   once per telling.
2. Free phase (student), adapters on: the cortex reads X once from the base state.
3. Update, per layer, at the synapse's own layer (no feedback path, no weight transport):
   - value path: e_i = x_{i+1} - x*_{i+1}; dB_v = (h A_v)^T e_i (the channel mix ADDS to the residual, so this is the
     exact delta rule for that layer's own squared error);
   - key path: e_h = (h - h*) * 2 relu(pre); dB_k = (k_in A_k)^T e_h;
   - A factors: v1's Kolen-Pollack copy of B, one hop inside the adapter's own rank units (dA = x^T (e Bcopy));
   - step: normalised LMS (divided by the pre-synaptic energy of that factor), so one lr serves every layer;
   - every error comes from the student's own forward, so a later layer corrects only what earlier layers left.
4. Keep: one keep text per step, the same rule, target = the pretrained cortex's own activity on that text (the error
   is the adapters' own effect), scaled by the keep weight.
Clip (v1's update-norm clip, divisive-normalisation stand-in) and B decay with the KP copies are v1's, unchanged.

Arms: v3, gradient (brain.sleep.sleep, reference), no_sleep, sham_ctx (the clamped phase reads ANOTHER fact's telling
before X, seeded derangement), no_target (lesion: no first reading, target = own activity, must stay inert),
top_only (depth lesion: targets at the last block only), grad_distill (report only: backprop on KL to the same
teacher's final logits). Gate G0 before any night: the teacher's top-1 at the answer position is the answer for >= 10
of 12 facts.

    POOL_NODE=1 python -m brain.experiments.local_rule_v3 run --model M.pth --seed 7 --arms g0,no_sleep,gradient \\
        --out-dir D --tag ref
    POOL_NODE=1 python -m brain.experiments.local_rule_v3 run --model M.pth --seed 7 --arms v3 --lr 0.1 --keep 1 \\
        --out-dir D --tag lr0.1_keep1
    python -m brain.experiments.local_rule_v3 pick --dir D --out D/pick.json   # G0 unit + 2 refs + 9 cells required
    POOL_NODE=1 python -m brain.experiments.local_rule_v3 run --model M.pth --seed 42 --pick D/pick.json --out-dir S \\
        --tag six
    python -m brain.experiments.local_rule_v3 score --dir S --tag six --pick D/pick.json
    python -m brain.experiments.local_rule_v3 combine --v01 V01.json --v04 V04.json      # GO-V3 / GO-V3-PARTIAL
Prereg: brain/research/2026-11-27-local-rule-v3-prereg.md (DRAFT pending pre-data review).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from brain.experiments.local_rule_v2 import (PROMPTS_SHA256, _finite, _load_dir, _mean, _model_id, _one, atomic_write,
                                             heldout_drift, prompts_sha, provenance, read_pick, split_prompts, unit_path)
from brain.experiments.local_sleep import LocalCortex
from brain.plastic_cortex import PlasticCortex

SEEDS = (42, 43, 44, 100, 101, 102)
DEV_SEED = 7
ARMS = ("no_sleep", "gradient", "v3", "sham_ctx", "no_target", "top_only", "grad_distill")
LOCAL_ARMS = ("v3", "sham_ctx", "no_target", "top_only")          # one (lr, keep) across these
TEACHER_OF = {"v3": "double", "sham_ctx": "sham", "no_target": "none", "top_only": "double"}
GATED_ARMS = ("no_sleep", "gradient", "v3", "sham_ctx", "no_target", "top_only")   # grad_distill is report only
GATED = ("recall", "false_recall", "generic_kl", "heldout_kl")
DEV_LRS = (0.03, 0.1, 0.3)
DEV_KEEPS = (0.3, 1.0, 3.0)
DRIFT_MULT = 1.25       # (c) generic and held-out KL each <= 1.25 x gradient's
KL_NIGHT_MAX = 1.0      # (c) no v3 night with generic or held-out KL above 1
RECALL_PLUS = 2         # (a) bar = max(0.5 x gradient, no_sleep + 2)
SHAM_PLUS = 1           # (b) sham_ctx mean <= no_sleep + 1
WIN_SEEDS = 5           # (b) v3 > sham_ctx on >= 5 of 6 seeds
FALSE_PLUS = 1          # (e) false recall <= gradient + 1
SHALLOW_MIN = 1.0       # GO-SHALLOW if v3 - top_only < 1
G0_MIN = 10             # G0: teacher top-1 == answer for >= 10 of 12 facts
GRAD_MIN = 2            # UNDEFINED if mean gradient recall < 2
INERT_NORM = 1e-6       # (d) no_target adapter norm "at zero"
ENGAGE_NORM_MIN = 0.5   # sham_ctx adapter norm >= 0.5 x that seed's v3 norm
LOSS_WIN = 20           # head / tail = mean of the first / last 20 steps
SHAM_SEED_OFFSET = 7919
NLMS_EPS = 1e-6


# ------------------------------------------------------------------ guard
def guard(env=None):
    env = os.environ if env is None else env
    if env.get("POOL_NODE") != "1" and env.get("BRAIN_ALLOW_W3090") != "1":
        raise SystemExit("local_rule_v3 run: refuses to run unless POOL_NODE=1 or BRAIN_ALLOW_W3090=1")


# ------------------------------------------------------------------ recording cortex
class V3Cortex(LocalCortex):
    """LocalCortex that, when `rec` is a list, records per block (i, k_in, pre, h, y) of the channel mix (adapters on
    or off as `enabled` says) and, in `rec_t`, the time-mix output; `activity` rebuilds the residual after every block
    from them with the same adds in the same order as PlasticCortex.forward (bit-identical; tested)."""
    rec_t = None

    def _tmix(self, i, x, x_prev, v_first, state):
        out = super()._tmix(i, x, x_prev, v_first, state)
        if self.rec_t is not None:
            self.rec_t.append(out[0])
        return out

    def _cmix(self, i, x, x_prev):
        if self.rec is None:
            return PlasticCortex._cmix(self, i, x, x_prev)
        assert not torch.is_grad_enabled(), "local rule v3 forward must run under no_grad"
        f, s = f"blocks.{i}.ffn.", str(i)
        xx = torch.cat((x_prev.unsqueeze(0), x[:-1, :])) - x
        k_in = x + xx * self.z[f + "x_k"]
        K, V = self.z[f + "key.weight"], self.z[f + "value.weight"]
        if self.enabled and s in self.A_k:
            pre = k_in @ K + (k_in @ self.A_k[s]) @ self.B_k[s]
            h = torch.relu(pre) ** 2
            y = h @ V + (h @ self.A_v[s]) @ self.B_v[s]
        else:
            pre = k_in @ K
            h = torch.relu(pre) ** 2
            y = h @ V
        self.rec.append((i, k_in, pre, h, y))
        return y, x[-1, :]


def adopt3(plastic) -> V3Cortex:
    """A V3Cortex sharing the adapters (Parameters) of an existing PlasticCortex; the original is untouched."""
    lc = V3Cortex.__new__(V3Cortex)
    lc.__dict__.update(plastic.__dict__)
    return lc


def activity(pc, ids, state, adapters=True):
    """One no_grad forward of `ids`. Returns (logits (T,V), {layer: (k_in, pre, h, res_after)}) with res_after the
    residual after the block. adapters False = the pretrained cortex (the clamped phase)."""
    assert getattr(pc, "source_tag", None) is None, "local rule v3 does not support a source tag"
    was = pc.enabled
    pc.rec, pc.rec_t, pc.enabled = [], [], bool(adapters) and was
    try:
        with torch.no_grad():
            logits, _ = pc(ids, state)
            x = pc.z["emb.weight"][ids]
            out = {}
            cm = {r[0]: r for r in pc.rec}
            for i in range(pc.L):
                x = x + pc.rec_t[i]
                _, k_in, pre, h, y = cm[i]
                x = x + y
                out[i] = (k_in, pre, h, x)
    finally:
        pc.rec, pc.rec_t, pc.enabled = None, None, was
    return logits, out


# ------------------------------------------------------------------ the clamped phase
def derangement(n, seed):
    """A seeded permutation of range(n) with no fixed point (n >= 2)."""
    if n < 2:
        raise ValueError("sham_ctx needs at least two tellings")
    rng = random.Random(seed + SHAM_SEED_OFFSET)
    p = list(range(n))
    while any(i == j for i, j in enumerate(p)):
        rng.shuffle(p)
    return p


def context_ids(ep_ids, j, kind, perm=None):
    """What the clamped phase reads before telling j: itself ("double"), another fact's telling ("sham"), nothing
    ("none", the lesion)."""
    if kind == "double":
        return list(ep_ids[j])
    if kind == "sham":
        return list(ep_ids[perm[j]])
    if kind == "none":
        return []
    raise ValueError(kind)


def teacher(pc, ctx, ids, state):
    """Clamped phase: adapters off, read ctx + ids; return the activity and logits on the ids positions only."""
    logits, act = activity(pc, list(ctx) + list(ids), state, adapters=False)
    n = len(ctx)
    return logits[n:], {i: (h[n:], res[n:]) for i, (_k, _p, h, res) in act.items()}


# ------------------------------------------------------------------ the rule
def nlms(a, e, eps=NLMS_EPS):
    """Normalised LMS step for a factor whose pre-synaptic activity is a (T, r) and post-synaptic error e (T, n):
    a^T e / (sum ||a_t||^2 + eps). At lr 1 a single row is fully corrected (tested)."""
    return a.T @ e / (float(a.pow(2).sum()) + eps)


class KPCopies:
    """v1's Kolen-Pollack copies of B (no head / mix / hidden feedback: no error travels between layers here)."""

    def __init__(self, pc, seed=0):
        g = torch.Generator().manual_seed(seed)
        s0 = str(pc.layers[0])
        r, hdim = pc.B_k[s0].shape
        dev = pc.z["emb.weight"].device
        self.v = {i: (torch.randn(pc.C, r, generator=g) / pc.C ** 0.5).to(dev) for i in pc.layers}   # ~ B_v^T
        self.k = {i: (torch.randn(hdim, r, generator=g) / hdim ** 0.5).to(dev) for i in pc.layers}     # ~ B_k^T


def _clip(d, clip):
    if clip:
        n = float(d.norm())
        d.mul_(min(1.0, clip / (n + 1e-12)))
    return d


def v3_update(pc, kp, act, tgt, *, lr, layers, decay=0.0, clip=1.0, weight=1.0):
    """One local step on `pc`'s adapters from the student's activity `act` and per-layer targets `tgt` {i: (h*, res*)}.
    Only `layers` learn. Returns {layer: (value error norm^2 / T, key error norm^2 / T)}. No autograd anywhere."""
    errs = {}
    with torch.no_grad():
        for i in layers:
            s = str(i)
            k_in, pre, h, res = act[i]
            ht, rt = tgt[i]
            T = res.shape[0]
            e = weight * (res - rt)                                   # value path: residual after the block
            e_h = weight * (h - ht) * 2 * torch.relu(pre)             # key path: hidden, own derivative
            errs[i] = (float(e.pow(2).sum()) / T, float(e_h.pow(2).sum()) / T)
            a_v, a_k = h @ pc.A_v[s], k_in @ pc.A_k[s]
            dBv, dBk = nlms(a_v, e), nlms(a_k, e_h)
            dAv, dAk = nlms(h, e @ kp.v[i]), nlms(k_in, e_h @ kp.k[i])
            for p, d in ((pc.B_v[s], dBv), (pc.B_k[s], dBk), (pc.A_v[s], dAv), (pc.A_k[s], dAk)):
                p.sub_(lr * _clip(d, clip))
            kp.v[i].sub_(lr * dBv.T)          # the copy gets B's own (clipped) update
            kp.k[i].sub_(lr * dBk.T)
            if decay:
                for p in (pc.B_v[s], pc.B_k[s], kp.v[i], kp.k[i]):
                    p.mul_(1 - decay)
    return errs


def v3_sleep(plastic, base_state, episodes, generic, *, steps=240, lr=0.1, keep=1.0, decay=1e-3, seed=0, clip=1.0,
             teacher_kind="double", top_only=False, log=None, info=None):
    """The v3 night. episodes: (text, strength) as brain.sleep.sleep; generic: keep texts. Same RNG draws as v1/v2
    (one episode by strength, then one keep text, per step). Returns per-step (replay error, keep error, keep KL);
    `info` gets the engagement record."""
    pc = adopt3(plastic)
    rng = random.Random(seed)
    kp = KPCopies(pc, seed)
    ids_of = pc.cortex.pipe.encode
    weights = [max(s, 1e-3) for _, s in episodes]
    ep_ids = [ids_of(t) for t, _ in episodes]
    gen_ids = [ids_of(g if isinstance(g, str) else g[0]) for g in generic]
    perm = derangement(len(ep_ids), seed) if teacher_kind == "sham" else None
    layers = [pc.layers[-1]] if top_only else list(pc.layers)
    tgts, ctx_len = [], []
    for j, ids in enumerate(ep_ids):                              # targets cached once per telling
        ctx = context_ids(ep_ids, j, teacher_kind, perm)
        ctx_len.append(len(ctx))
        tgts.append(teacher(pc, ctx, ids, base_state)[1])
    keep_t, keep_ref = [], []
    for g in gen_ids:                                            # keep target: the pretrained cortex, no context
        lg, t = teacher(pc, [], g, base_state)
        keep_t.append(t)
        keep_ref.append(F.log_softmax(lg, -1))
    pc.enabled = True
    losses, per_layer = [], []
    for step in range(steps):
        j = rng.choices(range(len(episodes)), weights=weights)[0]
        _, act = activity(pc, ep_ids[j], base_state)
        er = v3_update(pc, kp, act, tgts[j], lr=lr, layers=layers, decay=decay, clip=clip)
        g = rng.randrange(len(gen_ids))
        lg, gact = activity(pc, gen_ids[g], base_state)
        ek = v3_update(pc, kp, gact, keep_t[g], lr=lr, layers=layers, clip=clip, weight=keep)
        kl = float(F.kl_div(F.log_softmax(lg, -1), keep_ref[g], log_target=True, reduction="batchmean"))
        losses.append((sum(v for v, _ in er.values()), sum(v for v, _ in ek.values()), kl))
        per_layer.append({str(i): v for i, (v, _) in er.items()})
        if log and step % 20 == 0:
            log(f"v3 step {step}: replay err {losses[-1][0]:.4f} keep err {losses[-1][1]:.4f} keep kl {kl:.4f}")
    if info is not None:
        head, tail = losses[:LOSS_WIN], losses[-LOSS_WIN:]
        info.update(kind="local", teacher=teacher_kind, steps_run=len(losses), layers_on=layers,
                    targets_cached=len(tgts), target_ctx_tokens=_mean(ctx_len) if ctx_len else 0.0,
                    err_head=_mean(x[0] for x in head) if head else None,
                    err_tail=_mean(x[0] for x in tail) if tail else None,
                    keep_err_head=_mean(x[1] for x in head) if head else None,
                    keep_err_tail=_mean(x[1] for x in tail) if tail else None,
                    err_layer_first={k: _mean(d[k] for d in per_layer[:LOSS_WIN]) for k in (per_layer[0] if per_layer else {})},
                    err_layer_last={k: _mean(d[k] for d in per_layer[-LOSS_WIN:]) for k in (per_layer[0] if per_layer else {})},
                    sham_perm=perm)
    return losses


def distill_sleep(plastic, base_state, episodes, generic, *, steps=240, lr=5e-4, kl_weight=1.0, seed=0, log=None,
                  info=None):
    """grad_distill (report only): Adam on KL(teacher || student) over the telling's positions (the teacher = the
    clamped-phase double reading's logits) plus kl_weight x the keep KL to the pretrained cortex, as brain.sleep."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    pc = adopt3(plastic)
    ids_of = pc.cortex.pipe.encode
    weights = [max(s, 1e-3) for _, s in episodes]
    ep_ids = [ids_of(t) for t, _ in episodes]
    gen_ids = [ids_of(g if isinstance(g, str) else g[0]) for g in generic]
    tlog = [F.log_softmax(teacher(pc, ids, ids, base_state)[0], -1)[:-1] for ids in ep_ids]
    pc.enabled = False
    with torch.no_grad():
        ref = [F.log_softmax(pc(g, base_state)[0], -1) for g in gen_ids]
    pc.enabled = True
    opt = torch.optim.Adam([p for p in plastic.parameters() if p.requires_grad], lr=lr)
    losses = []
    for step in range(steps):
        j = rng.choices(range(len(episodes)), weights=weights)[0]
        lg, _ = pc(ep_ids[j], base_state)
        d = F.kl_div(F.log_softmax(lg[:-1], -1), tlog[j], log_target=True, reduction="batchmean")
        g = rng.randrange(len(gen_ids))
        gl, _ = pc(gen_ids[g], base_state)
        keep = F.kl_div(F.log_softmax(gl, -1), ref[g], log_target=True, reduction="batchmean")
        opt.zero_grad()
        (d + kl_weight * keep).backward()
        opt.step()
        losses.append((float(d.detach()), float(keep.detach())))
        if log and step % 20 == 0:
            log(f"distill step {step}: kl to teacher {float(d):.4f} keep {float(keep):.4f}")
    if info is not None:
        info.update(_loss_engagement("grad_distill", losses))
    return losses


def _loss_engagement(kind, losses):
    head, tail = losses[:LOSS_WIN], losses[-LOSS_WIN:]
    return {"kind": kind, "steps_run": len(losses), "loss_head": _mean(x[0] for x in head) if head else None,
            "loss_tail": _mean(x[0] for x in tail) if tail else None}


# ------------------------------------------------------------------ G0 (the teacher knows the answer in context)
def answer_index(pipe, ids, text, key):
    """Index of the first token of `ids` whose decoded span reaches the first character of `key` in `text`."""
    off = text.lower().find(key.lower())
    if off < 0:
        raise ValueError(f"key {key!r} not in {text!r}")
    for j in range(len(ids)):
        if len(pipe.decode(ids[: j + 1])) > off:
            return j
    raise ValueError(f"key {key!r} beyond the tokens of {text!r}")


def g0_check(pc, base_state, told):
    """told: [(telling text, answer key)]. For each: does the teacher (adapters off, second reading) put top-1 on the
    answer's first token? Also the single-reading rate (no context), reported. No night runs."""
    pc = adopt3(pc)
    pipe = pc.cortex.pipe
    items = []
    for text, key in told:
        ids = pipe.encode(text)
        a = answer_index(pipe, ids, text, key)
        if a == 0:
            raise ValueError(f"answer at position 0 of {text!r}: no prediction position")
        t_logits, _ = teacher(pc, ids, ids, base_state)
        s_logits, _ = teacher(pc, [], ids, base_state)
        items.append({"key": key, "answer_pos": a, "answer_tok": int(ids[a]),
                      "teacher_top1": int(torch.argmax(t_logits[a - 1])), "single_top1": int(torch.argmax(s_logits[a - 1]))})
    n_ok = sum(it["teacher_top1"] == it["answer_tok"] for it in items)
    return {"n": len(items), "n_ok": n_ok, "single_ok": sum(it["single_top1"] == it["answer_tok"] for it in items),
            "min_ok": G0_MIN, "pass": g0_pass(n_ok), "items": items}


def g0_pass(n_ok):
    return isinstance(n_ok, int) and n_ok >= G0_MIN


# ------------------------------------------------------------------ registered rules (pure; tested on synthetic rows)
def dev_bar(ns_recall, grad_recall):
    """The recall bar, the same in dev and in the verdict: max(0.5 x gradient, no_sleep + 2)."""
    return max(0.5 * grad_recall, ns_recall + RECALL_PLUS)


def dev_pick(g0, ref, rows):
    """g0: the G0 result ({pass}); ref: {"no_sleep": {recall}, "gradient": {recall, generic_kl, heldout_kl_dev}} on the dev
    seed; rows: [{lr, keep, recall, generic_kl, heldout_kl_dev, adapter_norm, err_head, err_tail}] of the v3 arm.
    G0 not passed -> STOP-G0. A non-finite ref -> UNDEFINED-DEV. Qualify: all finite; recall >= bar; generic_kl <= 1.25 x
    gradient's and <= 1; heldout_kl_dev <= 1.25 x gradient's and <= 1; engaged (adapter_norm > 0, err_tail < err_head).
    Pick max recall; ties: lower heldout_kl_dev, lower generic_kl, lower lr, lower keep. None -> NEGATIVE-DEV."""
    if not (isinstance(g0, dict) and g0.get("pass") is True):
        return {"verdict": "STOP-G0", "why": f"G0 not passed: {None if not isinstance(g0, dict) else g0.get('n_ok')}"
                f" of {None if not isinstance(g0, dict) else g0.get('n')} (needs >= {G0_MIN})", "pick": None}
    need = [ref["no_sleep"]["recall"], ref["gradient"]["recall"], ref["gradient"]["generic_kl"],
            ref["gradient"]["heldout_kl_dev"]]
    if not all(_finite(x) for x in need):
        return {"verdict": "UNDEFINED-DEV", "why": "non-finite ref", "pick": None, "n_rows": len(rows)}
    floor = dev_bar(ref["no_sleep"]["recall"], ref["gradient"]["recall"])
    cap_g = DRIFT_MULT * ref["gradient"]["generic_kl"]
    cap_h = DRIFT_MULT * ref["gradient"]["heldout_kl_dev"]
    keys = ("recall", "generic_kl", "heldout_kl_dev", "adapter_norm", "err_head", "err_tail")
    ok = [r for r in rows if all(_finite(r.get(k)) for k in keys)
          and r["recall"] >= floor and r["generic_kl"] <= cap_g and r["heldout_kl_dev"] <= cap_h
          and r["generic_kl"] <= KL_NIGHT_MAX and r["heldout_kl_dev"] <= KL_NIGHT_MAX
          and r["adapter_norm"] > 0 and r["err_tail"] < r["err_head"]]
    out = {"floor": floor, "cap_generic": cap_g, "cap_heldout_dev": cap_h, "n_rows": len(rows), "n_qualify": len(ok),
           "rows": [{k: r.get(k) for k in ("lr", "keep") + keys} for r in rows]}
    if not ok:
        return {**out, "verdict": "NEGATIVE-DEV", "pick": None}
    best = sorted(ok, key=lambda r: (-r["recall"], r["heldout_kl_dev"], r["generic_kl"], r["lr"], r["keep"]))[0]
    return {**out, "verdict": "PICKED", "pick": {"lr": best["lr"], "keep": best["keep"]}}


def _wins(recs, hi, lo, k, seeds):
    return sum(recs[hi][s][k] > recs[lo][s][k] for s in seeds)


def _eng(recs, arm, s):
    return recs[arm][s].get("engagement") or {}


def _falls(e, head="err_head", tail="err_tail"):
    hd, tl = e.get(head), e.get(tail)
    return _finite(hd) and _finite(tl) and tl < hd


def engagement(recs, seeds=SEEDS, last_layer=None):
    """{"core": [...], "controls": [...], "report": [...]}: why an arm's night was not what its role needs, per seed.
    core (no_sleep, gradient, v3): no_sleep adapter norm 0; gradient norm > 0 and loss tail < head; v3 norm > 0, finite,
    replay error tail < head, teacher "double", every layer on. controls: sham_ctx norm >= 0.5 x v3's and > 0, error tail <
    head, teacher "sham" with a context read; top_only norm > 0, error tail < head, exactly one layer on (the last when
    given); no_target ran its steps with teacher "none" and no context (its inertness is clause (d)). report
    (grad_distill): norm > 0 and loss tail < head; it gates only the NEGATIVE reading."""
    core, ctl, rep = [], [], []
    for s in seeds:
        n = lambda arm: recs[arm][s].get("adapter_norm")
        if n("no_sleep") != 0:
            core.append(f"no_sleep s{s}: adapter_norm {n('no_sleep')} != 0")
        if not (_finite(n("gradient")) and n("gradient") > 0 and _falls(_eng(recs, "gradient", s), "loss_head", "loss_tail")):
            core.append(f"gradient s{s} not engaged: norm {n('gradient')}, loss {_eng(recs, 'gradient', s)}")
        e = _eng(recs, "v3", s)
        if not (_finite(n("v3")) and n("v3") > 0 and _falls(e) and e.get("teacher") == "double"
                and e.get("steps_run", 0) > 0 and len(e.get("layers_on") or []) > 1):
            core.append(f"v3 s{s} not engaged: norm {n('v3')}, err {e.get('err_head')}->{e.get('err_tail')}, "
                        f"teacher {e.get('teacher')}, layers {e.get('layers_on')}")
        e = _eng(recs, "sham_ctx", s)
        v = n("v3")
        if not (_finite(n("sham_ctx")) and n("sham_ctx") > 0 and _finite(v) and n("sham_ctx") >= ENGAGE_NORM_MIN * v
                and _falls(e) and e.get("teacher") == "sham" and (e.get("target_ctx_tokens") or 0) > 0):
            ctl.append(f"sham_ctx s{s} not engaged: norm {n('sham_ctx')} vs v3 {v} (needs > 0 and >= "
                       f"{ENGAGE_NORM_MIN} x v3), err {e.get('err_head')}->{e.get('err_tail')}, teacher {e.get('teacher')}")
        e = _eng(recs, "top_only", s)
        lo = e.get("layers_on") or []
        if not (_finite(n("top_only")) and n("top_only") > 0 and _falls(e) and len(lo) == 1
                and (last_layer is None or lo == [last_layer])):
            ctl.append(f"top_only s{s} not engaged: norm {n('top_only')}, err {e.get('err_head')}->{e.get('err_tail')}, "
                       f"layers {lo}")
        e = _eng(recs, "no_target", s)
        if not (e.get("steps_run", 0) > 0 and e.get("teacher") == "none" and e.get("target_ctx_tokens") == 0):
            ctl.append(f"no_target s{s} did not run its lesion night: {e}")
        if not (_finite(n("grad_distill")) and n("grad_distill") > 0
                and _falls(_eng(recs, "grad_distill", s), "loss_head", "loss_tail")):
            rep.append(f"grad_distill s{s} not engaged: norm {n('grad_distill')}, loss {_eng(recs, 'grad_distill', s)}")
    return {"core": core, "controls": ctl, "report": rep}


def _passes_a_c(recs, arm, seeds, bar, cg, ch):
    m = lambda k: _mean(recs[arm][s][k] for s in seeds)
    return (m("recall") >= bar and m("generic_kl") <= cg and m("heldout_kl") <= ch
            and all(recs[arm][s]["generic_kl"] <= KL_NIGHT_MAX and recs[arm][s]["heldout_kl"] <= KL_NIGHT_MAX
                    for s in seeds))


def verdict(recs, last_layer=None):
    """Per model. recs: {arm: {seed: {recall, false_recall, generic_kl, heldout_kl (VERDICT half), adapter_norm,
    engagement}}}, every arm of ARMS on exactly SEEDS (else ValueError). Order:
    1. UNDEFINED: any gated value of a gated arm non-finite; any v3 adapter norm non-finite; mean gradient recall < 2;
       a core engagement gate fails (no_sleep, gradient, v3); (d) violated (no_target mean recall > no_sleep mean, or a
       no_target adapter norm > 1e-6: a leak).
    2. core (a) v3 recall >= bar; (c) v3 generic and held-out KL each <= 1.25 x gradient's and no v3 night above KL 1;
       (e) v3 false recall <= gradient + 1. A failing core clause -> NEGATIVE, with a reading from grad_distill.
    3. a control engagement gate fails (sham_ctx, top_only, no_target) -> UNDEFINED.
    4. (b) v3 > sham_ctx on >= 5 seeds AND sham_ctx mean <= no_sleep + 1; fails -> NEGATIVE.
    5. v3 mean - top_only mean < 1 -> GO-SHALLOW, else GO-V3 (per model; `combine` names the two-scale verdict)."""
    for arm in ARMS:
        if arm not in recs or set(recs[arm]) != set(SEEDS):
            raise ValueError(f"verdict needs every arm on exactly the seeds {SEEDS}: {arm} has {sorted(recs.get(arm, {}))}")
    seeds = list(SEEDS)
    why = [f"{arm} s{s} {k} non-finite" for arm in GATED_ARMS for s in seeds for k in GATED
           if not _finite(recs[arm][s].get(k))]
    why += [f"v3 s{s} adapter_norm non-finite" for s in seeds if not _finite(recs["v3"][s].get("adapter_norm"))]
    if why:
        return {"n_seeds": len(seeds), "verdict": "UNDEFINED", "why": why}
    m = lambda arm, k: _mean(recs[arm][s][k] for s in seeds)
    G, NS = m("gradient", "recall"), m("no_sleep", "recall")
    bar = dev_bar(NS, G)
    cg, ch = DRIFT_MULT * m("gradient", "generic_kl"), DRIFT_MULT * m("gradient", "heldout_kl")
    nwin = _wins(recs, "v3", "sham_ctx", "recall", seeds)
    nt_norms = [recs["no_target"][s].get("adapter_norm") for s in seeds]
    cl = {"a": m("v3", "recall") >= bar,
          "b": nwin >= WIN_SEEDS and m("sham_ctx", "recall") <= NS + SHAM_PLUS,
          "c_generic": m("v3", "generic_kl") <= cg,
          "c_heldout": m("v3", "heldout_kl") <= ch,
          "c_night": all(recs["v3"][s]["generic_kl"] <= KL_NIGHT_MAX and recs["v3"][s]["heldout_kl"] <= KL_NIGHT_MAX
                         for s in seeds),
          "d": m("no_target", "recall") <= NS and all(_finite(x) and abs(x) <= INERT_NORM for x in nt_norms),
          "e": m("v3", "false_recall") <= m("gradient", "false_recall") + FALSE_PLUS}
    eng = engagement(recs, seeds, last_layer)
    out = {"n_seeds": len(seeds), "bar": bar, "cap_generic": cg, "cap_heldout": ch, "v3_beats_sham": nwin,
           "v3_minus_top_only": m("v3", "recall") - m("top_only", "recall"),
           "means": {a: {k: (m(a, k) if all(_finite(recs[a][s].get(k)) for s in seeds) else None)
                         for k in GATED + ("adapter_norm",)} for a in ARMS},
           "clauses": cl, "engagement": eng}
    why = []
    if G < GRAD_MIN:
        why.append(f"gradient mean recall {G} < {GRAD_MIN}")
    why += eng["core"]
    if not cl["d"]:
        why.append(f"(d) violated: no_target recall {m('no_target', 'recall')} vs no_sleep {NS}, norms {nt_norms}")
    if why:
        return {**out, "verdict": "UNDEFINED", "why": why}
    if not all(cl[k] for k in ("a", "c_generic", "c_heldout", "c_night", "e")):
        return {**out, "verdict": "NEGATIVE", "reading": negative_reading(recs, seeds, bar, cg, ch, eng)}
    if eng["controls"]:
        return {**out, "verdict": "UNDEFINED", "why": eng["controls"]}
    if not cl["b"]:
        return {**out, "verdict": "NEGATIVE", "reading": "contrast failed: in-context activity of another fact does "
                "as well (or the sham moved recall); not credited to the told content"}
    if out["v3_minus_top_only"] < SHALLOW_MIN:
        return {**out, "verdict": "GO-SHALLOW"}
    return {**out, "verdict": "GO-V3"}


def negative_reading(recs, seeds, bar, cg, ch, eng):
    """What a core NEGATIVE means, from grad_distill (report only): passes (a)+(c) -> the delta rule fails even with
    aligned targets (the delta-rule family stops at this adapter rank); fails -> the single-form target is the limit
    (next: question-form cues); not engaged or non-finite -> undetermined."""
    gd = [recs["grad_distill"][s] for s in seeds]
    if eng["report"] or not all(_finite(r.get(k)) for r in gd for k in GATED):
        return "undetermined: grad_distill not engaged or non-finite"
    if _passes_a_c(recs, "grad_distill", seeds, bar, cg, ch):
        return "rule-limit: grad_distill passes (a) and (c) on the same target; the delta-rule family stops at this rank"
    return "target-limit: grad_distill fails (a) or (c) too; the single-form target is the limit (next: question-form cues)"


def combine(v01, v04):
    """Two-scale verdict from the per-model verdict words: GO-V3 (both GO-V3); GO-V3-PARTIAL (exactly one GO-V3, the
    other defined); UNDEFINED if either is UNDEFINED*/ERROR*/HOLD*/missing; else GO-SHALLOW if either is GO-SHALLOW;
    else NEGATIVE (NEGATIVE or NEGATIVE-DEV at both)."""
    ws = [v01, v04]
    if any(not isinstance(w, str) or w.startswith(("UNDEFINED", "ERROR", "HOLD", "STOP")) for w in ws):
        return "UNDEFINED"
    go = sum(w == "GO-V3" for w in ws)
    if go == 2:
        return "GO-V3"
    if go == 1:
        return "GO-V3-PARTIAL"
    if "GO-SHALLOW" in ws:
        return "GO-SHALLOW"
    return "NEGATIVE"


# ------------------------------------------------------------------ CLI
def cmd_pick(a):
    """Dev pick. Refuses (exit non-zero, no pick.json) unless the dir holds exactly the G0 unit, the 2 ref units and one
    v3 unit per (lr, keep) cell, all seed 7, one model, one commit, the pinned prompts."""
    units = [u for u in _load_dir(a.dir, "*.json") if isinstance(u, dict) and "arm" in u and u.get("seed") == DEV_SEED]
    ref_u = [u for u in units if u["tag"] == a.ref_tag and u["arm"] in ("no_sleep", "gradient")]
    g0_u = [u for u in units if u["tag"] == a.ref_tag and u["arm"] == "g0"]
    v3_u = [u for u in units if u["tag"] != a.ref_tag and u["arm"] == "v3"]
    bad = []
    if sorted(u["arm"] for u in ref_u) != ["gradient", "no_sleep"]:
        bad.append(f"ref units {sorted(u['arm'] for u in ref_u)} != [gradient, no_sleep]")
    if len(g0_u) != 1:
        bad.append(f"{len(g0_u)} G0 units != 1")
    cells = sorted((u.get("lr"), u.get("keep")) for u in v3_u)
    if cells != sorted(itertools.product(DEV_LRS, DEV_KEEPS)):
        bad.append(f"v3 cells {cells} != the 9 registered (lr, keep)")
    if bad:
        raise SystemExit("pick refused: " + "; ".join(bad))
    allu = g0_u + ref_u + v3_u
    model, commit = _one(allu, "model_id", "pick"), _one(allu, "commit", "pick")
    _one(ref_u + v3_u, "prompts_sha256", "pick")
    if not commit:
        raise SystemExit("pick refused: units carry no commit")
    ref = {u["arm"]: u["result"] for u in ref_u}
    rows = [{"lr": u["lr"], "keep": u["keep"], **u["result"],
             **{k: (u.get("engagement") or {}).get(k) for k in ("err_head", "err_tail")}} for u in v3_u]
    res = {**dev_pick(g0_u[0]["result"], ref, rows), "model_id": model, "commit": commit,
           "prompts_sha256": ref_u[0].get("prompts_sha256"), "g0": {k: g0_u[0]["result"].get(k) for k in ("n", "n_ok", "single_ok", "pass")}}
    print(json.dumps(res, indent=1))
    if a.out:
        atomic_write(a.out, res)


def check_units(units, pick=None):
    """One instrument: one model, commit and prompts list; one (lr, keep) across the local arms; the pick's when given."""
    model, commit = _one(units, "model_id", "score"), _one(units, "commit", "score")
    _one(units, "prompts_sha256", "score")
    loc = [u for u in units if u["arm"] in LOCAL_ARMS]
    lr, keep = _one(loc, "lr", "score (local arms)"), _one(loc, "keep", "score (local arms)")
    if pick is not None:
        want = (pick["pick"]["lr"], pick["pick"]["keep"], pick["model_id"], pick["commit"])
        if (lr, keep, model, commit) != want:
            raise SystemExit(f"units (lr, keep, model, commit) {(lr, keep, model, commit)} != pick {want}")
    return {"lr": lr, "keep": keep, "model_id": model, "commit": commit}


def cmd_score(a):
    units = [u for u in _load_dir(a.dir, f"{a.tag}_s*_*.json") if u.get("seed") in SEEDS and u.get("tag") == a.tag
             and u.get("arm") in ARMS]
    recs = {}
    for u in units:
        rec = dict(u["result"])
        rec["engagement"] = u.get("engagement") or {}
        recs.setdefault(u["arm"], {})[u["seed"]] = rec
    missing = [(arm, s) for arm in ARMS for s in SEEDS if s not in recs.get(arm, {})]
    if missing:
        raise SystemExit(f"missing units: {missing}")
    inst = check_units(units, read_pick(a.pick) if a.pick else None)
    last = _one(units, "last_layer", "score")
    res = {**verdict(recs, last), "instrument": inst}
    print(json.dumps(res, indent=1))
    if a.out:
        atomic_write(a.out, res)


def cmd_combine(a):
    vs = []
    for p in (a.v01, a.v04):
        with open(p) as fh:
            vs.append(json.load(fh).get("verdict"))
    res = {"v01": vs[0], "v04": vs[1], "verdict": combine(*vs)}
    print(json.dumps(res, indent=1))
    if a.out:
        atomic_write(a.out, res)


def run_units(a, c, base, *, PROMPTS, GENERIC, build_arm_episodes, evaluate, sleep, keys_all, told, PlasticCortex):
    """The per-(seed, arm) loop of `run` on a loaded cortex `c` (tests pass a stub cortex). Each arm is one atomic unit;
    existing units are skipped (resume). Arm "g0" (if asked) runs first; a failing G0 writes its unit and stops before
    any night (exit non-zero)."""
    dev_p, ver_p = split_prompts(PROMPTS, a.prompts_sha)
    psha = prompts_sha(PROMPTS)
    if a.pick:
        p = read_pick(a.pick)
        if p.get("model_id") != _model_id(a.model):
            raise SystemExit(f"pick model {p.get('model_id')} != --model {_model_id(a.model)}")
        a.lr, a.keep = p["pick"]["lr"], p["pick"]["keep"]
    prov = provenance(a.device, "float32")
    if a.pick and p.get("commit") != prov["commit"]:
        raise SystemExit(f"pick commit {p.get('commit')} != this code's commit {prov['commit']}")
    order = (["g0"] if "g0" in a.arm_list else []) + [x for x in a.arm_list if x != "g0"]
    todo = [arm for arm in order if not os.path.exists(unit_path(a.out_dir, a.tag, a.seed, arm))]
    g0f = unit_path(a.out_dir, a.tag, a.seed, "g0")
    if "g0" in a.arm_list and "g0" not in todo:
        with open(g0f) as fh:
            if not json.load(fh)["result"].get("pass"):
                raise SystemExit(f"G0 FAILED (unit {g0f}): no nights run")
    eps = build_arm_episodes(a.seed)
    base_unit = {"tag": a.tag, "seed": a.seed, "model": a.model, "model_id": _model_id(a.model), "steps": a.steps,
                 "prompts_sha256": psha, "pick": a.pick, **prov}
    for arm in todo:
        t1 = time.time()
        torch.manual_seed(a.seed)
        pc = PlasticCortex(c)
        last = pc.layers[-1]
        if arm == "g0":
            r = g0_check(pc, base, told)
            unit = {**base_unit, "arm": "g0", "last_layer": last, "elapsed_s": time.time() - t1,
                    "engagement": {"kind": "gate", "n": r["n"]}, "result": r}
            atomic_write(g0f, unit)
            print("g0", {k: r[k] for k in ("n", "n_ok", "single_ok", "pass")}, flush=True)
            if not r["pass"]:
                raise SystemExit(f"G0 FAILED: teacher top-1 is the answer for {r['n_ok']} of {r['n']} (needs >= {G0_MIN}): "
                                 "no nights run")
            continue
        losses, eng = [], {}
        lg = lambda msg, arm=arm: print(arm, msg, flush=True)
        if arm == "no_sleep":
            eng = {"kind": "none", "steps_run": 0}
        elif arm == "gradient":
            losses = sleep(pc, base, eps["sleep_true"], GENERIC, steps=a.steps, lr=a.grad_lr, kl_weight=a.kl_weight,
                           seed=a.seed, log=lg)
            eng = _loss_engagement("gradient", losses)          # (replay CE, keep KL) per step
        elif arm == "grad_distill":
            distill_sleep(pc, base, eps["sleep_true"], GENERIC, steps=a.steps, lr=a.grad_lr, kl_weight=a.kl_weight,
                          seed=a.seed, log=lg, info=eng)
        else:
            v3_sleep(pc, base, eps["sleep_true"], GENERIC, steps=a.steps, lr=a.lr, keep=a.keep, decay=a.decay,
                     seed=a.seed, teacher_kind=TEACHER_OF[arm], top_only=(arm == "top_only"), log=lg, info=eng)
        r = evaluate(pc, base, keys_all)
        # dev/verdict firewall: the VERDICT half is never computed on the dev seed
        r["heldout_kl"] = None if a.seed == DEV_SEED else heldout_drift(pc, base, ver_p)
        r["heldout_kl_dev"] = heldout_drift(pc, base, dev_p)
        r["adapter_norm"] = pc.adapter_norm()
        r["factor_norms"] = pc.factor_norms()
        eng["adapter_norm"] = r["adapter_norm"]
        unit = {**base_unit, "arm": arm, "last_layer": last,
                "lr": a.lr if arm in LOCAL_ARMS else None, "keep": a.keep if arm in LOCAL_ARMS else None,
                "decay": a.decay if arm in LOCAL_ARMS else None,
                "grad_lr": a.grad_lr if arm in ("gradient", "grad_distill") else None,
                "teacher": TEACHER_OF.get(arm), "elapsed_s": time.time() - t1, "engagement": eng, "result": r}
        atomic_write(unit_path(a.out_dir, a.tag, a.seed, arm), unit)
        print(arm, {k: v for k, v in r.items() if not k.endswith("_items")}, flush=True)
    return todo


def cmd_run(a):
    guard()
    want = [x.strip() for x in a.arms.split(",") if x.strip()]
    if [x for x in want if x not in ARMS + ("g0",)]:
        raise SystemExit(f"unknown arms; choose from {ARMS + ('g0',)}")
    a.arm_list = want
    if not [arm for arm in want if not os.path.exists(unit_path(a.out_dir, a.tag, a.seed, arm))]:
        print("all units done", flush=True)
        return
    from brain.brain import BIRTH
    from brain.cortex import Cortex
    from brain.experiments.facts import FACTS
    from brain.experiments.mix_night_screen import PROMPTS
    from brain.experiments.sleep_probe import GENERIC, _told, build_arm_episodes, evaluate
    from brain.plastic_cortex import PlasticCortex as PC
    from brain.sleep import sleep
    t0 = time.time()
    c = Cortex(model_path=a.model, device=a.device)
    _, base = c.read(BIRTH, None)
    base = [s.float() for s in base]
    told = [(_told(s)[0], k) for s, _, k in FACTS]
    todo = run_units(a, c, base, PROMPTS=PROMPTS, GENERIC=GENERIC, build_arm_episodes=build_arm_episodes,
                     evaluate=evaluate, sleep=sleep, keys_all=[k for _, _, k in FACTS], told=told, PlasticCortex=PC)
    print(f"done {len(todo)} units in {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--arms", default=",".join(ARMS))
    r.add_argument("--lr", type=float, default=0.1)
    r.add_argument("--keep", type=float, default=1.0)
    r.add_argument("--decay", type=float, default=1e-3)
    r.add_argument("--kl-weight", type=float, default=1.0)
    r.add_argument("--grad-lr", type=float, default=5e-4)
    r.add_argument("--steps", type=int, default=240)
    r.add_argument("--device", default="cpu")
    r.add_argument("--tag", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--pick", default=None, help="dev pick.json: lr and keep from it; its model and commit must match")
    r.add_argument("--prompts-sha", default=PROMPTS_SHA256)
    p = sub.add_parser("pick")
    p.add_argument("--dir", required=True)
    p.add_argument("--ref-tag", default="ref")
    p.add_argument("--out", default=None)
    s = sub.add_parser("score")
    s.add_argument("--dir", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--out", default=None)
    s.add_argument("--pick", default=None)
    c = sub.add_parser("combine")
    c.add_argument("--v01", required=True)
    c.add_argument("--v04", required=True)
    c.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    {"run": cmd_run, "pick": cmd_pick, "score": cmd_score, "combine": cmd_combine}[a.cmd](a)


if __name__ == "__main__":
    main()
