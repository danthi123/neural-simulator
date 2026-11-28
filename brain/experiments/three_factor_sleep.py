# Capability: Sleep consolidation
# Record: brain/research/2026-11-28-local-rule-research-round.md
# Expected-fail: the perturbation estimate has high variance (about 2 x layers x 8 x T noise dimensions per text, some
#   10^4 at 0.1B), so at a 30x forward budget the accumulated update may stay too weakly aligned (cosine under 0.1) or
#   too slow (replay loss not under 1.0); the rank units' output projection B is fixed random in this rule (only A
#   learns), so the adapters span a fixed random subspace per layer. A negative is a verdict on this perturbation
#   method at this budget and parameterisation, not on three-factor learning.
"""Three-factor sleep: reward-gated perturbation on the adapters' rank units (candidate 1 of the 11-28 research note,
MeZO family). Forward-only: every night forward runs under torch.no_grad; no feedback path, no weight transport.

Rank units. Each channel-mix adapter is delta = A @ B with 8 rank units in between: u = x @ A (x the adapter's
pre-synaptic activity: k_in for the key adapter, the hidden h for the value adapter), output u @ B. PlasticCortex
starts B at zero, which silences the rank units' effect, so noise on them could not change anything. This rule
re-parameterises at the start of the night (`reparam_rank`): A = 0, B = a fixed random projection (seeded, entries
N(0, 1/r)). The product is still zero, so the cortex starts exactly pretrained (tested). Only A learns; B stays fixed.

Per step (one episode by strength, then one keep text, the same RNG draws as brain.sleep / v1 / v3):
1. Draw unit Gaussian noise xi (T, 8) per adapter per text; inject +sigma xi and -sigma xi into the rank-unit
   activities (estimator "antithetic", 4 forwards per perturbation: two texts x two signs) or 0 and +sigma xi
   ("forward", 2 + 2 forwards per perturbation, the clean pass shared).
2. One scalar for the whole cortex: s = (L+ - L-) / (2 sigma) (or (L+ - L0) / sigma), L = replay CE + kl_weight x
   keep KL to the pretrained cortex. It says whether that noise made the night's objective go down.
3. Update = scalar x noise x pre-synaptic activity: dA = s x^T xi per adapter (x = the mean of the two passes'
   pre-synaptic activity, or the clean pass's). E[dA] = dL/dA (tested on a tiny cortex: positive cosine).
4. Step: A -= lr sqrt(P) dA / m, m = an EMA of the past estimate norms (normalised SGD, so lr is a per-parameter RMS
   step like Adam's and one lr grid serves every arm; m uses only past steps, so the step's direction stays unbiased).
5. Optional (MeZO-SVRG form, `--svrg q`): every q steps an anchor copy of A and a full-batch estimate over every telling
   and keep text; each step adds s(A) N(A) - s(A~) N(A~) with the SAME noise at the anchor, plus the full estimate.

Budget: 30 x the gradient arm's forward passes. The gradient night runs 2 forwards per step (replay + keep), 240 steps
= 480 forwards (+ 240 backwards, not counted); the rule gets 30 x 480 = 14400 text forwards, every one counted, anchor
refreshes included; the number of steps follows (antithetic, 1 perturbation: 3600 steps).

Arms: three_factor; tf_shuffled (the scalar is a draw from earlier steps' scalars: same magnitudes, no relation to this
step's noise); tf_wrong_sign (-s: ascent in expectation); no_sleep; gradient (brain.sleep.sleep, the reference);
gradient_rank (report only: Adam on the SAME re-parameterisation, A from 0 with B fixed; reads a NEGATIVE as rule-limit
or parameterisation-limit).

Instrument (backprop as an instrument only, never the rule): at 9 points of the night (8 windows) the true gradient of
the full fixed-order objective (every telling's replay CE + kl_weight x every keep text's KL) with respect to A; per
window, cosine between the rule's accumulated update (-dA over the window) and that gradient (`cos_window` = mean over
windows); also over the whole night. Engagement also has the fixed-order objective and replay CE at start and end.
Read-outs, hippocampus off: recall /12 (sleep_probe.evaluate) and the graded read-out (log-probability and rank of the
answer word at the question).

    POOL_NODE=1 python -m brain.experiments.three_factor_sleep run --model M.pth --seed 7 \\
        --arms no_sleep,gradient,gradient_rank --out-dir D --tag ref
    POOL_NODE=1 python -m brain.experiments.three_factor_sleep run --model M.pth --seed 7 --arms three_factor \\
        --lr 3e-4 --sigma 0.03 --out-dir D --tag lr0.0003_sig0.03
    python -m brain.experiments.three_factor_sleep pick --dir D --out D/pick.json   # 3 refs + 9 cells required
    POOL_NODE=1 python -m brain.experiments.three_factor_sleep run --model M.pth --seed 42 --pick D/pick.json \\
        --out-dir S --tag six
    python -m brain.experiments.three_factor_sleep score --dir S --tag six --pick D/pick.json
    python -m brain.experiments.three_factor_sleep combine --v01 V01.json --v04 V04.json
Prereg: brain/research/2026-11-28-three-factor-sleep-prereg-DRAFT.md (DRAFT; nothing has run).
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from brain.experiments.local_rule_v2 import (PROMPTS_SHA256, _finite, _load_dir, _mean, _model_id, _one, atomic_write,
                                             heldout_drift, prompts_sha, provenance, read_pick, split_prompts, unit_path)
from brain.experiments.local_rule_v3 import _loss_engagement, replay_ce
from brain.plastic_cortex import PlasticCortex

SEEDS = (42, 43, 44, 100, 101, 102)
DEV_SEED = 7
ARMS = ("no_sleep", "gradient", "three_factor", "tf_shuffled", "tf_wrong_sign", "gradient_rank")
TF_ARMS = ("three_factor", "tf_shuffled", "tf_wrong_sign")          # one (lr, sigma, estimator, ...) across these
SCALAR_OF = {"three_factor": "true", "tf_shuffled": "shuffled", "tf_wrong_sign": "wrong_sign"}
GATED_ARMS = ("no_sleep", "gradient", "three_factor", "tf_shuffled", "tf_wrong_sign")   # gradient_rank: report only
GATED = ("recall", "false_recall", "generic_kl", "heldout_kl")
REF_ARMS = ("gradient", "gradient_rank", "no_sleep")
DEV_LRS = (3e-5, 1e-4, 3e-4)
DEV_SIGMAS = (0.01, 0.03, 0.1)
BUDGET_MULT = 30        # the rule's text forwards = 30 x the gradient arm's (2 per step)
BUDGET_USE_MIN = 0.9    # engaged: used >= 90% of the budget (and never more than it)
KILL_REPLAY = 1.0       # kill: fixed-order replay CE at the night's end not under 1.0 ...
KILL_COS = 0.1          # ... or the mean window cosine to the true gradient under 0.1
DRIFT_MULT = 1.25       # (c) generic and held-out KL each <= 1.25 x gradient's
KL_NIGHT_MAX = 1.0      # (c) no three_factor night with generic or held-out KL above 1
RECALL_PLUS = 2         # (a) bar = max(0.5 x gradient, no_sleep + 2): the note's "recall at the floor"
CTRL_PLUS = 1           # (b) each control's mean recall <= no_sleep + 1
WIN_SEEDS = 5           # (b) three_factor > each control on >= 5 of 6 seeds
FALSE_PLUS = 1          # (e) false recall <= gradient + 1
GRAD_MIN = 2            # UNDEFINED if mean gradient recall < 2
ENGAGE_NORM_MIN = 0.5   # each control's adapter norm >= 0.5 x that seed's three_factor norm
CORR_TRUE_MIN = 0.99    # three_factor applies its own scalar (corr(used, own) >= 0.99)
CORR_SHUF_MAX = 0.2     # tf_shuffled: |corr(used, own)| <= 0.2
N_WINDOWS = 8
SHUF_BUF = 256
SHUF_SEED_OFFSET = 7919
RANK_SEED_OFFSET = 104729
NOISE_SEED_OFFSET = 15485863
EMA = 0.9


# ------------------------------------------------------------------ guard
def guard(env=None):
    env = os.environ if env is None else env
    if env.get("POOL_NODE") != "1" and env.get("BRAIN_ALLOW_W3090") != "1":
        raise SystemExit("three_factor_sleep run: refuses to run unless POOL_NODE=1 or BRAIN_ALLOW_W3090=1")


# ------------------------------------------------------------------ the perturbable cortex
class TFCortex(PlasticCortex):
    """PlasticCortex whose channel-mix rank units can take additive noise (`pert` {layer: (noise_k (T,r), noise_v
    (T,r))}) and record each adapter's pre-synaptic activity (`rec` list of (i, k_in, h)). With both None it is
    PlasticCortex exactly (the same code path)."""
    pert = None
    rec = None

    def _cmix(self, i, x, x_prev):
        s = str(i)
        if (self.pert is None and self.rec is None) or not (self.enabled and s in self.A_k):
            return PlasticCortex._cmix(self, i, x, x_prev)
        assert not torch.is_grad_enabled(), "three-factor forward must run under no_grad"
        f = f"blocks.{i}.ffn."
        xx = torch.cat((x_prev.unsqueeze(0), x[:-1, :])) - x
        k_in = x + xx * self.z[f + "x_k"]
        u_k = k_in @ self.A_k[s]
        if self.pert is not None:
            u_k = u_k + self.pert[i][0]
        h = torch.relu(k_in @ self.z[f + "key.weight"] + u_k @ self.B_k[s]) ** 2
        u_v = h @ self.A_v[s]
        if self.pert is not None:
            u_v = u_v + self.pert[i][1]
        if self.rec is not None:
            self.rec.append((i, k_in, h))
        return h @ self.z[f + "value.weight"] + u_v @ self.B_v[s], x[-1, :]


def adopt_tf(plastic) -> TFCortex:
    """A TFCortex sharing the adapters (Parameters) of an existing PlasticCortex; the original is untouched."""
    lc = TFCortex.__new__(TFCortex)
    lc.__dict__.update(plastic.__dict__)
    return lc


def reparam_rank(pc, seed):
    """A = 0, B = fixed random N(0, 1/r) (seeded), B frozen (requires_grad False). The product stays zero, so the cortex
    is exactly pretrained, and the rank units now reach the output."""
    g = torch.Generator().manual_seed(seed + RANK_SEED_OFFSET)
    with torch.no_grad():
        for s in pc.A_k:
            for A, B in ((pc.A_k[s], pc.B_k[s]), (pc.A_v[s], pc.B_v[s])):
                r = B.shape[0]
                A.zero_()
                B.copy_((torch.randn(*B.shape, generator=g) / r ** 0.5).to(B.device, B.dtype))
                B.requires_grad_(False)


def a_params(pc):
    """[(key, Parameter)] of the learned factors, in a fixed order."""
    return [((s, "k"), pc.A_k[s]) for s in pc.A_k] + [((s, "v"), pc.A_v[s]) for s in pc.A_v]


def _flat_cos(a, b):
    na = math.sqrt(sum(float(x.pow(2).sum()) for x in a.values()))
    nb = math.sqrt(sum(float(x.pow(2).sum()) for x in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return sum(float((a[k] * b[k]).sum()) for k in a) / (na * nb)


def _norm(d):
    return math.sqrt(sum(float(x.pow(2).sum()) for x in d.values()))


# ------------------------------------------------------------------ one perturbation sample
def draw_noise(pc, T, gen):
    """Unit Gaussian noise {layer: (xi_k (T,r), xi_v (T,r))} for one text, from the CPU generator `gen`."""
    dev = pc.z["emb.weight"].device
    out = {}
    for s in pc.A_k:
        r_k, r_v = pc.A_k[s].shape[1], pc.A_v[s].shape[1]
        out[int(s)] = (torch.randn(T, r_k, generator=gen).to(dev), torch.randn(T, r_v, generator=gen).to(dev))
    return out


def text_loss(logits, ids, kind, ref):
    if kind == "replay":
        return float(F.cross_entropy(logits[:-1], torch.tensor(ids[1:], device=logits.device)))
    return float(F.kl_div(F.log_softmax(logits, -1), ref, log_target=True, reduction="batchmean"))


def _pass(pc, texts, state, noises, sign, counter):
    """One forward per text with noise sign x noise (sign 0 or noises None: clean). Returns (weighted objective, recs)."""
    tot, recs = 0.0, []
    for (ids, kind, ref, w), nz in zip(texts, noises):
        pc.pert = None if (nz is None or sign == 0) else {i: (sign * a, sign * b) for i, (a, b) in nz.items()}
        pc.rec = []
        try:
            with torch.no_grad():
                lg = pc(ids, state)[0]
            rec = {i: (k, h) for i, k, h in pc.rec}
        finally:
            pc.pert, pc.rec = None, None
        counter["n"] += 1
        tot += w * text_loss(lg, ids, kind, ref)
        recs.append(rec)
    return tot, recs


def zo_sample(pc, texts, state, sigma, noises, estimator, counter):
    """One perturbation sample. texts: [(ids, kind, ref, weight)]; noises: one draw_noise per text (unit). Returns
    (s, N, L_a, L_b): the scalar s and the noise x pre-synaptic term N {(layer, k|v): x^T xi summed over texts}, so the
    estimate of dL/dA is s x N."""
    if estimator == "antithetic":
        Lp, rp = _pass(pc, texts, state, [{i: (sigma * a, sigma * b) for i, (a, b) in n.items()} for n in noises], 1, counter)
        Lm, rm = _pass(pc, texts, state, [{i: (sigma * a, sigma * b) for i, (a, b) in n.items()} for n in noises], -1, counter)
        s = (Lp - Lm) / (2 * sigma)
        pre = [{i: ((p[i][0] + m[i][0]) / 2, (p[i][1] + m[i][1]) / 2) for i in p} for p, m in zip(rp, rm)]
        La, Lb = Lp, Lm
    elif estimator == "forward":
        L0, r0 = _pass(pc, texts, state, [None] * len(texts), 0, counter)
        L1, _ = _pass(pc, texts, state, [{i: (sigma * a, sigma * b) for i, (a, b) in n.items()} for n in noises], 1, counter)
        s = (L1 - L0) / sigma
        pre, La, Lb = r0, L1, L0
    else:
        raise ValueError(estimator)
    N = {}
    with torch.no_grad():
        for p, n in zip(pre, noises):
            for i, (k_in, h) in p.items():
                xi_k, xi_v = n[i]
                for key, d in (((str(i), "k"), k_in.T @ xi_k), ((str(i), "v"), h.T @ xi_v)):
                    N[key] = N[key] + d if key in N else d
    return s, N, La, Lb


# ------------------------------------------------------------------ budget
def step_cost(estimator, n_pert, svrg):
    per = 4 * n_pert if estimator == "antithetic" else 2 + 2 * n_pert
    return per * (2 if svrg else 1)


def refresh_cost(estimator, n_full, n_texts):
    return 2 * n_texts * n_full if estimator == "antithetic" else n_texts * (1 + n_full)


def plan_steps(budget, estimator, n_pert, svrg, svrg_full, n_texts):
    """(steps, forwards) that fit the budget: a step runs only if all its forwards (and the anchor refresh due at it)
    fit."""
    used, t = 0, 0
    while True:
        c = step_cost(estimator, n_pert, svrg) + (refresh_cost(estimator, svrg_full, n_texts) if svrg and t % svrg == 0 else 0)
        if used + c > budget:
            return t, used
        used += c
        t += 1


# ------------------------------------------------------------------ the instrument (backprop, never the rule)
def full_objective(pc, state, tell_ids, gen_ids, keep_ref, kl_weight, grad=False):
    """Fixed-order objective: mean replay CE over every telling + kl_weight x mean keep KL over every keep text. With
    grad, also its true gradient {(layer, k|v): dL/dA} by autograd (instrument only), backpropagated ONE TEXT AT A TIME
    and summed, so only one text's graph is alive at once (pre-data review item 1: peak memory as gradient sleep's).
    Returns (L, replay, keep, G)."""
    params = a_params(pc)
    dev = pc.z["emb.weight"].device
    nt, ng = len(tell_ids), len(gen_ids)
    G = {k: torch.zeros_like(p) for k, p in params} if grad else None
    rep_v, kp_v = 0.0, 0.0
    terms = [("t", ids, None, 1.0 / nt) for ids in tell_ids] + \
            [("k", g, r, kl_weight / ng) for g, r in zip(gen_ids, keep_ref)]
    for kind, ids, r, w in terms:
        with torch.enable_grad() if grad else torch.no_grad():
            if kind == "t":
                li = F.cross_entropy(pc(ids, state)[0][:-1], torch.tensor(ids[1:], device=dev))
            else:
                li = F.kl_div(F.log_softmax(pc(ids, state)[0], -1), r, log_target=True, reduction="batchmean")
            if grad:
                gs = torch.autograd.grad(w * li, [p for _, p in params], allow_unused=True)
                for (k, _), g in zip(params, gs):
                    if g is not None:
                        G[k] += g.detach()
        if kind == "t":
            rep_v += float(li.detach()) / nt
        else:
            kp_v += float(li.detach()) / ng
    return rep_v + kl_weight * kp_v, rep_v, kp_v, G


def _corr(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


class _Scalar:
    """The third factor as applied: true (own s), wrong_sign (-s), shuffled (a draw from earlier scalars; 0 before any)."""

    def __init__(self, mode, seed):
        assert mode in ("true", "shuffled", "wrong_sign")
        self.mode, self.buf, self.rng = mode, collections.deque(maxlen=SHUF_BUF), random.Random(seed + SHUF_SEED_OFFSET)

    def __call__(self, s):
        if self.mode == "true":
            return s
        if self.mode == "wrong_sign":
            return -s
        out = self.rng.choice(list(self.buf)) if self.buf else 0.0
        self.buf.append(s)
        return out


# ------------------------------------------------------------------ the night
def tf_sleep(plastic, base_state, episodes, generic, *, grad_steps=240, budget_mult=BUDGET_MULT, lr=3e-4, sigma=0.03,
             kl_weight=1.0, seed=0, scalar="true", estimator="antithetic", n_pert=1, svrg=0, svrg_full=1, max_step=0.0,
             log=None, info=None):
    """The three-factor night (re-parameterises `plastic` first, see reparam_rank). episodes: (text, strength);
    generic: keep texts. Budget = budget_mult x 2 x grad_steps text forwards. Returns per-step (L_a, L_b, s, s_used);
    `info` gets the engagement record."""
    reparam_rank(plastic, seed)
    pc = adopt_tf(plastic)
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed + NOISE_SEED_OFFSET)
    third = _Scalar(scalar, seed)
    ids_of = pc.cortex.pipe.encode
    weights = [max(s, 1e-3) for _, s in episodes]
    ep_ids = [ids_of(t) for t, _ in episodes]
    gen_ids = [ids_of(g if isinstance(g, str) else g[0]) for g in generic]
    pc.enabled = False
    with torch.no_grad():
        keep_ref = [F.log_softmax(pc(g, base_state)[0], -1) for g in gen_ids]
    pc.enabled = True
    budget = int(budget_mult * 2 * grad_steps)
    n_steps, planned = plan_steps(budget, estimator, n_pert, svrg, svrg_full, len(ep_ids) + len(gen_ids))
    bounds = sorted({round(n_steps * k / N_WINDOWS) for k in range(N_WINDOWS + 1)})
    params = a_params(pc)
    P = sum(p.numel() for _, p in params)
    counter = {"n": 0}
    snap = lambda: {k: p.detach().clone() for k, p in params}
    A0 = snap()
    full = lambda: full_objective(pc, base_state, ep_ids, gen_ids, keep_ref, kl_weight, grad=True)
    L0, rep0, kp0, G0 = full()
    win_G, win_A, cos_w, ests_cos = G0, A0, [], []
    ema, skipped, losses, own, used = None, 0, [], [], []
    anchor, g_full = None, None
    full_texts = [(i, "replay", None, 1.0 / len(ep_ids)) for i in ep_ids] + \
                 [(g, "keep", r, kl_weight / len(gen_ids)) for g, r in zip(gen_ids, keep_ref)]

    def swap(vals):
        with torch.no_grad():
            for k, p in params:
                p.copy_(vals[k])

    for t in range(n_steps):
        if t in bounds and t > 0:
            A_now = snap()
            cos_w.append(_flat_cos({k: -(A_now[k] - win_A[k]) for k in A_now}, win_G))
            _, _, _, win_G = full()
            win_A = A_now
        if svrg and t % svrg == 0:
            anchor = snap()
            g_full = {k: torch.zeros_like(p) for k, p in params}
            for _ in range(svrg_full):
                nz = [draw_noise(pc, len(ids), gen) for ids, *_ in full_texts]
                s, N, _, _ = zo_sample(pc, full_texts, base_state, sigma, nz, estimator, counter)
                s = third(s)
                for k in N:
                    g_full[k] += s * N[k] / svrg_full
        j = rng.choices(range(len(episodes)), weights=weights)[0]
        g = rng.randrange(len(gen_ids))
        texts = [(ep_ids[j], "replay", None, 1.0), (gen_ids[g], "keep", keep_ref[g], kl_weight)]
        est = {k: torch.zeros_like(p) for k, p in params}
        for _ in range(n_pert):
            nz = [draw_noise(pc, len(ids), gen) for ids, *_ in texts]
            s, N, La, Lb = zo_sample(pc, texts, base_state, sigma, nz, estimator, counter)
            su = third(s)
            own.append(s)
            used.append(su)
            losses.append((La, Lb, s, su))
            for k in N:
                est[k] += su * N[k] / n_pert
            if svrg:
                cur = snap()
                swap(anchor)
                try:
                    s2, N2, _, _ = zo_sample(pc, texts, base_state, sigma, nz, estimator, counter)
                finally:
                    swap(cur)
                s2 = third(s2)
                for k in N2:
                    est[k] -= s2 * N2[k] / n_pert
        if svrg:
            for k in est:
                est[k] += g_full[k]
        ests_cos.append(_flat_cos({k: -v for k, v in est.items()}, {k: -v for k, v in win_G.items()}))
        n = _norm(est)
        if not math.isfinite(n) or n == 0:
            skipped += 1
            continue
        ref = ema if ema else n
        ema = n if ema is None else EMA * ema + (1 - EMA) * n
        scale = lr * math.sqrt(P) / ref
        if max_step and n * scale > max_step:
            scale = max_step / n
        with torch.no_grad():
            for k, p in params:
                p.sub_(scale * est[k])
        if log and t % 200 == 0:
            log(f"tf step {t}/{n_steps}: L {La:.4f}/{Lb:.4f} s {s:.4f} used {su:.4f} forwards {counter['n']}")
    A_end = snap()
    L1, rep1, kp1, _ = full_objective(pc, base_state, ep_ids, gen_ids, keep_ref, kl_weight, grad=False)
    if n_steps > 0:
        cos_w.append(_flat_cos({k: -(A_end[k] - win_A[k]) for k in A_end}, win_G))
    if info is not None:
        info.update(kind="three_factor", scalar=scalar, estimator=estimator, n_pert=n_pert, svrg=svrg,
                    svrg_full=svrg_full if svrg else None, sigma=sigma, lr=lr, max_step=max_step,
                    parameterisation="rank: A from 0 learned, B fixed random N(0,1/r)", n_params=P,
                    budget=budget, budget_mult=budget_mult, grad_forwards=2 * grad_steps,
                    forwards_used=counter["n"], forwards_planned=planned, steps_run=n_steps, skipped_steps=skipped,
                    objective_start=L0, objective_end=L1, replay_start=rep0, replay_end=rep1, keep_start=kp0,
                    keep_end=kp1, cos_windows=cos_w, cos_window=_mean(cos_w) if cos_w else None,
                    cos_night=_flat_cos({k: -(A_end[k] - A0[k]) for k in A0}, G0),
                    cos_step_mean=_mean(ests_cos) if ests_cos else None,
                    scalar_corr=_corr(own, used), scalar_abs_mean=_mean(abs(x) for x in own) if own else None,
                    used_abs_mean=_mean(abs(x) for x in used) if used else None)
    return losses


def gradient_rank_sleep(plastic, base_state, episodes, generic, *, sleep, steps, lr, kl_weight, seed, log=None, info=None):
    """Report-only arm: brain.sleep.sleep (Adam, backprop) on the three-factor re-parameterisation (A from 0, B fixed)."""
    reparam_rank(plastic, seed)
    ce0 = replay_ce(plastic, base_state, episodes)
    losses = sleep(plastic, base_state, episodes, generic, steps=steps, lr=lr, kl_weight=kl_weight, seed=seed, log=log)
    if info is not None:
        ce1 = replay_ce(plastic, base_state, episodes)
        info.update(_loss_engagement("gradient_rank", losses), loss_start=ce0, loss_end=ce1, replay_end=ce1,
                    parameterisation="rank: A from 0, B fixed",
                    forwards=2 * steps, backwards=steps)
    return losses


# ------------------------------------------------------------------ graded read-out (hippocampus off)
def graded(pc, state, facts):
    """Per fact: log-probability of the answer word (" " + its surface form in the telling) right after
    "User: {q}\\n\\nAssistant:", summed over its tokens, and the rank of its first token (1 = top). Changes nothing."""
    pipe = pc.cortex.pipe
    items = []
    with torch.no_grad():
        for s, q, k in facts:
            i = s.lower().find(k)
            surf = s[i:i + len(k)] if i >= 0 else k
            prompt = f"User: {q}\n\nAssistant:"
            ids = pipe.encode(prompt + " " + surf)
            a = next(j for j in range(len(ids)) if len(pipe.decode(ids[: j + 1])) > len(prompt))
            if a == 0:
                raise ValueError(f"answer at position 0 for {q!r}")
            lg = F.log_softmax(pc(ids, state)[0].float(), -1)
            lp = sum(float(lg[j - 1, ids[j]]) for j in range(a, len(ids)))
            first = lg[a - 1]
            rank = 1 + int((first > first[ids[a]]).sum())
            items.append({"key": k, "logprob": lp, "rank": rank})
    ranks = sorted(it["rank"] for it in items)
    n = len(ranks)
    med = (ranks[n // 2] if n % 2 else (ranks[n // 2 - 1] + ranks[n // 2]) / 2) if n else None
    return {"answer_logprob": _mean(it["logprob"] for it in items), "answer_rank_median": med,
            "answer_log10_rank": _mean(math.log10(it["rank"]) for it in items),
            "answer_top10": sum(it["rank"] <= 10 for it in items), "graded_items": items}


# ------------------------------------------------------------------ registered rules (pure; tested on synthetic rows)
def dev_bar(ns_recall, grad_recall):
    """The recall bar ("the floor"), the same in dev and in the verdict: max(0.5 x gradient, no_sleep + 2)."""
    return max(0.5 * grad_recall, ns_recall + RECALL_PLUS)


def killed(e):
    """The note's kill rule on one night's engagement: replay CE at the end not under 1.0, or mean window cosine to the
    true gradient under 0.1 (non-finite counts as killed). Returns the reasons (empty = alive)."""
    why = []
    if not (_finite(e.get("replay_end")) and e["replay_end"] < KILL_REPLAY):
        why.append(f"replay_end {e.get('replay_end')} not under {KILL_REPLAY}")
    if not (_finite(e.get("cos_window")) and e["cos_window"] >= KILL_COS):
        why.append(f"cos_window {e.get('cos_window')} under {KILL_COS}")
    return why


def _budget_ok(e):
    b, u = e.get("budget"), e.get("forwards_used")
    return (_finite(b) and _finite(u) and e.get("budget_mult") == BUDGET_MULT and b == BUDGET_MULT * e.get("grad_forwards", -1)
            and BUDGET_USE_MIN * b <= u <= b)


def _tf_engaged(e, norm):
    return (_finite(norm) and norm > 0 and _finite(e.get("objective_start")) and _finite(e.get("objective_end"))
            and e["objective_end"] < e["objective_start"] and _budget_ok(e) and e.get("scalar") == "true"
            and _finite(e.get("scalar_corr")) and e["scalar_corr"] >= CORR_TRUE_MIN)


def dev_pick(ref, rows):
    """ref: {"no_sleep": {recall}, "gradient": {recall, generic_kl, heldout_kl_dev}} on the dev seed (gradient_rank is
    reported, never gates); rows: [{lr, sigma, recall, generic_kl, heldout_kl_dev, adapter_norm, answer_logprob,
    engagement}] of the three_factor arm. Order: non-finite ref or gradient recall < 2 -> UNDEFINED-DEV; every cell
    killed (killed()) -> KILL-DEV; qualify = alive, finite, recall >= bar, generic and held-out KL <= 1.25 x gradient's and
    <= 1, engaged (norm > 0, fixed-order objective end < start, budget used, own scalar); none -> NEGATIVE-DEV. Pick max
    recall; ties: higher answer_logprob, lower heldout_kl_dev, lower generic_kl, lower lr, lower sigma."""
    need = [ref["no_sleep"]["recall"], ref["gradient"]["recall"], ref["gradient"]["generic_kl"],
            ref["gradient"]["heldout_kl_dev"]]
    if not all(_finite(x) for x in need):
        return {"verdict": "UNDEFINED-DEV", "why": "non-finite ref", "pick": None, "n_rows": len(rows)}
    if ref["gradient"]["recall"] < GRAD_MIN:
        return {"verdict": "UNDEFINED-DEV", "why": f"dev gradient recall {ref['gradient']['recall']} < {GRAD_MIN}",
                "pick": None, "n_rows": len(rows)}
    floor = dev_bar(ref["no_sleep"]["recall"], ref["gradient"]["recall"])
    cap_g, cap_h = DRIFT_MULT * ref["gradient"]["generic_kl"], DRIFT_MULT * ref["gradient"]["heldout_kl_dev"]
    keys = ("recall", "generic_kl", "heldout_kl_dev", "adapter_norm", "answer_logprob")
    out = {"floor": floor, "cap_generic": cap_g, "cap_heldout_dev": cap_h, "n_rows": len(rows),
           "rows": [{**{k: r.get(k) for k in ("lr", "sigma") + keys},
                     **{k: (r.get("engagement") or {}).get(k) for k in ("replay_end", "cos_window", "cos_night",
                                                                         "objective_start", "objective_end")},
                     "killed": killed(r.get("engagement") or {})} for r in rows]}
    alive = [r for r in rows if not killed(r.get("engagement") or {})]
    out["n_alive"] = len(alive)
    if not alive:
        return {**out, "verdict": "KILL-DEV", "pick": None}
    ok = [r for r in alive if all(_finite(r.get(k)) for k in keys)
          and r["recall"] >= floor and r["generic_kl"] <= cap_g and r["heldout_kl_dev"] <= cap_h
          and r["generic_kl"] <= KL_NIGHT_MAX and r["heldout_kl_dev"] <= KL_NIGHT_MAX
          and _tf_engaged(r.get("engagement") or {}, r["adapter_norm"])]
    out["n_qualify"] = len(ok)
    if not ok:
        return {**out, "verdict": "NEGATIVE-DEV", "pick": None}
    best = sorted(ok, key=lambda r: (-r["recall"], -r["answer_logprob"], r["heldout_kl_dev"], r["generic_kl"],
                                     r["lr"], r["sigma"]))[0]
    return {**out, "verdict": "PICKED", "pick": {"lr": best["lr"], "sigma": best["sigma"]}}


def _eng(recs, arm, s):
    return recs[arm][s].get("engagement") or {}


def _falls(e, head, tail):
    hd, tl = e.get(head), e.get(tail)
    return _finite(hd) and _finite(tl) and tl < hd


def engagement(recs, seeds=SEEDS):
    """{"core", "controls", "report"}: why an arm's night was not what its role needs, per seed. core: no_sleep norm 0;
    gradient norm > 0 and fixed-order replay CE end < start; three_factor norm > 0, fixed-order objective end < start,
    the budget used (>= 90%, never above, budget = 30 x the gradient forwards), its own scalar (corr >= 0.99). controls:
    tf_shuffled norm > 0 and >= 0.5 x three_factor's, budget used, |corr(used, own)| <= 0.2; tf_wrong_sign norm > 0 and
    >= 0.5 x three_factor's, budget used, corr <= -0.99, objective end > start (ascent: the sign is live). report:
    gradient_rank norm > 0 and replay CE end < start."""
    core, ctl, rep = [], [], []
    for s in seeds:
        n = lambda arm: recs[arm][s].get("adapter_norm")
        if n("no_sleep") != 0:
            core.append(f"no_sleep s{s}: adapter_norm {n('no_sleep')} != 0")
        if not (_finite(n("gradient")) and n("gradient") > 0 and _falls(_eng(recs, "gradient", s), "loss_start", "loss_end")):
            core.append(f"gradient s{s} not engaged: norm {n('gradient')}, loss {_eng(recs, 'gradient', s)}")
        e = _eng(recs, "three_factor", s)
        if not _tf_engaged(e, n("three_factor")):
            core.append(f"three_factor s{s} not engaged: norm {n('three_factor')}, objective {e.get('objective_start')}"
                        f"->{e.get('objective_end')}, forwards {e.get('forwards_used')}/{e.get('budget')}, scalar "
                        f"{e.get('scalar')} corr {e.get('scalar_corr')}")
        v = n("three_factor")
        e = _eng(recs, "tf_shuffled", s)
        c = e.get("scalar_corr")
        if not (_finite(n("tf_shuffled")) and n("tf_shuffled") > 0 and _finite(v) and n("tf_shuffled") >= ENGAGE_NORM_MIN * v
                and _budget_ok(e) and e.get("scalar") == "shuffled" and _finite(c) and abs(c) <= CORR_SHUF_MAX):
            ctl.append(f"tf_shuffled s{s} not engaged: norm {n('tf_shuffled')} vs {v}, forwards "
                       f"{e.get('forwards_used')}/{e.get('budget')}, corr {c}")
        e = _eng(recs, "tf_wrong_sign", s)
        c = e.get("scalar_corr")
        if not (_finite(n("tf_wrong_sign")) and n("tf_wrong_sign") > 0 and _finite(v)
                and n("tf_wrong_sign") >= ENGAGE_NORM_MIN * v and _budget_ok(e) and e.get("scalar") == "wrong_sign"
                and _finite(c) and c <= -CORR_TRUE_MIN and _falls(e, "objective_end", "objective_start")):
            ctl.append(f"tf_wrong_sign s{s} not engaged: norm {n('tf_wrong_sign')} vs {v}, objective "
                       f"{e.get('objective_start')}->{e.get('objective_end')} (must rise), corr {c}")
        if not (_finite(n("gradient_rank")) and n("gradient_rank") > 0
                and _falls(_eng(recs, "gradient_rank", s), "loss_start", "loss_end")):
            rep.append(f"gradient_rank s{s} not engaged: norm {n('gradient_rank')}")
    return {"core": core, "controls": ctl, "report": rep}


def verdict(recs):
    """Per model. recs: {arm: {seed: {recall, false_recall, generic_kl, heldout_kl (VERDICT half), adapter_norm,
    answer_logprob, ..., engagement}}}, every arm of ARMS on exactly SEEDS (else ValueError). Order:
    1. UNDEFINED: a gated value of a gated arm non-finite, or a three_factor norm non-finite; mean gradient recall < 2;
       a core engagement gate fails.
    2. KILL: mean three_factor replay_end not under 1.0, or mean cos_window under 0.1 (the note's kill).
    3. core (a) three_factor recall >= bar; (c) generic and held-out KL each <= 1.25 x gradient's, no night above KL 1;
       (e) false recall <= gradient + 1. A failing core clause -> NEGATIVE (reading from gradient_rank).
    4. a control engagement gate fails -> UNDEFINED.
    5. (b) three_factor > tf_shuffled and > tf_wrong_sign each on >= 5 seeds, each control's mean recall <= no_sleep + 1;
       fails -> NEGATIVE.
    6. GO-TF. The graded read-out (answer log-probability, rank) is reported beside recall; it gates nothing."""
    for arm in ARMS:
        if arm not in recs or set(recs[arm]) != set(SEEDS):
            raise ValueError(f"verdict needs every arm on exactly the seeds {SEEDS}: {arm} has {sorted(recs.get(arm, {}))}")
    seeds = list(SEEDS)
    why = [f"{arm} s{s} {k} non-finite" for arm in GATED_ARMS for s in seeds for k in GATED
           if not _finite(recs[arm][s].get(k))]
    why += [f"three_factor s{s} adapter_norm non-finite" for s in seeds if not _finite(recs["three_factor"][s].get("adapter_norm"))]
    if why:
        return {"n_seeds": len(seeds), "verdict": "UNDEFINED", "why": why}
    m = lambda arm, k: _mean(recs[arm][s][k] for s in seeds)
    me = lambda arm, k: _mean(_eng(recs, arm, s).get(k) for s in seeds) if all(_finite(_eng(recs, arm, s).get(k)) for s in seeds) else None
    G, NS = m("gradient", "recall"), m("no_sleep", "recall")
    bar = dev_bar(NS, G)
    cg, ch = DRIFT_MULT * m("gradient", "generic_kl"), DRIFT_MULT * m("gradient", "heldout_kl")
    w_sh = sum(recs["three_factor"][s]["recall"] > recs["tf_shuffled"][s]["recall"] for s in seeds)
    w_ws = sum(recs["three_factor"][s]["recall"] > recs["tf_wrong_sign"][s]["recall"] for s in seeds)
    kill = {"replay_end": me("three_factor", "replay_end"), "cos_window": me("three_factor", "cos_window")}
    cl = {"a": m("three_factor", "recall") >= bar,
          "b_shuffled": w_sh >= WIN_SEEDS and m("tf_shuffled", "recall") <= NS + CTRL_PLUS,
          "b_wrong_sign": w_ws >= WIN_SEEDS and m("tf_wrong_sign", "recall") <= NS + CTRL_PLUS,
          "c_generic": m("three_factor", "generic_kl") <= cg,
          "c_heldout": m("three_factor", "heldout_kl") <= ch,
          "c_night": all(recs["three_factor"][s]["generic_kl"] <= KL_NIGHT_MAX
                         and recs["three_factor"][s]["heldout_kl"] <= KL_NIGHT_MAX for s in seeds),
          "e": m("three_factor", "false_recall") <= m("gradient", "false_recall") + FALSE_PLUS,
          "kill_replay": _finite(kill["replay_end"]) and kill["replay_end"] < KILL_REPLAY,
          "kill_cos": _finite(kill["cos_window"]) and kill["cos_window"] >= KILL_COS}
    eng = engagement(recs, seeds)
    graded_keys = ("answer_logprob", "answer_log10_rank", "answer_top10")
    out = {"n_seeds": len(seeds), "bar": bar, "cap_generic": cg, "cap_heldout": ch, "tf_beats_shuffled": w_sh,
           "tf_beats_wrong_sign": w_ws, "kill": kill, "clauses": cl, "engagement": eng,
           "means": {a: {k: (m(a, k) if all(_finite(recs[a][s].get(k)) for s in seeds) else None)
                         for k in GATED + ("adapter_norm",) + graded_keys} for a in ARMS}}
    why = []
    if G < GRAD_MIN:
        why.append(f"gradient mean recall {G} < {GRAD_MIN}")
    why += eng["core"]
    if why:
        return {**out, "verdict": "UNDEFINED", "why": why}
    if not (cl["kill_replay"] and cl["kill_cos"]):
        return {**out, "verdict": "KILL", "reading": f"killed by the note's rule: replay_end {kill['replay_end']} "
                f"(needs < {KILL_REPLAY}), cos_window {kill['cos_window']} (needs >= {KILL_COS})"}
    if not all(cl[k] for k in ("a", "c_generic", "c_heldout", "c_night", "e")):
        return {**out, "verdict": "NEGATIVE", "reading": negative_reading(recs, seeds, bar, cg, ch, eng)}
    if eng["controls"]:
        return {**out, "verdict": "UNDEFINED", "why": eng["controls"]}
    if not (cl["b_shuffled"] and cl["b_wrong_sign"]):
        return {**out, "verdict": "NEGATIVE", "reading": "contrast failed: a scalar unrelated to the noise (or of the "
                "wrong sign) does as well, or a control moved recall; not credited to the third factor"}
    return {**out, "verdict": "GO-TF"}


def negative_reading(recs, seeds, bar, cg, ch, eng):
    """gradient_rank (report only) passes (a) and (c) -> rule-limit (the same parameterisation stores by gradient; the
    perturbation estimate does not); fails -> parameterisation-limit (A only with a fixed random B does not store even
    by gradient; next: learn B or the key-value write); not engaged or non-finite -> undetermined."""
    gr = [recs["gradient_rank"][s] for s in seeds]
    if eng["report"] or not all(_finite(r.get(k)) for r in gr for k in GATED):
        return "undetermined: gradient_rank not engaged or non-finite"
    m = lambda k: _mean(r[k] for r in gr)
    if (m("recall") >= bar and m("generic_kl") <= cg and m("heldout_kl") <= ch
            and all(r["generic_kl"] <= KL_NIGHT_MAX and r["heldout_kl"] <= KL_NIGHT_MAX for r in gr)):
        return "rule-limit: gradient_rank passes (a) and (c) on the same parameterisation; the perturbation estimate fails"
    return "parameterisation-limit: gradient_rank fails (a) or (c) too; A-only rank units with a fixed random B do not store"


def combine(v01, v04):
    """Two-scale word: GO-TF (both); GO-TF-PARTIAL (exactly one, the other defined); UNDEFINED if either is UNDEFINED*/
    ERROR*/HOLD*/STOP*/missing; else NEGATIVE (NEGATIVE, NEGATIVE-DEV, KILL, KILL-DEV)."""
    ws = [v01, v04]
    if any(not isinstance(w, str) or w.startswith(("UNDEFINED", "ERROR", "HOLD", "STOP")) for w in ws):
        return "UNDEFINED"
    go = sum(w == "GO-TF" for w in ws)
    return "GO-TF" if go == 2 else "GO-TF-PARTIAL" if go == 1 else "NEGATIVE"


# ------------------------------------------------------------------ CLI
TF_KEYS = ("lr", "sigma", "estimator", "n_pert", "svrg", "svrg_full", "budget_mult", "max_step")


def cmd_pick(a):
    """Dev pick. Refuses (exit non-zero, no pick.json) unless the dir holds exactly the 3 ref units and one
    three_factor unit per (lr, sigma) cell, all seed 7, one model, one commit, the pinned prompts."""
    units = [u for u in _load_dir(a.dir, "*.json") if isinstance(u, dict) and "arm" in u and u.get("seed") == DEV_SEED]
    ref_u = [u for u in units if u["tag"] == a.ref_tag and u["arm"] in REF_ARMS]
    tf_u = [u for u in units if u["tag"] != a.ref_tag and u["arm"] == "three_factor"]
    bad = []
    if sorted(u["arm"] for u in ref_u) != sorted(REF_ARMS):
        bad.append(f"ref units {sorted(u['arm'] for u in ref_u)} != {sorted(REF_ARMS)}")
    cells = sorted((u.get("lr"), u.get("sigma")) for u in tf_u)
    if cells != sorted(itertools.product(DEV_LRS, DEV_SIGMAS)):
        bad.append(f"three_factor cells {cells} != the 9 registered (lr, sigma)")
    if bad:
        raise SystemExit("pick refused: " + "; ".join(bad))
    allu = ref_u + tf_u
    model, commit = _one(allu, "model_id", "pick"), _one(allu, "commit", "pick")
    _one(allu, "prompts_sha256", "pick")
    for k in TF_KEYS[2:]:
        _one(tf_u, k, "pick (three_factor cells)")
    if not commit:
        raise SystemExit("pick refused: units carry no commit")
    ref = {u["arm"]: u["result"] for u in ref_u}
    rows = [{"lr": u["lr"], "sigma": u["sigma"], **u["result"], "engagement": u.get("engagement") or {}} for u in tf_u]
    res = {**dev_pick(ref, rows), "model_id": model, "commit": commit, "prompts_sha256": ref_u[0].get("prompts_sha256"),
           "tf_config": {k: tf_u[0].get(k) for k in TF_KEYS[2:]},
           "ref": {arm: {k: r.get(k) for k in ("recall", "generic_kl", "heldout_kl_dev", "answer_logprob")}
                   for arm, r in ref.items()}}
    print(json.dumps(res, indent=1))
    if a.out:
        atomic_write(a.out, res)


def check_units(units, pick=None):
    """One instrument: one model, commit and prompts list; one TF config across the TF arms; the pick's when given."""
    model, commit = _one(units, "model_id", "score"), _one(units, "commit", "score")
    _one(units, "prompts_sha256", "score")
    loc = [u for u in units if u["arm"] in TF_ARMS]
    cfg = {k: _one(loc, k, "score (three-factor arms)") for k in TF_KEYS}
    if pick is not None:
        want = (pick["pick"]["lr"], pick["pick"]["sigma"], pick["model_id"], pick["commit"])
        got = (cfg["lr"], cfg["sigma"], model, commit)
        if got != want:
            raise SystemExit(f"units (lr, sigma, model, commit) {got} != pick {want}")
        for k, v in (pick.get("tf_config") or {}).items():
            if cfg.get(k) != v:
                raise SystemExit(f"units {k} {cfg.get(k)} != pick {v}")
    return {**cfg, "model_id": model, "commit": commit}


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
    res = {**verdict(recs), "instrument": inst}
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


def run_units(a, c, base, *, PROMPTS, GENERIC, build_arm_episodes, evaluate, sleep, keys_all, facts, PlasticCortex):
    """The per-(seed, arm) loop of `run` on a loaded cortex `c` (tests pass a stub cortex). Each arm is one atomic unit;
    existing units are skipped (resume)."""
    dev_p, ver_p = split_prompts(PROMPTS, a.prompts_sha)
    psha = prompts_sha(PROMPTS)
    if a.pick:
        p = read_pick(a.pick)
        if p.get("model_id") != _model_id(a.model):
            raise SystemExit(f"pick model {p.get('model_id')} != --model {_model_id(a.model)}")
        a.lr, a.sigma = p["pick"]["lr"], p["pick"]["sigma"]
        for k, v in (p.get("tf_config") or {}).items():
            if getattr(a, k) != v:
                raise SystemExit(f"--{k.replace('_', '-')} {getattr(a, k)} != pick {v}")
    prov = provenance(a.device, "float32")
    if a.pick and p.get("commit") != prov["commit"]:
        raise SystemExit(f"pick commit {p.get('commit')} != this code's commit {prov['commit']}")
    todo = [arm for arm in a.arm_list if not os.path.exists(unit_path(a.out_dir, a.tag, a.seed, arm))]
    eps = build_arm_episodes(a.seed)["sleep_true"]
    base_unit = {"tag": a.tag, "seed": a.seed, "model": a.model, "model_id": _model_id(a.model), "steps": a.steps,
                 "prompts_sha256": psha, "pick": a.pick, **prov}
    for arm in todo:
        t1 = time.time()
        torch.manual_seed(a.seed)
        pc = PlasticCortex(c)
        eng = {}
        lg = lambda msg, arm=arm: print(arm, msg, flush=True)
        if arm == "no_sleep":
            eng = {"kind": "none", "steps_run": 0}
        elif arm == "gradient":
            ce0 = replay_ce(pc, base, eps)
            losses = sleep(pc, base, eps, GENERIC, steps=a.steps, lr=a.grad_lr, kl_weight=a.kl_weight, seed=a.seed, log=lg)
            eng = {**_loss_engagement("gradient", losses), "loss_start": ce0, "loss_end": replay_ce(pc, base, eps),
                   "forwards": 2 * a.steps, "backwards": a.steps}
            eng["replay_end"] = eng["loss_end"]
        elif arm == "gradient_rank":
            gradient_rank_sleep(pc, base, eps, GENERIC, sleep=sleep, steps=a.steps, lr=a.grad_rank_lr, kl_weight=a.kl_weight,
                                seed=a.seed, log=lg, info=eng)
        else:
            tf_sleep(pc, base, eps, GENERIC, grad_steps=a.steps, budget_mult=a.budget_mult, lr=a.lr, sigma=a.sigma,
                     kl_weight=a.kl_weight, seed=a.seed, scalar=SCALAR_OF[arm], estimator=a.estimator, n_pert=a.n_pert,
                     svrg=a.svrg, svrg_full=a.svrg_full, max_step=a.max_step, log=lg, info=eng)
        r = evaluate(pc, base, keys_all)
        r.update(graded(pc, base, facts))
        # dev/verdict firewall: the VERDICT half is never computed on the dev seed
        r["heldout_kl"] = None if a.seed == DEV_SEED else heldout_drift(pc, base, ver_p)
        r["heldout_kl_dev"] = heldout_drift(pc, base, dev_p)
        r["adapter_norm"] = pc.adapter_norm()
        r["factor_norms"] = pc.factor_norms()
        eng["adapter_norm"] = r["adapter_norm"]
        tf = arm in TF_ARMS
        unit = {**base_unit, "arm": arm, **{k: (getattr(a, k) if tf else None) for k in TF_KEYS},
                "grad_lr": {"gradient": a.grad_lr, "gradient_rank": a.grad_rank_lr}.get(arm), "scalar": SCALAR_OF.get(arm),
                "elapsed_s": time.time() - t1, "engagement": eng, "result": r}
        atomic_write(unit_path(a.out_dir, a.tag, a.seed, arm), unit)
        print(arm, {k: v for k, v in r.items() if not k.endswith("_items")}, flush=True)
    return todo


def cmd_run(a):
    guard()
    want = [x.strip() for x in a.arms.split(",") if x.strip()]
    if not want or [x for x in want if x not in ARMS]:
        raise SystemExit(f"unknown arms; choose from {ARMS}")
    a.arm_list = want
    if not [arm for arm in want if not os.path.exists(unit_path(a.out_dir, a.tag, a.seed, arm))]:
        print("all units done", flush=True)
        return
    from brain.brain import BIRTH
    from brain.cortex import Cortex
    from brain.experiments.facts import FACTS
    from brain.experiments.mix_night_screen import PROMPTS
    from brain.experiments.sleep_probe import GENERIC, build_arm_episodes, evaluate
    from brain.plastic_cortex import PlasticCortex as PC
    from brain.sleep import sleep
    t0 = time.time()
    c = Cortex(model_path=a.model, device=a.device)
    _, base = c.read(BIRTH, None)
    base = [s.float() for s in base]
    todo = run_units(a, c, base, PROMPTS=PROMPTS, GENERIC=GENERIC, build_arm_episodes=build_arm_episodes,
                     evaluate=evaluate, sleep=sleep, keys_all=[k for _, _, k in FACTS], facts=FACTS, PlasticCortex=PC)
    print(f"done {len(todo)} units in {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--arms", default=",".join(ARMS))
    r.add_argument("--lr", type=float, default=3e-4)
    r.add_argument("--sigma", type=float, default=0.03)
    r.add_argument("--estimator", default="antithetic", choices=["antithetic", "forward"])
    r.add_argument("--n-pert", type=int, default=1)
    r.add_argument("--svrg", type=int, default=0, help="anchor refresh every q steps (MeZO-SVRG form); 0 = off")
    r.add_argument("--svrg-full", type=int, default=1, help="perturbations per full-batch anchor estimate")
    r.add_argument("--budget-mult", type=float, default=BUDGET_MULT)
    r.add_argument("--max-step", type=float, default=0.0, help="global step-norm clip; 0 = off")
    r.add_argument("--kl-weight", type=float, default=1.0)
    r.add_argument("--grad-lr", type=float, default=5e-4)
    r.add_argument("--grad-rank-lr", type=float, default=5e-5,
                   help="gradient_rank's Adam lr: fixed B ~ N(0,1/8) is ~3x the gradient arm's learned B, so 5e-4/10")
    r.add_argument("--steps", type=int, default=240, help="the gradient arm's steps (sets the rule's budget)")
    r.add_argument("--device", default="cpu")
    r.add_argument("--tag", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--pick", default=None, help="dev pick.json: lr and sigma from it; model, commit, config must match")
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
