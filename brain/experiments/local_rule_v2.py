# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-27-local-sleep-01b-read.md
# Expected-fail: recall may fall under the bar: the protected subspace holds the chat frame and common directions that the
#   told-fact texts share, so learning rides only on what is new in them; still no credit through the WKV state. A
#   negative is a verdict on this projection method at this scale, not on local learning.
"""Local rule v2: the local delta rule of brain/experiments/local_sleep.py, with its adapter plasticity confined to the
input directions that ordinary (keep-bank) text does NOT use in the pretrained cortex. Drift is attacked at the source.

Why (the record, 2026-11-27): under the drift cap the v1 rule is nearly inert at 0.1B (recall 0.17/12, NEGATIVE); at
the lr where it learns, it drifts 3.5x the gradient arm (0.1B lr 0.1-0.3: recall 2 at KL 0.11; 0.4B lr 0.3: recall 4 at
KL 0.089 vs cap 0.040). The SHY cut after the night (g 0.6) brings 0.4B drift to 0.035 but the gradient sleep given the
same cut drops to 0.017, so the cut is not specific. The keep term of v1 (a feedback-carried KL error and a B decay)
competes with learning step by step; it is a penalty, so less drift costs learning. Here nothing competes: the adapter
delta at layer i on an input x is (x A) B, so if every column of A is orthogonal to the span of the keep-bank inputs of
that layer, the delta on keep text is zero whatever B learns, and (layer by layer, by induction) the keep text's later
inputs are unchanged too.

Rule (per layer, per adapter path, local to the synapse):
- Before the night, the pretrained cortex reads the keep bank once (adapters at zero); each layer keeps the principal
  subspace U of its OWN pre-synaptic activity (k_in for the key path, the hidden h for the value path), uncentred, the
  smallest rank holding a fraction `tau` of the energy.
- A starts and stays in the orthogonal complement: A <- (I - U U^T) A at the start and after every local step. B is
  unchanged (its pre-synaptic activity is x A, already protected). The v1 rule, feedback, KP copies, clip, B decay and
  keep-error step are otherwise IDENTICAL (subspace="none" reproduces local_sleep.local_sleep exactly; tested).
- Biology stand-in: a decorrelating (anti-Hebbian lateral) circuit trained on ordinary activity converges to this
  projector (Oja/Foldiak subspace learning); the SVD here is the declared stand-in for that converged circuit. Method
  family: orthogonal weight modification / gradient projection memory (Zeng et al. 2019; Saha et al. 2021), here with a
  local rule rather than backprop gradients. No host English: the keep bank and battery are the existing ones.
- Shams (rank-matched per layer and path): subspace="random" projects out a random orthonormal subspace (same loss of
  free directions, no activity at all); subspace="tokens" projects out the top principal subspace of the pretrained
  cortex's own activity on RANDOM TOKEN sequences of the keep bank's lengths (real activity, including the shared mean
  direction every input carries, but none of the keep text's content). The random sham alone removes only ~m/D of A, so
  it is close to the lesion; the token sham asks whether any real-activity subspace of that rank does it.
- Declared: on keep text x A ~ 0 under the projection, so the v1 keep-error step moves B by ~0 and its A move lies in U
  and is projected out; it is kept (the v1 loop, unchanged) and its effective step norm is logged next to the replay
  step's (info["step_norms"]). The clip acts on the update before projection, so the applied step can be smaller.

Held-out drift: the keep bank (sleep_probe.GENERIC) is also the battery's drift text, so a projector fitted on it is
judged on texts it never saw: brain/experiments/mix_night_screen.PROMPTS (24, pinned by sha256), KL(on || off) over all
positions of the "User: ...\\n\\nAssistant:" frame of sleep_probe.ask. Split: even indices (12) are the DEV half (used
only by the seed-7 pick, `heldout_kl_dev`), odd indices (12) the VERDICT half (`heldout_kl`, gates the six seeds).

    POOL_NODE=1 python -m brain.experiments.local_rule_v2 run --model M.pth --seed 7 --arms no_sleep,gradient \\
        --out-dir D --tag ref
    POOL_NODE=1 python -m brain.experiments.local_rule_v2 run --model M.pth --seed 7 --arms v2 --lr 0.3 --tau 0.99 \\
        --out-dir D --tag lr0.3_tau0.99
    python -m brain.experiments.local_rule_v2 pick --dir D --out D/pick.json   # dev pick (seed 7): all 11 units required
    POOL_NODE=1 python -m brain.experiments.local_rule_v2 run --model M.pth --seed 42 --pick D/pick.json --out-dir S \\
        --tag six                                                    # lr, tau from the pick; model + commit must match
    python -m brain.experiments.local_rule_v2 score --dir S --tag six --pick D/pick.json   # six-seed verdict
Prereg: brain/research/2026-11-27-local-rule-v2-prereg.md (REGISTERED 2026-11-27 after pre-data sonnet review).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import math
import os
import random
import socket
import subprocess
import time

import torch
import torch.nn.functional as F

from brain.experiments.local_sleep import PRESETS, Feedback, _forward, adopt, commit_id, local_update

SEEDS = (42, 43, 44, 100, 101, 102)
DEV_SEED = 7
ARMS = ("no_sleep", "gradient", "v2", "v2_shuffled", "noproj", "sham", "sham_tok")
LOCAL_ARMS = ("v2", "v2_shuffled", "noproj", "sham", "sham_tok")      # one (lr, tau) across these
SUBSPACE_OF = {"v2": "keep", "v2_shuffled": "keep", "noproj": "none", "sham": "random", "sham_tok": "tokens"}
DEV_LRS = (0.1, 0.3, 1.0)
DEV_TAUS = (0.9, 0.99, 0.999)
DRIFT_MULT = 1.25
RECALL_PLUS = 2
SHUF_MAX = 1.0
FALSE_PLUS = 0.5
SHUF_SEEDS = 5
SHAM_SEED_OFFSET = 7919
LESION_RATIO = 1.5      # (f), (g): the control's mean held-out drift >= 1.5 x v2's ...
SPEC_SEEDS = 5          # ... and above v2's on >= 5 of 6 seeds
RANK_MIN = 4            # dev: a config protects >= 4 directions in every layer and path (more than the shared mean)
PROMPTS_SHA256 = "77e31aa1935c86bf6d6c04ea8e87009b73668a0196f3d831a81766c9d620d8d6"   # mix_night_screen.PROMPTS @ 4389b39
GATED = ("recall", "false_recall", "generic_kl", "heldout_kl")
CONTROL_ARMS = ("v2_shuffled", "noproj", "sham", "sham_tok")   # each must have had a live night before it can count
ENGAGE_NORM_MIN = 0.5   # a control's adapter norm >= 0.5 x that seed's v2 norm (no upper bound: see the prereg)
LOSS_WIN = 20           # replay loss head / tail = mean of the first / last 20 steps; a live night has tail < head


# ------------------------------------------------------------------ guard and provenance
def guard(env=None):
    env = os.environ if env is None else env
    if env.get("POOL_NODE") != "1" and env.get("BRAIN_ALLOW_W3090") != "1":
        raise SystemExit("local_rule_v2 run: refuses to run unless POOL_NODE=1 or BRAIN_ALLOW_W3090=1")


def provenance(device="cpu", dtype="float32") -> dict:
    c = commit_id()
    if c is None:
        try:
            c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                               cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or None
        except OSError:
            c = None
    p = {"host": socket.gethostname(), "commit": c, "device": str(device), "precision": str(dtype)}
    if os.environ.get("HOST", "").strip():
        p["host_env"] = os.environ["HOST"].strip()
    return p


def atomic_write(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + ".tmp", path)


def unit_path(out_dir, tag, seed, arm):
    return os.path.join(out_dir, f"{tag}_s{seed}_{arm}.json")


def prompts_sha(prompts):
    return hashlib.sha256(json.dumps(list(prompts)).encode()).hexdigest()


def split_prompts(prompts, sha=PROMPTS_SHA256):
    """(dev half, verdict half) of the pinned held-out prompts: even / odd indices. Refuses a changed list."""
    got = prompts_sha(prompts)
    if sha is not None and got != sha:
        raise SystemExit(f"held-out prompts sha256 {got} != pinned {sha}: the list changed; re-register before running")
    return list(prompts[0::2]), list(prompts[1::2])


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ------------------------------------------------------------------ the protected subspace
def basis(X, tau):
    """Orthonormal (D, m) basis of the uncentred principal subspace of the rows of X holding >= tau of the energy.
    tau >= 1: the whole numerical row space. Empty X or zero energy: m = 0."""
    D = X.shape[1]
    if X.numel() == 0:
        return X.new_zeros(D, 0)
    _, S, Vh = torch.linalg.svd(X.double(), full_matrices=False)
    e = S ** 2
    tot = float(e.sum())
    if tot <= 0:
        return X.new_zeros(D, 0)
    if tau >= 1.0:
        m = int((S > S[0] * 1e-6).sum())
    else:
        m = int(torch.searchsorted(torch.cumsum(e, 0) / tot, torch.tensor(tau, dtype=e.dtype)).item()) + 1
    m = min(m, len(S))
    return Vh[:m].T.to(X.dtype).contiguous()


def basis_top(X, m):
    """Orthonormal (D, <=m) basis: the top-m right singular vectors of the rows of X (uncentred)."""
    D = X.shape[1]
    if m == 0 or X.numel() == 0:
        return X.new_zeros(D, 0)
    _, S, Vh = torch.linalg.svd(X.double(), full_matrices=False)
    m = min(m, int((S > S[0] * 1e-6).sum()) if float(S[0]) > 0 else 0)
    return Vh[:m].T.to(X.dtype).contiguous()


def random_basis(D, m, gen, dtype=torch.float32, device="cpu"):
    if m == 0:
        return torch.zeros(D, 0, dtype=dtype, device=device)
    q, _ = torch.linalg.qr(torch.randn(D, m, generator=gen, dtype=torch.float64))
    return q[:, :m].to(dtype=dtype, device=device).contiguous()


def keep_activity(plastic, gen_ids, base_state):
    """Per layer, the pretrained cortex's own channel-mix pre-synaptic activity on the keep bank: (k_in rows, h rows).
    B is zeroed for the read (so this IS the pretrained cortex) and restored after."""
    pc = adopt(plastic)
    saved = {s: (pc.B_k[s].detach().clone(), pc.B_v[s].detach().clone()) for s in pc.B_k}
    acts = {i: ([], []) for i in pc.layers}
    was = pc.enabled
    try:
        with torch.no_grad():
            for s in pc.B_k:
                pc.B_k[s].zero_()
                pc.B_v[s].zero_()
        pc.enabled = True
        for ids in gen_ids:
            _forward(pc, ids, base_state)
            for i, k_in, _pre, h in pc.rec:
                acts[i][0].append(k_in)
                acts[i][1].append(h)
    finally:
        pc.rec = None
        pc.enabled = was
        with torch.no_grad():
            for s, (bk, bv) in saved.items():
                pc.B_k[s].copy_(bk)
                pc.B_v[s].copy_(bv)
    return {i: (torch.cat(a), torch.cat(b)) for i, (a, b) in acts.items()}


def token_ids_like(gen_ids, vocab, seed):
    """Random token sequences of the keep bank's lengths (ids 1..vocab-1), seeded: the token sham's input."""
    g = torch.Generator().manual_seed(seed + SHAM_SEED_OFFSET + 1)
    return [torch.randint(1, vocab, (len(ids),), generator=g).tolist() for ids in gen_ids]


def build_subspaces(plastic, gen_ids, base_state, tau, kind="keep", seed=0):
    """{layer: (U_k (C,m_k), U_v (hid,m_v))} and an info dict. kind "keep": the keep bank's principal subspace;
    "random": random orthonormal subspaces of the SAME ranks (sham); "tokens": the top subspace of the same ranks of
    the pretrained activity on random token sequences of the keep bank's lengths (sham); "none": {}."""
    if kind == "none":
        return {}, {"kind": "none"}
    acts = keep_activity(plastic, gen_ids, base_state)
    U = {i: (basis(Xk, tau), basis(Xh, tau)) for i, (Xk, Xh) in acts.items()}
    info = {"kind": kind, "tau": tau, "m_k": {str(i): u[0].shape[1] for i, u in U.items()},
            "m_v": {str(i): u[1].shape[1] for i, u in U.items()},
            "keep_tokens": int(next(iter(acts.values()))[0].shape[0]) if acts else 0}
    if kind == "random":
        g = torch.Generator().manual_seed(seed + SHAM_SEED_OFFSET)
        U = {i: (random_basis(uk.shape[0], uk.shape[1], g, uk.dtype, uk.device),
                 random_basis(uv.shape[0], uv.shape[1], g, uv.dtype, uv.device)) for i, (uk, uv) in U.items()}
    elif kind == "tokens":
        pc = adopt(plastic)
        was = pc.enabled
        pc.enabled = False
        with torch.no_grad():
            vocab = int(pc(gen_ids[0], base_state)[0].shape[-1])
        pc.enabled = was
        tacts = keep_activity(plastic, token_ids_like(gen_ids, vocab, seed), base_state)
        U = {i: (basis_top(tacts[i][0], uk.shape[1]), basis_top(tacts[i][1], uv.shape[1])) for i, (uk, uv) in U.items()}
        info["m_k_used"] = {str(i): u[0].shape[1] for i, u in U.items()}
        info["m_v_used"] = {str(i): u[1].shape[1] for i, u in U.items()}
    elif kind != "keep":
        raise ValueError(kind)
    info["rank_min"] = min([min(u[0].shape[1], u[1].shape[1]) for u in U.values()], default=0)
    # energy of the keep activity that stays inside each subspace (1.0 = the adapter is blind to the keep bank)
    info["keep_energy_inside"] = {str(i): [float((acts[i][0] @ uk).pow(2).sum() / acts[i][0].pow(2).sum().clamp_min(1e-30)),
                                           float((acts[i][1] @ uv).pow(2).sum() / acts[i][1].pow(2).sum().clamp_min(1e-30))]
                                  for i, (uk, uv) in U.items()}
    return U, info


def project_adapters(pc, U):
    """A <- (I - U U^T) A for every protected layer (both paths). No-op for U == {}."""
    with torch.no_grad():
        for i, (uk, uv) in U.items():
            s = str(i)
            if uk.shape[1]:
                pc.A_k[s].sub_(uk @ (uk.T @ pc.A_k[s]))
            if uv.shape[1]:
                pc.A_v[s].sub_(uv @ (uv.T @ pc.A_v[s]))


# ------------------------------------------------------------------ the night
def v2_sleep(plastic, base_state, episodes, generic, *, steps=240, lr=0.3, kl_weight=1.0, decay=1e-3, seed=0, mode="kp",
             clip=1.0, feedback="transport", tau=0.99, subspace="keep", log=None, info=None):
    """local_sleep.local_sleep with A confined to the complement of the keep subspace. subspace "none" is the v1 loop
    step for step (same RNG draws, same updates). `info` (a dict) receives the subspace description."""
    pc = adopt(plastic)
    rng = random.Random(seed)
    fb = Feedback(pc, mode, seed, **PRESETS[feedback])
    ids_of = pc.cortex.pipe.encode
    weights = [max(s, 1e-3) for _, s in episodes]
    gen_ids = [ids_of(g if isinstance(g, str) else g[0]) for g in generic]
    U, sinfo = build_subspaces(plastic, gen_ids, base_state, tau, subspace, seed)
    if info is not None:
        info.update(sinfo)
    project_adapters(pc, U)
    pc.enabled = False
    with torch.no_grad():
        ref = [F.softmax(pc(g, base_state)[0], -1) for g in gen_ids]
    pc.enabled = True
    losses, norms = [], {"replay": 0.0, "keep": 0.0}
    for step in range(steps):
        text, _ = rng.choices(episodes, weights=weights)[0]
        ids = ids_of(text)
        logits = _forward(pc, ids, base_state)
        p = F.softmax(logits, -1)
        tgt = F.one_hot(torch.tensor(ids[1:], device=p.device), p.shape[1]).to(p.dtype)
        err = torch.cat((p[:-1] - tgt, torch.zeros_like(p[:1])))
        ce = float(F.cross_entropy(logits[:-1], torch.tensor(ids[1:], device=p.device)))
        snap = _snap(pc) if info is not None else None
        local_update(pc, fb, err, lr=lr, decay=decay, clip=clip)
        project_adapters(pc, U)
        if snap is not None:
            norms["replay"] += _moved(pc, snap) / steps
        j = rng.randrange(len(gen_ids))
        gl = _forward(pc, gen_ids[j], base_state)
        gp = F.softmax(gl, -1)
        snap = _snap(pc) if info is not None else None
        local_update(pc, fb, kl_weight * (gp - ref[j]), lr=lr, clip=clip)
        project_adapters(pc, U)
        if snap is not None:
            norms["keep"] += _moved(pc, snap) / steps
        kl = float(F.kl_div(F.log_softmax(gl, -1), ref[j], reduction="batchmean"))
        losses.append((ce, kl))
        if log and step % 20 == 0:
            log(f"v2 sleep step {step}: replay {ce:.3f} keep {kl:.4f}")
    pc.rec = None
    if info is not None:
        info["step_norms"] = norms     # mean applied (post-projection) step norm of the two local updates per step
    return losses


def _snap(pc):
    return [t.detach().clone() for d in (pc.A_k, pc.A_v, pc.B_k, pc.B_v) for t in d.values()]


def _moved(pc, snap):
    now = [t.detach() for d in (pc.A_k, pc.A_v, pc.B_k, pc.B_v) for t in d.values()]
    return float(sum((a - b).pow(2).sum() for a, b in zip(now, snap)) ** 0.5)


def heldout_drift(pc, state, prompts):
    """Mean KL(on || off) over all positions of each held-out prompt in the ask() frame."""
    tot = 0.0
    with torch.no_grad():
        for q in prompts:
            ids = pc.cortex.pipe.encode(f"User: {q}\n\nAssistant:")
            on = F.log_softmax(pc(ids, state)[0], -1)
            was, pc.enabled = pc.enabled, False
            off = F.log_softmax(pc(ids, state)[0], -1)
            pc.enabled = was
            tot += float(F.kl_div(on, off, log_target=True, reduction="batchmean"))
    return tot / max(len(prompts), 1)


# ------------------------------------------------------------------ registered rules (pure; tested on synthetic rows)
def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def dev_bar(ns_recall, grad_recall):
    """The recall bar, the same in dev and in the verdict: max(0.5 x gradient, no_sleep + 2)."""
    return max(0.5 * grad_recall, ns_recall + RECALL_PLUS)


def dev_pick(ref, rows):
    """ref: {"no_sleep": {recall}, "gradient": {recall, generic_kl, heldout_kl_dev}} on the dev seed. rows: [{lr, tau,
    recall, generic_kl, heldout_kl_dev, rank_min}] of the v2 arm. Qualify: all finite, recall >= dev_bar (the verdict's
    bar, from the ref gradient), generic_kl <= 1.25 x gradient's, heldout_kl_dev <= 1.25 x gradient's, rank_min >=
    RANK_MIN. Pick max recall; ties: lower heldout_kl_dev, then lower lr, then lower tau. A non-finite ref -> UNDEFINED-DEV
    (the scale is not closed); none qualifying -> NEGATIVE-DEV (no six seeds). Only the DEV half of the held-out prompts
    is read here."""
    need = [ref["no_sleep"]["recall"], ref["gradient"]["recall"], ref["gradient"]["generic_kl"],
            ref["gradient"]["heldout_kl_dev"]]
    if not all(_finite(x) for x in need):
        return {"verdict": "UNDEFINED-DEV", "why": "non-finite ref", "pick": None, "n_rows": len(rows)}
    floor = dev_bar(ref["no_sleep"]["recall"], ref["gradient"]["recall"])
    cap_g = DRIFT_MULT * ref["gradient"]["generic_kl"]
    cap_h = DRIFT_MULT * ref["gradient"]["heldout_kl_dev"]
    ok = [r for r in rows if all(_finite(r.get(k)) for k in ("recall", "generic_kl", "heldout_kl_dev", "rank_min"))
          and r["recall"] >= floor and r["generic_kl"] <= cap_g and r["heldout_kl_dev"] <= cap_h
          and r["rank_min"] >= RANK_MIN]
    out = {"floor": floor, "cap_generic": cap_g, "cap_heldout_dev": cap_h, "n_rows": len(rows), "n_qualify": len(ok),
           "rows": [{k: r.get(k) for k in ("lr", "tau", "recall", "generic_kl", "heldout_kl_dev", "rank_min")}
                    for r in rows]}
    if not ok:
        return {**out, "verdict": "NEGATIVE-DEV", "pick": None}
    best = sorted(ok, key=lambda r: (-r["recall"], r["heldout_kl_dev"], r["lr"], r["tau"]))[0]
    return {**out, "verdict": "PICKED", "pick": {"lr": best["lr"], "tau": best["tau"]}}


def _wins(recs, hi, lo, k, seeds):
    return sum(recs[hi][s][k] > recs[lo][s][k] for s in seeds)


def engagement(recs, seeds=SEEDS):
    """Why each control arm (v2_shuffled, noproj, sham, sham_tok) did NOT have a live night, per seed (empty = all live).
    Live: adapter_norm finite, > 0 and >= ENGAGE_NORM_MIN x that seed's v2 adapter_norm; replay_loss_tail <
    replay_loss_head (both finite); sham_tok: the rank-matched subspace was built at full rank (m_k_used == m_k and
    m_v_used == m_v on every layer). An inert control would pass (b) or (g) for free (v1 was nearly inert)."""
    why = []
    for arm in CONTROL_ARMS:
        for s in seeds:
            r, v = recs[arm][s], recs["v2"][s].get("adapter_norm")
            n = r.get("adapter_norm")
            if not (_finite(n) and n > 0 and _finite(v) and n >= ENGAGE_NORM_MIN * v):
                why.append(f"{arm} s{s} not engaged: adapter_norm {n} vs v2 {v} (needs > 0 and >= {ENGAGE_NORM_MIN} x v2)")
            hd, tl = r.get("replay_loss_head"), r.get("replay_loss_tail")
            if not (_finite(hd) and _finite(tl) and tl < hd):
                why.append(f"{arm} s{s} not engaged: replay loss head {hd} tail {tl} (needs tail < head)")
            if arm == "sham_tok":
                sub = r.get("subspace") or {}
                if not (sub.get("m_k") and sub.get("m_k_used") == sub.get("m_k")
                        and sub.get("m_v") and sub.get("m_v_used") == sub.get("m_v")):
                    why.append(f"sham_tok s{s} not rank-matched: m_k/m_v used {sub.get('m_k_used')}/"
                               f"{sub.get('m_v_used')} != {sub.get('m_k')}/{sub.get('m_v')}")
    return why


def verdict(recs):
    """recs: {arm: {seed: {recall, false_recall, generic_kl, heldout_kl, adapter_norm}}}, every arm of ARMS on exactly
    SEEDS (else ValueError). heldout_kl is the VERDICT half of the held-out prompts.
    UNDEFINED: any gated value (recall, false recall, generic or held-out drift, any arm; v2 adapter norm) non-finite;
    gradient mean recall < 2; a v2 adapter norm zero; mean v2_shuffled recall > 1.0.
    core: (a) v2 recall >= max(0.5 x gradient, no_sleep + 2); (b) v2 > v2_shuffled on >= 5 seeds and shuffled mean <= 1;
    (c1) v2 generic drift <= 1.25 x gradient; (c2) v2 held-out drift <= 1.25 x gradient; (d) v2 false recall <=
    gradient + 0.5. Not core -> NEGATIVE.
    (f) lesion, HELD-OUT drift only (the keep projection zeroes generic drift by construction, so generic is reported,
    not gated): mean noproj held-out > the c2 cap AND >= 1.5 x mean v2 held-out AND noproj > v2 on >= 5 of 6 seeds.
    (g) each sham (random, tokens) fails: [held-out drift: mean sham >= 1.5 x v2 AND sham > v2 on >= 5 seeds] OR
    [recall: mean sham < the (a) bar AND v2 > sham on >= 5 seeds].
    Engagement (`engagement`): if (a), (c1), (c2) and (d) all pass (they read only v2, gradient, no_sleep) and any
    control arm was not live on any seed -> UNDEFINED (an inert control cannot pass (b), (f) or (g)); a failing
    (a)/(c1)/(c2)/(d) is NEGATIVE whatever the controls did.
    GO-V2 = core, f, g. GO-PLAIN-LOCAL = core, not f. GO-NONSPECIFIC = core, f, not g (names the sham that passed)."""
    for arm in ARMS:
        if arm not in recs or set(recs[arm]) != set(SEEDS):
            raise ValueError(f"verdict needs every arm on exactly the seeds {SEEDS}: {arm} has "
                             f"{sorted(recs.get(arm, {}))}")
    seeds = list(SEEDS)
    why = [f"{arm} s{s} {k} non-finite" for arm in ARMS for s in seeds for k in GATED
           if not _finite(recs[arm][s].get(k))]
    why += [f"v2 s{s} adapter_norm non-finite" for s in seeds if not _finite(recs["v2"][s].get("adapter_norm"))]
    if why:
        return {"n_seeds": len(seeds), "verdict": "UNDEFINED", "why": why}
    m = lambda arm, k: _mean(recs[arm][s][k] for s in seeds)
    G, NS = m("gradient", "recall"), m("no_sleep", "recall")
    bar = dev_bar(NS, G)
    cg, ch = DRIFT_MULT * m("gradient", "generic_kl"), DRIFT_MULT * m("gradient", "heldout_kl")
    v2h = m("v2", "heldout_kl")
    shuf = m("v2_shuffled", "recall")
    nwin = _wins(recs, "v2", "v2_shuffled", "recall", seeds)
    f_win = _wins(recs, "noproj", "v2", "heldout_kl", seeds)
    g_detail = {}
    for sh in ("sham", "sham_tok"):
        dw, rw = _wins(recs, sh, "v2", "heldout_kl", seeds), _wins(recs, "v2", sh, "recall", seeds)
        drift_spec = m(sh, "heldout_kl") >= LESION_RATIO * v2h and dw >= SPEC_SEEDS
        recall_spec = m(sh, "recall") < bar and rw >= SPEC_SEEDS
        g_detail[sh] = {"drift_specific": drift_spec, "recall_specific": recall_spec, "heldout_wins": dw,
                        "recall_wins": rw, "fails": drift_spec or recall_spec}
    cl = {"a": m("v2", "recall") >= bar,
          "b": nwin >= SHUF_SEEDS and shuf <= SHUF_MAX,
          "c1": m("v2", "generic_kl") <= cg,
          "c2": v2h <= ch,
          "d": m("v2", "false_recall") <= m("gradient", "false_recall") + FALSE_PLUS,
          "f": m("noproj", "heldout_kl") > ch and m("noproj", "heldout_kl") >= LESION_RATIO * v2h
               and f_win >= SPEC_SEEDS,
          "g": all(d["fails"] for d in g_detail.values())}
    out = {"n_seeds": len(seeds), "bar": bar, "cap_generic": cg, "cap_heldout": ch, "v2_beats_shuffled": nwin,
           "noproj_heldout_wins": f_win, "noproj_over_v2_heldout": (m("noproj", "heldout_kl") / v2h) if v2h else None,
           "shams": g_detail,
           "means": {a: {k: m(a, k) for k in ("recall", "false_recall", "generic_kl", "heldout_kl")} for a in ARMS},
           "clauses": cl}
    eng = engagement(recs, seeds)
    out["engagement"] = eng or "all controls live"
    out["control_norm_ratio"] = {a: _mean(recs[a][s]["adapter_norm"] / recs["v2"][s]["adapter_norm"] for s in seeds)
                                 if all(_finite(recs[a][s].get("adapter_norm")) for s in seeds)
                                 and all(recs["v2"][s]["adapter_norm"] for s in seeds) else None
                                 for a in CONTROL_ARMS}
    why = []
    if G < 2:
        why.append("gradient mean recall < 2")
    if any(recs["v2"][s]["adapter_norm"] == 0 for s in seeds):
        why.append("v2 adapter norm zero")
    if shuf > SHUF_MAX:
        why.append("v2_shuffled mean recall > 1.0")
    if why:
        return {**out, "verdict": "UNDEFINED", "why": why}
    if not all(cl[k] for k in ("a", "c1", "c2", "d")):
        return {**out, "verdict": "NEGATIVE"}
    if eng:
        return {**out, "verdict": "UNDEFINED", "why": eng}
    if not cl["b"]:
        return {**out, "verdict": "NEGATIVE"}
    if not cl["f"]:
        return {**out, "verdict": "GO-PLAIN-LOCAL"}
    if not cl["g"]:
        return {**out, "verdict": "GO-NONSPECIFIC", "passing_shams": [k for k, d in g_detail.items() if not d["fails"]]}
    return {**out, "verdict": "GO-V2"}


# ------------------------------------------------------------------ CLI
def _load_dir(d, pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(d, pattern))):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def _one(units, key, what):
    vals = {json.dumps(u.get(key)) for u in units}
    if len(vals) != 1:
        raise SystemExit(f"{what}: units differ in {key}: {sorted(vals)} (stale or mixed units; use a fresh dir)")
    return units[0].get(key)


def _model_id(path):
    return os.path.basename(str(path)) if path else None


def cmd_pick(a):
    """Dev pick. Refuses (exit non-zero, no pick.json) unless the dir holds exactly the 2 ref units and one v2 unit for
    each of the 9 (lr, tau) cells, all seed 7, one model, one commit, the pinned prompts: a crash never closes a scale."""
    units = [u for u in _load_dir(a.dir, "*.json") if isinstance(u, dict) and "arm" in u and u.get("seed") == DEV_SEED]
    ref_u = [u for u in units if u["tag"] == a.ref_tag]
    v2_u = [u for u in units if u["tag"] != a.ref_tag and u["arm"] == "v2"]
    bad = []
    if sorted(u["arm"] for u in ref_u) != ["gradient", "no_sleep"]:
        bad.append(f"ref units {sorted(u['arm'] for u in ref_u)} != [gradient, no_sleep]")
    cells = sorted((u.get("lr"), u.get("tau")) for u in v2_u)
    if cells != sorted(itertools.product(DEV_LRS, DEV_TAUS)):
        bad.append(f"v2 cells {cells} != the 9 registered (lr, tau)")
    if bad:
        raise SystemExit("pick refused: " + "; ".join(bad))
    allu = ref_u + v2_u
    model, commit = _one(allu, "model_id", "pick"), _one(allu, "commit", "pick")
    _one(allu, "prompts_sha256", "pick")
    if not commit:
        raise SystemExit("pick refused: units carry no commit")
    ref = {u["arm"]: u["result"] for u in ref_u}
    rows = [{"lr": u["lr"], "tau": u["tau"], **u["result"], "rank_min": (u.get("subspace") or {}).get("rank_min")}
            for u in v2_u]
    res = {**dev_pick(ref, rows), "model_id": model, "commit": commit, "prompts_sha256": allu[0].get("prompts_sha256"),
           "ranks": {f"lr{u['lr']}_tau{u['tau']}": {k: (u.get("subspace") or {}).get(k) for k in ("m_k", "m_v")}
                     for u in v2_u}}
    print(json.dumps(res, indent=1))
    if a.out:
        atomic_write(a.out, res)


def read_pick(path):
    with open(path) as fh:
        p = json.load(fh)
    if p.get("verdict") != "PICKED":
        raise SystemExit(f"{path}: {p.get('verdict')}: six seeds are not run")
    return p


def check_units(units, pick=None):
    """One instrument: one model, commit and prompts list across all units; one (lr, tau) across the local arms; the
    pick's lr, tau, model and commit when a pick is given. Raises SystemExit naming the mismatch."""
    model, commit = _one(units, "model_id", "score"), _one(units, "commit", "score")
    _one(units, "prompts_sha256", "score")
    loc = [u for u in units if u["arm"] in LOCAL_ARMS]
    lr = _one(loc, "lr", "score (local arms)")
    tau = _one([u for u in loc if u["arm"] != "noproj"], "tau", "score (projected arms)")
    if pick is not None:
        want = (pick["pick"]["lr"], pick["pick"]["tau"], pick["model_id"], pick["commit"])
        if (lr, tau, model, commit) != want:
            raise SystemExit(f"units (lr, tau, model, commit) {(lr, tau, model, commit)} != pick {want}")
    return {"lr": lr, "tau": tau, "model_id": model, "commit": commit}


def cmd_score(a):
    units = [u for u in _load_dir(a.dir, f"{a.tag}_s*_*.json") if u["seed"] in SEEDS and u["tag"] == a.tag]
    recs = {}
    for u in units:
        rec = dict(u["result"])
        if u.get("subspace") is not None:
            rec["subspace"] = u["subspace"]          # the sham_tok rank check of `engagement` reads it
        recs.setdefault(u["arm"], {})[u["seed"]] = rec
    missing = [(arm, s) for arm in ARMS for s in SEEDS if s not in recs.get(arm, {})]
    if missing:
        raise SystemExit(f"missing units: {missing}")
    inst = check_units(units, read_pick(a.pick) if a.pick else None)
    res = {**verdict(recs), "instrument": inst}
    print(json.dumps(res, indent=1))
    if a.out:
        atomic_write(a.out, res)


def run_units(a, c, base, *, PROMPTS, GENERIC, build_arm_episodes, evaluate, sleep, keys_all, PlasticCortex):
    """The per-(seed, arm) loop of `run` on a loaded cortex `c` (tests pass a stub cortex). Each arm is one atomic unit;
    existing units are skipped (resume)."""
    dev_p, ver_p = split_prompts(PROMPTS, a.prompts_sha)
    psha = prompts_sha(PROMPTS)
    if a.pick:
        p = read_pick(a.pick)
        if p.get("model_id") != _model_id(a.model):
            raise SystemExit(f"pick model {p.get('model_id')} != --model {_model_id(a.model)}")
        a.lr, a.tau = p["pick"]["lr"], p["pick"]["tau"]
    prov = provenance(a.device, "float32")
    if a.pick and p.get("commit") != prov["commit"]:
        raise SystemExit(f"pick commit {p.get('commit')} != this code's commit {prov['commit']}")
    todo = [arm for arm in a.arm_list if not os.path.exists(unit_path(a.out_dir, a.tag, a.seed, arm))]
    eps = build_arm_episodes(a.seed)
    for arm in todo:
        t1 = time.time()
        torch.manual_seed(a.seed)
        pc = PlasticCortex(c)
        losses, sinfo = [], {}
        lg = lambda msg, arm=arm: print(arm, msg, flush=True)
        if arm == "gradient":
            losses = sleep(pc, base, eps["sleep_true"], GENERIC, steps=a.steps, lr=a.grad_lr, kl_weight=a.kl_weight,
                           seed=a.seed, log=lg)
        elif arm != "no_sleep":
            e = eps["sleep_shuffled"] if arm == "v2_shuffled" else eps["sleep_true"]
            losses = v2_sleep(pc, base, e, GENERIC, steps=a.steps, lr=a.lr, kl_weight=a.kl_weight, decay=a.decay,
                              seed=a.seed, mode="kp", feedback=a.feedback, tau=a.tau, subspace=SUBSPACE_OF[arm], log=lg,
                              info=sinfo)
        r = evaluate(pc, base, keys_all)
        # dev/verdict firewall: the VERDICT half is never computed on the dev seed (the pick reads only the DEV half)
        r["heldout_kl"] = None if a.seed == DEV_SEED else heldout_drift(pc, base, ver_p)
        r["heldout_kl_dev"] = heldout_drift(pc, base, dev_p)
        r["adapter_norm"] = pc.adapter_norm()
        r["factor_norms"] = pc.factor_norms()
        r["final_replay_loss"] = losses[-1][0] if losses else None
        r["first_replay_loss"] = losses[0][0] if losses else None
        r["replay_loss_head"] = _mean(x[0] for x in losses[:LOSS_WIN]) if losses else None
        r["replay_loss_tail"] = _mean(x[0] for x in losses[-LOSS_WIN:]) if losses else None
        unit = {"tag": a.tag, "seed": a.seed, "arm": arm, "model": a.model, "model_id": _model_id(a.model),
                "steps": a.steps, "lr": a.lr if arm in LOCAL_ARMS else None,
                "tau": a.tau if arm in LOCAL_ARMS and arm != "noproj" else None,
                "decay": a.decay, "feedback": a.feedback if arm in LOCAL_ARMS else None,
                "grad_lr": a.grad_lr if arm == "gradient" else None, "subspace": sinfo, "prompts_sha256": psha,
                "pick": a.pick, **prov, "elapsed_s": time.time() - t1, "result": r}
        atomic_write(unit_path(a.out_dir, a.tag, a.seed, arm), unit)
        print(arm, {k: v for k, v in r.items() if not k.endswith("_items")}, flush=True)
    return todo


def cmd_run(a):
    guard()
    want = [x.strip() for x in a.arms.split(",") if x.strip()]
    if [x for x in want if x not in ARMS]:
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
    from brain.plastic_cortex import PlasticCortex
    from brain.sleep import sleep
    t0 = time.time()
    c = Cortex(model_path=a.model, device=a.device)
    _, base = c.read(BIRTH, None)
    base = [s.float() for s in base]
    todo = run_units(a, c, base, PROMPTS=PROMPTS, GENERIC=GENERIC, build_arm_episodes=build_arm_episodes,
                     evaluate=evaluate, sleep=sleep, keys_all=[k for _, _, k in FACTS], PlasticCortex=PlasticCortex)
    print(f"done {len(todo)} units in {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--arms", default=",".join(ARMS))
    r.add_argument("--lr", type=float, default=0.3)
    r.add_argument("--tau", type=float, default=0.99)
    r.add_argument("--decay", type=float, default=1e-3)
    r.add_argument("--feedback", default="transport", choices=list(PRESETS))
    r.add_argument("--kl-weight", type=float, default=1.0)
    r.add_argument("--grad-lr", type=float, default=5e-4)
    r.add_argument("--steps", type=int, default=240)
    r.add_argument("--device", default="cpu")
    r.add_argument("--tag", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--pick", default=None, help="dev pick.json: lr and tau from it; its model and commit must match")
    r.add_argument("--prompts-sha", default=PROMPTS_SHA256)
    p = sub.add_parser("pick")
    p.add_argument("--dir", required=True)
    p.add_argument("--ref-tag", default="ref")
    p.add_argument("--out", default=None)
    s = sub.add_parser("score")
    s.add_argument("--dir", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--out", default=None)
    s.add_argument("--pick", default=None, help="dev pick.json: the units must carry its lr, tau, model and commit")
    a = ap.parse_args(argv)
    {"run": cmd_run, "pick": cmd_pick, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
