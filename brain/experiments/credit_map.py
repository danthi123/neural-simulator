# Capability: Sleep consolidation
# Record: brain/research/2026-11-28-local-rule-research-round.md
# Expected-fail: a diagnostic, not a method. It can come out UNDEFINED at seed 7 / 0.1B: gradient recall there is near the
#   floor (3-5 of 12), a single-layer night may not move the graded read-out by 0.1 nats in any layer, Adam's accumulated
#   update may not even agree with itself across replay orders (positive control). Backprop is used as an instrument only.
"""T0 credit map (research note section 1): did the local rules move the adapters along the direction that gives recall?

All arms at one seed (dev seed 7), one model (0.1B), the same told episodes and keep texts as sleep_probe. Every arm starts
from the same adapter init (torch.manual_seed(seed) then PlasticCortex) unless it is layer-restricted.
1. Cosine (per layer, in effective weight space dW = A @ B, key path, value path and both together "kv") between the
   accumulated update of v1 (local_sleep, transport feedback, the 11-27 pick lr 0.01) and of v3 (local_rule_v3, the
   0.1B best dev cell lr 0.3 keep 1) and the gradient-sleep update (brain.sleep.sleep). Instruments beside it:
   grad_reseed (gradient sleep, another replay order: the positive control, "can this measure see alignment at all"),
   grad_shuffled (gradient sleep on the shuffled-name episodes: the content-free direction and the instrument sanity),
   grad_init / grad_init_ans (the first-order direction at init: -A0 @ dL/dB of the replay CE over every telling, all
   positions / answer-predicting positions only; no night).
2. Gradient sleep restricted to ONE layer (PlasticCortex(layers=[i])), all positions (grad_layer<i>) and answer-predicting
   positions only (grad_ans_layer<i>), plus all layers answer positions only (grad_ans): recall /12 and the graded
   read-out per layer. Carrier layers R = layers whose single-layer, all-position night gains >= 0.5 x the best layer's
   gain in mean answer log-probability over no_sleep (and the best gain > 0.1 nats).
3. v3's grad_distill arm (backprop to the double-reading teacher's logits): bad target or bad rule (report).
Graded read-out on every arm (hippocampus off, the ask() frame): log p of the told word holding the key (leading space)
right after "User: <question>\\n\\nAssistant:", summed over its tokens, and the rank of its first token (1 = top-1).

Verdict on the misdirected-credit claim (prereg brain/research/2026-11-28-credit-map-T0-prereg.md):
CONFIRMED / FALSIFIED / UNDEFINED, `verdict()` below (pure, tested with each control failing alone).

    POOL_NODE=1 python -m brain.experiments.credit_map run --model M.pth --seed 7 --out-dir D --tag t0
    python -m brain.experiments.credit_map score --dir D --tag t0 --out D/verdict.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from brain.experiments.local_rule_v2 import _finite, _load_dir, _mean, _model_id, _one, atomic_write, provenance, unit_path
from brain.experiments.local_rule_v3 import answer_span, distill_sleep, replay_ce, v3_sleep
from brain.experiments.local_sleep import local_sleep

DEV_SEED = 7
RESEED_OFFSET = 1          # grad_reseed replays in the order of seed + 1 (same adapter init)
V1 = dict(lr=0.01, decay=1e-3, feedback="transport", mode="kp")   # brain/experiments/results/2026-11-27-local-sleep-01b-read.md
V3 = dict(lr=0.3, keep=1.0, decay=1e-3)                            # 0.1B best dev cell, -local-rule-v3-read.md
CORE = ("no_sleep", "gradient", "grad_reseed", "grad_shuffled", "v1", "v3", "grad_distill", "grad_ans", "grad_init",
        "grad_init_ans")
DELTA_ARMS = ("gradient", "grad_reseed", "grad_shuffled", "v1", "v3", "grad_distill", "grad_ans", "grad_init", "grad_init_ans")
PAIRS = tuple([(x, "gradient") for x in ("v1", "v3", "grad_distill", "grad_ans", "grad_shuffled", "grad_reseed")]
              + [(x, r) for x in ("v1", "v3", "gradient") for r in ("grad_init", "grad_init_ans")]
              + [(x, "grad_shuffled") for x in ("v1", "v3")])
RULES = ("v1", "v3")
COS_LOW = 0.1              # prediction: cosine under 0.1 in the carrier layers
COS_HIGH = 0.3             # falsifier: cosine above 0.3
POS_COS_MIN = 0.5          # positive control: grad_reseed vs gradient, mean over R, must reach this (above COS_HIGH)
GAIN_MIN = 0.1             # nats: the best single-layer gain in mean answer log p must exceed this, else no map
CARRIER_FRAC = 0.5         # R = layers with gain >= 0.5 x the best layer's gain
GRAD_MIN = 2               # gradient recall /12 at the seed, else the instrument is too weak
SHUF_PLUS = 1              # grad_shuffled recall <= no_sleep + 1 (content-specific test)
RECALL_PLUS = 2            # distill reading: the v3 bar max(0.5 x gradient, no_sleep + 2)


# ------------------------------------------------------------------ guard
def guard(env=None):
    env = os.environ if env is None else env
    if env.get("POOL_NODE") != "1" and env.get("BRAIN_ALLOW_W3090") != "1":
        raise SystemExit("credit_map run: refuses to run unless POOL_NODE=1 or BRAIN_ALLOW_W3090=1")


def layer_arms(L):
    return [f"grad_layer{i}" for i in range(L)] + [f"grad_ans_layer{i}" for i in range(L)]


def all_arms(L):
    return list(CORE) + layer_arms(L) + ["cosine"]


def arm_layer(arm):
    """grad_layer3 -> 3, grad_ans_layer3 -> 3, else None."""
    for p in ("grad_ans_layer", "grad_layer"):
        if arm.startswith(p) and arm[len(p):].isdigit():
            return int(arm[len(p):])
    return None


# ------------------------------------------------------------------ answer positions and the masked gradient night
def answer_rows(pipe, text, key):
    """Logit rows of `text` that predict the tokens of `key` (row j-1 predicts token j)."""
    ids = pipe.encode(text)
    a, b = answer_span(pipe, ids, text, key)
    if a == 0:
        raise ValueError(f"answer at position 0 of {text!r}: no prediction row")
    return list(range(a - 1, b))


def masked_ce(logits, ids, rows):
    r = torch.tensor(rows, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits[r], torch.tensor(ids, device=logits.device)[r + 1])


def masked_sleep(plastic, base_state, episodes, rows, generic, *, steps=240, lr=5e-4, kl_weight=1.0, seed=0, log=None):
    """brain.sleep.sleep with the replay CE on `rows[j]` of episode j only (None = every row, which is brain.sleep's loss;
    tested equal). Same RNG draws, optimiser and keep KL as brain.sleep.sleep (no guard)."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    c = plastic.cortex
    opt = torch.optim.Adam([p for p in plastic.parameters() if p.requires_grad], lr=lr)
    weights = [max(s, 1e-3) for _, s in episodes]
    ep_ids = [c.pipe.encode(t) for t, _ in episodes]
    gen_ids = [c.pipe.encode(g if isinstance(g, str) else g[0]) for g in generic]
    plastic.enabled = False
    with torch.no_grad():
        ref = [F.log_softmax(plastic(ids, base_state)[0], -1) for ids in gen_ids]
    plastic.enabled = True
    losses = []
    for step in range(steps):
        j = rng.choices(range(len(episodes)), weights=weights)[0]
        ids = ep_ids[j]
        logits, _ = plastic(ids, base_state)
        rr = list(range(len(ids) - 1)) if rows[j] is None else rows[j]
        replay = masked_ce(logits, ids, rr)
        g = rng.randrange(len(gen_ids))
        g_logits, _ = plastic(gen_ids[g], base_state)
        keep = F.kl_div(F.log_softmax(g_logits, -1), ref[g], log_target=True, reduction="batchmean")
        opt.zero_grad()
        (replay + kl_weight * keep).backward()
        opt.step()
        losses.append((float(replay), float(keep)))
        if log and step % 20 == 0:
            log(f"masked sleep step {step}: replay {float(replay):.3f} keep {float(keep):.4f}")
    return losses


def masked_replay_ce(pc, base_state, episodes, rows):
    """Fixed-order, forward-only engagement probe of masked_sleep: mean masked CE over every telling. Changes nothing."""
    out = []
    with torch.no_grad():
        for (text, _), rr in zip(episodes, rows):
            ids = pc.cortex.pipe.encode(text)
            lg, _ = pc(ids, base_state)
            out.append(float(masked_ce(lg, ids, list(range(len(ids) - 1)) if rr is None else rr)))
    return _mean(out)


def init_direction(pc, base_state, episodes, rows):
    """The first-order update direction at init: dW = A0 @ (-dL/dB), L = mean replay CE over every telling (on `rows`;
    None entries = every row). B starts at zero, so dL/dA = 0 and the direction lives in B. Adapters unchanged.
    Returns (factors {A_k, B_k, A_v, B_v: {layer: tensor}}, gradient norm)."""
    for p in pc.parameters():
        p.grad = None
    loss = 0.0
    for (text, _), rr in zip(episodes, rows):
        ids = pc.cortex.pipe.encode(text)
        lg, _ = pc(ids, base_state)
        loss = loss + masked_ce(lg, ids, list(range(len(ids) - 1)) if rr is None else rr) / len(episodes)
    loss.backward()
    f = {"A_k": {}, "B_k": {}, "A_v": {}, "B_v": {}}
    gn = 0.0
    for s in pc.A_k:
        for a, b in (("A_k", "B_k"), ("A_v", "B_v")):
            g = getattr(pc, b)[s].grad
            g = torch.zeros_like(getattr(pc, b)[s]) if g is None else g
            gn += float(g.pow(2).sum())
            f[a][s] = getattr(pc, a)[s].detach().cpu().clone()
            f[b][s] = (-g).detach().cpu().clone()
    for p in pc.parameters():
        p.grad = None
    return f, gn ** 0.5


# ------------------------------------------------------------------ read-outs
def surface(sentence, key):
    """The told word containing `key`, as told: from the key's start to the end of its word ("peanut" in "...peanuts."
    -> "peanuts"), so the read-out scores the tokens the telling used, not a re-tokenised stem."""
    i = sentence.lower().find(key.lower())
    if i < 0:
        raise ValueError(f"key {key!r} not in {sentence!r}")
    j = i + len(key)
    while j < len(sentence) and sentence[j].isalnum():
        j += 1
    return sentence[i:j]


def graded(pc, state, facts):
    """Answer log-probability and rank at the question, hippocampus off. facts: [(sentence, question, key)]. Per fact:
    the ask() prompt, then " <key as told>" teacher-forced; logp = summed log p of the key's tokens, rank = rank of its
    first token at the answer position (1 = top-1)."""
    pipe = pc.cortex.pipe
    items = []
    with torch.no_grad():
        for s, q, k in facts:
            p = pipe.encode(f"User: {q}\n\nAssistant:")
            a = pipe.encode(" " + surface(s, k))
            lg, _ = pc(p + a, state)
            lp = F.log_softmax(lg.float(), -1)
            rows = [len(p) - 1 + t for t in range(len(a))]
            tok = [float(lp[r, a[t]]) for t, r in enumerate(rows)]
            first = lp[rows[0]]
            rank = int((first > first[a[0]]).sum()) + 1
            items.append({"key": k, "n_tok": len(a), "logp": sum(tok), "logp_first": tok[0], "rank_first": rank})
    ranks = sorted(it["rank_first"] for it in items)
    return {"mean_logp": _mean(it["logp"] for it in items), "mean_logp_first": _mean(it["logp_first"] for it in items),
            "mean_rank_first": _mean(ranks), "median_rank_first": ranks[len(ranks) // 2] if ranks else None,
            "n_top1": sum(r == 1 for r in ranks), "n_top10": sum(r <= 10 for r in ranks), "items": items}


def factors(pc):
    return {n: {s: t.detach().cpu().clone() for s, t in getattr(pc, n).items()} for n in ("A_k", "B_k", "A_v", "B_v")}


def layer_norms(pc):
    with torch.no_grad():
        return {s: float(((pc.A_k[s] @ pc.B_k[s]).pow(2).sum() + (pc.A_v[s] @ pc.B_v[s]).pow(2).sum()) ** 0.5)
                for s in pc.A_k}


def _ip(A1, B1, A2, B2):
    """<A1 B1, A2 B2>_F without forming the products: sum((A1^T A2) * (B1 B2^T)). float64."""
    A1, B1, A2, B2 = (t.double() for t in (A1, B1, A2, B2))
    return float(((A1.T @ A2) * (B1 @ B2.T)).sum())


def layer_cosines(f1, f2):
    """{layer: {k, v, kv}} cosines of the effective updates A @ B (key path, value path, both concatenated) for every
    layer the two factor sets share. A zero norm gives None."""
    out = {}
    for s in sorted(set(f1["A_k"]) & set(f2["A_k"]), key=int):
        ip, n1, n2 = {}, {}, {}
        for p, (a, b) in (("k", ("A_k", "B_k")), ("v", ("A_v", "B_v"))):
            ip[p] = _ip(f1[a][s], f1[b][s], f2[a][s], f2[b][s])
            n1[p] = _ip(f1[a][s], f1[b][s], f1[a][s], f1[b][s])
            n2[p] = _ip(f2[a][s], f2[b][s], f2[a][s], f2[b][s])
        c = lambda i, x, y: (i / math.sqrt(x * y)) if x > 0 and y > 0 else None
        out[s] = {"k": c(ip["k"], n1["k"], n2["k"]), "v": c(ip["v"], n1["v"], n2["v"]),
                  "kv": c(ip["k"] + ip["v"], n1["k"] + n1["v"], n2["k"] + n2["v"]),
                  "norm_a": math.sqrt(max(n1["k"] + n1["v"], 0.0)), "norm_b": math.sqrt(max(n2["k"] + n2["v"], 0.0))}
    return out


def delta_path(out_dir, tag, seed, arm):
    return os.path.join(out_dir, f"{tag}_s{seed}_{arm}.delta.pt")


def save_delta(path, f):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(f, path + ".tmp")
    os.replace(path + ".tmp", path)


# ------------------------------------------------------------------ the verdict (pure; tested on synthetic units)
def _r(units, arm, k):
    return ((units.get(arm) or {}).get("result") or {}).get(k)


def _logp(units, arm):
    return ((_r(units, arm, "graded") or {}).get("mean_logp"))


def _e(units, arm):
    return (units.get(arm) or {}).get("engagement") or {}


def _night_ok(units, arm, layers, strict=True):
    """norm > 0, finite fixed-order loss at start and end, steps run, the layers given; strict = the loss also fell.
    Single-layer nights are not strict: the map exists to find layers that do nothing, so a flat layer is a reading."""
    e, n = _e(units, arm), _r(units, arm, "adapter_norm")
    return (_finite(n) and n > 0 and _finite(e.get("loss_start")) and _finite(e.get("loss_end"))
            and (not strict or e["loss_end"] < e["loss_start"]) and e.get("steps_run", 0) > 0
            and e.get("layers_on") == layers)


def engagement(units, L):
    """{"gated": [...], "report": [...]} reasons an arm's run is not what its role needs."""
    allL = list(range(L))
    gated, rep = [], []
    if _r(units, "no_sleep", "adapter_norm") != 0:
        gated.append(f"no_sleep adapter_norm {_r(units, 'no_sleep', 'adapter_norm')} != 0")
    for arm in ("gradient", "grad_reseed", "grad_shuffled"):
        if not _night_ok(units, arm, allL):
            gated.append(f"{arm} not engaged: norm {_r(units, arm, 'adapter_norm')}, {_e(units, arm)}")
    for i in range(L):
        if not _night_ok(units, f"grad_layer{i}", [i], strict=False):
            gated.append(f"grad_layer{i} not engaged: norm {_r(units, f'grad_layer{i}', 'adapter_norm')}, "
                         f"{_e(units, f'grad_layer{i}')}")
    e, n = _e(units, "v1"), _r(units, "v1", "adapter_norm")
    if not (_finite(n) and n > 0 and e.get("steps_run", 0) > 0 and e.get("feedback") == V1["feedback"]):
        gated.append(f"v1 not engaged: norm {n}, {e}")
    e, n = _e(units, "v3", ), _r(units, "v3", "adapter_norm")
    if not (_finite(n) and n > 0 and _finite(e.get("err_start")) and _finite(e.get("err_end"))
            and e["err_end"] < e["err_start"] and e.get("teacher") == "double" and e.get("layers_on") == allL):
        gated.append(f"v3 not engaged: norm {n}, err {e.get('err_start')}->{e.get('err_end')}, layers {e.get('layers_on')}")
    for arm in ["grad_ans"] + [f"grad_ans_layer{i}" for i in range(L)]:
        if not _night_ok(units, arm, allL if arm == "grad_ans" else [arm_layer(arm)], strict=arm == "grad_ans"):
            rep.append(f"{arm} not engaged: norm {_r(units, arm, 'adapter_norm')}, {_e(units, arm)}")
    if not _night_ok(units, "grad_distill", allL):
        rep.append(f"grad_distill not engaged: norm {_r(units, 'grad_distill', 'adapter_norm')}, {_e(units, 'grad_distill')}")
    for arm in ("grad_init", "grad_init_ans"):
        g = _e(units, arm).get("grad_norm")
        if not (_finite(g) and g > 0):
            rep.append(f"{arm} not engaged: grad_norm {g}")
    return {"gated": gated, "report": rep}


def _cos(units, a, b, i, part="kv"):
    return ((((_r(units, "cosine", "pairs") or {}).get(f"{a}|{b}") or {}).get(str(i))) or {}).get(part)


def carriers(units, L):
    """(R, gains {layer: gain}, best): R = layers with single-layer gain >= CARRIER_FRAC x best, best > GAIN_MIN."""
    ns = _logp(units, "no_sleep")
    gains = {i: _logp(units, f"grad_layer{i}") - ns for i in range(L)}
    best = max(gains.values()) if gains else float("nan")
    R = [i for i, g in gains.items() if best > GAIN_MIN and g >= CARRIER_FRAC * best]
    return R, gains, best


def distill_reading(units, rep):
    if any(x.startswith("grad_distill") for x in rep) or not _finite(_r(units, "grad_distill", "recall")):
        return "undetermined: grad_distill not engaged or non-finite"
    bar = max(0.5 * _r(units, "gradient", "recall"), _r(units, "no_sleep", "recall") + RECALL_PLUS)
    if _r(units, "grad_distill", "recall") >= bar:
        return f"rule-limit: grad_distill recall {_r(units, 'grad_distill', 'recall')} >= bar {bar} on v3's own target"
    return f"target-limit: grad_distill recall {_r(units, 'grad_distill', 'recall')} < bar {bar}: v3's target itself does not give recall"


def position_reading(units, rep):
    if any(x.startswith("grad_ans ") or x.startswith("grad_ans not") for x in rep):
        return "undetermined: grad_ans not engaged"
    g, a, ns = _logp(units, "gradient"), _logp(units, "grad_ans"), _logp(units, "no_sleep")
    if not all(_finite(x) for x in (g, a, ns)) or g - ns <= 0:
        return "undetermined: no gradient gain to compare"
    return (f"answer-positions-only keeps {(a - ns) / (g - ns):.2f} of the all-position gain "
            f"(recall {_r(units, 'grad_ans', 'recall')} vs {_r(units, 'gradient', 'recall')})")


def verdict(units, L):
    """units: {arm: unit} for every arm of all_arms(L). Order:
    1. UNDEFINED: an arm missing; a gated value non-finite (recall, graded mean_logp, adapter_norm of the gated arms; a
       cosine of a rule or of the positive control in a carrier layer); a gated engagement gate fails (no_sleep norm 0;
       gradient, grad_reseed, grad_shuffled: norm > 0, fixed-order loss end < start, steps run, the layers given; every
       grad_layer<i>: the same without the loss decrease (finite loss only); v1 norm > 0 with transport feedback; v3 norm > 0, error end < start, every layer); gradient recall
       < GRAD_MIN; grad_shuffled recall > no_sleep + SHUF_PLUS; no carrier map (best single-layer gain <= GAIN_MIN); the
       positive control mean over R of cos(grad_reseed, gradient) < POS_COS_MIN.
    2. FALSIFIED: v1 or v3 has mean over R of cos(rule, gradient) > COS_HIGH.
    3. CONFIRMED: v1 and v3 have |cos(rule, gradient)| < COS_LOW in EVERY layer of R, and |R| < L / 2.
    4. else UNDEFINED (|cosine| between the bars, a rule anti-aligned: mean over R < -COS_HIGH, a different finding than
       near-zero projection; or recall not concentrated in a minority of layers).
    grad_distill, grad_ans, grad_ans_layer<i>, grad_init* are report-only: their engagement gates only their readings.
    Their units must still EXIST (a missing unit of any arm, report-only included, is UNDEFINED: the run is incomplete)."""
    need = all_arms(L)
    miss = [a for a in need if a not in units]
    base = {"L": L, "thresholds": {"cos_low": COS_LOW, "cos_high": COS_HIGH, "pos_cos_min": POS_COS_MIN,
                                   "gain_min": GAIN_MIN, "carrier_frac": CARRIER_FRAC, "grad_min": GRAD_MIN}}
    if miss:
        return {**base, "verdict": "UNDEFINED", "why": [f"missing arms {miss}"]}
    gated_arms = ["no_sleep", "gradient", "grad_reseed", "grad_shuffled", "v1", "v3"] + [f"grad_layer{i}" for i in range(L)]
    why = [f"{a} {k} non-finite" for a in gated_arms for k in ("recall", "adapter_norm") if not _finite(_r(units, a, k))]
    why += [f"{a} graded mean_logp non-finite" for a in gated_arms if not _finite(_logp(units, a))]
    if why:
        return {**base, "verdict": "UNDEFINED", "why": why}
    eng = engagement(units, L)
    why = list(eng["gated"])
    NS, G = _r(units, "no_sleep", "recall"), _r(units, "gradient", "recall")
    if G < GRAD_MIN:
        why.append(f"gradient recall {G} < {GRAD_MIN}: the instrument is too weak at this seed")
    if _r(units, "grad_shuffled", "recall") > NS + SHUF_PLUS:
        why.append(f"grad_shuffled recall {_r(units, 'grad_shuffled', 'recall')} > no_sleep {NS} + {SHUF_PLUS}: test broken")
    R, gains, best = carriers(units, L)
    out = {**base, "carriers": R, "gains": {str(i): g for i, g in gains.items()}, "best_gain": best,
           "recall": {a: _r(units, a, "recall") for a in need if a != "cosine"},
           "mean_logp": {a: _logp(units, a) for a in need if a != "cosine"},
           "engagement": eng, "distill_reading": distill_reading(units, eng["report"]),
           "position_reading": position_reading(units, eng["report"])}
    if not R:
        why.append(f"no carrier map: best single-layer gain {best} <= {GAIN_MIN} nats")
    cos = {r: {str(i): _cos(units, r, "gradient", i) for i in R} for r in RULES + ("grad_reseed",)}
    why += [f"cos({r}, gradient) layer {i} non-finite" for r, d in cos.items() for i, c in d.items() if not _finite(c)]
    if why:
        return {**out, "cos_R": cos, "verdict": "UNDEFINED", "why": why}
    mean_R = {r: _mean(d.values()) for r, d in cos.items()}
    anti = [r for r in RULES if mean_R[r] < -COS_HIGH]
    out.update(cos_R=cos, cos_mean_R=mean_R, minority=len(R) < L / 2, anti_aligned=anti)
    if mean_R["grad_reseed"] < POS_COS_MIN:
        return {**out, "verdict": "UNDEFINED", "why": [f"positive control: cos(grad_reseed, gradient) mean over R "
                                                       f"{mean_R['grad_reseed']:.3f} < {POS_COS_MIN}"]}
    hi = [r for r in RULES if mean_R[r] > COS_HIGH]
    if hi:
        rd = {r: ("direction fine, no recall: the limit is step size or drift budget (08-11 per-arm lr artifact); next is a "
                  "per-layer lr sweep, not a new rule" if _r(units, r, "recall") <= NS + SHUF_PLUS
                  else "aligned and recalling at this config") for r in hi}
        return {**out, "verdict": "FALSIFIED", "falsified_by": hi, "reading": rd}
    low = all(abs(cos[r][str(i)]) < COS_LOW for r in RULES for i in R)
    if low and out["minority"]:
        return {**out, "verdict": "CONFIRMED"}
    w = []
    if anti:
        w.append(f"anti-aligned: {anti} mean over R of cos(rule, gradient) < -{COS_HIGH} "
                 f"({ {r: round(mean_R[r], 3) for r in anti} }): the rule moves against the gradient, not orthogonal to it")
    if not low:
        w.append(f"|cosine| between the bars: means over R {({r: round(mean_R[r], 3) for r in RULES})}")
    if not out["minority"]:
        w.append(f"recall not concentrated: {len(R)} carrier layers of {L}")
    return {**out, "verdict": "UNDEFINED", "why": w}


# ------------------------------------------------------------------ run
def _episodes(build_arm_episodes, seed, facts, pipe):
    eps = build_arm_episodes(seed)
    true = eps["sleep_true"]
    if len(true) != len(facts):
        raise SystemExit(f"{len(true)} true episodes != {len(facts)} facts")
    rows = [answer_rows(pipe, t, k) for (t, _), (_, _, k) in zip(true, facts)]
    return true, eps["sleep_shuffled"], rows


def cosine_unit(a, base_unit):
    fs = {arm: torch.load(delta_path(a.out_dir, a.tag, a.seed, arm)) for arm in DELTA_ARMS}
    pairs = {f"{x}|{y}": layer_cosines(fs[x], fs[y]) for x, y in PAIRS}
    return {**base_unit, "arm": "cosine", "engagement": {"kind": "instrument", "n_pairs": len(pairs)},
            "result": {"pairs": pairs, "space": "effective dW = A @ B per layer (k, v, kv)"}}


def run_units(a, c, base, *, GENERIC, build_arm_episodes, evaluate, sleep, keys_all, facts, PlasticCortex):
    """Every arm one atomic unit (json; delta arms also write their factors .pt BEFORE the json); existing units are
    skipped (resume). "cosine" runs last, from the saved factors. Returns the arms run."""
    L = c.model.n_layer
    want = all_arms(L) if a.arm_list == ["all"] else list(a.arm_list)
    bad = [x for x in want if x not in all_arms(L)]
    if bad:
        raise SystemExit(f"unknown arms {bad}")
    want = [x for x in want if x != "cosine"] + (["cosine"] if "cosine" in want else [])
    todo = [arm for arm in want if not os.path.exists(unit_path(a.out_dir, a.tag, a.seed, arm))]
    prov = provenance(a.device, "float32")
    base_unit = {"tag": a.tag, "seed": a.seed, "model": a.model, "model_id": _model_id(a.model), "steps": a.steps,
                 "L": L, "grad_lr": a.grad_lr, "kl_weight": a.kl_weight, "v1": V1, "v3": V3, **prov}
    true, shuf, rows = _episodes(build_arm_episodes, a.seed, facts, c.pipe)
    allrows = [None] * len(true)
    for arm in todo:
        t1 = time.time()
        if arm == "cosine":
            miss = [x for x in DELTA_ARMS if not os.path.exists(delta_path(a.out_dir, a.tag, a.seed, x))]
            if miss:
                raise SystemExit(f"cosine: delta files missing for {miss}; run those arms first")
            unit = cosine_unit(a, base_unit)
            unit["elapsed_s"] = time.time() - t1
            atomic_write(unit_path(a.out_dir, a.tag, a.seed, arm), unit)
            print("cosine", {k: {s: v["kv"] for s, v in d.items()} for k, d in unit["result"]["pairs"].items()
                             if k.endswith("|gradient")}, flush=True)
            continue
        li = arm_layer(arm)
        torch.manual_seed(a.seed)
        pc = PlasticCortex(c, layers=None if li is None else [li])
        lg = lambda msg, arm=arm: print(arm, msg, flush=True)
        on = list(pc.layers)
        eng = {"kind": arm, "layers_on": on}
        fac = None
        if arm == "no_sleep":
            eng.update(kind="none", steps_run=0)
        elif arm in ("gradient", "grad_reseed", "grad_shuffled") or (li is not None and arm.startswith("grad_layer")):
            eps = shuf if arm == "grad_shuffled" else true
            sd = a.seed + (RESEED_OFFSET if arm == "grad_reseed" else 0)
            l0 = replay_ce(pc, base, eps)
            losses = sleep(pc, base, eps, GENERIC, steps=a.steps, lr=a.grad_lr, kl_weight=a.kl_weight, seed=sd, log=lg)
            eng.update(kind="gradient", replay_seed=sd, steps_run=len(losses), loss_start=l0, loss_end=replay_ce(pc, base, eps),
                       loss_tail=losses[-1][0] if losses else None)
        elif arm == "grad_ans" or (li is not None and arm.startswith("grad_ans_layer")):
            l0 = masked_replay_ce(pc, base, true, rows)
            losses = masked_sleep(pc, base, true, rows, GENERIC, steps=a.steps, lr=a.grad_lr, kl_weight=a.kl_weight,
                                  seed=a.seed, log=lg)
            eng.update(kind="gradient_answer_rows", steps_run=len(losses), loss_start=l0,
                       loss_end=masked_replay_ce(pc, base, true, rows), n_rows=sum(len(r) for r in rows))
        elif arm == "v1":
            l0 = replay_ce(pc, base, true)
            losses = local_sleep(pc, base, true, GENERIC, steps=a.steps, lr=V1["lr"], kl_weight=a.kl_weight,
                                 decay=V1["decay"], seed=a.seed, mode=V1["mode"], feedback=V1["feedback"], log=lg)
            eng.update(kind="local_v1", feedback=V1["feedback"], steps_run=len(losses), loss_start=l0,
                       loss_end=replay_ce(pc, base, true))
        elif arm == "v3":
            info = {}
            v3_sleep(pc, base, true, GENERIC, steps=a.steps, lr=V3["lr"], keep=V3["keep"], decay=V3["decay"], seed=a.seed,
                     teacher_kind="double", log=lg, info=info)
            eng.update(info)
            eng["kind"] = "local_v3"
        elif arm == "grad_distill":
            info = {}
            distill_sleep(pc, base, true, GENERIC, steps=a.steps, lr=a.grad_lr, kl_weight=a.kl_weight, seed=a.seed,
                          log=lg, info=info)
            eng.update(info, kind="grad_distill", layers_on=on)
        elif arm in ("grad_init", "grad_init_ans"):
            fac, gn = init_direction(pc, base, true, allrows if arm == "grad_init" else rows)
            eng.update(kind="init_gradient", grad_norm=gn, steps_run=0)
        else:
            raise SystemExit(f"no runner for arm {arm}")
        r = evaluate(pc, base, keys_all)
        r["graded"] = graded(pc, base, facts)
        r["adapter_norm"] = pc.adapter_norm()
        r["layer_norms"] = layer_norms(pc)
        eng["adapter_norm"] = r["adapter_norm"]
        if arm in DELTA_ARMS:
            save_delta(delta_path(a.out_dir, a.tag, a.seed, arm), fac if fac is not None else factors(pc))
        unit = {**base_unit, "arm": arm, "layer": li, "elapsed_s": time.time() - t1, "engagement": eng, "result": r}
        atomic_write(unit_path(a.out_dir, a.tag, a.seed, arm), unit)
        print(arm, {k: v for k, v in r.items() if not k.endswith("_items") and k != "graded"},
              {k: v for k, v in r["graded"].items() if k != "items"}, flush=True)
    return todo


def cmd_run(a):
    guard()
    a.arm_list = [x.strip() for x in a.arms.split(",") if x.strip()]
    from brain.brain import BIRTH
    from brain.cortex import Cortex
    from brain.experiments.facts import FACTS
    from brain.experiments.sleep_probe import GENERIC, build_arm_episodes, evaluate
    from brain.plastic_cortex import PlasticCortex as PC
    from brain.sleep import sleep
    t0 = time.time()
    c = Cortex(model_path=a.model, device=a.device)
    _, base = c.read(BIRTH, None)
    base = [s.float() for s in base]
    todo = run_units(a, c, base, GENERIC=GENERIC, build_arm_episodes=build_arm_episodes, evaluate=evaluate, sleep=sleep,
                     keys_all=[k for _, _, k in FACTS], facts=list(FACTS), PlasticCortex=PC)
    print(f"done {len(todo)} units in {time.time() - t0:.0f}s", flush=True)


def cmd_score(a):
    units = [u for u in _load_dir(a.dir, f"{a.tag}_s{a.seed}_*.json") if isinstance(u, dict) and u.get("tag") == a.tag
             and u.get("seed") == a.seed and "arm" in u]
    if not units:
        raise SystemExit(f"score: no units for tag {a.tag} seed {a.seed} in {a.dir}")
    inst = {"model_id": _one(units, "model_id", "score"), "commit": _one(units, "commit", "score"),
            "L": _one(units, "L", "score"), "steps": _one(units, "steps", "score")}
    if not inst["commit"]:
        raise SystemExit("score refused: units carry no commit")
    res = {**verdict({u["arm"]: u for u in units}, inst["L"]), "instrument": inst, "seed": a.seed,
           "hosts": sorted({u.get("host") for u in units}, key=str), "devices": sorted({u.get("device") for u in units}, key=str)}
    print(json.dumps({k: v for k, v in res.items() if k not in ("recall", "mean_logp")}, indent=1))
    if a.out:
        atomic_write(a.out, res)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--seed", type=int, default=DEV_SEED)
    r.add_argument("--arms", default="all", help="all, or a comma list of CORE, grad_layer<i>, grad_ans_layer<i>, cosine")
    r.add_argument("--steps", type=int, default=240)
    r.add_argument("--grad-lr", type=float, default=5e-4)
    r.add_argument("--kl-weight", type=float, default=1.0)
    r.add_argument("--device", default="cpu")
    r.add_argument("--tag", required=True)
    r.add_argument("--out-dir", required=True)
    s = sub.add_parser("score")
    s.add_argument("--dir", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--seed", type=int, default=DEV_SEED)
    s.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    {"run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
