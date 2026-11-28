# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-28-translation-layer-DESIGN.md
# Expected-fail: likely (design odds of dense headroom 40%). A residual write may not reach the matrix memory a delta
#   writes (design section 4 item 4) and local rule v3 matched the telling's activity without answering; BARRIER is a
#   live outcome and is a verdict on residual writing of a pooled telling vector at this scale (1.5B DEV), not on the
#   capability. Strict scorer only (paraphrase is MISSING, PREREQUISITES.md).
"""Translation layer S1a, dictionary free: can any residual write carry a told fact? (forward only, nothing trained)

Prereg: brain/research/2026-11-28-translation-s1a-prereg.md (rule, gates, fallbacks). Design section 3, S1a.
Population: the told questions of the banked KR lives (voice_tag.told_questions at night 14, 25/22/20 = 67 item-lives over
seeds 319/320/321), fresh state per question, the banked n14 brain (A0snap adapters, voice_ceiling_oracle.session). An
item-life with no stored right episode is outside the population (counted). Layer L = n_layers // 2 (12 at 1.5B).

Arms (one battery each; vector arms at every gain of GAINS, each with its SHAM):
  OFF            hippocampus off, nothing written (must reproduce the banked voice-ceiling H rows)
  C              the telling heard first, then the question (ceiling; voice_split ctx)
  DELTA          the right episode forced, reinstated by the brain's own path at the shipped gain (today's memory)
  DELTA-NOTEXT   DELTA with every stored episode's text withheld (matrix delta only)
  SHAM-DELTA     another told episode forced (memory_ceiling.pick_sham: strength tercile, nearest age)
  DELTA-NULL     the DELTA driver with nothing forced (recall returns no episode): must give OFF's rows (driver link)
  DENSE@g        R(item) - mean of the other told items' R added at layer L on every position (Provenance "all");
                 R = the residual after block L averaged over the tokens of "User: <telling>" read from base
  SPARSE-RAW@g   the top-64 coordinates (by magnitude) of DENSE, nothing else
  SPARSE-PROJ@g  a random overcomplete projection (16384 x C, seeded) of DENSE, top-64 code, decoded by the transpose
  SHAM-<X>@g     arm X built from another told item (voice_ceiling_oracle.wrong_map), same gain
  ORACLE-RES     report only: C - H at the label (the item's own question), at its natural norm
Every vector arm writes g * nu * v / |v|, nu = the life's median |DENSE| over told items (all arms norm matched).
Gain per write arm and life is picked on the DEV_FACTS36 bank only (most strict-correct, neutral perplexity ratio <= 1.05,
ties to the smaller gain); told answers never choose a gain. DELTA anchor (two links): REPRO re-runs memory_ceiling's
ORACLE unit on night 14 and compares every battery row to the banked unit (the forced-episode path reproduces; REPRO2, only
if REPRO misses, checks this commit's self-determinism), and DELTA-NULL must reproduce this commit's OFF rows (the
fresh-state DELTA driver adds nothing but the forced episode; OFF itself is anchored to the banked voice-ceiling H rows).
The DELTA unit also reports its replies against the banked ORACLE told rows (report only: the banked battery is sequential
with carried state, a different condition; banked OFF told rows match the fresh-state H rows on only 12/22).

    POOL_NODE=1 python -m brain.experiments.translation_s1a run --life L.json --snaps DIR --trunk T --model M \\
        --out-dir D --work W --mc-ref results/2026-11-27-memory-ceiling/pulled --vc-ref vc_full_s<seed>.json
    python -m brain.experiments.translation_s1a score --dir D [--seeds 319 320 321] [--out verdict.json]
Host guard: `run` refuses unless POOL_NODE=1 or BRAIN_ALLOW_W3090=1.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import re
import socket
import statistics
import time

import torch
import torch.nn.functional as F

from brain import source_monitor as SM
from brain.experiments import cascade_screen as CS
from brain.experiments import memory_ceiling as MC
from brain.experiments import voice_ceiling_oracle as VC
from brain.experiments import voice_split as VS
from brain.experiments import voice_tag as VT

NIGHT = VT.NIGHT
KR_SEEDS = (319, 320, 321)                     # the only lives with banked n14 snapshots and ceiling rows (DEV)
PROJECT_SEEDS = (42, 43, 44, 100, 101, 102)
K_TOP = 64
M_PROJ = 16384
PROJ_SEED = 20261128
GAINS = (0.5, 1.0, 2.0, 4.0, 8.0)
WRITE = ("DENSE", "SPARSE-RAW", "SPARSE-PROJ")
PPL_MAX = 1.05                                 # neutral-text perplexity ratio bar (admissible gain)
PPL_PASSAGES, PPL_VECTORS = 4, 4               # first 4 off-frame passages x first 4 vectors of the set
OFFFRAME_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "mix_night_offframe.json")
OFFFRAME_SHA256 = "b1a95684b8caeec1f6564493afa35a6cddc280d791f8466a2c73ccbc49e6f268"
C_MIN = 0.6                                    # C strict-correct rate below this: fallback / UNDEFINED
REPRO_MIN = 0.9                                # DELTA anchor: banked ORACLE n14 replies identical, all 20 rows per life
SELF_DET_MIN = 0.95                            # REPRO2 vs REPRO, every row
DRIVER_MIN = 0.95                              # DELTA-NULL vs OFF: replies and onsets identical, per life
MARGIN_PER_LIFE = 3                            # equivalence margin: 3 item-lives per life (design "under 3 items per life")
REF_MARGIN = 0.05                              # refusal non-inferiority margin: 5% of item-lives
ALPHA = 0.05
BOOT, BOOT_SEED = 2000, 20261128
NOOP_TOL, NOOP_MAX = 1e-6, 0.5
NORM_RTOL = 1e-3
HOLM = (("S1", "DENSE", "SHAM-DENSE"), ("S2", "SPARSE-RAW", "OFF"), ("S3", "SPARSE-PROJ", "OFF"),
        ("S4", "SPARSE-RAW", "SHAM-SPARSE-RAW"), ("S5", "SPARSE-PROJ", "SHAM-SPARSE-PROJ"),
        ("S6", "DENSE", "DELTA-NOTEXT"))
FIXED = ("OFF", "C", "DELTA", "DELTA-NOTEXT", "SHAM-DELTA", "DELTA-NULL")


def guard(env=None):
    env = os.environ if env is None else env
    if env.get("POOL_NODE") != "1" and env.get("BRAIN_ALLOW_W3090") != "1":
        raise SystemExit("translation_s1a run: refuses unless POOL_NODE=1 (pool node) or BRAIN_ALLOW_W3090=1")


def gname(arm: str, g: float) -> str:
    return f"{arm}@{g:g}"


def vec_names(gains=GAINS) -> list:
    return [gname(p + x, g) for x in WRITE for g in gains for p in ("", "SHAM-")]


def required_units(gains=GAINS) -> list:
    return ["META", "PPL", *[f"DEV-{gname(x, g)}" for x in WRITE for g in gains], *FIXED, *vec_names(gains), "REPRO"]


# ------------------------------------------------------------------ residual reads and vectors
@contextlib.contextmanager
def grab_layer(n_layers: int, layer: int, sink: list):
    """Every position's residual after block `layer` (the input of layer_norm call 2 * (layer + 1); the same index
    voice_ceiling_oracle.grab_residuals reads at the last position, and where a Provenance tag at `layer` lands)."""
    orig, n, target = F.layer_norm, [0], 2 * (layer + 1)

    def ln(x, *a, **k):
        if n[0] == target:
            sink.append(x.detach().clone())
        n[0] += 1
        return orig(x, *a, **k)
    F.layer_norm = ln
    try:
        yield
    finally:
        F.layer_norm = orig
    if n[0] != 2 * n_layers + 1:
        raise RuntimeError(f"residual capture saw {n[0]} layer_norm calls, expected {2 * n_layers + 1}")


def telling_residual(b, text: str, layer: int) -> torch.Tensor:
    """(C,): the residual after block `layer`, averaged over the tokens of `text`, read from the birth base through the
    brain's live cortex (adapters on; no tag attached)."""
    p = b._plastic()
    if p.source_tag is not None:
        raise RuntimeError("residual read with a tag attached")
    if VC.REQUIRE_ACTIVE and not b.cortex.active:
        raise RuntimeError("the cortex is not running through the consolidated adapters")
    sink: list = []
    with torch.no_grad(), grab_layer(p.L, layer, sink):
        b.cortex.read(text, [x.clone() for x in b.base])
    if len(sink) != 1 or sink[0].dim() != 2:
        raise RuntimeError(f"captured {len(sink)} residual blocks")
    return sink[0].float().mean(0).cpu()


def centred(R: dict) -> dict:
    """{item: R[item] - mean of the OTHER items' R} (leave-self-out centring: the shared frame is removed)."""
    if len(R) < 2:
        raise ValueError("centring needs at least 2 items")
    out = {}
    for k, v in R.items():
        out[k] = v - torch.stack([w for j, w in R.items() if j != k]).mean(0)
    return out


def topk_abs(v: torch.Tensor, k: int) -> torch.Tensor:
    out = torch.zeros_like(v)
    idx = torch.topk(v.abs(), min(k, v.numel())).indices
    out[idx] = v[idx]
    return out


def proj_matrix(c: int, m: int = M_PROJ, seed: int = PROJ_SEED) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, c, generator=g) / math.sqrt(c)


def write_vectors(R: dict, k: int = K_TOP, m: int = M_PROJ, P=None) -> dict:
    """{arm: {item: (C,)}} for DENSE, SPARSE-RAW, SPARSE-PROJ from the per-item token-mean residuals R."""
    dense = centred(R)
    c = next(iter(dense.values())).numel()
    P = proj_matrix(c, m) if P is None else P
    return {"DENSE": dense, "SPARSE-RAW": {i: topk_abs(v, k) for i, v in dense.items()},
            "SPARSE-PROJ": {i: P.t() @ topk_abs(P @ v, k) for i, v in dense.items()}}


def scaled(v: torch.Tensor, g: float, nu: float) -> torch.Tensor:
    n = float(v.norm())
    if n == 0:
        raise RuntimeError("a zero write vector")
    return v * (g * nu / n)


def nu_of(vecs: dict) -> float:
    return float(statistics.median(float(v.norm()) for v in vecs["DENSE"].values()))


# ------------------------------------------------------------------ neutral-text perplexity
def offframe_passages(n=PPL_PASSAGES) -> list:
    import hashlib
    raw = open(OFFFRAME_FILE, "rb").read()
    if hashlib.sha256(raw).hexdigest() != OFFFRAME_SHA256:
        raise RuntimeError("off-frame passage file changed (sha256)")
    return json.loads(raw)[:n]


def neutral_nll(b, passages: list, vec, layer: int) -> float:
    """Token-mean NLL (nats) of the passages read from base through the live cortex with `vec` (or nothing) added at
    `layer` on every position. Full-output forward of PlasticCortex (the live path; read_nll is the pretrained one)."""
    p = b._plastic()
    was = p.enabled
    p.enabled = True
    if vec is not None:
        SM.attach(p, vec, layer, False, SM.Provenance("all"))
    try:
        tot, n = 0.0, 0
        dt = p.z["emb.weight"].dtype
        with torch.no_grad():
            for t in passages:
                ids = b.cortex.pipe.encode(t)
                lg, _ = p(ids, [x.to(dt).clone() for x in b.base])
                lp = torch.log_softmax(lg.float(), -1)
                tot += float(-lp[:-1].gather(1, torch.tensor(ids[1:])[:, None]).sum())
                n += len(ids) - 1
        return tot / max(n, 1)
    finally:
        if vec is not None:
            SM.detach(p)
        p.enabled = was


# ------------------------------------------------------------------ statistics (exact, item-life paired)
def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def cp_upper(k: int, n: int, alpha: float = ALPHA) -> float:
    """One-sided (1 - alpha) Clopper-Pearson upper bound of a binomial proportion (bisection, exact)."""
    if n == 0 or k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _binom_cdf(k, n, mid) > alpha else (lo, mid)
    return hi


def cp_lower(k: int, n: int, alpha: float = ALPHA) -> float:
    if n == 0 or k <= 0:
        return 0.0
    return 1.0 - cp_upper(n - k, n, alpha)


def holm(ps: dict) -> dict:
    """Holm step-down adjusted p-values {name: p_adj}."""
    order = sorted(ps, key=lambda k: (ps[k], k))
    m, out, run = len(order), {}, 0.0
    for j, k in enumerate(order):
        run = max(run, min(1.0, (m - j) * ps[k]))
        out[k] = run
    return out


def paired(x: dict, y: dict, boot: int = BOOT) -> dict:
    """x, y: {(seed, item): bool}. b = x right and y wrong, c the reverse. Exact two-sided McNemar p (memory_ceiling.
    sign_p); exact non-superiority / non-inferiority bounds on b - c from the one-sided 95% Clopper-Pearson bound on
    b / (b + c); item-clustered bootstrap 5th / 95th percentiles of b - c (items resampled, every life's cell of a drawn
    item comes with it)."""
    keys = sorted(set(x) & set(y))
    b = sum(bool(x[k]) and not y[k] for k in keys)
    c = sum(bool(y[k]) and not x[k] for k in keys)
    n = b + c
    out = {"N": len(keys), "b": b, "c": c, "diff": b - c, "p": MC.sign_p(b, c),
           "upper": n * (2 * cp_upper(b, n) - 1) if n else 0.0, "lower": n * (2 * cp_lower(b, n) - 1) if n else 0.0}
    by: dict = {}
    for k in keys:
        by.setdefault(k[1], []).append(int(bool(x[k])) - int(bool(y[k])))
    cl = list(by.values())
    if cl and boot:
        rng = random.Random(BOOT_SEED)
        ms = []
        for _ in range(boot):
            pick = [cl[rng.randrange(len(cl))] for _ in cl]
            s, m = sum(sum(p) for p in pick), sum(len(p) for p in pick)
            ms.append(s / m * len(keys))
        ms.sort()
        out["boot_lower"], out["boot_upper"] = ms[int(0.05 * boot)], ms[int(0.95 * boot) - 1]
    else:
        out["boot_lower"] = out["boot_upper"] = 0.0
    return out


# ------------------------------------------------------------------ row measures
def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (t or "").lower())).strip()


def echo_verbatim(reply: str, telling: str | None) -> bool:
    """The reply holds the whole told sentence (case and punctuation ignored): the 11-27 hand-read echo."""
    return bool(telling) and _norm(telling) in _norm(reply)


def echo_fp(reply: str, keys, telling) -> bool:
    from brain.experiments.strict_scorer import is_echo
    return bool(telling) and any(is_echo(reply, k, telling) for k in keys or ())


def row_measures(row: dict, led: dict) -> dict:
    it = led.get(row.get("item") or "") or {}
    br = row.get("brain") or ""
    kn = VS.knows(row, led)
    return {"ok": bool(row.get("ok")), "knows": kn, "refusal": CS.refusal(br),
            "echo_verbatim": echo_verbatim(br, it.get("telling")), "echo_fp": echo_fp(br, it.get("keys"), it.get("telling")),
            "uv_knows": kn and CS.own2(br)}


# ------------------------------------------------------------------ batteries
def delta_battery(res, snaps, trunk, cortex, work, qs, name, forced: dict, notext=False, night=NIGHT):
    """The DELTA family: fresh state per question, hippocampus on, the forced episode returned by recall and reinstated by
    the brain's own Hippocampus.reinstate (shipped GAIN). notext: every episode's text withheld for the whole battery."""
    from brain.eval import life
    seed, led = res["seed"], res["ledger"]
    t0 = time.time()
    with VC.session(res, snaps, trunk, cortex, work, night, f"b_{name}") as b:
        hc = b.hippocampus
        b.set_lesions(hippocampus_off=False, consolidated_off=False)
        base = [x.clone() for x in b.base]
        texts = [e.text for e in hc.episodes]
        if notext:
            for e in hc.episodes:
                e.text = ""
        yid, mid = VT._onset_ids(b.cortex.pipe)
        cur, calls, rein = {"i": None}, [], []
        orig_re = hc.reinstate

        def recall(cortex_, base_, cue, speaker="User", question=True, exclude=()):
            i = cur["i"]
            if i is None or cue.strip() != qs[i]["text"].strip():
                raise RuntimeError(f"{name}: recall cue {cue!r} is not question {i}")
            calls.append(i)
            k = forced.get(i)
            return (hc.episodes[k], 0.0) if k is not None else (None, 0.0)

        def reinstate(cortex_, state, ep):
            rein.append((cur["i"], round(sum(float(x.float().norm()) for x in ep.delta if x is not None), 4)))
            return orig_re(cortex_, state, ep)
        hc.recall, hc.reinstate = recall, reinstate
        rows = []
        for i, q in enumerate(qs):
            cur["i"] = None
            b.state = [x.clone() for x in base]
            k = forced.get(i)
            st = orig_re(b.cortex, b.state, hc.episodes[k]) if k is not None else b.state
            onset = VC._onset(b, q, st, yid, mid)
            cur["i"] = i
            n_c, n_r = len(calls), len(rein)
            life._reseed(seed, 500 + night)
            r = life.run_battery(b, [q], led, night)[0]
            mine = [d for j, d in rein[n_r:] if j == i]
            want = None if k is None else ("" if notext else texts[k])
            rows.append({**{kk: r.get(kk) for kk in VS.KEEP}, "onset": onset, "forced_idx": k,
                         "recall_calls": len(calls) - n_c, "reinstated": len(mine), "delta_norm": mine[0] if mine else None,
                         "recalled_ok": r.get("recalled") == want, "gain": float(hc.GAIN)})
    return {"rows": rows, "seconds": round(time.time() - t0, 1), "notext": notext}


def dev_battery(res, snaps, trunk, cortex, work, facts, vec_fn, layer, name, night=NIGHT):
    """DEV_FACTS36 gain fit: fresh state per fact, hippocampus off, the fact's write vector on every position, the
    original cue asked; strict correct_answer against the statement."""
    from brain.eval import life
    from brain.experiments.strict_scorer import correct_answer
    seed = res["seed"]
    t0 = time.time()
    rows = []
    with VC.session(res, snaps, trunk, cortex, work, night, f"d_{name}") as b:
        base = [x.clone() for x in b.base]
        for f in facts:
            b.state = [x.clone() for x in base]
            v = vec_fn(f["key"])
            prov = SM.Provenance("all")
            VC.set_vector(b, v, layer, prov)
            try:
                life._reseed(seed, 500 + night)
                reply, _ = b.hear(f["cue"])
            finally:
                VC.clear_vector(b)
            rows.append({"key": f["key"], "cue": f["cue"], "brain": reply, "vec_norm": float(v.norm()),
                         "ok": bool(correct_answer(reply, f["key"], f["cue"], f["statement"]))})
    return {"rows": rows, "ok": sum(r["ok"] for r in rows), "n": len(rows), "seconds": round(time.time() - t0, 1)}


def dev_facts(ledger: dict) -> list:
    """DEV_FACTS36 (original cue), minus any fact whose key is a key of a ledger item of this life (no told overlap)."""
    from brain.experiments.facts_bank import DEV_FACTS36
    lk = {k.lower() for it in ledger.values() for k in (it.get("keys") or ())}
    return [{"statement": s, "cue": o, "key": keys[0]} for s, o, _, _, keys in DEV_FACTS36
            if not any(k.lower() in lk for k in keys)]


def decisions(res, snaps, trunk, cortex, work, qs, layer, night=NIGHT):
    """Right / sham episode per question, and the residual reads (told tellings, dev statements). One session."""
    from brain import clock
    from brain.eval import life
    led = res["ledger"]
    LB = life.use_split(res.get("split", "dev"))
    with VC.session(res, snaps, trunk, cortex, work, night, "meta") as b:
        hc = b.hippocampus
        b.set_lesions(hippocampus_off=False, consolidated_off=False)
        texts = [e.text for e in hc.episodes]
        now = clock.now()
        eff = [hc.effective_strength(e, now) for e in hc.episodes]
        created = [e.created for e in hc.episodes]
        told_map = MC.told_episode_idx(texts, led, LB, life.episode_is_source)
        right, sham = {}, {}
        for i, q in enumerate(qs):
            it = LB.ALL[q["item"]]
            cand = [j for j, t in enumerate(texts) if life.episode_is_source(it, t)]
            if not cand:
                continue
            if len(cand) > 1:
                sims = hc.similarities(b.cortex, b.base, q["text"])
                right[i] = max(cand, key=lambda j: (float(sims[j]), -j))
            else:
                right[i] = cand[0]
            s = MC.pick_sham(right[i], q["item"], told_map, eff, created)
            if s is not None:
                sham[i] = s
        b.set_lesions(hippocampus_off=True, consolidated_off=False)
        items = sorted({q["item"] for q in qs})
        R = {it: telling_residual(b, f"User: {led[it]['telling']}\n\n", layer) for it in items}
        facts = dev_facts(led)
        Rd = {f["key"]: telling_residual(b, f"User: {f['statement']}\n\n", layer) for f in facts}
        gain = float(hc.GAIN)
    return {"right": right, "sham": sham, "R": R, "Rd": Rd, "facts": facts, "store": len(texts), "gain_delta": gain}


# ------------------------------------------------------------------ run (resumable per unit)
def _anchor_frac(rec):
    """The REPRO gate's share: every battery row (banked_all, 20 per life), identical replies; None if nothing aligned."""
    return (rec.get("banked_all") or {}).get("frac")


def _load_banked(mc_ref, seed, night):
    dec = json.load(open(os.path.join(mc_ref, f"dec_s{seed}_n{night}.json")))
    ban = json.load(open(os.path.join(mc_ref, f"u_s{seed}_n{night}_ORACLE.json")))
    return dec, ban


def compare_rows(a: list, b: list, told_only=True) -> dict:
    pairs = [(x, y) for x, y in zip(a, b) if x.get("you") == y.get("you") and (not told_only or y.get("cls") == "told")]
    same = sum(x.get("brain") == y.get("brain") for x, y in pairs)
    return {"n": len(pairs), "same": same, "frac": (same / len(pairs)) if pairs else None,
            "aligned": len(a) == len(b) and all(x.get("you") == y.get("you") for x, y in zip(a, b))}


def run(res, snaps, trunk, cortex, out_dir, work, mc_ref=None, vc_ref=None, units=None, gains=GAINS, tag="", limit=0,
        night=NIGHT, log=print):
    from brain.eval import life
    life.use_split(res.get("split", "dev"))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    seed = res["seed"]
    qs = VT.told_questions(res, night)
    if limit:
        qs = qs[:limit]
    L = cortex.n_layers // 2
    com = VC.commit()
    spec = VC._fp(seed, night, [q["text"] for q in qs], L, com, list(gains), K_TOP, M_PROJ, tag, limit)
    prov = {"seed": seed, "host": socket.gethostname(), "commit": com, "device": str(getattr(cortex, "device", "cpu")),
            "model": res.get("model"), "layer": L, "night": night, "spec": spec, "tag": tag}
    pre = f"u{('-' + tag) if tag else ''}_s{seed}_"
    want = None if not units else [u.strip() for u in units.split(",") if u.strip()]
    done = {}

    def unit(name, fn):
        path = os.path.join(out_dir, f"{pre}{name}.json")
        if os.path.exists(path):
            u = json.load(open(path))
            if u.get("spec") != spec:
                raise SystemExit(f"{path}: written under spec {u.get('spec')} (commit {u.get('commit')}), now {spec}; "
                                 "use a fresh out dir (no mixing)")
            done[name] = u
            return u
        if want is not None and not any(name == w or name.startswith(w) for w in want):
            return None
        u = {**prov, "name": name, "rec": fn()}
        VC._save_json(path, u)
        done[name] = u
        log(seed, name, u["rec"].get("seconds"))
        return u

    vp = os.path.join(work, f"vecs{('-' + tag) if tag else ''}_s{seed}.pt")
    state = {}

    def dec():
        if "d" not in state:
            if os.path.exists(vp):
                blob = torch.load(vp, map_location="cpu", weights_only=False)
                if blob.get("spec") == spec:
                    state["d"] = blob["d"]
            if "d" not in state:
                state["d"] = decisions(res, snaps, trunk, cortex, work, qs, L, night)
                VC._save_pt(vp, {"spec": spec, "d": state["d"]})
            d = state["d"]
            P = proj_matrix(next(iter(d["R"].values())).numel())
            state["V"] = write_vectors(d["R"], P=P)
            state["nu"] = nu_of(state["V"])
            state["Vd"] = write_vectors(d["Rd"], P=P) if len(d["Rd"]) >= 2 else None
            state["wm"] = VC.wrong_map(list(d["R"]))
        return state["d"]

    def meta():
        d = dec()
        led = res["ledger"]
        V = state["V"]
        return {"n_questions": len(qs), "questions": [q["text"] for q in qs],
                "population": [qs[i]["text"] for i in sorted(d["right"])],
                "no_right_episode": [q["text"] for i, q in enumerate(qs) if i not in d["right"]],
                "right_idx": {str(i): v for i, v in d["right"].items()}, "sham_idx": {str(i): v for i, v in d["sham"].items()},
                "items": {q["text"]: q["item"] for q in qs}, "wrong_map": state["wm"], "nu": state["nu"],
                "gains": list(gains), "gain_delta": d["gain_delta"], "store": d["store"],
                "ledger": {it: {"keys": list(led[it].get("keys") or ()), "telling": led[it].get("telling")}
                           for it in sorted({q["item"] for q in qs})},
                "dev_facts": [f["key"] for f in d["facts"]],
                "norms": {a: {it: float(v.norm()) for it, v in V[a].items()} for a in WRITE},
                "nonzero": {a: {it: int((v != 0).sum()) for it, v in V[a].items()} for a in WRITE},
                "seconds": 0.0}

    def told_vec(arm, g, sham=False):
        dec()

        def fn(i, q):
            it = state["wm"][q["item"]] if sham else q["item"]
            return scaled(state["V"][arm][it], g, state["nu"])
        return fn

    def ppl():
        d = dec()
        t0 = time.time()
        passages = offframe_passages()
        out = {"dev": {}, "told": {}, "base_nll": None}
        with VC.session(res, snaps, trunk, cortex, work, night, "ppl") as b:
            base_nll = neutral_nll(b, passages, None, L)
            out["base_nll"] = base_nll
            for set_name, V, keys in (("told", state["V"], list(d["R"])[:PPL_VECTORS]),
                                      ("dev", state["Vd"], list(d["Rd"])[:PPL_VECTORS])):
                if V is None:
                    continue
                for arm in WRITE:
                    for g in gains:
                        nl = [neutral_nll(b, passages, scaled(V[arm][k], g, state["nu"]), L) for k in keys]
                        out[set_name].setdefault(arm, {})[f"{g:g}"] = math.exp(sum(nl) / len(nl) - base_nll)
        out["seconds"] = round(time.time() - t0, 1)
        return out

    unit("META", meta)
    unit("PPL", ppl)
    for arm in WRITE:
        for g in gains:
            unit(f"DEV-{gname(arm, g)}", lambda arm=arm, g=g: dev_battery(
                res, snaps, trunk, cortex, work, dec()["facts"],
                lambda key: scaled(state["Vd"][arm][key], g, state["nu"]), L, f"{arm}{g:g}", night))
    vc_rows = None
    if vc_ref and os.path.exists(vc_ref):
        vc_rows = json.load(open(vc_ref)).get("batteries") or {}

    def off():
        rec = VC.battery(res, snaps, trunk, cortex, work, qs, "OFF", "B", night=night)
        if vc_rows and "H" in vc_rows:
            rec["reproduction"] = VC.reproduce(rec["rows"], vc_rows["H"]["rows"])
        return rec

    def ctx():
        rec = VC.battery(res, snaps, trunk, cortex, work, qs, "C", "ctx", night=night)
        if vc_rows and "C" in vc_rows:
            rec["reproduction"] = VC.reproduce(rec["rows"], vc_rows["C"]["rows"])
        return rec
    unit("OFF", off)
    unit("C", ctx)
    def delta():
        rec = delta_battery(res, snaps, trunk, cortex, work, qs, "DELTA", dec()["right"], False, night)
        if mc_ref and os.path.exists(os.path.join(mc_ref, f"u_s{seed}_n{night}_ORACLE.json")):
            ban = _load_banked(mc_ref, seed, night)[1]
            told = {r["you"]: r for r in ban["rows"] if r.get("cls") == "told"}
            pr = [(r, told[r["you"]]) for r in rec["rows"] if r["you"] in told]
            rec["vs_banked_oracle"] = {"n": len(pr), "same": sum(a["brain"] == b["brain"] for a, b in pr),
                                       "report_only": True}
        return rec
    unit("DELTA", delta)
    unit("DELTA-NOTEXT", lambda: delta_battery(res, snaps, trunk, cortex, work, qs, "NOTEXT", dec()["right"], True, night))
    unit("SHAM-DELTA", lambda: delta_battery(res, snaps, trunk, cortex, work, qs, "SHAMD", dec()["sham"], False, night))
    unit("DELTA-NULL", lambda: delta_battery(res, snaps, trunk, cortex, work, qs, "NULLD", {}, False, night))
    for arm in WRITE:
        for g in gains:
            for sh in (False, True):
                nm = gname(("SHAM-" if sh else "") + arm, g)
                unit(nm, lambda arm=arm, g=g, sh=sh, nm=nm: VC.battery(
                    res, snaps, trunk, cortex, work, qs, nm, "B", told_vec(arm, g, sh), L, "all", night))

    def oracle_res():
        reads = VC.collect_reads(res, snaps, trunk, cortex, work, qs, night, log=lambda *a: None)
        dq = [c[L] - h[L] for c, h in zip(reads["C"], reads["H"])]
        rec = VC.battery(res, snaps, trunk, cortex, work, qs, "ORACLE-RES", "B", lambda i, q: dq[i], L, "all", night)
        rec["report_only"] = True
        return rec
    unit("ORACLE-RES", oracle_res)

    def repro(work_dir):
        if not mc_ref:
            raise SystemExit("REPRO needs --mc-ref (the banked memory-ceiling dec_/u_ files)")
        dcs, ban = _load_banked(mc_ref, seed, night)
        if dcs.get("void"):
            return {"void": True, "seconds": 0.0}
        t0 = time.time()
        u = MC.run_arm(res, night, os.path.join(snaps, f"n{night}"), trunk, cortex, work_dir, True, "ORACLE", dcs)
        return {"rows": u["rows"], "banked": compare_rows(u["rows"], ban["rows"]),
                "banked_all": compare_rows(u["rows"], ban["rows"], told_only=False), "banked_commit": ban.get("commit"),
                "seconds": round(time.time() - t0, 1)}
    r = unit("REPRO", lambda: repro(os.path.join(work, "repro")))
    if r is not None and not r["rec"].get("void") and not (_anchor_frac(r["rec"]) or 0) >= REPRO_MIN:
        def repro2():
            x = repro(os.path.join(work, "repro2"))
            x["self"] = compare_rows(x["rows"], r["rec"]["rows"], told_only=False)
            return x
        unit("REPRO2", repro2)
    return done


# ------------------------------------------------------------------ verdict (prereg section 5; fixed before data)
def pick_gain(dev_ok: dict, ppl_dev: dict, gains=GAINS) -> tuple:
    """(gain, admissible_any): the gain with the most dev strict-correct among gains whose dev perplexity ratio is <=
    PPL_MAX; ties to the smaller gain; with no admissible gain the smallest gain, flagged."""
    adm = [g for g in sorted(gains) if ppl_dev.get(f"{g:g}", math.inf) <= PPL_MAX]
    if not adm:
        return min(gains), False
    best = max(dev_ok.get(f"{g:g}", 0) for g in adm)
    return next(g for g in adm if dev_ok.get(f"{g:g}", 0) == best), True


def _cells(units, seeds, name, measure, pop, gain_of=None):
    """{(seed, item): measure} over the population rows of battery `name` (name may be a function of the seed)."""
    out = {}
    for s in seeds:
        nm = name(s) if callable(name) else name
        u = units[s].get(nm)
        if u is None:
            continue
        led = units[s]["META"]["rec"]["ledger"]
        for r in u["rec"]["rows"]:
            if r.get("you") in pop[s]:
                out[(s, r["item"])] = row_measures(r, led)[measure]
    return out


def _arm_stats(units, seeds, name, pop) -> dict:
    out = {}
    for m in ("ok", "knows", "refusal", "echo_verbatim", "echo_fp", "uv_knows"):
        c = _cells(units, seeds, name, m, pop)
        out[m] = sum(c.values())
        out["n"] = len(c)
    return out


def _engagement(units, s, pop, gains) -> list:
    """Failed engagement records of one life (each a short string)."""
    bad = []
    U, meta = units[s], units[s]["META"]["rec"]
    nu = meta["nu"]
    rows = lambda n: [r for r in U[n]["rec"]["rows"] if r.get("you") in pop]
    if any(r.get("recalled") is not None or r.get("vec_norm") is not None for r in rows("OFF")):
        bad.append(f"s{s} OFF: a row recalled or wrote")
    if not all(r.get("told_first") for r in rows("C")):
        bad.append(f"s{s} C: telling not heard first")
    sham_pop = {meta["questions"][int(i)] for i in meta["sham_idx"]}
    for n in ("DELTA", "DELTA-NOTEXT", "SHAM-DELTA"):
        rr = [r for r in rows(n) if n != "SHAM-DELTA" or r["you"] in sham_pop]
        if not rr or not all(r.get("forced_idx") is not None and r.get("recall_calls") == 1 and r.get("reinstated") == 1
                             and (r.get("delta_norm") or 0) > 0 and r.get("recalled_ok") for r in rr):
            bad.append(f"s{s} {n}: forced episode not reinstated on every row")
    rn = rows("DELTA-NULL")
    if not rn or not all(r.get("forced_idx") is None and r.get("recall_calls") == 1 and r.get("reinstated") == 0 for r in rn):
        bad.append(f"s{s} DELTA-NULL: a row was forced or recall was not called once")
    if U["DELTA-NOTEXT"]["rec"].get("notext") is not True:
        bad.append(f"s{s} DELTA-NOTEXT: text not withheld")
    off_on = {r["you"]: r.get("onset") for r in rows("OFF")}
    for nm in vec_names(gains):
        g = float(nm.split("@")[1])
        rec = U[nm]["rec"]
        m = rec.get("mask") or {}
        rr = rows(nm)
        if not m.get("tokens") or m.get("tagged") != m.get("tokens"):
            bad.append(f"s{s} {nm}: footprint not every position")
        if not rr or not all(r.get("label_tagged") is True and r.get("vec_norm") is not None
                             and abs(r["vec_norm"] - g * nu) <= NORM_RTOL * g * nu for r in rr):
            bad.append(f"s{s} {nm}: written norm not g * nu or label untagged")
        sh = [abs(r["onset"] - off_on[r["you"]]) for r in rr if r.get("onset") is not None and off_on.get(r["you"]) is not None]
        if not sh or sum(d < NOOP_TOL for d in sh) / len(sh) > NOOP_MAX:
            bad.append(f"s{s} {nm}: no-op (onset equals OFF on more than half the rows)")
    return bad


def verdict(units: dict, expected_seeds=KR_SEEDS, gains=GAINS) -> dict:
    """units: {seed: {unit name: unit dict}}. UNDEFINED > HEADROOM > BARRIER > NEITHER; modifiers -DEV (KR lives),
    -UNANCHORED (DELTA anchor missed, self-deterministic), -FALLBACK-KNOWS (C strict < 0.6, knows >= 0.6). Every -DEV
    verdict is a lead, never a generalisation claim. `acts` = the prereg section 8 next step of the outcome is taken
    (HEADROOM-DEV: the S0 prereg; BARRIER-DEV: S0 deferred); False under UNANCHORED or FALLBACK-KNOWS and for UNDEFINED."""
    T = {"seeds": sorted(units), "expected_seeds": list(expected_seeds)}

    def undef(why):
        return {**T, "verdict": "UNDEFINED", "why": why, "acts": False}
    seeds = sorted(units)
    if seeds != sorted(expected_seeds):
        return undef(f"seeds {seeds} != expected {sorted(expected_seeds)}")
    req = required_units(gains)
    miss = [(s, n) for s in seeds for n in req if n not in units[s]]
    if miss:
        return undef(f"units missing {miss[:8]}{' ...' if len(miss) > 8 else ''}")
    for k in ("commit", "model", "layer", "night", "tag"):
        vals = {str(u.get(k)) for s in seeds for u in units[s].values()}
        if len(vals) != 1:
            return undef(f"units disagree on {k}: {sorted(vals)}")
    if any(len({u.get("spec") for u in units[s].values()}) != 1 for s in seeds):
        return undef("mixed spec within a life")
    T["label"] = ("DEV (KR lives 319/320/321; lead only, not a generalisation claim)" if set(seeds) == set(KR_SEEDS)
                  else "SIX (project seeds)" if set(seeds) == set(PROJECT_SEEDS) else "OTHER")
    pop = {s: set(units[s]["META"]["rec"]["population"]) for s in seeds}
    T["population"] = {s: len(pop[s]) for s in seeds}
    T["no_right_episode"] = {s: len(units[s]["META"]["rec"]["no_right_episode"]) for s in seeds}
    N = sum(T["population"].values())
    if N == 0:
        return undef("empty population")
    # gates: OFF reproduction, engagement, DELTA anchor, C
    nq = {s: units[s]["META"]["rec"]["n_questions"] for s in seeds}
    T["off_reproduction"] = {s: units[s]["OFF"]["rec"].get("reproduction") for s in seeds}
    T["c_reproduction_report"] = {s: units[s]["C"]["rec"].get("reproduction") for s in seeds}
    badrep = [s for s in seeds if not VC.repro_ok(T["off_reproduction"][s], nq[s])]
    if badrep:
        return undef(f"OFF does not reproduce the banked voice-ceiling H rows on {badrep}")
    eng = [x for s in seeds for x in _engagement(units, s, pop[s], gains)]
    T["engagement_failed"] = eng
    if eng:
        return undef(f"engagement: {eng[:6]}")
    drv = {}
    for s in seeds:
        d = VC.reproduce([r for r in units[s]["DELTA-NULL"]["rec"]["rows"] if r.get("you") in pop[s]],
                         [r for r in units[s]["OFF"]["rec"]["rows"] if r.get("you") in pop[s]])
        drv[s] = d
        if not (d["n"] == len(pop[s]) and (d["frac"] or 0) >= DRIVER_MIN and (d["onset_frac"] or 0) >= DRIVER_MIN):
            return {**undef(f"s{s}: the DELTA driver with nothing forced does not reproduce OFF ({d})"), "delta_driver": drv}
    T["delta_driver"] = drv
    anchor, unanch = {}, []
    for s in seeds:
        r = units[s]["REPRO"]["rec"]
        if r.get("void"):
            return undef(f"s{s}: the banked memory-ceiling night {units[s]['META'].get('night')} is void")
        fr = _anchor_frac(r)
        anchor[s] = {"all": r.get("banked_all"), "told": r.get("banked")}
        if not ((r.get("banked_all") or {}).get("n") and fr is not None and fr >= REPRO_MIN):
            r2 = (units[s].get("REPRO2") or {}).get("rec")
            sd = (r2 or {}).get("self") or {}
            if not (sd.get("n") and sd.get("frac") is not None and sd["frac"] >= SELF_DET_MIN):
                return undef(f"s{s}: DELTA misses its banked rows ({fr}) and this commit is not self-deterministic ({sd.get('frac')})")
            unanch.append(s)
    T["delta_anchor"], T["unanchored"] = anchor, unanch
    cst = _arm_stats(units, seeds, "C", pop)
    T["C"] = cst
    measure = "ok"
    if cst["ok"] / N < C_MIN:
        if cst["knows"] / N < C_MIN:
            return undef(f"C strict {cst['ok']}/{N} and knows {cst['knows']}/{N} both under {C_MIN}")
        measure = "knows"
    T["measure"] = measure
    # gains (dev only)
    picks, ppl_t = {}, {}
    for s in seeds:
        P = units[s]["PPL"]["rec"]
        ppl_t[s] = P["told"]
        for arm in WRITE:
            dev_ok = {f"{g:g}": units[s][f"DEV-{gname(arm, g)}"]["rec"]["ok"] for g in gains}
            g, adm = pick_gain(dev_ok, (P.get("dev") or {}).get(arm, {}), gains)
            picks.setdefault(arm, {})[s] = {"gain": g, "dev_admissible": adm, "dev_ok": dev_ok}
    T["picks"], T["ppl_told"] = picks, ppl_t
    at = lambda arm, sham=False: (lambda s: gname(("SHAM-" if sham else "") + arm, picks[arm][s]["gain"]))
    cell = lambda name: _cells(units, seeds, name, measure, pop)
    ref = lambda name: _cells(units, seeds, name, "refusal", pop)
    arms = {"OFF": "OFF", "C": "C", "DELTA": "DELTA", "DELTA-NOTEXT": "DELTA-NOTEXT", "SHAM-DELTA": "SHAM-DELTA",
            "ORACLE-RES": "ORACLE-RES"}
    for x in WRITE:
        arms[x], arms["SHAM-" + x] = at(x), at(x, True)
    T["arms"] = {k: _arm_stats(units, seeds, v, pop) for k, v in arms.items() if k != "ORACLE-RES"}
    if all("ORACLE-RES" in units[s] for s in seeds):
        T["arms"]["ORACLE-RES (report)"] = _arm_stats(units, seeds, "ORACLE-RES", pop)
    T["per_gain"] = {gname(p + x, g): _arm_stats(units, seeds, gname(p + x, g), pop)
                     for x in WRITE for g in gains for p in ("", "SHAM-")}
    T["text_path_rows"] = sum(a["brain"] != b["brain"] for s in seeds
                              for a, b in zip(units[s]["DELTA"]["rec"]["rows"], units[s]["DELTA-NOTEXT"]["rec"]["rows"])
                              if a.get("you") in pop[s])
    spop = {s: {units[s]["META"]["rec"]["questions"][int(i)] for i in units[s]["META"]["rec"]["sham_idx"]} for s in seeds}
    skeys = {(s, r["item"]) for s in seeds for r in units[s]["SHAM-DELTA"]["rec"]["rows"] if r.get("you") in spop[s]}
    in_sham = lambda c: {k: v for k, v in c.items() if k in skeys}
    rep = {"DELTA-OFF": paired(cell("DELTA"), cell("OFF")),
           "DELTA-SHAM-DELTA (sham population)": paired(in_sham(cell("DELTA")), in_sham(cell("SHAM-DELTA"))),
           "C-OFF": paired(cell("C"), cell("OFF")),
           "DELTA vs banked ORACLE told rows (sequential battery; report)":
               {s: units[s]["DELTA"]["rec"].get("vs_banked_oracle") for s in seeds}}
    T["report"] = rep
    # primary and Holm family (at each life's dev-picked gain)
    prim = paired(cell(at("DENSE")), cell("OFF"))
    T["primary"] = prim
    fam = {}
    for k, a, b in HOLM:
        fam[k] = {"x": a, "y": b, **paired(cell(arms[a]), cell(arms[b]))}
    adj = holm({k: v["p"] for k, v in fam.items()})
    for k in fam:
        fam[k]["p_holm"] = adj[k]
        fam[k]["holds"] = fam[k]["b"] > fam[k]["c"] and adj[k] < ALPHA
    T["holm"] = fam
    delta = MARGIN_PER_LIFE * len(seeds)
    T["margin_items"] = delta
    nin = paired(cell(at("DENSE")), cell("DELTA-NOTEXT"))
    rf = paired(ref(at("DENSE")), ref("DELTA-NOTEXT"))
    head = {"H1 DENSE-OFF positive, exact McNemar p < 0.05 and clustered-bootstrap 5th percentile > 0":
                prim["b"] > prim["c"] and prim["p"] < ALPHA and prim["boot_lower"] > 0,
            "H2 DENSE ppl ratio <= 1.05 at the picked gain on every life":
                all(ppl_t[s]["DENSE"][f"{picks['DENSE'][s]['gain']:g}"] <= PPL_MAX for s in seeds),
            "H3 DENSE-SHAM-DENSE positive, Holm p < 0.05 (S1) and clustered-bootstrap 5th percentile > 0":
                fam["S1"]["holds"] and fam["S1"]["boot_lower"] > 0,
            "H4 DENSE not below DELTA-NOTEXT by the margin (exact lower bound > -margin)": nin["lower"] > -delta,
            "H5 refusals not above DELTA-NOTEXT (exact upper bound < 5% of item-lives)": rf["upper"] < REF_MARGIN * N}
    T["headroom_clauses"] = head
    T["noninferiority_vs_notext"], T["refusal_vs_notext"] = nin, rf
    # null decision: equivalence at every admissible gain
    bar, adm_any = {}, {}
    for x in WRITE:
        adm = [g for g in gains if all(ppl_t[s][x][f"{g:g}"] <= PPL_MAX for s in seeds)]
        adm_any[x] = adm
        for g in adm:
            vo = paired(cell(gname(x, g)), cell("OFF"))
            vs = paired(cell(gname(x, g)), cell(gname("SHAM-" + x, g)))
            bar[gname(x, g)] = {"vs_off": vo, "vs_sham": vs,
                                "B1 upper(X-OFF) < margin (exact and bootstrap)": vo["upper"] < delta and vo["boot_upper"] < delta,
                                "B2 upper(X-SHAM) < margin (exact and bootstrap)": vs["upper"] < delta and vs["boot_upper"] < delta}
    T["barrier_cells"], T["admissible_gains"] = bar, adm_any
    bclauses = {"B3 every write arm has an admissible gain": all(adm_any[x] for x in WRITE),
                "B4 no positive finding: H1 false and none of S1-S5 holds":
                    not head[next(k for k in head if k.startswith("H1"))]
                    and not any(fam[k]["holds"] for k in ("S1", "S2", "S3", "S4", "S5"))}
    for k in ("B1 upper(X-OFF) < margin (exact and bootstrap)",
              "B2 upper(X-SHAM) < margin (exact and bootstrap)"):
        bclauses[k] = all(c[k] for c in bar.values()) if bar else False
    T["barrier_clauses"] = bclauses
    if all(head.values()):
        out = "HEADROOM"
    elif all(bclauses.values()):
        out = "BARRIER"
    else:
        out = "NEITHER"
    T["failed"] = [k for k, v in {**head, **bclauses}.items() if not v]
    mods = (["DEV"] if set(seeds) == set(KR_SEEDS) else []) + (["UNANCHORED"] if unanch else []) + \
        (["FALLBACK-KNOWS"] if measure == "knows" else [])
    T["verdict"] = "-".join([out] + mods)
    T["acts"] = not unanch and measure == "ok"
    return T


def load_units(d: str, tag: str = "") -> dict:
    pre = f"u{('-' + tag) if tag else ''}_s"
    by: dict = {}
    for f in sorted(os.listdir(d)):
        if f.startswith(pre) and f.endswith(".json") and not f.endswith(".tmp"):
            m = re.match(rf"{re.escape(pre)}(\d+)_(.+)\.json$", f)
            if m:
                by.setdefault(int(m.group(1)), {})[m.group(2)] = json.load(open(os.path.join(d, f)))
    return by


def main(argv=None, cortex=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--life", required=True)
    r.add_argument("--snaps", required=True)
    r.add_argument("--trunk", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--work", default=None)
    r.add_argument("--model", required=True, help="checkpoint path (the engine passes $BRAIN_MODEL_DIR/<file>.pth)")
    r.add_argument("--device", default="cpu")
    r.add_argument("--mc-ref", default=None)
    r.add_argument("--vc-ref", default=None)
    r.add_argument("--units", default=None, help="comma list of unit names / prefixes to run (smoke); default all")
    r.add_argument("--limit", type=int, default=0, help="first N questions only (smoke; never scored)")
    r.add_argument("--tag", default="")
    r.add_argument("--gains", type=float, nargs="+", default=None,
                   help="gain grid (smoke only: a grid other than GAINS is never scored, the verdict requires every GAINS unit)")
    sc = sub.add_parser("score")
    sc.add_argument("--dir", required=True)
    sc.add_argument("--seeds", type=int, nargs="+", default=list(KR_SEEDS))
    sc.add_argument("--tag", default="")
    sc.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        guard()
        res = json.load(open(a.life))
        if cortex is None:
            from brain.cortex import Cortex, resolve_model
            torch.set_num_threads(os.cpu_count())
            cortex = Cortex(model_path=resolve_model(a.model), device=a.device)
        run(res, a.snaps, a.trunk, cortex, a.out_dir, a.work or os.path.join(a.out_dir, f"work_s{res['seed']}"),
            a.mc_ref, a.vc_ref, a.units, tuple(a.gains) if a.gains else GAINS, a.tag, a.limit)
        return None
    v = verdict(load_units(a.dir, a.tag), tuple(a.seeds))
    if a.out:
        VC._save_json(a.out, v)
    print(json.dumps({k: v.get(k) for k in ("verdict", "why", "failed", "primary", "acts")}, indent=1, default=str))
    return v


if __name__ == "__main__":
    main()
