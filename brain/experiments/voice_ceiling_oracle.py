# Capability: Self and identity
# Record: brain/research/2026-11-27-voice-direction-ceiling-prereg.md
# Expected-fail: the oracle direction is the upper bound of the whole one-vector residual-tag class; CLASS-CLOSED (the voice arc stays killed) is a live outcome, and a direction that moves the voice may also lower knows (MIXED)
"""Oracle source-direction ceiling (forward only, CPU, nothing trained, hippocampus off).

Question: can ANY one-vector residual signal on the brain's own-production positions move told-fact voice in the slept
weights without losing the fact? The vector is not learned: it is read off the cortex itself as the residual difference
between the same question asked with the telling in context (C) and without (H), at the label position, averaged over
the OTHER items (leave-one-item-out), and added back on the source-monitor footprint (brain/source_monitor.py
Provenance("own"): the "Assistant:" label, tokens fed back after sampling) through PlasticCortex's tag hook.

Banked n14 adapters (A0snap), fresh state per question, hippocampus off. Arms (all hippocampus off, asked as
brain/experiments/voice_tag.battery asks A0snap):
  H       nothing; must reproduce the A0snap hippocampus-off rows (voice_tag / voice_split), else UNDEFINED
  C       the telling heard first, then the question (voice_split ctx): the reference ceiling
  ORACLE  v_L on the own footprint        SHAM  norm-matched random vector, same footprint
  ALL     v_L on every position           WRONG v built from C-reads whose preceding turn is an unrelated item's telling
Reads taken once per seed: H, C and the WRONG context at every block's residual (last position of "User: q\\n\\nAssistant:").
Rule and verdicts (CLASS-ALIVE / CLASS-CLOSED / MIXED / UNDEFINED): brain/research/2026-11-27-voice-direction-ceiling-prereg.md.
  python -m brain.experiments.voice_ceiling_oracle run --life lifeKR_s<seed>.json --snaps <dir> --trunk <trunk> --out <json> --work <dir>
  python -m brain.experiments.voice_ceiling_oracle table <outs...> --lives <lifeKR...> --out <json>
  python -m brain.experiments.voice_ceiling_oracle handread <outs...> --lives <lifeKR...> --out-prefix <path>
Host guard: `run` refuses unless POOL_NODE=1 or BRAIN_ALLOW_W3090=1."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from brain import clock
from brain import source_monitor as SM
from brain.experiments import cascade_screen as CS
from brain.experiments import voice_split as VS
from brain.experiments import voice_tag as VT

NIGHT = VT.NIGHT
EXPECTED_SEEDS = (319, 320, 321)
REPORT_LAYERS = (6, 18)                       # report only; the primary layer is n_layers // 2
ARMS = ("H", "C", "ORACLE", "SHAM", "ALL", "WRONG")
VEC_ARMS = ("ORACLE", "SHAM", "ALL", "WRONG")
REPORT_ARMS = ("ORACLE", "SHAM")              # run at the report layers (override with --report-arms)
BOOT, BOOT_SEED = 2000, 20261128
REPRO_MIN = 0.9                               # share of H replies identical to the A0snap hippocampus-off reference
ITEMS_PRIMARY = ("pixel", "marguerite")       # must individually move; volvo is report only
ITEMS_REPORT = ("pixel", "marguerite", "volvo")
ONSET_TOL = 1e-4                              # |onset_H - onset_ref| below this counts as the same read
NOOP_TOL, NOOP_MAX = 1e-6, 0.5                # an arm whose onset equals H's on more than half the rows did nothing
REQUIRE_ACTIVE = True                         # the banked adapters must be live, else a tag never applies (tests turn off)


# ------------------------------------------------------------------ guard, provenance, atomic writes
def guard(env=None):
    env = os.environ if env is None else env
    if env.get("POOL_NODE") != "1" and env.get("BRAIN_ALLOW_W3090") != "1":
        raise SystemExit("refusing to run: set POOL_NODE=1 (pool node) or BRAIN_ALLOW_W3090=1 (the w3090 node)")


def commit() -> str:
    if os.environ.get("BRAIN_COMMIT"):
        return os.environ["BRAIN_COMMIT"]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        return open(os.path.join(root, "COMMIT")).read().strip()
    except OSError:
        pass
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _save_pt(path, obj):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _fp(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ the residual read
@contextlib.contextmanager
def grab_residuals(n_layers: int, sink: list):
    """While active, record the last position's residual after every block of PlasticCortex.forward.
    The forward calls F.layer_norm twice per block (ln1, ln2) and once at the end: the input of call 2*(j+1) is the
    residual after block j (the last, j = n_layers-1, is the ln_out input). No change to PlasticCortex."""
    orig, n = F.layer_norm, [0]

    def ln(x, *a, **k):
        i = n[0]
        n[0] += 1
        if i >= 2 and i % 2 == 0:
            sink.append(x[-1].detach().clone())
        return orig(x, *a, **k)
    F.layer_norm = ln
    try:
        yield
    finally:
        F.layer_norm = orig
    if n[0] != 2 * n_layers + 1:
        raise RuntimeError(f"residual capture saw {n[0]} layer_norm calls, expected {2 * n_layers + 1}")


def capture_label(b, text: str, state) -> torch.Tensor:
    """(n_layers, C): the residual after every block at the last position of `text`, read from `state` through the
    brain's live cortex (the consolidated adapters on). No tag attached (checked)."""
    p = b._plastic()
    if p.source_tag is not None:
        raise RuntimeError("capture with a tag attached")
    if REQUIRE_ACTIVE and not b.cortex.active:
        raise RuntimeError("the cortex is not running through the consolidated adapters (nothing banked, or lesioned)")
    sink: list = []
    with torch.no_grad(), grab_residuals(p.L, sink):
        b.cortex.read(text, state)
    if len(sink) != p.L:
        raise RuntimeError(f"captured {len(sink)} residuals for {p.L} layers")
    return torch.stack(sink)


def set_vector(b, vec, layer: int, prov: SM.Provenance):
    SM.attach(b._plastic(), vec, layer, False, prov)


def clear_vector(b):
    p = b._plastic()
    if p.source_tag is not None:
        SM.detach(p)


# ------------------------------------------------------------------ vectors
def wrong_map(items) -> dict:
    """item -> a different item whose telling is read as the unrelated preceding turn (the next item in sorted order)."""
    its = sorted(set(items))
    if len(its) < 3:
        raise ValueError(f"WRONG needs at least 3 told items, got {len(its)}")
    return {it: its[(i + 1) % len(its)] for i, it in enumerate(its)}


def loo_means(per_item: dict, drop_of=None) -> dict:
    """{item: (n_layers, C) mean over the OTHER items} where per_item[item] is the item's (n_layers, C) mean difference.
    drop_of(item) -> extra items excluded for that item (WRONG: the items whose unrelated turn WAS this item's telling)."""
    out = {}
    for it in per_item:
        skip = {it} | set(drop_of(it) if drop_of else ())
        rest = [v for k, v in per_item.items() if k not in skip]
        if not rest:
            raise ValueError(f"no other item to average for {it}")
        out[it] = torch.stack(rest).mean(0)
    return out


def item_means(qs, tensors) -> dict:
    """Per-item mean of the per-question (n_layers, C) tensors (items weighted equally downstream)."""
    d: dict = {}
    for q, t in zip(qs, tensors):
        d.setdefault(q["item"], []).append(t)
    return {k: torch.stack(v).mean(0) for k, v in d.items()}


def sham_vector(seed: int, qi: int, layer: int, like: torch.Tensor) -> torch.Tensor:
    """A random direction with exactly like's norm (per question, deterministic)."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + qi * 101 + layer)
    d = torch.randn(like.shape, generator=g, dtype=torch.float64)
    return (d / d.norm() * float(like.double().norm())).to(like.dtype)


# ------------------------------------------------------------------ one opened brain
@contextlib.contextmanager
def session(res, snaps, trunk, cortex, work, night, tag, adapters=None):
    """The banked night-N brain (A0snap adapters unless `adapters`), fresh clock, hippocampus off. Same open order as
    voice_tag.battery."""
    from brain.eval import life
    seed = res["seed"]
    cur = os.path.join(snaps, f"n{night}")
    sim = life.SimClock()
    clock.set(sim.now)
    home = os.path.join(work, f"s{seed}_{tag}")
    try:
        sim.to(life.day_start(night + 1, True))
        life._reseed(seed, 500 + night)
        b = VT._open(cur, {"cortex_adapters.pt": adapters} if adapters else {}, home, cortex, dict(res["brain_kw"]), trunk)
        try:
            b.hippocampus.FREEZE_USE = False
            b.cortex.refresh()
            b.set_lesions(hippocampus_off=True, consolidated_off=False)
            if getattr(b, "abstain", False):
                raise RuntimeError("the abstain gate is on: hear() could route off the consolidated synapses")
            if REQUIRE_ACTIVE and not b.cortex.active:
                raise RuntimeError("the cortex is not running through the consolidated adapters (nothing banked, or lesioned)")
            yield b
        finally:
            b.release()
            shutil.rmtree(home, ignore_errors=True)
    finally:
        clock.set(None)


def _telling(res, q):
    return (res["ledger"].get(q["item"]) or {}).get("telling")


def collect_reads(res, snaps, trunk, cortex, work, qs, night=NIGHT, log=print):
    """H, C and WRONG-context residuals for every question: {"H","C","W": [(n_layers, C) per question], ...}."""
    from brain.eval import life
    seed = res["seed"]
    wm = wrong_map([q["item"] for q in qs])
    H, Cc, W = [], [], []
    t0 = time.time()
    with session(res, snaps, trunk, cortex, work, night, "reads") as b:
        base = [x.clone() for x in b.base]
        for q in qs:
            text = f"User: {q['text']}\n\n{b._asst}"
            tell, wtell = _telling(res, q), (res["ledger"].get(wm[q["item"]]) or {}).get("telling")
            if not tell or not wtell:
                raise ValueError(f"item {q['item']} or {wm[q['item']]} has no telling in the ledger")
            H.append(capture_label(b, text, [x.clone() for x in base]))
            for turn, dst in ((tell, Cc), (wtell, W)):
                b.state = [x.clone() for x in base]
                life._reseed(seed, 500 + night)
                b.hear(turn)
                dst.append(capture_label(b, text, b.state))
            b.state = [x.clone() for x in base]
    log(seed, "reads", len(qs), round(time.time() - t0, 1), "s")
    return {"H": H, "C": Cc, "W": W, "wrong_of": wm, "seconds": round(time.time() - t0, 1)}


def build_vectors(qs, reads) -> dict:
    """Leave-one-item-out vectors for every layer: {"oracle": {item: (n_layers, C)}, "wrong": {...}} + norms/cosines."""
    dC = item_means(qs, [c - h for c, h in zip(reads["C"], reads["H"])])
    dW = item_means(qs, [w - h for w, h in zip(reads["W"], reads["H"])])
    wm = reads["wrong_of"]
    oracle = loo_means(dC)
    wrong = loo_means(dW, drop_of=lambda it: [k for k, w in wm.items() if w == it])
    return {"oracle": oracle, "wrong": wrong, "dC": dC, "dW": dW}


def vector_summary(vecs, layers) -> dict:
    out = {}
    for L in layers:
        no = [float(v[L].norm()) for v in vecs["oracle"].values()]
        nw = [float(v[L].norm()) for v in vecs["wrong"].values()]
        cos = [float(F.cosine_similarity(vecs["oracle"][k][L], vecs["wrong"][k][L], dim=0)) for k in vecs["oracle"]]
        out[str(L)] = {"oracle_norm_mean": sum(no) / len(no), "wrong_norm_mean": sum(nw) / len(nw),
                       "cos_oracle_wrong_mean": sum(cos) / len(cos)}
    return out


# ------------------------------------------------------------------ batteries
def _onset(b, q, state, yid, mid):
    with torch.no_grad():
        lg, _ = b.cortex.read(f"User: {q['text']}\n\n{b._asst}", [x.clone() for x in state])
        lp = torch.log_softmax(lg.float(), -1)
        return float(torch.logsumexp(lp[yid], 0) - lp[mid])


def _label_tagged(b, kind: str, text: str) -> bool:
    """Would the footprint tag the `Assistant:` label position (the last token of the question prompt)? Probed on a
    throwaway Provenance so the arm's engagement stats stay clean."""
    return bool(SM.Provenance(kind).mask(b.cortex.pipe, b.cortex.pipe.encode(text))[-1])


def battery(res, snaps, trunk, cortex, work, qs, name, mode="B", vec_fn=None, layer=None, kind=None, night=NIGHT):
    """Ask every told question (fresh state each, hippocampus off). mode "B": the plain question (H); "ctx": the item's
    telling heard first (C). vec_fn(i, q) -> vector or None, applied at `layer` with Provenance(kind), only mode "B"."""
    from brain.eval import life
    seed, led = res["seed"], res["ledger"]
    prov = SM.Provenance(kind) if kind else None
    t0 = time.time()
    with session(res, snaps, trunk, cortex, work, night, f"b_{name}") as b:
        base = [x.clone() for x in b.base]
        yid, mid = VT._onset_ids(b.cortex.pipe)
        rows, norms = [], []
        if prov is not None and REQUIRE_ACTIVE and not b.cortex.active:
            raise RuntimeError("vector arm on an inactive cortex")
        for i, q in enumerate(qs):
            b.state = [x.clone() for x in base]
            told_first = False
            if mode == "ctx":
                tell = _telling(res, q)
                if tell:
                    life._reseed(seed, 500 + night)
                    b.hear(tell)
                    told_first = True
            v = vec_fn(i, q) if vec_fn else None
            lt = vn = None
            if v is not None:
                lt = _label_tagged(b, kind, f"User: {q['text']}\n\n{b._asst}")
                set_vector(b, v, layer, prov)
                vn = float(v.norm())
                norms.append(vn)
            onset = _onset(b, q, b.state, yid, mid)
            life._reseed(seed, 500 + night)             # paired draws: every arm samples each reply from the same RNG state
            r = life.run_battery(b, [q], led, night)[0]
            rows.append({**{k: r.get(k) for k in VS.KEEP}, "onset": onset, "told_first": told_first,
                         "label_tagged": lt, "vec_norm": vn})
            if v is not None:
                clear_vector(b)
    return {"rows": rows, **VS.voice(rows, led), "seconds": round(time.time() - t0, 1),
            "mask": None if prov is None else dict(prov.stats), "vec_norm_mean": (sum(norms) / len(norms)) if norms else None}


def _vec_fn(arm, seed, vecs, layer):
    """The per-question vector of a vector arm at `layer` (leave-one-item-out: the question's own item is never in it)."""
    def fn(i, q):
        o = vecs["oracle"][q["item"]][layer]
        if arm in ("ORACLE", "ALL"):
            return o
        if arm == "SHAM":
            return sham_vector(seed, i, layer, o)
        if arm == "WRONG":
            return vecs["wrong"][q["item"]][layer]
        raise ValueError(arm)
    return fn


# ------------------------------------------------------------------ reproduction
def load_ref_rows(path, night=NIGHT):
    """(kind, rows): the A0snap hippocampus-off rows of a voice_tag output ("vt"), or the hippocampus-off cond-B rows of
    a voice_split output ("vs")."""
    j = json.load(open(path))
    if "batteries" in j:
        return "vt", j["batteries"]["A0snap"]["hoff"]["rows"]
    if "nights" in j:
        n = next(x for x in j["nights"] if x["night"] == night)
        return "vs", n["arms"]["hippocampus-off"]["B"]["rows"]
    raise ValueError(f"{path}: neither a voice_tag nor a voice_split output")


def reproduce(rows, ref_rows, ledger=None) -> dict:
    """Share of H rows whose reply (and onset) is identical to the reference row asked the same question. `n` is the
    number of questions matched; the gate (repro_ok) also needs n to cover the questions."""
    ref = {r["you"]: r for r in ref_rows}
    pairs = [(r, ref[r["you"]]) for r in rows if r["you"] in ref]
    same = sum(a["brain"] == b["brain"] for a, b in pairs)
    osame = sum(a.get("onset") is not None and b.get("onset") is not None and abs(a["onset"] - b["onset"]) < ONSET_TOL
                for a, b in pairs)
    out = {"n": len(pairs), "same": same, "frac": (same / len(pairs)) if pairs else None,
           "onset_same": osame, "onset_frac": (osame / len(pairs)) if pairs else None}
    if ledger is not None:
        refd = [r for r in ref_rows if r["you"] in {x["you"] for x in rows}]
        out["ref_knows"] = sum(VS.knows(r, ledger) for r in refd if CS.is_told(r, ledger))
        out["ref_uv_knows"] = sum(VS.knows(r, ledger) and CS.own2(r["brain"]) for r in refd if CS.is_told(r, ledger))
    return out


def repro_ok(r, n_questions) -> bool:
    """H reproduction gate: the reference covers at least REPRO_MIN of the questions, and replies AND onsets match on
    at least REPRO_MIN of the matched rows."""
    return bool(r and n_questions and r.get("n") is not None and r["n"] >= REPRO_MIN * n_questions
                and r.get("frac") is not None and r["frac"] >= REPRO_MIN
                and r.get("onset_frac") is not None and r["onset_frac"] >= REPRO_MIN)


# ------------------------------------------------------------------ run (resumable per unit)
def run(res, snaps, trunk, cortex, work, out_path=None, night=NIGHT, report_layers=REPORT_LAYERS,
        report_arms=REPORT_ARMS, refs=(), log=print):
    os.makedirs(work, exist_ok=True)
    seed = res["seed"]
    qs = VT.told_questions(res, night)
    nl = cortex.n_layers
    L = nl // 2
    assert L >= 1, nl
    rep = [x for x in report_layers if x != L and 0 <= x < nl]
    layers = [L] + rep
    spec = _fp(seed, night, [q["text"] for q in qs], layers, commit())
    R = {"seed": seed, "host": socket.gethostname(), "commit": commit(), "device": str(getattr(cortex, "device", "cpu")),
         "night": night, "layer": L, "report_layers": rep, "skipped_layers": [x for x in report_layers if x not in rep and x != L],
         "n_questions": len(qs), "spec": spec, "model": res.get("model"), "batteries": {}, "units": {}}

    def save():
        if out_path:
            _save_json(out_path, R)

    def unit(name, fn):
        path = os.path.join(work, f"unit_s{seed}_{name}.json")
        if os.path.exists(path):
            u = json.load(open(path))
            if u.get("spec") == spec:
                log(seed, name, "resumed")
                return u["rec"]
        rec = fn()
        _save_json(path, {"spec": spec, "rec": rec})
        return rec

    rp = os.path.join(work, f"reads_s{seed}.pt")
    reads = None
    if os.path.exists(rp):
        blob = torch.load(rp, map_location="cpu", weights_only=False)
        if blob.get("spec") == spec:
            reads = blob["reads"]
            log(seed, "reads resumed")
    if reads is None:
        reads = collect_reads(res, snaps, trunk, cortex, work, qs, night, log)
        _save_pt(rp, {"spec": spec, "reads": reads})
    vecs = build_vectors(qs, reads)
    R["vectors"] = vector_summary(vecs, layers)
    R["reads_seconds"] = reads["seconds"]
    save()

    def go(arm, layer):
        name = arm if layer is None else f"{arm}@{layer}"
        if arm == "H":
            rec = unit(name, lambda: battery(res, snaps, trunk, cortex, work, qs, name, "B", night=night))
        elif arm == "C":
            rec = unit(name, lambda: battery(res, snaps, trunk, cortex, work, qs, name, "ctx", night=night))
        else:
            kind = "all" if arm == "ALL" else "own"
            rec = unit(name, lambda: battery(res, snaps, trunk, cortex, work, qs, name, "B",
                                             _vec_fn(arm, seed, vecs, layer), layer, kind, night))
        R["batteries"][name] = rec
        log(seed, name, {k: rec[k] for k in ("knows", "uv_knows", "seconds")})
        save()
        return rec

    go("H", None)
    go("C", None)
    for arm in VEC_ARMS:
        go(arm, L)
    for layer in rep:
        for arm in report_arms:
            go(arm, layer)
    R["reproduction"] = {}
    for p in refs:
        kind, rows = load_ref_rows(p, night)
        R["reproduction"][kind] = {**reproduce(R["batteries"]["H"]["rows"], rows, res["ledger"]), "ref": os.path.basename(p)}
    save()
    return R


# ------------------------------------------------------------------ scoring (rule: the prereg, fixed before data)
def _told(o, name, led):
    return [r for r in o["batteries"][name]["rows"] if r.get("item") and CS.is_told(r, led)]


def _stats(rows, led) -> dict:
    kn = [r for r in rows if VS.knows(r, led)]
    uv = sum(CS.own2(r["brain"]) for r in kn)
    return {"told": len(rows), "knows": len(kn), "uv_knows": uv, "p_uv": (uv / len(kn)) if kn else None,
            "refusal": sum(bool(CS.refusal(r["brain"])) for r in rows)}


def _pool(outs, ledgers, name, items=None, seeds=None):
    rows = []
    for o in outs:
        if seeds is not None and o["seed"] not in seeds:
            continue
        rows += [r for r in _told(o, name, ledgers[o["seed"]]) if items is None or r["item"] in items]
    return rows


def _stats_pool(outs, ledgers, name, items=None, seeds=None):
    """Pooled stats; knows is judged against each seed's own ledger."""
    t = {"told": 0, "knows": 0, "uv_knows": 0, "refusal": 0}
    for o in outs:
        if seeds is not None and o["seed"] not in seeds:
            continue
        rows = [r for r in _told(o, name, ledgers[o["seed"]]) if items is None or r["item"] in items]
        s = _stats(rows, ledgers[o["seed"]])
        for k in t:
            t[k] += s[k]
    t["p_uv"] = (t["uv_knows"] / t["knows"]) if t["knows"] else None
    return t


def _cells(outs, ledgers, name, f):
    """{(seed, item, question): f(row, ledger)} over told rows, None dropped."""
    c = {}
    for o in outs:
        led = ledgers[o["seed"]]
        for r in _told(o, name, led):
            v = f(r, led)
            if v is not None:
                c[(o["seed"], r["item"], r["you"])] = v
    return c


def _row_onset(r, led):
    return r.get("onset")


def boot_diff(a: dict, b: dict, draws=BOOT, seed=BOOT_SEED) -> dict:
    """Paired mean difference a - b over common cells; 95% percentile CI of an ITEM-clustered bootstrap (resample item
    names, every seed's cells of a drawn item come with it)."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"diff": None, "ci": None, "n": 0}
    by: dict = {}
    for k in keys:
        s = by.setdefault(k[1], [0.0, 0])
        s[0] += a[k] - b[k]
        s[1] += 1
    cl = list(by.values())
    rng = random.Random(seed)
    ms = []
    for _ in range(draws):
        pick = [cl[rng.randrange(len(cl))] for _ in cl]
        ms.append(sum(p[0] for p in pick) / sum(p[1] for p in pick))
    ms.sort()
    tot = sum(p[0] for p in cl) / sum(p[1] for p in cl)
    return {"diff": tot, "ci": [ms[int(0.025 * draws)], ms[int(0.975 * draws) - 1]], "n": len(keys), "clusters": len(cl)}


def boot_p_uv(outs, ledgers, a, b, draws=BOOT, seed=BOOT_SEED) -> dict:
    """p_uv(a) - p_uv(b) (P(user voice | knows)), item-clustered bootstrap; report only (the bar is a point estimate)."""
    per: dict = {}
    for o in outs:
        led = ledgers[o["seed"]]
        for tag, name in (("a", a), ("b", b)):
            for r in _told(o, name, led):
                x = per.setdefault(r["item"], {"a": [0, 0], "b": [0, 0]})[tag]
                if VS.knows(r, led):
                    x[1] += 1
                    x[0] += CS.own2(r["brain"])
    cl = list(per.values())

    def d(pick):
        ra = sum(p["a"][1] for p in pick)
        rb = sum(p["b"][1] for p in pick)
        if not ra or not rb:
            return None
        return sum(p["a"][0] for p in pick) / ra - sum(p["b"][0] for p in pick) / rb
    rng = random.Random(seed)
    ms = sorted(x for x in (d([cl[rng.randrange(len(cl))] for _ in cl]) for _ in range(draws)) if x is not None)
    return {"diff": d(cl), "ci": [ms[int(0.025 * len(ms))], ms[int(0.975 * len(ms)) - 1]] if ms else None}


def arm_clauses(outs, ledgers, L, seeds_expected) -> dict:
    """Every CLASS-ALIVE clause for ORACLE at layer L against H, SHAM, ALL, WRONG (prereg section 4)."""
    O, S, A, W = (f"{a}@{L}" for a in ("ORACLE", "SHAM", "ALL", "WRONG"))
    ons = {n: _cells(outs, ledgers, n, _row_onset) for n in ("H", O, S, A, W)}
    c = {"onset_ORACLE-H": boot_diff(ons[O], ons["H"]), "onset_ORACLE-SHAM": boot_diff(ons[O], ons[S]),
         "onset_ORACLE-ALL": boot_diff(ons[O], ons[A]), "onset_ORACLE-WRONG": boot_diff(ons[O], ons[W])}
    h, o = _stats_pool(outs, ledgers, "H"), _stats_pool(outs, ledgers, O)
    c["H"], c["ORACLE"] = h, o
    c["SHAM"], c["ALL"], c["WRONG"] = (_stats_pool(outs, ledgers, n) for n in (S, A, W))
    lb = lambda k: c[k]["ci"] is not None and c[k]["ci"][0] > 0
    ok = {"onset": lb("onset_ORACLE-H"), "sham": lb("onset_ORACLE-SHAM"), "all": lb("onset_ORACLE-ALL"),
          "wrong": lb("onset_ORACLE-WRONG")}
    ok["voice"] = o["p_uv"] is not None and h["p_uv"] is not None and o["p_uv"] <= h["p_uv"] - 0.10
    ok["voice_sham"] = (o["p_uv"] is not None and c["SHAM"]["p_uv"] is not None
                        and o["p_uv"] <= c["SHAM"]["p_uv"] - 0.10)     # the voice control: random push of equal norm
    ok["knows"] = h["knows"] > 0 and o["knows"] >= 0.9 * h["knows"]
    ok["refusals"] = (o["refusal"] - h["refusal"]) / max(h["told"], 1) < 0.10
    per_seed = {}
    for s in seeds_expected:
        d = boot_diff(*(_cells([x for x in outs if x["seed"] == s], ledgers, n, _row_onset) for n in (O, "H")), draws=1)
        pu_h = _stats_pool(outs, ledgers, "H", seeds={s})["p_uv"]
        pu_o = _stats_pool(outs, ledgers, O, seeds={s})["p_uv"]
        per_seed[s] = {"onset_diff": d["diff"], "p_uv_H": pu_h, "p_uv_ORACLE": pu_o,
                       "holds": d["diff"] is not None and d["diff"] > 0 and pu_h is not None and pu_o is not None
                       and pu_o < pu_h}
    c["per_seed"] = per_seed
    ok["seeds"] = sum(v["holds"] for v in per_seed.values()) >= 2
    per_item = {}
    for it in ITEMS_REPORT:
        d = boot_diff({k: v for k, v in ons[O].items() if k[1] == it}, {k: v for k, v in ons["H"].items() if k[1] == it},
                      draws=1)
        sh, so = _stats_pool(outs, ledgers, "H", items={it}), _stats_pool(outs, ledgers, O, items={it})
        per_item[it] = {"onset_diff": d["diff"], "n": d["n"], "p_uv_H": sh["p_uv"], "p_uv_ORACLE": so["p_uv"],
                        "knows_H": sh["knows"], "knows_ORACLE": so["knows"],
                        "holds": (None if sh["p_uv"] is None or so["p_uv"] is None else
                                  (d["diff"] is not None and d["diff"] > 0 and so["p_uv"] < sh["p_uv"]))}
    c["per_item"] = per_item
    # a primary item with no knows rows in an arm is "not evaluable" (reported), not a failure of direction
    c["items_evaluable"] = [it for it in ITEMS_PRIMARY if per_item[it]["holds"] is not None]
    ok["items"] = bool(c["items_evaluable"]) and all(per_item[it]["holds"] is not False for it in ITEMS_PRIMARY)
    c["item_rows"] = {it: {"n_told_cells": per_item[it]["n"], "knows_H": per_item[it]["knows_H"],
                           "knows_ORACLE": per_item[it]["knows_ORACLE"]} for it in ITEMS_REPORT}
    c["ok"] = ok
    return c


def table(outs, ledgers, expected=EXPECTED_SEEDS, refs=None) -> dict:
    """Verdict rule (prereg section 4): UNDEFINED > CLASS-CLOSED > CLASS-ALIVE > MIXED."""
    T = {"seeds": sorted(o["seed"] for o in outs)}
    if len(set(T["seeds"])) != len(T["seeds"]):
        T["verdict"], T["why"] = "UNDEFINED", f"duplicate seeds {T['seeds']}"
        return T
    for k in ("commit", "night"):
        vals = {str(o.get(k)) for o in outs}
        if len(vals) != 1:
            T["verdict"], T["why"] = "UNDEFINED", f"outputs disagree on {k}: {sorted(vals)}"
            return T
    refs = refs or {}
    repro, bad = {}, []
    for o in outs:
        r = (o.get("reproduction") or {}).get("vt") or refs.get(o["seed"])
        repro[o["seed"]] = r
        if not repro_ok(r, o.get("n_questions")):
            bad.append(o["seed"])
    T["reproduction"] = repro
    T["reproduction_vs_report"] = {o["seed"]: (o.get("reproduction") or {}).get("vs") for o in outs}
    prim = {o.get("layer") for o in outs}
    if len(prim) != 1 or None in prim:
        T["verdict"], T["why"] = "UNDEFINED", f"seeds disagree on the primary layer {sorted(map(str, prim))}"
        return T
    L = prim.pop()
    T["primary_layer"] = L
    missing = [s for s in expected if s not in T["seeds"]]
    names = [f"{a}@{L}" for a in VEC_ARMS] + ["H", "C"]
    absent = [(o["seed"], n) for o in outs for n in names if n not in o["batteries"]]
    deg = [o["seed"] for o in outs if any(
        (m := (o["batteries"].get(f"{a}@{L}") or {}).get("mask")) is None or m["tagged"] == 0
        or (a != "ALL" and m["tagged"] == m["tokens"])
        or any(r.get("label_tagged") is not True for r in o["batteries"][f"{a}@{L}"]["rows"])      # label never tagged
        for a in ("ORACLE", "SHAM", "WRONG", "ALL"))]
    if missing or absent:
        T["verdict"], T["why"] = "UNDEFINED", f"seeds missing {missing}; batteries missing {absent}"
        return T
    noop = []
    for o in outs:
        one = [o]
        for a in ("ORACLE", "SHAM", "WRONG", "ALL"):
            x, h = _cells(one, ledgers, f"{a}@{L}", _row_onset), _cells(one, ledgers, "H", _row_onset)
            sh = [abs(x[k] - h[k]) for k in x if k in h]
            if not sh or sum(d < NOOP_TOL for d in sh) / len(sh) > NOOP_MAX:
                noop.append((o["seed"], a))
    T["noop_arms"] = noop
    c = arm_clauses(outs, ledgers, L, list(expected))
    T["clauses"] = c
    T["C_moves_onset"] = boot_diff(_cells(outs, ledgers, "C", _row_onset), _cells(outs, ledgers, "H", _row_onset))
    T["C"] = _stats_pool(outs, ledgers, "C")
    T["p_uv_ORACLE-H"] = boot_p_uv(outs, ledgers, f"ORACLE@{L}", "H")
    T["report_layers"] = {}
    for rl in sorted({x for o in outs for x in o.get("report_layers", [])}):
        if all(f"ORACLE@{rl}" in o["batteries"] and f"SHAM@{rl}" in o["batteries"] for o in outs):
            T["report_layers"][str(rl)] = {
                "onset_ORACLE-H": boot_diff(_cells(outs, ledgers, f"ORACLE@{rl}", _row_onset), _cells(outs, ledgers, "H", _row_onset)),
                "onset_ORACLE-SHAM": boot_diff(_cells(outs, ledgers, f"ORACLE@{rl}", _row_onset), _cells(outs, ledgers, f"SHAM@{rl}", _row_onset)),
                "ORACLE": _stats_pool(outs, ledgers, f"ORACLE@{rl}")}
    if bad or c["H"]["knows"] < 10 or deg or noop:
        T["verdict"] = "UNDEFINED"
        T["why"] = (f"H reproduction (coverage, replies, onsets >= {REPRO_MIN}) failed or missing on {bad}; H knows {c['H']['knows']} "
                    f"(< 10 = undefined); footprint degenerate on {deg}; arms whose onset equals H on over half the rows {noop}")
        return T
    ok = c["ok"]
    failed = [k for k, v in ok.items() if not v]
    T["failed"] = failed
    if not ok["onset"] or not ok["voice"]:
        T["verdict"], T["why"] = "CLASS-CLOSED", f"the oracle direction fails {[k for k in ('onset', 'voice') if not ok[k]]}: no residual tag is built at 1.5B"
    elif not failed:
        T["verdict"], T["why"] = "CLASS-ALIVE", "every clause met"
    else:
        T["verdict"], T["why"] = "MIXED", f"onset and voice move; clauses failed: {failed}"
    return T


# ------------------------------------------------------------------ blind hand-read
def handread(outs, ledgers, n=40, seed=BOOT_SEED, arms=None):
    """40 pooled told rows from H, C and ORACLE (primary layer), shuffled, arm labels in a separate key."""
    L = outs[0]["layer"]
    arms = arms or ("H", "C", f"ORACLE@{L}")
    pool = [(o["seed"], a, r) for o in outs for a in arms for r in _told(o, a, ledgers[o["seed"]])]
    rng = random.Random(seed)
    rng.shuffle(pool)
    pick = pool[:n]
    blind = [{"id": i, "question": r["you"], "reply": r["brain"]} for i, (_, _, r) in enumerate(pick)]
    key = [{"id": i, "seed": s, "arm": a, "item": r["item"]} for i, (s, a, r) in enumerate(pick)]
    return blind, key


def main(argv=None, cortex=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--life", required=True)
    r.add_argument("--snaps", required=True)
    r.add_argument("--trunk", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--night", type=int, default=NIGHT)
    r.add_argument("--ref", nargs="*", default=[], help="voice_tag and/or voice_split output(s) of the same seed (H reproduction)")
    r.add_argument("--report-layers", type=int, nargs="*", default=list(REPORT_LAYERS))
    r.add_argument("--report-arms", nargs="*", default=list(REPORT_ARMS), choices=VEC_ARMS)
    t = sub.add_parser("table")
    t.add_argument("outs", nargs="+")
    t.add_argument("--lives", nargs="+", required=True)
    t.add_argument("--out")
    h = sub.add_parser("handread")
    h.add_argument("outs", nargs="+")
    h.add_argument("--lives", nargs="+", required=True)
    h.add_argument("--out-prefix", required=True)
    h.add_argument("--n", type=int, default=40)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        guard()
        res = json.load(open(a.life))
        if cortex is None:
            from brain.cortex import Cortex
            torch.set_num_threads(os.cpu_count())
            cortex = Cortex(model_path=os.path.join(os.environ["BRAIN_MODEL_DIR"], res["model"] + ".pth"), device="cpu")
        run(res, a.snaps, a.trunk, cortex, a.work, out_path=a.out, night=a.night, report_layers=tuple(a.report_layers),
            report_arms=tuple(a.report_arms), refs=[p for p in a.ref if os.path.exists(p)])
        return
    outs = [json.load(open(p)) for p in a.outs]
    ledgers = {}
    for p in a.lives:
        lr = json.load(open(p))
        ledgers[lr["seed"]] = lr["ledger"]
    if a.cmd == "table":
        T = table(outs, ledgers)
        print(json.dumps(T, indent=1, default=str))
        if a.out:
            _save_json(a.out, T)
    else:
        blind, key = handread(outs, ledgers, a.n)
        _save_json(a.out_prefix + "_blind.json", blind)
        _save_json(a.out_prefix + "_key.json", key)


if __name__ == "__main__":
    main()
