# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-26-routing-screen3-prereg.md (part J); results/2026-11-25-routing-ceiling-read.md;
#   brain/experiments/jlens.py (J matrices)
# Expected-fail: partly. The J-lens is untested on a recurrent cortex; near-duplicate separation and a paraphrase-robust
#   cue are MISSING (PREREQUISITES.md). Rank only, offline; nothing here is a brain default.
"""J-space rank ceiling on the KR night snapshots (prereg part J): where does the right episode rank under keys read
from the residual stream, with and without the Jacobian lens?

For each night the routing ceiling scored (its JSON gives the rows, cues and right episodes), the snapshot is opened as
routing_replay.open_snapshot opens it (no deltas; keys from keys.pt), and S0 is recomputed on every row: it must give
the ceiling's S0 rank, else the night is void here. Every stored episode's text (+ "\\n\\n") and every cue
("User: {cue}\\n\\n") is read from base through the night's cortex (LiveCortex.plastic, adapters as the brain has them).
Keys at each layer l (12, 18):
  H<l>   mean over positions of the residual h_l, centred cosine (no Jacobian, no readout)
  L<l>   logit lens: mean over positions of softmax(W_U LN_out(h_l,t)), top 25 tokens kept, cosine
  Jt<l>  J-lens: as L<l> with J_l h_l,t
  Jv<l>  centred cosine of J_l (mean h_l)
Every key is a centred cosine on the night's stored mean. Controls (review 11-26): Rv<l>:<seed> = R (mean h_l) with
R a random matrix with J_l's singular values; S_all_rk = S_all_live's math on episodes re-keyed from their text. Each
key is also ranked with permanent episodes left out ("<key>/np"), and the top's permanence is logged.

    python -m brain.experiments.jspace_ceiling run --life L.json --ceil CEIL.json --snaps DIR --J J.pt --out O.json
    python -m brain.experiments.jspace_ceiling table O_s319.json O_s320.json O_s321.json [--out T.json]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
import torch.nn.functional as F

from brain import clock
from brain.experiments import jlens
from brain.experiments.routing_ceiling import centred_cos_one, live_store, rank_of, sall_live

LAYERS = (12, 18)
TOPK = 25
KEYS = tuple(f"{k}{l}" for l in LAYERS for k in ("H", "L", "Jt", "Jv"))
RV_SEEDS = (42, 43, 44)
RV_KEYS = tuple(f"Rv{l}:{s}" for l in LAYERS for s in RV_SEEDS)
H_ITEMS = ("pixel", "admin", "marguerite", "volvo")     # asked every night (the ceiling read's 128 H rows)


def spectrum_controls(J: dict) -> dict:
    """{(l, seed): R} with R = U diag(svdvals(J_l)) V^T, U and V random orthogonal (QR of a seeded Gaussian): a linear map
    with J_l's spectrum and no J_l directions (review 11-26; an orthogonal map alone leaves the centred cosine as H)."""
    out = {}
    for l in LAYERS:
        sv = torch.linalg.svdvals(J[l].double())
        C = sv.shape[0]
        for sd in RV_SEEDS:
            g = torch.Generator().manual_seed(sd)
            U = torch.linalg.qr(torch.randn(C, C, generator=g, dtype=torch.float64))[0]
            V = torch.linalg.qr(torch.randn(C, C, generator=g, dtype=torch.float64))[0]
            out[(l, sd)] = ((U * sv) @ V.T).float()
    return out


def text_reads(p, pipe, base, text: str, J: dict) -> dict:
    """{l: {"h": mean h_l (C,), "Jh": J_l mean h_l, "lp": mean logit-lens probs (V,), "jp": mean J-lens probs}}."""
    dt = p.z["emb.weight"].dtype
    st = [s.to(dt) if i % 3 != 1 else s.float() for i, s in enumerate(base)]
    with torch.no_grad():
        hs, _, _ = jlens.residuals(p, pipe.encode(text), st)
        out = {}
        for l in LAYERS:
            h = hs[l].float()
            Jl = J[l].to(h.device)
            jh = h @ Jl.T                                       # J_l h_l,t for every position t
            out[l] = {"h": h.mean(0).cpu(), "Jh": jh.mean(0).cpu(),
                      "lp": F.softmax(jlens.readout(p, h.to(dt)).float(), -1).mean(0).cpu(),
                      "jp": F.softmax(jlens.readout(p, jh.to(dt)).float(), -1).mean(0).cpu()}
    return out


def topk_sparse(P: torch.Tensor, k: int = TOPK) -> torch.Tensor:
    v, i = torch.topk(P, k, dim=-1)
    out = torch.zeros_like(P)
    return out.scatter(-1, i, v)


def cos_rows(K: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return F.normalize(K, dim=1) @ F.normalize(q, dim=0)


def key_sims(store: list, cue: dict, R: dict) -> dict:
    """Every key is a centred cosine on the night's stored mean (L / Jt too: review 11-26)."""
    sims = {}
    for l in LAYERS:
        H = torch.stack([s[l]["h"] for s in store])
        sims[f"H{l}"] = centred_cos_one(H, cue[l]["h"])
        Jv = torch.stack([s[l]["Jh"] for s in store])
        sims[f"Jv{l}"] = centred_cos_one(Jv, cue[l]["Jh"])
        Lp = torch.stack([topk_sparse(s[l]["lp"]) for s in store])
        sims[f"L{l}"] = centred_cos_one(Lp, topk_sparse(cue[l]["lp"]))
        Jp = torch.stack([topk_sparse(s[l]["jp"]) for s in store])
        sims[f"Jt{l}"] = centred_cos_one(Jp, topk_sparse(cue[l]["jp"]))
        for sd in RV_SEEDS:
            Rm = R[(l, sd)]
            sims[f"Rv{l}:{sd}"] = centred_cos_one(H @ Rm.T, Rm @ cue[l]["h"])
    return sims


def top_tokens(pipe, P: torch.Tensor, k: int = 8) -> list:
    v, i = torch.topk(P, k)
    return [(pipe.decode([int(t)]), round(float(x), 4)) for x, t in zip(v, i)]


def ceiling_night_j(res, cnight, snap, trunk, cortex, work, weekend, J, R) -> dict:
    from brain.eval import life
    from brain.experiments.routing_replay import open_snapshot
    night = cnight["night"]
    sim = life.SimClock()
    clock.set(sim.now)
    home = os.path.join(work, f"s{res['seed']}_n{night}")
    try:
        sim.to(life.day_start(night + 1, weekend))
        b = open_snapshot(snap, home, cortex, res["brain_kw"], trunk=trunk, with_deltas=False)
        try:
            cx, base, eps = b.cortex, b.base, b.hippocampus.episodes
            p = cx.plastic
            if p is None:
                raise RuntimeError(f"night {night}: the snapshot brain has no plastic cortex (no adapters)")
            was = p.enabled
            p.enabled = bool(cx.active)
            try:
                texts = [e.text for e in eps]
                perm = [bool(e.permanent) for e in eps]
                K0 = torch.stack([e.key for e in eps]).float()
                store = [text_reads(p, cx.pipe, base, t + "\n\n", J) for t in texts]
                # S_all_rk (review 11-26): S_all_live's math on every episode re-keyed from its text (all layers)
                blocks = {}
                for t in texts:
                    d = cx.matrix_delta(cx.read(t + "\n\n", base)[1], base)
                    for li, x in enumerate(d):
                        if x is not None and li % 3 == 1:
                            blocks.setdefault(li, []).append(x.flatten().float().cpu())
                live_rk = live_store({li: torch.stack(v) for li, v in blocks.items()})
                del blocks
                rows, insane = [], 0
                for r in cnight["rows"]:
                    if not r["right"]:
                        continue
                    _, st = cx.read(f"User: {r['cue'].strip()}\n\n", base)
                    dl = cx.matrix_delta(st, base)
                    q0 = cx.key_of(dl).cpu()
                    s0 = rank_of(centred_cos_one(K0, q0), r["right"])
                    ok = s0["rank"] == r["keys"]["S0"]["rank"]
                    insane += not ok
                    cue = text_reads(p, cx.pipe, base, f"User: {r['cue'].strip()}\n\n", J)
                    sims = key_sims(store, cue, R)
                    sims["S_all_rk"] = sall_live(live_rk, dl, len(eps))
                    row = {"night": night, "i": r["i"], "item": r["item"], "cue": r["cue"], "right": r["right"],
                           "S0_ok": ok, "keys": {"S0": s0}, "top_perm": {},
                           "ref": {k: r["keys"][k] for k in ("S_all_live", "S12", "S18") if k in r["keys"]},
                           "cue_top": {l: {"L": top_tokens(cx.pipe, cue[l]["lp"]), "Jt": top_tokens(cx.pipe, cue[l]["jp"])}
                                       for l in LAYERS}}
                    for k, s in sims.items():
                        row["keys"][k] = rank_of(s, r["right"])
                        row["top_perm"][k] = perm[int(torch.argmax(s))]
                        snp = s.clone()
                        snp[torch.tensor(perm, dtype=torch.bool)] = -1e9
                        row["keys"][k + "/np"] = rank_of(snp, r["right"])
                    rows.append(row)
            finally:
                p.enabled = was
        finally:
            b.release()
            shutil.rmtree(home, ignore_errors=True)
    finally:
        clock.set(None)
    return {"night": night, "store": len(texts), "n_perm": sum(perm), "rows": rows, "insane_rows": insane,
            "void": insane > 0}


def run(res, ceil, snaps, trunk, cortex, work, J, weekend=True, log=print) -> dict:
    out = {"seed": res["seed"], "layers": list(LAYERS), "topk": TOPK, "rv_seeds": list(RV_SEEDS), "nights": []}
    R = spectrum_controls(J)
    os.makedirs(work, exist_ok=True)
    for n in ceil["nights"]:
        if n.get("void") or not n.get("sane", True) or not n.get("rows"):
            out["nights"].append({"night": n["night"], "void": True, "rows": [], "why": "void in the ceiling"})
            continue
        rec = ceiling_night_j(res, n, os.path.join(snaps, f"n{n['night']}"), trunk, cortex, work, weekend, J, R)
        log(f"s{res['seed']} n{rec['night']}: store {rec['store']} perm {rec['n_perm']} rows {len(rec['rows'])} "
            f"insane {rec['insane_rows']}")
        out["nights"].append(rec)
    return out


def _r1(rows, k):
    return sum(1 for r in rows if (r["keys"].get(k) or r["ref"].get(k) or {}).get("rank") == 1)


def table(outs: list) -> dict:
    """Prereg part J bars (as amended 11-26, before data), per J key at its own layer l:
    J1  pooled r1 >= max(H_l, L_l, Rv_l draws) + 10, and >= that control on >= 2 of 3 seeds
    J1n on the non-H rows, >= max(H_l, L_l) + 5
    J2  pooled >= max(S_all_live, S_all_rk) + 10, and >= S_all_live on >= 2 of 3 seeds
    J2n on the non-H rows, >= S_all_live - 2. "/np" keys never pass. A seed with > 2 void nights here is void."""
    by_seed, void = {}, []
    for o in outs:
        ok = [n for n in o["nights"] if not n.get("void")]
        if sum(1 for n in o["nights"] if n.get("void") and not n.get("why")) > 2:
            void.append(o["seed"])
        by_seed[str(o["seed"])] = [r for n in ok for r in n["rows"]]
    seeds = [s for s in by_seed if int(s) not in void]
    pooled = [r for s in seeds for r in by_seed[s]]
    nonH = [r for r in pooled if r["item"] not in H_ITEMS]
    allk = (list(KEYS) + list(RV_KEYS) + [k + "/np" for k in KEYS] + ["S_all_rk", "S0", "S_all_live", "S12", "S18"])
    r1 = {k: _r1(pooled, k) for k in allk}
    r1n = {k: _r1(nonH, k) for k in allk}
    seed_r1 = {s: {k: _r1(by_seed[s], k) for k in allk} for s in seeds}
    selfcap = {k: sum(1 for r in pooled if r["top_perm"].get(k)) for k in KEYS}
    bars = []
    for l in LAYERS:
        rv = [f"Rv{l}:{sd}" for sd in RV_SEEDS]
        for k in (f"Jt{l}", f"Jv{l}"):
            ctrl = max([r1[f"H{l}"], r1[f"L{l}"]] + [r1[x] for x in rv])
            ctrl_s = {s: max([seed_r1[s][f"H{l}"], seed_r1[s][f"L{l}"]] + [seed_r1[s][x] for x in rv]) for s in seeds}
            j1 = r1[k] >= ctrl + 10 and sum(seed_r1[s][k] >= ctrl_s[s] for s in seeds) >= 2
            j1n = r1n[k] >= max(r1n[f"H{l}"], r1n[f"L{l}"]) + 5
            j2 = (r1[k] >= max(r1["S_all_live"], r1["S_all_rk"]) + 10
                  and sum(seed_r1[s][k] >= seed_r1[s]["S_all_live"] for s in seeds) >= 2)
            j2n = r1n[k] >= r1n["S_all_live"] - 2
            bars.append({"key": k, "r1": r1[k], "r1_nonH": r1n[k], "control": ctrl, "J1": j1, "J1n": j1n,
                         "J2": j2, "J2n": j2n, "pass": j1 and j1n and j2 and j2n})
    other = [k for k in KEYS if k[0] in "HL" and r1[k] >= max(r1["S_all_live"], r1["S_all_rk"]) + 10]
    status = ("VOID" if not seeds else "LEAD" if any(b["pass"] for b in bars) else "KILLED")
    return {"status": status, "void_seeds": void, "n_rows": len(pooled), "n_nonH": len(nonH), "r1": r1,
            "r1_nonH": r1n, "seed_r1": seed_r1, "self_capture_top": selfcap, "j_bars": bars,
            "non_J_keys_clearing_J2": other}


def main(argv=None, cortex=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--life", required=True)
    r.add_argument("--ceil", required=True, help="routing_ceiling run output for this seed")
    r.add_argument("--snaps", required=True)
    r.add_argument("--trunk", required=True)
    r.add_argument("--J", required=True, help="jlens compute output (J.pt)")
    r.add_argument("--model", default=None)
    r.add_argument("--device", default="cpu")
    r.add_argument("--work", default=None)
    r.add_argument("--no-weekend", action="store_true")
    r.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("outs", nargs="+")
    t.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        with open(a.life) as fh:
            res = json.load(fh)
        with open(a.ceil) as fh:
            ceil = json.load(fh)
        if ceil["seed"] != res["seed"]:
            raise SystemExit(f"ceiling seed {ceil['seed']} != life seed {res['seed']}")
        Jd = torch.load(a.J, map_location="cpu", weights_only=False)
        missing = [l for l in LAYERS if l not in Jd["J"]]
        if missing:
            raise SystemExit(f"{a.J} lacks layers {missing}")
        if cortex is None:
            from brain.cortex import Cortex, resolve_model
            cortex = Cortex(model_path=resolve_model(a.model), device=a.device)
        out = run(res, ceil, a.snaps, a.trunk, cortex, a.work or a.out + ".work", Jd["J"], not a.no_weekend)
        out["J_meta"] = Jd["meta"]
        with open(a.out, "w") as fh:
            json.dump(out, fh)
        return out
    outs = []
    for f in a.outs:
        with open(f) as fh:
            outs.append(json.load(fh))
    tab = table(outs)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(tab, fh, indent=1)
    print(json.dumps({k: tab[k] for k in ("status", "void_seeds", "n_rows", "n_nonH", "j_bars", "non_J_keys_clearing_J2",
                                          "self_capture_top")}, indent=1))
    for k, v in tab["r1"].items():
        print(f"{k:14s} r1 {v}  nonH {tab['r1_nonH'][k]}")
    return tab


if __name__ == "__main__":
    main()
