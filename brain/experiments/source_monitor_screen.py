# Capability: Self and identity
# Record: brain/research/2026-11-27-source-monitor-port-prereg-DRAFT.md
# (also: brain/experiments/voice_tag.py (machinery); brain/experiments/results/2026-11-26-source-tag-read.md (the killed heard-side tag))
# Expected-fail: yes, named in the prereg: the tag is host-marked by provenance, the keep bank is the only own-production data in a K-P night, and a fact stored conditional on the tag lowers knows; GO about 10-15%
"""Source-monitor screen (DEV, offline, CPU): the efference-copy tag (brain/source_monitor.py) on KR night snapshots.

Same night, same store, same battery and same rule as brain/experiments/voice_tag.py (night N of a KR life re-slept from
night N-1's slow store, every told item's own probe question asked from a fresh state, hippocampus-off primary). What
differs is where the vector sits. voice_tag put it on "User:" turns; here it sits on what the brain itself produced
(the keep bank's own samples, "Assistant:" turns, tokens fed back after sampling) and never on a told telling.
Arms: A0 (no tag), TAG (learned efference tag), SHAM (fixed random vector, same norm, same mask), ALL (learned vector on
every position: the no-contrast control, report only), LESION (TAG's adapters, tag removed at test), A0snap (banked night).
Rule: voice_tag.table, unchanged, plus report-only TAG-ALL onset and the masked-token counts.
  python -m brain.experiments.source_monitor_screen run --life lifeKR_s<seed>.json --snaps <dir> --trunk <trunk> --out <json> --work <dir>
  python -m brain.experiments.source_monitor_screen table <outs...> --lives <lifeKR...> --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import socket

import torch

from brain import source_monitor as SM
from brain.experiments import voice_tag as VT


def bank_texts(snaps, night):
    """The keep bank the night-N sleep drew from: night N-1's keep_bank.json (copied into the re-sleep home by VT)."""
    path = os.path.join(snaps, f"n{night - 1}", "keep_bank.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: the efference tag needs the life's keep bank (Brain(keep_bank=...)); UNDEFINED")
    blob = json.load(open(path))
    return blob["texts"], blob.get("frame")


def run(res, snaps, trunk, cortex, work, out_path=None, night=VT.NIGHT, log=print, smoke=False):
    os.makedirs(work, exist_ok=True)
    seed = res["seed"]
    qs = VT.told_questions(res, night)
    L = cortex.n_layers // 2
    assert L >= 1, cortex.n_layers
    texts, frame = bank_texts(snaps, night)
    R = {"seed": seed, "host": socket.gethostname(), "device": str(getattr(cortex, "device", "cpu")), "night": night,
         "layer": L, "n_questions": len(qs), "bank": {"n": len(texts), "frame": frame}, "sleeps": {}, "batteries": {},
         "masked": {}}

    def save():
        if out_path:
            json.dump(R, open(out_path + ".tmp", "w"), indent=1)
            os.replace(out_path + ".tmp", out_path)

    def keep(name, rec, prov=None):
        path = os.path.join(work, f"s{seed}_adapters_{name}.pt")
        torch.save({"rank": 8, "layers": None, "adapters": rec["state"]}, path)
        R["sleeps"][name] = {k: v for k, v in rec.items() if k not in ("state", "tag")}
        R["sleeps"][name]["tag_norm"] = None if rec["tag"] is None else float(rec["tag"].norm())
        if prov is not None:
            R["masked"][name] = dict(prov.stats)   # engagement record: tagged vs total tokens over the whole sleep
        return path

    def provenance(kind):
        p = SM.Provenance(kind)
        R["bank"]["registered"] = p.register(cortex.pipe, texts) if kind == "own" else 0
        return p

    a0 = VT.resleep(res, snaps, trunk, cortex, work, "none", night=night, log=log)
    a0_path = keep("A0", a0)
    R["reproduction"] = VT.reproduction(a0["state"], os.path.join(snaps, f"n{night}", "cortex_adapters.pt"))
    log(seed, "reproduction", R["reproduction"])
    save()
    plan = [("A0snap", os.path.join(snaps, f"n{night}", "cortex_adapters.pt"), None, (VT.CONDS[0],), None),
            ("A0", a0_path, None, VT.CONDS, None)]
    own = provenance("own")
    with SM.channel(VT, own):
        tg = VT.resleep(res, snaps, trunk, cortex, work, "learn", layer=L, night=night, log=log)
    tg_path = keep("TAG", tg, own)
    if own.stats["registered_hits"] == 0 or own.stats["tagged"] == 0 or own.stats["tagged"] == own.stats["tokens"]:
        R["masked"]["TAG_degenerate"] = True   # nothing tagged, or everything: no contrast, the screen is UNDEFINED
    plan += [("TAG", tg_path, tg["tag"], VT.CONDS, "own"), ("LESION", tg_path, None, VT.CONDS, "own")]
    if not smoke:
        g = torch.Generator().manual_seed(seed * 1000 + 7)
        d = torch.randn(tg["tag"].shape, generator=g, dtype=torch.float64)
        sham = (d / d.norm() * float(tg["tag"].norm())).to(tg["tag"].dtype)
        sp = provenance("own")
        with SM.channel(VT, sp):
            sh = VT.resleep(res, snaps, trunk, cortex, work, "fixed", vec=sham, layer=L, night=night, log=log)
        sh_path = keep("SHAM", sh, sp)
        ap = provenance("all")
        with SM.channel(VT, ap):
            al = VT.resleep(res, snaps, trunk, cortex, work, "learn", layer=L, night=night, log=log)
        al_path = keep("ALL", al, ap)
        torch.save({"tag": tg["tag"], "sham": sham, "all": al["tag"], "layer": L}, os.path.join(work, f"s{seed}_tags.pt"))
        save()
        with SM.channel(VT, SM.Provenance("own")):
            R["engagement"] = {"TAG": VT.onset_kl(res, snaps, trunk, cortex, work, qs, tg_path, tg["tag"], L, "TAG", night),
                               "SHAM": VT.onset_kl(res, snaps, trunk, cortex, work, qs, sh_path, sham, L, "SHAM", night)}
        with SM.channel(VT, SM.Provenance("all")):
            R["engagement_ALL"] = VT.onset_kl(res, snaps, trunk, cortex, work, qs, al_path, al["tag"], L, "ALL", night)
        save()
        plan += [("SHAM", sh_path, sham, VT.CONDS, "own"), ("ALL", al_path, al["tag"], VT.CONDS, "all")]
    else:
        with SM.channel(VT, SM.Provenance("own")):
            R["engagement"] = {"TAG": VT.onset_kl(res, snaps, trunk, cortex, work, qs, tg_path, tg["tag"], L, "TAG", night),
                               "SHAM": [0.0] * len(qs)}
    for name, path, vec, conds, kind in plan:
        prov = SM.Provenance(kind) if kind else None
        if prov is None:
            R["batteries"][name] = VT.battery(res, snaps, trunk, cortex, work, qs, path, vec, L, name, night, conds)
        else:
            with SM.channel(VT, prov):
                R["batteries"][name] = VT.battery(res, snaps, trunk, cortex, work, qs, path, vec, L, name, night, conds)
        log(seed, name, {c: (v["uv_knows"], v["knows"], v["seconds"]) for c, v in R["batteries"][name].items()})
        save()
    return R


def table(outs, ledgers):
    """voice_tag.table (rule unchanged) plus report-only items: TAG-ALL onset, masked-token shares, degenerate flag."""
    T = VT.table(outs, ledgers)
    if all("ALL" in o["batteries"] for o in outs):
        ons = {a: VT._cells(outs, ledgers, a, "hoff", lambda r, led: r.get("onset")) for a in ("TAG", "ALL")}
        T["onset_TAG-ALL"] = VT._boot_diff(ons["TAG"], ons["ALL"])
        T["engagement_ALL_mean_kl"] = sum(x for o in outs for x in o["engagement_ALL"]) / \
            max(sum(len(o["engagement_ALL"]) for o in outs), 1)
    T["masked_share_TAG"] = {o["seed"]: (o["masked"]["TAG"]["tagged"] / max(o["masked"]["TAG"]["tokens"], 1))
                             for o in outs if "TAG" in o.get("masked", {})}
    degenerate = [o["seed"] for o in outs if o.get("masked", {}).get("TAG_degenerate")]
    if degenerate:
        T["verdict"], T["why"] = "UNDEFINED", f"efference mask tagged none or all tokens on seeds {degenerate}"
    return T


def main(argv=None, cortex=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--life", required=True)
    r.add_argument("--snaps", required=True)
    r.add_argument("--trunk", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--night", type=int, default=VT.NIGHT)
    r.add_argument("--smoke", action="store_true")
    t = sub.add_parser("table")
    t.add_argument("outs", nargs="+")
    t.add_argument("--lives", nargs="+", required=True)
    t.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        res = json.load(open(a.life))
        if cortex is None:
            from brain.cortex import Cortex
            cortex = Cortex(model_path=os.path.join(os.environ["BRAIN_MODEL_DIR"], res["model"] + ".pth"), device="cpu")
        R = run(res, a.snaps, a.trunk, cortex, a.work, out_path=a.out, night=a.night, smoke=a.smoke)
        json.dump(R, open(a.out, "w"), indent=1)
    else:
        outs = [json.load(open(p)) for p in a.outs]
        ledgers = {json.load(open(p))["seed"]: json.load(open(p))["ledger"] for p in a.lives}
        T = table(outs, ledgers)
        print(json.dumps(T, indent=1))
        if a.out:
            json.dump(T, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
