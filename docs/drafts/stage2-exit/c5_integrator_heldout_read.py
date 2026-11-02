"""C5 integrator decision memo (DRAFT, 2026-11-01): a read-only re-score of the stable-recal held-out battery.

The battery (RULING canary-recal-1028; lines research/coordination/stable-recal-pool-lines.txt; revision F = 34a4e5f75)
ran the Stage-2 runner's `intact_a` arm of all five groups on dev seeds 1-12 under three settings: `off` (production
integrator), `stable` (SIM_STABLE_INTEGRATION=1) and `stable_recal` (SIM_STABLE_INTEGRATION=1 SIM_STABLE_RECAL=1),
wrapped by research/runners/membrane_census.py, which embeds the full Stage-2 shard at `workload.shard`. This script
scores each embedded shard with the REGISTERED Stage-2 scorers (`SCORERS[group]`, the correctness clause only: no
lesion, no null, no permuted arm exists in this battery) and compares each setting's primary reply fields against
`off`. Nothing is run; dev seeds only; no verdict.

    .venv/bin/python docs/drafts/stage2-exit/c5_integrator_heldout_read.py \
        --out docs/drafts/stage2-exit/c5_integrator_heldout_read.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import stage2_exit_gate as G  # noqa: E402

ROOT = "research/findings/raw/_stable_recal/dev"
SETTINGS = ("off", "stable", "stable_recal")
GROUPS = ("conv", "infer", "ask", "emo", "inner")
# seeds 7 and 8 are in-sample for the re-calibration (the swap workspace was fitted on 7 and 8, the bus on 7)
IN_SAMPLE = (7, 8)


def _shard(setting, seed, group):
    p = os.path.join(_REPO, ROOT, setting, "s%d" % seed, "%s__intact_a.json" % group)
    if not os.path.exists(p):
        return None, p
    with open(p) as f:
        rec = json.load(f)
    return ((rec.get("workload") or {}).get("shard")), p


def _answered(resp):
    p = G._primary(resp)
    return None if p is None else (not p["abstained"])


def read(seeds=tuple(range(1, 13))):
    cells, missing = {}, []
    for seed in seeds:
        for group in GROUPS:
            base, _ = _shard("off", seed, group)
            for setting in SETTINGS:
                sh, p = _shard(setting, seed, group)
                if sh is None or "responses" not in sh:
                    missing.append(os.path.relpath(p, _REPO))
                    continue
                sc = G.SCORERS[group](sh)
                c = {"all": bool(sc.get("all"))}
                if group == "inner":
                    c.update({k: sc.get(k) for k in ("private_voices_stored_fact", "inner_terminal_lynx",
                                                     "inner_terminal_vole", "single_hop", "moat_tail",
                                                     "chain_hops_internal_ge2", "private_utterance")})
                if base is not None and setting != "off":
                    labs = [l for l in base["responses"] if l in sh["responses"]]
                    c["primary_differs_vs_off"] = [l for l in labs if G.reply_differs(base["responses"], sh["responses"], l)]
                    c["answered_off_not_here"] = [l for l in labs if _answered(base["responses"][l]) is True
                                                  and _answered(sh["responses"][l]) is False]
                cells["%s/s%d/%s" % (setting, seed, group)] = c
    summary = {}
    for setting in SETTINGS:
        for group in GROUPS:
            for part, ss in (("held_out", [s for s in seeds if s not in IN_SAMPLE]), ("in_sample", [s for s in seeds if s in IN_SAMPLE])):
                vals = [cells.get("%s/s%d/%s" % (setting, s, group)) for s in ss]
                vals = [v for v in vals if v is not None]
                summary["%s/%s/%s" % (setting, group, part)] = {
                    "n": len(vals), "all_true": sum(1 for v in vals if v["all"]),
                    "seeds_false": [s for s in ss if (cells.get("%s/s%d/%s" % (setting, s, group)) or {}).get("all") is False],
                    "answers_lost_vs_off": sum(len(v.get("answered_off_not_here") or []) for v in vals)}
    return {"backend": "numpy", "device": "cpu", "analysis_only": True, "what": "stable-recal held-out battery re-scored with the registered Stage-2 scorers (intact_a only; "
                    "correctness clause; no lesion/null/permuted arm exists in this battery)",
            "root": ROOT, "revision": "34a4e5f756024b4f5aa5d145b3b29661ad976d0b (the battery's pool pin)",
            "in_sample_seeds": list(IN_SAMPLE), "cells": cells, "summary": summary, "missing": missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = read()
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    for k, v in res["summary"].items():
        print(k, v)
    print("missing:", res["missing"])


if __name__ == "__main__":
    main()
