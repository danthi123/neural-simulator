#!/usr/bin/env python3
"""Vikunja #203 follow-up (pi lane): attribute the comprehension organ's frozen-synapse drift to a WRITE PATH.

The #203 merge (main `dc6abf989`) gates six `cp_connections.data` write sites behind
`cfg.enforce_plastic_mask_beyond_hebbian` (umbrella env `BRAIN_ENFORCE_PLASTIC_MASK`, DEFAULT OFF), and its
reply-level arm found the flag changes the production comprehension organ's reply ZERO times on 6 seeds -- while
the 2026-09-02 read-isolation audit had measured a real 13.8 -> 56.1 weight drift on this very organ.

This runner answers the question the reply arm left open: WHICH write path moves the organ's frozen weights, and
does any of it escape the six gated sites? It runs `research.runners.pi203_write_path_tap` (which records every
write to `cp_connections.data`, attributed to the sim line that performed it, with the delta split into frozen vs
plastic) over the production battery, on DEV seeds only (7, 8 -- the generalization seeds 42/43/44/100/101/102 are
untouched by this instrument), for both organs that "the comprehension organ" can mean:

  * `standalone`  : `ComprehensionProductionOrgan(seed)` -- the 1032-neuron `SpikingRoleCompetition` bridge the
                    2026-09-02 audit measured (24764 synapses, 1724 declared frozen).
  * `pooled`      : `get_organ(seed)` -- the wave3 merged one-brain pool (531916 synapses) that the reply arm ran.

and both arms (flag OFF / flag ON), plus a TAP-OFF control run per cell so the instrument's own non-invasiveness
is measured rather than asserted (tapped vs untapped reply digests must be identical).

Protocol fixed BEFORE the run (this docstring is the pre-registration; the criteria are quoted in the artifact):
  1. every write site that touches a mask-frozen synapse is listed with its measured delta, ON and OFF;
  2. a site is "uncovered by the six fixes" only if it still moves frozen synapses with the flag ON;
  3. the tap is void if tapped and untapped reply digests differ for the same cell.

Run:
    PYTHONPATH=. CUDA_VISIBLE_DEVICES= SIM_BACKEND=numpy OMP_NUM_THREADS=1 \
      python -m research.runners.pi203_write_path_attribution \
      --out research/findings/raw/_pi203_frozen_synapse_residual/write_path_attribution.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time

from tools.lab import attributable_to, lever, project_cost, assert_backend

SEEDS = (7, 8)                                   # DEV seeds only (42/43/44/100/101/102 untouched)
ORGANS = ("standalone", "pooled")
ARMS = ("off", "on")
TAPS = ("on", "off")
RUNNER = "research.runners.pi203_write_path_tap"


def _run_cell(seed: int, organ: str, arm: str, tap: str, items: int = 0) -> dict:
    env = dict(os.environ)
    env["SIM_BACKEND"] = "numpy"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = "."
    out = "/tmp/pi203_tap_%s_s%d_%s_%s.json" % (organ, seed, arm, tap)
    p = subprocess.run([sys.executable, "-m", RUNNER, "--seed", str(seed), "--arm", arm,
                        "--organ", organ, "--tap", tap, "--items", str(items), "--out", out],
                       env=env, capture_output=True, text=True, timeout=2400)
    if p.returncode != 0 or not os.path.exists(out):
        return {"seed": seed, "organ": organ, "arm": arm, "tap": tap,
                "error": p.stderr[-400:], "returncode": p.returncode}
    with open(out) as f:
        d = json.load(f)
    return d


def _field_diff(a: dict, b: dict) -> list:
    """Which reply-level fields differ between two battery digests (per item)."""
    diffs = []
    for label in sorted(a.get("battery_items", {})):
        if a["battery_items"][label] != b.get("battery_items", {}).get(label):
            diffs.append(label)
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--organs", default=",".join(ORGANS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--items", type=int, default=0, help="limit battery items in every cell (0 = all)")
    args = ap.parse_args()
    organs = [o for o in args.organs.split(",") if o]
    seeds = [int(s) for s in args.seeds.split(",") if s]

    assert_backend("numpy", "this attribution runs on the CPU/numpy path by design (dev-seed smokes)")
    cells = [(s, o, a, t, args.items) for s in seeds for o in organs for a in ARMS for t in TAPS]
    t0 = time.time()
    runs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_cell, *c): c for c in cells}
        for i, fut in enumerate(concurrent.futures.as_completed(futs)):
            c = futs[fut]
            runs[c] = fut.result()
            if i == 0:
                project_cost("tap cell", 1, len(cells), runs[c].get("elapsed_seconds", 0.0))

    runs = [runs[c] for c in cells]
    by_key = {(r["seed"], r["organ"], r["arm"], r["tap"]): r for r in runs}

    # (3) instrument control: tapped vs untapped, same cell.
    instrument_control = {}
    for s in seeds:
        for o in organs:
            for a in ARMS:
                t = by_key.get((s, o, a, "on"))
                u = by_key.get((s, o, a, "off"))
                instrument_control["%s/%s/seed%d" % (o, a, s)] = {
                    "tapped_digest": t.get("battery_digest"), "untapped_digest": u.get("battery_digest"),
                    "identical": t.get("battery_digest") == u.get("battery_digest"),
                }

    # (1)+(2) per-site attribution: the same site, flag OFF vs flag ON.
    attribution = {}
    for o in organs:
        for s in seeds:
            off = by_key.get((s, o, "off", "on"))
            on = by_key.get((s, o, "on", "on"))
            if not off or not on:
                continue
            sites_off = {x["site"]: x for x in off.get("write_sites", [])}
            sites_on = {x["site"]: x for x in on.get("write_sites", [])}
            rows = []
            for site in sorted(set(sites_off) | set(sites_on)):
                a = sites_off.get(site, {"frozen_sum_abs_delta": 0.0, "frozen_max_abs_delta": 0.0, "writes": 0})
                b = sites_on.get(site, {"frozen_sum_abs_delta": 0.0, "frozen_max_abs_delta": 0.0, "writes": 0})
                rows.append({
                    "site": site, "function": a.get("function", b.get("function", "")),
                    "writes_off": a["writes"], "writes_on": b["writes"],
                    "frozen_sum_abs_delta_off": a["frozen_sum_abs_delta"],
                    "frozen_sum_abs_delta_on": b["frozen_sum_abs_delta"],
                    "frozen_potentiation_off": a.get("frozen_potentiation", 0.0),
                    "frozen_depression_off": a.get("frozen_depression", 0.0),
                    "frozen_potentiation_on": b.get("frozen_potentiation", 0.0),
                    "frozen_depression_on": b.get("frozen_depression", 0.0),
                    "residual_frozen_delta_share_on": attributable_to(
                        "frozen-synapse delta at %s (OFF = ungated treatment, ON = gated control)" % site,
                        a["frozen_sum_abs_delta"], b["frozen_sum_abs_delta"]),
                })
            attribution["%s/seed%d" % (o, s)] = {
                "total_frozen_delta_off": off.get("total_writes", 0) and sum(
                    x["frozen_sum_abs_delta"] for x in off.get("write_sites", [])),
                "total_frozen_delta_on": sum(x["frozen_sum_abs_delta"] for x in on.get("write_sites", [])),
                "coordinate_drift_off": off.get("coordinate_drift"),
                "coordinate_drift_on": on.get("coordinate_drift"),
                "frozen_drift_regions_off": off.get("frozen_drift_regions"),
                "sites": rows,
            }

    # reply-level: flag ON vs OFF, same organ/seed, tap ON (and cross-checked on tap OFF)
    reply_level = {}
    for o in organs:
        for s in seeds:
            for t in TAPS:
                a = by_key.get((s, o, "off", t))
                b = by_key.get((s, o, "on", t))
                if not a or not b:
                    continue
                diffs = _field_diff(a, b)
                m_off, m_on = a.get("reply_margins", {}), b.get("reply_margins", {})
                moved = {k: [m_off.get(k), m_on.get(k)] for k in m_off if m_off.get(k) != m_on.get(k)}
                reply_level["%s/%s/seed%d" % (o, t, s)] = {
                    "items_differing": len(diffs),
                    "n_items": a.get("n_items"),
                    "margin_diffs": moved,
                    "max_abs_margin_diff": max((abs(x - y) for x, y in (tuple(v) for v in moved.values())), default=0.0),
                }

    # THE LEVER: does the flag actually move the measured quantity? (standalone organ = the drift organ)
    lever_check = {}
    for s in seeds:
        before = by_key.get((s, "standalone", "off", "on"), {}).get("total_writes", 0)
        after = by_key.get((s, "standalone", "on", "on"), {}).get("total_writes", 0)
        lever_check["seed%d_total_weight_writes" % s] = lever("BRAIN_ENFORCE_PLASTIC_MASK", before, after,
                                                              required=False)

    # (0) does the instrument account for EVERY change it sees? The tap sums every write's signed delta;
    # the coordinate snapshot is the net. If they agree, no write path escaped the hooks.
    completeness = {}
    for r in runs:
        if r.get("tap") != "on" or "error" in r:
            continue
        cd = r.get("coordinate_drift", {})
        completeness["%d/%s/%s" % (r["seed"], r["organ"], r["arm"])] = {
            "signed_residual": cd.get("completeness_residual_signed"),
            "abs_residual": cd.get("completeness_residual_abs"),
            "net_frozen_signed_drift": cd.get("frozen_signed_drift"),
            "written_net_frozen_delta": cd.get("frozen_signed_drift", 0.0) + (cd.get("completeness_residual_signed") or 0.0),
        }

    payload = {
        "runner": RUNNER + " (orchestrator)",
        "cmd": " ".join(sys.argv),
        "backend": "numpy", "device": "cpu",
        "protocol": {
            "seeds": list(SEEDS), "organs": list(ORGANS), "arms": list(ARMS), "taps": list(TAPS),
            "criteria": ["list every write site touching a mask-frozen synapse with its measured delta, OFF and ON",
                         "a site is UNCOVERED by the six #203 fixes only if it still moves frozen synapses with the flag ON",
                         "the tap is void if tapped and untapped reply digests differ in the same cell"],
        },
        "elapsed_seconds": round(time.time() - t0, 2),
        "n_runs": len(runs),
        "runs": [{k: v for k, v in r.items() if k != "write_sites"} for r in runs],
        "write_sites_by_run": {"%d/%s/%s/tap=%s" % (r["seed"], r["organ"], r["arm"], r["tap"]): r.get("write_sites", []) for r in runs},
        "attribution": attribution,
        "instrument_completeness": completeness,
        "instrument_control": instrument_control,
        "reply_level": reply_level,
        "lever_check": lever_check,
        "config_flags_observed": sorted({json.dumps(r.get("config", {}), sort_keys=True) for r in runs}),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print("[attribution] %d runs, %s total -> %s" % (len(runs), payload["elapsed_seconds"], args.out))
    for k, v in attribution.items():
        print("  %s: frozen delta OFF=%s ON=%s" % (k, v["total_frozen_delta_off"], v["total_frozen_delta_on"]))
    bad = [k for k, v in instrument_control.items() if not v["identical"]]
    print("  instrument control: %d/%d cells tapped==untapped; exceptions: %s"
          % (len(instrument_control) - len(bad), len(instrument_control), bad))
    print("  completeness signed residuals:", {k: v["signed_residual"] for k, v in completeness.items()})
    print("  reply level:", json.dumps(reply_level))


if __name__ == "__main__":
    main()
