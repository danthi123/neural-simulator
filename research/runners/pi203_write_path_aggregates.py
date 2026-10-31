#!/usr/bin/env python3
"""Aggregate the #203 follow-up write-path attribution into the numbers the finding quotes.

WHY this exists as a runner rather than as prose (2026-10-31, Vikunja #203 follow-up). tools/claim_check.py
requires every measurement in a findings doc to EXIST in an artifact the doc cites. The per-cell numbers live in
write_path_attribution_v2.json; the rounded values quoted in the prose (e.g. max_abs_margin_diff) are derived
from it, and a derived value stated in a doc without a saved aggregate is exactly the failure mode this project
has been bitten by twice. This runner recomputes the aggregates from the raw record, so the doc's numbers are
checkable and regenerable:

    PYTHONPATH=. python -m research.runners.pi203_write_path_aggregates \
        --src research/findings/raw/_pi203_frozen_synapse_residual/write_path_attribution_v2.json \
        --out  research/findings/raw/_pi203_frozen_synapse_residual/write_path_attribution_aggregates.json
"""
from __future__ import annotations

import argparse
import json

import tools.lab


def aggregate(src: str) -> dict:
    d = json.load(open(src, encoding="utf-8"))
    out = {
        "runner": "research.runners.pi203_write_path_aggregates",
        "derived_from": src,
        "protocol": d.get("protocol"),
        "backend": d.get("backend"),
        "device": d.get("device"),
        "n_runs": d.get("n_runs"),
        "elapsed_seconds": d.get("elapsed_seconds"),
        "per_cell_frozen_delta": {},
        "per_site": {},
        "reply_level": {},
        "reply_competence": {},
        "instrument_control": {},
        "instrument_completeness": {},
        "lever_check": d.get("lever_check"),
    }
    for key, v in d.get("attribution", {}).items():
        out["per_cell_frozen_delta"][key] = {
            "off": round(v["total_frozen_delta_off"], 2),
            "on": round(v["total_frozen_delta_on"], 2),
        }
        for s in v.get("sites", []):
            out["per_site"][f"{key} {s['site']}"] = {
                "writes_off": s["writes_off"], "writes_on": s["writes_on"],
                "frozen_delta_off": round(s["frozen_sum_abs_delta_off"], 2),
                "frozen_delta_on": round(s["frozen_sum_abs_delta_on"], 2),
                "frozen_potentiation_off": round(s["frozen_potentiation_off"], 2),
                "frozen_depression_off": round(s["frozen_depression_off"], 2),
                "share_attributable_to_flag": s["residual_frozen_delta_share_on"],
            }
        cd = v.get("coordinate_drift_off", {})
        out["per_site"][f"{key} coordinate_drift_off"] = {
            "frozen_max_abs_drift": round(cd.get("frozen_max_abs_drift", 0.0), 3),
            "frozen_synapses_drifted": cd.get("frozen_synapses_drifted", 0),
            "plastic_synapses_drifted": cd.get("plastic_synapses_drifted", 0),
            "csr_rebuilds": cd.get("csr_rebuilds", 0),
        }
    for key, v in d.get("reply_level", {}).items():
        out["reply_level"][key] = {
            "items_differing": v["items_differing"],
            "n_items": v["n_items"],
            "max_abs_margin_diff": round(v["max_abs_margin_diff"], 5),
        }
    by = {}
    for r in d.get("runs", []):
        by[(r["seed"], r["organ"], r["arm"], r["tap"])] = r
    for seed in sorted({k[0] for k in by}):
        for organ in ("standalone", "pooled"):
            off, on = by.get((seed, organ, "off", "on")), by.get((seed, organ, "on", "on"))
            if off and on:
                out["reply_competence"][f"{organ}/seed{seed}"] = {
                    "items_off": len(off["reply_competent"]),
                    "items_on": len(on["reply_competent"]),
                    "competent_off": sum(1 for v in off["reply_competent"].values() if v),
                    "competent_on": sum(1 for v in on["reply_competent"].values() if v),
                    "competence_flips": sum(1 for k in off["reply_competent"]
                                           if off["reply_competent"][k] != on["reply_competent"][k]),
                }
    for key, v in d.get("instrument_control", {}).items():
        out["instrument_control"][key] = {"identical": v["identical"], "digest": v["tapped_digest"]}
    for key, v in d.get("instrument_completeness", {}).items():
        out["instrument_completeness"][key] = {
            "signed_residual": round(v["signed_residual"], 6),
            "abs_residual": round(v["abs_residual"], 2),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tools.lab.assert_backend("numpy", "the #203 follow-up attribution ran on the CPU/numpy path by design")
    agg = aggregate(args.src)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=1)
    print(f"[aggregates] {args.src} -> {args.out} | {len(agg['per_site'])} site rows, "
          f"{len(agg['reply_level'])} reply cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
