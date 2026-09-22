"""LOAD-BEARING BORDERLINE — DIAGNOSIS SUMMARY + EARNED VERDICT (research/lbf-borderline-diagnosis, 2026-09-21).

Reads the six per-seed operating-point artifacts produced by research.runners._lbf_borderline_operating_point
(op_s{42,43,44,100,101,102}.json) + the committed AWS ground-truth verdicts (the adequate6 load-bearing battery +
the pmem_v2 6-seed), tabulates the per-seed MARGIN to the decision threshold for each of the four seed-dependent
borderline faculties, and EARNS a verdict (tools.verdict.Verdict) on the DIAGNOSIS OUTCOME. Builds no brain (a pure
read + aggregation; no memcap needed).

Run: .venv/bin/python -m research.runners._lbf_borderline_diagnosis_summary \
       --out research/findings/raw/_lbf_borderline/diagnosis_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from tools.verdict import Verdict
# attribution discipline (tools.lab): the source-provenance intact/lesion opponent read is a treatment/control pair;
# ASK whose the difference is (fraction of the read owned by the lesioned pathway = (intact - lesion)/intact).
from tools.lab import attributable_to

SEEDS = ["42", "43", "44", "100", "101", "102"]
BORDER = ["episodic-memory", "prospective-memory", "affect-marker-spiking-wta", "source-provenance-honesty"]
COMPLETE_MIN, FIRE_THR, DEAD_MARGIN = 0.20, 0.20, 0.05


def _ground_truth(root):
    gt = {}
    for f in glob.glob(os.path.join(root, "_adequate6", "load_bearing_adequate_s*.json")):
        if f.endswith(".prov.json"):
            continue
        s = f.split("_s")[-1].split(".")[0]
        for p in json.load(open(f))["per_faculty"]:
            if p["faculty"] in BORDER:
                gt.setdefault(p["faculty"], {})[s] = p.get("load_bearing")
    for f in glob.glob(os.path.join(root, "_pmem_v2_6seed", "pmem_v2_s*.json")):
        if f.endswith(".prov.json"):
            continue
        s = f.split("_s")[-1].split(".")[0]
        try:
            gt.setdefault("prospective-memory", {})[s] = json.load(open(f))["per_faculty"][0].get("load_bearing")
        except Exception:
            pass
    return gt


def run(op_dir, lb_root):
    op = {}
    for s in SEEDS:
        fn = os.path.join(op_dir, "op_s%s.json" % s)
        op[s] = json.load(open(fn))["per_faculty"] if os.path.exists(fn) else None
    gt = _ground_truth(lb_root)

    table = {fac: {} for fac in BORDER}
    n_errors = 0
    for fac in BORDER:
        for s in SEEDS:
            if not op.get(s):
                table[fac][s] = {"error": "op-file-missing"}
                n_errors += 1
                continue
            r = op[s].get(fac, {})
            if "error" in r:
                table[fac][s] = {"error": r["error"]}
                n_errors += 1
                continue
            table[fac][s] = {"margin_to_threshold": r.get("margin_to_threshold"),
                             "predicted_load_bearing": r.get("predicted_load_bearing"),
                             "aws_load_bearing": gt.get(fac, {}).get(s),
                             "intact": r.get("intact"), "lesion": r.get("lesion"), "ladder": r.get("ladder")}

    # ── derived diagnostic statistics ────────────────────────────────────────────────────────────────────────
    def col(fac, key, sub=None):
        out = []
        for s in SEEDS:
            c = table[fac][s]
            if "error" in c:
                continue
            v = c.get(key) if sub is None else (c.get(key) or {}).get(sub)
            out.append(v)
        return out

    pmem_m = col("prospective-memory", "margin_to_threshold")
    epi_cue = col("episodic-memory", "intact", "apical_cue")
    prov_di = [ (c.get("intact") or {}).get("d") for s in SEEDS for c in [table["source-provenance-honesty"][s]] if "error" not in c ]
    prov_dl = [ (c.get("lesion") or {}).get("d") for s in SEEDS for c in [table["source-provenance-honesty"][s]] if "error" not in c ]
    aff_m = col("affect-marker-spiking-wta", "margin_to_threshold")   # = intact_margin - DEAD_MARGIN

    pmem_agree = sum(1 for s in SEEDS if "error" not in table["prospective-memory"][s]
                     and table["prospective-memory"][s]["predicted_load_bearing"]
                     == table["prospective-memory"][s]["aws_load_bearing"])
    pmem_brackets = (min(pmem_m) < 0) and (max(pmem_m) > 0)
    min_prov_di = min(prov_di) if prov_di else None
    max_prov_dl = max(prov_dl) if prov_dl else None
    # attribution: fraction of the source-provenance opponent read owned by the lesioned pathway, aggregated (mean
    # intact d vs mean lesion d). ~1.0 -> the intact read is entirely the learned trace's, the lesion collapses it;
    # the seed-dependence is therefore NOT in the read but in the degenerate control's coin-flip (see per-seed).
    _mi = (sum(prov_di) / len(prov_di)) if prov_di else None
    _ml = (sum(prov_dl) / len(prov_dl)) if prov_dl else None
    prov_attributable = attributable_to("source-provenance intact-vs-lesion d (mean over seeds)", _mi, _ml) \
        if (_mi is not None and _ml is not None) else None
    # affect near-threshold: |intact_margin - DEAD_MARGIN| = |margin_to_threshold| < 0.02
    aff_near = sum(1 for m in aff_m if m is not None and abs(m) < 0.02)
    min_epi_recall_margin = (min(epi_cue) - COMPLETE_MIN) if epi_cue else None

    stats = {"n_errors": n_errors, "pmem_agreement_6seed": pmem_agree, "pmem_margin_brackets_thr": bool(pmem_brackets),
             "pmem_margins": pmem_m, "min_intact_prov_d": min_prov_di, "max_lesion_prov_d": max_prov_dl,
             "affect_near_threshold_count": aff_near, "affect_margins_to_thr": aff_m,
             "min_epi_recall_margin": min_epi_recall_margin, "epi_intact_apical_cue": epi_cue,
             "provenance_d_attributable_to_lesioned_pathway": prov_attributable}

    # ── EARN the verdict on the diagnosis outcome ────────────────────────────────────────────────────────────
    v = Verdict("lbf-borderline-operating-point-diagnosis")
    v.require("24 organ-reads complete (0 errors across 4 faculties x 6 seeds)", n_errors, expect=0)
    v.require("prospective-memory margin-sign predicts the AWS verdict 6/6 (a GENUINE near-threshold fragility)",
              pmem_agree, expect=6)
    v.require("prospective-memory operating point BRACKETS FIRE_THR (min margin <0 < max margin -> the flip is a "
              "threshold crossing, not wiring)", pmem_brackets, expect=True)
    v.require("source-provenance intact opponent read is ROBUST at every seed (min intact d >= 0.99) -> its "
              "borderline is NOT a brain operating point", (min_prov_di is not None and min_prov_di >= 0.99),
              expect=True)
    v.require("source-provenance lesion is a DEGENERATE control (lesion d == 0 at every seed) -> the seed-dependence "
              "is the tie-break coin-flip on the discretized label (an INSTRUMENT artifact)",
              (max_prov_dl is not None and max_prov_dl < 1e-6), expect=True)
    v.require("affect-marker WTA winner margin sits AT DEAD_MARGIN (|margin - 0.05| < 0.02) at >= 5/6 seeds -> the "
              "verdict flips with seed/build-context/hardware", aff_near >= 5, expect=True)
    v.require("episodic RECALL margin is robust at every seed (min apical_cue - COMPLETE_MIN > 0.2) -> the fragile "
              "step is the STORE (assembly formation), not the recall read",
              (min_epi_recall_margin is not None and min_epi_recall_margin > 0.2), expect=True)
    decided = v.decide(go=True)

    rep = {"runner": "research.runners._lbf_borderline_diagnosis_summary", "kind": "diagnosis-summary",
           "faculties": BORDER, "seeds": SEEDS, "thresholds": {"COMPLETE_MIN": COMPLETE_MIN, "FIRE_THR": FIRE_THR,
           "DEAD_MARGIN": DEAD_MARGIN}, "per_faculty_per_seed": table, "stats": stats}
    rep.update(decided)   # status / go / preconditions / undefined_reasons at TOP LEVEL (verdict-preconditions gate)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-dir", default="research/findings/raw/_lbf_borderline")
    ap.add_argument("--lb-root", default="research/findings/raw/_load_bearing")
    ap.add_argument("--out", default="research/findings/raw/_lbf_borderline/diagnosis_verdict.json")
    args = ap.parse_args()
    rep = run(args.op_dir, args.lb_root)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2, default=str)
    print("STATUS:", rep.get("status"), "go:", rep.get("go"))
    print("stats:", json.dumps(rep["stats"], default=str))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
