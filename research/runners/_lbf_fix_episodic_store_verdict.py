"""Assemble the BRAIN_EPISODIC_STORE_VERIFY verdict from the three verification legs, carrying a tools.verdict.Verdict
preconditions block (so the verdict-preconditions gate is satisfied):

  1. BYTE-IDENTICAL OFF   — the store's default (flag unset) formed-weight hash == the pre-fix HEAD module on all 6
                            seeds (research/findings/raw/_lbf_fix_episodic_store/byte_identical_6seed.json).
  2. RECOVERY MECHANISM   — a forced-degenerate ensemble (guaranteed non-completing) is re-encoded to in_memory=True
                            (lesion still collapses) on all 6 seeds (.../recovery_6seed.json).
  3. IN-SITU BATTERY      — LB_EPISODIC_DRIVE_PROBE=1 + BRAIN_EPISODIC_STORE_VERIFY=1, --only episodic-memory, one
                            --seed per run: episodic.in_memory flips intact True vs lesion False (load-bearing),
                            null-control clean, on all 6 seeds (fix_sweep/s<seed>/out.json).

Usage: .venv/bin/python -m research.runners._lbf_fix_episodic_store_verdict \
         --sweep-dir <fix_sweep dir> --out research/findings/raw/_lbf_fix_episodic_store/verdict.json
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path

SEEDS = [42, 43, 44, 100, 101, 102]
RAW = "research/findings/raw/_lbf_fix_episodic_store"


def _episodic_row(summary):
    per = summary.get("per_faculty") or summary.get("per") or []
    return next((r for r in per if r.get("faculty") == "episodic-memory"), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True, help="dir with s<seed>/out.json per-seed fix-ON battery summaries")
    ap.add_argument("--out", default=os.path.join(RAW, "verdict.json"))
    a = ap.parse_args()

    bi = json.load(open(os.path.join(RAW, "byte_identical_6seed.json")))
    rec = json.load(open(os.path.join(RAW, "recovery_6seed.json")))

    per_seed = []
    for s in SEEDS:
        p = os.path.join(a.sweep_dir, "s%d" % s, "out.json")
        row = {"seed": s, "found": os.path.exists(p)}
        if os.path.exists(p):
            r = _episodic_row(json.load(open(p)))
            if r:
                inmem = {d["field"]: (d.get("on"), d.get("off")) for d in (r.get("diffs") or [])
                         if d.get("field") == "episodic.in_memory"}
                row.update(load_bearing=bool(r.get("load_bearing")), verdict=r.get("verdict"),
                           treatment_diffs=r.get("treatment_diffs"), control_diffs=r.get("control_diffs"),
                           null_control_clean=r.get("null_control_clean"),
                           lesion_reproduced=r.get("lesion_reproduced"),
                           in_memory_intact_vs_lesion=inmem.get("episodic.in_memory"))
            else:
                row["error"] = "no episodic row"
        per_seed.append(row)

    n_lb = sum(1 for r in per_seed if r.get("load_bearing") is True)
    all_null_clean = all(r.get("null_control_clean") is True for r in per_seed if r.get("found"))
    n_found = sum(1 for r in per_seed if r.get("found"))

    # ATTRIBUTION (tools.lab): the episodic.in_memory flip must be attributable to the LESION, not run-to-run noise.
    # treatment = summed intact-vs-lesion decision diffs; control = summed intact-vs-intact-rebuild (null) diffs.
    from tools.lab import attributable_to
    tot_treat = sum(int(r.get("treatment_diffs") or 0) for r in per_seed if r.get("found"))
    tot_ctrl = sum(int(r.get("control_diffs") or 0) for r in per_seed if r.get("found"))
    attributable_fraction = attributable_to(
        "episodic.in_memory flip: lesion(treatment) vs null-rebuild(control), summed over seeds",
        tot_treat, tot_ctrl)

    from tools.verdict import Verdict
    v = Verdict("BRAIN_EPISODIC_STORE_VERIFY: episodic-memory load-bearing 6/6 (esp s44)", chance=0.0)
    v.require("6 seeds present in the in-situ battery sweep", n_found == 6, expect=True,
              note="fix_sweep/s{42,43,44,100,101,102}/out.json")
    v.require("byte-identical OFF: default-flag formed-weight hash == pre-fix HEAD on all 6 seeds",
              bool(bi.get("all_match")) and bi.get("n_match") == 6, expect=True,
              note="the shipped brain + the battery default are byte-identical when the flag is off")
    v.require("recovery mechanism: forced-degenerate -> re-encode -> in_memory=True (lesion collapses) on all 6 seeds",
              rec.get("n_recovered") == 6, expect=True,
              note="the re-encode loop reliably recovers a non-completing ensemble on every seed's own network")
    v.require("in-situ battery fix-ON: episodic-memory load-bearing on all 6 seeds (esp s44)",
              n_lb == 6, expect=True, note="episodic.in_memory intact True vs lesion False")
    v.require("in-situ null control clean on every built seed", bool(all_null_clean and n_found > 0), expect=True,
              note="intact-vs-intact-rebuild = 0 decision diffs -> the flip is attributable to the lesion")
    GO = bool(n_found == 6 and n_lb == 6 and all_null_clean and bi.get("all_match") and rec.get("n_recovered") == 6)
    vb = v.decide(go=GO)

    out = {"runner": "_lbf_fix_episodic_store_verdict",
           "faculty": "episodic-memory load-bearing robustness (BRAIN_EPISODIC_STORE_VERIFY), 6-seed",
           "flag": "BRAIN_EPISODIC_STORE_VERIFY", "default": "OFF", "seeds": SEEDS,
           "GO": GO, "verdict": vb["status"], "n_load_bearing": n_lb, "n_seeds_found": n_found,
           "byte_identical_off": {"all_match": bi.get("all_match"), "n_match": bi.get("n_match")},
           "recovery_6seed": {"n_recovered": rec.get("n_recovered"), "n_seeds": rec.get("n_seeds")},
           "attribution": {"treatment_diffs_total": tot_treat, "control_diffs_total": tot_ctrl,
                           "attributable_fraction": attributable_fraction},
           "per_seed_in_situ": per_seed,
           **{k: vb[k] for k in ("preconditions", "disabled_processes", "undefined_reasons")}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({"GO": GO, "verdict": vb["status"], "n_load_bearing": n_lb, "n_seeds_found": n_found,
                      "byte_identical": bi.get("all_match"), "recovery": rec.get("n_recovered")}, default=str))
    for r in per_seed:
        print("  seed %-3s load_bearing=%s verdict=%s treat=%s null_clean=%s in_mem(intact,lesion)=%s" % (
            r["seed"], r.get("load_bearing"), r.get("verdict"), r.get("treatment_diffs"),
            r.get("null_control_clean"), r.get("in_memory_intact_vs_lesion")))


if __name__ == "__main__":
    main()
