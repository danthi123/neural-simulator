"""Aggregate the izh-substeps T2 organ probe (research/runners/izh_substep_organ_probe.py) over dev seeds 71-78.

Reads <t2>/<arm>/s<seed>.json, checks instrument validity (cfg read-back + engagement counters), scores the bars
pre-stated before the runs (B1 fix-on arms keep every membrane in range on 8/8 seeds; B2 fix-off breaches exactly on
72/75/77 and matches the step-1 grid's events 1-2 v_range; B3 fix-on + bounded-off still breaches >= 100 cells per seed),
and writes <t2>/T2_summary.json. Observe-only; no simulation.

    python -m research.runners.izh_substep_t2_aggregate --t2 research/findings/raw/_izh_substeps/t2 \
        --grid <step-1 grid dir holding s<seed>/s<seed>/seam_b.json for stn100_gpe100>
"""
import argparse
import json, os, collections

_ap = argparse.ArgumentParser()
_ap.add_argument("--t2", required=True)
_ap.add_argument("--grid", required=True)
_args = _ap.parse_args()
T2 = _args.t2
GRID = _args.grid
ARMS = ["fix_off", "fix_a", "fix_b", "fix_a_unbounded", "fix_a_nopreset", "fix_a_n2"]
SEEDS = range(71, 79)
R = {a: {s: json.load(open(f"{T2}/{a}/s{s}.json")) for s in SEEDS} for a in ARMS}
out = {"arms": {}, "bars": {}}
for a in ARMS:
    rows = {}
    for s in SEEDS:
        r = R[a][s]
        rows[s] = {"cfg": {k: r["cfg_readback"][k] for k in ("izh_membrane_substeps", "izh_substep_refractory_rule",
                   "enable_bounded_conductance_integration", "preset_centred_parameter_heterogeneity", "refractory_period_steps")},
                   "counters": r["counters_end"], "n_steps": r["n_steps"], "n_breach": r["membrane"]["n_breach_cells"],
                   "v_min": round(r["membrane"]["v_min"], 3), "v_max": round(r["membrane"]["v_max"], 3),
                   "winners": r["winners"], "n_match": r["n_match"], "n_abstain": r["n_abstain"],
                   "breach_cells": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for c in r["breach_cells"]][:5] if a != "fix_a_unbounded" else None,
                   "breach_regions": dict(collections.Counter(c["region"] for c in r["breach_cells"]))}
    out["arms"][a] = rows
# validity
val = {}
for a in ARMS:
    ok = True
    for s in SEEDS:
        r = R[a][s]; c = r["counters_end"]; cb = r["cfg_readback"]
        exp_n = 1 if a == "fix_off" else (2 if a == "fix_a_n2" else 4)
        exp_rule = "no_engine_timer" if a == "fix_b" else "model_reset"
        ok &= cb["izh_membrane_substeps"] == exp_n
        ok &= (exp_n == 1) or cb["izh_substep_refractory_rule"] == exp_rule
        ok &= cb["enable_bounded_conductance_integration"] == (a != "fix_a_unbounded")
        ok &= cb["preset_centred_parameter_heterogeneity"] == (a != "fix_a_nopreset")
        ok &= c["substep_calls"] == (0 if a == "fix_off" else r["n_steps"])
        if a in ("fix_off", "fix_b"):
            ok &= c["refractory_model_resets"] == 0
    val[a] = bool(ok)
out["validity"] = val
# B1
out["bars"]["B1"] = {a: {"seeds_clean": sum(R[a][s]["membrane"]["n_breach_cells"] == 0 for s in SEEDS),
                         "pass": all(R[a][s]["membrane"]["n_breach_cells"] == 0 for s in SEEDS)} for a in ("fix_a", "fix_b", "fix_a_nopreset")}
# B2
breach_seeds = sorted(s for s in SEEDS if R["fix_off"][s]["membrane"]["n_breach_cells"] > 0)
gm = {}
for s in SEEDS:
    g = json.load(open(f"{GRID}/s{s}/s{s}/seam_b.json"))["events"]
    mine = {e["label"]: [e["v_min"], e["v_max"]] for e in R["fix_off"][s]["events"]}
    grid = {k: [e for e in g[k] if e.get("kind") == "bg_neural"][0]["v_range"] for k in ("hsb_empty1", "hsb_q")}
    gm[s] = {"match": all(mine[k] == grid[k] for k in grid), "mine": {k: mine[k] for k in grid}, "grid": grid}
out["bars"]["B2"] = {"breach_seeds": breach_seeds, "breach_set_ok": breach_seeds == [72, 75, 77],
                     "grid_v_range_events12_match_seeds": sum(v["match"] for v in gm.values()), "detail": gm,
                     "pass": breach_seeds == [72, 75, 77] and all(v["match"] for v in gm.values())}
out["bars"]["B3"] = {"n_breach_per_seed": {s: R["fix_a_unbounded"][s]["membrane"]["n_breach_cells"] for s in SEEDS},
                     "pass": all(R["fix_a_unbounded"][s]["membrane"]["n_breach_cells"] >= 100 for s in SEEDS)}
# per-region summaries (seed-pooled)
reg = {}
for a in ARMS:
    agg = collections.defaultdict(lambda: collections.Counter())
    for s in SEEDS:
        for nm, pr in R[a][s]["per_region"].items():
            agg[nm]["spikes"] += pr["spikes"]; agg[nm]["isi1"] += pr["isi1_spikes"]; agg[nm]["silent"] += pr["silent_model_resets"]
    reg[a] = {nm: dict(c) for nm, c in agg.items() if c["isi1"] or c["silent"]}
out["regions_resets_isi1"] = reg
# min C per region + a-priori multipliers (worst cell, max over seeds) from the fix_off (n=1) and fix_a (n=4) builds
mc = {}
for s in SEEDS:
    for nm, pr in R["fix_off"][s]["per_region"].items():
        m = mc.setdefault(nm, {"min_C": [], "E_i": pr["E_i"], "n1": collections.defaultdict(list), "n4": collections.defaultdict(list)})
        m["min_C"].append(round(pr["min_C"], 2))
        for k, v in pr["mult_n1"].items(): m["n1"][k].append(v)
    for nm, pr in R["fix_a"][s]["per_region"].items():
        for k, v in pr["mult_n_arm"].items(): mc[nm]["n4"][k].append(v)
out["minC_mult"] = {nm: {"min_C_min": min(m["min_C"]), "min_C_per_seed": m["min_C"], "E_i": m["E_i"],
                         "n1": {k: max(v) for k, v in m["n1"].items()}, "n4": {k: max(v) for k, v in m["n4"].items()}}
                    for nm, m in mc.items()}
spk = {}
for a in ARMS:
    for s in SEEDS:
        for nm, pr in R[a][s]["per_region"].items():
            spk.setdefault(nm, {}).setdefault(a, 0)
            spk[nm][a] += pr["spikes"]
out["spikes_pooled"] = spk
json.dump(out, open(f"{T2}/T2_summary.json", "w"), indent=1)
print("validity", val)
for b in ("B1", "B2", "B3"):
    print(b, {k: v for k, v in out["bars"][b].items() if k != "detail"})
for a in ARMS:
    print("==", a, "breach", [out["arms"][a][s]["n_breach"] for s in SEEDS], "match", [out["arms"][a][s]["n_match"] for s in SEEDS],
          "abst", [out["arms"][a][s]["n_abstain"] for s in SEEDS], "vmin", [out["arms"][a][s]["v_min"] for s in SEEDS])
    print("   winners", [out["arms"][a][s]["winners"] for s in SEEDS])
    print("   resets", [out["arms"][a][s]["counters"]["refractory_model_resets"] for s in SEEDS])
for a in ("fix_off", "fix_a", "fix_b", "fix_a_n2", "fix_a_nopreset"):
    print("regions", a, reg[a])
print("fix_off breach cells", {s: out["arms"]["fix_off"][s]["breach_cells"] for s in (72, 75, 77)})
for nm, m in sorted(out["minC_mult"].items(), key=lambda x: x[1]["min_C_min"]):
    print(f"  {nm:16s} minC {m['min_C_min']:7.2f} E_i {m['E_i']}  n1 rest {m['n1']['step_rest_g0']:.3f} Ei_g0 {m['n1']['step_Ei_g0']:.3f} Ei_ginf {m['n1']['step_Ei_ginf']:.3f} | n4 sub_rest {m['n4']['substep_rest']:.3f} sub_Ei {m['n4']['substep_Ei']:.3f} step_rest {m['n4']['step_rest_g0']:.3f} Ei_g0 {m['n4']['step_Ei_g0']:.3f} Ei_ginf {m['n4']['step_Ei_ginf']:.3f}")
for nm in ("selector_reset", "latch_fs_0", "latch_fs_1", "hold_inh", "commit_fs_0", "commit_fs_1", "str_d1_0", "str_d2_1", "gpe_0", "gpi_0", "selector_stn"):
    print("spikes", nm, spk[nm])
