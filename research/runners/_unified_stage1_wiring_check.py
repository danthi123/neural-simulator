"""Stage-1 parser-silence: CPU-ONLY wiring check (no GPU, no simulation).

diag3 gen-OFF: parse_role has 720 excitatory / 0 inhibitory incoming edges and fires (v_max 34.8). The decisive
question for gen-ON is whether the generalization stack adds EXTRA / INHIBITORY edges INTO parse_role (accidental
wiring) or shifts the parse_conj->parse_role wiring (RNG/order) -- vs. leaving it identical (=> the suppression is a
dynamic/size effect, not wiring). That question is answered by the union WIRING PLAN alone, which is a pure CPU
computation (RegionManager.initialize + build_wiring_plan) -- NO bridge build, NO GPU, NO convergence training. This
runs instantly and is immune to the desktop GPU contention.

Replicates build_merged_nav_conv_bridge's region/pathway union (lines 462-520) for gen-OFF and gen-ON, builds the
RegionManager + wiring plan, and reports every edge INTO parse_role (count, pre-region, exc/inh) + whether the
parse_conj->parse_role edge set is byte-identical across the two.

Run: SIM_BACKEND=numpy python -m research.runners._unified_stage1_wiring_check --seed 42
"""
import argparse
import json

import numpy as np

from sim.regions import RegionManager, BrainRegion
from research.runners.g11_bg_runner import build_bg_brain_regions
from research.runners.nav_conv_merged_bridge import (
    parser_regions_pathways, _generalization_regions_pathways, PARSER_R,
)


def build_union(gen_flag, seed, n_cortex=100, rf_D=128):
    nav_regions, nav_pathways = build_bg_brain_regions(
        n_cortex=n_cortex, enable_spiking_wta_readout=True)
    parser_regions, parser_pathways = parser_regions_pathways(PARSER_R)
    from research.runners.rf_phasor_composer import DEFAULT_VOCAB
    V = len(sorted(set(DEFAULT_VOCAB)))
    n_dlpfc = max(600, 60 * V)
    dlpfc_regions = [
        BrainRegion(name="cortex_ctx", n_neurons=n_dlpfc, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="dlpfc_wm", n_neurons=n_dlpfc, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
    ]
    rf_regions = [BrainRegion(name="rf", n_neurons=7 * int(rf_D), exc_fraction=1.0,
                              internal_density=0.0, enable_nmda=False)]
    perception_regions = [BrainRegion(name="cortex_it", n_neurons=256, exc_fraction=0.8,
                                      internal_density=0.0, enable_nmda=False)]
    gen_regions, gen_pathways = ([], [])
    if gen_flag:
        gen_regions, gen_pathways = _generalization_regions_pathways(100, 100)
    union_regions = (list(nav_regions) + list(parser_regions) + list(dlpfc_regions)
                     + list(rf_regions) + list(perception_regions) + list(gen_regions))
    union_pathways = list(nav_pathways) + list(parser_pathways) + list(gen_pathways)
    rm = RegionManager(union_regions, union_pathways)
    rm.initialize(seed=seed)
    plan = rm.build_wiring_plan(seed=seed)
    return rm, plan


def _pre_post(pop):
    pre = pop.get("pre_indices", pop.get("pre"))
    post = pop.get("post_indices", pop.get("post"))
    return np.asarray(pre, dtype=np.int64), np.asarray(post, dtype=np.int64)


def edges_into_parse_role(rm, plan):
    role_idx = np.asarray(list(rm.indices("parse_role")), dtype=np.int64)
    role_set = set(int(i) for i in role_idx)
    # region lookup for a pre neuron
    region_of = {}
    for region in rm.regions():
        for i in rm.indices(region.name):
            region_of[int(i)] = region.name
    inh_set = set()
    for region in rm.regions():
        inh_set.update(int(i) for i in rm.inhibitory_indices(region.name))

    by_pre_region = {}
    n_exc = n_inh = 0
    conj_role_edges = []
    for pop_name, pop in plan.items():
        pre, post = _pre_post(pop)
        if post.size == 0:
            continue
        mask = np.isin(post, role_idx)
        if not mask.any():
            continue
        pre_in = pre[mask]
        post_in = post[mask]
        for p, q in zip(pre_in.tolist(), post_in.tolist()):
            rg = region_of.get(int(p), "?")
            by_pre_region[rg] = by_pre_region.get(rg, 0) + 1
            if int(p) in inh_set:
                n_inh += 1
            else:
                n_exc += 1
            if rg == "parse_conj":
                conj_role_edges.append((int(p), int(q)))
    return {
        "n_into_role": n_exc + n_inh, "n_exc": n_exc, "n_inh": n_inh,
        "by_pre_region": by_pre_region,
        "conj_role_edge_set": sorted(conj_role_edges),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="research/findings/raw/_unified_stage1_wiring_check.json")
    args = ap.parse_args()

    results = {}
    for gen_flag in (False, True):
        rm, plan = build_union(gen_flag, args.seed)
        info = edges_into_parse_role(rm, plan)
        role_base = int(rm.indices("parse_role")[0])
        conj_base = int(rm.indices("parse_conj")[0])
        n_total = rm.total_neurons()
        results[("gen_on" if gen_flag else "gen_off")] = {
            "n_total_neurons": int(n_total), "parse_role_base": role_base, "parse_conj_base": conj_base,
            **{k: v for k, v in info.items() if k != "conj_role_edge_set"},
            "_conj_role_edge_set": info["conj_role_edge_set"],
        }
        print(f"[wiring] gen={'ON ' if gen_flag else 'OFF'}: n_neurons={n_total} parse_role@{role_base} "
              f"edges_into_role={info['n_into_role']} (exc {info['n_exc']}, inh {info['n_inh']}) "
              f"by_pre_region={json.dumps(info['by_pre_region'])}", flush=True)

    off, on = results["gen_off"], results["gen_on"]
    conj_role_identical = (off["_conj_role_edge_set"] == on["_conj_role_edge_set"])
    extra_into_role = on["n_into_role"] - off["n_into_role"]
    inh_appeared = on["n_inh"] > off["n_inh"]
    verdict = {
        "conj_role_wiring_identical": conj_role_identical,
        "extra_edges_into_role_on_gen": extra_into_role,
        "inhibitory_edges_appeared_on_gen": inh_appeared,
        "diagnosis": (
            "ACCIDENTAL INHIBITORY wiring into parse_role on gen-ON (the suppressor)" if inh_appeared else
            "EXTRA edges into parse_role on gen-ON (wiring leak)" if extra_into_role > 0 else
            "parse_conj->parse_role wiring DIFFERS on gen-ON (RNG/order shift)" if not conj_role_identical else
            "WIRING IDENTICAL -> the suppression is a DYNAMIC/size effect, not wiring (parser silence is NOT a wiring bug)"),
    }
    print(f"\n[wiring] VERDICT {json.dumps(verdict, indent=2)}", flush=True)
    with open(args.out, "w") as f:
        json.dump({"verdict": verdict, "gen_off": {k: v for k, v in off.items() if k != "_conj_role_edge_set"},
                   "gen_on": {k: v for k, v in on.items() if k != "_conj_role_edge_set"}}, f, indent=2)
    print(f"[wiring] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
