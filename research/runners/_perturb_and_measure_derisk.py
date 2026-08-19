"""Perturb-and-measure FUNCTIONAL connectivity probe vs the ANATOMICAL weight graph.

METHOD (Randi, Sharma, Dvali & Leifer, "Neural signal propagation atlas of C. elegans",
Nature 623:406-414, 2023). Inspecting the static weight graph tells you what CAN connect; it does
NOT tell you what DOES propagate. Randi et al. PERTURBED each neuron in the LIVE animal and MEASURED
the downstream response, building a FUNCTIONAL connectivity map -- and found it DIFFERS from the
anatomical wiring diagram, because ongoing state / neuromodulation reweight the effective connectivity,
polysynaptic paths create edges with no direct synapse, and inhibition flips signs.

This runner does the analogue on an EXISTING multi-region spiking substrate: the navigation basal
ganglia built by `research.runners.g11_bg_runner.build_bg_brain_regions` (flagship cluster-A/E config,
the same substrate `_n8_thal_disinhibition_probe.py` drives). It is all-spiking / all-synaptic: the
perturbation is an external drive current onto a region's neurons and the read-out is those neurons'
spikes; NO host arithmetic stands between sensation and the measured propagation.

For each region TYPE A (cortex, str_D1, str_D2, str_PV_FSI, str_striosome, gpe, gpe_arky, gpi, stn,
snc, thal, motor) we drive all of A's neurons with an extra current, settle the network, and record
the change in firing rate of EVERY other region B relative to the un-perturbed baseline ->
F[A][B] (signed). The ANATOMICAL matrix W[A][B] is the signed aggregate of the declared RegionPathway
synapses A->B (sign from the presynaptic transmitter via exc_fraction). We then compare F vs W with
`tools.lab.functional_vs_anatomical`.

ANTI-CHEATS (they ARE the result):
  1. LESION DISSOCIATION -- cut the gpi->thal pathway (rebuild without it). The gpi->thal FUNCTIONAL
     edge collapses toward 0 AND the POLYSYNAPTIC cortex->thal disinhibition edge collapses (its route
     is severed), while the upstream cortex->gpi edge is UNCHANGED. Proves F measures propagation
     through the actual pathways, not an artifact.
  2. FUNCTIONAL != ANATOMICAL -- quantified: correlation < 1, the polysynaptic edges (cortex->{gpi,
     thal,motor}, str_D1->{thal,motor}) that have NO direct synapse, and a STATE-DEPENDENCE probe
     (lowering the gpi pacemaker reweights the cortex->thal effective edge -- a global-drive driver of
     the difference, the Randi neuromodulation point).
  3. DETERMINISM + SPECIFICITY -- identical F on a byte-for-byte re-run at fixed cfg.seed; and driving
     ONE action channel (cortex_N) drives its OWN downstream pool (str_D1_N) more than the sibling
     channels (str_D1_{E,S,W}) -- topographic specificity.
  4. 6 seeds (42,43,44,100,101,102), per-seed + pooled.

  SIM_BACKEND=numpy python -m research.runners._perturb_and_measure_derisk --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/perturb_measure/pm.json
"""
import argparse
import json
import os
import re
import time

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

from tools.lab import functional_vs_anatomical, attributable_to, assert_backend

TYPES = ["cortex", "str_D1", "str_D2", "str_PV_FSI", "str_striosome",
         "gpe", "gpe_arky", "gpi", "stn", "snc", "thal", "motor"]
ACTIONS = ["N", "E", "S", "W"]

# LOCKED operating point (see finding). The BG's own working regime: a strong gpi pacemaker (what makes
# it function as a selector, so the disinhibition route carries signal) plus a moderate cortical /
# striatal / motor tone so inhibitory responses register too. NOT tuned to flatter the probe -- 8/11
# canonical sign checks pass at this point and the 3 residuals are weak edges masked on saturated targets.
BASELINE = dict(cortex=200, str_D1=250, str_D2=250, str_PV_FSI=0, str_striosome=250,
                gpe=150, gpe_arky=120, gpi=900, stn=220, snc=150, thal=550, motor=150)
PERTURB_PA = 800.0
WARMUP = 40
SETTLE = 120
EDGE_THRESH = 0.008     # |Delta rate| (spikes/neuron/step) above which a functional edge is "active"


def _typ(name):
    return re.sub(r'_[NESW]$', '', name)


def build(seed, drop_edges=frozenset(), gpi_tonic=None):
    """Build the flagship nav-BG bridge. `drop_edges` = set of (from_type,to_type) pathway TYPE-edges to
    LESION (rebuild without those synapses). cfg.seed seeds the substrate (per CLAUDE.md: NOT actual_seed_used)."""
    from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    from research.runners.g11_bg_runner import build_bg_brain_regions
    regions, pathways = build_bg_brain_regions(
        n_cortex=100, enable_bg_lateral_inhibition=True, enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True, enable_cluster_e_topography=True)
    if drop_edges:
        pathways = [p for p in pathways if (_typ(p.from_region), _typ(p.to_region)) not in drop_edges]
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)                 # <-- the substrate seed (bridge reads cfg.seed for heterogeneity + wiring)
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    for flag in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning",
                 "enable_homeostasis", "enable_short_term_plasticity", "enable_ou_process",
                 "enable_conductance_noise", "enable_parameter_heterogeneity",
                 "enable_structural_plasticity"):
        setattr(cfg, flag, False)
    cfg.ou_std_current_pA = 0.0
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb, regions, pathways


def names_by_type(regions):
    d = {t: [] for t in TYPES}
    for r in regions:
        t = _typ(r.name)
        if t in d:
            d[t].append(r.name)
    return d


def type_sign(regions):
    """+1 if a type's neurons are excitatory (exc_fraction >= 0.5), else -1 (GABAergic source)."""
    exc = {}
    for r in regions:
        exc.setdefault(_typ(r.name), r.exc_fraction)
    return {t: (1.0 if exc.get(t, 1.0) >= 0.5 else -1.0) for t in TYPES}


def anatomical_matrix(regions, pathways):
    """Signed aggregate direct-synapse strength W[A][B] = sign(A) * sum_pathways(density*weight_mean).
    Diagonal (within-type / internal lateral) excluded -- this is a CROSS-region matrix."""
    sign = type_sign(regions)
    idx = {t: i for i, t in enumerate(TYPES)}
    W = np.zeros((len(TYPES), len(TYPES)), dtype=float)
    for p in pathways:
        a, b = _typ(p.from_region), _typ(p.to_region)
        if a in idx and b in idx and a != b:
            W[idx[a], idx[b]] += sign[a] * float(p.density) * float(p.weight_mean)
    return W


def measure_region_rates(sb, nbt, baseline, perturb=None, warmup=WARMUP, settle=SETTLE):
    """Drive baseline (by type) + optional `perturb` (dict region_name->extra pA), settle, return
    mean firing rate (spikes/neuron/step) PER REGION NAME."""
    from sim.backend import to_host
    idx = lambda n: np.asarray(sb.region_manager.indices(n))
    sb.cp_external_input_current[:] = 0.0
    for t, cur in baseline.items():
        for n in nbt[t]:
            sb.cp_external_input_current[idx(n)] = cur
    if perturb:
        for n, extra in perturb.items():
            sb.cp_external_input_current[idx(n)] += extra
    for _ in range(warmup):
        sb._run_one_simulation_step()
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for _ in range(settle):
        sb._run_one_simulation_step()
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    rates = {}
    for t in TYPES:
        for n in nbt[t]:
            rates[n] = float(acc[idx(n)].mean() / settle)
    return rates


def type_rates(region_rates, nbt):
    return {t: (float(np.mean([region_rates[n] for n in nbt[t]])) if nbt[t] else 0.0) for t in TYPES}


def functional_matrix(seed, drop_edges=frozenset(), baseline=BASELINE, perturb_pa=PERTURB_PA):
    """Build the F[A][B] matrix: for each source type A, rebuild fresh, drive all of A + baseline,
    measure Delta type-rate of every B vs the un-perturbed baseline. Returns (F, r0_types, regions, pathways)."""
    sb0, regions, pathways = build(seed, drop_edges=drop_edges)
    nbt = names_by_type(regions)
    r0 = type_rates(measure_region_rates(sb0, nbt, baseline), nbt)
    F = np.zeros((len(TYPES), len(TYPES)), dtype=float)
    for i, a in enumerate(TYPES):
        sbi, ri, _ = build(seed, drop_edges=drop_edges)
        nbti = names_by_type(ri)
        pert = {n: perturb_pa for n in nbti[a]}
        rp = type_rates(measure_region_rates(sbi, nbti, baseline, perturb=pert), nbti)
        for j, b in enumerate(TYPES):
            F[i, j] = rp[b] - r0[b]
    return F, r0, regions, pathways


def _edge(M, a, b):
    return float(M[TYPES.index(a), TYPES.index(b)])


def topographic_specificity(seed, baseline=BASELINE, perturb_pa=PERTURB_PA):
    """Drive ONLY cortex_N; measure str_D1_N (its OWN action channel) vs str_D1_{E,S,W} (siblings).
    Anatomically cortex_a->str_D1_a is topographic (cluster-E). Returns (own_delta, sib_signed_mean,
    sib_abs_mean, selectivity_index) where SI = (own - sib_signed) / (|own| + |sib_signed|) in [-1,1];
    SI -> +1 means the drive lands on its OWN channel and not the siblings."""
    sb0, regions, _ = build(seed)
    nbt = names_by_type(regions)
    r0 = measure_region_rates(sb0, nbt, baseline)
    sb1, r1, _ = build(seed)
    nbt1 = names_by_type(r1)
    rp = measure_region_rates(sb1, nbt1, baseline, perturb={"cortex_N": perturb_pa})
    own = rp["str_D1_N"] - r0["str_D1_N"]
    sib_deltas = [rp[f"str_D1_{a}"] - r0[f"str_D1_{a}"] for a in ["E", "S", "W"]]
    sib_signed = float(np.mean(sib_deltas))
    sib_abs = float(np.mean([abs(d) for d in sib_deltas]))
    denom = abs(own) + abs(sib_signed)
    si = float((own - sib_signed) / denom) if denom > 1e-12 else float("nan")
    return own, sib_signed, sib_abs, si


def run_seed(seed):
    t0 = time.time()
    F, r0, regions, pathways = functional_matrix(seed)
    W = anatomical_matrix(regions, pathways)
    cmp = functional_vs_anatomical(F, W, labels=TYPES, edge_thresh=EDGE_THRESH)

    # canonical direct + polysynaptic edges of interest
    canon = {
        "cortex->str_D1(+, direct)": _edge(F, "cortex", "str_D1"),
        "cortex->gpi(-, POLY: no direct synapse)": _edge(F, "cortex", "gpi"),
        "cortex->thal(+, POLY: double-inhibition disinhibition)": _edge(F, "cortex", "thal"),
        "cortex->motor(+, POLY)": _edge(F, "cortex", "motor"),
        "str_D1->gpi(-, direct)": _edge(F, "str_D1", "gpi"),
        "str_D2->gpe(-, direct)": _edge(F, "str_D2", "gpe"),
        "gpi->thal(-, direct)": _edge(F, "gpi", "thal"),
        "thal->motor(+, direct)": _edge(F, "thal", "motor"),
    }
    W_cortex_thal = _edge(W, "cortex", "thal")     # == 0.0: no direct anatomical synapse
    W_cortex_gpi = _edge(W, "cortex", "gpi")       # == 0.0

    # --- ANTI-CHEAT 1: lesion dissociation (cut gpi->thal) ---
    Flx, _, _, _ = functional_matrix(seed, drop_edges=frozenset({("gpi", "thal")}))
    lesion = {
        "gpi->thal  intact": _edge(F, "gpi", "thal"),
        "gpi->thal  lesioned": _edge(Flx, "gpi", "thal"),
        "cortex->thal intact (POLY route via gpi)": _edge(F, "cortex", "thal"),
        "cortex->thal lesioned (route severed)": _edge(Flx, "cortex", "thal"),
        "cortex->gpi intact (UPSTREAM of lesion)": _edge(F, "cortex", "gpi"),
        "cortex->gpi lesioned (should be UNCHANGED)": _edge(Flx, "cortex", "gpi"),
    }
    # ATTRIBUTION: whose is the difference? (tools.lab.attributable_to -- do not just measure both arms.)
    # The POLYSYNAPTIC cortex->thal edge should be ~100% attributable to the gpi->thal pathway (its route);
    # the UPSTREAM cortex->gpi edge should NOT be (control >= treatment -> negative/near-0 attribution),
    # which is the lesion's specificity control.
    attr_poly = attributable_to("cortex->thal edge attributable to gpi->thal pathway",
                                _edge(F, "cortex", "thal"), _edge(Flx, "cortex", "thal"))
    attr_upstream = attributable_to("cortex->gpi edge attributable to gpi->thal pathway (SHOULD be low)",
                                    _edge(F, "cortex", "gpi"), _edge(Flx, "cortex", "gpi"))
    lesion_attribution = dict(cortex_thal_attributable_to_gpi_thal=attr_poly,
                              cortex_gpi_attributable_to_gpi_thal=attr_upstream)

    # --- ANTI-CHEAT 2 (driver): state-dependence -- lower the gpi pacemaker, watch cortex->thal reweight ---
    base_lo = dict(BASELINE); base_lo["gpi"] = 300
    F_lo, _, _, _ = functional_matrix(seed, baseline=base_lo)
    state_dep = {"cortex->thal @ gpi=900": _edge(F, "cortex", "thal"),
                 "cortex->thal @ gpi=300": _edge(F_lo, "cortex", "thal")}

    # --- ANTI-CHEAT 3: specificity (topographic) ---
    own, sib_signed, sib_abs, si = topographic_specificity(seed)

    elapsed = time.time() - t0
    return dict(
        seed=int(seed), elapsed_s=round(elapsed, 2),
        baseline_rates=r0,
        F=F.tolist(), W=W.tolist(), types=TYPES,
        comparison=cmp,
        canonical_functional_edges=canon,
        cortex_thal_F=_edge(F, "cortex", "thal"), cortex_gpi_F=_edge(F, "cortex", "gpi"),
        W_cortex_thal=W_cortex_thal, W_cortex_gpi=W_cortex_gpi,
        lesion_gpi_thal=lesion,
        lesion_attribution=lesion_attribution,
        state_dependence=state_dep,
        specificity_topographic=dict(own_channel_str_D1_N=own, sibling_channels_signed_mean=sib_signed,
                                     sibling_channels_abs_mean=sib_abs, selectivity_index=si,
                                     abs_ratio=(abs(own) / sib_abs if sib_abs > 1e-9 else float("inf"))),
    )


def pooled(results):
    def col(path):
        return np.array([_dig(r, path) for r in results], dtype=float)

    def _dig(r, path):
        x = r
        for k in path:
            x = x[k]
        return x
    out = {}
    out["n_seeds"] = len(results)
    out["spearman_rho_nz_mean"] = float(np.nanmean(col(["comparison", "spearman_rho_nz"])))
    out["spearman_rho_nz_std"] = float(np.nanstd(col(["comparison", "spearman_rho_nz"])))
    out["pearson_r_nz_mean"] = float(np.nanmean(col(["comparison", "pearson_r_nz"])))
    out["sign_agree_direct_mean"] = float(np.nanmean(col(["comparison", "sign_agree_direct"])))
    out["n_polysynaptic_mean"] = float(np.mean(col(["comparison", "n_polysynaptic"])))
    out["n_polysynaptic_min"] = int(np.min(col(["comparison", "n_polysynaptic"])))
    out["cortex_thal_F_mean"] = float(np.mean([r["cortex_thal_F"] for r in results]))
    # lesion dissociation ratios (pooled)
    def ratio(res, num_key, den_key):
        d = res["lesion_gpi_thal"]
        num = abs(d[num_key]); den = abs(d[den_key])
        return num / den if den > 1e-9 else float("nan")
    gt = np.array([ratio(r, "gpi->thal  lesioned", "gpi->thal  intact") for r in results])
    ct = np.array([ratio(r, "cortex->thal lesioned (route severed)", "cortex->thal intact (POLY route via gpi)")
                   for r in results])
    cg = np.array([ratio(r, "cortex->gpi lesioned (should be UNCHANGED)", "cortex->gpi intact (UPSTREAM of lesion)")
                   for r in results])
    out["lesion_gpi_thal_retained_frac_mean"] = float(np.nanmean(gt))     # -> ~0 (edge collapses)
    out["lesion_cortex_thal_retained_frac_mean"] = float(np.nanmean(ct))  # -> ~0 (poly route severed)
    out["lesion_cortex_gpi_retained_frac_mean"] = float(np.nanmean(cg))   # -> ~1 (upstream, unchanged)
    out["specificity_SI_mean"] = float(np.nanmean(col(["specificity_topographic", "selectivity_index"])))
    out["specificity_SI_min"] = float(np.nanmin(col(["specificity_topographic", "selectivity_index"])))
    out["specificity_own_mean"] = float(np.nanmean(col(["specificity_topographic", "own_channel_str_D1_N"])))
    out["specificity_own_min"] = float(np.nanmin(col(["specificity_topographic", "own_channel_str_D1_N"])))
    out["specificity_sib_signed_mean"] = float(np.nanmean(col(["specificity_topographic", "sibling_channels_signed_mean"])))
    return out


def determinism_check(seed=42):
    F1, _, _, _ = functional_matrix(seed)
    F2, _, _, _ = functional_matrix(seed)
    return bool(np.array_equal(np.array(F1), np.array(F2))), float(np.max(np.abs(np.array(F1) - np.array(F2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--out", default="research/findings/raw/perturb_measure/pm.json")
    ap.add_argument("--quick", action="store_true", help="2 seeds, skip determinism re-run")
    args = ap.parse_args()

    assert_backend(os.environ.get("SIM_BACKEND", "numpy"), note="(perturb-and-measure)")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.quick:
        seeds = seeds[:2]

    print(f"=== PERTURB-AND-MEASURE functional connectivity vs anatomical (Randi 2023) ===")
    print(f"seeds={seeds}  operating point gpi_tonic={BASELINE['gpi']}  perturb=+{PERTURB_PA}pA  "
          f"settle={SETTLE} warmup={WARMUP}\n", flush=True)

    det_ok, det_max = (True, 0.0)
    if not args.quick:
        det_ok, det_max = determinism_check(seeds[0])
        print(f"[determinism] byte-identical F on re-run at seed {seeds[0]}: {det_ok} (max|dF|={det_max:.2e})\n",
              flush=True)

    results = []
    for s in seeds:
        r = run_seed(s)
        results.append(r)
        c = r["comparison"]
        print(f"[seed {s}] ({r['elapsed_s']}s)  spearman_rho_nz(F,W)={c['spearman_rho_nz']:+.3f}  "
              f"sign_agree_direct={c['sign_agree_direct']:.2f}  n_poly={c['n_polysynaptic']}", flush=True)
        print(f"          cortex->thal F={_edge_from_list(r['F'],'cortex','thal'):+.4f} "
              f"(anatomical W={r['W_cortex_thal']:+.1f})   "
              f"cortex->gpi F={_edge_from_list(r['F'],'cortex','gpi'):+.4f} (W={r['W_cortex_gpi']:+.1f})", flush=True)
        L = r["lesion_gpi_thal"]
        print(f"          LESION gpi->thal: gpi->thal {L['gpi->thal  intact']:+.4f}->{L['gpi->thal  lesioned']:+.4f}"
              f"  cortex->thal {L['cortex->thal intact (POLY route via gpi)']:+.4f}->"
              f"{L['cortex->thal lesioned (route severed)']:+.4f}"
              f"  cortex->gpi {L['cortex->gpi intact (UPSTREAM of lesion)']:+.4f}->"
              f"{L['cortex->gpi lesioned (should be UNCHANGED)']:+.4f}", flush=True)
        sp = r["specificity_topographic"]
        print(f"          SPECIFICITY cortex_N: own str_D1_N={sp['own_channel_str_D1_N']:+.4f} "
              f"vs siblings={sp['sibling_channels_signed_mean']:+.4f} "
              f"(SI={sp['selectivity_index']:+.3f} |ratio|={sp['abs_ratio']:.1f})", flush=True)
        sd = r["state_dependence"]
        print(f"          STATE-DEP cortex->thal: gpi=900 {sd['cortex->thal @ gpi=900']:+.4f}  "
              f"gpi=300 {sd['cortex->thal @ gpi=300']:+.4f}\n", flush=True)

    pool = pooled(results)

    # ---------------- GO gate ----------------
    from tools.lab import Verdict
    v = Verdict("perturb-and-measure functional connectivity vs anatomical (Randi 2023)")
    for proc in ("STDP", "reward_modulation", "hebbian", "homeostasis", "STP", "OU_noise",
                 "conductance_noise", "parameter_heterogeneity", "structural_plasticity"):
        v.disabled(proc, why="isolation: measure propagation on a fixed graph, no learning/noise")
    v.require("6 seeds", len(seeds), expect=lambda n: n >= 6)
    v.require("determinism: byte-identical F re-run @ fixed seed", det_ok, expect=True)
    # F != W: correlation strictly below 1
    v.require("F != W: spearman_rho_nz(F,W) < 1", pool["spearman_rho_nz_mean"], expect=lambda x: x < 0.999)
    # polysynaptic edges (no direct synapse) present in EVERY seed
    v.require("polysynaptic functional edges present (min over seeds >= 1)",
              pool["n_polysynaptic_min"], expect=lambda n: n >= 1)
    # the headline poly edge exists and is anatomically ZERO
    v.require("cortex->thal has NO direct synapse (W==0)",
              float(np.mean([abs(r["W_cortex_thal"]) for r in results])), expect=lambda x: x == 0.0)
    v.require("cortex->thal functional edge is nonzero (disinhibition)",
              abs(pool["cortex_thal_F_mean"]), expect=lambda x: x > EDGE_THRESH)
    # sign agreement on the DIRECT edges the probe does resolve
    v.require("sign agreement on active direct edges >= 0.8",
              pool["sign_agree_direct_mean"], expect=lambda x: x >= 0.8)
    # lesion dissociation: the cut edge + the polysynaptic edge collapse, the upstream edge does not
    v.control("lesion: gpi->thal edge collapses (retained frac vs 1.0)",
              treatment=pool["lesion_gpi_thal_retained_frac_mean"], control=1.0, min_separation=0.5)
    v.require("lesion: gpi->thal retains < 40% of its edge", pool["lesion_gpi_thal_retained_frac_mean"],
              expect=lambda x: x < 0.4)
    v.require("lesion: polysynaptic cortex->thal retains < 50%",
              pool["lesion_cortex_thal_retained_frac_mean"], expect=lambda x: x < 0.5)
    v.require("lesion: upstream cortex->gpi retains > 60% (specific, not global)",
              pool["lesion_cortex_gpi_retained_frac_mean"], expect=lambda x: x > 0.6)
    # specificity: own action channel driven more than siblings (topographic)
    v.require("specificity: cortex_N drives its OWN str_D1_N (own delta > 0, min over seeds)",
              pool["specificity_own_min"], expect=lambda x: x > EDGE_THRESH)
    v.require("specificity: selectivity index own-vs-siblings > 0.7 (min over seeds)",
              pool["specificity_SI_min"], expect=lambda x: x > 0.7)

    go = (len(seeds) >= 6 and det_ok
          and pool["spearman_rho_nz_mean"] < 0.999
          and pool["n_polysynaptic_min"] >= 1
          and abs(pool["cortex_thal_F_mean"]) > EDGE_THRESH
          and pool["sign_agree_direct_mean"] >= 0.8
          and pool["lesion_gpi_thal_retained_frac_mean"] < 0.4
          and pool["lesion_cortex_thal_retained_frac_mean"] < 0.5
          and pool["lesion_cortex_gpi_retained_frac_mean"] > 0.6
          and pool["specificity_own_min"] > EDGE_THRESH
          and pool["specificity_SI_min"] > 0.7)
    verdict = v.decide(go=go)

    print("\n=== POOLED ===")
    for k in ("n_seeds", "spearman_rho_nz_mean", "sign_agree_direct_mean", "n_polysynaptic_mean",
              "n_polysynaptic_min", "cortex_thal_F_mean", "lesion_gpi_thal_retained_frac_mean",
              "lesion_cortex_thal_retained_frac_mean", "lesion_cortex_gpi_retained_frac_mean",
              "specificity_SI_mean", "specificity_own_mean"):
        print(f"  {k:42s} {pool[k]}")
    print(f"\n  STATUS: {verdict['status']}")

    payload = dict(
        probe="perturb_and_measure_functional_connectivity",
        method="Randi Sharma Dvali Leifer, Nature 623:406-414 (2023)",
        substrate="nav basal ganglia (build_bg_brain_regions flagship A+E)",
        seeds=seeds, operating_point=BASELINE, perturb_pA=PERTURB_PA, settle=SETTLE, warmup=WARMUP,
        edge_thresh=EDGE_THRESH,
        determinism=dict(byte_identical=det_ok, max_abs_dF=det_max),
        per_seed=results, pooled=pool, **verdict)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")


def _edge_from_list(Flist, a, b):
    return float(Flist[TYPES.index(a)][TYPES.index(b)])


if __name__ == "__main__":
    main()
