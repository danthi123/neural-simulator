"""Inductive-coverage (premise-diversity) de-risk -- category-based induction on a spiking substrate.

FACULTY: premise-integrating induction. The strength of a generalization to a superordinate category ("all
birds have property P") scales with premise COVERAGE / DIVERSITY, not premise count alone (Osherson, Smith,
Wilkie, Lopez & Shafir 1990, "Category-Based Induction," Psychological Review 97(2):185-200 -- the similarity-
COVERAGE model; confirmed via WebSearch, not in the local corpus).

BIOLOGY BINDING: research/biology/inductive-coverage-premise-diversity.md
  * Kandel Ch 30 states the coverage principle outright for motor primitives -- "[t]he makeup of the population
    of such primitives then determines which structural constraints are imposed on learning ... a behavior for
    which the [motor] system has many primitives will be easy to learn." Breadth of the active population governs
    generalization; category-based induction is the semantic-memory instance.
  * Kandel Ch 17 (Sensory Coding) names the COMPANION PROCESS: inhibitory networks "allow the context of a
    stimulus to modify the strength of excitation evoked by that stimulus, an important process called
    normalization." A saturating/normalizing response is what makes broad coverage beat concentrated potentiation.
  * Here that concavity is supplied BRAIN-SIDE by the soft-bound Hebbian rule itself (delta_w = rate*(w_max - w)):
    a synapse fired by TWO premises reaches w2 < 2*w1. Diverse premises spread w1 across two subcategory cores;
    within-subcategory premises concentrate w2 on one. Diverse wins iff 2*w1 > w2 -- guaranteed by the concavity.

MECHANISM (one SimulationBridge, two regions -- reuse the convergence-runner pattern):
  * CONCEPT region -- population codes with STRUCTURE: a shared per-SUPERORDINATE core (all members),
    a per-SUBCATEGORY core (members of one subcat), and a per-concept UNIQUE tail. Same-subcat concepts overlap
    on the subcat core; all concepts overlap on the category core. Scattered (permuted) assignment (kills a
    contiguous-layout index bias).
  * PROPERTY region -- two disjoint population blocks: the TAUGHT property assembly + a FOIL (never taught).
  * plastic concept->property pathway (near-floor init, rate-Hebbian coincidence = the property learning).

PREMISES (the manipulation -- ONLY the premise set differs across arms; held-out members are never premises):
  * 1-premise            : {a0}                (1 premise, 1 subcat covered)
  * within-subcategory   : {a0, a1}            (2 premises, SAME subcat -> still 1 subcat covered)
  * diverse              : {a0, b0}            (2 premises, DIFFERENT subcats -> 2 subcats covered)
Training co-activates each premise's concept code + the TAUGHT property block, so rate-Hebbian potentiates the
concept->property synapses for the premise-active concept neurons.

INDUCTION READOUT (the conclusion = the superordinate "all category members"): drive each HELD-OUT category
member's concept code (one held-out per subcat, spanning the superordinate; none are premises), read the TAUGHT
property block's GRADED population depolarization (the documented rate-code-wall read -- point-neuron property
cells do not spike from concept drive alone, so the assembly's subthreshold depolarization above rest IS its
neural response), minus the FOIL block (removes any non-specific drive). Average over the held-out set =
induction strength to the superordinate. NO host similarity formula: the strength is the property assembly's own
membrane response through LEARNED synapses.

GATE (6 seeds 42/43/44/100/101/102 for the real sweep; this file runs a 1-seed numpy SMOKE in the foreground):
  GO       : COVERAGE effect -- diverse-2 > within-2 (MATCHED premise count, isolates coverage from count) by a
             positive margin AND diverse-2 > 1-premise, on the median seed; AND both anti-cheats fire:
             (1) PERMUTED concept codes (category/subcat sharing destroyed) -> all arms collapse to ~floor and the
                 diverse>within ordering vanishes (ordering-collapse);
             (2) PREMISE-LESION (no co-activation training) -> generalization collapses to ~floor for every arm.
  PARTIAL  : coverage ordering holds but weakly / one anti-cheat marginal (localize: operating point -- keep
             hebbian_rate*epochs in the SUB-SATURATION regime, see the binding's constraints_config).
  NEGATIVE : no coverage effect even in the sub-saturation regime (diverse-2 <= within-2).

NOTE on the honest ordering: premise MONOTONICITY (2 premises > 1) co-exists via the same concavity on the
category core, so within-2 typically also beats 1-premise. The load-bearing DIVERSITY claim is therefore the
MATCHED-count contrast diverse-2 vs within-2 -- that isolates coverage. Reported explicitly.

Reuse-by-import: the SimulationBridge two-region + rate-Hebbian + graded-depol-readout pattern from
`_genfrontier_onsubstrate_convergence_derisk`. NO sim/ edits.

Run (1-seed numpy SMOKE, foreground):  python -u -m research.runners._inductive_coverage_derisk --smoke
Run (6-seed real sweep, GPU):          SIM_BACKEND=cupy python -u -m research.runners._inductive_coverage_derisk --seeds 42,43,44,100,101,102
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402

_V_REST = -65.0  # Izhikevich resting potential; depolarization above rest = the graded population response.


# ===========================================================================
# Category structure: ONE superordinate, K subcategories x M members.
# ===========================================================================
def category_layout(n_sub, n_per_sub):
    F = n_sub * n_per_sub
    sub_ids = np.repeat(np.arange(n_sub), n_per_sub)  # concept -> subcategory id
    return F, sub_ids


def structured_concept_sets(n_concept, n_sub, n_per_sub, n_cat_core, n_sub_core, n_uniq, seed, permute=False):
    """For each concept i: index set = SUPERORDINATE core (n_cat_core, shared by ALL members) + SUBCATEGORY core
    (n_sub_core, shared within its subcat) + UNIQUE tail (n_uniq, disjoint). Scattered across the region via a
    permutation (kills a contiguous-layout index bias).

    permute=True is the ANTI-CHEAT: every concept gets its OWN disjoint block of the SAME total size (no shared
    category/subcat cores) -- the similarity STRUCTURE is destroyed while cardinality is matched, so held-out
    members share no neurons with premises and the coverage effect must vanish."""
    F, sub_ids = category_layout(n_sub, n_per_sub)
    rng = np.random.default_rng(seed)
    code_size = n_cat_core + n_sub_core + n_uniq
    if permute:
        n_need = F * code_size
        assert n_need <= n_concept, f"concept region too small (permuted): need {n_need}, have {n_concept}"
        perm = rng.permutation(n_concept)[:n_need]
        sets = [perm[i * code_size:(i + 1) * code_size] for i in range(F)]
        return sets, sub_ids
    n_need = n_cat_core + n_sub * n_sub_core + F * n_uniq
    assert n_need <= n_concept, f"concept region too small: need {n_need}, have {n_concept}"
    perm = rng.permutation(n_concept)[:n_need]
    cat_core = perm[:n_cat_core]
    base = n_cat_core
    sub_core = [perm[base + s * n_sub_core: base + (s + 1) * n_sub_core] for s in range(n_sub)]
    base2 = n_cat_core + n_sub * n_sub_core
    sets = []
    for i in range(F):
        uniq = perm[base2 + i * n_uniq: base2 + (i + 1) * n_uniq]
        sets.append(np.concatenate([cat_core, sub_core[sub_ids[i]], uniq]))
    return sets, sub_ids


# ===========================================================================
# The bridge: concept region (index-set codes) + property region (2 disjoint blocks: taught + foil).
# ===========================================================================
def build_bridge(n_concept, n_prop_per, seed, a):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    n_prop = 2 * n_prop_per  # block 0 = TAUGHT property, block 1 = FOIL
    cfg.brain_regions = [
        BrainRegion(name="concept", n_neurons=n_concept, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="property", n_neurons=n_prop, exc_fraction=1.0, internal_density=0.0),
    ]
    cfg.region_pathways = [RegionPathway(from_region="concept", to_region="property", density=1.0,
                                         weight_mean=0.05, weight_jitter=0.0, plastic=True)]
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed  # cfg.seed is what actually seeds the substrate
    cfg.enable_ou_process = False
    # RATE-Hebbian coincidence (NOT STDP): concept(X) AND property fire together = a symmetric coincidence, which
    # STDP's delta_t kernel reads as ~0. Soft-bound Hebbian delta = rate*(max - w) accumulates the co-occurrence
    # AND supplies the concavity (w2 < 2*w1) that makes coverage beat concentration.
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = a.hebbian_rate
    cfg.hebbian_max_weight = a.hebbian_max
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_weight_decay = 0.00001
    rt = RuntimeState(); rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt,
                              gpu_config=GPUConfig())
    bridge._initialize_simulation_data()
    conc_region = np.asarray(bridge.region_manager.indices("concept"))
    prop_region = np.asarray(bridge.region_manager.indices("property"))
    prop_blocks = prop_region.reshape(2, n_prop_per)  # 0 = taught, 1 = foil
    return bridge, conc_region, prop_region, prop_blocks


def _xp_of(bridge):
    try:
        import cupy as cp  # noqa
        if isinstance(getattr(bridge, "cp_membrane_potential_v", None), cp.ndarray):
            return cp
    except Exception:
        pass
    return None


_DYN_ATTRS = ("cp_membrane_potential_v", "cp_recovery_variable_u", "cp_conductance_g_e",
              "cp_conductance_g_i", "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise",
              "cp_external_input_current")


def snapshot_dynamic_state(bridge):
    """Copy the DYNAMIC (non-weight) state -- membrane potential, recovery variable, synaptic conductances,
    external drive -- so it can be restored to a clean resting baseline before a read-out. Taken right after
    build (rest state). Synaptic WEIGHTS are a separate array and are NOT touched, so training persists.

    Why this is load-bearing: training drives the taught assembly hard, leaving it PRIMED (depolarized, elevated
    conductance). Without a reset, any read-out drive tips the primed block over threshold regardless of whether
    the cue overlaps the premises -- the permuted-code anti-cheat exposed exactly this (it fired at ceiling with
    zero premise overlap). Restoring the rest state makes the read reflect ONLY the learned weights + the cue."""
    snap = {}
    for name in _DYN_ATTRS:
        arr = getattr(bridge, name, None)
        if arr is not None:
            snap[name] = arr.copy()
    return snap


def restore_dynamic_state(bridge, snap):
    for name, arr in snap.items():
        getattr(bridge, name)[:] = arr


def _set_drive(bridge, xp, conc_region, prop_region, conc_local, conc_scale,
               prop_block_local=None, prop_scale=0.0):
    n_conc = conc_region.shape[0]
    full_conc = np.zeros(n_conc, np.float32)
    full_conc[conc_local] = conc_scale
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[conc_region] = xp.asarray(full_conc) if xp is not None else full_conc
    if prop_block_local is not None and prop_scale > 0.0:
        n_prop = prop_region.shape[0]
        full_prop = np.zeros(n_prop, np.float32)
        full_prop[prop_block_local] = prop_scale
        bridge.cp_external_input_current[prop_region] = xp.asarray(full_prop) if xp is not None else full_prop


def _prop_depol_step(bridge, prop_region):
    """Per-step property-region depolarization above rest -- the property assembly's OWN graded membrane
    response (the documented rate-code-wall read: a point-neuron property cell does not reliably SPIKE from
    concept synaptic drive alone from a clean rest state, so its subthreshold depolarization is its neural
    response to the cue). Read after a dynamic-state reset (snapshot_dynamic_state), so it reflects the LEARNED
    weights via the cue, not training-priming."""
    v = getattr(bridge, "cp_membrane_potential_v", None)
    if v is None:
        return np.zeros(prop_region.shape[0], np.float64)
    vh = np.asarray(to_host(v))[prop_region].astype(np.float64)
    return (vh - _V_REST).clip(min=0.0)


def train_property(bridge, xp, conc_region, prop_region, prop_blocks, conc_sets, premises, a):
    """Co-activate each PREMISE concept's code + the TAUGHT property block, repeated, so rate-Hebbian potentiates
    the concept->property synapses for the premise-active concept neurons."""
    taught_local = prop_blocks[0] - prop_region[0]
    diag = {"conc": 0, "prop": 0}
    for ep in range(a.epochs):
        order = np.random.RandomState(a.seed_base * 7 + ep).permutation(np.asarray(premises))
        for si, p in enumerate(order):
            conc_local = np.asarray(conc_sets[p]) - conc_region[0]
            _set_drive(bridge, xp, conc_region, prop_region, conc_local, a.conc_scale,
                       prop_block_local=taught_local, prop_scale=a.prop_scale)
            first = (ep == 0 and si == 0)
            for _ in range(a.scene_steps):
                bridge._run_one_simulation_step()
                if first:
                    fs = getattr(bridge, "cp_firing_states", None)
                    if fs is not None:
                        h = np.asarray(to_host(fs))
                        diag["conc"] += int(h[conc_region].sum())
                        diag["prop"] += int(h[prop_region].sum())
    bridge.cp_external_input_current[:] = 0.0
    return diag


def read_property_response(bridge, xp, conc_region, prop_region, prop_blocks, conc_idx, scale, steps):
    """Drive ONLY the concept code and accumulate the property region's graded depolarization above rest.
    Returns (taught_depol, foil_depol) -- per-block mean depolarization summed over `steps`. The CALLER restores
    the rest state before this."""
    conc_local = np.asarray(conc_idx) - conc_region[0]
    _set_drive(bridge, xp, conc_region, prop_region, conc_local, scale)
    acc = np.zeros(prop_region.shape[0], np.float64)
    for _ in range(steps):
        bridge._run_one_simulation_step()
        acc += _prop_depol_step(bridge, prop_region)
    bridge.cp_external_input_current[:] = 0.0
    block_local = prop_blocks - prop_region[0]  # 2 x n_prop_per
    per_block = acc[block_local].mean(axis=1)
    return float(per_block[0]), float(per_block[1])  # taught, foil


def taught_depol_over_heldout(bridge, xp, conc_region, prop_region, prop_blocks, conc_sets, heldout, a,
                              rest_snap):
    """Per-HELD-OUT-member taught-block graded depolarization (list, len=len(heldout)) + the mean foil-block
    depol (a specificity reference). The rest state is restored before EACH member read (no cross-cue or
    training-priming contamination -- see snapshot_dynamic_state)."""
    taught_list, foil_list = [], []
    for m in heldout:
        restore_dynamic_state(bridge, rest_snap)
        taught, foil = read_property_response(bridge, xp, conc_region, prop_region, prop_blocks,
                                              conc_sets[m], a.read_scale, a.read_steps)
        taught_list.append(float(taught)); foil_list.append(float(foil))
    return taught_list, float(np.mean(foil_list))


# ===========================================================================
# floor baseline + one arm.
# ===========================================================================
def floor_baseline(heldout, conc_sets, n_concept, seed, a):
    """Per-held-out taught-block depol on an UNTRAINED bridge (all concept->property synapses at floor). This is
    the intrinsic + floor-driven response of the taught block to each held-out cue -- the reference the learned
    increment is measured above. Offset-free by construction (same block, same cue, only the WEIGHTS differ from
    a trained arm)."""
    bridge, conc_region, prop_region, prop_blocks = build_bridge(n_concept, a.n_prop_per, seed, a)
    xp = _xp_of(bridge)
    rest_snap = snapshot_dynamic_state(bridge)
    taught_list, _foil = taught_depol_over_heldout(bridge, xp, conc_region, prop_region, prop_blocks,
                                                   conc_sets, heldout, a, rest_snap)
    return np.asarray(taught_list, np.float64)


def run_arm(premises, heldout, conc_sets, base, n_concept, seed, a):
    """Build a fresh bridge, TRAIN on `premises` (co-activate premise concept + taught property), then read the
    taught-block depol over held-out. Induction strength = mean over held-out of (taught depol - floor base) --
    the LEARNING-INDUCED increment above the untrained floor. Returns (strength, per_held_increment, foil, diag)."""
    bridge, conc_region, prop_region, prop_blocks = build_bridge(n_concept, a.n_prop_per, seed, a)
    xp = _xp_of(bridge)
    rest_snap = snapshot_dynamic_state(bridge)  # clean rest state (pre-training), restored before each read
    diag = train_property(bridge, xp, conc_region, prop_region, prop_blocks, conc_sets, premises, a)
    taught_list, foil = taught_depol_over_heldout(bridge, xp, conc_region, prop_region, prop_blocks,
                                                  conc_sets, heldout, a, rest_snap)
    inc = np.asarray(taught_list, np.float64) - base
    return float(inc.mean()), [float(v) for v in inc], foil, diag


def run_seed(seed, a):
    """Build the arms for one seed. Premises/held-out are fixed by construction; only the premise SET differs."""
    def cid(s, j):
        return s * a.n_per_sub + j
    conc_sets, _ = structured_concept_sets(a.n_concept, a.n_sub, a.n_per_sub, a.n_cat_core, a.n_sub_core,
                                           a.n_uniq, seed, permute=False)
    conc_sets_perm, _ = structured_concept_sets(a.n_concept, a.n_sub, a.n_per_sub, a.n_cat_core, a.n_sub_core,
                                                a.n_uniq, seed, permute=True)
    # held-out: LAST member of every subcategory (spans the superordinate; never a premise)
    heldout = [cid(s, a.n_per_sub - 1) for s in range(a.n_sub)]
    prem_1 = [cid(0, 0)]                    # 1-premise
    prem_within = [cid(0, 0), cid(0, 1)]    # 2 premises, SAME subcat
    prem_diverse = [cid(0, 0), cid(1, 0)]   # 2 premises, DIFFERENT subcats

    base = floor_baseline(heldout, conc_sets, a.n_concept, seed, a)         # untrained-floor reference (structured)
    base_perm = floor_baseline(heldout, conc_sets_perm, a.n_concept, seed, a)  # untrained-floor (permuted codes)

    out = {"seed": seed}
    s1, i1, f1, d1 = run_arm(prem_1, heldout, conc_sets, base, a.n_concept, seed, a)
    sw, iw, fw, dw = run_arm(prem_within, heldout, conc_sets, base, a.n_concept, seed, a)
    sd, idv, fd, dd = run_arm(prem_diverse, heldout, conc_sets, base, a.n_concept, seed, a)
    out["one_premise"] = s1; out["within_subcat"] = sw; out["diverse"] = sd
    out["one_premise_inc"] = i1; out["within_inc"] = iw; out["diverse_inc"] = idv
    out["diag_diverse"] = dd
    # ANTI-CHEAT 1: permuted concept codes (category/subcat sharing destroyed) -- held-out shares no neurons with
    # premises, so the learned increment collapses to ~0 and the diverse>within ordering vanishes.
    sd_perm, _, _, _ = run_arm(prem_diverse, heldout, conc_sets_perm, base_perm, a.n_concept, seed, a)
    sw_perm, _, _, _ = run_arm(prem_within, heldout, conc_sets_perm, base_perm, a.n_concept, seed, a)
    out["diverse_permuted"] = sd_perm; out["within_permuted"] = sw_perm
    # ANTI-CHEAT 2: premise-lesion (no training) -- read a fresh untrained bridge vs the SAME floor base -> ~0.
    s_les, _, _, _ = run_arm([], heldout, conc_sets, base, a.n_concept, seed, a)  # empty premise set = no learning
    out["diverse_lesion"] = s_les
    return out


# ===========================================================================
# Verdict.
# ===========================================================================
def verdict(seed_rows, a):
    def med(k):
        return float(np.median([r[k] for r in seed_rows]))
    one = med("one_premise"); within = med("within_subcat"); diverse = med("diverse")
    div_perm = med("diverse_permuted"); win_perm = med("within_permuted")
    lesion = med("diverse_lesion")
    cov_margin = diverse - within          # THE coverage effect (matched premise count)
    dvs1 = diverse - one
    # anti-cheat thresholds: collapse = permuted/lesion strength is a small fraction of the diverse strength
    floor = max(1e-6, 0.10 * abs(diverse))
    ac1_ok = (div_perm < floor) and (abs(diverse - within) > abs(div_perm - win_perm))  # ordering collapses
    ac2_ok = (abs(lesion) < floor)
    coverage_ok = (cov_margin > a.margin_eps) and (dvs1 > a.margin_eps)

    print("\n================= INDUCTIVE-COVERAGE DE-RISK — VERDICT =================")
    print(f"seeds: {[r['seed'] for r in seed_rows]}  (median reported)")
    print(f"  induction strength (taught-block depol INCREMENT above untrained floor, held-out mean over subcats):")
    print(f"    1-premise            : {one:+.4f}")
    print(f"    within-subcategory(2): {within:+.4f}")
    print(f"    diverse(2)           : {diverse:+.4f}")
    print(f"  COVERAGE effect  diverse(2) - within(2)  = {cov_margin:+.4f}   (matched premise count; >{a.margin_eps} = coverage)")
    print(f"  diverse(2) - 1-premise                   = {dvs1:+.4f}")
    print(f"  premise-monotonicity within(2)-1-prem    = {within-one:+.4f}   (co-existing Osherson effect; not the claim)")
    print(f"  ANTI-CHEAT 1 permuted codes: diverse {div_perm:+.4f} within {win_perm:+.4f}  (both -> ~floor {floor:.4f}; ordering collapses) : {'PASS' if ac1_ok else 'FAIL'}")
    print(f"  ANTI-CHEAT 2 premise-lesion: strength {lesion:+.4f}  (-> ~floor {floor:.4f}) : {'PASS' if ac2_ok else 'FAIL'}")
    if coverage_ok and ac1_ok and ac2_ok:
        v = "GO"
    elif coverage_ok and (ac1_ok or ac2_ok):
        v = "PARTIAL"
    elif coverage_ok:
        v = "PARTIAL (coverage effect present; an anti-cheat did not fire — inspect operating point)"
    else:
        v = "NEGATIVE (no coverage effect: diverse(2) <= within(2))"
    print(f"  VERDICT: {v}")
    print("=======================================================================\n")
    return v, {"one": one, "within": within, "diverse": diverse, "cov_margin": cov_margin,
               "dvs1": dvs1, "div_perm": div_perm, "win_perm": win_perm, "lesion": lesion,
               "ac1_ok": ac1_ok, "ac2_ok": ac2_ok, "coverage_ok": coverage_ok}


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--smoke", action="store_true", help="1-seed numpy smoke (seed 42), small + fast")
    ap.add_argument("--out", default="")
    # category structure
    ap.add_argument("--n-sub", type=int, default=3)          # subcategories in the superordinate
    ap.add_argument("--n-per-sub", type=int, default=4)      # members per subcategory
    ap.add_argument("--n-concept", type=int, default=1024)   # concept region size (fits permuted anti-cheat: F*code_size)
    ap.add_argument("--n-cat-core", type=int, default=40)    # shared superordinate core
    ap.add_argument("--n-sub-core", type=int, default=24)    # per-subcategory core
    ap.add_argument("--n-uniq", type=int, default=12)        # per-concept unique tail
    ap.add_argument("--n-prop-per", type=int, default=100)   # property block size (taught / foil)
    # learning / drive (operating point: keep hebbian_rate*epochs sub-saturation, see biology binding)
    ap.add_argument("--hebbian-rate", type=float, default=0.05)
    ap.add_argument("--hebbian-max", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--scene-steps", type=int, default=16)
    ap.add_argument("--conc-scale", type=float, default=300.0)
    ap.add_argument("--prop-scale", type=float, default=600.0)
    ap.add_argument("--read-scale", type=float, default=300.0)
    ap.add_argument("--read-steps", type=int, default=16)
    ap.add_argument("--margin-eps", type=float, default=0.02)  # depol-increment units (mV*steps): coverage-gap threshold
    return ap


def main():
    a = build_args().parse_args()
    if a.smoke:
        a.seeds = "42"
        # keep the smoke small + fast on numpy
        a.epochs = min(a.epochs, 6)
    a.seed_base = 12345
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    t0 = time.time()
    rows = []
    for sd in seeds:
        r = run_seed(sd, a)
        rows.append(r)
        print(f"[seed {sd}] 1p={r['one_premise']:+.4f} within={r['within_subcat']:+.4f} "
              f"diverse={r['diverse']:+.4f} | perm(div)={r['diverse_permuted']:+.4f} "
              f"lesion={r['diverse_lesion']:+.4f} | diag={r['diag_diverse']}")
    v, summ = verdict(rows, a)
    print(f"(elapsed {time.time()-t0:.1f}s, backend={'cupy' if os.environ.get('SIM_BACKEND')=='cupy' else 'numpy'})")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"verdict": v, "summary": summ, "seeds": rows, "args": vars(a)}, f, indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
