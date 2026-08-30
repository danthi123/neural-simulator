"""Consolidation dendritic LINE/BUMP continuous-attractor de-risk (the opsweep's own NEXT-(b), 2026-08-xx).

The confirmed boundary (`_consol_dendritic_opsweep.py`, 343-cell NO-GO): a comprehensive operating-point sweep over
N INDEPENDENT point-plateau `comp_attr_s` slots (each an isolated two-compartment bistable subunit, competing only
through ONE globally-symmetric WTA/FS inhibitory pool) found NO operating point that separates the consolidation
targets -- every top candidate has `dend_separated=0/3`, `dend_ratio~=1.00` (barely beats a linear readout). The
design doc's own prescription for this exact failure
(`2026-07-25-consolidation-dendritic-surpass-DESIGN-...md`, "Option-1 BUILT+TESTED" section, NEXT-(b)):

    "if no operating point separates -> the deeper dendritic LINE (bump) attractor -- a graded moving bump over
    the slots (Ecker/continuous-attractor style) rather than N independent point-plateaus, the months-scale surpass."

THE MECHANISM this runner builds (Ben-Yishai 1995 / Zhang 1996 ring-attractor; Ecker continuous-attractor-network
style; Wu-Zhang 2016 review): lay the N fact-slots out on a RING of `n_ring = N*(ring_spacing+1)` dendritic subunit
positions instead of N isolated islands. The N fact-slots sit at evenly-spaced ring positions -- named `comp_attr_i`,
IDENTICAL to the opsweep's naming, so `cdrive_probe`/`slot_ignition`/`_slot_idx` are reused BYTE-FOR-BYTE (import,
zero copy) -- and the intervening positions are structural FILLER dendritic subunits (`comp_ring_fill_p`) that carry
the ring's spatial continuity but receive no ca1 feedforward. Local coupling is Mexican-hat, the textbook
ring-attractor kernel:
  * ring-distance==1 (nearest neighbor): WEAK LATERAL EXCITATION (`comp_lateral_exc`) -- a locally-favored position
    is reinforced by its neighborhood, the "line" ingredient the opsweep's isolated point-plateaus structurally
    lacked (each slot's bistability there was PURELY self-loop, zero cross-slot cooperation).
  * ring-distance>=`comp_surround_start`: INHIBITION (`comp_surround_inhib`) via a per-position FS pool -- the SAME
    per-slot-FS mechanism already shipped in `nmda_compositional_consolidation.py` (`comp_per_slot_fs`), now
    DISTANCE-GATED (near positions are EXCLUDED from a position's own FS-pool inhibition targets) instead of
    "inhibit every other slot the same amount regardless of distance" (the opsweep's global-WTA topology, which
    treats the correct neighbor identically to the wrong slot two positions away and cannot support a graded bump).
The bet the opsweep's own design foregrounds: even a weak, noisy `ca1_i->slot_i` STDP advantage (measured too small
to separate an ISOLATED point-plateau, `dend_ratio~=1.00`) can be AMPLIFIED into one stable, spatially localized
bump by the ring's own recurrent line dynamics -- exactly how ring-attractor networks turn a noisy directional cue
into a sharp, stable head-direction/orientation peak (Zhang 1996; Ben-Yishai et al. 1995) -- instead of requiring the
feedforward drive to already be separated (what an isolated point-plateau + a single global WTA needs).

REUSE, NOT REINVENTION (no sim/ edit, no existing-runner edit): the AdEx/Izhikevich neuron model, the two-compartment
coincidence-plateau dendritic op (`enable_coincidence_detection`, `enable_two_compartment_dap`,
`coincidence_weighted_drive`, the bistable KIR down-state), the hippocampal encode/replay pipeline
(`encode_facts_with_reinstatement`, `coactivation_replay`), and the c_drive/ignition READOUTS (`cdrive_probe`,
`slot_ignition`) are ALL imported unmodified from `nmda_compositional_consolidation.py` /
`_consol_dendritic_opsweep.py`. The only NEW code is the ring topology itself (lateral-excitation +
distance-gated-surround-inhibition wiring in `build_substrate_lineattractor`), built entirely from the pre-existing
`BrainRegion` / `RegionPathway` primitives (zero sim/ edit).

VERDICT STRUCTURE matches the opsweep EXACTLY (`dend_selective`, `dend_separated`, `dend_ratio`, `lin_selective`,
`candidate`, plus `op`/`config_index`/`seed`) so results are directly comparable cell-for-cell. The LINEAR control
arm keeps the ring topology (lateral excitation + surround inhibition) FIXED and disables ONLY the dendritic
coincidence-plateau (`comp_dendritic=False`) -- the same anti-cheat discipline as the opsweep: isolates whether the
DENDRITIC plateau specifically is load-bearing, not just "any recurrent ring topology helps."

cfg.seed is set directly (NOT actual_seed_used) -- the repo's determinism rule (`bridge.py:2136`).

  python -m research.runners._consol_dendritic_lineattractor_derisk --list-configs
  python -m research.runners._consol_dendritic_lineattractor_derisk --config-index 7 --seed 42 \
      --out research/findings/raw/consol_lineattractor
  python -m research.runners._consol_dendritic_lineattractor_derisk --config-index 0 --seed 42 --cycles 1 \
      --dendritic-only --out /tmp/smoke   # BOUNDED smoke: 1 replay cycle, dendritic arm only

NOTE: `_phase1_recipe(True)` (tiny_synth) is NOT usable here -- the reused `encode_facts_with_reinstatement`
(`nmda_compositional_consolidation.py`) hardcodes `_phase1_recipe(False)` internally for its stimulus-pattern dims
(`n_lang_input=2048` etc.), so a substrate built at tiny_synth dims mismatches its own encoding call (a
`ValueError: shape mismatch` -- confirmed by trying it). The substrate here is therefore ALWAYS full `_phase1_recipe`
scale, exactly like the opsweep; bound a smoke run via `--cycles` + `--dendritic-only` instead of network size.
"""
from __future__ import annotations
import argparse, itertools, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SIM_BACKEND", "numpy")   # the pool is CPU/numpy; GPU box can override to cupy
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import research.runners.concept_pool_demo as cpd
from research.runners.unified_per_regime_monitor_runner import _phase1_recipe
from research.runners.text_minimal_isolation import build_biological_brain_regions
from research.runners.nmda_compositional_consolidation import (
    CONSOLIDATED_FACTS, encode_facts_with_reinstatement, coactivation_replay,
    _mean_gate_weight, _NOUN_POOLS, _ADJ_POOLS)
# byte-for-byte reuse of the opsweep's c_drive / ignition readouts (they key off region names
# `comp_attr_{s}` for s in range(N) -- this runner's fact-slot positions use the SAME names).
from research.runners._consol_dendritic_opsweep import cdrive_probe, slot_ignition
from sim.backend import get_backend

N = len(CONSOLIDATED_FACTS)

# ---- the operating-point grid (ring/bump-specific knobs on top of the opsweep's calibrated ones) -------------------
GRID = dict(
    self_regen=[0.0, 0.10, 0.20],        # v-gated SUSTAIN latch (opsweep's calibrated low-end; unchanged meaning)
    k_thresh=[2.0, 3.0, 4.0],            # coincidence threshold on the per-step weighted ca1->slot drive
    lateral_exc=[0.5, 1.5, 3.0],         # NEW: ring-distance==1 lateral excitation (the "line" ingredient)
    surround_inhib=[3.0, 6.0],           # NEW: ring-distance>=surround_start inhibition (the Mexican-hat surround)
    slot_drive=[700.0, 1400.0],          # co-activation slot drive (opsweep's low/high)
    ring_spacing=[1, 2],                 # NEW: filler positions between adjacent fact-slots (n_ring = N*(spacing+1))
)
_KEYS = list(GRID.keys())
_CONFIGS = [dict(zip(_KEYS, vals)) for vals in itertools.product(*[GRID[k] for k in _KEYS])]

# same BASE the opsweep uses for the non-ring parts of the substrate (ca1 wiring, nmda self-loops, replay drives);
# comp_no_pool_slot=True keeps the confirmed write-selectivity killer (concept-pool->ALL-slots broadcast) OFF.
BASE = dict(ca1_concept_density=0.25, ca1_concept_weight=0.0, nmda_self_weight=12.0, nmda_self_density=0.15,
            nmda_recurrent_ratio=0.6, cross_pool_density=0.10, stdp_w_max=8.0, enable_global_nmda=False,
            enable_hebbian=True, skip_nmda_additions=True, comp_self_weight=12.0, comp_no_pool_slot=True,
            comp_attractor_n_per=120, comp_fill_n_per=40, comp_kir_g=3.0,
            comp_lateral_density=0.3, comp_surround_start=2, comp_surround_density=0.5, comp_slot_to_fs=20.0)
REPLAY_CYCLES = 40   # matches the opsweep default; --cycles overrides for a bounded smoke


# ---------------------------------------------------------------------------
# Substrate: the ring/bump continuous attractor (reuses build_biological_brain_regions + the ca1/nmda_slow wiring
# verbatim from nmda_compositional_consolidation.build_substrate; only section (4), the comp-attractor slots, is
# NEW -- a ring instead of N isolated islands + one global WTA).
# ---------------------------------------------------------------------------
def build_substrate_lineattractor(seed, args):
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.regions import RegionPathway, BrainRegion

    dims = _phase1_recipe(False)   # full scale -- MUST match encode_facts_with_reinstatement's hardcoded dims (see module docstring NOTE)
    n_lang_input = int(dims["n_lang_input"])
    n_per_pool = int(dims["n_per_pool"])
    n_fs_per_pool = int(dims["n_fs_per_pool"])
    n_dlpfc_verb = int(dims["n_dlpfc_verb"])

    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=n_per_pool,
        motor_internal_density=0.10,
        motor_exc_weight_mean=2.0,
        motor_inh_weight_mean=4.0,
        text_input_to_motor_density=0.30,
        text_input_to_motor_weight=3.0,
        text_input_to_motor_jitter=0.5,
        enable_motor_fs=True,
        n_motor_fs_per_action=n_fs_per_pool,
        enable_language_output=True,
        n_lang_output=n_lang_input,
        motor_to_language_output_weight=2.0,
        enable_noun_pools=True,
        noun_pool_names=cpd.NOUN_NAMES,
        n_noun_per_pool=n_per_pool,
        n_noun_fs_per_pool=n_fs_per_pool,
        enable_verb_pools=True,
        verb_pool_names=cpd.VERB_NAMES,
        n_verb_per_pool=n_per_pool,
        n_verb_fs_per_pool=n_fs_per_pool,
        enable_adjective_pools=True,
        adjective_pool_names=cpd.ADJECTIVE_NAMES,
        n_adjective_per_pool=n_per_pool,
        n_adjective_fs_per_pool=n_fs_per_pool,
        # WEAK concept dynamics (v14/v16 Phase-1 stability) -- unchanged from build_substrate.
        concept_pool_internal_density=0.05,
        concept_pool_exc_weight_mean=0.3,
        concept_pool_inh_weight_mean=0.8,
        enable_cross_pool_concept_pathways=True,
        cross_pool_concept_density=float(args.cross_pool_density),
        enable_hippocampus_consolidation=True,
        enable_dlpfc_verb=True,
        n_dlpfc_verb=n_dlpfc_verb,
        dlpfc_verb_internal_density=0.15,
    )
    pathways = list(pathways)
    regions = list(regions)
    concept_pools = _NOUN_POOLS + ["verb_pool_%s" % v for v in cpd.VERB_NAMES] + _ADJ_POOLS
    skip_nmda = bool(getattr(args, "skip_nmda_additions", False))

    # (1) ca1 -> concept-pool consolidation wire -- UNCHANGED from build_substrate.
    n_ca1_wire = 0
    for pool in concept_pools:
        pathways.append(RegionPathway(
            from_region="ca1", to_region=pool,
            density=float(args.ca1_concept_density),
            weight_mean=float(args.ca1_concept_weight), weight_jitter=0.3,
            plastic=True, plasticity_gate="ca1_to_concept_pool",
        ))
        n_ca1_wire += 1

    # (3) nmda_slow self-loop attractor on noun+adjective word-concept pools -- UNCHANGED from build_substrate.
    nmda_pools = _NOUN_POOLS + _ADJ_POOLS
    n_self = 0
    if not skip_nmda:
        for pool in nmda_pools:
            pathways.append(RegionPathway(
                from_region=pool, to_region=pool,
                density=float(args.nmda_self_density),
                weight_mean=float(args.nmda_self_weight), weight_jitter=0.05,
                plastic=False, exc_receptor="nmda_slow",
                transmission_gate="nmda_attractor",
            ))
            n_self += 1

    # (4) THE LINE/BUMP RING (this runner's new ingredient -- replaces build_substrate's isolated
    #     comp_attr_s + one global/per-slot-uniform WTA with a ring of n_ring = N*(ring_spacing+1) dendritic
    #     subunit positions, Mexican-hat coupled).
    comp_dend = bool(getattr(args, "comp_dendritic", False))
    ring_spacing = max(0, int(getattr(args, "ring_spacing", 1)))
    n_ring = N * (ring_spacing + 1)
    n_per = int(getattr(args, "comp_attractor_n_per", 120))
    n_fill = int(getattr(args, "comp_fill_n_per", max(20, n_per // 3)))
    lateral_exc = float(getattr(args, "comp_lateral_exc", 1.0))
    lateral_density = float(getattr(args, "comp_lateral_density", 0.3))
    surround_inhib = float(getattr(args, "comp_surround_inhib", 5.0))
    surround_start = max(1, int(getattr(args, "comp_surround_start", 2)))
    surround_density = float(getattr(args, "comp_surround_density", 0.5))
    slot_to_fs = float(getattr(args, "comp_slot_to_fs", 20.0))

    def _fact_pos(i):
        return i * (ring_spacing + 1)
    fact_of_pos = {_fact_pos(i): i for i in range(N)}

    def _pos_name(p):
        return f"comp_attr_{fact_of_pos[p]}" if p in fact_of_pos else f"comp_ring_fill_{p}"

    def _ring_dist(a, b):
        d = abs(a - b) % n_ring
        return min(d, n_ring - d)

    n_comp = 0
    if n_ring > 0:
        pos_names = [_pos_name(p) for p in range(n_ring)]
        # per-position excitatory dendritic subunit + self nmda_slow bistable loop (SAME mechanism as the
        # opsweep's isolated comp_attr_s -- the departure is what connects them, not the subunit itself).
        for p in range(n_ring):
            nm = pos_names[p]
            npos = n_per if p in fact_of_pos else n_fill
            regions = list(regions) + [BrainRegion(
                name=nm, n_neurons=npos, exc_fraction=1.0, internal_density=0.20,
                exc_weight_mean=2.0, inh_weight_mean=0.0, weight_jitter=0.3, plastic_internal=False)]
            pathways.append(RegionPathway(
                from_region=nm, to_region=nm, density=0.20,
                weight_mean=float(getattr(args, "comp_self_weight", 12.0)), weight_jitter=0.05,
                plastic=False, exc_receptor="nmda_slow", transmission_gate="nmda_attractor"))
            if p in fact_of_pos:
                # ca1 -> this fact's ring position (plastic; potentiates during co-activation replay). With
                # comp_dendritic: route through the two-compartment WEIGHTED-coincidence plateau.
                pathways.append(RegionPathway(
                    from_region="ca1", to_region=nm,
                    density=float(args.ca1_concept_density), weight_mean=float(args.ca1_concept_weight),
                    weight_jitter=0.3, plastic=True, plasticity_gate="ca1_to_comp_attr",
                    coincidence_detector=comp_dend))
            n_comp += 1
        # ring-distance==1: LATERAL EXCITATION -- the "line" continuity the opsweep's isolated slots lacked.
        for p in range(n_ring):
            for q in range(n_ring):
                if p != q and _ring_dist(p, q) == 1:
                    pathways.append(RegionPathway(
                        from_region=pos_names[p], to_region=pos_names[q],
                        density=lateral_density, weight_mean=lateral_exc, weight_jitter=0.2, plastic=False))
        # per-position FS pool: fed by its own position, inhibits ONLY ring-distance>=surround_start positions
        # (the Mexican-hat surround; DISTANCE-GATED, unlike the opsweep's "inhibit every other slot equally").
        for p in range(n_ring):
            far = [q for q in range(n_ring) if _ring_dist(p, q) >= surround_start]
            if not far:
                continue
            fs_nm = f"comp_ring_fs_{p}"
            npos = n_per if p in fact_of_pos else n_fill
            regions = list(regions) + [BrainRegion(
                name=fs_nm, n_neurons=int(max(10, npos * 0.5)), exc_fraction=0.0, internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.1, plastic_internal=False)]
            pathways.append(RegionPathway(from_region=pos_names[p], to_region=fs_nm,
                                          density=1.0, weight_mean=slot_to_fs, weight_jitter=0.2, plastic=False))
            for q in far:
                pathways.append(RegionPathway(from_region=fs_nm, to_region=pos_names[q],
                                              density=surround_density, weight_mean=surround_inhib,
                                              weight_jitter=0.2, plastic=False))
        # concept pools -> the FACT positions only (comp_no_pool_slot=True in BASE drops this -- the confirmed
        # write-selectivity-killer broadcast; filler ring positions never receive concept-pool input).
        if not bool(getattr(args, "comp_no_pool_slot", False)):
            for pool in (_NOUN_POOLS + _ADJ_POOLS):
                for i in range(N):
                    pathways.append(RegionPathway(
                        from_region=pool, to_region=f"comp_attr_{i}", density=0.15,
                        weight_mean=float(getattr(args, "comp_pool_slot_weight", 1.5)), weight_jitter=0.3,
                        plastic=True, plasticity_gate="concept_to_comp_attr"))

    print(f"  augment(ring): +{n_ca1_wire} ca1->concept wires, +{n_self} nmda_slow self-loops, "
          f"+{n_comp} ring positions (N={N} facts, spacing={ring_spacing}, n_ring={n_ring}, "
          f"n_per={n_per}, n_fill={n_fill}) lateral_exc={lateral_exc} "
          f"surround_inhib={surround_inhib}(start>={surround_start})", flush=True)

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = int(seed)                      # <-- SEEDS THE SUBSTRATE (not actual_seed_used)
    cfg.enable_nmda = bool(args.enable_global_nmda)
    cfg.nmda_tau_decay = 100.0
    cfg.enable_nmda_recurrent = (not skip_nmda)
    cfg.nmda_recurrent_tau_decay_ms = 100.0
    cfg.nmda_recurrent_ratio = float(args.nmda_recurrent_ratio)
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = bool(args.enable_hebbian)
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = float(args.stdp_w_max)
    cfg.fast_spike_reset = True

    if comp_dend:   # dendritic plateau (gap5 GO_CFG + r-iii operating point) -- identical to build_substrate.
        cfg.enable_coincidence_detection = True
        cfg.coincidence_weighted_drive = True
        cfg.coincidence_k_threshold = float(getattr(args, "comp_k_thresh", 3.0))
        cfg.enable_two_compartment_dap = True
        cfg.coincidence_plateau_self_regen = float(getattr(args, "comp_self_regen", 0.15))
        cfg.coincidence_plateau_v_hold = float(getattr(args, "comp_v_hold", -50.0))
        cfg.apical_kir_g = float(getattr(args, "comp_kir_g", 3.0))
        cfg.apical_g_couple = float(getattr(args, "comp_gc", 1.0))
        cfg.apical_g_couple_to_soma = float(getattr(args, "comp_gc_read", 5.0))
        cfg.apical_R = float(getattr(args, "comp_apical_R", 50.0))

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def run_arm(seed, op, dendritic):
    a = dict(BASE)
    a.update(comp_dendritic=bool(dendritic), comp_k_thresh=op["k_thresh"], comp_self_regen=op["self_regen"],
              comp_lateral_exc=op["lateral_exc"], comp_surround_inhib=op["surround_inhib"],
              ring_spacing=op["ring_spacing"])
    b = build_substrate_lineattractor(seed, SimpleNamespace(**a))
    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS)
    w0 = _mean_gate_weight(b, "ca1_to_comp_attr")
    coactivation_replay(b, CONSOLIDATED_FACTS, tags, REPLAY_CYCLES, seed,
                        coactivate=True, attractor_on=True, slot_drive_pA=op["slot_drive"])
    w1 = _mean_gate_weight(b, "ca1_to_comp_attr")
    cd = cdrive_probe(b, tags)
    ig = slot_ignition(b, tags)
    return dict(arm="dendritic" if dendritic else "linear", w_ca1slot_pre=round(w0, 5), w_ca1slot_post=round(w1, 5),
                dw=round(w1 - w0, 5), cdrive=cd, ignition=ig)


def main():
    global REPLAY_CYCLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-index", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="research/findings/raw/consol_lineattractor")
    ap.add_argument("--list-configs", action="store_true")
    ap.add_argument("--dendritic-only", action="store_true", help="skip the LINEAR control arm (half the runtime)")
    ap.add_argument("--cycles", type=int, default=None, help="replay cycles (default 40; lower for smoke)")
    args = ap.parse_args()
    if args.cycles is not None:
        REPLAY_CYCLES = int(args.cycles)
    if args.list_configs:
        print(len(_CONFIGS)); return
    ci = args.config_index
    if ci is None or not (0 <= ci < len(_CONFIGS)):
        print(f"ERR --config-index must be 0..{len(_CONFIGS)-1}"); sys.exit(2)
    op = _CONFIGS[ci]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    outp = Path(args.out) / f"la{ci:03d}_seed{args.seed}.json"
    t0 = time.time()
    rec = dict(config_index=ci, seed=args.seed, op=op, backend=get_backend()[1],
               replay_cycles=REPLAY_CYCLES, n_facts=N,
               n_ring=N * (int(op["ring_spacing"]) + 1))

    def _verdict():
        d = rec.get("dendritic")
        if not d:
            return {}
        return dict(
            dend_selective=d["ignition"]["selective"], dend_ratio=d["cdrive"]["mean_ratio"],
            dend_separated=d["cdrive"]["n_separated"],
            lin_selective=(rec.get("linear") or {}).get("ignition", {}).get("selective"),
            # SAME GO-candidate definition as the opsweep: dendritic separates AND is selective >= ceil(N/2)
            # AND beats the linear (ring-topology-only, no coincidence-plateau) control.
            candidate=bool(d["cdrive"]["n_separated"] >= (N + 1) // 2
                           and d["ignition"]["selective"] >= (N + 1) // 2
                           and (rec.get("linear") is None
                                or d["ignition"]["selective"] > rec["linear"]["ignition"]["selective"])))

    def _flush():
        rec["VERDICT"] = _verdict()
        rec["elapsed_s"] = round(time.time() - t0, 1)
        outp.write_text(json.dumps(rec, indent=2))   # INCREMENTAL: preserve the dendritic arm if the linear arm times out

    try:
        rec["dendritic"] = run_arm(args.seed, op, dendritic=True)
        _flush()                                                      # <- dendritic result persisted before the slow linear arm
        if not args.dendritic_only:
            rec["linear"] = run_arm(args.seed, op, dendritic=False)   # coincidence OFF, SAME ring -> load-bearing check
            _flush()
    except Exception as e:
        import traceback
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-2000:]
        _flush()
    v = rec.get("VERDICT", {})
    print(f"[la{ci:03d} seed{args.seed}] {op} -> dend_sel={v.get('dend_selective')} ratio={v.get('dend_ratio')} "
          f"sep={v.get('dend_separated')}/{N} lin_sel={v.get('lin_selective')} CANDIDATE={v.get('candidate')} "
          f"({rec['elapsed_s']}s){' ERR='+rec['error'] if 'error' in rec else ''}", flush=True)


if __name__ == "__main__":
    main()
