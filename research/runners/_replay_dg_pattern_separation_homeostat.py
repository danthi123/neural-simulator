"""Stabilize the DG k-WTA with an intrinsic-excitability HOMEOSTAT (board #73).

Attacks the RESIDUAL localized by board #43 / the #71 bridge finding
(`2026-08-19-replay-separator-bridge-rebound-and-write-runaway-FIXED-single-recall-ceiling-kWTA-stability-residual.md`):
the two rebound + Hebbian-write-runaway blockers are already FIXED in the imported
runner ``_replay_dg_pattern_separation_bridge`` (shunting inhibition + a
transmission-gated write). What remains is k-WTA STABILITY -- on the Izhikevich
substrate one memory's granule engram collapses to near-dense (150-200 of 200
cells), that dense engram SUBSUMES the other memory's sparse engram, and the dense
memory's answer then wins BOTH probes (the anti-symmetric signature; both_win 0/6).

THE NAMED, UNTRIED MECHANISM (this file): an intrinsic-excitability HOMEOSTAT on the
granule cells -- each granule adapts its OWN firing threshold toward a target
activity set-point (a per-cell firing-rate homeostat / adaptive threshold). A cell
that fires too often raises its threshold, so the active population stays bounded at
~k regardless of input, and the runaway that lets one engram capture all cells is
prevented. Biology: intrinsic plasticity / firing-rate homeostasis (Turrigiano 2011
Annu Rev Neurosci 34:89-103; Desai 2003 J Physiol Paris 97:391-402).

IMPLEMENTATION -- NO sim/ edit. The production substrate ALREADY carries this
mechanism: ``BrainRegion.enable_homeostasis=True`` scopes the engine's per-neuron
adaptive-threshold homeostat (``cp_neuron_firing_thresholds`` driven by
``fused_homeostasis_update``: ema<-(1-a)ema+a*fired; err=ema-target;
thr<-thr+err*adapt) to ONE region via ``cp_homeostasis_neuron_mask``, while the
GLOBAL ``cfg.enable_homeostasis`` stays False (so every OTHER region keeps its normal
vpeak spike threshold, byte-identical). Precedent: the per-region-homeostasis nav
finding (`2026-06-09-navfaithful-afferent-critic-homeostasis-PASS.md`). We enable it
ONLY on ``dg`` and set the homeostat operating point (target rate, EMA/adapt speed,
threshold band around vt=-40) so it acts WITHIN the replay protocol -- fast intrinsic
plasticity / SFA-timescale, tens of ms (the regime the CA1-sparsification gate named).

This is the exact "next mechanism 1" the #71 finding named:
  "A slow per-granule adaptive threshold (intrinsic excitability homeostasis;
   Turrigiano) that drives every granule toward a target firing fraction, run
   alongside the fast basket inhibition, would cap the dense-collapse..."

We reuse the #71 runner by import (its shunting reversal + transmission-gated write +
all measurement/probe/scramble/direct-readout machinery) and only (a) add a
homeostat-capable ``build_bridge`` and (b) run BOTH the homeostat-ON arm and a
homeostat-LESION arm (dg_homeostat=False -> reproduces the #71 residual) so the
dissociation proves the homeostat does the work.

ANTI-CHEATS (design first, from the #71 NO-GO bar):
  1. Two SIMILAR memories BOTH discriminable -- both_win across 6 seeds (the bar #71
     missed). Per-memory selectivity reported for BOTH memories.
  2. The homeostat holds sparsity: engram size bounded near k for BOTH memories (no
     dense-collapse); sizes reported per memory per seed. LESION -> dense-collapse +
     the anti-symmetric failure RETURN (reproduces #71). Dissociation.
  3. No regression: single-memory recall stays +1.00; DISSIMILAR pairs both-win;
     scramble-teach still inverts.
  4. 6 seeds (42/43/44/100/101/102), per-seed + pooled, deterministic (cfg.seed).

Run:
    OMP_NUM_THREADS=2 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_homeostat \
        --seeds 42 43 44 100 101 102 \
        --out research/findings/raw/kwta_homeostat/homeostat_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import research.runners._replay_dg_pattern_separation_bridge as base  # noqa: E402
from research.runners._replay_dg_pattern_separation_bridge import (  # noqa: E402
    BridgeConfig,
    SEEDS,
    DG_WRITE_GATE,
    DG_COMPETITION_GATE,
    ANSWER_INHIBITION_GATE,
    DG_ANSWER_TX_GATE,
)
from research.runners._replay_dg_pattern_separation_gate import (  # noqa: E402
    _perforant_edges,
)
from research.runners._replay_cortical_consolidation_gate import (  # noqa: E402
    _all_to_all,
)
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402


@dataclass(frozen=True)
class HConfig(BridgeConfig):
    """#71 BridgeConfig + the intrinsic-excitability homeostat knobs (additive)."""
    # Master switch. False -> build is byte-identical to the #71 runner (the LESION).
    dg_homeostat: bool = True
    # Homeostat operating point (applied to the dg region only, via the per-region
    # mask; global cfg.enable_homeostasis stays False).
    # target_rate = per-step EMA firing-probability set-point at the biological DG
    # sparsity (~2-5% active).
    homeostat_target_rate: float = 0.05
    # FAST intrinsic plasticity so it bites within a replay event (tens of ms /
    # tens of steps), not the default 5000-step homeostatic timescale.
    homeostat_ema_alpha: float = 0.20        # tau ~5 steps
    homeostat_adapt_rate: float = 1.5        # mV per unit-error per step
    # Threshold band ENTIRELY BELOW vt=-40 (granule RS: vr=-60, vt=-40, vpeak=+35):
    # the adapted threshold REPLACES vpeak, and sub-vt detection fires the cell
    # BEFORE the quadratic upstroke engages -> the adaptive threshold has authority
    # over the active set (a LIF-like regime; a threshold at/above vt loses authority
    # because the quadratic runs v away past any finite threshold <= vpeak). This is
    # the strongest faithful attempt at the intrinsic-adaptive-threshold mechanism.
    homeostat_thresh_min: float = -56.0
    homeostat_thresh_max: float = -44.0


def build_bridge_homeostat(seed: int, cfg: HConfig):
    """Copy of ``base.build_bridge`` (keeps the shunting reversal + transmission-gated
    write) plus the per-region intrinsic-excitability homeostat on ``dg``.

    When ``cfg.dg_homeostat`` is False this is byte-identical to ``base.build_bridge``
    (no region opts in -> no homeostasis mask -> vpeak spike threshold everywhere ->
    the #71 residual). That False path IS the lesion control."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
    from sim.enums import NeuronModel, NeuronType
    from sim.regions import BrainRegion, RegionPathway

    exc = dict(exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
               weight_jitter=0.0, plastic_internal=False,
               izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)
    inh = dict(exc_fraction=0.0, internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
               weight_jitter=0.0, plastic_internal=False,
               izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name)

    dg_kwargs = dict(exc, syn_reversal_potential_i_override=float(cfg.dg_inh_reversal_mV))
    # THE HOMEOSTAT: scope the engine's per-neuron adaptive-threshold intrinsic
    # plasticity to the dg region only (additive; existing public BrainRegion field).
    if cfg.dg_homeostat:
        dg_kwargs["enable_homeostasis"] = True
    ans_kwargs = dict(exc, syn_reversal_potential_i_override=float(cfg.answer_inh_reversal_mV))

    core = CoreSimConfig()
    core.enable_brain_region_framework = True
    core.brain_regions = [
        BrainRegion("input", cfg.n_input, **exc),
        BrainRegion("dg", cfg.n_dg, **dg_kwargs),
        BrainRegion("dg_fs", cfg.n_dg_fs, **inh),
        BrainRegion("answer", cfg.n_answer, **ans_kwargs),
        BrainRegion("answer_fs", cfg.n_answer_fs, **inh),
    ]
    core.region_pathways = [RegionPathway("input", "dg", density=0.01, weight_mean=0.01, plastic=False)]
    core.num_neurons = 0
    core.connections_per_neuron = 0
    core.neuron_model_type = NeuronModel.IZHIKEVICH.name
    core.neural_profile_name = "GENERIC_UNSTRUCTURED"
    core.dt_ms = 1.0
    core.seed = core.heterogeneity_seed = core.ou_seed = int(seed)
    core.enable_stdp = False
    core.enable_hebbian_learning = True
    core.hebbian_rate_window = True
    core.hebbian_learning_rate = float(cfg.wake_learning_rate)
    core.hebbian_max_weight = float(cfg.hebbian_max_weight)
    core.hebbian_min_weight = 0.0
    core.hebbian_weight_decay = 0.0
    core.hebbian_coactivity_decay = float(cfg.hebbian_coactivity_decay)
    core.hebbian_coactivity_thresh = float(cfg.hebbian_coactivity_thresh)
    core.enable_reward_modulation = False
    # GLOBAL homeostasis stays OFF -> only the dg per-region mask uses adapted
    # thresholds; every other region keeps its normal vpeak spike threshold.
    core.enable_homeostasis = False
    # The homeostat operating point (used by the dg-masked neurons only).
    if cfg.dg_homeostat:
        core.homeostasis_target_rate = float(cfg.homeostat_target_rate)
        core.homeostasis_ema_alpha = float(cfg.homeostat_ema_alpha)
        core.homeostasis_threshold_adapt_rate = float(cfg.homeostat_adapt_rate)
        core.homeostasis_threshold_min = float(cfg.homeostat_thresh_min)
        core.homeostasis_threshold_max = float(cfg.homeostat_thresh_max)
    core.enable_short_term_plasticity = False
    core.enable_structural_plasticity = False
    core.enable_ou_process = False
    core.ou_std_current_pA = 0.0
    core.fast_spike_reset = True
    core.propagation_strength = float(cfg.propagation_strength)
    core.max_synaptic_delay_ms = float(cfg.max_synaptic_delay_ms)

    runtime = RuntimeState()
    runtime.actual_seed_used = int(seed)
    bridge = SimulationBridge(core_config=core, viz_config=VisualizationConfig(),
                              runtime_state=runtime, gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(core.max_synaptic_delay_ms / core.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    regions = {name: np.asarray(bridge.region_manager.indices(name), dtype=np.int64)
               for name in ("input", "dg", "dg_fs", "answer", "answer_fs")}

    pp_pre, pp_post = _perforant_edges(seed, cfg, regions["input"], regions["dg"])
    i2fs_pre, i2fs_post = _all_to_all(regions["input"], regions["dg_fs"])
    d2fs_pre, d2fs_post = _all_to_all(regions["dg"], regions["dg_fs"])
    fs2d_pre, fs2d_post = _all_to_all(regions["dg_fs"], regions["dg"])
    a2fs_pre, a2fs_post = _all_to_all(regions["answer"], regions["answer_fs"])
    fs2a_pre, fs2a_post = _all_to_all(regions["answer_fs"], regions["answer"])
    dga_pre, dga_post = _all_to_all(regions["dg"], regions["answer"])

    def group(pre, post, weight, *, plastic, conn_type, plasticity_gate=None, transmission_gate=None):
        row = {"pre_indices": pre.tolist(), "post_indices": post.tolist(),
               "initial_weights": np.full(pre.size, weight, dtype=np.float32),
               "plastic": bool(plastic), "conn_type": conn_type, "count": int(pre.size)}
        if plasticity_gate:
            row["plasticity_gate"] = plasticity_gate
        if transmission_gate:
            row["transmission_gate"] = transmission_gate
        return row

    wiring = {
        "input_to_dg": group(pp_pre, pp_post, cfg.input_to_dg_weight, plastic=False, conn_type="E_TO_E"),
        "input_to_fs": group(i2fs_pre, i2fs_post, cfg.input_to_fs_weight, plastic=False, conn_type="E_TO_I"),
        "dg_to_fs": group(d2fs_pre, d2fs_post, cfg.dg_to_fs_weight, plastic=False, conn_type="E_TO_I"),
        "fs_to_dg": group(fs2d_pre, fs2d_post, cfg.fs_to_dg_weight, plastic=False,
                          conn_type="I_TO_E", transmission_gate=DG_COMPETITION_GATE),
        "answer_to_fs": group(a2fs_pre, a2fs_post, cfg.answer_to_fs_weight, plastic=False, conn_type="E_TO_I"),
        "fs_to_answer": group(fs2a_pre, fs2a_post, cfg.fs_to_answer_weight, plastic=False,
                              conn_type="I_TO_E", transmission_gate=ANSWER_INHIBITION_GATE),
        "dg_to_answer": group(dga_pre, dga_post, cfg.dg_answer_init_weight, plastic=True,
                              conn_type="E_TO_E", plasticity_gate=DG_WRITE_GATE,
                              transmission_gate=DG_ANSWER_TX_GATE),
    }
    inh_idx = np.concatenate([regions["dg_fs"], regions["answer_fs"]]).tolist()
    bridge.inject_explicit_wiring(wiring, output_inhibitory_indices=inh_idx)
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0)
    bridge.set_transmission_gate(ANSWER_INHIBITION_GATE, 1.0)
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 1.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)

    handles = {"regions": regions, "bridge_identity": id(bridge),
               "homeostat": bool(cfg.dg_homeostat),
               "homeostasis_mask_neurons": (int(bridge.cp_homeostasis_neuron_mask.sum())
                                            if bridge.cp_homeostasis_neuron_mask is not None else 0),
               "wiring_counts": {k: v["count"] for k, v in wiring.items()}}
    return bridge, handles


def _run_arm(seeds, cfg):
    """Run the full #71 pipeline (via base.run) with build_bridge swapped to the
    homeostat build. cfg.dg_homeostat selects ON vs LESION."""
    base.build_bridge = build_bridge_homeostat
    try:
        return base.run(seeds, cfg)
    finally:
        base.build_bridge = _ORIG_BUILD


_ORIG_BUILD = base.build_bridge


def _seed_row(on_seed, off_seed):
    """Assemble the dissociation row for one seed from the ON and LESION payloads."""
    on = on_seed["conditions"]["similar_separator_on"]
    off = off_seed["conditions"]["similar_separator_on"]
    return {
        "seed": on_seed["seed"],
        # anti-cheat 1: both similar memories discriminable
        "both_win_on": on["both_win"],
        "both_win_lesion": off["both_win"],
        "per_memory_selectivity_on": on["per_memory_selectivity"],
        "per_memory_selectivity_lesion": off["per_memory_selectivity"],
        "mean_selectivity_on": on["mean_selectivity"],
        "mean_selectivity_lesion": off["mean_selectivity"],
        # anti-cheat 2: engram sizes bounded (ON) vs dense-collapse (LESION)
        "dg_sizes_on": (on["dg_separation"]["dg_size_m0"], on["dg_separation"]["dg_size_m1"]),
        "dg_sizes_lesion": (off["dg_separation"]["dg_size_m0"], off["dg_separation"]["dg_size_m1"]),
        "dense_collapse_on": on["dense_engram_collapse"],
        "dense_collapse_lesion": off["dense_engram_collapse"],
        "dg_jaccard_on": on["dg_separation"]["dg_jaccard"],
        "dg_jaccard_lesion": off["dg_separation"]["dg_jaccard"],
        # anti-cheat 3: no regression (from the ON arm)
        "single_selectivity_on": on_seed["summary"]["single_selectivity"],
        "single_scramble_on": on_seed["summary"]["single_scramble_selectivity"],
        "dissimilar_both_win_on": on_seed["summary"]["dissimilar_both_win"],
        # homeostat provenance
        "homeostasis_mask_neurons_on": on_seed.get("homeostasis_mask_neurons"),
    }


def run(seeds, cfg_on):
    started = time.time()
    cfg_lesion = HConfig(**{**cfg_on.__dict__, "dg_homeostat": False})

    on_payload = _run_arm(seeds, cfg_on)
    lesion_payload = _run_arm(seeds, cfg_lesion)

    by_seed_on = {r["seed"]: r for r in on_payload["per_seed"]}
    by_seed_off = {r["seed"]: r for r in lesion_payload["per_seed"]}
    rows = [_seed_row(by_seed_on[int(s)], by_seed_off[int(s)]) for s in seeds]
    n = len(rows)

    def cnt(fn):
        return int(sum(1 for r in rows if fn(r)))

    # ---- The board #73 bar + anti-cheats ----
    both_win_on = cnt(lambda r: r["both_win_on"])
    both_win_lesion = cnt(lambda r: r["both_win_lesion"])
    dense_on = cnt(lambda r: r["dense_collapse_on"])
    dense_lesion = cnt(lambda r: r["dense_collapse_lesion"])
    single_ok = cnt(lambda r: r["single_selectivity_on"] >= 0.30)
    scramble_ok = cnt(lambda r: r["single_scramble_on"] <= r["single_selectivity_on"] - 0.30)
    dissim_ok = cnt(lambda r: r["dissimilar_both_win_on"])

    checks = {
        # AC1: both similar memories discriminable, all 6 seeds (the #71-missed bar).
        "both_similar_discriminable_6of6": both_win_on == n,
        # AC2: homeostat holds sparsity -- no dense-collapse under ON.
        "no_dense_collapse_on": dense_on == 0,
        # AC2 dissociation: LESION reproduces the #71 failure (dense-collapse returns
        # AND both_win fails).
        "lesion_reproduces_residual": (dense_lesion >= 1) and (both_win_lesion < n),
        # AC3: no regression.
        "single_recall_ceiling": single_ok == n,
        "scramble_inverts": scramble_ok == n,
        "dissimilar_both_win": dissim_ok == n,
    }
    go = (checks["both_similar_discriminable_6of6"]
          and checks["no_dense_collapse_on"]
          and checks["lesion_reproduces_residual"]
          and checks["single_recall_ceiling"]
          and checks["scramble_inverts"]
          and checks["dissimilar_both_win"])

    # ATTRIBUTION (tools.lab): whose is the dense-collapse? The homeostat arm's
    # dense-collapse count vs the lesion's -- how much of the ON dense-collapse is
    # NOT already present in the lesion (i.e. added BY the homeostat). A negative
    # fraction would mean the homeostat REDUCED dense-collapse (the hoped-for GO).
    dense_attrib = attributable_to("dense-collapse ON vs LESION", dense_on, dense_lesion)

    # VERDICT (tools.verdict): earn the verdict; the preconditions travel with the
    # artifact so the gate can enforce that what earned it is present.
    v = Verdict("DG k-WTA stability via intrinsic-excitability homeostat")
    v.require("pipeline intact: single-memory recall at ceiling (6/6)", single_ok == n, expect=True)
    v.require("read rides the learned mapping: scramble inverts (6/6)", scramble_ok == n, expect=True)
    v.require("baseline is the #71 residual: lesion dense-collapses", dense_lesion >= 1, expect=True)
    v.require("baseline is the #71 residual: lesion both_win fails", both_win_lesion == 0, expect=True)
    v.control("homeostat changes the DG code (manipulation landed)",
              treatment=dense_on, control=dense_lesion, min_separation=0.0)
    v.disabled("STDP / reward / short-term & structural plasticity / OU noise",
               why="isolation inherited from the #71 runner: the separator + Hebbian write are the only live plasticity")
    decided = v.decide(go=go)

    return {
        "gate": "replay_dg_pattern_separation_homeostat",
        "mechanism": "per-region intrinsic-excitability homeostat on dg (adaptive threshold, firing-rate set-point)",
        "seeds": [int(s) for s in seeds],
        "n_seeds": n,
        "status": decided["status"],
        "verdict": decided,
        "preconditions": decided["preconditions"],
        "dense_collapse_attributable_to_homeostat": dense_attrib,
        "checks": checks,
        "homeostat_config": {
            "target_rate": cfg_on.homeostat_target_rate,
            "ema_alpha": cfg_on.homeostat_ema_alpha,
            "adapt_rate": cfg_on.homeostat_adapt_rate,
            "thresh_min": cfg_on.homeostat_thresh_min,
            "thresh_max": cfg_on.homeostat_thresh_max,
        },
        "pooled": {
            "both_win_on_count": both_win_on,
            "both_win_lesion_count": both_win_lesion,
            "dense_collapse_on_count": dense_on,
            "dense_collapse_lesion_count": dense_lesion,
            "single_recall_ok_count": single_ok,
            "scramble_inverts_count": scramble_ok,
            "dissimilar_both_win_count": dissim_ok,
            "mean_selectivity_on": float(np.mean([r["mean_selectivity_on"] for r in rows])),
            "mean_selectivity_lesion": float(np.mean([r["mean_selectivity_lesion"] for r in rows])),
        },
        "per_seed": rows,
        "on_arm_full": on_payload,
        "lesion_arm_full": lesion_payload,
        "scaffolds": on_payload.get("scaffolds", []),
        "elapsed_seconds": time.time() - started,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--target-rate", type=float, default=None)
    ap.add_argument("--ema-alpha", type=float, default=None)
    ap.add_argument("--adapt-rate", type=float, default=None)
    ap.add_argument("--thresh-min", type=float, default=None)
    ap.add_argument("--thresh-max", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    overrides = {}
    if args.smoke:
        sm = base.smoke_config()
        overrides.update(sm.__dict__)
    for name, val in (("homeostat_target_rate", args.target_rate),
                      ("homeostat_ema_alpha", args.ema_alpha),
                      ("homeostat_adapt_rate", args.adapt_rate),
                      ("homeostat_thresh_min", args.thresh_min),
                      ("homeostat_thresh_max", args.thresh_max)):
        if val is not None:
            overrides[name] = val
    cfg = HConfig(**overrides)

    print(f"[kwta-homeostat] backend={os.environ.get('SIM_BACKEND','default')} "
          f"seeds={args.seeds} smoke={args.smoke} "
          f"target={cfg.homeostat_target_rate} alpha={cfg.homeostat_ema_alpha} "
          f"adapt={cfg.homeostat_adapt_rate} band=[{cfg.homeostat_thresh_min},{cfg.homeostat_thresh_max}]",
          flush=True)
    payload = run(args.seeds, cfg)
    for r in payload["per_seed"]:
        pm_on = r["per_memory_selectivity_on"]
        pm_le = r["per_memory_selectivity_lesion"]
        print(f"  seed {r['seed']}: "
              f"ON both_win={r['both_win_on']} sizes={r['dg_sizes_on']} dense={r['dense_collapse_on']} "
              f"m0={pm_on['m0']:+.2f} m1={pm_on['m1']:+.2f} | "
              f"LESION both_win={r['both_win_lesion']} sizes={r['dg_sizes_lesion']} dense={r['dense_collapse_lesion']} "
              f"m0={pm_le['m0']:+.2f} m1={pm_le['m1']:+.2f} | "
              f"single={r['single_selectivity_on']:+.2f} scr={r['single_scramble_on']:+.2f} "
              f"dissim_bw={r['dissimilar_both_win_on']}", flush=True)
    print(f"  STATUS: {payload['status']}", flush=True)
    print(f"  checks: {json.dumps(payload['checks'])}", flush=True)
    print(f"  pooled: {json.dumps(payload['pooled'])}", flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
