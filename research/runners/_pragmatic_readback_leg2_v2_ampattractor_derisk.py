"""D (Stage-4 CONVERSANT) - PRAGMATICS LEG-2 v2, DUAL LEVER: value-critic SIGNAL-AMPLIFICATION.

WALL (measured, banked in 2026-08-08-pragmatics-readback-leg2-v2-oracle-RESOLVED-convergence-NEGATIVE-*):
the DA-learned intent->utterance value/policy DIFFERENTIAL is REAL but TINY (~0.01-0.02, a ~1.2x afferent gap)
and sits BELOW per-neuron heterogeneity noise (~2-6% of the ~0.03 population rate). So the graded soft-WTA speaker
picks the heterogeneity winner, not the value winner: critic-argmax ~0.556, actor-WTA ~0.500, both near chance
(0.333, K=3). Already-tried-and-banked: action-localized credit, executed epsilon-greedy (do NOT re-run). A
concurrent agent is testing the DUAL (a HOMEOSTAT that REDUCES the noise) -- this runner does NOT touch that.

THIS LEVER (untried): AMPLIFY the small differential rather than reduce the noise. Add a RECURRENT VALUE ATTRACTOR
-- within-population positive-feedback self-excitation on the competing utterance assemblies (NMDA recurrence, the
same dense self-recurrent E loop the GNW workspace uses), so a small afferent drive difference is pumped to a large
firing difference (winner-take-MORE) while the shared FS pool supplies cross-population competition. Reuses ALL the
Leg-2 v2 machinery by import; the ONLY addition is the recurrent utter[u]->utter[u] loop (weight W_REC).

CHEAP DECISIVE ISOLATION TEST (this smoke): the amplifier's raw capability is tested on a CONTROLLED afferent gap
via the committed oracle-weight readout -- intent[t]->utter[t]=GAP, others=1.0, greedy neural WTA -- so we measure
"can recurrent amplification resolve a SMALL coherent afferent gap against static heterogeneity?" WITHOUT the
360-trial DA training. This is the ceiling for what the amplifier can do for the readout; if it fails here it
cannot rescue the trained convergence, and if it passes it is worth wiring into the full loop.

ARMS (all share ONE substrate per seed; differ ONLY in GAP and W_REC):
  REF   gap=8.0,  wrec=0    -- sanity: reproduces the v2 oracle result (acc ~1.0); validates the harness.
  BASE  gap=g,    wrec=0    -- the NEGATIVE: a small gap loses to heterogeneity (acc ~ chance). [no-lever control]
  AMP   gap=g,    wrec>0    -- the TREATMENT: does the recurrent attractor lift acc WELL ABOVE BASE?
  CTRL  gap=1.0,  wrec>0    -- SKEPTICAL zeroed-afferent control: gain applied, NO signal. If AMP works by
                              amplifying the SIGNAL it must beat CTRL; if AMP ~ CTRL the gain only sharpens the
                              heterogeneity (noise) winner -> NEGATIVE (order-preserving amplification).

READ (honesty): the WTA winner is a NEURAL read (highest late-window population rate after lateral inhibition),
NOT np.argmax over an imported table -- argmax is only over the 3 measured neural population rates (the body
reading motor output). No sim/ edit; reuse-by-import; additive/default-off.

Usage:
  SIM_BACKEND=numpy python -u -m research.runners._pragmatic_readback_leg2_v2_ampattractor_derisk --smoke \
      --seeds 42 43 44 100 --json research/findings/raw/_pragmatic_success/leg2_v2_ampattractor_smoke.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.regions import BrainRegion, RegionPathway  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402

from research.runners._gnw_rung1_ignition_curve_derisk import (  # noqa: E402
    _snapshot_state, _restore_state, SETTLE_STEPS, _build_assembly_loop_population,
)
from research.runners._gnw_rung3_report_reasoning_identity_derisk import _dense_projection  # noqa: E402
from research.runners._self_schema_region_derisk import WS_LOOP_GATE  # noqa: E402
from research.runners._pragmatic_success_coincidence_derisk import (  # noqa: E402
    ITEM, K, INTENT_PA,
)
# reuse the calibrated readout operating point from the banked v2 runner
from research.runners._pragmatic_success_readback_leg2_v2_derisk import (  # noqa: E402
    UTT_ITEM, UTT_FS_N, UTT_FS_W, FS_UTT_W, UTT_DRIVE_PA, SETTLE_MS, READ_UTT, W_OTHER,
)

CHANCE = 1.0 / K


def build_amp_bridge(seed, gap, wrec, nmda=False, jit=0.0):
    """ONE bridge: intent[K] -> utter[K] (afferent gap: intent[t]->utter[t]=gap, others=W_OTHER, FROZEN) with a
    shared FS pool for cross-population competition AND a within-population recurrent-E attractor loop on each
    utter[u] at weight `wrec` (wrec=0 -> no amplification). `nmda` toggles slow NMDA recurrence on the utter
    region (OFF matches the banked v2 graded regime; ON gives a slower/stickier attractor). All plasticity OFF --
    this is the amplifier CAPABILITY test on a controlled afferent gap, not a learning run."""
    xp, _ = get_backend()
    regions = [
        BrainRegion(name="intent", n_neurons=ITEM * K, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="utter", n_neurons=UTT_ITEM * K, exc_fraction=1.0, internal_density=0.0, enable_nmda=bool(nmda)),
        BrainRegion(name="utter_fs", n_neurons=UTT_FS_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
    ]
    pathways = [
        RegionPathway(from_region="utter", to_region="utter_fs", density=0.6, weight_mean=UTT_FS_W,
                      weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="utter_fs", to_region="utter", density=0.6, weight_mean=FS_UTT_W,
                      weight_jitter=0.0, plastic=False),
    ]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)                     # ⚠ seeds the substrate (het thresholds); see CLAUDE.md seed trap
    cfg.enable_nmda = bool(nmda)
    cfg.nmda_ratio = 0.5
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_homeostasis", "enable_reward_modulation",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = True   # THE noise we are fighting -- keep ON
    cfg.stdp_w_max = max(400.0, float(wrec) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(wrec) * 4.0)

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    rm = bridge.region_manager
    intent = np.asarray(rm.indices("intent"), dtype=np.int64)
    utter = np.asarray(rm.indices("utter"), dtype=np.int64)
    intent_k = {k: intent[k * ITEM:(k + 1) * ITEM] for k in range(K)}
    utter_k = {k: utter[k * UTT_ITEM:(k + 1) * UTT_ITEM] for k in range(K)}

    union = dict(rm.build_wiring_plan(seed=int(seed)))
    wrng = np.random.default_rng(int(seed) * 17 + 3)
    for t in range(K):
        for u in range(K):
            base = float(gap) if (t == u) else float(W_OTHER)
            proj = _dense_projection(intent_k[t], utter_k[u], base, WS_LOOP_GATE)
            if jit > 0.0:
                n = proj["initial_weights"].shape[0]
                proj["initial_weights"] = np.clip(
                    base * (1.0 + wrng.normal(0.0, jit, n)), 1e-3, None).astype(np.float32)
            union[f"i2u_{t}_{u}"] = proj
    if wrec > 0.0:
        for u in range(K):
            union[f"rec_{u}"] = _build_assembly_loop_population(utter_k[u], float(wrec))

    inh = list(rm.inhibitory_indices("utter_fs"))
    bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)      # everything frozen

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _snapshot_state(bridge, xp)
    idx = {"intent": {k: xp.asarray(intent_k[k]) for k in range(K)},
           "utter": {k: xp.asarray(utter_k[k]) for k in range(K)}}
    return bridge, xp, idx, snap


def _read_wta(bridge, xp, idx, snap, intent_t, settle_ms=SETTLE_MS, read_utt=READ_UTT):
    """Drive intent[t]; the graded/attractor competition settles; winner = utter population with the highest
    late-window rate (neural WTA). Returns (winner, rates[K])."""
    bridge.cp_external_input_current[:] = 0.0
    _restore_state(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0
    acc = np.zeros(K)
    for s in range(settle_ms):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[idx["intent"][intent_t]] = xp.float32(INTENT_PA)
        for u in range(K):
            if UTT_DRIVE_PA != 0.0:
                bridge.cp_external_input_current[idx["utter"][u]] = xp.float32(UTT_DRIVE_PA)
        bridge._run_one_simulation_step()
        if s >= settle_ms - read_utt:
            for u in range(K):
                acc[u] += float(to_host(bridge.cp_firing_states[idx["utter"][u]].astype(xp.float64).sum()))
    rates = acc / (read_utt * UTT_ITEM)
    return int(np.argmax(rates)), rates


def run_arm(seed, gap, wrec, nmda=False, jit=0.0):
    """Build the substrate at (gap,wrec) and read the WTA winner for every intent. acc = P(winner==intent).
    For the zeroed-afferent control (gap==W_OTHER) there is no true target, so acc is the alignment of the
    (noise-driven) winner with the nominal target index -- expected ~chance by symmetry."""
    bridge, xp, idx, snap = build_amp_bridge(seed, gap, wrec, nmda=nmda, jit=jit)
    winners, rates_all = {}, {}
    for t in range(K):
        w, rates = _read_wta(bridge, xp, idx, snap, t)
        winners[t] = w
        rates_all[t] = [round(float(x), 5) for x in rates]
    acc = float(np.mean([winners[t] == t for t in range(K)]))
    return {"seed": int(seed), "gap": gap, "wrec": wrec, "acc": acc,
            "winners": {str(t): int(winners[t]) for t in range(K)}, "rates": rates_all}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100])
    ap.add_argument("--gap-small", type=float, default=1.3, help="small afferent gap (mimics the ~1.2x value gap)")
    ap.add_argument("--gap-ref", type=float, default=8.0, help="large gap sanity (reproduces v2 oracle acc~1.0)")
    ap.add_argument("--wrec", type=float, nargs="+", default=[8.0, 20.0, 40.0], help="recurrent attractor weights")
    ap.add_argument("--gap-sweep", type=float, nargs="+", default=None,
                    help="if set, sweep BASE (wrec=0) over these gaps to locate the near-chance regime")
    ap.add_argument("--nmda", action="store_true", help="enable slow NMDA recurrence on the utter region")
    ap.add_argument("--jit", type=float, default=0.0,
                    help="per-synapse relative jitter on the intent->utter weights (0=clean coherent gap; >0 = a "
                         "NOISY differential surrogate for a DA-learned weight matrix)")
    ap.add_argument("--jit-sweep", type=float, nargs="+", default=None,
                    help="sweep BASE (wrec=0) over these jitters at --gap-small to locate the near-chance regime")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_pragmatic_success/leg2_v2_ampattractor_smoke.json")
    args = ap.parse_args()
    get_backend("numpy")
    seeds = args.seeds
    t0 = time.time()
    out = {"runner": "_pragmatic_readback_leg2_v2_ampattractor_derisk", "seeds": seeds, "chance": CHANCE,
           "operating_point": {"UTT_FS_W": UTT_FS_W, "FS_UTT_W": FS_UTT_W, "UTT_DRIVE_PA": UTT_DRIVE_PA,
                               "SETTLE_MS": SETTLE_MS, "READ_UTT": READ_UTT, "UTT_ITEM": UTT_ITEM}, "arms": {}}

    def _agg(rows):
        return {"mean_acc": round(float(np.mean([r["acc"] for r in rows])), 4),
                "per_seed_acc": [r["acc"] for r in rows], "rows": rows}

    nmda = bool(args.nmda)
    jit = float(args.jit)
    out["nmda"] = nmda
    out["jit"] = jit
    if args.gap_sweep is not None:
        print(f"[amp] GAP SWEEP (wrec=0) gaps={args.gap_sweep} seeds={seeds} nmda={nmda} jit={jit}", flush=True)
        for g in args.gap_sweep:
            rows = [run_arm(s, g, 0.0, nmda=nmda, jit=jit) for s in seeds]
            a = _agg(rows)
            out["arms"][f"BASE_gap{g}"] = a
            print(f"  gap={g:<5} wrec=0   mean_acc={a['mean_acc']:.3f} (chance {CHANCE:.3f})  "
                  f"per_seed={a['per_seed_acc']}", flush=True)

    if args.jit_sweep is not None:
        print(f"[amp] JIT SWEEP (wrec=0, gap={args.gap_small}) jits={args.jit_sweep} seeds={seeds}", flush=True)
        for j in args.jit_sweep:
            rows = [run_arm(s, args.gap_small, 0.0, nmda=nmda, jit=j) for s in seeds]
            a = _agg(rows)
            out["arms"][f"BASE_jit{j}"] = a
            print(f"  jit={j:<5} gap={args.gap_small} wrec=0   mean_acc={a['mean_acc']:.3f} (chance {CHANCE:.3f})  "
                  f"per_seed={a['per_seed_acc']}", flush=True)

    # REF sanity (validates the harness reproduces the v2 oracle acc~1.0) -- always CLEAN gap (jit=0)
    ref = _agg([run_arm(s, args.gap_ref, 0.0, nmda=nmda, jit=0.0) for s in seeds])
    out["arms"]["REF_biggap_norec"] = ref
    print(f"[amp] REF   gap={args.gap_ref} wrec=0 jit=0   mean_acc={ref['mean_acc']:.3f}  (expect ~1.0)", flush=True)

    # BASE negative (small gap, no amplification, at the operating jitter)
    base = _agg([run_arm(s, args.gap_small, 0.0, nmda=nmda, jit=jit) for s in seeds])
    out["arms"]["BASE_smallgap_norec"] = base
    print(f"[amp] BASE  gap={args.gap_small} wrec=0 jit={jit}   mean_acc={base['mean_acc']:.3f}  "
          f"(target ~chance {CHANCE:.3f})", flush=True)

    # AMP treatment + CTRL zeroed-afferent, per wrec (same operating jitter)
    for w in args.wrec:
        amp = _agg([run_arm(s, args.gap_small, w, nmda=nmda, jit=jit) for s in seeds])
        ctrl = _agg([run_arm(s, W_OTHER, w, nmda=nmda, jit=jit) for s in seeds])   # zeroed afferent, gain ON
        out["arms"][f"AMP_smallgap_wrec{w}"] = amp
        out["arms"][f"CTRL_zeroaff_wrec{w}"] = ctrl
        verdict = ("AMPLIFIES-SIGNAL" if (amp["mean_acc"] - base["mean_acc"] >= 0.20
                                          and amp["mean_acc"] - ctrl["mean_acc"] >= 0.20)
                   else "no-signal-gain")
        print(f"[amp] wrec={w:<5} AMP(smallgap)={amp['mean_acc']:.3f}  "
              f"CTRL(zeroaff)={ctrl['mean_acc']:.3f}  vs BASE={base['mean_acc']:.3f}  -> {verdict}", flush=True)

    out["elapsed_seconds"] = round(time.time() - t0, 1)
    # decision
    best_amp = max((out["arms"][f"AMP_smallgap_wrec{w}"]["mean_acc"] for w in args.wrec), default=0.0)
    best_ctrl_at_best = None
    for w in args.wrec:
        if abs(out["arms"][f"AMP_smallgap_wrec{w}"]["mean_acc"] - best_amp) < 1e-9:
            best_ctrl_at_best = out["arms"][f"CTRL_zeroaff_wrec{w}"]["mean_acc"]
            break
    out["decision"] = {
        "ref_acc": ref["mean_acc"], "base_acc": base["mean_acc"], "best_amp_acc": best_amp,
        "ctrl_at_best_amp": best_ctrl_at_best,
        "amp_beats_base_by": round(best_amp - base["mean_acc"], 4),
        "amp_beats_ctrl_by": round(best_amp - (best_ctrl_at_best if best_ctrl_at_best is not None else 0.0), 4),
        "verdict": ("POSITIVE" if (best_amp - base["mean_acc"] >= 0.20
                                   and best_amp - (best_ctrl_at_best or 1.0) >= 0.20)
                    else "NEGATIVE"),
    }
    Path(os.path.dirname(os.path.abspath(args.json))).mkdir(parents=True, exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(out, f, indent=2, default=str)
    d = out["decision"]
    print("\n" + "=" * 100, flush=True)
    print(f"[amp] DECISION={d['verdict']}  REF={d['ref_acc']:.3f} BASE={d['base_acc']:.3f} "
          f"bestAMP={d['best_amp_acc']:.3f} CTRL@best={d['ctrl_at_best_amp']}  "
          f"(AMP-BASE={d['amp_beats_base_by']:+.3f}, AMP-CTRL={d['amp_beats_ctrl_by']:+.3f})", flush=True)
    print(f"[amp] elapsed={out['elapsed_seconds']}s  wrote {args.json}\n" + "=" * 100, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
