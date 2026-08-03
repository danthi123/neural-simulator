"""LANE C integration: feed the dynamic metacognitive confidence assembly into self_schema.

This is the cheap Step-2 de-risk after
`_second_order_metacog_monitor_derisk --confidence-read learned_acc --learned-feature-mode dynamic` cleared 6 seeds.
It does NOT change production abstain/hedge behavior yet. It tests whether a self_schema confidence sub-block can read
the dynamic meta_schema/aPFC confidence population through fixed on-substrate synapses.

GO: self_schema confidence rate must separate correct vs error trials in the same type-2 SDT currency, while
meta-lesion, self-read lesion, and permuted confidence all collapse. This is still a functional correlate, not a
claim of subjective experience.
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

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion, RegionPathway
from sim.backend import get_backend, to_host

from research.runners import _second_order_metacog_monitor_derisk as meta
from research.runners._gnw_rung1_ignition_curve_derisk import (
    _build_assembly_loop_population, _snapshot_state,
    SETTLE_STEPS,
)
from research.runners._gnw_rung3_report_reasoning_identity_derisk import _dense_projection
from research.runners._self_schema_region_derisk import _spearman
from tools.lab import attributable_to


SELF_CONFID_SIZE = 60
META_TO_SELF_CONFID_GATE = "meta_to_self_confid_fixed"
DEFAULT_META_TO_SELF_CONFID_W = 10.0

DEFAULT_THRESHOLDS = {
    "type1_acc_lo": meta.DEFAULT_THRESHOLDS["type1_acc_lo"],
    "type1_acc_hi": meta.DEFAULT_THRESHOLDS["type1_acc_hi"],
    "type2_auc": meta.DEFAULT_THRESHOLDS["type2_auc"],
    "m_ratio": meta.DEFAULT_THRESHOLDS["m_ratio"],
    "chance_type2_auc": meta.DEFAULT_THRESHOLDS["chance_type2_auc"],
    "collapse_meta_d": meta.DEFAULT_THRESHOLDS["collapse_meta_d"],
    "max_d1_shift": meta.DEFAULT_THRESHOLDS["max_d1_shift"],
    "max_acc_shift": meta.DEFAULT_THRESHOLDS["max_acc_shift"],
    "self_vs_meta_spearman": 0.75,
}


def _learned_config(args):
    return {
        "calib_trials": int(min(args.learned_calib_trials, 64) if args.smoke else args.learned_calib_trials),
        "epochs": int(args.learned_epochs),
        "lr": float(args.learned_lr),
        "l2": float(args.learned_l2),
        "w_max": float(args.learned_w_max),
        "conf_min_pa": float(args.learned_conf_min_pa),
        "conf_max_pa": float(args.learned_conf_max_pa),
        "report_steps": int(args.learned_report_steps),
        "balance_classes": False,
        "symmetric_features": False,
        "response_homeostasis": False,
        "feature_mode": "dynamic",
    }


def build_bridge(seed: int, meta_to_self_w: float = DEFAULT_META_TO_SELF_CONFID_W,
                 lesion_self_read: bool = False):
    xp, _ = get_backend()

    n_ws = meta.ASSEMBLY_SIZE * meta.K_CLASSES
    regions = [
        BrainRegion(name="workspace", n_neurons=n_ws, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="workspace_fs", n_neurons=meta.WORKSPACE_FS_N, exc_fraction=0.0, internal_density=0.0,
                    enable_nmda=False),
        BrainRegion(name="meta_schema", n_neurons=meta.META_SIZE, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=True),
        BrainRegion(name="self_schema", n_neurons=SELF_CONFID_SIZE, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=False),
    ]
    pathways = [
        RegionPathway(from_region="workspace", to_region="workspace_fs", density=0.5,
                      weight_mean=meta.WS_TO_FS_WEIGHT, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="workspace_fs", to_region="workspace", density=0.5,
                      weight_mean=meta.FS_TO_WS_WEIGHT, weight_jitter=0.0, plastic=False),
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
    cfg.seed = int(seed)
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.nmda_tau_decay = float(meta.DEFAULT_NMDA_TAU)
    cfg.nmda_recurrent_tau_decay_ms = float(meta.DEFAULT_NMDA_TAU)
    for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = True
    cfg.stdp_w_max = max(400.0, float(meta.DEFAULT_ATTRACTOR_WEIGHT) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(meta.DEFAULT_ATTRACTOR_WEIGHT) * 4.0)

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    rm = bridge.region_manager
    ws = np.asarray(rm.indices("workspace"), dtype=np.int64)
    fs = np.asarray(rm.indices("workspace_fs"), dtype=np.int64)
    meta_idx = np.asarray(rm.indices("meta_schema"), dtype=np.int64)
    self_confid = np.asarray(rm.indices("self_schema"), dtype=np.int64)
    member_idx = {k: ws[k * meta.ASSEMBLY_SIZE:(k + 1) * meta.ASSEMBLY_SIZE] for k in range(meta.K_CLASSES)}

    union = dict(rm.build_wiring_plan(seed=int(seed)))
    for k in range(meta.K_CLASSES):
        union[f"loop_{k}"] = _build_assembly_loop_population(member_idx[k], float(meta.DEFAULT_ATTRACTOR_WEIGHT))
    w_self = 0.0 if lesion_self_read else float(meta_to_self_w)
    union["meta_to_self_confid"] = _dense_projection(meta_idx, self_confid, w_self, META_TO_SELF_CONFID_GATE)

    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(meta.WS_LOOP_GATE, 0.0)
    bridge.set_plasticity_gate(META_TO_SELF_CONFID_GATE, 0.0)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _snapshot_state(bridge, xp)

    idx = {
        "member_dev": {k: xp.asarray(v) for k, v in member_idx.items()},
        "fs_dev": xp.asarray(fs),
        "meta_dev": xp.asarray(meta_idx),
        "self_confid_dev": xp.asarray(self_confid),
        "confidence_read": meta.LEARNED_ACC_CONFIDENCE_READ,
    }
    return bridge, xp, idx, snap


def _run_report(bridge, xp, idx, confidence_current: float, report_steps: int) -> tuple[float, float]:
    report_steps = int(max(3, report_steps))
    late_start = report_steps - max(1, report_steps // 3)
    meta_acc = 0
    self_acc = 0
    meta_dev = idx["meta_dev"]
    self_dev = idx["self_confid_dev"]
    for t in range(report_steps):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[meta_dev] = xp.float32(float(confidence_current))
        bridge._run_one_simulation_step()
        if t >= late_start:
            meta_acc += int(to_host(bridge.cp_firing_states[meta_dev].astype(xp.float64).sum()))
            self_acc += int(to_host(bridge.cp_firing_states[self_dev].astype(xp.float64).sum()))
    bridge.cp_external_input_current[:] = 0.0
    denom_meta = float(report_steps - late_start) * meta.META_SIZE
    denom_self = float(report_steps - late_start) * SELF_CONFID_SIZE
    return meta_acc / denom_meta, self_acc / denom_self


def _run_block(seed, drive, monitor, learned_config, meta_to_self_w, lesion_meta=False,
               lesion_self_read=False, forced_confidence=None):
    bridge, xp, idx, snap = build_bridge(seed, meta_to_self_w=meta_to_self_w, lesion_self_read=lesion_self_read)
    n_trials = int(len(drive))
    response = np.zeros(n_trials, dtype=int)
    learned_conf = np.zeros(n_trials, dtype=np.float64)
    meta_rate = np.zeros(n_trials, dtype=np.float64)
    self_rate = np.zeros(n_trials, dtype=np.float64)
    for i in range(n_trials):
        tr = meta._run_workspace_decision_trace(
            bridge, xp, idx, snap, drive[i], feature_mode=learned_config["feature_mode"]
        )
        response[i] = meta._response_from_assembly(tr["assembly"])
        c = float(monitor.confidence_from_features(tr["features"]))
        if forced_confidence is not None:
            c = float(forced_confidence[i])
        learned_conf[i] = c
        current = 0.0 if lesion_meta else monitor.current_from_confidence(c)
        meta_rate[i], self_rate[i] = _run_report(bridge, xp, idx, current, learned_config["report_steps"])
    return response, learned_conf, meta_rate, self_rate


def evaluate_seed(seed, n_trials, args, thresholds, verbose=False):
    learned_config = _learned_config(args)
    stimulus, drive, _sig = meta.make_trials(
        seed, n_trials, args.base_pa, args.sig_lo, args.sig_hi, args.stim_noise
    )
    drive_offset_by_class = np.asarray([0.0, float(args.response1_tonic_pa)], dtype=np.float64)
    if float(args.response1_tonic_pa) != 0.0:
        drive = np.clip(drive + drive_offset_by_class, 0.0, None)
    monitor = meta.fit_learned_acc_apfc_monitor(
        seed, learned_config["calib_trials"], args.base_pa, args.sig_lo, args.sig_hi, args.stim_noise,
        args.attractor_weight, args.meta_exc_w, args.meta_inh_w, args.nmda_tau, learned_config,
        drive_offset_by_class=drive_offset_by_class,
    )

    response, learned_conf, meta_rate, self_rate = _run_block(
        seed, drive, monitor, learned_config, args.meta_to_self_w
    )
    type1_accuracy = float(np.mean(response == stimulus))
    d1, c1, hr, far = meta._type1_sdt(stimulus, response)
    t2 = meta._score_type2(stimulus, response, self_rate, c1, d1, seed=seed)
    meta_t2 = meta._score_type2(stimulus, response, meta_rate, c1, d1, seed=seed)
    self_vs_meta = _spearman(meta_rate, self_rate)
    self_vs_learned = _spearman(learned_conf, self_rate)

    response_m, _lc_m, _mr_m, self_rate_m = _run_block(
        seed, drive, monitor, learned_config, args.meta_to_self_w, lesion_meta=True
    )
    type1_accuracy_m = float(np.mean(response_m == stimulus))
    d1_m, c1_m, _, _ = meta._type1_sdt(stimulus, response_m)
    t2_m = meta._score_type2(stimulus, response_m, self_rate_m, c1_m, d1_m, seed=seed)

    response_s, _lc_s, meta_rate_s, self_rate_s = _run_block(
        seed, drive, monitor, learned_config, args.meta_to_self_w, lesion_self_read=True
    )
    type1_accuracy_s = float(np.mean(response_s == stimulus))
    d1_s, c1_s, _, _ = meta._type1_sdt(stimulus, response_s)
    t2_s = meta._score_type2(stimulus, response_s, self_rate_s, c1_s, d1_s, seed=seed)

    rng = np.random.default_rng(seed * 777 + 13)
    perm = rng.permutation(n_trials)
    response_p, _lc_p, _mr_p, self_rate_p = _run_block(
        seed, drive, monitor, learned_config, args.meta_to_self_w, forced_confidence=learned_conf[perm]
    )
    t2_p = meta._score_type2(stimulus, response_p, self_rate_p, c1, d1, seed=seed)

    meta_lesion_collapsed = bool(t2_m["type2_auc"] <= thresholds["chance_type2_auc"]
                                 and t2_m["meta_d"] <= thresholds["collapse_meta_d"])
    self_read_collapsed = bool(t2_s["type2_auc"] <= thresholds["chance_type2_auc"]
                               and t2_s["meta_d"] <= thresholds["collapse_meta_d"])
    permuted_collapsed = bool(t2_p["type2_auc"] <= thresholds["chance_type2_auc"]
                              and t2_p["meta_d"] <= thresholds["collapse_meta_d"])
    domain_ok = bool(abs(d1 - d1_m) <= thresholds["max_d1_shift"]
                     and abs(type1_accuracy - type1_accuracy_m) <= thresholds["max_acc_shift"]
                     and abs(d1 - d1_s) <= thresholds["max_d1_shift"]
                     and abs(type1_accuracy - type1_accuracy_s) <= thresholds["max_acc_shift"])
    go = bool(thresholds["type1_acc_lo"] <= type1_accuracy <= thresholds["type1_acc_hi"]
              and t2["type2_auc"] >= thresholds["type2_auc"]
              and t2["meta_d"] > 0.0 and t2["m_ratio"] >= thresholds["m_ratio"]
              and self_vs_meta >= thresholds["self_vs_meta_spearman"]
              and meta_lesion_collapsed and self_read_collapsed and permuted_collapsed and domain_ok)

    r = {
        "seed": int(seed), "n_trials": int(n_trials), "go": go,
        "drive_offset_by_class": [float(x) for x in drive_offset_by_class],
        "intact": {
            "type1_accuracy": type1_accuracy, "d1": d1, "c1": c1, "hr": hr, "far": far,
            "self_type2_auc": t2["type2_auc"], "self_meta_d": t2["meta_d"], "self_m_ratio": t2["m_ratio"],
            "meta_type2_auc": meta_t2["type2_auc"], "meta_meta_d": meta_t2["meta_d"],
            "self_vs_meta_spearman": self_vs_meta, "self_vs_learned_conf_spearman": self_vs_learned,
            "self_conf_correct_mean": t2["conf_correct_mean"], "self_conf_error_mean": t2["conf_error_mean"],
            "self_rate_range": [float(self_rate.min()), float(self_rate.max())],
            "meta_rate_range": [float(meta_rate.min()), float(meta_rate.max())],
        },
        "meta_lesion": {
            "type1_accuracy": type1_accuracy_m, "d1": d1_m,
            "self_type2_auc": t2_m["type2_auc"], "self_meta_d": t2_m["meta_d"],
            "collapsed": meta_lesion_collapsed,
            "meta_d_attributable": attributable_to(
                "self-schema meta-d from meta assembly (meta-lesion vs intact)",
                t2["meta_d"], t2_m["meta_d"], warn_below=-1.0,
            ),
        },
        "self_read_lesion": {
            "type1_accuracy": type1_accuracy_s, "d1": d1_s,
            "self_type2_auc": t2_s["type2_auc"], "self_meta_d": t2_s["meta_d"],
            "meta_type2_auc": meta._score_type2(stimulus, response_s, meta_rate_s, c1_s, d1_s, seed=seed)["type2_auc"],
            "collapsed": self_read_collapsed,
            "meta_d_attributable": attributable_to(
                "self-schema meta-d from meta->self readout (self-read lesion vs intact)",
                t2["meta_d"], t2_s["meta_d"], warn_below=-1.0,
            ),
        },
        "permuted_confidence": {
            "self_type2_auc": t2_p["type2_auc"], "self_meta_d": t2_p["meta_d"],
            "collapsed": permuted_collapsed,
            "meta_d_attributable": attributable_to(
                "self-schema meta-d from true trial pairing (permuted vs intact)",
                t2["meta_d"], t2_p["meta_d"], warn_below=-1.0,
            ),
        },
        "go_components": {
            "type1_window": thresholds["type1_acc_lo"] <= type1_accuracy <= thresholds["type1_acc_hi"],
            "self_type2": t2["type2_auc"] >= thresholds["type2_auc"],
            "self_meta": t2["meta_d"] > 0.0 and t2["m_ratio"] >= thresholds["m_ratio"],
            "self_tracks_meta": self_vs_meta >= thresholds["self_vs_meta_spearman"],
            "meta_lesion_collapses": meta_lesion_collapsed,
            "self_read_lesion_collapses": self_read_collapsed,
            "permuted_collapses": permuted_collapsed,
            "domain_dissociation": domain_ok,
        },
        "learned_monitor": monitor.to_json(),
    }
    if verbose:
        _print_seed(r)
    return r


def _print_seed(r):
    it = r["intact"]
    print(f"  [seed {r['seed']}] type1_acc={it['type1_accuracy']:.3f} d'={it['d1']:+.2f} "
          f"| self type2_auc={it['self_type2_auc']:.3f} meta_d={it['self_meta_d']:.2f} "
          f"M-ratio={it['self_m_ratio']:.2f} | self~meta={it['self_vs_meta_spearman']:+.2f}", flush=True)
    print(f"           self correct/error mean={it['self_conf_correct_mean']}/{it['self_conf_error_mean']} "
          f"self_rate={it['self_rate_range']} meta_rate={it['meta_rate_range']}", flush=True)
    print(f"    META-LESION self_auc={r['meta_lesion']['self_type2_auc']:.3f} "
          f"meta_d={r['meta_lesion']['self_meta_d']:.2f} collapsed={r['meta_lesion']['collapsed']}", flush=True)
    print(f"    SELF-READ   self_auc={r['self_read_lesion']['self_type2_auc']:.3f} "
          f"meta_d={r['self_read_lesion']['self_meta_d']:.2f} collapsed={r['self_read_lesion']['collapsed']}", flush=True)
    print(f"    PERMUTED    self_auc={r['permuted_confidence']['self_type2_auc']:.3f} "
          f"meta_d={r['permuted_confidence']['self_meta_d']:.2f} collapsed={r['permuted_confidence']['collapsed']}",
          flush=True)
    print(f"    >>> seed GO = {r['go']} {r['go_components']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Lane C self_schema reads dynamic metacognitive confidence.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--n-trials", type=int, default=160)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--base-pa", type=float, default=300.0)
    ap.add_argument("--sig-lo", type=float, default=40.0)
    ap.add_argument("--sig-hi", type=float, default=260.0)
    ap.add_argument("--stim-noise", type=float, default=70.0)
    ap.add_argument("--response1-tonic-pa", type=float, default=0.0,
                    help=("Default-off operating-point scout: tonic current added to class-1 workspace drive "
                          "during ACC calibration and evaluation. Positive values compensate a response-0 bias."))
    ap.add_argument("--attractor-weight", type=float, default=meta.DEFAULT_ATTRACTOR_WEIGHT)
    ap.add_argument("--meta-exc-w", type=float, default=meta.DEFAULT_META_EXC_W)
    ap.add_argument("--meta-inh-w", type=float, default=meta.DEFAULT_META_INH_W)
    ap.add_argument("--nmda-tau", type=float, default=meta.DEFAULT_NMDA_TAU)
    ap.add_argument("--learned-calib-trials", type=int, default=meta.DEFAULT_LEARNED_CALIB_TRIALS)
    ap.add_argument("--learned-epochs", type=int, default=meta.DEFAULT_LEARNED_EPOCHS)
    ap.add_argument("--learned-lr", type=float, default=meta.DEFAULT_LEARNED_LR)
    ap.add_argument("--learned-l2", type=float, default=meta.DEFAULT_LEARNED_L2)
    ap.add_argument("--learned-w-max", type=float, default=meta.DEFAULT_LEARNED_W_MAX)
    ap.add_argument("--learned-conf-min-pa", type=float, default=meta.DEFAULT_LEARNED_CONF_MIN_PA)
    ap.add_argument("--learned-conf-max-pa", type=float, default=meta.DEFAULT_LEARNED_CONF_MAX_PA)
    ap.add_argument("--learned-report-steps", type=int, default=meta.DEFAULT_LEARNED_REPORT_STEPS)
    ap.add_argument("--meta-to-self-w", type=float, default=DEFAULT_META_TO_SELF_CONFID_W)
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/lanes/metacog/metacog_self_schema_integration_smoke.json")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)
    seeds = [args.seed] if args.smoke else (args.seeds if args.seeds is not None else [args.seed])
    n_trials = min(args.n_trials, 64) if args.smoke else args.n_trials

    print(f"[metacog-self] Lane C self_schema<-dynamic-metacog integration | seeds={seeds} n_trials={n_trials} "
          f"backend={args.backend} meta_to_self_w={args.meta_to_self_w}", flush=True)
    print("[metacog-self] confidence source: dynamic ACC/aPFC meta_schema rate; self_schema reads it through "
          "fixed on-substrate meta->self_confid synapses.", flush=True)

    t0 = time.time()
    per_seed = [evaluate_seed(s, n_trials, args, DEFAULT_THRESHOLDS, verbose=True) for s in seeds]
    n_go = sum(1 for r in per_seed if r["go"])
    verdict = "GO" if n_go == len(per_seed) else ("PARTIAL" if n_go > 0 else "NEGATIVE")

    def _mean(path):
        vals = []
        for r in per_seed:
            v = r
            for k in path:
                v = v[k]
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    aggregate = {
        "mean_type1_accuracy": _mean(["intact", "type1_accuracy"]),
        "mean_d1": _mean(["intact", "d1"]),
        "mean_self_type2_auc": _mean(["intact", "self_type2_auc"]),
        "mean_self_meta_d": _mean(["intact", "self_meta_d"]),
        "mean_self_m_ratio": _mean(["intact", "self_m_ratio"]),
        "mean_self_vs_meta_spearman": _mean(["intact", "self_vs_meta_spearman"]),
        "all_meta_lesion_collapse": all(r["meta_lesion"]["collapsed"] for r in per_seed),
        "all_self_read_lesion_collapse": all(r["self_read_lesion"]["collapsed"] for r in per_seed),
        "all_permuted_collapse": all(r["permuted_confidence"]["collapsed"] for r in per_seed),
        "all_domain_dissociation": all(r["go_components"]["domain_dissociation"] for r in per_seed),
    }
    out = {
        "runner": "_laneC_self_schema_metacog_integration_derisk",
        "faculty": "F4 self-model/metacognition (self_schema reads dynamic metacognitive confidence)",
        "theory": "Fleming-Daw metacognitive confidence routed into Graziano-style self_schema report",
        "seeds": seeds, "n_trials": n_trials, "backend": args.backend,
        "thresholds": DEFAULT_THRESHOLDS,
        "meta_to_self_w": float(args.meta_to_self_w),
        "response1_tonic_pa": float(args.response1_tonic_pa),
        "verdict": verdict, "n_go": n_go, "n_seeds": len(seeds),
        "aggregate": aggregate,
        "per_seed": per_seed,
        "preconditions": [
            {
                "name": "per_seed_self_schema_type2_and_tracking_metrics_recorded",
                "ok": all(
                    "intact" in r
                    and "self_type2_auc" in r["intact"]
                    and "self_meta_d" in r["intact"]
                    and "self_vs_meta_spearman" in r["intact"]
                    for r in per_seed
                ),
            },
            {
                "name": "meta_lesion_self_read_lesion_permutation_and_domain_controls_recorded",
                "ok": all(
                    all(k in r for k in ("meta_lesion", "self_read_lesion", "permuted_confidence"))
                    and "domain_dissociation" in r["go_components"]
                    for r in per_seed
                ),
            },
            {
                "name": "verdict_derived_from_recorded_seed_go_flags",
                "ok": verdict == ("GO" if n_go == len(per_seed) else ("PARTIAL" if n_go > 0 else "NEGATIVE")),
            },
        ],
        "honest_scope": ("Runner-level integration de-risk: a self_schema confidence pool reads a dynamic "
                         "meta_schema/aPFC confidence population through fixed on-substrate synapses. Production "
                         "abstain/hedge is not changed yet; subjective experience is not claimed."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[metacog-self] === VERDICT: {verdict} ({n_go}/{len(seeds)} seeds GO) ===", flush=True)
    print(f"[metacog-self]   mean self_type2_auc={aggregate['mean_self_type2_auc']:.3f} "
          f"self_meta_d={aggregate['mean_self_meta_d']:.2f} self~meta={aggregate['mean_self_vs_meta_spearman']:+.2f}",
          flush=True)
    print(f"[metacog-self]   anti-cheats: meta-lesion={aggregate['all_meta_lesion_collapse']} "
          f"self-read-lesion={aggregate['all_self_read_lesion_collapse']} "
          f"permuted={aggregate['all_permuted_collapse']} domain={aggregate['all_domain_dissociation']}",
          flush=True)
    print(f"[metacog-self]   elapsed={time.time()-t0:.1f}s wrote {args.json}", flush=True)
    return 0 if n_go == len(seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
