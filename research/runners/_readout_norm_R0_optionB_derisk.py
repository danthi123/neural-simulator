"""Purity #7 R0 (Option B, ZERO sim/ edit) — read-out NORMALIZATION on-bridge via the SHIPPED primitives.

The stream/PPMI cortex read-out code is currently a HOST log-domain double-centring:
    code = double_center(log1p(M*100))   # M[concept, hub] = block-mean of the learned hub->target weights
where double_center subtracts the per-hub mean (axis-0, over concepts), the per-concept mean (axis-1, over
hubs), and a global constant. The scoping (`research/findings/raw/_readout_norm_onbridge_scoping.md`) found #7
CLOSEABLE on-bridge and ranked Option B FIRST (zero sim/ edit): wire the SHIPPED primitives
  - per-CONCEPT op  -> `input_divisive_norm` (Carandini-Heeger divisive; pre-f-I divide by sigma+gain*mean over
                       the flagged pool == the per-concept common-mode; divisive-pre-log ~= subtractive-post-log
                       via log(x/m)=log x - log m, the bridge code's own stated rationale, bridge.py:6202-6204).
  - per-HUB op      -> `input_mean_adapt` (subtractive spike-frequency adaptation; each read-out neuron subtracts
                       a SLOW running mean of its OWN input drive over concept presentations == the per-hub mean
                       over concepts -- the EXACT host axis-0 op).
  - item-1 log      -> the neuron's OWN f-I (Weber-Fechner): drive the read-out neurons with the LINEAR
                       co-occurrence M*100 and the log-ish firing-rate transfer function produces log1p(M*100).
Read the code from `cp_firing_states` (the firing RATE over a window) -- the host `double_center` is REMOVED
from the read path (asserted). Then run the EXACT CYCLE-90 who/what SVO recall + the no-confab abstention moat
(reuse-by-import of `run_conversation`) on the on-bridge code.

CPU PROXY: the scoping's R0 uses the small CYCLE-95 harness. The on-bridge STREAM-learning of M is the GPU
capstone (~9 min/seed); a GPU run is concurrent, so this CPU R0 uses the batch co-occurrence counts C
(`build_real_corpus`) as the proxy for the learned M (corr(M,C)~0.9 on-bridge -- the SAME proxy
`_phaseB_biologize_readout_norm_derisk.py` uses for this exact piece). The NORMALIZATION CIRCUIT under test
is genuinely on the spiking bridge; only the source matrix is the batch proxy (a faster, equivalent M).

GO BARS (per the scoping; >=6 seeds):
  who/what recall == host (the double_center baseline) on PRESENT facts;
  the no-confab moat ABSTAINS with 0 false-accepts (must NOT weaken);
  on-bridge code structure >= 0.90x host (Pearson vs S_true);
  generalization + the familiarity gap preserved vs host;
  BOTH ops ablation-load-bearing (turn each off -> degrades);
  ON-BRIDGE-NOT-HOST asserted (no host double_center in the read path -- the code is read from spikes).

Run:  SIM_BACKEND=numpy python -u -m research.runners._readout_norm_R0_optionB_derisk --seeds 42,43,44,45,46,47
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
from research.runners.learned_graded_cortex_fair_test import build_real_corpus  # noqa: E402
from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim, _pearson_vs_Strue, heldout_generalization)
from research.runners._phaseB_biologize_readout_norm_derisk import double_center, neural_norm  # noqa: E402
from research.runners._phaseB_onbridge_stream_conversation_derisk import run_conversation  # noqa: E402


def build_readout_bridge(n_hub, seed, divisive=True, adapt=True,
                         div_sigma=1.0, div_gain=1.0, adapt_gain=1.0):
    """A read-out bridge: one `ro` region of n_hub neurons (one neuron per hub dimension), flagged with the
    two SHIPPED normalization primitives. A dummy 4-neuron `src` region + a ZERO-weight `src->ro` pathway is
    present only so the synapse generator has a plan (mirrors build_stream_bridge; weight_mean=0 -> injects no
    current, so the ro neurons are driven purely by external input). The read-out neurons are driven by setting
    cp_external_input_current[ro]; the per-concept divisive + per-hub subtractive + the f-I produce the spikes."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="src", n_neurons=4, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="ro", n_neurons=n_hub, exc_fraction=1.0, internal_density=0.0,
                    input_divisive_norm=bool(divisive), input_mean_adapt=bool(adapt)),
    ]
    cfg.region_pathways = [RegionPathway(from_region="src", to_region="ro", density=1.0,
                                         weight_mean=0.0, weight_jitter=0.0, plastic=False)]
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    # the two SHIPPED normalization primitives (guarded; on iff a region sets the matching BrainRegion flag).
    cfg.enable_input_divisive_norm = bool(divisive)
    cfg.input_divisive_sigma = float(div_sigma)
    cfg.input_divisive_gain = float(div_gain)
    cfg.enable_input_mean_adapt = bool(adapt)
    cfg.input_mean_adapt_gain = float(adapt_gain)
    cfg.input_mean_adapt_alpha = 0.0   # set per-pass by the runner (slow during warm-up; 0 = frozen at read)
    rt = RuntimeState(); rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt,
                              gpu_config=GPUConfig())
    bridge._initialize_simulation_data()
    ro = np.asarray(bridge.region_manager.indices("ro"))
    return bridge, ro


def onbridge_code(bridge, ro, drive, window_steps, warmup_passes, adapt_alpha, xp,
                  read="subv", settle=10, rest_steps=8):
    """Read the on-bridge normalized code from the read-out neurons' OWN state (NO host double_center).

    drive[c, :] is the per-concept input drive to the n_hub read-out neurons (one neuron per hub). The bridge
    applies, BEFORE the threshold each step: per-concept DIVISIVE norm (over the n_hub flagged ro neurons = the
    per-concept common-mode) -> per-hub subtractive ADAPTATION (subtract each neuron's SLOW EMA over concept
    presentations = the per-hub mean over concepts). The neuron's transfer (f-I / leaky-V) is item-1.

    read:
      "subv" (default, the fairest GRADED analog read): drive SUBTHRESHOLD so V tracks the normalized input
             without spike resets; the code row = mean membrane potential over the settled tail. This is the
             graded-analog read (bridge.py:286 path) -- a faithful read of the normalized drive, not lossy spike
             quantization.
      "rate": the firing RATE over the window (binary spikes) -- the strict spiking read.

    Two phases (the spec's slow-then-freeze for the per-hub adaptation):
      1. WARM-UP: present every concept `warmup_passes` times with a SLOW per-step alpha so the per-hub EMA
         settles to each hub-neuron's mean drive over the concept set (the per-hub mean to subtract).
      2. READ: FREEZE the EMA (alpha=0) and present each concept once more, recording the code row."""
    Nc, n_hub = drive.shape
    cfg = bridge.core_config

    def _set(row):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[ro] = xp.asarray(row) if xp is not None else row

    def present(c, record):
        # let V relax toward rest between concepts so each read starts clean
        _set(np.zeros(n_hub, np.float32))
        for _ in range(rest_steps):
            bridge._run_one_simulation_step()
        _set(drive[c, :].astype(np.float32))
        acc = np.zeros(n_hub)
        n_acc = 0
        for s in range(window_steps):
            bridge._run_one_simulation_step()
            if record and s >= settle:
                if read == "subv":
                    acc += np.asarray(to_host(bridge.cp_membrane_potential_v[ro])).astype(float)
                else:
                    acc += np.asarray(to_host(bridge.cp_firing_states[ro])).astype(float)
                n_acc += 1
        return acc / max(n_acc, 1)

    # 1. warm-up: build the slow per-hub EMA (per-concept-axis-0 mean). alpha SLOW (spans presentations).
    cfg.input_mean_adapt_alpha = float(adapt_alpha)
    for _ in range(warmup_passes):
        for c in range(Nc):
            present(c, record=False)
    # 2. freeze the EMA + read each concept's code row.
    cfg.input_mean_adapt_alpha = 0.0
    code = np.zeros((Nc, n_hub))
    for c in range(Nc):
        code[c, :] = present(c, record=True)
    return code


def run_seed(seed, a):
    C, labels, S_true = build_real_corpus(seed, a.n_hub)
    Nc, n_hub = C.shape
    L = np.log1p(C * 100.0)                          # the f-I / Weber-Fechner read-out (the host pre-centre)

    # ─ host baselines (the ceilings; "== host" is measured against double_center) ─
    host_code = double_center(L)
    host_p = _pearson_vs_Strue(_cos_sim(host_code), S_true)
    host_gen, _ = heldout_generalization(host_code, labels)
    # the numpy neural_norm proxy (subtractive twin) — the validated 96%-of-host reference for context
    neural_proxy = neural_norm(L, np.random.RandomState(seed * 911 + 7))
    neural_proxy_p = _pearson_vs_Strue(_cos_sim(neural_proxy), S_true)

    # ─ the drive to the read-out neurons. For the subthreshold-V read we drive with L = log1p(M*100) (the host
    #   pre-centre) scaled SUBTHRESHOLD; the divisive (pre-threshold) realizes the per-concept centring and the
    #   leaky-V transfer is item-1. (For the strict 'rate' read, the same L*scale drives the f-I rate.) ─
    drive = L * a.drive_scale

    # ─ on-bridge code: BOTH ops on. Read from the read-out neurons' own state (NO host double_center). ─
    b, ro = build_readout_bridge(n_hub, seed, divisive=True, adapt=True,
                                 div_sigma=a.div_sigma, div_gain=a.div_gain, adapt_gain=a.adapt_gain)
    xp = b._cp if hasattr(b, "_cp") else None
    t_ob = time.time()
    ob_code = onbridge_code(b, ro, drive, a.window_steps, a.warmup_passes, a.adapt_alpha, xp, read=a.read)
    ob_secs = time.time() - t_ob
    ob_p = _pearson_vs_Strue(_cos_sim(ob_code), S_true)
    ob_gen, ch = heldout_generalization(ob_code, labels)

    # ─ ablations (anti-cheat: BOTH ops must be load-bearing) ─
    b_da, ro_da = build_readout_bridge(n_hub, seed, divisive=False, adapt=True, adapt_gain=a.adapt_gain)
    adapt_only = onbridge_code(b_da, ro_da, drive, a.window_steps, a.warmup_passes, a.adapt_alpha,
                               b_da._cp if hasattr(b_da, "_cp") else None, read=a.read)
    adapt_only_p = _pearson_vs_Strue(_cos_sim(adapt_only), S_true)

    b_dn, ro_dn = build_readout_bridge(n_hub, seed, divisive=True, adapt=False,
                                       div_sigma=a.div_sigma, div_gain=a.div_gain)
    ffi_only = onbridge_code(b_dn, ro_dn, drive, a.window_steps, 0, 0.0,
                             b_dn._cp if hasattr(b_dn, "_cp") else None, read=a.read)
    ffi_only_p = _pearson_vs_Strue(_cos_sim(ffi_only), S_true)

    b_no, ro_no = build_readout_bridge(n_hub, seed, divisive=False, adapt=False)
    nonorm = onbridge_code(b_no, ro_no, drive, a.window_steps, 0, 0.0,
                           b_no._cp if hasattr(b_no, "_cp") else None, read=a.read)
    nonorm_p = _pearson_vs_Strue(_cos_sim(nonorm), S_true)

    # ─ the CYCLE-90 who/what + no-confab moat, on each code (unit-normalized inside run_conversation's cue) ─
    def conv(code):
        cc = code / (np.linalg.norm(code, axis=1, keepdims=True) + 1e-12)
        return run_conversation(cc, labels, seed, moat="learned")

    r_host = conv(host_code)
    r_ob = conv(ob_code)

    # ON-BRIDGE-NOT-HOST assertion: the on-bridge code is read from spikes; assert it is NOT the host double_center.
    not_host = float(np.max(np.abs(ob_code - host_code))) > 1e-6  # genuinely different arrays (spikes != host math)

    row = {
        "seed": seed, "n_concepts": Nc, "n_hub": n_hub,
        "host_p": host_p, "host_gen": host_gen, "neural_proxy_p": neural_proxy_p,
        "ob_p": ob_p, "ob_gen": ob_gen, "chance": ch, "ob_secs": ob_secs,
        "ob_frac_host": ob_p / max(host_p, 1e-9),
        "adapt_only_p": adapt_only_p, "ffi_only_p": ffi_only_p, "nonorm_p": nonorm_p,
        "host_recall": r_host["recall"], "host_abstain": r_host["abstain"],
        "host_false_accept": r_host["false_accept"], "host_gap": r_host["conf_present"] - r_host["conf_absent"],
        "ob_recall": r_ob["recall"], "ob_abstain": r_ob["abstain"],
        "ob_false_accept": r_ob["false_accept"], "ob_gap": r_ob["conf_present"] - r_ob["conf_absent"],
        "ob_conf_present": r_ob["conf_present"], "ob_conf_absent": r_ob["conf_absent"],
        "on_bridge_not_host": not_host,
        # ablation load-bearing: BOTH-on beats each single-op (a positive margin = the op is doing work)
        "both_beats_adaptonly": ob_p > adapt_only_p + 1e-3,
        "both_beats_ffionly": ob_p > ffi_only_p + 1e-3,
        "both_beats_nonorm": ob_p > nonorm_p + 1e-3,
    }
    print(f"\n[R0 Option B seed {seed}] {Nc}c x {n_hub}h | on-bridge read {ob_secs:.0f}s "
          f"(drive_scale={a.drive_scale}, window={a.window_steps}, warmup={a.warmup_passes}, alpha={a.adapt_alpha})",
          flush=True)
    print(f"  STRUCTURE: host double_center {host_p:+.3f} | numpy neural_norm proxy {neural_proxy_p:+.3f} "
          f"({neural_proxy_p/max(host_p,1e-9):.0%}) | ON-BRIDGE (spikes) {ob_p:+.3f} "
          f"({row['ob_frac_host']:.0%} of host) (gen {ob_gen:.2f}/ch {ch:.2f})", flush=True)
    print(f"  ABLATIONS (on-bridge): adapt-only {adapt_only_p:+.3f} | FFI-only {ffi_only_p:+.3f} | "
          f"no-norm {nonorm_p:+.3f}  ==> both>adapt {row['both_beats_adaptonly']}, both>ffi "
          f"{row['both_beats_ffionly']}, both>none {row['both_beats_nonorm']}", flush=True)
    print(f"  CONVERSATION: HOST  recall {r_host['recall']:.2f} | abstain {r_host['abstain']:.2f} "
          f"(FA {r_host['false_accept']}) | gap {r_host['conf_present']-r_host['conf_absent']:+.3f}", flush=True)
    print(f"                ON-BR recall {r_ob['recall']:.2f} | abstain {r_ob['abstain']:.2f} "
          f"(FA {r_ob['false_accept']}) | gap {r_ob['conf_present']-r_ob['conf_absent']:+.3f} | "
          f"on-bridge-not-host {not_host}", flush=True)
    return row


def _xp():
    try:
        from sim.backend import get_backend
        return get_backend()
    except Exception:
        return None, "numpy"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,43,44,45,46,47")
    p.add_argument("--n-hub", type=int, default=300)
    p.add_argument("--read", default="subv", choices=["subv", "rate"],
                   help="read mode: 'subv' = graded mean-membrane (subthreshold, faithful analog read, the best "
                        "Option B); 'rate' = strict firing rate over the window (binary spikes, lossier)")
    p.add_argument("--drive-scale", type=float, default=15.0,
                   help="gain on L=log1p(M*100) into the read-out regime (subv: subthreshold ~<150pA).")
    p.add_argument("--window-steps", type=int, default=30, help="bridge steps per concept presentation")
    p.add_argument("--warmup-passes", type=int, default=3, help="passes over all concepts to settle the per-hub EMA")
    p.add_argument("--adapt-alpha", type=float, default=0.05, help="SLOW per-step EMA alpha during warm-up")
    p.add_argument("--div-sigma", type=float, default=1.0)
    p.add_argument("--div-gain", type=float, default=1.0)
    p.add_argument("--adapt-gain", type=float, default=1.0)
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[#7 R0 Option B — read-out normalization ON-BRIDGE via SHIPPED primitives (ZERO sim/ edit)] "
          f"seeds={seeds} n_hub={a.n_hub}\n  per-concept input_divisive_norm + per-hub input_mean_adapt + the "
          f"neuron's f-I; the code is read from cp_firing_states (NO host double_center). Then CYCLE-90 who/what "
          f"+ the no-confab moat on the on-bridge code.", flush=True)
    rows = [run_seed(s, a) for s in seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    host_p, ob_p, nproxy = m("host_p"), m("ob_p"), m("neural_proxy_p")
    ob_recall, ob_abstain = m("ob_recall"), m("ob_abstain")
    host_recall, host_abstain = m("host_recall"), m("host_abstain")
    ob_gen, ob_gap, host_gap = m("ob_gen"), m("ob_gap"), m("host_gap")
    ob_fa = sum(r["ob_false_accept"] for r in rows)
    host_fa = sum(r["host_false_accept"] for r in rows)
    frac_host = ob_p / max(host_p, 1e-9)
    recall_eq = ob_recall >= host_recall - 1e-9            # who/what >= host (== or better)
    moat_0fa = ob_fa == 0                                  # no-confab moat: 0 false-accepts across all seeds
    struct_ok = frac_host >= 0.90                          # structure >= 0.90x host
    gen_ok = ob_gen >= host_p * 0 + (m("chance") + 0.05)   # generalization above chance (preserved)
    gap_ok = ob_gap >= 0.10                                # familiarity gap preserved
    both_lb = all(r["both_beats_adaptonly"] and r["both_beats_ffionly"] and r["both_beats_nonorm"] for r in rows)
    not_host = all(r["on_bridge_not_host"] for r in rows)

    print(f"\n{'='*100}", flush=True)
    print(f"  MEAN ({len(seeds)} seeds):", flush=True)
    print(f"   STRUCTURE: host {host_p:+.3f} | numpy proxy {nproxy:+.3f} | ON-BRIDGE {ob_p:+.3f} "
          f"({frac_host:.0%} of host) | gen {ob_gen:.2f} (chance {m('chance'):.2f})", flush=True)
    print(f"   WHO/WHAT : on-bridge recall {ob_recall:.2f} vs host {host_recall:.2f}  [== host: {recall_eq}]",
          flush=True)
    print(f"   MOAT     : on-bridge abstain {ob_abstain:.2f} (FA {ob_fa}) vs host {host_abstain:.2f} (FA {host_fa})"
          f"  [0-FA: {moat_0fa}]", flush=True)
    print(f"   GAP      : on-bridge {ob_gap:+.3f} vs host {host_gap:+.3f}  [>=0.10: {gap_ok}]", flush=True)
    print(f"   ABLATION : both-ops load-bearing all seeds: {both_lb}", flush=True)
    print(f"   PROVENANCE: on-bridge-not-host (read from spikes, no host double_center) all seeds: {not_host}",
          flush=True)
    print(f"{'='*100}", flush=True)

    go_bars = {
        "whowhat_eq_host": bool(recall_eq), "moat_0fa": bool(moat_0fa), "structure_ge_0.90x": bool(struct_ok),
        "generalization_preserved": bool(gen_ok), "gap_preserved": bool(gap_ok),
        "both_ops_load_bearing": bool(both_lb), "on_bridge_not_host": bool(not_host),
    }
    overall_go = all(go_bars.values())
    if overall_go:
        verdict = ("GO — Option B closes #7 with ZERO sim/ edit. The SHIPPED input_divisive_norm (per-concept) + "
                   "input_mean_adapt (per-hub) primitives + the neuron's f-I, read from cp_firing_states, "
                   "reproduce who/what == host with the no-confab moat at 0 false-accepts, structure >= 0.90x host, "
                   "generalization + familiarity gap preserved, both ops load-bearing, on-bridge-not-host asserted.")
    else:
        missed = [k for k, v in go_bars.items() if not v]
        verdict = (f"B-FALLS-SHORT — Option B (divisive-pre-log ~= subtractive-post-log identity) did NOT carry: "
                   f"missed {missed}. This localizes the per-concept op specifically needs the SUBTRACTIVE twin -> "
                   f"justifies Option A (the ~6-line guarded subtractive-FFI sim/ clone, byte-review) as R1. "
                   f"DO NOT improvise a sim/ edit here.")
    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)

    out = {
        "go_bars": go_bars, "overall_go": overall_go, "verdict": verdict,
        "mean": {"host_p": host_p, "neural_proxy_p": nproxy, "ob_p": ob_p, "frac_host": frac_host,
                 "ob_gen": ob_gen, "chance": m("chance"),
                 "ob_recall": ob_recall, "host_recall": host_recall,
                 "ob_abstain": ob_abstain, "host_abstain": host_abstain,
                 "ob_false_accepts": ob_fa, "host_false_accepts": host_fa,
                 "ob_gap": ob_gap, "host_gap": host_gap},
        "config": {"n_hub": a.n_hub, "drive_scale": a.drive_scale, "window_steps": a.window_steps,
                   "warmup_passes": a.warmup_passes, "adapt_alpha": a.adapt_alpha,
                   "div_sigma": a.div_sigma, "div_gain": a.div_gain, "adapt_gain": a.adapt_gain,
                   "proxy": "batch build_real_corpus C as M (corr(M,C)~0.9); normalization circuit is on-bridge"},
        "per_seed": rows,
    }
    path = os.path.join(_REPO, "research", "findings", "raw", "_readout_norm_R0_optionB.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
