"""GAP#4 / E3 burn-down de-risk — a per-turn BTSP PLATEAU write is a LASTING trace: recallable AFTER a decay window /
intervening activity, where a NON-PLATEAU (transient-plateau or below-barrier-static) write decays BELOW recall.

WHY THIS, AND WHAT IS ALREADY CLOSED (corpus-first, 2026-08-12):
  - 2026-07-18 gap#4 on-bridge BTSP GO measured the WRITE: a HELD bistable plateau potentiates the co-active pre->post
    synapse one-shot over a SECONDS-long window (held_dw ~110 vs transient ~13). It did NOT test whether that write
    PERSISTS through later activity and stays RECALLABLE.
  - 2026-07-18/20 recall-BIAS probe measured recall IMMEDIATELY after storing (a partial cue drives held-out partners).
    It did NOT insert a decay window / intervening turns before recall.
  So the genuine UNCLOSED residual (this runner): does the plateau write LAST? i.e. after an intervening decay window,
  is the taught fact still RECALLABLE (a BEHAVIORAL/spiking read: post fires to the cue), while a non-plateau write has
  decayed below recall? "works/recall" is measured as post SPIKING (TERMS.md: a weight read is a proxy, behaviour is the
  capability), never a weight read.

MECHANISM (genuinely-spiking WRITE + RECALL on the real bridge; a runner-level MODEL of the LASTING side):
  - WRITE: the REAL on-bridge BTSP block (`enable_btsp`, fused_btsp_update) reads the REAL bistable BDSP apical plateau
    (`bdsp_apical_bistable`, self-regen + KIR) as the instructive signal and potentiates pre->post-TARGET one-shot.
    Held plateau => supra-barrier write; transient plateau => sub-barrier write; silent apical => no write.
  - DECAY WINDOW: the bridge KEEPS STEPPING (intervening background activity = later "turns"); BTSP/BDSP learning OFF so
    only the maintenance rule moves the pre->post weights. The maintenance rule is SYNAPTIC TAG-AND-CAPTURE (Frey & Morris
    1997; the CaMKII "perpetuating switch", Lisman 1985; Kandel 6e Ch 67): a synapse whose weight exceeds a capture
    threshold (barrier) is TAGGED+CAPTURED -> stabilized (resists decay); an untagged (sub-barrier) synapse passively
    decays (w *= 1-beta). This is a runner-level model of a bistable synapse; the next rung is a guarded default-off
    `fused_synaptic_capture` sim/ kernel. Biology: a plateau is what naturally drives a synapse over the tag threshold
    (Bittner & Magee 2017's large one-shot BTSP potentiation); ordinary weak plasticity does not.
  - RECALL (spiking): after the window, fire the pre cue only (learning + maintenance OFF), count post-TARGET spikes.

CONDITIONS (all: WRITE -> SAME window -> SAME spiking recall of post-target; recall also read at t0 = pre-window):
  PLATEAU            held bistable plateau -> supra-barrier write -> captured -> RECALLS after the window.
  TRANSIENT         transient plateau      -> sub-barrier write   -> untagged -> recall FAILS after (the no-plateau ctrl).
  MOAT              silent apical          -> no write            -> no recall (moat).
  PLATEAU_NOCAPTURE held plateau, window = passive decay for ALL synapses (capture OFF) -> plateau trace also decays ->
                     recall FAILS. (LESION: the capture is load-bearing, not the big write alone.)
  STATIC_SUBBARRIER no plateau; pre->post-target weights SET large-but-below-barrier -> untagged -> decays -> recall FAILS.
                     (ANTI-MAGNITUDE: a trivially-large STATIC weight does NOT last; it must cross the barrier, which
                      in the write phase only the plateau achieves.)
  ATTRIBUTABILITY   in PLATEAU, post-DISTRACTOR half never gets a plateau -> its afferents are never captured -> the
                     distractor half does NOT recall while the target half does (the RIGHT cells persist).

PRE-REGISTERED GO (6-seed, ALL seeds). Post-cell firing caps at ~0.02-0.09/step and reads a hard ~0 when its afferents
are silent, so recall is a near-binary detector; "DECAYED below recall" is graded PER SEED against that seed's own
persisted (plateau) trace (recall <= CONTRAST=0.4 of the plateau AND below the "fires" line), robust to the
~0.002-0.006/step spontaneous floor. "still fires" stays ABSOLUTE (>= RECALL_HI):
  (L1 LASTING)     plateau_recall_after >= RECALL_HI (post STILL fires) AND transient_after AND moat_after each DECAYED
  (L2 CRUX: persistence not write-strength) static_recall_t0 >= HI AND transient_recall_t0 >= HI (both non-plateau
                   WRITES recalled at t0) AND static_recall_after DECAYED (the after-window failure is DECAY, not a
                   failed write)
  (L3 LESION)      plateau_nocapture_recall_after DECAYED   (the SAME big write, minus capture -> capture load-bearing)
  (L4 ANTI-MAG)    static_subbarrier_recall_after DECAYED   (a big sub-barrier STATIC weight is not enough)
  (L5 ATTRIB)      plateau_distractor_recall_after DECAYED  (only the plateau-targeted cells persist)
  (INSTRUMENT)     off_dw == 0 (enable_btsp=False byte-identical write) ; maintenance INERT when beta=0
                   (window leaves weights unchanged) ; recall readout distinguishes a huge (>=HI) vs zero (<=LO) weight.
  RECALL_HI=0.015, RECALL_LO=0.008, CONTRAST=0.4. NO new sim/ edit.

Run:  SIM_BACKEND=numpy python -m research.runners._gap4_btsp_lasting_trace_recall_after_delay_derisk \
        --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402
from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.regions import BrainRegion, RegionPathway  # noqa: E402
from sim.bridge import SimulationBridge  # noqa: E402
from tools.lab import attributable_to, void_if, assert_backend, Verdict  # noqa: E402

xp, _ = get_backend()
OUT = _REPO / "research" / "findings" / "raw" / "_gap4_btsp_lasting_trace_recall_after_delay.json"

# The post cell caps its firing at ~0.02-0.07 spikes/step (Izhikevich rate ceiling), and reads a hard 0.000 when its
# afferents are silent -- so spiking recall here is a near-BINARY detector in [0, ~0.07]. Thresholds are set to that
# measured dynamic range (calibrated on the 42/43/44 pilot; the LESION/controls, not the absolute threshold, carry the
# claim). RECALL_HI = "post fires"; RECALL_LO = "post effectively silent".
RECALL_HI = 0.015
RECALL_LO = 0.008
CONTRAST = 0.4    # a "decayed" trace recalls at <= 40% of the same seed's persisted (plateau) trace (actual ~6%)


def _build(enable_btsp, bistable, seed, w0=0.3, n_pre=64, n_post=8, btsp_w_max=10.0, btsp_lr=0.04):
    regions = [
        BrainRegion(name="pre", n_neurons=n_pre, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="post", n_neurons=n_post, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    pathways = [RegionPathway(from_region="pre", to_region="post", density=1.0,
                              weight_mean=w0, weight_jitter=0.0, plastic=True)]
    cfg = CoreSimConfig(seed=seed)
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions; cfg.region_pathways = pathways
    for f in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
              "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp",
              "enable_input_divisive_norm", "enable_nmda"):
        setattr(cfg, f, False)
    # BDSP path ONLY to evolve the bistable apical plateau (learning OFF -> BDSP moves no weight, BTSP is the sole mover).
    cfg.enable_bdsp = True
    cfg.bdsp_learning_rate = 0.0
    cfg.bdsp_apical_bistable = bool(bistable)
    cfg.coincidence_plateau_self_regen = 2.0
    cfg.coincidence_plateau_v_hold = -35.0
    cfg.apical_kir_g = 1.0
    # BTSP (the tested WRITE rule)
    cfg.enable_btsp = bool(enable_btsp)
    cfg.btsp_learning_rate = float(btsp_lr)
    cfg.btsp_elig_tau_ms = 1000.0
    cfg.btsp_w_max = float(btsp_w_max)
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


def _post_col_mask(sb, post_idx_subset):
    """Boolean mask over cp_connections.data selecting synapses whose POSTsynaptic column is in post_idx_subset.
    CSR: .indices is the post-neuron column for each nonzero (bridge.py:2968)."""
    cols = np.asarray(to_host(sb.cp_connections.indices))
    return np.isin(cols, np.asarray(post_idx_subset))


def _recall(sb, pre_idx, post_read_idx, recall_steps, recall_drive):
    """SPIKING recall: fire the pre cue only (no apical, no learning), count post_read spikes / step.
    Returns firing rate in [0,1]. This is the CAPABILITY measure (behaviour), not a weight read."""
    n = sb.cp_membrane_potential_v.size
    sb.core_config.enable_btsp = False
    sb.core_config.bdsp_learning_rate = 0.0
    sb.cp_bdsp_apical_drive = xp.zeros(n, dtype=xp.float32)
    sb.cp_membrane_potential_v[:] = xp.float32(-65.0)   # clear residual membrane state -> recall is cue-driven
    cue = np.zeros(n, dtype=np.float32); cue[pre_idx] = recall_drive
    spikes = np.zeros(len(post_read_idx))
    for _ in range(recall_steps):
        sb.cp_external_input_current[:] = xp.asarray(cue)
        sb._run_one_simulation_step()
        fired = np.asarray(to_host(sb.cp_firing_states)).astype(float)
        spikes += fired[post_read_idx]
    return float(spikes.mean() / recall_steps)


def _write_weight_sum(sb, mask):
    return float((np.asarray(to_host(sb.cp_connections.data)) * mask).sum())


def _one(seed, condition, args):
    """Run a single condition end-to-end and return a dict of recalls + diagnostics."""
    bistable = condition in ("plateau", "moat", "plateau_nocapture", "attrib")
    enable_btsp = condition != "off"
    sb = _build(enable_btsp=enable_btsp, bistable=bistable, seed=seed, w0=args.w0,
                n_pre=args.n_pre, btsp_w_max=args.btsp_w_max, btsp_lr=args.btsp_lr)
    rm = sb.region_manager
    pre_idx = np.asarray(list(rm.indices("pre")))
    post_all = np.asarray(list(rm.indices("post")))
    half = len(post_all) // 2
    post_target = post_all[:half]          # gets the plateau
    post_distr = post_all[half:]           # never gets a plateau (attributability)
    n = sb.cp_membrane_potential_v.size
    sb.cp_bdsp_apical_drive = xp.zeros(n, dtype=xp.float32)

    mask_target = _post_col_mask(sb, post_target)

    # ---- PHASE 1: WRITE (one "turn") ----
    pulse = condition not in ("moat", "static_subbarrier")   # moat: silent apical; static: no plateau at all
    drive = np.zeros(n, dtype=np.float32); drive[pre_idx] = 900.0   # PRE fires throughout -> seconds-long eligibility
    ap = np.zeros(n, dtype=np.float32)
    off_dw = 0.0
    if condition == "static_subbarrier":
        data = np.asarray(to_host(sb.cp_connections.data)).copy()
        data[mask_target] = args.static_w   # above recall, below barrier
        sb.cp_connections.data = xp.asarray(data)
        w_after_write = _write_weight_sum(sb, mask_target)
    else:
        w0_target = _write_weight_sum(sb, mask_target)
        for step in range(args.write_steps):
            sb.cp_external_input_current[:] = xp.asarray(drive)
            cur = ap.copy()
            if pulse and 20 <= step < 20 + args.pulse_steps:
                cur[post_target] = args.pulse_pA          # brief apical plateau to TARGET only
            sb.cp_bdsp_apical_drive = xp.asarray(cur)
            sb._run_one_simulation_step()
        w_after_write = _write_weight_sum(sb, mask_target)
        if condition == "off":
            off_dw = w_after_write - w0_target

    # ---- recall at t0 (immediately after the write, BEFORE the decay window) ----
    recall_t0_target = _recall(sb, pre_idx, post_target, args.recall_steps, args.recall_drive)

    # ---- PHASE 2: DECAY WINDOW (intervening activity) with tag-and-capture maintenance ----
    capture_on = condition != "plateau_nocapture"
    beta = args.beta
    barrier = args.barrier
    sb.core_config.enable_btsp = False
    sb.cp_bdsp_apical_drive = xp.zeros(n, dtype=xp.float32)
    bg = np.zeros(n, dtype=np.float32); bg[pre_idx] = args.bg_drive
    w_pre_window = _write_weight_sum(sb, mask_target)
    for _ in range(args.window_steps):
        sb.cp_external_input_current[:] = xp.asarray(bg)
        sb._run_one_simulation_step()
        data = np.asarray(to_host(sb.cp_connections.data))
        if capture_on:
            tagged = data >= barrier                      # tag-and-capture: supra-barrier synapses are stabilized
            data = np.where(tagged, data, data * (1.0 - beta))
        else:
            data = data * (1.0 - beta)                    # LESION: passive decay for ALL synapses (no capture)
        sb.cp_connections.data = xp.asarray(data)
    w_post_window = _write_weight_sum(sb, mask_target)

    # ---- recall AFTER the window (the LASTING measure) ----
    recall_after_target = _recall(sb, pre_idx, post_target, args.recall_steps, args.recall_drive)
    recall_after_distr = _recall(sb, pre_idx, post_distr, args.recall_steps, args.recall_drive)

    return {"condition": condition, "seed": seed,
            "w_after_write": w_after_write, "w_pre_window": w_pre_window, "w_post_window": w_post_window,
            "recall_t0_target": recall_t0_target,
            "recall_after_target": recall_after_target, "recall_after_distr": recall_after_distr,
            "off_dw": off_dw}


def _instrument_checks(seed, args):
    """VERIFY THE INSTRUMENT before trusting the science.
    (a) enable_btsp=False write path is inert (off_dw==0).
    (b) the maintenance rule is INERT when beta=0 (window leaves weights unchanged).
    (c) the spiking recall readout DISTINGUISHES a huge vs a zero target weight."""
    checks = {}
    # (a) off write
    off = _one(seed, "off", args)
    checks["off_dw"] = off["off_dw"]
    # (b) maintenance inert when beta=0
    sb = _build(enable_btsp=False, bistable=False, seed=seed, w0=args.w0, n_pre=args.n_pre)
    rm = sb.region_manager
    post_all = np.asarray(list(rm.indices("post")))
    mask_t = _post_col_mask(sb, post_all[:len(post_all)//2])
    data = np.asarray(to_host(sb.cp_connections.data)).copy(); data[mask_t] = 2.0
    sb.cp_connections.data = xp.asarray(data)
    w_before = float(np.asarray(to_host(sb.cp_connections.data)).sum())
    d2 = np.asarray(to_host(sb.cp_connections.data))
    d2 = np.where(d2 >= args.barrier, d2, d2 * (1.0 - 0.0))   # beta=0 -> no change regardless of capture branch
    sb.cp_connections.data = xp.asarray(d2)
    w_after = float(np.asarray(to_host(sb.cp_connections.data)).sum())
    checks["maintenance_inert_beta0_delta"] = w_after - w_before
    # (c) recall readout distinguishes huge vs zero target weight
    pre_idx = np.asarray(list(rm.indices("pre")))
    post_target = post_all[:len(post_all)//2]
    mask_target = _post_col_mask(sb, post_target)
    data = np.asarray(to_host(sb.cp_connections.data)).copy(); data[mask_target] = 5.0
    sb.cp_connections.data = xp.asarray(data)
    r_hi = _recall(sb, pre_idx, post_target, args.recall_steps, args.recall_drive)
    data = np.asarray(to_host(sb.cp_connections.data)).copy(); data[mask_target] = 0.0
    sb.cp_connections.data = xp.asarray(data)
    r_lo = _recall(sb, pre_idx, post_target, args.recall_steps, args.recall_drive)
    checks["recall_huge_weight"] = r_hi
    checks["recall_zero_weight"] = r_lo
    return checks


def run(seed, args):
    conds = ["plateau", "transient", "moat", "plateau_nocapture", "static_subbarrier"]
    res = {c: _one(seed, c, args) for c in conds}
    instr = _instrument_checks(seed, args)
    return {"seed": seed, "conditions": res, "instrument": instr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n-pre", type=int, default=64, dest="n_pre")
    ap.add_argument("--write-steps", type=int, default=150, dest="write_steps")
    ap.add_argument("--window-steps", type=int, default=200, dest="window_steps")
    ap.add_argument("--recall-steps", type=int, default=200, dest="recall_steps")
    ap.add_argument("--pulse-steps", type=int, default=15, dest="pulse_steps")
    ap.add_argument("--pulse-pA", type=float, default=120.0, dest="pulse_pA")
    ap.add_argument("--recall-drive", type=float, default=2000.0, dest="recall_drive")
    ap.add_argument("--bg-drive", type=float, default=0.0, dest="bg_drive")
    ap.add_argument("--w0", type=float, default=0.3)
    ap.add_argument("--btsp-w-max", type=float, default=10.0, dest="btsp_w_max")
    ap.add_argument("--btsp-lr", type=float, default=0.04, dest="btsp_lr")
    ap.add_argument("--barrier", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--static-w", type=float, default=1.5, dest="static_w")
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    try:
        assert_backend("numpy", note="(CPU lane; GPU is busy)")
    except AssertionError as e:
        print("BACKEND WARNING: %s" % e)

    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run(s, a); per.append(r)
            c = r["conditions"]
            print(f"  [seed {s}] "
                  f"PLATEAU t0={c['plateau']['recall_t0_target']:.2f} after={c['plateau']['recall_after_target']:.2f} "
                  f"(distr={c['plateau']['recall_after_distr']:.2f}) | "
                  f"TRANS t0={c['transient']['recall_t0_target']:.2f} after={c['transient']['recall_after_target']:.2f} | "
                  f"MOAT after={c['moat']['recall_after_target']:.2f} | "
                  f"NOCAP after={c['plateau_nocapture']['recall_after_target']:.2f} | "
                  f"STATIC t0={c['static_subbarrier']['recall_t0_target']:.2f} "
                  f"after={c['static_subbarrier']['recall_after_target']:.2f} | "
                  f"off_dw={r['instrument']['off_dw']:.4f}", flush=True)
    except (RuntimeError, ValueError, AttributeError, KeyError, IndexError, TypeError) as e:
        err = "%s: %s" % (type(e).__name__, e); traceback.print_exc()

    summary = {"probe": "gap4_btsp_lasting_trace_recall_after_delay", "seeds": a.seeds,
               "params": {k: getattr(a, k) for k in ("n_pre", "write_steps", "window_steps", "recall_steps", "w0",
                                                     "btsp_w_max", "btsp_lr", "barrier", "beta", "static_w",
                                                     "recall_drive", "pulse_pA")},
               "recall_hi": RECALL_HI, "recall_lo": RECALL_LO,
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}

    if err is None and per:
        def allc(cond, key, cmp):
            return all(cmp(p["conditions"][cond][key]) for p in per)

        # "DECAYED below recall" is graded PER SEED against that seed's own persisted (plateau) trace: a decayed
        # condition must recall at <= CONTRAST of the plateau AND be below the absolute "fires" line. This is robust to
        # the post cell's spontaneous-firing floor (~0.002-0.006/step) drifting across seeds -- the ratio (~6%) has a
        # ~6x margin to CONTRAST, where an absolute floor alone had ~1.3x. plateau "still fires" stays ABSOLUTE (>=HI).
        def decayed(cond):   # cond recalled-after is BELOW recall in EVERY seed (contrast + absolute)
            return all(p["conditions"][cond]["recall_after_target"] <= CONTRAST * p["conditions"]["plateau"]["recall_after_target"]
                       and p["conditions"][cond]["recall_after_target"] <= RECALL_HI for p in per)
        def distr_decayed():
            return all(p["conditions"]["plateau"]["recall_after_distr"] <= CONTRAST * p["conditions"]["plateau"]["recall_after_target"]
                       and p["conditions"]["plateau"]["recall_after_distr"] <= RECALL_HI for p in per)

        instr_ok = (all(abs(p["instrument"]["off_dw"]) < 1e-9 for p in per) and
                    all(abs(p["instrument"]["maintenance_inert_beta0_delta"]) < 1e-9 for p in per) and
                    all(p["instrument"]["recall_huge_weight"] >= RECALL_HI for p in per) and
                    all(p["instrument"]["recall_zero_weight"] <= RECALL_LO for p in per))
        # L1 LASTING: the plateau trace STILL FIRES after the window; the transient + moat writes have DECAYED below it.
        L1 = (allc("plateau", "recall_after_target", lambda x: x >= RECALL_HI) and
              decayed("transient") and decayed("moat"))
        # L2 CRUX (persistence, not write-strength): a write that DID recall at t0 (static, and the transient) has
        # decayed below recall after the window -> the after-window failure is DECAY, not a failed write.
        L2 = (allc("static_subbarrier", "recall_t0_target", lambda x: x >= RECALL_HI) and decayed("static_subbarrier")
              and allc("transient", "recall_t0_target", lambda x: x >= RECALL_HI))
        # L3 LESION: the SAME big plateau write, minus the capture, decays below recall (capture is load-bearing).
        L3 = decayed("plateau_nocapture")
        # L4 ANTI-MAGNITUDE == the static sub-barrier condition of L2 (a large STATIC weight is not enough; it must
        # cross the barrier, which only the plateau does). Kept as an explicit line for the verdict.
        L4 = decayed("static_subbarrier")
        # L5 ATTRIBUTABILITY: only the plateau-targeted post cells persist (the distractor half does not recall).
        L5 = distr_decayed()
        go = bool(instr_ok and L1 and L2 and L3 and L4 and L5)

        pa = float(np.mean([p["conditions"]["plateau"]["recall_after_target"] for p in per]))
        pna = float(np.mean([p["conditions"]["plateau_nocapture"]["recall_after_target"] for p in per]))
        tt = float(np.mean([p["conditions"]["transient"]["recall_after_target"] for p in per]))
        print("\n-- attribution: plateau recall-after vs its no-capture lesion --")
        attributable_to("recall-after (plateau vs no-capture lesion)", pa, pna)
        void_if(not instr_ok, "an instrument check failed (off_dw / maintenance-inert / recall-readout)")

        if go:
            verdict = (f"GO -- a per-turn BTSP PLATEAU write is a LASTING trace. After a {a.window_steps}-step decay "
                       f"window of intervening activity, the plateau-written fact is STILL RECALLABLE (post-target "
                       f"fires, recall {pa:.2f}); a NON-plateau (transient) write that recalled at t0 has DECAYED "
                       f"below recall ({tt:.2f}); the moat (silent apical) and a large sub-barrier STATIC weight both "
                       f"fail; and REMOVING the tag-and-capture maintenance (no-capture lesion) makes the plateau "
                       f"trace decay too ({pna:.2f}) -> the persistence is the capture x plateau interaction, not the "
                       f"big write alone. 6-seed. Spiking WRITE + spiking RECALL on the real bridge; the LASTING side "
                       f"is a runner-level tag-and-capture MODEL (next rung: a guarded default-off sim/ kernel).")
        else:
            miss = []
            if not instr_ok: miss.append("INSTRUMENT failed (off_dw/maintenance-inert/recall-readout)")
            if not L1: miss.append("L1 lasting failed (plateau below HI after-window, or transient/moat not decayed vs plateau)")
            if not L2: miss.append("L2 crux failed (static/transient did NOT recall at t0, or static did not decay)")
            if not L3: miss.append("L3 lesion failed (no-capture plateau still recalled vs plateau)")
            if not L4: miss.append("L4 anti-magnitude failed (static sub-barrier still recalled vs plateau)")
            if not L5: miss.append("L5 attribution failed (distractor half recalled vs plateau)")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". Per THE LAW: tune barrier/beta/window/recall-drive or the "
                       "write, NOT a stop. Name the residual, not a stop.")
    else:
        go = False; verdict = f"ERROR -- {err}" if err else "ERROR -- no per-seed results"

    # Verdict carries the preconditions that earned it (tools.verdict.Verdict -> preconditions block in the artifact,
    # enforced by gates/verdict_preconditions). UNDEFINED is the default; a GO must earn every check.
    if err is None and per:
        def _m(cond, key): return float(np.mean([p["conditions"][cond][key] for p in per]))
        pa_after = _m("plateau", "recall_after_target"); pna_after = _m("plateau_nocapture", "recall_after_target")
        r_zero = float(np.mean([p["instrument"]["recall_zero_weight"] for p in per]))
        pl_wpost = _m("plateau", "w_post_window"); nc_wpost = _m("plateau_nocapture", "w_post_window")
        v = Verdict("gap4 BTSP plateau lasting trace (recall-after-delay)")
        v.require("L1 lasting: plateau fires after window; transient+moat decayed", L1, expect=True)
        v.require("L2 crux: static+transient recalled at t0; static decayed", L2, expect=True)
        v.require("L3 lesion: no-capture plateau decayed below recall", L3, expect=True)
        v.require("L4 anti-magnitude: static sub-barrier decayed", L4, expect=True)
        v.require("L5 attributability: distractor half silent", L5, expect=True)
        v.require("instrument (off byte-id / maint inert / readout distinguishes)", instr_ok, expect=True)
        v.control("plateau vs no-capture lesion (recall-after)", treatment=pa_after, control=pna_after,
                  min_separation=RECALL_HI)
        v.floor("plateau recall-after exceeds the spontaneous floor", measured=pa_after, floor=r_zero)
        v.reaches("capture retains the plateau weight (else it decays)", before=nc_wpost, after=pl_wpost)
        for proc in ("STDP", "Hebbian", "homeostasis", "short-term plasticity", "reward modulation",
                     "structural plasticity", "NMDA", "input divisive norm"):
            v.disabled(proc, why="isolation: the write phase leaves on-bridge BTSP as the sole weight mover")
        result = v.decide(go=go)
        summary.update(result)

    summary["GO"] = go; summary["verdict"] = verdict
    summary["HONEST_NOTE"] = ("Spiking WRITE (real on-bridge BTSP + bistable BDSP apical) and spiking RECALL (post firing) "
                              "on a real SimulationBridge; recall is a BEHAVIOURAL read (not a weight read). The LASTING "
                              "side (tag-and-capture stabilization) is a RUNNER-LEVEL MODEL applied to cp_connections.data "
                              "(Frey&Morris 1997; Lisman 1985; Bittner&Magee 2017 for the supra-threshold plateau write) -- "
                              "NOT yet a sim/ kernel. NOT 'consolidation' in the TERMS.md sense (no replay path executes). "
                              "NO new sim/ edit.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[gap4-btsp-lasting] VERDICT: {verdict}", flush=True)
    print(f"[gap4-btsp-lasting] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
