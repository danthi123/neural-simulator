"""Round-3 scratch PROTOTYPE of the step-0a(i) calibration (read-only on the repo; numpy backend; plasticity off).

NOT the step-0a record: the organ's circuit constants (FS weights, presence current) are not yet frozen, so these numbers
are a forecast that tells the packet which way the per-target thresholds can move. The step-0a module
(research/runners/_deixis_calibration.py) re-measures on the frozen circuit.

Two targets, measured separately (finding: one FS-free curve was applied to both):
  P  : TOK (12, RS, 1500 pA) -> P pool (12, RS); P's only hearing-mode inputs are TOK and C (dx_conv is closed at
       hearing), so the bare curve IS P's real wiring.
  E  : TOK (12) -> E_target (12, RS); five entity assemblies (target + 4 present) all project onto one shared FS pool
       (16, IZH2007_FS_CORTICAL_INTERNEURON preset) at W_EF, which inhibits every E at W_FE (GABA_A).
       Conditions: 'quiet' (no presence drive), 'presence' (every present entity incl. the target at I_PRES, the
       exposure presence rate's current), 'release' (quiet + H lines on the target's C cells at their subthreshold
       drive and the C -> E release projection open: C must stay silent, so this equals 'quiet' unless C leaks).
Heterogeneity: enable_parameter_heterogeneity=True with preset_centred_parameter_heterogeneity ON (the organ's
setting) and OFF (the 10-20 hybrid, reported). The FS pool's measured median a/b/d/C are printed per setting.
Seeds 61-70: outside the census seeds 1-15, outside the gate seeds.
Criterion (unchanged from 2.6): theta_fire(target) = the smallest grid w at which every target neuron fires at least
once in a 40-step token on every seed in every condition; theta_silent(target) = the largest grid w at which the
target emits at most one spike on every seed in every condition.
"""
import os, sys, json
os.environ.setdefault("SIM_BACKEND", "numpy")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np, logging
logging.disable(logging.CRITICAL)
from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.bridge import SimulationBridge
from sim.regions import BrainRegion

RS, FS = "IZH2007_RS_CORTICAL_PYRAMIDAL", "IZH2007_FS_CORTICAL_INTERNEURON"
NT = 12; NE = 12; NFS = 16; N_PRESENT = 5; NC = 4
DRIVE = 1500.0
W_EF, W_FE = float(os.environ.get("W_EF", 20.0)), float(os.environ.get("W_FE", 20.0))
I_PRES = float(os.environ.get("I_PRES", 200.0))
W_HC = float(os.environ.get("W_HC", 3.0)); W_CE = float(os.environ.get("W_CE", 20.0))
GRID = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20)


def _dense(pre, post, w, gate=None, receptor=None):
    pre = np.asarray(pre, np.int64); post = np.asarray(post, np.int64)
    a = np.repeat(pre, post.size); b = np.tile(post, pre.size)
    d = {"pre_indices": a, "post_indices": b, "initial_weights": np.full(a.size, float(w), np.float32),
         "plastic": False, "plasticity_gate": "dx_fixed", "conn_type": "x", "count": int(a.size)}
    if gate: d["transmission_gate"] = gate
    if receptor: d["receptor"] = receptor
    return d


def build(seed, target, preset_centred):
    regs = [BrainRegion(name="tok", n_neurons=NT, exc_fraction=1.0, internal_density=0.0, izh_neuron_type=RS,
                        enable_nmda=False, plastic_internal=False)]
    if target == "P":
        regs.append(BrainRegion(name="p", n_neurons=NE, exc_fraction=1.0, internal_density=0.0, izh_neuron_type=RS,
                                enable_nmda=False, plastic_internal=False))
    else:
        regs.append(BrainRegion(name="e", n_neurons=NE * N_PRESENT, exc_fraction=1.0, internal_density=0.0,
                                izh_neuron_type=RS, enable_nmda=False, plastic_internal=False))
        regs.append(BrainRegion(name="fs", n_neurons=NFS, exc_fraction=0.0, internal_density=0.0, izh_neuron_type=FS,
                                enable_nmda=False, plastic_internal=False))
        regs.append(BrainRegion(name="h", n_neurons=NC, exc_fraction=1.0, internal_density=0.0, izh_neuron_type=RS,
                                enable_nmda=False, plastic_internal=False))
        regs.append(BrainRegion(name="c", n_neurons=NC, exc_fraction=1.0, internal_density=0.0, izh_neuron_type=RS,
                                enable_nmda=False, plastic_internal=False))
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True; cfg.enable_neuromodulator_subsystem = False; cfg.enable_nmda = False
    cfg.dt_ms = 1.0; cfg.seed = int(seed); cfg.connections_per_neuron = 0; cfg.num_traits = 1
    for f in ("enable_stdp", "enable_reward_modulation", "enable_homeostasis", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_watts_strogatz", "enable_ou_process", "enable_hebbian_learning"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = True
    cfg.preset_centred_parameter_heterogeneity = bool(preset_centred)
    cfg.brain_regions = regs; cfg.region_pathways = []
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    idx = {k: np.asarray(v, np.int64) for k, v in b.region_manager.region_indices_dict().items()}
    return b, idx


def trial(seed, target, w, cond, preset_centred, steps=40):
    b, idx = build(seed, target, preset_centred)
    tok = idx["tok"]
    if target == "P":
        tgt = idx["p"]; plan = {"route": _dense(tok, tgt, w, gate="dx_route")}; inh = np.array([], np.int64)
    else:
        e = idx["e"]; tgt = e[:NE]; fs = idx["fs"]
        plan = {"route": _dense(tok, tgt, w, gate="dx_route"), "ef": _dense(e, fs, W_EF), "fe": _dense(fs, e, W_FE, receptor="gaba_a"),
                "hc": _dense(idx["h"], idx["c"], W_HC), "ce": _dense(idx["c"], tgt, W_CE, gate="dx_release_e")}
        inh = fs
    b.inject_explicit_wiring(plan, output_inhibitory_indices=inh)
    b.set_transmission_gate("dx_route", 1.0)
    if target == "E": b.set_transmission_gate("dx_release_e", 1.0)
    b.cp_external_input_current[:] = 0.0
    for _ in range(60): b._run_one_simulation_step()
    n = b.core_config.num_neurons
    cur = np.zeros(n, np.float32); cur[tok] = DRIVE
    if target == "E" and cond == "presence": cur[idx["e"]] = I_PRES
    if target == "E" and cond == "release": cur[idx["h"]] = DRIVE
    b.cp_external_input_current[:] = cur
    sp = 0; fired = np.zeros(tgt.size, bool); csp = 0; fssp = 0; base = 0
    for _ in range(steps):
        b._run_one_simulation_step(); f = np.asarray(b.cp_firing_states).astype(bool)
        sp += int(f[tgt].sum()); fired |= f[tgt]
        if target == "E": csp += int(f[idx["c"]].sum()); fssp += int(f[idx["fs"]].sum())
    return sp, float(fired.mean()), csp, fssp


def fs_params(seed, preset_centred):
    b, idx = build(seed, "E", preset_centred)
    out = {}
    for nm in ("cp_izh_a", "cp_izh_b", "cp_izh_d", "cp_izh_C"):
        a = getattr(b, nm, None)
        if a is not None:
            a = np.asarray(a); out[nm] = (float(np.median(a[idx["fs"]])), float(np.median(a[idx["e"]])))
    return out


if __name__ == "__main__":
    seeds = list(range(61, 71)); res = {"W_EF": W_EF, "W_FE": W_FE, "I_PRES": I_PRES, "W_HC": W_HC, "W_CE": W_CE, "seeds": seeds}
    for pc in (True, False):
        print("== preset_centred_parameter_heterogeneity =", pc, "| FS / E median a,b,d,C (seed 61):", fs_params(61, pc), flush=True)
        for target, conds in (("P", ("bare",)), ("E", ("quiet", "presence", "release"))):
            curve = {}
            for w in GRID:
                rows = {c: [trial(s, target, w, c, pc) for s in seeds] for c in conds}
                curve[w] = rows
                msg = " | ".join("%s spk min %d med %d max %d frac-min %.2f%s" % (
                    c, min(r[0] for r in rr), int(np.median([r[0] for r in rr])), max(r[0] for r in rr), min(r[1] for r in rr),
                    (" C %d FS %d" % (max(r[2] for r in rr), int(np.median([r[3] for r in rr])))) if target == "E" else "")
                    for c, rr in rows.items())
                print("  %s w=%4.1f  %s" % (target, w, msg), flush=True)
            # thresholds: envelope over the HEARING conditions only (the routes transmit only at hearing); the
            # 'presence' condition (PART rendered, exposure only, routes closed) is reported as the increment over w=0
            hear = [c for c in conds if c != "presence"]
            fire = [w for w in GRID if all(min(r[1] for r in curve[w][c]) >= 1.0 for c in hear)]
            silent = [w for w in GRID if all(max(r[0] for r in curve[w][c]) <= 1 for c in hear)]
            if "presence" in conds:
                inc = {w: max(curve[w]["presence"][i][0] - curve[0]["presence"][i][0] for i in range(len(seeds))) for w in GRID}
                print("  presence (reported): max increment over w=0 per w:", inc, flush=True)
            th_f = min(fire) if fire else None; th_s = max([w for w in silent if th_f is None or w < th_f], default=None)
            print("  => %s theta_fire %s theta_silent %s" % (target, th_f, th_s), flush=True)
            res["%s_pc%d" % (target, int(pc))] = dict(theta_fire=th_f, theta_silent=th_s,
                                                      curve={str(w): {c: v for c, v in d.items()} for w, d in curve.items()})
    tag = "calib_proto_WEF%g_WFE%g_IP%g.json" % (W_EF, W_FE, I_PRES)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), tag), "w"))
