"""_deixis_hearing_core — step-0a module 2 (PACKET Appendix C, frozen).

The hearing-circuit builder: the Appendix C populations and mode gates, built
on the brain-region framework so the cell presets apply (RS for every
excitatory population, IZH2007_FS_CORTICAL_INTERNEURON for the shared FS pool
and, in the CAUSAL+WTA variant, the per-relation inhibitory pools).

Frozen by Appendix C:
  TOK[w]  12 per toy form: the replica's 30 forms + the 5 markers + LFT,
          plus 12 spares (444 neurons); the homograph shares N_SELF's
          assembly (no extra assembly).
  E[e]    12 per entity x 10 entities = 120; FS = 16, one shared pool.
  H_k[e]  4 per relation-entity = 200; C_k[e] 4 = 200; P_k 12 x 5 = 60.
  FS constants: E->FS 8 dense, FS->E 30 GABA_A — frozen (ruling binding 2).
  H_k[e]->C_k[e] at W_HC (ungated); E[e]->C_k[e] at W_EC (transmission gate
  dx_teach); P_k->C_k[e] at W_PC (gate dx_release); C_k[e]->P_k at 20 (gate
  dx_conv); C_k[e]->E[e] at 20 (gate dx_release, hearing only).
  Learned blocks: TOK->P (plasticity gate dx_ptr, transmission gate dx_ptr_tx)
  and TOK->E (dx_name / dx_name_tx), w_init 0, clip [0, 20] via
  hebbian_min_weight/hebbian_max_weight.
  CAUSAL+WTA variant (wta=True): per-P inhibitory pool NI 8 (word-role
  organ's W_RI 200, W_IR 80; P_k->pw_k 200, pw_k->P_j 80 GABA_A, j != k).
  The pool cells use the FS interneuron preset — calib_proto's
  inhibitory-region convention; Appendix C freezes NI and the two weights.

Appendix C config flags: population-scale, centered, seed-locked
heterogeneity ON; enable_stdp / reward modulation / homeostasis /
short-term and structural plasticity / Watts-Strogatz / OU off; enable_nmda
off; enforce_plastic_mask_in_hebbian=True; no hebbian_mean_subtract;
propagation_strength the engine default 0.05. enable_hebbian_learning stays
at the engine default (Appendix C does not list it among the off flags and
attaches enforce_plastic_mask_in_hebbian=True to it); weights are immutable
outside EXPOSURE because the mode gates close dx_ptr/dx_name — the gate
readback STOP-PLAST watches.

Sign semantics (probe-verified, no sim/ edit): the engine derives each
synapse's conn_type from the pre/post inhibitory flags, not from the plan's
receptor field — receptor only routes conductance kinetics. Excitatory
projections therefore carry no receptor key (absent == ampa-effective);
inhibitory ones carry receptor="gaba_a" exactly as calib_proto does, and
output_inhibitory_indices MUST include the FS (+WTA pool) neurons so their
outputs are inhibitory.

Modes (Appendix C):
  EXPOSURE  transmission dx_teach 1, dx_conv 1, dx_release 0, dx_ptr_tx 0,
            dx_name_tx 0; plasticity dx_ptr 1, dx_name 1.
  HEARING   transmission 0, 0, 1, 1, 1; plasticity 0, 0.

CLI: --selftest builds EXPOSURE + HEARING (+ the wta variant) at seed 61,
asserts population sizes, gate tables and gate-registered synapse counts,
runs 50 quiet steps, and prints one JSON record. The packet runs it inside
pytest on ONE pool node (host class: CPU pool).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

import numpy as np  # noqa: E402

from sim.bridge import SimulationBridge  # noqa: E402
from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig  # noqa: E402
from sim.regions import BrainRegion  # noqa: E402

RS = "IZH2007_RS_CORTICAL_PYRAMIDAL"
FS_PRESET = "IZH2007_FS_CORTICAL_INTERNEURON"

# The replica's 30 forms (fix/replica3.py::TOK): 10 fixed + GIVTOK 5 +
# N_* persons 7 + T_* objects 3 + 5 pool markers; then the 5 markers and LFT.
GIVTOK = ["ANA", "DEM", "POSS_G", "ORD1", "ORD2"]
FIXED = ["SHS", "SHA", "POSS_S", "POSS_A", "PLUR", "OOI1", "PREV", "CN", "IDIO", "FILL"]
PERS = ["V1", "V2", "V3", "V4", "V5", "B1", "SELF"]
OBJ = ["O1", "O2", "O3"]
POOLS_REL = ["SPK", "ADR", "GIV", "LEFT", "NEWEST"]
MARKERS = ["ART", "ASP", "IND", "AGR", "DEF"]
FORMS = FIXED + GIVTOK + ["N_" + p for p in PERS] + ["T_" + o for o in OBJ] \
    + ["P%d" % i for i in range(1, 6)] + MARKERS + ["LFT"]  # 36 toy forms; P1..P5 per replica vocab
ENTITIES = PERS + OBJ  # 10

N_UNIT = 12          # neurons per form / per entity / per P pool
N_SPARE = 12         # spare TOK neurons
N_FS = 16            # shared FS pool
N_H_UNIT = 4         # per relation-entity
N_P_UNIT = 12
N_WTA = 8            # NI, per-P inhibitory pool (CAUSAL+WTA)
W_EF_FROZEN = 8.0    # Appendix C, ruling binding 2: never tuned
W_FE_FROZEN = 30.0
W_CP = 20.0          # C_k[e] -> P_k
W_CE = 20.0          # C_k[e] -> E[e]
W_RI = 200.0         # P_k -> pw_k   (word-role organ)
W_IR = 80.0          # pw_k -> P_j   (word-role organ)
LEARN_CLIP = (0.0, 20.0)

N_TOK = len(FORMS) * N_UNIT + N_SPARE
N_E = len(ENTITIES) * N_UNIT
N_HC = len(POOLS_REL) * len(ENTITIES) * N_H_UNIT
N_P = len(POOLS_REL) * N_P_UNIT

POPULATION_SIZES = {"tok": N_TOK, "e": N_E, "fs": N_FS, "h": N_HC,
                    "c": N_HC, "p": N_P}

TRANSMISSION_GATES = ("dx_teach", "dx_conv", "dx_release", "dx_ptr_tx", "dx_name_tx")
PLASTICITY_GATES = ("dx_ptr", "dx_name")
MODES = {
    "EXPOSURE": {"dx_teach": 1.0, "dx_conv": 1.0, "dx_release": 0.0,
                 "dx_ptr_tx": 0.0, "dx_name_tx": 0.0,
                 "dx_ptr": 1.0, "dx_name": 1.0},
    "HEARING": {"dx_teach": 0.0, "dx_conv": 0.0, "dx_release": 1.0,
                "dx_ptr_tx": 1.0, "dx_name_tx": 1.0,
                "dx_ptr": 0.0, "dx_name": 0.0},
}


def _dense(pre, post, w, transmission_gate=None, plasticity_gate="dx_fixed",
           plastic=False, receptor=None):
    """calib_proto's _dense, with the engine's literal plan keys
    (transmission_gate / plasticity_gate; probe-verified sign semantics)."""
    pre = np.asarray(pre, np.int64)
    post = np.asarray(post, np.int64)
    a = np.repeat(pre, post.size)
    b = np.tile(post, pre.size)
    d = {"pre_indices": a, "post_indices": b,
         "initial_weights": np.full(a.size, float(w), np.float32),
         "plastic": bool(plastic), "plasticity_gate": plasticity_gate,
         "conn_type": "x", "count": int(a.size)}
    if transmission_gate:
        d["transmission_gate"] = transmission_gate
    if receptor:
        d["receptor"] = receptor
    return d


def form_slices():
    """form index -> (tok slice for its 12 cells)."""
    return {f: slice(i * N_UNIT, (i + 1) * N_UNIT)
            for i, f in enumerate(FORMS)}


def entity_slices():
    return {e: slice(i * N_UNIT, (i + 1) * N_UNIT)
            for i, e in enumerate(ENTITIES)}


def hc_slices():
    """(relation, entity) -> h / c block slices (same layout in both regions)."""
    out = {}
    n = 0
    for k in range(len(POOLS_REL)):
        for e in range(len(ENTITIES)):
            out[(k, e)] = slice(n, n + N_H_UNIT)
            n += N_H_UNIT
    return out


def p_slices():
    return {k: slice(k * N_P_UNIT, (k + 1) * N_P_UNIT)
            for k in range(len(POOLS_REL))}


def wiring_plan(w_hc, w_ec, w_pc, wta=False, idx=None):
    """The frozen Appendix C plan (weights in engine units).

    `idx` is region_indices_dict() and is REQUIRED: the engine's
    inject_explicit_wiring builds a COO of shape (n, n) over GLOBAL
    neuron indices, so region-local indices would alias every region's
    edges onto the low global ids (learn blocks colliding with tok/e,
    COO->CSR dedup collapsing 94,160 planned edges to 55,424, and the
    plasticity-gate maps pointing past cp_connections.data).  The gate
    maps are rebuilt from this plan in (pre, post) sort order, so the
    plan must be the exact global edge set.
    """
    if idx is None:
        raise ValueError(
            "wiring_plan requires idx=region_indices_dict(): the engine "
            "injects global neuron indices (COO shape (n, n)); local "
            "indices collide inside sum_duplicates()")
    tok_all = np.asarray(idx["tok"], np.int64)
    e_all = np.asarray(idx["e"], np.int64)
    fs_all = np.asarray(idx["fs"], np.int64)
    h_all = np.asarray(idx["h"], np.int64)
    c_all = np.asarray(idx["c"], np.int64)
    p_all = np.asarray(idx["p"], np.int64)
    plan = {
        "ef": _dense(e_all, fs_all, W_EF_FROZEN),
        "fe": _dense(fs_all, e_all, W_FE_FROZEN, receptor="gaba_a"),
    }
    hc = hc_slices()
    es = entity_slices()
    ps = p_slices()
    for k in range(len(POOLS_REL)):
        for ei in range(len(ENTITIES)):
            cb = hc[(k, ei)]
            eb = es[ENTITIES[ei]]
            e_idx = e_all[eb]
            plan["hc_%d_%d" % (k, ei)] = _dense(h_all[cb], c_all[cb], w_hc)
            plan["ec_%d_%d" % (k, ei)] = _dense(e_idx, c_all[cb], w_ec,
                                                transmission_gate="dx_teach")
            plan["pc_%d_%d" % (k, ei)] = _dense(p_all[ps[k]], c_all[cb], w_pc,
                                                transmission_gate="dx_release")
            plan["cp_%d_%d" % (k, ei)] = _dense(c_all[cb], p_all[ps[k]], W_CP,
                                                transmission_gate="dx_conv")
            plan["ce_%d_%d" % (k, ei)] = _dense(c_all[cb], e_idx, W_CE,
                                                transmission_gate="dx_release")
    # Learned blocks: TOK -> P (dx_ptr) and TOK -> E (dx_name), w_init 0.
    plan["learn_p"] = _dense(tok_all, p_all, 0.0, plastic=True,
                             plasticity_gate="dx_ptr",
                             transmission_gate="dx_ptr_tx")
    plan["learn_e"] = _dense(tok_all, e_all, 0.0, plastic=True,
                             plasticity_gate="dx_name",
                             transmission_gate="dx_name_tx")
    if wta:
        pw_all = np.asarray(idx["pw"], np.int64)
        for k in range(len(POOLS_REL)):
            pwk = slice(k * N_WTA, (k + 1) * N_WTA)
            plan["ri_%d" % k] = _dense(p_all[ps[k]], pw_all[pwk], W_RI)
            others = np.concatenate(
                [p_all[ps[j]] for j in range(len(POOLS_REL)) if j != k])
            plan["ir_%d" % k] = _dense(pw_all[pwk], others, W_IR,
                                       receptor="gaba_a")
    return plan


def regions(wta=False):
    regs = [
        BrainRegion(name="tok", n_neurons=N_TOK, exc_fraction=1.0,
                    internal_density=0.0, izh_neuron_type=RS,
                    enable_nmda=False, plastic_internal=False),
        BrainRegion(name="e", n_neurons=N_E, exc_fraction=1.0,
                    internal_density=0.0, izh_neuron_type=RS,
                    enable_nmda=False, plastic_internal=False),
        BrainRegion(name="fs", n_neurons=N_FS, exc_fraction=0.0,
                    internal_density=0.0, izh_neuron_type=FS_PRESET,
                    enable_nmda=False, plastic_internal=False),
        BrainRegion(name="h", n_neurons=N_HC, exc_fraction=1.0,
                    internal_density=0.0, izh_neuron_type=RS,
                    enable_nmda=False, plastic_internal=False),
        BrainRegion(name="c", n_neurons=N_HC, exc_fraction=1.0,
                    internal_density=0.0, izh_neuron_type=RS,
                    enable_nmda=False, plastic_internal=False),
        BrainRegion(name="p", n_neurons=N_P, exc_fraction=1.0,
                    internal_density=0.0, izh_neuron_type=RS,
                    enable_nmda=False, plastic_internal=False),
    ]
    if wta:
        regs.append(BrainRegion(name="pw", n_neurons=len(POOLS_REL) * N_WTA,
                                exc_fraction=0.0, internal_density=0.0,
                                izh_neuron_type=FS_PRESET, enable_nmda=False,
                                plastic_internal=False))
    return regs


def config(seed):
    """Appendix C's frozen configuration."""
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.enable_neuromodulator_subsystem = False
    cfg.enable_nmda = False
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    for f in ("enable_stdp", "enable_reward_modulation", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity",
              "enable_watts_strogatz", "enable_ou_process"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = True
    cfg.preset_centred_parameter_heterogeneity = True
    cfg.enforce_plastic_mask_in_hebbian = True
    cfg.hebbian_min_weight = LEARN_CLIP[0]
    cfg.hebbian_max_weight = LEARN_CLIP[1]
    # hebbian_mean_subtract stays 0.0 ("no hebbian_mean_subtract");
    # propagation_strength stays the engine default 0.05;
    # enable_hebbian_learning stays the engine default (module docstring).
    return cfg


def set_mode(b, mode):
    """Apply one of the two frozen mode tables; returns the gate readback."""
    table = MODES[mode]
    for g in TRANSMISSION_GATES:
        b.set_transmission_gate(g, float(table[g]))
    for g in PLASTICITY_GATES:
        b.set_plasticity_gate(g, float(table[g]))
    return readback(b, mode)


def readback(b, mode=None):
    """Gate values as the engine holds them: transmission gates live in a
    private dict (the bridge exposes no public getter and no sim/ edit is
    allowed), plasticity gates through the public getter."""
    rec = {g: float(b._transmission_gate_values[g]) for g in TRANSMISSION_GATES}
    rec.update({g: float(b.get_plasticity_gate_value(g)) for g in PLASTICITY_GATES})
    if mode is not None:
        rec["_mode"] = mode
    return rec


def build(seed=61, mode="EXPOSURE", w_hc=6.0, w_ec=20.0, w_pc=20.0, wta=False):
    """Build the Appendix C circuit, install the plan, apply `mode`."""
    regs = regions(wta=wta)
    cfg = config(seed)
    cfg.brain_regions = regs
    cfg.region_pathways = []
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    idx = {k: np.asarray(v, np.int64)
           for k, v in b.region_manager.region_indices_dict().items()}
    inh = idx["fs"]
    if wta:
        inh = np.concatenate([inh, idx["pw"]])
    b.inject_explicit_wiring(
        wiring_plan(w_hc, w_ec, w_pc, wta=wta, idx=idx),
        output_inhibitory_indices=inh)
    b.cp_external_input_current[:] = 0.0
    gates = set_mode(b, mode)
    return b, idx, gates


def selftest(seed=61):
    rec = {"populations": dict(POPULATION_SIZES)}
    expected_trans = {
        "dx_teach": len(POOLS_REL) * len(ENTITIES) * (N_UNIT * N_H_UNIT),
        "dx_conv": len(POOLS_REL) * len(ENTITIES) * (N_H_UNIT * N_P_UNIT),
        "dx_ptr_tx": N_TOK * N_P,
        "dx_name_tx": N_TOK * N_E,
    }
    expected_trans["dx_release"] = 2 * len(POOLS_REL) * len(ENTITIES) * (
        N_P_UNIT * N_H_UNIT)
    b, idx, gates = build(seed=seed, mode="EXPOSURE")
    assert int(idx["tok"].size) == N_TOK and int(idx["e"].size) == N_E
    assert int(idx["fs"].size) == N_FS and int(idx["h"].size) == N_HC
    assert int(idx["c"].size) == N_HC and int(idx["p"].size) == N_P
    for g, want in expected_trans.items():
        got = len(b._transmission_gate_to_synapses[g])
        assert got == want, "dx %s: %d != %d" % (g, got, want)
    expected_plastic = {"dx_ptr": N_TOK * N_P, "dx_name": N_TOK * N_E}
    for g in PLASTICITY_GATES:
        assert b.plasticity_gate_synapse_count(g) == expected_plastic[g]
    # installed matrix must carry EVERY plan edge (global indices; local
    # ones alias and collapse inside COO->CSR dedup), and every gate map
    # must index inside cp_connections.data.
    want_nnz = (sum(expected_trans.values())
                + len(POOLS_REL) * len(ENTITIES) * N_H_UNIT * N_H_UNIT
                + N_E * N_FS + N_FS * N_E)
    got_nnz = int(b.cp_connections.nnz)
    assert got_nnz == want_nnz, "nnz %d != plan total %d" % (
        got_nnz, want_nnz)
    for g in PLASTICITY_GATES:
        mx = int(np.max(np.asarray(
            b._plasticity_gate_to_synapses[g])))
        assert mx < got_nnz, "gate %s index %d >= nnz %d" % (
            g, mx, got_nnz)
    for g in TRANSMISSION_GATES:
        mx = int(np.max(np.asarray(
            b._transmission_gate_to_synapses[g])))
        assert mx < got_nnz, "gate %s index %d >= nnz %d" % (
            g, mx, got_nnz)
    for g, v in MODES["EXPOSURE"].items():
        want = v
        got = (b._transmission_gate_values[g]
               if g in b._transmission_gate_values
               else b.get_plasticity_gate_value(g))
        assert float(got) == want, "EXPOSURE %s: %s != %s" % (g, got, want)
    for _ in range(50):
        b._run_one_simulation_step()
    h2, _, _ = build(seed=seed, mode="HEARING", wta=True)
    for g, want in MODES["HEARING"].items():
        got = (h2._transmission_gate_values[g]
               if g in h2._transmission_gate_values
               else h2.get_plasticity_gate_value(g))
        assert float(got) == want, "HEARING %s: %s != %s" % (g, got, want)
    assert len(h2.region_manager.region_indices_dict()["pw"]) == \
        len(POOLS_REL) * N_WTA
    rec.update({"ok": True, "seed": seed, "mode_tables": MODES,
                "gates_exposure": readback(b, "EXPOSURE"),
                "gates_hearing": readback(h2, "HEARING"),
                "plastic_synapses": int(b.plasticity_gate_synapse_count("dx_ptr")
                                        + b.plasticity_gate_synapse_count("dx_name"))})
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="build both modes (+wta), assert tables, run 50 quiet steps")
    ap.add_argument("--seed", type=int, default=61)
    args = ap.parse_args(argv)
    if args.selftest:
        print(json.dumps(selftest(seed=args.seed), indent=1, sort_keys=True))
        return 0
    ap.error("no action: pass --selftest")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
