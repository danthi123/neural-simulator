"""Gap #2 spiking slot binder — BUILD STEP 2b/2c: role-cued RETRIEVAL + the full multi-bind recovery test.

2026-07-17, per `2026-07-17-keystone-slot-binder-research-gate.md` #1. Composes the verified pieces:
  step 1 (GO): distinct role-drives -> distinct competitive slots (EMERGE-41 pooler).
  step 2a (GO): P NMDA-recurrent slots COEXIST (hold at zero input; no-recurrence collapses).
This runner adds the DECODE + the multi-bind recovery test:

  ONE bridge: `slot` region (K pools, NMDA-recurrent self-excitation -> HOLD) + shared `fs` (sparsity)
              + `filler` region (KF filler-output pools) + a PLASTIC slot->filler pathway (Hebbian, `enable_hebbian`).
  ROLE -> SLOT: role r drives slot pool r (distinct roles -> distinct slots, step 1). SVO fact = P roles.
  STORE a fact: for each (role r, filler f): drive slot r + TEACH filler pool f (co-activation) -> the
                slot_r -> filler_f synapses potentiate (Hebbian); the slot HOLDS (NMDA).
  RETRIEVE role r: drive slot r (the role cue) -> slot fires -> the potentiated slot_r->filler pathway
                   drives filler_f's pool -> argmax filler-pool rate = the decoded filler.

  SLOT-SEPARATED (each role -> its own slot) vs SHARED (all roles -> ONE slot -> fillers superpose = the
  write-rule ~2 cap). GO: slot-separated recovers a P>=3 fact >= 0.80 where SHARED caps ~2.
  ANTI-CHEATS: (1) SHARED/lesion-the-competition -> collapse toward ~2 (proves slot-separation is load-bearing);
  (2) no-NMDA-recurrence -> hold collapses (the slots don't persist); (3) permuted-role -> chance.

CPU/numpy; reuse build_persistent_slot for the NMDA slot substrate; a filler region + Hebbian slot->filler added.
"""
import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def build_binder_bridge(seed, K, KF, n_word=20, n_fs=24, n_fill=20, recur=25.0, fs_to_exc=10.0, nmda=True):
    """K NMDA-recurrent slot pools + shared FS + KF filler pools + a PLASTIC slot->filler pathway."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    regions = []
    for k in range(K):
        regions.append(BrainRegion(name=f"w{k}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0,
                                   exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False))
    regions.append(BrainRegion(name="fs", n_neurons=n_fs, exc_fraction=0.0, internal_density=0.0,
                               exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False))
    for f in range(KF):
        regions.append(BrainRegion(name=f"f{f}", n_neurons=n_fill, exc_fraction=1.0, internal_density=0.0,
                                   exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False))
    pathways = []
    for k in range(K):
        # NMDA self-excitation (the HOLD): w_k -> w_k via slow NMDA (Wang mechanism), like build_persistent_slot
        pathways.append(RegionPathway(from_region=f"w{k}", to_region=f"w{k}", density=1.0, weight_mean=recur,
                                      weight_jitter=0.0, plastic=False, exc_receptor="nmda_slow"))
        pathways.append(RegionPathway(from_region=f"w{k}", to_region="fs", density=1.0, weight_mean=1.4,
                                      weight_jitter=0.0, plastic=False))
        pathways.append(RegionPathway(from_region="fs", to_region=f"w{k}", density=1.0, weight_mean=fs_to_exc,
                                      weight_jitter=0.0, plastic=False))
        # PLASTIC slot -> ALL filler pools (Hebbian writes the association at store time; zero-init). PER-SLOT gate so a
        # bind's teach opens ONLY its own slot's synapses -> the Hebbian DECAY of inactive synapses cannot erode the
        # OTHER slots' already-written associations (the documented Hebbian-decay gotcha, seen as the multi-bind 0.00).
        for f in range(KF):
            pathways.append(RegionPathway(from_region=f"w{k}", to_region=f"f{f}", density=1.0, weight_mean=0.0,
                                          weight_jitter=0.0, plastic=True, plasticity_gate=f"slot{k}_to_filler"))
    # NOTE: a NAIVE always-on filler-WTA (f->ffs->f inhibition) HURT (0.56->0.11) -- it suppresses the target filler,
    # esp. during teach. A tuned WTA (weaker inhibition, readout-ONLY / disabled during teach) is the fresh-focus piece.
    # The best working config is read-calibration WITHOUT the WTA (maxw~250, lr=0.05): slot-sep 0.56 > shared 0.33.
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed); cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.enable_brain_region_framework = True; cfg.brain_regions = list(regions); cfg.region_pathways = list(pathways)
    cfg.enable_nmda = bool(nmda); cfg.enable_nmda_recurrent = bool(nmda); cfg.nmda_recurrent_tau_decay_ms = 100.0
    cfg.enable_stdp = False; cfg.enable_homeostasis = False; cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False; cfg.enable_structural_plasticity = False; cfg.fast_spike_reset = True
    # read-calibration (2026-07-17): the readout weight must clear the ~0.1 held-slot rate to fire the filler robustly.
    # maxw~250/lr=0.05 gives slot-sep 0.56 > shared 0.33 on spikes (directional GO); maxw=30 gave 0.00 (too weak).
    cfg.enable_hebbian_learning = True; cfg.hebbian_learning_rate = 0.05; cfg.hebbian_max_weight = 250.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    for k in range(K):
        b.set_plasticity_gate(f"slot{k}_to_filler", 0.0)     # all frozen until each slot's own store window
    b._K_slots = K
    return b


def _idx(b, name):
    return np.asarray(list(b.region_manager.indices(name)), int)


def run_seed(seed, K=4, KF=6, P=3, shared=False, nmda=True, permute=False, drive_steps=25, teach_steps=25,
             retr_steps=40, gain=400.0):
    from sim.backend import to_host, from_host
    rng = np.random.default_rng(seed * 131 + 5)
    b = build_binder_bridge(seed, K, KF, nmda=nmda)
    n = b.core_config.num_neurons
    slot_idx = [_idx(b, f"w{k}") for k in range(K)]
    fill_idx = [_idx(b, f"f{f}") for f in range(KF)]

    def drive(cur, steps, learn_slot=None):
        if learn_slot is not None:
            b.set_plasticity_gate(f"slot{learn_slot}_to_filler", 1.0)     # ONLY this slot's synapses learn
        dev = from_host(cur.astype(np.float64))
        for _ in range(steps):
            b.cp_external_input_current[:] = dev; b._run_one_simulation_step()
        if learn_slot is not None:
            b.set_plasticity_gate(f"slot{learn_slot}_to_filler", 0.0)

    def _reset():
        # clear v/u/firing + the slow-NMDA recurrent conductance (tau=100ms survives a v-reset) so a previously-held
        # slot does NOT co-fire with the NEXT bind's filler and write a SPURIOUS cross-association (the composition bug).
        if getattr(b, "cp_izh_c_reset", None) is not None:
            b.cp_membrane_potential_v[:] = b.cp_izh_c_reset
        else:
            b.cp_membrane_potential_v[:] = -65.0
        b.cp_recovery_variable_u[:] = 0.0
        if getattr(b, "cp_firing_states", None) is not None:
            b.cp_firing_states[:] = False
        for _a in ("cp_conductance_g_nmda_recurrent", "cp_conductance_g_e", "cp_conductance_g_i",
                   "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise"):
            _arr = getattr(b, _a, None)
            if _arr is not None:
                _arr[:] = 0.0

    # a fact = P (role r -> slot, filler f) binds. role r maps to slot (r if separated, else 0 = SHARED).
    fillers = rng.permutation(KF)[:P]
    slots_for_role = [(0 if shared else r) for r in range(P)]
    # STORE: NO reset -- the slots HOLD (NMDA) so a held slot drives its filler at retrieval. The PER-SLOT gate prevents
    # the spurious cross-write: during bind (r+1)'s teach, the already-held slot r's synapses are on slot r's FROZEN
    # gate, so they cannot co-associate with filler (r+1). (reset breaks retrieval -- the mechanism needs the hold.)
    for r in range(P):
        s = slots_for_role[r]; f = int(fillers[r])
        cur = np.zeros(n); cur[slot_idx[s]] = gain; cur[fill_idx[f]] = gain
        drive(cur, teach_steps, learn_slot=s)      # open ONLY slot s's gate
    # RETRIEVE each role: drive the ROLE's slot (the cue), read the filler-pool rates
    hits = 0
    for r in range(P):
        s = slots_for_role[(r + 1) % P] if permute else slots_for_role[r]   # permute = wrong slot cue
        cur = np.zeros(n); cur[slot_idx[s]] = gain                          # DRIVE the cued slot ABOVE the held rate
        dev = from_host(cur.astype(np.float64)); rate = np.zeros(KF)
        for _ in range(retr_steps):
            b.cp_external_input_current[:] = dev; b._run_one_simulation_step()
            fir = np.asarray(to_host(b.cp_firing_states)).astype(float)
            for f in range(KF): rate[f] += fir[fill_idx[f]].mean()
        pred = int(np.argmax(rate))
        hits += int(pred == int(fillers[r]))
    return hits / P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--P", type=int, nargs="+", default=[2, 3, 4])
    a = ap.parse_args()
    print(f"gap#2 spiking slot binder: role-cued retrieval, slot-separated vs shared (KF=6, chance {1/6:.2f})")
    rows = []
    for P in a.P:
        sep = np.mean([run_seed(s, P=P, shared=False) for s in a.seeds])
        shr = np.mean([run_seed(s, P=P, shared=True) for s in a.seeds])
        perm = np.mean([run_seed(s, P=P, shared=False, permute=True) for s in a.seeds])
        norec = np.mean([run_seed(s, P=P, shared=False, nmda=False) for s in a.seeds])
        rows.append({"P": P, "separated": sep, "shared": shr, "permuted": perm, "no_recur": norec})
        print(f"  P={P}: SLOT-SEP {sep:.2f} | shared(~2cap) {shr:.2f} | permuted-role {perm:.2f} | no-recur {norec:.2f}")
    go = all(r["separated"] >= 0.80 and r["separated"] > r["shared"] + 0.15 for r in rows if r["P"] >= 3)
    print(f"  {'GO' if go else 'PARTIAL/BOUNDARY'}: slot-separated >=0.80 & > shared at P>=3")
    json.dump(rows, open(os.path.join(_REPO, "research/findings/raw/_keystone2_spiking_slot_binder.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
