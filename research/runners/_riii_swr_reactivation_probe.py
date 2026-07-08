"""R-iii probe (the #5 gate's cheap diagnostic for the 2026-05-24 fully-spiking-SWR generative-replay NEGATIVE):
does the SWR replay REACTIVATE a specific stored engram, and is the diagnosed fix (the explicit CA3 DRIVE that the
validated Phase 1.3 replay uses, which the (c) loop's `trigger_swr_replay` OMITS) what makes it specific?

Builds a small fresh hippocampus substrate (EC/DG/CA3/CA1 trisynaptic + the ca3_swr_burst gate), tags K engrams on
CA3, then compares SWR reactivation SPECIFICITY in two conditions:
  - NO-DRIVE  (the (c) loop): open the ca3_swr_burst gate + run -- NO stimulate_tag (the diagnosed bug)
  - WITH-DRIVE (Phase 1.3):   stimulate_tag(correct) to seed + open the gate + run (the diagnosed fix)
Specificity = post-replay CA3 activity overlap with the CORRECT tag minus the mean overlap with the OTHER tags.
Diagnosis (per the 2026-05-24 finding): WITH-DRIVE specific + NO-DRIVE at chance -> the missing CA3 drive is the
bottleneck (fix = add stimulate_tag to trigger_swr_replay, ~10 lines). numpy-CPU. NO `sim/` edit.
"""
from __future__ import annotations
import argparse
import time
import numpy as np


def _build(seed, n_lang=512, n_ec=200, n_dg=400, n_ca3=120, n_ca1=120, train_ca3=False):
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    # train_ca3: strengthen the CA3 recurrent AUTOASSOCIATOR (Marr, D.13) so a seeded ensemble COMPLETES + sustains
    # via recurrence -- the fresh (untrained) substrate has no attractor, so a seed decays to non-specific.
    ca3w = 5.0 if train_ca3 else 1.5
    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang, n_motor_per_action=16, n_motor_fs_per_action=4, enable_motor_fs=True,
        enable_language_output=True, n_lang_output=n_lang, enable_hippocampus_consolidation=True,
        n_ec=n_ec, n_dg=n_dg, n_ca3=n_ca3, n_ca1=n_ca1, ca3_recurrent_density=0.30, ca3_recurrent_weight=ca3w)
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions); cfg.region_pathways = list(pathways)
    cfg.dt_ms = 1.0; cfg.seed = seed; cfg.enable_nmda = True
    cfg.enable_structural_plasticity = False; cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = bool(train_ca3); cfg.stdp_w_max = 10.0; cfg.fast_spike_reset = True
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def run_seed(seed, n_tags=4, encode_steps=60, replay_steps=100, drive_pA=200.0, train_ca3=False):
    from sim.backend import get_backend
    from sim.text_embeddings import orthogonal_drive_pattern
    cp, _ = get_backend()
    bridge = _build(seed, train_ca3=train_ca3)
    rm = bridge.region_manager
    lang = np.array(list(rm.indices("language_input")), dtype=np.int64)
    ca3 = np.array(list(rm.indices("ca3")), dtype=np.int64)
    n_total = bridge.cp_external_input_current.shape[0]

    # TAG K engrams: drive a distinct orthogonal lang pattern -> propagate the trisynaptic loop -> tag the CA3 ensemble
    tags = [f"concept{i}" for i in range(n_tags)]
    for i, tg in enumerate(tags):
        drive = orthogonal_drive_pattern(cue_idx=i, n_cues=n_tags, n_neurons=len(lang),
                                         drive_max_pA=drive_pA, sparsity=0.05)
        drive = cp.asarray(drive, dtype=cp.float32)
        bridge.start_engram_recording(tg)
        for _ in range(encode_steps):
            ext = cp.zeros(n_total, dtype=cp.float32); ext[cp.asarray(lang)] = drive
            bridge.cp_external_input_current[:] = ext
            bridge._run_one_simulation_step()
        bridge.cp_external_input_current[:] = 0.0
        bridge.commit_engram_tag(tg, top_k=60, region_filter=["ca3"])
    if train_ca3:                                            # freeze the trained autoassociator before the SWR test
        bridge.core_config.enable_hebbian_learning = False
    tag_idx = {tg: set(int(x) for x in bridge.get_engram_tag_indices(tg)) for tg in tags}

    def settle(steps=40):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(steps):
            bridge._run_one_simulation_step()

    def swr_capture(correct_tag, with_drive, seed_steps=20):
        """Run one SWR replay event; accumulate CA3 spike counts ONLY in the POST-seed window (steps >= seed_steps)
        so the count reflects recurrence-driven pattern COMPLETION, not the direct drive; return the top-active set.
        Both conditions use the identical capture window -- only the seed (steps < seed_steps) differs."""
        settle(40)
        counts = np.zeros(len(ca3))
        bridge.set_plasticity_gate("ca3_swr_burst", 1.0)
        for step in range(replay_steps):
            if with_drive and step < seed_steps:              # Phase 1.3: SEED with the CA3 drive for the first ~20ms
                bridge.stimulate_tag(correct_tag, drive_pA=drive_pA, additive=False)
            else:
                bridge.clear_tag_drive()
                bridge.cp_external_input_current[:] = 0.0
            bridge._run_one_simulation_step()
            if step >= seed_steps:                            # capture only the POST-seed completion window (both conds)
                fs = bridge.cp_firing_states
                fs = np.asarray(fs.get() if hasattr(fs, "get") else fs)
                counts += fs[ca3]
        bridge.set_plasticity_gate("ca3_swr_burst", 0.0)
        bridge.clear_tag_drive()
        k = max(20, int(0.5 * max(len(v) for v in tag_idx.values())))
        top = set(int(ca3[j]) for j in np.argsort(-counts)[:k])
        return top

    def specificity(top, correct):
        def ov(tg):
            g = tag_idx[tg]
            return len(top & g) / max(1, len(g))
        corr = ov(correct)
        others = np.mean([ov(tg) for tg in tags if tg != correct])
        return corr - others

    nod, wd = [], []
    for tg in tags:
        nod.append(specificity(swr_capture(tg, with_drive=False), tg))
        wd.append(specificity(swr_capture(tg, with_drive=True), tg))
    return {"no_drive_spec": float(np.mean(nod)), "with_drive_spec": float(np.mean(wd))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--train", action="store_true", help="train the CA3 recurrent autoassociator during encoding")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"[R-iii SWR reactivation probe] post-seed COMPLETION specificity, CA3-drive vs not "
          f"| train_ca3={a.train}", flush=True)
    ND, WD = [], []
    for s in seeds:
        t0 = time.time()
        r = run_seed(s, train_ca3=a.train)
        ND.append(r["no_drive_spec"]); WD.append(r["with_drive_spec"])
        print(f"  [seed {s}] NO-DRIVE spec={r['no_drive_spec']:+.3f}  WITH-DRIVE spec={r['with_drive_spec']:+.3f}  ({time.time()-t0:.0f}s)", flush=True)
    nd, wd = float(np.mean(ND)), float(np.mean(WD))
    confirms = wd - nd > 0.20 and wd > 0.20
    print(f"\n  AGGREGATE: NO-DRIVE spec={nd:+.3f}  WITH-DRIVE spec={wd:+.3f}", flush=True)
    print(f"  VERDICT: {'DIAGNOSIS-CONFIRMED' if confirms else 'INCONCLUSIVE/OTHER-BOTTLENECK'} -- "
          f"{'the explicit CA3 DRIVE makes SWR reactivation SPECIFIC while the (c) loop (no drive) is non-specific -> the diagnosed fix (add stimulate_tag to trigger_swr_replay) is the R-iii bottleneck (failure-mode #1)' if confirms else 'the CA3 drive does NOT cleanly restore specificity here -> the bottleneck is elsewhere (consolidation / decoder / substrate), per the 2026-05-24 failure-mode #2/#3'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
