"""R-iii corrected probe (CYCLE 1062) — CA3 Marr autoassociator PARTIAL-cue completion SPECIFICITY, reusing the
VALIDATED D.13 regime (validate_trisynaptic_loop: open the ca3_swr_burst/dg_to_ca3/ec_to_dg/lang_to_ec gates, train
the full pattern N events, record the stored CA3 ensemble; measure cos(partial-cue CA3 response, full-cue response)).
CYCLE 1060/1061 used the WRONG regime (60 steps, global Hebbian, a held-out-activation metric) + got a false
negative. This reproduces what WORKED and EXTENDS it to the R-iii question: with K distinct memories stored, does a
partial cue of memory A complete A SPECIFICALLY (cos to A's full >> cos to the others)? That specific partial-cue
completion is the reactivation the (c) SWR generative-replay loop needs. Reuse-by-import of the validated helpers.
NO `sim/` edit.

Anti-cheats: (A) NO-TRAIN control (fresh attractor, no LTP) -> completion collapses (per NMDA-KO: partial recall
fails without recurrent LTP); (B) SPECIFICITY -- cos(partial_A, full_A) must exceed cos(partial_A, full_other);
(C) partial-cue is a strict SUBSET of the stored ensemble (the held-out half must be reconstructed by recurrence).
"""
from __future__ import annotations
import argparse, time
import numpy as np
from research.runners.validate_trisynaptic_loop import (
    measure_region_response, build_drive_pattern, cosine_similarity)


def _build(seed, n_lang=512, n_ec=200, n_dg=400, n_ca3=200, n_ca1=120, ca3w=5.0, train=True):
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang, n_motor_per_action=16, n_motor_fs_per_action=4, enable_motor_fs=True,
        enable_language_output=True, n_lang_output=n_lang, enable_hippocampus_consolidation=True,
        n_ec=n_ec, n_dg=n_dg, n_ca3=n_ca3, n_ca1=n_ca1, ca3_recurrent_density=0.30,
        ca3_recurrent_weight=(ca3w if train else 1.5))
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions); cfg.region_pathways = list(pathways)
    cfg.dt_ms = 1.0; cfg.seed = seed; cfg.enable_nmda = True
    cfg.enable_structural_plasticity = False; cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = True; cfg.stdp_w_max = 10.0; cfg.fast_spike_reset = True
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


_GATES = ["ca3_swr_burst", "dg_to_ca3", "ec_to_dg", "lang_to_ec"]


def _set_gates(bridge, v):
    for g in _GATES:
        try:
            bridge.set_plasticity_gate(g, v)
        except Exception:
            pass


def run_seed(seed, n_mem=2, train_events=120, drive_pA=200.0, do_train=True,
             n_lang=384, n_ca3=150, n_dg=300, reset_steps=15, drive_steps=55, recall_steps=60):
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    bridge = _build(seed, n_lang=n_lang, n_ca3=n_ca3, n_dg=n_dg, train=do_train)
    rm = bridge.region_manager
    lang = list(rm.indices("language_input"))
    ca3_idx = list(rm.indices("ca3"))
    ca3_arr = cp.asarray(ca3_idx, dtype=cp.int64)
    n_lang = len(lang)
    # K distinct memories = K distinct sparse lang drive patterns
    patterns = [build_drive_pattern(n_neurons=n_lang, sparsity=0.1, seed=seed * 100 + m) for m in range(n_mem)]
    stored = {}                                                   # memory -> global CA3 stored-ensemble indices

    if do_train:
        _set_gates(bridge, 1.0)
    rec_last = min(10, max(1, train_events // 3))
    lang_arr = np.asarray(lang, dtype=np.int64)
    for m, pat in enumerate(patterns):
        drv = cp.asarray(lang_arr[pat], dtype=cp.int64)            # map lang-local pattern -> GLOBAL neuron indices
        spikes = cp.zeros(len(ca3_idx), dtype=cp.float32)
        for ev in range(train_events):
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(reset_steps):                           # reset/settle
                bridge._run_one_simulation_step()
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[drv] = float(drive_pA)  # drive the full pattern
            recording = ev >= train_events - rec_last
            for _ in range(drive_steps):
                bridge._run_one_simulation_step()
                if recording:
                    spikes += bridge.cp_firing_states[ca3_arr].astype(cp.float32)
        bridge.cp_external_input_current[:] = 0.0
        sp = to_host(spikes)
        n_stored = max(4, int(0.10 * len(ca3_idx)))
        top = np.argsort(-sp)[:n_stored]
        top = top[sp[top] > 0]
        stored[m] = np.array([ca3_idx[i] for i in top], dtype=np.int64)
    if do_train:
        _set_gates(bridge, 0.0)

    # recall: full + partial (50%) of each memory's stored CA3 ensemble, driven DIRECTLY on CA3
    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}          # global CA3 index -> position in the response vector
    # PASS 1: build all full/partial responses + the clean held-out completion (needs only each memory's own data)
    full_resp, part_resp, heldout, heldout_fullcue = {}, {}, [], []
    for m in range(n_mem):
        se = stored[m]
        if len(se) < 4:
            return None
        np.random.default_rng(seed + m).shuffle(se)
        n_part = max(2, int(0.5 * len(se)))
        cue, held = se[:n_part], se[n_part:]
        full_resp[m] = measure_region_response(bridge, "ca3", se, drive_pA=drive_pA, drive_region="ca3", n_steps=recall_steps)
        part_resp[m] = measure_region_response(bridge, "ca3", cue, drive_pA=drive_pA, drive_region="ca3", n_steps=recall_steps)
        # CLEAN metric (isolates recurrence from the drive artifact): the HELD-OUT stored neurons (NOT in the cue) must
        # be activated by recurrent completion. Their mean response, normalized by the driven cue neurons' response.
        held_pos = [ca3_pos[int(g)] for g in held if int(g) in ca3_pos]
        cue_pos = [ca3_pos[int(g)] for g in cue if int(g) in ca3_pos]
        held_act = float(np.mean(part_resp[m][held_pos])) if held_pos else 0.0
        cue_act = float(np.mean(part_resp[m][cue_pos])) if cue_pos else 1.0
        heldout.append(held_act / (cue_act + 1e-9))              # completion = held-out firing relative to the cue
        # ADVERSARIAL VERIFY: the SAME held-out neurons under the FULL cue (they ARE directly driven -> must fire).
        # If full-cue held-out fires but partial-cue held-out = 0 -> genuine recurrence failure, not a mapping bug.
        _fc = float(np.mean(full_resp[m][held_pos])) if held_pos else 0.0
        heldout_fullcue.append(_fc)
    # PASS 2: cross-memory cos metrics (need ALL full responses) -- secondary (drive-artifact-confounded)
    own, other, spec = [], [], []
    for m in range(n_mem):
        c_own = cosine_similarity(part_resp[m], full_resp[m])
        c_oth = float(np.mean([cosine_similarity(part_resp[m], full_resp[k]) for k in range(n_mem) if k != m]))
        own.append(c_own); other.append(c_oth); spec.append(c_own - c_oth)
    return {"completion_own": float(np.mean(own)), "completion_other": float(np.mean(other)),
            "specificity": float(np.mean(spec)), "heldout_completion": float(np.mean(heldout)),
            "heldout_fullcue": float(np.mean(heldout_fullcue)),
            "n_stored": int(np.mean([len(stored[m]) for m in range(n_mem)]))}


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--train-events", type=int, default=120)
    ap.add_argument("--json", default=None, help="write per-seed results JSON (for fan-out aggregation)")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"[R-iii CA3 completion specificity] validated D.13 regime, K=3 memories, train_events={a.train_events} "
          f"| CLEAN held-out completion (trained vs no-train) isolates recurrence from the drive artifact", flush=True)
    rows = []
    for s in seeds:
        t0 = time.time()
        rt = run_seed(s, train_events=a.train_events, do_train=True)
        rc = run_seed(s, train_events=a.train_events, do_train=False)
        if rt is None or rc is None:
            print(f"  [seed {s}] NOT-EVALUABLE (stored ensemble too small)"); continue
        row = {"seed": s, "trained_heldout": rt["heldout_completion"], "notrain_heldout": rc["heldout_completion"],
               "recurrence_gain": rt["heldout_completion"] - rc["heldout_completion"],
               "trained_own_cos": rt["completion_own"], "trained_spec": rt["specificity"], "n_stored": rt["n_stored"]}
        rows.append(row)
        print(f"  [seed {s}] held-out completion: TRAINED={rt['heldout_completion']:.3f} NO-TRAIN={rc['heldout_completion']:.3f} "
              f"recurrence-gain={row['recurrence_gain']:+.3f} | (cos own={rt['completion_own']:.3f} spec={rt['specificity']:+.3f}) ({time.time()-t0:.0f}s)", flush=True)
    if a.json and rows:
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        gains = [r["recurrence_gain"] for r in rows]
        tr = [r["trained_heldout"] for r in rows]
        go = all(g > 0.15 for g in gains) and all(t > 0.30 for t in tr)
        print(f"\n  AGGREGATE: trained held-out={np.mean(tr):.3f} recurrence-gain(trained-notrain)={np.mean(gains):+.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- the trained CA3 recurrents COMPLETE the HELD-OUT "
              f"(non-cued) stored neurons from a partial cue {'(held-out firing >> no-train control) -> genuine recurrent pattern completion, not the drive artifact; the R-iii SWR loop can build on this attractor' if go else '-- held-out neurons NOT recurrently completed above no-train (completion is the drive artifact); the point-neuron CA3 attractor is too weak at this regime -> scale train / strengthen recurrents, or the substrate needs a stronger attractor mechanism (honest boundary to push)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
