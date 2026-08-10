"""Gap #5 — CA3 point-neuron SELECTIVE completion: weight x density held-out sweep (NEW runner, NO sim/ edit).

Frontier (2026-08-10): the 6-seed episodic recall GO (0.646) leaves the residual at CA3 ATTRACTOR
STRENGTH / SPECIFICITY. The pending sweep (2026-07-17 finding) asks: does trained recurrent LTP yield
pattern-SELECTIVE completion on a POINT neuron (no dendritic bistability)? GO gate: recurrence-gain > 0.15,
trained held-out completion > 0.30, AND pattern-selective (held-out A members reactivate; permuted cue and
non-members do NOT).

This ISOLATES the attractor wall from the training-collapse confound (2026-07-17 root-cause: the substrate's
rate-Hebbian rule collapses ca3->ca3 to a uniform ~0.846 fixed point during training, so no pattern-specific
attractor ever forms). Method: build with a WEAK uniform baseline recurrent weight + FROZEN plasticity, then
HAND-INSTALL a perfect pattern-selective within-assembly potentiation directly on cp_connections.data (the
idealized outcome a perfect recurrent LTP would produce). If even a PERFECT selective attractor cannot complete
selectively at feasible weight, that is an honest negative on the point-neuron path -> redirect to the
dendritic-plateau completion readout (already 6-seed GO, 2026-07-18). Hand-install is a host idealization of LTP
(brain-based-only note): legitimate for a de-risk that bounds what the point-neuron substrate CAN do.

Skeptical controls (mandatory):
  (a) PERMUTED cue -- drive a random NON-A subset of the same size; A's held-out members must NOT reactivate.
  (b) UNTRAINED -- weak baseline recurrents (no selective install); completion must collapse (LTP load-bearing).
  (c) RECURRENCE-ZERO -- zero ALL ca3->ca3; held-out firing must collapse (isolates completion from any re-drive).
  plus NO-CUE rest firing (self-ignition / bistability check) and NON-MEMBER reactivation (specificity).
"""
from __future__ import annotations
import argparse, json, time
import numpy as np

from research.runners.validate_trisynaptic_loop import (
    measure_region_response, cosine_similarity)


def _build(seed, n_lang=384, n_ec=160, n_dg=300, n_ca3=200, n_ca1=120,
           ca3_density=0.30, base_w=5.0):
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang, n_motor_per_action=16, n_motor_fs_per_action=4, enable_motor_fs=True,
        enable_language_output=True, n_lang_output=n_lang, enable_hippocampus_consolidation=True,
        n_ec=n_ec, n_dg=n_dg, n_ca3=n_ca3, n_ca1=n_ca1, ca3_recurrent_density=ca3_density,
        ca3_recurrent_weight=float(base_w))
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions); cfg.region_pathways = list(pathways)
    cfg.dt_ms = 1.0; cfg.seed = seed; cfg.enable_nmda = True
    # FREEZE all plasticity so the hand-installed attractor cannot drift (avoids the Hebbian-collapse confound).
    cfg.enable_structural_plasticity = False; cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False; cfg.enable_stdp = False
    cfg.fast_spike_reset = True
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def _csr_row_col(cp, C):
    """Return (rows, cols) global pre/post index arrays aligned to C.data (CSR assumed: cp_connections[pre,post])."""
    n = C.shape[0]
    rows = cp.repeat(cp.arange(n, dtype=cp.int64), cp.diff(C.indptr))
    cols = C.indices.astype(cp.int64)
    return rows, cols


def run_seed(seed, n_ca3=200, n_lang=384, ca3_density=0.30, weights=(120, 600, 1500, 3000, 6000),
             n_assembly=3, assembly_frac=0.12, cue_frac=0.5, base_w=5.0,
             drive_pA=200.0, recall_steps=40, reset_steps=30, verbose=True):
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    bridge = _build(seed, n_lang=n_lang, n_ca3=n_ca3, ca3_density=ca3_density, base_w=base_w)
    rm = bridge.region_manager
    ca3_idx = np.asarray(list(rm.indices("ca3")), dtype=np.int64)
    n = bridge.cp_connections.shape[0]
    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}          # global CA3 index -> position in response vector

    C = bridge.cp_connections
    mtype = type(C).__name__
    if not hasattr(C, "indptr"):
        C = C.tocsr(); bridge.cp_connections = C                  # ensure in-place-writable CSR
    rows, cols = _csr_row_col(cp, C)
    baseline_data = C.data.copy()                                 # weak uniform baseline (untrained / restore point)

    # membership masks over the FULL neuron index space
    is_ca3 = cp.zeros(n, dtype=cp.bool_); is_ca3[cp.asarray(ca3_idx)] = True
    rec_mask = is_ca3[rows] & is_ca3[cols]                        # all ca3->ca3 synapses
    n_rec = int(to_host(cp.sum(rec_mask)))

    # K disjoint assemblies (sparse random subsets of CA3)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ca3_idx))
    a_size = max(6, int(assembly_frac * len(ca3_idx)))
    assemblies = []
    for a in range(n_assembly):
        sel = perm[a * a_size:(a + 1) * a_size]
        assemblies.append(ca3_idx[sel])
    withinA_masks = []
    for A in assemblies:
        is_A = cp.zeros(n, dtype=cp.bool_); is_A[cp.asarray(A)] = True
        withinA_masks.append(is_A[rows] & is_A[cols])

    def restore_baseline():
        C.data[:] = baseline_data

    def install_selective(W):
        restore_baseline()
        for m in withinA_masks:
            C.data[m] = cp.float32(W)

    def zero_recurrents():
        restore_baseline()
        C.data[rec_mask] = cp.float32(0.0)

    def resp(drive_global):
        return measure_region_response(bridge, "ca3", np.asarray(drive_global, dtype=np.int64),
                                       drive_pA=drive_pA, drive_region="ca3",
                                       n_steps=recall_steps, reset_steps=reset_steps)

    # --- partial-cue completion + selectivity for one assembly, given the CURRENT weights ---
    def eval_assembly(A):
        se = np.asarray(A, dtype=np.int64)
        r = np.random.default_rng(seed + int(se[0]))
        order = r.permutation(len(se))
        se = se[order]
        n_cue = max(2, int(cue_frac * len(se)))
        cue, held = se[:n_cue], se[n_cue:]
        cue_pos = [ca3_pos[int(g)] for g in cue]
        held_pos = [ca3_pos[int(g)] for g in held]
        member = set(int(g) for g in se)
        nonmember_pos = [i for g, i in ca3_pos.items() if g not in member]

        full = resp(se)
        part = resp(cue)
        # permuted cue: random NON-A CA3 cells, same count as the real cue
        nonA = np.asarray([g for g in ca3_idx if int(g) not in member], dtype=np.int64)
        perm_cue = r.choice(nonA, size=len(cue), replace=False)
        perm_resp = resp(perm_cue)

        cue_act = float(np.mean(part[cue_pos])) + 1e-9
        held_act = float(np.mean(part[held_pos])) if held_pos else 0.0
        nonmember_act = float(np.mean(part[nonmember_pos])) if nonmember_pos else 0.0
        # permuted-cue reactivation of A's held-out members (should be ~0)
        perm_held_act = float(np.mean(perm_resp[held_pos])) if held_pos else 0.0
        perm_cue_pos = [ca3_pos[int(g)] for g in perm_cue]
        perm_cueact = float(np.mean(perm_resp[perm_cue_pos])) + 1e-9

        return {
            "completion": held_act / cue_act,                    # held-out firing relative to the cue
            "held_act": held_act, "cue_act": cue_act - 1e-9,
            "nonmember_act": nonmember_act,
            "sel_held_over_nonmember": held_act / (nonmember_act + 1e-9),
            "perm_completion": perm_held_act / perm_cueact,      # permuted cue -> A held-out (specificity; want ~0)
            "full": full, "part": part,
        }

    def cross_specificity(evals):
        """cos(partial_m, full_m) - mean_k!=m cos(partial_m, full_k)."""
        own, oth = [], []
        for m in range(len(evals)):
            own.append(cosine_similarity(evals[m]["part"], evals[m]["full"]))
            oth.append(float(np.mean([cosine_similarity(evals[m]["part"], evals[k]["full"])
                                      for k in range(len(evals)) if k != m])))
        return float(np.mean(own)), float(np.mean(oth))

    # NO-CUE rest firing (self-ignition / non-silent check): drive nothing, measure CA3 firing
    def no_cue_rest():
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset_steps + recall_steps):
            bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
        ca3_arr = cp.asarray(ca3_idx)
        cnt = 0.0
        for _ in range(recall_steps):
            bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
            cnt += float(to_host(bridge.cp_firing_states[ca3_arr].astype(cp.float32).mean()))
        return cnt / recall_steps

    # ---- CONTROL: untrained (weak baseline, no selective install) ----
    restore_baseline()
    unt_evals = [eval_assembly(A) for A in assemblies]
    untrained_completion = float(np.mean([e["completion"] for e in unt_evals]))

    # ---- CONTROL: recurrence-zero ----
    zero_recurrents()
    zero_evals = [eval_assembly(A) for A in assemblies]
    zero_completion = float(np.mean([e["completion"] for e in zero_evals]))

    # ---- SWEEP: selective install at each W ----
    grid = []
    for W in weights:
        install_selective(W)
        rest = no_cue_rest()
        evals = [eval_assembly(A) for A in assemblies]
        own_cos, oth_cos = cross_specificity(evals)
        comp = float(np.mean([e["completion"] for e in evals]))
        nonmem = float(np.mean([e["nonmember_act"] for e in evals]))
        held = float(np.mean([e["held_act"] for e in evals]))
        sel_nm = float(np.mean([e["sel_held_over_nonmember"] for e in evals]))
        perm_comp = float(np.mean([e["perm_completion"] for e in evals]))
        from tools.lab import attributable_to
        # attribute completion to trained recurrent LTP (vs untrained) AND selectivity to the cue (vs permuted).
        attributable_to(f"[d{ca3_density} W{W:.0f}] completion: trained-recurrent vs untrained (LTP load-bearing)",
                        comp, untrained_completion)
        attributable_to(f"[d{ca3_density} W{W:.0f}] SELECTIVITY: partial-cue completion vs PERMUTED-cue (specificity)",
                        comp, perm_comp)
        row = {
            "seed": seed, "ca3_density": ca3_density, "W": float(W), "n_rec_syn": n_rec,
            "trained_completion": comp, "untrained_completion": untrained_completion,
            "recurrence_gain": comp - untrained_completion,
            "held_act": held, "nonmember_act": nonmem, "sel_held_over_nonmember": sel_nm,
            "perm_completion": perm_comp, "no_cue_rest_rate": rest,
            "spec_cos_own": own_cos, "spec_cos_other": oth_cos, "spec_cos_margin": own_cos - oth_cos,
            "zero_completion": zero_completion,
        }
        # GO per (W,density): magnitude + recurrence + selective (permuted fails, held>>nonmember, cos margin +, not self-igniting)
        row["GO"] = bool(comp > 0.30 and row["recurrence_gain"] > 0.15 and perm_comp < 0.30
                         and sel_nm > 2.0 and (own_cos - oth_cos) > 0.05 and rest < 0.05)
        grid.append(row)
        if verbose:
            print(f"  [s{seed} d{ca3_density} W{W:>6.0f}] comp={comp:.3f} (unt={untrained_completion:.3f} "
                  f"gain={row['recurrence_gain']:+.3f} zero={zero_completion:.3f}) | held/nonmem={sel_nm:6.2f} "
                  f"perm={perm_comp:.3f} rest={rest:.3f} cos(own={own_cos:.3f}-oth={oth_cos:.3f}) "
                  f"{'GO' if row['GO'] else '--'}", flush=True)
    return grid, mtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--densities", default="0.10,0.30,0.50")
    ap.add_argument("--weights", default="120,600,1500,3000,6000")
    ap.add_argument("--n-ca3", type=int, default=200)
    ap.add_argument("--assembly-frac", type=float, default=0.12)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    densities = [float(x) for x in a.densities.replace(",", " ").split()]
    weights = [float(x) for x in a.weights.replace(",", " ").split()]
    print(f"[gap5 CA3 selective-attractor sweep] seeds={seeds} densities={densities} weights={weights} "
          f"n_ca3={a.n_ca3} assembly_frac={a.assembly_frac}", flush=True)
    print("  GO gate: comp>0.30 & recurrence-gain>0.15 & perm<0.30 & held/nonmem>2 & cos-margin>0.05 & rest<0.05", flush=True)
    all_rows = []
    for s in seeds:
        for d in densities:
            t0 = time.time()
            grid, mtype = run_seed(s, n_ca3=a.n_ca3, ca3_density=d, weights=weights,
                                   assembly_frac=a.assembly_frac)
            all_rows.extend(grid)
            print(f"    (seed {s} density {d}: {time.time()-t0:.0f}s, cp_connections={mtype})", flush=True)
    if a.json:
        json.dump(all_rows, open(a.json, "w"), indent=1)
        print(f"  wrote {a.json}", flush=True)
    any_go = any(r["GO"] for r in all_rows)
    best = max(all_rows, key=lambda r: (r["GO"], r["trained_completion"])) if all_rows else None
    print(f"\n  VERDICT: {'SOME GO' if any_go else 'NO GO (point-neuron selective completion fails the gate)'}", flush=True)
    if best:
        print(f"  best cell: density={best['ca3_density']} W={best['W']:.0f} comp={best['trained_completion']:.3f} "
              f"gain={best['recurrence_gain']:+.3f} perm={best['perm_completion']:.3f} "
              f"held/nonmem={best['sel_held_over_nonmember']:.2f} rest={best['no_cue_rest_rate']:.3f} "
              f"GO={best['GO']}", flush=True)
    if not any_go:
        print("  => point-neuron recurrent LTP does NOT give pattern-selective completion above the gate at any "
              "swept weight x density -> redirect to the dendritic-plateau readout (2026-07-18 CLOSED 6-seed GO).", flush=True)


if __name__ == "__main__":
    main()
