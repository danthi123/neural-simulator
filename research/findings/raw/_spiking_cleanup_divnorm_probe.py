"""(de-risk A, Task 1) Does a SPIKING matched-filter cleanup + DIVISIVE NORMALIZATION (Carandini-Heeger,
conductance-based shunting) + temporal integration reach NUMPY PARITY at cleaning up the composer's REAL
noisy unbind estimate? This GATES the (A) build: if parity -> GO (build the spiking cleanup into the
composer); if it plateaus -> NEGATIVE honest boundary (the disclosed numpy argmax readout stands).

Prior cheap-first (`2026-06-04-spine-item2-cleanup-noisy-est-wall.md`): the bare matched filter plateaus at
~0.78 on the composer's real est (cue-cosine ~0.35) vs numpy 1.00. Diagnosed cause = SATURATION (true
concept + competitors all driven past saturation -> rates tie -> argmax can't separate). Lower gain alone
does NOT fix it. The diagnosed fix is the canonical cortical gain control: DIVISIVE NORMALIZATION.

Mechanism built here (NO sim/ edits; reuse-by-import):
  - M concept matched-filter neurons: input synaptic weights = each concept's ZCA-decorrelated code on the
    ON/OFF channels (exactly the existing matched-filter probe `build_cleanup_bridge`).
  - PLUS a divisive-normalization circuit: an INHIBITORY-TRAIT FS pool that pools the TOTAL concept-population
    activity (concept -> FS, excitatory) and feeds conductance-based SHUNTING inhibition back to every concept
    neuron (FS -> concept, I_TO_E with inhibitory trait => g_i*(E_i - V) divisive term). response_i ~
    drive_i / (sigma + sum_j drive_j). NOT subtractive WTA (which HURT, 0/45 -- prior finding).
    sigma is realized by E_inh (syn_reversal_potential_i): closer to rest = gentler shunt; more negative =
    stronger. We also sweep the FS pool weight + match gain + run-steps.
  - Temporal integration: accumulate concept firing over run-steps (the readout window).

CRITICAL mechanism detail (bridge.py 5046-5070): g_e vs g_i routing depends on whether the PRESYNAPTIC
neuron is INHIBITORY (cp_traits in inhibitory_trait_indices), NOT on the wiring plan's conn_type string.
So the FS pool MUST carry an inhibitory trait + enable_inhibitory_neurons=True to truly shunt (the prior
WTA probe left this False, so its 'I_TO_E' weights added to g_e = excitation -> that NEGATIVE was testing
lateral EXCITATION). Verified in `_divnorm_mechanism_sanity.py`.

The composer's REAL est is captured via `composer._unbind_onoff(bound, role)` at V=320 production codes
(the setup the capability matrix is validated on). Recovery = fraction of est's whose cleanup returns the
TRUE concept (same metric as the prior probes). We report numpy oracle vs spiking-WITH-divnorm vs
spiking-WITHOUT (the plateau baseline), so the with-vs-without contrast shows divisive norm is the lever.

  python -m research.findings.raw._spiking_cleanup_divnorm_probe --seed 42 --proj-dim 800 \
      --w-match 40 --w-cfs 150 --w-fs 20 --einh -75 --run-steps 300
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import get_backend, to_host
from research.runners.core_sim_composition import CoreSimComposer, onoff, _scale_to_current
from research.findings.raw._core_composer_grounded320_probe import production_codes

RESET_STEPS = 20
INPUT_DRIVE = 2500.0
INH_TRAIT = 1


def build_divnorm_bridge(seed, codes, w_match, w_cfs, w_fs, n_fs, einh, enable_divnorm, ou_std=20.0):
    """codes: (M, D) centered+normalized concept codes.
    Layout: input_ON[0,D) + input_OFF[D,2D) + concept[2D, 2D+M) + FS[2D+M, 2D+M+n_fs).
      - matched filter: input_ON/OFF -> concept (codes as receptive fields, E_TO_E).
      - divisive norm (if enable_divnorm): concept -> FS (E_TO_I pooling), FS -> concept (I_TO_E shunting);
        FS neurons carry the inhibitory trait so g_i routing fires.
    `einh` sets syn_reversal_potential_i (the divisive sigma surrogate); `ou_std` the background OU noise."""
    M, D = codes.shape
    base = 2 * D + M
    N = base + (n_fs if enable_divnorm else 0)
    cfg = CoreSimConfig()
    cfg.num_neurons = N
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed); cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0; cfg.num_traits = 2 if enable_divnorm else 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
              "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.ou_std_current_pA = float(ou_std)
    if enable_divnorm:
        cfg.enable_inhibitory_neurons = True
        cfg.inhibitory_trait_indices = [INH_TRAIT]
        cfg.syn_reversal_potential_i = float(einh)

    in_on = np.arange(0, D); in_off = np.arange(D, 2 * D); concept = np.arange(2 * D, 2 * D + M)
    fs = np.arange(base, base + n_fs)
    code_on = np.maximum(codes, 0.0); code_off = np.maximum(-codes, 0.0)
    mpre, mpost, mw = [], [], []                      # matched filter (codes as synaptic receptive fields)
    for c in range(M):
        for i in range(D):
            if code_on[c, i] > 0:
                mpre.append(int(in_on[i])); mpost.append(int(concept[c])); mw.append(float(code_on[c, i] * w_match))
            if code_off[c, i] > 0:
                mpre.append(int(in_off[i])); mpost.append(int(concept[c])); mw.append(float(code_off[c, i] * w_match))
    plan = {"match": {"pre_indices": mpre, "post_indices": mpost,
                      "initial_weights": np.array(mw, dtype=np.float32), "plastic": False,
                      "conn_type": "E_TO_E", "count": len(mpre)}}
    if enable_divnorm:
        cpre, cpost, cw = [], [], []                  # concept -> FS (pool total population activity)
        for c in range(M):
            for j in range(n_fs):
                cpre.append(int(concept[c])); cpost.append(int(fs[j])); cw.append(float(w_cfs))
        ipre, ipost, iw = [], [], []                  # FS -> concept (divisive shunting feedback)
        for j in range(n_fs):
            for c in range(M):
                ipre.append(int(fs[j])); ipost.append(int(concept[c])); iw.append(float(w_fs))
        plan["pool"] = {"pre_indices": cpre, "post_indices": cpost,
                        "initial_weights": np.array(cw, dtype=np.float32), "plastic": False,
                        "conn_type": "E_TO_I", "count": len(cpre)}
        plan["shunt"] = {"pre_indices": ipre, "post_indices": ipost,
                         "initial_weights": np.array(iw, dtype=np.float32), "plastic": False,
                         "conn_type": "I_TO_E", "count": len(ipre)}

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    xp, _ = get_backend()
    if enable_divnorm:                                # mark FS pool inhibitory BEFORE first step (mask cached then)
        tr = bridge.cp_traits
        tr[:] = 0
        tr[xp.asarray(fs, dtype=tr.dtype)] = INH_TRAIT
        bridge.cp_traits = tr
        bridge._cached_inhibitory_mask = None
    bridge.inject_explicit_wiring(plan)
    idx = {"in_on": xp.asarray(in_on, dtype=xp.int64), "in_off": xp.asarray(in_off, dtype=xp.int64),
           "concept": xp.asarray(concept, dtype=xp.int64),
           "fs": xp.asarray(fs, dtype=xp.int64) if enable_divnorm else None}
    return bridge, idx


def cleanup_divnorm(bridge, idx, D, M, est, concept_bias, run_steps, return_fs=False, input_drive=INPUT_DRIVE):
    xp, _ = get_backend()
    e_on, e_off = onoff(est)
    on_cur, off_cur = _scale_to_current(e_on, e_off, input_drive)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge._run_one_simulation_step()
    cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
    cur[idx["in_on"]] = xp.asarray(on_cur.astype(np.float32))
    cur[idx["in_off"]] = xp.asarray(off_cur.astype(np.float32))
    cur[idx["concept"]] = concept_bias
    bridge.cp_external_input_current[:] = cur
    acc = xp.zeros(M, dtype=xp.float64)
    fs_acc = 0.0
    for _ in range(run_steps):
        bridge._run_one_simulation_step()
        acc += bridge.cp_firing_states[idx["concept"]].astype(xp.float64)
        if return_fs and idx["fs"] is not None:
            fs_acc += float(bridge.cp_firing_states[idx["fs"]].astype(xp.float64).sum())
    bridge.cp_external_input_current[:] = 0.0
    rates = to_host(acc) / run_steps
    if return_fs:
        n_fs = len(idx["fs"]) if idx["fs"] is not None else 1
        return rates, fs_acc / run_steps / max(1, n_fs)
    return rates


def capture_real_est(seed, vocab, proj_dim, n_flat, n_attr):
    """Build the V=320 production-code composer, store facts of each kind, capture the REAL est
    (e_on - e_off) for each role via composer._unbind_onoff. Returns (items, code_mat, widx, words)."""
    codes_in = production_codes(vocab, 2000, 100, proj_dim, seed)
    words = [f"c{i:03d}" for i in range(vocab)]
    concepts = {w: codes_in[i] for i, w in enumerate(words)}
    comp = CoreSimComposer(seed=seed, proj_dim=proj_dim, concepts=concepts)
    code_mat = np.stack([comp.concepts[w] for w in comp.words])
    widx = {w: i for i, w in enumerate(comp.words)}
    rng = np.random.default_rng(seed + 1)

    def pick(k):
        return [str(x) for x in rng.choice(comp.words, size=k, replace=False)]

    items = []
    for _ in range(n_flat):
        a, ac, p = pick(3)
        comp.kb = []; comp.store(a, ac, p)
        bound = comp.kb[0][1]
        for role, true in (("agent", a), ("action", ac), ("patient", p)):
            e_on, e_off = comp._unbind_onoff(bound, role)
            items.append((e_on - e_off, true, "flat"))
    for _ in range(n_attr):
        a, ac, adj1, adj2, noun = pick(5)
        comp.kb = []; comp.store(a, ac, ((adj1, adj2), noun))
        bound = comp.kb[0][1]
        for role, true in (("patient", noun), ("attribute", adj1), ("attribute2", adj2)):
            e_on, e_off = comp._unbind_onoff(bound, role)
            items.append((e_on - e_off, true, "two_attr"))
    return items, code_mat, widx, comp.words


def evaluate(items, code_mat, widx, words, bridge, idx, D, M, concept_bias, run_steps,
             input_drive=INPUT_DRIVE):
    """Return (numpy_recovery, spiking_recovery, mean_cue_cos)."""
    np_ok = sp_ok = 0
    coss = []
    for est, true, _cat in items:
        coss.append(float(code_mat[widx[true]] @ est / (np.linalg.norm(est) + 1e-12)))
        np_ok += int(words[int(np.argmax(code_mat @ est))] == true)
        rates = cleanup_divnorm(bridge, idx, D, M, est, concept_bias, run_steps, input_drive=input_drive)
        sp_ok += int(words[int(np.argmax(rates))] == true)
    n = len(items)
    return np_ok / n, sp_ok / n, float(np.mean(coss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab", type=int, default=320)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n-flat", type=int, default=15)
    ap.add_argument("--n-attr", type=int, default=8)
    ap.add_argument("--w-match", type=float, default=40.0)
    ap.add_argument("--w-cfs", type=float, default=150.0, help="concept->FS pooling weight (divisive)")
    ap.add_argument("--w-fs", type=float, default=20.0, help="FS->concept shunting weight (divisive gain)")
    ap.add_argument("--n-fs", type=int, default=40)
    ap.add_argument("--einh", type=float, default=-75.0, help="syn_reversal_potential_i (divisive sigma surrogate)")
    ap.add_argument("--concept-bias", type=float, default=-150.0)
    ap.add_argument("--run-steps", type=int, default=300)
    ap.add_argument("--no-divnorm", action="store_true", help="baseline: matched filter only (the plateau)")
    ap.add_argument("--diagnose", action="store_true", help="print population regime stats, then exit")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items, code_mat, widx, words = capture_real_est(args.seed, args.vocab, args.proj_dim, args.n_flat, args.n_attr)
    M = len(words); D = code_mat.shape[1]

    enable = not args.no_divnorm
    bridge, idx = build_divnorm_bridge(args.seed, code_mat, args.w_match, args.w_cfs, args.w_fs,
                                       args.n_fs, args.einh, enable_divnorm=enable)

    if args.diagnose:
        # Print population regime for the first few items: true-concept rate, max-competitor rate,
        # mean rate over all M, and FS pool rate. Reveals whether the population is saturated (rates tie)
        # or in the responsive range (true separates from competitors).
        for est, true, cat in items[:6]:
            rates, fs_rate = cleanup_divnorm(bridge, idx, D, M, est, args.concept_bias, args.run_steps,
                                             return_fs=True)
            ti = widx[true]
            comp_rates = np.delete(rates, ti)
            order = np.argsort(rates)[::-1]
            win = words[order[0]]
            print(f"[diag] {cat} true={true} true_rate={rates[ti]:.3f} max_comp={comp_rates.max():.3f} "
                  f"mean={rates.mean():.3f} nonzero={int((rates>0).sum())}/{M} fs={fs_rate:.3f} "
                  f"win={'OK' if win==true else win}", flush=True)
        return

    np_rec, sp_rec, cue_cos = evaluate(items, code_mat, widx, words, bridge, idx, D, M,
                                       args.concept_bias, args.run_steps)

    res = {"seed": args.seed, "vocab": args.vocab, "proj_dim": args.proj_dim, "n_items": len(items),
           "divnorm": enable, "w_match": args.w_match, "w_cfs": args.w_cfs, "w_fs": args.w_fs,
           "n_fs": args.n_fs, "einh": args.einh, "run_steps": args.run_steps,
           "numpy": np_rec, "spiking": sp_rec, "cue_cos": cue_cos}
    tag = "divnorm" if enable else "NOdivnorm"
    print(f"[{tag}] seed={args.seed} V={args.vocab} n={len(items)} cue_cos={cue_cos:.3f}  "
          f"numpy={np_rec:.3f}  spiking={sp_rec:.3f}  "
          f"(w_match={args.w_match} w_cfs={args.w_cfs} w_fs={args.w_fs} einh={args.einh} steps={args.run_steps})",
          flush=True)
    print("[result] " + json.dumps(res), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
