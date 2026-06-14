"""D2 Phase 2 -- the learned graded cortex EMBEDDING via a bridge FORWARD PASS with the dendritic gain.

THE DECISIVE INTEGRATION TEST: does the on-bridge dendritic per-presynaptic-source divisive gain (D2 Phase
1, `enable_dendritic_divisive_gain`) produce a cortex whose concept CODES recover the paradigmatic category
structure the point-neuron substrate failed at (Option C Stage-B: brain Pearson(S_learned,S_true) = -0.008)?

THE REFRAME (D2 plan, 2026-06-14): the Option-C readout computed codes from the LEARNED RECURRENT W via a
numpy spreading-activation -- which the Phase-1 synaptic-current gain does NOT touch. So Phase 2 reads codes
via a BRIDGE FORWARD PASS:
  - HUB region (n_hub neurons) = the context. Each concept is presented by driving the hubs with that
    concept's co-occurrence COUNT pattern (external current ~ C[concept, :]).
  - READOUT region (n_readout neurons), random dense projection from the hubs (a Johnson-Lindenstrauss
    random cortical projection -> the readout-activity cosine preserves the hub-profile cosine).
  - The code = the readout region's firing over the presentation window.
  - The DENDRITIC GAIN scales each hub's firing into the readout by g_hub = sigma/(sigma + a_hub), a_hub =
    the hub's own firing-rate EMA: the high-frequency COMMON hubs are suppressed, the rare informative hubs
    pass -> the readout code reflects the category-distinguishing structure. With the gain OFF (the
    POINT-NEURON control) the common hubs dominate -> no structure (the Option-C failure).
The hub EMAs are converged to each hub's marginal by a WARM-UP pass (all concepts presented) before the
codes are read; the gain normalizes by that marginal -- the on-substrate realization of D1's per-hub gain.

GATES (multi-seed; the contrast IS the result):
  STRUCTURE  -- gain-ON readout codes recover the structure (Pearson(cos(codes),S_true) >= bar) WHILE the
                gain-OFF (point-neuron) control is ~0.  HEADLINE.
  HOST CEILING confirms the data carries it (PPMI+SVD on the counts >= bar).
ANTI-CHEATS: point-neuron-must-fail (gain off ~0 on the identical forward pass); S_true a-priori; permuted-
  similarity collapses; lesion (sigma huge -> gain~1) collapses to the point-neuron; not-collapsed.

NO new sim/ edits (uses the Phase-1 gain + the brain-region framework). CPU-numpy SMOKE validates the
forward-pass logic; the decisive run is GPU + the REAL Option-C TinyStories corpus (--real-corpus).

Run (CPU smoke -- synthetic counts, fast):
  SIM_BACKEND=numpy python -u -m research.runners.dendritic_cortex_forward_codes_derisk --smoke --seeds 42
Run (GPU decisive -- real Option-C corpus):
  SIM_BACKEND=cupy python -u -m research.runners.dendritic_cortex_forward_codes_derisk \
      --real-corpus --seeds 42,43,44 --out research/findings/raw/_dendritic_cortex_forward_multiseed.json
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

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    build_concept_hub_counts, _cos_sim, _pearson_vs_Strue, heldout_generalization, effective_rank,
)
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402


def _build_cortex_bridge(n_hub, n_readout, seed, enable_gain, sigma, alpha, readout_density):
    """A 2-region forward cortex: hub -> readout (random dense projection). The dendritic gain (if on)
    scales each hub's firing into the readout by its activity EMA."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="hub", n_neurons=n_hub, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="readout", n_neurons=n_readout, exc_fraction=1.0, internal_density=0.0),
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="hub", to_region="readout",
                      density=readout_density, weight_mean=1.0, weight_jitter=0.3),
    ]
    cfg.dt = 1.0
    cfg.seed = seed
    cfg.ou_seed = seed
    cfg.heterogeneity_seed = seed
    # the concept code must reflect the DRIVE-driven firing only -- spontaneous OU noise would give every
    # hub a baseline firing rate that swamps the per-hub co-occurrence marginal the gain normalizes by.
    cfg.enable_ou_process = False
    cfg.enable_dendritic_divisive_gain = enable_gain
    cfg.dendritic_divisive_sigma = sigma
    cfg.dendritic_gain_ema_alpha = alpha
    rt = RuntimeState()
    rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=rt, gpu_config=GPUConfig())
    bridge._initialize_simulation_data()
    return bridge, bridge.region_manager.indices("hub"), bridge.region_manager.indices("readout")


def _present(bridge, hub_idx, readout_idx, drive_vec, drive_scale, window, settle):
    """Drive the hubs with drive_vec (a concept's count pattern) for `window` steps; return the readout
    region's mean firing over the window (the code). `settle` lead-in steps let activity reach the readout."""
    import numpy as np
    hub_idx = np.asarray(hub_idx); readout_idx = np.asarray(readout_idx)
    xp = bridge._cp if hasattr(bridge, "_cp") else None
    drive = np.zeros(int(bridge.cp_membrane_potential_v.shape[0]), dtype=np.float64)
    drive[hub_idx] = np.asarray(drive_vec, dtype=np.float64) * drive_scale
    # set external current on the hubs
    bridge.cp_external_input_current[:] = 0.0
    if xp is not None:
        bridge.cp_external_input_current[hub_idx] = xp.asarray(drive[hub_idx].astype(np.float32))
    else:
        bridge.cp_external_input_current[hub_idx] = drive[hub_idx].astype(np.float32)
    acc = np.zeros(readout_idx.size, dtype=np.float64)
    nstep = 0
    for t in range(settle + window):
        bridge._run_one_simulation_step()
        if t >= settle:
            # read the readout's EXCITATORY CONDUCTANCE (the gain-normalized summed hub input -- the linear,
            # gain-affected dendritic quantity; the spiking-threshold read is a Phase-3 concern handled by
            # the dual/CLS cleanup). This is the direct measure of whether the gain shapes the code.
            ge = np.asarray(bridge.cp_conductance_g_e)[readout_idx].astype(np.float64)
            acc += ge
            nstep += 1
    bridge.cp_external_input_current[:] = 0.0
    return acc / max(1, nstep)


def _read_codes(bridge, hub_idx, readout_idx, C, drive_scale, window, settle, warmup_passes):
    """Warm-up (present all concepts so the hub EMAs converge to the marginals), then read each concept's
    readout code. Returns the [Nc x n_readout] code matrix."""
    Nc = C.shape[0]
    # warm-up: present every concept a few times so cp_dendritic_source_activity converges to each hub's
    # marginal (the divisor the gain normalizes by). No code read here.
    for _ in range(warmup_passes):
        for i in range(Nc):
            _present(bridge, hub_idx, readout_idx, C[i], drive_scale, window, settle)
    codes = np.zeros((Nc, np.asarray(readout_idx).size), dtype=np.float64)
    for i in range(Nc):
        codes[i] = _present(bridge, hub_idx, readout_idx, C[i], drive_scale, window, settle)
    return codes


def run_seed(seed, args):
    print(f"\n{'='*92}\n  D2 PHASE 2 -- DENDRITIC CORTEX FORWARD CODES (seed {seed})\n{'='*92}", flush=True)
    if args.real_corpus:
        from research.runners.option_c_stageB_fair_test import build_context_inclusive_cooccurrence
        from research.runners.option_c_real_cooccurrence_derisk import TAXONOMY_8x8, taxonomy_to_vocab_categories
        vocab, cat_ids, _ = taxonomy_to_vocab_categories(TAXONOMY_8x8)
        corpus = build_context_inclusive_cooccurrence(
            os.path.join(_REPO, "data", "corpus", "tinystories.txt"), vocab, cat_ids,
            window=2, n_context_hubs=args.n_hub, repeat_cap=40, seed=seed, verbose=False)
        # C restricted to TARGET rows x HUB columns (the concept x context-hub counts)
        from research.runners.learned_graded_embedding_diagnose import raw_count_matrix
        concepts = corpus["concepts"]; Nt = len(vocab)
        Cfull = raw_count_matrix(concepts, corpus["facts"])
        C = Cfull[:Nt, Nt:]                 # targets x hubs
        labels = np.asarray(cat_ids); S_true = corpus["S_true"]; n_hub = C.shape[1]
    else:
        C, labels, S_true, _ = build_concept_hub_counts(
            args.n_cat, args.per_cat, args.n_common, args.n_sig_per_cat,
            args.lam_common, args.lam_sig, args.lam_bg, seed)
        n_hub = C.shape[1]
    Nc = C.shape[0]
    host_sim = ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(args.host_svd, min(C.shape) - 1), alpha=args.host_alpha)
    host_pearson, _, _, _ = score(host_sim, labels)
    print(f"  {Nc} concepts x {n_hub} hubs; host PPMI ceiling={host_pearson:+.3f}; "
          f"n_readout={args.n_readout} drive_scale={args.drive_scale}", flush=True)

    # the DRIVE pattern: raw counts saturate the common hubs AND leave the rare category hubs below the
    # firing threshold (the diagnosed Phase-2 problem). PRESENCE drive (binary: does the concept co-occur
    # with the hub, count > threshold) puts ALL co-occurring hubs -- common AND rare -- above threshold so
    # they fire; the dendritic gain then down-weights the high-marginal common hubs, leaving the category
    # structure. (--raw-drive keeps the raw counts for comparison.)
    if args.drive_baseline > 0:
        # BASELINE-OFFSET RAW drive: co-occurring hubs (count > threshold) are driven by (count + baseline)
        # so the COMMON-MODE MAGNITUDE is kept (common count >> category) AND the low-count category hubs are
        # lifted above rheobase so they FIRE (readable). This is the regime where the gain's D1-level value
        # lives (common-mode-dominated) without the rheobase silencing the category signal -- the faithful
        # bridge analogue of D1's raw-count test.
        C_drive = np.where(C > args.presence_threshold, C + args.drive_baseline, 0.0)
    elif args.raw_drive:
        C_drive = C
    else:
        C_drive = (C > args.presence_threshold).astype(np.float64)

    def codes_for(enable_gain, sigma):
        bridge, hub_idx, ro_idx = _build_cortex_bridge(
            n_hub, args.n_readout, seed, enable_gain, sigma, args.alpha, args.readout_density)
        return _read_codes(bridge, hub_idx, ro_idx, C_drive, args.drive_scale, args.window, args.settle, args.warmup)

    # gain ON (dendritic) vs OFF (point-neuron control), identical pipeline
    t0 = time.time()
    dend_codes = codes_for(enable_gain=True, sigma=args.sigma)
    pn_codes = codes_for(enable_gain=False, sigma=args.sigma)
    dend_p = _pearson_vs_Strue(_cos_sim(dend_codes), S_true)
    pn_p = _pearson_vs_Strue(_cos_sim(pn_codes), S_true)
    dend_gen, chance = heldout_generalization(dend_codes, labels)
    pn_gen, _ = heldout_generalization(pn_codes, labels)
    dend_rank = effective_rank(dend_codes)
    dend_silent = float(np.mean(dend_codes.sum(1) == 0))
    print(f"  [DENDRITIC gain ON]  Pearson={dend_p:+.3f}  gen={dend_gen:.3f} (chance {chance:.3f})  "
          f"eff-rank={dend_rank:.1f}  silent={dend_silent:.2f}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"  [POINT-NEURON OFF]   Pearson={pn_p:+.3f}  gen={pn_gen:.3f}", flush=True)

    # anti-cheats
    rng = np.random.RandomState(seed * 2718281 + 3)
    S_perm = (rng.permutation(labels)[:, None] == rng.permutation(labels)[None, :]).astype(np.float64)
    dend_perm = _pearson_vs_Strue(_cos_sim(dend_codes), S_true if False else S_perm)
    lesion_codes = codes_for(enable_gain=True, sigma=1e6)   # gain ~1 -> collapses to point-neuron
    lesion_p = _pearson_vs_Strue(_cos_sim(lesion_codes), S_true)
    print(f"  [anti-cheat] permuted-S Pearson={dend_perm:+.3f} (~0)  lesion(sigma=1e6) Pearson={lesion_p:+.3f} "
          f"(-> ~point-neuron {pn_p:+.3f})", flush=True)

    point_neuron_fails = abs(pn_p) <= args.pn_fail_bar
    host_carries = host_pearson >= args.host_bar
    structure = (dend_p >= args.structure_bar) and point_neuron_fails and host_carries
    permuted_collapses = abs(dend_perm) <= args.pn_fail_bar
    lesion_collapses = lesion_p <= dend_p - 0.10
    not_collapsed = (dend_rank > 1.5) and (dend_silent < 0.5)
    gates = {"structure_contrast": bool(structure), "point_neuron_fails": bool(point_neuron_fails),
             "host_ceiling_carries": bool(host_carries), "permuted_collapses": bool(permuted_collapses),
             "lesion_collapses": bool(lesion_collapses), "not_collapsed": bool(not_collapsed)}
    print(f"  [seed {seed} gates] {gates}", flush=True)
    return {"seed": seed, "n_concepts": Nc, "n_hub": n_hub, "host_ceiling_pearson": host_pearson,
            "host_carries": bool(host_carries), "dend_pearson": dend_p, "pn_pearson": pn_p,
            "dend_gen": dend_gen, "pn_gen": pn_gen, "chance": chance,
            "permuted_pearson": dend_perm, "lesion_pearson": lesion_p, "gates": gates}


def decide_verdict(per_seed, seeds, args):
    def allg(k):
        return all(per_seed[str(s)]["gates"][k] for s in seeds)
    structure = allg("structure_contrast"); pn_fails = allg("point_neuron_fails")
    host_ok = allg("host_ceiling_carries")
    controls = allg("permuted_collapses") and allg("lesion_collapses")
    dmean = float(np.mean([per_seed[str(s)]["dend_pearson"] for s in seeds]))
    pmean = float(np.mean([per_seed[str(s)]["pn_pearson"] for s in seeds]))
    if not host_ok:
        verdict, why = "NEGATIVE_miscalibrated", "host ceiling did not carry on the counts -> re-tune."
    elif structure and controls:
        verdict = "GO"
        why = (f"the on-bridge dendritic gain produces a cortex whose FORWARD-PASS codes recover the "
               f"category structure (mean Pearson {dmean:+.3f}) WHILE the point-neuron (gain-off) control "
               f"fails ({pmean:+.3f}) on the identical pipeline, controls clean (permuted + lesion "
               f"collapse). The Option-C learn failure (-0.008) is FIXED by the dendritic substrate on the "
               f"real bridge -> Phase 2 PASSES; proceed to Phase 3 (the dual/CLS conversational gates).")
    elif dmean > pmean + 0.10 and controls:
        verdict, why = "BOUNDARY", (f"dendritic beats point-neuron (mean {dmean:+.3f} vs {pmean:+.3f}) but "
                                    f"below the structure bar -> partial; characterize before Phase 3.")
    else:
        verdict, why = "NEGATIVE", (f"the on-bridge gain does not beat the point neuron (dendritic "
                                    f"{dmean:+.3f} vs {pmean:+.3f}) -> the forward-pass realization needs "
                                    f"work (drive/EMA/projection); a decision-relevant Phase-2 finding.")
    return verdict, why, {"dend_pearson_mean": dmean, "pn_pearson_mean": pmean,
                          "structure_all": structure, "controls_all": controls, "host_all": host_ok}


def main():
    p = argparse.ArgumentParser(description="D2 Phase 2: dendritic cortex forward-pass codes")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--real-corpus", action="store_true", help="use the real Option-C TinyStories corpus")
    # synthetic toy (smoke)
    p.add_argument("--n-cat", type=int, default=8); p.add_argument("--per-cat", type=int, default=8)
    p.add_argument("--n-common", type=int, default=200); p.add_argument("--n-sig-per-cat", type=int, default=12)
    p.add_argument("--lam-common", type=float, default=40.0); p.add_argument("--lam-sig", type=float, default=4.0)
    p.add_argument("--lam-bg", type=float, default=0.3)
    p.add_argument("--n-hub", type=int, default=200, help="(real-corpus) number of context hubs")
    # cortex
    p.add_argument("--n-readout", type=int, default=400)
    p.add_argument("--readout-density", type=float, default=0.1)
    p.add_argument("--drive-scale", type=float, default=12.0, help="external current per unit count into the hubs")
    p.add_argument("--raw-drive", action="store_true",
                   help="drive hubs by RAW counts (saturates common hubs + silences rare ones -- the diagnosed problem); default = PRESENCE drive")
    p.add_argument("--presence-threshold", type=float, default=1.5,
                   help="count > this => the concept co-occurs with the hub (presence drive)")
    p.add_argument("--drive-baseline", type=float, default=0.0,
                   help="if >0: BASELINE-OFFSET RAW drive (count+baseline for co-occurring hubs) -- keeps the "
                        "common-mode magnitude while lifting category hubs above rheobase (the gain's regime)")
    p.add_argument("--window", type=int, default=20); p.add_argument("--settle", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2, help="warm-up passes over all concepts (EMA convergence)")
    # gain
    p.add_argument("--sigma", type=float, default=0.05); p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--host-svd", type=int, default=50); p.add_argument("--host-alpha", type=float, default=0.75)
    # bars
    p.add_argument("--structure-bar", type=float, default=0.25); p.add_argument("--pn-fail-bar", type=float, default=0.15)
    p.add_argument("--host-bar", type=float, default=0.30)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.smoke:
        os.environ.setdefault("SIM_BACKEND", "numpy")
        args.n_readout = 200; args.warmup = 2

    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    print(f"[D2 Phase 2 dendritic cortex] seeds={seeds} smoke={args.smoke} real_corpus={args.real_corpus} "
          f"backend={os.environ.get('SIM_BACKEND','auto')}", flush=True)
    per_seed = {str(s): run_seed(s, args) for s in seeds}
    verdict, why, detail = decide_verdict(per_seed, seeds, args)
    print(f"\n{'='*92}\n  D2 PHASE 2 VERDICT: {verdict}\n  {why}", flush=True)
    print(f"  ladder: DENDRITIC {detail['dend_pearson_mean']:+.3f}  vs  POINT-NEURON {detail['pn_pearson_mean']:+.3f}",
          flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n{'='*92}\n", flush=True)
    out = {"verdict": verdict, "why": why, "detail": detail, "seeds": seeds, "smoke": bool(args.smoke),
           "real_corpus": bool(args.real_corpus), "config": vars(args), "per_seed": per_seed,
           "note": ("D2 Phase 2: the on-bridge dendritic gain shapes the FORWARD-PASS cortex codes (hub -> "
                    "random readout projection). GO = the Option-C learn failure is fixed on the real "
                    "bridge. NO new sim/ edits (uses the Phase-1 gain).")}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(raw_dir, f"_dendritic_cortex_forward_{'smoke' if args.smoke else 'multiseed'}_{ts}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
