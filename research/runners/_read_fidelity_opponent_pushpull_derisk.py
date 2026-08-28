"""READ-FIDELITY de-risk, ITERATION 5 -- the RANKED-FIRST lever iteration 4 earned: an OPPONENT / PUSH-PULL
spiking read, on the surprise->source_provenance F2 crux.

BANKED SO FAR:
  iteration 1 (`_read_fidelity_nonrate_latency_derisk.py`,
    research/findings/2026-08-28-read-fidelity-nonrate-latency-UNDEFINED.md): first-spike LATENCY, UNDEFINED.
  iteration 2 (`_read_fidelity_nonrate_latency_dispersion_derisk.py`,
    research/findings/2026-08-28-read-fidelity-dispersion-instrument-fixed-still-NO-read-beats-rate.md):
    instrument fixed (permutation-over-neuron-identity), LATENCY 0/6, DISPERSION(ISI-CV) 1/6.
  iteration 3 (`_read_fidelity_decoder_separability_derisk.py`,
    research/findings/2026-08-28-read-fidelity-decoder-SIGNAL-FOUND-its-a-read-limit-not-wiring.md): a
    linear+nonlinear DECODER over the full per-neuron 10-bin spike-count profile separates generated-vs-
    perceived 6/6 (shuffle-clean, both weight conditions) -- the signal EXISTS in the distributed pattern.
  iteration 4 (`_read_fidelity_popvec_template_derisk.py`,
    research/findings/2026-08-28-read-fidelity-popvec-template-biological-read-NOGO-power-gap-not-signal-gap.md):
    a population-vector / matched-filter TEMPLATE (RECTIFIED to a SINGLE non-negative channel, per-bin gain
    only, Dale's-law-motivated) driving one LIF readout is NO-GO 0/6 (per-seed primary z =
    [0.05, 0.18, 0.04, 1.13, 0.09, 1.32]). Its own diagnosis: this is a READ-POWER gap, not a signal gap -- the
    single rectified channel throws away every bin where the PERCEIVED pool discriminates more than the
    GENERATED pool (clip(mu_gen-mu_perc, 0, None) zeros those bins outright), which is exactly the sign
    (negative-weight) information an unconstrained linear/MLP decoder is free to use and this template read
    is not. THE RANKED-FIRST NEXT LEVER (this file, per that finding's own ranking, over the dendritic-
    nonlinear alternative -- an internal precedent, the 2026-08-25 vision-2layer-granule-expansion finding,
    already showed nonlinearity alone does not reliably lift a linear ceiling on a related crux): recover the
    discarded sign information via a biologically MORE faithful mechanism, not a less faithful one -- an
    OPPONENT / PUSH-PULL population read.

THE DESIGN -- an OPPONENT (push-pull) spiking readout, the biological answer to "the template can't have signed
weights" that a REAL cortical neuron actually uses: excitation and inhibition are BOTH always-non-negative-
strength synaptic populations (Dale's law: no single synapse changes sign), and the SIGN of their combined
effect on the postsynaptic neuron comes from the two populations' PUSH (excitatory, depolarizing) vs PULL
(inhibitory, hyperpolarizing) opposition, not from any one synapse flipping polarity. This is not a house
hypothesis -- it is the literal, decades-characterized circuit motif of cat V1 simple cells (Hirsch, Alonso,
Reid & Martinez 1998, J Neurosci 18(22):9517-9528, "Synaptic integration in striate cortical simple cells",
PMID 9801388, doi:10.1523/JNEUROSCI.18-22-09517.1998; whole-cell recordings show a simple cell's response sign
at any instant is set by the antagonism between a PUSH excitatory conductance and a PULL inhibitory
conductance, "push-pull", never by an excitatory synapse turning inhibitory or vice versa) -- and the same
motif recurs across cat V1 layers/circuitry (Troyer, Krukowski, Priebe & Miller 1998 J Neurosci 18(15):5908-27
PMID 9671678; Martinez et al. 2005 Nat Neurosci 8(3):372-9 PMID 15711543; Taylor, Sedigh-Sarvestani, Vigeland,
Palmer & Contreras 2017 J Neurosci 38(3):595-612 PMID 29196320). Applying this motif here: iteration 4's single
rectified template `raw = clip(mu_gen-mu_perc, 0, None)/pooled_std` is exactly the PUSH (excitatory) half of a
push-pull pair. This run adds its missing PULL (inhibitory) half: `template_I = clip(mu_perc-mu_gen, 0,
None)/pooled_std` -- the MAGNITUDE of the bins iteration 4 discarded, delivered as a second, separately
non-negative INHIBITORY channel. Both channels individually obey Dale's law (template_E >= 0 always;
template_I >= 0 always -- neither is ever negative, so no single synaptic population ever flips excitatory/
inhibitory identity). The readout LIF neuron's own membrane integrates their NET drive, current_E - current_I,
exactly the way a real push-pull simple cell's membrane potential reflects the balance of its excitatory and
inhibitory conductances -- algebraically this reconstructs the FULL SIGNED template (template_E - template_I
== the un-rectified `raw`) that iteration 4's own docstring "considered and rejected" as a SINGLE signed
synapse, but now realized as two Dale's-law-faithful non-negative populations whose ANTAGONISM (not any one
synapse's sign-flip) carries the recovered information -- the same distinction the offline decoder's arbitrary
signed weights could exploit for free and the single-channel template could not.

WHY THE SAMPLE AXIS, THE STRATIFIED CV, THE CAPTURED RASTERS, THE ANTI-CHEAT ARE ALL UNCHANGED (inherited, not
re-derived here): iteration 2/3/4 already established (a) this pool family has no independent trials to
resample (fixed reset + fixed drive, `enable_ou_process=False`, `enable_short_term_plasticity=False`), so the
genuine resampling axis is NEURON IDENTITY via stratified k-fold CV, not read repetition; (b) the SAME trained
cross-edge + SAME captured rasters iterations 1-4 used must be reused verbatim to avoid a retraining confound;
(c) the neuron-identity permutation-null anti-cheat (K_SHUF draws, Z_FLOOR=2.0, SHUF_COLLAPSE_MAX_RATE=0.15) is
the validated instrument-validity bar. This file changes EXACTLY ONE thing versus iteration 4: the single
rectified matched-filter channel becomes an opponent (excitatory-push + inhibitory-pull) PAIR of channels
feeding the SAME LIF readout via net current -- everything else (pool build/train/lesion, raster capture,
binned features, CV fold machinery, LIF readout model, threshold calibration protocol, anti-cheat shape, gate
definition) is reused VERBATIM so this is a clean single-variable A/B against iteration 4.

THE GATE (pre-registered before any seed ran, UNCHANGED from iteration 4): PRIMARY = `delta_held_base` (the
cross-edge-attributable feature): intact real_margin_mean > 0 AND z >= Z_FLOOR AND shuffle_collapses AND the
margin is lesion-attributable (|lesion mean| < F2_LESION_RATIO * |intact mean|). `raw_held` is reported for
comparability but is NOT gating. GO requires the PRIMARY gate PASS on every one of 6 seeds.

DE-RISK ONLY -- no production wiring, no `sim/` edit, no default flip. Subclasses NOTHING new: reuses
`ReadFidelityPool` (build/train/lesion + the raster-capturing `_drive2`) VERBATIM from the committed iteration-1
file, `_capture_reads` VERBATIM from the committed iteration-2 file, `_binned_features`/`_avg_binned`/
`_stratified_folds`/`_seed_trap_check` VERBATIM from the committed iteration-3 file, and
`_expand_template_to_steps`/`_lif_batch_spike_counts`/`_calibrate_v_th`/`_arm_verdict` VERBATIM from the
committed iteration-4 file -- SAME trained cross-edge, SAME captured rasters, SAME CV machinery, SAME LIF
readout model, SAME verdict logic, no retraining confound. numpy CPU throughout; pool-runnable.

Run:
  SIM_BACKEND=numpy python -m research.runners._read_fidelity_opponent_pushpull_derisk --seeds 42 --smoke
  SIM_BACKEND=numpy python -m research.runners._read_fidelity_opponent_pushpull_derisk \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_read_fidelity_opponent_pushpull_derisk_6seed.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU only -- never touch the GPU
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sim.backend import to_host
from tools.lab import attributable_to
from research.runners._read_fidelity_nonrate_latency_derisk import (
    ReadFidelityPool, RECALL_STEPS, N_READS, PRE_STEPS, EPISODE_DRIVE_PA,
)
from research.runners._read_fidelity_nonrate_latency_dispersion_derisk import _capture_reads
from research.runners._read_fidelity_decoder_separability_derisk import (
    _binned_features, _avg_binned, _stratified_folds, _seed_trap_check,
)
from research.runners._onebrain_integration_surprise_episodic_crossedge import (
    F2_LESION_RATIO, CROSS_EDGE_LR, N_EPISODES, HMAX, CUE_PA, CTX_DRIVE_PA, _build_pool,
)
from research.runners._read_fidelity_popvec_template_derisk import (
    _expand_template_to_steps, _lif_batch_spike_counts, _calibrate_v_th, _arm_verdict,
)

# ---- this run's own pre-registered constants (declared BEFORE any measurement) ----
# ALL identical to iteration 4 -- the only changed variable is the template mechanism itself (below), not any
# of these knobs, so this is a clean single-variable A/B against the popvec/template NO-GO.
N_BINS = 10
Z_FLOOR = 2.0
SHUF_COLLAPSE_MAX_RATE = 0.15
K_FOLDS = 5
R_REPEATS = 5
K_SHUF = 20
R_REPEATS_SMOKE = 2
K_SHUF_SMOKE = 5
TAU_MEM_STEPS = 5.0
V_RESET = 0.0
CALIB_TARGET_FRAC = 0.5
V_TH_FLOOR = 1e-6
EPS = 1e-9
WEIGHT_CONDS = ("intact", "lesion")
FEAT_KINDS = ("raw_held", "delta_held_base")


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  The OPPONENT (push-pull) templates -- TWO Dale's-law-respecting, always-non-negative channels
#  (fit on TRAIN neurons only), where iteration 4 fit ONE rectified channel
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _opponent_templates(profiles_train, labels_train_bool):
    """profiles_train: (n_train, n_bins) per-neuron binned feature. labels_train_bool: (n_train,) bool, True
    = generated-pool. Returns (template_E, template_I), each a (n_bins,) NEVER-NEGATIVE array:
      template_E = clip(mu_gen - mu_perc, 0, None) / pooled_std   -- the PUSH (excitatory) channel, IDENTICAL
        to iteration 4's single rectified template (unchanged formula).
      template_I = clip(mu_perc - mu_gen, 0, None) / pooled_std   -- the PULL (inhibitory) OPPONENT channel:
        the MAGNITUDE of exactly the bins iteration 4's clip(...,0,None) zeroed out (where the PERCEIVED pool
        discriminates more than the GENERATED pool), now delivered as a SEPARATE non-negative channel instead
        of being discarded.
    Neither channel is ever negative (Dale's law preserved WITHIN each channel) -- the sign information is
    recovered by having TWO populations of opposite fixed polarity, not by any one synapse changing sign."""
    gen = profiles_train[labels_train_bool]
    perc = profiles_train[~labels_train_bool]
    assert gen.shape[0] >= 2 and perc.shape[0] >= 2, "need >=2 train neurons per class for a defined std"
    mu_gen, mu_perc = gen.mean(axis=0), perc.mean(axis=0)
    sd_gen, sd_perc = gen.std(axis=0, ddof=1), perc.std(axis=0, ddof=1)
    pooled_sd = 0.5 * (sd_gen + sd_perc)
    raw = (mu_gen - mu_perc) / (pooled_sd + EPS)
    template_E = np.clip(raw, 0.0, None)
    template_I = np.clip(-raw, 0.0, None)
    return template_E, template_I


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  One fold: fit the OPPONENT templates on TRAIN neurons, drive the spiking readout with the
#  TEST fold's own raw spikes via NET (excitatory-minus-inhibitory) synaptic current
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _fold_margin_opponent(raster_pairs, template_E_bins, template_I_bins, train_idx, test_idx, labels_bool,
                           steps, n_bins, feat_kind):
    """raster_pairs: N_READS list of (raster_base, raster_held), each (steps, n_union) bool -- the SAME
    already-captured rasters iteration 1-4 use, no new simulation. Returns (fold margin, calibrated v_th)."""
    tE_steps = _expand_template_to_steps(template_E_bins, steps, n_bins)
    tI_steps = _expand_template_to_steps(template_I_bins, steps, n_bins)
    n_union = labels_bool.size
    test_gen = np.zeros(n_union, dtype=bool); test_gen[test_idx] = labels_bool[test_idx]
    test_perc = np.zeros(n_union, dtype=bool); test_perc[test_idx] = ~labels_bool[test_idx]
    train_gen = np.zeros(n_union, dtype=bool); train_gen[train_idx] = labels_bool[train_idx]
    train_perc = np.zeros(n_union, dtype=bool); train_perc[train_idx] = ~labels_bool[train_idx]

    def _pop_signal(raster, gen_mask, perc_mask):
        # structural pool membership (fixed at pool-build time, independent of any read -- not leakage,
        # identical convention to iteration 4): generated-pool neurons contribute +1/spike, perceived-pool -1
        return (raster[:, gen_mask].sum(axis=1).astype(np.float64)
                - raster[:, perc_mask].sum(axis=1).astype(np.float64))

    def _net_current(raster, gen_mask, perc_mask):
        sig = _pop_signal(raster, gen_mask, perc_mask)
        I_push = tE_steps * sig    # EXCITATORY synaptic drive -- fixed non-negative gain (Dale's law)
        I_pull = tI_steps * sig    # INHIBITORY (opponent) synaptic drive -- ALSO a fixed non-negative gain;
                                    # it is the readout neuron's OWN membrane that subtracts it, exactly as a
                                    # real postsynaptic neuron sums an excitatory and an inhibitory conductance
        return I_push - I_pull     # net current into the readout neuron's membrane

    # threshold calibration from the TRAIN fold's own HELD-read net currents only (no test leakage)
    train_I = np.stack([_net_current(rh, train_gen, train_perc) for (_rb, rh) in raster_pairs])
    v_th = _calibrate_v_th(train_I, TAU_MEM_STEPS)

    held_I = np.stack([_net_current(rh, test_gen, test_perc) for (_rb, rh) in raster_pairs])
    base_I = np.stack([_net_current(rb, test_gen, test_perc) for (rb, _rh) in raster_pairs])
    n_reads = held_I.shape[0]
    counts = _lif_batch_spike_counts(np.concatenate([held_I, base_I], axis=0), TAU_MEM_STEPS, v_th)
    held_counts, base_counts = counts[:n_reads], counts[n_reads:]
    if feat_kind == "raw_held":
        return float(held_counts.mean()), float(v_th)
    return float((held_counts - base_counts).mean()), float(v_th)


def _one_cv_pass_opponent(labels_bool, rng, profiles_held, profiles_base, raster_pairs, feat_kind, steps, n_bins):
    """ONE stratified K_FOLDS-fold CV pass: fit the OPPONENT (push-pull) templates on each fold's TRAIN
    neurons, evaluate the spiking readout's net current on that fold's TEST neurons, average the per-fold
    margins. `labels_bool` may be the REAL generated/perceived labeling or a shuffled (anti-cheat) relabeling
    -- identical pipeline either way. Same CV shape as iteration 3/4's own `_one_cv_pass`."""
    prof = profiles_held if feat_kind == "raw_held" else (profiles_held - profiles_base)
    margins = []
    for train_idx, test_idx in _stratified_folds(labels_bool, K_FOLDS, rng):
        template_E, template_I = _opponent_templates(prof[train_idx], labels_bool[train_idx])
        m, _v_th = _fold_margin_opponent(raster_pairs, template_E, template_I, train_idx, test_idx,
                                          labels_bool, steps, n_bins, feat_kind)
        margins.append(m)
    return float(np.mean(margins))


def _combo_stats_opponent(raster_pairs, profiles_held, profiles_base, n_gen, n_perc, feat_kind, rng, repeats, k_shuf):
    """Real (REPEATS-repeated CV) vs a K_SHUF-draw neuron-identity permutation null -- IDENTICAL statistical
    shape to iteration 3/4's own `_combo_stats` (same Z_FLOOR, same SHUF_COLLAPSE_MAX_RATE, same anti-cheat
    definition), calling into the opponent (push-pull) CV pass instead of the single-channel one."""
    n_union = n_gen + n_perc
    real_label = np.zeros(n_union, dtype=bool); real_label[:n_gen] = True
    real_vals = np.array([_one_cv_pass_opponent(real_label, rng, profiles_held, profiles_base, raster_pairs,
                                                 feat_kind, RECALL_STEPS, N_BINS) for _ in range(repeats)])
    null_vals = []
    for _ in range(k_shuf):
        perm = rng.permutation(n_union)
        y_shuf = np.zeros(n_union, dtype=bool); y_shuf[perm[:n_gen]] = True
        null_vals.append(_one_cv_pass_opponent(y_shuf, rng, profiles_held, profiles_base, raster_pairs,
                                                feat_kind, RECALL_STEPS, N_BINS))
    null_vals = np.asarray(null_vals, dtype=np.float64)

    real_mean = float(real_vals.mean())
    real_std = float(real_vals.std(ddof=1)) if real_vals.size > 1 else 0.0
    null_mean = float(null_vals.mean())
    null_std = float(null_vals.std(ddof=1)) if null_vals.size > 1 else 0.0
    z = (real_mean - null_mean) / null_std if null_std > 0 else (float("inf") if real_mean != null_mean else 0.0)
    frac_null_clears = (float(np.mean(np.abs((null_vals - null_mean) / null_std) >= Z_FLOOR))
                         if null_std > 0 else float("nan"))
    shuffle_collapses = bool(frac_null_clears <= SHUF_COLLAPSE_MAX_RATE) if not np.isnan(frac_null_clears) else False
    return {
        "feat_kind": feat_kind, "real_margin_mean": real_mean, "real_margin_std": real_std,
        "real_margin_all": [float(x) for x in real_vals],
        "null_margin_mean": null_mean, "null_margin_std": null_std,
        "z": float(z), "n_repeats": int(repeats), "n_shuf": int(k_shuf),
        "frac_null_clears_floor": frac_null_clears, "shuffle_collapses": shuffle_collapses,
    }


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Per-seed run
# ─────────────────────────────────────────────────────────────────────────────────────────────
def run_seed(seed, repeats, k_shuf):
    t0 = time.time()
    pool = ReadFidelityPool(seed)
    traj = pool.train()
    emg_grew = bool(traj[-1]["w"] > 5 * 0.05)
    emg_specific = bool(abs(traj[-1]["w_other"] - 0.05) < 0.03)

    ix = pool.ix
    n_gen = int(ix["prov_generated"].size)
    n_perc = int(ix["prov_perceived"].size)
    union = np.concatenate([ix["prov_generated"], ix["prov_perceived"]])
    read_dict = {"gen": ix["prov_generated"], "perc": ix["prov_perceived"]}

    # ---- INTACT ----
    pairs_intact = _capture_reads(pool, read_dict, union)
    held_i = _avg_binned([h for (_b, h) in pairs_intact], N_BINS)
    base_i = _avg_binned([b for (b, _h) in pairs_intact], N_BINS)

    # ---- LESIONED (same event as iteration 1/2/3/4: zero surprise->provgen, in place) ----
    data = np.asarray(to_host(pool.b.cp_connections.data)).copy()
    data[pool.masks["surprise->provgen"]] = 0.0
    pool.b.cp_connections.data = pool.xp.asarray(data, dtype=pool.b.cp_connections.data.dtype)
    pairs_lesion = _capture_reads(pool, read_dict, union)
    held_l = _avg_binned([h for (_b, h) in pairs_lesion], N_BINS)
    base_l = _avg_binned([b for (b, _h) in pairs_lesion], N_BINS)

    feats = {"intact": (pairs_intact, held_i, base_i), "lesion": (pairs_lesion, held_l, base_l)}

    base_off = int(seed) * 179424673 + 337   # this module's own distinct RNG offset family (sibling offsets
                                               # already in use: *104729+17, *65599+41, *7919+101, *997+3,
                                               # *15485863+271, *50331653+191 -- none collide)
    combos = {}
    combo_i = 0
    for cond in WEIGHT_CONDS:
        raster_pairs, held_p, base_p = feats[cond]
        for feat_kind in FEAT_KINDS:
            rng = np.random.default_rng(base_off + combo_i * 8237)
            combo_i += 1
            combos[f"{cond}__{feat_kind}"] = _combo_stats_opponent(
                raster_pairs, held_p, base_p, n_gen, n_perc, feat_kind, rng, repeats, k_shuf)

    # Make the intact-vs-lesion attribution question EXPLICIT in this file (not only inside the imported
    # `_arm_verdict`, which already computes it internally on iteration 4's identical intact/lesion pair --
    # gap#5's own lesson: a treatment/control pair sitting one key apart in the same JSON must be SUBTRACTED,
    # visibly, in the file that owns the new mechanism, not just trusted to a cross-file import).
    attributable_to(
        "F2 opponent/push-pull spiking-readout margin (delta_held_base) -- intact vs lesioned cross-edge",
        combos["intact__delta_held_base"]["real_margin_mean"],
        combos["lesion__delta_held_base"]["real_margin_mean"])
    primary = _arm_verdict(combos["intact__delta_held_base"], combos["lesion__delta_held_base"],
                            "F2 opponent/push-pull spiking-readout margin (delta_held_base, cross-edge "
                            "attributable)")
    secondary = _arm_verdict(combos["intact__raw_held"], combos["lesion__raw_held"],
                              "F2 opponent/push-pull spiking-readout margin (raw_held, NOT gating -- may "
                              "include static structural asymmetry per iteration 3's own flag)")
    n_combos_shuffle_collapse = sum(c["shuffle_collapses"] for c in combos.values())

    return {
        "seed": int(seed), "elapsed_s": round(time.time() - t0, 1),
        "cue_concept": pool.cue_c, "assert_concept": pool.assert_cp,
        "final_weight_trained_block": float(traj[-1]["w"]), "final_weight_other_blocks": float(traj[-1]["w_other"]),
        "emergence_grew_from_near_zero": emg_grew, "emergence_other_blocks_stayed_near_seed": emg_specific,
        "n_gen": n_gen, "n_perc": n_perc,
        "combos": combos,
        "PRIMARY_delta_held_base": primary,
        "SECONDARY_raw_held_not_gating": secondary,
        "n_combos_shuffle_collapse": int(n_combos_shuffle_collapse), "n_combos": len(combos),
        "PASS": bool(primary["PASS"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--smoke", action="store_true", help="1 seed, lighter CV/shuffle budget")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]
    repeats = R_REPEATS_SMOKE if args.smoke else R_REPEATS
    k_shuf = K_SHUF_SMOKE if args.smoke else K_SHUF

    seed_trap = _seed_trap_check(seeds[0])
    print(f"[seed-trap] build-twice at seed={seeds[0]}: identical={seed_trap['identical']} "
          f"n_neurons={seed_trap['n_neurons']} hash={seed_trap['hash_build1']}", flush=True)

    runs = []
    for s in seeds:
        r = run_seed(s, repeats, k_shuf)
        runs.append(r)
        p = r["PRIMARY_delta_held_base"]
        c = r["combos"]["intact__delta_held_base"]
        print(f"[seed {s}] ({r['elapsed_s']}s) block(c={r['cue_concept']},c'={r['assert_concept']}) "
              f"w={r['final_weight_trained_block']:.2f} w_other={r['final_weight_other_blocks']:.3f} | "
              f"PRIMARY(delta_held_base) real={c['real_margin_mean']:.3f} null={c['null_margin_mean']:.3f} "
              f"z={c['z']:.2f} floor={p['floor_ok']} lesion_ok={p['lesion_ok']} "
              f"shuf_collapse={p['shuffle_collapses_intact']} PASS={p['PASS']} | "
              f"n_shuf_ok={r['n_combos_shuffle_collapse']}/{r['n_combos']}", flush=True)

    n_pass = sum(r["PASS"] for r in runs)
    n_shuf_ok = sum(r["n_combos_shuffle_collapse"] == r["n_combos"] for r in runs)
    all_go_raw = bool(n_pass == len(runs)) and not args.smoke

    dec, preconditions = None, []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("read_fidelity_opponent_pushpull_derisk")
        Vd.require("shuffle_anticheat_collapses_on_every_combo",
                   1 if all(r["n_combos_shuffle_collapse"] == r["n_combos"] for r in runs) else 0,
                   expect=lambda x: x >= 1,
                   note="the neuron-identity permutation null must individually clear Z_FLOOR on <= "
                        f"{SHUF_COLLAPSE_MAX_RATE} of its own draws, on EVERY combo, before the opponent "
                        "readout's verdict can be trusted -- same instrument-validity bar as iteration 2/3/4")
        Vd.require("emergence_grew_from_near_zero", 1 if all(r["emergence_grew_from_near_zero"] for r in runs) else 0,
                   expect=lambda x: x >= 1, note="the reused cross-edge trained normally (sanity on the shared substrate)")
        Vd.require("anti_cheat_random_assignment", 1 if len(set((r["cue_concept"], r["assert_concept"])
                   for r in runs)) > 1 else 0, expect=lambda x: x >= 1,
                   note="the per-seed block pair must actually vary (inherited from the parent runner)")
        dec = Vd.decide(all_go_raw, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    verdict_status = dec.get("status") if dec else None
    go = all_go_raw if dec is None else bool(dec.get("go"))
    if verdict_status == "UNDEFINED":
        tag = "UNDEFINED"
    elif args.smoke:
        tag = f"SMOKE ({'PASS' if runs[0]['PASS'] else 'NO-GO'}, 1-seed indicator)"
    else:
        tag = "GO -- the opponent/push-pull spiking readout CLEARS the crux" if all_go_raw else "NO-GO/PARTIAL"

    # per-seed z-shift vs iteration 4's own per-seed primary z (for the exact 6 pre-registered seeds, in order)
    ITERATION4_Z = {42: 0.05, 43: 0.18, 44: 0.04, 100: 1.13, 101: 0.09, 102: 1.32}
    z_shifts = []
    for r in runs:
        it4z = ITERATION4_Z.get(r["seed"])
        z_now = r["combos"]["intact__delta_held_base"]["z"]
        z_shifts.append({"seed": r["seed"], "z_iteration4": it4z, "z_opponent": z_now,
                          "shift": (None if it4z is None else round(z_now - it4z, 4))})

    verdict = (f"{tag}. A downstream spiking READOUT NEURON (LIF, threshold-calibrated from TRAIN data only) "
               f"whose synaptic drive is the NET of TWO Dale's-law-respecting, always-non-negative OPPONENT "
               f"channels (Hirsch, Alonso, Reid & Martinez 1998, J Neurosci 18(22):9517-9528, PMID 9801388: "
               f"cortical simple-cell push-pull -- excitation minus inhibition, never a single sign-flipping "
               f"synapse) fit on a TRAIN fold of neurons (stratified K_FOLDS={K_FOLDS}-fold CV over neuron "
               f"identity, R_REPEATS={repeats}) and evaluated on the HELD-OUT test fold's own raw spike "
               f"trains, on the SAME trained cross-edge and SAME captured rasters iterations 1-4 used (no "
               f"retraining confound). PRIMARY gate = delta_held_base (cross-edge-attributable component): "
               f"{n_pass}/{len(runs)} seeds PASS (z>=Z_FLOOR={Z_FLOOR} AND lesion-attributable AND shuffle "
               f"anti-cheat collapses). Anti-cheat: K_SHUF={k_shuf}-draw neuron-identity permutation null, "
               f"shuffle collapses on {sum(r['n_combos_shuffle_collapse'] for r in runs)}/"
               f"{sum(r['n_combos'] for r in runs)} combo-seed pairs. Per-seed z-shift vs iteration 4's single-"
               f"channel template: {z_shifts}."
               + (f" UNDEFINED, NOT a validated verdict either way: {len(dec.get('undefined_reasons', []))} "
                  f"precondition(s) unmet -- {'; '.join(dec.get('undefined_reasons', []))}."
                  if verdict_status == "UNDEFINED" else ""))

    payload = {
        "probe": "read_fidelity_opponent_pushpull_derisk", "verdict": verdict, "GO": go,
        "n_seeds": len(runs), "n_seeds_pass_primary": n_pass, "n_seeds_shuffle_ok": n_shuf_ok,
        "seeds": seeds, "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
        "preconditions": preconditions,
        "seed_trap_build_twice": seed_trap,
        "z_shift_vs_iteration4": z_shifts,
        "config": {
            "n_bins": N_BINS, "z_floor": Z_FLOOR, "shuf_collapse_max_rate": SHUF_COLLAPSE_MAX_RATE,
            "k_folds": K_FOLDS, "r_repeats": repeats, "k_shuf": k_shuf,
            "tau_mem_steps": TAU_MEM_STEPS, "v_reset": V_RESET, "calib_target_frac": CALIB_TARGET_FRAC,
            "v_th_floor": V_TH_FLOOR,
            "weight_conds": list(WEIGHT_CONDS), "feat_kinds": list(FEAT_KINDS),
            "recall_steps": RECALL_STEPS, "n_reads": N_READS, "pre_steps": PRE_STEPS,
            "episode_drive_pa": EPISODE_DRIVE_PA, "f2_lesion_ratio": F2_LESION_RATIO,
            "cross_edge_hebbian_lr": CROSS_EDGE_LR, "n_episodes": N_EPISODES,
            "hebbian_max_weight": HMAX, "cue_pa": CUE_PA, "ctx_drive_pa": CTX_DRIVE_PA,
            "rng_formula": "seed*179424673+337, +combo_index*8237 per combo (this module's own distinct offset "
                            "family; does not collide with _assign_blocks(*104729+17), _shuffle_mask(*65599+41), "
                            "iteration-2's permutation rng(*7919+101), _make_ambiguous_pattern(*997+3), "
                            "iteration-3's rng(*15485863+271, +combo*104651), or iteration-4's "
                            "rng(*50331653+191, +combo*7247)",
        },
        "mechanism": ("Reuses ReadFidelityPool (build/train/lesion + the raster-capturing `_drive2`) VERBATIM "
                      "from iteration-1, `_capture_reads` VERBATIM from iteration-2, "
                      "`_binned_features`/`_avg_binned`/`_stratified_folds`/`_seed_trap_check` VERBATIM from "
                      "iteration-3, and `_expand_template_to_steps`/`_lif_batch_spike_counts`/"
                      "`_calibrate_v_th`/`_arm_verdict` VERBATIM from iteration-4. Changes ONLY the template: "
                      "instead of iteration 4's single RECTIFIED matched-filter channel (Fisher-style, clipped "
                      "at 0), fits TWO always-non-negative channels on the TRAIN fold -- template_E = "
                      "clip(mu_gen-mu_perc, 0, None)/pooled_std (PUSH/excitatory, identical to iteration 4's "
                      "own channel) and template_I = clip(mu_perc-mu_gen, 0, None)/pooled_std (PULL/inhibitory "
                      "OPPONENT channel: the magnitude of exactly the bins iteration 4 discarded). The LIF "
                      "readout's input current is now the NET of the two channels' contributions "
                      "(I_push - I_pull) applied to the TEST fold's own raw per-step raster, signed by each "
                      "TEST neuron's known (structural, non-leaked) pool membership -- algebraically "
                      "reconstructing the FULL SIGNED template (template_E - template_I == the un-rectified "
                      "mean-difference) iteration 4 rejected as a single sign-flipping synapse, now realized "
                      "as two Dale's-law-faithful non-negative populations whose antagonism carries the sign."),
        "biology": ("Hirsch, Alonso, Reid & Martinez 1998, J Neurosci 18(22):9517-9528, 'Synaptic integration "
                    "in striate cortical simple cells', PMID 9801388, doi:10.1523/JNEUROSCI.18-22-9517.1998 -- "
                    "whole-cell recordings establishing that a cat V1 simple cell's response SIGN at any "
                    "instant is set by the antagonism (push-pull) between a separate excitatory (push) and "
                    "inhibitory (pull) synaptic drive, never by any single synapse changing polarity; the "
                    "circuit-level basis for how the SAME sign information a signed linear decoder can use for "
                    "free is instead realized biologically via two opposite-polarity populations. Confirmed "
                    "as a recurring cortical motif (not a one-off): Troyer, Krukowski, Priebe & Miller 1998, J "
                    "Neurosci 18(15):5908-5927, PMID 9671678 (correlation-based push-pull circuitry); Martinez "
                    "et al. 2005, Nat Neurosci 8(3):372-379, PMID 15711543 (push-pull receptive-field "
                    "structure across V1 layers); Taylor, Sedigh-Sarvestani, Vigeland, Palmer & Contreras 2017, "
                    "J Neurosci 38(3):595-612, PMID 29196320 (push-pull inhibition spanning the receptive "
                    "field). Retains Georgopoulos, Schwartz & Kettner 1986 (Science 233:1416-1419) and Salinas "
                    "& Abbott 1994 (J Neurosci 14(9):5667-5680) for the underlying population-vector/matched-"
                    "filter readout formalism, unchanged from iteration 4 -- this run's addition is ONLY the "
                    "opponent (two-channel, Dale's-law-respecting) realization of that filter's sign."),
        "scaffold_residuals": [
            "each opponent channel is INDIVIDUALLY rectified (Dale's law preserved per-channel), but the "
            "readout neuron's own membrane sums their NET (push-minus-pull) drive with UNCONSTRAINED "
            "arithmetic subtraction -- a real postsynaptic neuron's E/I balance is a conductance-based "
            "computation (driving-force-dependent, not a bare current subtraction); this is a current-based "
            "LIF simplification, same class of simplification iteration 4 already carried",
            "per-bin (diagonal) matched-filter normalization (pooled std per bin) for EACH channel "
            "separately, not a full covariance-whitened Fisher LDA across bins -- unchanged from iteration 4",
            "the LIF readout uses INSTANTANEOUS current injection per spike (no synaptic rise/decay "
            "kinetics) for EITHER channel -- a standard simplified LIF, not conductance-based synapses",
            "TAU_MEM_STEPS/CALIB_TARGET_FRAC/K_FOLDS/R_REPEATS/K_SHUF are host-chosen "
            "statistical-power/dynamics knobs, not computed features, UNCHANGED from iteration 4 (kept "
            "identical deliberately so the ONLY changed variable is the opponent-vs-single-channel template)",
            "N_READS=8 reads averaged into the per-combo margin (denoising only, per this pool family's "
            "deterministic dynamics); the genuine per-seed resampling axis is neuron identity via the "
            "K_FOLDS/R_REPEATS/K_SHUF draws, not read repetition -- unchanged from iteration 2-4",
            "the readout is trained/tested on NEURON IDENTITY folds, not independent TRIALS (this pool "
            "family has no independent trials to split) -- unchanged, inherited constraint",
            "same host-curated training schedule / topology as the parent crossedge runner (declared there, "
            "unchanged)",
            "the two channels' RELATIVE gain (push vs pull) is fixed 1:1 by construction (I_push - I_pull, "
            "no learned or biologically-derived E/I balance ratio) -- a genuinely LEARNED opponent gain (e.g. "
            "via a local delta/perceptron-style rule on each channel's overall scale) is the next estimator-"
            "power lever if this one under- or over-shoots",
        ],
        "runs": runs,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[READ-FIDELITY OPPONENT PUSH-PULL] VERDICT: {verdict}\n" + "=" * 100, flush=True)
    return 0 if (payload["GO"] or args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
