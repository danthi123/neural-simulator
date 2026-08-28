"""READ-FIDELITY de-risk, ITERATION 4 -- the lever iteration 3 earned: a BIOLOGICAL SPIKING read over the
DISTRIBUTED per-neuron pattern, on the surprise->source_provenance F2 crux.

BANKED SO FAR:
  iteration 1 (`_read_fidelity_nonrate_latency_derisk.py`,
    research/findings/2026-08-28-read-fidelity-nonrate-latency-UNDEFINED.md): first-spike LATENCY, UNDEFINED.
  iteration 2 (`_read_fidelity_nonrate_latency_dispersion_derisk.py`,
    research/findings/2026-08-28-read-fidelity-dispersion-instrument-fixed-still-NO-read-beats-rate.md):
    instrument fixed (permutation-over-neuron-identity), LATENCY 0/6, DISPERSION(ISI-CV) 1/6.
  iteration 3 (`_read_fidelity_decoder_separability_derisk.py`,
    research/findings/2026-08-28-read-fidelity-decoder-SIGNAL-FOUND-its-a-read-limit-not-wiring.md): a
    linear+nonlinear DECODER over the full per-neuron 10-bin spike-count profile separates generated-vs-
    perceived 6/6 (shuffle-clean, both weight conditions), where mean-rate/latency/dispersion ALL failed. This
    SETTLED the crux as a READ-FIDELITY LIMIT (the signal exists in the DISTRIBUTED pattern; scalar reads
    discard it), not a wiring/credit problem -- and flagged that only ~HALF the decodable separation is
    cross-edge-attributable (the rest is static structural asymmetry between the two hard-wired pools, present
    even under lesion). NEXT LEVER (this file): a read that (a) is BIOLOGICAL/SPIKING, not a bare host
    decoder-as-oracle, and (b) targets the CROSS-EDGE-ATTRIBUTABLE component specifically, like iteration 3's
    own `delta_held_base` anti-cheat.

THE DESIGN -- population-vector / matched-filter TEMPLATE READ, implemented as an actual spiking READOUT
NEURON (Georgopoulos, Schwartz & Kettner 1986, Science 233:1416-1419, "Neuronal population coding of movement
direction": a population's joint response is read out by a WEIGHTED LINEAR COMBINATION across the population,
not any one cell's own rate; Salinas & Abbott 1994, J Neurosci 14(9):5667-5680, "Vector reconstruction from
firing rates": the population vector is (close to) the OPTIMAL LINEAR ESTIMATOR when per-cell weights are
fit to the population's own response statistics, i.e. a matched filter, not a naive equal-weighted sum). The
existing rate read is literally the DEGENERATE equal-weight case of this family (every pool neuron gets
identical weight 1/n at every instant) -- exactly the summary iteration 3 showed throws signal away. This read
instead fits a WEIGHT PER TIME-BIN (a temporal receptive field / matched-filter kernel: Rieke, Warland, de
Ruyter van Steveninck & Bialek 1997, "Spikes: Exploring the Neural Code", ch.2's linear-filter/reverse-
correlation formalism) from TRAIN data, then reads a HELD-OUT population through an actual leaky-integrate-
and-fire neuron whose input current is that filter applied to the population's own spikes -- so the "decode"
is a spiking neuron's own membrane integration + threshold, not a host matmul standing in for one.

WHY THE SAMPLE AXIS IS NEURON IDENTITY, NOT TRIAL REPEATS (a constraint inherited, not chosen). Iteration 2's
own diagnosis (STEP 0 of `_read_fidelity_nonrate_latency_dispersion_derisk.py`, re-verified here by reusing its
machinery unchanged) established that this pool family runs `enable_ou_process=False` AND
`enable_short_term_plasticity=False` -- a FIXED reset + FIXED drive is a deterministic function of nothing else,
so `N_READS=8` "reads" are (at best) ~2 distinct raster states, not 8 independent trials. Fitting the template on
a TRAIN subset of "trials" and evaluating on a held-out subset of "trials" (the literal ask) is therefore not a
meaningful split for THIS pool -- there are no independent trials to split. The genuine, already-validated
resampling axis is NEURON IDENTITY (iteration 2's permutation-over-identity fix; iteration 3's stratified
neuron-identity k-fold CV). This run's "TRAIN vs TEST, no leakage" split is over neuron identity, exactly
matching iteration 3's own CV shape: K_FOLDS-fold stratified CV over the n_union=64 (prov_generated union
prov_perceived) neurons, template fit on the TRAIN fold, readout evaluated ONLY on the TEST fold's neurons.

WHAT IS "THE TEMPLATE" AND WHY IT GENERALIZES ACROSS UNSEEN NEURONS. A per-neuron-identity weight (a literal
Georgopoulos "preferred direction" per cell) cannot generalize to a held-out TEST neuron by construction -- a
test neuron was never in the fit, so it has no fitted per-neuron weight. What DOES generalize is a TEMPORAL
kernel: `template[bin] = clip(mean_bin_profile(TRAIN generated neurons) - mean_bin_profile(TRAIN perceived
neurons), 0, None) / (pooled_per_bin_std(TRAIN) + eps)` -- a per-BIN (not per-neuron) matched-filter weight,
answering "WHEN in the RECALL_STEPS=100 window does the generated-vs-perceived separation concentrate", fit
from a TRAIN subset of neurons and then applied IDENTICALLY (the same 10 numbers) to every TEST neuron's own
raw spike train, regardless of whether that specific neuron was ever seen during fitting. Two feature variants,
exactly iteration 3's own `FEAT_KINDS` (both computed to keep the two runs directly comparable): `raw_held`
(template fit on the held-read profile alone) and `delta_held_base` (template fit on held-minus-base, isolating
the surprise-hold-SPECIFIC contribution net of static baseline asymmetry -- per iteration 3's own honest
residual, THIS is the feature that targets the CROSS-EDGE-ATTRIBUTABLE half of the signal, and is the PRIMARY
gate here; `raw_held` is reported alongside for comparability but does not gate).
Rectified (clipped at 0, never negative): a signed kernel would need some synapses to flip excitatory/
inhibitory identity bin-to-bin, which no real synapse does (Dale's law); rectifying keeps every population-
neuron's contribution a fixed-sign (excitatory-for-generated / inhibitory-for-perceived, see below) input whose
GAIN is temporally gated, the biologically cleaner reading of "the readout listens harder in some time bins
than others", not "a synapse changes sign over time".

THE SPIKING READOUT (this is the "keep it spiking" requirement -- the projection is computed BY a leaky-
integrate-and-fire neuron's own membrane integration, not a numpy dot product the host then interprets).
For a TEST-fold split, define the population's SIGNED instantaneous input at step t (a fixed-sign per-neuron
contribution, Dale's-law-respecting: TEST generated-pool neurons contribute +1 per spike, TEST perceived-pool
neurons contribute -1 per spike -- pool membership is a STRUCTURAL/WIRING fact fixed at pool-build time,
independent of any read, so using it as a synapse's fixed sign is not label leakage, exactly as the ORIGINAL
rate-margin read also groups neurons by this same known membership):
    pop_signal[t] = (spikes among TEST generated neurons at t) - (spikes among TEST perceived neurons at t)
    I[t] = template[bin(t)] * pop_signal[t]              -- the readout neuron's own synaptic input current
A single LIF neuron (`_lif_batch_spike_counts`: dV = -V/tau + I(t), threshold-and-reset, Euler-integrated one
simulation step at a time, vectorized ACROSS independent read/fold draws for host speed only -- the per-row
recurrence is the identical scalar LIF model run one row at a time would compute) integrates I(t) over the
RECALL_STEPS window; its own SPIKE COUNT is the read. `margin = spike_count(held) - spike_count(base)`
(delta_held_base) or `spike_count(held)` alone (raw_held) -- averaged over the N_READS=8 (near-duplicated, see
above) captured read-pairs, the SAME denoising-only role N_READS played in every prior iteration. The host
NEVER reads a raw current or weight -- only `_lif_batch_spike_counts`' output spike counts.
Threshold calibration (v_th) is fit from the TRAIN fold's OWN current trace (never test data): a target
fraction of the peak subthreshold membrane excursion the template produces on TRAIN neurons' held reads, so the
readout sits in a graded, non-saturating spike-count regime without ever looking at the TEST fold.

ANTI-CHEAT (identical shape to iteration 3's own fix, same bars, no read-kind gets an easier floor): K_SHUF
independent neuron-identity relabelings (fresh random 32/32 split), the WHOLE pipeline (fold-split, template
fit, LIF readout) re-run per draw, giving a null margin distribution; `z = (real_mean-null_mean)/null_std`;
`shuffle_collapses` requires the fraction of null draws individually clearing Z_FLOOR against the null's own
mean/std to stay <= SHUF_COLLAPSE_MAX_RATE (both constants reused VERBATIM from iteration 3 -- same bar).

THE GATE (pre-registered before any seed ran): PRIMARY = `delta_held_base` (the cross-edge-attributable
feature): intact real_margin_mean > 0 AND z >= Z_FLOOR AND shuffle_collapses AND the margin is
lesion-attributable (|lesion mean| < F2_LESION_RATIO * |intact mean|, `attributable_to` reused verbatim).
`raw_held` is reported for comparability to iteration 3 but is NOT gating (it is the more permissive/
optimistic reading iteration 3 itself flagged as partly structural). GO requires the PRIMARY gate PASS on
every one of 6 seeds.

DE-RISK ONLY -- no production wiring, no `sim/` edit, no default flip. Subclasses NOTHING new: reuses
`ReadFidelityPool` (build/train/lesion + the raster-capturing `_drive2`) VERBATIM from the committed iteration-1
file, `_capture_reads` VERBATIM from the committed iteration-2 file, and `_binned_features`/`_avg_binned`/
`_stratified_folds`/`_seed_trap_check` VERBATIM from the committed iteration-3 file -- SAME trained cross-edge,
SAME captured rasters, SAME CV machinery, no retraining confound. numpy CPU throughout; pool-runnable.

Run:
  SIM_BACKEND=numpy python -m research.runners._read_fidelity_popvec_template_derisk --seeds 42 --smoke
  SIM_BACKEND=numpy python -m research.runners._read_fidelity_popvec_template_derisk \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_read_fidelity_popvec_template_derisk_6seed.json
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

# ---- this run's own pre-registered constants (declared BEFORE any measurement) ----
N_BINS = 10                       # UNCHANGED from iteration 3 -- same binning, same temporal resolution
Z_FLOOR = 2.0                      # UNCHANGED from iteration 2/3 -- same scale-free significance floor
SHUF_COLLAPSE_MAX_RATE = 0.15      # UNCHANGED from iteration 2/3 -- same anti-cheat bar
K_FOLDS = 5                        # UNCHANGED from iteration 3 -- stratified neuron-identity k-fold CV
R_REPEATS = 5                      # independent CV repeats (fresh fold split each) for the REAL-label decode
K_SHUF = 20                        # independent neuron-identity relabelings for the null distribution
R_REPEATS_SMOKE = 2
K_SHUF_SMOKE = 5
TAU_MEM_STEPS = 5.0                # readout LIF membrane time constant, in SIMULATION STEPS (roughly half a
                                    # bin's width of 10 steps -- a genuine leaky integrator that still resolves
                                    # bin-level structure, not so slow it smears the whole window into one number
                                    # (which would just be a scaled rate read again) nor so fast it degenerates
                                    # to raw coincidence counting)
V_RESET = 0.0
CALIB_TARGET_FRAC = 0.5            # readout threshold = this fraction of the TRAIN fold's own peak subthreshold
                                    # membrane excursion -- calibrated ONLY from TRAIN data, keeps the readout
                                    # graded (not silent, not saturating) without ever looking at TEST neurons
V_TH_FLOOR = 1e-6
EPS = 1e-9
WEIGHT_CONDS = ("intact", "lesion")
FEAT_KINDS = ("raw_held", "delta_held_base")   # delta_held_base is the PRIMARY/gating feature (cross-edge
                                                 # attributable); raw_held is reported, not gating (iteration 3's
                                                 # own flag: raw_held mixes in static structural asymmetry)


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  The matched-filter TEMPORAL template (fit on TRAIN neurons only) + its per-step expansion
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _fisher_template(profiles_train, labels_train_bool):
    """profiles_train: (n_train, n_bins) per-neuron binned feature. labels_train_bool: (n_train,) bool, True
    = generated-pool. Returns a (n_bins,) RECTIFIED matched-filter kernel (Georgopoulos 1986 population-vector
    weighting; Salinas & Abbott 1994's per-bin-std normalization makes it a matched filter, not a naive
    mean-diff) -- clipped at 0 so no bin ever flips synapse sign (Dale's law; see module docstring)."""
    gen = profiles_train[labels_train_bool]
    perc = profiles_train[~labels_train_bool]
    assert gen.shape[0] >= 2 and perc.shape[0] >= 2, "need >=2 train neurons per class for a defined std"
    mu_gen, mu_perc = gen.mean(axis=0), perc.mean(axis=0)
    sd_gen, sd_perc = gen.std(axis=0, ddof=1), perc.std(axis=0, ddof=1)
    pooled_sd = 0.5 * (sd_gen + sd_perc)
    raw = (mu_gen - mu_perc) / (pooled_sd + EPS)
    return np.clip(raw, 0.0, None)


def _expand_template_to_steps(template, steps, n_bins):
    """(n_bins,) -> (steps,), each bin's weight held constant across its own window -- the readout's temporal
    receptive field, upsampled from bin resolution to per-simulation-step resolution."""
    edges = np.linspace(0, steps, n_bins + 1).astype(int)
    out = np.empty(steps, dtype=np.float64)
    for bi in range(n_bins):
        out[edges[bi]:edges[bi + 1]] = template[bi]
    return out


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  The spiking readout: a leaky-integrate-and-fire neuron, batched across independent read/fold
#  draws for host speed only (each row is the SAME scalar LIF recurrence, run independently)
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _lif_batch_spike_counts(I_batch, tau_mem_steps, v_th, v_reset=V_RESET):
    """I_batch: (n_rows, steps) float64, each row an independent readout-neuron input-current trace. Standard
    discrete-time LIF (Euler): V[t] = V[t-1]*exp(-1/tau) + I[t]; spike + reset when V>=v_th. Vectorized across
    ROWS (independent draws) for host speed; the per-row recurrence over TIME is identical to running one LIF
    neuron at a time -- batching trades python-loop count for numpy width, not the neuron model."""
    decay = float(np.exp(-1.0 / tau_mem_steps))
    n_rows, steps = I_batch.shape
    v = np.zeros(n_rows, dtype=np.float64)
    counts = np.zeros(n_rows, dtype=np.float64)
    for t in range(steps):
        v = v * decay + I_batch[:, t]
        spiked = v >= v_th
        counts += spiked
        v = np.where(spiked, v_reset, v)
    return counts


def _calibrate_v_th(train_I_batch, tau_mem_steps, target_frac=CALIB_TARGET_FRAC, floor=V_TH_FLOOR):
    """Threshold calibration from TRAIN-fold current traces ONLY (never test data): the SAME LIF recurrence run
    WITHOUT a threshold (pure subthreshold leaky integration) to find the peak membrane excursion the template
    produces on TRAIN neurons' own held reads, then set v_th to a target fraction of that peak's median across
    the batch -- keeps the readout graded (not silent, not saturating every step) without ever looking at the
    TEST fold this fold's margin will be evaluated on."""
    decay = float(np.exp(-1.0 / tau_mem_steps))
    n_rows, steps = train_I_batch.shape
    v = np.zeros(n_rows, dtype=np.float64)
    vmax = np.zeros(n_rows, dtype=np.float64)
    for t in range(steps):
        v = v * decay + train_I_batch[:, t]
        vmax = np.maximum(vmax, np.abs(v))
    if float(vmax.max()) <= 0.0:
        return floor
    return max(floor, target_frac * float(np.median(vmax)))


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  One fold: fit the template on TRAIN neurons, drive the spiking readout with TEST neurons only
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _fold_margin(raster_pairs, template_bins, train_idx, test_idx, labels_bool, steps, n_bins, feat_kind):
    """raster_pairs: N_READS list of (raster_base, raster_held), each (steps, n_union) bool -- the SAME already-
    captured rasters iteration 2/3 use, no new simulation. Returns the fold's margin (float)."""
    template_steps = _expand_template_to_steps(template_bins, steps, n_bins)
    n_union = labels_bool.size
    test_gen = np.zeros(n_union, dtype=bool); test_gen[test_idx] = labels_bool[test_idx]
    test_perc = np.zeros(n_union, dtype=bool); test_perc[test_idx] = ~labels_bool[test_idx]
    train_gen = np.zeros(n_union, dtype=bool); train_gen[train_idx] = labels_bool[train_idx]
    train_perc = np.zeros(n_union, dtype=bool); train_perc[train_idx] = ~labels_bool[train_idx]

    def _current(raster, gen_mask, perc_mask):
        sig = (raster[:, gen_mask].sum(axis=1).astype(np.float64)
               - raster[:, perc_mask].sum(axis=1).astype(np.float64))
        return template_steps * sig

    # threshold calibration from the TRAIN fold's own HELD-read currents only (no test leakage)
    train_I = np.stack([_current(rh, train_gen, train_perc) for (_rb, rh) in raster_pairs])
    v_th = _calibrate_v_th(train_I, TAU_MEM_STEPS)

    held_I = np.stack([_current(rh, test_gen, test_perc) for (_rb, rh) in raster_pairs])
    base_I = np.stack([_current(rb, test_gen, test_perc) for (rb, _rh) in raster_pairs])
    n_reads = held_I.shape[0]
    counts = _lif_batch_spike_counts(np.concatenate([held_I, base_I], axis=0), TAU_MEM_STEPS, v_th)
    held_counts, base_counts = counts[:n_reads], counts[n_reads:]
    if feat_kind == "raw_held":
        return float(held_counts.mean()), float(v_th)
    return float((held_counts - base_counts).mean()), float(v_th)


def _one_cv_pass(labels_bool, rng, profiles_held, profiles_base, raster_pairs, feat_kind, steps, n_bins):
    """ONE stratified K_FOLDS-fold CV pass: fit the template on each fold's TRAIN neurons, evaluate the
    spiking readout on that fold's TEST neurons, average the per-fold margins. `labels_bool` may be the REAL
    generated/perceived labeling or a shuffled (anti-cheat) relabeling -- identical pipeline either way."""
    prof = profiles_held if feat_kind == "raw_held" else (profiles_held - profiles_base)
    margins = []
    for train_idx, test_idx in _stratified_folds(labels_bool, K_FOLDS, rng):
        template = _fisher_template(prof[train_idx], labels_bool[train_idx])
        m, _v_th = _fold_margin(raster_pairs, template, train_idx, test_idx, labels_bool, steps, n_bins, feat_kind)
        margins.append(m)
    return float(np.mean(margins))


def _combo_stats(raster_pairs, profiles_held, profiles_base, n_gen, n_perc, feat_kind, rng, repeats, k_shuf):
    """Real (REPEATS-repeated CV) vs a K_SHUF-draw neuron-identity permutation null -- same statistical shape
    as iteration 3's `_combo_stats` (same Z_FLOOR, same SHUF_COLLAPSE_MAX_RATE, same anti-cheat definition)."""
    n_union = n_gen + n_perc
    real_label = np.zeros(n_union, dtype=bool); real_label[:n_gen] = True
    real_vals = np.array([_one_cv_pass(real_label, rng, profiles_held, profiles_base, raster_pairs,
                                        feat_kind, RECALL_STEPS, N_BINS) for _ in range(repeats)])
    null_vals = []
    for _ in range(k_shuf):
        perm = rng.permutation(n_union)
        y_shuf = np.zeros(n_union, dtype=bool); y_shuf[perm[:n_gen]] = True
        null_vals.append(_one_cv_pass(y_shuf, rng, profiles_held, profiles_base, raster_pairs,
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


def _arm_verdict(intact_combo, lesion_combo, label):
    floor_ok = bool(intact_combo["real_margin_mean"] > 0 and intact_combo["z"] >= Z_FLOOR)
    denom = abs(intact_combo["real_margin_mean"])
    lesion_ok = bool(denom > 0 and abs(lesion_combo["real_margin_mean"]) < F2_LESION_RATIO * denom)
    frac = attributable_to(label, intact_combo["real_margin_mean"], lesion_combo["real_margin_mean"])
    return {"floor_ok": floor_ok, "lesion_ok": lesion_ok,
            "frac_attributable": None if frac is None else float(frac),
            "shuffle_collapses_intact": bool(intact_combo["shuffle_collapses"]),
            "PASS": bool(floor_ok and lesion_ok and intact_combo["shuffle_collapses"])}


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

    # ---- LESIONED (same event as iteration 1/2/3: zero surprise->provgen, in place) ----
    data = np.asarray(to_host(pool.b.cp_connections.data)).copy()
    data[pool.masks["surprise->provgen"]] = 0.0
    pool.b.cp_connections.data = pool.xp.asarray(data, dtype=pool.b.cp_connections.data.dtype)
    pairs_lesion = _capture_reads(pool, read_dict, union)
    held_l = _avg_binned([h for (_b, h) in pairs_lesion], N_BINS)
    base_l = _avg_binned([b for (b, _h) in pairs_lesion], N_BINS)

    feats = {"intact": (pairs_intact, held_i, base_i), "lesion": (pairs_lesion, held_l, base_l)}

    base_off = int(seed) * 50331653 + 191   # this module's own distinct RNG offset family (sibling offsets
                                              # already in use elsewhere in this file family: *104729+17,
                                              # *65599+41, *7919+101, *997+3, *15485863+271 -- none collide)
    combos = {}
    combo_i = 0
    for cond in WEIGHT_CONDS:
        raster_pairs, held_p, base_p = feats[cond]
        for feat_kind in FEAT_KINDS:
            rng = np.random.default_rng(base_off + combo_i * 7247)
            combo_i += 1
            combos[f"{cond}__{feat_kind}"] = _combo_stats(
                raster_pairs, held_p, base_p, n_gen, n_perc, feat_kind, rng, repeats, k_shuf)

    primary = _arm_verdict(combos["intact__delta_held_base"], combos["lesion__delta_held_base"],
                            "F2 popvec-template spiking-readout margin (delta_held_base, cross-edge attributable)")
    secondary = _arm_verdict(combos["intact__raw_held"], combos["lesion__raw_held"],
                              "F2 popvec-template spiking-readout margin (raw_held, NOT gating -- may include "
                              "static structural asymmetry per iteration 3's own flag)")
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
        Vd = Verdict("read_fidelity_popvec_template_derisk")
        Vd.require("shuffle_anticheat_collapses_on_every_combo",
                   1 if all(r["n_combos_shuffle_collapse"] == r["n_combos"] for r in runs) else 0,
                   expect=lambda x: x >= 1,
                   note="the neuron-identity permutation null must individually clear Z_FLOOR on <= "
                        f"{SHUF_COLLAPSE_MAX_RATE} of its own draws, on EVERY combo, before the popvec readout's "
                        "verdict can be trusted -- same instrument-validity bar as iteration 2/3")
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
        tag = "GO -- the spiking popvec/template readout CLEARS the crux" if all_go_raw else "NO-GO/PARTIAL"

    verdict = (f"{tag}. A downstream spiking READOUT NEURON (LIF, threshold-calibrated from TRAIN data only) "
               f"whose synaptic drive is a temporal matched-filter TEMPLATE (Georgopoulos 1986 population-"
               f"vector coding; Salinas & Abbott 1994 optimal-linear-estimator normalization) fit on a TRAIN "
               f"fold of neurons (stratified K_FOLDS={K_FOLDS}-fold CV over neuron identity, R_REPEATS={repeats}) "
               f"and evaluated on the HELD-OUT test fold's own raw spike trains, on the SAME trained cross-edge "
               f"and SAME captured rasters iteration 1/2/3 used (no retraining confound). PRIMARY gate = "
               f"delta_held_base (cross-edge-attributable component): {n_pass}/{len(runs)} seeds PASS "
               f"(z>=Z_FLOOR={Z_FLOOR} AND lesion-attributable AND shuffle anti-cheat collapses). "
               f"Anti-cheat: K_SHUF={k_shuf}-draw neuron-identity permutation null, shuffle collapses on "
               f"{sum(r['n_combos_shuffle_collapse'] for r in runs)}/{sum(r['n_combos'] for r in runs)} "
               f"combo-seed pairs."
               + (f" UNDEFINED, NOT a validated verdict either way: {len(dec.get('undefined_reasons', []))} "
                  f"precondition(s) unmet -- {'; '.join(dec.get('undefined_reasons', []))}."
                  if verdict_status == "UNDEFINED" else ""))

    payload = {
        "probe": "read_fidelity_popvec_template_derisk", "verdict": verdict, "GO": go,
        "n_seeds": len(runs), "n_seeds_pass_primary": n_pass, "n_seeds_shuffle_ok": n_shuf_ok,
        "seeds": seeds, "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
        "preconditions": preconditions,
        "seed_trap_build_twice": seed_trap,
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
            "rng_formula": "seed*50331653+191, +combo_index*7247 per combo (this module's own distinct offset "
                            "family; does not collide with _assign_blocks(*104729+17), _shuffle_mask(*65599+41), "
                            "iteration-2's permutation rng(*7919+101), _make_ambiguous_pattern(*997+3), or "
                            "iteration-3's rng(*15485863+271, +combo*104651)",
        },
        "mechanism": ("Reuses ReadFidelityPool (build/train/lesion + the raster-capturing `_drive2`) VERBATIM "
                      "from the committed iteration-1 file, `_capture_reads` VERBATIM from iteration-2, and "
                      "`_binned_features`/`_avg_binned`/`_stratified_folds`/`_seed_trap_check` VERBATIM from "
                      "iteration-3. Adds ONLY: a per-bin matched-filter TEMPLATE (Fisher-style rectified "
                      "mean-difference, normalized by pooled per-bin std) fit on a TRAIN fold of neurons; a "
                      "leaky-integrate-and-fire READOUT NEURON (`_lif_batch_spike_counts`, discrete-time Euler, "
                      "threshold calibrated from TRAIN-fold current traces only) whose input current is the "
                      "template applied to the TEST fold's own raw per-step raster, signed by each TEST "
                      "neuron's known (structural, non-leaked) pool membership; the SAME stratified k-fold CV "
                      "and neuron-identity permutation-null anti-cheat shape as iteration 3."),
        "biology": ("Georgopoulos, Schwartz & Kettner 1986 (Science 233:1416-1419, 'Neuronal population coding "
                    "of movement direction') for population-vector readout as a weighted linear combination "
                    "across a population, not any single cell's rate; Salinas & Abbott 1994 (J Neurosci "
                    "14(9):5667-5680, 'Vector reconstruction from firing rates') for the population vector as "
                    "an approximately-optimal (matched-filter-like) linear estimator when per-channel weights "
                    "are fit to the population's own response statistics; Rieke, Warland, de Ruyter van "
                    "Steveninck & Bialek 1997 ('Spikes: Exploring the Neural Code', ch.2) for the linear-filter/"
                    "matched-filter/reverse-correlation formalism grounding a temporal receptive-field kernel. "
                    "This read implements that population-vector/matched-filter motif as a literal spiking "
                    "readout: a leaky-integrate-and-fire neuron whose synaptic drive is the matched-filter "
                    "template, so the projection is computed by the readout neuron's OWN membrane integration "
                    "and threshold crossing, not a host matmul the host then reinterprets as a spike count."),
        "scaffold_residuals": [
            "the temporal template is RECTIFIED (clipped at 0) to keep every population-neuron's synapse a "
            "fixed sign (Dale's law) -- a signed (unrectified) kernel was considered and rejected for this "
            "reason (see module docstring), which may discard some genuinely discriminating anti-correlated bins",
            "per-bin (diagonal) matched-filter normalization (pooled std per bin), not a full covariance-"
            "whitened Fisher LDA across bins -- avoids overfitting/near-singular covariance at this fold's "
            "train-neuron count, at the cost of ignoring cross-bin correlations a fuller LDA could exploit",
            "the LIF readout uses INSTANTANEOUS current injection per spike (no synaptic rise/decay kinetics) "
            "-- a standard simplified LIF, not a conductance-based synapse model",
            "TAU_MEM_STEPS/CALIB_TARGET_FRAC/K_FOLDS/R_REPEATS/K_SHUF are host-chosen "
            "statistical-power/dynamics knobs, not computed features (same class as every prior iteration's own "
            "CV/anti-cheat knobs)",
            "N_READS=8 reads averaged into the per-combo margin (denoising only, per this pool family's "
            "deterministic dynamics -- iteration 2's own diagnosis, unchanged); the genuine per-seed resampling "
            "axis is neuron identity via the K_FOLDS/R_REPEATS/K_SHUF draws, not read repetition",
            "the readout is trained/tested on NEURON IDENTITY folds, not independent TRIALS (this pool family "
            "has no independent trials to split -- see module docstring); this is a deliberate, documented "
            "substitution for the literal ask, matching the constraint iteration 2/3 already established",
            "same host-curated training schedule / topology as the parent crossedge runner (declared there, "
            "unchanged)",
        ],
        "runs": runs,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[READ-FIDELITY POPVEC] VERDICT: {verdict}\n" + "=" * 100, flush=True)
    return 0 if (payload["GO"] or args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
