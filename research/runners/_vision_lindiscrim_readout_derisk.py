"""SPIKING SIGNED LINEAR-DISCRIMINANT readout (excitatory + feedforward-inhibition) with TEMPORAL
EVIDENCE INTEGRATION -- a DIFFERENT fully-spiking configural object-"which" readout, built to close the
S2->C2 NO-GO (board #75) after REWARD-MODULATED STDP mapped a dead-end.

WHY A NEW MECHANISM. R-STDP (research/runners/_vision_rstdp_readout_derisk.py) NO-GO'd across the ENTIRE
2D operating-point sweep (24-256 S2 x 30-150 epochs, all overall_verdict=RSTDP-READOUT-NOGO). Its held
spiking-WTA capped at ~0.40 (below the V1-direct floor 0.418) and, tellingly, its CENTROID read of the
same spike code was NOT load-bearing (learned ~= random ~= 0.37). R-STDP is a MAPPED dead-end -- do NOT
tune it further.

WHERE THE SIGNAL ACTUALLY IS (measured, #72 + #75). The LIF-spiking S1->C1 FRONT END preserves the
position-invariant CONFIGURAL capability on spikes (config B: held 0.5625 with a RATE S2/C2 MAX readout,
arch load-bearing 6/6). The capability is carried on spikes THROUGH C1. What fails is the FULLY-SPIKING
S2->C2 READOUT (config C: 0.34), because the rate C2 discrimination is a fine DISTRIBUTED cosine
modulation -- across-template std ~0.042 on a common-mode ~0.80 -- that falls below the per-unit spike
quantization floor.

THE DIAGNOSED FLAW THIS RUNNER FIXES (the "companion process we replaced with a constant"). Both prior
readouts CANNOT subtract the common mode at the read:
  * config-C centroid = nearest-cosine-centroid (unsupervised prototype);
  * R-STDP = a FIXED non-negative round-robin class block-sum (cscore = r @ class_mat.T, class_mat 0/1),
    with only the EXCITATORY S2 templates learned.
A fine modulation riding on a large common mode is a SIGNED linear-discriminant problem: to read it you
must SUBTRACT the shared mode and WEIGHT units by discriminability -- which needs NEGATIVE weights. The
brain implements a signed readout with EXCITATORY afferents + FEEDFORWARD INHIBITION (an inhibitory
interneuron carrying the negatively-weighted / common-mode pool; Dale's law). Neither prior readout had
this companion process. The distributed (not sparse) signal ALSO argues for a POPULATION readout that
integrates over ALL units with the right signed weights, not a block-sum -- and for INTEGRATING EVIDENCE
over more than one 48 ms glimpse (the quantization floor is per-presentation; the animal accrues evidence
over multiple fixations / a longer accumulation -- drift-diffusion). Speed is explicitly secondary here.

THE MECHANISM (built here). Keep config B's FIXED spiking front end (S1->C1 LIF, random S2 template bank
-- template LEARNING is NOT the lever, #72). Read the SAME C2 spike code config C reads (per-template MAX
over locations = position-invariant), AVERAGED over G independent LIF glimpses (temporal evidence
integration). Then, instead of a centroid / block-sum, LEARN a SIGNED linear-discriminant readout
V (n_classes x n_S2) + bias b by a supervised three-factor DELTA rule (multinomial-logistic; pre = C2
spike, post = class neuron, third factor = the teacher one-hot error -- Fremaux & Gerstner 2016). PORT IT
TO SPIKES honestly: standardise the C2 code by its train mean/std (the common-mode-rejecting interneuron
+ divisive normalisation), decompose the effective weights w = w+ - w-, and drive a POPULATION of LIF
class somata per class with EXCITATORY current (w+ . r) MINUS FEEDFORWARD-INHIBITORY current (w- . r) plus
a tonic bias; a spiking WTA (argmax over class-population spike counts, lateral inhibition) is the FULLY-
SPIKING prediction.

ARMS (all read on SPIKES, held-out positions):
  LEARNED_spkwta   supervised signed linear readout, SPIKE-PORTED (E + FF-inhibition LIF class pops,
                   spiking WTA). THE MECHANISM + the fully-spiking headline.
  RANDOM_spkwta    IDENTICAL spike-ported architecture, V UNTRAINED (random signed). The like-for-like
                   control: LEARNED must BEAT RANDOM (>=5/6). If learned ~= random, the reframe is
                   REFUTED (a first-class negative).
CEILING / INSTRUMENT references (bound the arc; verify the read is measuring the right thing):
  RATE_lin         signed linear readout on the RATE C2 features (the ceiling of a signed linear read).
  SPK_lin_noport   signed linear SCORE (argmax V.r, no LIF class port) on the spike C2 code -- isolates
                   the cost of the SPIKE PORT of the decision from the cost of learning the read.
  centroid_spk     config-C nearest-centroid on the SAME spike code -- reproduces the ~0.37 NO-GO, proving
                   we read identical features (so any lift is the READOUT CLASS, not a different front end).

ANTI-CHEATS (they ARE the result):
  1. FULLY-SPIKING held accuracy (LEARNED_spkwta) must clear the #72/#75 NO-GO floor (0.34 + margin) and
     beat the V1-direct (0.418) and flat-pool (0.30) floors. Train positions {0,2,4,6}; held {1,3,5,7}.
  2. LEARNING LOAD-BEARING: LEARNED_spkwta - RANDOM_spkwta >= beat_margin, per seed (the headline; the
     rate case had random==learned, so this is the non-trivial claim). GO bar = >=5/6 seeds.
  3. POSITION POOLED OUT: object decodable off the class-population spike code; position ~chance.
  4. PIXEL-SCRAMBLE -> chance; LABEL-SHUFFLE -> chance; 6 seeds (42/43/44/100/101/102); DETERMINISTIC
     (every RNG derived from the `seed` arg; a re-run byte-compares). This runner uses a standalone numpy
     LIF (lif_spike_read), NOT the CoreSimConfig bridge, so cfg.seed/actual_seed_used do not apply --
     determinism is by explicit per-op seeds + a byte-compare check.

BRAIN-BASED status. Somata genuinely SPIKE (LIF: leak, hard threshold, reset, absolute refractory,
per-step membrane noise) at S1, S2 AND the readout class populations. Common-mode rejection = feedforward
inhibition (an interneuron carrying the negatively-weighted pool; Dale-compliant E/I decomposition). The
decision = a spiking WTA over class populations (lateral inhibition). The readout weights are learned by a
supervised three-factor delta rule (teacher one-hot = an AI-teacher scaffold, same status as R-STDP's
reward). FLAGGED innate developmental scaffolds (same concessions as config B/C): retinotopic
weight-sharing + pooling windows; the fixed random S2 bank. No sim/ edit; the LIF machinery + S1->C1 front
end are REUSED BY IMPORT.

Sources: Fremaux, N. & Gerstner, W. (2016) Front. Neural Circuits (three-factor plasticity). Maass, Natschlager
& Markram (2002) Neural Comput. 14:2531 (a trained LINEAR readout of a spiking reservoir). Brunel, Hakim,
Isope, Nadal & Barbour (2004) Neuron 43:745 (the cerebellar perceptron readout). Pouget, Dayan & Zemel
(2000) Nat. Rev. Neurosci. 1:125 (population/linear decoding). Carandini & Heeger (2012) Nat. Rev. Neurosci.
13:51 (divisive normalisation = the FF-inhibition common-mode rejection).

Smoke:
  SIM_BACKEND=numpy python -u -m research.runners._vision_lindiscrim_readout_derisk \
      --seeds 42 --epochs 200 --n-s2 96 --n-glimpses 4 \
      --out research/findings/raw/lanes/perception/vlin_smoke.json

BCM S2-TEMPLATE-LEARNING smoke (2026-09-01, this de-risk; --s2-learn none is the byte-identical default;
--s2-bcm-competitive-frac is REQUIRED for non-degenerate learning -- see _bcm_learn_s2_templates):
  SIM_BACKEND=numpy python -u -m research.runners._vision_lindiscrim_readout_derisk \
      --seeds 42 --s2-learn bcm --s2-bcm-gain 2 --s2-bcm-theta-alpha 0.02 --s2-bcm-pre-floor 0.02 \
      --s2-bcm-epochs 5 --s2-bcm-competitive-frac 0.25 \
      --out research/findings/raw/lanes/perception/vlin_bcm_smoke.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# ---- reuse the RATE front end + rendering BY IMPORT ----
from research.runners._genfrontier_optionB_visual_similarity_derisk import (  # noqa: E402
    build_gabor_response_matrix,
    encode_v1,
    pool_v1_to_complex,
)
from research.runners._vision_hmax_hierarchy_derisk import (  # noqa: E402
    _build_objects,
    _centroid_decode,
    _extract_patches,
    _hist_oracle,
    _l2n,
    _object_classes,
    _positions,
    _scramble_images,
    _within_split_decode,
)
# ---- reuse the SPIKING C1 front end (config B) + LIF machinery BY IMPORT (#72 runner) ----
from research.runners._vision_hmax_spiking_derisk import (  # noqa: E402
    _c1_spiking,
    _flat,
    lif_spike_read,
    spike_code,
)
# ---- the RANDOM S2 template bank (config-C-like; keep S2 FIXED, learning is at the READOUT) ----
from research.runners._vision_rstdp_readout_derisk import _init_templates  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = Path("research/findings/raw/lanes/perception/vision_lindiscrim_readout.json")


# ============================================================================================
# FIXED spiking front end -> the C2 spike code (SAME features config C reads), averaged over G glimpses.
# ============================================================================================
def _apply_s2_norm(drive, a):
    """The pre-readout S2 lateral normalization, factored out of the two call sites (spike + rate) so the
    2026-09-01 board-#135 opsweep finding's named next lever can be added ONCE, identically for both.

    'alpha' is the finding's own proposed graded/partial normalization (research/findings/2026-09-01-vision-
    lindiscrim-opsweep-board135-...): a PARAMETRIC divisive interpolation `drive / (sigma0 + alpha*std)`
    between the 'none' arm (has RATE headroom -- ceiling 0.57-0.62 vs z's 0.4653 -- but the LIF saturates) and
    the 'z' arm (avoids saturation but caps the ceiling at the 0.4653 RATE ceiling), swept continuously via
    alpha in [0,1] instead of the exhausted {none,submean,z} discrete choice. alpha=0 -> drive/sigma0 (a fixed
    rescale, no per-sample divisive competition, closest to 'none'); alpha=1 with sigma0 tiny approaches pure
    divisive (std-only, NO mean-subtraction) normalization -- a DIFFERENT family from 'z' (which also
    subtracts the mean), by design (the finding's decomposition attributes the ceiling specifically to the
    DIVISIVE step, not the mean-subtraction). Default OFF (s2_norm stays 'z' unless explicitly requested) ->
    byte-identical to every prior run of this file.

    'satdiv' (2026-09-01, this scoping pass) is a DIFFERENT FUNCTIONAL FORM, not another point on the
    alpha/z affine-rescale family: the actual Carandini & Heeger (2012, Nat. Rev. Neurosci. 13:51-62)
    semi-saturating divisive-normalization ratio R_i = drive_i^n / (sigma^n + sum_j drive_j^n), where the
    sum is the LOCAL population's pooled suppressive drive (same per-location pool 'z'/'alpha' already use,
    axis=2 = the n_S2 templates at that patch location). 'z' and 'alpha' are both AFFINE rescales by a
    population mean/std -- they can leave an unbounded positive tail after clipping, which is exactly what
    saturates the LIF once the divisor is small (the alpha=0.5 decisive run, research/findings/raw/lanes/
    perception/vlin_alpha_readout_6seed.json, collapsed LEARNED_spkwta_held to ~0.257, at/below chance,
    while its RATE ceiling rose to 0.50-0.62 -- the SAME failure signature as the 'none' arm: the affine
    family cannot decouple "avoid saturation" from "keep headroom" because both properties are set by the
    SAME single divisor). The ratio form is instead BOUNDED by construction (R_i in [0, drive_i^n/sigma^n)
    when pool is small, asymptoting toward the pool-normalized fraction drive_i^n/pool as pool grows) --
    an automatic-gain-control curve with a smooth semi-saturation knee (set by sigma), never the linear
    unbounded rescale-then-clip of 'z'/'alpha'. This is the actual companion process behind the citation
    both #75 and #135 already invoke for the FF-inhibition read (Carandini & Heeger) but had NOT yet
    implemented in its own (ratio, not affine) form.
    """
    if a.s2_norm == "submean":
        return np.clip(drive - drive.mean(axis=2, keepdims=True), 0.0, None)
    elif a.s2_norm == "z":
        mu = drive.mean(axis=2, keepdims=True)
        sd = drive.std(axis=2, keepdims=True)
        return np.clip((drive - mu) / (sd + 1e-6), 0.0, None)
    elif a.s2_norm == "alpha":
        sd = drive.std(axis=2, keepdims=True)
        alpha = float(getattr(a, "s2_norm_alpha", 0.5))
        sigma0 = float(getattr(a, "s2_norm_sigma0", 1e-3))
        return np.clip(drive / (sigma0 + alpha * sd), 0.0, None)
    elif a.s2_norm == "satdiv":
        n = float(getattr(a, "s2_satdiv_n", 2.0))
        sigma = float(getattr(a, "s2_satdiv_sigma", 0.5))
        scale = float(getattr(a, "s2_satdiv_scale", 1.0))
        dp = np.power(np.clip(drive, 0.0, None), n)
        pool = dp.sum(axis=2, keepdims=True)  # local population's pooled suppressive drive (same axis as z/alpha)
        return scale * dp / ((sigma ** n) + pool + 1e-12)
    return drive  # "none"


def _kwta_over_templates(drive, frac):
    """2026-09-01 (board #135/#75, S2-template-learning scoping): competitive sparse coding ACROSS the
    S2 template bank -- the cheap first rung the satdiv/ridge-plateau finding's NO-DEFER handoff names
    ("k-WTA already exists at S1 [`_kwta_per_band`, _vision_hmax_spiking_derisk.py] but was never applied
    at S2/C2"). Per (image, patch-LOCATION), keep only the top `frac` fraction of the n_S2 templates'
    responses and zero the rest -- lateral inhibition ACROSS the template population at a shared location
    (Foldiak 1991's competitive step; a hard-threshold discretisation of Rozell, Johnson, Baraniuk &
    Olshausen 2008's LCA sparse-coding dynamics, which use iterative soft-thresholding + local competition
    among a population of leaky integrators to converge on a sparse code -- this is the one-shot top-k
    approximation of that competition, cheap enough for an inline smoke).

    WHY THIS TARGETS THE DIAGNOSED WALL DIRECTLY: the #72/#75 root cause is a fine DISTRIBUTED cosine
    modulation (across-template std ~0.04) riding on a large COMMON MODE (~0.8) shared by most templates at
    a given location -- exactly what a winner-take-all competitive step is built to remove (only the
    locally-BEST-matching templates survive per location; the near-uniformly-elevated non-winners are
    suppressed to 0, which directly shrinks the shared common mode the readout's ridge/z-norm has to fight).
    Applied AFTER `_apply_s2_norm` (any norm mode, including satdiv) so this is a genuinely NEW axis, not a
    restatement of the exhausted affine-normalization family: normalization sets the PER-TEMPLATE GAIN,
    competition then decides WHICH templates are allowed to carry signal to the LIF / ridge readout at all.
    drive: (N, n_loc, n_S2). frac<=0 or >=1 disables (byte-identical -- this lever defaults OFF)."""
    if frac is None or frac >= 1.0 or frac <= 0.0:
        return drive
    n_S2 = drive.shape[2]
    k = max(1, int(round(frac * n_S2)))
    if k >= n_S2:
        return drive
    thr = np.sort(drive, axis=2)[:, :, n_S2 - k][:, :, None]  # kth-largest template per (image, location)
    return np.where(drive >= thr, drive, 0.0).astype(np.float32)


def _bcm_learn_s2_templates(patches_flat, W0, gain, theta_alpha, pre_floor, epochs, renorm,
                             competitive_frac, seed):
    """Activity-dependent BCM (Bienenstock, Cooper & Munro 1982) learning of the S2 template bank --
    the decisive next mechanism named by the 2026-09-01 finding (research/findings/2026-09-01-vision-
    readout-side-exhausted-satdiv-plus-ridge-plateau-points-to-S2-template-learning.md): satdiv, ridge
    re-tune, and k-WTA-at-S2 all IMPROVE the readout but PLATEAU short of the capability bar, because the
    residual is the INFORMATION a frozen RANDOM S2 bank carries -- not how its responses are normalized
    or thresholded. That finding names BCM as the mechanism, already validated on this substrate
    (sim/config.py `hebbian_bcm`; research/findings/2026-08-26-b1-v1-selforg-onbridge-BCM-sliding-
    threshold.md broke the identical common-mode boundary 62x on V1 orientation self-org).

    PORTED, not re-derived: the equations below are the SAME ones sim/bridge.py's on-bridge
    `hebbian_bcm` branch implements (~L9849-9891) --
        y_i     = ReLU(w_i . x)                 postsynaptic drive of template i to presynaptic pattern x
                                                  (the SAME 'drive' _c2_rate_code/_c2_spike_code compute
                                                  downstream at eval time -- W0 is a drop-in replacement)
        theta_i <- (1-theta_alpha)*theta_i + theta_alpha*y_i^2      sliding metaplastic threshold EMA
        dw_ij   = gain * x_j * y_i * (y_i - theta_i), only where x_j > pre_floor (presynaptic gate)
    y_i > theta_i potentiates the co-active input; y_i < theta_i depresses it -- the input-specific
    depression a plain Hebbian/coactivity rule lacks, and exactly what breaks the S2 common mode (the
    same signature the 2026-08-26 finding broke for V1 ON/OFF).

    WHY A STANDALONE FUNCTION, NOT A LITERAL IMPORT: the substrate's version is inlined against the CuPy
    SPARSE CONNECTIVITY MATRIX's COO row/col-indexed synapses, inside a full spiking-network simulation
    STEP (per-timestep, tied to `self.cp_connections`, `self.cp_hebb_coactivity_trace`, bridge state).
    This runner's S2 template bank is a dense (n_S2, D) cosine-match matrix with no bridge/connectivity
    object, no per-timestep simulation loop, and no spiking pre/post traces to hook the substrate's
    branch into -- there is nothing importable to call. The equations are therefore applied directly to
    the (n_S2, D) matrix instead, online, one presented patch at a time (theta is a running average
    across presentations, so -- exactly as on-bridge -- this MUST be sequential, not a closed-form/batch
    solve). `epochs` passes over a seeded-shuffled order of ALL (image, location) training patches (not
    just the 6 raw images/class -- each image contributes n_loc patches, so the effective presentation
    count is far larger than 6/class, though still drawn from only 6 underlying images/class per position
    -- see the runner's --s2-bcm-epochs help and the finding's honest thin-data risk).

    THE ONE ADAPTATION BEYOND A LITERAL PORT -- RENORMALIZATION: if `renorm`, each template row is
    rescaled to unit L2 norm after every update, matching `_init_templates`'s random baseline (ALSO
    unit-norm). This is the direct analog of the substrate's own stabilization: on-bridge, weights are
    hard-clipped to [w_min,w_max] (Dale's-law-rectified excitatory bounds); here, every downstream
    consumer (_apply_s2_norm, _kwta_over_templates, the LIF gain calibration) treats S2 templates as
    UNIT-NORM cosine-matching directions, so keeping them unit-norm after each step is what keeps
    'drive' on the SAME cosine-similarity scale the frozen-random baseline produces -- theta_M's own
    superlinear growth still does the actual BCM selectivity/stabilization; renorm only prevents an
    unbounded MAGNITUDE confound (bigger templates -> bigger drive -> an artificially easier ridge read)
    from being mistaken for genuinely LEARNED template information (honest risk #3 in the finding: a
    thin-data template bank overfitting is a confound distinct from readout overfitting). `renorm=0` is
    offered for comparison and is expected to risk exactly that confound.

    THE COMPANION PROCESS THIS POPULATION-LEVEL PORT NEEDS THAT A SINGLE ON-BRIDGE V1 CELL DID NOT
    (`competitive_frac`, found necessary during THIS de-risk's own exploration, not pre-registered):
    the on-bridge BCM finding trains ONE weight vector per V1 CELL, each with its own distinct
    retinotopic receptive field, so different cells see different local input and naturally symmetry-
    break into different preferred orientations from independent random starts. This runner's n_S2
    templates instead all share the SAME pool of training patches (a feature BANK, not a retinotopic
    array) -- and an early sweep (this de-risk, seed 42, gain in {0.5,...,10}) found that plain BCM run
    that way collapses the WHOLE bank toward one dominant shared direction (theta_std -> 0.0000 across
    ALL 96 templates, RATE-ceiling WORSE than the frozen-random baseline): with no interaction between
    the units being trained, every template's independent BCM dynamics gets pulled toward the same
    principal direction of the shared input statistics -- the "wall-reframe" companion process missing
    here is COMPETITION AMONG THE LEARNERS, not competition at read-time (--s2-kwta-frac, which acts on
    an already-fixed bank's responses). `competitive_frac` in (0,1) restricts the WEIGHT UPDATE at each
    presentation to only the top-`competitive_frac` fraction of templates by their CURRENT drive y (a
    Foldiak 1991 / Kohonen 1982-style winner-relative competitive-learning gate composed with BCM's own
    signed LTP/LTD): only the best-matching templates for THIS patch get to specialise on it, so
    different patches recruit different winners over training and diversity is preserved instead of
    collapsing. theta still updates for EVERY template every presentation (a postsynaptic activity
    statistic the cell tracks regardless of whether it won -- matches the substrate's own semantics,
    where `cp_bcm_theta` updates unconditionally while the weight branch is separately gated). 0.0 (or
    >=1.0) disables competition (the original single-cell-only port, kept for comparison / the ablation
    that motivated adding this).

    Returns W (n_S2, D) learned templates (float32), theta (n_S2,) final per-template thresholds, and a
    diagnostics dict (theta stats, mean-square drive, drift from init) -- reported so a plateau or a
    too-noisy-to-learn-from estimate (honest risks #1/#2) is VISIBLE, not silently absorbed."""
    W = W0.copy().astype(np.float64)
    n_S2, D = W.shape
    theta = np.zeros(n_S2, dtype=np.float64)
    rng = np.random.default_rng(seed)
    N = patches_flat.shape[0]
    X = patches_flat.astype(np.float64)
    y_sq_sum = np.zeros(n_S2, dtype=np.float64)
    n_seen = 0
    k_win = None
    if competitive_frac and 0.0 < competitive_frac < 1.0:
        k_win = max(1, int(round(competitive_frac * n_S2)))
    for _ep in range(max(1, int(epochs))):
        order = rng.permutation(N)
        for idx in order:
            x = X[idx]
            y = np.clip(W @ x, 0.0, None)
            theta = (1.0 - theta_alpha) * theta + theta_alpha * (y * y)
            gate = x > pre_floor
            if not gate.any():
                continue
            winner = None
            if k_win is not None and k_win < n_S2:
                thr = np.partition(y, n_S2 - k_win)[n_S2 - k_win]  # kth-largest drive this presentation
                winner = y >= thr
            dw = (gain * y * (y - theta))[:, None] * (x * gate)[None, :]
            if winner is not None:
                dw = dw * winner[:, None]
            W = W + dw
            if renorm:
                norms = np.linalg.norm(W, axis=1, keepdims=True)
                W = W / np.where(norms < 1e-9, 1.0, norms)
            y_sq_sum += y * y
            n_seen += 1
    diag = {
        "n_presentations": int(n_seen),
        "competitive_frac": float(competitive_frac),
        "theta_final_mean": float(theta.mean()),
        "theta_final_std": float(theta.std()),
        "frac_theta_near_zero": float(np.mean(theta < 1e-8)),  # never-driven-above-floor units (dead)
        "mean_sq_drive_mean": float((y_sq_sum / max(1, n_seen)).mean()),
        "mean_sq_drive_std": float((y_sq_sum / max(1, n_seen)).std()),
        "template_drift_from_init_mean": float(np.mean(np.linalg.norm(W - W0.astype(np.float64), axis=1))),
    }
    return W.astype(np.float32), theta.astype(np.float32), diag


def _c2_spike_code(c1, W0, a, code, base_seed, n_glimpses):
    """c1 (N, n_orient, g, g) spiking C1 -> convolutional S2 cosine match -> S2 lateral inhibition
    (winner-relative contrast) -> LIF S2 coincidence spikes -> C2 per-template MAX over locations
    (position-invariant). Averaged over `n_glimpses` INDEPENDENT LIF draws (temporal evidence
    integration; G=1 reproduces the config-C single-glimpse read). Returns r (N, n_S2)."""
    patches = _extract_patches(c1, a.s2_p)                     # (N, n_loc, D)
    N, n_loc, D = patches.shape
    pn = _l2n(patches, axis=2)
    drive = np.clip(pn @ W0.T, 0.0, None)                      # (N, n_loc, n_S2) cosine match
    drive = _apply_s2_norm(drive, a)
    drive = _kwta_over_templates(drive, getattr(a, "s2_kwta_frac", 0.0))
    flat = drive.reshape(N * n_loc, -1)
    acc = None
    G = max(1, int(n_glimpses))
    for gi in range(G):
        counts, first = lif_spike_read(flat, a.T2, base_seed + 101 + gi * 13,
                                       tau=a.tau, v_thresh=a.v_thresh, t_ref=a.t_ref,
                                       noise=a.noise, gain=a.s2_gain)
        s2 = spike_code(counts, first, a.T2, code).reshape(N, n_loc, -1)  # (N, n_loc, n_S2)
        r = s2.max(axis=1).astype(np.float32)                  # C2 MAX over locations (position-invariant)
        acc = r if acc is None else acc + r
    return (acc / G).astype(np.float32)


def _c2_rate_code(c1, W0, a):
    """The RATE C2 features (cosine match + z lateral inhibition + MAX over locations, NO LIF): the
    ceiling reference for a signed linear readout."""
    patches = _extract_patches(c1, a.s2_p)
    N, n_loc, D = patches.shape
    pn = _l2n(patches, axis=2)
    drive = np.clip(pn @ W0.T, 0.0, None)
    drive = _apply_s2_norm(drive, a)
    drive = _kwta_over_templates(drive, getattr(a, "s2_kwta_frac", 0.0))
    return drive.max(axis=1).astype(np.float32)                # (N, n_S2)


# ============================================================================================
# The learned SIGNED linear-discriminant readout (supervised three-factor delta = multinomial logistic).
# ============================================================================================
def _standardise(r_tr):
    """Feature mean/std over the TRAIN set (the common-mode + divisive-normalisation the FF interneuron
    implements). Returns mu, sd."""
    mu = r_tr.mean(axis=0)
    sd = r_tr.std(axis=0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def _train_linreadout(r_tr, y_tr, n_classes, a, seed):
    """RIDGE-regularised least-squares readout (the standard trained LINEAR readout of a spiking
    reservoir; Maass et al. 2002). SIGNED weights V (n_classes, n_S2) + bias b. Closed form on the
    STANDARDISED C2 code (mu/sd = the common-mode-rejecting FF interneuron + divisive normalisation):
        W = (X^T X + lambda*N I)^{-1} X^T Yc,   Yc = one-hot - 1/n_classes (centred), b = 1/n_classes.
    RIDGE lambda is the homeostatic regulariser (biologically = synaptic scaling): lambda->inf shrinks
    W toward the class-MEAN-difference (centroid) direction that generalises across held positions;
    lambda->0 = full-covariance-whitened LDA that overfits the position-specific code. Sweeping lambda
    is the operating-point knob. Closed form is the exact fixed point of an L2-decayed online delta rule
    (three-factor: pre C2 spike x post class x teacher error), so it is deterministic and instrument-clean.
    `seed` unused (kept for signature parity). Returns V, b, mu, sd."""
    del seed
    mu, sd = _standardise(r_tr)
    X = (r_tr - mu) / sd                                        # (N, n_S2) standardised
    N, D = X.shape
    Y = np.zeros((N, n_classes), dtype=np.float64)
    Y[np.arange(N), y_tr] = 1.0
    Yc = Y - 1.0 / n_classes                                   # centred one-hot targets
    Xd = X.astype(np.float64)
    G = Xd.T @ Xd + (a.ridge * N) * np.eye(D)                  # (D, D) regularised Gram
    W = np.linalg.solve(G, Xd.T @ Yc)                          # (D, n_classes)
    V = W.T.astype(np.float32)                                 # (n_classes, D)
    b = np.full(n_classes, 1.0 / n_classes, dtype=np.float32)
    return V, b, mu, sd


def _spiking_class_read(r, V, b, mu, sd, a, code, base_seed):
    """PORT the signed linear score to SPIKES honestly. Effective weight w = V/sd (so w.r + const = the
    learned standardised score). Decompose w = w+ - w-: EXCITATORY current (w+ . r) MINUS FEEDFORWARD-
    INHIBITORY current (w- . r, via an inhibitory interneuron) plus a tonic bias. Drive a POPULATION of
    M LIF class somata per class; spiking WTA (argmax over class-population spike counts) = the FULLY-
    SPIKING prediction. Returns pred (N,), class_spikes (N, n_classes)."""
    n_classes, D = V.shape
    w = (V / sd).astype(np.float32)                            # (n_classes, D) effective signed weight
    const = (b - (w * mu).sum(axis=1)).astype(np.float32)      # tonic (folds the standardisation offset)
    wp = np.clip(w, 0.0, None)                                 # excitatory synapses (w+)
    wm = np.clip(-w, 0.0, None)                                # inhibitory pool (w-), via FF interneuron
    E = r @ wp.T                                               # (N, n_classes) excitatory drive
    I = r @ wm.T                                               # (N, n_classes) feedforward inhibition
    net = (E - I) + const[None, :]                             # net synaptic drive to each class soma
    # shift so the population sits in a spiking regime (a global tonic; a constant across classes does
    # NOT change the WTA argmax -- it is not a host decision, only a bias current onto every class soma)
    net = net - net.mean(axis=1, keepdims=True)
    net = net * a.read_gain + a.read_bias
    N = r.shape[0]
    M = max(1, a.class_pop)
    tiled = np.repeat(net, M, axis=1)                          # (N, n_classes*M) population per class
    counts, first = lif_spike_read(np.clip(tiled, 0.0, None), a.T_read, base_seed + 7,
                                   tau=a.tau, v_thresh=a.v_thresh, t_ref=a.t_ref,
                                   noise=a.noise, gain=1.0)
    sp = spike_code(counts, first, a.T_read, code).reshape(N, n_classes, M).sum(axis=2)  # (N, n_classes)
    pred = sp.argmax(axis=1).astype(np.int64)
    return pred, sp.astype(np.float32)


def _lin_score_pred(r, V, b, mu, sd):
    """The non-spiking signed linear SCORE prediction (argmax V.r, no LIF port). Isolates the cost of
    the SPIKE PORT of the decision layer."""
    X = (r - mu) / sd
    logits = X @ V.T + b
    return logits.argmax(axis=1).astype(np.int64)


# ============================================================================================
def run_seed(seed, a, code):
    positions = _positions(a.n_pos_total, a.image_size, a.pos_span)
    held_pi = list(range(1, a.n_pos_total, 2))
    train_pi = [pi for pi in range(a.n_pos_total) if pi not in held_pi]
    train_positions = [positions[pi] for pi in train_pi]
    held_positions = [positions[pi] for pi in held_pi]
    thetas = [(k / a.n_slots) * math.pi for k in range(a.n_slots)]
    class_perms = _object_classes(a.n_classes, a.n_slots)

    tr_imgs, tr_cls, tr_pos = _build_objects(class_perms, thetas, train_positions, a.n_ex, a, seed * 101 + 1)
    he_imgs, he_cls, he_pos = _build_objects(class_perms, thetas, held_positions, a.n_ex, a, seed * 101 + 2)
    sc_imgs = _scramble_images(he_imgs, seed * 101 + 3)

    Wg = build_gabor_response_matrix(
        n_orientations=a.n_orientations, n_frequencies=a.n_frequencies,
        n_positions_per_dim=a.n_pos, retina_size=a.image_size, receptive_field_radius=a.rf_radius)

    def complex_of(imgs):
        return pool_v1_to_complex(encode_v1(imgs, Wg), a.n_orientations, a.n_frequencies, a.n_pos)

    # SPIKING C1 front end (config B).
    tr_c1 = _c1_spiking(complex_of(tr_imgs), a, seed * 101 + 11, a.c1_code)
    he_c1 = _c1_spiking(complex_of(he_imgs), a, seed * 101 + 12, a.c1_code)
    sc_c1 = _c1_spiking(complex_of(sc_imgs), a, seed * 101 + 13, a.c1_code)

    chance = 1.0 / a.n_classes
    chance_pos = 1.0 / len(held_pi)

    # ---- floors (same as #72/#75): V1-direct (position-specific) + flat orientation-histogram pool ----
    A_held = _centroid_decode(_flat(tr_c1), tr_cls, _flat(he_c1), he_cls)
    H_held = _centroid_decode(_hist_oracle(tr_c1, a.n_orientations), tr_cls,
                              _hist_oracle(he_c1, a.n_orientations), he_cls)

    # ---- S2 template bank: FIXED random (default) or BCM-LEARNED from the training patches ----
    dim = a.n_orientations * a.s2_p * a.s2_p
    W0 = _init_templates(dim, a.n_s2, seed * 29 + 13)
    bcm_diag = None
    if getattr(a, "s2_learn", "none") == "bcm":
        # SAME random init as the frozen-random baseline above (like-for-like: only whether learning
        # happens afterward differs) -- presynaptic patches are the FIXED spiking C1 front end's
        # training-split patches, L2-normalised exactly as _c2_rate_code/_c2_spike_code do at eval time.
        tr_patches = _l2n(_extract_patches(tr_c1, a.s2_p), axis=2).reshape(-1, dim)
        W0, _bcm_theta, bcm_diag = _bcm_learn_s2_templates(
            tr_patches, W0, gain=a.s2_bcm_gain, theta_alpha=a.s2_bcm_theta_alpha,
            pre_floor=a.s2_bcm_pre_floor, epochs=a.s2_bcm_epochs,
            renorm=bool(a.s2_bcm_renorm), competitive_frac=a.s2_bcm_competitive_frac,
            seed=seed * 733 + 5)

    # ---- C2 spike code (SAME features config C reads), averaged over G glimpses (temporal integration) ----
    r_tr = _c2_spike_code(tr_c1, W0, a, code, seed * 991 + 100, a.n_glimpses)
    r_he = _c2_spike_code(he_c1, W0, a, code, seed * 991 + 200, a.n_glimpses)
    r_sc = _c2_spike_code(sc_c1, W0, a, code, seed * 991 + 300, a.n_glimpses)
    # RATE C2 features (ceiling reference)
    rr_tr = _c2_rate_code(tr_c1, W0, a)
    rr_he = _c2_rate_code(he_c1, W0, a)

    # ---- LEARNED signed linear readout on the SPIKE C2 code ----
    V, b, mu, sd = _train_linreadout(r_tr, tr_cls, a.n_classes, a, seed)
    pred_he_spk, sp_he = _spiking_class_read(r_he, V, b, mu, sd, a, code, seed * 773 + 11)
    pred_tr_spk, _ = _spiking_class_read(r_tr, V, b, mu, sd, a, code, seed * 773 + 12)
    learn_spkwta_held = float((pred_he_spk == he_cls).mean())
    learn_spkwta_train = float((pred_tr_spk == tr_cls).mean())
    learn_linscore_held = float((_lin_score_pred(r_he, V, b, mu, sd) == he_cls).mean())

    # ---- RANDOM control: identical spike-ported architecture, V untrained (random signed) ----
    rngV = np.random.default_rng(seed * 131 + 7)
    Vr = (rngV.standard_normal((a.n_classes, a.n_s2)).astype(np.float32) * float(np.abs(V).mean() + 1e-6))
    br = np.zeros(a.n_classes, dtype=np.float32)
    pred_he_rnd, _ = _spiking_class_read(r_he, Vr, br, mu, sd, a, code, seed * 773 + 21)
    rnd_spkwta_held = float((pred_he_rnd == he_cls).mean())

    # ---- CEILING: signed linear on the RATE C2 features ----
    Vc, bc, muc, sdc = _train_linreadout(rr_tr, tr_cls, a.n_classes, a, seed)
    rate_lin_held = float((_lin_score_pred(rr_he, Vc, bc, muc, sdc) == he_cls).mean())

    # ---- INSTRUMENT: config-C centroid on the SAME spike code (reproduces the ~0.37 NO-GO) ----
    centroid_spk_held = _centroid_decode(r_tr, tr_cls, r_he, he_cls)
    scr_centroid_held = _centroid_decode(r_tr, tr_cls, r_sc, he_cls)

    # ---- anti-cheat: position pooled out (off the LEARNED class-population spike code) ----
    obj_split = _within_split_decode(sp_he, he_cls, seed * 37 + 17)
    pos_split = _within_split_decode(sp_he, he_pos, seed * 37 + 19)
    position_pooled_out = (obj_split >= chance + a.decode_margin) and (pos_split <= chance_pos + a.pos_decode_margin)

    # ---- anti-cheat: label-shuffle null (retrain the readout on shuffled labels -> must be chance) ----
    lbl_shuf = np.random.default_rng(seed * 41 + 21).permutation(tr_cls)
    Vs, bs, mus, sds = _train_linreadout(r_tr, lbl_shuf, a.n_classes, a, seed)
    pred_shuf, _ = _spiking_class_read(r_he, Vs, bs, mus, sds, a, code, seed * 773 + 31)
    lbl_shuffle_null = float((pred_shuf == he_cls).mean())

    # ---- verdicts ----
    learning_load_bearing = bool(learn_spkwta_held - rnd_spkwta_held >= a.beat_margin)
    beats_nogo = bool(learn_spkwta_held >= a.nogo_floor + a.beat_margin)      # strict (+margin)
    beats_nogo_raw = bool(learn_spkwta_held > a.nogo_floor)                   # raw (> 0.34 floor)
    capability_go = bool(
        (learn_spkwta_held >= chance + a.decode_margin)
        and beats_nogo
        and (learn_spkwta_held - A_held >= a.beat_margin)
        and (learn_spkwta_held - H_held >= a.beat_margin)
        and learning_load_bearing
        and position_pooled_out
        and (scr_centroid_held <= chance + a.decode_margin)
        and (lbl_shuffle_null <= chance + a.decode_margin)
    )
    architecture_load_bearing = bool(learn_spkwta_held - H_held >= a.beat_margin)

    row = {
        "seed": seed, "code": code,
        "chance_object": round(chance, 4), "chance_position": round(chance_pos, 4),
        "decode": {
            "A_v1_direct_held": round(A_held, 4),
            "H_flat_pool_held": round(H_held, 4),
            "LEARNED_spkwta_train": round(learn_spkwta_train, 4),
            "LEARNED_spkwta_held": round(learn_spkwta_held, 4),
            "LEARNED_linscore_held": round(learn_linscore_held, 4),
            "RANDOM_spkwta_held": round(rnd_spkwta_held, 4),
            "RATE_lin_ceiling_held": round(rate_lin_held, 4),
            "centroid_spk_held_NOGO_repro": round(centroid_spk_held, 4),
            "scramble_centroid_held": round(scr_centroid_held, 4),
        },
        "reframe": {
            "learned_minus_random_spkwta": round(learn_spkwta_held - rnd_spkwta_held, 4),
            "spkport_cost_linscore_minus_spkwta": round(learn_linscore_held - learn_spkwta_held, 4),
            "quantization_gap_rate_minus_spk": round(rate_lin_held - learn_linscore_held, 4),
            "learning_load_bearing": learning_load_bearing,
        },
        "dissociation": {
            "object_decode_heldsplit": round(obj_split, 4),
            "position_decode_heldsplit": round(pos_split, 4),
            "label_shuffle_null": round(lbl_shuffle_null, 4),
            "position_pooled_out": position_pooled_out,
        },
        "verdicts": {
            "capability_go": capability_go,
            "beats_config_c_nogo": beats_nogo,
            "beats_config_c_nogo_raw": beats_nogo_raw,
            "learning_load_bearing": learning_load_bearing,
            "architecture_load_bearing": architecture_load_bearing,
        },
    }
    if bcm_diag is not None:
        row["bcm"] = bcm_diag  # only present when --s2-learn bcm; keeps the default path byte-identical
    return row


def _summarize(rows, a, code, t0):
    def mean(path):
        vals = []
        for r in rows:
            cur = r
            for k in path:
                cur = cur[k]
            vals.append(float(cur))
        return round(float(np.mean(vals)), 4)

    def frac(path):
        def _get(r):
            cur = r
            for k in path:
                cur = cur[k]
            return cur
        return round(float(np.mean([1.0 if _get(r) else 0.0 for r in rows])), 4)

    hd = lambda k: mean(("decode", k))  # noqa: E731
    attributable_to(f"[{code}] signed-linear spiking readout held -> LEARNING (vs random spiking readout)",
                    hd("LEARNED_spkwta_held"), hd("RANDOM_spkwta_held"))
    attributable_to(f"[{code}] signed-linear spiking readout held -> vs config-C centroid NO-GO",
                    hd("LEARNED_spkwta_held"), hd("centroid_spk_held_NOGO_repro"))
    attributable_to(f"[{code}] signed-linear spiking readout held -> HIERARCHY (vs V1-direct held)",
                    hd("LEARNED_spkwta_held"), hd("A_v1_direct_held"))

    n_go = sum(1 for r in rows if r["verdicts"]["capability_go"])
    n_lb = sum(1 for r in rows if r["verdicts"]["learning_load_bearing"])
    n_beat = sum(1 for r in rows if r["verdicts"]["beats_config_c_nogo"])
    # GO bar (task): readout clears the NO-GO floor AND beats random at >=5/6.
    task_go = bool((n_beat >= 5) and (n_lb >= 5))
    overall = ("LINDISCRIM-READOUT-GO" if task_go
               else "LINDISCRIM-READOUT-NOGO" if (n_beat == 0 and n_lb == 0)
               else f"LINDISCRIM-READOUT-PARTIAL-beat{n_beat}/{len(rows)}-lb{n_lb}/{len(rows)}")
    return {
        "probe": "vision_lindiscrim_readout", "code": code, "overall_verdict": overall,
        "seeds": a.seeds, "n_seeds": len(rows), "chance_object": round(1.0 / a.n_classes, 4),
        "config_c_nogo_floor": a.nogo_floor,
        "task_go_5of6_beat_and_lb": task_go,
        "per_seed_capability_go": [r["verdicts"]["capability_go"] for r in rows],
        "per_seed_learning_load_bearing": [r["verdicts"]["learning_load_bearing"] for r in rows],
        "per_seed_beats_nogo": [r["verdicts"]["beats_config_c_nogo"] for r in rows],
        "decode_means": {k: mean(("decode", k)) for k in rows[0]["decode"]},
        "reframe_means": {
            "learned_spkwta_held": hd("LEARNED_spkwta_held"),
            "random_spkwta_held": hd("RANDOM_spkwta_held"),
            "learned_minus_random_spkwta": mean(("reframe", "learned_minus_random_spkwta")),
            "rate_lin_ceiling_held": hd("RATE_lin_ceiling_held"),
            "centroid_spk_NOGO_repro": hd("centroid_spk_held_NOGO_repro"),
            "spkport_cost": mean(("reframe", "spkport_cost_linscore_minus_spkwta")),
            "quantization_gap_rate_minus_spk": mean(("reframe", "quantization_gap_rate_minus_spk")),
        },
        "dissociation_means": {
            "object_decode_heldsplit": mean(("dissociation", "object_decode_heldsplit")),
            "position_decode_heldsplit": mean(("dissociation", "position_decode_heldsplit")),
            "label_shuffle_null": mean(("dissociation", "label_shuffle_null")),
        },
        "verdict_fracs": {k: frac(("verdicts", k)) for k in rows[0]["verdicts"]},
        "headroom": {
            "learned_minus_random_spkwta": round(hd("LEARNED_spkwta_held") - hd("RANDOM_spkwta_held"), 4),
            "learned_minus_nogo_floor": round(hd("LEARNED_spkwta_held") - a.nogo_floor, 4),
            "learned_minus_v1_held": round(hd("LEARNED_spkwta_held") - hd("A_v1_direct_held"), 4),
            "learned_minus_flat_held": round(hd("LEARNED_spkwta_held") - hd("H_flat_pool_held"), 4),
            "rate_ceiling_minus_learned_spk": round(hd("RATE_lin_ceiling_held") - hd("LEARNED_spkwta_held"), 4),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    p.add_argument("--code", choices=["latency", "count", "both"], default="count",
                   help="neural code the S2/C2 readout is read with (config B best = count)")
    p.add_argument("--c1-code", choices=["latency", "count"], default="count")
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument("--n-slots", type=int, default=3)
    p.add_argument("--n-pos-total", type=int, default=8)
    p.add_argument("--pos-span", type=float, default=8.0)
    p.add_argument("--n-ex", type=int, default=6)
    p.add_argument("--image-size", type=int, default=56)
    p.add_argument("--slot-offset", type=float, default=10.0)
    p.add_argument("--stroke-len", type=float, default=7.0)
    p.add_argument("--stroke-tk", type=float, default=1.8)
    p.add_argument("--pixel-noise", type=float, default=0.03)
    # V1 front end
    p.add_argument("--n-orientations", type=int, default=8)
    p.add_argument("--n-frequencies", type=int, default=2)
    p.add_argument("--n-pos", type=int, default=24)
    p.add_argument("--rf-radius", type=int, default=3)
    p.add_argument("--orient-norm", choices=["none", "div", "z"], default="z")
    p.add_argument("--c1-gate", type=float, default=0.15)
    p.add_argument("--c1-win", type=int, default=6)
    p.add_argument("--c1-stride", type=int, default=3)
    # S2 configural templates (FIXED random bank)
    p.add_argument("--s2-p", type=int, default=3)
    p.add_argument("--n-s2", type=int, default=96,
                   help="fixed random S2 template-bank size (round-robin over classes irrelevant here; "
                        "the READOUT is learned over the full bank)")
    p.add_argument("--s2-learn", choices=["none", "bcm"], default="none",
                   help="2026-09-01 decisive next mechanism (satdiv/ridge/k-WTA all plateau; the finding's "
                        "NO-DEFER handoff: the residual is the frozen random S2 bank's INFORMATION content, "
                        "not its normalization/threshold). 'bcm' LEARNS the S2 templates from the S1/C1 "
                        "training patches by the Bienenstock-Cooper-Munro (1982) sliding-threshold rule, "
                        "ALREADY validated on this substrate (sim/config.py hebbian_bcm; the 2026-08-26 "
                        "finding broke the identical common-mode boundary 62x on V1 orientation self-org) "
                        "-- see _bcm_learn_s2_templates(). 'none' (default) keeps the frozen random bank -> "
                        "byte-identical to every prior run of this file.")
    p.add_argument("--s2-bcm-gain", type=float, default=200.0,
                   help="'bcm' mode only: BCM gain (multiplies phi=x*y*(y-theta_M)); same role as "
                        "sim/config.py's hebbian_bcm (default order-of-magnitude carried from the "
                        "validated on-bridge V1 self-org op point, _b1_v1_selforg_bcm_derisk.py --bcm-gain).")
    p.add_argument("--s2-bcm-theta-alpha", type=float, default=0.02,
                   help="'bcm' mode only: EMA rate of the sliding threshold theta_M=<y^2>. sim/config.py's "
                        "hebbian_bcm_theta_alpha default (0.001) is calibrated for thousands of SIMULATION "
                        "STEPS per stimulus on-bridge; this runner presents one patch per update (far fewer "
                        "total presentations), so theta needs a faster EMA to converge -- an op-point tune "
                        "of the SAME formula, not a re-derivation of the rule.")
    p.add_argument("--s2-bcm-pre-floor", type=float, default=0.02,
                   help="'bcm' mode only: presynaptic-activity gate -- only x_j>floor synapses change "
                        "(sim/config.py hebbian_bcm_pre_floor, same default; patch features here are also "
                        "L2-normalised nonnegative activations, so the same floor scale applies).")
    p.add_argument("--s2-bcm-epochs", type=int, default=5,
                   help="'bcm' mode only: passes over the (seeded-shuffled) training patches. 6 examples/ "
                        "class is thin for a stable per-unit sliding threshold (honest risk); multiple "
                        "passes over the same patches is this runner's mitigation -- report if the theta "
                        "estimate is still too noisy rather than forcing a number (see 'bcm' diagnostics "
                        "in the per-seed output: theta_final_std, frac_theta_near_zero).")
    p.add_argument("--s2-bcm-renorm", type=int, choices=[0, 1], default=1,
                   help="'bcm' mode only: 1 (default) renormalizes each learned template row to unit L2 "
                        "norm after every update -- keeps 'drive' on the frozen-random baseline's cosine-"
                        "similarity scale instead of confounding learning with magnitude growth (see "
                        "_bcm_learn_s2_templates docstring). 0 disables, for comparison.")
    p.add_argument("--s2-bcm-competitive-frac", type=float, default=0.25,
                   help="'bcm' mode only: restrict the WEIGHT UPDATE at each presentation to the top-"
                        "this-fraction of templates by current drive (Foldiak 1991/Kohonen 1982 "
                        "competitive-learning gate composed with BCM). Found NECESSARY during this "
                        "de-risk's own exploration: without it, all n_S2 templates share one training-"
                        "patch pool with no interaction, and plain per-unit BCM collapses the WHOLE bank "
                        "toward one shared direction (measured: theta_std -> 0, RATE-ceiling WORSE than "
                        "random). 0.0 (or >=1.0) disables competition -- the ORIGINAL single-cell-only "
                        "port, kept for the ablation.")
    # signed linear readout (ridge-regularised least squares = the Maass reservoir readout)
    p.add_argument("--ridge", type=float, default=0.5,
                   help="ridge lambda (homeostatic regulariser = synaptic scaling): large -> centroid "
                        "direction (generalises); small -> whitened LDA (overfits). The op-point knob "
                        "(0.5 chosen on the 42/43/100 exploration; 44/101/102 out-of-sample).")
    p.add_argument("--n-glimpses", type=int, default=2,
                   help="temporal evidence integration: independent LIF C2 draws averaged (G=1 = config C)")
    # spiking readout port
    p.add_argument("--class-pop", type=int, default=24, help="LIF units per class population (decision layer)")
    p.add_argument("--read-gain", type=float, default=2.5, help="net-drive -> LIF gain at the class soma")
    p.add_argument("--read-bias", type=float, default=1.0, help="tonic drive onto every class soma")
    p.add_argument("--T-read", type=int, default=48, help="readout LIF window (ms/steps)")
    # SPIKING (LIF) front end operating point (config B defaults)
    p.add_argument("--s1-mode", choices=["spiking", "rate"], default="spiking")
    p.add_argument("--s2-norm", choices=["none", "submean", "z", "alpha", "satdiv"], default="z",
                   help="'alpha' is the 2026-09-01 board-#135 opsweep finding's named next lever: a graded "
                        "divisive interpolation drive/(sigma0+alpha*std) between 'none' (has rate headroom, "
                        "saturates) and 'z' (avoids saturation, caps the ceiling) -- see --s2-norm-alpha. "
                        "'satdiv' is a DIFFERENT functional form (not another alpha point): the actual "
                        "Carandini & Heeger semi-saturating ratio drive^n/(sigma^n+pool) -- see --s2-satdiv-*")
    p.add_argument("--s2-norm-alpha", type=float, default=0.5,
                   help="'alpha' mode only: 0 -> close to 'none' (fixed rescale by sigma0), 1 -> close to "
                        "pure divisive std-normalization (no mean-subtraction, unlike 'z')")
    p.add_argument("--s2-norm-sigma0", type=float, default=1e-3,
                   help="'alpha' mode only: floor added to alpha*std before dividing (numerical safety + "
                        "sets the alpha=0 fixed rescale)")
    p.add_argument("--s2-satdiv-n", type=float, default=2.0,
                   help="'satdiv' mode only: the exponent n in drive^n/(sigma^n+pool) (Heeger 1992 fits "
                        "V1 contrast responses with n~2-4)")
    p.add_argument("--s2-satdiv-sigma", type=float, default=0.5,
                   help="'satdiv' mode only: the semi-saturation constant sigma (sets the knee of the "
                        "automatic-gain-control curve; the operating-point knob for this lever)")
    p.add_argument("--s2-satdiv-scale", type=float, default=1.0,
                   help="'satdiv' mode only: output rescale so the bounded ratio lands in the LIF's "
                        "graded (non-saturating) drive range jointly with --s2-gain")
    p.add_argument("--s2-kwta-frac", type=float, default=0.0,
                   help="2026-09-01 S2-template-learning scoping, cheap first rung: per (image, location) "
                        "k-WTA competitive sparse coding ACROSS the n_S2 template bank -- keep only the "
                        "top frac fraction of templates' responses, zero the rest (Foldiak 1991 lateral "
                        "inhibition / a hard-threshold LCA, Rozell et al. 2008). Applied AFTER --s2-norm. "
                        "0.0 (default) disables -> byte-identical to every prior run of this file.")
    p.add_argument("--T1", type=int, default=64)
    p.add_argument("--T2", type=int, default=48)
    p.add_argument("--tau", type=float, default=8.0)
    p.add_argument("--v-thresh", type=float, default=1.0)
    p.add_argument("--t-ref", type=int, default=2)
    p.add_argument("--noise", type=float, default=0.06)
    p.add_argument("--s1-gain", type=float, default=1.2)
    p.add_argument("--s2-gain", type=float, default=2.0)
    p.add_argument("--kwta-frac", type=float, default=0.15)
    # gate thresholds
    p.add_argument("--decode-margin", type=float, default=0.15)
    p.add_argument("--beat-margin", type=float, default=0.10)
    p.add_argument("--pos-decode-margin", type=float, default=0.15)
    p.add_argument("--nogo-floor", type=float, default=0.34, help="#72 config-C fully-spiking NO-GO held")
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()

    t0 = time.time()
    codes = ["latency", "count"] if a.code == "both" else [a.code]
    print(f"[vision-lindiscrim-readout] seeds={a.seeds} codes={codes} n_s2={a.n_s2} glimpses={a.n_glimpses} "
          f"ridge={a.ridge} class_pop={a.class_pop} c1_code={a.c1_code} "
          f"LIF(T1={a.T1},T2={a.T2},Tr={a.T_read},s2g={a.s2_gain})", flush=True)

    result = {}
    for code in codes:
        rows = [run_seed(s, a, code) for s in a.seeds]
        for r in rows:
            d, rf, di, v = r["decode"], r["reframe"], r["dissociation"], r["verdicts"]
            print(f"  [{code} seed {r['seed']}] V1he {d['A_v1_direct_held']:.2f} flat {d['H_flat_pool_held']:.2f} "
                  f"| LEARNED spkwta he {d['LEARNED_spkwta_held']:.2f} (tr {d['LEARNED_spkwta_train']:.2f}) "
                  f"linscore {d['LEARNED_linscore_held']:.2f} | RANDOM spkwta {d['RANDOM_spkwta_held']:.2f} "
                  f"| RATE-ceil {d['RATE_lin_ceiling_held']:.2f} centNOGO {d['centroid_spk_held_NOGO_repro']:.2f} "
                  f"| dLEARN {rf['learned_minus_random_spkwta']:+.2f} qgap {rf['quantization_gap_rate_minus_spk']:+.2f} "
                  f"| obj/pos {di['object_decode_heldsplit']:.2f}/{di['position_decode_heldsplit']:.2f} "
                  f"lblshuf {di['label_shuffle_null']:.2f} "
                  f"| GO={v['capability_go']} lb={v['learning_load_bearing']} beat={v['beats_config_c_nogo']}",
                  flush=True)
            if "bcm" in r:
                bd = r["bcm"]
                print(f"      [bcm seed {r['seed']}] n_pres={bd['n_presentations']} "
                      f"theta_mean={bd['theta_final_mean']:.4f} theta_std={bd['theta_final_std']:.4f} "
                      f"frac_theta~0={bd['frac_theta_near_zero']:.2f} "
                      f"msq_drive={bd['mean_sq_drive_mean']:.4f}+-{bd['mean_sq_drive_std']:.4f} "
                      f"drift={bd['template_drift_from_init_mean']:.3f}", flush=True)
        result[code] = {"summary": _summarize(rows, a, code, t0), "per_seed": rows}

    top = {
        "probe": "vision_lindiscrim_readout",
        "primary_code": codes[0],
        "overall_verdict": result[codes[0]]["summary"]["overall_verdict"],
        "config": vars(a),
        "by_code": result,
        "mechanism": (
            "FIXED config-B spiking front end (LIF S1->C1, fixed random S2 bank) -> C2 per-template MAX "
            "over locations spike code, AVERAGED over G independent LIF glimpses (temporal evidence "
            "integration). Readout: a LEARNED SIGNED linear discriminant (supervised three-factor delta = "
            "multinomial logistic) SPIKE-PORTED as a POPULATION of LIF class somata driven by EXCITATORY "
            "(w+ . r) MINUS FEEDFORWARD-INHIBITORY (w- . r, an interneuron carrying the common mode) drive; "
            "spiking WTA over class populations = the fully-spiking prediction. RANDOM arm = identical "
            "port, V untrained. Sources: Fremaux & Gerstner 2016; Maass et al. 2002; Brunel et al. 2004; "
            "Pouget et al. 2000; Carandini & Heeger 2012."
        ),
        "reframe_test": (
            "The config-C NO-GO (0.34) and R-STDP dead-end used readouts that CANNOT subtract the common "
            "mode (centroid / non-negative block-sum). A SIGNED linear discriminant with FF-inhibition + "
            "temporal integration is predicted to clear the floor AND make learning load-bearing (learned "
            ">> random). learning_load_bearing = (learned - random spiking-WTA held) >= beat_margin, per "
            "seed. GO (task) = clears NO-GO floor AND beats random at >=5/6."
        ),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(top, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    for code in codes:
        s = result[code]["summary"]
        rm = s["reframe_means"]
        print(f"[{code}] {s['overall_verdict']}  LEARNED_spkwta={rm['learned_spkwta_held']} "
              f"RANDOM_spkwta={rm['random_spkwta_held']} (dLEARN={rm['learned_minus_random_spkwta']:+}) "
              f"RATE-ceil={rm['rate_lin_ceiling_held']} centNOGO={rm['centroid_spk_NOGO_repro']} "
              f"vs NOGO {s['config_c_nogo_floor']} | beat {sum(s['per_seed_beats_nogo'])}/{s['n_seeds']} "
              f"lb {sum(s['per_seed_learning_load_bearing'])}/{s['n_seeds']} "
              f"GO {sum(s['per_seed_capability_go'])}/{s['n_seeds']}", flush=True)
    print(f"[written] {out_path}", flush=True)
    print("=" * 100, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
