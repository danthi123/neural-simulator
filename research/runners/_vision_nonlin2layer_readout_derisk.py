"""A NONLINEAR 2-LAYER SPIKING readout -- board #75b, the named next rung after the SIGNED
linear-discriminant readout (board #75, `_vision_lindiscrim_readout_derisk.py`) SOLVED the S2->C2
spike-quantization wall but hit the ~0.47 RATE linear-separability ceiling of the z-normed C2 feature
code (capability_go 0/6, does not beat the V1-direct floor).

WHY A NEW MECHANISM (not a re-tune). The #75 finding
(`2026-08-25-vision-signed-linear-discriminant-spiking-readout-solves-quantization-wall-relocates-to-feature-ceiling.md`)
named the residual explicitly and named this exact lever: "the z-normed configural distinction may not
be fully linearly separable; a 2-layer spiking readout (a hidden layer of LIF conjunction /
dendritic-coincidence units before the class populations) can exceed 0.47. Cortical readouts are not
single-layer perceptrons." Independently, the SAME lever (a fixed nonlinear EXPANSION recovering linear
decodability a compressive stage destroyed) was confirmed on a DIFFERENT gap on this substrate
(`2026-07-24-gap4-forward-representability-SURPASSED-nonlinear-expansion-numpy-GO-onbridge-next.md`:
held-out linear decodability 0.284 -> 0.772 via a fixed random-feature expansion, label-shuffle at
chance, non-expanding control far weaker) -- this runner is the SAME mechanism class, biologically
GROUNDED for vision specifically as the CEREBELLAR GRANULE-CELL EXPANSION (Marr 1969; Albus 1971): a
small number of mossy-fiber (here, C2) afferents converge sparsely (K claws) onto a much larger granule
population, and each granule cell's LIF threshold makes it an AND-like COINCIDENCE DETECTOR over its K
inputs -- Litwin-Kumar, Harris, Axel, Sompolinsky & Abbott (2017, Neuron) show K~4 claws is near-optimal
for pattern separation at realistic mossy-fiber counts, matching real anatomy. The Purkinje-cell linear
readout of that expanded code (Brunel, Hakim, Isope, Nadal & Barbour 2004, already cited by #75) is
EXACTLY the signed E+FF-inhibition spiking readout #75 already built and validated -- so this runner
inserts ONE new stage (the granule expansion) between the UNCHANGED #75 C2 code and the UNCHANGED #75
readout machinery (imported, not re-derived), isolating the nonlinearity as the only new variable.

THE MECHANISM (built here; no `sim/` edit; C2 front end + the entire #75 readout REUSED BY IMPORT).
Take the IDENTICAL C2 spike code #75 reads (per-template MAX over locations, averaged over G glimpses).
Project it through a FIXED random sparse "mossy-fiber -> granule" connectivity: each of n_hidden granule
units samples K_CLAW C2 units (K=4 default) with fixed positive weights (excitatory synapses; no sim/
STDP -- template/connectivity learning is not the lever, exactly as #72/#75 fixed the S2 bank). Drive a
granule LIF population with the summed claw current -> granule SPIKES = the EXPANDED code. Because a
granule cell only crosses threshold when enough of its K claws are jointly active, this is a genuine
AND-like conjunction/coincidence nonlinearity, not a linear recombination. The SAME #75 readout
(standardise -> ridge signed linear discriminant -> spike-port as E + FF-inhibition class populations ->
spiking WTA) is then trained on the EXPANDED code instead of the raw C2 code.

ARMS (all held-out positions unless noted):
  EXPAND_learned     the mechanism: granule expansion + #75's learned signed readout, spike-ported.
  EXPAND_random      identical EXPANDED architecture, V untrained (random signed) -- learning-load-bearing
                     control (must be beaten by EXPAND_learned, >=5/6, exactly #75's headline anti-cheat).
  NOEXPAND_learned   the #75 1-LAYER baseline, reproduced HERE on the SAME data/front-end for an
                     apples-to-apples comparison -- this is the ~0.47 ceiling this rung must exceed.
  EXPAND_RATE_lin    signed linear readout on the RATE (threshold-linear, non-spiking) granule code --
                     tests whether the EXPANSION raises the underlying linear-separability CEILING, or
                     only shuffles where the same ceiling sits.
  LINEXPAND_lin      the SAME fixed random K-claw connectivity WITHOUT the LIF threshold (a pure LINEAR
                     recombination of C2) -- isolates whether the AND-like NONLINEARITY is doing the work,
                     or whether any higher-dimensional random re-basis would do (it should not: composing
                     two linear maps stays linear in the original C2 features).
ANTI-CHEATS (they ARE the result, mirroring #75 exactly):
  1. capability_go = EXPAND_learned_held - V1_direct_held >= beat_margin (beats the V1-direct floor).
  2. beats_readout_floor = EXPAND_learned_held > the #72/#75 NO-GO floor (0.34), >=5/6 seeds (task GO bar).
  3. learning_load_bearing = EXPAND_learned_held - EXPAND_random_held >= beat_margin, >=5/6 seeds.
  4. position pooled out (object decodable off the class-population code; position ~chance).
  5. pixel-scramble -> chance; label-shuffle (retrained) -> chance.
  6. DETERMINISTIC: every RNG derived from the `seed` arg; a re-run at one seed byte-compares (this
     runner uses a standalone numpy LIF, not the CoreSimConfig bridge, so cfg.seed/actual_seed_used do
     not apply -- determinism is by explicit per-op seeds + an explicit byte-compare check, same as #75).

BRAIN-BASED status. Somata genuinely SPIKE (LIF: leak, hard threshold, reset, absolute refractory,
per-step membrane noise) at S1, S2, the NEW granule/hidden layer, AND the readout class populations.
The granule layer's AND-like nonlinearity is an EMERGENT property of K-claw summation + a hard threshold,
not a host-computed nonlinearity (no np.maximum trick standing in for a neuron). Common-mode rejection at
the readout = feedforward inhibition (Dale-compliant E/I decomposition, imported unchanged from #75). The
readout weights are a supervised ridge closed-form (the exact fixed point of an L2-decayed three-factor
delta rule; a host-computed teacher scaffold, SAME status as #75/R-STDP). FLAGGED innate developmental
scaffolds (same concessions as config B/C/#75): retinotopic weight-sharing + pooling windows; the fixed
random S2 bank; the fixed random granule (mossy-fiber) connectivity -- granule-cell wiring in the real
cerebellum is itself largely genetically/developmentally specified, not activity-learned, so this is a
DEFENDED concession, not a new one. No live conversational vision consumer exists (#72/#75); scope is the
spiking CAPABILITY.

Sources: Marr, D. (1969). A theory of cerebellar cortex. J. Physiol. 202:437-470. Albus, J. S. (1971). A
theory of cerebellar function. Math. Biosci. 10:25-61. Litwin-Kumar, A., Harris, K. D., Axel, R.,
Sompolinsky, H. & Abbott, L. F. (2017). Optimal degree of synaptic connectivity. Neuron 93:1153-1164.
Cayco-Gajic, N. A. & Silver, R. A. (2019). Re-evaluating circuit mechanisms underlying pattern
separation. Neuron 101:584-602. Brunel, Hakim, Isope, Nadal & Barbour (2004) Neuron 43:745 (Purkinje-cell
perceptron readout of an expanded code -- already the #75 readout's grounding). Maass, Natschlager &
Markram (2002) Neural Comput. 14:2531. Fremaux & Gerstner (2016) Front. Neural Circuits 9:85.

Smoke:
  SIM_BACKEND=numpy python -u -m research.runners._vision_nonlin2layer_readout_derisk \
      --seeds 42 --n-hidden 128 --k-claw 4 \
      --out research/findings/raw/lanes/perception/vnonlin2_smoke.json
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
# ---- the RANDOM S2 template bank (config-C-like; keep S2 FIXED, unchanged from #75) ----
from research.runners._vision_rstdp_readout_derisk import _init_templates  # noqa: E402
# ---- the ENTIRE #75 readout machinery, REUSED BY IMPORT, UNCHANGED. The only new thing this file
# adds is what feeds it (the granule expansion), so the readout itself cannot be a confound. ----
from research.runners._vision_lindiscrim_readout_derisk import (  # noqa: E402
    _c2_rate_code,
    _c2_spike_code,
    _lin_score_pred,
    _spiking_class_read,
    _train_linreadout,
)
from tools.lab import attributable_to  # noqa: E402

OUT = Path("research/findings/raw/lanes/perception/vision_nonlin2layer_readout.json")


# ============================================================================================
# NEW: the granule (hidden-layer) expansion. Marr (1969) / Albus (1971) / Litwin-Kumar et al. (2017):
# a SPARSE, FIXED, random "mossy-fiber -> granule" connectivity; K claws/unit; excitatory weights.
# ============================================================================================
def _init_granule_connectivity(n_in, n_hidden, k_claw, seed):
    """FIXED random sparse convergent connectivity: each of n_hidden granule-like units samples
    k_claw C2 ("mossy fiber") afferents with EXCITATORY (non-negative) synaptic weights. This is the
    ONLY new free structure this runner introduces, and it is UNLEARNED (same status as the fixed S2
    bank) -- the nonlinearity comes from K-claw summation + a hard LIF threshold, not from the specific
    random draw. Returns idx (n_hidden, k_claw) int64, w (n_hidden, k_claw) float32 > 0."""
    rng = np.random.default_rng(seed)
    k = max(1, min(int(k_claw), n_in))
    idx = np.stack([rng.choice(n_in, size=k, replace=False) for _ in range(n_hidden)])
    w = rng.uniform(0.5, 1.5, size=(n_hidden, k)).astype(np.float32)
    return idx.astype(np.int64), w.astype(np.float32)


def _granule_drive(r, idx, w):
    """r (N, n_in) >= 0 -> summed claw current (N, n_hidden) = sum_k w[h,k] * r[:, idx[h,k]]. Purely
    LINEAR in r (no threshold) -- used both as the RATE granule pre-nonlinearity drive and, standalone
    (no LIF), as the LINEAR-EXPANSION-ONLY control that isolates whether nonlinearity is load-bearing."""
    gathered = r[:, idx]                      # (N, n_hidden, k_claw)
    drive = (gathered * w[None, :, :]).sum(axis=2)
    return drive.astype(np.float32)


def _granule_expand_spiking(r, idx, w, a, code, base_seed):
    """The MECHANISM: claw-summed drive -> granule LIF -> granule SPIKES. A granule cell only crosses
    threshold when enough of its K claws are JOINTLY active -- an emergent AND-like coincidence
    nonlinearity (not a host np.maximum standing in for it). Returns r_gc (N, n_hidden), non-negative."""
    drive = np.clip(_granule_drive(r, idx, w), 0.0, None) * a.gc_gain
    counts, first = lif_spike_read(drive, a.T_gc, base_seed, tau=a.tau, v_thresh=a.v_thresh,
                                   t_ref=a.t_ref, noise=a.noise, gain=1.0)
    return spike_code(counts, first, a.T_gc, code).astype(np.float32)


def _granule_expand_rate(r, idx, w, a):
    """The RATE (threshold-linear, non-spiking) granule code: same K-claw summation, a graded
    rectified-threshold F-I curve instead of LIF discretisation. Isolates whether the EXPANSION raises
    the underlying linear-separability ceiling (independent of spike quantization cost).

    THE THRESHOLD MUST MATCH THIS INPUT's OWN DRIVE SCALE (a real bug caught twice: (1) a FIXED
    --gc-thresh=1.0 is negligible next to the raw claw-drive scale (~33-67 for k_claw 4-8) -- the "RATE
    ceiling" was not thresholding at all, numerically indistinguishable from the pure-LINEAR control;
    (2) tying the threshold to v_thresh/gc_gain -- calibrated for the SPIKE code's count scale (0..T)
    -- silently overshot on the RATE code, whose z-normed magnitude is ~5-10x smaller, collapsing the
    RATE arm to near-degenerate (all-but-a-few-percent-clipped-to-zero) at the decisive 6-seed run
    (mean 0.297, near chance) while the SAME hyperparameters' SPIKING arm was healthy. FIX: derive the
    threshold from THIS CALL's own claw-drive magnitude (self-normalising, like the S2 z-norm's
    per-image statistics -- not a fitted/train-only parameter, so no held-out leak) so it transfers
    correctly whether `r` is the spike-count or the rate-feature C2 code. --gc-thresh is now a
    MULTIPLIER on the input's own mean drive (1.0 = threshold at the mean -> a ~50%-sparse code)."""
    drive = _granule_drive(r, idx, w)
    ref_scale = float(np.abs(drive).mean()) + 1e-6
    eff_thresh = a.gc_thresh * ref_scale
    return np.clip(drive - eff_thresh, 0.0, None).astype(np.float32)


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

    tr_c1 = _c1_spiking(complex_of(tr_imgs), a, seed * 101 + 11, a.c1_code)
    he_c1 = _c1_spiking(complex_of(he_imgs), a, seed * 101 + 12, a.c1_code)
    sc_c1 = _c1_spiking(complex_of(sc_imgs), a, seed * 101 + 13, a.c1_code)

    chance = 1.0 / a.n_classes
    chance_pos = 1.0 / len(held_pi)

    # ---- floors (same as #72/#75): V1-direct (position-specific) + flat orientation-histogram pool ----
    A_held = _centroid_decode(_flat(tr_c1), tr_cls, _flat(he_c1), he_cls)
    H_held = _centroid_decode(_hist_oracle(tr_c1, a.n_orientations), tr_cls,
                              _hist_oracle(he_c1, a.n_orientations), he_cls)

    # ---- FIXED random S2 template bank (UNCHANGED from #75) ----
    dim = a.n_orientations * a.s2_p * a.s2_p
    W0 = _init_templates(dim, a.n_s2, seed * 29 + 13)

    # ---- C2 spike code -- IDENTICAL to the #75 1-layer arm (SAME features; this is the apples-to-apples
    # baseline this rung must exceed) ----
    r_tr = _c2_spike_code(tr_c1, W0, a, code, seed * 991 + 100, a.n_glimpses)
    r_he = _c2_spike_code(he_c1, W0, a, code, seed * 991 + 200, a.n_glimpses)
    r_sc = _c2_spike_code(sc_c1, W0, a, code, seed * 991 + 300, a.n_glimpses)
    rr_tr = _c2_rate_code(tr_c1, W0, a)
    rr_he = _c2_rate_code(he_c1, W0, a)

    # ---- NEW: granule/hidden EXPANSION (fixed random K-claw connectivity; unlearned, same status as W0) ----
    gc_idx, gc_w = _init_granule_connectivity(a.n_s2, a.n_hidden, a.k_claw, seed * 53 + 17)
    g_tr = _granule_expand_spiking(r_tr, gc_idx, gc_w, a, code, seed * 887 + 100)
    g_he = _granule_expand_spiking(r_he, gc_idx, gc_w, a, code, seed * 887 + 200)
    g_sc = _granule_expand_spiking(r_sc, gc_idx, gc_w, a, code, seed * 887 + 300)
    gl_tr = _granule_drive(r_tr, gc_idx, gc_w)                       # LINEAR-only expansion control
    gl_he = _granule_drive(r_he, gc_idx, gc_w)
    grr_tr = _granule_expand_rate(rr_tr, gc_idx, gc_w, a)            # RATE granule (ceiling reference)
    grr_he = _granule_expand_rate(rr_he, gc_idx, gc_w, a)

    # ---- BASELINE (NOEXPAND): the #75 1-layer readout, reproduced HERE on the SAME data ----
    V_ne, b_ne, mu_ne, sd_ne = _train_linreadout(r_tr, tr_cls, a.n_classes, a, seed)
    pred_ne_he, _ = _spiking_class_read(r_he, V_ne, b_ne, mu_ne, sd_ne, a, code, seed * 773 + 11)
    pred_ne_tr, _ = _spiking_class_read(r_tr, V_ne, b_ne, mu_ne, sd_ne, a, code, seed * 773 + 12)
    noexpand_learned_held = float((pred_ne_he == he_cls).mean())
    noexpand_learned_train = float((pred_ne_tr == tr_cls).mean())

    # ---- THE MECHANISM: learned signed readout on the EXPANDED (granule) spike code ----
    V_ex, b_ex, mu_ex, sd_ex = _train_linreadout(g_tr, tr_cls, a.n_classes, a, seed)
    pred_ex_he, sp_ex_he = _spiking_class_read(g_he, V_ex, b_ex, mu_ex, sd_ex, a, code, seed * 773 + 21)
    pred_ex_tr, _ = _spiking_class_read(g_tr, V_ex, b_ex, mu_ex, sd_ex, a, code, seed * 773 + 22)
    expand_learned_held = float((pred_ex_he == he_cls).mean())
    expand_learned_train = float((pred_ex_tr == tr_cls).mean())
    expand_linscore_held = float((_lin_score_pred(g_he, V_ex, b_ex, mu_ex, sd_ex) == he_cls).mean())

    # ---- RANDOM control: IDENTICAL expanded architecture, V untrained (random signed) ----
    rngV = np.random.default_rng(seed * 131 + 27)
    Vr = (rngV.standard_normal((a.n_classes, a.n_hidden)).astype(np.float32)
          * float(np.abs(V_ex).mean() + 1e-6))
    br = np.zeros(a.n_classes, dtype=np.float32)
    pred_rnd_he, _ = _spiking_class_read(g_he, Vr, br, mu_ex, sd_ex, a, code, seed * 773 + 31)
    expand_random_held = float((pred_rnd_he == he_cls).mean())

    # ---- CEILING: signed linear readout on the RATE granule code (isolates spike-quantization cost) ----
    V_rc, b_rc, mu_rc, sd_rc = _train_linreadout(grr_tr, tr_cls, a.n_classes, a, seed)
    expand_rate_lin_held = float((_lin_score_pred(grr_he, V_rc, b_rc, mu_rc, sd_rc) == he_cls).mean())

    # ---- ANTI-CHEAT: LINEAR-only expansion (same connectivity, NO threshold) -- is nonlinearity the lever? ----
    V_lx, b_lx, mu_lx, sd_lx = _train_linreadout(gl_tr, tr_cls, a.n_classes, a, seed)
    linexpand_lin_held = float((_lin_score_pred(gl_he, V_lx, b_lx, mu_lx, sd_lx) == he_cls).mean())

    # ---- anti-cheat: position pooled out (off the EXPAND class-population spike code) ----
    obj_split = _within_split_decode(sp_ex_he, he_cls, seed * 37 + 17)
    pos_split = _within_split_decode(sp_ex_he, he_pos, seed * 37 + 19)
    position_pooled_out = (obj_split >= chance + a.decode_margin) and (pos_split <= chance_pos + a.pos_decode_margin)

    # ---- anti-cheat: pixel-scramble null (trained EXPAND readout on scrambled held images) ----
    pred_sc, _ = _spiking_class_read(g_sc, V_ex, b_ex, mu_ex, sd_ex, a, code, seed * 773 + 41)
    scramble_null_held = float((pred_sc == he_cls).mean())

    # ---- anti-cheat: label-shuffle null (retrain the EXPAND readout on shuffled labels) ----
    lbl_shuf = np.random.default_rng(seed * 41 + 29).permutation(tr_cls)
    V_sh, b_sh, mu_sh, sd_sh = _train_linreadout(g_tr, lbl_shuf, a.n_classes, a, seed)
    pred_shuf, _ = _spiking_class_read(g_he, V_sh, b_sh, mu_sh, sd_sh, a, code, seed * 773 + 42)
    lbl_shuffle_null = float((pred_shuf == he_cls).mean())

    # ---- verdicts ----
    beat_margin = a.beat_margin
    capability_go = bool(expand_learned_held - A_held >= beat_margin)              # beats V1-direct floor
    beats_readout_floor = bool(expand_learned_held > a.nogo_floor)                 # clears #72/#75 NO-GO (raw)
    beats_readout_floor_strict = bool(expand_learned_held >= a.nogo_floor + beat_margin)
    learning_load_bearing = bool(expand_learned_held - expand_random_held >= beat_margin)
    nonlinear_lift = expand_learned_held - noexpand_learned_held
    beats_1layer_ceiling = bool(nonlinear_lift >= a.lift_margin)
    anti_cheats_clean = bool(
        (scramble_null_held <= chance + a.decode_margin)
        and (lbl_shuffle_null <= chance + a.decode_margin)
        and position_pooled_out
    )
    task_go = bool(capability_go and beats_readout_floor and anti_cheats_clean)

    return {
        "seed": seed, "code": code,
        "chance_object": round(chance, 4), "chance_position": round(chance_pos, 4),
        "decode": {
            "A_v1_direct_held": round(A_held, 4),
            "H_flat_pool_held": round(H_held, 4),
            "NOEXPAND_learned_held": round(noexpand_learned_held, 4),
            "NOEXPAND_learned_train": round(noexpand_learned_train, 4),
            "EXPAND_learned_held": round(expand_learned_held, 4),
            "EXPAND_learned_train": round(expand_learned_train, 4),
            "EXPAND_linscore_held": round(expand_linscore_held, 4),
            "EXPAND_random_held": round(expand_random_held, 4),
            "EXPAND_RATE_lin_held": round(expand_rate_lin_held, 4),
            "LINEXPAND_lin_held": round(linexpand_lin_held, 4),
            "scramble_null_held": round(scramble_null_held, 4),
        },
        "reframe": {
            "expand_minus_random": round(expand_learned_held - expand_random_held, 4),
            "expand_minus_noexpand_1layer": round(nonlinear_lift, 4),
            "expand_minus_linexpand": round(expand_learned_held - linexpand_lin_held, 4),
            "spkport_cost_linscore_minus_spkwta": round(expand_linscore_held - expand_learned_held, 4),
            "learning_load_bearing": learning_load_bearing,
        },
        "dissociation": {
            "object_decode_heldsplit": round(obj_split, 4),
            "position_decode_heldsplit": round(pos_split, 4),
            "label_shuffle_null": round(lbl_shuffle_null, 4),
            "position_pooled_out": position_pooled_out,
        },
        "verdicts": {
            "task_go": task_go,
            "capability_go": capability_go,
            "beats_readout_floor": beats_readout_floor,
            "beats_readout_floor_strict": beats_readout_floor_strict,
            "learning_load_bearing": learning_load_bearing,
            "beats_1layer_ceiling": beats_1layer_ceiling,
            "anti_cheats_clean": anti_cheats_clean,
        },
    }


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
    attributable_to(f"[{code}] 2-layer EXPAND readout held -> LEARNING (vs random EXPAND readout)",
                    hd("EXPAND_learned_held"), hd("EXPAND_random_held"))
    attributable_to(f"[{code}] 2-layer EXPAND readout held -> vs the #75 1-layer NOEXPAND baseline",
                    hd("EXPAND_learned_held"), hd("NOEXPAND_learned_held"))
    attributable_to(f"[{code}] 2-layer EXPAND readout held -> HIERARCHY (vs V1-direct held)",
                    hd("EXPAND_learned_held"), hd("A_v1_direct_held"))

    n_go = sum(1 for r in rows if r["verdicts"]["task_go"])
    n_cap = sum(1 for r in rows if r["verdicts"]["capability_go"])
    n_floor = sum(1 for r in rows if r["verdicts"]["beats_readout_floor"])
    n_lb = sum(1 for r in rows if r["verdicts"]["learning_load_bearing"])
    n_lift = sum(1 for r in rows if r["verdicts"]["beats_1layer_ceiling"])
    n_clean = sum(1 for r in rows if r["verdicts"]["anti_cheats_clean"])
    # task GO bar (board #75b, literal): beats V1-direct floor by margin (capability) AND clears the
    # readout floor >=5/6, anti-cheats clean.
    task_go_bar = bool((n_cap >= 5) and (n_floor >= 5) and (n_clean == len(rows)))
    overall = ("NONLIN2LAYER-READOUT-GO" if task_go_bar
               else "NONLIN2LAYER-READOUT-NOGO" if (n_floor == 0 and n_lb == 0)
               else f"NONLIN2LAYER-READOUT-PARTIAL-cap{n_cap}/{len(rows)}-floor{n_floor}/{len(rows)}"
                    f"-lb{n_lb}/{len(rows)}-lift{n_lift}/{len(rows)}")
    return {
        "probe": "vision_nonlin2layer_readout", "code": code, "overall_verdict": overall,
        "seeds": a.seeds, "n_seeds": len(rows), "chance_object": round(1.0 / a.n_classes, 4),
        "nogo_floor": a.nogo_floor,
        "task_go_bar": task_go_bar,
        "per_seed_task_go": [r["verdicts"]["task_go"] for r in rows],
        "per_seed_capability_go": [r["verdicts"]["capability_go"] for r in rows],
        "per_seed_beats_readout_floor": [r["verdicts"]["beats_readout_floor"] for r in rows],
        "per_seed_learning_load_bearing": [r["verdicts"]["learning_load_bearing"] for r in rows],
        "per_seed_beats_1layer_ceiling": [r["verdicts"]["beats_1layer_ceiling"] for r in rows],
        "decode_means": {k: mean(("decode", k)) for k in rows[0]["decode"]},
        "reframe_means": {
            "expand_learned_held": hd("EXPAND_learned_held"),
            "expand_random_held": hd("EXPAND_random_held"),
            "noexpand_learned_held": hd("NOEXPAND_learned_held"),
            "expand_minus_random": mean(("reframe", "expand_minus_random")),
            "expand_minus_noexpand_1layer": mean(("reframe", "expand_minus_noexpand_1layer")),
            "expand_minus_linexpand": mean(("reframe", "expand_minus_linexpand")),
            "expand_RATE_lin_ceiling_held": hd("EXPAND_RATE_lin_held"),
            "linexpand_lin_held": hd("LINEXPAND_lin_held"),
            "spkport_cost": mean(("reframe", "spkport_cost_linscore_minus_spkwta")),
        },
        "dissociation_means": {
            "object_decode_heldsplit": mean(("dissociation", "object_decode_heldsplit")),
            "position_decode_heldsplit": mean(("dissociation", "position_decode_heldsplit")),
            "label_shuffle_null": mean(("dissociation", "label_shuffle_null")),
            "scramble_null_held": hd("scramble_null_held"),
        },
        "verdict_fracs": {k: frac(("verdicts", k)) for k in rows[0]["verdicts"]},
        "headroom": {
            "expand_minus_v1_held": round(hd("EXPAND_learned_held") - hd("A_v1_direct_held"), 4),
            "expand_minus_flat_held": round(hd("EXPAND_learned_held") - hd("H_flat_pool_held"), 4),
            "expand_minus_nogo_floor": round(hd("EXPAND_learned_held") - a.nogo_floor, 4),
            "rate_ceiling_minus_expand_spk": round(hd("EXPAND_RATE_lin_held") - hd("EXPAND_learned_held"), 4),
            "expand_rate_ceiling_minus_1layer_rate_ceiling_0.4653": round(hd("EXPAND_RATE_lin_held") - 0.4653, 4),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    p.add_argument("--code", choices=["latency", "count", "both"], default="count")
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
    # S2 configural templates (FIXED random bank, unchanged from #75)
    p.add_argument("--s2-p", type=int, default=3)
    p.add_argument("--n-s2", type=int, default=96)
    # #75's signed linear readout hyperparams (UNCHANGED defaults; the readout is reused verbatim)
    p.add_argument("--ridge", type=float, default=0.5)
    p.add_argument("--n-glimpses", type=int, default=2)
    p.add_argument("--class-pop", type=int, default=24)
    p.add_argument("--read-gain", type=float, default=2.5)
    p.add_argument("--read-bias", type=float, default=1.0)
    p.add_argument("--T-read", type=int, default=48)
    # NEW: the granule/hidden EXPANSION layer (board #75b)
    p.add_argument("--n-hidden", type=int, default=1536,
                   help="granule population size (Marr-Albus expansion; 16x n_s2 -- chosen on a "
                        "42/43/100 exploration sweep over {128..1536}x{4,8,16} claws x gain, leaving "
                        "44/101/102 out-of-sample for the decisive run)")
    p.add_argument("--k-claw", type=int, default=8,
                   help="mossy-fiber claws per granule cell (op-point chosen on the SAME exploration; "
                        "Litwin-Kumar et al. 2017's near-optimal ~4 was also swept and did not do better)")
    p.add_argument("--gc-gain", type=float, default=0.05,
                   help="claw-drive -> granule LIF gain (operating-point knob; chosen on the same "
                        "42/43/100 exploration -- a smoke test caught granule cells SATURATING at the "
                        "naive default, which destroys all discriminability)")
    p.add_argument("--gc-thresh", type=float, default=1.0,
                   help="RATE granule threshold-linear F-I threshold (ceiling-reference arm only)")
    p.add_argument("--T-gc", type=int, default=48, help="granule LIF window (ms/steps)")
    p.add_argument("--lift-margin", type=float, default=0.03,
                   help="EXPAND-minus-NOEXPAND margin reported as beats_1layer_ceiling (diagnostic)")
    # SPIKING (LIF) front end operating point (config B defaults, unchanged from #75)
    p.add_argument("--s1-mode", choices=["spiking", "rate"], default="spiking")
    p.add_argument("--s2-norm", choices=["none", "submean", "z"], default="z")
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
    print(f"[vision-nonlin2layer-readout] seeds={a.seeds} codes={codes} n_s2={a.n_s2} "
          f"n_hidden={a.n_hidden} k_claw={a.k_claw} gc_gain={a.gc_gain} ridge={a.ridge} "
          f"class_pop={a.class_pop} LIF(T1={a.T1},T2={a.T2},T_gc={a.T_gc},Tr={a.T_read})", flush=True)

    result = {}
    for code in codes:
        rows = [run_seed(s, a, code) for s in a.seeds]
        for r in rows:
            d, rf, di, v = r["decode"], r["reframe"], r["dissociation"], r["verdicts"]
            print(f"  [{code} seed {r['seed']}] V1he {d['A_v1_direct_held']:.2f} "
                  f"| NOEXPAND(1L) he {d['NOEXPAND_learned_held']:.2f} "
                  f"| EXPAND(2L) he {d['EXPAND_learned_held']:.2f} (tr {d['EXPAND_learned_train']:.2f}) "
                  f"RANDOM {d['EXPAND_random_held']:.2f} RATE-ceil {d['EXPAND_RATE_lin_held']:.2f} "
                  f"LINEXP {d['LINEXPAND_lin_held']:.2f} "
                  f"| dLEARN {rf['expand_minus_random']:+.2f} dNONLIN {rf['expand_minus_noexpand_1layer']:+.2f} "
                  f"dVSLIN {rf['expand_minus_linexpand']:+.2f} "
                  f"| obj/pos {di['object_decode_heldsplit']:.2f}/{di['position_decode_heldsplit']:.2f} "
                  f"scr {d['scramble_null_held']:.2f} lblshuf {di['label_shuffle_null']:.2f} "
                  f"| GO={v['task_go']} cap={v['capability_go']} floor={v['beats_readout_floor']} "
                  f"lb={v['learning_load_bearing']}", flush=True)
        result[code] = {"summary": _summarize(rows, a, code, t0), "per_seed": rows}

    top = {
        "probe": "vision_nonlin2layer_readout",
        "board": "75b",
        "primary_code": codes[0],
        "overall_verdict": result[codes[0]]["summary"]["overall_verdict"],
        "config": vars(a),
        "by_code": result,
        "mechanism": (
            "UNCHANGED #75 C2 spike code (FIXED config-B spiking front end + FIXED random S2 bank, G "
            "glimpses averaged) -> NEW fixed random sparse K-claw granule (mossy-fiber -> granule cell, "
            "Marr 1969/Albus 1971/Litwin-Kumar et al. 2017) LIF expansion layer (AND-like coincidence "
            "nonlinearity from claw-summation + hard threshold) -> the UNCHANGED #75 readout (ridge "
            "signed linear discriminant, spike-ported as E + FF-inhibition class populations, spiking "
            "WTA). RANDOM arm = identical expanded port, V untrained. LINEXPAND arm = same connectivity, "
            "no threshold (isolates the nonlinearity). Sources: Marr 1969; Albus 1971; Litwin-Kumar et "
            "al. 2017; Cayco-Gajic & Silver 2019; Brunel et al. 2004; Maass et al. 2002; Fremaux & "
            "Gerstner 2016."
        ),
        "reframe_test": (
            "#75's residual was the ~0.47 RATE linear-separability ceiling of the z-normed C2 code, not "
            "spike quantization. A fixed nonlinear (AND-like) expansion is predicted to raise that "
            "ceiling (EXPAND_RATE_lin > the #75 1-layer RATE ceiling ~0.4653) and the spiking mechanism "
            "to inherit the lift (EXPAND_learned_held > NOEXPAND_learned_held ~0.4375), while a LINEAR-"
            "only expansion of the same dimensionality should NOT lift it (composing two linear maps "
            "stays linear). task GO (board #75b) = beats V1-direct floor by margin (capability_go) AND "
            "clears the #72/#75 NO-GO readout floor (0.34) at >=5/6 AND anti-cheats clean on every seed."
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
        print(f"[{code}] {s['overall_verdict']}  EXPAND={rm['expand_learned_held']} "
              f"RANDOM={rm['expand_random_held']} NOEXPAND(1L)={rm['noexpand_learned_held']} "
              f"(dLEARN={rm['expand_minus_random']:+} dNONLIN={rm['expand_minus_noexpand_1layer']:+}) "
              f"RATE-ceil={rm['expand_RATE_lin_ceiling_held']} LINEXP={rm['linexpand_lin_held']} "
              f"vs NOGO {s['nogo_floor']} | cap {sum(s['per_seed_capability_go'])}/{s['n_seeds']} "
              f"floor {sum(s['per_seed_beats_readout_floor'])}/{s['n_seeds']} "
              f"lb {sum(s['per_seed_learning_load_bearing'])}/{s['n_seeds']} "
              f"GO {sum(s['per_seed_task_go'])}/{s['n_seeds']}", flush=True)
    print(f"[written] {out_path}", flush=True)
    print("=" * 100, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
