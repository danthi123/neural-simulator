"""V1 SELF-ORGANIZED receptive-field bank wired into the PRODUCTION visual-cortex build (B1, 2026-08-26).

This is the production-integration glue for the B1 residual: `sim/visual_cortex.py:build_v1_simple_weights` builds
the V1 simple-cell RF weights from a HOST GABOR FORMULA (32 oriented templates = 8 orient x 4 freq, tiled over 256
positions). The OPERATION (V1 filter -> spikes) runs on-substrate; the STRUCTURE (the Gabor weights) is host-computed
-> a criterion-2 structure residual. This organ replaces the host Gabor bank with a bank that is SELF-ORGANIZED on the
real spiking substrate (the already-plastic retina->cortex_v1_simple pathway + the bridge's own rate-window Hebbian /
STDP + homeostasis), then transplants the learned bank onto the production V1 pathway -- exactly mirroring
`apply_v1_gabor_weights` (both install a precomputed relative-index bank via `set_pathway_weights(add_missing=True)`),
so the wiring is a drop-in.

    apply_v1_selforg_weights(bridge, ...)   <-- drop-in for  apply_v1_gabor_weights(bridge, ...)

REUSE-BY-IMPORT (no `sim/` edit): the on-bridge de-risk mechanism
`research/runners/_b1_v1_selforg_onbridge_derisk` supplies build_v1_bridge / build_isotropic_support / read_v1_rfs /
gabor_orientation_tuning / raw_weight_stats. This organ adds only (a) the CLOCK-ADVANCE FIX in its develop loop --
the de-risk's `develop` called `_run_one_simulation_step()` which does NOT advance `runtime_state.current_time_ms`, so
every STDP delta_t was 0 and `--rule stdp` was SILENTLY INERT in every prior on-bridge run (the runner's own guard
prints `STDP IS INERT`); this organ advances the clock so a timing rule is genuinely exercised -- (b) an optional DoG
center-surround front-end, and (c) the transplant + flag surface.

⚠️ CURRENT STATUS = BOUNDARY, so the flag DEFAULTS OFF and MUST NOT be flipped yet. The on-bridge realization hits a
robust COMMON-MODE CONVERGENCE boundary: under full-field gratings the potentiation-only rate-Hebbian rule drives the
ON and OFF channels of each pixel to near-identical weights, so the signed ON-OFF receptive field cancels and OSI ~ 0
(2026-08-14 finding). This organ additionally exercised, for the first time genuinely (clock-fixed), the STDP-LTD and
DoG-whitening levers: neither breaks the common mode (STDP net-depresses to a low-L2 floor; DoG does not create the
opponency). The named next mechanism (a possible `sim/` edit, flagged, NOT done here) is LEARNED (plastic) anti-Hebbian
recurrent inhibition -- SAILnet/Foldiak decorrelation -- which the fixed FS pool here cannot substitute for. Until that
lands and the 6-seed flip-soak clears OSI>=0.5, `apply_v1_selforg_weights` produces a NON-oriented (common-mode) bank;
flipping BRAIN_V1_SELFORG=1 would DEGRADE V1, so the default is the host Gabor bank.

FLAGS (default OFF; the byte-identical oracle is the flag simply unset -> the caller keeps calling apply_v1_gabor):
  * BRAIN_V1_SELFORG        in {1,true,yes,on} -> production uses the self-organized bank (currently BOUNDARY: OFF).
  * BRAIN_V1_SELFORG_LESION in {freeze, shuffle} -> the faculty's OWN lesion oracle:
        freeze  -> NO developmental learning (random-init isotropic support); OSI must collapse to the no-learning
                   control (any orientation that emerges is proof the LEARNING, not the support, produced it).
        shuffle -> develop on PIXEL-SHUFFLED (orientation-destroyed) input; OSI must NOT rise above the freeze control
                   (proof the orientation comes from the INPUT STATISTICS, catalog L.05, not the substrate alone).

Backend: uses the process backend (cupy in production, numpy in tests) via get_backend -- NO global-backend flip.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from sim.backend import get_backend, to_host
# reuse-by-import the adversarially-characterized on-bridge de-risk mechanism (no sim/ edit)
from research.runners._b1_v1_selforg_onbridge_derisk import (
    build_v1_bridge,
    build_isotropic_support,
    read_v1_rfs,
    gabor_orientation_tuning,
    raw_weight_stats,
    render_oriented_field,
    _drive_image,
    _freeze,
)
from sim.visual_cortex import (
    N_ORIENTATIONS,
    N_FREQUENCIES,
    V1_POSITIONS_PER_DIM,
    RETINA_SIZE,
)


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Flag surface (default OFF -- BOUNDARY; parent flips only after the 6-seed flip-soak clears OSI>=0.5).
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def enabled() -> bool:
    """Default OFF. BRAIN_V1_SELFORG in {1,true,yes,on} -> production uses the self-organized RF bank."""
    v = os.environ.get("BRAIN_V1_SELFORG")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def lesion_kind() -> str | None:
    """BRAIN_V1_SELFORG_LESION in {freeze, shuffle} -> the faculty's own lesion oracle (else None)."""
    v = os.environ.get("BRAIN_V1_SELFORG_LESION")
    if v is None:
        return None
    v = v.strip().lower()
    return v if v in ("freeze", "shuffle") else None


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# DoG center-surround front-end (optional; removes the local DC / common mode from the input, retinal whitening).
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _dog(signed: np.ndarray, sigma_c: float = 1.0, sigma_s: float = 2.5, surround_w: float = 0.85) -> np.ndarray:
    """Isotropic (orientation-free) difference-of-Gaussians on a 2D signed image. Separable, numpy-only."""
    def g1(sig):
        r = int(max(1, round(3 * sig)))
        x = np.arange(-r, r + 1, dtype=np.float32)
        k = np.exp(-(x * x) / (2 * sig * sig))
        return k / k.sum()

    def blur(img, sig):
        k = g1(sig)
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, img)
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)
        return out.astype(np.float32)

    return (blur(signed, sigma_c) - surround_w * blur(signed, sigma_s)).astype(np.float32)


def _render_field(rng, retina_size: int, shuffle: bool = False, dog: bool = False) -> np.ndarray:
    """A full-field oriented grating (random orientation/frequency/phase), windowed, split ON/OFF.
    shuffle -> pixel-permuted (orientation destroyed); dog -> center-surround whitened first."""
    import math
    H = W = retina_size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    theta = rng.uniform(0.0, math.pi)
    freq = rng.uniform(0.08, 0.30)
    phase = rng.uniform(0.0, 2 * math.pi)
    proj = xx * math.cos(theta) + yy * math.sin(theta)
    grating = np.cos(2 * math.pi * freq * proj + phase).astype(np.float32)
    cx = rng.uniform(0.25, 0.75) * W
    cy = rng.uniform(0.25, 0.75) * H
    sigma = rng.uniform(0.35, 0.6) * W
    env = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)).astype(np.float32)
    signed = grating * env
    if shuffle:
        flat = signed.reshape(-1).copy()
        rng.shuffle(flat)
        signed = flat.reshape(H, W)
    if dog:
        signed = _dog(signed)
    on = np.maximum(signed, 0.0)
    off = np.maximum(-signed, 0.0)
    return np.stack([on, off], axis=0).astype(np.float32)


def _develop(bridge, r0, n_retina, n_steps, drive_pA, present_steps, seed, xp, shuffle=False, dog=False):
    """Developmental phase with the CLOCK-ADVANCE FIX (so a timing rule is genuinely exercised)."""
    rng = np.random.default_rng(seed * 101 + (7 if shuffle else 3))
    dt_ms = getattr(bridge.core_config, "dt_ms", getattr(bridge.core_config, "dt", 1.0))
    done = 0
    while done < n_steps:
        img = _render_field(rng, bridge_retina_size(n_retina), shuffle=shuffle, dog=dog)
        _drive_image(bridge, r0, n_retina, img, drive_pA, xp)
        for _ in range(present_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_ms += dt_ms   # THE FIX: _run_one_simulation_step does not advance it
            done += 1
            if done >= n_steps:
                break
    bridge.cp_external_input_current[:] = 0.0


def bridge_retina_size(n_retina: int) -> int:
    """n_retina = 2*S*S (ON+OFF) -> S."""
    return int(round((n_retina / 2) ** 0.5))


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The core: self-organize an RF bank on a minimal internal substrate, return the LEARNED relative-index bank.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def selforg_v1_rf_bank(
    seed: int,
    n_orientations: int = N_ORIENTATIONS,
    n_frequencies: int = N_FREQUENCIES,
    n_positions_per_dim: int = V1_POSITIONS_PER_DIM,
    retina_size: int = RETINA_SIZE,
    receptive_field_radius: int = 4,
    dev_steps: int = 40000,
    drive_pA: float = 1200.0,
    present_steps: int = 40,
    rule: str = "hebbian",
    dog: bool = False,
    lesion: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Develop a self-organized retina->V1 RF bank on a minimal internal bridge; return the LEARNED bank
    (rel_pre, rel_post, weights) on the isotropic local support + a metrics dict.

    lesion='freeze'  -> skip development (random-init isotropic support = the no-learning control).
    lesion='shuffle' -> develop on pixel-shuffled (orientation-destroyed) input (the L.05 content control).
    """
    xp, backend = get_backend()
    xp = xp if backend == "cupy" else None

    n_v1 = n_orientations * n_frequencies * n_positions_per_dim * n_positions_per_dim
    bridge, r0, v0, n_retina, _ = build_v1_bridge(
        seed, n_orientations, n_frequencies, n_positions_per_dim, retina_size, receptive_field_radius,
        30.0, 7.0, 0.05, 0.00002, 70.0, 0.85, 0.03,
        syn_scaling=True, syn_scaling_rate=0.02, rule=rule,
    )

    # PRE-learning RFs (random init on isotropic support) = the no-learning baseline.
    rf_pre = read_v1_rfs(bridge, r0, v0, n_retina, n_v1,
                         n_orientations, n_frequencies, n_positions_per_dim, retina_size, receptive_field_radius)
    osi_pre_mean, osi_pre_frac = gabor_orientation_tuning(rf_pre)

    if lesion != "freeze":
        _develop(bridge, r0, n_retina, dev_steps, drive_pA, present_steps, seed, xp,
                 shuffle=(lesion == "shuffle"), dog=dog)
    _freeze(bridge)   # critical-period close: freeze all plasticity for a stable transplant

    rf_post = read_v1_rfs(bridge, r0, v0, n_retina, n_v1,
                          n_orientations, n_frequencies, n_positions_per_dim, retina_size, receptive_field_radius)
    osi_post_mean, osi_post_frac = gabor_orientation_tuning(rf_post)
    raw = raw_weight_stats(bridge, r0, v0, n_retina, n_v1, retina_size)

    # Extract the FULL learned pathway bank on the isotropic support (relative indices) for transplant.
    pre_rel, post_rel = build_isotropic_support(
        n_orientations, n_frequencies, n_positions_per_dim, retina_size, receptive_field_radius)
    block = bridge.cp_connections[r0:r0 + n_retina, v0:v0 + n_v1]
    Wfull = np.asarray(to_host(block.todense())).astype(np.float32)   # (n_retina, n_v1)
    learned_w = Wfull[pre_rel, post_rel].astype(np.float32)

    # GO gate: OSI>=0.5 AND clear lift over the no-learning baseline (the de-risk's self-org gate).
    go = bool(osi_post_frac >= 0.5 and osi_post_frac >= osi_pre_frac + 0.15)
    metrics = dict(
        seed=int(seed), backend=backend, rule=rule, dog=bool(dog), lesion=lesion, n_v1=int(n_v1),
        n_synapses=int(learned_w.size),
        osi_pre_frac=round(float(osi_pre_frac), 4), osi_pre_mean=round(float(osi_pre_mean), 4),
        osi_post_frac=round(float(osi_post_frac), 4), osi_post_mean=round(float(osi_post_mean), 4),
        on_minus_off_mean=raw["on_minus_off_mean"], l2_mean=raw["l2_mean"],
        weight_diagnosis=("COMMON-MODE" if abs(raw["on_minus_off_mean"]) < 0.05 * max(raw["raw_mean_abs"], 1e-9)
                          else "signed-RF"),
        go=go, verdict=("GO" if go else "BOUNDARY"),
    )
    return pre_rel.astype(np.int64), post_rel.astype(np.int64), learned_w, metrics


# per-process cache: developing a bank is expensive; a production build reuses it across calls at one (seed, arch).
_BANK_CACHE: dict = {}


def apply_v1_selforg_weights(
    bridge,
    seed: int = 42,
    n_orientations: int = N_ORIENTATIONS,
    n_frequencies: int = N_FREQUENCIES,
    n_positions_per_dim: int = V1_POSITIONS_PER_DIM,
    retina_size: int = RETINA_SIZE,
    receptive_field_radius: int = 4,
    weight_scale: float = 1.0,
    dev_steps: int = 40000,
    rule: str = "hebbian",
    dog: bool = False,
) -> int:
    """Drop-in replacement for `apply_v1_gabor_weights`: install a SELF-ORGANIZED retina->cortex_v1_simple bank on
    `bridge` (mirrors the Gabor path -- learn a relative-index bank, translate to global indices, install via
    `set_pathway_weights(add_missing=True)`). Honors BRAIN_V1_SELFORG_LESION (freeze/shuffle) for the lesion oracle.

    ⚠️ Currently a BOUNDARY: the learned bank is NON-oriented (common-mode); the caller MUST keep this behind an OFF
    flag until the flip-soak clears OSI>=0.5. Returns the count of synapses installed."""
    if bridge.region_manager is None:
        raise RuntimeError("apply_v1_selforg_weights: bridge.region_manager is None (region framework required).")
    retina_global = list(bridge.region_manager.indices("retina"))
    v1_global = list(bridge.region_manager.indices("cortex_v1_simple"))
    if not retina_global or not v1_global:
        raise RuntimeError("apply_v1_selforg_weights: 'retina'/'cortex_v1_simple' region not found.")

    lesion = lesion_kind()
    key = (seed, n_orientations, n_frequencies, n_positions_per_dim, retina_size, receptive_field_radius,
           dev_steps, rule, dog, lesion)
    if key not in _BANK_CACHE:
        _BANK_CACHE[key] = selforg_v1_rf_bank(
            seed, n_orientations, n_frequencies, n_positions_per_dim, retina_size, receptive_field_radius,
            dev_steps=dev_steps, rule=rule, dog=dog, lesion=lesion)
    rel_pre, rel_post, weights, metrics = _BANK_CACHE[key]

    retina_offset = int(retina_global[0])
    v1_offset = int(v1_global[0])
    global_pre = (rel_pre + retina_offset).astype(np.int64)
    global_post = (rel_post + v1_offset).astype(np.int64)
    scaled = (weights * float(weight_scale)).astype(np.float32)
    return int(bridge.set_pathway_weights(
        pathway_name="retina_to_v1_simple_selforg",
        pre_indices=global_pre, post_indices=global_post, weights=scaled, add_missing=True))
