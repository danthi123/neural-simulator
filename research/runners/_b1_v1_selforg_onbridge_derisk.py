"""B1 ON-BRIDGE lift: do V1 simple-cell RF self-organization ON THE REAL SPIKING
SimulationBridge (the numpy GO of 2026-06-21 was off-bridge; this is the on-substrate step).

CONTEXT
-------
  research/findings/2026-06-21-B1-v1-gabor-selforg-{scoping,derisk}.md  (numpy GO)
  research/findings/2026-07-23-perception-closure-scoping.md            (names B1 on-bridge as the not-done cheap lift)

  The V1 simple-cell RF weights are HOST-DESIGNED (a Gabor formula, 32 oriented templates =
  8 orient x 4 freq; sim/visual_cortex.py build_v1_simple_weights). The OPERATION (V1 filter ->
  spikes) already runs on-substrate; the STRUCTURE (the Gabor weights) is host-computed -> a
  criterion-2 (neuromorphic-hardware-port) structure residual. The numpy de-risk proved a
  self-organized RF bank (learned by a local rate-Hebbian rule from oriented-edge input)
  reproduces the host bank's orientation selectivity (OSI 1.0) + pixel-similarity geometry
  (RSA-to-host 0.988), with no-learning + noise-input controls collapsing on OSI. This runner
  does the SAME learning ON the REAL SimulationBridge (spiking), via the already-plastic
  retina->cortex_v1_simple pathway + the bridge's own rate-window Hebbian rule -- NO sim/ edit.

THE ON-BRIDGE MECHANISM (reuse-by-import, no sim/ edit)
------------------------------------------------------
  * Minimal bridge: a `retina` region (2*32*32 = 2048 ON/OFF neurons, externally driven) + a
    `cortex_v1_simple` region (8 orient x 4 freq x 16x16 pos = 8192), connected by the
    retina->cortex_v1_simple PLASTIC, gated (`visual_cortex_v1`) pathway (exactly the deployed
    g11 wiring). Izhikevich spiking neurons.
  * The pathway is installed on an ISOTROPIC LOCAL RF SUPPORT: each V1 cell connects to ALL
    retina pixels (both ON + OFF channels) within radius-`R` of its retinotopic centre -- a
    NON-oriented, biologically-legitimate retinotopic-locality prior. Initial weights are
    RANDOM (uniform). => any orientation in the final RFs MUST come from LEARNING, not the
    support and not a Gabor formula (the load-bearing anti-cheat).
  * Development: open `visual_cortex_v1`, drive the retina with full-field ORIENTED gratings
    (random orientation/phase/frequency each presentation), let the bridge's rate-window
    Hebbian (BCM-like: co-activity trace potentiation, soft-bounded, + weight decay) +
    homeostatic threshold adaptation refine the weights from spikes. Then FREEZE the gate
    (critical-period close). rate-Hebbian NOT symmetric STDP (CYCLE-95: STDP is the wrong rule
    for symmetric co-occurrence).
  * Read-out: (1) reconstruct each V1 cell's RF from the LEARNED weights (signed ON-OFF patch)
    and measure OSI per neuron; (2) encode the Option-B test shapes by the V1 FIRING code and
    measure RSA-to-host + within>between margin + orientation decode.

GO BAR (the numpy de-risk's bar, on the spiking substrate)
----------------------------------------------------------
  (1) The LEARNED weights develop orientation selectivity: POST-learning OSI-frac >> PRE-learning
      (random init) OSI-frac; POST mean OSI clears the self-org gate.
  (2) Geometry preserved: the V1-FIRING code RSA-to-host high + within>between margin positive.
ANTI-CHEATS (the discriminating controls -- OSI is where they collapse, per the numpy de-risk):
  (a) PRE vs POST: random-init RFs (isotropic support, random weights) are NOT oriented; learning
      makes them oriented.
  (b) SHUFFLE-STIMULUS: an identical bridge developed on PIXEL-SHUFFLED (orientation-destroyed)
      stimuli -> OSI does NOT rise (the structure comes from the input statistics, not the
      substrate; catalog L.05 "wave/image content matters").
  (c) The selectivity is from the LEARNED retina->V1 weights, never the fixed Gabor bank (never
      applied); the RF support is isotropic (carries no orientation).

Run:
  SIM_BACKEND=cupy python -u -m research.runners._b1_v1_selforg_onbridge_derisk \
      --seeds 42 43 44 --dev-steps 24000 \
      --out research/findings/raw/_b1_v1_selforg_onbridge_derisk.json
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402
# --- reuse-by-import: the numpy de-risk's metrics/shapes/host reference (apples-to-apples) ---
from research.runners._b1_v1_selforg_rf_derisk import (  # noqa: E402
    _render_bar_image,
    build_shape_set,
    build_fine_orientation_set,
    gabor_orientation_tuning,
    build_host_v1_matrix,
    encode_host_v1,
    within_between_margin,
    rsa_between_codes,
    rsa_pixel_provenance,
    orientation_decode_accuracy,
)
from sim.visual_cortex import (  # noqa: E402
    N_ORIENTATIONS,
    N_FREQUENCIES,
    V1_POSITIONS_PER_DIM,
    RETINA_SIZE,
)

PATCH = 9  # matches the numpy de-risk's OSI patch (radius-4 -> 9x9); gabor_orientation_tuning expects PATCH_PIX=81


# ============================================================================
# 1. Isotropic local RF support (NON-oriented) + random init weights.
# ============================================================================

def build_isotropic_support(n_orient, n_freq, n_pos, retina_size, radius):
    """(pre_rel, post_rel) for ALL retina pixels (ON+OFF) within `radius` of each V1 cell's
    retinotopic centre -- an ISOTROPIC local support (carries NO orientation). Every V1 cell at
    a position shares the same support; different cells get different RANDOM weights (below)."""
    stride = retina_size // n_pos
    pre, post = [], []
    # precompute the per-position support once (all cells at a position share it)
    for pos_y in range(n_pos):
        for pos_x in range(n_pos):
            cx = pos_x * stride + stride // 2
            cy = pos_y * stride + stride // 2
            support = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy > radius * radius:
                        continue  # disc, not square -> isotropic
                    px, py = cx + dx, cy + dy
                    if not (0 <= px < retina_size and 0 <= py < retina_size):
                        continue
                    for ch in (0, 1):  # ON, OFF
                        support.append(ch * (retina_size * retina_size) + py * retina_size + px)
            base_pos = pos_y * n_pos + pos_x
            for orient_i in range(n_orient):
                for freq_i in range(n_freq):
                    v1 = (orient_i * (n_freq * n_pos * n_pos)
                          + freq_i * (n_pos * n_pos) + base_pos)
                    for r_idx in support:
                        pre.append(r_idx)
                        post.append(v1)
    return np.asarray(pre, dtype=np.int64), np.asarray(post, dtype=np.int64)


# ============================================================================
# 2. Minimal on-bridge V1 (retina + cortex_v1_simple + plastic pathway).
# ============================================================================

def build_v1_bridge(seed, n_orient, n_freq, n_pos, retina_size, radius,
                    init_weight_mean, init_weight_jitter,
                    hebb_lr, hebb_decay, hebb_max, coact_decay, coact_thresh,
                    homeo_target=0.05, homeo_ema_alpha=0.005, homeo_adapt_rate=0.002,
                    syn_scaling=True, syn_scaling_rate=0.02,
                    n_inh=0, inh_exc_w=6.0, inh_inh_w=8.0, inh_density=0.25,
                    rule="hebbian", stdp_a_plus=0.006, stdp_a_minus=0.0075, stdp_tau=20.0):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType

    n_retina = 2 * retina_size * retina_size
    n_v1 = n_orient * n_freq * n_pos * n_pos

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    regions = [
        BrainRegion(name="retina", n_neurons=n_retina, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="cortex_v1_simple", n_neurons=n_v1, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
    ]
    # Declare the plastic pathway at density 0; we INSTALL the isotropic local support below
    # (add_missing=True). NO plasticity_gate: this is the only pathway, so gating is unneeded, and
    # a None gate takes the byte-identical clip/decay branch (avoids the gate-capacity foot-gun a
    # CSR-rebuild-after-add_missing would otherwise trip). Freeze = enable_hebbian_learning=False.
    pathways = [
        RegionPathway(from_region="retina", to_region="cortex_v1_simple",
                      density=0.001, weight_mean=init_weight_mean, weight_jitter=init_weight_jitter,
                      plastic=True),
    ]
    # OPTIONAL lateral inhibition (SAILnet/Foldiak competition = the ingredient the numpy mechanism
    # A used for orientation SELECTIVITY): a recurrent FS pool. V1 excites it; it inhibits V1 ->
    # soft k-WTA sparsening -> each V1 cell wins for a NARROW stimulus band -> reinforcement
    # concentrates -> oriented RF. Wired through the FRAMEWORK (real density) so the FS pool's
    # inhibitory trait is set (g11 sc_fs gotcha: set_pathway_weights would leave it excitatory).
    if n_inh > 0:
        regions.append(BrainRegion(
            name="v1_inh", n_neurons=int(n_inh), exc_fraction=0.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name))
        pathways.append(RegionPathway(from_region="cortex_v1_simple", to_region="v1_inh",
                                      density=inh_density, weight_mean=inh_exc_w, weight_jitter=0.1,
                                      plastic=False))
        pathways.append(RegionPathway(from_region="v1_inh", to_region="cortex_v1_simple",
                                      density=inh_density, weight_mean=inh_inh_w, weight_jitter=0.1,
                                      plastic=False))
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed   # cfg.seed seeds the substrate (NOT actual_seed_used)
    cfg.enable_ou_process = False
    cfg.enable_structural_plasticity = False   # keep nnz fixed (no synaptogenesis polluting the learned RF)
    # Learning rule. 'hebbian' = potentiation-only rate-window BCM (symmetric). 'stdp' = causal
    # spike-timing (LTP pre->post, LTD post->pre) -- the RIGHT rule for FEEDFORWARD sensory RF
    # refinement (Song-Miller-Abbott 2000; Burbank 2015 -> Gabor RFs). Its LTD provides the
    # input-specific DEPRESSION that potentiation-only Hebbian lacks (which saturates -> blobs).
    # NB CYCLE-95's "STDP is wrong" verdict was about SYMMETRIC word co-occurrence (no pre/post
    # order); feedforward retina->V1 HAS a causal order, so STDP applies.
    _stdp = rule in ("stdp", "both")
    _hebb = rule in ("hebbian", "both")
    cfg.enable_stdp = _stdp
    cfg.stdp_a_plus = stdp_a_plus
    cfg.stdp_a_minus = stdp_a_minus
    cfg.stdp_tau_plus_ms = stdp_tau
    cfg.stdp_tau_minus_ms = stdp_tau
    cfg.stdp_w_min = 0.0
    cfg.stdp_w_max = hebb_max         # keep the STDP soft-bound ABOVE the design weights (the w_max gotcha)
    cfg.enable_hebbian_learning = _hebb
    cfg.hebbian_rate_window = True          # BCM/rate-Hebbian co-activity (the right rule; CYCLE-95)
    cfg.hebbian_coactivity_decay = coact_decay
    cfg.hebbian_coactivity_thresh = coact_thresh
    cfg.hebbian_learning_rate = hebb_lr
    cfg.hebbian_weight_decay = hebb_decay
    cfg.hebbian_max_weight = hebb_max
    # Miller-MacKay subtractive normalization (2026-07-31): env-exposed so the ON/OFF arms are one flag.
    cfg.hebbian_mean_subtract = float(os.environ.get("HEBB_MEAN_SUB", "0.0"))
    cfg.hebbian_oja = float(os.environ.get("HEBB_OJA", "0.0"))
    cfg.hebbian_min_weight = 0.0
    cfg.enable_homeostasis = True           # threshold adaptation = the activity-normalizing competition term
    cfg.homeostasis_target_rate = homeo_target
    cfg.homeostasis_ema_alpha = homeo_ema_alpha        # faster EMA for the de-risk window
    cfg.homeostasis_threshold_adapt_rate = homeo_adapt_rate
    # Turrigiano multiplicative synaptic scaling = the COMPETITIVE weight normalization (Oja-like):
    # Hebbian selectively potentiates preferred synapses; scaling normalizes the postsynaptic total
    # down -> winners grow at the expense of losers -> BCM-like orientation selectivity emerges.
    cfg.enable_synaptic_scaling = bool(syn_scaling)
    cfg.synaptic_scaling_rate = syn_scaling_rate

    rt = RuntimeState()
    rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=rt, gpu_config=GPUConfig())
    bridge._initialize_simulation_data()

    rm = bridge.region_manager
    r0 = int(rm.indices("retina")[0])
    v0 = int(rm.indices("cortex_v1_simple")[0])

    # Install the isotropic local support with RANDOM init weights.
    pre_rel, post_rel = build_isotropic_support(n_orient, n_freq, n_pos, retina_size, radius)
    rng = np.random.default_rng(seed * 7 + 1)
    w_init = np.abs(rng.normal(init_weight_mean, init_weight_jitter, size=pre_rel.shape)).astype(np.float32)
    w_init = np.clip(w_init, 0.0, hebb_max)
    bridge.set_pathway_weights(
        pathway_name="retina_to_v1_simple_selforg",
        pre_indices=(pre_rel + r0), post_indices=(post_rel + v0),
        weights=w_init, add_missing=True,
    )
    return bridge, r0, v0, n_retina, n_v1


# ============================================================================
# 3. Stimulus: full-field oriented gratings (the V1-activating input statistics).
# ============================================================================

def render_oriented_field(rng, retina_size=RETINA_SIZE, shuffle=False):
    """A full-field ORIENTED grating (random orientation/frequency/phase) -> (2,H,W) ON/OFF image.
    Every retinotopic patch sees oriented structure. If shuffle=True, the pixels are permuted
    (orientation destroyed) -- the L.05 content-matters control."""
    H = W = retina_size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    theta = rng.uniform(0.0, math.pi)
    freq = rng.uniform(0.08, 0.30)
    phase = rng.uniform(0.0, 2 * math.pi)
    proj = xx * math.cos(theta) + yy * math.sin(theta)
    grating = np.cos(2 * math.pi * freq * proj + phase).astype(np.float32)  # signed [-1,1]
    # windowed to a random blob so it is a localized oriented edge, not a global plane wave
    cx = rng.uniform(0.25, 0.75) * W
    cy = rng.uniform(0.25, 0.75) * H
    sigma = rng.uniform(0.35, 0.6) * W      # moderate window: many patches lit per image, but not so global that
                                            # lateral competition becomes non-local (co-active cells compete locally)
    env = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)).astype(np.float32)
    signed = grating * env
    if shuffle:
        flat = signed.reshape(-1).copy()
        rng.shuffle(flat)          # destroy spatial (orientation) structure, keep the marginal
        signed = flat.reshape(H, W)
    on = np.maximum(signed, 0.0)
    off = np.maximum(-signed, 0.0)
    return np.stack([on, off], axis=0).astype(np.float32)


# ============================================================================
# 4. Drive helpers.
# ============================================================================

def _drive_image(bridge, r0, n_retina, image, drive_pA, xp):
    flat = image.reshape(-1).astype(np.float32) * drive_pA
    bridge.cp_external_input_current[:] = 0.0
    seg = xp.asarray(flat) if xp is not None else flat
    bridge.cp_external_input_current[r0:r0 + n_retina] = seg


def _freeze(bridge):
    """Critical-period close: freeze ALL plasticity (weights + thresholds) for a stable read-out."""
    bridge.core_config.enable_hebbian_learning = False
    bridge.core_config.enable_synaptic_scaling = False
    bridge.core_config.enable_homeostasis = False


def develop(bridge, r0, n_retina, n_steps, drive_pA, present_steps, seed, xp, shuffle=False):
    """Developmental phase: stream oriented gratings, run the plastic bridge."""
    rng = np.random.default_rng(seed * 101 + (7 if shuffle else 3))
    steps_done = 0
    while steps_done < n_steps:
        img = render_oriented_field(rng, shuffle=shuffle)
        _drive_image(bridge, r0, n_retina, img, drive_pA, xp)
        for _ in range(present_steps):
            bridge._run_one_simulation_step()
            steps_done += 1
            if steps_done >= n_steps:
                break
    bridge.cp_external_input_current[:] = 0.0


# ============================================================================
# 5. Read learned RFs -> OSI; encode test shapes by V1 FIRING -> RSA.
# ============================================================================

def read_v1_rfs(bridge, r0, v0, n_retina, n_v1, n_orient, n_freq, n_pos, retina_size, radius):
    """Reconstruct each V1 cell's signed ON-OFF RF patch (9x9) from the LEARNED weights."""
    block = bridge.cp_connections[r0:r0 + n_retina, v0:v0 + n_v1]
    W = np.asarray(to_host(block.todense())).astype(np.float32)  # (n_retina, n_v1): W[pre, post]
    stride = retina_size // n_pos
    half = PATCH // 2
    rfs = np.zeros((n_v1, PATCH * PATCH), dtype=np.float32)
    RS2 = retina_size * retina_size
    for c in range(n_v1):
        pos = c % (n_pos * n_pos)
        pos_x = pos % n_pos
        pos_y = pos // n_pos
        cx = pos_x * stride + stride // 2
        cy = pos_y * stride + stride // 2
        patch = np.zeros((PATCH, PATCH), dtype=np.float32)
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                px, py = cx + dx, cy + dy
                if not (0 <= px < retina_size and 0 <= py < retina_size):
                    continue
                on_idx = 0 * RS2 + py * retina_size + px
                off_idx = 1 * RS2 + py * retina_size + px
                patch[dy + half, dx + half] = W[on_idx, c] - W[off_idx, c]  # signed ON-OFF RF
        rfs[c] = patch.reshape(-1)
    return rfs


def raw_weight_stats(bridge, r0, v0, n_retina, n_v1, retina_size):
    """RAW retina->V1 weights split by ON/OFF channel, plus per-cell incoming L2 norm.

    ⚠️ WHY THIS EXISTS. Every fix proposed for lane D so far is predicated on "the weights collapsed" -- and that
    had NEVER been measured. The only weight statistic recorded (`saturation`) is computed from `rf_post`, which is
    the SIGNED ON-OFF DIFFERENCE, so it cannot distinguish the two hypotheses that need OPPOSITE fixes:

      * WEIGHT COLLAPSE          -> on_mean and off_mean both ~0, l2 per cell ~0.  Fix: the rule is destroying mass.
      * COMMON-MODE CONVERGENCE  -> on_mean and off_mean both LARGE and nearly EQUAL, l2 per cell LARGE.
                                    The signed difference vanishes while the raw weights are healthy.
                                    Fix: remove the common mode; adding weight mass would do nothing.

    `frac_rf_near_zero` reads ~0.45 under BOTH, and is additionally floored by geometry -- roughly 32 of the 81
    patch pixels fall outside the radius-4 disc and are structurally zero -- so it can never have answered this.
    Read-only: no bridge state is modified.
    """
    block = bridge.cp_connections[r0:r0 + n_retina, v0:v0 + n_v1]
    W = np.asarray(to_host(block.todense())).astype(np.float32)      # (n_retina, n_v1) = W[pre, post]
    RS2 = retina_size * retina_size
    on, off = W[0:RS2, :], W[RS2:2 * RS2, :]
    l2 = np.sqrt((W ** 2).sum(axis=0))                                # incoming L2 norm per V1 cell
    return dict(
        on_mean=round(float(on.mean()), 6), off_mean=round(float(off.mean()), 6),
        on_absmax=round(float(np.abs(on).max()), 6), off_absmax=round(float(np.abs(off).max()), 6),
        # THE DISCRIMINATOR: near 0 => the channels cancel (common mode); large => a genuine signed RF.
        on_minus_off_mean=round(float(on.mean() - off.mean()), 6),
        raw_mean_abs=round(float(np.abs(W).mean()), 6),
        l2_mean=round(float(l2.mean()), 6), l2_min=round(float(l2.min()), 6),
        l2_max=round(float(l2.max()), 6), l2_std=round(float(l2.std()), 6),
        frac_cells_l2_near_zero=round(float((l2 < 1e-6).mean()), 4),
        frac_raw_near_zero=round(float((np.abs(W) < 1e-6).mean()), 4),
    )


def plasticity_event_stats(bridge, v1_idx):
    """The plasticity-event count and the DEVELOPMENTAL activity EMA -- both already computed, neither ever read.

    `num_potentiation_events` is maintained by the Hebbian path in sim/bridge.py and nothing has ever looked at it,
    so "the rule fired but did nothing" and "the rule never fired" have been indistinguishable for this whole arc.
    Read-only, and defensive: a missing attribute records None rather than raising, because this is instrumentation
    and must never be able to fail the run it is measuring.
    """
    # NB `num_potentiation_events` is a LOCAL in the Hebbian block (sim/bridge.py:7865/7883), NOT an attribute --
    # reading it by that name returns None forever, which is what a first pass here did. It accumulates into
    # `_mock_total_plasticity_events` (bridge.py:7931), which is the field that actually exists.
    out = {}
    n_ev = getattr(bridge, "_mock_total_plasticity_events", None)
    try:
        out["total_plasticity_events"] = int(n_ev) if n_ev is not None else None
    except Exception:
        out["total_plasticity_events"] = None
    ema = getattr(bridge, "cp_neuron_activity_ema", None)
    if ema is not None and len(v1_idx):
        try:
            e = np.asarray(to_host(ema))[v1_idx].astype(float)
            out.update(v1_activity_ema_mean=round(float(e.mean()), 8),
                       v1_activity_ema_max=round(float(e.max()), 8),
                       v1_activity_ema_min=round(float(e.min()), 8),
                       frac_v1_ema_zero=round(float((e <= 0).mean()), 4))
        except Exception:
            out["v1_activity_ema_mean"] = None
    else:
        out["v1_activity_ema_mean"] = None
    return out


def encode_v1_firing(bridge, r0, v0, n_retina, n_v1, images, drive_pA, read_steps, settle_steps, xp):
    """Drive each test image; count V1 spikes over a read window -> the V1 FIRING code (N, n_v1)."""
    codes = np.zeros((images.shape[0], n_v1), dtype=np.float32)
    for i in range(images.shape[0]):
        _drive_image(bridge, r0, n_retina, images[i], drive_pA, xp)
        for _ in range(settle_steps):
            bridge._run_one_simulation_step()
        counts = np.zeros(n_v1, dtype=np.float32)
        for _ in range(read_steps):
            bridge._run_one_simulation_step()
            fired = to_host(bridge.cp_firing_states[v0:v0 + n_v1])
            counts += np.asarray(fired, dtype=np.float32)
        codes[i] = counts
    bridge.cp_external_input_current[:] = 0.0
    return codes


# ============================================================================
# 6. Per-seed run.
# ============================================================================

def run_seed(seed, a):
    from sim.backend import get_backend
    xp, backend = get_backend()
    xp = xp if backend == "cupy" else None

    n_orient, n_freq, n_pos = a.n_orient, a.n_freq, a.n_pos
    retina_size, radius = a.retina_size, a.radius
    n_v1 = n_orient * n_freq * n_pos * n_pos

    t0 = time.time()
    # ---- LEARN bridge (oriented input) ----
    bridge, r0, v0, n_retina, _ = build_v1_bridge(
        seed, n_orient, n_freq, n_pos, retina_size, radius,
        a.init_weight_mean, a.init_weight_jitter,
        a.hebb_lr, a.hebb_decay, a.hebb_max, a.coact_decay, a.coact_thresh,
        syn_scaling=bool(a.syn_scaling), syn_scaling_rate=a.syn_scaling_rate,
        n_inh=a.n_inh, inh_exc_w=a.inh_exc_w, inh_inh_w=a.inh_inh_w, inh_density=a.inh_density,
        homeo_target=a.homeo_target, homeo_ema_alpha=a.homeo_ema_alpha, homeo_adapt_rate=a.homeo_adapt_rate,
        rule=a.rule)

    # PRE-learning RFs (random init on isotropic support) = the no-learning control (a)
    rf_pre = read_v1_rfs(bridge, r0, v0, n_retina, n_v1, n_orient, n_freq, n_pos, retina_size, radius)
    osi_pre_mean, osi_pre_frac = gabor_orientation_tuning(rf_pre)

    develop(bridge, r0, n_retina, a.dev_steps, a.drive_pA, a.present_steps, seed, xp, shuffle=False)
    _freeze(bridge)   # critical-period close: freeze all plasticity for read-out

    rf_post = read_v1_rfs(bridge, r0, v0, n_retina, n_v1, n_orient, n_freq, n_pos, retina_size, radius)
    osi_post_mean, osi_post_frac = gabor_orientation_tuning(rf_post)

    # ---- SHUFFLE-STIMULUS control (b): identical bridge, orientation-destroyed input ----
    bridge_sh, r0s, v0s, n_ret_s, _ = build_v1_bridge(
        seed, n_orient, n_freq, n_pos, retina_size, radius,
        a.init_weight_mean, a.init_weight_jitter,
        a.hebb_lr, a.hebb_decay, a.hebb_max, a.coact_decay, a.coact_thresh,
        syn_scaling=bool(a.syn_scaling), syn_scaling_rate=a.syn_scaling_rate,
        n_inh=a.n_inh, inh_exc_w=a.inh_exc_w, inh_inh_w=a.inh_inh_w, inh_density=a.inh_density,
        homeo_target=a.homeo_target, homeo_ema_alpha=a.homeo_ema_alpha, homeo_adapt_rate=a.homeo_adapt_rate,
        rule=a.rule)
    develop(bridge_sh, r0s, n_ret_s, a.dev_steps, a.drive_pA, a.present_steps, seed, xp, shuffle=True)
    _freeze(bridge_sh)
    rf_shuf = read_v1_rfs(bridge_sh, r0s, v0s, n_ret_s, n_v1, n_orient, n_freq, n_pos, retina_size, radius)
    osi_shuf_mean, osi_shuf_frac = gabor_orientation_tuning(rf_shuf)

    # ---- Geometry: V1 FIRING code on the Option-B test shapes (RSA-to-host) ----
    rng = np.random.default_rng(seed)
    images, labels, _ = build_shape_set(a.n_categories, a.n_exemplars, rng, image_size=retina_size)
    # host reference (the real Gabor V1 code) at the SAME architecture
    Whost = build_host_v1_matrix()  # uses module defaults 8x4x16x16 / retina 32 -> requires those params
    host_code = encode_host_v1(images, Whost)
    host_w, host_b, host_m = within_between_margin(host_code, labels)
    host_rsa_pix = rsa_pixel_provenance(images, host_code)

    v1_code_post = encode_v1_firing(bridge, r0, v0, n_retina, n_v1, images,
                                    a.drive_pA, a.read_steps, a.settle_steps, xp)
    w_p, b_p, m_p = within_between_margin(v1_code_post, labels)
    rsa_host_post = rsa_between_codes(v1_code_post, host_code)
    rsa_pix_post = rsa_pixel_provenance(images, v1_code_post)

    # orientation decode (discriminating stimulus), on the V1 firing code
    oimgs, olabs = build_fine_orientation_set(a.n_orient_dec, a.n_orient_ex, seed + 100)
    host_ocode = encode_host_v1(oimgs, Whost)
    host_decode = orientation_decode_accuracy(host_ocode, olabs)
    v1_ocode = encode_v1_firing(bridge, r0, v0, n_retina, n_v1, oimgs,
                               a.drive_pA, a.read_steps, a.settle_steps, xp)
    decode_post = orientation_decode_accuracy(v1_ocode, olabs)

    # spike sanity: mean V1 firing per read step on the shape set
    v1_rate = float(v1_code_post.sum() / max(1, images.shape[0]) / max(1, a.read_steps) / n_v1)

    elapsed = time.time() - t0

    # ---- verdict ----
    osi_self_gate = 0.5          # majority of learned RFs oriented (numpy de-risk gate)
    osi_ctrl_ceiling = max(osi_pre_frac, osi_shuf_frac) + 0.05
    rsa_gate = 0.6
    margin_gate = min(0.15, 0.5 * host_m)

    learned_oriented = osi_post_frac >= osi_self_gate
    lift_over_controls = (osi_post_frac >= osi_pre_frac + 0.15) and (osi_post_frac >= osi_shuf_frac + 0.15)
    geometry_ok = (rsa_host_post >= rsa_gate) and (m_p >= margin_gate)

    if learned_oriented and lift_over_controls and geometry_ok:
        verdict = "GO"
    elif lift_over_controls and (osi_post_frac >= 0.3):
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"

    # WEIGHT SATURATION -- the quantity that DIAGNOSES this runner's failure, and which it never recorded.
    # The 2026-07-30 diagnosis predicts every retina->V1 synapse pins at hebbian_max_weight because synaptic
    # scaling is driven by a permanently-positive rate error (homeo_target 0.012 vs a measured rate of
    # 0.0004-0.0010) and is applied uniformly, ungated on spiking, for all 40k steps. If true, W[ON] and W[OFF]
    # both pin, their SIGNED difference is identically 0, and OSI reads exactly 0.0 -- i.e. learning DELETES the
    # chance structure present at random init rather than failing to add to it.
    # This is arithmetic until it is measured, and nothing here measured it. Note this runner's own help text at
    # :510 already warned the target "MUST be reachable by V1 (~0.012) or scaling saturates", and :201 already
    # named the missing "input-specific DEPRESSION that potentiation-only Hebbian lacks (which saturates ->
    # blobs)". Both failure modes were documented in this file BEFORE they occurred; nothing checked either.
    # A warning in a docstring cannot fail. A recorded number can.
    _rp = np.asarray(rf_post, dtype=float)
    _sat = dict(
        w_mean=round(float(np.abs(_rp).mean()), 6),
        w_absmax=round(float(np.abs(_rp).max()), 6),
        # rf_post is the SIGNED ON-OFF difference, so total saturation shows up as this collapsing to ~0.
        frac_rf_near_zero=round(float((np.abs(_rp) < 1e-6).mean()), 4),
        frac_cells_all_zero=round(float((np.abs(_rp) < 1e-6).all(axis=1).mean()), 4)
        if _rp.ndim == 2 else float("nan"),
    )

    # The RAW weights, which no lane-D run has ever recorded. `saturation` above is derived from the SIGNED
    # ON-OFF difference and is blind to the collapse-vs-common-mode distinction these two calls resolve.
    _raw = raw_weight_stats(bridge, r0, v0, n_retina, n_v1, retina_size)
    _plast = plasticity_event_stats(bridge, np.arange(v0, v0 + n_v1))
    print("  raw W: on_mean %.5f off_mean %.5f  on-off %.5f | l2 mean %.4f min %.4f max %.4f | "
          "cells l2~0 %.3f" % (_raw["on_mean"], _raw["off_mean"], _raw["on_minus_off_mean"],
                               _raw["l2_mean"], _raw["l2_min"], _raw["l2_max"],
                               _raw["frac_cells_l2_near_zero"]))
    print("  plasticity: events=%s | v1 activity EMA mean=%s frac_zero=%s"
          % (_plast.get("total_plasticity_events"), _plast.get("v1_activity_ema_mean"),
             _plast.get("frac_v1_ema_zero")))
    # Name the diagnosis the numbers support, so the artifact carries a verdict rather than raw columns someone
    # has to re-interpret. Threshold is relative to the raw weight scale, not an absolute.
    _scale = max(_raw["raw_mean_abs"], 1e-9)
    if _raw["l2_mean"] < 1e-6:
        _diag = "WEIGHT COLLAPSE — incoming mass is gone; a normalization fix would have nothing to normalize"
    elif abs(_raw["on_minus_off_mean"]) < 0.05 * _scale:
        _diag = ("COMMON-MODE CONVERGENCE — raw weights are healthy but ON and OFF cancel; adding weight mass "
                 "cannot help, the common mode has to be removed")
    else:
        _diag = "NEITHER — raw weights carry a signed ON-OFF structure; look downstream of the weights"
    print("  => %s" % _diag)

    return dict(
        seed=seed, backend=backend, n_v1=n_v1, elapsed_s=round(elapsed, 1),
        v1_firing_rate=round(v1_rate, 4),
        saturation=_sat,
        raw_weights=_raw,
        plasticity=_plast,
        weight_diagnosis=_diag,
        osi=dict(
            pre_random=dict(mean=round(osi_pre_mean, 4), frac_gt0_5=round(osi_pre_frac, 4)),
            post_learned=dict(mean=round(osi_post_mean, 4), frac_gt0_5=round(osi_post_frac, 4)),
            shuffle_ctrl=dict(mean=round(osi_shuf_mean, 4), frac_gt0_5=round(osi_shuf_frac, 4)),
        ),
        geometry=dict(
            host_reference=dict(margin=round(host_m, 4), rsa_vs_pixels=round(host_rsa_pix, 4),
                                orient_decode=round(host_decode, 4)),
            v1_firing_post=dict(within=round(w_p, 4), between=round(b_p, 4), margin=round(m_p, 4),
                                rsa_vs_host=round(rsa_host_post, 4), rsa_vs_pixels=round(rsa_pix_post, 4),
                                orient_decode=round(decode_post, 4)),
        ),
        gates=dict(osi_self_gate=osi_self_gate, osi_ctrl_ceiling=round(osi_ctrl_ceiling, 4),
                   rsa_gate=rsa_gate, margin_gate=round(margin_gate, 4)),
        learned_oriented=bool(learned_oriented), lift_over_controls=bool(lift_over_controls),
        geometry_ok=bool(geometry_ok), verdict=verdict,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-orient", type=int, default=N_ORIENTATIONS)
    ap.add_argument("--n-freq", type=int, default=N_FREQUENCIES)
    ap.add_argument("--n-pos", type=int, default=V1_POSITIONS_PER_DIM)
    ap.add_argument("--retina-size", type=int, default=RETINA_SIZE)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--dev-steps", type=int, default=40000)
    ap.add_argument("--present-steps", type=int, default=40,
                    help="steps per stimulus presentation (V1 has a ~25-step firing onset latency)")
    ap.add_argument("--drive-pA", type=float, default=1200.0)
    ap.add_argument("--settle-steps", type=int, default=25)   # let V1 charge past its onset latency before reading
    ap.add_argument("--read-steps", type=int, default=15)
    ap.add_argument("--init-weight-mean", type=float, default=30.0)
    ap.add_argument("--init-weight-jitter", type=float, default=7.0)
    ap.add_argument("--hebb-lr", type=float, default=0.05)
    ap.add_argument("--hebb-decay", type=float, default=0.00002)
    ap.add_argument("--hebb-max", type=float, default=70.0)
    ap.add_argument("--coact-decay", type=float, default=0.85)
    ap.add_argument("--coact-thresh", type=float, default=0.03)
    ap.add_argument("--syn-scaling", type=int, default=1)
    ap.add_argument("--syn-scaling-rate", type=float, default=0.02)
    ap.add_argument("--n-inh", type=int, default=0,
                    help="lateral-inhibition FS pool size (0 = none; SAILnet/Foldiak competition for selectivity)")
    ap.add_argument("--inh-exc-w", type=float, default=6.0)
    ap.add_argument("--inh-inh-w", type=float, default=12.0)
    ap.add_argument("--inh-density", type=float, default=0.25)
    ap.add_argument("--homeo-target", type=float, default=0.012,
                    help="homeostatic/scaling target rate -- MUST be reachable by V1 (~0.012) or scaling saturates weights")
    ap.add_argument("--homeo-ema-alpha", type=float, default=0.01)
    ap.add_argument("--homeo-adapt-rate", type=float, default=0.004)
    ap.add_argument("--rule", type=str, default="hebbian", choices=["hebbian", "stdp", "both"])
    ap.add_argument("--n-categories", type=int, default=4)
    ap.add_argument("--n-exemplars", type=int, default=4)
    ap.add_argument("--n-orient-dec", type=int, default=8)
    ap.add_argument("--n-orient-ex", type=int, default=8)
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_b1_v1_selforg_onbridge_derisk.json")
    a = ap.parse_args()

    os.environ.setdefault("SIM_BACKEND", "cupy")
    print(f"[B1 on-bridge V1 self-org] seeds={a.seeds} dev_steps={a.dev_steps} "
          f"arch={a.n_orient}x{a.n_freq}x{a.n_pos}x{a.n_pos} radius={a.radius}", flush=True)

    per_seed = []
    for s in a.seeds:
        r = run_seed(s, a)
        per_seed.append(r)
        print(json.dumps(r, indent=2), flush=True)

    def col(f):
        return [f(r) for r in per_seed]

    verdicts = [r["verdict"] for r in per_seed]
    all_go = all(v == "GO" for v in verdicts)
    overall = "GO" if all_go else ("PARTIAL" if all(v in ("GO", "PARTIAL") for v in verdicts) else "NEGATIVE")

    summary = dict(
        overall_verdict=overall,
        seeds=a.seeds,
        per_seed_verdicts=verdicts,
        osi_pre_frac_mean=round(float(np.mean(col(lambda r: r["osi"]["pre_random"]["frac_gt0_5"]))), 4),
        osi_post_frac_mean=round(float(np.mean(col(lambda r: r["osi"]["post_learned"]["frac_gt0_5"]))), 4),
        osi_shuffle_frac_mean=round(float(np.mean(col(lambda r: r["osi"]["shuffle_ctrl"]["frac_gt0_5"]))), 4),
        osi_post_mean_mean=round(float(np.mean(col(lambda r: r["osi"]["post_learned"]["mean"]))), 4),
        rsa_vs_host_mean=round(float(np.mean(col(lambda r: r["geometry"]["v1_firing_post"]["rsa_vs_host"]))), 4),
        margin_mean=round(float(np.mean(col(lambda r: r["geometry"]["v1_firing_post"]["margin"]))), 4),
        orient_decode_mean=round(float(np.mean(col(lambda r: r["geometry"]["v1_firing_post"]["orient_decode"]))), 4),
        host_decode_mean=round(float(np.mean(col(lambda r: r["geometry"]["host_reference"]["orient_decode"]))), 4),
        v1_firing_rate_mean=round(float(np.mean(col(lambda r: r["v1_firing_rate"]))), 4),
    )

    out = dict(summary=summary, per_seed=per_seed)
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 90, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[written] {outp}", flush=True)


if __name__ == "__main__":
    main()
