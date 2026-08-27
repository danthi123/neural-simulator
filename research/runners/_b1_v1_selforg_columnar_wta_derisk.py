"""B1 ON-BRIDGE V1 self-org — COLUMNAR k-WTA COMPETITION variant.

THE WALL (2026-08-14 / 2026-08-26): on the real spiking bridge, the plastic
retina->cortex_v1_simple rate-Hebbian rule learns a COMMON MODE — ON and OFF
channels potentiate to nearly-identical weights (`on_minus_off_mean` ~ 0), so the
signed ON-OFF receptive field cancels and osi_post_frac ~ 0.009 « the 0.5 gate.
The numpy GO (2026-06-21) avoided this by learning SIGNED bipolar patches with
sparse k-WTA competition + anti-Hebbian lateral inhibition; its own docstring
warns that "learning on the non-negative ON/OFF cone with weak inhibition
collapses to all-positive blobs — the documented failure." That IS the on-bridge
common mode.

WHY A NEW LEVER (not a re-run): the prior on-bridge inhibition
(`_b1_v1_selforg_onbridge_derisk --n-inh 64`) was a SINGLE GLOBAL FS pool with
UNIFORM RANDOM connectivity → uniform gain control, not competition. It lowered
everyone's gain together and never created the LOCAL cell-vs-cell competition that
drives specialization (finding 2026-08-14: "uniform gain control, not per-pair
decorrelation"). THIS runner installs COLUMNAR (iso-position) k-WTA competition:
one FS interneuron per retinotopic position pools ALL orientation/frequency
channels at that position and inhibits the whole hypercolumn back — so the cells
at a position COMPETE for each stimulus (winner-take-all). The mechanistic bet:
iso-position WTA splits a column's cells into ORIENTATION- and PHASE-opponent
partners (cell A wins horizontal-phase-0, cell B wins horizontal-phase-π; each
then fires SELECTIVELY and so develops a SIGNED RF), breaking the phase-average
that creates the common mode — the local competition a global pool cannot supply.
Companion process: per-cell homeostatic synaptic scaling (Turrigiano) + per-cell
threshold homeostasis normalize each cell's total input so a perpetual winner is
scaled/raised down and losers find a niche (the SAILnet homeostatic-competition
dynamic, with FIXED — not plastic — inhibition; plastic anti-Hebbian is the named
NEXT lever if this holds the wall).

BRAIN-BASED: the competition is real spiking FS interneurons inhibiting real
spiking pyramidal cells through synapses; homeostasis/scaling are the substrate's
own per-neuron mechanisms. NO sim/ edit (reuse-by-import + set_pathway_weights,
the same install path the Gabor bank + the FF support use).

GATE (the flip gate, 6 seeds): osi_post_frac must clear BOTH freeze (no-learning)
AND shuffle (orientation-destroyed input) lesion controls by +0.15. Secondary read:
RSA-to-host-Gabor + on_minus_off_mean (the common-mode discriminator).
Instrument check: the columnar inhibition MUST measurably suppress + sparsify V1
firing vs an inhibition-zeroed control (else the competition is inert and the test
is void), and freeze must genuinely differ from learn.

Run:
  SIM_BACKEND=cupy python -u -m research.runners._b1_v1_selforg_columnar_wta_derisk \
      --seeds 42 43 44 45 46 47 --dev-steps 24000 \
      --out research/findings/raw/_b1_v1_selforg_columnar_wta_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host, get_backend  # noqa: E402
from tools.lab import attributable_to  # noqa: E402  (explicit treatment-vs-control attribution)
# reuse-by-import: the proven on-bridge machinery + numpy metrics (apples-to-apples)
from research.runners._b1_v1_selforg_onbridge_derisk import (  # noqa: E402
    build_isotropic_support,
    render_oriented_field,
    read_v1_rfs,
    raw_weight_stats,
    encode_v1_firing,
    _drive_image,
    _freeze,
)
from research.runners._b1_v1_selforg_rf_derisk import (  # noqa: E402
    build_shape_set,
    build_fine_orientation_set,
    gabor_orientation_tuning,
    build_host_v1_matrix,
    encode_host_v1,
    within_between_margin,
    rsa_between_codes,
    orientation_decode_accuracy,
)
from sim.visual_cortex import (  # noqa: E402
    N_ORIENTATIONS,
    N_FREQUENCIES,
    V1_POSITIONS_PER_DIM,
    RETINA_SIZE,
)


# ============================================================================
# Columnar (iso-position) k-WTA inhibitory support.
# ============================================================================

def build_columnar_inh_support(n_orient, n_freq, n_pos, v0, h0):
    """Iso-position WTA wiring. One FS interneuron per retinotopic position pools
    ALL (orient x freq) channels at that position and inhibits the whole column.

    V1 cell index layout (matches build_isotropic_support):
        c = orient_i*(n_freq*n_pos*n_pos) + freq_i*(n_pos*n_pos) + base_pos
    so base_pos = c % (n_pos*n_pos) in [0, n_pos*n_pos). The interneuron for a
    column is inh[base_pos]. Returns (exc_pre, exc_post) [V1->inh] and
    (inh_pre, inh_post) [inh->V1] as GLOBAL indices.
    """
    n_v1 = n_orient * n_freq * n_pos * n_pos
    n_col = n_pos * n_pos
    exc_pre, exc_post, inh_pre, inh_post = [], [], [], []
    for c in range(n_v1):
        col = c % n_col
        # V1 cell excites its column's interneuron
        exc_pre.append(v0 + c)
        exc_post.append(h0 + col)
        # its column's interneuron inhibits the V1 cell
        inh_pre.append(h0 + col)
        inh_post.append(v0 + c)
    return (np.asarray(exc_pre, dtype=np.int64), np.asarray(exc_post, dtype=np.int64),
            np.asarray(inh_pre, dtype=np.int64), np.asarray(inh_post, dtype=np.int64))


def build_v1_bridge_columnar(seed, n_orient, n_freq, n_pos, retina_size, radius,
                             init_weight_mean, init_weight_jitter,
                             hebb_lr, hebb_decay, hebb_max, coact_decay, coact_thresh,
                             homeo_target, homeo_ema_alpha, homeo_adapt_rate,
                             syn_scaling, syn_scaling_rate,
                             inh_exc_w, inh_inh_w, inh_zero=False):
    """Minimal on-bridge V1 (retina + cortex_v1_simple) + a COLUMNAR FS inhibitory
    pool (one interneuron per position). `inh_zero=True` builds the SAME topology
    but with the inh->V1 weights set to 0 — the no-competition control that proves
    the columnar inhibition is what changes the firing/RF (instrument check)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType

    n_retina = 2 * retina_size * retina_size
    n_v1 = n_orient * n_freq * n_pos * n_pos
    n_col = n_pos * n_pos

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.enable_inhibitory_neurons = True   # explicit: the E/I-split current path must run for FS inhibition
    regions = [
        BrainRegion(name="retina", n_neurons=n_retina, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="cortex_v1_simple", n_neurons=n_v1, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="v1_inh", n_neurons=n_col, exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name),
    ]
    # FF plastic pathway declared at ~0 density; isotropic support installed below.
    # The two inhibitory pathways are declared at density 0 (registers the pathway,
    # NO random edges) — the COLUMNAR edges are installed via set_pathway_weights,
    # and the inhibitory SIGN comes from the v1_inh region trait (not the pathway),
    # so a set_pathway_weights-installed inh->V1 edge inhibits correctly.
    pathways = [
        RegionPathway(from_region="retina", to_region="cortex_v1_simple",
                      density=0.001, weight_mean=init_weight_mean, weight_jitter=init_weight_jitter,
                      plastic=True),
        RegionPathway(from_region="cortex_v1_simple", to_region="v1_inh",
                      density=0.0, weight_mean=inh_exc_w, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="v1_inh", to_region="cortex_v1_simple",
                      density=0.0, weight_mean=inh_inh_w, weight_jitter=0.0, plastic=False),
    ]
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    cfg.enable_structural_plasticity = False

    # Feedforward rate-Hebbian (BCM-like co-activity), the same rule as the wall runner.
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_rate_window = True
    cfg.hebbian_coactivity_decay = coact_decay
    cfg.hebbian_coactivity_thresh = coact_thresh
    cfg.hebbian_learning_rate = hebb_lr
    cfg.hebbian_weight_decay = hebb_decay
    cfg.hebbian_max_weight = hebb_max
    cfg.hebbian_mean_subtract = 0.0
    cfg.hebbian_oja = 0.0
    cfg.hebbian_min_weight = 0.0
    # Per-cell homeostasis: threshold adaptation + Turrigiano synaptic scaling (the companion).
    cfg.enable_homeostasis = True
    cfg.homeostasis_target_rate = homeo_target
    cfg.homeostasis_ema_alpha = homeo_ema_alpha
    cfg.homeostasis_threshold_adapt_rate = homeo_adapt_rate
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
    h0 = int(rm.indices("v1_inh")[0])

    # FF isotropic support + random init weights (any orientation must be LEARNED).
    pre_rel, post_rel = build_isotropic_support(n_orient, n_freq, n_pos, retina_size, radius)
    rng = np.random.default_rng(seed * 7 + 1)
    w_init = np.abs(rng.normal(init_weight_mean, init_weight_jitter, size=pre_rel.shape)).astype(np.float32)
    w_init = np.clip(w_init, 0.0, hebb_max)
    bridge.set_pathway_weights(pathway_name="retina_to_v1_selforg",
                               pre_indices=(pre_rel + r0), post_indices=(post_rel + v0),
                               weights=w_init, add_missing=True)

    # Columnar WTA inhibition.
    exc_pre, exc_post, inh_pre, inh_post = build_columnar_inh_support(n_orient, n_freq, n_pos, v0, h0)
    bridge.set_pathway_weights(pathway_name="v1_to_v1inh_columnar",
                               pre_indices=exc_pre, post_indices=exc_post,
                               weights=np.full(exc_pre.shape, inh_exc_w, dtype=np.float32),
                               add_missing=True)
    inh_w = 0.0 if inh_zero else inh_inh_w
    bridge.set_pathway_weights(pathway_name="v1inh_to_v1_columnar",
                               pre_indices=inh_pre, post_indices=inh_post,
                               weights=np.full(inh_pre.shape, inh_w, dtype=np.float32),
                               add_missing=True)
    return bridge, r0, v0, h0, n_retina, n_v1, n_col


def develop_col(bridge, r0, n_retina, n_steps, drive_pA, present_steps, seed, xp, retina_size, shuffle=False):
    rng = np.random.default_rng(seed * 101 + (7 if shuffle else 3))
    steps_done = 0
    while steps_done < n_steps:
        img = render_oriented_field(rng, retina_size=retina_size, shuffle=shuffle)
        _drive_image(bridge, r0, n_retina, img, drive_pA, xp)
        for _ in range(present_steps):
            bridge._run_one_simulation_step()
            steps_done += 1
            if steps_done >= n_steps:
                break
    bridge.cp_external_input_current[:] = 0.0


def measure_v1_sparsity(bridge, r0, v0, n_retina, n_v1, drive_pA, read_steps, settle_steps, xp, seed, retina_size):
    """Instrument: mean V1 firing rate + mean per-stimulus active fraction over a
    few oriented gratings. Used to prove the columnar inhibition suppresses/sparsifies."""
    rng = np.random.default_rng(seed * 991 + 5)
    rates, active_fracs = [], []
    for _ in range(6):
        img = render_oriented_field(rng, retina_size=retina_size, shuffle=False)
        _drive_image(bridge, r0, n_retina, img, drive_pA, xp)
        for _ in range(settle_steps):
            bridge._run_one_simulation_step()
        counts = np.zeros(n_v1, dtype=np.float32)
        for _ in range(read_steps):
            bridge._run_one_simulation_step()
            fired = to_host(bridge.cp_firing_states[v0:v0 + n_v1])
            counts += np.asarray(fired, dtype=np.float32)
        rates.append(float(counts.sum() / max(1, read_steps) / n_v1))
        active_fracs.append(float((counts > 0).mean()))
    bridge.cp_external_input_current[:] = 0.0
    return float(np.mean(rates)), float(np.mean(active_fracs))


def run_seed(seed, a, do_instrument=False):
    xp, backend = get_backend()
    xp = xp if backend == "cupy" else None
    n_orient, n_freq, n_pos = a.n_orient, a.n_freq, a.n_pos
    retina_size, radius = a.retina_size, a.radius
    n_v1 = n_orient * n_freq * n_pos * n_pos
    t0 = time.time()

    def build(inh_zero=False):
        return build_v1_bridge_columnar(
            seed, n_orient, n_freq, n_pos, retina_size, radius,
            a.init_weight_mean, a.init_weight_jitter,
            a.hebb_lr, a.hebb_decay, a.hebb_max, a.coact_decay, a.coact_thresh,
            a.homeo_target, a.homeo_ema_alpha, a.homeo_adapt_rate,
            bool(a.syn_scaling), a.syn_scaling_rate, a.inh_exc_w, a.inh_inh_w,
            inh_zero=inh_zero)

    # ---- LEARN arm (columnar WTA) ----
    bridge, r0, v0, h0, n_retina, _, n_col = build()

    # INSTRUMENT (seed-representative only): columnar inhibition must suppress + sparsify V1.
    instrument = {}
    if do_instrument:
        rate_on, act_on = measure_v1_sparsity(bridge, r0, v0, n_retina, n_v1,
                                              a.drive_pA, a.read_steps, a.settle_steps, xp, seed, retina_size)
        bctrl, r0c, v0c, _, n_retc, _, _ = build(inh_zero=True)
        rate_off, act_off = measure_v1_sparsity(bctrl, r0c, v0c, n_retc, n_v1,
                                               a.drive_pA, a.read_steps, a.settle_steps, xp, seed, retina_size)
        instrument = dict(
            v1_rate_inh_on=round(rate_on, 5), v1_rate_inh_off=round(rate_off, 5),
            v1_active_frac_inh_on=round(act_on, 4), v1_active_frac_inh_off=round(act_off, 4),
            inhibition_suppresses=bool(rate_on < rate_off * 0.95),
        )
        del bctrl

    rf_pre = read_v1_rfs(bridge, r0, v0, n_retina, n_v1, n_orient, n_freq, n_pos, retina_size, radius)
    osi_pre_mean, osi_pre_frac = gabor_orientation_tuning(rf_pre)

    develop_col(bridge, r0, n_retina, a.dev_steps, a.drive_pA, a.present_steps, seed, xp, retina_size, shuffle=False)
    _freeze(bridge)
    rf_post = read_v1_rfs(bridge, r0, v0, n_retina, n_v1, n_orient, n_freq, n_pos, retina_size, radius)
    osi_post_mean, osi_post_frac = gabor_orientation_tuning(rf_post)
    raw = raw_weight_stats(bridge, r0, v0, n_retina, n_v1, retina_size)

    # ---- SHUFFLE lesion (orientation-destroyed input) ----
    bridge_sh, r0s, v0s, _, n_ret_s, _, _ = build()
    develop_col(bridge_sh, r0s, n_ret_s, a.dev_steps, a.drive_pA, a.present_steps, seed, xp, retina_size, shuffle=True)
    _freeze(bridge_sh)
    rf_shuf = read_v1_rfs(bridge_sh, r0s, v0s, n_ret_s, n_v1, n_orient, n_freq, n_pos, retina_size, radius)
    osi_shuf_mean, osi_shuf_frac = gabor_orientation_tuning(rf_shuf)
    del bridge_sh

    # ---- FREEZE lesion (no learning; random init frozen) -> equals osi_pre by construction,
    #      but recomputed on a freshly built+frozen bridge as the honest control ----
    bridge_fz, r0f, v0f, _, n_ret_f, _, _ = build()
    _freeze(bridge_fz)  # freeze BEFORE development -> no plasticity runs
    develop_col(bridge_fz, r0f, n_ret_f, a.dev_steps, a.drive_pA, a.present_steps, seed, xp, retina_size, shuffle=False)
    rf_frz = read_v1_rfs(bridge_fz, r0f, v0f, n_ret_f, n_v1, n_orient, n_freq, n_pos, retina_size, radius)
    osi_frz_mean, osi_frz_frac = gabor_orientation_tuning(rf_frz)
    del bridge_fz

    # ---- Geometry: V1 FIRING code RSA-to-host on the Option-B shapes ----
    # build_host_v1_matrix() is hard-wired to the production arch (8x4x16x16, retina 32);
    # skip the host-reference geometry at any reduced arch (smoke), keep OSI as the primary read.
    is_prod_arch = (n_orient == N_ORIENTATIONS and n_freq == N_FREQUENCIES
                    and n_pos == V1_POSITIONS_PER_DIM and retina_size == RETINA_SIZE)
    if is_prod_arch:
        rng = np.random.default_rng(seed)
        images, labels, _ = build_shape_set(a.n_categories, a.n_exemplars, rng, image_size=retina_size)
        Whost = build_host_v1_matrix()
        host_code = encode_host_v1(images, Whost)
        host_w, host_b, host_m = within_between_margin(host_code, labels)
        v1_code = encode_v1_firing(bridge, r0, v0, n_retina, n_v1, images,
                                   a.drive_pA, a.read_steps, a.settle_steps, xp)
        w_p, b_p, m_p = within_between_margin(v1_code, labels)
        rsa_host = rsa_between_codes(v1_code, host_code)
        oimgs, olabs = build_fine_orientation_set(a.n_orient_dec, a.n_orient_ex, seed + 100)
        v1_ocode = encode_v1_firing(bridge, r0, v0, n_retina, n_v1, oimgs,
                                    a.drive_pA, a.read_steps, a.settle_steps, xp)
        decode = orientation_decode_accuracy(v1_ocode, olabs)
        v1_rate = float(v1_code.sum() / max(1, images.shape[0]) / max(1, a.read_steps) / n_v1)
    else:
        w_p = b_p = m_p = rsa_host = decode = host_m = float("nan")
        v1_rate = float("nan")
    del bridge

    # Explicit treatment-vs-control ATTRIBUTION: what fraction of the orientation effect is
    # present ABOVE the no-learning freeze control? (For a NO-GO this is the honest null read:
    # both arms ~0 => UNDEFINED, there is no oriented effect to attribute to the competition.)
    attributable_to("osi_learn_vs_freeze(seed%d)" % seed, osi_post_frac, osi_frz_frac)

    # ---- verdict (the flip gate: learn clears BOTH lesions by +0.15) ----
    lift_over_freeze = osi_post_frac >= osi_frz_frac + 0.15
    lift_over_shuffle = osi_post_frac >= osi_shuf_frac + 0.15
    learned_oriented = osi_post_frac >= 0.5
    if learned_oriented and lift_over_freeze and lift_over_shuffle:
        verdict = "GO"
    elif lift_over_freeze and lift_over_shuffle and osi_post_frac >= 0.3:
        verdict = "PARTIAL"
    else:
        verdict = "BOUNDARY"

    elapsed = round(time.time() - t0, 1)
    out = dict(
        seed=seed, backend=backend, n_v1=n_v1, n_col=n_col, elapsed_s=elapsed,
        v1_firing_rate=round(v1_rate, 4),
        osi=dict(
            pre_random=round(osi_pre_frac, 4),
            post_learned=round(osi_post_frac, 4),
            freeze_ctrl=round(osi_frz_frac, 4),
            shuffle_ctrl=round(osi_shuf_frac, 4),
            post_mean=round(osi_post_mean, 4),
        ),
        on_minus_off_mean=raw["on_minus_off_mean"],
        l2_mean=raw["l2_mean"], frac_cells_l2_near_zero=raw["frac_cells_l2_near_zero"],
        geometry=dict(within=round(w_p, 4), between=round(b_p, 4), margin=round(m_p, 4),
                      rsa_vs_host=round(rsa_host, 4), orient_decode=round(decode, 4),
                      host_margin=round(host_m, 4)),
        lift_over_freeze=bool(lift_over_freeze), lift_over_shuffle=bool(lift_over_shuffle),
        verdict=verdict,
    )
    if instrument:
        out["instrument"] = instrument
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46, 47])
    ap.add_argument("--n-orient", type=int, default=N_ORIENTATIONS)
    ap.add_argument("--n-freq", type=int, default=N_FREQUENCIES)
    ap.add_argument("--n-pos", type=int, default=V1_POSITIONS_PER_DIM)
    ap.add_argument("--retina-size", type=int, default=RETINA_SIZE)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--dev-steps", type=int, default=24000)
    ap.add_argument("--present-steps", type=int, default=40)
    ap.add_argument("--drive-pA", type=float, default=1200.0)
    ap.add_argument("--settle-steps", type=int, default=25)
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
    ap.add_argument("--inh-exc-w", type=float, default=25.0,
                    help="V1->interneuron weight (drives the columnar interneuron)")
    ap.add_argument("--inh-inh-w", type=float, default=45.0,
                    help="interneuron->V1 weight (columnar WTA suppression strength)")
    ap.add_argument("--homeo-target", type=float, default=0.012)
    ap.add_argument("--homeo-ema-alpha", type=float, default=0.01)
    ap.add_argument("--homeo-adapt-rate", type=float, default=0.004)
    ap.add_argument("--n-categories", type=int, default=4)
    ap.add_argument("--n-exemplars", type=int, default=4)
    ap.add_argument("--n-orient-dec", type=int, default=8)
    ap.add_argument("--n-orient-ex", type=int, default=8)
    ap.add_argument("--instrument-seed", type=int, default=42,
                    help="seed to also run the inhibition-efficacy instrument on")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_b1_v1_selforg_columnar_wta_6seed.json")
    a = ap.parse_args()

    os.environ.setdefault("SIM_BACKEND", "cupy")
    print(f"[B1 columnar-WTA V1 self-org] seeds={a.seeds} dev_steps={a.dev_steps} "
          f"arch={a.n_orient}x{a.n_freq}x{a.n_pos}x{a.n_pos} radius={a.radius} "
          f"inh_exc_w={a.inh_exc_w} inh_inh_w={a.inh_inh_w}", flush=True)

    per_seed = []
    for s in a.seeds:
        r = run_seed(s, a, do_instrument=(s == a.instrument_seed))
        per_seed.append(r)
        line = (f"  seed={s}: osi_post={r['osi']['post_learned']} "
                f"(pre={r['osi']['pre_random']} freeze={r['osi']['freeze_ctrl']} "
                f"shuf={r['osi']['shuffle_ctrl']}) on-off={r['on_minus_off_mean']} "
                f"rsa_host={r['geometry']['rsa_vs_host']} -> {r['verdict']} ({r['elapsed_s']}s)")
        print(line, flush=True)
        if "instrument" in r:
            ins = r["instrument"]
            print(f"    INSTRUMENT: v1_rate inh_on={ins['v1_rate_inh_on']} "
                  f"inh_off={ins['v1_rate_inh_off']} | active_frac on={ins['v1_active_frac_inh_on']} "
                  f"off={ins['v1_active_frac_inh_off']} | suppresses={ins['inhibition_suppresses']}", flush=True)

    def col(f):
        return [f(r) for r in per_seed]

    verdicts = [r["verdict"] for r in per_seed]
    all_go = all(v == "GO" for v in verdicts)
    overall = "GO" if all_go else ("PARTIAL" if all(v in ("GO", "PARTIAL") for v in verdicts) else "BOUNDARY")

    ins_seed = next((r for r in per_seed if "instrument" in r), None)
    summary = dict(
        overall_verdict=overall,
        flip_decision=("FLIP-ON" if all_go else "HOLD-OFF"),
        seeds=a.seeds,
        per_seed_verdicts=verdicts,
        osi_pre_frac_mean=round(float(np.mean(col(lambda r: r["osi"]["pre_random"]))), 4),
        osi_post_frac_mean=round(float(np.mean(col(lambda r: r["osi"]["post_learned"]))), 4),
        osi_post_frac_min=round(float(np.min(col(lambda r: r["osi"]["post_learned"]))), 4),
        osi_freeze_frac_mean=round(float(np.mean(col(lambda r: r["osi"]["freeze_ctrl"]))), 4),
        osi_shuffle_frac_mean=round(float(np.mean(col(lambda r: r["osi"]["shuffle_ctrl"]))), 4),
        on_minus_off_mean=round(float(np.mean(col(lambda r: r["on_minus_off_mean"]))), 6),
        rsa_vs_host_mean=round(float(np.mean(col(lambda r: r["geometry"]["rsa_vs_host"]))), 4),
        orient_decode_mean=round(float(np.mean(col(lambda r: r["geometry"]["orient_decode"]))), 4),
        v1_firing_rate_mean=round(float(np.mean(col(lambda r: r["v1_firing_rate"]))), 4),
        gate="all seeds osi_post_frac>=0.5 AND >=freeze+0.15 AND >=shuffle+0.15",
        instrument=(ins_seed["instrument"] if ins_seed else None),
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
