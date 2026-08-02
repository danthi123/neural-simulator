"""AWS job (b) — VALIDATE-OR-RETIRE the orphaned learned V2/IT ventral hierarchy ON-BRIDGE.

CONTEXT (the scoping this implements):
  research/findings/2026-07-23-perception-closure-scoping.md  -> residual #3.
  The DEPLOYED nav brain wires retina -> V1_simple(fixed Gabor) -> V1_complex(phase-pool) ->
  cortex_v2 (STDP, plastic) -> cortex_it (STDP, plastic) -> ventral "what" readouts. But EVERY
  validated perception-grounded result (genfrontier Option-B, EMERGE-34/36/53, the fully-spiking
  codon) runs V1(Gabor)->V1_complex as a RATE reference front end and then a SEPARATE competitive
  pooler / Marr-Albus codon for category discovery. NONE of them uses the deployed cortex_v2->cortex_it
  STDP regions. So the deployed learned ventral hierarchy is ORPHANED + UNPROVEN -- it may be INERT.

THE QUESTION (validate vs retire, cleanly distinguished):
  Perceive a set of objects (categories = shared visual features) through the REAL deployed
  retina -> V1(Gabor) -> V1_complex -> V2 -> IT hierarchy on a real SimulationBridge, and test whether
  the STDP-learned V2/IT layer develops useful CATEGORY-DISCRIMINATIVE + POSITION-INVARIANT codes (the
  DiCarlo property that V2/IT should ADD over V1's position-specific/retinotopic Gabors) -- or is inert.

  VALIDATE (GO): trained IT firing codes categorize an object at a HELD-OUT retinal position (position
    invariance) BETTER than the retinotopic V1_complex code AND better than a frozen (no-learning) IT,
    with the anti-cheats collapsing.
  RETIRE (NEGATIVE): trained IT is inert / adds nothing over V1_complex / adds nothing over frozen IT.
    This is a FIRST-CLASS honest result -- the scoping predicted it, and it standardizes the grounding
    path onto the validated V1->pooler codon.

THE DE-RISK DESIGN (categories = orientation; position DECOUPLED from category so invariance is testable):
  * A category = a base orientation theta_c. A position = a retinal centre (cx, cy). Rendering
    render_bar(cx, cy, theta_c, ...) DECOUPLES identity (orientation) from location -- unlike the stock
    build_shape_set which conflates category with position. TRAIN positions vs HELD-OUT positions are
    DISJOINT, so the position-invariance test is a genuine hold-out (V2/IT never saw the held position).
  * The bridge REPLICATES the deployed visual regions + pathways EXACTLY (params mirror
    g11_bg_runner.py:2704-2848); the front-end Gabor bank is installed by the DEPLOYED
    sim.visual_cortex.apply_v1_gabor_weights (reuse-by-import). V1 plasticity gate is CLOSED so the
    Gabor is frozen; V2/IT learn via STDP -- exactly the deployed intent ("V1 fixed Gabor, V2/IT learn").

ARMS (each measured; a control that is never invoked is the bug):
  * TRAINED     : enable_stdp=True, V1 gate closed -> the deployed learned V2/IT.
  * NO-LEARNING : enable_stdp=False (frozen random-init V2/IT) -- the validate-vs-retire discriminator
                  (does TRAINING add anything?).
  * LESION      : on the trained bridge, shut the V1_complex->V2 TRANSMISSION gate at read time ->
                  V2/IT get no feedforward stream -> IT code must collapse (proves IT's structure is
                  driven by the visual stream, load-bearing -- not spontaneous internal recurrence).
                  (v2_rate_intact vs v2_rate_lesion records the mechanism biting even if IT is inert.)

ANTI-CHEAT KIT (reused from the EMERGE / Option-B perception work):
  * POSITION-INVARIANCE (the DiCarlo "IT untangles" property, the discriminating test): classify a
    category at a HELD-OUT position from TRAIN-position centroids. IT (if it learned) >> V1_complex
    (retinotopic, a new position looks different).
  * PER-IMAGE PIXEL SCRAMBLE (input-destruction, LOAD-BEARING): shuffle each image's pixels ->
    destroys within-category VISUAL similarity -> IT category structure collapses. Structure is
    visual, not injected.
  * RSA PIXEL-PROVENANCE (LABEL-FREE): correlate the off-diagonal of the IT firing-cosine matrix with
    the raw-pixel cosine matrix. Intact tracks pixels; scramble collapses.
  * NO-LEARNING control: frozen V2/IT -> no added invariance over V1 (the flat baseline).
  * LESION control: silence the V2/IT input -> collapse.
  * 6 seeds (42,43,44,100,101,102).

READ-OUT (the honest "firing code"): per-neuron SPIKE COUNTS over a read window (the primary code).
  A graded membrane-DEPOLARIZATION code is also computed as context (the documented rate-code-wall
  fall-back if a point-neuron region does not spike -- see CLAUDE.md / the convergence de-risk). If IT
  never spikes the spike code is degenerate -> decode = chance -> the INERT/RETIRE signal, honestly.

Reuse-by-import (NO sim/ edit): sim.visual_cortex (apply_v1_gabor_weights, image_to_retina_drive) +
  sim.bridge/config/regions/enums (the region framework). The ONLY wiring addition vs the deployed
  pathways is a transmission_gate="v2it_stream" tag on cortex_v1_complex->cortex_v2 (default OPEN=1.0,
  i.e. byte-identical to the deployed always-on pathway) purely to enable the clean lesion.

Run (CPU smoke, small, 1 seed):
  SIM_BACKEND=numpy python -u -m research.runners._perception_v2it_validate_or_retire_derisk \
      --seeds 42 --n-categories 3 --n-train-pos 2 --n-held-pos 1 --n-ex 2 \
      --train-epochs 2 --scene-steps 6 --read-steps 15 --settle-steps 6 \
      --out research/findings/raw/_perception_v2it_smoke.json

Run (GPU 6-seed real):
  SIM_BACKEND=cupy python -u -m research.runners._perception_v2it_validate_or_retire_derisk \
      --seeds 42 43 44 100 101 102 \
      --out research/findings/raw/_perception_v2it_validate_or_retire.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# reuse-by-import: the project's REAL deployed front end + backend helpers (NO sim/ edit)
from sim.visual_cortex import apply_v1_gabor_weights, image_to_retina_drive  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402
# reuse-by-import: the Földiák (1991) trace / temporal-continuity hyper-parameters (bout_len, trace_decay)
# + the mechanism the --trace-rule mode ports here (EMERGE-50). NO sim/ edit; the mode is default-OFF.
from research.runners._emerge50_trace_rule_derisk import (  # noqa: E402
    BOUT_LEN as _TRACE_BOUT_LEN, TRACE_DECAY as _TRACE_DECAY,
)

REGIONS_READ = ["cortex_v1_complex", "cortex_v2", "cortex_it"]
_V_REST = -65.0   # Izhikevich resting potential; depolarization above this = graded response


# ============================================================================
# 1. Object rendering -- identity (orientation) DECOUPLED from position (retinal centre).
#    (mirrors _b1_v1_selforg_rf_derisk / Option-B _render_bar_image; self-contained.)
# ============================================================================
def render_bar(cx, cy, theta, length, thickness, rng, image_size, pixel_noise=0.04):
    """One oriented bar -> (2, H, W) ON/OFF image (channel-first, the retina convention)."""
    H = W = image_size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    perp = np.abs(dx * math.sin(theta) - dy * math.cos(theta))
    along = dx * math.cos(theta) + dy * math.sin(theta)
    bar = np.exp(-(perp * perp) / (2.0 * thickness * thickness))
    bar = bar * (np.abs(along) <= (length / 2.0)).astype(np.float32)
    on = bar.astype(np.float32)
    gx = np.gradient(on, axis=1)
    gy = np.gradient(on, axis=0)
    off = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    off = off / (off.max() + 1e-6) * 0.3
    on = np.clip(on + rng.normal(0.0, pixel_noise, size=on.shape).astype(np.float32), 0.0, 1.0)
    off = np.clip(off + rng.normal(0.0, pixel_noise * 0.5, size=off.shape).astype(np.float32), 0.0, 1.0)
    return np.stack([on, off], axis=0)


def _position_grid(n_positions, image_size, bar_len):
    """Deterministic spread of retinal centres, kept a bar-length's margin from the edges so bars fit.
    Returns a list of (cx, cy). The FIRST n_train_pos are train positions; the rest held-out."""
    margin = bar_len * 0.5 + 2.0
    lo, hi = margin, image_size - margin
    # spread on a near-square grid, then take the first n_positions in row-major order
    side = int(math.ceil(math.sqrt(n_positions)))
    xs = np.linspace(lo, hi, side)
    ys = np.linspace(lo, hi, side)
    pts = [(float(x), float(y)) for y in ys for x in xs]
    return pts[:n_positions]


def build_object_set(categories_theta, positions, n_ex, image_size, bar_len_frac, seed):
    """categories x positions x exemplars oriented bars. category = orientation index; position =
    retinal-centre index. Returns (images (N,2,H,W), cat_labels (N,), pos_labels (N,))."""
    rng = np.random.default_rng(seed)
    base_len = image_size * bar_len_frac
    images, cat_labels, pos_labels = [], [], []
    for ci, theta0 in enumerate(categories_theta):
        for pi, (cx0, cy0) in enumerate(positions):
            for _ in range(n_ex):
                theta = theta0 + rng.normal(0.0, math.radians(6.0))
                cx = cx0 + rng.normal(0.0, image_size * 0.02)
                cy = cy0 + rng.normal(0.0, image_size * 0.02)
                length = base_len * (1.0 + rng.normal(0.0, 0.06))
                thick = 1.6 * (1.0 + rng.normal(0.0, 0.08))
                images.append(render_bar(cx, cy, theta, length, thick, rng, image_size))
                cat_labels.append(ci)
                pos_labels.append(pi)
    return (np.asarray(images, dtype=np.float32),
            np.asarray(cat_labels, dtype=np.int64),
            np.asarray(pos_labels, dtype=np.int64))


def scramble_images(images, seed):
    """PER-IMAGE pixel scramble: shuffle each image's spatial layout with its OWN permutation (same
    permutation applied to both ON/OFF channels so channel alignment is preserved but SHAPE is
    destroyed). Destroys within-category VISUAL similarity while preserving intensity statistics."""
    rng = np.random.default_rng(seed)
    C, H, W = images.shape[1], images.shape[2], images.shape[3]
    out = np.empty_like(images)
    for i in range(images.shape[0]):
        perm = rng.permutation(H * W)
        out[i] = images[i].reshape(C, H * W)[:, perm].reshape(C, H, W)
    return out


# ============================================================================
# 2. Similarity / decoding metrics (Option-B definitions; self-contained).
# ============================================================================
def _cos_matrix(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-9, 1.0, n)
    Xn = X / n
    return Xn @ Xn.T


def within_between_margin(codes, labels):
    C = _cos_matrix(codes)
    N = codes.shape[0]
    same = labels[:, None] == labels[None, :]
    eye = np.eye(N, dtype=bool)
    within = C[same & ~eye]
    between = C[~same]
    w = float(within.mean()) if within.size else 0.0
    b = float(between.mean()) if between.size else 0.0
    return w, b, w - b


def rsa(codesA, codesB):
    """Off-diagonal RSA (label-free): correlate two code-sets' pairwise-cosine geometries."""
    Ca, Cb = _cos_matrix(codesA), _cos_matrix(codesB)
    iu = np.triu_indices(Ca.shape[0], k=1)
    a, b = Ca[iu], Cb[iu]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def rsa_pixels(images, codes):
    return rsa(codes, images.reshape(images.shape[0], -1).astype(np.float32))


def centroid_decode(train_codes, train_labels, test_codes, test_labels):
    """Nearest class-centroid (cosine) decode: build a centroid per class from TRAIN codes, classify
    each TEST code. Used for POSITION-INVARIANCE (train = train positions, test = held positions)."""
    def _norm(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(n < 1e-9, 1.0, n)
    Xtr, Xte = _norm(train_codes), _norm(test_codes)
    classes = np.unique(train_labels)
    cents = {}
    for c in classes:
        m = Xtr[train_labels == c]
        v = m.mean(axis=0)
        nv = np.linalg.norm(v)
        cents[c] = v / nv if nv > 1e-9 else v
    if not classes.size:
        return 0.0
    correct = 0
    for i in range(Xte.shape[0]):
        sims = [(float(Xte[i] @ cents[c]), c) for c in classes]
        pred = max(sims)[1]
        correct += int(pred == test_labels[i])
    return correct / Xte.shape[0]


# ============================================================================
# 3. The bridge -- REPLICATES the deployed visual hierarchy (g11_bg_runner.py:2704-2848).
# ============================================================================
def build_visual_bridge(seed, a, learn):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType
    RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name

    IMG = a.image_size
    n_retina = 2 * IMG * IMG
    n_v1s = a.n_orientations * a.n_frequencies * a.n_pos * a.n_pos
    n_v1c = a.n_orientations * a.n_pos * a.n_pos
    n_v2 = a.n_v2
    n_it = a.n_it

    # --- regions (exact deployed params) ---
    regions = [
        BrainRegion(name="retina", n_neurons=n_retina, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS),
        BrainRegion(name="cortex_v1_simple", n_neurons=n_v1s, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS),
        BrainRegion(name="cortex_v1_complex", n_neurons=n_v1c, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS),
        BrainRegion(name="cortex_v2", n_neurons=n_v2, exc_fraction=0.8, internal_density=0.05,
                    exc_weight_mean=2.0, inh_weight_mean=4.0, weight_jitter=0.2,
                    plastic_internal=True, izh_neuron_type=RS),
        BrainRegion(name="cortex_it", n_neurons=n_it, exc_fraction=0.8, internal_density=0.10,
                    exc_weight_mean=2.0, inh_weight_mean=4.0, weight_jitter=0.2,
                    plastic_internal=True, izh_neuron_type=RS),
    ]
    # --- pathways. DEFAULTS mirror the deployed params (g11_bg_runner.py:2817-2848) EXACTLY. The
    # weight/density KNOBS (default = deployed) let a `--fair-drive` run boost the stream so it
    # actually reaches IT -- because at the DEPLOYED operating point the hierarchy is INERT (V1_simple
    # barely depolarizes, V1_complex/V2/IT silent: the 2/1024-density random pool + weak weights don't
    # propagate spikes). Testing "does V2/IT ADD invariance" REQUIRES the stream to fire, else "IT is
    # dead" is an under-driving artifact, not a property of the learned layer. Both operating points are
    # recorded (the deployed-point propagation profile + the fair-point verdict).
    # +transmission_gate on v1c->v2 for the clean lesion (default OPEN(1.0) == deployed always-on).
    pathways = [
        RegionPathway(from_region="retina", to_region="cortex_v1_simple",
                      density=0.05, weight_mean=0.5, weight_jitter=0.5,
                      plastic=True, plasticity_gate="visual_cortex_v1"),
        RegionPathway(from_region="cortex_v1_simple", to_region="cortex_v1_complex",
                      density=a.n_frequencies / float(n_v1s) * float(a.v1c_pool_density_mult),
                      weight_mean=float(a.v1c_pool_weight), weight_jitter=0.0,
                      plastic=False),
        RegionPathway(from_region="cortex_v1_complex", to_region="cortex_v2",
                      density=0.10, weight_mean=float(a.v1c_to_v2_weight), weight_jitter=0.5,
                      plastic=True, plasticity_gate="visual_cortex_v2",
                      transmission_gate="v2it_stream"),
        RegionPathway(from_region="cortex_v2", to_region="cortex_it",
                      density=0.20, weight_mean=float(a.v2_to_it_weight), weight_jitter=0.5,
                      plastic=True, plasticity_gate="visual_cortex_it"),
    ]

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt_ms = a.dt
    # SEED THE SUBSTRATE (CLAUDE.md gotcha: cfg.seed drives heterogeneity, NOT actual_seed_used).
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = bool(a.enable_ou)
    # V2/IT learn via STDP (the deployed rule); Hebbian OFF (matches every g* runner; avoids decay of
    # the frozen Gabor). No-learning arm: STDP OFF -> frozen random-init V2/IT.
    cfg.enable_stdp = bool(learn)
    cfg.enable_hebbian_learning = False
    cfg.stdp_w_max = float(a.stdp_w_max)   # raise above design weights so STDP can grow V2/IT (gotcha)
    cfg.enable_reward_modulation = False

    rt = RuntimeState()
    rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt,
                              gpu_config=GPUConfig())
    bridge._initialize_simulation_data()

    # install the DEPLOYED Gabor front end (reuse-by-import), then FREEZE V1 so the front end is fixed
    n_gabor = apply_v1_gabor_weights(
        bridge,
        n_orientations=a.n_orientations, n_frequencies=a.n_frequencies,
        n_positions_per_dim=a.n_pos, retina_size=IMG,
        receptive_field_radius=a.rf_radius, weight_scale=a.v1_weight_scale,
    )
    bridge.set_plasticity_gate("visual_cortex_v1", 0.0)   # Gabor frozen (deployed intent)

    idxmap = {name: np.asarray(bridge.region_manager.indices(name)) for name in
              ["retina", "cortex_v1_simple", "cortex_v1_complex", "cortex_v2", "cortex_it"]}
    return bridge, idxmap, n_gabor


# ============================================================================
# 4. Drive + read.
# ============================================================================
def _set_input(bridge, xp, N, retina_idx, drive_vec):
    full = np.zeros(N, np.float32)
    full[retina_idx] = drive_vec
    bridge.cp_external_input_current[:] = xp.asarray(full) if xp is not None else full


def _blank(bridge, xp, N, settle_steps):
    bridge.cp_external_input_current[:] = xp.asarray(np.zeros(N, np.float32)) if xp is not None \
        else np.zeros(N, np.float32)
    for _ in range(settle_steps):
        bridge._run_one_simulation_step()


def read_one(bridge, xp, idxmap, N, image, drive_pA, read_steps, settle_steps):
    """Blank+settle to clear carry-over, then drive the retina with `image` and accumulate per-neuron
    SPIKE COUNTS (the firing code) + graded DEPOLARIZATION over read_steps, per read region."""
    _blank(bridge, xp, N, settle_steps)
    drive_vec = image_to_retina_drive(image, drive_max_pA=float(drive_pA))  # (2*H*W,)
    _set_input(bridge, xp, N, idxmap["retina"], drive_vec)
    spike_acc = {r: np.zeros(idxmap[r].shape[0], np.float64) for r in REGIONS_READ}
    depol_acc = {r: np.zeros(idxmap[r].shape[0], np.float64) for r in REGIONS_READ}
    for _ in range(read_steps):
        bridge._run_one_simulation_step()
        fs = np.asarray(to_host(bridge.cp_firing_states)).astype(np.float64)
        v = np.asarray(to_host(bridge.cp_membrane_potential_v)).astype(np.float64)
        for r in REGIONS_READ:
            spike_acc[r] += fs[idxmap[r]]
            depol_acc[r] += (v[idxmap[r]] - _V_REST).clip(min=0.0)
    bridge.cp_external_input_current[:] = xp.asarray(np.zeros(N, np.float32)) if xp is not None \
        else np.zeros(N, np.float32)
    return spike_acc, depol_acc


def read_set(bridge, xp, idxmap, N, images, a):
    """Read every image in a set -> per-region (spike-code matrix, depol-code matrix)."""
    spikes = {r: [] for r in REGIONS_READ}
    depols = {r: [] for r in REGIONS_READ}
    for img in images:
        s, d = read_one(bridge, xp, idxmap, N, img, a.drive, a.read_steps, a.settle_steps)
        for r in REGIONS_READ:
            spikes[r].append(s[r])
            depols[r].append(d[r])
    return ({r: np.asarray(spikes[r]) for r in REGIONS_READ},
            {r: np.asarray(depols[r]) for r in REGIONS_READ})


def train_v2it(bridge, xp, idxmap, N, images, cat_labels, a):
    """Expose the hierarchy to the TRAIN objects with STDP on so V2/IT shape to the V1->V2->IT patterns.
    Records first-scene firing at each stage (diagnostic: where does the stream die?)."""
    diag = {r: 0 for r in ["retina", "cortex_v1_simple", "cortex_v1_complex", "cortex_v2", "cortex_it"]}
    n_img = images.shape[0]
    for ep in range(a.train_epochs):
        order = np.random.RandomState(a.seed_base * 7 + ep).permutation(n_img)
        for si, k in enumerate(order):
            _blank(bridge, xp, N, a.settle_steps)
            drive_vec = image_to_retina_drive(images[k], drive_max_pA=float(a.drive))
            _set_input(bridge, xp, N, idxmap["retina"], drive_vec)
            first = (ep == 0 and si == 0)
            for _ in range(a.scene_steps):
                bridge._run_one_simulation_step()
                if first:
                    fs = np.asarray(to_host(bridge.cp_firing_states)).astype(np.int64)
                    for r in diag:
                        diag[r] += int(fs[idxmap[r]].sum())
    bridge.cp_external_input_current[:] = xp.asarray(np.zeros(N, np.float32)) if xp is not None \
        else np.zeros(N, np.float32)
    return diag


# ============================================================================
# 4b. FÖLDIÁK (1991) TRACE / temporal-continuity training curriculum (--trace-rule, default OFF).
#   Ports the EMERGE-50 mechanism (research/runners/_emerge50_trace_rule_derisk.py) to this on-bridge
#   DiCarlo harness. Instead of blank-separated random scenes, each category's object is presented across
#   a SEQUENCE of NEARBY retinal positions in a contiguous BOUT, with NO blank between within-bout scenes,
#   so the SUBSTRATE's own STDP pre-synaptic trace (the biological eligibility trace) carries the
#   V1_complex pre-activity across positions; a host-side slow-decaying retinal-DRIVE trace additionally
#   smooths the pre-activity over the bout. STDP potentiation then binds V2/IT columns to the POSITION-
#   INVARIANT category identity (invariance learned from temporal continuity, the DiCarlo/Földiák claim).
#   The blank BETWEEN bouts resets the trace (bout boundary), exactly like EMERGE-50 resets each pass.
# ============================================================================
def _build_trace_bouts(images, cat_labels, pos_labels, a, epoch, shuffle_temporal):
    """Return a list of bouts (each a list of image indices). GROUPED (default) = each bout is one
    category swept across NEARBY positions (position-sorted -> a smooth spatial trajectory) so the
    decaying trace links the SAME category across positions. SHUFFLED-TEMPORAL (the LOAD-BEARING control)
    = the SAME bout structure (count, length, between-bout blanking, trace) but each bout is a RANDOM mix
    of categories/positions -> consecutive scenes are NOT same-category-nearby-position -> the trace can
    no longer bind a category across positions. The ONLY difference is within-bout ORDER/composition, so
    a trace benefit that survives here (but collapses under shuffle) proves temporal continuity did it."""
    rng = np.random.default_rng(a.seed_base * 131 + epoch)
    n_img = images.shape[0]
    n_bouts = max(1, int(math.ceil(n_img / max(1, a.bout_len))))
    if shuffle_temporal:
        allidx = rng.permutation(n_img)
        bouts, ptr = [], 0
        for _ in range(n_bouts):
            bouts.append([int(allidx[(ptr + j) % n_img]) for j in range(min(a.bout_len, n_img))])
            ptr += a.bout_len
        return bouts
    n_cat = a.n_categories
    by_cat = {c: np.where(cat_labels == c)[0] for c in range(n_cat)}
    bouts = []
    for _ in range(n_bouts):
        c = int(rng.integers(n_cat))
        idxs = by_cat[c]
        if idxs.size == 0:
            continue
        seq = idxs[np.argsort(pos_labels[idxs], kind="stable")]   # nearby-position trajectory for cat c
        start = int(rng.integers(len(seq)))
        bouts.append([int(seq[(start + j) % len(seq)]) for j in range(min(a.bout_len, len(seq)))])
    return bouts


def train_v2it_trace(bridge, xp, idxmap, N, images, cat_labels, pos_labels, a, shuffle_temporal=False):
    """Földiák temporal-continuity curriculum (see 4b). Records first-scene per-stage firing (diagnostic)."""
    diag = {r: 0 for r in ["retina", "cortex_v1_simple", "cortex_v1_complex", "cortex_v2", "cortex_it"]}
    n_retina = 2 * a.image_size * a.image_size
    first = True
    for ep in range(a.train_epochs):
        bouts = _build_trace_bouts(images, cat_labels, pos_labels, a, ep, shuffle_temporal)
        for bout in bouts:
            _blank(bridge, xp, N, a.settle_steps)                 # blank BETWEEN bouts -> reset the trace
            drive_trace = np.zeros(n_retina, np.float32)          # host eligibility trace over retinal drive
            for k in bout:
                cur = image_to_retina_drive(images[k], drive_max_pA=float(a.drive))
                drive_trace = np.clip(drive_trace * a.trace_decay + cur, 0.0, float(a.drive))
                _set_input(bridge, xp, N, idxmap["retina"], drive_trace)
                for _ in range(a.scene_steps):
                    bridge._run_one_simulation_step()             # NO within-bout blank -> substrate trace carries
                    if first:
                        fs = np.asarray(to_host(bridge.cp_firing_states)).astype(np.int64)
                        for r in diag:
                            diag[r] += int(fs[idxmap[r]].sum())
                first = False
    bridge.cp_external_input_current[:] = xp.asarray(np.zeros(N, np.float32)) if xp is not None \
        else np.zeros(N, np.float32)
    return diag


# ============================================================================
# 5. Metric suite for one code-set (spike codes), per region.
# ============================================================================
def region_metrics(train_codes, train_cat, held_codes, held_cat, held_images,
                   scr_codes, scr_cat, scr_images, n_categories):
    """All metrics for ONE region's code matrix (spikes OR graded depol): POSITION-INVARIANCE decode
    (train-position centroid -> held-out-position classify), same on scramble, RSA-to-pixels intact +
    scramble, category margin, mean code activity."""
    chance = 1.0 / n_categories
    heldpos_decode = centroid_decode(train_codes, train_cat, held_codes, held_cat)
    scramble_decode = centroid_decode(train_codes, train_cat, scr_codes, scr_cat)
    _, _, held_margin = within_between_margin(held_codes, held_cat)
    rsa_pix_held = rsa_pixels(held_images, held_codes)
    rsa_pix_scr = rsa_pixels(scr_images, scr_codes)
    return dict(
        chance=round(chance, 4),
        heldpos_decode=round(float(heldpos_decode), 4),
        scramble_decode=round(float(scramble_decode), 4),
        held_within_between_margin=round(float(held_margin), 4),
        rsa_pixels_held=round(float(rsa_pix_held), 4),
        rsa_pixels_scramble=round(float(rsa_pix_scr), 4),
        mean_code_activity_heldset=round(float(held_codes.mean()), 4),
    )


def stage_check(stg, v1, frz, lesion_decode, chance, a):
    """Does a LEARNED stage (V2 or IT) develop useful codes? (all conditions must hold for VALIDATE):
    position-invariance above chance, ADDS invariance over the retinotopic V1_complex AND over a frozen
    (no-learning) counterpart, per-image scramble collapses it, RSA tracks pixels (drops under scramble),
    and the lesion (stream silenced) collapses it."""
    inv = stg["heldpos_decode"] >= chance + a.decode_margin
    over_v1 = stg["heldpos_decode"] >= v1["heldpos_decode"] + a.add_delta
    over_frz = stg["heldpos_decode"] >= frz["heldpos_decode"] + a.add_delta
    scram = stg["scramble_decode"] <= chance + a.decode_margin
    rsa_ok = (stg["rsa_pixels_held"] >= a.rsa_gate
              and stg["rsa_pixels_scramble"] <= stg["rsa_pixels_held"] - a.rsa_drop)
    lesion_ok = lesion_decode <= chance + a.decode_margin
    return dict(inv_above_chance=bool(inv), adds_over_v1=bool(over_v1), adds_over_frozen=bool(over_frz),
                scramble_collapses=bool(scram), rsa_visual=bool(rsa_ok), lesion_collapses=bool(lesion_ok),
                validate=bool(inv and over_v1 and over_frz and scram and rsa_ok and lesion_ok))


def _deployed_weight_args(a):
    """A copy of `a` forced to the DEPLOYED operating point (g11_bg_runner defaults) -- used to record
    the deployed-point propagation profile regardless of --fair-drive."""
    import copy
    ad = copy.copy(a)
    ad.v1_weight_scale, ad.v1c_pool_weight, ad.v1c_pool_density_mult = 10.0, 2.0, 1.0
    ad.v1c_to_v2_weight, ad.v2_to_it_weight, ad.drive = 1.0, 1.5, 200.0
    return ad


def propagation_profile(seed, a, image, steps=40):
    """Cheap 1-image propagation probe at `a`'s CURRENT weights (NO training): total spike counts at
    each stage over `steps`. Documents where/whether the stream propagates (e.g. deployed weights are
    inert: V1_simple barely depolarizes, V1_complex/V2/IT silent)."""
    b, idx, _ = build_visual_bridge(seed, a, learn=False)
    xp, _ = get_backend()
    if xp.__name__ == "numpy":
        xp = None
    N = int(b.cp_firing_states.shape[0])
    _blank(b, xp, N, 10)
    _set_input(b, xp, N, idx["retina"], image_to_retina_drive(image, drive_max_pA=float(a.drive)))
    fire = {r: 0 for r in ["retina", "cortex_v1_simple", "cortex_v1_complex", "cortex_v2", "cortex_it"]}
    for _ in range(steps):
        b._run_one_simulation_step()
        fs = np.asarray(to_host(b.cp_firing_states)).astype(np.int64)
        for r in fire:
            fire[r] += int(fs[idx[r]].sum())
    del b
    return fire


# ============================================================================
# 6. Per-seed run.
# ============================================================================
def run_seed(seed, a):
    a.seed_base = seed
    n_cat = a.n_categories
    chance = 1.0 / n_cat
    categories_theta = [ci / n_cat * math.pi for ci in range(n_cat)]
    n_pos_total = a.n_train_pos + a.n_held_pos
    positions = _position_grid(n_pos_total, a.image_size, a.image_size * a.bar_len_frac)
    train_positions = positions[:a.n_train_pos]
    held_positions = positions[a.n_train_pos:n_pos_total]
    assert len(held_positions) == a.n_held_pos and len(train_positions) == a.n_train_pos, \
        "position split degenerate -- increase image_size or reduce positions/bar_len_frac"

    # --- build the three test sets (identity DECOUPLED from position; held positions are held out) ---
    train_imgs, train_cat, train_pos = build_object_set(categories_theta, train_positions, a.n_ex,
                                                        a.image_size, a.bar_len_frac, seed * 101 + 1)
    held_imgs, held_cat, _ = build_object_set(categories_theta, held_positions, a.n_ex,
                                              a.image_size, a.bar_len_frac, seed * 101 + 2)
    scr_imgs = scramble_images(held_imgs, seed * 101 + 3)
    scr_cat = held_cat.copy()

    out = {"seed": seed, "n_categories": n_cat, "chance": round(chance, 4),
           "n_train_images": int(train_imgs.shape[0]), "n_held_images": int(held_imgs.shape[0]),
           "train_positions": [[round(c, 2) for c in p] for p in train_positions],
           "held_positions": [[round(c, 2) for c in p] for p in held_positions]}

    # ---- propagation profiles (document inertness): DEPLOYED weights vs THIS run's weights ----
    rep_img = held_imgs[0]
    out["propagation_profile_deployed"] = propagation_profile(seed, _deployed_weight_args(a), rep_img)
    out["propagation_profile_thisrun"] = propagation_profile(seed, a, rep_img)
    print(f"  [seed {seed}] propagation (fire counts) DEPLOYED-weights {out['propagation_profile_deployed']} "
          f"| this-run {out['propagation_profile_thisrun']}", flush=True)

    # ========================= ARM: TRAINED (the deployed learned V2/IT) =========================
    bt, idxmap, n_gabor = build_visual_bridge(seed, a, learn=True)
    xp, _ = get_backend()
    if xp.__name__ == "numpy":
        xp = None   # numpy path: set arrays directly (no device transfer)
    N = int(bt.cp_firing_states.shape[0])
    out["n_neurons_total"] = N
    out["n_gabor_synapses"] = int(n_gabor)

    if a.trace_rule:
        diag = train_v2it_trace(bt, xp, idxmap, N, train_imgs, train_cat, train_pos, a,
                                shuffle_temporal=False)
    else:
        diag = train_v2it(bt, xp, idxmap, N, train_imgs, train_cat, a)
    out["train_firing_diag_first_scene"] = diag

    tr_spk, tr_dep = read_set(bt, xp, idxmap, N, train_imgs, a)
    hd_spk, hd_dep = read_set(bt, xp, idxmap, N, held_imgs, a)
    sc_spk, sc_dep = read_set(bt, xp, idxmap, N, scr_imgs, a)

    # ROBUST read mode per region (rate-code wall): SPIKES if the region fired above floor, else the
    # graded DEPOLARIZATION assembly response (the documented point-neuron read). One mode/region, all
    # arms, for comparability.
    read_mode = {r: ("spikes" if float(hd_spk[r].mean()) > a.inert_floor else "depol")
                 for r in REGIONS_READ}
    out["read_mode"] = read_mode

    def pick(spk, dep, r):
        return spk[r] if read_mode[r] == "spikes" else dep[r]

    trained_metrics = {}
    for r in REGIONS_READ:
        m = region_metrics(pick(tr_spk, tr_dep, r), train_cat, pick(hd_spk, hd_dep, r), held_cat,
                           held_imgs, pick(sc_spk, sc_dep, r), scr_cat, scr_imgs, n_cat)
        m["read_mode"] = read_mode[r]
        m["mean_spikes_per_neuron_heldset"] = round(float(hd_spk[r].mean()), 4)
        trained_metrics[r] = m
    out["trained"] = trained_metrics

    # ========================= LESION (shut v1c->v2 transmission at read time) =========================
    v2_rate_intact = float(hd_spk["cortex_v2"].mean())
    bt.set_transmission_gate("v2it_stream", 0.0)   # LESION applied -- never silently skipped
    hd_spk_les, hd_dep_les = read_set(bt, xp, idxmap, N, held_imgs, a)
    bt.set_transmission_gate("v2it_stream", 1.0)   # restore
    v2_rate_lesion = float(hd_spk_les["cortex_v2"].mean())
    lesion_decode = {}
    for r in ("cortex_v2", "cortex_it"):
        les_code = hd_spk_les[r] if read_mode[r] == "spikes" else hd_dep_les[r]
        lesion_decode[r] = round(float(centroid_decode(pick(tr_spk, tr_dep, r), train_cat,
                                                        les_code, held_cat)), 4)
    out["lesion"] = {
        "applied": True,
        "v2_mean_spikes_intact": round(v2_rate_intact, 4),
        "v2_mean_spikes_lesion": round(v2_rate_lesion, 4),
        "v2_input_silenced": bool(v2_rate_lesion < v2_rate_intact - 1e-6 or v2_rate_intact == 0.0),
        "heldpos_decode": lesion_decode,
    }
    print(f"  [seed {seed}] LESION v2 spikes intact {v2_rate_intact:.3f} -> lesion {v2_rate_lesion:.3f} "
          f"(mechanism {'BITES' if v2_rate_lesion < v2_rate_intact - 1e-6 else 'no-op'}); "
          f"IT heldpos-decode intact {trained_metrics['cortex_it']['heldpos_decode']:.2f} -> lesion "
          f"{lesion_decode['cortex_it']:.2f}", flush=True)
    del bt

    # ========================= ARM: NO-LEARNING (frozen random-init V2/IT) =========================
    bf, idxmap_f, _ = build_visual_bridge(seed, a, learn=False)
    xpf, _ = get_backend()
    if xpf.__name__ == "numpy":
        xpf = None
    Nf = int(bf.cp_firing_states.shape[0])
    tr_spk_f, tr_dep_f = read_set(bf, xpf, idxmap_f, Nf, train_imgs, a)
    hd_spk_f, hd_dep_f = read_set(bf, xpf, idxmap_f, Nf, held_imgs, a)
    sc_spk_f, sc_dep_f = read_set(bf, xpf, idxmap_f, Nf, scr_imgs, a)

    def pick_f(spk, dep, r):
        return spk[r] if read_mode[r] == "spikes" else dep[r]

    frozen_metrics = {}
    for r in REGIONS_READ:
        m = region_metrics(pick_f(tr_spk_f, tr_dep_f, r), train_cat, pick_f(hd_spk_f, hd_dep_f, r),
                           held_cat, held_imgs, pick_f(sc_spk_f, sc_dep_f, r), scr_cat, scr_imgs, n_cat)
        m["read_mode"] = read_mode[r]
        frozen_metrics[r] = m
    out["no_learning"] = frozen_metrics
    del bf

    # ===== ARM: SHUFFLED-TEMPORAL (--trace-rule only; the LOAD-BEARING domain dissociation) =====
    # Same trace MECHANISM (bouts, between-bout blank, decaying drive trace, STDP on) but within-bout
    # ORDER randomized so the trace can no longer bind a category across nearby positions. If the trace
    # benefit is real (temporal continuity, not just "more training"), IT here collapses toward V1/frozen.
    if a.trace_rule:
        bsh, idxmap_sh, _ = build_visual_bridge(seed, a, learn=True)
        Nsh = int(bsh.cp_firing_states.shape[0])
        train_v2it_trace(bsh, xp, idxmap_sh, Nsh, train_imgs, train_cat, train_pos, a,
                         shuffle_temporal=True)
        tr_spk_sh, tr_dep_sh = read_set(bsh, xp, idxmap_sh, Nsh, train_imgs, a)
        hd_spk_sh, hd_dep_sh = read_set(bsh, xp, idxmap_sh, Nsh, held_imgs, a)
        sc_spk_sh, sc_dep_sh = read_set(bsh, xp, idxmap_sh, Nsh, scr_imgs, a)

        def pick_sh(spk, dep, r):
            return spk[r] if read_mode[r] == "spikes" else dep[r]

        shuffled_metrics = {}
        for r in REGIONS_READ:
            m = region_metrics(pick_sh(tr_spk_sh, tr_dep_sh, r), train_cat,
                               pick_sh(hd_spk_sh, hd_dep_sh, r), held_cat, held_imgs,
                               pick_sh(sc_spk_sh, sc_dep_sh, r), scr_cat, scr_imgs, n_cat)
            m["read_mode"] = read_mode[r]
            shuffled_metrics[r] = m
        out["shuffled_temporal"] = shuffled_metrics
        del bsh

    # ========================= per-seed VALIDATE / RETIRE decision =========================
    # A LEARNED stage (V2 or IT) VALIDATES if it develops position-invariant category codes beating the
    # retinotopic V1_complex AND a frozen counterpart, with lesion/scramble/RSA collapsing. VALIDATE =
    # either learned stage validates; RETIRE = neither adds invariance over V1/frozen (orphaned/inert).
    v1_t = trained_metrics["cortex_v1_complex"]
    it_check = stage_check(trained_metrics["cortex_it"], v1_t, frozen_metrics["cortex_it"],
                           lesion_decode["cortex_it"], chance, a)
    v2_check = stage_check(trained_metrics["cortex_v2"], v1_t, frozen_metrics["cortex_v2"],
                           lesion_decode["cortex_v2"], chance, a)
    it_fires = trained_metrics["cortex_it"]["mean_spikes_per_neuron_heldset"] > a.inert_floor
    v2_fires = trained_metrics["cortex_v2"]["mean_spikes_per_neuron_heldset"] > a.inert_floor

    validate = bool(it_check["validate"] or v2_check["validate"])
    it_t = trained_metrics["cortex_it"]
    it_f = frozen_metrics["cortex_it"]
    # RETIRE: neither learned stage adds position-invariant category structure over V1/frozen.
    it_adds = it_check["inv_above_chance"] and (it_check["adds_over_v1"] or it_check["adds_over_frozen"])
    v2_adds = v2_check["inv_above_chance"] and (v2_check["adds_over_v1"] or v2_check["adds_over_frozen"])
    retire = bool(not it_adds and not v2_adds)
    verdict = "VALIDATE" if validate else ("RETIRE" if retire else "PARTIAL")

    out["decision"] = dict(
        it_stage=it_check, v2_stage=v2_check, it_fires=it_fires, v2_fires=v2_fires, verdict=verdict)

    # --- TRACE-RULE GO gate (only when --trace-rule): the base DiCarlo validate for IT (invariance above
    # chance, adds over V1_complex AND frozen IT, per-image scramble collapses, RSA tracks pixels, lesion
    # collapses) PLUS the domain dissociation: IT must beat the SHUFFLED-TEMPORAL arm by add_delta (i.e.
    # temporal continuity, not extra training, produced the position-invariant code). ---
    if a.trace_rule:
        sh_it = out["shuffled_temporal"]["cortex_it"]["heldpos_decode"]
        beats_shuffled = bool(it_t["heldpos_decode"] >= sh_it + a.add_delta)
        trace_go = bool(it_check["validate"] and beats_shuffled and it_fires)
        out["decision"]["shuffled_temporal_it_heldpos_decode"] = round(float(sh_it), 4)
        out["decision"]["trace_beats_shuffled"] = beats_shuffled
        out["decision"]["trace_go"] = trace_go
        out["decision"]["verdict"] = "TRACE-GO" if trace_go else "TRACE-NOGO"
        print(f"  [seed {seed}] TRACE-RULE: IT heldpos {it_t['heldpos_decode']:.2f} vs shuffled-temporal "
              f"{sh_it:.2f} (beats+{a.add_delta}={beats_shuffled}); base-validate {it_check['validate']}; "
              f"IT fires {it_fires} ==> {'TRACE-GO' if trace_go else 'TRACE-NOGO'}", flush=True)
    print(f"  [seed {seed}] heldpos-decode(train->held pos): IT {it_t['heldpos_decode']:.2f}"
          f"[{read_mode['cortex_it']}] | V2 {trained_metrics['cortex_v2']['heldpos_decode']:.2f}"
          f"[{read_mode['cortex_v2']}] | V1_complex {v1_t['heldpos_decode']:.2f} | frozen-IT "
          f"{it_f['heldpos_decode']:.2f} (chance {chance:.2f}) | IT fires {it_fires} | IT scramble "
          f"{it_t['scramble_decode']:.2f} | RSA-pix held {it_t['rsa_pixels_held']:.2f}/scr "
          f"{it_t['rsa_pixels_scramble']:.2f}  ==> {verdict}", flush=True)
    return out


# ============================================================================
# 7. Main.
# ============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    # object/position set
    p.add_argument("--n-categories", type=int, default=4, help="# orientations = # categories")
    p.add_argument("--n-train-pos", type=int, default=4, help="# TRAIN retinal positions")
    p.add_argument("--n-held-pos", type=int, default=2, help="# HELD-OUT retinal positions (invariance)")
    p.add_argument("--n-ex", type=int, default=4, help="exemplars per (category, position)")
    p.add_argument("--bar-len-frac", type=float, default=0.35, help="bar length as fraction of image")
    # deployed visual-hierarchy geometry (mirrors g11_bg_runner deployed defaults)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--n-orientations", type=int, default=8)
    p.add_argument("--n-frequencies", type=int, default=2)
    p.add_argument("--n-pos", type=int, default=8, help="V1 positions per dim (deployed=8)")
    p.add_argument("--n-v2", type=int, default=256)
    p.add_argument("--n-it", type=int, default=64)
    p.add_argument("--rf-radius", type=int, default=4)
    # pathway weights/density -- DEFAULTS = deployed (g11_bg_runner.py:2817-2848). At the DEPLOYED point
    # the stream is inert; --fair-drive boosts these so it actually propagates to IT (a FAIR test).
    p.add_argument("--v1-weight-scale", type=float, default=10.0, help="retina->V1 Gabor scale (deployed 10)")
    p.add_argument("--v1c-pool-weight", type=float, default=2.0, help="V1_simple->V1_complex weight (deployed 2)")
    p.add_argument("--v1c-pool-density-mult", type=float, default=1.0,
                   help="multiply the deployed V1_simple->V1_complex pooling density (deployed 1x = ~2/cell)")
    p.add_argument("--v1c-to-v2-weight", type=float, default=1.0, help="V1_complex->V2 weight (deployed 1.0)")
    p.add_argument("--v2-to-it-weight", type=float, default=1.5, help="V2->IT weight (deployed 1.5)")
    p.add_argument("--fair-drive", action="store_true",
                   help="give the stream a FAIR chance to reach IT: boost V1 scale + pooling density/weight "
                        "+ pathway weights + drive + stdp_w_max to a propagation operating point (the "
                        "deployed weights are inert). Explicit weight/drive/stdp flags still override.")
    # drive + timing
    p.add_argument("--drive", type=float, default=250.0, help="retina drive pA (peak)")
    p.add_argument("--train-epochs", type=int, default=8)
    p.add_argument("--scene-steps", type=int, default=30,
                   help="steps per training scene (>=~30 so V2/IT fire during training -> STDP shapes them)")
    p.add_argument("--read-steps", type=int, default=60, help="steps to accumulate a read code")
    p.add_argument("--settle-steps", type=int, default=15, help="blank+settle steps before each read/scene")
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--stdp-w-max", type=float, default=6.0)
    p.add_argument("--enable-ou", action="store_true", help="OU background noise (default OFF for clean reads)")
    # --- FÖLDIÁK TRACE / temporal-continuity training mode (default OFF -> byte-identical to the base run) ---
    p.add_argument("--trace-rule", action="store_true",
                   help="Földiák (1991) temporal-continuity curriculum (ports EMERGE-50): present each "
                        "category across a SEQUENCE of NEARBY retinal positions in contiguous bouts (no "
                        "within-bout blank -> the substrate STDP pre-trace carries across positions) with a "
                        "slow-decaying retinal-DRIVE eligibility trace over V1_complex pre-activity fed into "
                        "STDP potentiation, so V2/IT learn POSITION-INVARIANT category codes from temporal "
                        "continuity. Adds the SHUFFLED-TEMPORAL control arm + a TRACE-GO/NOGO verdict.")
    p.add_argument("--bout-len", type=int, default=_TRACE_BOUT_LEN,
                   help="temporal-grouping window: # consecutive same-category nearby-position scenes per "
                        "bout (reuse-by-import: EMERGE-50 default)")
    p.add_argument("--trace-decay", type=float, default=_TRACE_DECAY,
                   help="slow eligibility-trace decay on the retinal drive across a bout "
                        "(reuse-by-import: EMERGE-50 default)")
    # verdict thresholds (recorded)
    p.add_argument("--inert-floor", type=float, default=0.02,
                   help="IT mean spikes/neuron below this = inert")
    p.add_argument("--decode-margin", type=float, default=0.15, help="decode must beat chance by this")
    p.add_argument("--add-delta", type=float, default=0.10,
                   help="IT must beat V1/frozen heldpos-decode by this to 'add' invariance")
    p.add_argument("--rsa-gate", type=float, default=0.20, help="intact RSA-to-pixels floor")
    p.add_argument("--rsa-drop", type=float, default=0.15, help="scramble must drop RSA by this")
    p.add_argument("--out", type=str,
                   default="research/findings/raw/_perception_v2it_validate_or_retire.json")
    a = p.parse_args()

    # --fair-drive: a propagation operating point (found by probe: gets the full stream retina->V1->
    # V1_complex->V2->IT to fire, IT ~50-130 spikes/read). stdp_w_max is raised WELL ABOVE the pathway
    # weights so STDP does not soft-bound-collapse the propagation (the CLAUDE.md w_max gotcha). Only
    # overrides args the user LEFT AT DEFAULT (explicit flags win).
    if a.fair_drive:
        fair = {"v1_weight_scale": 200.0, "v1c_pool_weight": 200.0, "v1c_pool_density_mult": 100.0,
                "v1c_to_v2_weight": 200.0, "v2_to_it_weight": 250.0, "drive": 2500.0, "stdp_w_max": 600.0}
        for k, v in fair.items():
            if getattr(a, k) == p.get_default(k):   # not explicitly overridden by the caller
                setattr(a, k, v)
        print(f"[fair-drive] boosted operating point: v1_scale={a.v1_weight_scale} "
              f"pool_w={a.v1c_pool_weight} pool_dmult={a.v1c_pool_density_mult} v1c->v2={a.v1c_to_v2_weight} "
              f"v2->it={a.v2_to_it_weight} drive={a.drive} stdp_w_max={a.stdp_w_max}", flush=True)

    xp, backend = get_backend()
    t0 = time.time()
    print(f"[v2it validate-or-retire] backend={backend}  seeds={a.seeds}\n"
          f"  Deployed hierarchy: retina({2*a.image_size**2}) -> V1_simple("
          f"{a.n_orientations*a.n_frequencies*a.n_pos*a.n_pos}, Gabor) -> V1_complex("
          f"{a.n_orientations*a.n_pos*a.n_pos}) -> V2({a.n_v2}) -> IT({a.n_it}).\n"
          f"  Does trained V2/IT add category-discriminative + POSITION-INVARIANT codes over V1/frozen, "
          f"or is it inert? (lesion / per-image scramble / RSA-pixel / no-learning anti-cheats)",
          flush=True)

    rows = [run_seed(s, a) for s in a.seeds]

    def col(f):
        return [f(r) for r in rows]
    chance = rows[0]["chance"]
    it_heldpos = col(lambda r: r["trained"]["cortex_it"]["heldpos_decode"])
    v1_heldpos = col(lambda r: r["trained"]["cortex_v1_complex"]["heldpos_decode"])
    frozen_heldpos = col(lambda r: r["no_learning"]["cortex_it"]["heldpos_decode"])
    it_scramble = col(lambda r: r["trained"]["cortex_it"]["scramble_decode"])
    it_rsa = col(lambda r: r["trained"]["cortex_it"]["rsa_pixels_held"])
    it_fires_all = all(r["decision"]["it_fires"] for r in rows)
    verdicts = [r["decision"]["verdict"] for r in rows]

    n_validate = sum(v == "VALIDATE" for v in verdicts)
    n_retire = sum(v == "RETIRE" for v in verdicts)
    if n_validate == len(rows):
        overall = "VALIDATE"
    elif n_retire == len(rows):
        overall = "RETIRE"
    elif n_validate >= len(rows) - n_retire and n_validate > n_retire:
        overall = "PARTIAL-LEAN-VALIDATE"
    elif n_retire > n_validate:
        overall = "RETIRE-MAJORITY"
    else:
        overall = "PARTIAL"

    summary = dict(
        overall_verdict=overall,
        seeds=a.seeds, backend=backend, chance=chance,
        per_seed_verdicts=verdicts,
        it_heldpos_decode_mean=round(float(np.mean(it_heldpos)), 4),
        it_heldpos_decode_min=round(float(np.min(it_heldpos)), 4),
        v1_complex_heldpos_decode_mean=round(float(np.mean(v1_heldpos)), 4),
        frozen_it_heldpos_decode_mean=round(float(np.mean(frozen_heldpos)), 4),
        it_scramble_decode_mean=round(float(np.mean(it_scramble)), 4),
        it_rsa_pixels_mean=round(float(np.mean(it_rsa)), 4),
        it_fires_all_seeds=bool(it_fires_all),
        interpretation=(
            "VALIDATE = trained IT firing codes are position-invariant + category-discriminative, "
            "beating both the retinotopic V1_complex and a frozen IT, with lesion/scramble/RSA "
            "collapsing. RETIRE = trained IT is inert / adds nothing over V1_complex or frozen IT "
            "(standardize grounding on the V1->pooler codon)."),
        knobs=vars(a),
    )

    # --- TRACE-RULE summary + GO gate (only when --trace-rule) ---
    if a.trace_rule:
        trace_go_flags = [bool(r["decision"].get("trace_go", False)) for r in rows]
        sh_heldpos = col(lambda r: r["shuffled_temporal"]["cortex_it"]["heldpos_decode"])
        n_trace_go = sum(trace_go_flags)
        if n_trace_go == len(rows):
            trace_overall = "TRACE-GO"
        elif n_trace_go == 0:
            trace_overall = "TRACE-NOGO"
        else:
            trace_overall = f"TRACE-PARTIAL-{n_trace_go}/{len(rows)}"
        summary["trace_rule"] = dict(
            enabled=True, bout_len=a.bout_len, trace_decay=a.trace_decay,
            trace_overall=trace_overall, per_seed_trace_go=trace_go_flags, n_trace_go=n_trace_go,
            it_heldpos_decode_mean=round(float(np.mean(it_heldpos)), 4),
            shuffled_temporal_it_heldpos_decode_mean=round(float(np.mean(sh_heldpos)), 4),
            go_gate=("TRACE-GO (all seeds) requires, per seed: trained IT (via the Földiák temporal-"
                     "continuity curriculum) heldpos-decode >= chance+decode_margin AND >= V1_complex"
                     "+add_delta AND >= frozen-IT+add_delta AND per-image scramble collapses AND RSA "
                     "tracks pixels AND the v1c->v2 lesion collapses it (the base DiCarlo validate) AND "
                     ">= SHUFFLED-TEMPORAL IT+add_delta (temporal continuity, not extra training, did it) "
                     "AND IT fires above inert_floor."),
        )

    out = dict(summary=summary, per_seed=rows)
    outp = os.path.join(_REPO, a.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as fh:
        json.dump(out, fh, indent=2, default=str)

    print(f"\n{'='*100}")
    print(json.dumps(summary, indent=2, default=str))
    print(f"{'='*100}")
    print(f"[written] {outp}\nTotal elapsed: {time.time()-t0:.1f}s", flush=True)
    # exit code (honest tri-state). --trace-rule: 0 TRACE-GO, 1 TRACE-NOGO, 2 partial. Base run: 0
    # VALIDATE, 1 RETIRE, 2 in between.
    if a.trace_rule:
        raise SystemExit(0 if trace_overall == "TRACE-GO" else (1 if trace_overall == "TRACE-NOGO" else 2))
    raise SystemExit(0 if overall == "VALIDATE" else (1 if overall in ("RETIRE", "RETIRE-MAJORITY") else 2))


if __name__ == "__main__":
    main()
