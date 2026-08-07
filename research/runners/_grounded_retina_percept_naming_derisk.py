"""Grounded RETINA->V1 PERCEPT driving the on-bridge naming map — the word is selected from a percept the brain
SEES, not a fixed code.

WHY THIS RUNNER EXISTS
----------------------
The 2026-08-07 on-bridge naming de-risk (`_grounded_message_to_word_onbridge_derisk.py`, 6-seed PARENT-VERIFIED
GO) put the referent-naming map on real Izhikevich neurons: a plastic percept->word pathway, decoded from
word-pool spike counts, no weight transport. But its finding names a residual, verbatim:
  * "the percept assemblies are deterministic rather than emerged from vision"
  * next-mechanism 2: "Drive the percept assemblies from the neural retina/visual features rather than fixed codes."
The percept was `make_assemblies` -- ARBITRARY sparse random codes with NO relation to any image. This runner
burns that residual down: the percept that drives the naming map is now the firing of real V1 simple cells
responding, through the project's Gabor retina->V1 receptive-field bank, to the OBJECT'S RENDERED IMAGE. The
retina sees the object; V1 fires; that V1 firing -- noisy, distributed, viewpoint-varying -- drives the word
pools through the learned naming synapses; the word is decoded from word-pool spike counts. The sensation->naming
seam is closed with real perception.

WHAT IS BRAIN-BASED vs SCAFFOLD (declared, not hidden)
------------------------------------------------------
  * BRAIN (this rung's deliverable): the percept driving the naming map is the FIRING of cortex_v1_simple
    neurons, produced by the retina's response to the object image through the FIXED Gabor RF bank (V1 simple-
    cell RFs are innate/early-developing -- Hubel-Wiesel; the same bank the deployed nav/EMERGE stacks use). The
    naming map v1->word is a PLASTIC synaptic pathway, learned by on-bridge rate-Hebbian coincidence, GATED (open
    only during the teacher naming event). At inference the gate is closed and ONLY the retina is driven -- V1
    fires, propagates through the learned synapses, and the referent is the argmax of WORD-POOL SPIKE COUNTS. The
    true label is never on the read path; no weight transport.
  * WORLD/BODY (legitimate host): rendering the object's image the retina then sees (an oriented-bar shape per
    object -- a NAMED scaffold: richer object images are a follow-on, exactly as EMERGE-34 declared). The
    fixed Gabor retina->V1 transform is the innate sensory front end (not learned here).
  * TEACHER (legitimate social environment): during a naming event the caregiver co-activates the object's word
    pool while the object is seen. Present only during LEARNING.
  * ARTICULATION (legitimate host body): each word pool has a FIXED binding to one WKV vocab token; the numpy WKV
    forward renders the brain-decoded word (the same fixed articulatory scaffold the on-bridge rung used).

GO GATE (all must hold) — mirrors the on-bridge naming controls, now on RETINAL percepts
----------------------------------------------------------------------------------------
  1. naming accuracy from RETINAL percepts (held-out jittered+noisy exemplars) >= threshold, well above chance.
  2. (a) LEARNED: an untrained RANDOM v1->word map decodes at chance.
  3. (b) RETINA-DERIVED: at inference ONLY the retina is externally driven (word/V1 never injected); V1 fires
     (>0) and each object's V1 code is object-specific.
  4. (c) DISCRIMINATION: the 4 objects produce distinguishable V1 codes (low cross-object cosine) that name 4
     distinct words.
  5. (d) GENERALIZATION: naming holds on HELD-OUT exemplars (viewpoint jitter + pixel noise) never taught.
  6. (e) NO weight transport: name_from_retina takes the IMAGE only; the label never on the read path.
  7. LESION (zero the v1->word pathway) collapses to chance AND emits no confident decode (fails SAFE).
  8. PERMUTATION: teaching a permuted object->word map decodes the permutation; the original word is rejected.
  9. NOVEL untaught object abstains (word-pool margin below the confidence threshold).
 10. anti-drift: the FIXED Gabor retina->V1 weights are unchanged by teaching (only v1->word learns).

Backend: pure numpy (real spiking Izhikevich) + the numpy WKV forward. CPU. Run:
  PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -m research.runners._grounded_retina_percept_naming_derisk \
      --seeds 42 43 44 100 101 102 \
      --out research/findings/raw/grounded_retina_percept_naming/retina_naming_6seed.json
Smoke (1 seed, fewer trials/steps):
  PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -m research.runners._grounded_retina_percept_naming_derisk \
      --seeds 42 --smoke --out research/findings/raw/grounded_retina_percept_naming/smoke.json
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
sys.path.insert(0, _HERE)  # for _wkv_faculty (sibling of the naming runners)

from sim.backend import get_backend, to_host  # noqa: E402
from sim.visual_cortex import (  # noqa: E402
    RETINA_SIZE, N_ORIENTATIONS, N_FREQUENCIES, V1_POSITIONS_PER_DIM, N_RETINA_CHANNELS,
    apply_v1_gabor_weights,
)
# Reuse the referents (object, agent, verb, word) + WKV articulation harness from the on-bridge naming rung.
from research.runners._grounded_message_to_word_derisk import REFERENTS, GROUNDED_CKPT  # noqa: E402
# Reuse the REAL visual front end: render objects as oriented-bar shapes (one visual category per referent).
from research.runners._genfrontier_optionB_visual_similarity_derisk import build_shape_set  # noqa: E402
from _wkv_faculty import WKVFaculty  # noqa: E402
from tools.verdict import Verdict, GO  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

N_V1 = N_ORIENTATIONS * N_FREQUENCIES * V1_POSITIONS_PER_DIM * V1_POSITIONS_PER_DIM   # 8192
N_RETINA = N_RETINA_CHANNELS * RETINA_SIZE * RETINA_SIZE                              # 2048

# spiking-decode confidence: margin between the two most-active word pools + an absolute floor.
CONF_MARGIN_SPK = 0.20
MIN_WORD_SPIKES = 6.0


def _csr_pre_post(bridge):
    csr = bridge.cp_connections
    indptr = np.asarray(to_host(csr.indptr), dtype=np.int64)
    post = np.asarray(to_host(csr.indices), dtype=np.int64)
    pre = np.empty(post.shape[0], dtype=np.int64)
    for row in range(int(csr.shape[0])):
        pre[indptr[row]:indptr[row + 1]] = row
    return pre, post


def _pathway_mask(bridge, pre_indices, post_indices):
    """Boolean mask (device+host) over cp_connections.data selecting synapses pre in `pre_indices`,
    post in `post_indices`."""
    pre, post = _csr_pre_post(bridge)
    mask_h = np.isin(pre, np.asarray(pre_indices)) & np.isin(post, np.asarray(post_indices))
    if not mask_h.any():
        raise RuntimeError("requested pathway has no synapses")
    xp, _ = get_backend()
    return xp.asarray(mask_h), mask_h


def build_retina_naming_bridge(seed, a):
    """ONE SimulationBridge: retina -> cortex_v1_simple (FIXED Gabor) -> word (PLASTIC naming, gated), plus the
    request/silence/gate_fs spiking gate. The percept driving the naming map is V1's firing to the object image."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType
    k = len(REFERENTS)
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="retina", n_neurons=N_RETINA, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="cortex_v1_simple", n_neurons=N_V1, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="word", n_neurons=k * a.word_nper, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="request", n_neurons=a.gate_n, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="silence", n_neurons=a.gate_n, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="gate_fs", n_neurons=a.gate_fs_n, exc_fraction=0.0, internal_density=0.0),
    ]
    cfg.region_pathways = [
        # FIXED innate retina->V1 Gabor RF bank (installed below via apply_v1_gabor_weights). Declared plastic=False
        # at density 0 so the Gabor transform never drifts -- only the naming map learns.
        RegionPathway(from_region="retina", to_region="cortex_v1_simple", density=0.0,
                      weight_mean=0.0, weight_jitter=0.0, plastic=False),
        # THE NAMING MAP: V1 firing -> word pools, plastic (rate-Hebbian), zero-ish init. Learning is toggled
        # by directly gating the pathway's per-synapse plasticity gain (see build below) rather than a NAMED
        # plasticity_gate -- the Gabor install below rebuilds the CSR via add_missing, which would invalidate a
        # named gate's synapse mapping (the B1 foot-gun); the substrate is frozen globally and only this mask opens.
        RegionPathway(from_region="cortex_v1_simple", to_region="word", density=1.0,
                      weight_mean=a.init_w, weight_jitter=0.0, plastic=True),
        # THE GATE: request/silence excite a shared FS inhibitor which inhibits both -> WTA race.
        RegionPathway(from_region="request", to_region="gate_fs", density=1.0,
                      weight_mean=a.gate_exc_w, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="silence", to_region="gate_fs", density=1.0,
                      weight_mean=a.gate_exc_w, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="gate_fs", to_region="request", density=1.0,
                      weight_mean=a.gate_inh_w, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="gate_fs", to_region="silence", density=1.0,
                      weight_mean=a.gate_inh_w, weight_jitter=0.0, plastic=False),
    ]
    cfg.dt = 1.0
    # ⛔ the CLAUDE.md gotcha: set cfg.seed (NOT actual_seed_used) so the SUBSTRATE is actually seeded.
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    # RATE-HEBBIAN (symmetric referent<->word coincidence); STDP is measured-negative on symmetric co-firing.
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = a.hebb_rate
    cfg.hebbian_max_weight = a.hebb_max
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_weight_decay = 1e-5
    # Freeze the substrate around the naming map so the learned pathway is stable across the long inference
    # battery and repeated presentation cannot fatigue a word pool (DECLARED disabled scope).
    cfg.enable_structural_plasticity = False
    cfg.enable_homeostasis = False
    cfg.homeostasis_threshold_adapt_rate = 0.0
    cfg.enable_synaptic_scaling = False
    rt = RuntimeState(); rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt,
                              gpu_config=GPUConfig())
    bridge._initialize_simulation_data()
    # Install the FIXED Gabor retina->V1 weights (the deployed innate RF bank). This add_missing REBUILDS the CSR.
    apply_v1_gabor_weights(bridge, weight_scale=a.gabor_scale)
    rm = bridge.region_manager
    idx = {name: np.asarray(rm.indices(name), dtype=np.int64)
           for name in ("retina", "cortex_v1_simple", "word", "request", "silence", "gate_fs")}
    idx["word_blocks"] = idx["word"].reshape(k, a.word_nper)
    idx["r0"] = int(idx["retina"][0])
    # FREEZE the whole substrate (gain 0 -> no potentiation, no decay, no clip anywhere) AFTER the CSR rebuild, so
    # the innate Gabor RF bank never drifts. Only the V1->word naming synapses are opened (mask below) during the
    # teacher naming event -- the decay/clip are gain-gated (bridge.py:7958,7982), so gain 0 preserves weights verbatim.
    bridge.set_global_plasticity_gain(0.0)
    naming_mask, _ = _pathway_mask(bridge, idx["cortex_v1_simple"], idx["word"])
    idx["naming_mask"] = naming_mask   # device bool mask over cp_connections.data (the plastic V1->word synapses)
    return bridge, idx


def _run(bridge, n):
    for _ in range(int(n)):
        bridge._run_one_simulation_step()


def _count(bridge, indices):
    fs = np.asarray(to_host(bridge.cp_firing_states))
    return float(fs[indices].sum())


def _drive_retina(bridge, idx, image, drive_pA, xp, extra=None):
    """Set the retina external input from the object image (channel-first flatten). Everything else is 0 --
    the ONLY externally driven region during inference is the retina."""
    flat = (image.reshape(-1).astype(np.float32)) * float(drive_pA)
    N = int(bridge.core_config.num_neurons)
    drive = xp.zeros(N, dtype=xp.float32)
    drive[xp.asarray(idx["retina"])] = xp.asarray(flat)
    if extra is not None:
        for units, amp in extra:
            drive[xp.asarray(units)] = np.float32(amp)
    bridge.cp_external_input_current[:] = drive
    return drive


def teach_naming(bridge, idx, images, labels, targets, a):
    """Gated local-Hebbian naming: OPEN the "naming" gate; for each teaching exemplar drive the RETINA with the
    object image (V1 fires via Gabor) AND co-drive the TEACHER-selected word pool. Repeated co-firing potentiates
    the V1->word synapses. Close the gate afterward (inference is plasticity-frozen)."""
    xp, _ = get_backend()
    m = idx["naming_mask"]
    bridge.cp_plasticity_rate_gain[m] = np.float32(1.0)   # OPEN only the V1->word synapses (rest stays frozen at 0)
    order = list(range(len(images)))
    for _ep in range(a.teach_epochs):
        for j in order:
            c = int(labels[j])
            wpool = idx["word_blocks"][targets[c]]
            _drive_retina(bridge, idx, images[j], a.retina_drive, xp,
                          extra=[(wpool, a.teach_drive)])
            for _ in range(a.teach_steps):
                bridge._run_one_simulation_step()
            bridge.cp_external_input_current[:] = 0.0
            _run(bridge, a.settle_steps)
    bridge.cp_plasticity_rate_gain[m] = np.float32(0.0)   # CLOSE -> inference reads only, weights preserved verbatim


def v1_code(bridge, idx, image, a):
    """Read the V1 firing code produced by the retina's response to `image` (word NOT driven). Returns the
    per-V1-cell spike-count vector over the read window -- the actual RETINAL PERCEPT."""
    xp, _ = get_backend()
    counts = np.zeros(N_V1)
    bridge.cp_external_input_current[:] = 0.0
    _run(bridge, a.washout_steps)
    for _ in range(a.decode_steps):
        _drive_retina(bridge, idx, image, a.retina_drive, xp)
        bridge._run_one_simulation_step()
        fs = np.asarray(to_host(bridge.cp_firing_states))
        counts += fs[idx["cortex_v1_simple"]]
    bridge.cp_external_input_current[:] = 0.0
    return counts


def name_from_retina(bridge, idx, image, a):
    """Inference: plasticity CLOSED. Drive ONLY the retina with the object image; V1 fires through the Gabor
    bank; the learned V1->word synapses propagate to the word pools; decode = argmax of WORD-POOL SPIKE COUNTS
    (NOT a host matmul; the label is never an argument). Returns (winner, margin, word_counts, top1, v1_spikes)."""
    xp, _ = get_backend()
    k = len(idx["word_blocks"])
    counts = np.zeros(k)
    v1_spikes = 0.0
    bridge.cp_external_input_current[:] = 0.0
    _run(bridge, a.washout_steps)
    for _ in range(a.decode_steps):
        _drive_retina(bridge, idx, image, a.retina_drive, xp)
        bridge._run_one_simulation_step()
        fs = np.asarray(to_host(bridge.cp_firing_states))
        for w in range(k):
            counts[w] += float(fs[idx["word_blocks"][w]].sum())
        v1_spikes += float(fs[idx["cortex_v1_simple"]].sum())
    bridge.cp_external_input_current[:] = 0.0
    order = np.argsort(-counts)
    top1, top2 = counts[order[0]], counts[order[1]] if k > 1 else 0.0
    margin = (top1 - top2) / (top1 + top2 + 1e-9)
    return int(order[0]), float(margin), counts, float(top1), v1_spikes


def confident(margin, top1):
    return bool(margin > CONF_MARGIN_SPK and top1 >= MIN_WORD_SPIKES)


def naming_acc(bridge, idx, images, labels, targets, a):
    """Fraction of held-out (jittered+noisy) percept presentations that decode (from spikes) to the taught word."""
    correct = total = 0
    v1_ok = True
    for j in range(len(images)):
        c = int(labels[j])
        w, _, _, _, v1s = name_from_retina(bridge, idx, images[j], a)
        correct += int(w == targets[c]); total += 1
        if v1s <= 0.0:
            v1_ok = False
    return correct / total, v1_ok


def gate_race(bridge, idx, hunger, satiety, food_cue, a):
    """Spiking request-vs-silence race on the same bridge (moat mirror)."""
    xp, _ = get_backend()
    N = int(bridge.core_config.num_neurons)
    drive = xp.zeros(N, dtype=xp.float32)
    drive[xp.asarray(idx["request"])] = np.float32(a.gate_drive * (1.2 * food_cue + 1.0 * hunger))
    drive[xp.asarray(idx["silence"])] = np.float32(a.gate_drive * (1.3 * satiety))
    bridge.cp_external_input_current[:] = 0.0
    _run(bridge, a.washout_steps)
    req = sil = 0.0
    for _ in range(a.gate_steps):
        bridge.cp_external_input_current[:] = drive
        bridge._run_one_simulation_step()
        req += _count(bridge, idx["request"]); sil += _count(bridge, idx["silence"])
    bridge.cp_external_input_current[:] = 0.0
    return ("request" if req > sil else "silence"), float(req), float(sil)


def _cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def _split_exemplars(images, labels, k, hold, rng):
    """Per category: shuffle exemplars, keep first (n-hold) for teaching, last `hold` held-out for inference."""
    train_i, held_i = [], []
    for c in range(k):
        ci = [i for i in range(len(labels)) if int(labels[i]) == c]
        rng.shuffle(ci)
        held_i += ci[len(ci) - hold:]
        train_i += ci[:len(ci) - hold]
    return train_i, held_i


def run_seed(seed, a, fac, verbose=True):
    xp, _ = get_backend()
    k = len(REFERENTS)
    rng = np.random.default_rng(seed)
    n_ex = a.smoke_ex if a.smoke else a.n_ex
    hold = a.hold
    # Render k object categories (one per referent) + 1 NOVEL untaught category through the REAL front end.
    imgs, labels, _ = build_shape_set(n_categories=k + 1, n_exemplars=n_ex, rng=rng)
    novel_c = k  # the extra category = the untaught object (moat/abstain)
    is_novel = labels == novel_c
    imgs_k, labels_k = imgs[~is_novel], labels[~is_novel]
    imgs_novel = imgs[is_novel]
    train_i, held_i = _split_exemplars(imgs_k, labels_k, k, hold, np.random.default_rng(seed + 9))
    ident = list(range(k))

    # --- 1. teach the identity naming map on TRAINING exemplars; measure HELD-OUT naming accuracy -----------
    bridge, idx = build_retina_naming_bridge(seed, a)
    gabor_mask, gabor_mask_h = _pathway_mask(bridge, idx["retina"], idx["cortex_v1_simple"])
    gabor_before = np.asarray(to_host(bridge.cp_connections.data))[gabor_mask_h].copy()
    teach_naming(bridge, idx, imgs_k[train_i], labels_k[train_i], ident, a)
    # anti-drift: the fixed Gabor retina->V1 weights must be UNCHANGED by teaching (only V1->word learns).
    gabor_after = np.asarray(to_host(bridge.cp_connections.data))[gabor_mask_h]
    gabor_frozen = bool(np.allclose(gabor_before, gabor_after))

    name_acc, v1_fires = naming_acc(bridge, idx, imgs_k[held_i], labels_k[held_i], ident, a)

    # --- 3+4. RETINA-DERIVED + DISCRIMINATION: per-object V1 codes, cross-object cosine, distinct words -----
    obj_codes = []
    obj_words = []
    for c in range(k):
        # a representative held-out exemplar per object
        ex = [j for j in held_i if int(labels_k[j]) == c][0]
        obj_codes.append(v1_code(bridge, idx, imgs_k[ex], a))
        w, _, _, _, _ = name_from_retina(bridge, idx, imgs_k[ex], a)
        obj_words.append(w)
    cross_cos = float(np.mean([_cos(obj_codes[i], obj_codes[j])
                               for i in range(k) for j in range(i + 1, k)]))
    v1_nonzero = bool(all(c.sum() > 0 for c in obj_codes))
    distinct_words = int(len(set(obj_words)))

    # --- 9. NOVEL untaught object must ABSTAIN (below confidence) -------------------------------------------
    nw, nmargin, _, ntop1, _ = name_from_retina(bridge, idx, imgs_novel[0], a)
    novel_abstains = bool(not confident(nmargin, ntop1))

    # --- moat: spiking gate routes silence -> ZERO word output; render the spike-decoded word via WKV --------
    hungry_dec, hreq, hsil = gate_race(bridge, idx, hunger=1.0, satiety=0.0, food_cue=1.0, a=a)
    sated_dec, sreq, ssil = gate_race(bridge, idx, hunger=0.0, satiety=1.0, food_cue=1.0, a=a)
    word_output_events = 0
    render_ok, spoken = [], []
    # ARTICULATION FAITHFULNESS over ALL held-out exemplars (statistically interpretable, not 3/4): the body must
    # speak the word the BRAIN DECODED FROM SPIKES (patient == chosen_word), whatever that decode is -- decode
    # CORRECTNESS is measured separately by name_acc. This isolates "the mouth says the word the brain selected".
    for j in held_i:
        c = int(labels_k[j])
        obj, agent, verb, word = REFERENTS[c]
        w, _, _, _, _ = name_from_retina(bridge, idx, imgs_k[j], a)
        chosen_word = REFERENTS[w][3]
        if hungry_dec == "request":
            word_output_events += 1
            frame = ["the", agent, verb, chosen_word]
            fac.n_invocations = 0
            utter = fac.answer(" ".join(frame) + " .", "q")
            patient = utter.split()[-1] if utter.split() else ""
            render_ok.append(patient == chosen_word and fac.n_invocations == 1)
            if len(spoken) < k:
                spoken.append(utter)
        if sated_dec == "request":
            word_output_events += 1  # must not happen when sated
    render_faithful = float(np.mean(render_ok)) if render_ok else 0.0
    silent_zero_output = bool(sated_dec == "silence")

    # --- 7. LESION: zero the learned V1->word pathway -> collapse to chance, no confident decode -------------
    mask_x, _ = _pathway_mask(bridge, idx["cortex_v1_simple"], idx["word"])
    saved_w = bridge.cp_connections.data[mask_x].copy()
    bridge.cp_connections.data[mask_x] = xp.asarray(np.zeros(int(mask_x.sum()), dtype=np.float32))
    lesion_acc, _ = naming_acc(bridge, idx, imgs_k[held_i][:max(1, len(held_i) // 2)],
                               labels_k[held_i][:max(1, len(held_i) // 2)], ident, a)
    lesionD = []
    for c in range(k):
        ex = [j for j in held_i if int(labels_k[j]) == c][0]
        lesionD.append(name_from_retina(bridge, idx, imgs_k[ex], a))
    lesion_confident = float(np.mean([confident(d[1], d[3]) for d in lesionD]))
    bridge.cp_connections.data[mask_x] = xp.asarray(saved_w)

    # --- 2. CHANCE control: an UNTRAINED RANDOM V1->word map (learned effect is not a substrate bias) -------
    bridge_r, idx_r = build_retina_naming_bridge(seed, a)
    mask_r, _ = _pathway_mask(bridge_r, idx_r["cortex_v1_simple"], idx_r["word"])
    rw = np.abs(rng.normal(0.0, a.hebb_max * 0.5, size=int(mask_r.sum()))).astype(np.float32)
    bridge_r.cp_connections.data[mask_r] = xp.asarray(rw)
    name_acc_chance, _ = naming_acc(bridge_r, idx_r, imgs_k[held_i], labels_k[held_i], ident, a)

    # --- 8. PERMUTATION control: teach a derangement -> decode the PERMUTATION, reject the original ----------
    perm = list(np.roll(ident, 1))
    bridge_p, idx_p = build_retina_naming_bridge(seed, a)
    teach_naming(bridge_p, idx_p, imgs_k[train_i], labels_k[train_i], perm, a)
    perm_followed, _ = naming_acc(bridge_p, idx_p, imgs_k[held_i], labels_k[held_i], perm, a)
    orig_accepted, _ = naming_acc(bridge_p, idx_p, imgs_k[held_i], labels_k[held_i], ident, a)

    row = {
        "seed": int(seed), "k": k, "n_ex": n_ex, "hold": hold,
        "name_acc": name_acc, "name_acc_chance": name_acc_chance,
        "v1_fires_every_trial": bool(v1_fires), "v1_nonzero_per_object": v1_nonzero,
        "cross_object_v1_cosine": cross_cos, "distinct_words": distinct_words,
        "gabor_frozen_after_teach": gabor_frozen,
        "render_faithful": render_faithful, "silent_zero_output": silent_zero_output,
        "hungry_decision": hungry_dec, "sated_decision": sated_dec,
        "gate_hungry_req_sil": [hreq, hsil], "gate_sated_req_sil": [sreq, ssil],
        "word_output_events": int(word_output_events),
        "lesion_acc": lesion_acc, "lesion_confident_frac": lesion_confident,
        "perm_followed": perm_followed, "orig_accepted_after_perm": orig_accepted,
        "novel_abstains": novel_abstains, "novel_margin": float(nmargin), "novel_top1": float(ntop1),
        "spoken": spoken,
    }
    if verbose:
        print("  seed %-4d name_acc=%.3f (chance=%.3f lesion=%.3f) xcos=%.3f dwords=%d gabor_frozen=%s "
              "render=%.3f gate[h=%s s=%s] perm=%.3f orig=%.3f novel_abstains=%s | %s"
              % (seed, name_acc, name_acc_chance, lesion_acc, cross_cos, distinct_words, gabor_frozen,
                 render_faithful, hungry_dec, sated_dec, perm_followed, orig_accepted, novel_abstains,
                 spoken[0] if spoken else "(silent)"), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--ckpt", default=GROUNDED_CKPT)
    ap.add_argument("--out", default="research/findings/raw/grounded_retina_percept_naming/retina_naming_6seed.json")
    ap.add_argument("--smoke", action="store_true", help="1 seed, fewer exemplars/steps")
    ap.add_argument("--smoke-ex", type=int, default=6)
    ap.add_argument("--n-ex", type=int, default=12, help="exemplars per object (train n_ex-hold, hold held-out)")
    ap.add_argument("--hold", type=int, default=4, help="held-out jittered exemplars per object (generalization)")
    # substrate / pools
    ap.add_argument("--word-nper", type=int, default=48)
    ap.add_argument("--gate-n", type=int, default=16)
    ap.add_argument("--gate-fs-n", type=int, default=8)
    ap.add_argument("--gabor-scale", type=float, default=300.0, help="Gabor RF weight multiplier (V1 excitability)")
    # naming pathway + teaching
    ap.add_argument("--init-w", type=float, default=0.03)
    ap.add_argument("--hebb-rate", type=float, default=0.1)
    ap.add_argument("--hebb-max", type=float, default=50.0)
    ap.add_argument("--teach-epochs", type=int, default=10)
    ap.add_argument("--teach-steps", type=int, default=40)
    ap.add_argument("--settle-steps", type=int, default=8)
    ap.add_argument("--retina-drive", type=float, default=2000.0, help="external pA scale on the object image")
    ap.add_argument("--teach-drive", type=float, default=600.0)
    # inference / decode
    ap.add_argument("--washout-steps", type=int, default=15)
    ap.add_argument("--decode-steps", type=int, default=120)
    # gate race
    ap.add_argument("--gate-drive", type=float, default=180.0)
    ap.add_argument("--gate-steps", type=int, default=40)
    ap.add_argument("--gate-exc-w", type=float, default=1.0)
    ap.add_argument("--gate-inh-w", type=float, default=2.0)
    a = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")

    seeds = a.seeds[:1] if a.smoke else a.seeds
    k = len(REFERENTS)
    chance = 1.0 / k
    t0 = time.time()
    print("[retina-percept naming de-risk] seeds=%s smoke=%s -- does the on-bridge naming map still select the "
          "correct referent word when its percept is the REAL V1 firing to the object's rendered image "
          "(retina->Gabor V1->word), across held-out viewpoint jitter + noise, all anti-cheats on spikes?"
          % (seeds, a.smoke), flush=True)
    fac = WKVFaculty(ckpt=a.ckpt, max_new=12)
    rows = [run_seed(s, a, fac) for s in seeds]

    agg = lambda key: float(np.mean([r[key] for r in rows]))
    mean_name = agg("name_acc"); mean_chance = agg("name_acc_chance"); mean_lesion = agg("lesion_acc")
    mean_render = agg("render_faithful"); mean_perm = agg("perm_followed"); mean_orig = agg("orig_accepted_after_perm")
    mean_xcos = agg("cross_object_v1_cosine")
    all_silent = all(r["silent_zero_output"] for r in rows)
    all_hungry_req = all(r["hungry_decision"] == "request" for r in rows)
    all_lesion_conf0 = all(r["lesion_confident_frac"] == 0.0 for r in rows)
    all_novel = all(r["novel_abstains"] for r in rows)
    all_v1_fires = all(r["v1_fires_every_trial"] for r in rows)
    all_v1_nonzero = all(r["v1_nonzero_per_object"] for r in rows)
    all_distinct = all(r["distinct_words"] == k for r in rows)
    all_gabor_frozen = all(r["gabor_frozen_after_teach"] for r in rows)
    NAME_GO = 0.80

    print("\n  attribution of the retinal-percept naming accuracy above the untrained random-map control:", flush=True)
    frac = attributable_to("retina-percept naming decode accuracy", mean_name, mean_chance)

    v = Verdict("grounded RETINA->V1 percept driving the naming map (fixed codes replaced by real vision)",
                chance=chance)
    v.require("backend is numpy (real spiking Izhikevich)", os.environ.get("SIM_BACKEND") == "numpy", expect=True)
    v.require("naming input is V1 firing driven by the retina (not a host code)", all_v1_fires and all_v1_nonzero,
              expect=True)
    v.disabled("carrier-frame phrasing + simple-bar object render",
               "host 'the <agent> <verb> ___' skeleton and oriented-bar object images are NAMED residuals; only "
               "the REFERENT word is brain-selected, from the real V1 response")
    v.disabled("structural plasticity / threshold homeostasis / synaptic scaling",
               "frozen around the naming map so the learned pathway is stable across the long inference battery")
    v.disabled("full spiking request/silence gate operating point",
               "the request/silence race is on-bridge/spiking; here it ROUTES whether the naming circuit is engaged")
    v.require("naming accuracy (held-out jittered percept) >= %.2f" % NAME_GO, mean_name, expect=lambda x: x >= NAME_GO)
    v.floor("naming accuracy above chance", mean_name, floor=chance)
    v.control("naming vs untrained random map (learned, not substrate bias)", mean_name, mean_chance,
              min_separation=0.30)
    v.require("V1 codes are object-DISCRIMINABLE (low cross-object cosine)", mean_xcos, expect=lambda x: x < 0.9)
    v.require("each object names a DISTINCT word (all seeds)", all_distinct, expect=True)
    v.require("FIXED Gabor retina->V1 weights unchanged by teaching (only V1->word learns)", all_gabor_frozen,
              expect=True)
    v.reaches("lesion collapses the decode toward chance", before=mean_name, after=mean_lesion)
    v.require("lesion never emits a confident decode (fails safe)", all_lesion_conf0, expect=True)
    v.require("render articulates the spike-decoded word == 1", mean_render, expect=lambda x: x == 1.0)
    v.require("spiking gate: hungry routes to request all seeds", all_hungry_req, expect=True)
    v.require("spiking gate: sated routes to silence -> zero word output all seeds", all_silent, expect=True)
    v.require("permutation followed >= %.2f" % NAME_GO, mean_perm, expect=lambda x: x >= NAME_GO)
    v.control("permuted map rejects the original word", mean_perm, mean_orig, min_separation=0.30)
    v.require("novel untaught object abstains (below confidence) all seeds", all_novel, expect=True)

    go = (all_v1_fires and all_v1_nonzero and all_distinct and all_gabor_frozen
          and mean_name >= NAME_GO and mean_name > chance and (mean_name - mean_chance) > 0.30
          and mean_xcos < 0.9 and mean_lesion < mean_name and all_lesion_conf0 and mean_render == 1.0
          and all_hungry_req and all_silent and mean_perm >= NAME_GO and (mean_perm - mean_orig) > 0.30
          and all_novel)
    decided = v.decide(go=go)

    out = {
        "verdict": decided,
        "mean_name_acc": mean_name, "mean_name_acc_chance": mean_chance, "mean_lesion_acc": mean_lesion,
        "mean_render_faithful": mean_render, "mean_cross_object_v1_cosine": mean_xcos,
        "naming_accuracy_attributable_fraction": frac,
        "all_v1_fires_every_trial": all_v1_fires, "all_v1_nonzero_per_object": all_v1_nonzero,
        "all_objects_name_distinct_words": all_distinct, "all_gabor_frozen_after_teach": all_gabor_frozen,
        "all_hungry_request": all_hungry_req, "all_sated_silence_zero_output": all_silent,
        "all_lesion_zero_confident": all_lesion_conf0,
        "mean_perm_followed": mean_perm, "mean_orig_accepted_after_perm": mean_orig,
        "all_novel_abstains": all_novel,
        "chance": chance, "name_go_threshold": NAME_GO, "smoke": bool(a.smoke),
        "backend": os.environ.get("SIM_BACKEND"),
        "conf_margin_spk": CONF_MARGIN_SPK, "min_word_spikes": MIN_WORD_SPIKES,
        "n_v1": N_V1, "n_retina": N_RETINA, "word_nper": a.word_nper,
        "n_ex": (a.smoke_ex if a.smoke else a.n_ex), "hold": a.hold, "n_referents": k,
        "ckpt": a.ckpt, "seeds": seeds,
        "config": {kk: getattr(a, kk) for kk in ("gabor_scale", "hebb_rate", "hebb_max", "teach_epochs",
                   "teach_steps", "retina_drive", "teach_drive", "decode_steps", "washout_steps")},
        "per_seed": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n  wrote %s  (%.1fs)" % (a.out, time.time() - t0), flush=True)
    print("  => %s" % decided["status"], flush=True)
    return 0 if decided["status"] == GO else 1


if __name__ == "__main__":
    raise SystemExit(main())
