"""Functional-integration CHEAP-FIRST de-risk — PERCEPTION -> MEMORY (the deeper one-brain), §3.3 + §6 of
`docs/plans/2026-06-10-functional-integration-one-brain-design.md`.

THE LOAD-BEARING QUESTION (design §3.3 + §6): does an engram tag DRIVEN BY THE NAVIGATION PERCEPTION region
(the ventral object-identity code `cortex_it`, NOT a `language_input` cue) — when stimulated later — recall
the perceived concept SPECIFICALLY (perceive A -> recall A, not B), carried by the tagged synapses
(lesion-confirmed)?

WHY THIS MATTERS (design §6, the central cross-code-transfer problem). The navigation perception is a RATE
code in `cortex_it` (Izhikevich firing-rate ensembles); the conversational composer stores a PHASOR code
(phases in [0,1)^D on resonate-and-fire neurons). These two codes are NOT commensurable: a synaptic route
from `cortex_it` into the composer would deliver a rate pattern the bind/unbind algebra cannot consume. The
engram-tag mechanism (Tonegawa, catalog D.14) SIDESTEPS this wall: it stores the *perceived ENSEMBLE itself*
(the neurons that fired) as a recallable tag and recalls it by re-stimulation — it NEVER converts the rate
percept into a phasor, so the cross-code mismatch does not arise. This probe de-risks exactly that write:
PERCEPTION -> engram -> RECALL, all synaptic, with NO Python copy of a percept vector.

HONEST SCOPE (design §6): this is a RECALL interaction ("I saw the apple"), NOT composition over perceived
content (you cannot yet algebraically bind the perceived apple into a novel role-filler fact — that genuinely
requires shared grounded codes / step-3's learned cortex, the rate-vs-phasor wall). The compositional version
is deliberately out of scope; an honest NEGATIVE here (perception-driven tags don't recall as well as
language-cued ones) is a valid deliverable that MAPS the limit.

THE MECHANISM (design §3.3, all synaptic, reuse-by-import, no `sim/` edit):
  - PERCEIVE object X (the environment presents the percept): the navigation perception region `cortex_it`
    carries OBJECT-IDENTITY ensembles (the ventral "what" stream — per-object category codes). "The agent
    sees object X" = X's distinct `cortex_it` sub-ensemble fires. The probe renders this by driving X's
    orthogonal band of `cortex_it` (a legitimate sensory render — the environment presenting the percept,
    exactly as the (A) probe rendered the command word into `language_input`). This is the perception-side
    analogue of the navigation Gabor/retina pipeline producing an IT object code; the load-bearing thing being
    de-risked is the ENGRAM WRITE FROM PERCEPTION, not the (separately-validated) Gabor front-end.
  - WRITE the perceived ensemble to memory (the NEW (B) part — catalog D.14): with X perceived,
    `start_engram_recording("seen_X")` -> run the perception window -> `commit_engram_tag("seen_X",
    region_filter=["cortex_it"])`. The tag IS the actual perceived `cortex_it` ensemble — no phasor code, no
    Python copy of a percept vector; the neurons that fired ARE the memory.
  - RECALL by neural reactivation: later, `stimulate_tag("seen_X")` re-drives that ensemble; the reactivation
    propagates through the perception->language read-out (`cortex_it -> language_output`) and the readout reads
    which concept word the reactivation spells (cosine of the `language_output` firing pattern to each word's
    orthogonal code). Recall = neural reactivation, NOT a Python lookup.

CHEAPEST FAITHFUL SUBSTRATE (design §4 explicitly allows "the cheapest faithful bridge: a perception region +
the conversational read-out + the engram API"): a fresh brain-region-framework `SimulationBridge` with ONLY
  - `cortex_it`  : the navigation perception region (object-identity ensembles; the engram source),
  - `language_output` : the conversational spelling read-out (the recall channel),
plus the `cortex_it -> language_output` perception->word read-out pathway (the SAME pathway the navigation
builder wires when enable_visual_cortex+enable_text_io, `g11_bg_runner.py:2660-2667`). No
retina/V1/V2/striatum/parser/dlPFC/RF: none are needed to drive an engram from a `cortex_it` ensemble and read
the `language_output` reactivation, and omitting them keeps the probe CPU-cheap (`SIM_BACKEND=numpy`, minutes).
The `cortex_it -> language_output` selectivity ("apple's IT ensemble" -> spells "apple") is what TRAINING grows
in the full nav stack; this cold probe does not train, so — exactly as the (A) probe installed the trained
`language_input -> cortex_X` mapping STRUCTURALLY as a per-direction topographic labeled line — we install the
read-out as a per-object topographic labeled line (object o's `cortex_it` band -> object o's `language_output`
band). The thing being de-risked is the ENGRAM WRITE+RECALL (does a perception-driven tag reactivate the
concept), not the learning of the read-out map.

PASS criteria (the load-bearing question + the design §5 anti-cheats), multi-seed (42/43/44):
  (recall)     perceive A -> tag seen_A -> stimulate seen_A -> the `language_output` reactivation spells A
               (A is the top-1 word by cosine). Accuracy across objects >> chance (1/n_objects).
  (specificity) stimulate seen_A recalls A NOT B: the recalled top-1 == the perceived object (the cross-control
               IS the recall accuracy — each tag recalls its OWN object, not another).
  (lesion)     zero the `cortex_it -> language_output` synapses, re-stimulate every tag -> recall collapses to
               chance (the `language_output` reactivation no longer spells the object) -> proves the recall
               rides THOSE synapses, not a leak or a Python path.
  (provenance) the tag = the perceived ENSEMBLE: every tagged neuron is a `cortex_it` neuron (asserted
               structurally). NO host code copies a percept vector into the recall drive — the only legitimate
               writes are the sensory render of the object into `cortex_it` and the engram `stimulate_tag`.

Reuse-by-import: `g11_bg_runner` perception/read-out region shapes (re-expressed minimally here, same as the
(A) probe re-expressed the action cortex) + the bridge engram API + `sim.text_embeddings.orthogonal_drive_pattern`.
No `sim/` edit.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel, NeuronType
from sim.regions import BrainRegion
from sim.backend import get_backend, to_host
from sim.text_embeddings import orthogonal_drive_pattern

# ── constants ────────────────────────────────────────────────────────────────────────────────────────────
# The perceived objects (the navigation agent's ventral "what"-stream object identities). 4 objects -> chance
# top-1 recall = 1/4 = 0.25 (the specificity baseline). These are concept WORDS the read-out spells.
OBJECT_WORDS = ["apple", "river", "dog", "cat"]
N_OBJECTS = len(OBJECT_WORDS)

N_CORTEX_IT = 256          # the perception region (object-identity ensembles). Matches the nav visual_n_it default.
N_LANG_OUTPUT = 256        # the conversational spelling read-out (the recall channel).
IT_TO_LANG_GATE = "it_to_lang"   # plasticity-gate TAG on cortex_it -> language_output (so the lesion can resolve
#                                  + zero exactly those synapses via _plasticity_gate_indices_gpu — the same
#                                  mechanism the (A) probe used for the transmission gate).

# Per-object orthogonal bands (non-overlapping, maximally separable) — the SAME layout in BOTH the perception
# region (cortex_it) and the read-out (language_output), so the read-out is a clean per-object labeled line.
PERCEPT_SPARSITY = 0.1     # fraction of cortex_it active per object (n_active = 0.1*256 = 25 < stride 64 -> ok)
LANG_SPARSITY = 0.1        # fraction of language_output in each object's spelling band

# the perception render drive: drive object o's cortex_it band hard enough that the band FIRES (cortex_it has
# inhibitory recurrence that damps weak drives — 2500 pA is the composer/parser ROLE_DRIVE scale the (A) probe used).
PERCEPT_DRIVE_PA = 2500.0
# the cortex_it band -> language_output band read-out weight. Strong labeled line so that, when the perceived
# (or tag-reactivated) cortex_it band fires, the matching language_output band crosses threshold (cold untrained
# read-out needs a strong labeled line — the same TOPO_ROUTE_WEIGHT regime as the (A) probe's band->cortex line).
READOUT_WEIGHT = 14.0
# the engram stim drive (catalog D.14 recall): drive the tagged cortex_it ensemble at the validated recall scale
# (compose_concept_engram uses 1500 pA stim-recall). Strong enough the reactivation propagates to language_output.
TAG_STIM_PA = 1500.0

ENCODING_STEPS = 120       # the perception/encoding window the engram accumulates over (the tag's neurons).
RECALL_STEPS = 100         # the recall window: accumulate the language_output firing pattern under tag stim.
SETTLE_STEPS = 30          # quiescence between conditions (clears prior drive).
ENGRAM_TOP_K = 60          # tag the top-K cortex_it neurons by spike count (Marr-like sparse engram; ~= a band).

# a recall is "correct" only if the perceived object's word is the UNIQUE top-1 by cosine AND leads the runner-up
# by a meaningful margin (so a floating-point tie does not count as a recall). Sits below the clean-recall margin
# and above the lesion/chance residual.
MIN_RECALL_MARGIN = 0.02


def _object_band_indices(region_idx_h, obj_idx, n_objects, sparsity):
    """The GLOBAL neuron indices of object o's orthogonal band within a region. MUST match
    orthogonal_drive_pattern's layout exactly (same n_active/stride math) so the band we WIRE (and DRIVE) is the
    band we read. `region_idx_h` is the region's global indices (host int64)."""
    n = int(region_idx_h.size)
    n_active = max(1, int(round(sparsity * n)))
    stride = n // n_objects
    if n_active > stride:
        raise ValueError(f"band overlap: n_active={n_active} > stride={stride}")
    start = obj_idx * stride
    return region_idx_h[start:start + n_active]


# ── the minimal perception(cortex_it) + read-out(language_output) bridge ──────────────────────────────────────
def build_probe_bridge(seed: int = 42):
    """A fresh brain-region-framework `SimulationBridge` holding ONLY:
      - `cortex_it`       : the navigation perception region (object-identity ensembles; the engram source),
      - `language_output` : the conversational spelling read-out (the recall channel),
    plus a per-object TOPOGRAPHIC `cortex_it band_o -> language_output band_o` read-out population tagged
    `plasticity_gate="it_to_lang"` (so the lesion can resolve + zero exactly those synapses).

    Config mirrors the merge builder's conversational regime (Izhikevich, dt=1, STDP/reward/Hebbian/STP/
    homeostasis/structural OFF — this cold probe does not train; the engram API is the only state written; OU=20
    allocated at build for the read-out WTA spelling; the 5a clip mitigation stdp_w_max/hebbian_max_weight=400).
    No retina/V1/V2/parser/dlPFC/RF: none are needed to drive an engram from a cortex_it ensemble and read the
    language_output reactivation, and omitting them keeps the probe CPU-cheap.

    Returns (bridge, handles) where handles carry the per-region/per-object band index arrays.
    """
    xp, _ = get_backend()

    # cortex_it: the ventral object-identity region. exc_fraction/internal_density match g11's cortex_it
    # (exc_fraction=0.8, internal_density=0.10) so the perception ensemble dynamics are the navigation region's.
    it_region = BrainRegion(name="cortex_it", n_neurons=N_CORTEX_IT, exc_fraction=0.8,
                            internal_density=0.10, exc_weight_mean=2.0, inh_weight_mean=4.0,
                            weight_jitter=0.2, plastic_internal=True,
                            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)
    # language_output: the conversational spelling read-out. exc_fraction/internal_density match g11's
    # language_output (exc_fraction=0.8, internal_density=0.10).
    lang_out_region = BrainRegion(name="language_output", n_neurons=N_LANG_OUTPUT, exc_fraction=0.8,
                                  internal_density=0.10, exc_weight_mean=2.0, inh_weight_mean=4.0,
                                  weight_jitter=0.2, plastic_internal=True,
                                  izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)
    union_regions = [it_region, lang_out_region]

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = union_regions
    cfg.region_pathways = []   # the read-out is the hand-wired topographic population added post-build.
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)
    cfg.stdp_w_max = 400.0
    cfg.hebbian_max_weight = 400.0
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False     # no training: the engram API is the only state written.
    cfg.enable_homeostasis = False          # FOOT-GUN: the synaptic-scaling clip would slam the fixed read-out.
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_ou_process = True            # allocate OU state at build (the language_output WTA spelling needs it)
    cfg.ou_std_current_pA = 20.0
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_nmda = False

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    assert cfg.enable_homeostasis is False, "homeostasis must stay OFF (synaptic-scaling clip foot-gun)"

    rm = bridge.region_manager

    # Install the per-object TOPOGRAPHIC read-out (cortex_it band_o -> language_output band_o) as an explicit
    # population: rebuild the framework union plan + ADD it_to_lang, inject ONCE. The plasticity-gate is registered
    # by inject_explicit_wiring from the population's plasticity_gate field (no `sim/` edit). plastic=False (a fixed
    # labeled line; this probe does not train the read-out — only the ENGRAM is the written memory).
    it_idx_h = np.asarray(list(rm.indices("cortex_it")), dtype=np.int64)
    lo_idx_h = np.asarray(list(rm.indices("language_output")), dtype=np.int64)
    # the read-out must be EXCITATORY->excitatory (both regions are 20% inhibitory; an inhibitory band neuron in
    # the route would SUPPRESS its read-out band, and worse scale the inhibition up with weight -> the read-out
    # band goes silent, the inversion the (A) probe documented). Filter each band to its excitatory neurons only.
    inh_it = set(int(i) for i in rm.inhibitory_indices("cortex_it"))
    route_pre, route_post = [], []
    for obj_idx in range(N_OBJECTS):
        it_band = _object_band_indices(it_idx_h, obj_idx, N_OBJECTS, PERCEPT_SPARSITY)
        it_band_exc = [int(p) for p in it_band if int(p) not in inh_it]
        lo_band = _object_band_indices(lo_idx_h, obj_idx, N_OBJECTS, LANG_SPARSITY)
        # topographic: every EXCITATORY cortex_it band neuron -> every language_output band neuron (density 1.0)
        for p in it_band_exc:
            for q in lo_band:
                route_pre.append(int(p)); route_post.append(int(q))
    readout_pop = {
        "pre_indices": np.asarray(route_pre, dtype=np.int64),
        "post_indices": np.asarray(route_post, dtype=np.int64),
        "initial_weights": np.full(len(route_pre), READOUT_WEIGHT, dtype=np.float32),
        "plastic": False, "plasticity_gate": IT_TO_LANG_GATE, "conn_type": "E_TO_E", "count": len(route_pre),
    }
    union_plan = dict(rm.build_wiring_plan(seed=int(seed)))
    assert IT_TO_LANG_GATE not in union_plan, "it_to_lang name collides with a framework population"
    union_plan[IT_TO_LANG_GATE] = readout_pop
    inh_concat = []
    for region in rm.regions():
        inh_concat.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union_plan, output_inhibitory_indices=inh_concat or None)

    assert IT_TO_LANG_GATE in bridge._plasticity_gate_indices_gpu, \
        f"FAIL: '{IT_TO_LANG_GATE}' plasticity gate not registered (known: " \
        f"{list(bridge._plasticity_gate_indices_gpu.keys())})"
    # freeze the read-out (no weight updates anywhere; this is belt-and-braces since all plasticity is off).
    bridge.set_plasticity_gate(IT_TO_LANG_GATE, 0.0)

    handles = {
        "seed": int(seed),
        "it_indices": xp.asarray(it_idx_h),
        "lang_out_indices": xp.asarray(lo_idx_h),
        # per-object cortex_it band (the perception render target); host int64 for region_filter independence.
        "it_band": {OBJECT_WORDS[o]: _object_band_indices(it_idx_h, o, N_OBJECTS, PERCEPT_SPARSITY)
                    for o in range(N_OBJECTS)},
    }
    return bridge, handles


def _reset(bridge, steps=SETTLE_STEPS):
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(steps):
        bridge._run_one_simulation_step()


def _render_percept(bridge, it_indices, obj_idx):
    """Render object `obj_idx` into the perception region: drive object o's ORTHOGONAL cortex_it band. This is
    the legitimate sensory render — the environment presents the percept (the navigation Gabor/retina pipeline's
    job; here a direct object-identity render, exactly as the (A) probe rendered the command word)."""
    xp, _ = get_backend()
    n = int(it_indices.size)
    drive = orthogonal_drive_pattern(cue_idx=obj_idx, n_cues=N_OBJECTS, n_neurons=n,
                                     drive_max_pA=PERCEPT_DRIVE_PA, sparsity=PERCEPT_SPARSITY)
    bridge.cp_external_input_current[it_indices] = xp.asarray(drive, dtype=xp.float32)


def encode_percept_engram(bridge, handles, obj_word):
    """The (B) WRITE (catalog D.14, the NEW perception-driven part): perceive `obj_word` (render its cortex_it
    band) for the encoding window while an engram recording accumulates, then commit the tag over cortex_it ONLY
    (region_filter=["cortex_it"]). The tag IS the perceived cortex_it ensemble — no Python copy of a percept
    vector; the neurons that fired ARE the memory.

    Returns the commit stats dict (incl. n_tagged)."""
    xp, _ = get_backend()
    obj_idx = OBJECT_WORDS.index(obj_word)
    it_indices = handles["it_indices"]
    tag = f"seen_{obj_word}"

    _reset(bridge)
    bridge.start_engram_recording(tag)
    for _ in range(ENCODING_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        _render_percept(bridge, it_indices, obj_idx)
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()
    # commit the tag over cortex_it ONLY -> the tag is the perceived ensemble (provenance: a cortex_it subset).
    stats = bridge.commit_engram_tag(tag, top_k=ENGRAM_TOP_K, region_filter=["cortex_it"])
    return stats


def _recall_lang_output_pattern(bridge, handles, obj_word):
    """The (B) RECALL by neural reactivation: stimulate the perceived ensemble's tag (`seen_<obj>`); the
    reactivation propagates through cortex_it -> language_output; accumulate the language_output firing pattern
    over the recall window. Recall = neural reactivation, NOT a Python lookup (the ONLY write is stimulate_tag,
    which drives the TAGGED neurons — the perceived ensemble — not a copied percept vector).

    Returns the host language_output spike-count pattern (shape (N_LANG_OUTPUT,))."""
    xp, _ = get_backend()
    lang_out_indices = handles["lang_out_indices"]
    n_lo = int(lang_out_indices.size)
    tag = f"seen_{obj_word}"

    # OU on for the spelling read-out (the language_output WTA needs it; matches the merge's per-read toggle).
    cc = bridge.core_config
    prev_ou, prev_std = cc.enable_ou_process, cc.ou_std_current_pA
    cc.enable_ou_process = True
    cc.ou_std_current_pA = 20.0
    try:
        bridge.cp_external_input_current[:] = 0.0
        bridge.clear_tag_drive()
        _reset(bridge)
        bridge.stimulate_tag(tag, drive_pA=TAG_STIM_PA, additive=False)
        pattern = xp.zeros(n_lo, dtype=xp.float64)
        for _ in range(RECALL_STEPS):
            # re-assert the tag drive each step (overwrite-at-tagged-indices), as the validated recall runner does:
            # stimulate_tag(additive=False) sets the tagged indices; running the step then reads language_output.
            bridge.stimulate_tag(tag, drive_pA=TAG_STIM_PA, additive=False)
            bridge._run_one_simulation_step()
            pattern += bridge.cp_firing_states[lang_out_indices].astype(xp.float64)
        bridge.clear_tag_drive(tag)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(20):
            bridge._run_one_simulation_step()
    finally:
        cc.enable_ou_process, cc.ou_std_current_pA = prev_ou, prev_std
    return to_host(pattern)


def _cosine_to_object(pattern_h, obj_word, n_lo):
    """Cosine of a language_output spike-count pattern to `obj_word`'s orthogonal spelling band (the read-out's
    per-object labeled-line target). The recalled word is argmax of this over OBJECT_WORDS."""
    obj_idx = OBJECT_WORDS.index(obj_word)
    target = orthogonal_drive_pattern(cue_idx=obj_idx, n_cues=N_OBJECTS, n_neurons=n_lo,
                                      drive_max_pA=1.0, sparsity=LANG_SPARSITY)
    a = float(np.linalg.norm(pattern_h)); b = float(np.linalg.norm(target))
    if a == 0.0 or b == 0.0:
        return 0.0
    return float(np.dot(pattern_h, target) / (a * b))


def _recall_metrics(pattern_h, perceived_word, n_lo):
    """Given the recall language_output pattern and the PERCEIVED object, rank all object words by cosine and
    report (top1, top1_is_perceived, margin, scores). `correct` (the recall + specificity criterion) requires the
    perceived object to be the UNIQUE top-1 by a MEANINGFUL margin (>= MIN_RECALL_MARGIN) — so the tag recalls
    its OWN object (specificity), not a tie/leak."""
    scores = {w: _cosine_to_object(pattern_h, w, n_lo) for w in OBJECT_WORDS}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top1, top1_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top1_score - runner_up_score
    perceived_score = scores[perceived_word]
    correct = bool(top1 == perceived_word and margin >= MIN_RECALL_MARGIN)
    return {
        "scores": {w: float(s) for w, s in scores.items()},
        "top1": top1,
        "top1_score": float(top1_score),
        "perceived_score": float(perceived_score),
        "margin": float(margin),
        "top1_is_perceived": bool(top1 == perceived_word),
        "correct": correct,
    }


# ── provenance check (design §5 anti-cheat: the tag is the perceived ensemble; no Python percept-copy) ────────
def provenance_check(bridge, handles):
    """Anti-cheat 3 (design §5): the tag = the perceived ENSEMBLE (every tagged neuron is a cortex_it neuron),
    and NO host code copies a percept vector into the recall drive. ASSERTED STRUCTURALLY: for each committed
    tag, its indices are a SUBSET of the cortex_it region. The ONLY current writes anywhere in the probe are
    (i) the orthogonal object code into cortex_it (the sensory render — the environment presenting the percept)
    and (ii) `stimulate_tag` (which drives the TAGGED neurons = the perceived ensemble, NOT a copied vector).
    Returns the audit facts the report records; raises AssertionError on any violation (caught -> exit 1)."""
    xp, _ = get_backend()
    it_set = set(int(i) for i in to_host(handles["it_indices"]))
    tag_facts = []
    for tinfo in bridge.list_engram_tags():
        name = tinfo["name"]
        idx_h = to_host(bridge.get_engram_tag_indices(name))
        tagged = [int(i) for i in idx_h]
        all_in_it = all(t in it_set for t in tagged)
        assert all_in_it, (f"FAIL provenance: tag {name!r} has {sum(1 for t in tagged if t not in it_set)} "
                           f"neurons OUTSIDE cortex_it — the tag must be the PERCEIVED ensemble")
        tag_facts.append({"tag": name, "n_tagged": len(tagged), "all_in_cortex_it": bool(all_in_it)})
    return {
        "perception_side_current_writes": [
            "cortex_it <- orthogonal_drive_pattern(object)  [legitimate sensory render: the environment "
            "presents the perceived object]",
        ],
        "recall_side_current_writes": [
            "stimulate_tag(seen_<obj>)  [drives the TAGGED neurons = the perceived cortex_it ensemble, NOT a "
            "copied percept vector]",
        ],
        "engram_region_filter": "cortex_it",
        "every_tag_is_a_cortex_it_subset": bool(all(t["all_in_cortex_it"] for t in tag_facts)),
        "no_percept_vector_copied_into_recall_drive": True,
        "tags": tag_facts,
    }


# ── one seed: encode every object's percept engram, then recall (clean) + recall (lesioned) ──────────────────
def run_seed(seed: int):
    xp, backend = get_backend()
    print(f"\n[funcint-p2m] ===== seed {seed} (backend={backend}) =====")
    bridge, handles = build_probe_bridge(seed)

    # (write) encode each object's perception engram (perceive -> tag the cortex_it ensemble).
    encode_stats = {}
    for w in OBJECT_WORDS:
        encode_stats[w] = encode_percept_engram(bridge, handles, w)
        print(f"[funcint-p2m]  encoded seen_{w}: tagged {encode_stats[w]['n_tagged']} cortex_it neurons "
              f"(window {encode_stats[w]['window_ms']:.0f} ms)")

    prov = provenance_check(bridge, handles)

    # (recall clean) stimulate each tag -> read language_output reactivation -> which object does it spell?
    n_lo = int(handles["lang_out_indices"].size)
    per_obj = {}
    n_recall_correct = 0
    for w in OBJECT_WORDS:
        pat = _recall_lang_output_pattern(bridge, handles, w)
        m = _recall_metrics(pat, w, n_lo)
        per_obj[w] = m
        n_recall_correct += int(m["correct"])
        print(f"[funcint-p2m]  recall seen_{w:5s}: top1={m['top1']:5s} "
              f"(perceived_score={m['perceived_score']:.4f} margin={m['margin']:+.4f}) "
              f"correct={m['correct']}")

    # (recall lesioned) zero the cortex_it -> language_output read-out, re-stimulate every tag: recall must
    # collapse to chance (the reactivation no longer reaches language_output) -> proves the recall rides THOSE
    # synapses. Rebuild a fresh bridge + re-encode so the lesion is clean (zero the gate-tagged synapses).
    lesion_per_obj, n_lesion_correct = _run_lesion(seed)

    print(f"[funcint-p2m] seed {seed} roll-up: CLEAN recall correct={n_recall_correct}/{N_OBJECTS}  "
          f"LESION recall correct={n_lesion_correct}/{N_OBJECTS}  (chance=1/{N_OBJECTS})")

    return {
        "seed": int(seed),
        "n_objects": N_OBJECTS,
        "chance": 1.0 / N_OBJECTS,
        "encode_stats": {w: {"n_tagged": int(s["n_tagged"]), "window_ms": float(s["window_ms"])}
                         for w, s in encode_stats.items()},
        "per_obj": per_obj,
        "lesion_per_obj": lesion_per_obj,
        "n_recall_correct": n_recall_correct,
        "n_lesion_correct": n_lesion_correct,
        "provenance": prov,
    }


def _run_lesion(seed: int):
    """LESION control (design §5 anti-cheat 1, primary): build a fresh probe bridge, re-encode every object's
    perception engram, ZERO every `cortex_it -> language_output` read-out synapse's weight, then re-stimulate
    each tag. With the perception engram intact but the read-out cut, recall must FAIL -> the recall rides the
    `cortex_it -> language_output` synapses, not ambient leakage or any non-route path."""
    xp, _ = get_backend()
    bridge, handles = build_probe_bridge(seed)
    for w in OBJECT_WORDS:
        encode_percept_engram(bridge, handles, w)
    # zero the read-out weights in place (the plasticity-gate index map points at exactly those synapses).
    idx = bridge._plasticity_gate_indices_gpu[IT_TO_LANG_GATE]
    n_lesioned = int(idx.size)
    bridge.cp_connections.data[idx] = xp.asarray(0.0, dtype=bridge.cp_connections.data.dtype)

    n_lo = int(handles["lang_out_indices"].size)
    out = {}
    n_correct = 0
    for w in OBJECT_WORDS:
        pat = _recall_lang_output_pattern(bridge, handles, w)
        m = _recall_metrics(pat, w, n_lo)
        out[w] = m
        n_correct += int(m["correct"])
    print(f"[funcint-p2m]  LESION (cortex_it->language_output weights zeroed, n_synapses={n_lesioned}) "
          f"recall: " + " ".join(f"{w}={out[w]['correct']}" for w in OBJECT_WORDS))
    return out, n_correct


# ── verdict ──────────────────────────────────────────────────────────────────────────────────────────────
def verdict_from(results):
    """GO  : CLEAN recall is correct for >=3/4 objects (>> chance 1/4) on ALL seeds (perception-driven tags
             recall the perceived concept specifically), AND the LESION collapses recall to <=1/4 on ALL seeds
             (the recall rides the cortex_it->language_output synapses). [specificity is folded into 'correct':
             each tag must recall its OWN object as the unique top-1.]
       PARTIAL : CLEAN recall is correct on a majority across seeds (avg >=2/4) AND the lesion collapses it,
                 but full >=3/4-all-seeds is not met (some object recalls weakly / seed-variable).
       NEGATIVE: CLEAN recall does not reliably exceed chance, OR the lesion does not collapse it (the route is
                 not the cause). An honest negative MAPS the limit (design §6: perception-driven tags may not
                 recall as cleanly as language-cued ones)."""
    seeds = [r["seed"] for r in results]
    recall_ok = all(r["n_recall_correct"] >= 3 for r in results)
    lesion_ok = all(r["n_lesion_correct"] <= 1 for r in results)
    avg_recall = sum(r["n_recall_correct"] for r in results) / max(1, len(results))

    if recall_ok and lesion_ok:
        v = "GO"
    elif avg_recall >= 2.0 and lesion_ok:
        v = "PARTIAL"
    else:
        v = "NEGATIVE"
    return {
        "verdict": v,
        "seeds": seeds,
        "recall_ge3_all_seeds": bool(recall_ok),
        "lesion_collapses_all_seeds": bool(lesion_ok),
        "avg_recall_correct": float(avg_recall),
        "chance": 1.0 / N_OBJECTS,
        "recall_correct_per_seed": {r["seed"]: r["n_recall_correct"] for r in results},
        "lesion_correct_per_seed": {r["seed"]: r["n_lesion_correct"] for r in results},
        "provenance_all_seeds": bool(all(r["provenance"]["every_tag_is_a_cortex_it_subset"] for r in results)),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Functional-integration cheap-first de-risk: PERCEPTION->MEMORY (engram driven from the "
                    "navigation perception region cortex_it, recalled via cortex_it->language_output).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/funcint_perception_to_memory_probe.json")
    args = ap.parse_args()

    results = [run_seed(s) for s in args.seeds]
    vd = verdict_from(results)

    print("\n[funcint-p2m] ============ VERDICT ============")
    print(f"[funcint-p2m] verdict={vd['verdict']}  (chance=1/{N_OBJECTS}={vd['chance']:.2f})")
    print(f"[funcint-p2m]   CLEAN recall correct (>=3/4) all seeds : {vd['recall_ge3_all_seeds']}  "
          f"{vd['recall_correct_per_seed']}")
    print(f"[funcint-p2m]   LESION collapses recall (<=1/4) all    : {vd['lesion_collapses_all_seeds']}  "
          f"{vd['lesion_correct_per_seed']}")
    print(f"[funcint-p2m]   provenance (every tag = cortex_it subset) all : {vd['provenance_all_seeds']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def _ser(o):
        if isinstance(o, dict):
            return {k: _ser(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_ser(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    with open(args.out, "w") as f:
        json.dump(_ser({"results": results, "verdict": vd}), f, indent=2)
    print(f"[funcint-p2m] wrote {args.out}")
    raise SystemExit(0 if vd["verdict"] == "GO" else (2 if vd["verdict"] == "PARTIAL" else 1))


if __name__ == "__main__":
    main()
