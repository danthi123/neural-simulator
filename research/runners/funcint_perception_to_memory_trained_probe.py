"""Functional-integration CHEAP-FIRST de-risk — PERCEPTION -> MEMORY with a **TRAINED (noisy) read-out**.

This is the pre-registered NEXT STEP after `funcint_perception_to_memory_probe.py` went GO (2026-06-16). That
confirmed probe established the perception -> engram -> recall loop, BUT used a CLEAN per-object TOPOGRAPHIC
labeled-line read-out (fixed weights, band_o -> band_o), so the recall cosines were ~1.0/0.0 — noise-free. Its
HONEST SCOPE explicitly flagged the **TRAINED (Hebbian-grown, NOISY) read-out** as the load-bearing next test.

THE SINGLE LOAD-BEARING QUESTION (this probe):
  Does the perception -> engram -> recall loop SURVIVE a TRAINED `cortex_it -> language_output` read-out instead
  of the clean labeled-line stand-in? The read-out is now LEARNED by Hebbian co-firing (the project's
  Pulvermuller / b3 / concept-pool / bio_three_factor embodied co-firing idiom), so it is LOSSY/NOISY — recall
  correctness becomes a genuine signal-above-chance, NOT ~1.0.

WHAT CHANGES vs the confirmed probe (everything else is reused VERBATIM in spirit):
  - The `cortex_it -> language_output` route is no longer a fixed per-object band->band labeled line. It is wired
    DENSE (every excitatory cortex_it neuron -> every excitatory language_output neuron) at a LOW initial weight,
    `plastic=True`, tagged `plasticity_gate="it_to_lang"`. The per-object SELECTIVITY is therefore something the
    network must LEARN, not something we wired in. (A band->band wiring + training would just re-grow the same
    labeled line; a DENSE plastic route is the honest "trained map" — the cross-object synapses are present and
    only stay weak because they never co-fire.)
  - A TRAINING PASS (the NEW part): for each object o, drive object o's `cortex_it` band (the perceived ensemble,
    presynaptic) AND object o's `language_output` band (the teacher, postsynaptic) TOGETHER for a training
    window, so the route GROWS o->o selectivity via the bridge's soft-bound Hebbian co-firing rule
    (`delta_w = lr*(w_max - w)` for synapses whose pre AND post both fire that step,
    `bridge.py:6476-6510`). Trials are INTERLEAVED (shuffled across objects, like the concept-pool recipe) to
    avoid last-object-wins order effects. Training is RESTRICTED to the route: `cp_plasticity_rate_gain` is set
    to 0 everywhere and 1 only on the `it_to_lang` synapses, so the internal region recurrence is NOT perturbed
    by the training co-firing (the Hebbian update multiplies delta_w by this per-synapse gain,
    `bridge.py:6490-6491`).
  - After training, plasticity is FROZEN (`set_plasticity_gate("it_to_lang", 0.0)` + `enable_hebbian_learning`
    flipped off) and the loop is tested EXACTLY as the confirmed probe does.

THE TRAINING DRIVE IS A LEARNING SIGNAL, NOT A RECALL-TIME PERCEPT COPY (anti-cheat 3 / provenance). During the
TRAIN phase, driving the language_output teacher band is the Pulvermuller embodied co-firing learning signal
(the same "elevate the motor/output target so the sites co-fire" pattern bio_three_factor uses). It is in a
SEPARATE phase from recall. At RECALL time the ONLY write is `stimulate_tag` (driving the perceived cortex_it
ensemble); `language_output` is NEVER driven at recall — the probe asserts this. So the recall genuinely rides
the TRAINED synapses (lesion-confirmed), not a copied percept vector.

ANTI-CHEATS (all carried from the confirmed probe — a "pass" without these is fake):
  1. LESION (primary): zero the TRAINED `cortex_it -> language_output` synapses -> recall must collapse to chance.
  2. SPECIFICITY: perceive A -> stimulate seen_A -> recall A NOT B (top-1 == perceived; vs chance 1/n_objects).
  3. PROVENANCE: the tag is a cortex_it subset (asserted); the recall's only write is `stimulate_tag` (no Python
     percept copy); the training co-firing drive is a LEARNING signal in a separate phase, asserted not present
     at recall.
  4. Multi-seed: >=3 (42/43/44); 6 if cheap.

GATE:
  GO       : trained-map recall correct >=3/4 objects on ALL seeds (>> chance 1/4) AND lesion collapses it on all.
  PARTIAL  : recall beats chance but is seed-variable/weak; lesion still collapses.
  NEGATIVE : the noisy trained map cannot carry the recall (an honest negative that MAPS the limit — the read-out
             needs more training / a cleaner code; localize WHY). A negative is a valid, useful deliverable.

HONEST SCOPE: this is still a RECALL interaction ("I saw the apple" -> later recall "apple"), NOT composition
over perceived content. The compositional version (algebraically binding the perceived apple into a novel fact)
needs shared grounded codes / the learned-cortex step-3 (the rate-vs-phasor wall) and is deliberately out of
scope. What this probe ADDS over the confirmed (B) GO: the read-out is no longer an idealized clean labeled line
but a TRAINED, lossy, noisier map — the next idealization peeled off.

Reuse-by-import from the confirmed probe: `OBJECT_WORDS`, `_object_band_indices`, `_render_percept`,
`encode_percept_engram`, `_recall_lang_output_pattern`, `_cosine_to_object`, `_recall_metrics`,
`provenance_check`, and the constants. NO `sim/` edit (the engram API + Hebbian co-firing + plasticity gate are
all reused as-is).
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

# Reuse the confirmed probe's vocabulary, band math, percept render, encode, recall, cosine, metrics, provenance.
from research.runners.funcint_perception_to_memory_probe import (
    OBJECT_WORDS, N_OBJECTS,
    N_CORTEX_IT, N_LANG_OUTPUT, IT_TO_LANG_GATE,
    PERCEPT_SPARSITY, LANG_SPARSITY, PERCEPT_DRIVE_PA, TAG_STIM_PA,
    ENGRAM_TOP_K, MIN_RECALL_MARGIN,
    _object_band_indices, _render_percept, encode_percept_engram,
    _recall_lang_output_pattern, _cosine_to_object, _recall_metrics, provenance_check,
)

# ── trained-read-out constants (the NEW knobs; everything else is imported) ───────────────────────────────────
# The route starts LOW (so the dense untrained map carries no selectivity) and Hebbian co-firing grows the o->o
# block toward hebbian_max_weight. cfg.hebbian_max_weight (below) caps the grown weight; READOUT_INIT_WEIGHT is
# the cold weight. Low-but-nonzero so the soft-bound rule (delta = lr*(w_max - w)) has headroom from step 1.
READOUT_INIT_WEIGHT = 0.05      # cold dense-route weight (== hebbian_min_weight floor; selectivity is LEARNED).
HEBBIAN_MAX_WEIGHT = 25.0       # the grown-weight ceiling for the route (so a trained o->o line can fire lang).
HEBBIAN_LR = 0.05               # per-co-fire LTP step (soft-bound). Brisk so the route trains in a short window.

# The teacher drive on the language_output target band during TRAINING (the Pulmuller co-firing learning signal,
# the postsynaptic half). Strong enough the lang band FIRES so pre(IT)&post(lang) co-fire -> Hebbian LTP.
LANG_TEACHER_PA = 2500.0
TRAIN_EVENTS_PER_OBJECT = 60    # co-firing trials per object (interleaved/shuffled across objects).
TRAIN_STEPS_PER_EVENT = 12      # steps per co-firing trial (drive IT band + lang band together, run, repeat).

# ── the minimal perception(cortex_it) + read-out(language_output) bridge, but with a DENSE PLASTIC route ──────
def build_trained_probe_bridge(seed: int = 42):
    """A fresh brain-region-framework `SimulationBridge` holding ONLY `cortex_it` (the perception/engram source)
    + `language_output` (the recall channel), with the `cortex_it -> language_output` read-out wired DENSE
    (every excitatory cortex_it neuron -> every excitatory language_output neuron), at a LOW initial weight,
    `plastic=True`, tagged `plasticity_gate="it_to_lang"`. The per-object selectivity is LEARNED by the training
    pass — NOT wired. Hebbian learning is ENABLED at build (the training pass uses it); the internal region
    recurrence is shielded from the training co-firing by zeroing `cp_plasticity_rate_gain` off-route.

    Returns (bridge, handles); handles carry the per-region/per-object band index arrays + the route's
    plasticity-gate synapse indices (for the gain mask + the lesion).
    """
    xp, _ = get_backend()

    # Same region shapes as the confirmed probe (the navigation perception + conversational read-out canon).
    it_region = BrainRegion(name="cortex_it", n_neurons=N_CORTEX_IT, exc_fraction=0.8,
                            internal_density=0.10, exc_weight_mean=2.0, inh_weight_mean=4.0,
                            weight_jitter=0.2, plastic_internal=False,   # internal recurrence FIXED (training is
                            #                                              route-only via the gain mask below).
                            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)
    lang_out_region = BrainRegion(name="language_output", n_neurons=N_LANG_OUTPUT, exc_fraction=0.8,
                                  internal_density=0.10, exc_weight_mean=2.0, inh_weight_mean=4.0,
                                  weight_jitter=0.2, plastic_internal=False,
                                  izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)
    union_regions = [it_region, lang_out_region]

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = union_regions
    cfg.region_pathways = []
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)
    # Hebbian co-firing trains the route. soft-bound w_max/lr as above; min weight = the cold floor.
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = HEBBIAN_LR
    cfg.hebbian_max_weight = HEBBIAN_MAX_WEIGHT
    cfg.hebbian_min_weight = READOUT_INIT_WEIGHT
    cfg.hebbian_weight_decay = 0.0          # no decay: training windows are short; decay would erode the route.
    cfg.stdp_w_max = HEBBIAN_MAX_WEIGHT
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_homeostasis = False          # synaptic-scaling clip foot-gun (same as the confirmed probe).
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_ou_process = True
    cfg.ou_std_current_pA = 20.0
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_nmda = False

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert cfg.enable_homeostasis is False, "homeostasis must stay OFF (synaptic-scaling clip foot-gun)"

    rm = bridge.region_manager
    it_idx_h = np.asarray(list(rm.indices("cortex_it")), dtype=np.int64)
    lo_idx_h = np.asarray(list(rm.indices("language_output")), dtype=np.int64)

    # DENSE plastic route: every EXCITATORY cortex_it neuron -> every EXCITATORY language_output neuron, at the
    # cold init weight. Excitatory-only on BOTH ends (an inhibitory route neuron would SUPPRESS its target; the
    # (A)/(B) probes documented that inversion). The per-object selectivity is grown by training, not wired.
    inh_it = set(int(i) for i in rm.inhibitory_indices("cortex_it"))
    inh_lo = set(int(i) for i in rm.inhibitory_indices("language_output"))
    it_exc = [int(p) for p in it_idx_h if int(p) not in inh_it]
    lo_exc = [int(q) for q in lo_idx_h if int(q) not in inh_lo]
    route_pre, route_post = [], []
    for p in it_exc:
        for q in lo_exc:
            route_pre.append(p); route_post.append(q)
    readout_pop = {
        "pre_indices": np.asarray(route_pre, dtype=np.int64),
        "post_indices": np.asarray(route_post, dtype=np.int64),
        "initial_weights": np.full(len(route_pre), READOUT_INIT_WEIGHT, dtype=np.float32),
        "plastic": True, "plasticity_gate": IT_TO_LANG_GATE, "conn_type": "E_TO_E", "count": len(route_pre),
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

    handles = {
        "seed": int(seed),
        "it_indices": xp.asarray(it_idx_h),
        "lang_out_indices": xp.asarray(lo_idx_h),
        "it_band": {OBJECT_WORDS[o]: _object_band_indices(it_idx_h, o, N_OBJECTS, PERCEPT_SPARSITY)
                    for o in range(N_OBJECTS)},
        # the language_output per-object band the TEACHER drives during training (the postsynaptic co-fire site).
        "lang_band": {OBJECT_WORDS[o]: _object_band_indices(lo_idx_h, o, N_OBJECTS, LANG_SPARSITY)
                      for o in range(N_OBJECTS)},
        "route_syn_idx": bridge._plasticity_gate_indices_gpu[IT_TO_LANG_GATE],  # for the gain mask + lesion.
    }
    return bridge, handles


def _settle(bridge, steps=30):
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(steps):
        bridge._run_one_simulation_step()


def train_readout(bridge, handles, seed: int):
    """TRAIN the dense `cortex_it -> language_output` route by Pulmuller co-firing (the NEW part). For each object
    o, drive object o's cortex_it band (perceived ensemble, presynaptic) AND object o's language_output band (the
    teacher, postsynaptic) TOGETHER for a short window; the bridge's soft-bound Hebbian rule grows the o->o
    synapses (pre & post co-fire). Trials are INTERLEAVED across objects (shuffled) so no single object's
    last-trained block dominates. Training is RESTRICTED to the route: `cp_plasticity_rate_gain` = 0 off-route,
    1 on the `it_to_lang` synapses, so the internal region recurrence is untouched by the training co-firing.

    Returns a dict of grown per-object on/off-diagonal mean route weights (a training-health probe)."""
    xp, _ = get_backend()
    it_indices = handles["it_indices"]
    route_idx = handles["route_syn_idx"]

    # Gate plasticity to the route ONLY (everything else 0): the Hebbian delta is multiplied by this per-synapse
    # gain, so off-route synapses (the internal recurrence) get 0 weight change even though they co-fire.
    if bridge.cp_plasticity_rate_gain is None:
        raise RuntimeError("cp_plasticity_rate_gain not allocated (the route's plasticity_gate should allocate it)")
    bridge.cp_plasticity_rate_gain[:] = 0.0
    bridge.cp_plasticity_rate_gain[route_idx] = 1.0

    # Build the interleaved trial list (shuffled object order), deterministic per seed.
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(TRAIN_EVENTS_PER_OBJECT):
        order = list(range(N_OBJECTS))
        rng.shuffle(order)
        trials.extend(order)

    _settle(bridge)
    for obj_idx in trials:
        # drive the perceived ensemble (IT band, presynaptic) + the teacher (lang band, postsynaptic) together.
        it_drive = orthogonal_drive_pattern(cue_idx=obj_idx, n_cues=N_OBJECTS, n_neurons=int(it_indices.size),
                                             drive_max_pA=PERCEPT_DRIVE_PA, sparsity=PERCEPT_SPARSITY)
        lang_drive = orthogonal_drive_pattern(cue_idx=obj_idx, n_cues=N_OBJECTS,
                                              n_neurons=int(handles["lang_out_indices"].size),
                                              drive_max_pA=LANG_TEACHER_PA, sparsity=LANG_SPARSITY)
        for _ in range(TRAIN_STEPS_PER_EVENT):
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[it_indices] = xp.asarray(it_drive, dtype=xp.float32)
            bridge.cp_external_input_current[handles["lang_out_indices"]] = xp.asarray(lang_drive, dtype=xp.float32)
            bridge._run_one_simulation_step()
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()

    # FREEZE: no further weight updates anywhere (gate the route off + flip Hebbian off globally).
    bridge.set_plasticity_gate(IT_TO_LANG_GATE, 0.0)
    bridge.cp_plasticity_rate_gain[:] = 0.0
    bridge.core_config.enable_hebbian_learning = False
    bridge.cp_external_input_current[:] = 0.0
    _settle(bridge)

    # Training-health probe: grown on-diagonal (o->o) vs off-diagonal (o->p!=o) route-weight means. A trained
    # map has on >> off; a NEGATIVE here localizes "route didn't learn selectivity" vs "read-out is fine but
    # the engram doesn't reactivate". (Diagnostic only — not a gate.)
    coo = bridge._get_cached_coo()
    rows = to_host(coo.row[route_idx]); cols = to_host(coo.col[route_idx])
    w = to_host(bridge.cp_connections.data[route_idx])
    it_band_set = {OBJECT_WORDS[o]: set(int(i) for i in handles["it_band"][OBJECT_WORDS[o]]) for o in range(N_OBJECTS)}
    lo_band_set = {OBJECT_WORDS[o]: set(int(i) for i in handles["lang_band"][OBJECT_WORDS[o]]) for o in range(N_OBJECTS)}
    def _obj_of(idx, band_sets):
        for o in range(N_OBJECTS):
            if int(idx) in band_sets[OBJECT_WORDS[o]]:
                return o
        return -1
    on_diag, off_diag = [], []
    for r, c, wi in zip(rows, cols, w):
        oi = _obj_of(r, it_band_set); oj = _obj_of(c, lo_band_set)
        if oi < 0 or oj < 0:
            continue
        (on_diag if oi == oj else off_diag).append(float(wi))
    return {
        "route_weight_on_diag_mean": float(np.mean(on_diag)) if on_diag else 0.0,
        "route_weight_off_diag_mean": float(np.mean(off_diag)) if off_diag else 0.0,
        "route_weight_max": float(w.max()) if w.size else 0.0,
        "n_train_trials": len(trials),
    }


# ── one seed: train the read-out, encode every object's percept engram, recall (clean) + recall (lesioned) ─────
def run_seed(seed: int):
    xp, backend = get_backend()
    print(f"\n[funcint-p2m-trained] ===== seed {seed} (backend={backend}) =====")
    bridge, handles = build_trained_probe_bridge(seed)

    # (TRAIN) grow the dense cortex_it -> language_output route by co-firing (the NEW part).
    train_stats = train_readout(bridge, handles, seed)
    print(f"[funcint-p2m-trained]  trained read-out: route w on-diag={train_stats['route_weight_on_diag_mean']:.3f} "
          f"off-diag={train_stats['route_weight_off_diag_mean']:.3f} max={train_stats['route_weight_max']:.2f} "
          f"({train_stats['n_train_trials']} trials)")

    # (write) encode each object's perception engram (perceive -> tag the cortex_it ensemble). Reused verbatim.
    encode_stats = {}
    for w in OBJECT_WORDS:
        encode_stats[w] = encode_percept_engram(bridge, handles, w)
        print(f"[funcint-p2m-trained]  encoded seen_{w}: tagged {encode_stats[w]['n_tagged']} cortex_it neurons "
              f"(window {encode_stats[w]['window_ms']:.0f} ms)")

    prov = provenance_check(bridge, handles)

    # (recall clean) stimulate each tag -> read language_output reactivation through the TRAINED route.
    n_lo = int(handles["lang_out_indices"].size)
    per_obj = {}
    n_recall_correct = 0
    for w in OBJECT_WORDS:
        pat = _recall_lang_output_pattern(bridge, handles, w)
        m = _recall_metrics(pat, w, n_lo)
        per_obj[w] = m
        n_recall_correct += int(m["correct"])
        print(f"[funcint-p2m-trained]  recall seen_{w:5s}: top1={m['top1']:5s} "
              f"(perceived_score={m['perceived_score']:.4f} margin={m['margin']:+.4f}) correct={m['correct']}")

    # (recall lesioned) zero the TRAINED route, re-stimulate every tag: recall must collapse to chance.
    lesion_per_obj, n_lesion_correct = _run_lesion(seed)

    print(f"[funcint-p2m-trained] seed {seed} roll-up: CLEAN recall correct={n_recall_correct}/{N_OBJECTS}  "
          f"LESION recall correct={n_lesion_correct}/{N_OBJECTS}  (chance=1/{N_OBJECTS})")

    return {
        "seed": int(seed),
        "n_objects": N_OBJECTS,
        "chance": 1.0 / N_OBJECTS,
        "train_stats": train_stats,
        "encode_stats": {w: {"n_tagged": int(s["n_tagged"]), "window_ms": float(s["window_ms"])}
                         for w, s in encode_stats.items()},
        "per_obj": per_obj,
        "lesion_per_obj": lesion_per_obj,
        "n_recall_correct": n_recall_correct,
        "n_lesion_correct": n_lesion_correct,
        "provenance": prov,
    }


def _run_lesion(seed: int):
    """LESION control (anti-cheat 1, primary): build a fresh trained probe bridge, RE-TRAIN the route + re-encode
    every object's perception engram, then ZERO every `cortex_it -> language_output` route synapse's weight, and
    re-stimulate each tag. With the engram intact but the TRAINED read-out cut, recall must FAIL -> the recall
    rides the trained route synapses, not ambient leakage or any non-route path."""
    xp, _ = get_backend()
    bridge, handles = build_trained_probe_bridge(seed)
    train_readout(bridge, handles, seed)
    for w in OBJECT_WORDS:
        encode_percept_engram(bridge, handles, w)
    idx = handles["route_syn_idx"]
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
    print(f"[funcint-p2m-trained]  LESION (trained cortex_it->language_output weights zeroed, n_synapses="
          f"{n_lesioned}) recall: " + " ".join(f"{w}={out[w]['correct']}" for w in OBJECT_WORDS))
    return out, n_correct


# ── verdict ──────────────────────────────────────────────────────────────────────────────────────────────
def verdict_from(results):
    """GO       : CLEAN recall correct >=3/4 objects on ALL seeds (>> chance 1/4) AND the LESION collapses recall
                  to <=1/4 on ALL seeds (the recall rides the TRAINED route).
       PARTIAL  : CLEAN recall beats chance on a majority across seeds (avg >=2/4) AND the lesion collapses it,
                  but full >=3/4-all-seeds is not met (some object recalls weakly / seed-variable).
       NEGATIVE : CLEAN recall does not reliably exceed chance, OR the lesion does not collapse it. An honest
                  negative MAPS the limit (the noisy trained map needs more training / a cleaner code)."""
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
        "route_on_off_diag_per_seed": {r["seed"]: {
            "on": r["train_stats"]["route_weight_on_diag_mean"],
            "off": r["train_stats"]["route_weight_off_diag_mean"]} for r in results},
        "provenance_all_seeds": bool(all(r["provenance"]["every_tag_is_a_cortex_it_subset"] for r in results)),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Functional-integration cheap-first de-risk: PERCEPTION->MEMORY with a TRAINED (Hebbian, "
                    "noisy) cortex_it->language_output read-out (the next idealization peeled off the GO probe).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/funcint_perception_to_memory_trained.json")
    args = ap.parse_args()

    results = [run_seed(s) for s in args.seeds]
    vd = verdict_from(results)

    print("\n[funcint-p2m-trained] ============ VERDICT ============")
    print(f"[funcint-p2m-trained] verdict={vd['verdict']}  (chance=1/{N_OBJECTS}={vd['chance']:.2f})")
    print(f"[funcint-p2m-trained]   CLEAN recall correct (>=3/4) all seeds : {vd['recall_ge3_all_seeds']}  "
          f"{vd['recall_correct_per_seed']}")
    print(f"[funcint-p2m-trained]   LESION collapses recall (<=1/4) all    : {vd['lesion_collapses_all_seeds']}  "
          f"{vd['lesion_correct_per_seed']}")
    print(f"[funcint-p2m-trained]   route w on/off-diag per seed           : {vd['route_on_off_diag_per_seed']}")
    print(f"[funcint-p2m-trained]   provenance (every tag = cortex_it subset) all : {vd['provenance_all_seeds']}")

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
    print(f"[funcint-p2m-trained] wrote {args.out}")
    raise SystemExit(0 if vd["verdict"] == "GO" else (2 if vd["verdict"] == "PARTIAL" else 1))


if __name__ == "__main__":
    main()
