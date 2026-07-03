"""EMERGE-93 -- RUNG A.2 toward the one-brain capstone: the two SPIKING components (composer + producer) fold onto ONE bridge.

RUNG A.1 (EMERGE-92) proved the producer runs as a `slots` slice on a shared bridge. RUNG A.2 folds the capstone's TWO
spiking bridges -- the RF-phasor composer (memory) and the Izhikevich slot producer (production) -- onto ONE
`SimulationBridge` as disjoint slices (`rf` + `slots`), and runs the whole EMERGE-90 conversational turn against the
shared slices. The rate reservoir (comprehension) has no bridge, so this is the honest "fold the two spiking bridges
into one" step: the capstone's composer + producer are no longer separate substrates -- they co-reside on one bridge.

Co-residence mechanics (both validated): the composer via `MergedRFComposer` (masked RF ops on the `rf` slice --
step-2b); the producer via the EMERGE-92 `shared_bridge=` slot slice. dt=1.0 (the RF ops are a dt-agnostic SEPARATE loop
`rf_resonate_steps`; the producer is dt=1.0). The composer's facts live in the complex RF synapses (`cp_rf_w_re/im`,
array-disjoint from `cp_connections` + not in the producer's wash-out snapshot), so the producer's inter-utterance
wash-out does NOT disturb the composer's memory.

Anti-cheats: the full turn on the SHARED-bridge composer + producer reproduces the EMERGE-90 separate-bridge result
(parse/recall/render 1.000 -> functional isolation: co-location changes nothing); the gate-first no-confab moat
(unstored -> abstain -> producer NEVER invoked); comprehension-lesion collapses render; producer-no-learn collapses the
spoken order; the two slices are index-disjoint with NO cross-region pathways. NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -u -m research.runners._emerge93_two_spiking_onebridge_turn_derisk \
          --seeds 42 43 44 100 101 102 --json research/findings/raw/_emerge93_two_spiking_onebridge_turn.json
"""
import argparse
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

import research.runners._emerge62_discover_function_words_derisk as m62  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import (  # noqa: E402
    _content_pools, _gen, _TRAIN_KINDS, _N_TRAIN_PER_CONSTRUCTION,
)
from research.runners._emerge88_reservoir_comprehends_composer_answers_derisk import ReservoirComprehender  # noqa: E402
from research.runners._emerge59_spiking_broca_frame_slots_derisk import N_PER, N_SLOT_POOLS  # noqa: E402
from research.runners._emerge72_construction_registry_derisk import (  # noqa: E402
    decision, RegistryBrocaProducer, RegistryProducer,
)
from research.runners._emerge74_transitive_ditransitive_derisk import (  # noqa: E402
    build_stream_svo, SVOConstructionRegistry, emerge_v3, _TRANS_VERBS, _SUBJ_SET, _OBJ_SET,
)
from research.runners.nav_conv_merged_bridge import MergedRFComposer  # noqa: E402

_D = 256
_N_TEST = 12          # fewer facts than EMERGE-90 -> the batched-scan rf region (2*K*D) stays modest


def _build_shared_bridge(seed, rf_size, n_slot_pools=N_SLOT_POOLS):
    """ONE bridge (dt=1.0) hosting `rf` (composer, sized for the batched scan) + `slots` (producer) + `_anchor`, as
    DISJOINT slices with NO cross-region pathways."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="rf", n_neurons=int(rf_size), exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="slots", n_neurons=n_slot_pools * N_PER, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="_anchor", n_neurons=4, exc_fraction=1.0, internal_density=1.0),
    ]
    cfg.region_pathways = []
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    rt = RuntimeState()
    rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    rf_base = int(b.region_manager.indices("rf")[0])
    return b, rf_base


def _speak(producer, subj, verb_bare, patient):
    return producer.speak(decision("ANSWER", construction="C_TRANS", subject=subj, verb=verb_bare, obj=patient))["surface"]


def _derisk_one(seed):
    # PRODUCTION registry (C_TRANS) + COMPREHENSION reservoir (rate; content abstracted -> vocabulary is free)
    reg = SVOConstructionRegistry(seed).build(build_stream_svo(seed))
    assert "C_TRANS" in reg.registered
    stream = m62.build_stream(seed, n_sentences=6000)
    words, freq, cover, _c = m62.compute_stats(stream)
    discovered, *_ = m62.discover_closed_class(words, freq, cover)
    subj_m62, verb_m62, obj_m62 = _content_pools(discovered)
    comp = ReservoirComprehender(seed, discovered)
    comp.fit(_gen(_TRAIN_KINDS, _N_TRAIN_PER_CONSTRUCTION, np.random.default_rng(seed * 101 + 5),
                  subj_m62, verb_m62, obj_m62))

    _open = lambda w: w not in comp.closed
    subjects = [w for w in sorted(_SUBJ_SET) if _open(w)]
    objects = [w for w in sorted(_OBJ_SET) if _open(w)]
    verbs = [v for v in _TRANS_VERBS if _open(v) and _open(emerge_v3(v))]
    v3_of = {v: emerge_v3(v) for v in verbs}
    bare_of = {emerge_v3(v): v for v in verbs}
    composer_vocab = sorted(set(subjects) | set(objects) | set(v3_of.values()))

    trng = np.random.default_rng(seed * 733 + 11)
    facts, seen = [], set()
    guard = 0
    while len(facts) < _N_TEST and guard < 6000:
        guard += 1
        s = str(trng.choice(subjects)); vb = str(trng.choice(verbs)); o = str(trng.choice(objects))
        v3 = v3_of[vb]
        if (s, v3) in seen or s == o:
            continue
        seen.add((s, v3))
        facts.append({"subj": s, "verb_bare": vb, "v3": v3, "obj": o, "sentence": ["the", s, v3, "the", o]})

    # ONE bridge for BOTH spiking components: rf sized for the batched scan (2*K*D) + margin; slots for the producer.
    rf_size = 2 * (_N_TEST + 3) * _D
    bridge, rf_base = _build_shared_bridge(seed, rf_size)
    # producer FIRST (clean post-init wash-out snapshot), THEN composer (RF ops after) -- both on the SHARED bridge.
    cq = RegistryProducer(seed=seed, registry_slots=reg.registered_fits(), shared_bridge=bridge, slot_region="slots")
    cq.learn()
    producer = RegistryBrocaProducer(cq)
    composer = MergedRFComposer(bridge, rf_base, rf_size, seed=seed, D=_D, vocab=composer_vocab)

    # THE TURN on the shared bridge: HEAR -> comprehend -> store -> ASK -> recall -> SPEAK
    parsed = [comp.comprehend(f["sentence"]) for f in facts]
    for pf in parsed:
        if {"agent", "action", "patient"} <= set(pf):
            composer.store(pf["agent"], pf["action"], pf["patient"])
    parse_acc = float(np.mean([int(pf.get("agent") == f["subj"] and pf.get("action") == f["v3"]
                                   and pf.get("patient") == f["obj"]) for pf, f in zip(parsed, facts)]))

    recall_hit = render_hit = 0
    for f in facts:
        patient = composer.query_patient(f["subj"], f["v3"])
        recall_hit += int(patient == f["obj"])
        if patient is None:
            producer.speak(decision("ABSTAIN"))
            continue
        render_hit += int(_speak(producer, f["subj"], bare_of.get(f["v3"], f["v3"]), patient)
                          == " ".join(["the", f["subj"], f["v3"], "the", f["obj"]]))
    recall = recall_hit / len(facts)
    render_exact = render_hit / len(facts)

    # MOAT (gate-first): unstored -> abstain -> producer NEVER invoked
    prod_before = producer.production_count
    stored_keys = {(f["subj"], f["v3"]) for f in facts}
    fa = tot = 0; mguard = 0
    while tot < 40 and mguard < 5000:
        mguard += 1
        s = str(trng.choice(subjects)); v3q = v3_of[str(trng.choice(verbs))]
        if (s, v3q) in stored_keys:
            continue
        tot += 1
        patient = composer.query_patient(s, v3q)
        if patient is None:
            producer.speak(decision("ABSTAIN"))
        else:
            fa += 1
    moat_fa = fa / max(1, tot)
    moat_invoked_on_abstain = int(producer.production_count - prod_before)

    # COMPREHENSION-LESION on a FRESH shared bridge (isolate the composer's state)
    bridge_l, rf_base_l = _build_shared_bridge(seed, rf_size)
    cq_l = RegistryProducer(seed=seed, registry_slots=reg.registered_fits(), shared_bridge=bridge_l, slot_region="slots")
    cq_l.learn()
    producer_l = RegistryBrocaProducer(cq_l)
    composer_l = MergedRFComposer(bridge_l, rf_base_l, rf_size, seed=seed, D=_D, vocab=composer_vocab)
    for f in facts:
        pf = comp.comprehend(f["sentence"], lesion=True)
        if {"agent", "action", "patient"} <= set(pf):
            composer_l.store(pf["agent"], pf["action"], pf["patient"])
    lesion_hit = 0
    for f in facts:
        patient = composer_l.query_patient(f["subj"], f["v3"])
        if patient is None:
            continue
        lesion_hit += int(_speak(producer_l, f["subj"], bare_of.get(f["v3"], f["v3"]), patient)
                          == " ".join(["the", f["subj"], f["v3"], "the", f["obj"]]))
    lesion_render_exact = lesion_hit / len(facts)

    # PRODUCER-NO-LEARN on the SHARED bridge (the composer's recall is correct; only the spoken order is scrambled)
    cq_nl = RegistryProducer(seed=seed, registry_slots=reg.registered_fits(), shared_bridge=bridge, slot_region="slots")
    producer_nl = RegistryBrocaProducer(cq_nl)   # NO .learn()
    nl_hit = 0
    for f in facts:
        patient = composer.query_patient(f["subj"], f["v3"])
        if patient is None:
            continue
        nl_hit += int(_speak(producer_nl, f["subj"], bare_of.get(f["v3"], f["v3"]), patient)
                      == " ".join(["the", f["subj"], f["v3"], "the", f["obj"]]))
    nolearn_render_exact = nl_hit / len(facts)

    return {
        "seed": seed, "n_facts": len(facts), "rf_size": rf_size,
        "parse_acc": parse_acc, "recall": recall, "render_exact": render_exact,
        "moat_false_accept": moat_fa, "moat_producer_invoked_on_abstain": moat_invoked_on_abstain,
        "lesion_render_exact": lesion_render_exact, "nolearn_render_exact": nolearn_render_exact,
    }


def _go(rows):
    def mean(k):
        return float(np.mean([r[k] for r in rows]))
    return {
        "n_seeds": len(rows),
        "parse_acc": mean("parse_acc"), "recall": mean("recall"), "render_exact": mean("render_exact"),
        "moat_false_accept": mean("moat_false_accept"),
        "moat_producer_invoked_on_abstain": int(max(r["moat_producer_invoked_on_abstain"] for r in rows)),
        "lesion_render_exact": mean("lesion_render_exact"), "nolearn_render_exact": mean("nolearn_render_exact"),
        "go": (mean("parse_acc") >= 0.90 and mean("recall") >= 0.90 and mean("render_exact") >= 0.90
               and mean("moat_false_accept") <= 0.05
               and max(r["moat_producer_invoked_on_abstain"] for r in rows) == 0
               and mean("lesion_render_exact") <= 0.30 and mean("nolearn_render_exact") <= 0.60),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    rows = []
    for s in args.seeds:
        d = _derisk_one(s)
        rows.append(d)
        print(f"[seed {s}] parse {d['parse_acc']:.3f} | recall {d['recall']:.3f} | render {d['render_exact']:.3f} | "
              f"moat-FA {d['moat_false_accept']:.3f} invoked-on-abstain {d['moat_producer_invoked_on_abstain']} | "
              f"lesion {d['lesion_render_exact']:.3f} | nolearn {d['nolearn_render_exact']:.3f}", flush=True)

    agg = _go(rows)
    verdict = "GO" if agg["go"] else "NO-GO"
    print(f"\n[emerge93] VERDICT: {verdict} -- the TWO SPIKING components (RF composer + Izhikevich producer) fold onto "
          f"ONE bridge as disjoint slices + run the whole turn: parse {agg['parse_acc']:.3f}; recall {agg['recall']:.3f}; "
          f"render_exact {agg['render_exact']:.3f}; moat {agg['moat_false_accept']:.3f} FA + "
          f"{agg['moat_producer_invoked_on_abstain']} invoked-on-abstain (gate-first); comprehension-lesion "
          f"{agg['lesion_render_exact']:.3f}; producer-no-learn {agg['nolearn_render_exact']:.3f}. RUNG A.2.", flush=True)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "agg": agg}, fh, indent=2)
        print(f"[emerge93] wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
