"""EMERGE-90 -- THE CONVERSATIONAL-TURN CAPSTONE: HEAR -> comprehend -> store -> ASK -> SPEAK the answer (SPIKING WORD-ORDER).

HONEST SCOPE up front (per the adversarial-verify verdict): this wires THREE co-executing components in ONE PROCESS --
a RATE reservoir (comprehension), an RF-phasor SPIKING bridge (memory), and an Izhikevich SPIKING bridge (production)
-- with host-dict hand-offs and NO shared bridge / NO cross-synaptic interaction. So it is NOT "one brain" in the
EMERGE-70/71 shared-substrate sense. TWO of the three are spiking; comprehension is the RATE reservoir (the spiking
`OnBridgeLSM` swap, EMERGE-89, is the named follow-on). The producer speaks the answer with SPIKING slot-ORDER; the
WORD SURFACES are the host-token spell (`spell=str(w)`) and 3sg inflection is host `emerge_v3` (the A->W neural spell,
EMERGE-67/68, is the fully-spiking-words follow-on). The construction `C_TRANS` is hand-named (its SHAPE is corpus-mined
EMERGE-72; the per-turn message->construction router is not exercised here).

Wires the two spiking components (memory + production) + a rate reservoir comprehender into ONE conversational turn:
  * COMPREHENSION -- the fronto-striatal reservoir (EMERGE-78/88) parses a heard transitive sentence into
    (agent, action, patient) from the closed-class configuration (content abstracted -> roles from STRUCTURE).
  * MEMORY -- the RF phasor composer stores the fact + answers who/what, with the no-confab moat.
  * PRODUCTION -- the self-organized spiking Broca producer (EMERGE-72/74 C_TRANS registry) SPEAKS the answer
    sentence ON SPIKES (the frame-slot competitive-queuing emission order on a real bridge).

The turn: HEAR "the dog chases the ball" -> the reservoir comprehends the roles -> the composer stores the fact ->
ASK "what does the dog chase?" -> the composer recalls the patient -> the producer SPEAKS "the dog chases the ball"
ON SPIKES. GATE-FIRST moat: an unstored query -> the composer abstains -> the producer is NEVER invoked.

Both halves are self-taught from corpus experience (the reservoir's form->role map; the producer's WHOLE grammar --
function words, slot order, slot inventory -- all discovered, EMERGE-62..65); no hand-written grammar rulebook, no
bolted-on language model. The whole turn co-executes on ONE numpy process (host-token spell = spiking ORDER, host word
surface; the A->W neural spell is the cupy follow-on). Reuse-by-import; NO `sim/` edit.

Anti-cheats: held-out CONTENT (fresh transitive draws); render_exact (the spoken surface == the ground-truth
transitive); the no-confab MOAT (unstored -> abstain -> the producer is NEVER invoked, production_count unchanged);
COMPREHENSION-LESION (the reservoir's closed-class identity collapsed -> wrong roles -> recall + render collapse);
PRODUCER-NO-LEARN (the producer's learned spiking ORDER removed -> the spoken order collapses) = the spoken order is
genuinely from the spiking producer, not a host join.

Run:  SIM_BACKEND=numpy python -u -m research.runners._emerge90_conversational_turn_capstone_derisk \
          --seeds 42 43 44 100 101 102 --json research/findings/raw/_emerge90_conversational_turn_capstone.json
"""
import argparse
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

import research.runners._emerge62_discover_function_words_derisk as m62  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import (  # noqa: E402
    _content_pools, _gen, _TRAIN_KINDS, _N_TRAIN_PER_CONSTRUCTION,
)
from research.runners._emerge88_reservoir_comprehends_composer_answers_derisk import (  # noqa: E402
    ReservoirComprehender, _D, _N_TEST,
)
from research.runners._emerge72_construction_registry_derisk import (  # noqa: E402
    decision, RegistryBrocaProducer, RegistryProducer,
)
from research.runners._emerge74_transitive_ditransitive_derisk import (  # noqa: E402
    build_stream_svo, SVOConstructionRegistry, emerge_v3, _TRANS_VERBS, _SUBJ_SET, _OBJ_SET,
)
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402


def _build_facts(seed, subjects, objects, verbs, v3_of, n=_N_TEST):
    """Distinct-(agent, action) transitive facts with fresh CONTENT draws. Each carries the surface 3sg sentence to be
    HEARD and the ground-truth for scoring."""
    trng = np.random.default_rng(seed * 733 + 11)
    facts, seen = [], set()
    guard = 0
    while len(facts) < n and guard < 6000:
        guard += 1
        s = str(trng.choice(subjects)); vb = str(trng.choice(verbs)); o = str(trng.choice(objects))
        v3 = v3_of[vb]
        if (s, v3) in seen or s == o:
            continue
        seen.add((s, v3))
        facts.append({"subj": s, "verb_bare": vb, "v3": v3, "obj": o, "sentence": ["the", s, v3, "the", o]})
    return facts, seen, trng


def _speak_answer(producer, subj, verb_bare, patient):
    """SPEAK the transitive answer on spikes via the registry producer (gate=ANSWER)."""
    out = producer.speak(decision("ANSWER", construction="C_TRANS", subject=subj, verb=verb_bare, obj=patient))
    return out["surface"], bool(out["produced"])


def _derisk_one(seed, spiking_reservoir=False):
    # ---- PRODUCTION: the self-organized spiking producer over the corpus-mined registry (C_TRANS registered) --------
    tokens = build_stream_svo(seed)
    reg = SVOConstructionRegistry(seed).build(tokens)
    assert "C_TRANS" in reg.registered, "C_TRANS must mine from the SVO corpus"
    producer = RegistryBrocaProducer(reg.render_cq())          # host-token spell (spiking ORDER on a real bridge)

    # ---- COMPREHENSION: the reservoir form->role map (content abstracted, so vocabulary is free) --------------------
    # Default: the EMERGE-88 RATE reservoir. `spiking_reservoir=True` (EMERGE-91): the EMERGE-82/89 on-bridge SPIKING
    # reservoir (`OnBridgeLSM`, identical `final_state(U)` signature) -> comprehension is spiking too, on a real bridge
    # (a heavier fit -> the EMERGE-82 reduced train). The rest of the turn is byte-identical either way.
    stream = m62.build_stream(seed, n_sentences=6000)
    words, freq, cover, _c = m62.compute_stats(stream)
    discovered, *_ = m62.discover_closed_class(words, freq, cover)
    subj_m62, verb_m62, obj_m62 = _content_pools(discovered)
    if spiking_reservoir:
        from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder
        from research.runners._emerge82_onbridge_lsm_derisk import OnBridgeLSM, _N_POOL, _N_TRAIN_PER
        enc = Encoder(discovered)
        comp = ReservoirComprehender(seed, discovered, res=OnBridgeLSM(enc.dim, seed=seed, n=_N_POOL), enc=enc)
        _n_train = _N_TRAIN_PER
    else:
        comp = ReservoirComprehender(seed, discovered)
        _n_train = _N_TRAIN_PER_CONSTRUCTION
    comp.fit(_gen(_TRAIN_KINDS, _n_train, np.random.default_rng(seed * 101 + 5),
                  subj_m62, verb_m62, obj_m62))

    # ---- shared transitive vocab (the producer speaks any fillers; the reservoir abstracts content) -----------------
    # Isolate the WIRE from EMERGE-62's closed-class-discovery precision: use only content words the reservoir sees as
    # genuinely OPEN (a noun/verb the discovery FALSE-POSITIVED as closed would be read as a function word -> a parse
    # miss that reflects the discovery's precision, a SEPARATELY-characterized property, not the comprehension->
    # production wire under test here).
    _open = lambda w: w not in comp.closed
    subjects = [w for w in sorted(_SUBJ_SET) if _open(w)]
    objects = [w for w in sorted(_OBJ_SET) if _open(w)]
    verbs = [v for v in _TRANS_VERBS if _open(v) and _open(emerge_v3(v))]
    v3_of = {v: emerge_v3(v) for v in verbs}
    bare_of = {emerge_v3(v): v for v in verbs}                 # the morphological de-inflection lexicon (3sg -> bare)
    facts, seen, trng = _build_facts(seed, subjects, objects, verbs, v3_of)
    composer_vocab = sorted(set(subjects) | set(objects) | set(v3_of.values()))

    # ---- THE TURN: HEAR -> comprehend -> store --------------------------------------------------------------------
    composer = RFPhasorComposer(seed=seed, D=_D, vocab=composer_vocab)
    parsed = [comp.comprehend(f["sentence"]) for f in facts]
    for pf in parsed:
        if {"agent", "action", "patient"} <= set(pf):
            composer.store(pf["agent"], pf["action"], pf["patient"])
    parse_acc = float(np.mean([int(pf.get("agent") == f["subj"] and pf.get("action") == f["v3"]
                                   and pf.get("patient") == f["obj"]) for pf, f in zip(parsed, facts)]))

    # ---- ASK -> recall -> SPEAK the answer on spikes --------------------------------------------------------------
    recall_hit = render_hit = 0
    for f in facts:
        patient = composer.query_patient(f["subj"], f["v3"])
        recall_hit += int(patient == f["obj"])
        if patient is None:
            producer.speak(decision("ABSTAIN"))               # gate-first: the producer is NOT run
            continue
        surface, _p = _speak_answer(producer, f["subj"], bare_of.get(f["v3"], f["v3"]), patient)
        render_hit += int(surface == " ".join(["the", f["subj"], f["v3"], "the", f["obj"]]))
    recall = recall_hit / len(facts)
    render_exact = render_hit / len(facts)

    # ---- MOAT (gate-first): unstored (agent, action) -> abstain -> the producer is NEVER invoked -------------------
    prod_count_before = producer.production_count
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
            producer.speak(decision("ABSTAIN"))               # gate-first: producer NOT run
        else:
            fa += 1                                            # a false-accept (confab) that WOULD be spoken
    moat_fa = fa / max(1, tot)
    moat_producer_invoked_on_abstain = int(producer.production_count - prod_count_before)   # must be 0

    # ---- COMPREHENSION-LESION: reservoir lesioned -> wrong roles -> recall + render collapse -----------------------
    composer_l = RFPhasorComposer(seed=seed, D=_D, vocab=composer_vocab)
    for f in facts:
        pf = comp.comprehend(f["sentence"], lesion=True)
        if {"agent", "action", "patient"} <= set(pf):
            composer_l.store(pf["agent"], pf["action"], pf["patient"])
    lesion_hit = 0
    for f in facts:
        patient = composer_l.query_patient(f["subj"], f["v3"])
        if patient is None:
            continue
        surface, _p = _speak_answer(producer, f["subj"], bare_of.get(f["v3"], f["v3"]), patient)
        lesion_hit += int(surface == " ".join(["the", f["subj"], f["v3"], "the", f["obj"]]))
    lesion_render_exact = lesion_hit / len(facts)

    # ---- PRODUCER-NO-LEARN: remove the learned spiking ORDER -> the spoken order collapses ------------------------
    producer_nl = RegistryBrocaProducer(RegistryProducer(seed=seed, registry_slots=reg.registered))   # NO .learn()
    nl_hit = 0
    for f in facts:
        patient = composer.query_patient(f["subj"], f["v3"])
        if patient is None:
            continue
        surface, _p = _speak_answer(producer_nl, f["subj"], bare_of.get(f["v3"], f["v3"]), patient)
        nl_hit += int(surface == " ".join(["the", f["subj"], f["v3"], "the", f["obj"]]))
    nolearn_render_exact = nl_hit / len(facts)

    return {
        "seed": seed, "n_facts": len(facts), "n_registered_constructions": reg.n_registered(),
        "parse_acc": parse_acc, "recall": recall, "render_exact": render_exact,
        "moat_false_accept": moat_fa, "moat_producer_invoked_on_abstain": moat_producer_invoked_on_abstain,
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
        # GO: the reservoir parses (parse), the composer recalls (recall), the producer SPEAKS the answer on spikes
        # (render_exact); the moat holds gate-first (0 false-accepts, producer never invoked on abstain); and both the
        # comprehension AND the learned spiking order are load-bearing (both lesions collapse the render).
        "go": (mean("parse_acc") >= 0.90 and mean("recall") >= 0.90 and mean("render_exact") >= 0.90
               and mean("moat_false_accept") <= 0.05
               and max(r["moat_producer_invoked_on_abstain"] for r in rows) == 0
               and mean("lesion_render_exact") <= 0.30 and mean("nolearn_render_exact") <= 0.60),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--spiking-reservoir", action="store_true",
                    help="EMERGE-91: use the on-bridge SPIKING reservoir (OnBridgeLSM) for comprehension too")
    args = ap.parse_args()

    t0 = time.time()
    rows = []
    for s in args.seeds:
        d = _derisk_one(s, spiking_reservoir=args.spiking_reservoir)
        rows.append(d)
        print(f"[seed {s}] parse {d['parse_acc']:.3f} | recall {d['recall']:.3f} | render {d['render_exact']:.3f} | "
              f"moat-FA {d['moat_false_accept']:.3f} invoked-on-abstain {d['moat_producer_invoked_on_abstain']} | "
              f"lesion-render {d['lesion_render_exact']:.3f} | nolearn-render {d['nolearn_render_exact']:.3f}",
              flush=True)

    agg = _go(rows)
    agg["elapsed_seconds"] = round(time.time() - t0, 1)
    agg["spiking_reservoir"] = bool(args.spiking_reservoir)
    verdict = "GO" if agg["go"] else "NO-GO"
    tag = "emerge91 SPIKING-reservoir" if args.spiking_reservoir else "emerge90"
    res_label = "on-bridge SPIKING reservoir" if args.spiking_reservoir else "RATE reservoir"
    print(f"\n[{tag}] VERDICT: {verdict} -- the CONVERSATIONAL TURN (3 co-executing components, 1 process; NOT one "
          f"shared bridge): HEAR->comprehend ({res_label})->store (RF spiking)->ASK->SPEAK the answer with SPIKING "
          f"WORD-ORDER (Izhikevich producer; words=host-token spell). parse {agg['parse_acc']:.3f}; recall "
          f"{agg['recall']:.3f}; render_exact {agg['render_exact']:.3f}; no-confab moat {agg['moat_false_accept']:.3f} "
          f"false-accept + {agg['moat_producer_invoked_on_abstain']} producer-invocations-on-abstain (gate-first); "
          f"comprehension-lesion collapses render to {agg['lesion_render_exact']:.3f}; producer-no-learn collapses "
          f"render to {agg['nolearn_render_exact']:.3f}.", flush=True)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "agg": agg}, fh, indent=2)
        print(f"[emerge90] wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
