"""RUNG B-1 / FULLY-SPIKING-WORDS one-brain transitive turn (cupy) — comprehension + memory + production (ORDER AND WORDS) all on spikes, one process.

The one-brain-substrate capstone (EMERGE-95) runs the whole transitive turn — the spiking reservoir COMPREHENDS, the RF
composer REMEMBERS, the Izhikevich producer SPEAKS the emission ORDER — as three disjoint slices on ONE `SimulationBridge`
(validated on cupy here: EMERGE-95 seed 42 GO on `SIM_BACKEND=cupy`). But the producer's word SURFACES were the host-token
spell. This runner composes it with the A→W SPIKING word-spell (the transitive BRIDGE-A + the function BRIDGE-F, thread A)
+ the EMERGE-77 2-stage CALIBRATED order read (the cupy near-tie fix) — so the whole turn is spiking end-to-end: the
reservoir comprehends on the shared bridge, the composer stores/recalls on the shared bridge, and the producer speaks the
answer with the spiking ORDER (shared bridge) AND every WORD decoded from `language_output` spikes (BRIDGE-A/F) — all
co-executing in ONE cupy process (the EMERGE-70/71 one-process result).

hear "the dog chases the ball" -> reservoir(shared bridge) parses roles -> composer(shared bridge) stores + recalls ->
producer speaks "the dog chases the ball" with the spiking order + the A→W spiking words. GATE-FIRST moat.

Reuse-by-import (EMERGE-95 one-brain turn + EMERGE-88 comprehender + EMERGE-77 calibrated producer + the thread-A A→W
transitive spell); NO `sim/` edit. GPU/cupy (the A→W read-out is the validated scale; EMERGE-95 co-executes on cupy).

Anti-cheats: parse (reservoir comprehends) / recall (composer) / render_exact (producer speaks ORDER+WORDS on spikes);
gate-first no-confab moat; comprehension-lesion collapses; content-lesion (zero the A→W pool→language_output) collapses
the words (genuinely spiking).

Run:  SIM_BACKEND=cupy python -u -m research.runners._rungB1_fully_spiking_words_capstone_derisk \
          --seeds 42 43 44 --json research/findings/raw/_rungB1_fully_spiking_words_capstone.json
"""
import argparse
import json
import os

os.environ.setdefault("SIM_BACKEND", "cupy")
import numpy as np  # noqa: E402

import research.runners._emerge62_discover_function_words_derisk as m62  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _content_pools, _gen, _TRAIN_KINDS, _N_TRAIN_PER_CONSTRUCTION  # noqa: E402
from research.runners._emerge88_reservoir_comprehends_composer_answers_derisk import ReservoirComprehender  # noqa: E402
import research.runners._emerge95_three_spiking_onebridge_turn_derisk as m95  # noqa: E402
from research.runners._emerge72_construction_registry_derisk import decision, RegistryBrocaProducer  # noqa: E402
from research.runners._emerge77_ditransitive_render_derisk import DitransRegistryProducer  # noqa: E402
from research.runners._emerge74_transitive_ditransitive_derisk import build_stream_svo, SVOConstructionRegistry, emerge_v3  # noqa: E402
import research.runners._rungB1_aw_neural_words_transitive_derisk as AW  # noqa: E402

_D = m95._D
_N_TEST = 12


def _facts(seed, closed=frozenset(), n=_N_TEST):
    """Transitive facts drawn from the A→W 16-word vocab (so every content word is spike-spellable by BRIDGE-A). The
    fact CONTENT is restricted to words the reservoir sees as genuinely OPEN (a subject/object EMERGE-62 false-positived
    as closed -- e.g. 'cat' -- would be read as a function word by the reservoir -> a parse miss reflecting the closed-
    class-discovery precision, a SEPARATELY-characterized property, not the fully-spiking turn under test; the A→W cache
    still spells the excluded word, it is just not used as a fact filler). Mirrors EMERGE-90/95's open-word filter."""
    subjects = [w for w in AW._TRANS_SUBJECTS if w not in closed]
    objects = [w for w in AW._TRANS_OBJECTS if w not in closed]
    verbs = [v for v in AW._TRANS_VERBS_BARE if v not in closed and emerge_v3(v) not in closed]
    trng = np.random.default_rng(seed * 733 + 11)
    out, seen = [], set()
    guard = 0
    while len(out) < n and guard < 5000:
        guard += 1
        s = str(trng.choice(subjects)); vb = str(trng.choice(verbs)); o = str(trng.choice(objects))
        v3 = emerge_v3(vb)
        if (s, v3) in seen:
            continue
        seen.add((s, v3))
        out.append({"subj": s, "verb_bare": vb, "v3": v3, "obj": o, "sentence": ["the", s, v3, "the", o]})
    return out


def _build(seed, discovered, reg, composer_vocab, spell):
    """EMERGE-95's 3-slice shared bridge (reservoir + rf + slots) on cupy, but with the CALIBRATED producer + the A→W
    spiking spell. Mirrors m95._build_components; the only changes are the producer class (calibrated order) + spell."""
    enc = Encoder(discovered)
    rf_size = 2 * (_N_TEST + 3) * _D
    bridge, rf_base = m95._build_shared_bridge(seed, rf_size)
    comp = ReservoirComprehender(seed, discovered, res=m95.SharedBridgeReservoirLSM(enc.dim, seed, bridge), enc=enc)
    cq = DitransRegistryProducer(seed=seed, registry_slots=reg.registered_fits(), n_slot_pools=6, calibrate=True,
                                 shared_bridge=bridge, slot_region="slots")
    cq.learn()
    producer = RegistryBrocaProducer(cq, spell=spell)
    from research.runners.nav_conv_merged_bridge import MergedRFComposer
    composer = MergedRFComposer(bridge, rf_base, rf_size, seed=seed, D=_D, vocab=composer_vocab)
    return comp, composer, producer


def _speak(producer, subj, verb_bare, patient):
    return producer.speak(decision("ANSWER", construction="C_TRANS", subject=subj, verb=verb_bare, obj=patient))["surface"]


def _derisk_one(seed, aw_engine, aw_engine_lesion):
    reg = SVOConstructionRegistry(seed).build(build_stream_svo(seed))
    assert "C_TRANS" in reg.registered
    stream = m62.build_stream(seed, n_sentences=6000)
    words, freq, cover, _c = m62.compute_stats(stream)
    discovered, *_ = m62.discover_closed_class(words, freq, cover)
    subj_m62, verb_m62, obj_m62 = _content_pools(discovered)
    facts = _facts(seed, closed=set(discovered))
    composer_vocab = sorted(set(AW._TRANS_SUBJECTS) | set(AW._TRANS_OBJECTS) | set(emerge_v3(v) for v in AW._TRANS_VERBS_BARE))

    # THE FULLY-SPIKING TURN: reservoir + composer + producer on ONE cupy bridge; producer words via the A→W spell.
    comp, composer, producer = _build(seed, discovered, reg, composer_vocab, aw_engine.spell)
    comp.fit(_gen(_TRAIN_KINDS, _N_TRAIN_PER_CONSTRUCTION, np.random.default_rng(seed * 101 + 5), subj_m62, verb_m62, obj_m62))

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
        render_hit += int(_speak(producer, f["subj"], f["verb_bare"], patient)
                          == " ".join(["the", f["subj"], f["v3"], "the", f["obj"]]))
    recall = recall_hit / len(facts)
    render_exact = render_hit / len(facts)

    # MOAT (gate-first)
    prod_before = producer.production_count
    stored_keys = {(f["subj"], f["v3"]) for f in facts}
    fa = tot = 0; mguard = 0
    trng = np.random.default_rng(seed * 733 + 91)
    while tot < 30 and mguard < 5000:
        mguard += 1
        s = str(trng.choice(AW._TRANS_SUBJECTS)); v3q = emerge_v3(str(trng.choice(AW._TRANS_VERBS_BARE)))
        if (s, v3q) in stored_keys:
            continue
        tot += 1
        if composer.query_patient(s, v3q) is None:
            producer.speak(decision("ABSTAIN"))
        else:
            fa += 1
    moat_fa = fa / max(1, tot)
    moat_invoked_on_abstain = int(producer.production_count - prod_before)

    # CONTENT-LESION: zero the A→W content pool->language_output -> the words collapse (genuinely spiking)
    comp_l, composer_l, producer_l = _build(seed, discovered, reg, composer_vocab, aw_engine_lesion.spell)
    comp_l.fit(_gen(_TRAIN_KINDS, _N_TRAIN_PER_CONSTRUCTION, np.random.default_rng(seed * 101 + 5), subj_m62, verb_m62, obj_m62))
    for f in facts:
        pf = comp_l.comprehend(f["sentence"])
        if {"agent", "action", "patient"} <= set(pf):
            composer_l.store(pf["agent"], pf["action"], pf["patient"])
    les_hit = 0
    for f in facts:
        patient = composer_l.query_patient(f["subj"], f["v3"])
        if patient is None:
            continue
        les_hit += int(_speak(producer_l, f["subj"], f["verb_bare"], patient)
                       == " ".join(["the", f["subj"], f["v3"], "the", f["obj"]]))
    content_lesion_render = les_hit / len(facts)

    return {
        "seed": seed, "n_facts": len(facts),
        "parse_acc": parse_acc, "recall": recall, "render_exact_allword": render_exact,
        "moat_false_accept": moat_fa, "moat_producer_invoked_on_abstain": moat_invoked_on_abstain,
        "content_lesion_render": content_lesion_render,
    }


def _go(rows):
    def mean(k):
        return float(np.mean([r[k] for r in rows]))
    return {
        "n_seeds": len(rows),
        "parse_acc": mean("parse_acc"), "recall": mean("recall"), "render_exact_allword": mean("render_exact_allword"),
        "moat_false_accept": mean("moat_false_accept"),
        "moat_producer_invoked_on_abstain": int(max(r["moat_producer_invoked_on_abstain"] for r in rows)),
        "content_lesion_render": mean("content_lesion_render"),
        "go": (mean("parse_acc") >= 0.90 and mean("recall") >= 0.90 and mean("render_exact_allword") >= 0.90
               and mean("moat_false_accept") <= 0.05
               and max(r["moat_producer_invoked_on_abstain"] for r in rows) == 0
               and mean("content_lesion_render") <= 0.30),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    aw = AW.TransUnifiedSpell(load=True)
    aw_lesion = AW.TransUnifiedSpell(load=True, content_lesion=True)
    rows = []
    for s in args.seeds:
        d = _derisk_one(s, aw, aw_lesion)
        rows.append(d)
        print(f"[seed {s}] parse {d['parse_acc']:.3f} | recall {d['recall']:.3f} | render(ALL-WORD) "
              f"{d['render_exact_allword']:.3f} | moat-FA {d['moat_false_accept']:.3f} invoked {d['moat_producer_invoked_on_abstain']} | "
              f"content-lesion {d['content_lesion_render']:.3f}", flush=True)

    agg = _go(rows)
    verdict = "GO" if agg["go"] else "NO-GO"
    print(f"\n[fully-spiking-words] VERDICT: {verdict} -- the FULLY-SPIKING transitive turn on ONE cupy process: comprehend "
          f"(spiking reservoir) -> store/recall (spiking composer) -> SPEAK the answer with spiking ORDER + every WORD on "
          f"spikes. parse {agg['parse_acc']:.3f}; recall {agg['recall']:.3f}; all-word render {agg['render_exact_allword']:.3f}; "
          f"moat {agg['moat_false_accept']:.3f} FA + {agg['moat_producer_invoked_on_abstain']} invoked (gate-first); "
          f"content-lesion {agg['content_lesion_render']:.3f}.", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "agg": agg}, fh, indent=2)
        print(f"[fully-spiking-words] wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
