"""COMMUNICABLE TURN -- STAGE B: the PRODUCTION-AGENT WIRE-IN + Q-PERSISTENCE validator.

Stage A (`_communicable_turn_stageA_derisk.py`, GO 3-seed) validated the fused `CommunicableTurn` (the 3
mechanisms + the moat + the channels compose, every invariant holds). Stage B WIRES it into the production agent
as an OPT-IN mode (default-OFF = byte-identical to today) and PERSISTS the learned talkativeness Q into the
developed-brain bundle (so the talkativeness learned from interaction carries across sessions -- the develop-loop
tie-in). This runner VALIDATES the wire-in end-to-end:

  (1) DEFAULT-OFF BYTE-IDENTITY: a `BrainConversationalAgent` / `MultiTurnAgent` built with the DEFAULT (no
      communicable_mode) NEVER constructs the orchestrator (no corpus pass, no proposer/accumulator build) and its
      existing API (hear / what_does / who_does / is_it_true) behaves EXACTLY as before. (The pytest suites
      `tests/test_brain_conversational_agent.py` + `tests/test_multi_turn_agent.py` are the authoritative
      byte-identity check -- they pass verbatim with the default; this runner re-asserts the no-build invariant +
      a couple of behavioural anchors programmatically.)

  (2) COMMUNICABLE-MODE-ON GATE REPRODUCES THROUGH THE PRODUCTION AGENT: with communicable_mode=True, the agent's
      `converse(msg, ...)` reproduces the Stage A invariants ON THE PRODUCTION AGENT'S OWN composer/moat --
        - a KNOWN-fact question on a STORED cue answers CERTAIN; an unstored cue ABSTAINS (the no-confab moat);
        - an OPINION on a grounded topic emits a NOVEL + topic-relevant + FLAGGED (never-certain, never-stored)
          hypothesis; the shuffled-PPMI-graph control collapses groundedness;
        - perceived FEEDBACK ('elaborate') RAISES the next-turn talkativeness there (the brain's three-factor
          plasticity), and the DA-LESION ABOLISHES that rise (the change is the brain's reward system).

  (3) Q-PERSISTENCE: teach a topic -> save the developed brain -> load it -> the talkativeness Q for that topic
      PERSISTS (the bundle round-trips the learned speak-value). A FROZEN control (no further feedback) keeps Q
      stable.  All via `developed_brain_io.save_developed_brain` / `load_developed_brain` (the standard bundle).

THE MOAT (HARD, throughout): 0 known-fact-channel leaks; every novel emission FLAGGED + NEVER stored; an unstored
cue ABSTAINS.  If the wire-in REGRESSES the existing behaviour (default-OFF not byte-identical) OR the moat leaks
in communicable-mode, this reports it PRECISELY and the verdict is HONEST_NEGATIVE -- it does NOT fake a GO.

DRAW SELECTOR (owner-steer #3): this VALIDATION GATE uses the HOST-oracle generative draw (`--draw host`, default)
-- the SAME PPMI likelihood as the spiking draw, fast on CPU. The spiking draw (`--draw spiking`) is the
DEFAULT-ON PRODUCTION path but is ~40s/topic in the fused turn on CPU (the megakernel perf lever is Stage C), so
gating on it 3-seed would be intractable; it is separately GO 6-seed (`_followon2_spiking_wta_sampler_derisk`).
The load-bearing SPIKING speak DECISION (the SpikingSpeakAccumulator) stays spiking in BOTH draws -- the
brain-based speak choice is unchanged.  CPU (`SIM_BACKEND=numpy`); reuse-by-import; NO `sim/` edit.

Run:
  SIM_BACKEND=numpy python -u -m research.runners._communicable_turn_stageB_wirein \
      --seeds 42,43,44 --draw host --out research/findings/raw/_communicable_turn_stageB_wirein.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from collections import defaultdict

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402
from research.runners.multi_turn_agent import MultiTurnAgent  # noqa: E402
from research.runners._communicable_turn_stageA_derisk import (  # noqa: E402
    build_communicable_brain,
    _default_corpus,
)
from research.runners._genfrontier_b2_generative_replay_derisk import shuffle_graph  # noqa: E402
from research.runners.option_c_real_cooccurrence_derisk import (  # noqa: E402
    TAXONOMY_8x8,
    taxonomy_to_vocab_categories,
)
from research.runners.developed_brain_io import (  # noqa: E402
    save_developed_brain,
    load_developed_brain,
)


# ============================================================================================================
# (1) DEFAULT-OFF BYTE-IDENTITY -- the orchestrator is NEVER built at the default; the existing API is unchanged.
# ============================================================================================================
_BYTE_ID_NOUNS = ["dog", "cat", "fish", "bird", "worm", "ball"]
_BYTE_ID_VOCAB = _BYTE_ID_NOUNS + ["chase", "eat", "go", "come", "north", "south"]


def check_default_off_byte_identity(seed):
    """A DEFAULT agent must (a) NOT have a built communicable orchestrator, (b) report communicable_mode False, and
    (c) behave EXACTLY as before on hear/what_does/who_does/is_it_true. (The full byte-identity guarantee is the
    pytest suites; this asserts the no-build invariant + the behavioural anchors the wire-in could have disturbed.)

    Uses an explicit small vocab (the SAME pattern as tests/test_multi_turn_agent.py -- per-seed random codes, no
    denoise64 cache dependency) so the byte-identity check runs anywhere on numpy-CPU."""
    concepts = {w: None for w in _BYTE_ID_VOCAB}
    a = BrainConversationalAgent(seed=seed, concepts=concepts, composer_kind="rf", enable_neural_render=False)
    no_build = (a.communicable_mode is False) and (getattr(a, "_communicable", "MISSING") is None) \
        and (getattr(a, "_communicable_brain", "MISSING") is None)
    # behaviour anchors: store + recall + abstain (the byte-identical default path)
    a.composer.kb = []
    a.hear("dog go north")
    a.hear("cat come south", polarity="AFFIRM")
    qa_ok = (a.what_does("dog", "go") == "north") and (a.who_does("come", "south") == "cat") \
        and (a.what_does("bird", "eat") is None) and (a.is_it_true("cat", "come", "south") == "yes")
    # MultiTurnAgent default-off pass-through
    mt = MultiTurnAgent(referent_concepts=_BYTE_ID_NOUNS, concepts=concepts, seed=seed)
    mt_no_build = (mt.communicable_mode is False) and (getattr(mt.agent, "_communicable", "MISSING") is None)
    # calling converse() at the default must RAISE (the orchestrator is intentionally not built)
    raised = False
    try:
        a.converse("hi")
    except RuntimeError:
        raised = True
    return {"orchestrator_not_built": bool(no_build), "default_api_unchanged": bool(qa_ok),
            "multiturn_not_built": bool(mt_no_build), "converse_raises_when_off": bool(raised),
            "ok": bool(no_build and qa_ok and mt_no_build and raised)}


# ============================================================================================================
# (2) COMMUNICABLE-MODE-ON GATE THROUGH THE PRODUCTION AGENT -- the Stage A invariants on the agent's OWN composer.
# ============================================================================================================
def check_communicable_on(seed, vocab, corpus, draw, n_attempts, advantage_bar, n_rounds):
    """Build a PRODUCTION agent with communicable_mode=True over a curriculum of STORED facts, then drive its
    `converse()` across the four channels + measure the Stage A invariants IN THE PRODUCTION PATH."""
    # Build a self-contained communicable brain to discover the affirmed/negated facts + grounded topics for THIS
    # seed (the SAME construction the agent uses internally), so we can teach the SAME curriculum to the agent.
    ref = build_communicable_brain(seed=seed, host_oracle_sampler=(draw == "host"), corpus=corpus,
                                   n_attempts=n_attempts)
    affirmed, negated = ref["affirmed"], ref["negated"]
    turn_ref = ref["turn"]
    grounded = [t for t in ref["topic_pool"] if turn_ref.propose_candidates_about(t, n_attempts=n_attempts)]
    if len(grounded) < 6 or not affirmed:
        return {"insufficient": True, "n_grounded": len(grounded), "n_affirmed": len(affirmed)}

    # The PRODUCTION agent: a fresh BrainConversationalAgent over the full taxonomy vocab, communicable_mode=True.
    # We TEACH it the curriculum into its OWN composer (the production fact store + the same no-confab moat), so the
    # known-fact channel reads the agent's own facts. Facts go in via composer.store (the production demos' path;
    # equivalent to hear() for the known-fact/moat invariants Stage B validates, and avoids the per-fact Hebbian
    # parser comprehension cost x ~36 facts x 2 agents/seed -- the parser is orthogonal to the communicable
    # orchestration this gate exercises; agent.hear() is separately covered byte-identical in (1)).
    concepts = {w: None for w in vocab}
    agent = BrainConversationalAgent(seed=seed, concepts=concepts, composer_kind="rf", enable_neural_render=False,
                                     communicable_mode=True, communicable_draw=draw,
                                     communicable_config={"n_attempts": n_attempts})
    agent.composer.kb = []
    for ag, ac, pt in affirmed:
        agent.composer.store(ag, ac, pt, polarity="AFFIRM")
    for ag, ac, pt in negated:
        agent.composer.store(ag, ac, pt, polarity="NEGATE")
    # building the orchestrator now reads the agent's OWN composer (its facts) + the same moat.
    agent._ensure_communicable()

    # --- (A) KNOWN-FACT channel through converse(): stored cue CERTAIN, unstored cue ABSTAIN ---
    cue_to_pat = defaultdict(set)
    for ag, ac, pt in affirmed:
        cue_to_pat[(ag, ac)].add(pt)
    known_ok = 0
    known_total = 0
    for (ag, ac), pats in cue_to_pat.items():
        r = agent.converse(f"what does {ag} {ac}?", cue=(ag, ac))
        known_total += 1
        if (r["channel"] == "known" and r["certain"] and not r["abstained"]
                and r["recalled_svo"] is not None and r["recalled_svo"][2] in pats):
            known_ok += 1
    # an unstored cue must ABSTAIN (the moat). pick an (agent, action) pair NEVER stored.
    all_cues = set(cue_to_pat.keys())
    agents_pool = sorted({a_ for a_, _, _ in affirmed})
    actions_pool = sorted({v_ for _, v_, _ in affirmed})
    unstored_cue = None
    for ag in agents_pool:
        for ac in actions_pool:
            if (ag, ac) not in all_cues:
                unstored_cue = (ag, ac); break
        if unstored_cue:
            break
    abstain_ok = True
    if unstored_cue is not None:
        ru = agent.converse(f"what does {unstored_cue[0]} {unstored_cue[1]}?", cue=unstored_cue)
        abstain_ok = bool(ru["abstained"] and ru["recalled_svo"] is None and ru["certain"])
    known_channel_ok = (known_total > 0) and (known_ok == known_total) and abstain_ok

    # --- (B) NOVEL channel through converse(): emit NOVEL + topic-relevant + FLAGGED, never stored ---
    all_stored = set(affirmed) | set(negated)
    n_facts_before = len(agent.composer.kb)
    opinion_recs = [agent.converse(f"what do you think about {t}?", topic=t, n_attempts=n_attempts)
                    for t in grounded]
    emitted = [r for r in opinion_recs if r.get("emitted")]
    n_emit = len(emitted)
    all_novel = (n_emit > 0) and all(tuple(r["proposed_triple"]) not in all_stored for r in emitted)
    all_topic_rel = (n_emit > 0) and all(r.get("topic_in_proposition") for r in emitted)
    all_flagged = (n_emit > 0) and all(r.get("hedge") is not None and r.get("certain") is False for r in emitted)
    # the novel channel NEVER stored: the composer kb is unchanged by the opinions
    store_unchanged = (len(agent.composer.kb) == n_facts_before)

    # --- MOAT (HARD): every emitted novel proposition still ABSTAINS on the known-fact channel ---
    moat_leaks = 0
    for r in emitted:
        a_, v_, p_ = r["proposed_triple"]
        if agent.what_does(a_, v_) == p_:
            moat_leaks += 1
        if agent.is_it_true(a_, v_, p_) == "yes":
            moat_leaks += 1
    # the stored facts STILL answer (positive control)
    from collections import Counter
    cue_count = Counter((ag, ac) for ag, ac, _ in affirmed)
    unique_cue_facts = [(ag, ac, pt) for ag, ac, pt in affirmed if cue_count[(ag, ac)] == 1]
    stored_still_answer = all(agent.is_it_true(ag, ac, pt) == "yes" for ag, ac, pt in affirmed) and \
        all(agent.what_does(ag, ac) == pt for ag, ac, pt in unique_cue_facts)
    moat_ok = (moat_leaks == 0) and all_flagged and store_unchanged and stored_still_answer

    # --- shuffled-PPMI-graph anti-cheat: the emitted propositions' groundedness collapses on a shuffled graph ---
    P, row = ref["P"], ref["row"]
    tau_pct = 50.0
    pos = P[P > 0]
    tau = float(np.percentile(pos, tau_pct)) if pos.size else 0.0
    P_shuf = shuffle_graph(P, np.random.default_rng(seed * 17 + 5))
    pos_s = P_shuf[P_shuf > 0]
    tau_s = float(np.percentile(pos_s, tau_pct)) if pos_s.size else 0.0
    emit_triples = [tuple(r["proposed_triple"]) for r in emitted]

    def _plausible(Pm, taum, tp):
        a_, ac_, p_ = tp
        return (Pm[row[a_], row[ac_]] >= taum) and (Pm[row[ac_], row[p_]] >= taum)

    true_pass = sum(1 for tp in emit_triples if _plausible(P, tau, tp))
    shuf_pass = sum(1 for tp in emit_triples if _plausible(P_shuf, tau_s, tp))
    true_frac = true_pass / max(1, len(emit_triples))
    shuf_frac = shuf_pass / max(1, len(emit_triples))
    grounded_adv = true_frac / max(shuf_frac, 1.0 / max(1, len(emit_triples)))
    grounded_ok = (len(emit_triples) > 0) and (grounded_adv >= advantage_bar)

    # --- (C) FEEDBACK raises talkativeness + the DA-lesion abolishes it (through the production agent) ---
    fb_topic = grounded[0]
    q_before = agent.speak_value_Q().get(fb_topic, 0.0)
    # the teaching path through converse() ('tell me more' -> classified teaching -> three-factor Q update)
    teach_rec = agent.converse("tell me more", topic=fb_topic, n_attempts=n_attempts)
    teaching_path_ok = (teach_rec.get("channel") == "teaching" and teach_rec.get("polarity") == +1
                        and teach_rec.get("feedback_topic") == fb_topic)
    # accumulate the remaining rounds via the cheap direct feedback (the rerun's spiking decide is UX, not the
    # Q-rise measurement, which reads speak_value_Q()).
    for _ in range(n_rounds - 1):
        agent.communicable_feedback(fb_topic, +1)
    q_after = agent.speak_value_Q().get(fb_topic, 0.0)
    feedback_raises = q_after > q_before + 1e-9
    # the negative 'stop' lowers it back
    stop_rec = agent.converse("that's enough", topic=fb_topic, n_attempts=n_attempts)
    stop_path_ok = (stop_rec.get("channel") == "teaching" and stop_rec.get("polarity") == -1)
    for _ in range(n_rounds - 1):
        agent.communicable_feedback(fb_topic, -1)
    q_after_stop = agent.speak_value_Q().get(fb_topic, 0.0)
    negative_lowers = q_after_stop < q_after - 1e-9

    # DA-LESION: a fresh production agent; the SAME teaching round with DA pinned does NOT raise Q.
    agent_les = BrainConversationalAgent(seed=seed, concepts=concepts, composer_kind="rf",
                                         enable_neural_render=False, communicable_mode=True,
                                         communicable_draw=draw, communicable_config={"n_attempts": n_attempts})
    agent_les.composer.kb = []
    for ag, ac, pt in affirmed:
        agent_les.composer.store(ag, ac, pt, polarity="AFFIRM")
    agent_les._ensure_communicable()
    q_les_before = agent_les.speak_value_Q().get(fb_topic, 0.0)
    for _ in range(n_rounds):
        agent_les.communicable_feedback(fb_topic, +1, lesion_DA=True)   # SNc lesioned -> no learning
    q_les_after = agent_les.speak_value_Q().get(fb_topic, 0.0)
    lesion_abolishes = abs(q_les_after - q_les_before) < 1e-12

    # phatic: a non-factual reply, makes no claim
    phatic_rec = agent.converse("hi")
    phatic_ok = (phatic_rec.get("channel") == "phatic" and phatic_rec.get("reply")
                 and not phatic_rec.get("is_factual_claim", True))

    rec = {
        "n_affirmed": len(affirmed), "n_negated": len(negated), "n_grounded": len(grounded),
        # (A) known-fact
        "known_ok": known_ok, "known_total": known_total, "abstain_ok": bool(abstain_ok),
        "known_channel_ok": bool(known_channel_ok),
        # (B) novel
        "n_emitted": n_emit, "all_novel": bool(all_novel), "all_topic_relevant": bool(all_topic_rel),
        "all_flagged": bool(all_flagged), "store_unchanged": bool(store_unchanged),
        "grounded_advantage": grounded_adv, "grounded_ok": bool(grounded_ok),
        # moat
        "moat_leaks": moat_leaks, "stored_still_answer": bool(stored_still_answer), "moat_ok": bool(moat_ok),
        # (C) feedback + lesion
        "feedback_topic": fb_topic, "q_before": q_before, "q_after_teach": q_after, "q_after_stop": q_after_stop,
        "feedback_raises": bool(feedback_raises), "negative_lowers": bool(negative_lowers),
        "teaching_path_ok": bool(teaching_path_ok), "stop_path_ok": bool(stop_path_ok),
        "lesion_q_before": q_les_before, "lesion_q_after": q_les_after, "lesion_abolishes": bool(lesion_abolishes),
        # phatic
        "phatic_ok": bool(phatic_ok),
        "insufficient": False,
    }
    rec["all_ok"] = bool(known_channel_ok and all_novel and all_topic_rel and all_flagged and store_unchanged
                         and grounded_ok and moat_ok and feedback_raises and negative_lowers
                         and teaching_path_ok and stop_path_ok and lesion_abolishes and phatic_ok)
    return rec


# ============================================================================================================
# (3) Q-PERSISTENCE -- teach a topic -> SAVE the developed brain -> LOAD it -> the talkativeness Q PERSISTS.
# ============================================================================================================
def check_q_persistence(seed, vocab, corpus, draw, n_attempts, n_rounds):
    """Teach a topic's talkativeness through the production agent, SAVE the developed brain (the bundle), LOAD it
    back (communicable_mode=True), and assert the talkativeness Q for that topic PERSISTS. A FROZEN control (no
    further feedback after load) keeps Q stable."""
    ref = build_communicable_brain(seed=seed, host_oracle_sampler=(draw == "host"), corpus=corpus,
                                   n_attempts=n_attempts)
    affirmed = ref["affirmed"]
    turn_ref = ref["turn"]
    grounded = [t for t in ref["topic_pool"] if turn_ref.propose_candidates_about(t, n_attempts=n_attempts)]
    if len(grounded) < 1 or not affirmed:
        return {"insufficient": True}
    fb_topic = grounded[0]

    concepts = {w: None for w in vocab}
    agent = BrainConversationalAgent(seed=seed, concepts=concepts, composer_kind="rf", enable_neural_render=False,
                                     communicable_mode=True, communicable_draw=draw,
                                     communicable_config={"n_attempts": n_attempts})
    agent.composer.kb = []
    for ag, ac, pt in affirmed:
        agent.composer.store(ag, ac, pt, polarity="AFFIRM")
    # TEACH the topic (raise its talkativeness)
    for _ in range(n_rounds):
        agent.communicable_feedback(fb_topic, +1)
    Q_taught = agent.speak_value_Q()
    q_taught_topic = Q_taught.get(fb_topic, 0.0)

    # SAVE the developed brain to a temp bundle
    tmpdir = tempfile.mkdtemp(prefix="commB_qpersist_")
    bundle = os.path.join(tmpdir, "brain")
    manifest = save_developed_brain(agent, bundle, seed=seed, composer_kind="rf")
    n_in_manifest = manifest.get("n_speak_value_Q", 0)
    # the file was written
    q_file = os.path.join(bundle, "speak_value_Q.json")
    file_written = os.path.exists(q_file)

    # LOAD it back with communicable_mode=True
    agent2, manifest2 = load_developed_brain(bundle, communicable_mode=True, communicable_draw=draw)
    # configure the same n_attempts on the reload so the orchestrator builds identically
    agent2._communicable_config["n_attempts"] = n_attempts
    Q_loaded = agent2.speak_value_Q()                 # this seeds + reads the restored Q
    q_loaded_topic = Q_loaded.get(fb_topic, 0.0)

    persisted = abs(q_loaded_topic - q_taught_topic) < 1e-6 and q_loaded_topic > 1e-9
    # the full Q dict round-trips (every taught entry matches)
    full_match = all(abs(Q_loaded.get(t, 0.0) - q) < 1e-6 for t, q in Q_taught.items()) and \
        (set(Q_taught.keys()) == set(Q_loaded.keys()) or
         all(abs(Q_loaded.get(t, 0.0) - q) < 1e-6 for t, q in Q_taught.items()))

    # FROZEN control: with no further feedback, the loaded Q is stable across a build + a read
    agent2._ensure_communicable()
    q_frozen = agent2.speak_value_Q().get(fb_topic, 0.0)
    frozen_stable = abs(q_frozen - q_loaded_topic) < 1e-9

    # cleanup
    import shutil
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass

    return {
        "insufficient": False, "feedback_topic": fb_topic,
        "q_taught_topic": q_taught_topic, "q_loaded_topic": q_loaded_topic,
        "n_speak_value_Q_in_manifest": n_in_manifest, "file_written": bool(file_written),
        "persisted": bool(persisted), "full_dict_match": bool(full_match), "frozen_stable": bool(frozen_stable),
        "ok": bool(persisted and full_match and frozen_stable and file_written and n_in_manifest > 0),
    }


# ============================================================================================================
# Per-seed orchestration.
# ============================================================================================================
def run_seed(seed, vocab, corpus, a):
    t0 = time.time()
    byte_id = check_default_off_byte_identity(seed)
    on = check_communicable_on(seed, vocab, corpus, a.draw, a.n_attempts, a.advantage_bar, a.n_rounds)
    qp = check_q_persistence(seed, vocab, corpus, a.draw, a.n_attempts, a.n_rounds)
    elapsed = time.time() - t0

    print(f"\n[stageB seed {seed}] elapsed {elapsed:.1f}s", flush=True)
    print(f"  (1) DEFAULT-OFF byte-identity: {byte_id['ok']} "
          f"(no-build {byte_id['orchestrator_not_built']}, api-unchanged {byte_id['default_api_unchanged']}, "
          f"mt-no-build {byte_id['multiturn_not_built']}, converse-raises {byte_id['converse_raises_when_off']})",
          flush=True)
    if on.get("insufficient"):
        print(f"  (2) COMMUNICABLE-ON: INSUFFICIENT (grounded {on.get('n_grounded')}, affirmed {on.get('n_affirmed')})",
              flush=True)
    else:
        print(f"  (2) COMMUNICABLE-ON gate: {on['all_ok']}", flush=True)
        print(f"      known {on['known_ok']}/{on['known_total']} + abstain {on['abstain_ok']} -> "
              f"{on['known_channel_ok']} | novel emit {on['n_emitted']} (novel {on['all_novel']}, topic-rel "
              f"{on['all_topic_relevant']}, flagged {on['all_flagged']}, never-stored {on['store_unchanged']})",
              flush=True)
        print(f"      MOAT {on['moat_leaks']} leaks (stored-answer {on['stored_still_answer']}) -> {on['moat_ok']} | "
              f"shuffled-graph adv {on['grounded_advantage']:.1f}x ({on['grounded_ok']})", flush=True)
        print(f"      FEEDBACK Q[{on['feedback_topic']}] {on['q_before']:.3f} -> teach {on['q_after_teach']:.3f} "
              f"({on['feedback_raises']}) -> stop {on['q_after_stop']:.3f} (neg-lowers {on['negative_lowers']}) | "
              f"LESION {on['lesion_q_before']:.3f}->{on['lesion_q_after']:.3f} ({on['lesion_abolishes']})", flush=True)
    if qp.get("insufficient"):
        print(f"  (3) Q-PERSISTENCE: INSUFFICIENT", flush=True)
    else:
        print(f"  (3) Q-PERSISTENCE: {qp['ok']} (taught Q[{qp['feedback_topic']}]={qp['q_taught_topic']:.3f} -> "
              f"loaded {qp['q_loaded_topic']:.3f}; persisted {qp['persisted']}, full-dict {qp['full_dict_match']}, "
              f"frozen-stable {qp['frozen_stable']}, manifest-n {qp['n_speak_value_Q_in_manifest']})", flush=True)

    return {"seed": seed, "elapsed_s": elapsed, "byte_identity": byte_id, "communicable_on": on,
            "q_persistence": qp}


def decide_verdict(rows):
    """STAGE B GO iff, across ALL seeds: (1) default-OFF byte-identity holds (the orchestrator is never built, the
    existing API is unchanged); (2) the communicable-mode-ON gate reproduces through the production agent (known
    CERTAIN/abstain, novel flagged/never-stored, 0 moat leaks, feedback raises, lesion abolishes); (3) the
    talkativeness Q persists across a bundle save->load. Else HONEST_NEGATIVE + the precise failing piece."""
    usable = [r for r in rows if not r["communicable_on"].get("insufficient")
              and not r["q_persistence"].get("insufficient")]
    if not usable:
        return "INVALID_insufficient_grounded_topics", {"note": "fewer than 6 grounded topics / no facts in every seed"}

    byte_all = all(r["byte_identity"]["ok"] for r in rows)
    on_all = all(r["communicable_on"]["all_ok"] for r in usable)
    moat_all = all(r["communicable_on"]["moat_ok"] for r in usable)
    qp_all = all(r["q_persistence"]["ok"] for r in usable)

    detail = {
        "n_seeds": len(rows),
        "n_usable_seeds": len(usable),
        "default_off_byte_identity_all_seeds": bool(byte_all),
        "communicable_on_gate_all_seeds": bool(on_all),
        "moat_ok_all_seeds": bool(moat_all),
        "moat_leaks_total": int(sum(r["communicable_on"].get("moat_leaks", 0) for r in usable)),
        "q_persistence_all_seeds": bool(qp_all),
        "n_emitted_mean": float(np.mean([r["communicable_on"]["n_emitted"] for r in usable])),
        "grounded_advantage_min": float(min(r["communicable_on"]["grounded_advantage"] for r in usable)),
        "feedback_raises_all_seeds": bool(all(r["communicable_on"]["feedback_raises"] for r in usable)),
        "lesion_abolishes_all_seeds": bool(all(r["communicable_on"]["lesion_abolishes"] for r in usable)),
    }

    if not byte_all:
        verdict = "HONEST_NEGATIVE_default_off_not_byte_identical"   # the wire-in regressed the default path
    elif not moat_all:
        verdict = "HONEST_NEGATIVE_moat_leak_in_communicable_mode"   # the load-bearing safety invariant
    elif not on_all:
        verdict = "HONEST_NEGATIVE_communicable_gate_regressed_through_agent"
    elif not qp_all:
        verdict = "HONEST_NEGATIVE_talkativeness_Q_does_not_persist"
    else:
        verdict = "GO"
    return verdict, detail


def main():
    p = argparse.ArgumentParser(description="Communicable turn -- Stage B: validate the production-agent wire-in "
                                            "(default-OFF byte-identity + the communicable-mode-ON gate) + the "
                                            "talkativeness-Q persistence into the developed-brain bundle.")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--draw", choices=["host", "spiking"], default="host",
                   help="generative DRAW for this VALIDATION GATE: 'host' (default, fast on CPU; same PPMI "
                        "likelihood) or 'spiking' (the production default, ~40s/topic on CPU). The load-bearing "
                        "SPIKING speak DECISION stays spiking in both.")
    p.add_argument("--n-attempts", type=int, default=500, help="generative-replay samples per topic")
    p.add_argument("--n-rounds", type=int, default=12, help="feedback rounds for the talkativeness teach")
    p.add_argument("--advantage-bar", type=float, default=3.0, help="grounded shuffled-graph advantage ratio bar")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    logging.getLogger().setLevel(logging.WARNING)
    for nm in ("SIM_BRIDGE", "sim", "sim.bridge"):
        logging.getLogger(nm).setLevel(logging.WARNING)

    seeds = [int(s.strip()) for s in a.seeds.split(",")]
    t0 = time.time()
    print(f"[stageB] seeds={seeds} draw={a.draw} -- validate the production-agent communicable wire-in "
          f"(default-OFF byte-identity + communicable-ON gate + Q-persistence).", flush=True)

    vocab, cat_ids, _cat_names = taxonomy_to_vocab_categories(TAXONOMY_8x8)
    corpus = _default_corpus(vocab, cat_ids)             # the SAME TinyStories PPMI corpus Stage A uses (cached)

    rows = [run_seed(s, vocab, corpus, a) for s in seeds]
    verdict, detail = decide_verdict(rows)

    print(f"\n{'='*100}", flush=True)
    print(f"  STAGE B VERDICT: {verdict}", flush=True)
    print(f"  (1) DEFAULT-OFF byte-identity all seeds: {detail.get('default_off_byte_identity_all_seeds')}", flush=True)
    print(f"  (2) COMMUNICABLE-ON gate all seeds: {detail.get('communicable_on_gate_all_seeds')} "
          f"(moat {detail.get('moat_ok_all_seeds')}, {detail.get('moat_leaks_total')} leaks; novel emit mean "
          f"{detail.get('n_emitted_mean', float('nan')):.1f}; shuffled-graph adv min "
          f"{detail.get('grounded_advantage_min', float('nan')):.1f}x; feedback-raises "
          f"{detail.get('feedback_raises_all_seeds')}; lesion-abolishes {detail.get('lesion_abolishes_all_seeds')})",
          flush=True)
    print(f"  (3) Q-PERSISTENCE all seeds: {detail.get('q_persistence_all_seeds')}", flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*100}\n", flush=True)

    out = {
        "probe": "communicable_turn_stageB_wirein",
        "verdict": verdict,
        "seeds": seeds,
        "stage": "B -- the production-agent wire-in (default-OFF byte-identity) + talkativeness-Q persistence",
        "stage_a": "research/findings/raw/_communicable_turn_stageA_derisk.json (GO 3-seed -- the fused CommunicableTurn)",
        "wire_in": {
            "agent_opt_in": "BrainConversationalAgent(communicable_mode=True[, communicable_draw, communicable_config, "
                            "speak_value_Q]) -> converse(msg, cue=, topic=) + communicable_feedback(topic, polarity) "
                            "+ speak_value_Q(); enable_communicable_mode() at runtime. DEFAULT OFF = byte-identical "
                            "(the CommunicableTurn is NEVER constructed).",
            "multiturn_opt_in": "MultiTurnAgent passes communicable_mode through to the inner agent + delegates "
                                "converse / communicable_feedback / speak_value_Q.",
            "factory": "research/runners/_communicable_turn_stageA_derisk.build_communicable_brain (the Stage A "
                       "brain-assembly hoisted into a reusable builder; the agent attaches it over its OWN composer).",
            "persistence": "research/runners/developed_brain_io.py: save_developed_brain writes speak_value_Q.json "
                           "(the learned-talkativeness Q) + a manifest n_speak_value_Q field; load_developed_brain "
                           "reads it back + seeds the rebuilt agent's CommunicableTurn (carries talkativeness across "
                           "sessions -- the develop-loop tie-in).",
            "draw_selector": ("owner-steer #3: communicable_draw in {'spiking' (production default), 'host' "
                              "(fast-interactive / numpy-CPU / test oracle)}. This validation gate uses --draw "
                              f"{a.draw}. The spiking draw is ~40s/topic in the fused turn on CPU (the megakernel "
                              "perf lever is Stage C); the load-bearing SPIKING speak DECISION stays spiking in both."),
            "no_sim_edit": True,
        },
        "config": {"draw": a.draw, "n_attempts": a.n_attempts, "n_rounds": a.n_rounds,
                   "advantage_bar": a.advantage_bar},
        "stage_b_gate": (
            "GO = (1) default-OFF byte-identity (the orchestrator is never built; the existing API + the pytest "
            "suites are unchanged); (2) the communicable-mode-ON gate reproduces THROUGH the production agent "
            "(known CERTAIN + abstain, novel flagged + never-stored + grounded, 0 moat leaks, feedback raises + "
            "the DA-lesion abolishes); (3) the talkativeness Q persists across a developed-brain bundle save->load "
            "(+ a frozen control). NEVER weakens the no-confab moat."),
        "detail": detail,
        "per_seed": rows,
        "elapsed_total_s": time.time() - t0,
    }
    if a.out is None:
        a.out = os.path.join(_REPO, "research", "findings", "raw", "_communicable_turn_stageB_wirein.json")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {a.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
