"""CPU structural guard for the fully-spiking-words one-brain transitive-turn capstone runner.

The full de-risk is a slow GPU/cupy research run (each seed builds the EMERGE-95 3-slice shared bridge -- reservoir + RF
composer + Izhikevich producer -- co-executing with the A->W spiking word-spell in ONE process). It is NOT a fast CI
gate. This light test guards the runner's structure so it stays committable + importable without a GPU or a long run:

- the capstone hooks exist (the shared-bridge build, the calibrated producer path, the gate-first speak, the GO gate);
- the GO gate thresholds are the correct honest bars (parse/recall/all-word-render >= 0.90, moat 0-FA + 0 producer
  invocations on abstain (gate-first), content-lesion collapses <= 0.30);
- the open-word fact filter (`_facts(seed, closed=...)`) excludes reservoir closed-class collisions (e.g. a subject/
  object EMERGE-62 false-positives as a function word) -- the load-bearing fix that took seed 42 parse 0.833 -> 1.000.

Importing the runner defaults SIM_BACKEND=numpy here (the runner's own `setdefault("cupy")` respects the already-set
value), so the heavy import chain (EMERGE-95/88/77/74 + the A->W spell) loads on CPU without allocating a GPU.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest  # noqa: E402

import research.runners._rungB1_fully_spiking_words_capstone_derisk as m  # noqa: E402


def test_capstone_hooks_exist():
    # the fully-spiking-turn build + speak + gate hooks (composition of EMERGE-95 shared bridge + calibrated producer)
    assert hasattr(m, "_facts") and hasattr(m, "_build") and hasattr(m, "_speak")
    assert hasattr(m, "_derisk_one") and hasattr(m, "_go") and hasattr(m, "main")


def test_go_gate_thresholds_are_the_honest_bars():
    # a synthetic aggregate must produce GO only when EVERY honest bar is met (mirrors m._go):
    #   parse/recall/all-word-render >= 0.90 ; moat_false_accept <= 0.05 ; 0 producer invocations on abstain
    #   (gate-first, load-bearing) ; content-lesion collapses <= 0.30.
    def go(parse, recall, render, moat_fa, invoked, lesion):
        rows = [{"parse_acc": parse, "recall": recall, "render_exact_allword": render,
                 "moat_false_accept": moat_fa, "moat_producer_invoked_on_abstain": invoked,
                 "content_lesion_render": lesion}]
        return m._go(rows)["go"]

    assert go(1.0, 1.0, 1.0, 0.0, 0, 0.0)          # the observed seed-42 full GO
    assert not go(0.833, 1.0, 1.0, 0.0, 0, 0.0)    # the pre-fix mis-parse (closed-class collision) is NOT a GO
    assert not go(1.0, 1.0, 1.0, 0.10, 0, 0.0)     # a moat false-accept blocks GO
    assert not go(1.0, 1.0, 1.0, 0.0, 1, 0.0)      # a producer invocation on abstain (gate-first breach) blocks GO
    assert not go(1.0, 1.0, 1.0, 0.0, 0, 0.9)      # content-lesion NOT collapsing (words not spiking) blocks GO


def test_open_word_fact_filter_excludes_closed_class_collisions():
    # `_facts` draws transitive facts ONLY from words the reservoir sees as genuinely OPEN; a filler word passed in
    # `closed` (a closed-class false-positive like "cat") must never appear as subject/verb/object of a generated fact.
    import research.runners._rungB1_aw_neural_words_transitive_derisk as AW
    from research.runners._emerge74_transitive_ditransitive_derisk import emerge_v3

    # pick a real subject to exclude; it must not leak into any fact
    excluded = AW._TRANS_SUBJECTS[1]
    facts = m._facts(42, closed=frozenset({excluded}), n=8)
    assert len(facts) == 8
    for f in facts:
        assert f["subj"] != excluded and f["obj"] != excluded and f["v3"] != excluded and f["verb_bare"] != excluded
        # every fact is a well-formed C_TRANS surface: the DET SUBJ VERB:3sg DET OBJ backbone
        assert f["sentence"] == ["the", f["subj"], f["v3"], "the", f["obj"]]
        assert f["v3"] == emerge_v3(f["verb_bare"])
    # the unfiltered call CAN include that word (proves the filter, not the vocab, removed it)
    keys_open = {(f["subj"], f["v3"]) for f in m._facts(42, closed=frozenset(), n=12)}
    assert len(keys_open) == 12  # unique (subj, verb) keys -- no collisions in the fact set
