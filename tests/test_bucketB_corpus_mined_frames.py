"""Regression guard for Bucket-B B-mine-1 -- a CORPUS-MINED verb-frame LEXICON.

The B1-for-relations -> B1-for-frames step: the hand-authored FRAME_LEXICON (go->GOAL, give->THEME+RECIPIENT, ...)
is DERIVED from corpus argument co-occurrence over the brain's OWN learned verbs -- structure ACQUIRED, not given.
De-risked GO (research/findings/2026-06-27-burndown-Bmine1-corpus-mined-frames-GO.md). This guard pins:

  - the DERIVE LOGIC on a fixed synthetic per-verb argument distribution (no spaCy/corpus needed -> fast CI):
    go/come/walk/run -> GOAL; give -> THEME+RECIPIENT (the dative double-object rule); look -> LOCATION; a rare
    verb below threshold -> un-mineable; a transitive verb -> patient;
  - the additive ArgStructureComposer `frame_lexicon=` override is byte-identical to the hand frames by default,
    and renders/recalls through a SUPPLIED mined lexicon;
  - COMPOSER PARITY: render + query_role on derived frames == on the hand frames for the validated facts;
  - ** PERMUTED-MINING ** collapses (random frames -> the render/recall breaks -> the corpus, not the apparatus,
    carries the frames);
  - the agrammatism ablation + the no-confab moat hold on the mined frames.

Plus a CORPUS-GATED test that runs the REAL mining (skips if data/corpus/tinystories.txt + the brain NPZ are
absent, mirroring the regimeb guard). CPU/numpy.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest  # noqa: E402

from research.runners.argstructure_composer import (  # noqa: E402
    ArgStructureComposer, FRAME_LEXICON, FUNCTION_WORDS, frame_for, frame_id, content_slot_count, realized_units)
from research.runners._bucketB_corpus_mined_frames_derisk import (  # noqa: E402
    derive_frame_lexicon, compare_frames, composer_parity, permuted_mining, agrammatism_and_moat,
    _roles_of, PARITY_FACTS, run_seed, VALIDATED_VERBS)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CORPUS = os.path.join(_REPO, "data", "corpus", "tinystories.txt")
_NPZ = os.path.join(_REPO, "bridges", "firstchat", "brainALL_w7000.npz_seed42.npz")


# A FIXED synthetic per-verb argument distribution that reproduces the corpus signal shape (so the derive logic is
# pinned without spaCy/corpus): motion verbs (to-dominant, no dobj), give (dobj=THEME + dative=RECIPIENT), look
# (at/in -> LOCATION), a transitive verb (dobj only), and a rare verb below the freq threshold.
SYNTH_STATS = {
    "go":   {"freq": 1000, "dobj": 2, "dative": 0, "preps": {"to": 300, "into": 40, "for": 80, "with": 30},
             "ex_dobj": None, "ex_dative": None, "ex_prep": {"to": "go to the park"}},
    "come": {"freq": 1000, "dobj": 0, "dative": 0, "preps": {"to": 90, "into": 60, "from": 40},
             "ex_dobj": None, "ex_dative": None, "ex_prep": {"to": "come to the house", "into": "came into the room"}},
    "walk": {"freq": 800, "dobj": 5, "dative": 0, "preps": {"to": 150, "in": 60, "around": 40},
             "ex_dobj": None, "ex_dative": None, "ex_prep": {"to": "walked to the shop"}},
    "run":  {"freq": 800, "dobj": 6, "dative": 0, "preps": {"to": 250, "after": 30},
             "ex_dobj": None, "ex_dative": None, "ex_prep": {"to": "ran to the tree"}},
    "give": {"freq": 600, "dobj": 280, "dative": 120, "preps": {"to": 18, "at": 5},
             "ex_dobj": "gave a cookie", "ex_dative": "gave Lily a cookie", "ex_prep": {"to": "gave it to mom"}},
    "look": {"freq": 900, "dobj": 6, "dative": 0, "preps": {"at": 400, "for": 120, "in": 80},
             "ex_dobj": None, "ex_dative": None, "ex_prep": {"at": "looked at the dog"}},
    "chase": {"freq": 300, "dobj": 200, "dative": 0, "preps": {"around": 20},
              "ex_dobj": "chased the cat", "ex_dative": None, "ex_prep": {}},
    "send": {"freq": 12, "dobj": 4, "dative": 1, "preps": {"to": 2},     # rare -> un-mineable (freq < min_freq)
             "ex_dobj": None, "ex_dative": None, "ex_prep": {}},
}


@pytest.fixture(scope="module")
def mined():
    frames, vpr, prov = derive_frame_lexicon(SYNTH_STATS, min_freq=30, dobj_thresh=0.20,
                                             role_thresh=0.10, min_role_count=20)
    return frames, vpr, prov


def test_composer_frame_lexicon_default_is_byte_identical():
    """The additive frame_lexicon= override defaults to the hand FRAME_LEXICON -> the module frame helpers + a
    default composer render are byte-identical to the prior behaviour (the production path is unchanged)."""
    assert frame_for("go") == FRAME_LEXICON["go"]
    assert frame_id("give") == 4 and content_slot_count("give") == 4
    assert _roles_of(realized_units("go", {"agent": "boy", "GOAL": "park"})) == ("agent", "action", "GOAL")
    c = ArgStructureComposer(seed=42, D=64, vocab=["boy", "go", "park"])    # no frame_lexicon -> hand frames
    c.store_fact({"agent": "boy", "action": "go", "GOAL": "park"})
    assert c.render({"agent": "boy", "action": "go", "GOAL": "park"}) == "the boy goes to the park"


def test_derive_motion_verbs_get_goal(mined):
    frames, _vpr, _prov = mined
    for v in ("go", "come", "walk", "run"):
        assert _roles_of(frames[v]) == ("agent", "action", "GOAL"), f"{v} -> {_roles_of(frames[v])}"


def test_derive_give_is_ditransitive_theme_recipient(mined):
    """give attests a direct object (THEME) AND a dative indirect object (RECIPIENT) -> the ditransitive frame,
    matching the hand FRAME_LEXICON -- recovered from the double-object dative the corpus actually states."""
    frames, _vpr, prov = mined
    assert _roles_of(frames["give"]) == ("agent", "action", "THEME", "RECIPIENT")
    assert prov["give"]["ditransitive"] is True
    # the RECIPIENT slot's lead is the canonical prepositional-dative 'to the' (matching the hand frame's surface)
    recip_unit = [u for u in frames["give"] if u[1] == "RECIPIENT"][0]
    assert recip_unit[2] == ("to", "the")


def test_derive_look_gets_locative_from_at(mined):
    """look has no hand frame (falls to _default transitive); the corpus says it takes `at`/`in` -> a LOCATION
    oblique -- a corpus-JUSTIFIED difference (every slot attested)."""
    frames, _vpr, _prov = mined
    assert "LOCATION" in _roles_of(frames["look"])


def test_derive_transitive_and_unmined(mined):
    """A transitive verb (chase: dobj only) -> patient; a rare verb (send: freq below threshold) -> un-mineable."""
    frames, _vpr, prov = mined
    assert _roles_of(frames["chase"]) == ("agent", "action", "patient")
    assert "send" not in frames and prov["send"]["attested"] is False


def test_every_mined_slot_is_attested(mined):
    """PROVENANCE: every derived slot of every mined verb is corpus-attested (count > 0 with a logged example)."""
    _frames, _vpr, prov = mined
    for v, p in prov.items():
        if not p.get("attested"):
            continue
        for s in p["slots"]:
            assert s["count"] > 0, f"{v}:{s['role']} slot not attested"


def test_composer_parity_render_and_recall(mined):
    """COMPOSER PARITY: render + query_role on the MINED frames == on the hand frames for the validated facts
    (answer-identical, or a corpus-justified frame difference)."""
    frames, _vpr, _prov = mined
    ok, details = composer_parity(42, frames, PARITY_FACTS)
    assert ok, [d for d in details if not d["pair_ok"]]
    # the headline render is byte-identical on the mined frames
    gives = [d for d in details if d["fact"]["action"] == "go"]
    assert gives and gives[0]["mined_render"] == "the boy goes to the park"


def test_permuted_mining_collapses(mined):
    """** THE DECISIVE CONTROL ** -- random (deranged) frames break the render/recall: the corpus, not the
    apparatus, carries the frames."""
    frames, _vpr, _prov = mined
    pm_acc, _permuted = permuted_mining(42, frames, PARITY_FACTS)
    assert pm_acc <= 0.5, f"permuted-mining did not collapse ({pm_acc:.2f})"


def test_agrammatism_and_moat_on_mined(mined):
    frames, _vpr, _prov = mined
    agram_ok, reparse_ok, moat_ok = agrammatism_and_moat(42, frames, PARITY_FACTS)
    assert agram_ok and reparse_ok and moat_ok


# --------------------------------------------------------------------------------------------------------------
# CORPUS-GATED end-to-end test: run the REAL mining (spaCy + corpus). Skips if the corpus/brain artifacts are
# absent (mirrors the regimeb guard's skip-if-data-absent discipline).
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not (os.path.exists(_CORPUS) and os.path.exists(_NPZ)),
                    reason="needs data/corpus/tinystories.txt + bridges/firstchat/brainALL_w7000.npz_seed42.npz")
def test_real_corpus_mining_go():
    """End-to-end: mine the frames from the real corpus, run the de-risk gate on seed 42 -- the validated verbs
    match-or-justify the hand frames, parity holds, permuted-mining collapses, moat 0-FA."""
    import numpy as np
    from research.runners._bucketB_corpus_mined_frames_derisk import (
        mine_verb_argstats, derive_frame_lexicon as _derive)
    d = np.load(_NPZ, allow_pickle=True)
    vocab = set(str(w).lower() for w in d["vocab"])
    stats, _n = mine_verb_argstats(_CORPUS, vocab, 400000, target_verbs=None)
    frames, _vpr, prov = _derive(stats)
    cf = compare_frames(frames)
    # go/walk/run match the hand GOAL frame exactly; the validated verbs are all match-or-corpus-justified
    assert cf["go"][0] == "match" and cf["walk"][0] == "match" and cf["run"][0] == "match"
    n_unjustified = sum(1 for v in VALIDATED_VERBS if cf[v][0] == "differ"
                        and not (prov.get(v, {}).get("attested") and
                                 all(s.get("count", 0) > 0 for s in prov.get(v, {}).get("slots", []))))
    assert n_unjustified == 0, f"{n_unjustified} unjustified frame differences"
    r = run_seed(42, frames, prov, cf)
    assert r["parity_ok"] and r["moat_ok"] and r["agrammatism_ok"] and r["reparse_ok"]
    assert r["permuted_mining_acc"] <= 0.5 and r["mined_acc"] - r["permuted_mining_acc"] >= 0.4
