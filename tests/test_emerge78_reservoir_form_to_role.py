"""CPU tests for EMERGE-78 -- the fronto-striatal reservoir form->role labeler (hardened after adversarial verification).

Verifies: the non-local (relative-clause) shapes exist + are labeled; the shipped hand labeler returns None on the
multi-argument shapes; content abstraction; the reservoir resolves the relative-clause HEAD (a genuine NON-LOCAL / global
dependency) where BOTH a left-context governing-cue baseline AND a symmetric +-2 window baseline are at chance; controls
(scramble, non-degenerate lesion) collapse; and the seed-42 de-risk clears the hardened GO gates. All CPU/numpy, offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import pytest

import research.runners._emerge78_reservoir_form_to_role_derisk as m78
from research.runners._emerge72_construction_registry_derisk import label_sentence_ext


def test_nonlocal_relative_clause_shapes_labeled():
    assert set(m78._RELHEAD_KINDS) == {"subjrel", "objrel"}
    rng = np.random.default_rng(0)
    subj, verb, obj = ["dog", "cat", "fox"], ["chase", "see"], ["ball", "bone"]
    sr, srr = m78._make_sentence("subjrel", rng, subj, verb, obj)
    orl, orr = m78._make_sentence("objrel", rng, subj, verb, obj)
    # head (index 1) role differs: AGENT in a subject-relative, THEME in an object-relative (the non-local contrast)
    assert srr[1] == "AGENT" and orr[1] == "THEME"
    # the head has the IDENTICAL left context 'the [head] that' in both
    assert sr[0] == "the" and sr[2] == "that" and orl[0] == "the" and orl[2] == "that"


def test_hand_labeler_returns_none_on_multiarg_heldout():
    closed = {"the", "a", "can", "does", "not", "to", "on"}
    rng = np.random.default_rng(0)
    subj, verb, obj = ["dog", "cat"], ["give", "throw"], ["bone", "ball"]
    for k in m78._LOCAL_HELDOUT:
        toks, _r = m78._make_sentence(k, rng, subj, verb, obj)
        assert label_sentence_ext(toks, closed) is None, (k, toks)


def test_content_abstraction_and_nondegenerate_lesion():
    enc = m78.Encoder({"the", "to", "on", "can", "does", "not"})
    U = enc.encode(["the", "dog", "chases", "the", "ball"])
    assert U[1, enc.open_i] == 1.0 and U[2, enc.open_i] == 1.0   # content -> OPEN (no lexical identity)
    assert U[0, enc.idx["the"]] == 1.0
    # NON-DEGENERATE lesion: closed -> ONE generic marker, content still OPEN, STRUCTURE preserved (not all-identical)
    Ul = enc.encode(["the", "dog", "to", "the", "ball"], lesion=True)
    assert Ul[0, enc.closed_generic_i] == 1.0 and Ul[2, enc.closed_generic_i] == 1.0   # the, to -> generic
    assert Ul[1, enc.open_i] == 1.0                                                     # dog stays OPEN
    assert not np.all(Ul == Ul[0])                                                      # NOT a degenerate single row


def test_reservoir_deterministic_per_seed():
    r1 = m78.Reservoir(16, seed=42); r2 = m78.Reservoir(16, seed=42)
    assert np.allclose(r1.W_res, r2.W_res) and np.allclose(r1.W_in, r2.W_in)
    assert not np.allclose(r1.W_res, m78.Reservoir(16, seed=43).W_res)


def test_final_state_read_uses_whole_sentence():
    # the final-state read-out is what lets a word's role depend on a cue to its RIGHT (the rel-clause disambiguator)
    enc = m78.Encoder({"the", "to", "on"})
    res = m78.Reservoir(enc.dim, seed=42)
    f_short = res.final_state(enc.encode(["the", "dog", "runs"]))
    f_long = res.final_state(enc.encode(["the", "dog", "that", "runs", "the", "cat"]))
    assert f_short.shape == (res.n,) and not np.allclose(f_short, f_long)   # different sentences -> different final state


@pytest.mark.slow
def test_seed42_derisk_clears_hardened_go_gates():
    d = m78._derisk_one(42)
    # (A) CONSOLIDATION: the reservoir learns the full form->role map
    assert d["train_acc"] >= 0.95, d["train_acc"]
    # (B) NECESSITY: reservoir resolves the rel-clause head where BOTH local baselines are at chance
    assert d["relhead_reservoir"] >= 0.90, d["relhead_reservoir"]
    assert d["relhead_gov_baseline"] <= 0.65, d["relhead_gov_baseline"]       # left-context rule fails
    assert d["relhead_symwin_baseline"] <= 0.65, d["relhead_symwin_baseline"]  # symmetric +-2 window ALSO fails (global)
    # (C)/(D) controls collapse on the rel-head
    assert d["relhead_scramble"] <= d["chance_binary"] + 0.18
    assert (d["relhead_reservoir"] - d["relhead_lesion"]) >= 0.25             # closed-class IDENTITY load-bearing
    # honest: the shipped hand labeler cannot do the multi-arg shapes
    assert d["hand_labeler_local_heldout_acc"] <= 0.10
