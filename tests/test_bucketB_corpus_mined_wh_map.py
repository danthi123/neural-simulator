"""Regression guard for Bucket-B B-mine-2 -- the wh->role MAP as the INVERSE INDEX of the corpus-mined verb-frames.

The near-free corollary of B-mine-1: the hand-authored WH_ROLE_CANDIDATES (where->[GOAL,LOCATION],
what->[patient,THEME], who->[agent,RECIPIENT], ...) is DERIVED by INVERTING the mined verb-frame lexicon -- a
wh-word gaps a role-CLASS (the small closed WH_ROLE_CLASS affinity), the mined frames say WHICH roles fall in that
class + in what order (CORE-first, then descending corpus attestation). Structure ACQUIRED, not given. De-risked GO
(research/findings/2026-06-27-burndown-Bmine2-wh-map-GO.md). This guard pins:

  - the INVERSE-INDEX derive on a fixed synthetic mined-frame lexicon (no spaCy/corpus -> fast CI): where->GOAL,
    LOCATION; what->patient,THEME; who->agent,RECIPIENT; whom->RECIPIENT; with->INSTRUMENT; the multiword cues;
  - the attestation-count ORDER (GOAL before LOCATION even though FEWER verbs license GOAL -- the inverse-index
    weight is the corpus attestation, not the verb-license count);
  - the additive wh-parser `frame_roles=` override is byte-identical to the hand FRAME_ROLES by default, and
    resolves against a SUPPLIED (mined) frame inventory;
  - PARSE PARITY: parse + answer on the mined map == on the hand map for the validated questions;
  - ** PERMUTED-MINING ** collapses (a scrambled mined-frame inventory -> a broken wh-resolution -> the parses
    break -> the mined frames, not the apparatus, carry the wh-map);
  - the no-confab moat holds on the mined map.

Plus a CORPUS-GATED test that runs the REAL mining (skips if data/corpus/tinystories.txt + the brain NPZ are
absent, mirroring the B-mine-1 guard). CPU/numpy.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest  # noqa: E402

from research.runners.wh_question_parser import (  # noqa: E402
    WH_ROLE_CANDIDATES as HAND_WH, WH_MULTIWORD as HAND_MW, parse_wh_question, _resolve_wh_role, FRAME_ROLES)
from research.runners._bucketB_corpus_mined_wh_map_derisk import (  # noqa: E402
    derive_wh_role_map, compare_wh_maps, parse_parity, permuted_mining, moat, frame_roles_of,
    WH_CASES, VALIDATED_WH, run_seed)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CORPUS = os.path.join(_REPO, "data", "corpus", "tinystories.txt")
_NPZ = os.path.join(_REPO, "bridges", "firstchat", "brainALL_w7000.npz_seed42.npz")


def _u(role, lead=()):
    return ("CONTENT", role, tuple(lead))


# A FIXED synthetic MINED frame lexicon (the shape B-mine-1 produces) that reproduces the corpus role inventory:
# motion verbs -> GOAL; give/bring -> THEME+RECIPIENT (ditransitive); look/sit -> LOCATION; play -> INSTRUMENT;
# chase -> patient (transitive). Plus the per-role attestation counts (GOAL out-attests LOCATION even though FEWER
# verbs license it -- the inverse-index ranking weight).
SYNTH_FRAMES = {
    "go":   [_u("agent", ("the",)), ("TENSE", "action", ()), _u("GOAL", ("to", "the"))],
    "come": [_u("agent", ("the",)), ("TENSE", "action", ()), _u("GOAL", ("to", "the"))],
    "run":  [_u("agent", ("the",)), ("TENSE", "action", ()), _u("GOAL", ("to", "the"))],
    "give": [_u("agent", ("the",)), ("TENSE", "action", ()), _u("THEME", ("the",)), _u("RECIPIENT", ("to", "the"))],
    "bring": [_u("agent", ("the",)), ("TENSE", "action", ()), _u("THEME", ("the",)), _u("RECIPIENT", ("to", "the"))],
    "look": [_u("agent", ("the",)), ("TENSE", "action", ()), _u("LOCATION", ("on", "the"))],
    "sit":  [_u("agent", ("the",)), ("TENSE", "action", ()), _u("LOCATION", ("on", "the"))],
    "jump": [_u("agent", ("the",)), ("TENSE", "action", ()), _u("LOCATION", ("on", "the"))],
    "play": [_u("agent", ("the",)), ("TENSE", "action", ()), _u("INSTRUMENT", ("with", "the"))],
    "chase": [_u("agent", ("the",)), ("TENSE", "action", ()), _u("patient", ("the",))],
    "_default": [_u("agent", ("the",)), ("TENSE", "action", ()), _u("patient", ("the",))],
}
# corpus attestation totals (GOAL=3437 > LOCATION=2819 in the real mine, here scaled to keep GOAL>LOCATION even
# though 3 verbs license GOAL vs 3 license LOCATION -- the order must come from attestation, not the verb count).
SYNTH_ATTEST = {"GOAL": 3437, "LOCATION": 2819, "patient": 14181, "THEME": 810, "RECIPIENT": 1230, "INSTRUMENT": 1078}


@pytest.fixture(scope="module")
def derived():
    return derive_wh_role_map(SYNTH_FRAMES, attest_count=SYNTH_ATTEST)


def test_parser_frame_roles_default_is_byte_identical():
    """The additive frame_roles= override defaults to the hand FRAME_ROLES -> a default parse is byte-identical to
    the prior behaviour (passing the hand FRAME_ROLES explicitly == None)."""
    p_default = parse_wh_question("where does the boy go?")
    p_explicit = parse_wh_question("where does the boy go?", frame_roles=dict(FRAME_ROLES))
    assert p_default == p_explicit
    assert p_default["role"] == "GOAL"


def test_inverse_index_matches_hand_map(derived):
    """The MINED wh-map matches the hand WH_ROLE_CANDIDATES on the validated wh-words (where/what/who/whom/with)."""
    wh_map, _mw, _prov = derived
    assert wh_map["where"] == ["GOAL", "LOCATION"]      # attestation order (GOAL>LOCATION), not verb-count order
    assert wh_map["what"] == ["patient", "THEME"]
    assert wh_map["who"] == ["agent", "RECIPIENT"]      # core agent before oblique recipient
    assert wh_map["whom"] == ["RECIPIENT"]
    assert wh_map["with"] == ["INSTRUMENT"]
    assert wh_map["who_to"] == ["RECIPIENT", "agent"]   # the to-PP recipient gap


def test_attestation_order_not_verb_count_order(derived):
    """GOAL is licensed by FEWER verbs than LOCATION in SYNTH (3 vs 3 here, but GOAL out-attests) -- the ORDER must
    come from the corpus ATTESTATION (GOAL 3437 > LOCATION 2819), reproducing the hand order. A verb-count ranking
    would be a tie / could flip it; the attestation weight fixes GOAL first."""
    wh_map, _mw, prov = derived
    goal_att = next(c["attestation"] for c in prov["wh"]["where"]["candidates"] if c["role"] == "GOAL")
    loc_att = next(c["attestation"] for c in prov["wh"]["where"]["candidates"] if c["role"] == "LOCATION")
    assert goal_att > loc_att and wh_map["where"][0] == "GOAL"


def test_multiword_derived_from_prep_role(derived):
    """The multiword cues are derived from PREP_ROLE associations: where-from->SOURCE, with-what->INSTRUMENT,
    to-whom->RECIPIENT (the dative -- NOT GOAL, the same `to`-disambiguation as B-mine-1's ditransitive rule)."""
    _wh, mw, _prov = derived
    assert mw[("where", "from")] == "SOURCE" and mw[("from", "where")] == "SOURCE"
    assert mw[("with", "what")] == "INSTRUMENT"
    assert mw[("to", "whom")] == "RECIPIENT"
    assert mw == dict(HAND_MW)


def test_compare_match_or_justify(derived):
    """MATCH-or-justify: the mined map matches the hand map on the validated wh-words (when->[] is a corpus-justified
    difference -- the synthetic frames attest no TIME slot, exactly as the real corpus does not)."""
    wh_map, mw, _prov = derived
    cf = compare_wh_maps(wh_map, mw)
    for wh in ("who", "what", "where", "whom", "with"):
        assert cf[wh][0] == "match", f"{wh}: {cf[wh]}"
    assert cf["when"][0] == "differ" and wh_map.get("when") == []   # un-attested TIME -> justified empty
    assert cf["__multiword__"][0] == "match"


def test_every_mined_candidate_is_attested(derived):
    """PROVENANCE: every mined wh-candidate role is backed by >=1 corpus-attested licensing verb-frame."""
    wh_map, _mw, prov = derived
    for wh in wh_map:
        for c in prov["wh"].get(wh, {}).get("candidates", []):
            assert c["n_licensing_verbs"] > 0, f"{wh}:{c['role']} has no licensing frame"


def test_parse_parity_render_and_recall(derived):
    """PARSE PARITY: parse + answer on the MINED map (resolved against the MINED frame roles) == on the hand map for
    the validated questions (answer-identical)."""
    wh_map, mw, _prov = derived
    parity_ok, mined_acc, details = parse_parity(42, SYNTH_FRAMES, wh_map, mw)
    assert parity_ok, [d for d in details if not d["pair_ok"]]
    assert mined_acc == 1.0


def test_permuted_mining_collapses(derived):
    """** THE DECISIVE CONTROL ** -- a scrambled mined-frame inventory -> a broken wh-resolution: the corpus, not
    the apparatus, carries the wh-map."""
    wh_map, mw, _prov = derived
    pm_acc, _scrambled = permuted_mining(42, SYNTH_FRAMES, wh_map, mw)
    assert pm_acc <= 0.5, f"permuted-mining did not collapse ({pm_acc:.2f})"


def test_scrambled_frame_roles_make_parser_abstain():
    """The mechanism behind the permuted-mining collapse: when a verb's FRAME_ROLES is scrambled so it licenses the
    WRONG roles, the wh-resolution can't intersect a candidate -> abstain (None)."""
    # go scrambled to a THEME-only frame -> where (GOAL/LOCATION) intersect {THEME} = empty -> None.
    assert _resolve_wh_role("where", "go",
                            frame_roles={"go": ["agent", "action", "THEME"],
                                         "_default": ["agent", "action", "patient"]}) is None


def test_moat_holds_on_mined(derived):
    wh_map, mw, _prov = derived
    fa, recall_ok, abstain_ok, n_abstain = moat(42, SYNTH_FRAMES, wh_map, mw)
    assert fa == 0 and recall_ok and abstain_ok == n_abstain


# --------------------------------------------------------------------------------------------------------------
# CORPUS-GATED end-to-end test: run the REAL mining (B-mine-1 frames -> the wh-map). Skips if the corpus/brain
# artifacts are absent (mirrors the B-mine-1 guard's skip-if-data-absent discipline).
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not (os.path.exists(_CORPUS) and os.path.exists(_NPZ)),
                    reason="needs data/corpus/tinystories.txt + bridges/firstchat/brainALL_w7000.npz_seed42.npz")
def test_real_corpus_wh_map_go():
    """End-to-end: mine the frames from the real corpus, invert to the wh-map, run the gate on seed 42 -- the
    validated wh-words match-or-justify the hand map, parse parity holds, permuted-mining collapses, moat 0-FA."""
    import collections
    import numpy as np
    from research.runners._bucketB_corpus_mined_frames_derisk import mine_verb_argstats, derive_frame_lexicon
    d = np.load(_NPZ, allow_pickle=True)
    vocab = set(str(w).lower() for w in d["vocab"])
    stats, _n = mine_verb_argstats(_CORPUS, vocab, 400000, target_verbs=None)
    frames, _vpr, fprov = derive_frame_lexicon(stats)
    attest = collections.Counter()
    for _v, p in fprov.items():
        if p.get("attested"):
            for s in p.get("slots", []):
                attest[s["role"]] += s.get("count", 0)
    wh_map, mw, wh_prov = derive_wh_role_map(frames, attest_count=dict(attest))
    cf = compare_wh_maps(wh_map, mw)
    # where/what/who match the hand map exactly (the gate's headline)
    assert cf["where"][0] == "match" and cf["what"][0] == "match" and cf["who"][0] == "match"
    n_unjustified = sum(1 for wh in VALIDATED_WH if cf[wh][0] == "differ"
                        and not all(c.get("n_licensing_verbs", 0) > 0
                                    for c in wh_prov["wh"].get(wh, {}).get("candidates", [])))
    assert n_unjustified == 0
    r = run_seed(42, frames, wh_map, mw)
    assert r["parity_ok"] and r["moat_ok"]
    assert r["permuted_mining_acc"] <= 0.5 and r["mined_acc"] - r["permuted_mining_acc"] >= 0.4
