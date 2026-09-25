"""The closed-class parse instrument (`_lexicon_closed_class_parse_diag`): its adjudication rule, its three-valued
independent ground truth, and its pre-registered G2/G3 scoring. Pure; no corpus or lexicon build."""
import json

from research.runners import _lexicon_closed_class_parse_diag as P

_NOUNS = {"owl", "apple", "marble", "basket", "room", "box", "leaves"}
_CLOSED = {"when", "most", "the", "where"}


def _gc(w):
    return P.NOUN if w in _NOUNS else (P.NON if w in _CLOSED else P.UNKNOWN)


def test_new_noun_without_drop_matches():
    r = P.adjudicate(["wolf"], ["wolf", "owl"], 5, _gc)
    assert r["match"] and r["changed"] and r["admitted"] == ["owl"]


def test_closed_class_admission_is_a_mismatch():
    r = P.adjudicate(["dog", "bird"], ["dog", "when", "bird"], 5, _gc)
    assert not r["match"] and r["bad_admits"] == ["when"]


def test_unknown_admission_is_listed_not_counted():
    r = P.adjudicate(["dog"], ["dog", "east"], 5, _gc)
    assert r["match"] and r["unknown_admits"] == ["east"] and r["bad_admits"] == []


def test_cap_displacement_by_genuine_nouns_is_explained():
    r = P.adjudicate(["sally", "anne", "box"], ["marble", "basket", "sally", "leaves", "room"], 5, _gc)
    assert r["match"] and r["dropped"] == ["anne", "box"] and r["unexplained_drops"] == []


def test_cap_displacement_with_a_closed_class_word_is_not_explained():
    r = P.adjudicate(["sally", "anne"], ["marble", "sally", "when", "room", "basket"], 5, _gc)
    assert not r["match"] and r["unexplained_drops"] == ["anne"]


def test_drop_without_full_cap_is_not_explained():
    r = P.adjudicate(["sally", "anne"], ["sally", "owl"], 5, _gc)
    assert not r["match"] and r["unexplained_drops"] == ["anne"]


def test_order_change_is_a_mismatch():
    assert not P.adjudicate(["a1", "b1"], ["b1", "a1"], 5, _gc)["match"]


def test_ground_truth_three_valued():
    gt_class, gt_pos = P.load_gt()
    for w in ("owl", "marble", "apple", "today", "story"):
        assert gt_class(w) == P.NOUN, w
    for w in ("what", "who", "when", "before", "most", "the", "wonderful", "crazy", "moves"):
        assert gt_class(w) == P.NON, w
    for w in ("east", "anne", "sally"):
        assert gt_class(w) == P.UNKNOWN, w
    assert gt_pos("most") == "ADJ" and gt_pos("what") is None


def _seed_file(tmp_path, seed, intact_mm, lesion_mm, variant="junction", sha="c"):
    arm = lambda n: {"n_mismatch": n, "offending_words": [], "unknown_admits": [],  # noqa: E731
                     "new_gt_nouns_recovered": [], "tom_fb_anne_kept": True}
    d = {"seed": seed, "variant": variant, "arms": {"intact": arm(intact_mm), "coincidence": arm(lesion_mm)},
         "corpus_sha256": sha, "fixture_sha256": "f", "corpus_pos_map_sha256": "m", "closed_class_sha256": "k"}
    (tmp_path / f"junction_s{seed}.json").write_text(json.dumps(d))


def test_score_go_needs_seed42_and_five_of_six(tmp_path):
    for s in P.EVAL_SEEDS:
        _seed_file(tmp_path, s, 0, 3)
    out = P.score(str(tmp_path))
    assert out["G2_parse_match"] == (6, True) and out["G3_conjunction_lesion"] == (6, True)
    assert out["verdict"] == "G2+G3 PASS, G1 NOT SCORED"


def test_score_fails_g2_when_production_seed_mismatches(tmp_path):
    for s in P.EVAL_SEEDS:
        _seed_file(tmp_path, s, 1 if s == 42 else 0, 3)
    out = P.score(str(tmp_path))
    assert out["G2_parse_match"] == (5, False) and out["verdict"] == "NO-GO"


def test_score_fails_g3_when_the_lesion_does_not_move(tmp_path):
    for s in P.EVAL_SEEDS:
        _seed_file(tmp_path, s, 0, 0 if s in (42, 43) else 2)
    out = P.score(str(tmp_path))
    assert out["G3_conjunction_lesion"] == (4, False) and out["verdict"] == "NO-GO"


def test_score_incomplete_and_mixed_input(tmp_path):
    for s in P.EVAL_SEEDS[:5]:
        _seed_file(tmp_path, s, 0, 3)
    assert P.score(str(tmp_path))["verdict"] == "INCOMPLETE"
    _seed_file(tmp_path, 102, 0, 3, sha="other")
    assert P.score(str(tmp_path))["verdict"] == "MIXED-INPUT"


def test_score_reads_route_verdict(tmp_path):
    for s in P.EVAL_SEEDS:
        _seed_file(tmp_path, s, 0, 3)
    route = tmp_path / "route"
    route.mkdir()
    (route / "verdict.json").write_text(json.dumps({"verdict": "GO", "inputs": [["e", "c"]]}))
    assert P.score(str(tmp_path), str(route))["verdict"] == "GO"
    (route / "verdict.json").write_text(json.dumps({"verdict": "NO-GO", "inputs": [["e", "c"]]}))
    assert P.score(str(tmp_path), str(route))["verdict"] == "NO-GO"
    (route / "verdict.json").write_text(json.dumps({"verdict": "GO", "inputs": [["e", "zzz"]]}))
    assert P.score(str(tmp_path), str(route))["verdict"] == "MIXED-INPUT"


# ── AMENDMENT 3: G3' (two drive-REMOVING lesions) and the junction_elemental scoring ─────────────────────────────
def _earm(mm, silent, drive):
    return {"n_mismatch": mm, "offending_words": [], "unknown_admits": [], "new_gt_nouns_recovered": [],
            "tom_fb_anne_kept": True, "silent_non_fraction": silent,
            "mean_afferent_drive": None if drive is None else {"total": drive}}


def _elem_file(tmp_path, seed, intact=(0, 0.2, 10.0), elemental=(0, 0.5, 6.0), conjunctive=(3, 0.3, 4.0),
               variant="junction_elemental", sha="c"):
    d = {"seed": seed, "variant": variant,
         "arms": {"intact": _earm(*intact), "elemental": _earm(*elemental), "conjunctive": _earm(*conjunctive)},
         "corpus_sha256": sha, "fixture_sha256": "f", "corpus_pos_map_sha256": "m", "closed_class_sha256": "k"}
    (tmp_path / f"junction_elemental_s{seed}.json").write_text(json.dumps(d))


def test_g3prime_needs_both_levers_and_no_added_drive():
    ok = P.g3prime_seed({"intact": _earm(0, 0.2, 10.0), "elemental": _earm(0, 0.5, 6.0),
                         "conjunctive": _earm(3, 0.3, 4.0)})
    assert ok["g3prime_pass"] and not ok["void"]
    no_a = P.g3prime_seed({"intact": _earm(2, 0.2, 10.0), "elemental": _earm(0, 0.5, 6.0),
                           "conjunctive": _earm(2, 0.3, 4.0)})
    assert not no_a["g3a_conjunctive_raises_mismatch"] and not no_a["g3prime_pass"]
    no_b = P.g3prime_seed({"intact": _earm(0, 0.2, 10.0), "elemental": _earm(0, 0.2, 6.0),
                           "conjunctive": _earm(3, 0.3, 4.0)})
    assert not no_b["g3b_elemental_raises_silent_non"] and not no_b["g3prime_pass"]
    flooded = P.g3prime_seed({"intact": _earm(0, 0.2, 10.0), "elemental": _earm(0, 0.5, 6.0),
                              "conjunctive": _earm(3, 0.3, 40.0)})
    assert flooded["void"] and not flooded["g3prime_pass"], "a lesion that ADDS drive cannot pass G3'"


def test_score_elemental_go_and_g4(tmp_path):
    for s in P.EVAL_SEEDS:
        _elem_file(tmp_path, s)
    out = P.score(str(tmp_path))
    assert out["variant"] == "junction_elemental"
    assert out["G2_parse_match"] == (6, True) and out["G3prime_dissociation"] == (6, True)
    assert out["G4_silent_non"] == (6, True) and out["verdict"] == "G2+G3'+G4 PASS, G1 NOT SCORED"
    _elem_file(tmp_path, 101, intact=(0, 0.31, 10.0))                   # G4 must hold at EVERY seed
    out = P.score(str(tmp_path))
    assert out["G4_silent_non"] == (5, False) and out["verdict"] == "NO-GO"


def test_score_elemental_incomplete_and_mixed_variant(tmp_path):
    for s in P.EVAL_SEEDS[:5]:
        _elem_file(tmp_path, s)
    assert P.score(str(tmp_path))["verdict"] == "INCOMPLETE"
    _seed_file(tmp_path, 102, 0, 3)                                       # a round-2 junction run mixed in
    assert P.score(str(tmp_path))["verdict"] in ("INCOMPLETE", "MIXED-INPUT")
