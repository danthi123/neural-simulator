"""Hand labels for the unique (turn, svo, surface) sentences of corpus.json, numbered by unique_sentences.py (turn order, then
first appearance).  Assigned by reading each sentence against ITS TURN'S GATHERED SET, BEFORE any B2h (slug-norm) verdict was
computed on this corpus; the commit that adds this file/labels.json precedes the measurement commit.  The stock verifier ran only
inside the generation process (it rejected everything); no verdict was consulted.

Rubric (the memo's, moat-memo/labels.py): does the sentence assert anything NOT supported by the turn's gathered facts?
  F faithful   - states a gathered fact (de-slug, case, plural, article, light relation paraphrase), adds nothing else.
  B borderline - near-paraphrase with small drift: relation weakened/changed (citizenship -> lives in), subject or object
                 generalised or partial, garbled but content-equal, evaluative trailing filler ("is fascinating").
  U unsupported- adds an entity/property/number/claim, swaps roles, misnames the entity or object (typos included), asserts the
                 subject IS a country / other category it is not, non-English, truncated nonsense.
  N no-claim   - fragment/persona that asserts nothing about the facts.
When torn between B and U the label is U (the bar counts U, so a doubtful sentence is held against the verifier).
usage (cwd = worktree): make_labels.py <corpus.json> <out labels.json> [--partial]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unique_sentences import unique

LAB = {}
NOTE = {}
def _set(nums, lab):
    for i in nums.split():
        assert int(i) not in LAB, i
        LAB[int(i)] = lab

# h01 1962_asian_games
_set("6 8", "F"); _set("4 5 10", "B"); _set("1 2 3 7 9", "U")
# h02 almer_a_spain
_set("12", "F"); _set("18", "B"); _set("11 13 14 15 16 17", "U")
# h03 aus_fb_a_league_br
_set("22 23 25", "B"); _set("19 20 21 24 26 27 28", "U")
# h04 bison_ravi
_set("29 30 32 34 37", "F"); _set("31 35", "B"); _set("33 36 38", "U")
# h05 carl_lamarc
_set("39 40 42 44 45", "F"); _set("41 46", "B"); _set("43 48", "U"); _set("47", "N")
# h06 city_of_green_bay
_set("49 52 53 55 56 57", "F"); _set("50 54", "B"); _set("51", "U")
# h07 danny_lanois
_set("58 59 61 63 67", "F"); _set("60 64 66", "B"); _set("62 65", "U")
# h08 e_e_horton
_set("71 73 77", "F"); _set("69 70 76", "B"); _set("68 72 74 75", "U")
# h09 fernand_joseph_desire_contandin
_set("78 79 81 83", "F"); _set("80 87", "B"); _set("82 84 85 86", "U")
# h10 geography_of_adana
_set("89 90 91 92 94", "F"); _set("88 93", "B"); _set("95", "U")
# h11 ha_noi_vietnam
_set("97 101", "F"); _set("98", "B"); _set("96 99 100 102 103 104 105", "U")
# h12 history_of_ferrara
_set("106 107 112 113", "F"); _set("108 110", "B"); _set("109 111", "U")
# h13 ida_lundgren
_set("114 115 117 119 120", "F"); _set("116 121", "B"); _set("118 122 123", "U")
# h14 ji_nan_city
_set("129 130", "F"); _set("125", "B"); _set("124 126 127 128 131 132", "U")
# h15 keifer_sutherland
_set("133 134 136 138 139", "F"); _set("135 140 141", "B"); _set("137 142", "U")
# h16 larry_shaw_director
_set("143 144 146 148 149", "F"); _set("145 150", "B"); _set("147 151", "U")
# ---- turns h17..h29 are labelled in the next block (generated on the GPU after the reboot) ----

if __name__ == "__main__":
    corpus = json.load(open(sys.argv[1])); out = sys.argv[2]; partial = "--partial" in sys.argv
    uniq = unique(corpus)
    nums = {u["num"] for u in uniq}
    assert set(LAB) <= nums, sorted(set(LAB) - nums)
    if not partial:
        assert set(LAB) == nums, ("unlabelled", sorted(nums - set(LAB))[:20])
    labels = [{"num": u["num"], "turn": u["turn"], "svo": u["svo"], "surface": u["surface"], "label": LAB[u["num"]]}
              for u in uniq if u["num"] in LAB]
    json.dump({"_doc": "hand labels (rubric in make_labels.py), assigned before any slug-norm verdict on this corpus",
               "backend": "numpy", "device": "cpu", "script": "research/findings/raw/_moat_slug_norm_heldout/make_labels.py",
               "argv": sys.argv, "n_labels": len(labels), "partial": partial, "labels": labels}, open(out, "w"), indent=1)
    import collections
    print(len(labels), dict(collections.Counter(l["label"] for l in labels)))
