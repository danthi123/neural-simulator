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
import json, os, socket, sys

import research.runners  # noqa: F401  (the provenance door: stamps labels.json with run id, host, backend)

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
# ---- turns h17..h29 (generated on the GPU after the reboot) ----
# h17 luk_cs_p_l
_set("153 157 158 159 161", "F"); _set("154 155", "B"); _set("156 160", "U"); _set("152", "N")
# h18 merano_italy
_set("163 164 166 167", "F"); _set("165", "B"); _set("162 168 169", "U")
# h19 nc_arts
_set("174 175", "F"); _set("179", "B"); _set("170 171 172 173 176 177 178", "U")
# h20 ousepachan
_set("185", "F"); _set("182 183", "B"); _set("180 181 184 186 187 189", "U"); _set("188", "N")
# h21 prefecture_of_nagasaki
_set("193 196 197", "F"); _set("190", "B"); _set("192 194 195 198 199", "U"); _set("191", "N")
# h22 rep_blica_del_paraguay
_set("201 204 209", "F"); _set("200 203", "B"); _set("202 205 206 207 208", "U")
# h23 s_te_france
_set("211 215", "F"); _set("216 217", "B"); _set("210 212 213 214", "U")
# h24 shelley_winters  (a spelling variant of the name counts as a misnaming -> U; "doubtful goes to U")
_set("218 223", "F"); _set("220 221", "B"); _set("219 222 224 225 226 227", "U")
# h25 steven_lipson
_set("228 229 233 237", "F"); _set("230 231 234 235 236", "B"); _set("232", "U")
# h26 the_weather_in_stavenger
_set("238 242", "F"); _set("239", "B"); _set("240 241 243 244 245", "U")
# h27 universitat_des_saarlandes
_set("248 249 254 255", "F"); _set("252", "B"); _set("246 247 250 251 253", "U")
# h28 waadt_canton_of
_set("261", "F"); _set("259 262", "B"); _set("256 257 258 260 263", "U")
# h29 zhejiang_province
_set("269 270", "B"); _set("264 265 266 267 268 271 272", "U")

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
               "backend": "numpy", "device": "cpu (labels are hand-assigned; this script only writes them out)", "host": socket.gethostname(),
               "script": "research/findings/raw/_moat_slug_norm_heldout/make_labels.py", "argv": sys.argv, "n_labels": len(labels), "partial": partial, "labels": labels}, open(out, "w"), indent=1)
    import collections
    print(len(labels), dict(collections.Counter(l["label"] for l in labels)))
