"""Hand labels for the unique (turn, svo, surface) sentences of s3_corpus.json (stratum S3: entities with 1-3 AFFIRM facts, real Qwen on
the GPU; rule in select_s3.py), numbered by unique_sentences.py (turn order, then first appearance).  Assigned by reading each sentence
against ITS TURN'S GATHERED SET, BEFORE any B2h (slug-norm) verdict existed on this corpus; the commit that adds this file/s3_labels.json
precedes the measurement commit.  The only verifier that ever ran on these sentences is the stock one inside the generation process
(flag unset); no B2h verdict, replay or composer-path flag-on result had been computed when these labels were assigned.

Rubric (identical to make_labels.py, the S1 rubric; the memo's, moat-memo/labels.py): does the sentence assert anything NOT supported by
the turn's gathered facts?
  F faithful   - states a gathered fact (de-slug, case, plural, article, light relation paraphrase: country -> "is located in",
                 subclass_of -> "is classified as"), adds nothing else.
  B borderline - near-paraphrase with small drift: relation weakened/changed, subject or object generalised or partial, garbled but
                 content-equal, evaluative trailing filler.
  U unsupported- adds an entity/property/number/claim, swaps roles, misnames the entity or object (typos and variant spellings of a
                 gathered name included), asserts the subject IS something it is not, non-English, truncated nonsense.
  N no-claim   - fragment/persona that asserts nothing about the facts.
When torn between B and U the label is U (the bar counts U, so a doubtful sentence is held against the verifier).
usage (cwd = worktree, as a module): python -m research.findings.raw._moat_slug_norm_heldout.make_labels_s3 <s3_corpus.json> <out s3_labels.json> [--partial]"""
import json, os, socket, sys

import research.runners  # noqa: F401  (the provenance door: stamps the labels file with run id, host, backend)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unique_sentences import unique

LAB = {}
def _set(nums, lab):
    for i in nums.split():
        assert int(i) not in LAB, i
        LAB[int(i)] = lab

# s3a01 academica_de_coimbra
_set("5 9", "F"); _set("1 6", "B"); _set("2 3 4 7 8 10", "U")
# s3a02 angola_national_football_team
_set("15 20", "F"); _set("11 16 17 18", "B"); _set("12 13 14 19", "U")
# s3a03 bac_u_county
_set("23", "B"); _set("21 22 24", "U")
# s3a04 bobolice_commune
_set("28 29", "F"); _set("25", "B"); _set("26 27 30", "U")
# s3a05 canton_of_craonne
_set("33", "F"); _set("35 38", "B"); _set("31 32 34 36 37 39 40", "U")
# s3a06 city_of_district_significance_ukraine
_set("44", "F"); _set("41 42 43", "U")
# s3a07 country_mansion
_set("45 47 48 49", "F"); _set("50", "B"); _set("46", "U")
# s3a08 directeurs_sportifs
_set("53 54 56 57", "F"); _set("58", "B"); _set("51 52 55", "U")
# s3a09 emarcy_records
_set("59 62 64 66", "F"); _set("61 63 65", "B"); _set("60", "U")
# s3a10 fc_magna_wiener_neustadt
_set("71", "F"); _set("72", "B"); _set("67 68 69 70 73 74", "U")
# s3a11 french_communist_party
_set("77 81 82", "F"); _set("75 79", "B"); _set("76 78 80", "U")
# s3a12 gorky_film_studio
_set("90", "F"); _set("83 84 85 86 87 88 89", "U")

# s3a13 hertha_berlin_ii
_set("95 97 99", "F"); _set("91 96", "B"); _set("92 93 94 98", "U")
# s3a14 indoor_bandy
_set("102 104 105", "F"); _set("100 103", "B"); _set("101", "U")
# s3a15 k_v_c_westerlo
_set("110", "F"); _set("111", "B"); _set("106 107 108 109 112 113", "U")
# s3a16 lauri_first_name
_set("114 115", "U")
# s3a17 madurai_india
_set("122", "F"); _set("116 117 118 119 120 121 123 124", "U")
# s3a18 milford_new_brunswick
_set("125 127 128 131 132 133 134", "F"); _set("126", "B"); _set("129 130", "U")
# s3a19 national_coat_of_arms
_set("136 137 138", "F"); _set("135", "B")
# s3a20 o_t_bonner_middle_school
_set("139 143 144", "F"); _set("145", "B"); _set("140 141 142 146", "U")
# s3a21 parke_county_in
_set("151 152", "F"); _set("147 155", "B"); _set("148 149 150 153 154 156", "U")
# s3a22 premiers_of_victoria
_set("159 160 161 164", "F"); _set("158", "B"); _set("157 162 163", "U")
# s3a23 renderware
_set("165 166 167 169 170", "F"); _set("168", "U")
# s3a24 samut_songkhram_btu_f_c
_set("175", "F"); _set("172 176", "B"); _set("171 173 174 177 178", "U")

# s3a25 sire_records_group
_set("183 184 185 186", "F"); _set("179 180", "B"); _set("181 182", "U")
# s3a26 sten_first_name
_set("188", "F"); _set("187", "B"); _set("189 190 191 192", "U")
# s3a27 the_baa_baas
_set("193 196", "B"); _set("194 195 197 198", "U")
# s3a28 turkey_national_under_17_football_team
_set("203 205", "F"); _set("199 204", "B"); _set("200 201 202 206", "U")
# s3a29 veterinary_physician (one gathered fact)
_set("207 208", "F")
# s3a30 worcester_rfc
_set("212", "B"); _set("209 210 211 213 214", "U")
# s3b01 adrijan (one gathered fact)
_set("215", "B"); _set("216", "U")
# s3b02 anita_given_name (one gathered fact)
_set("217", "B"); _set("218", "U")
# s3b03 badminton_at_the_2006_asian_games_women (one gathered fact)
_set("220", "B"); _set("219", "U")
# s3b04 business_chains (one gathered fact)
_set("222", "F"); _set("221", "B")
# s3b05 classical_style (one gathered fact)
_set("224", "B"); _set("223", "N")
# s3b06 corrado_given_name (one gathered fact)
_set("226", "F"); _set("225", "B")

# s3b07 defensive_secondary (one gathered fact)
_set("228", "F"); _set("227", "B")
# s3b08 earle_given_name (one gathered fact)
_set("229 230", "U")
# s3b09 equestrian_statue (one gathered fact)
_set("232", "F"); _set("231", "B")
# s3b10 field_goal_kicker (one gathered fact)
_set("233", "F"); _set("234", "U")
# s3b11 gaclbusiness (one gathered fact)
_set("235", "B"); _set("236", "U")
# s3b12 gonzalo_given_name (one gathered fact)
_set("237", "B"); _set("238", "U")
# s3b13 heikki (one gathered fact)
_set("239", "F")
# s3b14 independent_spirit_award_for_best_first (one gathered fact)
_set("240", "F"); _set("241", "U")
# s3b15 jock_first_name (one gathered fact)
_set("242 243", "U")
# s3b16 lady_in_waiting (one gathered fact)
_set("244 245", "B")
# s3b17 london_midland_region (one gathered fact)
_set("246 247", "U")
# s3b18 marvel_superhero (one gathered fact)
_set("249", "F"); _set("248", "B")

# s3b19 mikkel (one gathered fact)
_set("250 251", "U")
# s3b20 naval_artillery_support (one gathered fact)
_set("252 253", "F")
# s3b21 ocean_liners (one gathered fact)
_set("255", "F"); _set("254", "U")
# s3b22 paavo (one gathered fact)
_set("256 257", "U")
# s3b23 pittsburgh_lrt (one gathered fact)
_set("258 259", "F")
# s3b24 quarter_back (one gathered fact)
_set("260 261", "B")
# s3b25 rustic_architecture (one gathered fact)
_set("263", "F"); _set("262", "U")
# s3b26 shane_name (one gathered fact)
_set("264", "B"); _set("265", "U")
# s3b27 supervising_direction (one gathered fact)
_set("267", "F"); _set("266", "U")
# s3b28 think_thank (one gathered fact)
_set("268 269", "U")
# s3b29 unit_of_volume (one gathered fact)
_set("270 271", "F")
# s3b30 web_gallery_of_art (one gathered fact)
_set("272 273", "F")
# s3b31 z_rich_sc_lions (one gathered fact)
_set("275", "B"); _set("274", "U")

if __name__ == "__main__":
    corpus = json.load(open(sys.argv[1])); out = sys.argv[2]; partial = "--partial" in sys.argv
    uniq = unique(corpus)
    nums = {u["num"] for u in uniq}
    assert set(LAB) <= nums, sorted(set(LAB) - nums)
    if not partial:
        assert set(LAB) == nums, ("unlabelled", sorted(nums - set(LAB))[:20])
    labels = [{"num": u["num"], "turn": u["turn"], "svo": u["svo"], "surface": u["surface"], "label": LAB[u["num"]]}
              for u in uniq if u["num"] in LAB]
    json.dump({"_doc": "hand labels for stratum S3 (rubric in make_labels_s3.py), assigned before any slug-norm verdict on this corpus",
               "backend": "numpy", "device": "cpu (labels are hand-assigned; this script only writes them out)", "host": socket.gethostname(),
               "script": "research/findings/raw/_moat_slug_norm_heldout/make_labels_s3.py", "argv": sys.argv, "n_labels": len(labels),
               "partial": partial, "labels": labels}, open(out, "w"), indent=1)
    import collections
    print(len(labels), dict(collections.Counter(l["label"] for l in labels)))
