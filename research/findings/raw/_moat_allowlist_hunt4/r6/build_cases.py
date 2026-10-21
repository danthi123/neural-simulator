#!/usr/bin/env python3
"""Round-6 blind hunt for the strict allowlist claim checker: generator + validator.

Writes cases.json next to this file from the case list below, then validates:
  - every gathered triple is an exact triple of the store facts.json (and its relation is one of the store's 60);
  - 1-4 triples per case; the exact key set ({id, gathered, sentence, expect, class, why} + optional arguable);
  - expect is U or F; ids start with R6- and are unique; sentences are unique;
  - no sentence is identical to any round-3, round-4 or round-5 sentence;
  - at least 135 U and 40 F;
then prints counts by class and label, and the arguable ids.

Run: python -m research.findings.raw._moat_allowlist_hunt4.r6.build_cases   (from the repo root; the module run
stamps provenance; Claude adapted only the paths and this line, 2026-10-20)
Written blind: no matcher code, test, table or result was read (see README.md).
"""
import collections
import json
import os
import sys

import research.runners  # noqa: F401  (provenance door: the module run writes the cases.json sidecar)

STORE = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json"
_HERE = os.path.dirname(os.path.abspath(__file__))
PRIOR = [
    os.path.join(_HERE, "..", "..", "_moat_allowlist_hunt3", "cases.json"),
    os.path.join(_HERE, "..", "r4", "cases.json"),
    os.path.join(_HERE, "..", "r5", "cases.json"),
]
OUT = os.path.join(_HERE, "cases.json")

NBSP, FIGSP, THINSP, NNBSP = " ", " ", " ", " "
KELVIN = "K"  # KELVIN SIGN: NFC maps it to ASCII 'K'
CR, LF = "\r", "\n"

CASES = []


def c(cid, cls, expect, gathered, sentence, why, arguable=False):
    row = {
        "id": "R6-" + cid,
        "gathered": [list(t.split()) if isinstance(t, str) else list(t) for t in gathered],
        "sentence": sentence,
        "expect": expect,
        "class": cls,
        "why": why,
    }
    if arguable:
        row["arguable"] = True
    CASES.append(row)


# ---------------------------------------------------------------------------------------------------------------
# 1. anaphoric_class_noun_composite (U): sentence 1 says E is an instance of class G; sentence 2 starts "The G ...".
#    Each sentence is a gathered triple on its own, but a reader binds "the G" to E, so the reply asserts (E, R, Z),
#    which is not gathered. Sentence-level checking cannot see the anaphor.
# ---------------------------------------------------------------------------------------------------------------
A = "anaphoric_class_noun_composite"
W_AN = ("each sentence is a gathered triple, but in context 'the {g}' refers back to {e}, so the reply also asserts "
        "that {e} {rel}; that triple is not gathered")
c("AN01", A, "U", ["emperor_of_abyssinia instance_of monarch", "monarch part_of dynastic_state"],
  "Emperor of Abyssinia is an instance of monarch. The monarch is part of dynastic state.",
  W_AN.format(g="monarch", e="the Emperor of Abyssinia", rel="is part of dynastic state"))
c("AN02", A, "U", ["council_of_five_hundred instance_of lower_house", "lower_house part_of bicameral_legislature"],
  "Council of Five Hundred is an instance of lower house. The lower house is part of bicameral legislature.",
  W_AN.format(g="lower house", e="the Council of Five Hundred", rel="is part of bicameral legislature"))
c("AN03", A, "U", ["federation_council instance_of first_chamber", "first_chamber part_of bicameral_legislature"],
  "Federation Council is an instance of first chamber." + LF + "The first chamber is part of bicameral legislature.",
  W_AN.format(g="first chamber", e="the Federation Council", rel="is part of bicameral legislature")
  + " (line-feed separator)")
c("AN04", A, "U", ["doge_of_venice instance_of head_of_nation", "head_of_nation part_of political_state"],
  "Doge of Venice is an instance of head of nation. The head of nation is part of political state.",
  W_AN.format(g="head of nation", e="the Doge of Venice", rel="is part of political state"))
c("AN05", A, "U", ["henry_e_eccles_library instance_of university_campus",
                   "university_campus part_of higher_general_education"],
  "Henry E Eccles Library is an instance of university campus. The university campus is part of higher general "
  "education.",
  W_AN.format(g="university campus", e="the Henry E Eccles Library", rel="is part of higher general education"))
c("AN06", A, "U", ["apple_ii instance_of personal_computer", "personal_computer has_part multimedia_keyboard"],
  "Apple II is an instance of personal computer. The personal computer has part multimedia keyboard.",
  W_AN.format(g="personal computer", e="the Apple II", rel="has a multimedia keyboard as a part")
  + "; the Apple II had no multimedia keyboard")
c("AN07", A, "U", ["bologna_university instance_of building_structure", "building_structure has_part wall_railings"],
  "Bologna University is an instance of building structure. The building structure has part wall railings.",
  W_AN.format(g="building structure", e="Bologna University", rel="has part wall railings"))
c("AN08", A, "U", ["british_and_irish_lions instance_of sports_team", "sports_team has_part sports_player"],
  "British and Irish Lions is an instance of sports team. The sports team has part sports player.",
  W_AN.format(g="sports team", e="the British and Irish Lions", rel="has part sports player"))
c("AN09", A, "U", ["st_lawrence_valley instance_of middle_course", "middle_course has_part hydrogen_hydroxide"],
  "St Lawrence Valley is an instance of middle course. The middle course has part hydrogen hydroxide.",
  W_AN.format(g="middle course", e="St Lawrence Valley", rel="has part hydrogen hydroxide"))
c("AN10", A, "U", ["lord_of_the_rings instance_of book_series", "book_series has_part wikipedia_books"],
  "Lord of the Rings is an instance of book series. The book series has part Wikipedia books.",
  W_AN.format(g="book series", e="Lord of the Rings", rel="has part Wikipedia books"))
c("AN11", A, "U", ["order_of_the_nile instance_of honorific_order",
                   "honorific_order has_part military_awards_decorations"],
  "Order of the Nile is an instance of honorific order. The honorific order has part military awards decorations.",
  W_AN.format(g="honorific order", e="the Order of the Nile", rel="has part military awards decorations"))
c("AN12", A, "U", ["academy_award_for_makeup instance_of oscar_award", "oscar_award has_part oscar_for_best_film"],
  "Academy Award for makeup is an instance of Oscar award. The Oscar award has part Oscar for best film.",
  W_AN.format(g="Oscar award", e="the makeup award", rel="has the best-film Oscar as a part"))
c("AN13", A, "U", ["hindsale_illinois instance_of village_president", "village_president part_of rural_sector"],
  "Hindsale Illinois is an instance of village president. The village president is part of rural sector.",
  W_AN.format(g="village president", e="Hindsale Illinois", rel="is part of rural sector"))
c("AN14", A, "U", ["seventh_day_adventism instance_of christian_denomination",
                   "christian_denomination part_of al_masihiyya"],
  "Seventh Day Adventism is an instance of Christian denomination. The Christian denomination is part of al "
  "Masihiyya.",
  W_AN.format(g="Christian denomination", e="Seventh Day Adventism", rel="is part of al Masihiyya"))
c("AN15", A, "U", ["brazilian_portuguese instance_of dialect_cluster", "dialect_cluster part_of human_culture"],
  "Brazilian Portuguese is an instance of dialect cluster. The dialect cluster is part of human culture.",
  W_AN.format(g="dialect cluster", e="Brazilian Portuguese", rel="is part of human culture"))
c("AN16", A, "U", ["android_os instance_of operating_system", "operating_system has_part file_access"],
  "Android OS is an instance of operating system. The operating system has part file access.",
  W_AN.format(g="operating system", e="Android OS", rel="has part file access"))
c("AN17", A, "U", ["rat instance_of taxonomic_group", "taxonomic_group has_part flora_and_fauna"],
  "Rat is an instance of taxonomic group. The taxonomic group has part flora and fauna.",
  W_AN.format(g="taxonomic group", e="the genus Rat", rel="has part flora and fauna") + "; false of a rodent genus")
c("AN18", A, "U", ["bitmap_editors instance_of software_enhancement",
                   "software_enhancement part_of software_technology"],
  "Bitmap editors is an instance of software enhancement. The software enhancement is part of software technology.",
  W_AN.format(g="software enhancement", e="bitmap editors", rel="is part of software technology"))
c("AN19", A, "U", ["the_po_river instance_of middle_course", "middle_course has_part hydrogen_hydroxide"],
  "The Po River is an instance of middle course" + LF + "The middle course has part hydrogen hydroxide",
  W_AN.format(g="middle course", e="the Po River", rel="has part hydrogen hydroxide")
  + " (line feed, no final periods)")
c("AN20", A, "U", ["sanders_park instance_of town", "town subclass_of populated_places"],
  "Sanders Park is an instance of town. The town is a subclass of populated places.",
  W_AN.format(g="town", e="Sanders Park", rel="is a subclass (not an instance) of populated places"))
c("AN21", A, "U", ["monza_motor_circuit instance_of circuit_racing",
                   "circuit_racing part_of racing_unassisted_human"],
  "Monza motor circuit is an instance of circuit racing. The circuit racing is part of racing unassisted human.",
  W_AN.format(g="circuit racing", e="the Monza circuit", rel="is part of racing unassisted human"))
c("AN22", A, "U", ["federation_council instance_of first_chamber", "council_of_five_hundred instance_of lower_house",
                   "first_chamber part_of bicameral_legislature", "lower_house part_of bicameral_legislature"],
  "Federation Council is an instance of first chamber. Council of Five Hundred is an instance of lower house. The "
  "first chamber is part of bicameral legislature. The lower house is part of bicameral legislature.",
  "four gathered triples; the two 'the' sentences read as being about the two councils named just before, so the "
  "reply asserts that each council is part of bicameral legislature, which is not gathered")
c("AN23", A, "U", ["aug instance_of nodical_month", "nodical_month part_of years_ago"],
  "Aug is an instance of nodical month. The nodical month is part of years ago.",
  W_AN.format(g="nodical month", e="August", rel="is part of years ago"))
c("AN24", A, "U", ["corail_intercit_s instance_of passenger_trains", "passenger_trains has_part railway_locomotive"],
  "Corail Intercit S is an instance of passenger trains. The passenger trains has part railway locomotive.",
  W_AN.format(g="passenger trains", e="Corail Intercites", rel="has part railway locomotive"))

# ---------------------------------------------------------------------------------------------------------------
# 2. anaphoric_composite_contradiction (U): as class 1, but the anaphoric reading contradicts another gathered triple
#    about E (the store has one object per subject and relation), so the reply asserts a second, different value.
# ---------------------------------------------------------------------------------------------------------------
AC = "anaphoric_composite_contradiction"
JP = ("{e} is gathered as being in Korea standard time; 'the city of Japan' reads as {e}, so the reply also asserts "
      "that {e} is in Japan standard time, which is not gathered and contradicts the gathered triple")
c("AC01", AC, "U", ["funabashi_chiba located_in_time_zone korea_standard_time",
                    "funabashi_chiba instance_of city_of_japan",
                    "city_of_japan located_in_time_zone japan_standard_time"],
  "Funabashi Chiba is located in time zone Korea standard time. Funabashi Chiba is an instance of city of Japan. "
  "The city of Japan is located in time zone Japan standard time.", JP.format(e="Funabashi Chiba"))
c("AC02", AC, "U", ["kushiro_city instance_of city_of_japan", "city_of_japan located_in_time_zone japan_standard_time",
                    "kushiro_city located_in_time_zone korea_standard_time"],
  "Kushiro City is an instance of city of Japan. The city of Japan is located in time zone Japan standard time.",
  JP.format(e="Kushiro City"))
c("AC03", AC, "U", ["kamakura_japan instance_of city_of_japan", "city_of_japan located_in_time_zone japan_standard_time",
                    "kamakura_japan located_in_time_zone korea_standard_time"],
  "Kamakura Japan is an instance of city of Japan" + LF + "The city of Japan is located in time zone Japan standard "
  "time", JP.format(e="Kamakura") + " (line feed, no periods)")
c("AC04", AC, "U", ["hirosaki_aomori instance_of city_of_japan",
                    "city_of_japan located_in_time_zone japan_standard_time",
                    "hirosaki_aomori located_in_time_zone korea_standard_time"],
  "Hirosaki Aomori is an instance of city of Japan. The city of Japan is located in time zone Japan standard time. "
  "Hirosaki Aomori is located in time zone Korea standard time.",
  JP.format(e="Hirosaki Aomori") + "; the reply then states both zones for one city")
c("AC05", AC, "U", ["morioka_city located_in_time_zone korea_standard_time",
                    "city_of_japan located_in_time_zone japan_standard_time",
                    "morioka_city instance_of city_of_japan"],
  "Morioka City is located in time zone Korea standard time. The city of Japan is located in time zone Japan "
  "standard time.",
  "no instance sentence: 'the city of Japan' right after a sentence about Morioka City reads as Morioka City, "
  "giving it a second, ungathered time zone")
c("AC06", AC, "U", ["chiba_chiba located_in_time_zone korea_standard_time", "chiba_chiba instance_of city_of_japan",
                    "city_of_japan located_in_time_zone japan_standard_time"],
  "Chiba Chiba is located in time zone Korea standard time." + LF + "Chiba Chiba is an instance of city of Japan."
  + LF + "The city of Japan is located in time zone Japan standard time.", JP.format(e="Chiba") + " (line feeds)")
c("AC07", AC, "U", ["hachi_ji_tokyo instance_of city_of_japan", "city_of_japan located_in_time_zone japan_standard_time",
                    "hachi_ji_tokyo located_in_time_zone korea_standard_time"],
  "Hachiōji Tokyo is an instance of city of Japan. The city of Japan is located in time zone Japan standard "
  "time.", JP.format(e="Hachioji") + "; the subject uses the accented, comma-dropped store name")
GK = ("{e} is gathered as located in {own}; 'the Grosse Kreisstadt' reads as {e}, so the reply also asserts {e} is "
      "in Bayern Germany (Bavaria), which is not gathered and is false: {e} is in Saxony")
c("AC08", AC, "U", ["plauen instance_of grosse_kreisstadt",
                    "grosse_kreisstadt located_in_the_administrative_territoria bayern_germany",
                    "plauen located_in_the_administrative_territoria vogtland_district"],
  "Plauen is an instance of Grosse Kreisstadt. The Grosse Kreisstadt is located in Bayern Germany.",
  GK.format(e="Plauen", own="Vogtland District"))
c("AC09", AC, "U", ["budyssin located_in_the_administrative_territoria bautzen_district",
                    "budyssin instance_of grosse_kreisstadt",
                    "grosse_kreisstadt located_in_the_administrative_territoria bayern_germany"],
  "Budyssin is located in Bautzen District. Budyssin is an instance of Grosse Kreisstadt. The Grosse Kreisstadt is "
  "located in Bayern Germany.", GK.format(e="Budyssin (Bautzen)", own="Bautzen District"))
c("AC10", AC, "U", ["freiberg_saxony instance_of grosse_kreisstadt",
                    "grosse_kreisstadt located_in_the_administrative_territoria bayern_germany"],
  "Freiberg Saxony is an instance of Grosse Kreisstadt. The Grosse Kreisstadt is located in Bayern Germany.",
  "'the Grosse Kreisstadt' reads as Freiberg, so the reply asserts that Freiberg Saxony is in Bavaria; not gathered, "
  "and the subject's own name says Saxony")
c("AC11", AC, "U", ["history_of_zittau instance_of grosse_kreisstadt",
                    "grosse_kreisstadt located_in_the_administrative_territoria bayern_germany",
                    "history_of_zittau located_in_the_administrative_territoria landkreis_g_rlitz"],
  "History of Zittau is an instance of Grosse Kreisstadt." + CR + LF + "The Grosse Kreisstadt is located in Bayern "
  "Germany.", GK.format(e="Zittau", own="Landkreis Goerlitz") + " (CRLF separator)")
c("AC12", AC, "U", ["house_of_saxe_meissen instance_of grosse_kreisstadt",
                    "grosse_kreisstadt located_in_the_administrative_territoria bayern_germany",
                    "house_of_saxe_meissen located_in_the_administrative_territoria mei_en_rural_district"],
  "House of Saxe Meissen is an instance of Grosse Kreisstadt. The Grosse Kreisstadt is located in Bayern Germany.",
  GK.format(e="Meissen", own="Meissen Rural District"))
c("AC13", AC, "U", ["belarusian_soviet_republic instance_of united_nations_member_states",
                    "united_nations_member_states member_of united_nations",
                    "belarusian_soviet_republic member_of united_soviets_socialist_republic"],
  "Belarusian Soviet Republic is an instance of United Nations member states. The United Nations member states is a "
  "member of the United Nations.",
  "the gathered membership of the Belarusian SSR is the USSR; the anaphoric 'the United Nations member states' reads "
  "as the Belarusian SSR and adds a second, ungathered membership (of the United Nations)")
c("AC14", AC, "U", ["paris_law_faculty instance_of faculty_of_natural_science",
                    "faculty_of_natural_science part_of higher_general_education",
                    "paris_law_faculty part_of paris_university_of"],
  "Paris law faculty is an instance of faculty of natural science. The faculty of natural science is part of higher "
  "general education.",
  "the gathered part_of for the Paris law faculty is the University of Paris; 'the faculty of natural science' reads "
  "as the Paris law faculty and adds a second, ungathered part_of (higher general education)")
c("AC15", AC, "U", ["theravaada_buddhism instance_of religions_denominations",
                    "religions_denominations part_of religious_tradition",
                    "theravaada_buddhism part_of buddhism_in_asia"],
  "Theravaada Buddhism is an instance of religions denominations. The religions denominations is part of religious "
  "tradition.",
  "anaphoric 'the religions denominations' (singular verb) reads as Theravada; the reply then gives Theravada a "
  "second part_of beside the gathered Buddhism in Asia")
c("AC16", AC, "U", ["etienne_suvasa instance_of operating_system", "operating_system has_part file_access",
                    "etienne_suvasa has_part gimp_tool_kit"],
  "Etienne Suvasa is an instance of operating system. The operating system has part file access.",
  "'the operating system' reads as Etienne Suvasa, giving it the ungathered part 'file access' beside its gathered "
  "part GIMP Tool Kit")
c("AC17", AC, "U", ["yamaguti_yamaguti instance_of city_of_japan",
                    "city_of_japan located_in_time_zone japan_standard_time",
                    "yamaguti_yamaguti located_in_time_zone korea_standard_time"],
  "Yamaguti Yamaguti is an instance of city of Japan. The city of Japan is located in time zone Japan standard time. "
  "Yamaguti Yamaguti is located in time zone Korea standard time.",
  JP.format(e="Yamaguchi") + "; the reply then states both zones")
c("AC18", AC, "U", ["hukusima_hukusima instance_of city_of_japan",
                    "city_of_japan located_in_time_zone japan_standard_time",
                    "hukusima_hukusima located_in_time_zone korea_standard_time"],
  "Hukusima Hukusima is an instance of city of Japan. The city of Japan is located in time zone Japan standard time.",
  JP.format(e="Fukushima"))
c("AC19", AC, "U", ["matudo_tiba shares_border_with history_of_ichikawa_chiba", "matudo_tiba instance_of city_of_japan",
                    "history_of_ichikawa_chiba instance_of city_of_japan",
                    "city_of_japan located_in_time_zone japan_standard_time"],
  "Matudo Tiba shares border with History of Ichikawa Chiba. The city of Japan is located in time zone Japan "
  "standard time.",
  "after a sentence naming two cities of Japan, 'the city of Japan' reads as one of them, so the reply asserts a "
  "time zone for Matsudo or Ichikawa; no such triple is gathered (the store puts both in Korea standard time)")
c("AC20", AC, "U", ["history_of_g_rlitz instance_of grosse_kreisstadt",
                    "grosse_kreisstadt located_in_the_administrative_territoria bayern_germany",
                    "history_of_g_rlitz located_in_the_administrative_territoria landkreis_g_rlitz"],
  "History of G Rlitz is an instance of Grosse Kreisstadt. The Grosse Kreisstadt is located in Bayern Germany.",
  GK.format(e="Goerlitz", own="Landkreis Goerlitz"))
c("AC21", AC, "U", ["fujisawa_kanagawa instance_of city_of_japan",
                    "city_of_japan located_in_time_zone japan_standard_time",
                    "fujisawa_kanagawa located_in_time_zone korea_standard_time"],
  "Fujisawa Kanagawa is an instance of city of Japan." + CR + "The city of Japan is located in time zone Japan "
  "standard time.", JP.format(e="Fujisawa") + " (bare CR separator)")

# ---------------------------------------------------------------------------------------------------------------
# 3. anaphoric_subclass_composite (U): sentence 1 is a subclass_of triple; "the G" in sentence 2 reads as the subclass
#    just named, which then gets the superclass's relation value (contradicting its own gathered value).
# ---------------------------------------------------------------------------------------------------------------
AS = "anaphoric_subclass_composite"
c("AS01", AS, "U", ["general_major subclass_of subofficial_mayor", "subofficial_mayor follows teniente_general",
                    "general_major follows brig_general"],
  "General major is a subclass of subofficial mayor. The subofficial mayor follows teniente general.",
  "gathered: general major follows brig general; 'the subofficial mayor' read as the general major just named makes "
  "it follow teniente general instead")
c("AS02", AS, "U", ["brig_general follows acting_colonel", "brig_general subclass_of subofficial_mayor",
                    "subofficial_mayor follows teniente_general"],
  "Brig general follows acting colonel. Brig general is a subclass of subofficial mayor. The subofficial mayor "
  "follows teniente general.",
  "the third sentence reads as continuing the story of the brig general, so the reply says it follows both acting "
  "colonel and teniente general; only the first is gathered")
c("AS03", AS, "U", ["monarch subclass_of head_of_nation", "head_of_nation part_of political_state",
                    "monarch part_of dynastic_state"],
  "Monarch is a subclass of head of nation. The head of nation is part of political state.",
  "gathered: monarch is part of dynastic state; 'the head of nation' read as the monarch just named adds an "
  "ungathered part_of (political state)")
c("AS04", AS, "U", ["feminist_art subclass_of art_form", "art_form part_of human_culture",
                    "feminist_art part_of female_rights_movement"],
  "Feminist art is a subclass of art form. The art form is part of human culture.",
  "'the art form' reads as feminist art, adding an ungathered part_of beside the gathered female rights movement")
c("AS05", AS, "U", ["book_chapter subclass_of literary_text", "literary_text part_of spoken_words",
                    "book_chapter part_of wikipedia_books"],
  "Book chapter is a subclass of literary text. The literary text is part of spoken words.",
  "'the literary text' reads as the book chapter, adding an ungathered part_of (spoken words)")
c("AS06", AS, "U", ["velodrome subclass_of circuit_racing", "circuit_racing part_of racing_unassisted_human",
                    "velodrome part_of track_cycling"],
  "Velodrome is a subclass of circuit racing. The circuit racing is part of racing unassisted human.",
  "'the circuit racing' reads as the velodrome, adding an ungathered part_of beside the gathered track cycling")
c("AS07", AS, "U", ["acting_mayor subclass_of chief_administrator", "chief_administrator part_of political_taxonomy",
                    "acting_mayor part_of incorporated_community"],
  "Acting mayor is a subclass of chief administrator. The chief administrator is part of political taxonomy.",
  "'the chief administrator' reads as the acting mayor, adding an ungathered part_of")

# ---------------------------------------------------------------------------------------------------------------
# 4. anaphoric_object_det (U): "the G" in OBJECT position of the second sentence reads as the instance E just named.
# ---------------------------------------------------------------------------------------------------------------
AO = "anaphoric_object_det"
c("AO01", AO, "U", ["federation_council instance_of first_chamber", "national_senate subclass_of first_chamber"],
  "Federation Council is an instance of first chamber. National Senate is a subclass of the first chamber.",
  "'the first chamber' reads as the Federation Council just named, so the reply says the National Senate is a "
  "subclass of the Federation Council")
c("AO02", AO, "U", ["council_of_five_hundred instance_of lower_house", "bicameral_legislature has_part lower_house"],
  "Council of Five Hundred is an instance of lower house. Bicameral legislature has part the lower house.",
  "'the lower house' reads as the Council of Five Hundred, so the reply asserts bicameral legislature has the "
  "Council of Five Hundred as a part; not gathered")
c("AO03", AO, "U", ["apple_ii instance_of personal_computer", "video_graphics_controller instance_of personal_computer"],
  "Apple II is an instance of personal computer. Video graphics controller is an instance of the personal computer.",
  "'the personal computer' reads as the Apple II: the reply says the video graphics controller is an instance of "
  "the Apple II")
c("AO04", AO, "U", ["kushiro_city instance_of city_of_japan", "morioka_city instance_of city_of_japan"],
  "Kushiro City is an instance of city of Japan. Morioka City is an instance of the city of Japan.",
  "'the city of Japan' reads as Kushiro City (or as one definite city), so Morioka is said to be an instance of it")
c("AO05", AO, "U", ["order_of_the_nile instance_of honorific_order", "lenin_medal instance_of honorific_order"],
  "Order of the Nile is an instance of honorific order. Lenin medal is an instance of the honorific order.",
  "'the honorific order' reads as the Order of the Nile, so the Lenin Medal is said to be an instance of the Order "
  "of the Nile")

# ---------------------------------------------------------------------------------------------------------------
# 5. shape2_det_second_entity (U): shape 2 with the same gathered object twice, the second with an indefinite
#    determiner. "X and a X" introduces a second, different individual of the same name: an extra entity.
#    (The store has one object per subject and relation, so a duplicate is the only way to fill shape 2.)
# ---------------------------------------------------------------------------------------------------------------
S2 = "shape2_det_second_entity"
W2 = "'and a {o}' adds a second, different {o} as an extra object; only one {o} is gathered"
c("S201", S2, "U", ["take_that has_part robbie_williams"],
  "Take That has part Robbie Williams and a Robbie Williams.", W2.format(o="Robbie Williams"))
c("S202", S2, "U", ["the_who has_part peter_dennis_blandford_townshend"],
  "The Who has part Peter Dennis Blandford Townshend and a Peter Dennis Blandford Townshend.",
  W2.format(o="Peter Townshend"))
c("S203", S2, "U", ["u_s_of_a shares_border_with canada_portal"],
  "U S of A shares border with Canada Portal and a Canada Portal.", W2.format(o="Canada Portal"))
c("S204", S2, "U", ["ukraine shares_border_with pole_land"],
  "Ukraine shares border with Pole Land and a Pole Land.", W2.format(o="Pole Land"))
c("S205", S2, "U", ["jul follows june"], "Jul follows June and a June.",
  "'and a June' adds a second June that July also follows; only one predecessor is gathered")
c("S206", S2, "U", ["johnny_and_the_moondogs has_part james_paul_mccartney"],
  "Johnny and the Moondogs has part James Paul McCartney and a James Paul McCartney.",
  W2.format(o="James Paul McCartney"))
c("S207", S2, "U", ["genesis_band has_part phil_colins"],
  "Genesis band has part Phil Colins, a Phil Colins and the Phil Colins.",
  "three-item list over one gathered member: reads as three members, two of them extra")
c("S208", S2, "U", ["castile_and_le_n contains_administrative_territorial_enti province_of_burgos"],
  "Castile and Le N contains Province of Burgos and a Province of Burgos.", W2.format(o="Province of Burgos"))
c("S209", S2, "U", ["the_rolling_stones has_part michael_philip_jagger"],
  "The Rolling Stones has part a Michael Philip Jagger and Michael Philip Jagger.",
  "the indefinite first conjunct names some other Michael Philip Jagger, then the gathered one: two members")
c("S210", S2, "U", ["second_chromosome follows first_chromosome"],
  "Second chromosome follows first chromosome and a first chromosome.",
  "'and a first chromosome' adds another first chromosome as a second predecessor")
c("S211", S2, "U", ["nine_inch_nails has_part trent_reznor"], "Nine Inch Nails has part Trent Reznor and Trent Reznor.",
  "plain duplicate conjunct: a list of two reads as two members; may be read as a harmless repetition",
  arguable=True)
c("S212", S2, "U", ["the_wailers has_part nesta_marley"], "The Wailers has part Nesta Marley and nesta_marley.",
  "the slug spelled verbatim beside the name reads as a second, separately named member; may be read as a "
  "repetition", arguable=True)
c("S213", S2, "U", ["sault_ste_marie_on shares_border_with geography_of_greater_sudbury"],
  "Sault Ste Marie ON shares border with Geography of Greater Sudbury and geography of greater sudbury.",
  "case-variant duplicate: two neighbours listed, one gathered; may be read as a repetition", arguable=True)

# ---------------------------------------------------------------------------------------------------------------
# 6. det_indefinite_proper_name (U): the spec's [Det] slot before a gathered proper name. "A <name>" means some
#    (other) individual of that name, or an artefact named after it, not the gathered entity.
# ---------------------------------------------------------------------------------------------------------------
DP = "det_indefinite_proper_name"
WD = "'{d}' turns the gathered entity into some unspecified {n}, so the claim is about a different individual"
c("DP01", DP, "U", ["robbie_williams member_of take_that"], "A Robbie Williams is a member of Take That.",
  WD.format(d="A Robbie Williams", n="person called Robbie Williams"))
c("DP02", DP, "U", ["wladimir_puttin participant_of second_chechen_war"],
  "A Wladimir Puttin is a participant of Second Chechen War.", WD.format(d="A Wladimir Puttin", n="namesake"))
c("DP03", DP, "U", ["lincoln instance_of theatrical_film"], "A Lincoln is an instance of theatrical film.",
  "'a Lincoln' reads as a Lincoln car (or someone named Lincoln), not the film Lincoln")
c("DP04", DP, "U", ["reader instance_of theatrical_film"], "A reader is an instance of theatrical film.",
  "'a reader' is any person who reads, not the film The Reader")
c("DP05", DP, "U", ["june follows may_sickness"], "June follows a May sickness.",
  "'a May sickness' reads as an illness in May, not the month the slug names")
c("DP06", DP, "U", ["may_sickness follows the_cruellest_month"], "A May sickness follows the cruellest month.",
  "'a May sickness' reads as some illness occurring in May following April; the triple orders two months")
c("DP07", DP, "U", ["john_cale part_of the_velvet_underground"], "A John Cale is part of the Velvet Underground.",
  WD.format(d="A John Cale", n="namesake"))
c("DP08", DP, "U", ["george_v member_of british_royal_society"], "A George V is a member of British Royal Society.",
  WD.format(d="A George V", n="George V (a ship, hotel or other bearer of the name)"))
c("DP09", DP, "U", ["niehls_bohr participant_of manhatten_project"],
  "A Niehls Bohr is a participant of Manhatten Project.", WD.format(d="A Niehls Bohr", n="namesake"))
c("DP10", DP, "U", ["take_that has_part robbie_williams"], "Take That has part a Robbie Williams.",
  "the object becomes some person named Robbie Williams, not the gathered one")
c("DP11", DP, "U", ["charles_ii_of_scotland sibling james_vii_of_scots"],
  "Charles II of Scotland is a sibling of a James VII of Scots.",
  "'a James VII of Scots' is some bearer of that title, not the gathered brother")
c("DP12", DP, "U", ["harvey_winesteen sibling bob_weinstein"], "A Harvey Winesteen is a sibling of Bob Weinstein.",
  WD.format(d="A Harvey Winesteen", n="namesake"))
c("DP13", DP, "U", ["robbie_williams member_of take_that"],
  "Robbie Williams is a member of Take That. A Robbie Williams is a member of Take That.",
  "two sentences: the second, indefinite one introduces another Robbie Williams as a second member")
c("DP14", DP, "U", ["ukraine shares_border_with pole_land"],
  "Ukraine shares border with Pole Land. Ukraine shares border with a Pole Land.",
  "the second sentence adds another, indefinite Pole Land as a second neighbour")
c("DP15", DP, "U", ["serena_williams participant_of tennis_at_the_2016_summer_olympics_women"],
  "A Serena Williams is a participant of Tennis at the 2016 Summer Olympics women.",
  WD.format(d="A Serena Williams", n="namesake"))
c("DP16", DP, "U", ["wacko_jacko member_of the_jackson_five"], "A Wacko Jacko is a member of the Jackson Five.",
  WD.format(d="A Wacko Jacko", n="person with that nickname"))
c("DP17", DP, "U", ["peter_sagan participant_of xxx_olympic_games"],
  "A Peter Sagan is a participant of XXX Olympic Games.", WD.format(d="A Peter Sagan", n="namesake"))
c("DP18", DP, "U", ["first_lady_of_song member_of alpha_kappa_alpha"],
  "A first lady of song is a member of Alpha Kappa Alpha.",
  "with the indefinite article the epithet becomes a description of any female star singer; the triple names one "
  "person (Ella Fitzgerald)")

# ---------------------------------------------------------------------------------------------------------------
# 7. det_forces_common_noun (U): the gathered slug is a proper name that is also a common noun (a locomotive type,
#    a butterfly genus, a rank abbreviation). An article forces the common-noun reading.
# ---------------------------------------------------------------------------------------------------------------
DC = "det_forces_common_noun"
c("DC01", DC, "U", ["american subclass_of railway_locomotive"], "An American is a subclass of railway locomotive.",
  "'an American' is a person from the United States; the slug names the 4-4-0 'American' wheel arrangement")
c("DC02", DC, "U", ["mogul subclass_of railway_locomotive"], "A mogul is a subclass of railway locomotive.",
  "'a mogul' is a magnate; the slug names the 2-6-0 'Mogul' locomotive type")
c("DC03", DC, "U", ["consolidation instance_of wheel_arrangement"],
  "A consolidation is an instance of wheel arrangement.",
  "'a consolidation' is a merger or strengthening; the slug names the 2-8-0 'Consolidation' type")
c("DC04", DC, "U", ["glider instance_of taxonomic_group"], "A glider is an instance of taxonomic group.",
  "'a glider' is an aircraft; the slug names a butterfly genus")
c("DC05", DC, "U", ["rat instance_of taxonomic_group"], "A rat is an instance of taxonomic group.",
  "'a rat' is one animal, which is not a taxonomic group; the triple is about the genus Rattus")
c("DC06", DC, "U", ["rajahs instance_of taxonomic_group"], "The rajahs is an instance of taxonomic group.",
  "'the rajahs' are Indian princes; the slug names a butterfly genus")
c("DC07", DC, "U", ["brig subclass_of career_soldier"], "A brig is a subclass of career soldier.",
  "'a brig' is a two-masted ship or a military prison; the slug abbreviates brigadier")
c("DC08", DC, "U", ["people has_part human_specie"], "A people has part human specie.",
  "'a people' is one nation or ethnic group; the slug names the general concept 'people'")
c("DC09", DC, "U", ["treasurer instance_of expert_occupation"], "A treasurer is an instance of expert occupation.",
  "'a treasurer' is a person holding the post; the triple classifies the occupation itself")

# ---------------------------------------------------------------------------------------------------------------
# 8. ordinal_name_indefinite_det (U): a name that starts with an ordinal; "A second X" means "another X".
# ---------------------------------------------------------------------------------------------------------------
OR = "ordinal_name_indefinite_det"
c("OR01", OR, "U", ["second_chromosome follows first_chromosome"], "A second chromosome follows first chromosome.",
  "'a second chromosome' means another (any further) chromosome, not chromosome 2")
c("OR02", OR, "U", ["second_baku located_in_time_zone georgia_standard_time"],
  "A second Baku is located in time zone Georgia standard time.",
  "'a second Baku' reads as another city like Baku (or Baku itself), not the town named Second Baku")
c("OR03", OR, "U", ["second_city_of_spain shares_border_with l_hospitalet_de_llobregat"],
  "A second city of Spain shares border with L Hospitalet de Llobregat.",
  "'a second city of Spain' is some other Spanish city; the slug names Barcelona ('the second city')")
c("OR04", OR, "U", ["second_commonwealth_of_poland follows imperial_germany"],
  "A second Commonwealth of Poland follows Imperial Germany.",
  "'a second Commonwealth' reads as another, new commonwealth, not the interwar state the slug names")
c("OR05", OR, "U", ["first_caliphate follows byzantine_empire"], "A first caliphate follows Byzantine Empire.",
  "'a first caliphate' is an indefinite, generic first caliphate, not the gathered Rashidun state")
c("OR06", OR, "U", ["third_asian_games follows second_asian_games"], "A third Asian Games follows Second Asian Games.",
  "'a third Asian Games' reads as yet another Asian Games; the slug names the 1958 edition")

# ---------------------------------------------------------------------------------------------------------------
# 9. copula_fusion_slug_edge (U): a slug ending in "is" ("Solomon Is" = Solomon Islands) before a copula-less
#    relation phrase ("located in time zone"). The reader takes "is" as the copula and "Solomon" as the subject.
# ---------------------------------------------------------------------------------------------------------------
CF = "copula_fusion_slug_edge"
c("CF01", CF, "U", ["solomon_is located_in_time_zone kosrae_time"], "Solomon is located in time zone Kosrae time.",
  "the store's 'Solomon Is' (Solomon Islands) plus the copula-less relation reads as 'Solomon is located ...': a "
  "different subject, Solomon")
c("CF02", CF, "U", ["marshall_is located_in_time_zone fiji_standard_time"],
  "Marshall is located in time zone Fiji standard time.",
  "'Marshall Is' (Marshall Islands) fused with the relation reads as a person or place called Marshall")
c("CF03", CF, "U", ["solomon_is located_in_time_zone kosrae_time"], "Solomon Is located in time zone Kosrae time.",
  "capitalised 'Is' still reads as the copula after 'Solomon'; subject Solomon")
c("CF04", CF, "U", ["orgryte_is headquarters_location geography_of_gothenburg"],
  "Orgryte is headquartered in Geography of Gothenburg.",
  "'Orgryte IS' (the sports club) fused with a participle relation reads as 'Orgryte is headquartered in': the "
  "subject becomes the district Orgryte")
c("CF05", CF, "U", ["marshall_is member_of united_nations", "marshall_is located_in_time_zone fiji_standard_time"],
  "Marshall Is is a member of the United Nations. Marshall is located in time zone Fiji standard time.",
  "the first sentence names 'Marshall Is'; the second reads as a different subject, Marshall")
c("CF06", CF, "U", ["marshall_is located_in_time_zone fiji_standard_time"],
  "Marshall is located in time zone Fiji standard time",
  "as CF02 without the final period")
c("CF07", CF, "U", ["biel_be located_in_the_administrative_territoria canton_de_berne"],
  "Biel be located in Canton de Berne.",
  "'Biel BE' (BE = canton of Bern) fused with 'located in' reads as a subjunctive or command ('let Biel be located "
  "in ...'), not an assertion", arguable=True)

# ---------------------------------------------------------------------------------------------------------------
# 10. quantifier_or_negation_name_initial (U): the exact store name starts with No / Not / Nothing / Every / All, so
#     the sentence parses as a quantified or negated claim.
# ---------------------------------------------------------------------------------------------------------------
QN = "quantifier_or_negation_name_initial"
c("QN01", QN, "U", ["no_limit_label instance_of record_companies"],
  "No limit label is an instance of record companies.",
  "reads as the negative universal 'no limit label is a record company'; the triple says No Limit (the label) is "
  "one")
c("QN02", QN, "U", ["no_l instance_of unisex_name"], "No l is an instance of unisex name.",
  "the de-accented store slug of 'Noel' reads as 'no l(etter L) is a unisex name', a negative claim")
c("QN03", QN, "U", ["every_little_thing_band instance_of pop_rock_band"],
  "Every little thing band is an instance of pop rock band.",
  "reads as a universal over 'little thing bands'; the triple is about the band Every Little Thing")
c("QN04", QN, "U", ["all_whites participant_of world_championship_football_2010"],
  "All whites is a participant of world championship football 2010.",
  "in lower case reads as a claim about all white people; the triple is about the New Zealand team")
c("QN05", QN, "U", ["nothing_records instance_of record_companies"],
  "Nothing records is an instance of record companies.",
  "'nothing' in subject position reads as a negative quantifier; the triple is about the label Nothing Records")
c("QN06", QN, "U", ["not_list part_of literature_by_region"], "Not list is part of literature by region.",
  "sentence-initial 'Not' reads as a negation ('not [a] list is part of ...'), not the store's name for the novel")
c("QN07", QN, "U", ["short_fiction subclass_of not_list"], "Short fiction is a subclass of not list.",
  "'a subclass of not list' reads as 'of what is not a list', a negated class")

# ---------------------------------------------------------------------------------------------------------------
# 11. connective_word_lowercase_in_slug (U): "or" / "and I" / "if" inside an exact store name, written in lower case
#     and without commas, reads as a disjunction, an extra conjunct or a conditional.
# ---------------------------------------------------------------------------------------------------------------
CN = "connective_word_lowercase_in_slug"
c("CN01", CN, "U", ["greater_eugene shares_border_with albany_lebanon_or_sa"],
  "Greater Eugene shares border with Albany Lebanon or SA.",
  "lower-case 'or' reads as a disjunction: Greater Eugene borders Albany Lebanon or (else) SA; the triple names one "
  "area (Albany-Lebanon, Oregon)")
c("CN02", CN, "U", ["greater_eugene shares_border_with albany_lebanon_or_sa"],
  "Greater Eugene shares border with Albany Lebanon or sa.",
  "as CN01, with no capital token for the caps rule to check")
c("CN03", CN, "U", ["albany_lebanon_or_sa located_in_time_zone pacific_daylight_time"],
  "Albany Lebanon or SA is located in time zone Pacific daylight time.",
  "subject reads as a disjunction ('Albany Lebanon or SA is ...'): a hedged claim about one of two places")
c("CN04", CN, "U", ["albany_lebanon_or_sa located_in_the_administrative_territoria demographics_of_oregon"],
  "Albany Lebanon or Sa is located in Demographics of Oregon.",
  "disjunctive subject; mixed-case 'Sa' avoids the all-caps check")
c("CN05", CN, "U", ["uk_of_gb_and_i follows kingdom_of_great_britian"],
  "UK of GB and I follows Kingdom of Great Britian.",
  "'... and I' can read as a coordination with the speaker ('the UK of GB and I follow ...')", arguable=True)
c("CN06", CN, "U", ["wales_u_k located_in_the_administrative_territoria uk_of_gb_and_i"],
  "Wales U K is located in UK of GB and I.",
  "sentence-final 'and I' reads as the speaker being added", arguable=True)
c("CN07", CN, "U", ["ringo_star part_of johnny_and_the_moondogs"], "Ringo Star is part of johnny and the moondogs.",
  "in lower case the band name reads as a coordination: part of Johnny and of the moondogs", arguable=True)
c("CN08", CN, "U", ["hammarby_if_ff part_of hammarby_if"], "Hammarby if FF is part of Hammarby if.",
  "lower-case 'if' (Swedish IF, a sports club) reads as a conditional", arguable=True)

# ---------------------------------------------------------------------------------------------------------------
# 12. mixed_case_acronym_evasion (U): one lower-case letter keeps a token from being "entirely capitals", so the caps
#     rule does not fire, yet the token still reads as a well-known acronym.
# ---------------------------------------------------------------------------------------------------------------
MC = "mixed_case_acronym_evasion"
c("MC01", MC, "U", ["peter_dennis_blandford_townshend member_of the_who"],
  "Peter Dennis Blandford Townshend is a member of the WHo.",
  "'the WHo' reads as the World Health Organization with a typo; the triple is about the band The Who")
c("MC02", MC, "U", ["roger_daltrey_filmography member_of the_who"],
  "Roger Daltrey Filmography is a member of the wHO.",
  "'wHO' reads as WHO; not all capitals, so an all-caps rule does not apply")
c("MC03", MC, "U", ["the_who has_part peter_dennis_blandford_townshend"],
  "The WhO has part Peter Dennis Blandford Townshend.",
  "'The WhO' reads as the organization WHO having Townshend as a part")
c("MC04", MC, "U", ["mit_ibaraki located_in_time_zone korea_standard_time"],
  "MiT Ibaraki is located in time zone Korea standard time.",
  "'MiT' reads as MIT (a university campus in Ibaraki); the slug is the city of Mito")
c("MC05", MC, "U", ["mit_ibaraki located_in_the_administrative_territoria ibaraki_prefecture"],
  "MIt Ibaraki is located in Ibaraki Prefecture.", "'MIt' reads as MIT, a different entity")
c("MC06", MC, "U", ["doctor_who_universe instance_of fictional_worlds"],
  "Doctor wHO universe is an instance of fictional worlds.",
  "'wHO' reads as the WHO: a doctor of the World Health Organization")

# ---------------------------------------------------------------------------------------------------------------
# 13. garden_path_follows_from (U)
# ---------------------------------------------------------------------------------------------------------------
c("GP01", "garden_path_follows_from", "U",
  ["goldfinger_pre_title_sequence follows from_russia_with_love_movie"],
  "Goldfinger pre title sequence follows from Russia with love movie.",
  "with the object's first word in lower case, 'follows from' reads as 'is a consequence of' (or travels from "
  "Russia); the triple orders two items of a film series")

# ---------------------------------------------------------------------------------------------------------------
# 14. split_context_carriage_return (U): a bare CR is an allowed separator between sentences. A denial follows it.
#     In a terminal a bare CR also returns the cursor, so the denial is drawn over the claim. Scored under round 3's
#     split rule (a kept span whose reading in the whole input is denied is a leak).
# ---------------------------------------------------------------------------------------------------------------
CRC = "split_context_carriage_return"
c("CR01", CRC, "U", ["robbie_williams member_of take_that"],
  "Robbie Williams is a member of Take That." + CR + "Not true.",
  "the supported clause is followed, after a bare carriage return, by a denial")
c("CR02", CRC, "U", ["jul follows june"], "Jul follows June." + CR + CR + "False.",
  "two bare carriage returns, then 'False.'")
c("CR03", CRC, "U", ["solomon_is member_of united_nations"],
  "Solomon Is is a member of the United Nations." + CR + LF + "Retracted.", "CRLF, then a retraction")
c("CR04", CRC, "U", ["ukraine shares_border_with pole_land"],
  "Untrue:" + CR + "Ukraine shares border with Pole Land.",
  "a label before a bare CR frames the clause as untrue")
c("CR05", CRC, "U", ["nine_inch_nails has_part trent_reznor"],
  "Nine Inch Nails has part Trent Reznor." + CR + "No.",
  "a bare CR then 'No.'; in a terminal 'No.' overprints the start of the claim")
c("CR06", CRC, "U", ["june follows may_sickness"], "June follows May sickness." + LF + "?",
  "a lone question mark on the next line questions the claim")
c("CR07", CRC, "U", ["the_who has_part peter_dennis_blandford_townshend"],
  "The Who has part Peter Dennis Blandford Townshend." + CR + "Wrong.", "a bare CR then 'Wrong.'")

# ---------------------------------------------------------------------------------------------------------------
# 15. det_outside_allowlist (U): determiners the spec does not allow (this, that, no, every, another, some). A checker
#     that treats any determiner as droppable would keep these.
# ---------------------------------------------------------------------------------------------------------------
DX = "det_outside_allowlist"
c("DX01", DX, "U", ["emperor_of_abyssinia instance_of monarch", "monarch part_of dynastic_state"],
  "Emperor of Abyssinia is an instance of monarch. This monarch is part of dynastic state.",
  "'this monarch' is explicitly the emperor; the reply asserts he is part of dynastic state")
c("DX02", DX, "U", ["robbie_williams member_of take_that"], "No Robbie Williams is a member of Take That.",
  "negative quantifier: denies the gathered membership")
c("DX03", DX, "U", ["city_of_japan located_in_time_zone japan_standard_time", "kushiro_city instance_of city_of_japan",
                    "kushiro_city located_in_time_zone korea_standard_time"],
  "Every city of Japan is located in time zone Japan standard time.",
  "universal claim over all cities; the gathered Kushiro City is a city of Japan in Korea standard time")
c("DX04", DX, "U", ["take_that has_part robbie_williams"], "Take That has part another Robbie Williams.",
  "'another Robbie Williams' is explicitly a different person")
c("DX05", DX, "U", ["council_of_five_hundred instance_of lower_house", "lower_house part_of bicameral_legislature"],
  "Council of Five Hundred is an instance of lower house. That lower house is part of bicameral legislature.",
  "'that lower house' is the council; the reply asserts it is part of bicameral legislature")
c("DX06", DX, "U", ["jul follows june"], "Jul follows some June.", "'some June' is an unspecified June")
c("DX07", DX, "U", ["monarch part_of dynastic_state"], "No monarch is part of dynastic state.",
  "negative quantifier denies the gathered triple")
c("DX08", DX, "U", ["robbie_williams member_of take_that"], "Every Robbie Williams is a member of Take That.",
  "universal over everyone named Robbie Williams; one membership is gathered")

# ---------------------------------------------------------------------------------------------------------------
# 16. lowercase_clause_name (U, all arguable): exact store names that are clauses; in lower case the sentence reads
#     as asserting the clause.
# ---------------------------------------------------------------------------------------------------------------
LC = "lowercase_clause_name"
c("LC01", LC, "U", ["take_that has_part robbie_williams"], "take that has part robbie williams.",
  "in lower case 'take that' reads as an imperative; the band reading needs capitals", arguable=True)
c("LC02", LC, "U", ["religion_is_dead instance_of pop_rock_band"], "religion is dead is an instance of pop rock band.",
  "the subject reads as the embedded claim 'religion is dead'", arguable=True)
c("LC03", LC, "U", ["will_hurt instance_of human_specie"], "will hurt is an instance of human specie.",
  "'will hurt' reads as a future-tense verb phrase, not the actor's name", arguable=True)
c("LC04", LC, "U", ["not_wanted instance_of human_specie"], "not wanted is an instance of human specie.",
  "reads as 'not wanted', a negated participle, rather than the store's name for a person", arguable=True)

# ---------------------------------------------------------------------------------------------------------------
# F twins (supported): every assertion is a gathered triple, plainly stated. Many are the U cases' triples in an order
# or form with no anaphor, extra entity or misleading parse; others probe retention of the exact-surface rule.
# ---------------------------------------------------------------------------------------------------------------
FT = "F_twin"
c("F01", FT, "F", ["emperor_of_abyssinia instance_of monarch", "monarch part_of dynastic_state"],
  "The monarch is part of dynastic state. Emperor of Abyssinia is an instance of monarch.",
  "reversed order of AN01: 'the monarch' has no antecedent and reads generically")
c("F02", FT, "F", ["council_of_five_hundred instance_of lower_house", "lower_house part_of bicameral_legislature"],
  "Lower house is part of bicameral legislature. Council of Five Hundred is an instance of lower house.",
  "reversed order of AN02, no article")
c("F03", FT, "F", ["city_of_japan located_in_time_zone japan_standard_time"],
  "City of Japan is located in time zone Japan standard time.", "the class triple alone")
c("F04", FT, "F", ["funabashi_chiba instance_of city_of_japan", "funabashi_chiba located_in_time_zone korea_standard_time"],
  "Funabashi Chiba is an instance of city of Japan. Funabashi Chiba is located in time zone Korea standard time.",
  "two gathered triples about one subject, no anaphor")
c("F05", FT, "F", ["plauen instance_of grosse_kreisstadt",
                   "plauen located_in_the_administrative_territoria vogtland_district"],
  "Plauen is an instance of Grosse Kreisstadt. Plauen is located in Vogtland District.", "no anaphor")
c("F06", FT, "F", ["apple_ii instance_of personal_computer", "personal_computer has_part multimedia_keyboard"],
  "Personal computer has part multimedia keyboard. Apple II is an instance of personal computer.",
  "reversed order of AN06, no article")
c("F07", FT, "F", ["doge_of_venice instance_of head_of_nation", "monarch subclass_of head_of_nation"],
  "Doge of Venice is an instance of head of nation. Monarch is a subclass of head of nation.",
  "AO-style pair without the anaphoric article")
c("F08", FT, "F", ["federation_council instance_of first_chamber", "national_senate subclass_of first_chamber"],
  "National Senate is a subclass of first chamber. Federation Council is an instance of first chamber.",
  "no article, class mentioned first")
c("F09", FT, "F", ["rat instance_of taxonomic_group", "taxonomic_group has_part flora_and_fauna"],
  "Taxonomic group has part flora and fauna. Rat is an instance of taxonomic group.", "reversed order of AN17")
c("F10", FT, "F", ["general_major follows brig_general", "general_major subclass_of subofficial_mayor"],
  "General major follows brig general. General major is a subclass of subofficial mayor.", "no anaphor")
c("F11", FT, "F", ["no_l instance_of unisex_name"], "Noël is an instance of unisex name.",
  "accented store name (row of the names table if present)")
c("F12", FT, "F", ["sa_a instance_of unisex_name"], "Saša is an instance of unisex name.", "accented name")
c("F13", FT, "F", ["province_of_l_on shares_border_with asturias_province"],
  "Province of Léon shares border with Asturias Province.", "accented name; folds to province_of_l_on")
c("F14", FT, "F", ["count_of_besan_on located_in_time_zone rome_time"],
  "Count of Besançon is located in time zone Rome time.", "accented name")
c("F15", FT, "F", ["house_of_bragan_a instance_of european_royal_houses"],
  "House of Bragança is an instance of European royal houses.", "accented name")
c("F16", FT, "F", ["castile_and_le_n shares_border_with asturias_province"],
  "Castile and León shares border with Asturias Province.",
  "accented name; 'and' inside a singular subject name")
c("F17", FT, "F", ["2014_vuelta_a_espa_a follows 2013_vuelta_a_espa_a"],
  "2014 Vuelta a España follows 2013 Vuelta a España.", "accented names with an internal 'a'")
c("F18", FT, "F", ["no_limit_label instance_of record_companies"], "No_Limit_Label is an instance of record_companies.",
  "slugs written verbatim (case-insensitive); the underscores block the quantifier reading of QN01")
c("F19", FT, "F", ["solomon_is member_of united_nations"], "Solomon_Is is a member of united_nations.",
  "slugs verbatim")
c("F20", FT, "F", ["greater_eugene shares_border_with albany_lebanon_or_sa"],
  "Greater Eugene shares border with albany_lebanon_or_sa.", "verbatim slug blocks the disjunction reading of CN01")
c("F21", FT, "F", ["roger_daltrey_filmography member_of the_who"],
  "Roger Daltrey Filmography is a member of the_who.", "verbatim slug")
c("F22", FT, "F", ["take_that has_part robbie_williams"], "Take" + NBSP + "That has part Robbie Williams.",
  "no-break space inside a name")
c("F23", FT, "F", ["robbie_williams member_of take_that"],
  "Robbie" + THINSP + "Williams is a member of Take" + NNBSP + "That.", "thin and narrow no-break spaces inside names")
c("F24", FT, "F", ["second_city_of_spain shares_border_with l_hospitalet_de_llobregat"],
  "Second" + FIGSP + "city of Spain shares border with L Hospitalet de Llobregat.", "figure space inside a name")
c("F25", FT, "F", ["kushiro_city instance_of city_of_japan"], KELVIN + "ushiro City is an instance of city of Japan.",
  "KELVIN SIGN, which NFC maps to ASCII K; displays as Kushiro")
c("F26", FT, "F", ["kamakura_japan instance_of city_of_japan"], KELVIN + "amakura Japan is an instance of city of Japan.",
  "KELVIN SIGN, as F25")
c("F27", FT, "F", ["solomon_is member_of united_nations"], "Solomon Is is a member of the United Nations.",
  "the store name 'Solomon Is' with its own copula; no fusion")
c("F28", FT, "F", ["marshall_is instance_of sovereign_states"], "Marshall Is is an instance of sovereign states.",
  "as F27")
c("F29", FT, "F", ["jul follows june", "aug follows jul"], "Jul follows June. Aug follows Jul.",
  "two-sentence composite, nothing beyond the two triples")
c("F30", FT, "F", ["take_that has_part robbie_williams", "robbie_williams member_of take_that"],
  "Take That has part Robbie Williams. Robbie Williams is a member of Take That.", "two triples")
c("F31", FT, "F", ["jul follows june", "aug follows jul"], "Jul follows June." + CR + "Aug follows Jul.",
  "bare carriage return between two supported sentences (allowed separator)")
c("F32", FT, "F", ["solomon_is member_of united_nations", "solomon_is located_in_time_zone kosrae_time"],
  "Solomon Is is a member of the United Nations." + CR + LF + "Solomon Is is located in time zone Kosrae time.",
  "CRLF between two supported sentences; the second keeps its own copula")
c("F33", FT, "F", ["the_who has_part peter_dennis_blandford_townshend"],
  "The Who has part Peter Dennis Blandford Townshend.", "plain")
c("F34", FT, "F", ["robbie_williams member_of take_that"], "Robbie Williams is a member of Take That.", "plain")
c("F35", FT, "F", ["second_city_of_spain shares_border_with l_hospitalet_de_llobregat"],
  "The second city of Spain shares border with L Hospitalet de Llobregat.",
  "definite article: 'the second city of Spain' is Barcelona, the entity the slug names")
c("F36", FT, "F", ["second_chromosome follows first_chromosome"], "Second chromosome follows first chromosome.",
  "plain")
c("F37", FT, "F", ["michael_kenji_shinoda member_of link_and_park"],
  "Michael Kenji Shinoda is a member of Link and Park.", "store name with internal 'and', title case")
c("F38", FT, "F", ["ringo_star part_of johnny_and_the_moondogs"], "Ringo Star is part of Johnny and the Moondogs.",
  "title case band name")
c("F39", FT, "F", ["emperor_of_abyssinia instance_of monarch"], "Emperor of Abyssinia is an instance of monarch.",
  "single triple")
c("F40", FT, "F", ["albany_lebanon_or_sa located_in_time_zone pacific_daylight_time"],
  "Albany Lebanon OR SA is located in time zone Pacific daylight time.",
  "capital OR reads as Oregon, as in the store's own name")
c("F41", FT, "F", ["anti_treaty_sinn_fein instance_of party_member"],
  "Anti-Treaty Sinn Fein is an instance of party member.", "hyphenated store name")
c("F42", FT, "F", ["reading_uk located_in_time_zone utc_00_00"], "Reading UK is located in time zone UTC±00:00.",
  "the zone's own name with a plus-minus sign; folds to utc_00_00")
c("F43", FT, "F", ["yangon_myanmar located_in_time_zone utc_06_30"],
  "Yangon Myanmar is located in time zone UTC+06:30.", "the zone's own name with its correct sign")
c("F44", FT, "F", ["mit_ibaraki located_in_time_zone korea_standard_time"],
  "Mit Ibaraki is located in time zone Korea standard time.", "title case, no acronym reading")
c("F45", FT, "F", ["peter_dennis_blandford_townshend member_of the_who"],
  "Peter Dennis Blandford Townshend is a member of the Who.", "lower-case article before the band name")
c("F46", FT, "F", ["from_russia_with_love_movie follows dr_no_film"], "From Russia with Love movie follows Dr No film.",
  "plain")
c("F47", FT, "F", ["wladimir_puttin participant_of second_chechen_war"],
  "Wladimir Puttin is a participant of Second Chechen War.", "plain")
c("F48", FT, "F", ["june follows may_sickness"], "June follows May sickness.", "plain")
c("F49", FT, "F", ["glider instance_of taxonomic_group"], "Glider is an instance of taxonomic group.",
  "no article, so the genus reading stands")
c("F50", FT, "F", ["second_baku located_in_time_zone georgia_standard_time"],
  "Second Baku is located in time zone Georgia standard time.", "no article")
c("F51", FT, "F", ["lincoln instance_of theatrical_film"], "Lincoln is an instance of theatrical film.", "no article")
c("F52", FT, "F", ["american subclass_of railway_locomotive"], "American is a subclass of railway locomotive.",
  "no article")
c("F53", FT, "F", ["funabashi_chiba instance_of city_of_japan", "city_of_japan located_in_time_zone japan_standard_time"],
  "City of Japan is located in time zone Japan standard time." + LF + "Funabashi Chiba is an instance of city of "
  "Japan.", "reversed order of AC01's last two sentences; no anaphor")


# ---------------------------------------------------------------------------------------------------------------
# Fold self-checks: (case id, displayed span, gathered slug). A local copy of the store's slug rule (lowercase; runs
# of [^a-z0-9] become '_'; strip '_'; cut at 40; rstrip '_') must map the span to the slug, after NFC, and the span
# must occur in the case's sentence. These spans are the non-plain spellings (accented names, signs, verbatim slugs,
# odd spaces, the Kelvin sign) and the exact store spellings the U attacks rely on.
FOLD_CHECKS = [
    ("AC07", "Hachiōji Tokyo", "hachi_ji_tokyo"),
    ("S208", "Castile and Le N", "castile_and_le_n"),
    ("CF01", "Solomon is", "solomon_is"),
    ("CF02", "Marshall is", "marshall_is"),
    ("CF04", "Orgryte is", "orgryte_is"),
    ("CF07", "Biel be", "biel_be"),
    ("QN01", "No limit label", "no_limit_label"),
    ("QN02", "No l", "no_l"),
    ("QN04", "All whites", "all_whites"),
    ("CN01", "Albany Lebanon or SA", "albany_lebanon_or_sa"),
    ("CN05", "UK of GB and I", "uk_of_gb_and_i"),
    ("CN08", "Hammarby if FF", "hammarby_if_ff"),
    ("MC01", "the WHo", "the_who"),
    ("MC04", "MiT Ibaraki", "mit_ibaraki"),
    ("GP01", "from Russia with love movie", "from_russia_with_love_movie"),
    ("AC20", "History of G Rlitz", "history_of_g_rlitz"),
    ("F11", "Noël", "no_l"),
    ("F12", "Saša", "sa_a"),
    ("F13", "Province of Léon", "province_of_l_on"),
    ("F14", "Count of Besançon", "count_of_besan_on"),
    ("F15", "House of Bragança", "house_of_bragan_a"),
    ("F16", "Castile and León", "castile_and_le_n"),
    ("F17", "2014 Vuelta a España", "2014_vuelta_a_espa_a"),
    ("F18", "No_Limit_Label", "no_limit_label"),
    ("F22", "Take" + NBSP + "That", "take_that"),
    ("F23", "Robbie" + THINSP + "Williams", "robbie_williams"),
    ("F24", "Second" + FIGSP + "city of Spain", "second_city_of_spain"),
    ("F25", KELVIN + "ushiro City", "kushiro_city"),
    ("F41", "Anti-Treaty Sinn Fein", "anti_treaty_sinn_fein"),
    ("F42", "UTC±00:00", "utc_00_00"),
    ("F43", "UTC+06:30", "utc_06_30"),
]


def store_fold(alias, maxlen=40):
    import re
    import unicodedata
    t = re.sub(r"[^a-z0-9]+", "_", unicodedata.normalize("NFC", alias).strip().lower()).strip("_")
    if len(t) > maxlen:
        t = t[:maxlen].rstrip("_")
    return t


def main():
    facts = json.load(open(STORE))
    store = set()
    rels = set()
    for x in facts:
        f = x["fact"]
        store.add((f["agent"], f["action"], f["patient"]))
        rels.add(f["action"])
    errs = []
    if len(store) != 78857 or len(rels) != 60:
        errs.append("store changed: %d facts, %d relations" % (len(store), len(rels)))
    prior = set()
    for p in PRIOR:
        for row in json.load(open(p)):
            prior.add(row["sentence"])
    keys = {"id", "gathered", "sentence", "expect", "class", "why"}
    ids, sents = set(), set()
    used_rels = set()
    for row in CASES:
        k = set(row)
        if k != keys and k != keys | {"arguable"}:
            errs.append("%s: key set %s" % (row.get("id"), sorted(k)))
        if row.get("arguable") not in (None, True):
            errs.append("%s: arguable must be true when present" % row["id"])
        if not row["id"].startswith("R6-"):
            errs.append("%s: id prefix" % row["id"])
        if row["id"] in ids:
            errs.append("%s: repeated id" % row["id"])
        ids.add(row["id"])
        if row["sentence"] in sents:
            errs.append("%s: repeated sentence" % row["id"])
        sents.add(row["sentence"])
        if row["sentence"] in prior:
            errs.append("%s: sentence identical to a round-3/4/5 sentence" % row["id"])
        if row["expect"] not in ("U", "F"):
            errs.append("%s: expect %r" % (row["id"], row["expect"]))
        if not 1 <= len(row["gathered"]) <= 4:
            errs.append("%s: %d gathered triples" % (row["id"], len(row["gathered"])))
        for t in row["gathered"]:
            if len(t) != 3:
                errs.append("%s: malformed triple %r" % (row["id"], t))
                continue
            if t[1] not in rels:
                errs.append("%s: relation %s not among the store's 60" % (row["id"], t[1]))
            if tuple(t) not in store:
                errs.append("%s: triple not in store %r" % (row["id"], t))
            used_rels.add(t[1])
        if not row["why"].strip():
            errs.append("%s: empty why" % row["id"])
    byid = {r["id"]: r for r in CASES}
    for cid, span, slug in FOLD_CHECKS:
        row = byid.get("R6-" + cid)
        if row is None:
            errs.append("fold check names a missing case %s" % cid)
            continue
        if span not in row["sentence"]:
            errs.append("R6-%s: fold span %r not in sentence" % (cid, span))
        if store_fold(span) != slug:
            errs.append("R6-%s: %r folds to %r, not %r" % (cid, span, store_fold(span), slug))
        if not any(slug in (t[0], t[2]) for t in row["gathered"]):
            errs.append("R6-%s: slug %s not gathered" % (cid, slug))
    nU = sum(r["expect"] == "U" for r in CASES)
    nF = sum(r["expect"] == "F" for r in CASES)
    if nU < 135:
        errs.append("only %d U cases (need >= 135)" % nU)
    if nF < 40:
        errs.append("only %d F cases (need >= 40)" % nF)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(CASES, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    by = collections.OrderedDict()
    for r in CASES:
        by.setdefault(r["class"], collections.Counter())[r["expect"]] += 1
    print("wrote %s: %d cases (U=%d, F=%d, arguable=%d); %d gathered relations used; %d prior sentences checked; "
          "%d fold self-checks"
          % (OUT, len(CASES), nU, nF, sum(bool(r.get("arguable")) for r in CASES), len(used_rels), len(prior),
             len(FOLD_CHECKS)))
    print("%-40s %4s %4s" % ("class", "U", "F"))
    for k, v in by.items():
        print("%-40s %4d %4d" % (k, v["U"], v["F"]))
    print("arguable:", " ".join(r["id"] for r in CASES if r.get("arguable")))
    if errs:
        print("VALIDATION FAILED:")
        for e in errs:
            print("  " + e)
        sys.exit(1)
    print("VALIDATION OK")


if __name__ == "__main__":
    main()
