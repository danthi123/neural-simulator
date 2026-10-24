#!/usr/bin/env python3
"""Round-7 blind hunt: generator + validator for the strict allowlist claim checker.

Writes cases.json next to this file from the CASES list below, then checks:
  - every gathered triple is an exact triple of the store facts.json (78,857 facts, 60 relations);
  - every relation is one of the store's relations;
  - 1-4 triples per case, the exact key set, ids prefixed R7- and unique, sentences unique;
  - no sentence is identical to any round-3 .. round-6 sentence;
  - at least 170 U and 40 F;
  - display-space self-checks: for each listed span, mapping the four display-identical spaces
    (U+00A0, U+2007, U+2009, U+202F) to U+0020, collapsing runs and lower-casing gives the slug with spaces.
Prints counts by class and label. Runs no checker.
Run: python -m research.findings.raw._moat_allowlist_hunt4.r7.build_cases   (from the repo root; the module run stamps provenance;
Claude adapted only the paths and this line, 2026-10-20)
"""
import collections
import json
import os
import sys
import unicodedata

import research.runners  # noqa: F401  (provenance door: the module run writes the cases.json sidecar)

STORE = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json"
_HERE = os.path.dirname(os.path.abspath(__file__))
PRIOR_ROOT = os.path.join(_HERE, "..", "..")
PRIOR = [
    PRIOR_ROOT + "/_moat_allowlist_hunt3/cases.json",
    PRIOR_ROOT + "/_moat_allowlist_hunt4/r4/cases.json",
    PRIOR_ROOT + "/_moat_allowlist_hunt4/r5/cases.json",
    PRIOR_ROOT + "/_moat_allowlist_hunt4/r6/cases.json",
]
OUT = os.path.join(_HERE, "cases.json")

NB = " "   # NO-BREAK SPACE
FS = " "   # FIGURE SPACE
TS = " "   # THIN SPACE
NN = " "   # NARROW NO-BREAK SPACE
DISPLAY_SPACES = (NB, FS, TS, NN)

# ---------------------------------------------------------------- gathered sets (exact store triples)
SOL_TZ = ["solomon_is", "located_in_time_zone", "kosrae_time"]
SOL_UN = ["solomon_is", "member_of", "united_nations"]
SOL_INST = ["solomon_is", "instance_of", "sovereign_states"]
MAR_TZ = ["marshall_is", "located_in_time_zone", "fiji_standard_time"]
MAR_DBS = ["marshall_is", "described_by_source", "armenian_soviet_encyclopedia"]
MAR_UN = ["marshall_is", "member_of", "united_nations"]
MAR_INST = ["marshall_is", "instance_of", "sovereign_states"]
THUN = ["thun_be", "located_in_the_administrative_territoria", "canton_de_berne"]
RW = ["robbie_williams", "member_of", "take_that"]
RW_POB = ["robbie_williams", "place_of_birth", "stoke_on_trent"]
TT = ["take_that", "has_part", "robbie_williams"]
TT_INST = ["take_that", "instance_of", "boy_band"]
JUL = ["jul", "follows", "june"]
JUL_FB = ["jul", "followed_by", "aug"]
JUL_NAMED = ["jul", "named_after", "julius_ceasar"]
JUNE = ["june", "follows", "may_sickness"]
JUNE_FB = ["june", "followed_by", "jul"]
AUG = ["aug", "follows", "jul"]
UKR = ["ukraine", "shares_border_with", "pole_land"]
UKR_UN = ["ukraine", "member_of", "united_nations"]
UKR_KYIV = ["ukraine", "contains_administrative_territorial_enti", "m_kyiv"]
UKR_FOL = ["ukraine", "follows", "ukrainian_s_s_r"]
UKR_TZ = ["ukraine", "located_in_time_zone", "west_africa_summer_time"]
POL = ["pole_land", "shares_border_with", "federal_republic_of_germany"]
POL_UN = ["pole_land", "member_of", "united_nations"]
POL_MAZ = ["pole_land", "contains_administrative_territorial_enti", "mazowsze_voivodeship"]
POL_TZ = ["pole_land", "located_in_time_zone", "rome_time"]
WHO = ["the_who", "has_part", "peter_dennis_blandford_townshend"]
PDBT = ["peter_dennis_blandford_townshend", "member_of", "the_who"]
PDBT_POB = ["peter_dennis_blandford_townshend", "place_of_birth", "england_london"]
YES = ["yes_band", "has_part", "rick_wakeman"]
RICK = ["rick_wakeman", "member_of", "yes_band"]
RICK_EDU = ["rick_wakeman", "educated_at", "royal_college_of_music"]
RICK_INSTR = ["rick_wakeman", "instrument", "keyboard_hammer"]
POLAND2 = ["second_commonwealth_of_poland", "follows", "imperial_germany"]
RABIN = ["trevor_rabin", "part_of", "yes_band"]
CHR = ["second_chromosome", "follows", "first_chromosome"]
ASIAN3 = ["third_asian_games", "follows", "second_asian_games"]
ASIAN1 = ["first_asian_games", "followed_by", "second_asian_games"]
JACKO = ["wacko_jacko", "member_of", "the_jackson_five"]
J5 = ["the_jackson_five", "has_part", "wacko_jacko"]
SAGAN = ["peter_sagan", "participant_of", "xxx_olympic_games"]
NEWBERN = ["new_bern_nc", "twinned_administrative_body", "city_of_berne"]
JERESY = ["new_jeresy", "shares_border_with", "ny_state"]
OLDZEE = ["old_zealand", "named_after", "open_sea"]
GUINEA = ["new_guinea_papua", "shares_border_with", "commonwealth_australia"]
EUGENE_TW = ["eugene_or", "twinned_administrative_body", "kathmandu_nepal"]
EUGENE_IN = ["eugene_or", "located_in_the_administrative_territoria", "greater_eugene"]
CLEVE_TW = ["cleveland_oh", "twinned_administrative_body", "ciudad_de_los_reyes"]
CLEVE_IN = ["cleveland_oh", "located_in_the_administrative_territoria", "cuyahoga_county_oh"]
DAYTON = ["dayton_oh", "twinned_administrative_body", "capital_of_serbia"]
FRANKLIN = ["franklin_county_oh", "named_after", "cilla_single"]
INDY = ["indianapolis_in", "shares_border_with", "morgan_county_in"]
PARKE = ["parke_county_in", "shares_border_with", "vigo_county"]
DOWN = ["co_down", "shares_border_with", "co_antrim"]
DERRY = ["co_derry", "shares_border_with", "co_antrim"]
JAPAN_TZ = ["city_of_japan", "located_in_time_zone", "japan_standard_time"]
KUSHIRO_I = ["kushiro_city", "instance_of", "city_of_japan"]
KUSHIRO_TZ = ["kushiro_city", "located_in_time_zone", "korea_standard_time"]
GKS = ["grosse_kreisstadt", "located_in_the_administrative_territoria", "bayern_germany"]
FREIBERG_I = ["freiberg_saxony", "instance_of", "grosse_kreisstadt"]
FREIBERG_IN = ["freiberg_saxony", "located_in_the_administrative_territoria", "landkreis_mittelsachsen"]
BRIDGES = ["jeff_bridges", "sibling", "beau_bridges"]
GIBB = ["robin_john_gibb", "member_of", "bee_gee"]


def g(*triples):
    return [list(t) for t in triples]


# ---------------------------------------------------------------- cases
# (id, gathered, sentence, expect, class, why[, arguable])
CASES = []


def add(cid, gathered, sentence, expect, cls, why, arguable=False):
    CASES.append((cid, gathered, sentence, expect, cls, why, arguable))


C = "copula_fusion_display_space"
A_WHY = ("the subject slug ends in the word 'is' ({slug}); its last space is the display-identical {sp}, so the "
         "text reads exactly like the plain sentence '{read}': the reader takes '{stem}' as the subject and 'is' as "
         "the copula, naming a different entity than the gathered islands; build 5's clause-word rule would catch "
         "this with an ordinary space, so the case asks whether its split on spaces also splits on {sp}")
add("R7-A01", g(SOL_TZ), "Solomon" + NB + "is located in time zone Kosrae time.", "U", C,
    A_WHY.format(slug="solomon_is", sp="U+00A0", read="Solomon is located in time zone Kosrae time", stem="Solomon"))
add("R7-A02", g(SOL_TZ), "Solomon" + FS + "is located in time zone Kosrae time.", "U", C,
    A_WHY.format(slug="solomon_is", sp="U+2007", read="Solomon is located in time zone Kosrae time", stem="Solomon"))
add("R7-A03", g(SOL_TZ), "Solomon" + TS + "is located in time zone Kosrae time.", "U", C,
    A_WHY.format(slug="solomon_is", sp="U+2009", read="Solomon is located in time zone Kosrae time", stem="Solomon"))
add("R7-A04", g(SOL_TZ), "Solomon" + NN + "is located in time zone Kosrae time.", "U", C,
    A_WHY.format(slug="solomon_is", sp="U+202F", read="Solomon is located in time zone Kosrae time", stem="Solomon"))
add("R7-A05", g(MAR_TZ), "Marshall" + NB + "is located in time zone Fiji standard time.", "U", C,
    A_WHY.format(slug="marshall_is", sp="U+00A0", read="Marshall is located in time zone Fiji standard time",
                 stem="Marshall"))
add("R7-A06", g(MAR_TZ), "Marshall" + FS + "is located in time zone Fiji standard time.", "U", C,
    A_WHY.format(slug="marshall_is", sp="U+2007", read="Marshall is located in time zone Fiji standard time",
                 stem="Marshall"))
add("R7-A07", g(MAR_TZ), "Marshall" + TS + "is located in time zone Fiji standard time.", "U", C,
    A_WHY.format(slug="marshall_is", sp="U+2009", read="Marshall is located in time zone Fiji standard time",
                 stem="Marshall"))
add("R7-A08", g(MAR_TZ), "Marshall" + NN + "is located in time zone Fiji standard time.", "U", C,
    A_WHY.format(slug="marshall_is", sp="U+202F", read="Marshall is located in time zone Fiji standard time",
                 stem="Marshall"))
add("R7-A09", g(MAR_DBS), "Marshall" + NB + "is described by source Armenian Soviet Encyclopedia.", "U", C,
    A_WHY.format(slug="marshall_is", sp="U+00A0",
                 read="Marshall is described by source Armenian Soviet Encyclopedia", stem="Marshall")
    + "; the relation is the telegraphic 'described by source', so the parse needs no copula of its own")
add("R7-A10", g(MAR_DBS), "Marshall" + NN + "is described by source Armenian Soviet Encyclopedia.", "U", C,
    A_WHY.format(slug="marshall_is", sp="U+202F",
                 read="Marshall is described by source Armenian Soviet Encyclopedia", stem="Marshall"))
add("R7-A11", g(MAR_DBS), "Marshall is described by source Armenian Soviet Encyclopedia.", "U", C,
    "control with an ordinary space: 'Marshall is described by source ...' names Marshall, not Marshall Is. (the "
    "islands); a new relation for round 6's copula-fusion class, which build 5's clause-word rule should reject")
add("R7-A12", g(SOL_INST), "Solomon" + NB + "is an instance of sovereign states.", "U", C,
    A_WHY.format(slug="solomon_is", sp="U+00A0", read="Solomon is an instance of sovereign states",
                 stem="Solomon") + "; it parses only if the shape-3 noun reading has a copula-less form")
add("R7-A13", g(MAR_INST), "Marshall" + TS + "is an instance of sovereign states.", "U", C,
    A_WHY.format(slug="marshall_is", sp="U+2009", read="Marshall is an instance of sovereign states",
                 stem="Marshall"))
add("R7-A14", g(SOL_UN), "Solomon" + NB + "is a member of the United Nations.", "U", C,
    A_WHY.format(slug="solomon_is", sp="U+00A0", read="Solomon is a member of the United Nations", stem="Solomon"))
add("R7-A15", g(MAR_UN), "Marshall" + FS + "is member of the United Nations.", "U", C,
    A_WHY.format(slug="marshall_is", sp="U+2007", read="Marshall is member of the United Nations",
                 stem="Marshall") + "; probes a telegraphic 'member of' reading")
add("R7-A16", g(SOL_TZ), "solomon" + NB + "is located in time zone kosrae time.", "U", C,
    A_WHY.format(slug="solomon_is", sp="U+00A0", read="solomon is located in time zone kosrae time",
                 stem="solomon") + "; all lower case, so no capital marks 'is' as part of a name")
add("R7-A17", g(SOL_TZ), "Solomon" + NB + "is located in time zone Kosrae" + NB + "time.", "U", C,
    A_WHY.format(slug="solomon_is", sp="U+00A0", read="Solomon is located in time zone Kosrae time",
                 stem="Solomon") + "; the object also uses U+00A0, as a renderer that joins every name would")
add("R7-A18", g(MAR_TZ, MAR_DBS),
    "Marshall" + NB + "is located in time zone Fiji standard time.\nMarshall" + NB
    + "is described by source Armenian Soviet Encyclopedia.", "U", C,
    "two sentences, each 'Marshall' plus U+00A0 plus 'is' before a copula-less relation; both read as claims "
    "about Marshall, not Marshall Is. (the islands)")
add("R7-A19", g(SOL_TZ), "Solomon" + NB + "is located in Kosrae time.", "U", C,
    A_WHY.format(slug="solomon_is", sp="U+00A0", read="Solomon is located in Kosrae time", stem="Solomon")
    + "; probes a 'located in' reading of the time-zone relation")
add("R7-A21", g(SOL_TZ), "Solomon" + TS + TS + "is located in time zone Kosrae time.", "U", C,
    A_WHY.format(slug="solomon_is", sp="a run of two U+2009", read="Solomon is located in time zone Kosrae time",
                 stem="Solomon") + "; the build-4 exact check collapses runs of spaces inside a name")
add("R7-A22", g(MAR_TZ), "Marshall" + NB + TS + "is located in time zone Fiji standard time.", "U", C,
    A_WHY.format(slug="marshall_is", sp="a mixed run U+00A0 U+2009",
                 read="Marshall is located in time zone Fiji standard time", stem="Marshall"))
add("R7-A20", g(THUN), "Thun" + NB + "be located in Canton de Berne.", "U", C,
    "the subject slug thun_be ends in 'be' (canton BE); with U+00A0 the text reads 'Thun be located in Canton de "
    "Berne', a subjunctive or imperative about Thun rather than a statement about the town Thun BE", True)

C = "relation_token_reuse_hedge"
add("R7-B01", g(RW), "Robbie Williams is a member of Take That in name only.", "U", C,
    "'in name only' denies real membership; it reuses no new content word beyond 'name'")
add("R7-B02", g(RW), "Robbie Williams is a member of Take That part of the time.", "U", C,
    "'part of the time' (relation tokens 'part', 'of', 'time') makes the membership intermittent")
add("R7-B03", g(TT), "Take That has part Robbie Williams only in part.", "U", C,
    "'only in part' repeats the relation word 'part' as a hedge: Robbie Williams belongs only partly")
add("R7-B04", g(UKR), "Ukraine shares border with Pole Land at no point.", "U", C,
    "'at no point' negates the shared border")
add("R7-B05", g(JUL), "Jul follows June at no time.", "U", C, "'at no time' negates the order")
add("R7-B06", g(PDBT), "Peter Dennis Blandford Townshend is a member of The Who part time.", "U", C,
    "'part time' adds a part-time membership claim the triple does not state")
add("R7-B07", g(WHO), "The Who has part Peter Dennis Blandford Townshend at times.", "U", C,
    "'at times' makes the part intermittent")
add("R7-B08", g(SOL_TZ), "Solomon Is is located in time zone Kosrae time part of the time.", "U", C,
    "'part of the time' (relation tokens 'time', 'part') says the islands are in Kosrae time only sometimes")
add("R7-B09", g(RICK), "Rick Wakeman is a member of Yes band for a time only.", "U", C,
    "'for a time only' says the membership was temporary and has ended")
add("R7-B10", g(JUL_NAMED), "Jul is named after Julius Ceasar only in part.", "U", C,
    "'only in part' hedges the naming")
add("R7-B11", g(POL_UN), "Pole Land is a member of the United Nations in name only.", "U", C,
    "'in name only' denies real membership")
add("R7-B12", g(SOL_INST), "Solomon Is is an instance of sovereign states in some instances only.", "U", C,
    "'in some instances only' (relation token 'instance') hedges the classification")
add("R7-B13", g(MAR_TZ), "Marshall Is is located in time zone Fiji standard time at times only.", "U", C,
    "'at times only' makes the time zone intermittent")
add("R7-B14", g(POL), "Pole Land shares border with Federal Republic of Germany at no time.", "U", C,
    "'at no time' negates the border")

C = "modality_aspect"
add("R7-C01", g(JUL), "Jul may follow June.", "U", C, "epistemic 'may' hedges a fact the triple states outright")
add("R7-C02", g(UKR), "Ukraine might share border with Pole Land.", "U", C, "'might' hedges the border")
add("R7-C03", g(RW), "Robbie Williams could be a member of Take That.", "U", C, "'could be' hedges the membership")
add("R7-C04", g(RICK), "Rick Wakeman used to be a member of Yes band.", "U", C,
    "'used to be' asserts the membership has ended, which the present-tense triple does not say")
add("R7-C05", g(POL_UN), "Pole Land is about to become a member of the United Nations.", "U", C,
    "'is about to become' asserts Pole Land is not yet a member, contradicting the triple")
add("R7-C06", g(UKR), "Ukraine is set to share border with Pole Land.", "U", C,
    "'is set to' places the border in the future, so it does not hold now")
add("R7-C07", g(JUL), "Jul is expected to follow June.", "U", C, "'is expected to' turns the fact into a prediction")
add("R7-C08", g(SOL_TZ), "Solomon Is is thought to be located in time zone Kosrae time.", "U", C,
    "'is thought to be' hedges the time zone")
add("R7-C09", g(PDBT), "Peter Dennis Blandford Townshend is rumoured to be a member of The Who.", "U", C,
    "'is rumoured to be' reports the membership as rumour")
add("R7-C10", g(TT), "Take That seems to have part Robbie Williams.", "U", C, "'seems to' hedges the part")
add("R7-C11", g(MAR_UN), "Marshall Is appears to be a member of the United Nations.", "U", C,
    "'appears to be' hedges the membership")
add("R7-C12", g(UKR), "Ukraine has stopped sharing border with Pole Land.", "U", C,
    "'has stopped' asserts the border no longer exists")
add("R7-C13", g(YES), "Yes band has yet to have part Rick Wakeman.", "U", C,
    "'has yet to' asserts Rick Wakeman is not yet part of the band")
add("R7-C14", g(AUG), "Aug might not follow Jul.", "U", C, "modal plus negation doubts the order")
add("R7-C15", g(CHR), "Second chromosome would probably follow first chromosome.", "U", C,
    "'would probably' hedges the order")

C = "comparative_superlative"
add("R7-D01", g(RW), "Robbie Williams is the most famous member of Take That.", "U", C,
    "superlative ranking among members, not in the triple")
add("R7-D02", g(UKR), "Ukraine shares its longest border with Pole Land.", "U", C,
    "claims the Pole Land border is Ukraine's longest; the triple says nothing of length (and it is false)")
add("R7-D03", g(RICK), "Rick Wakeman is the oldest member of Yes band.", "U", C, "age ranking not in the triple")
add("R7-D04", g(POL_UN, UKR_UN), "Pole Land is a larger member of the United Nations than Ukraine.", "U", C,
    "a size comparison between two members that no gathered triple states")
add("R7-D05", g(PDBT), "Peter Dennis Blandford Townshend is the best known member of The Who.", "U", C,
    "fame ranking not in the triple")
add("R7-D06", g(SOL_UN), "Solomon Is is the largest member of the United Nations.", "U", C,
    "size ranking, not in the triple and false")
add("R7-D07", g(UKR), "Ukraine shares a longer border with Pole Land than with any other country.", "U", C,
    "a comparison with Ukraine's other borders, none gathered")
add("R7-D08", g(MAR_INST), "Marshall Is is the smallest instance of sovereign states.", "U", C,
    "size ranking not in the triple")
add("R7-D09", g(JUL, AUG), "Jul follows June more closely than Aug follows Jul.", "U", C,
    "compares the closeness of two orderings; neither triple grades closeness")
add("R7-D10", g(TT), "Take That has part Robbie Williams more prominently than any other member.", "U", C,
    "a prominence ranking and other members, none gathered")

C = "apposition_relative_smuggle"
add("R7-E01", g(RW), "Robbie Williams, who is a member of The Who, is a member of Take That.", "U", C,
    "the relative clause smuggles membership of The Who")
add("R7-E02", g(UKR), "Ukraine, which is part of Pole Land, shares border with Pole Land.", "U", C,
    "the relative clause smuggles Ukraine being part of Pole Land")
add("R7-E03", g(JUL, AUG), "Jul, which follows Aug, follows June.", "U", C,
    "the relative clause reverses the gathered (aug, follows, jul)")
add("R7-E04", g(TT), "Take That, a band that has part Rick Wakeman, has part Robbie Williams.", "U", C,
    "the appositive smuggles Rick Wakeman as a part of Take That")
add("R7-E05", g(SOL_UN), "Solomon Is, a member of the European Union, is a member of the United Nations.", "U", C,
    "the appositive smuggles EU membership")
add("R7-E06", g(MAR_TZ, MAR_UN),
    "Marshall Is, which is located in time zone Kosrae time, is a member of the United Nations.", "U", C,
    "the relative clause gives the wrong time zone (gathered: Fiji standard time)")
add("R7-E07", g(POL, UKR_KYIV), "Pole Land, which contains M Kyiv, shares border with Federal Republic of Germany.",
    "U", C, "the relative clause moves M Kyiv into Pole Land (gathered: Ukraine contains it)")
add("R7-E08", g(RICK), "Rick Wakeman, the founder of Yes band, is a member of Yes band.", "U", C,
    "the appositive smuggles 'founder'")
add("R7-E09", g(JUNE), "June, the month after Jul, follows May sickness.", "U", C,
    "the appositive puts June after Jul, which is false")
add("R7-E10", g(UKR), "Ukraine shares border with Pole Land, which is part of Ukraine.", "U", C,
    "the relative clause makes Pole Land part of Ukraine")
add("R7-E11", g(RW, WHO), "Robbie Williams is a member of Take That, which has part Peter Dennis Blandford Townshend.",
    "U", C, "the relative clause gives Take That the gathered part of The Who")
add("R7-E12", g(PDBT), "Peter Dennis Blandford Townshend is a member of The Who, a band that has part Robbie Williams.",
    "U", C, "the appositive puts Robbie Williams in The Who")

C = "ellipsis_gapping"
add("R7-GP01", g(JUL, JUNE_FB), "Jul follows June, and June Jul.", "U", C,
    "the gapped conjunct asserts 'June follows Jul', the reverse of the gathered order")
add("R7-GP02", g(UKR, POL, UKR_KYIV), "Ukraine shares border with Pole Land, and Pole Land with M Kyiv.", "U", C,
    "the gapped conjunct asserts Pole Land borders M Kyiv")
add("R7-GP03", g(RW, RICK), "Robbie Williams is a member of Take That, and Rick Wakeman of Take That too.", "U", C,
    "the gapped conjunct puts Rick Wakeman in Take That")
add("R7-GP04", g(TT, YES), "Take That has part Robbie Williams, and Yes band Robbie Williams as well.", "U", C,
    "the gapped conjunct puts Robbie Williams in Yes band")
add("R7-GP05", g(RICK, PDBT),
    "Rick Wakeman is a member of Yes band, and Peter Dennis Blandford Townshend of Yes band too.", "U", C,
    "the gapped conjunct puts Townshend in Yes band")
add("R7-GP06", g(JUL, AUG), "Jul follows June, and June Aug.", "U", C,
    "the gapped conjunct asserts 'June follows Aug'")
add("R7-GP07", g(SOL_TZ, MAR_TZ), "Solomon Is is located in time zone Kosrae time, and Marshall Is too.", "U", C,
    "the elided conjunct puts Marshall Is in Kosrae time (gathered: Fiji standard time)")
add("R7-GP08", g(UKR_UN, UKR_KYIV), "Ukraine is a member of the United Nations, and so is M Kyiv.", "U", C,
    "the elided conjunct makes the city M Kyiv a UN member")
add("R7-GP09", g(MAR_DBS, SOL_UN),
    "Marshall Is is described by source Armenian Soviet Encyclopedia, and Solomon Is as well.", "U", C,
    "the elided conjunct asserts a source for Solomon Is that is not gathered")

C = "counterfactual_conditional"
add("R7-CF01", g(RW), "Robbie Williams would have been a member of Take That, but he never joined.", "U", C,
    "the counterfactual and 'never joined' deny the membership")
add("R7-CF02", g(UKR), "Had Ukraine shared border with Pole Land, the two would be neighbours.", "U", C,
    "a counterfactual antecedent implies there is no shared border")
add("R7-CF03", g(RICK), "Rick Wakeman would be a member of Yes band if he had not left.", "U", C,
    "implies he left, so he is not a member")
add("R7-CF04", g(SOL_TZ), "Solomon Is would be located in time zone Kosrae time if it lay further east.", "U", C,
    "implies it is not in Kosrae time")
add("R7-CF05", g(JUL), "Jul would follow June if the calendar were different.", "U", C,
    "implies Jul does not follow June in the actual calendar")
add("R7-CF06", g(TT), "Take That would have part Robbie Williams had he stayed.", "U", C,
    "implies he did not stay, so he is not a part")
add("R7-CF07", g(POL), "Pole Land would share border with Federal Republic of Germany if the border had not moved.",
    "U", C, "implies there is no shared border now")
add("R7-CF08", g(PDBT), "Peter Dennis Blandford Townshend could have been a member of The Who.", "U", C,
    "'could have been' implies he was not")

C = "report_attribution"
add("R7-H01", g(JUL), "According to rumour, Jul follows June.", "U", C, "reported as rumour, not asserted")
add("R7-H02", g(RW), "Some fans wrongly claim Robbie Williams is a member of Take That.", "U", C,
    "'wrongly claim' denies the membership")
add("R7-H03", g(RICK), "Critics deny that Rick Wakeman is a member of Yes band.", "U", C,
    "reports a denial of the membership")
add("R7-H04", g(UKR), "It is disputed whether Ukraine shares border with Pole Land.", "U", C,
    "presents the border as disputed")
add("R7-H05", g(SOL_UN), "Nobody believes Solomon Is is a member of the United Nations.", "U", C,
    "asserts universal disbelief in the membership")
add("R7-H06", g(MAR_TZ), "Wikipedia mistakenly says Marshall Is is located in time zone Fiji standard time.", "U", C,
    "'mistakenly' denies the time zone")
add("R7-H07", g(TT), "\"Take That has part Robbie Williams\" is a myth.", "U", C, "calls the quoted fact a myth")
add("R7-H08", g(PDBT), "Rumours that Peter Dennis Blandford Townshend is a member of The Who are false.", "U", C,
    "calls the membership false")

C = "quantity_numeral"
add("R7-I01", g(TT), "Take That has part two Robbie Williams.", "U", C, "asserts two individuals named Robbie Williams")
add("R7-I02", g(UKR), "Ukraine shares border with Pole Land twice.", "U", C, "asserts two separate borders")
add("R7-I03", g(UKR), "Ukraine shares 9 km of border with Pole Land.", "U", C,
    "a border length the triple lacks, and false")
add("R7-I04", g(JUL), "Jul follows June by three months.", "U", C, "a three-month gap, false and not gathered")
add("R7-I05", g(RW), "Robbie Williams is one of nine members of Take That.", "U", C, "a member count not gathered")
add("R7-I06", g(YES), "Yes band has part Rick Wakeman and four other keyboardists.", "U", C,
    "four further parts not gathered")
add("R7-I07", g(WHO), "The Who has part 2 Peter Dennis Blandford Townshend.", "U", C, "a digit count of two")
add("R7-I08", g(SOL_TZ), "Solomon Is is located in time zone Kosrae time for half of the year.", "U", C,
    "limits the time zone to half the year")
add("R7-I09", g(MAR_UN), "Marshall Is is a member of the United Nations three times over.", "U", C,
    "a threefold membership")
add("R7-I10", g(JUNE), "June follows May sickness 12 times a year.", "U", C, "a frequency of 12 a year")
add("R7-I11", g(POL), "Pole Land shares border with Federal Republic of Germany along 4000 km.", "U", C,
    "a border length not gathered and false")
add("R7-I12", g(RICK), "Rick Wakeman is a member of Yes band and of 5 other bands.", "U", C,
    "five further memberships not gathered")

C = "genitive_of_scope"
add("R7-J01", g(RW), "Robbie Williams is a member of the board of Take That.", "U", C,
    "'the board of' makes him a board member, a different body")
add("R7-J02", g(UKR), "Ukraine shares border with the capital of Pole Land.", "U", C,
    "the capital of Pole Land, a different entity that Ukraine does not border")
add("R7-J03", g(RICK), "Rick Wakeman is a member of the fan club of Yes band.", "U", C, "a fan club, not the band")
add("R7-J04", g(PDBT), "Peter Dennis Blandford Townshend is a member of the tribute band of The Who.", "U", C,
    "a tribute band, not The Who")
add("R7-J05", g(SOL_UN), "Solomon Is is a member of the Security Council of the United Nations.", "U", C,
    "the Security Council, not the UN")
add("R7-J06", g(TT), "Take That has part the brother of Robbie Williams.", "U", C,
    "the brother of Robbie Williams is another person")
add("R7-J07", g(JUL_NAMED), "Jul is named after the son of Julius Ceasar.", "U", C,
    "names the son, not Julius Ceasar")
add("R7-J08", g(POL), "Pole Land shares border with the south of Federal Republic of Germany.", "U", C,
    "locates the border in Germany's south, which is false and not gathered")
add("R7-J09", g(RW), "Robbie Williams is a member of the rival of Take That.", "U", C,
    "the rival of Take That is another band")

C = "symmetric_reciprocal"
add("R7-K01", g(JUL), "Jul and June follow each other.", "U", C, "reciprocal: also asserts June follows Jul")
add("R7-K02", g(RW), "Robbie Williams and Take That are members of each other.", "U", C,
    "reciprocal: also makes Take That a member of Robbie Williams")
add("R7-K03", g(TT), "Take That and Robbie Williams have part one another.", "U", C,
    "reciprocal: also makes Take That a part of Robbie Williams")
add("R7-K04", g(CHR), "Second chromosome and first chromosome follow each other.", "U", C,
    "reciprocal: also asserts the first follows the second")
add("R7-K05", g(AUG), "Aug and Jul are followed by each other.", "U", C, "reciprocal: also asserts Aug precedes Jul")
add("R7-K06", g(WHO), "The Who and Peter Dennis Blandford Townshend are part of each other.", "U", C,
    "reciprocal: also makes the band part of Townshend")
add("R7-K07", g(RICK), "Yes band and Rick Wakeman are members of one another.", "U", C,
    "reciprocal: also makes Yes band a member of Rick Wakeman")
add("R7-K08", g(JUNE), "June and May sickness follow one another.", "U", C,
    "reciprocal: also asserts May follows June")

C = "time_index_change"
add("R7-L01", g(RW), "Robbie Williams has been a member of Take That since 1975.", "U", C,
    "a start date not gathered (and before the band existed)")
add("R7-L02", g(UKR), "Ukraine has shared border with Pole Land only since 2015.", "U", C, "a start date not gathered")
add("R7-L03", g(RICK), "Rick Wakeman left Yes band.", "U", C, "asserts he left, so the membership ended")
add("R7-L04", g(RW), "Robbie Williams quit Take That.", "U", C, "asserts he quit")
add("R7-L05", g(JUL), "Jul followed June until 1900.", "U", C, "the order is said to have ended in 1900")
add("R7-L06", g(POL_UN), "Pole Land joined the United Nations in 2004.", "U", C, "a joining date not gathered, false")
add("R7-L07", g(SOL_TZ), "Solomon Is was located in time zone Kosrae time until last year.", "U", C,
    "says the time zone no longer applies")
add("R7-L08", g(MAR_UN), "Marshall Is is no longer a member of the United Nations.", "U", C,
    "'no longer' denies the membership")
add("R7-L09", g(TT), "Take That had part Robbie Williams before 1950.", "U", C, "a date not gathered")
add("R7-L10", g(PDBT), "Peter Dennis Blandford Townshend became a member of The Who in 2020.", "U", C,
    "a joining date not gathered, false")

C = "cross_triple_swap"
add("R7-M01", g(SOL_TZ, MAR_TZ), "Solomon Is is located in time zone Fiji standard time.", "U", C,
    "gives Solomon Is the time zone gathered for Marshall Is")
add("R7-M02", g(SOL_TZ, MAR_TZ), "Marshall Is is located in time zone Kosrae time.", "U", C,
    "gives Marshall Is the time zone gathered for Solomon Is")
add("R7-M03", g(RW, RICK), "Robbie Williams is a member of Yes band.", "U", C, "swaps the bands of two gathered facts")
add("R7-M04", g(RW, RICK), "Rick Wakeman is a member of Take That.", "U", C, "swaps the bands of two gathered facts")
add("R7-M05", g(UKR, POL), "Ukraine shares border with Federal Republic of Germany.", "U", C,
    "a neighbour of Pole Land given to Ukraine")
add("R7-M06", g(JUL, JUL_NAMED, JUNE), "June is named after Julius Ceasar.", "U", C,
    "Jul's eponym given to June")
add("R7-M07", g(WHO, TT), "The Who has part Robbie Williams.", "U", C, "Take That's member given to The Who")
add("R7-M08", g(WHO, TT), "Take That has part Peter Dennis Blandford Townshend.", "U", C,
    "The Who's member given to Take That")
add("R7-M09", g(EUGENE_TW, EUGENE_IN), "Greater Eugene is twinned with Kathmandu Nepal.", "U", C,
    "the twinning of eugene_or given to greater_eugene, a gathered slug that contains 'eugene'")
add("R7-M10", g(THUN, NEWBERN), "New Bern NC is located in Canton de Berne.", "U", C,
    "puts the North Carolina town in the Swiss canton gathered for Thun")
add("R7-M11", g(CLEVE_TW, CLEVE_IN), "Cuyahoga County OH is twinned with Ciudad de los Reyes.", "U", C,
    "the twinning of cleveland_oh given to its county")
add("R7-M12", g(DAYTON, FRANKLIN), "Dayton OH is named after Cilla Single.", "U", C,
    "Franklin County OH's eponym given to Dayton OH")
add("R7-M13", g(INDY, PARKE), "Indianapolis IN shares border with Vigo County.", "U", C,
    "Parke County's neighbour given to Indianapolis")
add("R7-M14", g(RABIN, RW), "Trevor Rabin is part of Take That.", "U", C, "Trevor Rabin moved from Yes band to Take That")
add("R7-M15", g(UKR_FOL, UKR_KYIV), "M Kyiv follows Ukrainian S S R.", "U", C,
    "Ukraine's predecessor given to its capital")
add("R7-M16", g(POL_TZ, UKR_TZ), "Ukraine is located in time zone Rome time.", "U", C,
    "Pole Land's time zone given to Ukraine")
add("R7-M17", g(DOWN, DERRY), "Co Down shares border with Co Derry.", "U", C,
    "two counties that each border Co Antrim, asserted to border each other")

C = "shape2_cross_relation_conjunct"
add("R7-N01", g(TT, TT_INST), "Take That has part Robbie Williams and boy band.", "U", C,
    "the second conjunct is Take That's class, asserted as a part")
add("R7-N02", g(UKR, UKR_KYIV), "Ukraine shares border with Pole Land and M Kyiv.", "U", C,
    "M Kyiv lies inside Ukraine; it is not a neighbour")
add("R7-N03", g(JUL, JUL_FB), "Jul follows June and Aug.", "U", C, "Jul does not follow Aug; Aug follows Jul")
add("R7-N04", g(RW, RW_POB), "Robbie Williams is a member of Take That and Stoke on Trent.", "U", C,
    "his birthplace asserted as something he is a member of")
add("R7-N05", g(RICK, RICK_INSTR), "Rick Wakeman is a member of Yes band and Keyboard Hammer.", "U", C,
    "his instrument asserted as something he is a member of")
add("R7-N06", g(POL, POL_MAZ), "Pole Land shares border with Federal Republic of Germany and Mazowsze Voivodeship.",
    "U", C, "a province inside Pole Land asserted as a neighbour")
add("R7-N07", g(JUNE, JUNE_FB), "June follows May sickness and Jul.", "U", C, "June does not follow Jul")
add("R7-N08", g(PDBT, PDBT_POB),
    "Peter Dennis Blandford Townshend is a member of The Who and England London.", "U", C,
    "his birthplace asserted as a membership")

C = "near_negative_adverb"
add("R7-P01", g(UKR), "Ukraine nearly shares border with Pole Land.", "U", C, "'nearly' implies it does not")
add("R7-P02", g(RW), "Robbie Williams is almost a member of Take That.", "U", C, "'almost' implies he is not")
add("R7-P03", g(TT), "Take That hardly has part Robbie Williams.", "U", C, "'hardly' implies almost not")
add("R7-P04", g(SOL_UN), "Solomon Is is scarcely a member of the United Nations.", "U", C,
    "'scarcely' implies almost not")
add("R7-P05", g(POL), "Pole Land all but shares border with Federal Republic of Germany.", "U", C,
    "'all but' implies it does not")
add("R7-P06", g(JUL), "Jul almost follows June.", "U", C, "'almost' implies it does not")
add("R7-P07", g(RICK), "Rick Wakeman is hardly a member of Yes band.", "U", C, "'hardly' implies almost not")
add("R7-P08", g(MAR_TZ), "Marshall Is is nearly located in time zone Fiji standard time.", "U", C,
    "'nearly' implies it is not")

C = "exclusive_focus"
add("R7-Q01", g(RW), "Only Robbie Williams is a member of Take That.", "U", C, "exclusivity: no other member")
add("R7-Q02", g(TT), "Take That has part only Robbie Williams.", "U", C, "exclusivity: no other part")
add("R7-Q03", g(UKR), "Ukraine shares border only with Pole Land.", "U", C, "exclusivity: no other neighbour")
add("R7-Q04", g(RICK), "Rick Wakeman alone is a member of Yes band.", "U", C, "exclusivity: no other member")
add("R7-Q05", g(SOL_UN), "Solomon Is is the only member of the United Nations.", "U", C,
    "exclusivity: no other UN member")
add("R7-Q06", g(PDBT), "Peter Dennis Blandford Townshend is the sole member of The Who.", "U", C,
    "exclusivity: no other member")
add("R7-Q07", g(YES), "Yes band has part Rick Wakeman exclusively.", "U", C, "exclusivity: no other part")

C = "generic_quantified_class_subject"
add("R7-R01", g(JAPAN_TZ, KUSHIRO_I, KUSHIRO_TZ), "Each city of Japan is located in time zone Japan standard time.",
    "U", C, "a universal over instances; the gathered instance Kushiro City is in Korea standard time")
add("R7-R02", g(JAPAN_TZ, KUSHIRO_I, KUSHIRO_TZ), "Any city of Japan is located in time zone Japan standard time.",
    "U", C, "free-choice 'any' over instances, contradicted by the gathered Kushiro City")
add("R7-R03", g(GKS, FREIBERG_I, FREIBERG_IN), "Each Grosse Kreisstadt is located in Bayern Germany.", "U", C,
    "a universal over instances; the gathered instance Freiberg Saxony is in Landkreis Mittelsachsen")
add("R7-R04", g(GKS, FREIBERG_I, FREIBERG_IN), "All Grosse Kreisstadt towns are located in Bayern Germany.", "U", C,
    "a universal over towns, contradicted by the gathered Freiberg Saxony")

C = "lexical_negation_antonym"
add("R7-S01", g(RW), "Robbie Williams is excluded from Take That.", "U", C, "'excluded from' denies membership")
add("R7-S02", g(UKR), "Ukraine fails to share border with Pole Land.", "U", C, "'fails to' denies the border")
add("R7-S03", g(TT), "Take That lacks Robbie Williams.", "U", C, "'lacks' denies the part")
add("R7-S04", g(JUL), "Jul fails to follow June.", "U", C, "'fails to' denies the order")
add("R7-S05", g(RICK), "Rick Wakeman was expelled from Yes band.", "U", C, "'expelled' ends the membership")
add("R7-S06", g(SOL_UN), "Solomon Is is absent from the United Nations.", "U", C, "'absent from' denies membership")
add("R7-S07", g(MAR_TZ), "Marshall Is lies outside time zone Fiji standard time.", "U", C,
    "'lies outside' denies the time zone")
add("R7-S08", g(POL), "Pole Land refuses to share border with Federal Republic of Germany.", "U", C,
    "'refuses to' implies it does not")
add("R7-S09", g(WHO), "The Who is missing Peter Dennis Blandford Townshend.", "U", C, "'is missing' denies the part")
add("R7-S10", g(JUNE), "June precedes May sickness.", "U", C, "the antonym 'precedes' reverses the order")
add("R7-S11", g(AUG), "Aug precedes Jul.", "U", C, "'precedes' reverses the order")
add("R7-S12", g(JUL), "Jul comes before June.", "U", C, "'comes before' reverses the order")

C = "name_truncation_other_entity"
add("R7-W01", g(NEWBERN), "Bern is twinned with City of Berne.", "U", C,
    "'Bern' (the Swiss capital) is not New Bern NC")
add("R7-W02", g(JERESY), "Jeresy shares border with NY State.", "U", C,
    "'Jeresy' without 'New' names the island of Jersey, not the state")
add("R7-W03", g(POLAND2), "Commonwealth of Poland follows Imperial Germany.", "U", C,
    "'Commonwealth of Poland' (the Polish-Lithuanian Commonwealth, 1569-1795) is not the Second Commonwealth "
    "of Poland, and it did not follow Imperial Germany")
add("R7-W04", g(ASIAN3), "Asian Games follows Second Asian Games.", "U", C,
    "the whole Asian Games series, not the third edition")
add("R7-W05", g(GUINEA), "Guinea shares border with Commonwealth Australia.", "U", C,
    "'Guinea' is the West African country, not New Guinea Papua")
add("R7-W06", g(GUINEA), "Papua shares border with Commonwealth Australia.", "U", C,
    "'Papua' (the Indonesian province) is not New Guinea Papua")
add("R7-W07", g(ASIAN1), "Asian Games is followed by Second Asian Games.", "U", C,
    "the series, not the first edition")
add("R7-W08", g(CHR), "Chromosome follows first chromosome.", "U", C,
    "'chromosome' in general, not the second chromosome")

C = "name_extension_other_entity"
add("R7-X01", g(RW), "Robbie Williams is a member of Take That Tribute Band.", "U", C, "a tribute band, not Take That")
add("R7-X02", g(SOL_UN), "Solomon Is is a member of the United Nations Security Council.", "U", C,
    "the Security Council, a different body")
add("R7-X03", g(POL_UN), "Pole Land is a member of the United Nations Human Rights Council.", "U", C,
    "the Human Rights Council, a different body")
add("R7-X04", g(JUL), "Jul follows June Carter.", "U", C, "June Carter is a person, not the month")
add("R7-X05", g(RICK), "Rick Wakeman is a member of Yes band Fan Club.", "U", C, "a fan club, not the band")
add("R7-X06", g(UKR), "Ukraine shares border with Lesser Pole Land.", "U", C,
    "Lesser Poland is a province that does not border Ukraine")
add("R7-X07", g(SAGAN), "Peter Sagan is a participant of XXX Olympic Games qualifiers.", "U", C,
    "the qualifiers, a different event")
add("R7-X08", g(JACKO), "Wacko Jacko is a member of the Jackson Five tribute act.", "U", C,
    "a tribute act, not the band")

C = "numeral_inside_name_change"
add("R7-Y01", g(JACKO), "Wacko Jacko is a member of the Jackson Six.", "U", C, "a different number inside the name")
add("R7-Y02", g(SAGAN), "Peter Sagan is a participant of XXXI Olympic Games.", "U", C,
    "XXXI (2016) is not XXX (2012)")
add("R7-Y03", g(ASIAN3), "Fourth Asian Games follows Second Asian Games.", "U", C,
    "the fourth games, not the third, as subject")
add("R7-Y04", g(CHR), "Second chromosome follows third chromosome.", "U", C, "the third chromosome, not the first")

C = "shape2_duplicate_variant"
add("R7-Z01", g(TT), "Take That has part Robbie Williams and a Robbie" + NB + "Williams.", "U", C,
    "'and a Robbie Williams' adds a second individual; the second copy uses U+00A0 inside the name")
add("R7-Z02", g(J5), "The Jackson Five has part Wacko Jacko and a wacko_jacko.", "U", C,
    "'and a wacko_jacko' (verbatim slug) adds a second individual")
add("R7-Z03", g(YES), "Yes band has part Rick Wakeman and another Rick Wakeman.", "U", C,
    "'another' asserts a second Rick Wakeman")

# ---------------------------------------------------------------- supported twins
C = "F_twin"
FT = [
    ("R7-F01", g(SOL_TZ), "Solomon Is is located in time zone Kosrae time.", "copula after the full name"),
    ("R7-F02", g(MAR_TZ), "Marshall Is is located in time zone Fiji standard time.", "copula after the full name"),
    ("R7-F03", g(MAR_DBS), "Marshall Is is described by source Armenian Soviet Encyclopedia.",
     "copula after the full name"),
    ("R7-F04", g(SOL_TZ), "Solomon" + NB + "Is is located in time zone Kosrae time.",
     "U+00A0 inside the name, then the copula: retention of a display space"),
    ("R7-F05", g(MAR_UN), "Marshall" + TS + "Is is a member of the United Nations.",
     "U+2009 inside the name, then the copula"),
    ("R7-F06", g(SOL_INST), "Solomon Is is an instance of sovereign states.", "shape 3"),
    ("R7-F07", g(SOL_UN), "solomon_is is a member of united_nations.", "verbatim slugs, as the stub renders them"),
    ("R7-F08", g(MAR_INST), "Marshall" + NN + "Is is an instance of sovereign states.",
     "U+202F inside the name, then the copula"),
    ("R7-F09", g(THUN), "Thun BE is located in Canton de Berne.", "the canton letters in capitals"),
    ("R7-F10", g(RW), "Robbie Williams is a member of Take That", "no final period"),
    ("R7-F11", g(TT), "Take That has part Robbie Williams.", "plain"),
    ("R7-F12", g(JUL), "Jul follows June.", "plain"),
    ("R7-F13", g(JUNE), "June follows May sickness", "no final period"),
    ("R7-F14", g(AUG), "Aug follows Jul.", "plain"),
    ("R7-F15", g(UKR), "Ukraine shares border with Pole Land.", "plain"),
    ("R7-F16", g(POL), "Pole Land shares border with Federal Republic of Germany.", "plain"),
    ("R7-F17", g(POL_UN), "Pole Land is a member of the United Nations.", "shape 3 with 'the' before the object"),
    ("R7-F18", g(UKR_UN), "Ukraine is a member of the United Nations.", "shape 3"),
    ("R7-F19", g(WHO), "The Who has part Peter Dennis Blandford Townshend", "no final period"),
    ("R7-F20", g(PDBT), "Peter Dennis Blandford Townshend is a member of The Who", "shape 3, no final period"),
    ("R7-F21", g(YES), "Yes band has part Rick Wakeman.", "plain"),
    ("R7-F22", g(RICK), "Rick Wakeman is a member of Yes band.", "shape 3"),
    ("R7-F23", g(JUL_NAMED), "Jul is named after Julius Ceasar.", "copula plus participle"),
    ("R7-F24", g(CHR), "Second chromosome follows the first chromosome.", "'the' before the object"),
    ("R7-F25", g(ASIAN3), "Third Asian Games follows Second Asian Games.", "ordinal names, plain"),
    ("R7-F26", g(JACKO), "Wacko Jacko is a member of the Jackson Five.", "number word inside the name"),
    ("R7-F27", g(SAGAN), "Peter Sagan is a participant of XXX Olympic Games.", "Roman numeral in capitals"),
    ("R7-F28", g(NEWBERN), "New Bern NC is twinned with City of Berne.", "twinning wording"),
    ("R7-F29", g(JERESY), "New Jeresy shares border with NY State.", "store misspelling kept"),
    ("R7-F30", g(OLDZEE), "Old Zealand is named after open sea.", "plain"),
    ("R7-F31", g(GUINEA), "New Guinea Papua shares border with Commonwealth Australia.", "plain"),
    ("R7-F32", g(RABIN), "Trevor Rabin is part of Yes band.", "plain"),
    ("R7-F33", g(EUGENE_TW), "Eugene OR is twinned with Kathmandu Nepal.", "state letters in capitals"),
    ("R7-F34", g(CLEVE_TW), "Cleveland OH is twinned with Ciudad de los Reyes.", "plain"),
    ("R7-F35", g(DAYTON), "Dayton OH is twinned with Capital of Serbia.", "plain"),
    ("R7-F36", g(INDY), "Indianapolis IN shares border with Morgan County IN.", "plain"),
    ("R7-F37", g(DOWN), "Co Down shares border with Co Antrim.", "plain"),
    ("R7-F38", g(UKR_FOL), "Ukraine follows Ukrainian S S R.", "single letters inside the name"),
    ("R7-F39", g(UKR_KYIV), "Ukraine contains M Kyiv.", "plain"),
    ("R7-F40", g(KUSHIRO_TZ), "Kushiro City is located in time zone Korea standard time.",
     "the store's own (wrong in the world) value, restated"),
    ("R7-F41", g(JAPAN_TZ), "The city of Japan is located in time zone Japan standard time.",
     "class entity with 'the', alone (no instance named, so no anaphor)"),
    ("R7-F42", g(FREIBERG_IN), "Freiberg Saxony is located in Landkreis Mittelsachsen.", "plain"),
    ("R7-F43", g(JUL, JUL_FB, JUL_NAMED),
     "Jul follows June. Jul is followed by Aug. Jul is named after Julius Ceasar.",
     "three distinct gathered facts in three sentences"),
    ("R7-F44", g(UKR, POL),
     "Ukraine shares border with Pole Land. Pole Land shares border with Federal Republic of Germany.",
     "two distinct gathered facts"),
    ("R7-F45", g(UKR), "Ukraine shares border with Pole" + NB + "Land.", "U+00A0 inside the object name"),
    ("R7-F46", g(PDBT), "Peter Dennis" + TS + "Blandford Townshend is a member of The Who.",
     "U+2009 inside the subject name"),
    ("R7-F47", g(RW, TT_INST), "Robbie Williams is a member of Take That. Take That is an instance of boy band.",
     "two distinct gathered facts"),
    ("R7-F48", g(BRIDGES), "Jeff Bridges is a sibling of Beau Bridges.", "shape 3, symmetric relation"),
    ("R7-F49", g(GIBB), "Robin John Gibb is a member of Bee Gee.", "store's singular band name"),
    ("R7-F50", g(POL_MAZ), "Pole Land contains Mazowsze Voivodeship.", "plain"),
    ("R7-F51", g(ASIAN1), "First Asian Games is followed by Second Asian Games.", "copula plus participle"),
    ("R7-F53", g(POLAND2), "Second Commonwealth of Poland follows Imperial Germany.", "ordinal name, plain"),
    ("R7-F54", g(RICK_EDU), "Rick Wakeman is educated at Royal College of Music.", "copula plus participle"),
    ("R7-F52", g(MAR_TZ, MAR_DBS),
     "Marshall Is is located in time zone Fiji standard time.\nMarshall Is is described by source Armenian Soviet "
     "Encyclopedia.", "two distinct facts separated by a line feed"),
]
for cid, gathered, sentence, why in FT:
    add(cid, gathered, sentence, "F", C, "supported twin: every assertion is a gathered triple; " + why)

# display-space self-checks: (case id, span as written, gathered slug)
SPAN_CHECKS = [
    ("R7-A01", "Solomon" + NB + "is", "solomon_is"), ("R7-A02", "Solomon" + FS + "is", "solomon_is"),
    ("R7-A03", "Solomon" + TS + "is", "solomon_is"), ("R7-A04", "Solomon" + NN + "is", "solomon_is"),
    ("R7-A05", "Marshall" + NB + "is", "marshall_is"), ("R7-A06", "Marshall" + FS + "is", "marshall_is"),
    ("R7-A07", "Marshall" + TS + "is", "marshall_is"), ("R7-A08", "Marshall" + NN + "is", "marshall_is"),
    ("R7-A09", "Marshall" + NB + "is", "marshall_is"), ("R7-A10", "Marshall" + NN + "is", "marshall_is"),
    ("R7-A12", "Solomon" + NB + "is", "solomon_is"), ("R7-A13", "Marshall" + TS + "is", "marshall_is"),
    ("R7-A14", "Solomon" + NB + "is", "solomon_is"), ("R7-A15", "Marshall" + FS + "is", "marshall_is"),
    ("R7-A16", "solomon" + NB + "is", "solomon_is"), ("R7-A17", "Kosrae" + NB + "time", "kosrae_time"),
    ("R7-A19", "Solomon" + NB + "is", "solomon_is"), ("R7-A20", "Thun" + NB + "be", "thun_be"),
    ("R7-A21", "Solomon" + TS + TS + "is", "solomon_is"), ("R7-A22", "Marshall" + NB + TS + "is", "marshall_is"),
    ("R7-Z01", "Robbie" + NB + "Williams", "robbie_williams"),
    ("R7-F04", "Solomon" + NB + "Is", "solomon_is"), ("R7-F05", "Marshall" + TS + "Is", "marshall_is"),
    ("R7-F08", "Marshall" + NN + "Is", "marshall_is"), ("R7-F45", "Pole" + NB + "Land", "pole_land"),
    ("R7-F46", "Peter Dennis" + TS + "Blandford Townshend", "peter_dennis_blandford_townshend"),
]


def main():
    errors = []
    facts = json.load(open(STORE))
    triples = set()
    rels = set()
    for x in facts:
        f = x["fact"]
        triples.add((f["agent"], f["action"], f["patient"]))
        rels.add(f["action"])
    if len(facts) != 78857 or len(rels) != 60:
        errors.append("store shape changed: %d facts, %d relations" % (len(facts), len(rels)))

    prior = set()
    for p in PRIOR:
        for c in json.load(open(p)):
            prior.add(c["sentence"])

    out = []
    ids, sents = set(), set()
    keys = {"id", "gathered", "sentence", "expect", "class", "why"}
    for cid, gathered, sentence, expect, cls, why, arguable in CASES:
        rec = {"id": cid, "gathered": gathered, "sentence": sentence, "expect": expect, "class": cls, "why": why}
        if arguable:
            rec["arguable"] = True
        if not cid.startswith("R7-"):
            errors.append("bad id prefix " + cid)
        if cid in ids:
            errors.append("repeated id " + cid)
        ids.add(cid)
        if sentence in sents:
            errors.append("repeated sentence %s %r" % (cid, sentence))
        sents.add(sentence)
        if sentence in prior:
            errors.append("sentence repeats a round-3..6 sentence: %s %r" % (cid, sentence))
        if expect not in ("U", "F"):
            errors.append("bad expect " + cid)
        if not (1 <= len(gathered) <= 4):
            errors.append("gathered size %s" % cid)
        for t in gathered:
            if len(t) != 3 or tuple(t) not in triples:
                errors.append("not a store triple %s %r" % (cid, t))
            elif t[1] not in rels:
                errors.append("unknown relation %s %r" % (cid, t))
        extra = set(rec) - keys - {"arguable"}
        if extra or not keys <= set(rec):
            errors.append("key set " + cid)
        if unicodedata.normalize("NFC", sentence) != sentence:
            errors.append("sentence not NFC " + cid)
        out.append(rec)

    by_id = {r["id"]: r for r in out}
    for cid, span, slug in SPAN_CHECKS:
        r = by_id.get(cid)
        if r is None or span not in r["sentence"]:
            errors.append("span check: span not in sentence " + cid)
            continue
        norm = span
        for sp in DISPLAY_SPACES:
            norm = norm.replace(sp, " ")
        norm = " ".join(w for w in unicodedata.normalize("NFC", norm).lower().split(" ") if w)  # collapse runs
        if norm != slug.replace("_", " "):
            errors.append("span check: %s %r -> %r != %r" % (cid, span, norm, slug))
        if not any(t[0] == slug or t[2] == slug for t in r["gathered"]):
            errors.append("span check: slug not gathered " + cid)

    n_u = sum(1 for r in out if r["expect"] == "U")
    n_f = sum(1 for r in out if r["expect"] == "F")
    n_arg = sum(1 for r in out if r.get("arguable"))
    if n_u < 170:
        errors.append("only %d U" % n_u)
    if n_f < 40:
        errors.append("only %d F" % n_f)

    if errors:
        for e in errors:
            print("ERROR:", e)
        sys.exit(1)

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
        fh.write("\n")

    print("wrote %s: %d cases (U=%d, F=%d, arguable=%d)" % (OUT, len(out), n_u, n_f, n_arg))
    print("relations used: %d of %d" % (len({t[1] for r in out for t in r["gathered"]}), len(rels)))
    print("prior sentences checked: %d" % len(prior))
    print("display-space span checks passed: %d" % len(SPAN_CHECKS))
    tab = collections.OrderedDict()
    for r in out:
        tab.setdefault(r["class"], collections.Counter())[r["expect"]] += 1
        if r.get("arguable"):
            tab[r["class"]]["arg"] += 1
    print("%-34s %4s %4s %4s" % ("class", "U", "F", "arg"))
    for cls, c in tab.items():
        print("%-34s %4d %4d %4d" % (cls, c["U"], c["F"], c["arg"]))
    print("arguable:", ", ".join(r["id"] for r in out if r.get("arguable")))


if __name__ == "__main__":
    main()
