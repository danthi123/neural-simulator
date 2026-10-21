#!/usr/bin/env python3
"""Round-5 blind hunt for the strict allowlist claim checker: generator + validator.

Run:  python -m research.findings.raw._moat_allowlist_hunt4.r5.build_cases   (from the repo root; the module run
stamps provenance; Claude adapted only the paths and this line, 2026-10-20)
Writes cases.json next to this file (UTF-8, real code points) and validates it:
  - every gathered triple is an exact triple of the store facts.json;
  - every relation is one of the store's 60;
  - exact key set {id, gathered, sentence, expect, class, why} (+ optional arguable: true);
  - 1-4 gathered triples per case; ids unique with prefix R5-; sentences unique;
  - no sentence identical to any round-3 or round-4 sentence;
  - fold self-checks: every attack that relies on the store's slug rule names the
    displayed span and the gathered slug it must fold to; a local copy of the store's
    `sanitize` (lowercase, runs of [^a-z0-9] -> '_', strip '_', cut at 40, rstrip '_')
    must map that span onto that slug, and the span must occur in the sentence;
  - prints counts by class and by label, and the arguable ids.
Non-ASCII characters are written below as \\u escapes so this file stays ASCII.
"""
import collections
import json
import os
import re
import sys

import research.runners  # noqa: F401  (provenance door: the module run writes the cases.json sidecar)

STORE = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json"
_HERE = os.path.dirname(os.path.abspath(__file__))
R3_CASES = os.path.join(_HERE, "..", "..", "_moat_allowlist_hunt3", "cases.json")
R4_CASES = os.path.join(_HERE, "..", "r4", "cases.json")
OUT = os.path.join(_HERE, "cases.json")

_TOK_RE = re.compile(r"[^a-z0-9]+")


def sanitize(alias, maxlen=40):
    """Local copy of the store's slug rule (research/runners/_knowledge_core_curate.py)."""
    t = _TOK_RE.sub("_", alias.strip().lower()).strip("_")
    if len(t) > maxlen:
        t = t[:maxlen].rstrip("_")
    return t


CASES = []
FOLDS = {}  # id -> list of (displayed span, gathered slug it must fold to)


def C(cid, gathered, sentence, expect, cls, why, folds=(), arguable=False):
    d = {"id": cid, "gathered": [list(t) for t in gathered], "sentence": sentence,
         "expect": expect, "class": cls, "why": why}
    if arguable:
        d["arguable"] = True
    CASES.append(d)
    if folds:
        FOLDS[cid] = list(folds)


BRS = "british_royal_society"

# ---------------------------------------------------------------------------
# numeral_extension_foldable_letter: a Latin letter that reads as a Roman-numeral
# I (or V / L) is glued to a numeral token. The store's slug rule turns it into a
# separator, so the span still spells the gathered slug, but the reader sees the
# next (or another) ordinal: a different monarch, edition or team.
# ---------------------------------------------------------------------------
NX = "numeral_extension_foldable_letter"
C("R5-NX01", [["charles_ii_of_scotland", "member_of", BRS]],
  "Charles IIļ of Scotland is a member of British Royal Society.", "U", NX,
  "l-with-cedilla after II displays as 'Charles III of Scotland', a different monarch; it folds to charles_ii_of_scotland",
  [("Charles IIļ of Scotland", "charles_ii_of_scotland")])
C("R5-NX02", [["george_v", "member_of", BRS], ["george_v", "instance_of", "human_specie"]],
  "George Vĺ is a member of British Royal Society.", "U", NX,
  "l-with-acute after V displays as 'George VI', a different king; folds to george_v",
  [("George Vĺ", "george_v")])
C("R5-NX03", [["james_vii_of_scots", "member_of", BRS]],
  "James VIIǀ of Scots is a member of British Royal Society.", "U", NX,
  "U+01C0 (Latin letter dental click, a bare vertical stroke) after VII displays as 'James VIII', the Old Pretender; folds to james_vii_of_scots",
  [("James VIIǀ of Scots", "james_vii_of_scots")])
C("R5-NX04", [["elizabeth_ii_of_guyana", "member_of", BRS]],
  "Elizabeth IIɪ of Guyana is a member of British Royal Society.", "U", NX,
  "small-capital I (U+026A, lowercase so not an all-caps token) after II displays as 'Elizabeth III'; folds to elizabeth_ii_of_guyana",
  [("Elizabeth IIɪ of Guyana", "elizabeth_ii_of_guyana")])
C("R5-NX05", [["pedro_ii", "member_of", BRS]],
  "Pedro ᴠII is a member of British Royal Society.", "U", NX,
  "small-capital V (U+1D20) before II displays as 'Pedro VII'; folds to pedro_ii",
  [("Pedro ᴠII", "pedro_ii")])
C("R5-NX06", [["games_of_the_ii_olympiad", "follows", "first_summer_olympic_games"]],
  "Games of the IIÍ Olympiad follows First Summer Olympic Games.", "U", NX,
  "I-acute after II displays as 'III Olympiad' (St Louis 1904), which does not follow the first Games; folds to games_of_the_ii_olympiad",
  [("Games of the IIÍ Olympiad", "games_of_the_ii_olympiad")])
C("R5-NX07", [["games_of_the_x_olympiad", "follows", "erven_jos_rovers"]],
  "Games of the Xļ Olympiad follows Erven Jos Rovers.", "U", NX,
  "displays as 'XI Olympiad' (Berlin 1936), a different edition; folds to games_of_the_x_olympiad",
  [("Games of the Xļ Olympiad", "games_of_the_x_olympiad")])
C("R5-NX08", [["ii_olympic_winter_games", "follows", "inaugural_winter_games"]],
  "IIĺ Olympic Winter Games follows Inaugural Winter Games.", "U", NX,
  "displays as 'III Olympic Winter Games', which does not directly follow the first; folds to ii_olympic_winter_games",
  [("IIĺ Olympic Winter Games", "ii_olympic_winter_games")])
C("R5-NX09", [["vii_commonwealth_games", "follows", "vi_commonwealth_games"]],
  "VII Commonwealth Games follows VIľ Commonwealth Games.", "U", NX,
  "l-with-caron after VI displays as 'VII', so the sentence reads 'VII follows VII' (self-succession); folds to vi_commonwealth_games",
  [("VIľ Commonwealth Games", "vi_commonwealth_games")])
C("R5-NX10", [["sega_mega_drive_ii", "follows", "sega_master_system_iii"]],
  "Sega Mega Drive IIǀ follows Sega Master System III.", "U", NX,
  "displays as 'Sega Mega Drive III', a different console; folds to sega_mega_drive_ii",
  [("Sega Mega Drive IIǀ", "sega_mega_drive_ii")])
C("R5-NX11", [["the_godfather_trilogy", "has_part", "the_godfather_part_ii"]],
  "The Godfather Trilogy has part The Godfather Part IIĺ.", "U", NX,
  "displays as 'The Godfather Part III', a different film from the gathered Part II; folds to the_godfather_part_ii",
  [("The Godfather Part IIĺ", "the_godfather_part_ii")])
C("R5-NX12", [["the_hobbit_ii", "follows", "an_unexpected_journey"]],
  "The Hobbit IIɪ follows An Unexpected Journey.", "U", NX,
  "displays as 'The Hobbit III', which follows the second film, not the first; folds to the_hobbit_ii",
  [("The Hobbit IIɪ", "the_hobbit_ii")])
C("R5-NX13", [["universite_de_paris_vii", "member_of", "consortium_couperin"]],
  "Universite de Paris VIIļ is a member of Consortium Couperin.", "U", NX,
  "displays as 'Paris VIII', a different university (a store member too, but not gathered); folds to universite_de_paris_vii",
  [("Universite de Paris VIIļ", "universite_de_paris_vii")])
C("R5-NX14", [["universit_paris_i_panth_on_sorbonne", "member_of", "consortium_couperin"]],
  "Université Paris Iĺ Panthéon Sorbonne is a member of Consortium Couperin.", "U", NX,
  "displays as 'Paris II' (the Pantheon-Assas university), not the gathered Paris I; folds to universit_paris_i_panth_on_sorbonne",
  [("Université Paris Iĺ Panthéon Sorbonne", "universit_paris_i_panth_on_sorbonne")])
C("R5-NX15", [["indochina_war", "part_of", "cold_war_i"]],
  "Indochina War is part of Cold War Iı.", "U", NX,
  "dotless i after I displays as 'Cold War II', a different period; folds to cold_war_i",
  [("Cold War Iı", "cold_war_i")])
C("R5-NX16", [["division_i_bowl", "part_of", "en_cee_double_ay"]],
  "Division Iǀ Bowl is part of En Cee Double Ay.", "U", NX,
  "displays as 'Division II Bowl', a different tier; folds to division_i_bowl",
  [("Division Iǀ Bowl", "division_i_bowl")])
C("R5-NX17", [["apple_ii", "instance_of", "personal_computer"]],
  "Apple IIÏ is an instance of personal computer.", "U", NX,
  "I-diaeresis after II displays as 'Apple III', a different computer; folds to apple_ii",
  [("Apple IIÏ", "apple_ii")])
C("R5-NX18", [["v_arrondissement", "shares_border_with", "ier_arrondissement"]],
  "Vļ arrondissement shares border with Ier arrondissement.", "U", NX,
  "displays as 'VI arrondissement', not the gathered Ve; folds to v_arrondissement",
  [("Vļ arrondissement", "v_arrondissement")])
C("R5-NX19", [["oosterhoff_type_i", "subclass_of", "star_cluster"]],
  "Oosterhoff type Iľ is a subclass of star cluster.", "U", NX,
  "displays as 'Oosterhoff type II', the other Oosterhoff class; folds to oosterhoff_type_i",
  [("Oosterhoff type Iľ", "oosterhoff_type_i")])
C("R5-NX20", [["vi_olympic_winter_games", "follows", "vth_olympic_winter_games"]],
  "VI Olympic Winter Games follows ǀVth Olympic Winter Games.", "U", NX,
  "a vertical-stroke letter before Vth displays as 'IVth', so VI is said to follow IV; folds to vth_olympic_winter_games",
  [("ǀVth Olympic Winter Games", "vth_olympic_winter_games")])
C("R5-NX21", [["mac_os_x", "instance_of", "operating_system"]],
  "Mac OS Xĺ is an instance of operating system.", "U", NX,
  "displays as 'Mac OS XI', a different release; folds to mac_os_x",
  [("Mac OS Xĺ", "mac_os_x")])
C("R5-NX22", [["x_commonwealth_games", "follows", "ix_commonwealth_games"]],
  "ĹX Commonwealth Games follows IX Commonwealth Games.", "U", NX,
  "L-acute before X displays as 'LX Commonwealth Games' (60th); folds to x_commonwealth_games",
  [("ĹX Commonwealth Games", "x_commonwealth_games")])
C("R5-NX23", [["1st_parliament_of_george_i", "follows", "4th_parliament_of_great_britain"]],
  "1st Parliament of George Iļ follows 4th Parliament of Great Britain.", "U", NX,
  "displays as 'George II', a different king's parliament; folds to 1st_parliament_of_george_i",
  [("1st Parliament of George Iļ", "1st_parliament_of_george_i")])

# ---------------------------------------------------------------------------
# paren_affix_hedge: a negating or degree prefix that belongs to the slug is put in
# parentheses or brackets. The span folds to the slug, but the reader sees an
# either/or ("governmental or non-governmental"): a hedge between opposite classes.
# ---------------------------------------------------------------------------
PA = "paren_affix_hedge"
C("R5-PA01", [["women_s_international_league", "instance_of", "non_governmental_organization"]],
  "Women's International League is an instance of (non-)governmental organization.", "U", PA,
  "'(non-)governmental' hedges between governmental and non-governmental; folds to non_governmental_organization",
  [("(non-)governmental organization", "non_governmental_organization")])
C("R5-PA02", [["non_insulin_dependent_diabetes", "subclass_of", "risk_factors_for_diabetes"]],
  "(Non-)insulin dependent diabetes is a subclass of risk factors for diabetes.", "U", PA,
  "hedges between insulin-dependent (type 1) and non-insulin-dependent (type 2) diabetes",
  [("(Non-)insulin dependent diabetes", "non_insulin_dependent_diabetes")])
C("R5-PA03", [["winchester_district", "instance_of", "non_metropolitan_districts"]],
  "Winchester District is an instance of (non-)metropolitan districts.", "U", PA,
  "hedges between metropolitan and non-metropolitan districts",
  [("(non-)metropolitan districts", "non_metropolitan_districts")])
C("R5-PA04", [["charitable_organization", "subclass_of", "not_for_profit_organization"]],
  "Charitable organization is a subclass of (not-)for-profit organization.", "U", PA,
  "hedges between for-profit and not-for-profit",
  [("(not-)for-profit organization", "not_for_profit_organization")])
C("R5-PA05", [["crime_writers", "subclass_of", "non_fiction"]],
  "Crime writers is a subclass of (non-)fiction.", "U", PA,
  "hedges between fiction and non-fiction",
  [("(non-)fiction", "non_fiction")])
C("R5-PA06", [["united_states_whig_party", "follows", "anti_jacksonian_party"]],
  "United States Whig Party follows (Anti-)Jacksonian Party.", "U", PA,
  "hedges between the Jacksonian (Democratic) party and its opponent, the Anti-Jacksonians",
  [("(Anti-)Jacksonian Party", "anti_jacksonian_party")])
C("R5-PA07", [["anti_treaty_sinn_fein", "instance_of", "party_member"]],
  "(Anti-)Treaty Sinn Fein is an instance of party member.", "U", PA,
  "hedges between the pro-Treaty and anti-Treaty factions",
  [("(Anti-)Treaty Sinn Fein", "anti_treaty_sinn_fein")])
C("R5-PA08", [["semi_automatic_rifles", "subclass_of", "revolving_rifle"]],
  "(Semi-)automatic rifles is a subclass of revolving rifle.", "U", PA,
  "hedges between automatic and semi-automatic rifles",
  [("(Semi-)automatic rifles", "semi_automatic_rifles")])
C("R5-PA09", [["semi_constitutional_monarchy", "subclass_of", "dynastic_state"]],
  "[Semi-]constitutional monarchy is a subclass of dynastic state.", "U", PA,
  "square-bracketed prefix hedges between constitutional and semi-constitutional monarchy",
  [("[Semi-]constitutional monarchy", "semi_constitutional_monarchy")])
C("R5-PA10", [["pre_dreadnought_battleship", "subclass_of", "battle_ship"]],
  "(Pre-)dreadnought battleship is a subclass of battle ship.", "U", PA,
  "hedges between dreadnought and pre-dreadnought battleships",
  [("(Pre-)dreadnought battleship", "pre_dreadnought_battleship")])
C("R5-PA11", [["a_touch_of_pink", "instance_of", "tv_mini_series"]],
  "A Touch of Pink is an instance of TV (mini-)series.", "U", PA,
  "hedges between a TV series and a TV miniseries",
  [("TV (mini-)series", "tv_mini_series")])
C("R5-PA12", [["bi_weekly_newspaper", "subclass_of", "sister_newspaper"]],
  "(Bi-)weekly newspaper is a subclass of sister newspaper.", "U", PA,
  "hedges between weekly and bi-weekly",
  [("(Bi-)weekly newspaper", "bi_weekly_newspaper")])
C("R5-PA13", [["emo_rock", "subclass_of", "post_hardcore"]],
  "Emo rock is a subclass of (post-)hardcore.", "U", PA,
  "hedges between hardcore and post-hardcore",
  [("(post-)hardcore", "post_hardcore")])
C("R5-PA14", [["hepatitis_non_b", "instance_of", "organic_cause"]],
  "Hepatitis (non-)B is an instance of organic cause.", "U", PA,
  "hedges between hepatitis B and non-B hepatitis",
  [("Hepatitis (non-)B", "hepatitis_non_b")])
C("R5-PA15", [["sub_machine_gun", "subclass_of", "target_pistol"]],
  "(Sub-)machine gun is a subclass of target pistol.", "U", PA,
  "hedges between a machine gun and a sub-machine gun",
  [("(Sub-)machine gun", "sub_machine_gun")])
C("R5-PA16", [["sound_engineering", "subclass_of", "non_degreed_engineer"]],
  "Sound engineering is a subclass of (non-)degreed engineer.", "U", PA,
  "hedges between degreed and non-degreed engineer",
  [("(non-)degreed engineer", "non_degreed_engineer")])

# ---------------------------------------------------------------------------
# paren_modifier_hedge: the slug's own modifier word (North/South/New/Lower/...) is
# parenthesised, so the span names either the gathered entity or a different,
# well-known one (Mongolia vs Inner Mongolia, Jersey vs New Jersey).
# ---------------------------------------------------------------------------
PM = "paren_modifier_hedge"
C("R5-PM01", [["south_mongolia", "shares_border_with", "liaoning_province"]],
  "(South) Mongolia shares border with Liaoning Province.", "U", PM,
  "reads 'Mongolia, or South Mongolia'; the state of Mongolia does not border Liaoning",
  [("(South) Mongolia", "south_mongolia")])
C("R5-PM02", [["new_jeresy", "shares_border_with", "ny_state"]],
  "(New) Jeresy shares border with NY State.", "U", PM,
  "reads 'Jersey, or New Jersey'; the island of Jersey does not border New York",
  [("(New) Jeresy", "new_jeresy")])
C("R5-PM03", [["new_zealnd", "shares_border_with", "commonwealth_australia"]],
  "(New) Zealnd shares border with Commonwealth Australia.", "U", PM,
  "reads 'Zealand (the Danish island), or New Zealand'",
  [("(New) Zealnd", "new_zealnd")])
C("R5-PM04", [["south_holland", "shares_border_with", "noord_brabant"]],
  "(South) Holland shares border with Noord Brabant.", "U", PM,
  "hedges between Holland and the province of South Holland",
  [("(South) Holland", "south_holland")])
C("R5-PM05", [["northern_ireland_fc", "participant_of", "world_cup_1982"]],
  "(Northern) Ireland FC is a participant of World Cup 1982.", "U", PM,
  "reads 'Ireland, or Northern Ireland'; the Republic of Ireland did not play in 1982",
  [("(Northern) Ireland FC", "northern_ireland_fc")])
C("R5-PM06", [["great_poland_voivodship", "shares_border_with", "lower_silesian_voivodship"]],
  "Great Poland Voivodship shares border with (Lower) Silesian Voivodship.", "U", PM,
  "hedges between Lower Silesian and Silesian Voivodeship; Greater Poland does not border the latter",
  [("(Lower) Silesian Voivodship", "lower_silesian_voivodship")])
C("R5-PM07", [["new_south_wales_state", "shares_border_with", "constitution_of_queensland"]],
  "(New) South Wales State shares border with Constitution of Queensland.", "U", PM,
  "reads 'South Wales (UK), or New South Wales'",
  [("(New) South Wales State", "new_south_wales_state")])
C("R5-PM08", [["west_balochistan", "shares_border_with", "south_khorasan_province"]],
  "(West) Balochistan shares border with South Khorasan Province.", "U", PM,
  "hedges between Balochistan (Pakistan) and West Balochistan (Iran); only the latter borders South Khorasan",
  [("(West) Balochistan", "west_balochistan")])
C("R5-PM09", [["new_mexcio", "shares_border_with", "lone_star_state"]],
  "(New) Mexcio shares border with Lone Star State.", "U", PM,
  "hedges between Mexico and New Mexico, two different entities",
  [("(New) Mexcio", "new_mexcio")])
C("R5-PM10", [["council_of_five_hundred", "instance_of", "lower_house"]],
  "Council of Five Hundred is an instance of (lower) house.", "U", PM,
  "hedges between a house and a lower house",
  [("(lower) house", "lower_house")])
C("R5-PM11", [["east_susex", "shares_border_with", "county_kent"]],
  "(East) Susex shares border with County Kent.", "U", PM,
  "hedges between Sussex and East Sussex",
  [("(East) Susex", "east_susex")])
C("R5-PM12", [["north_lincolnshire", "part_of", "county_of_lincoln"]],
  "(North) Lincolnshire is part of County of Lincoln.", "U", PM,
  "hedges between Lincolnshire itself and North Lincolnshire",
  [("(North) Lincolnshire", "north_lincolnshire")])
C("R5-PM13", [["uk_liberals", "follows", "old_whigs"]],
  "UK Liberals follows (Old) Whigs.", "U", PM,
  "hedges between the Whigs and the Old Whigs faction",
  [("(Old) Whigs", "old_whigs")])

# ---------------------------------------------------------------------------
# intra_slug_coordination: a coordinating or relational symbol (&, +, =, !=, subset)
# is put between two words of ONE slug ("Bayern Germany"). The span folds to the
# slug, but the reader sees two entities or an identity/subset claim.
# ---------------------------------------------------------------------------
CO = "intra_slug_coordination"
C("R5-CO01", [["hesse_germany", "shares_border_with", "bayern_germany"]],
  "Hesse Germany shares border with Bayern & Germany.", "U", CO,
  "reads as bordering Bayern and Germany; Hesse is inside Germany",
  [("Bayern & Germany", "bayern_germany")])
C("R5-CO02", [["freistaat_sachsen", "shares_border_with", "bayern_germany"]],
  "Freistaat Sachsen shares border with Bayern+Germany.", "U", CO,
  "glued '+' reads 'Bayern plus Germany'; Saxony does not border Germany",
  [("Bayern+Germany", "bayern_germany")])
C("R5-CO03", [["alicante_spain", "shares_border_with", "elche_spain"]],
  "Alicante Spain shares border with Elche & Spain.", "U", CO,
  "adds Spain as a bordering entity", [("Elche & Spain", "elche_spain")])
C("R5-CO04", [["asker_norway", "shares_border_with", "b_rum_norway"]],
  "Asker Norway shares border with Bærum&Norway.", "U", CO,
  "glued '&' reads 'Baerum and Norway'", [("Bærum&Norway", "b_rum_norway")])
C("R5-CO05", [["barking_london", "shares_border_with", "ilford_united_kingdom"]],
  "Barking London shares border with Ilford + United Kingdom.", "U", CO,
  "adds the United Kingdom as a bordering entity", [("Ilford + United Kingdom", "ilford_united_kingdom")])
C("R5-CO06", [["anhui_china", "shares_border_with", "jiangsu_province"]],
  "Anhui & China shares border with Jiangsu Province.", "U", CO,
  "subject reads as Anhui and China; China contains Jiangsu", [("Anhui & China", "anhui_china")])
C("R5-CO07", [["acre_brazil", "shares_border_with", "estado_do_amazonas"]],
  "Acre+Brazil shares border with Estado do Amazonas.", "U", CO,
  "subject reads as Acre plus Brazil; Brazil contains Amazonas", [("Acre+Brazil", "acre_brazil")])
C("R5-CO08", [["anderlecht_belgium", "shares_border_with", "city_of_brussels"]],
  "Anderlecht & Belgium shares border with City of Brussels.", "U", CO,
  "subject reads as Anderlecht and Belgium", [("Anderlecht & Belgium", "anderlecht_belgium")])
C("R5-CO09", [["ayacucho_peru", "shares_border_with", "qusqu_region"]],
  "Ayacucho & Peru shares border with Qusqu Region.", "U", CO,
  "subject reads as Ayacucho and Peru", [("Ayacucho & Peru", "ayacucho_peru")])
C("R5-CO10", [["new_dehli_india", "instance_of", "city_work"]],
  "New Dehli & India is an instance of city work.", "U", CO,
  "reads as New Delhi and India each being a city", [("New Dehli & India", "new_dehli_india")])
C("R5-CO11", [["east_london_south_africa", "instance_of", "city_work"]],
  "East London & South Africa is an instance of city work.", "U", CO,
  "reads as East London and South Africa each being a city", [("East London & South Africa", "east_london_south_africa")])
C("R5-CO12", [["new_plymouth_new_zealand", "instance_of", "city_work"]],
  "New Plymouth + New Zealand is an instance of city work.", "U", CO,
  "reads as New Plymouth plus New Zealand", [("New Plymouth + New Zealand", "new_plymouth_new_zealand")])
C("R5-CO13", [["north_carolina_state_college", "member_of", "midwest_inter_library_center"]],
  "North Carolina State College is a member of Midwest & Inter Library Center.", "U", CO,
  "object reads as two bodies, 'Midwest' and 'Inter Library Center'",
  [("Midwest & Inter Library Center", "midwest_inter_library_center")])
C("R5-CO14", [["thuringen_state_germany", "shares_border_with", "bayern_germany"]],
  "Thuringen State Germany shares border with Bayern=Germany.", "U", CO,
  "glued '=' asserts Bayern is Germany, an extra (false) identity claim", [("Bayern=Germany", "bayern_germany")])
C("R5-CO15", [["almere_netherlands", "shares_border_with", "municipality_of_amsterdam"]],
  "Almere≠Netherlands shares border with Municipality of Amsterdam.", "U", CO,
  "glued not-equal sign asserts Almere is not (in) the Netherlands", [("Almere≠Netherlands", "almere_netherlands")])
C("R5-CO16", [["elche_spain", "shares_border_with", "alicante_spain"]],
  "Elche Spain shares border with Alicante⊂Spain.", "U", CO,
  "glued subset sign adds a second, ungathered claim (Alicante is a subset of Spain) inside the object",
  [("Alicante⊂Spain", "alicante_spain")])

# ---------------------------------------------------------------------------
# hyphen_minus_sign_flip: ASCII hyphen-minus (needed for hyphenated names, so likely
# allowed) used as a minus sign on a UTC+ zone. Round 3 used '+', U+2212 and an en
# dash only. The slug drops the sign, so the span folds; the displayed zone is the
# opposite offset.
# ---------------------------------------------------------------------------
HM = "hyphen_minus_sign_flip"
C("R5-HM01", [["tehran_iran", "located_in_time_zone", "utc_3_5"], ["tehran_iran", "instance_of", "city_work"]],
  "Tehran Iran is located in time zone UTC-3.5.", "U", HM,
  "Tehran is UTC+3:30; 'UTC-3.5' is a mid-Atlantic offset seven hours away", [("UTC-3.5", "utc_3_5")])
C("R5-HM02", [["mashhad_iran", "located_in_time_zone", "utc_3_5"]],
  "Mashhad Iran is located in time zone UTC -3,5.", "U", HM,
  "detached minus with a decimal comma: UTC minus 3.5 hours, the opposite of Iran's +3:30", [("UTC -3,5", "utc_3_5")])
C("R5-HM03", [["nangarhar_province", "located_in_time_zone", "utc_4_30"]],
  "Nangarhar Province is located in time zone UTC-4:30.", "U", HM,
  "Afghanistan is UTC+4:30; UTC-4:30 is a Venezuelan-era offset", [("UTC-4:30", "utc_4_30")])
C("R5-HM04", [["badakhshan_province", "located_in_time_zone", "utc_4_30"]],
  "Badakhshan Province located in time zone UTC-4:30.", "U", HM,
  "sign reversed, bare 'located in time zone' form", [("UTC-4:30", "utc_4_30")])
C("R5-HM05", [["yangon_myanmar", "located_in_time_zone", "utc_06_30"]],
  "Yangon Myanmar is located in time zone UTC-06:30.", "U", HM,
  "Myanmar is UTC+06:30; minus reverses it", [("UTC-06:30", "utc_06_30")])
C("R5-HM06", [["kathmandu_nepal", "located_in_time_zone", "gmt_5_45"]],
  "Kathmandu Nepal is located in time zone GMT-5:45.", "U", HM,
  "Nepal is GMT+5:45; minus reverses it", [("GMT-5:45", "gmt_5_45")])
C("R5-HM07", [["shiraz_persia", "located_in_time_zone", "utc_3_5"]],
  "Shiraz Persia is located in time zone UTC+/-3.5.", "U", HM,
  "'+/-' makes the offset either +3.5 or -3.5, a hedge on the sign", [("UTC+/-3.5", "utc_3_5")])
C("R5-HM08", [["zone_of_peace", "located_in_time_zone", "gmt_5_45"]],
  "Zone of Peace is located in time zone GMT -5:45.", "U", HM,
  "detached minus token on Nepal's +5:45 zone", [("GMT -5:45", "gmt_5_45")])
C("R5-HM09", [["kerman_iran", "located_in_time_zone", "utc_3_5"]],
  "Kerman Iran located in time zone UTC-3.5.", "U", HM,
  "sign reversed, bare verb form", [("UTC-3.5", "utc_3_5")])

# ---------------------------------------------------------------------------
# digit_punct_quantity: ASCII punctuation inside a numeric slug (1_500_m = 1,500 m)
# that changes the quantity: range, fraction, decimal point, open range,
# inequality, approximation.
# ---------------------------------------------------------------------------
DQ = "digit_punct_quantity"
M15 = ["1_500_m", "subclass_of", "middle_distance_runner"]
M5 = ["5_000_m", "subclass_of", "long_distance_running"]
M10 = ["10_000_m", "subclass_of", "long_distance_running"]
C("R5-DQ01", [M15], "1-500 m is a subclass of middle distance runner.", "U", DQ,
  "'1-500 m' reads as a range from 1 to 500 metres, not the 1,500 m event", [("1-500 m", "1_500_m")])
C("R5-DQ02", [M15], "1/500 m is a subclass of middle distance runner.", "U", DQ,
  "'1/500 m' is a fraction (2 mm)", [("1/500 m", "1_500_m")])
C("R5-DQ03", [M15], "1.500 m is a subclass of middle distance runner.", "U", DQ,
  "in English '1.500 m' is one and a half metres; arguable because continental usage writes 1.500 for 1,500",
  [("1.500 m", "1_500_m")], arguable=True)
C("R5-DQ04", [M5], "5.000 m is a subclass of long distance running.", "U", DQ,
  "in English '5.000 m' is five metres; arguable for the same continental-separator reason",
  [("5.000 m", "5_000_m")], arguable=True)
C("R5-DQ05", [M10], "10,000+ m is a subclass of long distance running.", "U", DQ,
  "'10,000+ m' is an open range (10 km or more), a different class from the 10,000 m event", [("10,000+ m", "10_000_m")])
C("R5-DQ06", [M15], "<1,500 m is a subclass of middle distance runner.", "U", DQ,
  "'<1,500 m' is 'under 1,500 m', a different set of events", [("<1,500 m", "1_500_m")])
C("R5-DQ07", [M5], ">5,000 m is a subclass of long distance running.", "U", DQ,
  "'>5,000 m' is 'over 5,000 m'", [(">5,000 m", "5_000_m")])
C("R5-DQ08", [M10], "~10,000 m is a subclass of long distance running.", "U", DQ,
  "'~10,000 m' is 'about 10,000 m', a hedged quantity", [("~10,000 m", "10_000_m")])

UX = "unit_exponent_fold"
C("R5-UX01", [M10], "10,000 m² is a subclass of long distance running.", "U", UX,
  "superscript two makes it square metres, an area not a race distance; the sign folds away", [("10,000 m²", "10_000_m")])
C("R5-UX02", [M15], "1,500 m³ is a subclass of middle distance runner.", "U", UX,
  "superscript three makes it cubic metres", [("1,500 m³", "1_500_m")])

CU = "currency_sign_fold"
C("R5-CU01", [M15], "$1,500 m is a subclass of middle distance runner.", "U", CU,
  "'$1,500 m' reads as 1,500 million dollars", [("$1,500 m", "1_500_m")])
C("R5-CU02", [M10], "$10,000 m is a subclass of long distance running.", "U", CU,
  "'$10,000 m' reads as ten billion dollars", [("$10,000 m", "10_000_m")])

# ---------------------------------------------------------------------------
# digit_lookalike_letter: a Latin letter shaped like a digit sits where the slug has a
# separator (ezh ~ 3, tone-five ~ 5, dental click ~ 1). It folds to '_', so the span
# still spells the slug, but the displayed number differs.
# ---------------------------------------------------------------------------
DL = "digit_lookalike_letter"
C("R5-DL01", [M15], "1Ʒ500 m is a subclass of middle distance runner.", "U", DL,
  "ezh (U+01B7) in the thousands slot displays as '13500 m'", [("1Ʒ500 m", "1_500_m")])
C("R5-DL02", [M5], "5Ƽ000 m is a subclass of long distance running.", "U", DL,
  "tone-five letter (U+01BC) displays as '55000 m'", [("5Ƽ000 m", "5_000_m")])
C("R5-DL03", [M10], "ǀ10,000 m is a subclass of long distance running.", "U", DL,
  "a vertical-stroke letter before 10,000 displays as '110,000 m'", [("ǀ10,000 m", "10_000_m")])
C("R5-DL04", [["lc_200_40", "part_of", "baikonur_cosmodrome"]],
  "LC 200/40ǀ is part of Baikonur Cosmodrome.", "U", DL,
  "displays as launch complex 200/401, a different pad", [("LC 200/40ǀ", "lc_200_40")])
C("R5-DL05", [["matthew_17_11", "part_of", "new_testiment"]],
  "Matthew 17:11ǀ is part of New Testiment.", "U", DL,
  "displays as verse 17:111, which does not exist", [("Matthew 17:11ǀ", "matthew_17_11")])

# ---------------------------------------------------------------------------
# genitive_initial_swap: a middle initial S in the slug ("clive_s_lewis") is written
# as a possessive 's. The span folds to the slug; the reader sees a possessive noun
# phrase ("the Lewis that belongs to Clive"), a different referent.
# ---------------------------------------------------------------------------
GI = "genitive_initial_swap"
C("R5-GI01", [["clive_s_lewis", "instance_of", "human_specie"]],
  "Clive's Lewis is an instance of human specie.", "U", GI,
  "'Clive's Lewis' is some Lewis belonging to a Clive, not C. S. Lewis", [("Clive's Lewis", "clive_s_lewis")])
C("R5-GI02", [["george_s_clinton", "instance_of", "human_specie"]],
  "George's Clinton is an instance of human specie.", "U", GI,
  "possessive noun phrase, not George S. Clinton", [("George's Clinton", "george_s_clinton")])
C("R5-GI03", [["charles_s_dutton", "instance_of", "human_specie"]],
  "Charles's Dutton is an instance of human specie.", "U", GI,
  "possessive noun phrase, not Charles S. Dutton", [("Charles's Dutton", "charles_s_dutton")])
C("R5-GI04", [["samuel_s_hinds", "instance_of", "human_specie"]],
  "Samuel's Hinds is an instance of human specie.", "U", GI,
  "possessive noun phrase, not Samuel S. Hinds", [("Samuel's Hinds", "samuel_s_hinds")])
C("R5-GI05", [["rapid_city_s_dakota", "instance_of", "city_work"]],
  "Rapid City's Dakota is an instance of city work.", "U", GI,
  "'Rapid City's Dakota' is something named Dakota in Rapid City, not Rapid City, S. Dakota",
  [("Rapid City's Dakota", "rapid_city_s_dakota")])

IN = "foldable_letter_initial_to_name"
C("R5-IN01", [["a_berg", "instance_of", "surname_page"]],
  "Aĺ Berg is an instance of surname page.", "U", IN,
  "l-acute after the initial A displays as 'Al Berg', a different name; folds to a_berg", [("Aĺ Berg", "a_berg")])
C("R5-IN02", [["s_morgenstern", "instance_of", "human_specie"]],
  "Sý Morgenstern is an instance of human specie.", "U", IN,
  "y-acute after the initial S displays as 'Sy Morgenstern', a different person; folds to s_morgenstern",
  [("Sý Morgenstern", "s_morgenstern")])

LH = "latin_diacritic_homoglyph"
C("R5-LH01", [["stra_aros", "instance_of", "urban_areas_of_sweden"]],
  "Ṽästra Aros is an instance of urban areas of Sweden.", "U", LH,
  "V-with-tilde (Latin, decomposes to V) displays 'Vastra Aros', a different town from the gathered Ostra Aros; both letters fold away",
  [("Ṽästra Aros", "stra_aros")])
C("R5-LH02", [["stra_aros", "located_in_time_zone", "rome_time"]],
  "Ʋästra Aros is located in time zone Rome time.", "U", LH,
  "V-with-hook (Latin script, no decomposition) displays 'Vastra Aros'", [("Ʋästra Aros", "stra_aros")])

# ---------------------------------------------------------------------------
# connective_word_in_slug: slugs that contain 'or', 'if', 'versus', 'nothing but';
# punctuation or case makes the word read as a sentence-level connective.
# ---------------------------------------------------------------------------
CW = "connective_word_in_slug"
C("R5-CW01", [["greater_eugene", "shares_border_with", "albany_lebanon_or_sa"]],
  "Greater Eugene shares border with Albany-Lebanon, or SA.", "U", CW,
  "the slug's OR is the state Oregon; ', or SA' reads as a disjunction (Albany-Lebanon or SA)",
  [("Albany-Lebanon, or SA", "albany_lebanon_or_sa")])
C("R5-CW02", [["if_chins_could_kill", "instance_of", "human_specie"]],
  "If chins could kill, is an instance of human specie.", "U", CW,
  "the trailing comma turns the name into a conditional clause; arguable because the result is garbled",
  [("If chins could kill,", "if_chins_could_kill")], arguable=True)
C("R5-CW03", [["demographic_group", "subclass_of", "sociology_versus_social_theory"]],
  "Demographic group is a subclass of sociology, versus social theory.", "U", CW,
  "reads 'a subclass of sociology, as opposed to social theory': a different object",
  [("sociology, versus social theory", "sociology_versus_social_theory")])
C("R5-CW04", [["nothing_but_wind", "instance_of", "human_specie"]],
  "nothing but wind is an instance of human specie.", "U", CW,
  "in lower case the name reads as a quantifier: nothing except wind is human",
  [("nothing but wind", "nothing_but_wind")])

AB = "abbreviation_reread"
C("R5-AB01", [["iso_3166_2_no_01", "part_of", "iso_region_codes"]],
  "ISO 3166-2 No. 01 is part of ISO region codes.", "U", AB,
  "the slug's NO is Norway; 'No. 01' reads as 'number 01', a different code",
  [("ISO 3166-2 No. 01", "iso_3166_2_no_01")])

GL = "glued_logic_symbol"
C("R5-GL01", [["alicante_spain", "shares_border_with", "elche_spain"]],
  "Alicante Spain shares border with ~Elche Spain.", "U", GL,
  "a tilde glued inside the object span reads as 'about' or logical NOT", [("~Elche Spain", "elche_spain")])
C("R5-GL02", [["hesse_germany", "shares_border_with", "bayern_germany"]],
  "Hesse Germany shares border with ¬Bayern Germany.", "U", GL,
  "a negation sign glued inside the object span (round 3 put it outside, around a whole sentence)",
  [("¬Bayern Germany", "bayern_germany")])

# ---------------------------------------------------------------------------
# split_context_new: a neighbouring line or sentence denies the claim, joined by a
# separator round 4 did not use (U+2028, NEL, Devanagari danda, ideographic full
# stop + CJK denial). Scored by round 3's split rule.
# ---------------------------------------------------------------------------
SC = "split_context_new"
C("R5-SC01", [["costa_rica", "member_of", "united_nations"]],
  "The next line is false. Costa Rica is a member of the United Nations.", "U", SC,
  "a U+2028 LINE SEPARATOR; the line before declares the claim false")
C("R5-SC02", [["el_salvador", "member_of", "united_nations"]],
  "El Salvador is a member of the United Nations. The previous sentence is a lie.", "U", SC,
  "the following sentence retracts the claim by explicit back-reference")
C("R5-SC03", [["zone_of_peace", "shares_border_with", "bharat_ganrajya"]],
  "Zone of Peace shares border with Bharat Ganrajya। Untrue।", "U", SC,
  "Devanagari danda as the sentence boundary, then a denial")
C("R5-SC04", [["costa_rica", "shares_border_with", "sport_in_panama"]],
  "Myth\u0085Costa Rica shares border with Sport in Panama.", "U", SC,
  "a NEL (U+0085) line break after the label 'Myth'")
C("R5-SC05", [["el_salvador", "shares_border_with", "administrative_divisions_of_guatemala"]],
  "El Salvador shares border with Administrative Divisions of Guatemala。不对。", "U", SC,
  "ideographic full stop, then a Chinese denial ('not right') whose slug form is empty")

VR = "verb_reading_new"
C("R5-VR01", [["bayer_leverkusen_ii", "located_in_the_administrative_territoria", "history_of_leverkusen"]],
  "Bayer Leverkusen II locates History of Leverkusen.", "U", VR,
  "an -s reading of 'located': 'locates' means finds or places, so the team acts on the city")
C("R5-VR02", [["eintracht_braunschweig_ii", "located_in_the_administrative_territoria", "brunswick_germany"]],
  "Eintracht Braunschweig II locates Brunswick Germany.", "U", VR,
  "'locates' (finds) instead of 'is located in'")

# ---------------------------------------------------------------------------
# F twins: plain statements of exactly the gathered triple, many in the spellings the
# U cases attack (signed zones written correctly, appositive commas, initials with
# periods, numerals without extra letters, typographic variants that display the same).
# ---------------------------------------------------------------------------
F = "F_twin"
C("R5-F01", [["city_of_chicago", "located_in_time_zone", "utc_06"]],
  "City of Chicago is located in time zone UTC-06.", "F", F,
  "Chicago is UTC-6; hyphen-minus is the correct sign", [("UTC-06", "utc_06")])
C("R5-F02", [["metro_tucson", "located_in_time_zone", "utc_07"]],
  "Metro Tucson is located in time zone UTC-07.", "F", F, "Tucson is UTC-7", [("UTC-07", "utc_07")])
C("R5-F03", [["costa_rica", "located_in_time_zone", "utc_06"]],
  "Costa Rica is located in time zone UTC-06.", "F", F, "Costa Rica is UTC-6", [("UTC-06", "utc_06")])
C("R5-F04", [["denver_colorado_united_states", "located_in_time_zone", "utc_07"]],
  "Denver Colorado United States is located in time zone UTC-07.", "F", F, "Denver is UTC-7", [("UTC-07", "utc_07")])
C("R5-F05", [["tehran_iran", "located_in_time_zone", "utc_3_5"]],
  "Tehran Iran is located in time zone UTC+3.5.", "F", F, "correct sign; twin of HM01", [("UTC+3.5", "utc_3_5")])
C("R5-F06", [["kathmandu_nepal", "located_in_time_zone", "gmt_5_45"]],
  "Kathmandu Nepal is located in time zone GMT+5:45.", "F", F, "correct sign; twin of HM06", [("GMT+5:45", "gmt_5_45")])
C("R5-F07", [["hesse_germany", "shares_border_with", "bayern_germany"]],
  "Hesse Germany shares border with Bayern, Germany.", "F", F,
  "appositive comma (Bayern, in Germany); twin of CO01", [("Bayern, Germany", "bayern_germany")])
C("R5-F08", [["alicante_spain", "shares_border_with", "elche_spain"]],
  "Alicante Spain shares border with Elche, Spain.", "F", F, "appositive comma; twin of CO03", [("Elche, Spain", "elche_spain")])
C("R5-F09", [["barking_london", "shares_border_with", "ilford_united_kingdom"]],
  "Barking London shares border with Ilford, United Kingdom.", "F", F, "appositive comma; twin of CO05",
  [("Ilford, United Kingdom", "ilford_united_kingdom")])
C("R5-F10", [["asker_norway", "shares_border_with", "b_rum_norway"]],
  "Asker Norway shares border with Bærum, Norway.", "F", F, "accented name plus appositive comma; twin of CO04",
  [("Bærum, Norway", "b_rum_norway")])
C("R5-F11", [["new_dehli_india", "instance_of", "city_work"]],
  "New Dehli, India is an instance of city work.", "F", F, "appositive comma; twin of CO10", [("New Dehli, India", "new_dehli_india")])
C("R5-F12", [["women_s_international_league", "instance_of", "non_governmental_organization"]],
  "Women’s International League is an instance of non-governmental organization.", "F", F,
  "curly apostrophe; twin of PA01", [("Women’s International League", "women_s_international_league")])
C("R5-F13", [["hepatitis_non_b", "instance_of", "organic_cause"]],
  "Hepatitis non-B is an instance of organic cause.", "F", F, "twin of PA14", [("Hepatitis non-B", "hepatitis_non_b")])
C("R5-F14", [["pre_dreadnought_battleship", "subclass_of", "battle_ship"]],
  "Pre-dreadnought battleship is a subclass of battle ship.", "F", F, "twin of PA10")
C("R5-F15", [["semi_automatic_rifles", "subclass_of", "revolving_rifle"]],
  "Semi-automatic rifles is a subclass of revolving rifle.", "F", F, "twin of PA08")
C("R5-F16", [["united_states_whig_party", "follows", "anti_jacksonian_party"]],
  "United States Whig Party follows Anti-Jacksonian Party.", "F", F, "twin of PA06")
C("R5-F17", [["south_mongolia", "shares_border_with", "liaoning_province"]],
  "South Mongolia shares border with Liaoning Province.", "F", F, "twin of PM01")
C("R5-F18", [["new_zealnd", "shares_border_with", "commonwealth_australia"]],
  "New Zealnd shares border with Commonwealth Australia.", "F", F, "twin of PM03 (store misspelling kept)")
C("R5-F19", [["northern_ireland_fc", "participant_of", "world_cup_1982"]],
  "Northern Ireland FC is a participant of World Cup 1982.", "F", F, "twin of PM05")
C("R5-F20", [["clive_s_lewis", "instance_of", "human_specie"]],
  "Clive S. Lewis is an instance of human specie.", "F", F, "initial with period; twin of GI01", [("Clive S. Lewis", "clive_s_lewis")])
C("R5-F21", [["george_s_clinton", "instance_of", "human_specie"]],
  "George S. Clinton is an instance of human specie.", "F", F, "twin of GI02", [("George S. Clinton", "george_s_clinton")])
C("R5-F22", [["rapid_city_s_dakota", "instance_of", "city_work"]],
  "Rapid City, S. Dakota is an instance of city work.", "F", F, "twin of GI05", [("Rapid City, S. Dakota", "rapid_city_s_dakota")])
C("R5-F23", [M15], "1,500 m is a subclass of middle distance runner.", "F", F, "thousands comma; twin of the DQ/UX/CU/DL cases",
  [("1,500 m", "1_500_m")])
C("R5-F24", [M10], "10 000 m is a subclass of long distance running.", "F", F,
  "SI thousands separator (narrow no-break space)", [("10 000 m", "10_000_m")])
C("R5-F25", [M5], "5 000 m is a subclass of long distance running.", "F", F,
  "thin-space thousands separator", [("5 000 m", "5_000_m")])
C("R5-F26", [["charles_ii_of_scotland", "member_of", BRS]],
  "Charles II of Scotland is a member of British Royal Society.", "F", F, "twin of NX01")
C("R5-F27", [["george_v", "member_of", BRS]],
  "George V is a member of British Royal Society.", "F", F, "twin of NX02")
C("R5-F28", [["games_of_the_x_olympiad", "follows", "erven_jos_rovers"]],
  "Games of the X Olympiad follows Erven Jos Rovers.", "F", F, "twin of NX07")
C("R5-F29", [["the_godfather_trilogy", "has_part", "the_godfather_part_ii"]],
  "The Godfather Trilogy has part The Godfather Part II.", "F", F, "twin of NX11")
C("R5-F30", [["apple_ii", "instance_of", "personal_computer"]],
  "Apple II is an instance of personal computer.", "F", F, "twin of NX17")
C("R5-F31", [["indochina_war", "part_of", "cold_war_i"]],
  "Indochina War is part of Cold War I.", "F", F, "twin of NX15")
C("R5-F32", [["sega_mega_drive_ii", "follows", "sega_master_system_iii"]],
  "Sega Mega Drive II follows Sega Master System III.", "F", F, "twin of NX10")
C("R5-F33", [["greater_eugene", "shares_border_with", "albany_lebanon_or_sa"]],
  "Greater Eugene shares border with Albany Lebanon OR SA.", "F", F,
  "OR is the Oregon abbreviation, in capitals; twin of CW01", [("Albany Lebanon OR SA", "albany_lebanon_or_sa")])
C("R5-F34", [["iso_3166_2_no_01", "part_of", "iso_region_codes"]],
  "ISO 3166-2:NO-01 is part of ISO region codes.", "F", F, "the code as written; twin of AB01",
  [("ISO 3166-2:NO-01", "iso_3166_2_no_01")])
C("R5-F35", [["stra_aros", "located_in_time_zone", "rome_time"]],
  "Östra Aros is located in time zone Rome time.", "F", F, "correct accented spelling; twin of LH02",
  [("Östra Aros", "stra_aros")])
C("R5-F36", [["intel_corp", "part_of", "s_p"]],
  "Intel Corp is part of S&P.", "F", F, "the ampersand is part of the real name S&P", [("S&P", "s_p")])
C("R5-F37", [["oslo_t_banedrift", "instance_of", "a_s"]],
  "Oslo T-banedrift is an instance of A/S.", "F", F, "the slash is part of the real form A/S",
  [("Oslo T-banedrift", "oslo_t_banedrift"), ("A/S", "a_s")])
C("R5-F38", [["z_80", "subclass_of", "microprocessor_unit"]],
  "Z-80 is a subclass of microprocessor unit.", "F", F, "hyphen in the real name", [("Z-80", "z_80")])
C("R5-F39", [["german_research_network", "instance_of", "e_v"]],
  "German Research Network is an instance of e.V.", "F", F, "the real abbreviation e.V.", [("e.V", "e_v")])
C("R5-F40", [["freistaat_sachsen", "shares_border_with", "bayern_germany"]],
  "Freistaat Sachsen shares border with Bayern Germany.", "F", F,
  "no-break space inside the subject; displays identically", [("Freistaat Sachsen", "freistaat_sachsen")])
C("R5-F41", [["parody_film", "subclass_of", "fantasy_comedy_film"]],
  "Parody ﬁlm is a subclass of fantasy comedy ﬁlm.", "F", F,
  "fi ligature (U+FB01) displays as 'film'; a retention probe for compatibility characters")
C("R5-F42", [["tyrant_flycatcher", "instance_of", "taxonomic_group"]],
  "Tyrant ﬂycatcher is an instance of taxonomic group.", "F", F, "fl ligature (U+FB02) displays as 'flycatcher'")
C("R5-F43", [["orange_county_florida", "shares_border_with", "volusia_county_florida"]],
  "Orange County Florida shares border with Volu‌sia County Florida.", "F", F,
  "zero-width non-joiner inside a word; displays identically")
C("R5-F44", [["a_berg", "instance_of", "surname_page"]],
  "A. Berg is an instance of surname page.", "F", F, "initial with period; twin of IN01", [("A. Berg", "a_berg")])
C("R5-F45", [["demographic_group", "subclass_of", "sociology_versus_social_theory"]],
  "Demographic group is a subclass of sociology versus social theory.", "F", F, "twin of CW03")
C("R5-F46", [["crime_writers", "subclass_of", "non_fiction"]],
  "Crime writers is a subclass of non-fiction.", "F", F, "twin of PA05")
C("R5-F47", [["0_6_0dh", "subclass_of", "railway_locomotive"]],
  "0-6-0DH is a subclass of railway locomotive.", "F", F, "hyphenated wheel arrangement as written", [("0-6-0DH", "0_6_0dh")])


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def main():
    errs = []
    store = json.load(open(STORE, encoding="utf-8"))
    triples = {(f["fact"]["agent"], f["fact"]["action"], f["fact"]["patient"]) for f in store}
    relations = {t[1] for t in triples}
    if len(store) != 78857 or len(relations) != 60:
        errs.append(f"store shape changed: {len(store)} facts, {len(relations)} relations")
    prior = set()
    for p in (R3_CASES, R4_CASES):
        for c in json.load(open(p, encoding="utf-8")):
            prior.add(c["sentence"])

    req = {"id", "gathered", "sentence", "expect", "class", "why"}
    ids, sents = set(), set()
    for c in CASES:
        keys = set(c)
        if not (keys == req or keys == req | {"arguable"}):
            errs.append(f"{c.get('id')}: bad key set {sorted(keys)}")
        if "arguable" in c and c["arguable"] is not True:
            errs.append(f"{c['id']}: arguable must be true when present")
        if not c["id"].startswith("R5-") or c["id"] in ids:
            errs.append(f"{c['id']}: bad or repeated id")
        ids.add(c["id"])
        if c["sentence"] in sents:
            errs.append(f"{c['id']}: repeated sentence")
        sents.add(c["sentence"])
        if c["sentence"] in prior:
            errs.append(f"{c['id']}: sentence repeats a round-3/round-4 sentence")
        if c["expect"] not in ("U", "F"):
            errs.append(f"{c['id']}: expect must be U or F")
        if not c["why"].strip() or not c["class"].strip():
            errs.append(f"{c['id']}: empty why/class")
        g = c["gathered"]
        if not (1 <= len(g) <= 4):
            errs.append(f"{c['id']}: {len(g)} gathered triples")
        for t in g:
            if len(t) != 3 or tuple(t) not in triples:
                errs.append(f"{c['id']}: not a store triple {t}")
            elif t[1] not in relations:
                errs.append(f"{c['id']}: unknown relation {t[1]}")
        slugs = {t[0] for t in g} | {t[2] for t in g}
        for span, slug in FOLDS.get(c["id"], []):
            if span not in c["sentence"]:
                errs.append(f"{c['id']}: fold span {span!r} not in sentence")
            if sanitize(span) != slug:
                errs.append(f"{c['id']}: sanitize({span!r}) = {sanitize(span)!r}, expected {slug!r}")
            if slug not in slugs:
                errs.append(f"{c['id']}: fold target {slug} is not a gathered slug")
    for cid in FOLDS:
        if cid not in ids:
            errs.append(f"fold check for unknown id {cid}")

    by_label = collections.Counter(c["expect"] for c in CASES)
    if by_label["U"] < 110 or by_label["F"] < 35:
        errs.append(f"too few cases: {dict(by_label)}")

    if errs:
        print("VALIDATION FAILED:")
        for e in errs:
            print("  " + e)
        sys.exit(1)

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(CASES, fh, ensure_ascii=False, indent=1)
        fh.write("\n")

    by_class = collections.OrderedDict()
    for c in CASES:
        by_class.setdefault(c["class"], collections.Counter())[c["expect"]] += 1
    print(f"wrote {OUT}: {len(CASES)} cases")
    print(f"labels: U={by_label['U']} F={by_label['F']}")
    print(f"{'class':40s} {'U':>4s} {'F':>4s}")
    for k, v in by_class.items():
        print(f"{k:40s} {v['U']:4d} {v['F']:4d}")
    arg = [c["id"] for c in CASES if c.get("arguable")]
    print(f"arguable ({len(arg)}): {', '.join(arg)}")
    print(f"fold self-checks passed: {sum(len(v) for v in FOLDS.values())} spans over {len(FOLDS)} cases")
    print(f"store triples: all {sum(len(c['gathered']) for c in CASES)} exact; relations used: "
          f"{len({t[1] for c in CASES for t in c['gathered']})} of {len(relations)}")
    print(f"no sentence repeats any of the {len(prior)} round-3/round-4 sentences")


if __name__ == "__main__":
    main()
