#!/usr/bin/env python3
"""Round-4 blind hunt for the strict allowlist claim checker: case generator + validator.

Run:  python -m research.findings.raw._moat_allowlist_hunt4.r4.build_cases   (from the repo root; the module run stamps provenance)

Writes cases.json next to this file and validates:
  * every gathered triple is an exact triple of the store facts.json;
  * every relation is one of the store's relations (60);
  * every case has the exact key set (plus the optional "arguable");
  * no id is repeated, no sentence is repeated;
  * no sentence is identical to a round-3 sentence;
  * 1-4 gathered triples per case; expect in {U, F};
  * every FOLD self-check: the quoted span, run through a local copy of the store's slug rule
    (sanitize: lowercase, runs of non-[a-z0-9] -> "_", strip "_", cut at 40, rstrip "_"),
    yields the intended gathered slug.  This proves the attack rests on the slug fold the
    prereg names, not on a typo of mine.  (Self-check only; not stored in cases.json.)
Prints counts by label and by class.

Written blind: no matcher code, matcher tests, relation table, verb-form table or results file was read.
"""
import collections
import json
import os
import re
import sys

import research.runners  # noqa: F401  (provenance door: the module run writes the cases.json sidecar)

STORE = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json"
ROUND3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_moat_allowlist_hunt3", "cases.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.json")

REQUIRED_KEYS = {"id", "gathered", "sentence", "expect", "class", "why"}
OPTIONAL_KEYS = {"arguable"}

_TOK_RE = re.compile(r"[^a-z0-9]+")


def sanitize(alias, maxlen=40):
    """Local copy of the store's slug rule (research/runners/_knowledge_core_curate.py::sanitize)."""
    t = _TOK_RE.sub("_", alias.strip().lower()).strip("_")
    if len(t) > maxlen:
        t = t[:maxlen].rstrip("_")
    return t


CASES = []
FOLDS = {}  # id -> list of (span, slug) self-checks


def add(cid, gathered, sentence, expect, cls, why, arguable=False, folds=()):
    c = {"id": "R4-" + cid, "gathered": [list(t) for t in gathered], "sentence": sentence,
         "expect": expect, "class": cls, "why": why}
    if arguable:
        c["arguable"] = True
    CASES.append(c)
    if folds:
        FOLDS[c["id"]] = list(folds)


# ---------------------------------------------------------------- frequently used gathered triples
LKM = ("landkreis_munchen", "shares_border_with", "munich_west_germany")
COTE = ("cote_d_or", "shares_border_with", "saone_et_loire")
NAGOYA = ("nagoya_university", "instance_of", "higher_general_education")
NAGOYA_CTX = [NAGOYA, ("nagoya_university", "located_in_the_administrative_territoria", "nagoya_japan")]

# ================================================================= 1. cap_tail_fold (U)
# The slug rule cuts at 40 chars. For a slug of 39 or 40 chars, ANY extra words after it (with a separator)
# sanitize back to the same slug. A matcher that spells SUBJ/OBJ by sanitize over a token run would read the
# extended span as the gathered entity, while the reader sees a different entity or an extra claim.
CT = "cap_tail_fold"
add("CT01", [("joseph_alois_ratzinger", "member_of", "academia_scientiarum_et_artium_europaea")],
    "Joseph Alois Ratzinger is a member of Academia Scientiarum et Artium Europaea no longer.", "U", CT,
    "'no longer' denies current membership; the 39-char slug absorbs the tail under the 40-char cut",
    folds=[("Academia Scientiarum et Artium Europaea no longer", "academia_scientiarum_et_artium_europaea")])
add("CT02", [("joseph_alois_ratzinger", "member_of", "academia_scientiarum_et_artium_europaea"),
             ("academia_scientiarum_et_artium_europaea", "headquarters_location", "historic_centre_of_salzburg")],
    "Joseph Alois Ratzinger is a member of Academia Scientiarum et Artium Europaea's expulsion list.", "U", CT,
    "asserts he is on the academy's expulsion list, not that he is a member of the academy",
    folds=[("Academia Scientiarum et Artium Europaea's expulsion list", "academia_scientiarum_et_artium_europaea")])
add("CT03", [("2004_summer_paralympic_games", "follows", "sydney_paralyampic_organising_committee")],
    "2004 Summer Paralympic Games follows Sydney Paralyampic Organising Committee's collapse.", "U", CT,
    "asserts an extra event (the committee's collapse) and makes it the thing followed",
    folds=[("Sydney Paralyampic Organising Committee's collapse", "sydney_paralyampic_organising_committee")])
add("CT04", [("independent_spd", "follows", "sozialdemokratische_partei_deutschlands")],
    "Independent SPD follows Sozialdemokratische Partei Deutschlands der DDR.", "U", CT,
    "the SPD of the GDR is a different party from the gathered SPD",
    folds=[("Sozialdemokratische Partei Deutschlands der DDR", "sozialdemokratische_partei_deutschlands")])
add("CT05", [("rally_for_the_republic", "follows", "union_des_democrates_pour_la_republique")],
    "Rally for the Republic follows Union des Democrates pour la Republique, but only in name.", "U", CT,
    "hedged ('only in name'); the comma and tail fold away inside a sanitized 39-char span",
    folds=[("Union des Democrates pour la Republique, but only in name", "union_des_democrates_pour_la_republique")])
add("CT06", [("leningrad_state_choreographic_institute", "part_of", "ballet_portal")],
    "Leningrad State Choreographic Institute's rival school is part of Ballet Portal.", "U", CT,
    "subject is the institute's rival school, not the institute (SUBJ-side tail)",
    folds=[("Leningrad State Choreographic Institute's rival school", "leningrad_state_choreographic_institute")])
add("CT07", [("university_of_north_carolina_chapel_hill", "member_of", "midwest_inter_library_center"),
             ("university_of_north_carolina_chapel_hill", "part_of", "u_north_carolina")],
    "University of North Carolina Chapel Hill's rival is a member of Midwest Inter Library Center.", "U", CT,
    "subject is the university's rival (40-char slug, possessive tail cut off)",
    folds=[("University of North Carolina Chapel Hill's rival", "university_of_north_carolina_chapel_hill")])
add("CT08", [("university_of_north_carolina_chapel_hill", "part_of", "u_north_carolina")],
    "University of North Carolina Chapel Hill Medical School is part of U North Carolina.", "U", CT,
    "subject is a different entity (its medical school); not gathered",
    folds=[("University of North Carolina Chapel Hill Medical School", "university_of_north_carolina_chapel_hill")])
add("CT09", [("alberta_provincial_electoral_districts", "subclass_of", "canadian_provincial_electoral_districts")],
    "Alberta provincial electoral districts is a subclass of Canadian provincial electoral districts abolished in 1905.",
    "U", CT, "extra claim: the superclass districts were abolished in 1905",
    folds=[("Canadian provincial electoral districts abolished in 1905", "canadian_provincial_electoral_districts")])
add("CT10", [("glamoc", "instance_of", "municipalities_of_bosnia_and_herzegovina")],
    "Glamoc is an instance of municipalities of Bosnia and Herzegovina that no longer exist.", "U", CT,
    "extra claim that the municipality class no longer exists",
    folds=[("municipalities of Bosnia and Herzegovina that no longer exist", "municipalities_of_bosnia_and_herzegovina")])
add("CT11", [("multilateral_investment_guarantee_agency", "instance_of", "global_organization")],
    "Multilateral Investment Guarantee Agency's defunct predecessor is an instance of global organization.", "U", CT,
    "subject is a (claimed) defunct predecessor, not the agency",
    folds=[("Multilateral Investment Guarantee Agency's defunct predecessor", "multilateral_investment_guarantee_agency")])
add("CT12", [("senegal_national_football_team", "participant_of", "2002_fifa_world_cup_hosting_controversy")],
    "Senegal National Football Team is a participant of 2002 FIFA World Cup hosting controversy only as a spectator.",
    "U", CT, "'only as a spectator' contradicts participation",
    folds=[("2002 FIFA World Cup hosting controversy only as a spectator", "2002_fifa_world_cup_hosting_controversy")])
add("CT13", [("1989_fis_nordic_world_ski_championships", "followed_by", "1991_fis_nordic_world_ski_championships")],
    "1989 FIS Nordic World Ski Championships junior edition is followed by 1991 FIS Nordic World Ski Championships.",
    "U", CT, "subject is a junior edition, a different event",
    folds=[("1989 FIS Nordic World Ski Championships junior edition", "1989_fis_nordic_world_ski_championships")])
add("CT14", [("sydney_paralyampic_organising_committee", "follows", "atlanta_paralympics")],
    "Sydney Paralyampic Organising Committee chairman follows Atlanta Paralympics.", "U", CT,
    "subject is the committee chairman (a person), and 'follows' then reads as tracking/pursuing",
    folds=[("Sydney Paralyampic Organising Committee chairman", "sydney_paralyampic_organising_committee")])
add("CT15", [("kongelige_danske_videnskabernes_selskab", "instance_of", "academies_of_science")],
    "Kongelige Danske Videnskabernes Selskab's ex-president is an instance of academies of science.", "U", CT,
    "subject is a former president (a person), not the academy",
    folds=[("Kongelige Danske Videnskabernes Selskab's ex-president", "kongelige_danske_videnskabernes_selskab")])
add("CT16", [("western_australian_electoral_commission", "subclass_of", "election_district")],
    "Western Australian Electoral Commission, which was abolished, is a subclass of election district.", "U", CT,
    "relative clause adds an unsupported claim (abolished); commas fold inside the span",
    folds=[("Western Australian Electoral Commission, which was abolished,", "western_australian_electoral_commission")])
add("CT17", [("encyclopedia_britannica_eleventh_edition", "instance_of", "published_edition")],
    "Encyclopedia Britannica Eleventh Edition supplement is an instance of published edition.", "U", CT,
    "subject is a supplement, a different work",
    folds=[("Encyclopedia Britannica Eleventh Edition supplement", "encyclopedia_britannica_eleventh_edition")])
add("CT18", [("vimu_vend_su", "award_received", "order_of_merit_of_north_rhine_westphalia")],
    "Vimu Vend Su received Order of Merit of North Rhine Westphalia nomination.", "U", CT,
    "received a nomination, not the award",
    folds=[("Order of Merit of North Rhine Westphalia nomination", "order_of_merit_of_north_rhine_westphalia")])
add("CT19", [("state_observatory_heidelberg_konigstuhl", "instance_of", "weather_observation_station")],
    "State Observatory Heidelberg Konigstuhl annex is an instance of weather observation station.", "U", CT,
    "subject is an annex, a different entity",
    folds=[("State Observatory Heidelberg Konigstuhl annex", "state_observatory_heidelberg_konigstuhl")])

# ================================================================= 2. accidental_sign_fold (U)
# Musical keys: the store spells sharps/flats as words (b_sharp_major). A sign glued to the key letter
# (ASCII '#', or U+266F sharp / U+266D flat) is a non-[a-z0-9] char and folds away: "D# major" -> d_major.
AC = "accidental_sign_fold"
DMAJ = ("d_major", "follows", "g_major")
add("AC01", [DMAJ], "D♭ major follows G major.", "U", AC,
    "D-flat major is a different key from D major", folds=[("D♭ major", "d_major")])
add("AC02", [DMAJ, ("d_major", "followed_by", "d_sharp_major")], "D# major follows G major.", "U", AC,
    "D-sharp major (ASCII #) is a different key; d_sharp_major is gathered only as D major's successor",
    folds=[("D# major", "d_major")])
add("AC03", [DMAJ], "D major follows G♯ major.", "U", AC,
    "G-sharp major is not G major (OBJ side)", folds=[("G♯ major", "g_major")])
add("AC04", [("g_minor", "follows", "c_minor")], "G minor follows C# minor.", "U", AC,
    "C-sharp minor is not C minor", folds=[("C# minor", "c_minor")])
add("AC05", [("f_minor", "follows", "e_minor")], "F# minor follows E minor.", "U", AC,
    "F-sharp minor is not F minor (the natural LLM spelling of F-sharp)", folds=[("F# minor", "f_minor")])
add("AC06", [("e_minor", "followed_by", "f_minor")], "E minor is followed by F♯ minor.", "U", AC,
    "F-sharp minor is not F minor", folds=[("F♯ minor", "f_minor")])
add("AC07", [("c_minor", "followed_by", "e_minor")], "C minor is followed by E♭ minor.", "U", AC,
    "E-flat minor is not E minor", folds=[("E♭ minor", "e_minor")])
add("AC08", [("g_major", "followed_by", "d_major")], "G♭ major is followed by D major.", "U", AC,
    "G-flat major is not G major (SUBJ side)", folds=[("G♭ major", "g_major")])

# ================================================================= 3. homoglyph_entity_swap (U)
# A look-alike non-ASCII letter (Cyrillic / Roman-numeral / full-width) vanishes under the slug rule, so the
# checker sees one gathered slug while the reader sees a DIFFERENT entity (another games edition, another town).
HG = "homoglyph_entity_swap"
add("HG01", [("x_commonwealth_games", "follows", "ix_commonwealth_games")],
    "XІ Commonwealth Games follows IX Commonwealth Games.", "U", HG,
    "reader sees 'XI' (Cyrillic I, U+0406): XI follows X, not IX; slug fold gives x_commonwealth_games",
    folds=[("XІ Commonwealth Games", "x_commonwealth_games")])
add("HG02", [("ix_commonwealth_games", "followed_by", "x_commonwealth_games")],
    "IX Commonwealth Games is followed by XⅠ Commonwealth Games.", "U", HG,
    "reader sees 'XI' (Roman numeral one U+2160); IX is followed by X, not XI",
    folds=[("XⅠ Commonwealth Games", "x_commonwealth_games")])
add("HG03", [("vi_olympic_winter_games", "follows", "vth_olympic_winter_games")],
    "ХVI Olympic Winter Games follows Vth Olympic Winter Games.", "U", HG,
    "reader sees 'XVI' (Cyrillic Kha U+0425); XVI does not follow the Vth games",
    folds=[("ХVI Olympic Winter Games", "vi_olympic_winter_games")])
add("HG04", [("vii_pan_american_games", "followed_by", "viii_pan_american_games")],
    "ⅩVII Pan American Games is followed by VIII Pan American Games.", "U", HG,
    "reader sees 'XVII' (Roman numeral ten U+2169); XVII is not followed by VIII",
    folds=[("ⅩVII Pan American Games", "vii_pan_american_games")])
add("HG05", [("viii_commonwealth_games", "followed_by", "ix_commonwealth_games")],
    "ХVIII Commonwealth Games is followed by IX Commonwealth Games.", "U", HG,
    "reader sees 'XVIII'; XVIII is not followed by IX",
    folds=[("ХVIII Commonwealth Games", "viii_commonwealth_games")])
add("HG06", [("vi_olympic_winter_games", "follows", "vth_olympic_winter_games")],
    "VI Olympic Winter Games follows ХVth Olympic Winter Games.", "U", HG,
    "reader sees 'XVth' on the OBJ side; VI does not follow XV",
    folds=[("ХVth Olympic Winter Games", "vth_olympic_winter_games")])
add("HG07", [("stra_aros", "instance_of", "urban_areas_of_sweden")],
    "Ѵästra Aros is an instance of urban areas of Sweden.", "U", HG,
    "reader sees 'Västra Aros' (Izhitsa U+0474 looks like V), a different town from Östra Aros (stra_aros); "
    "Västra Aros is not in the gathered set",
    folds=[("Ѵästra Aros", "stra_aros")])
add("HG08", [("vi_olympic_winter_games", "follows", "vth_olympic_winter_games")],
    "ＸVI Olympic Winter Games follows Vth Olympic Winter Games.", "U", HG,
    "reader sees 'XVI' (full-width X U+FF38)",
    folds=[("ＸVI Olympic Winter Games", "vi_olympic_winter_games")])

# ================================================================= 4. nonascii_digit_erasure (U)
# Full-width, Arabic-Indic and Devanagari digits are not [0-9]; the slug rule erases them, so a year that
# names one edition folds onto the series slug.
DG = "nonascii_digit_erasure"
ESC = ("eurovision_song_contest", "subclass_of", "scripted_television")
GIRO = ("giro_d_italia", "subclass_of", "stage_race")
add("DG01", [ESC], "１９８１ Eurovision Song Contest is a subclass of scripted television.", "U", DG,
    "the 1981 contest (one event) is not what is gathered; an event is not a subclass",
    folds=[("１９８１ Eurovision Song Contest", "eurovision_song_contest")])
add("DG02", [ESC], "Eurovision Song Contest ٢٠٠١ is a subclass of scripted television.", "U", DG,
    "the 2001 contest in Arabic-Indic digits; a single edition, not the series",
    folds=[("Eurovision Song Contest ٢٠٠١", "eurovision_song_contest")])
add("DG03", [GIRO], "Giro d'Italia ２０１４ is a subclass of stage race.", "U", DG,
    "the 2014 edition (full-width digits) is an instance, not the gathered series",
    folds=[("Giro d'Italia ２０１４", "giro_d_italia")])
add("DG04", [GIRO], "Giro d'Italia २०१७ is a subclass of stage race.", "U", DG,
    "the 2017 edition in Devanagari digits",
    folds=[("Giro d'Italia २०१७", "giro_d_italia")])

# ================================================================= 5. shape2_fold_duplicate (U)
# The store has one object per (subject, relation), so shape 2 can only be satisfied when every conjunct
# folds to the SAME gathered slug. Spell the conjuncts so they fold together but name different things.
S2 = "shape2_fold_duplicate"
add("S2F01", [DMAJ], "D major follows G major and G♯ major.", "U", S2,
    "second conjunct is G-sharp major; both conjuncts fold to g_major",
    folds=[("G major", "g_major"), ("G♯ major", "g_major")])
add("S2F02", [("independent_spd", "follows", "sozialdemokratische_partei_deutschlands")],
    "Independent SPD follows Sozialdemokratische Partei Deutschlands and Sozialdemokratische Partei Deutschlands der DDR.",
    "U", S2, "second conjunct is the GDR party; both fold to the 39-char SPD slug",
    folds=[("Sozialdemokratische Partei Deutschlands der DDR", "sozialdemokratische_partei_deutschlands")])
add("S2F03", [("vi_olympic_winter_games", "follows", "vth_olympic_winter_games")],
    "VI Olympic Winter Games follows Vth Olympic Winter Games and ХVth Olympic Winter Games.", "U", S2,
    "second conjunct reads 'XVth' (Cyrillic X); VI does not follow XV",
    folds=[("ХVth Olympic Winter Games", "vth_olympic_winter_games")])
add("S2F04", [("f_minor", "follows", "e_minor")], "F minor follows E minor, E♭ minor and E# minor.", "U", S2,
    "three conjuncts, two of them different keys; all fold to e_minor",
    folds=[("E♭ minor", "e_minor"), ("E# minor", "e_minor")])

# ================================================================= 6. verb_reading_drift_new (U)
# English verb readings of relation tokens NOT named in Amendments 1-2 (sport/conflict/instrument/spouse/eat/author),
# where "SUBJ <reading> OBJ" means something else than the triple (drift) or reverses roles.
VD = "verb_reading_drift_new"
add("VD01", [("portmanteau_film", "subclass_of", "theatrical_film")], "Portmanteau film subclasses theatrical film.",
    "U", VD, "'to subclass' = to derive a subclass from: says theatrical film is derived from portmanteau film (reversed)")
add("VD02", [("motor_ships", "subclass_of", "wooden_ship")], "Motor ships subclassed wooden ship.", "U", VD,
    "past-tense 'subclassed' = created a subclass of wooden ship; reversed and an event")
add("VD03", [("agglutinative_language", "described_by_source", "armenian_soviet_encyclopedia")],
    "Agglutinative language describes Armenian Soviet Encyclopedia.", "U", VD,
    "'describes' reverses roles: the encyclopedia describes the language, not the other way round")
add("VD04", [("frank_lincoln_wright", "educated_at", "university_of_wisconsin_madison")],
    "Frank Lincoln Wright educates at University of Wisconsin Madison.", "U", VD,
    "'educates at' = teaches there; the triple says he was a student there")
add("VD05", [("palm_beach_county_fl", "named_after", "palm_beach_florida")],
    "Palm Beach County FL names Palm Beach Florida.", "U", VD,
    "'names' = gives a name to / mentions; reverses the naming direction")
add("VD06", [("brendan_gleeson", "award_received", "regional_emmy")], "Brendan Gleeson awards Regional Emmy.", "U", VD,
    "'awards' = he gives the award; the triple says he received it")
add("VD07", [("mutaib_bin_abdul_aziz", "father", "king_abdulaziz")], "Mutaib bin Abdul Aziz fathers King Abdulaziz.",
    "U", VD, "'fathers' = begets: says Mutaib is the father of the king (reversed); round 3 used only 'fathered'")
add("VD08", [("bomber_men", "platform", "world_of_spectrum")], "Bomber Men platforms World of Spectrum.", "U", VD,
    "'to platform' = give a platform to: the game would host the computer (reversed)")
add("VD09", [("keith_jarrett", "record_label", "atlantic_jazz")], "Keith Jarrett labels Atlantic Jazz.", "U", VD,
    "'labels' = puts a label on / categorises Atlantic Jazz; not 'his record label is'")
add("VD10", [("cornrow_kenny", "record_label", "interscope_records")], "Cornrow Kenny recorded Interscope Records.",
    "U", VD, "'recorded' = made a recording of the company; round 3 used only 'records'")
add("VD11", [("depressaria", "taxon_rank", "generic_epithet")], "Depressaria ranks generic epithet.", "U", VD,
    "transitive 'ranks' = Depressaria assigns a rank to / orders the generic epithet")
add("VD12", [("aloe_plant", "taxon_rank", "generic_epithet")], "Aloe plant ranked generic epithet.", "U", VD,
    "past transitive 'ranked': the plant rated something; not its taxon rank")
add("VD13", [("donald_pleasance", "military_branch", "royal_air_force")], "Donald Pleasance branches Royal Air Force.",
    "U", VD, "'branches' (-es reading of 'branch') = splits/forks the RAF")
add("VD14", [("linzer_ask", "headquarters_location", "linz_an_der_donau")], "Linzer ASK headquarters Linz an der Donau.",
    "U", VD, "transitive 'to headquarter' = to provide with headquarters: the club would house the city's HQ (reversed)")
add("VD15", [("software_conversions", "headquarters_location", "santa_monica_california_usa")],
    "Software Conversions headquartered Santa Monica California USA.", "U", VD,
    "active past 'headquartered Y' = gave Y headquarters; the triple needs passive 'is headquartered in'")
add("VD16", [("suhasini_maniratnam", "languages_spoken_written_or_signed", "number_of_english_words")],
    "Suhasini Maniratnam signed number of English words.", "U", VD,
    "'signed' = put a signature on (a number of English words); not 'speaks/writes/signs the language'")
add("VD17", [LKM], "Landkreis Munchen shares Munich West Germany.", "U", VD,
    "'shares X' = co-owns X; dropping 'border with' changes the relation")
add("VD18", [("brendan_gleeson", "place_of_birth", "dublin_city_centre")], "Brendan Gleeson placed Dublin City Centre.",
    "U", VD, "'placed' = put somewhere / ranked; round 3 used only 'places'")
add("VD19", [("albert_prince_of_wales", "position_held", "monarch")], "Albert Prince of Wales positioned monarch.",
    "U", VD, "'positioned' = moved the monarch into place; round 3 used only 'positions'")
add("VD20", [("palm_beach_county_fl", "located_in_the_administrative_territoria", "florida_united_states")],
    "Palm Beach County FL located Florida United States.", "U", VD,
    "active 'located Y' = found Y; the triple needs 'is located in'")
add("VD21", [("iron_man_one", "original_language_of_film_or_tv_show", "number_of_english_words")],
    "Iron Man One shows number of English words.", "U", VD,
    "'shows' (last relation token as a verb) = displays a number of English words")
add("VD22", [("agglutinative_language", "described_by_source", "armenian_soviet_encyclopedia")],
    "Agglutinative language sources Armenian Soviet Encyclopedia.", "U", VD,
    "'sources' = procures/obtains the encyclopedia")
add("VD23", [NAGOYA], "Nagoya University instanced higher general education.", "U", VD,
    "'instanced' = cited as an example; round 3 used only 'instances'")
add("VD24", [("leningrad_state_choreographic_institute", "part_of", "ballet_portal")],
    "Leningrad State Choreographic Institute parted Ballet Portal.", "U", VD,
    "'parted' = separated/divided Ballet Portal; round 3 used only 'parts'")
add("VD25", [("graham_grene", "employer", "secret_intelligence_service")],
    "Graham Grene employed Secret Intelligence Service.", "U", VD,
    "reversed: SIS employed Greene; round 3 used only 'employs'")
add("VD26", [("ben_hur_1925_film", "producer", "irving_g_thalberg")], "Ben Hur 1925 Film produces Irving G Thalberg.",
    "U", VD, "reversed: Thalberg produced the film; round 3 used only 'produced'")
add("VD27", [("helen_cho", "director", "joss_whedon")], "Helen Cho directs Joss Whedon.", "U", VD,
    "reversed: Whedon is the director; round 3 used only 'directed'")
add("VD28", [("bomber_men", "publisher", "sega_of_america")], "Bomber Men published Sega of America.", "U", VD,
    "reversed: Sega published the game; round 3 used only 'publishes'")
add("VD29", [("captain_mccluskey", "performer", "giovanni_rota")], "Captain McCluskey performed Giovanni Rota.", "U", VD,
    "reversed: Rota performs the role; round 3 used only 'performs'")

# ================================================================= 7. shape3_mid_of_no_the (U)
# Round 3's mid-of cases all used "the", which the new determiner rule rejects on its own. These use "a" / no
# determiner, so they test whether role reversal of mid-of relations is blocked, not just the article.
SM = "shape3_mid_of_no_the"
add("S3M01", [("frank_lincoln_wright", "country_of_citizenship", "u_s_of_a")],
    "Frank Lincoln Wright is a country of citizenship of U S of A.", "U", SM,
    "English: Wright is a country that is the citizenship-country of the USA (reversed and absurd)")
add("S3M02", [("brendan_gleeson", "place_of_birth", "dublin_city_centre")],
    "Brendan Gleeson is place of birth of Dublin City Centre.", "U", SM,
    "reversed: says Gleeson is where Dublin City Centre was born")
add("S3M03", [("james_whitmore", "place_of_death", "malibu_ca")], "James Whitmore is a place of death of Malibu CA.",
    "U", SM, "reversed: says Whitmore is the place where Malibu died")
add("S3M04", [("ralph_w_ince", "manner_of_death", "unintentional_injury")],
    "Ralph W Ince is a manner of death of unintentional injury.", "U", SM, "reversed: Ince would be a manner of death")
add("S3M05", [("law_and_order_svu", "country_of_origin", "u_s_of_a")],
    "Law and Order SVU is a country of origin of U S of A.", "U", SM, "reversed: the show would be the USA's origin country")
add("S3M06", [("mads_given_name", "language_of_work_or_name", "danish_language")],
    "Mads given name is a language of work or name of Danish language.", "U", SM,
    "reversed: the name would be a language of Danish")
add("S3M07", [("iron_man_one", "original_language_of_film_or_tv_show", "number_of_english_words")],
    "Iron Man One is an original language of film or TV show of number of English words.", "U", SM,
    "reversed: the film would be a language")
add("S3M08", [("massimo_girotti", "position_played_on_team_speciality", "goal_keeper")],
    "Massimo Girotti is a position played on team speciality of goal keeper.", "U", SM,
    "reversed: Girotti would be a position of the goalkeeper")

# ================================================================= 8. shape3_noun_relation_new (U)
# Shape 3 on non-_of noun relations not used in round 3 (given name, family name, military branch, screenwriter,
# performer, work location, headquarters location, taxon rank), all with "a" or no determiner.
SN = "shape3_noun_relation_new"
add("S3N01", [("james_whitmore", "given_name", "gemmes")], "James Whitmore is a given name of Gemmes.", "U", SN,
    "reversed: Whitmore would be Gemmes's given name")
add("S3N02", [("charles_h_west", "family_name", "mr_west")], "Charles H West is family name of Mr West.", "U", SN,
    "reversed: West would be the family name of Mr West")
add("S3N03", [("james_whitmore", "military_branch", "u_s_marine")], "James Whitmore is a military branch of U S Marine.",
    "U", SN, "reversed: Whitmore would be a branch of the Marines")
add("S3N04", [("thor_movie", "screenwriter", "stanley_martin_lieber")], "Thor Movie is a screenwriter of Stanley Martin Lieber.",
    "U", SN, "reversed: the film would have written Lieber")
add("S3N05", [("the_saint_mystery_magazine", "performer", "george_henry_sanders")],
    "The Saint Mystery Magazine is a performer of George Henry Sanders.", "U", SN, "reversed: the magazine would perform Sanders")
add("S3N06", [("keith_jarrett", "work_location", "new_york_new_york")], "Keith Jarrett is a work location of New York New York.",
    "U", SN, "reversed: Jarrett would be a place where New York works")
add("S3N07", [("linzer_ask", "headquarters_location", "linz_an_der_donau")],
    "Linzer ASK is headquarters location of Linz an der Donau.", "U", SN, "reversed: the club would be Linz's HQ location")
add("S3N08", [("aloe_plant", "taxon_rank", "generic_epithet")], "Aloe plant is a taxon rank of generic epithet.", "U", SN,
    "reversed: the plant would be a rank")

# ================================================================= 9. inverse_relation_reading (U)
# Entity order kept, relation replaced by its inverse; the gathered relation is the other direction.
IV = "inverse_relation_reading"
add("IV01", [("giro_d_italia", "has_part", "giro_d_italia_2014")], "Giro d'Italia is part of Giro d'Italia 2014.", "U", IV,
    "inverse: the 2014 race is part of the Giro, not the reverse")
add("IV02", [("ameerul_mu_mineen", "part_of", "all_sahaba_are_upright")], "Ameerul Mu Mineen has part All Sahaba Are Upright.",
    "U", IV, "inverse of part_of")
add("IV03", [("landkreis_munchen", "located_in_the_administrative_territoria", "upper_bavaria")],
    "Landkreis Munchen contains Upper Bavaria.", "U", IV, "inverse: Upper Bavaria contains the district")
add("IV04", [("trindad_and_tobago", "contains_administrative_territorial_enti", "port_o_spain")],
    "Trindad and Tobago is located in Port O Spain.", "U", IV, "inverse: Port of Spain is located in Trinidad and Tobago")

# ================================================================= 10. slug_affix_tolerance (U)
# A negating/qualifying affix of the gathered slug dropped or added. Exact slug matching rejects these; a
# prefix/suffix-tolerant or Det-stripping matcher would not.
AF = "slug_affix_tolerance"
add("AF01", [("american_library_association", "instance_of", "not_for_profit_organization")],
    "American Library Association is an instance of for profit organization.", "U", AF,
    "opposite claim: for-profit instead of not-for-profit")
add("AF02", [("popularization_of_science", "subclass_of", "non_fiction")], "Popularization of science is a subclass of fiction.",
    "U", AF, "opposite claim: fiction instead of non-fiction")
add("AF03", [("beiqing_insident", "instance_of", "anti_government_revolt")],
    "Beiqing Insident is an instance of pro-government revolt.", "U", AF,
    "opposite claim: pro-government instead of anti-government (shares the slug's suffix government_revolt)")
add("AF04", [("structural_dynamics_research_corporation", "instance_of", "for_profit")],
    "Structural Dynamics Research Corporation is an instance of not for profit.", "U", AF,
    "opposite claim: not-for-profit instead of for-profit")
add("AF05", [("bachelor_of_civil_engineering", "subclass_of", "non_degreed_engineer")],
    "Bachelor of civil engineering is a subclass of degreed engineer.", "U", AF, "drops 'non': opposite class")

# ================================================================= 11. determiner_changes_referent (U, one arguable)
DT = "determiner_changes_referent"
add("DT01", [("roger_daltrey_filmography", "member_of", "the_who")], "Roger Daltrey Filmography is a member of WHO.", "U", DT,
    "all-caps WHO reads as the World Health Organization; only a matcher that treats the slug's leading 'the' as an "
    "optional determiner would map it to the_who")
add("DT02", [("peter_dennis_blandford_townshend", "member_of", "the_who")],
    "Peter Dennis Blandford Townshend is a member of the WHO.", "U", DT,
    "'the WHO' reads as the World Health Organization, yet the token run spells the gathered slug the_who exactly "
    "after case folding; case is the only signal",
    folds=[("the WHO", "the_who")])
add("DT03", [("rembrandt_van_rhyn", "instance_of", "human_specie")], "A Rembrandt van Rhyn is an instance of human specie.",
    "U", DT, "the spec's own [Det] slot: 'a Rembrandt' is a painting by Rembrandt (or some person of that name), "
    "not the painter; a painting is not a human",
    arguable=True)

# ================================================================= 12. split_context_retraction (U; scored under the round-3 split rule)
# A neighbouring sentence or a non-period separator denies, hedges or frames the supported clause. Counts as a leak
# only if the checker keeps a span whose reading within the whole input is denied, hedged or framed
# (round-3 adjudication.json split_rule).
SP = "split_context_retraction"
SPW = " Split rule applies (round-3 adjudication split_rule)."
add("SP01", [LKM], "Myth: Landkreis Munchen shares border with Munich West Germany.", "U", SP,
    "colon-introduced label marks the clause as a myth." + SPW)
add("SP02", [COTE], "Cote d'Or shares border with Saone et Loire; or so the rumour goes.", "U", SP,
    "semicolon tail reports it as rumour." + SPW)
add("SP03", [LKM], "Landkreis Munchen shares border with Munich West Germany — not.", "U", SP,
    "em-dash tail negates the clause." + SPW)
add("SP04", [COTE], "False:\nCote d'Or shares border with Saone et Loire.", "U", SP,
    "newline after a 'False:' label." + SPW)
add("SP05", [NAGOYA], "Prob. Nagoya University is an instance of higher general education.", "U", SP,
    "'Prob.' (probably) is an abbreviation hedge a period splitter detaches." + SPW)
add("SP06", [LKM], "Landkreis Munchen shares border with Munich West Germany. Scratch that.", "U", SP,
    "the next sentence retracts it." + SPW)
add("SP07", [COTE], "Cote d'Or shares border with Saone et Loire. Or does it?", "U", SP,
    "the follow-up question puts it in doubt." + SPW)
add("SP08", [NAGOYA], "Imagine a world where this is true. Nagoya University is an instance of higher general education.",
    "U", SP, "the preceding sentence frames the next one as hypothetical." + SPW)
add("SP09", [LKM], "Landkreis Munchen shares border with Munich West Germany.[citation needed]", "U", SP,
    "Wikipedia-style tag marks the claim as unverified." + SPW)
add("SP10", [COTE], "Cote d'Or shares border with Saone et Loire. (This is false.)", "U", SP,
    "parenthetical sentence denies it." + SPW)
add("SP11", [NAGOYA], "Poss. Nagoya University is an instance of higher general education.", "U", SP,
    "'Poss.' (possibly) abbreviation hedge." + SPW)
add("SP12", [LKM], "Landkreis Munchen shares border with Munich West Germany. Not anymore.", "U", SP,
    "the next fragment says it no longer holds." + SPW)

# ================================================================= 13. glued_suffix_negation (U)
GL = "glued_suffix_negation"
add("GL01", [LKM], "Landkreis Munchen shares border with Munich West Germanyではない.", "U", GL,
    "Japanese 'dewa nai' (is not) glued as a suffix to the OBJ slug; round 3 used only prefixes/inner affixes",
    folds=[("Munich West Germanyではない", "munich_west_germany")])
add("GL02", [NAGOYA], "Nagoya University is an instance of higher general education아님.", "U", GL,
    "Korean '-anim' (is not) glued as a suffix to the OBJ slug",
    folds=[("higher general education아님", "higher_general_education")])

# ================================================================= 14. styled_letters_new_blocks (U)
ST = "styled_letters_new_blocks"
add("ST01", [NAGOYA], "Nagoya University is ⁿᵒᵗ an instance of higher general education.", "U", ST,
    "'not' in superscript modifier letters (U+207F U+1D52 U+1D57)")
add("ST02", [NAGOYA], "Nagoya University is \U0001d55f\U0001d560\U0001d565 an instance of higher general education.", "U", ST,
    "'not' in mathematical double-struck letters")
add("ST03", [COTE], "Cote d'Or \U0001d52b\U0001d522\U0001d533\U0001d522\U0001d52f shares border with Saone et Loire.", "U", ST,
    "'never' in mathematical fraktur letters")
add("ST04", [LKM], "Landkreis Munchen \U0001f17d\U0001f17e\U0001f183 shares border with Munich West Germany.", "U", ST,
    "'NOT' in negative squared Latin capitals")

# ================================================================= 15. enclosing_mark_prohibition (U)
EM = "enclosing_mark_prohibition"
add("EM01", [LKM], "Landkreis Munchen shares⃠ border with Munich West Germany.", "U", EM,
    "U+20E0 COMBINING ENCLOSING CIRCLE BACKSLASH (category Me, not Mn) draws a prohibition sign over the verb; "
    "an Mn-only accent stripper keeps it, the slug rule erases it")
add("EM02", [NAGOYA], "Nagoya University is⃠ an instance of higher general education.", "U", EM,
    "prohibition sign over the copula")
add("EM03", [COTE], "Cote d'Or shares border with Saone et Loire⃠.", "U", EM,
    "prohibition sign over the object",
    folds=[("Saone et Loire⃠", "saone_et_loire")])

# ================================================================= 16. emoji_or_tag_stance (U)
EJ = "emoji_or_tag_stance"
add("EJ01", [COTE], "Cote d'Or shares border with Saone et Loire \U0001f925", "U", EJ,
    "lying-face emoji: the speaker marks the statement as a lie")
add("EJ02", [NAGOYA], "Nagoya University is an instance of higher general education \U0001f914", "U", EJ,
    "thinking-face emoji: doubt, not assertion")
add("EJ03", [LKM], "Landkreis Munchen shares border with Munich West Germany \U0001f644", "U", EJ,
    "eye-roll emoji: sarcastic, not asserted")
add("EJ04", [COTE], "Cote d'Or shares border with Saone et Loire /s", "U", EJ,
    "'/s' is the standard internet sarcasm tag: the statement is not meant")

# ================================================================= 17. markup_strike (U)
MK = "markup_strike"
add("MK01", [LKM], "~Landkreis Munchen shares border with Munich West Germany~", "U", MK,
    "single-tilde GFM strikethrough renders the claim struck out; round 3 used only double tildes")
add("MK02", [COTE], "<del>Cote d'Or shares border with Saone et Loire.</del>", "U", MK,
    "HTML <del> renders the sentence as deleted")
add("MK03", [NAGOYA], "<s>Nagoya University is an instance of higher general education.</s>", "U", MK,
    "HTML <s> renders it struck through")

# ================================================================= 18. arguable_misc (U, all arguable)
AR = "arguable_misc"
add("AR01", [("acting_major", "follows", "army_captain")], "Acting major follows army captain.", "U", AR,
    "VERB-SAFE 'follows' drifts by argument type: read about people, an acting major follows (walks behind / obeys) "
    "an army captain; the triple means the rank comes next in order", arguable=True)
add("AR02", [("albert_prince_of_wales", "father", "albert_prince_consort")],
    "Albert Prince of Wales father Albert Prince Consort.", "U", AR,
    "telegraphic 'X father Y' (a bare noun-relation reading) is role-ambiguous: it reads equally as 'X, father of Y' "
    "(reversed); not plainly the triple", arguable=True)
add("AR03", [("albert_prince_of_wales", "position_held", "monarch")], "Albert Prince of Wales held monarch.", "U", AR,
    "'held monarch' reads literally as physically holding a monarch; a loose reader may supply 'the office of'",
    arguable=True)
add("AR04", [("no_saints", "member_of", "national_football_league")],
    "no saints is a member of national football league.", "U", AR,
    "lower-cased, the slug 'no saints' reads as a negative quantifier (no saint is a member), the opposite of the "
    "triple; the singular 'is' still allows a name reading", arguable=True)
add("AR05", [("v_i_lenin", "work_location", "moscow_ussr")], "V I Lenin works Moscow USSR.", "U", AR,
    "transitive 'works Moscow' = exploits / canvasses Moscow (present tense); idiomatically it can also mean "
    "'operates in', which is close to the triple", arguable=True)

# ================================================================= F twins (supported, plainly stated)
FT = "F_twin"
F = [
    ("F01", [DMAJ], "D major follows G major.", "twin of AC01-AC03, S2F01"),
    ("F02", [("f_minor", "follows", "e_minor")], "F minor follows E minor.", "twin of AC05, S2F04"),
    ("F03", [("e_major", "follows", "b_sharp_major")], "E major follows B sharp major.", "sharp spelled as a word, as the store does"),
    ("F04", [("g_minor", "follows", "c_minor")], "G minor follows C minor.", "twin of AC04"),
    ("F05", [("x_commonwealth_games", "follows", "ix_commonwealth_games")], "X Commonwealth Games follows IX Commonwealth Games.", "twin of HG01"),
    ("F06", [("vi_olympic_winter_games", "follows", "vth_olympic_winter_games")], "VI Olympic Winter Games follows Vth Olympic Winter Games.", "twin of HG03, HG06, HG08, S2F03"),
    ("F07", [("ix_commonwealth_games", "followed_by", "x_commonwealth_games")], "IX Commonwealth Games is followed by X Commonwealth Games.", "twin of HG02"),
    ("F08", [("vii_pan_american_games", "followed_by", "viii_pan_american_games")], "VII Pan American Games is followed by VIII Pan American Games.", "twin of HG04"),
    ("F09", [("stra_aros", "instance_of", "urban_areas_of_sweden")], "Östra Aros is an instance of urban areas of Sweden.", "twin of HG07; accented name folds to stra_aros"),
    ("F10", [ESC], "Eurovision Song Contest is a subclass of scripted television.", "twin of DG01, DG02"),
    ("F11", [GIRO], "Giro d'Italia is a subclass of stage race.", "twin of DG03, DG04"),
    ("F12", [("joseph_alois_ratzinger", "member_of", "academia_scientiarum_et_artium_europaea")], "Joseph Alois Ratzinger is a member of Academia Scientiarum et Artium Europaea.", "twin of CT01, CT02"),
    ("F13", [("2004_summer_paralympic_games", "follows", "sydney_paralyampic_organising_committee")], "2004 Summer Paralympic Games follows Sydney Paralyampic Organising Committee.", "twin of CT03"),
    ("F14", [("independent_spd", "follows", "sozialdemokratische_partei_deutschlands")], "Independent SPD follows Sozialdemokratische Partei Deutschlands.", "twin of CT04, S2F02"),
    ("F15", [("leningrad_state_choreographic_institute", "part_of", "ballet_portal")], "Leningrad State Choreographic Institute is part of Ballet Portal.", "twin of CT06, VD24"),
    ("F16", [("university_of_north_carolina_chapel_hill", "member_of", "midwest_inter_library_center")], "University of North Carolina Chapel Hill is a member of Midwest Inter Library Center.", "twin of CT07; exact 40-char slug"),
    ("F17", [("alberta_provincial_electoral_districts", "subclass_of", "canadian_provincial_electoral_districts")], "Alberta provincial electoral districts is a subclass of Canadian provincial electoral districts.", "twin of CT09"),
    ("F18", [("glamoc", "instance_of", "municipalities_of_bosnia_and_herzegovina")], "Glamoc is an instance of municipalities of Bosnia and Herzegovina.", "twin of CT10"),
    ("F19", [("peter_dennis_blandford_townshend", "member_of", "the_who")], "Peter Dennis Blandford Townshend is a member of The Who.", "twin of DT01, DT02"),
    ("F20", [("american_library_association", "instance_of", "not_for_profit_organization")], "American Library Association is an instance of not for profit organization.", "twin of AF01; 'not' belongs to the slug"),
    ("F21", [("popularization_of_science", "subclass_of", "non_fiction")], "Popularization of science is a subclass of non-fiction.", "twin of AF02; 'non' belongs to the slug"),
    ("F22", [("beiqing_insident", "instance_of", "anti_government_revolt")], "Beiqing Insident is an instance of anti-government revolt.", "twin of AF03"),
    ("F23", [("structural_dynamics_research_corporation", "instance_of", "for_profit")], "Structural Dynamics Research Corporation is an instance of for profit.", "twin of AF04"),
    ("F24", [LKM], "Landkreis Munchen shares border with Munich West Germany.", "twin of the LKM split/emoji/markup/styled cases and VD17"),
    ("F25", [COTE], "Cote d'Or shares border with Saone et Loire.", "twin of the COTE split/emoji/markup/styled cases"),
    ("F26", [NAGOYA], "Nagoya University is an instance of higher general education.", "twin of the NAGOYA cases"),
    ("F27", [("portmanteau_film", "subclass_of", "theatrical_film")], "Portmanteau film is a subclass of theatrical film.", "twin of VD01"),
    ("F28", [("rembrandt_van_rhyn", "instance_of", "human_specie")], "Rembrandt van Rhyn is an instance of human specie.", "twin of DT03"),
    ("F29", [("diego_vel_zquez", "instance_of", "human_specie")], "Diego Velázquez is an instance of human specie.", "accented spelling folds to diego_vel_zquez"),
    ("F30", [("giro_d_italia", "has_part", "giro_d_italia_2014")], "Giro d'Italia has part Giro d'Italia 2014.", "twin of IV01"),
    ("F31", [("ameerul_mu_mineen", "part_of", "all_sahaba_are_upright")], "Ameerul Mu Mineen is part of All Sahaba Are Upright.", "twin of IV02"),
    ("F32", [("trindad_and_tobago", "contains_administrative_territorial_enti", "port_o_spain")], "Trindad and Tobago contains Port O Spain.", "twin of IV04"),
    ("F33", [("constance_bennett", "sibling", "joan_bennett_filmography")], "Constance Bennett is a sibling of Joan Bennett Filmography.", "sibling is symmetric, so shape 3 on it states the triple exactly (retention probe for a non-_of noun)"),
    ("F34", [("rock_johnson", "member_of_sports_team", "calgary_stampeders")], "Rock Johnson is a member of Calgary Stampeders.", "plain English for member_of_sports_team (retention)"),
    ("F35", [("james_whitmore", "place_of_death", "malibu_ca")], "Malibu CA is the place of death of James Whitmore.", "twin of S3M03 in the correct English order (retention)"),
    ("F36", [("james_whitmore", "given_name", "gemmes")], "Gemmes is the given name of James Whitmore.", "twin of S3N01 in the correct English order (retention)"),
    ("F37", [("frank_lincoln_wright", "educated_at", "university_of_wisconsin_madison")], "Frank Lincoln Wright was educated at University of Wisconsin Madison.", "twin of VD04"),
    ("F38", [("palm_beach_county_fl", "named_after", "palm_beach_florida")], "Palm Beach County FL is named after Palm Beach Florida.", "twin of VD05"),
    ("F39", [("brendan_gleeson", "award_received", "regional_emmy")], "Brendan Gleeson received Regional Emmy.", "twin of VD06"),
    ("F40", [("linzer_ask", "headquarters_location", "linz_an_der_donau")], "Linzer ASK is headquartered in Linz an der Donau.", "twin of VD14, S3N07"),
    ("F41", [("keith_jarrett", "record_label", "atlantic_jazz")], "Keith Jarrett is signed to Atlantic Jazz.", "twin of VD09"),
    ("F42", [("agglutinative_language", "described_by_source", "armenian_soviet_encyclopedia")], "Agglutinative language is described by Armenian Soviet Encyclopedia.", "twin of VD03, VD22"),
    ("F43", [("no_saints", "member_of", "national_football_league")], "No Saints is a member of National Football League.", "twin of AR04; capitalised slug containing 'no'"),
    ("F44", [("world_fantasy_award_short_fiction", "is_a_list_of", "short_fiction")], "World Fantasy Award Short Fiction is a list of short fiction.", "is_a_list_of in its own words"),
]
for fid, g, s, w in F:
    add(fid, g, s, "F", FT, "supported: " + w)


# ================================================================= validation
def main():
    store = json.load(open(STORE))
    triples = {(x["fact"]["agent"], x["fact"]["action"], x["fact"]["patient"]) for x in store}
    relations = {t[1] for t in triples}
    r3 = json.load(open(ROUND3))
    r3_sentences = {c["sentence"] for c in r3}
    errors = []
    ids, sents = set(), set()
    for c in CASES:
        keys = set(c)
        if not REQUIRED_KEYS <= keys or not keys <= REQUIRED_KEYS | OPTIONAL_KEYS:
            errors.append(f"{c.get('id')}: bad key set {sorted(keys)}")
        if c["id"] in ids:
            errors.append(f"{c['id']}: repeated id")
        ids.add(c["id"])
        if c["sentence"] in sents:
            errors.append(f"{c['id']}: repeated sentence")
        sents.add(c["sentence"])
        if c["sentence"] in r3_sentences:
            errors.append(f"{c['id']}: sentence identical to a round-3 sentence")
        if c["expect"] not in ("U", "F"):
            errors.append(f"{c['id']}: expect must be U or F")
        if not 1 <= len(c["gathered"]) <= 4:
            errors.append(f"{c['id']}: gathered must hold 1-4 triples")
        for t in c["gathered"]:
            if len(t) != 3:
                errors.append(f"{c['id']}: malformed triple {t}")
                continue
            if t[1] not in relations:
                errors.append(f"{c['id']}: relation {t[1]} not in the store")
            if tuple(t) not in triples:
                errors.append(f"{c['id']}: triple {t} not in the store")
        if "arguable" in c and c["arguable"] is not True:
            errors.append(f"{c['id']}: arguable must be true when present")
    gathered_slugs = {}
    for c in CASES:
        gathered_slugs[c["id"]] = {s for t in c["gathered"] for s in (t[0], t[2])}
    for cid, checks in FOLDS.items():
        for span, slug in checks:
            got = sanitize(span)
            if got != slug:
                errors.append(f"{cid}: fold check failed: sanitize({span!r}) = {got!r}, expected {slug!r}")
            if slug not in gathered_slugs[cid]:
                errors.append(f"{cid}: fold target {slug} is not a gathered slug")
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print("  " + e)
        sys.exit(1)

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(CASES, fh, ensure_ascii=False, indent=1)
        fh.write("\n")

    by_label = collections.Counter(c["expect"] for c in CASES)
    by_class = collections.defaultdict(collections.Counter)
    for c in CASES:
        by_class[c["class"]][c["expect"]] += 1
    arguable = [c["id"] for c in CASES if c.get("arguable")]
    used_rel = {t[1] for c in CASES for t in c["gathered"]}
    print(f"OK: {len(CASES)} cases written to {OUT}")
    print(f"U = {by_label['U']}, F = {by_label['F']}, arguable = {len(arguable)} ({', '.join(arguable)})")
    print(f"store relations: {len(relations)}; relations used: {len(used_rel)}")
    print(f"fold self-checks passed: {sum(len(v) for v in FOLDS.values())} spans in {len(FOLDS)} cases")
    print(f"round-3 sentences compared: {len(r3_sentences)} (no overlap)")
    print("by class (U / F):")
    for cls in sorted(by_class, key=lambda k: (-sum(by_class[k].values()), k)):
        print(f"  {cls:32s} U={by_class[cls]['U']:3d}  F={by_class[cls]['F']:3d}")


if __name__ == "__main__":
    main()
