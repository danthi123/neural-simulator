"""REVIEWED PER-RELATION VERB-READING TABLE for the allowlist claim checker (`BRAIN_MOAT_ALLOWLIST`, research/runners/_moat_allowlist.py).

Prereg research/findings/2026-10-20-moat-allowlist-claim-checker-PREREGISTRATION.md, AMENDMENT 1 item 2; build 2 (2026-10-20) replaces the
role test with a MEANING test. The verb-form table the matcher reuses (B2h's `_build_inflection_map` + VERB_SYNONYMS + the guarded -es / -ies
readings) reads a relation's -s / -ed / -ing forms and its synonyms as the relation. Under the flag a relation takes those readings (shapes 1
and 2), or the shape-3 `X is [a|an] REL_of Y` reading, ONLY if its row below is VERB_SAFE or NOT_A_VERB. A ROLE_REVERSING or MEANING_DRIFT row,
and a relation with no row at all, gets none: every shape-1/2/3 sentence on it is rejected, with the class in the reason.

THE MEANING TEST (build 2, controller ruling after the round-3 blind hunt kept "X sports Y", "X conflicts Y", "X instruments Y"): a relation
keeps a reading only if a native reader would paraphrase "SUBJ <reading> OBJ" back to EXACTLY the triple's meaning. Keeping the subject's
role is not enough. The test is per READING (controller ruling 2026-10-20): a reading that fails it is listed in DROPPED_READINGS and gets no
verb reading while the row's passing readings stay; a row is MEANING_DRIFT when its own-label readings fail. When unsure: MEANING_DRIFT (or
ROLE_REVERSING when the roles are the problem).
  VERB_SAFE       the reading is an English verb phrase (finite, or a reduced passive such as "X named after Y") that paraphrases to the triple.
  NOT_A_VERB      the reading is a noun phrase or the property label read telegraphically ("X place of birth Y" -> "X was born in Y"); no form
                  is an English verb with another meaning. A participle INSIDE a label ("award received", "position held") modifies the noun;
                  the bare participle is never a reading, because the matcher only takes the whole slug.
  ROLE_REVERSING  the reading is an English verb whose subject is the triple's OBJECT ("X fathers Y" for (X, father, Y)), or the bare reading
                  is ambiguous between the two role orders ("X child Y" between two people).
  MEANING_DRIFT   the reading is an English sentence that keeps the roles but asserts a DIFFERENT relation ("X sports Y" = X wears Y, not
                  the sport of X is Y), or the paraphrase is not certainly the triple (conservative).
Tense and agreement are not a meaning change: the held-out F labels and the round-3 F labels treat "X followed Y" as the triple (X, follows, Y).
Multi-word slugs: their inflected readings ("family_names", "award_receiveds") exist only as underscore TOKENS; prose with spaces
normalises to the slug only when spelled exactly ("family name"), so for a multi-word relation the English reading is the bare phrase.
Build 5 (review 2026-10-20): an underscore token is still read as words ("has_parted" reads "has parted"), so a multi-word relation keeps
only the slug and the stub's own +s / +es token (`stub_third_person`); every other listed reading of it is dropped (`reading_dropped`).

INVENTORY (where the rows come from; column `sources`):
  store           EVERY relation of the production LTM store the matcher's gathered sets and the one-fact stub population come from:
                  /home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json (78,857 facts, all AFFIRM, 60 relations;
                  column `n_store` = its fact count = its stub sentences in the exhaustive stub population). The held-out strata's gathered
                  sets (S1-S3) and the golden / must-keep fixtures use no relation outside it. wikidata_core_15k (34 relations) is a subset.
  tiny_demo       research/runners/brain_chat_tui.py `_build_tiny_demo` (the TUI's default brain without --load): use, learn, store,
                  chase, eat.
  self_knowledge  research/findings/raw/_curriculum_self_knowledge.json "facts" (the --self-knowledge brain): 33 relations, each an
                  English 3rd-person verb.
  leak_hunt       relations of the out-of-repo leak-hunt fixtures (tests/fixtures/moat_allowlist_bank.json) that are in no store above:
                  border, borders, capital, founded_in, industry, located_in. Test vocabulary; classified so the skeptic's supported
                  twins keep their meaning.
NOT CLASSIFIED (fail closed, no verb reading under the flag): the extra relations of the larger bundles wikidata_500k_fast (135 beyond the
store), wikidata_1M_fast (314), wikidata_core_15k_grounded_v1 (alias_of) and wikidata_core_15k_smoke (4). Loading one of them under the flag
needs rows here first; until then every sentence on those relations is rejected.

`readings` is the exact set of tokens the verb-form table maps to the relation when it is gathered (pinned against the live table in
tests/test_moat_allowlist.py, so this file cannot go stale silently). Classified by Claude (build agent) 2026-10-20 for controller review.

BUILD 6: ONE REVIEWED PHRASE PER NO-READING RELATION (Claude rulings 2026-10-21, preflip topic 2). The nine ROLE-REVERSING / MEANING-DRIFT
relations still have NO VERB reading (shapes 1-3 reject "X sports Y", "X fathers Y", "X is a child of Y" read through shape 3), but a row of
PHRASE_READINGS voices each of them through the matcher's shape 4, "[Det] A PHRASE [Det] B [.]": PHRASE is the exact token sequence of the
row, FWD reads (A, rel, B) and INV reads (B, rel, A). One wording per relation, present tense as for every relation; "the" only where the
property is functional (father: single-best-value; parent_taxon: "closest parent taxon"), "a"/"an" everywhere else (the shape-3 rule that
"the" asserts uniqueness). Spouse stays orientation-exact (no symmetric reading). A relation without a row keeps no reading. Every row has
pinned role-reversal tests in tests/test_moat_allowlist_build6.py (the reversed sentence rejects).
"""
from __future__ import annotations

from collections import namedtuple

VERB_SAFE = "VERB-SAFE"
ROLE_REVERSING = "ROLE-REVERSING"
NOT_A_VERB = "NOT-A-VERB"
MEANING_DRIFT = "MEANING-DRIFT"
CLASSES = (VERB_SAFE, ROLE_REVERSING, NOT_A_VERB, MEANING_DRIFT)
# the classes that keep their verb reading (shapes 1/2) and, for an `_of` relation, shape 3
VERB_READING_CLASSES = frozenset((VERB_SAFE, NOT_A_VERB))
# the classes whose stock-kept stub sentences are counted as stock leaks, apart from HURTS (stub_population_exhaustive.py). They have no VERB
# reading; build 6: a PHRASE_READINGS row (below) can still voice them through shape 4, and the stub writes that phrase under the flag.
NO_READING_CLASSES = (ROLE_REVERSING, MEANING_DRIFT)
UNCLASSIFIED = "UNCLASSIFIED"
STORE_FACTS = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json"

Row = namedtuple("Row", "cls sources n_store readings why")

RELATIONS = {
    'author': Row(ROLE_REVERSING, ('store',), 27,
        ('author', 'authored', 'authores', 'authoring', 'authors'),
        '"X authors/authored Y" -> "X wrote Y"; the triple says Y wrote X (roles reversed)'),
    'award_received': Row(NOT_A_VERB, ('store',), 1556,
        ('award_received', 'award_receiveded', 'award_receivedes', 'award_receiveding', 'award_receiveds'),
        'property label: "X award received Y" -> "X received the award Y" = the triple; the bare "received" is never a reading'),
    'cast_member': Row(NOT_A_VERB, ('store',), 200,
        ('cast_member', 'cast_membered', 'cast_memberes', 'cast_membering', 'cast_members'),
        'lexicalised noun "cast member": "X cast member Y" -> "Y is in the cast of X" = the triple; the bare "cast" is never a reading'),
    'child': Row(ROLE_REVERSING, ('store',), 81,
        ('child', 'childed', 'childes', 'childing', 'childs'),
        'conservative: bare "X child Y" between two people reads both "Y is a child of X" and "X is a child of Y"; "X childs Y" is archaic'),
    'composer': Row(NOT_A_VERB, ('store',), 140,
        ('composer', 'composered', 'composeres', 'composering', 'composers'),
        '"X composer Y" -> "the composer of X is Y" = the triple; no form is an English verb'),
    'conflict': Row(MEANING_DRIFT, ('store',), 280,
        ('conflict', 'conflicted', 'conflictes', 'conflicting', 'conflicts'),
        '"X conflicts/conflicted Y" -> "X contradicts / clashed with Y", not "X took part in the conflict Y"'),
    'contains_administrative_territorial_enti': Row(VERB_SAFE, ('store',), 410,
        ('contains_administrative_territorial_enti', 'contains_administrative_territorial_entied', 'contains_administrative_territorial_enties', 'contains_administrative_territorial_entiing', 'contains_administrative_territorial_entis'),
        '"X contains administrative territorial enti(ty) Y" -> "the administrative unit Y lies in X" = the triple'),
    'country': Row(NOT_A_VERB, ('store',), 12743,
        ('countries', 'country', 'countryed', 'countryes', 'countrying', 'countrys'),
        '"X country Y" -> "the country of X is Y" = the triple; countries/countrys/countryed are no English verb'),
    'country_of_citizenship': Row(NOT_A_VERB, ('store',), 3151,
        ('country_of_citizenship', 'country_of_citizenshiped', 'country_of_citizenshipes', 'country_of_citizenshiping', 'country_of_citizenships'),
        '"X country of citizenship Y" -> "X is a citizen of Y" = the triple'),
    'country_of_origin': Row(NOT_A_VERB, ('store',), 538,
        ('country_of_origin', 'country_of_origined', 'country_of_origines', 'country_of_origining', 'country_of_origins'),
        '"X country of origin Y" -> "X comes from the country Y" = the triple'),
    'described_by_source': Row(VERB_SAFE, ('store',), 2024,
        ('described_by_source', 'described_by_sourced', 'described_by_sourcees', 'described_by_sources', 'described_by_sourcing'),
        '"X described by source Y" -> "the source Y describes X" = the triple'),
    'director': Row(NOT_A_VERB, ('store',), 41,
        ('director', 'directored', 'directores', 'directoring', 'directors'),
        '"X director Y" -> "the director of X is Y" = the triple; no form is an English verb'),
    'educated_at': Row(VERB_SAFE, ('store',), 1156,
        ('educated_at', 'educated_ated', 'educated_ates', 'educated_ating', 'educated_ats'),
        '"X educated at Y" -> "X was educated at Y" = the triple ("educate" needs an object, so the objectless form reads only as the reduced passive)'),
    'employer': Row(NOT_A_VERB, ('store',), 154,
        ('employer', 'employered', 'employeres', 'employering', 'employers'),
        '"X employer Y" -> "Y employs X" = the triple; no form is an English verb'),
    'family_name': Row(NOT_A_VERB, ('store',), 428,
        ('family_name', 'family_named', 'family_namees', 'family_names', 'family_naming'),
        '"X family name Y" -> "the family name of X is Y" = the triple'),
    'father': Row(ROLE_REVERSING, ('store',), 125,
        ('father', 'fathered', 'fatheres', 'fathering', 'fathers'),
        '"X fathers/fathered Y" -> "X is the father of Y"; the triple says Y is the father of X (roles reversed)'),
    'followed_by': Row(VERB_SAFE, ('store',), 413,
        ('followed_bies', 'followed_by', 'followed_byed', 'followed_byes', 'followed_bying', 'followed_bys'),
        '"X followed by Y" -> "Y comes after X" = the triple'),
    'follows': Row(VERB_SAFE, ('store',), 417,
        ('follows', 'followsed', 'followses', 'followsing'),
        '"X follows Y" -> "X comes after Y" = the triple; all 417 store subjects are series items, none a person (for whom it would read "is a follower of")'),
    'genre': Row(NOT_A_VERB, ('store',), 1250,
        ('genre', 'genred', 'genrees', 'genres', 'genring'),
        '"X genre Y" -> "the genre of X is Y" = the triple; no form is an English verb'),
    'given_name': Row(NOT_A_VERB, ('store',), 2032,
        ('given_name', 'given_named', 'given_namees', 'given_names', 'given_naming'),
        'lexicalised noun "given name": "X given name Y" -> "the first name of X is Y" = the triple; the bare "given" is never a reading'),
    'has_part': Row(VERB_SAFE, ('store',), 256,
        ('has_part', 'has_parted', 'has_partes', 'has_parting', 'has_parts'),
        '"X has part Y" -> "Y is a part of X" = the triple'),
    'headquarters_location': Row(NOT_A_VERB, ('store',), 2649,
        ('headquarters_location', 'headquarters_locationed', 'headquarters_locationes', 'headquarters_locationing', 'headquarters_locations'),
        '"X headquarters location Y" -> "X has its headquarters in Y" = the triple'),
    'instance_of': Row(NOT_A_VERB, ('store',), 13562,
        ('instance_of', 'instance_ofed', 'instance_ofes', 'instance_ofing', 'instance_ofs'),
        '"X instance of Y" / "X is an instance of Y" -> "X is a Y" = the triple; takes shape 3'),
    'instrument': Row(MEANING_DRIFT, ('store',), 410,
        ('instrument', 'instrumented', 'instrumentes', 'instrumenting', 'instruments'),
        '"X instruments/instrumented Y" -> "X equips / orchestrates Y", not "X plays the instrument Y"'),
    'is_a_list_of': Row(VERB_SAFE, ('store',), 58,
        ('is_a_list_of', 'is_a_list_ofed', 'is_a_list_ofes', 'is_a_list_ofing', 'is_a_list_ofs'),
        '"X is a list of Y" is the triple verbatim; takes shape 3'),
    'iucn_conservation_status': Row(NOT_A_VERB, ('store',), 5,
        ('iucn_conservation_status', 'iucn_conservation_statused', 'iucn_conservation_statuses', 'iucn_conservation_statusing'),
        '"X iucn conservation status Y" -> "the IUCN status of X is Y" = the triple'),
    'language_of_work_or_name': Row(NOT_A_VERB, ('store',), 1128,
        ('language_of_work_or_name', 'language_of_work_or_named', 'language_of_work_or_namees', 'language_of_work_or_names', 'language_of_work_or_naming'),
        '"X language of work or name Y" -> "X is in the language Y" = the triple'),
    'languages_spoken_written_or_signed': Row(NOT_A_VERB, ('store',), 1548,
        ('languages_spoken_written_or_signed', 'languages_spoken_written_or_signeded', 'languages_spoken_written_or_signedes', 'languages_spoken_written_or_signeding', 'languages_spoken_written_or_signeds'),
        'property label: "X languages spoken written or signed Y" -> "X speaks, writes or signs Y" = the triple; the bare "spoken"/"signed" is never a reading'),
    'located_in_the_administrative_territoria': Row(VERB_SAFE, ('store',), 4442,
        ('located_in_the_administrative_territoria', 'located_in_the_administrative_territoriaed', 'located_in_the_administrative_territoriaes', 'located_in_the_administrative_territoriaing', 'located_in_the_administrative_territorias'),
        '"X located in the administrative territoria(l entity) Y" -> "X lies in the administrative unit Y" = the triple'),
    'located_in_time_zone': Row(VERB_SAFE, ('store',), 3647,
        ('located_in_time_zone', 'located_in_time_zoned', 'located_in_time_zonees', 'located_in_time_zones', 'located_in_time_zoning'),
        '"X located in time zone Y" -> "X is in the time zone Y" = the triple ("locate" needs an object, so only the reduced passive reads)'),
    'manner_of_death': Row(NOT_A_VERB, ('store',), 929,
        ('manner_of_death', 'manner_of_deathed', 'manner_of_deathes', 'manner_of_deathing', 'manner_of_deaths'),
        '"X manner of death Y" -> "X died by Y" = the triple'),
    'member_of': Row(NOT_A_VERB, ('store',), 1168,
        ('member_of', 'member_ofed', 'member_ofes', 'member_ofing', 'member_ofs'),
        '"X member of Y" / "X is a member of Y" = the triple; takes shape 3 (a/an or no determiner only)'),
    'member_of_political_party': Row(NOT_A_VERB, ('store',), 270,
        ('member_of_political_parties', 'member_of_political_party', 'member_of_political_partyed', 'member_of_political_partyes', 'member_of_political_partying', 'member_of_political_partys'),
        '"X member of political party Y" -> "X belongs to the party Y" = the triple'),
    'member_of_sports_team': Row(NOT_A_VERB, ('store',), 14,
        ('member_of_sports_team', 'member_of_sports_teamed', 'member_of_sports_teames', 'member_of_sports_teaming', 'member_of_sports_teams'),
        '"X member of sports team Y" -> "X plays for the team Y" = the triple'),
    'military_branch': Row(NOT_A_VERB, ('store',), 206,
        ('military_branch', 'military_branched', 'military_branches', 'military_branching'),
        '"X military branch Y" -> "X served in the branch Y" = the triple'),
    'named_after': Row(VERB_SAFE, ('store',), 399,
        ('named_after', 'named_aftered', 'named_afteres', 'named_aftering', 'named_afters'),
        '"X named after Y" -> "X is named after Y" = the triple ("name" needs an object, so only the reduced passive reads)'),
    'occupation': Row(NOT_A_VERB, ('store',), 2981,
        ('occupation', 'occupationed', 'occupationes', 'occupationing', 'occupations'),
        '"X occupation Y" -> "the occupation of X is Y" = the triple; no form is an English verb'),
    'operator': Row(NOT_A_VERB, ('store',), 89,
        ('operator', 'operatored', 'operatores', 'operatoring', 'operators'),
        '"X operator Y" -> "Y operates X" = the triple; no form is an English verb'),
    'original_language_of_film_or_tv_show': Row(NOT_A_VERB, ('store',), 281,
        ('original_language_of_film_or_tv_show', 'original_language_of_film_or_tv_showed', 'original_language_of_film_or_tv_showes', 'original_language_of_film_or_tv_showing', 'original_language_of_film_or_tv_shows'),
        '"X original language of film or tv show Y" -> "X was made in the language Y" = the triple'),
    'parent_taxon': Row(ROLE_REVERSING, ('store',), 182,
        ('parent_taxon', 'parent_taxoned', 'parent_taxones', 'parent_taxoning', 'parent_taxons'),
        'conservative: bare "X parent taxon Y" between two taxa also reads "X is the parent taxon of Y", and "X parents Y" makes X the parent'),
    'part_of': Row(NOT_A_VERB, ('store',), 610,
        ('part_of', 'part_ofed', 'part_ofes', 'part_ofing', 'part_ofs'),
        '"X part of Y" / "X is a part of Y" = the triple; takes shape 3'),
    'participant_of': Row(NOT_A_VERB, ('store',), 195,
        ('participant_of', 'participant_ofed', 'participant_ofes', 'participant_ofing', 'participant_ofs'),
        '"X participant of Y" / "X is a participant of Y" = the triple; takes shape 3'),
    'performer': Row(NOT_A_VERB, ('store',), 7,
        ('performer', 'performered', 'performeres', 'performering', 'performers'),
        '"X performer Y" -> "Y performs X" = the triple; no form is an English verb'),
    'place_of_birth': Row(NOT_A_VERB, ('store',), 2198,
        ('place_of_birth', 'place_of_birthed', 'place_of_birthes', 'place_of_birthing', 'place_of_births'),
        '"X place of birth Y" -> "X was born in Y" = the triple'),
    'place_of_death': Row(NOT_A_VERB, ('store',), 1506,
        ('place_of_death', 'place_of_deathed', 'place_of_deathes', 'place_of_deathing', 'place_of_deaths'),
        '"X place of death Y" -> "X died in Y" = the triple'),
    'platform': Row(ROLE_REVERSING, ('store',), 21,
        ('platform', 'platformed', 'platformes', 'platforming', 'platforms'),
        '"X platforms Y" -> "X gives Y a platform"; the triple says Y is the platform of X (roles reversed)'),
    'position_held': Row(NOT_A_VERB, ('store',), 105,
        ('position_held', 'position_helded', 'position_heldes', 'position_helding', 'position_helds'),
        'property label: "X position held Y" -> "X held the position Y" = the triple'),
    'position_played_on_team_speciality': Row(NOT_A_VERB, ('store',), 10,
        ('position_played_on_team_specialities', 'position_played_on_team_speciality', 'position_played_on_team_specialityed', 'position_played_on_team_specialityes', 'position_played_on_team_specialitying', 'position_played_on_team_specialitys'),
        'property label: "X position played on team speciality Y" -> "X plays the position Y" = the triple'),
    'producer': Row(NOT_A_VERB, ('store',), 73,
        ('producer', 'producered', 'produceres', 'producering', 'producers'),
        '"X producer Y" -> "Y produced X" = the triple; no form is an English verb'),
    'publisher': Row(NOT_A_VERB, ('store',), 14,
        ('publisher', 'publishered', 'publisheres', 'publishering', 'publishers'),
        '"X publisher Y" -> "Y published X" = the triple; no form is an English verb'),
    'record_label': Row(NOT_A_VERB, ('store',), 548,
        ('record_label', 'record_labeled', 'record_labeles', 'record_labeling', 'record_labels'),
        '"X record label Y" -> "the record label of X is Y" = the triple'),
    'screenwriter': Row(NOT_A_VERB, ('store',), 17,
        ('screenwriter', 'screenwritered', 'screenwriteres', 'screenwritering', 'screenwriters'),
        '"X screenwriter Y" -> "Y wrote the screenplay of X" = the triple; no form is an English verb'),
    'shares_border_with': Row(VERB_SAFE, ('store',), 2351,
        ('shares_border_with', 'shares_border_withed', 'shares_border_withes', 'shares_border_withing', 'shares_border_withs'),
        '"X shares border with Y" -> "X borders Y" = the triple'),
    'sibling': Row(NOT_A_VERB, ('store',), 138,
        ('sibling', 'siblinged', 'siblinges', 'siblinging', 'siblings'),
        '"X sibling Y" -> "Y is a sibling of X" = the triple (symmetric); no form is an English verb'),
    'sport': Row(MEANING_DRIFT, ('store',), 4116,
        ('sport', 'sported', 'sportes', 'sporting', 'sports'),
        '"X sports/sported Y" -> "X wears / shows off Y", not "the sport of X is Y"'),
    'spouse': Row(MEANING_DRIFT, ('store',), 202,
        ('spouse', 'spoused', 'spousees', 'spouses', 'spousing'),
        'conservative: "X spouses/spoused Y" is the archaic verb "to marry" (an act), not "Y is the spouse of X"'),
    'subclass_of': Row(NOT_A_VERB, ('store',), 2004,
        ('subclass_of', 'subclass_ofed', 'subclass_ofes', 'subclass_ofing', 'subclass_ofs'),
        '"X subclass of Y" / "X is a subclass of Y" = the triple; takes shape 3'),
    'taxon_rank': Row(NOT_A_VERB, ('store',), 721,
        ('taxon_rank', 'taxon_ranked', 'taxon_rankes', 'taxon_ranking', 'taxon_ranks'),
        '"X taxon rank Y" -> "the taxon rank of X is Y" = the triple'),
    'twinned_administrative_body': Row(NOT_A_VERB, ('store',), 2437,
        ('twinned_administrative_bodies', 'twinned_administrative_body', 'twinned_administrative_bodyed', 'twinned_administrative_bodyes', 'twinned_administrative_bodying', 'twinned_administrative_bodys'),
        'property label: "X twinned administrative body Y" -> "X is twinned with Y" = the triple (symmetric)'),
    'work_location': Row(NOT_A_VERB, ('store',), 194,
        ('work_location', 'work_locationed', 'work_locationes', 'work_locationing', 'work_locations'),
        '"X work location Y" -> "X works in Y" = the triple'),
    'chase': Row(VERB_SAFE, ('tiny_demo',), 0,
        ('chase', 'chased', 'chasees', 'chases', 'chasing', 'pursue', 'pursued', 'pursues', 'pursuing'),
        '"X chases Y" is the triple verbatim; the table synonym "X pursues Y" -> "X chases Y" = the triple'),
    'eat': Row(VERB_SAFE, ('tiny_demo',), 0,
        ('ate', 'consume', 'consumed', 'consumes', 'consuming', 'devour', 'devoured', 'devouring', 'devours', 'eat', 'eated', 'eaten', 'eates', 'eating', 'eats'),
        '"X eats Y" is the triple verbatim; the table synonyms "X devours Y" (eats greedily) and "X consumes Y" (also: uses up) fail the meaning test and are DROPPED_READINGS'),
    'learn': Row(VERB_SAFE, ('tiny_demo',), 0,
        ('learn', 'learned', 'learnes', 'learning', 'learns'),
        '"X learns Y" is the triple verbatim (the slug is the English verb)'),
    'store': Row(VERB_SAFE, ('tiny_demo',), 0,
        ('store', 'stored', 'storees', 'stores', 'storing'),
        '"X stores Y" is the triple verbatim (the slug is the English verb)'),
    'use': Row(VERB_SAFE, ('tiny_demo',), 0,
        ('use', 'used', 'usees', 'uses', 'using'),
        '"X uses Y" is the triple verbatim (the slug is the English verb)'),
    'abstains': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('abstains', 'abstainsed', 'abstainses', 'abstainsing'),
        '"X abstains Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'adds': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('adds', 'addsed', 'addses', 'addsing'),
        '"X adds Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'admits': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('admits', 'admitsed', 'admitses', 'admitsing'),
        '"X admits Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'answers': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('answers', 'answersed', 'answerses', 'answersing'),
        '"X answers Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'assists': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('assists', 'assistsed', 'assistses', 'assistsing'),
        '"X assists Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'binds': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('binds', 'bindsed', 'bindses', 'bindsing'),
        '"X binds Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'carry': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('carries', 'carry', 'carryed', 'carryes', 'carrying', 'carrys'),
        '"X carry Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'checks': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('checks', 'checksed', 'checkses', 'checksing'),
        '"X checks Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'comes': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('comes', 'comesed', 'comeses', 'comesing'),
        '"X comes Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'consolidates': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('consolidates', 'consolidatesed', 'consolidateses', 'consolidatesing'),
        '"X consolidates Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'controls': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('controls', 'controlsed', 'controlses', 'controlsing'),
        '"X controls Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'develops': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('develops', 'developsed', 'developses', 'developsing'),
        '"X develops Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'fires': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('fires', 'firesed', 'fireses', 'firesing'),
        '"X fires Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'form': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('form', 'formed', 'formes', 'forming', 'forms'),
        '"X form Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'grows': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('grows', 'growsed', 'growses', 'growsing'),
        '"X grows Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'has': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('has', 'hased', 'hases', 'hasing'),
        '"X has Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'hold': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('hold', 'holded', 'holdes', 'holding', 'holds'),
        '"X hold Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'learns': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('learns', 'learnsed', 'learnses', 'learnsing'),
        '"X learns Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'persists': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('persists', 'persistsed', 'persistses', 'persistsing'),
        '"X persists Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'phrases': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('phrases', 'phrasesed', 'phraseses', 'phrasesing'),
        '"X phrases Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'prevents': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('prevents', 'preventsed', 'preventses', 'preventsing'),
        '"X prevents Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'reads': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('reads', 'readsed', 'readses', 'readsing'),
        '"X reads Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'refuses': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('refuses', 'refusesed', 'refuseses', 'refusesing'),
        '"X refuses Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'remembers': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('remembers', 'remembersed', 'rememberses', 'remembersing'),
        '"X remembers Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'replays': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('replays', 'replaysed', 'replayses', 'replaysing'),
        '"X replays Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'retrieves': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('retrieves', 'retrievesed', 'retrieveses', 'retrievesing'),
        '"X retrieves Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'runs': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('runs', 'runsed', 'runses', 'runsing'),
        '"X runs Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'send': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('send', 'sended', 'sendes', 'sending', 'sends'),
        '"X send Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'simulates': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('simulates', 'simulatesed', 'simulateses', 'simulatesing'),
        '"X simulates Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'stores': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('stores', 'storesed', 'storeses', 'storesing'),
        '"X stores Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'strengthens': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('strengthens', 'strengthensed', 'strengthenses', 'strengthensing'),
        '"X strengthens Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'teaches': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('teaches', 'teachesed', 'teacheses', 'teachesing'),
        '"X teaches Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'uses': Row(VERB_SAFE, ('self_knowledge',), 0,
        ('uses', 'usesed', 'useses', 'usesing'),
        '"X uses Y" is the triple verbatim (the slug is the English verb itself; its other forms only change tense)'),
    'border': Row(VERB_SAFE, ('leak_hunt',), 0,
        ('border', 'bordered', 'borderes', 'bordering', 'borders'),
        '"X borders Y" -> "X and Y share a border" = the triple'),
    'borders': Row(VERB_SAFE, ('leak_hunt',), 0,
        ('borders', 'bordersed', 'borderses', 'bordersing'),
        '"X borders Y" -> "X and Y share a border" = the triple'),
    'capital': Row(NOT_A_VERB, ('leak_hunt',), 0,
        ('capital', 'capitaled', 'capitales', 'capitaling', 'capitals'),
        '"X capital Y" -> "the capital of X is Y" = the triple; "capitals" is no English verb (never shape 3: no _of)'),
    'founded_in': Row(VERB_SAFE, ('leak_hunt',), 0,
        ('founded_in', 'founded_ined', 'founded_ines', 'founded_ining', 'founded_ins'),
        '"X founded in Y" -> "X was founded in Y" = the triple ("found" needs an object, so only the reduced passive reads)'),
    'industry': Row(NOT_A_VERB, ('leak_hunt',), 0,
        ('industries', 'industry', 'industryed', 'industryes', 'industrying', 'industrys'),
        '"X industry Y" -> "X is in the industry Y" = the triple; no form is an English verb'),
    'located_in': Row(VERB_SAFE, ('leak_hunt',), 0,
        ('located_in', 'located_ined', 'located_ines', 'located_ining', 'located_ins'),
        '"X located in Y" -> "X is located in Y" = the triple ("locate" needs an object, so only the reduced passive reads)'),
}


# build 6: the reviewed relation phrases of shape 4 (one per no-reading relation; module docstring). `text` is the exact phrase, lower-case
# ASCII; `orientation` FWD = "A PHRASE B" encodes (A, rel, B), INV = it encodes (B, rel, A); `why` quotes the Wikidata property description.
FWD = "FWD"
INV = "INV"
PhraseRow = namedtuple("PhraseRow", "text orientation why")

PHRASE_READINGS = {
    'sport': PhraseRow('is a sport of', INV,
        'P641 "sport that the subject participates or participated in or is associated with" (can have one or more values): '
        '"Y is a sport of X" for (X, sport, Y); "the sport of" would assert the only sport'),
    'instrument': PhraseRow('plays', FWD,
        'P1303 "musical instrument that a person plays or teaches or used in a music occupation": "X plays Y" for (X, instrument, Y) '
        '(the description\'s own verb; present tense as for every relation; "Y is an instrument of X" drifts to a tool/agent of X)'),
    'conflict': PhraseRow('participated in', FWD,
        'P607, label "participated in conflict": "battles, wars or other military engagements in which the person or item participated"; '
        '"X participated in Y" for (X, conflict, Y) ("fought in" adds combat)'),
    'spouse': PhraseRow('is a spouse of', INV,
        'P26 (symmetric, multi-valued): "Y is a spouse of X" for (X, spouse, Y); orientation-exact by ruling (no symmetric reading); "the" '
        'would assert the only spouse, "wife"/"husband" adds gender'),
    'parent_taxon': PhraseRow('is the parent taxon of', INV,
        'P171 "closest parent taxon of the taxon in question": "Y is the parent taxon of X" for (X, parent_taxon, Y); "the" rests on '
        '"closest" (functional)'),
    'father': PhraseRow('is the father of', INV,
        'P22 "male parent of the subject" (single-best-value constraint): "Y is the father of X" for (X, father, Y); "a father of" is wrong'),
    'child': PhraseRow('is a child of', INV,
        'P40 (multi-valued): "Y is a child of X" for (X, child, Y); "the child" asserts an only child, "son"/"daughter" adds gender'),
    'author': PhraseRow('is an author of', INV,
        'P50 (multi-valued): "Y is an author of X" for (X, author, Y); "the author" is false for a store row such as '
        '(new_testiment, author, peter_the_disciple)'),
    'platform': PhraseRow('is a platform for', INV,
        'P400 "platform for which a work was developed or released": "Y is a platform for X" for (X, platform, Y)'),
}


def phrase_reading(rel):
    """-> the reviewed shape-4 PhraseRow of `rel`, or None (no row: no phrase reading)."""
    return PHRASE_READINGS.get(rel)


# per-reading meaning test: readings of a VERB_SAFE / NOT_A_VERB row that fail it (today only table synonyms). Each is rejected like a
# MEANING_DRIFT reading, with that class in the reason; the row's other readings stay.
DROPPED_READINGS = {
    'eat': frozenset(('consume', 'consumed', 'consumes', 'consuming', 'devour', 'devoured', 'devouring', 'devours')),
}


def stub_third_person(rel):
    """The ONE inflected token the template stub mouth renders for `rel` (research/runners/_grounded_lang_p3_derisk.py `_inflect` for a
    relation with no irregular form: +es after s/sh/ch/x/z, else +s; pinned equal to `_inflect` over the table in
    tests/test_moat_allowlist_build5.py)."""
    return rel + ("es" if rel.endswith(("s", "sh", "ch", "x", "z")) else "s")


def reading_dropped(rel, tok):
    """True iff `tok` is a reading of `rel` that failed the per-reading meaning test, or (build 5, review of 2026-10-20) an
    underscore inflection of a MULTI-WORD relation other than the stub's own token: "has_parted" reads as "has parted" (separated
    from), not as has_part, and no mouth needs the -ed / -ing / -ies / second -es forms of a multi-word slug ("record_labeled",
    "member_of_political_parties"). A multi-word relation keeps the slug itself and `stub_third_person(rel)`; its English reading
    is the label written with spaces, which the exact check compares to the slug."""
    if "_" in rel and tok != rel and tok != stub_third_person(rel):
        return True
    return tok in DROPPED_READINGS.get(rel, ())


def relation_class(rel):
    """-> the class of `rel`, or UNCLASSIFIED when it has no row (no verb reading under the flag)."""
    row = RELATIONS.get(rel)
    return row.cls if row is not None else UNCLASSIFIED


def has_verb_reading(rel):
    """True iff `rel` may take a verb reading (shapes 1/2) or shape 3 under the flag."""
    return relation_class(rel) in VERB_READING_CLASSES
