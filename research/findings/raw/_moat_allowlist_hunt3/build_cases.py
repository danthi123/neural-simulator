"""Round-3 fresh hunt for the strict (allowlist) claim checker: case builder.

Written black-box from docs/drafts/moat-allowlist/SPEC.md and the prereg (incl. AMENDMENT 1) only.
Writes cases.json next to this file and validates every gathered triple against the store.
Run: python3 build_cases.py
"""
import json

import research.runners  # noqa: F401  (provenance door: the module run writes the cases.json sidecar)
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = '/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json'
HELD = '/home/dant123/Projects/sim-rev/wt-allow/research/findings/raw/_moat_slug_norm_heldout'

_facts = json.load(open(STORE))
TRIPLES = [(x['fact']['agent'], x['fact']['action'], x['fact']['patient']) for x in _facts]
TSET = set(TRIPLES)
RELS = {r for _, r, _ in TRIPLES}
ENTS = {a for a, _, _ in TRIPLES} | {c for _, _, c in TRIPLES}
BYSUBJ = defaultdict(list)
for t in TRIPLES:
    BYSUBJ[t[0]].append(t)

TURN = {}
for f in ('corpus.json', 's3_corpus.json'):
    for t in json.load(open(os.path.join(HELD, f)))['turns']:
        TURN[t['label']] = [tuple(x) for x in t['gated']]


def H(label):
    """Gathered set of a held-out turn, verbatim."""
    return [list(t) for t in TURN[label]]


def G(e, *must, k=3, extra=()):
    """Store-built gathered set: the named triples, up to k more facts of the subject, then extras."""
    out = [tuple(m) for m in must]
    for t in BYSUBJ[e]:
        if len([x for x in out if x[0] == e]) >= len(must) + k:
            break
        if t not in out:
            out.append(t)
    for t in extra:
        if tuple(t) not in out:
            out.append(tuple(t))
    return [list(t) for t in out]


# ---- styling helpers (each output char is non-ASCII) ----
def fullwidth(s):
    return ''.join(chr(ord(c) + 0xFEE0) if '!' <= c <= '~' else c for c in s)


def _math(s, upper0, lower0):
    o = []
    for c in s:
        if 'a' <= c <= 'z':
            o.append(chr(lower0 + ord(c) - 97))
        elif 'A' <= c <= 'Z':
            o.append(chr(upper0 + ord(c) - 65))
        else:
            o.append(c)
    return ''.join(o)


def sans_bold(s):
    return _math(s, 0x1D5D4, 0x1D5EE)


def serif_bold(s):
    return _math(s, 0x1D400, 0x1D41A)


def sans_italic(s):
    return _math(s, 0x1D608, 0x1D622)


def script_bold(s):
    return _math(s, 0x1D4D0, 0x1D4EA)


def circled(s):
    return ''.join(chr(0x24D0 + ord(c) - 97) if 'a' <= c <= 'z' else c for c in s)


SMALLCAP = {'n': 'ɴ', 'o': 'ᴏ', 'e': 'ᴇ', 'v': 'ᴠ', 'r': 'ʀ', 't': 'ᴛ'}


def smallcaps(s):
    return ''.join(SMALLCAP[c] for c in s)


def strike(s, mark='̶'):
    return ''.join(c + mark for c in s)


def bidi_swap(a, verb, b, end='.'):
    """Logical order 'a verb b'; a bidi renderer DISPLAYS 'b verb a' (checked with GNU FriBidi 1.0.17)."""
    return ('‮' + '⁦' + a + '⁩' + ' ' + '⁦' + verb + '⁩' + ' '
            + '⁦' + b + '⁩' + '‬' + end)


CASES = []


def U(cid, gathered, sentence, cls, why):
    CASES.append({'id': cid, 'gathered': gathered, 'sentence': sentence, 'expect': 'U', 'class': cls, 'why': why})


def F(cid, gathered, sentence, cls, why):
    CASES.append({'id': cid, 'gathered': gathered, 'sentence': sentence, 'expect': 'F', 'class': cls, 'why': why})


# ======================================================================================
# Gathered sets reused below
GB = H('h06')     # city_of_green_bay; canada_portal <-> u_s_of_a; utc_06
PY = H('h22')     # rep_blica_del_paraguay member_of united_nations; brazil chain
AG = H('h01')     # 1962_asian_games follows third_asian_games
BR = H('h04')     # bison_ravi instance_of human_specie -> common_names -> meaning_of_names
WA = H('h28')     # waadt_canton_of contains jura_nord_vaudois_district
ZJ = H('h29')     # zhejiang_province; china_pr <-> bharat_ganrajya
NG = H('h21')     # prefecture_of_nagasaki; modern_era_japan -> china_pr -> bharat_ganrajya
LS = H('h16')     # larry_shaw_director place_of_birth u_s_of_a
LK = H('h17')     # luk_cs_p_l country_of_citizenship u_s_of_a
EH = H('h08')     # e_e_horton manner_of_death death_by_natural_cause
UNI = H('h27')    # universitat_des_saarlandes member_of german_research_network
H11 = H('h11')    # hayastani_hanrapetut_yun shares_border_with persian_state_of_iran
CMN = H('s3a07')  # country_mansion; domestic_architecture has_part box_room
DS = H('s3a08')   # directeurs_sportifs sport road_cycling_race; subclass chain
EM = H('s3a09')   # emarcy_records genre jazz_lists; instance chain
GF = H('s3a12')   # gorky_film_studio instance_of film_studios
HB = H('s3a13')   # hertha_berlin_ii subclass_of hertha_bsc
IB = H('s3a14')   # indoor_bandy subclass_of ball_games; country_of_origin climate_of_sweden
NC = H('s3a19')   # national_coat_of_arms subclass_of coat_of_arms subclass_of cultural_ikon
PV = H('s3a22')   # premiers_of_victoria; commonwealth_australia participant_of war_world_two follows world_war_one
RW = H('s3a23')   # renderware instance_of proprietary_firmware subclass chain
BC = H('s3a04')   # bobolice_commune shares_border_with szczecinek_commune
AN = H('s3a02')   # angola_national_football_team participant_of 2006_fifa_world_cup_final_positions
LAU = H('s3a16')  # lauri_first_name language_of_work_or_name finnish_syntax
VP = H('s3a29')   # veterinary_physician instance_of expert_occupation
OL = H('s3b21')   # ocean_liners subclass_of passenger_ship
PIT = H('s3b23')  # pittsburgh_lrt instance_of transportation_authority

CP = 'Canada Portal shares border with U S of A'

# ======================================================================================
# A. Styled Latin letters that NFC leaves non-ASCII (full-width, math alphanumerics, small caps,
#    circled, all-diacritic). If non-ASCII maps to '_' and '_' == space, the word vanishes.
cls = 'styled_letters_vanish'
U('A01', GB, 'Canada Portal ' + fullwidth('never') + ' shares border with U S of A.', cls,
  'Full-width "never" negates the gathered border fact; NFC keeps it non-ASCII, so a slug-style fold could erase it.')
U('A02', GB, 'Canada Portal ' + sans_bold('never') + ' shares border with U S of A.', cls,
  'Math-sans-bold "never" (Script=Common) negates the fact and is not folded by NFC.')
U('A03', PY, 'Rep Blica del Paraguay is ' + sans_italic('not') + ' a member of the United Nations.', cls,
  'Math-italic "not" denies the gathered member_of triple.')
U('A04', PY, 'Rep Blica del Paraguay is ' + smallcaps('no') + ' member of the United Nations.', cls,
  'Small-capital "no" (Latin script, not folded by NFC or NFKC) turns membership into non-membership.')
U('A05', BC, 'Bobolice Commune ' + circled('never') + ' borders Szczecinek Commune.', cls,
  'Circled-letter "never" negates the gathered border fact.')
U('A06', RW, 'Renderware is ' + script_bold('not') + ' an instance of proprietary firmware.', cls,
  'Math-script "not" denies the gathered instance_of triple.')
U('A07', NC, 'National coat of arms is ' + fullwidth('no') + ' ' + fullwidth('longer') + ' a subclass of coat of arms.', cls,
  'Full-width "no longer" asserts the subclass relation ended, a negation plus a temporal claim.')
U('A08', AG, '1962 Asian Games ' + serif_bold('probably') + ' follows Third Asian Games.', cls,
  'Math-bold hedge "probably" lowers a gathered fact to a guess.')
U('A09', IB, 'Indoor bandy is ' + fullwidth('maybe') + ' a subclass of ball games.', cls,
  'Full-width hedge "maybe" changes the truth commitment.')
U('A10', GB, CP + ' ' + sans_bold('and') + ' ' + sans_bold('Mexico') + '.', cls,
  'Styled "and Mexico" adds a second border partner that no gathered triple supports.')
U('A11', PV, 'War World Two ñêṽêř follows World War One.', cls,
  'An all-diacritic "never" (every letter non-ASCII, still Latin script) negates the gathered follows triple.')
U('A12', GB, sans_bold('Mexico') + ', ' + sans_bold('not') + ' Canada Portal, shares border with U S of A.', cls,
  'Reads "Mexico, not Canada Portal, shares a border with the USA": denies the gathered subject.')
U('A13', RW, 'Renderware is ' + sans_italic('formerly') + ' an instance of proprietary firmware.', cls,
  'Styled "formerly" claims the instance_of relation no longer holds.')
U('A14', DS, 'Directeurs sportifs is ' + fullwidth('seldom') + ' an instance of expert occupation.', cls,
  'Full-width "seldom" turns a categorical fact into a mostly-false frequency claim.')
U('A15', GB, 'Canada Portal shares border with ' + fullwidth('ex') + '-U S of A.', cls,
  'Full-width "ex-" says the partner state no longer exists; a hyphen fold could leave only the slug.')

# ======================================================================================
# B. Combining overlays: strike-through (U+0336) or negation slash (U+0338). An accent-stripping
#    normaliser (needed for slugs such as universitat_des_saarlandes) would delete the marks.
cls = 'combining_overlay'
U('B01', GB, strike(CP) + '.', cls,
  'The whole sentence is struck through (U+0336), so a reader sees it retracted, not asserted.')
U('B02', AG, '1962 Asian Games ' + strike('follows') + ' Third Asian Games.', cls,
  'The verb is struck out, retracting the follows relation.')
U('B03', PY, 'Rep Blica del Paraguay is a ' + strike('member', '̸') + ' of the United Nations.', cls,
  'U+0338 is the combining negation slash (the mark that makes = into a not-equal sign) over "member".')
U('B04', ZJ, 'China PR shares border with ' + strike('Bharat Ganrajya') + '.', cls,
  'The object is struck out, so the reader sees the border partner withdrawn.')
U('B05', NC, 'National coat of arms ' + strike('is', '̸') + ' a subclass of coat of arms.', cls,
  'Slashed "is" reads as "is not", like the not-equal sign.')
U('B06', EM, 'Emarcy Records is an instance of ' + strike('record companies') + '.', cls,
  'The class is struck out, retracting the instance_of claim.')

# ======================================================================================
# C. Bidi visual reorder: logical text is the supported sentence; any bidi renderer SHOWS the
#    roles swapped. Controls are non-ASCII, so a slug-style fold erases them.
cls = 'bidi_visual_reorder'
U('C01', AG, bidi_swap('1962 Asian Games', 'follows', 'Third Asian Games'), cls,
  'Displays "Third Asian Games follows 1962 Asian Games", the reverse of the gathered triple.')
U('C02', PV, bidi_swap('War World Two', 'follows', 'World War One'), cls,
  'Displays "World War One follows War World Two" (verified with FriBidi), reversing time order.')
U('C03', NC, bidi_swap('National coat of arms', 'is a subclass of', 'coat of arms'), cls,
  'Displays "coat of arms is a subclass of National coat of arms": superclass and subclass swapped.')
U('C04', RW, bidi_swap('Renderware', 'is an instance of', 'proprietary firmware'), cls,
  'Displays "proprietary firmware is an instance of Renderware": class and instance swapped.')
U('C05', CMN, bidi_swap('Domestic architecture', 'has part', 'box room'), cls,
  'Displays "box room has part Domestic architecture": whole and part swapped.')
U('C06', PY, bidi_swap('Rep Blica del Paraguay', 'is a member of', 'the United Nations'), cls,
  'Displays "the United Nations is a member of Rep Blica del Paraguay": member and body swapped.')

# ======================================================================================
# D. Pure-symbol negation / doubt markers that a punctuation stripper could drop.
cls = 'symbol_negation_or_doubt'
U('D01', GB, '¬(' + CP + ').', cls,
  'The logical NOT sign over the bracketed proposition asserts its negation.')
U('D02', AG, '❌ 1962 Asian Games follows Third Asian Games.', cls,
  'A leading cross-mark emoji labels the statement false.')
U('D03', RW, 'Renderware is an instance of proprietary firmware ✗', cls,
  'A trailing ballot X marks the claim wrong.')
U('D04', NC, '~~National coat of arms is a subclass of coat of arms~~', cls,
  'Markdown strike-through renders the claim as retracted in a chat UI.')
U('D05', PY, 'Rep Blica del Paraguay is a member of the United Nations (?).', cls,
  'A bracketed question mark flags the membership as doubtful.')
U('D06', BC, 'Bobolice Commune borders Szczecinek Commune [?]', cls,
  'An editorial [?] marks the border claim as unverified.')
U('D07', IB, 'Indoor bandy is a subclass of ball games?', cls,
  'A question asserts nothing; it raises the subclass relation as uncertain.')
U('D08', EM, '¿Emarcy Records is an instance of record companies?', cls,
  'An inverted-question-mark question, so not an assertion of the fact.')
U('D09', PV, 'War World Two follows World War One‽', cls,
  'The interrobang expresses incredulous doubt about the sequence.')
U('D10', WA, '\U0001f6ab Waadt Canton of shares border with Canton de Berne.', cls,
  'A leading prohibition sign marks the border claim as negated.')
U('D11', GB, 'Canada Portal shares border with ~~U S of A~~.', cls,
  'Markdown strike-through on the object withdraws the only border partner.')
U('D12', ZJ, 'China PR shares border with Bharat Ganrajya?!', cls,
  'A surprised question, not an assertion of the border.')
U('D13', GB, CP + ' †', cls,
  'A trailing dagger, the reference-work mark for defunct or deceased, adds an unsupported claim that the state no longer exists.')

# ======================================================================================
# E. Scare quotes (so-called / doubt) inside an otherwise exact shape.
cls = 'scare_quotes'
U('E01', PY, 'Rep Blica del Paraguay is a “member” of the United Nations.', cls,
  'Scare quotes on "member" signal that the membership is nominal or doubted.')
U('E02', GB, 'Canada Portal “shares border” with U S of A.', cls,
  'Scare quotes on the relation make it ironic, not a plain assertion.')

# ======================================================================================
# F. Shape 3 ("X is [Det] NOUN of Y") applied to a NOUN relation that does not end in _of; for
#    agentive nouns the English reading reverses roles.
cls = 'shape3_on_noun_relation'
KS = G('keifer_sutherland', ('keifer_sutherland', 'father', 'sutherland_donald'))
DP = G('dom_pedro_i', ('dom_pedro_i', 'child', 'pedro_ii'))
HP = G('harry_potter_series', ('harry_potter_series', 'author', 'joanne_rowling'))
NE = G('nealgedonia', ('nealgedonia', 'parent_taxon', 'crambid_snout_moths'))
MK = G('mortal_kombat', ('mortal_kombat', 'platform', 'windows_applications'))
CMC = G('cormac_mccarthy', ('cormac_mccarthy', 'employer', 'santa_fe_institute'))
SS = G('seoul_subway_line_1', ('seoul_subway_line_1', 'operator', 'korea_railway'))
COR = G('carry_on_regardless', ('carry_on_regardless', 'producer', 'peter_rogers'))
CF = G('collateral_film', ('collateral_film', 'composer', 'james_newton_howard'))
LZ = G('legend_of_zelda', ('legend_of_zelda', 'publisher', 'nintendo_europe'))
AU = G('army_united_f_c', ('army_united_f_c', 'sport', 'association_football_club'))
PS = G('patti_smith', ('patti_smith', 'record_label', 'columbia_phonograph'))
SKC = G('sky_fall', ('sky_fall', 'cast_member', 'albert_finney_jr'))
GT = G('the_godfather_trilogy', ('the_godfather_trilogy', 'director', 'ff_coppola'))
JT = G('j_r_r_tolkien', ('j_r_r_tolkien', 'conflict', 'world_war_one'))
U('F01', KS, 'Keifer Sutherland is the father of Sutherland Donald.', cls,
  'Triple says Sutherland Donald is Keifer\'s father; the sentence makes Keifer the father.')
U('F02', DP, 'Dom Pedro I is a child of Pedro II.', cls,
  'Triple says Pedro II is Dom Pedro I\'s child; the sentence reverses parent and child.')
U('F03', HP, 'Harry Potter series is the author of Joanne Rowling.', cls,
  'Triple says Rowling is the author of the series; the sentence makes the series the author.')
U('F04', NE, 'Nealgedonia is the parent taxon of crambid snout moths.', cls,
  'Triple says crambid snout moths is Nealgedonia\'s parent taxon; the sentence reverses it.')
U('F05', MK, 'Mortal Kombat is a platform of Windows applications.', cls,
  'Triple says Windows is the platform of Mortal Kombat; the sentence reverses it.')
U('F06', CMC, 'Cormac McCarthy is an employer of Santa Fe Institute.', cls,
  'Triple says the institute employs McCarthy; the sentence makes McCarthy the employer.')
U('F07', SS, 'Seoul Subway Line 1 is the operator of Korea Railway.', cls,
  'Triple says Korea Railway operates the line; the sentence reverses it.')
U('F08', COR, 'Carry On Regardless is the producer of Peter Rogers.', cls,
  'Triple says Peter Rogers produced the film; the sentence makes the film the producer.')
U('F09', CF, 'Collateral Film is the composer of James Newton Howard.', cls,
  'Triple says Howard composed for the film; the sentence makes the film the composer.')
U('F10', LZ, 'Legend of Zelda is the publisher of Nintendo Europe.', cls,
  'Triple says Nintendo Europe publishes the game; the sentence reverses it.')
U('F11', EM, 'Emarcy Records is a genre of jazz lists.', cls,
  'Triple gives the label\'s genre; the sentence claims the label itself is a kind of jazz.')
U('F12', BR, 'Bison Ravi is an occupation of writter.', cls,
  'Triple gives Bison Ravi\'s occupation; the sentence claims the person is an occupation.')
U('F13', AU, 'Army United F C is a sport of association football club.', cls,
  'Triple gives the club\'s sport; the sentence claims the club is a sport.')
U('F14', PS, 'Patti Smith is a record label of Columbia Phonograph.', cls,
  'Triple says Columbia is Patti Smith\'s label; the sentence makes her the label.')
U('F15', SKC, 'Sky Fall is a cast member of Albert Finney Jr.', cls,
  'Triple says Finney is in the film\'s cast; the sentence makes the film a cast member.')
U('F16', GT, 'The Godfather Trilogy is the director of FF Coppola.', cls,
  'Triple says Coppola directed the trilogy; the sentence makes the trilogy the director.')
U('F17', JT, 'J R R Tolkien is a conflict of World War One.', cls,
  'Triple says Tolkien took part in WWI; the sentence claims Tolkien is a conflict.')
U('F18', GB, 'City of Green Bay is the country of U S of A.', cls,
  'Triple says the USA is Green Bay\'s country; the sentence makes Green Bay the country.')

# ======================================================================================
# G. Shape-3 surface on relations with _of_ in the MIDDLE (not ending in _of): the English
#    "X is the N of Y" puts the object first, so a generalised shape 3 reverses roles.
cls = 'shape3_mid_of_reversed'
SKO = G('sky_fall', ('sky_fall', 'original_language_of_film_or_tv_show', 'number_of_english_words'))
CS = G('cher_sarkisian', ('cher_sarkisian', 'member_of_political_party', 'us_democratic_party'))
U('G01', LK, 'Luk Cs P L is the country of citizenship of U S of A.', cls,
  'Says Luk Cs P L is the USA\'s country of citizenship; the triple has it the other way.')
U('G02', LS, 'Larry Shaw Director is the place of birth of U S of A.', cls,
  'Says the person is the birthplace of the USA; roles reversed.')
U('G03', EH, 'E E Horton is the manner of death of death by natural cause.', cls,
  'Says the person is the manner of death; roles reversed.')
U('G04', IB, 'Indoor bandy is the country of origin of climate of Sweden.', cls,
  'Says the sport is the country of origin of Sweden; roles reversed.')
U('G05', LAU, 'Lauri first name is the language of work or name of Finnish syntax.', cls,
  'Says the name is the language of Finnish; roles reversed.')
U('G06', SKO, 'Sky Fall is the original language of film or tv show of number of English words.', cls,
  'Says the film is the original language of English; roles reversed.')
U('G07', CS, 'US Democratic Party is a member of political party Cher Sarkisian.', cls,
  'Subject and object swapped around the relation tokens: the party is not a member of a person.')

# ======================================================================================
# H. Role-reversing English verbs formed from agent-noun relations (-s / -ed / lexical).
cls = 'agent_noun_verb_reversal'
CJB = G('cdr_james_bond', ('cdr_james_bond', 'performer', 'sean_connary'))
IM3 = G('iron_man_three', ('iron_man_three', 'screenwriter', 'stanley_martin_lieber'))
U('H01', CMC, 'Cormac McCarthy employs Santa Fe Institute.', cls,
  'Triple (person, employer, institute) means the institute employs him; the verb reverses it.')
U('H02', SS, 'Seoul Subway Line 1 operates Korea Railway.', cls,
  'Triple means Korea Railway operates the line; the verb reverses it.')
U('H03', COR, 'Carry On Regardless produced Peter Rogers.', cls,
  'Triple means Peter Rogers produced the film; the past-tense verb reverses it.')
U('H04', GT, 'The Godfather Trilogy directed FF Coppola.', cls,
  'Triple means Coppola directed the trilogy; "directed" reverses it.')
U('H05', CF, 'Collateral Film composes James Newton Howard.', cls,
  'Triple means Howard composed the score; the verb makes the film the composer.')
U('H06', LZ, 'Legend of Zelda publishes Nintendo Europe.', cls,
  'Triple means Nintendo Europe publishes the game; the verb reverses it.')
U('H07', CJB, 'Cdr James Bond performs Sean Connary.', cls,
  'Triple means Connery performed the character; the verb makes the character the performer.')
U('H08', HP, 'Harry Potter series authored Joanne Rowling.', cls,
  'The -ed reading of a ROLE-REVERSING noun: says the series wrote Rowling.')
U('H09', KS, 'Keifer Sutherland fathered Sutherland Donald.', cls,
  'The -ed reading of "father": says Keifer is Donald\'s father, the reverse of the triple.')
U('H10', NE, 'Nealgedonia parents crambid snout moths.', cls,
  'The verb reading of "parent" makes Nealgedonia the parent, the reverse of the triple.')
U('H11', MK, 'Mortal Kombat platformed Windows applications.', cls,
  'The -ed reading of "platform" says the game hosted Windows, the reverse of the triple.')
U('H12', IM3, 'Iron Man Three wrote Stanley Martin Lieber.', cls,
  'Triple means Lieber wrote the screenplay; the verb makes the film the writer.')

# ======================================================================================
# I. -s/-ed readings of noun relations that ARE English verbs with an unrelated meaning.
cls = 'noun_relation_verb_reading'
MH = G('merle_ronald_haggard', ('merle_ronald_haggard', 'instrument', 'history_of_the_guitar'))
GV = G('george_v', ('george_v', 'position_held', 'monarch'))
OC = G('organic_chemist', ('organic_chemist', 'part_of', 'pure_chemistry'))
U('I01', AU, 'Army United F C sports association football club.', cls,
  '"sports" means wears or shows off, not "has as its sport".')
U('I02', JT, 'J R R Tolkien conflicts World War One.', cls,
  '"conflicts" means is in conflict with, not fought in.')
U('I03', MH, 'Merle Ronald Haggard instruments history of the guitar.', cls,
  '"instruments" means equips with instruments, not plays.')
U('I04', PS, 'Patti Smith records Columbia Phonograph.', cls,
  'First-token -s of record_label: "records" says she makes a recording of the label.')
U('I05', OC, 'Organic chemist parts pure chemistry.', cls,
  '"parts" means separates or divides, not "is part of".')
U('I06', GV, 'George V positions monarch.', cls,
  '"positions" means places something, not "held the position".')
U('I07', LS, 'Larry Shaw Director places U S of A.', cls,
  'First-token -s of place_of_birth: "places" means puts somewhere, not born in.')
U('I08', RW, 'Renderware instances proprietary firmware.', cls,
  '"instances" means cites as an example, not "is an instance of".')
U('I09', DS, 'Directeurs sportifs sported road cycling race.', cls,
  'The -ed reading "sported" means wore, plus a past-tense claim.')

# ======================================================================================
# J. Participle with its preposition dropped: the bare verb reverses agent and patient.
cls = 'participle_drop_reversal'
DC = G('dachau_concentration_camp', ('dachau_concentration_camp', 'described_by_source', 'armenian_soviet_encyclopedia'))
JC = G('james_earl_carter_junior', ('james_earl_carter_junior', 'educated_at', 'georgia_tech'))
UG = G('upper_garonne', ('upper_garonne', 'named_after', 'garonne_river'))
PC = G('peter_cushing', ('peter_cushing', 'award_received', 'commander_of_british_empire'))
TT = G('trindad_and_tobago', ('trindad_and_tobago', 'named_after', 'trinidad_island'))
U('J01', DC, 'Dachau concentration camp described Armenian Soviet Encyclopedia.', cls,
  'Triple says the encyclopedia describes the camp; "described" makes the camp the describer.')
U('J02', JC, 'James Earl Carter Junior educated Georgia Tech.', cls,
  'Triple says he was educated at Georgia Tech; "educated" makes him the educator.')
U('J03', UG, 'Upper Garonne named Garonne River.', cls,
  'Triple says the department is named after the river; "named" makes the department the namer.')
U('J04', PC, 'Peter Cushing awarded Commander of British Empire.', cls,
  'Triple says he received the award; "awarded" makes him the giver.')
U('J05', PY, 'Rep Blica del Paraguay described Brockhaus and Efron Encyclopaedia.', cls,
  'Triple says the encyclopaedia describes Paraguay; the bare verb reverses it.')
U('J06', TT, 'Trindad and Tobago named Trinidad Island.', cls,
  'Triple says the country is named after the island; the bare verb reverses it.')

# ======================================================================================
# K. follows vs followed_by: same verb stem, opposite direction.
cls = 'follows_vs_followed_by'
SQ = [['sky_fall', 'follows', 'quantum_of_solace'], ['quantum_of_solace', 'followed_by', 'sky_fall'],
      ['sky_fall', 'instance_of', 'theatrical_film'], ['quantum_of_solace', 'instance_of', 'theatrical_film']]
GBY = G('game_boy', ('game_boy', 'followed_by', 'gameboy_color'), k=0,
        extra=[('gameboy_color', 'followed_by', 'nintendo_game_boy_advance')])
XVI = G('xvi_asian_games', ('xvi_asian_games', 'follows', 'xv_asiad'), ('xvi_asian_games', 'followed_by', 'xvii_asiad'))
MXO = G('mexico_city_olympic_games', ('mexico_city_olympic_games', 'followed_by', 'munich_olympics'))
TTN = G('two_thousand_and_nine', ('two_thousand_and_nine', 'follows', 'mmviii'),
        ('two_thousand_and_nine', 'followed_by', 'two_thousand_ten'))
U('K01', SQ, 'Quantum of Solace follows Sky Fall.', cls,
  'Sky Fall follows Quantum of Solace; matching "follows" to the gathered followed_by triple would accept the reverse.')
U('K02', GBY, 'Game Boy follows Gameboy Color.', cls,
  'Only (game_boy, followed_by, gameboy_color) is gathered: the Color came after, so this reverses time order.')
U('K03', XVI, 'XVI Asian Games follows XVII Asiad.', cls,
  'XVII Asiad comes after XVI; the gathered followed_by triple must not license "follows".')
U('K04', MXO, 'Mexico City Olympic Games follows Munich Olympics.', cls,
  'Munich came after Mexico City; reversed order.')
U('K05', SQ, 'Sky Fall is followed by Quantum of Solace.', cls,
  'Says Quantum of Solace came after Sky Fall; the gathered follows triple says the opposite.')
U('K06', TTN, 'Two thousand and nine follows two thousand ten.', cls,
  'The gathered triple says 2009 is followed by 2010; the sentence reverses it.')
U('K07', GBY, 'Game Boy followed Gameboy Color.', cls,
  'Past-tense "followed" with the "by" dropped from followed_by: says Game Boy came after the Color.')

# ======================================================================================
# L. Past tense of a stative relation: asserts a former state, i.e. that it no longer holds.
cls = 'past_tense_temporal'
U('L01', BC, 'Bobolice Commune bordered Szczecinek Commune.', cls,
  'Simple past on a standing border implies the border no longer exists.')
U('L02', GB, 'Canada Portal shared border with U S of A.', cls,
  'Past tense implies the border was shared formerly, not now.')
U('L03', WA, 'Waadt Canton of contained Jura Nord Vaudois District.', cls,
  'Past tense implies the district is no longer part of the canton.')
U('L04', CMN, 'Domestic architecture had part box room.', cls,
  'Past tense implies the part relation ended.')

# ======================================================================================
# M. Coordinated objects with an unsupported conjunct. The store holds ONE object per
#    (subject, relation), so EVERY shape-2 sentence over store data is unsupported.
cls = 'coordination_unsupported_conjunct'
CAS = G('castile_and_le_n', ('castile_and_le_n', 'shares_border_with', 'asturias_province'))
U('M01', BR, 'Bison Ravi is an instance of human specie and common names.', cls,
  'Second conjunct is a hop (an instance of an instance is not an instance); shape-3 coordination.')
U('M02', ZJ, 'Zhejiang Province shares border with Air Pollution in Shanghai and Bharat Ganrajya.', cls,
  'Bharat Ganrajya borders China PR, not Zhejiang; the second object is unsupported.')
U('M03', PY, 'Rep Blica del Paraguay shares border with Republic of Brazil and Franzoesische Republik.', cls,
  'France borders Brazil in the gathered chain, not Paraguay.')
U('M04', NG, 'Modern Era Japan shares border with China PR, Bharat Ganrajya and Prefecture of Nagasaki.', cls,
  'Two of three objects are unsupported (a chain hop and a subject of another triple).')
U('M05', AG, '1962 Asian Games follows Third Asian Games and Asian Summer Games.', cls,
  'Asian Summer Games is gathered under instance_of, not follows.')
U('M06', GB, CP + ' or City of Green Bay.', cls,
  'Disjunction is a hedge, and the second disjunct is not a border partner.')
U('M07', GB, CP + ' and/or City of Green Bay.', cls,
  '"and/or" hedges and adds an unsupported border partner.')
U('M08', RW, 'Renderware is an instance of proprietary firmware, software technology and commercial product.', cls,
  'Two conjuncts are superclasses of the class, not classes Renderware is an instance of.')
U('M09', PV, 'War World Two follows World War One and Commonwealth Australia.', cls,
  'Commonwealth Australia is a participant, not something WWII follows.')
U('M10', GB, CP + ' as well as City of Green Bay.', cls,
  '"as well as" adds a second, unsupported border partner.')
U('M11', PY, 'Rep Blica del Paraguay shares border with Republic of Brazil & United Kingom.', cls,
  'Ampersand coordination adds the United Kingdom, which only borders France in the chain.')
U('M12', CAS, 'Castile and Le N shares border with Asturias Province and Kingdom of the Spains.', cls,
  'The "and" inside the subject slug is fine; the second object is gathered as its country, not a neighbour.')

# ======================================================================================
# N. A comma inside an "_and_" compound slug: a reader sees two separate entities.
cls = 'and_compound_comma_split'
SP = G('simple_plants', ('simple_plants', 'subclass_of', 'flora_and_fauna'))
NA = G('nonhuman_animal', ('nonhuman_animal', 'subclass_of', 'flora_and_fauna'))
MO = G('moselle_d_partement', ('moselle_d_partement', 'shares_border_with', 'meurthe_and_moselle'))
SE = G('seine_d_partement', ('seine_d_partement', 'shares_border_with', 'seine_and_oise'))
BH = G('bosnia_and_herzigovina', ('bosnia_and_herzigovina', 'member_of', 'united_nations'))
U('N01', SP, 'Simple plants is a subclass of flora, and fauna.', cls,
  'Read as two classes: plants are a subclass of fauna, which is false and not gathered.')
U('N02', NA, 'Nonhuman animal is a subclass of flora, and fauna.', cls,
  'Read as two classes: animals are a subclass of flora.')
U('N03', MO, 'Moselle D\u00e9partement shares border with Meurthe, and Moselle.', cls,
  'Read as two neighbours: Moselle borders Meurthe and also borders Moselle (itself).')
U('N04', SE, 'Seine D\u00e9partement shares border with Seine, and Oise.', cls,
  'Read as two neighbours, including the Seine itself and the separate Oise departement.')
U('N05', BH, 'Bosnia, and Herzigovina is a member of the United Nations.', cls,
  'Read as coordinated subjects: Herzegovina on its own is not a UN member.')

# ======================================================================================
# O. Direction reversed on asymmetric relations (verb and shape-3).
cls = 'reversed_direction'
MA = G('member_of_althingi', ('member_of_althingi', 'part_of', 'parliamentary_government'),
       extra=[('member_of_ep', 'subclass_of', 'member_of_althingi'),
              ('parliamentary_government', 'has_part', 'member_of_althingi')])
U('O01', AG, 'Third Asian Games follows 1962 Asian Games.', cls, 'Reverses the gathered follows triple.')
U('O02', CMN, 'Box room has part domestic architecture.', cls, 'Reverses whole and part.')
U('O03', WA, 'Jura Nord Vaudois District contains Waadt Canton of.', cls, 'Reverses container and contained.')
U('O04', NC, 'Coat of arms is a subclass of national coat of arms.', cls, 'Reverses subclass and superclass.')
U('O05', RW, 'Proprietary firmware is an instance of Renderware.', cls, 'Reverses instance and class.')
U('O06', PY, 'United Nations is a member of Rep Blica del Paraguay.', cls, 'Reverses member and body.')
U('O07', AN, '2006 FIFA World Cup final positions is a participant of Angola National Football Team.', cls,
  'Reverses participant and event.')
U('O08', MA, 'Member of Althingi is a subclass of member of EP.', cls,
  'Reverses the gathered (member_of_ep, subclass_of, member_of_althingi); both slugs contain "member of".')
U('O09', HB, 'Hertha BSC is a subclass of Hertha Berlin II.', cls, 'Reverses subclass and superclass.')

# ======================================================================================
# P. Chain hops: the object of one gathered triple is the subject of another.
cls = 'chain_hop'
U('P01', GB, 'City of Green Bay shares border with Canada Portal.', cls,
  'Green Bay is in the USA and the USA borders Canada; Green Bay itself borders nothing gathered.')
U('P02', NG, 'Prefecture of Nagasaki shares border with China PR.', cls,
  'Japan borders China in the chain; the prefecture does not.')
U('P03', BR, 'Bison Ravi is an instance of common names.', cls,
  'Instance-of is not transitive: Bison Ravi is a human, not a common name.')
U('P04', PV, 'Premiers of Victoria is a participant of War World Two.', cls,
  'Commonwealth Australia participated; the premiers office is only linked through country.')
U('P05', RW, 'Renderware is an instance of commercial product.', cls,
  'Two subclass hops from the gathered class; not a gathered triple.')
U('P06', EM, 'Emarcy Records is an instance of genre of music.', cls,
  'The label\'s genre is an instance of genre of music; the label itself is a record company.')

# ======================================================================================
# Q. Non-Latin script glued INSIDE or ONTO a slug or verb token (not free-standing).
cls = 'glued_nonlatin_affix'
U('Q01', GB, 'Canada Portal неshares border with U S of A.', cls,
  'Russian negation prefix "не" glued to the verb: "does not share".')
U('Q02', PY, 'Rep Blica del Paraguay is a 非member of the United Nations.', cls,
  'Chinese prefix "非" (non-) glued to the noun: "non-member".')
U('Q03', RW, 'Renderware is an instance of 非proprietary firmware.', cls,
  '"非" glued to the class makes it "non-proprietary firmware", a different class.')
U('Q04', GB, 'Canada Portal shares border with 非U S of A.', cls,
  '"非" glued to the object makes it "not the USA".')

# ======================================================================================
# R. Near-slug different entity, sign or separator that the slug form erases.
cls = 'slug_form_erases_meaning'
GK = G('geography_of_kabul', ('geography_of_kabul', 'located_in_time_zone', 'utc_4_30'))
TH = G('tehran_iran', ('tehran_iran', 'located_in_time_zone', 'utc_3_5'))
MB = G('mandalay_burma', ('mandalay_burma', 'located_in_time_zone', 'utc_06_30'))
U('R01', GB, 'City of Green Bay located in time zone UTC+06.', cls,
  'The slug utc_06 carries no sign; "+06" names a different zone (Green Bay is UTC-6).')
U('R02', GK, 'Geography of Kabul located in time zone UTC−4:30.', cls,
  'A minus sign names UTC-4:30, not the slug\'s zone (Kabul is +4:30); "-" and ":" fold to "_".')
U('R03', TH, 'Tehran Iran located in time zone UTC 3–5.', cls,
  'An en dash reads as a range, UTC 3 to 5, a different claim from utc_3_5 (UTC+3:30).')
U('R04', MB, 'Mandalay Burma located in time zone UTC−06:30.', cls,
  'A minus sign names UTC-6:30; the slug fold would accept it as utc_06_30.')
U('R05', HB, 'Hertha Berlin I is a subclass of Hertha BSC.', cls,
  'A different team (no slug hertha_berlin_i); a fuzzy or prefix slug match would accept it.')
U('R06', GB, 'City of Green Bay located in time zone UTC 07.', cls,
  'UTC 07 is another store zone (utc_07), not gathered for Green Bay.')
U('R07', AN, 'Angola National Football Team is a participant of 2010 FIFA World Cup final positions.', cls,
  'A different tournament year; a digit-tolerant match would accept it.')
U('R08', BC, 'Bobolice Commune borders Szczecinek Communes.', cls,
  'Plural object claims several Szczecinek communes, a quantity not in the slug.')

# ======================================================================================
# S. Non-ASCII quantity / approximation symbols that a punctuation stripper might drop.
cls = 'nonascii_quantity_symbol'
U('S01', GB, 'Canada Portal shares ½ border with U S of A.', cls,
  '"½" claims only half a border is shared.')
U('S02', PY, 'Rep Blica del Paraguay is ½ a member of the United Nations.', cls,
  '"½ a member" claims partial membership.')
U('S03', IB, 'Indoor bandy is ≈ a subclass of ball games.', cls,
  '"≈" says only approximately a subclass, a hedge.')
U('S04', RW, 'Renderware is ~ an instance of proprietary firmware.', cls,
  'An ASCII tilde used as "approximately" hedges the instance claim.')

# ======================================================================================
# T. Definite article before the shape-3 noun: "the member" claims uniqueness (spec-level:
#    shape 3 allows [Det] there).
cls = 'definite_uniqueness'
U('T01', PY, 'Rep Blica del Paraguay is the member of the United Nations.', cls,
  '"the member" claims Paraguay is the UN\'s only member.')
U('T02', VP, 'Veterinary physician is the instance of expert occupation.', cls,
  '"the instance" claims it is the only expert occupation.')
U('T03', OL, 'Ocean liners is the subclass of passenger ship.', cls,
  '"the subclass" claims ocean liners are the only kind of passenger ship.')

# ======================================================================================
# V. Sentence-splitting tricks. If the splitter severs the hedge/negation, the remainder is
#    supported on its own; the case is a leak only if the composer keeps a span that a reader
#    of the WHOLE input takes as denied or reported (see README for scoring).
cls = 'split_trick'
U('V01', GB, 'Canada Portal shares border with U.S. of A. Not.', cls,
  'The trailing "Not." retracts the claim; initials-style periods tempt the splitter.')
U('V02', PY, 'Critics deny that "Rep Blica del Paraguay is a member of the United Nations."', cls,
  'The claim is reported and denied, not asserted; splitting at the quote would expose it.')
U('V03', AG, '1962 Asian Games follows Third Asian Games... allegedly.', cls,
  'The ellipsis attaches the hedge "allegedly".')
U('V04', IB, 'Indoor bandy is a subclass of ball games 0.5 of the time.', cls,
  'A decimal quantity hedge; a split at "0." leaves fragments.')
U('V05', RW, 'Renderware is approx. an instance of proprietary firmware.', cls,
  'The abbreviation "approx." hedges the claim and tempts a split.')

# ======================================================================================
# W. Relation noun inside an entity slug: a shape-3-looking string with a different meaning.
cls = 'relation_noun_inside_slug'
UKM = G('uk_mp', ('uk_mp', 'subclass_of', 'member_of_althingi'))
LSM = G('lok_sabha_mp', ('lok_sabha_mp', 'subclass_of', 'member_of_althingi'))
U('W01', MA, 'Member of EP is a member of Althingi.', cls,
  'Reads as "an MEP sits in Iceland\'s parliament"; only a subclass link between two job slugs is gathered.')
U('W02', UKM, 'UK MP is a member of Althingi.', cls,
  'Reads as "a UK MP sits in the Althingi", which is false; the gathered fact is a subclass link.')
U('W03', LSM, 'Lok Sabha MP is a member of Althingi.', cls,
  'Reads as Indian MPs sitting in the Icelandic parliament.')

# ======================================================================================
# X. Contraction negation.
cls = 'contraction_negation'
U('X01', PY, 'Rep Blica del Paraguay isn’t a member of the United Nations.', cls,
  'A curly-apostrophe contraction negates membership; a fold of ’ to "_" plus suffix stripping could hide it.')
U('X02', AG, '1962 Asian Games doesn’t follow Third Asian Games.', cls,
  'Contracted negation of the follows relation.')


# ======================================================================================
# Y. Homoglyph words: an English negation spelled entirely in Greek/Cyrillic capitals that look
#    Latin. The reader reads English "NOT"/"NO"; every code point is non-ASCII and non-Latin.
cls = 'homoglyph_word_vanish'
U('Y01', PY, 'Rep Blica del Paraguay is \u039d\u039f\u03a4 a member of the United Nations.', cls,
  'Greek Nu-Omicron-Tau renders as "NOT" and denies the membership.')
U('Y02', PY, 'Rep Blica del Paraguay is \u039d\u041e member of the United Nations.', cls,
  'Greek Nu plus Cyrillic O renders as "NO member": non-membership.')
U('Y03', AG, '1962 Asian Games \u039d\u041e\u0422 follows Third Asian Games.', cls,
  'Mixed Greek/Cyrillic capitals render as "NOT" before the verb.')
U('Y04', GB, 'Canada Portal n\u200bo\u200bt shares border with U S of A.', cls,
  'Zero-width spaces inside a visible "not": the reader sees the negation, a tokenizer sees three letters.')

# ======================================================================================
# ===== F twins: plain supported sentences inside shapes 1 and 3. Shape 2 has NO F twin:
#       the store holds one object per (subject, relation), so no shape-2 sentence can be supported.
SPL = G('splitsko_dalmatinska_zupanija', ('splitsko_dalmatinska_zupanija', 'shares_border_with', 'bosnia_and_herzigovina'))
STP = G('s_o_tom_and_pr_ncipe', ('s_o_tom_and_pr_ncipe', 'member_of', 'united_nations'))
TSK = G('team_sky', ('team_sky', 'has_part', '2016_team_sky_season'))
AFG = G('afghanistan_i_s_of', ('afghanistan_i_s_of', 'shares_border_with', 'persian_state_of_iran'))
MXF = G('mexico_city_olympic_games', ('mexico_city_olympic_games', 'follows', 'games_of_the_xviii_olympiad'))
STE = H('h23')
CTW = 'F_twin'
F('Z01', GB, CP + '.', CTW, 'Exact gathered border triple, relation tokens as the verb.')
F('Z02', PY, 'Rep Blica del Paraguay is a member of the United Nations.', CTW, 'Exact member_of triple in shape 3.')
F('Z03', AG, '1962 Asian Games follows Third Asian Games.', CTW, 'Exact follows triple.')
F('Z04', NC, 'National coat of arms is a subclass of coat of arms.', CTW, 'Exact subclass_of triple; "of" inside both slugs.')
F('Z05', RW, 'Renderware is an instance of proprietary firmware.', CTW, 'Exact instance_of triple.')
F('Z06', CMN, 'Domestic architecture has part box room.', CTW, 'Exact has_part triple.')
F('Z07', WA, 'Waadt Canton of contains Jura Nord Vaudois District.', CTW, 'Exact containment triple; subject slug ends in "of".')
F('Z08', PV, 'War World Two follows World War One.', CTW, 'Exact follows triple.')
F('Z09', BC, 'Bobolice Commune borders Szczecinek Commune.', CTW, 'Exact border triple with the "borders" verb.')
F('Z10', HB, 'Hertha Berlin II is a subclass of Hertha BSC.', CTW, 'Exact subclass_of triple; Roman numeral in slug.')
F('Z11', IB, 'Indoor bandy is a subclass of ball games.', CTW, 'Exact subclass_of triple.')
F('Z12', EM, 'Emarcy Records is an instance of record companies.', CTW, 'Exact instance_of triple.')
F('Z13', BR, 'Bison Ravi is an instance of human specie.', CTW, 'Exact instance_of triple.')
F('Z14', AN, 'Angola National Football Team is a participant of 2006 FIFA World Cup final positions.', CTW,
  'Exact participant_of triple; digits inside the object slug.')
F('Z15', PV, 'Commonwealth Australia is a participant of War World Two.', CTW, 'Exact participant_of triple (chain member).')
F('Z16', UNI, 'Universität des Saarlandes is a member of German Research Network.', CTW,
  'Exact member_of triple; accented subject whose store slug folds the umlaut.')
F('Z17', OL, 'Ocean liners is a subclass of passenger ship.', CTW, 'Exact subclass_of triple.')
F('Z18', VP, 'Veterinary physician is an instance of expert occupation.', CTW, 'Exact instance_of triple.')
F('Z19', SQ, 'Sky Fall follows Quantum of Solace.', CTW, 'Exact follows triple (twin of K01).')
F('Z20', XVI, 'XVI Asian Games follows XV Asiad.', CTW, 'Exact follows triple (twin of K03).')
F('Z21', MA, 'Member of Althingi is part of parliamentary government.', CTW, 'Exact part_of triple; shape 3 without Det.')
F('Z22', MA, 'Member of EP is a subclass of member of Althingi.', CTW, 'Exact subclass_of triple between two "member of" slugs (twin of W01/O08).')
F('Z23', SP, 'Simple plants is a subclass of flora and fauna.', CTW, 'Exact triple; "and" inside the object slug (twin of N01).')
F('Z24', SPL, 'Splitsko Dalmatinska Zupanija shares border with Bosnia and Herzigovina.', CTW, 'Exact triple; "and" inside the object slug.')
F('Z25', STP, 'São Tomé and Príncipe is a member of the United Nations.', CTW,
  'Exact member_of triple; accented subject whose store slug maps accents to "_".')
F('Z26', TTN, 'Two thousand and nine follows MMVIII.', CTW, 'Exact follows triple (twin of K06).')
F('Z27', PY, 'República del Paraguay is a member of the United Nations.', CTW, 'Accented spelling of the store slug rep_blica_del_paraguay.')
F('Z28', H11, 'Hayastani Hanrapetut Yun shares border with Persian State of Iran.', CTW, 'Exact border triple.')
F('Z29', NG, 'Modern Era Japan shares border with China PR.', CTW, 'Exact border triple (twin of P02).')
F('Z30', DS, 'Road cycling race is a subclass of cycle race.', CTW, 'Exact subclass_of triple.')
F('Z31', EM, 'Genre of music is an instance of sociological concept.', CTW, 'Exact instance_of triple; "of" in subject slug.')
F('Z32', GF, 'Gorky Film Studio is an instance of film studios.', CTW, 'Exact instance_of triple.')
F('Z33', PIT, 'Pittsburgh LRT is an instance of a transportation authority.', CTW, 'Exact instance_of triple; Det before the object.')
F('Z34', TSK, 'Team Sky has part 2016 Team Sky season.', CTW, 'Exact has_part triple; digits in the object slug.')
F('Z35', AFG, 'Afghanistan I S of shares border with Persian State of Iran.', CTW, 'Exact border triple; subject slug ends in "of".')
F('Z36', MXF, 'Mexico City Olympic Games follows Games of the XVIII Olympiad.', CTW, 'Exact follows triple; "the" inside the object slug.')
F('Z37', GB, 'CANADA PORTAL SHARES BORDER WITH U S OF A.', CTW, 'Case-only variant of Z01.')
F('Z38', GB, 'Canada_Portal shares border with U_S_of_A.', CTW, 'Underscore-spelled slugs, allowed by the spec\'s normalisation.')
F('Z39', STE, 'Sète France is an instance of communes in France.', CTW, 'Accented spelling of the store slug s_te_france.')
F('Z40', BR, 'Human specie is an instance of common names.', CTW, 'Exact chain triple (twin of P03).')
F('Z41', CMN, 'The domestic architecture has part the box room.', CTW, 'Exact has_part triple with Det before subject and object.')
ABG = G('a_berg', ('a_berg', 'instance_of', 'surname_page'))
F('Z43', ABG, 'A Berg is an instance of surname page.', CTW, 'Exact triple; the capital A is part of the slug a_berg, not a Det.')
F('Z42', AG, '1962 Asian Games followed Third Asian Games.', CTW, 'Past tense of an event-order relation is still true (contrast L01-L04).')


def validate():
    ids = [c['id'] for c in CASES]
    assert len(ids) == len(set(ids)), 'duplicate ids'
    bad = []
    for c in CASES:
        assert set(c) == {'id', 'gathered', 'sentence', 'expect', 'class', 'why'}
        assert c['expect'] in ('U', 'F')
        assert c['gathered'], c['id']
        for t in c['gathered']:
            if tuple(t) not in TSET:
                bad.append((c['id'], t))
            if t[1] not in RELS:
                bad.append((c['id'], 'REL', t[1]))
    return bad


if __name__ == '__main__':
    bad = validate()
    if bad:
        raise SystemExit('NOT IN STORE: %r' % bad)
    out = os.path.join(HERE, 'cases.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(CASES, f, ensure_ascii=False, indent=1)
    from collections import Counter
    print('wrote', out, len(CASES))
    print(Counter(c['expect'] for c in CASES))
    for (k, e), n in sorted(Counter((c['class'], c['expect']) for c in CASES).items()):
        print('%4d  %s  %s' % (n, e, k))
