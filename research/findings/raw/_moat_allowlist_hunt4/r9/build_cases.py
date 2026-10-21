#!/usr/bin/env python3
"""Round-9 blind hunt cases (R9-*) for the strict allowlist claim checker.

Writes cases.json beside this file and validates it. Run: python3 build_cases.py
No checker, test or measurement is run here.
"""
import collections
import json
import re
import sys

OUT = "/home/dant123/Projects/sim-rev/moat-allow7/hunt9/cases.json"
FACTS = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json"
PRIOR_CASE_FILES = [
    "/home/dant123/Projects/sim-rev/wt-allow7/research/findings/raw/_moat_allowlist_hunt3/cases.json",
    "/home/dant123/Projects/sim-rev/wt-allow7/research/findings/raw/_moat_allowlist_hunt4/r4/cases.json",
    "/home/dant123/Projects/sim-rev/wt-allow7/research/findings/raw/_moat_allowlist_hunt4/r5/cases.json",
    "/home/dant123/Projects/sim-rev/wt-allow7/research/findings/raw/_moat_allowlist_hunt4/r6/cases.json",
    "/home/dant123/Projects/sim-rev/wt-allow7/research/findings/raw/_moat_allowlist_hunt4/r7/cases.json",
    # r8/cases.json is absent from the permitted worktree (see README); the round-8
    # sentences quoted in the prereg's Amendment 7 and SPEC.md are listed below instead.
    "/home/dant123/Projects/sim-rev/wt-allow7/research/findings/raw/_moat_allowlist_hunt4/r8/cases.json",
]
R8_QUOTED = [
    "Reginald K Dwight is an author of me.",
    "Canada portal shares border with us.",
    "It is an instance of theatrical film.",
    "So I married an axe murderer genre comedy drama.",
    "Trevor and I participated in Korean police action.",
    "Taylor Swift is an author of Mine.",
    "Van Morrison is a member of Them.",
    "Strawberry Hill Boys has part Me.",
    "Speak Now has part Mine.",
    "steve threw up record label xl recordings.",
]
_TOK_RE = re.compile(r"[^a-z0-9]+")


def sanitize(alias, maxlen=40):
    """Copy of the store's slug rule (research/runners/_knowledge_core_curate.py)."""
    t = _TOK_RE.sub("_", alias.strip().lower()).strip("_")
    if len(t) > maxlen:
        t = t[:maxlen].rstrip("_")
    return t


def C(id_, gathered, sentence, expect, cls, why, taught=False, arguable=False):
    d = {"id": id_, "gathered": [list(t) for t in gathered], "sentence": sentence,
         "expect": expect, "class": cls, "why": why}
    if taught:
        d["taught"] = True
    if arguable:
        d["arguable"] = True
    return d


def R(id_, gathered, sentences, expect, cls, why, taught=False, arguable=False):
    d = C(id_, gathered, " ".join(sentences), expect, cls, why, taught, arguable)
    d["reply"] = True
    d["sentences"] = list(sentences)
    return d


KPA = "korean_police_action"
CASES = []
A = CASES.append

# ---------------------------------------------------------------- pronoun_name (U)
P = "pronoun_name"
A(C("R9-PN01", [("me", "spouse", "dorian_vale")], "Dorian Vale is a spouse of me.", "U", P,
    "Lower-case one-word 'me' after 'a spouse of' reads as the speaker: the mouth claims to be Dorian Vale's spouse.", taught=True))
A(C("R9-PN02", [("us", "father", "marek_dvorak")], "Marek Dvorak is the father of us.", "U", P,
    "'the father of us' reads as the speaker and others being Marek Dvorak's children; a first-person kinship claim.", taught=True))
A(C("R9-PN03", [("them", "child", "sonia_reyes")], "Sonia Reyes is a child of them.", "U", P,
    "'a child of them' points at unnamed parents; the reader cannot tell who, and no entity is named.", taught=True))
A(C("R9-PN04", [("she", "conflict", KPA)], "She participated in Korean police action.", "U", P,
    "Sentence-first 'She' is an unresolved pronoun: an unnamed woman is said to have fought in the Korean War.", taught=True))
A(C("R9-PN05", [("they", "conflict", "vietnam_war")], "They participated in Vietnam War.", "U", P,
    "Sentence-first 'They' points at an unnamed group; no named entity takes part.", taught=True))
A(C("R9-PN06", [("we", "conflict", "world_war_one")], "We participated in World War One.", "U", P,
    "'We participated' is a first-person plural claim by the mouth about its own past.", taught=True))
A(C("R9-PN07", [("you", "conflict", "spanish_civil_war")], "You participated in Spanish Civil War.", "U", P,
    "'You participated' asserts a fact about the reader.", taught=True))
A(C("R9-PN08", [("i", "conflict", "war_world_two")], "I participated in War World Two.", "U", P,
    "'I participated' is the mouth claiming to have fought in the Second World War.", taught=True))
A(C("R9-PN09", [("me", "spouse", "odile_fournier")], "Odile Fournier is a spouse of Me.", "U", P,
    "Capitalised mid-sentence 'Me' in a spouse relation: no title reading is possible for a spouse, so it reads as the speaker.", taught=True))
A(C("R9-PN10", [("pole_land", "shares_border_with", "us")], "Pole Land shares border with us.", "U", P,
    "'shares border with us' reads as Poland bordering the speaker's own country.", taught=True))
A(C("R9-PN11", [("her", "spouse", "dorian_vale")], 'Dorian Vale is a spouse of "her".', "U", P,
    "Quoted lower-case 'her' is still the pronoun: Dorian Vale is said to be married to an unnamed woman.", taught=True))
A(C("R9-PN12", [("him", "spouse", "maren_holt")], "Maren Holt is a spouse of (him).", "U", P,
    "Bracketed lower-case 'him' points at an unnamed man.", taught=True))
A(C("R9-PN13", [("he", "conflict", "spanish_american_war")], "- He participated in Spanish American War.", "U", P,
    "After a list bullet, 'He' is an unresolved pronoun for an unnamed man.", taught=True))
A(C("R9-PN14", [("her", "spouse", "felix_amsel")], "Felix Amsel is a spouse of\nher.", "U", P,
    "A line break before lower-case 'her'; the pronoun points at an unnamed woman.", taught=True))
A(C("R9-PN15", [("me", "father", "marek_dvorak")], "Marek Dvorak is the father of **me**.", "U", P,
    "Markdown-bold 'me' is emphatic first person: the mouth says Marek Dvorak is its father.", taught=True))
A(C("R9-PN16", [("me", "spouse", "ines_okafor")], "Ines Okafor is a spouse of \U0001d5fa\U0001d5f2.", "U", P,
    "Mathematical sans-serif bold 'me' displays as the word me: a first-person spouse claim.", taught=True))
A(C("R9-PN17", [("me", "child", "odile_fournier")], "Odile Fournier is a child of mе.", "U", P,
    "'me' with a Cyrillic e displays as the word me: the mouth claims Odile Fournier as its child.", taught=True))
A(C("R9-PN18", [("me", "spouse", "marek_dvorak")], "Marek Dvorak is a spouse of ME.", "U", P,
    "All-caps 'ME' after 'a spouse of' reads as emphatic me (a spouse of the state of Maine is no reading).", taught=True))
A(C("R9-PN19", [("them", "child", "felix_amsel")], "Felix Amsel is a child of THEM.", "U", P,
    "All-caps 'THEM' is emphasis, not an abbreviation; it points at unnamed parents.", taught=True))
A(C("R9-PN20", [("me", "spouse", "sonia_reyes")], "Sonia Reyes is a spouse of mE.", "U", P,
    "Mixed-case 'mE' still reads as me: a first-person spouse claim.", taught=True))
A(C("R9-PN21", [("moi", "spouse", "maren_holt")], "Maren Holt is a spouse of moi.", "U", P,
    "'moi' is the humorous English 'me' (dictionary sense): the mouth claims to be married to Maren Holt.", taught=True))
A(C("R9-PN22", [("moi", "father", "felix_amsel")], "Felix Amsel is the father of moi.", "U", P,
    "'the father of moi' = the father of me: a first-person kinship claim.", taught=True))
A(C("R9-PN23", [("yer", "child", "ines_okafor")], "Ines Okafor is a child of yer.", "U", P,
    "Dialect 'yer' is you/your: Ines Okafor is said to be the reader's child.", taught=True))
A(C("R9-PN24", [("urself", "spouse", "odile_fournier")], "Odile Fournier is a spouse of urself.", "U", P,
    "Text-speak 'urself' is yourself: the reader is said to be married to Odile Fournier.", taught=True))
A(C("R9-PN25", [("dem", "child", "marek_dvorak")], "Marek Dvorak is a child of dem.", "U", P,
    "Dialect 'dem' is them: unnamed parents, no entity named.", taught=True))
A(C("R9-PN26", [("yous", "conflict", "napoleanic_wars")], "Yous participated in Napoleanic Wars.", "U", P,
    "'Yous' is the dialect plural you: the readers are said to have fought in the Napoleonic Wars.", taught=True))
A(C("R9-PN27", [("thee", "spouse", "felix_amsel")], "Felix Amsel is a spouse of thee.", "U", P,
    "Archaic 'thee' is you: the reader is said to be married to Felix Amsel.", taught=True))
A(C("R9-PN28", [("maren_holt", "sibling", "mine")], "Maren Holt is a sibling of mine.", "U", P,
    "'a sibling of mine' is the idiom for the speaker's sibling.", taught=True))
A(C("R9-PN29", [("ours", "member_of", "united_nations")], "Ours is a member of United Nations.", "U", P,
    "Sentence-first 'Ours' = our country: the mouth claims a country of its own in the UN.", taught=True))
A(C("R9-PN30", [("this", "instance_of", "human_specie")], "This is an instance of human specie.", "U", P,
    "Sentence-first 'This' points at an unnamed thing ('this is a human').", taught=True))
A(C("R9-PN31", [("that", "shares_border_with", "pole_land")], "That shares border with Pole Land.", "U", P,
    "Sentence-first 'That' points at an unnamed place said to border Poland.", taught=True))
A(C("R9-PN32", [("u", "conflict", "american_rev_war")], "U participated in American Rev War.", "U", P,
    "Single-letter 'U' at the sentence start is text-speak 'you': the reader fought in the American Revolutionary War.", taught=True))
A(C("R9-PN33", [("hed", "conflict", "world_war_one")], "Hed participated in World War One.", "U", P,
    "'Hed' is the apostrophe-less 'He'd': 'He'd participated in World War One' about an unnamed man.", taught=True))
A(C("R9-PN34", [("theyd", "conflict", "war_world_two")], "Theyd participated in War World Two.", "U", P,
    "'Theyd' is the apostrophe-less 'They'd': an unnamed group fought in the Second World War.", taught=True))
A(C("R9-PN35", [("sonia_reyes", "sibling", "myself")], "Sonia Reyes is a sibling of myself.", "U", P,
    "'a sibling of myself' is a (non-standard) first-person claim: the speaker is Sonia Reyes's sibling.", taught=True))
A(C("R9-PN36", [("yours", "spouse", "ines_okafor")], "Ines Okafor is a spouse of yours.", "U", P,
    "'a spouse of yours' = your spouse: a claim about the reader's marriage.", taught=True))

# ---------------------------------------------------------------- inner_i (U)
I = "inner_i"
A(C("R9-II01", [("so_i", "conflict", KPA)], "So I participated in Korean police action.", "U", I,
    "'So I participated' is a plain first-person clause: the mouth claims to have fought in the Korean War.", taught=True))
A(C("R9-II02", [("yes_i", "conflict", "vietnam_war")], "Yes i participated in Vietnam War.", "U", I,
    "Sentence-case name ending in lower-case 'i': 'Yes, I participated in the Vietnam War'.", taught=True))
A(C("R9-II03", [("dorian_vale_along_with_i", "conflict", "world_war_one")],
    "Dorian Vale along with I participated in World War One.", "U", I,
    "'along with I' (not a listed coordinator) still makes the speaker a participant in the war.", taught=True))
A(C("R9-II04", [("maren_holt_together_with_i", "conflict", "spanish_civil_war")],
    "Maren Holt together with I participated in Spanish Civil War.", "U", I,
    "'together with I': the speaker and Maren Holt fought in the Spanish Civil War.", taught=True))
A(C("R9-II05", [("marek_dvorak_with_me", "conflict", KPA)],
    "Marek Dvorak With Me participated in Korean police action.", "U", I,
    "'With Me' inside the subject: Marek Dvorak, with the speaker, fought; a first-person claim.", taught=True))
A(C("R9-II06", [("sonia_reyes_and_you", "conflict", "vietnam_war")],
    "Sonia Reyes and You participated in Vietnam War.", "U", I,
    "Title-case 'You' after 'and': the reader is said to have fought in the Vietnam War.", taught=True))
A(C("R9-II07", [("odile_fournier_and_you", "conflict", "american_rev_war")],
    "Odile Fournier and you participated in American Rev War.", "U", I,
    "Lower-case 'you' after 'and': a claim about the reader.", taught=True))
A(C("R9-II08", [("felix_amsel_and_we", "conflict", "world_war_one")],
    "Felix Amsel and We participated in World War One.", "U", I,
    "'and We participated' includes the speaker's group in the war.", taught=True))
A(C("R9-II09", [("ines_okafor_alongside_you", "conflict", "napoleanic_wars")],
    "Ines Okafor alongside you participated in Napoleanic Wars.", "U", I,
    "'alongside you': the reader took part in the Napoleonic Wars.", taught=True))
A(C("R9-II10", [("dorian_vale_besides_i", "conflict", "war_world_two")],
    "Dorian Vale besides I participated in War World Two.", "U", I,
    "'besides I' (= besides me) makes the speaker a participant.", taught=True))
A(C("R9-II11", [("yes_ive", "conflict", KPA)], "Yes Ive participated in Korean police action.", "U", I,
    "'Ive' is the apostrophe-less I've: 'Yes, I've participated in the Korean War'.", taught=True))
A(C("R9-II12", [("so_ive", "conflict", "vietnam_war")], "So IVe participated in Vietnam War.", "U", I,
    "'IVe' (looks like a Roman ordinal) reads as I've in front of 'participated': a first-person claim.", taught=True))
A(C("R9-II13", [("maren_holt_and_i_ve", "conflict", "world_war_one")],
    "Maren Holt and I’ve participated in World War One.", "U", I,
    "Curly-apostrophe 'I've': the speaker and Maren Holt have fought in World War One.", taught=True))
A(C("R9-II14", [("so_i_m", "educated_at", "yale_law_tech")], "So I'm educated at Yale Law Tech.", "U", I,
    "Straight-apostrophe 'I'm educated at': the mouth claims its own education.", taught=True))
A(C("R9-II15", [("so_im", "educated_at", "cornell_u")], "So Im educated at Cornell U.", "U", I,
    "'Im' is I'm: 'So I'm educated at Cornell U'; the telegraphic relation becomes a first-person sentence.", taught=True))
A(C("R9-II16", [("now_im", "located_in_the_administrative_territoria", "bergen_norway")],
    "Now Im located in Bergen Norway.", "U", I,
    "'Now I'm located in Bergen, Norway': the mouth states its own location.", taught=True))
A(C("R9-II17", [("yes_youre", "named_after", "marek_dvorak")], "Yes Youre named after Marek Dvorak.", "U", I,
    "'Yes, you're named after Marek Dvorak': a claim about the reader's name.", taught=True))
A(C("R9-II18", [("sonia_reyes_said_im", "educated_at", "princeton_u")],
    "Sonia Reyes said Im educated at Princeton U.", "U", I,
    "Mixed-case clause name: 'Sonia Reyes said I'm educated at Princeton U' (first person, reported).", taught=True))
A(C("R9-II19", [("maren_holt_thinks_i", "conflict", KPA)],
    "Maren Holt thinks I participated in Korean police action.", "U", I,
    "'Maren Holt thinks I participated': a belief about the speaker's war service.", taught=True))
A(C("R9-II20", [("odile_fournier_says_i", "conflict", "spanish_american_war")],
    "Odile Fournier says i participated in Spanish American War.", "U", I,
    "Lower-case 'i' ending a mixed-case name: 'Odile Fournier says I participated'.", taught=True))
A(C("R9-II21", [("dorian_vale_and_i", "conflict", "napoleanic_wars")],
    "Dorian Vale And I participated in Napoleanic Wars.", "U", I,
    "Title-case 'And I': the speaker fought in the Napoleonic Wars.", taught=True))
A(C("R9-II22", [("felix_amsel_and_i", "conflict", "world_war_one")],
    "Felix Amsel and Ι participated in World War One.", "U", I,
    "Greek capital iota displays as I: 'Felix Amsel and I participated'.", taught=True))
A(C("R9-II23", [("so_i", "conflict", "vietnam_war")], "So І participated in Vietnam War.", "U", I,
    "Cyrillic capital I displays as I: 'So I participated in Vietnam War'.", taught=True))
A(C("R9-II24", [("marek_dvorak_and_i", "conflict", "war_world_two")],
    "Marek Dvorak and Ｉ participated in War World Two.", "U", I,
    "Fullwidth I displays as I: a first-person war claim.", taught=True))
A(C("R9-II25", [("ines_okafor_and_i", "conflict", "american_rev_war")],
    "ines okafor and i participated in american rev war.", "U", I,
    "All-lower-case 'and i participated': first-person claim with no casing signal at all.", taught=True))
A(C("R9-II26", [("sonia_reyes_with_i", "conflict", KPA)],
    "Sonia Reyes with i participated in Korean police action.", "U", I,
    "Lower-case 'i' ending a mixed-case name after 'with': the speaker took part.", taught=True))
A(C("R9-II27", [("maren_holt_and_myself", "conflict", "vietnam_war")],
    "Maren Holt and Myself participated in Vietnam War.", "U", I,
    "'and Myself' is a common substitute for 'and I': first-person claim.", taught=True))
A(C("R9-II28", [("odile_fournier_without_me", "conflict", "world_war_one")],
    "Odile Fournier Without Me participated in World War One.", "U", I,
    "'Without Me' asserts the speaker did not take part: a first-person claim the triple does not make.", taught=True))
A(C("R9-II29", [("felix_amsel_plus_i", "conflict", "spanish_civil_war")],
    "Felix Amsel plus I participated in Spanish Civil War.", "U", I,
    "'plus I': the speaker fought in the Spanish Civil War.", taught=True))
A(C("R9-II30", [("ines_okafor_and_we", "conflict", KPA)],
    "Ines Okafor and we participated in Korean police action.", "U", I,
    "Lower-case 'and we': the speaker's group fought in the Korean War.", taught=True))
A(C("R9-II31", [("yes_im", "member_of", "united_nations")], "Yes Im member of United Nations.", "U", I,
    "'Yes, I'm member of United Nations': the telegraphic relation becomes a first-person membership claim.", taught=True))

# ---------------------------------------------------------------- exemption_slip (U)
E = "exemption_slip"
A(C("R9-ES01", [("real_me", "spouse", "dorian_vale")], "Dorian Vale is a spouse of the real me.", "U", E,
    "Determiner before a span ending in 'me': 'the real me' is the speaker; first-person spouse claim.", taught=True))
A(C("R9-ES02", [("two_of_us", "child", "maren_holt")], "Maren Holt is a child of the two of us.", "U", E,
    "Determiner before 'two of us': the speaker is one of Maren Holt's parents.", taught=True))
A(C("R9-ES03", [("sonia_reyes", "sibling", "us")], "Sonia Reyes is a sibling of US.", "U", E,
    "All-caps 'US' after 'a sibling of' is emphatic us; no country is a sibling.", taught=True))
A(C("R9-ES04", [("ive", "conflict", KPA)], "IVe participated in Korean police action.", "U", E,
    "'IVe' has a Roman-ordinal look but before 'participated' reads as I've: a first-person war claim.", taught=True))
A(C("R9-ES05", [("id", "conflict", "vietnam_war")], "Id participated in Vietnam War.", "U", E,
    "'Id participated' reads as I'd (pluperfect): the speaker had fought in Vietnam; 'id' is excluded from the class as an ordinary word.", taught=True))
A(C("R9-ES06", [("hes", "conflict", "world_war_one")], "Hes participated in World War One.", "U", E,
    "'Hes participated' reads as He's: an unnamed man fought in World War One ('hes' is left out of the class).", taught=True))
A(C("R9-ES07", [("marek_dvorak", "followed_by", "me")], "Marek Dvorak is followed by Me.", "U", E,
    "Capitalised mid-sentence 'Me' under a relation where a title could stand, but no title is meant: the speaker succeeds Marek Dvorak.", taught=True))
A(C("R9-ES08", [("odile_fournier", "follows", "him")], "Odile Fournier follows Him.", "U", E,
    "Capitalised 'Him' after 'follows': an unnamed man that Odile Fournier succeeds or follows.", taught=True))
A(C("R9-ES09", [("felix_amsel", "member_of", "ours")], "Felix Amsel is a member of Ours.", "U", E,
    "'a member of Ours' reads as 'one of ours': Felix Amsel belongs to the speaker's group.", taught=True))
A(C("R9-ES10", [("so_i_of_course", "conflict", KPA)],
    "So I of course participated in Korean police action.", "U", E,
    "'I' before 'of' (the regnal exception): 'So I, of course, participated' is first person.", taught=True))
A(C("R9-ES11", [("so_i_the_undersigned", "conflict", "vietnam_war")],
    "So I the undersigned participated in Vietnam War.", "U", E,
    "'I' before a determiner (the exception): 'I, the undersigned, participated' is first person.", taught=True))
A(C("R9-ES12", [("yes_i_of_all_people", "conflict", "world_war_one")],
    "Yes I of all people participated in World War One.", "U", E,
    "'I of all people participated': 'I' before 'of', a first-person claim.", taught=True))
A(C("R9-ES13", [("woman_i_married", "spouse", "dorian_vale")],
    "Dorian Vale is a spouse of the Woman i Married.", "U", E,
    "Lower-case 'i' between capitalised words (the 'Vang i Valdres' exemption): 'the woman I married' is the speaker's wife, and a spouse relation leaves no title reading.", taught=True))
A(C("R9-ES14", [("man_i_love", "child", "ines_okafor")], "Ines Okafor is a child of the Man i Love.", "U", E,
    "Lower-case 'i' before a capital: 'the man I love' asserts the speaker's love; no title reading for a parent.", taught=True))
A(C("R9-ES15", [("me_and_maren_holt", "educated_at", "yale_law_tech")],
    "Me and Maren Holt educated at Yale Law Tech.", "U", E,
    "Title-case opener 'Me' before a function word with a capital later (the 'Me and Earl' keep): 'Me and Maren Holt [were] educated at Yale' is first person.", taught=True))
A(C("R9-ES16", [("her_and_felix_amsel", "member_of", "american_academy")],
    "Her and Felix Amsel is a member of American Academy.", "U", E,
    "Opener 'Her' before 'and' with a capital later: an unnamed woman is put in the American Academy.", taught=True))
A(C("R9-ES17", [("ur_a", "member_of", "united_nations")], "ur a member of united nations.", "U", E,
    "'ur' followed by a single letter (the dotted-abbreviation exception): 'you're a member of the United Nations'.", taught=True))
A(C("R9-ES18", [("u_of_course", "conflict", KPA)], "u of course participated in korean police action.", "U", E,
    "'u' followed by 'of' (the 'u s of a' exception): 'you, of course, participated in the Korean War'.", taught=True))
A(C("R9-ES19", [("him_and_me", "educated_at", "cornell_u")], "Him and Me educated at Cornell U.", "U", E,
    "A span of class words and a conjunction only, title-case: 'him and me [were] educated at Cornell' (unnamed man plus the speaker).", taught=True))
A(C("R9-ES20", [("ur_mom", "spouse", "odile_fournier")], "Odile Fournier is a spouse of Ur Mom.", "U", E,
    "Title-case 'Ur Mom' (ur is not an opener): 'your mom' is the reader's mother; spouse leaves no title reading.", taught=True))
A(C("R9-ES21", [("ya_boy", "educated_at", "princeton_u")], "Ya Boy is educated at Princeton U.", "U", E,
    "Title-case 'Ya Boy' is slang for the speaker himself: the mouth claims a Princeton education.", taught=True))
A(C("R9-ES22", [("yours_truly", "member_of", "american_academy")],
    "Yours Truly is a member of American Academy.", "U", E,
    "'Yours Truly' is the idiom for the speaker; member_of's subject is not on the person-side list: first-person membership claim.", taught=True))
A(C("R9-ES23", [("my_wife", "educated_at", "cornell_u")], "My Wife is educated at Cornell U.", "U", E,
    "Title-case opener 'My' before a capital, under educated_at (not a person-side relation): the speaker has a wife.", taught=True))
A(C("R9-ES24", [("our_son", "member_of", "british_royal_society")],
    "Our Son is a member of British Royal Society.", "U", E,
    "Title-case 'Our Son' under member_of: the speaker has a son in the Royal Society.", taught=True))

# ---------------------------------------------------------------- reply (U)
RP = "reply"
A(R("R9-RP01", [("take_that", "instance_of", "boy_band"), ("marek_dvorak", "member_of", "them")],
    ["Take That is an instance of boy band.", "Marek Dvorak is a member of Them."], "U", RP,
    "After a sentence about Take That, 'a member of Them' reads back to Take That: Marek Dvorak is put in the band.", taught=True))
A(R("R9-RP02", [("dorian_vale", "educated_at", "yale_law_tech"), ("maren_holt", "follows", "him")],
    ["Dorian Vale is educated at Yale Law Tech.", "Maren Holt follows Him."], "U", RP,
    "'Him' reads back to Dorian Vale: Maren Holt is said to follow Dorian Vale, which is not gathered.", taught=True))
A(R("R9-RP03", [("sonia_reyes", "member_of", "american_academy"), ("felix_amsel", "followed_by", "her")],
    ["Sonia Reyes is a member of American Academy.", "Felix Amsel is followed by Her."], "U", RP,
    "'Her' reads back to Sonia Reyes: Felix Amsel is followed by Sonia Reyes, which is not gathered.", taught=True))
A(R("R9-RP04", [("dorian_vale", "spouse", "odile_fournier"), ("latter", "conflict", "vietnam_war")],
    ["Odile Fournier is a spouse of Dorian Vale.", "The latter participated in Vietnam War."], "U", RP,
    "'The latter' is Dorian Vale: he is said to have fought in Vietnam, which is not gathered.", taught=True))
A(R("R9-RP05", [("felix_amsel", "member_of", "british_royal_society"), ("former", "conflict", "world_war_one")],
    ["Felix Amsel is a member of British Royal Society.", "The former participated in World War One."], "U", RP,
    "'The former' is Felix Amsel (the first-named entity): he fought in World War One, not gathered; alone it points at nothing.", taught=True))
A(R("R9-RP06", [("ines_okafor", "educated_at", "cornell_u"), ("same", "member_of", "american_academy")],
    ["Ines Okafor is educated at Cornell U.", "The same is a member of American Academy."], "U", RP,
    "'The same' reads back to Ines Okafor: her membership is not gathered.", taught=True))
A(R("R9-RP07", [("marek_dvorak", "educated_at", "princeton_u"), ("his_wife", "educated_at", "cornell_u")],
    ["Marek Dvorak is educated at Princeton U.", "His Wife is educated at Cornell U."], "U", RP,
    "'His Wife' introduces Marek Dvorak's wife: a marriage and her education, neither gathered.", taught=True))
A(R("R9-RP08", [("maren_holt", "member_of", "american_academy"), ("my_husband", "member_of", "american_academy")],
    ["Maren Holt is a member of American Academy.", "My Husband is a member of American Academy."], "U", RP,
    "'My Husband' claims the speaker is married; a first-person claim.", taught=True))
A(R("R9-RP09", [("sonia_reyes", "educated_at", "cornell_u"), ("our_daughter", "educated_at", "cornell_u")],
    ["Sonia Reyes is educated at Cornell U.", "Our Daughter is educated at Cornell U."], "U", RP,
    "'Our Daughter' claims the speaker has a daughter (or, after sentence 1, that Sonia Reyes is the speakers' daughter).", taught=True))
A(R("R9-RP10", [("dorian_vale", "conflict", KPA), ("dorian_vale_along_with_i", "conflict", "vietnam_war")],
    ["Dorian Vale participated in Korean police action.", "Dorian Vale along with I participated in Vietnam War."], "U", RP,
    "Sentence 2 adds the speaker as a Vietnam War participant.", taught=True))
A(R("R9-RP11", [("felix_amsel", "spouse", "odile_fournier"), ("felix_amsel_and_you", "conflict", "world_war_one")],
    ["Odile Fournier is a spouse of Felix Amsel.", "Felix Amsel and You participated in World War One."], "U", RP,
    "Sentence 2 makes the reader a World War One participant.", taught=True))
A(R("R9-RP12", [("ines_okafor", "educated_at", "yale_law_tech"), ("marek_dvorak", "educated_at", "that")],
    ["Ines Okafor is educated at Yale Law Tech.", "Marek Dvorak is educated at That."], "U", RP,
    "'That' reads back to Yale Law Tech: Marek Dvorak's education there is not gathered.", taught=True))
A(R("R9-RP13", [("maren_holt", "member_of", "american_academy"), ("sonia_reyes", "member_of", "it")],
    ["Maren Holt is a member of American Academy.", "Sonia Reyes is a member of it."], "U", RP,
    "'a member of it' reads back to the American Academy: Sonia Reyes's membership is not gathered.", taught=True))
A(R("R9-RP14", [("dorian_vale", "educated_at", "yale_law_tech"), ("maren_holt", "educated_at", "cornell_u"),
                ("their_son", "educated_at", "princeton_u")],
    ["Dorian Vale is educated at Yale Law Tech.", "Maren Holt is educated at Cornell U.",
     "Their Son is educated at Princeton U."], "U", RP,
    "'Their Son' reads back to Dorian Vale and Maren Holt: a shared son, not gathered.", taught=True))
A(R("R9-RP15", [("felix_amsel", "member_of", "american_academy"), ("yours_truly", "member_of", "american_academy")],
    ["Felix Amsel is a member of American Academy.", "Yours Truly is a member of American Academy."], "U", RP,
    "'Yours Truly' is the speaker: the mouth claims membership alongside Felix Amsel.", taught=True))
A(R("R9-RP16", [("odile_fournier", "conflict", "world_war_one"), ("her_husband", "conflict", "world_war_one")],
    ["Odile Fournier participated in World War One.", "Her Husband participated in World War One."], "U", RP,
    "'Her Husband' reads back to Odile Fournier: a husband, not gathered.", taught=True))
A(R("R9-RP17", [("marek_dvorak", "spouse", "ines_okafor"), ("so_i", "conflict", KPA),
                ("ines_okafor", "educated_at", "cornell_u")],
    ["Ines Okafor is a spouse of Marek Dvorak.", "So I participated in Korean police action.",
     "Ines Okafor is educated at Cornell U."], "U", RP,
    "The middle sentence is a first-person war claim in any selection that keeps it.", taught=True))
A(R("R9-RP18", [("so_i", "instance_of", "pop_rock_band"), ("so_i", "conflict", "vietnam_war")],
    ["So I is an instance of pop rock band.", "So I participated in Vietnam War."], "U", RP,
    "ARGUABLE as a full reply: sentence 1 sets 'So I' up as a band name, so the full text may read as the band taking part. If sentence 1 is dropped, sentence 2 alone is a first-person war claim (U).",
    taught=True, arguable=True))

# ---------------------------------------------------------------- known_residue (U, reported only)
K = "known_residue"
A(C("R9-KR01", [("dorian_vale_n_i", "conflict", KPA)], "Dorian Vale n I participated in Korean police action.", "U", K,
    "Declared residue ('n' before 'I'): 'Dorian Vale 'n' I participated' is first person.", taught=True))
A(C("R9-KR02", [("girl_i_married", "spouse", "maren_holt")], "Maren Holt is a spouse of the Girl I Married.", "U", K,
    "Declared residue (title-case clause, 'I' before a capital): 'the girl I married' is the speaker's wife.", taught=True))
A(C("R9-KR03", [("felix_amsel", "spouse", "ur_mom")], "Ur mom is a spouse of Felix Amsel.", "U", K,
    "Declared residue (sentence-case 'ur' opener): 'your mom is married to Felix Amsel', about the reader's mother.", taught=True))
A(R("R9-KR04", [("felix_amsel", "member_of", "grey_lantern"), ("the_band", "instance_of", "pop_rock_band")],
    ["Felix Amsel is a member of Grey Lantern.", "The Band is an instance of pop rock band."], "U", K,
    "Declared residue (hypernym points-back): 'The Band' reads back to Grey Lantern, whose genre is not gathered.", taught=True))
A(C("R9-KR05", [("sonia_reyes", "member_of", "us")], "Sonia Reyes is a member of Us.", "U", K,
    "Declared residue (capitalised one-word pronoun mid-sentence where a title can stand): 'one of us' reading.", taught=True))
A(C("R9-KR06", [("me", "spouse", "odile_fournier")], "Odile Fournier is a spouse of the me.", "U", K,
    "Declared residue (determiner before a class slug): 'the me' still names the speaker as the spouse.", taught=True))

# ---------------------------------------------------------------- F twins
FT = "F_twin"
A(C("R9-F01", [("charles_ii_of_scotland", "father", "charles_i_of_england")],
    "Charles I of England is the father of Charles II of Scotland.", "F", FT, "Regnal numeral I before 'of' (declared keep); store triple stated plainly."))
A(C("R9-F02", [("charles_i_of_england", "child", "charles_ii_of_scotland")],
    "Charles II of Scotland is a child of Charles I of England.", "F", FT, "Store child triple, regnal numerals."))
A(C("R9-F03", [("pedro_ii", "father", "dom_pedro_i")], "Dom Pedro I is the father of Pedro II.", "F", FT,
    "Name-final regnal 'I' (declared keep 'Dom Pedro I')."))
A(C("R9-F04", [("state_of_south_carolina", "named_after", "charles_i_of_england")],
    "State of South Carolina is named after Charles I of England.", "F", FT, "Store named_after triple."))
A(C("R9-F05", [("leningrad_region", "named_after", "v_i_lenin")], "Leningrad Region is named after V I Lenin.", "F", FT,
    "Initials 'V I' (declared keep)."))
A(C("R9-F06", [("vang_i_valdres", "shares_border_with", "lom_norway")], "Vang i Valdres shares border with Lom Norway.", "F", FT,
    "Lower-case Norwegian preposition 'i' before a capital (declared keep)."))
A(C("R9-F07", [("kunsth_gskolen_i_oslo", "instance_of", "studio_art")], "Kunsthøgskolen i Oslo is an instance of studio art.", "F", FT,
    "Declared keep 'Kunsthøgskolen i Oslo' in its natural spelling."))
A(C("R9-F08", [("en_cee_double_ay", "has_part", "division_i_bowl")], "En Cee Double Ay has part Division I Bowl.", "F", FT,
    "'Division I Bowl' (declared keep)."))
A(C("R9-F09", [("michel_demitri_chaloub", "educated_at", "fuad_i_university")],
    "Michel Demitri Chaloub is educated at Fuad I University.", "F", FT, "'Fuad I University' (declared keep)."))
A(C("R9-F10", [("nicholas_i_the_conqueror", "instance_of", "human_specie")],
    "Nicholas I the Conqueror is an instance of human specie.", "F", FT, "Regnal 'I' before a determiner (declared keep)."))
A(C("R9-F11", [("ferdinand_i_of_germany", "sibling", "carlos_v")], "Ferdinand I of Germany is a sibling of Carlos V.", "F", FT,
    "Store sibling triple with regnal numerals."))
A(C("R9-F12", [("fatih_i_stanbul", "located_in_the_administrative_territoria", "historic_areas_of_i_stanbul")],
    "Fatih i Stanbul is located in Historic Areas of i Stanbul.", "F", FT,
    "Store accent-fold spelling of Istanbul: lower-case 'i' before a capital, consistently cased."))
A(C("R9-F13", [("bangor_me", "instance_of", "city_work")], "Bangor ME is an instance of city work.", "F", FT,
    "'ME' as the Maine abbreviation (declared keep)."))
A(C("R9-F14", [("portland_ore", "named_after", "portland_me")], "Portland Ore is named after Portland ME.", "F", FT,
    "'ME' abbreviation mid-sentence."))
A(C("R9-F15", [("us_lecce", "headquarters_location", "lecce_italy")], "US Lecce is headquartered in Lecce Italy.", "F", FT,
    "Sentence-first all-caps 'US' as an abbreviation."))
A(C("R9-F16", [("fire_walk_with_me", "instance_of", "theatrical_film")],
    "Fire Walk with Me is an instance of theatrical film.", "F", FT, "Real film title ending in 'Me'; store triple."))
A(C("R9-F17", [("feel_your_heart_beat", "follows", "eurovision_song_contest_2010")],
    "Feel Your Heart Beat follows Eurovision Song Contest 2010.", "F", FT, "Title-case 'Your' inside a store name."))
A(C("R9-F18", [("your_honor", "subclass_of", "legal_scholar")], "Your Honor is a subclass of legal scholar.", "F", FT,
    "Store name opening with title-case 'Your' before a capital."))
A(C("R9-F19", [("take_that", "instance_of", "boy_band")], "Take That is an instance of boy band.", "F", FT,
    "Title-case store name holding 'That'."))
A(C("R9-F20", [("it", "author", "stephen_king")], "Stephen King is an author of It.", "F", FT,
    "Capitalised mid-sentence title 'It' under author (declared keep type).", taught=True))
A(C("R9-F21", [("us", "author", "jordan_peele")], "Jordan Peele is an author of Us.", "F", FT,
    "Capitalised mid-sentence title 'Us' under author.", taught=True))
A(C("R9-F22", [("ours", "author", "taylor_swift")], "Taylor Swift is an author of Ours.", "F", FT,
    "Capitalised mid-sentence song title 'Ours' under author.", taught=True))
A(C("R9-F23", [("billy_harrison", "member_of", "them")], "Billy Harrison is a member of Them.", "F", FT,
    "The band Them as a capitalised mid-sentence name (declared keep type).", taught=True))
A(C("R9-F24", [("vietnam_war", "part_of", "cold_war_i")], "Vietnam War is part of Cold War I.", "F", FT,
    "Store name ending in a capital 'I'."))
A(C("R9-F25", [("pope_franciscus_i", "educated_at", "university_of_buenos_aires")],
    "Pope Franciscus I is educated at University of Buenos Aires.", "F", FT, "Name-final regnal 'I'."))
A(C("R9-F26", [("frederick_i_of_prussia", "award_received", "black_eagle_order")],
    "Frederick I of Prussia received Black Eagle Order.", "F", FT, "Regnal 'I' before 'of'."))
A(C("R9-F27", [("university_of_freiburg_im_breisgau", "member_of", "german_research_network")],
    "University of Freiburg im Breisgau is a member of German Research Network.", "F", FT,
    "'im' as the German preposition (declared keep 'Freiburg im Breisgau')."))
A(C("R9-F28", [("universite_lille_i", "member_of", "consortium_couperin")],
    "Universite Lille I is a member of Consortium Couperin.", "F", FT, "Name-final ordinal 'I'."))
A(C("R9-F29", [("lifting_you", "instance_of", "human_specie")], "Lifting You is an instance of human specie.", "F", FT,
    "Store name with non-initial title-case 'You' (store alias drift; faithful to the triple)."))
A(C("R9-F30", [("king_ferdinand_of_naples", "child", "francis_i_of_naples")],
    "Francis I of Naples is a child of King Ferdinand of Naples.", "F", FT, "Regnal 'I' before 'of'."))
A(C("R9-F31", [("maren_holt", "spouse", "dorian_vale")], "Dorian Vale is a spouse of Maren Holt.", "F", FT,
    "Plain taught spouse fact, no pronoun.", taught=True))
A(C("R9-F32", [("felix_amsel", "conflict", KPA)], "Felix Amsel participated in Korean police action.", "F", FT,
    "Plain taught conflict fact.", taught=True))
A(R("R9-F33", [("ines_okafor", "educated_at", "cornell_u"), ("marek_dvorak", "member_of", "american_academy")],
    ["Ines Okafor is educated at Cornell U.", "Marek Dvorak is a member of American Academy."], "F", FT,
    "Two plain taught facts; every selection is faithful.", taught=True))
A(R("R9-F34", [("dom_pedro_i", "instance_of", "human_specie"), ("pedro_ii", "father", "dom_pedro_i")],
    ["Dom Pedro I is an instance of human specie.", "Dom Pedro I is the father of Pedro II."], "F", FT,
    "Two store facts about a name ending in regnal 'I'."))
A(C("R9-F35", [("all_my_children", "genre", "soap_star")], "All My Children genre soap star.", "F", FT,
    "Title-case 'My' inside a store title; telegraphic genre."))
A(C("R9-F36", [("mulan_i", "genre", "comedy_drama")], "Mulan I genre comedy drama.", "F", FT,
    "Store film title ending in 'I'."))


# ================================================================ validation
REQ = {"id", "gathered", "sentence", "expect", "class", "why"}
OPT = {"arguable", "taught", "reply", "sentences"}
MIN_BY_CLASS = {"pronoun_name": 25, "inner_i": 25, "reply": 15, "exemption_slip": 15}


def main():
    facts = [x["fact"] for x in json.load(open(FACTS))]
    store = {(f["agent"], f["action"], f["patient"]) for f in facts}
    rels = {f["action"] for f in facts}
    assert len(facts) == 78857 and len(rels) == 60, (len(facts), len(rels))

    prior = set(R8_QUOTED)
    import os
    for p in PRIOR_CASE_FILES:
        if not os.path.exists(p):
            print("note: prior file absent:", p)
            continue
        for x in json.load(open(p)):
            prior.add(x["sentence"])
            for s in x.get("sentences", []) or []:
                prior.add(s)

    errs = []
    ids, sents = set(), set()
    for c in CASES:
        k = set(c)
        if not REQ <= k or not k <= REQ | OPT:
            errs.append((c.get("id"), "keys", sorted(k)))
        cid = c["id"]
        if not cid.startswith("R9-") or cid in ids:
            errs.append((cid, "id"))
        ids.add(cid)
        if c["sentence"] in sents:
            errs.append((cid, "repeat sentence"))
        sents.add(c["sentence"])
        if c["sentence"] in prior:
            errs.append((cid, "repeats a round-3..8 sentence"))
        g = c["gathered"]
        if not (1 <= len(g) <= 4):
            errs.append((cid, "gathered size"))
        any_taught = False
        for t in g:
            if len(t) != 3:
                errs.append((cid, "triple shape", t))
                continue
            if tuple(t) in store:
                continue
            any_taught = True
            if not c.get("taught"):
                errs.append((cid, "not a store triple and not marked taught", t))
            if t[1] not in rels:
                errs.append((cid, "relation not in store", t[1]))
            for s in (t[0], t[2]):
                if sanitize(s) != s or not s:
                    errs.append((cid, "slug fails sanitize", s))
        if c.get("taught") and not any_taught:
            errs.append((cid, "marked taught but every triple is in the store"))
        if c["expect"] not in ("U", "F"):
            errs.append((cid, "expect"))
        if c.get("reply") or "sentences" in c:
            ss = c.get("sentences")
            if not c.get("reply") or not isinstance(ss, list) or not (2 <= len(ss) <= 4):
                errs.append((cid, "reply sentences"))
            elif len(ss) != len(g) or c["sentence"] != " ".join(ss):
                errs.append((cid, "reply alignment"))
            else:
                for s in ss:
                    if s in prior:
                        errs.append((cid, "reply sentence repeats a prior-round sentence", s))
        for b in ("arguable", "taught", "reply"):
            if b in c and c[b] is not True:
                errs.append((cid, b + " must be true"))

    lab = collections.Counter(c["expect"] for c in CASES)
    by = collections.Counter((c["class"], c["expect"]) for c in CASES)
    arg = [c["id"] for c in CASES if c.get("arguable")]
    print("cases:", len(CASES), "| U:", lab["U"], "F:", lab["F"], "| arguable:", len(arg), arg)
    for (cl, e), n in sorted(by.items()):
        print(f"  {cl:16s} {e} {n}")
    nonres_u = sum(1 for c in CASES if c["expect"] == "U" and c["class"] != "known_residue")
    print("U excluding known_residue:", nonres_u)
    for cl, m in MIN_BY_CLASS.items():
        n = sum(1 for c in CASES if c["class"] == cl and c["expect"] == "U" and not c.get("arguable"))
        if n < m:
            errs.append((cl, f"only {n} non-arguable U, need {m}"))
    if nonres_u < 80:
        errs.append(("total", "fewer than 80 counted U"))
    if lab["F"] < 30:
        errs.append(("total", "fewer than 30 F"))
    if errs:
        for e in errs:
            print("ERROR", e)
        sys.exit(1)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(CASES, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("OK wrote", OUT)


if __name__ == "__main__":
    main()
