#!/usr/bin/env python3
"""Round-8 blind hunt: generator + validator for cases.json.

Writes cases.json beside this file from the CASES list below, then validates:
  * every gathered triple is an exact triple of the store facts.json (except class `taught`, whose invented slugs must
    follow the store's sanitize rule and whose relations must be among the store's 60);
  * 1-4 triples per case, the exact key set, ids prefixed R8- and unique, sentences unique;
  * reply cases carry 2-4 sentences, one per gathered triple, and `sentence` is their single-space join;
  * no sentence (and no reply part) is identical to any round-3 to round-7 sentence;
  * the per-class minimums of the round-8 brief.
It prints counts by class and label. It runs no checker. Run: python3 build_cases.py
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import research.runners  # noqa: F401  (provenance door: the module run writes the cases.json sidecar; the only line added to the skeptic's file)

HERE = Path(__file__).resolve().parent
OUT = HERE / "cases.json"
FACTS = Path("/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json")
PRIOR_ROOT = Path("/home/dant123/Projects/sim-rev/wt-allow6/research/findings/raw")
PRIOR_FILES = [
    PRIOR_ROOT / "_moat_allowlist_hunt3" / "cases.json",
    PRIOR_ROOT / "_moat_allowlist_hunt4" / "r4" / "cases.json",
    PRIOR_ROOT / "_moat_allowlist_hunt4" / "r5" / "cases.json",
    PRIOR_ROOT / "_moat_allowlist_hunt4" / "r6" / "cases.json",
    PRIOR_ROOT / "_moat_allowlist_hunt4" / "r7" / "cases.json",
]

# Display spaces (render like U+0020).
NB = " "   # no-break space
FG = " "   # figure space
TH = " "   # thin space
NN = " "   # narrow no-break space

_TOK_RE = re.compile(r"[^a-z0-9]+")


def sanitize(alias, maxlen=40):
    """Local copy of the store's slug rule (research/runners/_knowledge_core_curate.py)."""
    t = _TOK_RE.sub("_", alias.strip().lower()).strip("_")
    if len(t) > maxlen:
        t = t[:maxlen].rstrip("_")
    return t


CASES = []


def S(cid, cls, gathered, sentence, expect, why, arguable=False):
    c = {"id": "R8-" + cid, "gathered": [list(t) for t in gathered], "sentence": sentence,
         "expect": expect, "class": cls, "why": why}
    if arguable:
        c["arguable"] = True
    CASES.append(c)


def R(cid, cls, gathered, sentences, expect, why, arguable=False):
    c = {"id": "R8-" + cid, "gathered": [list(t) for t in gathered], "sentence": " ".join(sentences),
         "expect": expect, "class": cls, "why": why, "reply": True, "sentences": list(sentences)}
    if arguable:
        c["arguable"] = True
    CASES.append(c)


# ---------------------------------------------------------------------------------------------------------------
# reply_joined: multi-sentence replies run as one reply (joined check, drop-then-prefix selection)
# ---------------------------------------------------------------------------------------------------------------
RJ = "reply_joined"
R("RJ01", RJ, [("peter_fonda", "father", "henry_fonda"), ("jayne_seymour_fonda", "father", "henry_fonda")],
  ["Henry Fonda is the father of Peter Fonda.", "Jayne Seymour Fonda is the father of Henry Fonda."], "U",
  "sentence 2 renders (jayne_seymour_fonda, father, henry_fonda) with the father phrase reversed: it says the "
  "daughter is Henry Fonda's father")
R("RJ02", RJ, [("jeff_bridges", "father", "lloyd_bridges"), ("jeff_bridges", "sibling", "beau_bridges")],
  ["Lloyd Bridges is the father of Jeff Bridges.", "So is Beau Bridges."], "U",
  "the elliptical second sentence reads 'Beau Bridges is the father of Jeff Bridges too'; Beau is his sibling")
R("RJ03", RJ, [("henry_fonda", "child", "jayne_seymour_fonda"), ("peter_fonda", "father", "henry_fonda"),
               ("peter_fonda", "sibling", "jayne_seymour_fonda")],
  ["Jayne Seymour Fonda is a child of Henry Fonda.", "Henry Fonda is the father of Peter Fonda.",
   "Peter Fonda is her only sibling."], "U",
  "sentence 3 adds 'only' (no other siblings) and a pronoun; the gathered set says nothing about other siblings")
R("RJ04", RJ, [("ringo_star", "instrument", "drum_kit"), ("george_harrison", "instrument", "history_of_the_guitar")],
  ["Ringo Star plays drum kit.", "George Harrison does not."], "U",
  "sentence 2 renders George Harrison's instrument triple as a negation of playing the drum kit, which nothing "
  "gathered states")
R("RJ05", RJ, [("w_churchill", "conflict", "world_war_one"), ("charles_degaulle", "conflict", "war_world_two")],
  ["W Churchill participated in world war one.",
   "Charles Degaulle participated in war world two, but not in world war one."], "U",
  "sentence 2 adds a denial (not in world war one) that no gathered triple supports")
R("RJ06", RJ, [("genesis_band", "has_part", "phil_colins"), ("phil_colins", "member_of", "genesis_band")],
  ["Genesis band has part Phil Colins.", "Phil Colins is no longer a member of Genesis band."], "U",
  "sentence 2 says the membership ended; the gathered member_of triple states it plainly")
R("RJ07", RJ, [("october", "follows", "mean_fomhair"), ("the_month_of_november", "follows", "october"),
               ("xber", "follows", "the_month_of_november")],
  ["October follows Mean Fomhair.", "The month of November follows October.",
   "Xber follows the month of November, and nothing follows Xber."], "U",
  "sentence 3 adds 'nothing follows Xber', false against the store and not gathered")
R("RJ08", RJ, [("october", "followed_by", "the_month_of_november"), ("the_month_of_november", "followed_by", "xber")],
  ["October is followed by the month of November.", "Xber is followed by the month of November."], "U",
  "sentence 2 reverses followed_by: it puts Xber before November")
R("RJ09", RJ, [("mean_fomhair", "follows", "aug"), ("october", "follows", "mean_fomhair")],
  ["Mean Fomhair follows Aug.", "Perhaps October follows Mean Fomhair."], "U",
  "sentence 2 is hedged with a leading 'Perhaps' inside a composer reply, where it is no declared non-claim")
R("RJ10", RJ, [("united_kingom", "shares_border_with", "franzoesische_republik"),
               ("franzoesische_republik", "member_of", "united_nations_security_council")],
  ["United Kingom shares border with Franzoesische Republik.",
   "My guess is that Franzoesische Republik is a member of United Nations Security Council."], "U",
  "sentence 2 is a first-person guess frame: it hedges a gathered fact and makes a first-person claim")
R("RJ11", RJ, [("henry_fonda", "conflict", "war_world_two"), ("he_s_dead_jim", "occupation", "doctor_behaviour")],
  ["Henry Fonda participated in war world two.", "He s dead Jim occupation doctor behaviour."], "U",
  "the store name he_s_dead_jim, displayed after a sentence about Henry Fonda, reads 'He's dead, Jim': it asserts "
  "that Henry Fonda is dead, which is not gathered")
R("RJ12", RJ, [("w_churchill", "conflict", "world_war_one"), ("he_s_dead_jim", "given_name", "l_n_rd")],
  ["W Churchill participated in world war one.", "He s dead Jim given name L N Rd."], "U",
  "as RJ11 with W Churchill as the antecedent of 'He s dead'; arguable because the telegraphic tail ('given name "
  "L N Rd') may make a reader take the whole run as a garbled name", arguable=True)
R("RJ13", RJ, [("canada_portal", "member_of", "united_nations"), ("not_wanted", "instance_of", "human_specie")],
  ["Canada portal is a member of the United Nations.", "Not wanted is an instance of human specie."], "U",
  "the store name 'Not wanted' opening the next sentence reads as a comment on the membership just stated (not "
  "wanted); arguable because a reader may take it as a garbled name",
  arguable=True)
R("RJ14", RJ, [("trevor_horn", "member_of", "yes_band"), ("trevor_rabin", "part_of", "yes_band"),
               ("yes_band", "genre", "symphonic_rock")],
  ["Trevor Horn is a member of Yes band.", "Trevor Rabin left Yes band.", "Yes band genre symphonic rock."], "U",
  "sentence 2 says Trevor Rabin left the band; the gathered triple states he is part of it")
R("RJ15", RJ, [("the_rolling_stones", "has_part", "michael_philip_jagger"),
               ("keith_richards", "member_of", "the_rolling_stones")],
  ["The Rolling Stones has part Michael Philip Jagger.",
   "Keith Richards is a member of the Rolling Stones as well, and the two founded it."], "U",
  "sentence 2 adds that the two founded the band, not gathered")
R("RJ16", RJ, [("john_huston", "father", "walter_thomas_huston"), ("huston_anjelica", "father", "john_huston")],
  ["Walter Thomas Huston is the father of John Huston.", "He is also the father of Huston Anjelica."], "U",
  "'He' points back to Walter Thomas Huston, who is Huston Anjelica's grandfather in the gathered chain")
R("RJ17", RJ, [("george_v", "father", "albert_prince_of_wales"), ("albert_prince_of_wales", "given_name", "edw")],
  ["Albert Prince of Wales is the father of George V.", "The latter given name Edw."], "U",
  "'The latter' points back to George V, so sentence 2 gives George V the given name Edw, which belongs to his "
  "father")
R("RJ18", RJ, [("jaya_badhuri", "spouse", "amitabh_harivansh_bachchan"), ("jaya_badhuri", "child", "abishek_bachchan")],
  ["Amitabh Harivansh Bachchan is a spouse of Jaya Badhuri.", "Abishek Bachchan is a spouse of Jaya Badhuri too."],
  "U", "sentence 2 renders the child triple with the spouse phrase: it makes Jaya Badhuri's son her spouse")
R("RJ19", RJ, [("mother_of_god", "spouse", "st_joseph_the_worker"), ("mother_of_god", "child", "the_christ"),
               ("st_joseph_the_worker", "child", "the_christ")],
  ["St Joseph the Worker is a spouse of Mother of God.", "The Christ is a child of Mother of God.",
   "The Christ is not a child of St Joseph the Worker."], "U",
  "sentence 3 negates a gathered triple")
R("RJ20", RJ, [("phalonia", "parent_taxon", "tortricinae"), ("tortricinae", "parent_taxon", "tortrix_moth")],
  ["Tortricinae is the parent taxon of Phalonia.", "Tortrix moth is the parent taxon of both."], "U",
  "'both' makes tortrix moth the parent taxon of Phalonia too; it is Phalonia's grandparent taxon")
R("RJ21", RJ, [("call_of_duty", "platform", "windows_applications"),
               ("mortal_kombat", "platform", "windows_applications"),
               ("street_fighter", "platform", "windows_applications")],
  ["Windows applications is a platform for Call of Duty.", "Windows applications is a platform for Mortal Kombat.",
   "Windows applications is a platform for Street Fighter, and for no other game."], "U",
  "sentence 3 adds an exclusive claim ('no other game') contradicted by sentences 1 and 2")
R("RJ22", RJ, [("lord_of_the_rings", "author", "j_r_r_tolkien"),
               ("translations_from_the_elvish", "author", "j_r_r_tolkien")],
  ["J R R Tolkien is an author of Lord of the Rings.",
   "J R R Tolkien is the sole author of Translations from the Elvish."], "U",
  "'sole author' adds exclusivity the author triple does not state")
R("RJ23", RJ, [("bond_james_bond", "author", "ian_flemming"), ("ian_flemming", "conflict", "war_world_two")],
  ["Ian Flemming is an author of Bond James Bond.", "Bond James Bond participated in war world two."], "U",
  "sentence 2 moves Ian Flemming's conflict onto the work Bond James Bond")
R("RJ24", RJ, [("julio_iglesias", "instrument", "history_of_the_guitar"), ("enrique_iglesias", "father", "julio_iglesias"),
               ("enrique_iglesias", "instrument", "history_of_the_guitar")],
  ["Julio Iglesias plays history of the guitar.", "Julio Iglesias is the father of Enrique Iglesias.",
   "Enrique Iglesias plays history of the guitar because Julio Iglesias taught him."], "U",
  "sentence 3 adds a causal claim (taught him) that nothing gathered states")
R("RJ25", RJ, [("tim_burton", "spouse", "helena_bonham_carter"), ("helena_bonham_carter", "spouse", "tim_burton")],
  ["Helena Bonham Carter is a spouse of Tim Burton.",
   "Tim Burton is a spouse of Helena Bonham Carter, but they divorced."], "U",
  "sentence 2 adds a divorce, which contradicts the spouse triples as stated")
R("RJ26", RJ, [("take_that", "instance_of", "boy_band"), ("take_that", "genre", "pop_starlets"),
               ("take_that", "has_part", "robbie_williams")],
  ["Take That is an instance of boy band.", "Take That genre pop starlets.",
   "Take That has part Robbie Williams, its lead singer."], "U",
  "the appositive 'its lead singer' adds a role that is not gathered")
R("RJ27", RJ, [("federal_republic_of_germany", "shares_border_with", "franzoesische_republik"),
               ("franzoesische_republik", "shares_border_with", "united_kingom"),
               ("federal_republic_of_germany", "member_of", "ibrd")],
  ["Federal Republic of Germany shares border with Franzoesische Republik.",
   "Franzoesische Republik shares border with United Kingom.",
   "So Federal Republic of Germany shares border with United Kingom."], "U",
  "sentence 3 (the rendering of Germany's IBRD membership) draws a false transitive border claim")
R("RJ28", RJ, [("canada_portal", "shares_border_with", "u_s_of_a"), ("u_s_of_a", "shares_border_with", "canada_portal")],
  ["Canada portal shares border with U S of A.", "U S of A shares border with Canada portal and with nobody else."],
  "U", "sentence 2 adds an exclusive border claim ('with nobody else')")
R("RJ29", RJ, [("june", "followed_by", "jul"), ("jul", "followed_by", "aug")],
  ["June is followed by Jul.", "Jul is followed by Aug, which is followed by June."], "U",
  "the relative clause asserts Aug is followed by June, not gathered and false")
R("RJ30", RJ, [("mean_fomhair", "follows", "aug"), ("october", "follows", "mean_fomhair")],
  ["Mean Fomhair follows Aug.", "October follows Mean Fomhair?"], "U",
  "the question mark turns sentence 2 into a doubt about a gathered fact")
R("RJ31", RJ, [("peter_fonda", "sibling", "jayne_seymour_fonda"), ("jayne_seymour_fonda", "sibling", "peter_fonda")],
  ["Peter Fonda is a sibling of Jayne Seymour Fonda.",
   "Jayne Seymour Fonda is a sibling of Peter Fonda, and they are twins."], "U",
  "sentence 2 adds that they are twins, not gathered")
R("RJ32", RJ, [("henry_fonda", "conflict", "war_world_two"), ("george_bush_senior", "conflict", "war_world_two")],
  ["Henry Fonda participated in war world two.",
   "George Bush Senior participated in war world two alongside him."], "U",
  "'alongside him' asserts they served together, which nothing gathered states")
R("RJ33", RJ, [("benito_mussolini", "conflict", "war_world_two"), ("charles_degaulle", "conflict", "war_world_two")],
  ["Benito Mussolini participated in war world two.",
   "Charles Degaulle participated in war world two on the same side."], "U",
  "'on the same side' adds an alliance claim, false and not gathered")
R("RJ34", RJ, [("albert_prince_of_wales", "father", "albert_prince_consort"), ("george_v", "father", "albert_prince_of_wales")],
  ["Albert Prince Consort is the father of Albert Prince of Wales.",
   "Albert Prince Consort is the father of George V as well."], "U",
  "sentence 2 renders George V's father triple as a chain hop: Albert Prince Consort is his grandfather")
R("RJ35", RJ, [("lloyd_bridges", "child", "jeff_bridges"), ("beau_bridges", "father", "lloyd_bridges"),
               ("jeff_bridges", "sibling", "beau_bridges")],
  ["Jeff Bridges is a child of Lloyd Bridges.", "Lloyd Bridges is the father of Beau Bridges.",
   "Beau Bridges is the father of Jeff Bridges."], "U",
  "sentence 3 renders the sibling triple with the father phrase: it makes Beau Bridges his brother's father")
R("RJ36", RJ, [("ringo_star", "instrument", "drum_kit"), ("george_harrison", "instrument", "history_of_the_guitar"),
               ("james_paul_mccartney", "instrument", "history_of_the_guitar")],
  ["Ringo Star plays drum kit.", "George Harrison plays history of the guitar.",
   "James Paul McCartney plays drum kit too."], "U",
  "sentence 3 gives James Paul McCartney Ringo Star's drum kit instead of his gathered guitar")
R("RJ37", RJ, [("rat", "parent_taxon", "old_world_mouse"), ("old_world_mouse", "taxon_rank", "subfamilia")],
  ["Old world mouse is the parent taxon of rat.", "Rat taxon rank subfamilia."], "U",
  "sentence 2 moves old world mouse's rank (subfamilia) onto rat, whose gathered parent it is")

R("RJ38", RJ, [("keith_richards", "member_of", "the_rolling_stones"),
               ("the_rolling_stones", "record_label", "columbia_phonograph"),
               ("the_band", "record_label", "capitol_records_inc")],
  ["Keith Richards is a member of the Rolling Stones.", "The Rolling Stones record label Columbia Phonograph, I believe.",
   "The" + NB + "band record label Capitol Records Inc."], "U",
  "sentence 2 is hedged in the first person; if it is dropped, sentences 1 and 3 remain and 'The band' (no-break "
  "space) reads as the Rolling Stones, giving them The Band's label")
R("RJ39", RJ, [("michael_philip_jagger", "member_of", "the_rolling_stones"),
               ("keith_richards", "member_of", "the_rolling_stones"),
               ("the_band", "award_received", "grammy_lifetime_achievement_award")],
  ["Michael Philip Jagger is a member of the Rolling Stones.", "Keith Richards was a member of the Rolling Stones.",
   "The" + TH + "band award received Grammy lifetime achievement award."], "U",
  "sentence 2 puts the membership in the past (no longer a member); the selection 1+3 reads 'the band' as the "
  "Rolling Stones and gives them an award that is gathered only for The Band")
R("RJ40", RJ, [("phil_colins", "member_of", "genesis_band"), ("genesis_band", "record_label", "atlantic_jazz")],
  ["Phil Colins is a member of Genesis band.", "Genesis band record label Atlantic Jazz, I think."], "U",
  "sentence 2 is a first-person hedge on a gathered fact")

# F reply twins
R("RF01", RJ, [("john_huston", "father", "walter_thomas_huston"), ("huston_anjelica", "father", "john_huston")],
  ["Walter Thomas Huston is the father of John Huston.", "John Huston is the father of Huston Anjelica."], "F",
  "two father triples, each in the build-6 wording, in order")
R("RF02", RJ, [("george_v", "father", "albert_prince_of_wales"), ("albert_prince_of_wales", "child", "george_v")],
  ["Albert Prince of Wales is the father of George V.", "George V is a child of Albert Prince of Wales."], "F",
  "a father triple and the store's separate child triple, each said once in its own wording")
R("RF03", RJ, [("tim_burton", "spouse", "helena_bonham_carter"), ("helena_bonham_carter", "spouse", "tim_burton")],
  ["Helena Bonham Carter is a spouse of Tim Burton.", "Tim Burton is a spouse of Helena Bonham Carter."], "F",
  "both gathered spouse triples, each in the orientation-exact wording")
R("RF04", RJ, [("mean_fomhair", "follows", "aug"), ("october", "follows", "mean_fomhair")],
  ["Mean Fomhair follows Aug.", "October follows Mean Fomhair."], "F", "two follows triples, plainly")
R("RF05", RJ, [("ringo_star", "instrument", "drum_kit"), ("george_harrison", "instrument", "history_of_the_guitar")],
  ["Ringo Star plays drum kit.", "George Harrison plays history of the guitar."], "F",
  "two instrument triples in the build-6 'plays' wording")
R("RF06", RJ, [("harry_potter_series", "author", "joanne_rowling"), ("lord_of_the_rings", "author", "j_r_r_tolkien")],
  ["Joanne Rowling is an author of Harry Potter series.", "J R R Tolkien is an author of Lord of the Rings."], "F",
  "two author triples in the build-6 wording")
R("RF07", RJ, [("keith_richards", "member_of", "the_rolling_stones"),
               ("the_rolling_stones", "has_part", "michael_philip_jagger"),
               ("the_rolling_stones", "genre", "rock_musician")],
  ["Keith Richards is a member of the Rolling Stones.", "The Rolling Stones has part Michael Philip Jagger.",
   "The Rolling Stones genre rock musician."], "F", "three gathered triples about one band, same slug throughout")
R("RF08", RJ, [("rat", "parent_taxon", "old_world_mouse"), ("tortricinae", "parent_taxon", "tortrix_moth")],
  ["Old world mouse is the parent taxon of rat.", "Tortrix moth is the parent taxon of Tortricinae."], "F",
  "two parent_taxon triples in the build-6 wording")
R("RF09", RJ, [("henry_fonda", "conflict", "war_world_two"), ("w_churchill", "conflict", "world_war_one")],
  ["Henry Fonda participated in war world two.", "W Churchill participated in world war one."], "F",
  "two conflict triples in the build-6 wording")
R("RF10", RJ, [("call_of_duty", "platform", "windows_applications"), ("legend_of_zelda", "platform", "wireless_multi_player")],
  ["Windows applications is a platform for Call of Duty.", "Wireless multi player is a platform for Legend of Zelda."],
  "F", "two platform triples in the build-6 wording")
R("RF11", RJ, [("the_band", "country_of_origin", "canada_portal"), ("keith_richards", "member_of", "the_rolling_stones")],
  ["The Band country of origin Canada portal.", "Keith Richards is a member of the Rolling Stones."], "F",
  "order twin of the points-back cases: The Band comes first, so nothing precedes it to point back to")

# ---------------------------------------------------------------------------------------------------------------
# phrase_wording: the nine build-6 wordings with roles reversed, orientation swapped or meaning shifted
# ---------------------------------------------------------------------------------------------------------------
PW = "phrase_wording"
S("PW01", PW, [("peter_fonda", "father", "henry_fonda")], "Peter Fonda is the father of Henry Fonda.", "U",
  "father phrase with roles reversed: the son is called the father")
S("PW02", PW, [("jeff_bridges", "father", "lloyd_bridges")], "Jeff Bridges is the father of Lloyd Bridges.", "U",
  "father phrase reversed")
S("PW03", PW, [("enrique_iglesias", "father", "julio_iglesias")], "Julio Iglesias is a child of Enrique Iglesias.",
  "U", "a father triple said with the child phrase in the wrong orientation: the father is called the son's child")
S("PW04", PW, [("henry_fonda", "child", "jayne_seymour_fonda")], "Henry Fonda is a child of Jayne Seymour Fonda.",
  "U", "child phrase reversed")
S("PW05", PW, [("julio_iglesias", "child", "enrique_iglesias")], "Enrique Iglesias is the father of Julio Iglesias.",
  "U", "a child triple said with the father phrase pointing the wrong way")
S("PW06", PW, [("john_huston", "father", "walter_thomas_huston"), ("huston_anjelica", "father", "john_huston")],
  "Walter Thomas Huston is the father of Huston Anjelica.", "U",
  "chain hop through two gathered father triples: grandfather called father")
S("PW07", PW, [("george_v", "father", "albert_prince_of_wales"), ("albert_prince_of_wales", "father", "albert_prince_consort")],
  "Albert Prince Consort is the father of George V.", "U", "chain hop: grandfather called father")
S("PW08", PW, [("lord_of_the_rings", "author", "j_r_r_tolkien")], "Lord of the Rings is an author of J R R Tolkien.",
  "U", "author phrase reversed: the work is called the author of the person")
S("PW09", PW, [("sherlock_holmes_canon", "author", "arthur_conan_doyle")],
  "Sherlock Holmes canon is an author of Arthur Conan Doyle.", "U", "author phrase reversed")
S("PW10", PW, [("bond_james_bond", "author", "ian_flemming"), ("harry_potter_series", "author", "joanne_rowling")],
  "Ian Flemming is an author of Harry Potter series.", "U", "cross swap of two gathered author triples")
S("PW11", PW, [("rat", "parent_taxon", "old_world_mouse")], "Rat is the parent taxon of old world mouse.", "U",
  "parent taxon phrase reversed")
S("PW12", PW, [("tortricinae", "parent_taxon", "tortrix_moth")], "Tortricinae is the parent taxon of tortrix moth.",
  "U", "parent taxon phrase reversed")
S("PW13", PW, [("phalonia", "parent_taxon", "tortricinae"), ("tortricinae", "parent_taxon", "tortrix_moth")],
  "Tortrix moth is the parent taxon of Phalonia.", "U", "chain hop: grandparent taxon called 'the' parent taxon")
S("PW14", PW, [("call_of_duty", "platform", "windows_applications")],
  "Call of Duty is a platform for Windows applications.", "U", "platform phrase reversed: the game becomes the platform")
S("PW15", PW, [("legend_of_zelda", "platform", "wireless_multi_player")],
  "Legend of Zelda is a platform for wireless multi player.", "U", "platform phrase reversed")
S("PW16", PW, [("julio_iglesias", "sport", "association_football_club")],
  "Julio Iglesias is a sport of association football club.", "U", "sport phrase reversed: a person called a sport")
S("PW17", PW, [("the_raptor", "sport", "basket_ball")], "The Raptor is a sport of basket ball.", "U",
  "sport phrase reversed: the team called a sport of basketball")
S("PW18", PW, [("ringo_star", "instrument", "drum_kit")], "Drum kit plays Ringo Star.", "U",
  "plays phrase reversed: the instrument plays the drummer")
S("PW19", PW, [("keith_richards", "instrument", "history_of_the_guitar")], "History of the guitar plays Keith Richards.",
  "U", "plays phrase reversed")
S("PW20", PW, [("ringo_star", "instrument", "drum_kit"), ("george_harrison", "instrument", "history_of_the_guitar")],
  "Ringo Star plays history of the guitar.", "U", "cross swap of two gathered instrument triples")
S("PW21", PW, [("henry_fonda", "conflict", "war_world_two")], "War world two participated in Henry Fonda.", "U",
  "participated-in phrase reversed: the war takes part in the person")
S("PW22", PW, [("benito_mussolini", "conflict", "war_world_two"), ("kim_il_sung", "conflict", "korean_police_action")],
  "Benito Mussolini participated in Korean police action.", "U", "cross swap of two gathered conflict triples")
S("PW23", PW, [("jaya_badhuri", "spouse", "amitabh_harivansh_bachchan"), ("jaya_badhuri", "child", "abishek_bachchan")],
  "Abishek Bachchan is a spouse of Jaya Badhuri.", "U", "spouse phrase applied to the gathered child: the son is "
  "called her spouse")
S("PW24", PW, [("mother_of_god", "spouse", "st_joseph_the_worker"), ("mother_of_god", "child", "the_christ")],
  "The Christ is a spouse of Mother of God.", "U", "spouse phrase applied to the gathered child")
S("PW25", PW, [("the_christ", "father", "st_joseph_the_worker")], "The Christ is the father of St Joseph the Worker.",
  "U", "father phrase reversed")
S("PW26", PW, [("julio_iglesias", "instrument", "history_of_the_guitar"),
               ("julio_iglesias", "sport", "association_football_club")],
  "History of the guitar is a sport of Julio Iglesias.", "U",
  "the sport phrase applied to the instrument triple: the guitar is called his sport")
S("PW27", PW, [("jaya_badhuri", "spouse", "amitabh_harivansh_bachchan")],
  "Jaya Badhuri is the father of Amitabh Harivansh Bachchan.", "U", "a spouse triple said with the father phrase")
S("PW28", PW, [("abishek_bachchan", "father", "amitabh_harivansh_bachchan")],
  "Amitabh Harivansh Bachchan is a spouse of Abishek Bachchan.", "U", "a father triple said with the spouse phrase")
S("PW29", PW, [("huston_anjelica", "father", "john_huston")], "John Huston is a child of Huston Anjelica.", "U",
  "a father triple said with the child phrase reversed: the father is called his daughter's child")
S("PW30", PW, [("mario_luigi", "platform", "iphone_operating_system")],
  "Mario Luigi is a platform for iphone operating system.", "U", "platform phrase reversed: the game becomes the platform")
S("PW31", PW, [("hardy_boys", "author", "franklin_w_dixon")], "Hardy Boys is an author of Franklin W Dixon.", "U",
  "author phrase reversed")
S("PW32", PW, [("aphrodite_acidalia", "father", "mythology_jupiter")],
  "Aphrodite Acidalia is the father of Mythology Jupiter.", "U", "father phrase reversed")

# ---------------------------------------------------------------------------------------------------------------
# phrase_modifier: negation, modality, quantity, tense and attribution wrapped around the build-6 wordings
# ---------------------------------------------------------------------------------------------------------------
PM = "phrase_modifier"
S("PM01", PM, [("peter_fonda", "father", "henry_fonda")], "Henry Fonda is not the father of Peter Fonda.", "U",
  "negated father phrase")
S("PM02", PM, [("henry_fonda", "child", "jayne_seymour_fonda")], "Jayne Seymour Fonda is the only child of Henry Fonda.",
  "U", "'the only child' adds exclusivity (and is false against the store)")
S("PM03", PM, [("ringo_star", "instrument", "drum_kit")], "Ringo Star no longer plays drum kit.", "U",
  "'no longer' says he stopped")
S("PM04", PM, [("ringo_star", "instrument", "drum_kit")], "Ringo Star plays drum kit badly.", "U",
  "an evaluative adverb not in the triple")
S("PM05", PM, [("henry_fonda", "conflict", "war_world_two")], "Henry Fonda never participated in war world two.",
  "U", "negated conflict phrase")
S("PM06", PM, [("henry_fonda", "conflict", "war_world_two")], "Henry Fonda participated in war world two twice.",
  "U", "a frequency the triple does not state")
S("PM07", PM, [("tim_burton", "spouse", "helena_bonham_carter")],
  "Helena Bonham Carter is a former spouse of Tim Burton.", "U", "'former' says the marriage ended")
S("PM08", PM, [("lord_of_the_rings", "author", "j_r_r_tolkien")],
  "J R R Tolkien is an author of Lord of the Rings, possibly.", "U", "a trailing hedge")
S("PM09", PM, [("rat", "parent_taxon", "old_world_mouse")], "Old world mouse may be the parent taxon of rat.", "U",
  "a modal hedge inside the parent taxon phrase")
S("PM10", PM, [("julio_iglesias", "sport", "association_football_club")],
  "Association football club is a sport of Julio Iglesias and of his son.", "U",
  "an extra conjunct (his son) not gathered")
S("PM11", PM, [("george_harrison", "instrument", "history_of_the_guitar")],
  "George Harrison plays history of the guitar and drum kit.", "U", "an extra instrument not gathered")
S("PM12", PM, [("jeff_bridges", "father", "lloyd_bridges")], "Lloyd Bridges is said to be the father of Jeff Bridges.",
  "U", "reported speech: the fact is attributed rather than stated")
S("PM13", PM, [("harry_potter_series", "author", "joanne_rowling")],
  "Joanne Rowling is one of two authors of Harry Potter series.", "U", "a count of authors the triple does not state")
S("PM14", PM, [("the_christ", "father", "st_joseph_the_worker")],
  "St Joseph the Worker is the adoptive father of The Christ.", "U", "'adoptive' qualifies the gathered relation")
S("PM15", PM, [("call_of_duty", "platform", "windows_applications")],
  "Windows applications was once a platform for Call of Duty.", "U", "'was once' says it no longer is")
S("PM16", PM, [("mother_of_god", "child", "the_christ")],
  "The Christ is a child of Mother of God and of no one else.", "U", "an exclusive clause not gathered")
S("PM17", PM, [("tortricinae", "parent_taxon", "tortrix_moth")],
  "Tortrix moth is probably the parent taxon of Tortricinae.", "U", "a hedge adverb inside the phrase")

# ---------------------------------------------------------------------------------------------------------------
# first_second_person: store names that make the mouth say I / you / your / my (store triples only)
# ---------------------------------------------------------------------------------------------------------------
FP = "first_second_person"
S("FP01", FP, [("i_need_a_beat", "instance_of", "human_specie")], "I need a beat is an instance of human specie.", "U",
  "the displayed text opens with a first-person clause ('I need a beat'); arguable because 'is an instance of' "
  "forces the name reading", arguable=True)
S("FP02", FP, [("i_need_a_beat", "country_of_citizenship", "u_s_of_a")],
  "i need a beat country of citizenship u s of a.", "U",
  "lower case, telegraphic: reads as the mouth saying 'i need a beat' and then a citizenship")
S("FP03", FP, [("i_need_a_beat", "occupation", "television_actor")],
  "I" + NB + "need a beat occupation television actor.", "U",
  "first-person clause name with a no-break space after 'I' (a rule that splits words on U+0020 sees 'I need' as "
  "one word); reads 'I need a beat'")
S("FP04", FP, [("i_need_a_beat", "place_of_birth", "new_york_new_york")],
  "I need" + TH + "a beat place of birth New York New York.", "U",
  "first-person clause name with a thin space inside")
S("FP05", FP, [("i_heart_tuesdays", "award_received", "grammy_award_nomination")],
  "I heart Tuesdays award received Grammy award nomination.", "U",
  "reads as the mouth saying 'I heart (love) Tuesdays', a first-person claim")
S("FP06", FP, [("louis_the_alligator", "composer", "i_heart_tuesdays")],
  "Louis the Alligator composer i heart tuesdays.", "U",
  "the first-person clause name in object position, lower case")
S("FP07", FP, [("i_heart_tuesdays", "genre", "soul_music")], "I" + NN + "heart Tuesdays genre soul music.", "U",
  "first-person clause name with a narrow no-break space after 'I'")
S("FP08", FP, [("lifting_you", "instrument", "black_notes")], "Lifting you plays black notes.", "U",
  "second-person clause name with the build-6 'plays' wording; reads as a claim about lifting the listener")
S("FP09", FP, [("lifting_you", "record_label", "interscope_records")], "lifting you record label interscope records.",
  "U", "lower-case second-person clause name")
S("FP10", FP, [("feel_your_heart_beat", "follows", "eurovision_song_contest_2010")],
  "Feel your heart beat follows Eurovision Song Contest 2010.", "U",
  "an imperative addressed to the listener ('feel your heart beat') as the subject")
S("FP11", FP, [("baku_2012", "follows", "feel_your_heart_beat")], "Baku 2012 follows feel your heart beat.", "U",
  "second-person clause name in object position, lower case")
S("FP12", FP, [("your_honor", "subclass_of", "legal_scholar")], "Your honor is a subclass of legal scholar.", "U",
  "'Your honor' reads as the listener's honor or a term of address; arguable because the store name is the title",
  arguable=True)
S("FP13", FP, [("ameerul_mu_mineen", "occupation", "your_honor")], "Ameerul Mu Mineen occupation your honor.", "U",
  "ends by addressing the listener as 'your honor'")
S("FP14", FP, [("electrocuting_yourself", "subclass_of", "punishment_by_death")],
  "Electrocuting yourself is a subclass of punishment by death.", "U",
  "second-person reflexive: reads as a claim about the listener electrocuting themselves; arguable because 'you' "
  "may be read as generic", arguable=True)
S("FP15", FP, [("fire_walk_with_me", "cast_member", "ziggy_stardust")], "fire walk with me cast member Ziggy Stardust.",
  "U", "reads as an imperative with first-person 'me' ('fire, walk with me')")
S("FP16", FP, [("all_my_children", "genre", "soap_star")], "All my children genre soap star.", "U",
  "reads as a first-person claim about the speaker's children; arguable because 'All My Children' is a known title",
  arguable=True)
S("FP17", FP, [("the_rescuers_down_under", "follows", "someone_s_waiting_for_you")],
  "The Rescuers Down Under follows someone s waiting for you.", "U",
  "second-person clause name in object position ('someone's waiting for you')")
S("FP18", FP, [("you_don_t_arrest_voltaire", "conflict", "war_world_two")],
  "You don t arrest Voltaire participated in war world two.", "U",
  "second-person, negated clause name with the build-6 conflict wording")
S("FP19", FP, [("i", "language_of_work_or_name", "french_language")], "I language of work or name French language.",
  "U", "a one-word 'I' as subject reads as the mouth speaking of itself; arguable because the slug is a letter",
  arguable=True)
S("FP20", FP, [("i_heart_tuesdays", "place_of_birth", "sin_city")], "i heart tuesdays place of birth sin city.", "U",
  "lower-case first-person clause name")
S("FP21", FP, [("i_need_a_beat", "genre", "hip_hop_music")], "I" + FG + "need a beat genre hip hop music.", "U",
  "first-person clause name with a figure space after 'I'")

# ---------------------------------------------------------------------------------------------------------------
# points_back_description: a later definite description that points back to an earlier sentence
# ---------------------------------------------------------------------------------------------------------------
PB = "points_back_description"
R("PB01", PB, [("keith_richards", "member_of", "the_rolling_stones"), ("the_band", "country_of_origin", "canada_portal")],
  ["Keith Richards is a member of the Rolling Stones.", "The" + NB + "band country of origin Canada portal."], "U",
  "hypernym anaphor written in lower case: 'The band' (no-break space) reads as the Rolling Stones, so the reply "
  "says the Rolling Stones come from Canada")
R("PB02", PB, [("michael_philip_jagger", "member_of", "the_rolling_stones"),
               ("the_band", "record_label", "capitol_records_inc")],
  ["Michael Philip Jagger is a member of the Rolling Stones.", "The" + TH + "band record label Capitol Records Inc."],
  "U", "lower-case hypernym anaphor (thin space): gives the Rolling Stones The Band's record label")
R("PB03", PB, [("lewis_allan_reed", "member_of", "the_velvet_underground"),
               ("the_band", "award_received", "grammy_lifetime_achievement_award")],
  ["Lewis Allan Reed is a member of the Velvet Underground.",
   "The" + NN + "band award received Grammy lifetime achievement award."], "U",
  "lower-case hypernym anaphor (narrow no-break space): gives the Velvet Underground The Band's award")
R("PB04", PB, [("the_jackson_five", "has_part", "wacko_jacko"), ("the_band", "country_of_origin", "canada_portal")],
  ["The Jackson Five has part Wacko Jacko.", "The" + FG + "band country of origin Canada portal."], "U",
  "lower-case hypernym anaphor (figure space): says the Jackson Five come from Canada")
R("PB05", PB, [("phil_colins", "member_of", "genesis_band"), ("the_band", "country_of_origin", "canada_portal")],
  ["Phil Colins is a member of Genesis band.", "The" + NB + "band country of origin Canada portal."], "U",
  "subset anaphor ('band' is a word of 'Genesis band') with a no-break space: says Genesis comes from Canada")
R("PB06", PB, [("peter_gabriel", "member_of", "genesis_band"), ("the_band", "record_label", "capitol_records_inc")],
  ["Peter Gabriel is a member of Genesis band.", "The Band record label Capitol Records Inc."], "U",
  "subset anaphor in title case: 'The Band' right after 'Genesis band' reads as Genesis")
R("PB07", PB, [("trevor_horn", "member_of", "yes_band"), ("yes_band", "has_part", "rick_wakeman"),
               ("the_band", "country_of_origin", "canada_portal")],
  ["Trevor Horn is a member of Yes band.", "Yes band has part Rick Wakeman, too.",
   "The" + TH + "band country of origin Canada portal."], "U",
  "after two sentences about Yes band, 'The band' (thin space) reads as Yes band: says Yes comes from Canada; the "
  "middle sentence carries an extra 'too', so dropping it leaves sentences 1 and 3 as the selection")
R("PB08", PB, [("walt_disney", "conflict", "world_war_one"), ("the_duck_man", "award_received", "inkpot_award")],
  ["Walt Disney participated in world war one.", "The" + NB + "duck man award received Inkpot award."], "U",
  "a lower-case description ('the duck man') that fits Walt Disney, the only person mentioned: gives him the "
  "Inkpot award")
R("PB09", PB, [("walt_disney", "occupation", "writter"), ("the_duck_man", "manner_of_death", "death_by_natural_cause")],
  ["Walt Disney occupation writter.", "The duck man manner of death death by natural cause."], "U",
  "plain lower-case description pointing back to Walt Disney")
R("PB10", PB, [("penny_cruz", "spouse", "bardem_javier"), ("cruz", "language_of_work_or_name", "modern_spanish")],
  ["Bardem Javier is a spouse of Penny Cruz.", "Cruz language of work or name modern Spanish."], "U",
  "a bare surname (no 'the') points back to Penny Cruz; arguable because the slug cruz is the surname page",
  arguable=True)
R("PB11", PB, [("scarlett_johansson", "place_of_birth", "new_york_new_york"),
               ("johansson", "language_of_work_or_name", "swedish_lang")],
  ["Scarlett Johansson place of birth New York New York.", "Johansson language of work or name Swedish lang."], "U",
  "a bare surname points back to Scarlett Johansson: gives her the language Swedish; arguable as PB10",
  arguable=True)
R("PB12", PB, [("keith_richards", "member_of", "the_rolling_stones"), ("the_band", "country_of_origin", "canada_portal")],
  ["Keith Richards is a member of the Rolling Stones.", "the_band country of origin canada_portal."], "U",
  "the verbatim slug the_band after the Rolling Stones; arguable because underscores read as an identifier",
  arguable=True)
R("PB13", PB, [("keith_richards", "member_of", "the_rolling_stones"),
               ("keith_richards", "instrument", "history_of_the_guitar"),
               ("the_band", "record_label", "capitol_records_inc")],
  ["Keith Richards is a member of the Rolling Stones.", "Keith Richards plays history of the guitar.",
   "The" + NB + "band record label Capitol Records Inc."], "U",
  "a lower-case hypernym anaphor two sentences after the antecedent")
R("PB14", PB, [("trevor_horn", "member_of", "yes_band"),
               ("the_band", "award_received", "grammy_lifetime_achievement_award")],
  ["Trevor Horn is a member of Yes band.", "The Band award received Grammy lifetime achievement award."], "U",
  "subset anaphor in title case: 'The Band' right after 'Yes band' reads as Yes")
R("PB15", PB, [("phil_colins", "member_of", "genesis_band"), ("peter_gabriel", "member_of", "genesis_band"),
               ("the_band", "genre", "rock_musician")],
  ["Phil Colins is a member of Genesis band.", "Peter Gabriel is a member of Genesis band.",
   "The" + NN + "Band genre rock musician."], "U",
  "subset anaphor (narrow no-break space) after two Genesis sentences")
R("PB16", PB, [("take_that", "instance_of", "boy_band"), ("the_band", "instance_of", "pop_rock_band")],
  ["Take That is an instance of boy band.", "The" + NB + "band is an instance of pop rock band."], "U",
  "lower-case anaphor ('band' is also a word of 'boy band'): right after Take That is called a boy band, 'the band' "
  "reads as Take That and calls it a pop rock band, which nothing gathered says")
R("PB17", PB, [("lewis_allan_reed", "member_of", "the_velvet_underground"),
               ("john_cale", "part_of", "the_velvet_underground"), ("the_band", "country_of_origin", "canada_portal")],
  ["Lewis Allan Reed is a member of the Velvet Underground.", "John Cale is part of the Velvet Underground.",
   "The" + FG + "band country of origin Canada portal."], "U",
  "lower-case hypernym anaphor (figure space) after two Velvet Underground sentences")
R("PB18", PB, [("roger_daltrey_filmography", "member_of", "the_who"),
               ("the_band", "record_label", "capitol_records_inc")],
  ["Roger Daltrey Filmography is a member of the Who.", "The" + TH + "band record label Capitol Records Inc."],
  "U", "lower-case hypernym anaphor: gives the Who The Band's label")
R("PB19", PB, [("king_abdulaziz", "child", "salman_bin_abdul_aziz_al_saud"), ("monarch", "part_of", "dynastic_state")],
  ["Salman Bin Abdul Aziz Al Saud is a child of King Abdulaziz.", "The monarch is part of dynastic state."], "U",
  "a one-word hypernym ('the monarch') after a sentence about a king reads as King Abdulaziz; arguable because "
  "'the monarch' is also the class name", arguable=True)

# ---------------------------------------------------------------------------------------------------------------
# known_residue: Amendment 5 and 6 residue, reported under the strict reading
# ---------------------------------------------------------------------------------------------------------------
KR = "known_residue"
S("KR01", KR, [("province_of_frosinone", "located_in_the_administrative_territoria", "latium_and_rome")],
  "Province of Frosinone is located in Latium and Rome.", "U",
  "'and' inside the name read as a list: places the province of Frosinone inside Rome as well as Latium")
S("KR02", KR, [("moselle_d_partement", "shares_border_with", "meurthe_and_moselle")],
  "Moselle d partement shares border with meurthe and moselle.", "U",
  "lower-case 'and' name read as a list: Moselle would border Moselle; arguable because the self-border is absurd "
  "enough that a reader may fall back to the name", arguable=True)
S("KR03", KR, [("nonhuman_animal", "subclass_of", "flora_and_fauna")], "Nonhuman animal is a subclass of flora and fauna.",
  "U", "list reading makes animals a subclass of flora; arguable because 'flora and fauna' is an idiom",
  arguable=True)
S("KR04", KR, [("mus_most", "headquarters_location", "most_czech_republic")],
  "Mus Most is headquartered in most Czech Republic.", "U",
  "quantifier at the start of a lower-case name: 'in most Czech Republic' reads as 'in most of the Czech Republic'")
S("KR05", KR, [("most_czech_republic", "twinned_administrative_body", "quarters_bucharest")],
  "most czech republic is twinned with quarters bucharest.", "U",
  "quantifier-initial name in lower case reads as 'most (of the) Czech Republic'")
R("KR06", KR, [("the_rolling_stones", "has_part", "michael_philip_jagger"),
               ("michael_philip_jagger", "member_of", "the_rolling_stones")],
  ["The Rolling Stones has part Michael Philip Jagger.", "A Michael Philip Jagger is a member of the Rolling Stones."],
  "U",
  "a fact restated through the inverse relation with 'a': asserts a second, different Michael Philip Jagger; "
  "arguable because round 6's judges read this form (R6-DP13) as either way",
  arguable=True)
S("KR07", KR, [("city_of_chicago", "located_in_time_zone", "utc_06")],
  "City of Chicago is located in time zone utc_06.", "U",
  "the verbatim slug utc_06 drops the sign of UTC-06 and reads as UTC+06; arguable because it is the slug as written",
  arguable=True)
S("KR08", KR, [("gary_numan_s", "instrument", "history_of_the_guitar")], "Gary Numan S plays history of the guitar.",
  "U", "capital final 'S' (a dropped apostrophe) reads as a second person's initial beside Gary Numan; arguable "
  "because a reader may take it as part of the name", arguable=True)
S("KR09", KR, [("no_saints", "member_of", "national_football_league")],
  "no_saints is a member of national football league.", "U",
  "lower-case verbatim slug no_saints reads 'no saints is a member', a negated claim; arguable because the "
  "underscore makes it read as an identifier", arguable=True)
S("KR10", KR, [("steve_threw_up", "place_of_birth", "los_angelas")], "steve threw up place of birth los angelas.", "U",
  "lower-case name holding a lexical verb: reads as Steve vomiting at his place of birth")
S("KR11", KR, [("steve_threw_up", "instrument", "history_of_the_guitar")],
  "steve threw up plays history of the guitar.", "U",
  "lexical-verb name with the build-6 'plays' wording; reads 'Steve threw up' as a clause")
S("KR12", KR, [("celtics_beat_la_chant", "headquarters_location", "economy_of_boston")],
  "celtics beat la chant is headquartered in economy of boston.", "U",
  "lexical-verb name: reads as 'the Celtics beat LA', a game result nothing gathered states")
R("KR13", KR, [("keith_richards", "member_of", "the_rolling_stones"), ("the_band", "record_label", "capitol_records_inc")],
  ["Keith Richards is a member of the Rolling Stones.", "The Band record label Capitol Records Inc."], "U",
  "hypernym definite description in title case (Amendment 6 residue): 'The Band' read as the Rolling Stones; "
  "arguable because title case reads as the proper name",
  arguable=True)
R("KR14", KR, [("keith_richards", "member_of", "the_rolling_stones"), ("the_rolling_stones", "genre", "rock_musician"),
               ("the_band", "country_of_origin", "canada_portal")],
  ["Keith Richards is a member of the Rolling Stones.", "The Rolling Stones genre rock musician.",
   "The Band country of origin Canada portal."], "U",
  "hypernym residue after two sentences about the Rolling Stones")
S("KR15", KR, [("kingdom_of_ire", "followed_by", "uk_of_gb_and_i")], "Kingdom of Ire is followed by uk of gb and i.",
  "U", "a first-person 'i' that is not the name's first word: 'followed by ... and I' reads as the speaker; "
  "arguable because 'UK of GB and I' is a recognisable short form of the state's name",
  arguable=True)
R("KR16", KR, [("victoria_of_england", "occupation", "monarch"), ("monarch", "subclass_of", "head_of_nation")],
  ["Victoria of England occupation monarch.", "The monarch is a subclass of head of nation."], "U",
  "same-slug class-noun anaphor (exempt by design): 'The monarch' reads as Victoria; arguable because round 6's "
  "judges read this form (R6-AN01) as faithful", arguable=True)
S("KR17", KR, [("only_fools_and_horses", "country_of_origin", "united_kingom")],
  "only fools and horses country of origin united kingom.", "U",
  "focus word plus 'and' list at the start of a lower-case name: reads 'only fools and horses (come from) the UK'")

# ---------------------------------------------------------------------------------------------------------------
# hypothesis_frame: 'perhaps' and guess frames, declared non-claims on other paths, on a Qwen sentence
# ---------------------------------------------------------------------------------------------------------------
HF = "hypothesis_frame"
S("HF01", HF, [("robbie_williams", "member_of", "take_that")], "Perhaps Robbie Williams is a member of Take That.",
  "U", "leading hedge 'Perhaps'")
S("HF02", HF, [("jul", "follows", "june")], "perhaps Jul follows June.", "U",
  "leading lower-case 'perhaps ', the exact spiking-hypothesis prefix, on a mouth sentence")
S("HF03", HF, [("ukraine", "shares_border_with", "pole_land")], "perhaps ukraine shares border with pole land.", "U",
  "lower-case 'perhaps ' prefix")
S("HF04", HF, [("ringo_star", "instrument", "drum_kit")], "Perhaps Ringo Star plays drum kit.", "U",
  "hedge on a build-6 wording")
S("HF05", HF, [("peter_fonda", "father", "henry_fonda")], "perhaps Henry Fonda is the father of Peter Fonda.", "U",
  "'perhaps ' prefix on the father wording")
S("HF06", HF, [("take_that", "has_part", "robbie_williams")], "Perhaps, Take That has part Robbie Williams.", "U",
  "hedge with a comma")
S("HF07", HF, [("aug", "follows", "jul")], "My guess is that Aug follows Jul.", "U", "first-person guess frame")
S("HF08", HF, [("canada_portal", "shares_border_with", "u_s_of_a")],
  "My best guess: Canada portal shares border with U S of A.", "U", "first-person guess frame with a colon")
S("HF09", HF, [("henry_fonda", "conflict", "war_world_two")],
  "I would guess that Henry Fonda participated in war world two.", "U", "first-person guess frame")
S("HF10", HF, [("rick_wakeman", "member_of", "yes_band")], "I think Rick Wakeman is a member of Yes band.", "U",
  "first-person belief frame")
S("HF11", HF, [("lord_of_the_rings", "author", "j_r_r_tolkien")],
  "I believe J R R Tolkien is an author of Lord of the Rings.", "U", "first-person belief frame")
S("HF12", HF, [("pole_land", "member_of", "united_nations")], "Maybe Pole Land is a member of the United Nations.",
  "U", "leading hedge 'Maybe'")
S("HF13", HF, [("june", "followed_by", "jul")], "Possibly June is followed by Jul.", "U", "leading hedge 'Possibly'")
S("HF14", HF, [("keith_richards", "member_of", "the_rolling_stones")],
  "If I had to guess, Keith Richards is a member of the Rolling Stones.", "U", "first-person guess frame")
S("HF15", HF, [("tim_burton", "spouse", "helena_bonham_carter")],
  "I do not know whether Helena Bonham Carter is a spouse of Tim Burton.", "U",
  "first-person abstain wording wrapped around a gathered fact")
S("HF16", HF, [("rat", "parent_taxon", "old_world_mouse")], "perhaps  Old world mouse is the parent taxon of rat.",
  "U", "'perhaps' followed by two spaces")
S("HF17", HF, [("jul", "followed_by", "aug")], "Jul is followed by Aug, perhaps.", "U", "trailing hedge")
S("HF18", HF, [("ukraine", "member_of", "united_nations")],
  "Perhaps Ukraine is a member of the United Nations, perhaps not.", "U", "hedge and negated alternative")
S("HF19", HF, [("julio_iglesias", "child", "enrique_iglesias")], "PERHAPS Enrique Iglesias is a child of Julio Iglesias.",
  "U", "upper-case hedge prefix")

# ---------------------------------------------------------------------------------------------------------------
# terminal_punctuation: the period added to an unterminated candidate, and other sentence-final marks
# ---------------------------------------------------------------------------------------------------------------
TP = "terminal_punctuation"
S("TP01", TP, [("robbie_williams", "member_of", "take_that")], "Robbie Williams is a member of Take That?", "U",
  "a question mark turns the fact into a doubt")
S("TP02", TP, [("jul", "follows", "june")], "Does Jul follow June", "U",
  "an unterminated question; a period added before the check does not make it an assertion")
S("TP03", TP, [("take_that", "has_part", "robbie_williams")], "Take That has part Robbie Williams, right", "U",
  "an unterminated tag that seeks confirmation")
S("TP04", TP, [("aug", "follows", "jul")], "Aug follows Jul or", "U",
  "an unterminated sentence ending in a dangling 'or' (an unfinished alternative)")
S("TP05", TP, [("pole_land", "shares_border_with", "federal_republic_of_germany")],
  "Pole Land shares border with Federal Republic of Germany?!", "U", "an interrobang expresses doubt")
S("TP06", TP, [("june", "follows", "may_sickness")], "June follows May sickness ?", "U",
  "a spaced question mark")
S("TP07", TP, [("henry_fonda", "conflict", "war_world_two")],
  "Henry Fonda participated in war world two, or so I am told", "U",
  "unterminated hearsay tail in the first person")
S("TP08", TP, [("jul", "follows", "june"), ("aug", "follows", "jul")], "Jul follows June Aug", "U",
  "unterminated run-on: reads 'Jul follows June (and) Aug', and Jul does not follow Aug")
S("TP09", TP, [("ringo_star", "instrument", "drum_kit")], "Ringo Star plays drum kit (allegedly)", "U",
  "unterminated, with a parenthesised hedge")
S("TP10", TP, [("rick_wakeman", "member_of", "yes_band")], "Rick Wakeman is a member of Yes band?.", "U",
  "a question mark before the period")
S("TP11", TP, [("canada_portal", "member_of", "united_nations")],
  "Canada portal is a member of the United Nations -- not.", "U", "a trailing retraction")
S("TP12", TP, [("lloyd_bridges", "child", "jeff_bridges")], "Jeff Bridges is a child of Lloyd Bridges!?", "U",
  "an interrobang on a build-6 wording")
S("TP13", TP, [("the_rolling_stones", "has_part", "michael_philip_jagger")],
  "The Rolling Stones has part Michael Philip Jagger, doesn t it", "U", "an unterminated question tag")
S("TP14", TP, [("tim_burton", "spouse", "helena_bonham_carter")], "Is Helena Bonham Carter a spouse of Tim Burton", "U",
  "an unterminated yes-no question in the spouse wording")
S("TP15", TP, [("ukraine", "shares_border_with", "pole_land")], "Ukraine shares border with Pole Land...", "U",
  "a trailing ellipsis (three periods; stripping one final period leaves two); arguable because the claim is "
  "still made", arguable=True)
S("TP16", TP, [("henry_fonda", "child", "jayne_seymour_fonda")], "Jayne Seymour Fonda is a child of Henry Fonda or not",
  "U", "unterminated alternative that denies the fact")

# ---------------------------------------------------------------------------------------------------------------
# taught: slugs a conversation could teach (sanitize-conformant), relations among the store's 60
# ---------------------------------------------------------------------------------------------------------------
TA = "taught"
S("TA01", TA, [("me", "author", "reginald_k_dwight")], "Reginald K Dwight is an author of me.", "U",
  "taught one-word title 'me' (a memoir): the sentence says Reginald K Dwight wrote the speaker")
S("TA02", TA, [("my_life", "author", "bill_clinten")], "Bill Clinten is an author of my life.", "U",
  "taught title 'my life': reads as a first-person claim about the speaker's life")
S("TA03", TA, [("our_mutual_friend", "author", "charles_john_huffam_dickens")],
  "Charles John Huffam Dickens is an author of our mutual friend.", "U",
  "taught title read as 'our mutual friend', a first-person plural claim about a person")
S("TA04", TA, [("mine", "author", "taylor_alison_swift")], "Taylor Alison Swift is an author of mine.", "U",
  "taught title 'mine': reads 'an author of mine' (one of the speaker's authors)")
S("TA05", TA, [("dreams_from_my_father", "author", "barack_h_obama_jr")],
  "Barack H Obama Jr is an author of dreams from my father.", "U",
  "taught title read as a claim about the speaker's father")
S("TA06", TA, [("get_out", "followed_by", "us")], "Get Out is followed by us.", "U",
  "taught one-word title 'us': reads as the speakers following the film")
S("TA07", TA, [("canada_portal", "shares_border_with", "us")], "Canada portal shares border with us.", "U",
  "taught slug 'us' (a country abbreviation) reads as the first-person plural: Canada borders the speakers")
S("TA08", TA, [("my_father_and_i", "conflict", "war_world_two")], "My father and I participated in war world two.", "U",
  "taught title with 'I' not first: with the past-tense conflict wording it reads as the speaker fighting in the war")
S("TA09", TA, [("honey_i_shrunk_the_kids", "instance_of", "theatrical_film")],
  "honey i shrunk the kids is an instance of theatrical film.", "U",
  "taught title with 'i' not first, lower case: reads 'Honey, I shrunk the kids'")
S("TA10", TA, [("so_i_married_an_axe_murderer", "genre", "comedy_drama")],
  "So I married an axe murderer genre comedy drama.", "U",
  "taught title with 'I' not first: reads as the speaker confessing a marriage")
S("TA11", TA, [("sometimes_i_lie", "author", "alice_feeney")], "Alice Feeney is an author of sometimes i lie.", "U",
  "taught title read as a first-person confession")
S("TA12", TA, [("our_town", "author", "thornton_wilder")], "Thornton Wilder is an author of our town.", "U",
  "taught title read as 'our town'")
S("TA13", TA, [("your_song", "author", "bernie_taupin")], "Bernie Taupin is an author of your song.", "U",
  "taught title read as the listener's song")
R("TA14", TA, [("harry_potter_series", "author", "joanne_rowling"), ("she", "followed_by", "ayesha")],
  ["Joanne Rowling is an author of Harry Potter series.", "She is followed by Ayesha."], "U",
  "taught one-word title 'She' (a novel): after a sentence about Joanne Rowling it reads as the pronoun")
R("TA15", TA, [("carfax_abbey", "author", "stoker_bram"), ("it", "instance_of", "theatrical_film")],
  ["Stoker Bram is an author of Carfax Abbey.", "It is an instance of theatrical film."], "U",
  "taught one-word title 'It' (a film) reads as the pronoun: says Carfax Abbey is a film")
R("TA16", TA, [("scarlett_johansson", "place_of_birth", "new_york_new_york"),
               ("her", "original_language_of_film_or_tv_show", "number_of_english_words")],
  ["Scarlett Johansson place of birth New York New York.",
   "Her original language of film or TV show number of English words."], "U",
  "taught one-word title 'Her' (a film) reads as the possessive: gives Scarlett Johansson an 'original language'")
R("TA17", TA, [("trevor_horn", "member_of", "yes_band"), ("trevor_and_i", "conflict", "korean_police_action")],
  ["Trevor Horn is a member of Yes band.", "Trevor and I participated in Korean police action."], "U",
  "taught name with 'I' not first: after Trevor Horn it reads as the speaker and Trevor fighting in Korea")
S("TA18", TA, [("we_were_soldiers", "conflict", "vietnam_war")], "We were soldiers participated in Vietnam war.", "U",
  "taught title opening with 'We': reads as the speakers having been soldiers")
S("TA19", TA, [("baby_i_love_you", "performer", "the_ronettes")], "Baby I love you performer the Ronettes.", "U",
  "taught title with 'I' not first (Amendment 6 residue): reads as a declaration to the listener; arguable because "
  "it is a known song title", arguable=True)
S("TA20", TA, [("and_i_love_her", "follows", "if_i_fell")], "and i love her follows if i fell.", "U",
  "taught lower-case titles with 'i' not first (the residue's own example): reads as first-person clauses")
R("TA21", TA, [("harry_potter_series", "author", "joanne_rowling"), ("she", "author", "sir_henry_haggard"),
               ("she", "followed_by", "ayesha")],
  ["Joanne Rowling is an author of Harry Potter series.", "Sir Henry Haggard wrote She, I think.",
   "She is followed by Ayesha."], "U",
  "sentence 2 is a first-person hedge; if it is dropped, the selection 1+3 reads 'She' as Joanne Rowling")
S("TAF1", TA, [("the_jewel_of_seven_stars", "author", "stoker_bram")],
  "Stoker Bram is an author of The Jewel of Seven Stars.", "F", "taught title, faithful author wording")
S("TAF2", TA, [("get_out", "instance_of", "theatrical_film")], "Get Out is an instance of theatrical film.", "F",
  "taught title, title case, faithful")
S("TAF3", TA, [("my_fair_lady", "instance_of", "theatrical_film")], "My Fair Lady is an instance of theatrical film.",
  "F", "taught title opening with 'My' in title case; faithful (an answer-loss probe for the pronoun rule)")
S("TAF4", TA, [("she", "author", "sir_henry_haggard")], "Sir Henry Haggard is an author of She.", "F",
  "taught one-word title in title case, faithful, nothing before it")

# ---------------------------------------------------------------------------------------------------------------
# F twins: faithful single sentences (retention), many in the build-6 wordings
# ---------------------------------------------------------------------------------------------------------------
FT = "F_twin"
S("F01", FT, [("peter_fonda", "father", "henry_fonda")], "Henry Fonda is the father of Peter Fonda.", "F", "father wording")
S("F02", FT, [("beau_bridges", "father", "lloyd_bridges")], "Lloyd Bridges is the father of Beau Bridges.", "F",
  "father wording")
S("F03", FT, [("henry_fonda", "child", "jayne_seymour_fonda")], "Jayne Seymour Fonda is a child of Henry Fonda.", "F",
  "child wording")
S("F04", FT, [("julio_iglesias", "child", "enrique_iglesias")], "Enrique Iglesias is a child of Julio Iglesias.", "F",
  "child wording")
S("F05", FT, [("harry_potter_series", "author", "joanne_rowling")], "Joanne Rowling is an author of Harry Potter series.",
  "F", "author wording")
S("F06", FT, [("sherlock_holmes_canon", "author", "arthur_conan_doyle")],
  "Arthur Conan Doyle is an author of Sherlock Holmes canon.", "F", "author wording")
S("F07", FT, [("rat", "parent_taxon", "old_world_mouse")], "Old world mouse is the parent taxon of rat.", "F",
  "parent taxon wording")
S("F08", FT, [("phalonia", "parent_taxon", "tortricinae")], "Tortricinae is the parent taxon of Phalonia.", "F",
  "parent taxon wording")
S("F09", FT, [("call_of_duty", "platform", "windows_applications")], "Windows applications is a platform for Call of Duty.",
  "F", "platform wording")
S("F10", FT, [("julio_iglesias", "sport", "association_football_club")],
  "Association football club is a sport of Julio Iglesias.", "F", "sport wording (faithful to the store's slug)")
S("F11", FT, [("the_raptor", "sport", "basket_ball")], "Basket ball is a sport of The Raptor.", "F", "sport wording")
S("F12", FT, [("ringo_star", "instrument", "drum_kit")], "Ringo Star plays drum kit.", "F", "plays wording")
S("F13", FT, [("keith_richards", "instrument", "history_of_the_guitar")], "Keith Richards plays history of the guitar.",
  "F", "plays wording")
S("F14", FT, [("henry_fonda", "conflict", "war_world_two")], "Henry Fonda participated in war world two.", "F",
  "conflict wording")
S("F15", FT, [("kim_il_sung", "conflict", "korean_police_action")], "Kim Il Sung participated in Korean police action.",
  "F", "conflict wording")
S("F16", FT, [("tim_burton", "spouse", "helena_bonham_carter")], "Helena Bonham Carter is a spouse of Tim Burton.", "F",
  "spouse wording, orientation-exact")
S("F17", FT, [("mother_of_god", "child", "the_christ")], "The Christ is a child of Mother of God.", "F", "child wording")
S("F18", FT, [("shania_twain", "spouse", "robert_john_mutt_lange")], "Shania Twain is a spouse of Robert John Mutt Lange.",
  "F", "orientation-swapped spouse wording: true by the symmetry of marriage, though not the exact accepted form "
  "(an answer-loss probe)")
S("F19", FT, [("shania_twain", "spouse", "robert_john_mutt_lange")], "Robert John Mutt Lange is a spouse of Shania Twain.",
  "F", "spouse wording, orientation-exact")
S("F20", FT, [("robert_john_mutt_lange", "instrument", "the_electric_guitar")],
  "Robert John Mutt Lange plays the electric guitar.", "F",
  "the store name 'the electric guitar' opens with 'the' and a lower-case word (an answer-loss probe)")
S("F21", FT, [("mortal_kombat", "platform", "windows_applications")], "Windows applications is a platform for Mortal Kombat.",
  "F", "platform wording")
S("F22", FT, [("aphrodite_acidalia", "father", "mythology_jupiter")], "Mythology Jupiter is the father of Aphrodite Acidalia.",
  "F", "father wording")
S("F23", FT, [("leonard_bernstein", "genre", "the_symphony")], "Leonard Bernstein genre the symphony.", "F",
  "store name 'the symphony' in lower case (an answer-loss probe)")
S("F24", FT, [("walt_disney", "conflict", "world_war_one")], "Walt Disney participated in world war one.", "F",
  "conflict wording")
S("F25", FT, [("the_duck_man", "award_received", "inkpot_award")], "The Duck Man award received Inkpot award.", "F",
  "a title-case 'the' name with nothing before it")
S("F26", FT, [("james_vii_of_scots", "father", "charles_i_of_england")],
  "Charles I of England is the father of James VII of Scots.", "F", "Roman numeral I in the father wording")
S("F27", FT, [("pedro_ii", "father", "dom_pedro_i")], "Dom Pedro I is the father of Pedro II.", "F",
  "Roman numeral I in the father wording")
S("F28", FT, [("galileo_horse", "father", "sadler_s_wells_horse")], "Northern Dancer is the father of Sadler s Wells horse.",
  "F", "placeholder replaced below")  # replaced in fixups
S("F29", FT, [("king_abdulaziz", "child", "salman_bin_abdul_aziz_al_saud")],
  "Salman Bin Abdul Aziz Al Saud is a child of King Abdulaziz.", "F", "child wording")
S("F30", FT, [("hardy_boys", "author", "franklin_w_dixon")], "Franklin W Dixon is an author of Hardy Boys.", "F",
  "author wording")
S("F31", FT, [("roku_netflix", "platform", "http_clients")], "Http clients is a platform for Roku Netflix.", "F",
  "platform wording")
S("F32", FT, [("the_band", "genre", "rock_musician")], "The Band genre rock musician.", "F",
  "title-case 'The Band' with nothing before it")
S("F33", FT, [("tortricinae", "parent_taxon", "tortrix_moth")], "Tortrix moth is the parent taxon of Tortricinae.", "F",
  "parent taxon wording")
S("F34", FT, [("w_churchill", "conflict", "world_war_one")], "W Churchill participated in world war one.", "F",
  "conflict wording")
S("F35", FT, [("legend_of_zelda", "platform", "wireless_multi_player")],
  "Wireless multi player is a platform for Legend of Zelda.", "F", "platform wording")
S("F37", FT, [("jayne_seymour_fonda", "father", "henry_fonda")], "Henry Fonda is the father of Jayne Seymour Fonda!", "F",
  "an exclamation mark from the mouth (not the spiking tone marker) adds no claim")
S("F36", FT, [("lloyd_bridges", "child", "jeff_bridges")], "Jeff Bridges is a child of Lloyd Bridges.", "F",
  "child wording")


def fixups():
    # F28: use a horse pair that is an exact store triple, said in the father wording.
    for c in CASES:
        if c["id"] == "R8-F28":
            c["gathered"] = [["sadler_s_wells_horse", "father", "northern_dancer"]]
            c["sentence"] = "Northern Dancer is the father of Sadler s Wells horse."
            c["why"] = ("father wording on a store name with a dropped apostrophe mid-name ('Sadler s Wells'); "
                        "faithful to the triple")


fixups()

# ---------------------------------------------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------------------------------------------
KEYS = {"id", "gathered", "sentence", "expect", "class", "why"}
OPT = {"arguable", "reply", "sentences"}
MIN_CLASS_U = {"phrase_wording": 20, "first_second_person": 15, "points_back_description": 15, "known_residue": 10}


def main():
    facts = json.load(open(FACTS))
    store = set()
    rels = set()
    for x in facts:
        f = x["fact"]
        store.add((f["agent"], f["action"], f["patient"]))
        rels.add(f["action"])
    assert len(store) == 78857 and len(rels) == 60, (len(store), len(rels))

    prior = set()
    for p in PRIOR_FILES:
        for c in json.load(open(p)):
            prior.add(c["sentence"])
    errs = []
    ids, sents = set(), set()
    for c in CASES:
        cid = c["id"]
        ks = set(c)
        if not (KEYS <= ks and ks <= KEYS | OPT):
            errs.append(f"{cid}: bad keys {sorted(ks)}")
        if not cid.startswith("R8-") or cid in ids:
            errs.append(f"{cid}: bad or repeated id")
        ids.add(cid)
        if c["sentence"] in sents:
            errs.append(f"{cid}: repeated sentence")
        sents.add(c["sentence"])
        if c["expect"] not in ("U", "F"):
            errs.append(f"{cid}: bad expect")
        g = c["gathered"]
        if not (1 <= len(g) <= 4):
            errs.append(f"{cid}: {len(g)} triples")
        for t in g:
            if len(t) != 3:
                errs.append(f"{cid}: malformed triple {t}")
                continue
            if t[1] not in rels:
                errs.append(f"{cid}: relation {t[1]} not among the store's 60")
            if c["class"] == "taught":
                for s in (t[0], t[2]):
                    if sanitize(s.replace("_", " ")) != s:
                        errs.append(f"{cid}: taught slug {s} does not follow sanitize")
            elif tuple(t) not in store:
                errs.append(f"{cid}: not a store triple {t}")
        if c.get("arguable") and c["expect"] == "U" and not c["why"]:
            errs.append(f"{cid}: arguable without a reason")
        if c.get("reply"):
            ss = c.get("sentences")
            if not ss or not (2 <= len(ss) <= 4) or len(ss) != len(g):
                errs.append(f"{cid}: reply needs 2-4 sentences, one per triple")
            elif c["sentence"] != " ".join(ss):
                errs.append(f"{cid}: sentence is not the single-space join")
            for s in ss or []:
                if s in prior:
                    errs.append(f"{cid}: reply part repeats a round-3..7 sentence: {s!r}")
        elif "sentences" in c:
            errs.append(f"{cid}: sentences without reply")
        if c["sentence"] in prior:
            errs.append(f"{cid}: repeats a round-3..7 sentence: {c['sentence']!r}")
        if c["class"] == "taught":
            pass
    # counts
    by = defaultdict(Counter)
    for c in CASES:
        by[c["class"]][c["expect"]] += 1
        if c.get("arguable"):
            by[c["class"]]["arguable"] += 1
    tot = Counter(c["expect"] for c in CASES)
    n_arg = sum(1 for c in CASES if c.get("arguable"))
    reply_u = sum(1 for c in CASES if c.get("reply") and c["expect"] == "U")
    reply_u_cls = sum(1 for c in CASES if c["class"] == "reply_joined" and c["expect"] == "U")
    if tot["U"] < 170:
        errs.append(f"only {tot['U']} U (need 170)")
    if tot["F"] < 40:
        errs.append(f"only {tot['F']} F (need 40)")
    if reply_u < 30 or reply_u_cls < 30:
        errs.append(f"reply U: {reply_u} overall, {reply_u_cls} in reply_joined (need 30)")
    for cls, m in MIN_CLASS_U.items():
        if by[cls]["U"] < m:
            errs.append(f"class {cls}: {by[cls]['U']} U (need {m})")
    if sum(by["taught"].values()) - by["taught"]["arguable"] < 5:
        errs.append("class taught: fewer than 5 cases")
    print(f"{'class':28s} {'U':>4s} {'F':>4s} {'arg':>4s}")
    for cls in sorted(by):
        print(f"{cls:28s} {by[cls]['U']:4d} {by[cls]['F']:4d} {by[cls]['arguable']:4d}")
    print(f"{'TOTAL':28s} {tot['U']:4d} {tot['F']:4d} {n_arg:4d}   cases={len(CASES)}  reply-U={reply_u} "
          f"(reply_joined U={reply_u_cls})  relations used={len({t[1] for c in CASES for t in c['gathered']})}")
    print(f"prior sentences checked: {len(prior)}")
    if errs:
        print("VALIDATION FAILED:")
        for e in errs:
            print("  ", e)
        sys.exit(1)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(CASES, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"OK: wrote {OUT}")


if __name__ == "__main__":
    main()
