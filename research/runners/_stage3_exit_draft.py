"""STAGE 3 (ADOLESCENT) EXIT GATE -- DRAFT RUNNER (not yet registered; dev seeds only).

The third developmental stage-exit attempt, run END TO END on the INTEGRATED production brain: the live
`webapp.server.brain_chat` handler at production defaults, one fresh brain per arm in its own process (the Stage-2
harness, research/runners/stage2_exit_gate.py on branch research/stage2-exit-gate, cloned). Master roadmap section 4,
STAGE 3 -- "3-hop novel conclusion via workspace; self-report tracks ground-truth confidence; false-belief passes;
deep held-out inference after teacher-corrected episodes >= replay-only" -- operationalized at the REPLY level, with
the design synthesized from three independent drafts (docs/drafts/stage3-exit/PLAN.md: validity, ownership,
feasibility) under Claude's rulings of 2026-10-13.

WHAT IS NEW AGAINST STAGE 2 (the two-clause verdict). Every criterion carries
  CAPABILITY  the Stage-2 machinery: correct on 6/6, a silencing LESION with a reply-level effect reproduced against
              its own `.rep0`, a PERMUTED exposure the reply must follow, plus ONE CONTENT-SPECIFIC intervention
              (a steer / neural clamp predicting one specific DIFFERENT answer: a host relay that never reads the
              stimulated population cannot produce it);
  OWNERSHIP   who decided the answer: F (hold the neural state, set the host decision to its alternative), ORDER /
              RECORD (permute or delete information only the host holds), RI (re-decide the recorded spiking read
              with the host threshold x/÷1.5, offline), G (silence or drive one population, host code untouched).
A criterion PASSes only with CAPABILITY PASS and OWNERSHIP in {OWNED, OWNED(read-out)}; "capability PASS,
host-owned" is reported as its own outcome. R1: a decisive verdict computed by host code with no spiking read to
vary is ownership HOST, never UNDEFINED. R2: UNDEFINED is reserved for instrument failure (an arm missing, a patch
with applied=False, a dirty null, an errored turn or organ error). R3: host actuation in read-out mode (helper
currents, query clocks, admission by verb) is OWNED(read-out).

THE FOUR CRITERIA (one group each):
  C1 3-hop novel conclusion via the workspace   group `chain`   a BRANCHING graph (puma -> shrew -> slug; puma -> gecko
                                                                 -> mantis -> flea) with a FORK probe; declared departure
                                                                 BRAIN_RECONSOLIDATION=0 in every chain arm (a one-to-many
                                                                 teach otherwise rewrites the first fact in place: a named
                                                                 production gap, PLAN.md)
  C2 self-report tracks ground-truth confidence  group `meta`    the brain's NATURAL uncertainty: category-inheritance
                                                                 members whose frames are consistent (one builder category)
                                                                 vs conflicting (two); the opt-out (abstain) is a counted
                                                                 report level (Kepecs-style); type-2 AUROC stratified to the
                                                                 GUESS stratum so a taught-vs-guess host proxy cannot carry it
  C3 false belief passes                         group `tom`     classic + COUNTERBALANCED (belief = the second-named
                                                                 location) + true-belief control + two unwitnessed moves
                                                                 with a return after; store lesion; witness flip (F);
                                                                 belief-store steer (G); spiking-witness CANDIDATE arms
  C4 deep held-out inference after teacher-      group `credit`  a hidden feature (hunts the wasp) shared by the corrected
     corrected episodes >= replay-only                           and a HELD-OUT member no correction mentioned; the
                                                                 replay_only comparator counted; kb_pop (RECORD) +
                                                                 corr_freeze (G)

ARMS: intact_a, intact_b (the null), a silencing LESION + `.rep0`, PERMUTED exposure, the content-specific STEER/G
arm, the OWNERSHIP arms (F / ORDER / RECORD), comparators (C4), and INFORMATIONAL / CANDIDATE arms (never counted).
No seed threading (as Stage 2). Harness identical to Stage 2 (numpy, one BLAS thread, stub renderer, no LLM,
tiny-demo + shipped LTM, BRAIN_CHAT_SEED=<seed>, PYTHONHASHSEED=0, `rich` omitted). No sim/ or webapp/ edit.

GATE SEEDS ARE REFUSED: this is a DRAFT. A gate seed (42/43/44/100/101/102) runs only from a pool job emitted against
a COMMITTED preregistration (none exists yet; PREREG below names the draft text). Dev seeds 1-12 need nothing.

  .venv/bin/python -m research.runners._stage3_exit_draft --selftest
  bash tools/memcap.sh 10 -- .venv/bin/python -m research.runners._stage3_exit_draft --shard --seed 7 --group chain \
      --arm intact_a --out research/findings/raw/_stage3_exit/dev/s7/chain__intact_a.json
  .venv/bin/python -m research.runners._stage3_exit_draft --aggregate research/findings/raw/_stage3_exit/dev \
      --seeds 7 --out research/findings/raw/_stage3_exit/dev/aggregate_s7.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.lab import attributable_to  # noqa: E402  (the lesion-vs-null attribution, gates/attribution_required)
from research.runners import host_share_schema as HSS  # noqa: E402  (ONE definition of host share; plan step 2)

GATE_SEEDS = (42, 43, 44, 100, 101, 102)
DEV_SEEDS = tuple(range(1, 13))
EXPECTED_SOURCE = "tiny-demo +LTM"
GATE_OUT_DIR = "research/findings/raw/_stage3_exit/gate"
DEV_OUT_DIR = "research/findings/raw/_stage3_exit/dev"
# the DRAFT design text; NOT a registered preregistration. A gate-seed job must cite the committed prereg once Claude
# registers one (research/findings/<date>-stage3-exit-gate-PREREGISTRATION.md); until then every gate seed is refused.
PREREG = "docs/drafts/stage3-exit/PLAN.md"
PREREG_REGISTERED = False
MAX_AGE_H = 240
MEM_GB = 9               # Stage-2 A2 measured peak RSS 6.3-8.1 GB; refined by the dev-7 smoke (ceil(1.3 x peak))
MEMCAP_GB = 10
AUROC_BAR = 0.75         # C2: half the conflicting items opt out while every other guess item is answered correctly
MIN_CORRECT_GUESS = 2    # C2: fewer correct guess items than this -> capability absent (no generalization to score)
RI_MIN_FRACTION = 0.85   # read-out invariance: the decisive read is unchanged on >= this fraction of items / cycles
RI_FACTOR = 1.5          # the host threshold x/÷ this factor
HOST_SHARE_TARGET = 0.2  # Phase D's number (reported, never a pass condition here)
# C2 read-out invariance grid: the organ's own host cuts (category_inherit_organ.FLOOR_RATE / TIE_MARGIN)
RI_TIE_MARGINS = (1.1, 1.25, 1.6)
RI_FLOOR_RATES = (0.013, 0.02, 0.03)

BASE_ENV = {"SIM_BACKEND": "numpy", "BRAIN_CHAT_RENDERER": "stub", "SIM_DISABLE_LLM": "1", "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "BRAIN_LTM_SHIP_DEFAULT": "1", "PYTHONHASHSEED": "0"}

VERB_3SG = {"chase": "chases", "eat": "eats", "hunt": "hunts", "fly": "flies", "nest": "nests", "swim": "swims",
            "hide": "hides"}
_3SG_TO_LEMMA = {v: k for k, v in VERB_3SG.items()}
# every word this gate introduces must be absent from the tiny-demo vocabulary and BOTH shipped LTM bundles
# (wikidata_100k, wikidata_core_15k); checked 2026-10-13 against ~/Projects/sim-data/knowledge_bundles (all absent)
NOVEL_WORDS = ("puma", "shrew", "gecko", "slug", "mantis", "flea",
               "tern", "kite", "carp", "bream", "grub", "weed", "lark", "finch", "dace", "roach", "bat", "duck", "seal",
               "auk", "moth", "snail", "squid", "krill", "reef", "lake", "tree", "sky",
               "robin", "wren", "worm", "swift", "canary", "jay", "rook", "wasp", "cricket", "seed", "nut")

# ══════════════════════════════════════════════ C1 chain (branching graph + fork) ═══════════════════════════════════
_CHAIN_TEACH = [("t_mantis", "the mantis chases the flea", "teach"),      # out of order: the 3-hop terminal is neither
                ("t_puma_1", "the puma chases the shrew", "teach"),       # the first- nor the last-taught patient
                ("t_shrew", "the shrew chases the slug", "teach"),
                ("t_puma_2", "the puma chases the gecko", "teach"),       # the SECOND patient of puma (the fork)
                ("t_gecko", "the gecko chases the mantis", "teach")]
_CHAIN_PROBES = [("p3", "does puma chase flea", "probe"),                 # 3 hops on branch 2: puma->gecko->mantis->flea
                 ("p2", "does puma chase slug", "probe"),                 # 2 hops on branch 1: puma->shrew->slug
                 ("m_adj", "does puma chase gecko", "probe"),             # a TOLD fact: yes, never framed derived
                 ("m_cross", "does shrew chase flea", "control"),         # abstain (the branches do not cross)
                 ("m_rev", "does flea chase puma", "control"),            # abstain (direction)
                 ("r_fork", "what does the puma chase all the way", "probe"),   # the FORK: both terminals or neither
                 ("r_all", "what does the gecko chase all the way", "probe"),   # unambiguous 2-hop chase -> flea, derived
                 ("c_direct", "what does the gecko chase", "control"),    # one hop; must SURVIVE the workspace lesion
                 ("c_tail", "what does the flea chase all the way", "control"),  # the tail: abstain
                 ("c_dog", "what does the dog chase", "control")]
CHAIN_TURNS = _CHAIN_TEACH + _CHAIN_PROBES
# permuted: branch 1 is re-routed through the mantis (slug never taught): p2 abstains, m_cross becomes a 2-hop yes,
# the fork's two branches CONVERGE on the flea (so naming the flea alone is honest), r_all/c_direct unchanged
CHAIN_PERMUTED = [("t_mantis", "the mantis chases the flea", "teach"), ("t_puma_1", "the puma chases the shrew", "teach"),
                  ("t_shrew", "the shrew chases the mantis", "teach"), ("t_puma_2", "the puma chases the gecko", "teach"),
                  ("t_gecko", "the gecko chases the mantis", "teach")] + _CHAIN_PROBES
# ORDER (ownership): the two puma teaches swapped; the host first-match now picks the gecko branch
CHAIN_ORDER_SWAP = [("t_mantis", "the mantis chases the flea", "teach"), ("t_puma_1", "the puma chases the gecko", "teach"),
                    ("t_shrew", "the shrew chases the slug", "teach"), ("t_puma_2", "the puma chases the shrew", "teach"),
                    ("t_gecko", "the gecko chases the mantis", "teach")] + _CHAIN_PROBES
CHAIN_FORK_TERMINALS = ("slug", "flea")
STEER_AT_CUE, STEER_CONCEPT = "flea", "slug"     # steer_leaf: at the leaf `flea` drive the slot carrying `slug`
SILENCE_CONCEPT = "mantis"                       # silence_mantis: the proposal `mantis` gets no drive (branch 2 only)
# the derived frame lead the C2 compositional route and the (required-in-pin) multistep fix both print
DERIVED_LEAD = "I derived this from"

# ══════════════════════════════════════════════ C2 meta (natural uncertainty) ═══════════════════════════════════════
def _fx(a, v, p, prep=None):
    return "the %s %s %sthe %s" % (a, VERB_3SG[v], (prep + " ") if prep else "", p)


_PREP = {"fly": "in", "nest": "in", "swim": "in", "hide": "in"}
# the builders: two categories, three frames each; the FOOD frame is what a member inherits
_META_BUILDERS = {"tern": [("eat", "grub"), ("fly", "sky"), ("nest", "tree")],
                  "kite": [("eat", "grub"), ("fly", "sky"), ("nest", "tree")],
                  "carp": [("eat", "weed"), ("swim", "lake"), ("hide", "reef")],
                  "bream": [("eat", "weed"), ("swim", "lake"), ("hide", "reef")]}
# the probe members: two frames each. CONSISTENT = both frames from one builder category (the food is inferable);
# CONFLICTING = one frame from each (a near-tie by construction; the true food is NEVER taught, so any answer is
# wrong and the abstain is the correct opt-out)
_META_MEMBERS = {"lark": [("fly", "sky"), ("nest", "tree")], "finch": [("fly", "sky"), ("nest", "tree")],
                 "dace": [("swim", "lake"), ("hide", "reef")], "roach": [("swim", "lake"), ("hide", "reef")],
                 "bat": [("fly", "sky"), ("swim", "lake")], "duck": [("fly", "sky"), ("hide", "reef")],
                 "seal": [("swim", "lake"), ("nest", "tree")], "auk": [("nest", "tree"), ("hide", "reef")]}
META_MEMBER_ORDER = ("lark", "bat", "finch", "duck", "dace", "seal", "roach", "auk")
META_CONSISTENT = ("lark", "finch", "dace", "roach")
META_CONFLICTING = ("bat", "duck", "seal", "auk")
META_TAUGHT = ("tern", "kite", "carp", "bream")
# the frozen WORLD table (correctness): the inferable food for a consistent member, an untaught food for a conflicting one
META_WORLD = {"lark": "grub", "finch": "grub", "dace": "weed", "roach": "weed",
              "bat": "moth", "duck": "snail", "seal": "squid", "auk": "krill",
              "tern": "grub", "kite": "grub", "carp": "weed", "bream": "weed"}
# PERMUTED: the builders' foods swapped AND the consistent/conflicting ROLES swapped between the two member sets, so a
# name-keyed confidence (abstain on bat/duck/seal/auk by name) cannot track, and the answers must follow the food
_META_BUILDERS_PERM = {"tern": [("eat", "weed"), ("fly", "sky"), ("nest", "tree")],
                       "kite": [("eat", "weed"), ("fly", "sky"), ("nest", "tree")],
                       "carp": [("eat", "grub"), ("swim", "lake"), ("hide", "reef")],
                       "bream": [("eat", "grub"), ("swim", "lake"), ("hide", "reef")]}
_META_MEMBERS_PERM = {"bat": [("fly", "sky"), ("nest", "tree")], "duck": [("fly", "sky"), ("nest", "tree")],
                      "seal": [("swim", "lake"), ("hide", "reef")], "auk": [("swim", "lake"), ("hide", "reef")],
                      "lark": [("fly", "sky"), ("swim", "lake")], "finch": [("fly", "sky"), ("hide", "reef")],
                      "dace": [("swim", "lake"), ("nest", "tree")], "roach": [("nest", "tree"), ("hide", "reef")]}
META_CONSISTENT_PERM = ("bat", "duck", "seal", "auk")
META_CONFLICTING_PERM = ("lark", "finch", "dace", "roach")
META_WORLD_PERM = {"bat": "weed", "duck": "weed", "seal": "grub", "auk": "grub",
                   "lark": "moth", "finch": "snail", "dace": "squid", "roach": "krill",
                   "tern": "weed", "kite": "weed", "carp": "grub", "bream": "grub"}
STEER_MEMBER, STEER_FRAME = "bat", ("eat", "grub")   # steer_frame: drive the `eat grub` frame while the bat is read
NOISE_SIGMAS = (1.5, 3.0)                            # noise_taught (INFO): r_*_1 at sigma[0], r_*_2 at sigma[1]; dev-calibrated


def _meta_turns(builders, members):
    turns = []
    for a, frs in builders.items():
        for v, p in frs:
            turns.append(("b_%s_%s" % (a, v), _fx(a, v, p, _PREP.get(v)), "teach"))
    for a in META_MEMBER_ORDER:                     # a fixed member order and positional labels: the permuted table
        for i, (v, p) in enumerate(members[a]):     # keeps every label (only the messages change)
            turns.append(("m_%s_%d" % (a, i), _fx(a, v, p, _PREP.get(v)), "teach"))
    for a in ("lark", "bat", "dace", "duck", "finch", "seal", "roach", "auk"):     # guess probes, roles interleaved
        turns.append(("q_%s" % a, "what does the %s eat" % a, "probe"))
    for a in META_TAUGHT:
        turns.append(("r_%s_1" % a, "what does the %s eat" % a, "probe"))
    turns.append(("c_dog", "what does the dog chase", "control"))
    for a in META_TAUGHT:
        turns.append(("r_%s_2" % a, "what does the %s eat" % a, "probe"))
    return turns


META_TURNS = _meta_turns(_META_BUILDERS, _META_MEMBERS)
META_PERMUTED = _meta_turns(_META_BUILDERS_PERM, _META_MEMBERS_PERM)
META_GUESS_LABELS = tuple("q_%s" % a for a in META_CONSISTENT + META_CONFLICTING)
META_TAUGHT_LABELS = tuple("r_%s_1" % a for a in META_TAUGHT)
META_TAUGHT_LABELS_2 = tuple("r_%s_2" % a for a in META_TAUGHT)

# ══════════════════════════════════════════════ C3 tom (false belief) ══════════════════════════════════════════════
_FILLER = "what does the cat eat"
TOM_TURNS = [
    # S1 classic: the belief is the FIRST-named location; a delay before the query
    ("s1_place", "Sally puts the marble in the basket.", "expose"), ("s1_leave", "Sally leaves the room.", "expose"),
    ("s1_move", "Anne moves the marble to the box.", "expose"),
    ("s1_delay1", "what does the dog chase", "control"), ("s1_delay2", "how do you feel", "control"),
    ("q1", "Where will Sally look for the marble?", "probe"),
    # S2 COUNTERBALANCED: the belief is the SECOND-named location (Tom's own witnessed move), defeating the slot-0 default
    ("s2_place", "Tom puts the key in the drawer.", "expose"), ("s2_move1", "Tom moves the key to the jar.", "expose"),
    ("s2_leave", "Tom leaves the room.", "expose"), ("s2_move2", "Mia moves the key to the cup.", "expose"),
    ("q2", "Where will Tom look for the key?", "probe"),
    # S3 TRUE-BELIEF control: returns before the move
    ("s3_place", "Lea puts the coin in the bag.", "expose"), ("s3_leave", "Lea leaves the room.", "expose"),
    ("s3_return", "Lea returns.", "expose"), ("s3_move", "Ben moves the coin to the tin.", "expose"),
    ("q3", "Where will Lea look for the coin?", "probe"),
    # S4 two unwitnessed moves, returns AFTER: present at query, belief stale
    ("s4_place", "Kate puts the ring in the chest.", "expose"), ("s4_leave", "Kate leaves the room.", "expose"),
    ("s4_move1", "Ben moves the ring to the pot.", "expose"), ("s4_move2", "Ben moves the ring to the shelf.", "expose"),
    ("s4_return", "Kate returns.", "expose"),
    ("q4", "Where will Kate look for the ring?", "probe"),
    ("m1", "Where will Ben look for the ring?", "control"),        # agent mismatch: names no location
    ("c_dog", "what does the dog chase", "control"),
]
# PERMUTED: witnessing swapped (a leave / return becomes a filler): Sally stays -> box; Tom stays -> cup; Lea never
# returns -> bag; Kate stays -> shelf. Labels identical; only expose turns change.
_TOM_PERM_SWAP = {"s1_leave": _FILLER, "s2_leave": _FILLER, "s3_return": _FILLER, "s4_leave": _FILLER}
TOM_PERMUTED = [(l, _TOM_PERM_SWAP.get(l, m), r) for l, m, r in TOM_TURNS]
TOM_EXPECT = {"q1": "basket", "q2": "jar", "q3": "tin", "q4": "chest"}
TOM_EXPECT_PERM = {"q1": "box", "q2": "cup", "q3": "bag", "q4": "shelf"}
TOM_WITNESS_FLIP_ALT = dict(TOM_EXPECT_PERM)    # the F alternative: every MOVE's witnessing inverted = the same table
TOM_QUERIES = ("q1", "q2", "q3", "q4")
STEER_STORE_SLOT, STEER_STORE_LOC = 1, "box"    # steer_store: on q1 drive the `box` belief assembly (slot 1) before the read

# ══════════════════════════════════════════════ C4 credit (held-out deep inference) ═════════════════════════════════
_CREDIT_EXPOSE = [("x_robin_fly", "the robin flies in the sky", "teach"), ("x_robin_eat", "the robin eats the worm", "teach"),
                  ("x_wren_fly", "the wren flies in the sky", "teach"), ("x_wren_eat", "the wren eats the worm", "teach"),
                  ("x_lark_fly", "the lark flies in the sky", "teach"), ("x_lark_hunt", "the lark hunts the wasp", "teach"),
                  ("x_swift_fly", "the swift flies in the sky", "teach"), ("x_swift_hunt", "the swift hunts the wasp", "teach"),
                  ("x_canary_fly", "the canary flies in the sky", "teach"), ("x_canary_hunt", "the canary hunts the wasp", "teach"),
                  ("x_jay_fly", "the jay flies in the sky", "teach"), ("x_jay_hunt", "the jay hunts the cricket", "teach"),
                  ("x_rook_fly", "the rook flies in the sky", "teach"), ("x_rook_hunt", "the rook hunts the cricket", "teach")]
_CREDIT_EPISODES = [("e1", "what does the lark eat", "episode"),              # the brain's own guess (worm) ...
                    ("k1", "the lark eats the seed", "teach"),             # ... corrected by the teacher
                    ("e2", "what does the swift eat", "episode"),
                    ("k2", "the swift eats the seed", "teach")]
_CREDIT_PROBES = [("H", "what does the canary eat", "probe"),               # HELD OUT: no correction mentioned the canary
                  ("J", "what does the jay eat", "control"),               # a cricket-hunter: the correction must NOT reach it
                  ("c_lark", "what does the lark eat", "control"),         # the corrected item retained
                  ("c_robin", "what does the robin eat", "control"),       # an uncorrected told fact
                  ("c_dog", "what does the dog chase", "control")]
CREDIT_TURNS = _CREDIT_EXPOSE + _CREDIT_EPISODES + _CREDIT_PROBES
IDLE = "say something"


def _credit_variant(e=None, k=None):
    sw = {}
    if e is not None:
        sw.update(e1=e, e2=e)
    if k is not None:
        sw.update(k1=k[0], k2=k[1])
    return [(l, sw.get(l, m), r) for l, m, r in CREDIT_TURNS]


CREDIT_REPLAY_ONLY = _credit_variant(e=IDLE)                                   # the same facts, no error before them
CREDIT_NO_TEACHER = _credit_variant(k=(IDLE, IDLE))                            # the teacher never corrects
CREDIT_PERMUTED = _credit_variant(k=("the lark eats the nut", "the swift eats the nut"))
CREDIT_CORRECTED = (("lark", "seed"), ("swift", "seed"))                       # kb_pop / corr_freeze targets
CREDIT_CORRECTED_PERM = (("lark", "nut"), ("swift", "nut"))

# ══════════════════════════════════════════════ GROUPS ══════════════════════════════════════════════════════════════
GROUPS = {
    "chain": {
        "criterion": "C1", "primary": True,
        "env": {"BRAIN_RECONSOLIDATION": "0"},   # DECLARED DEPARTURE (every chain arm): a one-to-many teach otherwise
        #                                          rewrites the first fact in place (PLAN.md, named production gap)
        "turns": CHAIN_TURNS, "permuted_turns": CHAIN_PERMUTED,
        "variants": {"order_swap": CHAIN_ORDER_SWAP},
        "observe_patch": "chase_rates",          # harness-wide observation (records the chase's per-slot rates for RI)
        "lesions": {"ws_lesion": {"env": {"BRAIN_GNW_MULTISTEP_LESION": "1", "BRAIN_TRANSITIVE_LESION": "1"}, "patch": None,
                                  "cuts": "the GNW workspace's assembly self-recurrence (one mechanism, read by both the "
                                          "'all the way' chase and the transitive yes/no route)"}},
        "interventions": {
            "steer_leaf": {"kind": "steer", "env": {}, "patch": "steer_leaf",
                           "note": "at the cycle whose cue is the leaf `flea` the slot carrying `slug` is driven at IGNITE_PA: "
                                   "r_all must name slug (a host relay that never reads the workspace cannot)"},
            "silence_mantis": {"kind": "G", "env": {}, "patch": "silence_mantis",
                               "note": "the proposal `mantis` gets no drive: r_all (and p3 if answered) collapse, p2 unchanged"},
            "order_swap": {"kind": "ORDER", "env": {}, "patch": None,
                           "note": "the two puma teaches swapped: primary fields on p3/p2/m_adj/r_fork must be identical"},
            "kselect_order": {"kind": "F", "env": {}, "patch": "kselect_order",
                              "note": "the host first-match over the stored (puma, chase) facts returns the LAST match "
                                      "(host_share k-select 'order'): F on p3 and p2"},
        },
        "informational": {"reach_cand": {"env": {"BRAIN_TRANSITIVE_SPIKING_REACH": "1"}, "patch": None,
                                         "note": "CANDIDATE route: the detect population replaces the host membership verdict"}},
    },
    "meta": {
        "criterion": "C2", "primary": True, "env": {},
        "turns": META_TURNS, "permuted_turns": META_PERMUTED, "variants": {}, "observe_patch": None,
        "lesions": {"inherit_lesion": {"env": {"BRAIN_CATEGORY_INHERIT_LESION": "1"}, "patch": None,
                                       "cuts": "both learned category pathways at read time (the organ reaches no frame)"}},
        "interventions": {
            "steer_frame": {"kind": "G", "env": {}, "patch": "steer_frame",
                            "note": "the `eat grub` frame assembly is driven while the bat (a near-tie) is read: the bat must "
                                    "answer grub; the lark (consistent) unchanged"},
        },
        "informational": {"metacog_lesion": {"env": {"BRAIN_METACOG_LESION": "1"}, "patch": None,
                                             "note": "the verbal hedge channel (predicted inert on guess turns)"},
                          "noise_taught": {"env": {}, "patch": "noise_taught",
                                           "note": "the Stage-2-style store-noise instrument on the taught recalls only "
                                                   "(secondary; sigma levels to be dev-calibrated)"}},
    },
    "tom": {
        "criterion": "C3", "primary": True, "env": {},
        "turns": TOM_TURNS, "permuted_turns": TOM_PERMUTED, "variants": {}, "observe_patch": None,
        "lesions": {"store_lesion": {"env": {"BRAIN_FALSE_BELIEF_STORE_LESION": "1"}, "patch": None,
                                     "cuts": "the spiking belief store's assembly self-recurrence (world/self untouched)"}},
        "interventions": {
            "steer_store": {"kind": "G", "env": {}, "patch": "steer_store",
                            "note": "on q1 the `box` belief assembly is driven with the organ's own write current before "
                                    "the read: q1 must name box"},
            "witness_flip": {"kind": "F", "env": {}, "patch": "witness_flip",
                             "note": "the host `witnessed` boolean inverted on every MOVE: F on q1-q4"},
        },
        "informational": {
            "cand_intact": {"env": {"BRAIN_FALSE_BELIEF_SPIKING_WITNESS": "1", "BRAIN_FALSE_BELIEF_WITNESS_OP": "4"}, "patch": None,
                            "note": "CANDIDATE: the spiking witness (read-out mode, op 4) decides the write"},
            "cand_witness_lesion": {"env": {"BRAIN_FALSE_BELIEF_SPIKING_WITNESS": "1", "BRAIN_FALSE_BELIEF_WITNESS_OP": "4",
                                            "BRAIN_FALSE_BELIEF_WITNESS_LESION": "1"}, "patch": None,
                                    "note": "CANDIDATE lesion: presence's influence removed (every move written)"},
            "cand_presence_clamp": {"env": {"BRAIN_FALSE_BELIEF_SPIKING_WITNESS": "1", "BRAIN_FALSE_BELIEF_WITNESS_OP": "4"},
                                    "patch": "presence_clamp",
                                    "note": "CANDIDATE G: presence silenced across S3's move (q3 -> bag) and driven across "
                                            "S1's move (q1 -> box)"},
        },
    },
    "credit": {
        "criterion": "C4", "primary": True, "env": {},
        "turns": CREDIT_TURNS, "permuted_turns": CREDIT_PERMUTED,
        "variants": {"replay_only": CREDIT_REPLAY_ONLY, "no_teacher": CREDIT_NO_TEACHER}, "observe_patch": None,
        "lesions": {"no_teacher": {"env": {}, "patch": None, "variant": True,
                                   "cuts": "the teacher (the correction turns become idle turns): is the teacher load-bearing?"},
                    "learn_lesion": {"env": {"BRAIN_CATEGORY_INHERIT_LESION": "lateral_learn"}, "patch": None,
                                     "cuts": "the frame->frame plasticity gate (never opened): no deep association forms"}},
        "comparators": {"replay_only": {"note": "the same facts at the same positions, no prediction by the brain before "
                                                "them (counted: corrected >= replay_only on H)"}},
        "interventions": {
            "kb_pop": {"kind": "RECORD", "env": {}, "patch": "kb_pop",
                       "note": "after k1/k2 the corrected fact's host kb record is popped (the organ untouched): H unchanged "
                               "means the content is substrate-held"},
            "corr_freeze": {"kind": "G", "env": {}, "patch": "corr_freeze",
                            "note": "the organ's plasticity gates stay shut on k1/k2 only: H must revert to no_teacher's reply"},
        },
        "informational": {},
    },
}
PRIMARY_GROUPS = tuple(g for g, d in GROUPS.items() if d["primary"])
CRITERIA = ("C1", "C2", "C3", "C4")
INTACT_ARMS = ("intact_a", "intact_b", "corrected_a", "corrected_b")


def _intact_names(group):
    return ("corrected_a", "corrected_b") if group == "credit" else ("intact_a", "intact_b")


def turns_for(group, seed, arm):
    g = GROUPS[group]
    base = arm.split(".rep")[0]
    if base == "permuted":
        return list(g["permuted_turns"])
    if base in g.get("variants", {}):
        return list(g["variants"][base])
    return list(g["turns"])


def arms_for(group, with_repeats=True, with_informational=True):
    g = GROUPS[group]
    arms = list(_intact_names(group))
    for les in g["lesions"]:
        arms.append(les)
        if with_repeats:
            arms.append(les + ".rep0")
    arms.append("permuted")
    arms += list(g.get("comparators", {}))
    arms += [a for a in g["interventions"] if a not in arms]
    if with_informational:
        arms += list(g["informational"])
    return arms


def arm_spec(group, arm):
    """(env, patch, kind) for an arm. kind in intact | lesion | permuted | comparator | steer | G | F | ORDER | RECORD |
    informational."""
    g = GROUPS[group]
    base = arm.split(".rep")[0]
    if base in _intact_names(group):
        return {}, None, "intact"
    if base == "permuted":
        return {}, None, "permuted"
    if base in g["lesions"]:
        d = g["lesions"][base]
        return dict(d["env"]), d["patch"], "lesion"
    if base in g.get("comparators", {}):
        return {}, None, "comparator"
    if base in g["interventions"]:
        d = g["interventions"][base]
        return dict(d["env"]), d["patch"], d["kind"]
    if base in g["informational"]:
        d = g["informational"][base]
        return dict(d["env"]), d["patch"], "informational"
    raise ValueError("unknown arm %s for group %s" % (arm, group))


def counted_arms(group):
    return arms_for(group, with_informational=False)

# m_adj (PLAN.md section 3.3): a told-directly conclusion that is also chain-reachable is over-marked derived at HEAD; the
# builder may land a stored-fact lookup fix. False -> m_adj is REPORTED (sub-clause m_adj_told_plain), not counted;
# True (set at GATE_SHA by an instrument-only amendment) -> m_adj must be answered gecko with `derived` False.
M_ADJ_TOLD_FIX_IN_PIN = False


# ══════════════════════════════════════════════ production frame strings (imported, never retyped) ═════════════════
def _production_derived_lex():
    """The lead strings production prints before a derived claim, read off the production functions themselves:
    `compositional_chain_route.frame_derived_answer` (the hop-list lead and its generic fallback) and
    `source_provenance_honesty.provenance_framed_text` (the swap wording the #129 monitor prints when it reads
    GENERATED; it drops the hop list). Returns (leads_before_claim, swap_marker, from_production)."""
    try:
        from research.runners.compositional_chain_route import frame_derived_answer
        from research.runners.source_provenance_honesty import provenance_framed_text, PROVENANCE_GENERATED
        marker = "\x00CLAIM\x00"
        lead = frame_derived_answer(marker, [["a", "v", "b"]]).split(marker)[0].replace("a v b.", "").strip()
        fallback = frame_derived_answer(marker, []).split(marker)[0].strip()
        swap = provenance_framed_text("fact", marker, PROVENANCE_GENERATED)
        swap_lead, swap_tail = swap.split(marker)[0].strip(), swap.split(marker)[1].strip().lstrip(",").strip()
        return (lead, fallback, swap_lead), swap_tail, True
    except Exception:
        return (("I derived this from:", "I reasoned this myself rather than being told it directly.", "I believe"),
                "but I reasoned that myself rather than being told it directly.", False)


DERIVED_LEADS, DERIVED_SWAP_TAIL, DERIVED_LEX_FROM_PRODUCTION = _production_derived_lex()


def _hedge_text():
    try:
        from research.runners.metacog_production_organ import hedge_prefix
        return hedge_prefix().strip()
    except Exception:
        return "My decision-margin reads this as low-confidence, so take it as uncertain:"


HEDGE_TEXT = _hedge_text()


# ══════════════════════════════════════════════ WORKER (inside the arm subprocess) ═════════════════════════════════
PATCH_STATE = {"label": None}      # the worker stamps the current turn label; every patch event records it


def _ev(rec, **kw):
    """One intervention event: counted toward `calls`; toward `n_applied` unless it carries `error`."""
    rec["calls"] += 1
    if not kw.get("error"):
        rec["n_applied"] += 1
    e = dict(label=PATCH_STATE.get("label"))
    e.update(kw)
    rec["events"].append(e)
    if len(rec["events"]) > 400:
        rec["events"] = rec["events"][-400:]


def install_patch(name, seed):
    """The in-process manipulations (no sim/ or webapp/ edit). Returns a record dict the worker stores: `calls`,
    `n_applied` (calls without an error), per-turn `events`, and optional `_pre_turn` / `_post_turn` hooks (callables,
    stripped before the record is written). Every patch is DEFENSIVE: a production internal that is not where this
    draft expects it records an error event (-> applied False -> that seed UNDEFINED) instead of crashing the arm.
    PLAN.md section 10 lists every patch under "needs dev-7 verification"."""
    if name is None:
        return None
    rec = {"patch": name, "calls": 0, "n_applied": 0, "events": []}
    try:
        _install(name, seed, rec)
    except Exception as e:            # the patch could not be installed at all: recorded, never a crash
        rec["install_error"] = "%s: %s" % (type(e).__name__, e)
    return rec


def _install(name, seed, rec):
    if name == "chase_rates":
        # OBSERVATION ONLY (every chain arm): the chase's per-slot late-window rates + THR per cycle, for RI
        from research.runners import _gnw_reentrant_metacog_gated_deliberation_derisk as D
        orig = D._deliberate_hop_conf
        rec["thr"] = float(D.THR)

        def _hop(bridge, xp, slots_dev, snap, target, distractors, nmda_attr=D.NMDA_ATTR_DEFAULT):
            out = orig(bridge, xp, slots_dev, snap, target, distractors, nmda_attr=nmda_attr)
            try:
                committed, rates, winner, n_ignited, conf, assignment = out
                _ev(rec, target=target, rates=[round(float(r), 6) for r in list(rates)], winner=int(winner),
                    n_ignited=int(n_ignited), committed=committed, thr=float(D.THR))
            except Exception as e:
                _ev(rec, error="%s: %s" % (type(e).__name__, e))
            return out
        D._deliberate_hop_conf = _hop
        return

    if name in ("steer_leaf", "silence_mantis"):
        from research.runners import _gnw_reentrant_metacog_gated_deliberation_derisk as D
        from research.runners import one_brain_composer as OBC
        orig_hop = D._deliberate_hop_conf
        orig_qp = OBC.OneBrainComposer.query_patient
        cue = {"x": None}

        def _qp(self, agent, action, *a, **k):          # the current cue x (PROPOSE is query_patient(x, rel))
            cue["x"] = agent
            return orig_qp(self, agent, action, *a, **k)
        OBC.OneBrainComposer.query_patient = _qp

        def _hop(bridge, xp, slots_dev, snap, target, distractors, nmda_attr=D.NMDA_ATTR_DEFAULT):
            if name == "steer_leaf" and target is None and cue["x"] == STEER_AT_CUE:
                # the leaf: production drives NOTHING here; drive the slot carrying STEER_CONCEPT at IGNITE_PA (the
                # current a proposal receives); a host relay that never reads the workspace cannot produce it
                _ev(rec, cue=cue["x"], target_in=None, target_out=STEER_CONCEPT, drive_pa=float(D.IGNITE_PA))
                target, distractors = STEER_CONCEPT, []
            elif name == "silence_mantis" and target == SILENCE_CONCEPT:
                _ev(rec, cue=cue["x"], target_in=target, target_out=None)
                target, distractors = None, []
            return orig_hop(bridge, xp, slots_dev, snap, target, distractors, nmda_attr=nmda_attr)
        D._deliberate_hop_conf = _hop
        return

    if name == "kselect_order":
        # F (host_share k-select "order"): when >= 2 stored facts match (agent, action) the host first-match is
        # replaced by the LAST match; the neural state (the store, the workspace) is untouched
        from research.runners import one_brain_composer as OBC
        orig_qp = OBC.OneBrainComposer.query_patient

        def _qp(self, agent, action, *a, **k):
            ans = orig_qp(self, agent, action, *a, **k)
            try:
                kb = getattr(self, "kb", []) or []
                matches = [f.get("patient") for f, _h in kb if f.get("agent") == agent and f.get("action") == action]
            except Exception as e:
                _ev(rec, error="%s: %s" % (type(e).__name__, e))
                return ans
            if len(matches) >= 2 and isinstance(matches[-1], str):
                _ev(rec, agent=agent, action=action, host=ans, matches=[str(m) for m in matches], returned=matches[-1])
                return matches[-1]
            return ans
        OBC.OneBrainComposer.query_patient = _qp
        return

    if name == "witness_flip":
        # F: the host `witnessed` boolean inverted on every MOVE (the first observe_event per organ is the PLACE)
        from research.runners import tom_false_belief_chat_organ as T
        orig = T.FalseBeliefChatOrgan.observe_event

        def _obs(self, loc, witnessed, *, lesion=False):
            n = getattr(self, "_s3_n_obs", 0) + 1
            self._s3_n_obs = n
            w_in = bool(witnessed)
            w_out = w_in if n == 1 else (not w_in)
            _ev(rec, n_obs=n, loc=int(loc), witnessed_in=w_in, witnessed_applied=w_out, flipped=bool(n > 1))
            return orig(self, loc, w_out, lesion=lesion)
        T.FalseBeliefChatOrgan.observe_event = _obs
        return

    if name == "steer_store":
        # G: on the FIRST query (q1) drive the `box` belief assembly (slot 1) with the organ's OWN write current
        # (helper_pa over drive_steps, the witness gate as for a witnessed write) before the read
        from research.runners import tom_false_belief_chat_organ as T
        orig = T.FalseBeliefChatOrgan.query

        def _q(self, *, lesion=False):
            if rec["calls"] == 0:
                try:
                    self._ensure_bridge()
                    self._isolated(lambda: self._write_event(int(STEER_STORE_SLOT), True, False))
                    _ev(rec, slot=int(STEER_STORE_SLOT), loc=STEER_STORE_LOC, helper_pa=getattr(self, "helper_pa", None),
                        drive_steps=getattr(self, "drive_steps", None))
                except Exception as e:
                    _ev(rec, error="%s: %s" % (type(e).__name__, e))
            return orig(self, lesion=lesion)
        T.FalseBeliefChatOrgan.query = _q
        return

    if name == "presence_clamp":
        # CANDIDATE G (spiking witness): presence silenced across S3's move (Lea; q3 -> bag), driven across S1's move
        # (Sally; q1 -> box). The tracked agent is read off the live scenario registry.
        from research.runners import tom_spiking_witness_organ as SW
        from webapp import false_belief_chat as FBC
        orig = SW.SpikingWitnessFalseBeliefOrgan.observe_event
        orig_run = SW.SpikingWitnessFalseBeliefOrgan._run

        def _tracked():
            try:
                scens = list(FBC._SESSIONS.values())
                return str(scens[-1].tracked_agent) if scens else None
            except Exception:
                return None

        def _obs(self, loc, *, lesion=False, silence_presence=False):
            n = getattr(self, "_s3_n_obs", 0) + 1
            self._s3_n_obs = n
            agent = _tracked()
            try:
                if n >= 2 and agent == "Lea":
                    _ev(rec, n_obs=n, agent=agent, mode="silence", loc=int(loc), pa=float(SW.SILENCE_PA))
                    return orig(self, loc, lesion=lesion, silence_presence=True)
                if n >= 2 and agent == "Sally":
                    _ev(rec, n_obs=n, agent=agent, mode="drive", loc=int(loc), pa=float(SW.EV_DRIVE_PA))

                    def _run_drive(steps, drive, les, silence=False, count=()):
                        extra = list(drive) + [(self._idx["presence"], float(SW.EV_DRIVE_PA))]
                        return orig_run(self, steps, extra, les, silence=silence, count=count)
                    self._run = _run_drive
                    try:
                        return orig(self, loc, lesion=lesion, silence_presence=silence_presence)
                    finally:
                        del self._run
            except Exception as e:
                _ev(rec, error="%s: %s" % (type(e).__name__, e))
            return orig(self, loc, lesion=lesion, silence_presence=silence_presence)
        SW.SpikingWitnessFalseBeliefOrgan.observe_event = _obs
        return

    if name == "steer_frame":
        # G: while the bat (a near-tie) is read, the `eat grub` frame assembly receives the organ's own context drive
        from research.runners import category_inherit_organ as CI
        orig_read = CI.CategoryInheritOrgan.read

        def _read(self, agent):
            if str(agent).lower() != STEER_MEMBER:
                return orig_read(self, agent)
            try:
                cs = self.context_slot.get(STEER_FRAME)
                if cs is None:
                    _ev(rec, agent=agent, error="frame slot for %s not recruited" % (STEER_FRAME,))
                    return orig_read(self, agent)
                c_dev = self._slot_idx("c", cs)
                b, f32 = self.bridge, self.xp.float32
                orig_step = b._run_one_simulation_step

                def _step():
                    b.cp_external_input_current[c_dev] = f32(CI.CONTEXT_DRIVE_PA)
                    orig_step()
                b._run_one_simulation_step = _step
                try:
                    out = orig_read(self, agent)
                finally:
                    del b._run_one_simulation_step
                _ev(rec, agent=agent, frame=list(STEER_FRAME), slot=int(cs), drive_pa=float(CI.CONTEXT_DRIVE_PA),
                    rates={"%s %s" % k: round(float(v), 5) for k, v in (out or {}).items()})
                return out
            except Exception as e:
                _ev(rec, agent=agent, error="%s: %s" % (type(e).__name__, e))
                return orig_read(self, agent)
        CI.CategoryInheritOrgan.read = _read
        return

    if name == "noise_taught":
        # INFO: Stage-2-style store noise on the taught recalls (r_*_1 at sigma[0], r_*_2 at sigma[1]); restored after
        import zlib
        import numpy as np
        from research.runners._emergent_graceful_degradation_derisk import _noise
        state = {"saved": None, "composer": None}

        def _pre(label, chat):
            m = re.match(r"^r_([a-z]+)_([12])$", label or "")
            if not m:
                return
            comp = getattr(getattr(chat, "inner", None), "composer", None)
            if comp is None or not getattr(comp, "store_conns", None):
                _ev(rec, error="no composer store on %s" % label)
                return
            sigma = float(NOISE_SIGMAS[int(m.group(2)) - 1])
            rng = np.random.default_rng([int(seed), zlib.crc32(label.encode())])
            state["saved"], state["composer"] = list(comp.store_conns), comp
            comp.store_conns = _noise(comp.store_conns, sigma, rng)
            comp._store_dirty = True
            comp._persistent_dirty = True
            _ev(rec, sigma=sigma, n_synapses=len(state["saved"]))

        def _post(label, chat):
            comp = state["composer"]
            if comp is not None and state["saved"] is not None:
                comp.store_conns = state["saved"]
                comp._store_dirty = True
                comp._persistent_dirty = True
            state["saved"], state["composer"] = None, None
        rec["_pre_turn"], rec["_post_turn"] = _pre, _post
        return

    if name == "kb_pop":
        # RECORD: after k1/k2 the corrected fact's HOST kb record is popped; the block's synapses and the organ untouched
        want = dict(CREDIT_CORRECTED)

        def _post(label, chat):
            if label not in ("k1", "k2"):
                return
            comp = getattr(getattr(chat, "inner", None), "composer", None)
            kb = getattr(comp, "kb", None)
            if not kb:
                _ev(rec, error="no kb after %s" % label)
                return
            f = kb[-1][0] if isinstance(kb[-1], (list, tuple)) else None
            if isinstance(f, dict) and want.get(f.get("agent")) == f.get("patient"):
                kb.pop()
                _ev(rec, popped=[f.get("agent"), f.get("action"), f.get("patient")], n_kb=len(kb))
            else:
                _ev(rec, error="last kb record after %s is not the corrected fact: %r" % (label, f))
        rec["_post_turn"] = _post
        return

    if name == "corr_freeze":
        # G: the organ's plasticity gates stay shut on the two correction facts only (observe skipped)
        from research.runners import category_inherit_organ as CI
        orig = CI.CategoryInheritOrgan.observe
        want = dict(CREDIT_CORRECTED)

        def _obs(self, agent, action, patient):
            a, p = str(agent).lower(), str(patient).lower()
            if want.get(a) == p:
                _ev(rec, agent=a, action=str(action).lower(), patient=p, gates="held shut (observe skipped)")
                return {"observed": False, "reason": "corr_freeze"}
            return orig(self, agent, action, patient)
        CI.CategoryInheritOrgan.observe = _obs
        return

    raise ValueError(name)


def _facts_of(chat):
    out = []
    try:
        for f in getattr(chat, "stored_facts", None) or []:
            if isinstance(f, (list, tuple)) and len(f) == 3 and all(isinstance(x, str) for x in f):
                out.append([f[0], f[1], f[2]])
    except Exception:
        pass
    return out


def _find_chat(srv, session):
    chat = None
    try:
        for _name, _obj in list(vars(srv).items()):
            if isinstance(_obj, dict):
                for key, val in list(_obj.items()):
                    if isinstance(key, tuple) and key and key[0] == session and hasattr(val, "stored_facts"):
                        chat = val
    except Exception:
        return None
    return chat


INTERNAL_TRACE_ATTRS = {"gnw_multistep": "_last_gnw_multistep"}


def capture_internal(chat, prev_ids=None):
    out = {}
    if chat is None:
        return out
    for name, attr in INTERNAL_TRACE_ATTRS.items():
        obj = getattr(chat, attr, None)
        if not isinstance(obj, dict):
            continue
        if prev_ids is not None:
            if prev_ids.get(name) == id(obj):
                continue
            prev_ids[name] = id(obj)
        out[name] = {k: v for k, v in obj.items() if not callable(v)}
    return out


def process_record(srv):
    rec = {}
    try:
        import resource
        rec["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:
        rec["peak_rss_mb"] = None
    try:
        b = srv._resolve_ltm_bundle()
        rec["ltm_bundle"] = os.path.basename(str(b).rstrip("/")) if b else None
    except Exception as e:
        rec["ltm_bundle"] = "unresolved: %s" % type(e).__name__
    return rec


def _clean_rec(rec):
    return None if rec is None else {k: v for k, v in rec.items() if not callable(v)}


def run_worker(seed, group, arm, out_path, labels=None):
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    env, patch, kind = arm_spec(group, arm)
    for k, v in dict(GROUPS[group]["env"], **env).items():
        os.environ[k] = v
    os.environ["BRAIN_CHAT_SEED"] = str(int(seed))
    observe_rec = install_patch(GROUPS[group].get("observe_patch"), seed)
    patch_rec = install_patch(patch, seed)
    hooks = [r for r in (observe_rec, patch_rec) if r]
    hash_ok = os.environ.get("PYTHONHASHSEED") == "0"
    corpus_present = os.path.exists(os.path.join(_REPO, "data", "corpus", "tinystories.txt"))
    t0 = time.time()
    from webapp.server import brain_chat, BrainChatRequest
    import_s = time.time() - t0
    turns = turns_for(group, seed, arm)
    if labels:
        keep = set(labels.split(","))
        turns = [t for t in turns if t[0] in keep]
    session = "s3_%s_%s_%d" % (group, arm.replace(".", "_"), int(seed))
    responses, timings, internal = {}, {}, {}
    builtin_facts = None
    from webapp import server as _srv
    prev_ids = {}
    chat = None
    for i, (label, msg, role) in enumerate(turns):
        PATCH_STATE["label"] = label
        for r in hooks:
            if callable(r.get("_pre_turn")):
                try:
                    r["_pre_turn"](label, chat)
                except Exception as e:
                    _ev(r, error="pre_turn %s: %s" % (type(e).__name__, e))
        t1 = time.time()
        try:
            r = brain_chat(BrainChatRequest(session=session, message=msg, brain="tiny-demo", renderer="stub",
                                            reset=(i == 0)))      # `rich` OMITTED = the production default path
            responses[label] = json.loads(r.body)
        except Exception as e:
            responses[label] = {"_error": "%s: %s" % (type(e).__name__, e)}
        timings[label] = round(time.time() - t1, 1)
        chat = _find_chat(_srv, session)
        for r in hooks:
            if callable(r.get("_post_turn")):
                try:
                    r["_post_turn"](label, chat)
                except Exception as e:
                    _ev(r, error="post_turn %s: %s" % (type(e).__name__, e))
        rec = capture_internal(chat, prev_ids)
        if rec:
            internal[label] = rec
        if builtin_facts is None:
            builtin_facts = _facts_of(chat) if chat is not None else []
    PATCH_STATE["label"] = None
    out = {"runner": "research/runners/_stage3_exit_draft.py", "prereg": PREREG, "prereg_registered": PREREG_REGISTERED,
           "seed": int(seed), "group": group, "arm": arm, "kind": kind, "criterion": GROUPS[group]["criterion"],
           "env": dict(GROUPS[group]["env"], **env), "patch": _clean_rec(patch_rec), "observe_patch": _clean_rec(observe_rec),
           "pythonhashseed_ok": hash_ok, "corpus_present": corpus_present, "builtin_facts": builtin_facts,
           "session": session, "turns": [{"label": l, "message": m, "role": r} for l, m, r in turns],
           "responses": responses, "internal": internal, "timings_s": timings, "import_s": round(import_s, 1),
           "wall_s": round(time.time() - t0, 1), "elapsed_seconds": round(time.time() - t0, 1),
           "expected_source": EXPECTED_SOURCE, "process": process_record(_srv),
           "derived_lex_from_production": DERIVED_LEX_FROM_PRODUCTION,
           "sources": sorted({str(r.get("source")) for r in responses.values() if isinstance(r, dict)})}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("[stage3 worker] seed=%d %s/%s -> %d turns, %.0fs -> %s" % (seed, group, arm, len(responses),
                                                                     out["wall_s"], out_path), flush=True)
    return 0


def run_shard(seed, group, arm, out_path, backend="numpy", labels=None, allow_gate_seed=False, prereg=None):
    env = dict(os.environ)
    env.update(BASE_ENV)
    env["SIM_BACKEND"] = backend
    if backend == "cupy":
        env.pop("CUDA_VISIBLE_DEVICES", None)
    part = out_path + ".part.json"
    cmd = [sys.executable, "-u", "-m", "research.runners._stage3_exit_draft", "--worker", "--seed", str(int(seed)),
           "--group", group, "--arm", arm, "--out", part]
    if labels:
        cmd += ["--labels", labels]
    if allow_gate_seed:
        cmd += ["--allow-gate-seed"]
    if prereg:
        cmd += ["--prereg", prereg]
    p = subprocess.run(cmd, env=env, cwd=_REPO)
    if p.returncode != 0 or not os.path.exists(part):
        print("[stage3 shard] worker FAILED rc=%s for seed=%s %s/%s" % (p.returncode, seed, group, arm), flush=True)
        return 1
    with open(part) as f:
        d = json.load(f)
    d["backend"] = backend
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(d, f, indent=1, default=str)
    for suffix in ("", ".prov.json"):
        try:
            os.remove(part + suffix)
        except OSError:
            pass
    return 0


# ══════════════════════════════════════════════ READERS (reply level first) ════════════════════════════════════════
def _primary(resp):
    """The reply-level primary fields (docs/TERMS.md): what the user sees on that turn. `derived_from` is included so a
    lesion discrimination never reads only `recalled_svo` (an intact derived answer and a lesion abstain both carry
    None there; coordinator finding 5)."""
    if not isinstance(resp, dict) or "_error" in resp:
        return None
    df = resp.get("derived_from") or None
    return {"answer": str(resp.get("answer")), "abstained": bool(resp.get("abstained")),
            "recalled_svo": (list(resp["recalled_svo"]) if resp.get("recalled_svo") else None),
            "derived": bool(resp.get("derived")),
            "derived_from": ([list(x) for x in df] if isinstance(df, (list, tuple)) else None)}


_FRAME_RE = re.compile(r"\bthe ([a-z]+) ([a-z]+) (?:(in|on|at|to) )?the ([a-z]+)\b", re.IGNORECASE)


def frames(answer):
    return [(m.group(1).lower(), m.group(2).lower(), (m.group(3) or None), m.group(4).lower())
            for m in _FRAME_RE.finditer(str(answer or ""))]


def _patient(resp):
    """The patient the reply names: the recalled SVO's, else a derived answer's terminal, else the last frame's."""
    p = _primary(resp)
    if not p:
        return None
    if p["recalled_svo"] and len(p["recalled_svo"]) == 3:
        return str(p["recalled_svo"][2]).lower()
    if p["derived"]:
        df = p["derived_from"] or []
        if df and len(df[-1]) == 3:
            return str(df[-1][2]).lower()
        fr = frames(p["answer"])
        if fr:
            return fr[-1][3]
    return None


def _abstained(resp):
    p = _primary(resp)
    return None if p is None else p["abstained"]


def _hedged(resp):
    p = _primary(resp)
    return None if p is None else (HEDGE_TEXT in p["answer"])


def _guess(resp):
    p = _primary(resp)
    return None if p is None else ("guess" in p["answer"].lower())


def level(resp):
    """C2's reply-level confidence report: 2 = answered without the hedge; 1 = answered with the production hedge;
    0 = abstain (the opt-out). None on an errored turn."""
    p = _primary(resp)
    if p is None:
        return None
    if p["abstained"]:
        return 0
    return 1 if HEDGE_TEXT in p["answer"] else 2


def _word_in(word, text):
    return re.search(r"\b%s\b" % re.escape(word), str(text or ""), re.IGNORECASE) is not None


def derived_framed(resp, agent, terminal, verb="chase", n_hops=None, max_hops=None):
    """The HONESTY clause (PLAN.md 3.2), scored from the RECORD: `derived` True; `recalled_svo` None; `derived_from`
    runs from the asked agent to the named terminal; its length equals `n_hops` when given (the chase's own
    resolved_hops), or lies in [2, max_hops]; and, at the reply level, a PRODUCTION frame string (imported) appears
    before the first mention of the claim. Never sentence by sentence (the production lead sits in its own sentence)."""
    p = _primary(resp)
    if p is None or not p["derived"] or p["recalled_svo"] is not None:
        return False
    df = p["derived_from"] or []
    if not df or len(df[0]) != 3 or len(df[-1]) != 3:
        return False
    if str(df[0][0]).lower() != agent or str(df[-1][2]).lower() != terminal:
        return False
    if n_hops is not None and len(df) != int(n_hops):
        return False
    if n_hops is None and (len(df) < 2 or (max_hops is not None and len(df) > int(max_hops))):
        return False
    ans = p["answer"]
    claim = _fx(agent, verb, terminal)
    ci = ans.lower().find(claim.lower())
    leads = [ans.find(s) for s in DERIVED_LEADS if s and ans.find(s) >= 0]
    if DERIVED_SWAP_TAIL and DERIVED_SWAP_TAIL in ans and leads:
        return True                                           # the #129 swap wording ("I believe ..., but I reasoned ...")
    if not leads:
        return False
    return ci < 0 or min(leads) < ci


def transitive_trace(resp):
    t = resp.get("transitive_trace") if isinstance(resp, dict) else None
    return t if isinstance(t, dict) else None


def n_advance(resp):
    t = transitive_trace(resp)
    if not t:
        return None
    return sum(1 for e in (t.get("chase") or []) if isinstance(e, dict) and e.get("action") == "ADVANCE")


def multistep_trace(shard, label):
    rec = ((shard.get("internal") or {}).get(label) or {}).get("gnw_multistep")
    return rec if isinstance(rec, dict) and "resolved_hops" in rec else None


def belief_named(resp):
    p = _primary(resp)
    if p is None:
        return None
    m = re.search(r"will look in the (\w+)", p["answer"])
    return m.group(1).lower() if m else None


def fb_info(resp):
    t = resp.get("false_belief_tom") if isinstance(resp, dict) else None
    return t if isinstance(t, dict) else None


def ci_read(resp):
    """The category-inheritance organ's read trace on this turn (`category_inherit.read`), or None."""
    t = resp.get("category_inherit") if isinstance(resp, dict) else None
    r = t.get("read") if isinstance(t, dict) else None
    return r if isinstance(r, dict) else None


def organ_errors(resp):
    out = []
    if not isinstance(resp, dict):
        return out
    for k, v in resp.items():
        if isinstance(v, dict):
            if v.get("error"):
                out.append(k)
            for k2, v2 in v.items():
                if isinstance(v2, dict) and v2.get("error"):
                    out.append("%s.%s" % (k, k2))
    return out


def patch_applied(rec):
    """R2: an in-process patch counts as applied only when it recorded at least one error-free event."""
    if not isinstance(rec, dict):
        return True                      # an env-flag arm: production reads the flag per call
    if rec.get("install_error"):
        return False
    return int(rec.get("n_applied", 0)) > 0


def arm_validity(group, arm, shard):
    """Why one COUNTED arm cannot be scored (empty list = valid). Stage-2 section 7 + R2."""
    why = []
    if not isinstance(shard, dict) or "responses" not in shard:
        return ["%s: arm missing or unreadable" % arm]
    resp = shard.get("responses") or {}
    errored = sorted(l for l, r in resp.items() if not isinstance(r, dict) or "_error" in r)
    if errored:
        why.append("%s: errored turn(s) %s" % (arm, ",".join(errored)[:120]))
    org = {l: organ_errors(r) for l, r in resp.items()}
    org = {l: v for l, v in org.items() if v}
    for l, rec in (shard.get("internal") or {}).items():
        ms = (rec or {}).get("gnw_multistep") or {}
        if str(ms.get("reason") or "").startswith("error"):
            org.setdefault(l, []).append("gnw_multistep")
    if org:
        why.append("%s: organ error %s" % (arm, "; ".join("%s:%s" % (l, "+".join(v)) for l, v in sorted(org.items()))[:160]))
    srcs = set(shard.get("sources") or [])
    if srcs != {EXPECTED_SOURCE}:
        why.append("%s: unexpected brain source %s" % (arm, sorted(srcs)))
    if not shard.get("pythonhashseed_ok"):
        why.append("%s: PYTHONHASHSEED not 0" % arm)
    if not shard.get("corpus_present", True):
        why.append("%s: corpus absent: corpus-learned production organs degraded" % arm)
    _env, patch, _kind = arm_spec(group, arm)
    if patch is not None and not patch_applied(shard.get("patch")):
        why.append("%s: patch %s not applied (calls without error = 0)" % (arm, patch))
    if GROUPS[group].get("observe_patch") and shard.get("observe_patch") is not None \
            and not patch_applied(shard.get("observe_patch")):
        why.append("%s: observation patch %s recorded nothing" % (arm, GROUPS[group]["observe_patch"]))
    return why


def null_clean(a, b, labels):
    diffs = []
    for l in labels:
        pa, pb = _primary((a or {}).get(l)), _primary((b or {}).get(l))
        if pa is None or pb is None:
            diffs.append((l, "missing_or_error"))
        elif pa != pb:
            diffs.append((l, "differs"))
    return (not diffs), diffs


def reply_differs(a, b, label):
    pa, pb = _primary((a or {}).get(label)), _primary((b or {}).get(label))
    if pa is None or pb is None:
        return None
    return pa != pb


def n_field_diffs(a, b, labels):
    n = 0
    for l in labels:
        pa, pb = _primary((a or {}).get(l)), _primary((b or {}).get(l))
        if pa is None or pb is None:
            continue
        n += sum(1 for k in pa if pa[k] != pb[k])
    return n


# ══════════════════════════════════════════════ SCORING (pure functions over shard dicts) ══════════════════════════
def _yes(resp, terminal):
    return _abstained(resp) is False and _patient(resp) == terminal


def score_chain(shard, permuted=False, m_adj_fix=None):
    """C1. See PLAN.md section 3. Every clause reads derived/derived_from beside recalled_svo."""
    if m_adj_fix is None:
        m_adj_fix = M_ADJ_TOLD_FIX_IN_PIN
    resp = shard.get("responses") or {}
    p3, p2, madj, mx, mrev = (resp.get(k) or {} for k in ("p3", "p2", "m_adj", "m_cross", "m_rev"))
    rfork, rall, cdir, ctail, cdog = (resp.get(k) or {} for k in ("r_fork", "r_all", "c_direct", "c_tail", "c_dog"))
    ms = multistep_trace(shard, "r_all")
    hops_all = ms.get("resolved_hops") if ms else None
    ok = {}
    ok["p3"] = _yes(p3, "flea") and derived_framed(p3, "puma", "flea", max_hops=n_advance(p3) or 3)
    if permuted:
        ok["p2_abstains"] = _abstained(p2) is True
        ok["m_cross"] = _yes(mx, "flea") and derived_framed(mx, "shrew", "flea", max_hops=n_advance(mx) or 2)
        ok["fork_honest"] = _yes(rfork, "flea")                       # both branches converge: one honest terminal
    else:
        ok["p2"] = _yes(p2, "slug") and derived_framed(p2, "puma", "slug", max_hops=n_advance(p2) or 2)
        ok["m_cross"] = _abstained(mx) is True
        a = str(rfork.get("answer") or "")
        ok["fork_honest"] = (_abstained(rfork) is True) or all(_word_in(t, a) for t in CHAIN_FORK_TERMINALS)
    ok["m_adj_told_plain"] = _yes(madj, "gecko") and not bool(madj.get("derived"))
    if m_adj_fix:
        ok["m_adj"] = ok["m_adj_told_plain"]
    ok["m_rev"] = _abstained(mrev) is True
    ok["r_all"] = _yes(rall, "flea") and derived_framed(rall, "gecko", "flea", n_hops=hops_all)
    ok["c_direct"] = _patient(cdir) == "mantis" and not bool(cdir.get("derived"))
    ok["c_tail"] = _abstained(ctail) is True
    ok["c_dog"] = _patient(cdog) == "cat"
    yes_probe = mx if permuted else p3
    tt = transitive_trace(yes_probe) or {}
    ok["via_workspace"] = bool(tt.get("route") not in (None, "direct")) and (n_advance(yes_probe) or 0) >= 2 \
        and ms is not None and (hops_all or 0) >= 2
    counted = [k for k in ok if k not in ("m_adj_told_plain",)]
    ok["all"] = all(ok[k] for k in counted)
    ok["routes"] = {l: ((transitive_trace(resp.get(l) or {}) or {}).get("route")) for l in ("p3", "p2", "m_adj", "m_cross", "m_rev")}
    ok["n_advance"] = {l: n_advance(resp.get(l) or {}) for l in ("p3", "p2", "m_cross")}
    ok["r_all_internal"] = ({"resolved_hops": hops_all, "terminal": ms.get("terminal"), "lesion": ms.get("lesion"),
                             "reason": ms.get("reason")} if ms else None)
    ok["patients"] = {l: _patient(resp.get(l) or {}) for l in ("p3", "p2", "m_adj", "r_fork", "r_all", "c_direct")}
    ok["derived_flags"] = {l: bool((resp.get(l) or {}).get("derived")) for l in ("p3", "p2", "m_adj", "r_all", "c_direct")}
    return ok


def type2_auroc(levels, correct):
    """Type-2 AUROC: P(L_correct > L_incorrect) + 0.5 P(tie) over every (correct, incorrect) pair. None if either side
    is empty."""
    pos = [l for l, c in zip(levels, correct) if c and l is not None]
    neg = [l for l, c in zip(levels, correct) if (not c) and l is not None]
    if not pos or not neg:
        return None
    s = 0.0
    for a in pos:
        for b in neg:
            s += 1.0 if a > b else (0.5 if a == b else 0.0)
    return s / float(len(pos) * len(neg))


def score_meta(shard, permuted=False):
    """C2. The type-2 AUROC over the GUESS stratum (8 members) with the frozen world table; taught recalls and the
    built-in beside it. See PLAN.md section 4."""
    resp = shard.get("responses") or {}
    world = META_WORLD_PERM if permuted else META_WORLD
    consistent = META_CONSISTENT_PERM if permuted else META_CONSISTENT
    conflicting = META_CONFLICTING_PERM if permuted else META_CONFLICTING
    items = list(consistent) + list(conflicting)
    levels, correct, guess, patients, reads = {}, {}, {}, {}, {}
    for a in items:
        r = resp.get("q_%s" % a) or {}
        levels[a] = level(r)
        patients[a] = _patient(r)
        correct[a] = (patients[a] == world[a]) and (_abstained(r) is False)
        guess[a] = _guess(r)
        reads[a] = ci_read(r)
    L = [levels[a] for a in items]
    C = [correct[a] for a in items]
    auroc = type2_auroc(L, C)
    n_correct = sum(1 for a in items if correct[a])
    ok = {"auroc": auroc, "n_correct_guess": n_correct, "n_items": len(items),
          "capability_present": n_correct >= MIN_CORRECT_GUESS,
          "auroc_ok": (auroc is not None and n_correct >= MIN_CORRECT_GUESS and auroc >= AUROC_BAR),
          "taught": all(_patient(resp.get("r_%s_1" % a) or {}) == world[a] for a in META_TAUGHT),
          "c_dog": _patient(resp.get("c_dog") or {}) == "cat"}
    ok["all"] = bool(ok["auroc_ok"] and ok["taught"] and ok["c_dog"])
    ok["levels"] = levels
    ok["correct"] = correct
    ok["patients"] = patients
    ok["guess_flag"] = guess
    ok["optout_consistent"] = sum(1 for a in consistent if levels[a] == 0)
    ok["optout_conflicting"] = sum(1 for a in conflicting if levels[a] == 0)
    ok["hedged_any"] = any(levels[a] == 1 for a in items) or any(
        _hedged(resp.get(l) or {}) for l in META_TAUGHT_LABELS + META_TAUGHT_LABELS_2)
    ok["taught_levels"] = {l: level(resp.get(l) or {}) for l in META_TAUGHT_LABELS + META_TAUGHT_LABELS_2}
    ok["reads_present"] = sum(1 for a in items if reads[a] is not None)
    return ok


def score_tom(shard, permuted=False):
    """C3. Each query names the expected location in the "<X> will look in the <L>" frame; the mismatch control names
    none; the built-in survives. See PLAN.md section 5."""
    resp = shard.get("responses") or {}
    exp = TOM_EXPECT_PERM if permuted else TOM_EXPECT
    ok = {}
    for q in TOM_QUERIES:
        ok[q] = belief_named(resp.get(q) or {}) == exp[q]
    ok["m1"] = belief_named(resp.get("m1") or {}) is None and _primary(resp.get("m1") or {}) is not None
    ok["c_dog"] = _patient(resp.get("c_dog") or {}) == "cat"
    ok["all"] = all(ok[k] for k in list(TOM_QUERIES) + ["m1", "c_dog"])
    ok["named"] = {q: belief_named(resp.get(q) or {}) for q in TOM_QUERIES}
    ok["fb"] = {}
    for q in TOM_QUERIES:
        t = fb_info(resp.get(q) or {}) or {}
        ok["fb"][q] = {k: t.get(k) for k in ("reason", "belief_location", "reality_location", "store_lesioned",
                                             "belief_rates", "presence_rate")}
    return ok


def score_credit(shard, permuted=False):
    """C4. The held-out member names the corrected food; the disjoint-feature control does not; the corrected and
    the uncorrected told facts recall; the built-in survives. See PLAN.md section 6."""
    resp = shard.get("responses") or {}
    want = "nut" if permuted else "seed"
    H, J = resp.get("H") or {}, resp.get("J") or {}
    ok = {"H": _patient(H) == want and _abstained(H) is False,
          "J": _patient(J) != want,
          "c_lark": _patient(resp.get("c_lark") or {}) == want,
          "c_robin": _patient(resp.get("c_robin") or {}) == "worm",
          "c_dog": _patient(resp.get("c_dog") or {}) == "cat"}
    ok["all"] = all(ok[k] for k in ("H", "J", "c_lark", "c_robin", "c_dog"))
    ok["H_patient"] = _patient(H)
    ok["H_guess"] = _guess(H)
    ok["H_level"] = level(H)
    ok["H_read"] = ci_read(H)
    ok["J_patient"] = _patient(J)
    ok["e1_patient"] = _patient(resp.get("e1") or {})
    return ok


SCORERS = {"chain": score_chain, "meta": score_meta, "tom": score_tom, "credit": score_credit}
# the turns a lesion is scored on (a primary reply field must differ from intact_a on at least one of them)
SCORED_LABELS = {
    "chain": {"ws_lesion": ["p3", "p2", "r_all"]},
    "meta": {"inherit_lesion": ["q_lark", "q_finch", "q_dace", "q_roach"]},
    "tom": {"store_lesion": ["q1", "q2", "q3", "q4"]},
    "credit": {"no_teacher": ["H"], "learn_lesion": ["H"]},
}
# the turns a lesion must NOT change (dissociation: the cut is on the mechanism, not on recall)
UNCHANGED_LABELS = {"chain": {"ws_lesion": ["c_direct"]}, "meta": {"inherit_lesion": list(META_TAUGHT_LABELS)},
                    "tom": {}, "credit": {}}


def lesion_engaged(group, les, shard):
    """Validity rule 2: the lesion's target acted on a scored turn with the lesion flag read as on (else UNDEFINED)."""
    resp = shard.get("responses") or {}
    if group == "chain":
        t = transitive_trace(resp.get("p3") or {}) or {}
        ms = multistep_trace(shard, "r_all") or {}
        return bool(t.get("lesioned")) and bool(ms.get("lesion"))
    if group == "meta":
        return any((ci_read(resp.get(l) or {}) or {}).get("lesion") == "all" for l in SCORED_LABELS["meta"][les])
    if group == "tom":
        return any(bool((fb_info(resp.get(q) or {}) or {}).get("store_lesioned")) for q in TOM_QUERIES)
    if group == "credit":
        if les == "no_teacher":
            return all(str((resp.get(k) or {}).get("answer") or "") != "" for k in ("k1", "k2"))
        return (ci_read(resp.get("H") or {}) or {}).get("lesion") == "lateral_learn"
    return False


# ══════════════════════════════════════════════ READ-OUT INVARIANCE (offline, from recorded rates) ═════════════════
def _cls(rates, thr):
    return min(2, sum(1 for r in rates if r >= thr))


def ri_chain(observe_rec, labels):
    """The chase's recorded per-slot rates re-decided at THR x/÷ RI_FACTOR: the ignition class (0 commit / 1 advance /
    >=2 conflict) must be unchanged. Returns (fraction_stable, n_cycles)."""
    evs = [e for e in ((observe_rec or {}).get("events") or []) if e.get("label") in labels and e.get("rates")]
    if not evs:
        return None, 0
    stable = 0
    for e in evs:
        thr = float(e.get("thr") or 0.0)
        base = _cls(e["rates"], thr)
        if all(_cls(e["rates"], thr * f) == base for f in (1.0 / RI_FACTOR, RI_FACTOR)):
            stable += 1
    return stable / float(len(evs)), len(evs)


def _decide_frames(rates, verb, floor, margin):
    adm = sorted(((float(r), fr.split(" ", 1)[1]) for fr, r in (rates or {}).items() if fr.split(" ", 1)[0] == verb),
                 key=lambda t: (-t[0], t[1]))
    if not adm or adm[0][0] < floor:
        return None
    if len(adm) > 1 and adm[1][0] >= floor and adm[0][0] < margin * adm[1][0]:
        return None
    return adm[0][1]


def ri_inherit(shard, labels, verb="eat"):
    """Each guess item re-decided from the organ's recorded rates over the TIE_MARGIN x FLOOR_RATE grid: answered-vs-
    abstained must be unchanged at every grid point. Returns (fraction_stable, n_items_with_read)."""
    resp = shard.get("responses") or {}
    n, stable = 0, 0
    for l in labels:
        r = resp.get(l) or {}
        rd = ci_read(r)
        if not rd or not isinstance(rd.get("rates"), dict):
            continue
        n += 1
        answered = (_abstained(r) is False)
        if all(((_decide_frames(rd["rates"], verb, fl, mg) is not None) == answered)
               for mg in RI_TIE_MARGINS for fl in RI_FLOOR_RATES):
            stable += 1
    return (stable / float(n) if n else None), n


def ri_tom(shard):
    """The belief argmax vs a rate-ratio >= 2 read-out on every scored query. Returns (fraction_agree, n)."""
    resp = shard.get("responses") or {}
    n, agree = 0, 0
    for q in TOM_QUERIES:
        t = fb_info(resp.get(q) or {}) or {}
        br = t.get("belief_rates")
        if not isinstance(br, dict) or not br:
            continue
        n += 1
        vals = sorted(((float(v), str(k)) for k, v in br.items()), key=lambda x: -x[0])
        top = vals[0]
        second = vals[1][0] if len(vals) > 1 else 0.0
        ratio_slot = top[1] if (top[0] > 0 and top[0] >= 2.0 * second) else None
        argmax_slot = str(t.get("belief_loc_idx")) if t.get("belief_loc_idx") is not None else top[1]
        if ratio_slot == argmax_slot:
            agree += 1
    return (agree / float(n) if n else None), n


# ══════════════════════════════════════════════ OWNERSHIP (per group, per seed) ════════════════════════════════════
def _outcome(resp):
    """The seam outcome of a reply: ('abstain', None) or ('yes', patient)."""
    if _primary(resp) is None:
        return None
    return ("abstain", None) if _abstained(resp) else ("yes", _patient(resp))


def f_event(seam, seed, label, intact_resp, f_resp, alternative, applied):
    """One F event in the host-share schema (S and D are not measured here; `f_outcome` = returned_to_neural when
    the F-arm reply equals intact's). `alternative` is the preregistered host alternative outcome. Not live when
    intact's outcome already equals the alternative (no real choice; ownership design section 0)."""
    oi, of = _outcome(intact_resp), _outcome(f_resp)
    ev = {"seam": seam, "seed": seed, "label": label, "live": False, "F": None, "S": None, "D": None, "f_outcome": None,
          "intact": oi, "under_F": of, "alternative": alternative}
    if oi is None or oi == alternative:
        ev["why"] = "not live: intact outcome equals the host alternative" if oi is not None else "intact errored"
        return ev
    ev["live"] = True
    if not applied or of is None:
        ev["why"] = "F undefined: patch not applied or errored turn"
        return ev
    if of == alternative:
        ev["F"] = 1
    else:
        ev["F"] = 0
        ev["f_outcome"] = "returned_to_neural" if of == oi else "masked"
    return ev


def _own_verdict(undefined, host, g_live, g_ok, ri_ok, capability_present, readout):
    if undefined:
        return "UNDEFINED"
    if host or (g_live and g_ok is False) or ri_ok is False:
        return "HOST"
    if not g_live:
        return "N/A" if not capability_present else "UNDEFINED"
    return "OWNED(read-out)" if readout else "OWNED"


def ownership_chain(seed, have, row):
    ia = have.get("intact_a") or {}
    ra = ia.get("responses") or {}
    out = {"seam": "C1 k-select order", "events": [], "reasons": [], "host": [], "undefined": []}
    order = have.get("order_swap")
    if order is None or "responses" not in order:
        out["undefined"].append("order_swap arm missing")
        out["order_change"] = None
    else:
        d = [reply_differs(ra, order["responses"], l) for l in ("p3", "p2", "m_adj", "r_fork")]
        out["order_change"] = any(x is True for x in d)
        out["order_diffs"] = dict(zip(("p3", "p2", "m_adj", "r_fork"), d))
        if out["order_change"]:
            out["host"].append("ORDER: the kb order decides (order_swap changed %s)" % [l for l, x in out["order_diffs"].items() if x])
    F = have.get("kselect_order")
    f_applied = F is not None and patch_applied((F or {}).get("patch"))
    rf = (F or {}).get("responses") or {}
    for label, alt in (("p3", ("yes", "flea")), ("p2", ("abstain", None))):
        ev = f_event(out["seam"], seed, label, ra.get(label), rf.get(label), alt, f_applied)
        out["events"].append(ev)
        if ev["live"] and ev["F"] is None:
            out["undefined"].append("F undefined on %s (%s)" % (label, ev.get("why")))
        if ev.get("F") == 1:
            out["host"].append("F=1 on %s: the host first-match decided" % label)
    G = have.get("silence_mantis")
    g_live = G is not None and patch_applied((G or {}).get("patch"))
    out["g_live"] = g_live
    if G is None:
        out["undefined"].append("silence_mantis arm missing")
    out["g"] = None
    if g_live:
        rg = G["responses"]
        changed_all = reply_differs(ra, rg, "r_all")
        p2_same = reply_differs(ra, rg, "p2") is False
        p3_ok = (not row["intact"].get("p3")) or reply_differs(ra, rg, "p3") is True
        out["g"] = bool(changed_all and p2_same and p3_ok)
        out["g_detail"] = {"r_all_changed": changed_all, "p2_unchanged": p2_same, "p3_rule": p3_ok}
    frac, n = ri_chain(ia.get("observe_patch"), ["p3", "p2", "r_all"])
    out["ri"] = {"fraction_stable": frac, "n_cycles": n, "factor": RI_FACTOR}
    ri_ok = None if frac is None else (frac >= RI_MIN_FRACTION)
    if frac is None:
        out["undefined"].append("RI: no chase rates recorded on the scored turns")
    elif not ri_ok:
        out["host"].append("RI fails: the ignition class moves with THR on %.0f%% of cycles" % (100 * (1 - frac)))
    out["verdict"] = _own_verdict(out["undefined"], out["host"], g_live, out["g"], ri_ok, bool(row.get("correct")), False)
    return out


def ownership_meta(seed, have, row):
    ia = have.get("intact_a") or {}
    ra = ia.get("responses") or {}
    out = {"seam": "C2 opt-out margin", "events": [], "reasons": [], "host": [], "undefined": []}
    frac, n = ri_inherit(ia, META_GUESS_LABELS)
    out["ri"] = {"fraction_stable": frac, "n_items_with_read": n, "grid": {"tie_margin": RI_TIE_MARGINS, "floor_rate": RI_FLOOR_RATES}}
    ri_ok = None
    if n < 7:
        out["undefined"].append("RI: the organ's read recorded on only %d of 8 guess items" % n)
    else:
        ri_ok = frac >= RI_MIN_FRACTION
        if not ri_ok:
            out["host"].append("RI fails: the host margin/floor decides %.0f%% of the guess items" % (100 * (1 - frac)))
    G = have.get("steer_frame")
    g_live = G is not None and patch_applied((G or {}).get("patch"))
    out["g_live"] = g_live
    out["g"] = None
    if G is None:
        out["undefined"].append("steer_frame arm missing")
    if g_live:
        rg = G["responses"]
        bat = rg.get("q_%s" % STEER_MEMBER) or {}
        out["g"] = bool(_abstained(bat) is False and _patient(bat) == STEER_FRAME[1]
                        and reply_differs(ra, rg, "q_lark") is False)
        out["g_detail"] = {"bat_patient": _patient(bat), "bat_abstained": _abstained(bat),
                           "lark_unchanged": reply_differs(ra, rg, "q_lark") is False}
    out["verdict"] = _own_verdict(out["undefined"], out["host"], g_live, out["g"], ri_ok,
                                  bool((row.get("intact") or {}).get("capability_present")), True)
    return out


def ownership_tom(seed, have, row):
    ia = have.get("intact_a") or {}
    ra = ia.get("responses") or {}
    out = {"seam": "C3 witnessing", "events": [], "reasons": [], "host": [], "undefined": []}
    F = have.get("witness_flip")
    f_applied = F is not None and patch_applied((F or {}).get("patch"))
    rf = (F or {}).get("responses") or {}
    for q in TOM_QUERIES:
        oi = belief_named(ra.get(q) or {})
        of = belief_named(rf.get(q) or {}) if q in rf else None
        alt = TOM_WITNESS_FLIP_ALT[q]
        ev = {"seam": out["seam"], "seed": seed, "label": q, "live": False, "F": None, "S": None, "D": None,
              "f_outcome": None, "intact": oi, "under_F": of, "alternative": alt}
        if oi is None or oi == alt:
            ev["why"] = "not live: intact names the alternative or nothing"
        else:
            ev["live"] = True
            if not f_applied or q not in rf or _primary(rf.get(q)) is None:
                ev["why"] = "F undefined: patch not applied or errored turn"
                out["undefined"].append("F undefined on %s" % q)
            elif of == alt:
                ev["F"] = 1
                out["host"].append("F=1 on %s: the host `witnessed` boolean decided" % q)
            else:
                ev["F"] = 0
                ev["f_outcome"] = "returned_to_neural" if of == oi else "masked"
        out["events"].append(ev)
    G = have.get("steer_store")
    g_live = G is not None and patch_applied((G or {}).get("patch"))
    out["g_live"] = g_live
    out["g"] = None
    if G is None:
        out["undefined"].append("steer_store arm missing")
    if g_live:
        out["g"] = belief_named(G["responses"].get("q1") or {}) == STEER_STORE_LOC
        out["g_detail"] = {"q1_named": belief_named(G["responses"].get("q1") or {})}
    frac, n = ri_tom(ia)
    out["ri"] = {"fraction_agree": frac, "n_queries": n}
    ri_ok = None
    if n < len(TOM_QUERIES):
        out["undefined"].append("RI: belief rates recorded on only %d of %d queries" % (n, len(TOM_QUERIES)))
    else:
        ri_ok = frac >= RI_MIN_FRACTION
        if not ri_ok:
            out["host"].append("RI fails: the argmax and a ratio>=2 read-out disagree")
    out["verdict"] = _own_verdict(out["undefined"], out["host"], g_live, out["g"], ri_ok, bool(row.get("correct")), True)
    return out


def ownership_credit(seed, have, row):
    ia = have.get("corrected_a") or {}
    ra = ia.get("responses") or {}
    out = {"seam": "C4 kb record", "events": [], "reasons": [], "host": [], "undefined": []}
    R = have.get("kb_pop")
    r_applied = R is not None and patch_applied((R or {}).get("patch"))
    ev = {"seam": out["seam"], "seed": seed, "label": "H", "live": True, "F": None, "S": None, "D": None,
          "f_outcome": None, "intact": _outcome(ra.get("H")), "under_F": _outcome(((R or {}).get("responses") or {}).get("H")),
          "alternative": "H changes when the host record is gone"}
    if not r_applied or ev["under_F"] is None or ev["intact"] is None:
        ev["live"] = False
        ev["why"] = "RECORD undefined: patch not applied or errored turn"
        out["undefined"].append("kb_pop undefined")
    else:
        changed = reply_differs(ra, R["responses"], "H")
        ev["F"] = 1 if changed else 0
        ev["f_outcome"] = None if changed else "returned_to_neural"
        if changed:
            out["host"].append("F=1 on H: the host kb record supplied the held-out answer")
    out["events"].append(ev)
    G = have.get("corr_freeze")
    NT = have.get("no_teacher")
    g_live = G is not None and patch_applied((G or {}).get("patch")) and NT is not None and "responses" in NT
    out["g_live"] = g_live
    out["g"] = None
    if G is None:
        out["undefined"].append("corr_freeze arm missing")
    cap = bool((row.get("intact") or {}).get("H"))
    if g_live and cap:
        changed = reply_differs(ra, G["responses"], "H")
        same_as_nt = reply_differs(G["responses"], NT["responses"], "H") is False
        out["g"] = bool(changed and same_as_nt)
        out["g_detail"] = {"H_changed": changed, "H_equals_no_teacher": same_as_nt}
    elif g_live and not cap:
        g_live = False               # the capability is absent: the clamp has nothing to revert (N/A)
        out["g_live"] = False
    frac, n = ri_inherit(ia, ["H"])
    out["ri"] = {"fraction_stable": frac, "n_items_with_read": n}
    ri_ok = None
    if n < 1:
        if cap:
            out["undefined"].append("RI: no organ read recorded on H")
    else:
        ri_ok = frac >= RI_MIN_FRACTION
        if not ri_ok:
            out["host"].append("RI fails: H's read moves with the host margin/floor")
    out["verdict"] = _own_verdict(out["undefined"], out["host"], g_live, out["g"], ri_ok, cap, True)
    return out


OWNERSHIP = {"chain": ownership_chain, "meta": ownership_meta, "tom": ownership_tom, "credit": ownership_credit}
STEER_ARM = {"chain": "steer_leaf", "meta": "steer_frame", "tom": "steer_store", "credit": "corr_freeze"}


def steer_effect(group, have, row):
    """The content-specific intervention's predicted specific answer (capability clause 'steerable')."""
    arm = STEER_ARM[group]
    S = have.get(arm)
    if S is None or "responses" not in S:
        return None, False
    applied = patch_applied(S.get("patch"))
    r = S["responses"]
    if group == "chain":
        rall = r.get("r_all") or {}
        return (_patient(rall) == STEER_CONCEPT and _abstained(rall) is False), applied
    if group == "meta":
        bat = r.get("q_%s" % STEER_MEMBER) or {}
        return (_abstained(bat) is False and _patient(bat) == STEER_FRAME[1]), applied
    if group == "tom":
        return (belief_named(r.get("q1") or {}) == STEER_STORE_LOC), applied
    if group == "credit":
        ia = have.get("corrected_a") or {}
        return (reply_differs(ia.get("responses") or {}, r, "H") is True), applied
    return None, applied


def shard_path(out_dir, seed, group, arm):
    return os.path.join(out_dir, "s%d" % int(seed), "%s__%s.json" % (group, arm.replace(":", "_")))


def _load_shards(root, seeds):
    out = {}
    for seed in seeds:
        for group in GROUPS:
            for arm in arms_for(group):
                p = shard_path(root, seed, group, arm)
                if os.path.exists(p):
                    try:
                        with open(p) as f:
                            out[(int(seed), group, arm)] = json.load(f)
                    except Exception as e:
                        out[(int(seed), group, arm)] = {"_load_error": repr(e)}
    return out


def _labels(group, seed, arm=None):
    return [t[0] for t in turns_for(group, seed, arm or _intact_names(group)[0])]


# ══════════════════════════════════════════════ AGGREGATE ══════════════════════════════════════════════════════════
def aggregate(root, seeds=GATE_SEEDS, informational=True):
    shards = _load_shards(root, seeds)
    seeds = [int(s) for s in seeds]
    per_group, missing, panel_events = {}, [], []
    for group in GROUPS:
        g = GROUPS[group]
        ia_name, ib_name = _intact_names(group)
        rows, defined = {}, []
        for seed in seeds:
            need = counted_arms(group)
            have = {a: shards.get((seed, group, a)) for a in arms_for(group)}
            for a in need:
                if have[a] is None:
                    missing.append(shard_path(root, seed, group, a))
            row = {"seed": seed, "undefined_reasons": []}
            ia, ib = have.get(ia_name), have.get(ib_name)
            if ia is None or ib is None or "_load_error" in (ia or {}) or "_load_error" in (ib or {}):
                row["undefined_reasons"].append("intact arm missing")
            else:
                labels = _labels(group, seed)
                ok, diffs = null_clean(ia["responses"], ib["responses"], labels)
                row["null_clean"], row["null_diffs"] = ok, diffs[:8]
                if not ok:
                    row["undefined_reasons"].append("null failed (intact_a != intact_b) on %d turn(s)" % len(diffs))
                row["sources"] = sorted(set(ia.get("sources") or []) | set(ib.get("sources") or []))
                for arm_name in need:
                    sh = have.get(arm_name)
                    if sh is None or "responses" not in sh:
                        row["undefined_reasons"].append("%s arm missing or unreadable" % arm_name)
                    else:
                        row["undefined_reasons"] += arm_validity(group, arm_name, sh)
                row["ltm_bundles"] = sorted({str(((have.get(a) or {}).get("process") or {}).get("ltm_bundle"))
                                             for a in need if have.get(a) is not None})
                ra, rb = ia["responses"], ib["responses"]
                row["intact"] = SCORERS[group](ia)
                row["correct"] = bool(row["intact"]["all"])
                # ── lesions: effect present, reproduced, ENGAGED, and (where registered) dissociated ──
                row["lesions"] = {}
                for les in g["lesions"]:
                    L, R = have.get(les), have.get(les + ".rep0")
                    rec = {"present": L is not None, "repeat_present": R is not None}
                    if L is not None and "responses" in L:
                        labs = SCORED_LABELS[group][les]
                        d = [reply_differs(ra, L["responses"], l) for l in labs]
                        rec["differs"] = dict(zip(labs, d))
                        rec["reply_effect"] = any(x is True for x in d)
                        keep = UNCHANGED_LABELS[group].get(les, [])
                        rec["dissociated"] = all(reply_differs(ra, L["responses"], l) is False for l in keep) if keep else True
                        rec["engaged"] = lesion_engaged(group, les, L)
                        rec["effect"] = bool(rec["reply_effect"] and rec["dissociated"])
                        rec["attributable"] = attributable_to("%s/%s seed %d (primary reply fields on %s)" % (group, les, seed, ",".join(labs)),
                                                              float(n_field_diffs(ra, L["responses"], labs)), float(n_field_diffs(ra, rb, labs)))
                        rec["lesion_score"] = {k: v for k, v in SCORERS[group](L).items()
                                               if k in ("all", "p3", "p2", "r_all", "auroc", "n_correct_guess", "H", "q1", "q2", "q3", "q4")}
                        rec["reproduced"] = (null_clean(L["responses"], R["responses"], _labels(group, seed, les))[0]
                                             if R is not None and "responses" in R else None)
                        rec["applied"] = patch_applied(L.get("patch"))
                        if not rec["engaged"]:
                            row["undefined_reasons"].append("%s: lesion not engaged on a scored turn (instrument)" % les)
                    else:
                        rec["effect"] = None
                        rec["reproduced"] = None
                    row["lesions"][les] = rec
                # ── permuted exposure ──
                P = have.get("permuted")
                if P is not None and "responses" in P:
                    sp = SCORERS[group](P, permuted=True)
                    row["permuted"] = sp
                    row["tracks"] = bool(sp["all"])
                else:
                    row["tracks"] = None
                # ── the content-specific intervention (steer / G) ──
                eff, applied = steer_effect(group, have, row)
                row["steer"] = {"arm": STEER_ARM[group], "effect": eff, "applied": applied}
                if eff is None:
                    row["undefined_reasons"].append("%s arm missing or unreadable" % STEER_ARM[group])
                elif not applied:
                    row["undefined_reasons"].append("%s: patch not applied" % STEER_ARM[group])
                # ── comparator (credit): corrected >= replay_only on H ──
                if group == "credit":
                    C = have.get("replay_only")
                    if C is not None and "responses" in C:
                        sc = SCORERS[group](C)
                        row["comparator"] = {"replay_H": sc["H"], "replay_H_patient": sc["H_patient"],
                                             "ge": int(bool(row["intact"]["H"])) >= int(bool(sc["H"])),
                                             "strict": bool(row["intact"]["H"]) and not sc["H"]}
                    else:
                        row["comparator"] = None
                        row["undefined_reasons"].append("replay_only arm missing or unreadable")
                # ── ownership ──
                row["ownership"] = OWNERSHIP[group](seed, have, row)
                for ev in row["ownership"]["events"]:
                    panel_events.append(dict(ev, group=group))
                if informational:
                    row["informational"] = {}
                    for inf in g["informational"]:
                        I = have.get(inf)
                        if I is not None and "responses" in I:
                            si = SCORERS[group](I)
                            row["informational"][inf] = {k: v for k, v in si.items()
                                                         if k in ("all", "p3", "p2", "r_all", "auroc", "hedged_any", "taught_levels",
                                                                  "q1", "q2", "q3", "q4", "named", "H", "H_patient", "routes")}
                            pi = I.get("patch")
                            if isinstance(pi, dict):
                                row["informational"][inf]["patch_calls"] = pi.get("calls")
                                row["informational"][inf]["applied"] = patch_applied(pi)
            rows[seed] = row
            if not row["undefined_reasons"] and "correct" in row:
                defined.append(seed)
        n_def = len(defined)
        n_correct = sum(1 for s in defined if rows[s]["correct"])
        les_names = list(g["lesions"])
        n_les = {l: sum(1 for s in defined if (rows[s]["lesions"].get(l) or {}).get("effect") is True) for l in les_names}
        n_rep = {l: sum(1 for s in defined if (rows[s]["lesions"].get(l) or {}).get("reproduced") is True) for l in les_names}
        n_tracks = sum(1 for s in defined if rows[s].get("tracks") is True)
        n_steer = sum(1 for s in defined if (rows[s].get("steer") or {}).get("effect") is True)
        n_ge = sum(1 for s in defined if (rows[s].get("comparator") or {}).get("ge")) if group == "credit" else None
        cap_ok = (n_def == len(seeds) and n_correct == n_def and all(n_les[l] == n_def for l in les_names)
                  and all(n_rep[l] == n_def for l in les_names) and n_tracks == n_def and n_steer == n_def
                  and (group != "credit" or n_ge == n_def))
        capability = "INCOMPLETE" if n_def < len(seeds) else ("PASS" if cap_ok else "FAIL")
        why = []
        if capability == "FAIL":
            if n_correct < n_def:
                if group == "meta" and any(not (rows[s]["intact"].get("capability_present")) for s in defined):
                    why.append("capability absent: fewer than %d correct guess items on %d of %d seeds" % (
                        MIN_CORRECT_GUESS, sum(1 for s in defined if not rows[s]["intact"].get("capability_present")), n_def))
                else:
                    why.append("capability absent on %d of %d seeds" % (n_def - n_correct, n_def))
                if group == "chain":
                    for k in ("fork_honest", "via_workspace", "r_all", "p3"):
                        n_k = sum(1 for s in defined if rows[s]["intact"].get(k))
                        if n_k < n_def:
                            why.append("%s %d/%d" % ({"via_workspace": "not via the workspace"}.get(k, k), n_k, n_def))
            for l in les_names:
                if n_les[l] < n_def:
                    why.append("%s: reply-level effect (with dissociation) on %d of %d seeds (not load-bearing)" % (l, n_les[l], n_def))
                if n_rep[l] < n_def:
                    why.append("%s: reproduced on %d of %d seeds" % (l, n_rep[l], n_def))
            if n_tracks < n_def:
                why.append("permuted exposure tracked on %d of %d seeds (not exposure-driven)" % (n_tracks, n_def))
            if n_steer < n_def:
                why.append("%s effect on %d of %d seeds (not steerable)" % (STEER_ARM[group], n_steer, n_def))
            if group == "credit" and n_ge is not None and n_ge < n_def:
                why.append("corrected >= replay_only on %d of %d seeds" % (n_ge, n_def))
        elif capability == "INCOMPLETE":
            why.append("%d of %d seeds defined" % (n_def, len(seeds)))
        own_rows = [rows[s]["ownership"]["verdict"] for s in defined]
        if n_def < len(seeds) or any(v == "UNDEFINED" for v in own_rows):
            ownership = "UNDEFINED"
        elif any(v == "HOST" for v in own_rows):
            ownership = "HOST"
        elif any(v == "N/A" for v in own_rows):
            ownership = "N/A"
        else:
            ownership = own_rows[0] if own_rows else "UNDEFINED"
        own_why = sorted({r for s in defined for r in rows[s]["ownership"]["host"] + rows[s]["ownership"]["undefined"]})
        if capability == "INCOMPLETE":
            criterion = "INCOMPLETE"
        elif capability == "FAIL":
            criterion = "FAIL"
        elif ownership in ("OWNED", "OWNED(read-out)"):
            criterion = "PASS"
        elif ownership == "HOST":
            criterion = "CAPABILITY PASS, HOST-OWNED"
        else:
            criterion = "INCOMPLETE"
        sub = {}
        for s in defined:
            for k, v in rows[s]["intact"].items():
                if isinstance(v, bool):
                    sub[k] = sub.get(k, 0) + (1 if v else 0)
        per_group[group] = {"criterion": g["criterion"], "primary": g["primary"], "verdict": criterion,
                            "capability": capability, "ownership": ownership, "why": why, "ownership_why": own_why,
                            "n_defined": n_def, "n_correct": n_correct, "n_lesion_effect": n_les, "n_reproduced": n_rep,
                            "n_permuted_tracks": n_tracks, "n_steer_effect": n_steer, "n_ge_replay": n_ge,
                            "n_ownership": {v: own_rows.count(v) for v in set(own_rows)},
                            "sub_clauses_n_true": sub, "seeds": rows}
    crit = {c: per_group[[g for g in PRIMARY_GROUPS if GROUPS[g]["criterion"] == c][0]]["verdict"] for c in CRITERIA}
    seams = sorted({e["seam"] for e in panel_events})
    panel = HSS.panel(panel_events, seams)
    panel["per_seam"] = {s: dict(HSS.seam_stats([e for e in panel_events if e["seam"] == s]),
                                 **{"class": HSS.classify_seam(HSS.seam_stats([e for e in panel_events if e["seam"] == s]))})
                         for s in seams}
    panel["target_host_share"] = HOST_SHARE_TARGET
    panel["host_decision_debt"] = ["C1 derived flag + lead (a template over an ADVANCE count)", "C1 DETECT regexes and `target in path`",
                                   "C2 evidence scalar (host arithmetic)", "C2/C4 admission by verb, floor, tie margin",
                                   "C3 location->slot map and argmax", "C4 host SVO extractor (sense boundary)"]
    from tools.verdict import Verdict
    prim_rows = [per_group[g]["seeds"][s] for g in PRIMARY_GROUPS for s in seeds]
    v = Verdict("Stage-3 (ADOLESCENT) exit on the integrated production brain (DRAFT)")
    v.require("no counted shard missing", len(missing), expect=0)
    v.require("null clean (intact_a == intact_b) on every (group, seed)", sum(1 for r in prim_rows if r.get("null_clean") is not True), expect=0)

    def _n_rows_with(tag):
        return sum(1 for r in prim_rows if any(tag in x for x in r.get("undefined_reasons", [])))
    v.require("every response from the expected brain", _n_rows_with("unexpected brain source"), expect=0,
              note="expected `source` = %s in every counted arm" % EXPECTED_SOURCE)
    v.require("PYTHONHASHSEED=0 in every counted arm", _n_rows_with("PYTHONHASHSEED not 0"), expect=0)
    v.require("no errored turn and no organ error in any counted arm", _n_rows_with("errored turn") + _n_rows_with("organ error"), expect=0)
    v.require("the corpus present in every counted arm", _n_rows_with("corpus absent"), expect=0)
    v.require("every in-process intervention applied (an error-free event recorded)", _n_rows_with("not applied"), expect=0)
    v.require("every lesion engaged on a scored turn", _n_rows_with("not engaged"), expect=0)
    v.require("six seeds defined for every criterion", sum(1 for g in PRIMARY_GROUPS if per_group[g]["n_defined"] < len(seeds)), expect=0)
    v.require("ownership instruments defined for every criterion", sum(1 for g in PRIMARY_GROUPS if per_group[g]["ownership"] == "UNDEFINED"), expect=0)
    earned = v.decide(go=all(x == "PASS" for x in crit.values()), verbose=False)
    stage = ("UNDEFINED (INCOMPLETE): " + "; ".join(earned["undefined_reasons"])[:400]) if earned["status"] == "UNDEFINED" \
        else ("PASS" if earned["go"] else "FAIL")
    return {"stage": "STAGE-3 (ADOLESCENT) exit (DRAFT runner)", "verdict": stage, "GO": bool(earned["go"]),
            "preconditions": earned["preconditions"], "undefined_reasons": earned["undefined_reasons"],
            "criteria": crit, "groups": per_group, "panel": panel, "seeds": seeds, "missing_shards": missing,
            "bars": {"auroc": AUROC_BAR, "min_correct_guess": MIN_CORRECT_GUESS, "ri_min_fraction": RI_MIN_FRACTION,
                     "ri_factor": RI_FACTOR, "seeds_required": len(seeds), "m_adj_told_fix_in_pin": M_ADJ_TOLD_FIX_IN_PIN},
            "runner": "research/runners/_stage3_exit_draft.py", "prereg": PREREG, "prereg_registered": PREREG_REGISTERED,
            "expected_source": EXPECTED_SOURCE, "derived_lex_from_production": DERIVED_LEX_FROM_PRODUCTION}


def report(agg):
    lines = ["**Stage-3 exit verdict: %s** (criteria: %s)" % (agg["verdict"], ", ".join("%s %s" % (k, v) for k, v in agg["criteria"].items())),
             "", "| group | crit | criterion | capability | ownership | correct | lesion effect | reproduced | tracks | steer | why |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g, d in agg["groups"].items():
        lines.append("| %s | %s | %s | %s | %s | %d/%d | %s | %s | %d/%d | %d/%d | %s |" % (
            g, d["criterion"], d["verdict"], d["capability"], d["ownership"], d["n_correct"], d["n_defined"],
            ", ".join("%s %d/%d" % (k, v, d["n_defined"]) for k, v in d["n_lesion_effect"].items()),
            ", ".join("%s %d/%d" % (k, v, d["n_defined"]) for k, v in d["n_reproduced"].items()),
            d["n_permuted_tracks"], d["n_defined"], d["n_steer_effect"], d["n_defined"],
            "; ".join(d["why"] + d["ownership_why"]) or "-"))
    p = agg["panel"]
    lines += ["", "Ownership panel (host-share schema; S/D not measured): host_decision_share %s over %s F events; neural %s; target <= %s" % (
        p.get("host_decision_share"), p.get("n_events_F_defined"), p.get("neural_decision_fraction"), p.get("target_host_share"))]
    for s, st in p.get("per_seam", {}).items():
        lines.append("- %s: HS %s, class %s, live seeds %s" % (s, st.get("HS"), st.get("class"), st.get("live_seeds")))
    lines += ["", "Sub-clauses (seeds true / seeds defined; reported):"]
    for g, d in agg["groups"].items():
        if d["sub_clauses_n_true"]:
            lines.append("- %s: %s" % (g, ", ".join("%s %d/%d" % (k, v, d["n_defined"]) for k, v in sorted(d["sub_clauses_n_true"].items()))))
    if agg["missing_shards"]:
        lines += ["", "Missing shards: %d" % len(agg["missing_shards"])]
    return "\n".join(lines)


# ══════════════════════════════════════════════ POOL JOBS ═══════════════════════════════════════════════════════════
def pool_job_lines(sha, seeds=GATE_SEEDS, out_dir=GATE_OUT_DIR, informational=True, max_age_h=MAX_AGE_H, prereg=PREREG):
    out = []
    max_age_h = min(int(max_age_h), 240)
    for seed in seeds:
        for group in GROUPS:
            for arm in arms_for(group, with_informational=informational):
                p = shard_path(out_dir, seed, group, arm)
                gate = " --allow-gate-seed --prereg %s" % prereg if seed in GATE_SEEDS else ""
                cmd = ("cd ~/derisk-pool/revisions/%s && bash tools/mem_ok.sh %d 1 && mkdir -p %s && "
                       "bash tools/memcap.sh %d -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                       "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m "
                       "research.runners._stage3_exit_draft --shard --seed %d --group %s --arm %s%s --out %s"
                       % (sha, MEM_GB, os.path.dirname(p), MEMCAP_GB, seed, group, arm, gate, p))
                why = ("prereg %s (%s); Stage-3 exit gate seed %d %s %s; record: third integrated stage-exit measurement, "
                       "two-clause (capability + ownership) verdict; mem_gb=%d max_age_h=%d" % (prereg, sha[:9], seed, group, arm, MEM_GB, max_age_h))
                out.append((cmd, why))
    return out


# ══════════════════════════════════════════════ SYNTHETIC BRAINS (selftest, tests, bar-sanity) ═════════════════════
_AB = "I don't know about that."


def R(answer, svo=None, abst=False, derived=False, derived_from=None, **extra):
    d = {"answer": answer, "abstained": bool(abst), "recalled_svo": (list(svo) if svo else None), "derived": bool(derived),
         "derived_from": ([list(x) for x in derived_from] if derived_from else None), "source": EXPECTED_SOURCE}
    d.update(extra)
    return d


def _derived(agent, verb, path, lead=0, swap=False):
    """A derived reply as production frames it: derived_from = the hops along `path` (agent first)."""
    hops = [[path[i], verb, path[i + 1]] for i in range(len(path) - 1)]
    claim = _fx(agent, verb, path[-1])
    if swap:
        text = "%s %s, %s" % (DERIVED_LEADS[2], claim, DERIVED_SWAP_TAIL)
    else:
        text = "%s %s. %s" % (DERIVED_LEADS[lead], "; ".join("%s %s %s" % tuple(h) for h in hops), claim)
    return R(text, None, derived=True, derived_from=hops)


def _plain(a, v, p, prep=None):
    return R(_fx(a, v, p, prep), [a, v, p])


def _ttrace(route, path, lesioned=False):
    """A transitive_trace as the handler attaches it: one ADVANCE per committed hop, then a COMMIT."""
    chase = [{"action": "ADVANCE", "committed": c, "n_ignited": 1} for c in path[1:]] + [{"action": "COMMIT", "committed": None, "n_ignited": 0}]
    return {"route": route, "lesioned": bool(lesioned), "error": None, "chase": chase if route == "chase" else [], "path_len": len(path)}


def _rates(ignite, n=4, hi=0.9, lo=0.05):
    return [hi if (ignite and i == 0) else (lo if ignite else 0.0) for i in range(n)]


def chain_brain(kind, arm, permuted=False, m_adj_fix=None):
    """{label: response}, internal, patch record, observe record for one chain arm of a brain of `kind`."""
    base = arm.split(".rep")[0]
    r = {l: R(m) for l, m, role in turns_for("chain", 1, arm) if role == "teach"}
    lesion = base == "ws_lesion"
    fork_terms = {"slug": ["puma", "shrew", "slug"], "flea": ["puma", "gecko", "mantis", "flea"]}
    if permuted:
        fork_terms = {"flea": ["puma", "shrew", "mantis", "flea"]}
    branch_order = ["slug", "flea"] if not permuted else ["flea"]
    if base == "order_swap" and kind in ("main", "order_dependent"):
        branch_order = list(reversed(branch_order))
    first = branch_order[0]
    # which conclusions the brain reaches
    reach_p3 = kind not in ("main",) or permuted or first == "flea"
    if kind == "order_dependent":
        reach_p3 = base != "order_swap"            # order-invariant intact, flips under the swap (ORDER change)
    reach_p2 = (not permuted) and (kind not in ("main",) or first == "slug")
    if base == "kselect_order" and kind in ("main", "f_host"):
        reach_p3, reach_p2 = (not permuted) or True, False          # the host alternative: the gecko branch
    if lesion and kind not in ("closure", "closure_all_leaves", "lesion_inert"):
        reach_p3 = reach_p2 = False
    if base == "silence_mantis" and kind not in ("closure", "closure_all_leaves"):
        reach_p3 = False
    marked = kind != "unmarked"
    swap = kind == "perfect_swap"
    r["p3"] = (_derived("puma", "chase", ["puma", "gecko", "mantis", "flea"] if not permuted else ["puma", "shrew", "mantis", "flea"], swap=swap)
               if (reach_p3 and marked) else (_plain("puma", "chase", "flea") if reach_p3 else R(_AB, abst=True)))
    r["p3"]["transitive_trace"] = _ttrace("chase", (["puma", "gecko", "mantis", "flea"] if reach_p3 else ["puma", first] + (["slug"] if first == "slug" else [])), lesion)
    r["p2"] = (_derived("puma", "chase", ["puma", "shrew", "slug"], swap=swap) if (reach_p2 and marked)
               else (_plain("puma", "chase", "slug") if reach_p2 else R(_AB, abst=True)))
    r["p2"]["transitive_trace"] = _ttrace("chase", ["puma", "shrew", "slug"] if reach_p2 else ["puma", first], lesion)
    told_fix = M_ADJ_TOLD_FIX_IN_PIN if m_adj_fix is None else m_adj_fix
    if kind in ("main",) and not told_fix and first != "flea":
        r["m_adj"] = R(_AB, abst=True)
    else:
        r["m_adj"] = _plain("puma", "chase", "gecko")
    r["m_adj"]["transitive_trace"] = _ttrace("direct", ["puma", "gecko"], lesion)
    if permuted:
        r["m_cross"] = _derived("shrew", "chase", ["shrew", "mantis", "flea"], swap=swap) if (not lesion or kind in ("closure", "closure_all_leaves", "lesion_inert")) else R(_AB, abst=True)
        r["m_cross"]["transitive_trace"] = _ttrace("chase", ["shrew", "mantis", "flea"], lesion)
    else:
        r["m_cross"] = R(_AB, abst=True)
        r["m_cross"]["transitive_trace"] = _ttrace("chase", ["shrew", "slug"], lesion)
    r["m_rev"] = R(_AB, abst=True)
    r["m_rev"]["transitive_trace"] = _ttrace("chase", ["flea"], lesion)
    # the fork
    if permuted:
        r["r_fork"] = _derived("puma", "chase", ["puma", "shrew", "mantis", "flea"]) if marked else _plain("puma", "chase", "flea")
    elif kind in ("perfect", "perfect_swap", "unmarked", "lesion_inert", "lesion_kills_recall", "steer_inert", "ri_unstable",
                  "order_dependent", "f_host", "closure_all_leaves", "relay"):
        r["r_fork"] = R("Two chains run from the puma: the puma chases the slug; the puma chases the flea. I cannot say which you mean.", abst=True)
    else:                                                       # one branch only (main, closure, single_branch_fork)
        p = fork_terms[first]
        r["r_fork"] = _derived("puma", "chase", p) if (marked and kind != "main") else _plain("puma", "chase", p[-1])
    if lesion and kind not in ("closure", "closure_all_leaves", "lesion_inert"):
        r["r_fork"] = R(_AB, abst=True)
    # the unambiguous 2-hop chase from the gecko
    hops_all = 2
    if base == "steer_leaf" and kind not in ("closure", "closure_all_leaves", "relay", "steer_inert"):
        r["r_all"] = _derived("gecko", "chase", ["gecko", "mantis", "flea", "slug"]) if marked else _plain("gecko", "chase", "slug")
        hops_all = 3
    elif (lesion and kind not in ("closure", "closure_all_leaves", "lesion_inert")) or (base == "silence_mantis" and kind not in ("closure", "closure_all_leaves")):
        r["r_all"] = R(_AB, abst=True)
        hops_all = 0
    elif kind == "main":
        r["r_all"] = _plain("gecko", "chase", "flea")           # the terminal spoken as a recalled fact (no fix in pin)
    else:
        r["r_all"] = _derived("gecko", "chase", ["gecko", "mantis", "flea"]) if marked else _plain("gecko", "chase", "flea")
    r["c_direct"] = R(_AB, abst=True) if (lesion and kind == "lesion_kills_recall") else _plain("gecko", "chase", "mantis")
    r["c_tail"] = R(_AB, abst=True)
    r["c_dog"] = _plain("dog", "chase", "cat")
    internal = {"r_all": {"gnw_multistep": {"acted": True, "reason": "chain_terminal" if hops_all else "chase_abstained",
                                             "terminal": _patient(r["r_all"]), "resolved_hops": hops_all, "cycles": hops_all + 1,
                                             "lesion": bool(lesion)}},
                "c_tail": {"gnw_multistep": {"acted": False, "reason": "already_abstained", "resolved_hops": None, "cycles": None, "lesion": bool(lesion)}}}
    hi, lo = (0.55, 0.4) if kind == "ri_unstable" else (0.9, 0.05)
    obs = {"patch": "chase_rates", "calls": 0, "n_applied": 0, "events": [], "thr": 0.5}
    for l, n in (("p3", 3 if reach_p3 else 1), ("p2", 2 if reach_p2 else 1), ("r_all", hops_all)):
        for i in range(n):
            _ev(obs, label=l, target="x", rates=_rates(True, hi=hi, lo=lo), winner=0, n_ignited=1, committed="x", thr=0.5)
        _ev(obs, label=l, target=None, rates=_rates(False), winner=0, n_ignited=0, committed=None, thr=0.5)
    patch = None
    if base == "kselect_order":
        patch = {"patch": base, "calls": 2, "n_applied": 2, "events": [{"label": "p3", "matches": ["shrew", "gecko"], "returned": "gecko"},
                                                                       {"label": "p2", "matches": ["shrew", "gecko"], "returned": "gecko"}]}
    elif base == "steer_leaf":
        patch = {"patch": base, "calls": 1, "n_applied": 1, "events": [{"label": "r_all", "cue": "flea", "target_out": "slug"}]}
    elif base == "silence_mantis":
        patch = {"patch": base, "calls": 1, "n_applied": 1, "events": [{"label": "r_all", "target_in": "mantis", "target_out": None}]}
    return r, internal, patch, obs


def _ci(rates, lesion=None, patient=None, reason="reached"):
    return {"read": {"rates": rates, "lesion": lesion, "patient": patient, "reason": reason,
                     "floor_rate": 0.02, "tie_margin": 1.25}, "observe": None}


def _guess_reply(a, food):
    return R("Maybe the %s eats the %s -- that's a guess from what I've learned, not something I was taught." % (a, food), [a, "eat", food])


def meta_brain(kind, arm, permuted=False, rng=None):
    base = arm.split(".rep")[0]
    r = {l: R(m) for l, m, role in turns_for("meta", 1, arm) if role == "teach"}
    world = META_WORLD_PERM if permuted else META_WORLD
    consistent = META_CONSISTENT_PERM if permuted else META_CONSISTENT
    lesion = base == "inherit_lesion" and kind not in ("conflict_counter", "lesion_inert")
    les_tag = "all" if base == "inherit_lesion" else None
    for a in consistent + (META_CONFLICTING_PERM if permuted else META_CONFLICTING):
        truth = world[a]
        is_cons = a in consistent
        inferable = truth if is_cons else ("grub" if a in ("bat", "duck", "lark", "finch") else "weed")
        other = "weed" if inferable == "grub" else "grub"
        if is_cons:
            rates = {"eat %s" % inferable: 0.08, "eat %s" % other: 0.01}
        else:
            ratio = 1.4 if kind == "ri_unstable" else 1.04
            rates = {"eat %s" % inferable: round(0.048 * ratio, 4), "eat %s" % other: 0.048}
        answer = True
        if kind in ("perfect", "lesion_inert", "permuted_inert", "steer_inert", "ri_unstable", "conflict_counter"):
            answer = is_cons
        elif kind == "perfect_hedge":
            answer = True
        elif kind == "never_abstains":
            answer = True
        elif kind in ("all_abstain", "hedge_by_taught"):
            answer = False
        elif kind == "item_keyed":
            answer = a not in META_CONFLICTING            # by NAME, whatever the roles
        elif kind == "random_optout":
            answer = bool(rng.random() < 0.5) if rng is not None else (a in consistent)
        if base == "steer_frame" and a == STEER_MEMBER and kind not in ("steer_inert", "conflict_counter"):
            answer = True
            inferable = STEER_FRAME[1]
        if lesion:
            answer = False
        if answer:
            r["q_%s" % a] = _guess_reply(a, inferable)
            if kind == "perfect_hedge" and not is_cons:
                r["q_%s" % a]["answer"] = HEDGE_TEXT + " " + r["q_%s" % a]["answer"]
        else:
            r["q_%s" % a] = R(_AB, abst=True)
        r["q_%s" % a]["category_inherit"] = _ci(rates, les_tag, inferable if answer else None, "reached" if answer else ("tie" if not is_cons else "below_floor"))
    for a in META_TAUGHT:
        for k in ("r_%s_1" % a, "r_%s_2" % a):
            r[k] = _plain(a, "eat", world[a])
    r["c_dog"] = _plain("dog", "chase", "cat")
    patch = None
    if base == "steer_frame":
        patch = {"patch": base, "calls": 1, "n_applied": 1, "events": [{"label": "q_bat", "frame": list(STEER_FRAME)}]}
    return r, {}, patch, None


def tom_brain(kind, arm, permuted=False):
    base = arm.split(".rep")[0]
    r = {}
    for l, m, role in turns_for("tom", 1, arm):
        if role == "expose":
            r[l] = R(m)
    r["s1_delay1"] = _plain("dog", "chase", "cat")
    r["s1_delay2"] = R("My affect monitor reads neutral right now (valence differential +0.000, low arousal) -- that's a functional read of my mood state, not a felt experience.")
    exp = TOM_EXPECT_PERM if permuted else TOM_EXPECT
    locs = {"q1": ["basket", "box"], "q2": ["drawer", "jar", "cup"], "q3": ["bag", "tin"], "q4": ["chest", "pot", "shelf"]}
    reality = {"q1": "box", "q2": "cup", "q3": "tin", "q4": "shelf"}
    present_at_query = {"q1": False, "q2": False, "q3": True, "q4": True}
    objects = {"q1": "marble", "q2": "key", "q3": "coin", "q4": "ring"}
    agents = {"q1": "Sally", "q2": "Tom", "q3": "Lea", "q4": "Kate"}
    for q in TOM_QUERIES:
        named = exp[q]
        if kind == "first_location":
            named = locs[q][0]
        elif kind == "present_means_reality":
            named = reality[q] if present_at_query[q] else exp[q]
        if base == "witness_flip" and kind in ("host_rule_unread_store", "host_rule_store_veto", "host_written_belief", "f_host"):
            named = TOM_WITNESS_FLIP_ALT[q]
        if base == "steer_store" and q == "q1" and kind not in ("host_rule_store_veto", "steer_inert"):
            named = STEER_STORE_LOC
        silent = False
        if base == "store_lesion" and kind not in ("host_rule_unread_store", "lesion_inert"):
            silent = True
        slot = locs[q].index(named) if named in locs[q] else 0
        if kind == "ri_unstable":
            rates = {str(i): (0.09 if i == slot else (0.06 if i == (slot + 1) % len(locs[q]) else 0.0)) for i in range(len(locs[q]))}
        else:
            rates = {str(i): (0.09 if i == slot else 0.0) for i in range(len(locs[q]))}
        if silent:
            r[q] = R(_AB, abst=True, false_belief_tom={"acted": False, "reason": "belief_store_silent", "store_lesioned": True,
                                                        "belief_rates": {str(i): 0.0 for i in range(len(locs[q]))}})
        else:
            r[q] = R("%s will look in the %s for the %s." % (agents[q], named, objects[q]),
                     false_belief_tom={"acted": True, "reason": "queried", "belief_location": named, "reality_location": reality[q],
                                       "belief_loc_idx": slot, "belief_rates": rates, "store_lesioned": bool(base == "store_lesion")})
    r["m1"] = R(_AB, abst=True)
    r["c_dog"] = _plain("dog", "chase", "cat")
    patch = None
    if base == "witness_flip":
        patch = {"patch": base, "calls": 6, "n_applied": 6, "events": [{"label": "s1_move", "flipped": True}]}
    elif base == "steer_store":
        patch = {"patch": base, "calls": 1, "n_applied": 1, "events": [{"label": "q1", "slot": 1, "loc": "box"}]}
    return r, {}, patch, None


def credit_brain(kind, arm, permuted=False):
    base = arm.split(".rep")[0]
    r = {l: R(m) for l, m, role in turns_for("credit", 1, arm) if role == "teach" and not m.startswith("say")}
    for l, m, role in turns_for("credit", 1, arm):
        if m == IDLE:
            r[l] = R("Something's been on my mind — robin eat worm. What does robin eat?")
    want = "nut" if permuted else "seed"
    r["e1"] = _guess_reply("lark", "worm")
    r["e2"] = _guess_reply("swift", "worm")
    teacher = base not in ("no_teacher",)
    learns = kind in ("perfect", "lesion_inert", "permuted_inert", "steer_inert", "ri_unstable", "host_inferred_block", "activity_only", "exemplar_nn", "recent_correction")
    H = "worm"
    if learns and teacher and base not in ("learn_lesion",):
        H = want
    if kind == "exemplar_nn":
        H = want                                       # the teacher is not load-bearing: no_teacher reads the same
    if base == "replay_only" and kind not in ("exemplar_nn",):
        H = "worm"
    if base == "corr_freeze" and kind not in ("activity_only", "exemplar_nn", "steer_inert", "host_inferred_block"):
        H = "worm"
    if base == "learn_lesion" and kind not in ("lesion_inert",):
        H = None
    if base == "kb_pop" and kind == "host_inferred_block":
        H = None
    ratio = 1.4 if kind == "ri_unstable" else 4.0
    rates = {"eat %s" % want: round(0.02 * ratio, 4), "eat worm": 0.02}
    if H is None:
        r["H"] = R(_AB, abst=True, category_inherit=_ci(rates, "lateral_learn" if base == "learn_lesion" else None, None, "below_floor"))
    else:
        r["H"] = _guess_reply("canary", H)
        r["H"]["category_inherit"] = _ci(rates, None, H)
    r["J"] = _guess_reply("jay", want if kind == "recent_correction" else "worm")
    r["c_lark"] = _plain("lark", "eat", want) if teacher else R(_AB, abst=True)
    if base == "kb_pop" and teacher:
        r["c_lark"] = _guess_reply("lark", want)
    r["c_robin"] = _plain("robin", "eat", "worm")
    r["c_dog"] = _plain("dog", "chase", "cat")
    patch = None
    if base == "kb_pop":
        patch = {"patch": base, "calls": 2, "n_applied": 2, "events": [{"label": "k1", "popped": ["lark", "eat", want]}, {"label": "k2", "popped": ["swift", "eat", want]}]}
    elif base == "corr_freeze":
        patch = {"patch": base, "calls": 2, "n_applied": 2, "events": [{"label": "k1"}, {"label": "k2"}]}
    return r, {}, patch, None


BRAINS = {"chain": chain_brain, "meta": meta_brain, "tom": tom_brain, "credit": credit_brain}


def synthetic_shard(group, seed, arm, kind="perfect", rng=None, **kw):
    """One synthetic shard in the worker's format for `arm` of a brain of `kind` (the permuted arm reads the permuted
    table). `kw` is passed to the group's brain (e.g. m_adj_fix)."""
    base = arm.split(".rep")[0]
    permuted = base == "permuted"
    fn = BRAINS[group]
    if group == "meta":
        resp, internal, patch, obs = fn(kind, arm, permuted, rng=rng)
    elif group == "chain":
        resp, internal, patch, obs = fn(kind, arm, permuted, **kw)
    else:
        resp, internal, patch, obs = fn(kind, arm, permuted)
    turns = turns_for(group, seed, arm)
    return {"seed": int(seed), "group": group, "arm": arm, "kind": arm_spec(group, arm)[2], "responses": resp,
            "internal": internal, "patch": patch, "observe_patch": obs, "sources": [EXPECTED_SOURCE],
            "pythonhashseed_ok": True, "corpus_present": True, "builtin_facts": [["dog", "chase", "cat"], ["cat", "eat", "fish"]],
            "turns": [{"label": l, "message": m, "role": r} for l, m, r in turns],
            "process": {"peak_rss_mb": 6300.0, "ltm_bundle": "wikidata_100k"}}


def write_synthetic_tree(root, seeds, kinds_by_group, informational=False):
    """kinds_by_group: {group: kind | callable(arm) -> kind}."""
    for seed in seeds:
        for group in GROUPS:
            k = kinds_by_group.get(group, "perfect")
            for arm in arms_for(group, with_informational=informational):
                kind = k(arm) if callable(k) else k
                p = shard_path(root, seed, group, arm)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as f:
                    json.dump(synthetic_shard(group, seed, arm, kind), f)


# ══════════════════════════════════════════════ SELFTEST (no brain) ════════════════════════════════════════════════
def selftest(verbose=True):
    import tempfile
    checks = {}
    checks["every group has the intact pair, >=1 lesion (+rep0), permuted, a steer/G arm"] = all(
        arms_for(g)[:2] == list(_intact_names(g)) and "permuted" in arms_for(g) and STEER_ARM[g] in arms_for(g)
        and any(a.endswith(".rep0") for a in arms_for(g)) for g in GROUPS)
    checks["one primary group per criterion, four criteria"] = sorted(GROUPS[g]["criterion"] for g in PRIMARY_GROUPS) == list(CRITERIA)
    checks["permuted/variant tables differ from intact only on teach/expose turns, same labels"] = all(
        [(l, r) for l, _m, r in GROUPS[g]["turns"]] == [(l, r) for l, _m, r in tab]
        and all(r in ("teach", "expose", "episode") for (_l, m1, r), (_l2, m2, _r2) in zip(GROUPS[g]["turns"], tab) if m1 != m2)
        for g in GROUPS for tab in [GROUPS[g]["permuted_turns"]] + list(GROUPS[g].get("variants", {}).values()))
    checks["production derived-lead strings imported"] = DERIVED_LEX_FROM_PRODUCTION and DERIVED_LEADS[0].startswith("I derived this from")
    ok_p3 = _derived("puma", "chase", ["puma", "gecko", "mantis", "flea"])
    checks["derived_framed reads the record and the lead before the claim; a plain recall and an unframed derived fail"] = (
        derived_framed(ok_p3, "puma", "flea", n_hops=3) and derived_framed(_derived("puma", "chase", ["puma", "gecko", "mantis", "flea"], swap=True), "puma", "flea")
        and not derived_framed(_plain("puma", "chase", "flea"), "puma", "flea")
        and not derived_framed(R(_fx("puma", "chase", "flea") + " I derived this from: x.", None, derived=True, derived_from=[["puma", "chase", "gecko"], ["gecko", "chase", "flea"]]), "puma", "flea")
        and not derived_framed(ok_p3, "puma", "flea", n_hops=2))
    checks["type-2 AUROC: half the conflicting items opt out = 0.75; none = 0.5; all = 1.0"] = (
        type2_auroc([2, 2, 2, 2, 0, 0, 2, 2], [1, 1, 1, 1, 0, 0, 0, 0]) == 0.75
        and type2_auroc([2] * 8, [1, 1, 1, 1, 0, 0, 0, 0]) == 0.5 and type2_auroc([2, 2, 2, 2, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]) == 1.0)
    sc = {k: score_chain(synthetic_shard("chain", 1, "intact_a", k)) for k in ("perfect", "main", "closure", "unmarked", "single_branch_fork")}
    checks["chain scorer: perfect passes; main fails (p3, fork, derived); closure fails the fork; unmarked fails derived_framed; single-branch fails the fork"] = (
        sc["perfect"]["all"] and not sc["main"]["all"] and not sc["main"]["p3"] and not sc["main"]["fork_honest"] and not sc["main"]["r_all"]
        and sc["closure"]["p3"] and not sc["closure"]["fork_honest"] and not sc["unmarked"]["p3"] and not sc["single_branch_fork"]["fork_honest"])
    sm = {k: score_meta(synthetic_shard("meta", 1, "intact_a", k)) for k in ("perfect", "never_abstains", "all_abstain", "perfect_hedge")}
    checks["meta scorer: opt-out brain AUROC 1.0 passes; never-abstains 0.5 fails; all-abstain is capability absent; hedge brain passes"] = (
        sm["perfect"]["all"] and sm["perfect"]["auroc"] == 1.0 and not sm["never_abstains"]["all"] and sm["never_abstains"]["auroc"] == 0.5
        and not sm["all_abstain"]["capability_present"] and sm["perfect_hedge"]["all"])
    st = {k: score_tom(synthetic_shard("tom", 1, "intact_a", k)) for k in ("perfect", "first_location", "present_means_reality")}
    checks["tom scorer: perfect passes; first-location fails q2/q3; present-means-reality fails q4"] = (
        st["perfect"]["all"] and not st["first_location"]["q2"] and not st["first_location"]["q3"] and st["first_location"]["q1"]
        and not st["present_means_reality"]["q4"] and st["present_means_reality"]["q2"])
    scr = {k: score_credit(synthetic_shard("credit", 1, "corrected_a", k)) for k in ("perfect", "main", "recent_correction")}
    checks["credit scorer: perfect passes; main fails H; recent-correction fails J"] = (
        scr["perfect"]["all"] and not scr["main"]["H"] and not scr["recent_correction"]["J"] and scr["recent_correction"]["H"])
    with tempfile.TemporaryDirectory() as td:
        seeds = (1, 2)
        write_synthetic_tree(td, seeds, {})
        agg = aggregate(td, seeds=seeds)
        checks["aggregate: a perfect brain PASSES every criterion (capability + ownership) and the stage"] = (
            all(v == "PASS" for v in agg["criteria"].values()) and agg["verdict"] == "PASS" and agg["GO"] is True
            and agg["panel"]["host_decision_share"] == 0.0)
        write_synthetic_tree(td, seeds, {"chain": "main", "meta": "conflict_counter", "tom": "host_rule_unread_store", "credit": "main"})
        agg2 = aggregate(td, seeds=seeds)
        gc, gm, gt, gr = (agg2["groups"][g] for g in ("chain", "meta", "tom", "credit"))
        checks["aggregate: production-at-main shapes read FAIL / HOST with the named clauses"] = (
            gc["verdict"] == "FAIL" and gc["ownership"] == "HOST" and any("fork_honest" in w for w in gc["why"])
            and gm["capability"] == "FAIL" and gm["ownership"] == "HOST" and any("not load-bearing" in w for w in gm["why"])
            and any("not steerable" in w for w in gm["why"])   # a host conflict counter: lesion and steer inert
            and gt["capability"] == "FAIL" and gt["ownership"] == "HOST" and any("not load-bearing" in w for w in gt["why"])
            and gr["verdict"] == "FAIL" and any("capability absent" in w for w in gr["why"]) and agg2["verdict"] == "FAIL")
        checks["panel: the production-at-main witness flip reads F=1 (host share > 0)"] = agg2["panel"]["host_decision_share"] > 0.5
        write_synthetic_tree(td, seeds, {"chain": "lesion_inert"})
        agg3 = aggregate(td, seeds=seeds)
        checks["aggregate: an inert lesion reads FAIL 'not load-bearing'"] = (
            agg3["groups"]["chain"]["verdict"] == "FAIL" and any("not load-bearing" in w for w in agg3["groups"]["chain"]["why"]))
        write_synthetic_tree(td, seeds, {"chain": "order_dependent", "tom": "host_written_belief"})
        agg4 = aggregate(td, seeds=seeds)
        checks["aggregate: capability PASS with a host-owned decision is its own outcome"] = (
            agg4["groups"]["chain"]["verdict"] == "CAPABILITY PASS, HOST-OWNED" and agg4["groups"]["tom"]["ownership"] == "HOST")
        os.remove(shard_path(td, 2, "meta", "steer_frame"))
        agg5 = aggregate(td, seeds=seeds)
        checks["aggregate: a missing counted arm -> that seed UNDEFINED -> INCOMPLETE, never FAIL"] = (
            agg5["groups"]["meta"]["verdict"] == "INCOMPLETE" and agg5["groups"]["meta"]["n_defined"] == 1 and agg5["verdict"].startswith("UNDEFINED"))
        checks["report renders"] = "Stage-3 exit verdict" in report(agg5)
    lines = pool_job_lines("0123456789abcdef" * 2 + "01234567")
    checks["pool jobs: one per (seed, group, arm), pinned, memcapped, gate flag + max_age_h; <= 260 jobs"] = (
        len(lines) == len(GATE_SEEDS) * sum(len(arms_for(g)) for g in GROUPS) and len(lines) <= 260
        and all(("memcap.sh %d" % MEMCAP_GB) in c and "--allow-gate-seed" in c and "revisions/0123456789" in c
                and "max_age_h=240" in w and ("mem_gb=%d" % MEM_GB) in w for c, w in lines))
    ok = all(checks.values())
    if verbose:
        for k, v in checks.items():
            print(("PASS " if v else "FAIL ") + k)
        print("SELFTEST", "OK" if ok else "FAILED")
    return ok, checks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worker", action="store_true", help="internal: build one arm in THIS process")
    ap.add_argument("--shard", action="store_true", help="run one (seed, group, arm) arm in a fresh subprocess")
    ap.add_argument("--aggregate", default=None, help="shard directory to aggregate")
    ap.add_argument("--report", default=None, metavar="AGGREGATE_JSON")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit-pool-jobs", default=None, metavar="SHA", help="print the pool job lines (TSV)")
    ap.add_argument("--max-age-h", type=int, default=MAX_AGE_H)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds for --aggregate / --emit-pool-jobs")
    ap.add_argument("--group", default=None, choices=sorted(GROUPS))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--labels", default=None, help="DEV ONLY: run only these turn labels (comma-separated)")
    ap.add_argument("--backend", default="numpy", choices=("numpy", "cupy"))
    ap.add_argument("--prereg", default=None, help="the committed preregistration a GATE-seed run is governed by")
    ap.add_argument("--allow-gate-seed", action="store_true")
    ap.add_argument("--no-informational", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        ok, _ = selftest()
        sys.exit(0 if ok else 1)
    if a.emit_pool_jobs:
        seeds = tuple(int(s) for s in a.seeds.split(",")) if a.seeds else DEV_SEEDS[6:7]
        if any(s in GATE_SEEDS for s in seeds) and not PREREG_REGISTERED:
            sys.stderr.write("refused: gate-seed pool lines need a REGISTERED preregistration (PREREG_REGISTERED is False); "
                             "emit dev lines with --seeds 7\n")
            sys.exit(2)
        for cmd, why in pool_job_lines(a.emit_pool_jobs, seeds=seeds, informational=not a.no_informational,
                                       max_age_h=a.max_age_h, out_dir=(GATE_OUT_DIR if seeds[0] in GATE_SEEDS else DEV_OUT_DIR)):
            print("%s\t%s" % (cmd, why))
        return
    if a.aggregate:
        seeds = tuple(int(s) for s in a.seeds.split(",")) if a.seeds else GATE_SEEDS
        with contextlib.redirect_stdout(sys.stderr):
            agg = aggregate(a.aggregate, seeds=seeds, informational=not a.no_informational)
        out = a.out or os.path.join(a.aggregate, "aggregate.json")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(agg, f, indent=1, default=str)
        print(report(agg))
        print("\n-> %s" % out)
        return
    if a.report:
        with open(a.report) as f:
            print(report(json.load(f)))
        return
    if a.worker or a.shard:
        if a.seed is None or not a.group or not a.arm or not a.out:
            ap.error("--seed, --group, --arm and --out are required")
        if a.seed in GATE_SEEDS and (not a.allow_gate_seed or not PREREG_REGISTERED):
            sys.stderr.write("GATE seed %d refused: this is a DRAFT runner (PREREG_REGISTERED=%s); a gate seed runs only from a pool "
                             "job emitted against the committed prereg (%s). Dev seeds: 1-12.\n" % (a.seed, PREREG_REGISTERED, PREREG))
            sys.exit(2)
        if a.worker:
            sys.exit(run_worker(a.seed, a.group, a.arm, a.out, labels=a.labels))
        sys.exit(run_shard(a.seed, a.group, a.arm, a.out, backend=a.backend, labels=a.labels,
                           allow_gate_seed=a.allow_gate_seed, prereg=a.prereg))
    ap.error("choose one of --worker/--shard/--aggregate/--report/--selftest/--emit-pool-jobs")


if __name__ == "__main__":
    main()
