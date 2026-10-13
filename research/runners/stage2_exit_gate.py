"""STAGE 2 (CHILD) EXIT GATE -- the second developmental stage-exit attempt, run END TO END on the INTEGRATED production
brain: the live `webapp.server.brain_chat` handler at production defaults, one fresh brain per arm in its own process,
never an isolated de-risk runner. Master roadmap section 4, STAGE 2 -- "Stage-exit GO: grammatical grounded
conversation; inference on held-out; curiosity ask-rate >=2x on gaps + learning-progress; discrete-emotion read-out;
private->inner-speech chain" -- operationalized at the REPLY level (what the user sees: `answer`, `abstained`,
`recalled_svo`, the derived-answer lead, the curiosity follow-up question, the feel read-out text, the self-initiated
remark), with the internal read-outs as secondary. Preregistration (frozen bars, arms, turns, predictions, UNDEFINED
rules): research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md. Execution packet for the local model:
docs/drafts/stage2-exit/HANDOFF.md. The machinery (harness, arms, null, lesion-vs-null attribution, earned verdict,
pool job lines) is the Stage-1 exit gate's (research/runners/stage1_exit_gate.py, branch research/stage1-exit-gate),
reused with five new probe groups.

THE FIVE CRITERIA (one primary probe GROUP each):
  C1 grammatical grounded conversation  group `conv`  (a mixed teach/ask dialogue; every reply a well-formed bounded
                                                       frame naming a true fact; a prepositional frame; a PASSIVE
                                                       teach comprehended; an elaboration; the moat on OOV)
  C2 inference on held-out              group `infer` (a 2-hop conclusion never told, composed from two told premises
                                                       through the possessive-chain route; the transitive yes/no route
                                                       as an informational default-OFF arm)
  C3 curiosity ask-rate >=2x on gaps    group `ask`   (the follow-up question fires on NOVEL-topic gaps at >=2x the
     + learning-progress                                rate on TAUGHT-topic moats; a learnable gap closes after the
                                                       teacher's answer; an unlearnable gap stops being asked)
  C4 discrete-emotion read-out          group `emo`   (four appraisal situations -> the feel reply names four distinct,
                                                       correct emotion categories; the 2-D sign/arousal read reported)
  C5 private->inner-speech chain        group `inner` (PRIVATE: an idle turn voices an internally selected stored
                                                       thought; INNER: a self-cued multi-hop chain whose depth the
                                                       workspace decides reaches a never-told terminal)

ARMS (every group): intact_a (production, nothing installed), intact_b (an independent same-seed rebuild = the null),
one or two LESION arms (the responsible faculty cut through an env flag production already carries, or an in-process
monkeypatch in the arm's worker), a PERMUTED-EXPOSURE arm (the teacher's exposure permuted: the reply must follow the
exposure), and for C2 two INFORMATIONAL arms (never counted). Lesion arms get a `.rep0` repeat (a second fresh process)
so a reply-level effect is "reproduced against the lesion's own repeat" (docs/TERMS.md, load-bearing).
AMENDMENT A4 (2026-10-13, docs/drafts/stage2-exit/AMENDMENT-A4.md), group `ask` only: a COUNTED `reversed` arm (the same
probes with the probe blocks in reverse order, its own session; every C3 correctness clause must hold in BOTH orders), a
taught moat whose teach turn was not acquired is UNDEFINED (never counted as novel), and an informational `da_lesion`
arm (`BRAIN_CURIOSITY_DA_LESION=1`; reported, never counted). A FAIL of C1 / C3 also prints the gaps A4 names.

HARNESS (identical in every arm incl. intact_a; declared in the prereg): numpy/CPU, one BLAS thread, stub renderer, no
LLM, tiny-demo brain with the LTM tier at its shipped default (expected `source` = "tiny-demo +LTM"),
`BRAIN_CHAT_SEED=<seed>`, `PYTHONHASHSEED=0`, `rich` omitted (the production default path). No group forces a
cupy-default value here (unlike Stage 1's `lived`): every organ these five groups touch runs its production numpy
path. No sim/ or webapp/ edit anywhere.

  .venv/bin/python -m research.runners.stage2_exit_gate --selftest
  bash tools/memcap.sh 10 -- .venv/bin/python -m research.runners.stage2_exit_gate --shard --seed 7 --group ask \
      --arm intact_a --out research/findings/raw/_stage2_exit/dev/s7/ask__intact_a.json
  .venv/bin/python -m research.runners.stage2_exit_gate --aggregate research/findings/raw/_stage2_exit/gate \
      --seeds 42,43,44,100,101,102 --out research/findings/raw/_stage2_exit/gate/aggregate.json
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

GATE_SEEDS = (42, 43, 44, 100, 101, 102)
DEV_SEEDS = tuple(range(1, 13))
EXPECTED_SOURCE = "tiny-demo +LTM"
GATE_OUT_DIR = "research/findings/raw/_stage2_exit/gate"
PREREG = "research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md"
ASK_RATIO_BAR = 2.0           # the roadmap's own Stage-2 bar: ask-rate on gaps >= 2x the rate off gaps
ASK_GAP_FLOOR = 2.0 / 3.0     # and the gap rate itself must be >= 2/3 (two of the three novel-topic gaps asked)
MOOD_TOL = 0.03               # the ladder's neutral tolerance (LADDER_NEUTRAL_TOL): |differential| below = neutral
MAX_AGE_H = 240               # every pool line declares max_age_h (the dispatcher's 12 h debris guard; cap 240 h)
# the lane's memory declaration (prereg amendment A2): dev seed 7 measured peak RSS 6.3-6.5 GB (conv/infer/ask/inner)
# and 8.1 GB (emo) on the final runner, so mem_gb=9 (the dispatcher's budget) under a 10 GB memcap (the hard kill)
MEM_GB = 9
MEMCAP_GB = 10
# amendment A4 (a): the taught-moat ask-rate is read over the moats whose teach turn was ACQUIRED; with none acquired the
# rate does not exist and the seed is UNDEFINED for C3 (an acquisition failure is never read as a curiosity result)
ASK_MOAT_MIN_DEFINED = 1
# amendment A4 reporting rule (no bar change): a FAIL of these groups names these gaps, whichever sub-clause failed
A4_NAMED_GAPS = {
    "conv": ("wh-locative question parse",
             "locative frame storage (the teach echo drops the preposition)",
             "discourse elaboration for 'tell me more'"),
    "ask": ("global DA engagement threshold; no per-topic learning-progress signal",),
}

BASE_ENV = {"SIM_BACKEND": "numpy", "BRAIN_CHAT_RENDERER": "stub", "SIM_DISABLE_LLM": "1", "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "BRAIN_LTM_SHIP_DEFAULT": "1", "PYTHONHASHSEED": "0"}

# ── the frozen grammar the C1 checker reads (a fixed lemma -> 3sg table for the verbs this gate teaches or the brain
#    holds built in; a reply verb outside it, or in the wrong form, is not grammatical) ──────────────────────────────
VERB_3SG = {"chase": "chases", "eat": "eats", "hunt": "hunts", "sleep": "sleeps", "use": "uses", "learn": "learns",
            "store": "stores", "fly": "flies", "swim": "swims", "bite": "bites", "win": "wins", "cheat": "cheats"}
_3SG_TO_LEMMA = {v: k for k, v in VERB_3SG.items()}
# a taught fact that carries a preposition must be rendered WITH it (the frame is not transitive)
TAUGHT_PREPS = {("deer", "sleep", "meadow"): "in"}
# the 4 discrete categories the C4 read-out must name, and the words that count as naming each (frozen)
EMOTION_LEXICON = {
    "joy": ("happy", "joy", "joyful", "glad", "delighted", "pleased", "cheerful", "elated"),
    "fear": ("afraid", "scared", "fear", "fearful", "anxious", "frightened", "terrified"),
    "anger": ("angry", "anger", "mad", "furious", "rage", "irritated", "annoyed"),
    "sadness": ("sad", "sadness", "sorrow", "grief", "lonely", "unhappy", "miserable", "down"),
}
# the four situations (frozen; each >= 4 content words so the host SVO extractor never stores one as a fact; every
# affect word is STRONG under the production salience gate -- pinned by tests/test_stage2_exit_gate.py)
SITUATIONS = {
    "joy": "you win a gift and you are happy and glad and delighted",
    "fear": "a wolf is attacking you and you are scared and afraid",
    "anger": "someone cheats you and you are angry and mad and furious",
    "sadness": "you are alone and lonely and sad and you cry",
}
SITUATION_AFFECT_WORDS = {"joy": ("happy", "glad", "delighted"), "fear": ("scared", "afraid"),
                          "anger": ("angry", "mad", "furious"), "sadness": ("lonely", "sad", "cry")}
# the PERMUTED arm swaps the affect company between the two same-valence... no: between joy<->sadness and
# fear<->anger, keeping each situation's frame; the expected category then follows the words
SITUATION_SWAP = {"joy": "sadness", "sadness": "joy", "fear": "anger", "anger": "fear"}
# words this gate introduces that must be absent from the tiny-demo vocabulary and the shipped LTM core (pinned by a
# test against the bundle's facts file when the bundle is present)
NOVEL_WORDS = ("fox", "hare", "berry", "deer", "meadow", "mole", "owl", "zorblat", "heron", "frog", "gnat", "crow",
               "wren", "moth", "yak", "moose", "goat", "grass", "leaf", "hay", "eel", "crab", "newt", "ibex",
               "marmot", "toad", "lynx", "vole", "beetle", "mite")


def _situation_turns(swap=False):
    """C4: a baseline feel, then each of the four situations presented twice (bare, so only that turn's words can move
    the mood: two exposures put the EMA at 84% of the situation's value) followed by the production feel query. The
    order is fixed (joy, fear, anger, sadness). `swap=True` is the PERMUTED-EXPOSURE arm: each situation keeps its
    frame words but carries the OTHER situation's affect words (joy<->sadness, fear<->anger), so the expected category
    follows the affective content, not the position or the frame."""
    turns = [("feel_base", "how do you feel", "feel")]
    for cat in ("joy", "fear", "anger", "sadness"):
        msg = SITUATIONS[cat]
        if swap:
            src, dst = SITUATION_AFFECT_WORDS[cat], SITUATION_AFFECT_WORDS[SITUATION_SWAP[cat]]
            for i, w in enumerate(src):
                msg = re.sub(r"\b%s\b" % re.escape(w), dst[i % len(dst)], msg)
        turns += [("s_%s_1" % cat, msg, "expose"), ("s_%s_2" % cat, msg, "expose"),
                  ("feel_%s" % cat, "how do you feel", "feel")]
    return turns


_CONV_PROBES = [
    ("p_fox_chase", "what does the fox chase", "probe"),         # a bounded transitive recall: grammatical + grounded
    ("p_more", "tell me more", "probe"),                         # the discourse follow-up elaborates the held topic
    ("p_deer_sleep", "where does the deer sleep", "probe"),      # a PREPOSITIONAL frame: must keep its preposition
    ("p_owl_hunt", "what does the owl hunt", "probe"),           # comprehension of the PASSIVE teach: owl hunts mole
    ("c_mole_hunt", "what does the mole hunt", "control"),       # moat on the passive's patient (intact: abstain)
    ("c_dog_chase", "what does the dog chase", "control"),       # a built-in fact survives every arm
    ("c_oov", "what does the zorblat chase", "control"),         # an OOV subject: abstain, never a confabulation
]
_INFER_PROBES = [
    ("p_prey_eat", "what does the heron's prey eat", "probe"),   # the 2-hop conclusion never told (heron->frog->gnat)
    ("c_heron_hunt", "what does the heron hunt", "control"),     # premise 1 recalled directly
    ("c_frog_eat", "what does the frog eat", "control"),         # premise 2 recalled directly
    ("c_prey_chase", "what does the heron's prey chase", "control"),   # moat: hop 2 has no stored fact -> abstain
    ("p_yesno", "does crow chase moth", "probe"),                # transitive yes/no over a non-adjacent pair (reported;
    ("c_dog_chase", "what does the dog chase", "control"),       #   counted only in the informational transitive arm)
]
# the `ask` probes in three BLOCKS (amendment A4 names them so the reversed-order arm can reverse them); the registered
# order is DISCRIM + LEARNABLE + UNLEARNABLE, unchanged since the prereg (pinned by tests/test_stage2_exit_gate_a4.py)
_ASK_DISCRIM = [
    ("g_eel", "what does the eel chase", "probe"),               # NOVEL-topic gap: abstain + the follow-up question
    ("m_yak", "what does the yak chase", "probe"),               # TAUGHT-topic moat: abstain, no follow-up
    ("g_crab", "what does the crab chase", "probe"),
    ("m_moose", "what does the moose chase", "probe"),
    ("g_newt", "what does the newt chase", "probe"),
    ("m_goat", "what does the goat chase", "probe"),
]
_ASK_LEARNABLE = [
    ("lp_ibex_1", "what does the ibex chase", "probe"),          # a LEARNABLE gap: asked ...
    ("t_ibex", "the ibex chases the marmot", "teach"),           # ... the teacher answers ...
    ("lp_ibex_2", "what does the ibex chase", "probe"),          # ... the gap closes: recalled, not asked again
]
_ASK_UNLEARNABLE = [
    ("u_toad_1", "what does the toad chase", "probe"),           # an UNLEARNABLE gap: asked, never answered ...
    ("c_dog_1", "what does the dog chase", "control"),
    ("u_toad_2", "what does the toad chase", "probe"),
    ("c_dog_2", "what does the dog chase", "control"),
    ("u_toad_3", "what does the toad chase", "probe"),
    ("c_dog_3", "what does the dog chase", "control"),
    ("u_toad_4", "what does the toad chase", "probe"),           # ... zero learning progress: the asking must STOP
]
_ASK_PROBES = _ASK_DISCRIM + _ASK_LEARNABLE + _ASK_UNLEARNABLE
_ASK_TEACH = [("t_yak", "the yak eats the grass", "teach"), ("t_moose", "the moose eats the leaf", "teach"),
              ("t_goat", "the goat eats the hay", "teach")]
# amendment A4 (b): the REVERSED-ORDER arm. The teach block stays first (it is the exposure the moats need, not a probe
# block); the three probe blocks run in reverse order, each block's own order unchanged (the learnable block's teach turn
# must sit between its two probes). Its own session, a COUNTED arm.
_ASK_REVERSED = _ASK_TEACH + _ASK_UNLEARNABLE + _ASK_LEARNABLE + _ASK_DISCRIM
# the topic each `what does the <x> chase` probe asks about (the moat rule reads it)
ASK_PROBE_TOPIC = {l: m.split()[3] for l, m, _r in _ASK_PROBES if m.startswith("what does the ")}
_INNER_PROBES = [
    ("p_private", "say something", "probe"),                     # PRIVATE speech: an idle turn voices a stored thought
    ("p_chase_all", "what does the lynx chase all the way", "probe"),      # INNER chain: lynx->vole->beetle->mite
    ("p_chase_all_vole", "what does the vole chase all the way", "probe"),  # a 2-hop start
    ("c_lynx_chase", "what does the lynx chase", "control"),     # the single hop (identity)
    ("c_mite_all", "what does the mite chase all the way", "control"),      # chain tail (intact: abstain)
    ("c_beetle_all", "what does the beetle chase all the way", "control"),  # chain tail in the permuted arm
    ("c_dog_chase", "what does the dog chase", "control"),
]

GROUPS = {
    "conv": {
        "criterion": "C1", "primary": True, "env": {},
        "turns": [("t_fox_chase", "the fox chases the hare", "teach"),
                  ("t_fox_eat", "the fox eats the berry", "teach"),
                  ("t_deer_sleep", "the deer sleeps in the meadow", "teach"),
                  ("t_mole_passive", "the mole is hunted by the owl", "teach")] + _CONV_PROBES,
        "permuted_turns": [("t_fox_chase", "the hare chases the fox", "teach"),
                           ("t_fox_eat", "the fox eats the berry", "teach"),
                           ("t_deer_sleep", "the deer sleeps in the meadow", "teach"),
                           ("t_mole_passive", "the owl is hunted by the mole", "teach")] + _CONV_PROBES,
        "lesions": {"lesion_mouth": {"env": {"BRAIN_SPIKING_MOUTH": "0"}, "patch": None,
                                     "cuts": "the spiking Broca mouth (master kill: the recall surface falls back to "
                                             "the host template)"}},
        "informational": {},
    },
    "infer": {
        "criterion": "C2", "primary": True, "env": {},
        "turns": [("t_heron_hunt", "the heron hunts the frog", "teach"),
                  ("t_frog_eat", "the frog eats the gnat", "teach"),
                  ("t_crow_chase", "the crow chases the wren", "teach"),
                  ("t_wren_chase", "the wren chases the moth", "teach")] + _INFER_PROBES,
        "permuted_turns": [("t_heron_hunt", "the heron hunts the gnat", "teach"),
                           ("t_frog_eat", "the gnat eats the frog", "teach"),
                           ("t_crow_chase", "the crow chases the wren", "teach"),
                           ("t_wren_chase", "the wren chases the moth", "teach")] + _INFER_PROBES,
        "lesions": {"lesion_chain_route": {"env": {"BRAIN_CHAIN_ROUTE": "0"}, "patch": None,
                                           "cuts": "the possessive-chain 2-hop route (a HOST dispatch at HEAD: regex + "
                                                   "role-noun table + kb scan; declared in the prereg's audit)"}},
        "informational": {"transitive_chat": {"env": {"BRAIN_TRANSITIVE_CHAT": "1"}, "patch": None,
                                              "note": "the default-OFF transitive yes/no route over the spiking "
                                                      "workspace chase (2026-09-24 prereg); reported, never counted"},
                          "qp_lesion": {"env": {}, "patch": "qp_lesion",
                                        "note": "the composer's spiking query_patient read returns None: does the "
                                                "derived answer need it at all? (ownership; reported, never counted)"}},
    },
    "ask": {
        "criterion": "C3", "primary": True, "env": {},
        "turns": _ASK_TEACH + _ASK_PROBES,
        "permuted_turns": [("t_yak", "the eel eats the grass", "teach"), ("t_moose", "the crab eats the leaf", "teach"),
                           ("t_goat", "the newt eats the hay", "teach")] + _ASK_PROBES,
        "lesions": {"lesion_curiosity": {"env": {"BRAIN_CURIOSITY_LESION": "1"}, "patch": None,
                                         "cuts": "the curiosity drive pathway (from_novelty -> the spiking ASK pool)"},
                    "lesion_graded": {"env": {"BRAIN_CURIOSITY_GRADED_NOVELTY_LESION": "1"}, "patch": None,
                                      "cuts": "the graded per-topic novelty read (every topic reads ceiling novelty)"}},
        # amendment A4 (b): COUNTED order arms (production defaults, nothing installed; only the turn order differs)
        "order_arms": {"reversed": {"turns": _ASK_REVERSED,
                                    "note": "the probe blocks in reverse order (unlearnable, learnable, discrimination) "
                                            "after the same teach block; every C3 correctness clause must hold here too"}},
        # amendment A4 (c): the DA/engagement crave-threshold coupling pinned to gain 1.0 (reported, never counted)
        "informational": {"da_lesion": {"env": {"BRAIN_CURIOSITY_DA_LESION": "1"}, "patch": None,
                                        "note": "the default-ON DA/engagement crave-threshold modulation pinned to 0 "
                                                "(gain 1.0): does the order confound vanish, and does the unlearnable "
                                                "gap still stop? reported, never counted"}},
    },
    "emo": {
        "criterion": "C4", "primary": True, "env": {},
        "turns": _situation_turns(),
        "permuted_turns": _situation_turns(swap=True),
        "lesions": {"lesion_affect": {"env": {"BRAIN_AFFECT_LESION": "1"}, "patch": None,
                                      "cuts": "the graded-affect ladder read (affect_out clamped to 0)"}},
        "informational": {},
    },
    "inner": {
        "criterion": "C5", "primary": True, "env": {},
        "turns": [("t_lynx", "the lynx chases the vole", "teach"), ("t_vole", "the vole chases the beetle", "teach"),
                  ("t_beetle", "the beetle chases the mite", "teach")] + _INNER_PROBES,
        "permuted_turns": [("t_lynx", "the lynx chases the mite", "teach"), ("t_vole", "the mite chases the vole", "teach"),
                           ("t_beetle", "the vole chases the beetle", "teach")] + _INNER_PROBES,
        "lesions": {"lesion_workspace": {"env": {"BRAIN_GNW_MULTISTEP_LESION": "1"}, "patch": None,
                                         "cuts": "the GNW workspace's assembly self-recurrence (the re-entrant chase "
                                                 "cannot sustain ignition)"},
                    "lesion_selfinit": {"env": {"BRAIN_SELF_INITIATE_LESION": "1"}, "patch": None,
                                        "cuts": "the self-initiation store (no-encode: nothing surfaces to voice)"}},
        "informational": {},
    },
}
PRIMARY_GROUPS = tuple(g for g, d in GROUPS.items() if d["primary"])
CRITERIA = ("C1", "C2", "C3", "C4", "C5")


def order_arms(group):
    """The COUNTED order arms of a group (amendment A4: `ask` -> ("reversed",); every other group none)."""
    return tuple(GROUPS[group].get("order_arms") or {})


def turns_for(group, seed, arm):
    g = GROUPS[group]
    if arm == "permuted":
        return list(g["permuted_turns"])
    base = str(arm).split(".rep")[0]
    if base in (g.get("order_arms") or {}):
        return list(g["order_arms"][base]["turns"])
    return list(g["turns"])


def arms_for(group, with_repeats=True, with_informational=True):
    g = GROUPS[group]
    arms = ["intact_a", "intact_b"]
    for les in g["lesions"]:
        arms.append(les)
        if with_repeats:
            arms.append(les + ".rep0")
    arms.append("permuted")
    arms += list(order_arms(group))            # amendment A4: counted, so emitted, required and validated like the rest
    if with_informational:
        arms += list(g["informational"])
    return arms


def arm_spec(group, arm):
    """(env, patch, kind) for an arm. kind in intact | lesion | permuted | order | informational."""
    g = GROUPS[group]
    base = arm.split(".rep")[0]
    if base in ("intact_a", "intact_b"):
        return {}, None, "intact"
    if base == "permuted":
        return {}, None, "permuted"
    if base in (g.get("order_arms") or {}):
        return {}, None, "order"
    if base in g["lesions"]:
        d = g["lesions"][base]
        return dict(d["env"]), d["patch"], "lesion"
    if base in g["informational"]:
        d = g["informational"][base]
        return dict(d["env"]), d["patch"], "informational"
    raise ValueError("unknown arm %s for group %s" % (arm, group))


# ══════════════════════════════════════════════ WORKER (inside the arm subprocess) ═════════════════════════════════
def install_patch(name, seed):
    """The in-process manipulations (no sim/ or webapp/ edit). Returns a record dict the worker stores."""
    if name is None:
        return None
    if name == "qp_lesion":
        from research.runners import one_brain_composer as OBC
        rec = {"patch": name, "calls": 0}
        orig = OBC.OneBrainComposer.query_patient

        def _query_patient(self, *a, **k):       # the spiking recall read answers nothing
            rec["calls"] += 1
            return None
        OBC.OneBrainComposer.query_patient = _query_patient
        rec["orig"] = repr(orig)
        return rec
    raise ValueError(name)


def _facts_of(chat):
    """The brain's stored facts at this moment, as (agent, action, patient) string triples (the C1/C5 grounding set)."""
    out = []
    try:
        for f in getattr(chat, "stored_facts", None) or []:
            if isinstance(f, (list, tuple)) and len(f) == 3 and all(isinstance(x, str) for x in f):
                out.append([f[0], f[1], f[2]])
    except Exception:
        pass
    return out


def _find_chat(srv, session):
    """The session's live ChatBrain in the handler's session cache (keyed (session, brain, renderer)), or None."""
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
    """Read-only copies of the per-turn traces the handler stashes on the chat and does not attach to the response
    (`_last_gnw_multistep`: the C5 chase's acted/reason/terminal/resolved_hops/cycles). `prev_ids` (name -> id of the
    object seen last turn) drops a trace no wrapper rewrote this turn (each wrapper assigns a NEW dict per call)."""
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
    """The worker process's peak RSS (the lane's mem_gb/memcap claim is checked against it) and the LTM bundle the
    handler resolved (the shipped default is wikidata_100k with wikidata_core_15k as the fallback; both yield the
    same `source` string, so the bundle is recorded per arm)."""
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


def run_worker(seed, group, arm, out_path, labels=None):
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    env, patch, kind = arm_spec(group, arm)
    for k, v in dict(GROUPS[group]["env"], **env).items():
        os.environ[k] = v
    os.environ["BRAIN_CHAT_SEED"] = str(int(seed))
    patch_rec = install_patch(patch, seed)
    hash_ok = os.environ.get("PYTHONHASHSEED") == "0"
    # the corpus-learned production organs read data/corpus/tinystories.txt from the repo root and DEGRADE silently
    # without it (Stage-1 packet, section 3 (d)); recorded per arm and required by the aggregate's preconditions.
    corpus_present = os.path.exists(os.path.join(_REPO, "data", "corpus", "tinystories.txt"))
    t0 = time.time()
    from webapp.server import brain_chat, BrainChatRequest
    import_s = time.time() - t0
    turns = turns_for(group, seed, arm)
    if labels:
        keep = set(labels.split(","))
        turns = [t for t in turns if t[0] in keep]
    session = "s2_%s_%s_%d" % (group, arm.replace(".", "_"), int(seed))
    responses, timings, internal = {}, {}, {}
    builtin_facts = None
    from webapp import server as _srv
    prev_ids = {}
    for i, (label, msg, role) in enumerate(turns):
        t1 = time.time()
        try:
            r = brain_chat(BrainChatRequest(session=session, message=msg, brain="tiny-demo", renderer="stub",
                                            reset=(i == 0)))      # `rich` OMITTED = the production default path
            responses[label] = json.loads(r.body)
        except Exception as e:
            responses[label] = {"_error": "%s: %s" % (type(e).__name__, e)}
        timings[label] = round(time.time() - t1, 1)
        chat = _find_chat(_srv, session)
        # the internal traces the handler stashes on the chat but never attaches to the response (the C5 chase trace:
        # prereg 4.5's internal secondary); only a trace this turn wrote is kept (object identity), never a stale one
        rec = capture_internal(chat, prev_ids)
        if rec:
            internal[label] = rec
        if builtin_facts is None:
            # the build-time fact list (read once, after the first turn built the brain; the grounding set for C1/C5)
            builtin_facts = _facts_of(chat) if chat is not None else []
    out = {"runner": "research/runners/stage2_exit_gate.py", "prereg": PREREG,
           "seed": int(seed), "group": group, "arm": arm, "kind": kind, "criterion": GROUPS[group]["criterion"],
           "env": dict(GROUPS[group]["env"], **env), "patch": patch_rec, "pythonhashseed_ok": hash_ok,
           "corpus_present": corpus_present, "builtin_facts": builtin_facts,
           "session": session, "turns": [{"label": l, "message": m, "role": r} for l, m, r in turns],
           "responses": responses, "internal": internal, "timings_s": timings, "import_s": round(import_s, 1),
           "wall_s": round(time.time() - t0, 1), "elapsed_seconds": round(time.time() - t0, 1),
           "expected_source": EXPECTED_SOURCE, "process": process_record(_srv),
           "sources": sorted({str(r.get("source")) for r in responses.values() if isinstance(r, dict)})}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("[stage2 worker] seed=%d %s/%s -> %d turns, %.0fs -> %s" % (seed, group, arm, len(responses),
                                                                     out["wall_s"], out_path), flush=True)
    return 0


def run_shard(seed, group, arm, out_path, backend="numpy", labels=None, allow_gate_seed=False, prereg=None):
    """One (seed, group, arm) arm in a FRESH subprocess with the frozen env (PYTHONHASHSEED must be set before the
    interpreter starts, so it cannot be applied in-process)."""
    env = dict(os.environ)
    env.update(BASE_ENV)
    env["SIM_BACKEND"] = backend
    if backend == "cupy":
        env.pop("CUDA_VISIBLE_DEVICES", None)
    part = out_path + ".part.json"
    cmd = [sys.executable, "-u", "-m", "research.runners.stage2_exit_gate", "--worker", "--seed", str(int(seed)),
           "--group", group, "--arm", arm, "--out", part]
    if labels:
        cmd += ["--labels", labels]
    if allow_gate_seed:
        # the worker's own main() applies the gate-seed guard too: without the flag every gate shard dies rc=2
        cmd += ["--allow-gate-seed"]
    if prereg:
        cmd += ["--prereg", prereg]
    p = subprocess.run(cmd, env=env, cwd=_REPO)
    if p.returncode != 0 or not os.path.exists(part):
        print("[stage2 shard] worker FAILED rc=%s for seed=%s %s/%s" % (p.returncode, seed, group, arm), flush=True)
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
    """The reply-level primary fields (docs/TERMS.md): what the user sees on that turn."""
    if not isinstance(resp, dict) or "_error" in resp:
        return None
    return {"answer": str(resp.get("answer")), "abstained": bool(resp.get("abstained")),
            "recalled_svo": (list(resp["recalled_svo"]) if resp.get("recalled_svo") else None),
            "derived": bool(resp.get("derived"))}


def _patient(resp):
    """The patient the reply names: the recalled SVO's, else a derived answer's terminal hop, else None."""
    p = _primary(resp)
    if not p:
        return None
    if p["recalled_svo"] and len(p["recalled_svo"]) == 3:
        return str(p["recalled_svo"][2]).lower()
    if p["derived"]:
        df = resp.get("derived_from") or []
        if df and isinstance(df[-1], (list, tuple)) and len(df[-1]) == 3:
            return str(df[-1][2]).lower()
        fr = frames(p["answer"])
        if fr:
            return fr[-1][3]
    return None


def _abstained(resp):
    p = _primary(resp)
    return None if p is None else p["abstained"]


_FRAME_RE = re.compile(r"\bthe ([a-z]+) ([a-z]+) (?:(in|on|at|to) )?the ([a-z]+)\b", re.IGNORECASE)
_TEMPLATE_RE = re.compile(r"\bThe [a-z]+ [a-z]+ [a-z]+\.")     # the host template stub's surface: "The robin flys sky."


def frames(answer):
    """Every bounded frame `the S V [prep] the O` in the reply text -> [(S, Vform, prep|None, O)], lowercased."""
    return [(m.group(1).lower(), m.group(2).lower(), (m.group(3) or None), m.group(4).lower())
            for m in _FRAME_RE.finditer(str(answer or ""))]


def grammatical(answer):
    """C1's fixed grammar: at least one frame; every frame's verb is a known lemma in its 3sg form; a taught
    prepositional fact keeps its preposition and a transitive one carries none; no template-stub surface anywhere."""
    a = str(answer or "")
    fr = frames(a)
    if not fr or _TEMPLATE_RE.search(a):
        return False
    for s, v, prep, o in fr:
        lemma = _3SG_TO_LEMMA.get(v)
        if lemma is None:
            return False
        want_prep = TAUGHT_PREPS.get((s, lemma, o))
        if (want_prep or None) != (prep or None):
            return False
    return True


def grounded(answer, facts):
    """Every frame in the reply names a fact the brain holds (taught this session or built in)."""
    fr = frames(answer)
    if not fr:
        return False
    fset = {tuple(f) for f in facts}
    return all((s, _3SG_TO_LEMMA.get(v), o) in fset for s, v, _p, o in fr)


_ASK_RE = re.compile(r"what can you tell me about", re.IGNORECASE)


def asked(resp):
    """The reply-level read of the curiosity follow-up: the honest QUESTION frame is in the reply text."""
    p = _primary(resp)
    if p is None:
        return None
    return bool(_ASK_RE.search(p["answer"]))


_DIFF_RE = re.compile(r"valence differential ([+-]?\d+\.\d+)")


def parse_feel(answer):
    """The reply-level read of a feel turn: the sign word, the differential and the arousal band PRINTED IN THE REPLY,
    plus the discrete category the reply names (None when it names none)."""
    a = str(answer or "")
    m = _DIFF_RE.search(a)
    d = float(m.group(1)) if m else None
    sign = "positive" if "reads positive" in a else ("negative" if "reads negative" in a else
                                                     ("neutral" if "reads neutral" in a else None))
    aro = "high" if "high arousal" in a else ("low" if "low arousal" in a else ("moderate" if "moderate arousal" in a else None))
    low = a.lower()
    cats = [c for c, words in EMOTION_LEXICON.items() if any(re.search(r"\b%s\b" % re.escape(w), low) for w in words)]
    return {"differential": d, "sign": sign, "arousal": aro, "category": (cats[0] if len(cats) == 1 else None),
            "categories_named": cats}


def _multistep_info(resp):
    """Secondary (internal) read of the self-cued chain: the multistep chase's trace dict, wherever the handler
    attached it (a dict carrying `resolved_hops`/`cycles`)."""
    if not isinstance(resp, dict):
        return None
    for k, v in resp.items():
        if isinstance(v, dict) and "cycles" in v and "resolved_hops" in v:
            return v
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, dict) and "cycles" in v2 and "resolved_hops" in v2:
                    return v2
    return None


# groups whose scorer grounds replies against the arm's `builtin_facts` (an empty list would read as a FAIL)
GROUNDING_GROUPS = ("conv", "inner")


def organ_errors(resp):
    """Faculty sub-dicts of one response that carry an `error` (the handler catches an organ exception into
    `<organ>.error` and keeps going; FAILURE_LOG 2026-09-29, B2b: such a turn was scored as a lesion effect).
    Scans two levels deep; returns the dotted keys."""
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


def arm_validity(group, arm, shard):
    """Why one COUNTED arm cannot be scored (empty list = valid). Prereg section 7: every response's source is the
    expected brain, PYTHONHASHSEED=0 and the corpus present; an errored reply is UNDEFINED, never a pass or a fail
    (a lesion arm whose scored turn crashed must not read as 'no effect', nor an organ error as an effect); the
    grounding set is recorded wherever a scorer reads it. The reason strings are matched by the preconditions."""
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
    if group in GROUNDING_GROUPS and not (shard.get("builtin_facts") or []):
        why.append("%s: builtin_facts not recorded (the grounding set the scorer reads)" % arm)
    return why


def null_clean(a, b, labels):
    """The null: two independent same-seed intact builds give the SAME primary reply fields on every listed turn."""
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
    """How many primary reply FIELDS differ between two arms over the listed turns (the effect SIZE the attribution
    subtracts: the lesion's count is the treatment, the null's intact_a-vs-intact_b count is the control)."""
    n = 0
    for l in labels:
        pa, pb = _primary((a or {}).get(l)), _primary((b or {}).get(l))
        if pa is None or pb is None:
            continue
        n += sum(1 for k in pa if pa[k] != pb[k])
    return n


def named_category_changed(intact_scored, lesion_scored):
    """C4, amendment A3: per feel turn, did the lesion change the discrete category the reply NAMES (not merely the
    number it prints)? A reply that names nothing under both arms did not change it: the lesion shows no dependence of
    the category on the ladder there. Inputs are `score_emo` outputs."""
    return {c: intact_scored["named"][c] != lesion_scored["named"][c] for c in intact_scored["named"]}


def private_selected_by_state(intact_scored, permuted_scored):
    """C5, amendment A3: the idle turn's voiced fact must depend on what the brain was exposed to. The two registered
    contexts are the intact and the permuted exposure (the same `say something` turn after different teaching): the
    voiced triples must both be present (each stored fact is already required by `private_voices_stored_fact`) and
    DIFFER. A brain that voices one fixed stored fact on every idle turn fails this. Inputs are `score_inner` outputs."""
    iu, pu = intact_scored.get("private_utterance"), permuted_scored.get("private_utterance")
    return bool(iu) and bool(pu) and list(iu) != list(pu)


# ══════════════════════════════════════════════ SCORING (pure functions over shard dicts) ══════════════════════════
def _canonical_teach(message):
    """A canonical active-voice teach sentence `the S Vs [prep] the O` -> [S, lemma, O], else None."""
    m = re.match(r"^the ([a-z]+) ([a-z]+) (?:(?:in|on|at|to) )?the ([a-z]+)$", str(message or ""))
    if m and _3SG_TO_LEMMA.get(m.group(2)):
        return [m.group(1), _3SG_TO_LEMMA[m.group(2)], m.group(3)]
    return None


def _taught(shard_turns):
    """The facts the teacher asserted this session, as the host extractor would store a canonical active-voice
    3-content-word sentence (the PASSIVE teach is deliberately NOT listed here: what the brain stored from it is what
    the probes measure)."""
    facts = []
    for t in shard_turns or []:
        if t.get("role") != "teach":
            continue
        f = _canonical_teach(t["message"])
        if f:
            facts.append(f)
    return facts


def teach_acquired(resp, fact):
    """Amendment A4 (a): did the brain ACQUIRE this canonical teach? Read at the reply level on the teach turn itself:
    the turn did not abstain and its `recalled_svo` (the handler's echo of what it stored) is exactly the taught triple.
    A role-binding abstain ("... didn't resolve the PATIENT", dev 7 `the goat eats the hay`) is not acquired."""
    p = _primary(resp)
    if p is None or p["abstained"] or not p["recalled_svo"] or len(p["recalled_svo"]) != 3:
        return False
    return [str(x).lower() for x in p["recalled_svo"]] == [str(x).lower() for x in fact]


def acquired_topics(shard):
    """{agent: acquired?} over the shard's own canonical teach turns (a topic taught twice counts if either landed)."""
    resp = shard.get("responses") or {}
    out = {}
    for t in shard.get("turns") or []:
        if t.get("role") != "teach":
            continue
        f = _canonical_teach(t.get("message"))
        if f:
            out[f[0]] = bool(out.get(f[0])) or teach_acquired(resp.get(t.get("label")), f)
    return out


def score_conv(shard, permuted=False):
    """C1. Every scored reply is a well-formed bounded frame naming a true fact; the prepositional frame keeps its
    preposition; the PASSIVE teach is comprehended (owl hunts mole); the elaboration brings a new grounded fact; the
    moats hold. Permuted: fox/hare and owl/mole swapped -> the replies must follow."""
    resp = shard.get("responses") or {}
    facts = list(shard.get("builtin_facts") or []) + _taught(shard.get("turns"))
    if not permuted:
        facts.append(["owl", "hunt", "mole"])          # what a comprehended passive stores
    else:
        facts.append(["mole", "hunt", "owl"])
    fset = {tuple(f) for f in facts}
    a_fox, a_more, a_deer = [(resp.get(k) or {}).get("answer") for k in ("p_fox_chase", "p_more", "p_deer_sleep")]
    fr_fox, fr_more = frames(a_fox), frames(a_more)
    ok = {}
    if not permuted:
        ok["fox_chase"] = (_patient(resp.get("p_fox_chase")) == "hare" and grammatical(a_fox) and grounded(a_fox, facts))
        ok["more"] = (grammatical(a_more) and grounded(a_more, facts)
                      and any((s, _3SG_TO_LEMMA.get(v), o) != ("fox", "chase", "hare") for s, v, _p, o in fr_more))
        ok["owl_hunt"] = _patient(resp.get("p_owl_hunt")) == "mole"
        ok["mole_moat"] = _abstained(resp.get("c_mole_hunt")) is True
    else:
        ok["fox_chase_follows"] = _abstained(resp.get("p_fox_chase")) is True     # the fox no longer chases anything
        ok["owl_hunt_follows"] = _abstained(resp.get("p_owl_hunt")) is True       # the owl is now the hunted
        ok["mole_hunts_owl"] = _patient(resp.get("c_mole_hunt")) == "owl"
    ok["deer_sleep"] = (grammatical(a_deer) and any((s, v, p, o) == ("deer", "sleeps", "in", "meadow") for s, v, p, o in frames(a_deer)))
    ok["builtin_dog_cat"] = _patient(resp.get("c_dog_chase")) == "cat"
    ok["oov_abstains"] = _abstained(resp.get("c_oov")) is True
    ok["all"] = all(v for v in ok.values())
    # reported sub-bars (never counted separately; the criterion counts `all`)
    ok["sub_transitive"] = bool(ok.get("fox_chase", ok.get("fox_chase_follows")) and ok["builtin_dog_cat"] and ok.get("more", True))
    ok["sub_prepositional"] = ok["deer_sleep"]
    ok["sub_passive"] = bool(ok.get("owl_hunt", ok.get("owl_hunt_follows")) and ok.get("mole_moat", ok.get("mole_hunts_owl")))
    ok["frames_fox"] = fr_fox
    ok["frames_more"] = fr_more
    ok["frames_deer"] = frames(a_deer)
    ok["passive_stored_as"] = _patient(resp.get("c_mole_hunt"))
    ok["n_facts_known"] = len(fset)
    return ok


def score_infer(shard, permuted=False):
    """C2. The 2-hop conclusion (heron -> frog -> gnat) is named on the possessive-chain probe as a DERIVED answer; both
    premises recall directly; the moat holds. Permuted premises (heron hunts gnat; gnat eats frog) -> the conclusion
    follows (frog). The transitive yes/no probe is reported (counted only in the informational arm's own line)."""
    resp = shard.get("responses") or {}
    want, prem1, prem2_abstain = ("frog", "gnat", True) if permuted else ("gnat", "frog", False)
    p = resp.get("p_prey_eat") or {}
    ok = {"conclusion": _patient(p) == want,
          "derived_flag": bool(p.get("derived")),
          "derived_lead": "I derived this from" in str(p.get("answer") or ""),
          "premise1": _patient(resp.get("c_heron_hunt")) == prem1,
          "moat_prey_chase": _abstained(resp.get("c_prey_chase")) is True,
          "builtin_dog_cat": _patient(resp.get("c_dog_chase")) == "cat"}
    if permuted:
        ok["premise2_abstains"] = _abstained(resp.get("c_frog_eat")) is True     # the frog eats nothing now
    else:
        ok["premise2"] = _patient(resp.get("c_frog_eat")) == "gnat"
    ok["all"] = all(v for k, v in ok.items() if k not in ("derived_lead",))
    y = resp.get("p_yesno") or {}
    ok["yesno_reply"] = str(y.get("answer") or "")[:160]
    ok["yesno_derived_moth"] = bool(y.get("derived")) and _patient(y) == "moth"     # reported only
    ok["yesno_misparsed_as_first_hop"] = (not bool(y.get("derived"))) and _patient(y) == "wren"
    return ok


def score_ask(shard, permuted=False):
    """C3. Ask-rate on NOVEL-topic gaps >= 2x the rate on TAUGHT-topic moats (and >= 2/3 itself); the learnable gap is
    asked once, answered, then recalled and not asked; the unlearnable gap is asked at first and NOT asked by its 4th
    probe (zero learning progress stops the asking). Permuted: the OTHER set was taught, so the asks must follow.
    Reads by LABEL, so it scores the amendment-A4 reversed-order arm unchanged. Amendment A4 (a): the moat rate counts
    only the taught moats this arm ACQUIRED (`acquired_topics`); an unacquired one is reported in `moats_undefined`."""
    resp = shard.get("responses") or {}
    gaps, moats = (["m_yak", "m_moose", "m_goat"], ["g_eel", "g_crab", "g_newt"]) if permuted else \
                  (["g_eel", "g_crab", "g_newt"], ["m_yak", "m_moose", "m_goat"])
    # amendment A4 (a): a taught moat whose teach turn this arm did NOT acquire is UNDEFINED: excluded from the moat
    # rate and never moved into the gap set (it is reported, below). The gap set is always the three never-taught topics.
    acq = acquired_topics(shard)
    moats_def = [l for l in moats if acq.get(ASK_PROBE_TOPIC[l]) is True]
    moats_undef = [l for l in moats if l not in moats_def]
    a_gap = [asked(resp.get(l)) for l in gaps]
    a_moat = [asked(resp.get(l)) for l in moats_def]
    rate_gap = (sum(1 for x in a_gap if x) / float(len(a_gap))) if all(x is not None for x in a_gap) else None
    rate_moat = ((sum(1 for x in a_moat if x) / float(len(a_moat)))
                 if (len(moats_def) >= ASK_MOAT_MIN_DEFINED and all(x is not None for x in a_moat)) else None)
    ok = {"ask_rate_gap": rate_gap, "ask_rate_moat": rate_moat,
          "discriminates": (rate_gap is not None and rate_moat is not None and rate_gap >= ASK_GAP_FLOOR
                            and rate_gap >= ASK_RATIO_BAR * rate_moat),
          "lp_learnable": (asked(resp.get("lp_ibex_1")) is True and _patient(resp.get("lp_ibex_2")) == "marmot"
                           and asked(resp.get("lp_ibex_2")) is False),
          "lp_unlearnable": (asked(resp.get("u_toad_1")) is True and asked(resp.get("u_toad_4")) is False),
          "all_gaps_abstain": all(_abstained(resp.get(l)) is True for l in gaps + moats + ["lp_ibex_1", "u_toad_1", "u_toad_4"]),
          "builtin_dog_cat": all(_patient(resp.get(l)) == "cat" for l in ("c_dog_1", "c_dog_2", "c_dog_3"))}
    ok["all"] = bool(ok["discriminates"] and ok["lp_learnable"] and ok["lp_unlearnable"] and ok["all_gaps_abstain"]
                     and ok["builtin_dog_cat"])
    # amendment A4 (a), reported: which taught moats counted, and whether the moat rate exists at all (the aggregate
    # makes a seed UNDEFINED, never FAIL, when it does not)
    ok["n_moats_defined"] = len(moats_def)
    ok["moats_undefined"] = moats_undef
    ok["moat_rate_defined"] = len(moats_def) >= ASK_MOAT_MIN_DEFINED
    ok["moat_items"] = {l: {"topic": ASK_PROBE_TOPIC[l], "teach_acquired": bool(acq.get(ASK_PROBE_TOPIC[l])),
                            "asked": asked(resp.get(l)), "counted": l in moats_def} for l in moats}
    ok["asked_by_label"] = {l: asked(resp.get(l)) for l in gaps + moats + ["lp_ibex_1", "lp_ibex_2", "u_toad_1", "u_toad_2", "u_toad_3", "u_toad_4"]}
    # secondary (internal) reads: the organ's own novelty / want / threshold per probe
    sec = {}
    for l in gaps + moats + ["u_toad_1", "u_toad_4"]:
        c = (resp.get(l) or {}).get("curiosity") or {}
        if isinstance(c, dict):
            sec[l] = {"curious": c.get("curious"), "want_hz": c.get("want_hz"), "threshold": c.get("threshold"),
                      "novelty": ((c.get("graded_novelty") or {}).get("value") if isinstance(c.get("graded_novelty"), dict) else None),
                      "topic": c.get("topic")}
    ok["curiosity_internal"] = sec
    return ok


def score_emo(shard, permuted=False):
    """C4. After each situation the feel reply must NAME the intended discrete emotion category (joy / fear / anger /
    sadness), four distinct categories, all correct. Reported beside it (never counted): the 2-D read (sign + arousal
    band), its sign correctness and how many distinct 2-D reads the four situations produced."""
    resp = shard.get("responses") or {}
    cats = ("joy", "fear", "anger", "sadness")
    reads = {c: parse_feel((resp.get("feel_%s" % c) or {}).get("answer")) for c in cats}
    want = {c: (SITUATION_SWAP[c] if permuted else c) for c in cats}
    named = {c: reads[c]["category"] for c in cats}
    ok = {"named": named, "want": want,
          "correct_each": {c: named[c] == want[c] for c in cats},
          "n_distinct_named": len({v for v in named.values() if v}),
          "signs": {c: reads[c]["sign"] for c in cats}, "arousal": {c: reads[c]["arousal"] for c in cats},
          "differentials": {c: reads[c]["differential"] for c in cats}}
    ok["all"] = all(ok["correct_each"].values()) and ok["n_distinct_named"] == 4
    want_sign = {c: ("positive" if want[c] == "joy" else "negative") for c in cats}
    ok["sign_correct"] = {c: reads[c]["sign"] == want_sign[c] for c in cats}
    ok["sign_correct_all"] = all(ok["sign_correct"].values())
    ok["n_distinct_2d"] = len({(reads[c]["sign"], reads[c]["arousal"]) for c in cats if reads[c]["sign"] is not None})
    ok["n_parsed"] = sum(1 for c in cats if reads[c]["differential"] is not None)
    return ok


def score_inner(shard, permuted=False):
    """C5. PRIVATE: the idle turn voices an internally selected STORED fact (the self-initiated frame, never the neutral
    idle line). INNER: the chase-form probes reach the never-told chain terminal (lynx -> vole -> beetle -> mite) with
    the hops internal (the trace, secondary); the single hop and the chain-tail moat hold. Permuted chain
    (lynx -> mite -> vole -> beetle) -> the terminal follows (beetle) and the moat moves to the beetle."""
    resp = shard.get("responses") or {}
    facts = list(shard.get("builtin_facts") or []) + _taught(shard.get("turns"))
    fset = {tuple(f) for f in facts}
    pv = resp.get("p_private") or {}
    a = str(pv.get("answer") or "")
    m = re.search(r"Something's been on my mind\s+[-—]+\s+([a-z]+) ([a-z]+) ([a-z]+)", a)
    utt = (m.group(1), m.group(2), m.group(3)) if m else None
    private = bool(utt) and (utt in fset or (utt[0], utt[1], utt[2]) in fset)
    si = pv.get("self_initiated") if isinstance(pv.get("self_initiated"), dict) else None
    if permuted:
        term, tail, other_tail = "beetle", "c_beetle_all", "c_mite_all"
        single, tail2_patient = "mite", "beetle"
    else:
        term, tail, other_tail = "mite", "c_mite_all", "c_beetle_all"
        single, tail2_patient = "vole", "mite"
    ok = {"private_voices_stored_fact": private, "private_utterance": (list(utt) if utt else None),
          "private_internal_n_utt": (si or {}).get("n_utt") if si else None,
          "inner_terminal_lynx": _patient(resp.get("p_chase_all")) == term,
          "inner_terminal_vole": _patient(resp.get("p_chase_all_vole")) == term,
          "single_hop": _patient(resp.get("c_lynx_chase")) == single,
          "moat_tail": _abstained(resp.get(tail)) is True,
          "other_tail_terminal": _patient(resp.get(other_tail)) == tail2_patient,
          "builtin_dog_cat": _patient(resp.get("c_dog_chase")) == "cat"}
    ok["all"] = bool(ok["private_voices_stored_fact"] and ok["inner_terminal_lynx"] and ok["inner_terminal_vole"]
                     and ok["single_hop"] and ok["moat_tail"] and ok["builtin_dog_cat"])
    def _trace(label):
        rec = ((shard.get("internal") or {}).get(label) or {}).get("gnw_multistep")
        return rec if isinstance(rec, dict) and "resolved_hops" in rec else _multistep_info(resp.get(label))
    info = _trace("p_chase_all")
    iv = _trace("p_chase_all_vole")
    ok["chain_internal_vole"] = ({"cycles": iv.get("cycles"), "resolved_hops": iv.get("resolved_hops"),
                                  "terminal": iv.get("terminal"), "reason": iv.get("reason")} if iv else None)
    ok["chain_internal"] = ({"cycles": info.get("cycles"), "resolved_hops": info.get("resolved_hops"),
                             "terminal": info.get("terminal"), "reason": info.get("reason")} if info else None)
    ok["chain_hops_internal_ge2"] = bool(info and (info.get("resolved_hops") or 0) >= 2)
    return ok


SCORERS = {"conv": score_conv, "infer": score_infer, "ask": score_ask, "emo": score_emo, "inner": score_inner}
# the turns a lesion is scored on (a primary reply field must differ from intact_a on at least one of them)
SCORED_LABELS = {
    "conv": {"lesion_mouth": ["p_fox_chase", "c_dog_chase"]},
    "infer": {"lesion_chain_route": ["p_prey_eat"]},
    "ask": {"lesion_curiosity": ["g_eel", "g_crab", "g_newt", "lp_ibex_1", "u_toad_1"],
            "lesion_graded": ["m_yak", "m_moose", "m_goat"]},
    "emo": {"lesion_affect": ["feel_joy", "feel_fear", "feel_anger", "feel_sadness"]},
    "inner": {"lesion_workspace": ["p_chase_all", "p_chase_all_vole"], "lesion_selfinit": ["p_private"]},
}


ASK_TAUGHT_MOATS = ("m_yak", "m_moose", "m_goat")       # the taught-topic moats of the registered (non-permuted) teaching


def effective_scored_labels(group, les, intact_shard, lesion_shard):
    """The lesion's scored turns. Amendment A4 (a), `ask` only: a taught moat whose teach turn was not acquired in BOTH
    intact_a and the lesion arm is dropped (an effect on a topic the brain never learned is not a graded-novelty
    effect). Only removes turns, so it can only make the lesion clause harder to meet."""
    labs = list(SCORED_LABELS[group][les])
    if group != "ask":
        return labs
    ai, al = acquired_topics(intact_shard or {}), acquired_topics(lesion_shard or {})
    return [l for l in labs if l not in ASK_TAUGHT_MOATS
            or (ai.get(ASK_PROBE_TOPIC[l]) is True and al.get(ASK_PROBE_TOPIC[l]) is True)]


def _moat_reason(arm, scored):
    """Amendment A4 (a): the UNDEFINED reason when an arm's taught-moat ask-rate does not exist (matched by the
    stage precondition's tag `taught moat acquired`)."""
    if scored.get("moat_rate_defined", True):
        return []
    n_def = int(scored.get("n_moats_defined") or 0)
    und = scored.get("moats_undefined") or []
    return ["%s: taught moat acquired on %d of %d (need >= %d; undefined %s): the moat ask-rate is undefined (A4 (a))"
            % (arm, n_def, n_def + len(und), ASK_MOAT_MIN_DEFINED, ",".join(und))]


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


def _labels(group, seed, arm="intact_a"):
    return [t[0] for t in turns_for(group, seed, arm)]


def aggregate(root, seeds=GATE_SEEDS, informational=True):
    shards = _load_shards(root, seeds)
    seeds = [int(s) for s in seeds]
    per_group = {}
    missing = []
    for group in GROUPS:
        g = GROUPS[group]
        rows, defined = {}, []
        for seed in seeds:
            need = [a for a in arms_for(group, with_informational=False)]
            have = {a: shards.get((seed, group, a)) for a in arms_for(group)}
            for a in need:
                if have[a] is None:
                    missing.append(shard_path(root, seed, group, a))
            row = {"seed": seed, "undefined_reasons": []}
            ia, ib = have.get("intact_a"), have.get("intact_b")
            if ia is None or ib is None or "_load_error" in (ia or {}) or "_load_error" in (ib or {}):
                row["undefined_reasons"].append("intact arm missing")
            else:
                labels = _labels(group, seed)
                ok, diffs = null_clean(ia["responses"], ib["responses"], labels)
                row["null_clean"], row["null_diffs"] = ok, diffs[:8]
                if not ok:
                    row["undefined_reasons"].append("null failed (intact_a != intact_b) on %d turn(s)" % len(diffs))
                srcs = set(ia.get("sources") or []) | set(ib.get("sources") or [])
                row["sources"] = sorted(srcs)
                for les in g["lesions"]:
                    for arm_name in (les, les + ".rep0"):
                        sh = have.get(arm_name)
                        if sh is None or "responses" not in sh:
                            row["undefined_reasons"].append("%s arm missing or unreadable" % arm_name)
                if have.get("permuted") is None or "responses" not in (have.get("permuted") or {}):
                    row["undefined_reasons"].append("permuted arm missing or unreadable")
                for oa in order_arms(group):          # amendment A4 (b): a counted arm, so missing = UNDEFINED
                    if have.get(oa) is None or "responses" not in (have.get(oa) or {}):
                        row["undefined_reasons"].append("%s arm missing or unreadable" % oa)
                # every COUNTED arm must be a valid measurement (prereg section 7), not only the intact pair
                for arm_name in need:
                    sh = have.get(arm_name)
                    if sh is not None and "responses" in sh:
                        row["undefined_reasons"] += arm_validity(group, arm_name, sh)
                row["ltm_bundles"] = sorted({str(((have.get(a) or {}).get("process") or {}).get("ltm_bundle"))
                                             for a in need if have.get(a) is not None})
                ra, rb = ia["responses"], ib["responses"]
                # ── correctness (intact_a) ──
                row["intact"] = SCORERS[group](ia)
                row["correct"] = bool(row["intact"]["all"])
                if group == "emo" and row["intact"]["n_parsed"] < 4:
                    row["undefined_reasons"].append("only %d of 4 feel reads parsed" % row["intact"]["n_parsed"])
                if group == "ask":
                    row["undefined_reasons"] += _moat_reason("intact_a", row["intact"])
                # ── amendment A4 (b): every correctness clause must hold in EVERY counted order, not only the
                #    registered one (so the probe order cannot carry the verdict) ──
                if order_arms(group):
                    row["correct_by_order"] = {"registered": bool(row["intact"]["all"])}
                    row["orders"] = {}
                    for oa in order_arms(group):
                        O = have.get(oa)
                        if O is not None and "responses" in O:
                            so = SCORERS[group](O)
                            row["orders"][oa] = so
                            row["correct_by_order"][oa] = bool(so["all"])
                            if group == "ask":
                                row["undefined_reasons"] += _moat_reason(oa, so)
                        else:
                            row["correct_by_order"][oa] = None
                    row["correct"] = all(v is True for v in row["correct_by_order"].values())
                # ── lesion: the reply-level effect must be present AND reproduced ──
                row["lesions"] = {}
                for les in g["lesions"]:
                    L, R = have.get(les), have.get(les + ".rep0")
                    rec = {"present": L is not None, "repeat_present": R is not None}
                    if L is not None and "responses" in L:
                        labs = effective_scored_labels(group, les, ia, L)
                        if group == "ask" and labs != SCORED_LABELS[group][les]:
                            rec["scored_labels_dropped_A4"] = [l for l in SCORED_LABELS[group][les] if l not in labs]
                        if not labs:
                            row["undefined_reasons"].append("%s: no scored turn left after the A4 (a) moat rule" % les)
                        d = [reply_differs(ra, L["responses"], l) for l in labs]
                        rec["differs"] = dict(zip(labs, d)); rec["effect"] = any(x is True for x in d)
                        if group == "emo":
                            # amendment A3: the lesion must change the NAMED category on every feel turn, not only the
                            # printed number (a category read off the heard words survives a ladder lesion)
                            rec["reply_effect"] = rec["effect"]
                            rec["category_changed"] = named_category_changed(row["intact"], SCORERS[group](L))
                            rec["effect"] = bool(rec["effect"]) and all(rec["category_changed"].values())
                        # ATTRIBUTION (tools.lab): treatment = primary-field differences intact_a-vs-lesion on the
                        # scored turns; control = the same count intact_a-vs-intact_b (the run-to-run variation).
                        rec["attributable"] = attributable_to("%s/%s seed %d (primary reply fields on %s)" % (group, les, seed, ",".join(labs)),
                                                              float(n_field_diffs(ra, L["responses"], labs)), float(n_field_diffs(ra, rb, labs)))
                        rec["lesion_score"] = {k: v for k, v in SCORERS[group](L).items()
                                               if k in ("all", "correct_each", "discriminates", "ask_rate_gap", "ask_rate_moat",
                                                        "inner_terminal_lynx", "private_voices_stored_fact", "conclusion", "fox_chase")}
                        if R is not None and "responses" in R:
                            rec["reproduced"], _ = null_clean(L["responses"], R["responses"], _labels(group, seed))
                        else:
                            rec["reproduced"] = None
                        p = L.get("patch")
                        if isinstance(p, dict) and "calls" in p:
                            rec["patch_calls"] = p["calls"]; rec["applied"] = p["calls"] > 0
                        else:
                            rec["applied"] = True      # env-flag lesions are read per call by production code
                    else:
                        rec["effect"] = None; rec["reproduced"] = None
                    row["lesions"][les] = rec
                # ── permuted exposure: the reply must follow the exposure ──
                P = have.get("permuted")
                if P is not None and "responses" in P:
                    sp = SCORERS[group](P, permuted=True); row["permuted"] = sp
                    if group == "emo":
                        # tracking is only DEFINED when the intact arm names the categories at all
                        row["tracks"] = sp["all"] if row["correct"] else None
                    elif group == "inner":
                        # amendment A3: the voiced idle fact must follow the exposure too (a fixed fact does not)
                        row["private_selected_by_state"] = private_selected_by_state(row["intact"], sp)
                        row["tracks"] = bool(sp["all"]) and row["private_selected_by_state"]
                    else:
                        row["tracks"] = bool(sp["all"])
                        if group == "ask":
                            # amendment A4 (a): the permuted teaching's own moats (eel/crab/newt) need their teach acquired
                            row["undefined_reasons"] += _moat_reason("permuted", sp)
                else:
                    row["tracks"] = None
                # ── informational arms (never counted) ──
                if informational:
                    row["informational"] = {}
                    for inf in g["informational"]:
                        I = have.get(inf)
                        if I is not None and "responses" in I:
                            si = SCORERS[group](I)
                            row["informational"][inf] = {k: v for k, v in si.items()
                                                         if k in ("all", "conclusion", "derived_flag", "yesno_derived_moth",
                                                                  "yesno_misparsed_as_first_hop", "yesno_reply",
                                                                  # amendment A4 (c): the DA-lesion arm's C3 reads
                                                                  "ask_rate_gap", "ask_rate_moat", "discriminates",
                                                                  "lp_learnable", "lp_unlearnable", "moats_undefined",
                                                                  "asked_by_label")}
                            pi = I.get("patch")
                            if isinstance(pi, dict) and "calls" in pi:
                                row["informational"][inf]["patch_calls"] = pi["calls"]
            rows[seed] = row
            if not row["undefined_reasons"] and "correct" in row:
                defined.append(seed)
        n_def = len(defined)
        n_correct = sum(1 for s in defined if rows[s]["correct"])
        les_names = list(g["lesions"])
        n_les = {les: sum(1 for s in defined if (rows[s]["lesions"].get(les) or {}).get("effect") is True) for les in les_names}
        # amendment A3 (emo): how many seeds the printed number moved vs the NAMED category moved under each lesion
        n_les_num = {les: sum(1 for s in defined if (rows[s]["lesions"].get(les) or {}).get("reply_effect") is True) for les in les_names}
        n_rep = {les: sum(1 for s in defined if (rows[s]["lesions"].get(les) or {}).get("reproduced") is True) for les in les_names}
        n_tracks = sum(1 for s in defined if rows[s].get("tracks") is True)
        n_tr_def = sum(1 for s in defined if rows[s].get("tracks") is not None)
        verdict = "INCOMPLETE" if n_def < len(seeds) else (
            "PASS" if (n_correct == n_def and all(n_les[l] == n_def for l in les_names)
                       and all(n_rep[l] == n_def for l in les_names) and n_tr_def == n_def and n_tracks == n_def)
            else "FAIL")
        why = []
        # amendment A4 (b): how many defined seeds were correct in each counted order
        n_by_order = {}
        if order_arms(group):
            for o in ("registered",) + order_arms(group):
                n_by_order[o] = sum(1 for s in defined if (rows[s].get("correct_by_order") or {}).get(o) is True)
        if verdict == "FAIL":
            if n_correct < n_def:
                per_order = ", ".join("%s %d of %d" % (o, k, n_def) for o, k in n_by_order.items())
                why.append("capability absent on %d of %d seeds%s" % (
                    n_def - n_correct, n_def, (" (correct per order: %s)" % per_order) if per_order else ""))
            else:
                for l in les_names:
                    if n_les[l] < n_def:
                        why.append("%s: reply-level effect on %d of %d seeds (not load-bearing)%s" % (
                            l, n_les[l], n_def, (" [the printed number moved on %d, the named category on every turn on %d]"
                                                 % (n_les_num[l], n_les[l])) if group == "emo" else ""))
                    if n_rep[l] < n_def:
                        why.append("%s: reproduced on %d of %d seeds" % (l, n_rep[l], n_def))
                if n_tracks < n_tr_def:
                    why.append("permuted exposure tracked on %d of %d seeds (not exposure-driven)" % (n_tracks, n_tr_def))
                elif n_tr_def < n_def:
                    why.append("permuted tracking undefined on %d of %d seeds" % (n_def - n_tr_def, n_def))
        elif verdict == "INCOMPLETE":
            why.append("%d of %d seeds defined" % (n_def, len(seeds)))
        # the sub-clauses, counted over the defined seeds (reported beside the verdict; never a pass condition alone)
        sub = {}
        for s in defined:
            it = rows[s]["intact"]
            for k, v in it.items():
                if k.startswith("sub_") or k in ("discriminates", "lp_learnable", "lp_unlearnable", "private_voices_stored_fact",
                                                 "inner_terminal_lynx", "conclusion", "derived_flag", "sign_correct_all",
                                                 "chain_hops_internal_ge2", "yesno_derived_moth"):
                    sub.setdefault(k, 0)
                    sub[k] += 1 if v else 0
        if group == "inner":
            sub["private_selected_by_state"] = sum(1 for s in defined if rows[s].get("private_selected_by_state"))
        for oa in order_arms(group):        # amendment A4 (b): the same clauses read in each counted order arm
            for k in ("all", "discriminates", "lp_learnable", "lp_unlearnable"):
                sub["%s_%s" % (oa, k)] = sum(1 for s in defined if ((rows[s].get("orders") or {}).get(oa) or {}).get(k))
        if group == "ask":
            sub["all_taught_moats_acquired"] = sum(1 for s in defined if not rows[s]["intact"].get("moats_undefined"))
        per_group[group] = {"criterion": g["criterion"], "primary": g["primary"], "verdict": verdict, "why": why,
                            "n_defined": n_def, "n_correct": n_correct, "n_lesion_effect": n_les,
                            "n_reproduced": n_rep, "n_permuted_tracks": n_tracks, "sub_clauses_n_true": sub, "seeds": rows,
                            "n_correct_by_order": n_by_order,
                            # amendment A4 reporting rule: the gaps a FAIL of C1 / C3 names (no bar change)
                            "named_gaps": list(A4_NAMED_GAPS.get(group, ())) if verdict == "FAIL" else []}
    crit = {}
    for c in CRITERIA:
        prim = [g for g in PRIMARY_GROUPS if GROUPS[g]["criterion"] == c][0]
        crit[c] = per_group[prim]["verdict"]
    # ── the verdict is EARNED (tools.verdict): every validity precondition measured and holding, else UNDEFINED ──
    from tools.verdict import Verdict
    prim_rows = [per_group[g]["seeds"][s] for g in PRIMARY_GROUPS for s in seeds]
    v = Verdict("Stage-2 (CHILD) exit on the integrated production brain")
    v.require("no primary shard missing", sum(1 for m in missing if any("/%s__" % g in m for g in PRIMARY_GROUPS)), expect=0)
    v.require("null clean (intact_a == intact_b) on every primary (group, seed)",
              sum(1 for r in prim_rows if r.get("null_clean") is not True), expect=0,
              note="a failed null makes that seed UNDEFINED; the criterion then cannot reach 6 defined seeds")
    def _n_rows_with(tag):
        return sum(1 for r in prim_rows if any(tag in x for x in r.get("undefined_reasons", [])))
    v.require("every response from the expected brain", _n_rows_with("unexpected brain source"), expect=0,
              note="expected `source` = %s in every counted arm" % EXPECTED_SOURCE)
    v.require("PYTHONHASHSEED=0 in every primary arm", _n_rows_with("PYTHONHASHSEED not 0"), expect=0)
    v.require("no errored turn and no organ error in any counted arm", _n_rows_with("errored turn") + _n_rows_with("organ error"),
              expect=0, note="an errored reply or a caught organ exception is an instrument failure, never an effect or a FAIL")
    v.require("the grounding set (builtin_facts) recorded in every conv/inner counted arm", _n_rows_with("builtin_facts not recorded"), expect=0)
    v.require("the corpus-learned production organs had their corpus (data/corpus/tinystories.txt) in every primary arm",
              sum(1 for r in prim_rows if any("corpus absent" in x for x in r.get("undefined_reasons", []))), expect=0,
              note="a worktree or pool checkout without the corpus builds a DEGRADED brain; pool_provision.sh syncs it")
    v.require("every in-process lesion applied (patch called at least once)",
              sum(1 for r in prim_rows for les in (r.get("lesions") or {}).values() if les.get("applied") is False), expect=0)
    v.require("the ask group's taught-moat ask-rate defined in every arm that reads it (amendment A4 (a))",
              _n_rows_with("A4 (a)"), expect=0,
              note="a taught moat whose teach turn was not acquired is UNDEFINED, never novel; with none acquired the "
                   "moat rate does not exist (an acquisition gap, not a curiosity result)")
    v.require("six seeds defined for every primary criterion", sum(1 for g in PRIMARY_GROUPS if per_group[g]["n_defined"] < len(seeds)), expect=0)
    earned = v.decide(go=all(x == "PASS" for x in crit.values()), verbose=False)
    if earned["status"] == "UNDEFINED":
        stage = "UNDEFINED (INCOMPLETE): " + "; ".join(earned["undefined_reasons"])[:400]
    else:
        stage = "PASS" if earned["go"] else "FAIL"
    return {"stage": "STAGE-2 (CHILD) exit", "verdict": stage, "GO": bool(earned["go"]),
            "preconditions": earned["preconditions"], "undefined_reasons": earned["undefined_reasons"],
            "criteria": crit, "groups": per_group,
            "seeds": seeds, "missing_shards": missing,
            "bars": {"ask_ratio": ASK_RATIO_BAR, "ask_gap_floor": ASK_GAP_FLOOR, "seeds_required": len(seeds),
                     "emotion_categories": 4},
            "runner": "research/runners/stage2_exit_gate.py", "prereg": PREREG, "expected_source": EXPECTED_SOURCE}


def report(agg):
    lines = ["**Stage-2 exit verdict: %s** (primary criteria: %s)" % (
        agg["verdict"], ", ".join("%s %s" % (k, v) for k, v in agg["criteria"].items())), "",
        "| group | criterion | verdict | correct | lesion effect | reproduced | permuted tracks | why |",
        "|---|---|---|---|---|---|---|---|"]
    for g, d in agg["groups"].items():
        lines.append("| %s | %s | %s | %d/%d | %s | %s | %d/%d | %s |" % (
            g, d["criterion"], d["verdict"], d["n_correct"], d["n_defined"],
            ", ".join("%s %d/%d" % (k, v, d["n_defined"]) for k, v in d["n_lesion_effect"].items()),
            ", ".join("%s %d/%d" % (k, v, d["n_defined"]) for k, v in d["n_reproduced"].items()),
            d["n_permuted_tracks"], d["n_defined"], "; ".join(d["why"]) or "-"))
    lines += ["", "Sub-clauses (seeds true / seeds defined; reported, never a pass condition alone):"]
    for g, d in agg["groups"].items():
        if d["sub_clauses_n_true"]:
            lines.append("- %s: %s" % (g, ", ".join("%s %d/%d" % (k, v, d["n_defined"]) for k, v in d["sub_clauses_n_true"].items())))
    named = [(g, d) for g, d in agg["groups"].items() if d.get("named_gaps")]
    if named:
        lines += ["", "Named gaps (amendment A4 reporting rule; printed for a FAIL of C1 / C3, whichever sub-clause failed):"]
        for g, d in named:
            lines.append("- %s (%s): %s" % (d["criterion"], g, "; ".join(d["named_gaps"])))
    ask = agg["groups"]["ask"]["seeds"]
    lines += ["", "Per-seed curiosity ask-rates (novel-topic gaps vs taught-topic moats; the bar is gap >= 2/3 and gap >= 2x moat; "
                  "amendment A4: every clause in BOTH orders, a taught moat not acquired is undefined):"]
    for s, row in ask.items():
        it = row.get("intact") or {}
        lines.append("- seed %s: gap %s, moat %s, learnable %s, unlearnable-stops %s, moats undefined %s; %s" % (
            s, it.get("ask_rate_gap"), it.get("ask_rate_moat"), it.get("lp_learnable"), it.get("lp_unlearnable"),
            it.get("moats_undefined") or "none", "; ".join(row.get("undefined_reasons") or []) or "defined"))
        for oa, so in (row.get("orders") or {}).items():
            lines.append("  - %s order: gap %s, moat %s, learnable %s, unlearnable-stops %s, moats undefined %s" % (
                oa, so.get("ask_rate_gap"), so.get("ask_rate_moat"), so.get("lp_learnable"), so.get("lp_unlearnable"),
                so.get("moats_undefined") or "none"))
        for inf, si in (row.get("informational") or {}).items():
            lines.append("  - %s (informational, never counted): gap %s, moat %s, learnable %s, unlearnable-stops %s" % (
                inf, si.get("ask_rate_gap"), si.get("ask_rate_moat"), si.get("lp_learnable"), si.get("lp_unlearnable")))
    emo = agg["groups"]["emo"]["seeds"]
    lines += ["", "Per-seed discrete-emotion read (category named per situation; the 2-D sign/arousal read beside):"]
    for s, row in emo.items():
        it = row.get("intact") or {}
        lines.append("- seed %s: named %s; signs %s; arousal %s; distinct 2-D reads %s" % (
            s, it.get("named"), it.get("signs"), it.get("arousal"), it.get("n_distinct_2d")))
    if agg["missing_shards"]:
        lines += ["", "Missing shards: %d" % len(agg["missing_shards"])]
    return "\n".join(lines)


# ══════════════════════════════════════════════ POOL JOBS ═══════════════════════════════════════════════════════════
def pool_job_lines(sha, seeds=GATE_SEEDS, out_dir=GATE_OUT_DIR, informational=True, max_age_h=MAX_AGE_H):
    """One pool job per (seed, group, arm). Dev seed 7 peaked at 6.3-8.1 GB RSS (emo highest) -> mem_gb=MEM_GB under a
    MEMCAP_GB cap. Every line declares max_age_h (the dispatcher's 12 h debris guard would otherwise skip a deep,
    legitimate queue)."""
    out = []
    max_age_h = min(int(max_age_h), 240)
    for seed in seeds:
        for group in GROUPS:
            for arm in arms_for(group, with_informational=informational):
                p = shard_path(out_dir, seed, group, arm)
                cmd = ("cd ~/derisk-pool/revisions/%s && bash tools/mem_ok.sh %d 1 && mkdir -p %s && "
                       "bash tools/memcap.sh %d -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                       "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m "
                       "research.runners.stage2_exit_gate --shard --seed %d --group %s --arm %s --allow-gate-seed "
                       "--prereg %s --out %s" % (sha, MEM_GB, os.path.dirname(p), MEMCAP_GB, seed, group, arm, PREREG, p))
                why = ("prereg %s (%s); Stage-2 exit gate seed %d %s %s; record: second integrated stage-exit "
                       "measurement (before_you_build + RAG 2026-09-30: only isolated de-risks and single-faculty "
                       "wire-ins exist); mem_gb=%d max_age_h=%d" % (PREREG, sha[:9], seed, group, arm, MEM_GB, max_age_h))
                out.append((cmd, why))
    return out


# ══════════════════════════════════════════════ SELFTEST (no brain) ════════════════════════════════════════════════
def _fake(answer, abstained=False, svo=None, extra=None):
    d = {"answer": answer, "abstained": abstained, "recalled_svo": svo, "source": EXPECTED_SOURCE}
    if extra:
        d.update(extra)
    return d


_AB = "I don't know about that."
_ASK = " My curiosity is piqued — I haven't learned about %s yet: what can you tell me about %s?"


def _fake_shard(group, responses, permuted=False, arm=None):
    arm = arm or ("permuted" if permuted else "intact_a")
    return {"seed": 1, "group": group, "arm": arm, "responses": responses, "sources": [EXPECTED_SOURCE],
            "pythonhashseed_ok": True, "builtin_facts": [["dog", "chase", "cat"], ["cat", "eat", "fish"]],
            "turns": [{"label": l, "message": m, "role": r} for l, m, r in turns_for(group, 1, arm)]}


_ROLE_ABSTAIN = ("I caught the verb 'eat' with the goat and the hay, but my role-binding didn't resolve the PATIENT — "
                 "which of them is the 'eat' done to, the goat or the hay?")      # dev 7's reply to `the goat eats the hay`


def _synthetic_ask(kind, unacquired=()):
    """A synthetic `ask` session: intact (asks on novel topics only, learnable closes, unlearnable stops) / lesioned
    (never asks) / graded-lesioned (asks everywhere) / permuted-consistent. The teach turns echo what was stored, as
    the live handler does (`recalled_svo` = the taught triple); `unacquired` names teach labels (t_yak/t_moose/t_goat)
    that end in dev 7's role-binding abstain instead (amendment A4 (a)), and that topic then reads NOVEL (asked)."""
    taught_msgs = {l: m for l, m, _r in turns_for("ask", 1, "permuted" if kind == "permuted" else "intact_a")
                   if l in ("t_yak", "t_moose", "t_goat")}
    r = {l: (_fake(_ROLE_ABSTAIN, True) if l in unacquired else _fake(m, svo=_canonical_teach(m)))
         for l, m in taught_msgs.items()}
    r["t_ibex"] = _fake("the ibex chases the marmot", svo=["ibex", "chase", "marmot"])
    for l in ("c_dog_1", "c_dog_2", "c_dog_3"):
        r[l] = _fake("the dog chases the cat", svo=["dog", "chase", "cat"])
    novel = {"g_eel": "eel", "g_crab": "crab", "g_newt": "newt"}
    taught = {"m_yak": "yak", "m_moose": "moose", "m_goat": "goat"}
    if kind == "permuted":
        novel, taught = ({"m_yak": "yak", "m_moose": "moose", "m_goat": "goat"}, {"g_eel": "eel", "g_crab": "crab", "g_newt": "newt"})
    ask_novel = kind != "lesion"
    ask_taught = kind == "graded_lesion"
    # a taught topic whose teach was NOT acquired is novel to this brain: it asks there exactly as on a novel gap
    unacq_topics = {(_canonical_teach(taught_msgs[l]) or [None])[0] for l in unacquired}
    for l, t in novel.items():
        r[l] = _fake(_AB + (_ASK % (t, t) if ask_novel else ""), True)
    for l, t in taught.items():
        r[l] = _fake(_AB + (_ASK % (t, t) if (ask_taught or (ask_novel and t in unacq_topics)) else ""), True)
    r["lp_ibex_1"] = _fake(_AB + (_ASK % ("ibex", "ibex") if ask_novel else ""), True)
    r["lp_ibex_2"] = _fake("the ibex chases the marmot", svo=["ibex", "chase", "marmot"])
    for l in ("u_toad_1", "u_toad_2", "u_toad_3"):
        r[l] = _fake(_AB + (_ASK % ("toad", "toad") if ask_novel else ""), True)
    r["u_toad_4"] = _fake(_AB + (_ASK % ("toad", "toad") if kind == "no_lp" else ""), True)
    return r


def _synthetic_feel(d, category=None, arousal="moderate"):
    mood = "positive" if d > MOOD_TOL else ("negative" if d < -MOOD_TOL else "neutral")
    cat = (" I would call it %s." % category) if category else ""
    return _fake("My affect monitor reads %s right now (valence differential %+.3f, %s arousal) -- that's a "
                 "functional read of my mood state, not a felt experience.%s" % (mood, d, arousal, cat))


def _synthetic_emo(kind):
    vals = {"joy": (0.3, "happy"), "fear": (-0.3, "afraid"), "anger": (-0.25, "angry"), "sadness": (-0.2, "sad")}
    r = {"feel_base": _synthetic_feel(0.0)}
    for c, (d, w) in vals.items():
        r["s_%s_1" % c] = _fake(_AB, True); r["s_%s_2" % c] = r["s_%s_1" % c]
        if kind == "discrete":
            r["feel_%s" % c] = _synthetic_feel(d, category=w)
        elif kind == "lesion":
            r["feel_%s" % c] = _synthetic_feel(0.0)
        else:   # production defaults: sign only, no category
            r["feel_%s" % c] = _synthetic_feel(d)
    return r


def selftest(verbose=True):
    import tempfile
    checks = {}
    checks["every group has intact_a, intact_b, >=1 lesion (+rep0), permuted"] = all(
        arms_for(g)[:2] == ["intact_a", "intact_b"] and "permuted" in arms_for(g)
        and any(a.endswith(".rep0") for a in arms_for(g)) for g in GROUPS)
    checks["exactly one primary group per criterion, five criteria"] = sorted(GROUPS[g]["criterion"] for g in PRIMARY_GROUPS) == list(CRITERIA)
    checks["permuted exposure differs from intact only on teach/expose turns, same labels"] = all(
        [(l, r) for l, _m, r in GROUPS[g]["turns"]] == [(l, r) for l, _m, r in GROUPS[g]["permuted_turns"]]
        and any(m1 != m2 for (_l, m1, _r), (_l2, m2, _r2) in zip(GROUPS[g]["turns"], GROUPS[g]["permuted_turns"]))
        and all(r in ("teach", "expose") for (_l, m1, r), (_l2, m2, _r2) in zip(GROUPS[g]["turns"], GROUPS[g]["permuted_turns"]) if m1 != m2)
        for g in GROUPS)
    checks["situations are never 3-content-word assertions and keep their frame under the swap"] = all(
        len([w for w in m.split() if w not in ("the", "a", "an", "is", "are", "and", "you", "your", "it")]) >= 4
        for _l, m, r in GROUPS["emo"]["turns"] + GROUPS["emo"]["permuted_turns"] if r == "expose")
    checks["grammar: the spiking frame passes, the template stub and a wrong inflection fail, the preposition is required"] = (
        grammatical("Setting the held thread aside — On fox, then — the fox chases the hare")
        and not grammatical("The fox chases hare.") and not grammatical("The robin flys sky.")
        and not grammatical("the deer sleeps the meadow") and grammatical("the deer sleeps in the meadow")
        and not grammatical(_AB))
    checks["grounded: a frame outside the fact set is not grounded"] = (
        grounded("the fox chases the hare", [["fox", "chase", "hare"]]) and not grounded("the fox chases the owl", [["fox", "chase", "hare"]]))
    checks["feel parser reads sign, differential, arousal and the named category"] = (
        parse_feel(_synthetic_feel(0.123, "happy", "high")["answer"]) == {"differential": 0.123, "sign": "positive", "arousal": "high", "category": "joy", "categories_named": ["joy"]}
        and parse_feel(_synthetic_feel(-0.05)["answer"])["category"] is None and parse_feel(_AB)["differential"] is None)
    checks["asked(): the follow-up frame is read from the reply text"] = (
        asked(_fake(_AB + _ASK % ("eel", "eel"), True)) is True and asked(_fake(_AB, True)) is False)
    sa = {k: score_ask(_fake_shard("ask", _synthetic_ask(k))) for k in ("intact", "lesion", "graded_lesion", "no_lp")}
    sp = score_ask(_fake_shard("ask", _synthetic_ask("permuted"), permuted=True), permuted=True)
    checks["ask scorer: intact passes; drive lesion and graded lesion fail discrimination; no-LP fails the stop clause; permuted tracks"] = (
        sa["intact"]["all"] and not sa["lesion"]["all"] and not sa["graded_lesion"]["discriminates"]
        and sa["no_lp"]["discriminates"] and not sa["no_lp"]["lp_unlearnable"] and sp["all"])
    se = {k: score_emo(_fake_shard("emo", _synthetic_emo(k))) for k in ("discrete", "defaults", "lesion")}
    checks["emo scorer: four named categories pass; sign-only defaults fail with signs correct; lesion reads neutral"] = (
        se["discrete"]["all"] and not se["defaults"]["all"] and se["defaults"]["sign_correct_all"] and se["defaults"]["n_distinct_named"] == 0
        and not se["lesion"]["sign_correct_all"])
    # end-to-end aggregate on a synthetic shard tree: PASS for ask (all arms), INCOMPLETE elsewhere (missing shards)
    with tempfile.TemporaryDirectory() as td:
        seeds = (1, 2)
        for seed in seeds:
            kinds = {"intact_a": "intact", "intact_b": "intact", "lesion_curiosity": "lesion", "lesion_curiosity.rep0": "lesion",
                     "lesion_graded": "graded_lesion", "lesion_graded.rep0": "graded_lesion", "permuted": "permuted",
                     "reversed": "intact"}
            for arm, kind in kinds.items():
                p = shard_path(td, seed, "ask", arm); os.makedirs(os.path.dirname(p), exist_ok=True)
                json.dump(dict(_fake_shard("ask", _synthetic_ask(kind), arm=arm), seed=seed, arm=arm), open(p, "w"))
        agg = aggregate(td, seeds=seeds)
        checks["aggregate: synthetic ask group PASSES 2/2; the stage is UNDEFINED while other groups are missing"] = (
            agg["groups"]["ask"]["verdict"] == "PASS" and agg["verdict"].startswith("UNDEFINED") and len(agg["missing_shards"]) > 0
            and agg["GO"] is False and len(agg["preconditions"]) >= 5 and all(c["ok"] is not None for c in agg["preconditions"]))
        checks["attribution: a genuine lesion effect with a clean null is 100% attributable"] = (
            agg["groups"]["ask"]["seeds"][1]["lesions"]["lesion_curiosity"]["attributable"] == 1.0)
        # a lesion arm with NO effect -> the group FAILs as 'not load-bearing' even though intact is correct
        for seed in seeds:
            for arm in ("lesion_curiosity", "lesion_curiosity.rep0"):
                json.dump(dict(_fake_shard("ask", _synthetic_ask("intact")), seed=seed, arm=arm), open(shard_path(td, seed, "ask", arm), "w"))
        agg2 = aggregate(td, seeds=seeds)
        checks["aggregate: correct-but-not-load-bearing reads FAIL with the reason named; no effect -> attribution UNDEFINED"] = (
            agg2["groups"]["ask"]["verdict"] == "FAIL" and any("not load-bearing" in w for w in agg2["groups"]["ask"]["why"])
            and agg2["groups"]["ask"]["seeds"][1]["lesions"]["lesion_curiosity"]["attributable"] is None)
        # the no-LP shape (asks every time on the unlearnable gap): capability absent, the sub-clause table shows which
        for seed in seeds:
            json.dump(dict(_fake_shard("ask", _synthetic_ask("no_lp")), seed=seed, arm="intact_a"), open(shard_path(td, seed, "ask", "intact_a"), "w"))
            json.dump(dict(_fake_shard("ask", _synthetic_ask("no_lp")), seed=seed, arm="intact_b"), open(shard_path(td, seed, "ask", "intact_b"), "w"))
        agg2c = aggregate(td, seeds=seeds)
        checks["aggregate: the predicted no-learning-progress shape reads FAIL 'capability absent' with discriminates 2/2 and lp_unlearnable 0/2"] = (
            agg2c["groups"]["ask"]["verdict"] == "FAIL" and any("capability absent" in w for w in agg2c["groups"]["ask"]["why"])
            and agg2c["groups"]["ask"]["sub_clauses_n_true"]["discriminates"] == 2 and agg2c["groups"]["ask"]["sub_clauses_n_true"]["lp_unlearnable"] == 0)
        # a MISSING permuted arm makes the seed UNDEFINED (an instrument failure), never a FAIL clause
        os.remove(shard_path(td, 2, "ask", "permuted"))
        agg2b = aggregate(td, seeds=seeds)
        checks["aggregate: a missing counted arm -> that seed UNDEFINED (INCOMPLETE), not FAIL"] = (
            agg2b["groups"]["ask"]["verdict"] == "INCOMPLETE" and agg2b["groups"]["ask"]["n_defined"] == 1
            and any("permuted arm missing" in r for r in agg2b["groups"]["ask"]["seeds"][2]["undefined_reasons"]))
        json.dump(dict(_fake_shard("ask", _synthetic_ask("permuted"), permuted=True), seed=2, arm="permuted"), open(shard_path(td, 2, "ask", "permuted"), "w"))
        # a failed null makes the seed UNDEFINED -> INCOMPLETE, never a pass or a fail
        json.dump(dict(_fake_shard("ask", _synthetic_ask("lesion")), seed=1, arm="intact_b"), open(shard_path(td, 1, "ask", "intact_b"), "w"))
        agg3 = aggregate(td, seeds=seeds)
        checks["aggregate: a failed null -> that seed UNDEFINED -> group INCOMPLETE"] = (
            agg3["groups"]["ask"]["verdict"] == "INCOMPLETE" and agg3["groups"]["ask"]["n_defined"] == 1)
        checks["report renders"] = "Stage-2 exit verdict" in report(agg3)
    # ── amendment A4 (2026-10-13), group `ask` ──
    checks["A4 (b): the reversed arm is counted, carries the same probes, the probe blocks reversed, the teach block first"] = (
        "reversed" in arms_for("ask", with_informational=False) and "reversed" not in arms_for("conv")
        and sorted(_labels("ask", 1, "reversed")) == sorted(_labels("ask", 1))
        and _labels("ask", 1, "reversed")[:3] == ["t_yak", "t_moose", "t_goat"]
        and _labels("ask", 1, "reversed")[3] == "u_toad_1" and _labels("ask", 1, "reversed")[-1] == "m_goat"
        and arm_spec("ask", "reversed")[2] == "order")
    s_goat = score_ask(_fake_shard("ask", _synthetic_ask("intact", unacquired=("t_goat",))))
    checks["A4 (a): an unacquired taught moat is undefined (reported), never counted as a moat nor as novel"] = (
        s_goat["moats_undefined"] == ["m_goat"] and s_goat["n_moats_defined"] == 2 and s_goat["ask_rate_moat"] == 0.0
        and s_goat["ask_rate_gap"] == 1.0 and s_goat["moat_items"]["m_goat"]["asked"] is True and s_goat["all"]
        and score_ask(_fake_shard("ask", _synthetic_ask("intact")))["moats_undefined"] == [])
    with tempfile.TemporaryDirectory() as td:
        seeds = (1, 2)
        kinds = {"intact_a": "intact", "intact_b": "intact", "lesion_curiosity": "lesion", "lesion_curiosity.rep0": "lesion",
                 "lesion_graded": "graded_lesion", "lesion_graded.rep0": "graded_lesion", "permuted": "permuted",
                 "reversed": "intact", "da_lesion": "intact"}

        def _write(over=None, unacq=()):
            for seed in seeds:
                for arm, kind in dict(kinds, **(over or {})).items():
                    p = shard_path(td, seed, "ask", arm); os.makedirs(os.path.dirname(p), exist_ok=True)
                    json.dump(dict(_fake_shard("ask", _synthetic_ask(kind, unacquired=unacq), arm=arm), seed=seed, arm=arm),
                              open(p, "w"))
        _write(over={"reversed": "lesion"})
        ag = aggregate(td, seeds=seeds)["groups"]["ask"]
        checks["A4 (b): correct in the registered order but not in the reversed one -> FAIL 'capability absent', per order"] = (
            ag["verdict"] == "FAIL" and any("capability absent" in w and "reversed 0 of 2" in w for w in ag["why"])
            and ag["n_correct_by_order"] == {"registered": 2, "reversed": 0}
            and ag["named_gaps"] == list(A4_NAMED_GAPS["ask"]))
        _write(unacq=("t_goat",))
        ag = aggregate(td, seeds=seeds)
        g = ag["groups"]["ask"]
        checks["A4 (a): goat not acquired in every arm -> still PASS; the graded lesion is scored on yak + moose only"] = (
            g["verdict"] == "PASS" and g["sub_clauses_n_true"]["all_taught_moats_acquired"] == 0
            and g["seeds"][1]["lesions"]["lesion_graded"]["scored_labels_dropped_A4"] == ["m_goat"]
            and "da_lesion" in g["seeds"][1]["informational"] and g["named_gaps"] == [])
        _write(unacq=("t_yak", "t_moose", "t_goat"))
        ag = aggregate(td, seeds=seeds)
        g = ag["groups"]["ask"]
        checks["A4 (a): no taught moat acquired -> the seed is UNDEFINED (precondition), never a FAIL"] = (
            g["verdict"] == "INCOMPLETE" and g["n_defined"] == 0 and ag["verdict"].startswith("UNDEFINED")
            and any("A4 (a)" in r for r in g["seeds"][1]["undefined_reasons"])
            and any(c["ok"] is False and "A4 (a)" in c["name"] for c in ag["preconditions"]))
        _write(over={"da_lesion": "lesion"})
        g = aggregate(td, seeds=seeds)["groups"]["ask"]
        checks["A4 (c): the DA-lesion arm is reported and never counted"] = (
            g["verdict"] == "PASS" and g["seeds"][1]["informational"]["da_lesion"]["discriminates"] is False
            and "da_lesion" not in arms_for("ask", with_informational=False))
    lines = pool_job_lines("0123456789abcdef" * 2 + "01234567")
    checks["pool jobs: one per (seed, group, arm), pinned, memcapped, prereg-tagged, gate flag + max_age_h present"] = (
        len(lines) == len(GATE_SEEDS) * sum(len(arms_for(g)) for g in GROUPS)
        and all(("memcap.sh %d" % MEMCAP_GB) in c and "--allow-gate-seed" in c and PREREG in c and "revisions/0123456789" in c
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
    ap.add_argument("--report", default=None, metavar="AGGREGATE_JSON", help="print the fixed Markdown result block")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit-pool-jobs", default=None, metavar="SHA", help="print the gate's pool job lines (TSV)")
    ap.add_argument("--max-age-h", type=int, default=MAX_AGE_H, help="max_age_h token on every pool line (cap 240)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds for --aggregate / --emit-pool-jobs")
    ap.add_argument("--group", default=None, choices=sorted(GROUPS))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--labels", default=None, help="DEV ONLY: run only these turn labels (comma-separated)")
    ap.add_argument("--backend", default="numpy", choices=("numpy", "cupy"))
    ap.add_argument("--prereg", default=None, help="the committed preregistration a GATE-seed run is governed by")
    ap.add_argument("--allow-gate-seed", action="store_true",
                    help="required to run a GATE seed (42/43/44/100/101/102); dev seeds need nothing")
    ap.add_argument("--no-informational", action="store_true", help="emit/aggregate without the informational arms")
    a = ap.parse_args()
    if a.selftest:
        ok, _ = selftest()
        sys.exit(0 if ok else 1)
    if a.emit_pool_jobs:
        seeds = tuple(int(s) for s in a.seeds.split(",")) if a.seeds else GATE_SEEDS
        for cmd, why in pool_job_lines(a.emit_pool_jobs, seeds=seeds, informational=not a.no_informational, max_age_h=a.max_age_h):
            print("%s\t%s" % (cmd, why))
        return
    if a.aggregate:
        seeds = tuple(int(s) for s in a.seeds.split(",")) if a.seeds else GATE_SEEDS
        with contextlib.redirect_stdout(sys.stderr):      # the attribution/verdict prints go to stderr: stdout is
            agg = aggregate(a.aggregate, seeds=seeds, informational=not a.no_informational)   # the report only
        out = a.out or os.path.join(a.aggregate, "aggregate.json")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(agg, f, indent=1, default=str)
        print(report(agg)); print("\n-> %s" % out)
        return
    if a.report:
        with open(a.report) as f:
            print(report(json.load(f)))
        return
    if a.worker or a.shard:
        if a.seed is None or not a.group or not a.arm or not a.out:
            ap.error("--seed, --group, --arm and --out are required")
        if a.seed in GATE_SEEDS and not a.allow_gate_seed:
            sys.stderr.write("GATE seed %d refused: a gate seed runs only from a pool job emitted against the committed "
                             "prereg (%s); pass --allow-gate-seed --prereg <path> from such a job. Dev seeds: 1-12.\n"
                             % (a.seed, PREREG))
            sys.exit(2)
        if a.worker:
            sys.exit(run_worker(a.seed, a.group, a.arm, a.out, labels=a.labels))
        sys.exit(run_shard(a.seed, a.group, a.arm, a.out, backend=a.backend, labels=a.labels,
                           allow_gate_seed=a.allow_gate_seed, prereg=a.prereg))
    ap.error("choose one of --worker/--shard/--aggregate/--report/--selftest/--emit-pool-jobs")


if __name__ == "__main__":
    main()
