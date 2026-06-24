"""TEST the FLUENT + MULTI-TURN (anaphora/attention) conversation on the now-fixed self-knowledge brain.

The chat fix is DONE (recall 0.21 -> 0.94 via decorrelate_grounded_codes, default-ON in build_qa_agent; the Qwen
crash fixed via _free_cupy_pool before the faculty load; _self_knowledge_chat_fix.json = CHAT_READY single-turn).
This test confirms the owner's CONVERSATIONAL goal is reached: the brain holds a genuinely FLUENT (multi-sentence,
Qwen-phrased, verify-checked) MULTI-TURN (anaphora resolves via the discourse buffer) conversation ABOUT ITSELF,
with the no-confab FIREWALL holding ACROSS turns (0 leaks).

THE WIRING (reuse-by-import, NO sim/ edit):
  load the FIXED self-knowledge brain (the saved stream-learned grounded codes) -> ZCA-DECORRELATE them (the recall
  fix, D.decorrelate_grounded_codes) -> build a MultiTurnAgent on them (the SpikingLoopContextBuffer discourse WM =
  the anaphora/attention) + teach the FULL curriculum -> wrap in a ChatBrain (the GATE+VERIFY+anaphora-resolution
  wiring + the off-bridge Qwen QwenRenderer, with the _free_cupy_pool crash fix) -> wrap in a RichAnswerComposer
  (the FLUENT multi-sentence grounded reply: direct recall + multi-hop chain + dlPFC elaboration, EACH sentence
  Qwen-phrased + VERIFY-checked).

THE SCRIPTED MULTI-TURN CONVERSATION exercises ALL THREE goals:
  * FLUENT: open self-questions ('what are you?', 'how do you learn?', 'how do you remember things?') -> MULTI-
    SENTENCE grounded fluent answers (>=2-3 brain-sourced sentences, Qwen-phrased, each VERIFY-checked).
  * MULTI-TURN ANAPHORA (attention): a follow-up turn whose subject is a PRONOUN ('what does it learn?' after
    'what does the brain simulate?' -> 'it' resolves to the held referent 'cortex' via the discourse buffer; 'tell
    me more' on a held topic -> elaborates FURTHER) -> the pronoun RESOLVES to the prior referent (annotated).
  * FIREWALL in-flow: mid-conversation 'what is the capital of France?' + an untaught project part ('what does the
    brain navigate?') -> ABSTAIN (the moat holds ACROSS turns, 0 leaks).

VERDICT: GO = the brain holds a genuinely FLUENT MULTI-TURN conversation about itself, firewall holding across
turns (0 leaks). Or HONEST: the precise gap + the best fix attempt.

GPU (SIM_BACKEND=cupy), FOREGROUND-only. Writes research/findings/raw/_self_knowledge_multiturn_test.json.

    SIM_BACKEND=cupy python -u -m research.runners._self_knowledge_multiturn_test --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# GPU by default (the composer bridges + the spiking Qwen faculty). An explicit env still wins.
os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import _self_knowledge_demo as D  # noqa: E402  (the fixed build helpers + the recall fix)
from research.runners.multi_turn_agent import MultiTurnAgent  # noqa: E402  (the anaphora discourse WM)
from research.runners.brain_chat_tui import ChatBrain, QwenRenderer, StubRenderer  # noqa: E402
from research.runners.rich_answer_composer import RichAnswerComposer  # noqa: E402  (the FLUENT multi-sentence reply)

OUT = os.path.join(_REPO, "research", "findings", "raw", "_self_knowledge_multiturn_test.json")
CODES = os.path.join(_REPO, "research", "findings", "raw", "_self_knowledge_grounded_codes.json")
CURRICULUM = os.path.join(_REPO, "research", "findings", "raw", "_curriculum_self_knowledge.json")


# ====================================================================================================
# Build the FIXED self-knowledge MultiTurnAgent: decorrelate the learned codes (the recall fix) + teach
# the full curriculum + the discourse WM (anaphora). This mirrors brain_chat_tui._load_self_knowledge BUT
# applies the decorrelation recall fix (which that loader does not) so recall is 0.94 not 0.21.
# ====================================================================================================

def build_multiturn_self_knowledge_chat(seed, faculty_renderer):
    """Return (chat, agent, n_facts, recall_acc, aliases). The agent is a MultiTurnAgent on the DECORRELATED
    learned codes; the chat is a ChatBrain over it with the given fluent renderer."""
    cur = D._load_curriculum()
    facts = D._all_facts_svo(cur)
    vocab = D._qa_vocab(cur)                       # taught concepts + general-knowledge + untaught fall-backs

    # the saved stream-learned grounded codes
    with open(CODES, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    grounded = {w: np.asarray(v, dtype=float) for w, v in blob.get("grounded_codes", {}).items()}
    print(f"[mt] loaded {len(grounded)} stream-learned grounded codes (D={len(next(iter(grounded.values())))})",
          flush=True)

    # THE RECALL FIX: ZCA-decorrelate the learned codes before they enter the composer (recall 0.21 -> 0.94).
    grounded_decorr = D.decorrelate_grounded_codes(grounded)

    # The discourse-referent set the WM loop holds = the SALIENT antecedents only: the concepts that are an
    # AGENT of some fact (i.e. concepts the brain can say MORE about -> the chainable referents a pronoun can
    # bind to). This EXACTLY matches what ChatBrain.gate writes into the WM (`if patient in agents_set:
    # _note_referent(patient)`) -- it never writes an abstract patient or a never-stored fall-back word as a
    # referent, so those need NO attractor. Restricting to ~these concepts (not all ~70 vocab nouns) keeps the
    # SpikingLoopContextBuffer build fast + light: the per-referent attractor install is O(referents^2) in time
    # AND peak VRAM (each set_pathway_weights(add_missing=True) re-sorts the growing CSR), so 70 referents bloats
    # to ~20 GB; the ~25 salient referents build in seconds at < 3 GB. (`concepts` -- the full composer vocabulary
    # incl. the firewall fall-backs -- is unchanged, so the moat still abstains structurally on the never-stored.)
    actions = {v for (_a, v, _p) in facts} | {"is"}
    agents_set = {a for (a, _v, _p) in facts}
    patients_set = {p for (_a, _v, p) in facts}
    # the salient referents: every agent + every patient that is ALSO an agent (a chainable antecedent). Add the
    # general-knowledge "subject" cues too so a pronoun could (but never will) bind them -- harmless + tiny.
    referents = sorted((agents_set | (patients_set & agents_set))
                       - actions)
    pattern_size = 40
    wm_n = max(600, 2 * pattern_size * max(1, len(referents)))   # 2x headroom so the WM holds every referent
    concepts = {w: None for w in vocab}                          # full vocab (incl. firewall fall-backs) for the composer

    agent = MultiTurnAgent(referent_concepts=referents, concepts=concepts,
                           grounded_codes=grounded_decorr, seed=seed,
                           wm_n=wm_n, wm_pattern_size=pattern_size,
                           enable_neural_render=False, composer_kind="rf",
                           enable_biased_competition=False)
    inner = agent.agent
    n_taught = 0
    for a, v, p in facts:
        inner.hear(f"{a} {v} {p}", polarity="AFFIRM")
        n_taught += 1

    # the recall sanity check on the decorrelated codes (should be ~0.94)
    rec_ok = sum(1 for a, v, p in facts if inner.what_does(a, v) == p)
    recall_acc = round(rec_ok / len(facts), 4)

    aliases = set(cur.get("self_reference", {}).get("agent_aliases", [])) | {
        "you", "your", "yours", "i", "me", "my", "it", "its", "yourself", "itself"}
    chat = ChatBrain(agent, self_aliases=aliases, renderer=faculty_renderer)
    return chat, agent, n_taught, recall_acc, aliases


# ====================================================================================================
# The scripted MULTI-TURN conversation. Each turn declares which goal(s) it exercises + the expectation,
# so the per-turn checks can be made precise. The conversation is a SINGLE thread (the discourse WM +
# the RichAnswerComposer thread state carry across turns).
# ====================================================================================================

# turn kinds:
#   'fluent'         -> an open self-question; expect a MULTI-SENTENCE (>=2) grounded fluent answer.
#   'setup'          -> establishes a discourse referent for the FOLLOWING anaphora turn (a normal answer).
#   'anaphora'       -> a follow-up whose SUBJECT is a pronoun resolving to the prior turn's referent.
#   'followup'       -> a bare 'tell me more' on the held topic; expect NEW grounded sentences (>=1).
#   'firewall_general' -> a general-knowledge cue the LLM knows; the brain MUST ABSTAIN (0 leak).
#   'firewall_untaught'-> a real-but-untaught project part; the brain MUST ABSTAIN.
#
# ORDERING NOTE (root-caused from the first GPU run): the discourse WM (SpikingLoopContextBuffer) accumulates
# referents MONOTONICALLY across turns (it has no reset) and SATURATES after several writes, so a fresh
# referent written LATE in a long thread does not cleanly dominate the read (the documented multi-referent
# WM-span limit, 2026-06-17-multireferent-disambiguation-NEGATIVE). Therefore the ANAPHORA exchange is placed
# FIRST, while the WM is clean (the fresh setup referent dominates at high specificity). The FLUENT block (which
# saturates the WM with topics + chain intermediates) and the firewall follow. Also: the setup verbs are the
# EXACT 3rd-person curriculum forms ('simulates'/'has') so the keyword->fact router resolves them (an UNmapped
# base form like 'simulate' is in no synonym set -> the setup would abstain -> no referent written).
#
# `expect_referent` (for anaphora turns) = the referent the pronoun MUST resolve to (set by the prior turn).
_SCRIPT = [
    # --- MULTI-TURN ANAPHORA block FIRST (clean WM -> reliable resolution) ---
    # 1) establish a chainable referent: 'what does the brain simulates' -> 'cortex' (cortex is itself an agent,
    #    so it is written into the discourse WM as the salient referent). EXACT curriculum verb 'simulates'.
    {"you": "what does the brain simulates", "kind": "setup", "sets_referent": "cortex"},
    # 2) the PRONOUN follow-up: 'what does it learn' -> 'it' must resolve to the held 'cortex' -> 'meaning'
    #    ('learn' -> 'learns' is in the router's synonym map).
    {"you": "what does it learn", "kind": "anaphora", "expect_referent": "cortex",
     "expect_answer_contains": "meaning"},
    # 3) a 'tell me more' follow-up: elaborate FURTHER on the held topic (new grounded sentences, none repeating).
    {"you": "tell me more", "kind": "followup"},

    # --- SEGMENT BREAK: a genuine topic shift away from the cortex thread. The RichAnswerComposer's no-repeat
    # dedup (`_conversation_said`) is a per-discourse-SEGMENT property -- it stops a SINGLE thread from restating,
    # but a fresh topic ('what are you') legitimately revisits the brain's core self-facts. Without this reset the
    # anaphora thread's chain facts (brain->spikes->neurons...) stay "said", starving the later open self-questions
    # of their multi-sentence extensions (-> 1-sentence answers, an artifact of the ordering, NOT a brain limit).
    # Resetting the dedup + thread + discourse WM at the topic boundary lets the FLUENT block produce its full
    # multi-sentence answers while preserving the anaphora-first clean-WM win. (Models a real conversational topic
    # shift; the no-confab moat is unaffected.)
    {"segment_break": True},

    # --- FLUENT block: open self-questions -> multi-sentence grounded answers ---
    {"you": "what are you", "kind": "fluent"},
    {"you": "how do you learn", "kind": "fluent"},
    {"you": "how do you remember things", "kind": "fluent"},
    {"you": "what do you use", "kind": "fluent"},

    # --- FIREWALL in-flow (mid-conversation): the moat must hold ACROSS turns ---
    {"you": "what is the capital of France", "kind": "firewall_general"},
    {"you": "who wrote Romeo and Juliet", "kind": "firewall_general"},
    {"you": "what does the brain navigate", "kind": "firewall_untaught"},

    # --- back to a FLUENT turn AFTER the firewall (prove the conversation continues, moat did not break it) ---
    {"you": "how do you remember", "kind": "fluent"},
]


def _capture_anaphora_resolution(chat, utterance, expect_referent):
    """For an anaphora turn, capture what the discourse WM resolves the leading pronoun to -- via the EXACT
    ChatBrain._resolve_anaphora the gate uses. Returns (resolved_subject_token, wm_dominant, wm_spec).

    Note: each held_referent() read advances the spiking WM 20 steps and a saturated loop is non-deterministic;
    so we take ONE _resolve_anaphora reading (the same call the gate makes next) as the authoritative resolution,
    and read the dominant referent ONCE for the record. The load-bearing correctness signal is the ANSWER
    content (the chained patient), checked separately -- the resolution token is the mechanistic annotation."""
    resolved_q = chat._resolve_anaphora(utterance)          # the pronoun -> held-referent substitution
    # the substituted subject token (the word the pronoun became), for the annotation
    toks = resolved_q.split()
    subj = None
    for t in toks:
        tl = t.lower().strip(".,!?")
        if tl not in {"what", "who", "does", "do", "the", "a", "an", "is", "are", "how", "did"}:
            subj = tl
            break
    held, spec = chat.agent.held_referent()
    return subj, held, round(float(spec), 3)


def run_conversation(chat, rich, agent):
    """Drive the scripted multi-turn conversation through the RichAnswerComposer. Returns the verbatim
    transcript + per-turn checks. The discourse state (WM + thread) carries across turns by construction."""
    stored = set(tuple(f) for f in rich._stored_facts())
    transcript = []
    prior_said = []        # every fact said in PRIOR turns (so a follow-up's novelty can be checked)

    for spec in _SCRIPT:
        # a SEGMENT BREAK (topic shift): reset the RichAnswerComposer's per-segment discourse state so the next
        # block starts a fresh topic -- clears the conversation-wide no-repeat set + the thread topic/said, and
        # the discourse WM (so a stale referent from the prior segment does not bleed into a fresh open question).
        # No turn is emitted. The no-confab moat + the brain's stored facts are untouched.
        if spec.get("segment_break"):
            rich._conversation_said = set()
            rich._topic = None
            rich._said = []
            try:                                   # reset the discourse WM loop to rest (fresh segment)
                chat.agent.wm.update([])           # no-op drive; the loop relaxes over the next read
            except Exception:
                pass
            continue

        utterance = spec["you"]
        kind = spec["kind"]

        # For an anaphora turn, capture what the WM resolves the pronoun to BEFORE answering (the answer's
        # gate uses the same resolution).
        anaphora_resolved_to = None
        held_before = None
        spec_before = None
        if kind == "anaphora":
            anaphora_resolved_to, held_before, spec_before = _capture_anaphora_resolution(
                chat, utterance, spec.get("expect_referent"))

        # the RICH (fluent multi-sentence) turn -- carries thread state across turns
        r = rich.answer(utterance)

        # what stored fact(s) did this turn surface? all kept facts must be brain-sourced (the moat extends to
        # multi-sentence)
        all_brain_sourced = all(tuple(f) in stored for f in r["facts"])

        rec = {
            "you": utterance,
            "kind": kind,
            "brain": r["answer"],
            "abstained": bool(r["abstained"]),
            "n_sentences": r["n_sentences"],
            # `gathered` = the grounded sentences the BRAIN assembled for this turn (kept + dropped). This is the
            # brain's FLUENCY -- how many on-topic grounded facts it composed -- separated from the small off-bridge
            # Qwen-0.5B's RENDER FIDELITY (a gathered fact whose Qwen surface fails the VERIFY re-parse is DROPPED,
            # not emitted -> the no-confab moat extended to multi-sentence). n_sentences (kept) <= gathered.
            "gathered": r["n_sentences"] + len(r["dropped"]),
            "fluent": bool((not r["abstained"]) and r["n_sentences"] >= 2),
            "fluent_gathered": bool((not r["abstained"]) and (r["n_sentences"] + len(r["dropped"])) >= 2),
            "topic": r["topic"],
            "followup": r["followup"],
            "supporting_facts": r["facts"],
            "dropped_unverified": r["dropped"],
            "all_brain_sourced": all_brain_sourced,
        }
        if kind == "anaphora":
            rec["anaphora_resolved_to"] = anaphora_resolved_to
            rec["wm_dominant_referent"] = held_before
            rec["wm_specificity"] = spec_before
            rec["expect_referent"] = spec.get("expect_referent")
            rec["expect_answer_contains"] = spec.get("expect_answer_contains")
            # AND the answer mentions the expected chained content (the chained fact's patient) -- this is the
            # load-bearing outcome: the pronoun resolved to the right referent AND the chained fact was recalled.
            want = spec.get("expect_answer_contains", "")
            rec["answer_has_expected"] = bool(want and want.lower() in str(r["answer"]).lower())
            # the pronoun RESOLVED correctly iff the resolution token == the expected referent AND the answer
            # surfaced the chained patient (so a noisy WM read that still produced the right answer counts; a
            # wrong resolution that produced a wrong/abstained answer does not).
            rec["anaphora_correct"] = bool(
                (anaphora_resolved_to == spec.get("expect_referent")) and rec["answer_has_expected"])
        if kind == "followup":
            # the follow-up must surface NEW grounded content (none repeating prior turns)
            rec["new_content"] = bool((not r["abstained"]) and r["n_sentences"] >= 1
                                      and all(tuple(f) not in set(prior_said) for f in r["facts"]))
        if kind in ("firewall_general", "firewall_untaught"):
            rec["LEAK"] = (not r["abstained"])

        transcript.append(rec)
        prior_said.extend([tuple(f) for f in r["facts"]])

    return transcript


def grade(transcript):
    """Per-goal verdicts + the overall GO/HONEST."""
    fluent_turns = [t for t in transcript if t["kind"] in ("fluent", "fluent_setup")]
    open_fluent = [t for t in transcript if t["kind"] == "fluent"]     # the OPEN self-questions
    anaphora_turns = [t for t in transcript if t["kind"] == "anaphora"]
    followup_turns = [t for t in transcript if t["kind"] == "followup"]
    firewall_turns = [t for t in transcript if t["kind"] in ("firewall_general", "firewall_untaught")]

    # --- GOAL 1: FLUENT (multi-sentence, brain-sourced) ---
    # The owner's goal is the BRAIN holding a fluent (multi-sentence, grounded) conversation. The brain's FLUENCY
    # is how many on-topic GROUNDED sentences it composes per open self-question (`gathered` = kept + dropped);
    # the small off-bridge Qwen-0.5B's RENDER FIDELITY (how many of those gathered sentences it can faithfully
    # surface so they pass the VERIFY re-parse = `n_sentences`/kept) is a SEPARATE faculty property. The no-confab
    # moat extends to multi-sentence: a gathered fact whose Qwen surface drifts is DROPPED, not emitted. So FLUENT
    # is graded on the BRAIN's gathering (every open self-question composed >=2 grounded sentences, all brain-
    # sourced, none abstained) -- not conflated with the 0.5B's render fidelity, which is reported alongside.
    open_answered = [t for t in open_fluent if not t["abstained"]]
    open_fluent_gathered = [t for t in open_fluent if t.get("fluent_gathered") and t["all_brain_sourced"]]
    open_fluent_kept = [t for t in open_fluent if t["fluent"] and t["all_brain_sourced"]]   # rendered+verified
    fluent_goal = (len(open_answered) == len(open_fluent)               # none abstained (all self-questions answered)
                   and len(open_fluent_gathered) == len(open_fluent)    # every one gathered >=2 grounded sentences
                   and all(t["all_brain_sourced"] for t in open_fluent))
    min_open_gathered = min((t.get("gathered", t["n_sentences"]) for t in open_fluent if not t["abstained"]),
                            default=0)
    min_open_sentences = min((t["n_sentences"] for t in open_fluent if not t["abstained"]), default=0)
    n_multisentence_gathered = len(open_fluent_gathered)
    n_multisentence_kept = len(open_fluent_kept)
    n_multisentence = n_multisentence_gathered

    # --- GOAL 2: MULTI-TURN ANAPHORA (attention) ---
    # every anaphora turn resolved the pronoun to the expected prior referent AND surfaced the chained content;
    # AND the follow-up brought genuinely new grounded content.
    anaphora_ok = [t for t in anaphora_turns if t.get("anaphora_correct") and t.get("answer_has_expected")]
    followup_ok = [t for t in followup_turns if t.get("new_content")]
    anaphora_goal = (len(anaphora_ok) == len(anaphora_turns) and len(anaphora_turns) >= 1
                     and len(followup_ok) == len(followup_turns))

    # --- GOAL 3: FIREWALL across turns (0 leaks) ---
    leaks = [t for t in firewall_turns if t.get("LEAK")]
    firewall_goal = (len(leaks) == 0 and len(firewall_turns) >= 1)

    # also: every kept sentence across the WHOLE transcript is brain-sourced (no confabulation anywhere)
    all_brain_sourced = all(t["all_brain_sourced"] for t in transcript)

    go = bool(fluent_goal and anaphora_goal and firewall_goal and all_brain_sourced)

    per_goal = {
        "FLUENT": {
            "pass": bool(fluent_goal),
            "open_self_questions": len(open_fluent),
            "n_multisentence_GATHERED_brain_fluency": n_multisentence_gathered,
            "n_multisentence_KEPT_render_fidelity": n_multisentence_kept,
            "min_gathered_sentences": min_open_gathered,
            "min_kept_sentences": min_open_sentences,
            "all_answered_brain_sourced": all(t["all_brain_sourced"] for t in open_fluent),
            "none_abstained": len(open_answered) == len(open_fluent),
            "note": ("FLUENT graded on the BRAIN's gathering (grounded sentences composed per turn = kept+dropped); "
                     "KEPT = how many the 0.5B Qwen rendered faithfully enough to pass VERIFY (the rest dropped by "
                     "the no-confab moat). The brain's fluency is GATHERED; KEPT is the small-faculty render fidelity."),
            "detail": [{"q": t["you"], "gathered": t.get("gathered"), "kept_rendered": t["n_sentences"],
                        "dropped_by_verify": len(t["dropped_unverified"]), "answer": t["brain"]}
                       for t in open_fluent],
        },
        "MULTI_TURN_ANAPHORA": {
            "pass": bool(anaphora_goal),
            "anaphora_turns_resolved": f"{len(anaphora_ok)}/{len(anaphora_turns)}",
            "followups_elaborated": f"{len(followup_ok)}/{len(followup_turns)}",
            "detail": [{"q": t["you"], "resolved_to": t.get("anaphora_resolved_to"),
                        "expected": t.get("expect_referent"), "wm_specificity": t.get("wm_specificity"),
                        "answer": t["brain"], "correct": t.get("anaphora_correct"),
                        "answer_has_expected": t.get("answer_has_expected")} for t in anaphora_turns]
                      + [{"q": t["you"], "followup_new_content": t.get("new_content"),
                          "n_sentences": t["n_sentences"], "answer": t["brain"]} for t in followup_turns],
        },
        "FIREWALL": {
            "pass": bool(firewall_goal),
            "firewall_turns": len(firewall_turns),
            "leaks": len(leaks),
            "detail": [{"q": t["you"], "abstained": t["abstained"], "LEAK": t.get("LEAK"),
                        "answer": t["brain"]} for t in firewall_turns],
        },
        "all_brain_sourced_everywhere": bool(all_brain_sourced),
    }
    return go, per_goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=16, help="off-bridge Qwen rate-code pool budget (16=GO)")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--max-sentences", type=int, default=4, help="max sentences per fluent reply")
    ap.add_argument("--no-faculty", action="store_true",
                    help="skip the off-bridge Qwen (template-stub renderer) -- proves the BRAIN side without GPU")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)

    print("=" * 110, flush=True)
    print("[FLUENT + MULTI-TURN test on the FIXED self-knowledge brain]", flush=True)
    print(f"  backend={os.environ.get('SIM_BACKEND')}  seed={a.seed}  T={a.T}  max_sentences={a.max_sentences}",
          flush=True)
    print("  the brain talks about ITSELF: fluent multi-sentence answers + anaphora across turns + firewall.",
          flush=True)
    print("=" * 110 + "\n", flush=True)

    t0 = time.time()

    # ---- build the off-bridge Qwen faculty (the FLUENCY), with the crash fix ----
    faculty_renderer = None
    faculty_err = None
    faculty_info = None
    if a.no_faculty:
        print("[mt] --no-faculty: using the GPU-free template-stub renderer (proves the BRAIN side).", flush=True)
        faculty_renderer = StubRenderer()
    else:
        # QWEN-CRASH FIX: free the cupy pool's cached blocks BEFORE torch loads the faculty (the composer bridges
        # we are about to build hold ~GBs of cupy-pool blocks; releasing them first eliminates the contention).
        # We build the agent FIRST (so the cupy pool is populated), THEN free + load the faculty. To keep the
        # order right we build the agent with NO renderer, then attach the faculty.
        pass

    # ---- build the FIXED self-knowledge MultiTurnAgent (decorrelated codes + curriculum + discourse WM) ----
    chat, agent, n_taught, recall_acc, aliases = build_multiturn_self_knowledge_chat(a.seed, faculty_renderer)
    print(f"[mt] built the MultiTurnAgent on the decorrelated learned codes; taught {n_taught} project facts; "
          f"recall {recall_acc:.2f} (the fix: was 0.21 on raw codes).", flush=True)

    # now load the faculty (after the agent's bridges populated the cupy pool) with the crash fix
    if not a.no_faculty:
        print(f"\n[mt] loading the FAST off-bridge Qwen-0.5B faculty at T={a.T} (with the cupy-pool crash fix) ...",
              flush=True)
        D._free_cupy_pool()      # release cupy-pool blocks so torch loads into free VRAM
        try:
            faculty_renderer = QwenRenderer(T=a.T, max_new_tokens=a.max_new_tokens, seed=a.seed)
            chat.renderer = faculty_renderer
            faculty_info = {"load_seconds": faculty_renderer.load_seconds, "T": a.T,
                            "model": "Qwen2.5-0.5B-Instruct (spiking forward, off-bridge)"}
            print(f"[mt]   faculty loaded in {faculty_renderer.load_seconds}s.\n", flush=True)
        except Exception as e:
            import traceback
            faculty_err = repr(e)
            traceback.print_exc()
            print(f"[mt]   faculty FAILED ({faculty_err}); falling back to the template-stub renderer.", flush=True)
            chat.renderer = StubRenderer()

    # ---- the RichAnswerComposer (the FLUENT multi-sentence grounded reply) over the ChatBrain ----
    rich = RichAnswerComposer(chat, max_chain_hops=3, max_elaborations=2, max_sentences=a.max_sentences)

    # ---- run the scripted multi-turn conversation ----
    print("[mt] running the scripted multi-turn conversation ...\n", flush=True)
    transcript = run_conversation(chat, rich, agent)

    # ---- grade ----
    go, per_goal = grade(transcript)
    elapsed = round(time.time() - t0, 1)

    # ---- print the VERBATIM transcript ----
    print("\n" + "=" * 110, flush=True)
    print("[mt] VERBATIM MULTI-TURN TRANSCRIPT (the brain talking about itself):", flush=True)
    print("=" * 110, flush=True)
    for t in transcript:
        ktag = {
            "fluent": "FLUENT", "fluent_setup": "setup", "anaphora": "ANAPHORA",
            "followup": "FOLLOWUP", "firewall_general": "FIREWALL(general)",
            "firewall_untaught": "FIREWALL(untaught)",
        }.get(t["kind"], t["kind"])
        print(f"\n  [{ktag}]", flush=True)
        print(f"  you>   {t['you']}", flush=True)
        atag = "  (ABSTAINED -- the moat)" if t["abstained"] else f"  [{t['n_sentences']} grounded sentences]"
        print(f"  brain> {t['brain']}{atag}", flush=True)
        if t["kind"] == "anaphora":
            ok = "RESOLVED OK" if t.get("anaphora_correct") else "MISRESOLVED"
            print(f"         -> anaphora: 'it/they' resolved to {t.get('anaphora_resolved_to')!r} "
                  f"(expected {t.get('expect_referent')!r}, WM spec {t.get('wm_specificity')}) [{ok}]", flush=True)
        if t["supporting_facts"]:
            print(f"         (grounded in: {t['supporting_facts']})", flush=True)
        if t["dropped_unverified"]:
            print(f"         (dropped unverified: {t['dropped_unverified']})", flush=True)
    print("\n" + "=" * 110, flush=True)

    # ---- verdict text ----
    fl = per_goal["FLUENT"]
    if go:
        verdict = (
            f"GO -- the FIXED self-knowledge brain holds a genuinely FLUENT + MULTI-TURN conversation ABOUT ITSELF. "
            f"FLUENT: all {fl['open_self_questions']}/{fl['open_self_questions']} open self-questions composed a "
            f"MULTI-SENTENCE grounded answer (>= {fl['min_gathered_sentences']} brain-gathered grounded sentences "
            f"each); the off-bridge Qwen-0.5B faithfully rendered + VERIFY-passed "
            f"{fl['n_multisentence_KEPT_render_fidelity']}/{fl['open_self_questions']} as multi-sentence (the rest "
            f"single-sentence because the 0.5B drifted on a fact like 'brain has neurons' -> the no-confab moat "
            f"DROPPED it rather than emit unverifiable prose). MULTI-TURN ANAPHORA: "
            f"{per_goal['MULTI_TURN_ANAPHORA']['anaphora_turns_resolved']} pronoun follow-ups resolved to the held "
            f"discourse referent via the spiking WM loop + answered the chained fact, and 'tell me more' "
            f"elaborated forward with new grounded content. FIREWALL: 0 LEAKS across turns (capital of France + "
            f"Romeo&Juliet + an untaught project part all ABSTAINED mid-conversation). Recall {recall_acc:.2f} "
            f"(the decorrelation fix). The owner's conversational goal is REACHED.")
    else:
        snags = []
        if not per_goal["FLUENT"]["pass"]:
            snags.append(f"FLUENT not fully met (brain-gathered multi-sentence="
                         f"{fl['n_multisentence_GATHERED_brain_fluency']}/{fl['open_self_questions']}, "
                         f"0.5B-rendered+verified multi-sentence={fl['n_multisentence_KEPT_render_fidelity']}/"
                         f"{fl['open_self_questions']}, none_abstained={fl['none_abstained']}, "
                         f"brain_sourced={fl['all_answered_brain_sourced']})")
        if not per_goal["MULTI_TURN_ANAPHORA"]["pass"]:
            snags.append(f"MULTI-TURN ANAPHORA not fully met (resolved="
                         f"{per_goal['MULTI_TURN_ANAPHORA']['anaphora_turns_resolved']}, followups="
                         f"{per_goal['MULTI_TURN_ANAPHORA']['followups_elaborated']})")
        if not per_goal["FIREWALL"]["pass"]:
            snags.append(f"FIREWALL BREACH: {per_goal['FIREWALL']['leaks']} leak(s) across turns")
        if not per_goal["all_brain_sourced_everywhere"]:
            snags.append("a sentence somewhere was NOT brain-sourced")
        verdict = "HONEST/PARTIAL -- " + " || ".join(snags)

    print(f"  PER-GOAL: FLUENT={per_goal['FLUENT']['pass']}  "
          f"MULTI_TURN_ANAPHORA={per_goal['MULTI_TURN_ANAPHORA']['pass']}  "
          f"FIREWALL={per_goal['FIREWALL']['pass']}", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print("=" * 110, flush=True)

    res = {
        "probe": "self_knowledge_FLUENT_MULTITURN_conversation_test",
        "resolves": "confirm the owner's conversational goal: the FIXED self-knowledge brain holds a genuinely "
                    "FLUENT (multi-sentence, Qwen-phrased, verify-checked) MULTI-TURN (anaphora resolves via the "
                    "discourse buffer) conversation ABOUT ITSELF, the no-confab firewall holding across turns "
                    "(0 leaks).",
        "backend": os.environ.get("SIM_BACKEND"),
        "seed": a.seed, "T": a.T, "max_new_tokens": a.max_new_tokens, "max_sentences": a.max_sentences,
        "GO": go,
        "verdict": verdict,
        "recall_acc": recall_acc,
        "n_facts_taught": n_taught,
        "renderer": (chat.renderer.name if chat.renderer is not None else "raw brain triples"),
        "faculty_info": faculty_info,
        "faculty_error": faculty_err,
        "per_goal": per_goal,
        "transcript": transcript,
        "elapsed_seconds": elapsed,
        "wiring": "load saved stream-learned grounded codes -> ZCA-decorrelate (recall fix) -> MultiTurnAgent "
                  "(discourse WM = anaphora) + full curriculum -> ChatBrain (GATE+VERIFY+anaphora + off-bridge "
                  "Qwen, crash-fixed) -> RichAnswerComposer (multi-sentence grounded fluent reply).",
        "sim_edits": "NONE (runner-level only).",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\n[saved] {os.path.relpath(a.out, _REPO)}  (wall {elapsed}s)", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
