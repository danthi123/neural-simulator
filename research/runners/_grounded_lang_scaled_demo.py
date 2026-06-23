"""SCALE the grounded-language capstone -- turn the end-to-end DE-RISK into a ROBUST capability at SCALE.

The de-risk (`_grounded_lang_integration_derisk.py`) was GO end-to-end at the P2 30-fact scale: the REAL
spiking Qwen2.5-0.5B faculty renders the brain's GATED+VERIFIED facts (grounded 4/4, moat 2/2, drift 1/1).
This runner SCALES it on the richer ~67-fact curriculum (`_grounded_lang_curriculum_v2_scaled.json`) with:
  - the FULL query set (18 recall, 8 moat, 2 chains) on the BRAIN half, multi-seed (>=3) -- cheap numpy-CPU;
  - the SUBJECT-FIRST constrain prompt by DEFAULT (the de-risk's loose "Turn the triple (a,v,p)..." prompt
    object-fronted 'Rabbit chased fox' for (fox,chase,rabbit); subject-first pins the agent first so the
    first render re-parses correctly, no regen needed);
  - reject->REGENERATE recovery (a VERIFY reject re-prompts the tighter exact-3-word template once);
  - a larger grounded-RENDER subset (~8-12 renders) + untaught (abstain) + drift adversarial on the spiking
    faculty, FOREGROUND (the slow part), 1-2 seeds (the brain half carries the multi-seed weight).

THE THREE LAYERS, per query (unchanged from the de-risk -- the moat the whole arc built is preserved EVEN
WITH a real generative LLM in the loop):
  (i)  GATE      -- the brain's composer exact-match recall returns the stored SVO OR ABSTAINS (what_does/
                    who_does -> None; is_it_true -> 'unknown'). On abstain the faculty is given NOTHING -> no
                    generation (the MOAT). The GATE is the numpy-CPU brain (parser+composer).
  (ii) CONSTRAIN -- the REAL spiking Qwen faculty renders the gated SVO into one short sentence, SUBJECT-FIRST
                    ("Write one short sentence. The subject is 'a'. The verb is 'v'. The object is 'p'. ..."),
                    keeping the verb verbatim. The faculty's freedom is determiners + inflection; the prompt +
                    grounded content pin the meaning + the role order.
  (iii) VERIFY   -- the faculty's GENERATED PROSE is re-parsed back into an SVO by the BRAIN (content extraction
                    handling determiners + verb inflection + the 'live in' phrasal; the SAME BridgeParser
                    re-assigns roles); the re-parsed {agent, action, patient} must MATCH the gated fact, else the
                    output is REJECTED (drift caught) -> regenerate tighter once -> verified-or-hard-abstain.

MEASURE (the scale question): does SUBJECT-FIRST LIFT the first-render correctness vs the de-risk's loose prompt
(fewer regens)? Does the moat stay 0-false-accept at the bigger vocab? Is the grounded-render correct-rate high?
Is every drift caught?

VERDICT: GO = at ~67 facts, recall high (multi-seed) + moat 0-false-accept (multi-seed) + grounded-render high
(subject-first) + drift caught -> the grounded-language capability is ROBUST at scale. Or HONEST: where it
degrades at scale (the composer's recall capacity at bigger vocab? the moat's false-accept rate? the render
correctness?) + the precise limit.

FOREGROUND/blocking by design. The brain (parser/composer) is numpy-CPU; the spiking faculty is PyTorch on the
GPU (RTX 3090). NO `sim/` edit (the faculty machinery is reused-by-import from the P1b runner; the brain half +
the VERIFY content-extractor from the integration de-risk).

Usage:
  python -m research.runners._grounded_lang_scaled_demo                  # brain seeds 42,43,44; render T=16 seed 42
  python -m research.runners._grounded_lang_scaled_demo --T 8            # faster faculty render (ppl 1.21x)
  python -m research.runners._grounded_lang_scaled_demo --render-seeds 42 43   # render on 2 seeds
  python -m research.runners._grounded_lang_scaled_demo --n-render 10    # grounded-render subset size
  python -m research.runners._grounded_lang_scaled_demo --no-faculty     # brain-half only (skip the GPU render)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

# the brain half is a numpy-CPU pipeline; pin it to numpy so the (parser/composer) build is portable + does not
# contend with the PyTorch CUDA faculty for the GPU. (The faculty forward is its own torch device.)
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# --- the BRAIN half (reused VERBATIM from P2; the parser+composer pipeline is GO) ---
from research.runners.core_sim_composition import Clause
from research.runners.brain_conversational_agent import BrainConversationalAgent
from research.runners._grounded_lang_p2_derisk import (
    _collect_vocab, _teach, _answer, _recall_ok, _moat_breach,
)
# --- the VERIFY content-extractor + the spiking faculty (reused VERBATIM from the integration de-risk / P1b) ---
from research.runners._grounded_lang_integration_derisk import (
    _build_inflection_map, _extract_svo_from_prose, SpikingQwenFaculty,
)

CURRICULUM = _REPO / "research" / "findings" / "raw" / "_grounded_lang_curriculum_v2_scaled.json"
OUT = _REPO / "research" / "findings" / "raw" / "_grounded_lang_scaled_demo.json"


# =================================================================================================
# VERIFY scale-fix: noun PLURALIZATION normalization (the de-risk extractor handles VERB inflection only).
# At the bigger curriculum the 0.5B faculty often pluralizes the subject/object of a generic fact -- "Wolves eat
# deer", "Frog eats flies" (BOTH semantically the TRUE fact) -- but the de-risk's `_extract_svo_from_prose`
# matches only the singular curriculum tokens, so a plural noun fails to re-parse and the (true) render is
# (conservative-correctly) rejected. That is a VERIFY-coverage gap, NOT a moat breach (the gate already abstained
# on untaught cues BEFORE any rendering) and NOT a faculty confabulation. A real downstream comprehension stage
# normalizes plurals; we add that here (NO sim/ edit, NO edit to the shared de-risk extractor) so the grounded
# render is robust to the faculty's free pluralization while the moat is untouched. We pre-normalize plural NOUNS
# in the prose back to their singular curriculum form, then delegate to the SAME de-risk extractor.
# =================================================================================================
def _build_plural_map(nouns):
    """surface plural -> singular curriculum noun. Covers regular -s/-es, y->ies (fly->flies), f/fe->ves
    (wolf->wolves, leaf->leaves, knife->knives). Only adds a key if it does not collide with an existing singular
    noun (so 'fish'/'deer'/'sheep' invariant plurals keep their singular meaning)."""
    sing = set(nouns)
    m = {}
    for n in sing:
        cands = []
        if n.endswith(("s", "sh", "ch", "x", "z")):
            cands.append(n + "es")
        else:
            cands.append(n + "s")
        if n.endswith("y") and len(n) > 1 and n[-2] not in "aeiou":
            cands.append(n[:-1] + "ies")                       # fly -> flies (consonant + y)
        if n.endswith("fe"):
            cands.append(n[:-2] + "ves")                       # knife -> knives
        elif n.endswith("f"):
            cands.append(n[:-1] + "ves")                       # wolf -> wolves, leaf -> leaves
        for c in cands:
            if c not in sing and c not in m:                   # never shadow a real singular token
                m[c] = n
    return m


def _extract_svo_plural_aware(prose, agents, actions, patients, inflect, plural_map):
    """Re-parse the faculty's prose into the canonical SVO, FIRST normalizing any plural agent/object noun back to
    its singular curriculum form (so 'Wolves eat deer' / 'Frog eats flies' re-parse to the true fact), then
    delegating to the de-risk's VERBATIM `_extract_svo_from_prose` (verb-inflection + role order unchanged)."""
    def repl(mo):
        w = mo.group(0)
        return plural_map.get(w.lower(), w)
    normalized = re.sub(r"[A-Za-z]+", repl, prose)
    return _extract_svo_from_prose(normalized, agents, actions, patients, inflect)


import re  # noqa: E402  (used by _extract_svo_plural_aware; kept local to this scale-fix block)


# =================================================================================================
# A SUBJECT-FIRST constrain prompt: the de-risk's loose "Turn the triple (a,v,p) into one sentence" let the 0.5B
# object-front ('Rabbit chased fox' for fox/chase/rabbit -> VERIFY rejected -> regen recovered). Pinning the
# subject/verb/object roles EXPLICITLY makes the first render assign the agent first, so it re-parses correctly
# without a regen. The faculty's freedom stays grammar/determiners/inflection; the prompt fixes role ORDER.
# =================================================================================================
SUBJECT_FIRST_TEMPLATE = (
    "Write one short, grammatical sentence. "
    "The subject is '{a}'. The action verb is '{v}'. The object is '{p}'. "
    "Put the subject '{a}' first, then the verb, then the object. Keep the verb '{v}'. "
    "Reply with only the sentence."
)
# the loose de-risk prompt, kept for an A/B (does subject-first LIFT the first-render correctness?).
LOOSE_TEMPLATE = ("Turn the triple ({a}, {v}, {p}) into one short grammatical sentence. "
                  "Keep the same verb '{v}'. Reply with only the sentence.")
# a TIGHTER regenerate prompt used after a VERIFY reject (forces the exact 3 words, subject order).
REGEN_TEMPLATE = ("Write exactly one short sentence that means '{a} {v} {p}'. "
                  "Put '{a}' first. Use the words {a}, {v}, and {p}. "
                  "Reply with only the sentence, nothing else.")


def _faculty_render(faculty, template, a, v, p, seed=None):
    """Render the gated SVO with `faculty` using an arbitrary prompt template. Returns (first_line, full, secs)."""
    return faculty._generate(template.format(a=a, v=v, p=p), seed=seed)


# =================================================================================================
# (1) THE BRAIN HALF AT SCALE -- the FULL query set, multi-seed (cheap numpy-CPU). This is the load-bearing
# scaling check: structured RECALL on every queries_recall + the no-confab MOAT on every queries_moat (the
# 0-false-accept anti-cheat at the BIGGER vocab) + the chains.
# =================================================================================================
def run_brain_half(cur, vocab, seed):
    agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf")
    taught = _teach(agent, cur)

    recall = []
    n_recall_ok = 0
    for q in cur.get("queries_recall", []):
        got = _answer(agent, q)
        ok = _recall_ok(q, got)
        n_recall_ok += int(ok)
        recall.append({"cue": q["cue"], "type": q["type"], "expect": q["expect"], "got": got, "ok": ok})

    moat = []
    n_breach = 0
    for q in cur.get("queries_moat", []):
        got = _answer(agent, q)
        breach = _moat_breach(q, got)
        n_breach += int(breach)
        moat.append({"cue": q["cue"], "type": q["type"], "got": got, "abstained": not breach,
                     "note": q.get("note", "")})

    # chains (bonus). The curriculum's two chains: a same-relation 2-hop (chase,chase -> mouse) AND a
    # MIXED-relation chain whose `expect_2hop` is actually the 1-hop target (a curriculum spec quirk). We test
    # BOTH the as-specified single-relation chase (what _answer/reason_chain in P2 does) AND, for the mixed
    # chain, the correct mixed-relation actions, so the trail shows the brain does the right thing either way.
    chain_results = []
    for ch in cur.get("chains", []):
        cue = ch["query"][0]
        rel = ch["query"][1]
        facts = ch.get("facts", [])
        same_rel_actions = [rel, rel]
        got_same = agent.reason_chain(cue, same_rel_actions)
        # the true relation sequence implied by the chain's facts (e.g. [chase, eat] for wolf->deer->leaf)
        mixed_actions = [f[1] for f in facts] if facts else same_rel_actions
        got_mixed = agent.reason_chain(cue, mixed_actions)
        true_terminal = facts[-1][2] if facts else None
        chain_results.append({
            "desc": ch.get("desc", ""), "cue": cue, "rel": rel,
            "same_rel_actions": same_rel_actions, "got_same_rel": got_same,
            "expect_2hop_as_specified": ch.get("expect_2hop"),
            "ok_as_specified": (got_same == ch.get("expect_2hop")),
            "mixed_actions": mixed_actions, "got_mixed": got_mixed,
            "true_terminal": true_terminal, "ok_mixed_true": (got_mixed == true_terminal),
        })

    n_recall = len(recall)
    return {
        "seed": seed,
        "taught": taught,
        "recall_correct": n_recall_ok,
        "recall_total": n_recall,
        "recall_rate": (n_recall_ok / n_recall) if n_recall else None,
        "moat_false_accepts": n_breach,
        "moat_total": len(moat),
        "recall_detail": recall,
        "moat_detail": moat,
        "chain_detail": chain_results,
    }, agent


# =================================================================================================
# (2) THE END-TO-END GROUNDED RENDER -- GATE -> CONSTRAIN(spiking, subject-first) -> VERIFY(reject->regen).
# Reuses a fresh agent (so the render is a clean grounded turn) at the render seed.
# =================================================================================================
def grounded_reply(agent, faculty, q, vocab_sets, mode="constrain", prompt="subject_first", allow_regen=True):
    """Run one query through the three-layer grounding loop with the REAL spiking Qwen faculty.

    `mode`: 'constrain' (the normal grounded render) or 'adversarial' (steer the faculty to a wrong patient).
    `prompt`: 'subject_first' (the default, fixes role order) or 'loose' (the de-risk prompt -- the A/B baseline).
    `allow_regen`: on a VERIFY reject in constrain mode, RE-PROMPT tighter ONCE and re-verify (the production
    recovery path). Disabled for adversarial (the drift MUST stay caught -- we never recover a steered-wrong fact).
    Returns a structured record (gate_svo, surface prose, reparsed SVO, verified, emitted, abstained, regen_used)."""
    agents, actions, patients, inflect, plural_map = vocab_sets
    qtype = q["type"]
    cue = q["cue"]
    truth = None

    # (i) GATE -- exact-match recall over the spiking store; abstains (None / 'unknown') when no fact matches
    if qtype == "patient":
        content = agent.what_does(cue[0], cue[1])
        gate_svo = [cue[0], cue[1], content] if content is not None else None
    elif qtype == "agent":
        content = agent.who_does(cue[0], cue[1])
        gate_svo = [content, cue[0], cue[1]] if content is not None else None
    elif qtype == "yesno":
        truth = agent.is_it_true(cue[0], cue[1], cue[2])
        gate_svo = [cue[0], cue[1], cue[2]] if truth != "unknown" else None
    else:
        raise ValueError(f"unknown query type {qtype!r}")

    rec = {"cue": cue, "type": qtype, "gate_svo": gate_svo, "gate_truth": truth, "prompt": prompt}

    # gate abstained -> the faculty is given NOTHING -> no generation (the MOAT)
    if gate_svo is None:
        rec.update({"surface": None, "surface_full": None, "reparse_svo": None,
                    "verified": None, "emitted": False, "abstained": True, "gen_seconds": 0.0,
                    "regen_used": False})
        return rec

    # (ii) CONSTRAIN -- the REAL spiking faculty renders the gated content into fluent prose
    a, v, p = gate_svo
    tmpl = SUBJECT_FIRST_TEMPLATE if prompt == "subject_first" else LOOSE_TEMPLATE
    if mode == "adversarial":
        wrong_p = q["wrong_patient"]
        surface, surface_full, gen_s = _faculty_render(faculty, tmpl, a, v, wrong_p)
    else:
        surface, surface_full, gen_s = _faculty_render(faculty, tmpl, a, v, p)

    # (iii) VERIFY -- re-parse the faculty's GENERATED PROSE back into an SVO; must match the GATED fact.
    def _verify(prose):
        csvo = _extract_svo_plural_aware(prose, agents, actions, patients, inflect, plural_map)
        if csvo is None:
            return None, False, "prose did not re-parse to a clean SVO"
        parsed_ = agent.parse(csvo, voice="active")            # the brain's comprehension of the recovered SVO
        rsvo = [parsed_.get("agent"), parsed_.get("action"), parsed_.get("patient")]
        return rsvo, (rsvo == gate_svo), (None if rsvo == gate_svo else "re-parsed SVO mismatches the gated fact")

    reparse_svo, verified, reason = _verify(surface)
    first_render_verified = bool(verified)                     # the A/B metric: did the FIRST render re-parse OK?
    regen_used = False
    if (not verified) and allow_regen and mode != "adversarial":
        # the constrain prompt produced an unverifiable render (drift/role-inversion). RE-PROMPT tighter ONCE.
        regen_used = True
        surface2, surface_full2, gen_s2 = _faculty_render(faculty, REGEN_TEMPLATE, a, v, p)
        reparse2, verified2, reason2 = _verify(surface2)
        rec["constrain_surface"] = surface                     # keep the first (rejected) render for the trail
        rec["constrain_reparse_svo"] = reparse_svo
        surface, surface_full, reparse_svo, verified, reason = surface2, surface_full2, reparse2, verified2, reason2
        gen_s += gen_s2

    rec.update({"surface": surface, "surface_full": surface_full, "reparse_svo": reparse_svo,
                "verified": bool(verified), "emitted": bool(verified), "abstained": False,
                "first_render_verified": first_render_verified, "regen_used": regen_used,
                "reject_reason": reason, "gen_seconds": round(gen_s, 2)})
    return rec


def run_render(cur, vocab, render_seed, faculty, n_render, prompt="subject_first", ab_loose=True):
    """The END-TO-END grounded render at one seed: a grounded subset + untaught (abstain) + drift adversarial.

    Returns a dict with grounded/untaught/drift detail + the A/B (subject-first vs loose first-render correctness)."""
    agent = BrainConversationalAgent(seed=render_seed, concepts={w: None for w in vocab}, composer_kind="rf")
    _teach(agent, cur)

    agents_set = {f[0] for f in cur.get("facts", [])}
    patients_set = {f[2] for f in cur.get("facts", [])}
    actions_set = {f[1] for f in cur.get("facts", [])}
    inflect = _build_inflection_map(sorted(actions_set))
    # the VERIFY plural-noun normalizer (the scale fix): surface plural agent/object -> singular curriculum noun
    plural_map = _build_plural_map(agents_set | patients_set)
    vocab_sets = (agents_set, actions_set, patients_set, inflect, plural_map)

    # --- (a) GROUNDED: a subset of patient/agent recall queries -> a fluent sentence whose re-parse matches ---
    grounded_queries = [q for q in cur.get("queries_recall", []) if q["type"] in ("patient", "agent")][:n_render]
    grounded = []
    for q in grounded_queries:
        rec = grounded_reply(agent, faculty, q, vocab_sets, mode="constrain", prompt=prompt)
        rec["ok"] = bool(rec["emitted"] and rec["verified"])
        grounded.append(rec)

    # --- (a') A/B: the SAME grounded queries with the LOOSE de-risk prompt (does subject-first LIFT first-render?) ---
    grounded_loose = []
    if ab_loose:
        for q in grounded_queries:
            # NO regen here -- we want the FIRST-render correctness of the loose prompt, head-to-head.
            rec = grounded_reply(agent, faculty, q, vocab_sets, mode="constrain", prompt="loose", allow_regen=False)
            rec["ok"] = bool(rec["emitted"] and rec["verified"])
            grounded_loose.append(rec)

    # --- (b) UNTAUGHT: all moat patient/agent cues -> the GATE abstains -> NO sentence (the MOAT) ---
    untaught_queries = [q for q in cur.get("queries_moat", []) if q["type"] in ("patient", "agent")]
    untaught = []
    for q in untaught_queries:
        rec = grounded_reply(agent, faculty, q, vocab_sets, mode="constrain", prompt=prompt)
        rec["held"] = (rec["abstained"] is True) and (rec["emitted"] is False)
        rec["note"] = q.get("note", "")
        untaught.append(rec)

    # --- (c) DRIFT/CONFAB: ~2 adversarial steered-to-wrong-fact -> VERIFY re-parse REJECTS it ---
    # take grounded facts, gate the TRUE fact, but steer the faculty to a DIFFERENT (wrong) patient.
    drift = []
    all_patients = sorted(patients_set)
    n_drift = min(2, len(grounded_queries))
    for base in grounded_queries[:n_drift]:
        if base["type"] != "patient":
            continue
        true_p = agent.what_does(base["cue"][0], base["cue"][1])
        wrong_p = next((x for x in all_patients if x != true_p), (true_p or "thing") + "_X")
        adv_q = {"type": "patient", "cue": base["cue"], "wrong_patient": wrong_p}
        rec = grounded_reply(agent, faculty, adv_q, vocab_sets, mode="adversarial")
        rec["true_patient"] = true_p
        rec["confab_patient"] = wrong_p
        rec["caught"] = (rec["gate_svo"] is not None) and (rec["emitted"] is False)
        drift.append(rec)

    # --- (c') REGENERATE-ON-REJECT demonstration: after a drift reject, re-prompt the TRUE fact tighter -> verified.
    regen = None
    if drift and drift[0]["caught"]:
        base = grounded_queries[0]
        tp = agent.what_does(base["cue"][0], base["cue"][1])
        if tp is not None:
            a, v, p = base["cue"][0], base["cue"][1], tp
            surface, surface_full, gen_s = _faculty_render(faculty, REGEN_TEMPLATE, a, v, p)
            content_svo = _extract_svo_plural_aware(surface, agents_set, actions_set, patients_set, inflect, plural_map)
            reparse = agent.parse(content_svo, voice="active") if content_svo else None
            reparse_svo = [reparse.get("agent"), reparse.get("action"), reparse.get("patient")] if reparse else None
            verified = (reparse_svo == [a, v, p])
            regen = {"gate_svo": [a, v, p], "surface": surface, "surface_full": surface_full,
                     "reparse_svo": reparse_svo, "verified": bool(verified), "emitted": bool(verified),
                     "gen_seconds": gen_s}

    n_grounded_ok = sum(r["ok"] for r in grounded)
    n_grounded_first_ok = sum(r.get("first_render_verified", False) for r in grounded)
    n_grounded_regen = sum(r.get("regen_used", False) for r in grounded)
    n_loose_first_ok = sum(r.get("first_render_verified", False) for r in grounded_loose) if ab_loose else None
    n_untaught_held = sum(r["held"] for r in untaught)
    n_drift_caught = sum(r["caught"] for r in drift)
    return {
        "render_seed": render_seed,
        "prompt": prompt,
        "grounded_correct": n_grounded_ok,
        "grounded_total": len(grounded),
        "grounded_first_render_ok": n_grounded_first_ok,         # verified on the FIRST render (no regen)
        "grounded_regen_used": n_grounded_regen,
        "loose_first_render_ok": n_loose_first_ok,               # A/B baseline (loose prompt, first render, no regen)
        "loose_total": (len(grounded_loose) if ab_loose else None),
        "untaught_held": n_untaught_held,
        "untaught_total": len(untaught),
        "drift_caught": n_drift_caught,
        "drift_total": len(drift),
        "grounded_detail": grounded,
        "grounded_loose_detail": grounded_loose,
        "untaught_detail": untaught,
        "drift_detail": drift,
        "regen_after_reject": regen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain-seeds", type=int, nargs="+", default=[42, 43, 44],
                    help="seeds for the cheap brain-half scale check (full query set, >=3)")
    ap.add_argument("--render-seeds", type=int, nargs="+", default=[42],
                    help="seeds for the slow spiking-render half (1-2 is fine; the brain half carries multi-seed)")
    ap.add_argument("--T", type=int, default=16, help="rate-code pool budget for the spiking faculty (16=GO 1.08x; 8=1.21x)")
    ap.add_argument("--max-new-tokens", type=int, default=24, help="faculty surface-form length cap (keep small)")
    ap.add_argument("--n-render", type=int, default=10, help="grounded-render subset size per render seed (~8-12)")
    ap.add_argument("--no-ab-loose", action="store_true", help="skip the loose-prompt A/B baseline (saves ~n-render gens/seed)")
    ap.add_argument("--no-faculty", action="store_true", help="brain-half only -- skip the GPU spiking render entirely")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t_start = time.time()
    with open(os.path.abspath(CURRICULUM), "r", encoding="utf-8") as fh:
        cur = json.load(fh)
    vocab = _collect_vocab(cur)
    # the adversarial path may fall back to '<token>_X'; add encodable fall-backs to the vocab (parity with the de-risk)
    vocab = sorted(set(vocab) | {p + "_X" for p in {f[2] for f in cur.get("facts", [])}})

    n_facts = (len(cur.get("facts", [])) + len(cur.get("attribute_facts", []))
               + len(cur.get("clause_facts", [])))
    print(f"[scaled] curriculum: ~{n_facts} facts ({len(cur.get('facts', []))} SVO + "
          f"{len(cur.get('attribute_facts', []))} attr + {len(cur.get('clause_facts', []))} clause); "
          f"vocab={len(vocab)}; recall_q={len(cur.get('queries_recall', []))} moat_q={len(cur.get('queries_moat', []))}",
          flush=True)

    # ---- (1) BRAIN HALF AT SCALE (cheap numpy-CPU, full query set, multi-seed) ----
    print(f"[scaled] (1) BRAIN HALF at scale -- backend={os.environ.get('SIM_BACKEND')}, "
          f"seeds={args.brain_seeds} (full {len(cur.get('queries_recall', []))}-recall + "
          f"{len(cur.get('queries_moat', []))}-moat) ...", flush=True)
    brain_per_seed = []
    for seed in args.brain_seeds:
        ts = time.time()
        try:
            r, _agent = run_brain_half(cur, vocab, seed)
        except Exception as e:
            r = {"seed": seed, "error": repr(e), "traceback": traceback.format_exc()}
            traceback.print_exc()
        brain_per_seed.append(r)
        if "error" not in r:
            cd = r["chain_detail"]
            print(f"[scaled]   seed {seed}: recall {r['recall_correct']}/{r['recall_total']} "
                  f"(={r['recall_rate']:.3f})  moat-false-accepts {r['moat_false_accepts']}/{r['moat_total']}  "
                  f"chains(as-spec {sum(c['ok_as_specified'] for c in cd)}/{len(cd)}, "
                  f"mixed-true {sum(c['ok_mixed_true'] for c in cd)}/{len(cd)})  [{time.time()-ts:.1f}s]", flush=True)

    ok_brain = [r for r in brain_per_seed if "error" not in r]
    brain_recall_perfect = bool(ok_brain) and all(r["recall_rate"] == 1.0 for r in ok_brain)
    brain_moat_clean = bool(ok_brain) and all(r["moat_false_accepts"] == 0 for r in ok_brain)
    brain_min_recall = min((r["recall_rate"] for r in ok_brain), default=None)
    brain_max_breach = max((r["moat_false_accepts"] for r in ok_brain), default=None)
    n_brain_ge3 = len(ok_brain) >= 3

    # ---- (2) END-TO-END GROUNDED RENDER (slow spiking faculty, FOREGROUND, render seeds) ----
    render_per_seed = []
    faculty_err = None
    faculty_info = None
    if not args.no_faculty:
        print(f"[scaled] (2) loading the REAL spiking Qwen faculty at T={args.T} (render seeds={args.render_seeds}) ...",
              flush=True)
        try:
            import torch
            if not torch.cuda.is_available():
                print("[scaled] WARNING: CUDA not available -- the spiking faculty is a GPU runner; will be slow.", flush=True)
            faculty = SpikingQwenFaculty(T=args.T, max_new_tokens=args.max_new_tokens, seed=args.render_seeds[0],
                                         device=("cuda" if torch.cuda.is_available() else "cpu"))
            faculty_info = {"load_seconds": faculty.load_seconds, "pools": faculty.pools,
                            "measured_ranges": faculty.measured_ranges}
            print(f"[scaled]   faculty loaded in {faculty.load_seconds}s; pools={faculty.pools}", flush=True)
            for rseed in args.render_seeds:
                ts = time.time()
                rr = run_render(cur, vocab, rseed, faculty, args.n_render,
                                prompt="subject_first", ab_loose=(not args.no_ab_loose))
                render_per_seed.append(rr)
                lf = rr["loose_first_render_ok"]
                print(f"[scaled]   render seed {rseed}: grounded {rr['grounded_correct']}/{rr['grounded_total']} "
                      f"(first-render {rr['grounded_first_render_ok']}/{rr['grounded_total']}, regen "
                      f"{rr['grounded_regen_used']}; loose-first {lf if lf is not None else 'n/a'}/"
                      f"{rr['grounded_total']})  moat-held {rr['untaught_held']}/{rr['untaught_total']}  "
                      f"drift-caught {rr['drift_caught']}/{rr['drift_total']}  [{time.time()-ts:.1f}s]", flush=True)
                # echo a couple of grounded replies inline
                for g in rr["grounded_detail"][:3]:
                    print(f"[scaled]       grounded {g['cue']} -> {g.get('surface')!r} "
                          f"(verified={g['verified']}, regen={g.get('regen_used')})", flush=True)
        except Exception as e:
            faculty_err = repr(e)
            traceback.print_exc()

    # ---- VERDICT ----
    ok_render = [r for r in render_per_seed]
    render_grounded_high = bool(ok_render) and all(
        r["grounded_correct"] == r["grounded_total"] and r["grounded_total"] > 0 for r in ok_render)
    render_moat_clean = bool(ok_render) and all(
        r["untaught_held"] == r["untaught_total"] and r["untaught_total"] > 0 for r in ok_render)
    render_drift_caught = bool(ok_render) and all(
        r["drift_caught"] == r["drift_total"] and r["drift_total"] > 0 for r in ok_render)

    # subject-first lift (A/B): mean first-render correctness, subject-first vs loose, across render seeds.
    sf_first = [r["grounded_first_render_ok"] / r["grounded_total"] for r in ok_render if r["grounded_total"]]
    loose_first = [r["loose_first_render_ok"] / r["grounded_total"] for r in ok_render
                   if r["grounded_total"] and r["loose_first_render_ok"] is not None]
    sf_first_mean = (sum(sf_first) / len(sf_first)) if sf_first else None
    loose_first_mean = (sum(loose_first) / len(loose_first)) if loose_first else None

    brain_ok = brain_recall_perfect and brain_moat_clean and n_brain_ge3
    if args.no_faculty or faculty_err is not None:
        # brain-half-only verdict (or faculty errored)
        go = bool(brain_ok)
        head = "GO (brain-half)" if go else "PARTIAL/NO-GO (brain-half)"
        extra = (f" -- the spiking render was {'SKIPPED (--no-faculty)' if args.no_faculty else f'ERRORED: {faculty_err}'}; "
                 "the END-TO-END grounded-render half is NOT validated by this run.")
        verdict = (
            f"{head}: at ~{n_facts} facts, recall {'1.0' if brain_recall_perfect else f'min {brain_min_recall}'} "
            f"({'all seeds' if brain_recall_perfect else 'PARTIAL'}) AND moat "
            f"{'0-false-accept' if brain_moat_clean else f'max {brain_max_breach} breach'} across "
            f"{len(ok_brain)} seeds.{extra}")
    else:
        go = bool(brain_ok and render_grounded_high and render_moat_clean and render_drift_caught)
        if go:
            verdict = (
                f"GO -- the grounded-language capability is ROBUST AT SCALE (~{n_facts} facts): "
                f"BRAIN-half recall {'1.0' if brain_recall_perfect else f'min {brain_min_recall}'} + moat "
                f"0-false-accept across {len(ok_brain)} seeds (>=3); the REAL spiking Qwen faculty (T={args.T}) "
                f"renders grounded facts FLUENTLY (grounded {sum(r['grounded_correct'] for r in ok_render)}/"
                f"{sum(r['grounded_total'] for r in ok_render)} across {len(ok_render)} render seed(s), each "
                f"re-parses to the taught fact) with SUBJECT-FIRST first-render {sf_first_mean:.2f} (vs loose "
                f"{loose_first_mean:.2f})" + (" -- a LIFT" if (sf_first_mean is not None and loose_first_mean is not None
                                                              and sf_first_mean > loose_first_mean) else "") +
                f"; untaught cues ABSTAIN (moat held all render seeds) AND adversarial DRIFT is caught-by-VERIFY "
                f"(all). The no-confab moat HOLDS at the bigger vocab, EVEN WITH a real generative LLM in the loop.")
        else:
            leaks = []
            if not brain_ok:
                leaks.append(f"BRAIN: recall min {brain_min_recall} (perfect={brain_recall_perfect}), moat max-breach "
                             f"{brain_max_breach} (clean={brain_moat_clean}), n_seeds {len(ok_brain)}")
            if not render_grounded_high:
                misses = [(rr["render_seed"], [g["cue"] for g in rr["grounded_detail"] if not g["ok"]])
                          for rr in ok_render]
                leaks.append(f"GROUNDED-RENDER not all correct: {misses}")
            if not render_moat_clean:
                leaks.append("MOAT leak: an untaught cue produced a sentence in the render half")
            if not render_drift_caught:
                leaks.append("VERIFY leak: an adversarial drift was NOT caught in the render half")
            verdict = "HONEST/PARTIAL -- " + " || ".join(leaks)

    summary = {
        "probe": "grounded_lang_scaled_demo_end_to_end_real_spiking_qwen_faculty",
        "resolves": "SCALE the grounded-language capstone -- the END-TO-END demo (GATE -> real spiking-Qwen "
                    "SUBJECT-FIRST render -> VERIFY/reject->regen) on a richer ~67-fact curriculum, the FULL query "
                    "set multi-seed for the brain half + a larger grounded-render subset; does the no-confab moat "
                    "stay 0-false-accept at the bigger vocab + does subject-first LIFT the render correctness?",
        "curriculum": os.path.relpath(os.path.abspath(CURRICULUM), str(_REPO)),
        "n_facts_total": n_facts,
        "vocab_size": len(vocab),
        "brain_backend": os.environ.get("SIM_BACKEND"),
        "brain_seeds": args.brain_seeds,
        "render_seeds": args.render_seeds,
        "T": args.T,
        "max_new_tokens": args.max_new_tokens,
        "n_render": args.n_render,
        "prompt_default": "subject_first",
        "verify_scale_fix": "noun-PLURALIZATION normalization added to the VERIFY re-parse (the de-risk extractor "
                            "handled verb inflection only). At ~67 facts the 0.5B faculty often pluralizes the "
                            "subject/object of a generic fact ('Wolves eat deer', 'Frog eats flies' -- BOTH the TRUE "
                            "fact); the de-risk extractor matched only singular curriculum tokens, so a plural noun "
                            "failed to re-parse and the (true) render was conservative-correctly rejected. This is a "
                            "VERIFY-coverage gap, NOT a moat breach (the gate abstains on untaught cues BEFORE any "
                            "render) and NOT a confabulation. The fix normalizes plural agent/object nouns back to "
                            "singular, then delegates to the SAME de-risk extractor. NO sim/ edit; NO edit to the "
                            "shared de-risk extractor (a local plural-aware wrapper in this runner).",
        "subject_first_template": SUBJECT_FIRST_TEMPLATE,
        "loose_template": LOOSE_TEMPLATE,
        "regen_template": REGEN_TEMPLATE,
        # --- brain-half rollup (the load-bearing scale check) ---
        "brain_recall_all_perfect": brain_recall_perfect,
        "brain_min_recall_rate": brain_min_recall,
        "brain_moat_all_clean": brain_moat_clean,
        "brain_max_moat_false_accepts": brain_max_breach,
        "brain_n_seeds_ok": len(ok_brain),
        # --- render-half rollup ---
        "render_grounded_all_correct": render_grounded_high,
        "render_moat_all_clean": render_moat_clean,
        "render_drift_all_caught": render_drift_caught,
        "subject_first_first_render_mean": sf_first_mean,
        "loose_first_render_mean": loose_first_mean,
        "subject_first_lift": (None if (sf_first_mean is None or loose_first_mean is None)
                               else round(sf_first_mean - loose_first_mean, 4)),
        "faculty_info": faculty_info,
        "faculty_error": faculty_err,
        "GO": go,
        "verdict": verdict,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "brain_per_seed": brain_per_seed,
        "render_per_seed": render_per_seed,
    }

    out_path = os.path.abspath(args.out)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False,
                  default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else o)

    print("\n" + "=" * 100, flush=True)
    print(f"[scaled] VERDICT: {verdict}", flush=True)
    print("=" * 100, flush=True)
    print(f"[scaled] wrote {out_path}", flush=True)
    return summary


if __name__ == "__main__":
    main()
