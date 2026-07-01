"""Phase-3 assembly: the FULL grounded conversational TURN end-to-end -- "talk to it like an LLM".

Assembles the validated pieces into ONE conversational turn: a free-text user question -> the brain COMPREHENDS it
(maps it to a structured query) -> the brain GATE recalls the fact OR abstains (the no-confab moat, gate-FIRST) ->
the RA-fine-tuned ~21M generator renders a FOCUSED fluent grounded answer (only when the gate has a fact) -> post-hoc
VERIFY re-parses the answer and rejects drift -> the reply (or "I don't know").

  user text --> [comprehend: interrogative parse] --> (qtype, cue)
             --> [GATE: brain what_does/who_does/is_it_true] --> fact | ABSTAIN
             --> if abstain: reply "I don't know" (model NEVER invoked = no-confab by construction)
             --> else: [RA-prompt -> fine-tuned 21M -> focused answer] -> [VERIFY re-parse] -> reply | regenerate/abstain

This is the owner's north star assembled from validated parts: the BRAIN supplies comprehension + knowledge +
grounding + the moat; the MINIMIZED (~21M, 15-25x < Qwen-0.5B), brain-trained, brain-gated generator supplies fluency.

SCAFFOLD (flagged): the interrogative parse (question -> structured query) is a light rule-based comprehension over
the brain's OWN vocab + wh/aux cue words; the brain-based replacement is a neural interrogative parser (a follow-on,
same family as the declarative BridgeParser). The surface morphology of the answer is the fine-tuned generator's.

METRICS (>=3 seeds): (a) GROUNDED-REPLY = a grounded question -> a focused fluent reply that VERIFY confirms states
the fact; (b) MOAT = an untaught question -> "I don't know" (gate-first, model not invoked); (c) DRIFT = an
adversarial (wrong-fact-in-context) turn -> VERIFY rejects -> abstain/regenerate (never emits the false fact).

GO = grounded-reply + moat + drift on the curriculum set, >=3 seeds. Also prints a scripted transcript (the demo).

Run: python -m research.runners._fluidconv_phase3_conversational_turn_derisk --seeds 42 43 44
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402
from research.runners._grounded_lang_p2_derisk import _collect_vocab, _teach, CURRICULUM  # noqa: E402
from research.runners._grounded_lang_integration_derisk import _build_inflection_map  # noqa: E402
from research.runners._fluidconv_phase1_grounded_continuation_derisk import _extract_all_svos, _fact_key  # noqa: E402
from research.runners._fluidconv_phase2_ra_finetune import VERBS, FT_CKPT  # noqa: E402
from research.runners._fluidconv_phase2_ra_qa_eval_derisk import FTFaculty, _v3  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_phase3_conversational_turn.json"


def parse_question(text, agents, actions, patients, inflect):
    """SCAFFOLD interrogative parse: free-text question -> (qtype, cue). Uses wh/aux cue words for the type + the
    brain's OWN vocab to pull content words. (The brain-based replacement is a neural interrogative parser.)"""
    toks = [t.strip("?.!,") for t in text.lower().split()]
    tset = set(toks)
    subj = next((t for t in toks if t in agents), None)
    obj = next((t for t in toks if t in patients and t != subj), None)
    verb = next((inflect.get(t) for t in toks if inflect.get(t) in actions), None)
    if "who" in tset:
        if verb and obj:
            return ("agent", (verb, obj))
    if ("does" in tset or "do" in tset or "is" in tset) and subj and verb and obj:
        return ("yesno", (subj, verb, obj))
    if ("tell" in tset or "about" in tset) and subj:
        # describe -> treat as a patient query on the subject's first known verb (the gate resolves the fact)
        return ("describe", (subj, verb))
    if "what" in tset and subj and verb:
        return ("patient", (subj, verb))
    # fallback: best-effort patient query if we have subject+verb
    if subj and verb:
        return ("patient", (subj, verb))
    return (None, None)


def conversational_turn(agent, faculty, user_text, vocab_sets, wrong_ctx=None):
    """One full grounded conversational turn. Returns a record (parsed query, gate result, reply, verified, abstained)."""
    agents, actions, patients, inflect, store_keys = vocab_sets
    qtype, cue = parse_question(user_text, agents, actions, patients, inflect)

    def _svos(t): return _extract_all_svos(t, agents, actions, patients, inflect)

    # (i) comprehend failed -> honest abstain
    if qtype is None:
        return {"user": user_text, "qtype": None, "reply": "I don't know.", "abstained": True, "verified": None}

    # (ii) GATE (brain recall / abstain) -- the moat, FIRST
    gate_fact = None
    if qtype in ("patient", "describe"):
        a, v = cue if qtype == "patient" else (cue[0], cue[1])
        # describe: if verb unknown, find the subject's first stored fact
        if v is None:
            v = next((vv for vv in sorted(actions) if agent.what_does(a, vv) is not None), None)
        p = agent.what_does(a, v) if v else None
        gate_fact = [a, v, p] if p is not None else None
    elif qtype == "agent":
        v, p = cue
        a = agent.who_does(v, p)
        gate_fact = [a, v, p] if a is not None else None
    elif qtype == "yesno":
        a, v, p = cue
        truth = agent.is_it_true(a, v, p)
        # yes/no answered from the brain directly (no generation needed for the polarity)
        reply = {"yes": f"Yes, the {a} {_v3(v)} {p}.", "no": "No.", "unknown": "I don't know."}[truth]
        return {"user": user_text, "qtype": qtype, "cue": cue, "gate_truth": truth,
                "reply": reply, "abstained": (truth == "unknown"), "verified": True}

    # gate abstained -> "I don't know" WITHOUT invoking the generator (no-confab by construction)
    if gate_fact is None or gate_fact[2] is None or gate_fact[1] is None:
        return {"user": user_text, "qtype": qtype, "cue": cue, "gate_fact": None,
                "reply": "I don't know.", "abstained": True, "verified": None}

    # (iii) RENDER: RA-prompt the fine-tuned generator with the gated fact (or an adversarial wrong ctx for the drift test)
    a, v, p = gate_fact
    ctx = wrong_ctx if wrong_ctx else f"the {a} {_v3(v)} {p} ."
    q_for_model = f"what does the {a} {v} ?" if qtype != "agent" else f"who {_v3(v)} {p} ?"
    ans = faculty.answer(ctx, q_for_model)

    # (iv) VERIFY: re-parse the answer; every known-entity SVO must match the gated fact (no drift)
    svos = _svos(ans)
    ungrounded = [s for s in svos if _fact_key(s) not in store_keys]
    states_fact = ([a, v, p] in svos) or (p in ans.split())
    verified = bool(states_fact and len(ungrounded) == 0)
    if not verified:
        # drift caught -> do NOT emit the unverified reply; honest fallback to the brain's own grounded statement
        return {"user": user_text, "qtype": qtype, "gate_fact": gate_fact, "model_answer": ans,
                "ungrounded": ungrounded, "reply": f"The {a} {_v3(v)} {p}.", "abstained": False,
                "verified": False, "drift_rejected": True}
    return {"user": user_text, "qtype": qtype, "gate_fact": gate_fact, "reply": ans,
            "abstained": False, "verified": True, "drift_rejected": False}


def run(cur, vocab, seed, faculty):
    agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf")
    _teach(agent, cur)
    facts = cur.get("facts", [])
    agents_set = {f[0] for f in facts}; patients_set = {f[2] for f in facts}; actions_set = {f[1] for f in facts}
    inflect = _build_inflection_map(sorted(actions_set))
    store_keys = {tuple(f) for f in facts}
    vs = (agents_set, actions_set, patients_set, inflect, store_keys)
    ft_verbs = {b for (b, _s, _p) in VERBS}

    # grounded questions (free text), verb-diverse: derive from queries_recall patient queries (spans eat/chase/make),
    # over facts the fine-tune saw in QA format. Resolve each cue's patient from the brain GATE (ground truth).
    gq = []
    for q in cur.get("queries_recall", []):
        if q["type"] != "patient" or q["cue"][1] not in ft_verbs:
            continue
        a, v = q["cue"]; p = agent.what_does(a, v)
        if p is not None:
            gq.append((f"what does the {a} {v} ?", (a, v, p)))
    gq = gq[:5]
    grounded = []
    for text, (a, v, p) in gq:
        rec = conversational_turn(agent, faculty, text, vs)
        rec["expected"] = [a, v, p]
        rec["ok"] = bool(rec["verified"] and not rec["abstained"] and (p in rec["reply"].split()))
        grounded.append(rec)

    # untaught questions -> "I don't know" (moat, gate-first)
    moat = []
    for uq in [x for x in cur.get("queries_moat", []) if x["type"] == "patient"][:3]:
        a, v = uq["cue"]
        rec = conversational_turn(agent, faculty, f"what does the {a} {v} ?", vs)
        rec["held"] = bool(rec["abstained"] and "know" in rec["reply"].lower())
        moat.append(rec)

    # drift: force a wrong fact into the model's context; VERIFY must reject -> no false emission
    drift = []
    for text, (a, v, p) in gq[:3]:
        wrong_p = next((o for o in sorted(patients_set) if o != p), p)
        rec = conversational_turn(agent, faculty, text, vs, wrong_ctx=f"the {a} {_v3(v)} {wrong_p} .")
        # caught = the reply does NOT assert the wrong patient (either VERIFY rejected, or the fallback restated truth)
        rec["wrong_p"] = wrong_p
        rec["caught"] = bool(wrong_p not in rec["reply"].split())
        drift.append(rec)

    n_ok = sum(r["ok"] for r in grounded); n_held = sum(r["held"] for r in moat); n_caught = sum(r["caught"] for r in drift)
    return {"seed": seed, "grounded_ok": n_ok, "grounded_total": len(grounded),
            "moat_held": n_held, "moat_total": len(moat), "drift_caught": n_caught, "drift_total": len(drift),
            "grounded_detail": grounded, "moat_detail": moat, "drift_detail": drift}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    if not os.path.exists(FT_CKPT):
        print(f"NOT-RUNNABLE: fine-tuned ckpt absent ({FT_CKPT})"); return 2
    t0 = time.time()
    with open(os.path.abspath(CURRICULUM), "r", encoding="utf-8") as fh:
        cur = json.load(fh)
    vocab = _collect_vocab(cur)
    err = None; per_seed = []
    try:
        faculty = FTFaculty()
        print(f"[phase3-turn] loaded RA-fine-tuned ~{faculty.npar:.1f}M (dev={faculty.device})\n", flush=True)
        for s in a.seeds:
            r = run(cur, vocab, s, faculty)
            per_seed.append(r)
            print(f"  [seed {s}] grounded-reply {r['grounded_ok']}/{r['grounded_total']} | moat {r['moat_held']}/"
                  f"{r['moat_total']} | drift-caught {r['drift_caught']}/{r['drift_total']}", flush=True)
        # scripted transcript (the demo, seed 42)
        print("\n  --- scripted transcript (seed 42) ---", flush=True)
        for r in per_seed[0]["grounded_detail"][:3]:
            print(f"    you> {r['user']}\n    brain> {r['reply']}", flush=True)
        for r in per_seed[0]["moat_detail"][:1]:
            print(f"    you> {r['user']}\n    brain> {r['reply']}   (untaught -> gate-first abstain)", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        g_ok = all(r["grounded_ok"] == r["grounded_total"] and r["grounded_total"] > 0 for r in per_seed)
        m_ok = all(r["moat_held"] == r["moat_total"] and r["moat_total"] > 0 for r in per_seed)
        d_ok = all(r["drift_caught"] == r["drift_total"] and r["drift_total"] > 0 for r in per_seed)
        go = bool(g_ok and m_ok and d_ok)
        verdict = (("GO -- the FULL grounded conversational TURN works end-to-end: free-text question -> brain "
                    "comprehend+gate -> RA-fine-tuned 21M focused fluent grounded reply -> VERIFY; untaught -> "
                    "gate-first 'I don't know' (model not invoked); adversarial drift -> VERIFY rejects. >=3 seeds. "
                    "'Talk to it like an LLM' assembled from validated parts (brain: comprehension+knowledge+moat; "
                    "minimized generator: fluency).") if go else
                   ("HONEST/PARTIAL -- " + "; ".join(
                       ([] if g_ok else [f"grounded-reply {[r['grounded_ok'] for r in per_seed]}/{[r['grounded_total'] for r in per_seed]}"]) +
                       ([] if m_ok else [f"moat {[r['moat_held'] for r in per_seed]} (abstain leak)"]) +
                       ([] if d_ok else [f"drift-caught {[r['drift_caught'] for r in per_seed]} (a false fact emitted)"]))))
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "fluidconv_phase3_conversational_turn", "GO": go, "verdict": verdict,
               "resolves": "the full grounded conversational turn end-to-end (comprehend -> gate -> fluent grounded "
                           "answer -> verify), 'talk to it like an LLM' from validated parts.",
               "seeds": a.seeds, "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per_seed,
               "SCAFFOLD_NOTE": "the interrogative parse is a light rule-based comprehension over the brain's vocab + "
                                "wh/aux cues; the brain-based replacement is a neural interrogative parser (follow-on).",
               "HONEST_CEILING": "single-turn grounded Q&A; multi-turn coherence (persistent discourse referents + "
                                 "the multi-referent WTA) + open breadth are the follow-ons."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[phase3-turn] VERDICT: {verdict}", flush=True)
    print(f"[phase3-turn] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
