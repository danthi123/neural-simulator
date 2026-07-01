"""Phase-4 DE-RISK: MULTI-TURN grounded conversation -- a pronoun in a follow-up turn resolves to the held referent.

Phase-3 gave the full single-turn grounded conversation (question -> comprehend -> gate -> RA-fine-tuned 21M focused
answer -> VERIFY). This closes the multi-turn axis (the owner's "fluid back-and-forth"): a persistent spiking
discourse working-memory loop holds the salient referent (the ANSWER of the prior turn) across a turn boundary, so a
later turn's PRONOUN ("what does IT eat?") resolves to it -- then routes through the same gate->answer->verify.

Uses the VALIDATED anaphora machinery (`MultiTurnAgent`, 2026-06-17 GO): `what_does("it", verb)` resolves the pronoun
INTERNALLY (via `_resolve`->`held_referent` over the spiking WM loop); after each turn the ANSWER (patient) is written
to the WM as the salient referent (`_write_referent`, exactly what `hear` does for a heard statement). The fluent
answer is rendered by the RA-fine-tuned 21M (`FTFaculty`) + post-hoc VERIFY. NO sim/ edit; reuse-by-import.

Discourse chain (curriculum-derived): (S1 v1 O1) and (O1 v2 O2), v1,v2 in the fine-tune's QA verbs -- e.g.
dog-chase-cat + cat-eat-fish:
  turn 1 "what does the dog chase?" -> the dog chases cat.   (writes 'cat' = the answer, salient)
  turn 2 "what does it eat?"        -> resolves it->cat -> the cat eats fish.

METRICS (>=3 seeds): (a) ANAPHORA = turn-2 pronoun resolves to O1 (the turn-1 answer) -> the correct grounded answer
(O2); (b) WM-LESION = wipe the WM before turn 2 -> the pronoun does NOT resolve -> abstain (the carry is
load-bearing); (c) EMPTY-WM MOAT = a turn-2 pronoun with NO prior turn -> abstain; (d) SINGLE-TURN unregressed.

GO = anaphora + WM-lesion-collapses + empty-WM-abstains + single-turn, >=3 seeds.

Run: python -m research.runners._fluidconv_phase4_multiturn_derisk --seeds 42 43 44
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners.multi_turn_agent import MultiTurnAgent  # noqa: E402
from research.runners._grounded_lang_p2_derisk import _collect_vocab, _teach, CURRICULUM  # noqa: E402
from research.runners._grounded_lang_integration_derisk import _build_inflection_map  # noqa: E402
from research.runners._fluidconv_phase1_grounded_continuation_derisk import _extract_all_svos, _fact_key  # noqa: E402
from research.runners._fluidconv_phase2_ra_finetune import VERBS, FT_CKPT  # noqa: E402
from research.runners._fluidconv_phase2_ra_qa_eval_derisk import FTFaculty, _v3  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_phase4_multiturn.json"


def _turn(mta, faculty, subj_or_it, verb, vs, *, lesion_wm=False, write_answer=True):
    """One turn: resolve (pronoun -> held referent, via the validated MultiTurnAgent), GATE, RA-render, VERIFY, and
    write the ANSWER as the salient referent for the next turn's pronoun."""
    agents, actions, patients, inflect, store_keys = vs
    if lesion_wm:
        mta.wm = None; mta._referent_history = []
        if getattr(mta, "bcw", None) is not None:
            mta.bcw = None
    resolved = mta._resolve(subj_or_it, query_verb=verb)     # 'it' -> held referent; a concrete word -> itself
    if resolved is None:
        return {"user": f"what does {subj_or_it} {verb}?", "reply": "I don't know.", "abstained": True,
                "resolved": None}
    p = mta.agent.what_does(resolved, verb)                  # GATE (moat gate-first)
    if p is None:
        return {"user": f"what does the {resolved} {verb}?", "reply": "I don't know.", "abstained": True,
                "resolved": resolved}
    ctx = f"the {resolved} {_v3(verb)} {p} ."
    ans = faculty.answer(ctx, f"what does the {resolved} {verb} ?")
    svos = _extract_all_svos(ans, agents, actions, patients, inflect)
    ungrounded = [s for s in svos if _fact_key(s) not in store_keys]
    verified = bool((([resolved, verb, p] in svos) or (p in ans.split())) and not ungrounded)
    reply = ans if verified else f"The {resolved} {_v3(verb)} {p}."
    if write_answer and p in mta.referents:
        mta._write_referent(p)                              # the ANSWER becomes the salient referent
    is_pron = isinstance(subj_or_it, str) and subj_or_it.lower() in ("it", "they", "that")
    user_echo = f"what does it {verb}?" if is_pron else f"what does the {resolved} {verb}?"
    return {"user": user_echo, "reply": reply, "abstained": False, "resolved": resolved,
            "answer_obj": p, "verified": verified}


def _find_chain(facts, ft_verbs):
    """A curriculum chain (S1,v1,O1),(O1,v2,O2) with v1,v2 in ft_verbs (O1 is object of fact-1 AND subject of fact-2)."""
    byav = {(a, v): p for a, v, p in facts}
    subj_facts = defaultdict(list)
    for a, v, p in facts:
        if v in ft_verbs:
            subj_facts[a].append((v, p))
    for a, v1, o1 in facts:
        if v1 not in ft_verbs:
            continue
        for v2, o2 in subj_facts.get(o1, []):
            return (a, v1, o1, v2, o2)
    return None


def run(cur, vocab, seed, faculty):
    facts = cur.get("facts", [])
    agents_set = {f[0] for f in facts}; patients_set = {f[2] for f in facts}; actions_set = {f[1] for f in facts}
    inflect = _build_inflection_map(sorted(actions_set)); store_keys = {tuple(f) for f in facts}
    vs = (agents_set, actions_set, patients_set, inflect, store_keys)
    ft_verbs = {b for (b, _s, _p) in VERBS}
    chain = _find_chain(facts, ft_verbs)
    if chain is None:
        return {"seed": seed, "error": "no ft-verb chain in curriculum"}
    s1, v1, o1, v2, o2 = chain
    # referents = a SMALL curated set (the chain concepts O1/O2/S1 + a few distractors), matching the validated
    # test scale (~6). The WM loop holds one pattern_size(=40)-neuron attractor per referent in n(=600) neurons, so
    # the referent count must stay well under n/pattern_size (~15) or read() overflows. Must include O1 (the anaphor
    # target) + O2 (the turn-2 answer, written as the next salient referent).
    referents = sorted({s1, o1, o2} | set(sorted(agents_set)[:5]))

    def _fresh():
        m = MultiTurnAgent(referent_concepts=referents, concepts={w: None for w in vocab},
                           seed=seed, defer_planner=True, enable_biased_competition=False, composer_kind="rf")
        _teach(m.agent, cur)
        return m

    result = {"seed": seed, "chain": list(chain)}

    # (a) ANAPHORA: turn-1 answer O1 -> written salient; turn-2 pronoun resolves it->O1 -> O2
    m = _fresh()
    t1 = _turn(m, faculty, s1, v1, vs)                       # "what does the dog chase?" -> cat (writes cat)
    t2 = _turn(m, faculty, "it", v2, vs)                     # "what does it eat?" -> it->cat -> fish
    result["turn1"] = t1; result["turn2"] = t2
    result["anaphora_ok"] = bool((t2.get("resolved") == o1) and (not t2["abstained"]) and (o2 in t2["reply"].split()))

    # (b) WM-LESION: wipe WM before turn 2 -> pronoun must NOT resolve
    m2 = _fresh()
    _turn(m2, faculty, s1, v1, vs)
    t2l = _turn(m2, faculty, "it", v2, vs, lesion_wm=True)
    result["turn2_lesioned"] = t2l
    result["lesion_ok"] = bool(t2l["abstained"] and t2l.get("resolved") is None)

    # (c) EMPTY-WM MOAT: pronoun with no prior turn -> abstain
    m3 = _fresh()
    t_empty = _turn(m3, faculty, "it", v2, vs)
    result["turn_empty"] = t_empty
    result["empty_ok"] = bool(t_empty["abstained"])

    # (d) SINGLE-TURN unregressed: a direct-subject turn answers grounded
    m4 = _fresh()
    ts = _turn(m4, faculty, s1, v1, vs)
    result["turn_single"] = ts
    result["single_ok"] = bool((not ts["abstained"]) and (o1 in ts["reply"].split()))
    return result


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
        print(f"[phase4-mt] loaded RA-fine-tuned ~{faculty.npar:.1f}M (dev={faculty.device})\n", flush=True)
        for s in a.seeds:
            r = run(cur, vocab, s, faculty)
            per_seed.append(r)
            print(f"  [seed {s}] chain={r.get('chain')} anaphora={r.get('anaphora_ok')} lesion={r.get('lesion_ok')} "
                  f"empty={r.get('empty_ok')} single={r.get('single_ok')}", flush=True)
        r0 = per_seed[0]
        if "turn1" in r0:
            print("\n  --- multi-turn transcript (seed 42) ---", flush=True)
            print(f"    you> {r0['turn1']['user']}\n    brain> {r0['turn1']['reply']}", flush=True)
            print(f"    you> {r0['turn2']['user']}\n    brain> {r0['turn2']['reply']}   "
                  f"(it -> {r0['turn2'].get('resolved')})", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        a_ok = all(r.get("anaphora_ok") for r in per_seed)
        l_ok = all(r.get("lesion_ok") for r in per_seed)
        e_ok = all(r.get("empty_ok") for r in per_seed)
        s_ok = all(r.get("single_ok") for r in per_seed)
        go = bool(a_ok and l_ok and e_ok and s_ok)
        verdict = (("GO -- MULTI-TURN grounded conversation: a turn-2 PRONOUN resolves to the turn-1 answer held in "
                    "the spiking WM loop -> correct grounded answer; WM-lesion collapses to abstain (load-bearing); "
                    "empty-WM pronoun abstains (moat); single-turn unregressed. >=3 seeds. Fluid back-and-forth on the "
                    "minimized, brain-trained, brain-gated stack.") if go else
                   ("HONEST/PARTIAL -- " + "; ".join(
                       ([] if a_ok else [f"anaphora {[r.get('anaphora_ok') for r in per_seed]}"]) +
                       ([] if l_ok else [f"WM-lesion {[r.get('lesion_ok') for r in per_seed]}"]) +
                       ([] if e_ok else [f"empty-WM moat {[r.get('empty_ok') for r in per_seed]}"]) +
                       ([] if s_ok else [f"single-turn {[r.get('single_ok') for r in per_seed]}"]))))
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "fluidconv_phase4_multiturn", "GO": go, "verdict": verdict,
               "resolves": "multi-turn grounded conversation: a follow-up pronoun resolves to the held discourse "
                           "referent (spiking WM loop) then routes through the Phase-3 gate->answer->verify.",
               "seeds": a.seeds, "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per_seed,
               "HONEST_CEILING": "anaphora over a single dominant held referent; >=2-referent disambiguation is the "
                                 "biased-competition path (opt-in, validated separately); open breadth is a follow-on."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[phase4-mt] VERDICT: {verdict}", flush=True)
    print(f"[phase4-mt] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
