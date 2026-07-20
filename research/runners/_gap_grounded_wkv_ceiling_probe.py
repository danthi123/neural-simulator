"""De-risk 0 (CEILING) — bound residual-B BEFORE spending training compute.

The north-star grounded-fluent-conversation frontier (research gate 2026-07-20) is a swap:
replace the fluid console's ~21M ANN renderer (FTFaculty) with the spiking WKV. The gate-first
moat + VERIFY + handoff are already GO (verified in `_fluidconv_chat_repl._answer:319-333`); the
one genuine residual is a format fine-tune so the WKV *answers* instead of *narrates* (residual-B).

This probe QUANTIFIES the raw (un-fine-tuned) WKV's ceiling on the 22-fact grounded curriculum,
using the console's OWN VERIFY (`_extract_all_svos`/`_fact_key`), so we know how big residual-B is
and whether a cheaper no-training path exists — the ceiling-first discipline
(feedback_run_ceiling_early_and_keep_gpu_busy).

Two prompting strategies (the raw WKV has NO punctuation/format tokens -> the FTFaculty
`facts:...answer:` format is un-representable; only NATURAL word prompts are):
  CONT     — prompt-condition on the grounded fact, generate, VERIFY (mirrors `_answer`).
  COMPLETE — prime "the A v3 P the A v3" (fact then subject+verb cue), is the next word P?
             (does prompt-conditioning carry the fact ~4 tokens into the read-out?)

Counts, per query: verified-fluent (states the fact, no ungrounded SVO) / would-confab (introduces
an ungrounded SVO) / fallback (neither -> the console would use the grounded template). Pure numpy,
no bridge, no training. NO `sim/` edit.
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
from research.runners._wkv_faculty import WKVFaculty, BIG_CKPT  # noqa: E402
from research.runners._fluidconv_phase1_grounded_continuation_derisk import _extract_all_svos, _fact_key  # noqa: E402
from research.runners._fluidconv_phase2_ra_qa_eval_derisk import _v3  # noqa: E402

CUR_PATH = "research/findings/raw/_grounded_lang_curriculum_p2.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=BIG_CKPT)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--curriculum", default=CUR_PATH)
    ap.add_argument("--out", default="research/findings/raw/_gap_grounded_wkv_ceiling.json")
    ap.add_argument("--show", type=int, default=8, help="print this many example generations")
    args = ap.parse_args()

    cur = json.load(open(args.curriculum))
    facts = [tuple(f) for f in cur.get("facts", [])]
    agents_set = {f[0] for f in facts}
    actions_set = {f[1] for f in facts}
    patients_set = {f[2] for f in facts}
    store_keys = {f for f in facts}
    from research.runners._grounded_lang_integration_derisk import _build_inflection_map
    inflect = _build_inflection_map(sorted(actions_set))

    fac = WKVFaculty(ckpt=args.ckpt, max_new=args.max_new)

    def _svos(t):
        return _extract_all_svos(t, agents_set, actions_set, patients_set, inflect)

    def _ung(t):
        return [s for s in _svos(t) if _fact_key(s) not in store_keys]

    # test on the 'patient' recall queries (mirror the phase2 eval scope), else fall back to all facts as (a,v)->p
    test = [(q["cue"][0], q["cue"][1]) for q in cur.get("queries_recall", []) if q["type"] == "patient"]
    if not test:
        test = [(a, v) for (a, v, p) in facts]
    # dedup, keep the taught patient
    seen = {}
    for a, v in test:
        p = next((pp for (aa, vv, pp) in facts if aa == a and vv == v), None)
        if p is not None and (a, v) not in seen:
            seen[(a, v)] = p
    cases = [(a, v, p) for (a, v), p in seen.items()]

    cont = {"verified": 0, "confab": 0, "fallback": 0}
    complete = {"top1": 0, "top5": 0, "n": 0}
    examples = []
    for (a, v, p) in cases:
        # --- CONT (mirror _answer) ---
        ctx = f"the {a} {_v3(v)} {p} ."
        ans = fac.answer(ctx, f"what does the {a} {v} ?")
        svos = _svos(ans); ung = _ung(ans)
        states_fact = ([a, v, p] in svos) or (p in ans.split())
        verified = bool(states_fact and not ung)
        if verified:
            cont["verified"] += 1
        elif ung:
            cont["confab"] += 1
        else:
            cont["fallback"] += 1
        # --- COMPLETE (fact-completion via priming) ---
        prime = ["the", a, _v3(v), p, "the", a, _v3(v)]
        prime = [w for w in prime if fac.in_vocab(w)]
        ranked = fac.next_ranked(prime)
        rank_words = [w for (w, _s) in ranked[:5]]
        complete["n"] += 1
        if rank_words and rank_words[0] == p:
            complete["top1"] += 1
        if p in rank_words:
            complete["top5"] += 1
        if len(examples) < args.show:
            examples.append({"cue": [a, v], "taught_p": p, "cont_ans": ans, "cont_verified": verified,
                             "cont_ung": ung, "complete_top5": rank_words})

    n = len(cases)
    print(f"\n=== CEILING (raw WKV, no fine-tune, n={n} grounded facts, ckpt={os.path.basename(args.ckpt)}) ===")
    print(f"CONT (prompt-condition + generate + console VERIFY):")
    print(f"   verified-fluent (states fact, no confab): {cont['verified']}/{n} = {cont['verified']/n:.2f}")
    print(f"   would-confab (introduces ungrounded SVO): {cont['confab']}/{n} = {cont['confab']/n:.2f}")
    print(f"   fallback (neither -> grounded template):  {cont['fallback']}/{n} = {cont['fallback']/n:.2f}")
    print(f"COMPLETE (fact-completion via priming, next-word == taught patient):")
    print(f"   top-1: {complete['top1']}/{complete['n']} = {complete['top1']/max(1,complete['n']):.2f}")
    print(f"   top-5: {complete['top5']}/{complete['n']} = {complete['top5']/max(1,complete['n']):.2f}")
    print(f"\nExamples:")
    for e in examples:
        print(f"   {e['cue']} (taught={e['taught_p']}): CONT={'OK' if e['cont_verified'] else 'ramble'} "
              f"ung={e['cont_ung']}  ans='{e['cont_ans'][:70]}'")
        print(f"       COMPLETE top5={e['complete_top5']}")
    verdict = ("residual-B (format fine-tune) NEEDED" if cont["verified"] / n < 0.5
               else "raw WKV already answers -- cheaper than expected")
    print(f"\nVERDICT: CONT verified {cont['verified']/n:.2f} -> {verdict}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"n": n, "cont": cont, "complete": complete, "examples": examples, "ckpt": args.ckpt,
               "verdict": verdict}, open(args.out, "w"), indent=2)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
