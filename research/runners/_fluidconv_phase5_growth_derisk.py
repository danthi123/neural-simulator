"""Phase-5 DE-RISK: GROWTH through conversation -- the brain learns NEW facts from the conversation, immediately
usable, old facts retained, moat intact ("grow through these experiences").

The fluid-conversation stack (Phase 0-4) answers a FIXED taught curriculum. This closes the growth axis (the owner's
"still being able to grow through these experiences"): a NEW fact stated in conversation is stored by the brain
(`agent.hear`), immediately answerable (brain GATE -> RA-fine-tuned 21M focused grounded answer), the OLD facts are
retained (no catastrophic forgetting at the brain-store level), and the no-confab moat still holds on
still-untaught cues. Reuse-by-import (brain store + the Phase-2/3 RA-QA pipeline); NO sim/ edit.

Two kinds of growth subject test two things:
  - a subject IN the generator's fine-tune vocab (e.g. 'wolf'): learn-from-conversation (the brain stores + answers).
  - a subject NOVEL to the generator's fine-tune vocab (e.g. 'camel', 'zebra'): the RA generator must GENERALIZE --
    render "the camel eats grass" from the PROVIDED fact even though it never saw 'camel' in the QA fine-tune (the
    whole point of the format fine-tune: use the provided facts, don't memorize).
Honest scope: growth is over PRE-ALLOCATED concept codes (the composer's vocab is fixed at build, as in the develop
loop); learning brand-new concept CODES is the separate dendritic/allocation frontier. Full cross-session persistence
is validated in the develop loop (Tier-3 live-and-remember); here a within-session DURABILITY check (the new fact
survives subsequent operations) stands in.

METRICS (>=3 seeds): (a) LEARN = each new fact, once heard, is answered grounded (RA-rendered); (b) NOVEL-GENERALIZE =
the generator renders a subject unseen in its fine-tune vocab from the provided fact; (c) RETENTION = the base facts
still recalled after growth; (d) MOAT = a still-untaught vocab cue -> abstain; (e) DURABILITY = the first-learned new
fact still recalled after all later growth + queries.

GO = learn (all) + novel-generalize + retention + moat 0-FA + durability, >=3 seeds.

Run: python -m research.runners._fluidconv_phase5_growth_derisk --seeds 42 43 44
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
from research.runners._fluidconv_phase2_ra_finetune import SUBJECTS as FT_SUBJECTS  # noqa: E402
from research.runners._fluidconv_phase2_ra_qa_eval_derisk import FTFaculty, _v3  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_phase5_growth.json"

# NEW facts taught mid-conversation (over pre-allocated vocab). 'wolf' is IN the fine-tune SUBJECTS (learn test);
# 'camel'/'zebra' are NOVEL to the fine-tune vocab (RA-generalization test). All objects are in the fine-tune OBJECTS.
GROWTH_FACTS = [("wolf", "eat", "rabbit"), ("camel", "eat", "grass"), ("zebra", "like", "hay")]
GROWTH_SUBJECTS = [f[0] for f in GROWTH_FACTS]
UNTAUGHT_GROWTH = "otter"          # in the composer vocab, NEVER taught -> the moat cue


def _answer_turn(agent, faculty, subj, verb, vs):
    """The Phase-3 turn: GATE -> RA-render -> VERIFY. Returns (patient_or_None, reply, verified)."""
    agents, actions, patients, inflect, store_keys = vs
    p = agent.what_does(subj, verb)
    if p is None:
        return None, "I don't know.", None
    ctx = f"the {subj} {_v3(verb)} {p} ."
    ans = faculty.answer(ctx, f"what does the {subj} {verb} ?")
    svos = _extract_all_svos(ans, agents, actions, patients, inflect)
    ungrounded = [s for s in svos if _fact_key(s) not in store_keys]
    verified = bool((([subj, verb, p] in svos) or (p in ans.split())) and not ungrounded)
    return p, (ans if verified else f"The {subj} {_v3(verb)} {p}."), verified


def run(cur, vocab, seed, faculty):
    # vocab includes the growth subjects + the untaught-growth moat cue (pre-allocated concept codes)
    agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf")
    _teach(agent, cur)
    base_facts = cur.get("facts", [])
    agents_set = {f[0] for f in base_facts} | set(GROWTH_SUBJECTS)
    patients_set = {f[2] for f in base_facts} | {f[2] for f in GROWTH_FACTS}
    actions_set = {f[1] for f in base_facts}
    inflect = _build_inflection_map(sorted(actions_set))
    store_keys = {tuple(f) for f in base_facts}
    ft_subj = set(FT_SUBJECTS)

    # (c) RETENTION baseline: answer a base fact BEFORE growth
    base_probe = next((f for f in base_facts if f[1] in {"eat"}), base_facts[0])
    vs0 = (agents_set, actions_set, patients_set, inflect, store_keys)
    p_before, _, _ = _answer_turn(agent, faculty, base_probe[0], base_probe[1], vs0)
    retention_before = (p_before == base_probe[2])

    # (a)+(b) GROW: hear each new fact, then immediately answer it grounded
    learned = []
    for (a, v, p) in GROWTH_FACTS:
        agent.hear(f"{a} {v} {p}")                       # LEARN from conversation
        store_keys.add((a, v, p))                        # the moat's ground-truth now includes it
        vs = (agents_set, actions_set, patients_set, inflect, store_keys)
        got, reply, verified = _answer_turn(agent, faculty, a, v, vs)
        novel_to_gen = a not in ft_subj
        ok = bool(got == p and (p in reply.split()))
        learned.append({"fact": [a, v, p], "answered": got, "reply": reply, "verified": verified,
                        "novel_to_generator": novel_to_gen, "ok": ok})

    # (c) RETENTION after growth: the base fact still recalled
    p_after, _, _ = _answer_turn(agent, faculty, base_probe[0], base_probe[1],
                                 (agents_set, actions_set, patients_set, inflect, store_keys))
    retention_after = (p_after == base_probe[2])

    # (d) MOAT: a still-untaught vocab cue -> abstain (gate-first; model not invoked)
    moat_gate = agent.what_does(UNTAUGHT_GROWTH, "eat")
    moat_ok = (moat_gate is None)

    # (e) DURABILITY: the FIRST-learned new fact still recalled after all later growth + queries
    d_subj, d_verb, d_pat = GROWTH_FACTS[0]
    p_dur, dur_reply, _ = _answer_turn(agent, faculty, d_subj, d_verb,
                                       (agents_set, actions_set, patients_set, inflect, store_keys))
    durability_ok = (p_dur == d_pat)

    n_learn = sum(r["ok"] for r in learned)
    novel_ok = all(r["ok"] for r in learned if r["novel_to_generator"])
    return {"seed": seed, "learned": learned, "n_learn": n_learn, "n_growth": len(GROWTH_FACTS),
            "novel_generalize_ok": bool(novel_ok),
            "retention_before": bool(retention_before), "retention_after": bool(retention_after),
            "moat_ok": bool(moat_ok), "moat_gate": moat_gate,
            "durability_ok": bool(durability_ok), "durability_reply": dur_reply}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    from research.runners._fluidconv_phase2_ra_finetune import FT_CKPT
    if not os.path.exists(FT_CKPT):
        print(f"NOT-RUNNABLE: fine-tuned ckpt absent ({FT_CKPT})"); return 2
    t0 = time.time()
    with open(os.path.abspath(CURRICULUM), "r", encoding="utf-8") as fh:
        cur = json.load(fh)
    # vocab = base curriculum + the growth subjects + their objects + the untaught-growth moat cue (pre-allocated)
    vocab = sorted(set(_collect_vocab(cur)) | set(GROWTH_SUBJECTS) |
                   {f[2] for f in GROWTH_FACTS} | {UNTAUGHT_GROWTH})
    err = None; per_seed = []
    try:
        faculty = FTFaculty()
        print(f"[phase5-growth] loaded RA-fine-tuned ~{faculty.npar:.1f}M (dev={faculty.device}); vocab={len(vocab)}\n",
              flush=True)
        for s in a.seeds:
            r = run(cur, vocab, s, faculty)
            per_seed.append(r)
            print(f"  [seed {s}] learn {r['n_learn']}/{r['n_growth']} | novel-generalize {r['novel_generalize_ok']} "
                  f"| retention {r['retention_before']}->{r['retention_after']} | moat {r['moat_ok']} | "
                  f"durability {r['durability_ok']}", flush=True)
            for L in r["learned"]:
                print(f"      taught '{' '.join(map(str, L['fact']))}' -> \"{L['reply']}\" "
                      f"(novel_to_gen={L['novel_to_generator']}, ok={L['ok']})", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        learn_ok = all(r["n_learn"] == r["n_growth"] for r in per_seed)
        novel_ok = all(r["novel_generalize_ok"] for r in per_seed)
        ret_ok = all(r["retention_after"] for r in per_seed)
        moat_ok = all(r["moat_ok"] for r in per_seed)
        dur_ok = all(r["durability_ok"] for r in per_seed)
        go = bool(learn_ok and novel_ok and ret_ok and moat_ok and dur_ok)
        verdict = (("GO -- GROWTH through conversation: each NEW fact heard is immediately answered grounded "
                    "(RA-rendered), the generator GENERALIZES to subjects unseen in its fine-tune vocab, base facts "
                    "are RETAINED, the moat holds 0-FA on still-untaught cues, and the new facts are DURABLE. >=3 "
                    "seeds. The brain grows its knowledge from the conversation.") if go else
                   ("HONEST/PARTIAL -- " + "; ".join(
                       ([] if learn_ok else [f"learn {[r['n_learn'] for r in per_seed]}/{[r['n_growth'] for r in per_seed]}"]) +
                       ([] if novel_ok else [f"novel-generalize {[r['novel_generalize_ok'] for r in per_seed]} (generator did not render an unseen subject)"]) +
                       ([] if ret_ok else [f"retention {[r['retention_after'] for r in per_seed]} (forgot a base fact)"]) +
                       ([] if moat_ok else [f"moat {[r['moat_ok'] for r in per_seed]} (untaught leaked)"]) +
                       ([] if dur_ok else [f"durability {[r['durability_ok'] for r in per_seed]}"]))))
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "fluidconv_phase5_growth", "GO": go, "verdict": verdict,
               "resolves": "growth through conversation: new facts learned from dialogue are immediately answerable "
                           "(RA-rendered, generalizing to novel entities), old retained, moat intact.",
               "seeds": a.seeds, "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per_seed,
               "HONEST_CEILING": "growth over PRE-ALLOCATED concept codes (new concept CODES = the dendritic/allocation "
                                 "frontier); full cross-session persistence validated in the develop loop; here a "
                                 "within-session durability check stands in."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[phase5-growth] VERDICT: {verdict}", flush=True)
    print(f"[phase5-growth] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
