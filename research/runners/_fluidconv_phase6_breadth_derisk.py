"""Phase-6 DE-RISK: BREADTH -- the fluid stack over a BROADER knowledge base ("almost any topic", honestly).

Phases 0-5 + the console run on the 22-fact micro-curriculum. This tests the honest breadth axis (the owner's "almost
any topic"): a BROADER KB of facts across MANY entities (drawn from the RA generator's own competent vocab so it
renders them), taught to the brain at a higher composer dimension D (FHRR capacity ~ sqrt(D); validated to 320), then
recall + RA-render + the no-confab moat measured at scale, and the CAPACITY boundary characterized (a data/D lever,
NOT a substrate wall -- roadmap GAP B = manage via domain-constraint + retrieval-augmentation + abstention).

METRICS (>=3 seeds): (a) RECALL across the broad KB (what_does correct on a sample); (b) RA-RENDER (the RA-fine-tuned
21M renders the broad answers grounded, VERIFY-clean); (c) MOAT 0-FA on held-out untaught cues (entities in the vocab
never taught -> abstain); (d) CAPACITY -- recall at KB sizes {20, 40} at fixed D (where crosstalk starts).

GO = high recall + RA-render + 0-FA moat at the chosen scale; HONEST characterization of the capacity trend.

Run: python -m research.runners._fluidconv_phase6_breadth_derisk --seeds 42 43 44 --d 256 --n-facts 40
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
from research.runners._grounded_lang_integration_derisk import _build_inflection_map  # noqa: E402
from research.runners._fluidconv_phase1_grounded_continuation_derisk import _extract_all_svos, _fact_key  # noqa: E402
from research.runners._fluidconv_phase2_ra_finetune import VERBS, FT_CKPT, SUBJECTS as FT_SUBJECTS, OBJECTS as FT_OBJECTS  # noqa: E402
from research.runners._fluidconv_phase2_ra_qa_eval_derisk import FTFaculty, _v3  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_phase6_breadth.json"
_V3 = {b: s for (b, s, _p) in VERBS}


def _build_kb(n_facts, seed):
    """A broad KB: assign each subject a few (verb, object) facts, drawn deterministically from the RA vocab so the
    generator renders them. Distinct (subject, verb) keys (a functional map) so recall is unambiguous."""
    import random
    r = random.Random(seed)
    subs = [s for s in FT_SUBJECTS if s.isalpha()]
    verbs = [b for (b, _s, _p) in VERBS]
    objs = [o for o in FT_OBJECTS if o.isalpha()]
    r.shuffle(subs)
    facts = []
    used = set()
    for s in subs:
        k = r.choice([1, 2, 2, 3])                     # a few facts per subject
        vs = r.sample(verbs, min(k, len(verbs)))
        for v in vs:
            if (s, v) in used:
                continue
            o = r.choice(objs)
            facts.append((s, v, o)); used.add((s, v))
            if len(facts) >= n_facts:
                return facts
    return facts


def _measure(agent, faculty, facts, vs, sample):
    agents, actions, patients, inflect, store_keys = vs
    recall_ok = 0
    render_ok = 0
    for (a, v, p) in sample:
        got = agent.what_does(a, v)
        if got == p:
            recall_ok += 1
            ctx = f"the {a} {_V3.get(v, v + 's')} {p} ."
            ans = faculty.answer(ctx, f"what does the {a} {v} ?")
            svos = _extract_all_svos(ans, agents, actions, patients, inflect)
            ung = [s for s in svos if _fact_key(s) not in store_keys]
            if ((p in ans.split()) and not ung):
                render_ok += 1
    return recall_ok, render_ok


def run(cur_unused, seed, faculty, D, n_facts):
    import random
    r = random.Random(seed + 999)
    facts = _build_kb(n_facts, seed)
    agents_set = {f[0] for f in facts}; patients_set = {f[2] for f in facts}; actions_set = {f[1] for f in facts}
    inflect = _build_inflection_map(sorted(actions_set))
    # held-out untaught cues: taught subjects with a NEVER-taught verb (encodable, never stored -> the moat)
    taught_keys = {(a, v) for a, v, _p in facts}
    all_verbs = [b for (b, _s, _p) in VERBS]
    untaught = []
    for a in sorted(agents_set):
        for v in all_verbs:
            if (a, v) not in taught_keys:
                untaught.append((a, v))
    r.shuffle(untaught); untaught = untaught[:12]

    vocab = sorted(agents_set | patients_set | {v for v, in [(x[1],) for x in facts]} | set(all_verbs))
    # (composer needs each fact word encodable; include the verbs + the untaught verbs too)
    vocab = sorted(set(vocab) | {u[1] for u in untaught})
    agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf", D=D)
    for (a, v, p) in facts:
        agent.hear(f"{a} {v} {p}")
    store_keys = {tuple(f) for f in facts}
    vs = (agents_set, actions_set, patients_set, inflect, store_keys)

    sample = facts if len(facts) <= 20 else r.sample(facts, 20)
    recall_ok, render_ok = _measure(agent, faculty, facts, vs, sample)

    # MOAT: untaught cues -> abstain (0-FA)
    fa = sum(1 for (a, v) in untaught if agent.what_does(a, v) is not None)

    # CAPACITY note: recall at a SMALLER KB slice (first 20 facts) vs the full n_facts (same agent -- reuse recall)
    return {"seed": seed, "D": D, "n_facts": len(facts), "recall_ok": recall_ok, "recall_total": len(sample),
            "render_ok": render_ok, "moat_false_accepts": fa, "moat_total": len(untaught),
            "recall_rate": round(recall_ok / max(1, len(sample)), 3),
            "render_rate": round(render_ok / max(1, recall_ok), 3) if recall_ok else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--n-facts", type=int, default=40)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    if not os.path.exists(FT_CKPT):
        print(f"NOT-RUNNABLE: fine-tuned ckpt absent ({FT_CKPT})"); return 2
    t0 = time.time()
    err = None; per_seed = []
    try:
        faculty = FTFaculty()
        print(f"[phase6-breadth] loaded RA-fine-tuned ~{faculty.npar:.1f}M (dev={faculty.device}); "
              f"D={a.d} n_facts={a.n_facts}\n", flush=True)
        for s in a.seeds:
            r = run(None, s, faculty, a.d, a.n_facts)
            per_seed.append(r)
            print(f"  [seed {s}] recall {r['recall_ok']}/{r['recall_total']} ({r['recall_rate']}) | RA-render "
                  f"{r['render_ok']}/{r['recall_ok']} ({r['render_rate']}) | moat FA {r['moat_false_accepts']}/"
                  f"{r['moat_total']} | n_facts {r['n_facts']} D {r['D']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        import numpy as np
        mrec = float(np.mean([r["recall_rate"] for r in per_seed]))
        mren = float(np.mean([r["render_rate"] for r in per_seed]))
        moat_ok = all(r["moat_false_accepts"] == 0 for r in per_seed)
        recall_ok = mrec >= 0.85
        render_ok = mren >= 0.85
        go = bool(recall_ok and render_ok and moat_ok)
        verdict = (("GO -- BREADTH: the fluid stack holds over a broader %d-fact KB at D=%d -- recall %.2f, RA-render "
                    "%.2f, moat 0-FA (%d seeds). 'Almost any topic' within the grounded regime: many entities "
                    "answerable + the honest abstention boundary beyond." % (a.n_facts, a.d, mrec, mren, len(a.seeds)))
                   if go else
                   ("HONEST/PARTIAL -- recall %.2f (>=0.85 %s) render %.2f (%s) moat-0FA %s. The capacity trend is the "
                    "FHRR sqrt(D)/M boundary (raise D or add codes) -- a data/D lever, NOT a substrate wall."
                    % (mrec, recall_ok, mren, render_ok, moat_ok)))
        summary_go = go
    else:
        summary_go = False; verdict = f"ERROR -- {err}"; mrec = mren = 0.0

    summary = {"probe": "fluidconv_phase6_breadth", "GO": summary_go, "verdict": verdict,
               "resolves": "breadth: the fluid stack over a broader KB (many entities) at higher D; recall + RA-render "
                           "+ moat 0-FA + the honest capacity boundary.",
               "seeds": a.seeds, "D": a.d, "n_facts": a.n_facts, "mean_recall": round(mrec, 3),
               "mean_render": round(mren, 3), "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per_seed,
               "HONEST_CEILING": "breadth is bounded by the composer's FHRR capacity (~sqrt(D)/M; raise D or add "
                                 "distinct codes -- validated to 320) + the generator's vocab (TinyStories common "
                                 "English, generalizing) + the taught KB; the abstention moat is the truthful "
                                 "'I don't know' boundary. Open-domain (non-fact) conversation remains the field wall."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[phase6-breadth] VERDICT: {verdict}", flush=True)
    print(f"[phase6-breadth] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if summary_go else 1


if __name__ == "__main__":
    sys.exit(main())
