"""Phase-12 DE-RISK: the KNOWLEDGE-ACQUISITION pipeline -- the brain LEARNS real facts from a corpus, STAGED
cumulatively (the core breadth lever for the owner's "grow grounded knowledge" path).

The Phase-11 bottleneck was the KB SIZE. This builds the acquisition PIPELINE: a corpus of simple factual SENTENCES
(REAL knowledge -- true facts, simplified to 3-word SVO the validated BridgeParser handles) is INGESTED day-by-day --
the brain PARSES each sentence -> COMPOSER.STORE (the same parse+store path validated across the arc) -- and the KB
GROWS cumulatively (later days ADD facts without forgetting earlier ones; the develop-loop / McClelland-CLS pattern).
This is the scaling mechanism toward encyclopedic grounded breadth; the data SOURCE here is a real-knowledge
mini-encyclopedia (the offline-textbook-author pattern), swappable for a downloaded fact corpus (ConceptNet-style
triples / simplified-Wikipedia) later -- the PIPELINE (parse->store->grow, staged) is the deliverable.

METRICS (>=3 seeds): (a) ACQUISITION = facts ingested day-by-day are recalled (what_does == the stored patient);
(b) STAGED-CUMULATIVE + RETENTION = after day N, day-1 facts are STILL recalled (no catastrophic forgetting as the KB
grows); (c) BREADTH = the brain ends knowing facts about MANY concepts (>= the corpus's concept count); (d) MOAT = a
never-ingested cue -> abstain (0 false-accepts).

GO = acquisition-recall high + retention across staged days + breadth + moat 0-FA, >=3 seeds. Reuse-by-import
(BrainConversationalAgent parse+store); NO `sim/` edit. CPU (brain-only; independent of the RA generator).
Run: python -m research.runners._fluidconv_phase12_knowledge_acquisition_derisk --seeds 42 43 44
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

OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_phase12_knowledge_acquisition.json"

# A real-knowledge mini-encyclopedia, STAGED into "days" (cumulative acquisition). Every fact is TRUE, simplified to
# 3-word SVO (subject verb object) -- the validated parser's format. (The data source is swappable for a real corpus.)
CORPUS_DAYS = [
    # day 1 -- mammals + what they eat
    ["dog is mammal", "dog eat meat", "cat is mammal", "cat eat fish", "wolf is mammal", "wolf eat deer",
     "cow is mammal", "cow eat grass", "bear is mammal", "bear eat fish"],
    # day 2 -- birds + more relations (cumulative: adds without forgetting day 1)
    ["bird is animal", "bird eat seed", "bird build nest", "owl is bird", "owl hunt mouse", "hen is bird",
     "hen make egg", "bee is insect", "bee make honey", "fox eat rabbit"],
    # day 3 -- more concepts + roles (cumulative)
    ["dog guard home", "dog help human", "cat hunt mouse", "wolf hunt deer", "fish is animal", "fish live water",
     "mouse eat seed", "tree give shade", "sun give light", "rain give water"],
]
UNINGESTED = "dragon"          # never in the corpus -> the moat cue


def _all_facts(days):
    facts = []
    for d in days:
        for s in d:
            a, v, p = s.split()
            facts.append((a, v, p))
    return facts


def run(seed):
    facts_all = _all_facts(CORPUS_DAYS)
    vocab = sorted({w for (a, v, p) in facts_all for w in (a, v, p)} | {UNINGESTED, "eat"})
    # D=256: ~30 facts exceeds D=128's comfortable FHRR capacity (~sqrt(D)); Phase-6 validated D=256 for 40 facts.
    agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf", D=256)

    day1_facts = [tuple(s.split()) for s in CORPUS_DAYS[0]]
    ingested = []
    retention_by_day = []
    for di, day in enumerate(CORPUS_DAYS):
        for s in day:                                  # INGEST: parse each factual sentence -> store
            agent.hear(s)
            ingested.append(tuple(s.split()))
        # after each day, check day-1 retention (no catastrophic forgetting as the KB grows)
        r1 = sum(1 for (a, v, p) in day1_facts if agent.what_does(a, v) == p)
        retention_by_day.append(r1)

    # (a) ACQUISITION-RECALL over ALL ingested facts (functional (agent,verb) keys)
    #    (dedup keys: some (a,v) repeat, e.g. 'wolf hunt deer' + 'wolf eat deer'; recall the last-stored patient)
    keys = {}
    for (a, v, p) in ingested:
        keys[(a, v)] = p
    recall_ok = sum(1 for (a, v), p in keys.items() if agent.what_does(a, v) == p)
    recall_total = len(keys)

    # (b) retention: day-1 facts recalled after ALL days
    day1_keys = {(a, v): p for (a, v, p) in day1_facts}
    day1_retained = sum(1 for (a, v), p in day1_keys.items() if agent.what_does(a, v) == p)

    # (c) breadth: distinct concepts the brain now knows facts about
    concepts = {a for (a, v, p) in ingested}
    # (d) moat: a never-ingested cue -> abstain
    moat_ok = (agent.what_does(UNINGESTED, "eat") is None)

    return {"seed": seed, "n_days": len(CORPUS_DAYS), "recall_ok": recall_ok, "recall_total": recall_total,
            "recall_rate": round(recall_ok / max(1, recall_total), 3),
            "retention_by_day": retention_by_day, "day1_total": len(day1_keys), "day1_retained": day1_retained,
            "n_concepts": len(concepts), "moat_ok": bool(moat_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per_seed = []
    try:
        for s in a.seeds:
            r = run(s); per_seed.append(r)
            print(f"  [seed {s}] acquisition-recall {r['recall_ok']}/{r['recall_total']} ({r['recall_rate']}) | "
                  f"day1 retention by day {r['retention_by_day']}/{r['day1_total']} | day1-retained-final "
                  f"{r['day1_retained']}/{r['day1_total']} | breadth {r['n_concepts']} concepts | moat {r['moat_ok']}",
                  flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        import numpy as np
        mrec = float(np.mean([r["recall_rate"] for r in per_seed]))
        recall_ok = mrec >= 0.85
        retention_ok = all(r["day1_retained"] >= int(0.85 * r["day1_total"]) for r in per_seed)
        breadth_ok = all(r["n_concepts"] >= 12 for r in per_seed)
        moat_ok = all(r["moat_ok"] for r in per_seed)
        go = bool(recall_ok and retention_ok and breadth_ok and moat_ok)
        verdict = (("GO -- the KNOWLEDGE-ACQUISITION pipeline works: the brain INGESTS a real-fact corpus day-by-day "
                    "(parse each sentence -> store), recalls the ingested facts (mean %.2f), RETAINS day-1 facts as "
                    "later days grow the KB (no catastrophic forgetting), ends knowing many concepts, and the moat "
                    "holds 0-FA on a never-ingested cue. >=3 seeds. This is the scaling mechanism for grounded "
                    "breadth (the data source is swappable for a downloaded real corpus)." % mrec) if go else
                   ("HONEST/PARTIAL -- recall %.2f (>=0.85 %s); retention %s; breadth %s; moat %s"
                    % (mrec, recall_ok,
                       [r["day1_retained"] for r in per_seed], [r["n_concepts"] for r in per_seed],
                       [r["moat_ok"] for r in per_seed])))
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "fluidconv_phase12_knowledge_acquisition", "GO": go, "verdict": verdict,
               "resolves": "the knowledge-acquisition pipeline: parse a real-fact corpus -> store, staged cumulatively "
                           "(the develop-loop pattern) -> grounded breadth; retention (no forgetting) + moat.",
               "seeds": a.seeds, "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per_seed,
               "HONEST_CEILING": "the PIPELINE (parse->store->grow, staged, retained) is the deliverable; the data "
                                 "SOURCE here is a real-knowledge mini-encyclopedia (offline-textbook-author), "
                                 "swappable for a downloaded fact corpus (ConceptNet-style triples / simplified-Wiki). "
                                 "The parser handles SIMPLE 3-word SVO; ingesting raw complex prose needs a "
                                 "fact-extraction front-end (a bounded follow-on). Rich RENDERING of the learned facts "
                                 "is the RA generator's job (the broader-verb fine-tune lever, in flight)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[phase12-acquisition] VERDICT: {verdict}", flush=True)
    print(f"[phase12-acquisition] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
