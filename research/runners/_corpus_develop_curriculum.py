#!/usr/bin/env python
"""Corpus-driven developmental curriculum -- the deep-knowledge SCALING of the longitudinal develop-loop's
cumulative new-concept growth.

The validated develop-loop (research.runners._longitudinal_develop_loop_gpu) grows a brain's vocabulary
day-by-day with no catastrophic forgetting (WAKE = stream-cortex learns the day's NEW concepts from the corpus;
CONVERSE = store facts + probe; SLEEP = self-replay + retention re-test; PERSIST = BridgeLineage save/resume),
but its GPUGradedCurriculum hardcodes a 6-concepts/day toy syllabus (~24 concepts total). This module builds a
GradedCurriculum syllabus from the POS-admitted CORPUS concepts (chunked N/"day", high-frequency first = the
developmental simple->rich shape) + the corpus-extracted SVO facts (each fact asserted on the day its concepts
first all become available). Feeding this to develop_gpu scales the no-forget growth to thousands of concepts.

Pair with: the multi-bridge cleanup (RoutedComposer) at recall (so grown vocab stays at recall>=0.95) + the
cp_connections-resume (--resume-bridge, CYCLE 632, corr 1.000) replacing the develop-loop's expensive re-hear.

Host-side curriculum prep (legitimate -- preparing the syllabus the brain then LEARNS). CPU.
"""
import argparse
import json
import sys

import numpy as np

from research.runners._longitudinal_develop_loop import GradedCurriculum  # noqa: E402


def build_corpus_syllabus(concepts_ranked, facts, *, concepts_per_day=24, max_facts_per_day=8,
                          max_recall_probes=3):
    """Build a develop-loop syllabus from corpus concepts + corpus facts.

    concepts_ranked : list[str]  -- concept words, HIGH-FREQUENCY FIRST (the order they are introduced).
    facts           : list[(agent, action, patient)]  -- corpus-extracted SVO tuples.

    Each "day" introduces the next `concepts_per_day` concepts and asserts the facts that became fully-available
    that day (all three words known AND >=1 word introduced today, so the fact carries new content). Returns the
    syllabus = list of day-dicts matching the develop-loop's schema
    {new_concepts, facts, probe_recall, probe_heldout, probe_yesno, probe_chain}.
    """
    cumulative = set()
    used = set()
    syllabus = []
    n_days = (len(concepts_ranked) + concepts_per_day - 1) // concepts_per_day
    for d in range(n_days):
        chunk = concepts_ranked[d * concepts_per_day:(d + 1) * concepts_per_day]
        if not chunk:
            break
        chunk_set = set(chunk)
        available = cumulative | chunk_set
        day_facts = []
        for i, (a, v, p) in enumerate(facts):
            if i in used:
                continue
            if a in available and v in available and p in available and (
                    a in chunk_set or v in chunk_set or p in chunk_set):
                day_facts.append((a, v, p))
                used.add(i)
                if len(day_facts) >= max_facts_per_day:
                    break
        probe_recall = [("patient", (a, v), p) for (a, v, p) in day_facts[:max_recall_probes]]
        probe_yesno = [(day_facts[0][0], day_facts[0][1], day_facts[0][2], "yes")] if day_facts else []
        syllabus.append({
            "new_concepts": list(chunk),
            "facts": day_facts,
            "probe_recall": probe_recall,
            "probe_heldout": [],
            "probe_yesno": probe_yesno,
            "probe_chain": [],
        })
        cumulative = available
    return syllabus


class CorpusGradedCurriculum(GradedCurriculum):
    """A GradedCurriculum whose syllabus is built from corpus concepts + corpus facts (deep-knowledge scale)."""

    def __init__(self, concepts_ranked, facts, *, concepts_per_day=24, max_facts_per_day=8):
        super().__init__(syllabus=build_corpus_syllabus(
            concepts_ranked, facts, concepts_per_day=concepts_per_day, max_facts_per_day=max_facts_per_day))


def _normalize_fact(f):
    """Accept a fact as [a,v,p] or {agent/subject, action/verb, patient/object} -> (a, v, p) or None."""
    if isinstance(f, dict):
        a = f.get("agent") or f.get("subject") or f.get("s")
        v = f.get("action") or f.get("verb") or f.get("v")
        p = f.get("patient") or f.get("object") or f.get("o")
    elif isinstance(f, (list, tuple)) and len(f) >= 3:
        a, v, p = f[0], f[1], f[2]
    else:
        return None
    return (a, v, p) if (a and v and p) else None


def load_concepts_and_facts(codes_npz, facts_json):
    """Concept vocab (high-freq-first, as saved by the curriculum trainer) + SVO facts (normalized)."""
    # allow_pickle: codes_npz is a TRUSTED, self-trained artifact (the curriculum trainer's --save-codes output);
    # its `vocab` is an object-dtype array of strings, which numpy can only read with allow_pickle=True. Not an
    # untrusted source.
    d = np.load(codes_npz, allow_pickle=True)
    concepts = [str(w) for w in d["vocab"]]
    raw = json.load(open(facts_json, encoding="utf-8"))
    items = raw["facts"] if isinstance(raw, dict) and "facts" in raw else raw
    facts = [t for t in (_normalize_fact(f) for f in items) if t is not None]
    return concepts, facts


def main():
    ap = argparse.ArgumentParser(description="Build + inspect a corpus-driven develop-loop syllabus.")
    ap.add_argument("--brain", default="bridges/firstchat/brain3000pos_w7000.npz_seed42.npz")
    ap.add_argument("--facts-json", default="research/findings/raw/_facts3000.json")
    ap.add_argument("--concepts-per-day", type=int, default=24)
    ap.add_argument("--max-facts-per-day", type=int, default=8)
    a = ap.parse_args()

    concepts, facts = load_concepts_and_facts(a.brain, a.facts_json)
    syll = build_corpus_syllabus(concepts, facts, concepts_per_day=a.concepts_per_day,
                                 max_facts_per_day=a.max_facts_per_day)
    n_assigned = sum(len(day["facts"]) for day in syll)
    print(f"[corpus-curriculum] {len(concepts)} concepts, {len(facts)} corpus facts")
    print(f"[corpus-curriculum] {len(syll)} days @ {a.concepts_per_day} concepts/day | "
          f"{n_assigned}/{len(facts)} facts assigned ({100*n_assigned/max(len(facts),1):.0f}% coverage)")
    print(f"[corpus-curriculum] sample fact: {facts[0] if facts else '(none)'}")
    for d in range(min(3, len(syll))):
        day = syll[d]
        print(f"  day {d}: +{len(day['new_concepts'])} concepts {day['new_concepts'][:6]}... | "
              f"{len(day['facts'])} facts e.g. {day['facts'][0] if day['facts'] else '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
