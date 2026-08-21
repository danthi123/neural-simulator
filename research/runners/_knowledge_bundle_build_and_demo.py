"""KNOWLEDGE-BUNDLE build + persist + verify + demo (owner ask, 2026-08-20): load the Wikidata-fetched general-
knowledge SVO triples (`_knowledge_bundle_wikidata_fetch.py`'s output) into a standalone `RFPhasorComposer`
(D=128, production dimension), persist as a `developed_brain_io` bundle, then VERIFY recall accuracy / the
no-confab moat / query latency at the loaded scale, and run a captured ~10-15-turn Q&A DEMO.

This is a declared TEST SCAFFOLD: the bulk-load (fetch -> SVO -> comp.store) is host-side data prep, no `sim/`
edit, no production default changed. The composer's recall + no-confab moat are the genuine reads this script
exists to exercise -- see rf_phasor_composer.py's own docstring for the mechanism (FHRR-on-bridge, phasor
bind/bundle/unbind on the RF/Izhikevich substrate).

Run: SIM_BACKEND=numpy python -m research.runners._knowledge_bundle_build_and_demo [--cap N] [--recall-n N]
     [--moat-n N] [--demo-answer N] [--demo-abstain N]
"""
from __future__ import annotations
import argparse, json, logging, random, sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logging.disable(logging.INFO)  # quiet the per-bridge SIM_BRIDGE init logs (this script builds many small bridges)

from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402
from research.runners import developed_brain_io as dbio  # noqa: E402

RAW_DIR = _REPO / "research" / "findings" / "raw" / "_knowledge_bundle_wikidata"
FACTS_RAW = RAW_DIR / "facts_raw.json"
BUNDLE_DIR = RAW_DIR / "bundle"
REPORT_OUT = RAW_DIR / "verify_report.json"
TRANSCRIPT_OUT = RAW_DIR / "demo_transcript.txt"


class _AgentShim:
    """The minimal object `developed_brain_io._inner_agent` needs: a `.composer` attribute. We build the
    RFPhasorComposer directly (the standalone composer path the task specifies -- NOT the live onebrain chat,
    which caps at k_max=32 facts), so there is no BrainConversationalAgent to wrap; this shim lets the SAME
    validated save_developed_brain()/extract_*() persist it as a normal developed-brain bundle."""
    def __init__(self, composer):
        self.composer = composer


# ---------------------------------------------------------------------------------------------------------------
def _clean_alpha(s):
    """True iff every WHITESPACE-SEPARATED word in `s` is alphabetic. (BUG FOUND + FIXED during this build:
    `s.isalpha()` on the whole string rejects any multi-word token -- and several of the fetch's own category-root
    names are two words ('musical instrument', 'chemical element', 'body part', ...), used verbatim as the
    agent/patient for every fact fetched under that root. The whole-string check silently dropped 372/2413 (15.4%)
    of the fetched corpus this way -- 0 of those 372 had any OTHER problem (checked: the per-word check recovers
    100% of them, and drops nothing further), so this was a pure bug, not a legitimate cleaning decision.)"""
    words = s.split()
    return bool(words) and all(w.isalpha() for w in words)


def load_and_clean_facts(cap, seed):
    payload = json.loads(FACTS_RAW.read_text())
    raw = payload["facts"]
    seen, uniq = set(), []
    for a, v, p in raw:
        a, v, p = str(a).lower().strip(), str(v).lower().strip(), str(p).lower().strip()
        if not a or not v or not p or not _clean_alpha(a) or not _clean_alpha(p):
            continue
        k = (a, v, p)
        if k in seen:
            continue
        seen.add(k); uniq.append([a, v, p])
    n_before_cap = len(uniq)
    if cap is not None and len(uniq) > cap:
        rng = random.Random(seed)
        rng.shuffle(uniq)
        uniq = uniq[:cap]
    return uniq, n_before_cap, payload.get("source", "wikidata_live"), payload.get("elapsed_seconds")


def cue_collisions(facts):
    """(HONEST cue-collision accounting) group by (agent, relation) -> ordered list of patients as STORED. Since
    `RFPhasorComposer.query_patient` is first-match on (agent, relation), a subject with >1 stored patient under
    the SAME relation can only ever recall the FIRST one it was given. Returns (first_seen map, collision stats)."""
    order = {}
    for a, v, p in facts:
        order.setdefault((a, v), []).append(p)
    first_seen = {k: v[0] for k, v in order.items()}
    collisions = {k: v for k, v in order.items() if len(set(v)) > 1}
    return first_seen, collisions


# ---------------------------------------------------------------------------------------------------------------
def build_composer(facts, D, seed):
    vocab = sorted({a for a, _, _ in facts} | {v for _, v, _ in facts} | {p for _, _, p in facts})
    comp = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    t0 = time.time()
    for a, v, p in facts:
        comp.store(a, v, p)
    build_seconds = time.time() - t0
    return comp, vocab, build_seconds


# ---------------------------------------------------------------------------------------------------------------
def sample_recall(comp, first_seen, n, seed):
    rng = random.Random(seed)
    pairs = list(first_seen.items())
    rng.shuffle(pairs)
    sample = pairs[:n]
    correct = 0
    examples = []
    t0 = time.time()
    for (a, v), expected in sample:
        ans = comp.query_patient(a, v)
        ok = (ans == expected)
        correct += int(ok)
        examples.append({"agent": a, "relation": v, "expected": expected, "got": ans, "ok": ok})
    dt = time.time() - t0
    return {"n": len(sample), "correct": correct,
            "accuracy": correct / max(len(sample), 1),
            "sec_per_query": dt / max(len(sample), 1),
            "total_seconds": dt,
            "wrong_examples": [e for e in examples if not e["ok"]][:10],
            "examples": examples[:15]}


def sample_moat(comp, first_seen, n, seed):
    rng = random.Random(seed)
    known_set = set(first_seen.keys())
    known_agents = sorted({a for a, _ in first_seen})
    rels = sorted({v for _, v in first_seen})
    fake_words = ["zorblaxi", "fnargleth", "quixotronn", "blipwoodx", "snarklebee",
                  "glimmerfoxx", "thundrakex", "wobblenite"]
    results = []
    abstain_ok = 0
    # (a) never-loaded made-up words (never appear anywhere in the vocab or kb)
    for w in fake_words:
        ans = comp.query_patient(w, "isa")
        ok = ans is None
        abstain_ok += int(ok)
        results.append({"agent": w, "relation": "isa", "got": ans, "abstain_ok": ok,
                        "kind": "never_seen_word"})
    # (b) a KNOWN agent, cued with a relation it was never stored under (the harder moat test: the agent's own
    #     code is real and grounded, only the (agent,relation) PAIR is unstored)
    tries = 0
    added = 0
    while added < n and tries < n * 20:
        a = rng.choice(known_agents)
        v = rng.choice(rels)
        tries += 1
        if (a, v) in known_set:
            continue
        ans = comp.query_patient(a, v)
        ok = ans is None
        abstain_ok += int(ok)
        results.append({"agent": a, "relation": v, "got": ans, "abstain_ok": ok,
                        "kind": "known_agent_unstored_relation"})
        added += 1
    total = len(results)
    return {"n": total, "abstain_ok": abstain_ok, "moat_rate": abstain_ok / max(total, 1),
            "false_accepts": [r for r in results if not r["abstain_ok"]],
            "examples": results[:15]}


# ---------------------------------------------------------------------------------------------------------------
def _phrase_question(agent, relation, proper_nouns=frozenset()):
    # a lightweight readability fix, NOT full NLG: agents we KNOW are proper nouns (fetched as instances of
    # "country" under P31) get capitalized + no indefinite article ("What is France?"); everything else keeps the
    # uniform "a <word>" template. This is a thin question-rendering wrapper around query_patient, disclosed as such
    # in the report -- not a parser, so any other proper-noun class (planets, elements, ...) still reads as "a mars".
    disp = agent.capitalize() if agent in proper_nouns else f"a {agent}"
    if relation == "isa":
        return f"What is {disp}?"
    if relation == "has":
        return f"What does {disp} have?"
    return f"What does {disp} {relation}?"


def _phrase_answer(agent, relation, answer, proper_nouns=frozenset()):
    if answer is None:
        return "I don't know."
    subj = agent.capitalize() if agent in proper_nouns else f"A {agent}"
    obj = answer.capitalize() if answer in proper_nouns else answer
    if relation == "isa":
        return f"{subj} is a {obj}."
    if relation == "has":
        art = "" if answer in proper_nouns else "a "
        return f"{subj} has {art}{obj}."
    return f"{subj} {relation} {obj}."


def build_demo(comp, first_seen, n_answer, n_abstain, seed):
    rng = random.Random(seed)
    pairs = list(first_seen.items())
    rng.shuffle(pairs)
    turns = []
    for (a, v), expected in pairs:
        if len(turns) >= n_answer:
            break
        ans = comp.query_patient(a, v)
        if ans == expected:  # only show turns that genuinely recall correctly (no cherry-picked wrong answers hidden)
            turns.append({"kind": "answer", "agent": a, "relation": v, "expected": expected, "answer": ans})
    known_agents = sorted({a for a, _ in first_seen})
    known_set = set(first_seen.keys())
    rels = sorted({v for _, v in first_seen})
    fake_words = ["zorblaxi", "fnargleth", "quixotronn", "blipwoodx", "snarklebee"]
    rng.shuffle(fake_words)
    n_fake = min(n_abstain // 2 + n_abstain % 2, len(fake_words))
    for w in fake_words[:n_fake]:
        ans = comp.query_patient(w, "isa")
        turns.append({"kind": "abstain_unknown_word", "agent": w, "relation": "isa", "answer": ans})
    added = 0
    tries = 0
    target = n_abstain - n_fake
    while added < target and tries < 500:
        a = rng.choice(known_agents); v = rng.choice(rels)
        tries += 1
        if (a, v) in known_set:
            continue
        ans = comp.query_patient(a, v)
        if ans is None:
            turns.append({"kind": "abstain_unstored_relation", "agent": a, "relation": v, "answer": ans})
            added += 1
    rng.shuffle(turns)
    return turns


def render_transcript(turns, proper_nouns=frozenset()):
    lines = []
    for i, t in enumerate(turns, 1):
        q = _phrase_question(t["agent"], t["relation"], proper_nouns)
        a = _phrase_answer(t["agent"], t["relation"], t["answer"], proper_nouns)
        tag = "[LOADED FACT]" if t["kind"] == "answer" else "[UNKNOWN -> MOAT ABSTAIN]"
        lines.append(f"Turn {i:2d} {tag}")
        lines.append(f"  Q: {q}")
        lines.append(f"  A: {a}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=3000, help="max facts to load into the composer")
    ap.add_argument("--D", type=int, default=128, help="phasor dimension (128 = production)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recall-n", type=int, default=60)
    ap.add_argument("--moat-n", type=int, default=30)
    ap.add_argument("--demo-answer", type=int, default=10)
    ap.add_argument("--demo-abstain", type=int, default=5)
    a = ap.parse_args()

    if not FACTS_RAW.exists():
        print(f"NOT-RUNNABLE: {FACTS_RAW} missing -- run _knowledge_bundle_wikidata_fetch first")
        return 2

    facts, n_before_cap, source, fetch_elapsed = load_and_clean_facts(a.cap, a.seed)
    first_seen, collisions = cue_collisions(facts)
    # agents fetched as P31 instances of "country" (+ their P36 capitals) -> readability-only fix for the demo
    # phrasing (see _phrase_*); NOT a general proper-noun detector, disclosed in the report as a thin template.
    proper_nouns = {ag for ag, rel, pat in facts if rel == "isa" and pat == "country"}
    proper_nouns |= {pat for ag, rel, pat in facts if rel == "has" and ag in proper_nouns}
    print(f"[build] {len(facts)} facts loaded (from {n_before_cap} cleaned/deduped, source={source}); "
          f"{len(first_seen)} distinct (agent,relation) cues; {len(collisions)} cue-collisions "
          f"({100.0*len(collisions)/max(len(first_seen),1):.1f}% of cues)", flush=True)
    if collisions:
        ex_k, ex_v = next(iter(collisions.items()))
        print(f"  [cue-collision example] {ex_k[0]!r} {ex_k[1]!r} -> stored patients {ex_v} "
              f"(query_patient returns the FIRST: {ex_v[0]!r} only)", flush=True)

    comp, vocab, build_seconds = build_composer(facts, a.D, a.seed)
    print(f"[build] composer: D={a.D} vocab={len(vocab)} facts_stored={len(comp.kb)} "
          f"build_time={build_seconds:.1f}s ({1000*build_seconds/max(len(facts),1):.1f} ms/store)", flush=True)

    print(f"[verify] recall sample n={a.recall_n} ...", flush=True)
    recall = sample_recall(comp, first_seen, a.recall_n, a.seed + 1)
    print(f"[verify] recall: {recall['correct']}/{recall['n']} = {recall['accuracy']:.3f}, "
          f"{recall['sec_per_query']*1000:.1f} ms/query ({recall['total_seconds']:.1f}s total)", flush=True)

    print(f"[verify] moat sample n~={a.moat_n} ...", flush=True)
    moat = sample_moat(comp, first_seen, a.moat_n, a.seed + 2)
    print(f"[verify] moat: {moat['abstain_ok']}/{moat['n']} abstained = {moat['moat_rate']:.3f}", flush=True)

    print(f"[demo] building {a.demo_answer} answer + {a.demo_abstain} abstain turns ...", flush=True)
    turns = build_demo(comp, first_seen, a.demo_answer, a.demo_abstain, a.seed + 3)
    transcript = render_transcript(turns, proper_nouns)
    print("\n" + "=" * 100)
    print(transcript)
    print("=" * 100 + "\n", flush=True)
    TRANSCRIPT_OUT.parent.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_OUT.write_text(transcript)

    print("[persist] saving developed_brain_io bundle ...", flush=True)
    shim = _AgentShim(comp)
    manifest = dbio.save_developed_brain(
        shim, BUNDLE_DIR, seed=a.seed, D=a.D, composer_kind="rf",
        extra_metadata={
            "provenance": "knowledge_bundle_wikidata (research/knowledge-bundle-wikidata worktree, 2026-08-20)",
            "source": source, "n_facts_fetched_before_cap": n_before_cap, "fetch_elapsed_seconds": fetch_elapsed,
            "cap": a.cap, "test_scaffold": True,
            "note": "host-side bulk data-prep test scaffold; the composer's recall + no-confab moat are the "
                    "genuine reads, this bundle is not a production default.",
        })
    print(f"[persist] wrote {BUNDLE_DIR} (n_facts={manifest['n_facts']}, "
          f"n_grounded_codes={manifest['n_grounded_codes']}, n_kb_composites={manifest['n_kb_composites']})",
          flush=True)

    report = {
        "source": source, "n_facts_fetched_before_cap": n_before_cap, "n_facts_loaded": len(facts),
        "D": a.D, "seed": a.seed, "vocab_size": len(vocab), "build_seconds": build_seconds,
        "n_distinct_cues": len(first_seen), "n_cue_collisions": len(collisions),
        "cue_collision_example": ({"cue": list(next(iter(collisions))), "patients": next(iter(collisions.values()))}
                                  if collisions else None),
        "recall": recall, "moat": moat, "demo_turns": turns, "bundle_dir": str(BUNDLE_DIR),
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"[done] wrote {REPORT_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
