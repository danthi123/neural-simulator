"""Board #112 next rung 1: verify the `in_vocab_scope` lead-in-phrase loophole fix (2026-09-01).

WHAT THIS IS. `research/findings/2026-09-01-wkv-mouth-fact-grounding-lever.md` Part 2 found a real bug, logged
to `research/FAILURE_LOG.md` (2026-09-01): `webapp.wkv_mouth_generator.in_vocab_scope("tell me about " +
<anything>)` returned True even for total nonsense, because the lead-in words "tell"/"me"/"about" are in this
checkpoint's V=1000 vocabulary AND were not excluded from `min_content_hits` scoring -- so the fixed lead-in
phrase alone satisfied the gate's content-word floor regardless of the actual topic. `webapp/wkv_mouth_generator.py`
now excludes a `_LEADIN_WORDS` set (every word appearing across `webapp.open_ended_chat._LEADINS`'s phrases,
duplicated locally to avoid a reverse import) from `content_hits`, alongside the pre-existing `_FUNCTION_WORDS`
exclusion. This runner verifies that fix two ways:

(1) DIRECT ADVERSARIAL CATCH -- several distinct `_LEADINS` phrases, each followed by nonsense, must now read
    False (they read True before this fix).
(2) STRATIFIED, REAL-DATA catch-rate + no-regression -- reproducing Part 2's exact real-store sample (n=600
    `wikidata_core_15k` agents, seed 42, `"tell me about " + agent.replace('_',' ')`), each agent is bucketed
    by how many GENUINE in-vocab content words its OWN slug (independent of any lead-in) carries, then the OLD
    (pre-fix, reimplemented locally byte-for-byte from the code this session found) and NEW gates are compared
    within each bucket:
      * bucket 0 (topic itself contributes ZERO content words -- the pure loophole population, previously
        passing ONLY because of the lead-in): OLD pass-rate should reproduce Part 2's ~68% overall figure's
        loophole-driven share; NEW pass-rate should collapse toward 0% -- this IS the catch rate.
      * bucket >=2 (topic ALONE already meets `min_content_hits=2`, genuinely in-scope by the gate's own
        design, independent of the lead-in): NEW pass-rate should stay ~100% -- this IS the no-regression
        claim, measured on real store data, not a synthetic example.
      * bucket 1 (topic carries exactly one real content word) is reported as an HONEST RESIDUAL, not folded
        into either claim: a single-content-word topic structurally cannot reach `min_content_hits=2` from the
        topic alone, so it now (correctly) fails post-fix even though the one word IS genuinely in-vocab. This
        was already true of the ORIGINAL (pre-loophole-bug) design intent -- the 2-hit floor was never
        satisfiable by a single word without the lead-in propping it up -- so this is not a regression against
        correct behavior, only against the bug.

MEMORY DISCIPLINE. Reads ONLY `facts.json` (~2 MB) via the SAME `_bundle_dir()` convention
`_wkv_mouth_fact_grounding_derisk.py` uses; no `SimulationBridge`/`ShardedPhasorStore`, no GPU. `_get_readout`
loads one ~1.4 MB checkpoint .npz. CPU/numpy only.

Run: `SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_invocab_scope_leadin_fix_verify`
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

from webapp import wkv_mouth_generator as WKV  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# The SAME `_LEADINS` list `webapp/open_ended_chat.py` uses to strip a lead-in before topic extraction --
# duplicated here (read directly from that module's source at the time this runner was written) purely to
# construct adversarial test PHRASES; this runner does not import `open_ended_chat` (keeps the same
# import-independence `wkv_mouth_generator.py` itself maintains).
_LEADIN_PHRASES = [
    "what can you tell me about", "what do you know about", "what do you think about",
    "can you tell me about", "do you know anything about", "do you know about", "tell me about",
    "who is", "who was", "tell me", "describe", "explain",
]


def _bundle_dir() -> str | None:
    roots = []
    env_root = os.environ.get("BRAIN_DATA_ROOT", "").strip()
    if env_root:
        roots.append(env_root)
    roots.append(str(REPO_ROOT.parent / "sim-data"))
    roots.append(str(Path.home() / "Projects" / "sim-data"))
    for r in roots:
        d = str(Path(r) / "knowledge_bundles" / "wikidata_core_15k")
        if os.path.isdir(d):
            return d
    return None


def _old_in_vocab_scope(text: str, vocab: set, min_frac=0.6, min_hits=2, min_content_hits=2) -> bool:
    """Byte-for-byte reimplementation of `in_vocab_scope` AS IT EXISTED before this session's fix (content_hits
    excludes ONLY `_FUNCTION_WORDS`, not the lead-in) -- used ONLY to reproduce the buggy baseline for
    comparison, never imported from the fixed module."""
    words = [w.lower() for w in WKV._WORD_RE.findall(text or "")]
    if not words:
        return False
    hits = [w for w in words if w in vocab]
    content_hits = [w for w in hits if w not in WKV._FUNCTION_WORDS]
    return (len(hits) >= min_hits and (len(hits) / len(words)) >= min_frac
            and len(content_hits) >= min_content_hits)


def _topic_content_word_count(agent_slug: str, vocab: set) -> int:
    """How many genuine in-vocab content words the BARE topic (no lead-in at all) carries -- the independent
    ground-truth bucketing key, computed the same way `in_vocab_scope`'s content_hits works, minus any lead-in
    concern (a bare entity slug is never itself a lead-in phrase)."""
    words = [w.lower() for w in WKV._WORD_RE.findall(agent_slug.replace("_", " "))]
    return sum(1 for w in words if w in vocab and w not in WKV._FUNCTION_WORDS and w not in WKV._LEADIN_WORDS)


def main(seed: int = 42) -> dict:
    out: dict = {"runner": "_wkv_invocab_scope_leadin_fix_verify", "seed": seed}
    _, vocab, _ = WKV._get_readout(seed)
    out["checkpoint_vocab_size"] = len(vocab)

    # ── Part A: direct adversarial catch, several distinct lead-in phrases + nonsense tails ───────────────────
    nonsense_tails = ["zzznonsenseword qqqgibberish", "xkcdplaceholder blorptastic", "wibbleflorp yazzledeen"]
    catches = []
    for phrase in _LEADIN_PHRASES:
        for tail in nonsense_tails:
            msg = f"{phrase} {tail}"
            old = _old_in_vocab_scope(msg, vocab)
            new = WKV.in_vocab_scope(msg, seed=seed)
            catches.append({"msg": msg, "old_pass": old, "new_pass": new, "caught": bool(old and not new)})
    n_old_false_positive = sum(1 for c in catches if c["old_pass"])
    n_now_caught = sum(1 for c in catches if c["caught"])
    n_still_leaking = sum(1 for c in catches if c["old_pass"] and c["new_pass"])
    out["part_a_direct_adversarial_catch"] = {
        "n_cases": len(catches),
        "n_old_false_positive": n_old_false_positive,
        "n_now_caught": n_now_caught,
        "n_still_leaking": n_still_leaking,
        "examples": catches[:6],
        "all_cases": catches,
    }

    # ── Part B: genuine multi-content-word sentences must still pass, with or without a recognized lead-in ────
    # Real TinyStories-register sentences (adapted from the existing `_wkv_learned_vs_native_head_ab.py` PROMPTS
    # battery, itself independently verified in-vocab) -- some carry a recognized lead-in prefix, some don't.
    genuine_sentences = [
        "once upon a time there was a little boy named tim who had a dog",
        "tell me a story about a happy dog and his best friend",
        "lily and her mom went to the park to play with a ball",
        "tell me about the big dog and the little cat and their ball",
        "what do you know about the happy boy and his best friend and their toy",
    ]
    genuine_results = []
    for s in genuine_sentences:
        old = _old_in_vocab_scope(s, vocab)
        new = WKV.in_vocab_scope(s, seed=seed)
        genuine_results.append({"msg": s, "old_pass": old, "new_pass": new, "regressed": bool(old and not new)})
    n_genuine_pass_before = sum(1 for r in genuine_results if r["old_pass"])
    n_genuine_pass_after = sum(1 for r in genuine_results if r["new_pass"])
    n_genuine_regressed = sum(1 for r in genuine_results if r["regressed"])
    out["part_b_genuine_content_no_regression"] = {
        "n_cases": len(genuine_results), "n_pass_before": n_genuine_pass_before,
        "n_pass_after": n_genuine_pass_after, "n_regressed": n_genuine_regressed,
        "cases": genuine_results,
    }

    # ── Part C: stratified real-store measurement (Part 2's exact sample), bucketed by genuine topic content ──
    bundle = _bundle_dir()
    out["bundle_dir"] = bundle
    part_c = None
    if bundle is not None:
        facts_path = Path(bundle) / "facts.json"
        raw = json.loads(facts_path.read_text(encoding="utf-8"))
        affirm = [r["fact"] for r in raw if r.get("fact", {}).get("polarity", "AFFIRM") == "AFFIRM"]
        by_agent: dict = {}
        for f in affirm:
            by_agent.setdefault(f["agent"], []).append(f["agent"])
        agents = sorted(by_agent.keys())
        rng = random.Random(seed)
        sample_agents = rng.sample(agents, min(600, len(agents)))

        buckets = {0: [], 1: [], "2plus": []}
        for a in sample_agents:
            msg = "tell me about " + a.replace("_", " ")
            old = _old_in_vocab_scope(msg, vocab)
            new = WKV.in_vocab_scope(msg, seed=seed)
            n_topic_content = _topic_content_word_count(a, vocab)
            key = 0 if n_topic_content == 0 else (1 if n_topic_content == 1 else "2plus")
            buckets[key].append({"agent": a, "old_pass": old, "new_pass": new})

        def _bucket_stats(items):
            n = len(items)
            if n == 0:
                return {"n": 0, "old_pass_frac": None, "new_pass_frac": None}
            return {
                "n": n,
                "old_pass": sum(1 for i in items if i["old_pass"]),
                "old_pass_frac": round(sum(1 for i in items if i["old_pass"]) / n, 4),
                "new_pass": sum(1 for i in items if i["new_pass"]),
                "new_pass_frac": round(sum(1 for i in items if i["new_pass"]) / n, 4),
            }

        stats0 = _bucket_stats(buckets[0])
        stats1 = _bucket_stats(buckets[1])
        stats2 = _bucket_stats(buckets["2plus"])
        overall_old = sum(1 for a in sample_agents
                           for _ in [0] if _old_in_vocab_scope("tell me about " + a.replace("_", " "), vocab))
        overall_new = sum(1 for a in sample_agents
                           for _ in [0] if WKV.in_vocab_scope("tell me about " + a.replace("_", " "), seed=seed))
        part_c = {
            "n_sampled_agents": len(sample_agents),
            "overall_old_pass_frac": round(overall_old / len(sample_agents), 4),
            "overall_new_pass_frac": round(overall_new / len(sample_agents), 4),
            "bucket_0_zero_genuine_content": stats0,
            "bucket_1_one_genuine_content_word": stats1,
            "bucket_2plus_genuine_content": stats2,
            "note": ("bucket_0 = topic's own slug has ZERO in-vocab content words (independent of any lead-in) "
                     "-- the pure loophole population; its old_pass_frac vs new_pass_frac IS the catch rate. "
                     "bucket_2plus = topic alone already meets min_content_hits=2 -- its new_pass_frac staying "
                     "~1.0 IS the no-regression claim, on real store data. bucket_1 is an honest residual: a "
                     "single real content word cannot reach the 2-hit floor alone, so it now correctly fails "
                     "post-fix (was only passing via the lead-in before)."),
        }
        out["part_c_stratified_real_store"] = part_c
    else:
        out["part_c_stratified_real_store"] = {"skipped": "no data lake (sim-data/knowledge_bundles/wikidata_core_15k not found)"}

    # ── verdict ──────────────────────────────────────────────────────────────────────────────────────────────
    v = Verdict("in_vocab_scope lead-in loophole fix")
    v.require("direct adversarial: every old false-positive is now caught",
              n_still_leaking, expect=0)
    v.require("no regression: every genuine multi-content-word sentence still passes",
              n_genuine_regressed, expect=0)
    v.require("genuine sentences still pass after the fix (same count as before)",
              n_genuine_pass_after, expect=n_genuine_pass_before)
    if part_c is not None:
        v.require("bucket_0 (zero genuine topic content) collapses toward 0 after the fix",
                  part_c["bucket_0_zero_genuine_content"]["new_pass_frac"], expect=lambda x: x is not None and x < 0.05)
        v.require("bucket_2plus (genuine 2+ content words) stays ~fully passing after the fix",
                  part_c["bucket_2plus_genuine_content"]["new_pass_frac"], expect=lambda x: x is not None and x >= 0.95)
        v.require("overall pass rate on the real-store sample drops (loophole was inflating it)",
                  part_c["overall_new_pass_frac"], expect=lambda x: x < part_c["overall_old_pass_frac"])
    verdict = v.decide(go=(n_still_leaking == 0 and n_genuine_regressed == 0))
    out["verdict"] = verdict
    out["verdict_preconditions"] = v.to_dict()

    print(json.dumps({k: out[k] for k in out if k not in ("part_a_direct_adversarial_catch", "part_c_stratified_real_store")},
                      indent=2, default=str))
    if part_c is not None:
        print("\n--- part C stratified (real store, n=%d) ---" % part_c["n_sampled_agents"])
        print(f"  overall: old={part_c['overall_old_pass_frac']}  new={part_c['overall_new_pass_frac']}")
        for k, label in (("bucket_0_zero_genuine_content", "bucket 0 (zero content)"),
                         ("bucket_1_one_genuine_content_word", "bucket 1 (one content word)"),
                         ("bucket_2plus_genuine_content", "bucket 2+ (2+ content words)")):
            s = part_c[k]
            print(f"  {label}: n={s['n']}  old_pass_frac={s.get('old_pass_frac')}  new_pass_frac={s.get('new_pass_frac')}")
    print(f"\nVERDICT: {verdict}")
    return out


if __name__ == "__main__":
    result = main(seed=42)
    out_path = REPO_ROOT / "research/findings/raw/_wkv_invocab_scope_leadin_fix_verify.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")
