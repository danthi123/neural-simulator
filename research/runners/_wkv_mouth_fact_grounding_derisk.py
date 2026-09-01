"""Board #112 "clean unlock": can a KNOWN-topic WKV-mouth reply SURFACE the brain's real recalled fact?

WHAT THIS IS. `research/findings/2026-09-01-open-ended-bundle-moat-safety-soak-fabrication-delta.md` (the moat
soak) named the next action: the from-scratch WKV spiking mouth's V=1000 TinyStories vocabulary structurally
cannot name a Wikidata entity's real facts, so flipping `BRAIN_OPEN_ENDED` trades grounded exact recall for
fact-thin free-gen on known topics. This runner (1) MAPS the mouth's real expressive ceiling against the real
shipped `wikidata_core_15k` store (not a guess), (2) exercises the ONE concrete grounding lever this arc built
(`webapp.wkv_mouth_generator.fact_grounding_ids` / `_apply_fact_boost`, wired into `webapp.open_ended_chat`
behind `BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND`, default OFF) on REAL recalled facts, and (3) reports both the
genuine capability AND the honest residual precisely, per the task's own instruction not to force a GO.

MEMORY DISCIPLINE. Reads ONLY `facts.json` from the shipped bundle (~2 MB; the SAME low-memory retrieval source
`webapp/open_ended_chat.py::build_index` uses) -- never the 15 MB `composites.npz` phasor store, never a full
`SimulationBridge`/15k-LTM brain build. The only spiking construction here is `FewSpikeWordRead`'s own ~512-
neuron few-spike Izhikevich bank (topk=64 * pop=8), the SAME small numpy-only bank the already-GO-verified
`_wkv_mouth_open_ended_wiring_verify.py` used. CPU/numpy only, no GPU.

Run: `SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_fact_grounding_derisk`
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402

from webapp import wkv_mouth_generator as WKV  # noqa: E402
from webapp import open_ended_chat as OE  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── locate the shipped bundle (mirrors webapp/server.py::_default_ltm_bundle_dir's candidate-root order) ─────────
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


FUNC = WKV._FUNCTION_WORDS


def _content_hits(s: str, vocab: set) -> list:
    return [w for w in WKV._WORD_RE.findall(s.lower()) if w in vocab and w not in FUNC]


def main() -> dict:
    out: dict = {"runner": "_wkv_mouth_fact_grounding_derisk"}
    bundle = _bundle_dir()
    out["bundle_dir"] = bundle
    if bundle is None:
        out["skipped"] = "no data lake (sim-data/knowledge_bundles/wikidata_core_15k not found)"
        print(json.dumps(out, indent=2))
        return out

    facts_path = Path(bundle) / "facts.json"
    raw = json.loads(facts_path.read_text(encoding="utf-8"))
    affirm = [r["fact"] for r in raw if r.get("fact", {}).get("polarity", "AFFIRM") == "AFFIRM"]
    out["n_facts_total"] = len(raw)
    out["n_facts_affirm"] = len(affirm)

    _, vocab, word_to_id = WKV._get_readout(42)
    out["checkpoint_vocab_size"] = len(vocab)

    # ── PART 1: corpus-level content-word coverage ceiling (per-fact, over ALL AFFIRM facts) ─────────────────────
    n = len(affirm)
    patient_content_hit = 0
    any_field_content_hit = 0
    for f in affirm:
        p_hits = _content_hits(f["patient"], vocab)
        if p_hits:
            patient_content_hit += 1
        if p_hits or _content_hits(f["action"], vocab) or _content_hits(f["agent"], vocab):
            any_field_content_hit += 1
    out["part1_corpus_coverage_ceiling"] = {
        "n_affirm_facts": n,
        "patient_content_word_hit": patient_content_hit,
        "patient_content_word_hit_frac": round(patient_content_hit / n, 5),
        "any_field_content_word_hit": any_field_content_hit,
        "any_field_content_word_hit_frac": round(any_field_content_hit / n, 5),
        "note": ("This checkpoint's V=1000 vocabulary is word-level and closed (no subword fallback): a fact "
                 "token with no trained embedding CANNOT be produced by this decoder at all, by construction. "
                 "The majority (~74%) of facts have zero content-word overlap -- the structural wall the task "
                 "named, now precisely quantified rather than assumed."),
    }

    # ── PART 2: the in_vocab_scope gate's OWN engagement rate on REAL known-topic queries, "tell me about X" ────
    # (the exact phrasing the moat-soak's own battery used) -- and a real bug this measurement found: the
    # lead-in phrase alone ("tell", "me", "about" are in-vocab and NOT in _FUNCTION_WORDS) can satisfy
    # `min_content_hits` on its own, so `in_vocab_scope` can pass regardless of whether the TOPIC itself is
    # remotely TinyStories-domain. Logged to research/FAILURE_LOG.md (see the finding doc).
    by_agent: dict = {}
    for f in affirm:
        by_agent.setdefault(f["agent"], []).append((f["agent"], f["action"], f["patient"]))
    agents = sorted(by_agent.keys())
    rng = random.Random(42)
    sample_agents = rng.sample(agents, min(600, len(agents)))

    scope_pass = 0
    scope_and_grounding = 0
    grounded_examples = []
    for a in sample_agents:
        msg = "tell me about " + a.replace("_", " ")
        if WKV.in_vocab_scope(msg, seed=42):
            scope_pass += 1
            triples = by_agent[a]
            ids = WKV.fact_grounding_ids(triples, seed=42)
            if ids:
                scope_and_grounding += 1
                if len(grounded_examples) < 40:
                    grounded_examples.append({"agent": a, "msg": msg, "triples": triples[:3], "fact_ids": ids})

    lead_in_alone_scope = WKV.in_vocab_scope("tell me about zzznonsenseword qqqgibberish", seed=42)
    out["part2_live_engagement"] = {
        "n_sampled_agents": len(sample_agents),
        "in_vocab_scope_pass": scope_pass,
        "in_vocab_scope_pass_frac": round(scope_pass / len(sample_agents), 4),
        "scope_pass_AND_has_fact_content_overlap": scope_and_grounding,
        "scope_pass_AND_has_fact_content_overlap_frac_of_sample": round(scope_and_grounding / len(sample_agents), 4),
        "conditional_grounding_rate_given_scope_pass": round(scope_and_grounding / max(1, scope_pass), 4),
        "lead_in_phrase_alone_passes_scope_on_nonsense_topic": lead_in_alone_scope,
        "note": ("`in_vocab_scope('tell me about <nonsense>')` == True: the lead-in phrase alone ('tell'/'me'/"
                 "'about', all in-vocab and NOT in _FUNCTION_WORDS) can satisfy min_content_hits=2 without the "
                 "topic itself contributing anything -- a real, previously-undiscovered gap in the ALREADY-"
                 "default-ON (2026-08-30) scope gate, found by this measurement, logged to FAILURE_LOG.md, NOT "
                 "fixed here (out of this task's scope; BRAIN_OPEN_ENDED itself stays default-OFF so this is not "
                 "live in production). It means the WKV mouth reaches real Wikidata topics far MORE than "
                 "Part 1's raw vocabulary ceiling alone would predict when phrased with a content-bearing "
                 "lead-in -- which makes fact-grounding MORE, not less, load-bearing: today those engaged turns "
                 "get zero grounding at all."),
    }

    # ── PART 3: the concrete lever -- before/after generation on REAL recalled facts, REAL in-vocab-scoped msgs ──
    demo_prompts = sorted(
        grounded_examples,
        key=lambda e: (len(set(e["fact_ids"])), e["agent"]),
        reverse=True,
    )
    # de-duplicate by first-fact action so the demo spans different relation types, not 8 near-identical ones.
    seen_actions = set()
    chosen = []
    for e in demo_prompts:
        act = e["triples"][0][1]
        if act in seen_actions:
            continue
        seen_actions.add(act)
        chosen.append(e)
        if len(chosen) >= 8:
            break

    part3 = []
    for e in chosen:
        agent, msg, triples, fact_ids = e["agent"], e["msg"], e["triples"], e["fact_ids"]
        ro, _v, _w2id = WKV._get_readout(42)
        boosted_words = [ro.words[i] for i in fact_ids]

        base_text, base_secs = WKV.generate(msg, seed=42, max_new_tokens=40,
                                            repetition_penalty=1.3, no_repeat_ngram_size=3, facts=None)
        boost_text, boost_secs = WKV.generate(msg, seed=42, max_new_tokens=40,
                                              repetition_penalty=1.3, no_repeat_ngram_size=3,
                                              facts=triples, fact_boost=6.0)
        base_words = set(base_text.split())
        boost_words = set(boost_text.split())
        surfaced_base = sorted(w for w in boosted_words if w in base_words)
        surfaced_boost = sorted(w for w in boosted_words if w in boost_words)
        newly_surfaced = sorted(set(surfaced_boost) - set(surfaced_base))

        base_filtered = OE.post_filter(base_text, agent, True, triples)
        boost_filtered = OE.post_filter(boost_text, agent, True, triples)

        part3.append({
            "agent": agent, "msg": msg, "triples": triples,
            "boosted_words": boosted_words,
            "baseline_raw": base_text, "boosted_raw": boost_text,
            "baseline_surfaces_fact_word": bool(surfaced_base),
            "boosted_surfaces_fact_word": bool(surfaced_boost),
            "newly_surfaced_by_boost": newly_surfaced,
            "baseline_filtered": base_filtered, "boosted_filtered": boost_filtered,
            "boosted_filtered_nonempty": bool(boost_filtered.strip()),
            "boosted_word_in_filtered_output": any(w in boost_filtered.split() for w in boosted_words),
        })

    n_demo = len(part3)
    n_base_surfaced = sum(1 for r in part3 if r["baseline_surfaces_fact_word"])
    n_boost_surfaced = sum(1 for r in part3 if r["boosted_surfaces_fact_word"])
    n_newly_surfaced = sum(1 for r in part3 if r["newly_surfaced_by_boost"])
    n_survives_filter = sum(1 for r in part3 if r["boosted_word_in_filtered_output"])
    out["part3_before_after_demo"] = {
        "n_examples": n_demo,
        "baseline_surfaced_fact_word": n_base_surfaced,
        "boosted_surfaced_fact_word": n_boost_surfaced,
        "boost_newly_surfaced_a_word_baseline_missed": n_newly_surfaced,
        "boosted_fact_word_survives_honesty_postfilter": n_survives_filter,
        "examples": part3,
    }

    # ── PART 4: mechanism no-op / determinism checks (the "byte-identical when off" property) ─────────────────────
    lg = np.arange(10, dtype=np.float64)
    lg_orig = lg
    same1 = WKV._apply_fact_boost(lg, [], 6.0) is lg_orig
    same2 = WKV._apply_fact_boost(lg, None, 6.0) is lg_orig
    same3 = WKV._apply_fact_boost(lg, [3, 5], 0.0) is lg_orig
    changed = WKV._apply_fact_boost(lg, [3, 5], 6.0)
    boost_actually_changes = not np.array_equal(changed, lg_orig) and changed[3] == 9.0 and changed[5] == 11.0

    t1, _ = WKV.generate("once upon a time there was a little boy who liked to play", seed=42, max_new_tokens=25,
                         facts=None)
    t2, _ = WKV.generate("once upon a time there was a little boy who liked to play", seed=42, max_new_tokens=25,
                         facts=None)
    off_deterministic = (t1 == t2)

    env_before = os.environ.pop("BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND", None)
    flag_default_off = OE.wkv_fact_grounding_enabled() is False
    os.environ["BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND"] = "1"
    flag_on_when_set = OE.wkv_fact_grounding_enabled() is True
    if env_before is None:
        os.environ.pop("BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND", None)
    else:
        os.environ["BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND"] = env_before

    out["part4_mechanism_noop_checks"] = {
        "apply_fact_boost_noop_empty_ids": same1,
        "apply_fact_boost_noop_none_ids": same2,
        "apply_fact_boost_noop_zero_boost": same3,
        "apply_fact_boost_actually_boosts_when_on": bool(boost_actually_changes),
        "generate_facts_none_deterministic_repeat": off_deterministic,
        "flag_default_off": flag_default_off,
        "flag_on_when_env_set": flag_on_when_set,
    }

    v = Verdict("board #112 clean-unlock: known-topic WKV-mouth reply surfaces the brain's real recalled fact")
    v.require("mechanism is an exact no-op off (empty ids)", same1, expect=True)
    v.require("mechanism is an exact no-op off (None ids)", same2, expect=True)
    v.require("mechanism is an exact no-op off (zero boost)", same3, expect=True)
    v.require("mechanism genuinely boosts when on", bool(boost_actually_changes), expect=True)
    v.require("flag default-off", flag_default_off, expect=True)
    v.control("fact-word surfaces in raw generation, boosted vs baseline",
              treatment=n_boost_surfaced, control=n_base_surfaced)
    verdict = v.decide(go=(n_boost_surfaced > n_base_surfaced) and boost_actually_changes and same1 and same2
                       and same3 and flag_default_off)
    out["verdict"] = verdict
    out["verdict_preconditions"] = v.to_dict()

    print(json.dumps({k: out[k] for k in out if k not in ("part3_before_after_demo",)}, indent=2, default=str))
    print("\n--- part3 demo summary ---")
    for r in part3:
        print(f"  {r['agent']:35s} boosted_words={r['boosted_words']!r:25s} "
              f"base_surfaced={r['baseline_surfaces_fact_word']!s:5s} boost_surfaced={r['boosted_surfaces_fact_word']!s:5s} "
              f"survives_filter={r['boosted_word_in_filtered_output']}")
    print(f"\nVERDICT: {verdict}")
    return out


if __name__ == "__main__":
    result = main()
    out_path = REPO_ROOT / "research/findings/raw/_wkv_mouth_fact_grounding_derisk.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")
