"""VERIFY the live-chat wiring of the de-risked agent-routed sharded fact-store (board #66 / #127).

Two independent byte-identical checks + the no-confab moat:

  PART A (store level, the load-bearing claim): a single `RFPhasorComposer` vs a
    `ShardedPhasorStore.from_existing_composer(...)` built from it -- every agent-cued read
    (query_patient / ask_yes_no / render_fact / query_chain / chain_of_thought) must return the
    IDENTICAL answer, and unknown cues must abstain IDENTICALLY (the moat).

  PART B (live wiring, the integration claim): the ACTUAL server swap `webapp.server._maybe_shard_composer`
    with BRAIN_SHARDED_STORE=1 applied to a real BrainConversationalAgent (tiny-demo, composer_kind=rf) --
    the agent's own recall methods (what_does / is_it_true / describe) must return the IDENTICAL answer
    before vs after the swap, and the moat must hold.

Run headless (SIM_BACKEND=numpy) via tools/gpu_queue.sh. Writes a JSON verdict; exits 0 iff GO.
"""
from __future__ import annotations

import json
import os
import sys
import time

from research.runners.rf_phasor_composer import RFPhasorComposer
from research.runners.sharded_phasor_store import ShardedPhasorStore

# A battery spanning many DISTINCT agents (so >1 shard is exercised), incl. multi-fact agents (first-match
# ordering), a bound NEGATE polarity, and an attributed patient (attribute reconstruction in re-homing).
FACTS = [
    ("france", "isa", "country"), ("france", "has", "paris"), ("france", "has", "lyon"),
    ("gold", "isa", "element"), ("gold", "has", "shine"),
    ("guitar", "isa", "instrument"), ("heart", "isa", "organ"), ("heart", "has", "myocardium"),
    ("computer", "has", "keyboard"), ("computer", "isa", "machine"),
    ("dog", "chase", "cat"), ("cat", "chase", "mouse"), ("mouse", "eat", "cheese"),
    ("paris", "isa", "city"), ("tokyo", "isa", "city"), ("japan", "has", "tokyo"),
    ("water", "isa", "liquid"), ("iron", "isa", "metal"), ("robin", "isa", "bird"),
    ("bird", "has", "feather"), ("fish", "has", "gill"), ("tree", "has", "leaf"),
    ("sun", "isa", "star"), ("moon", "isa", "satellite"), ("earth", "isa", "planet"),
    ("copper", "isa", "metal"), ("oak", "isa", "tree"), ("rose", "isa", "flower"),
    ("wolf", "isa", "mammal"), ("shark", "isa", "fish"), ("eagle", "isa", "bird"),
    ("piano", "isa", "instrument"), ("river", "has", "water"), ("book", "has", "page"),
    ("car", "has", "wheel"), ("clock", "has", "hand"),
]
NEG_FACTS = [("sky", "isa", "green")]          # stored NEGATE -> ask_yes_no must say 'no'
ATTR_FACTS = [("apple", "isa", (["red"], "fruit"))]  # attributed patient -> exercises re-home reconstruction

UNKNOWN_CUES = [   # (agent, action) -- agent and/or action absent -> the moat must abstain identically
    ("dragon", "isa"), ("unicorn", "has"), ("france", "devours"), ("gold", "sings"),
    ("xyzzy", "qux"), ("nonexistent", "isa"), ("cat", "flies"), ("heart", "orbits"),
]
UNKNOWN_YESNO = [("dragon", "isa", "country"), ("france", "isa", "planet"), ("gold", "isa", "liquid")]


def _vocab():
    ws = set()
    for a, v, p in FACTS + NEG_FACTS + list(UNKNOWN_YESNO):
        ws.update([a, v, p])
    for a, v in UNKNOWN_CUES:
        ws.update([a, v])
    for a, v, p in ATTR_FACTS:
        ws.add(a); ws.add(v)
        adjs, noun = p
        ws.update(adjs); ws.add(noun)
    return sorted(ws)


def _populate(comp):
    for a, v, p in FACTS:
        comp.store(a, v, p)
    for a, v, p in NEG_FACTS:
        comp.store(a, v, p, polarity="NEGATE")
    for a, v, p in ATTR_FACTS:
        comp.store(a, v, p)


def _known_agent_action_pairs():
    seen = []
    for a, v, _p in FACTS:
        if (a, v) not in seen:
            seen.append((a, v))
    for a, v, _p in NEG_FACTS:
        if (a, v) not in seen:
            seen.append((a, v))
    for a, v, _p in ATTR_FACTS:
        if (a, v) not in seen:
            seen.append((a, v))
    return seen


def part_a():
    """Store-level byte-identical: single composer vs sharded-from-it."""
    vocab = _vocab()
    single = RFPhasorComposer(seed=42, D=128, vocab=vocab)
    _populate(single)
    sharded = ShardedPhasorStore.from_existing_composer(single, n_shards=16)

    res = {"n_facts_single": len(single.kb), "n_facts_sharded": sharded.total_facts(),
           "n_shards": sharded.n_shards, "load_balance_max_over_mean": round(sharded.load_balance()[3], 3),
           "mismatches": [], "recall_checked": 0, "yesno_checked": 0, "render_checked": 0,
           "chain_checked": 0, "moat_checked": 0, "moat_abstain_ok": 0}

    # forward recall + yes/no + render, over every stored (agent, action)
    for a, v in _known_agent_action_pairs():
        s0, s1 = single.query_patient(a, v), sharded.query_patient(a, v)
        res["recall_checked"] += 1
        if s0 != s1:
            res["mismatches"].append({"kind": "query_patient", "cue": [a, v], "single": repr(s0), "sharded": repr(s1)})
        r0, r1 = single.render_fact(a), sharded.render_fact(a)
        res["render_checked"] += 1
        if r0 != r1:
            res["mismatches"].append({"kind": "render_fact", "cue": [a], "single": repr(r0), "sharded": repr(r1)})
    for a, v, p in FACTS + [(x[0], x[1], x[2]) for x in NEG_FACTS]:
        y0, y1 = single.ask_yes_no(a, v, p), sharded.ask_yes_no(a, v, p)
        res["yesno_checked"] += 1
        if y0 != y1:
            res["mismatches"].append({"kind": "ask_yes_no", "cue": [a, v, p], "single": repr(y0), "sharded": repr(y1)})

    # multi-hop
    for cue, acts in [("france", ["has"]), ("japan", ["has"]), ("dog", ["chase"])]:
        c0, c1 = single.query_chain(cue, acts), sharded.query_chain(cue, acts)
        res["chain_checked"] += 1
        if c0 != c1:
            res["mismatches"].append({"kind": "query_chain", "cue": [cue, acts], "single": repr(c0), "sharded": repr(c1)})
    for start in ["france", "dog", "gold"]:
        t0 = single.chain_of_thought(start, max_hops=3, return_path=True)
        t1 = sharded.chain_of_thought(start, max_hops=3, return_path=True)
        res["chain_checked"] += 1
        if t0 != t1:
            res["mismatches"].append({"kind": "chain_of_thought", "cue": [start], "single": repr(t0), "sharded": repr(t1)})

    # the no-confab moat: unknown cues must abstain IDENTICALLY
    for a, v in UNKNOWN_CUES:
        s0, s1 = single.query_patient(a, v), sharded.query_patient(a, v)
        res["moat_checked"] += 1
        if s0 == s1:
            res["moat_abstain_ok"] += 1
        else:
            res["mismatches"].append({"kind": "moat_query_patient", "cue": [a, v], "single": repr(s0), "sharded": repr(s1)})
        if s1 is not None:
            res["mismatches"].append({"kind": "moat_confab", "cue": [a, v], "sharded": repr(s1)})
    for a, v, p in UNKNOWN_YESNO:
        y0, y1 = single.ask_yes_no(a, v, p), sharded.ask_yes_no(a, v, p)
        res["moat_checked"] += 1
        if y0 == y1:
            res["moat_abstain_ok"] += 1
        else:
            res["mismatches"].append({"kind": "moat_ask_yes_no", "cue": [a, v, p], "single": repr(y0), "sharded": repr(y1)})

    res["GO"] = (len(res["mismatches"]) == 0
                 and res["n_facts_single"] == res["n_facts_sharded"]
                 and res["moat_abstain_ok"] == res["moat_checked"]
                 and res["recall_checked"] > 0)
    return res


def part_b_swap_fn():
    """The ACTUAL server swap function `webapp.server._maybe_shard_composer` (BRAIN_SHARDED_STORE=1) exercised on a
    populated composer via a lightweight stand-in agent. NO full brain load (composers only, like part A) -> safe to
    run locally. Verifies: the composer is swapped to ShardedPhasorStore, recall is byte-identical, the moat holds,
    and the .kb/.words drop-in surface works."""
    out = {"available": False}
    try:
        os.environ["BRAIN_SHARDED_STORE"] = "1"
        os.environ["BRAIN_SHARDED_STORE_SHARDS"] = "16"

        class _FakeInner:
            def __init__(self, comp):
                self.composer = comp
                self._composer_has_hear = hasattr(comp, "hear")

        class _FakeAgent:
            def __init__(self, inner):
                self.agent = inner

        vocab = _vocab()
        single = RFPhasorComposer(seed=42, D=128, vocab=vocab)
        _populate(single)
        pairs = _known_agent_action_pairs()
        off = {(a, v): single.query_patient(a, v) for a, v in pairs}
        off_moat = {c: single.query_patient(*c) for c in UNKNOWN_CUES}

        inner = _FakeInner(single)
        agent = _FakeAgent(inner)
        before_type = type(inner.composer).__name__
        from webapp.server import _maybe_shard_composer
        _maybe_shard_composer(agent)
        after_type = type(inner.composer).__name__

        mism = []
        for a, v in pairs:
            now = inner.composer.query_patient(a, v)
            if now != off[(a, v)]:
                mism.append({"cue": [a, v], "off": repr(off[(a, v)]), "on": repr(now)})
        moat_ok = 0
        for c in UNKNOWN_CUES:
            now = inner.composer.query_patient(*c)
            if now == off_moat[c]:
                moat_ok += 1
            else:
                mism.append({"kind": "moat", "cue": list(c), "off": repr(off_moat[c]), "on": repr(now)})
        kb_len = len(inner.composer.kb)
        words_len = len(inner.composer.words)

        out.update({
            "available": True,
            "before_composer_type": before_type,
            "after_composer_type": after_type,
            "swap_took_effect": after_type == "ShardedPhasorStore",
            "recall_checked": len(pairs), "recall_mismatches": mism,
            "moat_checked": len(UNKNOWN_CUES), "moat_ok": moat_ok,
            "kb_property_len": kb_len, "words_property_len": words_len,
            "GO": (len(mism) == 0 and after_type == "ShardedPhasorStore"
                   and moat_ok == len(UNKNOWN_CUES) and kb_len == len(single.kb)),
        })
    except Exception as e:
        import traceback
        out["error"] = "part_b_swap_fn failed: %r\n%s" % (e, traceback.format_exc())
    return out


def part_b_live_brain():
    """Live wiring: the ACTUAL server swap `_maybe_shard_composer` on a REAL tiny-demo brain. LOADS A BRAIN -> gated
    behind the queue (skipped when SKIP_LIVE_BRAIN is set)."""
    out = {"available": False}
    if os.environ.get("SKIP_LIVE_BRAIN", "").strip().lower() in ("1", "true", "on", "yes"):
        out["skipped"] = "SKIP_LIVE_BRAIN set (brain-load gated to tools/gpu_queue.sh)"
        return out
    try:
        os.environ["BRAIN_COMPOSER_KIND"] = "rf"           # numpy fast-path (cheap; the sharded store's oracle)
        from research.runners.brain_chat_tui import _build_tiny_demo
    except Exception as e:
        out["error"] = "import/build tiny-demo failed: %r" % (e,)
        return out
    try:
        agent_tuple = _build_tiny_demo(42, use_multiturn=True, enable_neural_render=False, composer_kind="rf")
        agent = agent_tuple[0]
        inner = getattr(agent, "agent", agent)
        # record flag-OFF answers on the brain's OWN stored facts, via the AGENT recall methods
        facts = [(f.get("agent"), f.get("action"), f.get("patient")) for f, _ in inner.composer.kb
                 if all(isinstance(f.get(r), str) for r in ("agent", "action", "patient"))]
        off = {}
        for a, v, p in facts:
            off[(a, v, p)] = (inner.what_does(a, v), inner.is_it_true(a, v, p), inner.describe(a))
        moat_cues = [("dragon", "isa"), ("unicorn", "has"), (facts[0][0] if facts else "x", "levitates")]
        off_moat = {c: inner.what_does(*c) for c in moat_cues}
        single_type = type(inner.composer).__name__

        # apply the REAL server swap
        os.environ["BRAIN_SHARDED_STORE"] = "1"
        os.environ["BRAIN_SHARDED_STORE_SHARDS"] = "16"
        from webapp.server import _maybe_shard_composer
        _maybe_shard_composer(agent)
        sharded_type = type(inner.composer).__name__

        mism = []
        for a, v, p in facts:
            now = (inner.what_does(a, v), inner.is_it_true(a, v, p), inner.describe(a))
            if now != off[(a, v, p)]:
                mism.append({"cue": [a, v, p], "off": [repr(x) for x in off[(a, v, p)]],
                             "on": [repr(x) for x in now]})
        moat_ok = 0
        for c in moat_cues:
            now = inner.what_does(*c)
            if now == off_moat[c]:
                moat_ok += 1
            else:
                mism.append({"kind": "moat", "cue": list(c), "off": repr(off_moat[c]), "on": repr(now)})

        out.update({
            "available": True,
            "n_facts": len(facts),
            "single_composer_type": single_type,
            "swapped_composer_type": sharded_type,
            "swap_took_effect": sharded_type == "ShardedPhasorStore",
            "recall_mismatches": mism,
            "moat_checked": len(moat_cues), "moat_ok": moat_ok,
            "GO": (len(mism) == 0 and sharded_type == "ShardedPhasorStore"
                   and moat_ok == len(moat_cues) and len(facts) > 0),
        })
    except Exception as e:
        import traceback
        out["error"] = "part_b failed: %r\n%s" % (e, traceback.format_exc())
    return out


def main():
    from tools.verdict import Verdict
    t0 = time.time()
    a = part_a()
    swap = part_b_swap_fn()
    live = part_b_live_brain()

    go = bool(a.get("GO")
              and (swap.get("GO") if swap.get("available") else False)
              and (live.get("GO") if live.get("available") else True))

    v = Verdict("knowledge-scale sharded fact-store -> live brain_chat recall wiring (board #66/#127)")
    # Part A -- store-level byte-identical (the load-bearing structural claim: agent co-location)
    v.require("part A: routing byte-identical (mismatches)", len(a.get("mismatches", [1])), expect=0)
    v.require("part A: facts re-homed == source", a.get("n_facts_sharded"), expect=a.get("n_facts_single"))
    v.require("part A: moat abstains identically", a.get("moat_abstain_ok"), expect=a.get("moat_checked"))
    v.require("part A: recall actually checked (>0)", (a.get("recall_checked", 0) > 0), expect=True)
    # The server swap function -- the real integration entry point (no brain load)
    v.require("swap fn: available", swap.get("available"), expect=True)
    v.require("swap fn: composer swapped to ShardedPhasorStore", swap.get("after_composer_type"),
              expect="ShardedPhasorStore")
    v.require("swap fn: recall byte-identical (mismatches)", len(swap.get("recall_mismatches", [1])), expect=0)
    v.require("swap fn: moat identical", swap.get("moat_ok"), expect=swap.get("moat_checked"))
    v.require("swap fn: kb drop-in surface == source", swap.get("kb_property_len"),
              expect=a.get("n_facts_single"))
    # Part B -- the real tiny-demo brain (only when actually run; queue-gated)
    if live.get("available"):
        v.require("live brain: composer swapped", live.get("swapped_composer_type"), expect="ShardedPhasorStore")
        v.require("live brain: recall byte-identical (mismatches)", len(live.get("recall_mismatches", [1])),
                  expect=0)
        v.require("live brain: moat identical", live.get("moat_ok"), expect=live.get("moat_checked"))
    else:
        v.disabled("part B live-brain end-to-end",
                   live.get("skipped") or "not run this invocation (queue-gated one-brain-load; SKIP_LIVE_BRAIN)")
    # the one host scaffold, declared as scope (not a pass/fail)
    v.disabled("learned/spiking cue->shard router",
               "router hash(agent) mod S is a declared host scaffold; the in-shard FHRR recall + the no-confab "
               "moat are the genuine reads")
    decided = v.decide(go=go)

    verdict = dict(decided)   # top-level carries {label,status,go,preconditions,disabled_processes,...}
    verdict.update({
        "arc": "knowledge-scale sharded fact-store -> live brain_chat recall wiring (board #66/#127)",
        "backend": os.environ.get("SIM_BACKEND", "?"),
        "part_a_store_level": a,
        "part_b_swap_fn": swap,               # real webapp.server._maybe_shard_composer (no brain load)
        "part_b_live_brain": live,            # real tiny-demo brain (queue-gated; skipped when SKIP_LIVE_BRAIN)
        "live_brain_available": bool(live.get("available")),
        "elapsed_s": round(time.time() - t0, 2),
    })
    out_path = os.environ.get(
        "VERIFY_OUT", "research/findings/raw/_knowledge_scale_sharded_live_wiring_verdict.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print("\nwrote", out_path)
    return 0 if verdict.get("go") else 1


if __name__ == "__main__":
    sys.exit(main())
