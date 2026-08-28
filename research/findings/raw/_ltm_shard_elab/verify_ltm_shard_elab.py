"""VERIFY: RichAnswerComposer elaboration can READ the routed cortical LTM shard (additive, default-OFF,
`BRAIN_ELABORATE_FROM_LTM_SHARD`) -- the convergent blocker behind confidence->forthcomingness (board #94) AND
knowledge-in-chat (owner priority #66).

BEFORE: `_stored_facts`/`_facts_about`/`_facts_mentioning` read `self.composer.kb`, and `TieredFactStore.__getattr__`
delegates `.kb` to the BUFFER tier -- so elaboration saw only the small conversational buffer, never the routed
`ShardedPhasorStore` LTM. An answer whose concept lives in long-term memory got a bare direct fact + same-relation
chain hops and NO breadth; on the true production floor (max_sentences=4) the reach-cap never had extra content to
trim, so a confident vs an uncertain turn kept identical sentences (the hollow-flip the owner's rule prohibits).

This numpy smoke (GPU-free wiring check; the DECISIVE 6-seed run is cupy) proves, on a tiny buffer + a real routed
LTM shard:
  (1) BYTE-IDENTICAL OFF -- with an LTM attached but the flag OFF, gather() is identical to no-LTM (the flag gates
      every draw); AND flag ON with NO ltm tier == flag OFF (guarded no-op).
  (2) MOAT -- an unknown entity abstains (0 facts) even with the flag ON + a populated LTM.
  (3) LOAD-BEARING -- for a concept whose knowledge is in the LTM, flag ON draws MORE grounded elaboration than
      OFF; varying the shard content changes the elaboration; lesioning the shard-read (flag OFF) reverts it.
  (4) CAP ENGAGES -- during the confidence reach (max_sentences=floor+1), flag ON produces > floor facts so
      `confidence_forthcoming_chat.apply_cap` truncates on a LOW read (`low_confidence_capped`) and keeps them on a
      HIGH read (`high_confidence`), whereas OFF stays at `nothing_to_cap` (the exact hollow shape, unblocked).
All facts drawn are genuine LTM store facts (brain-sourced by construction); the per-sentence VERIFY still gates
each rendered sentence.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.rich_answer_composer import RichAnswerComposer  # noqa: E402
from research.runners.tiered_fact_store import TieredFactStore, build_ltm_from_facts  # noqa: E402
from research.runners.brain_chat_tui import ChatBrain, StubRenderer, DEFAULT_SELF_ALIASES  # noqa: E402
from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402
from webapp import confidence_forthcoming_chat as CF  # noqa: E402


# A small conversational BUFFER (a couple of self-facts -- what a live chat has taught) + a routed cortical LTM.
_BUFFER_FACTS = [
    ("brain", "use", "spikes"),
    ("brain", "learn", "words"),
]

# The LTM knowledge: a concept ("canada") with several facts, plus a chainable second hop ("usa"->"mexico") so the
# multi-hop chain corner-walks the LTM too. Routed by AGENT (all of canada's facts land in canada's shard).
_LTM_FACTS = [
    {"agent": "canada", "action": "border", "patient": "usa"},
    {"agent": "usa", "action": "border", "patient": "mexico"},   # lets the same-relation chain walk a 2nd hop
    {"agent": "canada", "action": "has", "patient": "provinces"},
    {"agent": "canada", "action": "speak", "patient": "english"},
    {"agent": "canada", "action": "export", "patient": "oil"},
    {"agent": "canada", "action": "contain", "patient": "ontario"},
    {"agent": "canada", "action": "border", "patient": "arctic"},
]


def _build_chat(with_ltm=True, ltm_facts=None, seed=42):
    """A tiny ChatBrain (rf composer, stub renderer) at `seed`. When `with_ltm`, its composer is a
    TieredFactStore(buffer, ShardedPhasorStore(ltm_facts)); otherwise a plain buffer (no LTM tier)."""
    vocab_extra = sorted({w for f in (ltm_facts or _LTM_FACTS)
                          for w in (f["agent"], f["action"], f["patient"])})
    concepts = {w: None for w in sorted({w for f in _BUFFER_FACTS for w in f}) + vocab_extra}
    agent = BrainConversationalAgent(seed=seed, concepts=concepts, composer_kind="rf", enable_neural_render=False)
    for a, v, p in _BUFFER_FACTS:
        agent.hear(f"{a} {v} {p}", polarity="AFFIRM")
    if with_ltm:
        ltm = build_ltm_from_facts(list(ltm_facts or _LTM_FACTS), seed=seed, D=64)
        agent.composer = TieredFactStore(agent.composer, ltm)
    chat = ChatBrain(agent, self_aliases=DEFAULT_SELF_ALIASES, renderer=StubRenderer())
    return chat


def _gather_facts(chat, question, *, flag, max_sentences=4, max_elaborations=2, neural=False, seed=42):
    """gather() the fact-set for `question` under the given flag state + composer caps. Returns the list of
    [a,v,p]. Fresh composer each call (no thread-state carry). Neural planner OFF by default for a fast, bridge-free
    numpy check; the production path (neural=True) is exercised separately below."""
    if flag:
        os.environ["BRAIN_ELABORATE_FROM_LTM_SHARD"] = "1"
    else:
        os.environ.pop("BRAIN_ELABORATE_FROM_LTM_SHARD", None)
    rich = RichAnswerComposer(chat, max_chain_hops=3, max_elaborations=max_elaborations,
                              max_sentences=max_sentences, neural_planner=neural, planner_seed=seed)
    _topic, facts = rich.gather(question, followup=False)
    return [list(f) for f in facts]


def _answer(chat, question, *, flag, max_sentences=4, max_elaborations=2, neural=True, seed=42):
    if flag:
        os.environ["BRAIN_ELABORATE_FROM_LTM_SHARD"] = "1"
    else:
        os.environ.pop("BRAIN_ELABORATE_FROM_LTM_SHARD", None)
    rich = RichAnswerComposer(chat, max_chain_hops=3, max_elaborations=max_elaborations,
                              max_sentences=max_sentences, neural_planner=neural, planner_seed=seed)
    return rich, rich.answer(question)


def run_checks(seed=42):
    Q = "what does canada border"          # answer 'usa' lives ONLY in the LTM (buffer has no canada facts)
    results = {}

    # ---------- (1) BYTE-IDENTICAL OFF (same-composer invariants) -------------------------------------------
    # The end-to-end proof (the full smoke battery is byte-identical to HEAD with the flag off) is done separately
    # (smoke_HEAD_baseline.json vs smoke_flagoff.json). Here we prove the SAME-COMPOSER invariants that guarantee
    # it: on ONE composer, toggling the flag changes nothing unless an LTM tier is present AND the flag is ON.
    chat_ltm = _build_chat(with_ltm=True, seed=seed)
    chat_noltm = _build_chat(with_ltm=False, seed=seed)
    # (a) guarded no-op: flag ON but composer has NO ltm tier == flag OFF (identical gather on the SAME composer).
    g_on_noltm = _gather_facts(chat_noltm, Q, flag=True, seed=seed)
    g_off_noltm = _gather_facts(chat_noltm, Q, flag=False, seed=seed)
    noltm_noop = (g_on_noltm == g_off_noltm)
    # (b) with an LTM attached, flag OFF: _ltm_store() is None and the fact accessors return EXACTLY the buffer-only
    # filter (no LTM fact leaks). Assert directly on the accessors the elaboration reads.
    os.environ.pop("BRAIN_ELABORATE_FROM_LTM_SHARD", None)
    rich_off = RichAnswerComposer(chat_ltm, neural_planner=False)
    buf_stored = rich_off._stored_facts()
    about_off = rich_off._facts_about("canada")
    mentioning_off = rich_off._facts_mentioning("canada")
    off_is_buffer_only = (rich_off._ltm_store() is None
                          and rich_off._ltm_facts_about("canada") == []
                          and about_off == [[a, v, p] for (a, v, p) in buf_stored if a == "canada"]
                          and mentioning_off == [[a, v, p] for (a, v, p) in buf_stored if "canada" in (a, p)])
    # (c) same accessor, flag ON, DOES draw the LTM facts (the toggle is the only difference).
    os.environ["BRAIN_ELABORATE_FROM_LTM_SHARD"] = "1"
    rich_on = RichAnswerComposer(chat_ltm, neural_planner=False)
    about_on = rich_on._facts_about("canada")
    on_draws_ltm = (rich_on._ltm_store() is not None and len(about_on) > len(about_off))
    os.environ.pop("BRAIN_ELABORATE_FROM_LTM_SHARD", None)
    results["byte_identical_off"] = {
        "ok": bool(noltm_noop and off_is_buffer_only and on_draws_ltm),
        "noLTM_tier_flag_is_noop": bool(noltm_noop),
        "off_accessors_are_buffer_only": bool(off_is_buffer_only),
        "on_toggle_draws_ltm": bool(on_draws_ltm),
        "facts_about_canada_off": about_off, "facts_about_canada_on": about_on,
    }

    # ---------- (3) LOAD-BEARING (host + neural planner) ----------------------------------------------------
    # flag ON draws MORE grounded facts than OFF for an LTM concept; the extra facts are all real LTM store facts.
    g_off = _gather_facts(chat_ltm, Q, flag=False, seed=seed)
    g_on = _gather_facts(chat_ltm, Q, flag=True, seed=seed)
    ltm_keys = {(f["agent"], f["action"], f["patient"]) for f in _LTM_FACTS}
    extra = [f for f in g_on if f not in g_off]
    extra_all_ltm = all(tuple(f) in ltm_keys for f in extra)
    load_bearing_host = len(g_on) > len(g_off) and len(extra) >= 1 and extra_all_ltm
    # neural planner (production config) -- the topic 'canada' is absent from the buffer assoc graph, so before the
    # LTM topic-fill the neural elaboration was empty; ON it draws canada's routed LTM facts.
    g_off_neural = _gather_facts(chat_ltm, Q, flag=False, neural=True, seed=seed)
    g_on_neural = _gather_facts(chat_ltm, Q, flag=True, neural=True, seed=seed)
    load_bearing_neural = len(g_on_neural) > len(g_off_neural)
    results["load_bearing"] = {
        "ok": bool(load_bearing_host and load_bearing_neural),
        "host_off_n": len(g_off), "host_on_n": len(g_on), "host_off": g_off, "host_on": g_on,
        "extra_facts": extra, "extra_all_from_ltm": bool(extra_all_ltm),
        "neural_off_n": len(g_off_neural), "neural_on_n": len(g_on_neural),
    }

    # VARY THE SHARD CONTENT -> the elaboration changes (proves it rides the shard, not a constant).
    fewer = [f for f in _LTM_FACTS if f["patient"] not in ("provinces", "english", "oil")]
    chat_fewer = _build_chat(with_ltm=True, ltm_facts=fewer, seed=seed)
    g_on_fewer = _gather_facts(chat_fewer, Q, flag=True, seed=seed)
    varying_changes = (g_on_fewer != g_on)
    results["varying_shard_changes_elaboration"] = {
        "ok": bool(varying_changes), "full_shard_on": g_on, "fewer_shard_on": g_on_fewer,
    }

    # LESION the shard-read (flag OFF is the lesion of THIS coupling) -> the ON-only LTM elaboration extras vanish.
    lesion_reverts = (len(g_off) < len(g_on)) and all(f not in g_off for f in extra)
    results["lesion_reverts"] = {"ok": bool(lesion_reverts), "lesioned_gather": g_off, "intact_gather": g_on}

    # ---------- (2) MOAT: unknown entity abstains even with flag ON -----------------------------------------
    os.environ["BRAIN_ELABORATE_FROM_LTM_SHARD"] = "1"
    rich_moat = RichAnswerComposer(chat_ltm, max_chain_hops=3, max_elaborations=2, max_sentences=4,
                                   neural_planner=False, planner_seed=seed)
    r_unknown = rich_moat.answer("what does zzzznonexistent border")
    moat_ok = bool(r_unknown["abstained"]) and len(r_unknown["facts"]) == 0
    # and every fact any real turn surfaced is a genuine buffer-or-LTM fact (no confabulation)
    buf_keys = {(a, v, p) for (a, v, p) in RichAnswerComposer(chat_noltm)._stored_facts()}
    known_keys = buf_keys | ltm_keys
    no_confab = all(tuple(f) in known_keys for f in g_on)
    results["moat"] = {"ok": bool(moat_ok and no_confab), "unknown_abstained": bool(r_unknown["abstained"]),
                       "unknown_facts": r_unknown["facts"], "on_facts_all_brain_sourced": bool(no_confab),
                       "unknown_answer": r_unknown["answer"]}

    # ---------- (4) CONFIDENCE REACH-CAP ENGAGES ------------------------------------------------------------
    # Production floor = (max_sentences=4, max_elaborations=2). The reach bumps to (5, 3). With the flag ON, the
    # LTM-fed gather produces > floor(4) facts, so apply_cap TRUNCATES on a low read and KEEPS on a high read;
    # with the flag OFF the buffer-only gather stays <= floor -> `nothing_to_cap` (the hollow shape).
    FLOOR_S, FLOOR_E = 4, 2
    reach_s, reach_e = CF.reach_plan(FLOOR_S, FLOOR_E)      # (5, 3)

    def _cap_probe(chat, flag, confident):
        rich, r = _answer(chat, Q, flag=flag, max_sentences=reach_s, max_elaborations=reach_e, neural=True, seed=seed)
        _r_out, trace = CF.apply_cap(rich, r, FLOOR_S, confident)
        return len(r.get("facts", [])), trace

    n_on, tr_on_low = _cap_probe(_build_chat(True, seed=seed), True, False)   # ON + LOW read -> should truncate
    _n_on2, tr_on_high = _cap_probe(_build_chat(True, seed=seed), True, True)  # ON + HIGH read -> keep the reach fact
    n_off, tr_off = _cap_probe(_build_chat(True, seed=seed), False, False)    # OFF -> nothing beyond floor to cap
    cap_engages = (n_on > FLOOR_S
                   and tr_on_low.get("reason") == "low_confidence_capped"
                   and tr_on_high.get("reason") == "high_confidence" and tr_on_high.get("granted") is True
                   and tr_off.get("reason") == "nothing_to_cap")
    results["confidence_cap_engages"] = {
        "ok": bool(cap_engages), "floor": FLOOR_S, "reach": reach_s,
        "on_reach_facts": n_on, "off_reach_facts": n_off,
        "on_low_reason": tr_on_low.get("reason"), "on_low_kept": tr_on_low.get("kept_sentences"),
        "on_high_reason": tr_on_high.get("reason"), "on_high_granted": tr_on_high.get("granted"),
        "off_reason": tr_off.get("reason"),
    }

    # ---------- per-seed verdict ---------------------------------------------------------------------------
    os.environ.pop("BRAIN_ELABORATE_FROM_LTM_SHARD", None)
    checks = {k: v["ok"] for k, v in results.items()}
    go = all(checks.values())
    summary = {
        "host_off_n": len(g_off), "host_on_n": len(g_on), "neural_off_n": len(g_off_neural),
        "neural_on_n": len(g_on_neural), "reach_on_facts": n_on, "floor": FLOOR_S,
        "on_low_reason": tr_on_low.get("reason"), "on_high_reason": tr_on_high.get("reason"),
        "off_reason": tr_off.get("reason"),
    }
    return {"seed": int(seed), "GO": bool(go), "checks": checks, "detail": results, "summary": summary}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="verify RichAnswerComposer elaborates from the routed LTM shard.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="one or more seeds; GO requires every seed to pass all checks.")
    ap.add_argument("--out", default=os.path.join(_HERE, "verify_ltm_shard_elab.json"))
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)

    per_seed = [run_checks(s) for s in a.seeds]
    all_go = all(r["GO"] for r in per_seed)
    check_names = list(per_seed[0]["checks"].keys()) if per_seed else []
    per_check_all_seeds = {c: all(r["checks"].get(c) for r in per_seed) for c in check_names}
    # EARN the verdict: each check is a precondition that must hold on EVERY seed (tools.verdict.Verdict emits the
    # `preconditions` block the verdict-preconditions gate enforces). All six checks measured across all seeds ->
    # GO; any unmet -> UNDEFINED (never a bare GO). The flag is additive + default-OFF, so this is a wiring GO on
    # numpy; the decisive cross-backend verdict is the staged cupy 6-seed run.
    from tools.verdict import Verdict
    v = Verdict("rich-composer elaboration reads the routed LTM shard (BRAIN_ELABORATE_FROM_LTM_SHARD, default-OFF)")
    for c in check_names:
        v.require(c, per_check_all_seeds[c], expect=True, note=f"holds on all {len(a.seeds)} seeds")
    decided = v.decide(go=bool(all_go), verbose=False)
    out = {
        "probe": "rich_answer_composer_elaborate_from_ltm_shard",
        "flag": "BRAIN_ELABORATE_FROM_LTM_SHARD (additive, default-OFF, byte-identical when off)",
        "backend": os.environ.get("SIM_BACKEND"),
        "seeds": list(a.seeds),
        "n_seeds": len(a.seeds),
        "GO": bool(all_go),
        "status": decided["status"],
        "preconditions": decided["preconditions"],
        "undefined_reasons": decided["undefined_reasons"],
        "per_check_all_seeds": per_check_all_seeds,
        "per_seed": per_seed,
    }
    out_path = a.out if os.path.isabs(a.out) else os.path.join(_REPO, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("=" * 92)
    print(f"  seeds={a.seeds}  backend={os.environ.get('SIM_BACKEND')}")
    for c in check_names:
        print(f"  [{'PASS' if per_check_all_seeds[c] else 'FAIL'}] {c}  (all {len(a.seeds)} seeds)")
    print("=" * 92)
    for r in per_seed:
        s = r["summary"]
        print(f"  seed {r['seed']:>4}: GO={r['GO']}  host {s['host_off_n']}->{s['host_on_n']}  "
              f"neural {s['neural_off_n']}->{s['neural_on_n']}  reach_on={s['reach_on_facts']}(floor {s['floor']})  "
              f"low->{s['on_low_reason']} high->{s['on_high_reason']} off->{s['off_reason']}")
    print("=" * 92)
    print(f"  VERDICT: {'GO -- all seeds pass every check' if all_go else 'NO-GO'}  "
          f"(wrote {os.path.relpath(out_path, _REPO)})")
    return 0 if all_go else 1


if __name__ == "__main__":
    sys.exit(main())
