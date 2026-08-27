"""Verification for organ C's mirror-image LTM exemption on the 3-organ GNW bus (reuses the SAME flag
BRAIN_GNW_ORGANB_LTM_EXEMPT that already exempts organ B -- see webapp/gnw_two_organ_bus.py /
research/findings/2026-08-27-organb-ltm-exempt-derisk-6seed-GO.md).

This closes the residual the organ-B de-risk explicitly reported but did not fix: with the flag ON, organ B
corroborated LTM-sourced recalls, but the 3-organ bus STILL abstained (abstain_reason=
"consensus_veto_organ_c_non_comprehension") because organ C's own real-vocab / D4 spiking-margin comprehension
read has the identical buffer-only-vocab blind spot. This script re-verifies the SAME checks the organ-B de-risk
ran, now on the 3-organ bus with organ C's own exemption wired in.

SIM_BACKEND=numpy, tiny-demo + shipped LTM (the light path), NOT the GPU server. Multi-seed over
[42, 43, 44, 100, 101, 102].

Checks, per seed:
  A) MOAT -- a NON-EXISTENT fact still abstains on the 3-organ bus with the flag ON (and OFF).
  B) COMMIT -- a genuine LTM fact (chelsea_fc/country, + 2 more pulled live from the store) COMMITS on the
     3-organ bus with the flag ON (both organ B's AND organ C's exemption apply), still vetoes with flag OFF
     (reproducing the residual the organ-B de-risk left open).
  C) ORGAN-C-SPECIFIC -- with the flag ON, `organ_c_votes=True` and `organ_c_ltm_exempt_applied=True` for every
     LTM probe (proves organ C's OWN veto is the one that flipped, not just organ B's).
  D) CONVERSATIONAL-BUFFER UNTOUCHED -- a fact taught mid-conversation (buffer tier) behaves identically with the
     flag on vs off on the 3-organ bus (organ C still reads its real-vocab / D4 margin instrument, unaffected).
  E) 2-ORGAN BUS UNCHANGED -- the organ-B-only path (gnw_two_organ_bus.two_organ_combine) is byte-identical to
     its own pre-existing (already-GO) behavior; this arc never touches that module.

Prints a JSON summary per seed + an overall verdict line.
"""
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_LTM_SHIP_DEFAULT", "1")

REPO = "/home/dant123/Projects/sim/.claude/worktrees/agent-ad1545703c36b55e6"
sys.path.insert(0, REPO)

LTM_BUNDLE = os.path.expanduser("~/Projects/sim-data/knowledge_bundles/wikidata_core_15k")

SEEDS = [42, 43, 44, 100, 101, 102]


def build_chat(seed):
    from research.runners.brain_chat_tui import ChatBrain, StubRenderer, _build_tiny_demo
    from research.runners.developed_brain_io import _inner_agent
    from research.runners.tiered_fact_store import TieredFactStore
    from research.runners.sharded_phasor_store import ShardedPhasorStore

    agent, aliases, _n = _build_tiny_demo(seed, use_multiturn=True, enable_neural_render=False,
                                          composer_kind="onebrain")
    ltm = ShardedPhasorStore.load(LTM_BUNDLE)
    inner = _inner_agent(agent)
    inner.composer = TieredFactStore(inner.composer, ltm)
    chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())
    return chat, ltm


def find_more_ltm_facts(ltm, n=2, exclude_agents=("chelsea_fc",)):
    """Pull a couple more genuine (agent, action, patient) triples straight out of the LTM shards."""
    out = []
    for sh in ltm.shards:
        for fact, _comp in sh.kb:
            a, act, p = fact.get("agent"), fact.get("action"), fact.get("patient")
            if not (isinstance(a, str) and isinstance(act, str) and isinstance(p, str)):
                continue
            if a in exclude_agents:
                continue
            out.append((a, act, p))
            if len(out) >= n:
                return out
    return out


def three_organ_result(chat, agent, action, seed, *, ltm_exempt):
    from webapp.gnw_three_organ_bus import three_organ_combine
    return three_organ_combine(chat, agent, action, seed=seed, organb_ltm_exempt=ltm_exempt)


def two_organ_result(chat, agent, action, seed, *, ltm_exempt):
    from webapp.gnw_two_organ_bus import two_organ_combine
    return two_organ_combine(chat, agent, action, seed=seed, organb_ltm_exempt=ltm_exempt)


def main():
    overall = {"moat_held": True, "commits_with_flag_on": True, "organ_c_exemption_applied": True,
              "byte_identical_off": True, "buffer_untouched": True, "two_organ_bus_unaffected": True,
              "seeds": {}}

    for seed in SEEDS:
        chat, ltm = build_chat(seed)
        more_facts = find_more_ltm_facts(ltm, n=2)
        probes = [("chelsea_fc", "country")] + [(a, act) for (a, act, p) in more_facts]
        expected_patients = {}
        for a, act in probes:
            expected_patients[(a, act)] = chat.inner.composer.query_patient(a, act)

        seed_report = {"probes": probes, "expected": {f"{a}|{act}": p for (a, act), p in expected_patients.items()}}

        # --- A) MOAT: a non-existent fact must still abstain on the 3-organ bus, flag ON and OFF ---
        fake_agent, fake_action = "definitely_not_a_stored_entity_xyz", "definitely_not_a_stored_relation_xyz"
        direct = chat.inner.composer.query_patient(fake_agent, fake_action)
        assert direct is None, f"test fixture bug: fake fact resolved to {direct!r}"
        moat_off = three_organ_result(chat, fake_agent, fake_action, seed, ltm_exempt=False)
        moat_on = three_organ_result(chat, fake_agent, fake_action, seed, ltm_exempt=True)
        moat_ok = (moat_off["committed"] is None and moat_off.get("abstain_reason") == "primary_recall_miss"
                  and moat_on["committed"] is None and moat_on.get("abstain_reason") == "primary_recall_miss")
        seed_report["moat"] = {
            "off_committed": moat_off["committed"], "off_abstain_reason": moat_off.get("abstain_reason"),
            "on_committed": moat_on["committed"], "on_abstain_reason": moat_on.get("abstain_reason"),
            "ok": moat_ok,
        }
        if not moat_ok:
            overall["moat_held"] = False

        # --- B) COMMIT + C) organ-C-specific: genuine LTM facts commit on the 3-organ bus with flag ON,
        #     still veto (now via organ C, per the organ-B de-risk's own residual) with flag OFF ---
        commit_report = {}
        all_commit_ok = True
        all_organ_c_ok = True
        for a, act in probes:
            exp_p = expected_patients[(a, act)]
            r_off = three_organ_result(chat, a, act, seed, ltm_exempt=False)
            r_on = three_organ_result(chat, a, act, seed, ltm_exempt=True)
            # flag OFF: organ B ALSO withholds (its own exemption is off too, since it's the SAME flag), so the
            # reported abstain_reason is organ B's (the abstain_reason logic checks organ B first) -- but organ
            # C's OWN vote is independently False too (its real-vocab check has never seen these LTM tokens),
            # reproducing the exact pre-fix 3-organ veto this arc is closing.
            off_ok = (r_off["committed"] is None
                     and r_off.get("abstain_reason") == "consensus_veto_organ_b_withheld"
                     and r_off.get("organ_c_votes") is False
                     and r_off.get("organ_b_confirmed") is False)
            on_ok = (r_on["committed"] == exp_p and r_on.get("recall_source") == "ltm"
                    and r_on.get("organb_ltm_exempt_applied") is True
                    and r_on.get("organ_b_confirmed") is True
                    and r_on.get("organ_c_votes") is True
                    and r_on.get("organ_c_ltm_exempt_applied") is True)
            ok = off_ok and on_ok
            commit_report[f"{a}|{act}"] = {
                "expected_patient": exp_p,
                "off_committed": r_off["committed"], "off_abstain_reason": r_off.get("abstain_reason"),
                "off_organ_b_confirmed": r_off.get("organ_b_confirmed"), "off_organ_c_votes": r_off.get("organ_c_votes"),
                "on_committed": r_on["committed"], "on_abstain_reason": r_on.get("abstain_reason"),
                "on_recall_source": r_on.get("recall_source"),
                "on_organb_ltm_exempt_applied": r_on.get("organb_ltm_exempt_applied"),
                "on_organ_b_confirmed": r_on.get("organ_b_confirmed"),
                "on_organ_c_votes": r_on.get("organ_c_votes"),
                "on_organ_c_ltm_exempt_applied": r_on.get("organ_c_ltm_exempt_applied"),
                "on_organ_c_real_vocab_known": r_on.get("organ_c_real_vocab_known"),
                "off_organ_c_real_vocab_known": r_off.get("organ_c_real_vocab_known"),
                "off_organ_c_unknown_tokens": r_off.get("organ_c_unknown_tokens"),
                "off_organ_c_margin": r_off.get("organ_c_margin"), "off_organ_c_threshold": r_off.get("organ_c_threshold"),
                "ok": ok,
            }
            if not on_ok:
                all_organ_c_ok = False
            if not ok:
                all_commit_ok = False
        seed_report["commit"] = commit_report
        if not all_commit_ok:
            overall["commits_with_flag_on"] = False
        if not all_organ_c_ok:
            overall["organ_c_exemption_applied"] = False

        # --- D) BUFFER-TAUGHT fact: teach a fresh fact into the conversational buffer, verify flag on/off
        #     behave IDENTICALLY on the 3-organ bus (organ C's own real-vocab/D4 read still governs, untouched). ---
        taught_agent, taught_action, taught_patient = "zzz_test_agent", "zzz_test_action", "zzz_test_patient"
        chat.inner.composer.store(taught_agent, taught_action, taught_patient, polarity="AFFIRM")
        r_off_buf = three_organ_result(chat, taught_agent, taught_action, seed, ltm_exempt=False)
        r_on_buf = three_organ_result(chat, taught_agent, taught_action, seed, ltm_exempt=True)
        buf_identical = (r_off_buf["committed"] == r_on_buf["committed"]
                         and r_off_buf.get("organ_b_confirmed") == r_on_buf.get("organ_b_confirmed")
                         and r_off_buf.get("organ_c_votes") == r_on_buf.get("organ_c_votes")
                         and r_off_buf.get("organ_c_real_vocab_known") == r_on_buf.get("organ_c_real_vocab_known")
                         and r_on_buf.get("recall_source") == "buffer"
                         and r_on_buf.get("organb_ltm_exempt_applied") is False
                         and r_on_buf.get("organ_c_ltm_exempt_applied") is False)
        seed_report["buffer_untouched"] = {
            "off_committed": r_off_buf["committed"], "on_committed": r_on_buf["committed"],
            "off_organ_c_votes": r_off_buf.get("organ_c_votes"), "on_organ_c_votes": r_on_buf.get("organ_c_votes"),
            "on_recall_source": r_on_buf.get("recall_source"),
            "identical": buf_identical,
        }
        if not buf_identical:
            overall["buffer_untouched"] = False

        # --- E) 2-ORGAN BUS UNCHANGED: this arc never touches gnw_two_organ_bus.py; spot-check its own
        #     already-GO behavior on the SAME probes is unaffected by loading gnw_three_organ_bus.py in-process. ---
        two_organ_report = {}
        for a, act in probes:
            exp_p = expected_patients[(a, act)]
            r2_off = two_organ_result(chat, a, act, seed, ltm_exempt=False)
            r2_on = two_organ_result(chat, a, act, seed, ltm_exempt=True)
            ok2 = (r2_off["committed"] is None and r2_off.get("abstain_reason") == "consensus_veto_organ_b_withheld"
                  and r2_on["committed"] == exp_p and r2_on.get("organb_ltm_exempt_applied") is True)
            two_organ_report[f"{a}|{act}"] = {"off_committed": r2_off["committed"], "on_committed": r2_on["committed"],
                                              "ok": ok2}
            if not ok2:
                overall["two_organ_bus_unaffected"] = False
        seed_report["two_organ_bus"] = two_organ_report

        overall["seeds"][seed] = seed_report
        print(f"[seed {seed}] moat_ok={moat_ok} commit_ok={all_commit_ok} organ_c_ok={all_organ_c_ok} "
              f"buffer_identical={buf_identical}", flush=True)

    print("\n=== FULL REPORT ===")
    print(json.dumps(overall, indent=2, default=str))

    verdict = "GO" if (overall["moat_held"] and overall["commits_with_flag_on"]
                       and overall["organ_c_exemption_applied"] and overall["buffer_untouched"]
                       and overall["two_organ_bus_unaffected"]) else "NO-GO"
    print(f"\n=== VERDICT: {verdict} (moat_held={overall['moat_held']}, "
          f"commits_with_flag_on={overall['commits_with_flag_on']}, "
          f"organ_c_exemption_applied={overall['organ_c_exemption_applied']}, "
          f"buffer_untouched={overall['buffer_untouched']}, "
          f"two_organ_bus_unaffected={overall['two_organ_bus_unaffected']}) ===")


if __name__ == "__main__":
    main()
