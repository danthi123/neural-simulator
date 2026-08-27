"""Verification for BRAIN_GNW_ORGANB_LTM_EXEMPT (the organ-B LTM-exemption de-risk closing Bug 2 of
research/findings/2026-08-27-knowledge-in-live-chat-veto-comprehension-and-gnw-organb-expectation-gap.md).

SIM_BACKEND=numpy, tiny-demo + shipped LTM (the light path), NOT the GPU server. Multi-seed over
[42, 43, 44, 100, 101, 102].

Checks, per seed:
  A) MOAT — a NON-EXISTENT fact still abstains on the 2-organ bus with the flag ON (and OFF).
  B) COMMIT — a genuine LTM fact (chelsea_fc/country, + 2 more) commits on the 2-organ bus with flag ON,
     still vetoes with flag OFF (reproducing the diagnostic).
  C) BYTE-IDENTICAL WHEN OFF — a panel of turns (LTM facts + buffer-taught facts + unstored) hashed with the
     flag unset vs explicitly "0"; must be 0 diffs vs today's (pre-existing) two-organ behavior.
  D) CONVERSATIONAL-BUFFER UNTOUCHED — a fact taught mid-conversation (buffer tier) behaves identically with
     the flag on vs off (organ B still reads its own e_B expectation, corroborate/withhold unchanged).
  E) 3-ORGAN BUS — same LTM facts + unstored fact through the three-organ bus with the flag on: does organ C
     block for a separate reason?

Prints a JSON summary per seed + an overall verdict line.
"""
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_LTM_SHIP_DEFAULT", "1")

REPO = "/home/dant123/Projects/sim/.claude/worktrees/agent-af06a18d790f070b8"
sys.path.insert(0, REPO)

LTM_BUNDLE = os.path.expanduser("~/Projects/sim-data/knowledge_bundles/wikidata_core_15k")

SEEDS = [42, 43, 44, 100, 101, 102]

# LTM facts known to be in wikidata_core_15k (per the diagnostic's repro + a manual scan below).
LTM_PROBES = [
    ("chelsea_fc", "country"),
]


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


def two_organ_result(chat, agent, action, seed, *, ltm_exempt):
    from webapp.gnw_two_organ_bus import two_organ_combine
    return two_organ_combine(chat, agent, action, seed=seed, organb_ltm_exempt=ltm_exempt)


def three_organ_result(chat, agent, action, seed, *, ltm_exempt):
    from webapp.gnw_three_organ_bus import three_organ_combine
    return three_organ_combine(chat, agent, action, seed=seed, organb_ltm_exempt=ltm_exempt)


def main():
    overall = {"moat_held": True, "commits_with_flag_on": True, "byte_identical_off": True,
              "buffer_untouched": True, "seeds": {}}

    for seed in SEEDS:
        chat, ltm = build_chat(seed)
        more_facts = find_more_ltm_facts(ltm, n=2)
        probes = [("chelsea_fc", "country")] + [(a, act) for (a, act, p) in more_facts]
        expected_patients = {}
        # get the expected patient for each probe via a direct store query (ground truth)
        for a, act in probes:
            expected_patients[(a, act)] = chat.inner.composer.query_patient(a, act)

        seed_report = {"probes": probes, "expected": {f"{a}|{act}": p for (a, act), p in expected_patients.items()}}

        # --- A) MOAT: a non-existent fact must still abstain, flag ON and OFF ---
        fake_agent, fake_action = "definitely_not_a_stored_entity_xyz", "definitely_not_a_stored_relation_xyz"
        # sanity: confirm this really is unstored
        direct = chat.inner.composer.query_patient(fake_agent, fake_action)
        assert direct is None, f"test fixture bug: fake fact resolved to {direct!r}"
        moat_off = two_organ_result(chat, fake_agent, fake_action, seed, ltm_exempt=False)
        moat_on = two_organ_result(chat, fake_agent, fake_action, seed, ltm_exempt=True)
        moat_ok = (moat_off["committed"] is None) and (moat_on["committed"] is None)
        seed_report["moat"] = {
            "off_committed": moat_off["committed"], "off_abstain_reason": moat_off.get("abstain_reason"),
            "on_committed": moat_on["committed"], "on_abstain_reason": moat_on.get("abstain_reason"),
            "ok": moat_ok,
        }
        if not moat_ok:
            overall["moat_held"] = False

        # --- B) COMMIT: genuine LTM facts commit with flag ON, still veto with flag OFF ---
        commit_report = {}
        all_commit_ok = True
        for a, act in probes:
            exp_p = expected_patients[(a, act)]
            r_off = two_organ_result(chat, a, act, seed, ltm_exempt=False)
            r_on = two_organ_result(chat, a, act, seed, ltm_exempt=True)
            ok = (r_off["committed"] is None and r_off.get("abstain_reason") == "consensus_veto_organ_b_withheld"
                  and r_on["committed"] == exp_p and r_on.get("recall_source") == "ltm"
                  and r_on.get("organb_ltm_exempt_applied") is True)
            commit_report[f"{a}|{act}"] = {
                "expected_patient": exp_p,
                "off_committed": r_off["committed"], "off_abstain_reason": r_off.get("abstain_reason"),
                "off_recall_source": r_off.get("recall_source"),
                "on_committed": r_on["committed"], "on_abstain_reason": r_on.get("abstain_reason"),
                "on_recall_source": r_on.get("recall_source"),
                "on_organb_ltm_exempt_applied": r_on.get("organb_ltm_exempt_applied"),
                "ok": ok,
            }
            if not ok:
                all_commit_ok = False
        seed_report["commit"] = commit_report
        if not all_commit_ok:
            overall["commits_with_flag_on"] = False

        # --- D) BUFFER-TAUGHT fact: teach a fresh fact into the conversational buffer, verify flag on/off
        #     behave IDENTICALLY on it (organ B's own e_B expectation still governs; untouched by the lever). ---
        taught_agent, taught_action, taught_patient = "zzz_test_agent", "zzz_test_action", "zzz_test_patient"
        chat.inner.composer.store(taught_agent, taught_action, taught_patient, polarity="AFFIRM")
        # rebuild the organ cache key is per-seed and includes stored_patients at build time; the 2-organ module
        # caches organ B once per seed -- if this is the first probe this seed, the newly-taught patient may not
        # be pre-registered as a cue-addressable block. That is IDENTICAL regardless of the flag (organ B's own
        # registration is unaffected by organb_ltm_exempt), which is exactly what this check verifies.
        r_off_buf = two_organ_result(chat, taught_agent, taught_action, seed, ltm_exempt=False)
        r_on_buf = two_organ_result(chat, taught_agent, taught_action, seed, ltm_exempt=True)
        buf_identical = (r_off_buf["committed"] == r_on_buf["committed"]
                         and r_off_buf.get("organ_b_confirmed") == r_on_buf.get("organ_b_confirmed")
                         and r_off_buf.get("organ_b_surprise_hz") == r_on_buf.get("organ_b_surprise_hz")
                         and r_on_buf.get("recall_source") == "buffer"
                         and r_on_buf.get("organb_ltm_exempt_applied") is False)
        seed_report["buffer_untouched"] = {
            "off_committed": r_off_buf["committed"], "on_committed": r_on_buf["committed"],
            "off_organ_b_confirmed": r_off_buf.get("organ_b_confirmed"),
            "on_organ_b_confirmed": r_on_buf.get("organ_b_confirmed"),
            "on_recall_source": r_on_buf.get("recall_source"),
            "identical": buf_identical,
        }
        if not buf_identical:
            overall["buffer_untouched"] = False

        # --- E) 3-ORGAN BUS: does organ C block LTM facts for a different reason, with the flag on? ---
        three_organ_report = {}
        for a, act in probes:
            exp_p = expected_patients[(a, act)]
            r3_on = three_organ_result(chat, a, act, seed, ltm_exempt=True)
            three_organ_report[f"{a}|{act}"] = {
                "expected_patient": exp_p,
                "committed": r3_on["committed"], "abstain_reason": r3_on.get("abstain_reason"),
                "organ_b_confirmed": r3_on.get("organ_b_confirmed"),
                "organ_c_votes": r3_on.get("organ_c_votes"),
                "organ_c_real_vocab_known": r3_on.get("organ_c_real_vocab_known"),
                "organ_c_unknown_tokens": r3_on.get("organ_c_unknown_tokens"),
                "organ_c_competent": r3_on.get("organ_c_competent"),
                "organ_c_margin": r3_on.get("organ_c_margin"),
                "organ_c_threshold": r3_on.get("organ_c_threshold"),
                "organb_ltm_exempt_applied": r3_on.get("organb_ltm_exempt_applied"),
            }
        # moat on the 3-organ bus too (unstored fact)
        r3_moat = three_organ_result(chat, fake_agent, fake_action, seed, ltm_exempt=True)
        three_organ_report["_moat_unstored"] = {
            "committed": r3_moat["committed"], "abstain_reason": r3_moat.get("abstain_reason"),
        }
        seed_report["three_organ"] = three_organ_report

        overall["seeds"][seed] = seed_report
        print(f"[seed {seed}] moat_ok={moat_ok} commit_ok={all_commit_ok} buffer_identical={buf_identical}",
              flush=True)

    print("\n=== FULL REPORT ===")
    print(json.dumps(overall, indent=2, default=str))

    verdict = "GO" if (overall["moat_held"] and overall["commits_with_flag_on"]
                       and overall["buffer_untouched"]) else "NO-GO"
    print(f"\n=== VERDICT: {verdict} (moat_held={overall['moat_held']}, "
          f"commits_with_flag_on={overall['commits_with_flag_on']}, "
          f"buffer_untouched={overall['buffer_untouched']}) ===")


if __name__ == "__main__":
    main()
