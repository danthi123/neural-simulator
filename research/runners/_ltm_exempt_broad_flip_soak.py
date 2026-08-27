"""BROAD no-regression + moat soak for the GNW-consensus LTM-exemption PRODUCTION FLIP
(`BRAIN_GNW_ORGANB_LTM_EXEMPT`, default-ON since
research/findings/2026-08-27-ltm-exempt-production-flip-knowledge-answers-live-by-default.md).

WHY THIS RUNNER. The two de-risks that earned the flip (organ B and organ C, both 6/6 GO) each checked a HANDFUL
of facts per seed: ONE anchor LTM probe (chelsea_fc/country) + 2 more pulled from shard 0 only, ONE fixed fake
fact for the moat, and ONE taught buffer fact. That was enough to justify flipping the default; now that it IS
the default, the flip needs BROADER confidence: many more genuine facts spread across the whole 15k/75-shard
corpus, many more synthetic nonexistent probes (the moat -- the critical property), and many more buffer-taught
facts, still on BOTH the 2-organ and 3-organ bus. This runner does NOT re-derive any check -- it calls the exact
SAME combine-level entry points the two de-risks called (`webapp.gnw_two_organ_bus.two_organ_combine` /
`webapp.gnw_three_organ_bus.three_organ_combine`), just at a much larger N, and packages the result as a proper
GO-gated finding artifact (`tools.verdict.Verdict`) instead of a one-off script + hand-copied verdict.

SIM_BACKEND=numpy, tiny-demo + the shipped LTM bundle (the identical light path both de-risks used). NO GPU.

Checks, per seed (`--n-ltm-facts` / `--n-fake-probes` / `--n-buffer-facts` control N; defaults 20/20/10):
  A) MOAT (the critical property) -- N_FAKE_PROBES synthetic (agent,action) pairs, none in the store (verified per
     probe by a direct `composer.query_patient` sanity check), must ABSTAIN with
     `abstain_reason == "primary_recall_miss"` on BOTH buses with the flag ON (today's production default) --
     organ A's own forward-recall miss short-circuits BEFORE organ B or C is ever consulted, so the exemption can
     never manufacture an answer the store does not hold.
  B) COMMIT -- N_LTM_FACTS genuine (agent,action,patient) triples, pulled round-robin ACROSS the LTM's 75 shards
     (not just shard 0, unlike the de-risks' n=2 sample), must COMMIT (== the expected patient, `recall_source ==
     "ltm"`, the exemption flag applied) on BOTH buses with the flag ON, and must VETO (abstain, organ B/organ A
     withheld) with the flag OFF -- reproducing today's pre-flip behavior on the SAME facts.
  C) BUFFER UNTOUCHED -- N_BUFFER_FACTS freshly-taught conversational-buffer facts (a fresh "turn" each) must
     behave BYTE-IDENTICALLY flag on vs off (`committed` / `organ_b_confirmed` / `organ_b_surprise_hz` /
     `organ_c_votes` / `organ_c_real_vocab_known` all equal; `recall_source == "buffer"`; neither exemption flag
     ever applied) on BOTH buses -- the lever must never touch a recent-conversation recall.
  D) THE `=0` ESCAPE -- a direct, substrate-free env-level check: `BRAIN_GNW_ORGANB_LTM_EXEMPT` genuinely UNSET
     reads True (today's production default); explicit `"0"` reads False (the byte-identical escape back to
     pre-flip behavior); explicit `"1"` reads True.

VERDICT. GO iff, across every seed: the moat holds on EVERY fake probe (zero breaches -- any single moat breach is
an automatic NO-GO, flagged loud, because it would mean the exemption manufactured an answer for something the
brain never stored), EVERY genuine LTM fact commits ON / vetoes OFF on both buses, EVERY buffer-taught fact is
byte-identical on/off on both buses, and the env-level escape reads correctly.

Run (CPU cheap-first; EXPORT OMP/OPENBLAS/MKL=4):
    SIM_BACKEND=numpy OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u \
        -m research.runners._ltm_exempt_broad_flip_soak \
        --seeds 42 43 44 100 101 102 --n-ltm-facts 20 --n-fake-probes 20 --n-buffer-facts 10 \
        --json research/findings/raw/_ltm_exempt_broad_flip_soak.json

Pool-stagable (`tools/sweep_pool.sh`) once this runner is reachable from `main`: it is a headless numpy CLI with
no local state beyond the shipped LTM bundle, so a seed (or an N-axis) sweep round-robins cleanly across pool
nodes -- see the finding this runner ships with for the exact ready-to-run command.
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_LTM_SHIP_DEFAULT", "1")

import argparse
import json
import random

from tools.verdict import Verdict

LTM_BUNDLE_DEFAULT = os.path.expanduser("~/Projects/sim-data/knowledge_bundles/wikidata_core_15k")
DEFAULT_SEEDS = [42, 43, 44, 100, 101, 102]
ANCHOR_PROBE = ("chelsea_fc", "country")   # the de-risks' own anchor fact -- always included first


def build_chat(seed: int, ltm_bundle: str = LTM_BUNDLE_DEFAULT):
    """Identical build to both de-risks' `build_chat`: the tiny-demo brain + `TieredFactStore(buffer, ltm)`."""
    from research.runners.brain_chat_tui import ChatBrain, StubRenderer, _build_tiny_demo
    from research.runners.developed_brain_io import _inner_agent
    from research.runners.tiered_fact_store import TieredFactStore
    from research.runners.sharded_phasor_store import ShardedPhasorStore

    agent, aliases, _n = _build_tiny_demo(seed, use_multiturn=True, enable_neural_render=False,
                                          composer_kind="onebrain")
    ltm = ShardedPhasorStore.load(ltm_bundle)
    inner = _inner_agent(agent)
    inner.composer = TieredFactStore(inner.composer, ltm)
    chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())
    return chat, ltm


def find_ltm_facts_broad(ltm, n: int, *, exclude_agents=("chelsea_fc",), rng=None):
    """Pull up to `n` genuine (agent, action, patient) triples SPREAD ACROSS the LTM's 75 shards (round-robin one
    per shard per pass), broadening the de-risks' own `find_more_ltm_facts` (which only ever walked shard 0's
    local cluster). Always leads with the de-risks' own anchor probe (chelsea_fc/country)."""
    out = [ANCHOR_PROBE]
    seen = {ANCHOR_PROBE}
    shards = list(ltm.shards)
    if rng is not None:
        rng.shuffle(shards)
    idxs = [0] * len(shards)
    progressed = True
    while len(out) < n and progressed:
        progressed = False
        for si, sh in enumerate(shards):
            if len(out) >= n:
                break
            kb = sh.kb
            i = idxs[si]
            while i < len(kb):
                fact, _comp = kb[i]
                i += 1
                a, act, p = fact.get("agent"), fact.get("action"), fact.get("patient")
                if isinstance(a, str) and isinstance(act, str) and isinstance(p, str) \
                        and a not in exclude_agents and (a, act) not in seen:
                    seen.add((a, act))
                    out.append((a, act))
                    progressed = True
                    break
            idxs[si] = i
    return out[:n]


def fake_probes(n: int, *, seed: int):
    """N synthetic (agent,action) pairs guaranteed absent from any store -- broadens the de-risks' single fixed
    fake fact into a real breadth check on the moat (the critical property this flip must never regress)."""
    rng = random.Random(1_000_003 * int(seed) + 7)
    return [(f"definitely_not_a_stored_entity_{seed}_{i}_{rng.randrange(10**9)}",
             f"definitely_not_a_stored_relation_{seed}_{i}_{rng.randrange(10**9)}")
            for i in range(n)]


def buffer_facts(n: int, *, seed: int):
    return [(f"zzz_buf_agent_{seed}_{i}", f"zzz_buf_action_{seed}_{i}", f"zzz_buf_patient_{seed}_{i}")
            for i in range(n)]


def two_organ(chat, agent, action, seed, *, ltm_exempt):
    from webapp.gnw_two_organ_bus import two_organ_combine
    return two_organ_combine(chat, agent, action, seed=seed, organb_ltm_exempt=ltm_exempt)


def three_organ(chat, agent, action, seed, *, ltm_exempt):
    from webapp.gnw_three_organ_bus import three_organ_combine
    return three_organ_combine(chat, agent, action, seed=seed, organb_ltm_exempt=ltm_exempt)


def escape_check():
    """D) The `=0` escape, direct env-level (no substrate build): unset -> default-ON (True); '0' -> OFF (False);
    '1' -> ON (True). Restores whatever the caller's env held on the way out."""
    from webapp.gnw_two_organ_bus import organb_ltm_exempt_enabled
    saved = os.environ.pop("BRAIN_GNW_ORGANB_LTM_EXEMPT", None)
    try:
        unset_val = organb_ltm_exempt_enabled()
        os.environ["BRAIN_GNW_ORGANB_LTM_EXEMPT"] = "0"
        off_val = organb_ltm_exempt_enabled()
        os.environ["BRAIN_GNW_ORGANB_LTM_EXEMPT"] = "1"
        on_val = organb_ltm_exempt_enabled()
    finally:
        os.environ.pop("BRAIN_GNW_ORGANB_LTM_EXEMPT", None)
        if saved is not None:
            os.environ["BRAIN_GNW_ORGANB_LTM_EXEMPT"] = saved
    ok = bool(unset_val is True and off_val is False and on_val is True)
    return {"unset_is_default_on": unset_val, "explicit_0_is_off": off_val, "explicit_1_is_on": on_val, "ok": ok}


def evaluate_seed(seed: int, *, n_ltm_facts: int, n_fake_probes: int, n_buffer_facts: int,
                  ltm_bundle: str, verbose: bool = True) -> dict:
    chat, ltm = build_chat(seed, ltm_bundle)
    rng = random.Random(seed)
    ltm_probes = find_ltm_facts_broad(ltm, n_ltm_facts, rng=rng)
    fakes = fake_probes(n_fake_probes, seed=seed)
    bufs = buffer_facts(n_buffer_facts, seed=seed)

    expected = {(a, act): chat.inner.composer.query_patient(a, act) for a, act in ltm_probes}
    # a fact the direct store read itself abstains on is the DOCUMENTED separate key-routing gap the flip's own
    # finding flags (country entities keyed *_portal/*_core) -- not this arc's defect; exclude it from the commit
    # check (this arc verifies the CONSENSUS combine, not LTM key coverage) but report the drop count honestly.
    n_dropped_unresolvable = sum(1 for v in expected.values() if v is None)
    ltm_probes = [(a, act) for a, act in ltm_probes if expected[(a, act)] is not None]

    # --- A) MOAT: every fake probe abstains (primary_recall_miss) on BOTH buses, flag ON (today's default) ---
    moat_rows, moat_all_ok = [], True
    for fa, fac in fakes:
        direct = chat.inner.composer.query_patient(fa, fac)
        assert direct is None, f"test fixture bug: fake fact resolved to {direct!r}"
        r2 = two_organ(chat, fa, fac, seed, ltm_exempt=True)
        r3 = three_organ(chat, fa, fac, seed, ltm_exempt=True)
        ok = (r2["committed"] is None and r2.get("abstain_reason") == "primary_recall_miss"
              and r3["committed"] is None and r3.get("abstain_reason") == "primary_recall_miss")
        moat_rows.append({"agent": fa, "action": fac,
                          "two_organ_committed": r2["committed"], "two_organ_abstain": r2.get("abstain_reason"),
                          "three_organ_committed": r3["committed"], "three_organ_abstain": r3.get("abstain_reason"),
                          "ok": ok})
        moat_all_ok = moat_all_ok and ok

    # --- B) COMMIT: every genuine LTM fact commits (flag ON) / vetoes (flag OFF), BOTH buses ---
    commit_rows, commit_all_ok = [], True
    for a, act in ltm_probes:
        exp = expected[(a, act)]
        r2_on = two_organ(chat, a, act, seed, ltm_exempt=True)
        r2_off = two_organ(chat, a, act, seed, ltm_exempt=False)
        r3_on = three_organ(chat, a, act, seed, ltm_exempt=True)
        r3_off = three_organ(chat, a, act, seed, ltm_exempt=False)
        ok = (r2_on["committed"] == exp and r2_on.get("recall_source") == "ltm"
              and r2_on.get("organb_ltm_exempt_applied") is True and r2_off["committed"] is None
              and r3_on["committed"] == exp and r3_on.get("recall_source") == "ltm"
              and r3_on.get("organb_ltm_exempt_applied") is True and r3_on.get("organ_c_ltm_exempt_applied") is True
              and r3_off["committed"] is None)
        commit_rows.append({"agent": a, "action": act, "expected": exp,
                            "two_organ_on": r2_on["committed"], "two_organ_off": r2_off["committed"],
                            "three_organ_on": r3_on["committed"], "three_organ_off": r3_off["committed"],
                            "two_organ_off_abstain": r2_off.get("abstain_reason"),
                            "three_organ_off_abstain": r3_off.get("abstain_reason"), "ok": ok})
        commit_all_ok = commit_all_ok and ok

    # --- C) BUFFER: every taught buffer fact behaves byte-identically flag on/off, BOTH buses ---
    buf_rows, buf_all_ok = [], True
    for ta, tac, tp in bufs:
        chat.inner.composer.store(ta, tac, tp, polarity="AFFIRM")
        r2_off = two_organ(chat, ta, tac, seed, ltm_exempt=False)
        r2_on = two_organ(chat, ta, tac, seed, ltm_exempt=True)
        r3_off = three_organ(chat, ta, tac, seed, ltm_exempt=False)
        r3_on = three_organ(chat, ta, tac, seed, ltm_exempt=True)
        two_id = (r2_off["committed"] == r2_on["committed"]
                  and r2_off.get("organ_b_confirmed") == r2_on.get("organ_b_confirmed")
                  and r2_off.get("organ_b_surprise_hz") == r2_on.get("organ_b_surprise_hz")
                  and r2_on.get("recall_source") == "buffer" and r2_on.get("organb_ltm_exempt_applied") is False)
        three_id = (r3_off["committed"] == r3_on["committed"]
                    and r3_off.get("organ_b_confirmed") == r3_on.get("organ_b_confirmed")
                    and r3_off.get("organ_c_votes") == r3_on.get("organ_c_votes")
                    and r3_off.get("organ_c_real_vocab_known") == r3_on.get("organ_c_real_vocab_known")
                    and r3_on.get("recall_source") == "buffer" and r3_on.get("organb_ltm_exempt_applied") is False
                    and r3_on.get("organ_c_ltm_exempt_applied") is False)
        ok = bool(two_id and three_id)
        buf_rows.append({"agent": ta, "action": tac, "two_organ_identical": bool(two_id),
                         "three_organ_identical": bool(three_id), "ok": ok})
        buf_all_ok = buf_all_ok and ok

    seed_go = bool(moat_all_ok and commit_all_ok and buf_all_ok)
    result = {
        "seed": int(seed), "seed_go": seed_go, "n_dropped_unresolvable_ltm_probes": n_dropped_unresolvable,
        "moat": {"all_ok": moat_all_ok, "n": len(moat_rows), "rows": moat_rows},
        "commit": {"all_ok": commit_all_ok, "n": len(commit_rows), "rows": commit_rows},
        "buffer": {"all_ok": buf_all_ok, "n": len(buf_rows), "rows": buf_rows},
    }
    if verbose:
        print(f"[seed {seed}] moat_ok={moat_all_ok} ({len(moat_rows)} fake probes) | "
              f"commit_ok={commit_all_ok} ({len(commit_rows)} LTM facts, "
              f"{n_dropped_unresolvable} dropped-unresolvable) | "
              f"buffer_ok={buf_all_ok} ({len(buf_rows)} taught facts) | seed_go={seed_go}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description="Broad no-regression + moat soak for the LTM-exemption production "
                                              "flip (BRAIN_GNW_ORGANB_LTM_EXEMPT default-ON).")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--n-ltm-facts", type=int, default=20)
    ap.add_argument("--n-fake-probes", type=int, default=20)
    ap.add_argument("--n-buffer-facts", type=int, default=10)
    ap.add_argument("--ltm-bundle", type=str, default=LTM_BUNDLE_DEFAULT)
    ap.add_argument("--json", type=str, default="research/findings/raw/_ltm_exempt_broad_flip_soak.json")
    args = ap.parse_args()

    print(f"[ltm-exempt broad flip-soak] seeds={args.seeds} n_ltm_facts={args.n_ltm_facts} "
          f"n_fake_probes={args.n_fake_probes} n_buffer_facts={args.n_buffer_facts} "
          f"backend={os.environ.get('SIM_BACKEND')}\n", flush=True)

    esc = escape_check()
    print(f"[escape-check] unset={esc['unset_is_default_on']} '0'={esc['explicit_0_is_off']} "
          f"'1'={esc['explicit_1_is_on']} ok={esc['ok']}\n", flush=True)

    results = [evaluate_seed(s, n_ltm_facts=args.n_ltm_facts, n_fake_probes=args.n_fake_probes,
                             n_buffer_facts=args.n_buffer_facts, ltm_bundle=args.ltm_bundle, verbose=True)
               for s in args.seeds]

    n = len(results)
    all_moat_ok = all(r["moat"]["all_ok"] for r in results)
    all_commit_ok = all(r["commit"]["all_ok"] for r in results)
    all_buffer_ok = all(r["buffer"]["all_ok"] for r in results)
    n_seed_go = sum(int(r["seed_go"]) for r in results)

    n_moat_probes_total = sum(r["moat"]["n"] for r in results)
    n_moat_probes_ok = sum(sum(1 for row in r["moat"]["rows"] if row["ok"]) for r in results)
    n_commit_facts_total = sum(r["commit"]["n"] for r in results)
    n_commit_facts_ok = sum(sum(1 for row in r["commit"]["rows"] if row["ok"]) for r in results)
    n_buffer_facts_total = sum(r["buffer"]["n"] for r in results)
    n_buffer_facts_ok = sum(sum(1 for row in r["buffer"]["rows"] if row["ok"]) for r in results)

    flip_go = bool(all_moat_ok and all_commit_ok and all_buffer_ok and esc["ok"])

    v = Verdict("LTM-exemption production flip (BRAIN_GNW_ORGANB_LTM_EXEMPT) BROAD no-regression + moat soak "
               "(%d seeds, %d nonexistent probes, %d LTM facts, %d buffer facts)"
               % (n, n_moat_probes_total, n_commit_facts_total, n_buffer_facts_total))
    for r in results:
        seed = r["seed"]
        v.require(f"seed {seed}: MOAT holds on all {r['moat']['n']} nonexistent probes "
                  f"(primary_recall_miss, both buses)", r["moat"]["all_ok"], expect=True)
        v.require(f"seed {seed}: all {r['commit']['n']} genuine LTM facts commit flag-ON / veto flag-OFF, "
                  f"both buses", r["commit"]["all_ok"], expect=True)
        v.require(f"seed {seed}: all {r['buffer']['n']} buffer-taught facts byte-identical flag on/off, "
                  f"both buses", r["buffer"]["all_ok"], expect=True)
    v.require("env-level '=0' escape: unset=default-ON, explicit '0'=OFF, explicit '1'=ON", esc["ok"], expect=True)
    decided = v.decide(go=flip_go, verbose=True)
    verdict = decided["status"]
    if verdict == "NO-GO" and any(not r["moat"]["all_ok"] for r in results):
        verdict = "NO-GO-MOAT-BREACH"   # loud, distinct marker -- would gate a re-review of the production flip

    summary = {
        "runner": "_ltm_exempt_broad_flip_soak", "verdict": verdict, "flip_go": flip_go,
        "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
        "n_seeds": n, "n_seed_go": n_seed_go, "seeds": list(args.seeds),
        "escape_check": esc,
        "moat": {"all_ok": all_moat_ok, "n_probes_total": n_moat_probes_total, "n_probes_ok": n_moat_probes_ok},
        "commit": {"all_ok": all_commit_ok, "n_facts_total": n_commit_facts_total,
                  "n_facts_ok": n_commit_facts_ok},
        "buffer": {"all_ok": all_buffer_ok, "n_facts_total": n_buffer_facts_total,
                  "n_facts_ok": n_buffer_facts_ok},
        "config": {"n_ltm_facts": args.n_ltm_facts, "n_fake_probes": args.n_fake_probes,
                  "n_buffer_facts": args.n_buffer_facts, "ltm_bundle": args.ltm_bundle},
        "flag": "BRAIN_GNW_ORGANB_LTM_EXEMPT (production DEFAULT-ON since 2026-08-27; governs organ B on the "
               "2-organ bus AND organ C on the 3-organ bus via the SAME flag)",
        "per_seed": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 100}", flush=True)
    print(f"  LTM-EXEMPT BROAD FLIP-SOAK VERDICT: {verdict}  "
          f"(moat {n_moat_probes_ok}/{n_moat_probes_total} · commit {n_commit_facts_ok}/{n_commit_facts_total} · "
          f"buffer {n_buffer_facts_ok}/{n_buffer_facts_total} · escape_ok={esc['ok']} · "
          f"seed_go {n_seed_go}/{n})", flush=True)
    if not all_moat_ok:
        print("  !!!! MOAT BREACH DETECTED -- a nonexistent/unstored fact did NOT abstain on at least one probe. "
              "This gates a re-review of the production flip. !!!!", flush=True)
    print(f"    [saved] {args.json}\n{'=' * 100}", flush=True)
    return 0 if flip_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
