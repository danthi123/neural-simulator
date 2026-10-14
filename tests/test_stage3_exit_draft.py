"""Stage-3 exit gate DRAFT (research/runners/_stage3_exit_draft.py): the frozen turn tables pinned against the
PRODUCTION parsers, the scorers on synthetic records, the two-clause aggregate, the gate-seed refusal and the pool
lines. No brain build (the live arms are the dev-7 smoke and, after registration, the six-seed gate)."""
import json
import os
import re
import subprocess
import sys
import tempfile

import pytest

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import _stage3_exit_draft as G  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DRAFT = os.path.join(ROOT, "docs", "drafts", "stage3-exit")
_LTM_BUNDLES = ("wikidata_100k", "wikidata_core_15k")


def _all_tables(group):
    d = G.GROUPS[group]
    return [d["turns"], d["permuted_turns"]] + list(d.get("variants", {}).values())


# ── structure ────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_selftest_passes():
    ok, checks = G.selftest(verbose=False)
    assert ok, {k: v for k, v in checks.items() if not v}


def test_draft_is_unregistered_and_points_at_the_committed_plan():
    assert G.PREREG_REGISTERED is False
    assert os.path.exists(os.path.join(ROOT, G.PREREG))
    with open(os.path.join(ROOT, G.PREREG)) as f:
        plan = f.read()
    assert "M_ADJ_TOLD_FIX_IN_PIN" in plan and "host-decision debt" in plan and "DRAFT PREREGISTRATION TEXT" in plan


def test_one_primary_group_per_criterion_and_every_group_has_the_arm_kinds():
    assert sorted(G.GROUPS[g]["criterion"] for g in G.PRIMARY_GROUPS) == list(G.CRITERIA)
    for g in G.GROUPS:
        arms = G.arms_for(g)
        assert arms[:2] == list(G._intact_names(g)) and "permuted" in arms and G.STEER_ARM[g] in arms, (g, arms)
        kinds = {G.arm_spec(g, a)[2] for a in arms}
        assert {"intact", "lesion", "permuted"} <= kinds, (g, kinds)
        labels = {t[0] for t in G.GROUPS[g]["turns"]}
        for les in G.GROUPS[g]["lesions"]:
            assert les in G.SCORED_LABELS[g] and set(G.SCORED_LABELS[g][les]) <= labels, (g, les)
            assert les + ".rep0" in arms, (g, les)
        # ruling 2: every criterion keeps >= 1 content-specific intervention (steer / G) beside its silencing lesion
        assert any(v["kind"] in ("steer", "G") for v in G.GROUPS[g]["interventions"].values()), g


def test_permuted_and_variant_tables_keep_every_label_and_change_only_teaching():
    for g in G.GROUPS:
        base = G.GROUPS[g]["turns"]
        for tab in _all_tables(g)[1:]:
            assert [(l, r) for l, _m, r in base] == [(l, r) for l, _m, r in tab], g
            changed = [(l, r) for (l, m1, r), (_l, m2, _r) in zip(base, tab) if m1 != m2]
            assert changed and all(r in ("teach", "expose", "episode") for _l, r in changed), (g, changed)


def test_lesion_env_flags_are_read_by_production_code():
    files = []
    for sub in ("research/runners", "webapp"):
        d = os.path.join(ROOT, sub)
        files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".py")]
    blob = ""
    for f in files:
        with open(f, errors="ignore") as fh:
            blob += fh.read()
    for g, d in G.GROUPS.items():
        for k in d["env"]:
            assert k in blob, (g, k)
        for name, spec in list(d["lesions"].items()) + list(d["informational"].items()):
            assert spec["env"] or spec["patch"] or spec.get("variant"), (g, name)
            for k in spec["env"]:
                assert k in blob, (g, name, k)
            if name in d["lesions"]:
                assert spec["cuts"], (g, name)


# ── turn pins against the PRODUCTION parsers ─────────────────────────────────────────────────────────────────────────
def test_chain_probes_parse_as_production_reads_them():
    from webapp.reasoning_transitive_chat import parse_transitive_question
    from webapp.gnw_multistep_deliberation import detect_chase
    want_yesno = {"p3": ("puma", "chase", "flea"), "p2": ("puma", "chase", "slug"), "m_adj": ("puma", "chase", "gecko"),
                  "m_cross": ("shrew", "chase", "flea"), "m_rev": ("flea", "chase", "puma")}
    want_chase = {"r_fork": "puma", "r_all": "gecko", "c_tail": "flea"}
    for tab in _all_tables("chain"):
        msgs = {l: m for l, m, _r in tab}
        for l, t in want_yesno.items():
            assert parse_transitive_question(msgs[l]) == t, (l, msgs[l])
        for l, agent in want_chase.items():
            c = detect_chase(msgs[l])
            assert c and c.endswith("?") and "all the way" not in c and agent in c.split(), (l, c)
            assert parse_transitive_question(msgs[l]) is None, l
        for l in ("c_direct", "c_dog"):
            assert detect_chase(msgs[l]) is None and parse_transitive_question(msgs[l]) is None, l
        for l, m, r in tab:
            if r == "teach":
                assert detect_chase(m) is None and parse_transitive_question(m) is None, (l, m)


def test_teach_turns_are_three_token_assertions_and_probes_are_not():
    """The surprise organ's `extract_assertion` reads a bare `does X R Y` (no '?') as a 3-token assertion, so the
    C1 yes/no probes are claimed by the transitive route FIRST; every other non-teach turn is no assertion at all
    (PLAN.md section 10: the dev-7 smoke checks no probe is acquired as a fact)."""
    from research.runners.surprise_production_organ import extract_assertion
    from webapp.reasoning_transitive_chat import parse_transitive_question
    for g in ("chain", "meta", "credit"):
        for tab in _all_tables(g):
            for l, m, r in tab:
                if m == G.IDLE:
                    continue
                if r == "teach" and " in the " not in m:
                    a = extract_assertion(m)
                    assert a is not None and a[0] == m.split()[1] and a[1] in G.VERB_3SG.values(), (g, l, m, a)
                elif r != "teach" and parse_transitive_question(m) is None:
                    assert extract_assertion(m) is None, (g, l, m)


def test_tom_turns_classify_as_production_reads_them():
    from webapp.false_belief_chat import classify
    kinds = {"place": "place", "move": "move", "leave": "leave", "return": "return"}
    expect_q = {"q1": "sally", "q2": "tom", "q3": "lea", "q4": "kate", "m1": "ben"}
    for tab in _all_tables("tom"):
        for l, m, r in tab:
            c = classify(m)
            if r == "expose" and m != G._FILLER:
                kind = next(k for k in kinds if re.search(r"_%s\d?$" % k, l))
                assert c is not None and c["kind"] == kinds[kind], (l, m, c)
                if kind in ("place", "move"):
                    assert c["location"] == m.rstrip(".").split()[-1], (l, c)
            elif l in expect_q:
                assert c is not None and c["kind"] == "query" and c["agent"].lower() == expect_q[l], (l, c)
            else:
                assert c is None, (l, m, c)
    assert {G.TOM_EXPECT[q] for q in G.TOM_QUERIES} != {G.TOM_EXPECT_PERM[q] for q in G.TOM_QUERIES}
    # counterbalanced: S2's belief is the SECOND-named location; S3 is the true-belief control
    assert G.TOM_EXPECT["q2"] == "jar" and G.TOM_EXPECT["q3"] == "tin"


def test_only_the_idle_message_is_a_self_initiation_trigger():
    from research.runners.self_initiated_production_organ import is_selfinit_trigger
    assert is_selfinit_trigger(G.IDLE)
    for g in G.GROUPS:
        for tab in _all_tables(g):
            for l, m, _r in tab:
                if m != G.IDLE:
                    assert not is_selfinit_trigger(m), (g, l, m)


def test_credit_held_out_item_is_never_mentioned_by_a_correction():
    msgs = {l: m for l, m, _r in G.CREDIT_TURNS}
    assert "canary" in msgs["H"]
    for tab in _all_tables("credit"):
        for l, m, r in tab:
            if l.startswith("k"):
                assert "canary" not in m and "jay" not in m, (l, m)
    # the held-out item shares the corrected items' second frame (hunts the wasp), the control does not
    assert msgs["x_canary_hunt"].endswith("the wasp") and msgs["x_jay_hunt"].endswith("the cricket")


def test_novel_words_are_absent_from_the_tiny_demo_vocabulary_and_the_shipped_ltm():
    tiny = {"brain", "use", "spikes", "learn", "words", "store", "memory", "dog", "chase", "cat", "eat", "fish"}
    assert not (set(G.NOVEL_WORDS) & tiny)
    paths = [os.path.expanduser("~/Projects/sim-data/knowledge_bundles/%s/facts.json" % b) for b in _LTM_BUNDLES]
    present = [p for p in paths if os.path.exists(p)]
    if not present:
        pytest.skip("shipped LTM bundle not present on this machine")
    for p in present:
        with open(p, errors="ignore") as f:
            blob = f.read().lower()
        hits = [w for w in G.NOVEL_WORDS if re.search(r"\b%s\b" % re.escape(w), blob)]
        assert not hits, (p, hits)


# ── scorers on synthetic records ─────────────────────────────────────────────────────────────────────────────────────
def test_derived_lead_strings_come_from_production():
    assert G.DERIVED_LEX_FROM_PRODUCTION is True
    from research.runners.compositional_chain_route import frame_derived_answer
    assert frame_derived_answer("X", [["a", "v", "b"]]).startswith(G.DERIVED_LEADS[0])
    assert G.DERIVED_LEADS[1] in frame_derived_answer("X", [])
    assert G.DERIVED_SWAP_TAIL.startswith("but I reasoned that myself")


def test_derived_framed_is_scored_from_the_record_and_the_lead_precedes_the_claim():
    path = ["puma", "gecko", "mantis", "flea"]
    ok = G._derived("puma", "chase", path)
    assert G.derived_framed(ok, "puma", "flea", n_hops=3)
    assert G.derived_framed(ok, "puma", "flea", max_hops=3)
    assert not G.derived_framed(ok, "puma", "flea", n_hops=2)                 # hop count must match the chase
    assert G.derived_framed(G._derived("puma", "chase", path, lead=1), "puma", "flea")     # the generic fallback lead
    assert G.derived_framed(G._derived("puma", "chase", path, swap=True), "puma", "flea")  # the provenance swap wording
    assert not G.derived_framed(G._plain("puma", "chase", "flea"), "puma", "flea")          # a told fact
    claim = G._fx("puma", "chase", "flea")
    hops = [list(h) for h in zip(path[:-1], ["chase"] * 3, path[1:])]
    after = G.R(claim + ". " + G.DERIVED_LEADS[0] + " x.", None, derived=True, derived_from=hops)
    assert not G.derived_framed(after, "puma", "flea")                       # lead AFTER the claim
    unled = G.R(claim, None, derived=True, derived_from=hops)
    assert not G.derived_framed(unled, "puma", "flea")                       # derived in the record, unmarked in the reply
    svo = G.R(G.DERIVED_LEADS[0] + " x. " + claim, ["puma", "chase", "flea"], derived=True, derived_from=hops)
    assert not G.derived_framed(svo, "puma", "flea")                         # recalled_svo set: not a derivation
    wrong_agent = G.R(G.DERIVED_LEADS[0] + " x. " + claim, None, derived=True, derived_from=hops[1:])
    assert not G.derived_framed(wrong_agent, "puma", "flea")
    sf = G.R(G.DERIVED_LEADS[0] + " x. " + claim, None, derived=True, derived_from=None, supporting_facts=hops)
    assert not G.derived_framed(sf, "puma", "flea")                          # never supporting_facts


def test_level_and_type2_auroc():
    assert G.level(G.R(G._AB, abst=True)) == 0
    assert G.level(G.R(G.HEDGE_TEXT + " Maybe the bat eats the grub -- that's a guess.")) == 1
    assert G.level(G.R("Maybe the lark eats the grub -- that's a guess.")) == 2
    assert G.type2_auroc([2, 2, 2, 2, 0, 0, 2, 2], [1, 1, 1, 1, 0, 0, 0, 0]) == 0.75
    assert G.type2_auroc([2] * 8, [1, 1, 1, 1, 0, 0, 0, 0]) == 0.5
    assert G.type2_auroc([2, 2, 2, 2, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]) == 1.0
    assert G.type2_auroc([2, 2], [1, 1]) is None


def test_each_named_null_fails_its_own_clause():
    sc = {k: G.score_chain(G.synthetic_shard("chain", 1, "intact_a", k)) for k in ("perfect", "main", "closure", "unmarked", "single_branch_fork")}
    assert sc["perfect"]["all"] and sc["perfect"]["via_workspace"]
    assert not sc["main"]["p3"] and not sc["main"]["fork_honest"]
    assert sc["closure"]["p3"] and not sc["closure"]["fork_honest"]
    assert not sc["unmarked"]["p3"]
    assert not sc["single_branch_fork"]["fork_honest"]
    sm = {k: G.score_meta(G.synthetic_shard("meta", 1, "intact_a", k)) for k in ("perfect", "never_abstains", "all_abstain", "perfect_hedge")}
    assert sm["perfect"]["all"] and sm["perfect"]["auroc"] == 1.0
    assert not sm["never_abstains"]["all"] and sm["never_abstains"]["auroc"] == 0.5
    assert not sm["all_abstain"]["capability_present"]
    assert sm["perfect_hedge"]["all"]
    st = {k: G.score_tom(G.synthetic_shard("tom", 1, "intact_a", k)) for k in ("perfect", "first_location", "present_means_reality")}
    assert st["perfect"]["all"]
    assert not st["first_location"]["q2"] and not st["first_location"]["q3"] and st["first_location"]["q1"]
    assert not st["present_means_reality"]["q4"]
    scr = {k: G.score_credit(G.synthetic_shard("credit", 1, "corrected_a", k)) for k in ("perfect", "main", "recent_correction")}
    assert scr["perfect"]["all"] and not scr["main"]["H"] and not scr["recent_correction"]["J"]


def test_m_adj_expectation_depends_on_the_told_directly_fix_in_the_pin():
    """`m_adj` (a TOLD one-hop fact asked as a yes/no) is counted only when the pin carries the told-directly fix
    (`M_ADJ_TOLD_FIX_IN_PIN`); before it, production at main abstains there and the clause is reported, not counted."""
    perfect = G.synthetic_shard("chain", 1, "intact_a", "perfect", m_adj_fix=True)
    assert "m_adj" in G.score_chain(perfect, m_adj_fix=True) and G.score_chain(perfect, m_adj_fix=True)["all"]
    main_unfixed = G.synthetic_shard("chain", 1, "intact_a", "main", m_adj_fix=False)
    s = G.score_chain(main_unfixed, m_adj_fix=False)
    assert "m_adj" not in s and s["m_adj_told_plain"] is False                   # reported, not counted
    assert G.score_chain(main_unfixed, m_adj_fix=True)["m_adj"] is False          # counted once the fix is in the pin
    main_fixed = G.synthetic_shard("chain", 1, "intact_a", "main", m_adj_fix=True)
    assert G.score_chain(main_fixed, m_adj_fix=True)["m_adj"] is True
    assert G.M_ADJ_TOLD_FIX_IN_PIN is False                                       # s/main at 1187f7bee: 20614ef7d not merged


# ── the two-clause aggregate ─────────────────────────────────────────────────────────────────────────────────────────
def _agg(td, seeds, kinds):
    import contextlib
    import io
    G.write_synthetic_tree(td, seeds, kinds)
    with contextlib.redirect_stdout(io.StringIO()):
        return G.aggregate(td, seeds=seeds)


def test_aggregate_two_clause_verdicts_and_the_host_decision_debt_panel():
    seeds = (1, 2)
    with tempfile.TemporaryDirectory() as td:
        agg = _agg(td, seeds, {})
        assert all(v == "PASS" for v in agg["criteria"].values()) and agg["verdict"] == "PASS" and agg["GO"] is True
        assert agg["panel"]["host_decision_share"] == 0.0
        assert any("derived" in d for d in agg["panel"]["host_decision_debt"])     # derived_framed: honesty, never ownership
        assert not agg["groups"]["chain"]["ownership_why"]
        agg = _agg(td, seeds, {"chain": "order_dependent", "tom": "host_written_belief"})
        assert agg["groups"]["chain"]["verdict"] == "CAPABILITY PASS, HOST-OWNED"
        assert agg["groups"]["chain"]["capability"] == "PASS" and agg["groups"]["chain"]["ownership"] == "HOST"
        assert agg["groups"]["tom"]["ownership"] == "HOST" and agg["verdict"] != "PASS"
        agg = _agg(td, seeds, {"chain": "lesion_inert", "meta": "conflict_counter"})
        assert agg["groups"]["chain"]["verdict"] == "FAIL" and any("not load-bearing" in w for w in agg["groups"]["chain"]["why"])
        assert agg["groups"]["meta"]["ownership"] == "HOST"
        os.remove(G.shard_path(td, 2, "meta", G.STEER_ARM["meta"]))
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            agg = G.aggregate(td, seeds=seeds)
        assert agg["groups"]["meta"]["verdict"] == "INCOMPLETE" and agg["verdict"].startswith("UNDEFINED")


def test_defensive_patch_not_applied_reads_undefined_not_fail():
    seeds = (1,)
    with tempfile.TemporaryDirectory() as td:
        G.write_synthetic_tree(td, seeds, {})
        p = G.shard_path(td, 1, "chain", "steer_leaf")
        with open(p) as f:
            sh = json.load(f)
        sh["patch"] = {"patch": "steer_leaf", "calls": 0, "n_applied": 0, "events": []}
        with open(p, "w") as f:
            json.dump(sh, f)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            agg = G.aggregate(td, seeds=seeds)
        assert agg["groups"]["chain"]["verdict"] == "INCOMPLETE", agg["groups"]["chain"]


# ── bar-sanity records (synthetic, n=1000, seed 0) ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("c", ("c1", "c2", "c3", "c4"))
def test_bar_sanity_record_perfect_passes_and_every_null_reads_zero(c):
    p = os.path.join(DRAFT, "bar_sanity_record_s3_%s.json" % c)
    assert os.path.exists(p), p
    with open(p) as f:
        rec = json.load(f)
    assert rec["verdict"] == "OK", rec.get("failures")
    kinds = rec["kinds"]
    assert sum(1 for k in kinds.values() if k == "null") >= 10
    for g, kind in kinds.items():
        if kind == "perfect":
            assert rec["pass_rate"][g] >= 0.95, (g, rec["pass_rate"][g])
        elif kind == "null":
            assert rec["pass_rate"][g] == 0.0, (g, rec["pass_rate"][g])


def test_bar_sanity_common_feeds_the_real_aggregate():
    sys.path.insert(0, DRAFT)
    import bar_sanity_s3_common as C
    assert C.selfcheck(n=1) == 0
    for g in G.PRIMARY_GROUPS:
        gens = C.generators(g)
        assert "perfect" in gens and sum(1 for k in gens if k.startswith("null_")) >= 10
    for c in ("c1_chain", "c2_meta", "c3_tom", "c4_credit"):
        ns = {}
        with open(os.path.join(DRAFT, "bar_sanity_spec_s3_%s.py" % c)) as f:
            src = f.read()
        assert "SCORER_FILES" in src and "def decide" in src
        for rel in re.findall(r'"([^"]+\.py)"', src.split("SCORER_FILES")[1].split("]")[0]):
            assert os.path.exists(os.path.join(ROOT, rel)), rel
        del ns


# ── the gate-seed guard and the pool lines ───────────────────────────────────────────────────────────────────────────
def _run(*args):
    return subprocess.run([sys.executable, "-m", "research.runners._stage3_exit_draft"] + list(args),
                          capture_output=True, text=True, cwd=ROOT, env=dict(os.environ, SIM_NO_PROVENANCE="1"))


def test_gate_seed_is_refused_while_the_prereg_is_unregistered():
    p = _run("--shard", "--seed", "42", "--group", "chain", "--arm", "intact_a", "--out", "/nonexistent/x.json")
    assert p.returncode != 0 and "GATE seed" in p.stderr
    p = _run("--shard", "--seed", "42", "--group", "chain", "--arm", "intact_a", "--out", "/nonexistent/x.json",
             "--allow-gate-seed", "--prereg", G.PREREG)
    assert p.returncode != 0 and "GATE seed" in p.stderr
    p = _run("--emit-pool-jobs", "f" * 40, "--seeds", "42,43")
    assert p.returncode == 2 and "refused" in p.stderr


def test_emit_pool_jobs_defaults_to_the_dev_seed_and_the_dev_tree():
    p = _run("--emit-pool-jobs", "f" * 40)
    assert p.returncode == 0, p.stderr
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    assert len(lines) == sum(len(G.arms_for(g)) for g in G.GROUPS)
    for l in lines:
        assert "--seed 7 " in l and G.DEV_OUT_DIR in l and "--allow-gate-seed" not in l and "memcap.sh %d" % G.MEMCAP_GB in l


def test_pool_job_lines_are_pinned_memcapped_prereg_tagged_and_capped():
    sha = "f" * 40
    lines = G.pool_job_lines(sha)
    assert len(lines) == len(G.GATE_SEEDS) * sum(len(G.arms_for(g)) for g in G.GROUPS) <= 260
    for cmd, why in lines:
        assert ("revisions/%s" % sha) in cmd and ("memcap.sh %d" % G.MEMCAP_GB) in cmd and "--prereg %s" % G.PREREG in cmd
        assert "mem_ok.sh %d" % G.MEM_GB in cmd and "--allow-gate-seed" in cmd
        assert ("mem_gb=%d" % G.MEM_GB) in why and "prereg %s " % G.PREREG in why and "max_age_h=240" in why
    assert all("max_age_h=48" in w for _c, w in G.pool_job_lines(sha, max_age_h=48))
    assert all("max_age_h=240" in w for _c, w in G.pool_job_lines(sha, max_age_h=9999))
    assert G.MEM_GB == 9
