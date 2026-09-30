"""Stage-2 exit gate (research/runners/stage2_exit_gate.py): the frozen structure and the registered scorer, with no
brain build. The live arms are exercised by the dev smoke and the six-seed gate (see the preregistration)."""
import json
import os
import subprocess
import sys

import pytest

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import stage2_exit_gate as SG

# production resolves the shipped LTM as wikidata_100k first, wikidata_core_15k as the fallback
# (webapp/server.py::_default_ltm_bundle_dir, the #108 flip of 2026-09-02): the novel words must be absent from both
_LTM_BUNDLES = ("wikidata_100k", "wikidata_core_15k")
_LTM_FACTS = os.path.expanduser("~/Projects/sim-data/knowledge_bundles/wikidata_core_15k/facts.json")


def test_selftest_passes():
    ok, checks = SG.selftest(verbose=False)
    assert ok, {k: v for k, v in checks.items() if not v}


def test_one_primary_group_per_criterion_and_every_group_has_the_arm_kinds():
    assert sorted(SG.GROUPS[g]["criterion"] for g in SG.PRIMARY_GROUPS) == list(SG.CRITERIA)
    for g in SG.GROUPS:
        kinds = {SG.arm_spec(g, a)[2] for a in SG.arms_for(g)}
        assert {"intact", "lesion", "permuted"} <= kinds, (g, kinds)
        for les in SG.GROUPS[g]["lesions"]:
            assert les in SG.SCORED_LABELS[g], (g, les)
            labels = {t[0] for t in SG.GROUPS[g]["turns"]}
            assert set(SG.SCORED_LABELS[g][les]) <= labels, (g, les)


def test_lesion_arms_are_env_flags_production_reads_or_named_patches():
    for g, d in SG.GROUPS.items():
        for les, spec in d["lesions"].items():
            assert spec["env"] or spec["patch"], (g, les)
            assert spec["cuts"], (g, les)


def test_lesion_env_flags_are_read_by_production_code():
    """Every env-flag lesion names a variable some production module actually reads (a misspelled flag would be a
    silent no-op arm scored as 'not load-bearing')."""
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    files = [os.path.join(root, "webapp", "server.py"), os.path.join(root, "research", "runners", "brain_chat_tui.py")]
    for sub in ("research/runners", "webapp"):
        d = os.path.join(root, sub)
        files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".py")]
    blob = ""
    for f in set(files):
        try:
            with open(f, errors="ignore") as fh:
                blob += fh.read()
        except OSError:
            pass
    for g, d in SG.GROUPS.items():
        for les, spec in d["lesions"].items():
            for k in spec["env"]:
                assert k in blob, (g, les, k)
        for inf, spec in d["informational"].items():
            for k in spec["env"]:
                assert k in blob, (g, inf, k)


def test_situation_affect_words_are_strong_under_the_production_salience_gate():
    from research.runners.affect_production_organ import WARRINER, _STRONG_MARGIN
    for cat, words in SG.SITUATION_AFFECT_WORDS.items():
        for w in words:
            assert w in WARRINER, (cat, w)
            assert abs(WARRINER[w][0] - 5.0) >= _STRONG_MARGIN, (cat, w, WARRINER[w])
            want_pos = cat == "joy"
            assert (WARRINER[w][0] > 5.0) == want_pos, (cat, w, WARRINER[w])
        for w in words:
            assert w in SG.SITUATIONS[cat].split(), (cat, w)


def test_emotion_lexicon_names_each_category_once_and_never_the_read_out_boilerplate():
    for cat, words in SG.EMOTION_LEXICON.items():
        for w in words:
            others = [c for c, ws in SG.EMOTION_LEXICON.items() if c != cat and w in ws]
            assert not others, (cat, w, others)
    boiler = "My affect monitor reads negative right now (valence differential -0.050, moderate arousal) -- that's a functional read of my mood state, not a felt experience."
    assert SG.parse_feel(boiler)["category"] is None and SG.parse_feel(boiler)["categories_named"] == []


def test_novel_words_are_absent_from_the_tiny_demo_vocabulary_and_the_shipped_ltm_core():
    tiny = {"brain", "use", "spikes", "learn", "words", "store", "memory", "dog", "chase", "cat", "eat", "fish"}
    for w in SG.NOVEL_WORDS:
        assert w not in tiny, w
    import re
    paths = [os.path.expanduser("~/Projects/sim-data/knowledge_bundles/%s/facts.json" % b) for b in _LTM_BUNDLES]
    present = [p for p in paths if os.path.exists(p)]
    if not present:
        pytest.skip("shipped LTM bundle not present on this machine")
    for p in present:
        with open(p, errors="ignore") as f:
            blob = f.read().lower()
        hits = [w for w in SG.NOVEL_WORDS if re.search(r"\b%s\b" % re.escape(w), blob)]
        assert not hits, (p, hits)


def test_the_handoff_checks_the_ltm_bundle_production_resolves_first():
    """The pool check must look for the bundle the handler loads (wikidata_100k), not only the fallback core."""
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(root, "docs", "drafts", "stage2-exit", "HANDOFF.md")) as f:
        h = f.read()
    assert "grep -q wikidata_100k" in h


def test_grammar_checker_is_strict_about_form_and_prepositions():
    assert SG.grammatical("the fox chases the hare")
    assert SG.grammatical("the fox chases the hare the fox eats the berry — worth going further here.")
    assert not SG.grammatical("The fox chases hare.")            # the template stub's form (no object article)
    assert not SG.grammatical("The robin flys sky.")             # the template's naive inflection
    assert not SG.grammatical("the fox chased the hare")         # not the 3sg present form
    assert not SG.grammatical("the deer sleeps the meadow")      # a taught prepositional fact without its preposition
    assert SG.grammatical("the deer sleeps in the meadow")
    assert not SG.grammatical("I don't know about that.")
    assert SG.grounded("the fox chases the hare", [["fox", "chase", "hare"]])
    assert not SG.grounded("the fox chases the owl", [["fox", "chase", "hare"]])


def test_permuted_arms_keep_every_probe_and_change_only_the_teaching():
    for g in SG.GROUPS:
        it = [(l, m) for l, m, r in SG.GROUPS[g]["turns"] if r not in ("teach", "expose")]
        pe = [(l, m) for l, m, r in SG.GROUPS[g]["permuted_turns"] if r not in ("teach", "expose")]
        assert it == pe, g
    swapped = dict((l, m) for l, m, _r in SG.GROUPS["emo"]["permuted_turns"])
    assert "sad" in swapped["s_joy_1"] and "happy" in swapped["s_sadness_1"]
    assert "angry" in swapped["s_fear_1"] and "scared" in swapped["s_anger_1"]


def test_gate_seed_is_refused_without_the_flag():
    p = subprocess.run([sys.executable, "-m", "research.runners.stage2_exit_gate", "--shard", "--seed", "42",
                        "--group", "ask", "--arm", "intact_a", "--out", "/nonexistent/x.json"],
                       capture_output=True, text=True, env=dict(os.environ, SIM_NO_PROVENANCE="1"))
    assert p.returncode != 0 and "GATE seed" in p.stderr


def test_pool_job_lines_are_pinned_memcapped_prereg_tagged_and_carry_max_age():
    sha = "f" * 40
    lines = SG.pool_job_lines(sha)
    assert len(lines) == len(SG.GATE_SEEDS) * sum(len(SG.arms_for(g)) for g in SG.GROUPS) == 186
    for cmd, why in lines:
        assert ("revisions/%s" % sha) in cmd and "memcap.sh 9" in cmd and "--prereg %s" % SG.PREREG in cmd
        assert "mem_gb=7" in why and SG.PREREG in why and "max_age_h=240" in why
        assert "prereg %s " % SG.PREREG in why      # the queue's guard reads the path up to whitespace
    assert all("max_age_h=48" in w for _c, w in SG.pool_job_lines(sha, max_age_h=48))
    assert all("max_age_h=240" in w for _c, w in SG.pool_job_lines(sha, max_age_h=9999))   # capped


def test_multistep_info_reader_finds_the_trace_wherever_it_is_attached():
    info = {"cycles": 3, "resolved_hops": 2, "terminal": "mite"}
    assert SG._multistep_info({"gnw": {"multistep": info}}) == info
    assert SG._multistep_info({"gnw_multistep": info}) == info
    assert SG._multistep_info({"answer": "x"}) is None


# ── review fixes 2026-09-30 (adversarial review of the packet; failing-first) ────────────────────────────────────────
def test_gate_seed_shard_hands_the_gate_flag_to_its_worker(monkeypatch, tmp_path):
    """A pool line runs `--shard --seed 42 --allow-gate-seed --prereg <p>`; run_shard spawns `--worker` in a fresh
    process, and main()'s gate-seed guard applies to --worker too. The worker argv must therefore carry the flag, or
    every one of the 186 gate jobs dies with rc=2 before building a brain (measured at 8c934312a)."""
    seen = {}

    class _P:
        returncode = 0

    def _fake_worker(seed, group, arm, out_path, labels=None):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)     # the real worker makes its directory too
        with open(out_path, "w") as f:
            json.dump({"responses": {}, "seed": seed}, f)
        return 0

    def _fake_run(cmd, env=None, cwd=None):
        seen["cmd"] = list(cmd)
        assert cmd[1:4] == ["-u", "-m", "research.runners.stage2_exit_gate"], cmd
        monkeypatch.setattr(sys, "argv", ["stage2_exit_gate"] + list(cmd[4:]))
        with pytest.raises(SystemExit) as e:
            SG.main()
        p = _P()
        p.returncode = e.value.code
        seen["worker_rc"] = e.value.code
        return p

    monkeypatch.setattr(SG, "run_worker", _fake_worker)
    monkeypatch.setattr(SG.subprocess, "run", _fake_run)
    out = str(tmp_path / "s42" / "ask__intact_a.json")
    monkeypatch.setattr(sys, "argv", ["stage2_exit_gate", "--shard", "--seed", "42", "--group", "ask", "--arm", "intact_a",
                                      "--allow-gate-seed", "--prereg", SG.PREREG, "--out", out])
    with pytest.raises(SystemExit) as outer:
        SG.main()
    assert seen.get("worker_rc") == 0, seen
    assert outer.value.code == 0 and os.path.exists(out)
    # a dev seed never needs (or gets) the flag
    monkeypatch.setattr(sys, "argv", ["stage2_exit_gate", "--shard", "--seed", "7", "--group", "ask", "--arm", "intact_a",
                                      "--out", str(tmp_path / "s7.json")])
    with pytest.raises(SystemExit) as dev:
        SG.main()
    assert dev.value.code == 0 and "--allow-gate-seed" not in seen["cmd"]


def _ask_tree(td, seeds=(1, 2)):
    kinds = {"intact_a": "intact", "intact_b": "intact", "lesion_curiosity": "lesion", "lesion_curiosity.rep0": "lesion",
             "lesion_graded": "graded_lesion", "lesion_graded.rep0": "graded_lesion", "permuted": "permuted"}
    for seed in seeds:
        for arm, kind in kinds.items():
            p = SG.shard_path(td, seed, "ask", arm)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            d = dict(SG._fake_shard("ask", SG._synthetic_ask(kind), permuted=(kind == "permuted")), seed=seed, arm=arm,
                     corpus_present=True)
            with open(p, "w") as f:
                json.dump(d, f)


def _rewrite(td, seed, arm, fn):
    p = SG.shard_path(td, seed, "ask", arm)
    with open(p) as f:
        d = json.load(f)
    fn(d)
    with open(p, "w") as f:
        json.dump(d, f)


def test_an_errored_turn_in_a_lesion_arm_is_undefined_never_a_not_load_bearing_fail(tmp_path):
    """Prereg section 7: an errored reply makes the seed UNDEFINED; UNDEFINED is never counted as a pass or a fail.
    At 8c934312a an errored scored turn in a LESION arm read as 'no effect' and the group FAILed 'not load-bearing'."""
    td = str(tmp_path)
    _ask_tree(td)
    assert SG.aggregate(td, seeds=(1, 2))["groups"]["ask"]["verdict"] == "PASS"

    def _err(d):
        for l in SG.SCORED_LABELS["ask"]["lesion_curiosity"]:
            d["responses"][l] = {"_error": "RuntimeError: organ crashed"}
    for arm in ("lesion_curiosity", "lesion_curiosity.rep0"):
        _rewrite(td, 2, arm, _err)
    g = SG.aggregate(td, seeds=(1, 2))["groups"]["ask"]
    assert g["verdict"] == "INCOMPLETE", (g["verdict"], g["why"])
    assert any("errored" in r for r in g["seeds"][2]["undefined_reasons"]), g["seeds"][2]["undefined_reasons"]


def test_an_organ_error_in_a_counted_arm_is_undefined(tmp_path):
    """FAILURE_LOG 2026-09-29 (B2b): an organ that raised inside a lesion arm was caught into `<organ>.error` and the
    missing field was scored as the lesion's effect. An organ error in any counted arm is an instrument failure."""
    td = str(tmp_path)
    _ask_tree(td)

    def _organ_err(d):
        d["responses"]["g_eel"]["curiosity"] = {"on": True, "error": "ValueError: could not broadcast"}
    _rewrite(td, 1, "lesion_curiosity", _organ_err)
    g = SG.aggregate(td, seeds=(1, 2))["groups"]["ask"]
    assert g["verdict"] == "INCOMPLETE" and any("organ error" in r for r in g["seeds"][1]["undefined_reasons"])


def test_source_hash_and_corpus_are_required_in_every_counted_arm_not_only_the_intact_pair(tmp_path):
    """Prereg section 7: every response's source is `tiny-demo +LTM`, PYTHONHASHSEED=0 and corpus_present recorded.
    At 8c934312a only intact_a/intact_b were checked, so a lesion or permuted arm built on a degraded node counted."""
    td = str(tmp_path)
    _ask_tree(td)
    _rewrite(td, 1, "lesion_graded", lambda d: d.update(corpus_present=False))
    _rewrite(td, 2, "permuted", lambda d: d.update(sources=["tiny-demo"]))
    agg = SG.aggregate(td, seeds=(1, 2))
    g = agg["groups"]["ask"]
    assert g["n_defined"] == 0, [g["seeds"][s]["undefined_reasons"] for s in (1, 2)]
    assert any("corpus absent" in r for r in g["seeds"][1]["undefined_reasons"])
    assert any("unexpected brain source" in r for r in g["seeds"][2]["undefined_reasons"])
    pre = {c["name"]: c for c in agg["preconditions"]}
    assert pre["every response from the expected brain"]["ok"] is False
    _ask_tree(td)
    _rewrite(td, 2, "lesion_curiosity.rep0", lambda d: d.update(pythonhashseed_ok=False))
    g = SG.aggregate(td, seeds=(1, 2))["groups"]["ask"]
    assert any("PYTHONHASHSEED" in r for r in g["seeds"][2]["undefined_reasons"])


def test_grounding_set_is_required_where_a_scorer_reads_it(tmp_path):
    """C1 (`conv`) and C5 (`inner`) score grounding against `builtin_facts`; an arm that failed to record it would
    read as 'capability absent' (an instrument failure scored as a FAIL)."""
    assert SG.GROUNDING_GROUPS == ("conv", "inner")
    shard = {"responses": {"x": {"answer": "a", "source": SG.EXPECTED_SOURCE}}, "sources": [SG.EXPECTED_SOURCE],
             "pythonhashseed_ok": True, "corpus_present": True, "builtin_facts": []}
    assert any("builtin_facts" in r for r in SG.arm_validity("inner", "intact_a", shard))
    assert not SG.arm_validity("ask", "intact_a", shard)
    shard["builtin_facts"] = [["dog", "chase", "cat"]]
    assert not SG.arm_validity("inner", "intact_a", shard)


def test_worker_captures_the_multistep_trace_the_handler_does_not_attach():
    """Prereg 4.5 names the chase trace (cycles, resolved_hops, terminal) as C5's internal secondary; at HEAD the
    handler stashes it on `chat._last_gnw_multistep` and never puts it in the response, so the worker must read it."""
    class _Chat:
        _last_gnw_multistep = {"acted": True, "reason": "chain_terminal", "terminal": "mite", "resolved_hops": 3,
                               "cycles": 4}
    rec = SG.capture_internal(_Chat())
    assert rec["gnw_multistep"]["terminal"] == "mite"
    shard = {"responses": {"p_chase_all": {"answer": "x", "recalled_svo": ["lynx", "chase", "mite"]}},
             "internal": {"p_chase_all": rec}, "builtin_facts": [], "turns": []}
    s = SG.score_inner(shard)
    assert s["chain_internal"]["resolved_hops"] == 3 and s["chain_hops_internal_ge2"] is True
    assert SG.capture_internal(object()) == {}


def test_process_record_reports_peak_rss_and_the_resolved_ltm_bundle():
    class _Srv:
        @staticmethod
        def _resolve_ltm_bundle():
            return "/x/knowledge_bundles/wikidata_100k"
    rec = SG.process_record(_Srv)
    assert rec["peak_rss_mb"] > 0 and rec["ltm_bundle"] == "wikidata_100k"
