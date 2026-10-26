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
    # 186 at the prereg; amendment A4 adds the counted `ask/reversed` arm and the informational `ask/da_lesion` arm (198);
    # the C2 host-ownership test (2026-10-26) adds the informational `infer/kb_scramble` and `kb_scramble_spiking_control`
    assert len(lines) == len(SG.GATE_SEEDS) * sum(len(SG.arms_for(g)) for g in SG.GROUPS) == 210
    for cmd, why in lines:
        assert ("revisions/%s" % sha) in cmd and ("memcap.sh %d" % SG.MEMCAP_GB) in cmd and "--prereg %s" % SG.PREREG in cmd
        assert ("mem_gb=%d" % SG.MEM_GB) in why and SG.PREREG in why and "max_age_h=240" in why
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
             "lesion_graded": "graded_lesion", "lesion_graded.rep0": "graded_lesion", "permuted": "permuted",
             "reversed": "intact"}                    # amendment A4 (b): the reversed-order arm is counted
    for seed in seeds:
        for arm, kind in kinds.items():
            p = SG.shard_path(td, seed, "ask", arm)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            d = dict(SG._fake_shard("ask", SG._synthetic_ask(kind), arm=arm), seed=seed, arm=arm,
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
    def _sh(group, facts):     # the arm's registered turn table is recorded (amendment A4 review: required)
        return {"responses": {"x": {"answer": "a", "source": SG.EXPECTED_SOURCE}}, "sources": [SG.EXPECTED_SOURCE],
                "pythonhashseed_ok": True, "corpus_present": True, "builtin_facts": facts,
                "turns": [{"label": l, "message": m, "role": r} for l, m, r in SG.turns_for(group, 1, "intact_a")]}
    assert any("builtin_facts" in r for r in SG.arm_validity("inner", "intact_a", _sh("inner", [])))
    assert not SG.arm_validity("ask", "intact_a", _sh("ask", []))
    assert not SG.arm_validity("inner", "intact_a", _sh("inner", [["dog", "chase", "cat"]]))


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


# ── end to end over ALL five groups (review 2026-09-30): the bars are reachable by an ideal brain, and each clause bites ──
_S = SG.EXPECTED_SOURCE
_AB2 = "I don't know about that."


def _e2e_R(answer, svo=None, abst=False, **extra):
    d = {"answer": answer, "abstained": abst, "recalled_svo": svo, "source": _S, "derived": False}
    d.update(extra)
    return d


def _e2e_fr(a, v, p):
    return _e2e_R("the %s %s the %s" % (a, SG.VERB_3SG[v], p), [a, v, p])


def _e2e_teach_resps(turns):
    out = {}
    for l, m, r in turns:
        if r in ("teach", "expose"):
            out[l] = _e2e_R("ok " + m)
    return out


def _e2e_conv(kind):
    turns = SG.turns_for("conv", 1, "permuted" if kind == "permuted" else "intact_a")
    r = _e2e_teach_resps(turns)
    if kind == "permuted":
        r.update(p_fox_chase=_e2e_R(_AB2, abst=True), p_more=_e2e_R(_AB2, abst=True),
                 p_owl_hunt=_e2e_R(_AB2, abst=True), c_mole_hunt=_e2e_fr("mole", "hunt", "owl"))
    else:
        r.update(p_fox_chase=_e2e_fr("fox", "chase", "hare"), p_more=_e2e_fr("fox", "eat", "berry"),
                 p_owl_hunt=_e2e_fr("owl", "hunt", "mole"), c_mole_hunt=_e2e_R(_AB2, abst=True))
        if kind == "lesion":
            r["p_fox_chase"] = _e2e_R("The fox chases hare.", ["fox", "chase", "hare"])
    r["p_deer_sleep"] = _e2e_R("the deer sleeps in the meadow", ["deer", "sleep", "meadow"])
    r["c_dog_chase"] = _e2e_fr("dog", "chase", "cat")
    r["c_oov"] = _e2e_R(_AB2, abst=True)
    return r


def _e2e_infer(kind):
    turns = SG.turns_for("infer", 1, "permuted" if kind == "permuted" else "intact_a")
    r = _e2e_teach_resps(turns)
    if kind == "permuted":
        df = [["heron", "hunt", "gnat"], ["gnat", "eat", "frog"]]
        r.update(c_heron_hunt=_e2e_fr("heron", "hunt", "gnat"), c_frog_eat=_e2e_R(_AB2, abst=True))
    else:
        df = [["heron", "hunt", "frog"], ["frog", "eat", "gnat"]]
        r.update(c_heron_hunt=_e2e_fr("heron", "hunt", "frog"), c_frog_eat=_e2e_fr("frog", "eat", "gnat"))
    r["p_prey_eat"] = _e2e_R("I derived this from: ... the %s eats the %s" % (df[1][0], df[1][2]), None, derived=True,
                        derived_from=df)
    if kind == "lesion":
        r["p_prey_eat"] = _e2e_R(_AB2, abst=True)
    r.update(c_prey_chase=_e2e_R(_AB2, abst=True), p_yesno=_e2e_fr("crow", "chase", "wren"), c_dog_chase=_e2e_fr("dog", "chase", "cat"))
    return r


def _e2e_emo(kind):
    turns = SG.turns_for("emo", 1, "permuted" if kind == "permuted" else "intact_a")
    r = _e2e_teach_resps(turns)
    words = {"joy": "happy", "fear": "afraid", "anger": "angry", "sadness": "sad"}
    r["feel_base"] = SG._synthetic_feel(0.0)
    for c in ("joy", "fear", "anger", "sadness"):
        w = SG.SITUATION_SWAP[c] if kind == "permuted" else c
        d = 0.3 if w == "joy" else -0.3
        r["feel_%s" % c] = SG._synthetic_feel(0.0) if kind == "lesion" else SG._synthetic_feel(d, category=words[w])
    return r


def _e2e_inner(kind):
    turns = SG.turns_for("inner", 1, "permuted" if kind == "permuted" else "intact_a")
    r = _e2e_teach_resps(turns)
    # the idle turn voices the most recently taught fact: a deterministic function of the taught store (amendment A3:
    # the voiced fact must follow the exposure, so an ideal brain cannot voice one fixed fact)
    a, v, p = ("vole", "chase", "beetle") if kind == "permuted" else ("beetle", "chase", "mite")
    r["p_private"] = _e2e_R("Something's been on my mind — %s %s %s. What does %s %s?" % (a, v, p, a, v))
    if kind == "permuted":
        r.update(p_chase_all=_e2e_fr("lynx", "chase", "beetle"), p_chase_all_vole=_e2e_fr("vole", "chase", "beetle"),
                 c_lynx_chase=_e2e_fr("lynx", "chase", "mite"), c_beetle_all=_e2e_R(_AB2, abst=True),
                 c_mite_all=_e2e_fr("mite", "chase", "beetle"))
    else:
        r.update(p_chase_all=_e2e_fr("lynx", "chase", "mite"), p_chase_all_vole=_e2e_fr("vole", "chase", "mite"),
                 c_lynx_chase=_e2e_fr("lynx", "chase", "vole"), c_mite_all=_e2e_R(_AB2, abst=True),
                 c_beetle_all=_e2e_fr("beetle", "chase", "mite"))
    r["c_dog_chase"] = _e2e_fr("dog", "chase", "cat")
    if kind == "lesion_workspace":
        r.update(p_chase_all=_e2e_fr("lynx", "chase", "vole"), p_chase_all_vole=_e2e_fr("vole", "chase", "beetle"))
    if kind == "lesion_selfinit":
        r["p_private"] = _e2e_R("Nothing in particular is surfacing for me to bring up right now.", abst=True)
    return r


def _e2e_shard(group, arm, resp, seed):
    base = arm.split(".rep")[0]
    turns = SG.turns_for(group, seed, base)          # the arm's own turn table (amendment A4: `reversed` reorders)
    return {"seed": seed, "group": group, "arm": arm, "responses": resp, "sources": [_S], "pythonhashseed_ok": True,
            "corpus_present": True, "builtin_facts": [["dog", "chase", "cat"], ["cat", "eat", "fish"]],
            "turns": [{"label": l, "message": m, "role": r} for l, m, r in turns]}


def _e2e_build(td, seeds, mk):
    for seed in seeds:
        for group in SG.GROUPS:
            for arm in SG.arms_for(group):
                base = arm.split(".rep")[0]
                p = SG.shard_path(td, seed, group, arm)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                json.dump(_e2e_shard(group, arm, mk(group, base), seed), open(p, "w"))


def _e2e_ideal(group, base):
    if group == "ask":
        k = {"intact_a": "intact", "intact_b": "intact", "lesion_curiosity": "lesion", "lesion_graded": "graded_lesion",
             "permuted": "permuted", "reversed": "intact", "da_lesion": "intact"}[base]
        return SG._synthetic_ask(k)
    kind = "permuted" if base == "permuted" else ("lesion" if base.startswith("lesion") else "intact")
    if group == "inner" and base.startswith("lesion"):
        kind = base
    if base in ("transitive_chat", "qp_lesion"):
        kind = "intact"
    return {"conv": _e2e_conv, "infer": _e2e_infer, "emo": _e2e_emo, "inner": _e2e_inner}[group](kind)




def test_every_group_passes_on_an_ideal_brain_and_each_clause_bites(tmp_path):
    """The selftest drives only `ask` through the aggregate. Here every group: an IDEAL reply set PASSES (a bar no
    brain could reach would be a guaranteed FAIL); lesion arms equal to intact FAIL 'not load-bearing'; a permuted
    arm that ignores the exposure FAILs 'not exposure-driven'; all-abstain and echoed replies are never correct."""
    td = str(tmp_path)
    _e2e_build(td, (1, 2), _e2e_ideal)
    agg = SG.aggregate(td, seeds=(1, 2))
    assert agg["verdict"] == "PASS" and set(agg["criteria"].values()) == {"PASS"}, agg["criteria"]

    _e2e_build(td, (1, 2), lambda g, b: _e2e_ideal(g, "intact_a") if b.startswith("lesion") else _e2e_ideal(g, b))
    agg = SG.aggregate(td, seeds=(1, 2))
    for g, d in agg["groups"].items():
        assert d["verdict"] == "FAIL" and any("not load-bearing" in w for w in d["why"]), (g, d["why"])

    # amendment A4 review ruling (2026-10-13): the permuted arm is a CONTROL; its moat set is intact_a's, so an
    # exposure-blind brain (it replays the intact replies, teach echoes included) stays FAIL 'not exposure-driven'
    _e2e_build(td, (1, 2), lambda g, b: _e2e_ideal(g, "intact_a") if b == "permuted" else _e2e_ideal(g, b))
    agg = SG.aggregate(td, seeds=(1, 2))
    for g, d in agg["groups"].items():
        assert d["verdict"] == "FAIL" and any("not exposure-driven" in w for w in d["why"]), (g, d["why"])

    _e2e_build(td, (1, 2), lambda g, b: {l: _e2e_R(_AB2, abst=True) for l in _e2e_ideal(g, b)})
    agg = SG.aggregate(td, seeds=(1, 2))
    assert all(d["n_correct"] == 0 for d in agg["groups"].values()) and agg["GO"] is False

    def _echo(g, b):
        return {l: _e2e_R(m) for l, m, _r in SG.turns_for(g, 1, "permuted" if b == "permuted" else "intact_a")}
    _e2e_build(td, (1, 2), _echo)
    agg = SG.aggregate(td, seeds=(1, 2))
    assert all(d["n_correct"] == 0 for d in agg["groups"].values()) and agg["GO"] is False


# ── amendment A3 (2026-10-06): C4 names the category through the ladder; C5's idle fact follows the exposure ──
def _agg_with(tmp_path, over):
    """Aggregate over an ideal brain on seeds (1, 2) with `over(group, base)` -> a reply dict replacing that arm's."""
    td = str(tmp_path)
    _e2e_build(td, (1, 2), lambda g, b: (over(g, b) or _e2e_ideal(g, b)))
    return SG.aggregate(td, seeds=(1, 2))


def _lookup_emo(group, base):
    """A host word-lookup brain: the named category is read off the heard affect words, the printed numbers off the ladder;
    the ladder lesion clamps the numbers to neutral but the word stays."""
    if group != "emo" or base not in ("intact_a", "intact_b", "lesion_affect", "permuted"):
        return None
    turns = SG.turns_for("emo", 1, "permuted" if base == "permuted" else "intact_a")
    r = _e2e_teach_resps(turns)
    words = {"joy": "happy", "fear": "afraid", "anger": "angry", "sadness": "sad"}
    r["feel_base"] = SG._synthetic_feel(0.0)
    for c in ("joy", "fear", "anger", "sadness"):
        w = SG.SITUATION_SWAP[c] if base == "permuted" else c
        d = 0.0 if base == "lesion_affect" else (0.3 if w == "joy" else -0.3)
        r["feel_%s" % c] = SG._synthetic_feel(d, category=words[w])
    return r


def test_c4_a_category_lookup_independent_of_the_ladder_fails_the_lesion_clause(tmp_path):
    """A brain that names the emotion by looking up the heard affect words is correct and tracks the permuted words, and
    its ladder lesion changes the printed number on every feel turn: the OLD lesion clause (any primary field differs)
    passed it. The hardened clause needs the NAMED category to change, so it FAILs as not load-bearing."""
    agg = _agg_with(tmp_path, _lookup_emo)
    g = agg["groups"]["emo"]
    assert g["n_correct"] == 2 and g["n_permuted_tracks"] == 2 and g["n_reproduced"]["lesion_affect"] == 2
    assert g["n_lesion_effect"]["lesion_affect"] == 0 and g["verdict"] == "FAIL", g
    row = g["seeds"][1]["lesions"]["lesion_affect"]
    assert row["reply_effect"] is True and not any(row["category_changed"].values())
    assert any("not load-bearing" in w and "printed number moved on 2" in w for w in g["why"]), g["why"]
    # the ideal brain (the ladder lesion clears the word too) still PASSES, and records the category change per turn
    g2 = _agg_with(tmp_path, lambda g, b: None)["groups"]["emo"]
    assert g2["verdict"] == "PASS" and all(g2["seeds"][1]["lesions"]["lesion_affect"]["category_changed"].values())


def test_c4_named_category_changed_is_per_turn_and_none_to_none_is_not_a_change():
    named = lambda **k: {"named": dict({"joy": None, "fear": None, "anger": None, "sadness": None}, **k)}   # noqa: E731
    i = named(joy="joy", fear="fear", anger="anger", sadness="sadness")
    assert SG.named_category_changed(i, named()) == {c: True for c in i["named"]}
    assert SG.named_category_changed(i, i) == {c: False for c in i["named"]}
    assert not any(SG.named_category_changed(named(), named()).values())
    mixed = SG.named_category_changed(i, named(joy="joy"))
    assert mixed == {"joy": False, "fear": True, "anger": True, "sadness": True}


def _fixed_idle(group, base):
    """The same stored fact voiced on every idle turn, whatever the exposure (everything else ideal)."""
    if group != "inner" or base.startswith("lesion"):
        return None
    r = _e2e_ideal("inner", base)
    r["p_private"] = _e2e_R("Something's been on my mind — cat eat fish. What does cat eat?")
    return r


def test_c5_one_fixed_stored_fact_every_idle_turn_is_not_exposure_driven(tmp_path):
    """A brain voicing one fixed stored fact passed the old bar (the fact is stored, the self-initiation lesion removes it).
    Amendment A3: the voiced fact must also differ between the intact and the permuted exposure."""
    agg = _agg_with(tmp_path, _fixed_idle)
    g = agg["groups"]["inner"]
    assert g["n_correct"] == 2 and g["n_lesion_effect"] == {"lesion_workspace": 2, "lesion_selfinit": 2}
    assert g["n_permuted_tracks"] == 0 and g["verdict"] == "FAIL", g
    assert any("not exposure-driven" in w for w in g["why"]) and g["sub_clauses_n_true"]["private_selected_by_state"] == 0
    g2 = _agg_with(tmp_path, lambda g, b: None)["groups"]["inner"]
    assert g2["verdict"] == "PASS" and g2["sub_clauses_n_true"]["private_selected_by_state"] == 2


def test_c5_private_selected_by_state_needs_two_stored_and_different_triples():
    u = lambda t: {"private_utterance": t}                                                                   # noqa: E731
    assert SG.private_selected_by_state(u(["beetle", "chase", "mite"]), u(["vole", "chase", "beetle"]))
    assert not SG.private_selected_by_state(u(["cat", "eat", "fish"]), u(["cat", "eat", "fish"]))
    assert not SG.private_selected_by_state(u(None), u(["vole", "chase", "beetle"]))
    assert not SG.private_selected_by_state(u(["beetle", "chase", "mite"]), u(None))


def test_the_dev7_c4_and_c5_reads_under_the_hardened_scorer():
    """Reported, not tuned to: the dev-seed-7 shards on the branch. C4: no category is named by the dev brain, so the
    hardened clause cannot hold (and capability is absent anyway). C5: the idle fact is the self-initiation organ's fixed
    demo lexicon, so `private_voices_stored_fact` is already False. Skipped where the shards are not checked out."""
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "findings", "raw",
                     "_stage2_exit", "dev", "s7")
    ia, la = os.path.join(d, "emo__intact_a.json"), os.path.join(d, "emo__lesion_affect.json")
    ni = os.path.join(d, "inner__intact_a.json")
    if not (os.path.exists(ia) and os.path.exists(la) and os.path.exists(ni)):
        pytest.skip("dev-7 shards not present")
    ei, el = SG.score_emo(json.load(open(ia))), SG.score_emo(json.load(open(la)))
    assert not ei["all"] and not any(SG.named_category_changed(ei, el).values())
    assert not SG.score_inner(json.load(open(ni)))["private_voices_stored_fact"]


def test_pool_lines_declare_memory_above_the_measured_dev_peak():
    """Dev seed 7 (2026-09-30, workstation, final runner) measured peak RSS 6.3-6.5 GB for conv/infer/ask/inner and
    8.1 GB for emo. A line declaring mem_gb=7 under a 9 GB memcap under-declares the dispatcher's budget and leaves
    ~1 GB before the cap kills a shard; the declaration must cover the measured peak with >= 10% headroom."""
    peak_mb = 8101.1
    assert SG.MEM_GB * 1024 >= 1.10 * peak_mb and SG.MEMCAP_GB > SG.MEM_GB
    for cmd, why in SG.pool_job_lines("f" * 40):
        assert ("mem_ok.sh %d 1" % SG.MEM_GB) in cmd and ("memcap.sh %d --" % SG.MEMCAP_GB) in cmd, cmd
        assert ("mem_gb=%d " % SG.MEM_GB) in why, why


# ── amendment A4 review (2026-10-13): a C1 FAIL carries the named-gaps line through aggregate and report ──
def test_a_c1_fail_prints_the_named_gaps_line_through_aggregate_and_report(tmp_path):
    """NEEDS-CLAUDE #17 ruling: no wording change; a C1 FAIL names the wh-locative parse, locative frame storage and
    discourse elaboration, whichever sub-clause failed. Here the dev-7 shape: the deer reply drops its preposition."""
    def _no_prep(g, b):
        r = _e2e_ideal(g, b)
        if g == "conv" and b in ("intact_a", "intact_b"):
            r["p_deer_sleep"] = _e2e_R("the deer sleeps the meadow", ["deer", "sleep", "meadow"])
        return r
    td = str(tmp_path)
    _e2e_build(td, (1, 2), _no_prep)
    agg = SG.aggregate(td, seeds=(1, 2))
    conv = agg["groups"]["conv"]
    assert conv["verdict"] == "FAIL" and any("capability absent" in w for w in conv["why"])
    assert conv["named_gaps"] == list(SG.A4_NAMED_GAPS["conv"])
    assert all(d["named_gaps"] == [] for g, d in agg["groups"].items() if g != "conv")
    rep = SG.report(agg)
    assert "Named gaps (amendment A4 reporting rule, NEEDS-CLAUDE #17" in rep
    assert ("- C1 (conv): wh-locative question parse; locative frame storage (the teach echo drops the preposition); "
            "discourse elaboration for 'tell me more'") in rep
    # an ideal brain names nothing
    _e2e_build(td, (1, 2), _e2e_ideal)
    assert "Named gaps" not in SG.report(SG.aggregate(td, seeds=(1, 2)))


# ── recorder fix (RULING batch-1024 (10)): a taught fact must appear in builtin_facts ──
def test_builtin_facts_reads_a_taught_fact_through_the_registry_view():
    """The dev-7 pre-flight read builtin_facts: [] in all 33 shards while the brain recalled facts fine: the
    recorder scanned module-level dicts for the chat, but the server's `_BRAIN_CHATS` is a `ContextFieldView`
    (a MutableMapping, deliberately dict-shaped but NOT a dict) keyed (session, brain, renderer). The lookup
    must find the chat through that view and surface what the acquisition path leaves in stored_facts — the
    build-time facts plus one TAUGHT in the turn — which is exactly what SG.read_builtin_facts latches into the
    shard's builtin_facts field (no brain build: the live arms stay the dev smoke / gate's job)."""
    from types import SimpleNamespace
    from webapp import one_brain as OB

    session = "s2_taught_facts_7"
    key = (session, SG.BRAIN_NAME, SG.RENDERER)

    class _Chat:                      # the only attribute the recorder reads (the buffer tier acquisition writes)
        stored_facts = [["yak", "eat", "grass"], ["dog", "chase", "cat"]]   # the second = TAUGHT this session

    reg = OB.ContextRegistry()
    view = reg.view("chat")
    assert not isinstance(view, dict), "the registry view must stay non-dict — that mismatch is the regression"
    view.setdefault(key, _Chat())     # the production write shape (server.py: `_BRAIN_CHATS.setdefault(key, chat)`)
    srv = SimpleNamespace(_BRAIN_CHATS=view)

    chat = SG._find_chat(srv, session)
    assert chat is not None, "the registry view went unread (the dict-only module scan)"
    got = SG.read_builtin_facts(srv, session)
    assert got is not None
    assert ["yak", "eat", "grass"] in got
    assert ["dog", "chase", "cat"] in got          # the TAUGHT fact appears in builtin_facts
    # a conversation that does not exist yet reads as None: the read-once latch must never freeze [] on a miss
    absent = SimpleNamespace(_BRAIN_CHATS=OB.ContextRegistry().view("chat"))
    assert SG.read_builtin_facts(absent, "s2_absent_7") is None
