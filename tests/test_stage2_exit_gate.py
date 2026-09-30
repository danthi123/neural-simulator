"""Stage-2 exit gate (research/runners/stage2_exit_gate.py): the frozen structure and the registered scorer, with no
brain build. The live arms are exercised by the dev smoke and the six-seed gate (see the preregistration)."""
import json
import os
import subprocess
import sys

import pytest

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import stage2_exit_gate as SG

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
    if not os.path.exists(_LTM_FACTS):
        pytest.skip("shipped LTM bundle not present on this machine")
    with open(_LTM_FACTS, errors="ignore") as f:
        blob = f.read().lower()
    import re
    hits = [w for w in SG.NOVEL_WORDS if re.search(r"\b%s\b" % re.escape(w), blob)]
    assert not hits, hits


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
