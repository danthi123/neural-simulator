"""Stage-1 exit gate (research/runners/stage1_exit_gate.py): the frozen structure and the registered scorer, with no
brain build. The live arms are exercised by the dev smoke and the six-seed gate (see the preregistration)."""
import os
import subprocess
import sys

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import stage1_exit_gate as SG


def test_selftest_passes():
    ok, checks = SG.selftest(verbose=False)
    assert ok, {k: v for k, v in checks.items() if not v}


def test_one_primary_group_per_criterion_and_every_group_has_the_four_arm_kinds():
    assert sorted(SG.GROUPS[g]["criterion"] for g in SG.PRIMARY_GROUPS) == ["C1", "C2", "C3", "C4"]
    for g in SG.GROUPS:
        kinds = {SG.arm_spec(g, a)[2] for a in SG.arms_for(g)}
        assert {"intact", "lesion", "permuted"} <= kinds, (g, kinds)


def test_lesion_arms_are_env_flags_production_reads_or_named_patches():
    for g, d in SG.GROUPS.items():
        for les, spec in d["lesions"].items():
            assert spec["env"] or spec["patch"], (g, les)
            assert spec["cuts"], (g, les)


def test_held_out_words_match_the_production_lexicon_norms():
    from research.runners.affect_production_organ import WARRINER, _STRONG_MARGIN
    for w, v in SG.HELDOUT_WORDS:
        assert w in WARRINER and abs(WARRINER[w][0] - v) < 1e-9, (w, v, WARRINER.get(w))
        assert abs(v - 5.0) >= _STRONG_MARGIN, (w, v)      # passes the production salience gate


def test_held_out_words_are_in_the_learned_map():
    from research.runners.affect_production_organ import _get_learned_valence
    m = _get_learned_valence()
    missing = [w for w, _v in SG.HELDOUT_WORDS if w not in m]
    assert not missing, missing


def test_gate_seed_is_refused_without_the_flag():
    p = subprocess.run([sys.executable, "-m", "research.runners.stage1_exit_gate", "--shard", "--seed", "42",
                        "--group", "word", "--arm", "intact_a", "--out", "/nonexistent/x.json"],
                       capture_output=True, text=True, env=dict(os.environ, SIM_NO_PROVENANCE="1"))
    assert p.returncode != 0 and "GATE seed" in p.stderr


def test_pool_job_lines_are_pinned_memcapped_and_prereg_tagged():
    sha = "f" * 40
    lines = SG.pool_job_lines(sha)
    assert len(lines) == len(SG.GATE_SEEDS) * sum(len(SG.arms_for(g)) for g in SG.GROUPS)
    for cmd, why in lines:
        assert ("revisions/%s" % sha) in cmd and "memcap.sh 9" in cmd and "--prereg %s" % SG.PREREG in cmd
        assert "mem_gb=7" in why and SG.PREREG in why


def test_permuted_arm_of_val_novel_keeps_the_probe_words():
    intact = dict((l, m) for l, m, _r in SG.GROUPS["val_novel"]["turns"])
    perm = dict((l, m) for l, m, _r in SG.GROUPS["val_novel"]["permuted_turns"])
    assert intact["x_pos"] == perm["x_pos"] == "glorp" and intact["x_neg"] == perm["x_neg"] == "blick"
    assert intact["e_pos1"] != perm["e_pos1"]
