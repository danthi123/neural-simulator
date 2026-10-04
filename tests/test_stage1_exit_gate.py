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


def test_gate_seed_shard_hands_the_gate_flag_to_its_worker(monkeypatch, tmp_path):
    """Pre-data fix 2026-09-30 (pool canary): a pool line runs `--shard --seed 42 --allow-gate-seed --prereg <p>`;
    run_shard spawns `--worker` in a fresh process and main()'s gate-seed guard applies to --worker too, so the
    worker argv must carry the flag, or every one of the 204 gate jobs dies rc=2 before building a brain (measured
    on pool42 at bb6705de6 through the sandbox: 'GATE seed 42 refused' -> 'worker FAILED rc=2')."""
    import json
    import pytest
    seen = {}

    class _P:
        returncode = 0

    def _fake_worker(seed, group, arm, out_path, labels=None):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"responses": {}, "seed": seed}, f)
        return 0

    def _fake_run(cmd, env=None, cwd=None):
        seen["cmd"] = list(cmd)
        assert cmd[1:4] == ["-u", "-m", "research.runners.stage1_exit_gate"], cmd
        monkeypatch.setattr(sys, "argv", ["stage1_exit_gate"] + list(cmd[4:]))
        with pytest.raises(SystemExit) as e:
            SG.main()
        p = _P()
        p.returncode = e.value.code
        seen["worker_rc"] = e.value.code
        return p

    monkeypatch.setattr(SG, "run_worker", _fake_worker)
    monkeypatch.setattr(SG.subprocess, "run", _fake_run)
    out = str(tmp_path / "s42" / "val_novel__intact_a.json")
    monkeypatch.setattr(sys, "argv", ["stage1_exit_gate", "--shard", "--seed", "42", "--group", "val_novel", "--arm",
                                      "intact_a", "--allow-gate-seed", "--prereg", SG.PREREG, "--out", out])
    with pytest.raises(SystemExit) as outer:
        SG.main()
    assert seen.get("worker_rc") == 0, seen
    assert outer.value.code == 0 and os.path.exists(out)
    assert seen["cmd"][seen["cmd"].index("--prereg") + 1] == SG.PREREG
    # a dev seed never needs (or gets) the flag
    monkeypatch.setattr(sys, "argv", ["stage1_exit_gate", "--shard", "--seed", "7", "--group", "val_novel", "--arm",
                                      "intact_a", "--out", str(tmp_path / "s7.json")])
    with pytest.raises(SystemExit) as dev:
        SG.main()
    assert dev.value.code == 0 and "--allow-gate-seed" not in seen["cmd"]


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


def test_val_heldout_permuted_map_must_derange_every_held_out_word(tmp_path):
    """Claude ruling s1-c4-void (2026-10-04): on the first gate attempt the learned map was absent on the pool, the
    permuted-map patch deranged 0 of 24 words, and the inert control read as "not tracked". A permuted arm that did not
    derange all 24 held-out words now makes the seed UNDEFINED. Uses the committed seed-42 shards."""
    import json
    import shutil
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "findings", "raw", "_stage1_exit", "gate", "s42")
    dst = tmp_path / "s42"
    dst.mkdir()
    for name in os.listdir(src):
        if name.startswith("val_heldout__") and name.endswith((".json", ".rep0.json")) and ".prov." not in name:
            shutil.copy(os.path.join(src, name), dst / name)
    reasons = lambda: SG.aggregate(str(tmp_path), seeds=[42], informational=False)[
        "groups"]["val_heldout"]["seeds"][42]["undefined_reasons"]
    assert any("deranged 0 of 24" in r for r in reasons())
    p = dst / "val_heldout__permuted.json"
    d = json.loads(p.read_text())
    d["patch"]["n_words"] = len(SG.HELDOUT_WORDS)
    p.write_text(json.dumps(d))
    assert not any("deranged" in r for r in reasons())
