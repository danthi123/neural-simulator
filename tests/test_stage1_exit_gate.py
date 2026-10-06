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


def _ratings_and_values():
    """24 human ratings, 12 per sign, and a learned-map-like value vector strongly tracking them."""
    norms = [5.4 + 0.3 * i for i in range(12)] + [4.6 - 0.3 * i for i in range(12)]
    vals = [0.9 * n + 0.35 * ((i % 5) - 2) for i, n in enumerate(norms)]
    return vals, norms


def test_scrambled_draw_never_used_with_r_above_the_draw_bar():
    """RULING s1-c4-control: the C4 scrambled-values draw is rejection sampled, so a permutation whose |r(scrambled
    values, human ratings)| exceeds 0.10 is never used, for every gate seed."""
    vals, norms = _ratings_and_values()
    for seed in (42, 43, 44, 100, 101, 102):
        perm, r, within = SG._scrambled_valence_draw(vals, norms, seed)
        assert len(perm) == len(vals) and all(p != i for i, p in enumerate(perm)), (seed, perm)   # derangement
        assert r is None or abs(r) <= SG.SCRAMBLE_R_MAX, (seed, r)
        assert set(within) == {"positive", "negative"}, within                                    # report-only


def test_scrambled_draw_is_fixed_by_the_seed_and_rejects_naive_breaching_shuffles():
    """The RNG is fixed (seed = the brain seed): same seed, same draw. With values identical to the ratings, a naive
    shuffle of a monotone sequence correlates far past the bar — every accepted draw must still land within it."""
    _, norms = _ratings_and_values()
    vals = list(norms)
    p1, r1, _ = SG._scrambled_valence_draw(vals, norms, 42)
    p2, r2, _ = SG._scrambled_valence_draw(vals, norms, 42)
    assert p1 == p2 and r1 == r2
    for seed in (42, 43, 44, 100, 101, 102):
        _p, r, _w = SG._scrambled_valence_draw(vals, norms, seed)
        assert r is None or abs(r) <= SG.SCRAMBLE_R_MAX, (seed, r)


def test_permuted_map_patch_records_the_drawn_permutation_and_its_r():
    """The run output carries the drawn permutation, its r, and the report-only within-sign-group r."""
    from research.runners import affect_production_organ as AO
    saved = AO._LEARNED_VALENCE
    rec = None
    try:
        rec = SG.install_patch("permuted_map", 42)
    finally:
        AO._LEARNED_VALENCE = saved
    assert rec is not None
    assert rec["patch"] == "permuted_map" and rec["n_words"] >= 12
    assert len(rec["perm"]) == rec["n_words"] and all(p != i for i, p in enumerate(rec["perm"]))
    assert rec["scrambled_r"] is None or abs(rec["scrambled_r"]) <= SG.SCRAMBLE_R_MAX, rec["scrambled_r"]
    assert set(rec["within_sign_r"]) == {"positive", "negative"}


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


def test_no_effect_env_lesion_without_engagement_record_is_undefined(tmp_path):
    """Amendment A1.2: attempt 1's C3 lesion (BRAIN_SUBSTRATE_RECALL_LESION) cut a method the production bus path never
    calls, and its no-effect reply was scored as "not load-bearing". A no-effect env-flag lesion on a seed whose
    capability is present now makes the seed UNDEFINED; a patch lesion with a call count is scored as before."""
    import json
    import shutil
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "research", "findings", "raw", "_stage1_exit", "gate", "s42")
    dst = tmp_path / "s42"
    dst.mkdir()
    for name in os.listdir(src):
        if name.startswith("lived__") and ".prov." not in name and name.endswith(".json"):
            shutil.copy(os.path.join(src, name), dst / name)
    row = lambda: SG.aggregate(str(tmp_path), seeds=[42], informational=False)["groups"]["lived"]["seeds"][42]
    r = row()
    assert r["correct"] is True
    assert any("lesion_recall: an env-flag lesion" in u for u in r["undefined_reasons"])
    for name in ("lived__lesion_recall.json", "lived__lesion_recall.rep0.json"):   # the same shards with an engaged patch
        d = json.loads((dst / name).read_text())
        d["patch"] = {"patch": "lesion_lived_recall", "calls": 3, "calls_total": 9}
        (dst / name).write_text(json.dumps(d))
    r = row()
    assert not any("lesion_recall: an env-flag lesion" in u for u in r["undefined_reasons"])
    assert r["lesions"]["lesion_recall"]["applied"] is True
