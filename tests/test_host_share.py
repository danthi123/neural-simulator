"""Host-share instrument (research/runners/host_share.py + host_share_schema.py): aggregation logic and the
preregistered structure, with no brain build. The live arms are exercised by the dev run and the six-seed gate."""
import subprocess
import sys

from research.runners import host_share as HS
from research.runners import host_share_schema as SCHEMA


def test_runner_selftest_passes():
    ok, checks = HS.selftest(verbose=False)
    assert ok, {k: v for k, v in checks.items() if not v}


def test_schema_selftest_passes():
    ok, checks = SCHEMA.selftest()
    assert ok, {k: v for k, v in checks.items() if not v}


def test_host_causal_event_is_never_neural():
    # the audit's HIGH issue: F=1 with S=0 (host flip reaches the reply, neural rule agrees) must not count as neural
    assert SCHEMA.event_bucket(1, 0) == "host_causal_replaceable"
    p = SCHEMA.panel([dict(seam="k-select", live=True, F=1, S=0, seed=s) for s in (1, 2, 3, 4)], ["k-select"])
    assert p["neural_decision_fraction"] == 0.0 and p["host_decision_share"] == 1.0


def test_every_group_has_sham_and_each_seams_modes():
    for group, g in HS.GROUPS.items():
        arms = HS.arms_for(group)
        assert arms[:2] == ["intact_a", "intact_b"]
        for seam in g["seams"]:
            for mode in HS.SEAM_SPEC[seam]["modes"]:
                assert "%s:%s" % (seam, mode) in arms


def test_every_group_has_target_turns_with_cues():
    for group in HS.GROUPS:
        specs = HS.turn_specs(group)
        targets = [sp for sp in specs.values() if sp["target"]]
        assert targets and all(sp["cue"] for sp in targets)


def test_all_six_seams_are_live_somewhere_and_measured_split_is_frozen():
    assert {s for g in HS.GROUPS.values() for s in g["seams"]} == set(HS.SEAMS)
    assert set(HS.MEASURED_SEAMS) == {"k-select", "v-choice", "bg-select"}


def test_ltm_tier_is_pinned_and_expected_source_declared():
    assert HS.BASE_ENV.get("BRAIN_LTM_SHIP_DEFAULT") == "1" and HS.EXPECTED_SOURCE == "tiny-demo +LTM"


def test_gate_seed_is_refused_without_the_flag():
    p = subprocess.run([sys.executable, "-m", "research.runners.host_share", "--seed-all", "--seed", "42",
                        "--out-dir", "/nonexistent"], capture_output=True, text=True)
    assert p.returncode != 0 and "GATE seed" in p.stderr


def test_pool_job_lines_are_pinned_and_sized():
    lines = HS.pool_job_lines("0123456789abcdef")
    assert len(lines) == len(HS.GATE_SEEDS) * sum(len(HS.arms_for(g)) for g in HS.GROUPS)
    for cmd, why in lines:
        assert cmd.startswith("cd ~/derisk-pool/revisions/0123456789abcdef && ")
        assert "--allow-gate-seed" in cmd and "mem_gb=7" in why


def test_classifier_boundaries():
    st = lambda **kw: dict(dict(live_seeds=[1, 2, 3, 4], HS=None, NS=None, MS=None, SC=None), **kw)  # noqa: E731
    assert SCHEMA.classify_seam(st(HS=0.8, SC=0.2)) == "HOST-OWNED"
    assert SCHEMA.classify_seam(st(HS=0.8, SC=0.1)) == "HOST-OWNED (replaceable)"
    assert SCHEMA.classify_seam(st(HS=0.1, NS=0.9)) == "NEURAL-OWNED"
    assert SCHEMA.classify_seam(st(HS=0.1, NS=0.1, MS=0.9)) == "MASKED"
    assert SCHEMA.classify_seam(st(HS=0.5, NS=0.3, MS=0.2)) == "MIXED"
    assert SCHEMA.classify_seam(dict(st(HS=1.0), live_seeds=[1, 2, 3])) == "UNDEFINED"
