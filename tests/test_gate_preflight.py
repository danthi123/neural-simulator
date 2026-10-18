"""tools/gate_preflight.py + its pool_queue.sh guard: no GATE-seed line until the registered scorer read a dev-seed run
of the same prereg at the same revision CLEAN (Claude 2026-10-16, owner-approved acceleration step)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gate_preflight as G  # noqa: E402

POOL_QUEUE = ROOT / "tools" / "pool_queue.sh"
HEAD_SHA = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
PREREG = "CLAUDE.md"
MOD = "tools.fake_canary_runner"


def test_classify_clean_needs_a_defined_cell_and_only_coverage_gaps():
    ok = '{"per_seed": {"7": {"state": "DEFINED"}}, "undefined_reasons": ["UNDEFINED: missing seeds 42,43"]}'
    assert G.classify(ok) == ("CLEAN", [])
    assert G.classify('{"verdict": "UNDEFINED (INCOMPLETE)"}')[0] == "FLAGGED"          # no DEFINED cell at all


def test_classify_flags_instrument_failures():
    for bad in ('{"7": {"state": "UNDEFINED:engagement-route-word_role_commit:intact_a_x.json"}} DEFINED',
                '{"7": {"state": "UNDEFINED:engagement-lesion-flag:lesion_x.json"}} DEFINED',
                'DEFINED\nTraceback (most recent call last):\n  File "x"'):
        v, why = G.classify(bad)
        assert v == "FLAGGED" and why, bad


def _env(tmp_path):
    env = dict(os.environ)
    env.update(GATE_PREFLIGHT_PATH=str(tmp_path / "preflight.tsv"), POOL_QUEUE_PATH=str(tmp_path / "pool.queue"),
               POOL_CANARIES_PATH=str(tmp_path / "pool_canaries.tsv"),
               POOL_SSH_CONFIG=str(tmp_path / "no-such-ssh-config"), POOL_EXTRA_NODES_FILE=str(tmp_path / "none"))
    (tmp_path / "pool_canaries.tsv").write_text(
        "2026-10-16T00:00:00-04:00\t%s\t%s\tPASS\tpool41\tgate\ttest\tfixture\n" % (MOD, HEAD_SHA))
    return env


def _pf(tmp_path, *args):
    return subprocess.run([sys.executable, str(ROOT / "tools" / "gate_preflight.py"), *args], cwd=ROOT,
                          env=_env(tmp_path), capture_output=True, text=True, timeout=60)


def _add(tmp_path, seed="42"):
    line = "cd ~/derisk-pool/revisions/%s && true -m %s --seed %s" % (HEAD_SHA, MOD, seed)
    return subprocess.run(["bash", str(POOL_QUEUE), "add", line, "--checked", "prereg %s mem_gb=2 max_age_h=24" % PREREG],
                          cwd=ROOT, env=_env(tmp_path), capture_output=True, text=True, timeout=60)


def test_queue_refuses_a_gate_line_with_no_preflight_and_accepts_after_a_clean_record(tmp_path):
    r = _add(tmp_path)
    assert r.returncode == 2 and "pre-flight" in r.stderr, r.stdout + r.stderr
    rec = _pf(tmp_path, "record", "--prereg", PREREG, "--revision", HEAD_SHA, "--log", str(tmp_path / "pf.log"),
              "--", sys.executable, "-c", "print('{\"7\": {\"state\": \"DEFINED\"}}')")
    assert rec.returncode == 0, rec.stdout + rec.stderr
    r2 = _add(tmp_path)
    assert r2.returncode == 0, r2.stderr


def test_a_flagged_record_blocks_until_a_newer_clean_one(tmp_path):
    rec = _pf(tmp_path, "record", "--prereg", PREREG, "--revision", HEAD_SHA, "--log", str(tmp_path / "pf.log"),
              "--", sys.executable, "-c", "print('UNDEFINED:engagement-route-x DEFINED')")
    assert rec.returncode == 1
    r = _add(tmp_path)
    assert r.returncode == 2 and "FLAGGED" in r.stderr
    _pf(tmp_path, "record", "--prereg", PREREG, "--revision", HEAD_SHA, "--log", str(tmp_path / "pf2.log"),
        "--", sys.executable, "-c", "print('DEFINED')")
    assert _add(tmp_path).returncode == 0


def test_ack_needs_a_ruling_in_the_runbook(tmp_path):
    r = _pf(tmp_path, "ack", "--prereg", PREREG, "--revision", HEAD_SHA, "--ruling", "no-such-ruling-zzz-1016")
    assert r.returncode != 0
    assert _pf(tmp_path, "check", "--prereg", PREREG, "--revision", HEAD_SHA).returncode == 2


def test_dev_seed_lines_need_no_preflight(tmp_path):
    assert _add(tmp_path, seed="7").returncode == 0


def test_ack_accepts_a_real_section_6_ruling(tmp_path):
    # 2026-10-18: ack split on "## RULINGS" but the header is "## 6. RULINGS (...)", so every ack was refused.
    r = _pf(tmp_path, "ack", "--prereg", PREREG, "--revision", HEAD_SHA, "--ruling", "smoke-first-1018")
    assert r.returncode == 0, r.stderr


def test_ruling_section_excludes_needs_and_needs_an_entry():
    import gate_preflight as gp
    rb = ("## 5. Idle\n- early-key: not a ruling section\n"
          "## 6. RULINGS (Claude writes)\n- alpha-key: do it (2026-10-18)\n- 2026-10-18 RULING beta-1018 (Claude): ok\n"
          "  continuation mentions gamma-key\n"
          "## NEEDS CLAUDE\n- 2026-10-18 10:00 [delta-key] ASK: please\n")
    assert gp.ruling_present(rb, "alpha-key") and gp.ruling_present(rb, "beta-1018")
    assert not gp.ruling_present(rb, "gamma-key")      # a mention inside another entry is not a ruling
    assert not gp.ruling_present(rb, "delta-key")      # NEEDS lines never count
    assert not gp.ruling_present(rb, "early-key")      # nor other sections
    assert not gp.ruling_present(rb, "beta")           # no prefix matches
