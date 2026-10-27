"""research/runners/emo_flip_lines.py: the 72 gate-line TSV for the BRAIN_EMOTION_CATEGORY_BOUNDED flip
(research/coordination/emo-bounded-flip-gate-lines.txt). Nothing is queued by these tests.

What is pinned: the shape the prereg's section 3 fixes (60 Stage-2 emo lines + 12 row lines; ON adds the flag to the env block, OFF
leaves it unset); that every Stage-2 line IS the runner's own emitter's line with exactly three edits; that the scorer reads exactly
where the lines write; the pool_queue.sh gate-seed guard's own extractions (seeds, `prereg <path>` from --checked, the pinned
revision) on every line; and that the committed TSV is what the emitter prints.
"""
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import emo_flip_lines as L  # noqa: E402
from research.runners import emo_flip_score as S  # noqa: E402
from research.runners import stage2_exit_gate as G  # noqa: E402

FAKE = "0123456789abcdef0123456789abcdef01234567"          # a stand-in 40-hex revision for the substitution checks
TSV = os.path.join(REPO, "research", "coordination", "emo-bounded-flip-gate-lines.txt")
PREREG_PATH = "research/findings/2026-10-27-emo-bounded-flip-PREREGISTRATION.md"


@pytest.fixture(scope="module")
def lines():
    return L.all_lines()


def test_72_lines_60_stage2_and_12_rows_split_on_off(lines):
    assert len(lines) == 72
    s2 = [c for c, _ in lines if "research.runners.stage2_exit_gate" in c]
    rows = [c for c, _ in lines if "research.runners.load_bearing_fraction" in c]
    assert (len(s2), len(rows)) == (60, 12)
    on_s2 = [c for c in s2 if "gate_on/" in c]
    off_s2 = [c for c in s2 if "gate_off/" in c]
    assert (len(on_s2), len(off_s2)) == (30, 30)
    assert all(c.count("BRAIN_EMOTION_CATEGORY_BOUNDED=1") == 1 for c in on_s2)
    assert all("BRAIN_EMOTION_CATEGORY_BOUNDED" not in c for c in off_s2)                     # unset, not "0"
    assert sum("emo-bounded-flip-gate-on" in c for c in rows) == 6 and sum("emo-bounded-flip-gate-off" in c for c in rows) == 6
    assert all(c.count("BRAIN_EMOTION_CATEGORY_BOUNDED=1") == (1 if "gate-on" in c else 0) for c in rows)
    assert not any("BRAIN_EMOTION_CATEGORY_BOUNDED=0" in c for c, _ in lines)


def test_coverage_six_gate_seeds_five_arms_group_emo(lines):
    cells = set()
    for c, _ in lines:
        m = re.search(r"--shard --seed (\d+) --group (\S+) --arm (\S+) ", c)
        if m:
            cells.add((("on" if "gate_on/" in c else "off"), int(m.group(1)), m.group(2), m.group(3)))
    assert cells == {(f, s, "emo", a) for f in ("on", "off") for s in S.GATE_SEEDS for a in S.ARMS}
    row = set()
    for c, _ in lines:
        m = re.search(r"--only (\S+) --seed (\d+) --repeats 2 ", c)
        if m:
            row.add(("on" if "gate-on" in c else "off", m.group(1), int(m.group(2))))
    assert row == {(f, "emotion-category", s) for f in ("on", "off") for s in S.GATE_SEEDS}


def test_every_stage2_line_is_the_runners_own_line_with_three_edits(lines):
    for on in (True, False):
        out_dir = S.STAGE2_ON_DIR if on else S.STAGE2_OFF_DIR
        own = {}
        for cmd, why in G.pool_job_lines("<sha>", seeds=list(S.GATE_SEEDS), out_dir=out_dir, informational=False):
            m = re.search(r"--seed (\d+) --group emo --arm (\S+) ", cmd)
            if m:
                own[(int(m.group(1)), m.group(2))] = cmd
        mine = {}
        for cmd, _ in lines:
            m = re.search(r"--shard --seed (\d+) --group emo --arm (\S+) ", cmd)
            if m and ("gate_on/" in cmd) == on:
                mine[(int(m.group(1)), m.group(2))] = cmd
        assert set(mine) == {(s, a) for s in S.GATE_SEEDS for a in S.ARMS} and set(mine) <= set(own)
        for k, cmd in mine.items():
            back = cmd.replace("--prereg %s" % PREREG_PATH, "--prereg %s" % G.PREREG)
            if on:
                back = back.replace("-- env BRAIN_EMOTION_CATEGORY_BOUNDED=1 SIM_BACKEND=numpy", "-- env SIM_BACKEND=numpy")
            assert back == own[k], k                      # frozen harness, mem_ok 9 / memcap 10, --allow-gate-seed: untouched


def test_the_scorer_reads_exactly_where_the_lines_write(lines):
    import tools.lb_shard as LB
    for c, _ in lines:
        m = re.search(r"--out (\S+)$", c)
        out = m.group(1)
        sm = re.search(r"--shard --seed (\d+) --group emo --arm (\S+) ", c)
        if sm:
            d = S.STAGE2_ON_DIR if "gate_on/" in c else S.STAGE2_OFF_DIR
            assert out == G.shard_path(d, int(sm.group(1)), "emo", sm.group(2))
        else:
            seed = int(re.search(r"--seed (\d+)", c).group(1))
            tag = S.LB_ON_TAG if "gate-on" in c else S.LB_OFF_TAG
            assert out == LB.shard_out(tag, seed, S.LB_FACULTY)
            assert out == "%s/%s/s%d/%s/lb.json" % (S.LB_BASE, tag, seed, S.LB_FACULTY)       # what score_b8 / answer_rate open
    assert S.LB_FACULTY in LB.faculty_keys()                                                  # the registry row exists


def test_memory_is_the_stage2_emo_lines_figure(lines):
    for c, why in lines:
        assert "bash tools/mem_ok.sh 9 1 &&" in c and "mem_gb=9 max_age_h=240" in why
        assert ("memcap.sh 10 --" in c) == ("stage2_exit_gate" in c) and ("memcap.sh 12 --" in c) == ("load_bearing_fraction" in c)


def test_every_reason_names_the_prereg_first_and_is_one_clean_line(lines):
    for c, why in lines:
        assert "\t" not in c and "\t" not in why and "\n" not in c + why and all(ch.isprintable() for ch in c + why)
        m = re.search(r"prereg\s+(\S+)", why)
        assert m.group(1) == PREREG_PATH and "prereg" in os.path.basename(m.group(1)).lower()
        assert "<sha>" in c and "<sha>" in why


def test_the_pool_queue_gate_seed_guard_reads_each_substituted_line(lines):
    """pool_queue.sh add: seeds from --seed/--seeds, `prereg <path>` from --checked, the pinned revision from the command, and
    the prereg must exist at that revision. The extractions below are the guard's own greps, copied verbatim."""
    guard = r'''
JOB_SEED_NUMS=$(printf '%s' "$1" | grep -oE -- '--seeds?(=|[[:space:]]+)[0-9]+([,[:space:]]+[0-9]+)*' | grep -oE '[0-9]+')
PREREG_PATH=$(printf '%s' "$2" | grep -oE 'prereg[[:space:]]+[^[:space:]]+' | head -1 | awk '{print $2}')
GS_PINNED_REV=$(printf '%s' "$1" | grep -oE 'derisk-pool/revisions/[0-9a-f]{7,40}' | head -1 | sed 's#.*/##')
echo "$JOB_SEED_NUMS|$PREREG_PATH|$GS_PINNED_REV"
'''
    for cmd, why in lines:
        c, w = cmd.replace("<sha>", FAKE), why.replace("<sha>", FAKE)
        r = subprocess.run(["bash", "-c", guard, "_", c, w], capture_output=True, text=True, timeout=30)
        seeds, prereg, rev = r.stdout.strip().split("|")
        assert seeds.split() and set(seeds.split()) <= {str(s) for s in S.GATE_SEEDS} and len(set(seeds.split())) == 1
        assert prereg == PREREG_PATH and rev == FAKE


def test_substituted_lines_parse_as_shell_and_pass_the_queue_shape_check(lines):
    for cmd, _ in lines[::7] + lines[-2:]:
        c = cmd.replace("<sha>", FAKE)
        assert subprocess.run(["bash", "-n", "-c", c], capture_output=True, timeout=30).returncode == 0
        shape = subprocess.run(["bash", "-c", 'source tools/queue_job_shape_check.sh; queue_job_runnable_check "$1"', "_", c],
                               cwd=REPO, capture_output=True, text=True, timeout=60)
        assert shape.returncode == 0, shape.stderr


def test_the_gate_seed_guard_module_sees_a_gate_seed_and_the_prereg_on_the_stage2_lines(lines):
    from tools.gate_seed_guard import find_gate_hit_and_prereg
    for c, _ in lines:
        if "stage2_exit_gate" in c:
            assert find_gate_hit_and_prereg(c.replace("<sha>", FAKE)) == (True, PREREG_PATH)
        else:
            assert find_gate_hit_and_prereg(c.replace("<sha>", FAKE))[0] is True


def test_the_prereg_exists_in_this_tree_and_the_placeholder_substitutes_cleanly():
    assert os.path.exists(os.path.join(REPO, PREREG_PATH))
    assert "<sha>" not in L.render(FAKE) and L.render(FAKE).count(FAKE) >= 72


@pytest.mark.skipif(not os.path.exists(TSV), reason="the TSV is not committed in this checkout")
def test_the_committed_tsv_is_what_the_emitter_prints():
    assert open(TSV, encoding="utf8").read() == L.render()
    assert len(open(TSV, encoding="utf8").read().splitlines()) == 72
