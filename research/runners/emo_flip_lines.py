"""Gate-line emitter for the BRAIN_EMOTION_CATEGORY_BOUNDED default-ON flip (prereg section 3; `research/coordination/emo-bounded-flip-gate-lines.txt`).

    python -m research.runners.emo_flip_lines [<sha>] > research/coordination/emo-bounded-flip-gate-lines.txt

72 TSV pool lines (`command<TAB>--checked reason`), `<sha>` left as a placeholder unless one is given (gate revision F: the pool's
`~/derisk-pool/revisions/<F>`). It QUEUES NOTHING.

  * 60 Stage-2 emo lines: `stage2_exit_gate.pool_job_lines` for the six gate seeds, group emo, the five arms (intact_a, intact_b,
    permuted, lesion_affect, lesion_affect.rep0), ON and OFF. Taken from the runner's own emitter and edited in exactly three places:
    `--prereg` names the FLIP prereg, `--out` is under research/findings/raw/_emo_bounded_flip/gate_{on,off}, and the ON half adds
    `BRAIN_EMOTION_CATEGORY_BOUNDED=1` to the `env` block (the OFF half leaves the variable absent: unset, not "0").
    `mem_gb=9` under a 10 GB memcap, as the 10-26 Stage-2 gate lines.
  * 12 row lines: `tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 --tag emo-bounded-flip-gate-{on,off} --no-fixes --probe-set thin
    --faculties emotion-category [--extra-env BRAIN_EMOTION_CATEGORY_BOUNDED=1]`, wrapped in `mem_ok.sh 9 1` and `memcap.sh 12`
    (mem_gb=9 like the Stage-2 lines: the same tiny-demo brain; cap 12 as the earlier lb_shard row lines, because the battery's parent
    process shares the arm subprocess's cgroup and the dev smoke recorded no peak for this row).
STALENESS CAVEAT (written with the default-ON flip branch research/emo-bounded-default-on): "OFF = variable absent" is OFF only at a
revision where an unset BRAIN_EMOTION_CATEGORY_BOUNDED reads OFF, which is gate revision F and every revision before the flip. At a
revision that carries the flip an unset variable builds the bounded organ, so these OFF lines would run ON (the flip_offarm_staleness
pattern, `tools/gates/flip_offarm_staleness.py`): re-emit them with `BRAIN_EMOTION_CATEGORY_BOUNDED=0` for such a revision. Two reads
of the scorer refuse the mix: B5 (the OFF s101 intact category must reproduce the 10-26 gate's joy None) and B8's OFF-row cell check
(an OFF cell showing bounded_flag True is UNDEFINED, "the flag leaked in").
Every reason names `prereg <path>` first (the pool's gate-seed guard reads the prereg from --checked and requires a file whose name
contains 'prereg'); it carries no tab or control character.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import emo_flip_score as S  # noqa: E402
from research.runners import stage2_exit_gate as G  # noqa: E402

PLACEHOLDER = "<sha>"
FLAG_ENV = "BRAIN_EMOTION_CATEGORY_BOUNDED=1"
PREREG = S.PREREG
SEEDS = list(S.GATE_SEEDS)
MEM_GB = G.MEM_GB                    # 9, as the 10-26 Stage-2 lines
STAGE2_MEMCAP_GB = G.MEMCAP_GB       # 10
ROW_MEMCAP_GB = 12
MAX_AGE_H = G.MAX_AGE_H              # 240
RECORD = ("record: research/findings/2026-10-26-emotion-organ-bounded-conductance-derisk.md holds the organ-level and the seed-101 "
          "diagnostic evidence only, so no six-gate-seed reading of this flag exists; the scorer and bar-sanity records are filed "
          "in the prereg's Amendment 1")


def stage2_lines(sha=PLACEHOLDER):
    out = []
    for on in (True, False):
        out_dir = S.STAGE2_ON_DIR if on else S.STAGE2_OFF_DIR
        for cmd, _why in G.pool_job_lines(sha, seeds=SEEDS, out_dir=out_dir, informational=False, max_age_h=MAX_AGE_H):
            m = re.search(r"--seed (\d+) --group (\S+) --arm (\S+) ", cmd)
            seed, group, arm = int(m.group(1)), m.group(2), m.group(3)
            if group != "emo" or arm not in S.ARMS:
                continue
            assert cmd.count("--prereg %s" % G.PREREG) == 1 and cmd.count("-- env SIM_BACKEND=numpy") == 1
            cmd = cmd.replace("--prereg %s" % G.PREREG, "--prereg %s" % PREREG)
            if on:
                cmd = cmd.replace("-- env SIM_BACKEND=numpy", "-- env %s SIM_BACKEND=numpy" % FLAG_ENV)
            why = ("prereg %s (F %s); emo-bounded-flip gate seed %d emo %s flag %s (Stage-2 runner emo group, %s); "
                   "%s; mem_gb=%d max_age_h=%d" % (PREREG, sha, seed, arm, "ON" if on else "OFF",
                                                   FLAG_ENV if on else "BRAIN_EMOTION_CATEGORY_BOUNDED unset", RECORD,
                                                   MEM_GB, MAX_AGE_H))
            out.append((cmd, why))
    return out


def row_lines(sha=PLACEHOLDER):
    import tools.lb_shard as LB
    out = []
    for on in (True, False):
        tag = S.LB_ON_TAG if on else S.LB_OFF_TAG
        ns = argparse.Namespace(seeds=SEEDS, tag=tag, root="~/derisk-pool/revisions/%s" % sha, repeats=2,
                                faculties=[S.LB_FACULTY], extra_env=([FLAG_ENV] if on else None), probe_set="thin",
                                no_fixes=True, pin=None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            LB.cmd_jobs(ns)
        for ln in buf.getvalue().splitlines():
            m = re.match(r"^(cd \S+) && (mkdir -p \S+) && env (.*)$", ln)
            assert m, ln
            seed = int(re.search(r"--seed (\d+)", ln).group(1))
            cmd = "%s && bash tools/mem_ok.sh %d 1 && %s && bash tools/memcap.sh %d -- env %s" % (
                m.group(1), MEM_GB, m.group(2), ROW_MEMCAP_GB, m.group(3))
            why = ("prereg %s (F %s); emo-bounded-flip gate seed %d %s row flag %s (tools/lb_shard.py jobs --tag %s --no-fixes "
                   "--probe-set thin%s); %s; mem_gb=%d max_age_h=%d" % (
                       PREREG, sha, seed, S.LB_FACULTY, "ON" if on else "OFF", tag,
                       " --extra-env %s" % FLAG_ENV if on else "", RECORD, MEM_GB, MAX_AGE_H))
            out.append((cmd, why))
    return out


def all_lines(sha=PLACEHOLDER):
    return stage2_lines(sha) + row_lines(sha)


def render(sha=PLACEHOLDER):
    ls = all_lines(sha)
    for cmd, why in ls:
        assert "\t" not in cmd and "\t" not in why and "\n" not in cmd and "\n" not in why
    return "".join("%s\t%s\n" % (c, w) for c, w in ls)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sha", nargs="?", default=PLACEHOLDER, help="gate revision F (default: the literal %s)" % PLACEHOLDER)
    a = ap.parse_args(argv)
    sys.stdout.write(render(a.sha))
    return 0


if __name__ == "__main__":
    sys.exit(main())
