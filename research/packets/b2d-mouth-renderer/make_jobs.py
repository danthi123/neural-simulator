#!/usr/bin/env python3
"""make_jobs.py -- the B2d job files at a pin, for both arms (research/packets/b2d-mouth-renderer/HANDOFF.md).

  python research/packets/b2d-mouth-renderer/make_jobs.py stub  --pin <P> --out research/coordination/b2d_jobs.txt
  python research/packets/b2d-mouth-renderer/make_jobs.py mouth --pin <P> --seeds 42 43 44 100 101 102 \
      --out research/packets/b2d-mouth-renderer/jobs/mouth_<P7>.txt [--tag b2d0929-mouth]

stub  -- the b2d0929 arm (pool, numpy, stub renderer): exactly the pre-registration's Step 0 generation
         (`tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 --tag b2d0929 --no-fixes --probe-set adequate
         --repeats 2 --root '~/derisk-pool/revisions/<P>' --pin <P>`, then `.venv/bin/python
         tools/assert_flipped_defaults.py && ` inserted after the `cd`), 258 lines. Writes PIN.txt through lb_shard.
mouth -- the b2d0929-mouth arm (local GPU queue, cupy, LB_ARM_RENDERER=production): the same command shape for the
         three rows the pre-registration names (affect-coloring, affect-appraisal-interoceptive,
         affect-drives-response), with the same nine adequate LB_* probe flags, --repeats 2, one line per (seed, row),
         each under tools/mem_ok.sh + tools/memcap.sh. The only differences from the stub line are SIM_BACKEND=cupy,
         LB_ARM_RENDERER=production and the memory guard. Writes <shards>/<tag>/PIN.txt.

Both refuse a pin that is not a full 40-hex sha, and never queue anything.
"""
import argparse
import os
import re
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import lb_shard  # noqa: E402

SEEDS = [42, 43, 44, 100, 101, 102]
STUB_TAG = "b2d0929"
MOUTH_TAG = "b2d0929-mouth"
MOUTH_ROWS = ("affect-coloring", "affect-appraisal-interoceptive", "affect-drives-response")
ASSERT = ".venv/bin/python tools/assert_flipped_defaults.py && "
# mouth-arm memory: a cupy brain build plus the Qwen2.5-0.5B faculty in the worker; measured on dev seed 7 (HANDOFF §9)
MOUTH_MEM_OK = "bash tools/mem_ok.sh 10 4"
MOUTH_MEMCAP = "bash tools/memcap.sh 16 --"


def _check_pin(pin):
    if not re.fullmatch(r"[0-9a-f]{40}", pin or ""):
        raise SystemExit("pin must be a full 40-hex commit sha, got %r" % pin)


def stub_lines(pin, seeds=SEEDS, tag=STUB_TAG):
    """The pre-registration's Step 0 generation, verbatim, at `pin` (lb_shard.py jobs, then the assert insert)."""
    _check_pin(pin)
    root = "~/derisk-pool/revisions/%s" % pin
    cmd = [sys.executable, os.path.join(ROOT, "tools", "lb_shard.py"), "jobs", "--seeds"] + [str(s) for s in seeds] + [
        "--tag", tag, "--no-fixes", "--probe-set", "adequate", "--repeats", "2", "--root", root, "--pin", pin]
    env = dict(os.environ, SIM_NO_PROVENANCE="1")
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout
    prefix = "cd %s && " % root
    lines = []
    for ln in out.splitlines():
        if not ln.strip():
            continue
        if not ln.startswith(prefix):
            raise SystemExit("unexpected lb_shard line: %s" % ln[:120])
        lines.append(prefix + ASSERT + ln[len(prefix):])
    return lines


def mouth_lines(pin, seeds=SEEDS, tag=MOUTH_TAG, rows=MOUTH_ROWS):
    _check_pin(pin)
    probe_env = lb_shard.job_env("adequate", fixes=False)
    env = " ".join("%s=%s" % kv for kv in sorted(probe_env.items()))
    lines = []
    for seed in seeds:
        for row in rows:
            out = "%s/%s/s%d/%s/lb.json" % (lb_shard.OUT_BASE, tag, seed, row)
            lines.append(
                "cd ~/derisk-pool/revisions/%s && %smkdir -p %s && %s && %s env SIM_BACKEND=cupy OMP_NUM_THREADS=1 "
                "OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LB_ARM_RENDERER=production %s .venv/bin/python -u -m "
                "research.runners.load_bearing_fraction --only %s --seed %d --repeats 2 --out %s"
                % (pin, ASSERT, shlex.quote(os.path.dirname(out)), MOUTH_MEM_OK, MOUTH_MEMCAP, env,
                   shlex.quote(row), seed, shlex.quote(out)))
    return lines


def write_pin(tag, pin):
    pf = os.path.join(ROOT, lb_shard.OUT_BASE, tag, lb_shard.PIN_FILENAME)
    os.makedirs(os.path.dirname(pf), exist_ok=True)
    with open(pf, "w") as fh:
        fh.write(pin + "\n")
    return pf


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("arm", choices=("stub", "mouth"))
    ap.add_argument("--pin", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-pin-file", action="store_true", help="dev jobs: do not write the tag's PIN.txt")
    a = ap.parse_args(argv)
    if a.arm == "stub":
        tag = a.tag or STUB_TAG
        lines = stub_lines(a.pin, a.seeds, tag)
        if a.no_pin_file:
            pass  # lb_shard already wrote it for the tag; the stub arm always records its pin
    else:
        tag = a.tag or MOUTH_TAG
        lines = mouth_lines(a.pin, a.seeds, tag)
        if not a.no_pin_file:
            print("# pin recorded: %s" % write_pin(tag, a.pin), file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote %d %s lines (tag %s, pin %s) -> %s" % (len(lines), a.arm, tag, a.pin, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
