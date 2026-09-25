"""Replay tools/queue_job_shape_check.sh over every historical queued job line and record what it refuses.

Inputs are the two append-only histories of what was actually handed to a shell:
  * research/queue/pool.queue.claims -- one `<epoch>\t<job>` record per job the pool dispatcher popped;
  * research/queue/gpu_queue.log      -- one `<date> <time> START: <job>` line per job the GPU daemon started.
Every job line is checked exactly as a producer checks it at enqueue time (`bash tools/queue_job_shape_check.sh
'<job>'`), and each refusal is written out with its source line number, so every refusal can be reviewed as a
true or a false refusal. The check itself is a static parser (it never runs any part of a job), so replaying
thousands of real historical lines through it is safe.

    python tools/queue_job_shape_replay.py \
        --claims research/queue/pool.queue.claims --gpu-log research/queue/gpu_queue.log \
        --out research/coordination/queue_shape_replay_2026-09-25.json

Read-only on its inputs; writes only --out (and --fixtures-dir, when given: a deduplicated sample of accepted and
refused lines for tests/test_queue_job_shape_check.py, keyed on the first four whitespace tokens, because the
check steps over a leading `: mem_gb=N &&` and so the command that decides the verdict is often the third token).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHECK = os.path.join(_ROOT, "tools", "queue_job_shape_check.sh")
_START = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} START: (.*)$")


def check(job: str) -> tuple[int, str]:
    r = subprocess.run(["bash", _CHECK, job], capture_output=True, text=True, timeout=30)
    return r.returncode, r.stderr.strip()


def load(claims: str | None, gpu_log: str | None) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    if claims:
        with open(claims, "rb") as fh:   # binary: split on \n only, so line numbers match grep -n / sed -n
            for n, raw in enumerate(fh, 1):
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if "\t" in line:
                    rows.append(("pool.queue.claims", n, line.split("\t", 1)[1]))
    if gpu_log:
        with open(gpu_log, "rb") as fh:   # the log holds bare \r progress output; text mode would renumber lines
            for n, raw in enumerate(fh, 1):
                m = _START.match(raw.decode("utf-8", "replace").rstrip("\n"))
                if m:
                    rows.append(("gpu_queue.log", n, m.group(1)))
    return rows


def _git_sha() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True, text=True)
    return r.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--claims")
    ap.add_argument("--gpu-log")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures-dir")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    rows = load(a.claims, a.gpu_log)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(lambda r: check(r[2]), rows))

    refused = []
    per_source: dict[str, dict[str, int]] = {}
    for (src, n, job), (rc, msg) in zip(rows, results):
        s = per_source.setdefault(src, {"lines": 0, "refused": 0})
        s["lines"] += 1
        if rc != 0:
            s["refused"] += 1
            refused.append({"source": src, "line": n, "job": job[:240], "message": msg[:300]})

    out = {
        "provenance": {
            "cmd": "python " + " ".join([os.path.relpath(__file__, _ROOT)] + (argv if argv is not None else sys.argv[1:])),
            "script": "tools/queue_job_shape_replay.py",
            "check": "tools/queue_job_shape_check.sh",
            # the check as it was on disk for this run (it may be uncommitted work when the replay runs)
            "check_sha256": hashlib.sha256(open(_CHECK, "rb").read()).hexdigest(),
            "device": "cpu (host bash; no simulator backend)",
            "git_sha": _git_sha(),
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        },
        "inputs": {"claims": a.claims, "gpu_log": a.gpu_log},
        "per_source": per_source,
        "refused_total": len(refused),
        "refused": refused,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")

    if a.fixtures_dir:
        os.makedirs(a.fixtures_dir, exist_ok=True)
        buckets: dict[str, dict[str, list[str]]] = {}
        seen: set[tuple[str, str, str]] = set()
        prefix = {"pool.queue.claims": "pool_history", "gpu_queue.log": "gpu_history"}
        for (src, _n, job), (rc, _msg) in zip(rows, results):
            if "\n" in job:
                continue
            verdict = "good" if rc == 0 else "bad"
            key = (src, verdict, " ".join(job.split()[:4]))
            if key in seen:
                continue
            seen.add(key)
            buckets.setdefault(prefix[src], {}).setdefault(verdict, []).append(job)
        for name, by_verdict in buckets.items():
            for verdict, jobs in by_verdict.items():
                with open(os.path.join(a.fixtures_dir, f"{name}_{verdict}.txt"), "w", encoding="utf-8") as fh:
                    fh.write("\n".join(jobs) + "\n")

    print(json.dumps({"per_source": per_source, "refused_total": len(refused)}))
    for r in refused:
        print(f"REFUSED {r['source']}:{r['line']}: {r['job'][:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
