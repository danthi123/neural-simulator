"""Bundle a harness's unit JSONs into the one out.json the engine pulls (runs ON a node; stdlib only).

    python -m tools.engine.bundle DIR OUT [--glob 'u_*.json'] [--text 'time_*.txt'] [--t0 EPOCH]

For harnesses that write one atomic unit per (arm, seed, night) into a stable per-tree directory and resume by skipping
finished units (e.g. brain/experiments/mix_night_screen.py): the engine job runs the units, then this collects them.
Records each unit's mtime and the job wall time; --text also carries small text files verbatim (e.g. the /usr/bin/time -v
reports a measure-first step 0 is read from). Exits 1 (no out written) if no unit matched,
so an empty run is a failed attempt, never an empty result.
"""
import argparse
import fnmatch
import json
import os
import socket
import sys
import time


def bundle(d, pattern="u_*.json", t0=None, text=None):
    units, mtime, texts = {}, {}, {}
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if f.endswith(".tmp") or not os.path.isfile(p):
            continue
        if fnmatch.fnmatch(f, pattern):
            units[f] = json.load(open(p))
            mtime[f] = os.path.getmtime(p)
        elif text and fnmatch.fnmatch(f, text):
            texts[f] = open(p, errors="replace").read()[:65536]
    return {"dir": d, "units": units, "mtime": mtime, "texts": texts, "host": socket.gethostname(), "commit": os.environ.get("COMMIT"),
            "wall_s": round(time.time() - t0, 1) if t0 else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("out")
    ap.add_argument("--glob", default="u_*.json")
    ap.add_argument("--text", default=None)
    ap.add_argument("--t0", type=float, default=None)
    a = ap.parse_args(argv)
    b = bundle(a.dir, a.glob, a.t0, a.text)
    if not b["units"]:
        print(f"bundle: no {a.glob} in {a.dir}", file=sys.stderr)
        raise SystemExit(1)
    with open(a.out + ".tmp", "w") as fh:
        json.dump(b, fh)
    os.replace(a.out + ".tmp", a.out)
    print(f"bundle: {len(b['units'])} units -> {a.out}")


if __name__ == "__main__":
    main()
