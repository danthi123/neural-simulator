#!/usr/bin/env python3
"""battery_status.py -- read-only battery-readiness accounting for tools/status.sh and a battery's harvest step.

Reads a TSV registry (default: research/coordination/handoff_batteries.tsv) with one row per battery:
    name<TAB>raw_glob<TAB>expected_rows<TAB>harvest_cmd<TAB>finding_template

For each battery it counts LANDED rows (files matching `raw_glob`, resolved relative to --root, EXCLUDING any
path ending in `.prov.json` -- provenance sidecars are not result rows) and reports READY-TO-HARVEST when
landed >= expected AND the battery is not currently running.

"Currently running" is a crude substring check against --running-text/--running-stdin (free text describing
what the GPU queue + pool nodes are doing right now): if a battery's `name` appears in that text, it counts as
RUNNING regardless of its landed count. Crude ON PURPOSE -- a name collision can only produce a false RUNNING
(never a false READY), so this never tells the operator to harvest a battery that might still be writing files.

Never writes anything. Two modes:
    battery_status.py --tsv <path> [--root <dir>] [--running-stdin | --running-text <str>]
        one line per battery: NAME<TAB>landed<TAB>expected<TAB>STATUS   (STATUS: READY | RUNNING | WAITING)
    battery_status.py --tsv <path> [--root <dir>] --harvest <name> [--running-stdin | --running-text <str>]
        the battery's harvest_cmd + finding_template, then its landed files, one per line (sorted)
"""
from __future__ import annotations

import argparse
import glob as globmod
import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Battery:
    name: str
    raw_glob: str
    expected_rows: int
    harvest_cmd: str
    finding_template: str


def parse_tsv(path: str) -> list[Battery]:
    """Parse the registry. `#`-lines and blank lines are skipped; a `name` header row (if present) is skipped
    too, so the file may be read either with or without its header for convenience."""
    out: list[Battery] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cells = line.split("\t")
            if cells[0] == "name":
                continue
            if len(cells) != 5:
                raise ValueError(f"{path}:{lineno}: expected 5 tab-separated fields, got {len(cells)}: {line!r}")
            name, raw_glob, expected, harvest_cmd, template = cells
            try:
                expected_i = int(expected)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: expected_rows {expected!r} is not an integer") from exc
            out.append(Battery(name.strip(), raw_glob.strip(), expected_i, harvest_cmd.strip(), template.strip()))
    return out


def landed_files(root: str, raw_glob: str) -> list[str]:
    """Files matching `raw_glob` (repo-relative, may use `**`) under `root`, excluding *.prov.json sidecars."""
    pattern = os.path.join(root, raw_glob)
    matches = globmod.glob(pattern, recursive=True)
    return sorted(m for m in matches if not m.endswith(".prov.json") and os.path.isfile(m))


def is_running(name: str, running_text: str) -> bool:
    return bool(name) and name in running_text


def status_line(b: Battery, root: str, running_text: str) -> tuple[str, int, int, str]:
    landed = len(landed_files(root, b.raw_glob))
    if is_running(b.name, running_text):
        state = "RUNNING"
    elif landed >= b.expected_rows:
        state = "READY"
    else:
        state = "WAITING"
    return b.name, landed, b.expected_rows, state


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", required=True, help="path to the battery registry TSV")
    ap.add_argument("--root", default=os.getcwd(), help="repo root raw_glob is resolved against (default: cwd)")
    ap.add_argument("--running-stdin", action="store_true", help="read free-text 'what's running' from stdin")
    ap.add_argument("--running-text", default="", help="free-text 'what's running' (alternative to --running-stdin)")
    ap.add_argument("--harvest", metavar="NAME", help="print one battery's harvest recipe + landed files")
    args = ap.parse_args(argv)

    running_text = sys.stdin.read() if args.running_stdin else args.running_text
    try:
        batteries = parse_tsv(args.tsv)
    except (OSError, ValueError) as exc:
        print(f"battery_status: {exc}", file=sys.stderr)
        return 2

    if args.harvest:
        matches = [b for b in batteries if b.name == args.harvest]
        if not matches:
            print(f"battery_status: no battery named {args.harvest!r} in {args.tsv}", file=sys.stderr)
            return 1
        b = matches[0]
        name, landed, expected, state = status_line(b, args.root, running_text)
        print(f"# {name}: {landed}/{expected} rows landed, status={state}")
        print(f"# harvest_cmd: {b.harvest_cmd}")
        print(f"# finding_template: {b.finding_template}")
        if state != "READY":
            print(f"# NOTE: not READY yet -- listing landed files anyway (read-only, no gate to bypass).")
        for f in landed_files(args.root, b.raw_glob):
            print(f)
        return 0

    for b in batteries:
        name, landed, expected, state = status_line(b, args.root, running_text)
        print(f"{name}\t{landed}\t{expected}\t{state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
