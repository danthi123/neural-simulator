#!/usr/bin/env python3
"""tools/backlog.py — the mechanical WORK-BACKLOG GENERATOR (a scanner, NOT a dispatcher).

WHY THIS EXISTS. The project's full set of parallelizable work — de-risked-but-not-flipped faculties,
open walls, unclosed failures, findings' open residuals, host-scaffold burndowns, board next-rungs —
was only ever surfaced AD HOC, when the owner asked and Claude "looked from many angles". A dispatcher
cannot fan out to ALL available independent work if the enumeration is a judgement call made freshly
each time. This tool enumerates that work MECHANICALLY, from the tracked record, the SAME way every
run, so a downstream dispatcher/ratchet can consume a stable ranked list instead of a vibe.

IT IS A SCANNER. It reads files + the Vikunja board and prints/writes a ranked list. It NEVER spawns an
agent, launches a runner, queues a job, or steps a brain. Consuming the list (fan-out) is a separate,
deliberately-separate concern (see `--how` for the intended dispatcher contract).

SOURCES — one scanner FUNCTION each, so adding a source is one function + one line in `SCANNERS`:
  (1) scan_ledger_flips     PRODUCTION_INTEGRATION_LEDGER rows de_risked==YES & on_by_default!=YES
                            (the flip backlog — the owner's #1 lever: a proven faculty not yet default-on)
  (2) scan_walls_ledger     the roadmap §7 walls-ledger OPEN walls (each carries its biological surpass)
  (3) scan_failure_log      research/FAILURE_LOG.md entries with NO enforcing gate yet (NOT-GATEABLE /
                            empty gate — the open coverage gaps, with their stated candidate mitigation)
  (4) scan_findings         recent LIVE findings' "next lever / residual / open work" sections
  (5) scan_vikunja          open Vikunja board tasks + their stated next-rungs (the live board)
  (6) scan_ledger_scaffolds ledger rows on_by_default==YES but scaffold_retired!=YES (host shortcuts to
                            convert — the burn-down-to-one-brain backlog)
  (7) scan_next_actions     GAP_CLOSURE_MISSION.md forward-looking EXACT-NEXT lines that spell out a
                            genuinely-runnable command (the richest source of ready FREE-LANE work)

FREE-LANE `cmd` (the seam this generator closes). For the SUBSET of items that are genuinely runnable on a
free lane (a real runner + args exists — de-risk sweeps, verifications, soaks, param sweeps) the generator
emits `cmd` (the exact runnable command), a free `lane` (gpu-queue | pool-cpu), and structured
`dependencies` (ids this item is blocked on). Items that need a mind (builds/wiring/design) carry NO cmd →
they route to the agent lane. Until this existed, NO item carried a cmd, so the ratchet's free-lane
auto-dispatch was a known no-op; now the pool + GPU queue AUTO-FILL from the backlog.

ANTI-FABRICATION (the one hard guardrail). Every emitted item MUST trace to a real source line/anchor
(a file:line, a finding path, a vikunja #id). A `cmd` is minted ONLY when it names a research.runners
module that EXISTS on disk, carries no unresolved placeholder ($VAR / <X> / ellipsis), and — if it declares
an output artifact — that artifact is not already present; a command that cannot be derived truthfully is
left off (the item stays cmd-less). A scanner that finds nothing emits nothing — this tool NEVER invents
filler to inflate the count. `--selftest` proves both directions: that it surfaces the KNOWN current
backlog + mints a real cmd, AND that a scanner returning empty (or a cmd naming a fabricated module / a
placeholder template / an already-done run) FAILS LOUDLY.

OUTPUT: research/coordination/backlog.json (machine) + a compact human table (stdout). Items not already
represented on the Vikunja board are flagged `on_board=false` — a clearly-labelled "new (not on board)"
delta for reconciliation. This tool does NOT create board tasks (adding is a later, confirmed step).

    .venv/bin/python tools/backlog.py                 # scan, write json, print the ranked table
    .venv/bin/python tools/backlog.py --top 15        # table capped to the top 15
    .venv/bin/python tools/backlog.py --new-only      # only the "not on board" delta
    .venv/bin/python tools/backlog.py --json          # emit the full json to stdout
    .venv/bin/python tools/backlog.py --no-vikunja    # skip the network (file sources only)
    .venv/bin/python tools/backlog.py --selftest      # pass + demonstrated failing direction
    .venv/bin/python tools/backlog.py --how           # how a dispatcher/ratchet should consume this
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "docs", "PRODUCTION_INTEGRATION_LEDGER.yaml")
ROADMAP = os.path.join(ROOT, "docs", "plans", "2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md")
FAILURE_LOG = os.path.join(ROOT, "research", "FAILURE_LOG.md")
FINDINGS_DIR = os.path.join(ROOT, "research", "findings")
MISSION = os.path.join(ROOT, "GAP_CLOSURE_MISSION.md")
VIKUNJA = os.path.join(ROOT, "tools", "vikunja.sh")
OUT_JSON = os.path.join(ROOT, "research", "coordination", "backlog.json")

# Leverage weights by source (higher == dispatch first). Production-default flips are the owner's #1
# lever; open walls carry the biological surpass; board p0 work is owner-tracked; scaffolds + residuals
# are real but lower-yield. Per-item boosts (below) refine within a source.
BASE_LEVERAGE = {
    "ledger-flip":      100,
    "next-action":       82,   # a live board EXACT-NEXT that carries a genuinely-runnable command (free-lane)
    "vikunja":           80,
    "walls-ledger":      65,
    "ledger-scaffold":   45,
    "finding-residual":  35,
    "failure-log":       30,
}


# ─────────────────────────────────────────────────────────────────────────────
# lane + verify heuristics (best-effort, per the CLAUDE.md cost-routing rule)
# ─────────────────────────────────────────────────────────────────────────────
_GPU_HINTS = ("cupy", "gpu", "e-prop", "eprop", "train", "credit", "deep-credit", "scale",
              "spiking port", "on-spikes", "on the substrate", "btsp store", "reservoir",
              "microcircuit", "backprop", "sweep", "6-seed", "6 seed", "consolidation soak")
_CPU_HINTS = ("de-risk", "derisk", "grid", "tune", "param", "sweep", "smoke", "6-seed", "numpy-cpu")
_BUILD_HINTS = ("wire", "wiring", "integrat", "flip", "default-on", "on-by-default", "retire",
                "build", "gate", "port", "endpoint", "handler", "merge", "organ", "production")


def suggest_lane(text: str) -> str:
    """agent (build/integration judgment) | pool-cpu (CPU de-risk/sweep) | gpu-queue (GPU/train)."""
    t = text.lower()
    if any(h in t for h in _GPU_HINTS) and not any(h in t for h in ("flip", "wire", "retire", "endpoint")):
        return "gpu-queue"
    if any(h in t for h in _BUILD_HINTS):
        return "agent"
    if any(h in t for h in _CPU_HINTS):
        return "pool-cpu"
    return "agent"


def _slug(*parts: str) -> str:
    s = "-".join(p for p in parts if p)
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:64] or "item"


# ─────────────────────────────────────────────────────────────────────────────
# runnable-command extraction — THE SEAM this change closes. A backlog item is only
# free-lane AUTO-dispatchable if it carries a `cmd`; until now none did, so the
# ratchet's free-lane dispatch was a known no-op. ANTI-FABRICATION (hard rule): a
# `cmd` is minted ONLY when it (1) names a research.runners module that EXISTS on
# disk, (2) carries NO unresolved placeholder ($VAR / <X> / ellipsis / TODO), and
# (3) — when it declares an output artifact — that artifact is not already present
# (a completed run is not "ready"). A command we cannot derive truthfully is left
# off and the item stays cmd-less (→ agent lane, or NEEDS-COMMAND). We NEVER invent
# seeds/args to make an item look dispatchable.
# ─────────────────────────────────────────────────────────────────────────────
RUNNERS_DIR = os.path.join(ROOT, "research", "runners")

# a full command: optional leading ENV=val, the SANCTIONED interpreter (.venv/bin/python — the pool
# guard refuses a bare `python`), then `-m research.runners.MODULE ...`, non-greedily up to the next
# command-start or end-of-span (so multiple commands in one fenced block split cleanly).
_CMD_RE = re.compile(
    r"((?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*(?:\S*\.venv/bin/python\S*)\s+(?:-u\s+)?"
    r"-m\s+research\.runners\.[A-Za-z0-9_.]+.*?)"
    r"(?=\s+(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*\S*\.venv/bin/python|$)")
_PLACEHOLDER_RE = re.compile(r"\$\w|\$\{|<[A-Za-z]|\.\.\.|\bTODO\b|\{\{")


def _code_spans(text: str):
    """Inline-code / fenced-code spans, fence-aware (``` is consumed atomically so it cannot mis-pair
    the single-backtick scan). Every governed source embeds commands inside code spans."""
    spans, i, n = [], 0, len(text)
    while i < n:
        if text.startswith("```", i):
            j = text.find("```", i + 3)
            if j < 0:
                break
            spans.append(text[i + 3:j]); i = j + 3; continue
        if text[i] == "`":
            j = text.find("`", i + 1)
            if j < 0:
                break
            spans.append(text[i + 1:j]); i = j + 1; continue
        i += 1
    return spans


def _clean_cmd_span(s: str) -> str:
    s = re.sub(r"\n\s*>?\s*", " ", s)      # join blockquote-wrapped continuation lines (`> ...`)
    s = re.sub(r"\\\s+", " ", s)           # drop shell line-continuation backslashes
    return re.sub(r"\s+", " ", s).strip()


def _runner_module(cmd: str):
    m = re.search(r"-m\s+research\.runners\.([A-Za-z0-9_]+)", cmd or "")
    return m.group(1) if m else None


def _runner_exists(mod: str) -> bool:
    return bool(mod) and os.path.isfile(os.path.join(RUNNERS_DIR, mod + ".py"))


def _cmd_output_exists(cmd: str) -> bool:
    m = re.search(r"--(?:out|json|emit-map)\s+(\S+)", cmd or "")
    if not m:
        return False
    p = m.group(1)
    return os.path.exists(p if os.path.isabs(p) else os.path.join(ROOT, p))


def _lane_for_cmd(cmd: str) -> str:
    """Route by EXPLICIT backend signal only. Default to the GPU queue (a runner's native backend here
    is CuPy; the singleton GPU queue serialises it safely) — pool-cpu is chosen only for commands the
    author explicitly marked numpy, so a CuPy-only runner is never mis-routed onto a GPU-less pool node."""
    t = (cmd or "").lower()
    if "sim_backend=cupy" in t or "--backend cupy" in t or "--backend rf" in t or "--ckpt" in t:
        return "gpu-queue"
    if "sim_backend=numpy" in t or "--backend numpy" in t:
        return "pool-cpu"
    return "gpu-queue"


def extract_runnable_cmd(text: str):
    """Return (cmd, lane) for the FIRST genuinely-runnable command in `text`, else (None, None).

    This is the ONLY place a cmd is minted; every step is an anti-fabrication / anti-stale gate."""
    if not text or "research.runners." not in text:
        return None, None
    for s in _code_spans(text):
        if "research.runners." not in s:
            continue
        for m in _CMD_RE.finditer(_clean_cmd_span(s)):
            cmd = m.group(1).strip().rstrip("`").strip()
            cmd = re.split(r"\s+(?:;|&&|\|\||#)\s+", cmd)[0].strip()   # stop at a shell separator/comment
            mod = _runner_module(cmd)
            if not _runner_exists(mod):
                continue     # anti-fabrication: names no real module → not a real command
            if _PLACEHOLDER_RE.search(cmd):
                continue     # a template ($s / <S> / ...) is not runnable as-is → don't fabricate args
            if _cmd_output_exists(cmd):
                continue     # its declared artifact exists → run is DONE, not ready
            return cmd, _lane_for_cmd(cmd)
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# ledger parsing (shared by the flip + scaffold scanners)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_ledger_rows(text: str):
    """Yield (key, level_dict, host_scaffold, default_files, start_lineno) per `- key:` block.

    level_dict has de_risked/wired/on_by_default/scaffold_retired mapped to YES/PARTIAL/NO/? .
    Values may be bare (YES) or quoted ("YES"); comments after the value are ignored.
    """
    lines = text.split("\n")
    # find each row-start line so we can report a real anchor
    starts = [i for i, ln in enumerate(lines) if re.match(r"\s*-\s*key:\s*\S", ln)]
    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        block = lines[s:e]
        km = re.match(r"\s*-\s*key:\s*\"?([A-Za-z0-9_\-]+)", block[0])
        if not km:
            continue
        key = km.group(1)
        levels, host, dfiles = {}, "", []
        for ln in block:
            for f in ("de_risked", "wired", "on_by_default", "scaffold_retired"):
                if f not in levels:
                    m = re.match(r"\s*%s:\s*\"?([A-Za-z]+)" % f, ln)
                    if m:
                        levels[f] = m.group(1).upper()
            m = re.match(r"\s*host_scaffold_in_default:\s*\"?(.+)", ln)
            if m and not host:
                host = m.group(1).strip().strip('"').rstrip('"')
            m = re.match(r"\s*file:\s*(\S+)", ln)
            if m:
                dfiles.append(m.group(1))
        for f in ("de_risked", "wired", "on_by_default", "scaffold_retired"):
            levels.setdefault(f, "?")
        yield key, levels, host, dfiles, s + 1


def _faculty_desc(text: str, key: str) -> str:
    m = re.search(r"-\s*key:\s*\"?%s\"?\s*\n\s*faculty:\s*\"?(.+)" % re.escape(key), text)
    if not m:
        return key
    return m.group(1).strip().strip('"').rstrip('"')[:160]


# ─────────────────────────────────────────────────────────────────────────────
# (1) ledger FLIPS — de-risked but not on-by-default (the highest-leverage backlog)
# ─────────────────────────────────────────────────────────────────────────────
def scan_ledger_flips(ledger_text: str = None):
    if ledger_text is None:
        ledger_text = _read(LEDGER)
    if not ledger_text:
        return []
    items = []
    for key, lv, host, dfiles, lineno in _parse_ledger_rows(ledger_text):
        if lv["de_risked"] == "YES" and lv["on_by_default"] != "YES":
            desc = _faculty_desc(ledger_text, key)
            target = dfiles[0] if dfiles else "webapp/server.py + research/runners/*_production_organ.py"
            wired_note = "" if lv["wired"] == "YES" else " (wire first: wired=%s)" % lv["wired"]
            items.append(_item(
                source="ledger-flip",
                key="flip-" + key,
                what="Flip DE-RISKED faculty '%s' to ON-BY-DEFAULT%s" % (key, wired_note),
                detail=desc,
                anchor="docs/PRODUCTION_INTEGRATION_LEDGER.yaml:%d" % lineno,
                target=target,
                verify="build the DEFAULT ChatBrain, assert the faculty runs with no opt-in flag + a "
                       "lesion of its path flips the probe act (research/runners/_production_lesion_probe.py); 6-seed",
                deps="none (de_risked=YES)" if lv["wired"] == "YES" else "wire into /api/brain-chat first",
                lane="agent",
                boost=20 if lv["wired"] == "YES" else 5,
            ))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# (6) ledger SCAFFOLD burn-downs — on-by-default rows still carrying a host shortcut
# ─────────────────────────────────────────────────────────────────────────────
def scan_ledger_scaffolds(ledger_text: str = None):
    if ledger_text is None:
        ledger_text = _read(LEDGER)
    if not ledger_text:
        return []
    items = []
    for key, lv, host, dfiles, lineno in _parse_ledger_rows(ledger_text):
        # a burn-down item only when the faculty is otherwise live (on-by-default) but the host
        # shortcut it replaces is NOT yet gone. Flip rows (on_by_default!=YES) are handled by (1),
        # so this set is disjoint from the flip set by construction (dedup-by-design).
        if lv["on_by_default"] == "YES" and lv["scaffold_retired"] != "YES":
            desc = host or _faculty_desc(ledger_text, key)
            items.append(_item(
                source="ledger-scaffold",
                key="scaffold-" + key,
                what="Retire host scaffold for '%s' (scaffold_retired=%s)" % (key, lv["scaffold_retired"]),
                detail=desc[:200],
                anchor="docs/PRODUCTION_INTEGRATION_LEDGER.yaml:%d" % lineno,
                target=dfiles[0] if dfiles else "the faculty's production organ / webapp/server.py",
                verify="convert the named host shortcut to a spiking/synaptic mechanism (or bank an "
                       "honest negative); the default answer must stay stable (byte-identical escape)",
                deps="the spiking replacement must reach parity or an honest negative",
                lane="agent",
                boost=10 if lv["scaffold_retired"] == "PARTIAL" else 0,
            ))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# (2) roadmap §7 WALLS LEDGER — open walls + their named biological surpass
# ─────────────────────────────────────────────────────────────────────────────
_WALL_CLOSED_RE = re.compile(
    r"\bCLOSED\b|\bSURPASSED\b|\bRESOLVED\b|fully closed|closed end-to-end", re.I)
_WALL_OPEN_RE = re.compile(
    r"\bWALL\b|\bNO-GO\b|\bopen\b|\bFRONTIER\b|IN PROGRESS|residual|\bnext\b|not yet|BOUNDARY|"
    r"not the answer|un-?wired|never (been )?(run|combined)|has not been run", re.I)


def scan_walls_ledger(roadmap_text: str = None):
    if roadmap_text is None:
        roadmap_text = _read(ROADMAP)
    if not roadmap_text:
        return []
    lines = roadmap_text.split("\n")
    # find the §7 table header, then take table rows until the table ends
    hdr = None
    for i, ln in enumerate(lines):
        if ln.startswith("## 7.") and "WALLS LEDGER" in ln.upper():
            hdr = i
            break
    if hdr is None:
        return []
    items = []
    for j in range(hdr, len(lines)):
        ln = lines[j]
        if j > hdr and ln.startswith("## "):  # next section => table done
            break
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        wall_id, desc, status, surpass, retire = cells[0], cells[1], cells[2], cells[3], cells[4]
        if wall_id in ("#", "") or set(wall_id) <= set("-: "):     # header / separator rows
            continue
        # open unless the STATUS is unambiguously closed with no residual language
        closed = bool(_WALL_CLOSED_RE.search(status)) and not _WALL_OPEN_RE.search(status)
        if closed:
            continue
        name = re.sub(r"\*\*|\[mechanism:[^\]]*\]", "", wall_id).strip()
        mechs = re.findall(r"\[mechanism:\s*([^\]]+)\]", wall_id)
        items.append(_item(
            source="walls-ledger",
            key="wall-" + _slug(name),
            what="Open wall: %s — %s" % (name, desc[:90]),
            detail=("SURPASS: " + re.sub(r"\*\*", "", surpass)[:220]) if surpass else status[:200],
            anchor="docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md:%d" % (j + 1),
            target="research/runners/ (build the named biological surpass) — mechanism(s): %s"
                   % (", ".join(mechs) or "n/a"),
            verify="run the surpass mechanism vs a like-for-like control with anti-cheats; 6-seed GO "
                   "(the residual must move the right way, attributable, lesion-decisive)",
            deps=("retire/close at: " + retire[:80]) if retire else "",
            lane=suggest_lane(desc + " " + surpass),
            boost=10 if "gap#4" in wall_id or "gap#5" in wall_id else 0,
        ))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# (3) FAILURE_LOG — noticed failures with NO enforcing gate yet (open coverage gaps)
# ─────────────────────────────────────────────────────────────────────────────
def _failure_rows(text: str):
    for i, ln in enumerate(text.split("\n")):
        if not ln.strip().startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < 3 or not re.match(r"^\d{4}-\d{2}-\d{2}$", cells[0]):
            continue
        yield cells[0], cells[1], cells[2], i + 1


def scan_failure_log(log_text: str = None):
    if log_text is None:
        log_text = _read(FAILURE_LOG)
    if not log_text:
        return []
    items = []
    for date, failure, gate, lineno in _failure_rows(log_text):
        gate_clean = gate.strip("`").strip()
        is_notgateable = gate_clean.upper().startswith("NOT-GATEABLE")
        is_empty = not gate_clean
        if not (is_empty or is_notgateable):
            continue  # names a real gate/test => CLOSED, not backlog
        cand = ""
        cm = re.search(r"[Cc]andidate[:,]?\s*(.+)", gate_clean)
        if cm:
            cand = cm.group(1).strip().rstrip("`")
        # a bare "NOT-GATEABLE project-wide/generically" with no candidate is an accepted non-gap:
        # keep it, but rank it lowest (informational-open), so the count stays honest without noise.
        actionable = bool(cand) or is_empty
        items.append(_item(
            source="failure-log",
            key="failgap-" + _slug(date, failure[:40]),
            what="Open coverage gap (%s): %s" % (date, failure[:90]),
            detail=("CANDIDATE: " + cand[:220]) if cand else (gate_clean[:200] or "no gate named"),
            anchor="research/FAILURE_LOG.md:%d" % lineno,
            target="tools/gates/<new_module>.py (registry gate) or a tests/ regression",
            verify="new gate: implement check(paths)+selftest() that FAILS in its failing direction; "
                   "the registry refuses a gate whose selftest cannot fail",
            deps="",
            lane="agent",
            boost=8 if actionable else -15,
        ))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# (4) recent LIVE findings — "next lever / residual / open work" sections
# ─────────────────────────────────────────────────────────────────────────────
_RESID_HDR_RE = re.compile(
    r"^#{1,4}\s+.*(next lever|next rung|next step|next arc|residual|open work|open lever|remaining|"
    r"the wall|frontier|what.?s (left|next)|next experiment)", re.I)
_RESID_INLINE_RE = re.compile(
    r"^\**\s*(named next lever|next lever|next rung|next step|next\s*=|open work|residual\s*[:=]|"
    r"remaining\s*[:=])", re.I)


def _filename_date(path: str) -> str:
    m = re.match(r"(\d{4}-\d{2}-\d{2})-", os.path.basename(path))
    return m.group(1) if m else "0000-00-00"


def scan_findings(findings_dir: str = None, max_recent: int = 45, days: int = 45):
    findings_dir = findings_dir or FINDINGS_DIR
    files = [p for p in glob.glob(os.path.join(findings_dir, "*.md"))
             if re.match(r"\d{4}-\d{2}-\d{2}-", os.path.basename(p))]
    if not files:
        return []
    # recency by the FILENAME date, not mtime — a git checkout / worktree resets every mtime to now, which
    # would make "recent" arbitrary. The filename date is the finding's real date and is deterministic.
    files.sort(key=_filename_date, reverse=True)
    cutoff = time.strftime("%Y-%m-%d", time.localtime(time.time() - days * 86400))
    files = [p for p in files if _filename_date(p) >= cutoff][:max_recent]
    items = []
    seen_mech = set()
    for p in files:
        txt = _read(p)
        if not txt:
            continue
        # only LIVE findings carry a forward-open residual worth dispatching
        if re.search(r"^status:\s*(superseded|retracted|contributing|void)", txt, re.M | re.I):
            continue
        mm = re.search(r"^mechanism:\s*(\S+)", txt, re.M)
        mech = mm.group(1) if mm else os.path.basename(p)
        lines = txt.split("\n")
        hit_text, hit_line = None, None
        for i, ln in enumerate(lines):
            if _RESID_HDR_RE.match(ln) or _RESID_INLINE_RE.match(ln):
                # capture this line + the next couple of non-blank prose lines as the residual
                body = []
                for k in range(i + 1, min(i + 6, len(lines))):
                    s = lines[k].strip()
                    if s.startswith("#") or s.startswith("|"):
                        break
                    if s:
                        body.append(s)
                    if len(" ".join(body)) > 200:
                        break
                hit_text = re.sub(r"^#{1,4}\s+", "", ln).strip()
                if body:
                    hit_text += " — " + " ".join(body)
                hit_line = i + 1
                break
        if not hit_text:
            continue
        # a section headed "residual" whose BODY reports a closure (CLOSED/RESOLVED/GO) is not open
        # backlog — skip it (traceable non-fabrication: the header matched, the content says done).
        body_part = hit_text.split("—", 1)[-1].strip()
        if re.match(r"(CLOSED|RESOLVED|GO\b|GO:|none\b|no residual)", body_part, re.I):
            continue
        if mech in seen_mech:        # one residual per mechanism (dedup near-duplicate findings)
            continue
        seen_mech.add(mech)
        rel = os.path.relpath(p, ROOT)
        # if the next-lever SECTION (from the residual header to the next header) spells out a genuinely
        # runnable command, attach it → this item becomes free-lane AUTO-dispatchable, not agent-only.
        sect_end = len(lines)
        for k in range(hit_line, min(hit_line + 40, len(lines))):
            if lines[k].startswith("#") and k > hit_line:
                sect_end = k
                break
        section = "\n".join(lines[hit_line - 1:sect_end])
        cmd, cmd_lane = extract_runnable_cmd(section)
        items.append(_item(
            source="finding-residual",
            key="resid-" + _slug(mech),
            what="Residual/next-lever: %s" % hit_text[:110],
            detail=hit_text[:240],
            anchor="%s:%d" % (rel, hit_line),
            target="research/runners/ (per the finding's named next lever)",
            verify="run the named next lever with anti-cheats + like-for-like control; 6-seed if a claim",
            deps="mechanism: " + mech,
            lane=cmd_lane if cmd else suggest_lane(hit_text),
            cmd=cmd,
            dependencies=[],
            boost=0,
        ))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# (5) Vikunja board — open tasks + their next-rungs (the live board)
# ─────────────────────────────────────────────────────────────────────────────
# Vikunja priority: 5=DO-NOW/north-star meta, 4=urgent .. 1=low, 0=unset. The two meta rows (the
# north-star + the stage description) are not dispatchable work; everything else is.
_BOARD_META_IDS = set()  # populated by title heuristic below


def scan_vikunja(timeout: int = 25, project: int = 2):
    reachable = True
    try:
        raw = subprocess.run(
            ["bash", VIKUNJA, "--json", "list-tasks", str(project)],
            capture_output=True, text=True, timeout=timeout).stdout
        tasks = json.loads(raw)
        if not isinstance(tasks, list):
            tasks = []
    except Exception:
        return [], False
    items = []
    for t in tasks:
        if t.get("done"):
            continue
        pr = t.get("priority") or 0
        title = str(t.get("title", "")).strip()
        desc = re.sub(r"<[^>]+>", "", (t.get("description") or "")).strip()
        low = (title + " " + desc).lower()
        # skip the north-star / current-stage META rows (they describe the mission, not a task)
        if title.lower().startswith(("the goal", "current stage")) or pr >= 5:
            continue
        if "parked" in low and "later" in low:
            lane = "agent"
        else:
            lane = suggest_lane(low)
        # extract a next-rung sentence from the description if one is signposted
        rung = ""
        rm = re.search(r"(OPEN work|BLOCKED on|next|residual|the wall|remaining)[^.]{0,180}",
                       desc, re.I)
        if rm:
            rung = rm.group(0).strip()
        # an open board task whose body carries a genuinely-runnable command is free-lane work (open ⇒ not
        # yet done). The task description is HTML-stripped above, but code spans usually survive as backticked
        # text — pass the original description too so fenced/inline commands are seen.
        cmd, cmd_lane = extract_runnable_cmd((t.get("description") or "") + "\n" + desc)
        items.append(_item(
            source="vikunja",
            key="vik-%s" % t.get("id"),
            what="[board #%s p%d] %s" % (t.get("id"), pr, title[:90]),
            detail=(rung or desc)[:240],
            anchor="vikunja#%s" % t.get("id"),
            target="(see task) — production wiring / de-risk per the task body",
            verify="per the task's acceptance; production faculties need the lesion probe + 6-seed",
            deps="",
            lane=cmd_lane if cmd else lane,
            cmd=cmd,
            dependencies=[],
            boost=pr * 4 + (10 if "parked" not in low else -30),
            on_board=True,
        ))
    return items, reachable


# ─────────────────────────────────────────────────────────────────────────────
# (7) live board NEXT-ACTIONS — forward-looking runnable commands on the working board
# (GAP_CLOSURE_MISSION.md). This is the richest source of genuinely-ready FREE-LANE work:
# the owner writes the exact command to run next. Each emitted item carries a `cmd`
# (subject to the same anti-fabrication gates as everywhere else) so the ratchet can
# auto-dispatch it with zero agent tokens.
# ─────────────────────────────────────────────────────────────────────────────
_NEXT_SIGNPOST_RE = re.compile(
    r"NEXT|to run|run cheap|cheap-first|queue this|kick off|launch|when GPU|should run|TODO|next lever|"
    r"next rung|EXACT NEXT", re.I)


def scan_next_actions(mission_text: str = None):
    if mission_text is None:
        mission_text = _read(MISSION)
    if not mission_text:
        return []
    lines = mission_text.split("\n")
    items, seen = [], set()
    for i, ln in enumerate(lines):
        if "research.runners." not in ln:
            continue
        # forward-looking gate: a runnable command counts as a NEXT-ACTION only near a forward signpost
        # (so a past-tense "we ran X" results line is not mistaken for ready work). Look at a small window.
        ctx = " ".join(lines[max(0, i - 3):i + 1])
        if not _NEXT_SIGNPOST_RE.search(ctx):
            continue
        # the command can wrap across several blockquote lines — scan from here to the closing backtick
        block = "\n".join(lines[i - 1:i + 8]) if i > 0 else "\n".join(lines[i:i + 8])
        cmd, lane = extract_runnable_cmd(block)
        if not cmd:
            continue
        mod = _runner_module(cmd)
        if mod in seen:                     # one item per runner module (dedup near-duplicate next-actions)
            continue
        seen.add(mod)
        items.append(_item(
            source="next-action",
            key="next-" + _slug(mod),
            what="Run next-action (%s): %s" % (lane, mod),
            detail=cmd[:240],
            anchor="GAP_CLOSURE_MISSION.md:%d" % (i + 1),
            target="research/findings/raw/ (the command's declared output)",
            verify="the command completes and writes its artifact; read the verdict per the board's acceptance",
            deps="",
            lane=lane,
            cmd=cmd,
            dependencies=[],
            boost=0,
        ))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# item construction, dedup, reconciliation, ranking
# ─────────────────────────────────────────────────────────────────────────────
def _item(source, key, what, detail, anchor, target, verify, deps, lane, boost=0, on_board=False,
          cmd=None, dependencies=None):
    return {
        "id": key,
        "source": source,
        "what": what,
        "detail": detail,
        "anchor": anchor,
        "target": target,
        "verify": verify,
        "deps": deps,                     # PROSE dependency note (human + the ratchet's prose heuristic)
        "dependencies": list(dependencies or []),  # STRUCTURED: backlog ids this item is blocked on
        "cmd": cmd,                       # a genuinely-runnable command (free-lane) or None (agent lane)
        "lane": lane,
        "leverage": BASE_LEVERAGE.get(source, 20) + boost,
        "on_board": on_board,
        "sources": [source],
        "related_anchors": [],
    }


# canonical-token aliases: map the many phrasings of one faculty onto a shared strong token so a
# ledger flip, a board task, and a wall about the SAME thing reconcile instead of triple-counting.
_ALIASES = {
    "gap4": ["gap#4", "gap4", "deep credit", "deep-credit", "speak-with-own-neuron", "own neurons",
             "mouth", "e-prop", "eprop", "read-snr", "read snr", "credit across", "deep learning across"],
    "gap5": ["gap#5", "gap5", "memory-unblurring", "memory unblurring", "swr", "replay recall",
             "dendritic dap", "dap readout"],
    "d5consol": ["d5-live-consolidation", "d5 live consolidation", "learn-through-use",
                 "learn through use", "consolidation on the real brain", "unblurring to work on the real"],
    "dopamine": ["da-gated-encoding", "da-mode", "dopamine decide its mode", "da-gated", "snc"],
    "vision": ["object-anywhere", "object anywhere", "recognize an object wherever", "hmax",
               "visual identity", "position invarian", "perception-motor", "spiking readout"],
    "tiered": ["tiered-knowledge-ltm", "tiered knowledge", "llm-scale knowledge", "sharded fact"],
    "gnw3": ["gnw-three-organ-bus", "three-organ", "third", "comprehension monitor.*bus"],
    "gnwswap": ["gnw-thought-swap", "thought-swap", "thought swap"],
    "selfreward": ["self-model-reward-residual", "reward/value signal", "standalone reward"],
}


def _canon(text: str):
    """Return the set of canonical strong-tokens present in text (for conservative cross-source merge)."""
    t = text.lower()
    out = set()
    for canon, needles in _ALIASES.items():
        for n in needles:
            if re.search(n, t):
                out.add(canon)
                break
    return out


def dedup(items):
    """Conservative cross-source merge: fold items sharing a canonical strong-token into the
    highest-leverage representative, unioning sources + anchors. Items with no shared canonical token
    are NEVER merged (over-separation is safer than fabricated collapse)."""
    # sort so the highest-leverage item is the survivor
    items = sorted(items, key=lambda x: -x["leverage"])
    survivors, claimed = [], {}
    for it in items:
        toks = _canon(it["what"] + " " + it["detail"])
        home = None
        for tk in toks:
            if tk in claimed:
                home = claimed[tk]
                break
        if home is not None and home["source"] != it["source"]:
            # merge into the existing higher-leverage representative
            if it["source"] not in home["sources"]:
                home["sources"].append(it["source"])
            home["related_anchors"].append("%s(%s)" % (it["anchor"], it["source"]))
            home["on_board"] = home["on_board"] or it["on_board"]
            # never lose a real runnable command: if the survivor has none but a merged sibling does,
            # inherit it (and the sibling's free lane) so the merged item stays AUTO-dispatchable.
            if not home.get("cmd") and it.get("cmd"):
                home["cmd"] = it["cmd"]
                home["lane"] = it["lane"]
            for d in it.get("dependencies", []):
                if d not in home["dependencies"]:
                    home["dependencies"].append(d)
            continue
        survivors.append(it)
        for tk in toks:
            claimed.setdefault(tk, it)
    return survivors


def reconcile_board(items, board_items):
    """Set on_board for every non-board item by matching its text against the live board blob."""
    blob = " ".join((b["what"] + " " + b["detail"]).lower() for b in board_items)
    board_canon = set()
    for b in board_items:
        board_canon |= _canon(b["what"] + " " + b["detail"])
    for it in items:
        if it["source"] == "vikunja":
            it["on_board"] = True
            continue
        toks = _canon(it["what"] + " " + it["detail"])
        # matched if a canonical faculty token is shared, OR the ledger/wall key stem appears verbatim
        stem = it["id"].split("-", 1)[-1].replace("-", " ")
        it["on_board"] = bool(toks & board_canon) or (len(stem) > 6 and stem in blob)
    return items


# a real prerequisite in the PROSE deps (mirrors the ratchet's is_blocked vocabulary): an item that
# "must ... first / reach parity / requires / is blocked / pending / not yet" cannot START yet.
_PROSE_BLOCKED_RE = re.compile(
    r"\bfirst\b|must (reach|be|land|pass|exist|complete)|reach parity|\brequires?\b|depends on|"
    r"\bblocked\b|\bpending\b|\bawait|not yet|prerequisite", re.I)


def link_dependencies(items):
    """Populate STRUCTURED `dependencies` (backlog ids) — conservatively. A dependency id is added ONLY
    for an item whose PROSE deps already say it is blocked, and ONLY when an OPEN WALL for the SAME faculty
    (its biological surpass — the thing that must be built first) is present in this backlog. This never
    invents a block on a ready item, never contradicts the prose, and cannot cycle (walls are the sole
    upstream). It refines a prose block into concrete ids the ratchet can wait on; free-lane runnable
    items (a cmd, no prose block) stay independent (dependencies == [])."""
    upstream = {}   # canonical faculty token -> [wall ids] (the prerequisite to build first)
    for it in items:
        if it["source"] == "walls-ledger":
            for tk in _canon(it["what"] + " " + it["detail"]):
                upstream.setdefault(tk, []).append(it["id"])
    for it in items:
        if not _PROSE_BLOCKED_RE.search(it.get("deps") or ""):
            continue
        for tk in _canon(it["what"] + " " + it["detail"]):
            for dep_id in upstream.get(tk, []):
                if dep_id != it["id"] and dep_id not in it["dependencies"]:
                    it["dependencies"].append(dep_id)
    return items


# ─────────────────────────────────────────────────────────────────────────────
# driver
# ─────────────────────────────────────────────────────────────────────────────
def _read(path):
    try:
        return open(path, errors="ignore").read()
    except Exception:
        return ""


SCANNERS = [
    ("ledger-flip", scan_ledger_flips),
    ("ledger-scaffold", scan_ledger_scaffolds),
    ("walls-ledger", scan_walls_ledger),
    ("failure-log", scan_failure_log),
    ("finding-residual", scan_findings),
    ("next-action", scan_next_actions),
]


def generate(use_vikunja=True):
    items, notes = [], []
    for name, fn in SCANNERS:
        got = fn()
        items += got
        if not got:
            notes.append("scanner %s produced 0 items" % name)
    board_items, reachable = ([], False)
    if use_vikunja:
        board_items, reachable = scan_vikunja()
        if not reachable:
            notes.append("vikunja UNREACHABLE — board source skipped (not fabricated)")
    items += board_items
    items = dedup(items)
    items = reconcile_board(items, board_items)
    items = link_dependencies(items)      # structured deps (ids) derived after the full set + merges exist
    items = sorted(items, key=lambda x: (-x["leverage"], x["source"], x["id"]))
    for rank, it in enumerate(items, 1):
        it["rank"] = rank
    return items, {"vikunja_reachable": reachable, "notes": notes,
                   "n_board_open": len(board_items)}


def _render_table(items, meta, top=None, new_only=False, verbose=False):
    shown = [it for it in items if (it["on_board"] is False)] if new_only else items
    if top:
        shown = shown[:top]
    out = []
    out.append("=" * 100)
    out.append("WORK BACKLOG  —  %d independent items  (%s)  [board: %s]"
               % (len(items),
                  " · ".join("%s=%d" % (s, sum(1 for i in items if i["source"] == s))
                             for s, _ in SCANNERS + [("vikunja", None)]),
                  "reachable, %d open" % meta["n_board_open"] if meta["vikunja_reachable"]
                  else "UNREACHABLE"))
    out.append("=" * 100)
    out.append("%-4s %-5s %-17s %-9s %-4s %s" % ("#", "lev", "source", "lane", "new?", "what"))
    out.append("-" * 100)
    for it in shown:
        out.append("%-4d %-5d %-17s %-9s %-4s %s"
                   % (it["rank"], it["leverage"], it["source"], it["lane"],
                      "NEW" if not it["on_board"] else "", it["what"][:60]))
        if verbose:
            out.append("       └ %s  |  verify: %s" % (it["anchor"], it["verify"][:72]))
            if it["sources"] and len(it["sources"]) > 1:
                out.append("         merged from: %s" % ", ".join(it["sources"]))
    out.append("-" * 100)
    n_new = sum(1 for i in items if not i["on_board"])
    out.append("NEW (not on Vikunja board): %d items — reconcile the board before dispatch (this tool "
               "does NOT auto-create tasks)." % n_new)
    out.append("(anchors + verify + deps per item in the JSON; -v for inline anchors; --how for the "
               "dispatcher contract)")
    if meta["notes"]:
        out.append("notes: " + " ; ".join(meta["notes"]))
    return "\n".join(out)


HOW = """\
HOW A DISPATCHER / RATCHET SHOULD CONSUME research/coordination/backlog.json
───────────────────────────────────────────────────────────────────────────
This tool is the ENUMERATOR half of an enforced-parallelism engine; it never dispatches. A dispatcher
(or the heartbeat's parallel_audit) consumes the JSON like so:

  1. Read backlog.json (regenerate first: `python tools/backlog.py --no-vikunja` for a fast file-only pass,
     or the full pass when the board matters).
  2. Count in-flight lanes (parallel_audit.py already does: local cores, pool cores, GPU, live agents).
  3. Take the top-ranked items whose `lane` matches an IDLE capacity, in rank order, until capacity fills:
        lane=="pool-cpu"  -> tools/sweep_pool.sh          (0 agent tokens)
        lane=="gpu-queue" -> tools/gpu_queue.sh add '...' (sequential, VRAM-safe)
        lane=="agent"     -> a model-tiered subagent      (builds/wiring only; reserve tokens)
  4. Respect `deps`: an item whose deps are unmet is skipped, not dispatched.
  5. `on_board==false` items are the reconcile delta — surface them to the owner / add to Vikunja in a
     SEPARATE confirmed step before treating them as tracked work.
  6. The ratchet check: UNDER-PARALLELIZED iff (idle capacity for a lane) AND (a ready item for that lane
     with met deps) AND (in-flight lanes < ready items). That is the same verdict parallel_audit prints,
     now backed by the FULL mechanical enumeration instead of the board alone.
"""


# ─────────────────────────────────────────────────────────────────────────────
# selftest — proves BOTH directions (required)
# ─────────────────────────────────────────────────────────────────────────────
def selftest():
    problems = []

    # ---- PASS DIRECTION: the scanners surface the KNOWN current backlog ----
    flips = scan_ledger_flips()
    flip_keys = {it["id"] for it in flips}
    expected_flips = ["gnw-three-organ-bus", "da-gated-encoding", "d5-live-consolidation",
                      "tiered-knowledge-ltm", "self-model-reward-residual"]
    for k in expected_flips:
        if ("flip-" + k) not in flip_keys:
            problems.append("PASS-DIR: flip scanner missed the known de-risked-but-not-flipped faculty %r" % k)

    walls = scan_walls_ledger()
    if not any("gap#4" in w["what"] or "gap4" in w["id"] for w in walls):
        problems.append("PASS-DIR: walls scanner surfaced no gap#4 open wall (it is open per the record)")
    if len(walls) < 1:
        problems.append("PASS-DIR: walls scanner surfaced zero open walls")

    fails = scan_failure_log()
    if len(fails) < 1:
        problems.append("PASS-DIR: failure-log scanner surfaced zero unclosed coverage gaps "
                        "(the log has many NOT-GATEABLE rows)")

    scaffolds = scan_ledger_scaffolds()
    if len(scaffolds) < 5:
        problems.append("PASS-DIR: scaffold scanner surfaced <5 host-shortcut burn-downs "
                        "(the ledger headline says scaffold_retired total = 0)")

    # ---- FAILING DIRECTION: an empty scanner MUST be caught when the source has items ----
    # simulate a BYPASSED / broken flip scanner (returns []) while the source CLEARLY has flip rows.
    def _bypassed_flip_scanner(_txt=None):
        return []
    source_clearly_has_items = bool(
        re.search(r"de_risked:\s*\"?YES", _read(LEDGER))
        and re.search(r"on_by_default:\s*\"?NO", _read(LEDGER)))
    caught = _guard_scanner_nonempty("ledger-flip", _bypassed_flip_scanner,
                                     source_has_items=source_clearly_has_items)
    if not caught:
        problems.append("FAIL-DIR: the guard did NOT catch a flip scanner returning empty while the "
                        "ledger clearly holds de_risked=YES / on_by_default=NO rows")
    # and the guard must NOT fire when empty is legitimate (source genuinely has no items)
    if _guard_scanner_nonempty("x", lambda: [], source_has_items=False):
        problems.append("FAIL-DIR: the guard FALSE-fired on a legitimately-empty source")
    # the real scanner over an emptied source must return [] (no fabrication)
    if scan_ledger_flips(ledger_text="") != []:
        problems.append("FAIL-DIR: flip scanner FABRICATED items from empty source text")

    # ---- RUNNABLE-COMMAND EXTRACTION (the seam) — pass + anti-fabrication failing direction ----
    real_mod = None
    try:
        for f in sorted(os.listdir(RUNNERS_DIR)):
            if f.endswith(".py") and not f.startswith("__"):
                real_mod = f[:-3]
                break
    except OSError:
        pass
    if not real_mod:
        problems.append("PASS-DIR(cmd): no runner module found on disk to test extraction against")
    else:
        # PASS: a real module, explicit numpy, no output artifact → a pool-cpu cmd is minted
        c, lane = extract_runnable_cmd(
            "run `SIM_BACKEND=numpy .venv/bin/python -m research.runners.%s --seed 1`" % real_mod)
        if not c or ("research.runners." + real_mod) not in c or lane != "pool-cpu":
            problems.append("PASS-DIR(cmd): a real numpy runner command was not extracted as a pool-cpu cmd")
        # PASS: a cupy/GPU command routes to the GPU queue
        _, glane = extract_runnable_cmd(
            "`SIM_BACKEND=cupy .venv/bin/python -m research.runners.%s --seed 1`" % real_mod)
        if glane != "gpu-queue":
            problems.append("PASS-DIR(cmd): a cupy command did not route to gpu-queue (got %r)" % glane)
        # FAIL-DIR anti-fabrication: a nonexistent module yields NO command (never invented)
        if extract_runnable_cmd(
                "`.venv/bin/python -m research.runners._NOT_A_REAL_MODULE_ZZZ99 --seed 1`") != (None, None):
            problems.append("FAIL-DIR(cmd): a command naming a NONEXISTENT runner module was minted (fabrication)")
        # FAIL-DIR: a placeholder template is not runnable-as-is → NO command
        if extract_runnable_cmd(
                "`.venv/bin/python -m research.runners.%s --seed $s`" % real_mod) != (None, None):
            problems.append("FAIL-DIR(cmd): a placeholder ($s) template was minted as a runnable command")
        # FAIL-DIR anti-stale: a command whose declared output already exists is DONE, not ready
        if extract_runnable_cmd(
                "`.venv/bin/python -m research.runners.%s --out tools/backlog.py`" % real_mod) != (None, None):
            problems.append("FAIL-DIR(cmd): a command whose output artifact already exists was minted as ready")

    # every cmd the scanners actually emit over the LIVE record must name a real module (no fabrication)
    live_items, _meta = generate(use_vikunja=False)
    for it in live_items:
        if it.get("cmd") and not _runner_exists(_runner_module(it["cmd"])):
            problems.append("FAIL-DIR(cmd): item %s emitted a cmd naming a non-existent module: %s"
                            % (it["id"], it["cmd"][:80]))
        if it.get("cmd") and it["lane"] not in ("gpu-queue", "pool-cpu"):
            problems.append("FAIL-DIR(cmd): item %s has a cmd but a non-free lane %r" % (it["id"], it["lane"]))

    # ---- STRUCTURED DEPENDENCIES — derived, never invented on a ready item ----
    wall = _item("walls-ledger", "wall-gap4", "Open wall: gap#4 deep credit", "gap#4 deep-credit surpass",
                 "roadmap:1", "t", "v", "", "gpu-queue")
    blocked_scaffold = _item("ledger-scaffold", "scaffold-x", "Retire host scaffold for gap#4 deep credit",
                             "gap#4 deep-credit host shortcut", "led:1", "t", "v",
                             "the spiking replacement must reach parity", "agent")
    ready_flip = _item("ledger-flip", "flip-y", "Flip faculty vision to on-by-default", "object-anywhere",
                       "led:2", "t", "v", "none (de_risked=YES)", "agent")
    linked = link_dependencies([wall, blocked_scaffold, ready_flip])
    bs = next(i for i in linked if i["id"] == "scaffold-x")
    rf = next(i for i in linked if i["id"] == "flip-y")
    if "wall-gap4" not in bs["dependencies"]:
        problems.append("PASS-DIR(deps): a prose-blocked scaffold sharing gap#4 did NOT get the wall id as a dep")
    if rf["dependencies"]:
        problems.append("FAIL-DIR(deps): a READY (non-blocked) item was given a structured dependency (invented)")

    return problems


def _guard_scanner_nonempty(name, scanner, source_has_items):
    """The failing-direction detector: returns True (== 'caught a broken scanner') when the scanner
    yields nothing while its source demonstrably has items. This is the check a dispatcher/heartbeat
    runs so a silently-empty scanner is a LOUD failure, never a quietly-shorter backlog."""
    try:
        got = scanner()
    except Exception:
        return True  # a crashing scanner is also 'caught'
    return source_has_items and len(got) == 0


def main():
    ap = argparse.ArgumentParser(description="Mechanical work-backlog generator (scanner, not dispatcher).")
    ap.add_argument("--selftest", action="store_true", help="prove pass + failing direction, then exit")
    ap.add_argument("--json", action="store_true", help="emit the full backlog JSON to stdout")
    ap.add_argument("--top", type=int, default=None, help="cap the human table to the top N")
    ap.add_argument("--new-only", action="store_true", help="show only the 'not on board' delta")
    ap.add_argument("-v", "--verbose", action="store_true", help="show inline anchors + merge provenance")
    ap.add_argument("--no-vikunja", action="store_true", help="skip the network (file sources only)")
    ap.add_argument("--out", default=OUT_JSON, help="json output path")
    ap.add_argument("--how", action="store_true", help="print the dispatcher-consumption contract")
    args = ap.parse_args()

    if args.how:
        print(HOW)
        return 0

    if args.selftest:
        probs = selftest()
        if probs:
            print("⛔ backlog.py SELFTEST FAILED:")
            for p in probs:
                print("   - " + p)
            return 1
        print("✔ backlog.py selftest PASSED — pass direction (known backlog surfaced) + failing "
              "direction (an empty scanner over a non-empty source is caught) both demonstrated.")
        return 0

    items, meta = generate(use_vikunja=not args.no_vikunja)

    # always write the machine artifact
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root": ROOT,
        "meta": meta,
        "n_items": len(items),
        "n_new_not_on_board": sum(1 for i in items if not i["on_board"]),
        "items": items,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(_render_table(items, meta, top=args.top, new_only=args.new_only, verbose=args.verbose))
        print("\nwrote %s (%d items, %d new-not-on-board)"
              % (os.path.relpath(args.out, ROOT), len(items), payload["n_new_not_on_board"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
