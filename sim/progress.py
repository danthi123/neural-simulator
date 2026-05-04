"""Universal progress event format for research runners.

All runners SHOULD emit progress via `emit_progress(...)` instead of
ad-hoc print statements. The webapp's inflight panel parses these
structured events with a single regex, so adding a new runner doesn't
require touching webapp/server.py, app.js, or brain3d.js.

Event shape (JSON-encoded):
    {
        "kind": "training|replay|eval|step|complete|phase",
        "current": int,            # progress counter (units depend on kind)
        "total": int,              # max counter
        "phase": "P1|P2|P3|...",   # optional, label for the current phase
        "unit": "episodes|events|trials|steps",  # display unit
        "label": "freeform",       # optional, human-friendly identifier
        ...                        # arbitrary extra fields, e.g.
                                    # correct_pct, accuracy, pos, goal
    }

The runner emits one event per progress milestone:
    [PROGRESS] {"kind":"training","phase":"P2","current":50,...}

Wire format:
    Each event is one line on stdout, prefixed with "[PROGRESS] ".
    The JSON payload follows. flush=True so the webapp tail-reader
    sees it without buffering delay.

Rationale:
    Before: each runner had its own progress log format (e.g. "[ep N/M]",
    "[P3 SWR] N/M", "[isolation] N/M events"). The webapp's parser had
    one regex per format. Adding a runner required updating server.py +
    app.js + brain3d.js (3 files, all needing the same regex updated in
    different ways). After: ONE regex parses all formats; ONE frontend
    render function dispatches on `kind` with a generic fallback.

Migration:
    For backward compatibility during migration, runners that emit the
    structured format MAY also continue emitting the old human-readable
    format (e.g. add a separate `print(f"  [P2 ep {n}/{m}]...")` call
    for log readability). The webapp prefers the new format if present.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, Optional


# Webapp-side regex. Captures everything after "[PROGRESS] " for json.loads.
PROGRESS_LINE_RE = re.compile(r"\[PROGRESS\]\s+(\{.*\})")


def emit_progress(
    kind: str,
    current: Optional[int] = None,
    total: Optional[int] = None,
    *,
    phase: Optional[str] = None,
    unit: Optional[str] = None,
    label: Optional[str] = None,
    file=None,
    **extras: Any,
) -> None:
    """Emit one structured progress event.

    Args:
        kind: event category. One of:
            - "training": episode-based training (e.g. curriculum Phase 2).
            - "replay":   paired-stim/SWR replay (event-based training).
            - "eval":     evaluation trial (W->A, I->W, etc.).
            - "step":     per-step gridworld navigation.
            - "phase":    phase boundary marker ("PHASE 2 starting").
                          For phase markers, current/total can be None.
            - "complete": final completion + result summary.
        current: progress counter. None for phase markers / completion.
        total: max counter. None for phase markers / completion.
        phase: optional phase tag (e.g. "P2", "P3 SWR", "W->A eval").
        unit: optional display unit (e.g. "episodes", "events", "trials").
        label: optional freeform identifier for runs/conditions.
        file: optional file-like to write to (default sys.stdout).
        **extras: arbitrary additional fields (correct_pct, accuracy,
            pos, goal, n_steps, etc.). Will be JSON-serialized.

    Output: one line of `[PROGRESS] {json}` to stdout, flushed.
    """
    payload: Dict[str, Any] = {"kind": kind}
    if current is not None:
        payload["current"] = int(current)
    if total is not None:
        payload["total"] = int(total)
    if phase is not None:
        payload["phase"] = str(phase)
    if unit is not None:
        payload["unit"] = str(unit)
    if label is not None:
        payload["label"] = str(label)
    payload.update(extras)

    out = file if file is not None else sys.stdout
    print(f"[PROGRESS] {json.dumps(payload)}", file=out, flush=True)


def parse_progress_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single log line. Returns the structured progress dict if
    the line matches the [PROGRESS] format, else None.

    Used by webapp/server.py inflight scanner. Backward-compat parsers
    for old formats run AFTER this one — the new format takes priority.
    """
    m = PROGRESS_LINE_RE.search(line)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def parse_last_progress(text: str) -> Optional[Dict[str, Any]]:
    """Parse a multi-line log buffer, return the LAST progress event
    of any kind. The webapp uses this on the tail of a log file.
    """
    last = None
    for m in PROGRESS_LINE_RE.finditer(text):
        try:
            last = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
    return last


def parse_last_progress_by_kind(text: str) -> Dict[str, Dict[str, Any]]:
    """Parse a log buffer; return a dict mapping kind -> last event of
    that kind. Lets the webapp show "currently in eval phase, training
    ended at X/Y" without losing the prior phase's progress."""
    by_kind: Dict[str, Dict[str, Any]] = {}
    for m in PROGRESS_LINE_RE.finditer(text):
        try:
            evt = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        k = evt.get("kind")
        if k:
            by_kind[k] = evt
    return by_kind
