"""Failure class 13 — LIVE-STATE re-injection broken / over-cap (compaction-loss + attention-dilution).

WHY (owner, 2026-08-21, spec docs/plans/2026-08-21-enforcement-layer-self-maintaining-project-os.md §5).
Two root failure modes, both mechanical: (1) COMPACTION drops load-bearing info, (2) a FULL CONTEXT
DILUTES attention so in-context info stops driving behaviour. The fix is a capped LIVE-STATE file
(research/coordination/live_state.md) that a SessionStart + UserPromptSubmit hook re-injects at
turn-start and immediately post-compaction (re-read from the FILE, not the lossy summary). This gate
is the enforcement half — a re-injection layer that is only "remembered" is exactly the pattern the
whole enforcement engine exists to remove.

WHAT IT BLOCKS (all deterministic, root-derived, so absence of a problem is not evidence of health
only for the wiring pieces, which are opt-in files):

  1. DILUTION GUARD — the load-bearing check. If research/coordination/live_state.md is larger than
     tools/live_state.CAP_BYTES, it BLOCKS. An over-cap live-state is not "more context", it is more
     of the dilution the file exists to counter — it defeats its own purpose. The generator caps
     itself; this catches a hand-edit or a regression that lets it grow.

  2. MECHANISM-PRESENT — if the live-state FILE exists but .claude/hooks/live_state_inject.py is
     gone, the file is written but nothing re-injects it: silent death of the mechanism.

  3. WIRING — if .claude/settings.json parses, the SessionStart hook (post-compaction survival) AND
     the UserPromptSubmit hook (turn-start anti-dilution) must both invoke live_state_inject. Removing
     either silently reverts to "remembered, not enforced". Settings absent/unparseable ⇒ skipped
     (not this gate's file to police, and a partial checkout must not block).

WHAT IT CANNOT CATCH: whether the injected content is TRUE or current (the generator reads the
durable board; a stale board yields a stale-but-honest file — the summary_doc_freshness / stale_pointer
gates cover board staleness); whether Claude Code actually delivered the hook's stdout to the model
(outside the repo); the completed-task dump (not suppressible via any hook this build exposes — see the
noise-cut report). The FUNCTIONAL selftest below runs the real hook end-to-end, including the
source="compact" post-compaction path, so a broken hook is caught here rather than at the next
compaction.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile

NAME = "live-state-reinjection"
CLASS_ID = "13"
BLOCKING = True

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tools.live_state import CAP_BYTES, LIVE_STATE_REL   # noqa: E402  (one source for the cap)

INJECT_HOOK_REL = ".claude/hooks/live_state_inject.py"
SETTINGS_REL = ".claude/settings.json"
HOOK_TOKEN = "live_state_inject"


def _settings_commands(root):
    """{event_name: [command strings]} from settings.json, or None if absent/unparseable."""
    p = os.path.join(root, SETTINGS_REL)
    if not os.path.isfile(p):
        return None
    try:
        cfg = json.loads(io.open(p, encoding="utf-8", errors="ignore").read())
    except Exception:
        return None
    hooks = (cfg or {}).get("hooks") or {}
    out = {}
    for event, groups in hooks.items():
        cmds = []
        for g in groups or []:
            for h in (g or {}).get("hooks") or []:
                c = h.get("command")
                if c:
                    cmds.append(c)
        out[event] = cmds
    return out


def check(paths=None, root=ROOT):
    problems = []

    # 1. DILUTION GUARD ------------------------------------------------------------------------------
    lsf = os.path.join(root, LIVE_STATE_REL)
    file_exists = os.path.isfile(lsf)
    if file_exists:
        nb = os.path.getsize(lsf)
        if nb > CAP_BYTES:
            problems.append(
                "%s is %d bytes, over the %d-byte cap — an over-cap live-state DILUTES attention, "
                "the exact failure it exists to prevent. Regenerate: python tools/live_state.py"
                % (LIVE_STATE_REL, nb, CAP_BYTES))

    # 2. MECHANISM-PRESENT --------------------------------------------------------------------------
    if file_exists and not os.path.isfile(os.path.join(root, INJECT_HOOK_REL)):
        problems.append(
            "%s exists but %s is missing — the file is written but NOTHING re-injects it "
            "(silent death of the re-injection mechanism)." % (LIVE_STATE_REL, INJECT_HOOK_REL))

    # 3. WIRING -------------------------------------------------------------------------------------
    cmds = _settings_commands(root)
    if cmds is not None and os.path.isfile(os.path.join(root, INJECT_HOOK_REL)):
        for event, why in (("SessionStart", "post-compaction survival (source=compact)"),
                           ("UserPromptSubmit", "turn-start anti-dilution")):
            if not any(HOOK_TOKEN in c for c in cmds.get(event, [])):
                problems.append(
                    "%s does not wire a %s hook to %s (%s). Without it the layer is 'remembered, "
                    "not enforced'." % (SETTINGS_REL, event, HOOK_TOKEN, why))
    return problems


# --------------------------------------------------------------------------------------------------
# selftest — FAILING DIRECTION FIRST, then a real end-to-end hook run incl. the post-compaction path
# --------------------------------------------------------------------------------------------------
def _mk(root, size_bytes=None, with_hook=True, session=True, prompt=True):
    """Build a fixture repo root: a live-state file of a chosen size + optional hook + settings."""
    os.makedirs(os.path.join(root, "research/coordination"), exist_ok=True)
    os.makedirs(os.path.join(root, ".claude/hooks"), exist_ok=True)
    if size_bytes is not None:
        io.open(os.path.join(root, LIVE_STATE_REL), "w", encoding="utf-8").write("x" * size_bytes)
    if with_hook:
        io.open(os.path.join(root, INJECT_HOOK_REL), "w", encoding="utf-8").write("# stub\n")
    events = {}
    if session:
        events["SessionStart"] = [{"hooks": [{"type": "command",
                                              "command": "python3 .../%s.py" % HOOK_TOKEN}]}]
    if prompt:
        events["UserPromptSubmit"] = [{"hooks": [{"type": "command",
                                                  "command": "python3 .../%s.py" % HOOK_TOKEN}]}]
    io.open(os.path.join(root, SETTINGS_REL), "w", encoding="utf-8").write(
        json.dumps({"hooks": events}))
    return root


def selftest():
    bad = []

    # ---- FAILING DIRECTION FIRST: an over-cap file MUST be caught (the dilution guard) ----
    with tempfile.TemporaryDirectory() as d:
        _mk(d, size_bytes=CAP_BYTES + 1)
        if not any("over the" in p for p in check(root=d)):
            bad.append("did NOT catch an over-cap live-state file (dilution guard dead)")

    # a file exactly at the cap is fine (boundary), with everything wired: NO false positive
    with tempfile.TemporaryDirectory() as d:
        _mk(d, size_bytes=CAP_BYTES)
        probs = check(root=d)
        if probs:
            bad.append("FALSE POSITIVE on a valid at-cap, fully-wired fixture: %r" % probs)

    # mechanism gone: file present, inject hook missing MUST be caught
    with tempfile.TemporaryDirectory() as d:
        _mk(d, size_bytes=10, with_hook=False)
        if not any("NOTHING re-injects" in p for p in check(root=d)):
            bad.append("did NOT catch a present file with the inject hook missing")

    # wiring gone: SessionStart missing MUST be caught (post-compaction survival)
    with tempfile.TemporaryDirectory() as d:
        _mk(d, size_bytes=10, session=False)
        if not any("SessionStart" in p for p in check(root=d)):
            bad.append("did NOT catch a missing SessionStart wiring (post-compaction re-injection)")

    # wiring gone: UserPromptSubmit missing MUST be caught (turn-start anti-dilution)
    with tempfile.TemporaryDirectory() as d:
        _mk(d, size_bytes=10, prompt=False)
        if not any("UserPromptSubmit" in p for p in check(root=d)):
            bad.append("did NOT catch a missing UserPromptSubmit wiring (turn-start anti-dilution)")

    # no file at all: nothing to dilute, nothing to enforce ⇒ clean (no false positive)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, ".claude/hooks"))
        io.open(os.path.join(d, INJECT_HOOK_REL), "w").write("# stub\n")
        if check(root=d):
            bad.append("FALSE POSITIVE when no live-state file exists: %r" % check(root=d))

    # ---- FUNCTIONAL: run the REAL hook end-to-end, incl. the source="compact" post-compaction path.
    # A broken hook (emits nothing on compaction) is caught HERE, not at the next real compaction.
    hook = os.path.join(ROOT, INJECT_HOOK_REL)
    if os.path.isfile(hook):
        with tempfile.TemporaryDirectory() as data_root:
            os.makedirs(os.path.join(data_root, "research/coordination"))
            io.open(os.path.join(data_root, "GAP_CLOSURE_MISSION.md"), "w", encoding="utf-8").write(
                "# board\n**CURRENT ARC = fixture frontier**\n")
            env = dict(os.environ, LIVE_STATE_ROOT=data_root)
            for src, label in (("compact", "POST-COMPACTION SessionStart"), ("startup", "startup")):
                try:
                    r = subprocess.run(
                        [sys.executable, hook], input=json.dumps(
                            {"hook_event_name": "SessionStart", "source": src}),
                        capture_output=True, text=True, timeout=30, env=env)
                    if "⟦LIVE-STATE⟧" not in r.stdout or "CONSTRAINTS" not in r.stdout:
                        bad.append("hook emitted no LIVE-STATE on %s: %r" % (label, r.stdout[:120]))
                except Exception as e:
                    bad.append("hook crashed on %s: %s" % (label, e))
            # UserPromptSubmit must also emit (turn-start), reading the file the compact run wrote
            try:
                r = subprocess.run(
                    [sys.executable, hook],
                    input=json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": "hi"}),
                    capture_output=True, text=True, timeout=30, env=env)
                if "⟦LIVE-STATE⟧" not in r.stdout:
                    bad.append("hook emitted no LIVE-STATE on UserPromptSubmit: %r" % r.stdout[:120])
            except Exception as e:
                bad.append("hook crashed on UserPromptSubmit: %s" % e)

    return bad
