"""Post-fix verification on `SIM_BACKEND=cupy` for the 2026-08-25 DA-axis cupy-interop fix.

Checks (a)-(c) from the task:
  (a) observe_turn returns acted:True with NO `error:` reason.
  (b) da_level VARIES with engagement -- a rich/novel message yields a higher da_level/mode than an
      empty/low-engagement turn (load-bearing on input).
  (c) the LESION (`BRAIN_DA_DRIVES_LESION=1`) COLLAPSES da_level to its floor regardless of engagement.

Usage:
    SIM_BACKEND=cupy /home/dant123/Projects/sim/.venv/bin/python \
        research/findings/raw/da_axis_cupy_interop/verify_fix_cupy.py
"""
import os
import sys

assert os.environ.get("SIM_BACKEND") == "cupy", "run with SIM_BACKEND=cupy"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import webapp.da_mode_drives_chat as DAD  # noqa: E402


class _FakeChat:
    """Minimal stand-in for the production ChatBrain -- `observe_turn` only needs a place to stash
    `_da_drives_workspace` / `_last_da_drives`, exactly like `get_workspace`'s docstring says."""
    pass


def main():
    results = {}

    # (a) acted:True, no error reason, on a rich/novel first turn.
    os.environ.pop("BRAIN_DA_DRIVES_LESION", None)
    chat_a = _FakeChat()
    info_a = DAD.observe_turn(chat_a, "tell me something surprising and unusual about deep sea creatures")
    print("(a) acted/no-error:", info_a)
    assert info_a["acted"] is True, f"expected acted=True, got {info_a}"
    assert not str(info_a["reason"]).startswith("error:"), f"expected no error reason, got {info_a['reason']!r}"
    results["a_acted_no_error"] = True

    # (b) engagement variation: two FRESH workspaces (avoid EMA cross-turn confound) at the same seed,
    #     one with an empty/low-engagement turn, one with a rich/novel turn.
    os.environ.pop("BRAIN_DA_DRIVES_LESION", None)
    chat_low = _FakeChat()
    info_low = DAD.observe_turn(chat_low, "")
    chat_high = _FakeChat()
    info_high = DAD.observe_turn(chat_high,
        "tell me something surprising and unusual about deep sea bioluminescent creatures and their behavior")
    print("(b) low-engagement turn :", info_low)
    print("(b) high-engagement turn:", info_high)
    assert info_low["acted"] is True and info_high["acted"] is True
    assert info_high["da_level"] > info_low["da_level"], (
        f"expected da_level to rise with engagement: low={info_low['da_level']} high={info_high['da_level']}")
    assert info_low["mode"] in ("rest", "neutral") and info_high["mode"] in ("focus", "arousal"), (
        f"expected a mode swing rest/neutral -> focus/arousal, got low={info_low['mode']} high={info_high['mode']}")
    results["b_engagement_varies"] = {
        "low": {"da_level": info_low["da_level"], "mode": info_low["mode"]},
        "high": {"da_level": info_high["da_level"], "mode": info_high["mode"]},
    }

    # (c) lesion collapses da_level to the floor regardless of engagement -- same rich message, lesion ON.
    os.environ["BRAIN_DA_DRIVES_LESION"] = "1"
    chat_lesion = _FakeChat()
    info_lesion = DAD.observe_turn(chat_lesion,
        "tell me something surprising and unusual about deep sea bioluminescent creatures and their behavior")
    print("(c) lesioned (same rich message):", info_lesion)
    os.environ.pop("BRAIN_DA_DRIVES_LESION", None)
    assert info_lesion["acted"] is True
    assert info_lesion["mode"] == "rest", f"expected lesion to collapse mode to rest, got {info_lesion['mode']}"
    assert info_lesion["da_level"] < info_high["da_level"], (
        f"expected lesion da_level < unlesioned high da_level: "
        f"lesioned={info_lesion['da_level']} unlesioned-high={info_high['da_level']}")
    assert info_lesion["lead"] == "", f"expected the engagement suffix to vanish under lesion, got {info_lesion['lead']!r}"
    results["c_lesion_collapses"] = {"lesioned_da_level": info_lesion["da_level"], "mode": info_lesion["mode"]}

    print("\nALL CHECKS PASSED (a)(b)(c) on SIM_BACKEND=cupy")
    print(results)


if __name__ == "__main__":
    main()
