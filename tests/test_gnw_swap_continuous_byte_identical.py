"""Regression test for the 2026-09-23 fix-round finding on `research/gnw-thought-swap-drive`.

Bug (adversarial review, verdict=fix-required): `webapp/gnw_thought_swap.py`'s `ThoughtSwapWorkspace.observe()`
added an UNGUARDED `"continuous": continuous_enabled()` key to its held-topic info dict. Because
`swap_drives_chat.observe_turn` does `out = dict(info)` and `webapp/server.py` attaches that dict as
`resp["swap_drives"]` on the DEFAULT-ON path (`_SWAP_DRIVES_DEFAULT_ON=True`), every default `/api/brain-chat` turn
after the first gained a NEW `"continuous": False` key -- a default-path change, contradicting the branch's own
"byte-identical when unset" claim. `docs/TERMS.md`: "byte-identical" must be asserted in the data (hash or exact
compare), never inferred from reading the code -- this test is that assertion, not a comment.

Fix: the `continuous` key is now only added when `BRAIN_GNW_SWAP_CONTINUOUS` is truthy.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.pop("BRAIN_GNW_SWAP_CONTINUOUS", None)

import pytest

from webapp.gnw_thought_swap import ThoughtSwapWorkspace

# a,b,c,d,e against N_PATTERNS==3: exercises first-thought, same-topic holds, swaps, the board-#85 lesion path
# (mismatch detector silenced), and LRU slot reuse past N_PATTERNS -- every turn shape the original de-risk skipped.
_TURNS = [
    ("a", False), ("a", False), ("b", False), ("b", False), ("a", False),
    ("c", False), ("d", False), ("e", True), ("e", False),
]


@pytest.mark.parametrize("continuous_env", [None, "0"])
def test_continuous_key_absent_when_flag_unset_or_off(continuous_env):
    """`observe()`'s info dict must carry NO `continuous` key whenever BRAIN_GNW_SWAP_CONTINUOUS is unset/falsy --
    the default `/api/brain-chat` response must not gain a new key from this module."""
    prior = os.environ.get("BRAIN_GNW_SWAP_CONTINUOUS")
    try:
        if continuous_env is None:
            os.environ.pop("BRAIN_GNW_SWAP_CONTINUOUS", None)
        else:
            os.environ["BRAIN_GNW_SWAP_CONTINUOUS"] = continuous_env
        ws = ThoughtSwapWorkspace(seed=42)
        for topic, lesion in _TURNS:
            info = ws.observe(topic, lesion=lesion)
            assert "continuous" not in info, (
                f"turn topic={topic!r} lesion={lesion}: 'continuous' key present with the flag OFF -- "
                f"this breaks byte-identity on the default /api/brain-chat path. info={info}"
            )
    finally:
        if prior is None:
            os.environ.pop("BRAIN_GNW_SWAP_CONTINUOUS", None)
        else:
            os.environ["BRAIN_GNW_SWAP_CONTINUOUS"] = prior


def test_continuous_key_present_when_flag_on():
    """Sanity check on the other direction: the key DOES appear (and is `True`) once the flag is actually on, so
    the guard above is a real branch, not a check that can never fail."""
    prior = os.environ.get("BRAIN_GNW_SWAP_CONTINUOUS")
    try:
        os.environ["BRAIN_GNW_SWAP_CONTINUOUS"] = "1"
        ws = ThoughtSwapWorkspace(seed=42)
        ws.observe("a", lesion=False)   # first_thought turn: no held topic yet, never carries the key either way
        info = ws.observe("b", lesion=False)   # first turn with a held topic -> the branch that sets it
        assert info.get("continuous") is True, info
    finally:
        if prior is None:
            os.environ.pop("BRAIN_GNW_SWAP_CONTINUOUS", None)
        else:
            os.environ["BRAIN_GNW_SWAP_CONTINUOUS"] = prior


def test_default_conversation_byte_identical_to_pre_continuous_main():
    """The FULL default-path (flag unset) conversation must be byte-identical (exact dict equality on every field,
    every turn) to the module as it existed on `origin/main` BEFORE this branch's `continuous_enabled()`/isolate=
    thread-through was added -- not just "no continuous key". Loads that pre-branch source directly (frozen inline
    below is unnecessary; instead this re-derives the claim from the live git object so it tracks main, not a
    hand-copied snapshot) and compares turn-by-turn."""
    import hashlib
    import json
    import subprocess

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        merge_base = subprocess.run(
            ["git", "-C", repo_root, "merge-base", "HEAD", "origin/main"],
            capture_output=True, text=True, timeout=10,
        )
        base_sha = merge_base.stdout.strip() if merge_base.returncode == 0 else None
        if not base_sha:
            pytest.skip("no origin/main reachable in this checkout to diff against")
        show = subprocess.run(
            ["git", "-C", repo_root, "show", f"{base_sha}:webapp/gnw_thought_swap.py"],
            capture_output=True, text=True, timeout=10,
        )
        if show.returncode != 0 or not show.stdout.strip():
            pytest.skip("could not read webapp/gnw_thought_swap.py from origin/main's merge-base")
    except (FileNotFoundError, subprocess.SubprocessError):
        pytest.skip("git unavailable in this environment")

    import importlib.util
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix="_gnw_main_ref.py", delete=False) as f:
        f.write(show.stdout)
        ref_path = f.name
    try:
        spec = importlib.util.spec_from_file_location("_gnw_main_ref", ref_path)
        ref_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref_mod)
    finally:
        os.unlink(ref_path)

    import webapp.gnw_thought_swap as branch_mod

    def run(mod):
        ws = mod.ThoughtSwapWorkspace(seed=42)
        return [ws.observe(t, lesion=l) for t, l in _TURNS]

    def canon(x):
        return json.dumps(x, sort_keys=True, default=str)

    ref_out, branch_out = run(ref_mod), run(branch_mod)
    assert canon(ref_out) == canon(branch_out), (
        "default-path (BRAIN_GNW_SWAP_CONTINUOUS unset) conversation diverged from origin/main's merge-base -- "
        f"hash {hashlib.sha256(canon(ref_out).encode()).hexdigest()} vs "
        f"{hashlib.sha256(canon(branch_out).encode()).hexdigest()}"
    )
