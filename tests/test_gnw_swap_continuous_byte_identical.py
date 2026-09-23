"""Regression test for the 2026-09-23 fix-round finding on `research/gnw-thought-swap-drive`.

Bug (adversarial review, verdict=fix-required): `webapp/gnw_thought_swap.py`'s `ThoughtSwapWorkspace.observe()`
added an UNGUARDED `"continuous": continuous_enabled()` key to its held-topic info dict. Because
`swap_drives_chat.observe_turn` does `out = dict(info)` and `webapp/server.py` attaches that dict as
`resp["swap_drives"]` on the DEFAULT-ON path (`_SWAP_DRIVES_DEFAULT_ON=True`), every default `/api/brain-chat` turn
after the first gained a NEW `"continuous": False` key -- a default-path change, contradicting the branch's own
"byte-identical when unset" claim. `docs/TERMS.md`: "byte-identical" must be asserted in the data (hash or exact
compare), never inferred from reading the code -- this test is that assertion, not a comment.

Fix: the `continuous` key is now only added when `BRAIN_GNW_SWAP_CONTINUOUS` is truthy.

2ND FIX ROUND (2026-09-23, adversarial re-review, verdict=fix-required): `test_default_conversation_byte_identical_
to_pre_continuous_main` originally diffed against `git merge-base HEAD origin/main`, a MOVING reference that
becomes a no-op self-diff once this branch merges into main (tautological, cannot fail post-merge). Fixed to diff
against a FIXED pre-branch SHA instead -- see that test's own docstring.
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
    every turn) to the module as it existed BEFORE this branch's `continuous_enabled()`/isolate= thread-through was
    added -- not just "no continuous key". Loads that pre-branch source directly from a git object and compares
    turn-by-turn.

    2ND FIX ROUND (2026-09-23, adversarial re-review, verdict=fix-required): the PRIOR version of this test diffed
    against `git merge-base HEAD origin/main`. That is a MOVING reference -- once this branch is merged into main,
    `origin/main`'s merge-base with a checkout that already contains the merge commit IS that commit itself, so the
    diff becomes `webapp/gnw_thought_swap.py` against ITSELF and can never fail (tautological). FIXED: pinned to
    `_PRE_BRANCH_SHA` below, a FIXED (non-moving) commit -- the last commit to touch this file before this arc
    started -- verified an ancestor of both `HEAD` and `origin/main` at the time of this fix (`git merge-base
    --is-ancestor <sha> origin/main` / `HEAD`, both exit 0). A pinned SHA cannot become tautological on merge,
    unlike a merge-base against a branch this very test's own commit will land on."""
    import hashlib
    import json
    import subprocess

    # Fixed, non-moving reference: the commit immediately before `research/gnw-thought-swap-drive`'s first commit
    # touched this file (`git log --oneline -- webapp/gnw_thought_swap.py` on this branch). Do NOT replace with a
    # branch name / merge-base -- see the retraction above for why that becomes tautological after merge.
    _PRE_BRANCH_SHA = "5b718e73c"

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        verify = subprocess.run(
            ["git", "-C", repo_root, "cat-file", "-e", f"{_PRE_BRANCH_SHA}^{{commit}}"],
            capture_output=True, text=True, timeout=10,
        )
        if verify.returncode != 0:
            pytest.skip(f"pinned reference commit {_PRE_BRANCH_SHA} not reachable in this checkout (shallow clone?)")
        show = subprocess.run(
            ["git", "-C", repo_root, "show", f"{_PRE_BRANCH_SHA}:webapp/gnw_thought_swap.py"],
            capture_output=True, text=True, timeout=10,
        )
        if show.returncode != 0 or not show.stdout.strip():
            pytest.skip(f"could not read webapp/gnw_thought_swap.py from the pinned {_PRE_BRANCH_SHA}")
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
        f"default-path (BRAIN_GNW_SWAP_CONTINUOUS unset) conversation diverged from the pinned pre-branch reference "
        f"{_PRE_BRANCH_SHA} -- hash {hashlib.sha256(canon(ref_out).encode()).hexdigest()} vs "
        f"{hashlib.sha256(canon(branch_out).encode()).hexdigest()}"
    )
