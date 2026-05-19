"""Task 2 tests for the net-new regime-correct composition runner.

These pin the runner's CONTRACT, not toy accuracy numbers:
 (a) --tiny-synth smoke runs end-to-end, returns a dict with a `rungs`
     list and a `verdict` whose `gate` is one of the four legal states,
     and never raises;
 (b) every rung carries exactly the 7 required keys with correct
     types/ranges so the FROZEN verdict module does NOT VOID for a
     structural reason (it may legitimately FAIL on toy numbers --
     that is fine and asserted-around);
 (c) no shipped module on the runner's import graph pulls in
     torch.autograd / .backward (grep the source text);
 (d) the full pass and BOTH ablations for a given (seed, N) consume
     the SAME seed (one seed threaded into all three).

tiny_synth shrinks pools/episodes hard so this stays fast.
"""
from __future__ import annotations

import importlib
import math
import re
from pathlib import Path

import pytest

from research.runners.compose_retrieval_core import (
    compose_retrieval_verdict,
    REQUIRED_KEYS,
    _CR_LADDER,
)

_RUN_MOD = "research.runners.compose_retrieval_runner"
_LEGAL_GATES = {
    "VOID",
    "FAIL",
    "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE",
    "PASS",
}


@pytest.fixture(scope="module")
def tiny_result():
    """One end-to-end --tiny-synth run, shared across the assertions."""
    mod = importlib.import_module(_RUN_MOD)
    assert hasattr(mod, "run_compose_retrieval")
    # Single seed, default frozen ladder, tiny synth: must not raise.
    return mod.run_compose_retrieval(seeds=[42], tiny_synth=True)


def test_a_end_to_end_smoke_returns_well_formed(tiny_result):
    r = tiny_result
    assert isinstance(r, dict)
    assert isinstance(r.get("rungs"), list) and len(r["rungs"]) >= 1
    assert isinstance(r.get("verdict"), dict)
    assert r["verdict"].get("gate") in _LEGAL_GATES


def test_b_rungs_are_structurally_valid_for_frozen_verdict(tiny_result):
    rungs = tiny_result["rungs"]
    for rung in rungs:
        assert isinstance(rung, dict)
        # Exactly the required keys present with right types/ranges.
        for k in REQUIRED_KEYS:
            assert k in rung, f"missing required rung key {k!r}"
        assert isinstance(rung["N"], int) and not isinstance(rung["N"], bool)
        assert rung["N"] in _CR_LADDER
        assert isinstance(rung["n_seeds"], int) and not isinstance(
            rung["n_seeds"], bool
        )
        assert rung["n_seeds"] >= 1
        for k in (
            "full_acc",
            "recent_only_acc",
            "remote_only_acc",
            "abstain_correct_recent_only",
            "abstain_correct_remote_only",
        ):
            v = rung[k]
            assert isinstance(v, (int, float)) and not isinstance(v, bool)
            assert math.isfinite(v)
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"

    # The single-seed smoke uses n_seeds=1 (< MIN_SEEDS) so the FROZEN
    # verdict legitimately returns VOID for under-power -- that is NOT a
    # malformed-structure VOID. Prove the structure itself is accepted by
    # synthesizing the same numbers at >= MIN_SEEDS and confirming the
    # verdict no longer VOIDs for a structural reason (it may FAIL on toy
    # numbers; FAIL != VOID, which is exactly the point).
    bumped = []
    for rung in rungs:
        rr = dict(rung)
        rr["n_seeds"] = 3
        bumped.append(rr)
    v = compose_retrieval_verdict(bumped)
    assert v["gate"] in _LEGAL_GATES
    assert v["gate"] != "VOID", (
        "structurally-valid rungs must not VOID; got VOID with reason: "
        + str(v.get("reason"))
    )

    # And the runner's own embedded verdict never raised and is legal.
    assert compose_retrieval_verdict(rungs)["gate"] in _LEGAL_GATES


def test_c_no_autograd_on_shipped_paths():
    """The runner + the verdict core it imports must not pull autograd."""
    for mod_name in (_RUN_MOD, "research.runners.compose_retrieval_core"):
        mod = importlib.import_module(mod_name)
        src = Path(mod.__file__).read_text(encoding="utf-8", errors="ignore")
        assert "torch.autograd" not in src
        assert ".backward(" not in src
        assert not re.search(r"\bimport\s+torch\b", src)
        # autograd USAGE (import or attribute access), not the English
        # word in a comment that documents its ABSENCE.
        assert not re.search(r"\bimport\s+autograd\b", src)
        assert not re.search(r"\bfrom\s+\S*autograd\b", src)
        assert ".autograd" not in src


def test_d_full_and_both_ablations_consume_the_same_seed():
    """The composition controller must thread ONE seed per (seed, N)
    into the full run AND both ablations (recent_only, remote_only),
    so each ablation is 'full minus exactly one regime, same draws'."""
    mod = importlib.import_module(_RUN_MOD)
    seen = {"full": [], "recent_only": [], "remote_only": []}

    real_fn = mod._cell_passes

    def spy_fn(seed, N, tiny_synth, **kw):
        out = real_fn(seed, N, tiny_synth, **kw)
        # Every cell records which seed each of the three passes used.
        for arm in ("full", "recent_only", "remote_only"):
            assert arm in out and "seed" in out[arm]
            seen[arm].append((N, out[arm]["seed"]))
        return out

    mod._cell_passes = spy_fn
    try:
        mod.run_compose_retrieval(seeds=[42], tiny_synth=True)
    finally:
        mod._cell_passes = real_fn

    assert seen["full"], "no cells evaluated"
    # For every (N), the seed used by full == recent_only == remote_only.
    for arm in ("recent_only", "remote_only"):
        assert seen[arm] == seen["full"], (
            f"{arm} did not consume the same per-cell seed as full: "
            f"{seen[arm]} vs {seen['full']}"
        )
    # And the seed actually equals the requested 42 for the N=2 cell.
    n2 = [s for (N, s) in seen["full"] if N == 2]
    assert n2 and all(s == 42 for s in n2)
