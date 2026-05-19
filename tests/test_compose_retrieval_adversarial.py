"""ADVERSARIAL faithfulness pins for the regime-correct compositional
retrieval runner.

History (kept visible, not erased):
  * Task-3 dedicated adversarial review (2026-05-19) found FOUR
    confirmed defects: D1 dead ablation-accuracy bars, D2 single-path
    only blocked via an emergent abstention denominator, D2-KNOWN a
    CONFIRMED semantically-empty additive false-PASS, D3 the moat fed
    an out-of-calibration quantity.
  * The faithfulness-fix iteration (same date) CLOSED all four. These
    pins were INVERTED -- they no longer assert the defects exist; they
    assert each defect is CLOSED and stays closed (a regression trips a
    red test and re-enters review). The original defect intent is
    preserved in each docstring so the boundary remains legible.

These tests do NOT weaken any frozen bar and touch NO protected file.
"""
from __future__ import annotations

import importlib
import inspect
import sys

import pytest

_RUN = "research.runners.compose_retrieval_runner"


@pytest.fixture(scope="module")
def R():
    # --tiny-synth must be on argv before import (module-top backend
    # policy reads sys.argv at import time).
    if "--tiny-synth" not in sys.argv:
        sys.argv.append("--tiny-synth")
    return importlib.import_module(_RUN)


# ---------------------------------------------------------------------
# D1 (CLOSED): ablation-accuracy bars were structurally dead. The only
# `n_correct += 1` sat behind `if groundable:` with `groundable==False`
# on both ablation arms, so recent_only_acc / remote_only_acc were a
# structural constant 0.0 and the frozen _CR_ABLATION_MAX collapse bars
# were vacuously satisfied. FIX B made every arm run the IDENTICAL
# query->retrieve->compose->decode->score pipeline so the ablation
# accuracies are GENUINELY MEASURED. These pins now assert that.
# ---------------------------------------------------------------------
def test_D1_dead_groundable_shortcircuit_is_GONE(R):
    """The `groundable = have_remote and (not hippo_off)` per-arm gate
    that dominated the only accuracy increment is GONE -- the ablation
    accuracy is no longer a structural dead constant."""
    src = inspect.getsource(R._score_arm)
    assert "groundable = have_remote and (not hippo_off)" not in src, (
        "D1 regressed: the dead-bar `groundable` gate is back; ablation "
        "accuracy would again be a structural constant 0.0"
    )
    # An accuracy increment still exists, but it is reachable on every
    # arm (full / recent_only / remote_only) -- no per-arm short-circuit.
    assert src.count("n_correct += 1") >= 1


def test_D1_ablation_accuracy_is_now_a_live_measurement(R):
    """End-to-end (tiny-synth): an omniscient stub solver that answers
    correctly on EVERY arm drives BOTH ablation accuracies OFF 0.0 --
    empirical proof the bars are now genuine measurements (the exact
    inverse of the retired dead-constant pin)."""
    saved = (R._build_substrate, R._encode_recent_facts,
             R._build_remote_schema, R._hippo_silenced, R._compose_query)
    try:
        R._build_substrate = lambda s, t: (
            object(),
            {"n_lang_input": 64, "n_per_pool": 12,
             "n_fs_per_pool": 3, "sparsity": 0.05},
        )
        R._encode_recent_facts = lambda b, f, d, e: [
            f"fact_{i}" for i in range(len(f))
        ]
        R._build_remote_schema = lambda *a, **k: None
        R._hippo_silenced = lambda b, s=-2000.0: ((lambda: None), 0)
        facts_for = R._recent_facts

        def omniscient(bridge, cue, tag, dims, have_remote, recall_steps):
            for (noun, adj) in facts_for(8):
                if noun == cue:
                    return adj, [(adj, 9999.0, "c")], {}
            return None, [("x", 1.0, "c")], {}

        R._compose_query = omniscient
        res = R.run_compose_retrieval(seeds=[42, 43, 44],
                                      loads=(2, 4, 8), tiny_synth=True)
        for rung in res["rungs"]:
            assert rung["recent_only_acc"] > 0.0
            assert rung["remote_only_acc"] > 0.0
    finally:
        (R._build_substrate, R._encode_recent_facts,
         R._build_remote_schema, R._hippo_silenced,
         R._compose_query) = saved


# ---------------------------------------------------------------------
# D2 (CLOSED): single-path artifacts. Previously only blocked via the
# emergent abstention denominator (the dead accuracy bars were
# vacuously cleared). With FIX B the ablation accuracy is measured, so a
# single-path solver is now caught by EITHER a high measured ablation
# accuracy OR a collapsed abstain_correct. Pin both directions stay
# NOT-PASS.
# ---------------------------------------------------------------------
def _stub_network(R):
    R._build_substrate = lambda s, t: (
        object(),
        {"n_lang_input": 64, "n_per_pool": 12,
         "n_fs_per_pool": 3, "sparsity": 0.05},
    )
    R._encode_recent_facts = lambda b, f, d, e: [
        f"fact_{i}" for i in range(len(f))
    ]
    R._build_remote_schema = lambda *a, **k: None
    R._hippo_silenced = lambda b, s=-2000.0: ((lambda: None), 0)


def test_D2_pure_hippocampal_single_path_is_NOT_PASS(R):
    """A perfect hippocampal-only solver (composition a no-op) must NOT
    pass: it answers in recent_only (hippo still on), so recent_only_acc
    is high (> _CR_ABLATION_MAX) AND abstain_correct_recent_only
    collapses. Either trips the frozen verdict -> not PASS."""
    saved = (R._build_substrate, R._encode_recent_facts,
             R._build_remote_schema, R._hippo_silenced, R._compose_query)
    try:
        _stub_network(R)
        facts_for = R._recent_facts

        def hip_only(bridge, cue, tag, dims, have_remote, recall_steps):
            if tag is not None:                       # hippo path solves
                for (noun, adj) in facts_for(8):
                    if noun == cue:
                        return adj, [(adj, 9999.0, "c")], {}
            return None, [("x", 1.0, "c")], {}        # no tag -> abstain

        R._compose_query = hip_only
        res = R.run_compose_retrieval(seeds=[42, 43, 44],
                                      loads=(2, 4, 8), tiny_synth=True)
        assert res["verdict"]["gate"] != "PASS"
        r0 = res["rungs"][0]
        # The recent-only ablation is now a LIVE measurement: a hippo-
        # only solver answers there, so its measured accuracy is high
        # (the previously-dead bar now genuinely catches it) OR the
        # abstention denominator catches it. At least one must fire.
        caught = (
            r0["recent_only_acc"] > 0.40
            or r0["abstain_correct_recent_only"] < 0.90
        )
        assert caught, (
            "single-path hippo solver escaped BOTH the (now live) "
            "ablation-accuracy bar and the abstention bar"
        )
    finally:
        (R._build_substrate, R._encode_recent_facts,
         R._build_remote_schema, R._hippo_silenced,
         R._compose_query) = saved


def test_D2_pure_consolidated_single_path_is_NOT_PASS(R):
    """A perfect consolidated-only solver (the 2026-05-14 RETRACTED
    failure mode) must NOT pass: it answers in remote_only (consolidated
    still on), so remote_only_acc is high AND
    abstain_correct_remote_only collapses."""
    saved = (R._build_substrate, R._encode_recent_facts,
             R._build_remote_schema, R._hippo_silenced, R._compose_query)
    try:
        _stub_network(R)

        def cons_only(bridge, cue, tag, dims, have_remote, recall_steps):
            if have_remote:
                for (n, a) in R._recent_facts(8):
                    if n == cue:
                        return a, [(a, 9999.0, "c")], {}
            return None, [("x", 1.0, "c")], {}

        R._compose_query = cons_only
        res = R.run_compose_retrieval(seeds=[42, 43, 44],
                                      loads=(2, 4, 8), tiny_synth=True)
        assert res["verdict"]["gate"] != "PASS"
        r0 = res["rungs"][0]
        caught = (
            r0["remote_only_acc"] > 0.40
            or r0["abstain_correct_remote_only"] < 0.90
        )
        assert caught, (
            "single-path consolidated solver escaped BOTH the (now "
            "live) ablation-accuracy bar and the abstention bar"
        )
    finally:
        (R._build_substrate, R._encode_recent_facts,
         R._build_remote_schema, R._hippo_silenced,
         R._compose_query) = saved


def test_D2_semantically_empty_additive_solver_is_NOW_NOT_PASS(R):
    """PREVIOUSLY a CONFIRMED UNFIXED FALSE-PASS; NOW CLOSED.

    The original defect: a `_compose_query` doing ZERO neural
    computation and ZERO composition -- it read the answer out of the
    engram TAG STRING the runner constructed and emitted two hardcoded
    constant 400.0 addends -- scored a clean PASS across the frozen
    ladder, because each path alone (400) was sub-moat (vacuous
    abstain=1.0) while the additive sum (800) cleared the moat
    (full_acc=1.0) and the advertised collapse bars were dead.

    Two fixes close it together:
      (a) FIX C makes engram tags OPAQUE (fact_{i}) -- the tag string
          carries no answer, so a string-reading solver cannot recover
          the adjective at all;
      (b) FIX B makes the ablation accuracies a LIVE measurement and the
          decoded answer come from the validated neural readout, so a
          hardcoded-constant additive solver that bypasses the readout
          cannot produce a clean PASS.

    This pin is INVERTED: it asserts the gate is NOT PASS. If it ever
    PASSes again the defect regressed -- re-enter adversarial review."""
    saved = (R._build_substrate, R._encode_recent_facts,
             R._build_remote_schema, R._hippo_silenced, R._compose_query)
    try:
        _stub_network(R)

        from research.runners.abstention_gate import gate as _g
        from research.runners.abstention_gate import (
            DEFAULT_THRESHOLD as _M,
        )

        def additive(bridge, cue, tag, dims, have_remote, recall_steps):
            # Try to read adj from the tag string (the original cheat).
            # Opaque fact_{i} tags yield nothing -> no answer.
            adj = None
            if tag is not None:
                for sep in ("__", "_"):
                    parts = tag.split(sep)
                    if len(parts) >= 3 and parts[-1].isalpha() \
                            and parts[-1] != "fact":
                        adj = parts[-1]
            if adj is None:
                return None, [], {}
            cons = [(adj, 400.0, "c")] if have_remote else []
            hip = [(adj, 400.0, "c")] if tag is not None else []
            scores = {}
            for w, r, _ in cons:
                scores[w] = scores.get(w, 0.0) + r
            for w, r, _ in hip:
                scores[w] = scores.get(w, 0.0) + r
            ranked = sorted(((w, scores[w], "compose") for w in scores),
                            key=lambda t: -t[1])
            decided = _g(ranked, _M)
            return (None if decided is None else decided[0]), ranked, {}

        R._compose_query = additive
        res = R.run_compose_retrieval(seeds=[42, 43, 44],
                                      loads=(2, 4, 8), tiny_synth=True)
        assert res["verdict"]["gate"] != "PASS", (
            "REGRESSION: the semantically-empty additive false-PASS is "
            "back -- engram tags must be opaque AND the answer must come "
            "from the validated neural readout (re-enter review)"
        )
    finally:
        (R._build_substrate, R._encode_recent_facts,
         R._build_remote_schema, R._hippo_silenced,
         R._compose_query) = saved


# ---------------------------------------------------------------------
# D3 (CLOSED): the moat was fed an out-of-calibration quantity
# (`max(0,cos) * ||pattern||_2`) while abstention_gate's 650 threshold
# was calibrated on RAW lang_output firing rates (encoded mean ~796,
# control max ~584). FIX D feeds the moat the validated raw firing-rate
# confidence. Pin the energy-scaled-cosine hack is GONE and the moat is
# still the byte-unchanged 7/7 gate at 650.
# ---------------------------------------------------------------------
def test_D3_energy_scaled_cosine_hack_is_GONE(R):
    src = inspect.getsource(R._ranked_from_pattern)
    assert "* energy" not in src and (
        "np.linalg.norm(np.asarray(pattern))" not in src
    ), (
        "D3 regressed: the energy-scaled-cosine moat input is back; the "
        "650 threshold was calibrated on RAW firing rates"
    )
    # The validated calibrated quantity is the raw firing-rate readout.
    assert "firing" in src.lower() or "rate" in src.lower()
    # And the moat is genuinely the byte-unchanged 7/7 gate object.
    from research.runners import abstention_gate as ag
    assert R._abstain_gate is ag.gate
    assert R._MOAT == ag.DEFAULT_THRESHOLD == 650.0
