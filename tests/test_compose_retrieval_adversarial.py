"""ADVERSARIAL faithfulness pins for the regime-correct compositional
retrieval runner (Task 3 dedicated adversarial review, 2026-05-19).

These tests do NOT weaken any frozen bar and touch NO protected file.
They PIN the exact faithfulness boundary the adversarial review
established, so a future edit that silently changes any of these
properties trips a red test and re-enters review:

  D1 (dead ablation-accuracy bars): in `_score_arm` the ablation arms
      (recent_only / remote_only) have `groundable == False`, so
      `n_correct` can NEVER increment -> `recent_only_acc` and
      `remote_only_acc` are STRUCTURAL CONSTANT 0.0, not measurements.
      Consequence: the frozen verdict's `_CR_ABLATION_MAX` collapse
      bars are vacuously satisfied and prove nothing about collapse.
      The ONLY live ablation gate is the abstention-correctness bar.

  D2 (single-path artifact is blocked, but ONLY via the abstention
      denominator -- an emergent, not advertised, property): a pure
      hippocampal-only solver and a pure consolidated-only solver are
      each scored FAIL, because the still-present path answers in its
      ablation arm and collapses that arm's abstain_correct < 0.90.
      This pins that the protection EXISTS and is load-bearing -- if a
      refactor removes it, a single-path artifact would falsely clear
      the (dead) accuracy bars.

  D3 (moat fed an out-of-calibration quantity): `_ranked_from_pattern`
      feeds the no-confabulation moat `max(0,cos) * ||pattern||_2`,
      while abstention_gate's 650 threshold was calibrated on raw
      lang_output FIRING RATES (encoded mean ~796, control max ~584).
      This pins the scale-bridging hack is present and unvalidated so
      it stays visible to reviewers.

A future change that makes any ablation arm actually MEASURE accuracy
(closing D1) would legitimately break test_D1_* -- that is a desired
improvement and the test docstring says so explicitly; update the pin
deliberately, never silently.
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
# D1: ablation-accuracy bars are structurally dead.
# ---------------------------------------------------------------------
def test_D1_ablation_accuracy_is_a_dead_constant_zero(R):
    """recent_only_acc / remote_only_acc cannot be anything but 0.0
    because the only `n_correct += 1` site sits behind `if groundable:`
    and `groundable` is False on both ablation arms. This is the
    documented faithfulness limitation: _CR_ABLATION_MAX is a dead bar.
    """
    src = inspect.getsource(R._score_arm)
    assert src.count("n_correct += 1") == 1, (
        "more than one accuracy-increment site -- the dead-bar pin is "
        "stale; re-audit whether ablation accuracy is now measured"
    )
    assert "groundable = have_remote and (not hippo_off)" in src
    # The single increment must be dominated by `if groundable:`.
    inc = src.index("n_correct += 1")
    gctx = src.rindex("if groundable:", 0, inc)
    assert gctx != -1, "accuracy increment escaped the groundable gate"


def test_D1_runner_emits_zero_ablation_accuracy_on_real_smoke(R):
    """End-to-end (tiny-synth): both ablation accuracies are exactly
    0.0 regardless of network behaviour -- empirical proof the bars
    are not measurements."""
    res = R.run_compose_retrieval(seeds=[42], loads=(2,), tiny_synth=True)
    for rung in res["rungs"]:
        assert rung["recent_only_acc"] == 0.0
        assert rung["remote_only_acc"] == 0.0


# ---------------------------------------------------------------------
# D2: single-path artifacts are blocked -- but ONLY via the abstention
# denominator. Pin both directions so the protection cannot vanish in
# a refactor without a red test.
# ---------------------------------------------------------------------
def _stub_network(R):
    R._build_substrate = lambda s, t: (
        object(),
        {"n_lang_input": 64, "n_per_pool": 12,
         "n_fs_per_pool": 3, "sparsity": 0.05},
    )
    R._encode_recent_facts = lambda b, f, d, e: [
        f"recent__{n}__{a}" for (n, a) in f
    ]
    R._build_remote_schema = lambda *a, **k: None
    R._hippo_silenced = lambda b, s=-2000.0: ((lambda: None), 0)


def test_D2_pure_hippocampal_single_path_is_FAIL(R):
    """A perfect hippocampal-only solver (composition a no-op) must NOT
    pass: it answers in recent_only (hippo still on), collapsing
    abstain_correct_recent_only -> FAIL. Pins the emergent guard."""
    saved = (R._build_substrate, R._encode_recent_facts,
             R._build_remote_schema, R._hippo_silenced, R._compose_query)
    try:
        _stub_network(R)

        def hip_only(bridge, cue, tag, dims, have_remote, recall_steps):
            if tag is not None:                       # hippo path solves
                ans = tag.split("__")[2]
                return ans, [(ans, 9999.0, "c")], {}
            return None, [("x", 1.0, "c")], {}        # no tag -> abstain

        R._compose_query = hip_only
        res = R.run_compose_retrieval(seeds=[42, 43, 44],
                                      loads=(2, 4, 8), tiny_synth=True)
        assert res["verdict"]["gate"] == "FAIL"
        # Specifically: caught by the abstention bar, not the dead bar.
        r0 = res["rungs"][0]
        assert r0["full_acc"] >= 0.80
        assert r0["recent_only_acc"] == 0.0  # dead bar "cleared"
        assert r0["remote_only_acc"] == 0.0  # dead bar "cleared"
        assert r0["abstain_correct_recent_only"] < 0.90  # the real catch
    finally:
        (R._build_substrate, R._encode_recent_facts,
         R._build_remote_schema, R._hippo_silenced,
         R._compose_query) = saved


def test_D2_pure_consolidated_single_path_is_FAIL(R):
    """A perfect consolidated-only solver (the 2026-05-14 RETRACTED
    failure mode) must NOT pass: it answers in remote_only (consolidated
    still on), collapsing abstain_correct_remote_only -> FAIL."""
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
        assert res["verdict"]["gate"] == "FAIL"
        r0 = res["rungs"][0]
        assert r0["remote_only_acc"] == 0.0  # dead bar "cleared"
        assert r0["abstain_correct_remote_only"] < 0.90  # the real catch
    finally:
        (R._build_substrate, R._encode_recent_facts,
         R._build_remote_schema, R._hippo_silenced,
         R._compose_query) = saved


def test_D2_semantically_empty_additive_solver_scores_PASS_KNOWN_DEFECT(R):
    """CONFIRMED UNFIXED FALSE-PASS (reported, not papered over).

    A `_compose_query` that does ZERO neural computation, ZERO
    composition, and has ZERO semantic content -- it reads the answer
    out of the engram TAG STRING the runner itself constructs and emits
    two hardcoded constant 400.0 addends -- is scored PASS across the
    full frozen ladder (2,4,8) at >= MIN_SEEDS. Each path alone (400)
    is below the moat 650 so the ablation arms vacuously abstain
    (abstain_correct = 1.0); the additive sum (800) clears the moat so
    full_acc = 1.0; the advertised _CR_ABLATION_MAX collapse bars are
    dead (recent/remote_only_acc structurally 0.0). The instrument
    therefore CANNOT distinguish genuine regime composition from
    arithmetic summation of two sub-threshold signals -- the exact
    artifact class behind the project's 2026-05-14 retraction.

    Closing this needs (a) moat recalibration to the runner's actual
    `cos*||pattern||_2` input quantity (the 650 threshold was
    calibrated on raw firing RATES), OR (b) the ablation arms actually
    MEASURING accuracy so removing each path provably degrades full_acc.
    Both are runner-design changes beyond strengthen-only scope, so
    this is reported as a DEFECT, not silently fixed. If this test ever
    FAILS because the gate is no longer PASS, the defect was addressed:
    update this pin deliberately and record how."""
    saved = (R._build_substrate, R._encode_recent_facts,
             R._build_remote_schema, R._hippo_silenced, R._compose_query)
    try:
        _stub_network(R)

        from research.runners.abstention_gate import gate as _g
        from research.runners.abstention_gate import (
            DEFAULT_THRESHOLD as _M,
        )

        def additive(bridge, cue, tag, dims, have_remote, recall_steps):
            # The recent-fact tag NAME carries the (noun, adj) pair, so
            # a controller trivially reads adj from the tag string -- no
            # neural signal whatsoever. Two constant 400 addends:
            #   path alone   = 400  (< moat 650) -> abstains alone
            #   composed sum = 800  (> moat 650) -> answers in full
            if tag is not None:
                adj = tag.split("__")[2]
            else:
                adj = "cold"  # consolidated-alone is sub-moat anyway
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
        assert res["verdict"]["gate"] == "PASS", (
            "the semantically-empty additive false-PASS appears CLOSED "
            "-- if so this is an improvement; update this pin "
            "intentionally and record how the defect was addressed"
        )
        # And it is a CLEAN PASS on every rung (not WORKS-AT-SMALL):
        for rung in res["rungs"]:
            assert rung["full_acc"] >= 0.80
            assert rung["abstain_correct_recent_only"] >= 0.90
            assert rung["abstain_correct_remote_only"] >= 0.90
    finally:
        (R._build_substrate, R._encode_recent_facts,
         R._build_remote_schema, R._hippo_silenced,
         R._compose_query) = saved


# ---------------------------------------------------------------------
# D3: the moat is fed an out-of-calibration quantity. Pin the hack is
# present + visible (NOT removed silently, NOT silently re-scaled
# without recalibration evidence).
# ---------------------------------------------------------------------
def test_D3_moat_input_is_energy_scaled_cosine_not_raw_rate(R):
    src = inspect.getsource(R._ranked_from_pattern)
    assert "np.linalg.norm" in src and "* energy" in src, (
        "the energy-scaling hack changed -- abstention_gate(650) was "
        "calibrated on RAW firing rates; any new scale must come with "
        "explicit recalibration evidence, not a silent edit"
    )
    # And the moat is genuinely the byte-unchanged 7/7 gate object.
    from research.runners import abstention_gate as ag
    assert R._abstain_gate is ag.gate
    assert R._MOAT == ag.DEFAULT_THRESHOLD == 650.0
