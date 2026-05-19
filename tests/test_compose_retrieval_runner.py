"""Task 2 tests for the net-new regime-correct composition runner.

These pin the runner's CONTRACT, not toy accuracy numbers:
 (a) --tiny-synth smoke runs end-to-end, returns a dict with a `rungs`
     list and a `verdict` whose `gate` is one of the four legal states,
     and never raises;
 (b) every rung carries exactly the 7 required keys with correct
     types/ranges so the FROZEN verdict module does NOT VOID for a
     structural reason (it may legitimately FAIL/VOID on toy numbers --
     that is fine and asserted-around);
 (c) no shipped module on the runner's import graph pulls in
     torch.autograd / .backward (grep the source text);
 (d) the full pass and BOTH ablations for a given (seed, N) consume
     the SAME seed (one seed threaded into all three).

PLUS the four faithfulness-fix contract pins (2026-05-19 Task-3
adversarial re-review): the defects the dedicated review CONFIRMED are
now CLOSED and pinned closed:

 FIX A: the substrate is built EXACTLY as the validated v16 recipe
        (concept_pool_demo.build_concept_bridge) -- in particular the
        runner does NOT override cfg.num_traits (the validated recipe
        leaves the default).
 FIX B: all three arms (full / recent_only / remote_only) run the
        IDENTICAL query->retrieve->compose->decode->score pipeline and
        produce a GENUINELY MEASURED accuracy -- there is no
        `if groundable:` dead-bar short-circuit; the only accuracy
        increment is NOT dominated by a per-arm structural gate.
 FIX C: engram tag names are OPAQUE (never contain the answer word);
        the decoded answer is read from the validated NEURAL readout
        (lang_output firing pattern + cosine_to_word), not any string.
 FIX D: the moat is fed the validated RAW lang_output firing-rate
        confidence (the same quantity the validated concept readout
        produces), so the byte-unchanged 650 threshold is genuinely
        calibrated for it.

tiny_synth shrinks pools/episodes hard so this stays fast.
"""
from __future__ import annotations

import importlib
import inspect
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


# =====================================================================
#  FIX A -- substrate is the VALIDATED v16 recipe (no num_traits override)
# =====================================================================
def test_fixA_substrate_does_not_override_num_traits():
    """The validated recipe concept_pool_demo.build_concept_bridge does
    NOT set cfg.num_traits at all (it leaves the default). Forcing
    cfg.num_traits=1 diverges the substrate from the validated recipe.
    Pin that the runner's substrate builder performs no num_traits
    ASSIGNMENT (a comment documenting its deliberate ABSENCE is fine --
    same usage-vs-word discipline as the no-autograd pin)."""
    mod = importlib.import_module(_RUN_MOD)
    src = inspect.getsource(mod._build_substrate)
    # An ASSIGNMENT like `cfg.num_traits = 1` or `num_traits =` is the
    # divergence; the bare word in a comment that says we do NOT set it
    # is not. Pin the assignment is gone.
    assert not re.search(r"\bnum_traits\s*=", src), (
        "FIX A regressed: the runner assigns cfg.num_traits, diverging "
        "from the validated build_concept_bridge recipe (which leaves the "
        "default). Build the substrate exactly as the validated recipe."
    )
    # And it builds via the same CoreSimConfig field set the validated
    # recipe uses (sanity: the validated knobs are present).
    for fld in (
        "enable_brain_region_framework",
        "enable_nmda",
        "enable_hebbian_learning",
        "stdp_w_max",
        "fast_spike_reset",
    ):
        assert fld in src, f"validated recipe field {fld!r} missing"


def test_fixA_runner_builds_via_validated_recipe_builder():
    """The runner must construct the substrate by REUSING the validated
    v16 recipe's own region builder
    (text_minimal_isolation.build_biological_brain_regions) -- the ONLY
    builder that exposes enable_hippocampus_consolidation
    (concept_pool_demo.build_concept_bridge does not pass it) -- with
    the EXACT validated v16 weak-dynamics 16-pool kwargs, NOT a
    hand-rolled region set that can silently diverge. Spy the real
    builder and assert it is invoked with the validated recipe args
    plus the hippocampal recent-specific path."""
    import research.runners.text_minimal_isolation as tmi
    import research.runners.concept_pool_demo as cpd

    mod = importlib.import_module(_RUN_MOD)
    real = tmi.build_biological_brain_regions
    calls = []

    def spy(*a, **k):
        calls.append(k)
        return real(*a, **k)

    tmi.build_biological_brain_regions = spy
    try:
        mod.run_compose_retrieval(seeds=[42], loads=(2,), tiny_synth=True)
    finally:
        tmi.build_biological_brain_regions = real

    assert calls, (
        "FIX A: the runner did not build the substrate via the validated "
        "build_biological_brain_regions recipe builder"
    )
    k0 = calls[0]
    # The validated v16 weak-dynamics 16-pool recipe (mirrors
    # concept_pool_demo.build_concept_bridge weak_dynamics=True +
    # enable_adjective=True) PLUS the hippocampal recent-specific path.
    assert k0.get("enable_hippocampus_consolidation") is True
    assert k0.get("enable_noun_pools") is True
    assert k0.get("enable_verb_pools") is True
    assert k0.get("enable_adjective_pools") is True
    assert k0.get("noun_pool_names") == cpd.NOUN_NAMES
    assert k0.get("verb_pool_names") == cpd.VERB_NAMES
    assert k0.get("adjective_pool_names") == cpd.ADJECTIVE_NAMES
    # weak_dynamics=True validated concept-pool dynamics.
    assert k0.get("concept_pool_internal_density") == 0.05
    assert k0.get("concept_pool_exc_weight_mean") == 0.3
    assert k0.get("concept_pool_inh_weight_mean") == 0.8


# =====================================================================
#  FIX B -- ablation accuracies are GENUINELY MEASURED (no dead bar)
# =====================================================================
def test_fixB_no_dead_groundable_shortcircuit_in_score_arm():
    """The dead-bar artifact was: the only `n_correct += 1` sat behind
    `if groundable:` with `groundable == False` on both ablation arms,
    making recent_only_acc / remote_only_acc a structural constant 0.0.
    Pin that this short-circuit is gone: no `groundable`-named gate
    dominates the accuracy increment in _score_arm."""
    mod = importlib.import_module(_RUN_MOD)
    src = inspect.getsource(mod._score_arm)
    assert "groundable = have_remote and (not hippo_off)" not in src, (
        "FIX B regressed: the per-arm `groundable` dead-bar gate is back."
    )
    # There IS still exactly one accuracy increment, but it must be
    # reachable on EVERY arm (not dominated by an arm-structural gate).
    assert src.count("n_correct += 1") >= 1


def test_fixB_ablation_accuracy_is_a_live_measurement(monkeypatch):
    """End-to-end faithfulness: ablation accuracies are produced by the
    SAME pipeline as full and are a genuine function of the decoded
    answer -- NOT a hardcoded 0.0. Inject a stub `_compose_query` that
    deterministically answers correctly on whichever arm(s) we choose;
    the corresponding *_acc must move OFF 0.0."""
    mod = importlib.import_module(_RUN_MOD)
    saved = (mod._build_substrate, mod._encode_recent_facts,
             mod._build_remote_schema, mod._hippo_silenced,
             mod._compose_query)
    try:
        mod._build_substrate = lambda s, t: (
            object(),
            {"n_lang_input": 64, "n_per_pool": 12,
             "n_fs_per_pool": 3, "sparsity": 0.05},
        )
        mod._encode_recent_facts = lambda b, f, d, e: [
            f"fact_{i}" for i in range(len(f))
        ]
        mod._build_remote_schema = lambda *a, **k: None
        mod._hippo_silenced = lambda b, s=-2000.0: ((lambda: None), 0)

        # A solver that returns the CORRECT adjective on EVERY arm
        # regardless of which regime is present. If the ablation arms
        # genuinely run the same scoring pipeline, recent_only_acc and
        # remote_only_acc must be > 0 (the dead-bar artifact made them
        # impossible to be anything but 0.0).
        facts_for = mod._recent_facts

        def omniscient(bridge, cue, tag, dims, have_remote, recall_steps):
            for (noun, adj) in facts_for(8):
                if noun == cue:
                    return adj, [(adj, 9999.0, "c")], {}
            return None, [("x", 1.0, "c")], {}

        mod._compose_query = omniscient
        res = mod.run_compose_retrieval(seeds=[42, 43, 44],
                                        loads=(2, 4, 8), tiny_synth=True)
        for rung in res["rungs"]:
            assert rung["recent_only_acc"] > 0.0, (
                "FIX B regressed: recent_only_acc is still a dead "
                "constant -- an omniscient solver cannot move it"
            )
            assert rung["remote_only_acc"] > 0.0, (
                "FIX B regressed: remote_only_acc is still a dead "
                "constant -- an omniscient solver cannot move it"
            )
    finally:
        (mod._build_substrate, mod._encode_recent_facts,
         mod._build_remote_schema, mod._hippo_silenced,
         mod._compose_query) = saved


# =====================================================================
#  FIX C -- opaque tags + NEURAL readout (no tag-string reading)
# =====================================================================
def test_fixC_engram_tag_names_are_opaque():
    """Engram tag names must NOT carry the answer (no `f"{noun}_{adj}"`
    or `recent__{noun}__{adj}`). They must be opaque (e.g. fact_{i}) so
    a controller cannot read the answer out of the tag string."""
    mod = importlib.import_module(_RUN_MOD)
    src = inspect.getsource(mod._encode_recent_facts)
    # The old leaky pattern embedded noun/adj into the tag name.
    assert "f\"recent__{noun}__{adj}\"" not in src
    assert "{noun}__{adj}" not in src
    assert "{noun}_{adj}" not in src
    # An opaque, index-based tag name is used instead.
    assert "fact_" in src or "fact_{" in src, (
        "FIX C: expected opaque index-based tag names (e.g. fact_{i})"
    )
    # And the scorer must not split a tag string to recover the answer.
    sc = inspect.getsource(mod._score_arm)
    assert ".split(\"__\")" not in sc and ".split('__')" not in sc
    cq = inspect.getsource(mod._compose_query)
    assert ".split(\"__\")" not in cq and ".split('__')" not in cq


def test_fixC_decode_uses_validated_neural_readout():
    """The decoded answer must come from the validated NEURAL readout
    (lang_output firing pattern -> cosine_to_word), the same mechanism
    compose_concept_engram / compose_concept_chat use. Pin that
    _compose_query imports those validated readout helpers and uses
    cosine_to_word, with no synthetic arithmetic / tag-string parse."""
    mod = importlib.import_module(_RUN_MOD)
    cq = inspect.getsource(mod._compose_query)
    assert "lang_output_pattern_during_stim" in cq
    assert "lang_output_pattern_during_input" in cq
    rk = inspect.getsource(mod._ranked_from_pattern)
    assert "cosine_to_word" in rk, (
        "FIX C: ranking must use the validated cosine_to_word neural "
        "readout, not a hand-rolled string/arith decode"
    )


def test_fixC_tag_string_solver_cannot_PASS():
    """A 'solver' that does ZERO neural computation and reads the answer
    out of the engram tag string the runner constructs must NOT score
    PASS (defect CLOSED). Because tags are now opaque (fact_{i}) the
    tag string carries no answer; splitting it yields no adjective, so
    such a solver cannot clear the moat on the full arm -> NOT PASS."""
    mod = importlib.import_module(_RUN_MOD)
    saved = (mod._build_substrate, mod._encode_recent_facts,
             mod._build_remote_schema, mod._hippo_silenced,
             mod._compose_query)
    try:
        mod._build_substrate = lambda s, t: (
            object(),
            {"n_lang_input": 64, "n_per_pool": 12,
             "n_fs_per_pool": 3, "sparsity": 0.05},
        )
        # Opaque tags exactly like the real runner.
        mod._encode_recent_facts = lambda b, f, d, e: [
            f"fact_{i}" for i in range(len(f))
        ]
        mod._build_remote_schema = lambda *a, **k: None
        mod._hippo_silenced = lambda b, s=-2000.0: ((lambda: None), 0)

        from research.runners.abstention_gate import gate as _g
        from research.runners.abstention_gate import (
            DEFAULT_THRESHOLD as _M,
        )

        def tag_string_solver(bridge, cue, tag, dims, have_remote,
                              recall_steps):
            # Try every separator a cheater might use to recover an
            # answer from the tag string. Opaque fact_{i} yields none.
            adj = None
            if tag is not None:
                for sep in ("__", "_"):
                    parts = tag.split(sep)
                    if len(parts) >= 3 and parts[-1].isalpha() \
                            and parts[-1] != "fact":
                        adj = parts[-1]
            if adj is None:
                return None, [], {}
            ranked = [(adj, 9999.0, "compose")]
            decided = _g(ranked, _M)
            return (None if decided is None else decided[0]), ranked, {}

        mod._compose_query = tag_string_solver
        res = mod.run_compose_retrieval(seeds=[42, 43, 44],
                                        loads=(2, 4, 8), tiny_synth=True)
        assert res["verdict"]["gate"] != "PASS", (
            "FIX C regressed: a tag-string solver (zero neural compute) "
            "scored PASS -- the tag must be opaque and the answer must "
            "come from neural readout"
        )
    finally:
        (mod._build_substrate, mod._encode_recent_facts,
         mod._build_remote_schema, mod._hippo_silenced,
         mod._compose_query) = saved


def test_fixC_single_path_solver_cannot_PASS():
    """A single-path solver (answers from ONE regime only, composition
    a no-op) must NOT score PASS -- it answers in its still-present
    ablation arm, so that arm's measured accuracy is high AND its
    abstain_correct collapses. Either trips the frozen verdict."""
    mod = importlib.import_module(_RUN_MOD)
    saved = (mod._build_substrate, mod._encode_recent_facts,
             mod._build_remote_schema, mod._hippo_silenced,
             mod._compose_query)
    try:
        mod._build_substrate = lambda s, t: (
            object(),
            {"n_lang_input": 64, "n_per_pool": 12,
             "n_fs_per_pool": 3, "sparsity": 0.05},
        )
        mod._encode_recent_facts = lambda b, f, d, e: [
            f"fact_{i}" for i in range(len(f))
        ]
        mod._build_remote_schema = lambda *a, **k: None
        mod._hippo_silenced = lambda b, s=-2000.0: ((lambda: None), 0)

        facts_for = mod._recent_facts

        # Pure hippocampal single path: answers whenever a tag is
        # present (hippo on) -- i.e. on full AND on recent_only.
        def hip_only(bridge, cue, tag, dims, have_remote, recall_steps):
            if tag is not None:
                for (noun, adj) in facts_for(8):
                    if noun == cue:
                        return adj, [(adj, 9999.0, "c")], {}
            return None, [("x", 1.0, "c")], {}

        mod._compose_query = hip_only
        res = mod.run_compose_retrieval(seeds=[42, 43, 44],
                                        loads=(2, 4, 8), tiny_synth=True)
        assert res["verdict"]["gate"] != "PASS", (
            "FIX C regressed: a single-path (hippo-only) solver scored "
            "PASS -- the task must genuinely require BOTH regimes"
        )
    finally:
        (mod._build_substrate, mod._encode_recent_facts,
         mod._build_remote_schema, mod._hippo_silenced,
         mod._compose_query) = saved


# =====================================================================
#  FIX D -- moat is fed the validated RAW firing-rate confidence
# =====================================================================
def test_fixD_moat_input_is_raw_firing_rate_not_energy_scaled_cosine():
    """abstention_gate's 650 threshold was calibrated on RAW lang_output
    firing rates (encoded mean ~796, control max ~584;
    2026-05-16-G20-320-abstention-benchmark). The runner must feed the
    moat that calibrated quantity (a firing-rate confidence), NOT the
    old out-of-calibration `max(0,cos) * ||pattern||_2` hack."""
    mod = importlib.import_module(_RUN_MOD)
    rk = inspect.getsource(mod._ranked_from_pattern)
    # The retired hack multiplied cosine by the L2 energy of the pattern.
    assert "* energy" not in rk, (
        "FIX D regressed: moat still fed energy-scaled cosine; it must "
        "be fed the validated raw lang_output firing-rate confidence"
    )
    assert "np.linalg.norm(np.asarray(pattern))" not in rk
    # The validated calibrated quantity is the raw firing-rate readout.
    # Pin the function documents/uses a firing-rate confidence.
    assert "firing" in rk.lower() or "rate" in rk.lower(), (
        "FIX D: _ranked_from_pattern must produce the raw firing-rate "
        "confidence the validated readout/abstention benchmark calibrate"
    )
    # And the moat object is still the byte-unchanged 7/7 gate at 650.
    from research.runners import abstention_gate as ag
    assert mod._abstain_gate is ag.gate
    assert mod._MOAT == ag.DEFAULT_THRESHOLD == 650.0
