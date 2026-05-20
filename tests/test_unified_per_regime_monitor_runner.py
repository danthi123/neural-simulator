"""TDD tests for the net-new unified per-regime monitor + per-regime
encoding runner.

Written FIRST (red before the runner lands). The decisive multi-seed CuPy
run is a later controller-only task (Task 4); this suite screens only
the orchestration discipline of the unified architecture:

  (a) ``run_unified_per_regime_monitor(seeds=[42,43,44], loads=(2,3,5),
      tiny_synth=True, phase1_cache_dir=<tmp>)`` runs end-to-end (Phase-1
      training shrunk; compositional encoding one pair per rung) and
      returns a dict with ``rungs`` + ``verdict`` whose ``gate`` is one
      of the four frozen states, AND NEVER raises;
  (b) per-seed Phase-1 checkpoints exist after the run at the expected
      cache-dir path convention (``{phase1_cache_dir}/seed{seed}.simstate.h5``);
  (c) no shipped module text imports torch.autograd / ``.backward``;
  (d) OPAQUE tag names: the runner source contains no ``.split("_")``
      on tag names (Stage-1 / SPEAR / Pirazzini / Per-regime lesson);
  (e) BOTH moats are wired in: the runner source contains references
      to BOTH ``gate_direct`` and ``gate_compositional`` (or the
      equivalent module identifiers);
  (f) ``direct_retain_acc`` is computed from the SAME full-arm run as
      ``full_acc`` -- the runner source contains a single full-arm loop
      that accumulates BOTH direct-and-correct and overall-correct
      counters (no separate full vs direct-only draws);
  (g) Phase-1 caching works: a second invocation with the SAME
      ``phase1_cache_dir`` does NOT call ``run_concept_pool_demo``
      again (the expensive multi-event training is amortised across
      decisive-run invocations).

tiny_synth shrinks Phase-1 training events + compositional pair count
hard so the smoke is seconds (toy numbers explicitly NOT a result).
"""
from __future__ import annotations

import inspect
from pathlib import Path
from unittest import mock

import pytest

import research.runners.unified_per_regime_monitor_runner as urr
from research.runners.per_regime_monitor_core import (
    REQUIRED_KEYS,
    per_regime_monitor_verdict,
)
from research.runners.abstention_gate_compositional import (
    COMPOSITIONAL_THRESHOLD,
)
from research.runners.abstention_gate import (
    DEFAULT_THRESHOLD as _MOAT_DIRECT,
)
from research.runners.abstention_gate_direct_unified import (
    DIRECT_UNIFIED_THRESHOLD,
)


_VALID_GATES = {
    "VOID",
    "FAIL",
    "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE",
    "PASS",
}


def test_runner_module_exposes_entry_point():
    """Pin-style: the runner module exposes the entry point + main()."""
    assert hasattr(urr, "run_unified_per_regime_monitor")
    assert callable(urr.run_unified_per_regime_monitor)
    assert hasattr(urr, "main")


def test_run_signature_threads_phase1_cache_dir(tmp_path):
    """The entry point accepts ``seeds``, ``loads``, ``tiny_synth``,
    ``phase1_cache_dir``, ``out_path``, ``ckpt`` -- the orchestration
    contract the controller drives."""
    sig = inspect.signature(urr.run_unified_per_regime_monitor)
    assert "seeds" in sig.parameters
    assert "loads" in sig.parameters
    assert "tiny_synth" in sig.parameters
    assert "phase1_cache_dir" in sig.parameters
    assert "out_path" in sig.parameters
    assert "ckpt" in sig.parameters


def test_tiny_synth_end_to_end_well_formed(tmp_path):
    """(a) + (b): a tiny-synth multi-seed run runs end-to-end, returns
    a well-formed dict the frozen verdict accepts (one of four gate
    states, never raises), AND the Phase-1 cache files are present
    afterwards at the expected path convention.
    """
    cache_dir = tmp_path / "phase1"
    result = urr.run_unified_per_regime_monitor(
        seeds=[42, 43, 44],
        loads=(2, 3, 5),
        tiny_synth=True,
        phase1_cache_dir=str(cache_dir),
    )

    assert isinstance(result, dict)
    assert "rungs" in result and isinstance(result["rungs"], list)
    assert len(result["rungs"]) >= 1
    assert "verdict" in result and isinstance(result["verdict"], dict)

    gate = result["verdict"]["gate"]
    assert gate in _VALID_GATES

    # Every rung must carry EXACTLY the six required keys with correct
    # types/ranges so the frozen verdict does not VOID structurally.
    for r in result["rungs"]:
        assert isinstance(r, dict)
        for k in REQUIRED_KEYS:
            assert k in r, "rung missing required key %s" % k
        assert isinstance(r["N"], int) and not isinstance(r["N"], bool)
        assert isinstance(r["n_seeds"], int) and not isinstance(
            r["n_seeds"], bool
        )
        for ak in (
            "full_acc",
            "uniform_ctrl_acc",
            "direct_retain_acc",
            "abstain_correct",
        ):
            v = r[ak]
            assert isinstance(v, float) and not isinstance(v, bool)
            assert 0.0 <= v <= 1.0

    # Recomputed verdict from the rungs must also be valid (not VOID
    # for structural reasons).
    recomputed = per_regime_monitor_verdict(result["rungs"])
    assert recomputed["gate"] in _VALID_GATES
    assert recomputed["gate"] != "VOID", (
        "tiny-synth rungs must be structurally well-formed; VOID here "
        "means a malformed rung shape, got reason=%r"
        % recomputed.get("reason")
    )

    # (b) Phase-1 cache files exist at the expected path convention.
    for s in [42, 43, 44]:
        ckpt_path = cache_dir / ("seed%d.simstate.h5" % s)
        assert ckpt_path.exists(), (
            "Phase-1 cache checkpoint missing for seed %d at %s" % (s, ckpt_path)
        )


def test_tiny_synth_writes_json_with_out_path(tmp_path):
    """The runner writes the full result JSON when out_path is provided."""
    cache_dir = tmp_path / "phase1"
    out = tmp_path / "unified_eval_smoke.json"
    result = urr.run_unified_per_regime_monitor(
        seeds=[42, 43, 44],
        loads=(2,),
        tiny_synth=True,
        phase1_cache_dir=str(cache_dir),
        out_path=str(out),
    )
    assert out.exists()
    assert result["verdict"]["gate"] in _VALID_GATES
    assert result.get("tiny_synth") is True
    # The smoke must explicitly disclaim its toy numbers.
    note = result.get("note", "")
    assert "NOT a result" in note


def test_no_autograd_on_shipped_path():
    """(c): no shipped module text imports torch.autograd / .backward."""
    src = Path(urr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "torch.autograd" not in src
    assert ".backward(" not in src
    assert "import torch" not in src


def test_decode_is_neural_not_tag_string_parse():
    """(d): the runner source contains NO tag-string parse on tag names
    (.split("_")). Stage-1 / SPEAR / Pirazzini / Per-regime lesson: the
    answer is decoded from the validated neural readout, never out of
    an opaque tag name."""
    src = Path(urr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert '.split("_")' not in src
    assert ".split('_')" not in src
    # Tag names must be opaque: f"ep_{i}" or similar.
    assert ("ep_%d" in src) or ("ep_{i}" in src) or ('"ep_"' in src) \
        or ("f\"ep_" in src) or ("f'ep_" in src)


def test_both_moats_are_wired_in():
    """(e): BOTH moats are referenced in the runner source. The
    per-regime architecture routes direct queries through the
    substrate-specific direct gate (placeholder 0.0 until calibrated)
    AND compositional queries through the calibrated 5.69 compositional
    moat -- both are wired in by import. The existing 650 moat stays
    byte-unchanged as historical G.20 SharedPool calibration; it is
    still importable but no longer used to gate direct queries in the
    unified runner (substrate-mismatch defect #2 closure)."""
    src = Path(urr.__file__).read_text(encoding="utf-8", errors="ignore")
    # The runner imports / references BOTH gate modules.
    assert "abstention_gate" in src
    assert "abstention_gate_compositional" in src
    # The new substrate-specific direct gate is wired in too.
    assert "abstention_gate_direct_unified" in src
    # All three threshold constants present (calibrated quantities).
    assert "MOAT_DIRECT" in src or "DEFAULT_THRESHOLD" in src
    assert "COMPOSITIONAL_THRESHOLD" in src
    assert "DIRECT_UNIFIED_THRESHOLD" in src
    # Module-qualified gate calls visible. The unified runner now
    # routes direct queries through gate_direct_unified.
    assert (
        ("gate_direct_unified" in src)
        or ("abstention_gate_direct_unified.gate" in src)
    )
    assert (
        ("gate_compositional" in src)
        or ("abstention_gate_compositional.gate" in src)
    )
    # The uniform_ctrl arm is present as a named concept.
    assert "uniform_ctrl" in src


def test_direct_queries_route_through_new_substrate_specific_gate():
    """Defect #2 closure: direct queries now route through the new
    substrate-specific gate ``gate_direct_unified`` (placeholder
    threshold 0.0 until calibration ships the calibrated value via a
    controller commit). The existing 650 moat
    (``abstention_gate.gate_direct``, calibrated on G.20 SharedPool
    ``recall_rates``, scale ~500-800) is structurally unreachable by
    ``measure_pool_firing`` (per-neuron mean rate, scale ~0.5-2) and
    so cannot gate the unified runner's direct readout faithfully.

    Source-level pin: the new gate's import is present AND the runner
    routes direct queries through ``gate_direct_unified``, not through
    the historical 650 moat. We assert the call site exists.
    """
    src = Path(urr.__file__).read_text(encoding="utf-8", errors="ignore")
    # New gate import is present.
    assert "from research.runners.abstention_gate_direct_unified import" in src
    # The runner routes direct queries through the new gate (the call
    # site appears at least once in source). For direct queries the
    # existing 650 moat is no longer used in this runner.
    assert "gate_direct_unified(" in src
    # Placeholder threshold value matches the module constant.
    assert "DIRECT_UNIFIED_THRESHOLD" in src


def test_new_gate_module_threshold_pin():
    """Cross-pin: the runner imports the new gate AND the module's
    threshold constant is the documented placeholder (0.0). The
    controller commits the calibrated value in a SEPARATE commit
    (mirrors ``abe65f6`` for the compositional gate); until then the
    placeholder makes the gate effectively transparent so the runner
    can boot for the calibration step itself.
    """
    from research.runners.abstention_gate_direct_unified import (
        DIRECT_UNIFIED_THRESHOLD as _DUT,
        gate as _g,
        abstain as _a,
    )
    assert _DUT == 0.0
    # Surface-level invariants mirror the existing moats.
    assert _a(0.0) is True
    assert _a(0.1) is False
    assert _g(None) is None
    assert _g([]) is None


def test_calibration_mode_runs_direct_gate_calibration(tmp_path):
    """The runner accepts ``calibrate=True`` and produces a JSON
    payload containing BOTH a compositional-gate calibration result
    AND a direct-gate calibration result (per-seed thresholds +
    aggregate + status). Tiny-synth smoke; the decisive multi-seed
    CuPy calibration is a controller-only task.

    The status reporting parallels the compositional gate: MATCH /
    PENDING / MISMATCH / INSUFFICIENT-SEPARATION. INSUFFICIENT-
    SEPARATION is the EXPECTED status on tiny-synth toy data per the
    per-regime stage's pattern (the strengthen-only fix in
    ``per_regime_monitor_runner._calibration_status``); the runner
    must NOT crash on this case but report it cleanly.
    """
    cache_dir = tmp_path / "phase1"
    result = urr.run_unified_per_regime_monitor(
        seeds=[42, 43, 44],
        loads=(2,),
        tiny_synth=True,
        phase1_cache_dir=str(cache_dir),
        calibrate=True,
    )
    assert isinstance(result, dict)
    assert result.get("mode") == "calibration"
    # Compositional gate calibration block (parallels the per-regime
    # runner's calibration mode).
    comp = result.get("compositional_gate", result)
    assert "per_seed_calibrated_thresholds" in comp
    assert "aggregate_calibrated_threshold" in comp
    assert "committed_threshold" in comp
    assert "calibration_status" in comp
    assert comp["calibration_status"] in {
        "MATCH",
        "PENDING",
        "MISMATCH",
        "INSUFFICIENT-SEPARATION",
    }
    # Direct gate calibration block: the new substrate-specific
    # calibration result lives alongside the compositional one.
    direct = result.get("direct_gate")
    assert isinstance(direct, dict), (
        "calibrate-mode result must include a direct_gate block "
        "with per-seed thresholds + aggregate + status"
    )
    assert "per_seed_calibrated_thresholds" in direct
    assert "aggregate_calibrated_threshold" in direct
    assert "committed_threshold" in direct
    assert direct["committed_threshold"] == float(DIRECT_UNIFIED_THRESHOLD)
    assert "calibration_status" in direct
    assert direct["calibration_status"] in {
        "MATCH",
        "PENDING",
        "MISMATCH",
        "INSUFFICIENT-SEPARATION",
    }
    # Per-seed payload has the right shape (one entry per seed).
    assert len(direct["per_seed_calibrated_thresholds"]) == 3


def test_calibrate_mode_writes_json_with_out_path(tmp_path):
    """The runner writes the calibration result JSON when out_path is
    provided in calibrate mode."""
    cache_dir = tmp_path / "phase1"
    out = tmp_path / "unified_calibrate_smoke.json"
    result = urr.run_unified_per_regime_monitor(
        seeds=[42],
        loads=(2,),
        tiny_synth=True,
        phase1_cache_dir=str(cache_dir),
        calibrate=True,
        out_path=str(out),
    )
    assert out.exists()
    assert result.get("mode") == "calibration"


def test_direct_retain_is_read_from_same_full_run():
    """(f): the runner computes ``direct_retain_acc`` from the SAME
    full-arm run as ``full_acc`` -- the runner source contains a
    single accumulator structure that records BOTH the direct-only
    correctness count and the overall-correct count inside the same
    arm. Direct_retain_acc is the direct-query subset of full's run,
    not a separate draw.
    """
    src = Path(urr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "direct_retain_acc" in src
    # The marker for "same run, two counters" is a named direct-only
    # counter inside the full arm. Match either of the conventional
    # spellings used in the per-regime runner.
    assert (
        ("n_direct_correct_full" in src)
        or ("n_direct_correct" in src and "n_full_correct" in src)
        or ("direct_correct_full" in src)
    )


def test_phase1_caching_skips_retraining(tmp_path):
    """(g): a second invocation with the same ``phase1_cache_dir`` does
    NOT re-train (does not call ``concept_pool_demo.run_concept_pool_demo``)
    because the cached Phase-1 checkpoint already exists. The first
    invocation populates the cache; the second invocation must skip
    Phase-1 training entirely.
    """
    cache_dir = tmp_path / "phase1"
    # First invocation: Phase-1 training MUST run (no cache yet).
    urr.run_unified_per_regime_monitor(
        seeds=[42],
        loads=(2,),
        tiny_synth=True,
        phase1_cache_dir=str(cache_dir),
    )
    ckpt_path = cache_dir / "seed42.simstate.h5"
    assert ckpt_path.exists(), (
        "Phase-1 cache missing after first invocation"
    )

    # Second invocation with the same cache_dir: the runner MUST NOT
    # call ``run_concept_pool_demo`` again (the cache is warm). We
    # patch the cpd entry point and assert it is not invoked.
    import research.runners.concept_pool_demo as cpd_module
    with mock.patch.object(
        cpd_module, "run_concept_pool_demo",
        side_effect=AssertionError(
            "Phase-1 training must be skipped when the cache file exists"
        ),
    ):
        urr.run_unified_per_regime_monitor(
            seeds=[42],
            loads=(2,),
            tiny_synth=True,
            phase1_cache_dir=str(cache_dir),
        )


def test_main_entry_accepts_relevant_flags():
    """The CLI must expose --seeds, --loads, --tiny-synth, --phase1-cache-dir,
    --out, --ckpt so the controller can drive the full-scale decisive
    run without changing the runner."""
    src = Path(urr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert '"--tiny-synth"' in src
    assert '"--seeds"' in src
    assert '"--loads"' in src
    assert '"--phase1-cache-dir"' in src
    assert '"--out"' in src
    assert '"--ckpt"' in src


def test_phase1_recipe_kwargs_match_v14_v16_validated():
    """The runner's Phase-1 training call uses the v14/v16-validated
    recipe kwargs (the 88.75% multi-seed kwargs).

    Source check: the recipe constants the controller can audit appear
    in the source (the runner is not silently passing a different
    recipe that wouldn't be calibrated against the 650 direct moat).
    """
    src = Path(urr.__file__).read_text(encoding="utf-8", errors="ignore")
    # The v14/v16 recipe markers (concept_pool_demo.run_concept_pool_demo
    # is the validated entry; these kwargs name the validated dial-set).
    assert "run_concept_pool_demo" in src
    assert "weak_dynamics" in src
    assert "interleaved" in src
    assert "topographic_factor" in src
    assert "off_target_factor" in src
    assert "enable_adjective" in src
    assert "orthogonal_codes" in src
    assert "sparsity" in src


def test_compositional_encoding_uses_validated_pair_helper():
    """The runner uses the validated ``encode_concept_pair`` from
    ``compose_concept_engram`` (the protected one-shot binding helper
    that opens / closes the ``cross_pool_concept`` gate around the
    encoding window). Source check.
    """
    src = Path(urr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "encode_concept_pair" in src
    assert "compose_concept_engram" in src


def test_kill_safety_uses_train_checkpoint():
    """The runner mirrors the prior runners' kill-safety pattern via
    sim.train_checkpoint (save_checkpoint / load_checkpoint /
    resume_epoch)."""
    src = Path(urr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "sim.train_checkpoint" in src
    assert "save_checkpoint" in src
    assert "load_checkpoint" in src
    assert "resume_epoch" in src


# =====================================================================
# Adversarial-review-block fix (substrate defect #1):
# The prior runner built on ``cpd.build_concept_bridge`` whose substrate
# has NO hippocampal regions (no dg / ca3 / ca1) -- but compositional
# encoding via ``encode_concept_pair`` uses ``region_filter=
# ["dg","ca3","ca1"]``. ``commit_engram_tag`` silently swallows
# missing-region errors -> engram tags get ``n_tagged=0`` -> the
# compositional arm of ``full_acc`` is structurally inert.
#
# The fix rebuilds the substrate on
# ``text_minimal_isolation.build_biological_brain_regions(
# enable_hippocampus_consolidation=True, ...)`` -- the same builder
# Stage-1 / SPEAR / Pirazzini / Per-regime all used. The new substrate
# has BOTH hippocampus AND concept pools, so the engram region_filter
# now resolves to a real index set and ``commit_engram_tag`` produces
# tags with non-zero ``n_tagged``.
#
# These two pins ARE the success criterion for the substrate fix.
# =====================================================================


def test_substrate_has_hippocampal_regions(tmp_path):
    """The substrate the unified runner builds for a single (seed, N)
    cell MUST have the hippocampal trisynaptic-loop regions present
    (dg, ca3, ca1). Without these, ``commit_engram_tag(region_filter=
    ["dg","ca3","ca1"])`` produces zero-neuron tags and the
    compositional arm is structurally inert.

    We exercise the runner's actual cell-builder path -- not a probe --
    so the pin reflects what the decisive run will produce.
    """
    cache_dir = tmp_path / "phase1"
    # Train Phase-1 for one seed (writes cached checkpoint).
    urr.run_unified_per_regime_monitor(
        seeds=[42],
        loads=(2,),
        tiny_synth=True,
        phase1_cache_dir=str(cache_dir),
    )

    # Build the substrate the runner's evaluation arm uses (tiny-synth).
    bridge = urr._build_bridge_with_phase1_recipe(seed=42, tiny_synth=True)
    rm = bridge.region_manager

    # All three trisynaptic-loop regions must be present + non-empty.
    for region_name in ("dg", "ca3", "ca1"):
        try:
            idx = rm.indices(region_name)
        except KeyError as exc:
            raise AssertionError(
                "substrate missing hippocampal region %r -- the "
                "engram region_filter [dg, ca3, ca1] cannot resolve; "
                "compositional tags will be zero-neuron and the "
                "compositional arm structurally inert "
                "(prior adversarial-review-blocked defect #1)"
                % region_name
            ) from exc
        assert len(list(idx)) > 0, (
            "hippocampal region %r exists but has zero neurons -- "
            "engram region_filter would still produce zero-neuron tags"
            % region_name
        )


def test_compositional_encoding_produces_nonzero_engram_tag(tmp_path):
    """After ONE compositional encoding via the runner's encoding
    helper, the committed engram tag MUST have ``n_tagged > 0``. The
    prior adversarial review blocked precisely on this: zero-neuron
    engram tags (because the substrate had no hippocampal regions for
    the engram's region_filter to resolve against), which made the
    compositional arm structurally inert and the 5.69 gate always
    abstain on the zero-neuron-tag-stim noise.

    The fix is the substrate redesign: ``build_biological_brain_regions
    (enable_hippocampus_consolidation=True, ...)`` -- the same builder
    Stage-1 / SPEAR / Pirazzini / Per-regime all used -- has BOTH
    hippocampus AND concept pools, so the engram region_filter now
    resolves to a real index set.
    """
    cache_dir = tmp_path / "phase1"
    # Train Phase-1 for one seed (writes cached checkpoint).
    urr.run_unified_per_regime_monitor(
        seeds=[42],
        loads=(2,),
        tiny_synth=True,
        phase1_cache_dir=str(cache_dir),
    )

    bridge = urr._build_bridge_with_phase1_recipe(seed=42, tiny_synth=True)
    bridge.load_checkpoint(str(cache_dir / "seed42.simstate.h5"))
    urr._freeze_phase1_gates(bridge)

    # Encode ONE compositional pair via the runner's _encode_facts
    # helper (the SAME helper the evaluation arm calls). Use tiny-synth
    # encoding steps for fast smoke; the structural check is the
    # n_tagged > 0 result, not signal quality.
    recipe_dims = urr._phase1_recipe(tiny_synth=True)
    dims = {
        "n_lang_input": int(recipe_dims["n_lang_input"]),
        "n_per_pool": int(recipe_dims["n_per_pool"]),
        "n_fs_per_pool": int(recipe_dims["n_fs_per_pool"]),
        "sparsity": 0.05,
        "dt_ms": 0.5,
        "n_words_for_orthogonal": 16,
    }
    facts = [("apple", "big")]
    tags = urr._encode_facts(bridge, facts, dims, encoding_steps=8)
    assert len(tags) == 1, "_encode_facts must return one tag per fact"

    tag_records = bridge.list_engram_tags()
    assert tag_records, "no engram tags committed after compositional encoding"
    # Find the just-committed tag and inspect n_tagged.
    by_name = {t["name"]: t for t in tag_records}
    assert tags[0] in by_name, (
        "expected tag %r among list_engram_tags() result %r"
        % (tags[0], list(by_name.keys()))
    )

    # ``list_engram_tags`` reports the committed size as ``n_neurons``;
    # ``commit_engram_tag`` reports ``n_tagged`` in its own return-dict.
    # Both are the same quantity (the size of the int64 index array).
    n_tagged = int(by_name[tags[0]].get(
        "n_neurons",
        by_name[tags[0]].get("n_tagged", 0),
    ))
    assert n_tagged > 0, (
        "compositional engram tag has zero tagged neurons -- the "
        "substrate's region_filter regions [dg, ca3, ca1] resolved "
        "to an empty index set, so commit_engram_tag silently produced "
        "a zero-neuron tag (prior adversarial-review-blocked defect #1). "
        "Expected n_tagged > 0; got n_tagged = %d." % n_tagged
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
