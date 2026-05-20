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
    per-regime architecture routes direct queries through the existing
    650 moat AND compositional queries through the calibrated 5.69
    compositional moat -- both are wired in by import."""
    src = Path(urr.__file__).read_text(encoding="utf-8", errors="ignore")
    # The runner imports / references BOTH gate modules.
    assert "abstention_gate" in src
    assert "abstention_gate_compositional" in src
    # Both threshold constants present (calibrated quantities).
    assert "MOAT_DIRECT" in src or "DEFAULT_THRESHOLD" in src
    assert "COMPOSITIONAL_THRESHOLD" in src
    # Both gate-call aliases or module-qualified calls visible.
    assert (
        ("gate_direct" in src) or ("abstention_gate.gate" in src)
    )
    assert (
        ("gate_compositional" in src)
        or ("abstention_gate_compositional.gate" in src)
    )
    # The uniform_ctrl arm is present as a named concept.
    assert "uniform_ctrl" in src


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
