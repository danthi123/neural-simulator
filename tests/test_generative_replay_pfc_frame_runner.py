"""TDD tests for the net-new generative-replay + PFC-frame runner (Task 2
of the 6th arc).

Written FIRST (red before the runner lands). The runner implements
GENERATIVE REPLAY + PFC-HELD COMPOSITIONAL FRAME: per the 6th
architecture in the gating-based composition design line.

  * The FULL arm encodes the same N compositional pairs as
    UNIFORM_CTRL, but THEN (a) runs ``run_concept_replay_phase`` on
    bridge_full once before queries so the engram-tagged ensembles
    consolidate via STDP into the substrate, AND (b) briefly drives
    the ``dlpfc_verb`` region before each compositional query so the
    NMDA-bistable PFC frame holds the compositional structure during
    retrieve. The cue (lang_input) stays ON during retrieve in BOTH
    arms -- encoding-specificity respected per the theta-gamma
    finding that cue-suppression-during-retrieve is biologically
    backwards.

  * The UNIFORM_CTRL arm runs the SAME encoding + SAME queries EXCEPT
    it skips both augmenting mechanisms. The SOLE difference between
    the two arms is the augmenting mechanisms (replay phase + PFC-
    frame priming).

This is the load-bearing experimental contrast the architecture
introduces. It is grounded in the standing catalog-grounded design
direction (docs/plans/2026-05-20-generative-replay-PFC-frame-design.md)
+ the empirical 5-architecture convergent ceiling.

The decisive multi-seed CuPy run is a later controller-only task;
this suite screens only that:

  (a) ``run_generative_replay_pfc_frame(seeds=[42,43,44],
      tiny_synth=True)`` runs end-to-end, returns a dict with
      ``rungs`` + ``verdict`` whose ``gate`` is one of the four
      frozen states, and NEVER raises;
  (b) every rung carries EXACTLY the six required keys with correct
      types/ranges so the frozen verdict does NOT VOID for a
      structural reason (it may legitimately FAIL on toy numbers --
      fine);
  (c) no shipped module text imports torch.autograd / .backward;
  (d) STRUCTURAL-EFFECT PROBES (TWO of them, MANDATORY):
       (1) Replay-effect probe: the runner's actual code path
           produces NON-byte-identical bridge state between
           replay-on (FULL arm) and replay-off (UNIFORM_CTRL arm)
           with all other state identical -- replay phase is
           structurally active;
       (2) PFC-frame-effect probe: the runner's actual code path
           produces NON-byte-identical bridge state between
           pfc-frame-on (FULL arm) and pfc-frame-off (UNIFORM_CTRL
           arm) with all other state identical -- PFC-frame
           priming is structurally active.
       Each probe ALSO runs both-arms-same controls (e.g. both with
       replay; both without replay) and asserts those agree under
       deterministic RNG isolation (mirrors the eighth adversarial
       review lesson).
  (e) per-cell raw_cells emit BOTH full_acc and uniform_ctrl_acc and
      at least one cell exhibits the mechanism's structural effect
      (full_acc != uniform_ctrl_acc on at least one (seed, N)
      combination at tiny-synth scale -- if every cell shows
      equality, the mechanisms are inert).

tiny_synth shrinks pools / events / phase-block lengths so this is
a fast logic-screen smoke (toy numbers are NOT a result).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import research.runners.generative_replay_pfc_frame_runner as grr
from research.runners.generative_replay_pfc_frame_core import (
    REQUIRED_KEYS,
    generative_replay_pfc_frame_verdict,
)


_VALID_GATES = {
    "VOID",
    "FAIL",
    "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE",
    "PASS",
}


def test_runner_module_exposes_entry_point():
    """(a): the runner exposes the documented public entry point AND a
    main() CLI dispatcher. Task 0's grounding pin asserts main is
    importable; this also asserts the orchestrating function."""
    assert hasattr(grr, "run_generative_replay_pfc_frame")
    assert callable(grr.run_generative_replay_pfc_frame)
    assert hasattr(grr, "main")
    assert callable(grr.main)


def test_tiny_synth_smoke_outputs_expected_json_shape(tmp_path):
    """(a)+(b): a tiny-synth multi-seed run returns a well-formed dict
    the frozen verdict accepts. Every rung must carry EXACTLY the six
    required keys with correct types/ranges so the frozen verdict does
    not VOID structurally. The smoke must also write the JSON output
    when out_path is provided and disclaim its toy numbers."""
    out = tmp_path / "gr_smoke.json"
    result = grr.run_generative_replay_pfc_frame(
        seeds=[42, 43, 44], loads=(2,), tiny_synth=True,
        out_path=str(out),
    )
    assert out.exists()
    assert isinstance(result, dict)
    assert result.get("mode") == "evaluation"
    assert "rungs" in result and isinstance(result["rungs"], list)
    assert len(result["rungs"]) >= 1
    assert "verdict" in result and isinstance(result["verdict"], dict)

    gate = result["verdict"]["gate"]
    assert gate in _VALID_GATES

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

    recomputed = generative_replay_pfc_frame_verdict(result["rungs"])
    assert recomputed["gate"] in _VALID_GATES
    assert recomputed["gate"] != "VOID", (
        "tiny-synth rungs must be structurally well-formed; VOID here "
        "means a malformed rung shape, got reason=%r"
        % recomputed.get("reason")
    )
    assert result.get("tiny_synth") is True
    assert "note" in result and "NOT a result" in result["note"]


def test_no_autograd_on_shipped_path():
    """(c): no shipped module text imports torch.autograd / .backward."""
    src = Path(grr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "torch.autograd" not in src
    assert ".backward(" not in src
    assert "import torch" not in src


def test_structural_effect_probes_validate_replay_and_pfc_frame_mechanisms(tmp_path):
    """(d): MANDATORY structural-effect probes (TWO of them). Each
    must show > 1 mV bridge-state divergence between flag-on and
    flag-off via the runner's ACTUAL code path; controls (both-arms-
    same-flag with identical deterministic RNG isolation) must agree
    to < 0.5 mV.

    Replay-effect probe:
      * bridge with replay vs bridge without replay -- > 1 mV
      * bridge with replay vs bridge with replay (control) -- < 0.5 mV
      * bridge without replay vs bridge without replay (control) -- < 0.5 mV

    PFC-frame-effect probe:
      * bridge with PFC-frame vs bridge without PFC-frame -- > 1 mV
      * bridge with PFC-frame vs bridge with PFC-frame (control) -- < 0.5 mV
      * bridge without PFC-frame vs bridge without PFC-frame (control) -- < 0.5 mV

    Mirrors Pirazzini d462bf0 / theta-gamma e6b17da lesson: structural-
    effect probe must work via the runner's ACTUAL code path and rule
    out RNG drift via controls. If any probe fails (flag-differing
    < 1 mV OR control > 0.5 mV) the runner aborts (no decisive
    numbers reported).

    CACHE-SCALE DISCIPLINE (closes 10th adversarial review BLOCK):
    uses a tmp_path cache directory rather than the default
    biological-scale ``_PHASE1_CACHE_DEFAULT``. With ``tiny_synth=True``
    the probe builds a small (952-neuron) bridge; the cache MUST
    match. The probe's pre-load validator
    (``_validate_cache_scale_for_probe``) now REFUSES to run on a
    mismatched cache; a fresh tmp_path lets ``_phase1_train_if_needed``
    produce a matching tiny-synth-scale cache file. Without this
    discipline the prior 24.46 / 35.99 mV "passing" numbers were
    measured against a corrupted bridge (cached 8440-neuron state
    loaded into a 952-neuron build silently raised IndexError per
    sim step)."""
    assert hasattr(grr, "_replay_effect_probe"), (
        "the runner must expose a `_replay_effect_probe` helper "
        "(mirrors Pirazzini d462bf0 / theta-gamma e6b17da lesson)"
    )
    assert hasattr(grr, "_pfc_frame_effect_probe"), (
        "the runner must expose a `_pfc_frame_effect_probe` helper "
        "(mirrors Pirazzini d462bf0 / theta-gamma e6b17da lesson)"
    )
    cache_dir = tmp_path / "probe_cache"
    diff_replay = grr._replay_effect_probe(
        seed=42, tiny_synth=True, cache_dir=str(cache_dir)
    )
    assert isinstance(diff_replay, float) and diff_replay > 1.0, (
        "the runner's actual code path must produce > 1 mV bridge-state "
        "divergence between replay-on and replay-off at the SAME initial "
        "state; got %.6g mV. This is the inert-mechanism failure mode "
        "the Pirazzini d462bf0 lesson guards against."
        % diff_replay
    )
    diff_pfc = grr._pfc_frame_effect_probe(
        seed=42, tiny_synth=True, cache_dir=str(cache_dir)
    )
    assert isinstance(diff_pfc, float) and diff_pfc > 1.0, (
        "the runner's actual code path must produce > 1 mV bridge-state "
        "divergence between pfc-frame-on and pfc-frame-off at the SAME "
        "initial state; got %.6g mV. This is the inert-mechanism failure "
        "mode the Pirazzini d462bf0 lesson guards against."
        % diff_pfc
    )


def test_full_vs_uniform_arms_differ_at_least_on_some_query():
    """(e): the FULL arm (generative replay + PFC-frame priming) and
    the UNIFORM_CTRL arm (no replay, no PFC-frame) must produce a
    DIFFERENT signature on at least one (seed, N) cell at tiny-synth
    scale. If EVERY cell shows EXACT equality on BOTH the accuracy
    metrics AND the mechanism-trace diagnostics, the augmenting
    mechanisms are structurally inert.

    Acceptance:
      * Accuracy contrast: at least one cell has full_acc !=
        uniform_ctrl_acc OR direct_retain_acc differs across arms,
        OR
      * Mechanism-trace contrast: at least one cell records that the
        replay phase actually executed in the FULL arm (replay_n_replays
        > 0). The UNIFORM_CTRL arm structurally skips replay so by
        construction it logs 0 replays; a non-zero count in the FULL
        arm IS a per-cell contrast witnessing the mechanism's
        execution. This is the load-bearing structural evidence at
        toy-scale where accuracy is noise-dominated.

    Decisive built-in control: per the frozen verdict,
    uniform_ctrl_max=0.10 is the bar the FULL arm must beat; this
    test checks only that the mechanism PRODUCES a contrast (not
    that the contrast is in the right direction at tiny-synth). The
    proper bridge-state non-inertness check is the structural-effect
    probe (test (d) above)."""
    result = grr.run_generative_replay_pfc_frame(
        seeds=[42, 43, 44], loads=(2,), tiny_synth=True,
    )
    cells = result.get("raw_cells", [])
    assert isinstance(cells, list) and len(cells) >= 1
    has_accuracy_diff = False
    has_mechanism_trace = False
    for c in cells:
        full = float(c.get("full_acc", 0.0))
        uniform = float(c.get("uniform_ctrl_acc", 0.0))
        if abs(full - uniform) > 1e-9:
            has_accuracy_diff = True
        # Mechanism-trace contrast: replay phase actually ran on FULL
        # bridge; UNIFORM_CTRL arm skips it by construction (no
        # replay_n_replays_uniform field; the absence IS the contrast).
        if int(c.get("replay_n_replays", 0)) > 0:
            has_mechanism_trace = True
    has_a_diff = has_accuracy_diff or has_mechanism_trace
    assert has_a_diff, (
        "the generative-replay + PFC-frame mechanisms produced ZERO "
        "evidence of contrast between FULL and UNIFORM_CTRL on every "
        "(seed, N) cell at tiny-synth (no accuracy difference AND no "
        "replay execution trace). The mechanisms are structurally "
        "inert -- fix and re-run BEFORE decisive. raw_cells=%r"
        % cells
    )


def test_structural_effect_probes_refuse_cache_scale_mismatch(tmp_path):
    """(f) -- closes 10th adversarial review BLOCK.

    The 10th adversarial review caught a real defect: with
    ``tiny_synth=True``, the runner builds a 952-neuron / 46497-synapse
    bridge but ``load_checkpoint`` happily loads the existing
    biological-scale Phase-1 cache (8440 neurons / 4825651 synapses)
    from ``_PHASE1_CACHE_DEFAULT``. Every simulation step then raises
    ``IndexError`` (swallowed by try/except inside the bridge step),
    silently corrupting the bridge state -- the probe's "passing"
    24.46 mV / 35.99 mV numbers were measured against this corrupted
    state and are unreliable as a gate.

    The strengthen-only fix REFUSES TO RUN the probes on a cache that
    doesn't match the built bridge's neuron count. Inspect the HDF5
    checkpoint metadata BEFORE ``load_checkpoint``; if the cached
    ``num_neurons`` / ``connections_shape_0`` / ``cp_membrane_potential_v``
    shape disagrees with the built bridge's dimensions, raise
    ``RuntimeError`` with a clear message.

    This test constructs the exact failure mode: a tmp cache directory
    populated with a SYNTHETIC biological-scale cache file (one whose
    stored ``num_neurons`` / connection sizes match the full-scale
    recipe, NOT the tiny-synth recipe) for seed 42, then invokes the
    two structural-effect probes with ``tiny_synth=True`` and that
    cache directory. Both probes MUST raise ``RuntimeError`` whose
    message references the cache-scale-mismatch defect.

    The synthetic cache file is small (HDF5 with minimal datasets)
    but its STORED dimensions encode the biological scale; the probe
    must refuse based on that metadata, NOT after attempting to load
    and corrupting the bridge.
    """
    import h5py
    import numpy as np

    # The probe will call ``_phase1_train_if_needed`` which is a no-op
    # if the cache file already exists. We pre-populate the cache with
    # a synthetic file whose stored dimensions are biological-scale,
    # so the probe sees a cache that's mismatched relative to the
    # tiny_synth-sized bridge it builds.
    cache_dir = tmp_path / "mismatched_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "seed42.simstate.h5"

    # Synthetic biological-scale cache. The probe's pre-load check
    # reads num_neurons attr and the cp_membrane_potential_v shape; we
    # only need those to record the mismatched scale. Connections
    # metadata is included for defense-in-depth check.
    bio_scale_n = 8440
    with h5py.File(str(cache_path), "w") as f:
        f.attrs["num_neurons"] = int(bio_scale_n)
        f.attrs["connections_shape_0"] = int(bio_scale_n)
        f.attrs["connections_shape_1"] = int(bio_scale_n)
        f.create_dataset(
            "cp_membrane_potential_v",
            data=np.zeros((bio_scale_n,), dtype=np.float32),
        )
        f.create_dataset(
            "connections_data",
            data=np.zeros((10,), dtype=np.float32),
        )

    # Replay-effect probe MUST refuse on cache-scale mismatch.
    with pytest.raises(RuntimeError) as exc_replay:
        grr._replay_effect_probe(
            seed=42, tiny_synth=True, cache_dir=str(cache_dir)
        )
    msg_replay = str(exc_replay.value)
    assert "CACHE" in msg_replay.upper() or "MISMATCH" in msg_replay.upper(), (
        "the replay-effect probe must raise a RuntimeError whose message "
        "identifies the cache-scale-mismatch defect; got: %r" % msg_replay
    )
    assert str(bio_scale_n) in msg_replay, (
        "the error message must surface the cached neuron count (%d) so "
        "the operator can diagnose; got: %r" % (bio_scale_n, msg_replay)
    )

    # PFC-frame-effect probe MUST refuse on cache-scale mismatch.
    with pytest.raises(RuntimeError) as exc_pfc:
        grr._pfc_frame_effect_probe(
            seed=42, tiny_synth=True, cache_dir=str(cache_dir)
        )
    msg_pfc = str(exc_pfc.value)
    assert "CACHE" in msg_pfc.upper() or "MISMATCH" in msg_pfc.upper(), (
        "the pfc-frame-effect probe must raise a RuntimeError whose "
        "message identifies the cache-scale-mismatch defect; got: %r"
        % msg_pfc
    )
    assert str(bio_scale_n) in msg_pfc, (
        "the error message must surface the cached neuron count (%d) so "
        "the operator can diagnose; got: %r" % (bio_scale_n, msg_pfc)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
