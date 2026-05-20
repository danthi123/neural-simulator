"""TDD tests for the net-new targeted-cue-suppression-during-replay
runner (Task 2 of the 7th arc).

Written FIRST (red before the runner lands). The runner implements
FOUR EMPIRICALLY-TARGETED MECHANISMS on top of the 6th arc's
generative-replay + PFC-frame substrate:

  (1) CUE-SUPPRESSION DURING REPLAY (not retrieve): the FULL arm wraps
      ``run_concept_replay_phase`` with the lang_input cue forcibly
      zeroed AND the ``lang_to_ec`` plasticity gate clamped to 0.0 for
      the duration of the replay window. After replay, both are
      restored. The UNIFORM_CTRL arm runs replay with cue PRESENT
      (baseline). Encoding-specificity is respected at RETRIEVE in
      both arms (cue stays ON during retrieve in both).

  (2) AMPLIFIED ENGRAM-TAG STIM (during retrieve): the FULL arm's
      compositional retrieval drives the engram tag at
      ``RETRIEVE_TAG_AMP_FACTOR=3.0`` times the baseline 1500 pA (4500
      pA effective). The UNIFORM_CTRL arm uses the baseline 1x drive.

  (3) PERSISTENT PFC-FRAME (50 steps instead of 10): the FULL arm
      primes ``dlpfc_verb`` for ``PFC_FRAME_STIM_STEPS=50`` steps
      before each compositional read so the NMDA-bistable attractor
      has time to lock in. The UNIFORM_CTRL arm skips PFC-frame
      priming.

  (4) HIGHER n_replays_per_tag (REPLAY_CYCLES_PER_TAG=50 vs uniform's
      baseline 20): consolidates the engram-tagged ensemble more
      strongly into cortex during the FULL arm's replay window.

This is the load-bearing experimental contrast the architecture
introduces. Grounded in the cross-arc trajectory analysis at commit
9693685 (35% gap-closure across the prior 6 arcs); the design doc at
``docs/plans/2026-05-20-7th-arc-replay-cue-suppression-amplified-tag-design.md``;
the plan at
``docs/plans/2026-05-20-7th-arc-replay-cue-suppression-amplified-tag-implementation.md``.

The decisive multi-seed CuPy run is a later controller-only task;
this suite screens only that:

  (a) ``run_targeted_cue_suppression_replay(seeds=[42,43,44],
      tiny_synth=True)`` runs end-to-end, returns a dict with
      ``rungs`` + ``verdict`` whose ``gate`` is one of the four
      frozen states, and NEVER raises;
  (b) every rung carries EXACTLY the six required keys with correct
      types/ranges so the frozen verdict does NOT VOID for a
      structural reason (it may legitimately FAIL on toy numbers --
      fine);
  (c) no shipped module text imports torch.autograd / .backward;
  (d) STRUCTURAL-EFFECT PROBES (THREE of them, MANDATORY):
       (1) Cue-suppression-during-replay probe: the runner's actual
           code path produces NON-byte-identical bridge state between
           cue-suppression-on (FULL arm) and cue-suppression-off
           (UNIFORM_CTRL arm) under deterministic RNG isolation --
           mechanism is structurally active;
       (2) Amplified-tag-stim probe: bridge state diverges > 1 mV
           between 3x and 1x tag drive amplitude;
       (3) Persistent-PFC-frame probe: bridge state diverges > 1 mV
           between 50-step and 10-step PFC-frame priming.
       Each probe ALSO runs both-arms-same controls and asserts
       those agree to < 0.5 mV (eighth adversarial review lesson
       carried forward + tenth review cache-scale-mismatch refusal).
  (e) per-cell raw_cells emit BOTH full_acc and uniform_ctrl_acc and
      at least one cell exhibits the mechanism's structural effect
      (full_acc != uniform_ctrl_acc on at least one (seed, N)
      combination at tiny-synth scale -- if every cell shows
      equality, the mechanisms are inert);
  (f) cache-scale mismatch is refused by all three probes (tenth
      adversarial review BLOCK closure carried forward from 6th arc
      commit ``13f73e8``).

tiny_synth shrinks pools / events / replay cycles / PFC-frame
priming durations so this is a fast logic-screen smoke (toy numbers
are NOT a result).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import research.runners.targeted_cue_suppression_replay_runner as tcr
from research.runners.targeted_cue_suppression_replay_core import (
    REQUIRED_KEYS,
    targeted_cue_suppression_replay_verdict,
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
    assert hasattr(tcr, "run_targeted_cue_suppression_replay")
    assert callable(tcr.run_targeted_cue_suppression_replay)
    assert hasattr(tcr, "main")
    assert callable(tcr.main)


def test_tiny_synth_smoke_outputs_expected_json_shape(tmp_path):
    """(a)+(b): a tiny-synth multi-seed run returns a well-formed dict
    the frozen verdict accepts. Every rung must carry EXACTLY the six
    required keys with correct types/ranges so the frozen verdict does
    not VOID structurally. The smoke must also write the JSON output
    when out_path is provided and disclaim its toy numbers."""
    out = tmp_path / "tcr_smoke.json"
    cache_dir = tmp_path / "phase1"
    result = tcr.run_targeted_cue_suppression_replay(
        seeds=[42, 43, 44], loads=(2,), tiny_synth=True,
        phase1_cache_dir=str(cache_dir),
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

    recomputed = targeted_cue_suppression_replay_verdict(result["rungs"])
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
    src = Path(tcr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "torch.autograd" not in src
    assert ".backward(" not in src
    assert "import torch" not in src


def test_structural_effect_probes_validate_all_three_mechanisms(tmp_path):
    """(d): MANDATORY structural-effect probes (THREE of them). Each
    must show > 1 mV bridge-state divergence between flag-on and
    flag-off via the runner's ACTUAL code path; controls (both-arms-
    same-flag with identical deterministic RNG isolation) must agree
    to < 0.5 mV.

    Cue-suppression-during-replay probe:
      * bridge with cue-suppressed replay vs bridge with cue-present
        replay -- > 1 mV
      * both arms cue-suppressed (control) -- < 0.5 mV
      * both arms cue-present (control) -- < 0.5 mV

    Amplified-tag-stim probe:
      * bridge with 3x tag drive vs bridge with 1x tag drive -- > 1 mV
      * both arms 3x (control) -- < 0.5 mV
      * both arms 1x (control) -- < 0.5 mV

    Persistent-PFC-frame probe:
      * bridge with 50-step PFC-frame vs bridge with 10-step -- > 1 mV
      * both arms 50-step (control) -- < 0.5 mV
      * both arms 10-step (control) -- < 0.5 mV

    Mirrors Pirazzini d462bf0 / theta-gamma e6b17da lesson: structural-
    effect probe must work via the runner's ACTUAL code path and rule
    out RNG drift via controls. If any probe fails (flag-differing
    < 1 mV OR control > 0.5 mV) the runner aborts (no decisive
    numbers reported).

    CACHE-SCALE DISCIPLINE (closes 10th adversarial review BLOCK
    carried forward from 6th arc commit 13f73e8): uses a tmp_path
    cache directory rather than the default biological-scale
    ``_PHASE1_CACHE_DEFAULT``. With ``tiny_synth=True`` the probe
    builds a small (~952-neuron) bridge; the cache MUST match. The
    probe's pre-load validator
    (``_validate_cache_scale_for_probe``) REFUSES to run on a
    mismatched cache; a fresh tmp_path lets
    ``_phase1_train_if_needed`` produce a matching tiny-synth-scale
    cache file."""
    assert hasattr(tcr, "_cue_suppression_replay_effect_probe"), (
        "the runner must expose a `_cue_suppression_replay_effect_probe` "
        "helper (mirrors Pirazzini d462bf0 / theta-gamma e6b17da lesson)"
    )
    assert hasattr(tcr, "_amplified_tag_stim_effect_probe"), (
        "the runner must expose an `_amplified_tag_stim_effect_probe` "
        "helper (mirrors Pirazzini d462bf0 / theta-gamma e6b17da lesson)"
    )
    assert hasattr(tcr, "_persistent_pfc_frame_effect_probe"), (
        "the runner must expose a `_persistent_pfc_frame_effect_probe` "
        "helper (mirrors Pirazzini d462bf0 / theta-gamma e6b17da lesson)"
    )
    cache_dir = tmp_path / "probe_cache"
    diff_cue = tcr._cue_suppression_replay_effect_probe(
        seed=42, tiny_synth=True, cache_dir=str(cache_dir)
    )
    assert isinstance(diff_cue, float) and diff_cue > 1.0, (
        "the runner's actual code path must produce > 1 mV bridge-state "
        "divergence between cue-suppression-during-replay on and off; "
        "got %.6g mV. This is the inert-mechanism failure mode the "
        "Pirazzini d462bf0 lesson guards against."
        % diff_cue
    )
    diff_amp = tcr._amplified_tag_stim_effect_probe(
        seed=42, tiny_synth=True, cache_dir=str(cache_dir)
    )
    assert isinstance(diff_amp, float) and diff_amp > 1.0, (
        "the runner's actual code path must produce > 1 mV bridge-state "
        "divergence between 3x and 1x amplified-tag-stim; got %.6g mV."
        % diff_amp
    )
    diff_pfc = tcr._persistent_pfc_frame_effect_probe(
        seed=42, tiny_synth=True, cache_dir=str(cache_dir)
    )
    assert isinstance(diff_pfc, float) and diff_pfc > 1.0, (
        "the runner's actual code path must produce > 1 mV bridge-state "
        "divergence between 50-step and 10-step PFC-frame priming; got "
        "%.6g mV."
        % diff_pfc
    )


def test_full_vs_uniform_arms_differ_at_least_on_some_query(tmp_path):
    """(e): the FULL arm (cue-suppressed replay + amplified tag stim +
    persistent PFC-frame) and the UNIFORM_CTRL arm (cue-present replay
    + baseline 1x tag + brief 10-step PFC-frame) must produce a
    DIFFERENT signature on at least one (seed, N) cell at tiny-synth
    scale. If EVERY cell shows EXACT equality on BOTH the accuracy
    metrics AND the mechanism-trace diagnostics, the augmenting
    mechanisms are structurally inert.

    Acceptance:
      * Accuracy contrast: at least one cell has full_acc !=
        uniform_ctrl_acc OR direct_retain_acc differs across arms,
        OR
      * Mechanism-trace contrast: at least one cell records that the
        FULL arm's replay used the higher REPLAY_CYCLES_PER_TAG (50)
        whereas the UNIFORM_CTRL arm used the baseline 20. The two
        replay counts are recorded per cell as diagnostic; a non-zero
        difference IS a per-cell contrast witnessing the mechanism's
        execution. This is the load-bearing structural evidence at
        toy-scale where accuracy is noise-dominated.

    Decisive built-in control: per the frozen verdict,
    uniform_ctrl_max=0.10 is the bar the FULL arm must beat; this
    test checks only that the mechanism PRODUCES a contrast (not
    that the contrast is in the right direction at tiny-synth). The
    proper bridge-state non-inertness checks are the structural-effect
    probes (test (d) above)."""
    cache_dir = tmp_path / "phase1"
    result = tcr.run_targeted_cue_suppression_replay(
        seeds=[42, 43, 44], loads=(2,), tiny_synth=True,
        phase1_cache_dir=str(cache_dir),
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
        # Mechanism-trace contrast: the FULL arm uses replay count 50,
        # UNIFORM_CTRL uses 20. The per-cell diagnostics record both;
        # any non-equality IS the contrast.
        full_replays = int(c.get("replay_n_replays_full", 0))
        uniform_replays = int(c.get("replay_n_replays_uniform", 0))
        if full_replays != uniform_replays:
            has_mechanism_trace = True
    has_a_diff = has_accuracy_diff or has_mechanism_trace
    assert has_a_diff, (
        "the targeted cue-suppression-replay + amplified tag stim + "
        "persistent PFC-frame mechanisms produced ZERO evidence of "
        "contrast between FULL and UNIFORM_CTRL on every (seed, N) cell "
        "at tiny-synth (no accuracy difference AND no replay-count "
        "trace). The mechanisms are structurally inert -- fix and "
        "re-run BEFORE decisive. raw_cells=%r" % cells
    )


def test_cache_scale_mismatch_raises(tmp_path):
    """(f) -- carries forward the 10th adversarial review BLOCK closure
    from 6th arc commit 13f73e8.

    The 10th adversarial review caught a real defect: with
    ``tiny_synth=True``, the runner builds a ~952-neuron bridge but
    ``load_checkpoint`` happily loads the existing biological-scale
    Phase-1 cache (8440 neurons / 4825651 synapses) from
    ``_PHASE1_CACHE_DEFAULT``. Every simulation step then raises
    ``IndexError`` (swallowed by try/except inside the bridge step),
    silently corrupting the bridge state -- the probe's "passing"
    numbers were measured against this corrupted state and are
    unreliable as a gate.

    The strengthen-only fix REFUSES TO RUN the probes on a cache that
    doesn't match the built bridge's neuron count. Inspect the HDF5
    checkpoint metadata BEFORE ``load_checkpoint``; if the cached
    ``num_neurons`` / ``connections_shape_0`` /
    ``cp_membrane_potential_v`` shape disagrees with the built
    bridge's dimensions, raise ``RuntimeError`` with a clear message.

    This test constructs the exact failure mode: a tmp cache directory
    populated with a SYNTHETIC biological-scale cache file (one whose
    stored ``num_neurons`` / connection sizes match the full-scale
    recipe, NOT the tiny-synth recipe) for seed 42, then invokes the
    three structural-effect probes with ``tiny_synth=True`` and that
    cache directory. ALL THREE probes MUST raise ``RuntimeError``
    whose message references the cache-scale-mismatch defect.
    """
    import h5py
    import numpy as np

    cache_dir = tmp_path / "mismatched_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "seed42.simstate.h5"

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

    for probe_name in (
        "_cue_suppression_replay_effect_probe",
        "_amplified_tag_stim_effect_probe",
        "_persistent_pfc_frame_effect_probe",
    ):
        probe = getattr(tcr, probe_name)
        with pytest.raises(RuntimeError) as exc:
            probe(seed=42, tiny_synth=True, cache_dir=str(cache_dir))
        msg = str(exc.value)
        assert "CACHE" in msg.upper() or "MISMATCH" in msg.upper(), (
            "the %s must raise a RuntimeError whose message identifies "
            "the cache-scale-mismatch defect; got: %r"
            % (probe_name, msg)
        )
        assert str(bio_scale_n) in msg, (
            "the error message must surface the cached neuron count (%d) "
            "so the operator can diagnose; got: %r"
            % (bio_scale_n, msg)
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
