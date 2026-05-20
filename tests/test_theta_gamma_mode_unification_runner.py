"""TDD tests for the net-new theta-gamma mode-unification runner (Task 2).

Written FIRST (red before the runner lands). The runner implements
CUE SUPPRESSION DURING RETRIEVE: a three-phase theta cycle
(encode/gap/retrieve) where the FULL arm suppresses the cued-noun's
diffuse lang_input drive during the retrieve window so the engram
tag's selective bound-adj drive can dominate; the UNIFORM_CTRL arm
keeps the cue ON during retrieve as the decisive built-in control.

This is the load-bearing experimental contrast the architecture
introduces. It is grounded in the localisation finding (commit
110f7cd): the cued-noun's diffuse lang_input drive dominates the
engram tag's selective bound-adj drive at deployment.

The decisive multi-seed CuPy run is a later controller-only task;
this suite screens only that:

  (a) ``run_theta_gamma_mode_unification(seeds=[42,43,44],
      tiny_synth=True)`` runs end-to-end, returns a dict with
      ``rungs`` + ``verdict`` whose ``gate`` is one of the four
      frozen states, and NEVER raises;
  (b) every rung carries EXACTLY the six required keys with correct
      types/ranges so the frozen verdict does NOT VOID for a
      structural reason (it may legitimately FAIL on toy numbers --
      fine);
  (c) no shipped module text imports torch.autograd / .backward;
  (d) STRUCTURAL-EFFECT PROBE: the runner's actual code path
      produces NON-byte-identical bridge state between
      suppress_cue_during_retrieve=True (FULL arm) and
      suppress_cue_during_retrieve=False (UNIFORM_CTRL arm) -- the
      mechanism is structurally active (mirrors Pirazzini d462bf0
      lesson: must work via the runner's ACTUAL code path);
  (e) per-cell raw_cells emit BOTH full_acc and uniform_ctrl_acc and
      at least one cell exhibits the mechanism's structural effect
      (full_acc != uniform_ctrl_acc on at least one (seed, N)
      combination at tiny-synth scale -- if every cell shows
      equality, the mechanism is inert).

tiny_synth shrinks pools / events / phase-block lengths so this is
a fast logic-screen smoke (toy numbers are NOT a result).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import research.runners.theta_gamma_mode_unification_runner as tgr
from research.runners.theta_gamma_mode_unification_core import (
    REQUIRED_KEYS,
    theta_gamma_mode_unification_verdict,
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
    assert hasattr(tgr, "run_theta_gamma_mode_unification")
    assert callable(tgr.run_theta_gamma_mode_unification)
    assert hasattr(tgr, "main")
    assert callable(tgr.main)


def test_tiny_synth_smoke_outputs_expected_json_shape(tmp_path):
    """(a)+(b): a tiny-synth multi-seed run returns a well-formed dict
    the frozen verdict accepts. Every rung must carry EXACTLY the six
    required keys with correct types/ranges so the frozen verdict does
    not VOID structurally. The smoke must also write the JSON output
    when out_path is provided and disclaim its toy numbers."""
    out = tmp_path / "tg_smoke.json"
    result = tgr.run_theta_gamma_mode_unification(
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

    recomputed = theta_gamma_mode_unification_verdict(result["rungs"])
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
    src = Path(tgr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "torch.autograd" not in src
    assert ".backward(" not in src
    assert "import torch" not in src


def test_structural_effect_probe_runs_via_runner_actual_code_path():
    """(d): the structural-effect probe MUST be exposed by the runner
    AND must exercise the runner's actual code path (the per-step
    theta-cycle helper) -- NOT a synthetic-bypass probe.

    The probe's contract: build the same substrate twice from the
    SAME seed; run the runner's actual theta-cycle helper once with
    suppress_cue_during_retrieve=True and once with
    suppress_cue_during_retrieve=False at the SAME initial state.
    Compare bridge.cp_membrane_potential_v. Must differ by > 1 mV.

    Mirrors the Pirazzini d462bf0 lesson: the structural-effect
    probe must work via the runner's ACTUAL code path, not a
    synthetic per-step loop. If the probe fails the runner must
    raise (no decisive numbers are reported when the mechanism is
    structurally inert).

    Strengthened per the 8th adversarial review BLOCK: the probe ALSO
    runs CONTROL contrasts (both arms pass the SAME flag with the SAME
    deterministic RNG seed) and asserts those agree to < 0.5 mV. The
    earlier 30.24 mV claim turned out to be RNG-drift -- it reproduced
    under both-True and both-False just like under the flag-differing
    case. The strengthened probe rules that out by construction: if
    either control shows divergence > 0.5 mV the probe raises.
    """
    assert hasattr(tgr, "_structural_effect_probe"), (
        "the runner must expose a `_structural_effect_probe` helper "
        "(mirrors Pirazzini d462bf0 lesson)"
    )
    # The probe must not raise -- if it raises, either (a) the mechanism
    # is structurally inert (flag-differing < 1 mV) or (b) RNG isolation
    # is broken (a control shows > 0.5 mV divergence). Both conditions
    # are no-go.
    diff_mv = tgr._structural_effect_probe(seed=42, tiny_synth=True)
    assert isinstance(diff_mv, float) and diff_mv > 1.0, (
        "the runner's actual code path must produce > 1 mV bridge-state "
        "divergence between suppress_cue_during_retrieve=True and "
        "=False at the SAME initial state; got %.6g mV. This is the "
        "inert-mechanism failure mode the Pirazzini d462bf0 lesson "
        "guards against."
        % diff_mv
    )


def test_structural_effect_probe_controls_pass_at_runner_level():
    """8th adversarial review BLOCK closer: directly assert that the
    probe's CONTROL contrasts hold via the same helper the probe uses.

    The probe internally seeds the active backend's RNG to a fixed
    value before each call to _run_theta_cycle_query. We replicate the
    both-True and both-False contrasts here at the test boundary, using
    the _seed_query_rng / _restore_query_rng helpers the probe exposes,
    and assert each control is < 0.5 mV.

    If this test passes, the eighth adversarial review BLOCK is closed
    in the test surface as well (not just inside the probe): RNG drift
    is NOT the source of any bridge-state divergence reported by the
    probe or the per-cell eval arm.
    """
    from sim.backend import to_host
    import numpy as np

    # The probe-internal seed value (kept private). We mirror it here
    # so this test exercises the same isolation pattern the probe uses.
    PROBE_RNG_SEED = 999

    # Build the substrate the way the probe does.
    cache_dir = tgr._PHASE1_CACHE_DEFAULT
    tgr._phase1_train_if_needed(42, cache_dir, tiny_synth=True)
    cache_path = tgr._phase1_cache_path(cache_dir, 42)

    recipe_dims = tgr._phase1_recipe(True)
    all_words, word_to_idx = tgr._all_words_word_to_idx()
    n_words_for_orthogonal = max(
        tgr._N_WORDS_ORTHOGONAL, len(all_words)
    )
    dims = {
        "n_lang_input": int(recipe_dims["n_lang_input"]),
        "n_per_pool": int(recipe_dims["n_per_pool"]),
        "n_fs_per_pool": int(recipe_dims["n_fs_per_pool"]),
        "sparsity": 0.05,
        "dt_ms": 0.5,
        "n_words_for_orthogonal": int(n_words_for_orthogonal),
    }
    all_pools = tgr._all_pool_regions(enable_adjective=True)
    facts = tgr._unified_compositional_pairs(42, 1)
    cue_noun, _adj = facts[0]
    tag_name = "ep_0"

    ENCODE_RNG_SEED = 31337

    def _one_pair(flag_a: bool, flag_b: bool) -> float:
        bridge_a = tgr._build_bridge_with_phase1_recipe(42, True)
        bridge_b = tgr._build_bridge_with_phase1_recipe(42, True)
        bridge_a.load_checkpoint(str(cache_path))
        bridge_b.load_checkpoint(str(cache_path))
        tgr._freeze_phase1_gates(bridge_a)
        tgr._freeze_phase1_gates(bridge_b)

        # Deterministic RNG isolation: identical seed BEFORE _encode_facts
        # on each so the encoded states are byte-identical across arms.
        saved_enc_a = tgr._seed_query_rng(ENCODE_RNG_SEED)
        try:
            tgr._encode_facts(bridge_a, facts, dims, 8)
        finally:
            tgr._restore_query_rng(saved_enc_a)
        saved_enc_b = tgr._seed_query_rng(ENCODE_RNG_SEED)
        try:
            tgr._encode_facts(bridge_b, facts, dims, 8)
        finally:
            tgr._restore_query_rng(saved_enc_b)

        saved_a = tgr._seed_query_rng(PROBE_RNG_SEED)
        try:
            tgr._run_theta_cycle_query(
                bridge_a, cue_word=cue_noun, tag_name=tag_name,
                dims=dims,
                suppress_cue_during_retrieve=flag_a,
                tiny_synth=True,
                word_to_idx=word_to_idx,
                all_pools=all_pools,
            )
        finally:
            tgr._restore_query_rng(saved_a)
        saved_b = tgr._seed_query_rng(PROBE_RNG_SEED)
        try:
            tgr._run_theta_cycle_query(
                bridge_b, cue_word=cue_noun, tag_name=tag_name,
                dims=dims,
                suppress_cue_during_retrieve=flag_b,
                tiny_synth=True,
                word_to_idx=word_to_idx,
                all_pools=all_pools,
            )
        finally:
            tgr._restore_query_rng(saved_b)
        v_a = to_host(bridge_a.cp_membrane_potential_v)
        v_b = to_host(bridge_b.cp_membrane_potential_v)
        return float(
            np.max(np.abs(np.asarray(v_a) - np.asarray(v_b)))
        )

    diff_both_true = _one_pair(True, True)
    diff_both_false = _one_pair(False, False)
    diff_flag_diff = _one_pair(True, False)

    assert diff_both_true < 0.5, (
        "8th adversarial review BLOCK: with suppress=True on BOTH bridges "
        "and the SAME deterministic RNG seed, the two bridges must agree "
        "(div < 0.5 mV) -- RNG isolation is what closes the prior 30.24 "
        "mV artefact. Got %.6g mV." % diff_both_true
    )
    assert diff_both_false < 0.5, (
        "8th adversarial review BLOCK: with suppress=False on BOTH bridges "
        "and the SAME deterministic RNG seed, the two bridges must agree "
        "(div < 0.5 mV) -- RNG isolation is what closes the prior 30.24 "
        "mV artefact. Got %.6g mV." % diff_both_false
    )
    assert diff_flag_diff > 1.0, (
        "with suppress=True/False (flag-differing) and the SAME "
        "deterministic RNG seed, the mechanism MUST produce > 1 mV "
        "divergence -- this is the genuine cue-suppression effect "
        "(NOT RNG drift, since the controls above passed). Got %.6g mV."
        % diff_flag_diff
    )


def test_full_vs_uniform_arms_differ_at_least_on_some_query():
    """(e): the FULL arm (cue suppressed during retrieve) and the
    UNIFORM_CTRL arm (cue stays ON during retrieve) must produce a
    DIFFERENT result on at least one (seed, N) cell at tiny-synth
    scale. If full_acc == uniform_ctrl_acc on EVERY cell, the
    mechanism is structurally inert.

    Decisive built-in control: per the frozen verdict,
    uniform_ctrl_max=0.10 is the bar the FULL arm must beat; this
    test checks only that the mechanism PRODUCES a difference (not
    that the difference is in the right direction at tiny-synth)."""
    result = tgr.run_theta_gamma_mode_unification(
        seeds=[42, 43, 44], loads=(2,), tiny_synth=True,
    )
    cells = result.get("raw_cells", [])
    assert isinstance(cells, list) and len(cells) >= 1
    has_a_diff = False
    for c in cells:
        full = float(c.get("full_acc", 0.0))
        uniform = float(c.get("uniform_ctrl_acc", 0.0))
        if abs(full - uniform) > 1e-9:
            has_a_diff = True
            break
    assert has_a_diff, (
        "the cue-suppression-during-retrieve mechanism produced "
        "ZERO difference between FULL and UNIFORM_CTRL on every "
        "(seed, N) cell at tiny-synth. The mechanism is structurally "
        "inert -- fix and re-run BEFORE decisive. raw_cells=%r"
        % cells
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
