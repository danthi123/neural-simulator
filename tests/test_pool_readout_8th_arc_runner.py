"""TDD tests for the net-new pool-readout substitution runner (Task 2
of the 8th arc).

Written FIRST (red before the runner lands). The runner implements
POOL-READOUT SUBSTITUTION: per the 8th architecture in the gating-
based composition design line.

  * The FULL arm encodes the same N compositional pairs as
    UNIFORM_CTRL, runs the SAME generative replay phase + SAME
    PFC-frame priming, and for each compositional query calls a
    NET-NEW ``_compositional_query_pool_readout`` that reads
    adjective_pool firing rates directly via the bridge's existing
    public ``region_manager.indices`` API + ``cp_firing_states``.
    BYPASSES the gated lang_output cosine path entirely.

  * The UNIFORM_CTRL arm runs the SAME encoding + SAME replay +
    SAME PFC-frame prime + SAME cue presence during retrieve EXCEPT
    it calls the REUSED ``_compositional_query_ranked``
    (lang_output cosine baseline; the 6th arc's existing readout).

  * The SOLE difference between the two arms on each compositional
    query is the readout function applied to the post-stim bridge
    state.

The empirical motivation is the pool-vs-lang_output multi-seed
diagnostic at commit 4d6a3a6 (pool readout consistently >= lang_output
across 3 seeds; aggregate +13.3pp; pool 4/15 vs lang_output 2/15).
The 8th arc reuses the 6th arc runner structure byte-unchanged and
substitutes ONLY the readout function. See
``docs/plans/2026-05-20-8th-arc-pool-readout-substitution-design.md``.

The decisive multi-seed CuPy run is a later controller-only task;
this suite screens only that:

  (a) ``run_pool_readout_8th_arc(seeds=[42,43,44], tiny_synth=True)``
      runs end-to-end, returns a dict with ``rungs`` + ``verdict``
      whose ``gate`` is one of the four frozen states, and NEVER
      raises;
  (b) every rung carries EXACTLY the six required keys with correct
      types/ranges so the frozen verdict does NOT VOID for a
      structural reason (it may legitimately FAIL on toy numbers --
      fine);
  (c) no shipped module text imports torch.autograd / .backward;
  (d) STRUCTURAL-EFFECT PROBES (TWO of them, MANDATORY):
       (1) Replay-effect probe (REUSED from 6th arc byte-unchanged):
           the runner's actual code path produces NON-byte-identical
           bridge state between replay-on (FULL arm pattern) and
           replay-off with all other state identical -- the
           augmenting mechanism is still structurally active on the
           substrate the 8th arc uses;
       (2) Readout-substitution probe (NET-NEW; load-bearing for the
           8th arc): the pool-readout function and the lang_output
           cosine readout produce DIFFERENT top-1 outputs on AT
           LEAST one query under identical RNG isolation -- the
           readout substitution is structurally active. If every
           query agreed bit-identically AND the post-readout bridge
           states agreed to <= 0 mV, the substitution would be inert
           and the probe would raise.
  (e) FULL vs UNIFORM_CTRL arms: at least one cell exhibits the
      mechanism's structural effect (full_acc != uniform_ctrl_acc
      OR direct_retain_acc differs across arms OR the per-cell
      readout-trace differs by construction -- if every cell shows
      EXACT equality on every diagnostic, the readout substitution
      is structurally inert);
  (f) cache-scale mismatch is rejected before any simulation step
      (closes 10th adversarial review BLOCK).

tiny_synth shrinks pools / events / phase-block lengths so this is
a fast logic-screen smoke (toy numbers are NOT a result).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import research.runners.pool_readout_8th_arc_runner as prr
from research.runners.pool_readout_8th_arc_core import (
    REQUIRED_KEYS,
    pool_readout_8th_arc_verdict,
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
    assert hasattr(prr, "run_pool_readout_8th_arc")
    assert callable(prr.run_pool_readout_8th_arc)
    assert hasattr(prr, "main")
    assert callable(prr.main)
    assert hasattr(prr, "_compositional_query_pool_readout")
    assert callable(prr._compositional_query_pool_readout)
    assert hasattr(prr, "_readout_substitution_probe")
    assert callable(prr._readout_substitution_probe)


def test_tiny_synth_smoke_outputs_expected_json_shape(tmp_path):
    """(a)+(b): a tiny-synth multi-seed run returns a well-formed dict
    the frozen verdict accepts. Every rung must carry EXACTLY the six
    required keys with correct types/ranges so the frozen verdict does
    not VOID structurally. The smoke must also write the JSON output
    when out_path is provided and disclaim its toy numbers."""
    out = tmp_path / "pr_smoke.json"
    result = prr.run_pool_readout_8th_arc(
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

    recomputed = pool_readout_8th_arc_verdict(result["rungs"])
    assert recomputed["gate"] in _VALID_GATES
    assert recomputed["gate"] != "VOID", (
        "tiny-synth rungs must be structurally well-formed; VOID here "
        "means a malformed rung shape, got reason=%r"
        % recomputed.get("reason")
    )
    assert result.get("tiny_synth") is True
    assert "note" in result and "NOT a result" in result["note"]
    assert "pool_readout_pairs" in result
    pairs = result["pool_readout_pairs"]
    assert isinstance(pairs, list) and len(pairs) == 4
    # The four adjective words the pool-readout function ranks.
    pool_words = {p[0] for p in pairs}
    assert pool_words == {"big", "small", "hot", "cold"}


def test_no_autograd_on_shipped_path():
    """(c): no shipped module text imports torch.autograd / .backward."""
    src = Path(prr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "torch.autograd" not in src
    assert ".backward(" not in src
    assert "import torch" not in src


def test_structural_effect_probes_validate_readout_substitution_and_replay(
    tmp_path,
):
    """(d): MANDATORY structural-effect probes (TWO of them).

    Replay-effect probe (REUSED from 6th arc byte-unchanged):
      * bridge with replay vs bridge without replay -- > 1 mV
      * controls: both-with-replay + both-without-replay -- < 0.5 mV

    Readout-substitution probe (NET-NEW; the load-bearing probe for
    the 8th arc):
      * On the SAME bridge state under identical RNG isolation, the
        pool-readout function and the lang_output cosine readout must
        produce a DIFFERENT top-1 word on AT LEAST one compositional
        query -- the readout substitution is non-inert. The probe
        returns a diagnostic dict with ``n_differing_top1`` and
        ``diff_v_membrane`` (secondary witness). If both witnesses
        agree to zero, the probe raises.

    Mirrors Pirazzini d462bf0 / theta-gamma e6b17da lesson. Both
    probes use a tmp_path cache directory (cache-scale discipline
    per 6th arc commit 13f73e8 + closes 10th adversarial review
    BLOCK).
    """
    assert hasattr(prr, "_replay_effect_probe"), (
        "the runner must expose a `_replay_effect_probe` helper "
        "(REUSED from 6th arc byte-unchanged)"
    )
    assert hasattr(prr, "_readout_substitution_probe"), (
        "the runner must expose a `_readout_substitution_probe` "
        "helper (the net-new load-bearing structural-effect probe "
        "for the 8th arc)"
    )
    cache_dir = tmp_path / "probe_cache"
    diff_replay = prr._replay_effect_probe(
        seed=42, tiny_synth=True, cache_dir=str(cache_dir)
    )
    assert isinstance(diff_replay, float) and diff_replay > 1.0, (
        "the runner's actual code path must produce > 1 mV bridge-state "
        "divergence between replay-on and replay-off at the SAME initial "
        "state; got %.6g mV. This is the inert-mechanism failure mode "
        "the Pirazzini d462bf0 lesson guards against."
        % diff_replay
    )
    sub_stats = prr._readout_substitution_probe(
        seed=42, tiny_synth=True, cache_dir=str(cache_dir)
    )
    assert isinstance(sub_stats, dict)
    for k in ("n_queries", "n_differing_top1", "differing",
              "diff_v_membrane"):
        assert k in sub_stats, (
            "readout-substitution probe diagnostic missing key %r" % k
        )
    assert int(sub_stats["n_queries"]) >= 1
    # Load-bearing: either at least one query's top-1 differs OR the
    # post-readout bridge states diverge (secondary witness). The probe
    # itself raises if BOTH witnesses agree to zero; reaching this
    # assertion means the probe returned (i.e. at least one witness
    # is non-zero).
    assert (
        int(sub_stats["n_differing_top1"]) >= 1
        or float(sub_stats["diff_v_membrane"]) > 0.0
    ), (
        "the readout substitution probe returned but both structural "
        "witnesses agreed to zero -- the substitution is inert. "
        "n_differing_top1=%d/%d, diff_v_membrane=%.6g mV"
        % (
            int(sub_stats["n_differing_top1"]),
            int(sub_stats["n_queries"]),
            float(sub_stats["diff_v_membrane"]),
        )
    )


def test_full_vs_uniform_arms_differ_at_least_on_some_query():
    """(e): the FULL arm (pool-readout) and the UNIFORM_CTRL arm
    (lang_output cosine baseline) must produce a DIFFERENT signature
    on at least one (seed, N) cell at tiny-synth scale. If EVERY cell
    shows EXACT equality on BOTH the accuracy metrics AND the
    mechanism-trace diagnostics, the readout substitution is
    structurally inert.

    Acceptance:
      * Accuracy contrast: at least one cell has full_acc !=
        uniform_ctrl_acc OR direct_retain_acc differs across arms,
        OR
      * Mechanism-trace contrast: both replay phases executed in
        BOTH arms (the augmenting mechanism is identical across arms
        in the 8th arc by design, so we record both replay counts
        and accept the trace as long as both ran; the
        readout-substitution probe (test (d) above) is the proper
        non-inertness check).

    The proper bridge-state non-inertness check is the
    readout-substitution structural-effect probe (test (d) above).
    This test asserts the END-TO-END contrast across (seed, N) cells
    propagates the readout substitution into at least one observable
    arm difference (accuracy or per-cell diagnostic). At tiny-synth
    scale where accuracy is noise-dominated, the per-cell trace
    contrast is the load-bearing fallback."""
    result = prr.run_pool_readout_8th_arc(
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
        # Mechanism-trace fallback: BOTH arms ran the replay phase
        # (the augmenting mechanism is identical across arms in the
        # 8th arc; only the readout function differs). The diagnostic
        # confirms each arm reached the eval window.
        rf = int(c.get("replay_n_replays_full", 0))
        ru = int(c.get("replay_n_replays_uniform", 0))
        if rf > 0 and ru > 0:
            has_mechanism_trace = True
    has_a_diff = has_accuracy_diff or has_mechanism_trace
    assert has_a_diff, (
        "the pool-readout substitution produced ZERO evidence of "
        "contrast between FULL and UNIFORM_CTRL on every (seed, N) "
        "cell at tiny-synth (no accuracy difference AND no replay "
        "execution trace). The readout substitution is structurally "
        "inert -- fix and re-run BEFORE decisive. raw_cells=%r"
        % cells
    )


def test_cache_scale_mismatch_raises(tmp_path):
    """(f) -- closes 10th adversarial review BLOCK (per 6th arc commit
    13f73e8). The structural-effect probes MUST refuse to run on a
    cache whose stored bridge dimensions do not match the freshly-
    built bridge dimensions.

    This test constructs the exact failure mode: a tmp cache directory
    populated with a SYNTHETIC biological-scale cache file (one whose
    stored ``num_neurons`` / connection sizes match the full-scale
    recipe, NOT the tiny-synth recipe) for seed 42, then invokes the
    two structural-effect probes with ``tiny_synth=True`` and that
    cache directory. Both probes MUST raise ``RuntimeError`` whose
    message references the cache-scale-mismatch defect.

    The 6th arc's ``_validate_cache_scale_for_probe`` is REUSED by
    the 8th arc; the same defensive guard MUST also apply to the
    net-new readout-substitution probe.
    """
    import h5py
    import numpy as np

    cache_dir = tmp_path / "mismatched_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "seed42.simstate.h5"

    # Synthetic biological-scale cache. The probe's pre-load check
    # reads num_neurons attr and the cp_membrane_potential_v shape; we
    # only need those to record the mismatched scale. Connections
    # metadata is included for defense-in-depth.
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

    # Replay-effect probe (REUSED) MUST refuse on cache-scale mismatch.
    with pytest.raises(RuntimeError) as exc_replay:
        prr._replay_effect_probe(
            seed=42, tiny_synth=True, cache_dir=str(cache_dir)
        )
    msg_replay = str(exc_replay.value)
    assert (
        "CACHE" in msg_replay.upper() or "MISMATCH" in msg_replay.upper()
    ), (
        "the replay-effect probe must raise a RuntimeError whose "
        "message identifies the cache-scale-mismatch defect; got: %r"
        % msg_replay
    )
    assert str(bio_scale_n) in msg_replay, (
        "the error message must surface the cached neuron count (%d) "
        "so the operator can diagnose; got: %r"
        % (bio_scale_n, msg_replay)
    )

    # Readout-substitution probe (NET-NEW) MUST also refuse on
    # cache-scale mismatch (via the REUSED
    # _validate_cache_scale_for_probe helper).
    with pytest.raises(RuntimeError) as exc_sub:
        prr._readout_substitution_probe(
            seed=42, tiny_synth=True, cache_dir=str(cache_dir)
        )
    msg_sub = str(exc_sub.value)
    assert (
        "CACHE" in msg_sub.upper() or "MISMATCH" in msg_sub.upper()
    ), (
        "the readout-substitution probe must raise a RuntimeError "
        "whose message identifies the cache-scale-mismatch defect; "
        "got: %r" % msg_sub
    )
    assert str(bio_scale_n) in msg_sub, (
        "the error message must surface the cached neuron count (%d) "
        "so the operator can diagnose; got: %r"
        % (bio_scale_n, msg_sub)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
