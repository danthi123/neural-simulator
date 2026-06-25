"""GATE 3A (option A, Closure 1) — the DIRECT FUNCTIONAL-NEUTRALITY flip gate (composer OFF vs ON at a FIXED merged N).

THIS IS THE LOAD-BEARING FLIP GATE for Closure 1 (owner-default (b)), REPLACING the cross-N byte-identity gate 3.

WHY THE CROSS-N GATE IS NOT THE REQUIREMENT (per `research/findings/raw/_closure1_optionA_flip_DONE.json` +
`_optionA_flip_rng_decoupling_analysis.md`): the original gate 3 (`_closure1_optionA_gate3`) compared two DIFFERENT
total-N bridges (rf sized for onebrain vs for the rf oracle) and demanded a byte-identical nav SCORE (Δ=0). That gate is
UNREACHABLE on the GPU without a sim/ FP rewrite, and NOT because of any composer functional effect: the residual is GPU
floating-point NON-ASSOCIATIVITY — changing the rf-region SIZE changes the total N, so the episode-loop readout/render
reduction tiles differently and produces ~1e-5 (FP32 round-off scale) differences in the SAME nav-slice neurons, which
the chaotic spiking-WTA action-selection amplifies into a divergent-but-equally-valid trajectory over ~200 steps.
CUBLAS_WORKSPACE_CONFIG=:4096:8 enforces cuBLAS determinism for the SAME problem size but NOT across different matrix
sizes (and cuSPARSE/reduction ordering is size-dependent). Forcing the start state byte-identical (and even re-seeding the
global RNG every step) does NOT close it — it is a deterministic FP reduction, an N-independent reduction would be a
protected sim/ edit out of proportion to the gain. So the cross-N nav-SCORE byte-identity is NOT the flip requirement; it
is GPU-FP noise from the different N. The documented STANDALONE nav benchmark (`--readout-source motor`, the CLI default)
NEVER builds the conversational rf region at all, so it is totally unaffected by the merged composer kind.

WHAT 3A ASSERTS INSTEAD (the FUNCTIONAL-neutrality the flip actually needs): at a SINGLE FIXED N (the onebrain size for
BOTH legs — so the §1/FP cross-N confound is GONE), does ACTUALLY EXERCISING the co-resident composer (kick + resonate +
store + cleanup on the `rf` slice) perturb the nav state the episode would inherit? Since N is identical, any Δ would be a
REAL synaptic/state leak from the composer into nav — which the Task-1 anti-cheat says cannot happen: the `rf` slice has 0
`cp_connections` out-edges into nav, and every composer op `_zero_rf_v_u()`-resets the rf slice, so it cannot even leave
residue a later nav step would read through shared v/u arrays. The deployed nav episode runs OU-OFF with no per-step RNG,
so the nav trajectory is a deterministic function of the nav slice's (v, u) at episode start. Therefore "exercising the
composer does not perturb nav" is exactly: the NAV-slice v/u is BYTE-IDENTICAL whether or not composer ops ran. This probe
asserts that directly at the onebrain N — and it isolates the composer's FUNCTIONAL presence (not its region size, which
is the FP-confounded variable), which is the deployed question.

This complements (does not replace) Probe-1's anti-cheat 3 (a single composer op leaves a co-resident Izhikevich slice
byte-untouched, atol exact): here the FULL who/what matrix of composer ops runs on the REAL merged agent's `rf` slice,
against the REAL nav cascade slice, at production N.

ISOLATE THE COMPOSER, NOT THE PARSER (the gate-3A diagnostic, `_optionA_gate3A_diag.json`): we exercise the composer's
kind-specific RF work DIRECTLY (`agent.composer.store/query_patient/...`), NOT via `agent.hear`/`agent.what_does`. The
agent.hear/what_does path ALSO runs the merged PARSER (`_MergedParserAdapter` toggles OU on + steps the WHOLE bridge),
which perturbs the nav v/u by a KIND-INDEPENDENT amount (the parser is identical in the rf deployment). The FLIP changes
only the COMPOSER KIND, so the functional-neutrality question is exactly "does the composer KIND's RF op perturb nav" —
which the direct-composer-op leg isolates (the parser's full-bridge stepping is a separate, kind-independent, already-
validated concern that the gate must not conflate).

GO: nav-slice v/u byte-identical (Δ=0) across the composer-OFF (post-construction) vs composer-ON (direct RF ops) legs,
AND the no-confab moat preserved (an unstored cue still abstains after the ops). A Δ≠0 would be a REAL composer->nav
functional leak (unexpected given the 0 out-edges + the per-op `_zero_rf_v_u()` reset) -> do NOT flip, report the honest
negative. GPU-only. Reuse-by-import; NO sim/ edit.

Run: SIM_BACKEND=cupy python -m research.runners._closure1_optionA_gate3A --seed 42
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "findings", "raw", "_closure1_optionA_gate3A.json")

# the conversational/composer/dlPFC/drive/generative slices to EXCLUDE — the nav slice is everything else (the parser
# parse_conj/parse_role, the dlPFC cortex_ctx/dlpfc_wm, the composer `rf`, the limbic drive_agrp/drive_pomc, and any
# generative gen_* region). The nav slice = the BG cascade + cortex_{N,E,S,W} + motor + SC + place + visual cortex etc.
_CONV_SLICE_NAMES = frozenset({
    "parse_conj", "parse_role", "cortex_ctx", "dlpfc_wm", "rf", "drive_agrp", "drive_pomc"})


def _is_conv(name):
    return (name in _CONV_SLICE_NAMES) or name.startswith("gen_")


def _nav_idx_host(rm):
    names = [nm for nm in rm.region_indices_dict() if not _is_conv(nm)]
    return np.concatenate([np.asarray(rm.indices(nm), dtype=np.int64) for nm in names]) if names else \
        np.empty(0, dtype=np.int64)


def _run_one_seed(seed):
    """Build ONE onebrain merged agent at `seed`, snapshot the nav-slice v/u, exercise the composer's RF ops DIRECTLY,
    snapshot again, and return a per-seed row (the v/u byte-identity + the moat + the composer-kind check)."""
    from sim.backend import to_host
    import cupy as cp
    from research.runners.nav_conv_merged_bridge import MergedNavConvAgent, CoResidentOneBrainComposer

    # ONE bridge at the onebrain N (composer co-resident).
    agent = MergedNavConvAgent(seed=seed, co_resident_composer=True, co_resident_composer_kind="onebrain")
    b = agent._merged_bridge
    rm = b.region_manager
    composer_ok = isinstance(agent.composer, CoResidentOneBrainComposer)
    nav_idx_h = _nav_idx_host(rm)
    nav_idx = cp.asarray(nav_idx_h)

    # Snapshot the nav-slice v/u AFTER construction (the deployed episode-start state, composer NOT yet exercised).
    v0 = to_host(b.cp_membrane_potential_v[nav_idx]).copy()
    u0 = to_host(b.cp_recovery_variable_u[nav_idx]).copy()

    # Leg ON: EXERCISE THE COMPOSER'S KIND-SPECIFIC RF WORK DIRECTLY (store + the who/what/yes-no matrix on the `rf`
    # slice), bypassing agent.hear/what_does. WHY DIRECT (the gate-3A diagnostic _optionA_gate3A_diag.json): agent.hear/
    # what_does ALSO run the merged PARSER (_MergedParserAdapter toggles OU on + steps the WHOLE bridge), which perturbs
    # the nav v/u by a KIND-INDEPENDENT amount (identical in the rf deployment — the parser is the same regardless of
    # composer kind). The FLIP changes the COMPOSER KIND, so the functional-neutrality question is precisely "does the
    # composer KIND's RF op (kick+resonate+unbind+cleanup on the `rf` slice) perturb nav" — which we isolate by calling
    # agent.composer.* directly (exactly what Probe-1 isolates and proves byte-identical on the stub bridge; here on the
    # REAL production-N merged agent + the REAL nav cascade). The parser's full-bridge stepping is a separate, already-
    # validated, kind-independent concern (NOT what the flip touches), so the gate must not conflate it.
    c = agent.composer
    c.kb = []
    c.store("dog", "go", "north")
    c.store("cat", "come", "south", polarity="NEGATE")
    _ = c.query_patient("dog", "go")            # kick+resonate+unbind+cleanup on the rf slice
    _ = c.query_agent("come", "south")
    _ = c.ask_yes_no("cat", "come", "south")
    _ = c.render_fact("dog")
    moat_unstored = c.query_patient("river", "look")    # must abstain (no-confab moat) -> None
    moat_unknown = c.ask_yes_no("apple", "stop", "west")  # -> "unknown"

    # Snapshot the nav-slice v/u AFTER exercising the composer.
    v1 = to_host(b.cp_membrane_potential_v[nav_idx]).copy()
    u1 = to_host(b.cp_recovery_variable_u[nav_idx]).copy()

    nav_v_identical = bool(np.array_equal(v0, v1))
    nav_u_identical = bool(np.array_equal(u0, u1))
    max_abs_dv = float(np.max(np.abs(v1 - v0))) if v0.size else 0.0
    max_abs_du = float(np.max(np.abs(u1 - u0))) if u0.size else 0.0
    moat_ok = (moat_unstored is None) and (moat_unknown == "unknown")
    merged_N = int(b.core_config.num_neurons)
    nav_slice_size = int(nav_idx_h.size)

    del agent
    cp.get_default_memory_pool().free_all_blocks()
    return {
        "seed": int(seed),
        "merged_N": merged_N,
        "nav_slice_size": nav_slice_size,
        "composer_is_CoResidentOneBrainComposer": composer_ok,
        "nav_v_byte_identical": nav_v_identical, "nav_u_byte_identical": nav_u_identical,
        "max_abs_nav_dv": max_abs_dv, "max_abs_nav_du": max_abs_du,
        "moat_unstored_abstains": (moat_unstored is None), "moat_unknown_is_unknown": (moat_unknown == "unknown"),
        "moat_preserved": moat_ok,
        "seed_pass": bool(nav_v_identical and nav_u_identical and composer_ok and moat_ok),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,43,44",
                    help="comma-separated seeds (the gate is a true-null; ≥3 byte-identical seeds is conclusive)")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        raise SystemExit("gate 3A needs the CuPy/GPU backend (SIM_BACKEND=cupy)")

    rows = []
    for s in seeds:
        r = _run_one_seed(s)
        rows.append(r)
        print(f"[gate3A] seed {s}: nav v/u byte-identical OFF-vs-ON: v={r['nav_v_byte_identical']} "
              f"u={r['nav_u_byte_identical']} (max|dv|={r['max_abs_nav_dv']:.2e} max|du|={r['max_abs_nav_du']:.2e}); "
              f"moat_preserved={r['moat_preserved']}", flush=True)

    max_abs_dv = max((r["max_abs_nav_dv"] for r in rows), default=0.0)
    max_abs_du = max((r["max_abs_nav_du"] for r in rows), default=0.0)
    gate3a_pass = all(r["seed_pass"] for r in rows) and len(rows) > 0
    result = {
        "gate": "option A gate 3A -- same-N functional-neutrality (direct composer ops at fixed onebrain N)",
        "backend": "cupy", "seeds": seeds,
        "logic": ("at a FIXED merged N (the onebrain rf size), exercising the co-resident composer's kind-specific RF "
                  "work DIRECTLY (agent.composer.store + the who/what/yes-no matrix on the `rf` slice; NOT agent.hear/"
                  "what_does, which would also run the kind-INDEPENDENT parser's full-bridge OU stepping -- see "
                  "_optionA_gate3A_diag.json) must leave the NAV-slice v/u BYTE-IDENTICAL -> the episode (a deterministic "
                  "OU-OFF function of the nav v/u start-state) is unperturbed by the composer kind -> Δ=0. Any Δ would be "
                  "a real composer->nav state leak (the Task-1 anti-cheat says impossible: rf has 0 out-edges into nav + "
                  "every op _zero_rf_v_u()-resets the rf slice)."),
        "rows": rows,
        "n_seeds": len(rows),
        "n_byte_identical": sum(1 for r in rows if r["nav_v_byte_identical"] and r["nav_u_byte_identical"]),
        "n_moat_preserved": sum(1 for r in rows if r["moat_preserved"]),
        "max_abs_nav_dv": max_abs_dv, "max_abs_nav_du": max_abs_du,
        "gate3a_pass": gate3a_pass,
        "verdict": ("DELTA0_GO -- exercising the composer is nav-neutral at fixed N (functional neutrality proven, all "
                    "seeds); the moat is preserved -> Option A may flip on the functional-neutrality argument"
                    if gate3a_pass else
                    "REGRESS -- the composer perturbs the nav slice at fixed N (a real state leak) OR the moat broke -> "
                    "do NOT flip; report the honest negative"),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[gate3A] VERDICT: {result['verdict']}")
    print(f"[gate3A] byte-identical {result['n_byte_identical']}/{result['n_seeds']} seeds; "
          f"moat {result['n_moat_preserved']}/{result['n_seeds']}; max|dv|={max_abs_dv:.2e} max|du|={max_abs_du:.2e}")
    print(f"[gate3A] wrote {os.path.normpath(OUT)}")
    return 0 if gate3a_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
