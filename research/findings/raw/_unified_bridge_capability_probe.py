"""One-bridge unification — Step 1, Task 6: the capability NO-REGRESSION gate.

Question: at PRODUCTION scale (proj_dim=800) and with the REAL substrate concept codes (`denoise64` V=16),
does merging the conversational PARSER + COMPOSER onto ONE `SimulationBridge` change the composer's recall
versus the TWO separate bridges? This builds BOTH and compares the per-category capability matrix, multi-seed.

Terms (defined once — owner standing requirement, no undefined acronyms):
  * bridge           = one `sim.bridge.SimulationBridge` (a network of simulated Izhikevich neurons).
  * BASELINE / separate = the composer on its OWN bridge (`CoreSimComposer`) + (the parser, when present, on
    its own separate bridge). This is the pre-merge two-bridge arrangement.
  * UNIFIED / merged    = the parser + composer on ONE shared bridge (`UnifiedBrainBridge`), disjoint index
    slices. The parser slice (126 neurons) at indices 0..125; the composer slice (8*proj_dim) from 126.
  * capability matrix = the per-category recall test over 6 random-word trials each:
      FLAT          store(a, ac, p)                 -> query_patient == p
      ONE-ATTRIBUTE store(a, ac, (adj, noun))       -> query_patient == "adj noun"
      TWO-ATTRIBUTE store(a, ac, ((adj1, adj2),n))  -> query_patient == adjs sorted by words.index + noun
      NEGATION      store(...) polarity AFFIRM/NEGATE-> ask_yes_no == "yes"/"no"
  * regression       = the UNIFIED score for a category drops by MORE THAN 1 trial below the BASELINE score
    on any seed (±1 trial is the spiking-noise tolerance — shared OU background noise + the parser slice's
    activity can shift the composer's operating point by a hair). Two-attribute is a KNOWN V=16 boundary in
    BOTH arrangements, so it is compared unified-vs-separate, never vs a perfect score.

This REUSES the exact capability logic in `research/findings/raw/_decorrelate_v16_probe.py::run_matrix`
(imported, not re-implemented) — both `CoreSimComposer` and `UnifiedBrainBridge` expose the same
store / query_patient / query_agent / ask_yes_no API and `.words` / `.kb`, so the one helper runs against both.

GPU NOTE: this is a SPIKING capability probe. It runs on the validated production (CuPy/GPU) backend, NOT
NumPy (a prior subagent confirmed NumPy diverges from the validated behavior at seed 42). Do NOT set
SIM_BACKEND=numpy. Each seed builds a ~6526-neuron unified bridge AND a ~6400-neuron separate bridge and trains
the parser — expect several minutes per seed; that is expected for a faithful production-scale spiking run.

    python -m research.findings.raw._unified_bridge_capability_probe --proj-dim 800 --seeds 42 43 44

The function `run_capability_comparison(seeds, proj_dim, n)` returns a structured result the Task-6 test
(`tests/test_unified_brain_bridge.py::test_unified_capability_no_regression`) consumes to assert no regression.
"""
from __future__ import annotations

import argparse

import numpy as np

from research.runners.core_sim_composition import CoreSimComposer
from research.runners.unified_brain_bridge import UnifiedBrainBridge
from research.findings.raw._decorrelate_v16_probe import run_matrix

CATEGORIES = ("flat", "one_attr", "two_attr", "negation")


def _parser_smoke_on_unified(u):
    """Smoke the PARSER at production scale on the merged bridge: 'dog go north' must parse to
    agent=dog / action=go / patient=north, and the passive frame 'north go dog' must assign the SAME agent
    (voice-invariance — the parser's defining property). Returns (active_dict, passive_agent)."""
    active = u.parse("dog go north", voice="active")
    passive = u.parse("north go dog", voice="passive")
    return active, passive.get("agent")


def run_capability_comparison(seeds=(42, 43, 44), proj_dim=800, n=6):
    """For each seed: run the capability matrix on the SEPARATE-bridge baseline (`CoreSimComposer`) and on the
    UNIFIED one-bridge (`UnifiedBrainBridge`), using the SAME `run_matrix` helper and the SAME random word
    stream (seeded by seed+1, exactly as `_decorrelate_v16_probe.main`). Returns a dict:

        {seed: {"separate": {cat: (ok, total)}, "unified": {cat: (ok, total)},
                "parser": {"active": {...}, "passive_agent": "dog"}}}

    Raises FileNotFoundError if a seed's denoise64 cache is missing (the caller/test decides to skip)."""
    results = {}
    for seed in seeds:
        # BASELINE: composer on its own separate bridge, REAL denoise64 V=16 codes (default concepts).
        comp = CoreSimComposer(seed=seed, proj_dim=proj_dim)
        rng_sep = np.random.default_rng(seed + 1)
        sep_score = run_matrix(comp, n, rng_sep)

        # UNIFIED: parser + composer on ONE bridge; default concepts -> the SAME denoise64 codes.
        u = UnifiedBrainBridge(seed=seed, proj_dim=proj_dim)
        # Smoke the parser on the merged bridge BEFORE the capability matrix mutates u.kb.
        active, passive_agent = _parser_smoke_on_unified(u)
        rng_uni = np.random.default_rng(seed + 1)        # identical word stream to the baseline
        uni_score = run_matrix(u, n, rng_uni)

        results[seed] = {
            "separate": sep_score,
            "unified": uni_score,
            "parser": {"active": active, "passive_agent": passive_agent},
        }
    return results


def format_table(results):
    """A human-readable per-seed per-category table (separate vs unified) + the parser-on-merged-bridge line."""
    lines = []
    header = f"{'seed':>5}  {'category':<10}  {'separate':>10}  {'unified':>10}  {'delta':>6}  {'verdict'}"
    lines.append(header)
    lines.append("-" * len(header))
    for seed in sorted(results):
        r = results[seed]
        for cat in CATEGORIES:
            so, st = r["separate"][cat]
            uo, ut = r["unified"][cat]
            delta = uo - so
            # regression = unified worse than separate by more than 1 trial (±1 spiking-noise tolerance)
            verdict = "REGRESSION" if delta < -1 else "ok"
            lines.append(
                f"{seed:>5}  {cat:<10}  {so:>6}/{st:<3}  {uo:>6}/{ut:<3}  {delta:>+6}  {verdict}")
        p = r["parser"]
        lines.append(
            f"{seed:>5}  parser     active={p['active']}  passive_agent={p['passive_agent']!r}")
        lines.append("-" * len(header))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    try:
        results = run_capability_comparison(tuple(args.seeds), args.proj_dim, args.n)
    except FileNotFoundError as e:
        print(f"[cap-probe] denoise64 cache missing -> skip: {e}", flush=True)
        return
    print(format_table(results), flush=True)

    # Summarize any regression for the controller.
    regressions = []
    for seed in sorted(results):
        for cat in CATEGORIES:
            so = results[seed]["separate"][cat][0]
            uo = results[seed]["unified"][cat][0]
            if uo - so < -1:
                regressions.append((seed, cat, so, uo))
    if regressions:
        print("\n[cap-probe] REGRESSIONS (unified more than 1 trial below separate):", flush=True)
        for seed, cat, so, uo in regressions:
            print(f"  seed {seed} {cat}: separate {so} -> unified {uo} (drop {so - uo})", flush=True)
    else:
        print("\n[cap-probe] NO REGRESSION: every category within +-1 trial of the separate baseline.",
              flush=True)


if __name__ == "__main__":
    main()
