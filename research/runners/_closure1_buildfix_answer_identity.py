"""CLOSE-OUT build-fix guard — `CoResidentOneBrainComposer(build_parser=False)` is ANSWER-IDENTICAL to
`build_parser=True` (the Probe-1 / standalone oracle), and the no-confab moat is preserved.

The build fix for the traced close-out bug (`_closure1_optionA_gate3_flip.json`): the idle layout-only
`BridgeParser(shared_bridge=merged_bridge)` that `CoResidentOneBrainComposer.__init__` built was redundant on the
MergedNavConvAgent path (comprehension goes through `_MergedParserAdapter`) AND its shared-bridge merge wiped the
framework wiring + the COMMAND_GATE registration -> the agent COMMAND_GATE anti-cheat assert crashed at construction.
The fix adds `build_parser` (default True = Probe-1/standalone-parity, UNCHANGED); the merged agent passes False.

THE LOAD-BEARING CLAIM this guard proves: dropping the idle parser leaves the composer's RF numerical OUTPUT
byte-identical. The RF ops are reset-isolated from the parser -- every op `_zero_rf_v_u()`-resets the rf slice before
kicking, and the complex `cp_rf_w_*` synapses are array-disjoint from the parser's `cp_connections` "parse" population
-- so whatever the parser's Hebbian training left on the bridge is wiped before any RF read, and the codes come from
`RFPhasorComposer(seed=...)`'s own RNG (built BEFORE the parser). This guard verifies it EMPIRICALLY rather than only
by argument: build BOTH co-resident composers on the SAME merged-stub bridge construction (build_parser True vs False),
store the same facts, run the full who/what/yes-no + abstention matrix, assert verbatim-identical answers + the moat.

(Probe 1 itself is the COMPLEMENT: it uses the DEFAULT build_parser=True and is therefore byte-UNCHANGED by this fix --
re-running it confirms the relocated class is still == standalone OneBrainComposer to atol 1e-9.)

CPU / numpy / small. Reuse-by-import; NO `sim/` edit.
Run: SIM_BACKEND=numpy python -m research.runners._closure1_buildfix_answer_identity
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")  # logic/CPU; do not contend the GPU

from research.runners.one_brain_composer import OneBrainComposer
from research.runners.nav_conv_merged_bridge import CoResidentOneBrainComposer
# reuse Probe-1's stub bridge + fact set + matrix so this guard exercises the SAME construction Probe-1 validated.
from research.runners._consolidation_probe1_byteident import (
    build_merged_stub_bridge, _build_vocab, _store_all, _run_matrix, _compare,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "findings", "raw", "_closure1_buildfix_answer_identity.json")


def run(seed=42, D=48, nav_stub=37, persistent_loop=True, enable_spiking_cleanup=False):
    vocab = _build_vocab()
    common = dict(seed=seed, D=D, vocab=vocab, period=200, persistent_loop=persistent_loop,
                  enable_spiking_cleanup=enable_spiking_cleanup, integrated_loop=False, enable_batched=True,
                  enable_rf_cudagraph=False)

    # The oracle: the standalone OneBrainComposer (build a private bridge + a parser at [0:P]) -- the Probe-1 oracle.
    standalone = OneBrainComposer(**common)
    layout_span = int(standalone.n_total)

    # build_parser=True (== Probe-1's co-resident composer; the parser wires+trains on the stub bridge).
    merged_T = build_merged_stub_bridge(nav_stub + layout_span, seed=seed)
    co_T = CoResidentOneBrainComposer(merged_T, rf_base=nav_stub, build_parser=True, **common)
    assert co_T.parser is not None and co_T.parser.index_offset == nav_stub, "build_parser=True should build the parser"

    # build_parser=False (the merged-agent path; NO parser, NO destructive merge).
    merged_F = build_merged_stub_bridge(nav_stub + layout_span, seed=seed)
    co_F = CoResidentOneBrainComposer(merged_F, rf_base=nav_stub, build_parser=False, **common)
    assert co_F.parser is None, "build_parser=False should NOT build the parser"
    # the rf layout indices are byte-identical regardless of build_parser (the parser only adds bridge wiring/state).
    assert co_F.store_base == co_T.store_base and co_F.bat_c_base == co_T.bat_c_base, "rf layout diverged with build_parser"

    for comp in (standalone, co_T, co_F):
        _store_all(comp)

    sa_rows = _run_matrix(standalone)
    coT_rows = _run_matrix(co_T)
    coF_rows = _run_matrix(co_F)

    # build_parser=False vs the oracle (standalone) AND vs build_parser=True.
    n, n_match_F_vs_sa, n_abst_tot, n_abst_match_F, mism_F_vs_sa = _compare(sa_rows, coF_rows)
    _, n_match_F_vs_T, _, _, mism_F_vs_T = _compare(coT_rows, coF_rows)
    _, n_match_T_vs_sa, _, n_abst_match_T, _ = _compare(sa_rows, coT_rows)

    answer_identical = (len(mism_F_vs_sa) == 0 and len(mism_F_vs_T) == 0)
    moat_preserved = (n_abst_match_F == n_abst_tot and n_abst_tot > 0 and n_abst_match_T == n_abst_tot)

    return dict(
        n_matrix=n,
        n_match_buildparserFalse_vs_standalone=n_match_F_vs_sa,
        n_match_buildparserFalse_vs_buildparserTrue=n_match_F_vs_T,
        n_match_buildparserTrue_vs_standalone=n_match_T_vs_sa,
        n_abstain_total=n_abst_tot, n_abstain_match_buildparserFalse=n_abst_match_F,
        n_abstain_match_buildparserTrue=n_abst_match_T,
        coF_parser_is_none=(co_F.parser is None), coT_parser_index_offset=co_T.parser.index_offset,
        answer_identical=bool(answer_identical), moat_preserved=bool(moat_preserved),
        mismatches_F_vs_standalone=mism_F_vs_sa[:20], mismatches_F_vs_T=mism_F_vs_T[:20],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=48)
    ap.add_argument("--nav-stub", type=int, default=37)
    args = ap.parse_args()

    from sim.backend import get_backend
    results = {}
    for cfg_name, spiking in [("oracle_cleanup", False), ("spiking_cleanup", True)]:
        r = run(seed=args.seed, D=args.D, nav_stub=args.nav_stub, persistent_loop=True, enable_spiking_cleanup=spiking)
        results[cfg_name] = r
        print(f"[{cfg_name}] spiking_cleanup={spiking}: F-vs-standalone {r['n_match_buildparserFalse_vs_standalone']}/"
              f"{r['n_matrix']}, F-vs-T {r['n_match_buildparserFalse_vs_buildparserTrue']}/{r['n_matrix']}, "
              f"abstain(F) {r['n_abstain_match_buildparserFalse']}/{r['n_abstain_total']}, "
              f"answer_identical={r['answer_identical']}, moat_preserved={r['moat_preserved']}")
        if r["mismatches_F_vs_standalone"] or r["mismatches_F_vs_T"]:
            print(f"  MISMATCHES vs standalone: {r['mismatches_F_vs_standalone']}")
            print(f"  MISMATCHES vs build_parser=True: {r['mismatches_F_vs_T']}")

    overall = all(r["answer_identical"] and r["moat_preserved"] for r in results.values())
    out = {
        "guard": "closure1_buildfix_answer_identity",
        "what": "CoResidentOneBrainComposer(build_parser=False) == build_parser=True == standalone OneBrainComposer "
                "on the full who/what/yes-no + abstention matrix (verbatim), the no-confab moat preserved",
        "backend": get_backend()[1],
        "seed": args.seed, "D": args.D, "nav_stub_offset": args.nav_stub,
        "configs": results,
        "GO": bool(overall),
        "verdict": ("GO -- dropping the idle parser (build_parser=False) leaves the composer answer-identical; the "
                    "merged-agent onebrain composer is safe to build under the production COMMAND_GATE default"
                    if overall else
                    "NO-GO -- build_parser=False changed an answer (see mismatches); the idle parser was load-bearing"),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nGO={overall}  ->  wrote {os.path.normpath(OUT)}")
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
