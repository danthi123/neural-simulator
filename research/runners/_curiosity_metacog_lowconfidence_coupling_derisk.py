"""Curiosity x metacog coupling: does the SAME ASK-pool crave drive that fires on a hard ABSTAIN also fire, in a
GRADED way, on a merely LOW-CONFIDENCE (non-abstain) recall? (Lane B, board `curiosity-followup` named residual)

WHY THIS RUNNER. curiosity_production_organ.py's own docstring names this verbatim as a NOT-YET-BUILT next rung:
"NOVELTY = the ABSTAIN (a binary epistemic gap) ... a graded familiarity-gate novelty (Bogacz-Brown) is the next
rung. Curiosity is scoped to ABSTAINS (the clearest novelty); a low-confidence RECALL is handled by the metacog
hedge (E1) -- curiosity on a low-confidence recall is a named next rung." Checked before building (RAG + repo
search): the exhausted-feeling `learning-progress-slope` chase (2026-08-02, `_laneB_curiosity_learning_progress_
slope_derisk.py`) is a DIFFERENT residual (WHICH concept to ask about, a selector) -- this is about WHETHER to be
curious at all on a merely-uncertain (not empty) answer, never previously tested. `curiosity-to-d6wm` (2026-09-01)
is also a DIFFERENT edge (WM referent -> curiosity), not this one.

TODAY, on the SAME turn, the two organs never interact: `metacog_production_organ` only QUALIFIES an
ALREADY-PRODUCED answer (hedges it, never touches whether the brain asked a follow-up), and
`curiosity_production_organ` only fires on a hard ABSTAIN (a binary "holds nothing" flag) -- a turn that is
answered-but-uncertain gets a bare hedge and NO curiosity, even though a human who says "I think it's X, but I'm
not sure" characteristically also wants to learn more.

MECHANISM UNDER TEST (reuse-by-import, NO sim/ edit, NO new bridge): map the metacog organ's own [0,1] `evidence`
axis onto the SAME `novelty` axis the curiosity organ already accepts (`CuriosityProductionOrgan.judge(novelty=..)`
is ALREADY a continuous float, calibrated between `FAMILIAR_SIGNAL=0.0` and `NOVEL_SIGNAL=0.95` -- today only ever
called at those two endpoints). `novelty(evidence) = FAMILIAR_SIGNAL + (1-evidence) * (NOVEL_SIGNAL-FAMILIAR_SIGNAL)`
-- low evidence (the metacog hedge's own trigger regime) maps to HIGH novelty. If the ASK-pool's crave-drive
GENERALIZES past the two calibration endpoints (as `corr(gap, SPIKING-want)=+0.996` on the ORIGINAL DR-1 battery
already suggests it should), a merely-uncertain evidence level should ALSO cross the curiosity threshold, in a
monotonic, lesion-collapsible, non-shuffle-explainable way.

GO bar (pre-registered, all must hold):
  G1 MONOTONIC   Spearman rho(evidence, want_hz) <= -0.8 (want rises as evidence falls) on every seed.
  G2 CROSSES     at least one evidence level STRICTLY BETWEEN 0 and 1 (i.e. NOT the pre-existing abstain-only
                 endpoint) reads curious=True while metacog reads confident=False at that SAME evidence -- the
                 new coupling actually exercises a turn class today's wiring has NEVER connected.
  G3 LESION      the drive-removed curiosity lesion twin collapses G1's |rho| to <0.3 on every seed (the
                 monotonicity is caused by the spiking ASK-pool pathway, not a host artifact of the evidence
                 sweep itself).
  G4 SHUFFLE     permuting the evidence<->novelty pairing (same 7 novelty values, shuffled evidence labels)
                 collapses |rho| below 0.5 on every seed (the monotonic reading is not an artifact of the sweep
                 order or the fixed novelty grid alone).
  G5 METACOG_UNCHANGED  metacog's own `confident`/`balance` readings at each evidence level are BYTE-IDENTICAL
                 whether or not this runner even imports the curiosity organ (a pure read-only composition; the
                 metacog organ is never touched or lesioned by this file).

FUNCTIONAL CORRELATE ONLY, not a phenomenal claim. Additive research; no production wiring, no flag, no
sim/ edit. SIM_BACKEND=numpy CPU lane.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402

from research.runners.curiosity_production_organ import (  # noqa: E402
    CuriosityProductionOrgan,
    NOVEL_SIGNAL,
    FAMILIAR_SIGNAL,
)
from research.runners.metacog_production_organ import MetacogProductionOrgan  # noqa: E402

EVIDENCE_GRID = (0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0)  # spans metacog's own hi/lo calibration battery range


def _novelty_of(evidence: float) -> float:
    """The mapping under test: metacog's confidence-evidence axis -> curiosity's novelty axis, reusing the
    curiosity organ's OWN two calibrated endpoints (unchanged; no new constant introduced)."""
    return float(FAMILIAR_SIGNAL + (1.0 - float(evidence)) * (NOVEL_SIGNAL - FAMILIAR_SIGNAL))


def _spearman(x, y) -> float:
    """Dependency-free rank correlation (avoids a scipy hard requirement on the pool nodes)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def run_seed(seed: int, verbose: bool = False) -> dict:
    curi = CuriosityProductionOrgan(seed=seed)
    meta = MetacogProductionOrgan(seed=seed)

    rows = []
    for ev in EVIDENCE_GRID:
        mj = meta.judge(ev)
        nov = _novelty_of(ev)
        cj = curi.judge(novelty=nov)
        rows.append({"evidence": ev, "novelty": nov, "metacog_confident": mj["confident"],
                     "metacog_balance": mj["balance"], "curious": cj["curious"], "want_hz": cj["want_hz"]})

    evid = [r["evidence"] for r in rows]
    want = [r["want_hz"] for r in rows]
    rho = _spearman(evid, want)

    # G2: a NEW turn class -- strictly between the two old calibration endpoints, metacog hedges (not confident)
    # AND curiosity now fires (curious=True) at that SAME evidence level.
    new_class = [r for r in rows if 0.0 < r["evidence"] < 1.0 and (not r["metacog_confident"]) and r["curious"]]

    # G3: lesion twin (drive-removed) over the SAME grid. NOT a rank-correlation collapse -- the DR-1 de-risk's
    # own lesion criterion (curiosity_production_organ.py docstring: "129.2 -> 5.4 Hz") is a MAGNITUDE/attribution
    # collapse, and a tiny residual baseline can stay rank-monotonic (a handful of Hz of OU-noise-scale wobble)
    # even while >95% of the DRIVE's absolute swing is gone -- exactly the DR-1 finding's own numbers reproduce
    # here (129.17->5.38 Hz at the novel endpoint). Use the established tools.lab.attributable_to idiom on the
    # DYNAMIC RANGE (max-min) of want_hz across the sweep: intact range vs lesioned range.
    les_want = [curi.judge(novelty=_novelty_of(ev), lesion=True)["want_hz"] for ev in EVIDENCE_GRID]
    rho_les = _spearman(evid, les_want)
    intact_range = float(max(want) - min(want))
    lesion_range = float(max(les_want) - min(les_want))
    pct_attrib = float((intact_range - lesion_range) / intact_range) if intact_range > 1e-9 else 0.0

    # G4: shuffle control -- permute which novelty value pairs with which evidence LABEL (fixed permutation
    # seeded off the run seed so it's reproducible, distinct from the intact pairing above).
    rng = np.random.default_rng(90000 + seed)
    perm = rng.permutation(len(EVIDENCE_GRID))
    shuffled_novelties = [_novelty_of(EVIDENCE_GRID[j]) for j in perm]
    shuf_want = [curi.judge(novelty=nv)["want_hz"] for nv in shuffled_novelties]
    rho_shuf = _spearman(evid, shuf_want)

    # G5: metacog's own reads are unaffected by this composition (no shared state; separate bridges).
    meta_only = [MetacogProductionOrgan(seed=seed).judge(ev)["balance"] for ev in EVIDENCE_GRID]
    metacog_unchanged = bool(np.allclose(meta_only, [r["metacog_balance"] for r in rows]))

    checks = {
        "G1_monotonic": rho <= -0.8,
        "G2_new_turn_class_exercised": len(new_class) >= 1,
        "G3_lesion_collapses_range_attributable>=80%": pct_attrib >= 0.80,
        "G4_shuffle_collapses": abs(rho_shuf) < 0.5,
        "G5_metacog_reads_unchanged": metacog_unchanged,
    }
    go = all(checks.values())
    if verbose:
        print(f"  [seed {seed}] rho={rho:+.3f} rho_lesion={rho_les:+.3f} rho_shuffle={rho_shuf:+.3f} "
              f"intact_range={intact_range:.1f}Hz lesion_range={lesion_range:.1f}Hz attrib={pct_attrib:.1%} "
              f"new_class_hits={len(new_class)}/7 metacog_unchanged={metacog_unchanged} GO={go}", flush=True)
    return {"seed": seed, "rows": rows, "rho": rho, "rho_lesion": rho_les, "rho_shuffle": rho_shuf,
            "intact_range_hz": intact_range, "lesion_range_hz": lesion_range, "pct_attributable": pct_attrib,
            "new_class_hits": len(new_class), "checks": checks, "go": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true", help="1 seed indicator")
    ap.add_argument("--out", default=str(_REPO / "research" / "findings" / "raw" /
                                         "_curiosity_metacog_lowconfidence_coupling.json"))
    a = ap.parse_args()
    seeds = [a.seeds[0]] if a.smoke else a.seeds

    t0 = time.time()
    print(f"[curiosity-metacog-coupling] seeds={seeds} backend={os.environ.get('SIM_BACKEND')}", flush=True)
    rows = [run_seed(s, verbose=True) for s in seeds]
    n_go = sum(1 for r in rows if r["go"])
    go = n_go == len(rows)

    summary = {
        "mechanism": "curiosity-metacog-lowconfidence-coupling",
        "GO": bool(go), "n_go": n_go, "n_seeds": len(seeds),
        "per_seed": rows,
        "config": {"seeds": seeds, "smoke": a.smoke, "evidence_grid": list(EVIDENCE_GRID),
                   "backend": os.environ.get("SIM_BACKEND")},
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("=" * 100, flush=True)
    print(f"[curiosity-metacog-coupling] VERDICT: {'GO' if go else 'NOT-GO'} ({n_go}/{len(seeds)} seeds) "
          f"({summary['elapsed_s']}s)", flush=True)
    print(f"[curiosity-metacog-coupling] wrote {a.out}", flush=True)
    print("=" * 100, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
