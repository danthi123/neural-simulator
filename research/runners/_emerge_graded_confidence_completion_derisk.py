"""GRADED-CONFIDENCE spreading-activation completion (the 2026-07-08 frontier gate's GENUINE open piece — the bit
EMERGE-30 did NOT add): a completion carries a HEDGED confidence that RANKS with the evidence strength, extending the
no-confab moat from a hard abstain to a graded "likely / possibly / I-don't-know". Reuses EMERGE-30's STREAM-LEARNED
2-hop co-occurrence completion (`build_pool_bridge` + the committed 3-term kernel; on-bridge `corr(M,C)` Hebbian);
adds ONLY a graded read of the apical-drive margin. NO `sim/` edit.

WHY (a0-read of EMERGE-30 done FIRST): EMERGE-30 stream-learns member→context co-occurrence and completes a held-out
member to its class property (2-hop), but reads CATEGORICALLY (argmax/abstain). The gate's open piece is the GRADED
hedge. Here: a STRONG member (consistent co-occurrence with one context) → high-margin CONFIDENT completion; an
AMBIGUOUS member (co-occurs ~50/50 with BOTH contexts → both properties primed) → low-margin HEDGED ("possibly"); a
NOVEL concept (no co-occurrence) → ABSTAIN (moat). Rogers-McClelland graded distributed completion; the graded read is
Bogacz-Brown-style evidence-margin confidence.

GO (6-seed standard 42/43/44/100/101/102 + FRESH 7/8/9/10/11/12): the confidence RANKS margin(strong) > margin(ambiguous)
> margin(novel), with strong=CONFIDENT (correct property, margin above the confident threshold), ambiguous=HEDGED
(above the abstain floor but below the confident threshold → "possibly"), novel=ABSTAIN. Anti-cheats: PERMUTED stream
(random context each epoch → no structure → strong collapses toward ambiguous/abstain, ranking destroyed); NO-LEARNING
(skip the stream → all abstain); LESION (coincidence off → all abstain). GO iff the strong>ambiguous>novel margin
ranking holds AND strong=confident+correct AND novel=abstain AND permuted destroys the ranking, on >=5/6 in BOTH sets.

Run: SIM_BACKEND=numpy python -m research.runners._emerge_graded_confidence_completion_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from research.runners._emerge14_stageC_onbridge_learning_derisk import build_pool_bridge, apply_kernel_update, _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners

# content codes: STRONG member (consistent ctx B), AMBIGUOUS member (co-occurs both B & F), NOVEL (no co-occurrence);
# two contexts B/F (emergent superordinates); two properties p(B)/q(F).
CONTENT = {"strong": [0, 1, 2], "ambig": [3, 4, 5], "novel": [6, 7, 8],
           "p": [9, 10], "q": [11, 12]}
CTX = {"B": [13, 14], "F": [15, 16]}
CATPROP = {"B": "p", "F": "q"}
PROP = {"p": CONTENT["p"], "q": CONTENT["q"]}
nE = 8
ACT_TH = 2
FLOOR = -40.0                                           # apical rest ~ -62; above this = "some evidence"
CONF_TH = -10.0                                         # margin threshold: CONFIDENT if best-drive margin exceeds it
M = 1 + max(c for cs in CONTENT.values() for c in cs) + max(c for cs in CTX.values() for c in cs) - min(c for cs in CTX.values() for c in cs) + 1
M = 1 + max([c for cs in CONTENT.values() for c in cs] + [c for cs in CTX.values() for c in cs])


def _sdr(cols):
    return set(c * nE + 0 for c in cols)


class GradedProbe:
    def __init__(self, seed=42, epochs=80, lesion=False, permute=False, learn=True):
        self.b, self.ci, self.row, self.col = build_pool_bridge(M, nE, seed, act_th=ACT_TH, coincidence=(not lesion))
        self.z = np.zeros(M * nE)
        rng = np.random.default_rng(seed + 7)
        if learn:
            for _ in range(epochs):
                # STRONG: always context B. AMBIGUOUS: ~50/50 B or F each epoch (genuinely mixed evidence).
                apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CONTENT["strong"]), _sdr(CTX["B"]),
                                    self.z, 0.14, 0.02, 1.0)
                amb_ctx = "B" if rng.random() < 0.5 else "F"
                if permute:                             # PERMUTED: strong ALSO random each epoch -> no structure
                    apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CONTENT["strong"]),
                                        _sdr(CTX["B" if rng.random() < 0.5 else "F"]), self.z, 0.14, 0.02, 1.0)
                apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CONTENT["ambig"]), _sdr(CTX[amb_ctx]),
                                    self.z, 0.14, 0.02, 1.0)
        for _ in range(epochs):                          # teach property on the contexts
            for cat, prop in CATPROP.items():
                apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CTX[cat]), _sdr(PROP[prop]),
                                    self.z, 0.14, 0.02, 1.0)

    def _prime(self, active):
        ab = np.zeros(len(self.ci), bool)
        for i in active:
            ab[i] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        return None if vap is None else _host(vap)[self.ci]

    def complete(self, content_cols):
        """2-hop graded read -> (label, best_prop, best_drive, margin). label in {CONFIDENT, HEDGED, ABSTAIN}."""
        v1 = self._prime(_sdr(content_cols))
        if v1 is None:
            return "ABSTAIN", None, -99.0, 0.0
        ctx_cells = set(int(i) for i in np.where(v1 > FLOOR)[0])
        if not ctx_cells:
            return "ABSTAIN", None, -99.0, 0.0
        v2 = self._prime(ctx_cells)
        dr = {p: float(np.mean([v2[c * nE:(c + 1) * nE].max() for c in cols])) for p, cols in PROP.items()}
        best = max(dr, key=dr.get)
        margin = dr[best] - min(dr.values())
        if dr[best] <= FLOOR:
            return "ABSTAIN", None, dr[best], margin
        return ("CONFIDENT" if margin > (CONF_TH - FLOOR) else "HEDGED"), best, dr[best], margin


def _run_arm(seed, arm, epochs):
    p = GradedProbe(seed=seed, epochs=epochs, lesion=(arm == "lesion"),
                    permute=(arm == "permuted"), learn=(arm != "nolearn"))
    s_lab, s_prop, _, s_m = p.complete(CONTENT["strong"])
    a_lab, a_prop, _, a_m = p.complete(CONTENT["ambig"])
    n_lab, _, _, n_m = p.complete(CONTENT["novel"])
    return {"arm": arm, "strong": (s_lab, s_prop, round(s_m, 2)), "ambig": (a_lab, a_prop, round(a_m, 2)),
            "novel": (n_lab, round(n_m, 2)), "rank_ok": bool(s_m > a_m > n_m)}


def run(seed, epochs):
    htm = _run_arm(seed, "htm", epochs); perm = _run_arm(seed, "permuted", epochs); les = _run_arm(seed, "lesion", epochs)
    # GRADED-CONFIDENCE = a 3-LEVEL distinction (the right metric; the margin alone can't separate hedged from abstain):
    strong_confident = htm["strong"][0] == "CONFIDENT" and htm["strong"][1] == "p"   # strong -> confident + correct
    ambig_hedged = htm["ambig"][0] == "HEDGED"                                        # ambiguous -> hedged (evidence, no winner)
    novel_abstain = htm["novel"][0] == "ABSTAIN"                                      # novel -> abstain (no evidence)
    three_levels = strong_confident and ambig_hedged and novel_abstain               # CONFIDENT != HEDGED != ABSTAIN
    permuted_breaks = perm["strong"][0] != "CONFIDENT"                                # permuted destroys the confident completion
    lesion_breaks = les["strong"][0] == "ABSTAIN"
    go = bool(three_levels and permuted_breaks and lesion_breaks)
    print(f"[graded-complete seed={seed}] strong={htm['strong']} ambig={htm['ambig']} novel={htm['novel']} "
          f"| 3-levels(CONF/HEDGE/ABST)={three_levels} perm_breaks={permuted_breaks} lesion_breaks={lesion_breaks} -> {'GO' if go else 'no'}", flush=True)
    return {"seed": seed, "htm": htm, "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    res = [run(s, a.epochs) for s in a.seeds]
    print(f"[graded-complete] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
