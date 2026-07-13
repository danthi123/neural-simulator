"""SPREADING-ACTIVATION SEMANTIC COMPLETION (the 2026-07-08 open-domain frontier gate's #1 cheapest+highest-value piece,
toward OPEN conversation): a held-out concept that was NEVER taught a property, only CO-OCCURS with property-bearing
concepts, gets the property via GRADED (hedged) spreading activation on the spiking HTM cortex's LEARNED associative
codes — 2-hop propagation X →(learned co-occurrence)→ {A,B} →(taught property)→ p. Rogers-McClelland distributed feature
completion / Marr-1971 CA3 pattern completion (catalog D.13); the emergent-inference reframe (no inference engine).

Distinct from EMERGE-26/30 (clean is-a / discovered-category inheritance): here the association is LEARNED PURELY from
co-occurrence (no hand-designed is-a block, no clean category), the inference is 2-HOP, and the read is GRADED-CONFIDENCE
(a hedged TYPICAL-EXPECTATION, not a hard stated fact) — extending the no-confab moat from hard-abstain to graded-hedged
(toward "I was never told X's properties, but X is coded near {A,B} which {p} → X likely {p}, hedged").

MECHANISM (reuse-by-import the committed HTM pool-bridge; NO `sim/` edit): concepts = CONTENT-block sparse codes. TEACH
(1) CO-OCCURRENCE laterals: potentiate X↔A, X↔B (bidirectional content↔content) from the "stream"; (2) PROPERTY: A→p,
B→p. QUERY X (2-hop): present X → prime (hop 1 drives the co-occurring {A,B} cells via the learned laterals) → the
above-threshold driven cells become the hop-2 active set → prime again (drives {A,B}'s taught property p) → read the
graded apical drive per property → argmax if above a hedged floor, else ABSTAIN.

ANTI-CHEATS (6-seed 42/43/44/100/101/102): held-out (X's property NEVER taught); a CONTROL held-out Y co-occurs with
q-bearers {C,D} → must complete to q NOT p (isolates the association as the cause, not a global bias); PERMUTED co-
occurrence (X↔random concepts) → completion collapses; NO-PROPAGATION lesion (coincidence off / no laterals) → abstain;
a NOVEL concept with NO co-occurrence → ABSTAIN (the graded moat). GO = X→p AND Y→q on >=5/6 seeds AND permuted+lesion
collapse AND novel abstains.

Run: SIM_BACKEND=numpy python -m research.runners._emerge_spreading_activation_completion_derisk --seeds 42 43 44 100 101 102
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

# CONTENT-block codes: property-bearers A,B (→p) / C,D (→q); held-out X (co-occurs A,B) / Y (co-occurs C,D);
# NOVEL (no co-occurrence = moat); properties p,q (2 cols each).
CONTENT = {"A": [0, 1, 2], "B": [3, 4, 5], "C": [6, 7, 8], "D": [9, 10, 11],
           "X": [12, 13, 14], "Y": [15, 16, 17], "NOVEL": [18, 19, 20],
           "p": [21, 22], "q": [23, 24]}
COOCCUR = {"X": ["A", "B"], "Y": ["C", "D"]}          # learned co-occurrence (the ONLY link X/Y have to a property)
PROP_OF = {"A": "p", "B": "p", "C": "q", "D": "q"}    # taught class-bearer properties
PROPS = ["p", "q"]
nE = 8
ACT_TH = 2
FLOOR = -40.0
M = 1 + max(c for cs in CONTENT.values() for c in cs)


def _sdr(cols):
    return set(c * nE + 0 for c in cols)


class CompletionProbe:
    def __init__(self, seed=42, epochs=80, lesion=False, cooccur=None):
        self.cooccur = dict(COOCCUR if cooccur is None else cooccur)
        self.b, self.ci, self.row, self.col = build_pool_bridge(M, nE, seed, act_th=ACT_TH, coincidence=(not lesion))
        self.z = np.zeros(M * nE)
        for _ in range(epochs):
            # (1) property facts: bearer content -> property
            for bearer, prop in PROP_OF.items():
                apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CONTENT[bearer]), _sdr(CONTENT[prop]),
                                    self.z, 0.14, 0.02, 1.0)
            # (2) co-occurrence laterals: held-out concept <-> its co-occurring bearers (bidirectional content<->content)
            for hc, bearers in self.cooccur.items():
                for bz in bearers:
                    apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CONTENT[hc]), _sdr(CONTENT[bz]),
                                        self.z, 0.14, 0.02, 1.0)
                    apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CONTENT[bz]), _sdr(CONTENT[hc]),
                                        self.z, 0.14, 0.02, 1.0)

    def _drive_from(self, active_cols):
        ab = np.zeros(len(self.ci), bool)
        for i in _sdr(active_cols):
            ab[i] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        return None if vap is None else _host(vap)[self.ci]

    def query(self, concept, hops=2):
        """2-hop spreading activation: hop-1 primes from `concept` (drives its learned co-occurring cells); the above-
        FLOOR driven CONTENT columns become the hop-2 active set; hop-2 read gives the graded property drive."""
        active = list(CONTENT[concept])
        vap = None
        for h in range(hops):
            vap = self._drive_from(active)
            if vap is None:
                return "ABSTAIN", 0.0
            # next-hop active set = content columns whose driven cells exceed the floor (the spread frontier)
            nxt = [c for c in range(M) if vap[c * nE:(c + 1) * nE].max() > FLOOR]
            if h < hops - 1:
                active = nxt if nxt else active
        drive = {p: float(np.mean([vap[c * nE:(c + 1) * nE].max() for c in CONTENT[p]])) for p in PROPS}
        best = max(drive, key=drive.get)
        margin = drive[best] - min(drive.values())
        return (best if drive[best] > FLOOR else "ABSTAIN"), margin


def _permuted_cooccur(seed):
    """X/Y co-occur with RANDOM (wrong-property or neutral) concepts -> completion must collapse."""
    rng = np.random.RandomState(seed * 5 + 1)
    pool = ["A", "B", "C", "D"]
    return {"X": list(rng.choice(pool, 2, replace=False)), "Y": list(rng.choice(pool, 2, replace=False))}


def _run_arm(seed, arm, epochs):
    cooccur = _permuted_cooccur(seed) if arm == "permuted" else None
    p = CompletionProbe(seed=seed, epochs=epochs, lesion=(arm == "lesion"), cooccur=cooccur)
    xq = p.query("X")[0]; yq = p.query("Y")[0]; nov = p.query("NOVEL")[0]
    complete_correct = float(xq == "p" and yq == "q")          # X co-occurs p-bearers -> p; Y -> q
    moat = float(nov == "ABSTAIN")
    return arm, {"X": xq, "Y": yq, "NOVEL": nov, "complete_correct": complete_correct, "moat": moat}


def run(seed, epochs):
    out = {"seed": seed}
    for arm in ("htm", "permuted", "lesion"):
        _, r = _run_arm(seed, arm, epochs)
        out[arm] = r
    go = bool(out["htm"]["complete_correct"] == 1.0 and out["htm"]["moat"] == 1.0
              and out["permuted"]["complete_correct"] == 0.0 and out["lesion"]["complete_correct"] == 0.0)
    out["GO"] = go
    print(f"[spread-complete seed={seed}] htm: X->{out['htm']['X']} Y->{out['htm']['Y']} NOVEL->{out['htm']['NOVEL']} "
          f"(complete={out['htm']['complete_correct']:.0f} moat={out['htm']['moat']:.0f}) | "
          f"permuted_complete={out['permuted']['complete_correct']:.0f} lesion_complete={out['lesion']['complete_correct']:.0f} "
          f"-> {'GO' if go else 'no'}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    res = [run(s, a.epochs) for s in a.seeds]
    ng = sum(1 for r in res if r["GO"])
    print(f"[spread-complete] {ng}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
