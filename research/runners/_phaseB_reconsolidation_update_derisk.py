"""CYCLE 146 cheap-first de-risk — RECONSOLIDATION: does a prediction-error-gated, in-place fact UPDATE on the
composer KB recover a corrected memory, and is it DISTINGUISHABLE from naive-append / overwrite-always / nothing?

Scope per `research/findings/2026-06-17-reconsolidation-conversational-memory-scoping.md` (Option A). The
production conversational memory (`RFPhasorComposer`) is APPEND-ONLY: store() pushes onto self.kb and query_*
returns the FIRST match -- so "the dog went north" then "actually, south" yields TWO contradictory facts, the
stale one answered first. Reconsolidation (Nader 2000; Osan-Tort-Amaral 2011 mismatch-gated attractor update;
Sevenster 2013 prediction-error necessity) is the fix: a reactivated trace becomes labile and is UPDATED IN
PLACE -- but ONLY when retrieval carries a prediction error (a mismatch); a fully-predicted re-statement
re-stabilizes unchanged. Without that boundary condition "reconsolidation" degenerates to trivial last-write-wins.

THE LOAD-BEARING CLAIM under test: a reactivation-gated update REPLACES the reactivated fact in place AND is
cleanly distinguishable from (NAIVE-APPEND / OVERWRITE-ALWAYS / DO-NOTHING), with the prediction-error boundary
condition real (a PE~0 re-statement must NOT change the memory) and the no-confab moat respected (correcting a
NEVER-stored fact must ABSTAIN, not fabricate).

FOUR ARMS on the SAME corrective utterance ("dog go south" cueing the stored "dog go north"):
  1. RECONSOLIDATE     -- PE-gated in-place update (the mechanism).
  2. NAIVE-APPEND      -- current production: store() the correction -> two "dog go {north,south}" coexist.
  3. OVERWRITE-ALWAYS  -- ablation of the boundary condition: rewrite on ANY cue match regardless of PE.
  4. DO-NOTHING        -- ablation of the update: ignore the correction; stale fact persists.

Anti-cheats (decisive): C1 NO-PREDICTION-ERROR (re-state the SAME fact, PE~0 -> RECONSOLIDATE must NOT write;
OVERWRITE-ALWAYS writes anyway = the last-write-wins tell); C2 MOAT (correct a NEVER-stored subject -> abstain,
no fabrication); C3 LESION/PERMUTED (force PE=0 -> update collapses to DO-NOTHING; a wrong-cue correction ->
abstain, target intact). PE_LABILE is FROZEN at the measured same-vs-different PE midpoint (calibrate, not tuned).
6 seeds 42/43/44/100/101/102. Reuse-by-import only; NO sim/ edit; numpy/CPU (the RF bind/unbind/bundle still run
on the real resonate-and-fire spiking substrate via _resonate -- only the kb store + cleanup use the validated
numpy fast path).

Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_reconsolidation_update_derisk
      [--seeds 42,43,44,100,101,102]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402

# A small, well-separated SVO vocab (agents / actions / patients).  elephant is the NEVER-stored moat probe.
VOCAB = ["dog", "cat", "bird", "fish", "elephant",
         "go", "run", "fly", "swim",
         "north", "south", "east", "west"]
BASE_FACTS = [("dog", "go", "north"), ("cat", "run", "south"),
              ("bird", "fly", "east"), ("fish", "swim", "west")]
CORRECTION = ("dog", "go", "south")     # cue dog+go (stored north), new patient south = a real mismatch (high PE)
RESTATE = ("dog", "go", "north")        # same patient -> PE~0 (the boundary-condition probe)
NEVER = ("elephant", "go", "west")      # never stored -> must abstain (the moat)
PATIENTS = ["north", "south", "east", "west"]
D = 128                                 # match the production agent's composer dimension
ARMS = ["reconsolidate", "naive_append", "overwrite_always", "do_nothing"]


class ReconsolidatingComposer(RFPhasorComposer):
    """RFPhasorComposer + a mismatch-gated, in-place fact update (Option A). The kb stays the same (fact, comp)
    list; update_on_mismatch rewrites the matched entry instead of appending. Reuse-by-import; no sim/ edit."""

    def _find_fact(self, agent, action):
        """First kb entry whose CUE roles (agent+action) match, by the substrate unbind+cleanup. None = no trace."""
        for i, (fact, comp) in enumerate(self.kb):
            if self.unbind(comp, "agent") == agent and self.unbind(comp, "action") == action:
                return i, fact, comp
        return None

    def patient_pe(self, comp, new_patient):
        """Prediction error for the patient slot = 1 - phase-cos(recovered patient phasor, the asserted new code).
        ~0 when the new filler matches the stored one (a re-statement); ~1 when it mismatches (a real correction)."""
        rec = self._unbind_phases(comp, "patient")
        return 1.0 - float(np.mean(np.cos(2.0 * np.pi * (rec - self.concepts[new_patient]))))

    def _rewrite_patient(self, idx, fact, new_patient):
        f2 = dict(fact); f2["patient"] = new_patient
        self.kb[idx] = (f2, self._encode(f2))            # re-bind + replace IN PLACE (no append)

    def count_facts(self, agent, action):
        return sum(1 for fact, comp in self.kb
                   if self.unbind(comp, "agent") == agent and self.unbind(comp, "action") == action)

    def update_on_mismatch(self, agent, action, new_patient, mode, pe_labile, force_pe=None):
        """Route a corrective utterance. Returns {action, wrote, pe}. abstain = no trace to reactivate (the moat).
        force_pe (C3 lesion): override the measured PE (feed 0.0 -> reconsolidate must collapse to no-write)."""
        found = self._find_fact(agent, action)
        if found is None:
            return {"action": "abstain", "wrote": False, "pe": None}   # no trace -> no update, no fabrication
        idx, fact, comp = found
        pe = float(force_pe) if force_pe is not None else self.patient_pe(comp, new_patient)
        if mode == "reconsolidate":
            if pe >= pe_labile:
                self._rewrite_patient(idx, fact, new_patient); return {"action": "rewrite", "wrote": True, "pe": pe}
            return {"action": "restabilize", "wrote": False, "pe": pe}       # PE-gated: predicted -> no change
        if mode == "overwrite_always":
            self._rewrite_patient(idx, fact, new_patient); return {"action": "rewrite", "wrote": True, "pe": pe}
        if mode == "do_nothing":
            return {"action": "nothing", "wrote": False, "pe": pe}
        raise ValueError(mode)


def build(seed):
    c = ReconsolidatingComposer(seed=seed, D=D, vocab=VOCAB)
    for a, ac, p in BASE_FACTS:
        c.store(a, ac, p)
    return c


def calibrate_pe(seed):
    """Frozen PE_LABILE = midpoint of the measured same-vs-different PE distributions (the calibrate_threshold
    rule -- the data's own separation point, NOT tuned to a downstream probe)."""
    c = build(seed)
    same, diff = [], []
    for (a, ac, p) in BASE_FACTS:
        idx, fact, comp = c._find_fact(a, ac)
        same.append(c.patient_pe(comp, p))                       # vs the stored patient (a re-statement)
        for q in PATIENTS:
            if q != p:
                diff.append(c.patient_pe(comp, q))               # vs a different patient (a correction)
    same_m, diff_m = float(np.mean(same)), float(np.mean(diff))
    return 0.5 * (same_m + diff_m), same_m, diff_m


def baseline_ok(seed):
    """Sanity: the substrate must recover the stored facts before we test corrections (else the de-risk is
    confounded by a broken baseline, not by reconsolidation)."""
    c = build(seed)
    return all(c.query_patient(a, ac) == p for a, ac, p in BASE_FACTS)


def run_arm(seed, mode, pe_labile):
    c = build(seed)
    res = None
    if mode == "naive_append":
        c.store(*CORRECTION)                                     # current production path
    else:
        res = c.update_on_mismatch(*CORRECTION, mode=mode, pe_labile=pe_labile)
    return {"q_corr": c.query_patient("dog", "go"),              # corrected? south / north(stale)
            "n_doggo": c.count_facts("dog", "go"),               # exactly-one? 1 / 2(duplicate)
            "q_untouched": c.query_patient("cat", "run"),        # collateral? must stay south
            "res": res}


def c1_no_pe(seed, pe_labile):
    """Re-state the SAME fact (PE~0). RECONSOLIDATE must NOT write (boundary condition); OVERWRITE-ALWAYS writes
    anyway -- the tell that it is last-write-wins, not reconsolidation."""
    out = {}
    for mode in ("reconsolidate", "overwrite_always"):
        c = build(seed)
        r = c.update_on_mismatch(*RESTATE, mode=mode, pe_labile=pe_labile)
        out[mode] = {"wrote": r["wrote"], "pe": r["pe"], "q": c.query_patient("dog", "go")}
    return out


def c2_moat(seed, pe_labile):
    """Correct a NEVER-stored subject -> must abstain (no trace), not fabricate."""
    c = build(seed)
    r = c.update_on_mismatch(*NEVER, mode="reconsolidate", pe_labile=pe_labile)
    return {"action": r["action"], "q": c.query_patient("elephant", "go"),
            "n": c.count_facts("elephant", "go")}


def c3_lesion_permuted(seed, pe_labile):
    """lesion: force PE=0 -> the update must collapse to no-write (proves it is DRIVEN by the measured mismatch).
    permuted: correct dog with a WRONG action (fly is bird's) -> no cue match -> abstain; the real dog-go intact."""
    cl = build(seed)
    rl = cl.update_on_mismatch("dog", "go", "south", mode="reconsolidate", pe_labile=pe_labile, force_pe=0.0)
    lesion = {"wrote": rl["wrote"], "q": cl.query_patient("dog", "go")}
    cp = build(seed)
    rp = cp.update_on_mismatch("dog", "fly", "south", mode="reconsolidate", pe_labile=pe_labile)
    perm = {"action": rp["action"], "q_dogfly": cp.query_patient("dog", "fly"),
            "q_doggo": cp.query_patient("dog", "go")}
    return {"lesion": lesion, "perm": perm}


def run_seed(seed):
    pe_labile, same_m, diff_m = calibrate_pe(seed)
    base_ok = baseline_ok(seed)
    arms = {m: run_arm(seed, m, pe_labile) for m in ARMS}
    c1 = c1_no_pe(seed, pe_labile)
    c2 = c2_moat(seed, pe_labile)
    c3 = c3_lesion_permuted(seed, pe_labile)

    # Per-seed pass logic (pre-registered).
    reconsolidate_pass = (arms["reconsolidate"]["q_corr"] == "south"
                          and arms["reconsolidate"]["n_doggo"] == 1
                          and arms["reconsolidate"]["q_untouched"] == "south")
    naive_fails = (arms["naive_append"]["n_doggo"] == 2 and arms["naive_append"]["q_corr"] == "north")
    overwrite_passes_main = (arms["overwrite_always"]["q_corr"] == "south"
                             and arms["overwrite_always"]["n_doggo"] == 1)
    donothing_fails = (arms["do_nothing"]["q_corr"] == "north")
    # C1: reconsolidate did NOT write at PE~0 AND DID write on the correction; overwrite wrote even at PE~0 (the tell)
    recon_corr_wrote = arms["reconsolidate"]["res"]["wrote"]
    c1_boundary_ok = (c1["reconsolidate"]["wrote"] is False and recon_corr_wrote is True
                      and c1["overwrite_always"]["wrote"] is True)
    c2_ok = (c2["action"] == "abstain" and c2["q"] is None and c2["n"] == 0)
    c3_ok = (c3["lesion"]["wrote"] is False and c3["lesion"]["q"] == "north"
             and c3["perm"]["action"] == "abstain" and c3["perm"]["q_doggo"] == "north")
    arms_separated = (reconsolidate_pass and naive_fails and donothing_fails
                      and overwrite_passes_main and (c1["overwrite_always"]["wrote"] is True))

    row = {"seed": seed, "pe_labile": pe_labile, "pe_same": same_m, "pe_diff": diff_m, "baseline_ok": base_ok,
           "recon_pe": arms["reconsolidate"]["res"]["pe"], "c1_restate_pe": c1["reconsolidate"]["pe"],
           "reconsolidate_pass": bool(reconsolidate_pass), "naive_fails": bool(naive_fails),
           "overwrite_passes_main": bool(overwrite_passes_main), "donothing_fails": bool(donothing_fails),
           "c1_boundary_ok": bool(c1_boundary_ok), "c2_moat_ok": bool(c2_ok), "c3_ok": bool(c3_ok),
           "arms_separated": bool(arms_separated),
           "arms": {m: {k: v for k, v in arms[m].items() if k != "res"} for m in ARMS}}
    print(f"  [seed {seed}] PE same {same_m:.3f}/diff {diff_m:.3f} -> labile {pe_labile:.3f} | baseline {'OK' if base_ok else 'BROKEN'}",
          flush=True)
    print(f"           RECONSOLIDATE q={arms['reconsolidate']['q_corr']} n={arms['reconsolidate']['n_doggo']} "
          f"(pe {arms['reconsolidate']['res']['pe']:.3f}) | NAIVE q={arms['naive_append']['q_corr']} "
          f"n={arms['naive_append']['n_doggo']} | OVERWRITE q={arms['overwrite_always']['q_corr']} | "
          f"DO-NOTHING q={arms['do_nothing']['q_corr']}", flush=True)
    print(f"           C1 boundary {'OK' if c1_boundary_ok else 'FAIL'} (restate wrote={c1['reconsolidate']['wrote']} "
          f"pe {c1['reconsolidate']['pe']:.3f}; overwrite wrote={c1['overwrite_always']['wrote']}) | "
          f"C2 moat {'OK' if c2_ok else 'FAIL'} | C3 {'OK' if c3_ok else 'FAIL'} | "
          f"separated {'YES' if arms_separated else 'NO'}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_REPO, "research", "findings", "raw", "_phaseB_reconsolidation_update.json"))
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    print("[reconsolidation de-risk] does a PE-gated in-place fact UPDATE recover a corrected memory AND stay "
          "distinguishable from append/overwrite/nothing, with the boundary condition + moat intact?", flush=True)
    print(f"  seeds={seeds}  D={D}  base_facts={BASE_FACTS}  correction={CORRECTION}", flush=True)
    rows = [run_seed(s) for s in seeds]

    n = len(seeds)
    bar = int(np.ceil(5 / 6 * n))
    n_recon = sum(r["reconsolidate_pass"] for r in rows)
    n_c1 = sum(r["c1_boundary_ok"] for r in rows)
    n_c2 = sum(r["c2_moat_ok"] for r in rows)
    n_c3 = sum(r["c3_ok"] for r in rows)
    n_sep = sum(r["arms_separated"] for r in rows)
    n_base = sum(r["baseline_ok"] for r in rows)

    print(f"\n{'='*100}", flush=True)
    print(f"  MEAN ({n} seeds): RECONSOLIDATE corrects {n_recon}/{n} | C1 boundary {n_c1}/{n} | "
          f"C2 moat {n_c2}/{n} | C3 lesion/permuted {n_c3}/{n} | arms separated {n_sep}/{n} | baseline {n_base}/{n}",
          flush=True)
    print(f"  PE separation (mean): same {np.mean([r['pe_same'] for r in rows]):.3f} vs "
          f"diff {np.mean([r['pe_diff'] for r in rows]):.3f} (labile gate "
          f"{np.mean([r['pe_labile'] for r in rows]):.3f})", flush=True)

    go = (n_recon >= bar and n_c1 == n and n_c2 == n and n_c3 == n and n_sep >= bar and n_base == n)
    boundary = (n_recon >= bar and n_base == n and not go)      # capability works but a control is seed-fragile
    if go:
        verdict = "GO"
        print(f"  GO ({n_recon}/{n} correct, C1 {n_c1}/{n}, C2 {n_c2}/{n}, C3 {n_c3}/{n}, separated {n_sep}/{n}): a "
              f"prediction-error-gated in-place UPDATE recovers the corrected memory with exactly one fact, is "
              f"cleanly distinguishable from append/overwrite/nothing, the boundary condition is REAL (a PE~0 "
              f"re-statement does NOT change the memory while overwrite-always does), and the no-confab moat holds "
              f"(a never-stored correction abstains). ==> build Option A as a default-off additive "
              f"RFPhasorComposer.update_on_mismatch + a correction-turn hook in MultiTurnAgentV2; the synaptic "
              f"engram tier (Option B) is the named follow-on.", flush=True)
    elif boundary:
        verdict = "BOUNDARY"
        print(f"  BOUNDARY: the in-place update recovers the corrected memory ({n_recon}/{n}) but a control is "
              f"seed-fragile (C1 {n_c1}/{n}, C2 {n_c2}/{n}, C3 {n_c3}/{n}) -- the capability is there but the "
              f"PE boundary condition / moat is not robust at this D/vocab. Honest + informative; localize "
              f"(cleaner codes / higher D) before the production build.", flush=True)
    else:
        verdict = "NEGATIVE"
        print(f"  NEGATIVE: the composer layer cannot cleanly separate reconsolidation from last-write-wins / "
              f"fabrication ({n_recon}/{n} correct, C1 {n_c1}/{n}, C2 {n_c2}/{n}, baseline {n_base}/{n}). A real, "
              f"citable boundary -- in-place correction needs cleaner codes (the PPMI/learned-cortex arc) or the "
              f"synaptic tier directly. The honest negative is the deliverable.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*100}", flush=True)

    out = {"verdict": verdict, "seeds": seeds, "n_seeds": n, "pass_bar": bar, "D": D,
           "n_reconsolidate_correct": n_recon, "n_c1_boundary": n_c1, "n_c2_moat": n_c2, "n_c3": n_c3,
           "n_arms_separated": n_sep, "n_baseline_ok": n_base,
           "pe_same_mean": float(np.mean([r["pe_same"] for r in rows])),
           "pe_diff_mean": float(np.mean([r["pe_diff"] for r in rows])),
           "pe_labile_mean": float(np.mean([r["pe_labile"] for r in rows])),
           "base_facts": BASE_FACTS, "correction": CORRECTION, "per_seed": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
