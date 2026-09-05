"""GNW CONGRUENCE spiking read — retires the host string-id `==` that decides organ corroboration in the LIVE
N-organ ignition bus, reusing the already-6/6-seed-GO swap-intention MATCH-VETO circuit (`pred_k -> mm_k`).

WHAT THIS RETIRES (scaffold-retirement backlog rank-8, "GNW congruence host string-id"). The production organ-
combination bus (`webapp/gnw_bus_shadow.py::_organ_reads`, installed by DEFAULT since 2026-08-13's flip/retirement —
the mechanism `webapp/server.py::brain_reply` actually runs on every turn) builds its second and third organ votes
with a bare host comparison:

    cand_B = cand_A if composer.query_patient(agent, action) == cand_A else None       # organ B: VERIFY re-check
    cand_C = cand_A if composer.query_agent(action, cand_A) == agent else None          # organ C: reverse-binding

Both organs' reads (`query_patient`, `query_agent`) are genuinely spiking FHRR unbinds; the CONGRUENCE verdict that
turns each read into a vote — "does this second read MATCH the thing it is meant to corroborate" — is a raw Python
`==`. Downstream, `norgan_hop` (`_gnw_norgan_bus_derisk.py`) already runs a genuine coincidence-ignition consensus
over `[cand_A, cand_B, cand_C]` — but by the time votes reach it, organ B/C's congruence has ALREADY been decided
by host string equality; the ignition machinery only ever sees a pre-filtered `cand_A` (vote) or `None` (no vote),
never a genuine subthreshold vote whose fate the ignition dynamics themselves determine.

THE FIX — reuse the swap-intention circuit's OWN match-veto (NOT invent a new one). `research/runners/
_gnw_neural_swap_intention_derisk.py` (6/6-seed GO, `2026-09-0X` swap finding) already contains a spiking circuit
whose ENTIRE PURPOSE is "does a proposed content MATCH the held content": `pred_k` (driven by whichever pattern IS
established/ignited) inhibits `mm_k`'s ability to fire on a proposal for the SAME slot k — mm_k fires (a genuine
population-rate signal) IFF the proposal is for a slot that is NOT currently held. This module reuses that circuit
UNCHANGED (`build`, `run_intention_swap`, `MultiLoopSTD`, `SALIENT_PA` — reuse-by-import, NO `sim/` edit, NO
re-derivation) as a stateless pairwise CONGRUENCE READER: establish `held` as the incumbent, propose `proposed`,
and read the MATCH/MISMATCH verdict off `run_intention_swap`'s own spiking output (`held`, `swapped`, `mm_peak`) —
i.e. "the SAME populations already firing for ignition" (`pattern_k`, `pred_k`, `mm_k` ARE ignition-workspace
machinery, already load-bearing on the swap decision) — never a host `k_held == k_proposed`.

ADDRESSING VS DECIDING (the honest distinction this finding rests on). `held`/`proposed` still need SOME content->
slot address before any neuron can be driven — `_assign_slots` (imported unchanged from `_gnw_coincidence_
integrator_derisk`, the SAME first-seen-order dict every 6/6-GO'd coincidence-integrator finding in this repo
already uses for exactly this purpose) supplies it. This is a UNARY lookup (each string's slot depends only on
itself and prior registrations, never on the OTHER operand being compared this call) — the same class of "wiring"
the codebase already accepts as legitimate (a fixed receptive-field-like address, not a live decision; see
`ThoughtSwapWorkspace._slot_for`, `_ExpandedSurpriseOrgan._block_for`). What moved from host to spiking is NOT the
addressing — it is the MATCH VERDICT: nowhere in this module (nor in the reused `run_intention_swap`) is `k_held ==
k_proposed` computed and used to decide the outcome. The verdict is read from `pred_k -> mm_k`'s temporal, threshold-
crossing, LESIONABLE population dynamics, which happen to discriminate "same slot" from "different slot" only
because of anatomical wiring (`pred_k` projects ONLY to `mm_k`; `pattern_k` drives ONLY `pred_k`) plus the drive
protocol (establish `held`, then propose `proposed`) — exactly the same "wiring + threshold dynamics realize the
comparison, no homunculus compares indices" property the swap finding itself already earned its GO on.

GO GATE (6 seeds 42/43/44/100/101/102): a REAL battery (organ-A/B/C-shaped pairs built from the SAME `CHAINS`/
`RFPhasorComposer` fixture `_gnw_two_distinct_organs_derisk.py` itself uses — genuine `query_patient`/`query_agent`
reads on real stored facts, not synthetic strings) scores 100% PARITY between the spiking congruence read and the
host `==` ground truth, AND the TRIGGER-LESION (mm's proposal drive silenced — the swap finding's own load-bearing
lever, reused unchanged) collapses EVERY genuine-mismatch pair to a false "congruent" reading (proving the correct
NOT-congruent verdict on real mismatches depends on mm's actual firing, not the addressing), AND build-twice
determinism holds.

ANTI-CHEATS:
  - MISMATCH COLLAPSE UNDER LESION [the anti-host-if-else]: `trigger_lesion=True` zeroes mm's proposal drive only
    (mirrors the swap finding's own lever) -> mm never fires regardless of content -> EVERY pair (match or
    mismatch) reads "congruent". A host `==` shortcut hiding behind the addressing would be untouched by this
    lesion (a python `==` does not care about a spiking population's drive); the observed collapse proves the
    verdict is read from `mm`'s firing, not from `_assign_slots`.
  - REAL AFFERENTS, NOT HAND-PICKED: every `held`/`proposed` pair is an actual `composer.query_patient`/
    `query_agent` return value on real stored facts (the `_phaseB_multihop_query_chain_derisk` CHAINS fixture),
    including a genuine cross-chain mismatch (a REAL patient/agent from a DIFFERENT chain), not a synthetic string.
  - DETERMINISM: build the reader twice at one seed -> identical seed-derived Izhikevich-param hash.
  - NO `sim/` EDIT, NO RE-DERIVATION: `git diff sim/` is empty; every constant/primitive is reuse-by-import from the
    already-GO'd swap-intention + coincidence-integrator de-risks.

DISCIPLINE: additive, standalone (this module only). The production wire-in (a DEFAULT-OFF flag in
`webapp/gnw_bus_shadow.py::_organ_reads`) and its own byte-identical/parity hook-verify are a SEPARATE file
(`_gnw_congruence_spiking_hook_verify.py`) per this codebase's rank-12 precedent (mechanism GO gate vs. production-
dispatch GO gate are DISTINCT claims — TERMS.md `wired` requires the actual call path + a byte-identical assertion
in the data, not inferred from reading the diff).

Run (CPU cheap-first; the effect is a threshold-crossing bifurcation, not GPU-scale-dependent):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_congruence_spiking_read_derisk --calibrate --seed 42
  SIM_BACKEND=numpy python -u -m research.runners._gnw_congruence_spiking_read_derisk --smoke --seed 42 \
      --json research/findings/raw/_gnw_congruence_spiking_read_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._gnw_congruence_spiking_read_derisk --six-seed \
      --json research/findings/raw/_gnw_congruence_spiking_read_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# reuse-by-import the 6/6-seed-GO swap-intention circuit (build + MultiLoopSTD + the pred_k->mm_k MATCH VETO +
# run_intention_swap + the determinism hash) — NO sim/ edit, NO re-derivation. THIS is "the existing GNW ignition
# machinery (the same populations already firing for ignition)" rank-8 asks to reuse.
from research.runners._gnw_neural_swap_intention_derisk import (
    build as _build_swap_substrate,
    run_intention_swap,
    MultiLoopSTD,
    SALIENT_PA,
    _izh_hash,
)
# reuse-by-import the SAME first-seen-order content->slot addressing every coincidence-integrator finding uses.
from research.runners._gnw_coincidence_integrator_derisk import _assign_slots
# reuse-by-import the REAL fixture (facts + composer) the two/three-organ production bus's own de-risks are built on.
from research.runners._phaseB_multihop_query_chain_derisk import CHAINS, EAT, build_vocab, store_facts
from research.runners.rf_phasor_composer import RFPhasorComposer
from tools.verdict import Verdict
from tools.lab import attributable_to

D_COMPOSER = 64


# ── the reusable spiking congruence reader ──────────────────────────────────────────────────────────────────────
class SpikingCongruenceReader:
    """Stateless-per-call spiking congruence read: "does `proposed` MATCH `held`", reusing the swap-intention
    circuit's `pred_k -> mm_k` MATCH VETO instead of a host `held == proposed`. See the module docstring's
    "ADDRESSING VS DECIDING" section for why `_assign_slots` here is wiring, not the verdict."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._S = None
        self._std = None
        self._rng_state = None    # the reader's PRIVATE RNG timeline (see webapp.gnw_congruence_spiking's isolation)

    def _ensure(self):
        if self._S is None:
            self._S = _build_swap_substrate(seed=self.seed)
            self._std = MultiLoopSTD(self._S["bridge"], self._S["xp"], self._S["ws_used"], self._S["patterns_host"])

    def congruent(self, held, proposed, *, lesion: bool = False) -> dict:
        """`held`/`proposed` are read-only organ-candidate strings (or None = the organ had nothing to check).
        Returns `{"congruent": bool, "mm_peak": float, "swapped": bool, ...}`. `held` is NEVER compared to
        `proposed` in this function — see the class/module docstring."""
        if held is None or proposed is None:
            return {"congruent": False, "reason": "missing_content", "mm_peak": None, "swapped": None,
                    "held_flag": None, "k_held": None, "k_proposed": None, "same_slot": None}
        self._ensure()
        slot_of, _order = _assign_slots([held, proposed])     # UNARY per-string addressing (wiring, not a verdict)
        k_held, k_proposed = slot_of[held], slot_of[proposed]
        r = run_intention_swap(self._S, self._std, incumbent=k_held, proposed=k_proposed,
                               proposal_pa=SALIENT_PA, trigger_lesion=bool(lesion), isolate=True)
        is_congruent = bool(r["held"] and not r["swapped"])   # pred vetoed mm (or a trivial self-propose) -> MATCH
        return {"congruent": is_congruent, "mm_peak": float(r["mm_peak"]), "swapped": bool(r["swapped"]),
                "held_flag": bool(r["held"]), "k_held": int(k_held), "k_proposed": int(k_proposed),
                "same_slot": bool(k_held == k_proposed)}      # diagnostic/reporting ONLY — never feeds `is_congruent`


# ── a REAL organ-B/organ-C-shaped battery (built from real stored facts, not synthetic strings) ────────────────
def build_battery(seed: int):
    """Real `query_patient`/`query_agent` reads on the SAME `CHAINS` fixture `_gnw_two_distinct_organs_derisk.py`
    exercises. Each chain yields 4 pairs: organ-B match/mismatch (patient-space) + organ-C match/mismatch
    (agent-space). Mismatches use a REAL patient/agent from a DIFFERENT chain (a genuine cross-content probe, not a
    synthetic string). Returns a list of (label, held, proposed, ground_truth_congruent)."""
    vocab = build_vocab()
    composer = RFPhasorComposer(seed=seed, D=D_COMPOSER, vocab=vocab)
    store_facts(composer, CHAINS, distractor_rng=np.random.default_rng(seed * 53 + 1))  # matches the sibling derisks
    pairs = []
    n = len(CHAINS)
    for i, ch in enumerate(CHAINS):
        agent, patient = ch[0], ch[1]
        cand_A = composer.query_patient(agent, EAT)                     # organ A's real forward recall
        if cand_A != patient:
            continue                                                    # fixture precondition failed -> skip (never fabricate a pair)
        raw_B_same = composer.query_patient(agent, EAT)                 # organ B: a second REAL read of the SAME query
        pairs.append((f"organB_match_{agent}", cand_A, raw_B_same, True))
        other_patient = CHAINS[(i + 1) % n][1]                          # a REAL patient from a DIFFERENT chain
        pairs.append((f"organB_mismatch_{agent}", cand_A, other_patient, False))
        recovered_agent = composer.query_agent(EAT, cand_A)             # organ C: reverse-binding VERIFY
        if recovered_agent == agent:
            pairs.append((f"organC_match_{agent}", agent, recovered_agent, True))
        other_agent = CHAINS[(i + 1) % n][0]                            # a REAL agent from a DIFFERENT chain
        pairs.append((f"organC_mismatch_{agent}", agent, other_agent, False))
    return pairs


# ── one seed: parity vs the host `==` ground truth + trigger-lesion collapse + determinism ─────────────────────
def evaluate_seed(seed: int, verbose: bool = True):
    pairs = build_battery(seed)
    reader = SpikingCongruenceReader(seed=seed)
    rows = []
    n_correct = 0
    mismatch_rows = []
    for label, held, proposed, truth in pairs:
        r = reader.congruent(held, proposed, lesion=False)
        ok = bool(r["congruent"] == truth)
        n_correct += int(ok)
        row = {"label": label, "held": held, "proposed": proposed, "ground_truth": truth,
               "spiking_congruent": r["congruent"], "match": ok, "mm_peak": r["mm_peak"],
               "same_slot": r["same_slot"]}
        if not truth:
            mismatch_rows.append(row)
        rows.append(row)

    parity = n_correct / len(pairs) if pairs else 0.0
    intact_mismatch_correct = sum(1 for row in mismatch_rows if not row["spiking_congruent"])
    intact_mismatch_acc = (intact_mismatch_correct / len(mismatch_rows)) if mismatch_rows else 0.0

    # the CONTINUOUS signal underneath the (saturated, all-correct) boolean verdict above: the raw mm_k firing
    # rate the match/mismatch read is thresholded from. Reported so the boolean 100% parity is not the only
    # evidence on record — a genuinely varying physical quantity, not a bounded [0,1] performance score, so it is
    # NOT expected to sit at a ceiling and gives a human reader the actual margin the threshold is clearing.
    match_mm_peaks = [row["mm_peak"] for row in rows if row["ground_truth"]]
    mismatch_mm_peaks = [row["mm_peak"] for row in rows if not row["ground_truth"]]
    mean_match_mm_peak = float(np.mean(match_mm_peaks)) if match_mm_peaks else float("nan")
    mean_mismatch_mm_peak = float(np.mean(mismatch_mm_peaks)) if mismatch_mm_peaks else float("nan")
    mm_peak_margin = float(mean_mismatch_mm_peak - mean_match_mm_peak)

    # TRIGGER-LESION DISSOCIATION (reused unchanged from the swap finding's own load-bearing lever): silence mm's
    # proposal drive -> EVERY mismatch pair should now read "congruent" (a false MATCH) -> discrimination collapses.
    lesion_rows = []
    for label, held, proposed, truth in pairs:
        if truth:
            continue                                          # the lesion claim is about MISMATCH pairs specifically
        rl = reader.congruent(held, proposed, lesion=True)
        lesion_rows.append({"label": label, "held": held, "proposed": proposed,
                            "lesion_congruent": rl["congruent"], "mm_peak": rl["mm_peak"]})
    lesioned_mismatch_correct = sum(1 for row in lesion_rows if not row["lesion_congruent"])
    lesioned_mismatch_acc = (lesioned_mismatch_correct / len(lesion_rows)) if lesion_rows else 0.0
    lesion_collapses = bool(lesion_rows and lesioned_mismatch_correct == 0)

    attr = attributable_to("mismatch discrimination (intact vs trigger-lesioned)",
                           float(intact_mismatch_acc), float(lesioned_mismatch_acc), warn_below=0.5)

    # DETERMINISM: build the reader twice at one seed -> identical seed-derived Izhikevich-param hash.
    reader.congruent(pairs[0][1], pairs[0][2])                # force _ensure() if not already built
    h1 = _izh_hash(reader._S["bridge"])
    reader2 = SpikingCongruenceReader(seed=seed)
    reader2.congruent(pairs[0][1], pairs[0][2])
    h2 = _izh_hash(reader2._S["bridge"])
    seed_deterministic = bool(h1 == h2 and h1 != "")

    n_pairs = len(pairs)
    n_mismatch = len(mismatch_rows)
    parity_ok = bool(n_pairs > 0 and n_correct == n_pairs)
    seed_go = bool(parity_ok and lesion_collapses and seed_deterministic and n_mismatch >= 4)

    v = Verdict("GNW congruence spiking read (seed %d)" % seed)
    v.require("100%% PARITY vs the host `==` ground truth on a REAL organ-B/organ-C battery",
              parity_ok, expect=True, note="%d/%d" % (n_correct, n_pairs))
    v.require("TRIGGER-LESION collapses discrimination: every genuine mismatch reads 'congruent' when mm is silenced",
              lesion_collapses, expect=True, note="%d/%d mismatch pairs flipped" % (
                  len(lesion_rows) - lesioned_mismatch_correct, len(lesion_rows)))
    v.require("determinism: build-twice hash", seed_deterministic, expect=True)
    v.require("the battery contains a genuine mismatch arm (not a vacuous 100%% on an all-match battery)",
              bool(n_mismatch >= 4), expect=True)
    v.disabled("homeostasis", why="frozen weights; inherited from the reused swap-intention circuit")
    v.disabled("native_short_term_plasticity",
               why="inherited from the reused swap-intention circuit (STD targets only the incumbent E->E loop)")
    vd = v.decide(go=seed_go, verbose=verbose)

    result = {
        "seed": int(seed), "verdict": vd["status"], "seed_go": bool(seed_go and vd["status"] == "GO"),
        "n_pairs": n_pairs, "n_correct": n_correct, "parity": parity,
        "n_mismatch_pairs": n_mismatch, "intact_mismatch_acc": intact_mismatch_acc,
        "lesioned_mismatch_acc": lesioned_mismatch_acc, "lesion_collapses": lesion_collapses,
        "attributable_fraction": attr, "seed_deterministic": seed_deterministic,
        "mean_match_mm_peak": mean_match_mm_peak, "mean_mismatch_mm_peak": mean_mismatch_mm_peak,
        "mm_peak_margin": mm_peak_margin,
        "go_gate": {"parity_ok": parity_ok, "lesion_collapses": lesion_collapses,
                    "seed_deterministic": seed_deterministic, "has_mismatch_arm": bool(n_mismatch >= 4)},
        "rows": rows, "lesion_rows": lesion_rows,
        "preconditions": vd["preconditions"], "disabled_processes": vd["disabled_processes"],
        "undefined_reasons": vd["undefined_reasons"],
    }
    if verbose:
        print(f"[gnw-congruence-spiking seed={seed}] verdict={vd['status']} seed_go={result['seed_go']} "
              f"parity={n_correct}/{n_pairs} lesion_collapses={lesion_collapses} "
              f"({len(lesion_rows) - lesioned_mismatch_correct}/{len(lesion_rows)} mismatch pairs flipped) "
              f"det={seed_deterministic}", flush=True)
    return result


def run_calibrate(seed, args):
    print(f"[gnw-congruence-spiking calibrate] seed={seed} — the reused pred_k->mm_k primitive, MATCH vs MISMATCH",
          flush=True)
    reader = SpikingCongruenceReader(seed=seed)
    r_match = reader.congruent("dog", "dog")
    r_mismatch = reader.congruent("dog", "cat")
    r_lesion = reader.congruent("dog", "cat", lesion=True)
    ok = bool(r_match["congruent"] is True and r_mismatch["congruent"] is False and r_lesion["congruent"] is True)
    print(f"  MATCH    (held=dog, proposed=dog)          -> congruent={r_match['congruent']}   "
          f"mm_peak={r_match['mm_peak']:.3f}  (want True)", flush=True)
    print(f"  MISMATCH (held=dog, proposed=cat)          -> congruent={r_mismatch['congruent']}   "
          f"mm_peak={r_mismatch['mm_peak']:.3f}  (want False)", flush=True)
    print(f"  MISMATCH + TRIGGER-LESION (mm silenced)    -> congruent={r_lesion['congruent']}   "
          f"mm_peak={r_lesion['mm_peak']:.3f}  (want True: collapses)", flush=True)
    print(f"  PRIMITIVE {'HOLDS' if ok else 'FAILS'}", flush=True)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"runner": "_gnw_congruence_spiking_read_derisk", "mode": "calibrate", "seed": seed,
                       "match": r_match, "mismatch": r_mismatch, "mismatch_lesioned": r_lesion,
                       "primitive_ok": ok}, f, indent=2, default=str)
    return 0 if ok else 1


def run_smoke(seed, args):
    r = evaluate_seed(seed, verbose=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_congruence_spiking_read_derisk", "mode": "smoke", "seed": seed, "result": r},
                  f, indent=2, default=str)
    print(f"\n[gnw-congruence-spiking smoke] wrote {args.json}  seed_go={r['seed_go']}", flush=True)
    return 0 if r["seed_go"] else 1


def run_six_seed(args):
    seeds = [42, 43, 44, 100, 101, 102]
    print(f"[gnw-congruence-spiking six-seed] seeds={seeds}", flush=True)
    per_seed = [evaluate_seed(s, verbose=True) for s in seeds]
    n_go = sum(1 for r in per_seed if r["seed_go"])
    n_parity = sum(1 for r in per_seed if r["go_gate"]["parity_ok"])
    n_lesion = sum(1 for r in per_seed if r["go_gate"]["lesion_collapses"])
    n_det = sum(1 for r in per_seed if r["go_gate"]["seed_deterministic"])
    pooled_go = bool(n_go == len(seeds) and n_parity == len(seeds) and n_lesion == len(seeds) and n_det == len(seeds))
    verdict = "GO" if pooled_go else ("PARTIAL" if n_parity >= 1 else "NO-GO")

    v = Verdict("GNW congruence spiking read: 6-seed aggregate")
    v.require("100%% parity on 6/6 seeds", bool(n_parity == len(seeds)), expect=True)
    v.require("trigger-lesion collapses discrimination on 6/6 seeds", bool(n_lesion == len(seeds)), expect=True)
    v.require("determinism on 6/6 seeds", bool(n_det == len(seeds)), expect=True)
    vd = v.decide(go=pooled_go)

    summary = {"runner": "_gnw_congruence_spiking_read_derisk", "mode": "six_seed", "verdict": verdict,
               "pooled_go": pooled_go, "seeds": seeds, "verdict_status": vd["status"],
               "preconditions": vd["preconditions"], "disabled_processes": vd["disabled_processes"],
               "counts": {"seed_go": n_go, "parity_ok": n_parity, "lesion_collapses": n_lesion,
                          "seed_deterministic": n_det, "n_seeds": len(seeds)},
               "per_seed": per_seed}
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[gnw-congruence-spiking six-seed] verdict={verdict} seed_go {n_go}/6 parity {n_parity}/6 "
          f"lesion_collapses {n_lesion}/6 det {n_det}/6", flush=True)
    print(f"[gnw-congruence-spiking six-seed] wrote {args.json}", flush=True)
    return 0 if pooled_go else 1


def main():
    ap = argparse.ArgumentParser(description="GNW congruence spiking read: retire the host `==` congruence check "
                                             "in the N-organ bus with the swap-intention pred->mm MATCH VETO.")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--six-seed", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_congruence_spiking_read.json")
    args = ap.parse_args()

    if args.calibrate:
        return run_calibrate(args.seed, args)
    if args.six_seed:
        return run_six_seed(args)
    if args.smoke:
        return run_smoke(args.seed, args)
    r = evaluate_seed(args.seed, verbose=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_congruence_spiking_read_derisk", "mode": "single", "result": r}, f,
                  indent=2, default=str)
    print(f"[gnw-congruence-spiking] wrote {args.json} seed_go={r['seed_go']}", flush=True)
    return 0 if r["seed_go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
