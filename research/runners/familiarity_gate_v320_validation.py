"""V=320 production-scale validation: does a LEARNED anti-Hebbian familiarity gate AGREE with the production
host abstention (the no-confab moat) -- WITHOUT touching the moat?

CONTEXT. The production conversational composer (research/runners/rf_phasor_composer.py, the RFPhasorComposer /
BrainConversationalAgent) answers who/what fact-queries and ABSTAINS ("I don't know" = returns None) when no
stored fact matches. That abstention is currently a HOST check: query_agent / query_patient iterate the
knowledge base (self.kb) and abstain via a Python `if` on an exact relational match (rf_phasor_composer.py
around line 305). A cheap-first de-risk (research/findings/2026-06-10-cortex-learned-cleanup-derisk-PARTIAL.md)
showed a brain-based replacement works at TOY scale: a learned Bogacz-Brown anti-Hebbian familiarity signal
(a computed novelty/familiarity score; known cue ~ 0, unknown cue ~ 0.99, margin +0.98, lesionable).

WHAT THIS RUNNER DOES. Validates that the learned familiarity gate MATCHES the host abstention DECISION at
PRODUCTION scale (V=320 concepts), multi-seed -- ALONGSIDE the host check (compute BOTH, compare), never
replacing or weakening it. This is the precursor to ever wiring the gate in: the gate must be proven to AGREE
with the host check before any replacement is even proposed.

THE FAITHFUL CUE (the load-bearing design choice). The production moat is RELATIONAL: query_agent(action,
patient) abstains when no stored fact's (action, patient) matches; query_patient(agent, action) abstains when
no stored fact's (agent, action) matches. So the familiarity gate does NOT ask "are these bare concepts
known" -- it asks "is the QUERY CUE (the bound composite of the queried roles) familiar (a stored fact has
this partial structure) or novel (no stored fact does)". Per stored fact we IMPRINT the fact's PARTIAL-FACT
composite cue (e.g. for query_agent, the bound composite of {action, patient}) into the anti-Hebbian pool;
at query time the gate renders the SAME partial-fact composite for the query and reads its novelty. The
composite is produced BY THE COMPOSER's own RF bind/bundle ops (the substrate), not a host formula -- the gate
reads it; it does not compute the match. Known relational cue -> composite in the stored span -> familiar
(low novelty) -> ACCEPT (the host answers). Unknown relational cue -> composite outside the span -> novel
(high novelty) -> ABSTAIN (the host abstains). A learned, lesionable signal.

CODES. The PRODUCTION decorrelated phasor codes the RFPhasorComposer self-generates from the seed
(rng.uniform per concept + per role) -- between-code phase-cosine ~ 0 (decorrelated by construction), NOT the
de-risk's correlated denoise64 codes (cos ~ 0.70). This is the production regime the moat actually runs in.

TESTS (multi-seed 42/43/44, V=320; step down to a smaller V only if V=320 is too slow, and note it):
  1. AGREEMENT MATRIX: gate-vs-host accept/abstain agreement on KNOWN and UNKNOWN cues. Report the confusion
     (host-abstain vs gate-ACCEPT = the dangerous cell -- a confabulation risk -- must be 0).
  2. MOAT PRESERVATION: false-accept rate on the >=20-cue abstention floor must be 0 (multi-seed). Report the
     familiarity-score separation (known mean vs unknown mean = the margin) at V=320 -- does the +0.98 toy
     margin hold at 320 concepts, or shrink (a code-density effect)?
  3. THRESHOLD ROBUSTNESS: sweep the gate threshold; report the window where false-accepts stay 0 AND known
     cues are still accepted (the operating margin).
  4. ANTI-CHEAT LESION: lesion the familiarity gate's LEARNED weights -> the novelty separation must collapse
     (confirms the decision rides the learned gate, not an artifact).

MOAT-PRESERVING CONSTRAINTS. Does NOT modify rf_phasor_composer.py or any sim/ file. The host moat stays
exactly as is; the gate is validated ALONGSIDE it (both computed, compared). Reuse-by-import:
  - RFPhasorComposer (production composer + its decorrelated codes + RF bind/bundle/_encode ops + host moat).
  - AntiHebbianFamiliarity (the learned anti-Hebbian projector) from cortex_learned_cleanup_derisk.

Run (GPU for the V=320 bridge ops; the per-op RF bridges are tiny so CPU works too but GPU is the real path):
  SIM_BACKEND=cupy python -m research.runners.familiarity_gate_v320_validation --V 320 --seeds 42 43 44
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

# Reuse-by-import: the PRODUCTION composer (codes + RF ops + host moat) and the LEARNED familiarity gate.
from research.runners.rf_phasor_composer import RFPhasorComposer, ROLES
from research.runners.cortex_learned_cleanup_derisk import AntiHebbianFamiliarity


# ---------------------------------------------------------------------------
# The familiarity gate wired to the PRODUCTION composer (alongside the host moat).
# ---------------------------------------------------------------------------
class RelationalFamiliarityGate:
    """A LEARNED anti-Hebbian familiarity gate for the composer's RELATIONAL abstention. Imprints, per stored
    fact, the fact's PARTIAL-FACT composite cue (the bound composite of the roles a given query holds fixed)
    into a Bogacz-Brown anti-Hebbian pool; scores a query's familiarity by the novelty of the SAME partial-fact
    composite. The composite phases come from the composer's own RF _encode (the substrate produces the cue;
    the gate reads it). Two pools, one per query type:
      - 'who <action> <patient>?'  (query_agent)   cue = composite of {action, patient}
      - 'what does <agent> <action>?' (query_patient) cue = composite of {agent, action}
    The host moat (query_agent / query_patient returning None) is UNTOUCHED and computed alongside.
    """

    def __init__(self, composer: RFPhasorComposer):
        self.composer = composer
        D = composer.D
        # one anti-Hebbian pool per query type (the cue's role set differs)
        self.pool_agent = AntiHebbianFamiliarity(D)     # query_agent: cue roles = {action, patient}
        self.pool_patient = AntiHebbianFamiliarity(D)   # query_patient: cue roles = {agent, action}

    # --- cue rendering: the partial-fact composite via the composer's OWN RF bind/bundle ops ---
    def _partial_composite(self, roles_dict):
        """The bound composite phases for a partial fact (a subset of roles), produced by the composer's RF
        _encode (bind each role-filler, bundle). This IS the substrate op; the gate only reads the result."""
        return self.composer._encode(roles_dict)

    def cue_agent(self, action, patient):
        """The 'who <action> <patient>?' query cue: composite of {action, patient}."""
        return self._partial_composite({"action": action, "patient": patient})

    def cue_patient(self, agent, action):
        """The 'what does <agent> <action>?' query cue: composite of {agent, action}."""
        return self._partial_composite({"agent": agent, "action": action})

    def imprint_facts(self):
        """Imprint each stored fact's partial-fact composites into the matching pools (anti-Hebbian learning).
        Mirrors the host kb: every fact contributes its {action,patient} composite to pool_agent and its
        {agent,action} composite to pool_patient."""
        for fact, _comp in self.composer.kb:
            ap = {r: fact[r] for r in ("action", "patient") if r in fact}
            aa = {r: fact[r] for r in ("agent", "action") if r in fact}
            if "action" in ap and "patient" in ap and not _is_clause_filler(ap.get("patient")):
                self.pool_agent.imprint(self.cue_agent(ap["action"], ap["patient"]))
            if "agent" in aa and "action" in aa:
                self.pool_patient.imprint(self.cue_patient(aa["agent"], aa["action"]))

    def novelty_agent(self, action, patient):
        return self.pool_agent.novelty(self.cue_agent(action, patient))

    def novelty_patient(self, agent, action):
        return self.pool_patient.novelty(self.cue_patient(agent, action))

    def lesion(self):
        self.pool_agent.lesion()
        self.pool_patient.lesion()


def _is_clause_filler(x):
    return getattr(x, "_fields", None) == ("agent", "action", "patient")


# ---------------------------------------------------------------------------
# Build a V-concept production composer + a knowledge base of stored facts.
# ---------------------------------------------------------------------------
def build_composer_and_kb(V, n_facts, seed, D=128, period=200):
    """Build a production RFPhasorComposer over a V-concept vocab (its OWN decorrelated phasor codes), then
    store n_facts deterministic SVO facts. Returns (composer, stored_facts) where stored_facts is the list of
    (agent, action, patient) triples actually stored. The vocab is partitioned so cues can be chosen KNOWN
    (matching a stored fact) or UNKNOWN (a never-stored relational combination)."""
    vocab = ["c%04d" % i for i in range(V)]
    composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab, period=period)
    rng = np.random.default_rng(seed + 9999)
    stored = []
    seen_ap = set()   # (action, patient) pairs used -> so UNKNOWN cues avoid them
    seen_aa = set()   # (agent, action) pairs used
    # deterministic distinct SVO triples (distinct agent/action/patient so the relational structure is clean)
    while len(stored) < n_facts:
        a, ac, p = (vocab[i] for i in rng.choice(V, size=3, replace=False))
        if (ac, p) in seen_ap or (a, ac) in seen_aa:
            continue
        composer.store(a, ac, p)
        stored.append((a, ac, p))
        seen_ap.add((ac, p))
        seen_aa.add((a, ac))
    return composer, stored, vocab, seen_ap, seen_aa


def make_unknown_ap_cues(vocab, seen_ap, n, rng):
    """n KNOWN-impossible 'who <action> <patient>?' cues: (action, patient) pairs that are NOT any stored
    fact's (action, patient) -> the host MUST abstain on these (the abstention floor)."""
    cues = []
    tries = 0
    while len(cues) < n and tries < 100000:
        tries += 1
        ac, p = (vocab[i] for i in rng.choice(len(vocab), size=2, replace=False))
        if (ac, p) in seen_ap:
            continue
        cues.append((ac, p))
        seen_ap = seen_ap | {(ac, p)}   # distinct cues
    return cues


def make_unknown_aa_cues(vocab, seen_aa, n, rng):
    """n KNOWN-impossible 'what does <agent> <action>?' cues: (agent, action) NOT any stored fact's."""
    cues = []
    tries = 0
    seen = set(seen_aa)
    while len(cues) < n and tries < 100000:
        tries += 1
        a, ac = (vocab[i] for i in rng.choice(len(vocab), size=2, replace=False))
        if (a, ac) in seen:
            continue
        cues.append((a, ac))
        seen.add((a, ac))
    return cues


# ---------------------------------------------------------------------------
# Per-seed evaluation.
# ---------------------------------------------------------------------------
def evaluate_seed(V, n_facts, n_unknown, seed, D=128, period=200):
    composer, stored, vocab, seen_ap, seen_aa = build_composer_and_kb(V, n_facts, seed, D=D, period=period)

    # between-code phase-cosine of the PRODUCTION codes (auditable: the decorrelated regime).
    codes = np.stack([composer.concepts[w] for w in vocab])
    nb = min(len(vocab), 120)   # sample for speed at large V
    idx = np.random.default_rng(seed).choice(len(vocab), size=nb, replace=False)
    cs = [float(np.mean(np.cos(2 * np.pi * (codes[idx[i]] - codes[idx[k]]))))
          for i in range(nb) for k in range(i + 1, nb)]
    between_code_phase_cos = {"mean": float(np.mean(cs)), "max": float(np.max(cs)), "min": float(np.min(cs))}

    gate = RelationalFamiliarityGate(composer)
    gate.imprint_facts()

    rng = np.random.default_rng(seed + 123)

    # KNOWN cues: the stored facts' relational cues (the host ANSWERS on these).
    known_ap = [(ac, p) for (a, ac, p) in stored]          # query_agent cues
    known_aa = [(a, ac) for (a, ac, p) in stored]          # query_patient cues
    # UNKNOWN cues: relational combinations not in any stored fact (the host ABSTAINS = the abstention floor).
    unknown_ap = make_unknown_ap_cues(vocab, seen_ap, n_unknown, rng)
    unknown_aa = make_unknown_aa_cues(vocab, seen_aa, n_unknown, rng)

    # --- record HOST decision + gate novelty score for every cue (both query types) ---
    def host_accepts_agent(ac, p):
        return composer.query_agent(ac, p) is not None      # the production moat: not-None = accept (answer)

    def host_accepts_patient(a, ac):
        return composer.query_patient(a, ac) is not None

    rows = []   # (query_type, cue, host_accept(bool), novelty(float), is_known(bool))
    for (ac, p) in known_ap:
        rows.append(("agent", (ac, p), host_accepts_agent(ac, p), gate.novelty_agent(ac, p), True))
    for (ac, p) in unknown_ap:
        rows.append(("agent", (ac, p), host_accepts_agent(ac, p), gate.novelty_agent(ac, p), False))
    for (a, ac) in known_aa:
        rows.append(("patient", (a, ac), host_accepts_patient(a, ac), gate.novelty_patient(a, ac), True))
    for (a, ac) in unknown_aa:
        rows.append(("patient", (a, ac), host_accepts_patient(a, ac), gate.novelty_patient(a, ac), False))

    # separation of the LEARNED novelty score by host decision (the margin at V).
    nov_host_accept = np.array([n for (_, _, ha, n, _) in rows if ha])      # host answers (known/familiar)
    nov_host_abstain = np.array([n for (_, _, ha, n, _) in rows if not ha])  # host abstains (unknown/novel)
    known_max = float(nov_host_accept.max()) if nov_host_accept.size else float("nan")
    unknown_min = float(nov_host_abstain.min()) if nov_host_abstain.size else float("nan")
    margin = float(unknown_min - known_max) if (nov_host_accept.size and nov_host_abstain.size) else float("nan")
    # midpoint threshold (the gate's decision boundary) -- placed between the host-accept max and host-abstain min.
    thr = 0.5 * (known_max + unknown_min) if (nov_host_accept.size and nov_host_abstain.size) else \
        float(np.median([n for (_, _, _, n, _) in rows]))

    # --- gate decision at the midpoint threshold: ACCEPT if novelty < thr (familiar), else ABSTAIN ---
    def confusion_at(threshold):
        # rows of the abstention floor (host abstains): gate ACCEPT on these = FALSE-ACCEPT (the moat breach).
        host_accept_gate_accept = host_accept_gate_abstain = 0
        host_abstain_gate_accept = host_abstain_gate_abstain = 0
        for (_, _, ha, nov, _) in rows:
            gate_accept = nov < threshold
            if ha and gate_accept:
                host_accept_gate_accept += 1
            elif ha and not gate_accept:
                host_accept_gate_abstain += 1
            elif (not ha) and gate_accept:
                host_abstain_gate_accept += 1   # <<< DANGEROUS CELL (confabulation risk)
            else:
                host_abstain_gate_abstain += 1
        n = len(rows)
        agree = host_accept_gate_accept + host_abstain_gate_abstain
        return {
            "host_accept_gate_accept": host_accept_gate_accept,
            "host_accept_gate_abstain": host_accept_gate_abstain,
            "host_abstain_gate_accept": host_abstain_gate_accept,   # the moat-breach cell (must be 0)
            "host_abstain_gate_abstain": host_abstain_gate_abstain,
            "n": n, "n_agree": agree, "agreement": agree / n,
        }

    confusion_mid = confusion_at(thr)

    # --- abstention-floor false-accept rate (the load-bearing moat metric) at the midpoint threshold ---
    floor_rows = [(nov) for (_, _, ha, nov, _) in rows if not ha]   # host-abstain rows = the abstention floor
    floor_false_accepts_mid = int(sum(1 for nov in floor_rows if nov < thr))
    floor_n = len(floor_rows)

    # --- THRESHOLD ROBUSTNESS sweep: the window where false-accepts stay 0 AND known cues still accepted ---
    all_nov = np.array([n for (_, _, _, n, _) in rows])
    lo, hi = float(all_nov.min()), float(all_nov.max())
    sweep = []
    n_steps = 81
    for t in np.linspace(lo - 1e-6, hi + 1e-6, n_steps):
        fa = int(sum(1 for nov in floor_rows if nov < t))                 # false-accepts on the floor
        known_acc = int(sum(1 for (_, _, ha, nov, _) in rows if ha and nov < t))  # host-answer cues accepted
        known_tot = int(sum(1 for (_, _, ha, _, _) in rows if ha))
        sweep.append({"thr": float(t), "floor_false_accepts": fa,
                      "known_accepted": known_acc, "known_total": known_tot})
    # the safe window: thresholds with 0 false-accepts AND all known cues accepted.
    safe = [s for s in sweep if s["floor_false_accepts"] == 0 and s["known_accepted"] == s["known_total"]]
    if safe:
        window = {"lo": min(s["thr"] for s in safe), "hi": max(s["thr"] for s in safe),
                  "width": max(s["thr"] for s in safe) - min(s["thr"] for s in safe),
                  "exists": True, "n_thresholds": len(safe)}
    else:
        # fall back: the window with 0 false-accepts (even if not all known accepted), to characterize.
        zfa = [s for s in sweep if s["floor_false_accepts"] == 0]
        window = {"exists": False, "n_zero_false_accept_thresholds": len(zfa),
                  "max_known_accept_at_zero_fa": (max((s["known_accepted"] for s in zfa), default=0)),
                  "known_total": int(sum(1 for (_, _, ha, _, _) in rows if ha))}

    # --- LESION anti-cheat: zero the learned weights -> novelty separation must collapse ---
    # Recompute every cue's novelty AFTER lesion (re-derive the cue from the row); the host-accept vs
    # host-abstain separation must vanish (both N(x)=||x||^2 with W=0), proving the gate rode the LEARNED weights.
    gate.lesion()
    les_known, les_unknown = [], []
    for (qt, cue, ha, _n, _k) in rows:
        if qt == "agent":
            nv = gate.novelty_agent(cue[0], cue[1])
        else:
            nv = gate.novelty_patient(cue[0], cue[1])
        (les_known if ha else les_unknown).append(nv)
    les_known = np.array(les_known)
    les_unknown = np.array(les_unknown)
    lesion_margin = float(les_unknown.min() - les_known.max()) if (les_known.size and les_unknown.size) else float("nan")
    lesion_collapsed = bool(abs(lesion_margin) <= 1e-9) or bool(les_known.size and les_unknown.size and
                                                                np.allclose(les_known.mean(), les_unknown.mean(), atol=1e-6))

    return {
        "seed": seed, "V": V, "D": D, "n_facts": n_facts,
        "n_known_cues": len(known_ap) + len(known_aa),
        "n_unknown_cues": len(unknown_ap) + len(unknown_aa),
        "n_unknown_ap": len(unknown_ap), "n_unknown_aa": len(unknown_aa),
        "between_code_phase_cos": between_code_phase_cos,
        "novelty_host_accept_mean": float(nov_host_accept.mean()) if nov_host_accept.size else float("nan"),
        "novelty_host_accept_max": known_max,
        "novelty_host_abstain_mean": float(nov_host_abstain.mean()) if nov_host_abstain.size else float("nan"),
        "novelty_host_abstain_min": unknown_min,
        "separation_margin": margin,
        "threshold_midpoint": thr,
        "confusion_at_midpoint": confusion_mid,
        "abstention_floor_n": floor_n,
        "abstention_floor_false_accepts_at_midpoint": floor_false_accepts_mid,
        "agreement_at_midpoint": confusion_mid["agreement"],
        "threshold_window": window,
        "lesion_margin": lesion_margin,
        "lesion_collapsed": lesion_collapsed,
        "lesion_known_mean": float(les_known.mean()) if les_known.size else float("nan"),
        "lesion_unknown_mean": float(les_unknown.mean()) if les_unknown.size else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=320, help="concept vocab size (production scale = 320)")
    ap.add_argument("--n-facts", type=int, default=60, help="number of SVO facts stored in the knowledge base")
    ap.add_argument("--n-unknown", type=int, default=24,
                    help="abstention-floor size PER query type (>=20 required); total floor = 2x this")
    ap.add_argument("--D", type=int, default=128, help="phasor code dimension (production composer default)")
    ap.add_argument("--period", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "..", "findings", "raw", "_familiarity_gate_v320_validation.json"))
    args = ap.parse_args()

    print("=== V=%d familiarity-gate vs host-abstention agreement (ALONGSIDE the moat) ===" % args.V, flush=True)
    print("seeds=%s n_facts=%d abstention-floor=%d/query-type (total %d) D=%d"
          % (args.seeds, args.n_facts, args.n_unknown, 2 * args.n_unknown, args.D), flush=True)

    per_seed = []
    for seed in args.seeds:
        print("\n>>> seed %d <<<" % seed, flush=True)
        r = evaluate_seed(args.V, args.n_facts, args.n_unknown, seed, D=args.D, period=args.period)
        per_seed.append(r)
        bc = r["between_code_phase_cos"]
        print("  codes: between-code phase-cos mean=%.4f max=%.4f (decorrelated production regime)"
              % (bc["mean"], bc["max"]), flush=True)
        print("  novelty  host-ACCEPT (known): mean=%.4f max=%.4f" %
              (r["novelty_host_accept_mean"], r["novelty_host_accept_max"]), flush=True)
        print("  novelty  host-ABSTAIN (unkn): mean=%.4f min=%.4f" %
              (r["novelty_host_abstain_mean"], r["novelty_host_abstain_min"]), flush=True)
        print("  SEPARATION MARGIN (unk.min - known.max) = %+.4f" % r["separation_margin"], flush=True)
        c = r["confusion_at_midpoint"]
        print("  agreement @midpoint = %.4f (%d/%d)" % (c["agreement"], c["n_agree"], c["n"]), flush=True)
        print("  CONFUSION @midpoint: host-accept/gate-accept=%d  host-accept/gate-abstain=%d  "
              "host-abstain/gate-accept=%d (<<MOAT BREACH, must be 0)  host-abstain/gate-abstain=%d"
              % (c["host_accept_gate_accept"], c["host_accept_gate_abstain"],
                 c["host_abstain_gate_accept"], c["host_abstain_gate_abstain"]), flush=True)
        print("  ABSTENTION-FLOOR false-accepts @midpoint = %d / %d  (MUST be 0)"
              % (r["abstention_floor_false_accepts_at_midpoint"], r["abstention_floor_n"]), flush=True)
        w = r["threshold_window"]
        if w.get("exists"):
            print("  THRESHOLD WINDOW (0 false-accepts AND all known accepted): [%.4f, %.4f] width=%.4f (%d thr)"
                  % (w["lo"], w["hi"], w["width"], w["n_thresholds"]), flush=True)
        else:
            print("  THRESHOLD WINDOW: NO window with 0 false-accepts AND all known accepted "
                  "(max known accepted at 0-false-accept = %d/%d)"
                  % (w.get("max_known_accept_at_zero_fa", 0), w.get("known_total", 0)), flush=True)
        print("  LESION margin=%+.4f collapsed=%s (anti-cheat: separation rides LEARNED weights)"
              % (r["lesion_margin"], r["lesion_collapsed"]), flush=True)

    # --- aggregate verdict (multi-seed) ---
    total_floor_fa = sum(r["abstention_floor_false_accepts_at_midpoint"] for r in per_seed)
    all_zero_fa = all(r["abstention_floor_false_accepts_at_midpoint"] == 0 for r in per_seed)
    all_window = all(r["threshold_window"].get("exists", False) for r in per_seed)
    all_lesion = all(r["lesion_collapsed"] for r in per_seed)
    all_agree_perfect = all(r["confusion_at_midpoint"]["agreement"] >= 0.999 for r in per_seed)
    floors_ok = all(r["abstention_floor_n"] >= 40 for r in per_seed)  # >=20/query-type * 2 = 40 total

    margins = [r["separation_margin"] for r in per_seed if not np.isnan(r["separation_margin"])]
    mean_margin = float(np.mean(margins)) if margins else float("nan")

    if all_zero_fa and all_window and all_lesion and all_agree_perfect and floors_ok:
        verdict = "GO"
    elif all_zero_fa and all_lesion:
        # zero false-accepts (moat intact) but the window is tight or some known cues misclassified.
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"   # any false-accept = moat-breach risk = NEGATIVE

    print("\n=== VERDICT: %s ===" % verdict, flush=True)
    print("  multi-seed abstention-floor false-accepts (TOTAL across seeds) = %d  (MUST be 0)" % total_floor_fa,
          flush=True)
    print("  all seeds zero-false-accept=%s  robust-window=%s  lesion-collapses=%s  perfect-agreement=%s"
          % (all_zero_fa, all_window, all_lesion, all_agree_perfect), flush=True)
    print("  mean separation margin at V=%d = %+.4f  (toy de-risk was +0.98)" % (args.V, mean_margin), flush=True)

    out = {
        "probe": "familiarity_gate_v320_validation", "V": args.V, "D": args.D,
        "n_facts": args.n_facts, "abstention_floor_per_query_type": args.n_unknown,
        "seeds": args.seeds, "verdict": verdict,
        "multiseed_total_floor_false_accepts": total_floor_fa,
        "mean_separation_margin": mean_margin,
        "gates": {
            "all_zero_false_accepts": bool(all_zero_fa),
            "robust_threshold_window": bool(all_window),
            "lesion_collapses": bool(all_lesion),
            "perfect_agreement_at_midpoint": bool(all_agree_perfect),
            "abstention_floor_size_ok": bool(floors_ok),
        },
        "per_seed": per_seed,
    }
    op = os.path.normpath(args.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(out, open(op, "w", encoding="utf-8"), indent=2)
    print("\nwrote %s" % op, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
