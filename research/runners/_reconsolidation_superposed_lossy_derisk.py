"""Reconsolidation on a SUPERPOSED (shared-synapse) store -- the lossy / learned-binder path (WALL: reconsolidation).

WHY THIS LEVER (not a re-derivation). Reconsolidation is already banked GO three ways, but EVERY prior GO used a
BLOCK-MAJOR / list store -- each fact in its own composite / its own (1+D) trigger->readout block -- so update-
specificity ("correct one fact, leave the rest intact") is STRUCTURAL: the facts are PHYSICALLY SEPARATE, so a
rewrite cannot touch a neighbour by construction (see 2026-06-18-emergent-reconsolidation-in-loop-derisk.md:
"The store is block-major ... so isolation is STRUCTURAL"). The wall's genuinely un-tried residual -- the owner-OK'd
"moat-not-hard LOSSY path" (learned binders / PPMI / distributed VSA), named as the NEGATIVE branch of the scoping
doc (2026-06-17-reconsolidation-conversational-memory-scoping.md sec 5): "in-place fact-correction needs either a
cleaner-code representation (the PPMI/learned-cortex arc) or the synaptic tier directly" -- is a SUPERPOSED store
where facts SHARE one distributed trace (the biologically-real, capacity-efficient regime: real synapses are shared,
not one-slot-per-memory). Here isolation is NOT free -- it must SURVIVE CROSS-TALK.

THE LOAD-BEARING NEW QUESTION: on a SHARED superposed store, does prediction-error-gated reconsolidation still
correct a fact IN PLACE and leave the others intact -- and is the PE GATE what buys the isolation the block-major
store got for free? The failure mode is specific to superposition: the biological forget-step weakens the
REACTIVATED ESTIMATE of the stale filler (you can only weaken what reactivation surfaces), and that estimate is
corrupted by cross-talk from the other facts, so every rewrite injects residual noise into the neighbours. A re-
statement (PE~0) that needlessly rewrites therefore DEGRADES the store; the PE gate's NEW, weight-level role is to
SUPPRESS those needless writes -> protect the neighbours. This is a property the clean block-major list cannot even
exhibit.

THE STORE (faithful to the FHRR substrate, superposed not listed). Reuse RFPhasorComposer's codebook + its `_bind`
(which runs through the resonate-and-fire substrate `_resonate`). A fact (a,v,p) is a KEY->VALUE association:
    key(a,v)  = bind(code[a], code[v])          # a fact-specific key, on-substrate bind
    kv(a,v,p) = bind(key(a,v), code[p])          # the key-value association, on-substrate bind
    M         = SUM_f kv(a_f, v_f, p_f)           # ONE superposed distributed trace (magnitude-carrying HRR bundle)
Recover(a,v): rec = conj(key(a,v)) * M ; patient_hat = matched-filter cleanup(rec) over the patient codebook.
This is the standard Plate-HRR / Gayler-MAP holographic memory -- lossy (cross-talk ~ sqrt(K/D)), distributed, the
"learned binder" the wall names. (Honest scope: the superposition is held as a magnitude-carrying complex bundle;
the phase-only RF readout floor is the follow-on fidelity rung -- see the finding.)

RECONSOLIDATION on M (delta-rule / error-gated Hebbian, the distributed analogue of forget+re-store):
    reactivate -> rec, familiarity ; if familiarity < FAM_ABSTAIN -> ABSTAIN (moat: no trace, no fabrication)
    PE = 1 - phase_cos(rec, code[new]) ; p_est = cleanup(rec)   # the reactivated estimate of the stale filler
    if PE >= PE_LABILE:  M += kv(a,v,new) - kv(a,v,p_est)       # forget the reactivated estimate + store corrected
    else:                (re-stabilize -- no write)
Both gates are CALIBRATED FROM THE DATA (same-vs-different PE midpoint; stored-vs-random familiarity midpoint),
frozen before the probes -- never tuned to a downstream metric (the calibrate_threshold rule).

FOUR ARMS on the SAME corrective utterances (the distinguishability test):
  RECONSOLIDATE   -- PE-gated forget+store (the mechanism)
  NAIVE-ACCUMULATE-- M += kv(a,v,new) always, NO forget  (the "append" analogue on a superposed store: old+new
                     coexist under one key -> the readout BLENDS)
  OVERWRITE-ALWAYS-- forget+store on EVERY reactivation regardless of PE (ablate the gate) -> needless writes
                     inject cross-talk -> neighbours degrade
  DO-NOTHING      -- ignore the correction

GO bar (>=5/6 seeds, at the base set-size where baseline recall is clean): (i) REWRITE -- target recovers the new
patient; (ii) ISOLATION -- every other fact still recovers its patient (collateral ~0); (iii) C1 restabilize -- a
same-patient re-statement (PE~0) does NOT write and does NOT degrade the store; (iv) C2 moat -- a never-stored cue
ABSTAINS (no fabrication); (v) C3 -- forced-PE0 lesion collapses to no-write, permuted cue leaves the target intact;
AND the four arms cleanly separate, with the PE gate's protection of the neighbours attributable to the gate
(RECONSOLIDATE isolation >> OVERWRITE-ALWAYS isolation under repeated re-statements). Plus the lossy CAPACITY curve
(recall & isolation vs K). An honest NEGATIVE -- superposition cross-talk breaks in-place correction or the gate
cannot hold isolation -- maps the real boundary of the lossy path and names the next mechanism (delta-rule
iterative cleanup / a learned PPMI code / the synaptic tier), itself the deliverable.

Run (smoke): SIM_BACKEND=numpy python -m research.runners._reconsolidation_superposed_lossy_derisk --seeds 42
Reuse-by-import; NO sim/ edit. numpy/CPU. Nader 2000; Osan-Tort-Amaral 2011; Sevenster 2013; catalog J.27/J.34.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from research.runners.rf_phasor_composer import RFPhasorComposer
from tools.lab import lever, attributable_to


def _to_phasor(phases):
    return np.exp(2j * np.pi * np.asarray(phases))


def _phases(z):
    """complex -> phase in [0,1) (the substrate's phase unit)."""
    return (np.angle(z) / (2.0 * np.pi)) % 1.0


class SuperposedHRRMemory:
    """A distributed, LOSSY key-value store: all facts superposed into ONE complex bundle M. bind() runs on the RF
    substrate (composer._bind -> _resonate); the superposition + delta-rule forget/store are the distributed
    reconsolidation the wall's lossy path needs. Contrast the block-major list store (one composite per fact)."""

    def __init__(self, seed, D, vocab, patient_words):
        self.c = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
        self.D = D
        self.patient_words = list(patient_words)
        self.M = np.zeros(D, dtype=np.complex128)
        self.facts = {}                 # (agent, action) -> current patient (bookkeeping / ground truth, NOT the store)

    # --- on-substrate FHRR ops (bind through the resonate-and-fire substrate) ---
    def _key_phases(self, agent, action):
        return self.c._bind(self.c.concepts[agent], self.c.concepts[action])

    def _kv_phasor(self, agent, action, patient):
        return _to_phasor(self.c._bind(self._key_phases(agent, action), self.c.concepts[patient]))

    def store(self, agent, action, patient):
        self.M = self.M + self._kv_phasor(agent, action, patient)
        self.facts[(agent, action)] = patient

    def recover(self, agent, action):
        """Unbind the cue's key from the superposed trace, cleanup over the patient codebook.
        Returns (patient_hat, familiarity, rec_phases)."""
        key_z = _to_phasor(self._key_phases(agent, action))
        rec_z = np.conj(key_z) * self.M
        rec_ph = _phases(rec_z)
        sims = np.array([float(np.mean(np.cos(2.0 * np.pi * (rec_ph - self.c.concepts[w]))))
                         for w in self.patient_words])
        j = int(np.argmax(sims))
        return self.patient_words[j], float(sims[j]), rec_ph

    def pe(self, rec_ph, patient_word):
        return 1.0 - float(np.mean(np.cos(2.0 * np.pi * (rec_ph - self.c.concepts[patient_word]))))

    # --- calibrated, frozen gates (data's own separation point; not tuned to a probe) ---
    def calibrate_pe_labile(self):
        same, diff = [], []
        for (a, v), p in self.facts.items():
            _, _, rec_ph = self.recover(a, v)
            same.append(self.pe(rec_ph, p))
            for p2 in set(self.facts.values()):
                if p2 != p:
                    diff.append(self.pe(rec_ph, p2))
        if not same or not diff:
            return 0.5
        return 0.5 * (float(np.mean(same)) + float(np.mean(diff)))

    def calibrate_fam_abstain(self, rng, n_random=24):
        stored = [self.recover(a, v)[1] for (a, v) in self.facts]
        rand = []
        agents = sorted({a for (a, _) in self.facts})
        actions = sorted({v for (_, v) in self.facts})
        for _ in range(n_random):
            a = "__ghost_a%d" % rng.integers(1 << 30)
            v = "__ghost_v%d" % rng.integers(1 << 30)
            # ghost concept codes (never stored): random phasor phases, same distribution as the codebook
            self.c.concepts[a] = rng.uniform(0.0, 1.0, self.D)
            self.c.concepts[v] = rng.uniform(0.0, 1.0, self.D)
            rand.append(self.recover(a, v)[1])
        # midpoint of stored-familiarity vs random-cue-familiarity
        return 0.5 * (float(np.mean(stored)) + float(np.mean(rand)))

    def update_on_mismatch(self, agent, action, new_patient, *, mode, pe_labile, fam_abstain, force_pe=None):
        """mode in {reconsolidate, naive_accumulate, overwrite_always, do_nothing}. Returns dict(action, wrote, pe)."""
        p_hat, fam, rec_ph = self.recover(agent, action)
        # abstain (moat) applies to every WRITE-capable mode: no reactivatable trace -> no fabrication
        if mode in ("reconsolidate", "overwrite_always") and fam < fam_abstain:
            return {"action": "abstain", "wrote": False, "pe": None}
        pe = self.pe(rec_ph, new_patient) if force_pe is None else float(force_pe)

        if mode == "do_nothing":
            return {"action": "noop", "wrote": False, "pe": pe}
        if mode == "naive_accumulate":
            self.M = self.M + self._kv_phasor(agent, action, new_patient)   # add new WITHOUT forgetting old
            self.facts[(agent, action)] = new_patient
            return {"action": "accumulate", "wrote": True, "pe": pe}

        write = (pe >= pe_labile) if mode == "reconsolidate" else True      # overwrite_always ignores the gate
        if not write:
            return {"action": "restabilize", "wrote": False, "pe": pe}
        # forget the REACTIVATED ESTIMATE of the stale filler + store the corrected (delta-rule on the superposition)
        self.M = self.M + self._kv_phasor(agent, action, new_patient) - self._kv_phasor(agent, action, p_hat)
        self.facts[(agent, action)] = new_patient
        return {"action": "rewrite", "wrote": True, "pe": pe}


# ---- fact set: synthetic disjoint codebook so K scales freely and CROSS-TALK is the only limiter ----
def build_vocab(n_agent, n_action, n_patient):
    ag = ["A%d" % i for i in range(n_agent)]
    ac = ["V%d" % i for i in range(n_action)]
    pt = ["P%d" % i for i in range(n_patient)]
    return ag, ac, pt


def make_facts(rng, K, ag, ac, pt):
    """K facts with DISTINCT (agent,action) cues and randomly-assigned patients."""
    cues = []
    used = set()
    while len(cues) < K:
        a, v = ag[rng.integers(len(ag))], ac[rng.integers(len(ac))]
        if (a, v) not in used:
            used.add((a, v)); cues.append((a, v))
    facts = [(a, v, pt[rng.integers(len(pt))]) for (a, v) in cues]
    return facts


def fresh_store(seed, D, ag, ac, pt, facts):
    vocab = ag + ac + pt
    mem = SuperposedHRRMemory(seed, D, vocab, pt)
    for (a, v, p) in facts:
        mem.store(a, v, p)
    return mem


def recall_acc(mem, facts):
    ok = sum(1 for (a, v, p) in facts if mem.recover(a, v)[0] == p)
    return ok / len(facts)


def others_recall_acc(mem, facts, target_idx):
    others = [f for i, f in enumerate(facts) if i != target_idx]
    ok = sum(1 for (a, v, p) in others if mem.recover(a, v)[0] == mem.facts[(a, v)])
    return ok / max(1, len(others))


def evaluate_cell(seed, D, K, ag, ac, pt, rng, n_restate):
    """The full GO + arm-separation battery at ONE (D, K) operating point. Returns a metrics dict.
    Reused for the CLEAN base cell (the capability headline) and the LOSSY-STRESS cell (where cross-talk
    corrupts the reactivated estimate -> the PE gate's neighbour-protection becomes load-bearing)."""
    facts = make_facts(rng, K, ag, ac, pt)
    mem0 = fresh_store(seed, D, ag, ac, pt, facts)
    baseline = recall_acc(mem0, facts)
    gate = mem0.calibrate_pe_labile()
    fam = mem0.calibrate_fam_abstain(rng)

    ti = int(rng.integers(K))
    a, v, p_old = facts[ti]
    p_new = pt[(pt.index(p_old) + 7) % len(pt)]

    # (i) REWRITE + (ii) ISOLATION under a real correction
    mem = fresh_store(seed, D, ag, ac, pt, facts)
    r = mem.update_on_mismatch(a, v, p_new, mode="reconsolidate", pe_labile=gate, fam_abstain=fam)
    rewrite_ok = (r["action"] == "rewrite") and (mem.recover(a, v)[0] == p_new)
    isolation_ok = others_recall_acc(mem, facts, ti) == 1.0
    correction_pe = r["pe"]

    # (iii) C1 -- restabilize: re-STATE the SAME patient (PE~0) must NOT write, must NOT degrade the store
    mem = fresh_store(seed, D, ag, ac, pt, facts)
    before = recall_acc(mem, facts)
    rr = mem.update_on_mismatch(a, v, p_old, mode="reconsolidate", pe_labile=gate, fam_abstain=fam)
    after = recall_acc(mem, facts)
    restabilize_ok = (not rr["wrote"]) and (after >= before) and (mem.recover(a, v)[0] == p_old)
    restatement_pe = rr["pe"]

    # (iv) C2 -- moat: correct a NEVER-STORED cue -> abstain (no fabrication)
    mem = fresh_store(seed, D, ag, ac, pt, facts)
    mem.c.concepts["A_ghost"] = rng.uniform(0.0, 1.0, D)
    mem.c.concepts["V_ghost"] = rng.uniform(0.0, 1.0, D)
    gm = mem.update_on_mismatch("A_ghost", "V_ghost", pt[0], mode="reconsolidate", pe_labile=gate, fam_abstain=fam)
    moat_ok = (gm["action"] == "abstain") and (not gm["wrote"])

    # (v) C3 -- lesion (force PE=0 -> collapse to no-write) + permuted (correct a WRONG cue, target must be intact)
    mem = fresh_store(seed, D, ag, ac, pt, facts)
    lr = mem.update_on_mismatch(a, v, p_new, mode="reconsolidate", pe_labile=gate, fam_abstain=fam, force_pe=0.0)
    lesion_ok = (not lr["wrote"]) and (mem.recover(a, v)[0] == p_old)
    mem = fresh_store(seed, D, ag, ac, pt, facts)
    wa, wv, _ = facts[(ti + 1) % K]
    mem.update_on_mismatch(wa, wv, p_new, mode="reconsolidate", pe_labile=gate, fam_abstain=fam)
    permuted_ok = (mem.recover(a, v)[0] == p_old)     # the ORIGINAL target (untouched cue) unchanged

    # ---------- arm separation + the PE-gate protection (repeated re-statements stress test) ----------
    # A re-statement is a fully-predicted reactivation. RECONSOLIDATE gates it out (writes suppressed);
    # OVERWRITE-ALWAYS rewrites every time -> on a LOSSY store the reactivated estimate is corrupted by cross-talk,
    # so each ungated forget+store injects residual -> neighbours degrade.
    arms = {}
    for mode in ("reconsolidate", "naive_accumulate", "overwrite_always", "do_nothing"):
        mem = fresh_store(seed, D, ag, ac, pt, facts)
        writes = 0
        for _ in range(n_restate):
            res = mem.update_on_mismatch(a, v, p_old, mode=mode, pe_labile=gate, fam_abstain=fam)  # SAME patient
            writes += int(res["wrote"])
        others_acc = others_recall_acc(mem, facts, ti)
        target_ok = (mem.recover(a, v)[0] == p_old)
        arms[mode] = {"writes": writes, "others_recall_after_restatements": others_acc, "target_ok": bool(target_ok)}

    # naive-accumulate on a REAL correction: old+new coexist under one key -> the readout blends (the "append"
    # failure at the weight level -- no separate duplicate to answer-stale; the single trace is corrupted instead)
    mem = fresh_store(seed, D, ag, ac, pt, facts)
    mem.update_on_mismatch(a, v, p_new, mode="naive_accumulate", pe_labile=gate, fam_abstain=fam)
    naive_target = mem.recover(a, v)[0]

    return {
        "D": D, "K": K, "baseline_recall": baseline, "pe_labile": gate, "fam_abstain": fam,
        "rewrite_ok": bool(rewrite_ok), "isolation_ok": bool(isolation_ok), "correction_pe": correction_pe,
        "restabilize_ok": bool(restabilize_ok), "restatement_pe": restatement_pe,
        "moat_ok": bool(moat_ok), "lesion_ok": bool(lesion_ok), "permuted_ok": bool(permuted_ok),
        "arms_restatement_stress": arms,
        "naive_accumulate_target_after_correction": naive_target,
        "naive_accumulate_rewrite_clean": bool(naive_target == p_new),
    }


def run_seed(seed, D, K_base, K_sweep, n_restate, D_stress, K_stress):
    rng = np.random.default_rng(seed)
    # generous codebook: many patients so cleanup is a real matched-filter, plenty of cues
    ag, ac, pt = build_vocab(n_agent=8, n_action=8, n_patient=20)

    # ---------- capacity curve: baseline recall & reconsolidation isolation vs K (the lossy characterization) ----------
    capacity = []
    for K in K_sweep:
        facts = make_facts(rng, K, ag, ac, pt)
        mem = fresh_store(seed, D, ag, ac, pt, facts)
        base = recall_acc(mem, facts)
        ti = int(rng.integers(K))
        a, v, p_old = facts[ti]
        p_new = pt[(pt.index(p_old) + 7) % len(pt)]
        gate = mem.calibrate_pe_labile()
        fam = mem.calibrate_fam_abstain(rng)
        r = mem.update_on_mismatch(a, v, p_new, mode="reconsolidate", pe_labile=gate, fam_abstain=fam)
        rewrite_ok = (mem.recover(a, v)[0] == p_new)
        iso = others_recall_acc(mem, facts, ti)
        capacity.append({"K": K, "baseline_recall": base, "rewrite_ok": bool(rewrite_ok),
                         "isolation_after_1_correction": iso, "pe": r["pe"], "gate": gate})

    base = evaluate_cell(seed, D, K_base, ag, ac, pt, rng, n_restate)         # CLEAN cell -> the GO capability
    stress = evaluate_cell(seed, D_stress, K_stress, ag, ac, pt, rng, n_restate)  # LOSSY cell -> gate-protection

    return {"seed": seed, "D": D, "K_base": K_base, "capacity": capacity,
            "baseline_recall_at_base": base["baseline_recall"],
            "pe_labile": base["pe_labile"], "fam_abstain": base["fam_abstain"],
            "rewrite_ok": base["rewrite_ok"], "isolation_ok": base["isolation_ok"],
            "correction_pe": base["correction_pe"], "restabilize_ok": base["restabilize_ok"],
            "restatement_pe": base["restatement_pe"], "moat_ok": base["moat_ok"],
            "lesion_ok": base["lesion_ok"], "permuted_ok": base["permuted_ok"],
            "arms_restatement_stress": base["arms_restatement_stress"],
            "naive_accumulate_target_after_correction": base["naive_accumulate_target_after_correction"],
            "naive_accumulate_rewrite_clean": base["naive_accumulate_rewrite_clean"],
            "base_cell": base, "stress_cell": stress}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--K-base", type=int, default=6)
    ap.add_argument("--K-sweep", default="6,10,14")
    ap.add_argument("--n-restate", type=int, default=5)
    # LOSSY-STRESS cell: a lower-D / higher-K operating point where recovery is imperfect, so the reactivated
    # estimate the forget-step weakens is cross-talk-corrupted -> ungated (overwrite-always) writing damages the
    # neighbours and the PE gate's protection becomes load-bearing (a property the CLEAN base cell cannot exhibit).
    ap.add_argument("--D-stress", type=int, default=64)
    ap.add_argument("--K-stress", type=int, default=18)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "findings", "raw",
                                                  "_reconsolidation_superposed_lossy.json"))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    K_sweep = [int(k) for k in args.K_sweep.replace(",", " ").split()]

    results = [run_seed(s, args.D, args.K_base, K_sweep, args.n_restate, args.D_stress, args.K_stress)
               for s in seeds]

    def go_for(r):
        return (r["baseline_recall_at_base"] == 1.0 and r["rewrite_ok"] and r["isolation_ok"]
                and r["restabilize_ok"] and r["moat_ok"] and r["lesion_ok"] and r["permuted_ok"])

    n_go = sum(1 for r in results if go_for(r))

    print("\n===== RECONSOLIDATION ON A SUPERPOSED (LOSSY) STORE -- de-risk =====")
    print("D=%d  K_base=%d  K_sweep=%s  n_restate=%d  seeds=%s"
          % (args.D, args.K_base, K_sweep, args.n_restate, seeds))
    for r in results:
        print("\n-- seed %d --  baseline_recall@K%d=%.3f  gate=%.3f  fam_abstain=%.3f"
              % (r["seed"], args.K_base, r["baseline_recall_at_base"], r["pe_labile"], r["fam_abstain"]))
        print("   REWRITE=%s  ISOLATION=%s  restabilize=%s  moat=%s  lesion=%s  permuted=%s   [GO=%s]"
              % (r["rewrite_ok"], r["isolation_ok"], r["restabilize_ok"], r["moat_ok"],
                 r["lesion_ok"], r["permuted_ok"], go_for(r)))
        print("   correction PE=%.3f  vs  re-statement PE=%.3f  (gate %.3f)"
              % (r["correction_pe"], r["restatement_pe"], r["pe_labile"]))
        a = r["arms_restatement_stress"]
        print("   %d re-statements -> writes/others-recall:  recon=%d/%.3f  overwrite-always=%d/%.3f  "
              "naive-accum=%d/%.3f  do-nothing=%d/%.3f"
              % (args.n_restate,
                 a["reconsolidate"]["writes"], a["reconsolidate"]["others_recall_after_restatements"],
                 a["overwrite_always"]["writes"], a["overwrite_always"]["others_recall_after_restatements"],
                 a["naive_accumulate"]["writes"], a["naive_accumulate"]["others_recall_after_restatements"],
                 a["do_nothing"]["writes"], a["do_nothing"]["others_recall_after_restatements"]))
        print("   naive-accumulate on a real correction -> target=%s (clean rewrite=%s)"
              % (r["naive_accumulate_target_after_correction"], r["naive_accumulate_rewrite_clean"]))
        print("   capacity (K: baseline_recall / rewrite / isolation):  "
              + "  ".join("K%d:%.2f/%s/%.2f" % (c["K"], c["baseline_recall"], int(c["rewrite_ok"]),
                                                c["isolation_after_1_correction"]) for c in r["capacity"]))
        s = r["stress_cell"]
        sa = s["arms_restatement_stress"]
        print("   LOSSY-STRESS D%d/K%d: baseline_recall=%.2f  correction_PE=%.3f vs re-stmt_PE=%.3f (gate %.3f)  "
              "rewrite=%s isolation=%s restabilize=%s moat=%s"
              % (s["D"], s["K"], s["baseline_recall"], s["correction_pe"], s["restatement_pe"], s["pe_labile"],
                 s["rewrite_ok"], s["isolation_ok"], s["restabilize_ok"], s["moat_ok"]))
        print("      %d re-stmts -> writes/others-recall:  recon=%d/%.3f  overwrite-always=%d/%.3f  naive=%d/%.3f"
              % (args.n_restate,
                 sa["reconsolidate"]["writes"], sa["reconsolidate"]["others_recall_after_restatements"],
                 sa["overwrite_always"]["writes"], sa["overwrite_always"]["others_recall_after_restatements"],
                 sa["naive_accumulate"]["writes"], sa["naive_accumulate"]["others_recall_after_restatements"]))

    # treatment/control accounting AT THE CLEAN BASE CELL (control uncorrupted, so an attribution is meaningful).
    # Two DISTINCT, pre-registerable claims -- reported honestly whichever way they land:
    #   (1) does the PE GATE protect neighbours?  overwrite-always (ungated writes) vs reconsolidate (gated).
    #   (2) does the FORGET step protect neighbours?  naive-accumulate (no forget) vs reconsolidate (forget).
    def bdmg(r, mode):
        return 1.0 - r["base_cell"]["arms_restatement_stress"][mode]["others_recall_after_restatements"]
    dmg_over = float(np.mean([bdmg(r, "overwrite_always") for r in results]))
    dmg_recon = float(np.mean([bdmg(r, "reconsolidate") for r in results]))
    dmg_naive = float(np.mean([bdmg(r, "naive_accumulate") for r in results]))
    w_over = int(np.round(np.mean([r["base_cell"]["arms_restatement_stress"]["overwrite_always"]["writes"]
                                   for r in results])))
    w_recon = int(np.round(np.mean([r["base_cell"]["arms_restatement_stress"]["reconsolidate"]["writes"]
                                    for r in results])))
    print("\n-- treatment/control accounting @ CLEAN base cell (mean over seeds) --")
    lever("writes under re-statements (gate off->on)", w_over, w_recon, required=False)
    print("  claim 1 -- does the PE GATE protect neighbours? (overwrite-always -> reconsolidate)")
    attributable_to("neighbour-damage removed by the PE gate", dmg_over, dmg_recon)
    print("  claim 2 -- does the FORGET step protect neighbours? (naive-accumulate -> reconsolidate)")
    attributable_to("neighbour-damage removed by the forget step", dmg_naive, dmg_recon)

    print("\n===== %d/%d seeds GO =====" % (n_go, len(seeds)))

    summary = {"n_go": n_go, "n_seeds": len(seeds), "D": args.D, "K_base": args.K_base,
               "K_sweep": K_sweep, "n_restate": args.n_restate,
               "D_stress": args.D_stress, "K_stress": args.K_stress,
               "base_cell_damage_overwrite_always": dmg_over, "base_cell_damage_reconsolidate": dmg_recon,
               "base_cell_damage_naive_accumulate": dmg_naive,
               "results": results}
    outp = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote %s" % outp)


if __name__ == "__main__":
    main()
