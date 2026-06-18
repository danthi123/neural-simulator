"""ROADMAP PHASE 2 (the real "one brain"), the CUE-MATCHING SCAN over the persistent store -- the one operation STEP A3
needs that the multi-fact-store de-risk (GAP A) did not cover. GAP A fired a KNOWN trigger (query fact i by index); a
real who/what question ("who go north?") must find the MATCHING stored fact among K -- the resident equivalent of the
numpy composer's `_scan_first_match`/`query_patient`/`query_agent`.

3-role facts (agent, action, patient) live in ONE persistent bridge's complex weights (each a (1+D) trigger->readout
block, the GAP-A tiling extended from 2 to 3 roles via the validated 3-role coherence chain). A cue query scans the
blocks: for each block, reconstruct the composite (fire its trigger), unbind the CUE roles + cleanup, and if they ALL
match the cue, unbind + cleanup the ANSWER role -> the answer (first-match). If no block matches -> abstain (None) = the
no-confab moat over the store. All on-substrate; the host only supplies the cue codes (a programmatic text-in boundary)
and reads the winning concept.

GATE (exact/identity effect -> parity, 3 seeds x 2 D): on-bridge `query_patient`(agent,action)->patient AND
`query_agent`(action,patient)->agent over a K-fact store == the numpy `RFPhasorComposer` (`store`+`query_*`) == ground
truth, for EVERY stored fact. Anti-cheats: (i) the MOAT -- a cue present in NO stored fact -> abstain (None) [present
cues answer, absent abstain: the moat is a real discriminator, not always-None]; (ii) store-block LESION -> that fact's
cue stops matching (its recall collapses). Reuse-by-import (RFPhasorComposer + _build_rf_bridge); NO sim/ edit. GPU.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_cue_scan_derisk --seeds 42,43,44 --dims 64,128 --n-facts 8
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "cupy")

from sim.backend import to_host  # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer, _build_rf_bridge  # noqa: E402

AGENTS = ["dog", "cat", "bird", "river", "apple", "tree", "sun", "moon"]
ACTIONS = ["go", "come", "look", "stop", "swim", "walk", "run", "jump"]
PATIENTS = ["north", "east", "south", "west", "home", "hill", "lake", "sky"]
VOCAB = AGENTS + ACTIONS + PATIENTS
ROLES3 = ["agent", "action", "patient"]


class CueScanStore:
    """K 3-role facts tiled into ONE persistent bridge's complex weights; queried by CUE-matching scan."""

    def __init__(self, comp, n_unused):
        self.comp = comp; self.D = comp.D; self.V = len(VOCAB)
        self.n_unused = n_unused
        self.facts = []; self.store_conns = []; self.b = None

    def _layout(self, k_max):
        D = self.D
        self.work_base = 0                      # fill_0..2, bound_0..2, acc = 7 blocks
        self.store_base = 7 * D
        self.block = 1 + D
        self.q_base = self.store_base + k_max * self.block
        self.c_base = self.q_base + D
        return self.c_base + self.V

    def build(self, facts, seed):
        self.facts = list(facts)
        self._k_max = len(self.facts) + self.n_unused
        self.b = _build_rf_bridge(self._layout(self._k_max), seed)
        for i, fact in enumerate(self.facts):
            self._store_fact(i, fact)
        return self

    def n(self):
        return self.c_base + self.V

    def _store_fact(self, i, fact):
        """Build fact i's 3-role composite on-bridge (bind agent+action+patient, bundle into acc), append block i."""
        comp, b, D, P = self.comp, self.b, self.D, self.comp.period
        binds, bundle = [], []
        kick = np.zeros(self.n(), dtype=np.complex128)
        for ri, role in enumerate(ROLES3):
            zr = comp._to_phasor(comp.roles[role]); zf = comp._to_phasor(comp.concepts[fact[ri]])
            kick[ri * D:(ri + 1) * D] = zf
            binds += [((3 + ri) * D + k, ri * D + k, complex(zr[k])) for k in range(D)]
            bundle += [(6 * D + k, (3 + ri) * D + k, 1.0) for k in range(D)]
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
        b.rf_set_complex_weights(binds); b.rf_kick(kick, period=P, lam=0.0); b.rf_resonate_steps(P + 8)
        b.rf_set_complex_weights(bundle); b.rf_resonate_steps(P + 8)
        zc = comp._to_phasor(np.asarray(b.rf_read_phases())[6 * D:7 * D])
        trig = self.store_base + i * self.block
        self.store_conns += [(trig + 1 + k, trig, complex(zc[k])) for k in range(D)]

    def _reconstruct(self, block_idx, store):
        """Fire block_idx's trigger -> readout block reconstructs its composite (left in the readout register state)."""
        comp, b, D, P = self.comp, self.b, self.D, self.comp.period
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
        trig = self.store_base + block_idx * self.block
        kick = np.zeros(self.n(), dtype=np.complex128); kick[trig] = 1.0
        b.rf_set_complex_weights(store); b.rf_kick(kick, period=P, lam=0.0, neuron_mask=None); b.rf_resonate_steps(P + 8)
        return trig

    def _read_role(self, trig, role):
        """Unbind `role` from the already-reconstructed readout block (at trig+1..trig+D) -> cleanup -> (word, peak)."""
        comp, b, D, P, V = self.comp, self.b, self.D, self.comp.period, self.V
        zr_conj = np.conj(comp._to_phasor(comp.roles[role]))
        unbind = [(self.q_base + k, trig + 1 + k, complex(zr_conj[k])) for k in range(D)]
        b.rf_set_complex_weights(unbind); b.rf_resonate_steps(P + 8)
        clean = []
        for j in range(V):
            cc = np.conj(comp._to_phasor(comp.concepts[VOCAB[j]]))
            clean += [(self.c_base + j, self.q_base + k, complex(cc[k])) for k in range(D)]
        b.rf_set_complex_weights(clean); b.rf_resonate_steps(1)
        scores = np.maximum(np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)[self.c_base:self.c_base + V], 0.0)
        return VOCAB[int(np.argmax(scores))], float(scores.max())

    def query(self, cue, answer_role, lesion_block=None):
        """Scan the stored blocks; cue is {role: word}. Return the first fact's answer_role whose cue roles ALL match,
        else None (abstain). Reconstructing a block then reading multiple roles reuses the readout register (lam=0)."""
        store = self.store_conns if lesion_block is None else [
            c for c in self.store_conns
            if not (self.store_base + lesion_block * self.block <= c[1] < self.store_base + (lesion_block + 1) * self.block)]
        for i in range(len(self.facts)):
            trig = self._reconstruct(i, store)
            match = True
            for role, want in cue.items():
                w, _ = self._read_role(trig, role)
                # re-fire the trigger before the next read (the unbind/cleanup installs overwrote the store weights, and
                # the readout register self-rotated): cheapest correct path is reconstruct-per-read.
                self._reconstruct(i, store)
                if w != want:
                    match = False
                    break
            if match:
                ans, _ = self._read_role(trig, answer_role)
                return ans
        return None


def run_seed(seed, D, n_facts, n_unused):
    comp = RFPhasorComposer(seed=seed, D=D, vocab=VOCAB, period=200)
    # diagonal-distinct facts: (AGENTS[i], ACTIONS[i], PATIENTS[i]) -> every (agent,action) AND (action,patient) cue is
    # UNIQUE, so query_patient/query_agent are unambiguous (first-match == oracle). n_facts <= len(VOCAB lists) = 8.
    n_facts = min(n_facts, len(AGENTS))
    facts = [(AGENTS[i], ACTIONS[i], PATIENTS[i]) for i in range(n_facts)]
    store = CueScanStore(comp, n_unused=n_unused).build(facts, seed)
    # numpy oracle: same facts
    for (a, v, p) in facts:
        comp.store(a, v, p)

    okp = oka = hostp = hosta = 0
    for (a, v, p) in facts:
        ans_p = store.query({"agent": a, "action": v}, "patient")     # query_patient
        ans_a = store.query({"action": v, "patient": p}, "agent")     # query_agent
        okp += int(ans_p == p); oka += int(ans_a == a)
        hostp += int(ans_p == comp.query_patient(a, v)); hosta += int(ans_a == comp.query_agent(v, p))
    n = len(facts)

    # MOAT: a cue present in NO stored fact -> abstain (None). Build an unused (agent,action) combo.
    used = {(a, v) for (a, v, p) in facts}
    absent = next(((a, v) for a in AGENTS for v in ACTIONS if (a, v) not in used), None)
    moat_abstain = 1
    if absent is not None:
        moat_abstain = int(store.query({"agent": absent[0], "action": absent[1]}, "patient") is None)
    # present cue answers (not None) — the abstain is a real discriminator
    present_answers = int(store.query({"agent": facts[0][0], "action": facts[0][1]}, "patient") is not None)

    # LESION: drop fact 0's block -> its cue stops matching (query returns a DIFFERENT fact or abstains).
    les_ans = store.query({"agent": facts[0][0], "action": facts[0][1]}, "patient", lesion_block=0)
    lesion_collapse = int(les_ans != facts[0][2])

    row = {"seed": seed, "D": D, "n_facts": n_facts, "qpatient": okp / n, "qagent": oka / n,
           "host_p": hostp / n, "host_a": hosta / n, "moat_abstain": moat_abstain,
           "present_answers": present_answers, "lesion_collapse": lesion_collapse}
    print(f"  [seed {seed} D={D} K={n_facts}] query_patient={okp/n:.2f} query_agent={oka/n:.2f} | host_p={hostp/n:.2f} "
          f"host_a={hosta/n:.2f} | moat abstain={moat_abstain} present={present_answers} | lesion_collapse={lesion_collapse}",
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44"); ap.add_argument("--dims", type=str, default="64,128")
    ap.add_argument("--n-facts", type=int, default=8); ap.add_argument("--n-unused", type=int, default=2)
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_phaseB_onebrain_cue_scan.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]; dims = [int(d) for d in args.dims.split(",")]
    t0 = time.time()
    print(f"[one-brain cue-scan de-risk] does a CUE query find the matching fact among K={args.n_facts} stored on ONE "
          f"persistent bridge == the numpy composer, abstaining on an absent cue?\n", flush=True)
    rows = [run_seed(s, D, args.n_facts, args.n_unused) for s in seeds for D in dims]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    qp, qa, hp, ha = m("qpatient"), m("qagent"), m("host_p"), m("host_a")
    moat, present, les = m("moat_abstain"), m("present_answers"), m("lesion_collapse")
    n_full = sum(int(r["qpatient"] >= 0.99 and r["qagent"] >= 0.99 and r["host_p"] >= 0.99 and r["host_a"] >= 0.99
                     and r["moat_abstain"] >= 1 and r["present_answers"] >= 1) for r in rows)
    go = (n_full == len(rows)) and (moat >= 0.99) and (present >= 0.99) and (les >= 0.99)
    print(f"\n{'='*104}", flush=True)
    print(f"  MEAN ({len(rows)} seed*D, K={args.n_facts}): query_patient {qp:.3f} query_agent {qa:.3f} | host-parity "
          f"p {hp:.3f}/a {ha:.3f} | moat abstain {moat:.2f} present {present:.2f} | lesion_collapse {les:.2f} | "
          f"per-row full: {n_full}/{len(rows)}", flush=True)
    if go:
        print(f"  GO: a CUE query finds the matching fact among K={args.n_facts} stored on ONE persistent bridge -- "
              f"who/what recall == the numpy composer == ground truth every fact, an ABSENT cue abstains (the moat over "
              f"the store), a present cue answers, and a store-block lesion collapses that fact's cue. ==> the "
              f"cue-matching scan works on-substrate; STEP A3 (the production OneBrainComposer) can assemble parser "
              f"front-end + persistent store + cue-scan + moat.", flush=True)
    elif qp >= 0.95 and qa >= 0.95:
        print(f"  BOUNDARY: recall holds (qp {qp:.3f}/qa {qa:.3f}) but the moat ({moat:.2f}) or lesion ({les:.2f}) is "
              f"soft -- localize the abstain threshold / first-match. Reportable.", flush=True)
    else:
        print(f"  NEGATIVE: cue-scan recall qp {qp:.3f}/qa {qa:.3f} -- the matching scan does not recover on-substrate; "
              f"diagnose (per-block reconstruct phase / the match cleanup). Reportable.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*104}", flush=True)
    out = {"verdict": "GO" if go else ("BOUNDARY" if (qp >= 0.95 and qa >= 0.95) else "NEGATIVE"),
           "seeds": seeds, "dims": dims, "n_facts": args.n_facts, "qpatient": qp, "qagent": qa, "host_p": hp,
           "host_a": ha, "moat_abstain": moat, "present_answers": present, "lesion_collapse": les, "per_row": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
