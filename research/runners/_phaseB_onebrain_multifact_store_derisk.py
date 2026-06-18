"""ROADMAP PHASE 2 (the real "one brain"), STEP A1 -- the MULTI-FACT persistent synapse-store on ONE bridge. This is
GAP A from the production scoping (`2026-06-18-production-one-brain-composer-scoping.md`), the recommended FIRST
cheap-first de-risk: smaller than the parser front-end, reuses the validated `_store_substrate` mechanism.

The prior de-risks store ONE fact in an RF register (`v`/`u`) and query it immediately -- but a work-register reset
erases register state. A real knowledge base holds MANY facts that persist across turns AND across the per-op
work-register resets. The fix (the CYCLE-168 insight): stored facts live in SYNAPSES, not register state. This de-risk
builds that store: K facts, each occupying its own `(1+D)` trigger->readout block tiled into ONE persistent bridge's
complex weights, register-reset-safe, queried on-substrate.

Layout (D each unless noted; 2-role facts agent+action, reusing the GO step-2 bind+bundle chain to BUILD each composite
on-bridge): work registers a_in[0] v_in[1] a_bnd[2] v_bnd[3] acc[4]; then a STORE region of K_max tiled (1+D) blocks
(block i: trigger at store_base+i*(1+D), readout the next D); then a query register Q and V concept-score neurons.

STORE fact i: reset the work registers, bind(agent)+bind(action)+bundle into `acc` (the GO step-2 chain), read acc's
phasor, and APPEND block i's trigger->readout complex weights (`zc_i[k]` = the composite phasor) to the accumulated
store weights. The composite now lives in synapses; a register reset cannot erase it.
QUERY fact i, role r (on-substrate, all register state ZEROED first): install the WHOLE accumulated store, fire trigger
i (unit phasor) -> readout block i reconstructs composite i (other blocks' triggers at 0 -> their readouts stay 0, the
per-block isolation); swap to the unbind synapse readout_i->Q (conj role r) -> Q recovers the role; swap to the cleanup
synapse Q->concepts (conj codebook) -> the concept membranes are the matched-filter scores; argmax = the answer, max =
the familiarity peak.

GATE (exact/identity effect -> parity, 3 seeds x 2 D): every stored fact's BOTH roles recall the correct filler ==
the numpy `RFPhasorComposer` oracle (`_encode`+`_unbind_phases`+`_cleanup`), AND the work-register-reset invariant holds
(the query ZEROES all register state before each recall, so a correct recall proves the fact is in synapses).
Anti-cheats: (i) STORE-BLOCK LESION -- drop fact j's block weights -> fact j's recall collapses, the others intact (the
on-substrate store is load-bearing, not residual register state); (ii) the MOAT -- query an UNUSED store block (no
weights) -> the cleanup peak is LOW (abstain) vs a stored fact's HIGH peak (answer), clean separation at a measured
midpoint threshold (the no-confab moat, never weakened). Reuse-by-import (RFPhasorComposer + _build_rf_bridge);
ADDITIVE runner, NO sim/ edit. GPU.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_multifact_store_derisk --seeds 42,43,44 --n-facts 8
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

AGENTS = ["dog", "cat", "bird", "river", "apple"]
ACTIONS = ["go", "come", "look", "stop", "swim"]
VOCAB = AGENTS + ACTIONS


class MultiFactStore:
    """K facts tiled into ONE persistent bridge's complex weights; register-reset-safe; queried on-substrate."""

    def __init__(self, comp, n_unused):
        self.comp = comp
        self.D = comp.D
        self.V = len(VOCAB)
        self.n_unused = n_unused          # extra empty store blocks (for the moat: firing them -> abstain)
        self.facts = []                   # list of dicts (agent, action)
        self.store_conns = []             # accumulated trigger->readout complex weights for ALL stored facts
        self.b = None
        self._k_max = None

    # ---- layout helpers ----
    def _layout(self, k_max):
        D = self.D
        self.work_base = 0                            # a_in,v_in,a_bnd,v_bnd,acc = 5 blocks
        self.store_base = 5 * D
        self.block = 1 + D                            # per-fact: 1 trigger + D readout
        self.q_base = self.store_base + k_max * self.block
        self.c_base = self.q_base + D
        return self.c_base + self.V

    def build(self, facts, seed):
        """Build the persistent bridge sized for len(facts)+n_unused blocks, then STORE every fact on-substrate."""
        self.facts = list(facts)
        self._k_max = len(self.facts) + self.n_unused
        n_total = self._layout(self._k_max)
        self.b = _build_rf_bridge(n_total, seed)
        for i, fact in enumerate(self.facts):
            self._store_fact(i, fact)
        return self

    def _store_fact(self, i, fact):
        """Build fact i's composite ON-BRIDGE (bind agent + bind action + bundle into acc, the GO step-2 chain), read
        the composite phasor, and append block i's trigger->readout weights."""
        comp, b, D, P = self.comp, self.b, self.D, self.comp.period
        za_role = comp._to_phasor(comp.roles["agent"]); zv_role = comp._to_phasor(comp.roles["action"])
        za_fill = comp._to_phasor(comp.concepts[fact["agent"]]); zv_fill = comp._to_phasor(comp.concepts[fact["action"]])
        bind_a = [(2 * D + k, 0 * D + k, complex(za_role[k])) for k in range(D)]
        bind_v = [(3 * D + k, 1 * D + k, complex(zv_role[k])) for k in range(D)]
        bundle = ([(4 * D + k, 2 * D + k, 1.0) for k in range(D)] + [(4 * D + k, 3 * D + k, 1.0) for k in range(D)])
        kick = np.zeros(self.b_n(), dtype=np.complex128)
        kick[0 * D:1 * D] = za_fill; kick[1 * D:2 * D] = zv_fill
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0   # reset work registers (CYCLE-168)
        b.rf_set_complex_weights(bind_a + bind_v); b.rf_kick(kick, period=P, lam=0.0); b.rf_resonate_steps(P + 8)
        b.rf_set_complex_weights(bundle); b.rf_resonate_steps(P + 8)
        acc_phases = np.asarray(b.rf_read_phases())[4 * D:5 * D]
        zc = comp._to_phasor(acc_phases)                                        # the composite phasor (in synapses now)
        trig = self.store_base + i * self.block
        self.store_conns += [(trig + 1 + k, trig, complex(zc[k])) for k in range(D)]

    def b_n(self):
        return self.c_base + self.V

    def query(self, block_idx, role, lesion_block=None):
        """On-substrate query: reset ALL register state, install the (optionally lesioned) store, fire block_idx's
        trigger -> reconstruct -> unbind `role` -> cleanup. Returns (answer_word, peak_score)."""
        comp, b, D, P, V = self.comp, self.b, self.D, self.comp.period, self.V
        zr_conj = np.conj(comp._to_phasor(comp.roles[role]))
        store = self.store_conns if lesion_block is None else [
            c for c in self.store_conns
            if not (self.store_base + lesion_block * self.block <= c[1] < self.store_base + (lesion_block + 1) * self.block)]
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0   # ZERO all register state (reset-safety)
        trig = self.store_base + block_idx * self.block
        kick = np.zeros(self.b_n(), dtype=np.complex128); kick[trig] = 1.0      # unit phasor fires the trigger
        b.rf_set_complex_weights(store); b.rf_kick(kick, period=P, lam=0.0); b.rf_resonate_steps(P + 8)  # reconstruct
        unbind = [(self.q_base + k, trig + 1 + k, complex(zr_conj[k])) for k in range(D)]
        b.rf_set_complex_weights(unbind); b.rf_resonate_steps(P + 8)            # readout_i -> Q (role recovered)
        clean = []
        for j in range(V):
            cc = np.conj(comp._to_phasor(comp.concepts[VOCAB[j]]))
            clean += [(self.c_base + j, self.q_base + k, complex(cc[k])) for k in range(D)]
        b.rf_set_complex_weights(clean); b.rf_resonate_steps(1)                 # Q -> concept scores
        scores = np.maximum(np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)[self.c_base:self.c_base + V], 0.0)
        return VOCAB[int(np.argmax(scores))], float(scores.max())


def run_seed(seed, D, n_facts, n_unused):
    comp = RFPhasorComposer(seed=seed, D=D, vocab=VOCAB, period=200)
    facts = [{"agent": a, "action": v} for a, v in itertools.islice(itertools.product(AGENTS, ACTIONS), n_facts)]
    store = MultiFactStore(comp, n_unused=n_unused).build(facts, seed)

    ok_truth = ok_host = 0
    stored_peaks, unused_peaks = [], []
    for i, fact in enumerate(facts):
        for role in ("agent", "action"):
            ans, peak = store.query(i, role)
            truth = fact[role]
            host = comp._cleanup(comp._unbind_phases(comp._encode(fact), role), VOCAB)
            ok_truth += int(ans == truth); ok_host += int(ans == host)
            stored_peaks.append(peak)
    n = 2 * len(facts)

    # MOAT anti-cheat: fire UNUSED store blocks (no weights) -> low peak -> abstain.
    for u in range(len(facts), len(facts) + n_unused):
        _, peak = store.query(u, "agent")
        unused_peaks.append(peak)

    bmin, umax = (min(stored_peaks), max(unused_peaks)) if unused_peaks else (1.0, 0.0)
    thr = 0.5 * (np.mean(stored_peaks) + (np.mean(unused_peaks) if unused_peaks else 0.0))
    moat_sep = int(bmin > umax) if unused_peaks else 1

    # LESION anti-cheat: drop fact 0's store block -> its recall PEAK collapses (a lesioned block has no weights, so it
    # behaves like an unused block: peak below the moat threshold = abstain). We score by the PEAK, not the argmax,
    # because an all-zero cleanup defaults argmax to index 0 (== "dog" == fact-0's agent here) -- a false "recall".
    _, les0_peak = store.query(0, "agent", lesion_block=0)
    les0 = int(les0_peak > thr)                         # want 0 (lesioned fact -> peak collapses -> abstain)
    intact1_ans, intact1_peak = store.query(1, "agent", lesion_block=0)
    intact1 = int(intact1_ans == facts[1]["agent"] and intact1_peak > thr)  # want 1 (other facts intact + confident)
    row = {"seed": seed, "D": D, "n_facts": n_facts, "recall_truth": ok_truth / n, "recall_host": ok_host / n,
           "lesion_recall": les0, "intact_after_lesion": intact1, "moat_sep": moat_sep,
           "stored_peak_min": bmin, "unused_peak_max": umax}
    print(f"  [seed {seed} D={D} K={n_facts}] recall=={ok_truth/n:.2f} truth/{ok_host/n:.2f} host | "
          f"lesion {les0} intact {intact1} | moat sep={moat_sep} (stored>={bmin:.3g} unused<={umax:.3g})", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--dims", type=str, default="64,128")
    ap.add_argument("--n-facts", type=int, default=8)
    ap.add_argument("--n-unused", type=int, default=3)
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_phaseB_onebrain_multifact_store.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]; dims = [int(d) for d in args.dims.split(",")]
    t0 = time.time()
    print(f"[one-brain multi-fact store de-risk] do K={args.n_facts} facts live in ONE persistent bridge's synapses, "
          f"register-reset-safe, queried on-substrate == the numpy oracle?\n", flush=True)
    rows = [run_seed(s, D, args.n_facts, args.n_unused) for s in seeds for D in dims]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    rt, rh = m("recall_truth"), m("recall_host")
    les, intact, sep = m("lesion_recall"), m("intact_after_lesion"), m("moat_sep")
    n_full = sum(int(r["recall_truth"] >= 0.99 and r["recall_host"] >= 0.99) for r in rows)
    go = (n_full == len(rows)) and (les <= 0.01) and (intact >= 0.99) and (sep >= 0.99)
    print(f"\n{'='*108}", flush=True)
    print(f"  MEAN ({len(rows)} seed*D, K={args.n_facts}): recall {rt:.3f} truth / {rh:.3f} host | "
          f"lesioned-fact recall {les:.2f} (want 0) | intact-after-lesion {intact:.2f} | moat clean-sep {sep:.2f} | "
          f"per-row full: {n_full}/{len(rows)}", flush=True)
    if go:
        print(f"  GO: K={args.n_facts} facts live in ONE persistent bridge's COMPLEX SYNAPSES -- every fact's both "
              f"roles recall == the numpy oracle with ALL register state ZEROED before each query (so the fact is in "
              f"synapses, not register state), a store-block lesion collapses ONLY that fact, and the moat abstains on "
              f"unused blocks (clean separation). ==> GAP A resolved at K={args.n_facts}: the multi-fact store is "
              f"register-reset-safe + on-substrate-queryable. Next: scale K (16/32) + the parser front-end (GAP B).", flush=True)
    elif rh >= 0.95 and sep >= 0.99:
        print(f"  BOUNDARY: recall vs host {rh:.3f} (substrate-faithful) but vs truth {rt:.3f} -- the FHRR store "
              f"capacity / cross-talk degrades at K={args.n_facts}; cap K + shard (the validated 320-concept route) or "
              f"raise D. Moat + lesion hold. Reportable.", flush=True)
    else:
        print(f"  SUBSTRATE-COST/NEGATIVE: recall truth {rt:.3f} host {rh:.3f}, lesion {les:.2f}, intact {intact:.2f}, "
              f"moat sep {sep:.2f} -- diagnose (reset aliasing if intact<1; per-block phase drift if recall<host; moat "
              f"fragility if sep<1). The host-orchestrated numpy kb stays the default. Reportable finding.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*108}", flush=True)
    out = {"verdict": "GO" if go else ("BOUNDARY" if (rh >= 0.95 and sep >= 0.99) else "SUBSTRATE-COST"),
           "seeds": seeds, "dims": dims, "n_facts": args.n_facts, "n_unused": args.n_unused,
           "recall_truth": rt, "recall_host": rh, "lesion_recall": les, "intact_after_lesion": intact,
           "moat_sep": sep, "per_row": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
