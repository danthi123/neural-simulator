"""D3 CAPSTONE — the deployed brain answers "who was doing it BEFORE?" from a prior event **REMEMBERED BY SPIKES**,
with no state label anywhere.

EMERGENT + SPIKING-HOLD + DEPLOYED, in one register:
  * EMERGENT     -- the transition delta is learned from an agent-emission cross-entropy ALONE; the gate reads only the
                    OBSERVABLE clause code; the single slot->name read-out is fitted from clauses whose subject is
                    SPOKEN. NO (agent, patient) state label anywhere.
  * SPIKING HOLD -- the held prior event does NOT live in a Python variable. It lives in a PERSISTENT slow-NMDA
                    attractor (Wang 2002). Gate CLOSED = NO input at all -> the attractor sustains its own firing across
                    arbitrarily many clauses. Gate OPEN = CLEAR (an inhibitory reset longer than tau_NMDA) then LOAD --
                    exactly `sim/`'s `transmission_gate` semantics ("held normally CLOSED, opened on command").
  * DEPLOYED     -- it is a drop-in `event_register` on the real `MultiTurnAgent`, answering `who_agent_before()`.

The held agent is READ OUT OF SPIKES (whichever attractor pool is firing), then named by the label-free read-out.

ANTI-CHEATS (6-seed): (a) BEFORE >> a SINGLE-EVENT register (0.0 -- structurally cannot answer); (b) >> RECENCY and
>> naive "answer the current agent"; (c) a STATELESS held slot (the re-discretizer bridge every earlier rung used)
DEGRADES -- it cannot hold the prior event between clauses, which is the whole claim; (d) a GATE-LESION register (the
gate never opens, so nothing is ever shifted into the slot) COLLAPSES; (e) head-to-head against the rate gated-copy
deployment (0.711) and the replay deployment (0.367). Reuse-by-import; numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_gatedcopy_spiking_agent_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_selfsup_pair_derisk import make_pair_task
from research.runners._d3_event_gated_copy_derisk import train_gated_copy, _sm, _sig
from research.runners._d3_event_gatedcopy_agent_derisk import fit_slot_names_labelfree
from research.runners._d3_event_agent_derisk import D3EventRegister
from research.runners._d3_event_selfsup_pair_agent_derisk import make_discourse, COREF_W, PROMOTE_W
from research.runners._d3_persistent_slot_derisk import build_persistent_slot, _pool_idx, _reset
from research.runners.multi_turn_agent import MultiTurnAgent


def _run(sb, cur_vec, steps, idx, K):
    from sim.backend import to_host, from_host
    acc = np.zeros(K); dev = from_host(cur_vec)
    for _ in range(steps):
        sb.cp_external_input_current[:] = dev
        sb._run_one_simulation_step()
        fir = np.asarray(to_host(sb.cp_firing_states)).astype(float)
        for k in range(K):
            acc[k] += fir[idx[k]].mean()
    return acc / max(steps, 1)


class SpikingGatedCopyRegister:
    """The held prior event lives in a persistent slow-NMDA attractor. Drop-in for `D3EventRegister`
    (+ `mark_boundary`, `who_agent_prev`)."""

    def __init__(self, referents, seed=42, n_hid=128, epochs=40, recurrent=True, gate_lesion=False,
                 inter_clause=15, clear_steps=250, load_steps=80, clear_gain=1500.0, load_gain=400.0, read_steps=30):
        self.referents = list(referents); self.ref2idx = {r: i for i, r in enumerate(referents)}
        K = len(referents); self.K = K
        task = make_pair_task(seed, K=K)
        roll = train_gated_copy(task, seed=seed, n_hid=n_hid, epochs=epochs)
        self.W = roll.W; self.wg, self.bg = roll.gate
        self.ent, self.marks = task["ent"], task["marks"]
        self.ident = task["ident"]
        self.perm = fit_slot_names_labelfree(task, self.W, K)         # ONE label-free read-out for BOTH slots
        self.gate_lesion = bool(gate_lesion)
        self.inter_clause, self.clear_steps, self.load_steps = inter_clause, clear_steps, load_steps
        self.clear_gain, self.load_gain, self.read_steps = clear_gain, load_gain, read_steps

        self.sb = build_persistent_slot(seed, K, recur=(25.0 if recurrent else 0.0))
        self.idx = _pool_idx(self.sb, K)
        self.fs_idx = np.asarray(list(self.sb.region_manager.indices("fs")), dtype=int)
        self.n = self.sb.core_config.num_neurons
        self.zero = np.zeros(self.n, dtype=np.float64)
        self.reset()

    def reset(self):
        K = self.K
        _reset(self.sb)                                               # clears v/u/firing AND the conductances
        self.sc = np.zeros(K, np.float32); self.sc[self.ident] = 1.0
        self.pat = np.zeros(K, np.float32); self.pat[self.ident] = 1.0
        self.prev_winner = self.ident
        self._boundary = False

    def mark_boundary(self):
        self._boundary = True

    def is_pronoun_subject(self, word):
        w = (word or "").lower()
        return w in COREF_W or w in PROMOTE_W

    def observe(self, subject_word, object_word):
        o = self.ref2idx.get(object_word)
        if o is None:
            return
        sw = (subject_word or "").lower()
        if sw in COREF_W:
            sub = self.marks["HE"]
        elif sw in PROMOTE_W:
            sub = self.marks["IT"]
        else:
            s = self.ref2idx.get(sw)
            if s is None:
                return
            sub = self.ent[s]
        if self._boundary:
            mk = self.marks["RET"] if (sw in COREF_W or sw in PROMOTE_W) else self.marks["BND"]
        else:
            mk = self.marks["NOB"]
        self._boundary = False
        code = np.concatenate([mk, sub, self.ent[o]]).astype(np.float32)

        g = 0.0 if self.gate_lesion else float(_sig(code @ self.wg + self.bg))
        load_content = int(np.argmax(self.sc))                        # what a boundary copies: a_curr entering the clause
        if g > 0.5:                                                   # ---- gate OPEN: CLEAR then LOAD (a real WM update)
            cc = np.zeros(self.n, dtype=np.float64); cc[self.fs_idx] = self.clear_gain
            _run(self.sb, cc, self.clear_steps, self.idx, self.K)
            cl = np.zeros(self.n, dtype=np.float64); cl[self.idx[load_content]] = self.load_gain
            acc = _run(self.sb, cl, self.load_steps, self.idx, self.K)
        else:                                                         # ---- gate CLOSED: ZERO input; the attractor HOLDS
            assert not self.zero.any()
            acc = _run(self.sb, self.zero, self.inter_clause, self.idx, self.K)
        # a_prev is READ FROM SPIKES only. If the slot is SILENT it is EMPTY -- we must NOT fall back on a host variable,
        # or a stateless slot would be silently rescued by Python memory (measured: with a fallback the stateless control
        # scored identically to the spiking one, 0.833 vs 0.833 -- the exact fiction this rung exists to remove).
        self.prev_winner = int(np.argmax(acc)) if acc.max() > 1e-6 else self.ident

        emb, Wr, Wi, Wc, bc = (self.W["emb"], self.W["Wr"], self.W["Wi"], self.W["Wc"], self.W["bc"])
        sp_oh = np.zeros(self.K, np.float32); sp_oh[self.prev_winner] = 1.0
        h = np.tanh(np.concatenate([self.sc @ emb, sp_oh @ emb, self.pat @ emb]) @ Wr.T + code @ Wi.T)
        self.sc = _sm(h @ Wc.T + bc)
        self.pat = np.zeros(self.K, np.float32); self.pat[o] = 1.0

    def who_agent(self):
        return self.referents[int(self.perm[int(np.argmax(self.sc))])]

    def who_patient(self):
        return self.referents[int(np.argmax(self.pat))]

    def who_agent_prev(self):
        """Read the held event OUT OF SPIKES with zero input. NO host fallback: a silent slot means nothing is held."""
        read = _run(self.sb, self.zero, self.read_steps, self.idx, self.K)
        if read.max() <= 1e-6:
            return self.referents[int(self.perm[self.ident])]                # the slot is empty -- not a host memory
        return self.referents[int(self.perm[int(np.argmax(read))])]


def run_seed(seed, n_disc=20):
    referents = ["dog", "cat", "fish", "bird", "worm", "ball"]
    vocab = {w: None for w in (referents + ["chase"])}
    rng = np.random.RandomState(seed + 11)

    reg = SpikingGatedCopyRegister(referents, seed=seed)
    agent = MultiTurnAgent(referents, concepts=vocab, seed=seed, enable_biased_competition=True,
                           event_register=reg, enable_neural_render=False)
    stl = SpikingGatedCopyRegister(referents, seed=seed, recurrent=False)      # STATELESS held slot
    agent_stl = MultiTurnAgent(referents, concepts=vocab, seed=seed, enable_biased_competition=True,
                               event_register=stl, enable_neural_render=False)
    les = SpikingGatedCopyRegister(referents, seed=seed, gate_lesion=True)     # the gate never opens
    agent_les = MultiTurnAgent(referents, concepts=vocab, seed=seed, enable_biased_competition=True,
                               event_register=les, enable_neural_render=False)
    single = D3EventRegister(referents, seed=seed, spiking=False)
    agent_single = MultiTurnAgent(referents, concepts=vocab, seed=seed, enable_biased_competition=True,
                                  event_register=single, enable_neural_render=False)

    ok = ok_stl = ok_les = ok_single = ok_rec = ok_naive = ok_now = tot = tried = 0
    while tot < n_disc and tried < n_disc * 20:
        tried += 1
        clauses, true_now, true_before = make_discourse(rng, referents)
        if true_before == true_now or true_before == 0:
            continue
        reg.reset(); stl.reset(); les.reset(); single.reset()
        for c in clauses:
            agent.hear(c); agent_stl.hear(c); agent_les.hear(c); agent_single.hear(c)
        tb = referents[true_before]; tn = referents[true_now]
        ok += int(agent.who_agent_before() == tb)
        ok_stl += int(agent_stl.who_agent_before() == tb)
        ok_les += int(agent_les.who_agent_before() == tb)
        ok_single += int(agent_single.who_agent_before() == tb)
        ok_now += int(agent.who_agent_now() == tn)
        ok_naive += int(agent.who_agent_now() == tb)
        ok_rec += int(clauses[-1].split()[-1] == tb)
        tot += 1
    m = max(tot, 1)
    return {"seed": seed, "BEFORE_spiking": round(ok / m, 3), "BEFORE_stateless": round(ok_stl / m, 3),
            "BEFORE_gate_lesion": round(ok_les / m, 3), "BEFORE_single_event": round(ok_single / m, 3),
            "BEFORE_recency": round(ok_rec / m, 3), "BEFORE_naive_current": round(ok_naive / m, 3),
            "NOW_spiking": round(ok_now / m, 3), "n": tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-disc", type=int, default=20)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print("[D3 CAPSTONE] the deployed brain answers 'who was doing it BEFORE?' from a prior event REMEMBERED BY SPIKES (persistent slow-NMDA attractor), with no state label anywhere", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.n_disc); rows.append(r)
        print(f"  [seed {s}] BEFORE (spiking hold)={r['BEFORE_spiking']} || stateless-slot={r['BEFORE_stateless']} | gate-lesion={r['BEFORE_gate_lesion']} | "
              f"single-event={r['BEFORE_single_event']} | recency={r['BEFORE_recency']} | naive={r['BEFORE_naive_current']} || NOW={r['NOW_spiking']} (n={r['n']})", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        bp, bst, bl = _m("BEFORE_spiking"), _m("BEFORE_stateless"), _m("BEFORE_gate_lesion")
        bs_, br, bn, nw = _m("BEFORE_single_event"), _m("BEFORE_recency"), _m("BEFORE_naive_current"), _m("NOW_spiking")
        RATE_DEPLOY, REPLAY_DEPLOY, LABELLED = 0.711, 0.367, 0.928
        go = (bp > 0.5) and (bp - bst > 0.15) and (bp - bl > 0.25) and (bp - br > 0.3) and (bp - bn > 0.3)
        print(f"\n  AGGREGATE: BEFORE (spiking hold)={bp:.3f} | stateless slot={bst:.3f} | gate-lesion={bl:.3f} | single-event={bs_:.3f} | recency={br:.3f} | naive={bn:.3f} || NOW={nw:.3f}", flush=True)
        print(f"    references: rate gated-copy deployment {RATE_DEPLOY:.3f} | replay deployment {REPLAY_DEPLOY:.3f} | fully-LABELLED register {LABELLED:.3f}", flush=True)
        msg = ('the DEPLOYED agent answers who-was-doing-it-BEFORE (' + format(bp, '.2f') + ') from a prior event held in a '
               'PERSISTENT SLOW-NMDA ATTRACTOR -- gate CLOSED means ZERO input and the attractor sustains its own firing across '
               'arbitrarily many clauses; gate OPEN is a CLEAR-then-LOAD -- with the delta learned from agent-emission prediction '
               'ALONE and NO (agent,patient) state label anywhere. The held agent is READ OUT OF SPIKES. A STATELESS held slot -- the '
               're-discretizer every earlier rung used -- degrades (' + format(bst, '.2f') + '), a GATE-LESION register collapses ('
               + format(bl, '.2f') + '), a SINGLE-EVENT register cannot answer at all (' + format(bs_, '.2f') + '), and recency ('
               + format(br, '.2f') + ') + naive-current (' + format(bn, '.2f') + ') both fail -> EMERGENT + SPIKING-HOLD + DEPLOYED')
        bad = 'the spiking deployed register did not clearly beat its controls (read BEFORE vs stateless / gate-lesion / recency)'
        print("  VERDICT: " + ("GO" if go else "PARTIAL/NEGATIVE") + " -- " + (msg if go else bad) + ". NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
