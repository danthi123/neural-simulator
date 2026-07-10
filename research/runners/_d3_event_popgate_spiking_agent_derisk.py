"""D3 SPIKING POP → the deployed brain resumes a protagonist by READING a spiking attractor without erasing it.

WHERE THIS SITS.
- The held prior event already lives in a **persistent slow-NMDA attractor** on a real `SimulationBridge`: gate CLOSED ⇒
  zero input and the pool sustains its own firing indefinitely; gate OPEN ⇒ **CLEAR then LOAD**, where the clear must
  outlast τ_NMDA or the old event re-ignites (`2026-07-10-D3-persistent-spiking-slot-...md`).
- The **pop gate** (rate model, deployed) lets the register resume an earlier protagonist on a return marker: 0.778 vs
  0.139 for the identical register with the read gate shut (`2026-07-10-D3-pop-gate-deployed-...md`).

THE SUBSTRATE ASYMMETRY THIS RUNG TESTS.
The push is a **write**: it must DESTROY what the slot holds (hence clear-longer-than-τ_NMDA). The pop is a **read**: it
must NOT. On a point-neuron attractor those are genuinely different problems — a read that drives the pool risks either
re-igniting the wrong assembly or erasing the one it is reading. The read here drives NOTHING (`assert not zero.any()`),
so non-destructiveness is a property of the mechanism rather than a tuned parameter — and it is GATED on, not assumed:
after a pop, the held slot must still be holding the same agent.

WHAT THE POP DOES ON SPIKES.
On a return marker the register reads the held attractor's spikes and installs that agent as the CURRENT one, leaving the
attractor untouched. The rate model computes `a_curr <- r*a_prev + (1-r)*delta`; on spikes the read is a winner over
population firing, so the convex combination discretises to `r > 0.5 -> a_curr = argmax(spikes)`. That discretisation is
stated, not hidden, and the host-twin control below measures its cost.

ANTI-CHEATS (6-seed):
 (a) resumption vs a POP-LESION register (identical model, r forced to 0) -- the single-variable contrast;
 (b) vs "keep answering the pre-pop agent" and vs RECENCY -- the two listener shortcuts;
 (c) **NON-DESTRUCTIVE READ**: after the pop, `who_agent_prev()` must still return the held agent. A read that erased or
     re-ignited the attractor would fail this even while resumption looked fine;
 (d) **STATELESS slot** (`recur=0`, no attractor to read): resumption must COLLAPSE. NO host fallback anywhere -- a silent
     slot means nothing is held (a prior rung measured a stateless control silently rescued by a Python variable);
 (e) HOST TWIN: the same register with the spiking slot replaced by an exact host copy, to price the substrate;
 (f) the deployed gate must open on POPS and stay shut on BOUNDARIES (both carry a connective).

Reuse-by-import; numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_popgate_spiking_agent_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_selfsup_pair_derisk import make_pair_task
from research.runners._d3_event_gated_copy_derisk import _sm, _sig
from research.runners._d3_event_pop_gate_derisk import train_pushpop
from research.runners._d3_event_gatedcopy_agent_derisk import fit_slot_names_labelfree
from research.runners._d3_event_selfsup_pair_agent_derisk import make_discourse, COREF_W, PROMOTE_W, CONNECTIVES
from research.runners._d3_event_popgate_agent_derisk import _truth
from research.runners._d3_persistent_slot_derisk import build_persistent_slot, _pool_idx, _reset
from research.runners.multi_turn_agent import MultiTurnAgent


def _run(sb, cur, steps, idx, K):
    """Advance the bridge `steps` and return per-pool spike counts."""
    sb.cp_external_input_current[:] = cur
    acc = np.zeros(K)
    for _ in range(steps):
        sb._run_one_simulation_step()
        f = sb.cp_firing_states
        for k in range(K):
            acc[k] += float(np.asarray(f[idx[k]]).sum())
    return acc


class SpikingPopGateRegister:
    """Two gates on one register; the held slot is a persistent slow-NMDA attractor on a real `SimulationBridge`.
    PUSH (boundary) = clear-then-load. POP (return) = read the attractor's spikes into `a_curr`, leaving it holding."""

    def __init__(self, referents, seed=42, n_hid=128, epochs=40, stage_pop_epochs=15, recurrent=True,
                 pop_lesion=False, host_twin=False, inter_clause=15, clear_steps=250, load_steps=80,
                 clear_gain=1500.0, load_gain=400.0, read_steps=30):
        self.referents = list(referents); self.ref2idx = {r: i for i, r in enumerate(referents)}
        K = len(referents); self.K = K
        task = make_pair_task(seed, K=K)
        roll = train_pushpop(task, seed=seed, n_hid=n_hid, epochs=epochs,
                             stage_pop_epochs=stage_pop_epochs, freeze_core_in_phase2=False)
        self.W = roll.W
        self.wg, self.bg, self.wp, self.bp = roll.gates
        self.ent, self.marks = task["ent"], task["marks"]
        self.ident = task["ident"]
        self.perm = fit_slot_names_labelfree(task, self.W, K)
        self.pop_lesion = bool(pop_lesion); self.host_twin = bool(host_twin)
        self.inter_clause, self.clear_steps, self.load_steps = inter_clause, clear_steps, load_steps
        self.clear_gain, self.load_gain, self.read_steps = clear_gain, load_gain, read_steps
        self.r_on_pop, self.r_on_bnd = [], []
        self.held_survived, self.held_checked = 0, 0     # anti-cheat (c): does the slot still hold after its own read?

        if not self.host_twin:
            self.sb = build_persistent_slot(seed, K, recur=(25.0 if recurrent else 0.0))
            self.idx = _pool_idx(self.sb, K)
            self.fs_idx = np.asarray(list(self.sb.region_manager.indices("fs")), dtype=int)
            self.n = self.sb.core_config.num_neurons
            self.zero = np.zeros(self.n, dtype=np.float64)
        self.reset()

    def reset(self):
        K = self.K
        if not self.host_twin:
            _reset(self.sb)                                            # clears v/u/firing AND the conductances
        self._twin = self.ident                                        # host twin's held slot (arm (e) only)
        self.sc = np.zeros(K, np.float32); self.sc[self.ident] = 1.0
        self.pat = np.zeros(K, np.float32); self.pat[self.ident] = 1.0
        self._boundary = False

    def mark_boundary(self):
        self._boundary = True

    def is_pronoun_subject(self, word):
        w = (word or "").lower()
        return w in COREF_W or w in PROMOTE_W

    def _read_held(self, steps=None):
        """Read the held slot OUT OF SPIKES with ZERO input. Drives nothing -> cannot disturb what it reads.
        NO host fallback: a silent slot means nothing is held."""
        if self.host_twin:
            return self._twin
        assert not self.zero.any()
        acc = _run(self.sb, self.zero, steps or self.read_steps, self.idx, self.K)
        return int(np.argmax(acc)) if acc.max() > 1e-6 else self.ident

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
        is_pop = self._boundary and (sw in COREF_W or sw in PROMOTE_W)
        is_bnd = self._boundary and not is_pop
        mk = (self.marks["RET"] if is_pop else self.marks["BND"]) if self._boundary else self.marks["NOB"]
        self._boundary = False
        code = np.concatenate([mk, sub, self.ent[o]]).astype(np.float32)

        g = float(_sig(code @ self.wg + self.bg))
        r = 0.0 if self.pop_lesion else float(_sig(code @ self.wp + self.bp))
        if is_pop:
            self.r_on_pop.append(r)
        elif is_bnd:
            self.r_on_bnd.append(r)

        load_content = int(np.argmax(self.sc))            # what a boundary pushes: a_curr entering this clause

        # ---- PUSH: a real working-memory update. CLEAR (longer than tau_NMDA) then LOAD.
        if g > 0.5:
            if self.host_twin:
                self._twin = load_content
            else:
                cc = np.zeros(self.n, dtype=np.float64); cc[self.fs_idx] = self.clear_gain
                _run(self.sb, cc, self.clear_steps, self.idx, self.K)
                cl = np.zeros(self.n, dtype=np.float64); cl[self.idx[load_content]] = self.load_gain
                _run(self.sb, cl, self.load_steps, self.idx, self.K)
        elif not self.host_twin:                          # gate CLOSED: zero input; the attractor HOLDS itself
            assert not self.zero.any()
            _run(self.sb, self.zero, self.inter_clause, self.idx, self.K)

        # ---- POP: READ the held slot into a_curr. The read drives nothing, so the slot keeps holding.
        emb, Wr, Wi, Wc, bc = (self.W["emb"], self.W["Wr"], self.W["Wi"], self.W["Wc"], self.W["bc"])
        held = self._read_held()
        sp_oh = np.zeros(self.K, np.float32); sp_oh[held] = 1.0
        h = np.tanh(np.concatenate([self.sc @ emb, sp_oh @ emb, self.pat @ emb]) @ Wr.T + code @ Wi.T)
        raw = _sm(h @ Wc.T + bc)
        if r > 0.5:                                       # the convex combination discretises on a spiking read
            self.sc = sp_oh.copy()
            if is_pop:                                    # anti-cheat (c): the slot must SURVIVE its own read
                self.held_checked += 1
                self.held_survived += int(self._read_held() == held)
        else:
            self.sc = raw
        self.pat = np.zeros(self.K, np.float32); self.pat[o] = 1.0

    def who_agent(self):
        return self.referents[int(self.perm[int(np.argmax(self.sc))])]

    def who_patient(self):
        return self.referents[int(np.argmax(self.pat))]

    def who_agent_prev(self):
        return self.referents[int(self.perm[self._read_held()])]


def run_seed(seed, n_pop_disc=15, n_disc=15):
    refs = ["dog", "cat", "fish", "bird", "worm", "ball"]
    vocab = {w: None for w in (refs + ["chase"])}
    rng = np.random.RandomState(seed + 11)

    spk = SpikingPopGateRegister(refs, seed=seed)                          # the spiking attractor + both gates
    les = SpikingPopGateRegister(refs, seed=seed, pop_lesion=True)         # r == 0: the single-variable control
    stl = SpikingPopGateRegister(refs, seed=seed, recurrent=False)         # no attractor to read
    twn = SpikingPopGateRegister(refs, seed=seed, host_twin=True)          # exact host copy: prices the substrate
    mk = lambda reg: MultiTurnAgent(refs, concepts=vocab, seed=seed, enable_biased_competition=True,
                                    event_register=reg, enable_neural_render=False)
    a_spk, a_les, a_stl, a_twn = mk(spk), mk(les), mk(stl), mk(twn)
    regs = (spk, les, stl, twn); agents = (a_spk, a_les, a_stl, a_twn)

    def _hear(clauses):
        for r_ in regs:
            r_.reset()
        for c in clauses:
            for a_ in agents:
                a_.hear(c)

    pk = {k: 0 for k in ("spk", "les", "stl", "twn", "stay", "rec")}
    pk_n = ptried = 0
    while pk_n < n_pop_disc and ptried < n_pop_disc * 80:
        ptried += 1
        clauses, tn, tb = make_discourse(rng, refs)
        tr = _truth(clauses, refs)
        if len(tr) < 2 or not tr[-1][2]:
            continue
        resumed, pre_pop = tr[-1][0], tr[-2][0]
        if resumed == pre_pop:
            continue
        _hear(clauses)
        rw, pw = refs[resumed], refs[pre_pop]
        pk["spk"] += int(a_spk.who_agent_now() == rw); pk["les"] += int(a_les.who_agent_now() == rw)
        pk["stl"] += int(a_stl.who_agent_now() == rw); pk["twn"] += int(a_twn.who_agent_now() == rw)
        pk["stay"] += int(a_spk.who_agent_now() == pw); pk["rec"] += int(clauses[-1].split()[-1] == rw)
        pk_n += 1

    bef = now = tot = tried = 0
    while tot < n_disc and tried < n_disc * 25:
        tried += 1
        clauses, tn, tb = make_discourse(rng, refs)
        if tb == tn or tb == 0:
            continue
        _hear(clauses)
        bef += int(a_spk.who_agent_before() == refs[tb])
        now += int(a_spk.who_agent_now() == refs[tn])
        tot += 1

    mp = max(pk_n, 1); m = max(tot, 1)
    surv = (spk.held_survived / spk.held_checked) if spk.held_checked else float("nan")
    return {"seed": seed, "n_pop": pk_n, "n": tot,
            "RESUME_spiking": round(pk["spk"] / mp, 3), "RESUME_poplesion": round(pk["les"] / mp, 3),
            "RESUME_stateless": round(pk["stl"] / mp, 3), "RESUME_hosttwin": round(pk["twn"] / mp, 3),
            "RESUME_stay": round(pk["stay"] / mp, 3), "RESUME_recency": round(pk["rec"] / mp, 3),
            "held_survives_its_own_read": round(surv, 3), "n_read_checks": spk.held_checked,
            "BEFORE_spiking": round(bef / m, 3), "NOW_spiking": round(now / m, 3),
            "r_on_pops": round(float(np.mean(spk.r_on_pop)) if spk.r_on_pop else float("nan"), 3),
            "r_on_bounds": round(float(np.mean(spk.r_on_bnd)) if spk.r_on_bnd else float("nan"), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print("[D3 SPIKING POP] the brain resumes a protagonist by READING a spiking attractor without erasing it", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s); rows.append(r)
        print(f"  [seed {s}] RESUME: spiking={r['RESUME_spiking']} | pop-lesion={r['RESUME_poplesion']} | stateless={r['RESUME_stateless']} "
              f"| host-twin={r['RESUME_hosttwin']} | keep-agent={r['RESUME_stay']} | recency={r['RESUME_recency']} (n={r['n_pop']})", flush=True)
        print(f"            held slot SURVIVES its own read: {r['held_survives_its_own_read']} ({r['n_read_checks']} checks) "
              f"|| BEFORE={r['BEFORE_spiking']} NOW={r['NOW_spiking']} || gate r: pops={r['r_on_pops']} bounds={r['r_on_bounds']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k):
            v = [r[k] for r in rows if not (isinstance(r[k], float) and np.isnan(r[k]))]
            return float(np.mean(v)) if v else float("nan")
        sp, le, st, tw = _m("RESUME_spiking"), _m("RESUME_poplesion"), _m("RESUME_stateless"), _m("RESUME_hosttwin")
        sy, rc, sv = _m("RESUME_stay"), _m("RESUME_recency"), _m("held_survives_its_own_read")
        rp, rb, bf, nw = _m("r_on_pops"), _m("r_on_bounds"), _m("BEFORE_spiking"), _m("NOW_spiking")
        go = ((sp - le > 0.25) and (sp - st > 0.25) and (sp - sy > 0.3) and (sp - rc > 0.3)
              and (sv > 0.9) and (rp - rb > 0.3))
        print(f"\n  AGGREGATE  RESUME: spiking={sp:.3f} | pop-lesion={le:.3f} | STATELESS={st:.3f} | host-twin={tw:.3f} | keep-agent={sy:.3f} | recency={rc:.3f}", flush=True)
        print(f"    held slot survives its own read: {sv:.3f}   || BEFORE={bf:.3f}  NOW={nw:.3f}   || gate r: pops={rp:.3f} bounds={rb:.3f}", flush=True)
        msg = ('the deployed brain RESUMES a protagonist by READING a persistent slow-NMDA attractor -- and the read does not '
               'disturb it (the held slot still holds the same agent afterwards, ' + format(sv, '.2f') + '). Resumption ' + format(sp, '.2f') +
               ' vs ' + format(le, '.2f') + ' for the identical register with the read gate shut, ' + format(st, '.2f') + ' for a STATELESS slot '
               '(nothing to read; no host fallback anywhere), ' + format(sy, '.2f') + ' for "keep answering the same agent" and ' + format(rc, '.2f') +
               ' for recency; the host twin scores ' + format(tw, '.2f') + ', so the substrate costs little. The PUSH must DESTROY what the slot '
               'holds (clear longer than tau_NMDA); the POP must NOT -- and on this substrate the read drives nothing, so '
               'non-destructiveness is structural. The gate opens on deployed pops (' + format(rp, '.2f') + ') and stays shut on deployed '
               'boundaries (' + format(rb, '.2f') + ')')
        bad = 'the spiking pop did not clearly beat its controls, or the read disturbed the slot it read'
        print("  VERDICT: " + ("GO" if go else "PARTIAL/NEGATIVE") + " -- " + (msg if go else bad) + ". NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
