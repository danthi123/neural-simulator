"""D3 THE WHOLE EVENT PAIR ON SPIKES — two persistent attractors, two gates, both directions an attractor→attractor transfer.

WHERE THIS SITS.
Until now the register was a HOST vector (`a_curr`) with a SPIKING memory (`a_prev`) attached. The push wrote the spiking
slot (clear longer than tau_NMDA, then load) and the pop read it back out without erasing it (0.689 deployed, 6-seed GO).
`a_curr` was still re-chosen on the host every clause. This rung puts BOTH slots on persistent slow-NMDA attractors on
ONE bridge, so every state the register holds is a self-sustaining spiking assembly and both gates become transfers
between attractors:

    PUSH (boundary):  READ a_curr's spikes  ->  CLEAR a_prev (> tau_NMDA)  ->  LOAD a_prev
    POP  (return)  :  READ a_prev's spikes  ->  CLEAR a_curr (> tau_NMDA)  ->  LOAD a_curr
    otherwise      :  the transition proposes an agent            ->  CLEAR a_curr -> LOAD a_curr

ONE BRAIN, TWO SLOTS, SEPARATE INHIBITION. The slots share a bridge but each needs its OWN fast-spiking inhibitory pool:
a shared FS pool would mean clearing one slot SILENCES THE OTHER, and the pop (which writes a_curr while reading a_prev)
would destroy the very thing it is reading. That is a wiring fact, not a tuning choice, and the `pop_disturbs_prev`
counter below measures it directly.

THE ASYMMETRY, NOW ON BOTH SIDES. Every clause WRITES a_curr, and a persistent attractor RESISTS being overwritten by
input alone (measured: 0/6; the clear must outlast tau_NMDA or the incumbent re-ignites). So a_curr is cleared-then-loaded
each clause, while a_prev is written only on a boundary and otherwise holds itself with ZERO input across arbitrarily many
clauses.

ANTI-CHEATS (multi-seed):
 (a) resumption vs a POP-LESION register (identical model, read gate shut) -- the single-variable contrast;
 (b) vs "keep answering the pre-pop agent" and vs RECENCY;
 (c) **the POP must not disturb a_prev** -- it clears a_curr while reading a_prev; with a shared FS this fails;
 (d) **each slot survives its own read** (the read drives nothing);
 (e) STATELESS (recur=0 on both slots): nothing holds -> resumption and BEFORE collapse. NO host fallback anywhere;
 (f) HOST TWIN: both slots replaced by exact host copies, to price the substrate;
 (g) BEFORE (the held slot) must still work, and ordinary NOW must not regress.

Reuse-by-import; numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_pair_spiking_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_selfsup_pair_derisk import make_pair_task
from research.runners._d3_event_gated_copy_derisk import _sm, _sig
from research.runners._d3_event_pop_gate_derisk import train_pushpop
from research.runners._d3_event_gatedcopy_agent_derisk import fit_slot_names_labelfree
from research.runners._d3_event_selfsup_pair_agent_derisk import make_discourse, COREF_W, PROMOTE_W
from research.runners._d3_event_popgate_agent_derisk import _truth
from research.runners._d3_persistent_slot_derisk import _reset
from research.runners.multi_turn_agent import MultiTurnAgent

CURR, PREV = "c", "p"


def build_pair_slots(seed, K, n_word=20, n_fs=24, recur=25.0, exc_to_fs=1.4, fs_to_exc=10.0):
    """TWO slots x K attractor pools on ONE bridge, each slot with its OWN FS pool.

    Separate inhibition is load-bearing: the pop CLEARS a_curr while READING a_prev. A shared FS pool would silence both,
    so the pop would erase the very assembly it is reading."""
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel
    cfg = CoreSimConfig(); cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name; cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0; cfg.seed = int(seed); cfg.enable_brain_region_framework = True; cfg.ou_std_current_pA = 0.0
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp", "enable_input_divisive_norm"):
        setattr(cfg, flag, False)
    cfg.enable_nmda = True; cfg.enable_nmda_recurrent = True; cfg.nmda_recurrent_tau_decay_ms = 100.0

    regions, pathways = [], []
    for sl in (CURR, PREV):
        for k in range(K):
            regions.append(BrainRegion(name=f"{sl}{k}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0,
                                       exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                                       plastic_internal=False))
        regions.append(BrainRegion(name=f"fs_{sl}", n_neurons=n_fs, exc_fraction=0.0, internal_density=0.0,
                                   exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False))
        for k in range(K):
            pathways.append(RegionPathway(from_region=f"{sl}{k}", to_region=f"{sl}{k}", density=0.9,
                                          weight_mean=recur, weight_jitter=0.05, plastic=False,
                                          exc_receptor="nmda_slow"))       # slow-NMDA self-excitation = the hold
            pathways.append(RegionPathway(from_region=f"{sl}{k}", to_region=f"fs_{sl}", density=0.6,
                                          weight_mean=exc_to_fs, weight_jitter=0.1, plastic=False))
            pathways.append(RegionPathway(from_region=f"fs_{sl}", to_region=f"{sl}{k}", density=0.6,
                                          weight_mean=fs_to_exc, weight_jitter=0.1, plastic=False))
    cfg.brain_regions = regions; cfg.region_pathways = pathways
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


class PairSpikingRegister:
    """Both event slots are persistent slow-NMDA attractors on ONE bridge. Both gates are attractor->attractor transfers."""

    def __init__(self, referents, seed=42, n_hid=128, epochs=40, stage_pop_epochs=15, recurrent=True,
                 pop_lesion=False, host_twin=False, clear_steps=250, load_steps=80, inter_clause=15,
                 clear_gain=4000.0, load_gain=400.0, read_steps=30):
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
        self.clear_steps, self.load_steps, self.inter_clause = clear_steps, load_steps, inter_clause
        self.clear_gain, self.load_gain, self.read_steps = clear_gain, load_gain, read_steps
        self.r_on_pop, self.r_on_bnd = [], []
        self.surv_ok = self.surv_n = 0            # (d) a slot survives its own read
        self.popsafe_ok = self.popsafe_n = 0      # (c) the pop clears a_curr WITHOUT disturbing a_prev

        if not self.host_twin:
            self.sb = build_pair_slots(seed, K, recur=(25.0 if recurrent else 0.0))
            rm = self.sb.region_manager
            self.idx = {sl: {k: np.asarray(list(rm.indices(f"{sl}{k}")), dtype=int) for k in range(K)} for sl in (CURR, PREV)}
            self.fs = {sl: np.asarray(list(rm.indices(f"fs_{sl}")), dtype=int) for sl in (CURR, PREV)}
            self.n = self.sb.core_config.num_neurons
            self.zero = np.zeros(self.n, dtype=np.float64)
        self.reset()

    # ---- substrate primitives -------------------------------------------------
    def _step(self, cur, steps, sl=None):
        self.sb.cp_external_input_current[:] = cur
        acc = np.zeros(self.K)
        for _ in range(steps):
            self.sb._run_one_simulation_step()
            if sl is not None:
                f = self.sb.cp_firing_states
                for k in range(self.K):
                    acc[k] += float(np.asarray(f[self.idx[sl][k]]).sum())
        return acc

    def _read(self, sl):
        """Read a slot OUT OF SPIKES with ZERO input. Drives nothing -> cannot disturb what it reads.
        NO host fallback: a silent slot means nothing is held."""
        if self.host_twin:
            return self._twin[sl]
        assert not self.zero.any()
        acc = self._step(self.zero, self.read_steps, sl)
        return int(np.argmax(acc)) if acc.max() > 1e-6 else self.ident

    def _write(self, sl, content):
        """CLEAR then LOAD. Only this slot's FS pool is driven.

        A SHARPENING of the earlier rule. The push rung concluded "the clear must OUTLAST tau_NMDA, or the incumbent
        re-ignites" -- calibrated on a slot written ONCE per discourse. `a_curr` is written EVERY clause, so it carries a
        far larger residual NMDA conductance into each clear, and the validated 250-step / 1500 pA clear FAILED (write-
        read-back 10/12; the failures land exactly where the incumbent's residual g_nmda peaks, and the slot reads back the
        OLD content). MEASURED: a LONGER clear (400 steps @ 1500 pA) and a STRONGER one (250 steps @ 4000 pA) BOTH give
        12/12. So the governing quantity is the PRODUCT of inhibition strength and duration, not duration alone: held
        hyperpolarized hard enough, the Mg2+ block keeps the residual NMDA current from re-igniting the assembly without
        waiting out the time constant. The stronger clear is taken here (it is cheaper in wall-clock)."""
        if self.host_twin:
            self._twin[sl] = content
            return
        cc = np.zeros(self.n, dtype=np.float64); cc[self.fs[sl]] = self.clear_gain
        self._step(cc, self.clear_steps)
        cl = np.zeros(self.n, dtype=np.float64); cl[self.idx[sl][content]] = self.load_gain
        self._step(cl, self.load_steps)

    # ---- register API ---------------------------------------------------------
    def reset(self):
        if not self.host_twin:
            _reset(self.sb)
        self._twin = {CURR: self.ident, PREV: self.ident}
        self.pat = np.zeros(self.K, np.float32); self.pat[self.ident] = 1.0
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

        cur_agent = self._read(CURR)                       # both transition inputs come OUT OF SPIKES
        prev_agent = self._read(PREV)

        if g > 0.5:                                        # ---- PUSH: a_curr's assembly transfers into a_prev
            self._write(PREV, cur_agent)
            prev_agent = cur_agent

        emb, Wr, Wi, Wc, bc = (self.W["emb"], self.W["Wr"], self.W["Wi"], self.W["Wc"], self.W["bc"])
        c_oh = np.zeros(self.K, np.float32); c_oh[cur_agent] = 1.0
        p_oh = np.zeros(self.K, np.float32); p_oh[prev_agent] = 1.0
        h = np.tanh(np.concatenate([c_oh @ emb, p_oh @ emb, self.pat @ emb]) @ Wr.T + code @ Wi.T)
        raw = _sm(h @ Wc.T + bc)

        if r > 0.5:                                        # ---- POP: a_prev's assembly transfers into a_curr
            self._write(CURR, prev_agent)                  #      writes a_curr; MUST NOT disturb a_prev
            if not self.host_twin and is_pop:
                self.popsafe_n += 1
                self.popsafe_ok += int(self._read(PREV) == prev_agent)
                self.surv_n += 1
                self.surv_ok += int(self._read(CURR) == prev_agent)
        else:                                              # ---- the transition proposes; a_curr is rewritten
            self._write(CURR, int(np.argmax(raw)))
        self.pat = np.zeros(self.K, np.float32); self.pat[o] = 1.0

    def who_agent(self):
        return self.referents[int(self.perm[self._read(CURR)])]

    def who_patient(self):
        return self.referents[int(np.argmax(self.pat))]

    def who_agent_prev(self):
        return self.referents[int(self.perm[self._read(PREV)])]


def run_seed(seed, n_pop_disc=10, n_disc=10):
    refs = ["dog", "cat", "fish", "bird", "worm", "ball"]
    vocab = {w: None for w in (refs + ["chase"])}
    rng = np.random.RandomState(seed + 11)

    spk = PairSpikingRegister(refs, seed=seed)
    les = PairSpikingRegister(refs, seed=seed, pop_lesion=True)
    stl = PairSpikingRegister(refs, seed=seed, recurrent=False)
    twn = PairSpikingRegister(refs, seed=seed, host_twin=True)
    mk = lambda reg: MultiTurnAgent(refs, concepts=vocab, seed=seed, enable_biased_competition=True,
                                    event_register=reg, enable_neural_render=False)
    a_spk, a_les, a_stl, a_twn = mk(spk), mk(les), mk(stl), mk(twn)
    regs, agents = (spk, les, stl, twn), (a_spk, a_les, a_stl, a_twn)

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

    # ---- BEFORE, on discourses that actually REQUIRE the slot to hold.
    # A stateless slot (recur=0) cannot hold anything, but the LOAD pulse leaves a decaying trace inside the 30-step read
    # window. So if the LAST clause is a boundary, the push just drove the slot and a stateless control answers BEFORE
    # correctly WITHOUT any memory (measured: stateless BEFORE 0.6 on seed 100). That inflates the control and destroys its
    # discriminating power. Require >= 2 clauses since the last boundary, so the trace has decayed and only a real
    # attractor can still be holding. Same logic as every other pool here: the question must require the mechanism.
    MIN_SINCE_PUSH = 2
    bef = {k: 0 for k in ("spk", "stl", "twn")}
    now = tot = tried = 0
    while tot < n_disc and tried < n_disc * 40:
        tried += 1
        clauses, tn, tb = make_discourse(rng, refs)
        if tb == tn or tb == 0:
            continue
        lead = [i for i, c in enumerate(clauses)
                if c.split()[0].lower() in ("then", "but", "meanwhile") and c.split()[1].lower() not in ("he", "she", "they", "it")]
        if not lead or (len(clauses) - 1 - lead[-1]) < MIN_SINCE_PUSH:
            continue                                        # the push must be >= 2 clauses in the past
        _hear(clauses)
        bef["spk"] += int(a_spk.who_agent_before() == refs[tb])
        bef["stl"] += int(a_stl.who_agent_before() == refs[tb])
        bef["twn"] += int(a_twn.who_agent_before() == refs[tb])
        now += int(a_spk.who_agent_now() == refs[tn])
        tot += 1

    mp = max(pk_n, 1); m = max(tot, 1)
    return {"seed": seed, "n_pop": pk_n, "n": tot,
            "RESUME_pairspiking": round(pk["spk"] / mp, 3), "RESUME_poplesion": round(pk["les"] / mp, 3),
            "RESUME_stateless": round(pk["stl"] / mp, 3), "RESUME_hosttwin": round(pk["twn"] / mp, 3),
            "RESUME_stay": round(pk["stay"] / mp, 3), "RESUME_recency": round(pk["rec"] / mp, 3),
            "BEFORE_pairspiking": round(bef["spk"] / m, 3), "BEFORE_stateless": round(bef["stl"] / m, 3),
            "BEFORE_hosttwin": round(bef["twn"] / m, 3), "NOW_pairspiking": round(now / m, 3),
            "pop_leaves_prev_intact": round(spk.popsafe_ok / spk.popsafe_n, 3) if spk.popsafe_n else float("nan"),
            "slot_survives_own_read": round(spk.surv_ok / spk.surv_n, 3) if spk.surv_n else float("nan"),
            "n_pop_checks": spk.popsafe_n,
            "r_on_pops": round(float(np.mean(spk.r_on_pop)) if spk.r_on_pop else float("nan"), 3),
            "r_on_bounds": round(float(np.mean(spk.r_on_bnd)) if spk.r_on_bnd else float("nan"), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print("[D3 EVENT PAIR ON SPIKES] two persistent attractors, two gates, both directions an attractor->attractor transfer", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s); rows.append(r)
        print(f"  [seed {s}] RESUME: pair-spiking={r['RESUME_pairspiking']} | pop-lesion={r['RESUME_poplesion']} | stateless={r['RESUME_stateless']} "
              f"| host-twin={r['RESUME_hosttwin']} | keep-agent={r['RESUME_stay']} | recency={r['RESUME_recency']} (n={r['n_pop']})", flush=True)
        print(f"            POP leaves a_prev intact: {r['pop_leaves_prev_intact']} | slot survives its own read: {r['slot_survives_own_read']} "
              f"({r['n_pop_checks']} checks) || BEFORE={r['BEFORE_pairspiking']} (stateless {r['BEFORE_stateless']}) NOW={r['NOW_pairspiking']} "
              f"|| gate r: pops={r['r_on_pops']} bounds={r['r_on_bounds']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k):
            v = [r[k] for r in rows if not (isinstance(r[k], float) and np.isnan(r[k]))]
            return float(np.mean(v)) if v else float("nan")
        sp, le, st, tw = _m("RESUME_pairspiking"), _m("RESUME_poplesion"), _m("RESUME_stateless"), _m("RESUME_hosttwin")
        sy, rc = _m("RESUME_stay"), _m("RESUME_recency")
        ps, sv = _m("pop_leaves_prev_intact"), _m("slot_survives_own_read")
        bf, bst, nw = _m("BEFORE_pairspiking"), _m("BEFORE_stateless"), _m("NOW_pairspiking")
        rp, rb = _m("r_on_pops"), _m("r_on_bounds")
        go = ((sp - le > 0.25) and (sp - st > 0.25) and (sp - sy > 0.3) and (sp - rc > 0.3)
              and (ps > 0.9) and (bf - bst > 0.3) and (rp - rb > 0.3))
        print(f"\n  AGGREGATE  RESUME: pair-spiking={sp:.3f} | pop-lesion={le:.3f} | STATELESS={st:.3f} | host-twin={tw:.3f} | keep-agent={sy:.3f} | recency={rc:.3f}", flush=True)
        print(f"    POP leaves a_prev intact={ps:.3f} | slot survives its own read={sv:.3f} || BEFORE={bf:.3f} (stateless {bst:.3f}) NOW={nw:.3f} || gate r: pops={rp:.3f} bounds={rb:.3f}", flush=True)
        msg = ('the WHOLE event pair is now spiking: BOTH slots are persistent slow-NMDA attractors on ONE bridge, and BOTH gates '
               'are attractor->attractor transfers. Resumption ' + format(sp, '.2f') + ' vs ' + format(le, '.2f') + ' (read gate shut) and '
               + format(st, '.2f') + ' (STATELESS: nothing holds; no host fallback). The POP writes a_curr while READING a_prev and leaves '
               'a_prev intact (' + format(ps, '.2f') + ') -- which is why each slot needs its OWN inhibitory pool: a shared one would have '
               'the pop erase the very assembly it reads. BEFORE ' + format(bf, '.2f') + ' vs a stateless ' + format(bst, '.2f') +
               '; the host twin scores ' + format(tw, '.2f') + ', pricing the substrate')
        bad = 'the pair-spiking register did not clearly beat its controls, or the pop disturbed the slot it was reading'
        print("  VERDICT: " + ("GO" if go else "PARTIAL/NEGATIVE") + " -- " + (msg if go else bad) + ". NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
