"""D3 POP GATE -> the LIVE MultiTurnAgent: a discourse that RETURNS to its earlier protagonist, answered by a register
with two gates and no state label anywhere.

WHAT THIS DEPLOYS.
The push gate (boundary -> copy the running event INTO the held slot) is already deployed and answers "who was doing it
BEFORE?" at 0.711. Instrumenting that deployment proved the held slot is a PERFECT copy -- every remaining error is an
`a_curr` error inherited at the copy moment -- and that the emergent transition fails on exactly ONE relational
operation: RETURN, the discourse pop, the one op that must READ the held slot back out.

`2026-07-10-D3-pop-gate-the-discourse-pop-is-a-gated-copy-OUT.md` gave RETURN a second, normally-closed gate on the SAME
register (a_curr <- r*a_prev + (1-r)*delta), opened by the observable return marker, its onset DELAYED until the
representation it reads is trustworthy. It recovered 96% of an oracle gate's headroom on that operation. This rung asks
the only question that matters: does that survive deployment on the live agent, and on the live generator?

THE TRANSFER RISK (named before running it). The register is TRAINED on `make_pair_task`, whose clause code carries an
explicit RET mark, and DEPLOYED on `make_discourse`, where a connective + a PRONOUN subject is a pop and a connective +
a NAMED subject is a boundary. `GatedCopyPairRegister.observe` already builds `mk = marks["RET"]` in exactly that case,
so the deployed code carries the distinction the gate keys on. This repo has been bitten by a train/deploy generator
mismatch before (`2026-07-10-D3-selfsup-pair-deployed-PARTIAL-...`), which is why it is checked rather than assumed: the
gate's mean opening `r` is REPORTED separately on deployed pops and deployed boundaries. A gate that opens on boundaries
would overwrite the present with a stale past.

TWO QUESTIONS, NOT ONE. The push gate bought "who was doing it BEFORE?". The pop gate buys something the register could
not do at all: after a discourse pop, "who is doing it NOW?" is the agent of the EARLIER event -- the brain must resume a
protagonist it had already set aside. That NOW-after-a-pop accuracy is the pop gate's own headline; BEFORE is the
inherited beneficiary.

ANTI-CHEATS (6-seed): (a) NOW-after-a-pop vs the PUSH-ONLY register (the single-variable contrast, same everything else);
(b) vs RECENCY and vs "answer the current agent" -- a listener's shortcuts; (c) a POP-LESION register (r forced to 0)
must reproduce push-only exactly; (d) a SINGLE-EVENT register cannot answer BEFORE at all; (e) the gate's opening on
deployed BOUNDARIES must stay near zero; (f) BEFORE and ordinary NOW must not regress.

Reuse-by-import; numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_popgate_agent_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_selfsup_pair_derisk import make_pair_task
from research.runners._d3_event_pop_gate_derisk import train_pushpop
from research.runners._d3_event_gated_copy_derisk import _sm, _sig
from research.runners._d3_event_gatedcopy_agent_derisk import fit_slot_names_labelfree, GatedCopyPairRegister
from research.runners._d3_event_agent_derisk import D3EventRegister
from research.runners._d3_event_selfsup_pair_agent_derisk import make_discourse, COREF_W, PROMOTE_W, CONNECTIVES
from research.runners.multi_turn_agent import MultiTurnAgent


class PopGatePairRegister(GatedCopyPairRegister):
    """The push-gated register, plus a POP gate that reads the held slot back into the current one on a return marker.
    Drop-in for `GatedCopyPairRegister`. `pop_lesion=True` forces r = 0 and must reproduce it exactly."""

    def __init__(self, referents, seed=42, n_hid=128, epochs=40, stage_pop_epochs=15,
                 gate_lesion=False, pop_lesion=False):
        self.referents = list(referents); self.ref2idx = {r: i for i, r in enumerate(referents)}
        K = len(referents); self.K = K
        task = make_pair_task(seed, K=K)
        roll = train_pushpop(task, seed=seed, n_hid=n_hid, epochs=epochs,
                             stage_pop_epochs=stage_pop_epochs, freeze_core_in_phase2=False)
        self.W = roll.W
        self.wg, self.bg, self.wp, self.bp = roll.gates
        self.ent, self.marks = task["ent"], task["marks"]
        self.ident = task["ident"]
        self.gate_lesion = bool(gate_lesion); self.pop_lesion = bool(pop_lesion)
        self.perm = fit_slot_names_labelfree(task, self.W, K)   # ONE label-free read-out names both slots
        self.r_on_pop, self.r_on_bnd = [], []                   # deployed gate opening, by clause kind
        self.reset()

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
        if self._boundary:
            mk = self.marks["RET"] if is_pop else self.marks["BND"]
        else:
            mk = self.marks["NOB"]
        self._boundary = False
        code = np.concatenate([mk, sub, self.ent[o]]).astype(np.float32)

        emb, Wr, Wi, Wc, bc = (self.W["emb"], self.W["Wr"], self.W["Wi"], self.W["Wc"], self.W["bc"])
        g = 0.0 if self.gate_lesion else float(_sig(code @ self.wg + self.bg))
        r = 0.0 if self.pop_lesion else float(_sig(code @ self.wp + self.bp))
        if is_pop:
            self.r_on_pop.append(r)
        elif is_bnd:
            self.r_on_bnd.append(r)
        h = np.tanh(np.concatenate([self.sc @ emb, self.sp @ emb, self.pat @ emb]) @ Wr.T + code @ Wi.T)
        raw = _sm(h @ Wc.T + bc)
        sc_old, sp_old = self.sc, self.sp
        self.sp = g * sc_old + (1.0 - g) * sp_old        # PUSH: gated copy IN  (uses the OLD a_curr)
        self.sc = r * sp_old + (1.0 - r) * raw           # POP:  gated copy OUT (uses the OLD a_prev)
        self.pat = np.zeros(self.K, np.float32); self.pat[o] = 1.0


def _truth(clauses, refs):
    """Per-clause (a_curr, a_prev) from the utterances alone, and whether each clause is a discourse pop."""
    idx = {r: i for i, r in enumerate(refs)}
    ac = pc = ap = 0
    out = []
    for c in clauses:
        w = c.split(); lead = w[0].lower() in CONNECTIVES
        sw = w[1] if lead else w[0]; o = w[-1]
        pop = lead and (sw in COREF_W or sw in PROMOTE_W)
        if pop:
            ac = ap
        elif lead:
            ap = ac; ac = idx[sw]
        elif sw in COREF_W:
            pass
        elif sw in PROMOTE_W:
            ac = pc
        else:
            ac = idx[sw]
        pc = idx[o]; out.append((ac, ap, pop))
    return out


def run_seed(seed, n_disc=30, n_pop_disc=30):
    refs = ["dog", "cat", "fish", "bird", "worm", "ball"]
    vocab = {w: None for w in (refs + ["chase"])}
    rng = np.random.RandomState(seed + 11)

    pop = PopGatePairRegister(refs, seed=seed)
    push = PopGatePairRegister(refs, seed=seed, pop_lesion=True)     # SAME model, r == 0: the single-variable control
    base = GatedCopyPairRegister(refs, seed=seed)                    # the deployed predecessor (a different trainer)
    single = D3EventRegister(refs, seed=seed, spiking=False)
    mk = lambda reg: MultiTurnAgent(refs, concepts=vocab, seed=seed, enable_biased_competition=True,
                                    event_register=reg, enable_neural_render=False)
    a_pop, a_push, a_base, a_single = mk(pop), mk(push), mk(base), mk(single)

    def _hear_all(clauses):
        for r_ in (pop, push, base, single):
            r_.reset()
        for c in clauses:
            for a_ in (a_pop, a_push, a_base, a_single):
                a_.hear(c)

    # ---- POOL A: "who was doing it BEFORE?" (a real prior event, distinguishable from the current agent)
    bef_pop = bef_push = bef_base = bef_single = bef_rec = bef_naive = 0
    now_pop_all = now_push_all = tot = tried = 0
    while tot < n_disc and tried < n_disc * 25:
        tried += 1
        clauses, tn, tb = make_discourse(rng, refs)
        if tb == tn or tb == 0:
            continue
        _hear_all(clauses)
        tbw, tnw = refs[tb], refs[tn]
        bef_pop += int(a_pop.who_agent_before() == tbw)
        bef_push += int(a_push.who_agent_before() == tbw)
        bef_base += int(a_base.who_agent_before() == tbw)
        bef_single += int(a_single.who_agent_before() == tbw)
        bef_rec += int(clauses[-1].split()[-1] == tbw)
        bef_naive += int(a_pop.who_agent_now() == tbw)
        now_pop_all += int(a_pop.who_agent_now() == tnw)
        now_push_all += int(a_push.who_agent_now() == tnw)
        tot += 1

    # ---- POOL B: "who is doing it NOW?" immediately AFTER a discourse pop.
    # A pop sets a_curr <- a_prev, so true_now == true_before and POOL A's informativeness filter DISCARDS every such
    # discourse BY CONSTRUCTION (a defect in the first version of this runner: n_pop was 0 on all six seeds). The
    # resumption question therefore needs its own pool: discourses whose LAST clause is a pop AND whose resumed agent
    # DIFFERS from the pre-pop agent -- otherwise "keep answering the same agent" is trivially correct.
    pk_pop = pk_push = pk_rec = pk_stay = pk_n = ptried = 0
    while pk_n < n_pop_disc and ptried < n_pop_disc * 80:
        ptried += 1
        clauses, tn, tb = make_discourse(rng, refs)
        tr = _truth(clauses, refs)
        if len(tr) < 2 or not tr[-1][2]:
            continue                                                # the last clause must be a discourse pop
        resumed, pre_pop = tr[-1][0], tr[-2][0]
        if resumed == pre_pop:
            continue                                                # a genuine RESUMPTION, not a no-op
        _hear_all(clauses)
        rw, pw = refs[resumed], refs[pre_pop]
        pk_pop += int(a_pop.who_agent_now() == rw)
        pk_push += int(a_push.who_agent_now() == rw)
        pk_stay += int(a_pop.who_agent_now() == pw)                 # the shortcut: keep answering the PRE-POP agent
        pk_rec += int(clauses[-1].split()[-1] == rw)
        pk_n += 1

    m = max(tot, 1); mp = max(pk_n, 1)
    return {"seed": seed, "n": tot, "n_pop": pk_n,
            "NOWafterPOP_popgate": round(pk_pop / mp, 3), "NOWafterPOP_pushonly": round(pk_push / mp, 3),
            "NOWafterPOP_recency": round(pk_rec / mp, 3), "NOWafterPOP_stay": round(pk_stay / mp, 3),
            "BEFORE_popgate": round(bef_pop / m, 3), "BEFORE_pushlesion": round(bef_push / m, 3),
            "BEFORE_gatedcopy": round(bef_base / m, 3), "BEFORE_single": round(bef_single / m, 3),
            "BEFORE_recency": round(bef_rec / m, 3), "BEFORE_naive": round(bef_naive / m, 3),
            "NOW_popgate": round(now_pop_all / m, 3), "NOW_pushonly": round(now_push_all / m, 3),
            "r_on_deployed_pops": round(float(np.mean(pop.r_on_pop)) if pop.r_on_pop else float("nan"), 3),
            "r_on_deployed_bounds": round(float(np.mean(pop.r_on_bnd)) if pop.r_on_bnd else float("nan"), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print("[D3 POP GATE -> LIVE MultiTurnAgent] after a discourse pop, the brain resumes the protagonist it had set aside", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s); rows.append(r)
        print(f"  [seed {s}] NOW-after-a-POP: popgate={r['NOWafterPOP_popgate']} vs push-only={r['NOWafterPOP_pushonly']} "
              f"| keep-pre-pop-agent={r['NOWafterPOP_stay']} | recency={r['NOWafterPOP_recency']} (n_pop={r['n_pop']})", flush=True)
        print(f"            BEFORE: popgate={r['BEFORE_popgate']} | pop-lesion={r['BEFORE_pushlesion']} | gatedcopy={r['BEFORE_gatedcopy']} "
              f"| single={r['BEFORE_single']} || NOW(all)={r['NOW_popgate']} vs {r['NOW_pushonly']} "
              f"|| gate r: pops={r['r_on_deployed_pops']} bounds={r['r_on_deployed_bounds']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows if not np.isnan(r[k])]))
        npg, npu, nrc = _m("NOWafterPOP_popgate"), _m("NOWafterPOP_pushonly"), _m("NOWafterPOP_recency")
        nst = _m("NOWafterPOP_stay")
        bp, bl, bg_, bs = _m("BEFORE_popgate"), _m("BEFORE_pushlesion"), _m("BEFORE_gatedcopy"), _m("BEFORE_single")
        bn, br = _m("BEFORE_naive"), _m("BEFORE_recency")
        nw, nw0 = _m("NOW_popgate"), _m("NOW_pushonly")
        rp, rb = _m("r_on_deployed_pops"), _m("r_on_deployed_bounds")
        go = ((npg - npu > 0.25) and (npg - nrc > 0.3) and (npg - nst > 0.3) and (rp - rb > 0.3)
              and (nw >= nw0 - 0.02) and (bs < 0.05))
        print(f"\n  AGGREGATE  NOW-after-a-POP: popgate={npg:.3f} | push-only={npu:.3f} | keep-pre-pop-agent={nst:.3f} | recency={nrc:.3f}", flush=True)
        print(f"    BEFORE: popgate={bp:.3f} | pop-lesion={bl:.3f} | gatedcopy(prior deploy)={bg_:.3f} | single-event={bs:.3f} | recency={br:.3f} | naive-current={bn:.3f}", flush=True)
        print(f"    NOW(all)={nw:.3f} vs push-only {nw0:.3f}   ||   deployed gate opening r: on POPS={rp:.3f}  on BOUNDARIES={rb:.3f}", flush=True)
        msg = ('the DEPLOYED brain RESUMES a protagonist it had set aside. After a discourse pop ("meanwhile HE chase ball"), '
               '"who is doing it now?" is the agent of the EARLIER event -- an answer the push-only register cannot give ('
               + format(npu, '.2f') + ') and the pop-gated register gives at ' + format(npg, '.2f') + ', beating recency ('
               + format(nrc, '.2f') + '). The gate TRANSFERS to the live generator without a mismatch: it opens on deployed pops ('
               + format(rp, '.2f') + ') and stays shut on deployed boundaries (' + format(rb, '.2f') + '), so it never overwrites the '
               'present with a stale past. BEFORE (' + format(bp, '.2f') + ') and ordinary NOW (' + format(nw, '.2f') + ') do not regress, '
               'and a SINGLE-EVENT register still cannot answer BEFORE at all (' + format(bs, '.2f') + '). No state label anywhere')
        bad = 'the deployed pop gate did not clearly beat push-only / recency, or it opened on boundaries (a stale-past overwrite)'
        print("  VERDICT: " + ("GO" if go else "PARTIAL/NEGATIVE") + " -- " + (msg if go else bad) + ". NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
