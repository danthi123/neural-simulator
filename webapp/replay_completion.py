"""PATTERN COMPLETION IN SWR REPLAY (the DA tag-and-capture routes): a replay event reinstates the stored fact's whole
ensemble once its partial trace still selects the fact's own items, instead of reactivating it in proportion to the
partial trace's decode margin. Two default-OFF flags, one per replay route that reads a block:
  `BRAIN_AWAKE_REPLAY_COMPLETION` -- the quiet-rest bouts of webapp/awake_replay_capture.py (needs
                                     `BRAIN_AWAKE_REPLAY_CAPTURE`);
  `BRAIN_SLEEP_REPLAY_COMPLETION` -- the night's SWR epoch of webapp/sleep_replay_capture.py (needs
                                     `BRAIN_SLEEP_REPLAY_CAPTURE`).
Branch research/awake-replay-completion.

WHY (the wall, measured). The awake-rest route scored NO-GO 5/6
(research/findings/2026-09-25-awake-replay-capture-arc-no-go-6seed.md): on one gate seed the long-delay fact's read R
started at 0.207 (0.28-0.49 on the other five) and fell to 0.031 across the 48 five-minute bouts, while the other
seeds lost a few percent. Each bout re-induces e <- e + R (1 - e) with R the smallest cleanup decisiveness margin
(peak - runner_up) / peak over agent/action/patient. Between bouts e decays by exp(-5 min / 1.5 h) = 0.946, so a
block holds only if R(e) (1 - e) >= ~0.057 e at some e; a block whose margin is low at every e has no upper fixed
point and the loop that should keep it amplifies its decay instead (subcritical). The night's route has the same
proxy: its re-tag is R x |inc| and its SWR-coupled DA is tonic + span x sum R, so a low-margin fact that rest DID keep
expressed is still not captured (measured on dev seed 2 with the awake completion alone: expression 0.98 at the last
bout, all three items reinstated, night read 0.107, SWR DA 0.579, not captured).

THE WALL QUESTION: what does the real system run alongside this that the code replaced with a linear proxy? The
margin is a READ-OUT quantity: how far the fact's word stands above the most similar OTHER word of the vocabulary in
the cleanup. How strongly a reactivation re-induces LTP is a PARTICIPATION quantity: how many of the trace's own
synapses see their pre and post cells fire together in the replay event (Sadowski, Jones & Mellor 2016: the induced
change scales with the number of LTP-competent pairings); and the DA the event co-releases is, in the sleep route's
own model, proportional to the tags the reactivation sets (Clopath et al. 2008). In the hippocampus the two come apart
because of pattern completion: a replay event -- awake or asleep -- is a population burst of the CA3 recurrent network
that starts at a threshold level of firing (de la Prida et al. 2006), and "the reactivation of a subset of this
stored cell assembly would be sufficient to activate the entire original neural ensemble" (Kandel 6e ch.54, Marr's
proposal; Nakazawa et al. 2002; Guzman et al. 2016). So once the partial trace still selects the fact, the whole
ensemble fires; how close the runner-up word happens to be does not scale the burst. The linear proxy (reactivation
= margin) omits the completion. Biology binding: research/biology/awake-replay-pattern-completion.md.

WHAT HAPPENS (only with a flag ON, only inside that route's replay event):
  1. PARTIAL CUE. The route's own read runs unchanged (the block's trigger is driven through the store, unbound per
     role and read by the cleanup: `sleep_replay_capture.reactivation_strength`, R). It is kept on the record.
  2. ITEM COMPETITION (spiking). The same substrate read gives each concept unit its matched-filter drive (the
     rectified cleanup membrane; `_role_scores`, the ops of `OneBrainComposer._block_role_scores`). Per role the
     drive, divided by its own peak (the feedback-inhibition normalization the composer's spiking cleanup already
     uses), drives the composer's Izhikevich concept bank at its graded operating point (`_margin_drive_pA`, the
     measured point where the bank's spike counts are graded) for the cleanup window. The unit that FIRES most is the
     reinstated item; a silent or tied competition reinstates nothing for that role (no host argmax fallback).
  3. REINSTATEMENT (the completion). The reinstated items are re-bound to their roles and bundled on the composer's
     own resonate-and-fire work registers -- `OneBrainComposer._compose_phases`, the SAME substrate op that encoded
     the fact -- and read back as spike phases: the ensemble the burst reinstates at the block's readout.
  4. PAIRING COUNT. R_c = the in-phase coherence between the reinstated readout pattern and the block's own stored
     increment, Re(mean_k conj(d_k) z_k) clipped to [0, 1] (d = inc / |inc|): the fraction of the block's synapses
     whose post cell reinstates in phase with the synapse's increment (Sadowski's LTP-competent pairings). When the
     three content items are reinstated correctly the reinstated pattern is the stored composite less its polarity
     component (R_c ~0.75-0.79 measured; see COMPLETION_ROLES); a wrong item lowers it; a trace that no longer
     selects its items reinstates an unrelated pattern and R_c ~ 0. Unlike the margin R, R_c knows WHICH items the
     read selected: a decayed block whose cleanup confidently selects a wrong word (or a reserved slot) has a margin R
     but no coherence with its own increment.
  5. The route then uses R_c where it used R, and nothing else changes. Awake bout: e <- e + R_c (1 - e), the tag
     re-set to the same level (no PRP, z untouched, the awake-edge lesion still zeroes R_eff). Night epoch: the replay
     tag R_c x |inc|, the SWR-coupled DA tonic + span x min(1, sum R_c) onto the same spiking D1 pool, and (under the
     downscaling sub-flags) the protection 1 - delta (1 - R_c); both replay/DA lesions act on it unchanged.

THE THRESHOLD IS NOT A CONSTANT. Whether a partial trace ignites is decided by whether its own read still selects the
fact's items in the spiking competition, i.e. by the ratio of the decayed increment to the baseline synapses and to
the vocabulary's crosstalk on that block. No threshold, gain or iteration count is added.

THE COMPLETION LESION. `BRAIN_REPLAY_COMPLETION_LESION=1` cuts the completion's effect in BOTH routes: every read
above still runs (same compute, same substrate state) and is recorded, but each route uses the partial cue's own R,
i.e. exactly the Amendment-4 awake route and the rc/r2 night route.

HOST SHORTCUTS (declared, brain-based-only burn-down):
  - the op sequencing (read -> competition -> re-bind -> read-back) is host dispatch of substrate ops, as every
    composer op is; ONE pass (T = 1): no multi-cycle settle within the ripple;
  - the per-role peak normalization of the drive (the divisive feedback-inhibition stand-in the composer's spiking
    cleanup already uses) and the read of the winner as the unit with the largest spike count; the bank's units are
    independent (no lateral inhibition), so near-ties resolve as no reinstatement and an equal drive still picks the
    bank's most excitable unit;
  - R_c is host arithmetic on the reinstated spike phases against the ledger's stored increment (the same class as
    the margin arithmetic it replaces);
  - the reinstated pattern is credited to the ledger only along the stored increment: a wrong item's LTP (the
    component orthogonal to the increment) is DROPPED, which can only understate confabulation risk; every replay
    event records the reinstated items and R_c so a wrong completion is visible on the record;
  - the roles completed are the three content roles; a reserved (unrecruited) cleanup slot or a word without a code
    is never reinstated (no code growth from this read).
  - NOT MODELLED: literal CA3-CA3 collaterals. The composer store has no recurrent collaterals; the model's recurrent
    path for a stored fact is its own readout -> unbind -> cleanup -> re-bind loop, whose forward half every recall
    already runs. The repo's CA3 superposed-fact attractor (research/runners/ca3_superposed_fact_attractor.py) is a
    standalone binary k-WTA runner with its own EC codes and no chat write path, so the replay read cannot route
    through it without a second store; not used.
This is reactivation-driven re-potentiation / capture in the same store, not "consolidation" in the docs/TERMS.md
sense.

CONTRACT. DEFAULT-OFF. With both flags unset nothing here runs: the awake bout and the night epoch take their
pre-branch paths verbatim (tests/test_awake_replay_completion.py pins both against the pre-branch methods' store hash
and records). No `sim/` edit; no edit to one_brain_composer.py.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from webapp.sleep_replay_capture import FACT_ROLES, reactivation_strength

# The fact's CONTENT roles -- the same three the margin read R takes its minimum over. Polarity is bound in the
# stored composite too, but its 2-word competition resolves whatever the trace strength (AFFIRM wins at any e, even on
# the bare baseline), so reinstating it would add an unconditional floor to R_c that carries no evidence that THIS
# fact's ensemble ignited. Measured before this was fixed (a D=64 composer, dev seed 7): with polarity in, R_c = 0.081
# on the bare baseline (e = 0), from the polarity item alone. So the reinstated ensemble is the content items only; a
# fully reinstated fact then reads R_c < 1 (three of the four bound roles), which each route takes as it is.
COMPLETION_ROLES = FACT_ROLES


def _truthy(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "on", "yes")


def awake_completion_enabled() -> bool:
    """DEFAULT OFF. `BRAIN_AWAKE_REPLAY_COMPLETION` in {1,true,on,yes}: the awake bouts complete."""
    return _truthy("BRAIN_AWAKE_REPLAY_COMPLETION")


def sleep_completion_enabled() -> bool:
    """DEFAULT OFF. `BRAIN_SLEEP_REPLAY_COMPLETION` in {1,true,on,yes}: the night's SWR epoch completes."""
    return _truthy("BRAIN_SLEEP_REPLAY_COMPLETION")


def completion_lesioned() -> bool:
    """`BRAIN_REPLAY_COMPLETION_LESION` cuts the completion's effect in both routes (the reads still run): R_eff = R."""
    return _truthy("BRAIN_REPLAY_COMPLETION_LESION")


def read_blocks(comp, ledger, rng_ctx, seed, k_base):
    """The completion read of every managed block, each inside rng_ctx(seed, k_base + i) exactly as the route's own
    read loop. Returns (R list, R_read list, records): R = the partial cue's read, R_read = what the route uses (R_c,
    or R under the completion lesion), records = the per-block completion dicts."""
    use_c = not completion_lesioned()
    R, R_read, recs = [], [], []
    for i in range(len(ledger.blocks)):
        with rng_ctx(seed, k_base + i):
            c = completion_read(comp, ledger.block_offset + i, ledger.blocks[i])
        recs.append(c)
        R.append(None if c is None else float(c["R"]))
        R_read.append(None if c is None else (float(c["R_c"]) if use_c else float(c["R"])))
    return R, R_read, recs


def record(recs) -> list:
    """The per-block completion record a route stores (rounded like the route's own fields)."""
    return [None if c is None else {"R_c": round(c["R_c"], 9), "coherence_abs": round(c["coherence_abs"], 9),
                                    "items": c["items"], "spikes": c["spikes"], "n_items": c["n_items"]}
            for c in recs]


def _role_scores(comp, block_idx: int) -> Optional[dict]:
    """The matched-filter drive each concept unit receives when block `block_idx` is reactivated: the SAME substrate
    ops as `OneBrainComposer._block_role_scores` (kick the trigger through the store, unbind every role in parallel,
    one cleanup matvec) and the same rectification. Returns {role: (scores, vocab)} or None when the composer has no
    block-structured store read."""
    need = ("b", "D", "period", "V", "NP", "store_base", "block", "store_conns", "bind_roles", "main_roles",
            "q_base", "c_base", "n_main", "words", "pol_words", "rf_mask", "n_total")
    if not all(hasattr(comp, a) for a in need) or not hasattr(comp, "_block_role_scores"):
        return None
    b, D, Pd, V, NP = comp.b, comp.D, comp.period, comp.V, comp.NP
    from sim.backend import to_host
    comp._zero_rf_v_u()
    trig = comp.store_base + int(block_idx) * comp.block
    kick = np.zeros(comp.n_total, dtype=np.complex128)
    kick[trig] = 1.0
    b.rf_set_complex_weights(comp.store_conns)
    b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=comp.rf_mask)
    b.rf_resonate_steps(Pd + 8)
    unbind = []
    for ri, role in enumerate(comp.bind_roles):
        zc = comp._unbind_conj(role)
        unbind += [(comp.q_base + ri * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
    b.rf_set_complex_weights(unbind)
    b.rf_resonate_steps(Pd + 8)
    clean = []
    for ri, role in enumerate(comp.main_roles):
        for j in range(V):
            cc = comp._cleanup_conj(comp.words[j])
            clean += [(comp.c_base + ri * V + j, comp.q_base + ri * D + k, complex(cc[k])) for k in range(D)]
    pol_ri = comp.bind_roles.index("polarity")
    for j in range(NP):
        cc = comp._cleanup_conj(comp.pol_words[j])
        clean += [(comp.c_base + comp.n_main * V + j, comp.q_base + pol_ri * D + k, complex(cc[k])) for k in range(D)]
    b.rf_set_complex_weights(clean)
    b.rf_resonate_steps(1)
    mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
    out = {}
    for ri, role in enumerate(comp.main_roles):
        out[role] = (np.maximum(mem[comp.c_base + ri * V:comp.c_base + (ri + 1) * V], 0.0), list(comp.words))
    base = comp.c_base + comp.n_main * V
    out["polarity"] = (np.maximum(mem[base:base + NP], 0.0), list(comp.pol_words))
    return out


def spiking_pick(inner, scores) -> tuple:
    """The concept units' competition for one role, on the composer's Izhikevich concept bank: drive = scores / peak x
    `_margin_drive_pA` (the graded operating point), run `_cleanup_window` steps, count spikes. Returns
    (winner index or None, sorted top-two spike counts). None when the bank is silent or the top count is tied -- a
    competition that does not resolve reinstates nothing (no host argmax fallback)."""
    s = np.maximum(np.asarray(scores, dtype=float), 0.0)
    V = s.size
    if V == 0:
        return None, (0.0, 0.0)
    peak = float(s.max())
    if peak <= 1e-9:
        return None, (0.0, 0.0)
    drive = (s / peak) * float(inner._margin_drive_pA)
    bank = inner._izh_bank(V)
    bank.cp_membrane_potential_v[:] = bank._cleanup_v0      # reset to resting -> each competition is independent
    bank.cp_recovery_variable_u[:] = bank._cleanup_u0
    from sim.backend import get_backend, to_host
    xp, _ = get_backend()
    bank.cp_external_input_current[:] = xp.asarray(drive, dtype=bank.cp_external_input_current.dtype)
    firing = np.zeros(V)
    for _ in range(int(inner._cleanup_window)):
        bank._run_one_simulation_step()
        firing += np.asarray(to_host(bank.cp_firing_states)).astype(float)
    bank.cp_external_input_current[:] = 0.0
    top = np.sort(firing)[::-1]
    t2 = (float(top[0]), float(top[1]) if V > 1 else 0.0)
    if top[0] <= 0.0 or (V > 1 and top[0] == top[1]):
        return None, t2
    return int(np.argmax(firing)), t2


def completion_read(comp, block_idx: int, blk: dict) -> Optional[dict]:
    """One awake reactivation of block `block_idx` with pattern completion. Returns {"R": the partial cue's own read
    (Amendment 4), "R_c": the reinstated ensemble's in-phase coherence with the block's stored increment, "items":
    {role: reinstated word or None}, "spikes": {role: [top, runner_up]}, "coherence_abs": |mean conj(d) z|} or None
    when the composer has no block-structured read."""
    r = reactivation_strength(comp, int(block_idx))
    if r is None:
        return None
    sc = _role_scores(comp, int(block_idx))
    inner = getattr(comp, "comp", None)
    if sc is None or inner is None or not hasattr(inner, "_izh_bank"):
        return None
    items, spikes, fillers, roles = {}, {}, [], []
    for role in COMPLETION_ROLES:
        if role not in sc or role not in comp.bind_roles:
            continue
        scores, vocab = sc[role]
        j, t2 = spiking_pick(inner, scores)
        w = None if j is None else vocab[j]
        if isinstance(w, str) and (w.startswith("__free") or w not in inner.concepts):
            w = None                                   # an unrecruited slot / a word with no code is never reinstated
        items[role] = w
        spikes[role] = [t2[0], t2[1]]
        if w is not None:
            fillers.append(w)
            roles.append(role)
    order = [x for x in comp.bind_roles if x in roles]          # canonical bind order (as the write)
    fillers = [fillers[roles.index(x)] for x in order]
    inc = np.asarray(blk["inc"], dtype=np.complex128)
    d = inc / np.maximum(np.abs(inc), 1e-12)
    if not order:
        return {"R": float(r), "R_c": 0.0, "coherence_abs": 0.0, "items": items, "spikes": spikes, "n_items": 0}
    z = np.asarray(comp._compose_phases(fillers, order), dtype=np.complex128)
    m = complex(np.mean(np.conj(d) * z))
    return {"R": float(r), "R_c": float(min(1.0, max(0.0, m.real))), "coherence_abs": float(abs(m)),
            "items": items, "spikes": spikes, "n_items": len(order)}
