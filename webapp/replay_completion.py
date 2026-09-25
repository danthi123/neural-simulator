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
  2. ITEM COMPETITION (spiking, assembly-coded; Amendment 8 addendum 8a). The same substrate read gives each concept
     unit its matched-filter drive (the rectified cleanup membrane; `_role_scores`, the ops of
     `OneBrainComposer._block_role_scores`). Per role the drive, divided by its own peak (the feedback-inhibition
     normalization the composer's spiking cleanup already uses), drives an ASSEMBLY of `ASSEMBLY_CELLS` cells per
     candidate item in the composer's Izhikevich concept bank (`_izh_bank`: the same cell model, heterogeneity and seed,
     sized V x ASSEMBLY_CELLS) at `_margin_drive_pA` for the cleanup window. Each assembly's pooled spike count is its
     item's evidence. The reinstated item is the assembly that fires most, and only if its pooled count exceeds the
     runner-up's by at least `DISCRIMINATION_G` of its own count; a silent, tied or unresolved competition reinstates
     nothing for that role (no host argmax fallback).
     WHY assemblies (measured before this change, disclosed in addendum 8a): with ONE cell per item the bank's
     per-cell excitability spread (CV 0.18 of the spike count at 300 pA; 3-11 spikes over 120 steps at an identical
     drive) is as large as the drive differences it has to resolve, so the most excitable cell, not the stored item,
     won: 44 of 273 dev role reads disagreed with the matched-filter read, seed 14 tied 'cat' 4:4 at a 0.519 margin,
     and seed 3 reinstated the wrong word 'ball' in the agent slot at e = 0.3. "When many neurons contribute to the
     discrimination, the signal-to-noise ratio increases" (Kandel 6e ch.21, p.518; binding
     research/biology/awake-replay-pattern-completion.md): the stored item is an assembly, and its pooled rate
     averages out the single cells' excitability. With 64 cells per assembly the pooled-count CV is 0.027, and an
     equal drive to two assemblies of the bank did not separate them by `DISCRIMINATION_G` in 2000 measured pairs
     (largest 0.105), so an equal drive abstains instead of letting the most excitable unit win.
  3. IGNITION REQUIRES EVERY CONTENT ROLE (addendum 8a; unanimous, not a majority). The burst reinstates only when
     ALL `IGNITION_MIN_ITEMS` = 3 content roles resolve; otherwise nothing is reinstated for ANY role and R_c = 0,
     even for a role that did resolve on its own. de la Prida et al. 2006: a population burst starts at a threshold
     level of population firing, not a single unit's; Kandel ch.54 / Marr: completion runs from a SUBSET of the
     stored assembly (the biology gives a threshold, not its value). Two thresholds were tried and measured, in
     order, both disclosed in addendum 8a: (i) NO ignition threshold (discrimination alone) reinstated the bare
     baseline's crosstalk word 'brain' on dev seed 1 (e <= 0.03: the matched filter itself decodes it past
     `DISCRIMINATION_G`, the other two roles silent); (ii) a MAJORITY (2 of 3) closed that but, on a pre-existing
     3-fact vocabulary (dev seed 7's own composer, `tests/test_awake_replay_completion.py`), let a genuinely wrong
     item through at e = 0.3: two roles resolved, one of them decisively to a word borrowed from a DIFFERENT stored
     fact. Requiring all three is what stays clear of both (measured against the validated 15-dev-seed corpus,
     `research/findings/raw/_awake_replay_completion_dev/scan_assembly64/`: zero wrong reinstatements at either
     threshold, same count of fully-resolved facts, 77/255 -- unanimity's only measured cost there is the 29 blocks
     that had resolved exactly 2 of 3, which no longer ignite).
  4. REINSTATEMENT (the completion). The items that survive (2)+(3) are re-bound to their roles and bundled on the
     composer's own resonate-and-fire work registers -- `OneBrainComposer._compose_phases`, the SAME substrate op that
     encoded the fact -- and read back as spike phases: the ensemble the burst reinstates at the block's readout.
  5. PAIRING COUNT. R_c = the in-phase coherence between the reinstated readout pattern and the block's own stored
     increment, Re(mean_k conj(d_k) z_k) clipped to [0, 1] (d = inc / |inc|): the fraction of the block's synapses
     whose post cell reinstates in phase with the synapse's increment (Sadowski's LTP-competent pairings). When the
     three content items are reinstated correctly the reinstated pattern is the stored composite less its polarity
     component (R_c ~0.75-0.79 measured; see COMPLETION_ROLES); a wrong item lowers it; a trace that no longer
     selects its items reinstates an unrelated pattern and R_c ~ 0. Unlike the margin R, R_c knows WHICH items the
     read selected: a decayed block whose cleanup confidently selects a wrong word (or a reserved slot) has a margin R
     but no coherence with its own increment.
  6. The route then uses R_c where it used R, and nothing else changes. Awake bout: e <- e + R_c (1 - e), the tag
     re-set to the same level (no PRP, z untouched, the awake-edge lesion still zeroes R_eff). Night epoch: the replay
     tag R_c x |inc|, the SWR-coupled DA tonic + span x min(1, sum R_c) onto the same spiking D1 pool, and (under the
     downscaling sub-flags) the protection 1 - delta (1 - R_c); both replay/DA lesions act on it unchanged.

THE IGNITION POINT (corrected in addendum 8a; two constants, not none). Per role, an item is resolved while the
block's own read still separates it from every other candidate by the discrimination criterion `DISCRIMINATION_G` =
0.15 on the assemblies' pooled spike counts -- the composer's validated clean/noise separator on the same
normalized-decisiveness form (peak - runner_up) / peak (`confidence_gate` g = 0.15,
research/findings/2026-06-18-emergent-graceful-degradation-derisk.md), reused, not fitted here. The BURST ignites,
and a resolved item is actually reinstated, only when ALL `IGNITION_MIN_ITEMS` = 3 content roles resolve (de la
Prida et al. 2006; Kandel ch.54 / Marr) -- unanimous, not a majority: a 2-of-3 majority was tried first and withdrawn
after it let a genuinely wrong item through (measured on a pre-existing 3-fact vocabulary; see step 3 above). The
expression e at which a block stops igniting still differs from block to block (it depends on the decayed increment
against the baseline synapses and on the vocabulary's crosstalk), but it is set by these two constants. The first
build's statement that "no threshold is added" was wrong as built: with one cell per item the threshold was the
bank's per-cell excitability, and with no ignition requirement a single resolved role (crosstalk, not the fact)
could reinstate on its own. Kandel 6e ch.21 (p.518): a change in response must "significantly exceed the normal
variability in the response"; a difference inside the competition's own resolution is not a discrimination, so it
reinstates nothing.

THE COMPLETION LESION. `BRAIN_REPLAY_COMPLETION_LESION=1` cuts the completion's effect in BOTH routes: every read
above still runs (same compute, same substrate state) and is recorded, but each route uses the partial cue's own R,
i.e. exactly the Amendment-4 awake route and the rc/r2 night route.

HOST SHORTCUTS (declared, brain-based-only burn-down):
  - the op sequencing (read -> competition -> re-bind -> read-back) is host dispatch of substrate ops, as every
    composer op is; ONE pass (T = 1): no multi-cycle settle within the ripple;
  - the per-role peak normalization of the drive (the divisive feedback-inhibition stand-in the composer's spiking
    cleanup already uses), the pooled count of each assembly, and the read of the winner as the assembly with the
    largest pooled count, reinstated only when its lead over the runner-up is at least `DISCRIMINATION_G` of its own
    count (host arithmetic on the bank's spike counts). The bank's cells are independent (no lateral inhibition): the
    criterion stands in for the resolution a feedback-inhibition competition would reach itself (Buzsaki 2006, p.63:
    lateral inhibition works by "suppressing the similarly activated neighboring neurons"), the named next rung;
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

# ASSEMBLY CODING (addendum 8a). Cells per candidate item in the concept bank. Chosen a priori from the bank itself, not
# from a dev-seed outcome: at `_margin_drive_pA` a single cell's 120-step spike count has CV 0.182 across the bank's
# heterogeneous cells (4096 cells, seed 7), so an assembly of N cells has a pooled-count CV of ~0.182 / sqrt(N) and the
# difference of two assemblies' pooled excitability has sd ~0.182 x sqrt(2 / N). N = 64 puts 4 sd of that difference
# at 0.152, i.e. at the discrimination criterion below (measured on the same bank: pooled CV 0.027; the largest
# equal-drive normalized lead over 2000 random assembly pairs 0.105).
ASSEMBLY_CELLS = 64
# The discrimination criterion on the pooled counts' normalized lead (peak - runner_up) / peak: the composer's validated
# clean/noise separator on the same form (confidence_gate g = 0.15, 2026-06-18-emergent-graceful-degradation-derisk.md),
# reused, not fitted. Below it the competition has not resolved one item and the role reinstates nothing.
DISCRIMINATION_G = 0.15
# IGNITION REQUIRES ALL CONTENT ROLES (addendum 8a; unanimous -- every value tried below 3 was measured and
# withdrawn). The burst reinstates only when EVERY one of the fact's content roles resolves on its own (de la Prida
# et al. 2006: a population burst starts at a threshold level of POPULATION firing; Kandel ch.54 / Marr: completion
# runs from a SUBSET of the stored assembly -- the biology gives a threshold, not its value). Found necessary in two
# measured steps, both disclosed: with the discrimination criterion alone (no ignition threshold), a bare baseline's
# crosstalk decisively resolved ONE role ('brain' in the action slot, dev seed 1) while the other two stayed silent,
# and that single item was reinstated as a false memory; a MAJORITY (2 of 3) closed that, but on a pre-existing
# 3-fact vocabulary (dev seed 7's own composer) two roles resolved at e = 0.3 with one of them decisively wrong (a
# word borrowed from a different stored fact) and still ignited. Requiring all three costs nothing measured on the
# validated 15-dev-seed corpus (same 77/255 fully-resolved facts as at 2-of-3; only the blocks that had resolved
# exactly 2 of 3 no longer ignite there).
IGNITION_MIN_ITEMS = 3
_BANK_STATE_ZERO = ("cp_refractory_timers", "cp_firing_states", "cp_prev_firing_states")


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


def read_blocks(comp, ledger, rng_ctx, seed, k_base, reactivate_fn=None):
    """The completion read of every managed block, each inside rng_ctx(seed, k_base + i) exactly as the route's own
    read loop. `reactivate_fn` is the route's own partial-cue read (its injected `reactivate_fn`; default
    `sleep_replay_capture.reactivation_strength`). Returns (R list, R_read list, records): R = the partial cue's read,
    R_read = what the route uses (R_c, or R under the completion lesion), records = the per-block completion dicts."""
    use_c = not completion_lesioned()
    R, R_read, recs = [], [], []
    for i in range(len(ledger.blocks)):
        with rng_ctx(seed, k_base + i):
            c = completion_read(comp, ledger.block_offset + i, ledger.blocks[i], reactivate_fn=reactivate_fn)
        recs.append(c)
        R.append(None if c is None else float(c["R"]))
        R_read.append(None if c is None else (float(c["R_c"]) if use_c else float(c["R"])))
    return R, R_read, recs


def record(recs) -> list:
    """The per-block completion record a route stores (rounded like the route's own fields). `resolved` is each
    role's own competition winner (addendum 8a) whether or not the burst ignited; `items` is what was actually
    reinstated (== resolved when `ignited`, else all None) -- kept apart so a disclosed single-item resolve that did
    NOT ignite is visible on the record, not silently identical to a role that never resolved."""
    return [None if c is None else {"R_c": round(c["R_c"], 9), "coherence_abs": round(c["coherence_abs"], 9),
                                    "items": c["items"], "resolved": c.get("resolved", c["items"]),
                                    "ignited": c.get("ignited"), "spikes": c["spikes"], "n_items": c["n_items"]}
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


def _assembly_bank(inner, V: int):
    """The composer's own Izhikevich concept bank (`RFPhasorComposer._izh_bank`: the same cell model, heterogeneity and
    seed as its spiking cleanup), sized V x ASSEMBLY_CELLS; candidate j is the assembly of cells
    [j * ASSEMBLY_CELLS, (j + 1) * ASSEMBLY_CELLS)."""
    return inner._izh_bank(int(V) * ASSEMBLY_CELLS)


def _reset_bank(bank) -> None:
    """Every state array a step mutates back to rest (v, u to the build-time snapshot; refractory timers and firing
    flags to zero) and the drive off, so each competition is independent of any earlier use of this cached bank (the
    C2 read-isolation class: resetting v/u alone leaves refractory residue)."""
    bank.cp_membrane_potential_v[:] = bank._cleanup_v0
    bank.cp_recovery_variable_u[:] = bank._cleanup_u0
    for nm in _BANK_STATE_ZERO:
        a = getattr(bank, nm, None)
        if a is not None:
            a[:] = 0
    bank.cp_external_input_current[:] = 0.0


def assembly_counts(inner, scores) -> Optional[np.ndarray]:
    """Run the item competition for one role; return each candidate assembly's pooled spike count (length V), or None
    when the drive is empty or zero. Drive = scores / peak x `_margin_drive_pA`, the same current to every cell of a
    candidate's assembly, for `_cleanup_window` steps of the bank."""
    s = np.maximum(np.asarray(scores, dtype=float), 0.0)
    V = s.size
    if V == 0:
        return None
    peak = float(s.max())
    if peak <= 1e-9:
        return None
    drive = np.repeat((s / peak) * float(inner._margin_drive_pA), ASSEMBLY_CELLS)
    bank = _assembly_bank(inner, V)
    from sim.backend import get_backend, to_host
    xp, _ = get_backend()
    _reset_bank(bank)
    bank.cp_external_input_current[:] = xp.asarray(drive, dtype=bank.cp_external_input_current.dtype)
    firing = np.zeros(V * ASSEMBLY_CELLS)
    for _ in range(int(inner._cleanup_window)):
        bank._run_one_simulation_step()
        firing += np.asarray(to_host(bank.cp_firing_states)).astype(float)
    _reset_bank(bank)
    return firing.reshape(V, ASSEMBLY_CELLS).sum(axis=1)


def spiking_pick(inner, scores) -> tuple:
    """The item competition for one role (step 2 of the module docstring). Returns (winner index or None, (top,
    runner_up) pooled spike counts). None when the bank is silent, or when the top assembly's lead over the runner-up
    is below `DISCRIMINATION_G` of its own count (a tie included): a competition that does not resolve one item
    reinstates nothing (no host argmax fallback)."""
    pooled = assembly_counts(inner, scores)
    if pooled is None:
        return None, (0.0, 0.0)
    order = np.argsort(-pooled, kind="stable")
    top = float(pooled[order[0]])
    run = float(pooled[order[1]]) if pooled.size > 1 else 0.0
    if top <= 0.0 or (top - run) / top < DISCRIMINATION_G:
        return None, (top, run)
    return int(order[0]), (top, run)


def select_items(inner, sc: dict) -> dict:
    """The item competition for every role in `sc` (step 2), then the ignition requirement (step 3). `sc` = {role:
    (scores, vocab)}, exactly `_role_scores`'s return, restricted by the caller to the roles it wants resolved.
    Returns {"resolved": {role: word or None} -- each role's OWN competition winner (a reserved slot or a word
    without a code counts as None, no code growth), "spikes": {role: [top, runner_up]}, "ignited": bool, "items":
    {role: word or None} -- `resolved` when `ignited`, else every role forced to None}. A role's own resolve is
    never enough on its own: the burst needs ALL `IGNITION_MIN_ITEMS` of them to resolve (addendum 8a; unanimous,
    not a majority -- a 2-of-3 majority let a genuinely wrong item through, measured)."""
    resolved, spikes = {}, {}
    for role, (scores, vocab) in sc.items():
        j, t2 = spiking_pick(inner, scores)
        w = None if j is None else vocab[j]
        if isinstance(w, str) and (w.startswith("__free") or w not in inner.concepts):
            w = None                                   # an unrecruited slot / a word with no code is never reinstated
        resolved[role] = w
        spikes[role] = [t2[0], t2[1]]
    ignited = sum(1 for w in resolved.values() if w is not None) >= IGNITION_MIN_ITEMS
    items = dict(resolved) if ignited else {role: None for role in resolved}
    return {"resolved": resolved, "spikes": spikes, "ignited": ignited, "items": items}


def completion_read(comp, block_idx: int, blk: dict, reactivate_fn=None) -> Optional[dict]:
    """One awake reactivation of block `block_idx` with pattern completion. Returns {"R": the partial cue's own read
    (Amendment 4), "R_c": the reinstated ensemble's in-phase coherence with the block's stored increment, "items":
    {role: reinstated word or None} (all None unless the burst ignited), "resolved": {role: that role's OWN
    competition winner, ignited or not}, "ignited": bool, "spikes": {role: [top, runner_up]}, "coherence_abs":
    |mean conj(d) z|} or None when the composer has no block-structured read."""
    r = (reactivate_fn or reactivation_strength)(comp, int(block_idx))
    if r is None:
        return None
    sc = _role_scores(comp, int(block_idx))
    inner = getattr(comp, "comp", None)
    if sc is None or inner is None or not hasattr(inner, "_izh_bank"):
        return None
    sc = {role: v for role, v in sc.items() if role in COMPLETION_ROLES and role in comp.bind_roles}
    sel = select_items(inner, sc)
    items, spikes, ignited = sel["items"], sel["spikes"], sel["ignited"]
    roles = [role for role in comp.bind_roles if items.get(role) is not None]     # canonical bind order (as the write)
    fillers = [items[role] for role in roles]
    inc = np.asarray(blk["inc"], dtype=np.complex128)
    d = inc / np.maximum(np.abs(inc), 1e-12)
    if not roles:
        return {"R": float(r), "R_c": 0.0, "coherence_abs": 0.0, "items": items, "resolved": sel["resolved"],
                "ignited": ignited, "spikes": spikes, "n_items": 0}
    z = np.asarray(comp._compose_phases(fillers, roles), dtype=np.complex128)
    m = complex(np.mean(np.conj(d) * z))
    return {"R": float(r), "R_c": float(min(1.0, max(0.0, m.real))), "coherence_abs": float(abs(m)),
            "items": items, "resolved": sel["resolved"], "ignited": ignited, "spikes": spikes, "n_items": len(roles)}
