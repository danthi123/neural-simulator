"""D6 — LEARN-THROUGH-USE: the in-conversation fact WRITE by a LOCAL HEBBIAN RULE on the spiking substrate.

WHY (charter D6, docs/plans/2026-09-23-autonomous-charter.md; ledger row `in-loop-learning`). The production in-loop
learning path (the owner teaches "the wolf hunts the deer" -> later "what does the wolf hunt?" -> "deer") stores the
fact in `OneBrainComposer._write_block`: the bound composite is computed ON the substrate (bind + bundle through RF
synapses) but then READ TO HOST as phases and COPIED into the block's trigger->readout synapses as
`w_k = g * exp(2*pi*i*phase_k)`. The synaptic weight is therefore HOST-DESIGNED (a copy of an activity pattern into a
weight), not the output of a plasticity rule — the ledger's own named residual ("no lasting per-turn BTSP/plasticity
write by default"). Under the brain-based-only standard that is a shortcut.

WHAT THIS MODULE DOES (default-OFF, byte-identical when off). With `BRAIN_D6_HEBBIAN_STORE=1` the SAME fact block is
written by a LOCAL, activity-dependent rule instead of the copy:
  1. COMPOSE — the identical on-substrate bind/bundle op sequence `_compose_phases` runs (fill -> bound -> acc), but
     the composite is NOT read out to host: it stays as live oscillation in the `acc` register.
  2. ENCODING EPISODE — the acc register drives the new block's D readout cells through a one-to-one INSTRUCTIVE
     pathway (the analogue of the BTSP plateau / mossy-fiber "detonator": the teaching input that makes the
     postsynaptic cell fire in the pattern to be stored), while the block's TRIGGER (context/index) cell is
     activated. Both oscillate on the resonate-and-fire substrate for one encoding window.
  3. PLASTICITY — every trigger->readout_k synapse accumulates the complex (phase-coupled) Hebbian correlation of
     ITS OWN pre and post membrane phasors, dw_k = eta * z_post_k(t) * conj(z_pre(t)), read off the bridge's
     cp_membrane_potential_v / cp_recovery_variable_u each step. The weight's PHASE — the stored content — is the
     rule's output (phase-difference coding: the RF analogue of spike-timing-dependent potentiation, Izhikevich 2001
     resonate-and-fire; Hebbian 1949). No host copy of the composite enters the weight.
  4. BOUND — the magnitude saturates at w_max (a per-synapse saturating bound, like btsp_w_max). DECLARED CONSTANT:
     the RF recall is phase-based (magnitude-invariant above the read floor), so the bound sets only the
     magnitude, not the content; the measured content (phase) is 100% the Hebbian correlation. The DA encoding
     gain `g` is then applied by the unchanged `_write_block` exactly as on the direct path.

THE D6 LESION — `BRAIN_D6_HEBBIAN_FREEZE=1` sets eta=0 FOR IN-CONVERSATION WRITES ONLY (writes made inside the
`conversation_write(composer)` context, i.e. `ChatBrain._maybe_acquire`): the encoding episode RUNS identically (same
compose, same instructive drive, same trigger activation, same activity), but no synapse changes. Build-time
(developmental) facts are unaffected, so the recall READ path stays intact — the lesion is a pure WRITE freeze, the
exact contrast D6 needs ("the brain changes from use, and freezing the plasticity removes the change").

HONEST RESIDUALS (named, not hidden):
  * the rule is evaluated in this runner module from the bridge's membrane state (a local pre x post product per
    synapse), not inside a sim/ fused kernel (NO sim/ edit; a sim/ kernel is a pure speed follow-on);
  * WHICH block index a new fact takes (the next free trigger cell) is host bookkeeping (the kb list order), like
    the episodic organ's slot table;
  * the instructive pattern itself is the composer's on-substrate bind/bundle output — the FHRR binding algebra
    (role codes) is the standing composer idealization, unchanged by this module;
  * reconsolidation's in-place rewrite (`update_on_mismatch`) still uses the direct copy (not routed here) and so
    BYPASSES the freeze lesion; the D6 probe counts store writes after the teach turn and fails the lesion if any occur.

DECLARED HOST SHORTCUTS (2026-09-23 fix round, from the adversarial review; each is a named residual, not biology):
  (a) INSTRUCTIVE PATHWAY IS HOST-WIRED. The one-to-one acc->readout "teacher" synapses are installed by host code
      (`rf_set_complex_weights(teacher)`) fresh for each write, with unit weight. The learned phase is therefore a
      teacher-forced copy CARRIED THROUGH NEURAL ACTIVITY: the rule is local, but the instructive signal's wiring and
      its one-to-one topology are designed. Next method: a developmentally formed (plastic, competitive) CA3/EC->CA1-
      style instructive projection whose topology self-organizes, and a plateau that the substrate itself triggers.
  (b) PHASE-LOCK IS A HOST LOOP. `while _rf_counter % Pd != 0: _rf_advance_one()` advances the rhythm to the read's
      reference phase before the context cell is activated. The alignment target is not a tuned constant, but the
      gating is host control flow. Next method: theta-phase-gated encoding on the substrate (encoding at a fixed
      theta phase, Hasselmo 2002 SPEAR), i.e. an oscillatory inhibitory gate on the context cell.
  (c) MAGNITUDE IS CLAMPED. Every unfrozen synapse saturates at W_MAX (n_saturated == D on every write measured), so
      the rule stores NO graded strength -- only the phase is learned. A bound, not a homeostatic companion process
      (the CLAUDE.md "what does the real system run alongside this" question): the named companion is heterosynaptic
      /homeostatic normalization of the block.
  (d) THE PRUNE RETRACTION (`retract_unencoded_last`, BRAIN_D6_ENGRAM_PRUNE) IS A HOST DECISION about whether a
      memory exists, taken once at write time and acting by deleting host bookkeeping. BANKED as an invalid lesion
      instrument (it runs only in the freeze arm); superseded by the read-time view (`engram_readtime_enabled`).
  (e) The held/not-held decision of `engram_held` is a host threshold (`readout > floor`) on a neural read.
  (f) UNMEASURED SIDE EFFECT: the DA encoding gain `g` (`encoding_gain_fn`) scales a write; no D6 arm varies it, so
      whether a low-g production write can fall below the read floor (and so be "not held" / retracted) is unmeasured.
  (g) (fix round 3) THE READ-TIME VIEW IS PARTIAL. Only the four ENGRAM_READTIME_ROUTED readers consult the engram;
      ENGRAM_READTIME_NOT_ROUTED lists the host-kb readers that do not. Gate v3 measures (NOREC_H) rather than asserts
      that they are inert on the D6 protocol; off-protocol (e.g. the describe path) they are an open host shortcut.
      Next method: retire the kb list as a membership source entirely -- the block->words map becomes a learned
      readout of the trigger cell (index -> word codes) so there is no host record to leak.
  (h) THE BLOCK->WORDS MAP IS HOST BOOKKEEPING. After a substrate read picks a block, `kb[i]` maps it to words; the
      same holds for the fact shard and the seq fabric's fact list. Same next method as (g).
  (i) THE WRITE COUNTER COVERS `_write_block` ONLY. `apply_homeostatic_scaling` rewrites store_conns without it; the
      D6 worker counts those calls separately (`homeostatic_calls_after_teach`). Multiplicative scaling cannot
      un-zero a zeroed block, and the lever is read at probe time, so a lesion cannot be silently undone by it.
  (j) THE ABLATION AND RECORD-REMOVAL ARMS RUN EXPERIMENTER CODE (cache invalidation + a re-read of every block, and
      in NOREC_H a kb pop). They are lesions/controls, not brain mechanisms; the capability contrast (USE_H vs
      FREEZE_H) runs neither.

Reuse-by-import; no sim/ edit. See research/runners/d6_learn_through_use_lb.py for the 2x2 (use x plasticity)
lesion-verified probe through the real /api/brain-chat handler.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np

from sim.backend import to_host

_ON = ("1", "true", "yes", "on")

# The encoding window (resonate steps) and learning rate. The window matches the composer's own per-op resonate
# window (period + 8); eta is large enough that every synapse whose instructive drive is non-degenerate saturates at
# the bound within the window (the |post| grows ~linearly with t through the instructive pathway).
ETA = 1.0
W_MAX = 1.0


def hebbian_store_enabled() -> bool:
    """`BRAIN_D6_HEBBIAN_STORE` in {1,true,yes,on} -> fact blocks are written by the local Hebbian rule. Default OFF
    (unset) -> the direct composite copy, byte-identical to before this module existed."""
    return os.environ.get("BRAIN_D6_HEBBIAN_STORE", "").strip().lower() in _ON


def hebbian_freeze_lesioned() -> bool:
    """`BRAIN_D6_HEBBIAN_FREEZE` in {1,true,yes,on} -> eta=0 for IN-CONVERSATION writes (the D6 plasticity-freeze
    lesion). Only meaningful with BRAIN_D6_HEBBIAN_STORE on."""
    return os.environ.get("BRAIN_D6_HEBBIAN_FREEZE", "").strip().lower() in _ON


@contextlib.contextmanager
def conversation_write(composer):
    """Mark writes made inside this block as IN-CONVERSATION (learning from use) so the freeze lesion targets only
    them. A no-op attribute toggle when the store flag is off (the attribute is never read then)."""
    if composer is None or not hebbian_store_enabled():
        yield
        return
    prev = getattr(composer, "_d6_conv_write", False)
    n_before = len(getattr(composer, "kb", []) or [])
    composer._d6_conv_write = True
    try:
        yield
    finally:
        composer._d6_conv_write = prev
    # (BRAIN_D6_ENGRAM_PRUNE) an in-conversation encode that formed no engram leaves no bookkeeping record.
    if engram_prune_enabled() and len(getattr(composer, "kb", []) or []) > n_before:
        composer._d6_last_retract = retract_unencoded_last(composer)


def hebbian_encode(comp, block_idx, fillers, roles, *, freeze=False, eta=ETA, w_max=W_MAX):
    """Write-side of the D6 rule for ONE fact block on a OneBrainComposer `comp`. Returns (w, diag) where `w` is the
    (D,) complex weight vector for trigger->readout_k learned by the rule (0 when frozen) and `diag` records the
    episode (the instructive/composite phase agreement is measured by the caller's tests, not assumed)."""
    ob, b, D, P, Pd = comp.comp, comp.b, comp.D, comp.P, comp.period
    n = len(roles)
    acc = 2 * n
    # ---- 1. COMPOSE on the substrate (the identical op sequence of OneBrainComposer._compose_phases) -------------
    binds, bundle = [], []
    kick = np.zeros(comp.n_total, dtype=np.complex128)
    for i in range(n):
        zr = ob._to_phasor(ob.roles[roles[i]]); zf = ob._to_phasor(ob._filler_phases(fillers[i]))
        kick[P + i * D:P + (i + 1) * D] = zf
        binds += [(P + (n + i) * D + k, P + i * D + k, complex(zr[k])) for k in range(D)]
        bundle += [(P + acc * D + k, P + (n + i) * D + k, 1.0) for k in range(D)]
    comp._zero_rf_v_u()
    b.rf_set_complex_weights(binds); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=comp.rf_mask)
    b.rf_resonate_steps(Pd + 8)
    b.rf_set_complex_weights(bundle); b.rf_resonate_steps(Pd + 8)
    # ---- 2. ENCODING EPISODE: acc -> readout instructive pathway + trigger (context) cell activation -------------
    trig = comp.store_base + block_idx * comp.block
    acc_lo = P + acc * D
    teacher = [(trig + 1 + k, acc_lo + k, 1.0) for k in range(D)]
    b.rf_set_complex_weights(teacher)
    v, u = b.cp_membrane_potential_v, b.cp_recovery_variable_u
    xp = comp_backend_xp(v)
    v[trig + 1:trig + 1 + D] = 0.0; u[trig + 1:trig + 1 + D] = 0.0      # the new block's cells start at rest
    # PHASE-LOCKED context activation: the retrieval cue activates the trigger cell at the START of a substrate
    # oscillation cycle (rf_kick resets the rhythm counter to 0). The encoding activation must fire at that SAME
    # rhythm phase, or every learned weight carries the encode-vs-retrieve phase difference as a uniform rotation
    # (measured 2026-09-23: activating at counter 416 = 2 cycles + 16 steps rotated every synapse by ~0.49 rad and
    # cut the cleanup margins ~12-30%). So let the rhythm run (plasticity idle: the context cell is silent, pre=0)
    # to the next cycle boundary, then activate it. Not a tuned constant — the alignment target is the read's own
    # reference phase (0).
    pre_steps = 0
    while int(getattr(b, "_rf_counter", 0)) % Pd != 0:
        b._rf_advance_one(); pre_steps += 1
    v[trig] = 1.0; u[trig] = 0.0                                         # unit stimulus to the block's context cell
    # ---- 3. LOCAL HEBBIAN accumulation: each synapse sees only its own pre (trigger) and post (readout_k) state --
    S = xp.zeros(D, dtype=xp.complex128)
    steps = Pd + 8
    lr = 0.0 if freeze else float(eta)
    for _ in range(steps):
        b._rf_advance_one()
        if lr == 0.0:
            continue                                                     # FROZEN: identical activity, no weight change
        post = v[trig + 1:trig + 1 + D].astype(xp.float64) + 1j * u[trig + 1:trig + 1 + D].astype(xp.float64)
        pre = complex(float(v[trig])) + 1j * complex(float(u[trig]))
        S += lr * post * np.conj(pre)
    S = np.asarray(to_host(S), dtype=np.complex128)
    # ---- 4. per-synapse saturating bound (magnitude only; the PHASE = the content is the Hebbian correlation) ----
    mag = np.abs(S)
    scale = np.where(mag > w_max, w_max / np.maximum(mag, 1e-300), 1.0)
    w = S * scale
    comp._zero_rf_v_u()
    diag = {"frozen": bool(lr == 0.0), "steps": int(steps), "block": int(block_idx), "phase_lock_steps": pre_steps,
            "n_saturated": int(np.sum(mag > w_max)), "mean_abs_w": float(np.mean(np.abs(w))),
            "min_abs_w": float(np.min(np.abs(w))) if w.size else 0.0}
    return w, diag


def engram_vocab_enabled() -> bool:
    """`BRAIN_D6_ENGRAM_VOCAB` in {1,true,yes,on} -> the chat brain's KNOWN-FACT / known-word sets are derived from
    the engrams that actually REACTIVATE on the substrate, not from the host `kb` bookkeeping list. Default OFF.

    WHY (the 2026-09-23 D6 seed-42 smoke): with the fact WRITE frozen, the recall correctly abstained, but the reply
    still carried a trace of use -- the taught word ('wolf') read FAMILIAR (curiosity novelty 0.0 vs 0.97 in the
    shuffled control), was a GROUNDED common-ground topic, and triggered a thread swap -- because `hear()` appends
    the heard fact to the host `kb` list whether or not any synapse changed, and `ChatBrain._refresh_facts` builds
    `agents_set`/`actions_set`/`patients_set` (read by curiosity novelty, common ground, the thread swap) from that
    list. That is a host record doing the brain's remembering. With this flag the sets come from the substrate."""
    return os.environ.get("BRAIN_D6_ENGRAM_VOCAB", "").strip().lower() in _ON


def engram_held(comp, block_idx) -> dict:
    """Does fact block `block_idx` exist AS AN ENGRAM? Kick its trigger (context) cell, resonate one window, and read
    the mean |Z| over its D readout cells off the membrane (`OneBrainComposer._measure_block_readout`, a genuine
    neural read). The block is held iff that activity clears the substrate's own spike floor (`rf_kick`'s floor:
    a readout below it can never cross -> the engram is physically unrecallable). A frozen (never-potentiated)
    block reads exactly 0. The threshold is the read's own floor, not a tuned constant."""
    a = float(comp._measure_block_readout(block_idx))
    floor = float(getattr(comp.b, "_rf_floor", 1.0e-3))
    comp._zero_rf_v_u()
    return {"block": int(block_idx), "readout": a, "floor": floor, "held": bool(a > floor)}


def engram_prune_enabled() -> bool:
    """`BRAIN_D6_ENGRAM_PRUNE` in {1,true,yes,on} -> after an IN-CONVERSATION write, if the new block formed NO engram
    (engram_held False), the host bookkeeping entry for it is RETRACTED (kb entry popped, the unclaimed block's
    trigger cell returned to the free pool). Default OFF.

    WHY (the 2026-09-23 engram-variant seed-42 smoke): reading the known-word sets off engrams closed the curiosity
    leak (novelty 0.97 frozen vs 0.97 shuffled), but other organs read `composer.kb` DIRECTLY (e.g.
    webapp/gnw_thought_swap._known_concepts -> thread swap, common ground, GNW stop), so the frozen arm still treated
    the taught word as a grounded topic. Rather than patch every reader, keep the bookkeeping list CONSISTENT with the
    substrate: an encoding attempt that left no engram leaves no record."""
    return os.environ.get("BRAIN_D6_ENGRAM_PRUNE", "").strip().lower() in _ON


def retract_unencoded_last(comp) -> dict | None:
    """If the LAST kb block formed no engram on the substrate, retract it: pop the kb entry, drop its D store synapses
    (store_conns is block-major, so the last block is the tail), and dirty every cache keyed on the store. Returns
    the engram read (with `retracted`), or None when there is nothing to check."""
    n = len(getattr(comp, "kb", []) or [])
    if n == 0 or not hasattr(comp, "_measure_block_readout"):
        return None
    r = engram_held(comp, n - 1)
    r["retracted"] = False
    if not r["held"]:
        _ops(comp)["retractions"] += 1
        D = comp.D
        comp.kb.pop()
        del comp.store_conns[(n - 1) * D:]
        comp._store_dirty = True; comp._store_csr = None; comp._persistent_dirty = True
        if getattr(comp, "_csr_cache", None) is not None:
            comp._csr_cache = {}
        if getattr(comp, "integrated_loop", False):
            comp._seq_dirty = True
            if getattr(comp, "_fused", False):
                comp._fused_dirty = True
        comp._fact_shard = None; comp._fact_shard_built_K = -1
        r["retracted"] = True
    return r


def engram_readtime_enabled() -> bool:
    """`BRAIN_D6_ENGRAM_READTIME` in {1,true,yes,on} -> EVERY reader of "which facts does the brain hold" consults the
    substrate AT READ TIME (each turn), not the host `kb` list and not a write-time snapshot. Default OFF.

    WHY (2026-09-23 fix round, the adversarial review of the prune variant). `BRAIN_D6_ENGRAM_PRUNE` checked the
    engram ONCE, at write time, and then deleted the host record -- a host step that ran ONLY in the freeze arm, so a
    prune-variant C3 pass could come from the lesion arm running host code the treatment arm never runs, and later
    loss of the engram (ablation, decay, interference) would never be reflected. This flag replaces that with a READ:
    no record is ever deleted; instead FOUR named kb readers (ENGRAM_READTIME_ROUTED: `ChatBrain._refresh_facts`,
    re-run at the start of every turn, `webapp/gnw_thought_swap._known_concepts`,
    `webapp/gnw_multistep_deliberation._all_concepts`, the episodic content lookup in `webapp/server.brain_reply`) see
    only the facts whose engram reactivates NOW. SCOPE (narrowed 2026-09-23, fix round 3): these four are NOT every kb
    reader -- ENGRAM_READTIME_NOT_ROUTED lists the ones that still iterate the host list in every arm; gate v3's NOREC_H
    arm measures whether any of them carries an unheld fact's record into a reply on the D6 protocol. Across the
    Hebbian arms the host code PATH is the same; the lesion arms differ from USE_H in eta (FREEZE_H) or in an
    experimenter step after the teach turn (ABL_H: `ablate_block` zeroes the synapses AND invalidates the store caches,
    which then forces a re-read of every block; NOREC_H: `remove_block_record`). See `visible_kb`.

    DECLARED SHORTCUT: the held/not-held decision is a HOST THRESHOLD (`readout > floor`) on a genuine neural read
    (the readout activity after kicking the block's context cell) -- the same class as an argmax over spike rates.
    The kb list still maps block index -> words (host bookkeeping); what it no longer decides is membership."""
    return os.environ.get("BRAIN_D6_ENGRAM_READTIME", "").strip().lower() in _ON


def _store_digest(comp):
    """A digest of the store's synaptic weights (the only thing an engram read depends on) + the kb length."""
    import hashlib
    sc = getattr(comp, "store_conns", None) or []
    w = np.fromiter((complex(t[2]) for t in sc), dtype=np.complex128, count=len(sc))
    return (len(getattr(comp, "kb", []) or []), len(sc), hashlib.sha256(w.tobytes()).hexdigest())


def _ops(comp):
    ops = getattr(comp, "_d6_ops", None)
    if ops is None:
        ops = {"engram_reads": 0, "view_cache_hits": 0, "turn_refreshes": 0, "retractions": 0, "ablations": 0}
        try:
            comp._d6_ops = ops
        except Exception:
            pass
    return ops


def held_view(comp) -> list:
    """Per-kb-index engram read (`engram_held`) of every block, recomputed whenever the store's weights change (cached
    on a digest of store_conns, so a turn that reads the view several times kicks each block once). A frozen, ablated,
    or decayed block is re-read, never remembered from a past write."""
    key = _store_digest(comp)
    cache = getattr(comp, "_d6_held_cache", None)
    ops = _ops(comp)
    if cache is not None and cache[0] == key:
        ops["view_cache_hits"] += 1
        return cache[1]
    reads = [engram_held(comp, i) for i in range(len(comp.kb))]
    ops["engram_reads"] += len(reads)
    comp._d6_held_cache = (key, reads)
    return reads


def visible_kb(comp):
    """What a kb reader should iterate. Flag OFF (or a composer without an engram read, e.g. the rate composer) ->
    `comp.kb` ITSELF (the same object: byte-identical). Flag ON -> the (fact, handle) entries whose engram is held now."""
    kb = getattr(comp, "kb", None)
    if kb is None or not engram_readtime_enabled() or not hasattr(comp, "_measure_block_readout"):
        return kb
    reads = held_view(comp)
    return [e for e, r in zip(kb, reads) if r["held"]]


def readtime_refresh(chat) -> None:
    """Start-of-turn re-read (BRAIN_D6_ENGRAM_READTIME): refresh the chat brain's known-fact / known-word sets off the
    engrams as they are NOW. Called from `webapp.server.brain_reply` for every turn in every arm (host pipeline
    identical across arms)."""
    comp = getattr(getattr(chat, "inner", None), "composer", None)
    if comp is None or not hasattr(chat, "_refresh_facts"):
        return
    _ops(comp)["turn_refreshes"] += 1
    chat._refresh_facts()


def ablate_block(comp, block_idx) -> dict:
    """EXPERIMENTER LESION (the post-hoc engram ablation arm ABL_H of research/runners/d6_learn_through_use_lb.py):
    zero block `block_idx`'s D trigger->readout synapses through the composer's own in-place write path, WITHOUT
    touching the kb record. Not a brain mechanism -- a lesion, like a post-training lesion in an animal."""
    D = comp.D
    before = float(np.mean([abs(complex(t[2])) for t in comp.store_conns[block_idx * D:(block_idx + 1) * D]]))
    g_fn = getattr(comp, "encoding_gain_fn", None)
    comp.encoding_gain_fn = None                      # zeros are zeros whatever g is; keep the write exact
    try:
        comp._write_block(block_idx, np.zeros(D, dtype=np.complex128))
    finally:
        comp.encoding_gain_fn = g_fn
    comp._store_csr = None
    if getattr(comp, "_csr_cache", None) is not None:
        comp._csr_cache = {}
    comp._fact_shard = None; comp._fact_shard_built_K = -1
    after = float(np.mean([abs(complex(t[2])) for t in comp.store_conns[block_idx * D:(block_idx + 1) * D]]))
    _ops(comp)["ablations"] += 1
    return {"block": int(block_idx), "mean_abs_w_before": before, "mean_abs_w_after": after}


def remove_block_record(comp, block_idx) -> dict:
    """EXPERIMENTER CONTROL (arm NOREC_H of gate v3, research/runners/d6_learn_through_use_lb.py): remove the HOST RECORD
    of block `block_idx` -- its kb entry and its D store synapses -- WITHOUT removing any synaptic content: it refuses
    unless every one of the block's weights is exactly 0 (a frozen write) and the block is the LAST one (store_conns is
    block-major, so removing the tail renumbers nothing). NOREC_H == FREEZE_H on every turn then shows the host record
    of a fact the synapses do not hold carries nothing into any reply -- a measurement of what `visible_kb` covers, in
    place of the claim that every kb reader consults the engram (it does not: see ENGRAM_READTIME_ROUTED)."""
    D = comp.D
    n = len(getattr(comp, "kb", []) or [])
    if block_idx != n - 1:
        raise ValueError("remove_block_record: block %s is not the last block (n=%d)" % (block_idx, n))
    ws = [complex(t[2]) for t in comp.store_conns[block_idx * D:(block_idx + 1) * D]]
    if len(ws) != D or any(w != 0 for w in ws):
        raise ValueError("remove_block_record: block %s holds non-zero synapses; this control removes a RECORD only"
                         % block_idx)
    fact = dict(comp.kb[block_idx][0])
    comp.kb.pop()
    del comp.store_conns[block_idx * D:]
    comp._store_dirty = True; comp._store_csr = None; comp._persistent_dirty = True
    if getattr(comp, "_csr_cache", None) is not None:
        comp._csr_cache = {}
    if getattr(comp, "integrated_loop", False):
        comp._seq_dirty = True
        if getattr(comp, "_fused", False):
            comp._fused_dirty = True
    comp._fact_shard = None; comp._fact_shard_built_K = -1
    _ops(comp)["record_removals"] = _ops(comp).get("record_removals", 0) + 1
    return {"removed": True, "block": int(block_idx), "fact": fact, "n_kb_after": len(comp.kb)}


# The kb readers that consult the engram at read time under BRAIN_D6_ENGRAM_READTIME (the ONLY ones; 2026-09-23 fix
# round 3 narrowed the earlier "every kb reader" wording, which the re-review showed was false).
ENGRAM_READTIME_ROUTED = (
    "research/runners/brain_chat_tui.ChatBrain._refresh_facts (stored_facts / agents_set / actions_set / patients_set)",
    "webapp/gnw_thought_swap._known_concepts",
    "webapp/gnw_multistep_deliberation._all_concepts",
    "webapp/server.brain_reply episodic in-memory content lookup",
)
# kb readers NOT routed (they still iterate the host kb list in every arm). Named, not hidden; gate v3's NOREC_H arm
# MEASURES whether any of them carries the record of an unheld fact into a reply on the D6 protocol.
ENGRAM_READTIME_NOT_ROUTED = (
    "research/runners/brain_conversational_agent._assoc_graph (elaborate / ordered_associates; the describe path)",
    "research/runners/one_brain_composer.OneBrainComposer._assoc_graph and ._relation_assoc (chain_of_thought)",
    "research/runners/one_brain_composer.OneBrainComposer._ensure_sequencer (the seq fabric's fact list, kb[:K])",
    "research/runners/one_brain_composer.OneBrainComposer.query_role / _attributed_patient role-presence checks",
    "research/runners/one_brain_composer.OneBrainComposer._calibrate_pe_labile and the fact shard (_fact_shard)",
    "research/runners/one_brain_composer.OneBrainComposer kb[i] index->word lookups after a substrate read "
    "(block -> words bookkeeping, declared shortcut)",
    "webapp/continuous_engine homeostatic pass trigger (reads len(kb))",
)


def comp_backend_xp(arr):
    """numpy or cupy module matching `arr` (the bridge's device arrays)."""
    try:
        import cupy as _cp  # noqa: F401
        if type(arr).__module__.startswith("cupy"):
            return _cp
    except Exception:
        pass
    return np
