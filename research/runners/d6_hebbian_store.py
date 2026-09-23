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
  * reconsolidation's in-place rewrite (`update_on_mismatch`) still uses the direct copy (not routed here).

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
    composer._d6_conv_write = True
    try:
        yield
    finally:
        composer._d6_conv_write = prev


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


def comp_backend_xp(arr):
    """numpy or cupy module matching `arr` (the bridge's device arrays)."""
    try:
        import cupy as _cp  # noqa: F401
        if type(arr).__module__.startswith("cupy"):
            return _cp
    except Exception:
        pass
    return np
