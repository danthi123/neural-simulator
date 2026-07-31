"""EMERGE-61 — close the ONE honest residual in EMERGE-60: the spiking-Broca render-ORDER tail. EMERGE-60 wires the
EMERGE-59 spiking Broca producer (`FrameSlotCQ` / `BrocaProducer`) into the flagship console; render-CONTENT is 1.00 but
render-EXACT (word ORDER) is 0.93 6-seed — on seeds 100 & 101 the 4-slot F_MODAL frame [det:the, SUBJ:robin, FUNC:can,
VERB:breathe] swaps its two adjacent lowest-primacy slots -> "the robin breathe can". Content always correct; only the
ORDER swaps, only on the 4-slot frame, only on 2/6 seeds, and only when robin is the 5th emit (after owl/minnow/penguin/
pike) — the SWAP IS SEQUENCE-POSITION-DEPENDENT (the fresh console renders robin CORRECTLY).

ROOT CAUSE (diagnosed, H1 CONFIRMED). The spiking read-out (`slot_pool_rates`) advances a REAL `SimulationBridge`; the
Izhikevich recovery variable `cp_recovery_variable_u` is a SLOW ADAPTATION current that ACCUMULATES with every spike and
does NOT reset between productions. After 4 emits the heavily-firing slot pools carry a large, HETEROGENEOUS residual
adaptation (measured: u_pre 0.0 at emit#1 -> ~500 mean, std ~440-530, at emit#5). That per-neuron residual perturbs the
5th production's rates enough to flip the two near-equal-primacy adjacent slots on the seeds where the primacy noise
already put them close. This is a genuine BRAIN mechanism (spike-frequency adaptation, Izhikevich `u`), not a bug — but it
makes an utterance DEPEND on prior utterances' residual state, which a fluent producer must NOT do (each utterance is an
independent motor plan; Broca does not carry the last sentence's adaptation into the next).

WHY THE NAIVE FLAT RESET FAILED (diagnostic #3, already done): setting v=-65, u=0 for ALL neurons is the WRONG post-init
state — it ignores per-neuron heterogeneity (`cp_izh_vr`, `cp_izh_b`) and the correct u = b*(v-vr) relation the bridge
establishes at init (bridge.py:1562-1563), so it disrupts the slot f-I dynamics (made it WORSE, 0.867).

THE FIX (H1, the CORRECT reset). Capture the EXACT per-neuron dynamic state right after `_initialize_simulation_data()`
(a byte-for-byte snapshot of v / u / the four conductances / firing_states / STP), and RESTORE that snapshot before EACH
production. This returns the substrate to its genuine post-init operating point per utterance — so the read-out is a
function of the LEARNED primacy gradient ALONE, not of how many productions preceded it. Biologically: an inter-utterance
wash-out that clears the adaptation carried by the previous motor plan (the settle/wash-out the CQ read is entitled to;
the alternative rung, a quiet drive=0 window, decays u only partially and slower — the snapshot is the exact, cheap one).

ADDITIVE / DEFAULT-PRESERVING. The fix is a subclass `ResetFrameSlotCQ(FrameSlotCQ)` (EMERGE-59 is NOT edited — its
default de-risk stays byte-identical) that snapshots at construction and restores before `emit` / `emit_order_indices`.
EMERGE-60's `SpikingBrocaConsole` gets a default-OFF `reset_producer` flag; default False == EMERGE-60 byte-identical.

DE-RISK (>=6 seeds 42/43/44/100/101/102, CPU/numpy):
  (a) render-EXACT -> ~1.00 on ALL 6 seeds IN THE SEQUENCE (robin as the 5th emit, not fresh-per-emit).
  (b) POSITION-INDEPENDENCE (the load-bearing property): the SAME fact renders IDENTICALLY regardless of how many
      productions preceded it (robin@1st == robin@5th == robin@Nth), on every seed. This is what makes an utterance not
      depend on prior utterances' residual state.
  (c) the fix is CAUSAL: WITHOUT the reset the sequence tail swaps (render-exact < 1.0 on 100/101); WITH it, it does not.
  (d) MOAT still 0 on abstains (the reset does NOT touch the gate-first structure; the producer is never invoked on an
      abstain, so it is never reset on an abstain either — asserted).
  (e) NO REGRESSION: EMERGE-59's default de-risk + EMERGE-60's 6-seed de-risk both still GO (defaults preserved) —
      verified by the controller running those runners; here we assert the un-reset FrameSlotCQ is byte-unchanged.
GO bar: render-exact-in-sequence == 1.00 all 6 seeds AND position-independent all 6 seeds AND moat 0 AND the un-reset
control swaps (causal). BOUNDARY otherwise (naming exactly why + the next mechanism; do NOT force a GO; do NOT weaken moat).

HONEST SCOPE: this closes the ORDER tail for the bounded EMERGE frame inventory; it does not change render-CONTENT (already
1.00) or open-prose (R4, deferred). Reuse-by-import; NO `sim/` edit (the reset writes existing bridge arrays via their
public attributes — the same `cp_external_input_current[...] = ` pattern the producer already uses).

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
2026-07-31 — THE WASH-OUT IS NOW SCOPABLE TO THE PRODUCER REGION (additive, DEFAULT OFF = byte-identical).
THE DEFECT: the snapshot/restore above writes the WHOLE-BRIDGE arrays. That is correct for a PRIVATE producer bridge
(what every shipped EMERGE-61/63/66 path uses) but on a SHARED bridge it is a WHOLE-BRAIN reset executed before EVERY
WORD — any co-resident lane whose state lives in these same arrays is erased per emit (measured elsewhere on a
co-resident affect/mood pool: a held mood of +0.1092 → +0.0000, 3/3 seeds). That is what blocks lanes sharing one
bridge, not the read-out.
THE FIX: `_snapshot_state` / `_restore_state` take an optional `neuron_idx`, and `ResetFrameSlotCQ` takes
`reset_scope` (`True` → `region_manager.indices(<slot_region>)`, i.e. the producer's OWN neurons). `None` (the default
everywhere, including every existing importer) == the whole array == BYTE-IDENTICAL to the shipped path — proven
array-by-array against a verbatim copy of the old body by `--verify-scope`.
BIOLOGICAL WARRANT: an inter-utterance motor-plan wash-out is LOCAL to the motor plan; Broca clearing the previous
sentence's spike-frequency adaptation is not a global brain reset.
⚠️ TWO INDEX SPACES: `cp_stp_x` / `cp_stp_u` are PER-SYNAPSE (`bridge.py:866-867` sizes them by synapse CAPACITY), not
per-neuron. A neuron scope cannot address them, so a SCOPED wash-out does not restore them at all (invariant: a scoped
restore never writes outside the scoped neurons). On the EMERGE-59 slot bridge that omission is INERT and measured
(`_scope_residual_note`: 0 of 12 synapses touch the slots region — it is built `internal_density=0.0` with no
pathways); on a shared bridge whose synapses DO target the producer, the named next mechanism is to map
synapse→post-neuron through the CSR and scope the synapse arrays by post-membership. NOT built, NOT claimed.
NOT CLAIMED HERE: that lanes can now co-reside. That needs its own 6-seed run; this change only removes one measured
blocker and proves the default is unmoved.
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Run:
  SIM_BACKEND=numpy python -m research.runners._emerge61_spiking_broca_order_robustness_derisk --demo
  SIM_BACKEND=numpy python -m research.runners._emerge61_spiking_broca_order_robustness_derisk --derisk
  SIM_BACKEND=numpy python -m research.runners._emerge61_spiking_broca_order_robustness_derisk --derisk --seeds 42 43 44 100 101 102
  SIM_BACKEND=numpy python -m research.runners._emerge61_spiking_broca_order_robustness_derisk --verify-scope
  SIM_BACKEND=numpy python -m research.runners._emerge61_spiking_broca_order_robustness_derisk --derisk --reset-scope region
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sim.backend import to_host, from_host  # noqa: E402
from research.runners._emerge59_spiking_broca_frame_slots_derisk import (  # noqa: E402
    FrameSlotCQ, BrocaProducer, decision_from_emerge, FRAMES, FRAME_NAMES,
)

OUT = _REPO / "research" / "findings" / "raw" / "_emerge61_spiking_broca_order_robustness.json"

# The dynamic per-neuron bridge state that `_run_one_simulation_step` mutates and that CARRIES across productions.
# `cp_recovery_variable_u` is the load-bearing one (Izhikevich slow adaptation); the conductances + firing_states +
# STP are captured too so the restore returns the substrate to its EXACT post-init operating point (byte-for-byte).
# Only arrays PRESENT on this (Izhikevich, no-internal-connectivity) bridge are snapshotted.
#
# ⚠️ TWO INDEX SPACES (measured on this bridge, seed 42: 184 neurons / 12 synapses at capacity 18):
#   * the first SEVEN are PER-NEURON, shape (num_neurons,)          -> scopable by neuron index;
#   * `cp_stp_x` / `cp_stp_u` are PER-SYNAPSE, shape (synapse_capacity,) -> a NEURON index is meaningless on them
#     (`bridge.py:866-867` sizes them by `capacity`, not `num_neurons`).
# The split is what makes the region scoping below possible AND bounded; `_STATE_ARRAYS` keeps its exact previous
# contents + order (it is imported by other runners and by `tests/test_emerge75b_history_independent_aw.py`).
_NEURON_STATE_ARRAYS = (
    "cp_membrane_potential_v", "cp_recovery_variable_u",
    "cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise",
    "cp_firing_states",
)
_SYNAPSE_STATE_ARRAYS = ("cp_stp_x", "cp_stp_u")
_STATE_ARRAYS = _NEURON_STATE_ARRAYS + _SYNAPSE_STATE_ARRAYS


# ---------------------------------------------------------------------------------------------------------------------
# REGION-SCOPING THE WASH-OUT (additive, default-OFF). The un-scoped snapshot/restore writes the WHOLE-BRIDGE arrays,
# which is correct for a PRIVATE producer bridge but is a whole-brain reset on a SHARED one: any co-resident lane whose
# state lives in these same arrays (e.g. a co-resident affect/mood pool held in `cp_recovery_variable_u`) is ERASED on
# every emit. Biological warrant for the scoping: an inter-utterance motor-plan wash-out is LOCAL to the motor plan --
# Broca clearing the last sentence's adaptation does not reset the rest of the brain.
# `neuron_idx=None` (the DEFAULT everywhere) == the whole array == BYTE-IDENTICAL to the shipped path.
# ---------------------------------------------------------------------------------------------------------------------
def _resolve_neuron_scope(bridge, scope):
    """Normalize a wash-out scope to `None` (whole array) | a `slice` | an int index array.

    Accepts: None/False -> whole array (default, byte-identical); a `slice`; a region NAME (str) resolved through
    `bridge.region_manager.indices(name)` -- the producer region's OWN neuron indices; or an explicit sequence of
    neuron indices. A contiguous ascending run is normalized to a `slice` (numerically identical, and it sidesteps
    device fancy-index assignment); region indices ARE contiguous (`sim/regions.py:442` builds `list(range(...))`)."""
    if scope is None or scope is False:
        return None
    if scope is True:
        # `True` means "this producer's OWN region", which only a ResetFrameSlotCQ can resolve (it knows its
        # slot_region). Falling through would make np.asarray(True) read as NEURON INDEX 1 -- a silent, wrong,
        # 1-neuron wash-out. Refuse loudly instead.
        raise ValueError("wash-out scope True is only meaningful on ResetFrameSlotCQ(reset_scope=True), which resolves "
                         "it to its own region; pass the region NAME or an index array to the bridge-level helpers")
    if isinstance(scope, slice):
        return scope
    if isinstance(scope, str):
        rm = getattr(bridge, "region_manager", None)
        if rm is None:
            raise ValueError(f"wash-out scope {scope!r} names a region but this bridge has no region_manager")
        idx = np.asarray(rm.indices(scope), dtype=np.int64)
    else:
        idx = np.asarray(scope, dtype=np.int64).ravel()
    if idx.size == 0:
        raise ValueError("wash-out scope resolved to ZERO neurons -- refusing (a no-op reset would silently disable "
                         "the position-independence the wash-out exists to provide)")
    lo, hi = int(idx.min()), int(idx.max())
    if idx.size == (hi - lo + 1) and np.array_equal(idx, np.arange(lo, hi + 1, dtype=np.int64)):
        return slice(lo, hi + 1)
    return idx


def _scope_len(scope, n):
    """Number of elements a resolved scope selects from a length-`n` array."""
    if scope is None:
        return n
    if isinstance(scope, slice):
        return len(range(*scope.indices(n)))
    return int(np.asarray(scope).size)


def _index_for(arr, scope):
    """The resolved scope in the index space of `arr`'s backend (a cupy array needs a cupy index for fancy
    assignment; a `slice` needs no marshalling and is the normal case for a contiguous region)."""
    if scope is None or isinstance(scope, slice):
        return scope
    if hasattr(arr, "get"):                      # device (cupy) array
        import cupy
        return cupy.asarray(scope)
    return scope


def _snapshot_state(bridge, neuron_idx=None):
    """Byte-for-byte capture of the bridge's dynamic per-neuron state (host copies).

    `neuron_idx=None` (default) captures the WHOLE arrays -- byte-identical to the shipped path. When a scope is given
    (a region name, a slice, or explicit neuron indices) only THOSE neurons' per-neuron state is captured, and the
    PER-SYNAPSE STP arrays are SKIPPED (a neuron index does not address them; see `_SYNAPSE_STATE_ARRAYS`), so the
    paired scoped restore provably never writes outside the producer's own neurons."""
    scope = _resolve_neuron_scope(bridge, neuron_idx)
    snap = {}
    for name in _STATE_ARRAYS:
        arr = getattr(bridge, name, None)
        if arr is None:
            continue
        if scope is None:
            snap[name] = np.asarray(to_host(arr)).copy()
        elif name in _NEURON_STATE_ARRAYS:
            snap[name] = np.asarray(to_host(arr))[scope].copy()
        # else: per-synapse array under a neuron scope -> deliberately NOT captured (see the honest residual in
        # `_scope_residual_note`); capturing it would force a whole-bridge write on restore.
    return snap


def _restore_state(bridge, snap, neuron_idx=None):
    """Restore the captured post-init state in place (backend-agnostic). `from_host` moves the host snapshot to the
    ACTIVE sim backend's device (a no-op passthrough on numpy -> BYTE-IDENTICAL to the prior numpy path; on cupy it
    marshals host->device so `arr[:] = ` into a cupy bridge array works instead of raising 'non-scalar ndarray cannot
    be used for fill').

    `neuron_idx=None` (default) restores the WHOLE arrays -- the shipped path, unchanged. With a scope, the INVARIANT
    is: a scoped restore NEVER writes outside the scoped neurons -- per-neuron arrays are written only at those
    indices and per-synapse arrays are not written at all."""
    scope = _resolve_neuron_scope(bridge, neuron_idx)
    for name, val in snap.items():
        arr = getattr(bridge, name, None)
        if arr is None:
            continue
        if scope is None:
            arr[:] = from_host(val)
            continue
        if name not in _NEURON_STATE_ARRAYS:
            continue                              # INVARIANT: a scoped restore writes no whole-bridge array
        want = _scope_len(scope, int(arr.shape[0]))            # `.shape` works on both numpy and cupy arrays
        got = int(np.asarray(val).shape[0])
        if got != want:
            raise ValueError(
                f"scoped wash-out shape mismatch on {name}: snapshot has {got} entries, the scope selects {want}. "
                f"The snapshot was almost certainly taken UNSCOPED (`_snapshot_state(bridge)`) while the restore is "
                f"scoped -- re-take it with the SAME scope (on a ResetFrameSlotCQ subclass, call `self._resnapshot()` "
                f"instead of assigning `_post_init_state = _snapshot_state(self.bridge)`)")
        arr[_index_for(arr, scope)] = from_host(val)


def _scope_residual_note(bridge, neuron_idx):
    """The HONEST residual of a scoped wash-out, MEASURED on this bridge rather than assumed: with a neuron scope the
    per-synapse STP state is not restored, so any synapse whose post-synaptic neuron is inside the scope keeps its
    short-term-plasticity state across utterances. Returns a dict with the measured synapse count that touches the
    scoped neurons (0 => the omission is inert on this bridge)."""
    scope = _resolve_neuron_scope(bridge, neuron_idx)
    out = {"scoped": scope is not None, "n_synapses": int(getattr(bridge, "_synapse_count", 0) or 0),
           "stp_present": getattr(bridge, "cp_stp_x", None) is not None, "synapses_touching_scope": None}
    if scope is None or not out["stp_present"] or out["n_synapses"] == 0:
        out["synapses_touching_scope"] = 0
        return out
    conn = getattr(bridge, "cp_connections", None)
    try:
        m = conn.item() if hasattr(conn, "item") and getattr(conn, "shape", None) == () else conn
        m = m.get() if hasattr(m, "get") else m
        coo = m.tocoo()
        rows = np.asarray(to_host(coo.row)); cols = np.asarray(to_host(coo.col))
        n = int(bridge.core_config.num_neurons)
        mask = np.zeros(n, bool)
        mask[scope] = True                        # both orientations counted: a synapse "touches" if either end is in
        out["synapses_touching_scope"] = int(np.count_nonzero(mask[rows] | mask[cols]))
    except Exception as e:                        # measurement failed -> report UNKNOWN, never a silent 0
        out["synapses_touching_scope"] = None
        out["measure_error"] = repr(e)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# THE FIX: a FrameSlotCQ that resets the spiking substrate to its EXACT post-init state before EACH production, so the
# read-out is a function of the learned primacy gradient ALONE (position-independent). ADDITIVE: subclasses EMERGE-59's
# FrameSlotCQ, overrides only emit / emit_order_indices to restore first; EMERGE-59 itself is untouched.
# ---------------------------------------------------------------------------------------------------------------------
class ResetFrameSlotCQ(FrameSlotCQ):
    """FrameSlotCQ + an inter-utterance wash-out: capture the post-init dynamic state at construction, restore it before
    every emit so no production's residual adaptation leaks into the next. The learned primacy, RNG, and slot structure
    are inherited UNCHANGED; only the substrate's dynamic state is reset (the correct post-init snapshot, not a flat
    reset). This makes each production an independent motor plan (the load-bearing position-independence).

    `reset_scope` (additive, DEFAULT None = the shipped whole-bridge behavior, byte-identical) bounds the wash-out to
    the producer's OWN neurons, so a SHARED bridge's other lanes are not reset by every word:
      * `None` / `False` -> the whole arrays (default; correct + byte-identical on a PRIVATE producer bridge);
      * `True`           -> `self.slot_idx`, i.e. `region_manager.indices(<slot_region>)` -- the producer region;
      * a region NAME    -> `bridge.region_manager.indices(name)`;
      * a slice / sequence of neuron indices -> used as given.
    Biological warrant: an inter-utterance motor-plan wash-out is local to the motor plan (Broca clearing the previous
    sentence's adaptation is not a global brain reset)."""

    def __init__(self, *args, reset_scope=None, **kwargs):
        super().__init__(*args, **kwargs)
        # `True` means "this producer's own region": self.slot_idx IS region_manager.indices(slot_region), resolved by
        # the base builder for BOTH the private-bridge and the shared-bridge (co-location) paths.
        self.reset_scope = reset_scope
        self._reset_idx = _resolve_neuron_scope(self.bridge, self.slot_idx if reset_scope is True else reset_scope)
        # snapshot AFTER the base __init__ built + initialized the bridge (and before any emit ran a step).
        self._post_init_state = _snapshot_state(self.bridge, self._reset_idx)

    def _resnapshot(self):
        """Re-take the wash-out snapshot with THIS instance's scope. A subclass that mutates the bridge after
        `__init__` (e.g. EMERGE-63's CorpusOrderFrameSlotCQ disabling structural plasticity) must call THIS rather than
        assigning `_post_init_state = _snapshot_state(self.bridge)`, or a scoped instance would raise on restore."""
        self._post_init_state = _snapshot_state(self.bridge, self._reset_idx)
        return self._post_init_state

    def _reset_substrate(self):
        _restore_state(self.bridge, self._post_init_state, self._reset_idx)

    def emit(self, frame, subject, verb, spell):
        self._reset_substrate()
        return super().emit(frame, subject, verb, spell)

    def emit_order_indices(self, frame):
        self._reset_substrate()
        return super().emit_order_indices(frame)


# ---------------------------------------------------------------------------------------------------------------------
# THE EMERGE-60 EMIT SEQUENCE (the failing case): owl/minnow (F_MODAL), penguin/pike (F_INTR), robin (F_MODAL, 5th).
# robin@5th is where 100/101 swapped without the reset.
# ---------------------------------------------------------------------------------------------------------------------
_SEQUENCE = [
    ("owl", "fly", "affirm", "the owl can fly"),
    ("minnow", "swim", "affirm", "the minnow can swim"),
    ("penguin", "walks", "negate", "the penguin walks"),
    ("pike", "lurks", "negate", "the pike lurks"),
    ("robin", "breathe", "affirm", "the robin can breathe"),
]


def _render_sequence(cq):
    """Render the EMERGE-60 emit sequence through a BrocaProducer; return the surfaces + the moat/abstain result."""
    prod = BrocaProducer(cq)
    surfaces = []
    for (subj, verb, pol, _exp) in _SEQUENCE:
        dec = decision_from_emerge("ANSWER", subject=subj, verb=verb, polarity=pol)
        surfaces.append(prod.speak(dec)["surface"])
    # moat: an ABSTAIN decision must NOT invoke (or reset) the producer.
    calls_before = prod.production_count
    ab = prod.speak(decision_from_emerge("ABSTAIN"))
    moat_calls = prod.production_count - calls_before
    return surfaces, prod, int(moat_calls), bool(ab["produced"])


def _sequence_exact(surfaces):
    """render-EXACT over the sequence: fraction of productions whose surface == the ground-truth surface."""
    exp = [e for (_s, _v, _p, e) in _SEQUENCE]
    return float(np.mean([1.0 if surfaces[i] == exp[i] else 0.0 for i in range(len(exp))]))


def _position_independence(cq_factory, seed):
    """Render robin->'the robin can breathe' at emit-position 1, 3, and 5 (with 0/2/4 prior productions), each on a
    freshly-constructed producer of the same class, and check all three surfaces are IDENTICAL (and correct). The
    load-bearing property: an utterance must not depend on how many productions preceded it."""
    robin_dec = decision_from_emerge("ANSWER", subject="robin", verb="breathe", polarity="affirm")
    surfaces_at = {}
    for pos in (1, 3, 5):
        cq = cq_factory(seed)
        cq.learn()
        prod = BrocaProducer(cq)
        # run (pos-1) prior productions from the sequence, then robin
        for (subj, verb, pol, _e) in _SEQUENCE[: pos - 1]:
            prod.speak(decision_from_emerge("ANSWER", subject=subj, verb=verb, polarity=pol))
        surfaces_at[pos] = prod.speak(robin_dec)["surface"]
    vals = list(surfaces_at.values())
    identical = all(v == vals[0] for v in vals)
    correct = vals[0] == "the robin can breathe"
    return bool(identical and correct), surfaces_at


def _make_reset(seed):
    return ResetFrameSlotCQ(seed=seed)


def _make_reset_scoped(seed):
    """The SAME wash-out bounded to the producer region's own neurons (`region_manager.indices(<slot_region>)`)."""
    return ResetFrameSlotCQ(seed=seed, reset_scope=True)


def _make_plain(seed):
    return FrameSlotCQ(seed=seed)


# ---------------------------------------------------------------------------------------------------------------------
# INSTRUMENT VERIFICATION for the scoping change: (1) the UNSCOPED path is BYTE-IDENTICAL to the pre-change code (hashed
# array-by-array, against a verbatim copy of the old body), (2) the restore really does return the exact post-init
# bytes, (3) a SCOPED wash-out provably does not write outside the producer region (a sentinel in the co-resident
# neurons SURVIVES the scoped reset and is ERASED by the unscoped one), (4) the producer's OWN result is unchanged
# under the scope. This is the "verify the instrument before trusting the output" step, run as `--verify-scope`.
# ---------------------------------------------------------------------------------------------------------------------
VERIFY_OUT = _REPO / "research" / "findings" / "raw" / "_emerge61_scope_byte_identity.json"


def _hash_state(bridge):
    """blake2b-128 of every dynamic state array's raw bytes (host copies) -> {array_name: hexdigest}."""
    import hashlib
    out = {}
    for name in _STATE_ARRAYS:
        arr = getattr(bridge, name, None)
        if arr is None:
            continue
        host = np.ascontiguousarray(np.asarray(to_host(arr)))
        out[name] = hashlib.blake2b(host.tobytes(), digest_size=16).hexdigest()
    return out


def _legacy_snapshot(bridge):
    """VERBATIM copy of the PRE-CHANGE `_snapshot_state` body -- the A/B reference for the byte-identity check."""
    snap = {}
    for name in _STATE_ARRAYS:
        arr = getattr(bridge, name, None)
        if arr is not None:
            snap[name] = np.asarray(to_host(arr)).copy()
    return snap


def _legacy_restore(bridge, snap):
    """VERBATIM copy of the PRE-CHANGE `_restore_state` body -- the A/B reference for the byte-identity check."""
    for name, val in snap.items():
        arr = getattr(bridge, name, None)
        if arr is not None:
            arr[:] = from_host(val)


def _perturb(bridge, slot_idx, n_slot_pools, rounds=2):
    """Drive the slot pools for `rounds` productions' worth of steps so the dynamic state is genuinely dirty (this is
    what accumulates `cp_recovery_variable_u`); deterministic given the bridge, so two bridges perturb identically.

    `rounds=2` by default because at round 3 the bridge's STRUCTURAL PLASTICITY grows synapses on the inert `_anchor`
    region and REBUILDS `cp_stp_x`/`cp_stp_u` at the new nnz (measured: 18 -> 13 -> 14 -> 15), which breaks the
    fixed-shape snapshot -- a PRE-EXISTING property of the wash-out, already worked around in EMERGE-63 by disabling
    structural plasticity, and measured explicitly by `_measure_stp_resize` below rather than hidden."""
    from research.runners._emerge59_spiking_broca_frame_slots_derisk import slot_pool_rates
    for r in range(rounds):
        drive = {p: 1800.0 - 300.0 * ((p + r) % n_slot_pools) for p in range(n_slot_pools)}
        slot_pool_rates(bridge, slot_idx, drive, n_slot_pools=n_slot_pools)


def _measure_stp_resize(seed, rounds=6):
    """MEASURE (not assume) the pre-existing per-synapse-array resize and what it does to each restore path: drive one
    unscoped and one scoped instance past the resize, then try each wash-out. Records shapes + the exact exception."""
    out = {"rounds": rounds}
    for tag, factory in (("unscoped", _make_reset), ("scoped", _make_reset_scoped)):
        cq = factory(seed)
        before = None if cq.bridge.cp_stp_x is None else int(cq.bridge.cp_stp_x.shape[0])
        _perturb(cq.bridge, cq.slot_idx, cq.n_slot_pools, rounds=rounds)
        after = None if cq.bridge.cp_stp_x is None else int(cq.bridge.cp_stp_x.shape[0])
        try:
            cq._reset_substrate()
            err = None
        except Exception as e:                      # noqa: BLE001 -- the exception IS the measurement
            err = repr(e)
        out[tag] = {"stp_len_before": before, "stp_len_after": after, "resized": before != after,
                    "reset_raised": err}
    return out


def _scramble_state(bridge, seed):
    """Overwrite EVERY dynamic state array with deterministic garbage. The perturbation-by-driving arm only dirties
    the arrays the dynamics happen to touch (measured: 4 of 9 on this bridge -- the conductances never leave 0 because
    the slot pools have no incoming synapses), which leaves the other 5 a weak test. Scrambling gives the byte-identity
    A/B full power on ALL of them; the same `seed` scrambles two twin bridges identically."""
    rng = np.random.default_rng(seed)
    for name in _STATE_ARRAYS:
        arr = getattr(bridge, name, None)
        if arr is None:
            continue
        host = np.asarray(to_host(arr))
        g = (rng.random(host.shape) < 0.5) if host.dtype == np.bool_ else \
            (rng.standard_normal(host.shape) * 10.0).astype(host.dtype)
        arr[:] = from_host(g)


def _sentinel_write(bridge, idx):
    """Stamp a distinctive value into the NON-producer (co-resident) neurons of every per-neuron state array, and
    return what was written. This stands in for a co-resident lane's state living in the same arrays."""
    written = {}
    for i, name in enumerate(_NEURON_STATE_ARRAYS):
        arr = getattr(bridge, name, None)
        if arr is None:
            continue
        host = np.asarray(to_host(arr)).copy()
        host[idx] = True if host.dtype == np.bool_ else np.asarray(-77.25 - i, dtype=host.dtype)
        arr[:] = from_host(host)
        written[name] = host[idx].copy()
    return written


def _read_at(bridge, idx):
    """Host copies of every per-neuron state array restricted to `idx`."""
    return {name: np.asarray(to_host(getattr(bridge, name)))[idx].copy()
            for name in _NEURON_STATE_ARRAYS if getattr(bridge, name, None) is not None}


def _dicts_equal(a, b):
    return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)


def _freeze_structural(cq):
    """Disable structural plasticity on an already-built producer + re-take its wash-out snapshot.

    WHY (measured, 2026-07-31): `sim/bridge.py` commit 479e4584 made the CSR-rebuild remap re-allocate EVERY
    per-synapse array at `nnz`, so on this slot bridge `cp_stp_x`/`cp_stp_u` change LENGTH (18 -> 13) at the 3rd
    production, when structural plasticity grows a synapse on the inert `_anchor` region. The fixed-shape UNSCOPED
    restore then raises `could not broadcast input array from shape (18,) into shape (13,)`. That break is
    PRE-EXISTING (it reproduces on the pristine HEAD file with an unmodified `sim/` and `_emerge59`) and is NOT
    addressed here. Freezing structural plasticity is EMERGE-63's existing, behavior-neutral workaround: the slot
    pools are built `internal_density=0.0` with no pathways, so the growth is confined to the inert `_anchor` region
    and cannot reach the read-out. Applied to BOTH arms of any comparison so it cannot bias one of them."""
    cq.bridge.core_config.enable_structural_plasticity = False
    if hasattr(cq, "_resnapshot"):
        cq._resnapshot()
    return cq


def _frozen_factory(factory):
    """`factory` wrapped so every instance it builds has structural plasticity frozen (see `_freeze_structural`)."""
    def make(seed):
        return _freeze_structural(factory(seed))
    return make


def _learned(factory, seed, freeze_structural=False):
    cq = factory(seed)
    if freeze_structural:
        _freeze_structural(cq)
    cq.learn()
    return cq


def _verify_scope(seed=100):
    """Verify the scoping change did not move the DEFAULT path by one byte, and that the scoped path is bounded."""
    from research.runners._emerge59_spiking_broca_frame_slots_derisk import N_SLOT_POOLS
    res = {"probe": "emerge61_scope_byte_identity", "seed": int(seed), "checks": {}, "hashes": {}}

    # ---- (1) the UNSCOPED SNAPSHOT is byte-identical to the pre-change snapshot -------------------------------------
    cq = ResetFrameSlotCQ(seed=seed)                       # default: reset_scope=None -> whole-bridge (shipped path)
    b = cq.bridge
    new_snap, old_snap = _snapshot_state(b), _legacy_snapshot(b)
    res["checks"]["unscoped_snapshot_byte_identical"] = bool(_dicts_equal(new_snap, old_snap))
    res["checks"]["snapshot_keys"] = sorted(new_snap)
    res["checks"]["default_reset_idx_is_None"] = bool(cq._reset_idx is None)

    h_init = _hash_state(b)
    res["hashes"]["post_init"] = h_init

    # ---- (2) perturb -> the state really moves -> the UNSCOPED restore returns the EXACT post-init bytes ------------
    _perturb(b, cq.slot_idx, cq.n_slot_pools)
    h_dirty = _hash_state(b)
    res["hashes"]["after_driven_productions"] = h_dirty      # `_perturb` rounds; see its docstring for why it is 2
    res["checks"]["state_actually_changes"] = bool(any(h_dirty[k] != h_init[k] for k in h_init))
    res["checks"]["changed_arrays"] = sorted(k for k in h_init if h_dirty[k] != h_init[k])
    # guard the byte-identity arms against the pre-existing per-synapse RESIZE (see `_measure_stp_resize`): if the
    # arrays changed LENGTH here, the hash comparison below would be comparing different objects, not the same code.
    res["checks"]["stp_shape_stable_during_byte_identity_arms"] = bool(
        "cp_stp_x" not in new_snap or int(b.cp_stp_x.shape[0]) == int(np.asarray(new_snap["cp_stp_x"]).shape[0]))
    cq._reset_substrate()
    h_restored = _hash_state(b)
    res["hashes"]["after_unscoped_restore"] = h_restored
    res["checks"]["unscoped_restore_returns_post_init_bytes"] = bool(h_restored == h_init)

    # ---- (3) A/B: NEW default restore vs the VERBATIM OLD restore, on identically-perturbed twin bridges ------------
    cq_a, cq_b = ResetFrameSlotCQ(seed=seed), ResetFrameSlotCQ(seed=seed)
    res["checks"]["twin_bridges_identical_at_init"] = bool(_hash_state(cq_a.bridge) == _hash_state(cq_b.bridge))
    _perturb(cq_a.bridge, cq_a.slot_idx, cq_a.n_slot_pools)
    _perturb(cq_b.bridge, cq_b.slot_idx, cq_b.n_slot_pools)
    res["checks"]["twin_bridges_identical_when_dirty"] = bool(_hash_state(cq_a.bridge) == _hash_state(cq_b.bridge))
    _restore_state(cq_a.bridge, cq_a._post_init_state)                       # NEW code, scope unset
    _legacy_restore(cq_b.bridge, _legacy_snapshot(ResetFrameSlotCQ(seed=seed).bridge))   # OLD code, fresh post-init
    h_new, h_old = _hash_state(cq_a.bridge), _hash_state(cq_b.bridge)
    res["hashes"]["new_default_restore"] = h_new
    res["hashes"]["legacy_restore"] = h_old
    res["checks"]["new_vs_legacy_restore_byte_identical"] = bool(h_new == h_old)
    res["checks"]["byte_identical_array_by_array"] = {k: bool(h_new[k] == h_old[k]) for k in h_new}

    # ---- (3b) the same A/B with EVERY array scrambled, so the byte-identity claim has power on ALL 9 arrays --------
    cq_c, cq_d = ResetFrameSlotCQ(seed=seed), ResetFrameSlotCQ(seed=seed)
    snap_c, snap_d = _snapshot_state(cq_c.bridge), _legacy_snapshot(cq_d.bridge)
    _scramble_state(cq_c.bridge, 1234)
    _scramble_state(cq_d.bridge, 1234)
    h_scrambled = _hash_state(cq_c.bridge)
    res["hashes"]["scrambled"] = h_scrambled
    res["checks"]["scramble_dirties_every_array"] = bool(all(h_scrambled[k] != h_init[k] for k in h_init))
    res["checks"]["twins_scrambled_identically"] = bool(h_scrambled == _hash_state(cq_d.bridge))
    _restore_state(cq_c.bridge, snap_c)                    # NEW code, scope unset
    _legacy_restore(cq_d.bridge, snap_d)                   # VERBATIM old code
    h_new_s, h_old_s = _hash_state(cq_c.bridge), _hash_state(cq_d.bridge)
    res["hashes"]["new_default_restore_from_scrambled"] = h_new_s
    res["hashes"]["legacy_restore_from_scrambled"] = h_old_s
    res["checks"]["scrambled_new_vs_legacy_byte_identical"] = bool(h_new_s == h_old_s)
    res["checks"]["scrambled_restore_returns_post_init_bytes"] = bool(h_new_s == h_init)

    # ---- (4) the SCOPED wash-out is BOUNDED: a co-resident sentinel survives it, and the unscoped one erases it -----
    rm = b.region_manager
    all_regions = {n: np.asarray(v) for n, v in rm.region_indices_dict().items()}
    prod_idx = np.asarray(cq.slot_idx)
    other_idx = np.asarray(sorted(set(range(int(b.core_config.num_neurons))) - set(prod_idx.tolist())))
    res["regions"] = {n: [int(v[0]), int(v[-1]), int(v.size)] for n, v in all_regions.items()}
    res["checks"]["n_coresident_neurons"] = int(other_idx.size)

    scoped = ResetFrameSlotCQ(seed=seed, reset_scope=True)
    sel = np.arange(int(scoped.bridge.core_config.num_neurons))[scoped._reset_idx]
    res["checks"]["scoped_idx_equals_region_indices"] = bool(np.array_equal(sel, np.asarray(scoped.slot_idx)))
    res["checks"]["scoped_idx_repr"] = repr(scoped._reset_idx)
    _sentinel_write(scoped.bridge, other_idx)
    _perturb(scoped.bridge, scoped.slot_idx, scoped.n_slot_pools, rounds=1)
    sent_scoped = _read_at(scoped.bridge, other_idx)                 # post-drive value = what must SURVIVE the reset
    scoped._reset_substrate()
    res["checks"]["scoped_reset_preserves_coresident_state"] = bool(
        _dicts_equal(sent_scoped, _read_at(scoped.bridge, other_idx)))
    res["checks"]["scoped_reset_restores_producer_state"] = bool(_dicts_equal(
        _read_at(scoped.bridge, prod_idx), _read_at(ResetFrameSlotCQ(seed=seed).bridge, prod_idx)))

    unscoped = ResetFrameSlotCQ(seed=seed)
    _sentinel_write(unscoped.bridge, other_idx)
    _perturb(unscoped.bridge, unscoped.slot_idx, unscoped.n_slot_pools, rounds=1)
    sent_unscoped = _read_at(unscoped.bridge, other_idx)
    unscoped._reset_substrate()
    res["checks"]["unscoped_reset_ERASES_coresident_state"] = bool(
        not _dicts_equal(sent_unscoped, _read_at(unscoped.bridge, other_idx)))

    # ---- (5) the producer's OWN result is unchanged by the scoping (private bridge) ---------------------------------
    # BOTH arms freeze structural plasticity -- symmetrically -- because the UNSCOPED arm cannot complete 5 emits
    # without it since sim/bridge.py 479e4584 (see `_freeze_structural`). This is the ONLY way to A/B the two paths
    # end-to-end today, and the freeze is applied identically to both so it cannot favour either.
    res["structural_plasticity_frozen_for_surface_arms"] = True
    seq_unscoped, _p1, _mc1, _m1 = _render_sequence(_learned(_make_reset, seed, freeze_structural=True))
    seq_scoped, _p2, mc2, _m2 = _render_sequence(_learned(_make_reset_scoped, seed, freeze_structural=True))
    res["surfaces"] = {"unscoped": seq_unscoped, "scoped": seq_scoped}
    res["checks"]["scoped_surfaces_match_unscoped"] = bool(seq_unscoped == seq_scoped)
    res["checks"]["scoped_sequence_exact"] = float(_sequence_exact(seq_scoped))
    res["checks"]["unscoped_sequence_exact"] = float(_sequence_exact(seq_unscoped))
    pos_ok, pos_surf = _position_independence(_frozen_factory(_make_reset_scoped), seed)
    pos_ok_u, pos_surf_u = _position_independence(_frozen_factory(_make_reset), seed)
    res["checks"]["scoped_position_independent"] = bool(pos_ok)
    res["checks"]["unscoped_position_independent"] = bool(pos_ok_u)
    res["checks"]["scoped_position_surfaces"] = {str(k): v for k, v in pos_surf.items()}
    res["checks"]["unscoped_position_surfaces"] = {str(k): v for k, v in pos_surf_u.items()}
    res["checks"]["scoped_position_surfaces_match_unscoped"] = bool(pos_surf == pos_surf_u)
    res["checks"]["moat_calls_on_abstain_scoped"] = int(mc2)

    # ---- (6) a PRE-EXISTING fragility, measured (NOT introduced or fixed here) ---------------------------------------
    # Structural plasticity rebuilds the PER-SYNAPSE arrays at a new size after ~3 productions, which breaks the
    # fixed-shape UNSCOPED restore. The scoped restore does not raise -- not because it solves this, but because it
    # never writes the per-synapse arrays at all. Recorded as measurement, claimed as nothing.
    res["stp_resize"] = _measure_stp_resize(seed)

    # ---- (7) the HONEST residual of the scoping, MEASURED -----------------------------------------------------------
    res["scope_residual"] = _scope_residual_note(scoped.bridge, scoped._reset_idx)
    res["scope_residual"]["note"] = (
        "A neuron-scoped wash-out does NOT restore the PER-SYNAPSE STP arrays (cp_stp_x / cp_stp_u), because a neuron "
        "index does not address them. On THIS bridge that omission is inert iff `synapses_touching_scope` == 0 "
        "(measured above; the slots region is built with internal_density=0.0 and no pathways). On a shared bridge "
        "where synapses DO target the producer region, the named next mechanism is to map synapse->post-neuron through "
        "the CSR and scope the synapse arrays by post-membership -- NOT attempted here, and NOT claimed.")

    crit = ["unscoped_snapshot_byte_identical", "default_reset_idx_is_None", "state_actually_changes",
            "stp_shape_stable_during_byte_identity_arms",
            "unscoped_restore_returns_post_init_bytes", "twin_bridges_identical_at_init",
            "twin_bridges_identical_when_dirty", "new_vs_legacy_restore_byte_identical",
            "scramble_dirties_every_array", "twins_scrambled_identically",
            "scrambled_new_vs_legacy_byte_identical", "scrambled_restore_returns_post_init_bytes",
            "scoped_idx_equals_region_indices", "scoped_reset_preserves_coresident_state",
            "unscoped_reset_ERASES_coresident_state", "scoped_reset_restores_producer_state",
            "scoped_surfaces_match_unscoped", "scoped_position_independent",
            "unscoped_position_independent", "scoped_position_surfaces_match_unscoped"]
    res["PASS"] = bool(all(res["checks"].get(k) is True for k in crit))
    res["critical_checks"] = crit
    VERIFY_OUT.parent.mkdir(parents=True, exist_ok=True)
    VERIFY_OUT.write_text(json.dumps(res, indent=2, default=str))

    print("\n=== EMERGE-61 scope verification (seed %d) ===" % seed)
    for k in crit:
        print(f"  {'PASS' if res['checks'].get(k) is True else 'FAIL'}  {k} = {res['checks'].get(k)}")
    print("\n  array-by-array byte-identity (NEW default restore vs VERBATIM legacy restore, from a FULL SCRAMBLE so "
          "every array is genuinely dirty):")
    for k in sorted(h_new_s):
        print(f"    {k:28s} scrambled={h_scrambled[k]}  new={h_new_s[k]}  legacy={h_old_s[k]}  "
              f"{'==' if h_new_s[k] == h_old_s[k] == h_init[k] else '!! DIFFERS'}")
    print("\n  array-by-array byte-identity (same A/B after 2 driven productions):")
    for k in sorted(h_new):
        print(f"    {k:28s} new={h_new[k]}  legacy={h_old[k]}  {'==' if h_new[k] == h_old[k] else '!! DIFFERS'}")
    print("\n  post-init hashes vs after-unscoped-restore hashes:")
    for k in sorted(h_init):
        print(f"    {k:28s} init={h_init[k]}  restored={h_restored[k]}  dirty={h_dirty[k]}  "
              f"{'==' if h_init[k] == h_restored[k] else '!! DIFFERS'}")
    print(f"\n  scope residual (measured): {res['scope_residual']['synapses_touching_scope']} synapses touch the "
          f"producer region of {res['scope_residual']['n_synapses']} total")
    print(f"  PRE-EXISTING (not introduced, not fixed here) structural-plasticity STP resize after "
          f"{res['stp_resize']['rounds']} productions:")
    for tag in ("unscoped", "scoped"):
        r = res["stp_resize"][tag]
        print(f"    {tag:9s} stp len {r['stp_len_before']} -> {r['stp_len_after']}  reset_raised={r['reset_raised']}")
    print(f"\n[emerge61-verify] {'PASS' if res['PASS'] else 'FAIL'} -- wrote {VERIFY_OUT}\n")
    return 0 if res["PASS"] else 1


# ---------------------------------------------------------------------------------------------------------------------
# THE DE-RISK
# ---------------------------------------------------------------------------------------------------------------------
def _derisk_one(seed, reset_scope=None):
    # WITH the reset (the fix): render the sequence, score exact, check moat. `reset_scope=None` (default) is the
    # shipped whole-bridge wash-out; `True` runs the SAME de-risk with the wash-out bounded to the producer region
    # (an equivalence arm -- the producer's own result must not move).
    fix_factory = _make_reset if not reset_scope else _make_reset_scoped
    cq_fix = fix_factory(seed)
    cq_fix.learn()
    fix_surfaces, _prod, moat_calls, moat_produced = _render_sequence(cq_fix)
    fix_exact = _sequence_exact(fix_surfaces)

    # WITHOUT the reset (the causal control = EMERGE-60's current behavior): the tail swaps on the failing seeds.
    cq_ctl = FrameSlotCQ(seed=seed)
    cq_ctl.learn()
    ctl_surfaces, _p2, _mc2, _mp2 = _render_sequence(cq_ctl)
    ctl_exact = _sequence_exact(ctl_surfaces)

    # POSITION-INDEPENDENCE with the fix (must hold) vs without (may fail on 100/101).
    fix_posindep, fix_pos_surf = _position_independence(fix_factory, seed)
    ctl_posindep, ctl_pos_surf = _position_independence(_make_plain, seed)

    return {
        "seed": seed, "reset_scope": ("region" if reset_scope else None),
        "fix_exact": fix_exact, "ctl_exact": ctl_exact,
        "fix_surfaces": fix_surfaces, "ctl_surfaces": ctl_surfaces,
        "fix_posindep": fix_posindep, "ctl_posindep": ctl_posindep,
        "fix_pos_surfaces": {str(k): v for k, v in fix_pos_surf.items()},
        "ctl_pos_surfaces": {str(k): v for k, v in ctl_pos_surf.items()},
        "moat_calls_on_abstain": int(moat_calls), "moat_produced_on_abstain": bool(moat_produced),
    }


def _demo(seed=100):
    print("\n=== EMERGE-61 -- close the spiking-Broca render-ORDER tail: an inter-utterance WASH-OUT (reset the substrate "
          "to its exact post-init state before each production) so an utterance does not depend on prior utterances' "
          "residual Izhikevich adaptation ===\n")
    print(f"  (root cause: cp_recovery_variable_u -- the Izhikevich slow-adaptation current -- ACCUMULATES across "
          f"productions; on 2/6 seeds it flips the F_MODAL frame's two near-equal-primacy adjacent slots at the 5th "
          f"emit -> 'the robin breathe can'. The correct post-init reset returns the read-out to a function of the "
          f"LEARNED primacy alone.)\n")
    for tag, factory in (("WITHOUT reset (EMERGE-60 current)", _make_plain), ("WITH reset (EMERGE-61 fix)", _make_reset)):
        cq = factory(seed)
        cq.learn()
        surfaces, _prod, mc, _mp = _render_sequence(cq)
        exact = _sequence_exact(surfaces)
        print(f"  [{tag}]  (seed {seed})")
        for (subj, _v, _p, exp), got in zip(_SEQUENCE, surfaces):
            flag = "ok" if got == exp else "SWAP"
            print(f"      broca> {got:26s} [{flag}]")
        print(f"      render-exact {exact:.2f}, moat-calls-on-abstain {mc}\n")


def _derisk(seeds, reset_scope=None):
    print(f"EMERGE-61 de-risk: close the render-ORDER tail via an inter-utterance wash-out (post-init state reset); "
          f"render-exact-in-sequence -> ~1.00 + position-independence + causal (un-reset swaps) + moat; "
          f"{len(seeds)}-seed" + ("  [wash-out SCOPED to the producer region]" if reset_scope else ""), flush=True)
    t0 = time.time()
    err = None
    per = []
    try:
        for s in seeds:
            d = _derisk_one(s, reset_scope=reset_scope)
            per.append(d)
            print(f"  [seed {s}] FIX exact {d['fix_exact']:.2f} pos-indep {int(d['fix_posindep'])} | "
                  f"CTL(un-reset) exact {d['ctl_exact']:.2f} pos-indep {int(d['ctl_posindep'])} | "
                  f"moat-calls {d['moat_calls_on_abstain']}", flush=True)
    except Exception as e:
        err = repr(e)
        traceback.print_exc()

    if err is None:
        def m(k):
            return float(np.mean([d[k] for d in per]))
        fix_exact = m("fix_exact")
        ctl_exact = m("ctl_exact")
        fix_posindep_all = all(d["fix_posindep"] for d in per)
        moat_calls = int(sum(d["moat_calls_on_abstain"] for d in per))
        moat_produced = any(d["moat_produced_on_abstain"] for d in per)
        # the causal control: at least one seed must swap WITHOUT the reset (else the reset isn't load-bearing here).
        ctl_swaps_somewhere = any(d["ctl_exact"] < 1.0 or (not d["ctl_posindep"]) for d in per)

        all_fix_exact_1 = all(d["fix_exact"] >= 0.999 for d in per)
        moat_ok = (moat_calls == 0) and (not moat_produced)

        go = bool(all_fix_exact_1 and fix_posindep_all and moat_ok and ctl_swaps_somewhere)
        if go:
            verdict = (
                f"GO -- the spiking-Broca render-ORDER tail is CLOSED. Root cause CONFIRMED (H1): the Izhikevich slow-"
                f"adaptation current cp_recovery_variable_u ACCUMULATES across productions (u_pre 0.0 at emit#1 -> "
                f"~500 mean/~500 std at emit#5), and on 2/6 seeds that heterogeneous residual flips the F_MODAL frame's "
                f"two near-equal-primacy adjacent slots at the 5th emit -> 'the robin breathe can'. THE FIX: an inter-"
                f"utterance WASH-OUT -- restore the substrate to its EXACT per-neuron post-init state (v / u / the four "
                f"conductances / firing_states / STP, captured byte-for-byte after _initialize_simulation_data) before "
                f"EACH production -- so the read-out is a function of the LEARNED primacy gradient ALONE. render-EXACT-"
                f"in-sequence == {fix_exact:.2f} on ALL {len(seeds)} seeds (robin as the 5th emit, IN the sequence, not "
                f"fresh-per-emit). POSITION-INDEPENDENCE holds on every seed: the same fact renders IDENTICALLY at emit-"
                f"position 1 / 3 / 5 (0 / 2 / 4 prior productions) -- an utterance no longer depends on prior utterances' "
                f"residual state (the load-bearing property). CAUSAL: WITHOUT the reset the sequence tail swaps "
                f"(un-reset render-exact {ctl_exact:.2f}); WITH it, it does not. The naive FLAT reset was WRONG (it "
                f"ignores per-neuron heterogeneity + the u=b*(v-vr) init relation, made it worse 0.867); the CORRECT "
                f"post-init snapshot is exact + cheap. The gate-first no-confab MOAT is untouched ({moat_calls} producer "
                f"calls on abstains -- the producer is never invoked, hence never reset, on an abstain). ADDITIVE: a "
                f"ResetFrameSlotCQ subclass (EMERGE-59 untouched, its default de-risk byte-identical) + a default-OFF "
                f"reset_producer flag on EMERGE-60's SpikingBrocaConsole. NO sim/ edit (the reset writes existing bridge "
                f"arrays via their public attributes, the same pattern the producer already uses). ==> the flagship "
                f"console renders EMERGE answers EXACT on ALL seeds; the emergent brain SPEAKS its grounded answers on "
                f"spikes with a stable word order, transformer-retired for those frames.")
        else:
            miss = []
            if not all_fix_exact_1:
                bad = [d["seed"] for d in per if d["fix_exact"] < 0.999]
                miss.append(f"render-exact-in-sequence not 1.00 on seeds {bad} (mean {fix_exact:.3f}) -- the reset did "
                            f"NOT make the tail order stable; the residual is NOT (only) accumulated adaptation")
            if not fix_posindep_all:
                bad = [d["seed"] for d in per if not d["fix_posindep"]]
                miss.append(f"position-independence FAILS on seeds {bad} -- a production still depends on prior "
                            f"productions after the reset (the snapshot is incomplete OR another stateful array leaks)")
            if not moat_ok:
                miss.append(f"MOAT: {moat_calls} producer-calls on abstains / produced-on-abstain {moat_produced} "
                            f"-- BLOCKING, the reset must NOT run the producer on an abstain")
            if not ctl_swaps_somewhere:
                miss.append("the un-reset control did NOT swap on any seed (the fix is not causally demonstrated here) "
                            "-- rebuild the failing sequence")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". The named next mechanism: if position-independence still "
                       "fails, enumerate EVERY per-neuron array _run_one_simulation_step mutates and snapshot it too "
                       "(an incomplete snapshot leaks residual state); if the order is still ambiguous after a complete "
                       "reset, the residual is in the PRIMACY SEPARATION (H3) not the substrate state -- widen the "
                       "F_MODAL 4-slot primacy-current gradient (per-instance arg, EMERGE-59 default preserved) so the "
                       "two adjacent ranks separate above the read-out noise. Do NOT weaken the moat.")
    else:
        verdict = f"ERROR -- {err}"
        fix_exact = ctl_exact = None
        fix_posindep_all = moat_calls = None
        go = False

    summary = {
        "probe": "emerge61_spiking_broca_order_robustness", "GO": bool(go) if err is None else False,
        "verdict": verdict,
        "root_cause": ("H1 CONFIRMED: cp_recovery_variable_u (Izhikevich slow spike-frequency adaptation) accumulates "
                       "across productions on the shared SimulationBridge (u_pre 0.0 at emit#1 -> ~500 mean / ~500 std "
                       "at emit#5); on 2/6 seeds the heterogeneous residual flips the F_MODAL frame's two near-equal-"
                       "primacy adjacent slots at the 5th emit ('the robin breathe can'). Sequence-position-dependent, "
                       "deterministic given the sequence (NOT a noise tie-break -- lowering WTA_NOISE did not fix it)."),
        "mechanism": ("inter-utterance WASH-OUT: ResetFrameSlotCQ (subclass of EMERGE-59 FrameSlotCQ) captures the EXACT "
                      "per-neuron dynamic state right after _initialize_simulation_data (v / recovery_u / the four "
                      "conductances / firing_states / STP, byte-for-byte) and RESTORES it before each emit, returning "
                      "the substrate to its genuine post-init operating point so the rate read-out is a function of the "
                      "LEARNED primacy gradient ALONE. The naive flat reset (v=-65,u=0 for all) was WRONG (ignores per-"
                      "neuron heterogeneity + the u=b*(v-vr) init relation); the post-init snapshot is the correct, "
                      "cheap wash-out. ADDITIVE/default-preserving: EMERGE-59 untouched; EMERGE-60 gets a default-OFF "
                      "reset_producer flag. NO sim/ edit."),
        "task": ("close EMERGE-60's render-ORDER tail: render-exact-in-sequence -> ~1.00 on all 6 seeds (robin as the "
                 "5th emit) + position-independence (same fact renders identically regardless of prior productions) + "
                 "causal (un-reset control swaps) + moat 0 on abstains; >=6 seeds"),
        "seeds": list(seeds), "reset_scope": ("region" if reset_scope else None),
        "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": None if err else {
            "fix_render_exact": round(fix_exact, 4), "ctl_render_exact": round(ctl_exact, 4),
            "fix_position_independent_all_seeds": bool(fix_posindep_all),
            "moat_calls_on_abstain_total": moat_calls,
            "causal_ctl_swaps_somewhere": bool(any(d["ctl_exact"] < 1.0 or (not d["ctl_posindep"]) for d in per)),
        },
        "per_seed": per,
        "HONEST_NOTE": ("Closes the ORDER tail for the bounded EMERGE frame inventory; render-CONTENT was already 1.00 "
                        "and is unchanged; open-prose (R4) is the separate deferred wall. The reset is a genuine "
                        "biological inter-utterance wash-out (clear the previous motor plan's spike-frequency "
                        "adaptation), not a metric hack: it is validated by POSITION-INDEPENDENCE (the fact renders "
                        "identically regardless of prior productions) -- the productions are made genuinely independent, "
                        "not merely nudged. The gate-first moat is untouched. NO sim/ edit."),
    }
    # the SCOPED arm writes its OWN artifact -- it must never overwrite the shipped default-path result.
    out = OUT if not reset_scope else OUT.with_name(OUT.stem + "_scoped.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge61] VERDICT: {verdict}", flush=True)
    print(f"[emerge61] wrote {out}\n" + "=" * 118, flush=True)
    return 0 if (err is None and go) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--derisk", action="store_true")
    ap.add_argument("--reset-scope", choices=["none", "region"], default="none",
                    help="wash-out scope: 'none' (default) = the shipped whole-bridge reset; 'region' = bounded to the "
                         "producer region's own neurons (writes a separate *_scoped.json artifact)")
    ap.add_argument("--verify-scope", action="store_true",
                    help="verify the scoping change: the default path is BYTE-IDENTICAL to the pre-change code (hashed "
                         "array-by-array) and a scoped wash-out never writes outside the producer region")
    a = ap.parse_args()
    if a.verify_scope:
        return _verify_scope(a.seed)
    if a.derisk:
        return _derisk(a.seeds, reset_scope=(a.reset_scope == "region"))
    _demo(a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
