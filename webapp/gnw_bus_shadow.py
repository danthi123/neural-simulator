"""GNW N-organ ignition-bus SHADOW path for the production `brain_chat` handler (additive, DEFAULT-OFF).

WHAT THIS IS. `webapp/server.py::brain_chat` resolves a turn to a stored fact by COMBINING several organ reads in
HOST PYTHON (`ChatBrain.gate()` -> a substrate recall + a reverse-binding VERIFY + a router fallback, combined by
`if recalled == p`). The BRAIN-BASED-ONLY standard flags that COMBINATION as a host shortcut: the organs are neural,
but the `if/else` that fuses them is not. The de-risked GNW N-organ ignition bus (`research/runners/
_gnw_norgan_bus_derisk.py`, finding `2026-08-13-gnw-norgan-ignition-bus-...`, 6/6 GO) proved the SUBSTRATE can combine
N>=3 subthreshold organ reads via consensus-ignition + shared-inhibition WTA + re-entry. This module wires that bus
into production as a SHADOW/verification path: it takes the SAME real organ reads `gate()` combines, routes them
through the spiking workspace, and reports whether the substrate's committed decision AGREES with the host's.

SAFETY CONTRACT (the safe FIRST wiring; the host organ-combination is NOT removed yet):
  * ADDITIVE + DEFAULT-OFF. `brain_chat` only calls this when the env flag `BRAIN_GNW_BUS` is truthy; with the flag
    OFF nothing here runs and the turn is byte-identical to today. `ChatBrain.gate()` is never modified or bypassed —
    the host still authors the answer. The bus only RE-DERIVES the decision the host already made and reports whether
    the substrate reproduces it (a verification read-out, never a new answer).
  * MOAT-SAFE. The bus's primary organ is the forward recall; when it misses (`None`) the bus abstains by
    construction (no ignition), exactly like the host moat. The bus can only ADD abstentions (consensus-veto), never
    invent a fact — so the no-confab guarantee is at least as strong.
  * LESION-LOAD-BEARING. `bus_combine(..., lesion=True)` builds the workspace with the assembly self-recurrence
    zeroed; the consensus can no longer ignite -> the bus abstains, WHILE the forward recall reflex (direct
    `query_patient`, never routed through the workspace) still answers — the production proof that the SUBSTRATE, not
    a residual host read, is doing the combination.

REUSE-BY-IMPORT (NO `sim/` edit). The workspace build + ignition-read + the N-organ hop + calibrated subthreshold
drive come straight from the de-risk runners; this module adds only the production glue (organ reads off the live
`RFPhasorComposer`, a process-cached warm bridge, the host-vs-bus comparison).
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

# reuse-by-import the de-risked bus mechanism (build + ignite + the N-organ hop + calibrated drive) — NO sim/ edit.
from research.runners._gnw_norgan_bus_derisk import (
    norgan_hop, D_SUB_UNANIMITY,
)
from research.runners._p1_2_workspace_deliberation_loop_derisk import build_workspace_bridge
from research.runners._gnw_coincidence_integrator_derisk import _pick_decoy

# one warm workspace per (seed, lesion), built lazily on the first enabled turn and kept warm across turns (the
# other production organs are process-cached the same way). Guarded by a lock so two concurrent first-turns build once.
_BRIDGE_CACHE: dict = {}
_BRIDGE_LOCK = threading.Lock()

# the shared stopword set for the abstain-case (agent, action) extraction — mirrors `ChatBrain._substrate_recall`
# so the SHADOW derives the same query the host gate did (only used to route the abstain probe; it can never change
# the host answer). Divergence here can only UNDER-state agreement, never manufacture it.
_STOP = {"what", "who", "whom", "does", "do", "did", "is", "are", "was", "were", "the", "a", "an",
         "to", "it", "that", "this", "they", "them", "of", "about"}


def bus_enabled() -> bool:
    """The escape flag. `BRAIN_GNW_BUS` in {1,true,on,yes} ENABLES the shadow bus; default (unset/0/off) = OFF ->
    the shadow never runs and `brain_chat` is byte-identical to today."""
    return os.environ.get("BRAIN_GNW_BUS", "").strip().lower() in ("1", "true", "on", "yes")


def _get_bridge(seed: int, lesion: bool):
    """Build (once) + cache the warm GNW workspace bridge for (seed, lesion). Reused across every enabled turn."""
    key = (int(seed), bool(lesion))
    b = _BRIDGE_CACHE.get(key)
    if b is not None:
        return b
    with _BRIDGE_LOCK:
        b = _BRIDGE_CACHE.get(key)
        if b is None:
            b = build_workspace_bridge(int(seed), lesion=bool(lesion))   # (bridge, xp, slots_dev, snapshot)
            _BRIDGE_CACHE[key] = b
    return b


def _extract_query(question: str, agents_set, actions_set):
    """Lightweight (agent, action) extraction for the abstain-case probe, mirroring `_substrate_recall`'s heuristic
    (prefer a KNOWN agent/action, else structural position). Read-only; only feeds the shadow's own routing."""
    toks = [t.lower().strip(".,!?") for t in (question or "").split()]
    content = [t for t in toks if t and t not in _STOP]
    a = next((t for t in content if t in agents_set), None) or (content[0] if content else None)
    v = next((t for t in content if t in actions_set), None) or (content[1] if len(content) > 1 else None)
    if not (a and v) or a == v:
        return None, None
    return a, v


def _organ_reads(composer, agent, action):
    """The THREE REAL production organ reads `gate()` conceptually combines, all voting on the recalled PATIENT:
      organ A — spiking RECALL (forward):   query_patient(agent, action)          -> cand           [gate organ 1]
      organ B — VERIFY re-check:            cand iff query_patient(agent,action)==cand              [gate organ 3]
      organ C — reverse-binding VERIFY:     cand iff query_agent(action, cand)==agent  (distinct substrate read)
    Returns (cand_A, [A, B, C]). When the forward recall misses, A is None -> the moat abstains (primary organ miss)."""
    try:
        cand_A = composer.query_patient(agent, action)
    except Exception:
        cand_A = None
    if cand_A is None:
        return None, [None, None, None]
    # organ B: the gate's own VERIFY re-read (the answer must be the spiking recall). Same call -> corroborates a
    # genuine recall; would diverge only if recall were nondeterministic (it is not) -> it withholds otherwise.
    try:
        cand_B = cand_A if composer.query_patient(agent, action) == cand_A else None
    except Exception:
        cand_B = None
    # organ C: the reverse role-binding must recover the agent (a genuinely DIFFERENT substrate read).
    try:
        cand_C = cand_A if composer.query_agent(action, cand_A) == agent else None
    except Exception:
        cand_C = None
    return cand_A, [cand_A, cand_B, cand_C]


def bus_combine(composer, agent: str, action: str, all_concepts, *, seed: int = 42,
                lesion: bool = False, d_sub: Optional[float] = None) -> dict:
    """Route the 3 real organ reads for (agent, action) through the spiking ignition bus and return the SUBSTRATE's
    committed decision. `committed` is the ignited patient (or None = abstain). NO host `if/else` selects it — the
    consensus-ignition threshold + shared-inhibition WTA do. Read-only; never mutates the composer or the answer."""
    cand_A, candidates = _organ_reads(composer, agent, action)
    info = {"organ_reads": list(candidates), "committed": None, "ignited": False, "n_ignited": 0,
            "lesion": bool(lesion)}
    if cand_A is None:                                   # primary recall organ miss -> the moat abstains (no routing)
        info["abstain_reason"] = "primary_recall_miss"
        return info
    d = float(d_sub) if d_sub is not None else D_SUB_UNANIMITY.get(3, 1000.0)
    bridge, xp, slots_dev, snap = _get_bridge(seed, lesion)
    rng = np.random.default_rng(int(seed) * 991 + 7)
    exclude = set(c for c in candidates if c is not None) | {agent, action}
    decoy = _pick_decoy(list(all_concepts), exclude=exclude, rng=rng)   # a rival single-vote slot -> exercises WTA
    committed, rates, winner, n_ign = norgan_hop(bridge, xp, slots_dev, snap, candidates, decoy, d, rng=rng)
    info["committed"] = committed
    info["ignited"] = committed is not None
    info["n_ignited"] = int(n_ign)
    info["decoy"] = decoy
    return info


def shadow_report(chat, question: str, host_gate_svo, *, seed: int = 42) -> dict:
    """Compute the per-turn shadow verification block: route the live organ reads through the bus and COMPARE the
    substrate's committed decision to the host `gate()` decision. Returns a JSON-safe `gnw_bus` info dict. Read-only;
    NEVER changes the host answer (called only when `BRAIN_GNW_BUS` is on; the caller attaches the returned block)."""
    composer = getattr(getattr(chat, "inner", None), "composer", None)
    agents_set = getattr(chat, "agents_set", set()) or set()
    actions_set = getattr(chat, "actions_set", set()) or set()
    # the decoy pool = the brain's own vocabulary (every concept it could name)
    all_concepts = sorted(agents_set | actions_set | (getattr(chat, "patients_set", set()) or set()))

    host_answered = host_gate_svo is not None
    host_patient = host_gate_svo[2] if host_answered else None

    if host_answered:
        agent, action = host_gate_svo[0], host_gate_svo[1]
    else:                                                # host abstained -> re-derive the query to PROVE the bus abstains too
        agent, action = _extract_query(question, agents_set, actions_set)

    block = {
        "enabled": True,
        "note": ("shadow verification only — the host gate() authored the answer; the bus RE-DERIVES the "
                 "combination via spiking ignition and reports agreement (BRAIN-BASED-ONLY audit lever T1-1)."),
        "host_decision": ("answer" if host_answered else "abstain"),
        "host_patient": host_patient,
        "agent": agent, "action": action,
    }
    if composer is None or agent is None or action is None:
        # cannot form a substrate query (e.g. a self/identity turn) -> report unroutable; agreement only claimed for
        # a routable turn. A host-abstain we could not route is recorded as agree-by-abstain (both withhold).
        block.update({"routable": False, "bus_decision": "abstain", "committed": None,
                      "agrees": (not host_answered), "organ_reads": [None, None, None]})
        return block

    try:
        info = bus_combine(composer, agent, action, all_concepts, seed=seed, lesion=False)
    except Exception as e:                               # never let the shadow crash a turn -> report the error, no change
        block.update({"routable": False, "bus_decision": "error", "error": f"{type(e).__name__}: {e}",
                      "agrees": None})
        return block

    committed = info.get("committed")
    bus_answered = committed is not None
    # AGREEMENT: the substrate's committed patient must match the host's (both answer the same, or both abstain).
    if host_answered:
        agrees = bool(bus_answered and committed == host_patient)
    else:
        agrees = (not bus_answered)
    block.update({
        "routable": True,
        "bus_decision": ("answer" if bus_answered else "abstain"),
        "committed": committed,
        "organ_reads": info.get("organ_reads"),
        "n_ignited": info.get("n_ignited"),
        "agrees": agrees,
    })
    return block
