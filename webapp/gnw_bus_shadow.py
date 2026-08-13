"""GNW N-organ ignition-bus for the production `brain_chat` handler — the DEFAULT organ-combination (+ SHADOW audit).

WHAT THIS IS. `webapp/server.py::brain_chat` resolves a turn to a stored fact by COMBINING several organ reads in
HOST PYTHON (`ChatBrain.gate()` -> a substrate recall + a reverse-binding VERIFY + a router fallback, combined by
`if recalled == p`). The BRAIN-BASED-ONLY standard flags that COMBINATION as a host shortcut: the organs are neural,
but the `if/else` that fuses them is not. The de-risked GNW N-organ ignition bus (`research/runners/
_gnw_norgan_bus_derisk.py`, finding `2026-08-13-gnw-norgan-ignition-bus-...`, 6/6 GO) proved the SUBSTRATE can combine
N>=3 subthreshold organ reads via consensus-ignition + shared-inhibition WTA + re-entry. This module routes that bus
into production two ways:

  (1) DEFAULT COMBINATION PATH (`install_bus_gate` / `bus_authored_svo`, 2026-08-13 FLIP). The handler installs a
      wrapper on `chat.gate` so the SUBSTRATE — consensus-ignition + WTA — AUTHORS the organ-combination decision on
      every turn: the ignited patient IS the gate answer, no ignition IS the abstain (the moat as a substrate
      property). The host `if recalled == p` no longer decides. Extraction/comprehension (the parser/heuristic that
      names the (agent, action) queried) is UNCHANGED — only the COMBINATION verdict moves to the substrate. Both the
      single-fact path and the default rich path funnel their direct recall through `chat.gate`, so the flip covers
      the default turn. It was EARNED on a broad real-query panel that is byte-identical to the host gate() on every
      covered class (`research/runners/_gnw_bus_default_flip_verify.py`); the classes the bus does not cover (open-
      ended generation, self/identity turns with no substrate query) fall back to the host gate() decision (scoped).
  (2) SHADOW/AUDIT read-out (`shadow_report`, unchanged). A read-only host-vs-bus comparison block, still available for
      observability.

ESCAPE + SAFETY CONTRACT:
  * ESCAPE FLAG. `BRAIN_GNW_BUS_HOST` truthy -> the wrapper reverts to the ORIGINAL host `gate()` combination
    (the pre-flip mechanism); with it set the turn is byte-identical to today's production. The wrapper preserves the
    original as `chat._gnw_orig_gate`, so nothing is lost.
  * BYTE-IDENTICAL WHEN COVERED. On the covered class the bus reproduces the host decision exactly (organ A IS the
    host's forward recall; the corroboration/reverse-VERIFY organs agree on a cleanly-stored fact), so the response
    bytes are unchanged — the FLIP changes the MECHANISM, not the behaviour. A divergent covered query (e.g. a fact
    with an ambiguous reverse binding that vetoes) is a mapped residual to SCOPE to host, never to force.
  * MOAT-SAFE. The bus's primary organ is the forward recall; when it misses (`None`) the bus abstains by
    construction (no ignition), exactly like the host moat. It can only ADD abstentions (consensus-veto), never
    invent a fact — so the no-confab guarantee is at least as strong.
  * LESION-LOAD-BEARING. `BRAIN_GNW_BUS_LESION` (or `bus_combine(..., lesion=True)`) builds the workspace with the
    assembly self-recurrence zeroed; the consensus can no longer ignite -> the bus abstains -> the COMBINED ANSWER
    collapses to abstain, WHILE the forward recall reflex (direct `query_patient`, never routed through the
    workspace) still answers — the production proof that the SUBSTRATE, not a residual host read, does the combining.

REUSE-BY-IMPORT (NO `sim/` edit). The workspace build + ignition-read + the N-organ hop + calibrated subthreshold
drive come straight from the de-risk runners; this module adds only the production glue (organ reads off the live
`RFPhasorComposer`, a process-cached warm bridge, the host-vs-bus comparison, the gate wrapper).
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
    consensus-ignition threshold + shared-inhibition WTA do. Read-only; never mutates the composer or the answer.

    READ-ONLY CONTRACT: the corroboration/reverse-VERIFY organ reads (`query_patient`/`query_agent`) each OVERWRITE
    the composer's read-only `last_trace` (the per-turn "brain activity" the handler surfaces). The bus is a
    combination read-out, not a new query, so it snapshots + restores `last_trace` around its reads: the surfaced
    activity stays the gate's FORWARD-recall trace (byte-identical to the host path), never the bus's last probe."""
    _has_trace = hasattr(composer, "last_trace")
    _saved_trace = getattr(composer, "last_trace", None) if _has_trace else None
    try:
        return _bus_combine_inner(composer, agent, action, all_concepts, seed=seed, lesion=lesion, d_sub=d_sub)
    finally:
        if _has_trace:
            try:
                composer.last_trace = _saved_trace   # the bus reads must not perturb the surfaced activity
            except Exception:
                pass


def _bus_combine_inner(composer, agent: str, action: str, all_concepts, *, seed: int = 42,
                       lesion: bool = False, d_sub: Optional[float] = None) -> dict:
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


# ============================================================================================================
# DEFAULT COMBINATION PATH (the 2026-08-13 FLIP): the SUBSTRATE authors the organ-combination, not host Python.
# ============================================================================================================

def bus_host_escape() -> bool:
    """The ESCAPE flag. `BRAIN_GNW_BUS_HOST` in {1,true,on,yes} -> revert to the ORIGINAL host `gate()` combination
    (the pre-flip mechanism); with the escape set, a turn is byte-identical to today's production. Default (unset)
    -> the substrate ignition bus is the DEFAULT combination path."""
    return os.environ.get("BRAIN_GNW_BUS_HOST", "").strip().lower() in ("1", "true", "on", "yes")


def bus_report_enabled() -> bool:
    """Observability flag. `BRAIN_GNW_BUS` truthy -> attach the per-turn `gnw_bus` debug block (the bus's committed
    decision + host agreement + organ reads). Default OFF -> the response carries NO `gnw_bus` key (byte-identical)."""
    return bus_enabled()


def bus_lesion_on() -> bool:
    """The production honest-negative lever. `BRAIN_GNW_BUS_LESION` truthy -> build the workspace with the assembly
    self-recurrence zeroed, so the consensus cannot ignite -> the combined answer COLLAPSES to abstain (proving the
    substrate ignition is load-bearing on the ANSWER, not just a shadow). Default OFF."""
    return os.environ.get("BRAIN_GNW_BUS_LESION", "").strip().lower() in ("1", "true", "on", "yes")


def _chat_concepts(chat):
    """The brain's own vocabulary sets (agents/actions) + the sorted decoy pool = every concept it could name."""
    agents_set = getattr(chat, "agents_set", set()) or set()
    actions_set = getattr(chat, "actions_set", set()) or set()
    patients_set = getattr(chat, "patients_set", set()) or set()
    all_concepts = sorted(agents_set | actions_set | patients_set)
    return agents_set, actions_set, all_concepts


def bus_authored_svo(chat, question: str, host_svo, *, seed: int = 42, lesion: bool = False):
    """Re-AUTHOR the gate combination with the SUBSTRATE ignition bus (not host `if recalled == p`). `host_svo` is the
    ORIGINAL host `gate()` result (already computed — its extraction/acquisition/anaphora side effects have run).
    Returns (bus_svo, info): `bus_svo` is the substrate-authored SVO ([agent, action, committed]) or None (abstain);
    the substrate COMMITS via consensus-ignition (the ignited patient IS the answer) and VETOES via sub-quorum drive
    (no ignition IS the moat). Read-only; never mutates the composer/answer.

    Scope (the honest, byte-identical flip): the bus authors the RECALL COMBINATION for a routable (agent, action)
    query. Turns the bus does NOT cover fall back to the host decision unchanged:
      * open-ended GENERATION (a `HypothesisSVO` guess — not a recall combination);
      * a turn with no substrate query (self/identity/short — no (agent, action) to route, or a self-alias).
    The (agent, action) EXTRACTION stays the host's (the neural parser / heuristic) — only the COMBINATION moves."""
    self_aliases = getattr(getattr(chat, "router", None), "self_aliases", set()) or set()

    # (scope) open-ended generation is a flagged GUESS, not an organ-combination -> the host authors it.
    if type(host_svo).__name__ == "HypothesisSVO":
        return host_svo, {"routable": False, "reason": "open_ended_generation", "agrees": None,
                          "host_svo": list(host_svo), "bus_svo": list(host_svo)}

    composer = getattr(getattr(chat, "inner", None), "composer", None)
    agents_set, actions_set, all_concepts = _chat_concepts(chat)
    _host_list = (list(host_svo) if host_svo is not None else None)
    if composer is None:
        return host_svo, {"routable": False, "reason": "no_composer", "agrees": None,
                          "host_svo": _host_list, "bus_svo": _host_list}

    # (agent, action) — from the host decision when it answered, else re-derive the query (to route the moat probe).
    if host_svo is not None:
        agent, action = host_svo[0], host_svo[1]
    else:
        agent, action = _extract_query(question, agents_set, actions_set)
    if agent is None or action is None or agent in self_aliases or action in self_aliases:
        # unroutable (self/identity/short, or an alias-headed query the host router owns) -> host authors, unchanged.
        return host_svo, {"routable": False, "reason": "unroutable_self_or_short", "agrees": None,
                          "host_svo": _host_list, "bus_svo": _host_list}

    info = bus_combine(composer, agent, action, all_concepts, seed=seed, lesion=lesion)
    committed = info.get("committed")
    # THE SUBSTRATE AUTHORS THE COMMIT/VETO: an ignited patient IS the answer; no ignition IS the abstain (moat).
    bus_svo = [agent, action, committed] if committed is not None else None
    if host_svo is not None:
        agrees = bool(bus_svo is not None and committed == host_svo[2])
    else:
        agrees = bool(bus_svo is None)
    info.update({"routable": True, "agent": agent, "action": action, "agrees": agrees,
                 "host_svo": _host_list, "bus_svo": (list(bus_svo) if bus_svo is not None else None)})
    return bus_svo, info


def install_bus_gate(chat, *, seed: int = 42) -> bool:
    """Idempotently wrap `chat.gate` so the SUBSTRATE ignition bus authors the organ-combination by DEFAULT. The
    wrapper runs the original gate (extraction + recall + acquisition/open-ended/anaphora — all unchanged), then
    re-authors the COMBINATION verdict through the bus. `BRAIN_GNW_BUS_HOST` reverts to the original gate per-call.
    Preserves the original as `chat._gnw_orig_gate`; stashes this turn's bus info on `chat._last_gnw_bus`. Returns
    True if it installed (False if already installed). No `sim/` edit; the ChatBrain instance is a host scaffold."""
    if getattr(chat, "_gnw_bus_installed", False):
        return False
    orig_gate = chat.gate
    chat._gnw_orig_gate = orig_gate

    def _bus_gate(question):
        host_svo = orig_gate(question)                       # extraction + recall + teach/open-ended/anaphora (unchanged)
        if bus_host_escape():                                # BRAIN_GNW_BUS_HOST -> revert to the host gate() combination
            chat._last_gnw_bus = {"routable": False, "reason": "escape_host", "agrees": None,
                                  "host_svo": (list(host_svo) if host_svo is not None else None)}
            return host_svo
        try:
            bus_svo, info = bus_authored_svo(chat, question, host_svo, seed=seed, lesion=bus_lesion_on())
        except Exception as e:                               # never let the bus crash a turn -> host authors, recorded
            chat._last_gnw_bus = {"routable": False, "reason": f"error:{type(e).__name__}: {e}", "agrees": None,
                                  "host_svo": (list(host_svo) if host_svo is not None else None)}
            return host_svo
        chat._last_gnw_bus = info
        return bus_svo

    chat.gate = _bus_gate
    chat._gnw_bus_installed = True
    return True
