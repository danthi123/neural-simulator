"""GNW N-organ ignition-bus for the production `brain_chat` handler — the DEFAULT organ-combination (+ SHADOW audit).

WHAT THIS IS. `webapp/server.py::brain_chat` resolves a turn to a stored fact by COMBINING several organ reads in
HOST PYTHON (`ChatBrain.gate()` -> a substrate recall + a reverse-binding VERIFY + a router fallback, combined by
`if recalled == p`). The BRAIN-BASED-ONLY standard flags that COMBINATION as a host shortcut: the organs are neural,
but the `if/else` that fuses them is not. The de-risked GNW N-organ ignition bus (`research/runners/
_gnw_norgan_bus_derisk.py`, finding `2026-08-13-gnw-norgan-ignition-bus-...`, 6/6 GO) proved the SUBSTRATE can combine
N>=3 subthreshold organ reads via consensus-ignition + shared-inhibition WTA + re-entry. This module routes that bus
into production two ways:

  (1) DEFAULT COMBINATION PATH (`install_bus_gate` / `gate_via_bus`, 2026-08-13 FLIP -> SCAFFOLD-RETIREMENT). The
      handler installs a wrapper on `chat.gate` so the SUBSTRATE — consensus-ignition + WTA — AUTHORS the organ-
      combination decision on every turn: the ignited patient IS the gate answer, no ignition IS the abstain (the moat
      as a substrate property). The host `if recalled == p` no longer decides. Extraction/comprehension (the
      parser/heuristic that names the (agent, action) queried) is UNCHANGED — only the COMBINATION verdict moves to the
      substrate. RETIREMENT (the follow-on to the flip): the wrapper runs `chat.gate_extract` (extraction + side
      effects only) and `gate_via_bus` commits/vetoes the COVERED class WITHOUT EVER COMPUTING the host combination —
      no `_substrate_recall`, no `_gate_router_combine` on a routable factual recall. (The superseded `bus_authored_svo`
      remains as a read-only re-author for the shadow/flip verify; it computed the host verdict first, then overrode
      it — `gate_via_bus` never computes it.) Both the single-fact path and the default rich path funnel their direct
      recall through `chat.gate`, so the flip covers the default turn. EARNED on a broad real-query panel byte-identical
      to the host gate() on every covered class (`research/runners/_gnw_bus_default_flip_verify.py`) + a call-count
      retirement proof (`research/runners/_gnw_bus_scaffold_retire_verify.py`); the classes the bus does not cover
      (open-ended generation, acquisition, self/identity turns with no substrate query) stay HOST-authored (scoped).
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
# rank-13 scaffold-retirement de-risk flags (both default OFF; see the module-level comment in brain_chat_tui.py
# above `_neural_selfid_enabled`) — imported here so the BUS combiner path can mirror the SAME two extensions
# `ChatBrain.gate()` / `_substrate_recall` carry, rather than silently diverging from them under the flag.
from research.runners.brain_chat_tui import _neural_selfid_enabled, _neural_anaphora_abstain_enabled

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


def _congruence_spiking_enabled() -> bool:
    """rank-8 scaffold-retirement flag (env read only — NO import of `webapp.gnw_congruence_spiking` unless truthy,
    so the default-off path costs nothing). See that module's docstring for the mechanism/contract."""
    return os.environ.get("BRAIN_GNW_CONGRUENCE_SPIKING", "").strip().lower() in ("1", "true", "on", "yes")


def _spiking_congruent(held, proposed, *, seed: int) -> bool:
    """Lazy-imported wrapper around `webapp.gnw_congruence_spiking.spiking_congruent` — only ever called from
    `_organ_reads` when `_congruence_spiking_enabled()` is truthy."""
    from webapp.gnw_congruence_spiking import spiking_congruent
    return spiking_congruent(held, proposed, seed=seed)


def _organ_reads(composer, agent, action, *, seed: int = 42):
    """The THREE REAL production organ reads `gate()` conceptually combines, all voting on the recalled PATIENT:
      organ A — spiking RECALL (forward):   query_patient(agent, action)          -> cand           [gate organ 1]
      organ B — VERIFY re-check:            cand iff query_patient(agent,action)==cand              [gate organ 3]
      organ C — reverse-binding VERIFY:     cand iff query_agent(action, cand)==agent  (distinct substrate read)
    Returns (cand_A, [A, B, C]). When the forward recall misses, A is None -> the moat abstains (primary organ miss).

    RANK-8 SCAFFOLD-RETIREMENT (additive, DEFAULT-OFF; see `webapp/gnw_congruence_spiking.py`). Organs B/C's `==`
    congruence check above is the host string-id shortcut rank-8 targets: a genuinely spiking `pred_k->mm_k`
    match-veto circuit (reusing the SAME ignition-workspace populations already load-bearing on the neural
    thought-swap decision, 6/6-seed GO) can read "does this second read match the first" off neural competition
    instead. `BRAIN_GNW_CONGRUENCE_SPIKING` unset/off (DEFAULT) -> the two `==` checks below run EXACTLY as
    written -> byte-identical to pre-flip production; this module never imports `gnw_congruence_spiking`."""
    try:
        cand_A = composer.query_patient(agent, action)
    except Exception:
        cand_A = None
    # capture the FORWARD-recall trace RIGHT AFTER organ A (before the VERIFY/reverse probes overwrite it). This IS
    # the trace the host `what_does` (= `composer.query_patient(agent, action)`) leaves for the surfaced "brain
    # activity"; the retirement surfaces it so the covered-class activity stays byte-identical to the host (the bus's
    # OWN forward read, not the reverse-binding probe). query_patient assigns a fresh trace object per call, so this
    # reference is stable across organs B/C.
    trace_A = getattr(composer, "last_trace", None)
    if cand_A is None:
        return None, [None, None, None], trace_A
    congruence_spiking = _congruence_spiking_enabled()
    # organ B: the gate's own VERIFY re-read (the answer must be the spiking recall). Same call -> corroborates a
    # genuine recall; would diverge only if recall were nondeterministic (it is not) -> it withholds otherwise.
    try:
        raw_B = composer.query_patient(agent, action)
        if congruence_spiking:
            cand_B = cand_A if _spiking_congruent(cand_A, raw_B, seed=seed) else None
        else:
            cand_B = cand_A if raw_B == cand_A else None
    except Exception:
        cand_B = None
    # organ C: the reverse role-binding must recover the agent (a genuinely DIFFERENT substrate read).
    try:
        raw_C_agent = composer.query_agent(action, cand_A)
        if congruence_spiking:
            cand_C = cand_A if _spiking_congruent(agent, raw_C_agent, seed=seed) else None
        else:
            cand_C = cand_A if raw_C_agent == agent else None
    except Exception:
        cand_C = None
    return cand_A, [cand_A, cand_B, cand_C], trace_A


def bus_combine(composer, agent: str, action: str, all_concepts, *, seed: int = 42,
                lesion: bool = False, d_sub: Optional[float] = None, surface_forward_trace: bool = False) -> dict:
    """Route the 3 real organ reads for (agent, action) through the spiking ignition bus and return the SUBSTRATE's
    committed decision. `committed` is the ignited patient (or None = abstain). NO host `if/else` selects it — the
    consensus-ignition threshold + shared-inhibition WTA do. Read-only; never mutates the composer or the answer.

    READ-ONLY CONTRACT: the corroboration/reverse-VERIFY organ reads (`query_patient`/`query_agent`) each OVERWRITE
    the composer's read-only `last_trace` (the per-turn "brain activity" the handler surfaces). The bus is a
    combination read-out, not a new query, so it snapshots `last_trace` around its reads.
      * `surface_forward_trace=False` (default, the SHADOW / flip re-author) -> RESTORE the pre-bus `last_trace`: the
        surfaced activity stays whatever the caller's forward recall already set, never the bus's last probe.
      * `surface_forward_trace=True` (the SCAFFOLD-RETIREMENT `gate_via_bus` on the covered class, where the host
        `what_does` no longer runs) -> leave `last_trace` = ORGAN A's forward-recall trace (the bus's OWN forward read,
        which IS `composer.query_patient(agent, action)` == what the host `what_does` would have surfaced). The
        reverse-binding probe never surfaces. This keeps the covered-class activity BYTE-IDENTICAL to the host path."""
    _has_trace = hasattr(composer, "last_trace")
    _saved_trace = getattr(composer, "last_trace", None) if _has_trace else None
    info = None
    try:
        info = _bus_combine_inner(composer, agent, action, all_concepts, seed=seed, lesion=lesion, d_sub=d_sub)
        return info
    finally:
        _fwd = info.pop("_forward_trace", None) if isinstance(info, dict) else None
        if _has_trace:
            try:
                # surface organ A's forward-recall trace on the retirement path (== host); else restore the pre-bus trace
                composer.last_trace = _fwd if surface_forward_trace else _saved_trace
            except Exception:
                pass


def _bus_combine_inner(composer, agent: str, action: str, all_concepts, *, seed: int = 42,
                       lesion: bool = False, d_sub: Optional[float] = None) -> dict:
    cand_A, candidates, trace_A = _organ_reads(composer, agent, action, seed=seed)
    # `_forward_trace` is popped by `bus_combine` before returning (never surfaces in the JSON info block); it carries
    # organ A's forward-recall trace so the retirement path can leave it as the surfaced "brain activity" (== host).
    info = {"organ_reads": list(candidates), "committed": None, "ignited": False, "n_ignited": 0,
            "lesion": bool(lesion), "_forward_trace": trace_A}
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


def gate_via_bus(chat, question: str, *, seed: int = 42, lesion: bool = False):
    """AUTHOR the gate combination with the SUBSTRATE ignition bus WITHOUT ever computing the host `if recalled == p`
    combination on the covered class — the scaffold-retirement follow-on to `bus_authored_svo` (which computed the host
    verdict first, then overrode it). Runs `chat.gate_extract` (extraction + acquisition/anaphora/open-ended side
    effects — all HOST, unchanged), then routes ONLY the covered class through the substrate ignition:
      * ('done', svo)  -> an OUT-OF-SCOPE class (open-ended HypothesisSVO / acquisition) the host mechanism authored; the
                          bus does not combine it -> return it unchanged.
      * ('route', ...) -> the COVERED class: the bus recalls (agent, action) via 3 organ reads + consensus-ignites; the
                          ignited patient IS the answer, no ignition IS the abstain (the moat). THE HOST RECALL VERDICT
                          IS NEVER COMPUTED for this class (no `_substrate_recall`, no `_gate_router_combine`). A bus
                          VETO on an ANAPHORA turn falls through to the host router (an out-of-COVERED-scope residual).
      * ('decline' / 'router') -> out-of-scope (comprehension-decline / self/identity/short): the HOST router authors
                          (kept — the honest residual: extraction, side effects, and out-of-scope classes stay host).
    Returns (bus_svo, info). Read-only; never mutates the composer/answer (the bus organ reads save/restore last_trace).
    `info['authored_by']` and `info['host_combination_computed']` make the retirement auditable per turn."""
    mode_tuple = chat.gate_extract(question)
    mode = mode_tuple[0]

    if mode == 'done':                                       # out-of-scope: open-ended hypothesis / acquisition (host)
        svo = mode_tuple[1]
        _l = (list(svo) if svo is not None else None)
        reason = ("open_ended_generation" if type(svo).__name__ == "HypothesisSVO" else "acquisition_or_abstain")
        return svo, {"routable": False, "reason": reason, "agrees": None, "authored_by": "host_out_of_scope",
                     "host_combination_computed": False, "host_svo": _l, "bus_svo": _l}

    if mode == 'route':                                      # the COVERED class -> the SUBSTRATE authors the verdict
        _, q, agent, action, anaphora_used = mode_tuple
        composer = getattr(getattr(chat, "inner", None), "composer", None)
        agents_set, actions_set, all_concepts = _chat_concepts(chat)
        if composer is None:                                 # defensive: no composer -> degrade to the host router
            svo = chat._gate_router_combine(q)
            _l = (list(svo) if svo is not None else None)
            return svo, {"routable": False, "reason": "no_composer", "agrees": None, "authored_by": "host_router",
                         "host_combination_computed": True, "host_svo": _l, "bus_svo": _l}
        # surface_forward_trace=True: the host `what_does` no longer runs on this covered turn, so the bus leaves its
        # OWN forward-recall trace (organ A) as the surfaced "brain activity" -> byte-identical to the host path.
        info = bus_combine(composer, agent, action, all_concepts, seed=seed, lesion=lesion, surface_forward_trace=True)
        committed = info.get("committed")
        # SELF/IDENTITY candidate-relation retry (rank-13 de-risk, BRAIN_NEURAL_SELFID, default OFF): mirrors
        # `ChatBrain._substrate_recall`'s OWN candidate-relation retry for the IDENTICAL bare-identity shape
        # ('brain', 'isa') -- needed HERE too because this bus combiner reads `composer.query_patient` directly
        # (never calling `_substrate_recall`), so that retry would otherwise be invisible on THIS path, which is
        # the actual production default (`webapp/server.py::brain_reply` installs this combiner unconditionally).
        # Same MISS-ONLY, first-match-wins, host-router preference order (has/have/is/uses/use) -- no new
        # mechanism, a second call site for the identical recipe, reusing `bus_combine` AS-IS (no signature
        # change). On a hit, the RESOLVED action is what gets reported/committed.
        #
        # `action == "isa"` IS LOAD-BEARING (2026-09-05 finding): the plain-path sibling
        # (`ChatBrain._substrate_recall`) shipped WITHOUT this check, so ANY miss with agent=='brain' fired the
        # retry there -- including a query whose 'brain' came from `_resolve_anaphora` substituting an anaphoric
        # 'it' with a WRONGLY-identified discourse referent that happened to equal the literal string 'brain'
        # (seed=43's "...what does it fly?" -> misresolved to 'brain' -> this retry fabricated
        # ['brain','use','spikes'] for a question never about the brain). This bus path never had that bug
        # BECAUSE this `action == "isa"` guard was already here -- do not remove it to "match" the plain path;
        # the plain path was fixed to match THIS. See research/findings/2026-09-05-rank13-selfid-anaphora-
        # production-flip-CORRECTED-DIAGNOSIS.md.
        if committed is None and agent == "brain" and action == "isa" and _neural_selfid_enabled():
            for v_cand in ("has", "have", "is", "uses", "use"):
                cand_info = bus_combine(composer, agent, v_cand, all_concepts, seed=seed, lesion=lesion,
                                        surface_forward_trace=True)
                if cand_info.get("committed") is not None:
                    action, info, committed = v_cand, cand_info, cand_info.get("committed")
                    break
        if committed is not None:
            # THE SUBSTRATE AUTHORS THE COMMIT: the ignited patient IS the answer. Preserve gate()'s anaphora WM write
            # (a concrete patient that is itself an agent becomes the next-turn pronoun referent) — the ONLY side effect
            # the covered-class host verdict used to carry, replicated here so the retirement is behaviour-identical.
            if isinstance(committed, str) and committed in agents_set:
                try:
                    chat._note_referent(committed)
                except Exception:
                    pass
            bus_svo = [agent, action, committed]
            info.update({"routable": True, "agent": agent, "action": action, "anaphora_used": anaphora_used,
                         "authored_by": "bus", "host_combination_computed": False, "bus_svo": list(bus_svo)})
            return bus_svo, info
        # THE SUBSTRATE VETOES (no ignition = the moat). Match gate(): abstain unless anaphora, then the host router try.
        if not anaphora_used:
            info.update({"routable": True, "agent": agent, "action": action, "anaphora_used": False,
                         "authored_by": "bus", "host_combination_computed": False, "bus_svo": None})
            return None, info
        if _neural_anaphora_abstain_enabled():
            # ANAPHORA-MISS EXTENSION (rank-13 de-risk, default OFF): mirrors `gate()`'s SAME extension (see
            # brain_chat_tui.py) -- an anaphora-resolved query the substrate/bus can't confirm ABSTAINS instead of
            # falling to the host router's keyword "rescue" of a possibly-wrong WM referent.
            info.update({"routable": True, "agent": agent, "action": action, "anaphora_used": True,
                         "authored_by": "bus_veto_abstain", "host_combination_computed": False, "bus_svo": None})
            return None, info
        svo = chat._gate_router_combine(q)                   # anaphora-abstain fall-through -> host router (out of scope)
        info.update({"routable": True, "agent": agent, "action": action, "anaphora_used": True,
                     "authored_by": "bus_veto_then_host_router", "host_combination_computed": True,
                     "bus_svo": (list(svo) if svo is not None else None)})
        return svo, info

    if mode == 'decline':                                    # parser declined a factual-shaped Q (comprehension abstain)
        _, q, anaphora_used = mode_tuple
        if not anaphora_used:
            return None, {"routable": False, "reason": "parser_decline_abstain", "agrees": None,
                          "authored_by": "host_abstain", "host_combination_computed": False,
                          "host_svo": None, "bus_svo": None}
        if _neural_anaphora_abstain_enabled():
            return None, {"routable": False, "reason": "parser_decline_anaphora_abstain", "agrees": None,
                          "authored_by": "host_abstain", "host_combination_computed": False,
                          "host_svo": None, "bus_svo": None}
        svo = chat._gate_router_combine(q)
        _l = (list(svo) if svo is not None else None)
        return svo, {"routable": False, "reason": "parser_decline_anaphora_router", "agrees": None,
                     "authored_by": "host_router", "host_combination_computed": True, "host_svo": _l, "bus_svo": _l}

    # mode == 'router': unroutable (self/identity/short) -> the HOST router authors (out of scope, kept)
    q = mode_tuple[1]
    svo = chat._gate_router_combine(q)
    _l = (list(svo) if svo is not None else None)
    return svo, {"routable": False, "reason": "unroutable_self_or_short", "agrees": None, "authored_by": "host_router",
                 "host_combination_computed": True, "host_svo": _l, "bus_svo": _l}


def install_bus_gate(chat, *, seed: int = 42) -> bool:
    """Idempotently wrap `chat.gate` so the SUBSTRATE ignition bus AUTHORS the organ-combination by DEFAULT and the
    host `if recalled == p` combination is RETIRED on the covered class (the 2026-08-13 scaffold-retirement). The
    wrapper runs `chat.gate_extract` (extraction + acquisition/open-ended/anaphora side effects — all unchanged) via
    `gate_via_bus` and lets the substrate commit/veto WITHOUT ever computing the covered-class host verdict. Contrast
    the earlier FLIP wrapper, which ran the full `orig_gate` (host combination) and then OVERRODE it — here the host
    combination is not computed on the covered class at all. Escapes/safety are unchanged:
      * `BRAIN_GNW_BUS_HOST` truthy -> revert to the ORIGINAL host gate() combination per-call (byte-identical).
      * on any bus exception -> fall back to the original host gate() (a turn never crashes).
    Preserves the original as `chat._gnw_orig_gate`; stashes this turn's bus info on `chat._last_gnw_bus`. Returns True
    if it installed (False if already installed). No `sim/` edit; the ChatBrain instance is a host scaffold."""
    if getattr(chat, "_gnw_bus_installed", False):
        return False
    orig_gate = chat.gate
    chat._gnw_orig_gate = orig_gate

    def _bus_gate(question):
        if bus_host_escape():                                # BRAIN_GNW_BUS_HOST -> the ORIGINAL host gate() combination
            host_svo = orig_gate(question)                   # (extraction + recall + `if recalled == p` — pre-flip prod)
            chat._last_gnw_bus = {"routable": False, "reason": "escape_host", "agrees": None,
                                  "authored_by": "host_escape", "host_combination_computed": True,
                                  "host_svo": (list(host_svo) if host_svo is not None else None)}
            return host_svo
        try:
            bus_svo, info = gate_via_bus(chat, question, seed=seed, lesion=bus_lesion_on())
        except Exception as e:                               # never let the bus crash a turn -> host authors, recorded
            host_svo = orig_gate(question)
            chat._last_gnw_bus = {"routable": False, "reason": f"error:{type(e).__name__}: {e}", "agrees": None,
                                  "authored_by": "host_fallback_on_error", "host_combination_computed": True,
                                  "host_svo": (list(host_svo) if host_svo is not None else None)}
            return host_svo
        chat._last_gnw_bus = info
        return bus_svo

    chat.gate = _bus_gate
    chat._gnw_bus_installed = True
    return True
