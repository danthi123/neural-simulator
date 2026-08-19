"""GNW confidence/conflict-GATED deliberation for the production `brain_chat` gate — THE KEYSTONE, WIRED.

WHAT THIS IS (roadmap T1-1 rung d, the "ACT on the conflict/confidence signals we only REPORT" audit item, wired into
the LIVE default `/api/brain-chat` turn). The GNW N-organ ignition bus (`webapp/gnw_bus_shadow.py`) already lets the
SUBSTRATE author the organ-combination: the ignited patient IS the answer, no ignition IS the abstain. But when the
brain has TWO+ genuinely-competing stored answers for a query — >=2 facts bound under the SAME (agent, action) with
DIFFERENT patients — today's bus commits the arbitrary FIRST-match patient (query_patient is first-match; the 3 organs
all vote it), silently discarding the competitor. That is a SHAKY commit: the brain is not actually SURE which single
answer to give. This module wires the keystone's re-entrant, conflict-gated deliberation on top of the bus so the
WORKSPACE'S OWN SPIKING conflict read DECIDES commit-vs-abstain (deliberation-until-sure + halt-if-unsure):

  * PROPOSE (the declared modular-processor boundary, exactly as the keystone/coincidence-integrator declare): the
    composer enumerates the DISTINCT candidate patients bound under (agent, action) — a faithful `_iter_facts` +
    role-unbind read (the same unbind `query_patient` uses), read-only.
  * EVALUATE (the substrate): the candidates are driven EQUALLY into the P1.2 GNW workspace slots (equally-valid
    stored answers); mutual-inhibition WTA + the ignition knee settle the competition. The workspace conflict read is
    `n_ignited` (# slots over the knee, off `cp_firing_states`) + `conf` (the divisive-normalized winner-vs-runner-up
    NMDA balance off `cp_conductance_g_nmda`, the production-default `metacog_production_organ.nmda_norm_margin`).
  * ACC GATE (the keystone `acc_conflict_gate`, reads ONLY the spiking conf + n_ignited + its own retry budget):
    n_ignited==1 & conf>=theta_hi -> ADVANCE/COMMIT (a single clean winner -> the answer stands, byte-identical);
    n_ignited>=2 -> RETRY the deliberation (deterministic single-hop re-drive) up to R_max, then ABSTAIN; a single-but-
    low-conf read -> ABSTAIN. A sustained conflict/low-confidence read makes the brain ABSTAIN ("I don't know")
    instead of committing the shaky first-match. theta is SELF-CALIBRATED from a synthetic SOLO/CONFLICT/NULL battery
    (the keystone `calibrate_theta`), no per-seed hand-tuning.

THE DECISIVE PIECE + THE NAMED DEFERRED REMAINDER (scoped honestly). This wires the confidence/conflict-gated
ABSTAIN-vs-COMMIT on the single production recall hop — the "halt-if-unsure" half of the keystone, load-bearing and
byte-identical. The full MULTI-HOP "deliberation-until-sure" (the variable-depth transitive chase whose re-entrant
CYCLE COUNT emerges across a CHAIN of inferences) is NOT a single production recall turn; it stays the de-risk fixture
(`_gnw_reentrant_metacog_gated_deliberation_derisk`) and is the named deferred rung. On a single hop, RETRY re-drives
the same deterministic read, so a sustained conflict resolves to ABSTAIN after the budget.

CONTRACT:
  * DEFAULT-ON via the master flag `BRAIN_GNW_DELIBERATE` (unset/1/true/on/yes = ON). `BRAIN_GNW_DELIBERATE=0`
    (0/false/off/no) -> the wrapper is a PURE PASS-THROUGH (returns the bus svo untouched) -> byte-identical to today.
  * BYTE-IDENTICAL when there is no genuine conflict. A single-candidate recall drives ONE slot -> n_ignited==1 /
    conf~1 -> ADVANCE -> the answer is unchanged. Only a >=2-competing-candidate turn can change the outcome (the
    standard reactive panel — recall/abstain/learn/anaphora on unique (agent,action) keys — is byte-identical).
  * MOAT-SAFE. It can ONLY ADD abstentions on a genuine multi-answer conflict; it NEVER un-abstains (a bus abstain
    stays an abstain), NEVER invents a fact, and NEVER flips a confident single-answer recall (that is n_ignited==1).
  * LESION-LOAD-BEARING. `BRAIN_GNW_DELIBERATE_LESION=1` runs the conflict read on the workspace built with the
    assembly self-recurrence ZEROED -> the competing candidates cannot sustain co-ignition -> n_ignited collapses ->
    the ACC gate can no longer DETECT the conflict -> the brain COMMITS the shaky first-match again. The abstain is
    caused by the SPIKING workspace competition, not a host `if len(candidates) >= 2`.

REUSE-BY-IMPORT (NO `sim/` edit). The workspace build + the ignition/NMDA read + the ACC gate + the theta calibration
come straight from the keystone de-risk runner; this module adds only the production glue (the candidate enumeration
off the live composer + the gate wrapper). `git diff sim/` is empty.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

# reuse-by-import the keystone spiking mechanism (build + ignition/NMDA read + ACC gate + theta calibration) — NO sim/ edit.
from research.runners._p1_2_workspace_deliberation_loop_derisk import build_workspace_bridge, IGNITE_PA, K_SLOTS
from research.runners._gnw_reentrant_metacog_gated_deliberation_derisk import (
    _ignite_and_read_nmda, _conf_from_nmda, acc_conflict_gate, calibrate_theta,
    R_MAX_DEFAULT, ADVANCE, RETRY, COMMIT, ABSTAIN,
)

# ONE warm workspace per (seed, lesion) + one theta per seed, built lazily on the first conflict turn and kept warm.
_BRIDGE_CACHE: dict = {}
_THETA_CACHE: dict = {}
_LOCK = threading.Lock()

# The default seed for the deliberation workspace (matches the bus's default warm-bridge seed).
_DEFAULT_SEED = 42


def deliberate_enabled() -> bool:
    """The master flag. `BRAIN_GNW_DELIBERATE` DEFAULT-ON (unset/1/true/on/yes). Only an explicit 0/false/off/no
    disables it -> the wrapper becomes a pure pass-through (byte-identical to the pre-deliberation bus)."""
    return os.environ.get("BRAIN_GNW_DELIBERATE", "1").strip().lower() not in ("0", "false", "off", "no")


def deliberate_lesion_on() -> bool:
    """The load-bearing lesion lever. `BRAIN_GNW_DELIBERATE_LESION` truthy -> run the conflict read on the workspace
    with the assembly self-recurrence ZEROED -> the competing candidates cannot sustain co-ignition -> the conflict is
    not detected -> the brain COMMITS the shaky first-match (proving the spiking workspace does the deciding)."""
    return os.environ.get("BRAIN_GNW_DELIBERATE_LESION", "").strip().lower() in ("1", "true", "on", "yes")


def _get_bridge(seed: int, lesion: bool):
    """Cache the warm P1.2 workspace per (seed, lesion). The heavy build runs OUTSIDE the lock (the lock is held only
    for the dict read/store) so no caller ever holds `_LOCK` across a call that would re-acquire it — a nested
    acquisition on a non-reentrant Lock deadlocks."""
    key = (int(seed), bool(lesion))
    b = _BRIDGE_CACHE.get(key)
    if b is not None:
        return b
    built = build_workspace_bridge(int(seed), lesion=bool(lesion))       # (bridge, xp, slots_dev, snapshot) — no lock
    with _LOCK:
        b = _BRIDGE_CACHE.get(key)
        if b is None:
            _BRIDGE_CACHE[key] = built
            b = built
    return b


def _get_theta(seed: int):
    """Self-calibrate theta_hi/theta_lo from the synthetic SOLO/CONFLICT/NULL battery on the INTACT workspace (theta is
    a property of the healthy substrate; the lesion is applied only to the conflict-read bridge). Cached per seed. The
    calibration (and the `_get_bridge` it calls) runs OUTSIDE the lock — the lock only guards the dict store."""
    key = int(seed)
    t = _THETA_CACHE.get(key)
    if t is not None:
        return t
    b, xp, slots, snap = _get_bridge(seed, False)                        # self-contained (its own brief lock) — no nest
    cal = calibrate_theta(b, xp, slots, snap)
    t = (cal["theta_hi"], cal["theta_lo"], cal)
    with _LOCK:
        t2 = _THETA_CACHE.get(key)
        if t2 is None:
            _THETA_CACHE[key] = t
        else:
            t = t2
    return t


def all_candidate_patients(composer, agent, action):
    """PROPOSE — the DISTINCT patient concepts bound under (agent, action) across the store (the modular-processor
    boundary). A faithful role-unbind read mirroring `query_patient`'s non-batch match loop; order-preserving dedup.
    Read-only. Returns [] if the composer cannot be scanned (-> the caller treats it as a single-candidate commit)."""
    out = []
    try:
        it = composer._iter_facts()
    except Exception:
        return out
    for fact, comp in it:
        try:
            if composer.unbind(comp, "agent") == agent and composer.unbind(comp, "action") == action:
                p = composer.unbind(comp, "patient")
                if p is not None and p not in out:
                    out.append(p)
        except Exception:
            continue
    return out


def conflict_gate(n_candidates: int, *, seed: int = _DEFAULT_SEED, lesion: bool = False):
    """EVALUATE + ACC GATE: drive `n_candidates` workspace slots EQUALLY at IGNITE_PA, read the spiking (conf,
    n_ignited), apply the keystone `acc_conflict_gate`. On a single-production-hop, RETRY re-drives the SAME
    deterministic read, so we walk the retry budget then settle (a sustained conflict -> ABSTAIN). Returns
    (decision, conf, n_ignited)."""
    theta_hi, theta_lo, _cal = _get_theta(seed)
    bridge, xp, slots, snap = _get_bridge(seed, lesion)
    n = len(slots)
    drives = [0.0] * n
    for i in range(max(0, min(int(n_candidates), n))):
        drives[i] = IGNITE_PA
    rates, g_nmda = _ignite_and_read_nmda(bridge, xp, slots, snap, drives)
    conf, _winner, n_ign = _conf_from_nmda(rates, g_nmda)
    cyc = 0
    decision = acc_conflict_gate(conf, n_ign, cyc, R_MAX_DEFAULT, theta_hi, theta_lo)
    while decision == RETRY and cyc < R_MAX_DEFAULT:
        cyc += 1
        decision = acc_conflict_gate(conf, n_ign, cyc, R_MAX_DEFAULT, theta_hi, theta_lo)
    return decision, float(conf), int(n_ign)


def deliberate_svo(chat, bus_svo, bus_info, *, seed: int = _DEFAULT_SEED):
    """Given the bus's committed gate decision, let the substrate conflict read DECIDE commit-vs-abstain. Returns
    (svo_out, info). Only a covered-class factual recall that the bus AUTHORED and that has >=2 competing candidate
    patients can change: a sustained workspace conflict -> ABSTAIN (None). Everything else is returned UNCHANGED."""
    info = {"acted": False, "decision": None, "abstained": False, "n_candidates": None,
            "conf": None, "n_ignited": None, "lesion": deliberate_lesion_on()}

    # never un-abstain (moat), never touch an open-ended guess, only act on a bus-AUTHORED covered-class recall.
    if bus_svo is None:
        info["reason"] = "already_abstained"
        return None, info
    if type(bus_svo).__name__ == "HypothesisSVO":
        info["reason"] = "open_ended_generation"
        return bus_svo, info
    if not (isinstance(bus_info, dict) and bus_info.get("authored_by") == "bus"):
        info["reason"] = "out_of_scope_or_no_bus_info"
        return bus_svo, info
    if not (isinstance(bus_svo, (list, tuple)) and len(bus_svo) == 3):
        info["reason"] = "not_svo"
        return bus_svo, info

    agent, action = bus_svo[0], bus_svo[1]
    composer = getattr(getattr(chat, "inner", None), "composer", None)
    if composer is None or not hasattr(composer, "_iter_facts"):
        info["reason"] = "no_scannable_composer"
        return bus_svo, info

    # PROPOSE (read-only): snapshot/restore last_trace so the surfaced per-turn "brain activity" is byte-identical.
    _has_trace = hasattr(composer, "last_trace")
    _saved = getattr(composer, "last_trace", None) if _has_trace else None
    try:
        cands = all_candidate_patients(composer, agent, action)
    finally:
        if _has_trace:
            try:
                composer.last_trace = _saved
            except Exception:
                pass

    n_cand = len(cands)
    info.update({"acted": True, "n_candidates": n_cand, "candidates": list(cands)})
    if n_cand <= 1:
        # a single clean stored answer -> commit unchanged (byte-identical; the workspace is not even run).
        info["decision"] = COMMIT
        return bus_svo, info

    # >=2 genuinely-competing answers -> the SUBSTRATE decides via the workspace conflict read + ACC gate.
    decision, conf, n_ign = conflict_gate(min(n_cand, K_SLOTS), seed=seed, lesion=info["lesion"])
    info.update({"decision": decision, "conf": conf, "n_ignited": n_ign})
    if decision == ABSTAIN:
        info["abstained"] = True
        return None, info
    return bus_svo, info


def install_deliberation_gate(chat, *, seed: int = _DEFAULT_SEED) -> bool:
    """Idempotently wrap `chat.gate` (already the bus gate) so the substrate conflict read GATES commit-vs-abstain on
    a genuine multi-answer conflict. Preserves the pre-deliberation gate as `chat._gnw_predelib_gate`; stashes the
    per-turn info on `chat._last_gnw_delib`. When `BRAIN_GNW_DELIBERATE=0` the wrapper is a pure pass-through. On any
    exception the pre-deliberation gate's decision is returned unchanged (a turn never crashes). Returns True if it
    installed (False if already installed). No `sim/` edit; the ChatBrain instance is a host scaffold."""
    if getattr(chat, "_gnw_delib_installed", False):
        return False
    inner_gate = chat.gate
    chat._gnw_predelib_gate = inner_gate

    def _delib_gate(question):
        bus_svo = inner_gate(question)                       # the bus's committed decision (+ chat._last_gnw_bus)
        if not deliberate_enabled():                         # BRAIN_GNW_DELIBERATE=0 -> pure pass-through (byte-identical)
            chat._last_gnw_delib = {"acted": False, "reason": "disabled"}
            return bus_svo
        try:
            out_svo, info = deliberate_svo(chat, bus_svo, getattr(chat, "_last_gnw_bus", None), seed=seed)
        except Exception as e:                               # never let the deliberation crash / change a turn on error
            chat._last_gnw_delib = {"acted": False, "reason": f"error:{type(e).__name__}: {e}"}
            return bus_svo
        chat._last_gnw_delib = info
        return out_svo

    chat.gate = _delib_gate
    chat._gnw_delib_installed = True
    return True
