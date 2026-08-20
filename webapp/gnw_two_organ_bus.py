"""GNW TWO-GENUINELY-DISTINCT-ORGANS coincidence bus for the production `brain_chat` combine — DEFAULT-OFF.

WHAT THIS IS (and how it differs from `gnw_bus_shadow.py`). The production combine is already authored by the
GNW N-organ ignition bus (`gnw_bus_shadow.install_bus_gate`, default-ON): a turn's recall verdict is committed by
spiking consensus-ignition + WTA, not a host `if recalled == p`. BUT that bus's three organ reads all come from the
SAME FHRR composer (forward `query_patient`, a VERIFY re-read of the same call, and a reverse-binding `query_agent`)
— exactly the "caveat #1" that `research/findings/2026-08-20-gnw-workspace-integrates-two-genuinely-distinct-organs-
6seed-GO.md` closes: a GENUINELY DIFFERENT SPIKING SECOND ORGAN. Today the production `SurpriseProductionOrgan`
(a spiking Izhikevich predictive-coding mismatch circuit, `cp_firing_states[surprise]`) only OBSERVES the turn
(it prepends an honest "that surprises me" notice on an assertion); it is NOT load-bearing on the recall/abstain
combine. This module makes it load-bearing, behind a NEW default-off flag, so the owner can review before any flip.

  organ A — spiking RECALL (composer):  query_patient(agent, action)  -> cand                  [the FHRR read]
  organ B — spiking EXPECTATION-VIOLATION monitor (the PRODUCTION `SurpriseProductionOrgan`, NON-COMPOSER): holds
            its OWN independent expectation e_B[(agent,action)] = the brain's stored patient (an associative next-
            map trained into the Izhikevich circuit, a genuinely different substrate) and CONFIRMS cand against it
            by reading `cp_firing_states[surprise]`: cand matches e_B -> the asserted block is INHIBITED by the
            learned prediction -> CONFIRM (~0 Hz, below the organ's calibrated threshold) -> organ B casts its
            subthreshold vote for slot(cand); cand violates e_B (or the prediction edges are lesioned) -> the
            surprise pool FIRES (> threshold) -> organ B WITHHOLDS.

Each organ writes a SUBTHRESHOLD d_sub drive into the shared K-slot GNW workspace; only the COINCIDENCE of the two
crosses the ignition knee (WTA + NMDA attractor sustain it). The committed spiking winner IS the answer; no
coincidence IS the abstain (a live CONSENSUS-VETO — the brain withholds a conclusion its second organ cannot
corroborate). The AND-over-two-distinct-organs is the neuronal ignition THRESHOLD, not host control flow.

DISCIPLINE (the #1 requirement): ADDITIVE + DEFAULT-OFF + BYTE-IDENTICAL-WHEN-OFF. With `BRAIN_GNW_2ORGAN` unset/off
this module installs NOTHING (the server hook does not even import it), so the turn is byte-identical to today's
production (the existing N-organ bus authors it). Only when the flag is ON does the wrapper wrap `chat.gate`.

BACKEND CONSTRAINT (updated 2026-08-20, the cupy unblock). The surprise organ's confirm/contradict discrimination
turned on per-neuron firing THRESHOLDS drawn from the ACTIVE backend's RNG (numpy != cupy for the same seed), so on
cupy the FS prediction pool underfired (~3x) and CONFIRM fired spuriously — the organ stopped discriminating. The fix
is `backend_neutral_izh_initialization` (default-ON in `build_expectation_circuit`, which the production organ inherits):
it routes threshold init through a host RNG IDENTICAL across backends, so the organ's thresholds — and thus its
confirm<<contradict discrimination — now match numpy byte-for-byte on cupy (VERIFIED: confirm 0.110 << thr 1.387 <<
contradict 2.664 on the RTX 3090; per-turn read 67.6-67.8 ms). So the bus now ACTIVATES on BOTH numpy and cupy — but
ONLY when that backend-neutral INIT is on (`_organ_discriminates()`); if it is ever disabled the bus stays INERT on
cupy (one-time warning) and the turn degrades to the existing default combine — it NEVER runs the mis-discriminating
organ. (The companion `backend_neutral_izh_arithmetic` flag is a separate per-step strict kernel; it is verified INERT
for this organ — byte-identical discrimination AND 0.2 ms/read latency delta — so it is NOT required by the gate.)

LESION-LOAD-BEARING levers (the honest-negative deliverable):
  * `BRAIN_GNW_2ORGAN_WS_LESION=1` — build the workspace with the assembly self-recurrence zeroed: even 2*d_sub
    cannot sustain -> the combined answer COLLAPSES to abstain, WHILE the forward-recall reflex (direct
    query_patient, never routed through the workspace) still answers (the dissociation).
  * `BRAIN_GNW_2ORGAN_ORGANB_LESION=1` — zero the surprise circuit's patient_expected->surprise prediction edges:
    CONFIRM fires as high as CONTRADICT -> organ B can no longer corroborate a genuine match -> it withholds -> the
    coincidence collapses -> abstain. Proves the second vote is caused by organ B's learned SPIKING prediction.

REUSE-BY-IMPORT (NO `sim/` edit, NO re-derivation): the workspace build + ignition-read + the 2-organ coincidence
hop + organ B (the expanded production surprise organ) + the calibrated subthreshold drive come straight from
`research/runners/_gnw_two_distinct_organs_derisk.py` and its parents; this module adds only the production glue
(organ reads off the live brain, a process-cached warm organ B + workspace, the gate wrapper).
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

# reuse-by-import the de-risked two-distinct-organs mechanism (build + coincidence hop + organ B) — NO sim/ edit.
from research.runners._gnw_two_distinct_organs_derisk import (
    build_organ_b, organ_b_confirms, pre_register_expected, coincidence_hop,
    N_TRAINED_DEFAULT, N_NOVEL_DEFAULT,
)
from research.runners._p1_2_workspace_deliberation_loop_derisk import build_workspace_bridge
from research.runners._gnw_coincidence_integrator_derisk import _pick_decoy, D_SUB_DEFAULT


# ── flags (all DEFAULT-OFF) ─────────────────────────────────────────────────────────────────────────────────────
def two_organ_enabled() -> bool:
    """The master switch. `BRAIN_GNW_2ORGAN` in {1,true,on,yes} enables the two-distinct-organs combine.
    DEFAULT-ON (2026-08-20): unset = ON (the 6-seed GO + cupy-precision fix + production end-to-end verify cleared the
    default-on gate on both backends); explicit `BRAIN_GNW_2ORGAN=0`/`off` still disables (escape preserved). When
    disabled the module installs nothing and the turn is byte-identical to the pre-bus production path."""
    return os.environ.get("BRAIN_GNW_2ORGAN", "on").strip().lower() in ("1", "true", "on", "yes")


def ws_lesion_on() -> bool:
    """`BRAIN_GNW_2ORGAN_WS_LESION` truthy -> the workspace self-recurrence is zeroed (the coincidence cannot
    sustain -> the combined answer collapses to abstain; the forward reflex survives). Load-bearing lever."""
    return os.environ.get("BRAIN_GNW_2ORGAN_WS_LESION", "").strip().lower() in ("1", "true", "on", "yes")


def organb_lesion_on() -> bool:
    """`BRAIN_GNW_2ORGAN_ORGANB_LESION` truthy -> organ B's patient_expected->surprise prediction edges are zeroed
    (CONFIRM fires as high as CONTRADICT -> organ B withholds even on a match -> the coincidence collapses)."""
    return os.environ.get("BRAIN_GNW_2ORGAN_ORGANB_LESION", "").strip().lower() in ("1", "true", "on", "yes")


def _organ_backend_neutral_init_on() -> bool:
    """The production organ B is built by `build_expectation_circuit` via `_ExpandedSurpriseOrgan._build_one`, which
    does NOT override `backend_neutral_init` -> the organ inherits the builder's DEFAULT. Backend-neutral threshold
    INIT is the load-bearing cupy fix: it makes the organ's per-neuron thresholds (and thus its confirm<<contradict
    discrimination) IDENTICAL on numpy and cupy. Return True iff that default is on (so the organ discriminates on the
    GPU); if it is ever turned off, the cupy gate below stays inert rather than run the mis-discriminating organ."""
    try:
        import inspect
        from research.runners._spiking_expectation_rpe_derisk import build_expectation_circuit
        return inspect.signature(build_expectation_circuit).parameters["backend_neutral_init"].default is True
    except Exception:
        return False


def _organ_discriminates() -> bool:
    """The safety gate — the two-organ bus may activate ONLY where organ B's surprise discrimination is trustworthy.
    numpy: always (the organ was designed + calibrated there). cupy: ONLY when the production organ is built
    backend-neutral (INIT on), so its thresholds — and its confirm<<contradict discrimination — match numpy
    byte-for-byte on the GPU (the 2026-08-20 unblock). Any other backend, or INIT off on cupy: return False -> the bus
    stays INERT (the existing combine authors the turn, byte-identical) and NEVER runs the mis-discriminating organ."""
    try:
        from sim.backend import get_backend
        backend = get_backend()[1]
    except Exception:
        return False
    if backend == "numpy":
        return True
    if backend == "cupy":
        return _organ_backend_neutral_init_on()
    return False


def _backend_is_numpy() -> bool:
    """Kept for callers/tests that probe the raw backend; the ACTIVE gate is `_organ_discriminates()` (which now
    also admits cupy when the organ is built backend-neutral). Return True iff the process backend is numpy."""
    try:
        from sim.backend import get_backend
        return get_backend()[1] == "numpy"
    except Exception:
        return False


# ── process-cached warm organ B + workspace bridges (built lazily on the first enabled numpy turn) ────────────────
_STATE_LOCK = threading.Lock()
_ORGAN_CACHE: dict = {}          # seed -> the trained surprise organ (organ B), pre-registered on the brain's patients
_BRIDGE_CACHE: dict = {}         # (seed, lesion) -> (bridge, xp, slots_dev, snapshot)
_WARNED_CUPY = False


def _n_blocks_for(stored_patients) -> tuple[int, int]:
    """Size organ B so every stored patient gets its OWN cue-addressable block (avoid the round-robin wrap that would
    let two patients collide on one block). n_trained >= #distinct stored patients (>= the de-risk default)."""
    n = len(set(stored_patients))
    n_trained = max(int(N_TRAINED_DEFAULT), n)
    n_novel = max(int(N_NOVEL_DEFAULT), n)
    return n_trained, n_novel


def _get_organ(seed: int, stored_patients):
    """Build (once) + cache organ B trained + threshold-calibrated on the brain's own patient set. The organ's OWN
    Hebbian training / homeostat / calibration run inside `build_organ_b` (the production surprise organ), unchanged.
    We pre-register the brain's stored patients as cue-addressable blocks so e_B(x)=stored_patient predicts its own
    block (the topographic prediction), i.e. organ B corroborates iff the recall matches its independent expectation."""
    organ = _ORGAN_CACHE.get(int(seed))
    if organ is not None:
        return organ
    with _STATE_LOCK:
        organ = _ORGAN_CACHE.get(int(seed))
        if organ is None:
            n_trained, n_novel = _n_blocks_for(stored_patients)
            organ = build_organ_b(int(seed), n_trained=n_trained, n_novel=n_novel)
            pre_register_expected(organ, sorted(set(stored_patients)))
            _ORGAN_CACHE[int(seed)] = organ
    return organ


def _get_bridge(seed: int, lesion: bool):
    """Build (once) + cache the warm GNW workspace bridge for (seed, lesion). Reused across every enabled turn."""
    key = (int(seed), bool(lesion))
    b = _BRIDGE_CACHE.get(key)
    if b is not None:
        return b
    with _STATE_LOCK:
        b = _BRIDGE_CACHE.get(key)
        if b is None:
            b = build_workspace_bridge(int(seed), lesion=bool(lesion))   # (bridge, xp, slots_dev, snapshot)
            _BRIDGE_CACHE[key] = b
    return b


def _chat_concepts(chat):
    """The brain's own vocabulary sets + the (agent,action)->stored-patient expectation map organ B corroborates."""
    agents_set = getattr(chat, "agents_set", set()) or set()
    actions_set = getattr(chat, "actions_set", set()) or set()
    patients_set = getattr(chat, "patients_set", set()) or set()
    all_concepts = sorted(agents_set | actions_set | patients_set)
    stored_facts = getattr(chat, "stored_facts", []) or []
    e_b = {(a, v): p for (a, v, p) in stored_facts
           if isinstance(a, str) and isinstance(v, str) and isinstance(p, str)}
    stored_patients = sorted({p for (_a, _v, p) in stored_facts if isinstance(p, str)})
    return agents_set, actions_set, all_concepts, e_b, stored_patients


def two_organ_combine(chat, agent: str, action: str, *, seed: int = 42,
                      ws_lesion: bool = False, organb_lesion: bool = False) -> dict:
    """Route (agent, action) through the TWO-DISTINCT-ORGANS spiking coincidence and return the SUBSTRATE's committed
    decision. `committed` is the ignited patient (or None = abstain). NO host `if/else` selects it — organ A (FHRR
    recall) and organ B (the spiking surprise monitor's corroboration against its OWN expectation e_B) each write a
    subthreshold d_sub drive; only their coincidence crosses the ignition knee. Read-only; never mutates the answer.

    (agent A's forward `query_patient` read leaves the composer's `last_trace` = the forward-recall trace — the same
    trace the host `what_does` surfaces as "brain activity" — and organ B / the workspace use SEPARATE bridges, so the
    surfaced activity stays byte-identical to the host forward read.)"""
    composer = getattr(getattr(chat, "inner", None), "composer", None)
    _a, _v, all_concepts, e_b, stored_patients = _chat_concepts(chat)
    info = {"organ_a_recall": None, "organ_b_confirmed": None, "organ_b_surprise_hz": None,
            "committed": None, "ignited": False, "n_ignited": 0,
            "ws_lesion": bool(ws_lesion), "organb_lesion": bool(organb_lesion),
            "expected": e_b.get((agent, action))}
    if composer is None:
        info["abstain_reason"] = "no_composer"
        return info

    # organ A — the FHRR recall (the primary organ; a miss abstains by the moat, exactly like the host).
    try:
        cand = composer.query_patient(agent, action)
    except Exception:
        cand = None
    info["organ_a_recall"] = cand
    if cand is None:
        info["abstain_reason"] = "primary_recall_miss"
        return info

    # organ B — the spiking surprise monitor's corroboration of `cand` against its OWN expectation e_B[(agent,action)].
    organ = _get_organ(seed, stored_patients)
    exp = e_b.get((agent, action))
    confirmed, hz = organ_b_confirms(organ, exp, cand, lesion=bool(organb_lesion))
    info["organ_b_confirmed"] = bool(confirmed)
    info["organ_b_surprise_hz"] = (None if (hz is None or np.isnan(hz)) else float(hz))
    info["organ_b_threshold_hz"] = float(getattr(organ, "threshold", float("nan")))

    # the shared GNW workspace: organ A drives slot(cand); organ B drives slot(cand) IFF it corroborated; a decoy
    # exercises WTA. Only a >=2-vote slot crosses the knee -> a genuine coincidence-ignition commit (or abstain).
    bridge, xp, slots_dev, snap = _get_bridge(seed, bool(ws_lesion))
    decoy = _pick_decoy(list(all_concepts), exclude={cand, exp, agent, action},
                        rng=np.random.default_rng(int(seed) * 991 + 7))
    committed, _rates, _winner, n_ign, _b_slot = coincidence_hop(
        bridge, xp, slots_dev, snap, cand, (cand if confirmed else None), decoy, float(D_SUB_DEFAULT))
    info["committed"] = committed
    info["ignited"] = committed is not None
    info["n_ignited"] = int(n_ign)
    info["decoy"] = decoy
    if committed is None:
        info["abstain_reason"] = ("consensus_veto_organ_b_withheld" if not confirmed
                                  else "no_ignition")
    return info


def two_organ_gate_via(chat, question: str, *, seed: int = 42,
                       ws_lesion: bool = False, organb_lesion: bool = False):
    """AUTHOR the gate combination with the TWO-DISTINCT-ORGANS spiking coincidence for the COVERED routable class,
    mirroring `gnw_bus_shadow.gate_via_bus`'s routing EXACTLY (so extraction + acquisition/open-ended/anaphora side
    effects run ONCE, unchanged) but swapping the covered-class combine from the composer-only N-organ bus to the
    genuinely-distinct 2-organ coincidence. Returns (svo_or_None, info). Read-only; never mutates the composer/answer.

    Out-of-scope classes are HOST-authored, unchanged: open-ended HypothesisSVO / acquisition ('done'); a comprehension
    decline ('decline'); a self/identity/short turn ('router'). Only the routable factual recall ('route') is authored
    by the substrate coincidence — the ignited patient IS the answer, no coincidence IS the abstain (the moat)."""
    mode_tuple = chat.gate_extract(question)
    mode = mode_tuple[0]

    if mode == 'done':                                        # out-of-scope: open-ended hypothesis / acquisition (host)
        svo = mode_tuple[1]
        _l = (list(svo) if svo is not None else None)
        reason = ("open_ended_generation" if type(svo).__name__ == "HypothesisSVO" else "acquisition_or_abstain")
        return svo, {"routable": False, "reason": reason, "authored_by": "host_out_of_scope",
                     "host_combination_computed": False, "bus_svo": _l}

    if mode == 'route':                                       # the COVERED class -> the SUBSTRATE authors the verdict
        _, q, agent, action, anaphora_used = mode_tuple
        agents_set = getattr(chat, "agents_set", set()) or set()
        info = two_organ_combine(chat, agent, action, seed=seed,
                                 ws_lesion=ws_lesion, organb_lesion=organb_lesion)
        committed = info.get("committed")
        if committed is not None:
            # preserve gate()'s ONLY covered-class side effect: a concrete patient that is itself an agent becomes the
            # next-turn pronoun referent (the anaphora WM write), so the retirement is behaviour-identical.
            if isinstance(committed, str) and committed in agents_set:
                try:
                    chat._note_referent(committed)
                except Exception:
                    pass
            bus_svo = [agent, action, committed]
            info.update({"routable": True, "agent": agent, "action": action, "anaphora_used": anaphora_used,
                         "authored_by": "two_organ_bus", "host_combination_computed": False, "bus_svo": list(bus_svo)})
            return bus_svo, info
        # the SUBSTRATE VETOES (no coincidence = the moat). Match gate(): abstain unless anaphora, then the host router.
        if not anaphora_used:
            info.update({"routable": True, "agent": agent, "action": action, "anaphora_used": False,
                         "authored_by": "two_organ_bus", "host_combination_computed": False, "bus_svo": None})
            return None, info
        svo = chat._gate_router_combine(q)                    # anaphora-abstain fall-through -> host router (out of scope)
        info.update({"routable": True, "agent": agent, "action": action, "anaphora_used": True,
                     "authored_by": "two_organ_bus_veto_then_host_router", "host_combination_computed": True,
                     "bus_svo": (list(svo) if svo is not None else None)})
        return svo, info

    if mode == 'decline':                                     # parser declined a factual-shaped Q (comprehension abstain)
        _, q, anaphora_used = mode_tuple
        if not anaphora_used:
            return None, {"routable": False, "reason": "parser_decline_abstain", "authored_by": "host_abstain",
                          "host_combination_computed": False, "bus_svo": None}
        svo = chat._gate_router_combine(q)
        _l = (list(svo) if svo is not None else None)
        return svo, {"routable": False, "reason": "parser_decline_anaphora_router", "authored_by": "host_router",
                     "host_combination_computed": True, "bus_svo": _l}

    # mode == 'router': unroutable (self/identity/short) -> the HOST router authors (out of scope, kept)
    q = mode_tuple[1]
    svo = chat._gate_router_combine(q)
    _l = (list(svo) if svo is not None else None)
    return svo, {"routable": False, "reason": "unroutable_self_or_short", "authored_by": "host_router",
                 "host_combination_computed": True, "bus_svo": _l}


def install_two_organ_gate(chat, *, seed: int = 42) -> bool:
    """Idempotently wrap `chat.gate` so the TWO-DISTINCT-ORGANS spiking coincidence AUTHORS the organ-combination on
    the covered routable class. Installs ONLY when `BRAIN_GNW_2ORGAN` is truthy AND the organ discriminates on this
    backend (`_organ_discriminates()`: numpy always; cupy iff the organ is built backend-neutral, INIT on); otherwise
    installs NOTHING and returns False -> the turn is byte-identical to today's production (the existing N-organ bus
    authors it). On a backend where the organ would NOT discriminate (e.g. cupy with backend-neutral INIT disabled) it
    warns ONCE and stays inert. Preserves the original as `chat._two_organ_orig_gate`; stashes this turn's info on
    `chat._last_two_organ`. On any per-turn bus exception the wrapper falls back to the original gate (a turn never
    crashes). No `sim/` edit."""
    global _WARNED_CUPY
    if not two_organ_enabled():
        return False
    if not _organ_discriminates():
        if not _WARNED_CUPY:
            _WARNED_CUPY = True
            print("[gnw-2organ] BRAIN_GNW_2ORGAN is ON but organ B would NOT discriminate on this backend "
                  "(the surprise organ's confirm/contradict split needs backend-neutral per-neuron thresholds; "
                  "`backend_neutral_izh_initialization` is off, so cupy's RNG-drawn thresholds break the split). "
                  "The two-organ bus stays INERT (the existing combine authors the turn, byte-identical). Re-enable "
                  "the organ's backend-neutral init (or run SIM_BACKEND=numpy) to activate it.", flush=True)
        return False
    if getattr(chat, "_two_organ_installed", False):
        return False

    orig_gate = chat.gate
    chat._two_organ_orig_gate = orig_gate

    def _two_organ_gate(question):
        # RUNTIME re-check: if the flag is toggled off (or the organ would no longer discriminate on this backend)
        # after install, delegate to the ORIGINAL gate so the turn is byte-identical to today's production — the
        # wrapper never changes behaviour unless BRAIN_GNW_2ORGAN is actively on. (Belt-and-suspenders with the
        # flag-guarded server hook.)
        if not (two_organ_enabled() and _organ_discriminates()):
            return orig_gate(question)
        try:
            svo, info = two_organ_gate_via(chat, question, seed=seed,
                                           ws_lesion=ws_lesion_on(), organb_lesion=organb_lesion_on())
        except Exception as e:                                # never let the bus crash a turn -> original authors
            host_svo = orig_gate(question)
            chat._last_two_organ = {"routable": False, "reason": f"error:{type(e).__name__}: {e}",
                                    "authored_by": "orig_gate_fallback_on_error", "host_combination_computed": True,
                                    "bus_svo": (list(host_svo) if host_svo is not None else None)}
            return host_svo
        chat._last_two_organ = info
        return svo

    chat.gate = _two_organ_gate
    chat._two_organ_installed = True
    return True
