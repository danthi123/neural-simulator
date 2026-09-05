"""GNW GLOBAL-WORKSPACE STOP wired into the LIVE `/api/brain-chat` turn -- a conflict-triggered clear-all that empties
the held P1.2 workspace coalition to n_ignited=0 BEFORE the newcomer ignites, so an interrupt gives a CLEAN
single-content workspace instead of the current recurrence-weakening swap that lets stale content bleed in.

WHAT THIS IS (the distributed-overwrite GLOBAL STOP, `2026-08-18-gnw-distributed-overwrite-workspace-PARTIAL.md`,
global-stop capability 6/6 GO). The already-wired GNW chain resolves competition per-turn (the ignition bus commits,
the acc_conflict_gate abstains on a sustained co-ignition, the #85 swap evicts an incumbent by depressing its OWN
recurrence). What it did NOT have is a DISTRIBUTED CLEAR-ALL: a single control signal that drives a co-ignited
MULTI-content workspace to n_ignited=0 uniformly (the localist STN external-inhibition veto stuck at min n_post=2 --
a survivor always remained). The de-risk delivered exactly that: a DIVISIVELY-NORMALIZED workspace (a shared
inhibitory `norm_pool` returns broad conductance feedback -- the divisor -- so no content sits in a deep isolated
self-sufficient basin) plus a CONFLICT-TRIGGERED depression of the SHARED recurrent resource (Tsodyks-Markram STD,
globalized across the shared pool): because every ignited pattern draws on the shared recurrence, depleting it
de-ignites ALL content UNIFORMLY -> n_ignited -> 0. It is gain-withdrawal from a SHARED resource, not a huge external
g_i, so there is no driving-force reversal / rebound: self-extinction is the natural collapse.

WHERE IT ACTS (the consumer -- the interrupt / hard-topic-break turn). On a turn the already-computed spiking reads
flag as a strong interrupt / high-conflict:
  * the gnw-deliberation `acc_conflict_gate` reporting SUSTAINED multi-candidate co-ignition (`chat._last_gnw_delib`
    with `n_ignited >= 2`), OR
  * a HARD TOPIC-BREAK the #85 swap detector flags (`chat._last_swap_drives` with `swapped == True`),
this organ drives the held coalition (a stale incumbent + the newcomer == a 2-content conflict) into the
divisively-normalized distributed workspace and applies the conflict-triggered depression STOP: the held coalition
CLEARS to n_ignited=0, so the newcomer ignites into a CLEAN, single-content workspace instead of co-igniting with the
stale content (which would bleed into the reply). The clean clear is the substrate's own global stop, NOT a host state
reset (`host_workspace_reset_calls == 0` in the de-risk -- the emptying is the synaptic depression).

THE COUPLING (what makes it LOAD-BEARING, not observe-only; mirrors the #85 swap-DRIVES / #84 affect-DRIVES pattern).
  * A CLEAN neural stop (a 2-content conflict `n_pre >= 2` driven to `n_post == 0`) -> a short CLEARING lead prepended
    to the answer surface (`"Setting the held thread aside — "`) -- the honest EXPRESSION of the global stop the
    substrate just performed (the held coalition was cleared before the newcomer's answer). No trigger, or a stop that
    does NOT reach n_post==0, -> NO lead (byte-identical). The FACT after the lead is the SAME gate-matched,
    moat-verified answer: the stop frames HOW the reply opens (does it announce a clean-slate clear?), never WHICH
    fact is true and never whether an unmatched cue abstains. This is the single coupling this module wires.

THE HONESTY FLOOR (preserved BY CONSTRUCTION). The moat / recall / abstain verdict runs FIRST and unchanged; the stop
coupling only DECORATES an already-matched answer surface with a clearing lead. It never enters the certainty band,
never manufactures a fact, never flips an abstain into an assert. The content fields (`abstained`, `recalled_svo`,
`verified`) are BYTE-IDENTICAL with the coupling on or off; only the answer SURFACE (the optional lead) and the
additive `gnw_stop` trace change.

LESION (the load-bearing / brain-based proof -- the de-risk's OWN oracle). `BRAIN_GNW_STOP_LESION=1` ZEROES the
SHARED-RESOURCE-DEPRESSION term (the conflict boost gain -> 0), so the conflict-triggered depression never fires: the
two-content workspace STAYS >=2 co-ignited (`n_post >= 2`, the localist boundary the divisive-norm+STD stop surpasses)
-- the stale content is NOT cleared and would bleed in. Because the clean-stop condition (`n_post == 0`) is no longer
met, the clearing lead VANISHES and the surface reverts to the byte-identical no-lead (coupling-off) answer. So the
surface change RIDES the SPIKING depression of the shared recurrence, not a host `if interrupt`: zero the depression
term and the clearing acknowledgment disappears even though the world input (an interrupt) is unchanged.

CONTRACT (additive, reversible, byte-identical-off). 2026-08-26 FLIPPED DEFAULT-ON (wave 3, 6/6 flip-soak GO):
  * `stop_enabled()` (`BRAIN_GNW_STOP`, DEFAULT-ON) gates the whole block. `BRAIN_GNW_STOP=0` disables it: no
    workspace is built, no read runs, no `gnw_stop` key is attached, and NO clearing lead is prepended -> the turn is
    BYTE-IDENTICAL to pre-wiring.
  * The stop workspace runs on its PRIVATE RNG timeline and the host process-global RNG (numpy + the sim backend) is
    snapshotted/restored around every read (the #77/#85 global-RNG footgun, inherited): enabling this module cannot
    perturb the downstream RNG-dependent organs, so the OTHER response fields stay byte-identical.
  * The workspace build (~0.12s) is lazy on the first interrupt turn and kept warm; each subsequent interrupt turn
    runs one ~0.1s stop decision.

REUSE-BY-IMPORT (NO `sim/` edit). The divisively-normalized distributed workspace build, the Tsodyks-Markram shared-
recurrence depression (the STOP effector) and the conflict-triggered stop protocol come STRAIGHT from the de-risk
runner `research/runners/_gnw_distributed_overwrite_workspace_derisk.py` (global-stop 6/6 GO). This module adds only
the production glue (the interrupt trigger read off the live per-turn reads + the stop-verdict -> clearing-lead map).
`git diff sim/` is empty. FUNCTIONAL correlate only; NO phenomenal claim.

HONEST RESIDUALS (named, not claimed closed).
  1. The held coalition is instantiated as host-supplied external drive (a stale incumbent + the newcomer, world/body-
     legitimate as stimuli, exactly as the swap/deliberation organs drive their workspaces). The CLEAR itself -- the
     drive to n_ignited=0 -- is the substrate's own divisive-norm+STD global stop (lesion-proven).
  2. The verdict->CLEARING-STRING map is a HOST conditioned-articulation scaffold (the discourse "mouth"), exactly the
     sanctioned articulation-crutch pattern (owner: scaffold-ok-as-conditioned-articulation IF the faculty is
     load-bearing on the surface, which the lesion proves -- zero the depression term and the lead collapses).
  3. The conflict boost is a host-read margin scaling a neuromodulatory enhancement of the STD (a faithful
     conflict->neuromodulator effector), host-side until an ACC/BG circuit computes it from synaptic inputs -- the
     named next rung (inherited from the de-risk's remaining-scaffold #2).
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

# reuse-by-import the de-risk GLOBAL-STOP mechanism (divisive-norm workspace build + Tsodyks-Markram shared-recurrence
# depression + the conflict-triggered stop protocol) -- global-stop capability 6/6 GO. NO sim/ edit.
from research.runners._gnw_distributed_overwrite_workspace_derisk import (
    build_overwrite_bridge, WorkspaceDepression, run_conflict_stop,
    BOOST_GAIN, BOOST_SCALE, MARGIN_REF, PULSE_DURATION, N_PATTERNS,
)

_DEFAULT_SEED = 42

# The clearing EXPRESSION for a clean neural stop (the conditioned-articulation scaffold; DRIVEN by the stop verdict).
_STOP_LEAD = "Setting the held thread aside — "

# ONE warm distributed workspace per seed, built lazily on the first interrupt turn and kept warm.
_WS_CACHE: dict = {}
_LOCK = threading.Lock()


def stop_enabled() -> bool:
    """The master flag. `BRAIN_GNW_STOP` DEFAULT-ON (2026-08-26 flip, wave 3, 6/6 flip-soak GO): unset -> the
    global-stop consumer is enabled; an explicit off (0/false/off/no/'') -> the handler skips the block entirely
    (no build, no read, no key, no lead) -> BYTE-IDENTICAL to pre-wiring. Mirrors the server.py
    `_GNW_STOP_DEFAULT_ON` anchor and its `_gnw_stop_flag_on()` reader."""
    v = os.environ.get("BRAIN_GNW_STOP")
    return not (v is not None and v.strip().lower() in ("0", "false", "off", "no", ""))


def stop_lesion_on() -> bool:
    """The load-bearing lesion lever (the de-risk's OWN oracle). `BRAIN_GNW_STOP_LESION` truthy -> ZERO the
    shared-resource-depression term (the conflict boost gain -> 0): the conflict-triggered depression never fires, so
    the two-content workspace STAYS >=2 co-ignited (the stale content bleeds) -> the clean-stop condition fails -> the
    clearing lead VANISHES (proving the SPIKING depression of the shared recurrence does the clearing, not a host if)."""
    return os.environ.get("BRAIN_GNW_STOP_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


class _StopWorkspace:
    """A warm divisively-normalized distributed workspace + its Tsodyks-Markram shared-recurrence depression, run on a
    PRIVATE RNG timeline so enabling the organ never perturbs the host process-global RNG (byte-identity of the other
    response fields). One instance per seed, cached in `_WS_CACHE`."""

    def __init__(self, seed: int = _DEFAULT_SEED):
        self.seed = int(seed)
        self._built = None          # (bridge, xp, pats, privs, thal_dev, ws_used, snap, handles)
        self._std = None
        self._rng_state = None      # the stop's PRIVATE RNG timeline (the host process-global RNG is never advanced)
        self._lock = threading.Lock()

    def _isolated(self, fn):
        """Run `fn()` (build + spiking stop) on the stop's PRIVATE RNG timeline, leaving the host process-global RNG
        (numpy + the sim backend) BYTE-UNTOUCHED (the #77/#85 footgun: the workspace build reseeds cfg.seed and the
        stepping draws OU noise off the SAME process-global RNG the rest of the pipeline shares)."""
        xp = None
        try:
            from sim.backend import get_backend
            xp, _ = get_backend()
        except Exception:
            xp = None
        host_np = np.random.get_state()
        host_xp = None
        if xp is not None and xp is not np:
            try:
                host_xp = xp.random.get_random_state().get_state()
            except Exception:
                host_xp = None
        if self._rng_state is None:
            np.random.seed(self.seed)
            if xp is not None and xp is not np:
                try:
                    xp.random.seed(self.seed)
                except Exception:
                    pass
        else:
            try:
                np.random.set_state(self._rng_state["np"])
            except Exception:
                pass
            if xp is not None and xp is not np and self._rng_state.get("xp") is not None:
                try:
                    xp.random.get_random_state().set_state(self._rng_state["xp"])
                except Exception:
                    pass
        try:
            return fn()
        finally:
            st = {"np": np.random.get_state(), "xp": None}
            if xp is not None and xp is not np:
                try:
                    st["xp"] = xp.random.get_random_state().get_state()
                except Exception:
                    st["xp"] = None
            self._rng_state = st
            try:
                np.random.set_state(host_np)
            except Exception:
                pass
            if host_xp is not None:
                try:
                    xp.random.get_random_state().set_state(host_xp)
                except Exception:
                    pass

    def _ensure(self):
        if self._built is None:
            self._built = build_overwrite_bridge(seed=self.seed)
            bridge, xp, _pats, _privs, _thal, ws_used, _snap, _h = self._built
            self._std = WorkspaceDepression(bridge, xp, ws_used)
        return self._built, self._std

    def run(self, n_held: int, *, lesion: bool):
        """Drive `n_held` (clamped to [2, N_PATTERNS]) held contents into the workspace (a stale incumbent + the
        newcomer == a co-ignited conflict) and apply the conflict-triggered depression STOP. lesion=True zeroes the
        shared-resource-depression term (boost_gain=0). Returns (n_pre, n_post, boost, cleared). Runs on the private
        RNG timeline; the host RNG is restored on exit."""
        def _do():
            (bridge, xp, pats, privs, thal_dev, _ws_used, snap, handles), std = self._ensure()
            n = max(2, min(int(n_held), int(N_PATTERNS)))
            contents = tuple(range(n))
            boost_gain = 0.0 if lesion else BOOST_GAIN
            r = run_conflict_stop(bridge, xp, pats, privs, thal_dev, snap, std, handles["thal_tonic_pA"],
                                  do_stop=True, isolate=True, contents=contents,
                                  boost_gain=boost_gain, boost_scale=BOOST_SCALE, margin_ref=MARGIN_REF,
                                  pulse_duration=PULSE_DURATION)
            n_pre = int(r["n_ignited_pre"]); n_post = int(r["n_ignited_post"])
            cleared = bool(n_pre >= 2 and n_post == 0)     # the CLEAN global stop (the substrate's own, not a host reset)
            return n_pre, n_post, float(r["boost"]), cleared
        with self._lock:
            return self._isolated(_do)


def _get_workspace(seed: int) -> _StopWorkspace:
    key = int(seed)
    ws = _WS_CACHE.get(key)
    if ws is not None:
        return ws
    ws = _StopWorkspace(seed)
    with _LOCK:
        existing = _WS_CACHE.get(key)
        if existing is None:
            _WS_CACHE[key] = ws
        else:
            ws = existing
    return ws


def detect_trigger(chat):
    """Read the already-computed per-turn spiking reads for a strong interrupt / hard topic-break. Returns
    (triggered, reason, n_held, newcomer_topic). Two independent triggers (either fires the stop):
      * gnw-deliberation acc_conflict_gate SUSTAINED multi-candidate co-ignition: `chat._last_gnw_delib` with
        `n_ignited >= 2` (a genuinely-competing multi-answer conflict the workspace could not settle to one winner);
      * a HARD TOPIC-BREAK the #85 swap detector flags: `chat._last_swap_drives` with `swapped == True`.
    Read-only; never raises (an unreadable chat -> no trigger). n_held = # held contents to clear (>=2).

    RANK-12 (scaffold_retirement_backlog.md): the boolean COMBINATION above (`n_ignited>=2 OR swapped`) is host
    Python, even though each operand is itself a genuine spiking read-out of another organ. A de-risked (6/6 seeds
    GO) spiking ACC/BG hyperdirect circuit reads the SAME two afferents (`n_ignited`, the swap detector's `mm_peak`
    mismatch-population firing) directly as synaptic input and decides the trigger via spiking integration instead
    -- see `research/runners/_gnw_acc_bg_stop_trigger_derisk.py`. DEFAULT-ON since the 2026-09-05 production-flip
    (`BRAIN_GNW_STOP_TRIGGER_SPIKING`); an explicit falsy leaves EVERYTHING below this point UNCHANGED
    (byte-identical) -- the branch is additive, not a rewrite.

    STDLIB-RANDOM ISOLATION (2026-09-05 production-flip verify caught this): `_accbg`'s own `_isolated()` wrapper
    snapshots/restores `np.random`/`xp.random` around the circuit's build+step (the #77/#85 footgun, inherited) but
    NEVER touched Python's stdlib `random` module -- and the FIRST-import of `_accbg` (which transitively imports
    `research.runners._gnw_acc_bg_stop_trigger_derisk` -> `sim`/`webapp.gnw_deliberation`/`webapp.gnw_thought_swap`)
    happens INSIDE this function, before `_accbg`'s own isolation wrapper is ever entered, so no wrapper inside that
    module could have contained it. Measured effect (`_gnw_stop_trigger_production_flip_verify.py`): this leak
    deterministically flipped the ALREADY-SHIPPED (2026-08-26) STOP-*clear* workspace's own clean-stop outcome at
    seed 42 (`_gnw_global_stop_flip_soak.py`'s coupling check, re-run after this flip: 2/2 fresh-process runs
    reproduced the flip 100% of the time). Wrapping the WHOLE delegation (import + call) below in a stdlib-random
    snapshot/restore closes it at its actual source -- the caller, not a callee that cannot see its own import."""
    try:
        import random as _random
        _host_random_state = _random.getstate()
        try:
            from webapp import gnw_acc_bg_stop_trigger as _accbg
            if _accbg.stop_trigger_spiking_enabled():
                return _accbg.detect_trigger_spiking(chat)
        finally:
            _random.setstate(_host_random_state)
    except Exception:
        pass   # any import/circuit error -> fall through to the original host boolean-OR (never crash a turn)

    reason = None
    n_held = 2
    newcomer = None
    try:
        delib = getattr(chat, "_last_gnw_delib", None)
        if isinstance(delib, dict):
            n_ign = delib.get("n_ignited")
            if isinstance(n_ign, (int, float)) and int(n_ign) >= 2:
                reason = "delib_sustained_coignition"
                n_held = max(n_held, int(n_ign))
    except Exception:
        pass
    try:
        swap = getattr(chat, "_last_swap_drives", None)
        if isinstance(swap, dict) and bool(swap.get("swapped")):
            # a hard topic-break: the incumbent + the newcomer == a 2-content conflict to clear before igniting.
            reason = "swap_topic_break" if reason is None else "delib+swap"
            t = swap.get("new_topic") or swap.get("held_topic")
            newcomer = str(t) if t else None
    except Exception:
        pass
    return (reason is not None), reason, n_held, newcomer


def stop_lead(cleared: bool) -> str:
    """The clearing EXPRESSION for this turn (the conditioned-articulation scaffold; DRIVEN by the neural stop verdict).
    A CLEAN neural stop (the held coalition driven to n_ignited=0) -> the clearing lead; anything else -> '' so the
    surface is byte-identical. The FACT after it is unchanged (the moat/recall verdict is intact) -- this frames
    WHETHER the reply opens by announcing a clean-slate clear, never WHICH fact is true."""
    return _STOP_LEAD if cleared else ""


def observe_turn(chat, message: str = "", *, seed: int = _DEFAULT_SEED) -> Optional[dict]:
    """The production entry point (called AFTER the answer is composed so the deliberation/swap per-turn reads exist).
    Detect a strong-interrupt / hard-topic-break trigger; if present, run ONE neural global stop on the held coalition
    (a 2-content conflict) and map the CLEAN-stop verdict to a clearing lead. Returns the per-turn `gnw_stop` info
    (also stashed on `chat._last_gnw_stop`) or None when there is no trigger (byte-identical: no key, no lead). Never
    raises out -- on any error it returns an inert no-lead info so a turn can never crash."""
    triggered, reason, n_held, newcomer = detect_trigger(chat)
    if not triggered:
        # no interrupt this turn -> the stop does not run; no lead, no key -> byte-identical.
        chat._last_gnw_stop = None
        return None
    lesion = stop_lesion_on()
    try:
        ws = _get_workspace(seed)
        n_pre, n_post, boost, cleared = ws.run(n_held, lesion=lesion)
        lead = stop_lead(cleared)
        info = {
            "on": True, "acted": True, "reason": reason, "n_held": int(n_held),
            "newcomer_topic": newcomer, "n_ignited_pre": int(n_pre), "n_ignited_post": int(n_post),
            "boost": float(boost), "cleared": bool(cleared), "lesioned": bool(lesion), "lead": lead,
            "reason_lead": ("clean_global_stop" if lead else
                            ("lesion_stale_bleed" if lesion else "stop_incomplete")),
        }
    except Exception as e:   # never let the stop coupling crash / change a turn -> inert no-lead info
        info = {"on": True, "acted": False, "reason": reason, "error": f"{type(e).__name__}: {e}",
                "lead": "", "cleared": False, "lesioned": lesion}
    chat._last_gnw_stop = info
    return info
