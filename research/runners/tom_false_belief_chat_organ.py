"""A5 THEORY OF MIND: the W3 false-belief register wired for LIVE, INCREMENTAL conversation (2026-09-24).

REUSE, NOT REINVENT. `research/runners/_false_belief_register_derisk.py` is the 6/6-seed GO'd Sally-Anne
false-belief register (`research/findings/2026-07-24-W3-false-belief-register-ToM-6seed-GO-adversarially-
verified-immunity-claim-corrected.md`, commit `b5804d092`): three GNW single-content attractor stores
(`world`=reality, `belief`=the OTHER agent's tracked belief, `self`=the system's own belief), wired with
`sim`'s own `transmission_gate="witness_other"` gating the `world -> belief` write. That module runs a FIXED
two-event trial (place-then-move, `_run_tom_trial`) built for offline evaluation; it is NOT edited here.

THIS MODULE adds only the ORCHESTRATION a live conversation needs: an object can be placed, an agent can leave
and return, the object can be moved any number of times, and a belief can be queried at any point in that
sequence -- not just after exactly two fixed events. It does this by extracting the derisk's own
`_write_event` closure (`_false_belief_register_derisk.py:249-278`) into a standalone method that calls the
SAME primitives the derisk exports (`_restore_slice`, `build_tom_bridge`, `IGNITE_PA`, `HOLD_STEPS`,
`WRITE_DRIVE_STEPS`, `W_WRITE`, `K_LOC`, `STORE_ASSEMBLY`, `FREE_STEPS`, `_argmax_loc`) and a query method that
replays the derisk's own query window (zero-current settle -> late-window `cp_firing_states` rate read ->
`argmax`) at any point in the sequence, not only after the trial's final event. `git diff sim/` is empty.

VALIDATED OPERATING POINT (unchanged from the GO'd run): `helper_pa=5000.0` (the write-reliability fix,
2026-07-24), `drive_steps=WRITE_DRIVE_STEPS` (50), `w_write=W_WRITE` (26.0) -- the same knobs `main()`'s
argparse defaults used for the 6/6-seed GO, not the function-default `helper_pa=0.0` some direct callers of
`evaluate_seed` use for a cheaper/less-reliable smoke.

HONEST SCOPE (declared, not claimed closed):
  * WITNESSING is supplied by the CALLER (the webapp comprehension boundary in `webapp/false_belief_chat.py`
    decides whether the tracked agent was "present"); this organ only ENACTS a witnessed/unwitnessed write on
    the spiking substrate -- it does not itself compute presence.
  * The belief/reality/self READ-OUT is a host `argmax` over the late-window firing rate, the same instrument
    `_argmax_loc` already uses in the GO'd de-risk (the nav-readout-scaffold precedent the W3 finding itself
    names).
  * ONE scenario (one tracked agent, one object, up to `K_LOC=4` distinct locations) per organ instance. A
    conversation wanting a second concurrent false-belief scenario needs a second organ instance (the webapp
    layer starts a fresh one on a new PLACE sentence naming a different tracked agent).

LESION (load-bearing proof, mirrors the derisk's `lesion_other`): every write forces the witnessing gate OPEN
(`w_o=1.0` regardless of the caller's `witnessed`) and the query keeps it open too -- the belief store then
mirrors reality on every read, so an unwitnessed-move query answers with the CURRENT (true) location instead
of the stale pre-move one. This is a RUNTIME manipulation on the same intact bridge (nothing is deleted), the
identical class of ablation the derisk's own OTHER-LESION anti-cheat uses.

Backend: uses the process backend (cupy in production, numpy in tests/CPU smokes) via `sim.backend.get_backend`
-- no global-backend flip. Additive; default-OFF via the webapp-side flag (`webapp/false_belief_chat.py`); NO
`sim/` edit.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

# reuse-by-import: the validated GO'd W3 register module (bridge builder + write/query primitives + the
# validated operating-point constants). NO reimplementation of the mechanism.
import research.runners._false_belief_register_derisk as _FB
from sim.backend import get_backend, to_host

_DEFAULT_SEED = 42

# the validated GO'd operating point (main()'s argparse defaults, NOT evaluate_seed()'s cheaper function
# defaults) -- see the module docstring.
_HELPER_PA = 5000.0
_DRIVE_STEPS = None   # None -> _FB.WRITE_DRIVE_STEPS (50)
_W_WRITE = _FB.W_WRITE


class FalseBeliefChatOrgan:
    """ONE live change-of-location scenario: a bridge built once, then an arbitrary sequence of
    `observe_event` writes and `query` reads, any number of times, in any order a conversation narrates them."""

    def __init__(self, seed: int = _DEFAULT_SEED, *, helper_pa: float = _HELPER_PA,
                 drive_steps: Optional[int] = _DRIVE_STEPS, w_write: float = _W_WRITE):
        self.seed = int(seed)
        self.helper_pa = float(helper_pa)
        self.drive_steps = int(drive_steps) if drive_steps is not None else int(_FB.WRITE_DRIVE_STEPS)
        self.w_write = float(w_write)
        self._bridge = None
        self._xp = None
        self._idx = None
        self._snap = None
        self._lock = threading.Lock()
        self.n_events = 0

    # ── the #77 global-RNG footgun guard (same pattern as affective_tom_production_organ._isolated /
    #    affect_drives_chat._isolated): building/stepping this bridge re-seeds the backend RNG from cfg.seed,
    #    which runs off the SAME process-global RNG the rest of the pipeline shares. Snapshot + restore around
    #    every build/step so a triggered turn never perturbs a downstream RNG-dependent organ. ──────────────
    def _isolated(self, fn):
        xp = None
        try:
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
        try:
            return fn()
        finally:
            try:
                np.random.set_state(host_np)
            except Exception:
                pass
            if host_xp is not None:
                try:
                    xp.random.get_random_state().set_state(host_xp)
                except Exception:
                    pass

    def _ensure_bridge(self):
        if self._bridge is None:
            self._bridge, self._xp, self._idx, self._snap = self._isolated(
                lambda: _FB.build_tom_bridge(seed=self.seed, w_write=self.w_write))

    # ── WRITE: place / move the object to `loc` (0..K_LOC-1), `witnessed` decided by the CALLER (the host
    #    presence tracker). Mirrors `_false_belief_register_derisk._run_tom_trial`'s inner `_write_event`
    #    closure EXACTLY (clear-before-write via `_restore_slice`, transmission-gated ignite, then a HOLD),
    #    generalized to run any number of times instead of exactly twice. ────────────────────────────────────
    def observe_event(self, loc: int, witnessed: bool, *, lesion: bool = False) -> int:
        """Write one change-of-location event. `loc` must be in `range(K_LOC)`. Returns `loc` (for chaining)."""
        if not (0 <= int(loc) < _FB.K_LOC):
            raise ValueError(f"loc {loc} outside 0..{_FB.K_LOC - 1} (K_LOC={_FB.K_LOC})")
        self._ensure_bridge()
        self._isolated(lambda: self._write_event(int(loc), bool(witnessed), bool(lesion)))
        self.n_events += 1
        return int(loc)

    def _write_event(self, loc: int, witnessed: bool, lesion: bool) -> None:
        bridge, xp, idx, snap = self._bridge, self._xp, self._idx, self._snap
        world_dev, belief_dev, self_dev = idx["world_dev"], idx["belief_dev"], idx["self_dev"]
        belief_all, self_all = idx["belief_all"], idx["self_all"]

        w_o = 1.0 if lesion else (1.0 if witnessed else 0.0)
        write_belief = w_o > 0.0
        _FB._restore_slice(bridge, snap, idx["world_all"])
        _FB._restore_slice(bridge, snap, self_all)
        if write_belief:
            _FB._restore_slice(bridge, snap, belief_all)
        bridge.cp_external_input_current[:] = 0.0
        bridge.set_transmission_gate("witness_other", w_o)
        bridge.set_transmission_gate("witness_self", 1.0)

        write_pa = float(self.helper_pa) if self.helper_pa > 0.0 else float(_FB.IGNITE_PA)
        for _ in range(self.drive_steps):
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[world_dev[loc]] = xp.float32(_FB.IGNITE_PA)
            bridge.cp_external_input_current[self_dev[loc]] = xp.float32(write_pa)
            if write_belief:
                bridge.cp_external_input_current[belief_dev[loc]] = xp.float32(write_pa)
            bridge._run_one_simulation_step()

        bridge.set_transmission_gate("witness_other", 1.0 if lesion else 0.0)
        bridge.set_transmission_gate("witness_self", 0.0)
        for _ in range(_FB.HOLD_STEPS):
            bridge.cp_external_input_current[:] = 0.0
            bridge._run_one_simulation_step()

    # ── READ: the belief/reality/self argmax at THIS point in the narrated sequence. Mirrors the derisk's own
    #    query window exactly (zero-current settle over FREE_STEPS, late-third rate read, host argmax). Can be
    #    called any number of times, interleaved with further `observe_event` calls (a real conversation may
    #    ask "where will Sally look?" more than once, or ask again after another move). ─────────────────────
    def query(self, *, lesion: bool = False) -> dict:
        self._ensure_bridge()
        return self._isolated(lambda: self._query(bool(lesion)))

    def _query(self, lesion: bool) -> dict:
        bridge, xp, idx = self._bridge, self._xp, self._idx
        world_dev, belief_dev, self_dev = idx["world_dev"], idx["belief_dev"], idx["self_dev"]

        bridge.set_transmission_gate("witness_other", 1.0 if lesion else 0.0)
        bridge.set_transmission_gate("witness_self", 0.0)
        free_steps = int(_FB.FREE_STEPS)
        late_start = free_steps - max(1, free_steps // 3)
        k_loc = int(_FB.K_LOC)
        world_acc = {k: 0 for k in range(k_loc)}
        belief_acc = {k: 0 for k in range(k_loc)}
        self_acc = {k: 0 for k in range(k_loc)}
        for t in range(free_steps):
            bridge.cp_external_input_current[:] = 0.0
            bridge._run_one_simulation_step()
            if t >= late_start:
                for k in range(k_loc):
                    world_acc[k] += int(to_host(bridge.cp_firing_states[world_dev[k]].astype(xp.float64).sum()))
                    belief_acc[k] += int(to_host(bridge.cp_firing_states[belief_dev[k]].astype(xp.float64).sum()))
                    self_acc[k] += int(to_host(bridge.cp_firing_states[self_dev[k]].astype(xp.float64).sum()))
        nlate = float(free_steps - late_start) * float(_FB.STORE_ASSEMBLY)
        world_rates = {k: world_acc[k] / nlate for k in range(k_loc)}
        belief_rates = {k: belief_acc[k] / nlate for k in range(k_loc)}
        self_rates = {k: self_acc[k] / nlate for k in range(k_loc)}
        return {
            "world_loc": _FB._argmax_loc(world_rates),
            "belief_loc": _FB._argmax_loc(belief_rates),
            "self_loc": _FB._argmax_loc(self_rates),
            "world_rates": world_rates, "belief_rates": belief_rates, "self_rates": self_rates,
            "n_events": int(self.n_events), "lesioned": bool(lesion),
        }
