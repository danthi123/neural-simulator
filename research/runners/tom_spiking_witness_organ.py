"""Spiking WITNESS for the false-belief register: who saw the move is decided by spikes, not by a host boolean.

Packet `fb-spiking-witness` (session-2 packet factory, 2026-10-01; HANDOFF `docs/drafts/fb-spiking-witness/HANDOFF.md`,
prereg `research/findings/2026-10-01-fb-spiking-witness-PREREGISTRATION.md`).

WHAT IT REPLACES. In `research/runners/tom_false_belief_chat_organ.FalseBeliefChatOrgan` the caller passes
`witnessed: bool` (from `webapp/false_belief_chat._Scenario.present`, a host dict flipped by the LEAVE/RETURN regexes),
and that boolean decides three things on every move: whether the belief store is cleared (`_restore_slice`), whether
the 5000 pA helper ignites the new belief member, and the `witness_other` transmission gate. The integration-wave-2
review (runbook RULING `iw2-verify-go`) ruled that this host decision is why the false-belief chat cannot count as
substrate-owned: "Who saw the move is decided by the host witnessing parse, and that decision gates the write."

WHAT THIS ORGAN DOES INSTEAD (one bridge; the W3 register's three stores, unchanged, plus four populations):

  * `presence` -- an 80-neuron self-recurrent NMDA attractor of the same class as every store member here (the GNW
    Rung-1 loop). ON = the tracked agent is in the room. It is persistent activity, so it carries the state across
    any number of turns without a host variable.
  * `ev_arrive` -- a sensory population. The host renders "the tracked agent is perceived acting / arriving" (a PLACE
    by that agent, a RETURN) as a current into it; its spikes ignite `presence` through synapses.
  * `ev_leave` -> `leave_inh` -- a sensory population for "the tracked agent is perceived leaving" driving an
    inhibitory pool that silences `presence` through synapses.
  * `absence` + `arrive_inh` -- the flip-flop's other half (added after dev seed 7 showed a lone presence attractor
    re-igniting 25-50 ms after leave_inh released it): a departure also ignites `absence`, an arrival also silences
    it, and each attractor keeps the OTHER's inhibitory pool firing, so the state holds with no further input.
  * MODE "readout" (GRID 4-7): `presence` is read over PRES_READ_STEPS with a frozen threshold and that read gates the
    W3 register's validated write (clear, helper, `witness_other`); the decision state is spiking, the actuation is a
    host threshold read (the read-out class). MODE "relay" (GRID 0-3), fully synaptic:
  * `relay_k` (one per location) -- a coincidence population. It receives `world_k` (the perceived new location) and
    `presence` on separate subthreshold projections; it fires only when both are active, and its spikes drive
    `belief_k` (ignite) and the store's other members' inhibition (evict). No `witness_other` gate is ever opened and
    no host clear touches the belief store: whether the move is written is the AND of two spiking populations.

The host's remaining part is the world and the senses: which location the object is now in (the `world` store's drive,
as in the W3 register) and the rendered perceptual events (arrive / leave), routed to the tracked agent's population by
the sentence's parsed agent name. Those are declared residuals: the sentence classifier is a comprehension boundary,
and agent-name routing is host. The DECISION "was the tracked agent present when the object moved" is now the state of
`presence` at the moment `world_k` fires, read by `relay_k` -- substrate-owned.

LESIONS.
  * `lesion=True` (constructor; production flag `BRAIN_FALSE_BELIEF_WITNESS_LESION`): relay mode builds the bridge
    without the presence -> relay projection and drives every relay tonically; readout mode ignores the presence read.
    Either way every move is written whatever `presence` holds. The false-belief probe then reads reality ("box"); a witnessed move is unchanged. This is the same
    prediction the iw2 lesion made, now removing a spiking population's influence instead of a host boolean.
  * `silence_presence=True` (organ diagnostic only, never a production flag): `presence` gets a strong inhibitory
    clamp during every write, so NO move is written; a witnessed move then also reads the stale location. This shows
    the presence population is necessary for a write, not just sufficient to block one.

Backend: process backend (numpy on the pool). No `sim/` edit. Same global-RNG isolation as the W3 organ.
"""
from __future__ import annotations

import os
import sys
import threading
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.regions import BrainRegion, RegionPathway  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402

from research.runners import _false_belief_register_derisk as _FB  # noqa: E402
from research.runners._gnw_rung1_ignition_curve_derisk import (  # noqa: E402
    _build_assembly_loop_population, _snapshot_state, DEFAULT_ATTRACTOR_WEIGHT, SETTLE_STEPS,
)
from research.runners._gnw_rung3_report_reasoning_identity_derisk import _dense_projection  # noqa: E402
from research.runners._self_schema_region_derisk import (  # noqa: E402
    WS_LOOP_GATE, WS_TO_FS_WEIGHT, FS_TO_WS_WEIGHT, IGNITE_PA,
)

_DEFAULT_SEED = 42
K_LOC = _FB.K_LOC
STORE = _FB.STORE_ASSEMBLY            # 80
STORE_FS = _FB.STORE_FS_N             # 50

# ── the four new populations (sizes and weights frozen by the prereg; dev seeds 1-15 fixed them) ──────────────────
PRES_N = 80                           # presence attractor (same member size as a store slot)
PRES_NMDA = True                      # presence keeps the store class's NMDA attractor (dev-tuned below)
EV_N = 40                             # each sensory event population
LEAVE_INH_N = 40                      # the leave-driven inhibitory pool
RELAY_N = 40                          # per-location coincidence population

EV_DRIVE_PA = 2500.0                  # sensory current into an event population while the event is perceived
EV_STEPS = 40                         # perception window of an arrive/leave event
W_EV_PRES = 60.0                       # ev_arrive -> presence (dense)
W_EV_INH = 60.0                        # ev_leave -> leave_inh (dense)
W_HOLD_INH = 20.0                     # presence -> arrive_inh and absence -> leave_inh (the flip-flop's hold)
W_INH_PRES = 100.0                     # leave_inh -> presence (dense, inhibitory by trait)
W_WORLD_RELAY = 1.5                   # world_k -> relay_k (dense; subthreshold alone)
W_PRES_RELAY = 1.5                    # presence -> every relay_k (dense; subthreshold alone)
W_RELAY_BELIEF = 16.0                  # relay_k -> belief_k (dense; ignites the new belief member)
W_RELAY_EVICT = 8.0                   # relay_k -> evict_k (dense)
EVICT_N = 20                          # per-location eviction pool (inhibitory)
W_EVICT_BELIEF = 100.0                 # evict_k -> belief_j, j != k (dense, inhibitory by trait)
RELAY_BIAS_PA = -150.0                   # constant bias on relay cells (negative = hyperpolarizing)
QUERY_RELAY_BIAS = False              # opt-in (Claude 2026-10-06, relay2 review): also apply RELAY_BIAS_PA in
                                      # `_query`, which zeroes the input and steps outside `_run`. False =
                                      # the registered GRID behavior (the bias vanishes at query time).
TONIC_RELAY_PA = 900.0                # lesion: the tonic drive that replaces presence's input to every relay
SILENCE_PA = -4000.0                  # diagnostic: inhibitory clamp on presence during a write
BOUNDED_CONDUCTANCE = True             # the T13 exponential-Euler guard (strong inhibition overshoots without it)
LEAVE_RECEPTOR = "gaba_a"            # receptor of leave_inh -> presence
MODE = "relay"                        # "relay" (fully synaptic AND) | "readout" (threshold read gates the W3 write)
RELAY_MODES = ("relay", "relay_nmda")  # relay_nmda (2026-10-04, dev only, not in GRID): world -> relay on Mg2+-gated NMDA
NMDA_REC_TAU_MS = 100.0               # relay_nmda: slow NMDA decay (the engine's nmda_recurrent default)
NMDA_REC_STRENGTH = 0.05              # relay_nmda: per-spike slow-NMDA increment scale (engine default)
EVICT_RECEPTOR = "gaba_a"              # evict_k -> belief_j receptor (gaba_a = the registered GRID behavior)
RELAY_BELIEF_RECEPTOR = "ampa"        # relay_k -> belief_k exc receptor (ampa = the registered GRID behavior)
BELIEF_FS_RECEPTOR = "gaba_a"          # belief_fs -> belief receptor (gaba_a = the registered GRID behavior)
PRES_READ_STEPS = 20                  # readout mode: presence read window before each write
PRES_MIN = 0.03                       # readout mode: presence rate (spikes/neuron/step) that counts as present
HELPER_PA = 5000.0                    # readout mode: the W3 organ's helper ignite (FalseBeliefChatOrgan._HELPER_PA)  # transmission gate on presence -> relay (only the lesion closes it)
HOLD_STEPS = _FB.HOLD_STEPS
WRITE_STEPS = _FB.WRITE_DRIVE_STEPS   # 50, the W3 register's witnessed-event window


def _dense_inh(pre_idx, post_idx, weight):
    d = _dense_projection(np.asarray(pre_idx), np.asarray(post_idx), float(weight), WS_LOOP_GATE)
    d["conn_type"] = "I_TO_E"
    return d


def build_witness_bridge(seed: int = _DEFAULT_SEED, attractor_weight: float = DEFAULT_ATTRACTOR_WEIGHT,
                         lesion: bool = False):
    """The W3 register's three stores (world / belief / self, each with its shared inhibitory pool) plus presence,
    ev_arrive, ev_leave, leave_inh and (MODE "relay" only) the K_LOC relays and eviction pools, on ONE bridge. In MODE
    "relay", `lesion` builds the bridge WITHOUT the presence -> relay projection (a build-time cut: every arm is its own
    process, and a runtime transmission gate on that projection did not stop its current on dev seed 7). Returns
    (bridge, xp, idx, snap)."""
    relay_mode = MODE in RELAY_MODES
    xp, _ = get_backend()
    n_store = STORE * K_LOC
    regions = [
        BrainRegion(name="world", n_neurons=n_store, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="world_fs", n_neurons=STORE_FS, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="belief", n_neurons=n_store, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="belief_fs", n_neurons=STORE_FS, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="self", n_neurons=n_store, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="self_fs", n_neurons=STORE_FS, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="presence", n_neurons=PRES_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=PRES_NMDA),
        BrainRegion(name="ev_arrive", n_neurons=EV_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="ev_leave", n_neurons=EV_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="leave_inh", n_neurons=LEAVE_INH_N, exc_fraction=0.0, internal_density=0.0,
                    enable_nmda=False),
        BrainRegion(name="absence", n_neurons=PRES_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="arrive_inh", n_neurons=LEAVE_INH_N, exc_fraction=0.0, internal_density=0.0,
                    enable_nmda=False),
    ]
    if relay_mode:
        regions += [
            BrainRegion(name="relay", n_neurons=RELAY_N * K_LOC, exc_fraction=1.0, internal_density=0.0,
                        enable_nmda=False),
            BrainRegion(name="evict", n_neurons=EVICT_N * K_LOC, exc_fraction=0.0, internal_density=0.0,
                        enable_nmda=False),
        ]
    pathways = []
    for store in ("world", "belief", "self"):
        pathways.append(RegionPathway(from_region=store, to_region=f"{store}_fs", density=0.5,
                                      weight_mean=WS_TO_FS_WEIGHT, weight_jitter=0.0, plastic=False))
        _rec = BELIEF_FS_RECEPTOR if (store == "belief" and BELIEF_FS_RECEPTOR != "gaba_a") else None
        pathways.append(RegionPathway(from_region=f"{store}_fs", to_region=store, density=0.5,
                                      weight_mean=FS_TO_WS_WEIGHT, weight_jitter=0.0, plastic=False,
                                      **({"receptor": _rec} if _rec else {})))
    # leave_inh -> presence uses the engine's GABA_B receptor (E_K -90 mV): under this profile the fast GABA_A
    # reversal is shunting, and a GABA_A projection RAISED the attractor's rate on dev seed 7 (2026-10-01).
    # GABA_B's slow K+ current is also the biological terminator of persistent up-states.
    pathways.append(RegionPathway(from_region="leave_inh", to_region="presence", density=1.0,
                                  weight_mean=W_INH_PRES, weight_jitter=0.0, plastic=False, receptor=LEAVE_RECEPTOR))
    # the flip-flop's other half: arrive_inh silences `absence` (same receptor and weight)
    pathways.append(RegionPathway(from_region="arrive_inh", to_region="absence", density=1.0,
                                  weight_mean=W_INH_PRES, weight_jitter=0.0, plastic=False, receptor=LEAVE_RECEPTOR))
    cfg = CoreSimConfig()
    cfg.enable_gabab = True
    cfg.enable_bounded_conductance_integration = BOUNDED_CONDUCTANCE
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    if MODE == "relay_nmda" or RELAY_BELIEF_RECEPTOR == "nmda_slow":
        cfg.enable_nmda_recurrent = True
        cfg.nmda_recurrent_tau_decay_ms = float(NMDA_REC_TAU_MS)
        cfg.nmda_recurrent_propagation_strength = float(NMDA_REC_STRENGTH)
    for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = True
    cfg.stdp_w_max = max(400.0, float(attractor_weight) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(attractor_weight) * 4.0)
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                              gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    rm = bridge.region_manager

    def members(name, size, k):
        base = np.asarray(rm.indices(name), dtype=np.int64)
        return {i: base[i * size:(i + 1) * size] for i in range(k)}

    world_m, belief_m, self_m = members("world", STORE, K_LOC), members("belief", STORE, K_LOC), \
        members("self", STORE, K_LOC)
    relay_m = members("relay", RELAY_N, K_LOC) if relay_mode else {}
    evict_m = members("evict", EVICT_N, K_LOC) if relay_mode else {}
    pres = np.asarray(rm.indices("presence"), dtype=np.int64)
    ev_a = np.asarray(rm.indices("ev_arrive"), dtype=np.int64)
    ev_l = np.asarray(rm.indices("ev_leave"), dtype=np.int64)
    linh = np.asarray(rm.indices("leave_inh"), dtype=np.int64)

    union = dict(rm.build_wiring_plan(seed=int(seed)))
    for k in range(K_LOC):
        union[f"loop_world_{k}"] = _build_assembly_loop_population(world_m[k], float(attractor_weight))
        union[f"loop_belief_{k}"] = _build_assembly_loop_population(belief_m[k], float(attractor_weight))
        union[f"loop_self_{k}"] = _build_assembly_loop_population(self_m[k], float(attractor_weight))
        # the system's own store keeps the W3 always-witnessing write (witness_self opened at every event)
        union[f"write_self_{k}"] = _FB._gated_write_projection(world_m[k], self_m[k], _FB.W_WRITE, "witness_self")
        # the W3 register's own witnessing-gated write (used only by MODE "readout": its gate is set from a frozen
        # threshold read of the presence population, never from a host parse)
        union[f"write_belief_{k}"] = _FB._gated_write_projection(world_m[k], belief_m[k], _FB.W_WRITE, "witness_other")
        if not relay_mode:
            continue
        union[f"world_relay_{k}"] = _dense_projection(world_m[k], relay_m[k], W_WORLD_RELAY, WS_LOOP_GATE)
        if MODE == "relay_nmda":
            # the dendritic AND: the world input arrives on slow, Mg2+-blocked NMDA alone (no fast AMPA onset), so
            # it cannot fire a relay cell at rest; presence's AMPA depolarization relieves the block
            union[f"world_relay_{k}"]["exc_receptor"] = "nmda_slow"
        if not lesion:
            union[f"presence_relay_{k}"] = _dense_projection(pres, relay_m[k], W_PRES_RELAY, WS_LOOP_GATE)
        union[f"relay_belief_{k}"] = _dense_projection(relay_m[k], belief_m[k], W_RELAY_BELIEF, WS_LOOP_GATE)
        if RELAY_BELIEF_RECEPTOR != "ampa":
            union[f"relay_belief_{k}"]["exc_receptor"] = RELAY_BELIEF_RECEPTOR
        # targeted eviction: relay_k excites its own inhibitory pool, which inhibits every OTHER location's belief
        # member (the old belief), never belief_k (the new one)
        union[f"relay_evict_{k}"] = _dense_projection(relay_m[k], evict_m[k], W_RELAY_EVICT, WS_LOOP_GATE)
        others = np.concatenate([belief_m[j] for j in range(K_LOC) if j != k])
        union[f"evict_belief_{k}"] = _dense_inh(evict_m[k], others, W_EVICT_BELIEF)
        if EVICT_RECEPTOR != "gaba_a":
            union[f"evict_belief_{k}"]["receptor"] = EVICT_RECEPTOR
    union["loop_presence"] = _build_assembly_loop_population(pres, float(attractor_weight))
    union["arrive_presence"] = _dense_projection(ev_a, pres, W_EV_PRES, WS_LOOP_GATE)
    union["leave_inh"] = _dense_projection(ev_l, linh, W_EV_INH, WS_LOOP_GATE)
    # FLIP-FLOP (dev seed 7, 2026-10-01: a lone presence attractor silenced by leave_inh re-ignited 25-50 ms after
    # the inhibition ended). A leave also ignites `absence`; an arrival also drives `arrive_inh` onto `absence`. Each
    # attractor keeps the OTHER's inhibitory pool firing (presence -> arrive_inh, absence -> leave_inh), so the loser
    # stays down after the event: a bistable switch held by sustained cross-inhibition.
    absn = np.asarray(rm.indices("absence"), dtype=np.int64)
    ainh = np.asarray(rm.indices("arrive_inh"), dtype=np.int64)
    union["loop_absence"] = _build_assembly_loop_population(absn, float(attractor_weight))
    union["leave_absence"] = _dense_projection(ev_l, absn, W_EV_PRES, WS_LOOP_GATE)
    union["arrive_inh"] = _dense_projection(ev_a, ainh, W_EV_INH, WS_LOOP_GATE)
    union["hold_presence"] = _dense_projection(pres, ainh, W_HOLD_INH, WS_LOOP_GATE)
    union["hold_absence"] = _dense_projection(absn, linh, W_HOLD_INH, WS_LOOP_GATE)

    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)
    bridge.set_transmission_gate("witness_self", 0.0)
    bridge.set_transmission_gate("witness_other", 0.0)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    snap = _snapshot_state(bridge, xp)
    d = lambda a: xp.asarray(a)  # noqa: E731
    idx = {
        "world_dev": {k: d(v) for k, v in world_m.items()}, "belief_dev": {k: d(v) for k, v in belief_m.items()},
        "self_dev": {k: d(v) for k, v in self_m.items()}, "relay_dev": {k: d(v) for k, v in relay_m.items()},
        "world_all": d(np.concatenate([world_m[k] for k in range(K_LOC)])),
        "belief_all": d(np.concatenate([belief_m[k] for k in range(K_LOC)])),
        "self_all": d(np.concatenate([self_m[k] for k in range(K_LOC)])),
        "relay_all": d(np.concatenate([relay_m[k] for k in range(K_LOC)])) if relay_mode else None,
        "presence": d(pres), "ev_arrive": d(ev_a), "ev_leave": d(ev_l),
    }
    return bridge, xp, idx, snap


class SpikingWitnessFalseBeliefOrgan:
    """ONE live change-of-location scenario for ONE tracked agent. The caller narrates perceptual events
    (`perceive_arrive`, `perceive_leave`, `observe_event`) and reads `query()`. No method takes a `witnessed` argument:
    whether a move reaches the belief store is decided by the `presence` population's spikes at that moment."""

    def __init__(self, seed: int = _DEFAULT_SEED, *, lesion: bool = False):
        self.seed = int(seed)
        self.lesion = bool(lesion)
        self._bridge = self._xp = self._idx = self._snap = None
        self._lock = threading.Lock()
        self.n_events = 0
        self.last_write = {}

    def _isolated(self, fn):
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
            np.random.set_state(host_np)
            if host_xp is not None:
                try:
                    xp.random.get_random_state().set_state(host_xp)
                except Exception:
                    pass

    def _ensure(self):
        if self._bridge is None:
            self._bridge, self._xp, self._idx, self._snap = self._isolated(
                lambda: build_witness_bridge(self.seed, lesion=self.lesion))

    def _count(self, dev):
        return int(to_host(self._bridge.cp_firing_states[dev].astype(self._xp.float64).sum()))

    def _run(self, steps, drive, lesion, silence=False, count=()):
        """Step `steps` times with `drive` = [(dev_index_array, pA), ...] held on; return spike counts for `count` names."""
        b, xp, idx = self._bridge, self._xp, self._idx
        relay = MODE in RELAY_MODES
        lesion = self.lesion   # the lesion is a property of the bridge (fixed at construction)
        acc = {c: 0 for c in count}
        for _ in range(steps):
            b.cp_external_input_current[:] = 0.0
            for dev, pa in drive:
                b.cp_external_input_current[dev] = xp.float32(pa)
            # a constant hyperpolarizing bias on every relay cell (a cell property, the same every step): one input
            # alone stays subthreshold, so the relay is a coincidence detector
            if relay:
                b.cp_external_input_current[idx["relay_all"]] += xp.float32(RELAY_BIAS_PA)
            if relay and lesion:
                b.cp_external_input_current[idx["relay_all"]] += xp.float32(TONIC_RELAY_PA)
            if silence:
                b.cp_external_input_current[idx["presence"]] = xp.float32(SILENCE_PA)
            b._run_one_simulation_step()
            for c in count:
                acc[c] += self._count(idx[c] if isinstance(c, str) else idx["relay_dev"][c])
        return acc

    # ── perception (the host renders the senses; the substrate integrates them) ─────────────────────────────────
    def perceive_arrive(self, *, lesion: bool = False) -> None:
        """The tracked agent is perceived in the room (it placed the object, or it returned)."""
        self._ensure()
        self._isolated(lambda: self._event(self._idx["ev_arrive"], lesion))

    def perceive_leave(self, *, lesion: bool = False) -> None:
        """The tracked agent is perceived leaving."""
        self._ensure()
        self._isolated(lambda: self._event(self._idx["ev_leave"], lesion))

    def _event(self, dev, lesion):
        self._run(EV_STEPS, [(dev, EV_DRIVE_PA)], lesion)
        self._run(HOLD_STEPS, [], lesion)

    def observe_event(self, loc: int, *, lesion: bool = False, silence_presence: bool = False) -> int:
        """The object is perceived at `loc` (a placement or a move). The world store re-renders reality; the system's
        own store is written (it always witnesses); the tracked agent's belief store is written iff `relay_loc` fires,
        i.e. iff `presence` is active while `world_loc` is."""
        if not (0 <= int(loc) < K_LOC):
            raise ValueError(f"loc {loc} outside 0..{K_LOC - 1}")
        self._ensure()
        self._isolated(lambda: self._write(int(loc), bool(lesion), bool(silence_presence)))
        self.n_events += 1
        return int(loc)

    def _write(self, loc, lesion, silence):
        if MODE == "readout":
            return self._write_readout(loc, lesion, silence)
        b, idx, snap = self._bridge, self._idx, self._snap
        _FB._restore_slice(b, snap, idx["world_all"])      # the world store re-renders the perceived scene
        _FB._restore_slice(b, snap, idx["self_all"])
        b.set_transmission_gate("witness_self", 1.0)
        acc = self._run(WRITE_STEPS, [(idx["world_dev"][loc], IGNITE_PA), (idx["self_dev"][loc], IGNITE_PA * 2.0)],
                        lesion, silence=silence, count=("relay_all", "presence") + tuple(range(K_LOC)))
        b.set_transmission_gate("witness_self", 0.0)
        self._run(HOLD_STEPS, [], lesion)
        self.last_write = {"loc": loc, "relay_spikes": acc["relay_all"], "presence_spikes": acc["presence"],
                           "relay_by_loc": [acc[k] for k in range(K_LOC)]}

    def _write_readout(self, loc, lesion, silence):
        """MODE "readout": the decision is the presence population's state, read over PRES_READ_STEPS with a frozen
        threshold (PRES_MIN, spikes per neuron per step); the W3 register's validated write (clear-before-write,
        helper ignite, `witness_other` gate) is then gated by that read. `lesion` ignores the read (always written);
        `silence` clamps presence quiescent during the read (never written)."""
        b, idx, snap = self._bridge, self._idx, self._snap
        clamp = [(idx["presence"], SILENCE_PA)] if silence else []
        pres = self._run(PRES_READ_STEPS, clamp, False, count=("presence",))["presence"]
        pres_rate = pres / float(PRES_READ_STEPS * PRES_N)
        witnessed = True if self.lesion else (pres_rate >= PRES_MIN)
        _FB._restore_slice(b, snap, idx["world_all"])
        _FB._restore_slice(b, snap, idx["self_all"])
        if witnessed:
            _FB._restore_slice(b, snap, idx["belief_all"])
        b.set_transmission_gate("witness_other", 1.0 if witnessed else 0.0)
        b.set_transmission_gate("witness_self", 1.0)
        drive = [(idx["world_dev"][loc], IGNITE_PA), (idx["self_dev"][loc], HELPER_PA)]
        if witnessed:
            drive.append((idx["belief_dev"][loc], HELPER_PA))
        self._run(WRITE_STEPS, drive, False)
        b.set_transmission_gate("witness_other", 0.0)
        b.set_transmission_gate("witness_self", 0.0)
        self._run(HOLD_STEPS, [], False)
        self.last_write = {"loc": loc, "presence_rate": round(pres_rate, 4), "witnessed": bool(witnessed),
                           "relay_spikes": 0, "presence_spikes": pres, "relay_by_loc": [0] * K_LOC}

    def idle(self, steps: int = 200) -> None:
        """Time passes with nothing perceived (the persistence test: presence/absence must hold their state)."""
        self._ensure()
        self._isolated(lambda: self._run(int(steps), [], False))

    # ── read ─────────────────────────────────────────────────────────────────────────────────────────────────────
    def presence_rate(self, steps: int = 30) -> float:
        self._ensure()
        acc = self._isolated(lambda: self._run(steps, [], False, count=("presence",)))
        return acc["presence"] / float(steps * PRES_N)

    def query(self, *, lesion: bool = False) -> dict:
        self._ensure()
        return self._isolated(lambda: self._query(lesion))

    def _query(self, lesion):
        b, idx = self._bridge, self._idx
        free = int(_FB.FREE_STEPS)
        late = free - max(1, free // 3)
        acc = {s: {k: 0 for k in range(K_LOC)} for s in ("world", "belief", "self")}
        pres = 0
        query_bias = QUERY_RELAY_BIAS and MODE in RELAY_MODES
        for t in range(free):
            b.cp_external_input_current[:] = 0.0
            if query_bias:
                # the relay's bias is a cell property (see `_run`); without it the relays fire on presence alone
                # during the query and the off-target eviction pools silence the written slot (relay2 review)
                b.cp_external_input_current[idx["relay_all"]] += self._xp.float32(RELAY_BIAS_PA)
            b._run_one_simulation_step()
            if t >= late:
                for s in acc:
                    for k in range(K_LOC):
                        acc[s][k] += self._count(idx[f"{s}_dev"][k])
                pres += self._count(idx["presence"])
        n = float(free - late) * float(STORE)
        rates = {s: {k: acc[s][k] / n for k in range(K_LOC)} for s in acc}
        return {"world_loc": _FB._argmax_loc(rates["world"]), "belief_loc": _FB._argmax_loc(rates["belief"]),
                "self_loc": _FB._argmax_loc(rates["self"]), "world_rates": rates["world"],
                "belief_rates": rates["belief"], "self_rates": rates["self"],
                "presence_rate": pres / (float(free - late) * PRES_N), "n_events": int(self.n_events),
                "lesioned": bool(self.lesion)}


# ── the frozen operating-point grid (prereg section 3). Relay points first (the fully synaptic write is preferred);
#    readout points last. Dev seeds 1-8 (Claude, 2026-10-01, numpy, T13 guard on, the 10-case battery): GRID 4, 5 and 6
#    passed every case on every dev seed; GRID 7 (GABA_B leave receptor) failed on every dev seed; GRID 0 failed on
#    dev seeds 2-6 and 8 under the earlier 6-case battery (the relay is a working coincidence detector on dev seed 7 --
#    an unwitnessed move stays at the one-spike onset floor, 40 spikes, a witnessed one reaches 58-65 -- but its drive
#    did not ignite belief_k). Relay points 1-3 were not run. The pool sweep (HANDOFF step 1) decides; nothing here is
#    a gate result. ─────────────────────────────────────────────────────────────────────────────────────────────────
_SHARED = dict(W_EV_PRES=60.0, W_EV_INH=60.0, W_INH_PRES=100.0, LEAVE_RECEPTOR="gaba_a", BOUNDED_CONDUCTANCE=True)
GRID = [
    dict(_SHARED, MODE="relay", RELAY_BIAS_PA=-150.0, W_WORLD_RELAY=1.5, W_PRES_RELAY=1.5, W_RELAY_BELIEF=48.0,
         W_RELAY_EVICT=8.0, W_EVICT_BELIEF=100.0),
    dict(_SHARED, MODE="relay", RELAY_BIAS_PA=-100.0, W_WORLD_RELAY=2.5, W_PRES_RELAY=2.5, W_RELAY_BELIEF=48.0,
         W_RELAY_EVICT=8.0, W_EVICT_BELIEF=100.0),
    dict(_SHARED, MODE="relay", RELAY_BIAS_PA=-100.0, W_WORLD_RELAY=3.0, W_PRES_RELAY=3.0, W_RELAY_BELIEF=120.0,
         W_RELAY_EVICT=16.0, W_EVICT_BELIEF=200.0),
    dict(_SHARED, MODE="relay", RELAY_BIAS_PA=-300.0, W_WORLD_RELAY=5.0, W_PRES_RELAY=5.0, W_RELAY_BELIEF=120.0,
         W_RELAY_EVICT=16.0, W_EVICT_BELIEF=200.0),
    dict(_SHARED, MODE="readout", PRES_MIN=0.03, PRES_READ_STEPS=20),
    dict(_SHARED, MODE="readout", PRES_MIN=0.02, PRES_READ_STEPS=30),
    dict(_SHARED, MODE="readout", PRES_MIN=0.03, PRES_READ_STEPS=20, W_EV_PRES=40.0),
    dict(_SHARED, MODE="readout", PRES_MIN=0.03, PRES_READ_STEPS=20, LEAVE_RECEPTOR="gaba_b"),
]
_BASE_PARAMS = {k: globals()[k] for k in sorted({k for g in GRID for k in g})}


def set_operating_point(i: int) -> dict:
    """Copy GRID[i] into the module constants (every key of every grid point is reset first). Returns the params."""
    g = globals()
    for k, v in _BASE_PARAMS.items():
        g[k] = v
    for k, v in GRID[int(i)].items():
        g[k] = v
    return dict(GRID[int(i)])


# ── the organ battery (prereg section 4; scored by tools/s2p_score.py O) ───────────────────────────────────────────
CASES = [
    # name, events, kwargs, expected belief slot (A = arrive; L = leave; W = idle IDLE_STEPS; digits = object perceived
    # at that location: the first is the placement)
    ("false_belief", ["A", "0", "L", "1"], {}, 0),
    ("true_belief", ["A", "0", "1"], {}, 1),
    ("return_then_move", ["A", "0", "L", "A", "1"], {}, 1),
    ("two_moves_one_seen", ["A", "0", "L", "1", "A", "2"], {}, 2),
    ("two_unseen_moves", ["A", "0", "L", "1", "2"], {}, 0),
    ("absent_after_wait", ["A", "0", "L", "W", "1"], {}, 0),
    ("present_after_wait", ["A", "0", "W", "1"], {}, 1),
    ("return_after_wait", ["A", "0", "L", "W", "A", "W", "1"], {}, 1),
    ("lesion_false_belief", ["A", "0", "L", "1"], {"lesion": True}, 1),
    ("silence_true_belief", ["A", "0", "1"], {"silence": True}, 0),
]
IDLE_STEPS = 300      # "W": 300 ms with nothing perceived
BELIEF_MIN = 0.02     # the answered slot's late-window rate must reach this (an ignited member, not noise) ...
BELIEF_MARGIN = 2.0   # ... and be at least this multiple of every other slot's rate


def run_case(seed: int, events, lesion: bool = False, silence: bool = False) -> dict:
    o = SpikingWitnessFalseBeliefOrgan(seed, lesion=lesion)
    writes = []
    for e in events:
        if e == "A":
            o.perceive_arrive(lesion=lesion)
        elif e == "L":
            o.perceive_leave(lesion=lesion)
        elif e == "W":
            o.idle(IDLE_STEPS)
        else:
            # the placing write (first digit) is never silenced: the diagnostic clamps presence on later moves only
            o.observe_event(int(e), lesion=lesion, silence_presence=silence and len(writes) > 0)
            writes.append(dict(o.last_write))
    q = o.query(lesion=lesion)
    return {"belief_loc": q["belief_loc"], "world_loc": q["world_loc"],
            "belief_rates": {str(k): round(v, 5) for k, v in q["belief_rates"].items()},
            "presence_rate": round(q["presence_rate"], 5), "writes": writes}


def store_read_ok(rates: dict, slot: int) -> bool:
    r = {int(k): float(v) for k, v in rates.items()}
    top = r.get(int(slot), 0.0)
    rest = [v for k, v in r.items() if k != int(slot)]
    return top >= BELIEF_MIN and all(top >= BELIEF_MARGIN * v for v in rest)


def organ_battery(seed: int, op: int) -> dict:
    params = set_operating_point(op)
    cases = {}
    for name, events, kw, want in CASES:
        c = run_case(seed, events, **kw)
        c["expected_slot"] = want
        c["pass"] = bool(c["belief_loc"] == want and store_read_ok(c["belief_rates"], want))
        cases[name] = c
    return {"organ": "tom_spiking_witness", "seed": int(seed), "op_index": int(op), "params": params,
            "mode": params.get("MODE"), "cases": cases, "all_pass": all(c["pass"] for c in cases.values()),
            "backend": os.environ.get("SIM_BACKEND", "")}


GATE_SEEDS = (42, 43, 44, 100, 101, 102)


def main(argv=None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--op", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--sweep", action="store_true", help="every GRID point x --seeds, one file per (op, seed)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[])
    ap.add_argument("--out-dir")
    ap.add_argument("--allow-gate-seed", action="store_true")
    ap.add_argument("--prereg")
    a = ap.parse_args(argv)
    seeds = a.seeds if a.sweep else [a.seed]
    if any(s is None for s in seeds) or not seeds:
        ap.error("--seed (or --sweep --seeds ...) is required")
    if any(s in GATE_SEEDS for s in seeds):
        if not (a.allow_gate_seed and a.prereg and os.path.isfile(a.prereg)):
            print("GATE seed refused: pass --allow-gate-seed --prereg <committed prereg path>", file=sys.stderr)
            return 2
        if a.sweep:
            print("GATE seeds never run in --sweep (dev seeds only)", file=sys.stderr)
            return 2
    if a.sweep:
        os.makedirs(a.out_dir, exist_ok=True)
        for op in range(len(GRID)):
            for s in seeds:
                res = organ_battery(s, op)
                with open(os.path.join(a.out_dir, "op%d_s%d.json" % (op, s)), "w") as f:
                    json.dump(res, f, indent=1, sort_keys=True)
                print("op%d s%d all_pass=%s" % (op, s, res["all_pass"]), flush=True)
        return 0
    res = organ_battery(a.seed, a.op)
    if a.prereg:
        res["prereg"] = a.prereg
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("seed %d op %d all_pass=%s" % (a.seed, a.op, res["all_pass"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
