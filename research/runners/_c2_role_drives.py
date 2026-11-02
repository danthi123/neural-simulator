"""C2 match-unit smoke (RULING c2-drain-wta-1101): the agent/action drive sources and the row-shuffle lesion the composer's
sequencer may use, all research-only and default OFF.

The C2 match unit m{b} only sees its block's decoded agent/action line drives (`_seq_drives`, made by `make_block_drives`
from each block's cleanup scores through the divnorm score bridge). On the gate-reset wave the divnorm relay lit 4-7 words
per agent/action row, so a cue's two words were often both lit in a WRONG block (86 both-lit false fires). Two spiking
competition sources for those rows, and one lesion:

- `BRAIN_C2_ROLE_WTA=1`: `make_block_drives(retreat="wta")` on the existing, never-run NEF-FS lateral-inhibition WTA score
  bridge (`build_wta_score_bridge` / `wta_drive`, _phaseB_onebrain_sequencerK_k32_margin_derisk.py; finding
  2026-06-20-shortcut3-sequencer-deployment-scoping.md R1). Plumbing added here: the bridge is built lazily per composer
  (`_seq_score_wta`) with the composer's sigma/gain and the word pools' per-neuron parameters copied from `_seq_score`
  (as the patient bridge does), and each role read starts from a full synaptic reset (`wta_drive(full_reset=True)`).
  `BRAIN_C2_ROLE_WTA_I2E=0` is its lesion (same `wta_inh` population and synapses, zero I->E).
- `BRAIN_C2_ROLE_RELAY=1`: the patient-relay inhibition bridge generalized to the agent/action slices
  (`_c2_patient_inhib.composer_role_relay`; lesion `BRAIN_C2_ROLE_RELAY_I2E=0`).
- `BRAIN_C2_ROW_SHUFFLE=1` (lesion): block b is fed block perm[b]'s agent/action row, perm a fixed derangement drawn from
  the composer seed. The patient drives are not shuffled.

Unset: `role_drive_mode()` is None, `row_shuffle_perm` is None, and `_ensure_sequencer` runs the unchanged divnorm call.
No host argmax / top-k / threshold anywhere: which word lines are lit is decided by which word pools spike.
"""
from __future__ import annotations

import os

import numpy as np

ROLE_WTA_ENV = "BRAIN_C2_ROLE_WTA"
ROLE_WTA_I2E_ENV = "BRAIN_C2_ROLE_WTA_I2E"
ROW_SHUFFLE_ENV = "BRAIN_C2_ROW_SHUFFLE"
# the k32-margin builder's own defaults (build_wta_score_bridge / wta_drive), recorded with every WTA read
WTA_KNOBS = {"n_word": 20, "n_inh": 20, "w_exc_to_inh": 8.0, "w_inh_to_exc": 40.0, "settle": 40}
_ON = ("1", "true", "yes", "on")


def _flag(name) -> bool:
    v = os.environ.get(name)
    return v is not None and v.strip().lower() in _ON


def role_wta_enabled() -> bool:
    return _flag(ROLE_WTA_ENV)


def row_shuffle_enabled() -> bool:
    return _flag(ROW_SHUFFLE_ENV)


def role_wta_w_i2e() -> float:
    v = os.environ.get(ROLE_WTA_I2E_ENV)
    return float(WTA_KNOBS["w_inh_to_exc"]) if v in (None, "") else float(v)


def role_drive_mode():
    """None (the unchanged divnorm relay), "wta" or "relay". Both flags set is refused (the arms are separate)."""
    from research.runners._c2_patient_inhib import composer_role_relay_enabled
    w, r = role_wta_enabled(), composer_role_relay_enabled()
    if w and r:
        raise ValueError("BRAIN_C2_ROLE_WTA and BRAIN_C2_ROLE_RELAY are separate arms; set at most one")
    return "wta" if w else ("relay" if r else None)


def flags_record():
    """Every C2 drain/WTA env knob as set in this process (None = unset), for the shard's engagement record."""
    from research.runners._c2_patient_inhib import ROLE_RELAY_ENV, ROLE_RELAY_I2E_ENV
    from research.runners._phaseB_onebrain_sequencerK_derisk import QUIET_DRAIN_ENV
    names = (QUIET_DRAIN_ENV, ROLE_WTA_ENV, ROLE_WTA_I2E_ENV, ROLE_RELAY_ENV, ROLE_RELAY_I2E_ENV, ROW_SHUFFLE_ENV,
             "BRAIN_C2_GATE_RESET", "BRAIN_C2_COMPOSER_PATIENT_INHIB")
    return {n: os.environ.get(n) for n in names}


def row_shuffle_perm(seed, K):
    """The fixed derangement for `BRAIN_C2_ROW_SHUFFLE` (None when the flag is off or K < 2): block b reads row perm[b],
    perm[b] != b for every b. Drawn from RandomState(seed + 104729) by rejection, so it depends only on (seed, K)."""
    if not row_shuffle_enabled() or int(K) < 2:
        return None
    rng = np.random.RandomState(int(seed) + 104729)
    while True:
        p = rng.permutation(int(K))
        if not np.any(p == np.arange(int(K))):
            return [int(x) for x in p]


def role_wta_bridge(buf, w_i2e=None):
    """The composer's role-WTA score bridge (lazy, cached on `buf._seq_score_wta` keyed by (seed, V, sigma, gain, w_i2e)):
    `build_wta_score_bridge` at the composer's sigma/gain and the builder's own knobs, with the word pools' per-neuron
    Izhikevich parameters copied from `buf._seq_score` (the composer's divnorm word pools, same n_word) so only the
    `wta_inh` population and its synapses differ from the divnorm relay."""
    from research.runners._phaseB_onebrain_sequencerK_k32_margin_derisk import build_wta_score_bridge
    from sim.backend import to_host, from_host
    w = role_wta_w_i2e() if w_i2e is None else float(w_i2e)
    key = (int(buf.seed), int(buf.V), float(buf.sequencer_sigma), float(buf.sequencer_gain), w)
    cache = getattr(buf, "_seq_score_wta", None)
    if cache is None:
        cache = {}
        buf._seq_score_wta = cache
    if key not in cache:
        k = WTA_KNOBS
        sb = build_wta_score_bridge(buf.seed, int(buf.V), n_word=k["n_word"], n_inh=k["n_inh"], sigma=buf.sequencer_sigma,
                                    gain=buf.sequencer_gain, w_exc_to_inh=k["w_exc_to_inh"], w_inh_to_exc=w)
        ref = getattr(buf, "_seq_score", None)
        if ref is not None:
            nw = int(buf.V) * int(k["n_word"])
            for attr in [a for a in vars(ref) if a.startswith("cp_izh_")]:
                src, dst = getattr(ref, attr, None), getattr(sb, attr, None)
                if src is None or dst is None or not hasattr(dst, "shape") or len(dst.shape) != 1:
                    continue
                arr = np.array(to_host(dst), copy=True)
                arr[:nw] = np.asarray(to_host(src))[:nw]
                setattr(sb, attr, from_host(arr))
        cache[key] = sb
    return cache[key]


def role_block_drives(buf, fns, bscores):
    """The agent/action drives for the flagged role mode: (drives[(dA, dX)], accs[(accA, accX)], info). Mode "wta":
    `make_block_drives(retreat="wta")` on the role-WTA bridge; mode "relay": `composer_role_relay` per role slice."""
    mode = role_drive_mode()
    accs = []
    if mode == "wta":
        stats = {}
        w = role_wta_w_i2e()
        drives, _lit = fns["make_block_drives"](role_wta_bridge(buf, w), buf.V, bscores, input_gain=buf.sequencer_input_gain,
                                                retreat="wta", peak_mult=1.0, accs_out=accs,
                                                drive_kw={"stats": stats, "full_reset": True})
        info = dict(mode="wta", w_i2e=float(w), knobs=dict(WTA_KNOBS, w_inh_to_exc=float(w)),
                    inh_spikes=float(stats.get("inh_spikes", 0.0)), reads=int(stats.get("reads", 0)),
                    inh_ceiling=float(WTA_KNOBS["n_inh"] * WTA_KNOBS["settle"] * int(stats.get("reads", 0))))
        return drives, accs, info
    if mode == "relay":
        from research.runners._c2_patient_inhib import COMPOSER_KNOBS, composer_role_relay
        drives, inh, per = [], 0.0, []
        for (ag, ax) in bscores:
            dA, accA, iA = composer_role_relay(buf, ag)
            dX, accX, iX = composer_role_relay(buf, ax)
            drives.append((dA, dX))
            accs.append((accA, accX))
            inh += iA["inh_spikes"] + iX["inh_spikes"]
            per.append([iA["inh_spikes"], iX["inh_spikes"]])
        info = dict(mode="relay", w_i2e=float(iA["w_i2e"]) if bscores else None, knobs=dict(COMPOSER_KNOBS),
                    inh_spikes=float(inh), inh_spikes_per_block=per, reads=2 * len(bscores),
                    inh_ceiling=float(COMPOSER_KNOBS["n_inh"] * COMPOSER_KNOBS["settle"] * 2 * len(bscores)))
        return drives, accs, info
    raise ValueError("role_block_drives called with no role mode set")
