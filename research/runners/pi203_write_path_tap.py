"""Vikunja #203 follow-up (pi lane): WHICH write path moves the comprehension organ's frozen synapses?

The merged fix (main `dc6abf989`, `cfg.enforce_plastic_mask_beyond_hebbian`, umbrella env
`BRAIN_ENFORCE_PLASTIC_MASK`) gates six write sites to `cp_connections.data`, and the reply-level arm showed the
flag changes the production comprehension organ's reply ZERO times -- yet the original audit measured a real
13.8 -> 56.1 drift on THIS organ (research/findings/raw/_read_isolation_audit_29/audit_29runners.json), and the
1710/1724 synapses independently observed to change weight are the internal recurrent `sel_* <-> sel_FS_*`
pathways declared `plastic_internal=False`.

This instrument does not assume an answer. It records EVERY write to the organ's `cp_connections.data` during the
production battery, attributed to the exact `sim/bridge.py` line that performed it, and for each write: how many
synapses it touched, the summed and max |delta|, and how much of that delta landed on synapses the
`cp_synapse_plastic_mask` calls FROZEN at that instant (the mask is positional, so frozen-ness is evaluated at the
moment of the write, not after a re-ordering). It also snapshots every other weight-shaped store the organ could
read (rate gain, transmission gain, eligibility trace, coactivity trace, NMDA/leak conductances, the homeostatic
pair), counts every step in which the CSR data array was REPLACED (a rebuild/compaction site, which no in-place
tap can see), and reports each synapse's end-to-end drift by its (pre,post) COORDINATE so a re-ordering cannot
fake a zero.

Instrument, not computation: the tap is a numpy ndarray SUBCLASS wrapping the existing array. It records
(file:line, counts, deltas) and writes nothing back into the simulation. Its non-invasiveness is checked, not
assumed: the tapped battery's reply digests are compared with the same battery run through the untapped
production path (`research.runners._comprehension_learned_cues_joint_arm`), flag-for-flag.

Run (one seed, one arm, fresh process):
    CUDA_VISIBLE_DEVICES= SIM_BACKEND=numpy OMP_NUM_THREADS=1 PYTHONPATH=. python -m research.runners.pi203_write_path_tap \
        --seed 7 --arm on --out /tmp/pi203_tap_s7_on.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

import tools.lab

# Other weight-shaped stores the organ could read, snapshot before/after the battery.
STORES = (
    "cp_plasticity_rate_gain", "cp_transmission_gain", "cp_synapse_plastic_mask",
    "cp_eligibility_trace", "cp_hebb_coactivity_trace",
    "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise", "cp_conductance_g_nmda_recurrent",
    "cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab",
    "cp_neuron_firing_thresholds", "cp_neuron_activity_ema",
)

_LOG: list[dict] = []
_REBUILDS: list[int] = [0]
_MASK_BY_ARRAY: dict[int, object] = {}   # id(tapped array) -> the bridge whose mask governs it
_BRIDGE_BY_CSR: dict[int, object] = {}   # id(csr object) -> the bridge it belongs to
_INSTALLED_BRIDGES: set = set()          # bridges the tap is installed on
_CURRENT_BRIDGE = None                   # the organ's bridge in this process (fallback)
_KEEP_ALIVE: list = []                   # keep every tapped array referenced, so ids are never reused
_STEPS = 0
_ARRAY_REPLACED = 0
_BLIND: set = set()          # bridges whose weight buffer is NOT a numpy array (the cupy path): the tap can
                             # see coordinate drift but cannot hook individual writes. Blindness is reported, not
                             # hidden -- a tap that silently records nothing on the GPU looks identical to a run
                             # where nothing was written.


_SELF_FILE = os.path.basename(__file__)


def _caller_frame():
    """The first stack frame OUTSIDE this instrument: the sim line that performed the write."""
    f = sys._getframe(1)
    while f is not None and os.path.basename(f.f_code.co_filename) == _SELF_FILE:
        f = f.f_back
    return f


class TapArray(np.ndarray):
    """numpy ndarray subclass that records every in-place write and the line that performed it."""

    def __new__(cls, arr):
        return np.asarray(arr).view(cls)

    def _record(self, key, old, new, frame):
        oldf = np.asarray(old, dtype=np.float64)
        newf = np.asarray(new, dtype=np.float64)
        signed = newf - oldf
        delta = np.abs(signed)
        touched = int(delta.size)
        if touched == 0 or float(delta.max()) == 0.0:
            return
        bridge = getattr(self, "_tap_bridge", None) or _MASK_BY_ARRAY.get(id(self)) \
            or _MASK_BY_ARRAY.get(id(getattr(self, "base", None))) or _CURRENT_BRIDGE
        mask = _mask(bridge) if bridge is not None else None
        frozen_n, frozen_sum, frozen_max, frozen_pos, frozen_neg = 0, 0.0, 0.0, 0.0, 0.0
        if mask is not None:
            m = mask
            pos = np.arange(old.size) if isinstance(key, slice) else np.asarray(key).ravel()
            live = pos[pos < m.size]
            if live.size:
                frz = ~m[live]
                fs = signed[pos < m.size][frz]
                fd = np.abs(fs)
                frozen_n = int(fd.size)
                frozen_sum = float(fd.sum())
                frozen_max = float(fd.max()) if fd.size else 0.0
                frozen_pos = float(fs[fs > 0].sum())
                frozen_neg = float(-fs[fs < 0].sum())
        _LOG.append({
            "site": "%s:%d" % (os.path.basename(frame.f_code.co_filename), frame.f_lineno),
            "function": frame.f_code.co_name,
            "touched": touched,
            "sum_abs_delta": float(delta.sum()),
            "max_abs_delta": float(delta.max()),
            "frozen_writes": 1 if frozen_n else 0,
            "frozen_touched": frozen_n,
            "frozen_sum_abs_delta": frozen_sum,
            "frozen_max_abs_delta": frozen_max,
            "frozen_potentiation": frozen_pos,
            "frozen_depression": frozen_neg,
        })

    def __getitem__(self, key):
        out = np.ndarray.__getitem__(self, key)
        # A fancy-index read returns a COPY. Tag it: its in-place mutation is written back by the parent's
        # __setitem__, so recording it here as well would count every such write twice.
        if isinstance(out, TapArray):
            try:
                out._tap_temp = True
            except AttributeError:
                pass
        return out

    def __setitem__(self, key, value):
        frame = _caller_frame()
        old = np.asarray(self)[key]
        np.ndarray.__setitem__(self, key, value)
        self._record(key, old, np.asarray(self)[key], frame)

    def _inplace(self, op, value, frame):
        old = np.asarray(self).copy()
        getattr(np.ndarray, op)(self, value)
        if not getattr(self, "_tap_temp", False):
            self._record(slice(None), old, np.asarray(self), frame)
        return self

    def __iadd__(self, value):
        return self._inplace("__iadd__", value, _caller_frame())

    def __imul__(self, value):
        return self._inplace("__imul__", value, _caller_frame())

    def __isub__(self, value):
        return self._inplace("__isub__", value, _caller_frame())


def _npify(x):
    """cupy -> numpy without the implicit-conversion error; numpy passes through unchanged."""
    if hasattr(x, "get") and not isinstance(x, np.ndarray):
        return np.asarray(x.get())
    return np.asarray(x)


def _make_tap(arr, csr, bridge):
    if not isinstance(arr, np.ndarray):
        return arr          # cupy buffer: TapArray is a numpy subclass, it cannot wrap this
    tap = arr if isinstance(arr, TapArray) else TapArray(arr)
    try:
        # carry the bridge on the array itself: object identity of a numpy view is not stable
        # across scipy copies and re-views, so an attribute travels with the array
        tap._tap_bridge = bridge
        tap._tap_temp = False      # whatever lands in csr.data is the LIVE array, never a temp copy
    except AttributeError:
        pass
    _MASK_BY_ARRAY[id(tap)] = bridge
    _BRIDGE_BY_CSR[id(csr)] = bridge
    _KEEP_ALIVE.append(tap)
    return tap


def _record_replacement(old, new, csr, frame):
    """A whole-array replacement (`csr.data = <new array>`) is a WRITE too: measure it like one.
    A brand-new array (old is None) is not a change to existing weights; the coordinate snapshots cover it."""
    global _ARRAY_REPLACED
    _ARRAY_REPLACED += 1
    if old is None:
        return
    # a genuine CSR rebuild re-orders synapses: positions no longer identify the same synapse, so the
    # positional mask labels the wrong weights. That is defect (v), and it is what makes the tap's
    # position-based frozen split disagree with the coordinate-based drift.
    if getattr(old, "indptr", None) is not None and getattr(new, "indptr", None) is not None and (
            old.indptr.shape != new.indptr.shape or old.indices.shape != new.indices.shape):
        _REBUILDS[0] += 1
    bridge = _BRIDGE_BY_CSR.get(id(csr)) or _CURRENT_BRIDGE
    a, b = _npify(old).astype(np.float64), _npify(new).astype(np.float64)
    if a.shape != b.shape:
        _LOG.append({"site": "%s:%d" % (os.path.basename(frame.f_code.co_filename), frame.f_lineno),
                     "function": frame.f_code.co_name, "touched": 0, "sum_abs_delta": 0.0,
                     "max_abs_delta": 0.0, "frozen_writes": 0, "frozen_touched": 0,
                     "frozen_sum_abs_delta": 0.0, "frozen_max_abs_delta": 0.0,
                     "note": "data array resized %d -> %d (rebuild / compaction / formation)" % (a.size, b.size)})
        return
    _record_new(a, b, csr, bridge, frame, replacement=True)


def _record_new(old, new, csr, bridge, frame, replacement=False):
    signed = new - old
    delta = np.abs(signed)
    if delta.size == 0 or float(delta.max()) == 0.0:
        return
    mask = _mask(bridge) if bridge is not None else None
    frozen_n, frozen_sum, frozen_max, frozen_pos, frozen_neg = 0, 0.0, 0.0, 0.0, 0.0
    if mask is not None:
        m = np.asarray(mask)
        live = min(delta.size, m.size)
        frz = ~m[:live]
        fs = signed[:live][frz]
        fd = np.abs(fs)
        frozen_n, frozen_sum = int(fd.size), float(fd.sum())
        frozen_max = float(fd.max()) if fd.size else 0.0
        frozen_pos = float(fs[fs > 0].sum())
        frozen_neg = float(-fs[fs < 0].sum())
    rec = {"site": "%s:%d" % (os.path.basename(frame.f_code.co_filename), frame.f_lineno),
           "function": frame.f_code.co_name, "touched": int(delta.size),
           "sum_abs_delta": float(delta.sum()), "max_abs_delta": float(delta.max()),
           "frozen_writes": 1 if frozen_n else 0, "frozen_touched": frozen_n,
           "frozen_sum_abs_delta": frozen_sum, "frozen_max_abs_delta": frozen_max,
           "frozen_potentiation": frozen_pos, "frozen_depression": frozen_neg}
    if replacement:
        rec["note"] = "whole-array replacement: positions may have re-ordered, so the frozen split is positional"
    _LOG.append(rec)


def install_tap(bridge) -> None:
    """Wrap the organ's live `cp_connections.data`, and hook EVERY way that array can change:
      1. TapArray.__setitem__ / __iadd__ / __imul__ / __isub__  -- in-place writes,
      2. TapCSR.__setattr__                                     -- `csr.data = <new array>` (e.g. set_pathway_weights,
         sim/bridge.py:5766), which no in-place tap can see,
      3. SimulationBridge.__setattr__ hook                      -- `self.cp_connections = <new csr>` (formation,
         CSR rebuild, compaction).
    Every hook RECORDS; none writes back into the simulation. Non-invasiveness is measured (tapped vs untapped
    reply digests in the orchestrator), not assumed."""
    _INSTALLED_BRIDGES.add(id(bridge))
    global _CURRENT_BRIDGE, _STEPS, _ARRAY_REPLACED
    _LOG.clear()
    _REBUILDS[0] = 0
    _STEPS = 0
    _ARRAY_REPLACED = 0
    _CURRENT_BRIDGE = bridge
    BaseCSR = type(bridge.cp_connections)

    class TapCSR(BaseCSR):
        def __setattr__(self, name, value):
            if name == "data" and isinstance(value, np.ndarray):
                if isinstance(value, TapArray):
                    # `csr.data = <array read out of the live buffer, mutated, written back>`: the in-place
                    # mutation happened on a TEMP copy (suppressed by _tap_temp so the parent __setitem__
                    # does not double-count it), so the write-back IS the write. Record it, then untag.
                    old = getattr(self, "data", None)
                    if old is not None:
                        _record_replacement(old, value, self, _caller_frame())
                    try:
                        value._tap_temp = False
                    except AttributeError:
                        pass
                else:
                    old = getattr(self, "data", None)
                    _record_replacement(old, value, self, _caller_frame())
                    value = _make_tap(value, self, _BRIDGE_BY_CSR.get(id(self)) or _CURRENT_BRIDGE)
            object.__setattr__(self, name, value)

    def wrap(b, csr=None):
        global _ARRAY_REPLACED      # without this the += makes it LOCAL to wrap: UnboundLocalError the moment
                                    # the branch is reached -- which is exactly what happened on the cupy run
                                    # (research/findings/raw/_pi203_frozen_synapse_residual/cupy_frozen_flat_pytest.txt,
                                    # 5 failed), because on cupy the construction hook never pre-wraps the data.
        csr = csr if csr is not None else b.cp_connections
        if not isinstance(csr, TapCSR):
            csr = TapCSR(csr)
            b.__dict__["cp_connections"] = csr
        _BRIDGE_BY_CSR[id(csr)] = b
        if not isinstance(csr.data, TapArray):
            _ARRAY_REPLACED += 1
            tapped = _make_tap(csr.data, csr, b)
            if not isinstance(tapped, TapArray):
                _BLIND.add(id(b))
                _LOG.append({"site": "instrument", "function": "install_tap", "touched": 0,
                             "sum_abs_delta": 0.0, "max_abs_delta": 0.0, "frozen_writes": 0,
                             "frozen_touched": 0, "frozen_sum_abs_delta": 0.0, "frozen_max_abs_delta": 0.0,
                             "note": "cp_connections.data is a %s buffer, not a numpy array: the per-write tap "
                                     "cannot hook it. Coordinate drift is still measured; per-site write "
                                     "attribution is NOT, and any zero in the frozen column here is the "
                                     "coordinate measurement, not the tap."
                                     % type(csr.data).__module__})
            b.__dict__["cp_connections"].data = tapped

    wrap(bridge)

    from sim.bridge import SimulationBridge
    orig_setattr = SimulationBridge.__setattr__

    def hooked_setattr(self, name, value):
        orig_setattr(self, name, value)
        if id(self) in _INSTALLED_BRIDGES and name == "cp_connections" \
                and getattr(value, "format", "") == "csr" and not isinstance(value, TapCSR):
            wrap(self, value)

    SimulationBridge.__setattr__ = hooked_setattr

    orig = SimulationBridge._run_one_simulation_step

    def stepped(self):
        global _STEPS
        _STEPS += 1
        if id(self) in _INSTALLED_BRIDGES:
            wrap(self)
        return orig(self)

    SimulationBridge._run_one_simulation_step = stepped


def _region_attribution(bridge, frozen_coords, coords_pre, coords_end):
    """Name the populations the frozen-drift synapses belong to, so the finding says WHICH synapses moved."""
    rm = getattr(bridge, "region_manager", None)
    if rm is None or not frozen_coords:
        return {}
    try:
        idx2region = {}
        for name, idxs in rm.region_indices_dict().items():
            for i in idxs:
                idx2region[int(i)] = name
    except Exception:
        return {}
    pairs = {}
    for (pre, post) in frozen_coords:
        w0, w1 = coords_pre.get((pre, post)), coords_end.get((pre, post))
        if w0 is None or w1 is None or abs(w1 - w0) <= 1e-9:
            continue
        key = (idx2region.get(pre, "?"), idx2region.get(post, "?"))
        rec = pairs.setdefault(key, {"synapses": 0, "max_abs_drift": 0.0})
        rec["synapses"] += 1
        rec["max_abs_drift"] = max(rec["max_abs_drift"], abs(w1 - w0))
    return {"%s -> %s" % k: v for k, v in sorted(pairs.items(), key=lambda kv: -kv[1]["synapses"])}


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _coords(bridge) -> dict:
    """(pre, post) coordinate -> weight. Identity-based, so a CSR re-ordering cannot hide a drift."""
    coo = bridge.cp_connections.tocoo()
    return {(int(r), int(c)): float(v) for r, c, v in zip(coo.row, coo.col, coo.data)}


def _mask(bridge):
    m = getattr(bridge, "cp_synapse_plastic_mask", None)
    return None if m is None else _npify(m)


def tap_is_blind(bridge) -> bool:
    """True when this bridge's weight buffer cannot be hooked by the numpy-subclass tap (the cupy path)."""
    return id(bridge) in _BLIND


def _mask_label_by_coord(bridge) -> dict:
    """coordinate -> is_plastic as the mask CURRENTLY labels it. Comparing this before and after a run measures
    the misalignment defect (v) directly: a CSR rebuild re-orders synapses and the mask is not remapped
    unless the fix is on, so the labels then describe the wrong weights."""
    m = getattr(bridge, "cp_synapse_plastic_mask", None)
    mask = _npify(m) if m is not None else np.zeros(0, dtype=bool)
    coo = bridge.cp_connections.tocoo()
    return {(int(r), int(c)): bool(mask[k]) if k < mask.size else False
            for k, (r, c) in enumerate(zip(coo.row, coo.col))}


def _snapshot(bridge, name):
    arr = getattr(bridge, name, None)
    return None if arr is None else np.asarray(arr).copy()


def _aggregate(log):
    sites = {}
    for rec in log:
        s = sites.setdefault(rec["site"], {
            "site": rec["site"], "function": rec["function"], "writes": 0, "touched": 0,
            "sum_abs_delta": 0.0, "max_abs_delta": 0.0, "writes_touching_frozen": 0,
            "frozen_touched": 0, "frozen_sum_abs_delta": 0.0, "frozen_max_abs_delta": 0.0,
            "frozen_potentiation": 0.0, "frozen_depression": 0.0,
        })
        s["writes"] += 1
        s["touched"] += rec["touched"]
        s["sum_abs_delta"] += rec["sum_abs_delta"]
        s["max_abs_delta"] = max(s["max_abs_delta"], rec["max_abs_delta"])
        s["writes_touching_frozen"] += rec["frozen_writes"]
        s["frozen_touched"] += rec["frozen_touched"]
        s["frozen_sum_abs_delta"] += rec["frozen_sum_abs_delta"]
        s["frozen_max_abs_delta"] = max(s["frozen_max_abs_delta"], rec["frozen_max_abs_delta"])
        s["frozen_potentiation"] += rec["frozen_potentiation"]
        s["frozen_depression"] += rec["frozen_depression"]
    return sorted(sites.values(), key=lambda s: (-s["frozen_sum_abs_delta"], -s["sum_abs_delta"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--arm", choices=["on", "off"], default="on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", type=int, default=0, help="limit battery items (0 = all)")
    ap.add_argument("--organ", choices=["pooled", "standalone"], default="standalone",
                    help="pooled = get_organ() (the wave3 merged pool, what the reply arm used); "
                         "standalone = ComprehensionProductionOrgan(seed) (the bridge the 2026-09-02 audit measured)")
    ap.add_argument("--tap", choices=["on", "off"], default="on", help="off = the untapped control run")
    args = ap.parse_args()

    # EXPLICIT flag states, matching the reply-arm condition used in pi203_comprehension_no_regression.py.
    os.environ["BRAIN_LEARNED_ANIMACY_CUE"] = "1"
    os.environ["BRAIN_LEARNED_VERB_SELECTS"] = "1"
    os.environ["BRAIN_LEARNED_ANIMACY_LESION"] = "0"
    os.environ["BRAIN_LEARNED_VERB_SELECTS_LESION"] = "0"
    os.environ["BRAIN_ENFORCE_PLASTIC_MASK"] = "1" if args.arm == "on" else "0"

    import research.runners.comprehension_production_organ as CO
    from research.runners._comprehension_learned_cues_joint_arm import all_items

    t0 = time.time()
    if args.organ == "pooled":
        organ = CO.get_organ(seed=args.seed)
    else:
        organ = CO.ComprehensionProductionOrgan(seed=args.seed)
    organ.ensure_built()
    bridge = organ.comp.bridge
    cfg = bridge.core_config
    if args.tap == "on":
        install_tap(bridge)

    coords_pre = _coords(bridge)
    labels_pre = _mask_label_by_coord(bridge)
    mask_pre = _mask(bridge)
    stores_pre = {name: _snapshot(bridge, name) for name in STORES}
    nnz_pre = int(bridge.cp_connections.nnz)

    frozen_coords = set()
    if mask_pre is not None:
        coo_pre = bridge.cp_connections.tocoo()
        for k, (r, c) in enumerate(zip(coo_pre.row, coo_pre.col)):
            if k < mask_pre.size and not mask_pre[k]:
                frozen_coords.add((int(r), int(c)))

    items = all_items()
    if args.items:
        items = items[:args.items]

    rows = {}
    for label, n0, v, n1 in items:
        text = f"the {n0} {v} the {n1}"
        tr = CO.extract_transitive(text)
        rows[label] = {
            "text": text,
            "extract_transitive": list(tr) if tr else None,
            "competent": bool(organ.competent(*tr)) if tr else None,
            "judge": organ.judge(text),
            "repair_target": organ.repair_target(text),
        }

    coords_end = _coords(bridge)
    stores_post = {name: _snapshot(bridge, name) for name in STORES}
    common = set(coords_pre) & set(coords_end)
    drift = {cc: coords_end[cc] - coords_pre[cc] for cc in common}
    frozen_signed = [d for cc, d in drift.items() if cc in frozen_coords]
    frozen_drifts = [abs(d) for d in frozen_signed]
    plastic_drifts = [abs(d) for cc, d in drift.items() if cc not in frozen_coords]


    stores_delta = {}
    for name in STORES:
        a, b = stores_pre[name], stores_post[name]
        if a is None or b is None:
            stores_delta[name] = {"present": a is not None and b is not None}
            continue
        if a.shape != b.shape:
            stores_delta[name] = {"present": True, "shape_pre": list(a.shape), "shape_post": list(b.shape),
                                  "note": "array resized (lazy gate growth / rebuild / compaction)"}
            continue
        if a.dtype == np.bool_ or b.dtype == np.bool_:
            stores_delta[name] = {"present": True, "label_flips": int((a != b).sum())}
            continue
        d = np.abs(a - b)
        stores_delta[name] = {"present": True, "max_abs_delta": float(d.max()) if d.size else 0.0,
                             "changed_entries": int((d > 1e-9).sum())}

    sites = _aggregate(_LOG)
    written_net = sum(s["frozen_potentiation"] - s["frozen_depression"] for s in sites)
    out = {
        "seed": args.seed, "arm": args.arm,
        "enforced": os.environ.get("BRAIN_ENFORCE_PLASTIC_MASK", "") == "1",
        "backend": "numpy", "device": "cpu", "elapsed_seconds": round(time.time() - t0, 2),
        "n_items": len(items), "sim_steps": _STEPS, "nnz_pre": nnz_pre, "nnz_post": int(bridge.cp_connections.nnz),
        "csr_data_array_rebuilt_times": _ARRAY_REPLACED,
        "config": {
            "enable_hebbian_learning": bool(cfg.enable_hebbian_learning),
            "enable_stdp": bool(getattr(cfg, "enable_stdp", False)),
            "enable_structural_plasticity": bool(getattr(cfg, "enable_structural_plasticity", False)),
            "enable_reward_modulation": bool(getattr(cfg, "enable_reward_modulation", False)),
            "enable_homeostasis": bool(getattr(cfg, "enable_homeostasis", False)),
            "hebbian_rate_window": bool(getattr(cfg, "hebbian_rate_window", False)),
            "enforce_plastic_mask_in_hebbian": bool(getattr(cfg, "enforce_plastic_mask_in_hebbian", False)),
            "enforce_plastic_mask_beyond_hebbian": bool(getattr(cfg, "enforce_plastic_mask_beyond_hebbian", False)),
        },
        "organ": args.organ,
        "tap": args.tap,
        "experiment_engine_plasticity_this_phase": (
            None if getattr(bridge, "experiment_engine", None) is None
            else bool(getattr(bridge.experiment_engine, "plasticity_enabled_this_phase", True))),
        "mask_len_pre": int(mask_pre.size) if mask_pre is not None else 0,
        "frozen_synapses_pre": int((~mask_pre).sum()) if mask_pre is not None else 0,
        "write_sites": sites,
        "total_writes": len(_LOG),
        "frozen_drift_regions": _region_attribution(bridge, frozen_coords, coords_pre, coords_end),
        "stores_delta": stores_delta,
        "coordinate_drift": {
            "synapses_comparable_pre_post": len(common),
            "csr_rebuilds": _REBUILDS[0],
            "frozen_labelled_synapses": len(frozen_coords),
            "frozen_max_abs_drift": max(frozen_drifts, default=0.0),
            "frozen_sum_abs_drift": float(sum(frozen_drifts)),
            "frozen_signed_drift": float(sum(frozen_signed)),
            "frozen_synapses_drifted": sum(1 for d in frozen_drifts if d > 1e-9),
            "plastic_max_abs_drift": max(plastic_drifts, default=0.0),
            "plastic_sum_abs_drift": float(sum(plastic_drifts)),
            "plastic_synapses_drifted": sum(1 for d in plastic_drifts if d > 1e-9),
            # The tap sums every frozen-synapse delta it sees; the coordinate snapshot is the NET change.
            # If the hooks cover every write path, the signed sums agree. A nonzero signed residual means a
            # write path escaped the instrument (or the CSR re-ordered synapses between the snapshots).
            "completeness_residual_signed": round(written_net - float(sum(frozen_signed)), 6),
            "completeness_residual_abs": round(float(sum(s["frozen_sum_abs_delta"] for s in sites))
                                              - float(sum(frozen_drifts)), 6),
            "mask_labels_changed_by_rebuild": sum(1 for cc, lab in _mask_label_by_coord(bridge).items()
                                                  if cc in labels_pre and lab != labels_pre[cc]),
        },
        "battery_digest": _digest(rows),
        "battery_items": {k: _digest(v) for k, v in rows.items()},
        "reply_margins": {k: (v["judge"] or {}).get("margin") for k, v in rows.items()},
        "reply_competent": {k: v["competent"] for k, v in rows.items()},
    }
    # Ask the attribution question out loud rather than leaving two numbers one key apart: WHOSE drift is
    # this? The organ's own gross weight change splits into the population the mask declares frozen and the
    # population it declares plastic, and the recorded-write budget says which write sites own each.
    gross_drift = float(sum(frozen_drifts) + sum(plastic_drifts))
    out["drift_term_budget"] = tools.lab.term_budget(
        "gross weight drift in the organ (sum |dW|, by mask label)", gross_drift,
        {"mask-declared-frozen": float(sum(frozen_drifts)), "mask-declared-plastic": float(sum(plastic_drifts))})
    out["write_site_budget"] = {
        s["site"]: round(s["frozen_sum_abs_delta"], 4) for s in sites if s["frozen_sum_abs_delta"] > 0
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("[tap] seed=%s arm=%s steps=%d writes=%d sites=%d frozen_sum_delta=%.6g frozen_max_drift=%.6g -> %s"
          % (args.seed, args.arm, _STEPS, len(_LOG), len(sites),
             sum(s["frozen_sum_abs_delta"] for s in sites),
             out["coordinate_drift"]["frozen_max_abs_drift"], args.out))


if __name__ == "__main__":
    main()
