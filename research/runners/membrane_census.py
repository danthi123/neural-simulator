"""MEMBRANE-VALIDITY CENSUS -- an observe-only probe that runs one Stage-2 gate arm (the live production brain) and
records, for every `SimulationBridge` the arm builds, which membranes ever left the physiological range.

WHY (2026-10-26, finding 2026-10-26-stage2-fail-diagnoses.md). Two Stage-2 failures (the emotion organ's s101 joy miss
and the BG step-1 breach) were traced to the 1 ms explicit-Euler integrator going unstable in low-capacitance
heterogeneity draws under strong conductance: membranes below -110 mV, down to about -1900 mV, emitting spikes that
biology would not produce. Other organs that combine strong inhibition with low-C draws may carry the same latent
artefact. This is the census that finding says is owed before any claim about them.

WHAT IT RECORDS (per bridge instance, so per organ). The class method `SimulationBridge._run_one_simulation_step` is
wrapped (monkeypatch at CLASS level, restored on exit, no sim/ edit); after every step the bridge's membrane array v (mV)
is read, never written:
  * per region: minimum and maximum v ever seen; the minimum DRAWN capacitance C (Izhikevich `cp_izh_C`; AdEx the
    scalar `adex_C`); the steps and neuron-steps with v < V_LOW_MV (-110 mV) and with v above the neuron's vpeak + 5 mV
    (Izhikevich `cp_izh_vpeak`, AdEx `adex_V_peak`; HH and resonate-and-fire have no high check); the largest
    g_i*dt/C seen (the explicit-step hazard index: the finding puts the Euler overshoot at about 2 and above).
  * per out-of-range neuron (below -110 mV, above vpeak + 5 mV, or non-finite): global index, region, inhibitory flag,
    cell-type label, drawn C, k, vr, vt, vpeak, a, b, c_reset, d, first out-of-range step (per bridge and census-wide),
    extreme low and high v, steps out of range, and g_i, g_e and g_i*dt/C on the first such step.
  * per bridge: `cfg.enable_bounded_conductance_integration` read back from its own core config (every step; the set of
    values seen), the neuron model, dt, neuron count, who built it (the call chain at construction) and who steps it.
The v array is read AFTER the step, so a spike's reset to c has already happened: a value below -110 mV or above
vpeak + 5 mV at that point is a membrane the step left out of range. A LOW membrane is the explicit-Euler instability
(and its spurious spikes); a HIGH one is mostly a cell inside its refractory window, whose crossing the step withholds
(`fired = v_new >= threshold & not_in_refractory`, so no reset), and each row records `refractory` at first detection.
The high bound is max(vpeak, the homeostatic adapted threshold where the bridge uses one) + 5 mV. Coverage limit: only `_run_one_simulation_step`
is observed (the megakernel dispatch lives inside it); `rf_resonate_steps` (resonate-and-fire phasors, whose v is a
phasor component, not a membrane) is not.

CAVEAT (carried into every result). One seed is a census of ONE heterogeneity draw, and the seed-to-cell map is specific
to the numpy CPU random stream. "No out-of-range membrane on this seed" is not "this organ cannot produce one".

  .venv/bin/python -m pytest tests/test_membrane_census.py -q
  bash tools/memcap.sh 11 -- .venv/bin/python -m research.runners.membrane_census --shard --seed 7 --group emo \
      --arm intact_a --out research/findings/raw/_membrane_census/dev/s7/emo__intact_a.json
  .venv/bin/python -m research.runners.membrane_census --summarize research/findings/raw/_membrane_census/dev/s7/*.json
  .venv/bin/python -m research.runners.membrane_census --emit-pool-lines > research/coordination/membrane-census-pool-lines.txt
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import weakref

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

V_LOW_MV = -110.0                 # below this a membrane is out of physiological range (task definition)
VPEAK_MARGIN_MV = 5.0             # above vpeak + this a membrane the step failed to reset is out of range
MAX_DETAIL_PER_BRIDGE = 500       # per-neuron rows kept in the JSON per bridge (the counts are always exact)
MAX_CONTEXT_PER_BRIDGE = 5000     # first-step context (g_i, g_e) captured for at most this many neurons per bridge
CENSUS_VERSION = 1
DEFAULT_OUT_DIR = "research/findings/raw/_membrane_census/dev"
POOL_MEM_GB = 9                   # the Stage-2 lane's memory declaration (stage2_exit_gate.MEM_GB / MEMCAP_GB)
POOL_MEMCAP_GB = 10
POOL_SEEDS = tuple(range(1, 13))
CAVEAT = ("One seed is a census of one heterogeneity draw, and the seed-to-cell map is specific to the numpy CPU "
          "random stream; no out-of-range membrane on this seed does not mean the organ cannot produce one. Only "
          "SimulationBridge._run_one_simulation_step is observed (rf_resonate_steps is not).")

_THIS_FILE = os.path.abspath(__file__)


# ══════════════════════════════════════════════ small helpers ══════════════════════════════════════════════════════
def _f(x, nd=4):
    """A JSON-safe float (None for inf/nan sentinels)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return round(x, nd)


def _host(arr):
    """A numpy view/copy of a backend array (cupy -> host); None stays None."""
    if arr is None:
        return None
    from sim.backend import to_host
    return np.asarray(to_host(arr))


def _frame_label(frame):
    code = frame.f_code
    mod = frame.f_globals.get("__name__", "?")
    return "%s:%s" % (mod, getattr(code, "co_qualname", code.co_name))


def _caller_chain(depth_skip=2, limit=8):
    """The call chain above a bridge construction, skipping the bridge's own file and this module. Returns a list of
    'module:Qualname:lineno' strings, innermost first."""
    out = []
    try:
        f = sys._getframe(depth_skip)
    except ValueError:
        return out
    while f is not None and len(out) < limit:
        fn = os.path.abspath(f.f_code.co_filename)
        if fn != _THIS_FILE and not fn.endswith(os.sep + os.path.join("sim", "bridge.py")) \
                and not fn.startswith("<frozen") and "importlib" not in fn:
            out.append("%s:%d" % (_frame_label(f), f.f_lineno))
        f = f.f_back
    return out


def _spike_ceiling(bridge, vp):
    """Per-neuron voltage the step compares v_new against to decide a spike: cp_izh_vpeak, or max(vpeak, the homeostatic
    adapted threshold) where the bridge uses one (the step's own selection, bridge.py section 3). The high bound is this
    plus the margin, so a legitimately raised threshold is not read as an out-of-range membrane."""
    th = getattr(bridge, "cp_neuron_firing_thresholds", None)
    if th is None or vp is None:
        return vp
    th = _host(th)
    if th.shape != vp.shape:
        return vp
    if getattr(bridge.core_config, "enable_homeostasis", False):
        return np.maximum(vp, th)
    mask = getattr(bridge, "cp_homeostasis_neuron_mask", None)
    if mask is not None:
        return np.where(_host(mask).astype(bool), np.maximum(vp, th), vp)
    return vp


def _range_policy(model):
    """(check_low, high_kind) per neuron model. high_kind: 'vpeak' (per-neuron cp_izh_vpeak) | 'adex' (scalar) | None."""
    if model == "IZHIKEVICH":
        return True, "vpeak"
    if model == "ADEX":
        return True, "adex"
    if model == "HODGKIN_HUXLEY":
        return True, None
    return False, None                      # RESONATE_AND_FIRE: v is a phasor component, not a membrane


class _RegionMap:
    """Neuron -> region layout of one bridge, as run-length segments so per-region reductions are one reduceat call."""

    def __init__(self, bridge, n):
        names, region_of = [], np.full(n, -1, dtype=np.int32)
        inh = np.zeros(n, dtype=bool)
        rm = getattr(bridge, "region_manager", None)
        if rm is not None:
            try:
                d = rm.region_indices_dict()
            except Exception:
                d = {}
            for name, idx in d.items():
                ia = np.asarray(list(idx), dtype=np.int64)
                ia = ia[(ia >= 0) & (ia < n)]
                if ia.size == 0:
                    continue
                region_of[ia] = len(names)
                names.append(str(name))
                try:
                    ii = np.asarray(list(rm.inhibitory_indices(name)), dtype=np.int64)
                    ii = ii[(ii >= 0) & (ii < n)]
                    inh[ii] = True
                except Exception:
                    pass
        un = region_of < 0
        if un.any():
            names.append("<unassigned>" if names else "<all>")
            region_of[un] = len(names) - 1
        self.names = names
        self.region_of = region_of
        self.inhibitory = inh
        change = np.flatnonzero(np.diff(region_of)) + 1
        self.seg_starts = np.concatenate(([0], change)).astype(np.int64)
        self.seg_ends = np.append(self.seg_starts[1:], n).astype(np.int64)
        self.seg_region = region_of[self.seg_starts]
        self.sizes = np.bincount(region_of, minlength=len(names))
        self.span = {}
        for r, name in enumerate(names):
            m = self.seg_region == r
            self.span[name] = (int(self.seg_starts[m].min()), int(self.seg_ends[m].max()), int(m.sum()) == 1)
        self.key = (id(rm), n, len(names))

    def per_region_indices(self):
        order = np.argsort(self.region_of, kind="stable")
        bounds = np.concatenate(([0], np.cumsum(self.sizes)))
        return [order[bounds[r]:bounds[r + 1]] for r in range(len(self.names))]


class _BridgeRec:
    """Everything the census keeps for one bridge instance (holds the bridge only weakly)."""

    def __init__(self, bridge_id, bridge, n, owner_chain):
        cfg = bridge.core_config
        self.bridge_id = bridge_id
        self.ref = weakref.ref(bridge)
        self.n = n
        self.owner_chain = owner_chain
        self.model = str(getattr(cfg, "neuron_model_type", "?"))
        self.dt = float(getattr(cfg, "dt_ms", 1.0))
        self.n_steps = 0
        self.callers = collections.Counter()
        self.bounded_seen = set()
        self.map = None
        self._alloc_region_stats()
        self.c_first = None
        self.c_last = None
        self.first_step = self.first_gstep = self.vmin_n = self.vmax_n = None
        self.nlow = self.nhigh = self.nnonf = None
        self.ctx = {}
        self.steps_with_oor = 0
        self.prev_timers = None                       # post-step refractory timers of the previous observation
        self.prev_v = None                            # post-step v of the previous observation (the step's starting v)
        self.prev_u = None                            # ... and u
        self.n_spikes_total = 0                       # all spikes this bridge emitted while observed
        self.n_spikes_oor_cells = 0                   # spikes emitted by cells AFTER their first out-of-range step
        self.nspk_after = None

    def _alloc_region_stats(self, old=None, old_names=None):
        R = len(self.map.names) if self.map is not None else 0
        self.vmin_r = np.full(R, np.inf)
        self.vmax_r = np.full(R, -np.inf)
        self.g_r = np.full((3, R), -np.inf)          # per-region max of g_i, g_e, (g_i+g_e) times dt/C
        self.steps_low_r = np.zeros(R, dtype=np.int64)
        self.steps_high_r = np.zeros(R, dtype=np.int64)
        self.nsteps_low_r = np.zeros(R, dtype=np.int64)
        self.nsteps_high_r = np.zeros(R, dtype=np.int64)
        if old is not None and old_names is not None and self.map is not None:
            where = {nm: i for i, nm in enumerate(self.map.names)}
            for j, nm in enumerate(old_names):
                i = where.get(nm)
                if i is None:
                    continue
                for new, prev in ((self.vmin_r, old[0]), (self.vmax_r, old[1]),
                                  (self.steps_low_r, old[3]), (self.steps_high_r, old[4]),
                                  (self.nsteps_low_r, old[5]), (self.nsteps_high_r, old[6])):
                    new[i] = prev[j]
                self.g_r[:, i] = old[2][:, j]

    def set_map(self, m):
        old = None
        old_names = None
        if self.map is not None:
            old = (self.vmin_r, self.vmax_r, self.g_r, self.steps_low_r, self.steps_high_r, self.nsteps_low_r,
                   self.nsteps_high_r)
            old_names = list(self.map.names)
        self.map = m
        self._alloc_region_stats(old, old_names)

    def ensure_neuron_arrays(self):
        if self.first_step is None:
            n = self.n
            self.first_step = np.full(n, -1, dtype=np.int64)
            self.first_gstep = np.full(n, -1, dtype=np.int64)
            self.vmin_n = np.full(n, np.inf)
            self.vmax_n = np.full(n, -np.inf)
            self.nlow = np.zeros(n, dtype=np.int32)
            self.nhigh = np.zeros(n, dtype=np.int32)
            self.nnonf = np.zeros(n, dtype=np.int32)
            self.nspk_after = np.zeros(n, dtype=np.int32)


def _c_summary(bridge, rm):
    """Per-region drawn-capacitance summary (Izhikevich cp_izh_C per neuron; AdEx the scalar adex_C)."""
    cfg = bridge.core_config
    model = str(getattr(cfg, "neuron_model_type", "?"))
    out = {}
    C = _host(getattr(bridge, "cp_izh_C", None)) if model == "IZHIKEVICH" else None
    if C is not None and C.shape[0] != rm.region_of.shape[0]:
        C = None
    idxs = rm.per_region_indices()
    for r, name in enumerate(rm.names):
        ix = idxs[r]
        if C is not None and ix.size:
            c = C[ix].astype(np.float64)
            j = int(np.argmin(c))
            out[name] = {"n": int(ix.size), "c_min": _f(c[j]), "c_min_idx": int(ix[j]), "c_p01": _f(np.percentile(c, 1)),
                         "c_median": _f(np.median(c)), "c_max": _f(c.max())}
        elif model == "ADEX":
            v = _f(getattr(cfg, "adex_C", None))
            out[name] = {"n": int(ix.size), "c_min": v, "c_min_idx": None, "c_p01": v, "c_median": v, "c_max": v}
        else:
            out[name] = {"n": int(ix.size), "c_min": None, "c_min_idx": None, "c_p01": None, "c_median": None,
                         "c_max": None}
    return out


# ══════════════════════════════════════════════ the census ═════════════════════════════════════════════════════════
class Census:
    """Install/uninstall the class-level observer. Use as a context manager; the originals are restored on exit."""

    def __init__(self, v_low=V_LOW_MV, vpeak_margin=VPEAK_MARGIN_MV, track_conductance=True):
        self.v_low = float(v_low)
        self.margin = float(vpeak_margin)
        self.track_conductance = bool(track_conductance)
        self.total_steps = 0
        self.recs = []                                         # insertion order == first-seen order
        self._by_bridge = weakref.WeakKeyDictionary()
        self._owners = weakref.WeakKeyDictionary()
        self._lock = threading.RLock()
        self.observe_s = 0.0
        self.observer_errors = 0
        self.first_error = None
        self._installed = False
        self._orig_step = None
        self._orig_init = None
        self._cls = None

    # ── install / uninstall ──────────────────────────────────────────────────────────────────────────────────────
    def install(self):
        if self._installed:
            raise RuntimeError("membrane census already installed")
        from sim.bridge import SimulationBridge as SB
        if getattr(SB.__dict__["_run_one_simulation_step"], "_membrane_census_wrapper", False):
            raise RuntimeError("membrane census already installed (another Census holds the class patch)")
        self._cls = SB
        self._orig_step = SB.__dict__["_run_one_simulation_step"]
        self._orig_init = SB.__dict__["__init__"]
        census = self
        orig_step, orig_init = self._orig_step, self._orig_init

        def _init_wrapper(bridge, *a, **k):
            chain = _caller_chain(depth_skip=2)
            orig_init(bridge, *a, **k)
            try:
                census._owners[bridge] = chain
            except TypeError:
                pass

        def _step_wrapper(bridge, *a, **k):
            try:
                caller = sys._getframe(1)
                site = (caller.f_code, caller.f_lineno, caller.f_globals.get("__name__", "?"))
            except ValueError:
                site = None
            r = orig_step(bridge, *a, **k)
            t0 = time.perf_counter()
            try:
                census._observe(bridge, site)
            except Exception as e:                              # an observer fault must never kill the workload
                census.observer_errors += 1
                if census.first_error is None:
                    import traceback
                    census.first_error = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-1500:]
            census.observe_s += time.perf_counter() - t0
            return r

        _step_wrapper._membrane_census_wrapper = True
        _init_wrapper._membrane_census_wrapper = True
        self._step_wrapper, self._init_wrapper = _step_wrapper, _init_wrapper
        SB._run_one_simulation_step = _step_wrapper
        SB.__init__ = _init_wrapper
        self._installed = True
        return self

    def uninstall(self):
        if not self._installed:
            return
        SB = self._cls
        SB._run_one_simulation_step = self._orig_step
        SB.__init__ = self._orig_init
        self._installed = False

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()
        return False

    # ── per-step observation ─────────────────────────────────────────────────────────────────────────────────────
    def _rec_for(self, bridge, n):
        rec = self._by_bridge.get(bridge)
        if rec is None:
            with self._lock:
                rec = self._by_bridge.get(bridge)
                if rec is None:
                    chain = self._owners.get(bridge) or []
                    rec = _BridgeRec(len(self.recs), bridge, n, chain)
                    self._by_bridge[bridge] = rec
                    self.recs.append(rec)
        return rec

    def _observe(self, bridge, site):
        v = getattr(bridge, "cp_membrane_potential_v", None)
        if v is None or not getattr(bridge, "is_initialized", True):
            return
        vh = _host(v)
        n = int(vh.shape[0])
        if n == 0:
            return
        rec = self._rec_for(bridge, n)
        rec.n_steps += 1
        self.total_steps += 1
        rec.callers[site] += 1
        pre = rec.prev_timers                         # == the timers this step decided spikes with (None on the first)
        pre_v, pre_u = rec.prev_v, rec.prev_u
        rec.prev_v = np.array(vh, copy=True)
        u_now = getattr(bridge, "cp_recovery_variable_u", None)
        rec.prev_u = None if u_now is None else np.array(_host(u_now), copy=True)
        rt_now = getattr(bridge, "cp_refractory_timers", None)
        rec.prev_timers = None if rt_now is None else np.array(_host(rt_now), copy=True)
        fired = getattr(bridge, "cp_firing_states", None)
        if fired is not None:
            fh = _host(fired)
            if fh.shape[0] == n:
                fh = fh.astype(bool, copy=False)
                rec.n_spikes_total += int(np.count_nonzero(fh))
                if rec.first_step is not None:                    # spikes by cells that already left the range
                    sp = fh & (rec.first_step >= 0)
                    rec.nspk_after += sp
                    rec.n_spikes_oor_cells += int(np.count_nonzero(sp))
        cfg = bridge.core_config
        rec.bounded_seen.add(bool(getattr(cfg, "enable_bounded_conductance_integration", False)))
        m = rec.map
        if m is None or m.key != (id(getattr(bridge, "region_manager", None)), n, len(m.names)):
            m = _RegionMap(bridge, n)
            rec.set_map(m)
            rec.c_first = rec.c_first or _c_summary(bridge, m)
        seg_min = np.minimum.reduceat(vh, m.seg_starts)
        seg_max = np.maximum.reduceat(vh, m.seg_starts)
        np.fmin.at(rec.vmin_r, m.seg_region, seg_min)
        np.fmax.at(rec.vmax_r, m.seg_region, seg_max)
        check_low, high_kind = _range_policy(rec.model)
        # the explicit-step hazard index g*dt/C (Izhikevich; the finding puts the Euler overshoot at about 2 and above),
        # per region maximum over steps, for g_i, g_e and their sum (the end-of-step conductances the next step integrates)
        if self.track_conductance and rec.model == "IZHIKEVICH":
            gi = getattr(bridge, "cp_conductance_g_i", None)
            ge = getattr(bridge, "cp_conductance_g_e", None)
            C = getattr(bridge, "cp_izh_C", None)
            if C is not None and (gi is not None or ge is not None):
                try:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        k = rec.dt / _host(C)
                        gih = _host(gi) if gi is not None else np.zeros(n, dtype=np.float32)
                        geh = _host(ge) if ge is not None else np.zeros(n, dtype=np.float32)
                        for row, arr in enumerate((gih * k, geh * k, (gih + geh) * k)):
                            np.fmax.at(rec.g_r[row], m.seg_region, np.maximum.reduceat(arr, m.seg_starts))
                except Exception:
                    self.observer_errors += 1
        if not check_low:
            return
        gmin, gmax = float(seg_min.min()), float(seg_max.max())
        vp = None
        hi_min = np.inf
        if high_kind == "vpeak":
            vp = _spike_ceiling(bridge, _host(getattr(bridge, "cp_izh_vpeak", None)))
            if vp is not None:
                hi_min = float(vp.min()) + self.margin
        elif high_kind == "adex":
            hi_min = float(getattr(cfg, "adex_V_peak", np.inf)) + self.margin
        if gmin >= self.v_low and gmax <= hi_min:             # the fast path: nothing out of range (NaN falls through)
            return
        self._record_out_of_range(rec, bridge, vh, vp, high_kind, hi_min, pre, pre_v, pre_u)

    def _record_out_of_range(self, rec, bridge, vh, vp, high_kind, hi_min, pre=None, pre_v=None, pre_u=None):
        m = rec.map
        R = len(m.names)
        low = vh < self.v_low
        if high_kind == "vpeak" and vp is not None:
            high = vh > (vp + np.float32(self.margin))
        elif high_kind == "adex":
            high = vh > hi_min
        else:
            high = np.zeros(vh.shape, dtype=bool)
        nonf = ~np.isfinite(vh)
        bad = low | high | nonf
        bad_idx = np.flatnonzero(bad)
        if bad_idx.size == 0:
            return
        rec.steps_with_oor += 1
        lo_idx, hi_idx, nf_idx = np.flatnonzero(low), np.flatnonzero(high), np.flatnonzero(nonf)
        if lo_idx.size:
            c = np.bincount(m.region_of[lo_idx], minlength=R)
            rec.nsteps_low_r += c
            rec.steps_low_r += (c > 0)
        if hi_idx.size:
            c = np.bincount(m.region_of[hi_idx], minlength=R)
            rec.nsteps_high_r += c
            rec.steps_high_r += (c > 0)
        rec.ensure_neuron_arrays()
        new = bad_idx[rec.first_step[bad_idx] < 0]
        rec.first_step[new] = rec.n_steps
        rec.first_gstep[new] = self.total_steps
        rec.vmin_n[bad_idx] = np.fmin(rec.vmin_n[bad_idx], vh[bad_idx])
        rec.vmax_n[bad_idx] = np.fmax(rec.vmax_n[bad_idx], vh[bad_idx])
        rec.nlow[lo_idx] += 1
        rec.nhigh[hi_idx] += 1
        rec.nnonf[nf_idx] += 1
        if new.size and len(rec.ctx) < MAX_CONTEXT_PER_BRIDGE:
            gi, ge = _host(getattr(bridge, "cp_conductance_g_i", None)), _host(getattr(bridge, "cp_conductance_g_e", None))
            C = _host(getattr(bridge, "cp_izh_C", None)) if rec.model == "IZHIKEVICH" else None
            rt = rec.prev_timers                         # post-step timers (this observation)
            P = {}
            if rec.model == "IZHIKEVICH":
                for key, attr in (("C", "cp_izh_C"), ("k", "cp_izh_k"), ("vr", "cp_izh_vr"), ("vt", "cp_izh_vt"),
                                  ("vpeak", "cp_izh_vpeak"), ("a", "cp_izh_a"), ("b", "cp_izh_b"),
                                  ("c_reset", "cp_izh_c_reset"), ("d", "cp_izh_d_increment")):
                    P[key] = _host(getattr(bridge, attr, None))
            types = getattr(getattr(bridge, "runtime_state", None), "neuron_types_list_for_viz", None)
            if types is not None and len(types) != rec.n:
                types = None
            extra = {key: _host(getattr(bridge, attr, None)) for key, attr in (
                ("u", "cp_recovery_variable_u"), ("ext_pA", "cp_external_input_current"),
                ("g_nmda", "cp_conductance_g_nmda"), ("g_gabab", "cp_conductance_g_gabab"),
                ("g_nmda_rec", "cp_conductance_g_nmda_recurrent"))}
            for idx in new[:MAX_CONTEXT_PER_BRIDGE - len(rec.ctx)]:
                i = int(idx)
                d = {"params": {key: (_f(arr[i]) if arr is not None and arr.shape[0] > i else None) for key, arr in P.items()},
                     "type": (str(types[i]) if types is not None else None),
                     "v_before": (_f(pre_v[i], 2) if pre_v is not None and pre_v.shape[0] > i else None),
                     "v_first": _f(vh[i], 2),
                     "u_before": _f(pre_u[i], 2) if pre_u is not None and pre_u.shape[0] > i else None,
                     # the total input current the step integrated, by inverting the Euler update of the Izhikevich-2007
                     # model (valid when the cell did not spike/reset in the step, as for every row recorded here)
                     "I_implied_pA": self._implied_current(P, i, pre_v, pre_u, vh, rec.dt),
                     "u_first": _f(extra["u"][i], 2) if extra["u"] is not None and extra["u"].shape[0] > i else None,
                     "ext_pA": _f(extra["ext_pA"][i], 2) if extra["ext_pA"] is not None and extra["ext_pA"].shape[0] > i else None,
                     "g_nmda": _f(extra["g_nmda"][i]) if extra["g_nmda"] is not None and extra["g_nmda"].shape[0] > i else None,
                     "g_gabab": _f(extra["g_gabab"][i]) if extra["g_gabab"] is not None and extra["g_gabab"].shape[0] > i else None,
                     "g_nmda_rec": (_f(extra["g_nmda_rec"][i]) if extra["g_nmda_rec"] is not None
                                    and extra["g_nmda_rec"].shape[0] > i else None),
                     "g_i": _f(gi[i]) if gi is not None and gi.shape[0] > i else None,
                     "g_e": _f(ge[i]) if ge is not None and ge.shape[0] > i else None,
                     # refractory = in its window when the step decided spikes (a withheld crossing is not reset);
                     # refractory_post = timer still running after the step
                     "refractory": (bool(pre[i] > 0) if pre is not None and pre.shape[0] > i else None),
                     "refractory_post": (bool(rt[i] > 0) if rt is not None and rt.shape[0] > i else None)}
                ok = C is not None and float(C[i]) > 0
                for key, g in (("gi_dt_over_C", d["g_i"]), ("ge_dt_over_C", d["g_e"]),
                               ("gtot_dt_over_C", (d["g_i"] or 0.0) + (d["g_e"] or 0.0)
                                if d["g_i"] is not None or d["g_e"] is not None else None)):
                    d[key] = _f(g * rec.dt / float(C[i]), 3) if g is not None and ok else None
                rec.ctx[i] = d

    @staticmethod
    def _implied_current(P, i, pre_v, pre_u, vh, dt):
        try:
            if not P or pre_v is None or pre_u is None:
                return None
            C, k, vr, vt = (float(P[key][i]) for key in ("C", "k", "vr", "vt"))
            vb, ub = float(pre_v[i]), float(pre_u[i])
            return _f(C * (float(vh[i]) - vb) / dt - k * (vb - vr) * (vb - vt) + ub, 1)
        except Exception:
            return None

    # ── reporting ────────────────────────────────────────────────────────────────────────────────────────────────
    def _neuron_rows(self, rec):
        if rec.first_step is None:
            return [], 0, False
        idx = np.flatnonzero(rec.first_step >= 0)
        if idx.size == 0:
            return [], 0, False
        sev = np.zeros(idx.size)
        for j, i in enumerate(idx):
            c = rec.ctx.get(int(i)) or {}
            vpeak = (c.get("params") or {}).get("vpeak")
            s = np.inf if rec.nnonf[i] > 0 else 0.0
            if rec.nlow[i] > 0 and np.isfinite(rec.vmin_n[i]):
                s = max(s, self.v_low - float(rec.vmin_n[i]))
            if rec.nhigh[i] > 0 and np.isfinite(rec.vmax_n[i]):
                s = max(s, float(rec.vmax_n[i]) - ((vpeak if vpeak is not None else 0.0) + self.margin))
            sev[j] = s
        order = np.argsort(-sev, kind="stable")
        truncated = idx.size > MAX_DETAIL_PER_BRIDGE
        rows = []
        for j in order[:MAX_DETAIL_PER_BRIDGE]:
            i = int(idx[j])
            c = rec.ctx.get(i) or {}
            row = {"idx": i, "region": rec.map.names[int(rec.map.region_of[i])], "inhibitory": bool(rec.map.inhibitory[i]),
                   "type": c.get("type")}
            params = c.get("params") or {}
            for key in ("C", "k", "vr", "vt", "vpeak", "a", "b", "c_reset", "d"):
                row[key] = params.get(key)
            row.update({"first_step": int(rec.first_step[i]), "first_census_step": int(rec.first_gstep[i]),
                        "extreme_low_mV": _f(rec.vmin_n[i], 2) if rec.nlow[i] > 0 else None,
                        "extreme_high_mV": _f(rec.vmax_n[i], 2) if rec.nhigh[i] > 0 else None,
                        "n_steps_low": int(rec.nlow[i]), "n_steps_high": int(rec.nhigh[i]),
                        "n_steps_nonfinite": int(rec.nnonf[i]),
                        "n_spikes_after_first": int(rec.nspk_after[i])})
            for key in ("v_before", "v_first", "u_before", "u_first", "I_implied_pA", "ext_pA", "g_i", "g_e", "g_nmda", "g_gabab", "g_nmda_rec",
                        "refractory", "refractory_post", "gi_dt_over_C", "ge_dt_over_C", "gtot_dt_over_C"):
                row[key] = c.get(key)
            rows.append(row)
        return rows, int(idx.size), truncated

    def bridge_report(self, rec):
        bridge = rec.ref()
        m = rec.map
        if m is None:
            return None
        if bridge is not None:
            try:
                rec.c_last = _c_summary(bridge, m)
            except Exception:
                rec.c_last = None
        c_first = rec.c_first or {}
        c_last = rec.c_last or c_first
        rows, n_oor, truncated = self._neuron_rows(rec)
        regions = {}
        for r, name in enumerate(m.names):
            s0, s1, contig = m.span[name]
            cf = c_first.get(name, {})
            cl = c_last.get(name, {})
            n_low = n_high = 0
            if rec.first_step is not None:
                sel = m.region_of == r
                n_low = int(np.count_nonzero(rec.nlow[sel] > 0))
                n_high = int(np.count_nonzero(rec.nhigh[sel] > 0))
            regions[name] = {
                "n": int(m.sizes[r]), "start": s0, "stop": s1, "contiguous": contig,
                "v_min": _f(rec.vmin_r[r], 2), "v_max": _f(rec.vmax_r[r], 2),
                "c_min": cf.get("c_min"), "c_min_idx": cf.get("c_min_idx"), "c_p01": cf.get("c_p01"),
                "c_median": cf.get("c_median"), "c_min_at_end": cl.get("c_min"),
                "n_low_neurons": n_low, "n_high_neurons": n_high,
                "steps_low": int(rec.steps_low_r[r]), "steps_high": int(rec.steps_high_r[r]),
                "neuron_steps_low": int(rec.nsteps_low_r[r]), "neuron_steps_high": int(rec.nsteps_high_r[r]),
                "max_gi_dt_over_C": _f(rec.g_r[0][r], 3), "max_ge_dt_over_C": _f(rec.g_r[1][r], 3),
                "max_gtot_dt_over_C": _f(rec.g_r[2][r], 3)}
        vmins = [(rec.vmin_r[r], m.names[r]) for r in range(len(m.names)) if np.isfinite(rec.vmin_r[r])]
        vmaxs = [(rec.vmax_r[r], m.names[r]) for r in range(len(m.names)) if np.isfinite(rec.vmax_r[r])]
        vmin = min(vmins) if vmins else (None, None)
        vmax = max(vmaxs) if vmaxs else (None, None)
        top_caller = [("%s:%s:%d" % (mod, getattr(code, "co_qualname", code.co_name), ln) if code is not None else "?", c)
                      for (code, ln, mod), c in
                      collections.Counter({k: v for k, v in rec.callers.items() if k is not None}).most_common(3)]
        bounded = sorted(rec.bounded_seen)
        return {
            "bridge_id": rec.bridge_id, "organ": None, "built_by": None, "owner_chain": rec.owner_chain,
            "step_callers": [{"site": s, "steps": c} for s, c in top_caller],
            "neuron_model": rec.model, "n_neurons": rec.n, "dt_ms": rec.dt, "n_steps": rec.n_steps,
            "bounded_conductance_integration": (bounded[0] if len(bounded) == 1 else "mixed"),
            "bounded_values_seen": bounded,
            "v_min": _f(vmin[0], 2), "v_min_region": vmin[1], "v_max": _f(vmax[0], 2), "v_max_region": vmax[1],
            "n_oor_neurons": n_oor,
            "n_low_neurons": int(np.count_nonzero(rec.nlow > 0)) if rec.nlow is not None else 0,
            "n_high_neurons": int(np.count_nonzero(rec.nhigh > 0)) if rec.nhigh is not None else 0,
            "n_nonfinite_neurons": int(np.count_nonzero(rec.nnonf > 0)) if rec.nnonf is not None else 0,
            "n_steps_with_oor": rec.steps_with_oor,
            "n_spikes_total": rec.n_spikes_total, "n_spikes_from_oor_cells": rec.n_spikes_oor_cells,
            "alive_at_end": bridge is not None,
            "regions": regions, "oor_neurons": rows, "oor_neurons_truncated": truncated,
        }

    def report(self):
        bridges = [b for b in (self.bridge_report(r) for r in self.recs) if b is not None]
        relabel(bridges)
        return {"census_version": CENSUS_VERSION,
                "thresholds": {"v_low_mV": self.v_low, "vpeak_margin_mV": self.margin,
                               "high_check": "Izhikevich: per-neuron max(cp_izh_vpeak, homeostatic threshold) + margin; AdEx: adex_V_peak + margin; "
                                             "HH and resonate-and-fire: none"},
                "total_bridge_steps": self.total_steps, "n_bridges": len(bridges),
                "observer_errors": self.observer_errors, "first_observer_error": self.first_error,
                "observe_s": round(self.observe_s, 2),
                "bridges": bridges, "organs": organ_rows(bridges)}


def _stepper_label(callers):
    """module:Class (or module:function) of the code that stepped the bridge most. One builder can make several bridges
    that different organs step (the merged pool), so the stepper names the organ; the builder is kept as built_by."""
    if not callers:
        return None
    base = callers[0]["site"].rsplit(":", 1)[0]
    mod, _, q = base.partition(":")
    if "." in q:
        q = q.rsplit(".", 1)[0]
    return "%s:%s" % (mod, q)


def relabel(bridges):
    """(Re)derive each bridge's `organ` (stepper) and `built_by` from the call-site data stored with it. Idempotent."""
    for b in bridges:
        chain = b.get("owner_chain") or []
        b["built_by"] = chain[0].rsplit(":", 1)[0] if chain else None
        b["organ"] = _stepper_label(b.get("step_callers")) or b["built_by"] or "?"
    return bridges


def organ_rows(bridges):
    """One summary row per organ label (the owner that built the bridge)."""
    by = collections.OrderedDict()
    for b in bridges:
        by.setdefault(b["organ"], []).append(b)
    rows = []
    for organ, bs in by.items():
        vmins = [(b["v_min"], b["v_min_region"]) for b in bs if b["v_min"] is not None]
        vmin = min(vmins) if vmins else (None, None)
        vmaxs = [b["v_max"] for b in bs if b["v_max"] is not None]
        oor_regions = collections.OrderedDict()
        c_min, c_min_region = None, None
        gi_max, gi_max_region = None, None
        gt_max, gt_max_region = None, None
        for b in bs:
            for row in b["oor_neurons"]:
                d = oor_regions.setdefault(row["region"], {"n": 0, "C": []})
                d["n"] += 1
                d["C"].append(row["C"])
            for name, rg in b["regions"].items():
                if rg["c_min"] is not None and (c_min is None or rg["c_min"] < c_min):
                    c_min, c_min_region = rg["c_min"], name
                if rg["max_gi_dt_over_C"] is not None and (gi_max is None or rg["max_gi_dt_over_C"] > gi_max):
                    gi_max, gi_max_region = rg["max_gi_dt_over_C"], name
                gt = rg.get("max_gtot_dt_over_C")
                if gt is not None and (gt_max is None or gt > gt_max):
                    gt_max, gt_max_region = gt, name
        bounded = {str(b["bounded_conductance_integration"]) for b in bs}
        rows.append({"organ": organ, "n_bridges": len(bs), "neuron_models": sorted({b["neuron_model"] for b in bs}),
                     "n_neurons": sum(b["n_neurons"] for b in bs), "n_steps": sum(b["n_steps"] for b in bs),
                     "v_min": vmin[0], "v_min_region": vmin[1], "v_max": (max(vmaxs) if vmaxs else None),
                     "n_oor_neurons": sum(b["n_oor_neurons"] for b in bs),
                     "n_low_neurons": sum(b["n_low_neurons"] for b in bs),
                     "n_high_neurons": sum(b["n_high_neurons"] for b in bs),
                     "n_nonfinite_neurons": sum(b["n_nonfinite_neurons"] for b in bs),
                     "n_spikes_total": sum(b.get("n_spikes_total", 0) for b in bs),
                     "n_spikes_from_oor_cells": sum(b.get("n_spikes_from_oor_cells", 0) for b in bs),
                     "oor_regions": oor_regions,
                     "c_min": c_min, "c_min_region": c_min_region,
                     "max_gi_dt_over_C": gi_max, "max_gi_dt_over_C_region": gi_max_region,
                     "max_gtot_dt_over_C": gt_max, "max_gtot_dt_over_C_region": gt_max_region,
                     "built_by": sorted({b["built_by"] for b in bs if b.get("built_by")}),
                     "bounded_conductance_integration": (sorted(bounded)[0] if len(bounded) == 1 else "mixed")})
    return rows


def summary_line(row, tag=""):
    oor = "; ".join("%s x%d C=%s" % (r, d["n"], ",".join("%.2f" % c for c in d["C"][:6] if c is not None))
                    for r, d in row["oor_regions"].items()) or "-"
    return ("%s%s | bridges=%d n=%d steps=%d | v_min=%s (%s) v_max=%s | oor_neurons=%d (low %d high %d nonfinite %d) [%s] | "
            "spikes %d, of which %d from cells after leaving the range | min_C=%s (%s) max_(g_i+g_e)*dt/C=%s (%s) | bounded=%s" % (
                tag, row["organ"], row["n_bridges"], row["n_neurons"], row["n_steps"], row["v_min"], row["v_min_region"],
                row["v_max"], row["n_oor_neurons"], row["n_low_neurons"], row["n_high_neurons"],
                row["n_nonfinite_neurons"], oor, row.get("n_spikes_total", 0), row.get("n_spikes_from_oor_cells", 0),
                row["c_min"], row["c_min_region"], row["max_gtot_dt_over_C"],
                row["max_gtot_dt_over_C_region"], row["bounded_conductance_integration"]))


# ══════════════════════════════════════════════ worker / shard ═════════════════════════════════════════════════════
def run_worker(seed, group, arm, out_path, labels=None):
    """Run one Stage-2 gate arm IN THIS PROCESS under the census (the gate's own run_worker is the workload) and write
    the census JSON. The env (PYTHONHASHSEED etc.) is the parent shard's job, exactly as the gate's run_shard does it."""
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners import stage2_exit_gate as SG
    t0 = time.time()
    wl_path = out_path + ".workload.json"
    census = Census()
    with census:
        rc = SG.run_worker(seed, group, arm, wl_path, labels=labels)
    restored = {"step": census._cls.__dict__["_run_one_simulation_step"] is census._orig_step,
                "init": census._cls.__dict__["__init__"] is census._orig_init}
    wl = {"runner": "research/runners/stage2_exit_gate.py", "rc": rc}
    try:
        with open(wl_path) as f:
            d = json.load(f)
        resp = d.get("responses", {})
        wl.update({"n_turns": len(resp), "wall_s": d.get("wall_s"), "import_s": d.get("import_s"),
                   "sources": d.get("sources"), "corpus_present": d.get("corpus_present"),
                   "pythonhashseed_ok": d.get("pythonhashseed_ok"), "process": d.get("process"),
                   "turn_errors": {k: v["_error"] for k, v in resp.items() if isinstance(v, dict) and "_error" in v},
                   "responses_sha256": hashlib.sha256(json.dumps(resp, sort_keys=True, default=str).encode()).hexdigest()})
    except Exception as e:
        wl["error"] = "%s: %s" % (type(e).__name__, e)
    finally:
        try:
            os.remove(wl_path)
        except OSError:
            pass
    rep = census.report()
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True, text=True,
                             timeout=20).stdout.strip() or None
    except Exception:
        sha = None
    out = {"runner": "research/runners/membrane_census.py", "seed": int(seed), "group": group, "arm": arm,
           "git_sha": sha, "workload": wl, "wrapper_restored": restored, "wall_s": round(time.time() - t0, 1),
           "caveat": CAVEAT, "census": rep}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("[membrane census] seed=%d %s/%s -> %d bridges, %d steps, observer %.1fs, %d observer errors -> %s" % (
        seed, group, arm, rep["n_bridges"], rep["total_bridge_steps"], rep["observe_s"], rep["observer_errors"], out_path),
        flush=True)
    for row in rep["organs"]:
        print(summary_line(row, tag="  "), flush=True)
    return rc


def run_shard(seed, group, arm, out_path, backend="numpy", labels=None, allow_gate_seed=False):
    """Run the census worker in a FRESH subprocess with the Stage-2 gate's frozen env (PYTHONHASHSEED must be set before
    the interpreter starts), exactly as stage2_exit_gate.run_shard does for the arm itself."""
    from research.runners import stage2_exit_gate as SG
    env = dict(os.environ)
    env.update(SG.BASE_ENV)
    env["SIM_BACKEND"] = backend
    if backend == "cupy":
        env.pop("CUDA_VISIBLE_DEVICES", None)
    part = out_path + ".part.json"
    cmd = [sys.executable, "-u", "-m", "research.runners.membrane_census", "--worker", "--seed", str(int(seed)),
           "--group", group, "--arm", arm, "--out", part]
    if labels:
        cmd += ["--labels", labels]
    if allow_gate_seed:
        cmd += ["--allow-gate-seed"]
    p = subprocess.run(cmd, env=env, cwd=_REPO)
    if p.returncode != 0 or not os.path.exists(part):
        print("[membrane census] worker FAILED rc=%s for seed=%s %s/%s" % (p.returncode, seed, group, arm), flush=True)
        return 1
    with open(part) as f:
        d = json.load(f)
    d["backend"] = backend
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(d, f, indent=1, default=str)
    for suffix in ("", ".prov.json"):
        try:
            os.remove(part + suffix)
        except OSError:
            pass
    return 0


# ══════════════════════════════════════════════ pool lines / summary ═══════════════════════════════════════════════
def pool_lines(sha="<sha>", seeds=POOL_SEEDS, out_dir=DEFAULT_OUT_DIR, arm="intact_a"):
    """One plain command line per (dev seed, Stage-2 group), in the shape of the Stage-2 pool lines (no TSV columns)."""
    from research.runners import stage2_exit_gate as SG
    lines = []
    for seed in seeds:
        for group in SG.PRIMARY_GROUPS:
            d = "%s/s%d" % (out_dir, seed)
            lines.append("cd ~/derisk-pool/revisions/%s && bash tools/mem_ok.sh %d 1 && mkdir -p %s && bash tools/memcap.sh %d "
                         "-- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                         "NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m "
                         "research.runners.membrane_census --shard --seed %d --group %s --arm %s --out %s/%s__%s.json" % (
                             sha, POOL_MEM_GB, d, POOL_MEMCAP_GB, seed, group, arm, d, group, arm))
    return lines


def summarize(paths):
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        print("== seed %s %s/%s (%s) workload=%s observer_errors=%s wrapper_restored=%s" % (
            d.get("seed"), d.get("group"), d.get("arm"), os.path.basename(p), d.get("workload", {}).get("n_turns"),
            d["census"]["observer_errors"], d.get("wrapper_restored")))
        bridges = relabel(d["census"]["bridges"])
        for row in organ_rows(bridges):
            print(summary_line(row, tag="  "))


def main():
    from research.runners import stage2_exit_gate as SG
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worker", action="store_true", help="internal: run one arm under the census in THIS process")
    ap.add_argument("--shard", action="store_true", help="run one (seed, group, arm) census in a fresh subprocess")
    ap.add_argument("--summarize", nargs="+", metavar="JSON", help="print the per-organ summary lines of result JSONs")
    ap.add_argument("--emit-pool-lines", action="store_true", help="print the dev-seed 1-12 x 5-group pool command lines")
    ap.add_argument("--sha", default="<sha>", help="revision token for --emit-pool-lines (default: the literal <sha>)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--group", default=None, choices=sorted(SG.GROUPS))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--labels", default=None, help="DEV ONLY: run only these turn labels (comma-separated)")
    ap.add_argument("--backend", default="numpy", choices=("numpy", "cupy"))
    ap.add_argument("--allow-gate-seed", action="store_true", help="a GATE seed is refused without this (diagnostic probe)")
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize)
        return
    if a.emit_pool_lines:
        print("\n".join(pool_lines(a.sha)))
        return
    if a.worker or a.shard:
        if a.seed is None or not a.group or not a.arm or not a.out:
            ap.error("--seed, --group, --arm and --out are required")
        if a.seed in SG.GATE_SEEDS and not a.allow_gate_seed:
            sys.stderr.write("GATE seed %d refused: this census is a dev-seed diagnostic (dev seeds: 1-12).\n" % a.seed)
            sys.exit(2)
        if a.worker:
            sys.exit(run_worker(a.seed, a.group, a.arm, a.out, labels=a.labels))
        sys.exit(run_shard(a.seed, a.group, a.arm, a.out, backend=a.backend, labels=a.labels,
                           allow_gate_seed=a.allow_gate_seed))
    ap.error("choose one of --worker/--shard/--summarize/--emit-pool-lines")


if __name__ == "__main__":
    main()
