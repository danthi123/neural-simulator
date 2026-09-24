"""BRAIN_XEDGE_IN_WAVE3 (default OFF) — grow the d6 w{k}->sel cross-edge INSIDE the production merged cortical pool.

THE SEVERANCE THIS CLOSES (measured, not read off code: S02 probe, `research/findings/raw/_xedge_wave3_probe/s42.json`,
pool1 @8e1bbc470, shipped defaults). `comprehension_production_organ.get_organ()` resolves the Wave-3 merged pool
FIRST (production default), so the live comprehension organ rides the 11-organ MergedPool. The per-session d6
organ (`webapp.server._get_multiref_organ`) takes `shared=get_xedge_pool(seed).pool` -- a SEPARATE 1752-neuron
[d6 + comprehension + da_credit] pool that carries the learned w{k}->sel cross-edge. The live comprehension read
never touches that pool, so the cross-edge has no path to a reply: live cross-organ synapses in production = 0.
(The comprehension read's positional co-drive `wm_focus="w0"` lands on the WAVE-3 pool's own d6 w0 slice, which no
session holds and from which no synapse reaches sel.)

WHAT THIS MODULE BUILDS (additive; NO `sim/` edit; the comprehension `get_organ()` resolution order is UNCHANGED):
  1. ONE pool: the shipped Wave-3 11-organ reconciliation (`_wave3_descriptors()`, reuse-by-import, unchanged) +
     the R3 `da_credit` coincidence-detector organ (teach_agent/teach_patient/snc_a/snc_b -- the R3 spec by import,
     its 4 regions gain-0 FROZEN like every other organ's) + DECLARED `CrossEdge`s: the 6 plastic w{0,1,2}->
     sel_{agent,patient} candidate edges at W0=0.05 on gate `wm_to_sel_r2`, and R3's 4 fixed sel/teach->snc edges.
     Both endpoints of every new edge sit in frozen regions, so the pool's live Hebbian (surprise/world-model) can
     never move them; the global config union is UNCHANGED (no new config key).
  2. `Wave3XedgeView` -- the R3Pool surface the production xedge holder consumes (`amb_read`, `_episode`,
     `cross_weights`, `ix`, `masks`, `comp_organ`, ...), run ON THE MERGED POOL. Every view operation runs inside
     `pool.sequence_isolation()` (restores every per-neuron / per-synapse transient array, the timing counters and
     the global RNG cursor on exit; WEIGHTS persist -- that is the learning). A credited `_episode` additionally opens
     a LOCAL plasticity window: R3's exact STDP/three-factor config values, a locally-installed dopamine
     NeuromodulatorManager sourced from snc_a/snc_b firing (the curiosity organ's shipped local-neuromodulator
     pattern), the eligibility/last-spike arrays, and a whitelist gain vector (only `wm_to_sel_r2` open) -- all
     restored on exit. So the credit VALUE is still the coincidence population's own spikes (R3's mechanism,
     unchanged), and no other organ's synapse can move.
  3. PER-SESSION SLICE OWNERSHIP + HARD RESET. The per-session d6 organs now share the pool's d6 slice. Each session
     keeps its own codebook / binder / focus (per-instance, unchanged); the slice's TRANSIENT state (held bumps, NMDA
     rise buffers, refractory, pulse timers) belongs to exactly one session at a time. `MultiReferentWMOrgan.load()`
     announces its owner (`shared.xedge_session_enter`); when the owner changes, the d6 + da_credit slices are
     hard-reset to the pool's quiescent snapshot before the new session writes. The comprehension read already
     hard-resets the whole bridge before every read and the credit/amb reads run under sequence isolation, so they
     carry nothing across turns either.
  4. Routing (both DEFAULT OFF, `onebrain_xedge_wave3_flags.xedge_in_wave3_enabled`): `get_merged_cortical_pool`
     returns THIS pool, and `get_xedge_pool` builds its holder from THIS view -- so `get_organ()` (unchanged order),
     the per-session d6 `shared=`, the co-drive, the WM-resolved role read and the per-turn credit hook all land on
     ONE object. Unset -> neither routing point imports this module -> byte-identical to main.

DECLARED RESIDUALS (carried unchanged from the shipped xedge, none added):
  * The candidate topology (w0/w1/w2) and the live focus are POSITIONAL (`CAND_POOLS[0]`), not a semantic
    referent->pool binding; the co-drive is a host-timed 400 pA re-drive of that pool before the read.
  * WHICH discourse is presented (the teach drive direction's source text) is host/teacher scaffold; the credit
    value and direction are read off the brain's own amb_read resolution.
  * The learned cross-edge weight is PROCESS-shared (one brain learns from every conversation), exactly as the
    separate xedge pool's was. Only transient activity is session-owned.
  * Turn handling is assumed sequential (one bridge); concurrent turns on one bridge were never supported.
"""
from __future__ import annotations

import contextlib
import dataclasses

import numpy as np

from research.runners.onebrain_xedge_wave3_flags import xedge_in_wave3_enabled  # noqa: F401  (re-export)

DA_KEY = "da_credit"
DA_REGIONS = ("teach_agent", "teach_patient", "snc_a", "snc_b")


def _r3():
    """The R3 module, imported the way the production xedge build imports it: the R3-v3 module FIRST (its import
    raises R3's DA_SENSITIVITY 60 -> 10000, the functional-drive calibration production has always run with)."""
    import research.runners._onebrain_integration_r3v3_functional_drive  # noqa: F401  (side effect: DA_SENSITIVITY)
    import research.runners._onebrain_integration_r3_spiking_dopamine_credit as R3
    return R3


def _r2():
    import research.runners._onebrain_integration_r2_threefactor_selforganized as R2
    return R2


def da_credit_descriptor():
    from research.runners.onebrain_merge_framework import OrganDescriptor
    R3 = _r3()
    return OrganDescriptor(key=DA_KEY, regions=DA_REGIONS, spec_fn=R3._spec_da_credit, config={},
                           freeze_regions=DA_REGIONS,
                           scaffold_residuals=("teach_{agent,patient} drive = teacher/environment input (R3)",))


def cross_edges():
    """The R3 cross-edge set as declared `CrossEdge` rows, in R3's own union order (6 candidate edges, then the
    4 fixed coincidence edges)."""
    from research.runners.onebrain_merge_framework import CrossEdge
    R2, R3 = _r2(), _r3()
    edges = []
    for p in R2.CAND_POOLS:
        for tgt, suf in (("sel_agent", "sela"), ("sel_patient", "selp")):
            edges.append(CrossEdge(key=f"x_{p}_{suf}", source_key="d6_multiref_wm", source_region=p,
                                   target_key="comprehension", target_region=tgt, init_weight=float(R2.W0),
                                   plastic=True, gate=R2.GATE, freeze_rest=False, learn_rule="da_credit"))
    for key, src_k, src, tgt, w in (("teachA_snca", DA_KEY, "teach_agent", "snc_a", R3.W_TEACH),
                                    ("selA_snca", "comprehension", "sel_agent", "snc_a", R3.W_SEL),
                                    ("teachP_sncb", DA_KEY, "teach_patient", "snc_b", R3.W_TEACH),
                                    ("selP_sncb", "comprehension", "sel_patient", "snc_b", R3.W_SEL)):
        edges.append(CrossEdge(key=key, source_key=src_k, source_region=src, target_key=DA_KEY, target_region=tgt,
                               init_weight=float(w), plastic=False, freeze_rest=False, learn_rule="none"))
    return edges


def wave3_xedge_descriptors():
    from research.runners._onebrain_wave3_organread_verify import _wave3_descriptors
    return list(_wave3_descriptors()) + [da_credit_descriptor()]


def build_wave3_xedge_pool(seed: int):
    """A fresh (un-memoized) flag-ON pool -- the verify runners build it directly."""
    from research.runners.onebrain_merge_framework import merge_organs
    return merge_organs(wave3_xedge_descriptors(), int(seed), wire=True, cross_edges=cross_edges())


_POOL: dict = {}
_VIEW: dict = {}


def get_wave3_xedge_pool(seed: int = 42):
    """The process-shared flag-ON merged pool (memoized by seed). Reached from `get_merged_cortical_pool` only when
    `xedge_in_wave3_enabled()`. The view (and its coupling handles on the pool object) is installed at build, so
    whichever of comprehension / d6 / the xedge holder touches the pool first finds the handles already present."""
    key = int(seed)
    if key not in _POOL:
        pool = build_wave3_xedge_pool(key)
        _POOL[key] = pool
        _VIEW[key] = Wave3XedgeView(pool, key)
    return _POOL[key]


def get_wave3_xedge_view(seed: int = 42) -> "Wave3XedgeView":
    get_wave3_xedge_pool(seed)
    return _VIEW[int(seed)]


def _r3_plasticity_values(seed: int) -> dict:
    """R3Pool's EFFECTIVE plasticity operating point: R3 `_build_pool`'s config union (base + d6/comprehension
    REGISTRY configs minus enable_stdp/enable_reward_modulation + R3's `extra` descriptor) followed by
    `R3Pool.__init__`'s post-build overrides. Computed WITHOUT building a bridge. Applied only inside a credited
    episode's local plasticity window."""
    from research.runners.onebrain_merge_framework import REGISTRY, _base_config
    R2 = _r2()
    strip = ("enable_stdp", "enable_reward_modulation")
    union = {}
    for key in ("d6_multiref_wm", "comprehension"):
        union.update({k: v for k, v in REGISTRY[key].config.items() if k not in strip})
    union.update({"enable_stdp": True, "enable_reward_modulation": True, "reward_defer_stdp_weight_update": True})
    cfg = _base_config(int(seed))
    for k, v in union.items():
        setattr(cfg, k, v)
    cfg.stdp_w_max = R2.HMAX
    cfg.reward_learning_rate = R2.REWARD_LR
    cfg.reward_eligibility_tau_ms = R2.REWARD_TAU_MS
    cfg.reward_baseline = 0.0
    cfg.enable_hebbian_learning = False
    names = [f.name for f in dataclasses.fields(cfg)
             if f.name.startswith(("stdp_", "reward_")) or f.name in (
                 "enable_stdp", "enable_reward_modulation", "enable_inhibitory_stdp", "enable_d1_d2_asymmetry",
                 "enable_hebbian_learning")]
    out = {n: getattr(cfg, n) for n in names}
    out["current_reward_signal"] = 0.0
    return out


class Wave3XedgeView:
    """R3Pool's surface, run on the merged pool (see module docstring item 2)."""

    def __init__(self, pool, seed: int):
        R2 = _r2()
        from sim.backend import to_host
        self.seed = int(seed)
        self.mode = "intact"
        self.pool = pool
        pool.ensure_built()
        self.b = pool.bridge
        self.xp = pool.xp
        self.p_agent, self.p_patient = R2._role_assignment(self.seed)[:2]   # the 3rd (uncredited) pool is unused here
        rm = self.b.region_manager
        names = list(R2.CAND_POOLS) + [R2.BASE_POOL, "sel_agent", "sel_patient", "fs",
                                       "cue_animacy_pos", "cue_animacy_neg", "cue_verbfit_pos", "cue_verbfit_neg",
                                       "teach_agent", "teach_patient", "snc_a", "snc_b"]
        self.ix = {nm: np.asarray(rm.indices(nm), np.int64) for nm in names}
        coo = self.b.cp_connections.tocoo()
        row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
        self.masks = {}
        for p in R2.CAND_POOLS:
            self.masks[f"{p}->A"] = np.isin(row, self.ix[p]) & np.isin(col, self.ix["sel_agent"])
            self.masks[f"{p}->P"] = np.isin(row, self.ix[p]) & np.isin(col, self.ix["sel_patient"])
        self.da_masks = {
            "teach_agent->snc_a": np.isin(row, self.ix["teach_agent"]) & np.isin(col, self.ix["snc_a"]),
            "sel_agent->snc_a": np.isin(row, self.ix["sel_agent"]) & np.isin(col, self.ix["snc_a"]),
            "teach_patient->snc_b": np.isin(row, self.ix["teach_patient"]) & np.isin(col, self.ix["snc_b"]),
            "sel_patient->snc_b": np.isin(row, self.ix["sel_patient"]) & np.isin(col, self.ix["snc_b"]),
        }
        snap = pool.snap
        self.rest_v = np.asarray(to_host(snap["cp_membrane_potential_v"])).copy()
        self.rest_u = np.asarray(to_host(snap["cp_recovery_variable_u"])).copy()
        self._plast = _r3_plasticity_values(self.seed)
        # the candidate gate is FROZEN between credited steps (the pool's gain-0 freeze already zeroed it -- both
        # endpoints are frozen regions -- this makes the gate value explicit for set_plasticity_gate callers).
        self.b.set_plasticity_gate(R2.GATE, 0.0)
        # PER-SESSION SLICE OWNERSHIP (module docstring item 3)
        n = int(self.b.cp_membrane_potential_v.shape[0])
        sess = np.zeros(n, dtype=bool)
        for key in ("d6_multiref_wm", DA_KEY):
            for r in pool.organ_regions.get(key, ()):
                sess[np.asarray(rm.indices(r), np.int64)] = True
        self._session_mask_host = sess
        self._session_mask = self.xp.asarray(sess)
        self._session_syn_mask = self.xp.asarray(np.isin(row, np.flatnonzero(sess)))
        self._owner = None
        self.n_session_resets = 0
        self._comp_organ = None
        self._d6_organ = None
        # publish the coupling handles onto the pool object (the organs read them off `shared=`)
        from research.runners.onebrain_xedge_production import _CODRIVE_PARAMS
        pool.xedge_focus = None
        pool.xedge_codrive_params = dict(_CODRIVE_PARAMS)
        pool.xedge_amb_read = self.amb_read
        pool.xedge_balanced_cues = [("cue_animacy_pos", R2.AMBIG_PA), ("cue_animacy_neg", R2.AMBIG_PA)]
        pool.xedge_base_pool = R2.BASE_POOL
        pool.xedge_session_enter = self.session_enter
        pool.xedge_in_wave3 = True

    # ── organs on this pool (the PRODUCTION comprehension organ itself, so there is one comprehension organ) ──
    @property
    def comp_organ(self):
        if self._comp_organ is None:
            from research.runners import comprehension_production_organ as CO
            org = CO.get_organ(self.seed)
            if getattr(org, "_shared", None) is not self.pool:     # defensive: never read another pool's organ
                org = CO.ComprehensionProductionOrgan(seed=self.seed, shared=self.pool)
            org.ensure_built()
            self._comp_organ = org
        return self._comp_organ

    @property
    def d6_organ(self):
        if self._d6_organ is None:
            from research.runners.onebrain_merge_framework import _d6_organ
            self._d6_organ = _d6_organ(self.seed, self.pool)
        return self._d6_organ

    # ── per-session ownership of the d6 + da_credit slices ──
    def session_enter(self, owner) -> bool:
        """Called by `MultiReferentWMOrgan.load()` before it writes. Returns True iff a hard reset ran (the owner
        changed). The first owner inherits the quiescent build state (nothing to reset)."""
        oid = id(owner)
        reset = self._owner is not None and self._owner != oid
        if reset:
            self.hard_reset_session_slices()
            self.n_session_resets += 1
        self._owner = oid
        return bool(reset)

    def hard_reset_session_slices(self):
        """Return the d6 + da_credit slices' transient state to the pool's quiescent snapshot; every other organ's
        slice is untouched."""
        b, m = self.b, self._session_mask
        snap = self.pool.snap or {}
        zero_names = ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab", "cp_conductance_g_nmda",
                      "cp_conductance_g_nmda_rise", "cp_conductance_g_nmda_recurrent",
                      "cp_conductance_g_nmda_recurrent_rise", "cp_conductance_g_gabab_slow",
                      "cp_conductance_g_coincidence", "cp_conductance_g_coincidence_rise", "cp_firing_states",
                      "cp_prev_firing_states", "cp_refractory_timers", "cp_external_input_current")
        n = int(b.cp_membrane_potential_v.shape[0])
        for nm in set(zero_names) | set(snap.keys()):
            arr = getattr(b, nm, None)
            if arr is None or getattr(arr, "shape", (None,))[0] != n:
                continue
            if nm in snap and snap[nm] is not None and snap[nm].shape == arr.shape:
                arr[m] = self.xp.asarray(snap[nm])[m]
            else:
                arr[m] = 0
        nnz = int(b.cp_connections.nnz)
        for nm in ("cp_synapse_pulse_timers", "cp_synapse_pulse_progress"):
            arr = getattr(b, nm, None)
            if arr is not None and arr.shape[0] == nnz:
                arr[self._session_syn_mask] = 0

    # ── R3 primitives on the merged pool ──
    def _hard_reset(self):
        R2 = _r2()
        b, xp = self.b, self.xp
        b.cp_membrane_potential_v[:] = xp.asarray(self.rest_v)
        b.cp_recovery_variable_u[:] = xp.asarray(self.rest_u)
        for nm in R2._CONDUCT:
            a = getattr(b, nm, None)
            if a is not None:
                a[:] = 0
        if getattr(b, "cp_firing_states", None) is not None:
            b.cp_firing_states[:] = False
        if getattr(b, "cp_last_spike_time", None) is not None:
            b.cp_last_spike_time[:] = -1000.0
        if getattr(b, "cp_eligibility_trace", None) is not None:
            b.cp_eligibility_trace[:] = 0.0
        b.cp_external_input_current[:] = 0.0
        nm = getattr(b, "neuromodulator_manager", None)
        if nm is not None and "dopamine" in nm.modulator_names():
            nm.set_concentration("dopamine", 0.0)
            nm._rule_state["dopamine"] = {"err_ema": 0.0, "rate_ema": 0.0, "signed_rate_ema": 0.0}

    def _drive(self, pairs, steps, read=None):
        from sim.backend import to_host
        b, xp = self.b, self.xp
        cur = xp.zeros(b.core_config.num_neurons, dtype=xp.float32)
        for idx, pa in pairs:
            cur[xp.asarray(idx)] = xp.float32(pa)
        acc = {k: 0.0 for k in (read or {})}
        for _ in range(steps):
            b.cp_external_input_current[:] = cur
            b._run_one_simulation_step()
            b.runtime_state.current_time_ms += b.core_config.dt_ms
            if read:
                fs = b.cp_firing_states
                for k, idx in read.items():
                    acc[k] += float(to_host(fs[xp.asarray(idx)].astype(xp.float64).sum())) / idx.size
        b.cp_external_input_current[:] = 0.0
        return {k: v / steps for k, v in acc.items()}

    def _wmean(self, name):
        from sim.backend import to_host
        return float(np.asarray(to_host(self.b.cp_connections.data))[self.masks[name]].mean())

    def cross_weights(self):
        return {k: round(self._wmean(k), 4) for k in self.masks}

    @contextlib.contextmanager
    def _plasticity_window(self):
        """R3's three-factor operating point, LOCAL to one credited episode (restored on exit)."""
        R2, R3 = _r2(), _r3()
        from sim.neuromodulators import NeuromodulatorManager
        b, xp = self.b, self.xp
        cfg = b.core_config
        keys = list(self._plast) + ["enable_neuromodulator_subsystem", "neuromodulators"]
        _missing = object()
        saved_cfg = {k: getattr(cfg, k, _missing) for k in keys}
        saved_nm = getattr(b, "neuromodulator_manager", None)
        saved_elig = getattr(b, "cp_eligibility_trace", None)
        saved_lst = getattr(b, "cp_last_spike_time", None)
        saved_gain = b.cp_plasticity_rate_gain.copy()
        gate_val = float(b._plasticity_gate_values.get(b._canonicalize_gate_name(R2.GATE), 0.0))
        n = int(b.cp_membrane_potential_v.shape[0])
        nnz = int(b.cp_connections.nnz)
        try:
            for k, v in self._plast.items():
                setattr(cfg, k, v)
            cfg.enable_neuromodulator_subsystem = True
            cfg.neuromodulators = [R3._dopamine_cfg()]
            mgr = NeuromodulatorManager(cfg.neuromodulators, cfg.dt_ms)
            mgr.initialize(n, xp)
            if b.region_manager is not None:
                mgr.set_group_indices(b.region_manager.region_indices_dict())
            b.neuromodulator_manager = mgr
            if saved_elig is None:
                b.cp_eligibility_trace = xp.zeros(nnz, dtype=xp.float32)
            if saved_lst is None:
                b.cp_last_spike_time = xp.full(n, -1000.0, dtype=xp.float32)
            b.cp_plasticity_rate_gain[:] = 0.0                  # whitelist: ONLY the candidate cross-edge gate
            b.set_plasticity_gate(R2.GATE, gate_val)            # the caller's value (1.0 credited; 0.0 = lesion)
            yield
        finally:
            b.cp_plasticity_rate_gain[:] = saved_gain
            b.neuromodulator_manager = saved_nm
            b.cp_eligibility_trace = saved_elig
            b.cp_last_spike_time = saved_lst
            for k, v in saved_cfg.items():
                if v is _missing:
                    if hasattr(cfg, k):
                        try:
                            delattr(cfg, k)
                        except AttributeError:
                            pass
                else:
                    setattr(cfg, k, v)

    def _episode(self, pool_key, cue_pairs, credited, teach_pool):
        """R3Pool._episode on the merged pool, under sequence isolation + the local plasticity window."""
        R2, R3 = _r2(), _r3()
        with self.pool.sequence_isolation(), self._plasticity_window():
            self._hard_reset()
            self._drive([(self.ix[pool_key], R2.LOAD_PA)], R2.LOAD_STEPS)
            pairs = list(cue_pairs)
            if credited and teach_pool is not None:
                pairs.append((self.ix[teach_pool], R3.TEACH_PA))
            self._drive(pairs, R2.TRAIN_STEPS)

    def amb_read(self, hold_pool_key, cue_pairs, band=None):
        """R3Pool.amb_read on the merged pool (a frozen read; sequence-isolated)."""
        R2 = _r2()
        ix = self.ix
        with self.pool.sequence_isolation():
            margins, rates = [], {"sel_agent": 0.0, "sel_patient": 0.0, self.p_agent: 0.0, self.p_patient: 0.0,
                                  "fs": 0.0, "cue_animacy_pos": 0.0}
            for _ in range(R2.N_READS):
                self._hard_reset()
                if hold_pool_key is not None:
                    self._drive([(ix[hold_pool_key], R2.LOAD_PA)], R2.LOAD_STEPS)
                    self._drive([], 6)
                read = {"A": ix["sel_agent"], "P": ix["sel_patient"]}
                if band is not None:
                    for r in rates:
                        read[r] = ix[r]
                acc = self._drive([(ix[k], pa) for k, pa in cue_pairs], R2.READ_STEPS, read=read)
                margins.append(acc["A"] - acc["P"])
                if band is not None:
                    for r in rates:
                        rates[r] += acc[r]
        out = {"margin": float(np.mean(margins))}
        if band is not None:
            out["rates"] = {r: rates[r] / R2.N_READS for r in rates}
        return out

    def train(self, *a, **k):
        raise NotImplementedError("BRAIN_XEDGE_IN_WAVE3 supports the LEARNED edge only (PART-1 frozen host-schedule "
                                  "training stays on the separate xedge pool; xedge_in_wave3_enabled() is False "
                                  "when BRAIN_ONEBRAIN_XEDGE_LEARN=0)")

    def plasticity_values(self) -> dict:
        return {k: (v if isinstance(v, (bool, int, float, str)) or v is None else repr(v))
                for k, v in self._plast.items()}
