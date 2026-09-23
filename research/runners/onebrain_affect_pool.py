"""D3 ONE-BRAIN — the AFFECT organ (Gate-B graded mood ladder) migrated onto the shared cortical pool, with a
REAL cross-region synapse (the ladder's own held AROUSAL -> the D2 surprise pool; the LC-NE projection is its
biological motivation -- whether it acts as a GAIN or as additive DC drive on surprise is NOT tested, no claim).

WHY AFFECT IS THE NEXT ORGAN (verify-first inventory, 2026-09-23). The 11-organ Wave-3 pool
(`onebrain_wave3_pool_production.get_wave3_pool`, default-ON since 8ee5e6817) carries the cortical organs
(surprise / world-model / metacog / pragmatic / comprehension / source_provenance / self_schema / curiosity /
causal_whatif / prospective_memory / d6_multiref_wm). Of the charter-D3 core set, AFFECT is the one that still
runs on its OWN bridge in production: `affect_production_organ.AffectProductionOrgan.read_differential` dispatches
(default-ON interoceptive path) to `_appraisal_interoceptive_ladder_derisk.get_ladder(seed)` — a dedicated
standalone SimulationBridge (24 staggered-bistable rungs + readouts + opponent aggregates + 3 interoceptive relay
pools, ~690 neurons). The other residual non-pool organs are either per-SESSION (d6 / prospective_memory wrappers
— a process-shared pool cannot host them without a cross-session leak) or need a new online re-open mechanism
(source_provenance's incremental `encode_fact`), so affect has the best readiness: a process-shared singleton,
already proven co-residence-safe in kind (2026-08-13 per-region-OU affect merge GO 6/6), and its OU seam has a
shipped engine answer (`cfg.per_neuron_ou_seed`, the curiosity organ's local-OU read pattern).

WHAT THIS MODULE BUILDS (additive, DEFAULT-OFF, NO `sim/` edit):
  1. `AFFECT_DESCRIPTOR` — the ladder as a merge-framework `OrganDescriptor`: the EXACT region/pathway spec the
     production ladder uses (reused by import from `_stageA_full_integration_derisk` + the relay constants from
     `_appraisal_interoceptive_ladder_derisk`, same `dataclasses.replace(internal_density=LAD_RECUR_DENSITY)`
     restore and the same `speak_acc` exclusion). Name-keyed per-region param-het (`param_het=True`, the ladder's
     standalone uses global param-het). No region-name collision with any Wave-3 organ (every region is `aff_*`
     or `appr_intero_*`). All its pathways are `plastic=False` and every region `plastic_internal=False`, so the
     pool's global Hebbian flag cannot move a ladder weight — VERIFIED (not assumed) by the verify runner's
     weights-unchanged check, not by a gain-0 freeze (a freeze would forbid the cross-edge below, which has
     exactly one endpoint in the ladder).
  2. `PoolAffectLadder` — the SAME settle/ramp/drive-off/read protocol as `AppraisalInteroceptiveLadder.
     read_differential`, run on the pool slice. The ladder needs OU background noise (8 pA) that the frozen pool
     keeps OFF; it is built LOCALLY for the read window with the per-neuron-keyed OU stream (co-residence-
     invariant: a neuron's draw keys on (region, within-region rank)), inside `sequence_isolation` (restores every
     per-neuron + per-synapse array, the timing counters and the global RNG cursor on exit) — the curiosity
     organ's shipped local-OU pattern. The appraisal still enters ONLY as a current on the relay pools; the rungs
     are reached only through synapses (the anti-cheat assertion is kept).
  3. `AROUSAL_XEDGE` — the INTEGRATION synapse (not co-location): every held AROUSAL rung neuron
     (`aff_arousal_L1..L8`, 160 cells) projects all-to-all onto the D2 `surprise` pool, fixed weight
     (`plastic=False`), behind the transmission gate `affect_arousal_to_surprise` (the lesion handle). This is
     the rank-#3 cross-edge of the integration design (arousal -> surprise, Aston-Jones & Cohen 2005 LC-NE
     adaptive gain as the motivation) — but the 2026-09-02 de-risk drove a HOST-driven stand-in arousal source;
     here the source is the brain's OWN latched arousal ladder, so the extra excitatory drive onto surprise is set
     by the affect state the message appraisal left behind, carried end-to-end by spikes. (Gain vs additive DC is
     untested: at the tested weight confirm sits at a ~0 Hz floor, which a sub-threshold DC shift would also
     leave unchanged. No gain claim is made.)
  4. Production routing (both DEFAULT-OFF): `BRAIN_ONEBRAIN_AFFECT_POOL=1` -> `get_merged_cortical_pool` returns
     the 12-organ pool (every wired cortical organ still resolves to ONE pool object) AND the production affect
     read runs on it; `BRAIN_ONEBRAIN_AFFECT_XEDGE=1` (only meaningful with the pool flag) additionally installs
     the arousal->surprise synapse. Unset -> nothing here is imported by production -> byte-identical to main.

HONEST SCOPE / RESIDUALS (declared):
  * The production surprise read does a whole-bridge `_hard_reset` + `read_isolation`, and the affect read runs
    inside `sequence_isolation`; per-organ reads are ISOLATED, so in a live turn the held arousal of the affect
    read does not yet persist into the surprise read. The cross-edge is therefore verified at the MECHANISM level
    (ladder held -> surprise read in ONE continuous sequence on the one pool), not yet load-bearing on a chat
    reply. Persisting the affect state across organ reads (a turn-scoped shared-state protocol instead of
    per-organ isolation) is the named next rung.
  * The appraisal VALUE (Warriner-gated, DR-2 learned) and its injection current onto the relays remain the
    declared host sensory boundary (unchanged from production).
"""
from __future__ import annotations

import contextlib
import dataclasses as _dc

import numpy as np

from research.runners.onebrain_merge_framework import OrganDescriptor, CrossEdge
from research.runners import _stageA_full_integration_derisk as SA
from research.runners._appraisal_interoceptive_ladder_derisk import (
    APPR_INTERO_GATE, APPR_INTERO_N, APPR_INTERO_DENS, APPR_INTERO_I_PA, APPR_INTERO_W,
)

AFFECT_KEY = "affect_ladder"
XEDGE_KEY = "affect_arousal_to_surprise"
XEDGE_GATE = "affect_arousal_to_surprise"
# Per-synapse weight of the diffuse arousal->surprise projection. A HAND-SET CONSTANT. PROVENANCE (corrected in the
# 2026-09-23 fix round): 0.05 was hard-coded in the build commit 1fe3f56ef (08:59:38) while the seed-42 calibration
# (`--calibrate`, started 08:55:41, ~1268 s) was still running; the calibration result (f77556db, 09:17) then found
# 0.05 to be the smallest weight in its sweep that met its rule -- AGREEMENT after the fact, not a derivation. That
# calibration also measured the surprise organ at a NON-production operating point (the ladder's OU on every pool
# neuron), so it cannot license this value either. The 6-seed gate runs at this fixed hand-set value on all seeds.
XEDGE_W = 0.05
_AFF_N_RUNGS = 8
_RELAY_NAMES = {"vplus": "appr_intero_vplus", "vminus": "appr_intero_vminus", "arousal": "appr_intero_arousal"}


from research.runners.onebrain_affect_pool_flags import affect_pool_enabled, affect_xedge_enabled  # noqa: F401,E402


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  1. THE DESCRIPTOR — the production ladder's exact spec as data.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _ladder_spec(aff_n_rungs=_AFF_N_RUNGS, w=APPR_INTERO_W, dens=APPR_INTERO_DENS):
    """(regions, pathways, names) — byte-for-byte the region/pathway list `AppraisalInteroceptiveLadder.__init__`
    builds (reuse-by-import of the SEAM-C spec + the same relay pools/pathways), minus nothing else."""
    from sim.regions import BrainRegion, RegionPathway
    RS = "IZH2007_RS_CORTICAL_PYRAMIDAL"
    lad_regions, names = SA._ladder_region_specs(int(aff_n_rungs))
    rung_names = set(names["vplus"] + names["vminus"] + names["arousal"])
    lad_regions = [_dc.replace(r, internal_density=SA.LAD_RECUR_DENSITY) if r.name in rung_names else r
                   for r in lad_regions]
    lad_pathways = [p for p in SA._ladder_pathways(names) if p.to_region != "speak_acc"]

    def relay(name):
        return BrainRegion(name=name, n_neurons=APPR_INTERO_N, exc_fraction=1.0, internal_density=0.0,
                           exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.05,
                           plastic_internal=False, izh_neuron_type=RS, enable_nmda=False)

    relay_regions = [relay(n) for n in _RELAY_NAMES.values()]
    relay_pathways = []
    for sign, rname in _RELAY_NAMES.items():
        for rung in names[sign]:
            relay_pathways.append(RegionPathway(from_region=rname, to_region=rung, density=float(dens),
                                                weight_mean=float(w), weight_jitter=0.1, plastic=False,
                                                transmission_gate=APPR_INTERO_GATE))
    return lad_regions + relay_regions, lad_pathways + relay_pathways, names


def _spec_affect_ladder(seed):
    regions, pathways, names = _ladder_spec()
    return regions, pathways, {"names": names, "relay_names": dict(_RELAY_NAMES)}


def _affect_region_names():
    regions, _, _ = _ladder_spec()
    return tuple(r.name for r in regions)


# Config the ladder REQUIRES, unioned into the pool: NMDA on at the ladder's latch time-constant (== metacog's
# DEFAULT_NMDA_TAU 150, so no conflict), noise OFF at pool level (OU is built LOCALLY per read). Plasticity flags
# are deliberately NOT declared: the pool's global Hebbian stays as the other organs need it; every ladder edge
# is plastic=False, so it cannot learn (verified by the runner, not assumed).
_AFFECT_CONFIG = {
    "enable_nmda": True, "nmda_ratio": 0.5, "nmda_tau_decay": 150.0,   # == the standalone ladder's latch point
                                                                        # (and == metacog's union values)
    "enable_conductance_noise": False, "enable_ou_process": False, "ou_std_current_pA": 0.0,
    # the standalone ladder runs every plasticity/homeostasis rule OFF; these equal the Wave-3 union's values
    # (verified: no MergeConflict), so declaring them only matters when affect is built WITHOUT the superset.
    "enable_stdp": False, "enable_short_term_plasticity": False, "enable_homeostasis": False,
    "enable_reward_modulation": False, "enable_structural_plasticity": False,
}


def _affect_idx_fn(bridge):
    from research.runners.onebrain_merge_framework import _idx
    regs = _affect_region_names()
    return {n: np.asarray(_idx(bridge, n), dtype=np.int64) for n in regs}


def _arousal_idx(bridge):
    from research.runners.onebrain_merge_framework import _idx
    names = [f"aff_arousal_L{i + 1}" for i in range(_AFF_N_RUNGS)]
    return np.concatenate([np.asarray(sorted(int(i) for i in _idx(bridge, n)), dtype=np.int64) for n in names])


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  2. THE READ — the production protocol on the pool slice (local per-neuron OU, sequence-isolated).
# ─────────────────────────────────────────────────────────────────────────────────────────────
PROD_SWEEP = (-1.0, -0.5, 0.0, 0.5, 1.0)     # the production-realistic band (+ neutral) the answer is gated on


class PoolAffectLadder:
    """The Gate-B affect ladder read. shared=None -> the SHIPPED standalone production ladder
    (`_appraisal_interoceptive_ladder_derisk.AppraisalInteroceptiveLadder`, the default-ON production path today).
    shared=<MergedPool containing AFFECT_DESCRIPTOR> -> the same protocol on the pool slice."""

    def __init__(self, seed: int = 42, shared=None, i_pa: float = APPR_INTERO_I_PA):
        self.seed = int(seed)
        self._shared = shared
        self.i_pa = float(i_pa)
        self._standalone = None
        self._built = False
        self._reads = None
        self._answer = None

    # --- build ---
    def ensure_built(self):
        if self._built:
            return
        if self._shared is None:
            from research.runners._appraisal_interoceptive_ladder_derisk import AppraisalInteroceptiveLadder
            self._standalone = AppraisalInteroceptiveLadder(seed=self.seed, i_pa=self.i_pa)
        else:
            self._shared.ensure_built()
            b = self._shared.bridge
            idx = self._shared.idx(AFFECT_KEY)
            names = self._shared.meta[AFFECT_KEY]["names"]
            self.ladder = {
                "vplus": [idx[n] for n in names["vplus"]], "vminus": [idx[n] for n in names["vminus"]],
                "arousal": [idx[n] for n in names["arousal"]],
                "pos_readout": idx[names["pos_readout"]], "neg_readout": idx[names["neg_readout"]],
            }
            self.relay_idx = {s: idx[r] for s, r in _RELAY_NAMES.items()}
            self._ladder_flat = np.concatenate([np.concatenate(self.ladder[k]) for k in ("vplus", "vminus",
                                                                                          "arousal")])
            self.arousal_flat = np.concatenate(self.ladder["arousal"])
            self._xedge = XEDGE_GATE in getattr(b, "_transmission_gate_to_synapses", {})
        self._built = True

    # --- local OU (the curiosity-organ pattern, adapted to the ladder's 8 pA background) ---
    # Bridge-side OU state that `_initialize_ou_process_state` (re)writes; saved on entry and RESTORED on exit
    # (fix round 2026-09-23: the exit used to set these to None unconditionally, which would clobber an OU state a
    # caller had installed before the affect read -- harmless on today's noise-off pool, order-dependent otherwise).
    _OU_BRIDGE_ATTRS = ("cp_ou_current", "_region_ou_streams", "_ou_neuron_key_idx", "_ou_neuron_keys",
                        "_ou_pn_step", "ou_decay_factor", "ou_noise_std", "ou_mean", "cp_ou_neuron_mask")

    def affect_neuron_mask(self):
        """Backend bool mask over the pool: True on every affect-organ neuron (ladder rungs, readouts, aggregates,
        relays)."""
        pool = self._shared
        b = pool.bridge
        n = int(b.cp_membrane_potential_v.shape[0])
        m = pool.xp.zeros(n, dtype=bool)
        idx = np.concatenate([np.asarray(v, dtype=np.int64) for v in pool.idx(AFFECT_KEY).values()])
        m[pool.xp.asarray(idx)] = True
        return m

    @contextlib.contextmanager
    def local_ou(self, scope: str = "all"):
        """Local per-neuron OU background for an affect read window (+ Hebbian OFF), restored on exit.

        scope="all" (the production affect read, UNCHANGED): the OU stream is installed on every pool neuron; only
        the ladder is measured and the whole sequence is isolated, so it is inert for the affect read.
        scope="affect" (the arm-X instrument, fix round 2026-09-23): the engine's own `cp_ou_neuron_mask` seam
        (honoured by every step path) confines the OU CURRENT to the affect organ's neurons. Every other organ --
        in particular the surprise pool -- then runs noise-free, which is its production read operating point (the
        pool keeps OU off; the organ's threshold was calibrated noise-free). The per-neuron-keyed draws of the
        affect neurons are unchanged by the mask, and nothing projects INTO the affect regions, so the ladder's own
        dynamics are identical under either scope."""
        if scope not in ("all", "affect"):
            raise ValueError(f"local_ou scope must be 'all' or 'affect', got {scope!r}")
        pool = self._shared
        b = pool.bridge
        cfg = b.core_config
        n = int(b.cp_membrane_potential_v.shape[0])
        keys = ("enable_ou_process", "per_neuron_ou_seed", "ou_seed", "ou_std_current_pA",
                "ou_mean_current_pA", "ou_tau_ms", "enable_hebbian_learning")
        saved = {k: getattr(cfg, k, None) for k in keys}
        _missing = object()
        saved_b = {k: getattr(b, k, _missing) for k in self._OU_BRIDGE_ATTRS}
        mask = self.affect_neuron_mask() if scope == "affect" else None
        try:
            from sim.config import CoreSimConfig
            d = CoreSimConfig()
            # The standalone ladder runs with Hebbian OFF. The pool's GLOBAL Hebbian flag is on for surprise /
            # world-model training; left on during this read, the non-rate-window Hebbian path CLIPS even
            # plastic=False edges to hebbian_max_weight (measured 2026-09-23: an affect-only pool at the default
            # max 1.0 clipped the 28-weight latch recurrence to 1.0 in ONE read -> differential 0.0). Faithful +
            # protective: Hebbian OFF for the read window only (restored on exit).
            cfg.enable_hebbian_learning = False
            cfg.enable_ou_process = True
            cfg.per_neuron_ou_seed = True
            cfg.ou_seed = int(self.seed)
            cfg.ou_std_current_pA = float(SA.AFF_OU_PA)        # == the standalone ladder's background (8 pA)
            cfg.ou_mean_current_pA = float(d.ou_mean_current_pA)
            cfg.ou_tau_ms = float(d.ou_tau_ms)
            b._initialize_ou_process_state(cfg, n)
            if mask is not None:
                b.cp_ou_neuron_mask = mask
            yield
        finally:
            for k, v in saved_b.items():
                if v is _missing:
                    # absent before entry: the pre-fix exit state (None / step 0) for the per-read arrays; the
                    # scalar OU coefficients are only read when cp_ou_current is not None, so they are left as-is.
                    if k in ("cp_ou_current", "_region_ou_streams", "_ou_neuron_key_idx", "_ou_neuron_keys",
                             "cp_ou_neuron_mask"):
                        setattr(b, k, None)
                    elif k == "_ou_pn_step":
                        b._ou_pn_step = 0
                else:
                    setattr(b, k, v)
            for k, v in saved.items():
                setattr(cfg, k, v)

    def _drive_relays(self, m, pos_sign):
        b = self._shared.bridge
        b.cp_external_input_current[:] = 0.0
        cur = self.i_pa * float(m)
        b.cp_external_input_current[self.relay_idx["vplus"]] = np.float32(cur if pos_sign else 0.0)
        b.cp_external_input_current[self.relay_idx["vminus"]] = np.float32(0.0 if pos_sign else cur)
        b.cp_external_input_current[self.relay_idx["arousal"]] = np.float32(cur)
        # ANTI-CHEAT (kept from the standalone): the ladder rungs never receive a direct host current.
        assert float(np.abs(np.asarray(b.cp_external_input_current)[self._ladder_flat]).max()) == 0.0, \
            "ladder pools received a direct external current -- the appraisal->ladder path must be synaptic"

    def set_gates(self, lesion=False, intero_lesion=False, xedge_lesion=False):
        b = self._shared.bridge
        b.set_transmission_gate("affect_out", 0.0 if lesion else 1.0)
        b.set_transmission_gate(APPR_INTERO_GATE, 0.0 if intero_lesion else 1.0)
        if self._xedge:
            b.set_transmission_gate(XEDGE_GATE, 0.0 if xedge_lesion else 1.0)

    def restore_gates(self):
        self.set_gates(False, False, False)

    def run_appraisal_phase(self, appraisal, ramp_ms=SA.LAD_RAMP_MS, drive_off_ms=SA.LAD_DRIVE_OFF_MS):
        """settle(40) -> graded ramp 0->|appraisal| on the relays -> drive-off hold. Leaves the ladder HOLDING.
        Caller owns isolation + local OU + gates. Returns relay spike counts during the ramp."""
        b = self._shared.bridge
        m_abs = abs(float(appraisal))
        pos_sign = float(appraisal) >= 0.0
        spk = {"vplus": 0.0, "vminus": 0.0, "arousal": 0.0}
        for _ in range(40):
            self._drive_relays(0.0, pos_sign); b._run_one_simulation_step()
        for s in range(int(ramp_ms)):
            self._drive_relays(m_abs * (s + 1) / ramp_ms, pos_sign); b._run_one_simulation_step()
            fs = np.asarray(b.cp_firing_states)
            for k in spk:
                spk[k] += float(fs[self.relay_idx[k]].sum())
        for _ in range(int(drive_off_ms)):
            self._drive_relays(0.0, pos_sign); b._run_one_simulation_step()
        return spk

    def read_differential(self, appraisal: float, lesion: bool = False, intero_lesion: bool = False,
                          ramp_ms: int = SA.LAD_RAMP_MS, drive_off_ms: int = SA.LAD_DRIVE_OFF_MS,
                          read_ms: int = SA.LAD_READ_MS, ou_scope: str = "all") -> dict:
        """Same signature + return keys as `AppraisalInteroceptiveLadder.read_differential` (drop-in for the
        production dispatch). `ou_scope` (pool only; default "all" = the unchanged production read) selects the
        local-OU scope -- see `local_ou`; the arm-X instrument check reads it both ways and requires identity."""
        self.ensure_built()
        if self._shared is None:
            return self._standalone.read_differential(appraisal, lesion=lesion, intero_lesion=intero_lesion,
                                                      ramp_ms=ramp_ms, drive_off_ms=drive_off_ms,
                                                      read_ms=read_ms)
        from research.runners._gnw_rung1_ignition_curve_derisk import _restore_state
        pool = self._shared
        b = pool.bridge
        with pool.sequence_isolation():
            _restore_state(b, pool.snap)
            b.cp_external_input_current[:] = 0.0
            with self.local_ou(scope=ou_scope):
                self.set_gates(lesion=lesion, intero_lesion=intero_lesion)
                try:
                    spk = self.run_appraisal_phase(appraisal, ramp_ms, drive_off_ms)
                    pos = neg = 0.0
                    pos_sign = float(appraisal) >= 0.0
                    for _ in range(int(read_ms)):
                        self._drive_relays(0.0, pos_sign); b._run_one_simulation_step()
                        fs = np.asarray(b.cp_firing_states)
                        pos += float(fs[self.ladder["pos_readout"]].sum())
                        neg += float(fs[self.ladder["neg_readout"]].sum())
                finally:
                    self.restore_gates()
            b.cp_external_input_current[:] = 0.0
        denom = float(SA.LAD_N_RO * max(1, read_ms))
        pr, nr = pos / denom, neg / denom
        rr = {k: v / (APPR_INTERO_N * max(1, int(ramp_ms))) for k, v in spk.items()}
        return {"differential": float(pr - nr), "pos_rate": float(pr), "neg_rate": float(nr),
                "appraisal": float(appraisal), "lesioned": bool(lesion), "intero_lesioned": bool(intero_lesion),
                "relay_rate_vplus": rr["vplus"], "relay_rate_vminus": rr["vminus"],
                "relay_rate_arousal": rr["arousal"], "mechanism": "interoceptive_afferent_onebrain_pool"}

    # --- organ-read battery (merge-framework read_fn / answer_fn) ---
    def _battery(self):
        if self._reads is not None:
            return
        from research.runners.affect_production_organ import tone_level
        reads = {}
        levels = []
        for a in PROD_SWEEP:
            r = self.read_differential(a)
            reads[f"diff[{a:+.1f}]"] = r["differential"]
            reads[f"relay_arousal[{a:+.1f}]"] = r["relay_rate_arousal"]
            levels.append(int(tone_level(r["differential"])))
        reads["diff_lesion[+0.7]"] = self.read_differential(0.7, lesion=True)["differential"]
        reads["diff_intero_lesion[+1.0]"] = self.read_differential(1.0, intero_lesion=True)["differential"]
        reads["diff_intero_lesion[-1.0]"] = self.read_differential(-1.0, intero_lesion=True)["differential"]
        self._reads = reads
        self._answer = tuple(levels)

    def reads(self):
        self.ensure_built(); self._battery()
        return dict(self._reads)

    def answer(self):
        """The categorical downstream answer: the graded TONE LEVEL per production appraisal (what
        `content_plan` / `manner_for` consume)."""
        self.ensure_built(); self._battery()
        return self._answer


def _affect_organ(seed, shared=None):
    return PoolAffectLadder(int(seed), shared=shared)


def _affect_reads(organ):
    return organ.reads()


def _affect_answer(organ):
    return organ.answer()


AFFECT_DESCRIPTOR = OrganDescriptor(
    key=AFFECT_KEY,
    regions=(),                                 # discovered at build (organ_regions); _affect_region_names() lists them
    spec_fn=_spec_affect_ladder,
    config=_AFFECT_CONFIG,
    param_het=True,
    idx_fn=_affect_idx_fn,
    organ_cls=_affect_organ, read_fn=_affect_reads, answer_fn=_affect_answer,
    supports_shared=True,
    scaffold_residuals=("appraisal VALUE is the Warriner-gated DR-2 learned map + a host current onto the "
                        "interoceptive relay pools (declared sensory boundary, unchanged from production)",
                        "local per-neuron OU built per read (the frozen pool keeps noise off globally)"),
)
# `regions=()` would make MergedPool._keep_mask empty (it reads d.regions); fill it from the real spec.
AFFECT_DESCRIPTOR = _dc.replace(AFFECT_DESCRIPTOR, regions=_affect_region_names())


AROUSAL_XEDGE = CrossEdge(
    key=XEDGE_KEY,
    source_key=AFFECT_KEY, source_region="aff_arousal_L1..L8",
    target_key="surprise", target_region="surprise",
    init_weight=XEDGE_W, plastic=False, freeze_rest=False,
    learn_rule="none",
    source_idx_fn=_arousal_idx,
    transmission_gate=XEDGE_GATE,
)


def affect_descriptors():
    """The 12-organ family: the shipped Wave-3 11-organ reconciliation UNCHANGED (reuse-by-import) + affect."""
    from research.runners._onebrain_wave3_organread_verify import _wave3_descriptors
    return list(_wave3_descriptors()) + [AFFECT_DESCRIPTOR]


def build_affect_pool(seed: int, xedge: bool = False, xedge_w: float | None = None):
    from research.runners.onebrain_merge_framework import merge_organs
    edges = None
    if xedge:
        ce = AROUSAL_XEDGE if xedge_w is None else _dc.replace(AROUSAL_XEDGE, init_weight=float(xedge_w))
        edges = [ce]
    return merge_organs(affect_descriptors(), int(seed), wire=True, cross_edges=edges)


_POOL: dict = {}


def get_affect_pool(seed: int = 42):
    """Process-shared 12-organ pool (memoized by (seed, xedge)). Production reaches it only when
    `affect_pool_enabled()`."""
    key = (int(seed), bool(affect_xedge_enabled()))
    if key not in _POOL:
        _POOL[key] = build_affect_pool(key[0], xedge=key[1])
    return _POOL[key]


_LADDERS: dict = {}


def get_pool_ladder(seed: int = 42) -> PoolAffectLadder:
    pool = get_affect_pool(seed)
    k = id(pool)
    if k not in _LADDERS:
        _LADDERS[k] = PoolAffectLadder(int(seed), shared=pool)
    return _LADDERS[k]
