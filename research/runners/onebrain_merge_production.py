"""ONE-BRAIN MERGE — the FIRST production rung: TWO co-resident production organs on ONE shared spiking bridge.

THE GAP (production is still CO-RESIDENCY). Every Gate-B spiking organ builds its OWN `SimulationBridge`
(its own `cp_` neuron array, its own step). "One substrate" is de-risked byte-EXACT end-to-end
(`2026-08-13-one-brain-merge-CLOSED-per-region-threshold.md` closed the INIT-RNG cause via
`cfg.per_region_threshold_heterogeneity`; `...-homeostasis-GO.md` closed the homeostatic idle-drift cause via
`cfg.per_region_homeostasis_isolation`; `...-Norgan-GO.md` scaled it to N organs + DIFFERENT builders) but NO
production organ set actually shares one pool yet.

THIS RUNG. The two MOST COMPATIBLE production organs — the D2 SURPRISE expectation-violation organ
(`surprise_production_organ`, `build_expectation_circuit`) and the E2 affective WORLD-MODEL organ
(`worldmodel_production_organ`, `build_world_model_circuit`) — onto ONE shared `SimulationBridge`. They are the
safest first pair: their global configs are IDENTICAL where it matters (`dt_ms=1.0`, IZHIKEVICH,
GENERIC_UNSTRUCTURED, `enable_homeostasis=True`, `enable_gabab=True` with `gabab_conductance_max=0` so GABA_B is
inert in both, the same Hebbian block, `enable_nmda` unset in both) — so the config SUPERSET is a trivial union
with NO genuine single-valued conflict (contrast the mapped `dt_ms`/`homeostasis` conflict of the
expectation+Wong-Wang diffbuilder pair in `...-Norgan-GO.md`). Region names are disjoint (surprise: cue /
patient_expected / patient_asserted / surprise; worldmodel: state / pred_{pos,neg} / obs_{pos,neg} /
surprise_{pos,neg}).

GUARDED + DEFAULT-OFF. `BRAIN_ONEBRAIN_MERGE` (or `cfg.merge_production_organs`) default-OFF -> each organ builds
its OWN bridge exactly as today (production byte-identical, no regression). ON -> both organs share ONE bridge
built here, with the TWO merge flags ON (`per_region_threshold_heterogeneity=True`,
`per_region_homeostasis_isolation=True`). No cross-organ synapse is added on this rung (the load-bearing claim is
byte-identity of the two organs' reads merged-vs-co-resident; a genuine cross synapse is the named next step).

BYTE-IDENTITY, why it is EXACT. With both flags ON, each organ's per-neuron init is name-keyed (invariant to
co-residents) and idle co-resident neurons are FROZEN by the homeostasis-isolation gate (they do not drift while
the other organ trains/reads). Each organ trains + reads ONLY its own regions (disjoint names, NO cross synapse),
so on the shared bridge every read reproduces the standalone read bit-for-bit. Verified per-seed by
`_onebrain_merge_rung1_verify.py` (merged-vs-co-resident read deltas printed; expect 0.0).

NO `sim/` edit — the two merge flags already exist on `main` (guarded, default-off). Reuse-by-import: the region /
pathway SPECS + the post-build block-diagonal wiring are pulled from each de-risk builder; the two production
organ classes train + read their own slice on the shared bridge via a small additive `shared=` injection.
Process backend (cupy in production, numpy in tests).
"""
from __future__ import annotations

import contextlib
import os

from research.runners._spiking_expectation_rpe_derisk import (
    build_expectation_circuit,
    _install_block_diagonal,
    _idx,
)
from research.runners._affective_world_model_derisk import build_world_model_circuit

# The surprise organ's build parameters (must match SurpriseProductionOrgan's defaults so the merged slice is
# byte-identical to the standalone organ).
_SURPRISE_KW = dict(n_trained=8, n_novel=4, blk=24, cue_blk=24, cue_to_expected_weight=0.8)
# The worldmodel organ's build parameter (must match WorldModelProductionOrgan's default).
_WORLDMODEL_KW = dict(n_states=6)


def merge_enabled() -> bool:
    """Default-OFF opt-in. `BRAIN_ONEBRAIN_MERGE` in {1,true,yes,on} -> the surprise + worldmodel organs share
    ONE spiking bridge. Absent / anything else -> each organ builds its own bridge exactly as today."""
    v = os.environ.get("BRAIN_ONEBRAIN_MERGE")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


class MergedSubstrate:
    """ONE `SimulationBridge` holding the SURPRISE organ's regions and/or the WORLD-MODEL organ's regions, with
    BOTH merge flags ON (`per_region_threshold_heterogeneity`, `per_region_homeostasis_isolation`). Built ONCE
    (lazily), then SHARED: each production organ trains + reads its own region slice on `self.bridge`.

    `organs` selects which organs' regions are present — ("surprise", "worldmodel") for the real production merge,
    or a single-organ tuple for the byte-identity CO-RESIDENT baseline (an organ on its own bridge, both flags ON,
    the same construction path — so merged-vs-solo isolates the merge itself)."""

    def __init__(self, seed: int = 42, organs=("surprise", "worldmodel")):
        self.seed = int(seed)
        self.organs = tuple(organs)
        self.bridge = self.cfg = self.xp = None
        self.meta_surprise = None      # metaS: n_trained/n_novel/n_concepts/blk/cue_blk/W_exc/W_inh
        self.meta_worldmodel = None    # metaW: n_states/blk/npred/nobs/nsurp
        self._built = False

    def ensure_built(self):
        if self._built:
            return
        from sim.bridge import SimulationBridge
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.enums import NeuronModel
        from sim.backend import get_backend
        xp, _ = get_backend()

        # SPEC EXTRACTION (reuse-by-import): build each organ's standalone bridge purely to pull its real
        # BrainRegion / RegionPathway specs + meta. per_region_thresh on the throwaway is irrelevant (specs are
        # flag-independent); the MERGED cfg below sets both flags. These throwaways are not used afterward.
        _brS, cfgS, metaS = build_expectation_circuit(self.seed, per_region_thresh=True, **_SURPRISE_KW)
        _brW, cfgW, metaW = build_world_model_circuit(self.seed, **_WORLDMODEL_KW)
        self.meta_surprise = metaS
        self.meta_worldmodel = metaW

        # ── THE MERGED (or single-organ baseline) CONFIG SUPERSET. Globals replicate build_expectation_circuit
        #    exactly (they are identical to build_world_model_circuit where they matter); the only additions are
        #    per_region_homeostasis_isolation=True and the region/pathway UNION. ──
        cfg = CoreSimConfig()
        cfg.seed = int(self.seed); cfg.heterogeneity_seed = int(self.seed); cfg.ou_seed = int(self.seed)
        cfg.per_region_threshold_heterogeneity = True     # merge flag #1 (INIT byte-identity)
        cfg.per_region_homeostasis_isolation = True       # merge flag #2 (idle-drift byte-identity)
        cfg.dt_ms = 1.0
        cfg.num_traits = 1
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.connections_per_neuron = 0
        cfg.enable_brain_region_framework = True
        cfg.enable_stdp = False
        cfg.enable_hebbian_learning = True
        cfg.hebbian_learning_rate = 0.06
        cfg.hebbian_min_weight = 0.0
        cfg.hebbian_max_weight = 45.0
        cfg.hebbian_weight_decay = 0.0
        cfg.hebbian_rate_window = True
        cfg.hebbian_coactivity_decay = 0.85
        cfg.hebbian_coactivity_thresh = 0.20
        cfg.hebbian_mean_subtract = 1.0
        cfg.enable_reward_modulation = False
        cfg.enable_short_term_plasticity = False
        cfg.enable_structural_plasticity = False
        cfg.enable_parameter_heterogeneity = False
        cfg.enable_ou_process = False
        cfg.enable_conductance_noise = False
        cfg.current_reward_signal = 0.0
        cfg.reward_baseline = 0.0
        cfg.enable_gabab = True
        cfg.gabab_reversal_potential = -90.0
        cfg.gabab_tau_decay = 150.0
        cfg.gabab_propagation_strength = 0.22
        cfg.gabab_conductance_max = 0.0                   # GABA_B inert in BOTH organs -> tau/prop are don't-cares

        regions = []
        pathways = []
        if "surprise" in self.organs:
            regions += list(cfgS.brain_regions)
            pathways += list(cfgS.region_pathways)
        if "worldmodel" in self.organs:
            regions += list(cfgW.brain_regions)
            pathways += list(cfgW.region_pathways)
        cfg.brain_regions = regions
        cfg.region_pathways = pathways

        bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                  runtime_state=RuntimeState(), gpu_config=GPUConfig())
        bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        bridge.runtime_state.actual_seed_used = self.seed
        bridge._initialize_simulation_data(called_from_playback_init=False)

        # POST-BUILD WIRING. The surprise organ's builder installs its TOPOGRAPHIC block-diagonal edges AFTER
        # init (pathways are built full then masked concept c -> block c) — reproduce exactly on the merged
        # bridge. The worldmodel organ installs NO block-diagonal (its pathways stay full; the transition is
        # Hebbian-learned), so nothing extra is needed for it.
        if "surprise" in self.organs:
            blk = metaS["blk"]
            _install_block_diagonal(bridge, "patient_asserted", "surprise", blk, metaS["W_exc"])
            _install_block_diagonal(bridge, "patient_expected", "surprise", blk, metaS["W_inh"])
            _install_block_diagonal(bridge, "cue", "patient_expected", blk,
                                    float(_SURPRISE_KW["cue_to_expected_weight"]))
            bridge._blk = blk
        elif "worldmodel" in self.organs:
            bridge._blk = metaW["blk"]

        # Resting snapshot for hard resets (both organs' builders snapshot after their wiring; wiring changes
        # weights, not membrane state, so this is the deterministic init membrane state).
        bridge._rest_v = bridge.cp_membrane_potential_v.copy()
        bridge._rest_u = bridge.cp_recovery_variable_u.copy()

        self.bridge = bridge
        self.cfg = cfg
        self.xp = xp
        self._built = True

    _SURPRISE_REGIONS = ("cue", "patient_expected", "patient_asserted", "surprise")
    _WORLDMODEL_REGIONS = ("state", "pred_pos", "pred_neg", "obs_pos", "obs_neg",
                           "surprise_pos", "surprise_neg")

    def surprise_idx_map(self):
        """The surprise organ's region -> neuron-index map on the shared bridge."""
        self.ensure_built()
        return {n: self.xp.asarray(_idx(self.bridge, n)) for n in self._SURPRISE_REGIONS}

    def worldmodel_idx_map(self):
        """The worldmodel organ's region -> neuron-index map on the shared bridge."""
        self.ensure_built()
        return {n: self.xp.asarray(_idx(self.bridge, n)) for n in self._WORLDMODEL_REGIONS}

    def _keep_mask(self, active: str):
        """Cached xp boolean mask, True over the ACTIVE organ's neurons (the ones allowed to keep their
        homeostatic update across a read), False over the co-resident's (restored)."""
        cache = getattr(self, "_keep_mask_cache", None)
        if cache is None:
            cache = self._keep_mask_cache = {}
        if active not in cache:
            self.ensure_built()
            regions = self._SURPRISE_REGIONS if active == "surprise" else self._WORLDMODEL_REGIONS
            n = int(self.bridge.cp_membrane_potential_v.shape[0])
            mask = self.xp.zeros(n, dtype=bool)
            for r in regions:
                mask[self.xp.asarray(_idx(self.bridge, r))] = True
            cache[active] = mask
        return cache[active]

    # Per-neuron state arrays that PERSIST across an organ's own read (its `_hard_reset` does NOT clear all of
    # them — e.g. it resets `cp_refractory` by the wrong name and never touches `cp_prev_firing_states`), so a
    # co-resident organ's spontaneous firing during the active organ's read leaves a footprint on them. The
    # read-isolation guard snapshots + restores the co-resident's slice of ALL of these (membrane / conductances
    # are also restored — harmless, the co-resident's next read hard-resets them anyway — so the guard is robust
    # to which arrays a given organ's `_hard_reset` happens to clear).
    _PER_NEURON_STATE = (
        "cp_membrane_potential_v", "cp_recovery_variable_u",
        "cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab", "cp_conductance_g_nmda",
        "cp_firing_states", "cp_prev_firing_states", "cp_refractory_timers", "cp_refractory",
        "cp_neuron_firing_thresholds", "cp_neuron_activity_ema", "cp_external_input_current",
    )

    @contextlib.contextmanager
    def read_isolation(self, active: str):
        """Make a read of the `active` organ leave NO footprint on the CO-RESIDENT organ's persistent neural state.

        Two coupling paths exist on a shared, continuously-stepped substrate, both because a co-resident organ's
        HOMEOSTATICALLY-SILENCED FS neurons drop their threshold to ~rest and thus fire SPONTANEOUSLY while the
        active organ is read (an evolution the standalone bridge, stepped only during its OWN reads, never
        undergoes; participation-gated `per_region_homeostasis_isolation` cannot freeze them — they participate by
        firing): (1) intrinsic plasticity nudges the co-resident's thresholds + activity EMA; (2) the spontaneous
        spikes advance the co-resident's refractory timers + previous-firing state, which its own `_hard_reset`
        does not fully clear. Here we snapshot the FULL per-neuron state, run the read (the ACTIVE organ's slice
        self-adapts EXACTLY as standalone — preserved), then RESTORE the co-resident organ's slice. So each organ's
        neural evolution depends only on its OWN reads -> byte-identical to the standalone organ (there is no cross
        synapse, so the restored co-resident never influenced the read)."""
        b = self.bridge
        snaps = []
        for name in self._PER_NEURON_STATE:
            arr = getattr(b, name, None)
            snaps.append(None if arr is None else arr.copy())
        try:
            yield
        finally:
            keep = self._keep_mask(active)
            for name, snap in zip(self._PER_NEURON_STATE, snaps):
                if snap is None:
                    continue
                cur = getattr(b, name)
                setattr(b, name, self.xp.where(keep, cur, snap))


# The process-shared production merge substrate (built once on first use; holds BOTH organs).
_MERGED_SUBSTRATE: "MergedSubstrate | None" = None


def get_merged_substrate(seed: int = 42) -> MergedSubstrate:
    """The process-shared surprise+worldmodel merged substrate (the production merge, both organs on one pool)."""
    global _MERGED_SUBSTRATE
    if _MERGED_SUBSTRATE is None:
        _MERGED_SUBSTRATE = MergedSubstrate(seed=seed, organs=("surprise", "worldmodel"))
    return _MERGED_SUBSTRATE
