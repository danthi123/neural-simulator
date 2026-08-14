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

GUARDED + PRODUCTION-DEFAULT (flipped ON 2026-08-13). `BRAIN_ONEBRAIN_MERGE` default-ON -> both organs share ONE
bridge built here, with the TWO merge flags ON (`per_region_threshold_heterogeneity=True`,
`per_region_homeostasis_isolation=True`). `BRAIN_ONEBRAIN_MERGE=0` is the ESCAPE -> each organ builds its OWN
bridge, byte-identical to the pre-flip production. No cross-organ synapse is added on this rung (the load-bearing
claim is byte-identity of the two organs' reads merged-vs-co-resident-with-flags + answer-preservation vs the
pre-flip separate-bridge reads; a genuine cross synapse is the named next step). See `merge_enabled` /
`_MERGE_DEFAULT_ON` + the flip verify `research/runners/_onebrain_production_flip_verify.py`.

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

import numpy as np

from research.runners._spiking_expectation_rpe_derisk import (
    build_expectation_circuit,
    _install_block_diagonal,
    _idx,
    _host,
)
from research.runners._affective_world_model_derisk import build_world_model_circuit
from research.runners.rf_phasor_composer import RFPhasorComposer

# The surprise organ's build parameters (must match SurpriseProductionOrgan's defaults so the merged slice is
# byte-identical to the standalone organ).
_SURPRISE_KW = dict(n_trained=8, n_novel=4, blk=24, cue_blk=24, cue_to_expected_weight=0.8)
# The worldmodel organ's build parameter (must match WorldModelProductionOrgan's default).
_WORLDMODEL_KW = dict(n_states=6)

# ── COMPOSER-IN-POOL#1 sizing (the RECALL COMPOSER + its phase->spike TRANSDUCER cleanup on pool #1) ──
# The composer region holds the RF-phasor recall's resonate ops on a masked SLICE of the shared bridge. Size it
# for the largest RF op the recall issues: a 6-role encode/bundle (<=4*D) AND the K-fact batched moat scan
# (2*K*D) must fit -> max(7, 2*K)*D. D matches the production RF composer (BrainConversationalAgent D=128);
# `_COMPOSER_KMAX` caps the store size whose batched scan stays ON the shared pool (a larger store's scan
# gracefully FALLS BACK to a private per-op RF bridge -- byte-identical, but off-pool: the sizing residual named
# in the wire finding). The CLEANUP region is V word-blocks of the phase->spike transducer (idle in the
# byte-identical production turn -- the recall->surprise cross synapse is the NEXT behavioural rung, not wired
# here, so surprise stays byte-identical).
_COMPOSER_D = 128
_COMPOSER_KMAX = 16
_CLEANUP_BLK = 24
_CLEANUP_VOCAB = 16

# ── ONEBRAIN-COMPOSER-IN-POOL#1 sizing (the b-closer). The PRODUCTION-DEFAULT composer is `OneBrainComposer`
# (`composer_kind='onebrain'`), NOT the RF-phasor path wired above. Its whole who/what pipeline (parser + big-RF +
# k_max store + Q registers + batched cleanup) lives on ONE bridge sized to its full layout span
# `OneBrainComposer.n_total` (D=128, headroom vocab, k_max=32, attribute role -> 45856 at the tiny-demo vocab). To
# join pool #1, the pool reserves a region of EXACTLY that span (`_ONEBRAIN_SPAN`, registered at bind time before the
# process-global substrate is first built), and `Pool1BoundOneBrainComposer` REBASES the composer's whole RF layout
# onto that slice (the CoResidentOneBrainComposer index-shift, Probe-1 byte-identical) while its parser stays on a
# PRIVATE full-size bridge (byte-identical comprehension; the pool's Hebbian/homeostasis config differs from the
# composer bridge's, and `_run_one_simulation_step` steps ALL neurons, so the parser CANNOT run on the pool without
# breaking both recall byte-identity AND surprise/world-model byte-identity). Default None -> no onebrain region (the
# RF-path region + every existing caller is byte-unchanged).
_ONEBRAIN_SPAN: "int | None" = None

# COMPOSER-IN-POOL#1 DEFAULT (flipped ON 2026-08-14): the PRODUCTION-DEFAULT composer joins pool #1 by default. The
# b-closer (`Pool1BoundOneBrainComposer`) routes the SHIPPED `OneBrainComposer` (`composer_kind='onebrain'`) RF
# recall/store onto pool #1's shared bridge (one `cp_membrane_potential_v` with surprise + world-model), its parser on
# a private bridge. Earned by the DEFAULT-FLIP verify (`_onebrain_composer_pool1_production_verify.py --default-flip`,
# 6/6 seeds): DEFAULT-no-env vs ESCAPE `BRAIN_COMPOSER_MERGE=0` is byte-identical (recall/moat/surprise/world-model
# max delta 0.0), the moat abstains, it is genuinely one pool, both organs stay alive; determinism 9/9 + smoke
# byte-identical to the pre-flip baseline. `BRAIN_COMPOSER_MERGE=0` is the ESCAPE (each composer builds its own
# bridge, byte-identical to the pre-flip production). Ledger row: onebrain-composer-pool1-DEFAULT-FLIP. Finding:
# 2026-08-14-onebrain-composer-pool1-DEFAULT-FLIP-GO.md. (The RF-phasor `composer_kind='rf'` path also joins pool #1
# under this flag -- its own 6/6-GO wire, byte-identical.)
_COMPOSER_IN_POOL1_DEFAULT_ON = True


def composer_merge_enabled() -> bool:
    """DEFAULT-OFF (`_COMPOSER_IN_POOL1_DEFAULT_ON`). `BRAIN_COMPOSER_MERGE` in {1,true,yes,on} -> the RF-phasor
    RECALL COMPOSER (the `/api/brain-chat` recall organ) + its phase->spike TRANSDUCER cleanup region JOIN pool
    #1's shared bridge (alongside surprise + world-model, one `cp_membrane_potential_v`); in {0,false,no,off} or
    ABSENT -> the composer builds its OWN per-op RF bridges exactly as today (production byte-identical). Only
    the RF-phasor composer path (`composer_kind='rf'`) joins here; the production-default OneBrainComposer builds
    its own large co-resident bridge -- a separate, larger merge (see the wire finding's residual)."""
    v = os.environ.get("BRAIN_COMPOSER_MERGE")
    if v is None:
        return _COMPOSER_IN_POOL1_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "yes", "on")


# PRODUCTION DEFAULT (flipped ON 2026-08-13): the surprise + world-model production organs build on ONE shared
# `SimulationBridge` (one `cp_membrane_potential_v`) by default. This is byte-identical to the co-resident-WITH-
# merge-flags baseline (rung-1, 6/6 GO) and ANSWER-PRESERVING vs the pre-flip separate-bridge reads (every
# `surprised` bool + `pred_sign` identical across a broad panel + 6 seeds; the numeric Hz/margin reads shift — the
# inherent, characterized cost of a genuine shared pool, since one global RNG cannot reproduce BOTH organs'
# standalone threshold draws — but NO classification crosses a threshold). `BRAIN_ONEBRAIN_MERGE=0` is the ESCAPE:
# it reverts to two separate bridges, byte-identical to the pre-flip production. Verify:
# research/runners/_onebrain_production_flip_verify.py (FLIP-GO 6/6). Ledger row: onebrain-merge-organs.
_MERGE_DEFAULT_ON = True


def merge_enabled() -> bool:
    """Production-DEFAULT (`_MERGE_DEFAULT_ON`). `BRAIN_ONEBRAIN_MERGE` in {1,true,yes,on} -> the surprise +
    world-model organs share ONE spiking bridge; in {0,false,no,off} -> each builds its own bridge (the escape,
    byte-identical to the pre-flip production); ABSENT -> the production default (`_MERGE_DEFAULT_ON`, ON)."""
    v = os.environ.get("BRAIN_ONEBRAIN_MERGE")
    if v is None:
        return _MERGE_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "yes", "on")


class MergedSubstrate:
    """ONE `SimulationBridge` holding the SURPRISE organ's regions and/or the WORLD-MODEL organ's regions, with
    BOTH merge flags ON (`per_region_threshold_heterogeneity`, `per_region_homeostasis_isolation`). Built ONCE
    (lazily), then SHARED: each production organ trains + reads its own region slice on `self.bridge`.

    `organs` selects which organs' regions are present — ("surprise", "worldmodel") for the real production merge,
    or a single-organ tuple for the byte-identity CO-RESIDENT baseline (an organ on its own bridge, both flags ON,
    the same construction path — so merged-vs-solo isolates the merge itself)."""

    def __init__(self, seed: int = 42, organs=("surprise", "worldmodel"),
                 composer_D: int = _COMPOSER_D, composer_kmax: int = _COMPOSER_KMAX,
                 cleanup_blk: int = _CLEANUP_BLK, cleanup_vocab: int = _CLEANUP_VOCAB,
                 onebrain_span: "int | None" = None):
        self.seed = int(seed)
        self.organs = tuple(organs)
        # ONEBRAIN-COMPOSER region span (the b-closer): the OneBrainComposer.n_total the "onebrain_composer" region is
        # sized to (None unless that organ is present).
        self.onebrain_span = onebrain_span
        self.bridge = self.cfg = self.xp = None
        self.meta_surprise = None      # metaS: n_trained/n_novel/n_concepts/blk/cue_blk/W_exc/W_inh
        self.meta_worldmodel = None    # metaW: n_states/blk/npred/nobs/nsurp
        # COMPOSER-IN-POOL#1 (opt-in): the recall composer's region + the phase->spike transducer cleanup region.
        self.composer_D = int(composer_D)
        self.composer_kmax = int(composer_kmax)
        self.cleanup_blk = int(cleanup_blk)
        self.cleanup_vocab = int(cleanup_vocab)
        self.meta_composer = None      # {'D', 'cmp_n', 'kmax', 'cleanup_n'} when the composer region is present
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
        # COMPOSER-IN-POOL#1 (opt-in): append the recall COMPOSER region + the phase->spike TRANSDUCER cleanup
        # region AFTER surprise + world-model (name-keyed per-region init => the surprise/world-model slices are
        # INDEX- and byte-IDENTICAL to the 2-organ pool -- de-risk 6/6). NO cross synapse is added here (the
        # recall->surprise edge is the next behavioural rung), so both faculties' reads stay byte-identical and
        # the cleanup region is idle (frozen by per_region_homeostasis_isolation). Sizing: max(7, 2*K)*D so a
        # 6-role encode/bundle AND a K-fact batched moat scan both fit on the shared composer slice.
        if "composer" in self.organs:
            from sim.regions import BrainRegion
            cmp_n = max(7, 2 * self.composer_kmax) * self.composer_D
            regions.append(BrainRegion(
                name="composer", n_neurons=cmp_n, exc_fraction=1.0, internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False))
            self.meta_composer = {"D": self.composer_D, "cmp_n": cmp_n, "kmax": self.composer_kmax}
        if "cleanup" in self.organs:
            from sim.regions import BrainRegion
            cleanup_n = self.cleanup_vocab * self.cleanup_blk
            regions.append(BrainRegion(
                name="cleanup", n_neurons=cleanup_n, exc_fraction=1.0, internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False))
            if self.meta_composer is not None:
                self.meta_composer["cleanup_n"] = cleanup_n
        # ONEBRAIN-COMPOSER (the b-closer, opt-in): a SINGLE region sized to the production OneBrainComposer's full
        # layout span (parser + big-RF + store + Q + cleanup blocks), appended AFTER surprise + world-model (name-keyed
        # per-region init => surprise/world-model slices are index- and byte-IDENTICAL to the 2-organ pool). The
        # composer's RF recall/store ops run on this slice (masked, so surprise/world-model v/u stay byte-untouched);
        # its parser runs on a PRIVATE bridge (config-incompatible with the pool). NO cross synapse (the recall->surprise
        # edge is the next behavioural rung), so both organs' reads stay byte-identical and this slice is idle (frozen by
        # per_region_homeostasis_isolation) except during a composer op.
        if "onebrain_composer" in self.organs:
            from sim.regions import BrainRegion
            span = int(self.onebrain_span)
            regions.append(BrainRegion(
                name="onebrain_composer", n_neurons=span, exc_fraction=1.0, internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False))
            self.meta_composer = {"onebrain_span": span}
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

    def composer_idx(self):
        """The composer region's neuron indices on the shared bridge (host int array; contiguous block)."""
        self.ensure_built()
        return np.asarray(_host(_idx(self.bridge, "composer")))

    def cleanup_idx(self):
        """The phase->spike transducer cleanup region's neuron indices on the shared bridge (host int array)."""
        self.ensure_built()
        return np.asarray(_host(_idx(self.bridge, "cleanup")))

    def onebrain_composer_idx(self):
        """The ONEBRAIN-COMPOSER region's neuron indices on the shared bridge (host int array; contiguous block sized to
        the production OneBrainComposer's full layout span)."""
        self.ensure_built()
        return np.asarray(_host(_idx(self.bridge, "onebrain_composer")))

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
    """The process-shared merged substrate. Default: surprise + world-model on one pool (the production merge).
    When `composer_merge_enabled()` (opt-in `BRAIN_COMPOSER_MERGE`, default-off) the pool ALSO carries the recall
    COMPOSER region + the phase->spike TRANSDUCER cleanup region, so the RF-phasor recall can run on the shared
    bridge alongside the two organs (byte-identical surprise/world-model reads -- name-keyed init). Built ONCE;
    the flag is read at first build, so every caller (surprise organ, world-model organ, the composer bind) gets
    the SAME 2- or 4-organ singleton for this process."""
    global _MERGED_SUBSTRATE
    if _MERGED_SUBSTRATE is None:
        if composer_merge_enabled() and _ONEBRAIN_SPAN is not None:
            # ONEBRAIN-COMPOSER (the b-closer): the pool carries the production-default OneBrainComposer's full layout
            # span as a single "onebrain_composer" region (its parser stays on a private bridge). This is the branch the
            # DEFAULT flip exercises -- the SHIPPED composer path, not the RF-phasor path.
            _MERGED_SUBSTRATE = MergedSubstrate(seed=seed, organs=("surprise", "worldmodel", "onebrain_composer"),
                                                onebrain_span=int(_ONEBRAIN_SPAN))
        elif composer_merge_enabled():
            _MERGED_SUBSTRATE = MergedSubstrate(seed=seed, organs=("surprise", "worldmodel", "composer", "cleanup"))
        else:
            _MERGED_SUBSTRATE = MergedSubstrate(seed=seed, organs=("surprise", "worldmodel"))
    return _MERGED_SUBSTRATE


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  The production RF-phasor recall composer BOUND to pool #1's shared bridge (the composer slice).
# ─────────────────────────────────────────────────────────────────────────────────────────────
class Pool1BoundComposer(RFPhasorComposer):
    """An `RFPhasorComposer` whose RF resonate ops run on a masked SLICE of pool #1's shared bridge (one
    `cp_membrane_potential_v` shared with surprise + world-model). The de-risk `SharedBridgeComposer` index-shift
    mechanism, made production-robust with a GRACEFUL FALLBACK: an RF op too large for the composer region runs
    on a private per-op RF bridge instead (byte-identical -- a masked shared-slice RF op == a dedicated per-op RF
    bridge, de-risk 6/6 -- but off-pool). All other composer state (concept/role codes, kb, the no-confab moat)
    is unchanged, so recall + moat are byte-identical to a standalone `RFPhasorComposer`."""

    def bind_to_pool1(self, substrate: "MergedSubstrate"):
        """Bind this composer's RF ops onto `substrate`'s composer region. Requires the region D to match self.D."""
        substrate.ensure_built()
        if substrate.composer_D != self.D:
            raise ValueError(f"pool #1 composer region D={substrate.composer_D} != composer D={self.D}")
        cmp_idx = np.asarray(substrate.composer_idx())
        self._pool1 = substrate
        self._rf_base = int(cmp_idx.min())
        self._rf_size = int(len(cmp_idx))
        N = int(substrate.bridge.core_config.num_neurons)
        m = np.zeros(N, dtype=bool)
        m[cmp_idx] = True
        self._rf_mask = m
        return self

    def _resonate(self, n, conns, kick):
        n = int(n)
        pool1 = getattr(self, "_pool1", None)
        if pool1 is None or n > self._rf_size:
            # FALLBACK: an op bigger than the composer region (a large-K batched scan) runs on a private per-op
            # RF bridge -- byte-identical to the shared-slice op, but off the shared pool (the sizing residual).
            return super()._resonate(n, conns, kick)
        b = pool1.bridge
        base = self._rf_base
        N = int(b.core_config.num_neurons)
        shifted = [(base + int(post), base + int(pre), w) for (post, pre, w) in conns]
        b.rf_set_complex_weights(shifted)
        full_kick = np.zeros(N, dtype=np.complex128)
        kk = np.asarray(kick, dtype=np.complex128).reshape(-1)
        full_kick[base:base + n] = kk[:n]
        b.rf_kick(full_kick, period=self.period, lam=0.0, neuron_mask=self._rf_mask)
        b.rf_resonate_steps(self.period + 8)
        phases = np.asarray(b.rf_read_phases())
        if self.trace:
            self._last_resonate_n = n
        return phases[base:base + n]


def make_pool1_composer(seed: int = 42, **rf_kwargs) -> "Pool1BoundComposer":
    """Build the production RF-phasor recall composer BOUND to pool #1's shared bridge. `rf_kwargs` are exactly
    the `RFPhasorComposer` kwargs the caller would otherwise pass (D, vocab, period, grounded_codes, ...). The
    composer's D must match the pool's composer region (`_COMPOSER_D`)."""
    comp = Pool1BoundComposer(seed=seed, **rf_kwargs)
    comp.bind_to_pool1(get_merged_substrate(seed))
    return comp


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  The b-closer: the PRODUCTION-DEFAULT OneBrainComposer BOUND to pool #1's shared bridge.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _onebrain_layout_span(seed: int, D: int, vocab, k_max: int, enable_attributed: bool,
                          vocab_headroom: int) -> int:
    """The standalone `OneBrainComposer.n_total` (the full RF layout span) for these params -- the size the pool
    reserves for the "onebrain_composer" region. Mirrors `OneBrainComposer.__init__`'s layout math EXACTLY, INCLUDING
    the `vocab_headroom` reserved cleanup slots (which `CoResidentOneBrainComposer.n_total_for` OMITS) and the
    attribute role (n_roles=5). A drift here over/under-sizes the region and the rebase bounds-check in
    `Pool1BoundOneBrainComposer.__init__` fails loudly (never silently wrong)."""
    from research.runners.rf_phasor_composer import RFPhasorComposer
    comp = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    V = len(comp.words) + max(0, int(vocab_headroom))     # + headroom reserved slots (recruit-an-assembly pool)
    NP = len(comp.pol_words)
    D = int(D); k_max = int(k_max)
    P = 6 + 3 * 40
    n_roles = 5 if enable_attributed else 4               # agent/action/patient(+attribute)+polarity
    n_main = n_roles - 1
    store_base = P + (2 * n_roles + 1) * D
    block = 1 + D
    q_base = store_base + k_max * block
    cb = n_main * V + NP
    c_base = q_base + n_roles * D
    bat_q_base = c_base + cb
    bat_c_base = bat_q_base + k_max * n_roles * D
    return int(bat_c_base + k_max * cb)


# The b-closer class is a SUBCLASS of the (heavy) `OneBrainComposer`, built lazily + cached so importing this module
# never pulls one_brain_composer -> brain_conversational_agent -> (lazy) onebrain_merge_production at import time. The
# docstring for the class lives on `_POOL1_ONEBRAIN_DOC` (attached to the built class).
_POOL1_ONEBRAIN_CLASS = None
_POOL1_ONEBRAIN_DOC = """The production-DEFAULT `OneBrainComposer` with its RF RECALL/STORE ops bound to pool #1's
shared bridge (one `cp_membrane_potential_v` with surprise + world-model), while its PARSER stays on a PRIVATE
full-size bridge.

WHY THE SPLIT. `OneBrainComposer` runs TWO substrates on one bridge: the RF who/what pipeline (config-INDEPENDENT
resonate-and-fire dynamics -- `_rf_advance_one` reads only the per-op `_rf_omega/_rf_lambda/_rf_floor`, never the
cfg's Hebbian/homeostasis) AND a Hebbian PARSER (`BridgeParser`, Izhikevich `_run_one_simulation_step` over the WHOLE
bridge, trained with `hebbian_max_weight=400`). The RF ops port onto pool #1's slice BYTE-IDENTICALLY (the
`CoResidentOneBrainComposer` index-shift, Probe-1 GO at atol 1e-9). The parser CANNOT: pool #1's config differs
(`hebbian_max_weight=45`, `per_region_homeostasis_isolation=True`) AND its `_run_one_simulation_step` steps ALL
neurons -> the parser would (1) train differently (broken comprehension -> broken recall byte-identity) and (2)
advance + corrupt surprise/world-model (broken criterion-4 byte-identity). So the parser keeps its OWN bridge.

CONSTRUCTION. Build a FULL standalone `OneBrainComposer` (parser trained on its private big bridge; the complete
layout with `vocab_headroom` + recruit slots), then REBASE only the RF layout (`P/store_base/q_base/c_base/bat_q_base/
bat_c_base += rf_base`, `n_total = pool_N`, `rf_mask`/`_rf_reset_mask` = the composer's span on the pool, `self.b =
pool.bridge`). The parser handle keeps pointing at the private bridge, so `hear()` comprehends there and the RF
store/read run on the pool slice. Recall answers are byte-identical to the standalone by construction (identical
parser classification + rebased-RF identity); surprise/world-model stay byte-identical (masked RF writes leave their
v/u untouched, and the Izhikevich step is never called on the pool).

OVERFLOW. The pool region is sized to EXACTLY the standalone `n_total`, so every RF op fits by construction. If a
caller mis-sizes it (rf_base + span > pool_N), __init__ RAISES rather than silently truncating -- the spec's "never
silently wrong" invariant."""


def _pool1_onebrain_init(self, substrate, rf_base, seed=42, **ob_kwargs):
    from research.runners.one_brain_composer import OneBrainComposer
    # 1) FULL standalone build: parser + RF on a PRIVATE big bridge, the complete byte-identical layout.
    OneBrainComposer.__init__(self, seed=seed, **ob_kwargs)
    # 2) The parser keeps its OWN (private) bridge -- the one OneBrainComposer.__init__ just built + trained.
    self._parser_bridge = self.b
    # 3) REBASE the RF layout onto pool #1's "onebrain_composer" slice.
    substrate.ensure_built()
    pool = substrate.bridge
    N = int(pool.core_config.num_neurons)
    span = int(self.n_total)                                # the standalone layout span (pre-rebase n_total)
    cmp_idx = np.asarray(substrate.onebrain_composer_idx())
    base = int(cmp_idx.min())
    if int(rf_base) != base:
        raise ValueError(f"onebrain_composer region base {base} != requested rf_base {int(rf_base)}")
    if base + span > N:
        raise ValueError(f"onebrain_composer span {span} at base {base} exceeds pool N={N} "
                         f"(region reserved {len(cmp_idx)}) -- size _ONEBRAIN_SPAN to the composer's n_total")
    if len(cmp_idx) < span:
        raise ValueError(f"onebrain_composer region has {len(cmp_idx)} neurons < layout span {span}")
    self._rf_base = base
    self.P += base
    self.store_base += base
    self.q_base += base
    self.c_base += base
    self.bat_q_base += base
    self.bat_c_base += base
    self.n_total = N                                        # array-sizing is the full pool N
    self.b = pool                                           # RF ops now run on the pool slice
    self.rf_mask = np.zeros(N, dtype=bool)
    self.rf_mask[base:base + span] = True
    self._rf_reset_mask = self.rf_mask                      # per-op v/u reset restricted to the composer slice
    self._layout_span = span
    self._pool1 = substrate                                 # criterion-3: composer._pool1.bridge IS the pool bridge
    self._merged = pool                                     # parity with Pool1BoundComposer's anti-cheat attribute
    # Any store CSR cached against the pre-rebase n_total is stale; force a rebuild on first read.
    self._csr_cache = {}
    self._store_csr = None
    self._store_dirty = True


def _pool1_onebrain_class():
    """Lazily build + cache the `Pool1BoundOneBrainComposer` subclass of `OneBrainComposer`."""
    global _POOL1_ONEBRAIN_CLASS
    if _POOL1_ONEBRAIN_CLASS is None:
        from research.runners.one_brain_composer import OneBrainComposer
        _POOL1_ONEBRAIN_CLASS = type("Pool1BoundOneBrainComposer", (OneBrainComposer,),
                                     {"__init__": _pool1_onebrain_init, "__doc__": _POOL1_ONEBRAIN_DOC})
    return _POOL1_ONEBRAIN_CLASS


def make_pool1_onebrain_composer(seed: int = 42, D: int = 128, vocab=None, k_max: int = 32,
                                 enable_attributed: bool = False, vocab_headroom: int = 128,
                                 **ob_kwargs):
    """Build the production-default OneBrainComposer BOUND to pool #1 (its RF recall/store on the shared bridge; its
    parser on a private bridge). Registers the composer's full layout span as the pool's "onebrain_composer" region
    BEFORE the process-global substrate is first built (so surprise/world-model join the SAME pool), then binds. The
    `ob_kwargs` are exactly the `OneBrainComposer` kwargs the agent would otherwise pass (grounded_codes,
    enable_multiframe, enable_spiking_cleanup, integrated_loop, ...)."""
    global _ONEBRAIN_SPAN
    span = _onebrain_layout_span(seed, D, vocab, k_max, enable_attributed, vocab_headroom)
    _ONEBRAIN_SPAN = int(span)                              # registered so get_merged_substrate reserves the region
    substrate = get_merged_substrate(seed)
    substrate.ensure_built()
    base = int(np.asarray(substrate.onebrain_composer_idx()).min())
    cls = _pool1_onebrain_class()
    return cls(substrate, rf_base=base, seed=seed, D=D, vocab=vocab, k_max=k_max,
               enable_attributed=enable_attributed, vocab_headroom=vocab_headroom, **ob_kwargs)
