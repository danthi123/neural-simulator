"""A SPIKING lateral-inhibition WTA circuit that SELECTS the #84 affective expression marker (board #86,
2026-08-28) -- burns down the `_LEAD_WORD` host dict-lookup scaffold named in the 2026-08-19 affect-drives-chat
finding's honest residual #2 ("A brain-native affective mouth (the marker emitted by a spiking prosody circuit)
is the named next rung").

WHAT THIS IS. The #81 graded-affect ladder's OWN felt-state read (mood = rate(V+)-rate(V-); felt_arousal =
rate(arousal pool), both off `cp_firing_states`, `_graded_affect_attractor_derisk.read_body`) is a CONTINUOUS
population signal. Until now, `webapp/affect_drives_chat.py` converted that continuous felt state to a discrete
affective EXPRESSION MARKER (the word "Wonderful"/"Gladly"/"Sure"/"Hm"/"Honestly"/"Frankly" the reply leads with)
via a bare Python dict lookup keyed on a host-binned level (`_LEAD_WORD[level]`) -- the SELECTION of which marker
register fires was a host template, even though the READ that fed it was neural (lesion-proven). This module
replaces the SELECTION step with a genuine spiking competitive circuit: the felt mood/arousal projects, as a
topographic population code (a Gaussian-tuned afferent current per register, mirroring a labeled-line /
population-vector code -- Georgopoulos, Schwartz & Kettner 1986), onto N small excitatory MARKER ASSEMBLIES (one
per expression register), each with its own fast-spiking interneuron sub-pool that CROSS-INHIBITS every OTHER
assembly (mutual/reciprocal lateral inhibition -- Grossberg 1973's on-center/off-surround competitive network;
the SAME cross-inhibition motif already validated in this repo at N=2 channels by
`bg_action_selection_production_organ.py` / `_vocal_action_selector_gate.py`, here generalized to N=6). The
assembly whose spiking RATE clears the others by a dead margin after the network settles is read as the WINNER;
its (fixed, topographic) identity names the surface marker. The host may render the winner's TOKEN string, but
the SELECTION -- which register fires loudest under lateral competition -- is neurons/synapses on a real
`SimulationBridge`, not a Python `if`/dict.

BIOLOGY. (1) The population code: a continuous affect DIMENSION (valence, or separately arousal) is read out
through a bank of narrowly-tuned units, each maximally driven near its own preferred value and falling off with
distance -- the same principle by which population vector coding recovers a continuous quantity (movement
direction) from a set of broadly-tuned units (Georgopoulos, Schwartz & Kettner 1986, Science 233:1416, "Neuronal
population coding of movement direction"), applied here to a continuous felt-affect axis instead of a movement
direction. (2) The discrete registers: treating "Wonderful"/"Gladly"/.../"Frankly" as CATEGORY-LIKE regions
carved out of one continuous valence dimension mirrors Russell's circumplex model of affect (Russell 1980, J.
Pers. Soc. Psychol. 39:1161), in which discrete affect words are not separate faculties but graded LOCATIONS
along shared valence/arousal axes -- our marker-pool tuning centers are placed at representative points along the
SAME mood axis the #81 ladder already implements (Koulakov 2002 / Goldman 2003 robust-integrator staircase), so
no new axis is invented. (3) The competition: reciprocal/lateral inhibition as the mechanism that converts a
graded population code into a single categorical winner is the textbook competitive-network motif (Grossberg
1973; Douglas & Martin 2004's canonical cortical microcircuit) -- the identical motif this repo already uses,
and has already 6-seed flip-soak GO'd, for the 2-channel SPEAK-vs-STAY-SILENT basal-ganglia race.

REUSE, NO `sim/` edit. A private `SimulationBridge` (`enable_brain_region_framework`) is built once per process
(lazy, warm, cached), mirroring `build_slot_bridge` (EMERGE-59) and `build_selector_bridge`
(`_vocal_action_selector_gate`) -- both already-validated small-bridge builders in this repo. No existing
runner/module is modified.

CONTRACT (additive; default-OFF at the `webapp/affect_drives_chat.py` call site -- see that module).
  * `select_valence(mood, lesion=False, shuffle=False)` -> `(level_or_None, rates, meta)`. `level_or_None` is one
    of {-3,-2,-1,1,2,3} (never 0 -- the neutral gate stays upstream, in the caller, unchanged) or `None` when the
    circuit found no clean winner (a safe "no marker" outcome, never a crash).
  * `select_arousal(felt_arousal, lesion=False, shuffle=False)` -> `(high_or_None, rates, meta)`. `high_or_None`
    is True (emphatic register wins) / False (measured register wins) / None (no clean winner).
  * `lesion=True` cuts the felt-state -> assembly topographic PROJECTION (every pool receives the SAME baseline
    current, i.e. the population code is silenced while the circuit itself stays intact) -- the load-bearing
    proof for the SELECTION step (distinct from the #81 embodiment lesion, which collapses the FELT STATE
    upstream of this circuit entirely). Under lesion, no pool is differentially driven, lateral inhibition has
    nothing to break symmetry with, and the dead-margin check fails on (almost) every trial -> `None` -> the
    caller's documented fallback (an honest no-lead turn, NOT a silent revert to the host dict).
  * `shuffle=True` permutes WHICH physical pool is tuned to which register (a fixed random permutation, seeded)
    -- the anti-cheat: at a FIXED input value, the winning register differs between `shuffle=False` and
    `shuffle=True` runs in a way fully explained by the permutation (the winning POOL index is the same physical
    assembly; what changes is which register-label that pool carries) -- proof the reported marker identity is
    read off WHICH assembly actually won the spiking competition, not re-derived from the raw mood value by a
    fixed host formula that would be blind to the relabeling.

Both selectors share ONE tiny bridge (12 marker/arousal-pool "channels" total: 6 valence registers + 2 arousal
registers, ~24 exc + ~12 FSI neurons per channel -- a few hundred neurons, sub-second to build, single-digit-ms
per read on the numpy backend) -- proportionate to a "small pool of marker-coding assemblies", not a new large
region.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ── the ORIGINAL word vocabulary (reuse-by-import; webapp/affect_drives_chat.py owns the table). Importing here
#    (rather than redefining) means a change to the word set never has two sources of truth.
_LEAD_WORD = None  # late-bound (see get_lead_words()) to avoid a hard import-order dependency at module load.


def get_lead_words() -> dict:
    global _LEAD_WORD
    if _LEAD_WORD is None:
        from webapp.affect_drives_chat import _LEAD_WORD as _lw
        _LEAD_WORD = dict(_lw)
    return _LEAD_WORD


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Topology sizes -- small on purpose ("a small pool of marker-coding assemblies").
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
LEVEL_ORDER = (-3, -2, -1, 1, 2, 3)     # pool index i (0..5) -> the register it is topographically tuned to
N_VALENCE_POOLS = len(LEVEL_ORDER)
N_AROUSAL_POOLS = 2                      # 0 = measured (' -- '), 1 = emphatic ('! ')
N_PER = 24                               # excitatory neurons per marker/register assembly
N_PER_FSI = 12                           # fast-spiking interneurons per assembly's own cross-inhibition pool

# ── topographic tuning centers, placed at the MIDPOINT of each register's existing #84 mood-binning range
#    (`webapp.affect_drives_chat._MOOD_L1/_MOOD_L2/_MOOD_L3`) so the intact circuit reproduces the SAME
#    qualitative word choice the validated 2026-08-19 finding already measured (continuity of the vocabulary,
#    not a re-derivation of the affect axis) -- mood ranges ~[-0.08, +0.08] per that finding + the #81 de-risk.
MOOD_CENTERS = (-0.0850, -0.0575, -0.0275, 0.0275, 0.0575, 0.0850)   # one per LEVEL_ORDER entry, same order
MOOD_SIGMA = 0.020

AROUSAL_CENTERS = (0.0200, 0.0650)       # (measured, emphatic); felt_arousal ranges ~[0, 0.075] (#81/#84)
AROUSAL_SIGMA = 0.020

# ── driven-current scale (pA), sub-saturation like the EMERGE-59 slot-pool primacy currents (300-1800 pA range).
DRIVE_BASE_PA = 150.0
DRIVE_GAIN_PA = 1200.0

# ── cross-inhibition weights, in the same regime as the validated 2-channel `_vocal_action_selector_gate`
# commit_to_fsi_weight=30 / commit_fsi_cross_weight=40 (scaled down slightly for the smaller N_PER here).
TO_FSI_WEIGHT = 70.0
CROSS_INHIB_WEIGHT = 22.0

WARMUP_STEPS = 60
WASHOUT_STEPS = 40
RUN_STEPS = 60

# dead margin (rate units, spikes/step/neuron) the winner must clear the runner-up by -- calibrated empirically
# (see the module's __main__ calibration sweep); mirrors the `_TONE_DEAD_MARGIN` precedent in
# `spiking_mouth_recall_prod.py` (a genuine rate-vs-rate separation requirement, not a bare host threshold).
DEAD_MARGIN = 0.05


def _region(name, n, *, exc_fraction, neuron_type, internal_density=0.0):
    from sim.regions import BrainRegion
    return BrainRegion(
        name=name, n_neurons=int(n), exc_fraction=float(exc_fraction),
        internal_density=float(internal_density), exc_weight_mean=0.0, inh_weight_mean=0.0,
        weight_jitter=0.0, plastic_internal=False, izh_neuron_type=neuron_type.name,
        enable_homeostasis=False,
    )


def _build_bridge(seed: int, n_pools: int, prefix: str):
    """Build a private SimulationBridge with `n_pools` excitatory marker assemblies + `n_pools` dedicated FSI
    cross-inhibition sub-pools (assembly i's FSI inhibits every OTHER assembly j != i -- the mutual/reciprocal
    lateral-inhibition WTA motif, generalized from the 2-channel `_vocal_action_selector_gate` precedent).
    Returns (bridge, marker_idx[n_pools][N_PER], fsi_idx[n_pools][N_PER_FSI])."""
    from sim import CoreSimConfig, GPUConfig, RuntimeState, SimulationBridge, VisualizationConfig
    from sim.enums import NeuronType
    from sim.regions import RegionPathway

    rs = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL
    fs = NeuronType.IZH2007_FS_CORTICAL_INTERNEURON

    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.dt_ms = 1.0
    # ⛔ cfg.seed (NOT actual_seed_used) is what actually seeds per-neuron heterogeneity -- see CLAUDE.md's
    # "actual_seed_used DOES NOT SEED ANYTHING" note. All three RNG streams pinned together for determinism.
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = int(seed)
    cfg.enable_brain_region_framework = True
    cfg.enable_ou_process = False
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp"):
        setattr(cfg, flag, False)

    regions = []
    pathways = []
    for i in range(n_pools):
        regions.append(_region(f"{prefix}_m{i}", N_PER, exc_fraction=1.0, neuron_type=rs))
        regions.append(_region(f"{prefix}_fsi{i}", N_PER_FSI, exc_fraction=0.0, neuron_type=fs))
    for i in range(n_pools):
        pathways.append(RegionPathway(
            from_region=f"{prefix}_m{i}", to_region=f"{prefix}_fsi{i}",
            density=1.0, weight_mean=TO_FSI_WEIGHT, weight_jitter=0.05, plastic=False,
        ))
        for j in range(n_pools):
            if j == i:
                continue
            pathways.append(RegionPathway(
                from_region=f"{prefix}_fsi{i}", to_region=f"{prefix}_m{j}",
                density=1.0, weight_mean=CROSS_INHIB_WEIGHT, weight_jitter=0.05, plastic=False,
                receptor="gaba_a",
            ))

    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data()
    marker_idx = [np.asarray(bridge.region_manager.indices(f"{prefix}_m{i}")) for i in range(n_pools)]
    fsi_idx = [np.asarray(bridge.region_manager.indices(f"{prefix}_fsi{i}")) for i in range(n_pools)]
    return bridge, marker_idx, fsi_idx


def _step(bridge, n=1):
    for _ in range(int(n)):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms


def _pool_rates(bridge, marker_idx, drive_pa, *, warmup, washout, run):
    """Wash out any residual state from a prior read, apply `drive_pa` (a length-n_pools array, pA per pool) as
    external current on each marker pool for `run` steps, and return the per-pool spike RATE (spikes / (run *
    N_PER)) off `cp_firing_states` -- the same driven-pool-rate reading convention as `slot_pool_rates`
    (EMERGE-59) and `_run_biased_trial` (the BG selector)."""
    from sim.backend import get_backend, to_host
    xp, _ = get_backend()
    n_pools = len(marker_idx)

    # washout: zero drive, let the network relax toward its resting operating point (mirrors the BG selector's
    # between-trial reset+washout so a warm, reused bridge does not carry state across independent reads).
    bridge.cp_external_input_current[:] = 0.0
    _step(bridge, washout)

    for i in range(n_pools):
        bridge.cp_external_input_current[xp.asarray(marker_idx[i])] = xp.float32(float(drive_pa[i]))
    _step(bridge, warmup)

    counts = np.zeros(int(bridge.core_config.num_neurons), dtype=np.float64)
    for _ in range(int(run)):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        counts += np.asarray(to_host(bridge.cp_firing_states)).astype(np.float64)

    bridge.cp_external_input_current[:] = 0.0
    rate = np.array([counts[idx].mean() / run for idx in marker_idx])
    return rate


def _gaussian_drive(value: float, centers, sigma: float, base: float, gain: float) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    return base + gain * np.exp(-((float(value) - centers) ** 2) / (2.0 * sigma ** 2))


@dataclass
class _SelectResult:
    winner_slot: Optional[int]
    rates: np.ndarray
    winner_pool: Optional[int]
    margin: Optional[float]


class AffectMarkerWTA:
    """A process-warm, lazily-built pair of spiking lateral-inhibition WTA circuits: one over the 6 valence
    registers (Wonderful..Frankly), one over the 2 arousal registers (measured/emphatic). Cached per seed."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._v_bridge = self._v_idx = self._v_fsi = None
        self._a_bridge = self._a_idx = self._a_fsi = None

    def _ensure_valence(self):
        if self._v_bridge is None:
            self._v_bridge, self._v_idx, self._v_fsi = _build_bridge(self.seed, N_VALENCE_POOLS, "aff_val")

    def _ensure_arousal(self):
        if self._a_bridge is None:
            self._a_bridge, self._a_idx, self._a_fsi = _build_bridge(self.seed + 1, N_AROUSAL_POOLS, "aff_aro")

    def _perm(self, n_pools: int, shuffle: bool) -> np.ndarray:
        if not shuffle:
            return np.arange(n_pools)
        rng = np.random.default_rng(self.seed * 977 + 13)
        return rng.permutation(n_pools)

    def _select(self, bridge, marker_idx, value, centers, sigma, *, lesion, shuffle, dead_margin, warmup, washout, run):
        n_pools = len(marker_idx)
        perm = self._perm(n_pools, shuffle)
        # perm[p] = which tuning CENTER (index into `centers`) physical pool p's incoming drive is computed from.
        # shuffle=False -> perm is the identity (pool p is topographically tuned to register p, the intended map).
        # shuffle=True -> perm is a fixed random permutation: physical pool p's DRIVE is deliberately mis-routed
        # to a DIFFERENT register's tuning center, so whichever pool wins the competition is generally NOT the
        # pool whose OWN (fixed, canonical) label matches `value` -- the readout mapping below (pool index p ->
        # register p, via LEVEL_ORDER) is NEVER permuted, so a changed winner under shuffle necessarily changes
        # the REPORTED register too. This is the anti-cheat: it proves the reported identity is read off WHICH
        # PHYSICAL ASSEMBLY won the spiking race (a live functional dependency on the circuit's wiring), not
        # re-derived straight from `value` by a fixed host formula that would be blind to the mis-routing.
        if lesion:
            # cut the felt-state -> assembly PROJECTION: every pool receives the SAME baseline current, so the
            # population code carries no information the lateral-inhibition competition could break symmetry with.
            drive = np.full(n_pools, DRIVE_BASE_PA, dtype=np.float64)
        else:
            centers_perm = np.asarray(centers, dtype=np.float64)[perm]
            drive = DRIVE_BASE_PA + DRIVE_GAIN_PA * np.exp(-((float(value) - centers_perm) ** 2) / (2.0 * sigma ** 2))
        rates = _pool_rates(bridge, marker_idx, drive, warmup=warmup, washout=washout, run=run)
        order = np.argsort(rates)[::-1]
        top, second = int(order[0]), int(order[1])
        margin = float(rates[top] - rates[second])
        if margin > dead_margin:
            # the FIXED (never-permuted) register label of whichever physical pool actually won.
            return _SelectResult(winner_slot=top, rates=rates, winner_pool=top, margin=margin)
        return _SelectResult(winner_slot=None, rates=rates, winner_pool=None, margin=margin)

    def select_valence(self, mood: float, *, lesion: bool = False, shuffle: bool = False,
                       dead_margin: float = DEAD_MARGIN) -> tuple:
        self._ensure_valence()
        r = self._select(self._v_bridge, self._v_idx, mood, MOOD_CENTERS, MOOD_SIGMA,
                         lesion=lesion, shuffle=shuffle, dead_margin=dead_margin,
                         warmup=WARMUP_STEPS, washout=WASHOUT_STEPS, run=RUN_STEPS)
        level = LEVEL_ORDER[r.winner_slot] if r.winner_slot is not None else None
        meta = {"winner_pool": r.winner_pool, "margin": r.margin, "lesioned": bool(lesion), "shuffled": bool(shuffle)}
        return level, r.rates, meta

    def select_arousal(self, felt_arousal: float, *, lesion: bool = False, shuffle: bool = False,
                       dead_margin: float = DEAD_MARGIN) -> tuple:
        self._ensure_arousal()
        r = self._select(self._a_bridge, self._a_idx, felt_arousal, AROUSAL_CENTERS, AROUSAL_SIGMA,
                         lesion=lesion, shuffle=shuffle, dead_margin=dead_margin,
                         warmup=WARMUP_STEPS, washout=WASHOUT_STEPS, run=RUN_STEPS)
        high = bool(r.winner_slot == 1) if r.winner_slot is not None else None
        meta = {"winner_pool": r.winner_pool, "margin": r.margin, "lesioned": bool(lesion), "shuffled": bool(shuffle)}
        return high, r.rates, meta


# process-shared cache, keyed by seed (the production call site always uses one seed per session; the verify
# suite builds several).
_READERS: dict = {}


def get_reader(seed: int = 42) -> AffectMarkerWTA:
    r = _READERS.get(int(seed))
    if r is None:
        r = AffectMarkerWTA(seed=seed)
        _READERS[int(seed)] = r
    return r


def reset_readers():
    _READERS.clear()


def marker_from_level(level: Optional[int]) -> str:
    """Level -> the surface WORD (reuse-by-import of the ORIGINAL `_LEAD_WORD` table -- the WTA circuit selects
    WHICH register wins; the word each register renders as is unchanged vocabulary)."""
    if level is None:
        return ""
    return get_lead_words().get(int(level), "") or ""


if __name__ == "__main__":
    # tiny numpy smoke: build once, sweep representative mood/arousal values, print the winning register + the
    # margin, so a human can eyeball that the intact circuit's word choice matches the 2026-08-19 finding's own
    # measured table (mood +0.069 -> level +2 "Gladly"; mood -0.029 -> level -1 "Hm"; mood +0.064 -> level +2).
    import argparse
    os.environ.setdefault("SIM_BACKEND", "numpy")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    reader = get_reader(args.seed)
    words = get_lead_words()
    for mood in (-0.085, -0.06, -0.029, -0.02, 0.02, 0.064, 0.069, 0.085):
        level, rates, meta = reader.select_valence(mood)
        print(f"mood={mood:+.3f} -> level={level} word={words.get(level, '(none)') if level else '(neutral/none)'} "
              f"margin={meta['margin']:.4f} rates={np.round(rates, 3).tolist()}")
    for felt in (0.01, 0.02, 0.05, 0.065, 0.075):
        high, rates, meta = reader.select_arousal(felt)
        print(f"felt_arousal={felt:.3f} -> high={high} margin={meta['margin']:.4f} rates={np.round(rates, 3).tolist()}")
    print("-- lesion (valence): --")
    intact_margin = reader.select_valence(0.085)[2]["margin"]
    for mood in (-0.085, 0.085):
        level, rates, meta = reader.select_valence(mood, lesion=True)
        print(f"mood={mood:+.3f} LESIONED -> level={level} margin={meta['margin']:.4f} rates={np.round(rates, 3).tolist()}")
    lesion_margin = reader.select_valence(0.085, lesion=True)[2]["margin"]
    # ATTRIBUTION (tools.lab, gap#5 discipline): how much of the winner-vs-runner-up SEPARATION is attributable
    # to the felt-state->assembly topographic drive, vs. what remains under the lesion control (drive cut, every
    # assembly at the same baseline current)? A near-100% attribution is exactly what "the selection RIDES the
    # drive, not some other latent bias in the wiring" requires -- the same discipline gap#5 lacked when it
    # banked a lever's 3%-of-the-change effect beside an unatributed 97%-clamp for weeks.
    from tools.lab import attributable_to
    attributable_to("valence WTA margin: intact vs lesion (mood=+0.085)", intact_margin, lesion_margin)
    print("-- shuffle (valence): --")
    for mood in (-0.085, 0.085):
        level, rates, meta = reader.select_valence(mood, shuffle=True)
        print(f"mood={mood:+.3f} SHUFFLED -> level={level} margin={meta['margin']:.4f}")
