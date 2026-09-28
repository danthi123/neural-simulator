"""N-channel generalisation of the two-channel basal-ganglia action selector (the named next rung of the tone
seam).

THE NAMED NEXT RUNG (webapp/affect_tone_selection.py docstring, lines 33-37): the tone-selection seam's host
argmin (`select_by_mood`) is "NOT on the substrate. The named next rung is to run it as the basal-ganglia race
(research/runners/bg_action_selection_production_organ.py generalised to N channels, candidate salience =
congruence read off the organ populations)." This module IS that generalisation: the SAME topology as
`_vocal_action_selector_gate.build_selector_bridge` (a shared practice-arousal drive + a per-channel BG cascade +
cross-inhibition), with exactly two changes, both faithful to the two-channel mechanism:

  * the per-channel cross-inhibition generalises from `commit_fs_c -> commit_{1-c}` (the single other channel) to
    `commit_fs_c -> commit_{every other channel}` (and likewise the striatal-FSI cross-inhibition onto every other
    channel's D1/D2 pool);
  * the commit read generalises from "the other channel is suppressed below `clean_loser_ratio`" to "EVERY other
    channel is suppressed below `clean_loser_ratio`".

At N=2 this is byte-identical to the two-channel topology and commit read (the "every other channel" set is the
single other channel). At N>2 it is the faithful N-way race.

WHO OWNS THE SELECTION (the 2026-09-25 host-share question): the host NEVER does argmax/argmin over the candidates
to pick a winner. It supplies each channel's SALIENCE -- a scalar cortical afferent, here the congruence of the
candidate's organ-evoked valence with the held mood (`congruence_saliences`) -- and it reads the NEURAL winner:
the channel whose motor pool first crosses `commit_threshold_spikes` with every other channel suppressed below the
clean-loser ratio. No clean winner -> NO commit (`winner is None`); the host may then fall back to its declared
shortcut, but that fallback is the host's, not the race's. The anti-cheat is the finding's own: the
NO-AROUSAL lesion removes the shared practice-arousal drive, which leaves the MSNs sub-threshold at ANY salience
(so nothing commits -- the selection cannot be the host's), and the NO-DIRECT-PATH lesion cuts the D1->GPi gate.

No `sim/` edit; the de-risk primitives are reused by import from `_vocal_action_selector_gate`.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sim.backend import get_backend, to_host
from sim.enums import NeuronModel, NeuronType
from sim.regions import BrainRegion, RegionPathway

from research.runners._vocal_action_selector_gate import (
    SelectorConfig, selector_config, _region, _indices, _step, DIRECT_PATH_GATE,
)
from tools.lab import attributable_to

# the same salience->excitability gain as the two-channel production selector
# (bg_action_selection_production_organ.SALIENCE_GAIN_PA), so the N-way race operates at the same point.
SALIENCE_GAIN_PA = 600.0

# The fixed candidate set for the de-risk: the draft (the safe default, index 0, wins ties) + the four restyle
# tones, with organ-plausible evoked valences spanning both signs at two intensities (matching webapp STYLES).
CAND_STYLES = ["draft", "pos2", "pos1", "neg1", "neg2"]
CAND_VALENCES = [0.0, 0.7, 0.3, -0.3, -0.7]
N_CHANNELS = len(CAND_VALENCES)


def _set_nchannel_tonic(bridge, config, channels):
    """The equal tonic afferents, generalised to an explicit channel list (the two-channel helper hardcodes
    CHANNELS=(0,1), which is wrong for N!=2)."""
    xp, _ = get_backend()
    bridge.cp_external_input_current[:] = xp.float32(0.0)
    for ch in channels:
        bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, f"gpi_{ch}"))
        ] = xp.float32(config.gpi_tonic_pA)
        bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, f"thal_{ch}"))
        ] = xp.float32(config.thalamus_tonic_pA)


def build_nchannel_bridge(n_channels, seed, config=SelectorConfig(), *, commit_enable_nmda=True,
                          commit_internal_density=0.35, commit_internal_weight=0.5):
    """The N-channel generalisation of `build_selector_bridge`. At N=2 the region set and pathway set are
    byte-identical to the two-channel bridge (the "every other channel" loops reduce to the single other channel).

    `commit_internal_density` / `commit_internal_weight` default to the two-channel values (0.35 / 0.5) so N=2 stays
    byte-identical; they are exposed so the N>2 commit pool's self-sustaining recurrence can be tuned (the real BG
    commit is a TRANSIENT thalamic burst, not a self-sustaining firing -- see the N>2 wall in the module docstring)."""
    from sim import CoreSimConfig, GPUConfig, RuntimeState, SimulationBridge
    from sim import VisualizationConfig

    n = int(n_channels)
    if n < 2:
        raise ValueError("the N-channel race needs >= 2 competing channels")
    channels = tuple(range(n))

    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    cfg.enable_brain_region_framework = True
    cfg.enable_ou_process = True
    cfg.ou_mean_current_pA = 0.0
    cfg.ou_std_current_pA = float(config.ou_sigma_pA)
    cfg.ou_tau_ms = 15.0
    for flag in (
        "enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
        "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp",
    ):
        setattr(cfg, flag, False)

    rs = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL
    fs = NeuronType.IZH2007_FS_CORTICAL_INTERNEURON
    d1 = NeuronType.IZH2007_STRIATAL_MSN_D1
    d2 = NeuronType.IZH2007_STRIATAL_MSN_D2
    gpe = NeuronType.IZH2007_GPE_PACEMAKER
    gpi = NeuronType.IZH2007_GPI_OUTPUT
    stn = NeuronType.IZH2007_STN_BURST
    thal = NeuronType.IZH2007_THALAMIC_RELAY

    regions = [
        _region("practice_arousal", config.n_practice, exc_fraction=1.0, neuron_type=rs),
        _region("selector_stn", config.n_stn, exc_fraction=1.0, neuron_type=stn),
        _region("selector_reset", config.n_reset, exc_fraction=0.0, neuron_type=fs),
    ]
    for ch in channels:
        channel_regions = [
            _region(f"proposal_{ch}", config.n_proposal, exc_fraction=1.0, neuron_type=rs),
        ]
        if config.enable_striatal_fsi:
            channel_regions.append(_region(
                f"str_fsi_{ch}", config.n_commit_fs, exc_fraction=0.0, neuron_type=fs,
            ))
        channel_regions.extend([
            _region(f"str_d1_{ch}", config.n_striatum, exc_fraction=0.0, neuron_type=d1),
            _region(f"str_d2_{ch}", config.n_striatum, exc_fraction=0.0, neuron_type=d2),
            _region(f"gpe_{ch}", config.n_gpe, exc_fraction=0.0, neuron_type=gpe),
            _region(f"gpi_{ch}", config.n_gpi, exc_fraction=0.0, neuron_type=gpi),
            _region(f"thal_{ch}", config.n_thalamus, exc_fraction=1.0, neuron_type=thal),
            _region(f"commit_{ch}", config.n_commit, exc_fraction=1.0, neuron_type=rs,
                    internal_density=commit_internal_density, internal_weight=commit_internal_weight,
                    enable_nmda=commit_enable_nmda),
            _region(f"commit_fs_{ch}", config.n_commit_fs, exc_fraction=0.0, neuron_type=fs),
            _region(f"motor_{ch}", config.n_motor, exc_fraction=1.0, neuron_type=rs),
        ])
        regions.extend(channel_regions)

    pathways = []
    for ch in channels:
        others = [o for o in channels if o != ch]
        pathways.append(RegionPathway(
            from_region="practice_arousal", to_region=f"proposal_{ch}",
            density=1.0, weight_mean=config.arousal_to_proposal_weight,
            weight_jitter=0.0, plastic=False,
        ))
        if config.enable_striatal_fsi:
            pathways.append(RegionPathway(
                from_region=f"proposal_{ch}", to_region=f"str_fsi_{ch}",
                density=1.0, weight_mean=config.proposal_to_fsi_weight,
                weight_jitter=0.0, plastic=False,
            ))
            for o in others:
                pathways.extend([
                    RegionPathway(
                        from_region=f"str_fsi_{ch}", to_region=f"str_d1_{o}",
                        density=1.0, weight_mean=config.fsi_to_msn_weight,
                        weight_jitter=0.0, plastic=False, receptor="gaba_a",
                    ),
                    RegionPathway(
                        from_region=f"str_fsi_{ch}", to_region=f"str_d2_{o}",
                        density=1.0, weight_mean=config.fsi_to_msn_weight,
                        weight_jitter=0.0, plastic=False, receptor="gaba_a",
                    ),
                ])
        pathways.extend([
            RegionPathway(
                from_region=f"proposal_{ch}", to_region=f"str_d1_{ch}",
                density=config.proposal_to_msn_density, weight_mean=config.proposal_to_msn_weight,
                weight_jitter=0.05, plastic=False,
            ),
            RegionPathway(
                from_region=f"proposal_{ch}", to_region=f"str_d2_{ch}",
                density=config.proposal_to_msn_density, weight_mean=config.proposal_to_msn_weight,
                weight_jitter=0.05, plastic=False,
            ),
            RegionPathway(
                from_region=f"str_d1_{ch}", to_region=f"gpi_{ch}",
                density=1.0, weight_mean=config.d1_to_gpi_weight,
                weight_jitter=0.05, plastic=False, transmission_gate=DIRECT_PATH_GATE,
            ),
            RegionPathway(
                from_region=f"str_d2_{ch}", to_region=f"gpe_{ch}",
                density=0.60, weight_mean=2.5, weight_jitter=0.05, plastic=False,
            ),
            RegionPathway(
                from_region=f"gpe_{ch}", to_region="selector_stn",
                density=0.30, weight_mean=1.5, weight_jitter=0.05, plastic=False,
            ),
            RegionPathway(
                from_region="selector_stn", to_region=f"gpi_{ch}",
                density=0.40, weight_mean=1.0, weight_jitter=0.05, plastic=False,
            ),
            RegionPathway(
                from_region=f"gpi_{ch}", to_region=f"thal_{ch}",
                density=1.0, weight_mean=config.gpi_to_thalamus_weight,
                weight_jitter=0.05, plastic=False, receptor="gaba_a",
            ),
            RegionPathway(
                from_region=f"thal_{ch}", to_region=f"commit_{ch}",
                density=1.0, weight_mean=config.thalamus_to_commit_weight,
                weight_jitter=0.05, plastic=False,
            ),
            RegionPathway(
                from_region=f"commit_{ch}", to_region=f"commit_fs_{ch}",
                density=1.0, weight_mean=config.commit_to_fsi_weight,
                weight_jitter=0.0, plastic=False,
            ),
            RegionPathway(
                from_region=f"commit_{ch}", to_region=f"motor_{ch}",
                density=1.0, weight_mean=config.commit_to_motor_weight,
                weight_jitter=0.05, plastic=False,
            ),
        ])
        # cross-inhibition: this channel's commit_fs suppresses EVERY other channel's commit pool (the two-channel
        # `commit_fs_c -> commit_{1-c}`, generalised).
        for o in others:
            pathways.append(RegionPathway(
                from_region=f"commit_fs_{ch}", to_region=f"commit_{o}",
                density=1.0, weight_mean=config.commit_fsi_cross_weight,
                weight_jitter=0.0, plastic=False, receptor="gaba_a",
            ))
        reset_targets = [
            f"proposal_{ch}", f"str_d1_{ch}", f"str_d2_{ch}",
            f"thal_{ch}", f"commit_{ch}", f"motor_{ch}",
        ]
        if config.enable_striatal_fsi:
            reset_targets.append(f"str_fsi_{ch}")
        for target in reset_targets:
            pathways.append(RegionPathway(
                from_region="selector_reset", to_region=target,
                density=0.70, weight_mean=16.0, weight_jitter=0.0,
                plastic=False, receptor="gaba_a",
            ))

    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    xp, _ = get_backend()
    proposal_indices = np.concatenate([_indices(bridge, f"proposal_{ch}") for ch in channels])
    bridge.cp_ou_neuron_mask = xp.zeros(int(cfg.num_neurons), dtype=bool)
    bridge.cp_ou_neuron_mask[xp.asarray(proposal_indices)] = True
    return bridge


def _run_nchannel_trial(bridge, config, saliences, channels, *, arousal=True, gain=SALIENCE_GAIN_PA):
    """ONE N-way basal-ganglia race with a per-candidate salience bias. Mirrors the two-channel `_run_biased_trial`
    (the salience bias is a striatal-D1 EXCITABILITY modulation, GATED BY the shared arousal state), generalising
    the commit read to "every other channel suppressed below `clean_loser_ratio`"."""
    xp, _ = get_backend()
    _set_nchannel_tonic(bridge, config, channels)
    motor_idx = {ch: _indices(bridge, f"motor_{ch}") for ch in channels}
    if arousal:
        bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, "practice_arousal"))
        ] = xp.float32(config.practice_pA)
        # per-candidate salience -> a modulatory excitability bias on that channel's striatal D1 MSN pool, gated by
        # the shared arousal (the NO-AROUSAL lesion removes this gain AND the proposal->D1 barrage -> nothing
        # commits at any salience).
        d1_idx = {ch: xp.asarray(_indices(bridge, f"str_d1_{ch}")) for ch in channels}
        for ch in channels:
            idx = d1_idx[ch]
            bridge.cp_external_input_current[idx] = (
                bridge.cp_external_input_current[idx] + xp.float32(float(saliences[ch]) * gain)
            )

    n = len(channels)
    counts = np.zeros(n, dtype=np.int64)
    first_crossing = None
    simultaneous = False
    decision_step = None
    for step in range(int(config.action_steps)):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        previous = counts.copy()
        for i, ch in enumerate(channels):
            counts[i] += int(firing[motor_idx[ch]].sum())
        crossed = [
            i for i in range(n)
            if previous[i] < config.commit_threshold_spikes <= counts[i]
        ]
        if first_crossing is None and len(crossed) == 1:
            first_crossing = int(crossed[0])
            decision_step = int(step)
            break
        elif first_crossing is None and len(crossed) > 1:
            simultaneous = True
            decision_step = int(step)
            break

    winner = None
    loser_ratio = None
    if first_crossing is not None and not simultaneous:
        # clean iff EVERY other channel is suppressed below the clean-loser ratio (the two-channel check, generalised).
        loser_ratio = float(
            max(counts[i] for i in range(n) if i != first_crossing) / max(1, counts[first_crossing])
        )
        if loser_ratio <= config.clean_loser_ratio:
            winner = int(channels[first_crossing])

    # between-trial reset + washout (verbatim de-risk behavior so a cached bridge returns to the resting point).
    _set_nchannel_tonic(bridge, config, channels)
    bridge.cp_external_input_current[
        xp.asarray(_indices(bridge, "selector_reset"))
    ] = xp.float32(config.reset_pA)
    _step(bridge, config.reset_steps)
    _set_nchannel_tonic(bridge, config, channels)
    _step(bridge, config.washout_steps)
    return {
        "winner": winner,
        "first_crossing": int(channels[first_crossing]) if first_crossing is not None else None,
        "committed": winner is not None,
        "simultaneous": bool(simultaneous),
        "motor_spikes": counts.tolist(),
        "loser_ratio": loser_ratio,
        "decision_step": decision_step,
    }


def congruence_saliences(cand_valences, v_held):
    """The per-candidate salience = the congruence of the candidate's organ-evoked valence with the held mood. A
    scalar read-off (the declared cortical afferent): an exact valence match -> 1.0, an opposite valence -> 0.0.
    The host computes these scalars; the SELECTION (which channel wins) is the spiking race, never a host argmin."""
    vh = float(v_held)
    out = []
    for v in cand_valences:
        c = 1.0 - abs(float(v) - vh) / 2.0
        out.append(max(0.0, min(1.0, c)))
    return out


def host_argmin(cand_valences, v_held):
    """The DECLARED shortcut (`select_by_mood`), reproduced ONLY to compare the race against -- the index whose
    evoked valence is closest to the held valence, ties -> the lowest index. This is NOT the selection; the race is."""
    best, best_d = 0, None
    for i, v in enumerate(cand_valences):
        d = abs(float(v) - float(v_held))
        if best_d is None or d < best_d - 1e-12:
            best, best_d = i, d
    return best


class NToneBGSelector:
    """A warm N-channel BG selector for tone selection. Built ONCE (lazily) on a real `SimulationBridge` from the
    N-channel generalisation of the de-risk v2 topology. `lesion` in {None, 'arousal', 'direct_path'} installs the
    finding's own anti-cheat (same semantics as the two-channel organ). `select_once(saliences)` runs ONE salience-
    biased N-way race -> the selected channel index (or None if no clean winner)."""

    def __init__(self, seed=42, n_channels=N_CHANNELS, version="v2", lesion=None, config=None):
        self.seed = int(seed)
        self.n_channels = int(n_channels)
        self.channels = tuple(range(self.n_channels))
        # config=None -> the version default; a caller may pass a dataclasses.replace(...) override (e.g. the
        # N=5 GPi-tonic operating point) to run the SAME race at a different tonic operating point.
        self.config = config if config is not None else selector_config(version)
        self.lesion = lesion
        self._bridge = None
        self._built = False

    def ensure_built(self):
        if self._built:
            return
        b = build_nchannel_bridge(self.n_channels, self.seed, self.config)
        gate = 0.0 if self.lesion == "direct_path" else 1.0
        b.set_transmission_gate(DIRECT_PATH_GATE, gate)
        _set_nchannel_tonic(b, self.config, self.channels)
        _step(b, self.config.warmup_steps)
        self._bridge = b
        self._built = True

    def select_once(self, saliences) -> dict:
        self.ensure_built()
        arousal = self.lesion != "arousal"
        return _run_nchannel_trial(
            self._bridge, self.config, saliences, self.channels, arousal=arousal
        )


_ORGANS: dict = {}


def get_organ(seed=42, n_channels=N_CHANNELS, lesion=None, config=None) -> NToneBGSelector:
    """Process-shared selector cache, keyed by (seed, n_channels, lesion, tonic) so the intact + lesion variants
    (and different candidate counts / tonic operating points) coexist. config=None -> the version default tonic."""
    tonic = (int(config.gpi_tonic_pA), int(config.thalamus_tonic_pA)) if config is not None else None
    key = (int(seed), int(n_channels), lesion, tonic)
    org = _ORGANS.get(key)
    if org is None:
        org = NToneBGSelector(seed=seed, n_channels=n_channels, lesion=lesion, config=config)
        _ORGANS[key] = org
    return org


def _lesion_for(condition: str):
    return {"intact": None, "no_arousal": "arousal", "no_direct_path": "direct_path"}[condition]


def run_condition(seed, v_held, *, condition="intact", trials=20,
                  cand_valences=CAND_VALENCES, config=None) -> dict:
    """Run the N-way race `trials` times at one held mood + condition; return the winner distribution + the host
    argmin (for comparison) + the congruence saliences that drove the race. config=None -> the version default
    tonic; pass a dataclasses.replace(...) override to run the SAME race at a different tonic operating point."""
    sal = congruence_saliences(cand_valences, v_held)
    n = len(cand_valences)
    org = get_organ(seed=seed, n_channels=n, lesion=_lesion_for(condition), config=config)
    winner_counts = {i: 0 for i in range(n)}
    n_committed = 0
    for _ in range(int(trials)):
        res = org.select_once(sal)
        if res["committed"]:
            n_committed += 1
            winner_counts[int(res["winner"])] += 1
    committed = [i for i in range(n) if winner_counts[i] > 0]
    modal = max(committed, key=lambda i: winner_counts[i]) if committed else None
    return {
        "v_held": float(v_held),
        "condition": condition,
        "trials": int(trials),
        "saliences": [round(s, 4) for s in sal],
        "host_argmin": host_argmin(cand_valences, v_held),
        "n_committed": n_committed,
        "commit_rate": round(n_committed / max(1, int(trials)), 4),
        "winner_counts": winner_counts,
        "modal_winner": modal,
        "matches_host_argmin": (modal is not None and modal == host_argmin(cand_valences, v_held)),
    }


def run_seed(seed, *, v_held_grid=None, trials=20,
             conditions=("intact", "no_arousal", "no_direct_path"),
             cand_valences=CAND_VALENCES, config=None) -> dict:
    """The full de-risk: sweep the held mood, run the race under each condition, and check (a) the race commits
    when intact, (b) the arousal lesion floors the commit (the selection is the spiking race, not the host), (c)
    the winner is affect-dependent (it tracks the most-congruent candidate and CHANGES across the mood sweep).
    config=None -> the version default tonic; pass a dataclasses.replace(...) override to run the de-risk at a
    different tonic operating point (e.g. the N=5 GPi-tonic fix)."""
    if v_held_grid is None:
        v_held_grid = [-0.9, -0.45, 0.0, 0.45, 0.9]
    eff_config = config if config is not None else selector_config("v2")
    rows = []
    for cond in conditions:
        for vh in v_held_grid:
            rows.append(run_condition(seed, vh, condition=cond, trials=trials,
                                      cand_valences=cand_valences, config=config))

    intact = [r for r in rows if r["condition"] == "intact"]
    no_arousal = [r for r in rows if r["condition"] == "no_arousal"]
    no_direct = [r for r in rows if r["condition"] == "no_direct_path"]
    intact_commit = float(np.mean([r["commit_rate"] for r in intact])) if intact else 0.0
    no_arousal_commit = float(np.mean([r["commit_rate"] for r in no_arousal])) if no_arousal else 0.0
    no_direct_commit = float(np.mean([r["commit_rate"] for r in no_direct])) if no_direct else 0.0
    # attribution: WHOSE is the commit? the fraction of the intact commit rate NOT present in the lesioned
    # control (the manipulation removed) -- the gap#5 question in one call, so the load-bearing check below is
    # an attribution, not just a subtraction of two numbers one key apart.
    arousal_attr = attributable_to("arousal @ N-way race", intact_commit, no_arousal_commit)
    direct_attr = attributable_to("direct_path @ N-way race", intact_commit, no_direct_commit)
    match_rate = float(np.mean([1.0 if r["matches_host_argmin"] else 0.0 for r in intact])) if intact else 0.0
    modal_winners = {r["v_held"]: r["modal_winner"] for r in intact}
    distinct_winners = set(w for w in modal_winners.values() if w is not None)

    checks = {
        "race_commits_intact": intact_commit >= 0.9,
        "arousal_is_load_bearing": no_arousal_commit <= intact_commit - 0.75,
        "direct_path_is_load_bearing": (
            no_direct is not None and no_direct_commit <= intact_commit - 0.75
        ),
        "winner_tracks_host_argmin": match_rate >= 0.6,
        "winner_is_affect_dependent": len(distinct_winners) >= 2,
    }
    return {
        "seed": int(seed),
        "n_channels": len(cand_valences),
        "gpi_tonic_pA": float(eff_config.gpi_tonic_pA),
        "thalamus_tonic_pA": float(eff_config.thalamus_tonic_pA),
        "cand_styles": list(CAND_STYLES[:len(cand_valences)]),
        "cand_valences": [float(v) for v in cand_valences],
        "v_held_grid": [float(v) for v in v_held_grid],
        "trials": int(trials),
        "intact_commit_rate": round(intact_commit, 4),
        "no_arousal_commit_rate": round(no_arousal_commit, 4),
        "no_direct_path_commit_rate": round(no_direct_commit, 4),
        "arousal_attribution": (round(arousal_attr, 4) if arousal_attr is not None else None),
        "direct_path_attribution": (round(direct_attr, 4) if direct_attr is not None else None),
        "intact_match_rate_vs_host_argmin": round(match_rate, 4),
        "modal_winner_by_v_held": {str(k): v for k, v in modal_winners.items()},
        "rows": rows,
        "checks": checks,
        "go": bool(all(checks.values())),
    }


def main():
    ap = argparse.ArgumentParser(description="N-channel BG tone-selection race de-risk (affect-dependent + load-bearing)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--quick", action="store_true",
                    help="small mood grid + fewer trials + no direct-path condition (the numpy smoke test)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.quick:
        grid, trials, conditions = [-0.9, 0.0, 0.9], 8, ("intact", "no_arousal")
    else:
        grid, trials, conditions = None, args.trials, ("intact", "no_arousal", "no_direct_path")
    res = run_seed(args.seed, v_held_grid=grid, trials=trials, conditions=conditions)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
    print(json.dumps({
        "seed": res["seed"], "quick": bool(args.quick), "go": res["go"],
        "intact_commit_rate": res["intact_commit_rate"],
        "no_arousal_commit_rate": res["no_arousal_commit_rate"],
        "no_direct_path_commit_rate": res["no_direct_path_commit_rate"],
        "intact_match_rate_vs_host_argmin": res["intact_match_rate_vs_host_argmin"],
        "modal_winner_by_v_held": res["modal_winner_by_v_held"],
        "checks": res["checks"],
    }, indent=2))


if __name__ == "__main__":
    main()
