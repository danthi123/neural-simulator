"""ONE SELECTION SEAM WITH NO HOST AUTHORITY: an interoceptive need state reverses the basal-ganglia winner.

STATUS (coordinator ruling 2026-09-29): a MECHANISM DE-RISK ONLY, dev seeds only, never run on gate seeds. Its
commit read is the host first-crossing rule and several of its gates hold by construction (audit
.claude/worktrees/_audit_2026-09-29/selection-seam.md). The live seam is `bg_action_selection_production_organ`
region `neural-seam` + research/runners/host_share_bg_seam.py.

Plan: docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md, sections 2, 3 and 6 (step 3 of section 7).
Design: docs/drafts/selection-seam/DESIGN.md. Preregistration: research/findings/2026-09-29-need-reversal-selection-seam-PREREGISTRATION.md.

THE SEAM. The production basal-ganglia selector (`research/runners/bg_action_selection_production_organ.py`, Gate-A v2
topology from `_vocal_action_selector_gate.build_selector_bridge`, default-ON as BRAIN_BG_SELECT) chooses between
two actions by a spiking race: shared arousal -> cortical proposals -> striatal D1/D2 -> GPi/GPe/STN -> thalamus ->
commit burst -> motor pool; the first motor pool to cross the commit threshold with the other suppressed is the
winner. Its host authority sits upstream: the host computes a per-candidate SALIENCE from what the candidate IS
(`salience(msg)`), multiplies it by 600 pA and injects it into that candidate's D1 pool, and it pre-filters which
turns even reach the race (`if not (silent_sal > speak_sal): return None`). The race then transmits the host's
scoring. The same pattern scores the tone seam (`congruence_saliences`) and the value choice (`VALUE_HZ_GAIN_PA`).

THIS RUNNER replaces the host scoring with a neural route and measures who owns the choice:

  * two candidate ACTIONS, channel 0 and channel 1 of the unchanged selector; what each action does is known only to
    the WORLD (`WorldCandidates`), which applies the consequence after the neural commit;
  * an interoceptive NEED STATE: two Izhikevich relay populations, `intero_hunger` (AgRP-like) and `intero_satiety`
    (POMC-like), driven by the body's energy deficit h in [0, 1] through an afferent current (the body -> sensory
    boundary the homeostatic-drive and embodied-affect GOs use; h is body state, not a decision);
  * the need reaches the selection through SYNAPSES: intero_hunger -> str_d1 of channel 1, intero_satiety -> str_d1
    of channel 0 (glutamatergic, slow NMDA, frozen; the PVT -> nucleus accumbens route), behind the transmission gate
    `need_value` so it can be lesioned;
  * the host supplies only candidate-agnostic inputs (equal tonic currents, the shared arousal onset, the inter-trial
    reset) and receives only the index of the motor pool that committed.

ARMS
  host_salience (BEFORE; the production pattern): need synapses closed (gate 0); the host computes salience from the
      need AND the candidate's label (`host_value_rule`) and injects it as current (production `_run_biased_trial`).
  synaptic_need (AFTER): need synapses open (gate 1); host salience is exactly (0, 0).

MEASUREMENTS (plan section 2, kept separate): decoding (winner predicted from the host variable, the interoceptive
counts, the D1 counts), substitution (host scoring replaced by the neural route: fraction of trials whose winner
changes), intervention (hold the neural need fixed and flip every host decision variable, and the converse), plus the
code-path audit (a guarded world-side candidate table records every read, before and after the commit) and the
mechanism ablations (section 6: which mechanisms were NECESSARY for the reversal).

No `sim/` edit, no `webapp/` edit: the selector is imported and reused unchanged; the need route is added through the
builder's own `extra_regions` / `extra_pathways` seam.

Usage (dev seeds only until the preregistration is committed):
  SIM_BACKEND=numpy python -m research.runners._need_reversal_selection_seam --seeds 7 --trials 20 \
      --out research/findings/raw/_need_reversal_seam/dev_seed7.json
  SIM_BACKEND=numpy python -m research.runners._need_reversal_selection_seam --tune --seeds 7 \
      --out research/findings/raw/_need_reversal_seam/dev_tune_seed7.json
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import inspect
import json
import os
import textwrap
import time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import numpy as np

from sim.backend import get_backend, to_host
from sim.enums import NeuronType
from sim.regions import RegionPathway

from research.runners._vocal_action_selector_gate import (
    CHANNELS, DIRECT_PATH_GATE, _indices, _region, _set_equal_tonic_current, _step,
    build_selector_bridge, selector_config,
)
from research.runners.bg_action_selection_production_organ import SALIENCE_GAIN_PA
from tools.lab import attributable_to, membrane_bound_check

NEED_GATE = "need_value"
GATE_SEEDS = (42, 43, 44, 100, 101, 102)
DEV_SEEDS = tuple(range(1, 13))

# ── operating point: DEV-tuned on seeds 1-12 (research/findings/raw/_need_reversal_seam/dev_tune*/), frozen in the
#    preregistration before any gate seed. NMDA (voltage-gated) need synapses: at w=20 / 400 pA the 12 dev seeds gave
#    mean R 0.90 (min 0.85) with the need unable to commit on its own (no-arousal commit mean 0.07); AMPA at matched
#    selectivity let the need alone commit more often (w=60/350: 0.07 mean, 0.2 max; w=70/350: 0.6 max). ─────────
N_INTERO = 24
INTERO_MAX_PA = 400.0        # body afferent current at full deficit (hunger pop) / full surplus (satiety pop)
NEED_DENSITY = 0.5           # intero -> str_d1 connection probability
NEED_WEIGHT = 20.0           # intero -> str_d1 weight (the selector's proposal -> D1 weight is 400)
NEED_RECEPTOR = "nmda_slow"  # Mg2+-blocked slow NMDA: the need acts on MSNs the cortical barrage has depolarized
INACTIVATE_PA = -1000.0      # constant hyperpolarizing current = region inactivation for ablations. DEV-checked
#                              (seeds 1, 8): -1000 pA silences D2, GPe, STN, commit FS and thalamus with V >= -95 mV;
#                              -3000 drove the thalamic relay to -175 mV and into rebound firing (2 spikes/trial).

H_EXTREMES = (0.0, 1.0)
H_CURVE = (0.0, 0.25, 0.5, 0.75, 1.0)

# which channel each interoceptive population projects to under each mapping. The mapping is WIRING (the analogue of
# a learned association), fixed at build; the trial code never reads it.
MAPPINGS = {
    "matched": {"intero_satiety": (0,), "intero_hunger": (1,)},
    "shuffled": {"intero_satiety": (0, 1), "intero_hunger": (0, 1)},   # same expected synapse count, no mapping
}


# ═══ the WORLD side: what the actions are and what they do. Read only AFTER the neural commit. ══════════════════
class WorldCandidates:
    """World-side candidate table. Every read is logged with the phase it happened in, so the audit can show the
    selection path never inspected a candidate's identity before the neural commit."""

    def __init__(self, labels=None):
        self._labels = dict(labels or {0: "rest", 1: "forage"})
        self.phase = "pre_commit"
        self.reads = []

    def label_of(self, channel):
        self.reads.append((self.phase, int(channel)))
        return self._labels[int(channel)]

    def channels(self):
        return tuple(sorted(self._labels))

    def pre_commit_reads(self):
        return sum(1 for ph, _ in self.reads if ph == "pre_commit")


def world_consequence(winner, cands, h):
    """The body/world: foraging when in deficit reduces the deficit (drive reduction); resting does not. Applied
    AFTER the commit; returns the change in deficit. A non-commit does nothing."""
    if winner is None:
        return 0.0
    if cands.label_of(winner) == "forage":
        return -0.5 * float(h)
    return 0.0


def host_value_rule(h, cands):
    """The BEFORE arm's host scoring (the production pattern): the host decides how valuable each candidate is from
    WHAT it is and the need, then hands the race a salience vector. This is the Python multiplier the synaptic arm
    removes. It reads the candidate labels, so the audit counts it."""
    out = []
    for ch in cands.channels():
        out.append(float(h) if cands.label_of(ch) == "forage" else 1.0 - float(h))
    return tuple(out)


# ═══ the substrate ═══════════════════════════════════════════════════════════════════════════════════════════
@dataclasses.dataclass(frozen=True)
class NeedParams:
    n_intero: int = N_INTERO
    intero_max_pA: float = INTERO_MAX_PA
    need_density: float = NEED_DENSITY
    need_weight: float = NEED_WEIGHT
    mapping: str = "matched"
    # "ampa" = fast additive drive; "nmda_slow" = voltage-gated (Mg2+ block): the need input only acts on a D1 MSN
    # the cortical proposal barrage has already depolarized (the striatal up-state), so need BIASES but cannot DRIVE.
    receptor: str = NEED_RECEPTOR


def build_need_bridge(seed, params=NeedParams(), *, config=None, commit_enable_nmda=True):
    """The Gate-A v2 selector, unchanged, plus the two interoceptive relays and their synapses onto striatal D1."""
    config = config if config is not None else selector_config("v2")
    rs = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL
    extra_regions = [
        _region("intero_satiety", params.n_intero, exc_fraction=1.0, neuron_type=rs),
        _region("intero_hunger", params.n_intero, exc_fraction=1.0, neuron_type=rs),
    ]
    targets = MAPPINGS[params.mapping]
    extra_pathways = []
    for src in ("intero_satiety", "intero_hunger"):
        chans = targets[src]
        for ch in chans:
            extra_pathways.append(RegionPathway(
                from_region=src, to_region=f"str_d1_{ch}",
                density=float(params.need_density) / len(chans), weight_mean=float(params.need_weight),
                weight_jitter=0.05, plastic=False, transmission_gate=NEED_GATE, exc_receptor=params.receptor,
            ))
    updates = {"enable_nmda_recurrent": True} if params.receptor == "nmda_slow" else None
    bridge = build_selector_bridge(
        int(seed), config, commit_enable_nmda=commit_enable_nmda,
        extra_regions=extra_regions, extra_pathways=extra_pathways, core_config_updates=updates,
    )
    bridge.set_transmission_gate(DIRECT_PATH_GATE, 1.0)
    bridge.set_transmission_gate(NEED_GATE, 1.0)
    return bridge, config


def _apply_body(bridge, h, params, *, intero_swap=False, intero_silent=False):
    """The body -> sensory boundary: the deficit drives the hunger relay, the surplus drives the satiety relay.
    `intero_swap` crosses the afferents (the neural need state inverted while the body is unchanged)."""
    if intero_silent:
        return
    xp, _ = get_backend()
    hunger_pA = float(params.intero_max_pA) * float(h)
    satiety_pA = float(params.intero_max_pA) * (1.0 - float(h))
    if intero_swap:
        hunger_pA, satiety_pA = satiety_pA, hunger_pA
    bridge.cp_external_input_current[xp.asarray(_indices(bridge, "intero_hunger"))] = xp.float32(hunger_pA)
    bridge.cp_external_input_current[xp.asarray(_indices(bridge, "intero_satiety"))] = xp.float32(satiety_pA)


def _apply_clamps(bridge, clamps):
    xp, _ = get_backend()
    for region, pA in (clamps or {}).items():
        idx = xp.asarray(_indices(bridge, region))
        bridge.cp_external_input_current[idx] = bridge.cp_external_input_current[idx] + xp.float32(pA)


def _baseline_currents(bridge, config, h, params, body_kw, clamps):
    """Everything the host writes that is NOT a trial event: equal tonic, the body, and ablation clamps."""
    _set_equal_tonic_current(bridge, config)
    if h is not None:
        _apply_body(bridge, h, params, **body_kw)
    _apply_clamps(bridge, clamps)


def warm_up(bridge, config, h, params, *, body_kw=None, clamps=None):
    _baseline_currents(bridge, config, h, params, body_kw or {}, clamps)
    _step(bridge, config.warmup_steps)


def run_need_trial(bridge, config, params, h, *, arousal=True, host_salience=(0.0, 0.0), body_kw=None,
                   clamps=None, gain=SALIENCE_GAIN_PA, record_regions=()):
    """ONE race. Mirrors the production `_run_biased_trial` step for step (same commit read, reset and washout);
    adds only the body afferent, optional ablation clamps and the spike tallies for the decoding screen. It takes
    no candidate identity: its only candidate-specific input is `host_salience`, which the synaptic arm fixes at
    (0, 0). Returns the neural winner (motor-pool index) or None."""
    xp, _ = get_backend()
    body_kw = body_kw or {}
    _baseline_currents(bridge, config, h, params, body_kw, clamps)
    motor_idx = {ch: _indices(bridge, f"motor_{ch}") for ch in CHANNELS}
    if arousal:
        bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, "practice_arousal"))
        ] = xp.float32(config.practice_pA)
        sal = (float(host_salience[0]), float(host_salience[1]))
        if sal != (0.0, 0.0):
            for ch in CHANNELS:
                idx = xp.asarray(_indices(bridge, f"str_d1_{ch}"))
                bridge.cp_external_input_current[idx] = (
                    bridge.cp_external_input_current[idx] + xp.float32(sal[ch] * gain)
                )
    rec_idx = {name: _indices(bridge, name) for name in record_regions}
    rec = {name: 0 for name in record_regions}
    v_min, v_max = np.inf, -np.inf

    counts = np.zeros(2, dtype=np.int64)
    first_crossing = None
    simultaneous = False
    decision_step = None
    for step in range(int(config.action_steps)):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        for name, ii in rec_idx.items():
            rec[name] += int(firing[ii].sum())
        v = np.asarray(to_host(bridge.cp_membrane_potential_v), dtype=float)       # every step
        v_min, v_max = min(v_min, float(v.min())), max(v_max, float(v.max()))
        previous = counts.copy()
        for ch in CHANNELS:
            counts[ch] += int(firing[motor_idx[ch]].sum())
        crossed = [ch for ch in CHANNELS if previous[ch] < config.commit_threshold_spikes <= counts[ch]]
        if first_crossing is None and len(crossed) == 1:
            first_crossing = int(crossed[0])
            decision_step = int(step)
            break
        elif first_crossing is None and len(crossed) > 1:
            simultaneous = True
            decision_step = int(step)
            break
    v = np.asarray(to_host(bridge.cp_membrane_potential_v), dtype=float)
    v_min, v_max = min(v_min, float(v.min())), max(v_max, float(v.max()))

    winner = None
    loser_ratio = None
    if first_crossing is not None and not simultaneous:
        loser = 1 - first_crossing
        loser_ratio = float(counts[loser] / max(1, counts[first_crossing]))
        if loser_ratio <= config.clean_loser_ratio:
            winner = int(first_crossing)

    # between-trial reset + washout (verbatim production behavior; the body and the clamps stay on).
    _baseline_currents(bridge, config, h, params, body_kw, clamps)
    bridge.cp_external_input_current[
        xp.asarray(_indices(bridge, "selector_reset"))
    ] = xp.float32(config.reset_pA)
    for _ in range(int(config.reset_steps)):
        _step(bridge)
        v = np.asarray(to_host(bridge.cp_membrane_potential_v), dtype=float)
        v_min, v_max = min(v_min, float(v.min())), max(v_max, float(v.max()))
    _baseline_currents(bridge, config, h, params, body_kw, clamps)
    for _ in range(int(config.washout_steps)):
        _step(bridge)
        v = np.asarray(to_host(bridge.cp_membrane_potential_v), dtype=float)
        v_min, v_max = min(v_min, float(v.min())), max(v_max, float(v.max()))
    return {
        "winner": winner,
        "committed": winner is not None,
        "first_crossing": first_crossing,
        "simultaneous": bool(simultaneous),
        "motor_spikes": counts.tolist(),
        "loser_ratio": loser_ratio,
        "decision_step": decision_step,
        "spikes": rec,
        "v_min": v_min,
        "v_max": v_max,
    }


# ═══ one block of trials on a FRESH bridge (so every comparison starts from the identical seeded state) ═══════
RECORD = ("intero_hunger", "intero_satiety", "str_d1_0", "str_d1_1",
          # the ablation targets, recorded in EVERY block so the necessity table can tell "dispensable while active"
          # from "silent in the intact race anyway"
          "str_d2_0", "str_d2_1", "gpe_0", "gpe_1", "selector_stn", "gpi_0", "gpi_1", "thal_0", "thal_1",
          "commit_0", "commit_1", "commit_fs_0", "commit_fs_1", "motor_0", "motor_1")

ABLATIONS = {
    # name: (build kwargs, run kwargs)
    "need_synapse": ({}, {"need_gate": 0.0}),
    "intero_silent": ({}, {"body_kw": {"intero_silent": True}}),
    "arousal": ({}, {"arousal": False}),
    "direct_path": ({}, {"direct_gate": 0.0}),
    "d2_indirect": ({}, {"inactivate": ("str_d2_0", "str_d2_1")}),
    "gpe": ({}, {"inactivate": ("gpe_0", "gpe_1")}),
    "stn": ({}, {"inactivate": ("selector_stn",)}),
    "commit_cross_inhibition": ({}, {"inactivate": ("commit_fs_0", "commit_fs_1")}),
    "thalamus": ({}, {"inactivate": ("thal_0", "thal_1")}),
    "commit_nmda": ({"commit_enable_nmda": False}, {}),
    "ou_noise": ({"ou_sigma_pA": 0.0}, {}),
}


def run_block(seed, h, trials, *, params=NeedParams(), arm="synaptic_need", cands=None, host_belief_h=None,
              need_gate=None, direct_gate=1.0, arousal=True, body_kw=None, inactivate=(), commit_enable_nmda=True,
              ou_sigma_pA=None, host_override=None, strict_membrane=True):
    """`trials` races at body deficit `h` on a fresh bridge. `arm` selects who scores the candidates:
      synaptic_need: need gate open, host salience (0, 0);
      host_salience: need gate closed, host salience = host_value_rule(host_belief_h, cands).
    `host_override` (synaptic arm only) injects a host salience vector on top, to measure the host override effect.
    Returns per-trial rows + a digest of the full outcome sequence."""
    config = selector_config("v2")
    if ou_sigma_pA is not None:
        config = dataclasses.replace(config, ou_sigma_pA=float(ou_sigma_pA))
    cands = cands if cands is not None else WorldCandidates()
    bridge, config = build_need_bridge(seed, params, config=config, commit_enable_nmda=commit_enable_nmda)
    if need_gate is None:
        need_gate = 1.0 if arm == "synaptic_need" else 0.0
    bridge.set_transmission_gate(NEED_GATE, float(need_gate))
    bridge.set_transmission_gate(DIRECT_PATH_GATE, float(direct_gate))
    clamps = {r: INACTIVATE_PA for r in inactivate}
    body_kw = body_kw or {}
    warm_up(bridge, config, h, params, body_kw=body_kw, clamps=clamps)
    belief = h if host_belief_h is None else host_belief_h
    rows = []
    membrane_violation = None
    record = RECORD + tuple(r for r in inactivate if r not in RECORD)
    for _ in range(int(trials)):
        cands.phase = "pre_commit"
        if arm == "host_salience":
            sal = host_value_rule(belief, cands)
        elif arm == "synaptic_need":
            sal = (0.0, 0.0) if host_override is None else tuple(float(s) for s in host_override)
        else:
            raise ValueError(arm)
        r = run_need_trial(bridge, config, params, h, arousal=arousal, host_salience=sal, body_kw=body_kw,
                           clamps=clamps, record_regions=record)
        cands.phase = "post_commit"
        r["label"] = cands.label_of(r["winner"]) if r["winner"] is not None else None
        r["deficit_change"] = world_consequence(r["winner"], cands, h)
        r["host_salience"] = list(sal)
        rows.append(r)
        if strict_membrane:
            try:
                membrane_bound_check("need-seam trial (seed %d, h %.2f, arm %s)" % (seed, h, arm),
                                     np.array([r["v_min"], r["v_max"]]))
            except AssertionError as e:          # LeverError: this block is UNDEFINED, the run continues
                membrane_violation = str(e)[:300]
                break
    # TERMS.md "lesion": verify the manipulation still holds at the moment of measurement (read back, not assumed)
    gates_at_end = {g: _gate_value(bridge, g) for g in (NEED_GATE, DIRECT_PATH_GATE)}
    digest = hashlib.sha256(json.dumps(
        [[r["winner"], r["decision_step"], r["motor_spikes"], r["spikes"]] for r in rows], sort_keys=True,
    ).encode()).hexdigest()[:16]
    return {"rows": rows, "digest": digest, "pre_commit_reads": cands.pre_commit_reads(),
            "n_reads": len(cands.reads), "gates_at_end": gates_at_end, "membrane_violation": membrane_violation}


def _gate_value(bridge, name):
    """Mean runtime transmission gain of the synapses tagged `name`, read back from the bridge."""
    name = bridge._canonicalize_gate_name(name)
    idx = bridge._transmission_gate_indices_gpu.get(name)
    if idx is None or int(idx.size) == 0:
        return None
    return round(float(np.mean(np.asarray(to_host(bridge.cp_transmission_gain[idx]), dtype=float))), 4)


def summarize(block, forage_channel=1):
    rows = block["rows"]
    n = len(rows)
    committed = [r for r in rows if r["committed"]]
    n_c = len(committed)
    p_b = (sum(1 for r in committed if r["winner"] == forage_channel) / n_c) if n_c else None
    return {
        "n": n, "n_committed": n_c, "commit_rate": round(n_c / max(1, n), 4),
        "p_forage": (round(p_b, 4) if p_b is not None else None),
        "digest": block["digest"], "pre_commit_reads": block["pre_commit_reads"],
        "gates_at_end": block.get("gates_at_end"),
        "membrane_violation": block.get("membrane_violation"),
        "mean_spikes": {k: round(float(np.mean([r["spikes"].get(k, 0) for r in rows])), 1) for k in rows[0]["spikes"]}
        if rows else {},
        "v_range": [round(min(r["v_min"] for r in rows), 1), round(max(r["v_max"] for r in rows), 1)] if rows else None,
        # compact per-trial record for the decoding screen: [winner, host_sal_0, host_sal_1, hunger, satiety, d1_0, d1_1]
        "trials": [[r["winner"], r["host_salience"][0], r["host_salience"][1], r["spikes"].get("intero_hunger", 0),
                    r["spikes"].get("intero_satiety", 0), r["spikes"].get("str_d1_0", 0), r["spikes"].get("str_d1_1", 0)]
                   for r in rows],
    }


def reversal_index(s_hungry, s_sated, min_commit=0.5):
    """R = P(forage | committed, h=1) - P(forage | committed, h=0). UNDEFINED (None) if either extreme commits on
    fewer than `min_commit` of trials: a reversal cannot be read off a race that does not choose."""
    if s_hungry["commit_rate"] < min_commit or s_sated["commit_rate"] < min_commit:
        return None
    return round(float(s_hungry["p_forage"]) - float(s_sated["p_forage"]), 4)


def host_share(p_host_opt_intervened, p_host_opt_baseline):
    """Plan section 2 intervention measure. H = the option the intervention favors (opposite of the unintervened
    preference). HS = (P(H | intervened) - P(H | baseline)) / (1 - P(H | baseline)); 1 = the host fully controls the
    winner, 0 = no effect. None if undefined."""
    if p_host_opt_intervened is None or p_host_opt_baseline is None or p_host_opt_baseline >= 1.0:
        return None
    return round((p_host_opt_intervened - p_host_opt_baseline) / (1.0 - p_host_opt_baseline), 4)


# ═══ the code-path audit (static) ════════════════════════════════════════════════════════════════════════════
FORBIDDEN_IN_TRIAL = ("WorldCandidates", "label_of", "host_value_rule", "world_consequence", "argmax", "argmin",
                      "sorted", "labels", "cands")


def static_audit():
    """The trial function must take no candidate identity and reference none of the world-side names."""
    src = textwrap.dedent(inspect.getsource(run_need_trial))
    tree = ast.parse(src)
    fn = tree.body[0]
    params = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    hits = sorted(set(FORBIDDEN_IN_TRIAL) & (names | set(params)))
    return {"trial_params": params, "forbidden_hits": hits, "clean": not hits}


# ═══ the full per-seed battery ═══════════════════════════════════════════════════════════════════════════════
def _p_opt(summary, opt):
    if summary["p_forage"] is None:
        return None
    return summary["p_forage"] if opt == 1 else round(1.0 - summary["p_forage"], 4)


def run_seed(seed, *, trials=60, curve_trials=20, params=NeedParams(), ablations=True, curve=True):
    t0 = time.time()
    out = {"seed": int(seed), "params": dataclasses.asdict(params), "trials": int(trials)}
    S = {}

    def blk(key, h, **kw):
        b = run_block(seed, h, kw.pop("n", trials), params=kw.pop("params", params), **kw)
        S[key] = summarize(b)
        return b

    # AFTER: the synaptic-need arm, intact, at the two extremes
    for h in H_EXTREMES:
        blk(f"after/intact/h{h}", h, arm="synaptic_need")
    # BEFORE: the host-salience arm (production pattern)
    for h in H_EXTREMES:
        blk(f"before/intact/h{h}", h, arm="host_salience")
    # controls on the AFTER arm
    for h in H_EXTREMES:
        blk(f"after/need_lesion/h{h}", h, arm="synaptic_need", need_gate=0.0)
        blk(f"after/shuffled_synapses/h{h}", h, arm="synaptic_need",
            params=dataclasses.replace(params, mapping="shuffled"))
        blk(f"after/swapped_interoception/h{h}", h, arm="synaptic_need", body_kw={"intero_swap": True})
        blk(f"after/arousal_lesion/h{h}", h, arm="synaptic_need", arousal=False)
        blk(f"after/direct_path_lesion/h{h}", h, arm="synaptic_need", direct_gate=0.0)
    # intervention on the HOST decision variables (neural need fixed at h): swapped labels + inverted host belief
    for h in H_EXTREMES:
        swapped = WorldCandidates({0: "forage", 1: "rest"})
        blk(f"after/host_intervened/h{h}", h, arm="synaptic_need", cands=swapped, host_belief_h=1.0 - h)
        blk(f"before/host_intervened/h{h}", h, arm="host_salience", host_belief_h=1.0 - h)
        # the converse: intervene on the NEURAL need (interoception swapped), host belief left at the body truth
        blk(f"before/neural_intervened/h{h}", h, arm="host_salience", body_kw={"intero_swap": True})
        # host override effect: the host pushes the non-preferred option at the production gain
        push = (1.0, 0.0) if h == 1.0 else (0.0, 1.0)
        blk(f"after/host_override/h{h}", h, arm="synaptic_need", host_override=push)
    if curve:
        for h in H_CURVE:
            if h in H_EXTREMES:
                continue
            blk(f"after/intact/h{h}", h, arm="synaptic_need", n=curve_trials)
            blk(f"before/intact/h{h}", h, arm="host_salience", n=curve_trials)
    abl = {}
    if ablations:
        for name, (build_kw, run_kw) in ABLATIONS.items():
            kw = dict(run_kw)
            if "commit_enable_nmda" in build_kw:
                kw["commit_enable_nmda"] = build_kw["commit_enable_nmda"]
            if "ou_sigma_pA" in build_kw:
                kw["ou_sigma_pA"] = build_kw["ou_sigma_pA"]
            for h in H_EXTREMES:
                blk(f"ablation/{name}/h{h}", h, arm="synaptic_need", **kw)
            s1, s0 = S[f"ablation/{name}/h1.0"], S[f"ablation/{name}/h0.0"]
            inact = run_kw.get("inactivate", ())
            abl[name] = {
                "commit_h1": s1["commit_rate"], "commit_h0": s0["commit_rate"],
                "reversal": reversal_index(s1, s0),
                "inactivated_region_spikes": {r: [s1["mean_spikes"].get(r), s0["mean_spikes"].get(r)] for r in inact},
            }
    out["blocks"] = S
    out["ablations"] = abl

    # ── derived measures ──
    R = {}
    for key in ("after/intact", "before/intact", "after/need_lesion", "after/shuffled_synapses",
                "after/swapped_interoception", "after/host_intervened", "before/host_intervened",
                "before/neural_intervened", "after/host_override"):
        R[key] = reversal_index(S[f"{key}/h1.0"], S[f"{key}/h0.0"])
    out["reversal"] = R
    # host share (intervention on host variables, neural need fixed). Baseline preference at h: forage when hungry.
    hs = {}
    for arm in ("before", "after"):
        vals = []
        for h in H_EXTREMES:
            neural_pref = 1 if h == 1.0 else 0
            host_opt = 1 - neural_pref
            base = _p_opt(S[f"{arm}/intact/h{h}"], host_opt)
            intv = _p_opt(S[f"{arm}/host_intervened/h{h}"], host_opt)
            vals.append(host_share(intv, base))
        hs[arm] = vals
    # neural share: intervene on the neural need (interoception swapped), host variables at the body truth
    ns = {}
    for arm, key in (("before", "before/neural_intervened"), ("after", "after/swapped_interoception")):
        vals = []
        for h in H_EXTREMES:
            body_pref = 1 if h == 1.0 else 0
            neural_opt = 1 - body_pref       # the option the swapped interoception favors
            base = _p_opt(S[f"{arm}/intact/h{h}"], neural_opt)
            intv = _p_opt(S[f"{key}/h{h}"], neural_opt)
            vals.append(host_share(intv, base))
        ns[arm] = vals
    out["host_share"] = hs
    out["neural_share"] = ns
    out["after_host_intervention_byte_identical"] = all(
        S[f"after/host_intervened/h{h}"]["digest"] == S[f"after/intact/h{h}"]["digest"] for h in H_EXTREMES)
    # substitution: host scoring -> neural route, same seed and body: fraction of committed winners that differ is
    # read from the two arms' winner probabilities (different noise realizations, so it is a distributional read)
    out["substitution_delta_p_forage"] = {
        f"h{h}": (None if S[f"after/intact/h{h}"]["p_forage"] is None or S[f"before/intact/h{h}"]["p_forage"] is None
                  else round(S[f"after/intact/h{h}"]["p_forage"] - S[f"before/intact/h{h}"]["p_forage"], 4))
        for h in H_EXTREMES}
    out["decoding"] = {arm: decoding_screen(S, arm) for arm in ("before", "after")}
    # whose is the reversal / the commit? the fraction of the intact effect NOT present in each control
    R_int = R["after/intact"]
    c_int = min(S["after/intact/h1.0"]["commit_rate"], S["after/intact/h0.0"]["commit_rate"])

    def _att(label, treat, ctrl):
        if treat is None or ctrl is None:
            return None
        a = attributable_to(label, treat, ctrl)
        return round(a, 4) if a is not None else None

    out["attribution"] = {
        "reversal_to_need_synapses": _att("need synapses @ reversal", R_int, R["after/need_lesion"]),
        "reversal_to_synapse_mapping": _att("need->D1 mapping @ reversal", R_int, R["after/shuffled_synapses"]),
        "commit_to_arousal": _att("shared arousal @ commit", c_int, min(
            S["after/arousal_lesion/h1.0"]["commit_rate"], S["after/arousal_lesion/h0.0"]["commit_rate"])),
        "commit_to_direct_path": _att("direct path @ commit", c_int, min(
            S["after/direct_path_lesion/h1.0"]["commit_rate"], S["after/direct_path_lesion/h0.0"]["commit_rate"])),
    }
    out["touch_audit"] = {k: v["pre_commit_reads"] for k, v in S.items() if k.endswith("h1.0") or k.endswith("h0.0")}
    out["static_audit"] = static_audit()
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


# ═══ decoding screen (per trial): which variable predicts the winner? ════════════════════════════════════════
def decoding_screen(blocks, arm):
    """Across the intact blocks of one arm (both extremes + the curve), predict each committed winner by a sign rule
    from (a) the host decision variable (which candidate the host salience favors; a tie -> half credit, so the
    synaptic arm's constant (0, 0) decodes at chance by construction), (b) the interoceptive differential
    (hunger - satiety spikes), (c) the striatal D1 differential. A SCREEN (plan section 2), never evidence of who
    decided: in the host arm the neural counts also predict the winner, because the host current drives them."""
    rows = []
    for key, s in blocks.items():
        if key.startswith(f"{arm}/intact/"):
            rows.extend(t for t in s["trials"] if t[0] is not None)
    if not rows:
        return None

    def acc(pred):
        return round(float(np.mean([pred(t) for t in rows])), 4)

    def sign_pred(d, winner):
        return 0.5 if d == 0 else float((1 if d > 0 else 0) == winner)

    return {
        "n_committed": len(rows),
        "host_variable": acc(lambda t: sign_pred(t[2] - t[1], t[0])),
        "interoceptive_differential": acc(lambda t: sign_pred(t[3] - t[4], t[0])),
        "d1_differential": acc(lambda t: sign_pred(t[6] - t[5], t[0])),
    }


# ═══ closed loop: the body's deficit evolves with the brain's own choices (reported, not gated) ════════════════
def closed_loop_episode(seed, n_trials=150, *, params=NeedParams(), need_gate=1.0, h0=0.5, deplete=0.08,
                        eat=0.35):
    """ONE persistent bridge; each trial the deficit rises by `deplete`; a committed FORAGE (a world-side read after
    the commit) lowers it by `eat`. No host rule chooses: the host only renders the body and applies consequences.
    Intact, the need should gate foraging (forage when the deficit is high, rest when it is low) and hold the deficit
    in a band; with the need synapses closed the choice is blind to the deficit."""
    cands = WorldCandidates()
    bridge, config = build_need_bridge(seed, params)
    bridge.set_transmission_gate(NEED_GATE, float(need_gate))
    h = float(h0)
    warm_up(bridge, config, h, params)
    trace = []
    for _ in range(int(n_trials)):
        h = min(1.0, h + deplete)
        cands.phase = "pre_commit"
        r = run_need_trial(bridge, config, params, h)
        cands.phase = "post_commit"
        forage = r["winner"] is not None and cands.label_of(r["winner"]) == "forage"
        trace.append((round(h, 4), r["winner"], bool(forage)))
        if forage:
            h = max(0.0, h - eat)
    hs = np.array([t[0] for t in trace])
    fz = np.array([1.0 if t[2] else 0.0 for t in trace])
    corr = float(np.corrcoef(hs, fz)[0, 1]) if fz.std() > 0 and hs.std() > 0 else None
    return {"need_gate": need_gate, "n_trials": int(n_trials), "mean_deficit": round(float(hs.mean()), 4),
            "sd_deficit": round(float(hs.std()), 4), "forage_rate": round(float(fz.mean()), 4),
            "corr_deficit_forage": (round(corr, 4) if corr is not None else None),
            "p_forage_when_deficit_high": (round(float(fz[hs >= 0.6].mean()), 4) if (hs >= 0.6).any() else None),
            "p_forage_when_deficit_low": (round(float(fz[hs <= 0.4].mean()), 4) if (hs <= 0.4).any() else None),
            "pre_commit_reads": cands.pre_commit_reads(), "trace": trace}


# ═══ production anchor: the host share of the LIVE organ object (reported, not gated) ═══════════════════════════
def production_baseline(seed, trials=60, messages=("...", "?", "what does the dog chase?", "tell me about cats")):
    """The same intervention measure on `bg_action_selection_production_organ` itself, so the BEFORE number is
    anchored to the production object and not only to this runner's re-implementation of its race.
      * pre-filter share: the fraction of `messages` the host decides WITHOUT consulting the race
        (`decide_action` returns before the race unless the host salience already favors STAY-SILENT);
      * race host share: on a fresh organ, the host salience (speak, silent) = (0, 1) vs the flipped (1, 0),
        same seed; HS as in `host_share` with H = SPEAK (the option the flip favors)."""
    from research.runners import bg_action_selection_production_organ as BG
    prefilter = []
    for m in messages:
        sp, si = BG.salience(m)
        prefilter.append({"msg": m, "speak_salience": sp, "silent_salience": si,
                          "race_consulted": bool(si > sp)})

    def block(sal):
        org = BG.BGActionSelector(seed=seed)
        rows = [org.select_once(*sal) for _ in range(int(trials))]
        c = [r for r in rows if r["committed"]]
        p_speak = (sum(1 for r in c if r["winner"] == BG.SPEAK) / len(c)) if c else None
        return {"commit_rate": round(len(c) / max(1, len(rows)), 4),
                "p_speak": (round(p_speak, 4) if p_speak is not None else None)}

    base, flipped = block((0.0, 1.0)), block((1.0, 0.0))
    return {"prefilter": prefilter,
            "prefilter_host_decided_fraction": round(sum(1 for x in prefilter if not x["race_consulted"])
                                                     / len(prefilter), 4),
            "race_baseline": base, "race_host_flipped": flipped,
            "race_host_share": host_share(flipped["p_speak"], base["p_speak"])}


# ═══ the preregistered verdict (bars frozen in the PREREGISTRATION; changing them is an amendment) ═════════════
BARS = {
    "commit_min": 0.90,               # G1: the intact synaptic arm commits at both need extremes
    "R_min": 0.50,                    # G2: winner reversal R = P(forage|h=1) - P(forage|h=0)
    "p_h1_min": 0.70,                 # G2: forage is the majority winner when hungry ...
    "p_h0_max": 0.30,                 # G2: ... and the minority winner when sated
    "control_commit_min": 0.50,       # G3-G5 are evaluable only if the control arm still chooses
    "control_R_abs_max": 0.25,        # G3 need-synapse lesion, G4 shuffled synapses: the reversal vanishes
    "swap_R_max": -0.50,              # G5 swapped interoception: the winner follows the neural need state
    "hs_before_min": 0.80,            # G7 instrument positive control: the host owns the host-salience arm
    "ns_after_min": 0.50,             # G7 the neural need owns the synaptic arm
    "ns_before_max": 0.20,            # G7 the neural need has no causal route in the host-salience arm
    "inactivation_max_spikes": 1.0,   # necessity ablations: a silenced region fires <= 1 spike per trial (mean)
    "capability_commit_min": 0.80,    # necessity rule: capability present iff commit >= this at both extremes ...
    "capability_R_min": 0.50,         # ... and R >= this
}


def seed_verdict(r, bars=BARS):
    """Per-seed gate table G1-G7 + the preconditions, through tools.verdict (UNDEFINED unless earned)."""
    from tools.verdict import Verdict
    S = r["blocks"]
    g = {}
    v = Verdict("need-reversal selection seam, seed %d" % r["seed"])
    # preconditions: the manipulations hold at measurement (TERMS.md "lesion"), the static audit ran
    v.knob("need gate open, intact", requested=1.0, applied=S["after/intact/h1.0"]["gates_at_end"]["need_value"])
    v.knob("need gate closed, host arm", requested=0.0,
           applied=S["before/intact/h1.0"]["gates_at_end"]["need_value"])
    v.knob("need gate closed, lesion", requested=0.0,
           applied=S["after/need_lesion/h1.0"]["gates_at_end"]["need_value"])
    v.knob("direct path closed, lesion", requested=0.0,
           applied=S["after/direct_path_lesion/h1.0"]["gates_at_end"]["vocal_selector_direct_path"])
    v.require("static audit ran and is clean", r["static_audit"]["clean"])
    v.require("no membrane-bound violation in any gated block",
              all(not b.get("membrane_violation") for k, b in S.items() if not k.startswith("ablation/")))
    s1, s0 = S["after/intact/h1.0"], S["after/intact/h0.0"]
    v.require("intact synaptic arm chooses (R defined)", r["reversal"]["after/intact"] is not None)

    def evaluable(key):
        return (S[f"{key}/h1.0"]["commit_rate"] >= bars["control_commit_min"]
                and S[f"{key}/h0.0"]["commit_rate"] >= bars["control_commit_min"])

    R = r["reversal"]
    g["G1_commit"] = min(s1["commit_rate"], s0["commit_rate"]) >= bars["commit_min"]
    g["G2_reversal"] = (R["after/intact"] is not None and R["after/intact"] >= bars["R_min"]
                        and s1["p_forage"] >= bars["p_h1_min"] and s0["p_forage"] <= bars["p_h0_max"])
    for gate, key in (("G3_need_lesion", "after/need_lesion"), ("G4_shuffled_synapses", "after/shuffled_synapses")):
        g[gate] = (abs(R[key]) <= bars["control_R_abs_max"]) if evaluable(key) and R[key] is not None else None
    g["G5_swapped_interoception"] = ((R["after/swapped_interoception"] <= bars["swap_R_max"])
                                     if evaluable("after/swapped_interoception") else None)
    after_keys = [k for k in S if k.startswith("after/") and (k.endswith("h1.0") or k.endswith("h0.0"))]
    before_keys = [k for k in S if k.startswith("before/") and (k.endswith("h1.0") or k.endswith("h0.0"))]
    g["G6_code_path"] = (all(S[k]["pre_commit_reads"] == 0 for k in after_keys)
                         and r["static_audit"]["clean"]
                         and r["after_host_intervention_byte_identical"]
                         and all(S[k]["pre_commit_reads"] > 0 for k in before_keys))
    hs, ns = r["host_share"], r["neural_share"]
    if any(x is None for x in hs["before"] + hs["after"] + ns["before"] + ns["after"]):
        g["G7_host_share"] = None
    else:
        g["G7_host_share"] = (min(hs["before"]) >= bars["hs_before_min"] and max(abs(x) for x in hs["after"]) == 0.0
                              and min(ns["after"]) >= bars["ns_after_min"]
                              and max(ns["before"]) <= bars["ns_before_max"])
    # a gate that could not be evaluated (its arm did not choose) is an unmeasured precondition -> UNDEFINED;
    # an evaluated gate that fails is a NO-GO, never an UNDEFINED.
    for name, ok in g.items():
        if ok is None:
            v.require(name + " evaluable", None)
    decided = v.decide(go=all(ok is True for ok in g.values()), verbose=False)
    return {"seed": r["seed"], "gates": g, "decision": decided["status"], "verdict": v.to_dict()}


def necessity_table(per_seed, bars=BARS):
    """Section 6: which mechanisms were NECESSARY for the reversal. Per ablation per seed: capability present iff
    commit >= capability_commit_min at both extremes and R >= capability_R_min; UNDEFINED if an inactivation did not
    silence its region. NECESSARY iff absent on >= 5/6 evaluable seeds; NOT NECESSARY iff present on >= 5/6;
    otherwise CONTRIBUTING. NOT ENGAGED overrides when every inactivated region already fires <= 1 spike per trial
    in the intact race at both extremes on every seed (silencing a silent region is not a test)."""
    names = list(per_seed[0]["ablations"])
    out = {}
    for name in names:
        rows = []
        for r in per_seed:
            a = r["ablations"][name]
            silent = all(x is None or x <= bars["inactivation_max_spikes"]
                         for pair in a["inactivated_region_spikes"].values() for x in pair)
            present = (min(a["commit_h1"], a["commit_h0"]) >= bars["capability_commit_min"]
                       and a["reversal"] is not None and a["reversal"] >= bars["capability_R_min"])
            rows.append({"seed": r["seed"], "commit": [a["commit_h1"], a["commit_h0"]], "reversal": a["reversal"],
                         "inactivation_held": silent, "capability_present": (present if silent else None)})
        ev = [x for x in rows if x["capability_present"] is not None]
        n_abs = sum(1 for x in ev if not x["capability_present"])
        n_pres = sum(1 for x in ev if x["capability_present"])
        need = max(1, len(per_seed) - 1)
        # split (audit MEDIUM 7): commit floors -> the reversal cannot be read -> NECESSARY-FOR-COMMIT; commit kept but
        # the reversal collapses -> NECESSARY-FOR-REVERSAL.
        n_commit_floor = sum(1 for x in rows if min(x["commit"]) < 0.5)
        n_rev_collapse = sum(1 for x in rows if min(x["commit"]) >= 0.5 and (x["reversal"] is None
                                                                             or x["reversal"] < bars["capability_R_min"]))
        label = ("UNDEFINED" if len(ev) < need and n_commit_floor < need else
                 "NECESSARY-FOR-COMMIT" if n_commit_floor >= need else
                 "NECESSARY-FOR-REVERSAL" if n_rev_collapse >= need else
                 "NOT NECESSARY" if n_pres >= need else "CONTRIBUTING")
        if name in ("commit_cross_inhibition", "commit_nmda") and label == "NOT NECESSARY":
            label = "PRE-EMPTED BY HOST READOUT"     # the host first-crossing read stops the trial before they act
        targets = ABLATIONS[name][1].get("inactivate", ())
        intact_rates = {t: [r["blocks"][f"after/intact/h{h}"]["mean_spikes"].get(t) for r in per_seed
                            for h in H_EXTREMES] for t in targets}
        # an inactivation of a region that is ALREADY silent in the intact race tests nothing: label it so, instead
        # of calling the region "not necessary"
        if targets and all(x is not None and x <= bars["inactivation_max_spikes"]
                           for vals in intact_rates.values() for x in vals):
            label = "NOT ENGAGED"
        elif name == "d2_indirect" and all(x is not None and x <= bars["inactivation_max_spikes"] for r in per_seed
                                           for h in H_EXTREMES for x in [r["blocks"][f"after/intact/h{h}"][
                                               "mean_spikes"].get(g) for g in ("gpe_0", "gpe_1")]):
            label = "NOT ENGAGED (its only target, the GPe, is silent)"
        out[name] = {"label": label, "n_absent": n_abs, "n_present": n_pres, "n_evaluable": len(ev),
                     "target_spikes_per_trial_intact": intact_rates, "rows": rows}
    return out


def overall_verdict(seed_verdicts):
    """GO iff every seed is GO. PARTIAL iff G6 and G7 hold on every seed and G1-G5 on >= 4 seeds. UNDEFINED iff
    >= 3 seeds are UNDEFINED. NO-GO otherwise."""
    n = len(seed_verdicts)
    n_go = sum(1 for s in seed_verdicts if s["decision"] == "GO")
    n_undef = sum(1 for s in seed_verdicts if s["decision"] == "UNDEFINED")
    if n_undef >= 3:
        return "UNDEFINED", n_go
    if n_go == n:
        return "GO", n_go
    own = all(s["gates"]["G6_code_path"] is True and s["gates"]["G7_host_share"] is True for s in seed_verdicts)
    cap = sum(1 for s in seed_verdicts if all(s["gates"][k] is True for k in
                                              ("G1_commit", "G2_reversal", "G3_need_lesion",
                                               "G4_shuffled_synapses", "G5_swapped_interoception")))
    if own and cap >= 4:
        return "PARTIAL", n_go
    return "NO-GO", n_go


def overall_block(seed_verdicts, seeds):
    """The top-level read-out. The preregistered verdict exists only over exactly the six gate seeds; it is emitted
    with its own `preconditions` (tools.verdict) so gates/verdict_preconditions can see what earned it. Any other
    seed set gets a `dev_readout` string and NO top-level `verdict` key (a DEV read asserts nothing)."""
    label, n_go = overall_verdict(seed_verdicts)
    complete = sorted(seeds) == sorted(GATE_SEEDS)
    if not complete:
        return {"dev_readout": "DEV (not a verdict): " + label, "n_go": n_go, "gate_seed_set_complete": False}
    from tools.verdict import Verdict
    vo = Verdict("need-reversal selection seam, preregistered six-seed verdict")
    n_undef = sum(1 for s in seed_verdicts if s["decision"] == "UNDEFINED")
    vo.require("the six gate seeds were all run", complete)
    vo.require("fewer than three seeds UNDEFINED", n_undef < 3)
    decided = vo.decide(go=(label == "GO"), verbose=False)
    # the preregistered label (GO / PARTIAL / NO-GO / UNDEFINED) is the verdict; the Verdict status agrees with it
    # on GO and UNDEFINED, and reads NO-GO for both PARTIAL and NO-GO (a PARTIAL is not a GO).
    return {"verdict": label, "n_go": n_go, "gate_seed_set_complete": True,
            "status": decided["status"], "preconditions": decided["preconditions"]}


def aggregate(paths):
    """Combine per-seed outputs (one `--seeds <s>` run each) into the preregistered verdict: seed verdicts are
    recomputed from each seed's stored battery by the same `seed_verdict`, then `necessity_table` and
    `overall_verdict` run over all of them. Seeds must be distinct."""
    per_seed, extras = [], {"production_baseline": {}, "closed_loop": {}}
    meta = []
    for pth in paths:
        d = json.loads(Path(pth).read_text())
        meta.append({"path": str(pth), "backend": d.get("backend"), "label": d.get("label"), "seeds": d.get("seeds")})
        per_seed.extend(d.get("per_seed", []))
        for k in extras:
            extras[k].update(d.get(k, {}) or {})
    seeds = [r["seed"] for r in per_seed]
    if len(set(seeds)) != len(seeds):
        raise SystemExit("aggregate: duplicate seeds %s" % seeds)
    out = {"probe": "need_reversal_selection_seam", "aggregated_from": meta, "seeds": seeds, "bars": BARS}
    out["seed_verdicts"] = [seed_verdict(r) for r in per_seed]
    if all(r.get("ablations") for r in per_seed):
        out["necessity"] = necessity_table(per_seed)
    out.update(overall_block(out["seed_verdicts"], seeds))
    out["summary"] = {r["seed"]: {"reversal": r["reversal"], "host_share": r["host_share"],
                                  "neural_share": r["neural_share"], "decoding": r.get("decoding"),
                                  "substitution_delta_p_forage": r.get("substitution_delta_p_forage"),
                                  "attribution": r.get("attribution"),
                                  "after_host_intervention_byte_identical": r["after_host_intervention_byte_identical"]}
                      for r in per_seed}
    out.update({k: v for k, v in extras.items() if v})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--curve-trials", type=int, default=20)
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--no-curve", action="store_true")
    ap.add_argument("--need-weight", type=float, default=NEED_WEIGHT)
    ap.add_argument("--need-density", type=float, default=NEED_DENSITY)
    ap.add_argument("--intero-max-pa", type=float, default=INTERO_MAX_PA)
    ap.add_argument("--receptor", choices=("ampa", "nmda_slow"), default=NEED_RECEPTOR)
    ap.add_argument("--seed-range", type=str, default=None, help="e.g. 1-12 (dev) instead of --seeds")
    ap.add_argument("--production-baseline", action="store_true",
                    help="also measure the host share of the live production organ (reported, not gated)")
    ap.add_argument("--closed-loop", action="store_true",
                    help="also run the closed-loop body episode, intact and need-lesioned (reported, not gated)")
    ap.add_argument("--tune", action="store_true", help="dev operating-point sweep (intact AFTER arm)")
    ap.add_argument("--tune-weights", type=float, nargs="+", default=[20.0, 40.0, 60.0, 80.0, 120.0])
    ap.add_argument("--tune-intero", type=float, nargs="+", default=[200.0, 300.0, 400.0])
    ap.add_argument("--allow-gate-seeds", action="store_true",
                    help="required to run 42/43/44/100/101/102 (only under the committed preregistration)")
    ap.add_argument("--aggregate", type=str, nargs="+", default=None,
                    help="combine per-seed output JSONs into the preregistered verdict (no simulation)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)
    if args.aggregate:
        agg = aggregate(args.aggregate)
        print(json.dumps({"verdict": agg.get("verdict", agg.get("dev_readout")), "n_go": agg["n_go"],
                          "seed_decisions": {s["seed"]: s["decision"] for s in agg["seed_verdicts"]},
                          "necessity": {k: v["label"] for k, v in agg.get("necessity", {}).items()}}))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(agg, indent=1) + "\n")
        return 0
    if args.seed_range:
        a, b = (int(x) for x in args.seed_range.split("-"))
        args.seeds = list(range(a, b + 1))
    if set(args.seeds) & set(GATE_SEEDS) and not args.allow_gate_seeds:
        raise SystemExit("gate seeds requested without --allow-gate-seeds (preregistration first)")
    params = NeedParams(need_weight=args.need_weight, need_density=args.need_density,
                        intero_max_pA=args.intero_max_pa, receptor=args.receptor)
    xp, backend = get_backend()
    result = {"probe": "need_reversal_selection_seam", "backend": backend, "seeds": list(args.seeds),
              "label": "DEV, not a verdict" if not args.allow_gate_seeds else "GATE (preregistered)"}
    if args.tune:
        grid = []
        for w in args.tune_weights:
            for ipa in args.tune_intero:
                p = dataclasses.replace(params, need_weight=w, intero_max_pA=ipa)
                for seed in args.seeds:
                    s1 = summarize(run_block(seed, 1.0, args.trials, params=p))
                    s0 = summarize(run_block(seed, 0.0, args.trials, params=p))
                    sm = summarize(run_block(seed, 0.5, args.trials, params=p))
                    sa = summarize(run_block(seed, 1.0, args.trials, params=p, arousal=False))
                    row = {"seed": seed, "need_weight": w, "intero_max_pA": ipa,
                           "commit_h1": s1["commit_rate"], "commit_h0": s0["commit_rate"],
                           "p_forage_h1": s1["p_forage"], "p_forage_h0": s0["p_forage"],
                           "p_forage_h05": sm["p_forage"], "commit_h1_no_arousal": sa["commit_rate"],
                           "reversal": reversal_index(s1, s0), "spikes_h1": s1["mean_spikes"]}
                    print(json.dumps(row), flush=True)
                    grid.append(row)
        result["tune"] = grid
    else:
        rows = []
        for seed in args.seeds:
            r = run_seed(seed, trials=args.trials, curve_trials=args.curve_trials, params=params,
                         ablations=not args.no_ablations, curve=not args.no_curve)
            print(json.dumps({"seed": seed, "reversal": r["reversal"], "host_share": r["host_share"],
                              "neural_share": r["neural_share"],
                              "after_host_intervention_byte_identical": r["after_host_intervention_byte_identical"],
                              "static_audit_clean": r["static_audit"]["clean"],
                              "ablations": {k: [v["commit_h1"], v["commit_h0"], v["reversal"]]
                                            for k, v in r["ablations"].items()},
                              "elapsed_s": r["elapsed_s"]}), flush=True)
            rows.append(r)
        result["per_seed"] = rows
        if args.production_baseline:
            result["production_baseline"] = {s: production_baseline(s, trials=args.trials) for s in args.seeds}
            print(json.dumps({"production_baseline": {s: {k: v for k, v in b.items() if k != "prefilter"}
                                                      for s, b in result["production_baseline"].items()}}), flush=True)
        if args.closed_loop:
            result["closed_loop"] = {s: {"intact": closed_loop_episode(s, params=params),
                                         "need_lesion": closed_loop_episode(s, params=params, need_gate=0.0)}
                                     for s in args.seeds}
            print(json.dumps({"closed_loop": {s: {k: {kk: vv for kk, vv in v.items() if kk != "trace"}
                                                  for k, v in b.items()} for s, b in result["closed_loop"].items()}}),
                  flush=True)
        result["bars"] = BARS
        result["seed_verdicts"] = [seed_verdict(r) for r in rows]
        if all(r["ablations"] for r in rows):
            result["necessity"] = necessity_table(rows)
        result.update(overall_block(result["seed_verdicts"], [r["seed"] for r in rows]))
        print(json.dumps({"verdict": result.get("verdict", result.get("dev_readout")), "n_go": result["n_go"],
                          "seed_gates": {s["seed"]: s["gates"] for s in result["seed_verdicts"]},
                          "necessity": {k: v["label"] for k, v in result.get("necessity", {}).items()}}), flush=True)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
