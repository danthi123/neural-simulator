"""Board #91 READ-side fix, attempt 2: a per-granule CUE-RELATIVE (z-scored) excitability
normalization applied at read time, so the DG competition is decided by how much a
granule's drive under the CURRENT cue exceeds ITS OWN baseline excitability under a
broad reference set -- not by absolute drive.

WHERE THE RESIDUAL SITS (read verbatim before building -- do NOT re-derive):
  research/findings/2026-09-01-board91-memory-separator-readfix-popstate-reset-NOGO-relocalizes-to-consolidation-induced-generic-capture.md
  (#91 popreset attempt) found the #90/#78 collapse is NOT specific to the subordinate
  cue: after consolidation, driving ANY input -- the dominant cue, the subordinate cue,
  or a genuinely NOVEL, never-taught, independently-drawn cue -- reactivates the SAME
  dominant DG engram (mean novel->eng0 Jaccard 0.90 post-consolidation vs 0.44
  pre-consolidation, 6/6 seeds). Resetting the population-set-point controller's full
  host state (`bridge._pop_state`: ever/integ/drive/silent) at every read boundary is a
  clean 6/6-seed NO-GO with zero separation between ON/OFF -- the manipulation
  demonstrably lands and changes nothing about which engram wins. The finding's own
  explicit handoff: "attack the DG competition's cue-selectivity directly ... rather
  than hunting for more host state to reset," naming as candidate (3) "normalize each
  DG granule's effective threshold/gain by its OWN drive under the CURRENT cue relative
  to its drive under a broad reference set."

WIRING CHECK BEFORE BUILDING (this file, pure host arithmetic, no sim run needed): the
FIXED random perforant projection (input->dg, dg_fan_in=12 of n_input=48) gives every
granule an almost IDENTICAL expected raw overlap with any input assembly (mean~6,
std~1.5 for m0/m1/novel alike, measured directly from the wiring adjacency for 3
seeds) -- the wiring alone does not privilege specific granules for specific cues, so
the post-consolidation universal-winner set is NOT a static wiring artifact; it must be
a property of the CONSOLIDATED SUBSTRATE'S read-time competitive dynamics (a property
the #90/#91 per-neuron cp_* snapshot and pop_state clear could not fully explain --
banked, not re-derived here).

THE MECHANISM (candidate 3, made concrete; NO sim/ edit): on the SAME already-
consolidated bridge, before scoring the real cue,
  1. Drive N_REF independently-drawn, generic "reference" input patterns (a THIRD RNG
     stream, disjoint from m0/m1/novel) through the identical read protocol (competition
     ON, plasticity OFF) and record each granule's average firing count -- its OWN
     baseline excitability on THIS substrate instance, under THIS substrate's actual
     post-consolidation dynamics (whatever caused the universal-winner bias, this
     measurement inherits and captures it, by construction, regardless of root cause).
  2. z-score across granules: z_g = (ref_rate_g - mean(ref_rate)) / std(ref_rate).
  3. Inject a STATIC per-granule bias current bias_g = -gain * z_g into the DG
     population (on top of, not instead of, the ordinary synaptic drive) for every
     subsequent read on this bridge (m0, m1, novel, and the real behavioral probe).
     A granule that is excitable for ANY input (high z) gets suppressed; a granule
     that is a laggard for generic input gets facilitated -- levelling the competition
     so that CUE-SPECIFIC excess drive (not baseline excitability) decides the k-WTA
     winners.

Biology: divisive/subtractive contrast normalization relative to a population
reference (Carandini & Heeger 2012, Nat Rev Neurosci 13:51-62 -- response normalized
by a POOL, computed here per-granule against a broad generic-input pool rather than a
single scalar pool, i.e. a per-cell gain-control term instead of a single population
set-point); this is the CUE-RELATIVE form the #91 finding explicitly distinguished
from the population-SCALAR set-point already tried and refuted (#78/#91's pop_state).
It is also functionally a CA3/DG "sharpen against your own baseline" completion signal
in spirit (Neunuebel & Knierim 2014, J Neurosci 34:3999-4009: completion is read from
the CURRENT cue, and here the read is literally re-anchored to what is specific about
the current cue for THIS granule) without building a full recurrent attractor (that
heavier candidate-1 build remains banked if this cheaper lever is insufficient).

ANTI-CHEATS (design first, same bar the #90/#91 lineage set, now on BEHAVIORAL both_win
in addition to DG-reactivation Jaccard):
  1. both_win ON: BOTH memories' behavioral probe (dg->answer read) completes to their
     OWN answer assembly, on >=5/6 seeds.
  2. OFF arm (bias=None, otherwise byte-identical build/consolidate/probe): the defect
     REPRODUCES (both_win fails; m1 reactivates the dominant engram) -- dissociates the
     fix from a construction artifact.
  3. No regression: m0 (dominant) still wins its own probe ON, on every seed.
  4. A genuinely novel/untaught cue (independent RNG stream) does NOT spuriously
     complete to either taught DG engram under the SAME bias (Jaccard < 0.5 to both).
  Deterministic (cfg.seed); no sim/ edit; the bias is a runner-side per-granule current
  injection, a no-op when bias=None (guards the OFF arm as a literal skip, not a
  zero-valued call of the same code path, so the OFF arm is exactly the #90/#91
  pipeline).

TUNING DISCIPLINE: --gain is tuned on TUNE_SEEDS (42, 43, 100) only; the reported GO/NO-GO
verdict is evaluated on the FULL 6-seed battery including the held-out seeds
(44, 101, 102) never used to pick --gain.

Run (tune):
    OMP_NUM_THREADS=2 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_readfix_cuerelative --tune

Run (6-seed verdict, after fixing --gain from tuning):
    OMP_NUM_THREADS=2 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_readfix_cuerelative \
        --seeds 42 43 44 100 101 102 --gain <G> \
        --out research/findings/raw/sep_readfix/cuerelative_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import research.runners._replay_dg_pattern_separation_bridge as base  # noqa: E402
from research.runners._replay_dg_pattern_separation_bridge import (  # noqa: E402
    SEEDS,
    DG_COMPETITION_GATE,
    DG_ANSWER_TX_GATE,
    ANSWER_INHIBITION_GATE,
    DG_WRITE_GATE,
    _drain,
    _dg_engram,
)
from research.runners._replay_dg_pattern_separation_gate import (  # noqa: E402
    _answer_assemblies,
    _input_patterns,
    _jaccard,
)
from research.runners._replay_dg_pattern_separation_popsetpoint import (  # noqa: E402
    PConfig,
    build_bridge_popsetpoint,
)
from research.runners._replay_cortical_consolidation_gate import _zero_current  # noqa: E402
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

TUNE_SEEDS = (42, 43, 100)
HELDOUT_SEEDS = (44, 101, 102)
DEFAULT_GAIN = 150.0
DEFAULT_N_REF = 8


def _reference_patterns(seed: int, cfg: PConfig, n_ref: int) -> list[np.ndarray]:
    """N_REF generic input patterns from a THIRD RNG stream, disjoint from the
    m0/m1 ("similar", seed*97+7/19) and novel ("dissimilar", seed*97+19) draws."""
    rng = np.random.default_rng(seed * 727 + 13)
    pool = np.arange(cfg.n_input)
    return [np.sort(rng.choice(pool, cfg.input_assembly, replace=False)) for _ in range(n_ref)]


def _measure_reference_excitability(bridge, cfg: PConfig, regions, seed: int, n_ref: int) -> np.ndarray:
    """Each granule's average firing count under N_REF generic reference cues, on
    THIS (already-consolidated) bridge instance -- captures whatever post-
    consolidation bias exists, by construction, regardless of its physical cause."""
    from sim.backend import to_host
    dg_idx = regions["dg"]
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0)
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 1.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)
    totals = np.zeros(dg_idx.size, dtype=np.float64)
    for pat in _reference_patterns(seed, cfg, n_ref):
        _drain(bridge)
        for _ in range(cfg.probe_steps):
            _zero_current(bridge)
            bridge.cp_external_input_current[pat] = cfg.probe_input_drive_pA
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            totals += np.asarray(to_host(bridge.cp_firing_states[dg_idx]), dtype=np.float64)
    _drain(bridge)
    return totals / max(1, n_ref)


def _bias_from_reference(mean_rate: np.ndarray, gain: float) -> np.ndarray:
    mu = float(np.mean(mean_rate))
    sd = float(np.std(mean_rate)) + 1e-6
    z = (mean_rate - mu) / sd
    return (-float(gain) * z).astype(np.float32)


def _read_with_bias(bridge, cfg: PConfig, regions, inp, bias):
    """Identical to the #90/#91 `_read_reactivation` diagnostic except for the
    optional static per-granule DG bias current (None -> byte-identical OFF path)."""
    from sim.backend import to_host
    dg_idx = regions["dg"]
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0)
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 1.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)
    _drain(bridge)
    counts = np.zeros(dg_idx.size, dtype=np.float64)
    for _ in range(cfg.probe_steps):
        _zero_current(bridge)
        bridge.cp_external_input_current[inp] = cfg.probe_input_drive_pA
        if bias is not None:
            bridge.cp_external_input_current[dg_idx] = bias
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        counts += np.asarray(to_host(bridge.cp_firing_states[dg_idx]), dtype=np.float64)
    _zero_current(bridge)
    return dg_idx[counts > 0]


def _probe_with_bias(bridge, cfg: PConfig, regions, memories, target_name, bias):
    """Identical to `base._probe` except for the optional per-granule DG bias
    current (None -> byte-identical to base._probe)."""
    from sim.backend import to_host
    dg_idx = regions["dg"]
    answer_idx = regions["answer"]
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0)
    bridge.set_transmission_gate(ANSWER_INHIBITION_GATE, 1.0)
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 1.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)
    _drain(bridge)
    inp = memories[target_name]["input"]
    counts = np.zeros(answer_idx.size, dtype=np.float64)
    for _ in range(cfg.probe_steps):
        _zero_current(bridge)
        bridge.cp_external_input_current[inp] = cfg.probe_input_drive_pA
        if bias is not None:
            bridge.cp_external_input_current[dg_idx] = bias
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        counts += np.asarray(to_host(bridge.cp_firing_states[answer_idx]), dtype=np.float64)
    _zero_current(bridge)
    local = {int(g): i for i, g in enumerate(answer_idx)}
    other = [n for n in memories if n != target_name][0]
    correct_pos = np.asarray([local[int(i)] for i in memories[target_name]["answer"]], dtype=np.int64)
    wrong_pos = np.asarray([local[int(i)] for i in memories[other]["answer"]], dtype=np.int64)
    correct = float(counts[correct_pos].mean() / cfg.probe_steps)
    wrong = float(counts[wrong_pos].mean() / cfg.probe_steps)
    return {
        "target": target_name, "correct_rate": correct, "wrong_rate": wrong,
        "selectivity": (correct - wrong) / (correct + wrong + 1e-9),
        "target_assembly_wins": bool(correct > wrong),
        "total_answer_spikes": int(counts.sum()),
    }


def _build_and_consolidate(seed: int, cfg: PConfig, inputs):
    bridge, handles = build_bridge_popsetpoint(seed, cfg)
    regions = handles["regions"]
    answers = _answer_assemblies(seed, cfg, regions["answer"])
    mems = {"m0": {"input": inputs["m0"], "answer": answers["m0"]},
            "m1": {"input": inputs["m1"], "answer": answers["m1"]}}
    eng0, _ = _dg_engram(bridge, cfg, regions, inputs["m0"], True)
    eng1, _ = _dg_engram(bridge, cfg, regions, inputs["m1"], True)
    base._consolidate(bridge, cfg, regions, mems, True, seed)
    return bridge, regions, mems, eng0, eng1


def _install_static_dg_bias(bridge, dg_idx, bias) -> None:
    """Wrap `bridge._run_one_simulation_step` to ADD a constant per-granule bias
    current to `dg_idx` on EVERY step from here on -- during consolidation (the
    WRITE) as well as every later read/probe (the READ). Unlike the read-only
    variant above (which corrects the DG code only at read time, AFTER the
    dg->answer weights were already written against the UNCORRECTED, collapsed
    DG activity), this makes the WRITE see the same corrected code the READ
    will later see, so the two are consistent."""
    orig_step = bridge._run_one_simulation_step
    bias32 = np.asarray(bias, dtype=np.float32)

    def wrapped_step():
        bridge.cp_external_input_current[dg_idx] += bias32
        orig_step()

    bridge._run_one_simulation_step = wrapped_step


def _build_and_consolidate_writebias(seed: int, cfg: PConfig, inputs, gain: float, n_ref: int):
    """Bias is measured on the FRESH (pre-consolidation) bridge -- the intrinsic
    per-granule excitability gap (confirmed to track heterogeneous membrane
    capacitance cp_izh_C, not learning) is present from t=0, no consolidation
    needed to reveal it -- then installed as a PERSISTENT step-wrapper before
    `_consolidate` runs, so the WRITE and every subsequent READ share the same
    corrected DG code."""
    bridge, handles = build_bridge_popsetpoint(seed, cfg)
    regions = handles["regions"]
    dg_idx = regions["dg"]
    ref_rate = _measure_reference_excitability(bridge, cfg, regions, seed, n_ref)
    bias = _bias_from_reference(ref_rate, gain)
    _install_static_dg_bias(bridge, dg_idx, bias)
    answers = _answer_assemblies(seed, cfg, regions["answer"])
    mems = {"m0": {"input": inputs["m0"], "answer": answers["m0"]},
            "m1": {"input": inputs["m1"], "answer": answers["m1"]}}
    eng0, _ = _dg_engram(bridge, cfg, regions, inputs["m0"], True)
    eng1, _ = _dg_engram(bridge, cfg, regions, inputs["m1"], True)
    base._consolidate(bridge, cfg, regions, mems, True, seed)
    return bridge, regions, mems, eng0, eng1, bias


def _dg_row(r0, r1, rn, e0, e1):
    j_m0_e0, j_m0_e1 = _jaccard(r0, e0), _jaccard(r0, e1)
    j_m1_e0, j_m1_e1 = _jaccard(r1, e0), _jaccard(r1, e1)
    j_n_e0, j_n_e1 = _jaccard(rn, e0), _jaccard(rn, e1)
    return {
        "m0_to_eng0": j_m0_e0, "m0_to_eng1": j_m0_e1,
        "m1_to_eng0": j_m1_e0, "m1_to_eng1": j_m1_e1,
        "novel_to_eng0": j_n_e0, "novel_to_eng1": j_n_e1,
        "m0_reactivates_own": bool(j_m0_e0 > j_m0_e1),
        "m1_reactivates_own": bool(j_m1_e1 > j_m1_e0),
        "novel_no_spurious_completion": bool(j_n_e0 < 0.5 and j_n_e1 < 0.5),
    }


def _one_seed(seed: int, cfg: PConfig, gain: float, n_ref: int):
    seed = int(seed)
    inputs = _input_patterns(seed, cfg, "similar")
    novel = _input_patterns(seed, cfg, "dissimilar")["m1"]

    # ----- OFF arm: bias=None (byte-identical to the #90/#91 pipeline) -----
    b_off, regions, mems_off, eng0_off, eng1_off = _build_and_consolidate(seed, cfg, inputs)
    r0_off = _read_with_bias(b_off, cfg, regions, inputs["m0"], None)
    r1_off = _read_with_bias(b_off, cfg, regions, inputs["m1"], None)
    rn_off = _read_with_bias(b_off, cfg, regions, novel, None)
    p0_off = _probe_with_bias(b_off, cfg, regions, mems_off, "m0", None)
    p1_off = _probe_with_bias(b_off, cfg, regions, mems_off, "m1", None)

    # ----- ON arm: fresh build+consolidate (identical), then the cue-relative bias -----
    b_on, regions_o, mems_on, eng0_on, eng1_on = _build_and_consolidate(seed, cfg, inputs)
    ref_rate = _measure_reference_excitability(b_on, cfg, regions_o, seed, n_ref)
    bias = _bias_from_reference(ref_rate, gain)
    r0_on = _read_with_bias(b_on, cfg, regions_o, inputs["m0"], bias)
    r1_on = _read_with_bias(b_on, cfg, regions_o, inputs["m1"], bias)
    rn_on = _read_with_bias(b_on, cfg, regions_o, novel, bias)
    p0_on = _probe_with_bias(b_on, cfg, regions_o, mems_on, "m0", bias)
    p1_on = _probe_with_bias(b_on, cfg, regions_o, mems_on, "m1", bias)

    off_row = _dg_row(r0_off, r1_off, rn_off, eng0_off, eng1_off)
    on_row = _dg_row(r0_on, r1_on, rn_on, eng0_on, eng1_on)

    both_win_off = bool(p0_off["target_assembly_wins"] and p1_off["target_assembly_wins"])
    both_win_on = bool(p0_on["target_assembly_wins"] and p1_on["target_assembly_wins"])

    return {
        "seed": seed,
        "dg_jaccard_eng0_eng1": _jaccard(eng0_off, eng1_off),
        "bias_stats": {"mean": float(np.mean(bias)), "std": float(np.std(bias)),
                       "min": float(np.min(bias)), "max": float(np.max(bias))},
        "off": {**off_row, "p0": p0_off, "p1": p1_off, "both_win": both_win_off},
        "on": {**on_row, "p0": p0_on, "p1": p1_on, "both_win": both_win_on},
    }


def _one_seed_writebias(seed: int, cfg: PConfig, gain: float, n_ref: int):
    """WRITE+READ variant: the bias is installed as a persistent step-wrapper
    BEFORE `_consolidate`, so the WRITE and every READ see the same corrected
    DG code (see `_build_and_consolidate_writebias`)."""
    seed = int(seed)
    inputs = _input_patterns(seed, cfg, "similar")
    novel = _input_patterns(seed, cfg, "dissimilar")["m1"]

    # ----- OFF arm: unbiased build+consolidate+read (byte-identical to #90/#91) -----
    b_off, regions, mems_off, eng0_off, eng1_off = _build_and_consolidate(seed, cfg, inputs)
    r0_off = _read_with_bias(b_off, cfg, regions, inputs["m0"], None)
    r1_off = _read_with_bias(b_off, cfg, regions, inputs["m1"], None)
    rn_off = _read_with_bias(b_off, cfg, regions, novel, None)
    p0_off = _probe_with_bias(b_off, cfg, regions, mems_off, "m0", None)
    p1_off = _probe_with_bias(b_off, cfg, regions, mems_off, "m1", None)

    # ----- ON arm: bias installed BEFORE consolidate (write+read both corrected) -----
    b_on, regions_o, mems_on, eng0_on, eng1_on, bias = _build_and_consolidate_writebias(
        seed, cfg, inputs, gain, n_ref)
    r0_on = _read_with_bias(b_on, cfg, regions_o, inputs["m0"], None)
    r1_on = _read_with_bias(b_on, cfg, regions_o, inputs["m1"], None)
    rn_on = _read_with_bias(b_on, cfg, regions_o, novel, None)
    p0_on = _probe_with_bias(b_on, cfg, regions_o, mems_on, "m0", None)
    p1_on = _probe_with_bias(b_on, cfg, regions_o, mems_on, "m1", None)

    off_row = _dg_row(r0_off, r1_off, rn_off, eng0_off, eng1_off)
    on_row = _dg_row(r0_on, r1_on, rn_on, eng0_on, eng1_on)

    both_win_off = bool(p0_off["target_assembly_wins"] and p1_off["target_assembly_wins"])
    both_win_on = bool(p0_on["target_assembly_wins"] and p1_on["target_assembly_wins"])

    return {
        "seed": seed,
        "dg_jaccard_eng0_eng1": _jaccard(eng0_off, eng1_off),
        "bias_stats": {"mean": float(np.mean(bias)), "std": float(np.std(bias)),
                       "min": float(np.min(bias)), "max": float(np.max(bias))},
        "off": {**off_row, "p0": p0_off, "p1": p1_off, "both_win": both_win_off},
        "on": {**on_row, "p0": p0_on, "p1": p1_on, "both_win": both_win_on},
    }


def run_writebias(seeds, cfg: PConfig, gain: float, n_ref: int):
    started = time.time()
    rows = [_one_seed_writebias(s, cfg, gain, n_ref) for s in seeds]
    n = len(rows)

    def cnt(fn):
        return int(sum(1 for r in rows if fn(r)))

    both_win_on = cnt(lambda r: r["on"]["both_win"])
    both_win_off = cnt(lambda r: r["off"]["both_win"])
    m1_own_on = cnt(lambda r: r["on"]["m1_reactivates_own"])
    m0_win_on = cnt(lambda r: r["on"]["p0"]["target_assembly_wins"])
    m0_win_off = cnt(lambda r: r["off"]["p0"]["target_assembly_wins"])
    novel_ok_on = cnt(lambda r: r["on"]["novel_no_spurious_completion"])
    m0_no_regress = cnt(lambda r: (not r["off"]["p0"]["target_assembly_wins"])
                         or r["on"]["p0"]["target_assembly_wins"])

    heldout_rows = [r for r in rows if r["seed"] in HELDOUT_SEEDS]
    both_win_on_heldout = int(sum(1 for r in heldout_rows if r["on"]["both_win"]))

    checks = {
        "both_win_on_ge5of6": both_win_on >= max(1, round(0.833 * n)) if n >= 6 else both_win_on == n,
        "off_arm_reproduces_defect": both_win_off < n,
        "m0_no_regression_on": m0_no_regress == n,
        "novel_no_spurious_completion_on": novel_ok_on == n,
    }
    go = (checks["both_win_on_ge5of6"] and checks["off_arm_reproduces_defect"]
          and checks["m0_no_regression_on"] and checks["novel_no_spurious_completion_on"])

    both_win_attrib = attributable_to("both_win ON vs OFF (write+read bias)", both_win_on, both_win_off)

    v = Verdict("memory-separator READ+WRITE fix: persistent per-granule cue-relative DG excitability bias, installed before consolidation")
    v.require("baseline is the #90/#91 residual: OFF arm both_win fails on >=1 seed", both_win_off < n, expect=True)
    v.require("no regression: no seed where m0 won OFF and lost ON", m0_no_regress == n, expect=True)
    v.disabled("dg->answer WRITE family (BCM / competitive-heterosynaptic) and the #78/#91 pop-controller reset",
               why="this variant isolates a NEW lever (persistent per-granule cue-relative bias spanning write+read) "
                   "from the banked WRITE family and the banked pop_state reset; base._consolidate's WRITE RULE itself "
                   "is unmodified -- only the DG activity it writes FROM is corrected")
    decided = v.decide(go=go)

    return {
        "gate": "replay_dg_pattern_separation_readfix_cuerelative_writebias",
        "board": "91",
        "mechanism": f"persistent per-granule cue-relative z-scored DG excitability bias (gain={gain}, n_ref={n_ref}), "
                     "installed as a step-wrapper BEFORE consolidation so the WRITE (dg->answer coincidence) and every "
                     "READ (m0/m1/novel/probe) see the SAME corrected DG code (contrast: the read-only variant fixes "
                     "the isolated-reactivation diagnostic and the novel-cue anti-cheat but not behavioral both_win, "
                     "because the write still encodes the uncorrected/collapsed activity)",
        "seeds": [int(s) for s in seeds],
        "n_seeds": n,
        "gain": gain, "n_ref": n_ref,
        "tune_seeds": list(TUNE_SEEDS), "heldout_seeds": list(HELDOUT_SEEDS),
        "status": decided["status"],
        "verdict": decided,
        "preconditions": decided["preconditions"],
        "both_win_attributable_to_bias": both_win_attrib,
        "checks": checks,
        "pooled": {
            "both_win_on_count": both_win_on,
            "both_win_off_count": both_win_off,
            "both_win_on_heldout_count": both_win_on_heldout,
            "n_heldout": len(heldout_rows),
            "m1_reactivates_own_on_count": m1_own_on,
            "m0_wins_on_count": m0_win_on,
            "m0_wins_off_count": m0_win_off,
            "novel_no_spurious_completion_on_count": novel_ok_on,
            "mean_bias_std": float(np.mean([r["bias_stats"]["std"] for r in rows])),
        },
        "per_seed": rows,
        "elapsed_seconds": time.time() - started,
    }


def run(seeds, cfg: PConfig, gain: float, n_ref: int):
    started = time.time()
    rows = [_one_seed(s, cfg, gain, n_ref) for s in seeds]
    n = len(rows)

    def cnt(fn):
        return int(sum(1 for r in rows if fn(r)))

    both_win_on = cnt(lambda r: r["on"]["both_win"])
    both_win_off = cnt(lambda r: r["off"]["both_win"])
    m1_own_on = cnt(lambda r: r["on"]["m1_reactivates_own"])
    m1_own_off = cnt(lambda r: r["off"]["m1_reactivates_own"])
    m0_win_on = cnt(lambda r: r["on"]["p0"]["target_assembly_wins"])
    m0_win_off = cnt(lambda r: r["off"]["p0"]["target_assembly_wins"])
    novel_ok_on = cnt(lambda r: r["on"]["novel_no_spurious_completion"])
    # PER-SEED, OFF-relative non-regression: a seed regresses only if OFF won its
    # own probe and ON does not (comparing to an assumed 6/6 ceiling is the WRONG
    # bar -- the OFF arm itself is not at ceiling on this behavioral metric on
    # every seed; flagging that as "regression" would be a false negative).
    m0_no_regress = cnt(lambda r: (not r["off"]["p0"]["target_assembly_wins"])
                         or r["on"]["p0"]["target_assembly_wins"])

    heldout_rows = [r for r in rows if r["seed"] in HELDOUT_SEEDS]
    both_win_on_heldout = int(sum(1 for r in heldout_rows if r["on"]["both_win"]))

    checks = {
        "both_win_on_ge5of6": both_win_on >= max(1, round(0.833 * n)) if n >= 6 else both_win_on == n,
        "off_arm_reproduces_defect": both_win_off < n,
        "m0_no_regression_on": m0_no_regress == n,
        "novel_no_spurious_completion_on": novel_ok_on == n,
    }
    go = (checks["both_win_on_ge5of6"] and checks["off_arm_reproduces_defect"]
          and checks["m0_no_regression_on"] and checks["novel_no_spurious_completion_on"])

    both_win_attrib = attributable_to("both_win ON vs OFF", both_win_on, both_win_off)

    v = Verdict("memory-separator READ fix: per-granule cue-relative (z-scored) DG excitability bias")
    v.require("baseline is the #90/#91 residual: OFF arm both_win fails on >=1 seed", both_win_off < n, expect=True)
    v.require("no regression: no seed where m0 won OFF and lost ON", m0_no_regress == n, expect=True)
    v.disabled("dg->answer WRITE family (BCM / competitive-heterosynaptic) and the #78/#91 pop-controller reset",
               why="this file isolates a NEW read-side lever (per-granule cue-relative bias) from the banked WRITE family and the banked pop_state reset; base._consolidate + build_bridge_popsetpoint are unmodified")
    decided = v.decide(go=go)

    return {
        "gate": "replay_dg_pattern_separation_readfix_cuerelative",
        "board": "91",
        "mechanism": f"per-granule cue-relative z-scored DG excitability bias (gain={gain}, n_ref={n_ref}): "
                     "bias_g = -gain * (ref_rate_g - mean(ref_rate)) / std(ref_rate), ref_rate measured from "
                     "N_REF generic reference cues on the SAME post-consolidation bridge instance, injected "
                     "as a static per-granule current for every subsequent read (m0/m1/novel/probe)",
        "seeds": [int(s) for s in seeds],
        "n_seeds": n,
        "gain": gain, "n_ref": n_ref,
        "tune_seeds": list(TUNE_SEEDS), "heldout_seeds": list(HELDOUT_SEEDS),
        "status": decided["status"],
        "verdict": decided,
        "preconditions": decided["preconditions"],
        "both_win_attributable_to_bias": both_win_attrib,
        "checks": checks,
        "pooled": {
            "both_win_on_count": both_win_on,
            "both_win_off_count": both_win_off,
            "both_win_on_heldout_count": both_win_on_heldout,
            "n_heldout": len(heldout_rows),
            "m1_reactivates_own_on_count": m1_own_on,
            "m1_reactivates_own_off_count": m1_own_off,
            "m0_wins_on_count": m0_win_on,
            "m0_wins_off_count": m0_win_off,
            "novel_no_spurious_completion_on_count": novel_ok_on,
            "mean_bias_std": float(np.mean([r["bias_stats"]["std"] for r in rows])),
        },
        "per_seed": rows,
        "scaffolds": [
            "host-defined input (sensory) patterns + answer assemblies; host reinstatement of each "
            "memory's input during replay; the transmission/plasticity gate schedule (WRITE vs READ); "
            "the population set-point controller (#78, host PI loop) -- inherited, unmodified. NEW this "
            "file: the reference-cue excitability measurement and the resulting per-granule bias current "
            "are a runner-side host computation + current injection (not a sim/ edit); every read/probe "
            "and the dg->answer write stay on-substrate spiking.",
        ],
        "elapsed_seconds": time.time() - started,
    }


def tune(cfg: PConfig, gains, n_ref: int):
    print(f"[sep-readfix-cuerelative] TUNING on seeds={TUNE_SEEDS} n_ref={n_ref} gains={gains}", flush=True)
    best = None
    for g in gains:
        rows = [_one_seed(s, cfg, g, n_ref) for s in TUNE_SEEDS]
        bw_on = sum(1 for r in rows if r["on"]["both_win"])
        bw_off = sum(1 for r in rows if r["off"]["both_win"])
        m0_ok = sum(1 for r in rows if r["on"]["p0"]["target_assembly_wins"])
        novel_ok = sum(1 for r in rows if r["on"]["novel_no_spurious_completion"])
        print(f"  gain={g:8.1f}  both_win ON={bw_on}/{len(rows)} OFF={bw_off}/{len(rows)} "
              f"m0_ok={m0_ok}/{len(rows)} novel_ok={novel_ok}/{len(rows)}", flush=True)
        score = (bw_on, m0_ok, novel_ok)
        if best is None or score > best[0]:
            best = (score, g)
    print(f"  BEST gain (by tune-seed score) = {best[1]}", flush=True)
    return best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--gain", type=float, default=DEFAULT_GAIN)
    ap.add_argument("--n-ref", type=int, default=DEFAULT_N_REF)
    ap.add_argument("--tune", action="store_true", help="sweep --gain over TUNE_SEEDS only, print, exit")
    ap.add_argument("--mode", choices=["read", "writebias"], default="read",
                     help="'read' = bias only at read/probe time (write is uncorrected); "
                          "'writebias' = bias installed before consolidation (write+read both corrected)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = PConfig()
    print(f"[sep-readfix-cuerelative] backend={os.environ.get('SIM_BACKEND','default')} "
          f"seeds={args.seeds} gain={args.gain} n_ref={args.n_ref} mode={args.mode}", flush=True)

    if args.tune:
        gains = [0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 450.0, 600.0, 900.0]
        tune(cfg, gains, args.n_ref)
        return

    if args.mode == "writebias":
        payload = run_writebias(args.seeds, cfg, args.gain, args.n_ref)
        for r in payload["per_seed"]:
            off, on = r["off"], r["on"]
            print(f"  seed {r['seed']}: dgJ={r['dg_jaccard_eng0_eng1']:.2f} bias_std={r['bias_stats']['std']:.1f} | "
                  f"OFF both_win={off['both_win']} (m0_sel={off['p0']['selectivity']:+.2f} m1_sel={off['p1']['selectivity']:+.2f}) | "
                  f"ON  both_win={on['both_win']} (m0_sel={on['p0']['selectivity']:+.2f} m1_sel={on['p1']['selectivity']:+.2f}) "
                  f"m1->own={on['m1_reactivates_own']} m0->own={on['m0_reactivates_own']} "
                  f"novel_ok={on['novel_no_spurious_completion']} (novel->eng0={on['novel_to_eng0']:.2f})", flush=True)
        print(f"  STATUS: {payload['status']}", flush=True)
        print(f"  checks: {json.dumps(payload['checks'])}", flush=True)
        print(f"  pooled: {json.dumps(payload['pooled'])}", flush=True)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"  wrote {args.out}", flush=True)
        return

    payload = run(args.seeds, cfg, args.gain, args.n_ref)
    for r in payload["per_seed"]:
        off, on = r["off"], r["on"]
        print(f"  seed {r['seed']}: dgJ={r['dg_jaccard_eng0_eng1']:.2f} bias_std={r['bias_stats']['std']:.1f} | "
              f"OFF both_win={off['both_win']} (m0_sel={off['p0']['selectivity']:+.2f} m1_sel={off['p1']['selectivity']:+.2f}) "
              f"m1->own={off['m1_reactivates_own']} | "
              f"ON  both_win={on['both_win']} (m0_sel={on['p0']['selectivity']:+.2f} m1_sel={on['p1']['selectivity']:+.2f}) "
              f"m1->own={on['m1_reactivates_own']} m0->own={on['m0_reactivates_own']} "
              f"novel_ok={on['novel_no_spurious_completion']} (novel->eng0={on['novel_to_eng0']:.2f})", flush=True)
    print(f"  STATUS: {payload['status']}", flush=True)
    print(f"  checks: {json.dumps(payload['checks'])}", flush=True)
    print(f"  pooled: {json.dumps(payload['pooled'])}", flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
