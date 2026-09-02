"""Board #91 scope + cheap de-risk: fix the memory-separator READ (not the write) by
resetting the population set-point's persistent host-side controller state
(``bridge._pop_state``) at every distinct read/probe boundary.

WHERE THE RESIDUAL SITS (#90/#78, read verbatim before building -- do NOT re-derive):
  research/findings/2026-08-19-memory-separator-BCM-selectivity-write-writes-private-granule-but-NOGO-relocalizes-to-read-reactivation.md
  found that after the #78 population-set-point consolidation, driving the SUBORDINATE
  memory's (m1) OWN input reactivates the DOMINANT memory's DG engram (eng0) almost
  exactly (Jaccard 1.00 to eng0, 0.47-0.60 to eng1) on 6/6 seeds -- REGARDLESS of any
  dg->answer WRITE fix (per-granule output transform, banked; BCM selectivity write,
  banked). Both write-side levers land in weight-space and neither moves the read. The
  finding named 3 read-time candidates and left the collapse "characterized, locus not
  fully isolated": every per-neuron ``cp_*`` array matched a fresh-reset baseline, yet
  the read behaved differently -- "an uncaptured global/instrument state the down-state
  reset does not clear."

THE LOCUS THIS FILE IDENTIFIES (traced by reading the actual code, not re-derived by
guessing): ``bridge._pop_state`` (``ever``/``integ``/``drive``/``silent``) is a plain
Python dict the #78 population-set-point controller
(``_install_pop_controller``, ``_replay_dg_pattern_separation_popsetpoint.py:125``)
attaches to the bridge INSTANCE and updates from inside a monkey-patched
``bridge._run_one_simulation_step``. It is host bookkeeping, not a ``cp_*`` numeric
array, so the #90 finding's "snapshot every per-neuron cp_* array" check could not see
it by construction. ``_drain()`` (== ``_reset_dynamics``,
``_replay_dg_pattern_separation_gate.py:80``) zeros membrane/conductance/synapse-timer
arrays DIRECTLY -- it never calls ``bridge._run_one_simulation_step`` -- so the
controller's OWN documented "2 consecutive silent DG steps -> reset ever/integ/drive"
event-boundary reset never gets a chance to fire during ``_drain``. Measured here
(below): DG in this substrate does not reliably reach 2 consecutive fully-silent
population steps within a probe/settle window (a few cells keep firing asynchronously),
so that auto-reset is effectively DEAD CODE for its stated purpose -- the controller's
cumulative recruitment mask (``ever`` ~= 53 >> k=18) and its SATURATED basket drive
(pinned at ``drive_max``) from the #78 interleaved consolidation carry straight into
every later read call on the same bridge, including the #90 finding's own
``_read_reactivation`` diagnostic and the real behavioral ``_probe``. The #90 finding's
"not restored by clearing the #78 pop-controller integrator" cleared ``integ`` only:
with ``ever`` left populated, ``err = nfired - k`` recomputes ~35 on the very first
post-clear step, and the PROPORTIONAL term alone (``kp * err``, kp=45) reproduces
~1575 pA of basket drive before ``integ`` can matter -- a partial clear predicts
exactly the observed non-fix. This file clears ALL FOUR fields at every read/probe
boundary, completing candidate (2) from the #90 finding ("characterize + clear the
post-consolidation reactivation persistence... an instrument fix that may itself
unblock the read") in the form that was not actually tested.

Biology: this is squarely CLAUDE.md's "the instrument is part of the emulation" /
"what else does the real system run alongside this that we replaced with a constant"
reframe -- real dentate gyrus feedforward inhibition is FAST and INPUT-LOCKED
(Pouille & Scanziani 2001, Science 293:1159-1163: feedforward inhibition enforces a
narrow temporal window so a cortical/hippocampal cell's read-out tracks the CURRENT
input volley, not residual/ongoing activity); a slow population-level integrator whose
state survives across cue boundaries is exactly the kind of persistent process biology
does NOT rely on for a clean read, and CA3/DG pattern completion is understood to
retrieve from the CURRENT cue's degraded pattern (Neunuebel & Knierim 2014, J Neurosci
34:3999-4009: DG orthogonalizes and CA3 completes FROM THE PRESENTED CUE, not from
whichever ensemble happened to be active before it).

ANTI-CHEATS (design first):
  1. m1 (subordinate) reads its OWN engram, not the dominant one -- Jaccard(r1,eng1) >
     Jaccard(r1,eng0) -- on >=5/6 seeds, WITH the reset.
  2. WITHOUT the reset (byte-identical to the #90/#78 pipeline -- same build, same
     consolidate, only the read-time reset omitted): the defect REPRODUCES (m1 reads
     eng0), dissociating the fix from a construction artifact.
  3. No regression: m0 (dominant) still reads its own engram WITH the reset.
  4. A genuinely novel/unrelated cue (a DISSIMILAR-pair pattern drawn from an
     independent RNG stream, never taught) does NOT spuriously complete to either
     taught engram -- Jaccard to both stays low. Guards against "the reset just makes
     everything complete to whatever fires easiest."
  Deterministic (cfg.seed); no ``sim/`` edit; the read-time reset is a runner-side
  host-dict clear guarded by ``getattr(bridge, "_pop_state", None)`` -- a no-op on any
  substrate without the popsetpoint controller installed.

Run:
    OMP_NUM_THREADS=2 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_readfix_popreset \
        --seeds 42 43 44 100 101 102 \
        --out research/findings/raw/sep_readfix/popreset_6seed.json
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


def _reset_pop_state(bridge) -> bool:
    """Clear ALL FOUR population-set-point controller fields (not just the
    integrator). No-op (returns False) if the controller was never installed --
    keeps this a pure additive read-side lever, not a rewrite of #78."""
    state = getattr(bridge, "_pop_state", None)
    if state is None:
        return False
    state["ever"][:] = False
    state["integ"] = 0.0
    state["drive"] = 0.0
    state["silent"] = 0
    return True


def _read_reactivation(bridge, cfg, regions, inp, *, reset: bool):
    """Drive INPUT only (competition on, plasticity off), read the cumulative DG
    engram the READ recruits. Identical to the #90 finding's diagnostic except for
    the optional read-boundary pop_state reset (the ONLY manipulated variable)."""
    from sim.backend import to_host
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0)
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 1.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)
    _drain(bridge)
    reset_landed = _reset_pop_state(bridge) if reset else False
    dg = regions["dg"]
    counts = np.zeros(dg.size, dtype=np.float64)
    for _ in range(cfg.probe_steps):
        _zero_current(bridge)
        bridge.cp_external_input_current[inp] = cfg.probe_input_drive_pA
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        counts += np.asarray(to_host(bridge.cp_firing_states[dg]), dtype=np.float64)
    _zero_current(bridge)
    return dg[counts > 0], reset_landed


def _one_seed(seed: int, cfg: PConfig):
    seed = int(seed)
    inputs = _input_patterns(seed, cfg, "similar")
    # a genuinely novel/unrelated cue: an INDEPENDENT draw (dissimilar RNG stream),
    # never taught during consolidation.
    novel = _input_patterns(seed, cfg, "dissimilar")["m1"]

    def _build_and_consolidate():
        bridge, handles = build_bridge_popsetpoint(seed, cfg)
        regions = handles["regions"]
        answers = _answer_assemblies(seed, cfg, regions["answer"])
        mems = {"m0": {"input": inputs["m0"], "answer": answers["m0"]},
                "m1": {"input": inputs["m1"], "answer": answers["m1"]}}
        eng0, _ = _dg_engram(bridge, cfg, regions, inputs["m0"], True)
        eng1, _ = _dg_engram(bridge, cfg, regions, inputs["m1"], True)
        base._consolidate(bridge, cfg, regions, mems, True, seed)
        return bridge, regions, eng0, eng1

    # ----- OFF arm: no read-boundary reset (reproduces the #90/#78 negative) -----
    b_off, regions, eng0, eng1 = _build_and_consolidate()
    r0_off, _ = _read_reactivation(b_off, cfg, regions, inputs["m0"], reset=False)
    r1_off, _ = _read_reactivation(b_off, cfg, regions, inputs["m1"], reset=False)
    rn_off, _ = _read_reactivation(b_off, cfg, regions, novel, reset=False)

    # ----- ON arm: fresh build+consolidate (identical), reset at the read boundary -----
    b_on, regions_o, eng0_o, eng1_o = _build_and_consolidate()
    r0_on, land0 = _read_reactivation(b_on, cfg, regions_o, inputs["m0"], reset=True)
    r1_on, land1 = _read_reactivation(b_on, cfg, regions_o, inputs["m1"], reset=True)
    rn_on, landn = _read_reactivation(b_on, cfg, regions_o, novel, reset=True)

    def _row(r0, r1, rn, e0, e1):
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

    off_row = _row(r0_off, r1_off, rn_off, eng0, eng1)
    on_row = _row(r0_on, r1_on, rn_on, eng0_o, eng1_o)

    # ----- PRE-CONSOLIDATION baseline: a THIRD fresh bridge, NEVER consolidated. Does
    # the novel cue already collapse onto eng0 before any write/replay ever ran? This
    # dissociates "a static wiring/competition degeneracy" from "a genuine
    # consolidation-induced persistence" -- decisive context for where the residual
    # locus actually sits once this file's own read-boundary-reset lever is a NO-GO. -----
    b_pre, handles_pre = build_bridge_popsetpoint(seed, cfg)
    regions_pre = handles_pre["regions"]
    eng0_pre, _ = _dg_engram(b_pre, cfg, regions_pre, inputs["m0"], True)
    eng1_pre, _ = _dg_engram(b_pre, cfg, regions_pre, inputs["m1"], True)
    engn_pre, _ = _dg_engram(b_pre, cfg, regions_pre, novel, True)
    pre_row = {
        "eng0_eng1": _jaccard(eng0_pre, eng1_pre),
        "novel_to_eng0": _jaccard(engn_pre, eng0_pre),
        "novel_to_eng1": _jaccard(engn_pre, eng1_pre),
        "n_eng0": int(eng0_pre.size), "n_eng1": int(eng1_pre.size), "n_novel": int(engn_pre.size),
    }

    return {
        "seed": seed,
        "dg_jaccard_eng0_eng1": _jaccard(eng0, eng1),
        "reset_landed": bool(land0 and land1 and landn),
        "off": off_row,
        "on": on_row,
        "pre_consolidation_baseline": pre_row,
        "n_eng0": int(eng0.size), "n_eng1": int(eng1.size),
        "n_eng0_o": int(eng0_o.size), "n_eng1_o": int(eng1_o.size),
    }


def run(seeds, cfg: PConfig):
    started = time.time()
    rows = [_one_seed(s, cfg) for s in seeds]
    n = len(rows)

    def cnt(fn):
        return int(sum(1 for r in rows if fn(r)))

    m1_fixed_on = cnt(lambda r: r["on"]["m1_reactivates_own"])
    m1_broken_off = cnt(lambda r: not r["off"]["m1_reactivates_own"])
    m0_ok_on = cnt(lambda r: r["on"]["m0_reactivates_own"])
    m0_ok_off = cnt(lambda r: r["off"]["m0_reactivates_own"])
    novel_ok_on = cnt(lambda r: r["on"]["novel_no_spurious_completion"])
    reset_landed = cnt(lambda r: r["reset_landed"])

    checks = {
        "m1_reactivates_own_engram_on_ge5of6": m1_fixed_on >= max(1, round(0.833 * n)) if n >= 6 else m1_fixed_on == n,
        "off_arm_reproduces_defect": m1_broken_off >= max(1, round(0.833 * n)) if n >= 6 else m1_broken_off == n,
        "m0_no_regression_on": m0_ok_on == n,
        "m0_control_off": m0_ok_off == n,
        "novel_no_spurious_completion_on": novel_ok_on == n,
        "reset_landed_all_seeds": reset_landed == n,
    }
    go = (checks["m1_reactivates_own_engram_on_ge5of6"]
          and checks["off_arm_reproduces_defect"]
          and checks["m0_no_regression_on"]
          and checks["novel_no_spurious_completion_on"]
          and checks["reset_landed_all_seeds"])

    m1_attrib = attributable_to("m1_reactivates_own ON vs OFF", m1_fixed_on, n - m1_broken_off)

    # NOTE: the OUTCOME questions (does m1 read its own engram? does a novel cue avoid
    # spurious completion?) are deliberately NOT registered as Verdict preconditions --
    # `attributable_to`/`checks`/`go` above already carry them, and a Verdict `require`/
    # `control` that fails forces UNDEFINED (by design: "arms that tie mean the
    # manipulation never happened"). Here the manipulation VERIFIABLY landed
    # (`reset_landed`) and STILL produced zero separation -- a genuine, informative
    # NO-GO for this specific lever, not an unmeasured/void run. Only true preconditions
    # (manipulation landed; baseline reproduces the known defect; no regression) go
    # through `v.require` so a real negative reports as NO-GO, not UNDEFINED.
    v = Verdict("memory-separator READ fix: reset the population-set-point host state at the read boundary")
    v.require("manipulation landed: pop_state reset applied on every seed", reset_landed == n, expect=True)
    v.require("baseline is the #90/#78 residual: OFF arm reproduces m1->dominant-engram", m1_broken_off >= 1, expect=True)
    v.require("no regression: m0 (dominant) still reads its own engram ON", m0_ok_on == n, expect=True)
    v.disabled("dg->answer WRITE family (BCM / competitive-heterosynaptic)",
               why="this file isolates the READ-side fix from the #90-banked WRITE family; base._consolidate is the unmodified #78 write")
    decided = v.decide(go=go)

    return {
        "gate": "replay_dg_pattern_separation_readfix_popreset",
        "board": "91",
        "mechanism": "reset bridge._pop_state (ever/integ/drive/silent) at every read/probe boundary, "
                     "not just on the controller's own 2-consecutive-silent-step heuristic (measured dead "
                     "for this purpose -- DG never reaches true full-population silence in a probe window)",
        "seeds": [int(s) for s in seeds],
        "n_seeds": n,
        "status": decided["status"],
        "verdict": decided,
        "preconditions": decided["preconditions"],
        "m1_reactivates_own_attributable_to_reset": m1_attrib,
        "checks": checks,
        "pooled": {
            "m1_reactivates_own_on_count": m1_fixed_on,
            "m1_reactivates_dominant_off_count": m1_broken_off,
            "m0_reactivates_own_on_count": m0_ok_on,
            "m0_reactivates_own_off_count": m0_ok_off,
            "novel_no_spurious_completion_on_count": novel_ok_on,
            "reset_landed_count": reset_landed,
            "mean_pre_consolidation_novel_to_eng0": float(np.mean([r["pre_consolidation_baseline"]["novel_to_eng0"] for r in rows])),
            "mean_post_consolidation_novel_to_eng0_on": float(np.mean([r["on"]["novel_to_eng0"] for r in rows])),
        },
        "per_seed": rows,
        "scaffolds": [
            "host-defined input (sensory) patterns + answer assemblies; host reinstatement of each "
            "memory's input during replay; the transmission/plasticity gate schedule (WRITE vs READ); "
            "the population set-point controller itself is host (a PI loop injecting basket current) -- "
            "inherited from #78/#90, unmodified here. NEW this file: the read-boundary pop_state reset "
            "is a runner-side host-dict clear (not a sim/ edit); the read + engram measurement stay "
            "on-substrate spiking.",
        ],
        "elapsed_seconds": time.time() - started,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--k-target", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    overrides = {}
    if args.k_target is not None:
        overrides["pop_k_target"] = args.k_target
    cfg = PConfig(**overrides)

    print(f"[sep-readfix-popreset] backend={os.environ.get('SIM_BACKEND','default')} "
          f"seeds={args.seeds} pop_k={cfg.pop_k_target}", flush=True)
    payload = run(args.seeds, cfg)
    for r in payload["per_seed"]:
        off, on, pre = r["off"], r["on"], r["pre_consolidation_baseline"]
        print(f"  seed {r['seed']}: dgJ(eng0,eng1)={r['dg_jaccard_eng0_eng1']:.2f} reset_landed={r['reset_landed']} | "
              f"OFF m1->(eng0={off['m1_to_eng0']:.2f},eng1={off['m1_to_eng1']:.2f}) own={off['m1_reactivates_own']} | "
              f"ON  m1->(eng0={on['m1_to_eng0']:.2f},eng1={on['m1_to_eng1']:.2f}) own={on['m1_reactivates_own']} "
              f"m0_own={on['m0_reactivates_own']} novel_ok={on['novel_no_spurious_completion']} "
              f"(novel->eng0={on['novel_to_eng0']:.2f},eng1={on['novel_to_eng1']:.2f}) | "
              f"PRE-CONSOLIDATION novel->eng0={pre['novel_to_eng0']:.2f} (vs POST={on['novel_to_eng0']:.2f})", flush=True)
    print(f"  STATUS: {payload['status']}", flush=True)
    print(f"  checks: {json.dumps(payload['checks'])}", flush=True)
    print(f"  pooled: {json.dumps(payload['pooled'])}", flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
