"""Spiking REACHABILITY for the transitive chat: "does A reach D through the stored R-facts" is decided by spikes.

Packet `transitive-spiking-reach` (session-2 packet factory, 2026-10-01).

WHAT IT REPLACES. `webapp/reasoning_transitive_chat.resolve_transitive_query` walks the chain with a host loop
(`webapp.gnw_multistep_deliberation.multistep_chase` -> `confidence_gated_chase`, one `composer.query_patient` per
hop) and then decides "yes" with host code: `if target not in path[1:]`. The reachability verdict is a host membership
test over a host-built path.

WHAT THIS ORGAN DOES INSTEAD (one SimulationBridge per (composer fact set, relation); no hop loop, no membership test):

  * `wave` -- one assembly of N_W excitatory neurons per concept of the relation subgraph. A directed dense projection
    wave[x] -> wave[y] exists for every stored fact R(x, y) (weight W_HOP, transmission gate `reach_hop`). Activity
    started in wave[start] propagates x -> y -> z along the stored edges and dies at a leaf (no persistence).
  * `goal` -- one assembly per concept. The host renders the heard question word as a sensory current into
    goal[target] for the whole query window (the world/sense side).
  * `match` -- one assembly per concept. It receives wave[c] and goal[c] on separate subthreshold projections
    (W_WAVE_MATCH, W_GOAL_MATCH) and fires only on the coincidence of both.
  * `detect` -- ONE population receiving every match[c] (W_MATCH_DETECT). The host reads ONLY its spike count:
    reached = detect count >= DETECT_MIN. The host never reads which concept fired to decide yes/no.

The host's part: the sensory pulse into wave[start] (START_PA for START_STEPS), the held goal current, the fixed
QUERY_STEPS window clock, and the single detect-count threshold DETECT_MIN (frozen in the operating-point grid).

LESION (`lesion=True`, production flag `BRAIN_TRANSITIVE_REACH_LESION`): the transmission gate `reach_hop` is closed
(0). The wave cannot leave the start assembly, so match[target] never sees wave input, detect is silent, and the
organ abstains. Only that gate is touched.

PATH READ-OUT (narration only, never the decision). `fired_order` = concepts whose wave assembly fired, ordered by
first-spike time. `path` = a backtrack from the target along stored edges through fired concepts with strictly
earlier first-spike times, so that `derived_from` pairs are real stored facts even in a branching graph. For a plain
chain `path == fired_order` up to the target.

DECLARED RESIDUALS (honest).
  * EDGE TRANSCRIPTION. `reach_from_composer` transcribes the edges from the composer's fact index (`composer.kb`) at
    query time: the wave projections are wired by host code from the stored facts. This is a host transcription
    residual; learning the hop projections by STDP from the taught facts is the next rung.
  * Concept identity routing (word -> assembly) and the query clock are host (the senses).
  * The sentence parse ("does X R Y?") stays the declared comprehension boundary in the webapp module.

OPERATING POINT. Constants live in the frozen `GRID` (most conservative first); `PARAMS = dict(GRID[0])`,
`set_operating_point(i)` copies GRID[i] into the module. Tuned on dev seeds 1-8 only; no gate-seed run, no gate claim.

Backend: numpy. No `sim/` edit. Same global-RNG isolation as the ToM witness organ.

MEASURED (dev seeds 1-8 only, 2026-10-01; no gate seed run, no gate claim). Battery = CHAIN_A cat->ball->worm->river,
cases derived2 / derived3 / reversed / unrelated / branch / lesion_derived3 / path3.
  * GRID[0]..GRID[4] pass the whole battery on ALL of dev seeds 1-8. GRID[0] (the default): derived2 detect 40 on every
    seed, derived3 39-40, branch_river 40-45, branch_sun 40, reversed 0, lesion_derived3 0, unrelated 0-2 (seed 4: 1,
    seed 7: 2), path3 == [cat, ball, worm, river] on all 8 seeds. Detect saturates near N_DETECT (one spike per detect
    neuron) when reached, so DETECT_MIN = 10 sits far above the leak (<= 2) and below the signal (>= 36 across
    GRID[0..2]; GRID[3], GRID[4] reach as low as 20-23 on dev seeds 1 and 8 and pass anyway).
  * GRID[5] (W_HOP 30) FAILS: the wave is strong enough to fire match[c] on wave input alone, so `unrelated` reads
    ~26-60 (wave-alone leak). GRID[6] (W_HOP 14) FAILS: the wave decays before the third hop (derived3 1-10).
  * KNOWN LIMIT (not deferred, a finding): the operating window between "wave decays" (W_HOP <= 14) and "wave alone
    fires match" (W_HOP >= 24) is narrow, so a 7-hop chain a->...->h reaches only weakly (detect 1-16, flaky against
    DETECT_MIN) at W_HOP 20; 3-hop chains are solid. Next lever: lateral inhibition on `match` to remove the
    wave-alone leak, then raise W_HOP for depth.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.regions import BrainRegion  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402

from research.runners._gnw_rung1_ignition_curve_derisk import SETTLE_STEPS  # noqa: E402
from research.runners._gnw_rung3_report_reasoning_identity_derisk import _dense_projection  # noqa: E402

_DEFAULT_SEED = 42
GATE_SEEDS = (42, 43, 44, 100, 101, 102)
HOP_GATE = "reach_hop"                  # transmission gate on every wave -> wave projection (only the lesion closes it)
FROZEN_GATE = "reach_fixed"             # plasticity gate name carried by every (non-plastic) projection

# ── structure (not tuned) ──────────────────────────────────────────────────────────────────────────────────────────
N_W = 30                                # wave assembly size per concept
N_G = 30                                # goal assembly size per concept
N_M = 30                                # match assembly size per concept
N_DETECT = 40                           # the single detect population
QUERY_STEPS = 150                       # the fixed query window (host clock)
MIN_ASSEMBLY_SPIKES = 3                 # narration only: a wave assembly "fired" if it spiked at least this often

# ── the frozen operating-point grid (weights / currents / threshold); most conservative first ─────────────────────
GRID: List[dict] = [
    dict(W_HOP=20.0, W_WAVE_MATCH=3.0, W_GOAL_MATCH=1.5, W_MATCH_DETECT=6.0, START_PA=1500.0, START_STEPS=20,
         GOAL_PA=1000.0, DETECT_MIN=10),
    dict(W_HOP=20.0, W_WAVE_MATCH=3.0, W_GOAL_MATCH=1.0, W_MATCH_DETECT=6.0, START_PA=1500.0, START_STEPS=20,
         GOAL_PA=1500.0, DETECT_MIN=10),
    dict(W_HOP=20.0, W_WAVE_MATCH=3.0, W_GOAL_MATCH=2.0, W_MATCH_DETECT=6.0, START_PA=1500.0, START_STEPS=20,
         GOAL_PA=600.0, DETECT_MIN=10),
    dict(W_HOP=20.0, W_WAVE_MATCH=3.0, W_GOAL_MATCH=1.5, W_MATCH_DETECT=6.0, START_PA=1500.0, START_STEPS=20,
         GOAL_PA=600.0, DETECT_MIN=10),
    dict(W_HOP=20.0, W_WAVE_MATCH=3.0, W_GOAL_MATCH=1.0, W_MATCH_DETECT=6.0, START_PA=1500.0, START_STEPS=20,
         GOAL_PA=1000.0, DETECT_MIN=10),
    dict(W_HOP=30.0, W_WAVE_MATCH=3.0, W_GOAL_MATCH=1.5, W_MATCH_DETECT=6.0, START_PA=1500.0, START_STEPS=20,
         GOAL_PA=1000.0, DETECT_MIN=10),
    dict(W_HOP=14.0, W_WAVE_MATCH=3.0, W_GOAL_MATCH=1.5, W_MATCH_DETECT=6.0, START_PA=1500.0, START_STEPS=20,
         GOAL_PA=1000.0, DETECT_MIN=10),
]
PARAMS = dict(GRID[0])
W_HOP = W_WAVE_MATCH = W_GOAL_MATCH = W_MATCH_DETECT = START_PA = START_STEPS = GOAL_PA = DETECT_MIN = None


def set_operating_point(i: int) -> dict:
    """Copy GRID[i] into the module-level constants (and PARAMS). Returns the point."""
    global W_HOP, W_WAVE_MATCH, W_GOAL_MATCH, W_MATCH_DETECT, START_PA, START_STEPS, GOAL_PA, DETECT_MIN
    p = dict(GRID[int(i)])
    PARAMS.clear()
    PARAMS.update(p)
    W_HOP, W_WAVE_MATCH, W_GOAL_MATCH = p["W_HOP"], p["W_WAVE_MATCH"], p["W_GOAL_MATCH"]
    W_MATCH_DETECT, START_PA, START_STEPS = p["W_MATCH_DETECT"], p["START_PA"], int(p["START_STEPS"])
    GOAL_PA, DETECT_MIN = p["GOAL_PA"], int(p["DETECT_MIN"])
    return p


set_operating_point(0)


def _isolated(fn):
    """Run `fn` and restore the host numpy (and backend) global RNG state afterwards."""
    try:
        xp, _ = get_backend()
    except Exception:
        xp = None
    host_np = np.random.get_state()
    host_xp = None
    if xp is not None and xp is not np:
        try:
            host_xp = xp.random.get_random_state().get_state()
        except Exception:
            host_xp = None
    try:
        return fn()
    finally:
        np.random.set_state(host_np)
        if host_xp is not None:
            try:
                xp.random.get_random_state().set_state(host_xp)
            except Exception:
                pass


def concepts_of(edges: Sequence[Tuple[str, str]]) -> List[str]:
    """Concepts of the relation subgraph in first-appearance order (every agent / patient of a stored fact)."""
    out: List[str] = []
    for x, y in edges:
        for c in (x, y):
            if c not in out:
                out.append(c)
    return out


def build_reach_bridge(edges: Sequence[Tuple[str, str]], seed: int = _DEFAULT_SEED):
    """One bridge for the relation subgraph given by `edges` [(x, y), ...] (stored facts R(x, y)).
    Returns (bridge, xp, idx, snap) with idx = {concepts, wave/goal/match: {concept: host index array}, detect, ...}."""
    xp, _ = get_backend()
    cons = concepts_of(edges)
    n = len(cons)
    regions = [
        BrainRegion(name="wave", n_neurons=N_W * n, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="goal", n_neurons=N_G * n, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="match", n_neurons=N_M * n, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="detect", n_neurons=N_DETECT, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
    ]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = []
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)
    cfg.enable_nmda = False
    for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = True
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                              gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    rm = bridge.region_manager

    def members(name, size):
        base = np.asarray(rm.indices(name), dtype=np.int64)
        return {c: base[i * size:(i + 1) * size] for i, c in enumerate(cons)}

    wave, goal, match = members("wave", N_W), members("goal", N_G), members("match", N_M)
    detect = np.asarray(rm.indices("detect"), dtype=np.int64)

    union = dict(rm.build_wiring_plan(seed=int(seed)))
    for k, (x, y) in enumerate(edges):
        d = _dense_projection(wave[x], wave[y], W_HOP, FROZEN_GATE)
        d["transmission_gate"] = HOP_GATE
        union[f"hop_{k}"] = d
    for c in cons:
        union[f"wave_match_{c}"] = _dense_projection(wave[c], match[c], W_WAVE_MATCH, FROZEN_GATE)
        union[f"goal_match_{c}"] = _dense_projection(goal[c], match[c], W_GOAL_MATCH, FROZEN_GATE)
        union[f"match_detect_{c}"] = _dense_projection(match[c], detect, W_MATCH_DETECT, FROZEN_GATE)
    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(FROZEN_GATE, 0.0)
    if edges:
        bridge.set_transmission_gate(HOP_GATE, 1.0)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    idx = {"concepts": cons, "wave": wave, "goal": goal, "match": match, "detect": detect, "has_edges": bool(edges),
           "wave_dev": {c: xp.asarray(v) for c, v in wave.items()},
           "goal_dev": {c: xp.asarray(v) for c, v in goal.items()}}
    return bridge, xp, idx


def run_query(bridge, xp, idx, start: str, target: str, *, lesion: bool = False) -> dict:
    """Drive wave[start] (sensory pulse) and goal[target] (the heard word, held), run the fixed window, read detect.
    Returns {"detect_spikes", "reached", "first_spike": {concept: step | None}, "wave_spikes", "match_spikes"}.
    The decision is detect_spikes >= DETECT_MIN, nothing else."""
    cons = idx["concepts"]
    if idx["has_edges"]:
        bridge.set_transmission_gate(HOP_GATE, 0.0 if lesion else 1.0)
    first: Dict[str, Optional[int]] = {c: None for c in cons}
    wave_sp = {c: 0 for c in cons}
    match_sp = {c: 0 for c in cons}
    detect_sp = 0
    for t in range(QUERY_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        if target in idx["goal_dev"]:
            bridge.cp_external_input_current[idx["goal_dev"][target]] = xp.float32(GOAL_PA)
        if t < START_STEPS and start in idx["wave_dev"]:
            bridge.cp_external_input_current[idx["wave_dev"][start]] = xp.float32(START_PA)
        bridge._run_one_simulation_step()
        fs = to_host(bridge.cp_firing_states).astype(np.float64)
        detect_sp += int(fs[idx["detect"]].sum())
        for c in cons:
            w = int(fs[idx["wave"][c]].sum())
            if w:
                wave_sp[c] += w
                if first[c] is None and wave_sp[c] >= MIN_ASSEMBLY_SPIKES:
                    first[c] = t
            match_sp[c] += int(fs[idx["match"][c]].sum())
    bridge.cp_external_input_current[:] = 0.0
    return {"detect_spikes": int(detect_sp), "reached": bool(detect_sp >= DETECT_MIN), "first_spike": first,
            "wave_spikes": wave_sp, "match_spikes": match_sp}


def read_path(start: str, target: str, edges: Sequence[Tuple[str, str]], first_spike: Dict[str, Optional[int]]
              ) -> Tuple[List[str], List[str]]:
    """Narration only. Returns (path, fired_order). `fired_order`: concepts whose wave assembly fired, by first-spike
    time (ties: first-appearance order). `path`: backtrack from `target` through fired concepts with strictly earlier
    first-spike time along stored edges, ending at `start`; falls back to `fired_order` cut at the target."""
    fired = [c for c in first_spike if first_spike[c] is not None]
    order = sorted(fired, key=lambda c: (first_spike[c], fired.index(c)))
    if start in order:
        order = [start] + [c for c in order if c != start]
    path = [target]
    cur = target
    while cur != start and cur in fired:
        preds = [x for (x, y) in edges if y == cur and x in fired and first_spike[x] <= first_spike[cur] and x != cur
                 and x not in path]
        if not preds:
            break
        cur = max(preds, key=lambda x: (first_spike[x], -fired.index(x)))
        path.append(cur)
    path.reverse()
    if path and path[0] == start and path[-1] == target:
        return path, order
    cut = order[:order.index(target) + 1] if target in order else order
    return cut, order


def reach(edges: Sequence[Tuple[str, str]], start: str, target: str, *, seed: int, lesion: bool = False,
          bridge_bundle=None) -> dict:
    """Build the bridge for `edges`, run one query. Returns {reached, detect_spikes, path, fired_order, n_concepts,
    lesioned}. The builder is skipped for an empty graph (nothing can fire)."""
    edges = [(str(x), str(y)) for x, y in edges]
    cons = concepts_of(edges)
    if not edges or start not in cons or target not in cons:
        return {"reached": False, "detect_spikes": 0, "path": [start], "fired_order": [], "n_concepts": len(cons),
                "lesioned": bool(lesion)}

    def _go():
        bundle = bridge_bundle or build_reach_bridge(edges, seed)
        bridge, xp, idx = bundle
        r = run_query(bridge, xp, idx, start, target, lesion=lesion)
        path, order = read_path(start, target, edges, r["first_spike"])
        return {"reached": r["reached"], "detect_spikes": r["detect_spikes"], "path": path, "fired_order": order,
                "n_concepts": len(cons), "lesioned": bool(lesion), "first_spike": r["first_spike"],
                "wave_spikes": r["wave_spikes"], "match_spikes": r["match_spikes"]}

    return _isolated(_go)


def composer_edges(composer, relation: str) -> List[Tuple[str, str]]:
    """The (agent, patient) pairs of the composer's stored facts whose action lemma equals `relation`.
    HOST TRANSCRIPTION RESIDUAL: read off `composer.kb` at query time (see the module docstring)."""
    from research.runners.lexical_lemma import lemma_verb
    rel = lemma_verb(str(relation).lower())
    out: List[Tuple[str, str]] = []
    for item in list(getattr(composer, "kb", []) or []):
        fact = item[0] if isinstance(item, (tuple, list)) else item
        if not isinstance(fact, dict):
            continue
        a, act, p = fact.get("agent"), fact.get("action"), fact.get("patient")
        if not (isinstance(a, str) and isinstance(act, str) and isinstance(p, str)):
            continue
        if lemma_verb(act.lower()) != rel:
            continue
        e = (a.lower(), p.lower())
        if e not in out and e[0] != e[1]:
            out.append(e)
    return out


def reach_from_composer(composer, agent: str, relation: str, target: str, *, seed: int, lesion: bool = False) -> dict:
    """Spiking reachability over the composer's own stored facts. The edges are transcribed from `composer.kb` at query
    time (a host transcription residual; learning them by STDP is the next rung); the verdict is detect's spike count.
    Returns {reached, detect_spikes, path, n_concepts, lesioned}."""
    edges = composer_edges(composer, relation)
    r = reach(edges, str(agent).lower(), str(target).lower(), seed=int(seed), lesion=bool(lesion))
    return {"reached": bool(r["reached"]), "detect_spikes": int(r["detect_spikes"]), "path": list(r["path"]),
            "n_concepts": int(r["n_concepts"]), "lesioned": bool(lesion)}


# ── the organ battery (explicit fact list, no composer) ───────────────────────────────────────────────────────────
CHAIN_A = [("cat", "ball"), ("ball", "worm"), ("worm", "river")]
ISOLATED = [("dog", "bone")]
BRANCH = [("cat", "sun")]


def organ_battery(seed: int, op_index: Optional[int] = None) -> dict:
    """The organ battery on explicit edge lists. Per case: detect count, reached, pass."""
    if op_index is not None:
        set_operating_point(op_index)
    cases: Dict[str, dict] = {}

    def case(name, edges, start, target, want, lesion=False, want_path=None):
        r = reach(edges, start, target, seed=seed, lesion=lesion)
        ok = (r["reached"] is want)
        if want_path is not None:
            ok = ok and list(r["path"]) == list(want_path)
        cases[name] = {"detect": r["detect_spikes"], "reached": r["reached"], "want_reached": want,
                       "path": r["path"], "pass": bool(ok)}

    case("derived2", CHAIN_A, "cat", "worm", True)
    case("derived3", CHAIN_A, "cat", "river", True)
    case("reversed", CHAIN_A, "river", "cat", False)
    case("unrelated", CHAIN_A + ISOLATED, "cat", "bone", False)
    case("branch_river", CHAIN_A + BRANCH, "cat", "river", True)
    case("branch_sun", CHAIN_A + BRANCH, "cat", "sun", True)
    case("lesion_derived3", CHAIN_A, "cat", "river", False, lesion=True)
    case("path3", CHAIN_A, "cat", "river", True, want_path=["cat", "ball", "worm", "river"])
    cases["branch"] = {"detect": min(cases["branch_river"]["detect"], cases["branch_sun"]["detect"]),
                       "pass": bool(cases["branch_river"]["pass"] and cases["branch_sun"]["pass"])}
    return {"seed": int(seed), "op_index": None if op_index is None else int(op_index), "params": dict(PARAMS),
            "cases": cases, "all_pass": bool(all(c["pass"] for c in cases.values())), "backend": "numpy"}


def _refuse_gate_seeds(seeds, allow: bool, prereg: Optional[str]) -> None:
    bad = [s for s in seeds if int(s) in GATE_SEEDS]
    if bad and not (allow and prereg):
        raise SystemExit("refusing gate seed(s) %s: pass --allow-gate-seed --prereg <path> (dev seeds 1-8 only)" % bad)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--op", type=int, default=0, help="GRID index")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--ops", type=int, nargs="*", default=None, help="GRID indices for --sweep (default: all)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--allow-gate-seed", action="store_true")
    ap.add_argument("--prereg", default=None)
    a = ap.parse_args(argv)
    seeds = a.seeds if a.sweep else [a.seed]
    _refuse_gate_seeds(seeds, a.allow_gate_seed, a.prereg)
    if a.sweep:
        out_dir = a.out_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        for i in (a.ops if a.ops is not None else range(len(GRID))):
            for s in seeds:
                res = organ_battery(s, i)
                with open(os.path.join(out_dir, "op%d_s%d.json" % (i, s)), "w") as fh:
                    json.dump(res, fh, indent=1, default=str)
                print("op%d seed%d all_pass=%s %s" % (i, s, res["all_pass"],
                                                      {k: v["detect"] for k, v in res["cases"].items()}), flush=True)
        return 0
    res = organ_battery(a.seed, a.op)
    if a.prereg:
        res["prereg"] = a.prereg      # tools/s2p_score.py step O requires the registered prereg path here
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "cases"}, default=str))
    for k, v in res["cases"].items():
        print("  %-16s detect=%-4s pass=%s" % (k, v["detect"], v["pass"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
