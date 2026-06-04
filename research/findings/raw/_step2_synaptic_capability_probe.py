"""One-bridge unification — Step 2, Task 3: the SYNAPTIC-ROUTE no-regression gate.

Question: at PRODUCTION scale (proj_dim=2048) and with the REAL substrate concept codes (`denoise64`
V=16), does the SYNAPTIC route (`UnifiedBrainBridge.hear_synaptic` — the parser's role-ensemble firing
opens transmission gates that route a word's concept code into the composer's role bank, so comprehension
routes composition IN SPIKES) reproduce the Python parse+store hand-off path's FLAT subject-verb-object
(SVO) recall? This builds ONE unified bridge per seed and compares the two paths on the SAME bridge.

Terms (defined once — owner standing requirement, no undefined acronyms):
  * bridge          = one `sim.bridge.SimulationBridge` (a network of simulated Izhikevich neurons).
  * SYNAPTIC path   = comprehend→store via `u.hear_synaptic("agent action patient")`: the parser's spiking
    role assignment opens a transmission gate per role, routing that role's ±1 pattern into the composer's
    role bank while the word's concept code drives the fill bank. The cross-region hand-off is synaptic.
  * PYTHON path / ORACLE = `roles = u.parse(...); u.store(roles["agent"], roles["action"], roles["patient"])`
    — the parser returns a `{role: word}` dict that Python passes to the composer's store. The regression
    oracle (`BrainConversationalAgent.hear` uses exactly this), run on the SAME bridge.
  * FLAT SVO        = a 3-word fact `store(a, ac, p)`; recall = `query_patient(a, ac) == p` (the "what")
    AND `query_agent(ac, p) == a` (the "who"). The synaptic route ONLY affects this category — attribute /
    clause / negation facts are stored structurally via `composer.store` and are unchanged by this route,
    so the FLAT SVO category IS the gate.
  * regression      = the SYNAPTIC recall (who or what) drops by MORE THAN 1 trial below the PYTHON path's
    recall on any seed (±1 trial is the spiking-noise tolerance — the gate's EMA warm-up + the role-route
    operating point can shift a hair vs the direct role current the Python path drives).

Why the SAME bridge for both paths: the unified bridge built with `enable_synaptic_route=True` wires the
parser→gate→composer routes AND keeps the delegated `parse`/`store` (Python path) fully working. Running
both paths on one bridge holds the OU background noise, heterogeneity, and operating point identical between
them — the cleanest A/B (any per-seed difference is the route, not a different bridge). Each fact clears the
composer kb first (mirroring `_decorrelate_v16_probe.run_matrix`), stores one fact, queries, so the two
paths never see each other's stored vectors.

GPU NOTE: this is a heavy SPIKING probe. It runs on the validated production (CuPy/GPU) backend, NOT NumPy
(the parser's Hebbian convergence + the gate operating point are GPU-validated; NumPy diverges). Do NOT set
SIM_BACKEND=numpy. Each seed builds a ~16.5K-neuron unified bridge (parser 126 + composer 8*2048 + the
3*2048 role-src route pools) and trains the parser, then runs both paths over N flat facts — expect tens of
minutes per seed; that is expected for a faithful production-scale spiking run.

    python -m research.findings.raw._step2_synaptic_capability_probe --proj-dim 2048 --seeds 42 43 44 \
        --out research/findings/raw/_step2_synaptic_capability_probe.json

The function `run_synaptic_comparison(seeds, proj_dim, n)` returns a structured result the Task-3 test
(`tests/test_unified_brain_bridge.py::test_step2_synaptic_no_regression`) consumes to assert no regression.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from research.runners.unified_brain_bridge import UnifiedBrainBridge


def _flat_recall_synaptic(u, facts):
    """Run the SYNAPTIC route over `facts` (list of (a, ac, p)). Per fact: clear the composer kb, store via
    `u.hear_synaptic("a ac p")` (the parser-gated route), then count who/what recall. Returns
    (what_ok, who_ok) — how many of the N facts recalled the patient / agent correctly."""
    what_ok = who_ok = 0
    for a, ac, p in facts:
        u.kb = []                                  # one fact in memory at a time (mirrors run_matrix)
        u.hear_synaptic(f"{a} {ac} {p}")           # comprehend→store via the gated synaptic route
        what_ok += int(u.query_patient(a, ac) == p)
        who_ok += int(u.query_agent(ac, p) == a)
    return what_ok, who_ok


def _flat_recall_python(u, facts):
    """Run the PYTHON hand-off path (the regression ORACLE) over the SAME `facts` on the SAME bridge. Per
    fact: clear the composer kb, comprehend with `u.parse` → `{role: word}` dict, hand it to `u.store`
    (Python), then count who/what recall. Returns (what_ok, who_ok)."""
    what_ok = who_ok = 0
    for a, ac, p in facts:
        u.kb = []
        roles = u.parse(f"{a} {ac} {p}")           # Python comprehension → {role: word} dict
        u.store(roles["agent"], roles["action"], roles["patient"])   # Python hand-off store
        what_ok += int(u.query_patient(a, ac) == p)
        who_ok += int(u.query_agent(ac, p) == a)
    return what_ok, who_ok


def _abstention_holds(u, facts):
    """Abstention (the no-confab moat) must survive the synaptic route: store ONE flat fact via
    `hear_synaptic`, then query a cue whose (agent, action) is NOT the stored one — it must return None.
    Returns True if the unstored cue abstained. Uses two distinct vocabulary words not in the stored fact."""
    a, ac, p = facts[0]
    u.kb = []
    u.hear_synaptic(f"{a} {ac} {p}")
    # An (agent, action) cue guaranteed different from the stored (a, ac): pick two words != a, ac, p.
    others = [w for w in u.words if w not in (a, ac, p)]
    cue_a, cue_ac = others[0], others[1]
    return u.query_patient(cue_a, cue_ac) is None


def run_synaptic_comparison(seeds=(42, 43, 44), proj_dim=2048, n=6):
    """For each seed: build ONE `UnifiedBrainBridge(enable_synaptic_route=True)` and, on it, run N random
    distinct 3-word FLAT SVO facts through BOTH paths — the SYNAPTIC route (`hear_synaptic`) and the PYTHON
    hand-off (`parse`+`store`) — plus one abstention check. The word stream is seeded by seed+1 (matching
    `_decorrelate_v16_probe.main`'s convention) so the facts are deterministic per seed. Returns a dict:

        {seed: {"synaptic": {"what": (ok, n), "who": (ok, n)},
                "python":   {"what": (ok, n), "who": (ok, n)},
                "abstention": bool,
                "parser": {"active": {...}, "passive_agent": "dog"},
                "elapsed_s": float}}

    Raises FileNotFoundError if a seed's denoise64 cache is missing (the caller/test decides to skip)."""
    results = {}
    for seed in seeds:
        t0 = time.time()
        # ONE unified bridge with the synaptic route wired; default concepts -> the REAL denoise64 V=16 codes.
        u = UnifiedBrainBridge(seed=seed, proj_dim=proj_dim, enable_synaptic_route=True)

        # Confirm the parser still parses voice-invariantly on this merged production bridge (the synaptic
        # route is wired BEFORE training; this guards against the route perturbing the parser's convergence).
        active = u.parse("dog go north", voice="active")
        passive_agent = u.parse("north go dog", voice="passive").get("agent")

        # Deterministic per-seed flat-fact stream (3 distinct words each), identical for both paths.
        rng = np.random.default_rng(seed + 1)
        facts = [tuple(str(x) for x in rng.choice(u.words, size=3, replace=False)) for _ in range(n)]

        syn_what, syn_who = _flat_recall_synaptic(u, facts)
        py_what, py_who = _flat_recall_python(u, facts)
        abst = _abstention_holds(u, facts)

        results[seed] = {
            "synaptic": {"what": (syn_what, n), "who": (syn_who, n)},
            "python": {"what": (py_what, n), "who": (py_who, n)},
            "abstention": bool(abst),
            "parser": {"active": active, "passive_agent": passive_agent},
            "facts": facts,
            "elapsed_s": round(time.time() - t0, 1),
        }
        print(f"[syn-probe] seed {seed}: synaptic what={syn_what}/{n} who={syn_who}/{n} | "
              f"python what={py_what}/{n} who={py_who}/{n} | abstain={abst} | "
              f"{results[seed]['elapsed_s']}s", flush=True)
    return results


def format_table(results):
    """A human-readable per-seed table (synaptic vs python, who + what) + the parser-on-merged-bridge line."""
    lines = []
    header = (f"{'seed':>5}  {'metric':<6}  {'synaptic':>9}  {'python':>9}  {'delta':>6}  {'verdict'}")
    lines.append(header)
    lines.append("-" * len(header))
    for seed in sorted(results):
        r = results[seed]
        for metric in ("what", "who"):
            so, st = r["synaptic"][metric]
            po, pt = r["python"][metric]
            delta = so - po
            verdict = "REGRESSION" if delta < -1 else "ok"
            lines.append(
                f"{seed:>5}  {metric:<6}  {so:>5}/{st:<3}  {po:>5}/{pt:<3}  {delta:>+6}  {verdict}")
        lines.append(
            f"{seed:>5}  abstain={r['abstention']}  "
            f"parser active={r['parser']['active']}  passive_agent={r['parser']['passive_agent']!r}  "
            f"({r['elapsed_s']}s)")
        lines.append("-" * len(header))
    return "\n".join(lines)


def find_regressions(results):
    """Return the list of (seed, metric, python_ok, synaptic_ok) where the SYNAPTIC route dropped MORE THAN
    1 trial below the PYTHON path (the no-regression gate). Empty list == PASS."""
    regressions = []
    for seed in sorted(results):
        for metric in ("what", "who"):
            po = results[seed]["python"][metric][0]
            so = results[seed]["synaptic"][metric][0]
            if so - po < -1:
                regressions.append((seed, metric, po, so))
    return regressions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--proj-dim", type=int, default=2048)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", type=str, default=None, help="optional JSON dump of the structured results")
    args = ap.parse_args()
    try:
        results = run_synaptic_comparison(tuple(args.seeds), args.proj_dim, args.n)
    except FileNotFoundError as e:
        print(f"[syn-probe] denoise64 cache missing -> skip: {e}", flush=True)
        return
    print(format_table(results), flush=True)

    regressions = find_regressions(results)
    if regressions:
        print("\n[syn-probe] REGRESSIONS (synaptic more than 1 trial below python):", flush=True)
        for seed, metric, po, so in regressions:
            print(f"  seed {seed} {metric}: python {po} -> synaptic {so} (drop {po - so})", flush=True)
    else:
        print("\n[syn-probe] NO REGRESSION: every metric (who/what), every seed, within +-1 trial of "
              "the Python hand-off path. Abstention preserved on all seeds: "
              f"{all(results[s]['abstention'] for s in results)}.", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        print(f"[syn-probe] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
