"""The association graph's own derisk gates, run with and without the trait-centred heterogeneity variant
(research/stable-mechc2; dev seeds only, numpy CPU).

LearnedAssocGraph (research/runners/learned_assoc_graph.py) is built on `_D_sparse_heteroassoc.build`, a two-region
bridge whose cells are typed by trait, i.e. exactly the population SIM_TRAIT_CENTRED_HET changes. Its two existing gates:

  G1  learned_assoc_graph.main (the module's own parity harness): the substrate-learned concept graph against the Python
      co-occurrence oracle for 10 concepts and 4 stored facts; validated 2026-06-21 at 24/24 oracle edges recovered and
      9/9 concepts whose top learned associate is a true associate (research/findings/2026-06-21-A13-dialogue-assoc-graph-scoping.md).
      PASS here = every oracle edge recovered AND every concept's top learned associate in the oracle.
  G2  _D_sparse_heteroassoc.run (the D cue-recall resolution, research/findings/2026-06-05-D-cue-recall-RESOLVED-sparse-heteroassoc.md):
      cue -> associate TOP-1 completion, 2 of 2 pairs post-encode and post-SWR on every seed.
      PASS here = 2/2 post-encode AND 2/2 post-SWR.

  G3  (attribution, not a LearnedAssocGraph gate) the Broca frame-slot bridge's own per-seed de-risk,
      _emerge59_spiking_broca_frame_slots_derisk._derisk_one: the census found this trait-typed bridge (build_slot_bridge)
      among the three the variant changes in the brain, and the brain's replies came out with the function word
      misplaced ("the lynx the chases vole"). PASS here = the de-risk's own per-seed bars: held-out order and exact-slot
      rendering >= 0.90, >= 0.20 above the permuted and no-learning controls, function-word slots load-bearing
      (grammatical >= 0.99 with them, <= 0.01 ablated) and 0 producer calls on abstains.

All are run in this process through the modules' own entry points (G1: main() with sys.argv; G2: run() and the
module's own top-1 count; G3: _derisk_one), with the spikes and steps of every bridge step counted. The arm comes from the environment of
the caller: SIM_TRAIT_CENTRED_HET (unset = off) and SIM_STABLE_INTEGRATION; the record stores both.

  SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= .venv/bin/python -m research.runners.stable_mechc_lag_gate \\
      --seeds 1 2 3 4 5 6 7 8 --out research/findings/raw/_stable_mechc/lag_gate/off.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

DEV_SEEDS = tuple(range(1, 16))
RE_EDGES = re.compile(r"learned-graph edges recovered: (\d+)/(\d+)")
RE_TOP = re.compile(r"learned top-associate in oracle: (\d+)/(\d+)")


def parse_g1(text):
    """(edges_recovered, edges_oracle, top_match, top_total) from the module's printed lines; None when absent."""
    e, t = RE_EDGES.search(text), RE_TOP.search(text)
    if not e or not t:
        return None
    return int(e.group(1)), int(e.group(2)), int(t.group(1)), int(t.group(2))


def g1_pass(rec):
    return rec is not None and rec["edges_recovered"] == rec["edges_oracle"] and rec["top_match"] == rec["top_total"]


def g2_pass(rec):
    return rec["post_encode_top1"] == rec["n_pairs"] and rec["post_swr_top1"] == rec["n_pairs"]


class _Counter:
    """Count steps and spikes of every SimulationBridge step while installed (observe-only)."""

    def __init__(self):
        self.steps = 0
        self.spikes = 0
        self._orig = None

    def __enter__(self):
        from sim.backend import to_host
        from sim.bridge import SimulationBridge as SB
        self._cls, self._orig = SB, SB.__dict__["_run_one_simulation_step"]
        counter, orig = self, self._orig

        def wrapped(bridge, *a, **k):
            r = orig(bridge, *a, **k)
            counter.steps += 1
            counter.spikes += int(np.count_nonzero(np.asarray(to_host(bridge.cp_firing_states))))
            return r
        self._cls._run_one_simulation_step = wrapped
        return self

    def __exit__(self, *exc):
        self._cls._run_one_simulation_step = self._orig


def run_g1(seed):
    from research.runners import learned_assoc_graph as L
    buf = io.StringIO()
    old = sys.argv
    sys.argv = ["learned_assoc_graph", "--seed", str(int(seed))]
    t0 = time.time()
    with _Counter() as c, contextlib.redirect_stdout(buf):
        L.main()
    sys.argv = old
    text = buf.getvalue()
    p = parse_g1(text)
    rec = None if p is None else {"edges_recovered": p[0], "edges_oracle": p[1], "top_match": p[2], "top_total": p[3]}
    out = {"gate": "G1 learned_assoc_graph.main", "seed": int(seed), **(rec or {}), "pass": g1_pass(rec),
           "steps": c.steps, "spikes": c.spikes, "wall_s": round(time.time() - t0, 1),
           "printed": [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("[")][:12]}
    return out


def run_g2(seed):
    from research.runners import _D_sparse_heteroassoc as D
    t0 = time.time()
    with _Counter() as c, contextlib.redirect_stdout(io.StringIO()):
        pairs, base, post = D.run(int(seed), swr_cycles=40)
    nb = sum(1 for v in base.values() if v[0])
    npo = sum(1 for v in post.values() if v[0])
    out = {"gate": "G2 _D_sparse_heteroassoc.run", "seed": int(seed), "n_pairs": len(pairs),
           "post_encode_top1": nb, "post_swr_top1": npo,
           "ranks_encode": [base[k][1] for k in pairs], "ranks_swr": [post[k][1] for k in pairs],
           "margins_swr": [round(post[k][2], 4) for k in pairs],
           "steps": c.steps, "spikes": c.spikes, "wall_s": round(time.time() - t0, 1)}
    out["pass"] = g2_pass(out)
    return out


def g3_pass(d):
    return bool(d["main_order"] >= 0.90 and d["main_exact"] >= 0.90 and d["main_order"] >= d["perm_order"] + 0.20
                and d["main_exact"] >= d["perm_exact"] + 0.20 and d["main_order"] >= d["nolearn_order"] + 0.20
                and d["main_grammatical"] >= 0.99 and d["ablate_grammatical"] <= 0.01
                and d["moat_calls_on_abstain"] == 0 and not d["moat_produced_any"] and d["answer_produced"])


def run_g3(seed):
    from research.runners import _emerge59_spiking_broca_frame_slots_derisk as B
    t0 = time.time()
    with _Counter() as c, contextlib.redirect_stdout(io.StringIO()):
        d = B._derisk_one(int(seed))
    keep = ("main_order", "main_exact", "main_word", "perm_order", "perm_exact", "nolearn_order", "cross_order",
            "main_grammatical", "ablate_grammatical", "moat_calls_on_abstain", "moat_produced_any", "answer_produced")
    out = {"gate": "G3 _emerge59 Broca frame-slot derisk_one", "seed": int(seed), **{k: d[k] for k in keep},
           "steps": c.steps, "spikes": c.spikes, "wall_s": round(time.time() - t0, 1)}
    out["pass"] = g3_pass(d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--gates", nargs="+", default=["g1", "g2"], choices=["g1", "g2", "g3"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bad = [s for s in a.seeds if s not in DEV_SEEDS]
    if bad:
        raise SystemExit("dev seeds 1-15 only; refused: %s" % bad)
    env = {k: os.environ.get(k) for k in ("SIM_TRAIT_CENTRED_HET", "SIM_PRESET_CENTRED_HET", "SIM_STABLE_INTEGRATION")}
    rows = []
    for s in a.seeds:
        for g in a.gates:
            r = {"g1": run_g1, "g2": run_g2, "g3": run_g3}[g](s)
            rows.append(r)
            print("seed %d %s pass=%s %s" % (s, r["gate"], r["pass"], {k: r[k] for k in r if k in (
                "edges_recovered", "edges_oracle", "top_match", "top_total", "post_encode_top1", "post_swr_top1",
                "main_order", "main_exact", "main_grammatical", "spikes", "wall_s")}), flush=True)
    res = {"runner": "research/runners/stable_mechc_lag_gate.py", "env": env, "seeds": a.seeds, "rows": rows,
           "n_g1_pass": sum(1 for r in rows if r["gate"].startswith("G1") and r["pass"]),
           "n_g1": sum(1 for r in rows if r["gate"].startswith("G1")),
           "n_g2_pass": sum(1 for r in rows if r["gate"].startswith("G2") and r["pass"]),
           "n_g2": sum(1 for r in rows if r["gate"].startswith("G2")),
           "n_g3_pass": sum(1 for r in rows if r["gate"].startswith("G3") and r["pass"]),
           "n_g3": sum(1 for r in rows if r["gate"].startswith("G3")),
           "spikes_total": sum(r["spikes"] for r in rows)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("G1 %d/%d  G2 %d/%d  G3 %d/%d -> %s" % (res["n_g1_pass"], res["n_g1"], res["n_g2_pass"], res["n_g2"],
                                                  res["n_g3_pass"], res["n_g3"], a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
