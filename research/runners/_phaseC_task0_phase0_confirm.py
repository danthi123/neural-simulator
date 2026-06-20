"""Phase C — TASK 0 (PHASE-0 confirm, no new mechanism): the residual op-level spiking flags drive the
production who/what query path on the REAL `OneBrainComposer`, and the no-confab moat holds.

Per the Phase-C design (`2026-06-19-tier2-phaseC-integrated-loop-design.md` §5 Task 0): before adding any
integration, CONFIRM that the op-level spiking already in `OneBrainComposer` -- the masked RF megakernel
(`enable_rf_cudagraph`, the on-substrate resonate; ON by default) doing the bind/unbind/cleanup, and the
batched on-bridge read -- drives the production query path with the moat intact. This isolates any later
regression to the INTEGRATION (the S5/S6 seams), not the op-level spiking.

The "spiking flags" confirmed here:
  - enable_rf_cudagraph (default True): the resonate runs as ONE masked CUDA megakernel/step (the on-substrate
    fast path), == the per-step masked `_rf_advance_one` loop (the megakernel golden, test_rf_megakernel.py).
  - enable_batched (default True): the on-bridge batched read (all blocks in 3 resonate windows) == the
    per-block oracle (test_onebrain_batched_equals_per_block).
  - the on-bridge cleanup (Re(c) on cp_membrane_potential_v) feeding the read-out (the SAME op the S5 seam
    must couple on-substrate -- this confirms it is decisive at the op level BEFORE the integration consumes it).

GREEN: on a K=2 .. K=3 store, the who/what + yes/no matrix matches ground truth AND the host argmax read-out
(megakernel ON == megakernel OFF, i.e. the op-level spiking path == the loop), and EVERY absent/cross cue
abstains (false-accepts == 0) -- the moat. (The CI guard test_one_brain_composer_agent.py is the formal pin;
this runner is the explicit Phase-0 baseline + a megakernel-on/off identity check.)

GPU-only (the on-bridge parser + the masked megakernel need CuPy); a numpy run is a tiny no-megakernel smoke.

  SIM_BACKEND=cupy python -u -m research.runners._phaseC_task0_phase0_confirm --seeds 42,43,44 --dim 64
"""
from __future__ import annotations

import argparse
import json
import os

from sim.backend import is_gpu_backend
from research.runners.one_brain_composer import OneBrainComposer

# A small fixed store (K=3) + a moat battery. All words in VOCAB.
FACTS = [("dog", "go", "north"), ("cat", "run", "river"), ("bird", "look", "south")]
VOCAB = ["cat", "dog", "fox", "go", "north", "river", "run", "see", "look", "south",
         "tree", "bird", "sun", "moon"]

# present cues (each answers its block) + moat cues (absent agent / absent action / cross = agent0 + action1)
PRESENT = [(("dog", "go"), "north"), (("cat", "run"), "river"), (("bird", "look"), "south")]
ABSENT = [("fox", "go"), ("dog", "see"), ("dog", "run"), ("cat", "look")]


def _query_matrix(c):
    """who/what + yes/no over the present cues; abstain over the moat cues. Returns (rows, present_ok, moat_ok)."""
    rows = []
    present_ok = True
    for (a, x), truth in PRESENT:
        wp = c.query_patient(a, x)
        wa = c.query_agent(x, truth)
        yn = c.ask_yes_no(a, x, truth)
        ok = (wp == truth) and (wa == a) and (yn == "yes")
        present_ok = present_ok and ok
        rows.append(dict(cue=(a, x), kind="present", patient=wp, agent=wa, yes_no=yn, truth=truth, ok=ok))
    false_accepts = 0
    for (a, x) in ABSENT:
        wp = c.query_patient(a, x)
        if wp is not None:
            false_accepts += 1
        rows.append(dict(cue=(a, x), kind="moat", patient=wp, ok=(wp is None)))
    moat_ok = (false_accepts == 0)
    return rows, present_ok, moat_ok, false_accepts


def run_seed(seed, D, gpu):
    """Build the real OneBrainComposer; confirm the production query path + moat with the op-level spiking ON,
    and (on GPU) that the masked megakernel path == the per-step loop path (answer-identical)."""
    out = dict(seed=seed, D=D)
    # megakernel ON (the on-substrate resonate fast path, default) -- the production path
    c_on = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=8, enable_rf_cudagraph=gpu)
    for (a, x, p) in FACTS:
        c_on.store(a, x, p)
    rows_on, present_on, moat_on, fa_on = _query_matrix(c_on)
    out.update(rows_megakernel_on=rows_on, present_ok_on=present_on, moat_ok_on=moat_on, false_accepts_on=fa_on)

    # megakernel OFF (the per-step masked loop) -- must be answer-identical (the op-level spiking is byte-faithful)
    c_off = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=8, enable_rf_cudagraph=False)
    for (a, x, p) in FACTS:
        c_off.store(a, x, p)
    rows_off, present_off, moat_off, fa_off = _query_matrix(c_off)
    # answer-identity between the two op-level-spiking paths (the megakernel == the loop)
    ident = all(r1.get("patient") == r2.get("patient") and r1.get("agent") == r2.get("agent")
                and r1.get("yes_no") == r2.get("yes_no")
                for r1, r2 in zip(rows_on, rows_off))
    out.update(present_ok_off=present_off, moat_ok_off=moat_off, false_accepts_off=fa_off,
               megakernel_eq_loop=ident)
    out["ok"] = bool(present_on and moat_on and present_off and moat_off and ident)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--out", default="research/findings/raw/_phaseC_task0_phase0_confirm.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    gpu = is_gpu_backend()
    if not gpu:
        print("[warn] numpy backend: megakernel falls back to the loop (no-megakernel smoke only)", flush=True)

    results = []
    for s in seeds:
        r = run_seed(s, args.dim, gpu)
        results.append(r)
        tag = "OK" if r["ok"] else "FAIL"
        print(f"seed {s} D{args.dim}: {tag}  present(on/off)={r['present_ok_on']}/{r['present_ok_off']}  "
              f"moat(on/off)={r['moat_ok_on']}/{r['moat_ok_off']} fa={r['false_accepts_on']}/{r['false_accepts_off']}  "
              f"megakernel==loop={r['megakernel_eq_loop']}", flush=True)

    n = len(results)
    ok_n = sum(r["ok"] for r in results)
    moat_n = sum(r["moat_ok_on"] and r["moat_ok_off"] for r in results)
    verdict = "GREEN" if ok_n == n else "FAIL"
    summary = dict(n=n, ok_n=ok_n, moat_n=moat_n, gpu=gpu, verdict=verdict)
    print(f"\nSUMMARY: ok {ok_n}/{n}  moat {moat_n}/{n}  gpu={gpu}  -> {verdict}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=summary, results=results), f, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
