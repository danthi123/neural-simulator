"""Step-3 integration cheap-first smoke — does a LIVE cortex_it rate read OFF THE MERGED BRIDGE still COMPOSE?

The grounded-code→production-composer drop-in is ALREADY GO (D=2048, 6-seed) — but on a SEPARATE `build_probe_bridge`,
in isolation. The integration build's ONE unproven variable (per the controller-verified scoping
`2026-06-16-step3-integration-build-scoping.md` §3): reading the grounded rate from the MERGED bridge's `cortex_it`
slice — co-resident with the navigation cascade + parser + dlPFC + `rf` all firing under OU + tonic drive — and
composing on the merged `rf` slice. If the co-resident read degrades the grounded code below the compose bar, the
documented fix is a population-code lift / more read steps. This smoke isolates exactly that, CPU/numpy, BEFORE any
GPU build. It is the standing cheap-first-before-the-GPU-build move; the GPU behavioral integration build is gated.

What it does (numpy, CPU, tiny D + 4 objects, 3 seeds):
 1. Build the merged bridge WITH the new bare-`cortex_it` perception region: `build_merged_nav_conv_bridge(
    co_resident_rf=True, co_resident_perception=True)`. T0 BYTE-IDENTITY GATE: assert `cortex_it` is appended AFTER
    `rf` (its indices are the highest), so the navigation/parser/dlPFC/`rf` bases are byte-unchanged.
 2. For each object: read its LIVE `cortex_it` rate ON THE MERGED bridge (reuse the de-risk's `read_cortex_it_rate`
    against `rm.indices("cortex_it")`), map -> phases via the fixed `_projection`/`grounded_phases`, set
    `composer.concepts[o] = phases_o` on the co-resident `MergedRFComposer`.
 3. Split (agent, patient) distinct-object pairs into MEMORIZED vs HELD-OUT (the cheap-first probe anti-cheat). For
    each HELD-OUT fact: compose `_encode({"agent":a,"patient":b})` on the merged `rf` slice, `unbind` each role,
    cleanup -> the perceived object. Score a memorization-floor recall baseline on the SAME held-out facts.
 4. The no-confab MOAT: `composer.store`/`query_patient` of an unstored (agent, action) returns None.

GATE (mirrors the cheap-first probe + the production-composer de-risk, on the MERGED substrate; 3 seeds):
  GO       : held-out clean compose >= 0.90 AND held-out >= memorization-floor + 0.30 AND the moat abstains.
  NO-GO    : held-out collapses toward chance / the floor -> the merged-substrate cortex_it read is noisier than the
             standalone probe (localize to a population-code lift / more read steps — the documented fix).
  MOAT_BREACH (HARD STOP): an unstored query is accepted -> never weaken the moat.

  SIM_BACKEND=numpy python -m research.runners._step3_merged_grounded_compose_smoke
No `sim/` edit (the bare-cortex_it region is an additive default-off kwarg on the runner builder; reuse-by-import).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from research.runners.nav_conv_merged_bridge import build_merged_nav_conv_bridge, MergedRFComposer
from research.runners.funcint_perception_to_memory_probe import OBJECT_WORDS, N_OBJECTS
from research.runners._step3_grounded_codes_production_composer_derisk import (
    read_cortex_it_rate, _projection, grounded_phases,
)

D = 64                       # tiny composer dim (the smoke isolates the merged-bridge read, not D); rf_D matches.
ACTIONS = ["chase", "near"]  # native composer codes (verbs are not perceived) — for the moat store/query.
ABSENT = [("river", "chase"), ("apple", "near")]   # unstored (agent, action) -> the moat must abstain.


def run_seed(seed):
    vocab = list(OBJECT_WORDS) + ACTIONS
    bridge, handles = build_merged_nav_conv_bridge(
        seed=seed, vocab=vocab, n_cortex=100, co_resident_rf=True, rf_D=D, co_resident_perception=True)
    rm = bridge.region_manager
    it_indices = np.asarray(rm.indices("cortex_it"))   # rm.indices returns a list; the de-risk read needs an array
    rf_idx = np.asarray(rm.indices("rf"))
    N = int(bridge.core_config.num_neurons)

    # T0 BYTE-IDENTITY GATE: cortex_it is appended AFTER rf (highest indices) so nav/parser/dlPFC/rf bases unchanged.
    assert int(it_indices[0]) == int(rf_idx[-1]) + 1, "cortex_it must be appended immediately after rf"
    assert int(it_indices[-1]) == N - 1, "cortex_it must be the LAST region (byte-identity: no base shifts)"
    assert int(handles["rf_base"]) == int(rf_idx[0]), "rf_base must be unchanged (by-name index)"

    composer = MergedRFComposer(bridge, handles["rf_base"], handles["rf_size"], seed=seed, D=D, vocab=vocab, period=200)

    # GROUND each object's LIVE merged-bridge cortex_it rate into the co-resident composer codebook. Enable OU during
    # the read (the resting nav config has OU off) so the read reflects the ACTUAL episode's noisy co-resident
    # condition — the scoping's stated #1 risk (the co-resident read under OU + the whole stack).
    cc = bridge.core_config
    prev_ou, prev_std = cc.enable_ou_process, cc.ou_std_current_pA
    cc.enable_ou_process, cc.ou_std_current_pA = True, 20.0
    proj = _projection(D, int(it_indices.size), seed)
    try:
        for i, name in enumerate(OBJECT_WORDS):
            rate = read_cortex_it_rate(bridge, it_indices, i)            # live rate read OFF THE MERGED BRIDGE (OU on)
            composer.concepts[name] = grounded_phases(rate, proj)       # phases in [0,1)^D (the grounded code)
    finally:
        cc.enable_ou_process, cc.ou_std_current_pA = prev_ou, prev_std

    # held-out vs memorized split (the compose-vs-recall anti-cheat).
    names = list(OBJECT_WORDS)
    pairs = [(a, b) for a in range(N_OBJECTS) for b in range(N_OBJECTS) if a != b]
    rng = np.random.default_rng(seed * 99 + 7)
    rng.shuffle(pairs)
    memorized, held_out = pairs[:len(pairs) // 2], pairs[len(pairs) // 2:]

    # the memorization floor (a recall-only baseline): nearest stored composite -> its remembered filler.
    mem_store = [(composer._encode({"agent": names[ai], "patient": names[bi]}), ai, bi) for (ai, bi) in memorized]

    def _mem_recall(comp, role):
        best, bk = -1.0, 0
        for k, (f, a, b) in enumerate(mem_store):
            c = float(np.mean(np.cos(2.0 * np.pi * (f - comp))))         # composer's phase-cosine similarity
            if c > best:
                best, bk = c, k
        return mem_store[bk][1] if role == "agent" else mem_store[bk][2]

    clean_ok = mem_ok = 0
    for (ai, bi) in held_out:
        comp = composer._encode({"agent": names[ai], "patient": names[bi]})   # COMPOSE on the merged rf slice
        ra = composer.unbind(comp, "agent")
        rb = composer.unbind(comp, "patient")
        clean_ok += int(ra == names[ai]) + int(rb == names[bi])
        mem_ok += int(names[_mem_recall(comp, "agent")] == names[ai]) + int(names[_mem_recall(comp, "patient")] == names[bi])
    clean = clean_ok / (2 * len(held_out))
    floor = mem_ok / (2 * len(held_out))

    # the no-confab MOAT: store a couple facts, query an UNSTORED (agent, action) -> must abstain (None).
    composer.store(names[0], ACTIONS[0], names[1])      # apple chase river
    composer.store(names[2], ACTIONS[1], names[3])      # dog near cat
    moat_ok = sum(int(composer.query_patient(a, v) is None) for (a, v) in ABSENT)
    # and a positive control: a stored fact DOES retrieve (so the moat isn't trivially abstaining on everything).
    pos = int(composer.query_patient(names[0], ACTIONS[0]) == names[1])

    chance = 1.0 / N_OBJECTS
    go = (clean >= 0.90 and clean >= floor + 0.30 and moat_ok == len(ABSENT) and pos == 1)
    breach = (moat_ok < len(ABSENT))
    print(f"  [seed {seed}] merged-bridge held-out compose clean {clean:.3f} | mem-floor {floor:.3f} "
          f"(chance {chance:.3f}) | moat-abstain {moat_ok}/{len(ABSENT)} | pos-recall {pos}/1  "
          f"[{'GO' if go else ('MOAT_BREACH' if breach else 'NO-GO')}]", flush=True)
    return {"seed": seed, "clean": clean, "floor": floor, "chance": chance, "moat_ok": moat_ok,
            "moat_tot": len(ABSENT), "pos": pos, "go": go, "breach": breach,
            "n_neurons": N, "it_base": int(it_indices[0]), "rf_base": int(rf_idx[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=str, default="research/findings/raw/_step3_merged_grounded_compose_smoke.json")
    args = ap.parse_args()
    print("[step3-merged-smoke] does a LIVE cortex_it rate read OFF THE MERGED BRIDGE (co-resident w/ nav+parser+"
          "dlPFC+rf) still COMPOSE perceived objects into novel facts?", flush=True)
    rows = [run_seed(s) for s in args.seeds]
    breach = any(r["breach"] for r in rows)
    all_go = all(r["go"] for r in rows)
    clean = float(np.mean([r["clean"] for r in rows]))
    floor = float(np.mean([r["floor"] for r in rows]))
    verdict = "MOAT_BREACH" if breach else ("GO" if all_go else "NO-GO")
    print(f"\n{'='*98}\n  {len(rows)} seeds: held-out compose clean {clean:.3f} | mem-floor {floor:.3f} "
          f"==> {verdict}", flush=True)
    if verdict == "GO":
        print("  GO: the live cortex_it rate read OFF THE MERGED BRIDGE grounds codes that COMPOSE on the co-resident "
              "rf slice — held-out (never-composed) perceived-object facts recover >> the memorization floor, the "
              "no-confab moat abstains, and a stored fact still retrieves. ==> the integration build's one unproven "
              "variable (the co-resident read) is RETIRED; the GPU behavioral build (T3+) is de-risked (owner-gated).",
              flush=True)
    elif verdict == "MOAT_BREACH":
        print("  MOAT_BREACH (HARD STOP): an unstored query was accepted on the merged bridge — investigate before "
              "any build; never weaken the moat.", flush=True)
    else:
        print("  NO-GO: the co-resident merged-bridge cortex_it read is too noisy to ground a composable code "
              "(held-out ~ floor/chance) — localize to a population-code lift / more read steps (the documented fix) "
              "BEFORE any GPU spend. Honest negative that saves the GPU build.", flush=True)
    print(f"{'='*98}", flush=True)
    with open(args.out, "w") as f:
        json.dump({"verdict": verdict, "mean_clean": clean, "mean_floor": floor, "per_seed": rows}, f, indent=2,
                  default=str)
    print(f"  [saved] {args.out}", flush=True)
    raise SystemExit(0 if verdict == "GO" else 1)


if __name__ == "__main__":
    main()
