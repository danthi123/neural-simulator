"""D6 multi-referent WM -- EMERGENT free-slot-wins competitive register allocation (2026-09-01).

FRONTIER (named open residual, D6 finding 2026-08-12 + onebrain-xedge 2026-08-28): "the register assignment is
today a role-by-position host MARKER (referent 0 -> reg0, ...)" / "the SEMANTIC referent->pool binding... remains
a DECLARED residual". This verifies the fix built in `d6_multiref_wm_production_organ.py`
(`BRAIN_MULTIREF_COMPETITIVE=1`): register allocation is now routed by a genuine `MultiSlotHold.probe_occupancy()`
read (zero-input `cp_firing_states`, argmin over per-register band-max rate) instead of the referent's position
in the sentence/call. NO NEW spiking mechanism was invented for the HOLD or the BIND (both reuse the banked-GO
`MultiSlotHold`/`HebbianBinder` exactly as the D6 organ already did); the ONLY change is WHICH register a write
targets, and that decision is now made by reading the substrate's own state.

THE TEST (per seed, matching the task's own anti-cheat spec):
  (A) TOGETHER -- k=2 (then k=3) referents introduced in ONE `load()` call: verify they land in DISTINCT registers
      (the brain's own occupancy read separates them, not a host index) and are all recovered.
  (B) ORDER-INVARIANCE -- the SAME k=2 referents, introduced in the OPPOSITE order: still land in DISTINCT
      registers, still both recovered (the separation does not depend on which referent is mentioned first).
  (C) ANCHOR (the non-trivial case) -- ONE referent is pre-loaded (occupies a register), THEN two NEW referents
      are introduced (in both orders) WITHOUT an intervening reset, via the SAME write/hold/probe/read path
      `load()` uses (driven directly on the organ's own `buf` -- reused, not reimplemented): confirms the new
      referents avoid the ALREADY-OCCUPIED anchor register too, a claim a pure host-position marker cannot make
      because it never reads occupancy at all. All THREE referents must remain recoverable.
  LESION -- `competition_lesion=True` ablates ONLY the SELECTION (every write targets register 0, ignoring the
      probe); the HOLD's own recurrence is untouched (that lesion is `BRAIN_MULTIREF_LESION`, tested elsewhere by
      the D6 GO). Repeats (A) and (C) and reports the COLLISION this produces: >=2 referents share one register's
      local competition, so the read-back keeps only the local WTA winner -- exactly the SUPERPOSED-collide
      regime `_multi_slot_binding_derisk.eval_superposed_single` already validated as load-bearing separation,
      reproduced here as a genuine collision rather than assumed.
  BYTE-IDENTICAL-OFF -- `competitive=False` (the untouched default) must reproduce the EXACT pre-existing
      role-by-position registers ([0, 1, 2, ...]), confirming the flip is additive and default-OFF-safe.

Reuse-by-import (`MultiReferentWMOrgan`, `MultiSlotHold` via the organ); NO sim/ edit. SIM_BACKEND=numpy
(sub-1k-neuron D3/D6 loops are launch-bound: CPU faster, matching the parent de-risk's own convention).

Run (1-seed smoke, FOREGROUND):
  SIM_BACKEND=numpy python -m research.runners._d6_wm_competitive_slot_binding_verify --seeds 42
6-seed decisive:
  SIM_BACKEND=numpy python -m research.runners._d6_wm_competitive_slot_binding_verify \
    --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/_d6_wm_competitive_slot_binding/verify_6seed.json
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from research.runners.d6_multiref_wm_production_organ import MultiReferentWMOrgan

try:
    from tools.lab import lever, void_if
except Exception:  # tools.lab optional at import time; the runner still runs
    def lever(name, before, after, required=True, continuous=None):
        print(f"  LEVER {name}: {before} -> {after}"); return before != after
    def void_if(cond, reason):
        if cond: print(f"  VOID: {reason}")
        return bool(cond)

OUT = Path("research/findings/raw/_d6_wm_competitive_slot_binding/verify.json")


def _fresh_organ(seed):
    return MultiReferentWMOrgan(seed=seed)


def scenario_together(seed, refs, competitive, competition_lesion):
    """(A)/(LESION): k referents introduced in ONE `load()` call (the real production path, unmodified)."""
    o = _fresh_organ(seed)
    return o.load(list(refs), competitive=competitive, competition_lesion=competition_lesion)


def scenario_anchor(seed, anchor, new_refs, competitive, competition_lesion):
    """(C)/(LESION-C): pre-load `anchor` alone (occupies a register), THEN write the new referents WITHOUT an
    intervening reset -- so the anchor's bump is still alive (real occupancy) when the new referents are
    allocated. `load()` itself resets the bridge at entry (each production turn re-materializes the FULL held
    set from `_slot_of_ref`), so this drives the organ's OWN buffer directly through the identical
    write/hold/probe/read calls `load()` uses, to isolate the cross-write occupancy effect without re-deriving
    the write/read protocol."""
    o = _fresh_organ(seed)
    o.ensure_built()
    buf = o.buf
    buf.reset()
    all_refs = [anchor] + list(new_refs)
    locals_ = [o._local_slot(r) for r in all_refs]
    registers = []
    for i, loc in enumerate(locals_):
        if competitive:
            reg = 0 if competition_lesion else int(np.argmin(buf.probe_occupancy()))
        else:
            reg = i
        registers.append(reg)
        buf.write(reg, loc)
        buf.hold()
    buf.hold(); buf.hold()
    recovered = {}
    alive = []
    for i, reg in enumerate(registers):
        loc, amp = buf.read(reg)
        alive.append(float(amp))
        recovered[i] = o._ref_of_slot.get(loc, None)
    return {
        "input_order": all_refs, "registers": registers, "recovered": recovered,
        "all_recovered": bool(all(recovered.get(i) == all_refs[i] for i in range(len(all_refs)))),
        "distinct_registers": bool(len(set(registers)) == len(registers)),
        "hold_alive_min": float(min(alive)) if alive else 0.0,
    }


def run_seed(seed):
    out = {"seed": seed}

    # (A) TOGETHER k=2, both mention orders, intact competitive
    ab = scenario_together(seed, ["dog", "cat"], True, False)
    ba = scenario_together(seed, ["cat", "dog"], True, False)
    out["together_AB"] = ab; out["together_BA"] = ba
    out["together_k2_distinct_both_orders"] = bool(ab["distinct_registers"] and ba["distinct_registers"])
    out["together_k2_recovered_both_orders"] = bool(ab["all_recovered"] and ba["all_recovered"])
    out["together_k2_intact_collides"] = bool(not (out["together_k2_distinct_both_orders"]
                                                     and out["together_k2_recovered_both_orders"]))

    # (A) TOGETHER k=3
    k3 = scenario_together(seed, ["dog", "cat", "bird"], True, False)
    out["together_k3"] = k3
    out["together_k3_ok"] = bool(k3["distinct_registers"] and k3["all_recovered"])

    # (C) ANCHOR: a pre-occupied register, then two NEW referents, both mention orders
    anc_ab = scenario_anchor(seed, "horse", ["dog", "cat"], True, False)
    anc_ba = scenario_anchor(seed, "horse", ["cat", "dog"], True, False)
    out["anchor_AB"] = anc_ab; out["anchor_BA"] = anc_ba
    out["anchor_avoids_occupied_both_orders"] = bool(
        anc_ab["registers"][0] not in anc_ab["registers"][1:] and
        anc_ba["registers"][0] not in anc_ba["registers"][1:])
    out["anchor_all_three_recovered_both_orders"] = bool(anc_ab["all_recovered"] and anc_ba["all_recovered"])
    out["anchor_all_three_distinct_both_orders"] = bool(anc_ab["distinct_registers"] and anc_ba["distinct_registers"])

    # LESION: selection-only ablation (register 0 always) -- HOLD recurrence untouched
    les_together = scenario_together(seed, ["dog", "cat"], True, True)
    les_anchor = scenario_anchor(seed, "horse", ["dog", "cat"], True, True)
    out["lesion_together"] = les_together; out["lesion_anchor"] = les_anchor
    out["lesion_together_collides"] = bool((not les_together["distinct_registers"])
                                            or (not les_together["all_recovered"]))
    out["lesion_anchor_collides"] = bool((not les_anchor["distinct_registers"])
                                          or (not les_anchor["all_recovered"]))

    # BYTE-IDENTICAL-OFF: competitive=False reproduces the exact pre-existing positional registers
    off = scenario_together(seed, ["dog", "cat", "bird"], False, False)
    out["byte_identical_off"] = bool(off["registers"] == [0, 1, 2] and off["all_recovered"])

    out["seed_pass"] = bool(
        out["together_k2_distinct_both_orders"] and out["together_k2_recovered_both_orders"]
        and out["together_k3_ok"]
        and out["anchor_avoids_occupied_both_orders"] and out["anchor_all_three_recovered_both_orders"]
        and out["anchor_all_three_distinct_both_orders"]
        and out["lesion_together_collides"] and out["lesion_anchor_collides"]
        and out["byte_identical_off"]
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    smoke = len(a.seeds) < 6
    backend = os.environ.get("SIM_BACKEND", "numpy"); device = "cpu" if backend == "numpy" else "gpu"
    print(f"backend={backend} device={device} seeds={a.seeds} smoke={smoke}", flush=True)

    t0 = time.time(); err = None; per_seed = []
    try:
        for s in a.seeds:
            r = run_seed(s)
            per_seed.append(r)
            print(f"  seed={s} together_k2(distinct={r['together_k2_distinct_both_orders']} "
                  f"recov={r['together_k2_recovered_both_orders']}) k3_ok={r['together_k3_ok']} "
                  f"anchor(avoids={r['anchor_avoids_occupied_both_orders']} "
                  f"recov3={r['anchor_all_three_recovered_both_orders']}) "
                  f"LESION(together_collides={r['lesion_together_collides']} "
                  f"anchor_collides={r['lesion_anchor_collides']}) "
                  f"byte_id_off={r['byte_identical_off']} || seed_pass={r['seed_pass']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    verdict = None
    core = False
    if err is None and per_seed:
        n_pass = sum(1 for r in per_seed if r["seed_pass"])
        core = (n_pass == len(per_seed))
        void_if(len(per_seed) == 0, "no seeds ran")
        lever("collision count: LESIONed selection (register 0 forced) vs INTACT competitive (occupancy-read)",
              sum(1 for r in per_seed if r["lesion_together_collides"]),
              sum(1 for r in per_seed if r["together_k2_intact_collides"]), required=False)
        go = bool(core and not smoke)
        tag = "GO" if go else ("SMOKE-GO (1-seed indicator; run the 6-seed sweep)" if core else "HONEST-NEGATIVE")
        if core:
            verdict = (f"{tag} -- EMERGENT free-slot-wins register allocation ({n_pass}/{len(per_seed)} seeds "
                       f"pass): >=2 referents introduced together land in DISTINCT registers via a genuine "
                       f"MultiSlotHold.probe_occupancy() argmin read (not a host loop index), invariant to "
                       f"mention order; a referent introduced AFTER an already-held (anchor) referent correctly "
                       f"avoids the anchor's OCCUPIED register too (impossible for a pure position marker, which "
                       f"never reads occupancy); the selection-only LESION (register 0 forced regardless of the "
                       f"probe) COLLAPSES the separation into the already-validated SUPERPOSED-collide regime on "
                       f"every seed (load-bearing); competitive=False stays byte-identical to the pre-existing "
                       f"role-by-position path. Closes the D6/onebrain-xedge declared residual: register "
                       f"assignment is no longer a position-only host MARKER.")
        else:
            miss = [r["seed"] for r in per_seed if not r["seed_pass"]]
            verdict = f"HONEST-NEGATIVE -- {len(miss)}/{len(per_seed)} seeds failed (seeds {miss}); see per-seed detail."
    elif err is not None:
        verdict = f"ERROR -- {err}"
    else:
        verdict = "ERROR -- no seeds ran"

    summary = {
        "probe": "d6_wm_competitive_slot_binding", "verdict": verdict, "backend": backend, "sim_backend": backend,
        "device": device, "smoke": smoke, "cost_acknowledged": True, "seeds": a.seeds,
        "mechanism": "MultiSlotHold.probe_occupancy() (a genuine zero-input cp_firing_states read over every "
                     "register's band-max rate) replaces the role-by-position write marker; the referent is "
                     "routed to argmin(occupancy) -- the register the substrate itself currently shows as free "
                     "-- instead of its position in the sentence/call. Additive; BRAIN_MULTIREF_COMPETITIVE "
                     "default-OFF; BRAIN_MULTIREF_COMPETITION_LESION ablates ONLY the selection (forces "
                     "register 0), independent of the pre-existing BRAIN_MULTIREF_LESION (recur=0, the HOLD "
                     "itself).",
        "per_seed": per_seed, "elapsed_seconds": round(time.time() - t0, 1),
        "HONEST_NOTE": "reuse-by-import of the banked-GO MultiSlotHold (D3 slow-NMDA hold, ONE shared FS) + "
                       "RUNG6c HebbianBinder via the D6 production organ; NO sim/ edit, no new spiking "
                       "mechanism. UNCHANGED residuals (declared, not closed by this): referent EXTRACTION "
                       "(which tokens are referents) stays a host lexicon parse; the referent<->LOCAL-slot BIND "
                       "stays the host-numpy RUNG6c binder; the register READ stays a host argmax over firing "
                       "rate (the same read-out-instrument class used throughout this codebase's honest "
                       "functional read-outs, e.g. affect/comprehension/metacog). What moved: WHICH REGISTER "
                       "a referent lands in is now decided by a substrate READ, not a host loop index. This "
                       "substrate has no background OU noise (ou_std_current_pA=0), so a probe over an "
                       "all-baseline bank ties and breaks to the lowest free index -- a real (measured) tie, "
                       "not a formula, but deterministic absent prior occupancy; the ANCHOR scenario is the "
                       "non-trivial case where prior occupancy actually differs and the read demonstrably "
                       "steers the allocation away from it.",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[d6_wm_competitive_slot_binding] VERDICT: {verdict}", flush=True)
    print(f"[d6_wm_competitive_slot_binding] wrote {a.out}\n" + "=" * 112, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
