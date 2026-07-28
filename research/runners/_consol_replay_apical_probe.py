"""Does REPLAY, on its own, supply the apical instructive signal BTSP needs? (2026-07-26)

WHY THIS IS THE CRITICAL-PATH MEASUREMENT
-----------------------------------------
The whole consolidation arc rests on a claim that exists only as a COMMENT in
`_consol_cortical_store_probe.py`:

    "coactivation_replay drives the target slot SOMATICALLY, but BTSP's instructive signal is the
     APICAL plateau (max(v_apical - v_hold, 0)) - somatic drive supplies none, so pool->slot never
     receives a teaching signal"

That comment is why the host apical teaching clamp was introduced -- and the clamp is exactly the
shortcut that made today's result "host-supervised" rather than self-organized, and that (by routing
into the `if teaching_clamp:` branch) BYPASSES replay entirely so nothing is actually consolidated.

Today's standing lesson is that a claim living in a comment is a HYPOTHESIS, not a fact (a comment
cannot fail; three such claims were false today, including a "lesion" that never held). So MEASURE it:
run the real `coactivation_replay` and record `cp_v_apical` on the slot neurons throughout.

WHAT THE ANSWER MEANS
  * apical stays at/below v_hold during replay  -> the comment is RIGHT; the gap is real and precisely
    located: replay must be made to drive the APICAL compartment (the honest next mechanism).
  * apical rises above v_hold on the target slot -> the comment is WRONG, the clamp was never needed,
    and today's central shortcut can simply be deleted.

Either way this is decisive and costs one run. No sim/ edit; monkeypatches only this process's step
call to sample, and restores it.

  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_replay_apical_probe --seed 42
"""
import os, sys, json, argparse
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from types import SimpleNamespace
from research.runners.nmda_compositional_consolidation import (
    build_substrate, encode_facts_with_reinstatement, coactivation_replay,
    CONSOLIDATED_FACTS, _try_tgate, _try_pgate)
from research.runners._consol_direct_weight_probe import BASE
from sim.backend import get_backend, to_host

cp, BACKEND = get_backend()
N = len(CONSOLIDATED_FACTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--self-regen", type=float, default=None, help="coincidence_plateau_self_regen (runner default 0.15). This is a v-GATED SUSTAIN LATCH: once tripped the plateau holds itself up independently of ongoing drive, which would ERASE the graded differences weighted drive creates. 0.0 = no latch.")
    ap.add_argument("--weighted-coincidence", action="store_true", help="engine cfg.coincidence_weighted_drive (set EXPLICITLY both ways; comp_dendritic already defaults it True): grade the apical plateau by EFFECTIVE SYNAPTIC WEIGHT instead of the COUNT of coincident inputs. The count-based default is an all-or-none switch, so every slot crossing k gets a FULL plateau regardless of weight => the uniform signal measured. Config-only; no sim/ edit.")
    ap.add_argument("--out", default="research/findings/raw/cortical_store")
    args = ap.parse_args()

    a = dict(BASE)
    a.update(comp_dendritic=True, comp_wta_weight=5.0, comp_k_thresh=2.0, comp_self_regen=0.15,
             comp_kir_g=3.0, comp_v_hold=-50.0, comp_apical_R=0.15, comp_gc_read=0.5,
             comp_btsp=True, comp_btsp_lr=0.0005, comp_btsp_wmax=2000.0, comp_btsp_elig_tau=30.0,
             comp_no_pool_slot=False, comp_pool_slot_weight=1.5, comp_attractor_slots=N,
             enable_hebbian=True)
    b = build_substrate(args.seed, SimpleNamespace(**a))
    b.core_config.hebbian_max_weight = 2.5
    b.core_config.enable_stdp = False
    # ⚠️ comp_dendritic=True ALREADY SETS coincidence_weighted_drive=True
    # (nmda_compositional_consolidation.py:374). So a flag that only turns it ON is a NO-OP and the
    # A/B compares identical configs -- which is exactly what happened on the first run of this probe.
    # Set it EXPLICITLY in BOTH directions, and PRINT it, so the lever is verified rather than assumed.
    b.core_config.coincidence_weighted_drive = bool(args.weighted_coincidence)
    if args.self_regen is not None:
        b.core_config.coincidence_plateau_self_regen = float(args.self_regen)
    print(f"  LEVER: self_regen = {getattr(b.core_config, chr(39)+chr(39).join([]) or 'coincidence_plateau_self_regen', None)}")
    print(f"  LEVER: coincidence_weighted_drive = {b.core_config.coincidence_weighted_drive} "
          f"(comp_dendritic sets it True by default -- an ON-only flag would be a no-op)")

    rm = b.region_manager
    names = set(rm.region_names()) if hasattr(rm, "region_names") else set()
    slot = {i: np.asarray(sorted(rm.indices(f"comp_attr_{i}")), dtype=np.int64) for i in range(N)
            if f"comp_attr_{i}" in (names or {f"comp_attr_{i}"})}
    v_hold = float(getattr(b.core_config, "comp_v_hold", -50.0))

    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS)
    # ca1_concept_weight INITS AT 0.0 and must be grown by plasticity. If it is still ~0 the weighted
    # plateau has nothing to grade and would read as a (misleading) flat null -- so measure it explicitly.
    from research.runners.nmda_compositional_consolidation import _mean_gate_weight as _mgw
    try:
        print(f"  ca1->slot mean weight after encode: {_mgw(b, 'ca1_to_comp_attr'):.5f}  (inits at 0.0; weighted drive needs this > 0)")
    except Exception as _e:
        print(f"  ca1->slot weight read failed: {_e}")

    # ---- sample cp_v_apical on every slot at EVERY step of the real replay.
    samples = {j: [] for j in sorted(slot)}
    orig_step = b._run_one_simulation_step

    def sampling_step(*a_, **k_):
        r = orig_step(*a_, **k_)
        if getattr(b, "cp_v_apical", None) is not None:
            va = to_host(b.cp_v_apical)
            for j in sorted(slot):
                samples[j].append(float(va[slot[j]].max()))   # MAX: the most generous possible read
        return r

    b._run_one_simulation_step = sampling_step
    try:
        coactivation_replay(b, CONSOLIDATED_FACTS, tags, int(args.cycles), args.seed,
                            coactivate=True, attractor_on=True)
    finally:
        b._run_one_simulation_step = orig_step

    print(f"[seed {args.seed}] backend={BACKEND}  v_hold={v_hold}  steps sampled={len(samples[0])}")
    print("  (values are the MAX v_apical over each slot's neurons -- the most generous read possible)")
    any_above = False
    res = {}
    for j in sorted(slot):
        arr = np.asarray(samples[j])
        above = int((arr > v_hold).sum())
        any_above |= above > 0
        res[f"slot{j}"] = dict(max=round(float(arr.max()), 3), mean=round(float(arr.mean()), 3),
                               p99=round(float(np.percentile(arr, 99)), 3),
                               steps_above_v_hold=above, n_steps=int(arr.size))
        print(f"    slot {j}: max={arr.max():8.3f}  p99={np.percentile(arr,99):8.3f}  "
              f"mean={arr.mean():8.3f}  steps above v_hold({v_hold}): {above}/{arr.size}")
    print()
    # VERDICT LOGIC (hardened 2026-07-26): "above v_hold" is NOT sufficient for a usable teaching signal.
    # BTSP's instructive term is max(v_apical - v_hold, 0) applied PER POSTSYNAPTIC CELL, so a signal that is
    # (a) present on EVERY slot equally carries NO fact information and writes uniformly, and (b) a v_apical
    # outside the physiological range is the SAME artifact class as the 333x apical_R miscalibration this arc
    # already retracted once. Check selectivity and range, not just presence -- the first version of this
    # verdict printed "the clamp may be removable outright" on a saturating non-selective 400 mV signal.
    means = {j: float(np.asarray(samples[j]).mean()) for j in sorted(slot)}
    mx = max(means.values()); mn = min(means.values())
    selective = (mx - mn) / max(abs(mx), 1e-9) > 0.20          # >20% spread between slots
    physiological = all(np.asarray(samples[j]).max() <= 50.0 for j in sorted(slot))
    if any_above and selective and physiological:
        print("  ⇒ REPLAY SUPPLIES A USABLE, SELECTIVE, PHYSIOLOGICAL apical teaching signal. The comment that")
        print("    motivated the host clamp is WRONG and the clamp is removable.")
    elif any_above and not physiological:
        print(f"  ⛔ REPLAY DRIVES THE APICAL COMPARTMENT, BUT UNPHYSIOLOGICALLY (max {max(np.asarray(samples[j]).max() for j in sorted(slot)):.1f} mV,")
        print("     physiological range -90..+50). Same artifact class as the 333x apical_R miscalibration.")
        print(f"     Selective between slots? {'YES' if selective else 'NO'} (per-slot means {({k: round(v,1) for k,v in means.items()})}).")
        print("     ⇒ the comment's CONCLUSION (replay cannot teach selectively) may hold, but its REASON is wrong:")
        print("        the signal is not ABSENT, it is SATURATING and" + ("" if selective else " NON-SELECTIVE") + ".")
    elif any_above and not selective:
        print("  ⛔ REPLAY DRIVES EVERY SLOT'S APICAL COMPARTMENT EQUALLY -> the instructive signal carries NO")
        print(f"     fact information (per-slot means {({k: round(v,1) for k,v in means.items()})}); BTSP would write UNIFORMLY.")
    else:
        print("  ⇒ CONFIRMED: replay NEVER lifts any slot's apical compartment above v_hold, so BTSP's")
        print("    instructive signal max(v_apical - v_hold, 0) is IDENTICALLY ZERO throughout replay.")
        print("    The gap is real and precisely located: replay drives slots SOMATICALLY only. The honest")
        print("    next mechanism is to make a replay event depolarize the APICAL compartment of its target.")
    from pathlib import Path
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}/replay_apical_seed{args.seed}.json").write_text(json.dumps(
        dict(seed=args.seed, v_hold=v_hold, any_above_v_hold=bool(any_above), per_slot=res,
             argv=sys.argv[1:]), indent=2))
    print("REPLAY-APICAL-PROBE DONE")


if __name__ == "__main__":
    main()
