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
    ap.add_argument("--attractor-off", action="store_true", help="DEFECT-1 TARGETING test: coactivation_replay(attractor_on=False). The NMDA attractor (comp_self_weight=12) may latch a slot the next fact's 1400pA drive cannot displace — measured, the driven slot wins only 15/27 windows (chance 9/27) though competition is near-exclusive. NOTE: an earlier A/B of this flag was UNINTERPRETABLE because the weight read preceded replay; it is only meaningful against the CORE-AFTER-REPLAY numbers.")
    ap.add_argument("--hebb-max", type=float, default=2.5, help="hebbian_max_weight. THE TRAP: measured ca1->slot weights land at 2.55-2.87, ABOVE the 2.5 this probe was setting -- so every Hebbian potentiation was NEGATIVE and dragged all synapses to a common ceiling, producing the flat per-core weights. 8th instance of this trap today (STDP/BDSP/BTSP/Hebbian). Raise it above the design weights.")
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
    b.core_config.hebbian_max_weight = float(args.hebb_max)
    print('  LEVER: hebbian_max_weight = %.3f  (ca1->slot lands ~2.55-2.87; a bound BELOW that inverts the rule)'
          % b.core_config.hebbian_max_weight)
    b.core_config.enable_stdp = False
    # ⚠️ comp_dendritic=True ALREADY SETS coincidence_weighted_drive=True
    # (nmda_compositional_consolidation.py:374). So a flag that only turns it ON is a NO-OP and the
    # A/B compares identical configs -- which is exactly what happened on the first run of this probe.
    # Set it EXPLICITLY in BOTH directions, and PRINT it, so the lever is verified rather than assumed.
    b.core_config.coincidence_weighted_drive = bool(args.weighted_coincidence)
    # Read BEFORE and AFTER so the lever is proven to have moved, not merely issued. (The previous
    # version of this line used a generated getattr expression that resolved to the attribute name
    # "'" and printed None every run -- making a whole 6-run A/B uninterpretable. Keep it plain.)
    _sr_before = getattr(b.core_config, "coincidence_plateau_self_regen", None)
    if args.self_regen is not None:
        b.core_config.coincidence_plateau_self_regen = float(args.self_regen)
    _sr_after = getattr(b.core_config, "coincidence_plateau_self_regen", None)
    print(f"  LEVER: self_regen {_sr_before} -> {_sr_after}"
          f"{'  (UNCHANGED — lever did not move!)' if _sr_before == _sr_after and args.self_regen is not None else ''}")
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

    # ---- THE UPSTREAM MEASUREMENT I HAD NOT TAKEN: are the ca1->slot weights themselves FACT-SELECTIVE?
    # The weighted plateau grades by these weights. I reported only their MEAN (1.04-1.21) -- if the
    # per-slot weights are uniform, weighted drive has NOTHING to grade and the plateau CANNOT be
    # selective no matter what the plateau parameters do. That would relocate the failure from the
    # dendrite to the ca1->slot write. Measure per-slot, before touching the plateau.
    _c = b.cp_connections
    _nz = int(_c.nnz)
    _po = to_host(_c.indices).astype(np.int64)[:_nz]
    _ip = to_host(_c.indptr).astype(np.int64)
    _pr = np.repeat(np.arange(len(_ip) - 1), np.diff(_ip))[:_nz]
    _wd = to_host(_c.data).astype(np.float64)[:_nz]
    try:
        _ca1 = np.asarray(sorted(rm.indices("ca1")), dtype=np.int64)
        _m_ca1 = np.isin(_pr, _ca1)
        per_slot_w = {}
        for j in sorted(slot):
            m = _m_ca1 & np.isin(_po, slot[j])
            per_slot_w[j] = float(_wd[m].mean()) if m.sum() else 0.0
        _vals = [per_slot_w[j] for j in sorted(per_slot_w)]
        _spread = (max(_vals) - min(_vals)) / max(max(_vals), 1e-9) * 100
        print(f"  (UPSTREAM-raw) ca1->slot MEAN WEIGHT PER SLOT: {[round(v,4) for v in _vals]}  spread={_spread:.1f}%")
        print( "     ⚠️ THIS RAW MEAN IS THE KNOWN-DILUTING METRIC. It averages ALL ca1->slot synapses, while a")
        print( "        selective write lives only on the fact's small CA1 CORE -- the arc already established")
        print( "        that the raw mean washes it out and that a core/firing-weighted read is what makes it")
        print( "        visible. Do NOT read selectivity off the line above. The CORE-RESTRICTED read follows.")
        # CORE-RESTRICTED: for each fact i, use ONLY that fact's engram-tagged CA1 cells, and compare their
        # mean weight onto slot i (own) vs onto the other slots (other). This is the same correction that
        # made the ca1->slot 6-seed GO visible; the raw mean above is expected to be flat even when this is not.
        core_rows = []
        for i in sorted(slot):
            try:
                # get_engram_tag_indices returns a BACKEND array (CuPy on GPU). np.asarray(sorted(cupy_arr))
                # raises, and the bare `except` reported that as "no engram core" — a TYPE BUG presenting as a
                # scientific null (the tags DO exist: verified 12/12/10 indices for ep_0/1/2). Convert through
                # to_host explicitly, and SHOW the failure instead of swallowing it.
                _ci = b.get_engram_tag_indices(tags[i])
                core = np.asarray(to_host(_ci), dtype=np.int64).ravel() if _ci is not None else None
            except Exception as _ce:
                print("     fact %d: engram core read FAILED (%s: %s)" % (i, type(_ce).__name__, _ce))
                core = None
            if core is None or core.size == 0:
                core_rows.append(None); continue
            m_core = np.isin(_pr, core)
            ws = []
            for j in sorted(slot):
                m = m_core & np.isin(_po, slot[j])
                ws.append(float(_wd[m].mean()) if m.sum() else 0.0)
            own = ws[i]; other = np.mean([w for k, w in enumerate(ws) if k != i])
            core_rows.append((ws, own / other if other > 1e-12 else 0.0, int(core.size)))
        print("  (UPSTREAM-core) per-fact, restricted to that fact's CA1 engram core:")
        oks = 0
        for i, r in enumerate(core_rows):
            if r is None:
                print(f"     fact {i}: no engram core -- cannot evaluate"); continue
            ws, oo, ncore = r
            ok = (int(np.argmax(ws)) == i)
            oks += ok
            print(f"     fact {i}: weights->slots {[round(w,4) for w in ws]}  own/other={oo:.3f}  "
                  f"own_is_max={ok}  (core={ncore} cells)")
        _evaluable = sum(1 for r in core_rows if r is not None)
        if _evaluable == 0:
            print("     => UNDEFINED: no fact had an evaluable engram core. This is NOT 'own-is-max 0/N' —")
            print("        printing a score here would FABRICATE A NEGATIVE out of an INSTRUMENT FAILURE.")
            print("        (The first version of this probe did exactly that, three seeds running.)")
        else:
            print("     => own-is-max %d/%d evaluable. If THIS is selective while the raw mean is flat," % (oks, _evaluable))
        print( "        the ca1->slot write IS fact-specific and the flat plateau is a READ/GRADING problem,")
        print( "        not a write problem. If THIS is also flat, the write itself is not localizing.")
    except Exception as _e:
        print(f"  (UPSTREAM) per-slot ca1->slot read failed: {_e}")

    # ---- sample cp_v_apical on every slot at EVERY step of the real replay.
    samples = {j: [] for j in sorted(slot)}
    fire_samples = {j: [] for j in sorted(slot)}
    orig_step = b._run_one_simulation_step

    def sampling_step(*a_, **k_):
        r = orig_step(*a_, **k_)
        # per-slot SOMATIC firing too: the apical says every slot is depolarized, but the DRIVE is
        # selective (coactivation_replay drives only slot_idx[i]). If all slots also FIRE in every
        # window, coincidence is global and Hebbian potentiates every ca1->slot synapse alike --
        # which is the mechanism behind the flat write.
        _fs = to_host(b.cp_firing_states)
        for j in sorted(slot):
            fire_samples[j].append(float(_fs[slot[j]].sum()))
        if getattr(b, "cp_v_apical", None) is not None:
            va = to_host(b.cp_v_apical)
            for j in sorted(slot):
                samples[j].append(float(va[slot[j]].max()))   # MAX: the most generous possible read
        return r

    b._run_one_simulation_step = sampling_step
    try:
        coactivation_replay(b, CONSOLIDATED_FACTS, tags, int(args.cycles), args.seed,
                            coactivate=True, attractor_on=not args.attractor_off)
    finally:
        b._run_one_simulation_step = orig_step

    # ---- ⚠️ THE MEASUREMENT THAT WAS MISSING (fixed 2026-07-26). The per-core selectivity block above runs
    # BEFORE this point, i.e. it measures the ENCODE phase and NEVER the replay write. A "BOUNDARY LOCATED:
    # coactivation_replay produces a non-selective ca1->slot write" finding was committed on those numbers —
    # they could not have shown a replay effect at all, and the attractor-on/off arms came out BYTE-IDENTICAL
    # for exactly that reason. Re-measure the SAME quantity here, AFTER replay, and report the DELTA.
    _c2 = b.cp_connections
    _nz2 = int(_c2.nnz)
    _po2 = to_host(_c2.indices).astype(np.int64)[:_nz2]
    _ip2 = to_host(_c2.indptr).astype(np.int64)
    _pr2 = np.repeat(np.arange(len(_ip2) - 1), np.diff(_ip2))[:_nz2]
    _wd2 = to_host(_c2.data).astype(np.float64)[:_nz2]
    print("  (CORE-AFTER-REPLAY) per-fact ca1->slot, restricted to that fact's engram core:")
    _oks2, _ev2 = 0, 0
    for i in sorted(slot):
        try:
            _ci2 = b.get_engram_tag_indices(tags[i])
            _core2 = np.asarray(to_host(_ci2), dtype=np.int64).ravel() if _ci2 is not None else None
        except Exception as _e2:
            print("     fact %d: core read FAILED (%s)" % (i, type(_e2).__name__)); _core2 = None
        if _core2 is None or _core2.size == 0:
            print("     fact %d: no evaluable core" % i); continue
        _mc2 = np.isin(_pr2, _core2)
        _ws2 = []
        for j in sorted(slot):
            _m2 = _mc2 & np.isin(_po2, slot[j])
            _ws2.append(float(_wd2[_m2].mean()) if _m2.sum() else 0.0)
        _other2 = np.mean([w for k, w in enumerate(_ws2) if k != i])
        _oo2 = _ws2[i] / _other2 if _other2 > 1e-12 else 0.0
        _ok2 = int(np.argmax(_ws2)) == i
        _oks2 += _ok2; _ev2 += 1
        print("     fact %d: weights->slots %s  own/other=%.4f  own_is_max=%s  (core=%d)"
              % (i, [round(w, 4) for w in _ws2], _oo2, _ok2, _core2.size))
    if _ev2:
        print("     => AFTER REPLAY: own-is-max %d/%d evaluable." % (_oks2, _ev2))
    else:
        print("     => UNDEFINED (no evaluable cores) — not a negative.")

    # ---- WINDOW SEGMENTATION: reconstruct which fact was driven in each 30-step burst, exactly as
    # coactivation_replay does (np.random.default_rng(seed+777), reshuffled each cycle), then report
    # per-slot SOMATIC firing per window. THE QUESTION: does the DRIVEN slot dominate its own window?
    _burst = 30
    _rng = np.random.default_rng(int(args.seed) + 777)
    _order = []
    for _c in range(int(args.cycles)):
        _o = list(range(N)); _rng.shuffle(_o); _order.extend(_o)
    print("  (WINDOWS) per-slot SOMATIC spikes per replay burst (driven fact in brackets):")
    _dom = 0
    _nw = min(len(_order), len(fire_samples[0]) // _burst)
    for w in range(_nw):
        sl = slice(w * _burst, (w + 1) * _burst)
        tot = [float(np.asarray(fire_samples[j][sl]).sum()) for j in sorted(slot)]
        drv = _order[w]
        win = int(np.argmax(tot))
        _dom += (win == drv)
        if w < 6 or win != drv:
            print(f"     window {w:2d} [fact {drv}]: spikes={[int(t) for t in tot]}  argmax=slot {win}  "
                  f"{'driven slot dominates' if win == drv else '⛔ NON-DRIVEN slot dominates'}")
    print(f"     => driven slot dominated its own window in {_dom}/{_nw} windows (chance {_nw//N}/{_nw}).")
    print( "        If ~chance, coincidence during replay is GLOBAL and the flat ca1->slot write is explained:")
    print( "        every ca1 cell co-fires with every slot, so Hebbian/BTSP potentiate them all alike.")
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
