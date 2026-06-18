"""THROWAWAY de-risk (do NOT commit): find a NON-SATURATED, GRADED-delta SNc operating point
on the MERGED nav critic so a value-train CAN learn a meaningful V.

Context (read first):
  - 2026-06-18-merged-config-homeostasis-boundary-RESOLVED.md: a standalone-tuned organ fires
    ~6-10x weaker co-resident because the merged bridge keeps GLOBAL enable_homeostasis=False;
    fix = per-region BrainRegion.enable_homeostasis=True (the threshold-select bridge.py:6320-6323
    gives masked neurons the low ~-42mV threshold; the synaptic-scaling foot-gun is gated by the
    SEPARATE cfg.enable_synaptic_scaling, OFF).
  - 2026-06-18-organ-lift-homeo-generalize-derisk.md: the enabler GENERALIZES to the FULL nav SNc
    (446 Hz / 5.47x), BUT the homeostasis-boosted SNc SATURATES (~438 Hz) so the GABA_B value
    subtraction has the right DIRECTION (pred<unpred) but a WEAK gap (~1.05). The clean graded
    delta needs a NON-SATURATED SNc operating point.

THE DE-RISK QUESTION: at what SNc operating point (tonic / GIRK cap gabab_conductance_max / which
regions get per-region homeostasis) does the merged nav critic's GABA_B delta=r-V become GRADED
(gap clearly >1.3, not saturated ~1.05), so a value-train CAN learn a meaningful V?

KEY INSIGHT TO TEST (prompt #3): put per-region homeostasis on the CRITIC (striosome_value,
reward_us) but NOT the SNc, so the SNc stays at vpeak (high threshold => doesn't saturate) while
reward_us still drives it to burst and the critic still fires. The merged builder masks ALL THREE
(snc, reward_us, striosome_value); this de-risk additionally tests masking only {reward_us,
striosome_value} (SNc at vpeak).

The merged co_resident_nav_critic critic afferent is `vs_place_context` (dense Gaussian, NOT the
place self-org pool — the merged builder doesn't run the place self-org). So the critic afferent
here is driven DIRECTLY to a moderate rate (the value-train's job is to grow the
vs_place_context->striosome_value weight, which we do NOT do here — we drive the afferent and read
whether the critic CAN fire + whether the GABA_B subtraction is graded at a given SNc op point).

ONE build, sweep in-process. CPU-friendly. Run under SIM_BACKEND=numpy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CRITIC_AFFERENT = "vs_place_context"   # the merged co_resident_nav_critic afferent (dense Gaussian)
CRITIC = "striosome_value"
SNC = "snc"
REWARD_US = "reward_us"


def _host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return a


def _idx(bridge, name, xp):
    import numpy as np
    return xp.asarray(np.asarray(bridge.region_manager.indices(name), dtype=np.int64))


def _settle(bridge, xp, n_steps=80):
    """Clean-reset read protocol: zero external current + the slow GABA_B/GIRK conductance, run a
    silent gap so fast conductances + membranes decay to rest before the next frozen measurement
    (the GABA_B tau~150ms from a prior window must decay, else pred>unpred is an order artifact)."""
    bridge.cp_external_input_current[:] = 0.0
    if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
        bridge.cp_conductance_g_gabab[:] = 0.0
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)


def _rate(bridge, region_idx_map, drives, xp, n_steps=40):
    """Drive {region: pA}, step, return {region: Hz}."""
    bridge.cp_external_input_current[:] = 0.0
    for nm, pa in drives.items():
        bridge.cp_external_input_current[region_idx_map[nm]] = xp.float32(pa)
    counts = {nm: 0 for nm in region_idx_map}
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        for nm, gi in region_idx_map.items():
            counts[nm] += int(_host(bridge.cp_firing_states[gi]).sum())
    dur_s = n_steps * 1e-3
    return {nm: counts[nm] / max(len(_host(gi)), 1) / dur_s for nm, gi in region_idx_map.items()}


def _set_homeostasis_mask(bridge, masked_names, xp):
    """Rebuild cp_homeostasis_neuron_mask to cover exactly `masked_names` (a set of region names).
    The mask is allocated at init from the regions' enable_homeostasis flags (bridge.py:1227-1245);
    here we override it in-place so ONE build can test multiple masks. Returns n masked.

    threshold-select bridge.py:6320-6323 branch 2: global-off + mask -> cp.where(mask, adapted, vpeak).
    So a neuron IN the mask gets the low (~-42mV) adapted threshold; OUT keeps vpeak (+35mV)."""
    import numpy as np
    m = getattr(bridge, "cp_homeostasis_neuron_mask", None)
    n_total = int(len(_host(bridge.cp_membrane_potential_v)))
    new = np.zeros(n_total, dtype=bool)
    rm = bridge.region_manager
    for nm in masked_names:
        for i in rm.indices(nm):
            new[int(i)] = True
    bridge.cp_homeostasis_neuron_mask = xp.asarray(new)
    return int(new.sum())


def build_merged(seed, vocab=None):
    from research.runners.nav_conv_merged_bridge import build_merged_nav_conv_bridge
    b, _ = build_merged_nav_conv_bridge(seed=seed, vocab=vocab, co_resident_nav_critic=True)
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_navcritic_valuetrain_opmap_run2.json")
    args = ap.parse_args()

    from sim.backend import get_backend
    xp, backend = get_backend()
    print(f"[navcritic-valuetrain-opmap de-risk seed={args.seed}] backend={backend}")
    print("  building the MERGED nav+conv bridge (co_resident_nav_critic=True) ... (~2 min numpy)")
    b = build_merged(args.seed)
    cc = b.core_config

    # print the critic region names (sanity: which regions the full nav critic built on the merge)
    rm = b.region_manager
    names = set(r.name for r in rm.regions())
    for n in (SNC, CRITIC, REWARD_US, CRITIC_AFFERENT):
        print(f"    region {n!r}: present={n in names} "
              f"({len(list(rm.indices(n))) if n in names else 0} neurons)")
    assert all(n in names for n in (SNC, CRITIC, REWARD_US, CRITIC_AFFERENT)), \
        "FAIL: the merged nav critic did not build the expected regions"

    # which regions carry enable_homeostasis from the builder?
    builder_masked = sorted(r.name for r in rm.regions() if getattr(r, "enable_homeostasis", False))
    print(f"  builder set enable_homeostasis=True on: {builder_masked}")

    # match the limbic-measurement regime: OU on (the op point was pinned with OU), learning frozen,
    # homeostatic-threshold adaptation frozen (so the sweep isn't drift-contaminated).
    cc.enable_ou_process = True
    cc.ou_std_current_pA = 100.0
    cc.reward_learning_rate = 0.0
    cc.homeostasis_threshold_adapt_rate = 0.0

    region_idx_map = {nm: _idx(b, nm, xp) for nm in (SNC, CRITIC, REWARD_US, CRITIC_AFFERENT)}

    def meas(drives, n_steps=40):
        _settle(b, xp)
        return _rate(b, region_idx_map, drives, xp, n_steps=n_steps)

    results = {"seed": args.seed, "backend": backend, "builder_masked": builder_masked, "sweeps": []}

    # ---- The homeostasis-mask configs to test ----
    #   "builder"        : ALL THREE {snc, reward_us, striosome_value} (the merged default; SNc saturates).
    #   "critic_no_snc"  : ONLY {reward_us, striosome_value} -> SNc at vpeak (the prompt's KEY INSIGHT).
    #   "critic_only"    : ONLY {striosome_value} -> reward_us AND snc at vpeak (does reward_us still
    #                      burst the SNc from vpeak? if not, this isolates whether reward_us NEEDS the boost).
    # builder (all 3 masked) is KNOWN to saturate the SNc ~435 Hz (sweep run 1) — dropped. Focus on the
    # SNc-NON-saturating masks (the prompt's KEY INSIGHT: homeostasis on critic but NOT snc), now with the
    # critic driven ABOVE its ~600 pA rheobase so the GABA_B has a real V to subtract.
    mask_configs = {
        "critic_no_snc": {REWARD_US, CRITIC},   # SNc at vpeak (~155-170 Hz burst — non-saturated)
        "critic_only": {CRITIC},                # SNc + reward_us at vpeak (~80-100 Hz burst — clearly non-sat)
    }

    # SNc tonic range: the saturating default was ~160-210. Go LOWER (the prompt: try a wide low range)
    # to find a non-saturated point with GABA_B headroom.
    tonic_grid = [120, 160, 210]
    # GIRK cap (gabab_conductance_max): 0.0 = OFF/uncapped (merged default). Finite caps BOUND the GABA_B
    # K+ conductance so a fired critic can't over-clamp the SNc to 0 (the documented nav fix for exactly
    # this saturating-critic case). Smaller cap = stronger bound = MORE graded subtraction headroom.
    girk_caps = [0.0, 1.0]
    us_drive = 400.0          # the reward_us US drive (de-risk default)
    cue_drive = 800.0         # the critic-afferent drive (vs_place_context) — drive it to a moderate rate
    # afferent-weight caveat: the merged co_resident_nav_critic vs_place_context->striosome_value INIT
    # weight is 0.20 (untrained). So driving vs_place_context here may NOT fire the critic at init weight
    # (the de-risk's documented "sparse/weak afferent can't fire the MSN at init" boundary). We measure
    # BOTH the afferent-driven critic rate AND a DIRECT critic drive (proves the critic CAN fire at this
    # op point given a trained weight / direct depolarization), so V-learnability is decoupled from the
    # init-weight afferent firing.

    best = None  # (mask, tonic, girk, burst, gap, strio_via_afferent, strio_direct)

    for mask_name, mask_set in mask_configs.items():
        n_masked = _set_homeostasis_mask(b, mask_set, xp)
        # report the actual thresholds the mask produced on each region (verify the boost is applied)
        thr = getattr(b, "cp_neuron_firing_thresholds", None)
        snc_thr = float(_host(thr[region_idx_map[SNC]]).mean()) if thr is not None else float("nan")
        us_thr = float(_host(thr[region_idx_map[REWARD_US]]).mean()) if thr is not None else float("nan")
        crit_thr = float(_host(thr[region_idx_map[CRITIC]]).mean()) if thr is not None else float("nan")
        print(f"\n=== homeostasis mask = {sorted(mask_set)} ({n_masked} neurons) ===")
        print(f"    region adapted-thr (mask gives ~-42mV; vpeak=+35mV when OUT): "
              f"snc={snc_thr:.1f} reward_us={us_thr:.1f} striosome_value={crit_thr:.1f} "
              f"[masked: snc={SNC in mask_set} us={REWARD_US in mask_set} crit={CRITIC in mask_set}]")

        for girk in girk_caps:
            cc.gabab_conductance_max = float(girk)
            for tonic in tonic_grid:
                base = meas({SNC: tonic})
                unpred = meas({REWARD_US: us_drive, SNC: tonic})
                # critic fires from its AFFERENT (vs_place_context) at the INIT weight 0.20?
                crit_aff = meas({CRITIC_AFFERENT: cue_drive, SNC: tonic})
                strio_aff = crit_aff[CRITIC]
                # value subtraction via the AFFERENT path (the real loop: afferent->critic->GABA_B->snc)
                pred_aff = meas({CRITIC_AFFERENT: cue_drive, REWARD_US: us_drive, SNC: tonic})
                # ALSO drive the critic DIRECTLY (proves the critic CAN fire + subtract at this op point,
                # decoupled from the weak init afferent — this is the "trained-weight" / V-learned proxy).
                # The MSN-D1 rheobase is ~600 pA (f-I probe: 0 Hz below 600, 16-60 Hz at 600-800); 500 pA
                # was sub-rheobase (silent) — drive at 1000 pA so the trained-V critic fires ~80 Hz, the
                # value the GABA_B subtracts.
                crit_dir = meas({CRITIC: 1000.0, SNC: tonic})
                strio_dir = crit_dir[CRITIC]
                pred_dir = meas({CRITIC: 1000.0, REWARD_US: us_drive, SNC: tonic})

                burst = unpred[SNC] / max(base[SNC], 1e-6)
                gap_aff = unpred[SNC] / max(pred_aff[SNC], 1e-6)
                gap_dir = unpred[SNC] / max(pred_dir[SNC], 1e-6)
                # saturation flag: SNc near its ceiling leaves no GABA_B headroom. The IZH2007_DOPAMINE
                # max sustained rate is bounded; we flag "saturated" when the reward burst exceeds ~300 Hz
                # (the de-risk's ~438 saturating point) AND the gap is weak.
                saturated = unpred[SNC] >= 300.0
                row = dict(mask=mask_name, tonic=tonic, girk=girk,
                           base_hz=base[SNC], unpred_hz=unpred[SNC],
                           pred_aff_hz=pred_aff[SNC], pred_dir_hz=pred_dir[SNC],
                           strio_via_afferent_hz=strio_aff, strio_direct_hz=strio_dir,
                           reward_us_hz=unpred[REWARD_US], afferent_hz=crit_aff[CRITIC_AFFERENT],
                           burst=burst, gap_afferent=gap_aff, gap_direct=gap_dir,
                           saturated=bool(saturated))
                results["sweeps"].append(row)
                graded_dir = gap_dir > 1.3 and pred_dir[SNC] < unpred[SNC]
                flag = ""
                if burst >= 3.0 and graded_dir and not saturated:
                    flag = " <== GRADED+NONSAT (direct)"
                    if best is None or gap_dir > best[4]:
                        best = (mask_name, tonic, girk, burst, gap_dir, strio_aff, strio_dir)
                print(f"    {mask_name:13s} tonic={tonic:3d} girk={girk:3.1f} | "
                      f"base={base[SNC]:5.1f} unpred={unpred[SNC]:6.1f} predA={pred_aff[SNC]:6.1f} "
                      f"predD={pred_dir[SNC]:6.1f} | burst={burst:5.2f} gapA={gap_aff:4.2f} "
                      f"gapD={gap_dir:4.2f} | strioA={strio_aff:5.1f} strioD={strio_dir:5.1f} "
                      f"{'SAT' if saturated else ''}{flag}")

    # reset the GABA_B cap (cleanliness; the bridge is throwaway anyway)
    cc.gabab_conductance_max = 0.0

    print("\n\n========================= SUMMARY =========================")
    print(f"  best GRADED+NON-SATURATED point (direct-critic gap>1.3): {best}")
    if best is not None:
        m, t, g, bu, ga, sa, sd = best
        print(f"  => GO: mask={m}, snc_tonic={t}, girk_cap={g} -> burst={bu:.2f}x, gap_direct={ga:.2f}, "
              f"critic-via-afferent={sa:.1f}Hz, critic-direct={sd:.1f}Hz")
        results["verdict"] = "GO"
        results["best"] = dict(mask=m, snc_tonic=t, girk_cap=g, burst=bu, gap_direct=ga,
                               strio_via_afferent_hz=sa, strio_direct_hz=sd)
    else:
        print("  => NO graded+non-saturated point found with the swept (mask,tonic,girk). "
              "The value-train needs a different rate-normalizer (see findings doc).")
        results["verdict"] = "NEGATIVE"
        results["best"] = None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
