"""D5 STABILIZED SURFACED READ — de-risk a LOWER-VARIANCE conversation-visible recall-strength read.

THE RESIDUAL (verified 2026-08-21, do not re-derive): the D5 learn-through-use consolidation is wired default-OFF
(`BRAIN_D5_CONSOLIDATE`). It works, but the SURFACED graded read (`depth_hold` = mean-held max(cp_v_apical − v_hold, 0),
the substrate's own BTSP IS_post) is too NOISY at the mV scale, which blocks BOTH downstream verdicts:
  * memory-separator crosstalk (#73): "did A's consolidation move neighbour B?" is UNDECIDABLE because the repeated-read
    noise EXCEEDS the effect (a byte-identical-weight neighbour still moves several mV of pure read noise).
  * conversation-visibility (2026-08-20): the surfaced strength rises with use on only ~4/6 seeds; the same read noise
    buries the rise on the rest.

WHAT THE NOISE IS (DIAGNOSED this arc — see the finding): the production recall reads the LIVE bridge and only
`hard_silence`+`_reset_apical_latch` between reads, which does NOT return the bridge to the clean-rest baseline. So
residual state carries from one read into the next and the read becomes a stationary PERIOD-2 LIMIT CYCLE (lag-1 autocorr
≈ −0.8): consecutive live reads of the SAME memory ALTERNATE (e.g. depth_hold 0 ↔ 12 mV, binary 0 ↔ 0.43 — the reply
literally flips between "dog completes" and "dog does not"). At the near-saturated production encode the live read is even
WORSE — non-monotone in the very weight consolidation grows. Averaging/rate-coding the LIVE read cannot cure this (the
cycle is state-carryover, not zero-mean jitter, and at saturation there is no monotone signal to recover).

THE CURE, and it is the CLAUDE.md wall-reframe ("what companion process did we replace with a constant?"): the missing
process is the network's RETURN TO CLEAN-REST BASELINE between recalls (real recalls are separated in time; the network
settles). The incomplete reset approximated it with a constant. The faithful implementation is a COMPLETE reset — a
snapshot_state/restore to clean rest with the current weights injected — before the surfaced-strength read. That read is
then DETERMINISTIC (repeated-read std = 0) and a clean MONOTONE function of the stored weight (it is exactly the
snapshot-isolated read the step-6 / flip-verify instruments already validated). It is ADDITIVE: the binary moat gate
(`apical_cue`, `in_memory`) stays the LIVE read, byte-identical to production; only the surfaced STRENGTH number becomes
the stabilized isolated read. Host cost is only the snapshot/restore determinism guard — the SAME guard
consolidate_used_memory already declares.

THE DE-RISK measures, per graded read (depth_rest / depth_hold / soft) × {LIVE, ISOLATED}, on a weak-usable store
(step-6's adaptive encode, where the weight has rise headroom): the NOISE FLOOR (std over K repeated reads, no
consolidation) vs the EFFECT (rise after the real continuous_engine learn-through-use loop) → SNR = |effect|/noise_std.
GO: the ISOLATED read reaches SNR ≥ SNR_MIN (decidable — its noise ≈ 0) AND rises monotonically with use on MORE seeds
than the 4/6 depth_hold baseline (aim 6/6) AND the binary moat stays byte-identical; the LIVE read is shown undecidable
(the residual it was blocked on). HONEST-NEGATIVE is a first-class deliverable.

BRAIN-BASED (NO sim/ edit): both reads are the SAME spiking apical-dAP completion; the strengthening is the substrate's
OWN plateau-gated BTSP via the ACTUAL continuous_engine.consolidate_used_memory; host code is the clock, the encode
selection, and the snapshot/restore determinism guard. cupy is the D5 substrate.
  3-seed decisive: SIM_BACKEND=cupy python -m research.runners._d5_stabilized_read_derisk --seeds 42 43 44
  6-seed:          SIM_BACKEND=cupy python -m research.runners._d5_stabilized_read_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from sim.backend import get_backend  # noqa: E402
from research.runners._gap5_dendritic_dap_readout_completion_derisk import _apical_up_read  # noqa: E402
from research.runners._gap5_d5_latch_self_termination_derisk import snapshot_state, restore_state  # noqa: E402
# Reuse step-6's VALIDATED machinery (GradedEpisodicDapMemory read + adaptive weak-encode selection): the ISOLATED read
# below IS step-6's handler_read; this de-risk adds the LIVE-vs-ISOLATED noise-floor / SNR comparison step-6 never made.
from research.runners._d5_step6_graded_apical_read_derisk import (  # noqa: E402
    _select_encode, GRADED_READS)
from webapp import continuous_engine as CE  # noqa: E402  (the ACTUAL production wiring under test)
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_d5_stabilized_read" / "seed42.json"

SNR_MIN = 2.0               # decidability bar: repeated-read noise <= half the effect
MOVE_MARGIN = 1e-3          # a rise counts if the effect exceeds this (mV / [0,1])
MONO_TOL_FRAC = 0.02        # trajectory monotone if it never backtracks > this fraction of the total move
FAITHFUL_K = 3.0            # cue must beat perm/nocue by this factor
LESION_COLLAPSE_FRAC = 0.15  # formation-lesion read must be <= this fraction of the formed read
SOURCES = ("live", "iso")


def _g(rec, read):
    return float((rec.get("graded_cue") or {}).get(read, 0.0))


def _gp(rec, field, read):
    return float((rec.get(field) or {}).get(read, 0.0))


def _mono_rel(traj, tol_frac=MONO_TOL_FRAC):
    move = traj[-1] - traj[0]
    if move <= 0:
        return False
    tol = tol_frac * abs(move)
    return bool(all(traj[i + 1] >= traj[i] - tol for i in range(len(traj) - 1)))


def run_one(seed, a, backend, out_path):
    t0 = time.time()
    print("\n" + "=" * 118)
    print(f"[d5-stab] seed={seed} backend={backend} — LIVE vs snapshot-ISOLATED surfaced read: repeated-read noise vs "
          f"consolidation effect (SNR = |effect|/noise_std)", flush=True)
    result = {"seed": seed, "backend": backend, "params": vars(a)}
    try:
        cp, _ = get_backend()
        cache_key = ("d5-stab", seed)
        CE.forget_session(cache_key)
        te_grid = [int(x) for x in a.te_grid.split(",")] if a.te_grid else [a.train_events]

        # ── weak-usable store (step-6's adaptive selection): the lowest-binary completing headroom store; snap = a
        #    clean-rest snapshot; W_before = the stored weights. The weight has rise headroom (NOT saturated). ──────────
        org, te, rec0, snap, W_before, w_dog_before = _select_encode(seed, cp, te_grid, verbose=True)
        if org is None:
            result["verdict_status"] = "UNDEFINED"
            result["checks"] = {"instrument_valid": False, "reason": f"no te in {te_grid} landed a weak-usable store"}
            print(f"[d5-stab] seed={seed} INSTRUMENT-INVALID: no weak-encode store completed in headroom", flush=True)
            CE.forget_session(cache_key)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, default=str))
            return result
        mem = org.mem
        dslot = mem.topic_slot["dog"]; cslot = mem.topic_slot["cat"]

        def iso_read(topic, W, lesion=False):
            """The STABILIZED read: complete reset to the clean-rest snapshot + inject the current weights, then the
            SAME spiking apical-dAP recall (deterministic, weight-attributable)."""
            restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W)
            return org.recall(topic, lesion=lesion)

        def live_read(topic):
            """The CURRENT production read: org.recall on the LIVE warmed bridge (period-2 cycle)."""
            return org.recall(topic)

        # ── MOAT BYTE-IDENTITY: the surfaced isolated read is ADDITIVE — the binary gate (apical_cue) is the LIVE read,
        #    unchanged; pass it against production `_apical_up_read`. ──────────────────────────────────────────────────
        restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W_before)
        prod_up = _apical_up_read(mem.bridge, mem.R, [mem.held_pos_by_asm[dslot]], [mem.cue_by_asm[dslot]],
                                  mem.p["up_thresh"])
        gate_up = float(iso_read("dog", W_before)["apical_cue"])
        moat_byte_id = bool(abs(gate_up - prod_up) < 1e-9)
        print(f"[d5-stab] MOAT byte-id (gate stays the live binary): apical_cue={gate_up:.6f} prod_up={prod_up:.6f} "
              f"identical={moat_byte_id} te={te}", flush=True)
        inmem_T = bool(rec0["in_memory"])
        w_cat = float(cp.mean(W_before[mem.R.withinA_masks[cslot]]))

        # ── PHASE A: noise floor — K repeated reads, NO consolidation, LIVE (warmed) + ISOLATED ─────────────────────
        os.environ["BRAIN_D5_CONSOLIDATE"] = "0"
        for _ in range(a.warm_reads):          # warm the LIVE bridge into its steady period-2 regime
            live_read("dog")
        live_noise = [live_read("dog") for _ in range(a.k_noise)]
        iso_noise = [iso_read("dog", W_before) for _ in range(a.k_noise)]
        cat_iso = iso_read("cat", W_before)
        les_iso = iso_read("dog", W_before, lesion=True)   # formation-lesion faithfulness
        cat_never = bool(w_cat < 5.0 and not cat_iso["in_memory"])

        # ── PHASE B: the rise — N_TURNS consolidation ticks; each turn read LIVE + ISOLATED ─────────────────────────
        os.environ["BRAIN_D5_CONSOLIDATE"] = "1"
        W = W_before
        iso_traj = {r: [_g(iso_noise[0], r)] for r in GRADED_READS}
        live_traj = {r: [_g(live_noise[0], r)] for r in GRADED_READS}
        wdog_traj = [round(w_dog_before, 3)]; consolidated_rounds = 0
        for turn in range(a.n_turns):
            restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W)
            CE.mark_recall(cache_key, "dog")
            rec = CE.consolidate_used_memory(cache_key, org, n_episodes=a.n_episodes)
            if rec is not None:
                consolidated_rounds += 1
            W = mem.R.C.data.copy()
            wd = float(cp.mean(W[mem.R.withinA_masks[dslot]]))
            wdog_traj.append(round(wd, 3))
            iso_rec = iso_read("dog", W)
            live_rec = live_read("dog")
            for r in GRADED_READS:
                iso_traj[r].append(_g(iso_rec, r))
                live_traj[r].append(_g(live_rec, r))
            print(f"  [turn T+{turn+1}] consolidate={'ok' if rec else None} w_dog={wd:.1f} "
                  f"iso.depth_hold={_g(iso_rec, 'depth_hold'):.3f} live.depth_hold={_g(live_rec, 'depth_hold'):.3f}",
                  flush=True)
        consolidated = bool(consolidated_rounds > 0)
        w_dog_after = float(cp.mean(W[mem.R.withinA_masks[dslot]]))
        w_cat_after = float(cp.mean(W[mem.R.withinA_masks[cslot]]))
        cat_drift = abs(w_cat_after - w_cat)

        # OFF drift control per read/source (halves of the noise-floor reads; should be ~0)
        half = max(1, a.k_noise // 2)

        def drift_of(events, r):
            vals = [_g(e, r) for e in events]
            return float(np.mean(vals[half:]) - np.mean(vals[:half])) if len(vals) >= 2 else 0.0

        # ── per (read, source) metrics ─────────────────────────────────────────────────────────────────────────────
        per_read = {}
        for r in GRADED_READS:
            for src in SOURCES:
                events = live_noise if src == "live" else iso_noise
                traj = live_traj[r] if src == "live" else iso_traj[r]
                noise_vals = np.asarray([_g(e, r) for e in events], dtype=np.float64)
                noise_std = float(np.std(noise_vals))
                noise_mean = float(np.mean(noise_vals))
                effect = traj[-1] - traj[0]
                snr = abs(effect) / noise_std if noise_std > 1e-9 else (float("inf") if abs(effect) > MOVE_MARGIN else 0.0)
                rises = bool(effect > MOVE_MARGIN)
                monotone = _mono_rel(traj)
                decidable = bool(snr >= a.snr_min)
                # faithfulness measured on the ISOLATED read (deterministic, weight-attributable): cue-specific + the
                # formation-lesion collapses it. rec0 carries the borderline cue/perm/nocue graded reads.
                cue_v = _g(rec0, r); perm_v = _gp(rec0, "graded_perm", r); nocue_v = _gp(rec0, "graded_nocue", r)
                les_v = _g(les_iso, r)
                faithful_specific = bool(cue_v >= FAITHFUL_K * max(perm_v, nocue_v, 1e-9))
                faithful_lesion = bool(abs(les_v) <= LESION_COLLAPSE_FRAC * max(abs(cue_v), 1e-9))
                faithful = bool(faithful_specific and faithful_lesion)
                read_go = bool(decidable and rises and monotone and faithful and inmem_T and moat_byte_id)
                per_read[f"{r}.{src}"] = dict(
                    read=r, source=src, noise_mean=round(noise_mean, 5), noise_std=round(noise_std, 5),
                    noise_vals=[round(float(x), 4) for x in noise_vals],
                    effect=round(effect, 5), snr=round(float(snr), 3), rises=rises, monotone=monotone,
                    decidable=decidable, faithful=faithful, faithful_specific=faithful_specific,
                    faithful_lesion=faithful_lesion, perm=round(perm_v, 5), nocue=round(nocue_v, 5),
                    lesion=round(les_v, 5), traj=[round(x, 4) for x in traj], read_go=read_go)

        # headline: the best ISOLATED read (depth_hold preferred) vs the LIVE depth_hold baseline
        iso_keys = [k for k in per_read if per_read[k]["source"] == "iso"]
        best = max(iso_keys, key=lambda k: (per_read[k]["read_go"], per_read[k]["snr"]))
        bp = per_read[best]
        base = per_read["depth_hold.live"]

        v = Verdict(f"A snapshot-ISOLATED surfaced read makes the D5 recall-strength effect DECIDABLE (SNR≥{a.snr_min}) "
                    f"AND monotone-rising, where the LIVE read (period-2) is undecidable — best '{best}' (seed {seed})")
        v.disabled("host weight formula", "the strengthening is the substrate's OWN plateau-gated BTSP "
                                          "(fused_btsp_update) via continuous_engine.consolidate_used_memory")
        v.disabled("live-bridge state carryover on the surfaced read", "the ISOLATED read completely resets to the "
                   "clean-rest baseline (the return-to-rest the incomplete reset approximated) before the read — the "
                   "biology's own companion process, so the read is deterministic + weight-attributable")
        v.require("weak-store-completes-T", inmem_T, expect=True,
                  note="the weak encode COMPLETES via the BINARY moat gate (a genuinely usable memory with rise headroom)")
        v.require("moat-byte-identical", moat_byte_id, expect=True,
                  note="the surfaced isolated read is ADDITIVE: the binary gate apical_cue stays the live read == "
                       "production _apical_up_read")
        v.require("cat-never-recalled", cat_never, expect=True,
                  note="cat is a genuine never-formed control (no completion, baseline weight)")
        v.require("consolidation-ran", consolidated, expect=True,
                  note="the between-turn D5 consolidation actually potentiated dog's within-assembly weight")
        v.require("best-isolated-faithful", bp["faithful"], expect=True,
                  note="the isolated read is cue-specific AND collapses under the formation-lesion (moat preserved)")
        v.require("best-isolated-decidable", bp["decidable"], expect=True,
                  note=f"the isolated read's repeated-read noise is <= 1/{a.snr_min} of the effect (≈0 → decidable)")
        v.require("best-isolated-monotone", bp["monotone"], expect=True,
                  note="the isolated read rises monotonically with use (conversation-visible)")
        v.reaches("best-isolated-rises", bp["traj"][0], bp["traj"][-1],
                  note="the stabilized read rises after consolidation")
        v.control(f"isolated noise_std vs live noise_std ({bp['read']})",
                  treatment=base["noise_std"], control=bp["noise_std"], min_separation=0.0,
                  note="the live read carries the period-2 noise the isolated read removes")
        go = bool(bp["read_go"] and moat_byte_id and inmem_T and consolidated and cat_never)
        decided = v.decide(go=go)
        result["verdict"] = decided; result["verdict_status"] = decided["status"]
        result["attributable"] = attributable_to(
            f"[s{seed}] {bp['read']} noise removed by isolation: LIVE std vs ISOLATED std",
            base["noise_std"], bp["noise_std"])

        checks = dict(
            instrument_valid=True, moat_byte_id=moat_byte_id, inmem_T=inmem_T, cat_never=cat_never, te=te,
            apical_cue_sel=round(float(rec0["apical_cue"]), 4), consolidated=consolidated,
            consolidated_rounds=consolidated_rounds, best_isolated=best, best_read_go=bp["read_go"], best_snr=bp["snr"],
            live_depth_hold=dict(noise_std=base["noise_std"], effect=base["effect"], snr=base["snr"],
                                 rises=base["rises"], monotone=base["monotone"], decidable=base["decidable"],
                                 read_go=base["read_go"]),
            iso_depth_hold=dict(noise_std=per_read["depth_hold.iso"]["noise_std"],
                                effect=per_read["depth_hold.iso"]["effect"], snr=per_read["depth_hold.iso"]["snr"],
                                rises=per_read["depth_hold.iso"]["rises"], monotone=per_read["depth_hold.iso"]["monotone"],
                                decidable=per_read["depth_hold.iso"]["decidable"],
                                read_go=per_read["depth_hold.iso"]["read_go"]),
            per_read=per_read, wdog_traj=wdog_traj, w_dog_before=round(w_dog_before, 3),
            w_dog_after=round(w_dog_after, 3), cat_drift=round(cat_drift, 4),
            assembly_sizes=mem.assembly_sizes, warm_reads=a.warm_reads, k_noise=a.k_noise, n_turns=a.n_turns,
            n_episodes=a.n_episodes, snr_min=a.snr_min)
        result["checks"] = checks
        CE.forget_session(cache_key)
        del mem, org
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e); result["verdict_status"] = "ERROR"
        traceback.print_exc()
    finally:
        os.environ["BRAIN_D5_CONSOLIDATE"] = "0"

    result["elapsed_s"] = round(time.time() - t0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    status = result.get("verdict_status")
    print("=" * 118)
    print(f"[d5-stab] seed={seed} VERDICT: {status}")
    if result.get("checks", {}).get("instrument_valid"):
        c = result["checks"]
        print(f"    moat_byte_id={c['moat_byte_id']} inmem_T={c['inmem_T']} consolidated={c['consolidated']} te={c['te']} "
              f"| BEST={c['best_isolated']} SNR={c['best_snr']} read_go={c['best_read_go']}")
        print(f"    {'read.source':20s} {'noise_std':>9s} {'effect':>9s} {'SNR':>8s} {'rise':>5s} {'mono':>5s} "
              f"{'faith':>5s} {'GO':>4s}")
        for k in per_read:
            p = c["per_read"][k]
            print(f"    {k:20s} {p['noise_std']:9.4f} {p['effect']:9.4f} {p['snr']:8.2f} {str(p['rises']):>5s} "
                  f"{str(p['monotone']):>5s} {str(p['faithful']):>5s} {str(p['read_go']):>4s}")
    print(f"[d5-stab] wrote {out_path}")
    print("=" * 118)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--k-noise", type=int, default=8, dest="k_noise", help="repeated reads for the empirical noise floor")
    ap.add_argument("--n-turns", type=int, default=5, dest="n_turns", help="learn-through-use consolidation ticks")
    ap.add_argument("--n-episodes", type=int, default=1, dest="n_episodes", help="reactivation episodes per tick")
    ap.add_argument("--warm-reads", type=int, default=4, dest="warm_reads", help="live warm reads before the noise floor")
    ap.add_argument("--train-events", type=int, default=8, dest="train_events")
    ap.add_argument("--te-grid", type=str, default="5,6,7,8,10", dest="te_grid",
                    help="adaptive weak-encode sweep (lowest-completing headroom store; step-6's criterion → same store)")
    ap.add_argument("--snr-min", type=float, default=SNR_MIN, dest="snr_min")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    seeds = a.seeds if a.seeds else [a.seed]
    _, backend = get_backend()
    all_results = {}
    keys = [f"{r}.{s}" for r in GRADED_READS for s in SOURCES]
    tally = {k: dict(read_go=0, decidable=0, rises=0, monotone=0, faithful=0) for k in keys}
    valid_flags = []; moat_flags = []
    for seed in seeds:
        out_path = (Path(a.out) if len(seeds) == 1 else Path(a.out).parent / f"seed{seed}.json")
        res = run_one(seed, a, backend, out_path)
        all_results[seed] = res
        c = res.get("checks", {})
        valid = bool(c.get("instrument_valid") and c.get("inmem_T") and c.get("consolidated") and c.get("cat_never"))
        valid_flags.append(valid); moat_flags.append(bool(c.get("moat_byte_id")))
        if c.get("per_read"):
            for k in keys:
                p = c["per_read"].get(k, {})
                for m in ("read_go", "decidable", "rises", "monotone", "faithful"):
                    tally[k][m] += int(bool(p.get(m)))

    n = len(seeds)
    print("\n" + "#" * 118)
    print(f"[d5-stab] {n}-SEED SUMMARY (k_noise={a.k_noise} n_turns={a.n_turns} snr_min={a.snr_min}) seeds={seeds}")
    print(f"[d5-stab] instrument-valid {sum(valid_flags)}/{n} | moat-byte-id {sum(moat_flags)}/{n}")
    print(f"    {'read.source':20s} {'read_go':>8s} {'decidable':>10s} {'rises':>6s} {'monotone':>9s} {'faithful':>9s}")
    order = sorted(keys, key=lambda x: (tally[x]["read_go"], tally[x]["decidable"]), reverse=True)
    for k in order:
        t = tally[k]
        print(f"    {k:20s} {t['read_go']:>6d}/{n} {t['decidable']:>8d}/{n} {t['rises']:>4d}/{n} "
              f"{t['monotone']:>7d}/{n} {t['faithful']:>7d}/{n}")
    best_iso = max([k for k in keys if k.endswith(".iso")], key=lambda x: (tally[x]["read_go"], tally[x]["decidable"]))
    base = tally["depth_hold.live"]
    print(f"\n[d5-stab] LIVE baseline depth_hold.live: read_go={base['read_go']}/{n} decidable={base['decidable']}/{n} "
          f"rises={base['rises']}/{n} monotone={base['monotone']}/{n}")
    print(f"[d5-stab] BEST ISOLATED '{best_iso}': read_go={tally[best_iso]['read_go']}/{n} "
          f"decidable={tally[best_iso]['decidable']}/{n} monotone={tally[best_iso]['monotone']}/{n}")
    n_valid = sum(valid_flags)
    # GO: the ISOLATED read earns read_go on EVERY instrument-valid seed, strictly beats the LIVE baseline, and the moat
    # is byte-identical on every valid seed (>=1 valid seed required; a 6-seed run needs >=5 valid).
    go_overall = bool(n_valid >= max(1, n - 1) and tally[best_iso]["read_go"] == n_valid
                      and tally[best_iso]["read_go"] > base["read_go"] and sum(moat_flags) == n_valid)
    print(f"[d5-stab] DE-RISK VERDICT: {'GO' if go_overall else 'NEGATIVE / see table'} "
          f"(GO needs the isolated read read_go on all valid seeds, > the live baseline, moat byte-id)")
    print("#" * 118)
    summ_path = Path(a.out).parent / f"summary_{n}seed.json"
    summ_path.parent.mkdir(parents=True, exist_ok=True)
    summ_path.write_text(json.dumps({
        "seeds": seeds, "k_noise": a.k_noise, "n_turns": a.n_turns, "snr_min": a.snr_min, "backend": backend,
        "valid_flags": valid_flags, "moat_flags": moat_flags, "tally": tally, "best_isolated": best_iso,
        "baseline_read_go": base["read_go"], "best_read_go": tally[best_iso]["read_go"], "go_overall": go_overall,
        "params": vars(a), "per_seed": {str(s): all_results[s].get("checks", {}) for s in seeds}}, indent=2, default=str))
    print(f"[d5-stab] wrote {summ_path}")
    return 0 if go_overall else 1


if __name__ == "__main__":
    sys.exit(main())
