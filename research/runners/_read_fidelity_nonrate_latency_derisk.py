"""READ-FIDELITY de-risk: a NON-RATE (first-spike-LATENCY) read primitive vs the CURRENT mean-firing-RATE read,
on the surprise->episodic crux (`research/findings/2026-08-27-onebrain-surprise-episodic-crossedge-UNDEFINED.md`).

THE CRUX THIS ATTACKS. That finding's F2 arm (surprise -> source_provenance's `prov_generated` cross-edge, the
audit-sanctioned stand-in for the still-Group-C-deferred `d5_episodic`) reads a SIGNED MARGIN
`rate_generated - rate_perceived` -- a mean-firing-RATE read (`cp_firing_states` thresholded and averaged over
`RECALL_STEPS=100` steps, `N_READS=8` reads). That margin misses its pre-registered floor on ALL 6 seeds even
after a 3x learning-rate raise at an already-tripled episode budget (weight up to 155x its seed value, margin
essentially flat) -- and the finding's own read section names the diagnosis: **Sanzeni, Histed & Brunel (2020,
PLOS Comput Biol 16(9):e1008165, "Response nonlinearities in networks of spiking neurons")**, eq. 13, show a
network's rate-response transfer function goes SUBLINEAR and saturates with coupling strength because of the
refractory period -- exactly the shape here (weight scales ~2x, margin does not). The finding explicitly names
the untested fix: "a change to the read protocol ... to stay in the pre-saturation linear regime" (line 155-156
of that finding).

THE NON-RATE READ, AND WHY THIS ONE (biology, not an abstract trick). Four temporal-code candidates exist
(first-spike latency; phase/theta-referenced; rank-order; onset-transient/derivative). This crux's own signal
structure picks FIRST-SPIKE LATENCY: the cross-edge drives a mismatch-locked, single CONTRADICT-trial hold (no
oscillatory reference exists anywhere in this pairing to phase-lock against, ruling out (b); the comparison is
between exactly TWO pools -- generated vs perceived -- not a multi-item competition, so a RANK over >2 items
(c) degenerates to the same 2-way latency comparison anyway; and the read window is a single sustained hold, not
a transient onset event to differentiate (d)). First-spike latency is also the textbook answer for EXACTLY this
symptom (a rate code compressing near saturation): Gollisch & Meister (2008, Science 319:1108-1111,
"Rapid Neuronal Coding of Visual Stimuli") and Thorpe, Delorme & Van Rullum (2001, Neural Networks 14:715-725)
show spike TIMING carries graded stimulus information through a MUCH wider dynamic range than a windowed rate
because near/just-above threshold, small drive increases produce large monotonic decreases in time-to-first-spike
-- the opposite regime from a rate code, which only differentiates while spike counts are still resolvable and
compresses once the window fills up. Sanzeni et al. 2020's own mechanism (refractory-period-driven saturation) is
a property of SUSTAINED firing count, not of WHEN the first spike in a window occurs -- so a first-spike-latency
read is not merely "a different number", it targets the specific physical process the finding blamed.

THE READ (brain-based; spike TIMES only, no host-computed feature). For each of `prov_generated`/`prov_perceived`,
record the per-NEURON first step (0-based) at which `cp_firing_states` goes True within the (post-pre-phase)
RECALL_STEPS window; a neuron that never fires is right-censored at RECALL_STEPS (the slowest possible latency,
not a discard). The POPULATION onset is `mean(first_step) / RECALL_STEPS` -- an "onset FRACTION" in [0, 1],
chosen so the read lives on the same normalized [0,1]-ish scale as the rate margin (a fraction of steps firing)
and is thus visually/order-of-magnitude comparable, while remaining a genuinely different STATISTIC of the
IDENTICAL underlying `cp_firing_states` stream (computed from a single simulated trajectory that ALSO produces
the rate margin -- see `_drive2` below -- so rate-vs-latency is never confounded by two separately-simulated
runs). `margin_lat = onset_perceived - onset_generated` (positive = perceived is SLOWER than generated, i.e. the
same "shifted toward GENERATED" direction as the rate margin's `rate_generated - rate_perceived > 0`).

WHAT COUNTS AS "LIFTS THE CRUX" HERE (a scale-free floor, not a re-tuned magnitude). Reusing the rate arm's
absolute floor (`F2_INTACT_FLOOR=0.010`) on a differently-scaled statistic would be an unjustified magnitude
transplant -- exactly the "floor-tuning game" this codebase's own norms forbid. Instead this run measures, per
read-kind, the delta's own SEM across the N_READS=8 samples and requires `z = mean/sem >= Z_FLOOR` (a
STATISTICAL-SIGNIFICANCE floor, scale-free by construction, applied IDENTICALLY to the rate arm and the latency
arm so neither gets an easier bar) -- see `_stats`/`Z_FLOOR`. The lesion-attribution criterion is unchanged from
the original crux (`F2_LESION_RATIO=0.34`, a dimensionless ratio, already scale-free).

ANTI-CHEAT (the read must be genuinely from PER-NEURON spike-timing IDENTITY, not a freed host path or a
coincidental artifact of window shape). A FIXED, seed-keyed random permutation reassigns which neurons in
prov_generated union prov_perceived count as "generated" vs "perceived" (pool SIZES preserved), computed ONCE
per seed and applied to the SAME captured raster the real read uses. If the latency margin's significant shift
survives this shuffle, the read is not actually keying off which pool's neurons are which -- it would be reading
some pool-identity-independent artifact of the drive/window. `shuffle_collapses` requires z_shuffled < Z_FLOOR
on the intact edge.

DE-RISK ONLY -- no production wiring, no `sim/` edit, no default flip. Reuses `SurpriseEpisodicPool` from the
already-committed crossedge runner VERBATIM (same build/train/lesion machinery; this file adds ONLY the read),
so the rate-vs-latency comparison shares the identical substrate, training protocol, and lesion event as the
finding it is attacking -- no retraining-noise confound. numpy CPU throughout; pool-runnable (no GPU, no
gitignored asset).

Run:
  SIM_BACKEND=numpy python -m research.runners._read_fidelity_nonrate_latency_derisk --seeds 42 --smoke
  SIM_BACKEND=numpy python -m research.runners._read_fidelity_nonrate_latency_derisk \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_read_fidelity_nonrate_latency_derisk_6seed.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU only -- never touch the GPU
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sim.backend import to_host
from tools.lab import attributable_to
from research.runners._onebrain_integration_surprise_episodic_crossedge import (
    SurpriseEpisodicPool, RECALL_STEPS, N_READS, PRE_STEPS, EPISODE_DRIVE_PA,
    F2_INTACT_FLOOR, F2_LESION_RATIO, CROSS_EDGE_LR, N_EPISODES, HMAX, CUE_PA, CTX_DRIVE_PA,
)

# ---- this run's own pre-registered constants (declared BEFORE any measurement) ----
Z_FLOOR = 2.0   # scale-free significance floor: delta must clear 2 SEM (both read-kinds, same bar)


def _stats(samples):
    """mean/sem/z for a 1-D sample list -- the scale-free replacement for an absolute-magnitude floor."""
    arr = np.asarray(samples, dtype=np.float64)
    n = int(arr.size)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else float("inf")
    if sem > 0:
        z = mean / sem
    else:
        z = float("inf") if mean != 0 else 0.0
    return {"mean": mean, "std": std, "sem": sem, "n": n, "z": float(z)}


def _onset_fraction(raster, mask, steps):
    """raster: (steps, n_union) bool. mask: bool (n_union,) selecting this pool's columns (real OR shuffled
    identity). Returns the population's mean first-spike STEP (right-censored at `steps` for neurons that never
    fire within the window), normalized to [0, 1] -- a first-spike-LATENCY code (Thorpe et al. 2001;
    Gollisch & Meister 2008), not a spike-COUNT/RATE code."""
    sub = raster[:, mask]
    if sub.shape[1] == 0:
        return 1.0
    fired_any = sub.any(axis=0)
    first = np.where(fired_any, sub.argmax(axis=0), steps)   # argmax = index of the FIRST True per column
    return float(np.mean(first) / steps)


class ReadFidelityPool(SurpriseEpisodicPool):
    """Adds a raster-capturing drive (`_drive2`) and a paired rate+latency F2 read (`f2_reads`) on top of the
    UNCHANGED `SurpriseEpisodicPool` (build/train/lesion machinery reused verbatim, byte-identical to the
    parent's own protocol)."""

    def _drive2(self, pairs, steps, read, pre_pairs=None, pre_steps=0, latency_union=None):
        """VERBATIM mirror of the parent's `_drive` (same pre-phase, same current injection, same
        `enable_hebbian_learning=False` throughout -- F2 reads never learn in either the original or here) --
        with ONE addition: if `latency_union` (an index array) is given, ALSO capture the per-step firing raster
        over those neurons, from the SAME simulated trajectory that produces the rate `read` dict. This is why
        the rate margin and the latency margin can never disagree because of two different simulated runs: they
        are two different STATISTICS of one identical spike stream."""
        b, xp = self.b, self.xp
        b.core_config.enable_hebbian_learning = False
        if pre_pairs is not None and pre_steps > 0:
            precur = xp.zeros(b.core_config.num_neurons, dtype=xp.float32)
            for idx, pa in pre_pairs:
                precur[xp.asarray(idx)] = xp.float32(pa)
            for _ in range(pre_steps):
                b.cp_external_input_current[:] = precur
                b._run_one_simulation_step()
        cur = xp.zeros(b.core_config.num_neurons, dtype=xp.float32)
        for idx, pa in pairs:
            cur[xp.asarray(idx)] = xp.float32(pa)
        acc = {k: 0.0 for k in read}
        raster = None
        lat_idx = None
        if latency_union is not None:
            raster = np.zeros((steps, latency_union.size), dtype=bool)
            lat_idx = xp.asarray(latency_union)
        for t in range(steps):
            b.cp_external_input_current[:] = cur
            b._run_one_simulation_step()
            fs = b.cp_firing_states
            for k, idx in read.items():
                acc[k] += float(to_host(fs[xp.asarray(idx)].astype(xp.float64).sum())) / idx.size
            if raster is not None:
                raster[t] = np.asarray(to_host(fs[lat_idx]), dtype=bool)
        b.cp_external_input_current[:] = 0.0
        b.core_config.enable_hebbian_learning = False
        rate = {k: v / steps for k, v in acc.items()}
        return rate, raster

    def f2_reads(self, shuf_gen_mask):
        """N_READS (base, held) UNPAIRED reads (matches the original F2's own protocol shape), from each of
        which BOTH the rate margin and the (real-identity + shuffled-identity) latency margins are extracted.
        Returns raw per-read delta samples (held - base, per read) for all three, so `_stats` can compute a
        genuine SEM rather than a difference of two already-averaged means."""
        ix = self.ix
        ep_idx = ix["episode"][self.ambig_pattern]
        read = {"gen": ix["prov_generated"], "perc": ix["prov_perceived"]}
        n_gen = ix["prov_generated"].size
        union = np.concatenate([ix["prov_generated"], ix["prov_perceived"]])
        real_gen_mask = np.zeros(union.size, dtype=bool)
        real_gen_mask[:n_gen] = True

        def _one_read(hold):
            self._hard_reset()
            pairs = [(ep_idx, EPISODE_DRIVE_PA)]
            pre, pre_steps = None, 0
            if hold:
                pairs = pairs + self._contradict_pairs()
                pre, pre_steps = self._cue_pre_pairs(), PRE_STEPS
            rate, raster = self._drive2(pairs, RECALL_STEPS, read=read, pre_pairs=pre, pre_steps=pre_steps,
                                        latency_union=union)
            r_margin = rate["gen"] - rate["perc"]
            onset_gen = _onset_fraction(raster, real_gen_mask, RECALL_STEPS)
            onset_perc = _onset_fraction(raster, ~real_gen_mask, RECALL_STEPS)
            l_margin = onset_perc - onset_gen
            s_onset_gen = _onset_fraction(raster, shuf_gen_mask, RECALL_STEPS)
            s_onset_perc = _onset_fraction(raster, ~shuf_gen_mask, RECALL_STEPS)
            ls_margin = s_onset_perc - s_onset_gen
            return r_margin, l_margin, ls_margin

        d_rate, d_lat, d_lat_shuf = [], [], []
        for _ in range(N_READS):
            rb, lb, lsb = _one_read(False)
            rh, lh, lsh = _one_read(True)
            d_rate.append(rh - rb)
            d_lat.append(lh - lb)
            d_lat_shuf.append(lsh - lsb)
        return {"d_rate": d_rate, "d_lat": d_lat, "d_lat_shuf": d_lat_shuf}


def _shuffle_mask(seed, n_gen, n_perc):
    """ONE fixed, seed-keyed permutation of pool identity (sizes preserved), reused for BOTH the intact and
    lesioned arms so the anti-cheat is well-defined across the whole F2 comparison. Distinct RNG offset from
    every other seeded draw in this module family (`_assign_blocks` uses *104729+17)."""
    rng = np.random.default_rng(int(seed) * 65599 + 41)
    n = n_gen + n_perc
    perm = rng.permutation(n)
    mask = np.zeros(n, dtype=bool)
    mask[perm[:n_gen]] = True
    return mask


def _arm_verdict(intact_stats, lesion_stats, label):
    floor_ok = bool(intact_stats["mean"] > 0 and intact_stats["z"] >= Z_FLOOR)
    denom = abs(intact_stats["mean"])
    lesion_ok = bool(denom > 0 and abs(lesion_stats["mean"]) < F2_LESION_RATIO * denom)
    frac = attributable_to(label, intact_stats["mean"], lesion_stats["mean"])
    return {"floor_ok": floor_ok, "lesion_ok": lesion_ok,
            "frac_attributable": (None if frac is None else float(frac)),
            "PASS": bool(floor_ok and lesion_ok)}


def run_seed(seed):
    t0 = time.time()
    pool = ReadFidelityPool(seed)
    traj = pool.train()
    emg_grew = bool(traj[-1]["w"] > 5 * 0.05)
    emg_specific = bool(abs(traj[-1]["w_other"] - 0.05) < 0.03)

    n_gen = pool.ix["prov_generated"].size
    n_perc = pool.ix["prov_perceived"].size
    shuf_mask = _shuffle_mask(seed, n_gen, n_perc)

    intact = pool.f2_reads(shuf_mask)

    data = np.asarray(to_host(pool.b.cp_connections.data)).copy()
    data[pool.masks["surprise->provgen"]] = 0.0
    pool.b.cp_connections.data = pool.xp.asarray(data, dtype=pool.b.cp_connections.data.dtype)
    lesioned = pool.f2_reads(shuf_mask)

    rate_i, rate_l = _stats(intact["d_rate"]), _stats(lesioned["d_rate"])
    lat_i, lat_l = _stats(intact["d_lat"]), _stats(lesioned["d_lat"])
    latshuf_i, latshuf_l = _stats(intact["d_lat_shuf"]), _stats(lesioned["d_lat_shuf"])

    rate_arm = _arm_verdict(rate_i, rate_l, "F2 rate margin (delta_intact vs delta_lesion)")
    lat_arm = _arm_verdict(lat_i, lat_l, "F2 latency margin (delta_intact vs delta_lesion)")
    shuffle_collapses = bool(latshuf_i["z"] < Z_FLOOR)

    return {
        "seed": int(seed), "elapsed_s": round(time.time() - t0, 1),
        "cue_concept": pool.cue_c, "assert_concept": pool.assert_cp,
        "final_weight_trained_block": float(traj[-1]["w"]), "final_weight_other_blocks": float(traj[-1]["w_other"]),
        "emergence_grew_from_near_zero": emg_grew, "emergence_other_blocks_stayed_near_seed": emg_specific,
        "rate": {"intact": rate_i, "lesion": rate_l, **rate_arm},
        "latency": {"intact": lat_i, "lesion": lat_l, **lat_arm},
        "latency_shuffled_anticheat": {"intact": latshuf_i, "lesion": latshuf_l,
                                       "shuffle_collapses": shuffle_collapses},
        "PASS": bool(lat_arm["PASS"] and shuffle_collapses),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--smoke", action="store_true", help="1 seed indicator")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]

    runs = []
    for s in seeds:
        r = run_seed(s)
        runs.append(r)
        print(f"[seed {s}] ({r['elapsed_s']}s) block(c={r['cue_concept']},c'={r['assert_concept']}) "
              f"w={r['final_weight_trained_block']:.2f} w_other={r['final_weight_other_blocks']:.3f} | "
              f"RATE   delta_i={r['rate']['intact']['mean']:+.4f}(z={r['rate']['intact']['z']:.1f}) "
              f"delta_l={r['rate']['lesion']['mean']:+.4f} floor={r['rate']['floor_ok']} "
              f"lesion_ok={r['rate']['lesion_ok']} PASS={r['rate']['PASS']} | "
              f"LATENCY delta_i={r['latency']['intact']['mean']:+.4f}(z={r['latency']['intact']['z']:.1f}) "
              f"delta_l={r['latency']['lesion']['mean']:+.4f} floor={r['latency']['floor_ok']} "
              f"lesion_ok={r['latency']['lesion_ok']} PASS={r['latency']['PASS']} | "
              f"shuffle_z={r['latency_shuffled_anticheat']['intact']['z']:.1f} "
              f"collapses={r['latency_shuffled_anticheat']['shuffle_collapses']}",
              flush=True)

    n_rate_go = sum(r["rate"]["PASS"] for r in runs)
    n_lat_go = sum(r["latency"]["PASS"] for r in runs)
    n_shuf_collapse = sum(r["latency_shuffled_anticheat"]["shuffle_collapses"] for r in runs)
    all_go_raw = bool(n_lat_go == len(runs) and n_shuf_collapse == len(runs)) and not args.smoke

    dec, preconditions = None, []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("read_fidelity_nonrate_latency_derisk")
        Vd.require("latency_clears_scale_free_floor_and_lesion_attributable",
                   1 if all(r["latency"]["PASS"] for r in runs) else 0, expect=lambda x: x >= 1,
                   note="the latency margin must be significantly nonzero (z>=Z_FLOOR) AND vanish under lesion "
                        "(|delta_lesion| < F2_LESION_RATIO*|delta_intact|) on every seed")
        Vd.require("shuffle_control_collapses",
                   1 if all(r["latency_shuffled_anticheat"]["shuffle_collapses"] for r in runs) else 0,
                   expect=lambda x: x >= 1,
                   note="scrambling which neurons count as generated/perceived must destroy the latency "
                        "significance (z<Z_FLOOR) on every seed -- proves the read keys off genuine pool identity")
        Vd.require("emergence_grew_from_near_zero", 1 if all(r["emergence_grew_from_near_zero"] for r in runs) else 0,
                   expect=lambda x: x >= 1, note="the reused cross-edge trained normally (sanity on the shared substrate)")
        Vd.require("anti_cheat_random_assignment", 1 if len(set((r["cue_concept"], r["assert_concept"])
                   for r in runs)) > 1 else 0, expect=lambda x: x >= 1,
                   note="the per-seed block pair must actually vary (inherited from the parent runner)")
        dec = Vd.decide(all_go_raw, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    verdict_status = dec.get("status") if dec else None
    all_go = all_go_raw if dec is None else bool(dec.get("go"))
    if verdict_status == "UNDEFINED":
        tag = "UNDEFINED"
    elif args.smoke:
        tag = "SMOKE-GO (1-seed indicator)" if runs[0]["latency"]["PASS"] else "NO-GO/PARTIAL (1-seed indicator)"
    else:
        tag = "GO" if all_go_raw else "NO-GO/PARTIAL"
    verdict = (f"{tag} -- NON-RATE (first-spike-latency, normalized onset-fraction) read vs the CURRENT "
               f"mean-rate read on the surprise->episodic F2 crux "
               f"(research/findings/2026-08-27-onebrain-surprise-episodic-crossedge-UNDEFINED.md): "
               f"rate arm {n_rate_go}/{len(runs)} PASS (reproduces the original crux's own floor-miss shape), "
               f"latency arm {n_lat_go}/{len(runs)} PASS (z>=Z_FLOOR={Z_FLOOR} AND lesion-attributable), "
               f"shuffle-identity anti-cheat collapses on {n_shuf_collapse}/{len(runs)}. "
               f"Same trained cross-edge, same lesion event, same simulated trajectory feeds BOTH reads (no "
               f"retraining-noise confound). Biology: first-spike latency (Thorpe 2001; Gollisch & Meister 2008) "
               f"targets the SAME saturating regime Sanzeni/Histed/Brunel 2020 attribute to refractory-period-"
               f"driven rate compression, without re-tuning the floor's units (Z_FLOOR is scale-free, applied "
               f"identically to both arms)."
               + (f" UNDEFINED, NOT a validated verdict either way: {len(dec.get('undefined_reasons', []))} "
                  f"precondition(s) unmet -- {'; '.join(dec.get('undefined_reasons', []))}."
                  if verdict_status == "UNDEFINED" else ""))

    payload = {"probe": "read_fidelity_nonrate_latency_derisk", "verdict": verdict, "GO": all_go,
               "n_seeds": len(runs), "n_rate_pass": n_rate_go, "n_latency_pass": n_lat_go,
               "n_shuffle_collapse": n_shuf_collapse, "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
               "preconditions": preconditions,
               "config": {"Z_FLOOR": Z_FLOOR, "recall_steps": RECALL_STEPS, "n_reads": N_READS,
                          "pre_steps": PRE_STEPS, "episode_drive_pa": EPISODE_DRIVE_PA,
                          "f2_intact_floor_original_rate_units": F2_INTACT_FLOOR, "f2_lesion_ratio": F2_LESION_RATIO,
                          "cross_edge_hebbian_lr": CROSS_EDGE_LR, "n_episodes": N_EPISODES,
                          "hebbian_max_weight": HMAX, "cue_pa": CUE_PA, "ctx_drive_pa": CTX_DRIVE_PA,
                          "shuffle_seed_formula": "seed*65599+41"},
               "mechanism": ("Reuses SurpriseEpisodicPool (build/train/lesion) VERBATIM from the committed "
                             "crossedge runner. Adds a raster-capturing drive (`_drive2`) that extracts BOTH a "
                             "mean-RATE margin (rate_generated - rate_perceived, the original crux's own read) "
                             "AND a first-spike-LATENCY margin (onset_perceived - onset_generated, a normalized "
                             "onset-FRACTION in [0,1] per pool, right-censored at the window length) from the "
                             "SAME simulated trajectory. A seed-fixed shuffle of pool identity (sizes preserved) "
                             "re-extracts the latency margin from the identical raster as an anti-cheat: if the "
                             "shift survives scrambled identity, the read was not keyed to genuine per-neuron "
                             "spike-timing membership."),
               "biology": ("Thorpe, Delorme & Van Rullen 2001 (Neural Networks 14:715-725) and Gollisch & "
                           "Meister 2008 (Science 319:1108-1111): first-spike-latency codes carry graded "
                           "stimulus information through a wide dynamic range near/above threshold, the OPPOSITE "
                           "saturation regime from a windowed rate code (Sanzeni, Histed & Brunel 2020, PLOS "
                           "Comput Biol 16(9):e1008165, eq. 13: refractory-period-driven sublinear rate "
                           "saturation with coupling strength -- the diagnosed cause of the F2 floor-miss)."),
               "scaffold_residuals": ["the onset-fraction normalization (dividing by RECALL_STEPS) is a host "
                                      "choice of comparison SCALE, not a computed feature -- the underlying "
                                      "quantity (per-neuron first-spike step) is read directly from "
                                      "cp_firing_states",
                                      "N_READS=8 unpaired base/held samples per condition -- inherited from the "
                                      "parent runner's own calibration, not independently re-derived here",
                                      "same host-curated training schedule / topology as the parent crossedge "
                                      "runner (declared there, unchanged)"],
               "runs": runs}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[READ-FIDELITY] VERDICT: {verdict}\n" + "=" * 100, flush=True)
    return 0 if (all_go or args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
