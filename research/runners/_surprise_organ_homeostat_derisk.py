"""Surprise organ — a per-pool HOMEOSTATIC PREDICTION-GAIN companion closes the read-precision residual.

WHAT THIS CLOSES. The heterogeneous GNW-bus organ de-risk (`_gnw_bus_heterogeneous_organ_derisk`, 6/6 GO,
`2026-08-13-gnw-bus-heterogeneous-organ-GO.md`) named ONE separate-axis residual: the production surprise/familiarity
monitor's single-read confirm precision, `het_vote_rate` mean 0.9375 (3 seeds 8/8, 3 seeds 44/100/101 7/8). On the
1/8-per-seed marginal edge a genuinely FAMILIAR concept reads just ABOVE the surprise threshold, so the organ
WITHHOLDS its vote and the substrate (correctly) abstains — a stronger moat, but end-to-end parity with host recall
is lost. This runner attacks the surprise organ's OWN precision boundary.

THE ROOT CAUSE (measured, not assumed). Reading the surprise organ's per-block confirm rates directly, the residual
is NOT a global gain miss — it is a PER-BLOCK collapse of the top-down prediction RECALL. On the marginal block the
cue->patient_expected recall is near-silent (seed 101 block 4: recall 0.58 Hz vs 6-12 Hz on the other blocks; seed
100 block 2: recall 0.00 Hz), so the FS/PV prediction pool delivers almost NO subtractive inhibition to that block,
and the block's asserted-patient excitation fires UN-cancelled at contradict level (4-6 Hz) even though the assertion
is FAMILIAR. The uniform topographic prediction weight (0.8) is a fixed CONSTANT standing in for the homeostatic
gain-control the animal runs alongside predictive coding — exactly the wall reframe ("what else does the real system
run alongside this, that we replaced with a constant?"). The proxy is the per-block prediction gain.

THE COMPANION PROCESS (the biology). Predictive-coding error units are PRECISION-WEIGHTED: the gain of the prediction
that cancels an expected input is set by a homeostatic / divisive-normalization control (inhibitory E/I balancing,
Vogels-Sprekeler-Zenke-Ganguli-Gerstner 2011; homeostatic synaptic scaling, Turrigiano 2008; precision as
gain-control, Feldman & Friston 2010; Bastos et al. 2012). Here that companion is realized as a PER-BLOCK
homeostatic prediction-gain equalizer: for each stored (cue-addressable) block, if the CONFIRM error (the surprise
pool's spiking rate when the FAMILIAR patient is asserted) exceeds a low target, the top-down prediction gain
(cue->patient_expected) for THAT block is scaled up until the recalled prediction cancels the familiar assertion.
This equalizes the topographic prediction STRENGTH across blocks so every familiar edge reads reliably BELOW
threshold. The controller nulls a SPIKING error (confirm firing) by adjusting a SYNAPTIC gain — the same kind of
build-time calibration the organ already runs for its firing threshold; additive, NO `sim/` edit.

WHY SPECIFICITY IS PRESERVED BY CONSTRUCTION (the anti-"vote on everything" cheat). The prediction pathway is
TOPOGRAPHIC + block-diagonal: block c's prediction inhibits ONLY surprise block c. So boosting block c's prediction
gain cancels the CONFIRM read for block c (assert==expected==c) but leaves every CONTRADICT / NOVEL read untouched —
those drive a DIFFERENT block j!=c, which block c's prediction never inhibits. Measured: at every gain from 0.8 to
3.0 the contradict/novel rates stay 5-6 Hz UNCHANGED while confirm collapses to ~0. A homeostat that merely cranked a
GLOBAL gain (or inhibited the whole surprise pool) would suppress contradict/novel too and the organ would vote on
everything — the surprise-specificity control below FAILS that cheat.

THE GATE (6 seeds 42/43/44/100/101/102; SIM_BACKEND=numpy; a subthreshold-vs-suprathreshold bifurcation, not
GPU-scale-dependent). Re-runs [N]'s FULL heterogeneous-organ gate with the surprise organ swapped for the
homeostatted one (monkeypatch of the drop-in vote class; every one of [N]'s controls is preserved), and ADDS a
surprise-specificity control:
  * het_vote_rate == 1.0 (8/8) on EVERY seed         -> the read-precision residual CLOSES (was 0.9375).
  * consensus_acc == host_recall_acc (strict parity) -> end-to-end matches host recall (the residual's cost).
  * substrate_combines (ignite-when-voted / abstain-when-withheld) still True (the [N] substrate claim intact).
  * EVERY [N] collapse control still <= chance-ish (single, het-dropped, leave-one-out, disagree, shuffle-off,
    het-organ-lesion, workspace-lesion); reflex survives; het discriminates; moat abstains.
  * SURPRISE-SPECIFICITY (NEW): on a genuinely-NOVEL and a CONTRADICTING assertion the homeostatted organ STILL
    reads surprised (rate >= threshold) on every edge -> it did NOT learn to vote on everything.

DISCIPLINE: reuse-by-import (the production surprise organ + [N]'s gate). Additive, deterministic per seed, NO
`sim/` edit. Run:
  SIM_BACKEND=numpy python -u -m research.runners._surprise_organ_homeostat_derisk --smoke --seed 44
  SIM_BACKEND=numpy python -u -m research.runners._surprise_organ_homeostat_derisk --seeds 42 43 44 100 101 102 \
      --json research/findings/raw/_surprise_organ_homeostat/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# reuse-by-import: the production spiking surprise/familiarity monitor + its circuit read primitives.
from research.runners.surprise_production_organ import SurpriseProductionOrgan
from research.runners._spiking_expectation_rpe_derisk import (
    _hard_reset, _drive_read, _idx, _host, measure_conditions,
)
# reuse-by-import: [N]'s heterogeneous-organ bus gate (we monkeypatch the vote class + re-run its full gate).
import research.runners._gnw_bus_heterogeneous_organ_derisk as HB
from tools.lab import attributable_to, void_if


# ── the per-block homeostatic prediction-gain companion ───────────────────────────────────────────────────────────
def _install_block_gains(bridge, meta, src, dst, gains):
    """Set the TOPOGRAPHIC (block-diagonal) src->dst weights to a PER-BLOCK gain vector `gains[block]` (concept c of
    src -> concept c of dst), zeroing cross-concept edges. Operates on the CSR weight matrix (orientation-robust),
    the per-block generalization of `_spiking_expectation_rpe_derisk._install_block_diagonal`."""
    import scipy.sparse as sp
    src_idx = set(int(i) for i in _idx(bridge, src))
    dst_idx = set(int(i) for i in _idx(bridge, dst))
    src_base = min(src_idx); dst_base = min(dst_idx)
    blk = meta["blk"]
    M = bridge.cp_connections.tocsr()
    indptr = np.asarray(_host(M.indptr)); indices = np.asarray(_host(M.indices))
    data = np.asarray(_host(M.data)).astype(np.float32)
    # orientation: is a CSR row the post (dst) or the pre (src)? (same probe as _install_block_diagonal)
    row_is_dst = row_is_src = 0
    for r in range(M.shape[0]):
        r_in_dst = r in dst_idx; r_in_src = r in src_idx
        if not (r_in_dst or r_in_src):
            continue
        for off in range(int(indptr[r]), int(indptr[r + 1])):
            c = int(indices[off])
            if r_in_dst and c in src_idx:
                row_is_dst += 1
            if r_in_src and c in dst_idx:
                row_is_src += 1
    row_is_post = row_is_dst >= row_is_src
    for r in range(M.shape[0]):
        for off in range(int(indptr[r]), int(indptr[r + 1])):
            c = int(indices[off])
            post, pre = (r, c) if row_is_post else (c, r)
            if pre in src_idx and post in dst_idx:
                sc = (pre - src_base) // blk
                dc = (post - dst_base) // blk
                data[off] = float(gains[dc]) if sc == dc else 0.0
    bridge.cp_connections = sp.csr_matrix((data, indices, indptr), shape=M.shape)


class HomeostaticSurpriseOrgan(SurpriseProductionOrgan):
    """The production surprise organ + a PER-BLOCK homeostatic prediction-gain equalizer (the precision companion).

    After the base build+train, an iterative homeostatic controller measures each stored block's CONFIRM error (the
    surprise pool's spiking rate when the FAMILIAR patient is asserted) and scales UP that block's top-down prediction
    gain (cue->patient_expected) until the error falls to a low target — equalizing the topographic prediction
    strength across blocks so every familiar edge cancels below threshold. Contradict/novel reads (a DIFFERENT block)
    are untouched by construction, so surprise specificity is preserved. The organ's firing threshold is re-calibrated
    on the homeostatted circuit."""

    def __init__(self, seed: int = 42, cue_to_expected_weight: float = 0.8, n_reps: int = 22,
                 hz_target: float = 0.5, gain_eta: float = 0.18, gain_max: float = 3.0, homeo_reps: int = 12):
        super().__init__(seed=seed, cue_to_expected_weight=cue_to_expected_weight, n_reps=n_reps)
        self.hz_target = float(hz_target)      # per-block confirm set-point (well below any contradict/novel)
        self.gain_eta = float(gain_eta)        # homeostatic step: weight per Hz of confirm error over target
        self.gain_max = float(gain_max)        # cap on the per-block prediction gain (no runaway)
        self.homeo_reps = int(homeo_reps)
        self.pred_gains = None                 # per-trained-block cue->expected gain (the equalized precision)
        self.homeo_trace = []                  # per-rep max confirm error (convergence record)
        self.confirm_before = None             # per-block confirm at base gain (the residual, for transparency)
        self.confirm_after = None              # per-block confirm after equalization

    def _confirm_per_block(self) -> np.ndarray:
        """Per-stored-block CONFIRM surprise rate (Hz): drive cue i (prediction phase) then cue i + asserted i, read
        the surprise pool — the organ's OWN read path (identical to `measure_conditions`' confirm branch)."""
        nt = self.meta["n_trained"]
        rates = []
        for i in range(nt):
            _hard_reset(self.bridge)
            r = _drive_read(self.bridge, self.idx_map,
                            {"cue": (i, 600.0), "patient_asserted": (i, 600.0)},
                            60, self.xp, ["surprise"], pre_drives={"cue": (i, 600.0)}, pre_steps=60)
            rates.append(r["surprise"])
        return np.asarray(rates)

    def _homeostat(self):
        """Iterative per-block prediction-gain equalization. Homeostatic inhibitory-balance rule: where the CONFIRM
        error (post-synaptic surprise firing on a familiar assertion) exceeds the target, strengthen that block's
        top-down prediction gain; iterate until every block's familiar read is at target (or the cap is hit)."""
        nt = self.meta["n_trained"]
        base = self.cue_w
        gains = np.full(nt, base, dtype=np.float64)
        self.confirm_before = self._confirm_per_block()
        conf = self.confirm_before
        for _ in range(self.homeo_reps):
            over = np.maximum(0.0, conf - self.hz_target)
            self.homeo_trace.append(float(over.max()))
            if over.max() <= 0.0:
                break
            gains = np.clip(gains + self.gain_eta * over, base, self.gain_max)  # only strengthen (E/I balance)
            _install_block_gains(self.bridge, self.meta, "cue", "patient_expected", gains)
            conf = self._confirm_per_block()
        self.confirm_after = conf
        self.pred_gains = gains
        return gains

    def ensure_built(self):
        if self._built:
            return
        self.bridge, self.cfg, self.meta, self.xp, self.idx_map = self._build_one(lesion=False)
        self._novel_next = self.meta["n_trained"]
        # THE COMPANION PROCESS: equalize per-block prediction gain BEFORE calibrating the threshold.
        self._homeostat()
        # calibrate the confirm-vs-contradict threshold on the HOMEOSTATTED circuit.
        res = measure_conditions(self.bridge, self.cfg, self.idx_map, self.meta, self.xp)
        conf, contra, nov = res["confirm_hz"], res["contradict_hz"], res["novel_hz"]
        self.threshold = 0.5 * (conf + min(contra, nov))
        self.calib = {"confirm_hz": float(conf), "contradict_hz": float(contra), "novel_hz": float(nov),
                      "cue_to_expected_weight": self.cue_w, "homeostat": True,
                      "pred_gain_min": float(self.pred_gains.min()), "pred_gain_max": float(self.pred_gains.max()),
                      "confirm_before_max": float(self.confirm_before.max()),
                      "confirm_after_max": float(self.confirm_after.max())}
        self._built = True


class HomeostaticHeterogeneousOrganVote(HB.HeterogeneousOrganVote):
    """Drop-in for [N]'s `HeterogeneousOrganVote`, but the NON-COMPOSER surprise organ carries the homeostatic
    prediction-gain companion. Everything else (`vote`/`read_hz`/`threshold`/`calib`) is inherited unchanged."""

    def __init__(self, seed: int):
        self.organ = HomeostaticSurpriseOrgan(seed=seed)
        self.organ.ensure_built()


# ── the surprise-specificity control (the anti-"vote on everything" cheat) ────────────────────────────────────────
def surprise_specificity(seed):
    """Confirm the homeostatted organ STILL registers surprise on genuinely-novel and contradicting assertions
    (rate >= threshold on every stored fact) — a homeostat that suppressed the whole surprise pool would FAIL this.
    Returns (novel_registers_frac, contradict_registers_frac, mean_confirm_hz, mean_contra_hz, mean_novel_hz)."""
    o = HomeostaticSurpriseOrgan(seed=seed)
    o.ensure_built()
    res = measure_conditions(o.bridge, o.cfg, o.idx_map, o.meta, o.xp)
    thr = o.threshold
    contra = np.asarray(res["contradict_per"]); nov = np.asarray(res["novel_per"])
    conf = np.asarray(res["confirm_per"])
    return (float(np.mean(nov >= thr)), float(np.mean(contra >= thr)),
            float(conf.mean()), float(contra.mean()), float(nov.mean()),
            float(conf.max()), o)


def run_seed(seed, d_sub, D=256, verbose=True):
    """Re-run [N]'s FULL heterogeneous-organ gate with the homeostatted surprise organ (monkeypatched vote class),
    then ADD the surprise-specificity control. Returns [N]'s result dict augmented with the homeostat fields."""
    # monkeypatch: [N].run_seed constructs `HeterogeneousOrganVote(seed)` at module scope -> our subclass wins.
    _orig = HB.HeterogeneousOrganVote
    HB.HeterogeneousOrganVote = HomeostaticHeterogeneousOrganVote
    try:
        r = HB.run_seed(seed, d_sub, D=D, verbose=False)
    finally:
        HB.HeterogeneousOrganVote = _orig

    nov_reg, contra_reg, conf_hz, contra_hz, novel_hz, conf_max, organ = surprise_specificity(seed)
    r["novel_registers_frac"] = nov_reg
    r["contradict_registers_frac"] = contra_reg
    r["homeostat_confirm_before_max"] = float(organ.confirm_before.max())
    r["homeostat_confirm_after_max"] = float(organ.confirm_after.max())
    r["homeostat_pred_gain_max"] = float(organ.pred_gains.max())
    r["homeostat_pred_gain_min"] = float(organ.pred_gains.min())
    r["homeostat_trace"] = organ.homeo_trace
    r["surprise_specificity_ok"] = bool(nov_reg >= 0.999 and contra_reg >= 0.999)

    # THE CLOSED RESIDUAL: het_vote_rate == 1.0 (8/8) AND end-to-end parity with host recall, WHILE [N]'s
    # substrate-combination + every collapse control + surprise specificity hold.
    r["residual_closed"] = bool(
        abs(r["het_vote_rate"] - 1.0) < 1e-9 and                       # every familiar edge voted (was 0.9375)
        abs(r["consensus_acc"] - r["host_recall_acc"]) < 1e-9 and      # end-to-end == host recall (parity)
        r["substrate_combines"] and                                    # [N]'s substrate claim intact
        r["surprise_specificity_ok"] and                              # novel/contradict STILL register surprise
        r["seed_go"]                                                   # every [N] collapse/moat/discrimination holds
    )
    if verbose:
        print(f"[homeostat seed={seed}] het_vote_rate={r['het_vote_rate']:.3f} (was ~0.875 on 44/100/101) "
              f"| consensus={r['consensus_acc']:.3f} == host={r['host_recall_acc']:.3f} "
              f"| substrate_combines={r['substrate_combines']}", flush=True)
        print(f"    confirm_max {r['homeostat_confirm_before_max']:.2f} -> {r['homeostat_confirm_after_max']:.2f} Hz "
              f"(thr {r['het_threshold_hz']:.2f}) | pred_gain {r['homeostat_pred_gain_min']:.2f}-"
              f"{r['homeostat_pred_gain_max']:.2f} | trace {['%.2f'%t for t in r['homeostat_trace']]}", flush=True)
        print(f"    SPECIFICITY: novel_registers={nov_reg:.3f} contradict_registers={contra_reg:.3f} "
              f"(confirm {conf_hz:.2f} < thr {r['het_threshold_hz']:.2f} <= contra {contra_hz:.2f} / novel {novel_hz:.2f}) "
              f"specificity_ok={r['surprise_specificity_ok']}", flush=True)
        print(f"    [N] controls: seed_go={r['seed_go']} het_dropped={r['het_dropped_acc']:.3f} "
              f"disagree={r['disagree_acc']:.3f} het_organ_lesion={r['het_organ_lesion_acc']:.3f} "
              f"workspace_lesion={r['workspace_lesion_acc']:.3f} reflex={r['reflex_acc']:.3f} moat_ok={r['moat_ok']} "
              f"|| RESIDUAL_CLOSED={r['residual_closed']}", flush=True)
    return r


def run_smoke(seed, d_sub, D=256):
    print(f"[smoke] surprise-organ homeostat (per-block prediction-gain equalizer) on the heterogeneous bus, "
          f"seed={seed}", flush=True)
    r = run_seed(seed, d_sub, D=D, verbose=True)
    ok = r["residual_closed"]
    print(f"\n[smoke] SURPRISE-ORGAN HOMEOSTAT {'CLOSES' if ok else 'DOES NOT CLOSE'} the read-precision residual "
          f"(het_vote_rate {r['het_vote_rate']:.3f}, specificity_ok={r['surprise_specificity_ok']}).", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description="Surprise organ — per-block homeostatic prediction-gain companion.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--d-sub", type=float, default=None, help="per-organ subthreshold drive (default: unanimity N=3)")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_surprise_organ_homeostat/summary.json")
    args = ap.parse_args()

    d_sub = args.d_sub if args.d_sub is not None else HB.D_SUB_UNANIMITY.get(HB.N_ORGANS, 1000.0)
    if args.backend != "auto":
        from sim.backend import get_backend
        get_backend(args.backend)

    if args.smoke:
        return 0 if run_smoke(args.seed, d_sub, D=args.D) else 1

    print(f"[surprise-organ-homeostat] {len(HB.FIRST_EDGES)} first-edges | N_organs={HB.N_ORGANS} "
          f"(1 heterogeneous NON-COMPOSER, homeostatted) d_sub={d_sub:.0f} backend={args.backend}\n", flush=True)

    results = [run_seed(s, d_sub, D=args.D) for s in args.seeds]
    all_closed = all(r["residual_closed"] for r in results)
    n_closed = sum(int(r["residual_closed"]) for r in results)

    def mean(k):
        return float(np.mean([r[k] for r in results]))

    print("\n── attribution (tools.lab.attributable_to): the closed residual vs the surprise organ's read ──", flush=True)
    void_if(mean("consensus_acc") <= 1e-9, "intact consensus is ~0 — nothing to attribute")
    # how much of the confirm read did the homeostat remove? (before-max vs after-max, mean over seeds)
    attributable_to("familiar-read suppression @ the per-block prediction-gain companion",
                    mean("homeostat_confirm_before_max"), mean("homeostat_confirm_after_max"))

    summary = {
        "runner": "_surprise_organ_homeostat_derisk",
        "claim": ("a per-block homeostatic prediction-gain equalizer (the precision companion to the surprise "
                  "organ's predictive-coding read) closes the read-precision residual: het_vote_rate -> 1.0 (8/8) "
                  "and end-to-end parity with host recall, WHILE surprise specificity (novel/contradict still "
                  "register) and every [N] substrate/collapse control hold"),
        "seeds": list(args.seeds), "backend": args.backend,
        "all_residual_closed": all_closed, "n_residual_closed": n_closed, "n_seeds": len(results),
        "mean_het_vote_rate": mean("het_vote_rate"), "mean_consensus_acc": mean("consensus_acc"),
        "mean_host_recall_acc": mean("host_recall_acc"),
        "all_substrate_combines": all(r["substrate_combines"] for r in results),
        "mean_consensus_when_voted": mean("consensus_when_voted"),
        "mean_abstain_when_withheld": mean("abstain_when_withheld"),
        "all_surprise_specificity_ok": all(r["surprise_specificity_ok"] for r in results),
        "mean_novel_registers_frac": mean("novel_registers_frac"),
        "mean_contradict_registers_frac": mean("contradict_registers_frac"),
        "mean_confirm_before_max": mean("homeostat_confirm_before_max"),
        "mean_confirm_after_max": mean("homeostat_confirm_after_max"),
        "mean_het_dropped_acc": mean("het_dropped_acc"), "mean_leaveoneout_worst_acc": mean("leaveoneout_worst_acc"),
        "mean_disagree_acc": mean("disagree_acc"), "mean_shuffle_off_acc": mean("shuffle_off_acc"),
        "mean_het_organ_lesion_acc": mean("het_organ_lesion_acc"),
        "mean_workspace_lesion_acc": mean("workspace_lesion_acc"), "mean_reflex_acc": mean("reflex_acc"),
        "mean_single_organ_acc": mean("single_organ_acc"),
        "all_het_discriminate": all(r["het_discriminates"] for r in results),
        "all_moat_ok": all(r["moat_ok"] for r in results),
        "all_seed_go": all(r["seed_go"] for r in results),
        "per_seed": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    verdict = "GO" if all_closed else ("PARTIAL" if n_closed >= 1 else "NEGATIVE")
    print(f"\n{'='*100}", flush=True)
    print(f"  SURPRISE-ORGAN HOMEOSTAT VERDICT: {verdict}  ({n_closed}/{len(results)} seeds residual-closed)", flush=True)
    print(f"    het_vote_rate={summary['mean_het_vote_rate']:.3f} (was 0.9375) | consensus="
          f"{summary['mean_consensus_acc']:.3f} == host={summary['mean_host_recall_acc']:.3f} (parity)", flush=True)
    print(f"    substrate_combines_all={summary['all_substrate_combines']} "
          f"ignite-when-voted={summary['mean_consensus_when_voted']:.3f} "
          f"abstain-when-withheld={summary['mean_abstain_when_withheld']:.3f}", flush=True)
    print(f"    SPECIFICITY: all_ok={summary['all_surprise_specificity_ok']} "
          f"novel_registers={summary['mean_novel_registers_frac']:.3f} "
          f"contradict_registers={summary['mean_contradict_registers_frac']:.3f} "
          f"| confirm_max {summary['mean_confirm_before_max']:.2f} -> {summary['mean_confirm_after_max']:.2f} Hz", flush=True)
    print(f"    [N] controls: dropped={summary['mean_het_dropped_acc']:.3f} disagree={summary['mean_disagree_acc']:.3f} "
          f"het_organ_lesion={summary['mean_het_organ_lesion_acc']:.3f} "
          f"workspace_lesion={summary['mean_workspace_lesion_acc']:.3f} reflex={summary['mean_reflex_acc']:.3f} "
          f"discriminate_all={summary['all_het_discriminate']} moat_all={summary['all_moat_ok']}", flush=True)
    print(f"    [saved] {args.json}\n{'='*100}", flush=True)
    return 0 if all_closed else 1


if __name__ == "__main__":
    raise SystemExit(main())
