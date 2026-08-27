"""gap#4 / mouth read-SNR — the plan-of-record's ONE genuinely-open lever, staged as a ready-to-run de-risk.

THIS FILE WAS NOT RUN when it was written (2026-08-27 audit/scoping task; the session was kept to code-reading +
RAG per an explicit "no heavy sims" constraint after a recent OOM). It is a READY-TO-RUN orchestrator so idle GPU
compute has a queued job (`feedback_build_derisks_ahead_for_idle_compute`), not a claimed result.

THE WALL (read before touching this — do not re-derive):
  `research/findings/2026-08-19-mouth-substrate-forward-40k-coverage-EXCLUDED-real-credit-limit.md` + the plan of
  record `docs/plans/2026-08-19-deep-credit-plan-of-record.md`. The mouth read-out e-prop learning FORWARD — the
  margin that drives the local three-factor rule when computed by the ACTUAL Izhikevich substrate's graded
  conductance read (`BatchedSubstrateReadout` in `_wkv_mouth_readout_eprop_batched_substrate_derisk.py`) — plateaus
  at `sub_learned_recov_mean` ~0.34-0.37 regardless of coverage (8k or 40k training positions), while a
  matched-coverage host-linear-proxy forward (`W@h+head_b`) reaches ~0.86-0.90 at the SAME coverage. Coverage is
  EXCLUDED (identical at 8k/40k); the read-window/integration-time lever (120->360) was ALSO tested negative in that
  session. The wall is the few-spike/low-SNR READ that supplies the per-output error, not the amount of data, not
  the window, and not the feedback-alignment rule family (exhausted, 2026-07-12).

THE OPEN LEVER (plan-of-record, verbatim): "a read-SNR manipulation that is NOT integration-window — i.e. raise the
effective spike count of the read: higher firing-rate/gain, an ENSEMBLE read (average over a population), or a
MULTI-COMPARTMENT / DENDRITIC read (the BurstCCN 'two mechanisms our port lacks'; Urbanczik-Senn soma-vs-dendrite)."

WHAT IS ALREADY IN CODE BUT NEVER DECISIVELY MEASURED. The existing batched-substrate runner already exposes
`--sub-pop` (default 1): `GradedConductanceLogitRead._graded_margin` (`_wkv_graded_conductance_read_derisk.py:113`)
already SUMS P independent neurons' conductance per word-pool (`ge.reshape(V, P).sum(axis=1)`) — i.e. population
averaging is already wired, just never swept at the LEARNING forward. The batched-substrate runner's own CLI help
string calls this "graded read ~P-indep" (`_wkv_mouth_readout_eprop_batched_substrate_derisk.py:633`) — but that is
an unfiled, terse in-code comment from exploratory work, NOT a filed 6-seed (or even 1-seed) decisive measurement.
Per CLAUDE.md's own discipline ("the comfortable verdict is the START of the research, never the end"), a
one-line code comment asserting a null result is exactly the kind of claim that needs a real, filed run before it
is trusted — especially since it contradicts the mechanism that WAS 6-seed GO one level up the pipeline: reading the
open-ended WKV generator's NEXT WORD from a population of Izhikevich spikes (P>=8) reaches ideal-sampler parity
where a naive single-neuron read (P=1) recovers only ~46-56% of the distribution
(`research/findings/2026-08-13-gap1-A1-fewspike-izhikevich-read-of-fluent-wkv-generation-population-coding-is-the-
companion-process.md`). That lever is INFERENCE-time (reading an already-trained head); this one is LEARNING-time
(the forward whose error drives the weight update) — different question, same population-coding idea, and it has
never been decisively tried on the read that actually blocks gap#4.

THIS RUNNER, TWO LEVERS:

  --lever ensemble (READY TO RUN, this file DOES implement it): a subprocess SWEEP of the existing, already-6-seed-
    validated `_wkv_mouth_readout_eprop_batched_substrate_derisk` module over `--sub-pop` in {1,2,4,8,16,32}, at a
    fixed operating point (screen: cheap/fast; decisive: the exact 8k-coverage matched setting the 0.371 baseline
    used). Zero new mechanism code -- pure reuse-by-subprocess of the validated, anti-cheated runner, single
    variable changed. This is the SMALLEST test of the "ensemble" half of the open lever.

  --lever dendritic (IMPLEMENTED 2026-08-27 in the underlying runner as --dendritic; this scaffold invokes it): a
    genuine two-compartment Urbanczik & Senn (2014) read. The existing runner ALREADY shows the wiring pattern (its
    `bias_e`/`bias_i` tonic bias-input population, wired onto the SAME word-pools as an independent extra synaptic
    drive -- see `_wkv_mouth_endtoend_substrate_read_derisk.py` "BIOLOGIZE head_b" section) -- the dendritic lever
    reuses exactly that pattern but for the TEACHING signal instead of a tonic prior:
      * BASAL compartment (unchanged): the existing feedforward hid/hidinh graded-conductance margin. This alone
        stays the forward "prediction" used for the actual answer/eval -- untouched, so the demonstrated substrate
        read-out (recov_argmax 0.97 on a good W) is not disturbed.
      * APICAL compartment (NEW, reuses the bias-pop wiring template): a SECOND independent population driven by
        the ONE-HOT TRAINING TARGET (available only during learning, biologically a top-down/feedback projection,
        never part of the forward answer) -- same block-diagonal-per-batch-slot wiring as `bias_e`/`bias_i`, just
        target-driven instead of tonic.
      * LOCAL ERROR (the actual mechanism change): today's rule is
            err_j = softmax(basal_margin)_j - 1{j == target}          # needs the SAME noisy basal read to carry
                                                                       # both the ANSWER and the LEARNING signal
        the Urbanczik-Senn substitution is
            err_j = sigma(apical_margin_j) - sigma(basal_margin_j)    # a genuinely LOCAL prediction-error between
                                                                       # two INDEPENDENT synaptic reads, neither of
                                                                       # which alone needs to carry the whole load
        i.e. stop asking one noisy few-spike read to be both the prediction AND the teacher; give the teacher its
        own dendritic pathway (this is precisely the "two mechanisms our port lacks" gap named in the plan-of-
        record, and precisely the citation already sitting unused in the existing runner's own docstring: "Urbanczik
        & Senn Neuron 81:521 (2014) dendritic-prediction delta rule").
      * Smallest test: identical screen operating point + identical anti-cheat battery as `ensemble` below, PLUS
        two dendritic-specific anti-cheats: (a) SHUFFLE the apical-target mapping -> the local error becomes
        uninformative -> recovery collapses to the shuffle floor (currently ~0.002); (b) FREEZE the apical
        population's target-drive (silence it, tonic-zero) -> the rule degenerates to `err_j = -sigma(basal_margin)`
        for every class alike -> recovery collapses to the current ~0.34 plateau (the mechanism must REDUCE to
        today's wall when the new pathway is removed, not exceed it by accident of a different bug).
      * GO-gate (same bar the existing runner already uses): >=0.85 x copied-head substrate recovery (today's
        `integrated_go`), OR at minimum a decisive, anti-cheat-clean lift over the 0.34-0.37 plateau (e.g. >=0.55,
        the rough midpoint the WKV-fewspike population lever demonstrated between a degraded read and ideal-sampler
        parity), with `host_matmul_on_learning_forward == 0` preserved (the forward, apical included, stays 0 host
        matmul) and the two anti-cheats above collapsing.
    IMPLEMENTED 2026-08-27 (`research/mouth-read-snr-dendritic`): the apical population (a block-diagonal labelled-line
    excitatory teacher per (block,word) + a tonic inhibitory baseline), the apical substrate read `apical_margin`, the
    per-seed apical unit-calibration, the U-S local error, and the freeze-apical + shuffle-apical anti-cheats all live
    in `_wkv_mouth_readout_eprop_batched_substrate_derisk.py` behind `--dendritic` (byte-identical to the softmax rule
    when off; verified by a numpy off/off diff). This lever was escalated to AFTER the ensemble screen returned its
    verdict: the ensemble (--sub-pop) lever is INERT by construction (the word-pool members are deterministic
    conductance replicas of a SHARED noisy hidden drive -> common-mode -> no SNR averaging; see the 2026-08-27
    verdict finding), so the dendritic lever is the live contingency.

USAGE (screen -- cheap, ~minutes/arm on GPU, single seed, reduced coverage; NOT the decisive setting):
    SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_readout_snr_ensemble_dendritic_derisk \\
        --lever ensemble --coverage screen --pops 1,2,4,8,16 --seeds 42 \\
        --out-dir research/findings/raw/_wkv_mouth_readout_snr_ensemble/screen

USAGE (decisive -- the exact 8k-coverage operating point the 0.371 baseline used, 6 seeds; several GPU-hours):
    SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_readout_snr_ensemble_dendritic_derisk \\
        --lever ensemble --coverage decisive --pops 1,4,8,16 --seeds 42,43,44,100,101,102 \\
        --out-dir research/findings/raw/_wkv_mouth_readout_snr_ensemble/decisive

DRY RUN (prints the exact subprocess commands + exits, no execution -- the only mode exercised while writing this):
    .venv/bin/python -m research.runners._wkv_mouth_readout_snr_ensemble_dendritic_derisk \\
        --lever ensemble --coverage screen --pops 1,2,4,8,16 --seeds 42 --dry-run

QUEUE (do not run locally on this audit session -- stage it for the GPU queue instead):
    bash tools/gpu_queue.sh add 'SIM_BACKEND=cupy /home/dant123/Projects/sim/.venv/bin/python -u -m \
research.runners._wkv_mouth_readout_snr_ensemble_dendritic_derisk --lever ensemble --coverage screen \
--pops 1,2,4,8,16 --seeds 42 --out-dir research/findings/raw/_wkv_mouth_readout_snr_ensemble/screen'

ANTI-CHEATS (inherited, not reimplemented): every sweep arm is a full invocation of the existing, already-anti-
cheated `_wkv_mouth_readout_eprop_batched_substrate_derisk` runner (shuffle-teach / frozen / lesion-err collapse,
`host_matmul_on_learning_forward == 0`, cfg.seed build-twice hash) -- this file adds NO new mechanism for the
`ensemble` lever, only an outer parameter sweep + aggregation, so it inherits that runner's validated guarantees
verbatim. Runner-only, additive, default-off/manually-invoked, NO `sim/` edit.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.verdict import Verdict  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_UNDERLYING_MODULE = "research.runners._wkv_mouth_readout_eprop_batched_substrate_derisk"

# The exact 8k-coverage operating point the 0.371 (6-seed) baseline used (all defaults of the underlying runner
# EXCEPT --sub-pop, which is the single variable this sweep changes). See _wkv_mouth_readout_eprop_batched_substrate
# _derisk.py argparse defaults (n-train-pos 9600, n-eval-pos 800, epochs 8, batch 48) -- left at the module's own
# defaults here (not repeated) so a drift in the underlying runner's defaults cannot silently desync this sweep.
_DECISIVE_EXTRA_ARGS: list[str] = []

# A cheap SCREEN: ~8x fewer train positions, fewer epochs, smaller batch -- fast enough to scan 5 pop values before
# committing GPU-hours to the decisive 8k/6-seed setting. This is a SCAN for a signal, not a claim; a positive screen
# must be re-confirmed at --coverage decisive with the matched 6-seed battery before it is filed as a finding.
_SCREEN_EXTRA_ARGS: list[str] = [
    "--n-train-pos", "1200", "--n-eval-pos", "300", "--epochs", "3", "--batch", "16", "--n-sub-demo", "80",
]


def _cmd_for(pop: int, seeds: str, coverage: str, out_json: Path, python: str, backend: str,
             extra: list[str], lever: str = "ensemble") -> list[str]:
    cmd = [python, "-u", "-m", _UNDERLYING_MODULE,
           "--forward", "substrate", "--sub-pop", str(pop), "--seeds", seeds,
           "--json", str(out_json)]
    if lever == "dendritic":
        # the DENDRITIC (Urbanczik-Senn) lever is now IMPLEMENTED in the underlying runner as --dendritic (a second
        # target-driven apical substrate read; err = sigma(apical) - sigma(basal), per-unit not softmax). Same anti-cheat
        # battery + provenance + GO-gate as the substrate forward, PLUS the freeze-apical / shuffle-apical anti-cheats.
        cmd += ["--dendritic"]
    cmd += (_DECISIVE_EXTRA_ARGS if coverage == "decisive" else _SCREEN_EXTRA_ARGS)
    cmd += extra
    return cmd


def run_sweep(args) -> list[Path]:
    """Launch (or, --dry-run, print) one underlying-runner invocation per pop value. Returns the list of output
    JSON paths (whether or not they were actually produced this call -- --from-json reuses pre-existing ones)."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pops = [int(p) for p in args.pops.split(",") if p.strip()]
    paths = []
    for pop in pops:
        tag = f"dendritic_sub_pop{pop}" if args.lever == "dendritic" else f"sub_pop{pop}"
        out_json = out_dir / f"{tag}_{args.coverage}.json"
        paths.append(out_json)
        cmd = _cmd_for(pop, args.seeds, args.coverage, out_json, args.python, args.backend, args.extra_args,
                       lever=args.lever)
        env_prefix = f"SIM_BACKEND={args.backend} "
        printable = env_prefix + " ".join(cmd)
        if args.dry_run:
            print(f"[dry-run] would run: {printable}")
            continue
        # --aggregate-only NEVER launches anything, even for a missing arm (read whatever JSON already exists,
        # report the rest as absent -> aggregate() correctly reports UNDEFINED rather than silently kicking off a
        # heavy subprocess). Distinct from --from-json, which launches the MISSING arms (a resume of a partial
        # sweep) and only skips ones already on disk.
        if args.aggregate_only:
            if not out_json.exists():
                print(f"[missing] {out_json} absent (--aggregate-only never launches)")
            continue
        if args.from_json and out_json.exists():
            print(f"[skip] {out_json} already exists (--from-json)")
            continue
        print(f"[launch] {printable}", flush=True)
        env = dict(os.environ)
        env["SIM_BACKEND"] = args.backend
        t0 = time.time()
        subprocess.run(cmd, cwd=str(_REPO), env=env, check=True)
        print(f"[done] pop={pop} in {time.time() - t0:.0f}s -> {out_json}", flush=True)
    return paths


def _load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data.get("summary") or None


def aggregate(paths: list[Path], pops: list[int]) -> dict:
    """Read each sweep arm's summary (if it ran), build the pop -> recovery table, and decide the ensemble-lift
    verdict via tools.verdict.Verdict (UNDEFINED, not a fabricated GO/NO-GO, if any arm never ran)."""
    rows = {}
    for pop, path in zip(pops, paths):
        s = _load_summary(path)
        if s is not None:
            rows[pop] = s

    table = {
        pop: {
            "sub_learned_recov_mean": s.get("sub_learned_recov_mean"),
            "sub_recov_ratio_mean": s.get("sub_recov_ratio_mean"),
            "anticheats_collapse_count": s.get("anticheats_collapse_count"),
            "go_count": s.get("go_count"),
            "n_seeds": s.get("n_seeds"),
        }
        for pop, s in rows.items()
    }

    v = Verdict("ensemble (population) read lifts the mouth read-out e-prop learning forward off the ~0.34 plateau")
    baseline_pop = min(rows) if rows else None
    best_pop, best_recov = None, None
    p1_recov = rows.get(baseline_pop, {}).get("sub_learned_recov_mean") if baseline_pop is not None else None
    for pop, s in rows.items():
        r = s.get("sub_learned_recov_mean")
        if r is not None and (best_recov is None or r > best_recov):
            best_pop, best_recov = pop, r

    # NOTE ON VERDICT SEMANTICS: the preconditions registered below (`require`) must ALL hold for the verdict to
    # be anything other than UNDEFINED -- they are genuine INSTRUMENT checks (every arm produced data; the winning
    # arm's anti-cheats actually collapsed), not the hypothesis test itself. The hypothesis test (does raising P
    # lift the recovery by a margin that clears seed/run noise) is computed directly as `go` below and passed to
    # `decide()`, so a clean negative (lift < margin, preconditions all fine) correctly reports NO-GO -- a real,
    # filed, anti-cheated result confirming (or refuting) the existing terse in-code comment "graded read
    # ~P-indep" -- rather than being folded into `require` and forced to UNDEFINED (this file's first draft made
    # exactly that mistake; a synthetic-data smoke test caught it before any real run used it).
    v.require("every swept pop has a summary to read (no silently-missing arm)",
               len(rows), expect=lambda n: n == len(pops),
               note=f"{len(rows)}/{len(pops)} arms produced a summary")
    lift = None
    if p1_recov is not None and best_recov is not None:
        best_anticheats = rows.get(best_pop, {}).get("anticheats_collapse_count")
        best_nseeds = rows.get(best_pop, {}).get("n_seeds")
        v.require("best-pop arm's anti-cheats fully collapsed (a lift riding a broken anti-cheat is not trustworthy)",
                   best_anticheats,
                   expect=(lambda x: x is not None and best_nseeds is not None and x >= best_nseeds),
                   note=f"anticheats_collapse_count={best_anticheats} vs n_seeds={best_nseeds}")
        lift = best_recov - p1_recov
    go = bool(lift is not None and lift >= 0.10)
    decided = v.decide(go=go)

    return {
        "table_by_pop": table,
        "baseline_pop": baseline_pop, "baseline_recov": p1_recov,
        "best_pop": best_pop, "best_recov": best_recov, "lift": lift,
        "verdict": decided,
    }


def aggregate_dendritic(paths: list[Path], pops: list[int]) -> dict:
    """Read the dendritic run summaries and decide the Urbanczik-Senn verdict via tools.verdict.Verdict (UNDEFINED,
    not a fabricated GO/NO-GO, if the arm never ran). The underlying runner already computes the per-seed dendritic GO
    (integrated OR >=0.55 lift, with the freeze-apical + shuffle-apical anti-cheats collapsing and apical-read
    provenance); this aggregates its summary into the >=5/6-seed board verdict."""
    summaries = {}
    for pop, path in zip(pops, paths):
        s = _load_summary(path)
        if s is not None:
            summaries[pop] = s
    v = Verdict("the DENDRITIC (Urbanczik-Senn) two-compartment read lifts the mouth read-out e-prop learning forward "
                "off the ~0.37 plateau, teacher-load-bearing + provenance-clean")
    best = None
    for pop, s in summaries.items():
        r = s.get("sub_learned_recov_mean")
        if r is not None and (best is None or r > best[1]):
            best = (pop, r, s)
    v.require("every dendritic arm produced a summary (no silently-missing arm)", len(summaries),
              expect=lambda n: n == len(pops), note=f"{len(summaries)}/{len(pops)} arms produced a summary")
    go = False
    detail = {}
    if best is not None:
        pop, recov, s = best
        go_count = s.get("go_count") or 0
        ac_ok = s.get("dendritic_anticheats_ok_count")
        reads_ok = s.get("apical_reads_match_all")
        v.require("the winning arm's apical read ran every gradient step (provenance)", reads_ok,
                  expect=lambda x: bool(x), note=f"apical_reads_match_all={reads_ok}")
        v.require("the winning arm's dendritic anti-cheats collapse on >=5/6 seeds (teacher load-bearing)",
                  ac_ok, expect=(lambda x: x is not None and x >= 5), note=f"dendritic_anticheats_ok_count={ac_ok}")
        go = bool(go_count >= 5)                                       # >=5/6 per-seed dendritic GO (the runner's gate)
        detail = {"best_pop": pop, "sub_learned_recov_mean": recov,
                  "sub_freeze_apical_recov_mean": s.get("sub_freeze_apical_recov_mean"),
                  "go_count": go_count, "n_seeds": s.get("n_seeds"),
                  "dendritic_anticheats_ok_count": ac_ok, "apical_reads_match_all": reads_ok,
                  "sub_copied_recov_mean": s.get("sub_copied_recov_mean")}
    decided = v.decide(go=go)
    return {"summaries_by_pop": summaries, "detail": detail, "verdict": decided}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lever", choices=["ensemble", "dendritic"], required=True)
    ap.add_argument("--coverage", choices=["screen", "decisive"], default="screen")
    ap.add_argument("--pops", type=str, default="1,2,4,8,16",
                     help="comma-separated --sub-pop values to sweep (P=1 is the existing decisive baseline)")
    ap.add_argument("--seeds", type=str, default="42", help="comma-separated seeds, passed through verbatim")
    ap.add_argument("--out-dir", type=str, default="research/findings/raw/_wkv_mouth_readout_snr_ensemble/screen")
    ap.add_argument("--backend", type=str, default=os.environ.get("SIM_BACKEND", "cupy"))
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--dry-run", action="store_true", help="print the subprocess commands; run + aggregate nothing")
    ap.add_argument("--from-json", action="store_true",
                     help="LAUNCH only the arms missing from --out-dir (resume a partial sweep); an arm already "
                          "on disk is read, not re-run")
    ap.add_argument("--aggregate-only", action="store_true",
                     help="NEVER launch anything, even for a missing arm -- just read whatever --out-dir JSON "
                          "already exists (e.g. after tools/gpu_queue.sh finished a staged sweep) and report the "
                          "verdict; a missing arm is reported UNDEFINED, not silently launched")
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                     help="passed through verbatim to the underlying runner, e.g. --extra-args --verify-steps 4")
    ap.add_argument("--summary-json", type=str, default=None,
                     help="where to write the aggregated pop-vs-recovery table + verdict (default: <out-dir>/"
                          "aggregate_<lever>.json)")
    args = ap.parse_args()

    # the DENDRITIC mechanism is pop-INDEPENDENT (it is the apical teacher read, not the word-pool size); collapse the
    # ensemble default multi-pop sweep to a single arm so `--lever dendritic` runs the mechanism once per seed.
    if args.lever == "dendritic" and args.pops == "1,2,4,8,16":
        args.pops = "1"

    pops = [int(p) for p in args.pops.split(",") if p.strip()]
    paths = run_sweep(args)
    if args.dry_run:
        print("[dry-run] no aggregation -- nothing ran")
        return

    if args.lever == "dendritic":
        agg = aggregate_dendritic(paths, pops)
        out = args.summary_json or str(Path(args.out_dir) / f"aggregate_{args.lever}_{args.coverage}.json")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(agg, indent=2, default=str))
        print(json.dumps(agg["detail"], indent=2, default=str))
        print(f"[DENDRITIC] verdict={agg['verdict']['status']}")
        print(f"[done] -> {out}")
        return

    agg = aggregate(paths, pops)
    out = args.summary_json or str(Path(args.out_dir) / f"aggregate_{args.lever}_{args.coverage}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(agg, indent=2, default=str))
    print(json.dumps(agg["table_by_pop"], indent=2))
    print(f"[SWEEP] baseline P={agg['baseline_pop']} recov={agg['baseline_recov']} | "
          f"best P={agg['best_pop']} recov={agg['best_recov']} | verdict={agg['verdict']['status']}")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
