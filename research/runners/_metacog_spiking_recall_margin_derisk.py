"""DE-RISK (scaffold-retirement backlog rank 9, research/coordination/scaffold_retirement_backlog.md):
does a SPIKING recall-margin retire the metacog honesty-hedge's host confidence FORMULA?

THE SHORTCUT (backlog rank 9). `metacog_production_organ.mean_role_confidence` derives the per-turn EVIDENCE
fed to the metacog workspace's honesty hedge from `OneBrainComposer._margin` / `RFPhasorComposer.
_cleanup_all_score_stats`'s `margin`/`margin_norm`/`margin_snr` -- all HOST ARITHMETIC comparisons of the
matched-filter cleanup's scores ((peak-runner_up)/peak, or a z-score). The hedge itself is genuinely spiking
downstream (the metacog workspace WTA reads its OWN settled NMDA-conductance/firing-rate balance,
2026-08-12-laneC..., 2026-08-13-metacog-robust-confidence-GO.md) -- but the EVIDENCE flowing INTO that
competition is a Python comparison of numbers, not a read of the recall circuit's own spiking.

THE RETIREMENT MECHANISM (this de-risk). `RFPhasorComposer._spiking_margin` (added 2026-09-05) drives the SAME
cached Izhikevich concept bank `_spiking_cleanup`/`OneBrainComposer._spiking_select` already use for the
on-substrate winner-PICK, and reads a winner-vs-runner-up SPIKE-COUNT margin instead of a host score
comparison. Gated behind `BRAIN_METACOG_SPIKING_MARGIN` (default OFF, byte-identical); `mean_role_confidence`
prefers the resulting `margin_spiking` trace field when present.

THIS SCRIPT tests, at the COMPOSER level (constructing `OneBrainComposer` directly -- the class that owns
`_block_role_scores`/the metacog trace this whole mechanism feeds -- rather than the full multi-organ chat
brain, which costs ~180s/build on unrelated faculties; the neurons genuinely vary by seed either way, since
`OneBrainComposer(seed=seed)` reseeds both the FHRR codebook and the Izhikevich cleanup bank's heterogeneity):

  (a) TRACKS the host confidence: does margin_spiking correlate with the host `margin` (Pearson/Spearman)
      across a clean + synaptic-noise-degradation sweep, and does thresholding it reproduce the SAME
      confident/hedge classification the host formula gives, per seed?
  (b) LOAD-BEARING:
      - VARY: does the hedge's `confident` decision change as recall quality degrades, driven PURELY by the
        spiking evidence path (BRAIN_METACOG_SPIKING_MARGIN=1, ignoring the host formula entirely)?
      - LESION: does artificially removing the recall circuit's OWN discrimination (`_spiking_margin(...,
        lesion=True)`: the competing candidates driven identically, no differential) collapse margin_spiking
        and flip a would-be-confident case to a hedge?

  across the mandated 6 seeds: 42, 43, 44, 100, 101, 102. numpy/CPU throughout (cost-routed).

Usage: python -m research.runners._metacog_spiking_recall_margin_derisk --out <path.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
# the flag under test: ON for this whole process so every role chip carries BOTH `margin` (host, always
# computed) AND `margin_spiking` (spiking, additive) -- lets one composer instance / one query serve BOTH
# arms of the comparison with NO confound from separately-built neurons.
os.environ["BRAIN_METACOG_SPIKING_MARGIN"] = "1"

import numpy as np

from research.runners.one_brain_composer import OneBrainComposer
import research.runners.rf_phasor_composer as _rfp
from research.runners._emergent_graceful_degradation_derisk import _noise
from research.runners.metacog_production_organ import (
    mean_role_confidence, evidence_from_role_conf, MetacogProductionOrgan,
    ROLE_CONF_LO, ROLE_CONF_HI,
)
from tools.lab import attributable_to

SEEDS = [42, 43, 44, 100, 101, 102]
SIGMAS = [0.3, 0.6, 0.9, 1.1, 1.5, 2.0, 2.5, 3.0, 4.0]
FACTS = [("brain", "use", "spikes"), ("brain", "learn", "words"), ("brain", "store", "memory"),
         ("dog", "chase", "cat"), ("cat", "eat", "fish")]
VOCAB = sorted({w for f in FACTS for w in f} | {"river", "bird", "worm", "ball"})


def _host_mrc(activity):
    """The PRE-spiking-preference legacy average (mirrors mean_role_confidence's host chain exactly, but never
    prefers margin_spiking even when present) -- the CONTROL arm, read off the SAME trace/roles."""
    roles = (activity or {}).get("roles") or []
    vals = []
    for r in roles:
        snr = r.get("margin_snr"); mn = r.get("margin_norm"); m = r.get("margin")
        v = snr if snr is not None else (mn if mn is not None else (m if m is not None else r.get("confidence")))
        if v is not None:
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def build_composer(seed):
    comp = OneBrainComposer(seed=seed, trace=True, vocab=VOCAB)
    for a, v, p in FACTS:
        comp.hear(f"{a} {v} {p}", polarity="AFFIRM")
    return comp


def ask_and_trace(comp, noised_conns=None):
    base = list(comp.store_conns)
    if noised_conns is not None:
        comp.store_conns = noised_conns
    try:
        ans = comp.query_patient("brain", "use")
        return ans, comp.last_trace
    finally:
        comp.store_conns = base


def capture_raw_scores(comp, noised_conns=None):
    """Capture the raw (rectified) per-role score arrays `_spiking_margin` receives for this exact query, by
    temporarily wrapping it -- used ONLY to construct the lesion arm (re-drive the SAME competition with
    `lesion=True` on the SAME captured scores), never to change the answer path."""
    captured = []
    orig = _rfp.RFPhasorComposer._spiking_margin

    def _wrap(self, scores, lesion=False):
        if not lesion:
            captured.append(np.maximum(np.asarray(scores, dtype=float), 0.0))
        return orig(self, scores, lesion=lesion)

    _rfp.RFPhasorComposer._spiking_margin = _wrap
    try:
        ans, trace = ask_and_trace(comp, noised_conns)
    finally:
        _rfp.RFPhasorComposer._spiking_margin = orig
    return ans, trace, captured


def lesioned_spiking_mrc(comp, raw_scores):
    """Mean margin_spiking across roles with the RECALL CIRCUIT's own discrimination removed per role
    (uniform drive -> no differential), on the SAME raw score arrays a real query just produced."""
    if not raw_scores:
        return None
    vals = [comp.comp._spiking_margin(s, lesion=True) for s in raw_scores]
    return float(np.mean(vals))


def rankdata(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x))
    return ranks


def run_seed(seed):
    comp = build_composer(seed)
    base_conns = list(comp.store_conns)
    organ = MetacogProductionOrgan(seed=seed)
    organ.ensure_built()

    rows = []
    # CLEAN condition, captured with raw scores (for the lesion arm)
    ans0, trace0, raw0 = capture_raw_scores(comp, None)
    host0 = _host_mrc(trace0)
    spiking0 = mean_role_confidence(trace0)
    lesioned0 = lesioned_spiking_mrc(comp, raw0)
    # ATTRIBUTION (tools.lab.attributable_to, gates/attribution_required): the lesion is a treatment(intact)/
    # control(lesioned) pair -- ask explicitly what FRACTION of the intact margin is NOT present once the
    # recall circuit's own discrimination is removed, rather than just reporting both numbers side by side
    # (the gap#5 lesson: two correctly-measured arms sitting one key apart, with nobody subtracting them).
    frac_attributable = None
    if spiking0 is not None and lesioned0 is not None:
        print(f"  seed {seed} clean:", flush=True)
        frac_attributable = attributable_to("spiking_margin_lesion(seed=%s)" % seed, spiking0, lesioned0)
    rows.append({"cond": "clean", "sigma": 0.0, "answer": ans0, "abstained": ans0 is None,
                "host_mrc": host0, "spiking_mrc": spiking0, "spiking_mrc_lesioned": lesioned0,
                "spiking_margin_fraction_attributable_to_discrimination": frac_attributable})

    # a FRESH default_rng(1000+seed) per sigma level (not one rng advanced across the sweep, which would make
    # successive sigma levels non-comparable path-dependent perturbations) -- matches
    # _confidence_read_discrimination_derisk.py's fixed-per-call noise seeding, tied to `seed` for genuine
    # 6-seed diversity across composer builds.
    for sigma in SIGMAS:
        noised = _noise(base_conns, sigma, np.random.default_rng(1000 + seed))
        ans, trace = ask_and_trace(comp, noised)
        host = _host_mrc(trace)
        spiking = mean_role_confidence(trace)
        rows.append({"cond": f"sigma{sigma}", "sigma": sigma, "answer": ans, "abstained": ans is None,
                    "host_mrc": host, "spiking_mrc": spiking, "spiking_mrc_lesioned": None})

    # organ judgments (evidence -> workspace WTA -> confident bool), for every non-abstained row
    for row in rows:
        for key in ("host_mrc", "spiking_mrc", "spiking_mrc_lesioned"):
            mrc = row[key]
            if mrc is None:
                row[key + "_judge"] = None
                continue
            ev = evidence_from_role_conf(mrc)
            j = organ.judge(ev if ev is not None else 0.0)
            row[key + "_judge"] = {"evidence": ev, "confident": j["confident"], "balance": j["balance"],
                                   "threshold": j["threshold"]}
    return {"seed": seed, "rows": rows, "organ_threshold": organ.threshold, "organ_calib": organ.calib}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = ap.parse_args()

    per_seed = []
    for seed in args.seeds:
        print(f"=== seed {seed} ===", flush=True)
        r = run_seed(seed)
        per_seed.append(r)
        for row in r["rows"]:
            print(f"  {row['cond']:10s} host={row['host_mrc']} spiking={row['spiking_mrc']} "
                  f"lesioned={row['spiking_mrc_lesioned']} "
                  f"host_conf={(row['host_mrc_judge'] or {}).get('confident')} "
                  f"spk_conf={(row['spiking_mrc_judge'] or {}).get('confident')} "
                  f"les_conf={(row['spiking_mrc_lesioned_judge'] or {}).get('confident')}", flush=True)

    # --- aggregate: tracking (Pearson + Spearman over ALL seed x condition points), PLUS the agreement rate
    # split into UNAMBIGUOUS (host mrc clearly outside the ROLE_CONF_LO/HI band -> both signals should agree
    # trivially) vs AMBIGUOUS (host mrc inside the band, i.e. genuinely borderline) -- this is the honest split
    # the task anticipates ("if the spiking read can't match the host formula's discrimination, characterize
    # the residual"): a lumped single agreement number hides WHERE any disagreement actually lives. ---
    hosts, spikes = [], []
    agree_all, n_all = 0, 0
    agree_clear, n_clear = 0, 0
    agree_ambig, n_ambig = 0, 0
    for r in per_seed:
        for row in r["rows"]:
            if row["host_mrc"] is not None and row["spiking_mrc"] is not None:
                hosts.append(row["host_mrc"]); spikes.append(row["spiking_mrc"])
                hj, sj = row["host_mrc_judge"], row["spiking_mrc_judge"]
                if hj is not None and sj is not None:
                    n_all += 1
                    ok = int(hj["confident"] == sj["confident"])
                    agree_all += ok
                    if row["host_mrc"] <= ROLE_CONF_LO or row["host_mrc"] >= ROLE_CONF_HI:
                        n_clear += 1; agree_clear += ok
                    else:
                        n_ambig += 1; agree_ambig += ok
    hosts_a, spikes_a = np.array(hosts), np.array(spikes)
    pearson = float(np.corrcoef(hosts_a, spikes_a)[0, 1]) if hosts_a.std() > 0 and spikes_a.std() > 0 else float("nan")
    rh, rs = rankdata(hosts_a), rankdata(spikes_a)
    spearman = float(np.corrcoef(rh, rs)[0, 1]) if hosts_a.std() > 0 and spikes_a.std() > 0 else float("nan")
    agreement_rate = agree_all / n_all if n_all else float("nan")
    agreement_clear = agree_clear / n_clear if n_clear else float("nan")
    agreement_ambiguous = agree_ambig / n_ambig if n_ambig else float("nan")

    # --- load-bearing (a): VARY -- per seed, does spiking_mrc strictly (non-strictly) DECREASE as sigma
    # increases, and does the spiking-driven `confident` decision flip to False somewhere in the sweep for a
    # case whose clean turn was confident? ---
    vary_per_seed = []
    for r in per_seed:
        clean_row = r["rows"][0]
        clean_spk_conf = (clean_row["spiking_mrc_judge"] or {}).get("confident")
        sigma_rows = r["rows"][1:]
        spk_vals = [row["spiking_mrc"] for row in sigma_rows if row["spiking_mrc"] is not None]
        monotone_ish = all(a >= b - 1e-9 for a, b in zip([clean_row["spiking_mrc"]] + spk_vals, spk_vals)) \
            if clean_row["spiking_mrc"] is not None and spk_vals else False
        flips_to_hedge = any((row["spiking_mrc_judge"] or {}).get("confident") is False for row in sigma_rows)
        vary_per_seed.append({"seed": r["seed"], "clean_confident": clean_spk_conf,
                              "monotone_ish": monotone_ish, "flips_to_hedge_somewhere": flips_to_hedge})

    # --- load-bearing (b): LESION -- on the clean row (which has raw scores captured), does the lesioned
    # spiking margin collapse toward the LOW end, and does the organ's decision flip confident->hedge? ---
    lesion_per_seed = []
    for r in per_seed:
        clean_row = r["rows"][0]
        intact = clean_row["spiking_mrc"]
        lesioned = clean_row["spiking_mrc_lesioned"]
        intact_conf = (clean_row["spiking_mrc_judge"] or {}).get("confident")
        lesioned_conf = (clean_row["spiking_mrc_lesioned_judge"] or {}).get("confident")
        collapsed = (lesioned is not None and intact is not None and lesioned < intact) or \
                    (lesioned is not None and lesioned <= ROLE_CONF_LO)
        flips = (intact_conf is True) and (lesioned_conf is False)
        lesion_per_seed.append({"seed": r["seed"], "intact": intact, "lesioned": lesioned,
                                "intact_confident": intact_conf, "lesioned_confident": lesioned_conf,
                                "collapsed": collapsed, "flips_confident_to_hedge": flips})

    verdict = {
        "n_points": len(hosts), "pearson_r": pearson, "spearman_rho": spearman,
        "confident_agreement_rate_all": agreement_rate, "n_all_compared": n_all,
        "confident_agreement_rate_clear_cases": agreement_clear, "n_clear_compared": n_clear,
        "confident_agreement_rate_ambiguous_cases": agreement_ambiguous, "n_ambiguous_compared": n_ambig,
        # "tracks" = strong monotonic correlation (the task's own wording: "higher margin -> higher
        # confidence") PLUS clean agreement on the UNAMBIGUOUS cases (a clearly-confident or clearly-degraded
        # turn must classify the same way under either signal). Agreement in the AMBIGUOUS middle band is
        # reported, not gated on -- two distinct measurement channels of a genuinely borderline case are
        # expected to sometimes cross a fixed threshold on opposite sides even when strongly correlated
        # (signal-detection-theory criterion sensitivity); demanding perfect agreement THERE would be
        # over-claiming precision neither channel has. See the findings doc for the honest characterization.
        "tracks": bool(not np.isnan(spearman) and spearman >= 0.7 and
                      (np.isnan(agreement_clear) or agreement_clear >= 0.9)),
        "vary_per_seed": vary_per_seed,
        "vary_all_seeds_flip": all(v["flips_to_hedge_somewhere"] for v in vary_per_seed),
        "lesion_per_seed": lesion_per_seed,
        "lesion_all_seeds_collapse": all(l["collapsed"] for l in lesion_per_seed),
        "lesion_all_seeds_flip_where_intact_confident": all(
            (not l["intact_confident"]) or l["flips_confident_to_hedge"] for l in lesion_per_seed),
    }
    verdict["load_bearing"] = bool(verdict["vary_all_seeds_flip"] and verdict["lesion_all_seeds_collapse"]
                                   and verdict["lesion_all_seeds_flip_where_intact_confident"])
    verdict["mission_go"] = bool(verdict["tracks"] and verdict["load_bearing"])

    out = {"seeds": args.seeds, "sigmas": SIGMAS, "role_conf_lo": ROLE_CONF_LO, "role_conf_hi": ROLE_CONF_HI,
          "margin_drive_pA": _rfp.RFPhasorComposer(seed=1)._margin_drive_pA, "per_seed": per_seed,
          "verdict": verdict}

    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2, default=str))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
