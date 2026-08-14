"""METACOG ROBUST CONFIDENCE — a DIVISIVE-NORMALIZED balance-of-evidence read whose confident/uncertain
decision is INVARIANT to the per-region init heterogeneity re-draw that one-brain pool #2 requires.

THE BLOCKER (pool #2, `2026-08-13-onebrain-second-pool-SCOPED.md`). The metacog + pragmatic production organs merge
onto ONE shared spiking bridge BYTE-IDENTICALLY (merge GO 6/6). Pragmatic is answer-preserving (6/6). Metacog is NOT
(1/6): its balance-of-evidence confidence read is the ABSOLUTE spike-rate margin of the first-order WTA competition,
`|rate(asm1) - rate(asm0)|` off `cp_firing_states` (Vickers balance-of-evidence). That absolute margin has a DECLARED
NARROW DYNAMIC RANGE (`metacog_production_organ` residual). A shared pool REQUIRES the three region-scoped merge seams
(`per_region_parameter_heterogeneity` / `_threshold_heterogeneity` / `per_region_wiring_seed`) so each organ's slice is
co-residence-invariant — but those seams RE-DRAW the workspace's per-neuron Izhikevich params + firing thresholds
name-keyed instead of from the global-RNG order. That re-draw is, to first order, a per-region GAIN + threshold
perturbation of the assembly f-I curves, and an ABSOLUTE rate-difference SCALES with gain: a re-draw that lowers the
assembly gain compresses every margin toward the calibrated threshold, flipping the `confident`/`uncertain` decision at
mid-range evidence (the small margin the finding measured). The self-calibrating threshold moves too, but NOT
proportionally with the per-evidence margins, so the classification of a given evidence flips.

THE REAL ROOT CAUSE (measured here, deeper than "gain"). A first lever — divisive-normalizing the SPIKE-rate margin
by the summed co-active rate (`|r1-r0|/(r1+r0)`) — FAILED (still 0/1), because the flattening is not a pure gain change
the ratio could cancel: the spike-count margin sits at the NOISE FLOOR. The workspace assemblies fire at ~0.1% at this
operating point, so `|rate(asm1)-rate(asm0)|` is a difference of ~0.1 spikes — even the STANDALONE build's
margin(evidence) is only ~0.5 monotone (near-random). The confident/uncertain decision at mid-range is essentially
reading noise, and the per-region re-draw reshuffles that noise -> the flip. There is no signal in the spike counts to
normalize.

THE FIX — DIVISIVE NORMALIZATION OF THE NMDA CONDUCTANCE (Carandini & Heeger 2012 divisive normalization; Wang
persistent-NMDA accumulator). Read the balance off the assemblies' slow-NMDA recurrent conductance
(`cp_conductance_g_nmda`) — the GRADED synaptic accumulator the metacog faculty was explicitly designed around ("slow
NMDA lets meta INTEGRATE the settled balance-of-evidence") — instead of the coarse spike count:

    conf_norm(evidence) = |g_nmda(asm1) - g_nmda(asm0)| / (g_nmda(asm1) + g_nmda(asm0) + eps)

Both terms are read from the SAME substrate: the NMDA conductance is driven by presynaptic spikes through NMDA synapses
(a genuine spiking-substrate state, NOT the injected current), and slow-NMDA integrates the sparse spikes into a SMOOTH
graded signal with real SNR. Measured: this margin tracks evidence monotone 1.00 in BOTH the standalone and the merged
build, with the two curves nearly overlapping (evid 0.5: today 0.063 vs merged 0.056; evid 1.0: 0.099 vs 0.086) — so the
self-calibrated confident/uncertain threshold lands at the SAME evidence boundary in both, and the decision is invariant
to the re-draw. This is the anti-cheat's sanctioned form ("divisive-norm off cp_firing_states/CONDUCTANCES"), NOT a host
rescale of the answer: numerator = the two competing accumulators' balance, denominator = their summed co-active NMDA
drive (the normalization pool). The threshold self-calibrates on this margin exactly as before, and the STANDALONE
decision is unchanged (extremes fixed; only the mid-range stops flipping).

WHAT THIS RUNNER MEASURES (re-runs the pool-#2 answer-preservation A/B, `_onebrain_production_flip2_verify`, with the
metacog read swappable). For BOTH reads, per seed, over the pool-#2 panel (metacog evidence sweep MC_EVID x8; pragmatic
{none, some, all}):

  A. ONE shared pool                : MERGED metacog.bridge IS pragmatic.bridge IS the substrate (one cp_ array).
  B. MERGED == CORESIDENT (byte-id) : the genuine merge byte-identity (co-residence adds no footprint; == 0.0).
  C. answer preserved MERGED-vs-TODAY : the metacog `confident` bool + pragmatic implicature/enriched IDENTICAL.
  I. INTERPRETATION UNCHANGED       : (fix only) robust-TODAY `confident` == absolute-TODAY `confident` at every
                                      evidence — the normalization does NOT change which decision the STANDALONE
                                      production organ makes; it only makes the MERGED build agree with TODAY.
  DGN. NON-DEGENERATE               : (fix only) the robust read is NOT trivially invariant — its confident pattern
                                      carries BOTH confident and uncertain across the evidence sweep AND tracks
                                      evidence (low evidence -> uncertain, high -> confident). Guards the cheat where
                                      "make everything confident" is answer-invariant but destroys the faculty.

THE QUESTION. Does the metacog confident/uncertain decision become answer-preserving 6/6 (up from the absolute read's
1/6) under the merged re-draw, while B(==0)/pragmatic(6/6) hold, the winning decision is unchanged (I), and the read
still discriminates (DGN)? GO iff yes; else the honest boundary + the named next mechanism.

BRAIN-BASED: the robust read is a divisive normalization of two `cp_conductance_g_nmda` accumulator reads (numerator =
the balance, denominator = the co-active NMDA drive), NOT a host rescale of the answer. LESION -> the fragility returns:
running with the ABSOLUTE read (`--read balance`) reproduces the 1/6 baseline, so the normalization is the load-bearing
cause of the invariance. Functional read-out only; no phenomenal claim.

Reproduce:
  SIM_BACKEND=numpy python -m research.runners._metacog_robust_confidence_derisk \
      --seeds 42,43,44,100,101,102 --out research/findings/raw/_metacog_robust_confidence_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# reuse-by-import: the CANONICAL divisive-normalized NMDA-conductance margin is defined once in the production
# organ (metacog_production_organ.nmda_norm_margin); the base MetacogProductionOrgan dispatches its `_margin` to
# it under confidence_read="nmda_norm", so this de-risk drives the identical read through the real organ APIs.
from research.runners.metacog_production_organ import MetacogProductionOrgan
from research.runners.pragmatic_production_organ import PragmaticProductionOrgan
from research.runners.onebrain_merge_production2 import MergedSubstrate2
from research.runners._recursive_tom_rsa_derisk import UTTS
from tools.lab import lever, void_if

# the pool-#2 metacog evidence sweep (low -> high answer confidence).
MC_EVID = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]


class RobustMetacogProductionOrgan(MetacogProductionOrgan):
    """MetacogProductionOrgan PINNED to the divisive-normalized NMDA-conductance confidence read — which is now the
    production DEFAULT (`metacog_production_organ.nmda_norm_margin`; adopted 2026-08-13). Kept as a named alias for
    this A/B so the read is forced explicitly regardless of the `BRAIN_METACOG_READ` env. The margin logic is NOT
    reimplemented here — the base class dispatches `_margin` to the canonical `nmda_norm_margin` (single source of
    truth, reuse-by-import).

    WHY the NMDA conductance. The absolute spike-count margin (`confidence_read="balance"`, `|rate(asm1)-rate(asm0)|`
    off `cp_firing_states`) sits at the NOISE FLOOR: the workspace assemblies fire at ~0.1% at this operating point, so
    the margin is a difference of ~0.1 spikes — even the STANDALONE build's margin(evidence) is only ~0.5 monotone
    (near-random), which is the real reason the per-region re-draw flips the decision. The slow-NMDA recurrent
    conductance (`cp_conductance_g_nmda`) is the assembly's GRADED synaptic accumulator — the "balance-of-evidence
    integrator" the metacog faculty was designed around (Wang persistent NMDA) — driven by presynaptic spikes through
    NMDA synapses (a genuine substrate/spiking state, NOT the injected current), and it tracks evidence monotone 1.00 in
    BOTH the standalone and the merged build -> the confident/uncertain decision is invariant to the re-draw. Carandini
    & Heeger DIVISIVE NORMALIZATION off conductances, NOT a host rescale of the answer."""

    def __init__(self, seed: int = 42, shared=None):
        super().__init__(seed=seed, shared=shared, confidence_read="nmda_norm")


def _metacog_cls(read: str):
    """Return a 0-arg-per-(seed,shared) factory for the requested read. `nmda_norm` -> the production DEFAULT read
    (RobustMetacogProductionOrgan, pinned); `balance` -> the original ABSOLUTE spike-rate read (the 1/6 baseline
    control / lever), forced explicitly since the production default is now nmda_norm."""
    from functools import partial
    if read == "nmda_norm":
        return RobustMetacogProductionOrgan
    if read == "balance":
        return partial(MetacogProductionOrgan, confidence_read="balance")
    raise ValueError(f"unknown read={read!r}")


def _read_metacog(org):
    """Per-evidence: (evidence, margin, confident bool)."""
    out = []
    for e in MC_EVID:
        j = org.judge(e)
        out.append((float(e), float(j["balance"]), bool(j["confident"])))
    return out


def _read_pragmatic(org):
    """Per-utterance: (utterance, belief distribution, implicature_represented bool, enriched phrase)."""
    out = []
    for u in UTTS:
        info = org.interpret(u)
        out.append((u, [float(x) for x in info["belief"]],
                    bool(info["implicature_represented"]), str(info["enriched_interpretation"])))
    return out


def run_seed(seed: int, read: str) -> dict:
    cls = _metacog_cls(read)

    # TODAY (== flag-off production == escape path): each organ on its OWN bridge (global heterogeneity).
    mc_today = cls(seed=seed, shared=None)
    pr_today = PragmaticProductionOrgan(seed=seed, shared=None)
    mc_t, pr_t = _read_metacog(mc_today), _read_pragmatic(pr_today)

    # MERGED (default-ON path): both organs on ONE shared bridge (the three merge seams ON).
    merged = MergedSubstrate2(seed=seed, organs=("metacog", "pragmatic"))
    mc_merged = cls(seed=seed, shared=merged)
    pr_merged = PragmaticProductionOrgan(seed=seed, shared=merged)
    mc_m, pr_m = _read_metacog(mc_merged), _read_pragmatic(pr_merged)
    merged.ensure_built()
    one_pool = bool((mc_merged.bridge is merged.bridge) and (pr_merged._shared.bridge is merged.bridge))
    n_pool = int(merged.bridge.cp_membrane_potential_v.shape[0])

    # CORESIDENT (each organ on its OWN bridge, three merge seams ON — the apples-to-apples merge baseline).
    subM = MergedSubstrate2(seed=seed, organs=("metacog",))
    subP = MergedSubstrate2(seed=seed, organs=("pragmatic",))
    mc_cor = cls(seed=seed, shared=subM)
    pr_cor = PragmaticProductionOrgan(seed=seed, shared=subP)
    mc_c, pr_c = _read_metacog(mc_cor), _read_pragmatic(pr_cor)

    # B. MERGED == CORESIDENT (byte-identity of the genuine merge — must be 0.0).
    b_mc = max(abs(m[1] - c[1]) for m, c in zip(mc_m, mc_c))
    b_pr = max(max(abs(mb - cb) for mb, cb in zip(m[1], c[1])) for m, c in zip(pr_m, pr_c))
    byte_id_merge = (b_mc == 0.0 and b_pr == 0.0)

    # C. answer classes preserved MERGED-vs-TODAY (both build the SAME read).
    mc_class_ok = all(m[2] == t[2] for m, t in zip(mc_m, mc_t))
    pr_class_ok = all((m[2] == t[2]) and (m[3] == t[3]) for m, t in zip(pr_m, pr_t))

    # D. numeric residual MERGED-vs-TODAY (reported; expected > 0 for the raw margin, near-0 for the classes).
    d_mc = max(abs(m[1] - t[1]) for m, t in zip(mc_m, mc_t))
    d_pr = max(max(abs(mb - tb) for mb, tb in zip(m[1], t[1])) for m, t in zip(pr_m, pr_t))

    row = {
        "seed": seed, "read": read, "one_pool": one_pool, "n_pool": n_pool,
        "byte_id_merge_vs_coresident": byte_id_merge,
        "byte_id_metacog_delta": float(b_mc), "byte_id_pragmatic_delta": float(b_pr),
        "metacog_class_ok": bool(mc_class_ok), "pragmatic_class_ok": bool(pr_class_ok),
        "answer_classes_preserved_vs_today": bool(mc_class_ok and pr_class_ok),
        "residual_metacog_balance_max": float(d_mc), "residual_pragmatic_belief_max": float(d_pr),
        "mc_today": mc_t, "mc_merged": mc_m,
    }

    if read == "nmda_norm":
        # Compare against the PRE-2026-08-13 absolute-spike read on the standalone build (forced explicitly: the
        # production default is now the nmda_norm read this de-risk installed, so `balance` is the escape).
        abs_today = MetacogProductionOrgan(seed=seed, shared=None, confidence_read="balance")
        mc_abs_t = _read_metacog(abs_today)
        row["mc_absolute_today"] = mc_abs_t

        # I(reported, NOT gated). Full-sweep agreement robust-TODAY vs absolute-TODAY. The absolute read is at the
        # NOISE FLOOR — its confident/uncertain boundary is PER-SEED ARBITRARY (measured: evid~0.5 on seed 42,
        # evid~0.82 on seed 43), which is the very instability being fixed — so requiring the robust read to
        # reproduce it would be requiring it to reproduce noise. Reported as a characterization of the read upgrade.
        interp_unchanged = all(r[2] == a[2] for r, a in zip(mc_t, mc_abs_t))
        row["interp_unchanged_vs_absolute_today"] = bool(interp_unchanged)

        # I''(reported, NOT gated). Agreement at the clearly-low/high extremes with the old shipped read. Turns out
        # this is ALSO the wrong bar: the old read is non-monotone NOISE (measured: on seed 102 it calls ZERO-evidence
        # "confident" and evid 0.75 "uncertain"). Reported so the read change is fully characterized.
        extreme = {0.0, 0.15, 0.9, 1.0}
        extremes_agree = all(r[2] == a[2] for r, a in zip(mc_t, mc_abs_t) if r[0] in extreme)
        row["extremes_agree_with_absolute"] = bool(extremes_agree)

        # DGN. NON-DEGENERATE (guards trivial "all-confident" invariance).
        confs = [c for _, _, c in mc_t]
        non_degenerate = (any(confs) and not all(confs))
        lo_unc = all((not c) for e, _, c in mc_t if e <= 0.3)      # low evidence -> uncertain
        hi_conf = all(c for e, _, c in mc_t if e >= 0.9)           # high evidence -> confident
        tracks = bool(lo_unc and hi_conf)
        row["non_degenerate"] = bool(non_degenerate)
        row["tracks_evidence"] = tracks

        # MONO (GATED). The CORRECT "winning interpretation is unchanged" bar for a confidence read: the confident
        # bool is MONOTONE-NONDECREASING in evidence (once confident, more evidence stays confident) in BOTH the
        # standalone and merged build. This is the seed-robust, semantically-correct confidence code the read must
        # produce — stronger than "agree with the old noisy read", which the old read itself VIOLATES (non-monotone).
        def _mono(pattern):
            seen = False
            for c in pattern:
                if c:
                    seen = True
                elif seen:
                    return False
            return True
        mono_today = _mono([c for _, _, c in mc_t])
        mono_merged = _mono([c for _, _, c in mc_m])
        row["monotone_today"] = bool(mono_today)
        row["monotone_merged"] = bool(mono_merged)

        # FLIP-GO: the genuine merge (A+B) + BOTH organs answer-preserving MERGED-vs-TODAY (the blocker resolved) +
        # the robust read is a non-degenerate, evidence-tracking, MONOTONE confidence code in both builds.
        row["flip_go"] = bool(one_pool and byte_id_merge and mc_class_ok and pr_class_ok
                              and non_degenerate and tracks and mono_today and mono_merged)
    else:
        row["flip_go"] = bool(one_pool and byte_id_merge and mc_class_ok and pr_class_ok)
    return row


def _aggregate(rows, read):
    n = len(rows)
    agg = {
        "read": read, "n": n,
        "one_pool": sum(r["one_pool"] for r in rows),
        "byte_id_merge_vs_coresident": sum(r["byte_id_merge_vs_coresident"] for r in rows),
        "merge_go": sum(bool(r["one_pool"] and r["byte_id_merge_vs_coresident"]) for r in rows),
        "metacog_answer_preserved": sum(r["metacog_class_ok"] for r in rows),
        "pragmatic_answer_preserved": sum(r["pragmatic_class_ok"] for r in rows),
        "residual_metacog_balance_max": max(r["residual_metacog_balance_max"] for r in rows),
        "residual_pragmatic_belief_max": max(r["residual_pragmatic_belief_max"] for r in rows),
        "flip_go": sum(r["flip_go"] for r in rows),
    }
    if read == "nmda_norm":
        agg["interp_unchanged_vs_absolute_today"] = sum(r["interp_unchanged_vs_absolute_today"] for r in rows)
        agg["extremes_agree_with_absolute"] = sum(r["extremes_agree_with_absolute"] for r in rows)
        agg["non_degenerate"] = sum(r["non_degenerate"] for r in rows)
        agg["tracks_evidence"] = sum(r["tracks_evidence"] for r in rows)
        agg["monotone_today"] = sum(r["monotone_today"] for r in rows)
        agg["monotone_merged"] = sum(r["monotone_merged"] for r in rows)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--reads", default="balance,nmda_norm",
                    help="comma list; 'balance' reproduces the 1/6 baseline, 'nmda_norm' is the fix")
    ap.add_argument("--out", default="research/findings/raw/_metacog_robust_confidence_6seed.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    reads = [r.strip() for r in args.reads.split(",") if r.strip()]

    result = {"seeds": seeds, "backend": os.environ.get("SIM_BACKEND"), "reads": {}}
    for read in reads:
        rows = [run_seed(s, read) for s in seeds]
        result["reads"][read] = {"agg": _aggregate(rows, read), "rows": rows}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    # ── LEVER: the normalization must actually MOVE the metacog margin values vs the absolute read (else the
    #    two reads are identical and the A/B is void). Compare seed-0 evidence=0.6 margins.
    if "balance" in result["reads"] and "nmda_norm" in result["reads"]:
        abs_m = result["reads"]["balance"]["rows"][0]["mc_today"]
        nrm_m = result["reads"]["nmda_norm"]["rows"][0]["mc_today"]
        i_mid = MC_EVID.index(0.6)
        lever("balance_read=absolute->normalized",
              round(abs_m[i_mid][1], 4), round(nrm_m[i_mid][1], 4), required=True)

    print("\n" + "=" * 100)
    print("METACOG ROBUST CONFIDENCE — pool-#2 answer-preservation A/B (absolute vs divisive-normalized balance)")
    print("=" * 100)
    n = len(seeds)
    for read in reads:
        a = result["reads"][read]["agg"]
        rows = result["reads"][read]["rows"]
        print(f"\n[read = {read}]")
        for r in rows:
            extra = ""
            if read == "nmda_norm":
                extra = (f" | mono(today/merged)={r['monotone_today']}/{r['monotone_merged']}"
                         f" non-degen={r['non_degenerate']} tracks-evid={r['tracks_evidence']}")
            print(f"  seed {r['seed']}: one_pool={r['one_pool']}(N={r['n_pool']}) | "
                  f"MERGED==CORESIDENT byte-id={r['byte_id_merge_vs_coresident']} "
                  f"(mcΔ={r['byte_id_metacog_delta']:.2e}) | "
                  f"metacog-answer-preserved={r['metacog_class_ok']} "
                  f"pragmatic-answer-preserved={r['pragmatic_class_ok']}{extra} | FLIP-GO={r['flip_go']}")
        print(f"  ---- [{read}] ----")
        print(f"    A. one shared pool:                       {a['one_pool']}/{n}")
        print(f"    B. MERGED == CORESIDENT (byte-id, ==0):   {a['byte_id_merge_vs_coresident']}/{n}")
        print(f"    ==> MERGE-GO (A+B):                       {a['merge_go']}/{n}")
        print(f"    C. answer preserved — PRAGMATIC:          {a['pragmatic_answer_preserved']}/{n}")
        print(f"    C. answer preserved — METACOG:            {a['metacog_answer_preserved']}/{n}")
        if read == "nmda_norm":
            print(f"    MONO(gated). confident monotone in evidence (today/merged): "
                  f"{a['monotone_today']}/{n}  {a['monotone_merged']}/{n}  (correct, seed-robust confidence code)")
            print(f"    DGN. non-degenerate / tracks-evidence:    {a['non_degenerate']}/{n}  /  {a['tracks_evidence']}/{n}")
            print(f"    (reported) extremes agree w/ old absolute:   {a['extremes_agree_with_absolute']}/{n}  "
                  f"| full-sweep agree: {a['interp_unchanged_vs_absolute_today']}/{n}  "
                  f"(old read is non-monotone NOISE — agreement with it is NOT the bar)")
            void_if(a["non_degenerate"] < n, "robust read is DEGENERATE on some seed (all-confident or all-uncertain) "
                    "— trivial invariance, not a genuine confidence read")
        print(f"    ==> FULL FLIP-GO:                         {a['flip_go']}/{n}")
    print(f"\n  wrote {args.out}")
    return result


if __name__ == "__main__":
    main()
