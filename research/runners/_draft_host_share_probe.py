"""Host-share instrument, FIRST SEAM (plan section 2, runbook task 2).

SEAM UNDER PROBE
  `research/runners/metacog_production_organ.mean_role_confidence` -- the HOST ARITHMETIC that converts
  the composer's per-role matched-filter scores into a confidence scalar. On the DEFAULT production path
  (`BRAIN_METACOG_SPIKING_MARGIN` unset) it is pure host arithmetic: the per-role preference chain
  `margin_snr > margin_norm > margin > confidence`, averaged over roles. The one non-host member,
  `margin_spiking` (a genuine spiking winner-vs-runner-up read off the recall circuit's own Izhikevich
  cleanup competition), is flag-gated OFF and not wired (see
  research/findings/2026-09-05-metacog-spiking-recall-margin-derisk-PARTIAL.md).

  The confidence feeds `evidence_from_role_conf` -> `MetacogProductionOrgan.judge` -> the `confident` bool
  (the reply's COMMITMENT: assert vs honest hedge). So this is a COMMITMENT-STRENGTH seam: the host computes
  the evidence, the spiking WTA maps it to a commitment.

THREE MEASUREMENTS (plan section 2, kept conceptually separate)
  (a) DECODING -- record the host decision (`host_mrc`) and the per-role neural inputs on the SAME trace.
      Across the sweep, the host-vs-neural correlation is the screen: does the host's arithmetic faithfully
      track the spiking read, or compute something arbitrary? A screen, NEVER evidence the neural system made
      the decision.
  (b) SUBSTITUTION -- replace the host decision with a neural-derived one (the spiking margin, `spiking_mrc`)
      and observe the downstream effect: does the metacog `confident` bool change?
  (c) INTERVENTION (the primary causal measurement) -- hold the neural state (the trace) FIXED, sweep the host
      decision (`mrc`) over a grid, and measure whether the downstream `confident` follows the host's choice
      (monotonic, host-controllable) or saturates toward its own preference (the spiking WTA owns the
      commitment).

SCOPE / COST (mirrors `_metacog_spiking_recall_margin_derisk.py`)
  Builds `OneBrainComposer` DIRECTLY rather than the full multi-organ chat brain (~180s/build on ~15
  unrelated faculties) because the mechanism under test is entirely internal to the composer + the metacog
  organ. This trades exact production-config fidelity (no `enable_attributed` filler role, no
  `vocab_headroom` reserved slots) for a ~40x faster, still-genuinely-seed-varying probe. numpy/CPU
  throughout (cost-routed). NO `sim/` / `webapp/` edit: production code is imported and wrapped, never changed.

DEV SEEDS ONLY (runbook hard rule): 7, and 1-15 for a small dev sweep. NEVER 42/43/44/100/101/102. Every
result is labelled "DEV, not a verdict".

Usage:
  SIM_BACKEND=numpy python -m research.runners._draft_host_share_probe --seeds 7 \
      --out research/findings/raw/_draft_host_share_probe/dev_seed7_smoke.json
  # small dev sweep (under memcap, CPU):
  bash tools/memcap.sh 8 -- env SIM_BACKEND=numpy python -m research.runners._draft_host_share_probe \
      --seeds 1 2 3 4 5 6 7 --out research/findings/raw/_draft_host_share_probe/dev_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# numpy/CPU throughout (cost-routed; no GPU -- the GPU auto-swap would stop the local model mid-session).
os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
# ON for this whole process so every role chip carries BOTH `margin` (host, always computed) AND
# `margin_spiking` (spiking, additive) -- one composer instance / one query serves BOTH the host and the
# neural arms with NO confound from separately-built neurons. (The DEFAULT production path leaves this unset;
# this probe sets it deliberately so the substitution arm has a neural-derived read to substitute in.)
os.environ["BRAIN_METACOG_SPIKING_MARGIN"] = "1"

import numpy as np

from research.runners.one_brain_composer import OneBrainComposer
from research.runners.metacog_production_organ import (
    mean_role_confidence, evidence_from_role_conf, MetacogProductionOrgan,
    ROLE_CONF_LO, ROLE_CONF_HI,
)

# DEV facts + vocab (a tiny closed set so a dev-seed recall is reliable and the trace carries real per-role
# scores). Distinct from the gate-seed derisk's facts so this probe is not a copy of its panel.
FACTS = [("river", "flow", "sea"), ("bird", "fly", "sky"), ("worm", "dig", "soil"),
         ("ball", "roll", "hill"), ("stone", "sink", "pond")]
VOCAB = sorted({w for f in FACTS for w in f} | {"brain", "use", "spikes", "dog", "cat"})

# The intervention grid: host-decision (mrc) values to sweep with the neural state held fixed. Spans the
# evidence-stretch band [ROLE_CONF_LO, ROLE_CONF_HI] plus the out-of-band extremes so a saturated WTA (all 0
# or all 1) is visible.
MRC_GRID = [0.0, 0.15, 0.30, 0.40, 0.50, 0.60, 0.70, 1.0]


def _host_mrc(activity):
    """The HOST decision: mirror `mean_role_confidence`'s host chain EXACTLY but never prefer `margin_spiking`
    (the default production read). This is what the host computes on a default turn."""
    roles = (activity or {}).get("roles") or []
    vals = []
    for r in roles:
        snr = r.get("margin_snr")
        mn = r.get("margin_norm")
        m = r.get("margin")
        v = snr if snr is not None else (mn if mn is not None else (m if m is not None else r.get("confidence")))
        if v is not None:
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def _neural_mrc(activity):
    """The NEURAL-derived decision: `mean_role_confidence` with the spiking flag ON, so it prefers
    `margin_spiking` (the recall circuit's own spiking read) over the host chain. This is the substitution
    candidate -- a confidence the host did NOT compute."""
    return mean_role_confidence(activity)


def _role_fields(activity):
    """The per-role neural inputs on the trace (the raw fields the host arithmetic reads). Recorded for the
    decoding screen."""
    roles = (activity or {}).get("roles") or []
    out = []
    for r in roles:
        out.append({k: (None if r.get(k) is None else float(r.get(k)))
                    for k in ("margin", "margin_norm", "margin_snr", "margin_spiking", "confidence")})
    return out


def _confident(organ, mrc):
    """mrc -> evidence -> metacog WTA -> the downstream `confident` bool (the commitment)."""
    if mrc is None:
        return {"evidence": None, "confident": None, "balance": None}
    ev = evidence_from_role_conf(mrc)
    j = organ.judge(ev if ev is not None else 0.0)
    return {"evidence": (None if ev is None else float(ev)), "confident": bool(j["confident"]),
            "balance": float(j["balance"]), "threshold": float(j["threshold"])}


def build_composer(seed):
    comp = OneBrainComposer(seed=seed, trace=True, vocab=VOCAB)
    for a, v, p in FACTS:
        comp.hear(f"{a} {v} {p}", polarity="AFFIRM")
    return comp


def ask_and_trace(comp, a, v):
    ans = comp.query_patient(a, v)
    return ans, comp.last_trace


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run_seed(seed):
    comp = build_composer(seed)
    organ = MetacogProductionOrgan(seed=seed)
    organ.ensure_built()

    queries = [(f[0], f[1]) for f in FACTS]  # one query per taught fact (a small within-seed panel)
    rows = []
    for a, v in queries:
        ans, trace = ask_and_trace(comp, a, v)
        if ans is None or not trace:
            rows.append({"agent": a, "verb": v, "abstained": True, "answer": None})
            continue
        host = _host_mrc(trace)            # (a) the host decision
        neural = _neural_mrc(trace)        # the neural-derived decision (substitution candidate)
        fields = _role_fields(trace)       # (a) the per-role neural inputs
        # (b) SUBSTITUTION: downstream confident at the host decision vs the neural decision.
        sub_host = _confident(organ, host)
        sub_neural = _confident(organ, neural)
        # (c) INTERVENTION: hold the trace fixed, sweep the host decision, read the downstream confident.
        intervention = [{"mrc": m, **_confident(organ, m)} for m in MRC_GRID]
        conf_seq = [pt["confident"] for pt in intervention if pt["confident"] is not None]
        # host-controllable: can the host's mrc choice achieve BOTH confident values on this fixed trace?
        host_controls = (len(set(conf_seq)) == 2)
        # monotonic: a single 0->1 transition (the WTA faithfully transmits the host's evidence)?
        n_trans = sum(1 for i in range(1, len(conf_seq)) if conf_seq[i] != conf_seq[i - 1])
        monotonic = (n_trans <= 1)
        rows.append({
            "agent": a, "verb": v, "abstained": False, "answer": ans,
            "host_mrc": host, "neural_mrc": neural, "role_fields": fields,
            "substitution": {"host": sub_host, "neural": sub_neural,
                             "confident_changed": (sub_host["confident"] != sub_neural["confident"])},
            "intervention": intervention,
            "intervention_summary": {"host_controls_confident": host_controls, "monotonic": monotonic,
                                      "n_transitions": n_trans, "confident_seq": conf_seq},
        })

    # (a) DECODING screen across the non-abstained rows: does the host's arithmetic track the neural read?
    answered = [r for r in rows if not r["abstained"]]
    if len(answered) >= 2:
        host_vec = [r["host_mrc"] for r in answered if r["host_mrc"] is not None]
        neural_vec = [r["neural_mrc"] for r in answered if r["neural_mrc"] is not None]
        n = min(len(host_vec), len(neural_vec))
        decoding_corr = _pearson(host_vec[:n], neural_vec[:n]) if n >= 2 else None
    else:
        decoding_corr = None

    # a simple per-seed host-share indicator (DEV screen, NOT a verdict): the fraction of answered rows on
    # which the host's mrc choice controls the downstream confident AND the WTA transmits it monotonically.
    if answered:
        host_share = float(sum(1 for r in answered
                               if r["intervention_summary"]["host_controls_confident"]
                               and r["intervention_summary"]["monotonic"]) / len(answered))
    else:
        host_share = None
    return {"seed": seed, "rows": rows, "decoding_host_vs_neural_corr": decoding_corr,
            "host_share_indicator": host_share, "organ_threshold": float(organ.threshold),
            "organ_calib": organ.calib, "label": "DEV, not a verdict"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    per_seed = []
    for seed in args.seeds:
        print(f"=== host-share probe, dev seed {seed} ===", flush=True)
        r = run_seed(seed)
        per_seed.append(r)
        print(f"  answered {sum(1 for row in r['rows'] if not row['abstained'])}/{len(r['rows'])}, "
              f"decoding_corr={r['decoding_host_vs_neural_corr']}, "
              f"host_share_indicator={r['host_share_indicator']}", flush=True)

    result = {"seams": ["metacog mean_role_confidence (commitment-strength)"], "seeds": args.seeds,
              "per_seed": per_seed, "label": "DEV, not a verdict",
              "note": "host-share instrument first seam; plan section 2 (decoding/substitution/intervention)."}
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2))
        print(f"wrote {p}", flush=True)
    print(json.dumps({s["seed"]: {"answered": sum(1 for r in s["rows"] if not r["abstained"]),
                                  "host_share": s["host_share_indicator"]} for s in per_seed}, indent=2))


if __name__ == "__main__":
    main()
