#!/usr/bin/env python3
"""Step 1 (PACKET.md docs/drafts/word-role-frame line 81; RULINGS wrc-frame-go-1020,
wrc-frame-0-1021, wrc-frame-0-1021b, parallel-builds-1021).

Builds the production organ with the additive `frame=` stage (per-form trace
pops, the T->role projector at the role-layer entry step 0 chose, the
commit-gate-driven discharge pool, the slow-NMDA trace loop), trains it on the
same FACTS as step 0, and measures on dev seeds BEFORE any bar exists:

  (a) T->role co-firing events per presentation before discharge: drive-window
      steps where a trace pop fired AND a role ensemble fired AND the
      discharge pool was silent, per single-role word at pos=None;
  (b) NMDA frame gain under the per-item role-conductance reset: the same
      presentation run intact vs the frame lesion (GATE_TROLE_XMIT closed =
      the MM3 arm); gain for a word's own role = (intact+1)/(lesioned+1);
  (c) the -1021b tie-break arm: the pos=None (cat, eat) pair read in the 2x2
      {frame intact / frame lesion} x {exclusion intact / exclusion lesion}.

THRESHOLDS WRITTEN BEFORE THE RUN (packet: "if either is too small ... use an
AMPA frame WITH discharge"): (a) counts as present when >= 1 co-firing event
on >= 13/15 seeds; (b) counts as present when mean own-role gain >= 1.5 on
>= 13/15 seeds. Either failure -> report the AMPA-with-discharge fallback and
do not pick a bar here. The frame=None identity pins live in
tests/test_bg_commit.py:28-39 (commit battery) and
tests/test_word_role_commit.py:134-138 (count mode) and must pass unchanged.
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _ in range(3):
    _HERE = os.path.dirname(_HERE)
    if os.path.isdir(os.path.join(_HERE, "research")):
        break
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

from research.runners import word_role_parser_organ as W  # noqa: E402
from research.runners import declare_output  # noqa: E402
from tools.lab import assert_backend, attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

FACTS = [
    ("brain", "use", "spikes"),
    ("brain", "learn", "words"),
    ("brain", "store", "memory"),
    ("dog", "chase", "cat"),
    ("cat", "eat", "fish"),
    ("wolf", "hunt", "deer"),
]
AGENTS = ("dog", "wolf", "brain")
ACTIONS = ("chase", "eat", "hunt", "use")
PATIENTS = ("deer", "spikes", "memory", "words", "fish")
SINGLE_ROLE = AGENTS + ACTIONS + PATIENTS

OWN = {}
for _f in FACTS:
    for _k, _r in enumerate(W.ROLES):
        OWN.setdefault(_f[_k], _r)

GO_N = 15
MIN_SEEDS = 13
COFIRE_MIN = 1
GAIN_MIN = 1.5
THRESHOLDS = (
    "(a) >= %d co-firing event on >= %d/%d seeds; (b) mean own-role gain "
    ">= %.1f on >= %d/%d seeds; either absent -> AMPA-with-discharge "
    "fallback reported, no bar picked here." % (COFIRE_MIN, MIN_SEEDS, GO_N,
                                                GAIN_MIN, MIN_SEEDS, GO_N)
)


def run_frame(organ, seq, *, frame_lesion=False, excl_lesion=False,
              discharge_lesion=False):
    """Mirror of WordRoleParserOrgan._commit_run with the frame stage read:
    trace / discharge / co-firing counters. Frame path presents NO position."""
    def _go():
        b, xp, p, c = organ._bridge, organ._xp, organ.p, organ.c
        b.set_transmission_gate(W.GATE_WORD_TRANSMISSION, 1.0)
        b.set_transmission_gate(W.GATE_COMMIT_IN, 1.0)
        b.set_transmission_gate(W.GATE_SLOT_LOOP, 1.0)
        b.set_transmission_gate(W.GATE_COMMIT_EXCL,
                                0.0 if excl_lesion else 1.0)
        b.set_transmission_gate(W.GATE_TROLE_XMIT,
                                0.0 if frame_lesion else 1.0)
        b.set_transmission_gate(W.GATE_DISCHARGE,
                                0.0 if discharge_lesion else 1.0)
        b.set_transmission_gate(W.GATE_TRACE_LOOP, 1.0)
        rdev = {r: xp.asarray(organ.role_idx[r]) for r in W.ROLES}
        gdev = {r: xp.asarray(organ.gate_idx[r]) for r in W.ROLES}
        sdev = {r: xp.asarray(organ.slot_idx[r]) for r in W.ROLES}
        tdev = xp.asarray(organ.all_trace)
        ddev = xp.asarray(organ.discharge_idx)
        hold = int(c["HOLD_STEPS"])
        read = int(c["READ_STEPS"])
        out = []
        try:
            for i, (word, pos) in enumerate(seq):
                if i == 0:
                    W._reset_quiescent(b, xp, organ._snap)
                else:
                    organ._reset_keep_slots()
                cur = xp.zeros(organ._n, dtype=xp.float32)
                if word is not None:
                    dev = organ._word_dev(word)
                    if dev is not None:
                        cur[dev] = xp.float32(p["DRIVE_PA"])
                rc = {r: 0 for r in W.ROLES}
                gc = {r: 0 for r in W.ROLES}
                sc = {r: 0 for r in W.ROLES}
                trace_drive = 0
                cofire = 0
                first_disch = -1
                b.cp_external_input_current[:] = cur
                for step in range(int(c.get("WORD_STEPS",
                                            p["TEST_STEPS"]))):
                    b._run_one_simulation_step()
                    f = b.cp_firing_states
                    ts = int(W.to_host(f[tdev].astype(xp.float64).sum()))
                    ds = int(W.to_host(f[ddev].astype(xp.float64).sum()))
                    rsum = 0
                    for r in W.ROLES:
                        rc[r] += int(W.to_host(
                            f[rdev[r]].astype(xp.float64).sum()))
                        gc[r] += int(W.to_host(
                            f[gdev[r]].astype(xp.float64).sum()))
                        rsum += int(W.to_host(
                            f[rdev[r]].astype(xp.float64).sum()))
                    trace_drive += ts
                    if ts > 0 and ds > 0 and first_disch < 0:
                        first_disch = step
                    if ts > 0 and rsum > 0 and ds == 0:
                        cofire += 1
                b.cp_external_input_current[:] = 0.0
                trace_hold = 0
                for t in range(hold):
                    b._run_one_simulation_step()
                    f = b.cp_firing_states
                    trace_hold += int(W.to_host(
                        f[tdev].astype(xp.float64).sum()))
                    for r in W.ROLES:
                        gc[r] += int(W.to_host(
                            f[gdev[r]].astype(xp.float64).sum()))
                        if t >= hold - read:
                            sc[r] += int(W.to_host(
                                f[sdev[r]].astype(xp.float64).sum()))
                out.append({"word": word, "pos": pos, "role_counts": rc,
                            "gate_counts": gc, "slot_counts": sc,
                            "trace_drive": trace_drive,
                            "trace_hold": trace_hold,
                            "cofire_steps": cofire,
                            "first_discharge_step": first_disch})
        finally:
            b.cp_external_input_current[:] = 0.0
            for gname in (W.GATE_WORD_TRANSMISSION, W.GATE_COMMIT_IN,
                          W.GATE_SLOT_LOOP, W.GATE_COMMIT_EXCL,
                          W.GATE_TROLE_XMIT, W.GATE_DISCHARGE,
                          W.GATE_TRACE_LOOP):
                b.set_transmission_gate(gname, 1.0)
        return out
    return W._isolated(_go)


def own_gain(open_counts, lesion_counts, word):
    r = OWN.get(word)
    if r is None:
        return None
    return (float(open_counts[r]) + 1.0) / (float(lesion_counts[r]) + 1.0)


def seed_probe(seed):
    params, commit = W.commit_organ_params(0)
    organ = W.WordRoleParserOrgan(sorted(FACTS), seed, params, commit,
                                  frame=W.frame_organ_params()).train()
    rec = {"seed": int(seed), "frame_built": bool(organ.f is not None)}

    # (a)+(b): single-role words at pos=None, intact vs frame lesion.
    per = {}
    for w in SINGLE_ROLE:
        open_items = run_frame(organ, [(w, None)])
        les_items = run_frame(organ, [(w, None)], frame_lesion=True)
        per[w] = {
            "own_role": OWN.get(w),
            "cofire": int(open_items[0]["cofire_steps"]),
            "first_discharge_step": int(open_items[0]["first_discharge_step"]),
            "trace_drive": int(open_items[0]["trace_drive"]),
            "trace_hold": int(open_items[0]["trace_hold"]),
            "role_counts_open": open_items[0]["role_counts"],
            "role_counts_lesion": les_items[0]["role_counts"],
            "gain_own": own_gain(open_items[0]["role_counts"],
                                 les_items[0]["role_counts"], w),
        }
    rec["ab_singles"] = per
    gains = [v["gain_own"] for v in per.values() if v["gain_own"] is not None]
    cofire_any = any(v["cofire"] >= COFIRE_MIN for v in per.values())
    rec["a_cofire_present"] = bool(cofire_any)
    rec["b_gain_mean"] = float(np.mean(gains)) if gains else None
    rec["b_gain_present"] = bool(rec["b_gain_mean"] is not None
                                 and rec["b_gain_mean"] >= GAIN_MIN)

    # (c): the -1021b tie-break 2x2 on the pos=None (cat, eat) pair.
    arms = {}
    for fl in (False, True):
        for el in (False, True):
            key = "frame_%s/excl_%s" % ("lesion" if fl else "intact",
                                        "lesion" if el else "intact")
            items = run_frame(organ, [("cat", None), ("eat", None)],
                              frame_lesion=fl, excl_lesion=el)
            rd = organ._commit_read(items)
            arms[key] = {"per_item": rd["per_item"],
                         "new": [pi["new"] for pi in rd["per_item"]],
                         "role": [pi["role"] for pi in rd["per_item"]]}
    rec["c_tie_2x2"] = arms
    want = ["agent", "action"]
    correct = {k: bool(v["role"] == want) for k, v in arms.items()}
    rec["c_correct"] = correct
    rec["c_tie_broken_frame_intact"] = bool(
        correct["frame_intact/excl_intact"]
        and correct["frame_intact/excl_lesion"])
    # Attribution: reads earned by the frame vs the frame lesion (excl intact).
    rec["c_attributable"] = attributable_to(
        "tie-break read (frame intact vs frame lesion, excl intact)",
        float(correct["frame_intact/excl_intact"]),
        float(correct["frame_lesion/excl_intact"]))
    return rec


def evaluate(records):
    n = len(records)
    n_ok_frame = sum(1 for r in records if r["frame_built"])
    n_cofire = sum(1 for r in records if r["a_cofire_present"])
    n_gain = sum(1 for r in records if r["b_gain_present"])
    n_tie = sum(1 for r in records if r["c_tie_broken_frame_intact"])
    cofire_ok = n_cofire >= MIN_SEEDS
    gain_ok = n_gain >= MIN_SEEDS
    if cofire_ok and gain_ok:
        verdict = ("proceed as designed: NMDA frame co-firing %d/%d, gain "
                   "%d/%d (mean gain %.2f); tie broken with and without the "
                   "exclusion lesion on %d/%d seeds"
                   % (n_cofire, n, n_gain, n,
                      float(np.mean([r["b_gain_mean"] for r in records
                                     if r["b_gain_mean"] is not None])),
                      n_tie, n))
    else:
        verdict = ("FALLBACK (packet rule): frame measurements too small -- "
                   "co-firing %d/%d, gain %d/%d; use an AMPA frame WITH "
                   "discharge (never drop discharge), report the counts"
                   % (n_cofire, n, n_gain, n))
    return {"thresholds": THRESHOLDS, "n_seeds": n,
            "n_frame_built": n_ok_frame, "n_cofire_ok": n_cofire,
            "n_gain_ok": n_gain, "n_tie_broken": n_tie,
            "cofire_ok": cofire_ok, "gain_ok": gain_ok,
            "verdict": verdict}


def main(argv=None):
    ap = argparse.ArgumentParser(description="wrc-frame step 1 frame build")
    ap.add_argument("--seeds", default="1-15")
    ap.add_argument("--out",
                    default="research/findings/raw/_wrc_frame/step1")
    args = ap.parse_args(argv)
    backend = assert_backend("numpy", note="wrc-frame step 1 official run")
    if "-" in args.seeds and "," not in args.seeds:
        a, b = args.seeds.split("-", 1)
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
    out = args.out
    os.makedirs(out, exist_ok=True)
    recs = []
    for seed in seeds:
        rec = seed_probe(seed)
        rec["backend"] = backend
        rec["device"] = "cpu"
        path = os.path.join(out, "seed%d.json" % seed)
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=2, sort_keys=True)
        declare_output(path)
        recs.append(rec)
        print("seed %d: frame=%s cofire=%s gain_mean=%s gain_ok=%s "
              "tie=%s -> %s"
              % (seed, rec["frame_built"], rec["a_cofire_present"],
                 rec["b_gain_mean"], rec["b_gain_present"],
                 rec["c_tie_broken_frame_intact"],
                 "proceed" if (rec["a_cofire_present"]
                               and rec["b_gain_present"]) else "fallback"))
    summary = evaluate(recs)
    summary["seeds"] = seeds
    summary["backend"] = backend
    summary["device"] = "cpu"
    v = Verdict("wrc-frame step 1 frame build (thresholds in this probe)")
    v.require("15 dev seeds run", len(recs) == GO_N)
    v.require("backend numpy", backend == "numpy")
    v.require("frame stage built on every seed",
              summary["n_frame_built"] == GO_N)
    v.require("thresholds: co-firing present on >= %d seeds" % MIN_SEEDS,
              summary["cofire_ok"])
    v.require("thresholds: gain present on >= %d seeds" % MIN_SEEDS,
              summary["gain_ok"])
    dec = v.decide(go=bool(summary["cofire_ok"] and summary["gain_ok"]),
                   verbose=False)
    summary.update(dec)
    summary_path = os.path.join(out, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    declare_output(summary_path)
    print("VERDICT:", summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())