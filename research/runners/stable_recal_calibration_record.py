"""Writes research/findings/raw/_stable_recal/swap_calibration_record.json (research/stable-recal).

The record used to be assembled by hand. This generator replaces that: every figure that a cited artifact holds is READ
from the artifact (organ GO-gate counts and failing checks, run start times from the artifacts' provenance sidecars, the
w_gate_ws primitive sweeps and organ runs, the read-reliability of the chosen recurrent weight), and the production and
re-calibrated values are taken from `research.runners.stable_recal.SWAP`, so the record cannot disagree with the code.
What stays typed here is the JUDGMENT behind each constant (the `rule` strings), which no artifact can hold. It runs no
brain and needs no GPU. Run as a module so the provenance sidecar (swap_calibration_record.json.prov.json) is written:

  .venv/bin/python -m research.runners.stable_recal_calibration_record \
      --out research/findings/raw/_stable_recal/swap_calibration_record.json

Exit code 1 (and no record) when a cited artifact is missing or the record's values differ from `SWAP`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

RAW = "research/findings/raw/_stable_recal"
SWAP_SEEDS = (7, 8)
BUS_SEEDS = (7,)
BUS_SOURCES = ("research/findings/raw/_stable_integration/bus_ignition_curve_s7.json",
               "research/findings/raw/_stable_integration/bus_quorum_w5_s7.json")
ORGAN_SEEDS = tuple(range(1, 16))
SUSTAIN = RAW + "/swap_sustain_nws"
PRIM = RAW + "/swap_primitives"
GATE_TW9 = RAW + "/swap_organ_gate"
GATE_TW36 = RAW + "/swap_organ_gate_tw36"
GATE_WG1 = RAW + "/swap_organ_gate_wgate1"


def _load(path):
    with open(os.path.join(_REPO, path)) as f:
        return json.load(f)


def _arm(path):
    """The single arm of a probe artifact: (arm name, arm record)."""
    arms = _load(path)["arms"]
    (name, rec), = arms.items()
    return name, rec


def _started(path):
    return _load(path + ".prov.json")["started"]


def _organ_dir(files):
    """GO-gate summary of organ runs: count, failing checks -> seeds, and the runs' start times (provenance sidecars)."""
    go, fails, starts, per_seed = 0, {}, [], {}
    for path in files:
        _, rec = _arm(path)
        res = rec["result"]
        seed = int(res["seed"])
        go += bool(res["seed_go"])
        bad = sorted(k for k, v in res["go_gate"].items() if not v)
        for k in bad:
            fails.setdefault(k, []).append(seed)
        starts.append(_started(path))
        per_seed[seed] = {"go": bool(res["seed_go"]), "failed_checks": bad}
    return {"n_seeds": len(files), "n_go": go, "failing_checks_to_seeds": {k: sorted(v) for k, v in sorted(fails.items())},
            "first_run_started": min(starts), "last_run_started": max(starts), "per_seed": dict(sorted(per_seed.items()))}


def _read15(seed, w_rec):
    d = _load("%s/nws4_s%d.json" % (SUSTAIN, seed))
    rows = [r for r in d["rows"] if r["w_rec"] == w_rec and r["drive"] == 2500.0]
    assert len(rows) == 1, (seed, w_rec, len(rows))
    r = rows[0]["read15"]
    return {"plateau": r["plateau"], "worst_window": r["min"], "frac_windows_below_half": r["frac_below_half"]}


def _w_gate_ws_evidence():
    prim = {}
    for s in SWAP_SEEDS:
        _, rec = _arm("%s/w_gate_ws_s%d.json" % (PRIM, s))
        prim["s%d" % s] = {"%g" % r["w_gate_ws"]: {"gate_primitive_holds": r["gate_primitive_holds"],
                                                    "held_content_survives_own_gate": r["held_content_survives_own_gate"]}
                           for r in rec["rows"]}
    organ = {}
    for label, tmpl in (("w_gate_ws=4 (the re-calibration)", GATE_TW36 + "/recal_on_s%d.json"),
                        ("w_gate_ws=1 (all else as the re-calibration)", GATE_WG1 + "/wgate1_on_s%d.json")):
        organ[label] = {}
        for s in SWAP_SEEDS:
            _, rec = _arm(tmpl % s)
            res = rec["result"]
            organ[label]["s%d" % s] = {
                "seed_go": res["seed_go"], "swap_ok": res["go_gate"]["swap_ok"],
                "failed_checks": sorted(k for k, v in res["go_gate"].items() if not v),
                "newcomer_ignited_after_swap": res["residual"]["headline"]["new_ignited"],
                "newcomer_rate_after_swap": round(res["residual"]["headline"]["new_rate_post"], 4),
                "artifact": tmpl % s}
    return {"primitive_sweep": prim, "organ_gate": organ,
            "organ_values_run": [1, 4],
            "reading": "the vacancy-gate primitive holds at w_gate_ws 1, 2, 4, 8 on both seeds and fails at 16 (seed 7), 32 "
                       "(seed 8) and 100; the FULL ORGAN was run at 1 and 4 only: at 1 the swap fails on seed 7 (the newcomer "
                       "never ignites, rate 0.037) and passes on seed 8, at 4 both seeds swap. 2 and 8 were not run through the "
                       "organ; 4 is the middle of the primitive's admissible 2-8 band, not a measured optimum"}


def build():
    from research.runners import stable_recal as SR
    S = SR.SWAP

    def pr(name):
        return {"production": S[name][0], "recal": S[name][1]}

    prod_on = [p for s in SWAP_SEEDS for p in ("%s/production_on_s%d.json" % (GATE_TW9, s),)]
    prod_off = [p for s in SWAP_SEEDS for p in ("%s/production_off_s%d.json" % (GATE_TW9, s),)]
    tw9 = _organ_dir(["%s/recal_on_s%d.json" % (GATE_TW9, s) for s in ORGAN_SEEDS])
    tw36 = _organ_dir(["%s/recal_on_s%d.json" % (GATE_TW36, s) for s in ORGAN_SEEDS])
    po_on = _organ_dir(prod_on)
    po_off = _organ_dir(prod_off)
    n_checks = len(_arm(prod_on[0])[1]["result"]["go_gate"])
    nbad_on = sorted({len(v["failed_checks"]) for v in po_on["per_seed"].values()})
    sustain = {"s%d" % s: _read15(s, 6.5) for s in SWAP_SEEDS}
    # the claim "no other constant changed between the window-9 and window-36 runs" is checked, not typed
    cal9 = [_arm("%s/recal_on_s%d.json" % (GATE_TW9, s))[1]["calibration"] for s in ORGAN_SEEDS]
    cal36 = [_arm("%s/recal_on_s%d.json" % (GATE_TW36, s))[1]["calibration"] for s in ORGAN_SEEDS]
    same_but_window = all({k: v for k, v in b.items() if k != "timing_window"} == a and "timing_window" not in a
                          and b["timing_window"] == S["timing_window"][1] for a, b in zip(cal9, cal36))
    if not same_but_window:
        raise ValueError("the window-9 and window-36 organ runs differ in more than the timing window")

    const = {
        "w_rec": {**pr("w_rec"), "measured_at_recal_read15_nws4_drive2500": sustain,
                  "rule": "at nws 4 the sustain knee lies between 4.0 (0.28-0.38 of read windows below half the plateau) and 4.5 "
                          "(none; plateau 0.067-0.068); the production weight is deliberately supra-critical (the incumbent "
                          "resists inhibition), so 6.5 (about 1.4x the knee; plateau 0.084, worst window 0.069-0.076) was "
                          "chosen: a judgment, not a formula; the organ gate passes at it",
                  "source": ["%s/nws4_s%d.json" % (SUSTAIN, s) for s in SWAP_SEEDS]},
        "norm_ws_w": {**pr("norm_ws_w"),
                      "rule": "largest norm_pool->workspace weight whose sustained state reads reliably in the organ's 15-step "
                              "read window (read15: fraction of windows below half the plateau, steps 100-1000, thal 0, drive "
                              "2500, stable): at nws 16 and 8 every weight 4-6.5 has 0.53-0.93 of windows below half; at 6 still "
                              "0.23-0.42; at 5 none below half for w >= 5 but the worst window is only ~0.55 of the plateau "
                              "(0.041-0.045 vs 0.074); at 4 none below half for w >= 4.5 and the worst window is 0.82-0.90 of "
                              "the plateau. 4 was chosen over 5 for that margin (a judgment, recorded as such)",
                      "source": ["%s/nws%d_s%d.json" % (SUSTAIN, n, s) for n in (4, 5, 6, 8, 16) for s in SWAP_SEEDS]},
        "thal_tonic_pa": {**pr("thal_tonic_pa"),
                          "rule": "largest thalamic support with no spontaneous ignition over 1000 undriven steps on both seeds "
                                  "(w 6.5, nws 4): 25 pA leaves every pattern at 0.0 on seeds 7 and 8; at 50 pA seed 7 ignites "
                                  "pattern 0 by itself between steps 600 and 1000 (0.029 then 0.084), at 75 pA seed 8 ignites "
                                  "pattern 1 by itself; at 200 and 700 pA all three patterns fire together (0.05-0.06).",
                          "source": ["%s/swap_thal_support_w6p5_nws4_s%d.json" % (RAW, s) for s in SWAP_SEEDS]
                          + ["%s/swap_thal_support_s%d.json" % (RAW, s) for s in SWAP_SEEDS]},
        "solo_plateau": {**pr("solo_plateau"),
                         "rule": "the measured ignited plateau at (6.5, 4, 25) on the stable integrator; the ignition criterion "
                                 "stays IGNITE_FRAC (0.5) x plateau = 0.042, scoped to the swap organ (the global-stop workspace "
                                 "shares Rung-2's SOLO_PLATEAU and is not re-calibrated here)",
                         "source": ["%s/nws4_s%d.json" % (SUSTAIN, s) for s in SWAP_SEEDS]},
        "establish_pa": {**pr("establish_pa"),
                         "rule": "the production ratio of establishment drive to ignition knee (8000 / 2500 = 3.2) applied to "
                                 "the stable knee (300 pA sub, 400 pA ignites): 3.2 x 400",
                         "source": ["%s/swap_ignition_curve_w6p5_nws4_thal25_s%d.json" % (RAW, s) for s in SWAP_SEEDS]},
        "w_ws_occ": {**pr("w_ws_occ"),
                     "rule": "FORMULA, checked by a sweep: conserve the occupancy read's input (weight x plateau): 8 x (1/3) / "
                             "0.084 = 31.7; the w_ws_occ sweep 8, 16, 32, 64 keeps the gate and mismatch primitives holding on "
                             "both seeds (the veto strengthens with the weight)",
                     "source": ["%s/w_ws_occ_s%d.json" % (PRIM, s) for s in SWAP_SEEDS]},
        "w_pat_pred": {**pr("w_pat_pred"), "status": "FORMULA-ONLY (not measured)",
                       "rule": "the same conservation as w_ws_occ applied to the held-content prediction read (8 x (1/3) / "
                               "0.084 = 31.7, rounded to 32). No cited artifact varies w_pat_pred; the organ GO gate and the "
                               "mismatch sweeps were run with it at 32, which shows they pass with it there, not that 32 is "
                               "better than another value",
                       "source": []},
        "w_occ_gate": {**pr("w_occ_gate"),
                       "rule": "the vacancy veto knee: occupied-gate rate 0.215/0.199 at 20 (no veto), 0.093/0.077 at 40, "
                               "0.044/0.037 at 80 (empty gate ~0.19); 80 is where the veto saturates; no inversion up to 160 on "
                               "the stable integrator",
                       "source": ["%s/w_occ_gate_s%d.json" % (PRIM, s) for s in SWAP_SEEDS]},
        "w_pred_mm": {**pr("w_pred_mm"),
                      "rule": "the organ's own match bar (match mismatch rate < 0.02): 20 -> 0.082/0.084, 40 -> 0.038/0.038, "
                              "80 -> 0.009/0.013",
                      "source": ["%s/w_pred_mm_s%d.json" % (PRIM, s) for s in SWAP_SEEDS]},
        "w_gate_ws": {**pr("w_gate_ws"), "evidence": _w_gate_ws_evidence(),
                      "rule": "the gate primitive (empty admits, held vetoes, self survives) holds for 1-8 on both seeds and "
                              "fails at 16 (s7) / 32 (s8) and at the production 100; in the full organ w_gate_ws=1 fails the "
                              "swap on seed 7 (newcomer 0.037; swap_organ_gate_wgate1/) and 4 passes it; 4 is the middle of the "
                              "primitive's admissible 2-8 band (2 and 8 were not run through the organ)",
                      "source": ["%s/w_gate_ws_s%d.json" % (PRIM, s) for s in SWAP_SEEDS]
                      + ["%s/wgate1_on_s%d.json" % (GATE_WG1, s) for s in SWAP_SEEDS]},
        "timing_window": {**pr("timing_window"),
                          "rule": "the organ's stated rule is 3 ignition periods (TIMING_WINDOW comment in "
                                  "_gnw_neural_swap_intention_derisk.py); an ignited neuron spikes every 3 steps on the old "
                                  "integrator (9 = 3 spikes) and every 1/0.084 = 11.9 steps on the stable one (3 spikes = 36); a "
                                  "read-only timing instrument, not used by production replies. ORDER: the window was left at "
                                  "the production 9 first (GO 4 of 15, every failure the timing check) and was re-derived to 36 "
                                  "by this rule AFTER that result; see order_of_events",
                          "source": [GATE_TW9 + "/", GATE_TW36 + "/"]},
        "unchanged": {
            "salient_pa": 5000.0, "nonsalient_pa": 600.0, "boost_gain": 1.0, "max_boost": 0.16, "boost_window": 5,
            "evict_steps": 320, "reignite_hold": 150,
            "rule": "the mismatch detector reads 0.154 (salient) / 0.035-0.038 (non-salient) on stable, so the salient boost "
                    "reaches the 0.16 cap and the non-salient boost stays ~0.06; the eviction collapse knee on stable is "
                    "0.06-0.10 (held <= 0.05 s7, <= 0.08 s8; non-monotone above)",
            "source": ["%s/swap_collapse/collapse_s%d.json" % (RAW, s) for s in SWAP_SEEDS]},
    }
    for v in const.values():
        for p in v.get("source", []):
            if not os.path.exists(os.path.join(_REPO, p)):
                raise FileNotFoundError("cited source missing: " + p)

    # the typed prose above quotes these figures; refuse to write a record whose prose the artifacts do not support
    ev = const["w_gate_ws"]["evidence"]["organ_gate"]
    g4, g1 = ev["w_gate_ws=4 (the re-calibration)"], ev["w_gate_ws=1 (all else as the re-calibration)"]
    claims = {
        "window 9: GO 4 of 15, only the timing check fails": (tw9["n_go"] == 4 and list(tw9["failing_checks_to_seeds"]) == ["timing_ok"]),
        "window 36: GO 12 of 15, only the timing check fails, on seeds 1, 6, 12": (
            tw36["n_go"] == 12 and tw36["failing_checks_to_seeds"] == {"timing_ok": [1, 6, 12]}),
        "production on the stable integrator fails 9 of 11 checks": (nbad_on == [9] and n_checks == 11),
        "production on the old integrator is GO on both seeds": (po_off["n_go"] == 2),
        "w_gate_ws=4 swaps on seeds 7 and 8": (g4["s7"]["swap_ok"] and g4["s8"]["swap_ok"]),
        "w_gate_ws=1 fails the swap on seed 7 (newcomer rate 0.037, never ignites) and swaps on seed 8": (
            (not g1["s7"]["swap_ok"]) and (not g1["s7"]["newcomer_ignited_after_swap"])
            and g1["s7"]["newcomer_rate_after_swap"] == 0.0373 and g1["s8"]["swap_ok"]),
        "w_rec 6.5 reads without a window below half the plateau (plateau 0.084, worst window 0.069-0.076)": (
            all(v["frac_windows_below_half"] == 0.0 and 0.069 <= v["worst_window"] <= 0.076
                and round(v["plateau"], 3) == 0.084 for v in sustain.values())),
    }
    bad = [k for k, ok in claims.items() if not ok]
    if bad:
        raise ValueError("the record's prose is not supported by the cited artifacts: %s" % bad)
    ordered = tw9["last_run_started"] < tw36["first_run_started"]
    return {
        "runner": "research/runners/stable_recal_calibration_record.py (reads the cited artifacts; the rules are typed judgments)",
        "backend": "numpy", "device": "cpu",
        "what": "thought-swap workspace (research/runners/_gnw_neural_swap_intention_derisk.py) re-calibrated by its own rules "
                "on the stable integrator (SIM_STABLE_INTEGRATION=1); every value chosen on the isolated organ on dev seeds 7 "
                "and 8, never on a reply",
        "date": "2026-10-27",
        "seeds": list(SWAP_SEEDS),
        "organ_gate_seeds": list(ORGAN_SEEDS),
        "seed_roles": {
            "calibration_seeds": {"swap_workspace": list(SWAP_SEEDS), "gnw_bus": list(BUS_SEEDS),
                                  "bus_sources": list(BUS_SOURCES)},
            "reading": "dev seeds 7 and 8 are the CALIBRATION seeds of the swap workspace (bus: seed 7), so any OFF-versus-RECAL "
                       "comparison on them (the whole-brain dev check, reply identity, answer rate) is IN-SAMPLE. Held-out dev "
                       "seeds are the other organ-gate seeds (1-6, 9-15) for the isolated organ; no whole-brain comparison on a "
                       "held-out seed has been run (the pool battery of dev seeds 1-12 is written, not queued)"},
        "integrator_finding": "on the old integrator at the production constants the workspace ignition knee (2000 -> 2500 pA, "
                              "seed 7) is the drive at which 112-118 of 300 workspace cells leave the valid membrane range and "
                              "the 1/3 ignited rate is that runaway; on the stable integrator weight 34 with thal 700 is "
                              "depolarization block / spontaneous all-pattern activation and no weight 5-60 or normalization "
                              "0-64 gives a selective sustained ignition",
        "constants": const,
        "organ_go_gate": {
            "n_checks": n_checks,
            "production_old_integrator_s7_s8": {"n_go": po_off["n_go"], "n_seeds": po_off["n_seeds"],
                                                 "failing_checks_to_seeds": po_off["failing_checks_to_seeds"]},
            "production_stable_integrator_s7_s8": {"n_go": po_on["n_go"], "n_seeds": po_on["n_seeds"],
                                                    "failed_checks_per_seed": nbad_on,
                                                    "note": "nothing reaches the production ignition criterion"},
            "recal_stable_timing_window_9_dev1_15": tw9,
            "recal_stable_timing_window_36_dev1_15": tw36,
            "source": [GATE_TW9 + "/", GATE_TW36 + "/"]},
        "order_of_events": {
            "1_timing_window_9": "organ GO gate at the re-calibration with the production timing window 9: GO %d of %d on dev "
                                 "seeds 1-15; failing checks %s; runs started %s to %s (provenance sidecars)" % (
                                     tw9["n_go"], tw9["n_seeds"], tw9["failing_checks_to_seeds"], tw9["first_run_started"],
                                     tw9["last_run_started"]),
            "2_timing_window_re_derived": "after that result the window was re-derived to 36 from the organ's own rule (3 "
                                          "ignition periods = 3 spikes of an ignited neuron: 3 / 0.084 = 35.7); no other "
                                          "constant differs between the two sets of runs (checked from their recorded "
                                          "calibrations)",
            "3_timing_window_36": "same organ gate with window 36: GO %d of %d; failing checks %s; runs started %s to %s" % (
                tw36["n_go"], tw36["n_seeds"], tw36["failing_checks_to_seeds"], tw36["first_run_started"],
                tw36["last_run_started"]),
            "window_9_runs_started_before_window_36_runs": bool(ordered),
            "reading": "the %d of %d GO is the result AFTER a post-hoc correction of the timing instrument, not a pass of a "
                       "window fixed in advance; the window-9 result (%d of %d, %s) is what "
                       "the production window gives under this calibration" % (
                           tw36["n_go"], tw36["n_seeds"], tw9["n_go"], tw9["n_seeds"],
                           "every failure the timing check" if sorted(tw9["failing_checks_to_seeds"]) == ["timing_ok"]
                           else "failing checks %s" % sorted(tw9["failing_checks_to_seeds"]))},
        "probe": "research/runners/stable_integration_swap_ignition_probe.py",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = build()
    # the record must agree with the code: every constant's production and recal value is SWAP's by construction; assert
    # the order fact it states
    if not rec["order_of_events"]["window_9_runs_started_before_window_36_runs"]:
        print("the sidecar timestamps do not show the window-9 runs before the window-36 runs", file=sys.stderr)
        return 1
    out = os.path.join(_REPO, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print("[stable_recal_calibration_record] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
