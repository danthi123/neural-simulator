"""Aggregate the scratch outputs of the emotion-organ bounded-conductance de-risk into one committed JSON.

The raw per-seed records (about 20 MB: every read of every case) stay where the drivers wrote them (a scratch directory
on the workstation, not committed). This module reads them, computes the tallies the finding
`research/findings/2026-10-26-emotion-organ-bounded-conductance-derisk.md` cites, and writes ONE small artifact with the
sha256 of every input file and of the driver scripts that produced them (`docs/drafts/emo-bounded-cond/`). Run as a
module so the provenance door writes the `.prov.json` sidecar:

    python -m research.runners.emotion_category_bounded_aggregate --src <scratch dir> --pinned '<glob of pinned probe14_base_w*.json>' \\
        --out research/findings/raw/_emo_bounded/aggregate.json

Arms (see the finding): (a) the 10-case organ battery, flag off and on, held-out 301-360 and dev 1-8; (d) fresh seeds
401-420; (c) pinned-failure replay; (f) outlier-injection stress; (h) quench necessity; the handler shards (seed 101, group
emo, 4 arms); the determinism re-run. Nothing here runs a brain.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

CATS = ("joy", "fear", "anger", "sadness")
CASES = ("intact", "permuted", "affect_lesion", "word_lesion", "latch_lesion", "commit_lesion", "shuffle", "teach_swap",
         "ladder_only", "baseline")
READ_KEYS = ("category", "read_invariant", "co_commit", "slot_counts", "readout_counts", "gate_counts")
HANDLER_FEEL = ("feel_base", "feel_joy", "feel_fear", "feel_anger", "feel_sadness")
_INPUTS = []


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path):
    _INPUTS.append({"file": path, "sha256": _sha(path)})
    with open(path) as f:
        return json.load(f)


def _records(src, patterns):
    out = {}
    for pat in patterns:
        for f in sorted(glob.glob(os.path.join(src, "battery", pat))):
            for r in _load(f)["records"]:
                out[int(r["seed"])] = r
    return out


def _battery(recs, seeds):
    """Tallies of one flag over `seeds` (the dev battery's own 10-case verdicts, stored per record)."""
    seeds = [s for s in seeds if s in recs]
    fails = {s: [k for k, v in recs[s]["case_pass"].items() if not v] for s in seeds if not recs[s]["pass"]}
    slots = sorted(max(recs[s]["cases"]["intact"][c]["slot_counts"].values()) for s in seeds for c in CATS
                   if recs[s]["cases"]["intact"][c]["category"])
    reads = [x for s in seeds for d in recs[s]["cases"].values() for x in d.values()]
    cen = [recs[s]["census"] for s in seeds]
    eng = [recs[s]["engagement"] for s in seeds]
    return {
        "n_seeds": len(seeds), "n_pass_all_10_cases": sum(recs[s]["pass"] for s in seeds),
        "failing_seeds": {str(k): v for k, v in fails.items()},
        "per_case_pass": {c: sum(recs[s]["case_pass"][c] for s in seeds) for c in CASES},
        "intact_named_latch": {"n": len(slots), "min": slots[0], "median": slots[len(slots) // 2], "max": slots[-1],
                               "n_below_36": sum(x < 36 for x in slots)},
        "n_reads": len(reads), "n_co_commit_reads": sum(bool(x["co_commit"]) for x in reads),
        "census_all_steps_incl_training": {
            "seeds_with_excursion": sorted(s for s in seeds if recs[s]["census"]["n_oor_neurons_all_steps"] > 0),
            "n_seeds_with_excursion": sum(c["n_oor_neurons_all_steps"] > 0 for c in cen),
            "n_neurons_total": sum(c["n_oor_neurons_all_steps"] for c in cen),
            "min_v_mV": min(c["vmin"] for c in cen), "max_v_mV": max(c["vmax"] for c in cen)},
        "engagement": {"bounded_flag_values": sorted({str(x) for e in eng for x in e["bounded_flag"]}),
                       "bounded_env_values": sorted({x for e in eng for x in e["bounded_env"]}),
                       "bounded_readback_values": sorted({str(x) for e in eng for x in e["bounded_readback"]}),
                       "n_reads": sum(e["n_reads"] for e in eng), "oor_count_max": max(e["oor_count_max"] for e in eng),
                       "oor_count_session_max": max(e["oor_count_session_max"] for e in eng),
                       "n_seeds_oor_count_session_gt0": sum(e["oor_count_session_max"] > 0 for e in eng)},
    }


def _battery_block(src, pinned_glob):
    on = _records(src, ["on_w*.json", "on_dev.json"])
    off = _records(src, ["off_w*.json", "off_dev.json"])
    held, dev = list(range(301, 361)), list(range(1, 9))
    out = {"flag_on": {"held_out_301_360": _battery(on, held), "dev_1_8": _battery(on, dev)},
           "flag_off": {"held_out_301_360": _battery(off, held), "dev_1_8": _battery(off, dev)}}
    from research.runners import emotion_category_dev as D
    pinned = {}
    for f in sorted(glob.glob(pinned_glob)):
        for r in _load(f)["records"]:
            pinned[int(r["seed"])] = r
    same = [s for s in held if s in pinned and all(
        {k: off[s]["cases"][case][c][k] for k in READ_KEYS} == {k: pinned[s]["cases"][case][c][k] for k in READ_KEYS}
        for case in off[s]["cases"] for c in off[s]["cases"][case])]
    out["flag_off_vs_pinned_records"] = {"n_held_out_compared": sum(s in pinned for s in held), "n_identical": len(same),
                                         "pinned_n_pass_all_10_cases": sum(bool(D.seed_pass(pinned[s])["pass"]) for s in held if s in pinned)}
    paired = [max(on[s]["cases"]["intact"][c]["slot_counts"].values()) - max(off[s]["cases"]["intact"][c]["slot_counts"].values())
              for s in held if s in on and s in off and off[s]["pass"] for c in CATS]
    out["intact_latch_shift_on_minus_off"] = {"n": len(paired), "median": statistics.median(paired),
                                              "mean": round(statistics.mean(paired), 2), "min": min(paired), "max": max(paired)}
    fon, foff = _records(src, ["fon_*.json"]), _records(src, ["foff_*.json"])
    fresh = list(range(401, 421))
    out["fresh_401_420"] = {"flag_on": _battery(fon, fresh), "flag_off": _battery(foff, fresh)}
    spot = _records(src, ["spot_on.json"])
    out["determinism_flag_on_rerun"] = {str(s): bool(spot[s]["cases"] == on[s]["cases"]) for s in spot if s in on}
    return out


def _stress_block(src):
    out = {}
    for side in ("on", "off"):
        runs = {}
        for f in sorted(glob.glob(os.path.join(src, "battery", "stress_%s_inj_*.json" % side))):
            runs.update(_load(f)["runs"])
        by_c = {}
        for k, r in runs.items():
            cv = k.split("|")[2]
            d = by_c.setdefault(cv, {"n": 0, "ok": 0, "oor_session_max": 0})
            d["n"] += 1
            d["ok"] += bool(r["ok"])
            d["oor_session_max"] = max(d["oor_session_max"], r["eng"]["oor_session_max"])
        readback = sorted({str(x) for r in runs.values() for x in r["eng"]["readback"]})
        out["inject_flag_" + side] = {"seeds": sorted({int(k.split("|")[0]) for k in runs}), "by_C_pF": by_c,
                                      "bounded_readback_values": readback}
        pinned = _load(os.path.join(src, "battery", "stress_%s_pinned.json" % side))["runs"]
        out["pinned_replay_flag_" + side] = {s: {"ok": r["ok"], "intact": r["intact"], "permuted": r["permuted"], "eng": r["eng"]}
                                             for s, r in pinned.items()}
        q = _load(os.path.join(src, "battery", "stress_%s_quench.json" % side))["runs"]
        cl = {"co_latch_none": 0, "none_no_latch": 0, "stale_name_alone": 0, "correct_alone": 0}
        stale, correct = [], []
        for s, v in q.items():
            for c in CATS[1:]:
                r = v["closed"]["reads"][c]
                if r["cat"] == c:
                    cl["correct_alone"] += 1
                    correct.append([s, c])
                elif r["cat"] is None and len(r["latched"]) > 1:
                    cl["co_latch_none"] += 1
                elif r["cat"] is None:
                    cl["none_no_latch"] += 1
                else:
                    cl["stale_name_alone"] += 1
                    stale.append([s, c, r["cat"]])
        gains = sorted({(v["closed"]["quench"]["gain_min"], v["closed"]["quench"]["gain_max"], v["closed"]["quench"]["n_syn"],
                         v["open"]["quench"]["gain_min"]) for v in q.values()})
        out["quench_flag_" + side] = {
            "n_seeds": len(q), "open_all_four_named": sum(all(v["open"]["reads"][c]["cat"] == c for c in CATS) for v in q.values()),
            "closed_all_four_named": sum(all(v["closed"]["reads"][c]["cat"] == c for c in CATS) for v in q.values()),
            "closed_later_reads_n": 3 * len(q), "closed_later_reads": cl, "stale_reads": stale, "correct_alone_reads": correct,
            "lesion_engaged_closed_gain_min_max_nsyn_open_gain_min": [list(g) for g in gains],
            "bounded_readback_values": sorted({str(v[a]["reads"][c]["readback"]) for v in q.values() for a in ("open", "closed") for c in CATS})}
    return out


def _handler_block(src):
    H = os.path.join(src, "handler")
    from tools.answer_rate import rate
    shards = {}
    for side in ("on", "off"):
        for arm in ("intact_a", "permuted"):
            p = os.path.join(H, side, "s101", arm + ".json")
            shards[(side, arm)] = _load(p)
    out = {}
    for (side, arm), d in shards.items():
        R = d["responses"]
        feel = {}
        for lab in HANDLER_FEEL:
            ec = R[lab].get("emotion_category") or {}
            sl = ec.get("slot_counts") or {}
            feel[lab] = {"named": ec.get("category"), "latched": ec.get("latched"), "slot_max": max(sl.values()) if sl else None,
                         "co_commit": ec.get("co_commit"), "bounded_flag": ec.get("bounded_flag"), "bounded_env": ec.get("bounded_env"),
                         "bounded_readback": ec.get("bounded_readback"), "oor_count": ec.get("oor_count"),
                         "oor_count_session": ec.get("oor_count_session"), "organ_error": ec.get("error")}
        rt = rate(list(R.values()))
        out["%s_%s" % (side, arm)] = {
            "n_turns": len(R), "errored_turns": [l for l, r in R.items() if "_error" in r], "wall_s": d.get("wall_s"),
            "sources": d.get("sources"), "peak_rss_mb": (d.get("process") or {}).get("peak_rss_mb"),
            "feel": feel, "answered": rt["answered"], "answer_rate": rt["answer_rate"]}
    for arm in ("intact_a", "permuted"):
        Ro, Rf = shards[("on", arm)]["responses"], shards[("off", arm)]["responses"]
        nonfeel = [l for l in Rf if not (Rf[l] or {}).get("inner_state_readout")]

        def pre(a):
            a = str(a)
            for m in (" The emotion channel", " No single"):
                if m in a:
                    return a[:a.find(m)]
            return a
        out["paired_" + arm] = {
            "n_nonfeel_turns": len(nonfeel),
            "nonfeel_reply_text_identical": sum((Ro.get(l) or {}).get("answer") == (Rf.get(l) or {}).get("answer") for l in nonfeel),
            "nonfeel_whole_response_identical": sum(json.dumps(Ro.get(l), sort_keys=True) == json.dumps(Rf.get(l), sort_keys=True) for l in nonfeel),
            "feel_text_before_category_clause_identical": sum(pre((Ro.get(l) or {}).get("answer")) == pre((Rf.get(l) or {}).get("answer"))
                                                              for l in Rf if (Rf[l] or {}).get("inner_state_readout")),
            "n_feel_turns": sum(1 for l in Rf if (Rf[l] or {}).get("inner_state_readout")),
            "answer_rate_on": out["on_" + arm]["answer_rate"], "answer_rate_off": out["off_" + arm]["answer_rate"]}
    return out


def _attribution(res):
    """Whose is the difference between the arms? The flag is the only manipulation (same code, same seeds, flag taken from the
    environment), so the question is how much of each failure count is absent once it is on, and whether the lever moved the
    out-of-range census at all (a flag that left the census at 0 in both arms would have changed nothing)."""
    from tools.lab import attributable_to, lever
    b, st = res["battery"], res["stress"]

    def fails(blk):
        return blk["n_seeds"] - blk["n_pass_all_10_cases"]
    out = {}
    pairs = {"held_out_301_360": (b["flag_off"]["held_out_301_360"], b["flag_on"]["held_out_301_360"]),
             "dev_1_8": (b["flag_off"]["dev_1_8"], b["flag_on"]["dev_1_8"]),
             "fresh_401_420": (b["fresh_401_420"]["flag_off"], b["fresh_401_420"]["flag_on"])}
    for name, (off, on) in pairs.items():
        moved = lever("%s: seeds passing all 10 cases (flag off -> on)" % name, off["n_pass_all_10_cases"], on["n_pass_all_10_cases"],
                      required=False, continuous=(off["census_all_steps_incl_training"]["n_neurons_total"],
                                                  on["census_all_steps_incl_training"]["n_neurons_total"]))
        out[name] = {"lever_moved": bool(moved), "fails_off": fails(off), "fails_on": fails(on),
                     "attributable_to_flag": attributable_to("%s failing seeds" % name, fails(off), fails(on)),
                     "oor_neurons_off": off["census_all_steps_incl_training"]["n_neurons_total"],
                     "oor_neurons_on": on["census_all_steps_incl_training"]["n_neurons_total"]}
    for cv in ("39.6", "53.6"):
        off, on = st["inject_flag_off"]["by_C_pF"][cv], st["inject_flag_on"]["by_C_pF"][cv]
        out["inject_C_%s" % cv] = {"fails_off": off["n"] - off["ok"], "fails_on": on["n"] - on["ok"],
                                   "attributable_to_flag": attributable_to("injected C=%s failing runs" % cv, off["n"] - off["ok"], on["n"] - on["ok"])}
    for side in ("on", "off"):
        q = st["quench_flag_" + side]
        out["quench_%s" % side] = {"protocol_fails_closed": q["n_seeds"] - q["closed_all_four_named"], "protocol_fails_open": q["n_seeds"] - q["open_all_four_named"],
                                   "attributable_to_quench": attributable_to("flag %s: protocol failing seeds, quench closed vs open" % side,
                                                                             q["n_seeds"] - q["closed_all_four_named"], q["n_seeds"] - q["open_all_four_named"])}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="scratch dir holding battery/ and handler/ (the drivers' outputs)")
    ap.add_argument("--pinned", required=True, help="glob of the pinned probe14_base_w*.json records (Stage-2 pin code)")
    ap.add_argument("--drivers", default=os.path.join(_REPO, "docs", "drafts", "emo-bounded-cond"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"runner": "research/runners/emotion_category_bounded_aggregate.py", "analysis_only": True,
           "note": "aggregates driver outputs; runs no brain. Raw per-seed records stay in the scratch dir (sha256 below).",
           "battery": _battery_block(a.src, a.pinned), "stress": _stress_block(a.src), "handler_seed101_emo": _handler_block(a.src)}
    res["attribution"] = _attribution(res)
    ov = os.path.join(a.src, "overhead.json")
    if os.path.exists(ov):
        res["census_overhead"] = _load(ov)
    res["drivers"] = [{"file": os.path.relpath(p, _REPO), "sha256": _sha(p)} for p in sorted(glob.glob(os.path.join(a.drivers, "*"))) if os.path.isfile(p)]
    res["inputs"] = sorted({i["file"]: i for i in _INPUTS}.values(), key=lambda i: i["file"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=str)
    b = res["battery"]
    print("[emo-bounded aggregate] flag on held-out %d/%d, dev %d/%d; flag off held-out %d/%d; fresh on %d/%d off %d/%d -> %s" % (
        b["flag_on"]["held_out_301_360"]["n_pass_all_10_cases"], b["flag_on"]["held_out_301_360"]["n_seeds"],
        b["flag_on"]["dev_1_8"]["n_pass_all_10_cases"], b["flag_on"]["dev_1_8"]["n_seeds"],
        b["flag_off"]["held_out_301_360"]["n_pass_all_10_cases"], b["flag_off"]["held_out_301_360"]["n_seeds"],
        b["fresh_401_420"]["flag_on"]["n_pass_all_10_cases"], b["fresh_401_420"]["flag_on"]["n_seeds"],
        b["fresh_401_420"]["flag_off"]["n_pass_all_10_cases"], b["fresh_401_420"]["flag_off"]["n_seeds"], a.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
