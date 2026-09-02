"""FLIP-VERIFY for one-brain cross-edge C1 (D2 surprise -> E2 world-model error-gated online update) — proves the
wiring in `research/runners/onebrain_xedge_surprise_worldmodel_production.py` is FLIP-READY (default-ON worthy)
WITHOUT actually flipping the production default: the edge stays wired default-OFF; a later owner-gated flip is a
separate, later step.

THREE ARMS, through the REAL `webapp.server.brain_chat` handler on fresh per-config subprocess builds (mirrors
`onebrain_flip_verify_harness.py` / `_xedge_flip_production_verify.py`'s own shape):

  ARM A  BYTE-IDENTICAL-OFF — with the flag explicitly ="0" (never popped — the 2026-08-27 off-arm-staleness class,
         `tools/gates/flip_offarm_staleness.py`), the visible answer + worldmodel DECISION fields (kind,
         predicted_next_sign) reproduce env-unset (today's main, since C1 defaults OFF). Also asserts the new
         diagnostic key (`worldmodel.surprise_worldmodel_crossedge`) is ABSENT in both — proves merely importing the
         new module (even off) changes nothing observable.

  ARM B  VISIBLE-ON-REAL-TRAFFIC + LESION-ATTRIBUTABLE — C1's real-traffic signal is a DIAGNOSTIC FIELD
         (`resp["worldmodel"]["surprise_worldmodel_crossedge"]`), not a discourse-role decision, so this is NOT the
         reusable harness's repair-role/wm-resolved ARM B shape (that shape does not fit; ported the harness's
         ARM A/C machinery, wrote a diagnostic-shaped ARM B of its own — see the module docstring). Three real
         multi-turn conversations through the SAME live handler, one flag config each:
           - "on":       establish a POSITIVE expectation, then an ALTERNATING pos/neg sequence (every turn after
                         the first VIOLATES the just-updated persistence expectation — see the module's own analysis
                         of why alternation, not repetition, keeps the violation branch firing turn after turn).
                         Some turns must show `gate_opened=True` and `w_obs_after > w_obs_before`.
           - "on_expected": establish a POSITIVE expectation, then REPEAT the SAME positive text (never violates —
                         the persistence prior keeps confirming). The SAME gating code must show `gate_opened=False`
                         on every turn (n_gated==0) — selectivity, not a weaker bar.
           - "on_lesion": the SAME alternating sequence as "on", but BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL_LESION=1
                         (zeroes obs->surprise on the shared circuit BEFORE the sequence starts). Every gate stays
                         closed (n_gated==0) — the update VANISHES.
         `n_hollow` (the #94-class anti-hollow bar): 0 iff "on" shows >=1 visible gated turn AND "on_lesion" shows 0.

  ARM C  NO-REGRESSION — `onebrain_regression_battery.run_regression_battery(flag=FLAG, on_value="1")`: flipping
         the flag ON-vs-OFF must not change ANY of the ~38 registered default-ON faculties' DECIDED output.

De-risk: `research/runners/_crossedge_surprise_worldmodel_derisk.py`, 6/6-seed GO,
`research/findings/raw/_crossedge_surprise_worldmodel_6seed.json`. Production wiring:
`research/runners/onebrain_xedge_surprise_worldmodel_production.py` (default-OFF; this runner does NOT flip it —
FLIP_VERIFY_GO records READINESS for a later, separate, owner-gated flip).

Run (numpy, fast; RSS-light — watch `free -m`):
  SIM_BACKEND=numpy python -m research.runners._crossedge_surprise_worldmodel_flip_verify \
      --out research/findings/raw/_crossedge_surprise_worldmodel_flip_verify/numpy_smoke.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

FLAG = "BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL"
LESION_FLAG = "BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL_LESION"
RESET_KEYS = (FLAG, LESION_FLAG)

# Warriner-lexicon-reliable valence text (empirically checked 2026-09-02: appraise_text valence +0.81 / -0.73).
POS_TEXT = "that was wonderful, I am so happy"
NEG_TEXT = "that was terrible, I am so sad"

WELL_ITEMS = ["the wolf bites the apple", "the dog chases the cat"]
QUESTION = "what does the wolf bite"
N_TURNS = 8          # sequence length AFTER the establishing turn (ARM B)
SEED = 42

CONFIGS = {
    "A_baseline": {},
    "A_off":      {FLAG: "0"},
    "B_on":          {FLAG: "1"},
    "B_on_expected": {FLAG: "1"},
    "B_lesion":      {FLAG: "1", LESION_FLAG: "1"},
}


def _extract(d: dict) -> dict:
    wm = d.get("worldmodel") or {}
    return {"answer": d.get("answer"), "abstained": bool(d.get("abstained")),
            "worldmodel_kind": wm.get("kind"), "predicted_next_sign": wm.get("predicted_next_sign"),
            "surprised": wm.get("surprised"), "crossedge": wm.get("surprise_worldmodel_crossedge")}


def _decisions_equal(a: dict, b: dict) -> bool:
    """The DECISION surface (categorical only — never a continuous margin/rate, matching the regression battery's
    own instrument choice): the answer text, abstain, and the worldmodel kind/predicted-sign."""
    return (a.get("answer") == b.get("answer") and a.get("abstained") == b.get("abstained")
            and a.get("worldmodel_kind") == b.get("worldmodel_kind")
            and a.get("predicted_next_sign") == b.get("predicted_next_sign"))


def _turn(msg: str, session: str, reset: bool = False) -> dict:
    from webapp.server import brain_chat, BrainChatRequest
    try:
        r = brain_chat(BrainChatRequest(session=session, message=msg, brain="tiny-demo", renderer="stub",
                                        rich=False, reset=reset))
        return _extract(json.loads(r.body))
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}", "answer": None}


def run_worker(config: str, out_path: str) -> int:
    os.environ.setdefault("SIM_BACKEND", "numpy")
    for k in RESET_KEYS:                     # OFF-ARM DISCIPLINE: clear, then set EXACTLY this config's env
        os.environ.pop(k, None)
    for k, v in CONFIGS[config].items():
        os.environ[k] = v

    res: dict = {"config": config, "env": dict(CONFIGS[config])}

    if config in ("A_baseline", "A_off"):
        res["well"] = {m: _turn(m, "well_%d" % i, reset=True) for i, m in enumerate(WELL_ITEMS)}
        res["question"] = _turn(QUESTION, "q0", reset=True)

    if config in ("B_on", "B_on_expected", "B_lesion"):
        session = config
        _turn(POS_TEXT, session, reset=True)                  # establish a POSITIVE expectation
        turns = []
        for t in range(N_TURNS):
            if config == "B_on_expected":
                text = POS_TEXT                                # always confirms -> never violates
            else:
                text = NEG_TEXT if (t % 2 == 0) else POS_TEXT   # alternates -> violates the just-set expectation
            turns.append(_turn(text, session, reset=False))
        res["sequence_turns"] = turns

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print("[worker %s] -> %s" % (config, out_path), flush=True)
    return 0


# ── orchestrator ─────────────────────────────────────────────────────────────────────────────────────────────
def _free_mb() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 0


def _spawn(config: str, raw_dir: str, mem_floor: int) -> dict:
    fm = _free_mb()
    if fm and fm < mem_floor:
        return {"_abort": "free mem %dMB < %dMB floor before %s" % (fm, mem_floor, config)}
    cfg_out = os.path.join(raw_dir, "w_%s.json" % config)
    env = dict(os.environ)
    print("[orch] %s free=%dMB -> spawning worker" % (config, fm), flush=True)
    p = subprocess.run([sys.executable, "-u", "-m", "research.runners._crossedge_surprise_worldmodel_flip_verify",
                        "--worker", config, "--out", cfg_out], env=env)
    if p.returncode != 0 or not os.path.exists(cfg_out):
        return {"_failed": "%s rc=%d" % (config, p.returncode)}
    with open(cfg_out) as f:
        return json.load(f)


def aggregate(per: dict, battery: dict | None) -> dict:
    from tools.lab import attributable_to

    # ── ARM A: byte-identical-off ──
    base, off = per.get("A_baseline"), per.get("A_off")
    a_items, a_diffs = [], []
    if base and off:
        for m in WELL_ITEMS:
            eq = _decisions_equal(off["well"][m], base["well"][m])
            a_items.append(eq)
            if not eq:
                a_diffs.append({"item": m, "off": off["well"][m], "baseline": base["well"][m]})
        eqq = _decisions_equal(off["question"], base["question"])
        a_items.append(eqq)
        if not eqq:
            a_diffs.append({"item": QUESTION, "off": off["question"], "baseline": base["question"]})
        no_key = all(off["well"][m].get("crossedge") is None for m in WELL_ITEMS) and \
                 all(base["well"][m].get("crossedge") is None for m in WELL_ITEMS)
        a_items.append(no_key)
        if not no_key:
            a_diffs.append({"item": "crossedge_key_absent", "off": [off["well"][m].get("crossedge") for m in WELL_ITEMS],
                            "baseline": [base["well"][m].get("crossedge") for m in WELL_ITEMS]})
    a_pass = bool(a_items) and all(a_items)

    # ── ARM B: visible-on-real-traffic + lesion-attributable, diagnostic-field-shaped ──
    on, on_exp, les = per.get("B_on"), per.get("B_on_expected"), per.get("B_lesion")
    b_detail = {}
    n_gated_on = n_gated_exp = n_gated_les = 0
    growth_on = growth_exp = growth_les = 0.0
    if on:
        for t in on.get("sequence_turns", []):
            ce = t.get("crossedge") or {}
            if ce.get("gate_opened"):
                n_gated_on += 1
                growth_on += float(ce.get("w_obs_after", 0.0)) - float(ce.get("w_obs_before", 0.0))
    if on_exp:
        for t in on_exp.get("sequence_turns", []):
            ce = t.get("crossedge")
            if ce and ce.get("gate_opened"):
                n_gated_exp += 1
                growth_exp += float(ce.get("w_obs_after", 0.0)) - float(ce.get("w_obs_before", 0.0))
    if les:
        for t in les.get("sequence_turns", []):
            ce = t.get("crossedge") or {}
            if ce.get("gate_opened"):
                n_gated_les += 1
                growth_les += float(ce.get("w_obs_after", 0.0)) - float(ce.get("w_obs_before", 0.0))

    visible = bool(n_gated_on >= 1 and growth_on > 0.02)
    expected_selective = bool(n_gated_exp == 0)
    lesion_vanishes = bool(n_gated_les == 0 and abs(growth_les) < 1e-9)
    n_hollow = 0 if (visible and lesion_vanishes) else 1

    frac_vs_lesion = attributable_to("crossedge_surprise_worldmodel real-traffic weight growth vs lesion",
                                     growth_on, growth_les)
    frac_vs_expected = attributable_to("crossedge_surprise_worldmodel real-traffic weight growth vs expected "
                                       "(same gating code)", growth_on, growth_exp)
    b_detail = {"n_gated_on": n_gated_on, "growth_on": growth_on, "n_gated_expected": n_gated_exp,
               "growth_expected": growth_exp, "n_gated_lesion": n_gated_les, "growth_lesion": growth_les,
               "frac_attributable_vs_lesion": (None if frac_vs_lesion is None else float(frac_vs_lesion)),
               "frac_attributable_vs_expected": (None if frac_vs_expected is None else float(frac_vs_expected))}
    b_pass = bool(visible and expected_selective and lesion_vanishes
                  and frac_vs_lesion is not None and frac_vs_lesion >= 0.8
                  and frac_vs_expected is not None and frac_vs_expected >= 0.8)

    battery_pass = None
    if battery is not None:
        battery_pass = bool(battery.get("all_pass"))

    go = bool(a_pass and b_pass and (battery_pass if battery is not None else True))
    out = {
        "arm_A_byte_identical_off": {"pass": a_pass, "n_match": sum(a_items), "n_total": len(a_items), "diffs": a_diffs},
        "arm_B_visible_on_real_traffic": dict(b_detail, pass_=b_pass, visible=visible,
                                              expected_selective=expected_selective, lesion_vanishes=lesion_vanishes,
                                              n_hollow=n_hollow),
        "FLIP_VERIFY_GO": go,
    }
    out["arm_B_visible_on_real_traffic"]["pass"] = out["arm_B_visible_on_real_traffic"].pop("pass_")
    if battery is not None:
        out["arm_C_regression_battery"] = {"pass": battery_pass, "n_faculties": battery.get("n_faculties"),
                                           "n_regressed": battery.get("n_regressed"),
                                           "regressed": battery.get("regressed")}
    return out


def _orchestrate(out_path: str, mem_floor: int, run_battery: bool) -> int:
    raw_dir = os.path.dirname(out_path)
    os.makedirs(raw_dir, exist_ok=True)
    per, problems = {}, []
    for cfg in ("A_baseline", "A_off", "B_on", "B_on_expected", "B_lesion"):
        rec = _spawn(cfg, raw_dir, mem_floor)
        if "_abort" in rec:
            print("[orch] ABORT: %s" % rec["_abort"], flush=True)
            return 2
        if "_failed" in rec:
            problems.append(rec["_failed"])
            print("[orch] WORKER FAILED: %s" % rec["_failed"], flush=True)
            continue
        per[cfg] = rec

    battery_result = None
    if run_battery:
        try:
            from research.runners.onebrain_regression_battery import run_regression_battery
            battery_result = run_regression_battery(flag=FLAG, out_dir=raw_dir)
        except Exception as e:
            battery_result = {"all_pass": False, "error": "%s: %s" % (type(e).__name__, e)}

    agg = aggregate(per, battery_result)

    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("crossedge_surprise_worldmodel_flip_verify")
        b = agg["arm_B_visible_on_real_traffic"]
        Vd.require("arm_A_byte_identical_off", agg["arm_A_byte_identical_off"]["pass"], expect=True,
                   note="explicit flag=0 reproduces env-unset on answer/worldmodel-decision fields; the new "
                        "diagnostic key is absent in both")
        Vd.require("arm_B_visible_load_bearing", b["visible"], expect=True,
                   note="a real alternating-valence sequence through brain_chat opens the gate >=1 time and grows "
                        "the observed-pool transition")
        Vd.require("arm_B_expected_arm_selective", b["expected_selective"], expect=True,
                   note="the SAME gating code, on a confirming (never-violating) real sequence, opens the gate zero "
                        "times")
        Vd.require("arm_B_lesion_vanishes", b["lesion_vanishes"], expect=True,
                   note="BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL_LESION=1 on the SAME violating sequence opens the "
                        "gate zero times and produces zero weight growth")
        Vd.require("arm_B_n_hollow_zero", b["n_hollow"] == 0, expect=True,
                   note="the #94-class anti-hollow bar: visible AND lesion-vanishes together, not either alone")
        if battery_result is not None:
            Vd.require("arm_C_regression_battery_all_pass", bool(battery_result.get("all_pass")), expect=True,
                       note="flipping the flag ON-vs-OFF must not change any other default-ON faculty's decision")
        dec = Vd.decide(agg["FLIP_VERIFY_GO"], verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _ve:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_ve)}]

    payload = {
        "probe": "crossedge_surprise_worldmodel_flip_verify",
        "backend": os.environ.get("SIM_BACKEND", "numpy"), "brain": "tiny-demo", "renderer": "stub",
        "seed": SEED, "flip_target": "%s (stays default-OFF; this runner does NOT flip it)" % FLAG,
        "worker_problems": problems, "preconditions": preconditions, "aggregate": agg, "per_worker": per,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print("\n===== FLIP-VERIFY VERDICT (crossedge_surprise_worldmodel, real /api/brain-chat handler) =====", flush=True)
    for k in ("arm_A_byte_identical_off", "arm_B_visible_on_real_traffic"):
        print("  [%s] %s" % ("PASS" if agg[k]["pass"] else "FAIL", k), flush=True)
    if "arm_C_regression_battery" in agg:
        b = agg["arm_C_regression_battery"]
        print("  [%s] arm_C_regression_battery: %s/%s faculties pass"
              % ("PASS" if b["pass"] else "FAIL",
                 (b.get("n_faculties") or 0) - (b.get("n_regressed") or 0), b.get("n_faculties")), flush=True)
    if problems:
        print("  worker_problems: %s" % problems, flush=True)
    print("\n  FLIP_VERIFY_GO = %s   wrote %s" % (agg["FLIP_VERIFY_GO"], out_path), flush=True)
    return 0 if agg["FLIP_VERIFY_GO"] else 1


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None, help="internal: run one config's worker")
    ap.add_argument("--out", default="research/findings/raw/_crossedge_surprise_worldmodel_flip_verify/verify.json")
    ap.add_argument("--mem-floor", type=int, default=3000)
    ap.add_argument("--no-battery", action="store_true", help="skip the ARM C cross-faculty regression battery")
    args = ap.parse_args()
    if args.worker:
        return run_worker(args.worker, args.out)
    return _orchestrate(args.out, args.mem_floor, run_battery=(not args.no_battery))


if __name__ == "__main__":
    raise SystemExit(main())
