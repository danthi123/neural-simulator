"""A10 CAPABILITY-GATE runner (midnight plan S15c, 2026-09-24; rewritten in the fix round after the adversarial review
of 58c400ff6). Scores the pre-registration EXACTLY as written, with the design changes of AMENDMENT-1.

PRE-REGISTERED: research/findings/2026-09-24-reward-value-spiking-afferent-PREREGISTRATION.md (57f0ebfd0) and
research/findings/2026-09-24-reward-value-spiking-afferent-PREREG-AMENDMENT-1.md (committed before this version's
first run).

TWO MODES.
  --mode arms   (builds brains: run it on the pool, never on the RAM-starved local box). Five fresh tiny-demo brains,
                each ONE `onebrain_regression_battery --worker` subprocess running the battery's own `confirm`
                ("the dog chase the cat", session surp) and `contra` ("the dog chase the fish", session surp2)
                turns through the REAL webapp.server.brain_chat handler:
                  off_a, off_b : BRAIN_REWARD_VALUE_AFFERENT=0            (off_b = the OFF null control)
                  on_a,  on_b  : BRAIN_REWARD_VALUE_AFFERENT=1            (on_b  = the ON null control)
                  les          : BRAIN_REWARD_VALUE_AFFERENT=1 + BRAIN_REWARD_VALUE_LESION=1
                then, in this process, the BLOCK-ASYMMETRY CONTROL on a standalone copy of the organ's lesioned twin
                (see _block_control). The pre-patch reference for (A) is NOT built here: it is the same battery
                worker, same env, run at the pre-patch main revision (the merge parent) -- `--print-pre-cmd` prints
                the exact command.
  --mode score  (no brain build; runs anywhere). Reads the arms + the pre-patch reference and writes the verdict.

WHAT IS SCORED (the pre-registration's criteria, unchanged by the amendment):
  (A) OFF identity: no `reward_value` key on either OFF turn, AND `da_drives` (incl. afferent_pA and mode) and the
      reply `answer` on both turns EXACTLY equal to the pre-patch main build with the same env. The whole response
      is also compared (timing keys dropped, listed in the artifact). Asserted from data, never from the code.
  (B) ON: `reward_value.source == "surprise"` on both turns; contra > confirm on `reward_value.normalized` AND on
      `da_drives.afferent_pA` (the amendment's replacement for the removed `pa`); and
      `attributable_to(normalized differential, live vs lesion) >= 0.9`.
  (C) LESION: |normalized(contra) - normalized(confirm)| < 1e-6 under the lesion.
  GO iff A and B and C. Any criterion measured and false is NO-GO. UNDEFINED only when a real precondition fails
  (an arm did not build, a turn was not measured, the lesion cut did not hold at read time, a null control is not
  clean, the override never reached the workspace).

REPORTED SEPARATELY (not part of the pre-registered GO; the midnight plan's S15 success check, defined before any
data): on each turn, does the intact reply (answer / da_drives.mode / da_drives.lead) change under the lesion, with a
clean ON null control and (A) holding.

SEED 7 ONLY (dev/calibration; never a gate seed). Composer: the production default (onebrain) unless --composer rf.

Run (pool):  SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m research.runners._reward_value_afferent_derisk \
                 --mode arms --seed 7 --out research/findings/raw/_reward_value_afferent_derisk/v2/s7_arms.json
Score:       .venv/bin/python -m research.runners._reward_value_afferent_derisk --mode score \
                 --arms research/findings/raw/_reward_value_afferent_derisk/v2/s7_arms.json \
                 --pre research/findings/raw/_reward_value_afferent_derisk/v2/s7_pre_off_main.json \
                 --out research/findings/raw/_reward_value_afferent_derisk/v2/s7_verdict.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

AFF = "BRAIN_REWARD_VALUE_AFFERENT"
LES = "BRAIN_REWARD_VALUE_LESION"
TURNS = ("confirm", "contra")

# Heavy Gate-B organs unrelated to this coupling are switched off IDENTICALLY in every arm (and in the pre-patch
# reference). BRAIN_SURPRISE and BRAIN_DA_DRIVES stay at their default-ON anchors. The default LTM tier is off
# (BRAIN_LTM_SHIP_DEFAULT's own no-LTM escape): this coupling needs only the built-in dog-chase-cat fact.
_QUIET = {
    "BRAIN_AFFECT": "0", "BRAIN_WORLDMODEL": "0", "BRAIN_METACOG": "0", "BRAIN_MULTIREF": "0",
    "BRAIN_NONCONTRADICTION_GATE": "0", "BRAIN_EPISODIC_STORE": "0", "BRAIN_CURIOSITY": "0", "BRAIN_RICH": "0",
    "BRAIN_GNW_BUS": "0", "BRAIN_CONTINUOUS": "0", "BRAIN_CONTINUOUS_DRIVES": "0", "BRAIN_SWAP_DRIVES": "0",
    "BRAIN_RECONSOLIDATION": "0", "BRAIN_LTM_SHIP_DEFAULT": "off",
}

ARMS = {
    "off_a": {AFF: "0", LES: "0"},
    "off_b": {AFF: "0", LES: "0"},
    "on_a": {AFF: "1", LES: "0"},
    "on_b": {AFF: "1", LES: "0"},
    "les": {AFF: "1", LES: "1"},
}

# Keys dropped before the whole-response comparison: wall-clock quantities that differ run to run by nature.
_TIMING_KEYS = {"gen_seconds", "elapsed", "elapsed_s", "elapsed_ms", "latency", "latency_ms", "latency_s",
                "wall_s", "wall_ms", "seconds", "t_ms", "timing", "timings", "timestamp", "ts", "time_s", "phase_ms",
                "phase_timing", "phase_timings", "duration_s", "duration_ms"}


def arm_env(arm, seed, composer=None):
    env = dict(_QUIET)
    env["BRAIN_DA_DRIVES"] = "1"
    env["BRAIN_CHAT_SEED"] = str(int(seed))
    if composer:
        env["BRAIN_COMPOSER_KIND"] = str(composer)
    env.update(ARMS[arm])
    return env


def pre_cmd(seed, composer, revision, out):
    """The exact pool command for the pre-patch reference: the SAME battery worker and turns, with the OFF arm's env
    passed as process env (the worker only copies its --env JSON into os.environ before importing the server, so the
    two are the same process state)."""
    env = arm_env("off_a", seed, composer)
    kv = " ".join("%s=%s" % (k, env[k]) for k in sorted(env))
    return ("cd ~/derisk-pool/revisions/%s && env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 %s "
            ".venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns %s "
            "--out %s" % (revision, kv, ",".join(TURNS), out))


# ── mode arms ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _run_arm(arm, seed, composer, out_path):
    env = arm_env(arm, seed, composer)
    t0 = time.time()
    p = subprocess.run([sys.executable, "-u", "-m", "research.runners.onebrain_regression_battery", "--worker",
                        "--env", json.dumps(env, sort_keys=True), "--turns", ",".join(TURNS), "--out", out_path],
                       env=dict(os.environ))
    rec = {"arm": arm, "env": env, "out": out_path, "returncode": p.returncode, "seconds": round(time.time() - t0, 1)}
    rec["responses"] = json.load(open(out_path)) if (p.returncode == 0 and os.path.exists(out_path)) else None
    return rec


def _block_control(seed, s_blk, t_blk):
    """BLOCK-ASYMMETRY CONTROL for the within-lesion residual. The lesioned twin has ONE route into the surprise pool
    (patient_asserted -> surprise, block-diagonal; patient_expected -> surprise is zeroed and there is no other
    pathway), so the cue has no path to surprise there. Hypothesis: the confirm-vs-contra residual under the lesion is
    the difference between surprise block s and surprise block t, not anything prediction-related. Test: on a fresh
    standalone copy of the twin (same seed, same build path as production's `_ensure_les`), (1) replicate the two
    lesioned reads exactly, (2) read each block with NO cue at all (the prediction phase replaced by 60 undriven
    steps). If the cue-free t-minus-s difference equals the residual, the residual is block asymmetry."""
    from research.runners.surprise_production_organ import SurpriseProductionOrgan
    from research.runners._spiking_expectation_rpe_derisk import _drive_read, _hard_reset
    from webapp.reward_value_afferent_chat import pe_surprise_edge_abs_sum
    org = SurpriseProductionOrgan(seed=int(seed), shared=None)
    les = org._ensure_les()
    b, idx, xp, meta = les["bridge"], les["idx_map"], les["xp"], les["meta"]
    b._blk = meta["blk"]

    def read(drives, pre):
        _hard_reset(b)
        return float(_drive_read(b, idx, drives, 60, xp, ["surprise"], pre_drives=pre, pre_steps=60)["surprise"])

    out = {"seed": int(seed), "stored_block": s_blk, "asserted_block": t_blk,
           "cut_at_read": pe_surprise_edge_abs_sum(b, idx)}
    out["replicate_confirm_hz"] = read({"cue": (s_blk, 600.0), "patient_asserted": (s_blk, 600.0)}, {"cue": (s_blk, 600.0)})
    out["replicate_contra_hz"] = read({"cue": (s_blk, 600.0), "patient_asserted": (t_blk, 600.0)}, {"cue": (s_blk, 600.0)})
    out["cuefree_s_hz"] = read({"patient_asserted": (s_blk, 600.0)}, {})
    out["cuefree_t_hz"] = read({"patient_asserted": (t_blk, 600.0)}, {})
    out["cuefree_t_minus_s_hz"] = out["cuefree_t_hz"] - out["cuefree_s_hz"]
    return out


def mode_arms(seed, composer, out_path):
    base = os.path.splitext(out_path)[0]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    arms = {}
    for arm in ARMS:
        arms[arm] = _run_arm(arm, seed, composer, "%s_%s.json" % (base, arm))
        print("[A10 arms] %s rc=%s %.0fs" % (arm, arms[arm]["returncode"], arms[arm]["seconds"]), flush=True)
    ctrl = None
    try:
        les_rv = (((arms["les"]["responses"] or {}).get("contra") or {}).get("da_drives") or {}).get("reward_value") or {}
        s_blk, t_blk = les_rv.get("stored_block"), les_rv.get("asserted_block")
        if s_blk is not None and t_blk is not None:
            os.environ.setdefault("SIM_BACKEND", "numpy")
            ctrl = _block_control(seed, int(s_blk), int(t_blk))
        else:
            ctrl = {"error": "lesion arm recorded no block indices"}
    except Exception as e:
        ctrl = {"error": "%s: %s" % (type(e).__name__, e)}
    out = {"runner": "_reward_value_afferent_derisk", "mode": "arms", "seed": int(seed),
           "seed_kind": "dev-calibration (NOT a 6-seed gate seed)",
           "composer_forced": composer, "turns": list(TURNS), "arms": arms, "block_control": ctrl}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({"out": out_path, "rc": {a: arms[a]["returncode"] for a in arms}}), flush=True)
    return 0


# ── mode score ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _drop_timing(x):
    if isinstance(x, dict):
        return {k: _drop_timing(v) for k, v in x.items() if k not in _TIMING_KEYS}
    if isinstance(x, list):
        return [_drop_timing(v) for v in x]
    return x


def _canon(x):
    return json.dumps(_drop_timing(x), sort_keys=True, default=str)


def _dd(resp, turn):
    return ((resp or {}).get(turn) or {}).get("da_drives") or {}


def _rv(resp, turn):
    return _dd(resp, turn).get("reward_value") or {}


def _reply(resp, turn):
    r = (resp or {}).get(turn) or {}
    dd = r.get("da_drives") or {}
    return {"answer": r.get("answer"), "mode": dd.get("mode"), "lead": dd.get("lead"),
            "afferent_pA": dd.get("afferent_pA"), "da_level": dd.get("da_level")}


def _built(resp):
    if not isinstance(resp, dict):
        return False
    return all(isinstance(resp.get(t), dict) and not resp[t].get("_error") for t in TURNS)


def mode_score(arms_path, pre_path, out_path):
    from tools.lab import attributable_to, lever
    from tools.verdict import Verdict
    A = json.load(open(arms_path))
    arms = {k: v.get("responses") for k, v in A["arms"].items()}
    pre = json.load(open(pre_path)) if (pre_path and os.path.exists(pre_path)) else None
    ctrl = A.get("block_control") or {}

    v = Verdict("A10 prediction-error salience afferent into the SNc (flags BRAIN_REWARD_VALUE_*), seed-7 de-risk")
    # ── real preconditions ──
    for arm in ARMS:
        v.require("arm %s built and ran both turns" % arm, _built(arms.get(arm)), expect=True)
    v.require("pre-patch reference (main, same env) built and ran both turns", _built(pre), expect=True)
    on_rv = {t: _rv(arms.get("on_a"), t) for t in TURNS}
    les_rv = {t: _rv(arms.get("les"), t) for t in TURNS}
    v.require("ON: the surprise read drove both turns (reward_value.drives)",
              all(on_rv[t].get("drives") is True for t in TURNS), expect=True)
    v.require("ON: the override reached the workspace (da_drives.turn_signal_source == spiking_surprise)",
              all(_dd(arms.get("on_a"), t).get("turn_signal_source") == "spiking_surprise" for t in TURNS), expect=True)
    v.require("LESION: the lesioned read drove both turns", all(les_rv[t].get("drives") is True and les_rv[t].get("lesioned") is True
                                                                  for t in TURNS), expect=True)
    cut = [((les_rv[t].get("lesion_cut") or {}).get("holds")) for t in TURNS]
    v.require("LESION: the cut holds at read time on both turns (twin patient_expected<->surprise weights == 0)",
              all(c is True for c in cut), expect=True)
    null_on = all(_canon(_reply(arms.get("on_a"), t)) == _canon(_reply(arms.get("on_b"), t))
                  and on_rv[t].get("normalized") == _rv(arms.get("on_b"), t).get("normalized") for t in TURNS)
    v.require("ON null control clean (on_a == on_b: answer, mode, lead, afferent_pA, normalized)", null_on, expect=True)
    null_off = all(_canon((arms.get("off_a") or {}).get(t)) == _canon((arms.get("off_b") or {}).get(t)) for t in TURNS)
    v.require("OFF null control clean (off_a == off_b, whole response minus timing keys)", null_off, expect=True)
    intact_sum = [((les_rv[t].get("lesion_cut") or {}).get("intact_reference") or {}).get("total") for t in TURNS]
    twin_sum = [((les_rv[t].get("lesion_cut") or {}).get("lesion_twin") or {}).get("total") for t in TURNS]
    if intact_sum[0] is not None and twin_sum[0] is not None:
        v.reaches("lesion zeroes the prediction edges (intact organ sum -> twin sum)", intact_sum[0], twin_sum[0])
        lever("A10 lesion: patient_expected->surprise |W| sum", intact_sum[0], twin_sum[0], required=False)

    # ── (A) ──
    no_key = all("reward_value" not in _dd(arms.get("off_a"), t) for t in TURNS)
    a_dd_equal = all(_canon(_dd(arms.get("off_a"), t)) == _canon(_dd(pre, t)) for t in TURNS)
    a_answer_equal = all(((arms.get("off_a") or {}).get(t) or {}).get("answer") == ((pre or {}).get(t) or {}).get("answer")
                         for t in TURNS)
    a_whole_equal = all(_canon((arms.get("off_a") or {}).get(t)) == _canon((pre or {}).get(t)) for t in TURNS)
    go_a = bool(no_key and a_dd_equal and a_answer_equal)

    # ── (B) ──
    n_on = {t: on_rv[t].get("normalized") for t in TURNS}
    n_les = {t: les_rv[t].get("normalized") for t in TURNS}
    pa_on = {t: _dd(arms.get("on_a"), t).get("afferent_pA") for t in TURNS}
    src_ok = all(on_rv[t].get("source") == "surprise" for t in TURNS)
    diff_live = (n_on["contra"] - n_on["confirm"]) if None not in n_on.values() else None
    diff_les = (n_les["contra"] - n_les["confirm"]) if None not in n_les.values() else None
    attr = attributable_to("A10 normalized differential owed to the live surprise read (control = lesion)",
                           diff_live, diff_les) if (diff_live is not None and diff_les is not None) else None
    b_norm = diff_live is not None and diff_live > 0.0
    b_pa = None not in pa_on.values() and pa_on["contra"] > pa_on["confirm"]
    b_attr = attr is not None and attr >= 0.9
    go_b = bool(src_ok and b_norm and b_pa and b_attr)

    # ── (C) ──
    go_c = bool(diff_les is not None and abs(diff_les) < 1e-6)
    go = bool(go_a and go_b and go_c)
    v.disabled("heavy Gate-B organs (affect/worldmodel/metacog/multiref/..., LTM tier)",
               why="off identically in every arm and in the pre-patch reference, for speed and isolation")
    decided = v.decide(go=go, verbose=True)

    # ── S15 success check (reported, not part of the pre-registered GO) ──
    s15 = {}
    for t in TURNS:
        ri, rl, rb = _reply(arms.get("on_a"), t), _reply(arms.get("les"), t), _reply(arms.get("on_b"), t)
        s15[t] = {"intact": ri, "lesion": rl, "intact_rebuild": rb,
                  "reply_changes_under_lesion": bool(ri["answer"] != rl["answer"] or ri["mode"] != rl["mode"]
                                                      or ri["lead"] != rl["lead"]),
                  "answer_text_changes": bool(ri["answer"] != rl["answer"])}
    s15["clean_null"] = null_on
    s15["byte_identical_off"] = bool(go_a and a_whole_equal)

    # ── block-asymmetry control ──
    bc = dict(ctrl)
    hz_les = {t: les_rv[t].get("surprise_hz") for t in TURNS}
    if "cuefree_t_minus_s_hz" in ctrl and None not in hz_les.values():
        residual = hz_les["contra"] - hz_les["confirm"]
        bc["lesion_residual_hz"] = residual
        bc["replication_exact"] = bool(ctrl["replicate_confirm_hz"] == hz_les["confirm"]
                                       and ctrl["replicate_contra_hz"] == hz_les["contra"])
        bc["cuefree_fraction_of_residual"] = (ctrl["cuefree_t_minus_s_hz"] / residual) if residual else None

    out = {
        "runner": "_reward_value_afferent_derisk", "mode": "score", "seed": A.get("seed"),
        "seed_kind": A.get("seed_kind"), "composer_forced": A.get("composer_forced"),
        "composer_class": {t: on_rv[t].get("composer") for t in TURNS},
        "inputs": {"arms": arms_path, "pre": pre_path},
        "go": bool(decided["go"]), "status": decided["status"],
        "A_off_identity_vs_prepatch": {"no_reward_value_key": no_key, "da_drives_equal": a_dd_equal,
                                       "answer_equal": a_answer_equal, "whole_response_equal_minus_timing": a_whole_equal,
                                       "timing_keys_dropped": sorted(_TIMING_KEYS), "GO": go_a,
                                       "off": {t: _reply(arms.get("off_a"), t) for t in TURNS},
                                       "pre": {t: _reply(pre, t) for t in TURNS}},
        "B_on": {"source_ok": src_ok, "normalized": n_on, "afferent_pA": pa_on,
                 "normalized_contra_gt_confirm": b_norm, "afferent_pA_contra_gt_confirm": b_pa,
                 "attribution_vs_lesion": attr, "attribution_ge_0_9": b_attr, "GO": go_b,
                 "reward_value": on_rv, "replies": {t: _reply(arms.get("on_a"), t) for t in TURNS}},
        "C_lesion": {"normalized": n_les, "differential": diff_les, "lt_1e-6": go_c, "GO": go_c,
                     "reward_value": les_rv, "replies": {t: _reply(arms.get("les"), t) for t in TURNS}},
        "diff_live": diff_live, "diff_lesion": diff_les,
        "S15_success_check": s15,
        "block_asymmetry_control": bc,
        "preconditions": decided["preconditions"],
        "verdict": decided,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({"status": decided["status"], "A": go_a, "B": go_b, "C": go_c, "attr": attr, "out": out_path}))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("arms", "score"), default="arms")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--composer", choices=("onebrain", "rf"), default=None,
                    help="force BRAIN_COMPOSER_KIND (default: unset -> the production default)")
    ap.add_argument("--arms", default=None, help="score mode: the arms artifact")
    ap.add_argument("--pre", default=None, help="score mode: the pre-patch reference (battery worker at main)")
    ap.add_argument("--print-pre-cmd", default=None, metavar="REVISION",
                    help="print the pool command for the pre-patch reference at REVISION and exit")
    ap.add_argument("--pre-out", default="research/findings/raw/_reward_value_afferent_derisk/v2/s7_pre_off_main.json")
    ap.add_argument("--out", default="research/findings/raw/_reward_value_afferent_derisk/v2/s7_arms.json")
    a = ap.parse_args()
    if a.print_pre_cmd:
        print(pre_cmd(a.seed, a.composer, a.print_pre_cmd, a.pre_out))
        sys.exit(0)
    if a.mode == "arms":
        os.environ.setdefault("SIM_BACKEND", "numpy")
        sys.exit(mode_arms(a.seed, a.composer, a.out))
    sys.exit(mode_score(a.arms, a.pre, a.out))
