"""VERIFY (production wire): the RF-phasor RECALL COMPOSER joins production POOL #1 (surprise + world-model) on
ONE shared spiking bridge, BYTE-IDENTICAL through the REAL brain-chat handler path, the no-confab MOAT preserved,
with NO regression to surprise + world-model.

THE RUNG. `onebrain_merge_production.composer_merge_enabled()` (`BRAIN_COMPOSER_MERGE`, DEFAULT-OFF) makes the
production RF-phasor recall composer (`brain_conversational_agent` composer_kind='rf' -> `make_pool1_composer`)
run its resonate ops on pool #1's SHARED bridge (its masked composer slice), ONE `cp_membrane_potential_v` with
the D2 surprise organ + the E2 world-model organ. This runner exercises the REAL server brain-construction path
(`_build_tiny_demo(composer_kind='rf')`, the same brain `/api/brain-chat` builds when `BRAIN_COMPOSER_KIND=rf`)
and the REAL production reads: the recall (`what_does` -> `composer.query_patient`) + the moat, the surprise
organ (`SurpriseProductionOrgan.judge`), and the world-model organ (`WorldModelProductionOrgan.expectation` /
`read_surprise`).

VERDICT (GO): flag-ON vs flag-OFF is byte-identical across a broad panel --
  * COMPOSER recall byte-identical (stored who/what answers) + CORRECT + the moat abstains on the unstored cue.
  * SURPRISE reads byte-identical (per-case surprise_hz max delta 0.0, `surprised` bool identical) + faculty
    ALIVE (contradict >> confirm).
  * WORLD-MODEL reads byte-identical (pred_sign + pool rates + surprise_hz max delta 0.0) + faculty ALIVE.
  * GENUINELY ONE POOL when ON: the composer's `_pool1.bridge` IS the surprise organ's `_shared.bridge` (one
    object), N == surprise + world-model + composer + cleanup, one `cp_membrane_potential_v`.
Because the flag is read at first-build of a process-global singleton, ON and OFF are run in SEPARATE
subprocesses (`--emit`) and diffed by the `--compare` driver.

NO `sim/` edit. CPU-friendly (numpy). Run:
    SIM_BACKEND=numpy python -m research.runners._onebrain_composer_pool1_production_verify --compare \
        --out research/findings/raw/_onebrain_composer_pool1_production_verify.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  One process: build the production 'rf' brain + drive the real recall / surprise / world-model reads.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def run_panel(seed: int = 42) -> dict:
    # The composer KIND under test: `BRAIN_COMPOSER_KIND` (the driver sets it). 'rf' = the RF-phasor composer (the
    # original opt-in wire verify); 'onebrain' = the SHIPPED production-default composer (the DEFAULT-FLIP verify --
    # the b-closer routes it through pool #1 via make_pool1_onebrain_composer). Default 'onebrain' (the production
    # default) if the driver did not set it.
    kind = os.environ.get("BRAIN_COMPOSER_KIND", "onebrain")
    os.environ["BRAIN_COMPOSER_KIND"] = kind
    import research.runners.onebrain_merge_production as MP
    from research.runners.brain_chat_tui import _build_tiny_demo

    merge_on = MP.composer_merge_enabled()
    agent, _aliases, _n = _build_tiny_demo(seed, use_multiturn=True, enable_neural_render=False,
                                           composer_kind=kind)
    inner = getattr(agent, "agent", agent)
    composer = inner.composer

    # --- (1) RECALL panel (who/what) through the production recall path (`what_does` -> query_patient) ---
    # tiny-demo facts: brain->use->spikes, brain->learn->words, brain->store->memory, dog->chase->cat, cat->eat->fish
    cues = [("dog", "chase"), ("cat", "eat"), ("brain", "use"), ("brain", "learn"),
            ("brain", "store"), ("owl", "eat"), ("lion", "roar")]
    recall = {}
    for a, v in cues:
        ans = inner.what_does(a, v)
        recall[f"{a}|{v}"] = ans if isinstance(ans, str) else None
    # the no-confab MOAT: an unstored cue must abstain (None / "unknown"-like)
    moat_abstain = inner.what_does("lion", "roar")
    moat_ok = moat_abstain in (None, "unknown", "", "i don't know")
    # a direct query_patient on the composer (the exact organ the surprise/noncontradiction gates read)
    qp = {f"{a}|{v}": composer.query_patient(a, v) for a, v in cues}

    # --- (2) SURPRISE organ panel (the production D2 organ) ---
    import research.runners.surprise_production_organ as SP
    surprise = {}
    surprise_meta = {"enabled": bool(SP.surprise_enabled())}
    if SP.surprise_enabled():
        so = SP.get_organ(seed)
        so.ensure_built()
        cases = [
            ("dog", "chase", "cat", "cat"),      # confirm
            ("dog", "chase", "cat", "fish"),     # contradict
            ("brain", "use", "spikes", "words"),  # contradict
            ("brain", "learn", "words", "words"),  # confirm
        ]
        for a, v, ps, pa in cases:
            j = so.judge(a, v, ps, pa)
            surprise[f"{a}|{v}|{ps}|{pa}"] = {
                "hz": float(j["surprise_hz"]), "surprised": bool(j["surprised"]),
                "th": float(j["threshold"]),
            }
        surprise_meta.update({"threshold": float(so.threshold), "calib": so.calib})

    # --- (3) WORLD-MODEL organ panel (the production E2 organ) ---
    import research.runners.worldmodel_production_organ as WP
    worldmodel = {}
    worldmodel_meta = {"enabled": bool(WP.worldmodel_enabled())}
    if WP.worldmodel_enabled():
        wo = WP.get_organ(seed)
        wo.ensure_built()
        for sign in (+1, -1):
            e = wo.expectation(sign)
            worldmodel[f"exp{sign:+d}"] = {
                "pred_sign": int(e["pred_sign"]), "pp": float(e["pred_pos_rate"]),
                "pn": float(e["pred_neg_rate"]), "margin": float(e["pred_margin"]),
            }
            for obs in (+1, -1):
                r = wo.read_surprise(sign, obs)
                worldmodel[f"surp{sign:+d}|{obs:+d}"] = {
                    "hz": float(r["surprise_hz"]), "surprised": bool(r["surprised"]),
                }
        worldmodel_meta.update({"threshold": float(wo.threshold)})

    # --- (4) GENUINELY ONE POOL (only meaningful when the composer joined) ---
    one_pool = None
    if merge_on:
        sub = MP.get_merged_substrate(seed)
        sub.ensure_built()
        comp_pool = getattr(composer, "_pool1", None)
        so = SP.get_organ(seed) if SP.surprise_enabled() else None
        wo = WP.get_organ(seed) if WP.worldmodel_enabled() else None
        N = int(sub.bridge.core_config.num_neurons)
        vlen = int(sub.bridge.cp_membrane_potential_v.shape[0])
        composer_on = bool(comp_pool is sub.bridge or comp_pool is sub)
        # composer bound to THIS substrate's bridge:
        composer_bridge_is_pool = bool(comp_pool is not None and comp_pool.bridge is sub.bridge)
        surprise_bridge_is_pool = bool(so is not None and getattr(so, "_shared", None) is sub
                                       and so.bridge is sub.bridge)
        wm_bridge_is_pool = bool(wo is not None and getattr(wo, "_shared", None) is sub
                                 and wo._st is not None and wo._st["bridge"] is sub.bridge)
        one_pool = {
            "regions": list(sub.organs),
            "N": N, "v_len": vlen,
            "composer_on_pool1": composer_bridge_is_pool,
            "surprise_on_pool1": surprise_bridge_is_pool,
            "worldmodel_on_pool1": wm_bridge_is_pool,
            "one_v_array": bool(N == vlen),
            "genuinely_one_pool": bool(composer_bridge_is_pool and surprise_bridge_is_pool
                                       and wm_bridge_is_pool and N == vlen),
        }

    return {
        "merge_on": bool(merge_on),
        "recall": recall, "query_patient": qp,
        "moat_abstain": moat_abstain if isinstance(moat_abstain, str) else None,
        "moat_ok": bool(moat_ok),
        "surprise": surprise, "surprise_meta": surprise_meta,
        "worldmodel": worldmodel, "worldmodel_meta": worldmodel_meta,
        "one_pool": one_pool,
    }


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  The compare driver: run the panel in two SEPARATE subprocesses (flag OFF / ON) + diff.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _emit_subprocess(merge_val, seed: int, kind: str = "rf") -> dict:
    """Run ONE panel in a fresh subprocess (the merge flag is read at first-build of a process-global singleton, so
    ON/OFF/DEFAULT must be separate processes). `merge_val`: "0"/"1" sets `BRAIN_COMPOSER_MERGE`; None leaves it UNSET
    so the process reads the module DEFAULT (`_COMPOSER_IN_POOL1_DEFAULT_ON`) -- the DEFAULT-no-env leg of the flip
    verify. `kind` selects the composer under test ('rf' or 'onebrain')."""
    env = dict(os.environ)
    if merge_val is None:
        env.pop("BRAIN_COMPOSER_MERGE", None)     # DEFAULT-no-env: read _COMPOSER_IN_POOL1_DEFAULT_ON
    else:
        env["BRAIN_COMPOSER_MERGE"] = str(merge_val)
    env["BRAIN_COMPOSER_KIND"] = kind
    env.setdefault("SIM_BACKEND", "numpy")
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        outp = tf.name
    cmd = [sys.executable, "-m", "research.runners._onebrain_composer_pool1_production_verify",
           "--emit", "--seed", str(seed), "--out", outp]
    subprocess.run(cmd, env=env, check=True, cwd=os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    with open(outp) as f:
        return json.load(f)


def _dict_maxerr(a: dict, b: dict, keys) -> float:
    e = 0.0
    for k in a:
        for f in keys:
            if f in a[k] and f in b.get(k, {}):
                e = max(e, abs(float(a[k][f]) - float(b[k][f])))
    return e


def compare(seed: int = 42, kind: str = "rf", default_flip: bool = False) -> dict:
    """Diff two panels run in separate subprocesses.

    default_flip=False (the original opt-in wire verify): OFF (`BRAIN_COMPOSER_MERGE=0`, private) vs ON
    (`BRAIN_COMPOSER_MERGE=1`, pool). default_flip=True (the DEFAULT-FLIP verify): ESCAPE (`BRAIN_COMPOSER_MERGE=0`,
    the byte-identical revert) vs DEFAULT (NO env -> reads `_COMPOSER_IN_POOL1_DEFAULT_ON`; when the flag is ON this is
    the pool path). In both cases `off` = the private/escape baseline, `on` = the pool path -- so the diff logic below
    is shared. `kind` selects the composer ('rf' or the shipped 'onebrain')."""
    if default_flip:
        off = _emit_subprocess("0", seed, kind=kind)      # ESCAPE: BRAIN_COMPOSER_MERGE=0 (private, byte-id revert)
        on = _emit_subprocess(None, seed, kind=kind)      # DEFAULT: no env -> reads _COMPOSER_IN_POOL1_DEFAULT_ON
    else:
        off = _emit_subprocess("0", seed, kind=kind)
        on = _emit_subprocess("1", seed, kind=kind)

    # (1) recall byte-identical + correct + moat
    recall_byte_id = bool(off["recall"] == on["recall"])
    qp_byte_id = bool(off["query_patient"] == on["query_patient"])
    # correctness (the stored tiny-demo answers)
    expect = {"dog|chase": "cat", "cat|eat": "fish", "brain|use": "spikes",
              "brain|learn": "words", "brain|store": "memory", "owl|eat": None, "lion|roar": None}
    recall_correct = bool(all(on["recall"].get(k) == v for k, v in expect.items()))
    moat_ok = bool(off["moat_ok"] and on["moat_ok"] and on["moat_abstain"] in (None, "unknown", "", "i don't know"))

    # (2) surprise byte-identical + alive
    surp_err = _dict_maxerr(off["surprise"], on["surprise"], ["hz", "th"])
    surp_bool_id = bool(all(off["surprise"][k]["surprised"] == on["surprise"][k]["surprised"]
                            for k in off["surprise"]))
    surprise_byte_id = bool(surp_err <= 1e-9 and surp_bool_id and off["surprise"] and on["surprise"])
    # alive: a contradict case fires notably above a confirm case (ON pool)
    s = on["surprise"]
    conf_hz = s.get("dog|chase|cat|cat", {}).get("hz", 0.0)
    contra_hz = s.get("dog|chase|cat|fish", {}).get("hz", 0.0)
    surp_alive = bool(contra_hz >= max(2.0 * max(conf_hz, 1e-6), conf_hz + 1.0))

    # (3) world-model byte-identical + alive
    wm_err = _dict_maxerr(off["worldmodel"], on["worldmodel"], ["hz", "pp", "pn", "margin"])
    wm_sign_id = bool(all(off["worldmodel"][k].get("pred_sign") == on["worldmodel"][k].get("pred_sign")
                          for k in off["worldmodel"] if "pred_sign" in off["worldmodel"][k]))
    wm_bool_id = bool(all(off["worldmodel"][k].get("surprised") == on["worldmodel"][k].get("surprised")
                          for k in off["worldmodel"] if "surprised" in off["worldmodel"][k]))
    wm_byte_id = bool(wm_err <= 1e-9 and wm_sign_id and wm_bool_id and off["worldmodel"] and on["worldmodel"])
    w = on["worldmodel"]
    exp_hz = w.get("surp+1|+1", {}).get("hz", 0.0)
    vio_hz = w.get("surp+1|-1", {}).get("hz", 0.0)
    wm_alive = bool(vio_hz >= max(2.0 * max(exp_hz, 1e-6), exp_hz + 1.0))

    # (4) one pool
    one_pool = on.get("one_pool") or {}
    one_pool_ok = bool(one_pool.get("genuinely_one_pool"))

    go = bool(recall_byte_id and qp_byte_id and recall_correct and moat_ok
              and surprise_byte_id and surp_alive
              and wm_byte_id and wm_alive
              and one_pool_ok)

    # PRECONDITIONS carried WITH the verdict (tools.verdict.Verdict): each axis that must hold for the wire GO,
    # measured beside it -> the artifact travels with what earned it (gate: verdict-preconditions).
    from tools.verdict import Verdict
    _vlabel = (f"onebrain composer -> pool #1 DEFAULT-FLIP ({kind}; DEFAULT-no-env vs ESCAPE MERGE=0)"
               if default_flip else f"composer ({kind}) -> pool #1 production wire (opt-in, default-off)")
    V = Verdict(_vlabel)
    V.require("composer recall byte-identical (off==on)", recall_byte_id, expect=True)
    V.require("query_patient byte-identical (off==on)", qp_byte_id, expect=True)
    V.require("recall correct (stored answers)", recall_correct, expect=True)
    V.require("no-confab moat abstains on unstored cue", moat_ok, expect=True)
    V.require("surprise byte-identical (off==on)", surprise_byte_id, expect=True)
    V.require("surprise faculty alive (contradict>>confirm)", surp_alive, expect=True)
    V.require("worldmodel byte-identical (off==on)", wm_byte_id, expect=True)
    V.require("worldmodel faculty alive (violated>>expected)", wm_alive, expect=True)
    V.require("genuinely one pool when ON", one_pool_ok, expect=True)
    V.control("surprise separation (contradict vs confirm)", contra_hz, conf_hz, min_separation=1.0,
              note="the byte-identity is of a LIVE organ, not exact-of-dead")
    V.control("worldmodel separation (violated vs expected)", vio_hz, exp_hz, min_separation=1.0,
              note="the byte-identity is of a LIVE organ, not exact-of-dead")
    verdict = V.decide(go=go, verbose=False)

    return {
        "mode": ("onebrain_composer_pool1_DEFAULT_FLIP" if default_flip
                 else "onebrain_composer_pool1_production_verify"),
        "kind": kind, "default_flip": bool(default_flip), "seed": seed,
        "status": verdict["status"], "go": verdict["go"],
        "preconditions": verdict["preconditions"],
        "undefined_reasons": verdict["undefined_reasons"],
        "recall_byte_identical": recall_byte_id, "query_patient_byte_identical": qp_byte_id,
        "recall_correct": recall_correct, "moat_ok": moat_ok,
        "surprise_maxerr_hz": surp_err, "surprise_byte_identical": surprise_byte_id,
        "surprise_alive": surp_alive, "surprise_confirm_hz": conf_hz, "surprise_contradict_hz": contra_hz,
        "worldmodel_maxerr_hz": wm_err, "worldmodel_byte_identical": wm_byte_id,
        "worldmodel_alive": wm_alive, "worldmodel_expected_hz": exp_hz, "worldmodel_violated_hz": vio_hz,
        "one_pool": one_pool, "one_pool_ok": one_pool_ok,
        "GO": go,
        "off_recall": off["recall"], "on_recall": on["recall"],
        "off_surprise": off["surprise"], "on_surprise": on["surprise"],
        "off_worldmodel": off["worldmodel"], "on_worldmodel": on["worldmodel"],
    }


def _print_one(res):
    tag = "DEFAULT-FLIP" if res.get("default_flip") else "WIRE"
    print(f"=== COMPOSER({res.get('kind')}) -> POOL #1 {tag} VERIFY seed={res['seed']} "
          f"({'DEFAULT-no-env vs ESCAPE MERGE=0' if res.get('default_flip') else 'OFF vs ON'}) ===")
    print(f"  (1) recall byte-identical (who/what):    {res['recall_byte_identical']}")
    print(f"      query_patient byte-identical:        {res['query_patient_byte_identical']}")
    print(f"      recall CORRECT (stored answers):     {res['recall_correct']}")
    print(f"  (2) no-confab MOAT abstains (unstored):  {res['moat_ok']}")
    print(f"  (4) SURPRISE byte-identical:             {res['surprise_byte_identical']}  (max err {res['surprise_maxerr_hz']:.2e} Hz)")
    print(f"      surprise ALIVE (contradict>>confirm):{res['surprise_alive']}  (confirm {res['surprise_confirm_hz']:.2f} vs contradict {res['surprise_contradict_hz']:.2f} Hz)")
    print(f"  (4) WORLD-MODEL byte-identical:          {res['worldmodel_byte_identical']}  (max err {res['worldmodel_maxerr_hz']:.2e} Hz)")
    print(f"      world-model ALIVE (violated>>exp):   {res['worldmodel_alive']}  (expected {res['worldmodel_expected_hz']:.2f} vs violated {res['worldmodel_violated_hz']:.2f} Hz)")
    print(f"  (3) GENUINELY ONE POOL when DEFAULT:     {res['one_pool_ok']}  {res['one_pool']}")
    print(f"  ==> per-seed GO:                         {res['GO']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit", action="store_true", help="run ONE panel for the current env; write JSON to --out")
    ap.add_argument("--compare", action="store_true", help="run flag OFF/ON in subprocesses + diff")
    ap.add_argument("--default-flip", action="store_true",
                    help="DEFAULT-FLIP mode: diff DEFAULT-no-env vs ESCAPE BRAIN_COMPOSER_MERGE=0 over --seeds "
                         "(exercises the SHIPPED composer path routed to pool #1). Requires the default flag ON.")
    ap.add_argument("--kind", default=None, help="composer kind under test: 'rf' or 'onebrain' "
                    "(default 'onebrain' for --default-flip, else 'rf').")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds for --default-flip (default 42,43,44,100,101,102)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.emit:
        res = run_panel(args.seed)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(res, f, indent=2)
        else:
            print(json.dumps(res, indent=2))
        return

    if args.default_flip:
        # The DEFAULT-FLIP verify: the SHIPPED (onebrain) composer path routed to pool #1, DEFAULT-no-env (reads
        # _COMPOSER_IN_POOL1_DEFAULT_ON) vs ESCAPE MERGE=0, over 6 seeds. GO iff 6/6 seeds pass criteria 1-4.
        kind = args.kind or "onebrain"
        seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else ["42", "43", "44", "100", "101", "102"])]
        rows = [compare(s, kind=kind, default_flip=True) for s in seeds]
        for r in rows:
            _print_one(r)
        n_go = sum(1 for r in rows if r["GO"])
        agg = {
            "mode": "onebrain_composer_pool1_DEFAULT_FLIP_6seed", "kind": kind, "seeds": seeds,
            "n_go": n_go, "n_seeds": len(seeds), "all_go": bool(n_go == len(seeds)),
            "per_seed_go": {r["seed"]: r["GO"] for r in rows},
            "criteria_by_seed": {r["seed"]: {
                "1_recall_byte_id": r["recall_byte_identical"], "1_qp_byte_id": r["query_patient_byte_identical"],
                "1_recall_correct": r["recall_correct"], "2_moat_ok": r["moat_ok"],
                "3_one_pool": r["one_pool_ok"],
                "4_surprise_byte_id": r["surprise_byte_identical"], "4_surprise_alive": r["surprise_alive"],
                "4_wm_byte_id": r["worldmodel_byte_identical"], "4_wm_alive": r["worldmodel_alive"],
                "surprise_maxerr_hz": r["surprise_maxerr_hz"], "worldmodel_maxerr_hz": r["worldmodel_maxerr_hz"],
            } for r in rows},
            "rows": rows,
        }
        print(f"\n==> DEFAULT-FLIP {kind}: {n_go}/{len(seeds)} seeds GO (criteria 1-4).  "
              f"ALL-GO={agg['all_go']}")
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(agg, f, indent=2)
            print(f"  wrote {args.out}")
        return

    # default = single-seed compare (the original opt-in wire verify; kind default 'rf')
    res = compare(args.seed, kind=(args.kind or "rf"))
    _print_one(res)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
