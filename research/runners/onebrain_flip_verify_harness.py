"""REUSABLE flip-verify harness for the one-brain INTEGRATION program (Phase 1).

Generalizes `_xedge_flip_production_verify.py`'s three-arm verify into ONE parameterized entry point that any
future default-flip in the program can gate on. A flip is specified by an `EdgeSpec` (its flag, its `*_LEARN`/
`*_LESION` flags, its real-traffic probe items, its per-arm env configs, and a few edge-specific read-outs); the
harness then runs, through the REAL `webapp.server.brain_chat` handler on fresh per-(config,seed) brains:

  ARM A  BYTE-IDENTICAL-OFF  — with the flag explicitly ="0" (NEVER `os.environ.pop`, the 2026-08-27 off-arm
         staleness class — see `tools/gates/flip_offarm_staleness.py`), the visible ANSWER strings + comprehension/
         abstain DECISIONS reproduce env-unset (today's main). Protects every caller who opts out under the flip.

  ARM B  VISIBLE-ON-REAL-TRAFFIC + LESION-ATTRIBUTABLE — on a content-ambiguous item where THIS session holds the
         relevant state, the handler RESPONSE actually differs with the edge ON, and that difference VANISHES under
         the edge's `*_LESION`. Positional-residual-aware, `n_hollow` must be 0 (the #94 anti-hollow bar: no seed is
         internally-driven-yet-invisible), lesion-attribution via `tools.lab.attributable_to`.

  ARM C  NO-REGRESSION — the shipped default-ON config's well-formed / question / moat DECISIONS are unchanged vs
         off, learn-wiring is live, AND (the genuinely-new instrument) the shipped-faculty REGRESSION BATTERY
         (`onebrain_regression_battery.py`) asserts the flip does not silently break ANY of the OTHER ~29 default-ON
         faculties — the cross-faculty test that no per-faculty flip-verify has ever run.

WHY A HARNESS (owner, one-brain INTEGRATION program 2026-09-02, Phase 1). Every merge wave + cross-edge flip below
needs the same three-arm gate; re-deriving it per edge is how ARM C stayed a single faculty's fixed items for weeks
while a flip could silently regress the roster. This is the reusable, extensible gate.

DE-RISK (proves the generalization changed NOTHING for the one case with a known-good answer): the harness aggregate,
fed the BANKED d6-WM->comprehension per-worker data, reproduces that edge's existing ARM A/B/C verdict BYTE-FOR-BYTE
vs the reference `_xedge_flip_production_verify._aggregate` (all three banked cupy artifacts, GO and NO-GO). Run:
  SIM_BACKEND=numpy python -m research.runners.onebrain_flip_verify_harness --derisk

Full run (orchestrator; spawns ONE fresh brain per (config,seed) worker):
  SIM_BACKEND=cupy python -m research.runners.onebrain_flip_verify_harness --edge xedge \
      --out research/findings/raw/_flip_verify_harness/xedge_6seed.json
Tiny numpy smoke (mechanics only; 1 seed):
  SIM_BACKEND=numpy python -m research.runners.onebrain_flip_verify_harness --edge xedge --smoke \
      --out research/findings/raw/_flip_verify_harness/xedge_numpy_smoke.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from typing import Callable


# ── the edge specification (everything a flip needs to be verified) ──────────────────────────────────────────────
@dataclasses.dataclass
class EdgeSpec:
    """One default-flip's verify recipe. The reusable harness is edge-agnostic; ALL edge specifics live here."""
    name: str                                   # short id, e.g. "xedge"
    flag: str                                   # the primary flag being flipped default-ON (e.g. BRAIN_ONEBRAIN_XEDGE)
    lesion_flag: str                            # its *_LESION flag (ARM B attribution)
    b_edge: str                                 # "learn" or "frozen" — selects the ARM B config pair name
    configs: dict                               # arm-config-key -> {"env": {...}, "per_turn": bool}
    well_items: list                            # comprehensible items (no repair) — ARM A/C battery
    question: str                               # a question item -> None comprehension — ARM A/C battery
    amb_items: list                             # content-ambiguous items — ARM B visibility
    hold_turn: str                              # a >=2-referent turn that sets this session's focus (ARM B)
    all_seeds: list
    shipped_seed: int
    w0_role: Callable[[int], str]               # seed -> the discourse role the positional focus carries (ARM B)
    decisions_equal: Callable[[dict, dict], bool]   # the byte-identical decision-equality instrument (ARM A/C)
    prime_pool: Callable[[dict, int], dict]     # (config_spec, seed) -> pool_info; primes any process-global singleton
    extract_turn: Callable[[dict], dict]        # brain_chat response json -> the decision dict the arms read
    reset_env_keys: tuple                       # the flags to clear before setting a config's env (fresh subprocess)


# ── worker: drive the REAL handler on ONE freshly-built brain at ONE seed/config ─────────────────────────────────
def _brain_chat_turn(extract, msg, session, reset=False):
    """One real /api/brain-chat turn; a handler crash is a datum, never an abort."""
    from webapp.server import brain_chat, BrainChatRequest
    try:
        r = brain_chat(BrainChatRequest(session=session, message=msg, brain="tiny-demo",
                                        renderer="stub", rich=False, reset=reset))
        d = json.loads(r.body)
    except Exception as e:
        return {"_error": "%s: %s" % (type(e).__name__, e), "answer": None}
    return extract(d)


def run_worker(spec: EdgeSpec, config: str, seed: int, out_path: str) -> int:
    """Drive the arms' turns for ONE (config, seed) on a fresh brain. Mirrors the reference worker's per-arm turn
    plan (ARM A/C battery turns; ARM B novisi-vs-held visibility) so the aggregate reads an identical structure."""
    os.environ.setdefault("SIM_BACKEND", "numpy")
    # OFF-ARM DISCIPLINE (2026-08-27 class): a fresh subprocess inherits the orchestrator env; clear every edge flag,
    # then set EXACTLY this config's env EXPLICITLY. The config env sets "0"/"1" — never a pop — so an OFF arm stays
    # OFF even after this flag's default flips ON. (The harness NEVER relies on unset==OFF.)
    for k in spec.reset_env_keys:
        os.environ.pop(k, None)
    cfg_spec = spec.configs[config]
    for k, v in cfg_spec["env"].items():
        os.environ[k] = v

    pool_info = spec.prime_pool(cfg_spec, seed)

    def turn(msg, session, reset=False):
        return _brain_chat_turn(spec.extract_turn, msg, session, reset=reset)

    res: dict = {"config": config, "seed": seed, "env": cfg_spec["env"], "per_turn": cfg_spec.get("per_turn"),
                 "pool": pool_info}
    arm = config[0]

    if arm == "A" or config == "C_on_shipped":
        res["well"] = {m: turn(m, "well_%d" % i, reset=True) for i, m in enumerate(spec.well_items)}
        res["question"] = turn(spec.question, "q0", reset=True)
        res["amb_novisi"] = {m: turn(m, "nv_%d" % i, reset=True) for i, m in enumerate(spec.amb_items)}

    if config == "C_on_shipped":
        turn(spec.hold_turn, "moat", reset=True)
        res["moat_well_held"] = turn(spec.well_items[0], "moat", reset=False)
        traj = []
        for t in range(4):
            turn(spec.hold_turn, "learn", reset=(t == 0))
            r = turn(spec.well_items[0], "learn", reset=False)
            traj.append({"turn": t, "xedge_live_learn": r.get("xedge_live_learn")})
        res["learn_traj"] = traj
        # the shipped-faculty REGRESSION BATTERY runs alongside ARM C (its own on-vs-off collect is orchestrated).
        # Recorded per-worker as a marker; the actual on-vs-off battery is run by the orchestrator (see _orchestrate).

    if arm == "B":
        vis = {}
        for i, m in enumerate(spec.amb_items):
            novisi = turn(m, "nv_%d" % i, reset=True)
            turn(spec.hold_turn, "hd_%d" % i, reset=True)
            held = turn(m, "hd_%d" % i, reset=False)
            vis[m] = {"novisi": novisi, "held": held,
                      "answer_differs": bool(novisi.get("answer") != held.get("answer")),
                      "role_differs": bool(novisi.get("repair_role") != held.get("repair_role"))}
        res["visibility"] = vis

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print("[worker %s s%d] primed=%s -> %s" % (config, seed, pool_info.get("primed"), out_path), flush=True)
    return 0


# ── the generalized A/B/C aggregate (a faithful port of _xedge_flip_production_verify._aggregate) ───────────────
def aggregate(spec: EdgeSpec, per: dict, seeds: list, b_edge: str, battery: dict | None = None) -> dict:
    """The three-arm verdict, edge-parameterized. On the xedge spec + banked per-worker this is BYTE-IDENTICAL to
    the reference _aggregate (verified in --derisk). `battery` (optional) is the cross-faculty regression result;
    when present it becomes ARM C's additional gate and folds into FLIP_VERIFY_GO. When absent (e.g. banked data)
    the core three arms + GO are unchanged, so banked reproduction stays byte-identical."""
    well_items, question, amb_items = spec.well_items, spec.question, spec.amb_items
    _eq = spec.decisions_equal

    # ── ARM A: byte-identical-off ──
    base, off = per.get("A_baseline_s%d" % spec.shipped_seed), per.get("A_off_s%d" % spec.shipped_seed)
    a_items, a_diffs = [], []
    if base and off:
        for m in well_items:
            eq = _eq(off["well"][m], base["well"][m]); a_items.append(eq)
            if not eq:
                a_diffs.append({"item": m, "off": off["well"][m], "baseline": base["well"][m]})
        eqq = _eq(off["question"], base["question"]); a_items.append(eqq)
        if not eqq:
            a_diffs.append({"item": question, "off": off["question"], "baseline": base["question"]})
        for m in amb_items:
            eq = _eq(off["amb_novisi"][m], base["amb_novisi"][m]); a_items.append(eq)
            if not eq:
                a_diffs.append({"item": m + " (novisi)", "off": off["amb_novisi"][m], "baseline": base["amb_novisi"][m]})
    a_pass = bool(a_items) and all(a_items)

    # ── ARM B: visible-on-traffic + lesion-attributable, positional-residual-aware, per seed ──
    on_key, les_key = "B_on_%s" % b_edge, "B_lesion_%s" % b_edge
    b_seeds, n_visible, n_inert, n_hollow, all_revert = [], 0, 0, 0, True
    n_flips_lesion = 0
    shipped_visible = False
    for s in seeds:
        on = per.get("%s_s%d" % (on_key, s))
        les = per.get("%s_s%d" % (les_key, s))
        w0role = spec.w0_role(s)
        grown = w0role in ("agent", "patient")
        seed_rec = {"seed": s, "w0_role": w0role, "grown_focus": grown}
        if not on:
            seed_rec.update(missing=True); b_seeds.append(seed_rec); all_revert = False; continue
        m = amb_items[0]
        von = on["visibility"][m]
        held = von["held"]; novisi = von["novisi"]
        wm_res = (held.get("wm_resolved") is True)
        ans_differs = bool(held.get("answer") != novisi.get("answer"))
        role_off_content = (held.get("repair_role") != held.get("content_role"))
        visible = bool(grown and wm_res and ans_differs and role_off_content)
        hollow = bool(wm_res and not ans_differs)
        inert = bool((not grown) and (held.get("wm_resolved") is not True) and (not ans_differs))
        lesion_flip = False
        if les:
            lm = les["visibility"][m]["held"]
            reverts = bool(lm.get("wm_resolved") is not True and lm.get("answer") == novisi.get("answer"))
            lesion_flip = bool(lm.get("wm_resolved") is True or lm.get("answer") != novisi.get("answer"))
        else:
            reverts = False
        seed_rec.update(wm_resolved=wm_res, answer_differs=ans_differs, role_off_content=role_off_content,
                        visible=visible, hollow=hollow, correctly_inert=inert, lesion_reverts=reverts,
                        held_role=held.get("repair_role"), content_role=held.get("content_role"),
                        held_answer=held.get("answer"), novisi_answer=novisi.get("answer"),
                        wm_margin=held.get("wm_margin"))
        if visible:
            n_visible += 1
        if inert:
            n_inert += 1
        if hollow:
            n_hollow += 1
        if lesion_flip:
            n_flips_lesion += 1
        if not reverts:
            all_revert = False
        if s == spec.shipped_seed and visible and reverts:
            shipped_visible = True
        seed_rec["seed_ok"] = bool((visible and reverts) if grown else inert) and (not hollow)
        b_seeds.append(seed_rec)

    try:
        from tools.lab import attributable_to
        b_attribution = attributable_to("%s visible decision-flip: cross-edge ON vs lesioned" % spec.name,
                                        float(n_visible), float(n_flips_lesion))
    except Exception as _ae:
        b_attribution = None
        print("[aggregate] attribution call failed: %s: %s" % (type(_ae).__name__, _ae), flush=True)

    b_pass = bool(shipped_visible and (n_hollow == 0) and all_revert and (n_flips_lesion == 0)
                  and all(r.get("seed_ok") for r in b_seeds))

    # ── ARM C: no-regression on the shipped config + LEARN wiring live ──
    c = per.get("C_on_shipped_s%d" % spec.shipped_seed)
    c_items, c_diffs, moat_ok, learn_live = [], [], None, None
    if c and off:
        for m in well_items:
            eq = _eq(c["well"][m], off["well"][m]); c_items.append(eq)
            if not eq:
                c_diffs.append({"item": m, "on": c["well"][m], "off": off["well"][m]})
        eqq = _eq(c["question"], off["question"]); c_items.append(eqq)
        if not eqq:
            c_diffs.append({"item": question, "on": c["question"], "off": off["question"]})
        moat = c.get("moat_well_held") or {}
        moat_ok = bool(moat.get("comprehended") is True and not moat.get("abstained"))
        learn_live = bool((c.get("pool") or {}).get("learned") and (c.get("pool") or {}).get("live_per_turn"))
    c_pass = bool(c_items) and all(c_items) and bool(moat_ok) and bool(learn_live)

    battery_pass = None
    if battery is not None:
        battery_pass = bool(battery.get("all_pass"))

    go = bool(a_pass and b_pass and c_pass and (battery_pass if battery is not None else True))
    out = {
        "arm_A_byte_identical_off": {"pass": a_pass, "n_match": sum(a_items), "n_total": len(a_items), "diffs": a_diffs},
        "arm_B_visible_on_real_traffic": {
            "pass": b_pass, "b_edge": b_edge, "shipped_seed42_visible": shipped_visible,
            "n_visible_grown_focus": n_visible, "n_correctly_inert_ctrl_focus": n_inert,
            "n_hollow": n_hollow, "n_flips_surviving_lesion": n_flips_lesion,
            "flip_fraction_attributable_to_crossedge": b_attribution,
            "all_seeds_lesion_revert": all_revert, "per_seed": b_seeds,
            "note": ("through the real handler the WM focus is the positional CAND_POOLS[0]=w0; visibility requires "
                     "w0 to be a grown role for that seed (declared positional-binding residual). n_hollow==0 is the "
                     "anti-hollow bar; the shipped brain is seed 42 (w0=agent -> visible)."),
        },
        "arm_C_no_regression": {"pass": c_pass, "moat_well_held_ok": moat_ok, "learn_wiring_live": learn_live,
                                "diffs": c_diffs},
        "FLIP_VERIFY_GO": go,
    }
    if battery is not None:
        out["arm_C_regression_battery"] = {"pass": battery_pass, "n_faculties": battery.get("n_faculties"),
                                           "n_regressed": battery.get("n_regressed"),
                                           "regressed": battery.get("regressed")}
    return out


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


def _spawn(edge: str, config: str, seed: int, raw_dir: str, mem_floor: int) -> dict:
    fm = _free_mb()
    if fm and fm < mem_floor:
        return {"_abort": "free mem %dMB < %dMB floor before %s s%d" % (fm, mem_floor, config, seed)}
    cfg_out = os.path.join(raw_dir, "w_%s_s%d.json" % (config, seed))
    env = dict(os.environ)
    print("[orch] %s s%d free=%dMB -> spawning worker" % (config, seed, fm), flush=True)
    p = subprocess.run([sys.executable, "-u", "-m", "research.runners.onebrain_flip_verify_harness",
                        "--edge", edge, "--worker", config, "--seed", str(seed), "--out", cfg_out], env=env)
    if p.returncode != 0 or not os.path.exists(cfg_out):
        return {"_failed": "%s s%d rc=%d" % (config, seed, p.returncode)}
    with open(cfg_out) as f:
        return json.load(f)


def _orchestrate(spec: EdgeSpec, edge: str, out_path: str, seeds: list, b_edge: str, mem_floor: int,
                 smoke: bool, run_battery: bool) -> int:
    raw_dir = os.path.dirname(out_path)
    os.makedirs(raw_dir, exist_ok=True)
    jobs = [("A_baseline", spec.shipped_seed), ("A_off", spec.shipped_seed), ("C_on_shipped", spec.shipped_seed)]
    for s in seeds:
        jobs.append(("B_on_%s" % b_edge, s))
        jobs.append(("B_lesion_%s" % b_edge, s))
    if smoke:
        jobs = [("A_baseline", spec.shipped_seed), ("A_off", spec.shipped_seed),
                ("B_on_%s" % b_edge, spec.shipped_seed), ("B_lesion_%s" % b_edge, spec.shipped_seed)]
        seeds = [spec.shipped_seed]

    per, problems = {}, []
    for cfg, s in jobs:
        rec = _spawn(edge, cfg, s, raw_dir, mem_floor)
        if "_abort" in rec:
            print("[orch] ABORT: %s" % rec["_abort"], flush=True)
            return 2
        if "_failed" in rec:
            problems.append(rec["_failed"])
            print("[orch] WORKER FAILED: %s" % rec["_failed"], flush=True)
            continue
        per["%s_s%d" % (cfg, s)] = rec

    battery_result = None
    if run_battery and not smoke:
        try:
            from research.runners.onebrain_regression_battery import run_regression_battery
            battery_result = run_regression_battery(flag=spec.flag, out_dir=raw_dir)
        except Exception as e:
            battery_result = {"all_pass": False, "error": "%s: %s" % (type(e).__name__, e)}

    agg = aggregate(spec, per, seeds, b_edge, battery=battery_result)
    payload = {
        "probe": "onebrain_flip_verify_harness:%s" % spec.name,
        "backend": os.environ.get("SIM_BACKEND", "numpy"), "brain": "tiny-demo", "renderer": "stub",
        "smoke": bool(smoke), "b_edge": b_edge, "seeds": seeds, "shipped_seed": spec.shipped_seed,
        "flip_target": "%s (+its learn flag) default ON; =0 escape hatch preserved" % spec.flag,
        "worker_problems": problems, "aggregate": agg, "per_worker": per,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print("\n===== FLIP-VERIFY VERDICT (%s, real /api/brain-chat handler) =====" % spec.name, flush=True)
    for k in ("arm_A_byte_identical_off", "arm_B_visible_on_real_traffic", "arm_C_no_regression"):
        print("  [%s] %s" % ("PASS" if agg[k]["pass"] else "FAIL", k), flush=True)
    if "arm_C_regression_battery" in agg:
        b = agg["arm_C_regression_battery"]
        print("  [%s] arm_C_regression_battery: %s/%s faculties pass"
              % ("PASS" if b["pass"] else "FAIL",
                 (b.get("n_faculties") or 0) - (b.get("n_regressed") or 0), b.get("n_faculties")), flush=True)
    if problems:
        print("  worker_problems: %s" % problems, flush=True)
    print("\n  FLIP_VERIFY_GO = %s   wrote %s" % (agg["FLIP_VERIFY_GO"], out_path), flush=True)
    return 0


# ── the concrete xedge EdgeSpec (reuses the reference runner's helpers/constants -> exact reproduction) ─────────
def xedge_edge_spec() -> EdgeSpec:
    """The d6-WM->comprehension cross-edge (BRAIN_ONEBRAIN_XEDGE + _LEARN). Deliberately REUSES the reference
    runner's constants + helpers so the generalized aggregate is byte-identical on the banked verdict."""
    import research.runners._xedge_flip_production_verify as REF

    def _prime_pool(cfg_spec, seed):
        pool_info = {"primed": False}
        try:
            import research.runners.onebrain_xedge_production as OX
            OX.set_live_per_turn(bool(cfg_spec.get("per_turn")))
            if OX.xedge_enabled():
                pool = OX.get_xedge_pool(seed)
                if pool is not None and pool.ok:
                    pool_info = {"primed": True, "seed": pool.seed, "role": pool.role,
                                 "cross_weights": pool.cross_weights, "learned": bool(pool.learned),
                                 "live_per_turn": bool(pool.live_per_turn), "lesioned": OX.xedge_lesioned()}
                else:
                    pool_info = {"primed": False, "reason": "pool build failed / disabled"}
        except Exception as e:
            pool_info = {"primed": False, "error": "%s: %s" % (type(e).__name__, e)}
        return pool_info

    def _extract(d):
        rep = d.get("repair") or {}
        comp = d.get("comprehension") or {}
        return {
            "answer": d.get("answer"), "abstained": bool(d.get("abstained")),
            "comprehended": comp.get("comprehended"), "repair_role": rep.get("role"),
            "repair_kind": rep.get("kind"), "content_role": rep.get("content_role"),
            "wm_resolved": rep.get("wm_resolved"), "wm_margin": rep.get("wm_margin"),
            "repaired": rep.get("repaired"), "not_understood": bool(d.get("not_understood")),
            "multiref_n": (d.get("multiref") or {}).get("n_referents"),
            "xedge_live_learn": comp.get("xedge_live_learn"),
        }

    return EdgeSpec(
        name="xedge",
        flag="BRAIN_ONEBRAIN_XEDGE",
        lesion_flag="BRAIN_ONEBRAIN_XEDGE_LESION",
        b_edge="learn",
        configs=REF.CONFIGS,
        well_items=REF.WELL_ITEMS,
        question=REF.QUESTION,
        amb_items=REF.AMB_ITEMS,
        hold_turn=REF.HOLD_TURN,
        all_seeds=REF.ALL_SEEDS,
        shipped_seed=REF.SHIPPED_SEED,
        w0_role=REF._w0_role,
        decisions_equal=REF._decisions_equal,
        prime_pool=_prime_pool,
        extract_turn=_extract,
        reset_env_keys=("BRAIN_ONEBRAIN_XEDGE", "BRAIN_ONEBRAIN_XEDGE_LEARN", "BRAIN_ONEBRAIN_XEDGE_LESION"),
    )


_EDGES = {"xedge": xedge_edge_spec}


# ── the de-risk: reproduce the banked d6->comprehension verdict byte-for-byte ────────────────────────────────────
def derisk() -> int:
    """Feed the BANKED xedge per-worker data to BOTH the reference _aggregate and the harness aggregate; require the
    harness to reproduce the reference verdict byte-for-byte on every banked artifact (GO and NO-GO). This proves the
    generalization changed nothing for the one edge with a known-good answer. No brain builds (banked data)."""
    import research.runners._xedge_flip_production_verify as REF
    spec = xedge_edge_spec()
    raw = "research/findings/raw/_xedge_flip_verify"
    arts = [a for a in ("flip_verify_cupy_6seed.json", "flip_verify_cupy_6seed_strengthened.json",
                        "flip_verify_cupy_6seed_indirection.json")
            if os.path.exists(os.path.join(raw, a))]
    if not arts:
        print("DERISK: NO banked artifacts found under %s" % raw)
        return 2
    all_ok = True
    results = []
    for art in arts:
        d = json.load(open(os.path.join(raw, art)))
        per, seeds, b_edge = d["per_worker"], d["seeds"], d.get("b_edge", "learn")
        ref_agg = REF._aggregate(per, seeds, b_edge)
        harn_agg = aggregate(spec, per, seeds, b_edge)  # no battery -> core arms only
        # compare on the reference's own keys (harness may add optional battery keys not in banked data)
        shared = {k: harn_agg.get(k) for k in ref_agg}
        ref_s = json.dumps(ref_agg, sort_keys=True, default=str)
        harn_s = json.dumps(shared, sort_keys=True, default=str)
        banked_s = json.dumps(d["aggregate"], sort_keys=True, default=str)
        harn_vs_ref = (harn_s == ref_s)
        ref_vs_banked = (ref_s == banked_s)
        ok = bool(harn_vs_ref and ref_vs_banked)
        all_ok = all_ok and ok
        diff_keys = []
        if not harn_vs_ref:
            for k in ref_agg:
                if json.dumps(ref_agg[k], sort_keys=True, default=str) != json.dumps(shared.get(k), sort_keys=True, default=str):
                    diff_keys.append(k)
        results.append({"artifact": art, "banked_GO": d["aggregate"]["FLIP_VERIFY_GO"],
                        "harness_GO": harn_agg["FLIP_VERIFY_GO"], "harness_reproduces_reference": harn_vs_ref,
                        "reference_reproduces_banked": ref_vs_banked, "byte_identical": ok, "diff_keys": diff_keys})
        print("[derisk %s] banked_GO=%s harness_GO=%s harness==reference=%s reference==banked=%s%s"
              % (art, d["aggregate"]["FLIP_VERIFY_GO"], harn_agg["FLIP_VERIFY_GO"], harn_vs_ref, ref_vs_banked,
                 ("  DIFFERS: %s" % diff_keys) if diff_keys else ""), flush=True)
    verdict = {"DERISK_GO": all_ok, "n_artifacts": len(arts), "cases": results}
    out = "research/findings/raw/_flip_verify_harness/derisk_xedge_reproduction.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(verdict, open(out, "w"), indent=2, default=str)
    print("\n  DERISK_GO = %s  (harness reproduces the banked d6->comprehension verdict byte-for-byte on %d/%d "
          "artifacts)   wrote %s" % (all_ok, sum(r["byte_identical"] for r in results), len(arts), out), flush=True)
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default="xedge", choices=sorted(_EDGES), help="which flip's EdgeSpec to verify")
    ap.add_argument("--worker", default=None, help="internal: run one (config) worker")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="research/findings/raw/_flip_verify_harness/verify.json")
    ap.add_argument("--mem-floor", type=int, default=6000)
    ap.add_argument("--smoke", action="store_true", help="mechanics-only: 1 seed, ARM A + one ARM B pair")
    ap.add_argument("--no-battery", action="store_true", help="skip the ARM C cross-faculty regression battery")
    ap.add_argument("--derisk", action="store_true", help="reproduce the banked xedge verdict byte-for-byte (no builds)")
    args = ap.parse_args()
    if args.derisk:
        return derisk()
    spec = _EDGES[args.edge]()
    if args.worker:
        return run_worker(spec, args.worker, args.seed, args.out)
    return _orchestrate(spec, args.edge, args.out, spec.all_seeds, spec.b_edge, args.mem_floor,
                        args.smoke, run_battery=(not args.no_battery))


if __name__ == "__main__":
    raise SystemExit(main())
