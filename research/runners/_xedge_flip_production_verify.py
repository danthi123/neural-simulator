"""VERIFY-THEN-STAGE the production-default flip of the one-brain d6-WM->comprehension cross-edge, through the REAL
`/api/brain-chat` handler on the ACTUAL production brain (tiny-demo). The CONTROLLER reads the final JSON and makes
the flip decision; this runner only MEASURES + verdicts. It NEVER flips a default and NEVER blocks on a long run.

WHAT THE FLIP IS: `BRAIN_ONEBRAIN_XEDGE` + `BRAIN_ONEBRAIN_XEDGE_LEARN`, both default-OFF today, -> default-ON. The
cross-edge lets a HELD WM referent (d6 multi-referent organ) resolve an otherwise content-ambiguous thematic role in
the D4 comprehension read, so the clarification the brain asks (the `repair_target` role -> `clarification_question`
wording) changes with what the conversation is holding, and reverts when the cross-edge is lesioned.

THREE ARMS (all through `webapp.server.brain_chat`, the SAME entry the HTTP handler calls -- never a controlled organ
harness, per the #94 anti-hollow reversal):

  ARM A  BYTE-IDENTICAL-OFF (seed 42, the safety floor): with the flag explicitly =0 (the escape hatch) the visible
         ANSWER strings + the comprehension/abstain DECISIONS reproduce env-unset (today's main). This protects every
         caller who opts out under the flip. (Full back-to-back numeric identity of internal margins is NOT the
         instrument: a background-noise process advances between reads -- see the 2026-08-27 leak-fix finding -- so
         the reproducible claim is answer-string + decision-variable equality, which IS the user-facing output.)

  ARM B  VISIBLE-ON-REAL-TRAFFIC + LESION-ATTRIBUTABLE (6 seeds 42/43/44/100/101/102): on a content-ambiguous
         transitive where THIS session holds a WM referent, the `/api/brain-chat` RESPONSE's clarification wording
         actually DIFFERS from the no-focus response with the cross-edge ON, and that difference VANISHES under the
         lesion. THE POSITIONAL-BINDING RESIDUAL (declared in all three xedge findings) MATTERS HERE: through the real
         handler the WM focus is always `CAND_POOLS[0]=w0` (positional proxy), and the per-seed RANDOM role assignment
         makes w0 a GROWN role (p_agent/p_patient -> the edge transmits -> VISIBLE) for 4 of these 6 seeds
         (42/100/101/102) and the UNGROWN control pool (p_ctrl -> the edge is ~0.05 -> correctly INERT, wm not
         resolved) for the other 2 (43/44). So the honest per-seed expectation is: GROWN-focus seed -> visible +
         lesion-reverts; CTRL-focus seed -> correctly inert (NOT the hollow pattern). THE ANTI-HOLLOW BAR is that no
         seed is internally-driven-but-invisible (`wm_resolved` True yet the answer unchanged) -- `n_hollow` must be
         0. The SHIPPED production brain fixes the seed at 42 (w0=p_agent -> visible), so it is genuinely visible on
         real traffic. (The 6/6 mechanism-level lesion attribution that BYPASSES the positional proxy -- hold p_agent
         vs p_patient directly -- is the organ self-test `onebrain_xedge_production --verify-live --seeds ...`, staged
         alongside; this runner verifies the harder, real-traffic-visibility bar.)

  ARM C  NO-REGRESSION (seed 42, the SHIPPED per-turn config XEDGE+LEARN both on, `_LIVE_PER_TURN=True`): with the
         flags on, the well-formed / question / moat DECISIONS are unchanged vs off, no faculty degrades, and the
         per-turn LEARN wiring is live (`learned`/`live_per_turn` True; the edge starts at W0~0.05 -> it grows THROUGH
         the conversation, not pre-baked). Records the growth for the record; does not require a cold-start flip
         (that is the PART-3 "learns through the conversation" property, GO'd separately).

EDGE CONFIG for ARM B: both flags on with `set_live_per_turn(False)` so the IN-BRAIN self-supervised edge is at its
CONVERGED magnitude at build (the `--verify-live` protocol) -> visible in a few turns. The production default
(`_LIVE_PER_TURN=True`) reaches the SAME magnitude through use over a conversation (PART-3 finding), using the
IDENTICAL read mechanism; the frozen PART-1 edge closes the caveat identically too -- so this converged-edge
visibility test is faithful to the shipped read. `--b-edge frozen` switches ARM B to the PART-1 frozen edge
(`BRAIN_ONEBRAIN_XEDGE` only) if a frozen-only flip is preferred.

Run (orchestrator; spawns ONE fresh brain per (arm,config,seed) worker so the process-global `_POOL`/`_ORGAN`
singletons are built at the right seed -- they cannot be rebuilt at a new seed within one process):
  SIM_BACKEND=cupy python -m research.runners._xedge_flip_production_verify \
      --out research/findings/raw/_xedge_flip_verify/flip_verify_cupy_6seed.json
Tiny numpy smoke (mechanics only; 1 seed, ~a handful of turns):
  SIM_BACKEND=numpy python -m research.runners._xedge_flip_production_verify --smoke \
      --out research/findings/raw/_xedge_flip_verify/flip_verify_numpy_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


# ── the battery ──────────────────────────────────────────────────────────────────────────────────────────────
WELL_ITEMS = ["the wolf bites the apple", "the dog chases the cat"]   # comprehensible transitives (no repair)
QUESTION = "what does the wolf bite"                                  # a question -> None comprehension
AMB_ITEMS = ["the book carries the cup", "the cup holds the book"]    # content-ambiguous -> the repair/xedge path
HOLD_TURN = "the fox and the wolf walked in"                          # >=2 referents -> d6 sets this session's focus
ALL_SEEDS = [42, 43, 44, 100, 101, 102]
SHIPPED_SEED = 42

# arm/config -> the env flags it sets (before importing webapp.server) + the _LIVE_PER_TURN module toggle.
#   per_turn True  -> the SHIPPED per-turn edge (starts W0~0.05, grows one credited step per real chat turn)
#   per_turn False -> the build-curriculum edge at CONVERGED magnitude (the `--verify-live` protocol)
CONFIGS = {
    "A_baseline": {"env": {},                                                       "per_turn": True},
    "A_off":      {"env": {"BRAIN_ONEBRAIN_XEDGE": "0"},                            "per_turn": True},
    # ARM B edge configs (both flags = the named flip target); the orchestrator picks learn vs frozen via --b-edge.
    "B_on_learn":     {"env": {"BRAIN_ONEBRAIN_XEDGE": "1", "BRAIN_ONEBRAIN_XEDGE_LEARN": "1"}, "per_turn": False},
    "B_lesion_learn": {"env": {"BRAIN_ONEBRAIN_XEDGE": "1", "BRAIN_ONEBRAIN_XEDGE_LEARN": "1",
                               "BRAIN_ONEBRAIN_XEDGE_LESION": "1"},                              "per_turn": False},
    "B_on_frozen":     {"env": {"BRAIN_ONEBRAIN_XEDGE": "1"},                                    "per_turn": True},
    "B_lesion_frozen": {"env": {"BRAIN_ONEBRAIN_XEDGE": "1", "BRAIN_ONEBRAIN_XEDGE_LESION": "1"}, "per_turn": True},
    # ARM C = the SHIPPED default-ON config (both flags, per-turn live plasticity).
    "C_on_shipped": {"env": {"BRAIN_ONEBRAIN_XEDGE": "1", "BRAIN_ONEBRAIN_XEDGE_LEARN": "1"},   "per_turn": True},
}


def _w0_role(seed: int) -> str:
    """The DISCOURSE role the through-handler positional focus (`CAND_POOLS[0]`=w0) carries for `seed` -- the thing
    that decides whether the cross-edge is VISIBLE (grown p_agent/p_patient) or correctly INERT (ungrown p_ctrl) on
    real traffic. Mirrors `_role_assignment` exactly (RandomState(seed*7919+13).permutation(3))."""
    import numpy as np
    cand = ("w0", "w1", "w2")
    perm = np.random.RandomState(int(seed) * 7919 + 13).permutation(3)
    pools = [cand[i] for i in perm]
    p_agent, p_patient, p_ctrl = pools[0], pools[1], pools[2]
    if p_agent == "w0":
        return "agent"
    if p_patient == "w0":
        return "patient"
    return "ctrl"


# ── worker: drive the REAL handler on ONE freshly-built brain at ONE seed/config ─────────────────────────────────
def _run_worker(config: str, seed: int, out_path: str) -> int:
    os.environ.setdefault("SIM_BACKEND", "numpy")
    # clean every xedge flag, then set exactly this config's (a fresh subprocess inherits the orchestrator env).
    for k in ("BRAIN_ONEBRAIN_XEDGE", "BRAIN_ONEBRAIN_XEDGE_LEARN", "BRAIN_ONEBRAIN_XEDGE_LESION"):
        os.environ.pop(k, None)
    spec = CONFIGS[config]
    for k, v in spec["env"].items():
        os.environ[k] = v

    # PRIME the process-global xedge pool at THIS seed BEFORE importing the server: the server hardcodes
    # get_xedge_pool(42) / get_organ(seed=42), but both read the ALREADY-CACHED _POOL if it exists, so priming here
    # makes the whole handler path (comprehension organ + d6 slice + cross-edge role assignment) ride `seed`. Must
    # set _LIVE_PER_TURN before the build (it selects converged-vs-per-turn). No-op when the flag is off.
    pool_info = {"primed": False}
    try:
        import research.runners.onebrain_xedge_production as OX
        OX.set_live_per_turn(bool(spec["per_turn"]))
        if OX.xedge_enabled():
            pool = OX.get_xedge_pool(seed)   # builds + caches _POOL at `seed`
            if pool is not None and pool.ok:
                pool_info = {"primed": True, "seed": pool.seed, "role": pool.role,
                             "cross_weights": pool.cross_weights, "learned": bool(pool.learned),
                             "live_per_turn": bool(pool.live_per_turn), "lesioned": OX.xedge_lesioned()}
            else:
                pool_info = {"primed": False, "reason": "pool build failed / disabled"}
    except Exception as e:
        pool_info = {"primed": False, "error": f"{type(e).__name__}: {e}"}

    from webapp.server import brain_chat, BrainChatRequest

    def turn(msg, session, reset=False):
        try:
            r = brain_chat(BrainChatRequest(session=session, message=msg, brain="tiny-demo",
                                            renderer="stub", rich=False, reset=reset))
            d = json.loads(r.body)
        except Exception as e:   # a handler crash is itself a datum -> record, never abort the battery
            return {"_error": f"{type(e).__name__}: {e}", "answer": None}
        rep = d.get("repair") or {}
        comp = d.get("comprehension") or {}
        xll = comp.get("xedge_live_learn")
        return {
            "answer": d.get("answer"),
            "abstained": bool(d.get("abstained")),
            "comprehended": comp.get("comprehended"),
            "repair_role": rep.get("role"),
            "repair_kind": rep.get("kind"),
            "content_role": rep.get("content_role"),
            "wm_resolved": rep.get("wm_resolved"),
            "wm_margin": rep.get("wm_margin"),
            "repaired": rep.get("repaired"),
            "not_understood": bool(d.get("not_understood")),
            "multiref_n": (d.get("multiref") or {}).get("n_referents"),
            "xedge_live_learn": xll,
        }

    res: dict = {"config": config, "seed": seed, "env": spec["env"], "per_turn": spec["per_turn"],
                 "pool": pool_info}

    arm = config[0]
    if arm == "A" or config == "C_on_shipped":
        # NO-REGRESSION / BYTE-IDENTICAL battery: well-formed + question, each in its own fresh session.
        res["well"] = {m: turn(m, f"well_{i}", reset=True) for i, m in enumerate(WELL_ITEMS)}
        res["question"] = turn(QUESTION, "q0", reset=True)
        # the no-focus ambiguous reads (byte-identical + the ARM-B content baseline live here too).
        res["amb_novisi"] = {m: turn(m, f"nv_{i}", reset=True) for i, m in enumerate(AMB_ITEMS)}

    if config == "C_on_shipped":
        # MOAT: a well item read WHILE holding a focus must not flip to abstain/repair.
        turn(HOLD_TURN, "moat", reset=True)
        res["moat_well_held"] = turn(WELL_ITEMS[0], "moat", reset=False)
        # per-turn LEARN wiring: hold + resolve a few real turns and record the edge growing from W0~0.05.
        traj = []
        for t in range(4):
            turn(HOLD_TURN, "learn", reset=(t == 0))
            r = turn(WELL_ITEMS[0], "learn", reset=False)      # a comprehended turn -> credited plasticity step
            traj.append({"turn": t, "xedge_live_learn": r.get("xedge_live_learn")})
        res["learn_traj"] = traj
        try:
            import research.runners.onebrain_xedge_production as OX
            p = OX.get_xedge_pool()
            res["pool_after_learn"] = {"cross_weights": p.cross_weights, "n_live_credited": getattr(p, "n_live_credited", None)} if p else None
        except Exception as e:
            res["pool_after_learn"] = {"error": f"{type(e).__name__}: {e}"}

    if arm == "B":
        # VISIBILITY: for each ambiguous item, no-focus vs focus-held (same process, distinct sessions).
        vis = {}
        for i, m in enumerate(AMB_ITEMS):
            novisi = turn(m, f"nv_{i}", reset=True)            # fresh session, holds nothing -> content role
            turn(HOLD_TURN, f"hd_{i}", reset=True)             # session HOLDS >=2 referents -> focus = CAND_POOLS[0]
            held = turn(m, f"hd_{i}", reset=False)             # same session, now read the ambiguous item
            vis[m] = {
                "novisi": novisi, "held": held,
                "answer_differs": bool(novisi.get("answer") != held.get("answer")),
                "role_differs": bool(novisi.get("repair_role") != held.get("repair_role")),
            }
        res["visibility"] = vis

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"[worker {config} s{seed}] primed={pool_info.get('primed')} role={pool_info.get('role')} -> {out_path}",
          flush=True)
    return 0


# ── decision-level equality (the reproducible byte-identical instrument) ─────────────────────────────────────────
def _decisions_equal(a: dict, b: dict) -> bool:
    """The user-facing answer string + the comprehension/abstain verdict. Numeric margins are NOT compared (a
    background-noise process advances between reads; the reproducible claim is the decision + the answer text)."""
    return (a.get("answer") == b.get("answer") and a.get("abstained") == b.get("abstained")
            and a.get("comprehended") == b.get("comprehended") and a.get("repair_role") == b.get("repair_role")
            and a.get("not_understood") == b.get("not_understood"))


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


def _spawn(config: str, seed: int, raw_dir: str, mem_floor: int) -> dict:
    fm = _free_mb()
    if fm and fm < mem_floor:
        return {"_abort": f"free mem {fm}MB < {mem_floor}MB floor before {config} s{seed}"}
    cfg_out = os.path.join(raw_dir, f"w_{config}_s{seed}.json")
    env = dict(os.environ)
    print(f"[orch] {config} s{seed} free={fm}MB -> spawning worker", flush=True)
    p = subprocess.run([sys.executable, "-u", "-m", "research.runners._xedge_flip_production_verify",
                        "--worker", config, "--seed", str(seed), "--out", cfg_out], env=env)
    if p.returncode != 0 or not os.path.exists(cfg_out):
        return {"_failed": f"{config} s{seed} rc={p.returncode}"}
    with open(cfg_out) as f:
        return json.load(f)


def _aggregate(per: dict, seeds: list, b_edge: str) -> dict:
    on_key = f"B_on_{b_edge}"
    les_key = f"B_lesion_{b_edge}"

    # ── ARM A: byte-identical-off (A_off decisions+answers == A_baseline == today's main) ──
    base, off = per.get("A_baseline_s42"), per.get("A_off_s42")
    a_items, a_diffs = [], []
    if base and off:
        for m in WELL_ITEMS:
            eq = _decisions_equal(off["well"][m], base["well"][m]); a_items.append(eq)
            if not eq:
                a_diffs.append({"item": m, "off": off["well"][m], "baseline": base["well"][m]})
        eqq = _decisions_equal(off["question"], base["question"]); a_items.append(eqq)
        if not eqq:
            a_diffs.append({"item": QUESTION, "off": off["question"], "baseline": base["question"]})
        for m in AMB_ITEMS:
            eq = _decisions_equal(off["amb_novisi"][m], base["amb_novisi"][m]); a_items.append(eq)
            if not eq:
                a_diffs.append({"item": m + " (novisi)", "off": off["amb_novisi"][m], "baseline": base["amb_novisi"][m]})
    a_pass = bool(a_items) and all(a_items)

    # ── ARM B: visible-on-traffic + lesion-attributable, positional-residual-aware, per seed ──
    b_seeds, n_visible, n_inert, n_hollow, all_revert = [], 0, 0, 0, True
    n_flips_lesion = 0                                       # visible flips SURVIVING the cross-edge lesion (want 0)
    shipped_visible = False
    for s in seeds:
        on = per.get(f"{on_key}_s{s}")
        les = per.get(f"{les_key}_s{s}")
        w0role = _w0_role(s)
        grown = w0role in ("agent", "patient")
        seed_rec = {"seed": s, "w0_role": w0role, "grown_focus": grown}
        if not on:
            seed_rec.update(missing=True); b_seeds.append(seed_rec); all_revert = False; continue
        m = AMB_ITEMS[0]
        von = on["visibility"][m]
        held = von["held"]; novisi = von["novisi"]
        wm_res = (held.get("wm_resolved") is True)
        ans_differs = bool(held.get("answer") != novisi.get("answer"))
        role_off_content = (held.get("repair_role") != held.get("content_role"))
        visible = bool(grown and wm_res and ans_differs and role_off_content)
        hollow = bool(wm_res and not ans_differs)          # internally driven yet invisible = the #94 hollow pattern
        inert = bool((not grown) and (held.get("wm_resolved") is not True) and (not ans_differs))
        # lesion: the held read must revert to the content answer (no WM resolution surviving the lesion).
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
        if s == SHIPPED_SEED and visible and reverts:
            shipped_visible = True
        # per-seed correctness: grown -> visible+reverts ; ctrl -> inert ; never hollow.
        seed_rec["seed_ok"] = bool((visible and reverts) if grown else inert) and (not hollow)
        b_seeds.append(seed_rec)

    # LESION ATTRIBUTION (tools.lab): whose is the visible-flip? Subtract the two MEASURED arms -- the count of
    # visible decision-flips with the cross-edge ON (treatment) vs the count surviving the lesion (control). A clean
    # drive lives entirely in the treatment arm (control ~0 -> ~100% attributable); a flip present in both arms is
    # something else running in both. Mirrors the findings' max_abs_dNet_intact vs _lesioned attribution.
    try:
        from tools.lab import attributable_to
        b_attribution = attributable_to("xedge visible decision-flip: cross-edge ON vs lesioned",
                                        float(n_visible), float(n_flips_lesion))
    except Exception as _ae:
        b_attribution = None
        print(f"[aggregate] attribution call failed: {type(_ae).__name__}: {_ae}", flush=True)

    b_pass = bool(shipped_visible and (n_hollow == 0) and all_revert and (n_flips_lesion == 0)
                  and all(r.get("seed_ok") for r in b_seeds))

    # ── ARM C: no-regression on the shipped per-turn config + LEARN wiring live ──
    c = per.get("C_on_shipped_s42")
    c_items, c_diffs, moat_ok, learn_live = [], [], None, None
    if c and off:
        for m in WELL_ITEMS:
            eq = _decisions_equal(c["well"][m], off["well"][m]); c_items.append(eq)
            if not eq:
                c_diffs.append({"item": m, "on": c["well"][m], "off": off["well"][m]})
        eqq = _decisions_equal(c["question"], off["question"]); c_items.append(eqq)
        if not eqq:
            c_diffs.append({"item": QUESTION, "on": c["question"], "off": off["question"]})
        moat = c.get("moat_well_held") or {}
        moat_ok = bool(moat.get("comprehended") is True and not moat.get("abstained"))
        learn_live = bool((c.get("pool") or {}).get("learned") and (c.get("pool") or {}).get("live_per_turn"))
    c_pass = bool(c_items) and all(c_items) and bool(moat_ok) and bool(learn_live)

    go = bool(a_pass and b_pass and c_pass)
    return {
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


def _orchestrate(out_path: str, seeds: list, b_edge: str, mem_floor: int, smoke: bool) -> int:
    raw_dir = os.path.dirname(out_path)
    os.makedirs(raw_dir, exist_ok=True)
    # the (config, seed) matrix.
    jobs = [("A_baseline", 42), ("A_off", 42), ("C_on_shipped", 42)]
    for s in seeds:
        jobs.append((f"B_on_{b_edge}", s))
        jobs.append((f"B_lesion_{b_edge}", s))
    if smoke:   # mechanics only: 1 seed, ARM A + one ARM B on/lesion pair
        jobs = [("A_baseline", 42), ("A_off", 42), (f"B_on_{b_edge}", 42), (f"B_lesion_{b_edge}", 42)]
        seeds = [42]

    per, problems = {}, []
    for cfg, s in jobs:
        rec = _spawn(cfg, s, raw_dir, mem_floor)
        if "_abort" in rec:
            print(f"[orch] ABORT: {rec['_abort']}", flush=True)
            return 2
        if "_failed" in rec:
            problems.append(rec["_failed"])
            print(f"[orch] WORKER FAILED: {rec['_failed']}", flush=True)
            continue
        per[f"{cfg}_s{s}"] = rec

    agg = _aggregate(per, seeds, b_edge)
    payload = {
        "probe": "onebrain_xedge_production_default_flip_real_handler",
        "backend": os.environ.get("SIM_BACKEND", "numpy"), "brain": "tiny-demo", "renderer": "stub",
        "smoke": bool(smoke), "b_edge": b_edge, "seeds": seeds, "shipped_seed": SHIPPED_SEED,
        "flip_target": "BRAIN_ONEBRAIN_XEDGE + BRAIN_ONEBRAIN_XEDGE_LEARN default ON; =0 escape hatch preserved",
        "worker_problems": problems, "aggregate": agg, "per_worker": per,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print("\n===== FLIP-VERIFY VERDICT (real /api/brain-chat handler) =====", flush=True)
    for k in ("arm_A_byte_identical_off", "arm_B_visible_on_real_traffic", "arm_C_no_regression"):
        print(f"  [{'PASS' if agg[k]['pass'] else 'FAIL'}] {k}", flush=True)
    b = agg["arm_B_visible_on_real_traffic"]
    print(f"  ARM B: shipped_seed42_visible={b['shipped_seed42_visible']} "
          f"visible_grown={b['n_visible_grown_focus']} inert_ctrl={b['n_correctly_inert_ctrl_focus']} "
          f"hollow={b['n_hollow']} all_revert={b['all_seeds_lesion_revert']}", flush=True)
    if problems:
        print(f"  worker_problems: {problems}", flush=True)
    print(f"\n  FLIP_VERIFY_GO = {agg['FLIP_VERIFY_GO']}   wrote {out_path}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None, help="internal: run one (config) worker")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--b-edge", default="learn", choices=["learn", "frozen"],
                    help="ARM B edge: 'learn' (both flags, converged in-brain edge) or 'frozen' (BRAIN_ONEBRAIN_XEDGE only)")
    ap.add_argument("--mem-floor", type=int, default=6000, help="abort if MemAvailable drops below this (MB)")
    ap.add_argument("--smoke", action="store_true", help="mechanics-only: 1 seed, ARM A + one ARM B pair")
    args = ap.parse_args()
    if args.worker:
        return _run_worker(args.worker, args.seed, args.out)
    return _orchestrate(args.out, ALL_SEEDS, args.b_edge, args.mem_floor, args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
