# Capability: Memory strength and forgetting
# Record: brain/research/2026-11-27-fable-review-dg-port.md
# Expected-fail: partly. Replay draws follow effective strength, so replay-strengthening is rich-get-richer in expectation and
#   the ratio-preserving homeostat cannot undo it; lifting the store to the set-point can raise false fires. The counterfactual
#   store keeps the life's banked (S0) fetch history, which a life with the switch on would change. Fetch level plus an
#   oracle-validated reply projection only; never a reply claim.
"""Offline screen of reachable strength (brain/replay_strength.py, Brain(replay_strengthen=, strength_setpoint=)) on the KR
night snapshots. DRAFT prereg brain/research/2026-11-27-reachable-strength-prereg-DRAFT.md. GATED on the memory reply ceiling
verdict (brain/research/2026-11-27-memory-ceiling-prereg.md): `score` reads no bar unless that verdict is HEADROOM.

Stage `sims` (pool CPU, guarded; one cue read per battery row): per night snapshot, the brain's own store is opened
(routing_replay.open_snapshot, no deltas), the banked intact-frozen battery cues are read once, and the FULL raw similarity
vector over the store is kept per row, with the set of stored episodes that are the row item's own telling, and arm A's
decision (void rule as routing_replay: A must reproduce the banked fetch idx and text and its rightness). Also writes the
strength ledger of every snapshot (episodes.json scalars, meta sleeps / last_sleep, keep-bank size) and the night's sleep
settings (steps, sleep seed, generic text count) read from the opened Brain.

Stage `screen` (host arithmetic, seconds): the store's strengths are rebuilt night by night from the ledger under each arm
(the waking fetch events are the banked ones: uses increments between snapshots, at the snapshot's last_used), the nights run
brain/replay_strength.night_update with sleep.py's own draw sequence, and every row's decision is re-made from the banked
similarity vector with the arm's strengths through Hippocampus._compete (routing_replay.decide, FREEZE on):
  A      banked snapshot strengths (the life as is)
  NULL   the rebuild with no method: must equal the snapshot strengths and A's decision on every row (tripwire)
  RS     replay-strengthen (REPLAY_BOOST, decay clock restarts) + homeostatic scaling to SET_POINT     [the method]
  SCO    scaling only (no replay strengthening)                                                         [control]
  SHAM   NULL's effective strengths + RS's per-episode increment, the increments permuted (SHAM_DRAWS)   [no-contrast]
  FBL    RS with the waking fetch boost lesioned (a fetch changes no strength and no decay clock)         [lesion]
  report-only: RS-noreset, RS-b0.0625, RS-b0.25, RS-sp0.5, RS-sp2.0, NULL-nofetch (FBL's own no-fetch baseline)
`screen --tripwire-only` runs NULL and A alone (the pre-registration check); the full arm set refuses unless the memory
ceiling verdict is HEADROOM. Engagement gates (G1-G4) make an inert arm's verdict UNDEFINED.

Stage `score`: pooled counts and the pass rule fixed before data (verdict()); oracle-validated replies use the memory
ceiling's ORACLE reply units (u_s*_n*_ORACLE.json): a told row counts when the arm fetches a right episode AND the ORACLE reply
on that row was strict-ok.

    POOL_NODE=1 python -m brain.experiments.strength_screen sims --life L.json --snaps DIR --model M --out-dir O [--nights 5-15]
    python -m brain.experiments.strength_screen screen --dir O --seed S --tripwire-only          (before the gate: NULL + A)
    python -m brain.experiments.strength_screen screen --dir O --seed S --ceiling-verdict V      (all arms; needs HEADROOM)
    python -m brain.experiments.strength_screen score --dir O --seeds 319 320 321 --ceiling-dir C/mc_s* --ceiling-verdict V [--out F]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil

import torch

from brain.experiments import memory_ceiling as MC
from brain.experiments import routing_replay as RR
from brain.hippocampus import Episode, Hippocampus
from brain import replay_strength as RSM

STEP_HOURS = 14            # = brain.eval.life.STEP_HOURS (the night at 23:00); asserted equal in `sims`
SHAM_DRAWS = 20
SHAM_SEED = 20261127
ARM_CFG = {
    "NULL": {"fetch": True, "boost": None, "setpoint": None, "reset": True},
    "RS": {"fetch": True, "boost": RSM.REPLAY_BOOST, "setpoint": RSM.SET_POINT, "reset": True},
    "SCO": {"fetch": True, "boost": None, "setpoint": RSM.SET_POINT, "reset": True},
    "FBL": {"fetch": False, "boost": RSM.REPLAY_BOOST, "setpoint": RSM.SET_POINT, "reset": True},
    "RS-noreset": {"fetch": True, "boost": RSM.REPLAY_BOOST, "setpoint": RSM.SET_POINT, "reset": False},
    "RS-b0.0625": {"fetch": True, "boost": 0.0625, "setpoint": RSM.SET_POINT, "reset": True},
    "RS-b0.25": {"fetch": True, "boost": 0.25, "setpoint": RSM.SET_POINT, "reset": True},
    "RS-sp0.5": {"fetch": True, "boost": RSM.REPLAY_BOOST, "setpoint": 0.5, "reset": True},
    "RS-sp2.0": {"fetch": True, "boost": RSM.REPLAY_BOOST, "setpoint": 2.0, "reset": True},
    "NULL-nofetch": {"fetch": False, "boost": None, "setpoint": None, "reset": True},   # FBL's own no-fetch baseline
}
CONTROLS = ("RS", "SCO", "SHAM", "FBL")    # a missing one makes the verdict INCONCLUSIVE
REPORT_ONLY = ("RS-noreset", "RS-b0.0625", "RS-b0.25", "RS-sp0.5", "RS-sp2.0", "NULL-nofetch")
TRIPWIRE_ARMS = ("NULL",)                  # `screen --tripwire-only`: NULL and A only, no method or control arm

# Pass rule (DRAFT prereg section 5), fixed before data.
GAIN_FRAC, GAIN_MIN = 0.25, 10      # P1 right fetches >= max(ceil(1.25 * A), A + 10), pooled
DIRECTION_SEEDS, FALL_MAX = 2, 1    # P2 right rises on >= 2 of 3 seeds and falls by > 1 on none
OV_FRAC, OV_MIN = 0.25, 8           # P3 oracle-validated replies >= A's + max(8, ceil(0.25 * A's))
FIRE_FRAC = 0.5                     # P4 fires on fire rows <= floor(0.5 * A's), pooled
NEW_WRONG_MAX_SEED = 1              # P5 rows A fetched right that RS turns into a wrong fetch, per seed
C1_SCO = 5                          # C1 RS right - SCO right >= 5 (replay strengthening adds to scaling)
C2_SHAM = 10                        # C2 RS right - SHAM right (mean of draws) >= 10 (the per-episode assignment matters)
C3_FBL_SHARE = 0.5                  # C3 FBL right - A right >= 0.5 * (RS right - A right) (not riding the banked fetch history)
P1B_NONH = 6                        # P1b RS right - A right on the non-H told rows >= 6 (the ceiling's non-H bar)
#                                     P1c RS gains in more (seed, item) clusters than it loses in (ties dropped)
REGISTERED_SEEDS = (319, 320, 321)  # the KR lives; any other seed set is INCONCLUSIVE
# Engagement gates (review 11-27, B1): a failing gate makes the verdict UNDEFINED, never KILLED or NOT-ATTRIBUTED.
ENG_RS_DEC = 10          # G1 RS decisions != NULL on >= 10 rows pooled (= GAIN_MIN: fewer cannot reach P1's minimum gain)
ENG_CLAMP_SHARE = 0.25   # G1 RS's g clamped on <= 1/4 of its slept nights pooled (else the homeostat sits at its range)
ENG_EPS = 0.05           # G2 SCO's |g - 1| > 0.05 on >= half of its slept nights pooled (scaling actually scaled)
ENG_SCO_DEC = 1          # G2 SCO decisions != NULL on >= 1 row (a larger bar would turn a true "scaling does nothing" UNDEFINED)
ENG_FLOOR_SHARE = 0.05   # G3 sham_floored <= 5% of the sham's episode slots; FBL's store differs from RS's; SHAM != NULL
#                          G4 RS's pooled median tanh(eff) of the told row's best right episode > NULL's (the blamed quantity)
TOL = 1e-6


class ReconstructionError(RuntimeError):
    pass


# ------------------------------------------------------------------ guard / io
def guard(env=None):
    env = os.environ if env is None else env
    if env.get("POOL_NODE") != "1" and env.get("BRAIN_ALLOW_W3090") != "1":
        raise SystemExit("strength_screen sims: refuses to run unless POOL_NODE=1 (pool node) or BRAIN_ALLOW_W3090=1")


provenance = MC.provenance
atomic_write = MC.atomic_write


def day_start(d: int, weekend: bool) -> float:
    from brain.eval import life
    return life.day_start(d, weekend)


def night_time(k: int, weekend: bool) -> float:
    return day_start(k, weekend) + STEP_HOURS * 3600


# ------------------------------------------------------------------ stage sims (pool node)
def n_generic_of(b, snap: str) -> int:
    from brain import brain as bb
    if b._keep_bank_mode is None:
        return len(bb.ORDINARY)
    with open(os.path.join(snap, bb.KEEP_BANK_FILE)) as fh:
        return min(bb.KEEP_BANK_NIGHT, len(json.load(fh)["texts"]))


def ledger_of(snaps: str) -> dict:
    """Scalars of every night snapshot n<k> under snaps (episodes.json rows, meta sleeps / last_sleep / born)."""
    out = {}
    for d in sorted(os.listdir(snaps)):
        m = re.fullmatch(r"n(\d+)", d)
        if not m:
            continue
        p = os.path.join(snaps, d)
        meta = json.load(open(os.path.join(p, "meta.json")))
        out[m.group(1)] = {"rows": [{k: r.get(k) for k in ("file", "strength", "created", "last_used", "uses", "permanent")}
                                    for r in RR._episode_rows(p)],
                           "meta": {k: meta.get(k) for k in ("sleeps", "last_sleep", "born")}}
    return out


def sims_night(res: dict, night: int, snap: str, cortex, work: str, weekend: bool) -> dict:
    from brain import clock
    from brain.eval import life
    from brain.brain import _is_telling
    if life.STEP_HOURS != STEP_HOURS:
        raise RuntimeError("life.STEP_HOURS changed; strength_screen.STEP_HOURS must follow it")
    LB = life.use_split(res.get("split", "dev"))
    ledger, seed = res["ledger"], res["seed"]
    rows_b = RR.banked_rows(res, night)
    for r in rows_b:
        if _is_telling(r["you"]):
            raise ValueError(f"night {night}: battery cue {r['you']!r} is a telling")
    sim = life.SimClock()
    clock.set(sim.now)
    home = os.path.join(work, f"sims_s{seed}_n{night}")
    try:
        sim.to(life.day_start(night + 1, weekend))
        life._reseed(seed, 500 + night)
        b = RR.open_snapshot(snap, home, cortex, res["brain_kw"])
        try:
            if b._replay_priority is not None:
                raise SystemExit("strength_screen: replay_priority lives need the cortex for replay weights (not supported)")
            hc = b.hippocampus
            if type(hc)._compete is not Hippocampus._compete or type(hc).effective_strength is not Hippocampus.effective_strength:
                raise SystemExit(f"strength_screen: {type(hc).__name__} overrides the competition or decay (not supported)")
            texts = [e.text for e in hc.episodes]
            files = [r["file"] for r in RR._episode_rows(snap)]
            now = clock.now()
            out, void = [], []
            for i, r in enumerate(rows_b):
                cue = r["you"].strip()
                sims = hc.similarities(b.cortex, b.base, cue)
                d = RR.decide(hc, b.cortex, b.base, cue, now=now, sims=sims)
                cls = RR.row_class(r, ledger)
                rec = {"i": i, "cls": cls, "kind": r["kind"], "item": r.get("item"), "cue": cue,
                       "sims": [float(x) for x in sims], "a_idx": d["A"]["idx"], "banked_idx": (r.get("fetch") or {}).get("idx"),
                       "right_set": None}
                if cls == "told":
                    it = LB.ALL[r["item"]]
                    rec["right_set"] = [j for j, t in enumerate(texts) if life.episode_is_source(it, t)]
                    a_right = d["A"]["idx"] is not None and d["A"]["idx"] in rec["right_set"]
                    if a_right != bool(r.get("right_episode")):
                        void.append(i)
                if d["A"]["idx"] != rec["banked_idx"] or (texts[d["A"]["idx"]] if d["A"]["idx"] is not None else None) \
                        != r.get("recalled"):
                    void.append(i)
                out.append(rec)
            hcfg = {k: getattr(hc, k) for k in ("FAMILIARITY", "FAMILIARITY_SPONTANEOUS", "MARGIN", "TAU_DAYS", "DECAY",
                                                "USE_BOOST")}
            hcfg["class"] = type(hc).__name__
            sleep = {"steps": b.sleep_cfg["steps"], "sleep_seed": b.sleep_seed, "n_generic": n_generic_of(b, snap),
                     "sleep_extra": len(b.sleep_extra), "use_gate": b._use_gate}
        finally:
            b.release()
        return {"seed": seed, "night": night, "store": len(texts), "files": files, "t_battery": now, "weekend": weekend,
                "rows": out, "void_rows": sorted(set(void)), "void": bool(void), "hc": hcfg, "sleep": sleep}
    finally:
        clock.set(None)
        shutil.rmtree(home, ignore_errors=True)


def run_sims(res: dict, snaps: str, cortex, out_dir: str, weekend: bool, nights=RR.NIGHTS, log=print) -> None:
    """Per-night atomic units sims_s<seed>_n<night>.json (resume skips finished ones; a unit of another commit stops)."""
    seed = res["seed"]
    prov = provenance(getattr(cortex, "device", "?"))
    work = os.path.join(out_dir, f"work_s{seed}")
    os.makedirs(work, exist_ok=True)
    lp = os.path.join(out_dir, f"ledger_s{seed}.json")
    if os.path.exists(lp):
        old = json.load(open(lp))
        if old.get("commit") != prov["commit"] or old.get("weekend") != weekend:
            raise SystemExit(f"{lp}: written under another commit or weekend setting; use a fresh out dir (no mixing)")
    else:
        atomic_write(lp, {"seed": seed, "weekend": weekend, "snapshots": ledger_of(snaps), **prov})
    for n in nights:
        snap = os.path.join(snaps, f"n{n}")
        if not os.path.isdir(snap):
            log(f"s{seed} n{n}: missing snapshot, skipped")
            continue
        up = os.path.join(out_dir, f"sims_s{seed}_n{n}.json")
        if os.path.exists(up):
            if json.load(open(up)).get("commit") != prov["commit"]:
                raise SystemExit(f"{up}: written under another commit; use a fresh out dir (no mixing)")
            continue
        u = sims_night(res, n, snap, cortex, work, weekend)
        u.update(prov)
        atomic_write(up, u)
        log(f"s{seed} n{n}: store {u['store']} void rows {u['void_rows']}")
    shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------ stage screen: the counterfactual store
def history(ledger: dict, use_boost: float, weekend: bool) -> dict:
    """Episodes (salience, created, permanent, initial clock), the waking fetch events and the slept nights, from the
    snapshot scalars. Raises ReconstructionError when the store is not append-only or strength != salience + boosts."""
    snaps = ledger["snapshots"]
    nights = sorted(int(n) for n in snaps)
    if not nights:
        raise ReconstructionError("no snapshots")
    files = [r["file"] for r in snaps[str(nights[-1])]["rows"]]
    eps, events, prev_u, prev_n, approx = {}, [], {}, None, 0
    for n in nights:
        rows = snaps[str(n)]["rows"]
        if [r["file"] for r in rows] != files[:len(rows)]:
            raise ReconstructionError(f"snapshot n{n} is not a prefix of the last snapshot (store not append-only)")
        for r in rows:
            f, u = r["file"], int(r["uses"])
            sigma = float(r["strength"]) - use_boost * u
            if f not in eps:
                eps[f] = {"sigma": sigma, "created": float(r["created"]), "permanent": bool(r["permanent"]),
                          "lu0": float(r["last_used"]) if u == 0 else float(r["created"])}
            elif abs(eps[f]["sigma"] - sigma) > TOL:
                raise ReconstructionError(f"{f}: strength - USE_BOOST * uses moves between snapshots (not salience + boosts)")
            du = u - prev_u.get(f, 0)
            if du < 0:
                raise ReconstructionError(f"{f}: uses fell between snapshots")
            if du > 0:
                gap = (prev_n is None and n > 1) or (prev_n is not None and n - prev_n > 1)
                approx += gap
                events.append({"file": f, "count": du, "t": float(r["last_used"]), "day": n, "approx": gap})
            prev_u[f] = u
        prev_n = n
    created = {f: e["created"] for f, e in eps.items()}
    slept, sleeps_before = {}, {}
    count = 0
    for k in range(1, nights[-1] + 1):
        lo, hi = night_time(k - 1, weekend) if k > 1 else -math.inf, night_time(k, weekend)
        fresh = [f for f in files if lo < created[f] <= hi and not eps[f]["permanent"]]
        slept[k] = bool(fresh)
        sleeps_before[k] = count
        count += slept[k]
        m = snaps.get(str(k), {}).get("meta", {})
        if m.get("sleeps") is not None and m["sleeps"] != count:
            raise ReconstructionError(f"night {k}: rebuilt sleeps {count} != snapshot meta sleeps {m['sleeps']}")
    return {"files": files, "eps": eps, "events": events, "slept": slept, "sleeps_before": sleeps_before,
            "nights": nights, "approx_events": approx, "weekend": weekend}


def _hc(hcfg: dict) -> Hippocampus:
    hc = Hippocampus()
    for k in ("FAMILIARITY", "FAMILIARITY_SPONTANEOUS", "MARGIN", "TAU_DAYS", "DECAY", "USE_BOOST"):
        setattr(hc, k, hcfg[k])
    return hc


def simulate(hist: dict, cfg: dict, hcfg: dict, sleep: dict, batteries: dict, log: list | None = None) -> dict:
    """Rebuild the store under one arm. batteries: {night: battery time}. Returns {night: [Episode]} (store order).
    log: if a list, each slept night's engagement record is appended (g, g_raw, clamped, mean_eff_before, the number of
    episodes replayed at least once and of episodes whose strength or clock the night changed)."""
    hc = _hc(hcfg)
    files, eps, weekend = hist["files"], hist["eps"], hist["weekend"]
    state, order = {}, []
    by_day = {}
    for ev in hist["events"]:
        by_day.setdefault(ev["day"], []).append(ev)
    out = {}
    last = max(max(batteries, default=0), max(hist["slept"], default=0))
    for k in range(1, last + 1):
        T = night_time(k, weekend)
        for f in files:
            if f not in state and eps[f]["created"] <= T:
                e = eps[f]
                state[f] = Episode(text="", delta=None, key=None, strength=e["sigma"], created=e["created"],
                                   last_used=e["lu0"], permanent=e["permanent"])
                order.append(f)
        for ev in by_day.get(k, []):
            if cfg["fetch"]:
                ep = state[ev["file"]]
                ep.strength += hc.USE_BOOST * ev["count"]
                ep.last_used = max(ep.last_used, ev["t"])
        if hist["slept"].get(k) and (cfg["boost"] is not None or cfg["setpoint"] is not None):
            night_eps = [state[f] for f in order]
            w = [hc.effective_strength(e, T) for e in night_eps]
            if sleep.get("sleep_extra"):
                raise ReconstructionError("sleep_extra texts take draws; not supported offline")
            before = [(e.strength, e.last_used) for e in night_eps]
            rec = RSM.night_update(night_eps, w, sleep["n_generic"], sleep["steps"],
                                   sleep["sleep_seed"] + hist["sleeps_before"][k], T, hc.effective_strength,
                                   boost=cfg["boost"], setpoint=cfg["setpoint"], reset_clock=cfg["reset"])
            if log is not None:
                live = [not e.permanent for e in night_eps]
                log.append({"night": k, "g": rec.get("g"), "g_raw": rec.get("g_raw"), "clamped": bool(rec.get("clamped")),
                            "mean_eff_before": rec.get("mean_eff_before"), "n_live": sum(live),
                            "counted": sum(c > 0 for c, lv in zip(rec.get("counts") or [], live) if lv),
                            "changed": sum(lv and (abs(e.strength - b[0]) > TOL or e.last_used != b[1])
                                           for e, b, lv in zip(night_eps, before, live))})
        if k in batteries:
            out[k] = [Episode(text="", delta=None, key=None, strength=state[f].strength, created=state[f].created,
                              last_used=state[f].last_used, permanent=state[f].permanent) for f in order]
    return out


def decide_with(hcfg: dict, store: list, row: dict, now: float):
    hc = _hc(hcfg)
    hc.episodes = store
    hc.FREEZE_USE = True
    d = RR.decide(hc, None, None, row["cue"], now=now, sims=torch.tensor(row["sims"]))
    return d["A"]["idx"]


def _sham_store(store: list, null_store: list, now: float, hc: Hippocampus, rng: random.Random, count=None) -> list:
    """Increment sham: NULL's effective strengths plus RS's per-episode increment (eff_RS - eff_NULL) permuted among the
    non-permanent episodes (age 0, so strength = effective strength). The store's own structure (salience, recency, banked
    fetch boosts) stays where it is; only the method's add-on lands on the wrong episodes. Mean and the multiset of the
    increments are kept. A sum below 0 is floored at 0 (a strength is never negative); `count["floored"]` counts these."""
    if len(store) != len(null_store):
        raise ReconstructionError("RS and NULL stores differ in length")
    idx = [j for j, e in enumerate(store) if not e.permanent]
    if any(null_store[j].permanent for j in idx):
        raise ReconstructionError("RS and NULL disagree on permanence")
    base = [hc.effective_strength(null_store[j], now) for j in idx]
    delta = [hc.effective_strength(store[j], now) - b for j, b in zip(idx, base)]
    rng.shuffle(delta)
    out, floored = list(store), 0
    for j, b, d in zip(idx, base, delta):
        e = store[j]
        v = b + d
        if v < 0:
            v, floored = 0.0, floored + 1
        out[j] = Episode(text="", delta=None, key=None, strength=v, created=e.created, last_used=now, permanent=False)
    if count is not None:
        count["floored"] = count.get("floored", 0) + floored
        count["slots"] = count.get("slots", 0) + len(idx)
    return out


def _right(k, row):
    return k is not None and row["right_set"] is not None and k in row["right_set"]


def _median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else None


def screen_seed(ledger: dict, units: list, arms=tuple(ARM_CFG)) -> dict:
    """All arms for one seed. units: sims_s*_n*.json records of that seed. Per battery night it also keeps the engagement
    record (review 11-27, B1): decisions that differ from NULL (RS, SCO, SHAM) and from RS (FBL), FBL's store against RS's,
    the sham's floored share, and tanh(eff) of each told row's best right episode under NULL and RS; per arm and slept
    night, the night rule's record (g, clamp, replayed and changed episodes)."""
    units = sorted(units, key=lambda u: u["night"])
    hcfg, sleep = units[0]["hc"], units[0]["sleep"]
    for u in units:
        if u["hc"] != hcfg or u["sleep"] != sleep:
            raise ReconstructionError("hippocampus constants or sleep settings differ between nights")
    if sleep.get("use_gate") is not None:
        raise ReconstructionError("use_gate lives (a fetch may not boost) are not supported by the rebuild")
    if "NULL" not in arms:
        raise ReconstructionError("every screen needs NULL (the tripwire)")
    hist = history(ledger, hcfg["USE_BOOST"], ledger.get("weekend", True))
    batt = {u["night"]: u["t_battery"] for u in units}
    night_logs = {a: [] for a in arms}
    stores = {a: simulate(hist, ARM_CFG[a], hcfg, sleep, batt, night_logs[a]) for a in arms}
    seed = units[0]["seed"]
    nights, state_mismatch, null_mismatch = [], 0, 0
    sham_total = {"floored": 0, "slots": 0}
    snaps = ledger["snapshots"]
    for u in units:
        n, t = u["night"], u["t_battery"]
        if u["files"] != hist["files"][:len(u["files"])] or len(stores["NULL"][n]) != len(u["files"]):
            raise ReconstructionError(f"night {n}: the sims store is not the ledger's store")
        snap_rows = snaps.get(str(n), {}).get("rows")
        if snap_rows is not None:
            for e, r in zip(stores["NULL"][n], snap_rows):
                if abs(e.strength - r["strength"]) > TOL or abs(e.last_used - r["last_used"]) > 1.0:
                    state_mismatch += 1
        a_store = [Episode(text="", delta=None, key=None, strength=r["strength"], created=r["created"],
                           last_used=r["last_used"], permanent=r["permanent"]) for r in snap_rows] if snap_rows else None
        hc = _hc(hcfg)
        sham_count = {"floored": 0, "slots": 0}
        rows = []
        for row in u["rows"]:
            rec = {"i": row["i"], "cls": row["cls"], "kind": row["kind"], "item": row["item"], "cue": row.get("cue"),
                   "right_set": row["right_set"], "A": row["a_idx"]}
            for a in arms:
                rec[a] = decide_with(hcfg, stores[a][n], row, t)
            if a_store is not None and decide_with(hcfg, a_store, row, t) != row["a_idx"]:
                null_mismatch += 1
            if rec["NULL"] != row["a_idx"]:
                null_mismatch += 1
            if "RS" in arms:
                rng = random.Random(SHAM_SEED + 1000 * int(seed) + 10 * n + row["i"])
                draws = [decide_with(hcfg, _sham_store(stores["RS"][n], stores["NULL"][n], t, hc, rng, sham_count), row, t)
                         for _ in range(SHAM_DRAWS)]
                rec["SHAM"] = {"right": sum(_right(k, row) for k in draws) / SHAM_DRAWS,
                               "fire": sum(k is not None for k in draws) / SHAM_DRAWS,
                               "wrong": sum(k is not None and not _right(k, row) for k in draws) / SHAM_DRAWS,
                               "new_wrong_on_A_right": (sum(k is not None and not _right(k, row) for k in draws) / SHAM_DRAWS)
                               if _right(row["a_idx"], row) else 0.0,
                               "none_on_A_right": (sum(k is None for k in draws) / SHAM_DRAWS)
                               if _right(row["a_idx"], row) else 0.0,
                               "diff_null": sum(k != rec["NULL"] for k in draws) / SHAM_DRAWS}
            rows.append(rec)
        sham_total["floored"] += sham_count["floored"]
        sham_total["slots"] += sham_count["slots"]
        tanh_right, med = {}, {}
        for a in ("NULL", "RS"):       # the screen-2 diagnostic: tanh(effective strength) of the best right episode
            if a in arms:
                tanh_right[a] = [math.tanh(hc.effective_strength(stores[a][n][max(row["right_set"], key=lambda j: row["sims"][j])], t))
                                 for row in u["rows"] if row["right_set"]]
                med[a] = _median(tanh_right[a])
        eng = {"sham_floored": sham_count["floored"], "sham_slots": sham_count["slots"], "tanh_right": tanh_right}
        if "FBL" in arms and "RS" in arms:
            eng["fbl_store_vs_rs"] = sum(not x.permanent and (abs(x.strength - y.strength) > TOL or x.last_used != y.last_used)
                                         for x, y in zip(stores["FBL"][n], stores["RS"][n]))
        nights.append({"night": n, "void": u["void"], "void_rows": u["void_rows"], "rows": rows, "median_tanh_right": med,
                       "engagement": eng})
    return {"seed": seed, "arms": list(arms), "nights": nights, "state_mismatch": state_mismatch,
            "null_mismatch": null_mismatch, "approx_events": hist["approx_events"], "fetch_events": len(hist["events"]),
            "sham_floored": sham_total["floored"], "sham_slots": sham_total["slots"], "night_records": night_logs,
            "arm_cfg": {a: ARM_CFG[a] for a in arms},
            "constants": {"REPLAY_BOOST": RSM.REPLAY_BOOST, "SET_POINT": RSM.SET_POINT, "SCALE": [RSM.SCALE_MIN, RSM.SCALE_MAX],
                          "SHAM_DRAWS": SHAM_DRAWS}}


def input_sha(paths: list) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def _headroom(status) -> bool:
    return bool(status) and str(status).startswith("HEADROOM")


def run_screen(d: str, seed: int, ceiling_status: str | None = None, tripwire_only: bool = False) -> dict:
    """tripwire_only: NULL and A only (trip_s<seed>.json: the tripwire counters, no row, no method or control arm); this is
    the only screen allowed before registration. The full arm set (scr_s<seed>.json) refuses to run unless the memory
    ceiling verdict is HEADROOM (review 11-27, B2: no arm's counts exist on disk before the gate opens)."""
    if not tripwire_only and not _headroom(ceiling_status):
        raise SystemExit(f"screen s{seed}: the full arm set is GATED (memory ceiling verdict {ceiling_status!r}, not HEADROOM);"
                         " only --tripwire-only may run")
    lp = os.path.join(d, f"ledger_s{seed}.json")
    ups = sorted(os.path.join(d, f) for f in os.listdir(d) if re.fullmatch(rf"sims_s{seed}_n\d+\.json", f))
    if not os.path.exists(lp) or not ups:
        raise SystemExit(f"screen s{seed}: ledger or sims units missing in {d}")
    sha = input_sha([lp] + ups)
    prov = provenance("cpu (host arithmetic)")
    op = os.path.join(d, f"{'trip' if tripwire_only else 'scr'}_s{seed}.json")
    if os.path.exists(op):
        old = json.load(open(op))
        if old.get("input_sha") == sha and old.get("commit") == prov["commit"]:
            return old
    full = screen_seed(json.load(open(lp)), [json.load(open(p)) for p in ups],
                       arms=TRIPWIRE_ARMS if tripwire_only else tuple(ARM_CFG))
    if tripwire_only:
        out = {k: full[k] for k in ("seed", "arms", "null_mismatch", "state_mismatch", "approx_events", "fetch_events")}
        out["nights"] = [{"night": n["night"], "void": n["void"], "void_rows": n["void_rows"]} for n in full["nights"]]
        out["tripwire_only"] = True
    else:
        out = full
    out.update(prov)
    out["input_sha"] = sha
    out["sims_commits"] = sorted({json.load(open(p)).get("commit") or "" for p in ups})
    atomic_write(op, out)
    return out


# ------------------------------------------------------------------ score and verdict
def _ok_map(ceiling_dir, seed, night, arm="ORACLE"):
    """ceiling_dir: one directory or a list (the ceiling writes one mc_s<seed> directory per seed). Two different files
    for the same unit is an error, not a choice. Returns {row index: {"ok", "item", "you"}} over the told rows."""
    dirs = [ceiling_dir] if isinstance(ceiling_dir, str) else list(ceiling_dir)
    hits = [p for p in (os.path.join(x, f"u_s{seed}_n{night}_{arm}.json") for x in dirs) if os.path.exists(p)]
    if not hits:
        return None
    if len({open(p, "rb").read() for p in hits}) > 1:
        raise SystemExit(f"ceiling unit s{seed} n{night} {arm}: differing copies {hits}")
    p = hits[0]
    return {i: {"ok": bool(r.get("ok")), "item": r.get("item"), "you": r.get("you")}
            for i, r in enumerate(json.load(open(p))["rows"]) if r.get("cls") == "told"}


def _check_aligned(okm: dict, rows: list, seed, night) -> None:
    """The ceiling unit's row i must be this screen's row i (review 11-27, NB5): same told row count, same item, same cue."""
    told = {r["i"]: r for r in rows if r["cls"] == "told"}
    if set(okm) != set(told):
        raise SystemExit(f"ceiling unit s{seed} n{night}: told rows {sorted(okm)} != the screen's {sorted(told)}")
    for i, o in okm.items():
        r = told[i]
        if o["item"] != r.get("item") or (o["you"] is not None and r.get("cue") is not None
                                          and str(o["you"]).strip() != str(r["cue"]).strip()):
            raise SystemExit(f"ceiling unit s{seed} n{night} row {i}: item/cue {o['item']!r}/{o['you']!r} != the screen's "
                             f"{r.get('item')!r}/{r.get('cue')!r}")


def seed_counts(scr: dict, ceiling_dir) -> dict:
    """Per arm on the non-void nights: told right / wrong / no-fetch, right on the non-H told rows, right per item (the
    (seed, item) clusters), fires on fire rows, fetches on identity rows (report only), new wrong and new no-fetch on
    A-right rows, and oracle-validated replies (right fetch AND ORACLE reply strict-ok). missing_oracle: non-void nights
    with no ORACLE unit. engagement: the B1 gates' inputs (see verdict)."""
    if not ceiling_dir:
        raise ValueError("seed_counts: ceiling_dir is empty (the oracle-validated measure needs the ceiling's units)")
    arms = ["A"] + [a for a in scr["arms"]] + (["SHAM"] if "RS" in scr["arms"] else [])
    c = {a: {"told_n": 0, "right": 0.0, "wrong": 0.0, "none": 0.0, "nonH_n": 0, "right_nonH": 0.0, "cluster_right": {},
             "fire_n": 0, "fires": 0.0, "identity_n": 0, "identity_fetches": 0.0, "new_wrong_on_A_right": 0.0,
             "new_none_on_A_right": 0.0, "ov": 0.0} for a in arms}
    eng = {"RS_dec_vs_null": 0, "SCO_dec_vs_null": 0, "FBL_dec_vs_RS": 0, "SHAM_dec_vs_null": 0.0, "fbl_store_vs_rs": 0,
           "sham_floored": 0, "sham_slots": 0, "tanh_right": {"NULL": [], "RS": []}}
    missing, nv = [], 0
    for n in scr["nights"]:
        if n["void"]:
            nv += 1
            continue
        okm = _ok_map(ceiling_dir, scr["seed"], n["night"])
        if okm is None:
            missing.append(n["night"])
        else:
            _check_aligned(okm, n["rows"], scr["seed"], n["night"])
        e = n.get("engagement") or {}
        for k in ("sham_floored", "sham_slots", "fbl_store_vs_rs"):
            eng[k] += e.get(k, 0)
        for a in ("NULL", "RS"):
            eng["tanh_right"][a] += list((e.get("tanh_right") or {}).get(a, []))
        for r in n["rows"]:
            if "RS" in r:
                eng["RS_dec_vs_null"] += r["RS"] != r["NULL"]
            if "SCO" in r:
                eng["SCO_dec_vs_null"] += r["SCO"] != r["NULL"]
            if "FBL" in r and "RS" in r:
                eng["FBL_dec_vs_RS"] += r["FBL"] != r["RS"]
            if "SHAM" in r:
                eng["SHAM_dec_vs_null"] += r["SHAM"].get("diff_null", 0.0)
            for a in arms:
                x = c[a]
                if a == "SHAM":
                    s = r["SHAM"]
                    right, fire, wrong = s["right"], s["fire"], s["wrong"]
                    nwa, nna = s["new_wrong_on_A_right"], s.get("none_on_A_right", 0.0)
                else:
                    k = r[a]
                    right, fire = float(_right(k, r)), float(k is not None)
                    wrong = float(k is not None and not _right(k, r))
                    nwa = float(_right(r["A"], r) and k is not None and not _right(k, r))
                    nna = float(_right(r["A"], r) and k is None)
                if r["cls"] == "told":
                    x["told_n"] += 1
                    x["right"] += right
                    x["wrong"] += wrong
                    x["none"] += 1 - fire
                    x["new_wrong_on_A_right"] += nwa
                    x["new_none_on_A_right"] += nna
                    if r.get("kind") != "H":
                        x["nonH_n"] += 1
                        x["right_nonH"] += right
                    it = str(r.get("item"))
                    x["cluster_right"][it] = x["cluster_right"].get(it, 0.0) + right
                    if okm is not None and okm.get(r["i"], {}).get("ok"):
                        x["ov"] += right
                elif r["cls"] == "fire":
                    x["fire_n"] += 1
                    x["fires"] += fire
                elif r["cls"] == "identity":
                    x["identity_n"] += 1
                    x["identity_fetches"] += fire
    nrec = {}
    for a, recs in (scr.get("night_records") or {}).items():
        nrec[a] = {"nights": len(recs), "clamped": sum(bool(r.get("clamped")) for r in recs),
                   "g_off1": sum(r.get("g") is not None and abs(r["g"] - 1.0) > ENG_EPS for r in recs),
                   "counted": sum(r.get("counted", 0) for r in recs), "changed": sum(r.get("changed", 0) for r in recs),
                   "g": [r.get("g") for r in recs], "mean_eff_before": [r.get("mean_eff_before") for r in recs]}
    eng["night_records"] = nrec
    eng["fetch_events"] = scr.get("fetch_events")
    return {"arms": c, "void_nights": nv, "missing_oracle_nights": missing, "null_mismatch": scr["null_mismatch"],
            "state_mismatch": scr["state_mismatch"], "approx_events": scr["approx_events"],
            "engagement": eng if "night_records" in scr else None}


def engagement_gates(live: dict) -> dict:
    """The registered engagement gates (review 11-27, B1), pooled over the live seeds. Returns {gate: (holds, why)}.
    G1 and G4 are the method's (a failure makes any verdict UNDEFINED); G2 and G3 are the controls' (a failure makes a
    verdict that reads the controls, PASS or NOT-ATTRIBUTED, UNDEFINED; a P-clause KILLED does not read them)."""
    E = [m["engagement"] for m in live.values()]
    S = lambda k: sum(e[k] for e in E)
    NR = lambda a, k: sum((e["night_records"].get(a) or {}).get(k, 0) for e in E)
    out = {}
    rs_dec, rs_n, rs_cl = S("RS_dec_vs_null"), NR("RS", "nights"), NR("RS", "clamped")
    out["G1"] = (rs_dec >= ENG_RS_DEC and rs_n > 0 and rs_cl <= ENG_CLAMP_SHARE * rs_n,
                 f"G1 RS decisions != NULL {rs_dec} (min {ENG_RS_DEC}); g clamped on {rs_cl}/{rs_n} nights "
                 f"(max {ENG_CLAMP_SHARE:g})")
    sco_dec, sco_n, sco_off = S("SCO_dec_vs_null"), NR("SCO", "nights"), NR("SCO", "g_off1")
    out["G2"] = (sco_dec >= ENG_SCO_DEC and sco_n > 0 and sco_off >= 0.5 * sco_n,
                 f"G2 SCO |g-1|>{ENG_EPS:g} on {sco_off}/{sco_n} nights (min half); SCO decisions != NULL {sco_dec} "
                 f"(min {ENG_SCO_DEC})")
    fl, sl, fbl, sh = S("sham_floored"), S("sham_slots"), S("fbl_store_vs_rs"), S("SHAM_dec_vs_null")
    out["G3"] = (fbl > 0 and sh > 0 and sl > 0 and fl <= ENG_FLOOR_SHARE * sl,
                 f"G3 FBL store != RS on {fbl} episode-nights (min 1); SHAM decisions != NULL {sh:g} (> 0); "
                 f"sham floored {fl}/{sl} (max {ENG_FLOOR_SHARE:g})")
    tn = [x for e in E for x in e["tanh_right"]["NULL"]]
    tr = [x for e in E for x in e["tanh_right"]["RS"]]
    mn, mr = _median(tn), _median(tr)
    out["G4"] = (mn is not None and mr is not None and mr > mn,
                 f"G4 median tanh(eff) of the right episode RS {mr} vs NULL {mn} (RS must be higher)")
    return out


def verdict(meas: dict, ceiling_status: str | None, seeds=REGISTERED_SEEDS) -> dict:
    """meas: {seed: seed_counts(...)}. The rule of the prereg section 5, fixed before data."""
    out = {"ceiling_status": ceiling_status, "checks": {}}
    if not _headroom(ceiling_status):
        out["status"] = f"GATED: memory reply ceiling verdict is {ceiling_status!r}, not HEADROOM; no bar is read"
        return out
    why = []
    if tuple(sorted(int(s) for s in seeds)) != REGISTERED_SEEDS:
        why.append(f"seed set {sorted(seeds)} is not the registered {list(REGISTERED_SEEDS)}")
    if set(meas) - set(REGISTERED_SEEDS):
        why.append(f"unregistered seeds measured {sorted(set(meas) - set(REGISTERED_SEEDS))}")
    live = {s: m for s, m in meas.items() if s in REGISTERED_SEEDS and m["void_nights"] <= RR.VOID_NIGHTS_MAX}
    if len(live) < len(REGISTERED_SEEDS):
        why.append(f"{len(REGISTERED_SEEDS) - len(live)} seed(s) missing or void")
    for s, m in live.items():
        if m["null_mismatch"] or m["state_mismatch"]:
            why.append(f"s{s}: rebuild tripwire (null {m['null_mismatch']}, state {m['state_mismatch']})")
        if m["missing_oracle_nights"]:
            why.append(f"s{s}: ORACLE reply units missing for nights {m['missing_oracle_nights']}")
        lacking = [a for a in ("A",) + CONTROLS if a not in m["arms"]]
        if lacking:
            why.append(f"s{s}: arms missing {lacking}")
        if not m.get("engagement"):
            why.append(f"s{s}: engagement record missing")
    if why:
        out["status"] = "INCONCLUSIVE: " + "; ".join(why)
        return out
    P = lambda a, k: sum(m["arms"][a][k] for m in live.values())
    A_r, RS_r = P("A", "right"), P("RS", "right")
    need = max(math.ceil((1 + GAIN_FRAC) * A_r), A_r + GAIN_MIN)
    A_h, RS_h = P("A", "right_nonH"), P("RS", "right_nonH")
    cl = [(s, it) for s, m in live.items() for it in sorted(set(m["arms"]["A"]["cluster_right"]) | set(m["arms"]["RS"]["cluster_right"]))]
    cd = [live[s]["arms"]["RS"]["cluster_right"].get(it, 0.0) - live[s]["arms"]["A"]["cluster_right"].get(it, 0.0) for s, it in cl]
    c_win, c_loss = sum(d > 0 for d in cd), sum(d < 0 for d in cd)
    up = sum(m["arms"]["RS"]["right"] > m["arms"]["A"]["right"] for m in live.values())
    down = [s for s, m in live.items() if m["arms"]["RS"]["right"] < m["arms"]["A"]["right"] - FALL_MAX]
    A_ov, RS_ov = P("A", "ov"), P("RS", "ov")
    ov_need = A_ov + max(OV_MIN, math.ceil(OV_FRAC * A_ov))
    A_f, RS_f = P("A", "fires"), P("RS", "fires")
    fire_max = math.floor(FIRE_FRAC * A_f)
    nw = {s: m["arms"]["RS"]["new_wrong_on_A_right"] for s, m in live.items()}
    kills, controls = [], []
    if RS_r < need:
        kills.append(f"P1 right {RS_r:g} < {need} (A {A_r:g})")
    if RS_h - A_h < P1B_NONH:
        kills.append(f"P1b non-H right gain {RS_h - A_h:g} < {P1B_NONH} (A {A_h:g}, RS {RS_h:g})")
    if c_win <= c_loss:
        kills.append(f"P1c (seed, item) clusters: RS gains in {c_win}, loses in {c_loss} (needs more gains)")
    if up < DIRECTION_SEEDS or down:
        kills.append(f"P2 right rises on {up}/{len(live)} seeds, falls by >{FALL_MAX} on {down}")
    if RS_ov < ov_need:
        kills.append(f"P3 oracle-validated replies {RS_ov:g} < {ov_need:g} (A {A_ov:g})")
    if RS_f > fire_max:
        kills.append(f"P4 fires {RS_f:g} > {fire_max} (half of A's {A_f:g})")
    if any(x > NEW_WRONG_MAX_SEED for x in nw.values()):
        kills.append(f"P5 new wrong on A-right rows {nw} (max {NEW_WRONG_MAX_SEED}/seed)")
    SCO_r, SHAM_r, FBL_r = P("SCO", "right"), P("SHAM", "right"), P("FBL", "right")
    if RS_r - SCO_r < C1_SCO:
        controls.append(f"C1 RS - SCO right {RS_r - SCO_r:g} < {C1_SCO} (scaling alone does it)")
    if RS_r - SHAM_r < C2_SHAM:
        controls.append(f"C2 RS - SHAM right {RS_r - SHAM_r:g} < {C2_SHAM} (no per-episode contrast)")
    if FBL_r - A_r < C3_FBL_SHARE * (RS_r - A_r):
        controls.append(f"C3 FBL gain {FBL_r - A_r:g} < {C3_FBL_SHARE} x RS gain {RS_r - A_r:g} (rides the banked fetch history)")
    gates = engagement_gates(live)
    out["checks"] = {"A_right": A_r, "RS_right": RS_r, "need": need, "A_right_nonH": A_h, "RS_right_nonH": RS_h,
                     "clusters": len(cl), "cluster_wins": c_win, "cluster_losses": c_loss, "up_seeds": up,
                     "down_seeds": down, "A_ov": A_ov, "RS_ov": RS_ov, "ov_need": ov_need, "A_fires": A_f, "RS_fires": RS_f,
                     "fire_max": fire_max, "new_wrong": nw,
                     "new_none_on_A_right": {s: m["arms"]["RS"]["new_none_on_A_right"] for s, m in live.items()},
                     "SCO_right": SCO_r, "SHAM_right": SHAM_r, "FBL_right": FBL_r,
                     "gates": {g: {"holds": h, "why": w} for g, (h, w) in gates.items()}}
    if all("NULL-nofetch" in m["arms"] for m in live.values()):
        out["checks"]["NULL_nofetch_right_report_only"] = P("NULL-nofetch", "right")
    out["kills"], out["control_fails"] = kills, controls
    method_bad = [w for g, (h, w) in gates.items() if g in ("G1", "G4") and not h]
    control_bad = [w for g, (h, w) in gates.items() if g in ("G2", "G3") and not h]
    out["gate_fails"] = method_bad + control_bad
    if method_bad:
        out["status"] = "UNDEFINED (the method did not engage): " + "; ".join(method_bad)
    elif kills:
        out["status"] = "KILLED: " + "; ".join(kills)
    elif control_bad:
        out["status"] = "UNDEFINED (a control did not engage): " + "; ".join(control_bad)
    elif controls:
        out["status"] = "NOT-ATTRIBUTED: " + "; ".join(controls)
    else:
        out["status"] = "PASS (offline fetch level + oracle-validated projection: earns a live DEV six-seed life)"
    return out


def score(d: str, seeds, ceiling_dir, ceiling_status: str | None) -> dict:
    """GATED returns the status alone: no file is read and no arm count leaves this function before HEADROOM."""
    if not _headroom(ceiling_status):
        return verdict({}, ceiling_status, seeds)
    if not ceiling_dir:
        raise ValueError("score: --ceiling-dir is empty (the ORACLE reply units are required)")
    meas, commits = {}, set()
    for s in seeds:
        p = os.path.join(d, f"scr_s{s}.json")
        if not os.path.exists(p):
            continue
        scr = json.load(open(p))
        commits |= {scr.get("commit")} | set(scr.get("sims_commits") or [])
        meas[s] = seed_counts(scr, ceiling_dir)
    v = verdict(meas, ceiling_status, seeds)
    if len(commits - {None, ""}) > 1:
        v["status"] = f"INCONCLUSIVE: mixed commits {sorted(c for c in commits if c)}"
    v["per_seed"] = meas
    v["report_only"] = {a: sum(m["arms"][a]["right"] for m in meas.values() if a in m["arms"]) for a in REPORT_ONLY}
    return v


def ceiling_status_of(path: str | None):
    if not path or not os.path.exists(path):
        return None
    return json.load(open(path)).get("status")


# ------------------------------------------------------------------ CLI
def main(argv=None, cortex=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("sims")
    r.add_argument("--life", required=True)
    r.add_argument("--snaps", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--model", default=None)
    r.add_argument("--device", default="cpu")
    r.add_argument("--cuda-precision", default="fp16")
    r.add_argument("--nights", default="5-15")
    r.add_argument("--no-weekend", action="store_true")
    c = sub.add_parser("screen")
    c.add_argument("--dir", required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--tripwire-only", action="store_true", help="NULL and A only (the one screen allowed before the gate)")
    c.add_argument("--ceiling-verdict", default=None, help="the memory ceiling verdict JSON (the full screen needs HEADROOM)")
    s = sub.add_parser("score")
    s.add_argument("--dir", required=True)
    s.add_argument("--seeds", nargs="+", type=int, required=True)
    s.add_argument("--ceiling-dir", nargs="+", required=True, help="one or more dirs holding u_s*_n*_ORACLE.json (mc_s*)")
    s.add_argument("--ceiling-verdict", required=True)
    s.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "sims":
        guard()
        res = json.load(open(a.life))
        if cortex is None:
            from brain.cortex import Cortex, resolve_model
            cortex = Cortex(model_path=resolve_model(a.model), device=a.device, cuda_precision=a.cuda_precision)
        lo, _, hi = a.nights.partition("-")
        run_sims(res, a.snaps, cortex, a.out_dir, not a.no_weekend, range(int(lo), int(hi or lo) + 1))
        return None
    if a.cmd == "screen":
        out = run_screen(a.dir, a.seed, ceiling_status_of(a.ceiling_verdict), tripwire_only=a.tripwire_only)
        print(json.dumps({k: out[k] for k in ("seed", "null_mismatch", "state_mismatch", "approx_events")}))
        return out
    v = score(a.dir, a.seeds, a.ceiling_dir, ceiling_status_of(a.ceiling_verdict))
    if a.out:
        atomic_write(a.out, v)
    print(json.dumps({k: v[k] for k in ("status", "checks", "report_only") if k in v}, indent=1))
    return v


if __name__ == "__main__":
    main()
