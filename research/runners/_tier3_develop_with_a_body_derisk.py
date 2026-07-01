"""Tier-3 Option 2A 'develop-with-a-body' de-risk: a brain DEVELOPS over DAYS where each day's knowledge is LIVED.

Per the scoping (research/findings/2026-06-30-tier3-option2-develop-with-a-body-scoping.md, controller-verified):
Option 2 is the recommended second Tier-3 slice, and it is cheap -- largely done in two validated GO halves. The
live-perceive-ground-store day is Option 1's `live()` loop (_tier3_live_and_remember_derisk.py, 6/6 GO); the
day-loop scaffold (WAKE->SLEEP->GROW->PERSIST->METRICS) is develop_gpu's PATTERN. The genuine residual is the
runner-only JOIN: use Option-1's `live()` as the day's WAKE so each day's knowledge is LIVED (perceived+grounded
during a foraging day) instead of a SCRIPTED corpus curriculum.

This is Option 2A: a SELF-CONTAINED runner (owns its multi-day loop reusing `live()` + the develop-loop stage
PATTERNS -> CANNOT regress the existing develop_gpu). One PERSISTENT MergedNavConvAgent lives across days (the
brain that develops); each day it forages a cumulatively-richer world (objects introduced over days), grounding +
storing the LIVED chain of what it encountered; old lived facts are RETAINED (no-forgetting) as new days add facts;
the developing brain PERSISTS across a reset. NO `sim/` edit (reuse-by-import; `live()` gained one additive
default-off `commit_facts` param for the frozen-brain control -- default True is byte-identical to the 6/6 GO).

WHY LIVED, NOT SCRIPTED (the R-b discriminator): the day's facts are a consequence of WHICH objects the agent's
foraging brings it to (the world layout + its drive-biased behaviour), NOT an authored `curriculum.day_stream`
list. The permuted-world control proves it: a different object placement -> a different encounter chain -> different
lived facts.

GATES / ANTI-CHEATS (the validated-signal-by-its-function bar; ALL must hold)
----------------------------------------------------------------------------
  (1) DEVELOPS OVER DAYS: the lived-fact count GROWS day-over-day (the brain accumulates knowledge from experience).
  (2) NO-FORGETTING / RETENTION: on the last day, ALL accumulated lived facts -- including day-0's -- are still
      recalled (>= 0.8) as new days added facts.
  (3) FROZEN-BRAIN control: an identical multi-day life with `commit_facts=False` (the brain SEES/grounds but does
      NOT store) -> competence stays FLAT (recall ~0) -- competence must NOT rise without committing what it lived.
  (4) LIVED, NOT SCRIPTED: the accumulated facts are the ENCOUNTER CHAIN; a PERMUTED-world control (a different
      object placement order) yields a DIFFERENT fact set (the memory tracks the lived layout, not a script).
  (5) CONVERSE + no-confab MOAT: an unstored (object, "chase") cue ABSTAINS (None) every day; the conversational
      synapses stay BYTE-IDENTICAL across the whole multi-day life. A moat breach is a HARD STOP.
  (6) PERSISTENCE ACROSS RESET: reload resumes the EXACT developed brain (all accumulated lived facts); a
      NO-PERSISTENCE cold-start is empty.
  (*) REWARD-PROVENANCE: survival rides the intrinsic drive-reduction (rate_proxy, no host distance term); the
      agent stays ALIVE across the multi-day life (the drive keeps energy in-band).

HONEST SCOPE (deferred; flagged): the corridor + 4-object perceivable set (the gen stack renders OBJECT_WORDS,
N=4) bounds the developed graph to a short chain (~3 lived facts over the growth days) -- a RICHER multi-day
development needs more perceivable objects / a 2D world / the pair-accumulation upgrade (follow-ons). The learned
spatial policy stays the deferred Tier-4 dendrite wall (survival uses the validated rate-proxy stand-in).
Persistence is JSON re-instate (not the raw cp_connections tensor). Promotion to the 24/7 develop_gpu harness
(Option 2B, an additive default-off per_day_agent_factory seam) is a follow-on.

Run (GPU -- the merged bridge is GPU-only):
  python -m research.runners._tier3_develop_with_a_body_derisk --smoke                       # tiny GPU mechanics
  python -m research.runners._tier3_develop_with_a_body_derisk --seeds 42 43 44 100 101 102  # full 6-seed
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# reuse-by-import the VALIDATED Option-1 machinery (the perceive-ground-store day) verbatim.
from research.runners._tier3_live_and_remember_derisk import (
    live, LiveState, SpikingHunger, _lived_recall, _encode_code, _decode_code, _build_agent,
    OBJECT_WORDS, LINK_VERB, _survival, _drive_corr_sweep, L, HEALTHY, CRASH,
)
from sim.lineage import BridgeLineage

MOAT_VERB = "chase"          # a vocab verb NEVER used as a link -> an unstored (object, "chase") cue must abstain
DEV_ORDER = ["apple", "cat", "dog", "river"]   # the fixed cumulative-introduction order (a subset of OBJECT_WORDS)


class DevWorld:
    """The day-`day` world: a corridor with a CUMULATIVELY-richer object set (day d places the first n_intro(d)
    objects of `order` at descending cells, so a walk toward food (0) encounters them IN ORDER). Introducing a new
    object each early day is how the brain's lived knowledge GROWS from experience. `order` may be PERMUTED (the
    lived-not-scripted control) -> a different encounter chain -> different lived facts."""

    def __init__(self, day, order=None, base=2):
        order = list(order if order is not None else DEV_ORDER)
        n_intro = min(base + day, len(order))              # day0->2 objects, +1 per day, capped at len(order)
        self.placed = order[:n_intro]
        # descending cells L-2, L-3, ... so foraging 5->0 hits them in `placed` order (the chain).
        cells = list(range(L - 2, 0, -1))[:n_intro]        # L=6 -> [4,3,2,1]
        self.cell_to_obj = {c: o for c, o in zip(cells, self.placed)}
        self.held_out = []                                 # moat is via the unstored (obj, MOAT_VERB) cue, not a holdout


def _dev_moat(agent, encountered):
    """The no-confab moat for the develop loop: an unstored (object, MOAT_VERB) cue MUST abstain (None) -- MOAT_VERB
    is never used as a link, so no (obj, chase) fact was ever stored. Returns (n_abstain_ok, n_total)."""
    ok, tot = 0, 0
    for obj in sorted(set(encountered)):
        tot += 1
        try:
            if agent.composer.query_patient(obj, MOAT_VERB) is None:
                ok += 1
        except Exception:
            pass
    return ok, tot


def _run_multiday(agent, hunger, state, *, n_days, steps_per_day, order=None, commit_facts=True,
                  drive_read_every=10, cache=None, track=True):
    """Live `n_days` foraging days on the PERSISTENT agent+state (continuous body + accumulating memory). Each day
    a cumulatively-richer DevWorld; `live()` forages (survival) + perceives+grounds+stores the day's lived chain.
    Returns the per-day trace (facts_known, recall, moat, minE) when track, else None."""
    cache = cache if cache is not None else set()
    per_day = []
    for d in range(n_days):
        world_d = DevWorld(d, order=order)
        seg = live(agent, hunger, state, world_d, steps_per_day, drive_reward="rate_proxy",
                   drive_read_every=drive_read_every, perceive=True, commit_facts=commit_facts,
                   grounded_obj_cache=cache)
        if not track:
            continue
        # SLEEP/METRICS pattern: retention re-test = recall of ALL accumulated lived facts (old + new); moat.
        recall_ok, recall_tot = _lived_recall(agent, state.lived_facts)
        moat_ok, moat_tot = _dev_moat(agent, state.encountered)
        surv = _survival(seg["energies"])
        per_day.append({"day": d, "facts_known": len(state.lived_facts),
                        "n_placed": len(world_d.placed), "recall_ok": recall_ok, "recall_tot": recall_tot,
                        "moat_ok": moat_ok, "moat_tot": moat_tot, "min_energy": surv["min_energy"]})
    return per_day


def run_seed(seed, root, *, n_days=4, steps_per_day=700, drive_read_every=10, drive_window=40):
    """One seed: DEVELOP a persistent merged brain over `n_days` LIVED foraging days, then the anti-cheats
    (frozen-brain / permuted-world / persistence-across-reset). Builds: 1 develop agent + 1 frozen + 1 permuted +
    1 persistence agent."""
    from sim.backend import to_host

    out = {"seed": seed}

    # ── the DEVELOP run: one persistent agent lives n_days, accumulating LIVED knowledge ──
    agent = _build_agent(seed)
    bridge = agent._merged_bridge
    hunger = SpikingHunger(bridge, window=drive_window)
    pre_conn = to_host(bridge.cp_connections.data).copy()          # MOAT (in vivo): pre-life conversational synapses
    state = LiveState(seed)
    cache = set()
    per_day = _run_multiday(agent, hunger, state, n_days=n_days, steps_per_day=steps_per_day,
                            drive_read_every=drive_read_every, cache=cache)
    conv_byte_frozen = bool(np.array_equal(pre_conn, to_host(bridge.cp_connections.data)))
    corr = _drive_corr_sweep(hunger)
    out["per_day"] = per_day
    out["conv_byte_frozen"] = conv_byte_frozen
    out["corr_deficit_drive_sweep"] = corr
    final_facts = len(state.lived_facts)
    final_recall = per_day[-1] if per_day else {"recall_ok": 0, "recall_tot": 0}

    # ── persistence-across-reset: save the developed brain, reload into a FRESH agent, re-instate, recall matches;
    #    a cold-start (no re-instate) is empty. ──
    out["persist"] = _persistence_check(seed, root, state)

    # ── FROZEN-BRAIN control: an identical multi-day life but commit_facts=False (sees/grounds, does NOT store) ->
    #    competence must stay FLAT (no accumulated facts -> recall 0). ──
    fr_agent = _build_agent(seed)
    fr_hunger = SpikingHunger(fr_agent._merged_bridge, window=drive_window)
    fr_state = LiveState(seed)
    _run_multiday(fr_agent, fr_hunger, fr_state, n_days=n_days, steps_per_day=steps_per_day,
                  drive_read_every=drive_read_every, commit_facts=False, cache=set(), track=False)
    fr_recall_ok, fr_recall_tot = _lived_recall(fr_agent, fr_state.lived_facts)
    out["frozen"] = {"facts_known": len(fr_state.lived_facts), "recall_ok": fr_recall_ok, "recall_tot": fr_recall_tot}

    # ── LIVED-not-scripted control: a PERMUTED-world life (reversed introduction order) -> a DIFFERENT chain ->
    #    a DIFFERENT lived-fact set (the memory tracks the lived layout, not a script). ──
    pm_agent = _build_agent(seed)
    pm_hunger = SpikingHunger(pm_agent._merged_bridge, window=drive_window)
    pm_state = LiveState(seed)
    _run_multiday(pm_agent, pm_hunger, pm_state, n_days=n_days, steps_per_day=steps_per_day,
                  drive_read_every=drive_read_every, order=list(reversed(DEV_ORDER)), cache=set(), track=False)
    canon_facts = set(tuple(f) for f in state.lived_facts)
    perm_facts = set(tuple(f) for f in pm_state.lived_facts)
    out["permuted"] = {"canon_facts": sorted(canon_facts), "perm_facts": sorted(perm_facts),
                       "facts_differ": bool(canon_facts != perm_facts and len(perm_facts) >= 1)}

    out["verdict"] = _verdict(out)
    _print_seed(seed, out)
    return out


def _persistence_check(seed, root, state):
    """Save the developed brain (body + all accumulated lived facts + grounded codes), then in ONE fresh agent:
    query the facts BEFORE re-instating (the NO-PERSISTENCE cold start -> empty -> 0), then re-instate (grounded
    codes + re-store the facts) and query again (the PERSISTED resume -> recalls all). Cheap-first JSON re-instate."""
    seed_root = os.path.join(root, f"seed{seed}_persist")
    lineage = BridgeLineage(f"develop_body_{seed}", root=Path(seed_root))
    payload = {"body": state.body_payload(), "memory": state.memory_payload()}

    def save_fn(_unused, path_str):
        with open(path_str, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    lineage.save(None, save_fn=save_fn, tier="develop-with-a-body",
                 arch={"kind": "tier3_develop_with_a_body", "L": L}, snapshot=False)
    with open(lineage.load(), "r", encoding="utf-8") as fh:
        mem = json.load(fh)["memory"]
    lived_facts = [tuple(f) for f in mem["lived_facts"]]

    fresh = _build_agent(seed)
    c_ok, c_tot = _lived_recall(fresh, lived_facts)                # cold: empty kb -> 0
    for obj, code in mem["grounded_codes"].items():
        fresh.composer.concepts[obj] = _decode_code(code)
    for (a, v, p) in lived_facts:
        fresh.composer.store(a, v, p)
    p_ok, p_tot = _lived_recall(fresh, lived_facts)                # resumed: re-instated -> recalls all
    resumed_remembers = bool(p_tot > 0 and p_ok == p_tot)
    return {"resumed_remembers": resumed_remembers, "resumed_recall": [p_ok, p_tot], "cold_recall": [c_ok, c_tot],
            "no_persistence_differs": bool(resumed_remembers and c_ok == 0 and p_tot > 0)}


def _verdict(out):
    per_day = out.get("per_day", [])
    if len(per_day) < 2:
        return {"go": False, "reason": "need >= 2 days"}
    facts_seq = [d["facts_known"] for d in per_day]
    # (1) develops: facts grow over days (final > first, and monotonic non-decreasing).
    develops = bool(facts_seq[-1] > facts_seq[0] and all(b >= a for a, b in zip(facts_seq, facts_seq[1:])))
    # (2) retention/no-forget: on the LAST day, ALL accumulated facts recalled (>= 0.8) -- day-0 facts survive.
    last = per_day[-1]
    retention = bool(last["recall_tot"] >= 2 and (last["recall_ok"] / last["recall_tot"]) >= 0.8)
    # (3) frozen: competence flat (no committed facts -> recall ~0).
    fr = out.get("frozen", {})
    frozen_flat = bool(fr.get("facts_known", 1) == 0 and fr.get("recall_ok", 1) == 0)
    # (4) lived-not-scripted: the permuted world yields a different fact set.
    lived_not_scripted = bool(out.get("permuted", {}).get("facts_differ", False))
    # (5) moat: every day abstains on the (obj, MOAT_VERB) cue AND the conversational synapses are byte-frozen.
    moat = bool(all(d["moat_tot"] >= 1 and d["moat_ok"] == d["moat_tot"] for d in per_day)
                and out.get("conv_byte_frozen", False))
    # (6) persistence across reset.
    persistence = bool(out.get("persist", {}).get("resumed_remembers") and
                       out.get("persist", {}).get("no_persistence_differs"))
    # (*) alive across the multi-day life (survival healthy every day) + the drive is spiking.
    alive = bool(all(d["min_energy"] > HEALTHY for d in per_day))
    corr_ok = float(out.get("corr_deficit_drive_sweep", 0.0)) >= 0.9
    go = bool(develops and retention and frozen_flat and lived_not_scripted and moat and persistence and alive
              and corr_ok)
    return {"go": go, "develops": develops, "retention": retention, "frozen_flat": frozen_flat,
            "lived_not_scripted": lived_not_scripted, "moat": moat, "persistence": persistence, "alive": alive,
            "corr_ok": corr_ok, "facts_seq": facts_seq}


def _print_seed(seed, out):
    v = out["verdict"]
    pd = out["per_day"]
    fr = out.get("frozen", {})
    pm = out.get("permuted", {})
    pe = out.get("persist", {})
    facts_day = [d["facts_known"] for d in pd]
    moat_day = ["{}/{}".format(d["moat_ok"], d["moat_tot"]) for d in pd]
    print(f"  [seed {seed}] facts/day {facts_day} | last-recall "
          f"{pd[-1]['recall_ok']}/{pd[-1]['recall_tot']} | moat/day {moat_day} "
          f"| conv-frozen {out.get('conv_byte_frozen')} | corr {out.get('corr_deficit_drive_sweep', 0):+.2f}",
          flush=True)
    print(f"           frozen facts {fr.get('facts_known')} recall {fr.get('recall_ok')}/{fr.get('recall_tot')} | "
          f"permuted-differs {pm.get('facts_differ')} | persist resumed {pe.get('resumed_recall')} cold "
          f"{pe.get('cold_recall')} || {'GO' if v.get('go') else 'NO'}  {v}", flush=True)


def _run_smoke(a):
    """Tiny GPU mechanics check: 1 seed, few short days -- does the multi-day JOIN close (develops over days,
    retention, frozen-flat, permuted-differs, moat, persist)?"""
    root = tempfile.mkdtemp(prefix="develop_body_smoke_")
    try:
        r = run_seed(a.seeds[0], root, n_days=max(3, a.n_days), steps_per_day=min(a.steps_per_day, 300),
                     drive_read_every=a.drive_read_every)
        v = r["verdict"]
        ok = bool(v.get("develops") and v.get("retention") and v.get("frozen_flat")
                  and v.get("lived_not_scripted") and v.get("moat") and v.get("persistence"))
        print(f"[smoke] {'JOIN-CLOSES' if ok else 'CHECK'}  {v}", flush=True)
        return 0 if ok else 1
    finally:
        if not a.keep_lineage:
            shutil.rmtree(root, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-days", type=int, default=4)
    ap.add_argument("--steps-per-day", type=int, default=700)
    ap.add_argument("--drive-read-every", type=int, default=10)
    ap.add_argument("--out", default="research/findings/raw/_tier3_develop_with_a_body.json")
    ap.add_argument("--keep-lineage", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    print("[Tier-3 develop-with-a-body] does a brain DEVELOP over DAYS where each day's knowledge is LIVED "
          "(perceived), retaining old lived facts (no-forget) + persisting across a reset?\n"
          "  GATES: (1) develops (facts grow)  (2) retention (old facts recalled)  (3) frozen-brain flat  "
          "(4) lived-not-scripted (permuted-world differs)  (5) converse + no-confab MOAT  (6) persistence.\n",
          flush=True)

    if a.smoke:
        return _run_smoke(a)

    root = tempfile.mkdtemp(prefix="develop_body_")
    per_seed = []
    try:
        for seed in a.seeds:
            per_seed.append(run_seed(seed, root, n_days=a.n_days, steps_per_day=a.steps_per_day,
                                     drive_read_every=a.drive_read_every))
    finally:
        if not a.keep_lineage:
            shutil.rmtree(root, ignore_errors=True)

    n_go = sum(p["verdict"].get("go", False) for p in per_seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"per_seed": per_seed, "n_go": n_go, "n_seeds": len(per_seed)}, fh, indent=2, default=str)

    print(f"\n{'='*110}", flush=True)
    if per_seed and n_go == len(per_seed):
        print(f"  GO ({n_go}/{len(per_seed)} seeds): a brain DEVELOPS over DAYS from LIVED experience. A persistent "
              "merged one-brain forages a cumulatively-richer world, GROUNDS+STORES the objects it encounters (its "
              "lived-fact knowledge GROWS day-over-day), RETAINS old lived facts as new days add more (no-forget), "
              "answers about them + ABSTAINS on unstored cues (moat byte-frozen), and RESUMES the developed brain "
              "after a reset. A FROZEN brain (sees but doesn't store) stays flat; a PERMUTED world yields different "
              "lived facts (the knowledge is LIVED, not scripted). ⇒ the second Tier-3 slice: a brain that develops "
              "over time from what it LIVES. HONEST SCOPE: the corridor + 4-object perceivable set bounds the graph "
              "(~3 lived facts); richer world / pair-accumulation / the 24/7 develop_gpu harness (Option 2B) are "
              "follow-ons; the learned spatial policy stays the deferred Tier-4 dendrite wall.", flush=True)
    else:
        print(f"  PARTIAL/NEGATIVE ({n_go}/{len(per_seed)} seeds): localize (develops / retention / frozen / "
              "lived-not-scripted / moat / persistence). An honest negative that pins the wall is a valid "
              "deliverable.", flush=True)
    print(f"  [saved] {a.out}\n{'='*110}", flush=True)
    return 0 if (per_seed and n_go == len(per_seed)) else 1


if __name__ == "__main__":
    sys.exit(main())
