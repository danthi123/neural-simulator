"""A10 fix round 2 (after the review of 7d5c2743d): MODULE-LEVEL check that the A10 read leaves no footprint on the
PRODUCTION surprise organ, with a sensitivity control. Governed by
research/findings/2026-09-24-reward-value-spiking-afferent-PREREG-AMENDMENT-2.md (committed before this runner's
first run). An instrument check of the fix, not a capability verdict.

WHY. The surprise organ is process-shared and default-ON. With `BRAIN_REWARD_VALUE_AFFERENT=1` the da-drives block
reads it (A10) BEFORE the server's own surprise block reads it later in the same turn. Reads on the shared merged
pool depend on read history (the pool bridge has no `_rest_extra`, so `_hard_reset` leaves the surprise slice's
adaptive thresholds / activity EMA / refractory state where the last read left them). In the v2 seed-7 arms the
production CONFIRM read was 0.4050925925925926 Hz OFF and 0.3472222222222222 Hz ON. The fix snapshots and restores
the state the A10 read mutates (webapp/reward_value_afferent_chat.py).

WHAT IT RUNS (one process, the arms' env for the given seed; the organ is the production one: `get_organ(seed)` on
the merged cortical pool, exactly what the server's surprise block reads):
  0. build the organ and the lesion twin; hash the intact organ's read state (H0) and snapshot it (S0).
  1. REFERENCE (flag-OFF order): production CONFIRM read, production CONTRA read.
  2. restore S0 (hash must equal H0).
  3. ISOLATED (the fixed A10 path, intact): A10 CONFIRM read, production CONFIRM read, A10 CONTRA read, production
     CONTRA read. The hash is taken before and after each A10 read.
  4. restore S0. ISOLATED LESION: the same with BRAIN_REWARD_VALUE_LESION=1 (the A10 reads use the twin).
  5. restore S0. RAW (the v2 A10 path, the sensitivity control): `sorg.judge` CONFIRM exactly as v2's A10 did, then
     the production CONFIRM read.
  6. restore S0. (reported, not scored) the v1 runner's in-process order: CONFIRM, CONTRA, CONFIRM -- the value of
     the third read is what v1's ON arm recorded as its A10 CONFIRM read (the review's attribution of the v1->v2
     CONFIRM shift to read history).
"production read" = `sorg.judge(agent, action, stored, asserted, lesion=False)`, the call webapp/server.py makes.

SCORED (AMENDMENT-2, section "module-level footprint check"):
  preconditions: the organ is on the merged pool (production path); the reference CONFIRM and CONTRA reads equal
    the v2 OFF arm's handler-level `surprise.surprise_hz` (read from the artifact file); every restore of S0 hashes
    back to H0; the sensitivity control sees a shift (RAW production CONFIRM != REFERENCE CONFIRM), else the check
    could not have failed.
  GO iff: F1 every A10 read (intact and lesion) leaves the organ hash unchanged; F2 every production read in the
    isolated sequences equals the REFERENCE read for its turn exactly; F3 each intact A10 read equals the production
    read that follows it (the A10 read now sees what production sees).

Run (local, under memcap; ~6 min on numpy):
  bash tools/memcap.sh 10 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m \
      research.runners._reward_value_afferent_footprint --seed 7 \
      --out research/findings/raw/_reward_value_afferent_derisk/v3/footprint_module.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

CONFIRM = ("dog", "chase", "cat", "cat", "the dog chase the cat")
CONTRA = ("dog", "chase", "cat", "fish", "the dog chase the fish")
V2_OFF = "research/findings/raw/_reward_value_afferent_derisk/v2/s7_arms_off_a.json"


def _to_bytes(a):
    return (a.get() if hasattr(a, "get") else a).tobytes()


def state_hash(sorg, bridge=None) -> str:
    """sha256 over everything the A10 snapshot covers: every array / sparse / scalar attribute of the bridge (objects
    by type and identity), the runtime clock's scalars and the organ's block bookkeeping."""
    from webapp.reward_value_afferent_chat import _is_dense, _is_sparse, _SCALAR_TYPES
    b = sorg.bridge if bridge is None else bridge
    h = hashlib.sha256()
    for name in sorted(vars(b)):
        v = getattr(b, name)
        h.update(name.encode())
        if _is_dense(v):
            h.update(str((v.dtype, v.shape)).encode())
            h.update(_to_bytes(v))
        elif _is_sparse(v):
            for part in (v.data, v.indices, v.indptr):
                h.update(_to_bytes(part))
        elif isinstance(v, _SCALAR_TYPES):
            h.update(repr(v).encode())
        elif isinstance(v, (list, dict, set)):
            h.update(("%s:%d:%d" % (type(v).__name__, id(v), len(v))).encode())
        else:
            h.update(("%s:%d" % (type(v).__name__, id(v))).encode())
    rs = getattr(b, "runtime_state", None)
    if rs is not None and hasattr(rs, "__dict__"):
        for k in sorted(vars(rs)):
            if isinstance(getattr(rs, k), _SCALAR_TYPES):
                h.update(("%s=%r" % (k, getattr(rs, k))).encode())
    h.update(repr(sorted(getattr(sorg, "_block", {}).items())).encode())
    h.update(repr((getattr(sorg, "_cue_next", None), getattr(sorg, "_novel_next", None))).encode())
    return h.hexdigest()


def _chat():
    return types.SimpleNamespace(inner=types.SimpleNamespace(what_does=lambda a, v: "cat"))


def run(seed, out_path):
    from research.runners._reward_value_afferent_derisk import arm_env
    os.environ.update(arm_env("off_a", seed, None))
    os.environ.setdefault("SIM_BACKEND", "numpy")
    import research.runners.surprise_production_organ as SO
    import webapp.reward_value_afferent_chat as RVA
    from tools.verdict import Verdict

    t0 = time.time()
    sorg = SO.get_organ(seed=seed)
    sorg.ensure_built()
    sorg._ensure_les()
    build_s = round(time.time() - t0, 1)
    pool_kind = type(sorg._shared).__name__ if sorg._shared is not None else None
    H0 = state_hash(sorg)
    S0 = RVA._snapshot_read_state(sorg, sorg.bridge)

    def prod(turn):
        a, v, s, p, _ = turn
        return float(sorg.judge(a, v, s, p, lesion=False)["surprise_hz"])

    def a10(turn, lesion):
        if lesion:
            os.environ["BRAIN_REWARD_VALUE_LESION"] = "1"
        else:
            os.environ.pop("BRAIN_REWARD_VALUE_LESION", None)
        try:
            h_before = state_hash(sorg)
            info = RVA.spiking_reward_value(_chat(), turn[4], seed=seed)
            h_after = state_hash(sorg)
        finally:
            os.environ.pop("BRAIN_REWARD_VALUE_LESION", None)
        return {"surprise_hz": (info or {}).get("surprise_hz"), "drives": (info or {}).get("drives"),
                "footprint": (info or {}).get("footprint"), "error": (info or {}).get("error"),
                "hash_before": h_before, "hash_after": h_after, "hash_unchanged": h_before == h_after}

    def restore_S0():
        RVA._restore_read_state(sorg, sorg.bridge, S0)
        return state_hash(sorg)

    rec = {"seed": int(seed), "seed_kind": "dev-calibration (NOT a 6-seed gate seed)", "build_s": build_s,
           "pool_kind": pool_kind, "H0": H0}
    # 1. REFERENCE (flag-OFF order)
    rec["reference"] = {"confirm": prod(CONFIRM), "contra": prod(CONTRA)}
    rec["restore_1_hash_equal_H0"] = restore_S0() == H0
    # 3. ISOLATED intact
    iso = {"a10_confirm": a10(CONFIRM, False)}
    iso["prod_confirm"] = prod(CONFIRM)
    iso["a10_contra"] = a10(CONTRA, False)
    iso["prod_contra"] = prod(CONTRA)
    rec["isolated_intact"] = iso
    rec["restore_2_hash_equal_H0"] = restore_S0() == H0
    # 4. ISOLATED lesion
    isl = {"a10_confirm": a10(CONFIRM, True)}
    isl["prod_confirm"] = prod(CONFIRM)
    isl["a10_contra"] = a10(CONTRA, True)
    isl["prod_contra"] = prod(CONTRA)
    rec["isolated_lesion"] = isl
    rec["restore_3_hash_equal_H0"] = restore_S0() == H0
    # 5. RAW (the v2 A10 path): an unisolated judge, then production
    raw = {"raw_a10_confirm": prod(CONFIRM)}
    raw["prod_confirm"] = prod(CONFIRM)
    rec["raw_sensitivity"] = raw
    # 6. (reported, not scored) the v1 runner's order in ONE process (f3fa99c4a `main`): OFF confirm, OFF contra,
    #    then the ON arm's A10 CONFIRM read -- the third read of the organ and its second CONFIRM read. v1 recorded
    #    0.3472222222222222 Hz for that read (s7.json); the review attributes the v1->v2 shift to this order.
    rec["restore_4_hash_equal_H0"] = restore_S0() == H0
    v1 = {"off_confirm": prod(CONFIRM), "off_contra": prod(CONTRA)}
    v1["on_a10_confirm_third_read"] = prod(CONFIRM)
    rec["v1_order_reported"] = v1

    v2_off = None
    try:
        d = json.load(open(os.path.join(_REPO, V2_OFF)))
        v2_off = {t: ((d.get(t) or {}).get("surprise") or {}).get("surprise_hz") for t in ("confirm", "contra")}
    except Exception as e:
        rec["v2_off_error"] = "%s: %s" % (type(e).__name__, e)
    rec["v2_off_arm_surprise_hz"] = v2_off

    ref = rec["reference"]
    v = Verdict("A10 fix round 2: the A10 read leaves no footprint on the production surprise organ (module level)")
    v.require("organ is on the merged cortical pool (the production path)", pool_kind, expect=lambda k: k is not None)
    v.require("reference reads equal the v2 OFF arm's handler-level surprise_hz (confirm, contra)",
              None if v2_off is None else (v2_off.get("confirm") == ref["confirm"] and v2_off.get("contra") == ref["contra"]),
              expect=True)
    v.require("every restore of S0 hashes back to H0",
              rec["restore_1_hash_equal_H0"] and rec["restore_2_hash_equal_H0"] and rec["restore_3_hash_equal_H0"]
              and rec["restore_4_hash_equal_H0"],
              expect=True)
    v.require("A10 reads drove (intact and lesion, both turns)",
              all(x["drives"] is True for x in (iso["a10_confirm"], iso["a10_contra"], isl["a10_confirm"], isl["a10_contra"])),
              expect=True)
    sens = raw["prod_confirm"] != ref["confirm"]
    v.require("sensitivity: an UNISOLATED A10 read shifts the production CONFIRM read (the check can fail)", sens,
              expect=True)
    f1 = all(x["hash_unchanged"] for x in (iso["a10_confirm"], iso["a10_contra"], isl["a10_confirm"], isl["a10_contra"]))
    f2 = (iso["prod_confirm"] == ref["confirm"] and iso["prod_contra"] == ref["contra"]
          and isl["prod_confirm"] == ref["confirm"] and isl["prod_contra"] == ref["contra"])
    f3 = (iso["a10_confirm"]["surprise_hz"] == iso["prod_confirm"] and iso["a10_contra"]["surprise_hz"] == iso["prod_contra"])
    # whose was the v2 shift? the unisolated read (treatment) vs the isolated read (control), on the production
    # CONFIRM read's departure from the flag-OFF reference
    from tools.lab import attributable_to, lever
    shift_raw = ref["confirm"] - raw["prod_confirm"]
    shift_iso = ref["confirm"] - iso["prod_confirm"]
    rec["confirm_shift_raw_hz"], rec["confirm_shift_isolated_hz"] = shift_raw, shift_iso
    rec["confirm_shift_attributable_to_unisolated_read"] = (
        attributable_to("production CONFIRM shift owed to the unisolated A10 read (control = isolated read)",
                        shift_raw, shift_iso) if shift_raw != 0.0 else None)
    lever("A10 isolation: production CONFIRM read, unisolated -> isolated", raw["prod_confirm"], iso["prod_confirm"],
          required=False)
    decided = v.decide(go=bool(f1 and f2 and f3), verbose=True)
    rec.update({"F1_a10_reads_leave_hash_unchanged": f1, "F2_production_reads_equal_reference": f2,
                "F3_a10_read_equals_following_production_read": f3,
                "go": bool(decided["go"]), "status": decided["status"],
                "preconditions": decided["preconditions"], "verdict": decided,
                "runner": "_reward_value_afferent_footprint"})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=2, default=str)
    print(json.dumps({"status": decided["status"], "F1": f1, "F2": f2, "F3": f3, "sensitivity": sens,
                      "reference": ref, "out": out_path}), flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="research/findings/raw/_reward_value_afferent_derisk/v3/footprint_module.json")
    a = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    sys.exit(run(a.seed, a.out))
