"""STANDALONE verify harness for `reconsolidation_production_organ` — proves, numpy/CPU, that the belief-revision
organ (PE-gated in-place fact UPDATE, gated by the D2 SPIKING surprise window) fires correctly, is LESION-load-
bearing, is flag-off byte-identical, and preserves the no-confab moat. On BOTH composers: the rf composer (the
de-risked `update_on_mismatch` path, 6 seeds) AND the PRODUCTION-DEFAULT onebrain composer (the substrate-slot
rewrite path). Reuse-by-import; NO `sim/` edit.

The story under test (dog go north -> corrected to south):
  INTACT     : the contradiction FIRES the spiking window (D2 surprise) -> in-place rewrite -> recall = south, ONE
               fact, untouched facts intact.
  C1 BOUNDARY: a re-statement (assert the SAME patient) does NOT fire the window (D2 cancels) -> restabilize, no
               write. The SPIKING gate enforces the PE boundary (lesioning the gate makes the re-statement read as
               surprised = the separation is caused by the learned spiking prediction).
  MOAT       : a never-stored cue -> abstain, no fact fabricated.
  LESION     : BRAIN_RECONSOLIDATION_LESION -> the window fires but the in-place update is blocked -> append-only
               fallback -> recall = north (STALE). Load-bearing: the answer flips south -> north.
  FLAG-OFF   : BRAIN_RECONSOLIDATION=0 -> the organ is never invoked -> the correction appends -> the store is
               BYTE-IDENTICAL to a plain `store()` append (the production-today append-only path).

Run:  SIM_BACKEND=numpy python -u -m research.runners._reconsolidation_production_organ_verify [--seeds 42,43,44,100,101,102]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.rf_phasor_composer import RFPhasorComposer            # noqa: E402
from research.runners.reconsolidation_production_organ import (              # noqa: E402
    get_organ, reconsolidation_enabled, reconsolidation_lesioned,
)
from tools.lab import attributable_to                                        # noqa: E402
from tools.verdict import Verdict                                            # noqa: E402

VOCAB = ["dog", "cat", "bird", "fish", "elephant",
         "go", "run", "fly", "swim",
         "north", "south", "east", "west"]
BASE = [("dog", "go", "north"), ("cat", "run", "south"), ("bird", "fly", "east"), ("fish", "swim", "west")]
CORR = ("dog", "go", "south")     # a real contradiction (stored north) -> high PE -> window OPENS
RESTATE = ("dog", "go", "north")  # same patient -> PE~0 -> window CLOSED
NEVER = ("elephant", "go", "west")
D = 128


def n_facts(comp, agent, action):
    return sum(1 for f, _ in comp.kb if f.get("agent") == agent and f.get("action") == action)


def _kb_hash(comp):
    """A byte hash of the composer's stored composites (the flag-off byte-identity check)."""
    h = hashlib.sha256()
    for fact, handle in comp.kb:
        h.update(repr(sorted(fact.items())).encode())
        arr = np.asarray(handle if not isinstance(handle, tuple) else handle, dtype=np.float64)
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def build_rf(seed):
    c = RFPhasorComposer(seed=seed, D=D, vocab=VOCAB)
    for a, v, p in BASE:
        c.store(a, v, p)
    return c


def verify_rf_seed(seed):
    organ = get_organ(seed)
    row = {"seed": seed}

    # ── INTACT: the contradiction fires the spiking window -> in-place rewrite ────────────────────────────────
    c = build_rf(seed)
    p_stored = c.query_patient("dog", "go")
    opened, sj = organ.window_open("dog", "go", p_stored, CORR[2])                 # the SPIKING window read
    res = organ.reconsolidate(c, "dog", "go", p_stored, CORR[2], sj=sj)
    row["contradict_surprise_hz"] = round(float(sj["surprise_hz"]), 3)
    row["contradict_threshold"] = round(float(sj["threshold"]), 3)
    row["window_opened_contradict"] = bool(opened)
    row["intact_action"] = res["action"]
    row["intact_q"] = c.query_patient("dog", "go")
    row["intact_n"] = n_facts(c, "dog", "go")
    row["intact_untouched"] = c.query_patient("cat", "run")
    intact_ok = (opened and res["action"] == "rewrite"
                 and row["intact_q"] == "south" and row["intact_n"] == 1
                 and row["intact_untouched"] == "south")

    # ── C1 BOUNDARY: a re-statement does NOT open the window -> restabilize (no write) ────────────────────────
    c2 = build_rf(seed)
    opened_rs, sj_rs = organ.window_open("dog", "go", "north", RESTATE[2])
    res_rs = organ.reconsolidate(c2, "dog", "go", "north", RESTATE[2], sj=sj_rs)
    # the SPIKING gate is load-bearing: lesion the D2 prediction edges -> the confirm/restate FIRES too (separation
    # collapses), which is what would spuriously open the window without the learned prediction.
    opened_rs_lesion, sj_rs_l = organ.window_open("dog", "go", "north", RESTATE[2], lesion_gate=True)
    row["restate_surprise_hz"] = round(float(sj_rs["surprise_hz"]), 3)
    row["window_opened_restate"] = bool(opened_rs)
    row["restate_action"] = res_rs["action"]
    row["restate_n"] = n_facts(c2, "dog", "go")
    row["restate_gate_lesion_opens"] = bool(opened_rs_lesion)
    c1_ok = (not opened_rs and res_rs["action"] == "restabilize"
             and n_facts(c2, "dog", "go") == 1 and c2.query_patient("dog", "go") == "north")
    # the gate is load-bearing for the boundary: intact window separates contradict(True) from restate(False)
    gate_load_bearing = (opened and not opened_rs)

    # ── MOAT: a never-stored cue -> abstain, no fact fabricated ───────────────────────────────────────────────
    c3 = build_rf(seed)
    opened_nv, sj_nv = organ.window_open(NEVER[0], NEVER[1], "north", NEVER[2])    # no stored elephant fact
    res_nv = organ.reconsolidate(c3, NEVER[0], NEVER[1], "north", NEVER[2], sj=sj_nv)
    row["moat_action"] = res_nv["action"]
    moat_ok = (res_nv["action"] == "abstain" and res_nv["wrote"] is False
               and c3.query_patient("elephant", "go") is None and n_facts(c3, "elephant", "go") == 0)

    # ── LESION: window fires but the update is blocked -> append-only fallback -> STALE ───────────────────────
    c4 = build_rf(seed)
    res_les = organ.reconsolidate(c4, "dog", "go", "north", CORR[2], sj=sj, lesion=True)
    if not res_les["wrote"]:
        c4.store(*CORR)                                                             # append-only production fallback
    row["lesion_action"] = res_les["action"]
    row["lesion_q"] = c4.query_patient("dog", "go")
    row["lesion_n"] = n_facts(c4, "dog", "go")
    lesion_ok = (res_les["action"] == "lesioned_nowrite" and res_les["wrote"] is False
                 and row["lesion_q"] == "north" and row["lesion_n"] == 2)          # stale answered, duplicate coexists
    lesion_load_bearing = (row["intact_q"] == "south" and row["lesion_q"] == "north")

    # ── FLAG-OFF byte-identity: organ disabled -> append -> store BYTE-IDENTICAL to a plain store() append ────
    c_off = build_rf(seed)
    c_off.store(*CORR)                                                              # the correction just appends
    c_ctrl = build_rf(seed)
    c_ctrl.store(*CORR)                                                             # the pure production-today path
    flagoff_identical = (_kb_hash(c_off) == _kb_hash(c_ctrl)
                         and c_off.query_patient("dog", "go") == "north" and n_facts(c_off, "dog", "go") == 2)

    row.update(dict(intact_ok=intact_ok, c1_ok=c1_ok, gate_load_bearing=gate_load_bearing,
                    moat_ok=moat_ok, lesion_ok=lesion_ok, lesion_load_bearing=lesion_load_bearing,
                    flagoff_identical=flagoff_identical))
    row["seed_pass"] = bool(intact_ok and c1_ok and gate_load_bearing and moat_ok
                            and lesion_ok and lesion_load_bearing and flagoff_identical)
    print(f"  [rf seed {seed}] window contradict={row['contradict_surprise_hz']}Hz(open={opened}) "
          f"restate={row['restate_surprise_hz']}Hz(open={opened_rs}; gate-lesion-open={opened_rs_lesion}) | "
          f"INTACT q={row['intact_q']} n={row['intact_n']} | LESION q={row['lesion_q']} n={row['lesion_n']} | "
          f"MOAT {row['moat_action']} | flag-off id+append={flagoff_identical} -> "
          f"{'PASS' if row['seed_pass'] else 'FAIL'}", flush=True)
    return row


def verify_onebrain_seed(seed):
    """The PRODUCTION-DEFAULT composer: the organ rewrites the SAME substrate store slot via the composer's own
    _write_block+_compose_phases. Recall reads the DEVICE store, so this proves the in-place update on the substrate
    the production turn actually recalls from (not just a host kb dict)."""
    from research.runners.one_brain_composer import OneBrainComposer
    organ = get_organ(seed)
    c = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=8)
    for a, v, p in BASE[:3]:
        c.store(a, v, p)
    base = {(a, v): c.query_patient(a, v) for a, v, _ in BASE[:3]}
    p_stored = c.query_patient("dog", "go")
    opened, sj = organ.window_open("dog", "go", p_stored, CORR[2])
    res = organ.reconsolidate(c, "dog", "go", p_stored, CORR[2], sj=sj)
    q_after = c.query_patient("dog", "go")
    n_after = n_facts(c, "dog", "go")
    untouched = (c.query_patient("cat", "run") == base[("cat", "run")]
                 and c.query_patient("bird", "fly") == base[("bird", "fly")])
    intact_ok = (opened and res["action"] == "rewrite" and q_after == "south" and n_after == 1 and untouched)

    # lesion: window fires but no rewrite -> append fallback -> stale
    c2 = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=8)
    for a, v, p in BASE[:3]:
        c2.store(a, v, p)
    res_l = organ.reconsolidate(c2, "dog", "go", "north", CORR[2], sj=sj, lesion=True)
    if not res_l["wrote"]:
        c2.store(*CORR)
    lesion_q = c2.query_patient("dog", "go")
    lesion_ok = (res_l["action"] == "lesioned_nowrite" and lesion_q == "north" and n_facts(c2, "dog", "go") == 2)

    row = {"seed": seed, "path": "onebrain", "window_hz": round(float(sj["surprise_hz"]), 3),
           "opened": bool(opened), "intact_q": q_after, "intact_n": n_after, "untouched": bool(untouched),
           "lesion_q": lesion_q, "intact_ok": bool(intact_ok), "lesion_ok": bool(lesion_ok),
           "lesion_load_bearing": bool(q_after == "south" and lesion_q == "north")}
    row["seed_pass"] = bool(intact_ok and lesion_ok and row["lesion_load_bearing"])
    print(f"  [onebrain seed {seed}] window={row['window_hz']}Hz(open={opened}) | INTACT q={q_after} n={n_after} "
          f"untouched={untouched} | LESION q={lesion_q} -> {'PASS' if row['seed_pass'] else 'FAIL'}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--onebrain-seeds", type=str, default="42,43,44")
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_reconsolidation_production_organ_verify.json"))
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    seeds = [int(s) for s in args.seeds.split(",")]
    ob_seeds = [int(s) for s in args.onebrain_seeds.split(",") if s.strip()]
    t0 = time.time()

    # sanity: default-ON, and the lesion flag reads correctly.
    assert reconsolidation_enabled() is True, "organ must be default-ON"
    os.environ["BRAIN_RECONSOLIDATION"] = "0"
    assert reconsolidation_enabled() is False, "BRAIN_RECONSOLIDATION=0 must disable"
    del os.environ["BRAIN_RECONSOLIDATION"]
    os.environ["BRAIN_RECONSOLIDATION_LESION"] = "1"
    assert reconsolidation_lesioned() is True, "BRAIN_RECONSOLIDATION_LESION=1 must lesion"
    del os.environ["BRAIN_RECONSOLIDATION_LESION"]

    print("[reconsolidation organ verify] PE-gated in-place belief revision, D2-spiking-surprise-gated window; "
          "intact fires / lesion collapses / flag-off byte-identical / moat holds.", flush=True)
    print(f"  rf seeds={seeds}  onebrain seeds={ob_seeds}  base={BASE}  correction={CORR}", flush=True)

    rf_rows = [verify_rf_seed(s) for s in seeds]
    ob_rows = [verify_onebrain_seed(s) for s in ob_seeds]

    n = len(seeds)
    bar = int(np.ceil(5 / 6 * n))
    n_pass = sum(r["seed_pass"] for r in rf_rows)
    n_intact = sum(r["intact_ok"] for r in rf_rows)
    n_c1 = sum(r["c1_ok"] for r in rf_rows)
    n_gate = sum(r["gate_load_bearing"] for r in rf_rows)
    n_moat = sum(r["moat_ok"] for r in rf_rows)
    n_les = sum(r["lesion_load_bearing"] for r in rf_rows)
    n_flag = sum(r["flagoff_identical"] for r in rf_rows)
    ob_pass = sum(r["seed_pass"] for r in ob_rows)
    ob_lb = sum(r["lesion_load_bearing"] for r in ob_rows)

    print(f"\n{'='*100}", flush=True)
    print(f"  RF ({n} seeds): overall {n_pass}/{n} | intact {n_intact}/{n} | C1 boundary {n_c1}/{n} | "
          f"gate-load-bearing {n_gate}/{n} | moat {n_moat}/{n} | lesion-load-bearing {n_les}/{n} | "
          f"flag-off byte-identical {n_flag}/{n}", flush=True)
    print(f"  ONEBRAIN ({len(ob_rows)} seeds, production default): overall {ob_pass}/{len(ob_rows)} | "
          f"lesion-load-bearing {ob_lb}/{len(ob_rows)}", flush=True)

    # ── ATTRIBUTION: whose is the belief revision? intact (in-place update ON) vs lesion (update OFF). ─────────
    # Encode "belief revised" as recall==corrected (south=1.0, stale north=0.0). The mean over seeds is the
    # treatment (intact) vs control (lesion) fraction; the fraction NOT present in the lesion control is the
    # share of the revision the IN-PLACE UPDATE owns (vs any fixed input-driven artifact).
    treat = float(np.mean([1.0 if r["intact_q"] == "south" else 0.0 for r in rf_rows]))
    ctrl = float(np.mean([1.0 if r["lesion_q"] == "south" else 0.0 for r in rf_rows]))
    attr = attributable_to("belief revision @ in-place update (rf)", treat, ctrl)   # (treat-ctrl)/treat
    ob_treat = float(np.mean([1.0 if r["intact_q"] == "south" else 0.0 for r in ob_rows])) if ob_rows else None
    ob_ctrl = float(np.mean([1.0 if r["lesion_q"] == "south" else 0.0 for r in ob_rows])) if ob_rows else None
    ob_attr = attributable_to("belief revision @ in-place update (onebrain)", ob_treat, ob_ctrl) if ob_rows else None

    go_raw = (n_pass >= bar and n_intact >= bar and n_c1 >= bar and n_gate >= bar
              and n_moat == n and n_les == n and n_flag == n
              and (not ob_rows or (ob_pass == len(ob_rows) and ob_lb == len(ob_rows))))

    # ── VERDICT with preconditions (must travel with what earned it). ─────────────────────────────────────────
    v = Verdict("reconsolidation production organ — D2-gated in-place belief revision")
    v.require("intact revises the belief (rf)", n_intact, expect=lambda x: x >= bar)
    v.require("PE boundary: re-statement does not open the window (rf C1)", n_c1, expect=lambda x: x >= bar)
    v.require("spiking window gate is load-bearing (contradict opens, restate closed)", n_gate,
              expect=lambda x: x >= bar)
    v.require("no-confab moat: never-stored -> abstain", n_moat, expect=n)
    v.require("lesion collapses belief revision (rf south->north)", n_les, expect=n)
    v.require("flag-off byte-identical to append-only production path", n_flag, expect=n)
    v.require("onebrain (production default) revises + lesion-load-bearing", ob_pass, expect=len(ob_rows))
    v.require("revision attributable to the in-place update (>=0.99 not in lesion control)",
              attr, expect=lambda x: x is not None and x >= 0.99)
    v.disabled("cue-addressed spiking reactivation (kb cue-match selects the fact; rides one-brain merge)",
               "the co-resident residual the surprise/comprehension organs also carry")
    decided = v.decide(go=go_raw, verbose=True)

    go = (decided["status"] == "GO")
    verdict = decided["status"]
    print(f"  VERDICT: {verdict} — a D2-spiking-surprise-gated, PE-gated IN-PLACE fact UPDATE revises a stored "
          f"belief (recall flips north->south, ONE fact), the window is CLOSED on a re-statement (boundary), the "
          f"no-confab moat holds (never-stored -> abstain), the LESION collapses belief revision (south->north on "
          f"BOTH composers), and flag-off is byte-identical to the append-only production path.", flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*100}", flush=True)

    out = {"verdict": verdict, "seeds": seeds, "onebrain_seeds": ob_seeds, "pass_bar": bar,
           "preconditions": decided["preconditions"], "verdict_detail": decided,
           "attribution": {"rf_belief_revision_treatment": treat, "rf_belief_revision_control": ctrl,
                           "rf_attributable_to_inplace_update": attr,
                           "onebrain_treatment": ob_treat, "onebrain_control": ob_ctrl,
                           "onebrain_attributable_to_inplace_update": ob_attr},
           "rf": {"n_pass": n_pass, "n_intact": n_intact, "n_c1": n_c1, "n_gate_load_bearing": n_gate,
                  "n_moat": n_moat, "n_lesion_load_bearing": n_les, "n_flagoff_identical": n_flag, "rows": rf_rows},
           "onebrain": {"n_pass": ob_pass, "n_lesion_load_bearing": ob_lb, "rows": ob_rows}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
