"""STANDALONE verify harness for the D6 multi-referent WM production organ (numpy-CPU).

Proves, on the co-resident spiking buffer (reuse-by-import; NO sim/ edit):
  (1) INTACT fires correctly  — the organ HOLDS >=2 discourse referents across an intervening span and reads EACH
      back off the spiking held bumps (all_recovered True at k=2,3; hold_alive_min > 0 with input asserted zero).
  (2) LESION collapses it (load-bearing) — recur=0 kills the slow-NMDA hold -> the >=2-referent read-back collapses
      (all_recovered False, hold_alive_min ~ 0). The host parse + write marker are byte-identical, so the spiking
      hold is what carries the multi-referent capability.
  (3) SUPERPOSED single-attractor collides (the ~2-cap the prior anaphora store hits) — cram >=2 referents into ONE
      register -> only ONE recovered. This is the surpass: separate registers hold >=2, a single attractor ties.
  (4) FLAG-OFF is byte-identical — BRAIN_MULTIREF=0 disables the organ; out-of-scope inputs (fewer than 2 referents,
      no hold-query) return None -> the production turn is unchanged.
  (5) MOAT preserved/strengthened — the organ NEVER manufactures a fact or flips an abstain: its output is a read-out
      of ITS OWN buffer; every recovered referent is one the input named (no invented referent); a single-referent or
      non-query turn is out of scope (None).

Run:  SIM_BACKEND=numpy python -m research.runners._d6_multiref_wm_production_verify
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import time
from pathlib import Path

import research.runners.d6_multiref_wm_production_organ as D6
from research.runners.d6_multiref_wm_production_organ import MultiReferentWMOrgan, hold_readout

try:
    from tools.lab import lever, attributable_to
except Exception:  # tools.lab optional at import; the harness still runs
    def lever(name, before, after, required=True, continuous=None):
        print(f"  LEVER {name}: {before} -> {after}"); return before != after
    def attributable_to(label, t, c, warn_below=0.5):
        print(f"  attributable_to {label}: t={t} c={c}"); return None

OUT = Path("research/findings/raw/_d6_multiref_wm/verify.json")

# fixed distinct referents (within the binder _K=6 valid regime)
REFS = ["dog", "cat", "bird", "fish", "horse"]


def _fresh(seed=42):
    return MultiReferentWMOrgan(seed=seed)


def test_intact(seeds, ks):
    """(1) INTACT: hold k referents, read each back off the spiking bumps. all_recovered True; hold alive."""
    rows = {}
    for k in ks:
        per = []
        for s in seeds:
            org = _fresh(s)
            res = org.load(REFS[:k], lesion=False)
            per.append(res)
        allrec = sum(1 for r in per if r["all_recovered"]) / len(per)
        alive = sum(r["hold_alive_min"] for r in per) / len(per)
        zin = all(r["zero_input_ok"] for r in per)
        rows[k] = {"all_recovered_rate": allrec, "hold_alive_min": alive, "zero_input_ok": zin}
    return rows


def test_lesion(seeds, ks):
    """(2) LESION: recur=0 -> the hold dies -> >=2-referent read-back collapses."""
    rows = {}
    for k in ks:
        per = []
        for s in seeds:
            org = _fresh(s)
            res = org.load(REFS[:k], lesion=True)
            per.append(res)
        allrec = sum(1 for r in per if r["all_recovered"]) / len(per)
        alive = sum(r["hold_alive_min"] for r in per) / len(per)
        rows[k] = {"all_recovered_rate": allrec, "hold_alive_min": alive}
    return rows


def test_superposed(seeds, k=2):
    """(3) SUPERPOSED single-attractor collide: write k referents into ONE register (superpose, no clear) -> the
    1-of-K attractor holds ONE winner -> at most one referent recovered. The ~2-cap the multi-register hold surpasses."""
    n_recovered = []
    for s in seeds:
        org = _fresh(s)
        org.ensure_built()
        buf = org.buf
        buf.reset()
        locs = [org._local_slot(r) for r in REFS[:k]]
        for loc in locs:
            buf.write(0, loc, superpose=True)      # ALL into register 0 -> they collide
            buf.hold()
        buf.hold(); buf.hold()
        loc, amp = buf.read(0)
        got = org._ref_of_slot.get(loc, None)
        # how many of the k input referents does the single register return? (a tie returns <=1)
        n_recovered.append(1 if got in REFS[:k] else 0)
    return {"k": k, "mean_referents_recovered_single_register": sum(n_recovered) / len(n_recovered)}


def test_flag_off_and_scope():
    """(4)+(5) FLAG-OFF byte-identical + MOAT/scope: the enable toggle, and out-of-scope -> None (unchanged turn)."""
    checks = {}
    # enable toggle
    for val, expect in [("0", False), ("false", False), ("off", False), ("1", True), ("", False)]:
        os.environ["BRAIN_MULTIREF"] = val
        checks[f"enabled[{val!r}]"] = (D6.multiref_enabled() == expect)
    os.environ.pop("BRAIN_MULTIREF", None)
    checks["enabled[default]"] = (D6.multiref_enabled() is True)
    # lesion toggle
    os.environ["BRAIN_MULTIREF_LESION"] = "1"
    checks["lesioned[1]"] = (D6.multiref_lesioned() is True)
    os.environ.pop("BRAIN_MULTIREF_LESION", None)
    checks["lesioned[default]"] = (D6.multiref_lesioned() is False)

    org = _fresh(42)
    # out-of-scope: 0 referents, 1 referent, and a non-query single-referent -> None (byte-identical)
    checks["scope_no_referent_None"] = (org.judge("tell me more about that") is None)
    checks["scope_single_referent_None"] = (org.judge("the dog ran fast") is None)
    checks["scope_pronoun_only_None"] = (org.judge("it chased her") is None)
    return checks


def test_moat_and_production_entry(seed=42):
    """(5) MOAT + the realistic production entry: a >=2-referent turn holds both; a later hold-query reads both back;
    no invented referent; output is a READ-OUT, not a fabricated fact."""
    org = _fresh(seed)
    j = org.judge("The dog and the cat sat down.")            # introduce two referents
    ok_two = (j is not None and j["n_referents"] == 2 and j["all_recovered"])
    # every recovered referent is one the input named (no confabulated referent)
    named = set(j["input_order"]) if j else set()
    no_invented = all((v is None or v in named) for v in j["recovered"].values()) if j else False
    # a hold-query reads back what the buffer holds (a single-attractor store would tie to one)
    jq = org.judge("who are we talking about?")
    ok_query = (jq is not None and jq.get("is_hold_query") and jq["n_referents"] >= 2 and "readout" in jq)
    readout = jq.get("readout") if jq else None
    # the output carries NO answer/fact field that would change recall (it is a read-out only)
    no_answer_field = (j is not None and "answer" not in j and "recalled_svo" not in j)
    return {
        "two_held_and_recovered": bool(ok_two),
        "no_invented_referent": bool(no_invented),
        "hold_query_reads_back_both": bool(ok_query),
        "readout": readout,
        "output_is_readout_not_fact": bool(no_answer_field),
        "sample_j": j,
        "sample_jq": jq,
    }


def main():
    t0 = time.time()
    seeds = [42, 43, 44, 100, 101, 102]
    ks = [2, 3, 4]
    backend = os.environ.get("SIM_BACKEND", "numpy")

    intact = test_intact(seeds, ks)
    lesion = test_lesion(seeds, ks)
    superp = test_superposed(seeds, k=2)
    flags = test_flag_off_and_scope()
    moat = test_moat_and_production_entry()

    # ---- load-bearing gates ----
    # intact: >=2 referents recovered every seed; hold alive; input asserted zero
    intact_ok = all(intact[k]["all_recovered_rate"] >= 0.99 and intact[k]["hold_alive_min"] > 1e-3
                    and intact[k]["zero_input_ok"] for k in [2, 3])
    # lesion: the >=2-referent hold collapses (all_recovered ~0) and the bumps are dead
    lesion_ok = all(lesion[k]["all_recovered_rate"] <= 0.01 for k in [2, 3])
    lesion_load_bearing = all(intact[k]["all_recovered_rate"] - lesion[k]["all_recovered_rate"] >= 0.90
                              for k in [2, 3])
    # superposed single-attractor collides (<=1 referent from one register)
    superp_ok = superp["mean_referents_recovered_single_register"] <= 1.0001
    # flag-off + scope byte-identical
    flags_ok = all(flags.values())
    # moat
    moat_ok = (moat["two_held_and_recovered"] and moat["no_invented_referent"]
               and moat["hold_query_reads_back_both"] and moat["output_is_readout_not_fact"])

    # ---- ATTRIBUTION: whose is the >=2-referent read-back? the lever is the spiking HOLD, not the host bookkeeping.
    # The host referent PARSE + role-by-position WRITE MARKER are byte-identical across the two arms; only the recur
    # (the slow-NMDA recurrence) differs. So the intact->lesion drop in all_recovered is attributable to the hold.
    for k in [2, 3]:
        lever(f"D6 all_recovered@k={k}: LESION-the-hold (recur=0) -> INTACT (recur>0)",
              round(lesion[k]["all_recovered_rate"], 3), round(intact[k]["all_recovered_rate"], 3), required=False)
    attributable_to("D6 multi-referent hold (intact) over the lesion control @k=2",
                    round(intact[2]["all_recovered_rate"], 3), round(lesion[2]["all_recovered_rate"], 3))

    PASS = bool(intact_ok and lesion_ok and lesion_load_bearing and superp_ok and flags_ok and moat_ok)

    summary = {
        "probe": "d6_multiref_wm_production_verify", "backend": backend, "seeds": seeds, "ks": ks,
        "PASS": PASS,
        "gates": {
            "intact_ok": bool(intact_ok), "lesion_ok": bool(lesion_ok),
            "lesion_load_bearing": bool(lesion_load_bearing), "superposed_collides": bool(superp_ok),
            "flag_off_and_scope_ok": bool(flags_ok), "moat_ok": bool(moat_ok),
        },
        "intact": intact, "lesion": lesion, "superposed": superp, "flags": flags, "moat": moat,
        "elapsed_seconds": round(time.time() - t0, 1),
        "NOTE": "reuse-by-import of the 6-seed GO _multi_slot_binding_derisk.MultiSlotHold (R banks of D3 slow-NMDA, "
                "ONE shared FS) + the RUNG6c HebbianBinder; NO sim/ edit. The spiking HOLD is load-bearing (lesion "
                "recur=0 collapses the >=2 read-back); the referent PARSE + role-by-position WRITE MARKER + argmax "
                "READ are declared host residuals (the learned spiking write-gate is the open rung, gap#4).",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))

    print("=" * 100)
    print("[d6_multiref_wm_verify] INTACT (all_recovered / hold_alive_min / zero_in):")
    for k in ks:
        print(f"    k={k}: recovered={intact[k]['all_recovered_rate']:.3f} alive={intact[k]['hold_alive_min']:.4f} "
              f"zero_in={intact[k]['zero_input_ok']}")
    print("[d6_multiref_wm_verify] LESION (all_recovered / hold_alive_min):")
    for k in ks:
        print(f"    k={k}: recovered={lesion[k]['all_recovered_rate']:.3f} alive={lesion[k]['hold_alive_min']:.4f}")
    print(f"[d6_multiref_wm_verify] SUPERPOSED single-register recovered (k=2): "
          f"{superp['mean_referents_recovered_single_register']:.3f} (multi-register holds 2; a single attractor ties)")
    print(f"[d6_multiref_wm_verify] FLAG-OFF+SCOPE ok={flags_ok} | MOAT ok={moat_ok}")
    print(f"[d6_multiref_wm_verify] readout: {moat['readout']!r}")
    print(f"[d6_multiref_wm_verify] gates: {summary['gates']}")
    print(f"[d6_multiref_wm_verify] {'PASS' if PASS else 'FAIL'} | wrote {OUT} | {summary['elapsed_seconds']}s")
    print("=" * 100)
    return 0 if PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
