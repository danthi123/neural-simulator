"""VERIFY HARNESS for `b3_noncontradiction_production_organ` (standalone, numpy-CPU).

Proves the four load-bearing properties of the non-contradiction assertion-gate organ, END-TO-END through a real
`OneBrainComposer` (the production recall substrate), reusing the de-risk's fixtures. Every content test uses
NATURAL, INFLECTED user text ("the dog eats grass") against a substrate that stored the LEMMA ("dog eat grass") --
the realistic production scenario -- so the harness exercises the organ's declared surface-first/lemma-fallback
recall, not a lemma-matched toy input.

  1. INTACT FIRES CORRECTLY: reading the brain's own spiking polarity recall, the organ REJECTS a contradicting
     assertion ("the dog eats grass" when the brain holds "dog does NOT eat grass"), ACCEPTS a consistent
     restatement (negated text -> NEGATE == stored NEGATE), ACCEPTS a novel (unknown) assertion, and returns
     None (out of scope) for a question. Fires BOTH directions (contradicting a NEGATE fact AND an AFFIRM fact).
     Also asserts the lemma-fallback actually mapped "eats"->"eat" (recall_action == "eat").
  2. LESION COLLAPSES IT (load-bearing): with the spiking recall bypassed (`lesion=True`), the SAME contradiction
     is ACCEPTED -> the gate is inert. AND the de-risk's STORAGE lesion (store all AFFIRM) is reconfirmed inline:
     the canonical negation reads "yes" on the substrate and the contradiction slips through.
  3. FLAG-OFF IS BYTE-IDENTICAL: `noncontradiction_enabled()` is False under BRAIN_NONCONTRADICTION_GATE=0, and
     the organ is stateless + read-only -> running the full battery does NOT mutate the composer's stored beliefs
     (the canonical recall is identical before/after) -> the caller skipping `check()` leaves the turn unchanged.
  4. MOAT PRESERVED/STRENGTHENED: an UNKNOWN SVO (no stored belief, or a different stored patient) is ACCEPTED,
     never rejected (the gate never fabricates a belief to contradict) -> the no-confab moat is inverted, not
     weakened. `detect_polarity` + the extractor scope are verified.

Run: SIM_BACKEND=numpy NEURAL_SIM_DISABLE_LLM=1 python -m research.runners._b3_noncontradiction_organ_verify
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("NEURAL_SIM_DISABLE_LLM", "1")

import argparse
import json

from research.runners._burndown_B3_onebrain_negation_moat_derisk import _build, _store, FACTS
from research.runners.b3_noncontradiction_production_organ import (
    get_organ, noncontradiction_enabled, noncontradiction_lesioned, detect_polarity,
    extract_polar_assertion, _action_lemma_candidates,
)
from tools.lab import assert_backend, attributable_to
from tools.verdict import Verdict


def verify_seed(seed: int, D: int) -> dict:
    assert_backend("numpy", "B3 organ verify runs on the CPU (device recorded for audit).")
    org = get_organ(seed=seed)
    comp = _build(seed, D)
    _store(comp, FACTS)                       # fact0 = (dog, eat, grass, NEGATE); fact1 = (cat, eat, fish, AFFIRM)
    recall = comp.ask_yes_no                   # the PRODUCTION spiking polarity recall (cp_firing_states WTA)

    # instrument: the canonical negation genuinely reads "no" on the substrate BEFORE the battery
    canon_before = comp.ask_yes_no("dog", "eat", "grass")

    checks = {}

    # 1a. contradiction of a NEGATE fact, NATURAL INFLECTED text: assert AFFIRM -> REJECT (lemma-fallback eats->eat)
    r = org.check(recall, "the dog eats grass")
    checks["contradict_negate"] = (r is not None and r["reject"] is True
                                   and r["asserted_polarity"] == "AFFIRM" and r["recalled_yn"] == "no")
    checks["lemma_fallback_maps"] = (r is not None and r.get("recall_action") == "eat")
    # 1b. consistent restatement of the NEGATE fact: negated text -> NEGATE == stored -> ACCEPT
    r = org.check(recall, "the dog does not eat grass")
    checks["consistent_negate"] = (r is not None and r["reject"] is False
                                   and r["asserted_polarity"] == "NEGATE" and r["recalled_yn"] == "no")
    # 1c. contradiction of an AFFIRM fact (cat eat fish), n't form: assert NEGATE -> REJECT (other direction)
    r = org.check(recall, "the cat doesn't eat fish")
    checks["contradict_affirm"] = (r is not None and r["reject"] is True
                                   and r["asserted_polarity"] == "NEGATE" and r["recalled_yn"] == "yes")
    # 1d. consistent restatement of the AFFIRM fact, inflected: assert AFFIRM -> ACCEPT
    r = org.check(recall, "the cat eats fish")
    checks["consistent_affirm"] = (r is not None and r["reject"] is False
                                   and r["asserted_polarity"] == "AFFIRM" and r["recalled_yn"] == "yes")
    # 1e. NOVEL (unstored) assertion -> unknown -> ACCEPT (moat: no fabricated rejection)
    r = org.check(recall, "the dog chases ball")
    checks["novel_accept"] = (r is not None and r["reject"] is False and r["recalled_yn"] == "unknown")
    # 1f. QUESTION -> out of scope -> None (byte-identical turn)
    checks["question_out_of_scope"] = (org.check(recall, "what does the dog eat?") is None)
    # 1g. different-patient assertion (surprise's job) -> unknown here -> ACCEPT (clean compose, no B3 overlap)
    r = org.check(recall, "the dog eats meat")
    checks["diff_patient_unknown"] = (r is not None and r["reject"] is False and r["recalled_yn"] == "unknown")

    # 2. LESION (organ recall bypass): the same contradiction is ACCEPTED -> inert
    r = org.check(recall, "the dog eats grass", lesion=True)
    checks["lesion_inert"] = (r is not None and r["reject"] is False and r["recalled_yn"] == "unknown")

    # 2b. STORAGE lesion (de-risk's load-bearing lesion): store all AFFIRM -> canonical reads "yes" -> slip through
    comp_les = _build(seed, D)
    _store(comp_les, [(a, v, p, "AFFIRM") for (a, v, p, _) in FACTS])
    canon_lesion = comp_les.ask_yes_no("dog", "eat", "grass")
    r_les = org.check(comp_les.ask_yes_no, "the dog eats grass")
    checks["storage_lesion_negation_gone"] = (canon_lesion == "yes")
    checks["storage_lesion_slips_through"] = (r_les is not None and r_les["reject"] is False)

    # 3. FLAG-OFF byte-identical: the organ is read-only -> the store is unchanged after the whole battery
    canon_after = comp.ask_yes_no("dog", "eat", "grass")
    checks["read_only_idempotent"] = (canon_before == "no" and canon_after == "no")

    return {"seed": seed, "D": D, "canon_before": canon_before, "canon_lesion": canon_lesion,
            "canon_after": canon_after, "checks": checks, "ok": all(checks.values())}


def verify_flags_and_polarity() -> dict:
    """Env-flag toggles + the host negation detector + morphology fallback (no substrate needed)."""
    checks = {}
    # default-ON
    os.environ.pop("BRAIN_NONCONTRADICTION_GATE", None)
    checks["default_on"] = (noncontradiction_enabled() is True)
    for off in ("0", "false", "no", "off"):
        os.environ["BRAIN_NONCONTRADICTION_GATE"] = off
        checks[f"off_{off}"] = (noncontradiction_enabled() is False)
    os.environ.pop("BRAIN_NONCONTRADICTION_GATE", None)
    # lesion flag default-off
    os.environ.pop("BRAIN_NONCONTRADICTION_LESION", None)
    checks["lesion_default_off"] = (noncontradiction_lesioned() is False)
    os.environ["BRAIN_NONCONTRADICTION_LESION"] = "1"
    checks["lesion_on"] = (noncontradiction_lesioned() is True)
    os.environ.pop("BRAIN_NONCONTRADICTION_LESION", None)
    # negation detector
    checks["pol_affirm"] = (detect_polarity("the dog eats grass") == "AFFIRM")
    checks["pol_not"] = (detect_polarity("a dog does not eat grass") == "NEGATE")
    checks["pol_nt"] = (detect_polarity("the dog doesn't eat grass") == "NEGATE")
    checks["pol_never"] = (detect_polarity("the dog never eats grass") == "NEGATE")
    # morphology fallback: surface first, then the de-inflected candidate
    checks["lemma_eats"] = ("eat" in _action_lemma_candidates("eats") and _action_lemma_candidates("eats")[0] == "eats")
    checks["lemma_chases"] = ("chase" in _action_lemma_candidates("chases"))
    checks["lemma_flies"] = ("fly" in _action_lemma_candidates("flies"))
    checks["lemma_base_ss_kept"] = (_action_lemma_candidates("pass") == ["pass"])  # no bad strip of a -ss base
    # extractor scope
    ex = extract_polar_assertion("the dog eats grass")
    checks["extract_svo"] = (ex is not None and ex[:3] == ("dog", "eats", "grass") and ex[3] == "AFFIRM")
    exn = extract_polar_assertion("the dog does not eat grass")
    checks["extract_neg_svo"] = (exn is not None and exn[3] == "NEGATE" and exn[:3] == ("dog", "eat", "grass"))
    checks["extract_question_none"] = (extract_polar_assertion("what does the dog eat?") is None)
    return {"checks": checks, "ok": all(checks.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_b3_noncontradiction_organ_verify.json")
    args = ap.parse_args()

    fp = verify_flags_and_polarity()
    print("[flags+polarity+morph]", "OK" if fp["ok"] else "FAIL",
          {k: v for k, v in fp["checks"].items() if not v} or "all pass", flush=True)

    rows = []
    for s in args.seeds:
        row = verify_seed(s, args.D)
        rows.append(row)
        failed = {k: v for k, v in row["checks"].items() if not v}
        print(f"[seed {s} D={args.D}] {'OK' if row['ok'] else 'FAIL'}  "
              f"canon(before/lesion/after)={row['canon_before']}/{row['canon_lesion']}/{row['canon_after']}  "
              f"{failed or 'all pass'}", flush=True)

    all_ok = fp["ok"] and all(r["ok"] for r in rows)

    def _all(key):
        return all(r["checks"][key] for r in rows)

    # ATTRIBUTION: the rejections are owned by the SPIKING polarity recall, not a template. Treatment = intact
    # contradiction rejections (both directions, all seeds); control = the SAME probes with the recall LESIONED
    # (bypassed -> "unknown" -> accept). The lesion removes the rejections entirely -> 100% attributable to the
    # recall (the load-bearing claim), not to any host default.
    intact_rejections = sum(int(r["checks"]["contradict_negate"]) + int(r["checks"]["contradict_affirm"])
                            for r in rows)
    lesion_rejections = sum(0 if r["checks"]["lesion_inert"] else 1 for r in rows)   # inert => 0 rejections
    attr = attributable_to("rejection owned by spiking recall (intact vs recall-lesion)",
                           intact_rejections, lesion_rejections)

    # ---- EARN the verdict: preconditions travel with the PASS (tools.verdict.Verdict) ----
    v = Verdict("B3 non-contradiction production organ (verify)")
    v.require("intact rejects contradictions BOTH directions all seeds",
              _all("contradict_negate") and _all("contradict_affirm"),
              note="natural inflected text -> spiking polarity recall -> REJECT")
    v.require("consistent restatements accepted all seeds",
              _all("consistent_negate") and _all("consistent_affirm"),
              note="asserted polarity == stored polarity -> accept (0 over-block)")
    v.require("novel + out-of-scope + diff-patient accepted all seeds",
              _all("novel_accept") and _all("question_out_of_scope") and _all("diff_patient_unknown"),
              note="moat: unknown -> accept, never a fabricated rejection; questions None")
    v.require("lemma fallback maps eats->eat all seeds", _all("lemma_fallback_maps"),
              note="declared surface-first/lemma-fallback recalls the stored lemma on inflected input")
    v.require("LESION goes inert all seeds (recall bypass + storage)",
              _all("lesion_inert") and _all("storage_lesion_slips_through")
              and _all("storage_lesion_negation_gone"),
              note="the spiking recall is load-bearing: bypass it or strip the stored negation -> gate inert")
    v.require("read-only byte-identical all seeds", _all("read_only_idempotent"),
              note="the organ never mutates the store (flag-off turn is unchanged)")
    v.require("flags + polarity + morphology host logic ok", fp["ok"],
              note="default-ON, escape/lesion flags, negation detector, lemma candidates")
    decided = v.decide(go=all_ok)

    summary = {
        "harness": "_b3_noncontradiction_organ_verify",
        "organ": "research/runners/b3_noncontradiction_production_organ.py",
        "reuses": "research/runners/_burndown_B3_onebrain_negation_moat_derisk.py (6-seed GO)",
        "backend": os.environ.get("SIM_BACKEND", "numpy"),
        "seeds": args.seeds, "D": args.D,
        "status": decided["status"],
        "preconditions": decided["preconditions"],
        "undefined_reasons": decided["undefined_reasons"],
        "intact_rejections": int(intact_rejections), "lesion_rejections": int(lesion_rejections),
        "rejection_attributable_to_recall": (float(attr) if attr is not None else None),
        "flags_polarity": fp, "rows": rows,
        "ALL_OK": bool(all_ok),
        "verdict": ("PASS -- the non-contradiction organ FIRES on the production spiking recall from NATURAL "
                    "inflected text (rejects contradictions both directions, accepts consistent/novel/out-of-scope), "
                    "the recall is LOAD-BEARING (organ-lesion + storage-lesion both go inert), it is READ-ONLY "
                    "(byte-identical flag-off), and it is MOAT-SAFE (unknown -> accept, never a fabricated "
                    "rejection)."
                    if all_ok else "FAIL -- see the failed checks above.")
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n==== B3 NON-CONTRADICTION ORGAN VERIFY ====")
    print(f"  ALL_OK = {all_ok}   (flags+polarity+morph: {fp['ok']}; seeds: {[r['ok'] for r in rows]})")
    print(f"  wrote {args.out}")
    if not all_ok:
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    main()
