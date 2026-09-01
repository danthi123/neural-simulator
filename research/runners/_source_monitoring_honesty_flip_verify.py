"""VERIFY: BRAIN_SOURCE_MONITORING_FRAMES_HONESTY makes the #129 source-provenance organ's OWN judged label
LOAD-BEARING on the live chain-route (GENERATED) reply text (board #140 rung; 2026-09-01 owner task
"source-monitoring-drives-honesty-framing"). See webapp/source_monitoring_honesty_chat.py for the mechanism this
wires and research/findings/2026-09-01-production-default-flip-plan.md row #6 for the gap it closes.

TWO INDEPENDENT LAYERS OF EVIDENCE (deliberately not one 6x-repeated call, see "why two layers" below):

(1) MECHANISM-LEVEL 6-seed vary+lesion sweep, directly against `SourceProvenanceHonestyMonitor` +
    `provenance_framed_text` -- the EXACT functions `webapp/server.py`'s new board-#140 branch calls. Mirrors
    `tests/test_source_provenance_honesty_wirein.py::test_lesioning_the_monitor_collapses_the_perceived_vs_
    generated_distinction` (that test runs ONLY seed 42; this sweep runs the #129 de-risk's own registered
    6 seeds, research/findings/2026-08-25-laneC-source-provenance-opponent-perceived-vs-generated-6seed-GO.md).
    For each seed: N_PAIRS (perceived, generated) fact-pairs are encoded and judged, UNLESIONED and LESIONED
    (`SourceProvenanceHonestyMonitor(seed=s, lesion=...)`, the SAME failing-direction anti-cheat the #129 de-risk
    verified: Hebbian plasticity gate held shut at encode). Two things are measured per seed/lesion-state:
      - accuracy: does judge_fact's label match what was taught (the #129 GO's own headline metric).
      - vary_frac: over the SAME RAW TEXT, does `provenance_framed_text` produce DIFFERENT wording for the
        perceived-taught item vs the generated-taught item of pair i (the DIRECT "same fact, perceived vs
        generated -> different framing" bar the owner's task names, measured at the mechanism level).

(2) THROUGH-THE-REAL-`brain_chat`-HANDLER demonstration -- `webapp.server.brain_chat` (in-process, the actual
    function `/api/brain-chat` calls), teaching a real 2-hop chain ("the wolf hunts the deer" / "the deer eats
    the worm") through the tiny-demo numpy vocabulary (confirmed: {wolf, deer, worm, hunt, eat} are all in the
    live `tiny-demo` composer's 19-word real vocabulary; `research/runners/compositional_chain_route.py`'s own
    worked example uses the identical "X hunts Y; Y eats Z" -> "what does X's prey eat?" shape), then asking (a)
    a DIRECT recall question (PERCEIVED) and (b) the possessive-chain question (GENERATED, routed by
    `compositional_chain_route.resolve_compositional_chain`). Exercises the flag OFF (byte-identical), the flag
    ON unlesioned (the swap fires), and the flag ON lesioned (the swap is suppressed) -- proving the WIRING in
    `webapp/server.py` is correctly plumbed to the mechanism layer (1) already validated statistically.

WHY TWO LAYERS, NOT SIX HANDLER CALLS LABELED "6 SEEDS". The live `/api/brain-chat` handler's #129 organ is a
PROCESS-SHARED SINGLETON hardcoded to seed=42 (`webapp/server.py`'s `_get_source_provenance_organ()` ->
`source_provenance_production_organ.get_organ(seed=42, ...)`) -- an existing, unrelated production constant this
task does not touch. Six identical handler calls against one deterministic organ would return six IDENTICAL
results (the mechanism has no per-call randomness beyond a host coin-flip on a genuine tie, which a clearly-
taught pair essentially never hits) -- reporting that as "6-seed evidence" would be exactly the kind of
overclaim `docs/TERMS.md` exists to catch. The genuinely seed-varying, statistically meaningful check is layer
(1); layer (2) demonstrates the PLUMBING once, honestly, at the one seed production actually runs.

MOAT CHECK: across every condition below, `derived`/`recalled_svo` on the chain-route turn are asserted
unchanged (still `derived=True`, `recalled_svo=None`) and the terminal fact asserted in the text (“worm”) is
identical in every arm -- this flag only ever changes the HEDGE WORDING around an already-produced, already-
verified answer, never WHICH fact is stated.

Run (numpy-CPU, foreground, a few minutes): SIM_BACKEND=numpy .venv/bin/python -m
    research.runners._source_monitoring_honesty_flip_verify
"""
from __future__ import annotations

import json
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ["BRAIN_CHAT_RENDERER"] = "stub"
logging.getLogger().setLevel(logging.ERROR)

SEEDS = [42, 43, 44, 100, 101, 102]
N_PAIRS = 10

# NOTE (found live, 2026-09-01): do NOT silence BRAIN_NONCONTRADICTION_GATE here. `ChatBrain._maybe_acquire`
# (research/runners/brain_chat_tui.py:945) teaches a multi-word assertion ("the wolf hunts the deer") via B3's
# `extract_polar_assertion`, which strips articles/function words down to the 3-content-word SVO; with B3
# disabled it falls back to a LEGACY path that requires the raw input to be EXACTLY 3 whitespace tokens, so an
# article-bearing teach sentence silently fails to store (falls through to an honest abstain on every later
# recall) -- a real trap this runner's own drafting hit, not a defect in the #129/#140 wiring being verified.
# Leave every OTHER unrelated faculty at its OWN production default too (do not over-silence): this runner only
# needs BRAIN_SOURCE_PROVENANCE_HONESTY / BRAIN_SOURCE_MONITORING_FRAMES_HONESTY / the lesion flag to vary.
_QUIET = {}


def _setenv(env):
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# LAYER (1): mechanism-level 6-seed vary+lesion sweep.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def mechanism_sweep():
    from research.runners.source_provenance_honesty import (
        PROVENANCE_GENERATED,
        PROVENANCE_PERCEIVED,
        SourceProvenanceHonestyMonitor,
        provenance_framed_text,
    )
    out = []
    for seed in SEEDS:
        row = {"seed": seed}
        for lesion in (False, True):
            mon = SourceProvenanceHonestyMonitor(seed=seed, lesion=lesion)
            for i in range(N_PAIRS):
                mon.encode_fact(("perc", seed, i), PROVENANCE_PERCEIVED)
                mon.encode_fact(("gen", seed, i), PROVENANCE_GENERATED)
            correct = 0
            vary = 0
            for i in range(N_PAIRS):
                jp = mon.judge_fact(("perc", seed, i))
                jg = mon.judge_fact(("gen", seed, i))
                if jp["label"] == PROVENANCE_PERCEIVED:
                    correct += 1
                if jg["label"] == PROVENANCE_GENERATED:
                    correct += 1
                raw = f"the item {i} is true"
                fp = provenance_framed_text("x", raw, jp["label"])
                fg = provenance_framed_text("x", raw, jg["label"])
                if fp != fg:
                    vary += 1
            key = "lesioned" if lesion else "unlesioned"
            row[key] = {
                "accuracy": correct / (2 * N_PAIRS),
                "vary_frac": vary / N_PAIRS,
            }
        out.append(row)
    return out


# ─────────────────────────────────────────────────────────────────────────────────────────────
# LAYER (2): through the real brain_chat handler.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _turn(session, message):
    from webapp.server import brain_chat, BrainChatRequest as Req
    r = brain_chat(Req(session=session, message=message, brain="tiny-demo", renderer="stub", rich=False))
    return json.loads(bytes(r.body).decode("utf-8"))


def _teach_and_ask(tag, *, provenance_on, smh_on, lesion):
    # TEACH with the #129 organ OFF. The organ is a PROCESS-SHARED SINGLETON keyed by (seed, lesion) --
    # `source_provenance_production_organ.get_organ()` -- not by session, and `encode_fact` is idempotent
    # (first-write-wins per content key). A teach turn's own `gate_svo` ALSO flows through the SAME "SOURCE-
    # PROVENANCE HONESTY" block in webapp/server.py (it is not chain-routed, so it would encode PERCEIVED).
    # Teaching hop2 ("the deer eats the worm") with the organ ON would therefore pre-seed the key the chain
    # question's SAME terminal content later tries to encode as GENERATED -> idempotent no-op -> the chain
    # would (correctly, but confusingly for this demo) read back PERCEIVED, because that content really WAS
    # also directly taught. Keeping the organ OFF during teaching isolates the demonstration to the ONE
    # difference this task is about: perceived (direct recall) vs generated (chain-composed) framing of
    # content the organ has only ever seen through ONE of those two routes.
    _setenv(_QUIET)
    _setenv({"BRAIN_SOURCE_PROVENANCE_HONESTY": None, "BRAIN_SOURCE_PROVENANCE_HONESTY_LESION": None,
             "BRAIN_SOURCE_MONITORING_FRAMES_HONESTY": None})
    sess = f"smh_verify_{tag}"
    t1 = _turn(sess, "the wolf hunts the deer")
    t2 = _turn(sess, "the deer eats the worm")
    _setenv({
        "BRAIN_SOURCE_PROVENANCE_HONESTY": "1" if provenance_on else None,
        "BRAIN_SOURCE_PROVENANCE_HONESTY_LESION": "1" if lesion else None,
        "BRAIN_SOURCE_MONITORING_FRAMES_HONESTY": "1" if smh_on else None,
    })
    direct = _turn(sess, "what does the wolf hunt?")
    chain = _turn(sess, "what does the wolf's prey eat?")
    return {"teach1": t1, "teach2": t2, "direct": direct, "chain": chain}


def handler_demo():
    conditions = {
        "A_flag_on_unlesioned": dict(provenance_on=True, smh_on=True, lesion=False),
        "B_flag_on_lesioned": dict(provenance_on=True, smh_on=True, lesion=True),
        "C_organ_on_flag_off": dict(provenance_on=True, smh_on=False, lesion=False),
        "D_all_off": dict(provenance_on=False, smh_on=False, lesion=False),
    }
    out = {}
    for tag, kw in conditions.items():
        out[tag] = _teach_and_ask(tag, **kw)
    return out


def main():
    from tools.lab import attributable_to
    from tools.verdict import Verdict

    mech = mechanism_sweep()
    hand = handler_demo()

    # ---- LAYER (1) verdict lines ----
    mech_unlesioned_acc_ok = all(r["unlesioned"]["accuracy"] >= 0.9 for r in mech)
    mech_unlesioned_vary_ok = all(r["unlesioned"]["vary_frac"] >= 0.9 for r in mech)
    mech_lesioned_acc_collapses = all(r["lesioned"]["accuracy"] <= 0.75 for r in mech)
    mech_lesioned_vary_collapses = all(r["lesioned"]["vary_frac"] <= 0.75 for r in mech)

    # ---- ATTRIBUTION (tools.lab): what fraction of the unlesioned vary_frac/accuracy is NOT present in the
    # lesioned control -- makes explicit what checks (1c)/(1d)/(2f) below already show qualitatively (the
    # lesion collapses the effect), so a reader can't mistake "both arms measured" for "the lesion owns it".
    mean_vary_unlesioned = sum(r["unlesioned"]["vary_frac"] for r in mech) / len(mech)
    mean_vary_lesioned = sum(r["lesioned"]["vary_frac"] for r in mech) / len(mech)
    mean_acc_unlesioned = sum(r["unlesioned"]["accuracy"] for r in mech) / len(mech)
    mean_acc_lesioned = sum(r["lesioned"]["accuracy"] for r in mech) / len(mech)
    attr_vary = attributable_to("mechanism-sweep vary_frac (mean over 6 seeds)",
                                 mean_vary_unlesioned, mean_vary_lesioned)
    attr_acc = attributable_to("mechanism-sweep accuracy (mean over 6 seeds)",
                                mean_acc_unlesioned, mean_acc_lesioned)

    # ---- LAYER (2) verdict lines ----
    A = hand["A_flag_on_unlesioned"]
    B = hand["B_flag_on_lesioned"]
    C = hand["C_organ_on_flag_off"]
    D = hand["D_all_off"]

    def ans(rec):
        return (rec.get("answer") or "")

    def prov_label(rec):
        p = rec.get("provenance") or {}
        return p.get("label")

    a_direct_unchanged = ans(A["direct"]) == ans(C["direct"]) == ans(D["direct"])
    a_chain_swapped = ("I believe" in ans(A["chain"])) and ("reasoned that myself" in ans(A["chain"]))
    a_chain_label_generated = prov_label(A["chain"]) == "generated"
    b_chain_not_swapped = "I believe" not in ans(B["chain"])
    b_chain_matches_baseline_wording = ("I derived this from" in ans(B["chain"]))
    c_chain_matches_baseline_wording = ("I derived this from" in ans(C["chain"]))
    d_chain_matches_baseline_wording = ("I derived this from" in ans(D["chain"]))
    vary_effect = bool(ans(A["chain"]) != ans(C["chain"]))
    lesion_collapses_handler_swap = bool(ans(B["chain"]) == ans(C["chain"]) != ans(A["chain"]))

    # moat: derived/recalled_svo shape + terminal fact unchanged across every arm
    moat_shape_ok = all(
        (rec["chain"].get("derived") is True and rec["chain"].get("recalled_svo") is None)
        for rec in (A, B, C, D)
    )
    moat_fact_unchanged = all("worm" in ans(rec["chain"]) for rec in (A, B, C, D))

    go = bool(
        mech_unlesioned_acc_ok and mech_unlesioned_vary_ok and mech_lesioned_acc_collapses
        and mech_lesioned_vary_collapses and a_direct_unchanged and a_chain_swapped
        and a_chain_label_generated and b_chain_not_swapped and vary_effect
        and lesion_collapses_handler_swap and moat_shape_ok and moat_fact_unchanged
    )

    v = Verdict("BRAIN_SOURCE_MONITORING_FRAMES_HONESTY (board #140 rung) vary+lesion+moat verify")
    v.require("(1a) mechanism-level UNLESIONED accuracy >=0.9 on all 6 seeds", mech_unlesioned_acc_ok, expect=True,
              note=str([r["unlesioned"]["accuracy"] for r in mech]))
    v.require("(1b) mechanism-level UNLESIONED vary_frac >=0.9 on all 6 seeds (same content, different framing)",
              mech_unlesioned_vary_ok, expect=True, note=str([r["unlesioned"]["vary_frac"] for r in mech]))
    v.require("(1c) mechanism-level LESIONED accuracy collapses <=0.75 on all 6 seeds", mech_lesioned_acc_collapses,
              expect=True, note=str([r["lesioned"]["accuracy"] for r in mech]))
    v.require("(1d) mechanism-level LESIONED vary_frac collapses <=0.75 on all 6 seeds", mech_lesioned_vary_collapses,
              expect=True, note=str([r["lesioned"]["vary_frac"] for r in mech]))
    v.require("(2a) handler: direct-recall (PERCEIVED) reply text unchanged across A/C/D", a_direct_unchanged,
              expect=True, note=f"A={ans(A['direct'])!r} C={ans(C['direct'])!r} D={ans(D['direct'])!r}")
    v.require("(2b) handler: flag ON unlesioned chain reply SWAPPED to the #129 organ's own hedge wording",
              a_chain_swapped, expect=True, note=ans(A["chain"]))
    v.require("(2c) handler: flag ON unlesioned chain provenance.label == 'generated' (organ readback, not "
              "the is_chain_route flag)", a_chain_label_generated, expect=True, note=str(A["chain"].get("provenance")))
    v.require("(2d) handler: flag ON LESIONED chain reply is NOT swapped (falls back to frame_derived_answer)",
              b_chain_not_swapped, expect=True, note=ans(B["chain"]))
    v.require("(2e) handler VARY: flag-ON chain reply text != flag-OFF/organ-off chain reply text", vary_effect,
              expect=True, note=f"A={ans(A['chain'])!r} C={ans(C['chain'])!r}")
    v.require("(2f) handler LESION collapses the vary-effect: lesioned reply == organ-off reply != unlesioned reply",
              lesion_collapses_handler_swap, expect=True,
              note=f"B={ans(B['chain'])!r} C={ans(C['chain'])!r} A={ans(A['chain'])!r}")
    v.require("(2g) MOAT shape: derived=True/recalled_svo=None unchanged in every arm", moat_shape_ok, expect=True)
    v.require("(2h) MOAT content: the stated terminal fact ('worm') is identical in every arm", moat_fact_unchanged,
              expect=True)
    decided = v.decide(go=go, verbose=False)

    out = {
        "runner": "research/runners/_source_monitoring_honesty_flip_verify.py",
        "flag": "BRAIN_SOURCE_MONITORING_FRAMES_HONESTY",
        "attribution": {
            "vary_frac_attributable_to_lesion": attr_vary,
            "accuracy_attributable_to_lesion": attr_acc,
            "note": "fraction of the mean unlesioned effect NOT present in the mean lesioned control "
                    "(tools.lab.attributable_to) -- makes explicit what (1c)/(1d) already show qualitatively.",
        },
        "mechanism_sweep_6seed": mech,
        "handler_demo": hand,
        "GO": bool(decided["go"]),
        "status": decided["status"],
        "preconditions": decided["preconditions"],
    }
    op = os.path.join(_REPO, "research", "findings", "raw", "_source_monitoring_honesty",
                       "flip_verify.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)

    bar = "=" * 100
    print("\n" + bar)
    print("  BRAIN_SOURCE_MONITORING_FRAMES_HONESTY (board #140 rung) — vary+lesion+moat verify")
    print(bar)
    for r in mech:
        print(f"  seed {r['seed']:>4}: unlesioned acc={r['unlesioned']['accuracy']:.2f} "
              f"vary={r['unlesioned']['vary_frac']:.2f} | lesioned acc={r['lesioned']['accuracy']:.2f} "
              f"vary={r['lesioned']['vary_frac']:.2f}")
    print(f"  handler A (flag ON, unlesioned) chain: {ans(A['chain'])!r}")
    print(f"  handler B (flag ON, LESIONED)   chain: {ans(B['chain'])!r}")
    print(f"  handler C (organ ON, flag OFF)  chain: {ans(C['chain'])!r}")
    print(f"  handler D (all OFF)             chain: {ans(D['chain'])!r}")
    print(f"\n  VERDICT: {'GO' if decided['go'] else 'NO-GO'}")
    print(f"  [saved] {op}\n" + bar)
    return 0 if decided["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
