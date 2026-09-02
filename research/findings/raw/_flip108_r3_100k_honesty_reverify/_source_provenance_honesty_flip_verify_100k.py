"""R3 (board #108 cluster, 2026-09-02): re-verify source-provenance-honesty's #129 flip-gating check against the
BROADER `wikidata_100k` bundle (78,857 facts, vocab 23,914) instead of the shipped `wikidata_core_15k` (15,000
facts, vocab 7,032) that gated its original flip. See `research/findings/2026-09-01-source-provenance-honesty-129-
default-on-flip-GO.md`. This is a VERBATIM copy of `research/runners/_source_provenance_honesty_flip_verify.py`
with exactly two changes: (1) the output path (below) so the original 15k evidence file is never overwritten;
(2) NONE to the code path itself -- this script builds its brain via the REAL `brain_chat` handler / `tiny-demo`
default, which resolves its LTM bundle via `webapp/server.py::_resolve_ltm_bundle()`. Run this script with
`BRAIN_LTM_BUNDLE=<path to wikidata_100k>` set in the environment (the documented explicit-override path
`_resolve_ltm_bundle()` already supports) to point the SAME default-brain load path at the 100k bundle, and it
automatically picks up `webapp/server.py`'s post-R2 production defaults for that path (`enable_codebook_cache=True,
enable_decode_escalation=True`, board #108 R2) since those are threaded into `_load_or_build_ltm_store()` itself.

Run (queued via tools/gpu_queue.sh, NOT run concurrently with any other heavy brain-loading job):
  BRAIN_LTM_BUNDLE=/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k \\
  SIM_BACKEND=numpy .venv/bin/python \\
      research/findings/raw/_flip108_r3_100k_honesty_reverify/_source_provenance_honesty_flip_verify_100k.py

VERBATIM ORIGINAL DOCSTRING BELOW.
---
VERIFY: the `BRAIN_SOURCE_PROVENANCE_HONESTY` default-ON flip (board #129, 2026-09-01 owner auto-flip
directive) through the REAL `webapp.server.brain_chat` handler (in-process, SIM_BACKEND=numpy), 6-seed.

WHY THIS FLIP NOW MATTERS (it did not, before today): `webapp/server.py`'s board-#140 rung
(`BRAIN_SOURCE_MONITORING_FRAMES_HONESTY`, `webapp/source_monitoring_honesty_chat.py`) was flipped default-ON
earlier this session (commit `bedb9ad6e`) -- but that rung's ENTIRE branch lives nested INSIDE
`if _SP.source_provenance_enabled():` (webapp/server.py ~5884), which was still default-OFF. So the #140 flip
was, until this one lands, itself hollow in production: `BRAIN_SOURCE_PROVENANCE_HONESTY` unset -> the organ is
never built -> the #140 branch never runs -> zero effect on any real reply. This runner verifies that flipping
`_DEFAULT_ON = True` in `research/runners/source_provenance_production_organ.py` makes the composed pair
(#129 organ + #140 rung, BOTH now unset=on) genuinely LOAD-BEARING on the live chain-route (GENERATED) reply
text, that the `_LESION` escape collapses it (proving the reply text change rides the LEARNED opponent-
comparator trace, not a host if/else), that the `=0` escape is byte-identical to the pre-flip default-OFF
behavior, and that a battery of already-known (PERCEIVED) facts still recalls correctly with zero fabrication.

THREE CHECKS THE OWNER TASK NAMES, MAPPED TO THIS RUNNER:

  (1) LOAD-BEARING -- section A vs section B below (`prod_default` vs `prod_default_lesioned`): with BOTH env
      vars fully UNSET (the new production default), the chain-route reply text is SWAPPED to the #129 organ's
      own hedge wording ("I believe ..., but I reasoned that myself..."), driven by the monitor's live judged
      label (`provenance.label == 'generated'`); with ONLY `BRAIN_SOURCE_PROVENANCE_HONESTY_LESION=1` added (the
      #129 de-risk's own verified failing-direction anti-cheat: Hebbian plasticity gate held shut at encode),
      the swap collapses back to the pre-flip wording -- the change is driven by the LEARNED trace, not a flag.

  (2) BYTE-IDENTICAL-OFF -- section C vs section D: `BRAIN_SOURCE_PROVENANCE_HONESTY=0` EXPLICITLY (the
      documented reversible escape; never `os.environ.pop`, the staleness trap in FAILURE_LOG) produces output
      byte-identical to a PRE-FLIP EMULATION (this module's own `_DEFAULT_ON` monkey-patched back to `False`,
      env fully unset -- literally today's pre-flip code path) on both the direct-recall AND chain-route turns.

  (3) MOAT-SAFE + NO-REGRESSION -- section E: a battery of the tiny-demo brain's own PRE-TAUGHT facts (built
      into the brain at construction, not taught via chat -- `brain use spikes`, `dog chase cat`, `cat eat
      fish`) still recall correctly (non-abstained, correct object, `verified=True`) under the new default,
      IDENTICALLY to the flag-off baseline -- and a never-taught question still abstains in both, with no new
      fact manufactured, under the new default.

WHY TWO LAYERS AT THE MECHANISM LEVEL TOO (mirrors `_source_monitoring_honesty_flip_verify.py`'s own
justification, reused verbatim here for the same reason): the live handler's #129 organ is a PROCESS-SHARED
SINGLETON hardcoded to seed=42 (`webapp/server.py::_get_source_provenance_organ` ->
`source_provenance_production_organ.get_organ(seed=42, ...)`), an existing, unrelated production constant this
task does not touch. Six identical handler calls against one deterministic organ would be six IDENTICAL results,
not real 6-seed evidence. The genuinely seed-varying check is the mechanism-level sweep (section "MECH", the
#129 de-risk's own registered 6 seeds, directly against `SourceProvenanceHonestyMonitor` + `provenance_framed_
text` -- the EXACT functions the handler calls); the handler demonstration (sections A-E) proves the PLUMBING
+ the env-default semantics once, honestly, at the one seed production actually runs.

MOAT CHECK: across every handler condition, `derived`/`recalled_svo` on the chain-route turn are asserted
unchanged (`derived=True`, `recalled_svo=None`) and the terminal fact stated in the text ("worm") is identical
in every arm -- this flip only ever changes the HEDGE WORDING around an already-produced, already-verified
answer, never WHICH fact is stated.

Run (numpy-CPU, foreground, a few minutes):
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._source_provenance_honesty_flip_verify
"""
from __future__ import annotations

import json
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))   # research/findings/raw/<dir>/file.py -> repo root
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ["BRAIN_CHAT_RENDERER"] = "stub"
logging.getLogger().setLevel(logging.ERROR)

SEEDS = [42, 43, 44, 100, 101, 102]
N_PAIRS = 10

# See `_source_monitoring_honesty_flip_verify.py`'s identical note: B3's `extract_polar_assertion` needs the raw
# multi-word teach sentence; do not silence BRAIN_NONCONTRADICTION_GATE. Leave every OTHER faculty at its OWN
# production default (do not over-silence) -- this runner only varies the #129/#140/lesion env vars.
_QUIET: dict = {}


def _setenv(env):
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# MECHANISM: 6-seed vary+lesion sweep directly against the #129 mechanism (unchanged by this flip; re-confirmed
# here because the flip's whole premise rests on this mechanism being real).
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
            row[key] = {"accuracy": correct / (2 * N_PAIRS), "vary_frac": vary / N_PAIRS}
        out.append(row)
    return out


# ─────────────────────────────────────────────────────────────────────────────────────────────
# HANDLER: through the real webapp.server.brain_chat, at the production singleton's seed=42.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _turn(session, message):
    from webapp.server import brain_chat, BrainChatRequest as Req
    r = brain_chat(Req(session=session, message=message, brain="tiny-demo", renderer="stub", rich=False))
    return json.loads(bytes(r.body).decode("utf-8"))


def _reset_singleton():
    """The #129 organ is a process-shared singleton (`source_provenance_production_organ._ORGAN`); force a
    rebuild between conditions so a `lesion=True` condition doesn't reuse an unlesioned build (get_organ()
    already rebuilds on a (seed,lesion) key change, but resetting explicitly makes the boundary between
    conditions in this runner unambiguous rather than relying on that incidental behavior)."""
    import research.runners.source_provenance_production_organ as _SP
    _SP._ORGAN = None
    _SP._ORGAN_KEY = None


def _teach_chain(tag):
    """Teach the 2-hop chain with the #129 organ forced OFF (isolates the demo to perceived-vs-generated FRAMING
    of content the organ has only ever seen through one route -- see the sibling #140 verify runner's identical
    note for why teaching must happen with the organ off)."""
    _setenv(_QUIET)
    _setenv({"BRAIN_SOURCE_PROVENANCE_HONESTY": "0", "BRAIN_SOURCE_PROVENANCE_HONESTY_LESION": None,
             "BRAIN_SOURCE_MONITORING_FRAMES_HONESTY": None})
    sess = f"sph_verify_{tag}"
    t1 = _turn(sess, "the wolf hunts the deer")
    t2 = _turn(sess, "the deer eats the worm")
    return sess, t1, t2


def handler_conditions():
    """Sections A-D: the chain-route + direct-recall pair under each env condition. `env` is applied AFTER
    teaching (teaching always happens with the organ forced off; see `_teach_chain`)."""
    conditions = {
        # (1) LOAD-BEARING pair -----------------------------------------------------------------
        "A_prod_default": {"BRAIN_SOURCE_PROVENANCE_HONESTY": None, "BRAIN_SOURCE_PROVENANCE_HONESTY_LESION": None,
                            "BRAIN_SOURCE_MONITORING_FRAMES_HONESTY": None},   # fully unset == production going fwd
        "B_prod_default_lesioned": {"BRAIN_SOURCE_PROVENANCE_HONESTY": None,
                                     "BRAIN_SOURCE_PROVENANCE_HONESTY_LESION": "1",
                                     "BRAIN_SOURCE_MONITORING_FRAMES_HONESTY": None},
        # (2) BYTE-IDENTICAL-OFF pair -----------------------------------------------------------
        "C_explicit_off": {"BRAIN_SOURCE_PROVENANCE_HONESTY": "0", "BRAIN_SOURCE_PROVENANCE_HONESTY_LESION": None,
                            "BRAIN_SOURCE_MONITORING_FRAMES_HONESTY": None},
        "D_preflip_emulated": None,   # handled specially: monkey-patches _DEFAULT_ON=False, env fully unset
    }
    out = {}
    import research.runners.source_provenance_production_organ as _SP
    for tag, env in conditions.items():
        _reset_singleton()
        sess, t1, t2 = _teach_chain(tag)
        if tag == "D_preflip_emulated":
            _setenv(_QUIET)
            _setenv({"BRAIN_SOURCE_PROVENANCE_HONESTY": None, "BRAIN_SOURCE_PROVENANCE_HONESTY_LESION": None,
                     "BRAIN_SOURCE_MONITORING_FRAMES_HONESTY": None})
            _orig = _SP._DEFAULT_ON
            _SP._DEFAULT_ON = False   # literal pre-flip code path: unset now reads OFF again
            try:
                assert _SP.source_provenance_enabled() is False, "monkey-patch did not restore pre-flip OFF"
                direct = _turn(sess, "what does the wolf hunt?")
                chain = _turn(sess, "what does the wolf's prey eat?")
            finally:
                _SP._DEFAULT_ON = _orig
                _reset_singleton()
        else:
            _setenv(_QUIET)
            _setenv(env)
            direct = _turn(sess, "what does the wolf hunt?")
            chain = _turn(sess, "what does the wolf's prey eat?")
        out[tag] = {"teach1": t1, "teach2": t2, "direct": direct, "chain": chain}
    return out


def moat_battery():
    """Section E: MOAT-SAFE + NO-REGRESSION. The tiny-demo brain's own PRE-TAUGHT facts (built into the brain at
    construction -- research/runners/brain_chat_tui.py::_build_tiny_demo -- never taught via this session's
    chat) recalled under the new production default vs the explicit-off escape, plus one never-taught probe."""
    _reset_singleton()
    known = [
        ("what does the brain use?", "spikes"),
        ("what does the dog chase?", "cat"),
        ("what does the cat eat?", "fish"),
    ]
    unknown_q = "what does the bird eat?"   # 'bird' is in-vocab but never asserted as an eating relation
    out = {}
    for tag, env in (
        ("on_default", {"BRAIN_SOURCE_PROVENANCE_HONESTY": None, "BRAIN_SOURCE_PROVENANCE_HONESTY_LESION": None,
                         "BRAIN_SOURCE_MONITORING_FRAMES_HONESTY": None}),
        ("off", {"BRAIN_SOURCE_PROVENANCE_HONESTY": "0", "BRAIN_SOURCE_PROVENANCE_HONESTY_LESION": None,
                  "BRAIN_SOURCE_MONITORING_FRAMES_HONESTY": None}),
    ):
        _setenv(_QUIET)
        _setenv(env)
        sess = f"sph_moat_{tag}"
        rows = []
        for q, expect in known:
            r = _turn(sess, q)
            rows.append({"q": q, "expect": expect, **r})
        unk = _turn(sess, unknown_q)
        out[tag] = {"known": rows, "unknown": unk}
    return out


def main():
    from tools.lab import attributable_to
    from tools.verdict import Verdict

    mech = mechanism_sweep()
    hand = handler_conditions()
    moat = moat_battery()

    # ---- MECH verdict lines ----
    mech_unlesioned_acc_ok = all(r["unlesioned"]["accuracy"] >= 0.9 for r in mech)
    mech_unlesioned_vary_ok = all(r["unlesioned"]["vary_frac"] >= 0.9 for r in mech)
    mech_lesioned_acc_collapses = all(r["lesioned"]["accuracy"] <= 0.75 for r in mech)
    mech_lesioned_vary_collapses = all(r["lesioned"]["vary_frac"] <= 0.75 for r in mech)
    mean_vary_unlesioned = sum(r["unlesioned"]["vary_frac"] for r in mech) / len(mech)
    mean_vary_lesioned = sum(r["lesioned"]["vary_frac"] for r in mech) / len(mech)
    attr_vary = attributable_to("mechanism-sweep vary_frac (mean over 6 seeds)",
                                 mean_vary_unlesioned, mean_vary_lesioned)

    A, B, C, D = hand["A_prod_default"], hand["B_prod_default_lesioned"], hand["C_explicit_off"], hand["D_preflip_emulated"]

    def ans(rec):
        return (rec.get("answer") or "")

    def prov_label(rec):
        p = rec.get("provenance") or {}
        return p.get("label")

    # ---- (1) LOAD-BEARING ----
    a_chain_swapped = ("I believe" in ans(A["chain"])) and ("reasoned that myself" in ans(A["chain"]))
    a_chain_label_generated = prov_label(A["chain"]) == "generated"
    b_chain_not_swapped = "I believe" not in ans(B["chain"])
    vary_effect = bool(ans(A["chain"]) != ans(C["chain"]))
    lesion_collapses = bool(ans(B["chain"]) == ans(C["chain"]) != ans(A["chain"]))
    # UNLESIONED-ONLY invariant: A (prod-default) == C (explicit-off) == D (pre-flip-emulated). This is the
    # production-relevant moat-safety property -- normal traffic never sets BRAIN_SOURCE_PROVENANCE_HONESTY_LESION
    # (it defaults off), so this is what a real user's direct-recall turn actually sees, both before and after
    # the flip. B (the diagnostic LESION arm) is DELIBERATELY EXCLUDED here, not weakened: the #129 ledger's own
    # lesion_note already documents that under this lesion "both prov pools read exactly silent... any residual
    # accuracy is a noisy host tie-break on a fixed RNG stream, reported not gated" -- i.e. a fresh (never-before-
    # judged) PERCEIVED fact CAN land on either label once the pools are silenced, purely from the tie-break RNG.
    # Measured here: B's tie-break happens to read 'generated' for THIS content pattern (d==0.0 exactly, the
    # documented lesion_d_zero signature), spuriously hedging a directly-recalled fact -- reported below as an
    # observation, never gated, exactly the precedent this codebase already set for this mechanism.
    direct_unchanged_unlesioned = ans(A["direct"]) == ans(C["direct"]) == ans(D["direct"])
    b_direct_tie_break_note = ans(B["direct"])

    # ---- (2) BYTE-IDENTICAL-OFF: explicit-off (C) vs pre-flip-emulated (D) ----
    byte_identical_direct = (ans(C["direct"]) == ans(D["direct"])
                              and prov_label(C["direct"]) == prov_label(D["direct"]) is None)
    byte_identical_chain = (ans(C["chain"]) == ans(D["chain"])
                             and prov_label(C["chain"]) == prov_label(D["chain"]) is None)
    c_organ_stayed_off = C["chain"].get("provenance") is None and C["direct"].get("provenance") is None
    d_organ_stayed_off = D["chain"].get("provenance") is None and D["direct"].get("provenance") is None

    # ---- MOAT shape across A/B/C/D ----
    moat_shape_ok = all(
        (rec["chain"].get("derived") is True and rec["chain"].get("recalled_svo") is None)
        for rec in (A, B, C, D)
    )
    moat_fact_unchanged = all("worm" in ans(rec["chain"]) for rec in (A, B, C, D))

    # ---- (3) MOAT-SAFE + NO-REGRESSION (section E) ----
    on_rows, off_rows = moat["on_default"]["known"], moat["off"]["known"]
    known_correct_on = all((not r.get("abstained")) and r.get("verified") and r["expect"] in ans(r) for r in on_rows)
    known_correct_off = all((not r.get("abstained")) and r.get("verified") and r["expect"] in ans(r) for r in off_rows)
    known_answers_match = all(ans(a) == ans(b) for a, b in zip(on_rows, off_rows))
    unknown_abstains_on = bool(moat["on_default"]["unknown"].get("abstained"))
    unknown_abstains_off = bool(moat["off"]["unknown"].get("abstained"))
    no_fabrication = bool(unknown_abstains_on and unknown_abstains_off)

    go = bool(
        mech_unlesioned_acc_ok and mech_unlesioned_vary_ok and mech_lesioned_acc_collapses
        and mech_lesioned_vary_collapses
        and a_chain_swapped and a_chain_label_generated and b_chain_not_swapped
        and vary_effect and lesion_collapses and direct_unchanged_unlesioned
        and byte_identical_direct and byte_identical_chain and c_organ_stayed_off and d_organ_stayed_off
        and moat_shape_ok and moat_fact_unchanged
        and known_correct_on and known_correct_off and known_answers_match
        and no_fabrication
    )

    v = Verdict("BRAIN_SOURCE_PROVENANCE_HONESTY (board #129) default-ON flip -- handler re-verify")
    v.require("(mech-a) mechanism UNLESIONED accuracy >=0.9 on all 6 seeds", mech_unlesioned_acc_ok, expect=True,
              note=str([r["unlesioned"]["accuracy"] for r in mech]))
    v.require("(mech-b) mechanism UNLESIONED vary_frac >=0.9 on all 6 seeds", mech_unlesioned_vary_ok, expect=True,
              note=str([r["unlesioned"]["vary_frac"] for r in mech]))
    v.require("(mech-c) mechanism LESIONED accuracy collapses <=0.75 on all 6 seeds", mech_lesioned_acc_collapses,
              expect=True, note=str([r["lesioned"]["accuracy"] for r in mech]))
    v.require("(mech-d) mechanism LESIONED vary_frac collapses <=0.75 on all 6 seeds", mech_lesioned_vary_collapses,
              expect=True, note=str([r["lesioned"]["vary_frac"] for r in mech]))
    v.require("(1a) LOAD-BEARING: prod-default (env fully unset) chain reply SWAPPED to the organ's own hedge",
              a_chain_swapped, expect=True, note=ans(A["chain"]))
    v.require("(1b) LOAD-BEARING: swap driven by the organ's OWN live label == 'generated'", a_chain_label_generated,
              expect=True, note=str(A["chain"].get("provenance")))
    v.require("(1c) LOAD-BEARING: LESIONED prod-default chain reply is NOT swapped", b_chain_not_swapped,
              expect=True, note=ans(B["chain"]))
    v.require("(1d) LOAD-BEARING VARY: prod-default chain text != explicit-off chain text", vary_effect,
              expect=True, note=f"A={ans(A['chain'])!r} C={ans(C['chain'])!r}")
    v.require("(1e) LOAD-BEARING LESION collapses the vary-effect: lesioned == off != unlesioned", lesion_collapses,
              expect=True, note=f"B={ans(B['chain'])!r} C={ans(C['chain'])!r} A={ans(A['chain'])!r}")
    v.require("(1f) direct-recall (PERCEIVED) reply text unchanged A(unlesioned prod-default)==C(off)==D(pre-flip) "
              "-- the production-relevant moat check (lesion defaults OFF in real traffic; the diagnostic LESION "
              "arm B is a documented noisy tie-break, reported not gated -- see note)",
              direct_unchanged_unlesioned, expect=True,
              note=f"A={ans(A['direct'])!r} C={ans(C['direct'])!r} D={ans(D['direct'])!r} || OBSERVED (not gated) "
                   f"B(LESIONED)={b_direct_tie_break_note!r} -- a fresh PERCEIVED fact's tie-break landed on "
                   f"'generated' under the deliberately-silenced lesion pools (d==0.0), matching the #129 ledger's "
                   f"own documented lesion-arm noise; lesion defaults OFF in production so this never reaches a "
                   f"real user.")
    v.require("(2a) BYTE-IDENTICAL-OFF: explicit BRAIN_..=0 direct reply == pre-flip-emulated direct reply",
              byte_identical_direct, expect=True, note=f"C={ans(C['direct'])!r} D={ans(D['direct'])!r}")
    v.require("(2b) BYTE-IDENTICAL-OFF: explicit BRAIN_..=0 chain reply == pre-flip-emulated chain reply",
              byte_identical_chain, expect=True, note=f"C={ans(C['chain'])!r} D={ans(D['chain'])!r}")
    v.require("(2c) explicit-off: organ never built (no 'provenance' key) on direct+chain", c_organ_stayed_off,
              expect=True)
    v.require("(2d) pre-flip-emulated: organ never built (no 'provenance' key) on direct+chain", d_organ_stayed_off,
              expect=True)
    v.require("(moat-a) shape: derived=True/recalled_svo=None unchanged in A/B/C/D", moat_shape_ok, expect=True)
    v.require("(moat-b) content: stated terminal fact ('worm') identical in A/B/C/D", moat_fact_unchanged,
              expect=True)
    v.require("(3a) NO-REGRESSION: known-fact battery all correct+verified+non-abstained under prod-default",
              known_correct_on, expect=True, note=str(on_rows))
    v.require("(3b) NO-REGRESSION: same battery all correct+verified+non-abstained under explicit-off", known_correct_off,
              expect=True, note=str(off_rows))
    v.require("(3c) NO-REGRESSION: known-fact answer text identical prod-default vs off (recall unaffected)",
              known_answers_match, expect=True)
    v.require("(3d) MOAT: a never-taught question still abstains under BOTH prod-default and off (no fabrication)",
              no_fabrication, expect=True,
              note=f"on_abstained={unknown_abstains_on} off_abstained={unknown_abstains_off}")
    decided = v.decide(go=go, verbose=False)

    out = {
        "runner": "research/findings/raw/_flip108_r3_100k_honesty_reverify/"
                 "_source_provenance_honesty_flip_verify_100k.py (board #108 R3 re-verify of #129 at 100k scale)",
        "ltm_bundle_env": os.environ.get("BRAIN_LTM_BUNDLE"),
        "flag": "BRAIN_SOURCE_PROVENANCE_HONESTY",
        "flip": "_DEFAULT_ON = True (research/runners/source_provenance_production_organ.py)",
        "attribution": {"vary_frac_attributable_to_lesion": attr_vary},
        "mechanism_sweep_6seed": mech,
        "handler_conditions": hand,
        "moat_battery": moat,
        "GO": bool(decided["go"]),
        "status": decided["status"],
        "preconditions": decided["preconditions"],
    }
    op = os.path.join(_HERE, "source_provenance_honesty_flip_verify_100k.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)

    bar = "=" * 100
    print("\n" + bar)
    print("  BRAIN_SOURCE_PROVENANCE_HONESTY (board #129) default-ON flip -- handler re-verify")
    print(bar)
    for r in mech:
        print(f"  seed {r['seed']:>4}: unlesioned acc={r['unlesioned']['accuracy']:.2f} "
              f"vary={r['unlesioned']['vary_frac']:.2f} | lesioned acc={r['lesioned']['accuracy']:.2f} "
              f"vary={r['lesioned']['vary_frac']:.2f}")
    print(f"  A prod-default        chain: {ans(A['chain'])!r}")
    print(f"  B prod-default+LESION chain: {ans(B['chain'])!r}")
    print(f"  C explicit-off        chain: {ans(C['chain'])!r}")
    print(f"  D pre-flip-emulated   chain: {ans(D['chain'])!r}")
    print(f"\n  VERDICT: {'GO' if decided['go'] else 'NO-GO'}")
    print(f"  [saved] {op}\n" + bar)
    return 0 if decided["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
