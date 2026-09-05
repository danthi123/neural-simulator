"""PRODUCTION-FLIP VERIFICATION for Gate-B appraisal-via-interoceptive-afferent (scaffold-retirement backlog
rank 5, `research/coordination/scaffold_retirement_backlog.md`).

`research/findings/2026-09-05-gateB-appraisal-interoceptive-afferent-derisk-GO.md` already earned a 6-seed GO for the
MECHANISM with the flag explicitly forced ON (`BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE=1`) -- a de-risk, flag
default-OFF. THIS runner verifies the FLIP itself: `appraisal_interoceptive_enabled()`
(`research/runners/affect_production_organ.py`) now defaults ON (unset env -> the interoceptive path), and that
change is (1) NO-REGRESSION, (2) LOAD-BEARING NOT HOLLOW, (3) a genuine default change with an intact escape
hatch -- through the REAL production organs and the REAL `webapp.server.brain_chat` handler, not an isolated
stub.

TWO PHASES:

  PHASE 1 (mechanism-level, 6 seeds 42/43/44/100/101/102, via the REAL `AffectProductionOrgan` class):
    (a) the DEFAULT is genuinely ON now (env unset -> `read_differential(...)["mechanism"] ==
        "interoceptive_afferent"`, not inferred from reading the dispatch branch);
    (b) sweeping the appraisal moves the differential + the downstream `tone_level`/`content_plan`/`manner_for`
        with the correct sign in the production-realistic band (|appraisal|>=0.5) and ordered tracking
        (corr>=0.8) -- reproducing the de-risk's own load-bearing proof, now through the DEFAULT (unset) path,
        not an explicit flag=1;
    (c) the EXPLICIT ANTI-HOLLOW pairing the brief calls for: with the NEW relay->ladder synapse lesioned
        (`BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE_LESION=1`), the SAME appraisal sweep that varied
        tone_level/content_plan/manner_for in (b) now produces a CONSTANT downstream read regardless of
        appraisal -- the variation VANISHES, not just "the range shrinks" (a sharper, more literal reading of
        the brief's "byte-identical whether-varied-or-not = HOLLOW" than the de-risk's own <=0.25x-range bound);
    (d) the pre-existing readout lesion (`lesion=True`, `affect_out=0`) still collapses to exactly 0.0 --
        unchanged semantics;
    (e) the ESCAPE HATCH (`BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE=0`) still reaches the untouched original
        host-write code byte-for-byte, verified against the PRE-RECORDED 6-seed `host_diffs` in the existing
        de-risk artifact (`_appraisal_interoceptive_ladder_6seed.json`) -- rollback works on every seed, not
        just seed 42.

  A latent same-process multi-seed confound was found and fixed while building this phase:
  `_appraisal_interoceptive_ladder_derisk.get_ladder()` cached a SINGLE ladder keyed by nothing, so a second
  seed's `AffectProductionOrgan.read_differential()` in the SAME process would have silently reused the FIRST
  seed's neurons. Inert against today's production (every production organ is a single-seed-per-process
  singleton), but exactly the confound class `tests/test_determinism.py::TestSubstrateActuallySeeded` exists to
  catch for a same-process 6-seed loop -- fixed to a dict cache keyed by seed (see that module's own docstring).

  PHASE 2 (INTEGRATED handler-level, the fixed production seed=42 `webapp/server.py` always uses, via the REAL
  `webapp.server.brain_chat` endpoint function, numpy-CPU / `BRAIN_COMPOSER_KIND=rf` real production path):
    (a) NO-REGRESSION: a factual panel (stored / self / unstored / inconsistent) answered IDENTICALLY
        (recalled_svo / abstained / verified) whether the flag is at its NEW default (unset) or explicitly
        reverted to the pre-flip default (env=0) -- the mechanism swap never touches WHICH fact is recalled or
        whether the moat abstains, only how the answer is delivered;
    (b) THE HOLLOW-MOUTH DISCIPLINE, live (mirrors `research/findings/
        2026-09-04-linattn-flip-confirmation-affect-still-hollow-live-NOGO.md`'s method): four FRESH sessions,
        each PRIMED (through real conversational turns, `appraise_text` re-verified >=0.5 in magnitude every
        prime) to a strong POSITIVE or NEGATIVE mood, intact or with the intero-lesion held for the whole
        session, then asked the SAME stored factual query. Requires: intact POS tone_level/forthcomingness !=
        intact NEG (varying the input genuinely changes the live turn) AND lesioned POS == lesioned NEG (the
        variation VANISHES under the lesion) AND recalled_svo IDENTICAL across all four (mood colors HOW, never
        WHAT -- the honesty floor).

  Phase 2 disables the OTHER heavy default-on co-resident faculties unrelated to Gate-B appraisal (worldmodel,
  surprise, metacog, multiref, curiosity, episodic, discourse register, self-schema, source-provenance,
  pragmatic, comprehension-learned-cues, BG-select, spiking-mouth-recall, the GNW workspace buses, the onebrain
  xedge cross-session pool) -- speed only, mirrors `_gnw_bus_default_flip_verify.py`'s own precedent; none of
  them gates which fact a plain SVO query recalls or whether the moat abstains. `BRAIN_AFFECT`/`BRAIN_RICH`
  stay at their real production default throughout.

Run (numpy-CPU, fast rf recall -- no GPU, no queueing needed):
  SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._appraisal_interoceptive_production_flip_verify
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")   # the numpy fast production recall path (~s not ~180s)

from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402  (explicit intact-vs-lesion attribution -- gap#5 discipline)

FLAG = "BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE"
LESION_FLAG = "BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE_LESION"
SEEDS = (42, 43, 44, 100, 101, 102)
HOST_REF_ARTIFACT = (Path(_REPO) / "research" / "findings" / "raw" / "appraisal_interoceptive_ladder" /
                    "_appraisal_interoceptive_ladder_6seed.json")
OUT = (Path(_REPO) / "research" / "findings" / "raw" / "appraisal_interoceptive_ladder" /
       "_production_flip_verify.json")


def _flag(v: "str | None"):
    if v is None:
        os.environ.pop(FLAG, None)
    else:
        os.environ[FLAG] = v


def _lesion(on: bool):
    if on:
        os.environ[LESION_FLAG] = "1"
    else:
        os.environ.pop(LESION_FLAG, None)


def _load_host_reference() -> dict:
    """{seed: [differential, ...]} pre-recorded from the ORIGINAL host-write code by the de-risk battery,
    BEFORE this flip existed -- the golden rollback reference for all 6 seeds (not just seed 42's hardcoded
    `_PRE_EDIT_BASELINE` in the de-risk module)."""
    d = json.loads(HOST_REF_ARTIFACT.read_text())
    out = {}
    for row in d["per_seed"]:
        out[int(row["seed"])] = {"sweep": row["sweep"], "host_diffs": row["host_diffs"]}
    return out


# =============================================================================================================
# PHASE 1 -- mechanism-level, 6 seeds, through the REAL AffectProductionOrgan class.
# =============================================================================================================
def phase1_seed(seed: int, host_ref: dict) -> dict:
    from research.runners._appraisal_interoceptive_ladder_derisk import APPRAISAL_SWEEP, PRODUCTION_REALISTIC_ABS_MIN, _corr

    t0 = time.time()
    _flag(None); _lesion(False)   # the NEW default: unset env -> ON, no lesion
    from research.runners import affect_production_organ as AO
    organ = AO.AffectProductionOrgan(seed=seed)   # a FRESH instance (mirrors the de-risk's own per-seed pattern)

    # (a) the default is genuinely ON now.
    probe = organ.read_differential(0.7, lesion=False)
    default_is_on = (probe.get("mechanism") == "interoceptive_afferent")

    # (b) INTACT sweep + downstream.
    intact = [organ.read_differential(a, lesion=False) for a in APPRAISAL_SWEEP]
    diffs = [r["differential"] for r in intact]
    levels = [AO.tone_level(d) for d in diffs]
    plans = [AO.content_plan(lv) for lv in levels]
    manners = [AO.manner_for(lv, "cat", "sat", "mat") for lv in levels]
    n_sent = [p["max_sentences"] for p in plans]
    corr_new = _corr(APPRAISAL_SWEEP, diffs)
    range_new = float(max(diffs) - min(diffs))
    realistic = [(d, a) for d, a in zip(diffs, APPRAISAL_SWEEP) if abs(a) >= PRODUCTION_REALISTIC_ABS_MIN - 1e-9]
    signs_ok_realistic = all((d > 0) == (a > 0) for d, a in realistic if abs(a) > 1e-9)
    downstream_varies_intact = bool(len(set(n_sent)) > 1 and len(set(manners)) > 1)

    # (d) pre-existing readout lesion (affect_out=0) -- unchanged semantics.
    readout_les = [organ.read_differential(a, lesion=True)["differential"] for a in (0.7, -0.7, 1.0)]
    readout_lesion_collapses = all(v == 0.0 for v in readout_les)

    # (c) EXPLICIT ANTI-HOLLOW: the NEW synapse lesion must make the downstream read CONSTANT across the
    # SAME sweep that varied it above (the literal "vary -> changes; lesion -> vanishes" pairing) -- sharper
    # than a range-shrinks bound, applied THROUGH the production dispatch + the downstream consumers.
    _lesion(True)
    lesioned = [organ.read_differential(a, lesion=False) for a in APPRAISAL_SWEEP]
    _lesion(False)
    diffs_il = [r["differential"] for r in lesioned]
    levels_il = [AO.tone_level(d) for d in diffs_il]
    plans_il = [AO.content_plan(lv) for lv in levels_il]
    manners_il = [AO.manner_for(lv, "cat", "sat", "mat") for lv in levels_il]
    n_sent_il = [p["max_sentences"] for p in plans_il]
    range_il = float(max(diffs_il) - min(diffs_il))
    downstream_collapses_under_lesion = bool(len(set(n_sent_il)) == 1 and len(set(manners_il)) == 1)
    relay_enc_intact = _corr([abs(a) for a in APPRAISAL_SWEEP],
                             [r["relay_rate_vplus"] + r["relay_rate_vminus"] for r in intact])
    relay_enc_lesion = _corr([abs(a) for a in APPRAISAL_SWEEP],
                             [r["relay_rate_vplus"] + r["relay_rate_vminus"] for r in lesioned])
    # WHOSE was the collapse -- the treatment/control pair (intact range vs intero-lesioned range) demands an
    # explicit attribution, not just both numbers sitting one key apart (the gap#5 lesson: attribution_required).
    intero_owns_range = attributable_to(
        "intero_synapse_owns_appraisal_coupling(range intact vs lesion, through AffectProductionOrgan)",
        range_new, range_il)

    # (e) ESCAPE HATCH -- byte-identical to the PRE-RECORDED host-write reference, this seed, post-flip.
    _flag("0")
    off_organ = AO.AffectProductionOrgan(seed=seed)   # fresh instance -- no cached state assumption
    ref = host_ref[seed]
    off_diffs = [off_organ.read_differential(a, lesion=False)["differential"] for a in ref["sweep"]]
    off_mechanism_absent = ("mechanism" not in off_organ.read_differential(0.3, lesion=False))
    escape_exact_match = all(o == h for o, h in zip(off_diffs, ref["host_diffs"]))
    _flag(None); _lesion(False)   # restore the new default for whatever runs next in this process

    row = {
        "seed": int(seed), "elapsed_seconds": round(time.time() - t0, 1),
        "default_is_on": default_is_on,
        "intact_diffs": diffs, "intact_n_sentences": n_sent, "intact_manners": manners,
        "corr_new": corr_new, "range_new": range_new,
        "signs_ok_realistic_band": signs_ok_realistic, "downstream_varies_intact": downstream_varies_intact,
        "readout_lesion_values": readout_les, "readout_lesion_collapses": readout_lesion_collapses,
        "intero_lesion_diffs": diffs_il, "intero_lesion_n_sentences": n_sent_il, "range_intero_lesioned": range_il,
        "downstream_collapses_under_intero_lesion": downstream_collapses_under_lesion,
        "relay_enc_intact": relay_enc_intact, "relay_enc_under_intero_lesion": relay_enc_lesion,
        "intero_synapse_owns_range_frac": intero_owns_range,
        "off_diffs": off_diffs, "off_reference_diffs": ref["host_diffs"], "off_mechanism_absent": off_mechanism_absent,
        "escape_exact_match": escape_exact_match,
    }
    ok = bool(default_is_on and signs_ok_realistic and corr_new >= 0.8 and downstream_varies_intact
             and readout_lesion_collapses and downstream_collapses_under_lesion and relay_enc_lesion >= 0.8
             and off_mechanism_absent and escape_exact_match)
    row["GO"] = ok
    print(f"  [phase1 seed {seed}] default_on={default_is_on} corr={corr_new:+.3f} realistic_signs={signs_ok_realistic} "
          f"downstream_varies={downstream_varies_intact} downstream_collapses_under_lesion={downstream_collapses_under_lesion} "
          f"relay_enc_under_lesion={relay_enc_lesion:+.2f} escape_exact_match={escape_exact_match} "
          f"GO={ok} ({row['elapsed_seconds']}s)", flush=True)
    return row


# =============================================================================================================
# PHASE 2 -- INTEGRATED handler-level, production seed=42, through the REAL webapp.server.brain_chat endpoint.
# =============================================================================================================
STORED_PANEL = [
    ("what does dog chase?", ["dog", "chase", "cat"]),
    ("what does cat eat?", ["cat", "eat", "fish"]),
    ("what does brain use?", ["brain", "use", "spikes"]),
]
UNSTORED_PANEL = ["what does fish fly?", "what does bird sing?"]
INCONSISTENT_PANEL = ["what does dog eat?", "what does cat chase?"]
SELF_PANEL = ["what are you"]

POS_PRIME = "That is wonderful, amazing, fantastic, joyful news, I am so happy and thrilled!"
NEG_PRIME = "That is terrible, horrible, awful, miserable news, I am so sad and hateful!"
QUERY = "what does dog chase?"
QUERY_WANT = ["dog", "chase", "cat"]

# Heavy DEFAULT-ON faculties UNRELATED to the Gate-B appraisal mechanism under test, disabled for speed only --
# mirrors `_gnw_bus_default_flip_verify.py::_handler_escape_byte_identical`'s own "heavy Gate-B organs disabled
# for speed (they run identically on both arms)" precedent, extended to the other default-on co-resident organs
# a full production turn now cold-starts (worldmodel/surprise/metacog/multiref/curiosity/episodic/discourse/
# self-schema/source-provenance/pragmatic/comprehension-learned-cues/BG-select/spiking-mouth-recall/the GNW
# workspace buses/the onebrain xedge cross-session pool). NONE of these gate WHICH fact a simple SVO query
# recalls or whether the moat abstains (each is independently flag-guarded, never-crash-a-turn), so disabling
# them changes nothing this runner measures -- only how many OTHER co-resident bridges cold-start on the first
# turn. BRAIN_AFFECT and BRAIN_RICH are deliberately left at their production default (unset -> ON): they are
# the faculty under test and the surface (RichAnswerComposer forthcomingness) it drives.
_UNRELATED_HEAVY_FLAGS = (
    "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_METACOG", "BRAIN_MULTIREF", "BRAIN_NONCONTRADICTION_GATE",
    "BRAIN_RECONSOLIDATION", "BRAIN_EPISODIC", "BRAIN_EPISODIC_STORE", "BRAIN_CURIOSITY",
    "BRAIN_GNW_BUS", "BRAIN_GNW_2ORGAN", "BRAIN_GNW_3ORGAN", "BRAIN_ONEBRAIN_XEDGE",
    "BRAIN_SELF_SCHEMA", "BRAIN_SOURCE_PROVENANCE_HONESTY", "BRAIN_PRAGMATIC", "BRAIN_COMPREHENSION_GATE",
    "BRAIN_DISCOURSE_REGISTER", "BRAIN_BG_SELECT", "BRAIN_SPIKING_MOUTH_RECALL",
)


def _disable_unrelated_heavy_faculties():
    for k in _UNRELATED_HEAVY_FLAGS:
        os.environ[k] = "0"


def _call(session, message, rich=True):
    from webapp.server import brain_chat, BrainChatRequest as Req
    resp = brain_chat(Req(session=session, message=message, brain="tiny-demo", renderer="stub", rich=rich))
    return json.loads(resp.body.decode("utf-8"))


def phase2_no_regression() -> dict:
    """(a) the factual panel is answered IDENTICALLY (recalled_svo/abstained/verified) whether the flag sits at
    its NEW default (unset) or is explicitly reverted (env=0) -- separate sessions per arm so no session-mood
    state leaks between them; a fresh factual query each time (no priming) keeps the mood neutral in both arms,
    isolating the comparison to the mechanism swap itself."""
    rows = []
    ok = True
    panel = ([(q, w, "stored") for q, w in STORED_PANEL]
            + [(q, None, "unstored") for q in UNSTORED_PANEL]
            + [(q, None, "inconsistent") for q in INCONSISTENT_PANEL]
            + [(q, None, "self") for q in SELF_PANEL])
    for i, (q, want, cls) in enumerate(panel):
        _flag(None)
        new = _call(f"flip-new-{i}", q)
        _flag("0")
        old = _call(f"flip-old-{i}", q)
        _flag(None)
        identical = (new.get("recalled_svo") == old.get("recalled_svo")
                    and new.get("abstained") == old.get("abstained")
                    and new.get("verified") == old.get("verified"))
        answered_ok = True if want is None else (new.get("recalled_svo") == want and not new.get("abstained"))
        moat_ok = True if cls not in ("unstored", "inconsistent") else bool(new.get("abstained") and old.get("abstained"))
        both_answer = bool(new.get("answer")) and bool(old.get("answer"))
        ok = ok and identical and answered_ok and moat_ok and both_answer
        rows.append({"q": q, "cls": cls, "new_recalled": new.get("recalled_svo"), "old_recalled": old.get("recalled_svo"),
                    "new_abstained": new.get("abstained"), "old_abstained": old.get("abstained"),
                    "identical": identical, "answered_ok": answered_ok, "moat_ok": moat_ok, "both_answer": both_answer})
    return {"ok": bool(ok), "rows": rows}


def _prime_until_strong(session: str, prime_msg: str, want_sign: int, max_tries: int = 4) -> dict:
    """Sends `prime_msg` as real conversational turns (through the SAME EMA `_update_session_mood` production
    turns use) until the session's own reported appraisal_valence is a STRONG, correctly-signed mood
    (|valence|>=0.6 -- a safety margin above the 0.5 production-realistic-band floor the de-risk characterized,
    so the anti-hollow read below never lands in the honest relay-rheobase sub-threshold band). Returns the
    last turn's response (for inspection); asserts the empirical mood magnitude in the result."""
    last = None
    for _ in range(max_tries):
        last = _call(session, prime_msg)
        v = float((last.get("affect") or {}).get("appraisal_valence", 0.0))
        if abs(v) >= 0.6 and (v > 0) == (want_sign > 0):
            break
    return last


def phase2_hollow_check() -> dict:
    """(b) the hollow-mouth discipline, live: 4 fresh sessions (pos/neg priming x intact/intero-lesioned),
    then the SAME stored factual query. Requires intact pos != intact neg (varies) AND lesioned pos == lesioned
    neg (vanishes under the lesion) AND recalled_svo identical across all four (mood colors HOW, never WHAT)."""
    arms = {}
    for name, prime, want_sign, lesioned in (
        ("pos_intact", POS_PRIME, +1, False), ("neg_intact", NEG_PRIME, -1, False),
        ("pos_lesioned", POS_PRIME, +1, True), ("neg_lesioned", NEG_PRIME, -1, True),
    ):
        _flag(None); _lesion(lesioned)
        session = f"hollow-{name}"
        prime_resp = _prime_until_strong(session, prime, want_sign)
        query_resp = _call(session, QUERY)
        arms[name] = {
            "prime_valence": float((prime_resp.get("affect") or {}).get("appraisal_valence", 0.0)),
            "recalled_svo": query_resp.get("recalled_svo"), "abstained": query_resp.get("abstained"),
            "answer": query_resp.get("answer"), "n_sentences": query_resp.get("n_sentences"),
            "tone_level": (query_resp.get("affect") or {}).get("tone_level"),
            "forthcomingness": (query_resp.get("affect") or {}).get("forthcomingness"),
        }
    _lesion(False); _flag(None)

    pi, ni = arms["pos_intact"], arms["neg_intact"]
    pl, nl = arms["pos_lesioned"], arms["neg_lesioned"]
    primes_strong = all(abs(a["prime_valence"]) >= 0.6 for a in arms.values())
    varies_intact = bool(pi["tone_level"] != ni["tone_level"] or pi["forthcomingness"] != ni["forthcomingness"])
    vanishes_lesioned = bool(pl["tone_level"] == nl["tone_level"] and pl["forthcomingness"] == nl["forthcomingness"])
    facts_untouched = bool(pi["recalled_svo"] == QUERY_WANT and ni["recalled_svo"] == QUERY_WANT
                          and pl["recalled_svo"] == QUERY_WANT and nl["recalled_svo"] == QUERY_WANT
                          and not any(a["abstained"] for a in arms.values()))
    ok = bool(primes_strong and varies_intact and vanishes_lesioned and facts_untouched)
    return {"ok": ok, "arms": arms, "primes_strong": primes_strong, "varies_intact": varies_intact,
            "vanishes_lesioned": vanishes_lesioned, "facts_untouched": facts_untouched}


# =============================================================================================================
def main():
    t0 = time.time()
    host_ref = _load_host_reference()

    print("=" * 108, flush=True)
    print("  GATE-B APPRAISAL-INTEROCEPTIVE-AFFERENT PRODUCTION FLIP -- VERIFY", flush=True)
    print("=" * 108, flush=True)
    print(f"[phase 1] mechanism-level, {len(SEEDS)} seeds, via the real AffectProductionOrgan class", flush=True)
    p1_rows = [phase1_seed(s, host_ref) for s in SEEDS]

    print("[phase 2] integrated handler-level, seed=42, via the real webapp.server.brain_chat", flush=True)
    _disable_unrelated_heavy_faculties()
    p2_noreg = phase2_no_regression()
    print(f"  [phase2] no-regression panel: {sum(r['identical'] for r in p2_noreg['rows'])}/{len(p2_noreg['rows'])} "
          f"identical, ok={p2_noreg['ok']}", flush=True)
    p2_hollow = phase2_hollow_check()
    print(f"  [phase2] hollow-check: primes_strong={p2_hollow['primes_strong']} varies_intact={p2_hollow['varies_intact']} "
          f"vanishes_lesioned={p2_hollow['vanishes_lesioned']} facts_untouched={p2_hollow['facts_untouched']} "
          f"ok={p2_hollow['ok']}", flush=True)
    for name, a in p2_hollow["arms"].items():
        print(f"      {name:14s} prime_v={a['prime_valence']:+.3f} tone={a['tone_level']} "
              f"forthcoming={a['forthcomingness']} recalled={a['recalled_svo']} n_sent={a['n_sentences']}", flush=True)

    n_go_p1 = sum(1 for r in p1_rows if r["GO"])

    v = Verdict("Gate-B appraisal-via-interoceptive-afferent PRODUCTION FLIP (default-ON)")
    v.require("all 6 seeds ran", len(p1_rows) == len(SEEDS), expect=True)
    v.require("[1-mechanism] default is genuinely ON on every seed (unset env -> interoceptive dispatch)",
              all(r["default_is_on"] for r in p1_rows), expect=True)
    v.require("[1-mechanism] signs correct in the production-realistic band, every seed",
              all(r["signs_ok_realistic_band"] for r in p1_rows), expect=True)
    v.require("[1-mechanism] ordered tracking corr>=0.8, every seed", all(r["corr_new"] >= 0.8 for r in p1_rows),
              expect=True)
    v.require("[2-load-bearing] downstream (tone/content/manner) VARIES intact, every seed",
              all(r["downstream_varies_intact"] for r in p1_rows), expect=True)
    v.require("[2-anti-hollow] downstream COLLAPSES to constant under the intero-synapse lesion, every seed",
              all(r["downstream_collapses_under_intero_lesion"] for r in p1_rows), expect=True)
    v.require("[2-dissociation] relay_enc_under_intero_lesion>=0.8, every seed (still fires, still encodes "
              "the appraisal even while the synapse to the ladder is cut -- a genuine dissociation, not silence)",
              all(r["relay_enc_under_intero_lesion"] >= 0.8 for r in p1_rows), expect=True)
    v.require("[old semantics] readout lesion (affect_out=0) still collapses to exactly 0.0, every seed",
              all(r["readout_lesion_collapses"] for r in p1_rows), expect=True)
    v.require("[3-escape hatch] BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE=0 exact-matches the pre-recorded host "
              "reference, every seed (rollback verified on all 6 seeds, not just 42)",
              all(r["escape_exact_match"] for r in p1_rows), expect=True)
    v.require("[3-escape hatch] no 'mechanism' key leaks into the off-path response (genuinely the old branch)",
              all(r["off_mechanism_absent"] for r in p1_rows), expect=True)
    v.require("[1-no-regression, integrated] factual panel byte-identical recalled_svo/abstained/verified, "
              "new-default vs explicit-off, real handler", p2_noreg["ok"], expect=True)
    v.require("[2-anti-hollow, integrated] priming establishes a strong, correctly-signed mood every arm "
              "(|valence|>=0.6, real handler)", p2_hollow["primes_strong"], expect=True)
    v.require("[2-load-bearing, integrated] intact pos-primed turn genuinely differs from intact neg-primed "
              "(real handler, real answer)", p2_hollow["varies_intact"], expect=True)
    v.require("[2-anti-hollow, integrated] that difference VANISHES under the intero-synapse lesion (real "
              "handler) -- the explicit vary-vs-lesion pairing the brief requires", p2_hollow["vanishes_lesioned"],
              expect=True)
    v.require("[honesty floor, integrated] the recalled fact is IDENTICAL across all 4 mood/lesion arms (mood "
              "colors HOW, never WHAT)", p2_hollow["facts_untouched"], expect=True)
    v.disabled("heavy default-on faculties unrelated to Gate-B appraisal in phase 2 (%s)"
              % ", ".join(_UNRELATED_HEAVY_FLAGS),
              why="speed only -- none gates which fact a plain SVO query recalls or whether the moat abstains "
                  "(each independently flag-guarded, never-crash-a-turn); BRAIN_AFFECT + BRAIN_RICH (the "
                  "faculty under test + the surface it drives) stay at their real production default")

    go = bool(n_go_p1 == len(SEEDS) and p2_noreg["ok"] and p2_hollow["ok"])
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    verdict_str = (
        f"{'GO' if go else 'NO-GO'} -- production flip of Gate-B appraisal-via-interoceptive-afferent "
        f"(BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE default-ON). {n_go_p1}/{len(SEEDS)} seeds pass every mechanism-"
        f"level check (default genuinely on, sign+ordered-tracking in the realistic band, downstream load-"
        f"bearing intact AND provably hollow-under-lesion, the pre-existing readout lesion unchanged, the "
        f"escape hatch exact-matching the pre-recorded host reference). Integrated handler-level "
        f"(webapp.server.brain_chat, seed=42, real production organs): no-regression panel "
        f"{'holds' if p2_noreg['ok'] else 'FAILS'} (recalled_svo/abstained/verified byte-identical whichever "
        f"mechanism realizes the affect read); the live hollow-mouth check "
        f"{'holds' if p2_hollow['ok'] else 'FAILS'} (a strongly-primed positive vs negative mood genuinely "
        f"changes tone_level/forthcomingness through a real conversational turn, and that difference VANISHES "
        f"once the new synapse is lesioned, while the recalled fact never moves)."
    )
    print("\n" + "=" * 108, flush=True)
    print(f"  VERDICT: {verdict_str}", flush=True)
    print(f"  status={decided['status']}", flush=True)
    print("=" * 108, flush=True)

    out = {
        "runner": "_appraisal_interoceptive_production_flip_verify", "go": go, "status": decided["status"],
        "verdict": verdict_str, "n_seeds_go_phase1": n_go_p1, "n_seeds": len(SEEDS),
        "phase1_per_seed": p1_rows, "phase2_no_regression": p2_noreg, "phase2_hollow_check": p2_hollow,
        "preconditions": decided["preconditions"], "undefined_reasons": decided["undefined_reasons"],
        "elapsed_seconds": round(time.time() - t0, 1),
        "flag": FLAG, "lesion_flag": LESION_FLAG,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"  [saved] {OUT} ({out['elapsed_seconds']}s)", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
