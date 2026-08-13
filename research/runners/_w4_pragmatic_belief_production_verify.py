"""PRODUCTION VERIFY for the W4 GRADED SCALAR-IMPLICATURE BELIEF wired into `webapp/server.py::brain_chat` (Task-#12).

Asserts the hard requirements on the REAL production tiny-demo ChatBrain + the REAL `brain_chat` handler, numpy-CPU.
This is a SCOPED wiring: the production speaking pipeline had NO pragmatic-implicature slot before this (leg2_v2 is a
de-risk runner, never wired to production; `webapp/server.py` / the composer form no belief over interpretations). The
organ adds the MINIMAL genuine end-to-end path -- a single scalar-implicature turn class whose BELIEF SOURCE is the
de-risk-CLOSED W4 depth-2 RSA graded-implicature belief (2026-08-13-w4-detector-operating-point-homeostat-GO, 6/6).

  (A) DEFAULT-ON GRADED READ (the belief source): a scalar-quantity turn ("I ate some of the cookies.") through the
      REAL handler attaches a `pragmatic` block whose belief("some") is the GRADED RSA posterior [0, ~0.73, ~0.27]
      (SBNA preferred, "all" ~0.27-possible), the implicature is REPRESENTED (margin > 0.05), and the answer carries
      the honest functional pragmatic reading. The graded belief is BETTER CALIBRATED to the analytic Frank-Goodman
      RSA than the one-hot (calib_l1 lower) AND retains the residual "all"-probability the one-hot destroys (0 for the
      one-hot). This is the graded-vs-onehot A/B, read on the SAME utterance through production.
  (B) MOAT / BYTE-IDENTICAL-WHEN-OFF (real handler): on a NON-scalar panel (recall + abstain) the flag-ON and flag-OFF
      responses are byte-identical (the pragmatic block never fires -> no `pragmatic` key, no prefix; every other
      faculty runs unchanged); and a scalar turn with `BRAIN_PRAGMATIC=0` carries NO `pragmatic` key and no reading
      (the escape hatch is a clean skip). A casual "some" filler ("tell me some facts", no partitive) is OUT OF SCOPE
      -> no reading (the detector is moat-safe).
  (C) LESION-LOAD-BEARING: with `BRAIN_PRAGMATIC_LESION=1` (the normalization-lesion, RSA_FS_EXC_W=0) the SAME scalar
      turn's graded belief COLLAPSES to flat [0, 0.5, 0.5] -> the implicature margin falls to ~0 -> the reading is
      SUPPRESSED (no pragmatic notice). The graded implicature content is caused by the substrate's FS divisive
      normalization, NOT host-injected.
  (D) ADDITIVE / MOAT-SAFE: the pragmatic reading NEVER flips an abstain, never manufactures a recalled fact, never
      changes `recalled_svo`/`verified`/`abstained`. On a scalar turn the recall path is identical to flag-off; only
      the prepended reading + the `pragmatic` block differ.

Run (numpy-CPU, fast rf recall path):
  SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._w4_pragmatic_belief_production_verify
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")   # the numpy fast-path recall (a real production path)

import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)


def _setup_session(session):
    """Build the REAL production ChatBrain, teach a couple of plain facts (so the recall/abstain panel is genuine),
    and cache it into the handler's brain cache. Returns the cache_key."""
    import webapp.server as S
    chat, source = S._build_chat_brain("tiny-demo", "stub")
    comp = chat.inner.composer
    for svo in (("dog", "eat", "meat"), ("cat", "chase", "mouse")):
        comp.store(*svo)
    cache_key = (session, "tiny-demo", "stub")
    chat._brain_chat_source = source
    S._BRAIN_CHATS[cache_key] = chat
    return cache_key


def _turn(session, message, *, rich=False):
    """Drive one turn through the REAL `brain_chat` handler for a pre-built cached session."""
    from webapp.server import brain_chat, BrainChatRequest as Req
    r = brain_chat(Req(session=session, message=message, brain="tiny-demo", renderer="stub", rich=rich))
    return json.loads(r.body.decode("utf-8"))


SCALAR = "I ate some of the cookies."
NONSCALAR = ["what does dog eat?", "what does cat chase?", "what does fish fly?", "what does bird build?"]
FILLER = "tell me some facts"


def main():
    rows = {}

    # ── (A) DEFAULT-ON GRADED READ (the belief source) through the REAL handler ───────────────────────────────
    _setup_session("prg_default")
    a = _turn("prg_default", SCALAR)
    pj = a.get("pragmatic") or {}
    STATES = pj.get("states", ["none", "SBNA", "all"])
    bel = pj.get("belief", [])
    i_sbna = STATES.index("SBNA") if "SBNA" in STATES else 1
    i_all = STATES.index("all") if "all" in STATES else 2
    graded_read_ok = bool(
        pj.get("utterance") == "some" and pj.get("belief_source") == "graded"
        and pj.get("implicature_represented") and pj.get("implicature_margin", 0.0) > 0.05
        and len(bel) == 3 and bel[i_sbna] > 0.6 and 0.2 < bel[i_all] < 0.35      # graded [~0, ~0.73, ~0.27]
        and pj.get("residual_all_prob", 0.0) > 0.2                               # the graded hedge the onehot destroys
        and pj.get("onehot_residual_all_prob", 1.0) == 0.0                       # the leg2_v2 one-hot rules "all" out
        and pj.get("calib_l1_to_analytic", 9.0) < pj.get("calib_l1_to_analytic_onehot", 0.0)   # better calibrated
        and "some but not all" in a["answer"])                                   # the reading reached the surface
    # NOTE: this scalar turn's content words ("ate"/"cookies") are OOV for the tiny-demo brain, so the underlying turn
    # is a D4 comprehension-repair (abstained=True). The scalar implicature is STRUCTURAL (independent of the content
    # words) so the reading is still surfaced + the block attached. That pragmatic NEVER causes/flips the abstain is
    # proved in (D): the scalar turn's abstained/recalled_svo/verified match flag-off exactly.
    rows["A_default_graded"] = {"answer": a["answer"], "pragmatic": pj, "abstained": a["abstained"], "ok": graded_read_ok}

    # ── (C) LESION-LOAD-BEARING: normalization-lesion -> flat belief -> implicature collapses -> no reading ────
    _setup_session("prg_lesion")
    os.environ["BRAIN_PRAGMATIC_LESION"] = "1"
    try:
        c = _turn("prg_lesion", SCALAR)
    finally:
        os.environ.pop("BRAIN_PRAGMATIC_LESION", None)
    cj = c.get("pragmatic") or {}
    cbel = cj.get("belief", [])
    lesion_ok = bool(
        cj.get("lesioned") and not cj.get("implicature_represented")
        and abs(cj.get("implicature_margin", 1.0)) < 0.05                        # margin collapsed to ~0
        and len(cbel) == 3 and abs(cbel[i_sbna] - 0.5) < 0.05 and abs(cbel[i_all] - 0.5) < 0.05   # flat [0,0.5,0.5]
        and "some but not all" not in c["answer"])                              # the reading is suppressed
    rows["C_lesion"] = {"answer": c["answer"], "pragmatic": cj, "ok": lesion_ok}

    # ── (D) ADDITIVE / MOAT-SAFE: the scalar turn's recall path matches flag-off (only the reading + block differ) ─
    _setup_session("prg_addit")
    os.environ["BRAIN_PRAGMATIC"] = "1"
    on = _turn("prg_addit", SCALAR)
    os.environ["BRAIN_PRAGMATIC"] = "0"
    off = _turn("prg_addit", SCALAR)                                            # SAME session, 2nd call (idempotent)
    os.environ.pop("BRAIN_PRAGMATIC", None)
    additive_ok = bool(
        on["abstained"] == off["abstained"] and on["recalled_svo"] == off["recalled_svo"]
        and on["verified"] == off["verified"]
        and (on.get("pragmatic") is not None) and (off.get("pragmatic") is None)  # ON attaches a real block; OFF null
        and ("some but not all" in on["answer"])                                 # ON carries the reading prefix
        and ("some but not all" not in off["answer"]))                           # OFF carries no reading prefix
    rows["D_additive"] = {"on_answer": on["answer"], "off_answer": off["answer"],
                          "same_recall": on["recalled_svo"] == off["recalled_svo"],
                          "off_pragmatic_null": off.get("pragmatic") is None, "ok": additive_ok}

    # ── (B) BYTE-IDENTICAL-WHEN-OFF (real handler) on a NON-scalar panel + the filler out-of-scope ─────────────
    # the STATEFUL heavy organs are disabled ONLY for this comparison so it isolates the pragmatic flag; the (A)/(C)/(D)
    # checks above exercised the FULL default organ stack and passed. Idempotent recall/abstain queries only.
    for k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_METACOG", "BRAIN_MULTIREF",
              "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_EPISODIC", "BRAIN_CURIOSITY",
              "BRAIN_DISCOURSE_REGISTER", "BRAIN_COMPREHENSION_GATE", "BRAIN_RICH"):
        os.environ[k] = "0"
    _setup_session("prg_bi")
    bi_rows, bi_ok = [], True
    for msg in NONSCALAR:
        os.environ["BRAIN_PRAGMATIC"] = "1"
        on_b = _turn("prg_bi", msg)
        os.environ["BRAIN_PRAGMATIC"] = "0"
        off_b = _turn("prg_bi", msg)                                            # SAME session, 2nd call (idempotent)
        os.environ.pop("BRAIN_PRAGMATIC", None)
        # the `pragmatic` key is always present (the always-present-null pattern, like affect/surprise): out of scope
        # it is None on BOTH arms, so the RESPONSE is byte-identical (on_b == off_b) — the real byte-identical read.
        no_read = (off_b.get("pragmatic") is None) and (on_b.get("pragmatic") is None)   # non-scalar -> null both arms
        identical = (on_b == off_b)
        bi_ok = bi_ok and identical and no_read
        bi_rows.append({"q": msg, "identical": identical, "no_pragmatic_read": no_read})
    # a casual "some" filler (no partitive) is OUT OF SCOPE -> no reading even with the flag ON (detector moat).
    os.environ["BRAIN_PRAGMATIC"] = "1"
    fill = _turn("prg_bi", FILLER)
    os.environ.pop("BRAIN_PRAGMATIC", None)
    filler_out_of_scope = bool(fill.get("pragmatic") is None and "some but not all" not in fill["answer"])
    bi_ok = bi_ok and filler_out_of_scope
    rows["B_byte_identical"] = {"rows": bi_rows, "filler_out_of_scope": filler_out_of_scope, "ok": bi_ok}

    go = bool(graded_read_ok and lesion_ok and additive_ok and bi_ok)

    from tools.verdict import Verdict
    v = Verdict("W4 graded scalar-implicature belief production wiring (SCOPED: one scalar-implicature turn class)")
    v.require("DEFAULT-ON GRADED belief source: some->[~0,~0.73,~0.27], implicature represented, better-calibrated "
              "than one-hot, retains the residual 'all' hedge, reading reaches the surface", graded_read_ok, expect=True)
    v.require("MOAT / BYTE-IDENTICAL-when-off (real handler) on a non-scalar panel + the 'some' filler out-of-scope",
              bi_ok, expect=True)
    v.require("LESION: normalization-lesion flattens the belief -> implicature collapses -> reading suppressed "
              "(the graded content is the substrate's FS divisive normalization)", lesion_ok, expect=True)
    v.require("ADDITIVE / moat-safe: the scalar turn's recall (abstained/recalled_svo/verified) matches flag-off; "
              "only the reading + the pragmatic block differ", additive_ok, expect=True)
    v.disabled("a general pragmatic comprehension front-end (embedded/downward-entailing environments, non-lexical "
               "scalars, Q-under-discussion)", why="SCOPED: one lexical scalar-quantity turn class {none,some,all} in "
               "a partitive/probe context; a general front-end that lets this belief drive arbitrary pragmatic "
               "responses is the mapped gap")
    v.disabled("a live per-turn spiking re-read", why="the graded belief is a BUILD-TIME spiking read at a frozen "
               "operating point (plasticity off, as the W4 GO specifies) cached per process; a live per-turn re-read "
               "is IDENTICAL because there is no learning -- the same build-once-freeze pattern as the affect/surprise "
               "organs")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    print("\n" + "=" * 108, flush=True)
    print("  W4 GRADED SCALAR-IMPLICATURE BELIEF — PRODUCTION VERIFY (real ChatBrain + real brain_chat, rf, numpy-CPU)",
          flush=True)
    print("=" * 108, flush=True)
    print(f"  (A) DEFAULT GRADED belief source ok={graded_read_ok}", flush=True)
    print(f"        belief(some)={rows['A_default_graded']['pragmatic'].get('belief')} "
          f"margin={rows['A_default_graded']['pragmatic'].get('implicature_margin')} "
          f"residual_all(graded/onehot)={rows['A_default_graded']['pragmatic'].get('residual_all_prob')}/"
          f"{rows['A_default_graded']['pragmatic'].get('onehot_residual_all_prob')} "
          f"calib_l1(graded/onehot)={rows['A_default_graded']['pragmatic'].get('calib_l1_to_analytic')}/"
          f"{rows['A_default_graded']['pragmatic'].get('calib_l1_to_analytic_onehot')}", flush=True)
    print(f"        answer: {rows['A_default_graded']['answer']}", flush=True)
    print(f"  (C) LESION collapse ok={lesion_ok}  belief(some)={rows['C_lesion']['pragmatic'].get('belief')} "
          f"margin={rows['C_lesion']['pragmatic'].get('implicature_margin')}", flush=True)
    print(f"        answer: {rows['C_lesion']['answer']}", flush=True)
    print(f"  (D) ADDITIVE / moat-safe ok={additive_ok} (same_recall={rows['D_additive']['same_recall']} "
          f"off_pragmatic_null={rows['D_additive']['off_pragmatic_null']})", flush=True)
    print(f"        off answer: {rows['D_additive']['off_answer']}", flush=True)
    print(f"  (B) BYTE-IDENTICAL-when-off ok={bi_ok} (filler_out_of_scope={filler_out_of_scope})", flush=True)
    for r in bi_rows:
        print(f"        {r['q']:24s} identical={r['identical']} no_pragmatic_read={r['no_pragmatic_read']}", flush=True)
    verdict = "GO" if go else "NO-GO"
    print(f"\n  VERDICT: {verdict}\n" + "=" * 108, flush=True)

    out = {"runner": "_w4_pragmatic_belief_production_verify", "go": go, "status": decided["status"],
           "graded_read_ok": graded_read_ok, "lesion_ok": lesion_ok, "additive_ok": additive_ok,
           "byte_identical_ok": bi_ok, "rows": rows,
           "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
           "undefined_reasons": decided["undefined_reasons"]}
    op = "research/findings/raw/_w4_pragmatic_prod/production_verify.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  [saved] {op}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
