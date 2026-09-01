"""VERIFY (board #94 frontier, the precise next rung named by 2026-09-01-confidence-forthcomingness-ltm-
elaboration-load-bearing-GO.md): re-test confidence-forthcomingness through the REAL webapp.server.brain_chat
handler, on the LITERAL SHIPPED wikidata_core_15k LTM (BRAIN_LTM_SHIP_DEFAULT semantics), using a question that
routes via the NEW KB-relation router (`_kb_relation_question_route`, closed 2026-09-01, commit 8047b73a) --
the true production-floor re-test the earlier finding explicitly deferred ("not literally the shipped
wikidata_core_15k bundle").

FIXTURE: entity 'asimov_isaac' (10 real facts in the shipped bundle), relation 'employer' -- idiom-routed
question "who does asimov isaac work for?" (independently confirmed to route + recall correctly in
2026-09-01-nl-parser-kb-relation-question-routing-comprehension-GO.md's 29/29 sweep). Chosen because it has
enough OTHER own-agent facts (8) plus a real 2-hop chain-turn (university_of_boston -> country -> u_s_of_a ->
shares_border_with -> canada_portal) for the rich-answer gather to genuinely exceed the NEUTRAL_SENTENCES=4
floor (reach=5), giving the confidence-forthcoming cap real content to trim -- confirmed by direct scratchpad
probes (not committed) before writing this verify.

TWO NEWLY-SURFACED RESIDUALS THIS MEASUREMENT FOUND (both distinct from the NL-parser vocab gap this arc
already closed):

(1) CLAIM-ENTAILMENT-VERIFIER VOCAB/GRAMMAR GAP (`BRAIN_CLAIM_MOAT`, default ON). The claim-level moat
generalization (`ClaimEntailmentVerifier`, wired in `research/runners/brain_chat_tui.py::_verify_claim_set`)
cannot yet correctly clause-parse the StubRenderer's template surface form for underscored multi-word
Wikidata tokens (e.g. "The asimov_isaac employers university_of_boston.") -- it rejects EVERY gathered
sentence (0 kept, all `dropped`) even though the single-triple `chat._verify` independently confirms each one
is grounded and correct (scratchpad probe4, not committed: all 4 gathered facts pass `chat._verify` cleanly).
This finding uses the EXISTING `BRAIN_CLAIM_MOAT=0` escape (documented in brain_chat_tui.py, reverts to the
single-triple verify) to get PAST this residual and reach the confidence-forthcomingness measurement itself --
not a new mechanism, an existing flag. Verified per-seed below (`claim_moat_on_drops_everything` /
`claim_moat_off_restores`).

(2) TieredFactStore.last_trace DOES NOT PROPAGATE THE LTM TIER'S OWN MATCH TRACE (the metacog confidence
read's actual blocker here). `TieredFactStore._tiered()` always calls the BUFFER's own `query_patient` FIRST;
when the buffer abstains and the LTM tier answers, the OVERALL call returns the LTM's patient, but
`composer.last_trace` (== `buffer.last_trace`, via `__getattr__`) is left holding the BUFFER's OWN abstain
trace for that (agent, action) pair -- never updated to reflect that the LTM tier is what actually answered.
`RichAnswerComposer._chain_facts`'s existing "TRACE PRESERVATION" fix (2026-08-27) assumes `last_trace`
correctly reports whether the JUST-MADE `query_patient` call matched, which holds for a plain composer but not
for `TieredFactStore` on an LTM-sourced hop -- so `last_good_trace` is never captured from the (successful)
LTM-answered hops, and the metacog read that runs after `gather()` sees whatever trace the LAST internal probe
(nearly always the buffer's own abstain on some later chain/elaboration lookup) happened to leave behind.
`webapp/server.py` ALREADY detects this exact signature at runtime (issue #184's own warning: "an answer was
produced by a trace-capable composer but the confidence read came back empty this turn ... This is the
plumbing-bug signature (TieredFactStore.__setattr__ ate last_trace for a day the same way)") -- confirmed
firing verbatim in this measurement's scratchpad probe. The result: `confidence_forthcoming.confident` reads
`None` (not `True`) for this turn even though the recall is a clean, unambiguous, bus-committed (organs A/B/C
unanimous) match -- and `apply_cap` treats `None` identically to `False` (the documented SAFE direction), so
the reach is never granted regardless of the turn's true confidence. NOT fixed here: the repair needs
`TieredFactStore` to propagate whichever tier's own trace actually answered (or a per-tier trace registry),
which touches a class every other trace-based mechanism in this arc also depends on (GNW bus corroboration,
self-schema honesty, source-provenance) -- out of scope for this bounded measurement session; banked in
FAILURE_LOG.md and named precisely here for the next session, not silently dropped or characterized as a
"wall" (the actual blocking code path is small and located; the reason it is not patched here is regression
scope on a shared class, not difficulty).

CONSEQUENCE FOR THIS MEASUREMENT: with residual (2) open, the VARY check (high-confidence vs low-confidence
turn differ in n_sentences) does not observe a positive pair on the real shipped-KB path -- `confident` reads
None (never True) for the clean turn, so n_sentences stays at the floor (4) regardless of the turn's true
confidence. The LESION check is consequently MOOT here (nothing positive to collapse) rather than a genuine
lesion-attributable GO. This is reported as an honest PARTIAL/NO-GO on the vary+lesion criterion, not
recharacterized as a pass.

Usage:
  SIM_BACKEND=numpy .venv/bin/python research/findings/raw/_confidence_kb_relation_realtraffic/\\
      verify_confidence_kb_relation_realtraffic.py --seeds 42 \\
      --out research/findings/raw/_confidence_kb_relation_realtraffic/verify_..._seed42.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
for _k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_COMPREHENSION_GATE", "BRAIN_PRAGMATIC",
           "BRAIN_EPISODIC", "BRAIN_MULTIREF", "BRAIN_SELF_INITIATE", "BRAIN_GNW_DELIBERATE", "BRAIN_GNW_MULTISTEP",
           "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_PMEM", "BRAIN_CURIOSITY",
           "BRAIN_DISCOURSE_REGISTER", "BRAIN_AFFECT_DRIVES", "BRAIN_SWAP_DRIVES", "BRAIN_DA_DRIVES",
           "BRAIN_GNW_STOP", "BRAIN_SELF_SCHEMA", "BRAIN_AFFECTIVE_TOM", "BRAIN_GNW_2ORGAN", "BRAIN_GNW_3ORGAN",
           "BRAIN_BG_SELECT", "BRAIN_SILENT_WM", "BRAIN_SPIKING_MOUTH_RECALL"):
    os.environ[_k] = "0"
os.environ.pop("BRAIN_METACOG", None)                          # metacog stays default-ON (the confidence read)
os.environ.pop("BRAIN_CONFIDENCE_FORTHCOMING_FLOOR", None)     # NEVER override -- the true floor
os.environ["BRAIN_ELABORATE_FROM_LTM_SHARD"] = "1"
os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "1"
os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "1"
os.environ.pop("BRAIN_KB_RELATION_QUESTIONS", None)             # default-ON

LTM_BUNDLE = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_core_15k"
Q = "who does asimov isaac work for?"
EXPECTED_SVO = ["asimov_isaac", "employer", "university_of_boston"]

import webapp.server as S                                                            # noqa: E402
from research.runners.brain_chat_tui import ChatBrain, StubRenderer, _build_tiny_demo  # noqa: E402
from research.runners.developed_brain_io import _inner_agent                          # noqa: E402
from research.runners.tiered_fact_store import TieredFactStore                        # noqa: E402
from research.runners.sharded_phasor_store import ShardedPhasorStore                  # noqa: E402


def _real_kb_facts():
    with open(os.path.join(LTM_BUNDLE, "facts.json"), "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {(r["fact"]["agent"], r["fact"]["action"], r["fact"]["patient"]) for r in raw}


_KB_FACTS = _real_kb_facts()


def moat_ok(d):
    facts = d.get("supporting_facts") or []
    return all(tuple(f) in _KB_FACTS for f in facts)


def build_chat(seed):
    agent, aliases, _n = _build_tiny_demo(seed, use_multiturn=True, enable_neural_render=False,
                                          composer_kind="onebrain")
    ltm = ShardedPhasorStore.load(LTM_BUNDLE)
    inner = _inner_agent(agent)
    inner.composer = TieredFactStore(inner.composer, ltm)
    return ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())


_sid = [0]


def ask(chat, session_prefix="s"):
    ck = (f"{session_prefix}{_sid[0]:04d}", "tiny-demo", "stub")
    _sid[0] += 1
    S._BRAIN_CHATS[ck] = chat
    r = S.brain_chat(S.BrainChatRequest(session=ck[0], message=Q, brain="tiny-demo",
                                        reset=False, rich=True, renderer="stub"))
    return json.loads(bytes(r.body))


def run_seed(seed):
    out = {"seed": seed}
    chat = build_chat(seed)

    # --- (0) the claim-entailment-verifier residual, BEFORE/AFTER the existing escape -----------------------
    os.environ["BRAIN_CLAIM_MOAT"] = "1"     # explicit ON (the production default) -- documents the residual
    d_moat_on = ask(chat, session_prefix=f"m{seed}on")
    os.environ["BRAIN_CLAIM_MOAT"] = "0"     # the existing escape used for the rest of this measurement
    d_moat_off = ask(chat, session_prefix=f"m{seed}off")
    out["claim_moat_residual"] = {
        "on_abstained": d_moat_on.get("abstained"), "on_n_sentences": d_moat_on.get("n_sentences"),
        "off_abstained": d_moat_off.get("abstained"), "off_n_sentences": d_moat_off.get("n_sentences"),
        "on_drops_everything": bool(d_moat_on.get("abstained") and d_moat_on.get("n_sentences") == 0),
        "off_restores": bool((not d_moat_off.get("abstained")) and d_moat_off.get("n_sentences", 0) >= 4),
    }

    # --- (1) CLEAN turn (claim moat escaped) -------------------------------------------------------------
    os.environ.pop("BRAIN_METACOG_LESION", None)
    d_clean = ask(chat, session_prefix=f"c{seed}")
    cf_clean = d_clean.get("confidence_forthcoming") or {}
    clean = {
        "recalled_svo": d_clean.get("recalled_svo"), "abstained": d_clean.get("abstained"),
        "n_sentences": d_clean.get("n_sentences"), "confident": cf_clean.get("confident"),
        "reason": cf_clean.get("reason"), "cf": cf_clean, "moat_ok": moat_ok(d_clean),
        "recall_correct": (d_clean.get("recalled_svo") == EXPECTED_SVO),
    }
    out["clean"] = clean

    # --- (2) LESIONED turn (same question, BRAIN_METACOG_LESION=1) ---------------------------------------
    os.environ["BRAIN_METACOG_LESION"] = "1"
    d_lesion = ask(chat, session_prefix=f"l{seed}")
    os.environ.pop("BRAIN_METACOG_LESION", None)
    cf_lesion = d_lesion.get("confidence_forthcoming") or {}
    lesion = {
        "recalled_svo": d_lesion.get("recalled_svo"), "n_sentences": d_lesion.get("n_sentences"),
        "confident": cf_lesion.get("confident"), "reason": cf_lesion.get("reason"),
        "moat_ok": moat_ok(d_lesion),
    }
    out["lesion"] = lesion

    # --- verdict bookkeeping (honest: this measurement's checks, not a forced GO) ------------------------
    checks = {
        "kb_relation_route_recalls_correct_fact": clean["recall_correct"],
        "claim_moat_residual_confirmed": (out["claim_moat_residual"]["on_drops_everything"]
                                          and out["claim_moat_residual"]["off_restores"]),
        "moat_clean_every_arm": bool(clean["moat_ok"] and lesion["moat_ok"]
                                     and out["claim_moat_residual"]["off_restores"]),
        # the HONEST target check -- this is what would need to be True for a real vary+lesion GO. Recorded
        # (not silently dropped) even though it is expected to read False given residual (2) above.
        "vary_confident_high_on_clean_turn": bool(clean["confident"] is True),
        "vary_reach_granted_on_clean_turn": bool(d_clean.get("n_sentences", 0) > 4),
    }
    out["checks"] = checks
    # this measurement's own scope: confirm the TWO residuals precisely + the routing/recall/moat mechanics
    # are correct; the vary+lesion GO criterion is reported honestly, not forced.
    out["measurement_GO"] = bool(checks["kb_relation_route_recalls_correct_fact"]
                                 and checks["claim_moat_residual_confirmed"]
                                 and checks["moat_clean_every_arm"])
    out["vary_lesion_GO"] = bool(checks["vary_confident_high_on_clean_turn"]
                                 and checks["vary_reach_granted_on_clean_turn"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--out", default=os.path.join(_HERE, "verify_confidence_kb_relation_realtraffic.json"))
    args = ap.parse_args()

    import logging
    logging.disable(logging.INFO)

    t0 = time.time()
    per_seed = []
    for seed in args.seeds:
        r = run_seed(seed)
        per_seed.append(r)
        print(f"[{time.time()-t0:.0f}s] seed {seed}: measurement_GO={r['measurement_GO']} "
              f"vary_lesion_GO={r['vary_lesion_GO']} clean_confident={r['clean']['confident']} "
              f"clean_n={r['clean']['n_sentences']} lesion_n={r['lesion']['n_sentences']}", flush=True)
    dt = time.time() - t0

    measurement_all_go = all(r["measurement_GO"] for r in per_seed)
    vary_lesion_all_go = all(r["vary_lesion_GO"] for r in per_seed)

    out = {
        "probe": "confidence_forthcomingness_kb_relation_realtraffic",
        "question": Q, "expected_svo": EXPECTED_SVO, "ltm_bundle": LTM_BUNDLE,
        "flags": "BRAIN_ELABORATE_FROM_LTM_SHARD=1 + BRAIN_CONFIDENCE_FORTHCOMING=1 + BRAIN_LTM_SHIP_DEFAULT=1 "
                "+ BRAIN_KB_RELATION_QUESTIONS default-ON + BRAIN_CLAIM_MOAT=0 escape (residual 1), true floor",
        "backend": os.environ.get("SIM_BACKEND"), "seeds": args.seeds, "n_seeds": len(args.seeds),
        "elapsed_s": dt,
        "measurement_all_GO": measurement_all_go,       # routing/recall/moat mechanics + both residuals confirmed
        "vary_lesion_all_GO": vary_lesion_all_go,        # the ORIGINAL target criterion -- expected False (residual 2)
        "per_seed": per_seed,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print("=" * 100)
    print(f"  measurement_all_GO (routing+recall+moat+both residuals confirmed): {measurement_all_go}")
    print(f"  vary_lesion_all_GO (the original target -- confidence discriminates n_sentences): {vary_lesion_all_go}")
    print("=" * 100)
    print(f"  wrote {os.path.relpath(args.out, _REPO)}  ({dt:.1f}s)")
    return 0 if measurement_all_go else 1


if __name__ == "__main__":
    sys.exit(main())
