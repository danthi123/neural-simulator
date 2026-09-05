"""MEASURE+RETIRE de-risk -- one-brain Stage-2 BUILD-AHEAD (2026-09-04). Follow-on to Stage 1
(research/findings/2026-09-04-onebrain-stage1-qwen-fallback-retire-GO.md), targeting the roadmap's own Stage 2
(research/findings/2026-09-03-one-brain-mouth-integration-ROADMAP.md SS4): retire "Touchpoint A" -- the
Surface-A (production-default, `BRAIN_OPEN_ENDED` unset) open-prose recall fallback
(`RichAnswerComposer._render_one_verified`, research/runners/rich_answer_composer.py:859, falling through to
`chat.renderer.render_svo` on a bounded-transitive spiking-Broca miss).

THIS TASK IS PREP, NOT A LANDED RESULT. It builds the flag (`BRAIN_TOUCHPOINT_A_FACT_CLAUSE`, default OFF, see
`rich_answer_composer._touchpoint_a_fact_clause_enabled`), this measure+retire runner, and the GO-gate below --
then runs only a TINY --smoke (a couple of probes, confirming the wiring imports/parses/executes a real turn).
THE FULL DE-RISK (the real per-touchpoint probe battery, `--n-known 4` matching the precedent measurements) is
DEFERRED to when compute frees, per this task's own instruction. Nothing here claims the mechanism is a GO.

THE MEASURED CONTEXT (re-read, not re-run, from the already-committed artifact -- see rich_answer_composer.py's
own flag-block comment for the exact citation): post recall-gate-fix, 5/6 known-topic Surface-A rows now
return a genuine grounded multi-fact answer with `spiking_hit_count=0` and `render_calls>0` -- i.e. Touchpoint A
(Qwen on a CUDA host / the template-stub under SIM_BACKEND=numpy) renders 100% of this newly-reachable content.
`RELATION_LEXICON` (the closed-class lexicon driving `webapp.wkv_mouth_generator.render_fact_sentence`, the
SAME already-6-seed-GO `SpikingClauseProducer` Surface B's fact-clause fallback already reuses) was
independently measured (research/findings/2026-09-04-recall-gate-reaches-real-ltm-GO.md SS5) to cover 34/34
live relation types in the SAME sampled `wikidata_core_15k` store -- so the candidate fix reuses an
already-validated mechanism rather than building a new one; this runner is what actually exercises it against
Touchpoint A's own probe set (not yet done anywhere in the repo).

THE GO-GATE this runner computes (see `compute_go_gate`), distinguishing STRUCTURAL invariants (must ALWAYS
hold, smoke or full battery -- these are checked below even on the tiny smoke) from the READINESS signal (only
meaningful once the full battery runs):
  STRUCTURAL (anti-cheat; must hold on every probe, every run size):
    1. scope_untouched      -- unknown/dangerous/open_ended/greeting rows are answer-text BYTE-IDENTICAL
                                flag-off vs flag-on, and the fact-clause path is called ZERO times on them.
                                FIXED 2026-09-04 (FAILURE_LOG.md row 112): the Touchpoint-A render call in
                                `rich_answer_composer.py::_render_one_verified` now runs inside
                                `webapp.wkv_mouth_generator._RngIsolation.run` (matching `generate()`'s own
                                `_run()`), so a known-topic turn earlier in the SAME session no longer
                                perturbs a later unrelated turn's RNG-dependent state (e.g. the affect-driven
                                lead-in word) via the #77 reseed footgun.
    2. content_preserved    -- REDEFINED 2026-09-04 (OWNER DECISION -- see FAILURE_LOG.md row 113 and
                                research/coordination/build_ahead_ready.md item #3): NOT "every known-topic
                                row's `supporting_facts` is byte-identical flag-off vs flag-on" (the original
                                definition) -- Touchpoint-A deliberately RESCUES a fact the pre-existing
                                Qwen/template renderer failed to verify-render, and recovering MORE grounded
                                facts is the explicit GOAL of this mechanism, not a regression to guard
                                against. The gate now checks "no fact is LOST": flag-ON's UNION of
                                `supporting_facts` across every known-topic row in the battery must be a
                                SUPERSET of flag-OFF's union (a rescued fact appearing only flag-on is
                                EXPECTED and does not fail the gate; a fact flag-off surfaced that is absent
                                everywhere flag-on WOULD fail it). See `compute_go_gate`'s own docstring for
                                why this must be a battery-wide union, not a per-row comparison.
    3. flag_off_inert       -- the fact-clause render is called ZERO times across every flag-OFF row (proves
                                the flag genuinely gates the call, not merely its visible effect).
  READINESS (informational on a smoke; the actual retirement signal once N is large enough to matter):
    4. touchpoint_a_share_delta -- change in summed `render_calls` (Qwen/template fallback) across known-topic
                                    rows, flag-on minus flag-off (more negative = more of Touchpoint A retired).
    5. fact_clause_engaged      -- how many known-topic rows the new path actually fired on (0 on a tiny/
                                    synthetic smoke store is NOT a failure -- RELATION_LEXICON coverage was
                                    measured against the real wikidata_core_15k sample, not a smoke fixture).
`structural_checks_passed` (1-3) is what this session's own --smoke run must satisfy. 4-5 are reported, never
gate the smoke.

USAGE:
  # tiny CPU smoke (this session's own scope -- a couple of probes, confirms wiring end-to-end):
  CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m research.runners._touchpoint_a_fact_clause_derisk \\
      --smoke --out research/findings/raw/_touchpoint_a_fact_clause_smoke.json

  # the full de-risk (DEFERRED -- queue when compute frees, matching the precedent's own --n-known 4 battery):
  CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m research.runners._touchpoint_a_fact_clause_derisk \\
      --n-known 4 --out research/findings/raw/_touchpoint_a_fact_clause_full.json

REUSE-BY-IMPORT, NOT REIMPLEMENTATION: `_sample_known_topics`/`build_probes` are imported verbatim from the
precedent instrument (`_per_touchpoint_qwen_share_measure.py`); the Surface-A instrumentation wrapper below is
a DELIBERATE near-duplicate of that file's own closure (it cannot be imported -- it is nested inside that
file's `main()`), following the SAME "mirror rather than import, so this file stays independently readable"
convention `tests/test_no_qwen_fallback_flag.py`'s own header states explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""             # CPU-forced, matching the precedent instrument's own budget
os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "1"
os.environ["BRAIN_OPEN_ENDED"] = "0"                # this de-risk is Surface A ONLY -- explicit, never popped
os.environ["BRAIN_RICH_BATCH_RENDER"] = "0"         # the sequential _render_one_verified path -- the flag's own scope

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners._per_touchpoint_qwen_share_measure import (  # noqa: E402
    _sample_known_topics, build_probes,
)

_T0 = time.time()
_RSS_WARN_MB = 3800.0
_RSS_ABORT_MB = 4300.0
_SCOPE_GUARD_CLASSES = ("unknown", "dangerous", "open_ended_opinion", "greeting_social")


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def log(*a):
    print(f"[{time.time() - _T0:7.1f}s][rss={_rss_mb():7.1f}MB]", *a, flush=True)


def _smoke_probes(known_topics, unknown_topics):
    """The TINY probe set for --smoke: exactly one Touchpoint-A candidate (a known-topic recall, the row this
    flag's new branch can fire on) and one scope-guard probe (an unknown topic, which must NEVER engage the
    new path). Deliberately not a slice of `build_probes` (whose first N items are ALL known-topic when N>=1,
    which would never exercise the scope guard in a 2-probe slice)."""
    probes = []
    if known_topics:
        probes.append({"class": "known_factual", "msg": f"Tell me about {known_topics[0]}.", "topic": known_topics[0]})
    if unknown_topics:
        probes.append({"class": "unknown", "msg": f"Tell me about {unknown_topics[0]}.", "topic": unknown_topics[0]})
    return probes


def _install_instrumentation(chat, fc_counter):
    """Pure read-through wraps on THIS session's cached `chat` + `chat.renderer`, mirroring
    `_per_touchpoint_qwen_share_measure.py`'s own `_install_surface_a_instrumentation` (duplicated, not
    imported -- see module docstring), PLUS one new counter on the module-level
    `webapp.wkv_mouth_generator.render_fact_sentence` (patched once per process in `main()`, shared across both
    passes -- `fc_counter` is reset per-turn by the caller, not per-pass, so it composes with the outer patch)."""
    counts = {"spiking_hit": 0, "spiking_miss": 0, "render_calls": 0}
    orig_spiking = chat.spiking_recall_surface

    def _wrapped_spiking(a_, v_, p_):
        r = orig_spiking(a_, v_, p_)
        counts["spiking_hit" if r is not None else "spiking_miss"] += 1
        return r
    chat.spiking_recall_surface = _wrapped_spiking

    renderer_obj = getattr(chat, "renderer", None)
    if renderer_obj is not None:
        for _name in ("render_svo", "render_svo_regen"):
            if hasattr(renderer_obj, _name):
                _orig = getattr(renderer_obj, _name)

                def _make_wrapped(orig_fn):
                    def _wrapped(a_, v_, p_):
                        counts["render_calls"] += 1
                        return orig_fn(a_, v_, p_)
                    return _wrapped
                setattr(renderer_obj, _name, _make_wrapped(_orig))
    return counts


def _install_fact_clause_counter():
    """Module-level counting wrapper on `webapp.wkv_mouth_generator.render_fact_sentence` -- the ONLY way to
    observe this call from outside (it is imported fresh, LOCALLY, inside `_render_one_verified` on every
    invocation -- see that function -- so patching the module attribute is seen by every subsequent call,
    exactly the pattern `_open_ended_qwen_fact_clause_fallback_verify.py`'s own poison-pill relies on). Returns
    (counter_dict, restore_fn); counter_dict must be zeroed by the caller before each turn it wants isolated."""
    import webapp.wkv_mouth_generator as WKV
    orig = WKV.render_fact_sentence
    counter = {"calls": 0, "hits": 0}

    def _wrapped(facts, seed=42):
        counter["calls"] += 1
        r = orig(facts, seed=seed)
        if r:
            counter["hits"] += 1
        return r
    WKV.render_fact_sentence = _wrapped

    def _restore():
        WKV.render_fact_sentence = orig
    return counter, _restore


def _reset_cross_process_affect_wta_cache():
    """Clear a PROCESS-GLOBAL (not session-scoped) cache that would otherwise make the OFF and ON passes'
    SCOPE-GUARDED rows incomparable for a reason that has NOTHING to do with Touchpoint-A.

    DISCOVERED this session (see FAILURE_LOG.md 2026-09-04, the row logging this alongside the two Touchpoint-A
    rows this runner was built to close): `research.runners._affect_marker_wta_derisk.get_reader` caches its
    `AffectMarkerWTA` reader keyed ONLY by raw seed (`_READERS: dict`, module-level). Both passes build their
    session with the SAME `chat.inner.seed`, so `webapp.affect_drives_chat.expression_lead`'s spiking
    lateral-inhibition WTA circuit (board #86 -- the mechanism that picks the affective lead-in marker,
    "Sure -- "/"Wonderful -- "/etc., default-ON in production) is WARM and SHARED across the two passes, and
    its per-read "washout" (`_pool_rates`) is NOT sufficient to make the SAME continuous mood value always
    yield the SAME discrete winner. Measured directly (a 7-probe two-pass trace, both flag-off and flag-on):
    idx=6 (`unknown`, a scope-guarded row) reads the IDENTICAL mood=0.040375 in BOTH passes, but the WTA
    reports "no clean winner" (empty lead) as the SHARED reader's 8th-ever call (the OFF pass, which ran
    first) and "level=1 wins" (`"Sure -- "`) as the reader's 16th-ever call (the ON pass -- OFF's own 8 prior
    calls already consumed the reader's history) -- a call-SEQUENCE-POSITION-dependent decision, not a
    mood-dependent one. This is a genuine, PRE-EXISTING production defect (the SAME `_READERS` singleton is
    shared by every concurrent/sequential live chat session too, since `webapp/server.py`'s own call site never
    overrides `seed`), but fixing it AT THE SOURCE (session-scoping or fully RNG/state-isolating
    `AffectMarkerWTA`, mirroring `webapp.wkv_mouth_generator._RngIsolation`/`affect_drives_chat.
    AffectDrivesWorkspace._isolated`) is a separate, default-ON-production-facing task with its own blast
    radius -- NOT attempted here (flagged as a follow-up).

    For THIS runner's OWN purposes, resetting the reader immediately before EACH pass is a valid,
    non-invasive, TEST-HARNESS-LEVEL isolation fix (no production code, no flag, no default touched): it makes
    both passes' 8th call the reader's OWN 8th call since ITS OWN reset, restoring the apples-to-apples
    comparison `scope_untouched` needs. Best-effort: a missing/renamed helper must never break the battery."""
    try:
        from research.runners._affect_marker_wta_derisk import reset_readers
        reset_readers()
    except Exception:
        pass


def run_pass(S, flag_on, probes, session_suffix, fc_counter, rss_abort=True, evict_after=False):
    """Run the full probe list through ONE fresh session (flag_on fixed for the whole pass), returning the
    per-turn rows. A fresh session + `reset=True` warm-up per pass (never reused across off/on) keeps the two
    passes' discourse state independent, so a `known_followup` ("tell me more") row is directly comparable.
    `evict_after=True` pops this pass's session out of `S._BRAIN_CHATS` before returning -- the off/on passes
    each build an independent tiny-demo ChatBrain and the cache keeps BOTH resident by design (different cache
    keys), so evicting the first pass before building the second bounds peak RSS to roughly one session's
    worth instead of two stacking (the RSS-budget concern this whole file family already discloses).
    Also resets the process-global affect-marker-WTA reader cache first -- see
    `_reset_cross_process_affect_wta_cache`'s own docstring for why."""
    _reset_cross_process_affect_wta_cache()
    os.environ["BRAIN_TOUCHPOINT_A_FACT_CLAUSE"] = "1" if flag_on else "0"
    session = f"touchpoint_a_fc_derisk_{session_suffix}_{'on' if flag_on else 'off'}"
    RENDERER = None
    cache_key = (session, "tiny-demo", (RENDERER or S._default_brain_renderer()).lower())

    def _chat(msg, reset):
        resp = S.brain_chat(S.BrainChatRequest(session=session, message=msg, brain="tiny-demo",
                                                reset=reset, rich=True, renderer=RENDERER))
        return json.loads(bytes(resp.body))

    log(f"[{session}] warm-up ...")
    _warmup = _chat("(warm-up) tell me something.", reset=True)
    log(f"[{session}] warm-up done, renderer={_warmup.get('renderer')}")
    chat_obj = S._BRAIN_CHATS.get(cache_key)
    if chat_obj is None:
        raise RuntimeError(f"could not fetch cached chat for instrumentation -- cache_key mismatch: {cache_key}, "
                            f"available: {list(S._BRAIN_CHATS.keys())}")
    counts = _install_instrumentation(chat_obj, fc_counter)

    rows = []
    for i, probe in enumerate(probes):
        rss_now = _rss_mb()
        if rss_abort and rss_now >= _RSS_ABORT_MB:
            log(f"ABORT: rss={rss_now:.1f}MB >= {_RSS_ABORT_MB}MB -- stopping at {len(rows)}/{len(probes)}")
            break
        counts["spiking_hit"] = counts["spiking_miss"] = counts["render_calls"] = 0
        fc_counter["calls"] = fc_counter["hits"] = 0
        t0 = time.time()
        d = _chat(probe["msg"], reset=False)
        dt = round(time.time() - t0, 2)
        row = {
            "idx": i, "class": probe["class"], "topic": probe.get("topic"), "prompt": probe["msg"],
            "flag_on": flag_on, "answer": d.get("answer"), "abstained": d.get("abstained"),
            "n_sentences": d.get("n_sentences"), "renderer": d.get("renderer"),
            "supporting_facts": d.get("supporting_facts"),
            "spiking_hit_count": counts["spiking_hit"], "spiking_miss_count": counts["spiking_miss"],
            "render_calls": counts["render_calls"],
            "fact_clause_calls": fc_counter["calls"], "fact_clause_hits": fc_counter["hits"],
            "gen_seconds": dt, "rss_mb": round(rss_now, 1),
        }
        rows.append(row)
        log(f"  [{i+1}/{len(probes)}] {probe['class']:20s} {probe['msg']!r:45s} abst={row['abstained']} "
            f"n_sent={row['n_sentences']} spk_hit={row['spiking_hit_count']} render={row['render_calls']} "
            f"fc_calls={row['fact_clause_calls']} fc_hits={row['fact_clause_hits']} ({dt}s)")
    if evict_after and cache_key in S._BRAIN_CHATS:
        del S._BRAIN_CHATS[cache_key]
        log(f"[{session}] evicted from _BRAIN_CHATS cache (rss now {_rss_mb():.1f}MB)")
    return rows


def compute_go_gate(rows_off, rows_on):
    """See module docstring for the criteria. `scope_untouched`/`flag_off_inert` compare rows_off[i] to
    rows_on[i] index-by-index (both passes run the SAME probe list, in the SAME order).

    `content_preserved` (REDEFINED 2026-09-04, OWNER DECISION -- see FAILURE_LOG.md row 113 and
    research/coordination/build_ahead_ready.md item #3): the ORIGINAL definition ("every known-topic row's
    `supporting_facts` is byte-identical flag-off vs flag-on") cannot pass while Touchpoint-A is doing its
    job, because it deliberately RESCUES a fact the pre-existing Qwen/template renderer failed to
    verify-render -- recovering MORE grounded facts is the explicit GOAL, not a regression. Concretely (the
    exact case measured in `research/findings/raw/_touchpoint_a_fact_clause_full.json` before this fix): an
    `--n-known 4` battery's idx=3 `known_factual` row gains a fact flag-on (the rescue); idx=5's
    `known_followup` ("tell me more", SAME topic, SAME un-reset session -- see `run_pass`'s own docstring:
    one session per pass, never reset across probes) then has ONE FEWER fact left to tell flag-on, because
    the rescued fact was already told at idx=3 instead of surviving un-told until idx=5. The fact itself was
    never LOST -- it moved to an EARLIER turn in the same conversation -- but a PER-ROW comparison reads
    idx=5 as content flag-on dropped, which is exactly backwards.

    NEW definition: "no fact is lost". Let `facts_off_all`/`facts_on_all` be the UNION of every
    `supporting_facts` triple across every known-topic (non-scope-guard) row in the OFF/ON pass respectively
    -- a battery-wide union, not per-row and not per-topic, because a `known_followup` row's own reach is
    only meaningful relative to everything already told earlier in the SAME un-reset session. GO iff
    `facts_off_all` is a SUBSET of `facts_on_all` (equivalently: flag-ON's total surfaced set is a SUPERSET
    of flag-OFF's) -- every fact flag-OFF ever surfaced anywhere in the battery must also be surfaced
    somewhere flag-ON, regardless of which turn tells it. A fact present only in `facts_on_all` (a genuine
    rescue, reported as `facts_rescued` below) is EXPECTED and never fails the gate; a fact present only in
    `facts_off_all` (a genuine loss) does."""
    problems_scope, problems_content, problems_inert = [], [], []
    n = min(len(rows_off), len(rows_on))
    known_delta = 0
    fc_engaged = 0
    facts_off_all, facts_on_all = set(), set()
    for i in range(n):
        off, on = rows_off[i], rows_on[i]
        cls = off["class"]
        if off.get("fact_clause_calls", 0) not in (0, None):
            problems_inert.append(f"idx={i} class={cls}: flag-OFF row called fact_clause "
                                   f"{off['fact_clause_calls']}x (must be 0 -- the flag did not gate the call)")
        if cls in _SCOPE_GUARD_CLASSES:
            if on.get("fact_clause_calls", 0) not in (0, None):
                problems_scope.append(f"idx={i} class={cls}: flag-ON called fact_clause "
                                       f"{on['fact_clause_calls']}x on a non-Touchpoint-A class")
            if off.get("answer") != on.get("answer"):
                problems_scope.append(f"idx={i} class={cls}: answer text changed off->on "
                                       f"({off.get('answer')!r} -> {on.get('answer')!r})")
        else:
            facts_off_all.update(tuple(f) for f in (off.get("supporting_facts") or []))
            facts_on_all.update(tuple(f) for f in (on.get("supporting_facts") or []))
            known_delta += (on.get("render_calls") or 0) - (off.get("render_calls") or 0)
            if (on.get("fact_clause_calls") or 0) > 0:
                fc_engaged += 1
    lost_facts = facts_off_all - facts_on_all              # OFF surfaced it, ON never does anywhere -- a real loss
    rescued_facts = facts_on_all - facts_off_all           # ON surfaced it, OFF never does anywhere -- the intended rescue
    if lost_facts:
        problems_content.append(
            f"{len(lost_facts)} fact(s) flag-OFF surfaced somewhere in the battery are ABSENT from flag-ON's "
            f"entire surfaced set (a genuine loss, not a rescue -- flag-on's set must be a SUPERSET of "
            f"flag-off's): {sorted(lost_facts, key=repr)!r}")
    return {
        "n_rows_compared": n,
        "scope_untouched": not problems_scope, "scope_problems": problems_scope,
        "content_preserved": not problems_content, "content_problems": problems_content,
        "facts_off_union_count": len(facts_off_all), "facts_on_union_count": len(facts_on_all),
        "facts_rescued": sorted(rescued_facts, key=repr),   # informational: facts ONLY flag-on ever surfaces
        "flag_off_inert": not problems_inert, "flag_off_problems": problems_inert,
        "structural_checks_passed": not (problems_scope or problems_content or problems_inert),
        "touchpoint_a_render_calls_delta": known_delta,   # negative = fewer Qwen/template calls with flag ON
        "fact_clause_engaged_on_known_rows": fc_engaged,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny 2-probe wiring smoke, not the full battery")
    ap.add_argument("--n-known", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()

    log(f"=== touchpoint_a_fact_clause_derisk smoke={a.smoke} backend={os.environ.get('SIM_BACKEND')} ===")

    import webapp.server as S  # noqa: E402 -- heavy import, after env setup
    from research.runners._open_ended_state_driven_generation_derisk import (  # noqa: E402
        _UNKNOWN_ENTITIES, _QWEN_KNOWN_STORE_UNKNOWN,
    )

    ltm_dir = S._resolve_ltm_bundle()
    log("resolved LTM bundle dir:", ltm_dir)
    known_topics = _sample_known_topics(ltm_dir, a.n_known, seed=a.seed) if ltm_dir else []
    log("known topics sampled:", known_topics)

    if a.smoke:
        probes = _smoke_probes(known_topics, list(_UNKNOWN_ENTITIES))
    else:
        probes = build_probes(known_topics, list(_UNKNOWN_ENTITIES), list(_QWEN_KNOWN_STORE_UNKNOWN))
    log(f"probe set: {len(probes)} turns, classes={sorted({p['class'] for p in probes})}")

    fc_counter, restore_fc = _install_fact_clause_counter()
    tag = "s" if a.smoke else "full"
    try:
        rows_off = run_pass(S, False, probes, tag, fc_counter, rss_abort=True, evict_after=True)
        rows_on = run_pass(S, True, probes, tag, fc_counter, rss_abort=True, evict_after=False)
    finally:
        restore_fc()

    go_gate = compute_go_gate(rows_off, rows_on)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "runner": "_touchpoint_a_fact_clause_derisk", "smoke": a.smoke,
        "complete": True, "backend": os.environ.get("SIM_BACKEND"),
        "ltm_bundle_dir": ltm_dir, "known_topics": known_topics,
        "n_probes": len(probes), "seed": a.seed, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rows_off": rows_off, "rows_on": rows_on, "go_gate": go_gate,
        "wall_seconds": round(time.time() - _T0, 1), "peak_rss_mb": round(_rss_mb(), 1),
        "note": ("TINY SMOKE -- confirms the wiring imports/parses/executes real turns and the structural "
                 "anti-cheat invariants hold; NOT the full de-risk battery (deferred to compute-availability, "
                 "see module docstring)." if a.smoke else
                 "Full battery -- see go_gate.touchpoint_a_render_calls_delta / fact_clause_engaged_on_known_rows "
                 "for the actual retirement signal on this store."),
    }
    out_path.write_text(json.dumps(result, indent=1))
    log(f"wrote {out_path}")
    log(f"structural_checks_passed={go_gate['structural_checks_passed']} "
        f"render_calls_delta={go_gate['touchpoint_a_render_calls_delta']} "
        f"fact_clause_engaged={go_gate['fact_clause_engaged_on_known_rows']}")
    return result


if __name__ == "__main__":
    _result = main()
    sys.exit(0 if _result["go_gate"]["structural_checks_passed"] else 1)
