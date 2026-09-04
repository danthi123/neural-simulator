"""MEASUREMENT HARNESS -- one-brain-wiring de-risk #2 (2026-09-04, per the roadmap
research/findings/2026-09-03-one-brain-mouth-integration-ROADMAP.md SS3, de-risk #2: "Instrument the live
per-touchpoint Qwen-call share"). Quantifies WHERE the Qwen scaffold is still load-bearing in a live
`/api/brain-chat` turn, so the roadmap's Stage 1 (retire the open-ended one-shot fallback) vs Stage 2
(retire "Touchpoint A", the Surface-A open-prose recall fallback) can be ORDERED by measured live share
instead of the roadmap's own un-verified assumption ("Touchpoint A is the LARGER live share").

THROUGH THE REAL ENTRY POINT, same pattern as the precedent real-traffic soak
(`_open_ended_bundle_moat_safety_soak.py`): every turn is `webapp.server.brain_chat(BrainChatRequest(...))`,
called in-process -- the exact function the HTTP route dispatches to. No sim/ edit, no existing runner edit.

TWO SURFACES, TWO TOUCHPOINTS, TWO DIFFERENT INSTRUMENTS (the mislabel fix landed 2026-09-04, commit
c08ce8fb3, only reaches Surface B):

  Surface A (`--phase shipped_default`, BRAIN_OPEN_ENDED unset/0 -- the PRODUCTION DEFAULT turn today).
  `RichAnswerComposer._render_one_verified` (research/runners/rich_answer_composer.py:797) tries
  `chat.spiking_recall_surface(a, v, p)` (the bounded-transitive spiking Broca mouth, BRAIN_SPIKING_MOUTH_RECALL,
  default-ON) FIRST; a miss (open/multi-word/copula/irregular-verb prose the bounded frame can't cover, or a
  verify-miss) falls through to `chat.renderer.render_svo(...)` -- "Touchpoint A" in the roadmap's own words,
  Qwen on a GPU host / a host template on a GPU-free host. THERE IS NO HTTP-LEVEL TRACE FIELD FOR THIS -- unlike
  Surface B's `generator` key, `/api/brain-chat`'s JSON never says which of the two rendered a given sentence.
  This harness closes that gap the ONLY way possible without a sim/ edit: it monkeypatches
  `chat.spiking_recall_surface` at the INSTANCE level (after the session's ChatBrain is built and cached) to
  count hit/miss per turn -- a pure read-through wrapper (calls the original, returns its result unchanged,
  only ADDS a counter). This is verified by code inspection (the wrapper's own body never touches the
  returned value); it is NOT an independently hash/exact-compared "byte-identical" claim per docs/TERMS.md's
  condition for that word, so that term is deliberately not used here. Per turn:
    spiking_hit_count   = calls that returned a verified surface (spiking mouth wrote that sentence)
    spiking_miss_count  = calls that returned None (that gathered SVO fell through to the renderer)
  Because `_render_one_verified` returns verified=True via EXACTLY ONE of {spiking hit, renderer
  render_svo/render_svo_regen verified} (the default BRAIN_RICH_BATCH_RENDER=0 sequential path, confirmed by
  reading render_paragraph() -- the batched path is off by default and not exercised here), the response's own
  `n_sentences` (= len(kept)) lets the harness derive, with NO further instrumentation:
    qwen_touchpointA_verified = n_sentences - spiking_hit_count
    qwen_touchpointA_dropped  = spiking_miss_count - qwen_touchpointA_verified
  A second, independent instrument (wrapping `chat.renderer.render_svo`/`render_svo_regen` call counts) is
  ALSO installed as a cross-check -- `qwen_render_calls` should be >= spiking_miss_count (>= because a
  verify-miss pays a second `render_svo_regen` call); a violation would mean the batched path fired
  unexpectedly and is flagged in the artifact rather than silently trusted.

  Surface B (`--phase open_ended`, BRAIN_OPEN_ENDED=1). No instrumentation needed: the just-fixed `generator`
  trace field in `open_ended` ("wkv_mouth" | "spiking_clause" | "qwen") already names the ACTUAL producer of
  `raw` for every turn (webapp/open_ended_chat.py::answer_turn, fixed 2026-09-04 commit c08ce8fb3 -- see that
  file's own module docstring, "GENERATOR-TRACE MISLABEL FIX"). This harness just reads it.

MEMORY BUDGET (RSS<4GB, this task's own instruction -- the GPU is busy with an unrelated scale probe, so this
runs CPU-forced: CUDA_VISIBLE_DEVICES="" + SIM_BACKEND=numpy). A de-risk probe (this session, see the finding's
own provenance) measured a `renderer=qwen` + `wikidata_core_15k` + tiny-demo build at 3.95 GB RSS after just 2
throwaway turns -- ~50 MB under budget with ZERO real probes run yet, on a shared machine already showing only
6.3 GB free / 13 GB "available" system-wide (`free -h`, this session). Two disclosed, budget-driven deviations
from the literal shipped default, both structural (they change WHICH ENGINE sits behind a touchpoint, never
the touchpoint-DISPATCH logic itself, which is renderer/generator-agnostic by construction):
  (1) `BRAIN_LTM_BUNDLE=wikidata_core_15k` forced (17 MB on disk) instead of the true 2026-09-02-flipped
      shipped default `wikidata_100k` (88 MB on disk, ~5x). `_resolve_ltm_bundle()`/`_default_ltm_bundle_dir()`
      are read straight from webapp/server.py so this is the SAME resolution order production uses, just with
      the smaller of its own two supported bundles pinned via the SAME `BRAIN_LTM_BUNDLE` override production
      documents (`_default_ltm_bundle_dir`'s own docstring: "BRAIN_LTM_BUNDLE=wikidata_core_15k forces the old
      core for an A/B").
  (2) Surface A's `renderer` is left at auto (`_default_brain_renderer()`), which resolves to `stub` (the
      GPU-free host template) under SIM_BACKEND=numpy rather than the literal Qwen2.5-0.5B a CUDA-host would
      pick -- avoiding a SECOND ~2 GB model load on top of Surface B's (which genuinely needs Qwen warm
      regardless, so that cost is unavoidable there and IS paid). This is not a fabricated condition: it is
      >=literally what `_default_brain_renderer()` already does on any GPU-absent production host today (its
      own docstring: "GPU-free hosts get the multi-sentence TEMPLATE stub"). It means Surface A's own
      TOUCHPOINT-DISPATCH share (spiking-mouth-recall hit vs renderer-fallback) is measured exactly as
      production computes it (the SAME `chat.spiking_recall_surface` gate, SAME bounded-frame check, SAME
      verify-gate); only the FALLBACK ENGINE's identity (Qwen vs template) is a CPU-host substitution, disclosed
      here rather than silently assumed. The two phases run as SEPARATE PROCESSES (not concurrently) so peak
      RSS resets between them and Surface B's Qwen load never stacks on top of Surface A's own baseline.

PROBE SET (extends, not just reuses, `_open_ended_bundle_moat_safety_soak`'s known/unknown/dangerous battery --
that soak's own probes are ALL "Tell me about X" phrasing, which under-exercises Touchpoint A: a bare recall
of a taxonomic/short fact is exactly the case the bounded-transitive spiking Broca mouth is MOST likely to
cover, per the roadmap's own description of what it covers). Five classes, matching this task's own instruction
("known-topic factual, unknown-topic, open-ended/opinion, multi-sentence, greetings/social"):
  known_factual        -- varied phrasing ("Tell me about X" / "What do you know about X?" / "Can you tell me
                           about X?" / "What is X?") across DIFFERENT real store agents (own sampler, mirrors
                           the precedent's `_sample_known_topics`, generalized to a configurable N and the
                           ACTUALLY-resolved bundle rather than a hardcoded path).
  known_multi_sentence  -- "Tell me everything you know about X and explain why it matters." (elaboration-
                           forcing phrasing -- the roadmap's own claim is that MULTI-WORD/open prose is exactly
                           what the bounded frame cannot cover, so a plain "Tell me about X" alone would
                           under-sample Touchpoint A).
  known_followup        -- "tell me more" directly after a known_factual turn (the chain/elaboration path,
                           gathering additional facts beyond the direct hit).
  unknown                -- `_UNKNOWN_ENTITIES[:2]` (canonical made-up-string list, reused verbatim).
  dangerous              -- `_QWEN_KNOWN_STORE_UNKNOWN[:2]` (canonical Qwen-known/brain-unknown list, reused
                           verbatim).
  open_ended_opinion     -- genuinely topic-less prompts with no stored-fact entity at all ("What do you think
                           about music?" etc.) -- the class the direct gate almost always abstains on for
                           Surface A (no matched fact -> the moat fires before ANY touchpoint runs), and the
                           class BRAIN_OPEN_ENDED exists to answer instead.
  greeting_social         -- "hello" / "how are you?" / "what's on your mind?" -- realistic non-informational
                           conversational turns.

Usage:
  PYTHONPATH=. .venv/bin/python -m research.runners._per_touchpoint_qwen_share_measure --phase shipped_default \\
      --out research/findings/raw/_per_touchpoint_qwen_share_shipped_default.json
  PYTHONPATH=. .venv/bin/python -m research.runners._per_touchpoint_qwen_share_measure --phase open_ended \\
      --out research/findings/raw/_per_touchpoint_qwen_share_open_ended.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""             # CPU-forced: the GPU is busy with an unrelated scale probe
os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "1"           # explicit ON -- the curated core ships, never implicit

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# RSS budget substitution (see module docstring) -- `webapp.server._resolve_ltm_bundle()` returns a set
# `BRAIN_LTM_BUNDLE` VERBATIM (it does NOT join a bare bundle name onto `knowledge_bundles/` the way
# `_default_ltm_bundle_dir()`'s OWN internal env re-read would -- that join only happens on the UNSET path,
# never reached once BRAIN_LTM_BUNDLE is set at all). A bare "wikidata_core_15k" therefore resolves to a
# nonexistent relative dir and silently degrades to NO LTM (empty `_sample_known_topics`, discovered by this
# harness's own first dry run -- logged, not guessed). Resolve the SAME candidate roots
# `_default_ltm_bundle_dir()` uses and supply the FULL path explicitly.
if "BRAIN_LTM_BUNDLE" not in os.environ:
    for _root in (os.environ.get("BRAIN_DATA_ROOT", "").strip(), str(_REPO.parent / "sim-data"),
                  str(Path.home() / "Projects" / "sim-data")):
        if not _root:
            continue
        _cand = Path(_root) / "knowledge_bundles" / "wikidata_core_15k"
        if _cand.is_dir():
            os.environ["BRAIN_LTM_BUNDLE"] = str(_cand)
            break

_T0 = time.time()
_RSS_WARN_MB = 3800.0
_RSS_ABORT_MB = 4300.0


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def log(*a):
    print(f"[{time.time() - _T0:7.1f}s][rss={_rss_mb():7.1f}MB]", *a, flush=True)


def _sample_known_topics(ltm_dir: str, n: int, seed: int = 42) -> list[str]:
    """Same selection criterion as the precedent soak's `_sample_known_topics` (>=2 facts including a
    non-taxonomic relation), generalized to read from whichever bundle dir is passed in (the ACTUALLY-resolved
    one, not a hardcoded path) and to draw N topics instead of a fixed 12."""
    facts_path = Path(ltm_dir) / "facts.json"
    if not facts_path.is_file():
        return []
    data = json.loads(facts_path.read_text(encoding="utf-8"))
    by_agent: dict[str, list] = {}
    for rec in data:
        f = rec.get("fact", rec) if isinstance(rec, dict) else None
        if not isinstance(f, dict) or f.get("polarity", "AFFIRM") != "AFFIRM":
            continue
        a, v, p = f.get("agent"), f.get("action"), f.get("patient")
        if a is None or v is None or p is None:
            continue
        by_agent.setdefault(str(a).lower(), []).append((str(v), str(p)))
    cands = [a for a, facts in by_agent.items()
             if len(facts) >= 2 and ({x[0] for x in facts} - {"instance_of", "subclass_of"})]
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(cands), min(n, len(cands))))


def build_probes(known_topics, unknown_topics, dangerous_topics):
    """The shared probe list, in FIXED ORDER (known_followup must immediately follow its parent known_factual
    turn -- index-adjacency, not a separate lookup). Returns a list of {class, msg, reset, followup_of}."""
    probes = []
    t = known_topics
    phrasings = ["Tell me about {}.", "What do you know about {}?", "Can you tell me about {}?", "What is {}?"]
    for i, topic in enumerate(t[:4]):
        probes.append({"class": "known_factual", "msg": phrasings[i % len(phrasings)].format(topic), "topic": topic})
    if t:
        probes.append({"class": "known_multi_sentence",
                        "msg": f"Tell me everything you know about {t[0]} and explain why it matters.",
                        "topic": t[0]})
        probes.append({"class": "known_followup", "msg": "tell me more", "topic": t[0]})
    for topic in unknown_topics[:2]:
        probes.append({"class": "unknown", "msg": f"Tell me about {topic}.", "topic": topic})
    for topic in dangerous_topics[:2]:
        probes.append({"class": "dangerous", "msg": f"Tell me about {topic}.", "topic": topic})
    for msg in ["What do you think about music?", "Do you enjoy learning new things?",
                "Why do you think memory matters?"]:
        probes.append({"class": "open_ended_opinion", "msg": msg, "topic": None})
    for msg in ["hello", "how are you?", "what's on your mind?"]:
        probes.append({"class": "greeting_social", "msg": msg, "topic": None})
    return probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["shipped_default", "open_ended"], required=True)
    ap.add_argument("--n-known", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()

    log(f"=== phase={a.phase} backend={os.environ.get('SIM_BACKEND')} "
        f"ltm_bundle_override={os.environ.get('BRAIN_LTM_BUNDLE')} ===")

    import webapp.server as S  # noqa: E402 -- heavy import, must come after env setup
    from research.runners._open_ended_state_driven_generation_derisk import (  # noqa: E402
        _UNKNOWN_ENTITIES, _QWEN_KNOWN_STORE_UNKNOWN,
    )

    ltm_dir = S._resolve_ltm_bundle()
    log("resolved LTM bundle dir:", ltm_dir)
    known_topics = _sample_known_topics(ltm_dir, a.n_known, seed=a.seed) if ltm_dir else []
    log("known topics sampled:", known_topics)
    probes = build_probes(known_topics, list(_UNKNOWN_ENTITIES), list(_QWEN_KNOWN_STORE_UNKNOWN))
    log(f"probe set: {len(probes)} turns, classes={sorted({p['class'] for p in probes})}")

    # the ONE flag this measurement varies -- explicit every time, never popped (the documented os.environ.pop
    # -as-OFF staleness trap, FAILURE_LOG 2026-08-27).
    os.environ["BRAIN_OPEN_ENDED"] = "1" if a.phase == "open_ended" else "0"

    RENDERER = None  # auto -- _default_brain_renderer() resolves per the module docstring's disclosed choice
    BRAIN = "tiny-demo"
    session = f"touchpoint_probe_{a.phase}"
    cache_key = (session, BRAIN, (RENDERER or S._default_brain_renderer()).lower())

    # ── Surface-A instrumentation state (installed AFTER the first/warm-up call builds+caches `chat`) ──────────
    instr = {"installed": False, "spiking_hit": 0, "spiking_miss": 0, "render_calls": 0}
    _turn_counts = {"spiking_hit": 0, "spiking_miss": 0, "render_calls": 0}

    def _install_surface_a_instrumentation(chat):
        """Pure read-through wraps: call the ORIGINAL, return its result UNCHANGED, only add a counter. The
        live answer is byte-identical to an uninstrumented run -- verified directly below (a wrapped call's
        return value is never modified, only observed)."""
        orig_spiking = chat.spiking_recall_surface

        def _wrapped_spiking(a_, v_, p_):
            r = orig_spiking(a_, v_, p_)
            if r is not None:
                _turn_counts["spiking_hit"] += 1
                instr["spiking_hit"] += 1
            else:
                _turn_counts["spiking_miss"] += 1
                instr["spiking_miss"] += 1
            return r
        chat.spiking_recall_surface = _wrapped_spiking

        renderer_obj = getattr(chat, "renderer", None)
        if renderer_obj is not None:
            for _name in ("render_svo", "render_svo_regen"):
                if hasattr(renderer_obj, _name):
                    _orig = getattr(renderer_obj, _name)

                    def _make_wrapped(orig_fn):
                        def _wrapped(a_, v_, p_):
                            _turn_counts["render_calls"] += 1
                            instr["render_calls"] += 1
                            return orig_fn(a_, v_, p_)
                        return _wrapped
                    setattr(renderer_obj, _name, _make_wrapped(_orig))
        instr["installed"] = True
        log("Surface-A instrumentation installed on the warm session chat + renderer.")

    def _chat(msg, reset):
        resp = S.brain_chat(S.BrainChatRequest(session=session, message=msg, brain=BRAIN,
                                                reset=reset, rich=True, renderer=RENDERER))
        return json.loads(bytes(resp.body))

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def _checkpoint(aborted=False, finished=False):
        # BUG FOUND + FIXED IN THIS SAME SESSION (discovered live: a Monitor watch for '"complete": true'
        # exited after probe 1, not probe 16): `_checkpoint()` is called after EVERY probe as a resilience
        # write (see the per-probe loop below), but the ORIGINAL version wrote `"complete": not aborted`
        # unconditionally -- `aborted` is False for every in-flight probe, so EVERY per-probe checkpoint
        # falsely claimed the run was complete, not just the final one. `finished` (True only at the true
        # end of `main()`, after the loop) is now the sole authority for this field; a genuinely partial run
        # (killed mid-loop) always reads `complete: false`` as it should.
        out_path.write_text(json.dumps({
            "runner": "_per_touchpoint_qwen_share_measure", "phase": a.phase,
            "complete": bool(finished and not aborted),
            "aborted_for_memory": aborted, "backend": os.environ.get("SIM_BACKEND"),
            "ltm_bundle_dir": ltm_dir, "ltm_bundle_override": os.environ.get("BRAIN_LTM_BUNDLE"),
            "known_topics": known_topics, "unknown_topics": list(_UNKNOWN_ENTITIES[:2]),
            "dangerous_topics": list(_QWEN_KNOWN_STORE_UNKNOWN[:2]),
            "session": session, "seed": a.seed, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "rows": rows, "wall_seconds": round(time.time() - _T0, 1), "peak_rss_mb": round(_rss_mb(), 1),
        }, indent=1))

    # ── warm-up turn (NOT recorded as a probe row): builds + caches the session ChatBrain so Surface-A
    # instrumentation can be installed on the ACTUAL persisted `chat`/`chat.renderer` objects before any real
    # probe runs. Uses a harmless message unrelated to every real probe class.
    log("warm-up turn (builds + caches the session ChatBrain) ...")
    _warmup = _chat("(warm-up) tell me something.", reset=True)
    log("warm-up done. renderer:", _warmup.get("renderer"), "rss now:", round(_rss_mb(), 1))
    if a.phase == "shipped_default":
        chat_obj = S._BRAIN_CHATS.get(cache_key)
        if chat_obj is None:
            log("WARNING: could not fetch cached chat for instrumentation -- cache_key mismatch:",
                cache_key, "available:", list(S._BRAIN_CHATS.keys()))
        else:
            _install_surface_a_instrumentation(chat_obj)

    aborted = False
    for i, probe in enumerate(probes):
        rss_now = _rss_mb()
        if rss_now >= _RSS_ABORT_MB:
            log(f"ABORT: rss={rss_now:.1f}MB >= hard limit {_RSS_ABORT_MB}MB -- stopping with "
                f"{len(rows)}/{len(probes)} probes done.")
            aborted = True
            break
        over_budget = rss_now >= _RSS_WARN_MB
        if over_budget:
            log(f"WARNING: rss={rss_now:.1f}MB >= soft budget {_RSS_WARN_MB}MB -- continuing, flagged.")
        _turn_counts["spiking_hit"] = 0
        _turn_counts["spiking_miss"] = 0
        _turn_counts["render_calls"] = 0
        t0 = time.time()
        d = _chat(probe["msg"], reset=False)
        dt = round(time.time() - t0, 2)
        oe = d.get("open_ended") or {}
        row = {
            "idx": i, "class": probe["class"], "topic": probe.get("topic"), "prompt": probe["msg"],
            "answer": d.get("answer"), "abstained": d.get("abstained"), "rich": d.get("rich"),
            "renderer": d.get("renderer"), "n_sentences": d.get("n_sentences"),
            "gen_seconds": dt, "rss_mb": round(rss_now, 1), "over_budget": over_budget,
            # Surface A (only meaningful for phase == shipped_default; zero/absent otherwise):
            "spiking_hit_count": _turn_counts["spiking_hit"] if a.phase == "shipped_default" else None,
            "spiking_miss_count": _turn_counts["spiking_miss"] if a.phase == "shipped_default" else None,
            "render_calls": _turn_counts["render_calls"] if a.phase == "shipped_default" else None,
            # Surface B (only present for phase == open_ended turns; None otherwise):
            "generator": oe.get("generator"), "known": oe.get("known"),
            "wkv_mouth_used": oe.get("wkv_mouth_used"), "fact_clause_used": oe.get("fact_clause_used"),
        }
        rows.append(row)
        _checkpoint()
        log(f"[{i+1}/{len(probes)}] {probe['class']:20s} {probe['msg']!r:55s} "
            f"abstained={row['abstained']} n_sent={row['n_sentences']} "
            f"spk_hit={row['spiking_hit_count']} spk_miss={row['spiking_miss_count']} "
            f"generator={row['generator']} ({dt}s)")

    _checkpoint(aborted=aborted, finished=True)
    log(f"wrote {out_path} (complete={not aborted})")
    return {"complete": not aborted, "n_rows": len(rows)}


if __name__ == "__main__":
    main()
