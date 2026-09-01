"""BATCH SENTENCE RENDERING (2026-09-01, board frontier row "make each chat reply snappier by batching the
sentence rendering", branch `research/batch-sentence-rendering`).

THE WALL (memory `feedback_prioritize_orchestration_overhead`): the real-time chat wall is per-op LATENCY, not
VRAM -- the composer loop is launch-bound (many tiny sequential ops). This file locates the concrete instance
of that pattern in the RICH-answer production path and measures whether batching closes it.

WHERE THE LOOP LIVES. `RichAnswerComposer.render_paragraph` (research/runners/rich_answer_composer.py) renders
each gathered [a, v, p] (up to `max_sentences`, default 4) into a fluent sentence ONE AT A TIME:
    for svo in facts:
        sent, verified = self._render_one_verified(svo, gated=gated)
`_render_one_verified` calls `chat.renderer.render_svo(a, v, p)` -- for the production `QwenRenderer` this is
`SpikingQwenFaculty.render_svo` -> ONE `self.model.generate()` call per sentence. A 4-sentence rich reply
therefore pays 4 SEQUENTIAL `model.generate()` launches (each restarting tokenization + the model forward from
scratch for its own short prompt), even though the 4 prompts are mutually independent and could share one
batched forward pass.

THE FIX (additive, default-OFF `BRAIN_RICH_BATCH_RENDER`, see `rich_answer_composer._batch_render_enabled`).
`SpikingQwenFaculty.render_svo_batch` / `QwenRenderer.render_svo_batch` (both NEW, `_grounded_lang_
integration_derisk.py` / `brain_chat_tui.py`) left-pad the N CONSTRAIN prompts into ONE tensor and call
`model.generate()` ONCE. `RichAnswerComposer._render_paragraph_batched` (NEW) uses this for every fact that
falls through the cheap spiking-recall-surface check, VERIFY-gates each candidate exactly as the sequential
path does, and falls back to the single-item path (regen included) on any candidate that fails -- so it can
only be FASTER on the common case, never less safe.

THIS FILE measures, with the REAL production `QwenRenderer` (off-bridge spiking Qwen2.5-0.5B, GPU):
  (1) BASELINE latency: `render_paragraph(facts)` wall-clock, flag OFF (the pre-existing sequential loop),
      median over `--reps` repetitions.
  (2) BATCHED latency: the same call, flag ON, median over the same N repetitions.
  (3) BYTE-IDENTICAL-EQUIVALENCE: the batched path's (paragraph, kept, dropped) vs the sequential path's, for
      the SAME gathered facts -- reported honestly whether or not it holds (see the module docstring on
      `SpikingQwenFaculty._generate_batch` for the one honest mechanism-level reason it might not: the
      installed spiking ops draw graded-read pool noise from a shape-dependent RNG stream, so a batched
      forward's per-sequence noise realization is not bitwise-guaranteed identical to a lone forward's, even
      though both reseed to the identical value).
  (4) BYTE-IDENTICAL-WHEN-OFF (ASSERTED, not just inferred from output equality): `_render_paragraph_batched`
      is monkeypatched to raise if ever called; `render_paragraph(facts)` with the flag OFF must still succeed
      and produce its ordinary output -- PROVING the new code path is never reached when the flag is off, not
      merely that its output happens to match.

ISOLATION FROM THE SPIKING-MOUTH-RECALL PATH. `BRAIN_SPIKING_MOUTH_RECALL` is production DEFAULT-ON
(2026-08-26 flip): `chat.spiking_recall_surface` is checked FIRST for every fact, identically in both the
sequential and the batched render path, and a bounded-transitive-SVO fact never reaches Qwen's `render_svo` /
`render_svo_batch` at all when it succeeds there. This runner forces that flag OFF for its own measurement
(process-local env override, see below) so every gathered fact is actually rendered by Qwen -- a clean,
decisive read of the render-BATCHING mechanism this branch adds, isolated from that orthogonal, separately
GO-verified production feature (batching what the spiking mouth already resolves for free would only dilute
the very effect under test).

Usage:
    /home/dant123/Projects/sim/.venv/bin/python -m research.runners._batch_sentence_rendering_derisk \
        --reps 7 --out research/findings/raw/_batch_sentence_rendering_derisk.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging
logging.disable(logging.INFO)

from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.verdict import Verdict  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_batch_sentence_rendering_derisk.json"

BATCH_FLAG = "BRAIN_RICH_BATCH_RENDER"


def _build_rich_qwen(seed: int, T: int, max_new_tokens: int, max_sentences: int):
    """The tiny interlinked CPU brain (`rich_answer_composer._build_smoke_chat`) with its renderer swapped from
    the GPU-free `StubRenderer` to the REAL production `QwenRenderer` (off-bridge spiking Qwen2.5-0.5B) -- the
    same renderer class `brain_chat_tui --rich` uses at runtime. Returns (chat, rich)."""
    from research.runners.rich_answer_composer import _build_smoke_chat, RichAnswerComposer
    from research.runners.brain_chat_tui import QwenRenderer
    chat = _build_smoke_chat(seed, use_multiturn=True)
    print("[batch-render] loading the off-bridge spiking Qwen (calibration pass) ...", flush=True)
    t0 = time.time()
    chat.renderer = QwenRenderer(T=T, max_new_tokens=max_new_tokens, seed=seed)
    print(f"[batch-render] Qwen ready (load {round(time.time() - t0, 2)}s)", flush=True)
    rich = RichAnswerComposer(chat, max_chain_hops=5, max_elaborations=2, max_sentences=max_sentences)
    return chat, rich


# `BRAIN_SPIKING_MOUTH_RECALL` is production DEFAULT-ON (2026-08-26 flip): `chat.spiking_recall_surface` is
# checked FIRST for every fact, by BOTH the sequential and the batched render path, identically (see
# `RichAnswerComposer._render_one_verified` / `_render_paragraph_batched`) -- a bounded-transitive-SVO fact
# ("brain use spikes") is rendered on the spiking Broca WITHOUT EVER reaching the Qwen renderer's `render_svo`
# / `render_svo_batch` at all. Left as-is, a measured "N-sentence" turn may only send a FRACTION of those N
# facts to Qwen (the rest resolve on the cheap, GPU-free spiking mouth in BOTH conditions equally -- a fair
# comparison, but a SMALLER effective batch than `n_sentences` implies, diluting the very effect under test).
# This runner forces the flag off for its OWN measurement only (env var scoped to this process) so every
# gathered fact reaches the Qwen renderer -- a clean, decisive read of the render-BATCHING mechanism, isolated
# from that orthogonal, separately-verified production feature. Restored (well, this process never touches it
# again) -- a subprocess-local override, not a persistent config change.
os.environ["BRAIN_SPIKING_MOUTH_RECALL"] = "0"


def _median_latency(fn, reps: int):
    times = []
    result = None
    for i in range(reps):
        t0 = time.time()
        result = fn()
        times.append(time.time() - t0)
        print(f"    rep {i + 1}/{reps}: {times[-1]:.3f}s", flush=True)
    return statistics.median(times), times, result


def run(seed: int, T: int, max_new_tokens: int, reps: int, out_path: str, max_sentences: int, warmup: bool):
    chat, rich = _build_rich_qwen(seed, T, max_new_tokens, max_sentences)

    # gather a genuinely multi-fact turn (the RICH answer's whole point) so there is more than one sentence to
    # batch -- "what are you" -> the direct fact + the chain hops (+ elaboration), capped at max_sentences.
    topic, facts = rich.gather("what are you", followup=False)
    n_facts = len(facts)
    print(f"[batch-render] gathered {n_facts} facts for topic={topic!r}: {facts}", flush=True)
    if n_facts < 2:
        print("[batch-render] FATAL: fewer than 2 gathered facts -- nothing to batch; "
              "the smoke knowledge graph must yield a multi-sentence turn.", flush=True)
        return 1

    if warmup:
        # ONE untimed call per condition first -- absorbs first-call CUDA warm-up (kernel autotune / cache
        # build) so the TIMED reps below measure steady-state latency, not a one-off warm-up tax on rep 1.
        print("[batch-render] warm-up (untimed) ...", flush=True)
        os.environ.pop(BATCH_FLAG, None)
        rich.render_paragraph(facts)
        os.environ[BATCH_FLAG] = "1"
        rich.render_paragraph(facts)
        os.environ.pop(BATCH_FLAG, None)

    # ---- (1) BASELINE: the pre-existing sequential per-sentence loop (flag OFF / unset) ----
    os.environ.pop(BATCH_FLAG, None)
    print(f"[batch-render] BASELINE (flag OFF, sequential per-sentence render), {reps} reps:", flush=True)
    base_median, base_times, base_result = _median_latency(lambda: rich.render_paragraph(facts), reps)
    para_off, kept_off, dropped_off = base_result

    # ---- (2) BATCHED: the new one-launch render (flag ON) ----
    os.environ[BATCH_FLAG] = "1"
    print(f"[batch-render] BATCHED (flag ON, one launch for {n_facts} sentences), {reps} reps:", flush=True)
    batch_median, batch_times, batch_result = _median_latency(lambda: rich.render_paragraph(facts), reps)
    para_on, kept_on, dropped_on = batch_result
    os.environ.pop(BATCH_FLAG, None)

    speedup = (base_median / batch_median) if batch_median > 0 else float("nan")

    # ---- (3) BYTE-IDENTICAL EQUIVALENCE: batched output vs sequential output, same facts ----
    paragraph_identical = (para_off == para_on)
    kept_identical = (kept_off == kept_on)
    dropped_identical = (dropped_off == dropped_on)
    equivalence_ok = bool(paragraph_identical and kept_identical and dropped_identical)

    # ---- (4) BYTE-IDENTICAL-WHEN-OFF, ASSERTED: prove the batched code path is never REACHED with the flag
    # off (not merely that its output happens to match) -- monkeypatch the batched renderer to raise, then
    # confirm the flag-off call still succeeds untouched.
    os.environ.pop(BATCH_FLAG, None)
    _orig = rich._render_paragraph_batched

    def _poison(*_a, **_k):
        raise AssertionError("BATCHED PATH REACHED WITH THE FLAG OFF -- byte-identical-when-OFF violated")

    rich._render_paragraph_batched = _poison
    try:
        para_poisoned, kept_poisoned, dropped_poisoned = rich.render_paragraph(facts)
        off_never_calls_batched = True
        off_poisoned_matches_baseline = (para_poisoned == para_off and kept_poisoned == kept_off
                                         and dropped_poisoned == dropped_off)
    except AssertionError as exc:
        off_never_calls_batched = False
        off_poisoned_matches_baseline = False
        print(f"[batch-render] FLAG-OFF POISON TEST FAILED: {exc}", flush=True)
    finally:
        rich._render_paragraph_batched = _orig

    # also confirm a SINGLE-fact turn never batches even with the flag ON (nothing to batch -- the len>1 guard)
    os.environ[BATCH_FLAG] = "1"
    rich._render_paragraph_batched = _poison
    try:
        rich.render_paragraph(facts[:1])
        single_fact_never_batches = True
    except AssertionError:
        single_fact_never_batches = False
    finally:
        rich._render_paragraph_batched = _orig
        os.environ.pop(BATCH_FLAG, None)

    print(f"\n[batch-render] BASELINE median={base_median:.3f}s  BATCHED median={batch_median:.3f}s  "
          f"speedup={speedup:.2f}x  (n_sentences={n_facts})", flush=True)
    print(f"[batch-render] paragraph_off={para_off!r}", flush=True)
    print(f"[batch-render] paragraph_on ={para_on!r}", flush=True)
    print(f"[batch-render] equivalence_ok={equivalence_ok}  off_never_calls_batched={off_never_calls_batched}  "
          f"single_fact_never_batches={single_fact_never_batches}", flush=True)

    v = Verdict("batching the RICH-answer's per-sentence render loop into one model.generate() launch reduces "
               "wall-clock latency, byte-identical-when-OFF is PROVED (not just inferred), and batched-vs-"
               "sequential text equivalence is measured and reported honestly either way")
    v.require("flag OFF: the new batched code path is PROVABLY never reached (poison test)",
              off_never_calls_batched, expect=True)
    v.require("flag OFF + poisoned batched method: output still matches the untouched baseline",
              off_poisoned_matches_baseline, expect=True)
    v.require("flag ON + a single-fact turn: still never batches (nothing to batch)",
              single_fact_never_batches, expect=True)
    v.require("BATCHED median latency is lower than the BASELINE sequential median",
              batch_median, expect=lambda x: x < base_median)
    v.disabled("batched output is BYTE-IDENTICAL to the sequential output for this multi-sentence turn",
              why=f"equivalence_ok={equivalence_ok} (paragraph={paragraph_identical}, kept={kept_identical}, "
                  f"dropped={dropped_identical}) -- the installed spiking ops draw graded-read pool noise from "
                  f"a torch.randn() call SHAPED by the full (batch, seq, ...) tensor (see `_generate_batch`'s "
                  f"own docstring), so a batched forward's per-sequence noise realization is not bitwise-"
                  f"guaranteed identical to a lone forward's even at the identical reseed value. Reported "
                  f"honestly as measured, not asserted; the moat is unaffected either way (VERIFY still gates "
                  f"every candidate, batched or sequential, before it is kept).")
    go = bool(off_never_calls_batched and off_poisoned_matches_baseline and single_fact_never_batches
             and batch_median < base_median)
    decided = v.decide(go=go)

    art = {
        "probe": "batch_sentence_rendering_derisk", "backend": "cuda(qwen)+numpy(chat)",
        "seed": seed, "T": T, "max_new_tokens": max_new_tokens, "reps": reps, "warmup": warmup,
        "max_sentences_cfg": max_sentences, "n_sentences": n_facts,
        "topic": topic, "facts": [list(f) for f in facts],
        "baseline_median_s": base_median, "baseline_times_s": base_times,
        "batched_median_s": batch_median, "batched_times_s": batch_times,
        "speedup_x": speedup,
        "paragraph_off": para_off, "paragraph_on": para_on,
        "kept_off": kept_off, "kept_on": kept_on, "dropped_off": dropped_off, "dropped_on": dropped_on,
        "paragraph_identical": paragraph_identical, "kept_identical": kept_identical,
        "dropped_identical": dropped_identical, "equivalence_ok": equivalence_ok,
        "off_never_calls_batched": off_never_calls_batched,
        "off_poisoned_matches_baseline": off_poisoned_matches_baseline,
        "single_fact_never_batches": single_fact_never_batches,
        "verdict": decided, "preconditions": decided.get("preconditions", []), "GO": bool(go),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(art, fh, indent=2, ensure_ascii=False)
    print(f"[batch-render] wrote {os.path.relpath(out, _REPO)}", flush=True)
    return 0 if go else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--max-sentences", type=int, default=6,
                    help="RichAnswerComposer max_sentences (how many facts/sentences one rich turn gathers).")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the untimed warm-up call before each timed block (default: warm up).")
    args = ap.parse_args()
    return run(args.seed, args.T, args.max_new_tokens, args.reps, args.out, args.max_sentences,
              warmup=not args.no_warmup)


if __name__ == "__main__":
    sys.exit(main())
