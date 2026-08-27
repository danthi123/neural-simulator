---
type: finding
status: contributing
date: 2026-08-27
mechanism: confidence-caps-forthcomingness
lane: live-brain/introspection
artifacts:
  - research/findings/raw/_confidence_forthcoming/verify.json
runner: research/findings/raw/_confidence_forthcoming/verify_confidence_forthcoming.py
seed-waiver: production-INTEGRATION verify of an already-6/6-GO faculty (the nmda_norm metacog confidence read,
  2026-08-13-metacog-robust-confidence-GO.md). This doc verifies the deterministic CAP/TRUNCATION wiring glue on
  the real handler (single process, one seed=42 organ); the 6-seed statistical evidence for the underlying
  spiking margin read is the cited de-risk. The lesion + flag-off arms are decisive on the single wired seed.
---

# Confidence caps forthcomingness on the rich chat turn (board #94), GO (2026-08-27)

## Result
The brain's own spiking metacog confidence read (`metacog_production_organ`, the `nmda_norm` divisive-normalized
NMDA-conductance balance, already wired as the E1 honest hedge) now also caps HOW MUCH the rich (multi-sentence)
chat turn volunteers, not just whether it prepends a hedge phrase. A HIGH-confidence answer is allowed to chain
in ONE extra grounded fact beyond the floor the existing mood coupling (`content_plan`) already set; a
LOW-confidence (or unread) answer stays exactly at that floor. Additive, DEFAULT-OFF
(`BRAIN_CONFIDENCE_FORTHCOMING`), wired into `webapp/server.py::brain_chat`'s rich path; NO `sim/` edit
(reuse-by-import of the existing organ + composer methods). New module: `webapp/confidence_forthcoming_chat.py`.

## Design decision (the owner's build-note subtlety)
The metacog confidence read is the answer's own mean role-decode confidence, available only AFTER the rich
composer has gathered + rendered a turn — not before. This module takes POST-HOC TRUNCATION over a pre-computed
primary-recall probe: it requests a "reach" budget (floor + 1 fact) from `RichAnswerComposer` BEFORE
`rich.answer()` runs, reads the SAME post-answer confidence the E1 hedge already computes (ONE spiking read,
reused for both), and truncates the reach's extra fact back to the floor when not confident. A pre-flight probe
would need a second, separate call into the stochastic direct-recall gate before the composer's own gather()
runs it again — doubling per-turn cost for no new evidence. Because the composer's gather is deterministic, the
first `floor` facts of a reach-sized gather are provably identical to what a floor-only gather would have
produced, so truncating the tail reproduces the floor-only answer exactly; withheld facts are also removed from
the composer's discourse-thread "already said" registers so a later "tell me more" can still bring them up
honestly. Full reasoning lives in the `webapp/confidence_forthcoming_chat.py` module docstring.

## Verify — through the REAL `/api/brain-chat` handler, in-process, GO 4/4
<!--derived--> (rounded reads of the cited artifact; full precision is in the JSON)

- **(A) elaboration count differs, high vs low, base fact identical.** "what does the brain use" (forced-HIGH
  evidence) -> 2 sentences, `confident=True`, `granted=True`, direct fact `[brain, use, spikes]`. "what does the
  dog chase" (forced-LOW evidence) -> 1 sentence, `confident=False`, `granted=False`, direct fact
  `[dog, chase, cat]` — the SAME direct fact either way; only the bonus elaboration differs (2 vs 1). PASS.
- **(B) LESION collapses the difference.** `BRAIN_METACOG_LESION=1` on both turns -> both read `confident=False`
  unconditionally and both settle at 1 sentence (the intact HIGH turn's 2 sentences drop to 1) — the
  high-vs-low difference (B) is gone under the SAME lesion the E1 hedge already uses. PASS.
- **(C) byte-identical-off.** `BRAIN_CONFIDENCE_FORTHCOMING` unset on two independent runs of the same question
  -> no `confidence_forthcoming` key on either, and the two runs are byte-identical (same answer, same sentence
  count). PASS.
- **(D) moat-safe.** The HIGH-confidence (bonus-granted) reply's `verified=True` and its 2 `supporting_facts`
  exactly match its 2 rendered sentences — the honesty filter (per-sentence VERIFY) still governs the granted
  elaboration; nothing ungrounded was volunteered. PASS.

Artifact: research/findings/raw/_confidence_forthcoming/verify.json (A/B/C/D all pass, verdict GO) · runner
research/findings/raw/_confidence_forthcoming/verify_confidence_forthcoming.py.

## Honest residual (declared, not hidden)
On the small `tiny-demo` brain, the rich path's role-decode confidence extraction (`mean_role_confidence`) very
often returns None: `RichAnswerComposer._chain_facts` always probes one hop past a successful match, and that
probe's failure clobbers the composer's `last_trace`; the default-on GNW ignition buses also wrap `chat.gate` in
a way that does not exercise the plain trace-populating query path. Both are PRE-EXISTING properties of the E1
hedge's own evidence extraction (a declared host boundary), not something this feature changed — the SAME
confound would silence the E1 hedge on this demo brain today. To exercise the load-bearing SPIKING part of the
organ (the `nmda_norm` NMDA-conductance margin computation + its lesion) without also re-deriving a working
role-confidence extraction on this small demo, the verify script patches only the upstream evidence INPUT
(`evidence_from_role_conf`, the module's own declared host boundary) to the topic-keyed value the ORIGINAL E1
GateB measurement reported <!--derived--> (0.400 low / 0.476 high mean role-decode confidence, quoted from
`2026-08-12-GateB-metacog-confidence-readout-production-chat.md`); the organ build, the spiking
margin simulation, the threshold decision, and the lesion all ran unmodified. Fixing the role-confidence
extraction on this demo brain (or moving verification to a richer knowledge base) is the honest next rung, not
claimed closed here.

## Escape / lesion knobs
```
BRAIN_CONFIDENCE_FORTHCOMING=1          # enable (default OFF)
BRAIN_CONFIDENCE_FORTHCOMING_FLOOR="1,0"  # optional: force the floor (mirrors the #84 mood-INDUCE pattern)
BRAIN_METACOG_LESION=1                  # reused lesion (collapses confident->False unconditionally)
```
