---
type: finding
status: live
date: 2026-08-20
mechanism: open-text-moat-verifier
lane: D-pragmatics
seeds: [42]
seed-waiver: A deterministic frame-stripping fix + its before/after check — does stripping a leading reporting-verb+"that" frame let an embedded factual clause be extracted+entailed, without mis-stripping a relative clause. The evidence is per-item keep/suppress correctness on a fixed labelled set; single seed is the substrate build seed.
instrument: research/runners/_reporting_clause_strip_verify_derisk.py — strip_reporting_frame in front of the UNCHANGED extract (NPHeadBinder) + entail (FactStore/classify_claim) pipeline
runner: research/runners/_reporting_clause_strip_verify_derisk.py
artifacts:
  - research/findings/raw/_reporting_clause_strip_verify_derisk.json
external: NO-EXTERNAL-NEEDED — an in-repo lexical frame-detector for residual #3 named by our own 2026-08-20 fluency de-risk (same host-preprocessing category as the existing HEDGES/STOPWORDS/segment_clause rules); no capability wall or paradigm claim.
---
# Fluency residual #3 CLOSED — the reporting-clause frame is stripped; all THREE fluency-moat residuals now done

Artifact: research/findings/raw/_reporting_clause_strip_verify_derisk.json

**One line.** The last fluency residual: a clause like "Scientists confirmed that Mercury orbits the sun" failed the
moat two ways — a bare-SVO fact under the wrapper had too many content words to segment (`unparsed_abstain`), and a
multi-word-subject fact got its subject NP silently over-extended ("scientists confirmed amazon rainforest"). Both
wrongly SUPPRESSED true reworded facts. This lands the fix: a declared REPORTING_VERBS + "that" frame is stripped
BEFORE the SVO/NP passes, so the embedded clause recurses cleanly into the UNCHANGED extract+entail. With this, all
three fluency-moat residuals (hedge bypass, synonym brittleness, reporting-clause) are closed.

## Before / after (numpy, seed 42; reproduced identically)
| metric | before | after |
|---|---|---|
| recall-on-embedded-faithful (2 items) | 0/2 | **2/2** — "Scientists confirmed that X" now KEEPS a true X |
| precision-on-embedded-confab (2 items) | 2/2 | 2/2 — a false X under the wrapper still SUPPRESSED |
| relative-clause guard (3 items, incl. "Mercury is the planet THAT orbits the sun") | — | all 3 no-op (strip returns None, byte-identical) |
| regression (plain/passive/synonym/hedge, 19 items) | — | unchanged_overall True |

`strip_reporting_frame` anchors a small REPORTING_VERBS lexicon (confirmed/reported/said/found/showed/announced/…)
on the token immediately before the clause's first "that"; two thin wrappers try the strip first and fall through to
the UNCHANGED `extract_svo_npbind` on no match. The relative-clause guard is the important half: "the planet that
orbits" is NOT a reporting frame (no reporting verb before "that"), so it is left untouched — the anchor prevents
over-stripping.

## The one newly-named residual (honest, evidence-backed next step)
NESTED reports ("Scientists confirmed that researchers reported that Mercury orbits the sun") are NOT closed by the
single strip — only the outer frame is removed; the inner "reported that" defeats `segment_clause`, so the true
doubly-reported fact is suppressed via `unparsed_abstain` (0/1, fails safe — the confab variant is also suppressed).
This residual was not previously tracked; it is small, and the runner MEASURED the fix path rather than guessing —
looping the identical rule to a fixpoint (`strip_reporting_frame_recursive`) closes it (2/2). So the smallest safe
next step is confirmed by evidence.

## Fluency trilogy status
extraction reaches free prose (NPHeadBinder, cd3d5ff2) → moat entails (GO) → **hedge bypass CLOSED** (dd1b76b6) →
**synonym brittleness CLOSED** (620601a6) → **reporting-clause CLOSED** (this). Every fix imports the spiking
extraction + FactStore entailment UNCHANGED and adds only a declared host lexical rule (the same category as the
existing preprocessing). ⇒ the open-text moat is now safe + fluent for verbatim/passive/synonym/hedge/single-level-
reporting prose; the remaining edges (nested reports; the extraction is host-lexical NP-boundary detection feeding
the SPIKING role read-out) are named. The integration step — wiring the corrected routing + all three fixes into the
LIVE verifier path in brain_chat, then widening Qwen to free generation gated by it + lesion-testing the faculties
stay load-bearing over open output — is #99. (Agent-built, independently re-run + reproduced.)
