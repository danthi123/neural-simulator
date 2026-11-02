---
name: land-result
description: Use BEFORE committing or reporting a positive result (a GO, "works", "surpass", "inert", "byte-identical", "tests pass", a milestone met) or flipping a default, including an interim "lead". Six seeds, flag-off arm, numbers read from the artifact, a fresh-context skeptic, a lesion that changes the REPLY, then commit with caveats and update HANDOFF.md and the board.
---
# land-result: try to refute it, then land what survives

Each `(...)` names the incident behind a step. Full old checklist: `.claude/skills/_reference/verify-go.md`.

## 1. Evidence
- Six seeds (42/43/44/100/101/102) before any generalization; 3 seeds once showed "33%" that was 11% and not
  significant on 6 (memory 6seed_validation).
- A flag-off arm of the WHOLE battery on the gate seeds, with the flag set explicitly off: an "off" arm went stale
  when the default flipped (FAILURE_LOG 2026-08-27); packet-probe answer rate alone is blind (memory 10-17).
- Byte-identical means against a PINNED sha, never a moving ref like origin/main (FAILURE_LOG 2026-09-23).
- A null or shuffle control uses several permutations, not one seed-drawn draw (FAILURE_LOG 2026-09-23).
- Read every number from the result file or the run's printout, never from memory: chance quoted 0.200 where the
  run printed 0.167 (FAILURE_LOG 2026-07-31). Speed claims need measured wall time and GPU use (memory no_overclaim).
- Instrument sanity arm: report chance and the flag-off value next to the result. A read-isolation bug flipped
  nine banked GO 6/6 to NO-GO (acceleration report 2026-09-29, finding 2).
- Lesion: switching it off must change the REPLY, not only an internal field; report the reply-level number
  beside any decision-level one (memory faculties_must_drive 08-19; LBF 0.843 vs 0.535, 2026-09-25).

## 2. Fresh-context skeptic (always; it was the step that never ran, FAILURE_LOG 2026-07-31)
- Claude: one read-only subagent, model sonnet. Hermes or Pi: a fresh delegate session, or a NEEDS line for Claude.
- Give it ONLY the diff, the result file(s) and the runner source. Ask it to refute the claim; default REFUTED
  when unsure. Add the block below verbatim to its prompt.
- If a delegate drafted the prereg, diff every bar against the approved packet first (memory check_delegate_prereg_bars).
- An interim "lead" counts too: one written into two docs was a winner-slot artifact (verify-go reference 2026-07-25).

### Required refuter-lens block: paste VERBATIM into every refuter prompt (Claude review 2026-10-06)
<!-- REFUTER-LENS-BEGIN -->
REQUIRED LENSES (answer each from recorded data, never from inference):
(a) ENGAGEMENT: did each lesion, permutation or flag actually engage, as recorded in the data (a call count, n_words, a changed field, a trace with route / lesioned / error)? An arm with no record is UNDEFINED, never a PASS or a FAIL.
(b) ANSWER-RATE: for any default-on candidate, does flag-on lower the answer rate or raise abstentions ("I don't know about that.") against flag-off on the packet's own probes, on any seed? Run `tools/answer_rate.py` (prefixed or paired mode) or `iw2_score.py B --off-tag`; a FLIP-HURTS-ANSWERS result means the flip needs a Claude ruling (NEEDS line) before it lands.
(c) WHERE-RAN: which host produced each cited result (run `python tools/where_ran.py <result paths>` and quote its host summary), and does validity depend on it (pool/AWS nodes and the workstation differ in what inputs they hold, for example files under research/findings/raw/ that the provisioner does not ship)?
<!-- REFUTER-LENS-END -->
`tests/test_verify_go_lens_block.py` pins this block.

## 3. Words
- Check each load-bearing word (GO, works, closed, byte-identical, lesion, selective, consolidation, fully
  spiking...) against its code condition in `docs/TERMS.md`.

## 4. Land
- Survives: commit, naming the lenses run. Partly refuted: commit WITH the caveat (what it is vs what was claimed);
  never drop it silently. Refuted: bank the method as NO-GO with its root cause; the capability stays open.
- Validated and lesion-verified: flip it on by default yourself; stopping at "validated opt-in" is the old
  stall (memory close_arcs 06-24, flip_validated_fixes 09-23).
- Update HANDOFF.md (STATE, NEXT STEP) and the board item in the same step (vikunja skill; board drift
  FAILURE_LOG 2026-08-20). If a brain/ARCHITECTURE.md §4 milestone moved, say so in HANDOFF.md.
- Claude on W pushes with `bash tools/push_both.sh <branch>`; Hermes and Pi publish via sim-publish.

Done = committed with caveats, HANDOFF.md and the board item updated, skeptic verdict quoted in the commit.
