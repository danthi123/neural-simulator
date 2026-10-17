# HANDOFF packet: d6-hebbian-write (integration wave 1)

Run the shared flow in `../HANDOFF.md`; this file holds what is specific to this packet. Prereg:
`research/findings/2026-09-29-iw1-d6-hebbian-write-flip-PREREGISTRATION.md`.

**What flips.** One anchor, `_D6_LEARN_DEFAULT_ON` in `research/runners/d6_hebbian_store.py`, turns on together
`BRAIN_D6_HEBBIAN_STORE` (an in-conversation fact is written into its block's synapses by a local Hebbian rule, not
copied), `BRAIN_D6_ENGRAM_VOCAB` (the known-word sets come from engrams that reactivate) and
`BRAIN_D6_ENGRAM_READTIME` (four kb readers see only engrams held now). Ledger row `in-loop-learning`. Escape after the
flip: any of the three flags `=0`. The lesion `BRAIN_D6_HEBBIAN_FREEZE` stays opt-in.

**Why it was chosen.** Capability gate v3 GO 6/6 (the recall reply depends on the synaptic write; host record inert),
content provenance GO 6/6 with both mouths (the shipped default learns 0% of what it is told), AI-teacher GO 6/6.
Each of those findings names the missing legs: a combined-battery run and a production-default run. Phase: the
content of what the brain says it learned is owned by a synapse it wrote. Stage 1 exit: lived-fact recall.

## Step R for this packet (shared HANDOFF sections 3-5)

Job file `jobs/R_d6_<P7>.txt` (6 lines), tag `iw1-R-d6-<P7>`, row `d6-hebbian-write`: teach "the wolf hunts the deer"
-> "what does the cat eat" -> "what does the dog chase" -> "what does the wolf hunt". Score:
`tools/iw1_score.py R --packet d6-hebbian-write ...`.

| R verdict | per-seed detail | do |
|---|---|---|
| PASS | 6/6 DEFINED, S2 `regressed` | into the flip list (if D0 and D_d6-hebbian-write pass) |
| FAIL | a DEFINED seed reads `pass` | not flipped. Draft `research/findings/<date>-iw1-d6-hebbian-write-reply-level-NO-GO-<n>of6.md` (provisional): per-seed table from `R_d6.json` (answer of intact_a and lesion on the probe turn) |
| UNDEFINED | a seed's `state` is not DEFINED, or S2 reads `UNDEFINED:*` | shared re-run rule once; still UNDEFINED: not flipped, NEEDS-CLAUDE "iw1 d6 R UNDEFINED: <seed>: <fails>" |

## Production default (shared steps B and F)

F1: in `iw1-flip-<F7>` the row `d6-hebbian-write` must read reply-level load-bearing on 6/6 (in `B.json`,
`new_rows_s2_at_flip["d6-hebbian-write"] == 6`).

## Finding template (after the shared section 7 reads PASS)

`research/findings/<date>-iw1-d6-hebbian-write-default-on-flip-validated-PROVISIONAL.md`, frontmatter
`integration_faculty: in-loop-learning`, `status: provisional`, seeds 42/43/44/100/101/102, artifacts = the verdict
JSONs + the three shard tags. Sections: what flipped (the three flags, one anchor), R per-seed table (from R_d6.json),
B summary (S0/S2 robust core before/after from B.json), F1/F2/F3 lines, the prereg's declared residuals copied,
a "Per-turn cost" section (owner ruling 2026-09-29: the growth is accepted and must be reported): the read-time
view's per-turn seconds at N = 5 / 50 / 500 / 2000 taught facts, HEBB and COPY, quoted per seed from
`2026-09-25-d6-capacity-curve-N2000-6seed-CEILING-BETWEEN-50-AND-500.md`, plus the per-turn wall times of the
step-D/F identity arms at P and F; and "Verdict: provisional until Claude's adversarial review". Say "on by default on branch research/integration-wave1"
only; never "now on-by-default" (the PI gate's claim token) before the merge.

## Dev seed 7 (Claude, 2026-09-29, local numpy, worktree at the packet commit)

`research/findings/raw/_iw1/dev_s7/d6/` (row `d6-hebbian-write`, flags set as in step R, `--repeats 2`, commit
106a20da5; the local sidecars are not pool git_archive, so `iw1_score.py` would class the cell E on provenance --
that is expected for a dev run, and the gate seeds run on the pool). Registered trace verdict `regressed`, null clean,
lesion reproduced; `b2d_score.score_s2` on the same arm files reads `regressed` (reply level). Probe turn "what does
the wolf hunt": both intact builds answer "Setting the held thread aside -- On wolf, then -- the wolf hunts the deer"
(`recalled_svo` `['wolf', 'hunt', 'deer']`); the freeze lesion and its rebuild abstain with "I don't know about that.
My curiosity is piqued -- I haven't learned about wolf yet: what can you tell me about wolf?". The teach turn wrote
block 5 (128/128 synapses saturated), so the buffer held 6 facts: per-turn cost was not visible at this size.

## NEEDS CLAUDE (packet-specific)

1. RULED (owner, 2026-09-29): accepted (speed secondary); the finding MUST report the measured growth (template above). Was: per-turn
   cost grows with the number of taught facts (read-time view: 3-5 s per turn at 50 facts, minutes at 500,
   pool CPU). Before the merge, rule whether a default-on write needs a cap or a consolidation path for long chats.
