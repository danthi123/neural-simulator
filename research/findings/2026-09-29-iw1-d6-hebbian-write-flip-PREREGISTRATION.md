---
type: preregistration
status: preregistered
date: 2026-09-29
lane: integration (wave 1, packet d6-hebbian-write; ledger row in-loop-learning)
mechanism: D6 learn-through-use -- the in-conversation fact write by a local Hebbian rule on the substrate
  (research/runners/d6_hebbian_store.py) with the engram-derived vocabulary and the read-time view, three flags
  (BRAIN_D6_HEBBIAN_STORE, BRAIN_D6_ENGRAM_VOCAB, BRAIN_D6_ENGRAM_READTIME) now governed by one anchor
  `_D6_LEARN_DEFAULT_ON`, measured at reply level in the standard battery (LBF row d6-hebbian-write)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed with the anchor and the row, before any gate-seed job exists. No result is
  claimed here.
---

# Packet d6-hebbian-write: flip the D6 synaptic fact write default-on (pre-registration)

Shared steps (D, B, F, decision table, UNDEFINED rule): `2026-09-29-iw1-wave-battery-PREREGISTRATION.md`.

## Why this faculty (selection)

- **Reply impact.** The capability gate asks exactly the reply question: after "the wolf hunts the deer", does the
  reply to "what does the wolf hunt" depend on the synaptic write? K1-K7 GO 6/6
  (`2026-09-23-d6-learn-through-use-v3-capability-gate-GO-6of6.md`; the frozen write abstains, the host record is
  inert, the null is clean). Content provenance reads GO 6/6 with both mouths and puts the shipped default's
  learned-content fraction at 0.00 against 0.60-0.75 with these flags
  (`2026-09-24-content-provenance-learned-facts-prod-and-qwenforced-GO-6seed.md`). The chat-only AI teacher's taught
  facts are carried by this write, GO 6/6 (`2026-09-24-ai-teacher-chat-only-synaptic-learning-GO-6seed.md`).
- **Integration risk.** Every reader is wired already (`webapp/d6_hebbian_chat.py` docstring lists them); this
  packet adds one anchor and routes the five inline readers through it (tests/test_integration_wave1_anchors.py
  pins the truth table and that no production file reads the flags directly). The three findings name the missing
  legs themselves: "no combined-battery regression or production-default run".
- **Phase and stages.** Who owns the content: with the flags on, what the brain says it learned in conversation is
  carried by a synapse it wrote, not by the host list (K5). Stage 1 exit: lived-fact recall and word-to-concept
  recall. Owner 2026-09-24: knowledge is learned, never RAG.
- **Checked.** Not retracted (docs/RETRACTED.md has no D6 entry; `git log --all --grep` shows fixes, no
  retraction). Not on by default (all three readers default OFF at P; the ledger row names the fuller learn as the
  next depth).

## Step R: reply level, capability, at P

Row `d6-hebbian-write` (`research/runners/lbf_rows/d6_hebbian_write.py`): session `iw1d6`, turns teach "the wolf
hunts the deer" -> "what does the cat eat" -> "what does the dog chase" -> probe "what does the wolf hunt" (the
gate-v3 protocol). Intact = the three flags `=1`; lesion = the same plus `BRAIN_D6_HEBBIAN_FREEZE=1` (eta 0 for
in-conversation writes only). Fields: `answer`, `abstained`, `recalled_svo`. Tag `iw1-R-d6-<P7>`, six gate seeds,
`--repeats 2`, `tools/lb_shard.py jobs --no-fixes --faculties d6-hebbian-write --extra-env
BRAIN_D6_HEBBIAN_STORE=1 BRAIN_D6_ENGRAM_VOCAB=1 BRAIN_D6_ENGRAM_READTIME=1`.

- **R PASS:** `python tools/iw1_score.py R --packet d6-hebbian-write ...` reads all six seeds DEFINED and S2
  `regressed` (the lesion changes a reply field, the two intact builds agree, the lesion reproduces).
- **R FAIL:** any DEFINED seed reads `pass`. **UNDEFINED:** any seed not DEFINED after one re-run.
- Prediction: PASS (the same protocol is K1/K3 of the 6/6 gate, now through the battery's worker).

## Packet-specific production-default criteria (in addition to the shared B1, B2, F2, F3)

- **F1:** in `iw1-flip-<F7>` the row `d6-hebbian-write` (REQUIRED_ENV lifted by the flip) is reply-level
  load-bearing on 6/6 seeds with no BRAIN_* flag set.

## Declared residuals (unchanged by a flip; recorded so a GO is not read as more)

The write is local but teacher-forced: the instructive pathway is host-wired, the phase lock is a host loop, the
magnitude is clamped, and the capacity review measured the learned weights as a near-copy of the host pattern
(correlation 0.99996, quoted from the v3 finding's addendum) <!--derived--> Held/not-held is a host threshold on a neural read; the block-to-words map is host
bookkeeping; four kb readers are routed through the read-time view, the others are not
(`research/runners/d6_hebbian_store.py` shortcuts (a)-(j)). Per-turn cost grows with the number of taught facts
(ceiling between 50 and 500 on pool CPU). None of this is "self-organized" (docs/TERMS.md).
