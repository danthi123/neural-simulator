---
type: biology
id: repetition-retrieval-strengthen-same-trace
mechanism: Meeting the same information again -- told again after a gap, or recalled -- reactivates and strengthens the existing memory instead of adding a separate copy -- spaced repetitions recruit synapses the first episode missed, reactivation makes the trace labile and its restabilization folds the new learning into it, and retrieval is itself a stronger learning event than restudy, so repetition and use are how a brain decides that something matters
status: proposed
last_verified: 2026-09-25
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "Five spaced training sessions"
    note: "ch.53 (Aplysia): five spaced training sessions over about an hour produce long-term sensitization lasting a day or more, and spaced training over several days produces sensitization lasting weeks; the conversion needs new mRNA and protein synthesis."
  - path: "PMC3323981"
    anchor: "markedly enhanced previously saturated LTP if spaced apart by 1 h or longer, but were without effect when shorter intervals were used"
    note: "Kramar et al. 2012, PNAS 109:5121 (doi 10.1073/pnas.1120700109), abstract via PubMed 2026-09-25: the added LTP from a delayed bout recruited synapses the first bout missed, because fewer than half of spines are primed at baseline. A synaptic account of the spacing effect."
  - path: "PMID:16719566"
    anchor: "the ISI producing maximal retention increased as retention interval increased"
    note: "Cepeda, Pashler, Vul, Wixted & Rohrer 2006, Psychol Bull 132:354 (doi 10.1037/0033-2909.132.3.354), abstract via PubMed 2026-09-25: meta-analysis of 839 assessments of distributed practice; 'ISI and retention interval operate jointly to affect final-test retention', so a spacing comparison must hold the retention interval (last presentation to test) equal across its arms -- the lag design of the battery's Rsp / Rms cues."
  - path: "PMID:18276894"
    anchor: "Repeated studying after learning had no effect on delayed recall, but repeated testing produced a large positive effect"
    note: "Karpicke & Roediger 2008, Science 319:966 (doi 10.1126/science.1152408), abstract via PubMed 2026-09-25: the testing effect -- retrieval practice, not re-exposure, carries delayed retention."
  - path: "PMID:16507066"
    anchor: "took one or three immediate free-recall tests, without feedback, or restudied the material the same number of times"
    note: "Roediger & Karpicke 2006, Psychol Sci 17:249 (doi 10.1111/j.1467-9280.2006.01693.x), abstract read via PubMed 2026-09-25: on the delayed tests (2 days, 1 week) prior testing without feedback produced substantially greater retention than restudy. The strengthening is triggered by the act of retrieving, with nobody telling the learner whether the answer was right."
  - path: "PMID:16248758"
    anchor: "increased production of multiple-choice lures as incorrect answers on the final test"
    note: "Roediger & Marsh 2005, J Exp Psychol Learn Mem Cogn 31:1155 (doi 10.1037/0278-7393.31.5.1155), abstract read via PubMed 2026-09-25: reading more multiple-choice lures on a test increased the production of those lures as wrong answers on a later test ('may inadvertently lead to the creation of false knowledge'). This is the risk side of test-time learning, shown for wrong alternatives READ at test; it does not show directly that a produced wrong answer is strengthened. The case for a retrieval-triggered rule without a correctness grader rests on Roediger & Karpicke 2006 (no feedback needed); this source is why the design's no-confabulation gate must watch it."
  - path: "PMID:18849987"
    anchor: "one normal function of hippocampal memory reconsolidation in rats is to modify the strength of a contextual-fear memory as a result of further learning"
    note: "Lee 2008, Nat Neurosci 11:1264 (doi 10.1038/nn.2205), abstract via PubMed 2026-09-25: reactivation destabilizes the memory and restabilization updates its strength -- further learning strengthens the SAME trace."
  - path: "PMC4749834"
    anchor: "retrieving a memory shortly after it was encoded prevented loss of both central and peripheral details"
    note: "Sekeres et al. 2016, Learn Mem 23:72 (doi 10.1101/lm.039057.115), abstract via PubMed 2026-09-25."
findings:
  - research/findings/2026-09-25-prioritized-memory-remember-what-matters-DESIGN.md
---

# Repetition and recall strengthen the memory you already have

**The constants this replaces (read in the code and the record, 2026-09-25).**
- A re-told fact is stored as a NEW block. The load-dependent renormalization smoke (seed 42, 1 of 6,
  research/findings/2026-09-25-sleep-forgetting-interference-fi-seed42-smoke.md) shows it: on the re-mention arm the
  original block fell like the unmentioned fact's, and the recall matched the day-3 re-mention block. The protection
  came from a second copy. That is append-only storage, the RAG-like pattern the owner ruled out.
- `OneBrainComposer.update_on_mismatch` (reconsolidation) already finds the cued trace with the brain's own read and
  rewrites it in place when the new filler carries a prediction error; a fully predicted re-statement is
  "restabilize" with nothing written, so a repetition leaves the trace exactly as it was.
- Reads never write: a correct recall changes nothing (the systems-consolidation protocol rule "the recall read must
  not write" exists because a live Hebbian read once drifted the store, and it must be kept for the read itself).

**What the real system runs.** A repeat after a gap recruits synapses the first episode missed (Kramar 2012; Kandel
ch.53 for the spaced-training protocol); reactivation plus restabilization strengthens the same trace (Lee 2008);
retrieval is a stronger learning event than restudy, even without feedback (Karpicke & Roediger 2008; Roediger &
Karpicke 2006), wrong alternatives met at a test can become later answers (Roediger & Marsh 2005, shown for lures
read, not for answers produced), and early retrieval prevents detail loss (Sekeres 2016);
the best spacing grows with how long the memory must last (Cepeda 2006).

## How the code is expected to bind to it (design only, nothing implemented)

Design doc step 3: the composer's own cued-block read decides "this is a trace I already have" (the spiking cue-match
sequencer, so Step 3's arms run with `BRAIN_INTEGRATED_LOOP=1`; with it off, as in production today, the read is a
host first-match string compare); a predicted
re-statement, or the brain's own retrieval event (its cued read selected a managed block and the reply did not
abstain, whether or not the answer is right -- no host grader), then re-induces early LTP and re-sets the tag on THAT
block through the ledger (the awake-replay rule e <- e + R (1 - e), reused), after the reply, so the read itself never
writes; synapses left unprimed
by the first episode become ready on an hour scale, so a spaced repeat recruits them and a massed one does not. The
host `kb` list must not be the dedupe key. No `constraints_config`.

⚠️ **Provenance honesty.** The Kandel anchor is in the local corpus. The rest were read as abstracts through PubMed on
2026-09-25.
