# gap#3 WTA de-risk — SUPERSEDED (re-derivation of an already-CLOSED gap); a stale-pointer drift note (2026-07-23)

## What happened (honest)
The build-ahead queue built + ran `_gap3_multireferent_wta_disambiguation_derisk.py` (spiking WTA biased-competition
for multi-referent pronoun disambiguation) and it came back **5/6 aggregate GO** (wta 0.924 vs chance 0.455; bias-lesion
collapses to 0.424; recency 0.439 + salience 0.455 at chance BY CONSTRUCTION of a decorrelated battery). I initially
recorded this as "gap#3 mechanism CLOSED." **That is WRONG — gap#3 was already FULLY closed on 2026-07-18, more
completely than this de-risk.** Retracted.

## Why it's superseded (verified against the findings, not just the board)
`2026-07-18-gap3-A1-learned-feature-compatibility-cheap-first-GO.md` (verified to exist + match) already delivered:
- the biased-competition WTA disambiguation (6-seed GO, wired into `MultiTurnAgent`);
- **A1: the content-bias is NEURALIZED** — a learned SPIKING feature-compatibility map (`SpikingFeatureCompat`,
  corpus co-occurrence → feature-detector spikes on a real `SimulationBridge`) that **REPLACES** the host
  `content_bias_target` lexicon (learned == host 1.00; permuted-corpus collapses to 0.00), verified present in the
  deployed code (`multi_turn_agent.py` `build_referent_bias_from_experience`);
- A2 discourse-salience tie-break (6-seed GO); deployment default-on (learns the feat-compat from the SVO facts the
  agent HEARD). CI 7 gap3 + 8 regression.

This 2026-07-23 de-risk uses a **HOST-picked content-bias target** — i.e. it re-derives the WTA competition part
while sitting a full step BEHIND A1 (which already learned + spiked the bias). Its ONE non-redundant contribution is
the **decorrelated balanced battery** control design (randomize the recency cue and the salience cue INDEPENDENTLY of
the content-correct answer, so both documented 2026-06-17 negatives are uninformative *by construction*, not just
"fail once") — a cleaner anti-cheat than "permuted-corpus collapses," but of an ALREADY-closed capability. No new
gap movement.

## The drift + the lesson (drift mode #12 — stale-pointer / re-deriving concluded work)
The build-ahead's gap#3 builder READ the 2026-06-17 NEGATIVE (the wall) but did NOT RAG-check the CURRENT gap#3 status
(the 2026-07-18 closure) — so it built a de-risk for a gap that was already closed. I compounded it: I did not run the
a-1 check ("have we already closed gap#3?") before launching, and I reported a premature "CLOSED" to the owner.
**RULE reinforced: before building OR launching a de-risk for a gap, RAG-check the board's gap-status line AND the
most recent gap-N finding — the wall that motivated the mechanism is not the current state.** The build-ahead prompts
must point agents at the CURRENT gap status, not only the historical NEGATIVE that names the mechanism.

## Disposition
- gap#3 stays CLOSED per 2026-07-18 (unchanged). This de-risk adds no closure.
- The runner is retained (committed) as a redundant cross-check with the decorrelated-battery control; NOT a milestone.
- Applied the lesson to the rest of the build-ahead queue: re-checking each remaining candidate against its current
  gap status before launching (gap#4 seq-credit is a DEPRIORITIZED-per-board direction — deep-credit-beats-reservoir;
  gap#5 DG-detonator IS a legit attempt at the genuinely-open gap#5 replay-readout; gap#1 items open).
