# Direction E substrate Task 1 = BOUNDARY: both spatial (ec_context) AND temporal (theta-gamma) positional codes fail; v16 substrate fundamentally bounded for sequence storage

**Date:** 2026-05-24
**Status:** BOUNDARY (above chance, below strict bar; convergent with Direction A)
**Frozen bar:** 0.80 multi-seed STRICT TOP-1 (NEVER tuned)
**No-confab moat:** 7/7 green throughout
**Protected set:** byte-empty diff e8a99a2..HEAD

## Headline

Direction E substrate Task 1 (theta-gamma temporal phase positional
binding; catalog N.16 algebra pillar n=103 already validated) tested
on the v16 substrate produces multi-seed strict top-1 = **0.250**
(per-seed [0.000, 0.375, 0.375]; 4x chance 0.0625 but well below
the frozen 0.80 bar).

Combined with Direction A's earlier results (v1 0.333, v2 0.292),
**both spatial (ec_context) AND temporal (theta-gamma) positional
codes fail at the strict bar on the v16 substrate**. The engram tag
is load-bearing across both mechanisms; the positional cue (whether
spatial drive or temporal window) is NOT.

This is a precise biology-translatable finding: the v16 concept-pool
substrate (deliberately weak dynamics per the v14/v16 multi-concept
trainability design) cannot reliably do sequence-position retrieval
via either ec_context spatial codes or theta-gamma temporal phase
codes. The substrate that supports SIMULTANEOUS multitag binding
(91.7% multi-seed at pillar n=100/n=101) does NOT support SEQUENTIAL
slot-position retrieval.

## Smell test results (load-bearing strict top-1; multi-seed mean)

| Control | top-1 | Margin (main - control) | Interpretation |
|---------|-------|--------------------------|----------------|
| Main | 0.250 | -- | -- |
| (A) Wrong-slot window | 0.208 | **+0.042** | Slot window barely load-bearing |
| (B) No-stim | 0.000 | **+0.250** | Engram IS load-bearing |
| (C) No-window (full theta cycle) | 0.208 | **+0.042** | Slot window barely load-bearing |

The pattern is IDENTICAL to Direction A's smell test (engram
load-bearing; positional cue not). The mechanism degenerates to
multitag set-membership regardless of whether positional code is
spatial (ec_context) or temporal (theta-gamma).

## Comparison across the three substrate sequence-storage attempts

| Attempt | Mechanism | Multi-seed strict top-1 | Verdict |
|---------|-----------|--------------------------|---------|
| Direction A v1 | ec_context spatial, frozen plasticity | 0.333 | BOUNDARY |
| Direction A v2 | ec_context spatial, learned plasticity | 0.292 | BOUNDARY |
| Direction E Task 1 | theta-gamma temporal phase | 0.250 | BOUNDARY |

The three mechanisms cluster in ~0.25-0.33 strict top-1. Above chance
(0.0625) by 4-5x, but well below the bar. The convergent failure
across mechanisms localises the bottleneck precisely.

## Biology-translatable finding

The v16 substrate design (deliberate WEAK concept-pool dynamics per
the v14 "canon amplifies bias collapse" insight that enables
multi-concept trainability) trades off SIMULTANEOUS-binding capacity
against SEQUENTIAL-position retrieval. Specifically:
- The weak dynamics make all pool neurons fire ~equally during
  engram capture (top-K spike-count) regardless of which slot's
  drive caused them
- The engram tag captures roughly the same neurons across slots
- At retrieval, stim activates all slot-words' pools equally; no
  positional cue (spatial or temporal) provides enough selectivity
  to break the tie

Real biology solves this via:
- Stronger concept-pool dynamics (cortical pyramidals with NMDA
  bistability)
- Dedicated sequence-binding region (e.g., hippocampal CA3
  recurrent autoassociator + CA1 sequence cells + time cells per
  catalog D.11)
- Theta-gamma + hippocampal trisynaptic loop interactions

The v16 substrate has neither stronger dynamics (deliberately
omitted) nor hippocampus (the validated substrate-readiness chain's
HIPPO-OPTION3 variant n=97 has hippocampus -- but the multitag pillar
n=101 tested only static binding, not sequence retrieval).

## Code + reproducibility

- Task 1 runner: research/findings/raw/direction_E_substrate_task1_full.py
- Task 1 smell test: research/findings/raw/direction_E_substrate_task1_smell_test.py
- Task 1 post chain: research/findings/raw/direction_E_task1_post_chain.py
- Task 1 grounding pin: research/findings/raw/direction_E_substrate_task0_grounding.py (GREEN)
- Direction A v1+v2 chain: research/findings/raw/direction_A_*

Both remote (origin + gitea) propagated. Bar FROZEN at 0.80 throughout
the arc.

Cached per-seed bridges + trials:
- direction_A_ec_context_cache/ (Direction A; 75 MB)
- direction_E_substrate_task1_cache/ (Direction E Task 1; ~75 MB)

Smell test reproducible from caches; full re-run requires ~3 hr GPU
per direction.

## What's next (next genuinely-different direction)

The chain recommendation: NEGATIVE_HONEST_BOTH_SUBSTRATES_BOUNDED.
Architectural changes are needed to clear the bar. Candidate next
directions (each its own pre-registered test):

1. **Direction G: substrate with hippocampus + theta-gamma**. Re-run
   Task 1's mechanism but on HIPPO-OPTION3 substrate (n=97 builder
   with hippocampal trisynaptic loop). Hypothesis: CA3 recurrent
   autoassociator provides the per-slot pattern-completion that
   bare concept pools cannot. ~3-4 hr GPU.

2. **Direction H: substrate with stronger concept-pool dynamics**.
   Modify v16 builder to use canon dynamics (motor-pool style) on
   concept pools; test if this enables positional binding. RISK:
   may break v14/v16 multi-concept trainability (the original
   "canon amplifies bias collapse" finding). Pre-registered control:
   verify multi-concept Phase 1 binding still passes before claiming
   sequence-storage PASS.

3. **Direction I: dedicated sequence-binding region (PFC sequence
   buffer)**. Add a new BrainRegion (e.g., `pfc_seq_buffer`, ~200
   NMDA-bistable neurons) that holds the active sequence frame;
   binds (word, position) tuples; PFC -> concept pools provides
   the slot-selective drive. ~2-4 weeks design + build.

4. **Direction J: accept BOUNDARY**. The substrate's sequence storage
   limit is bounded; record honestly; move to other capabilities
   (e.g., scale concepts further via G.20 sparse encoding;
   improve cross-bridge composition; chat REPL with
   multitag-only sequences).

Direction G is the cheapest-falsifiable next test (reuses HIPPO-OPTION3
builder byte-unchanged; same Task 1 mechanism with hippocampus
present). If G passes, hippocampus IS the missing ingredient (biology-
translatable; catalog D.04 + D.11 vindicated). If G also fails, the
substrate's positional-binding limit is deeper than just hippocampus
absence.

## Honest scope

This is a BOUNDARY finding, not a NEGATIVE. The mechanisms produce
real signal (4-5x chance); just not robust enough for the bar. The
substrate IS doing something with positional codes; just not enough.
The honest characterization is: "spatial AND temporal positional
codes on v16 cortical-only substrate produce partial sequence-
position retrieval; full-bar retrieval requires architectural
augmentation (hippocampus OR stronger dynamics OR dedicated
sequence buffer)."

This is consistent with the prior overnight finding (n=99 (c) generative
replay arc characterised REPLAY_DOESNT_REACTIVATE); the substrate's
sequence-position structure was missing, and the candidate fixes
(ec_context spatial + theta-gamma temporal) BOTH provide only partial
recovery.

## Discipline preserved

- Bar FROZEN at 0.80 multi-seed STRICT TOP-1 throughout the arc
- No protected/frozen/moat module modified (e8a99a2..HEAD byte-empty
  diff across full protected set)
- No autograd
- Reuse-by-import only
- No-confab moat 7/7 green
- Both remotes propagated throughout
- Adversarial review caught critical methodology defect early; all
  STRENGTHEN-only fixes applied; no bar weakening
