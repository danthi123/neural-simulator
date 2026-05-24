# Direction K: honest re-classification — substrate grounding NOT load-bearing; pillar n=105 NOT validated

**Date:** 2026-05-24
**Status:** CHARACTERIZATION extending pillar n=104 (not a new pillar)
**Frozen bar:** 0.80 multi-seed STRICT TOP-1 (NEVER tuned throughout)
**Reviewer verdict:** BLOCK with 4 STRENGTHEN-only fixes; 2 implemented confirmed reviewer correct

## Headline

Direction K substrate-grounded FHRR sequence storage achieved
multi-seed strict top-1 = 1.000 (24/24 sequences correct across 3
seeds, no-teacher fair test). HOWEVER, fresh-agent adversarial
reviewer (ad5cdaf811e120e0d) returned BLOCK with the critical
finding: substrate grounding is NOT load-bearing at the tested
N_DIM=3200; the PASS is FHRR algebra at dim-overkill regime
(capacity ratio M/N = 3/3200 = 150x under FHRR theoretical capacity
0.15) working over arbitrary distinguishable codes.

Two reviewer-recommended STRENGTHEN probes confirmed the diagnosis:

**(#4) UNTRAINED-bridge exploit:** built FRESHLY UNTRAINED bridge
(no v16 training); ran the same FHRR sequence storage. RESULT:
PERFECT 1.000 multi-seed (per-seed [1.0, 1.0, 1.0]). Substrate
training contributes NOTHING.

**(#2) Dim-scaling probe:** swept N_DIM {64, 128, 256, 512, 1024,
3200} comparing substrate-like (Gaussian-with-bias matching 0.20
overlap) vs random sign vectors. RESULT: random sign vectors BEAT
substrate-like at EVERY tested N_DIM (random=1.000, substrate-like=
0.60-0.75).

## Honest finding

The FHRR algebra mechanism works at N_DIM=3200 / M=3 bundled items
for ANY distinguishable vector representation:
- Random sign vectors: 1.000 (the highest achievable)
- Substrate spike-count vectors (real): 1.000 (these are sparse + 
  positive; happen to be distinguishable enough)
- Gaussian-with-bias substrate proxy: 0.60-0.75 (this proxy is more
  overlapping than real substrate, hence lower)
- UNTRAINED-substrate spike counts: 1.000 (untrained substrate
  routes via random initial weights; still distinguishable enough)

**The substrate is NOT uniquely required.** Any 16 reasonably
distinguishable 3200-dim vectors would produce the same 1.000 PASS.

Per reviewer: "Pillar n=105 NOT VALIDATED as 'substrate-grounded
FHRR sequence storage'; the underlying mechanism is sound algebra
at dim-overkill regime, but the discriminating claim against pillar
n=104 (substrate as bottleneck) is not earned by this experiment."

## What's still true (pillar n=104 stands, extended)

Pillar n=104 records: v16 cortical-only substrate fundamentally
bounded for sequence-position retrieval via engram-tag mechanism
(4 convergent attempts at 0.25-0.33 multi-seed strict top-1).

This Direction K characterization EXTENDS n=104 with:
- The bound at 0.25-0.33 is in the ENGRAM-TAG MECHANISM, not the
  substrate per se
- FHRR algebra over arbitrary distinguishable representations clears
  the 0.80 bar at dim-overkill regime (1.000 multi-seed)
- The substrate provides distinguishable codes; the algebra does the
  binding; together they clear the bar
- The substrate's specific contribution (vs random codes) is NOT
  measurable in this test

The biology-translatable finding: the engram-tag mechanism specifically
fails because the top-K capture aggregates uniformly across slots; an
algebraic mechanism (FHRR or any other reversible composition) over
any reasonably distinct per-word representations works at sufficient
dim.

## Reviewer recommendations still pending (queued for future work)

**(#1) Scope-tighten pillar claim** — DONE: no pillar n=105
recorded; this characterization doc supersedes any earlier
"validated substrate-grounded FHRR" framing.

**(#3) Route through validated FHRR biologization stack** —
QUEUED: the Direction K runners use `cosine_real()` simple cosine
matching, NOT the validated biologized clean-up (resonate-and-fire
+ attractor + familiarity gate from pillar n=87). A future probe
should re-run with the biologized pipeline; if PASS holds AND
smell-test (B) drops, the biologized substrate-grounded claim is
honest.

## What this means for the project's conversational arc

After today's complete chain (~80 commits, 6.5 hr GPU total):
- Substrate CAN do SIMULTANEOUS multitag binding (pillar n=100/n=101
  at 91.7% multi-seed; validated; production-recipe-grade)
- Substrate CANNOT uniquely do SEQUENTIAL positional binding via
  engram-tag mechanism (pillar n=104; 4 convergent BOUNDARY attempts
  including hippocampus extension)
- FHRR algebra CAN do sequence retrieval at dim-overkill regime
  (Direction K characterization; works for any distinguishable
  codes; substrate's specific contribution not load-bearing)
- The honest path to substrate-grounded sequence storage requires
  EITHER:
    a. Lower-dim test where substrate's specific sparse positive
       structure beats random (reviewer fix #3 territory)
    b. Routing through validated biologized FHRR pipeline (pillar
       n=87 mechanism) — would demonstrate biology-faithful sequence
       storage explicitly
    c. Different substrate architecture (Direction H canon dynamics;
       Direction I PFC sequence buffer) — risky for v14/v16
       trainability OR longer build

## Discipline preserved

- Bar FROZEN at 0.80 multi-seed STRICT TOP-1 throughout
- No protected/frozen/moat module modified (e8a99a2..HEAD byte-empty
  diff)
- No autograd
- Reuse-by-import only
- No-confab moat 7/7 green
- Both remotes propagated throughout
- Adversarial reviewer BLOCK respected; STRENGTHEN-only fixes
  implemented; no bar weakening; honest re-classification
- HONEST PROPAGATION OF EVERY OUTCOME (positive AND negative)

## Discipline lesson

The Direction K arc demonstrates the reviewer-driven discipline at
its best: an initial 1.000 PASS claim was scrutinized via dedicated
fresh-agent adversarial review; the reviewer correctly identified
the dim-overkill artifact; two STRENGTHEN probes confirmed; the
claim was re-scoped honestly. This is exactly the pattern the
project's "scrutinize PASS harder than FAIL" discipline is for.

## Next direction (autonomous chain continues)

Per discipline (no hand-back), the next concrete direction:
- Reviewer fix #3 (biologized FHRR pipeline + substrate sequence
  storage) is the cheapest scientifically-meaningful next test
- Estimated cost: ~3-5 hr GPU + ~2 hr coding to integrate biologized
  pipeline with sequence task
- Alternative: pivot to other capabilities (substrate's validated
  multitag mechanism is already useful for sentence-level retrieval;
  can build chat REPL on top)

The autonomous decision: continue with biology-grounded testing per
the discipline. The next cheap-first probe is the biologized FHRR
pipeline + sequence storage test.
