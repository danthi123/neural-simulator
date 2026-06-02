# Richer conversation on the full-320 biological substrate -- KB capacity + negation/QA -- 2026-06-02

Follow-on to the full-320 flat-distinct composition milestone (2026-06-02-full-320-flat-distinct-composition-
RESOLVES-multiseed.md). The composition is validated; this asks how rich a CONVERSATION the validated 320
substrate supports -- knowledge-base size and the K=4 negation/yes-no/who-QA stack -- reusing the validated
spiking machinery on the cached 320 distinct codes (no retraining).

## (1) Knowledge-base capacity -- holds to AT LEAST 15 facts, perfect multi-seed
Stores N separately-bound facts (each a K=3 agent/action/patient spiking bind), then per fact runs the
relational query (cue an agent -> find its fact among N -> read its patient) + a role query + an absent-cue
abstention control. N in {5,10,15}, seeds 42/43/44, 6 KB draws per (N, seed). bias=-500, window=150.

| N facts | relational | role | abstention control |
|--------:|-----------:|-----:|-------------------:|
| 5  | 1.000 | 1.000 | 1.000 |
| 10 | 1.000 | 1.000 | 1.000 |
| 15 | 1.000 | 1.000 | 1.000 |

**VERDICT: reliable relational KB holds to at least N=15 facts, perfect multi-seed -- no ceiling reached.**
This is ~3x the prior cap (~5 spiking / ~12 numpy on the small-vocab denoise64 substrate, finding
2026-05-31, KB-scaling). 270 relational queries (15x6x3) all correct + perfect abstention.

### Scrutiny of the perfect scores (a PASS scrutinised harder than a FAIL)
A wall of 1.000s is suspicious by default; here it is mechanistically explained by the substrate change, not
a too-easy test:
- The prior ~5-fact cap was set by OVERLAPPING codes (denoise64 between-cos ~0.70) -> cleanup errors compound
  as facts accumulate. The 320 flat-distinct codes are near-orthogonal (between-cos mean 0.045) -> cleanup
  stays clean even with many competing facts. Distinct coding is exactly what should raise KB capacity.
- The absent-cue control is 1.000 (no false matches) -> the cleanup DISCRIMINATES (it is not trivially
  returning a fact for every cue). This is the anti-artifact check and it holds.
- Separate-fact storage (each fact an independent bound vector) means no superposition interference; the
  capacity question is purely cleanup reliability, which clean codes deliver.
- Honest caveat: the ceiling is ABOVE 15 (untested higher). The claim is "holds to >= 15, no ceiling found at
  the tested range," not a measured maximum. A higher-N probe (20/30/50) would locate the actual ceiling.

## (2) Negation + yes/no + who-QA (K=4 polarity stack)
PENDING (job bbxy4s5kl) -- folded in when it lands. Ports the validated negation mechanism (a bound POLARITY
tag AFFIRM/NEGATE, K=4) + who-QA to the 320 substrate; the genuine question is whether the 4th role still
answers yes/no correctly on the noisier real 320 codes (composition was validated at K=3). Multi-seed, with
the unknown-fact abstention control + a readable transcript.

## Why this matters (on the goal)
The owner's goal is conversation built on the brain-analogue mechanism. The composition milestone showed the
320-concept substrate composes structured facts robustly; this shows it supports a genuine MULTI-FACT
knowledge base (>= 15 facts, perfect retrieval + abstention) and (pending) negated/polar questions -- the
substrate behaves like a small queryable, honestly-abstaining knowledge base, in spiking, multi-seed. Scope
unchanged: concept codes GIVEN by sparse encoding; the composition + KB + QA on top are genuine.

## Reproduce
```
python -m research.findings.raw._insubstrate_flatdist320_kb_capacity_test       # KB capacity
python -m research.findings.raw._insubstrate_flatdist320_negation_qa_test       # negation/yes-no/who-QA
```
