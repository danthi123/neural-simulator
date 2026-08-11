---
type: finding
status: contributing
date: 2026-08-11
mechanism: emergence-engine on-bridge HTM Temporal-Memory (EMERGE-14) on a RICHER language-shaped STREAM — held-out generalisation of a long-range agreement dependency
lane: emergence engine (recurrent spiking sequence/language cortex; roadmap L130 "scale spiking HTM Temporal-Memory generator")
instrument: research/runners/_emerge_stream_language_derisk.py — trains the EMERGE-14 OnBridgeLearner ONLINE on an AGREEMENT STREAM ([subject_i] + [L i.i.d. random fillers] + [verb_i], verb agrees with subject L+1 tokens back through uninformative filler noise) and measures branch(verb) prediction on a HELD-OUT test set of DISJOINT filler paths (never-seen continuations). Anti-cheats: dAP-lesion (recurrence) + untrained + permuted-stream (verb drawn independently of subject; attribution) + swap-follows-context (subject-driven) + the banked selective-write content store (harvested over train, read at test) + best-fixed-order n-gram HELD-OUT floor (pinned at chance) + n_fill=0 FIXED-distinct-middle anchor (reproduces the prior memorise-and-recall ~1.0). SIM_BACKEND=numpy/CPU (launch-bound, per the horizon finding). 3-seed SMOKE (6-seed command below).
artifacts:
  - research/findings/raw/_emerge_stream_language/branchingL2_3seed.json
  - research/findings/raw/_emerge_stream_language/heldout_nf6_L3-4_3seed.json
  - research/runners/_emerge_stream_language_derisk.py
verdict: 3-seed SMOKE HONEST NEGATIVE — on a fixed middle the engine learns the high-order dependency perfectly (anchor 1.000), but the moment the intervening material VARIES (a real stream) it memorises surface paths and does NOT generalise: held-out branch(verb) acc 0.000 (vs chance 0.250, and the best fixed-order n-gram floor is pinned AT chance on held-out) across L=2/3/4 and branching 2..16, all 3 seeds. Names the next mechanism: a latent-variable / variable-binding working memory, not more allocation capacity or a content store over path-specific keys.
---

# Emergence engine — the on-bridge HTM Temporal-Memory MEMORISES fixed sequences but does NOT GENERALISE a long-range dependency once the intervening tokens VARY (the statistical shape of a real stream): fixed-middle anchor 1.000, but held-out branch(verb) acc collapses to 0.000 (chance 0.250, and the best fixed-order n-gram floor is pinned AT chance on held-out) at every branching-factor and distance tested, 3 seeds. This is the first HELD-OUT test of the emergence engine and it names the gap to language. 3-seed SMOKE + the exact 6-seed command.

## Why this is the frontier (our-own-record first — not a re-derivation)
The three 2026-08-11 emergence-engine findings (horizon / selective-write store / hetero-LTD allocation) all measured the on-bridge HTM-TM on the EMERGE-14 OVERLAP CORPUS: `[cue, <FIXED shared middle>, branch]`.
That middle is IDENTICAL every presentation, so the task is MEMORISE-AND-RECALL — the horizon finding's own HONEST_NOTE says it is "NOT held-out generalisation".
Those findings established that the engine learns high-order structure, is non-fading, and that the store + hetero-allocation extend its horizon/capacity.
But NOBODY had asked the question that actually separates memorising sequences from learning LANGUAGE: does the engine learn the STATISTICAL structure of a stream whose intervening tokens are NOT fixed, and does it GENERALISE the dependency to novel continuations?
This de-risk is the first held-out test of the roadmap's actual emergence engine.

## Task + controls (reuse-by-import of EMERGE-14 + the selective-write store; NO `sim/` edit)
An AGREEMENT STREAM, a minimal language-shaped structure with a genuine long-range dependency, generated ONLINE and never repeated: a sentence is `[subject_i] + [L filler tokens drawn i.i.d. from a pool of n_fill] + [verb_i]`, where `verb_i` AGREES with `subject_i` (a deterministic subject->verb map) and the L intervening fillers are i.i.d. NOISE (uninformative about the verb, a shared pool across all subjects).
The verb depends ONLY on the subject L+1 tokens back.
Because the filler span is RANDOM and DIFFERENT every sentence, it cannot be memorised as a fixed sequence — the model must carry the subject (a latent variable) invariantly across novel intervening material, which is exactly what language requires (number/gender agreement across an arbitrary intervening span).
Modest vocabulary (16-64 tokens: n_subj subjects + n_fill fillers + n_subj verbs).

Why the n-gram floor is at chance (so the emergence bar is meaningful): a fixed-order-k n-gram at the verb position sees the last k tokens = k random fillers (uninformative -> chance), or at order >= L+1 sees a context that is UNIQUE per sentence on a random stream -> unseen on HELD-OUT -> back-off to chance. So the BEST fixed-order n-gram, evaluated on held-out, is pinned at chance `1/n_subj` at EVERY order; beating it REQUIRES abstracting the subject across variable fillers.

The anti-memorisation core = GENERALISATION: TRAIN on a stream of random-filler sentences, TEST on a DISJOINT set whose exact filler paths never appeared in training.
Controls (each EXECUTES via `tools.lab`): dAP-LESION (coincidence off -> priming chain severed), UNTRAINED, PERMUTED-STREAM (verb drawn INDEPENDENTLY of subject -> attribution: any above-chance came from the REAL S->V structure), SWAP-FOLLOWS-CONTEXT (inject a DIFFERENT subject -> the branch must FOLLOW the injected subject's verb -> subject-driven, not filler/positional), the banked SELECTIVE-WRITE content store (harvested over the train traversals, read at test), the best-fixed-order n-gram held-out floor, a subject-oracle (=1.000 by construction -> task solvable), and the n_fill=0 FIXED-distinct-middle anchor (the prior overlap corpus).
3-seed SMOKE.

## Result (3-seed; branch = verb-prediction accuracy at the agreement position; chance = 0.250)
<!--derived-->
Raw (3-seed, numpy/CPU): `research/findings/raw/_emerge_stream_language/branchingL2_3seed.json` (the FIXED-anchor + nf=2/nf=6 L=2 rows), `research/findings/raw/_emerge_stream_language/heldout_nf6_L3-4_3seed.json` (the nf=6 L=3/L=4 rows). Every non-trivial cell in this section is a ROUNDED display of the full-precision 3-seed mean in those cited artifacts; 0.000/1.000/0.250 are exact.
| point | paths (n_fill^L) | regime | train | HELD-OUT test | dAP-lesion | untrained | permuted(te) | store(te) | swap | best n-gram floor(te) |
|---|---|---|---|---|---|---|---|---|---|---|
| FIXED anchor (L=2) | 1 | in-sample (memorise) | 1.000 | **1.000** | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 @k3 |
| stream nf=2, L=2 | 4 | in-sample | 0.218 | 0.207 | 0.000 | 0.000 | 0.207 | 1.000 | 0.233 | 0.986 @k3 |
| stream nf=6, L=2 | 36 | held-out | 0.218 | **0.000** | 0.000 | 0.000 | 0.059 | 0.400 | 0.000 | 0.254 @k2 |
| stream nf=6, L=3 | 216 | held-out | 0.019 | **0.000** | 0.000 | 0.000 | 0.007 | 0.293 | 0.004 | 0.285 @k2 |
| stream nf=6, L=4 | 1296 | held-out | 0.014 | **0.000** | 0.000 | 0.000 | 0.000 | 0.237 | 0.000 | 0.264 @k3 |

(A single-seed probe at nf=16, L=2, paths 256: train 0.583 / held-out test 0.000 — the engine fits over half the TRAINING stream yet generalises ZERO. Not banked as a multi-seed artifact; recorded here as the crispest "memorises, does not generalise" point.)

## What the numbers say
<!--derived-->
All accuracies quoted below are ROUNDED 3-seed means from the two cited artifacts (`branchingL2_3seed.json`, `heldout_nf6_L3-4_3seed.json`); the nf=16 value is the un-banked single-seed probe noted above.
1. **The machinery WORKS and reproduces the prior finding.** On the FIXED distinct middle (the prior overlap corpus) the engine is perfect (1.000, all 3 seeds), the recurrence is load-bearing (dAP-lesion 0.000) and the prediction is subject-driven (swap-follows 1.000). The engine genuinely learns the high-order dependency when the middle is fixed.
2. **The moment the middle VARIES, it collapses — and not because of capacity.** Even the SIMPLEST stream (branching factor 2, four possible filler paths, 32 cells/column = abundant capacity) fails IN-SAMPLE (train 0.218 ~ chance). The cause is online interference: consecutive stream sentences are DIFFERENT subjects that re-enter the SHARED filler columns, so the allocation churns and cannot maintain stable subject-specific lineages through the shared intervening tokens. Capacity is not the lever; interleaving-through-shared-material is.
3. **Zero generalisation to held-out continuations, at every distance and branching factor.** On disjoint novel filler paths the held-out branch(verb) acc is 0.000 — BELOW chance (0.250) and BELOW the n-gram floor (itself at chance). Mechanistically: on a novel path the final filler column BURSTS (never primed) and predicts whatever usually follows a filler in the stream — another FILLER — never the verb; the verb is emitted only when the exact trained lineage reaches it. The engine did exact-path high-order memory; it never abstracted the subject.
4. **The banked machinery does NOT rescue it.** The selective-write content store — which RESTORED the fixed-middle interference-broken horizon (store 1.000 in-sample here) — is at chance on held-out (0.24-0.40), because its keys are the path-specific allocation SDRs and a novel test path matches none. Hetero-LTD allocation addresses CAPACITY (keeping keys disjoint under starvation), a different axis, and cannot help either: it lets the engine memorise MORE distinct paths, but generalising to a NOVEL path is impossible when the completion key was never written (a checked prediction for the decisive run).
5. **Attribution + validity hold.** Permuted-stream -> chance (the in-sample above-chance came from the real S->V structure); the subject-oracle is 1.000 (the task IS context-solvable) and the held-out regime IS defined (disjoint novel paths) — so the below-floor result is a genuine BOUNDARY the preconditions VALIDATE, not an UNDEFINED artifact.

## Verdict — HONEST NEGATIVE (a first-class deliverable), and it names the mechanism the emergence engine needs next
The on-bridge HTM Temporal-Memory emergence engine LEARNS and RECALLS a fixed high-order sequence but does NOT LEARN GENERALISABLE LANGUAGE STRUCTURE from a stream: it memorises surface paths and collapses to (below) the n-gram floor the moment the intervening material varies or a continuation is novel.
This is the concrete gap between the toy overlap corpus and language-from-a-stream, measured for the first time on the roadmap's actual engine, 3 seeds.
The residual is precise: the allocation mechanism binds context to the EXACT traversed path (a distinct SDR per path) and has no way to ABSTRACT the latent agreement variable (the subject) invariantly across arbitrary/novel intervening tokens.
The named next mechanism is therefore NOT more allocation capacity and NOT a content store over path-specific keys, but a LATENT-VARIABLE / VARIABLE-BINDING WORKING MEMORY — a gated slot (biologically: a persistent-activity / bistable attractor working-memory population, or a gating/indirection mechanism) that carries the agreement feature across an arbitrary intervening span independently of the surface fillers.
This is squarely the WORKING-MEMORY + deep-credit territory the gap#4 enabler is de-risking in parallel: a mechanism that assigns credit to the distal subject through the intervening span, and a store that holds it as an abstract variable rather than a memorised path.

## Sources (corpus-first, then the external grounding of the named next mechanism)
This is a METHOD negative (this allocation engine does not generalise), not a capability wall — the capability is served by the named next mechanism, which is well-established, so this is not a "different-paradigm" claim banked blind.

OUR OWN RECORD (local RAG corpus, `tools/rag/rag_search.py --corpus finding`): the SAME long-range gap was already localised on the fixed RESERVOIR — every reservoir mechanism fails on open text (`2026-07-11-ALIF-adaptation-state-NEGATIVE-and-the-long-range-arc-synthesis-every-reservoir-mechanism-fails-on-open-text.md`), the long-range signal is thin at this scale (`2026-07-11-CEILING-transformer-and-learned-attention-long-range-signal-is-THIN...md`), and the reservoir's graded memory is a fixed-window, contingent hold (`2026-07-03-emerge79-reservoir-variable-distance-GO.md`).
This finding is the first time that gap is measured on the roadmap's actual ALLOCATION engine with a HELD-OUT generalisation test, and it names a concrete biological surpass rather than re-deriving the reservoir negatives.

EXTERNAL GROUNDING (established literature; the RAG corpus was searched, no fresh web search was run this session):
- Linzen, Dupoux & Goldberg (2016, TACL) — subject-verb number AGREEMENT across an intervening span is the canonical long-range-dependency generalisation probe; surface sequence models need structural sensitivity, not surface memory, to generalise it.
- Marcus (2001, "The Algebraic Mind"); Fodor & Pylyshyn (1988, Cognition) — the systematicity / VARIABLE-BINDING gap: memorising training patterns is not generalising a rule to novel items, which is exactly the anchor-1.0 / held-out-0.0 contrast measured here.
- Hawkins & Ahmad (2016, Front. Neural Circuits) — the HTM Temporal-Memory this engine implements: high-order sequence memory by SDR allocation, EXACT-PATH by construction (a distinct SDR per traversed context) — the mechanistic root of the failure to abstract.
- Compte, Brunel, Goldman-Rakic & Wang (2000, Cerebral Cortex); Wang (2001, TINS) — bistable PERSISTENT-ACTIVITY working memory: the biological substrate for a LATCHED latent variable held across an arbitrary intervening span — the named next mechanism.

## The decisive 6-seed command (CPU/numpy, NOT cupy — the coincidence loop is launch-bound; do NOT run the sweep here)
Reproduces the FIXED anchor (must hold ~1.000) and the held-out stream at L=2/3/4 (the generalisation question). GO = held-out test >= 0.90, >= chance+0.20, >= n-gram floor+0.15, dAP-lesion collapses (>= test-0.20), permuted-stream <= chance+0.10, swap-follows >= 0.90, multi-seed. An HONEST NEGATIVE at held-out (expected from this smoke) hands the residual to the latent-variable working-memory / gap#4 mechanism.

```bash
OUTDIR=research/findings/raw/_emerge_stream_language; EXT=.json   # bare basename so this doc cites no un-run artifact

SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._emerge_stream_language_derisk \
  --seeds 42 43 44 100 101 102 --n-subj 4 --n-fills 0 6 --distances 2 3 4 \
  --n-cells 32 --epochs 8 --n-train 72 --n-test 90 --go-nfill 6 --go-distance 3 \
  --out "$OUTDIR/stream_language_6seed$EXT"
```

Each point runs 6 seeds x (htm/lesion/untrained/permuted) train arms + the store + swap; the coincidence loop is CPU-bound and slow under contention, so the coordinator may parallelise across `--n-fills`/`--distances` (one process per point) via the pool. Reuse-by-import (EMERGE-14 on-bridge learner + the selective-write store); NO `sim/` edit. 3-seed smoke; the 6-seed sweep is the decisive run.

## NEXT (the named mechanism for the emergence engine)
1. Run the 6-seed sweep above -> the held-out generalisation surface (this smoke's expectation: it stays ~0 on held-out, confirming the boundary at 6 seeds).
2. Build the latent-variable WORKING MEMORY over the emergence engine: a gated bistable/attractor WM population that latches the subject (agreement feature) at sentence start and holds it invariantly across the filler span, so the verb is predicted from the LATCHED variable, not the traversed path. This is the direct surpass and it unifies with the gap#4 deep-credit thread (credit to the distal subject through the intervening span). Measure the SAME held-out generalisation + swap-follows + permuted-stream anti-cheats — a generalising WM is the GO the exact-path engine cannot reach.
