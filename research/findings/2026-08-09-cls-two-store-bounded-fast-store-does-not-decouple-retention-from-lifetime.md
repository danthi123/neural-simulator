---
type: finding
status: partial
mechanism: cls-two-store-bounded-fast-store-plus-slow-cortex
lane: breadth / catastrophic-forgetting / memory
date: 2026-08-09
---

# CLS two-store: a bounded fast store gives a bounded WORKING SET, not bounded LIFETIME retention (honest NEGATIVE with teeth)

**Date:** 2026-08-09
**Status:** NEGATIVE (the decoupling hypothesis is falsified; the wall's biological surpass is identified, not deferred)
**Runner:** `research/runners/_teacher_loop_cls_two_store_derisk.py` (reuse-by-import; NO sim/ edit)
**Backend:** numpy (tiny launch-bound teacher-loop net) · 6 seeds 42-47 · de-clamped `bdsp_wmax=1e9`
**Raws:** per-seed `research/findings/raw/teacher_loop_cls_two_store_s42.json` .. `_s47.json` +
`research/findings/raw/teacher_loop_cls_two_store_AGG.json`; 6-seed means (computed across seeds, cited below)
`research/findings/raw/teacher_loop_cls_two_store_6seed_means.json`; F-sweep
`research/findings/raw/teacher_loop_cls_two_store_F10_AGG.json`

## The question

The breadth crux was resolved by SIZING THE RESERVOIR to the fact count (flat capacity: 0.967 <!--derived--> at N=20,
holds at N=50; prior from the neurogenesis-capacity finding, not measured here). But a single flat store that grows with N means the per-step consolidation REPLAY SET grows with
everything ever learned (O(N) per sleep) — a wall for a year of real-world learning. The biology's answer is
Complementary Learning Systems (McClelland/McNaughton/O'Reilly 1995): a BOUNDED fast hippocampal store (recent
working set) + a slow distributed cortical store, with systems consolidation moving memories fast->slow via
interleaved replay, after which the hippocampal index DECAYS. **Hypothesis:** a FIXED-size fast store F + a slow
cortical readout, consolidated by neural self-replay with older facts EVICTED after consolidation, RETAINS N facts
even when N >> F — retention (and thus per-step replay cost) DECOUPLED from lifetime.

## The build (all brain-based; host code only for world/body)

- **Fast store** = `BoundedHippocampus` (subclass of the sleep-replay `Hippocampus`): fixed capacity F; after each
  sleep the index DECAYS (engrams older than the F most-recent are EVICTED). `capacity<=0` => the FLAT baseline
  (unbounded, replays all N).
- **Slow store** = a `NeurogenesisNet` reservoir born FULLY at start (a fixed large de-clamped leaky e-prop
  readout = matched_fixed cortex), NEVER grown per-fact (asserted constant across the curriculum — not the flat
  growing reservoir in disguise).
- **Consolidation = neural self-replay:** `_self_replay_consolidate` takes ONLY the fast store — the hippocampus
  GENERATES replay patterns from its stored engrams (brain-owned RNG; teacher + world absent, no `env` param) and
  the SAME transport-free e-prop rule moves the slow readout. No host weight/input copy fast->slow.
- **Arms** (same net build / seed / wake budget / slow reservoir; only the fast-store schedule differs):
  `two_store` (bounded F + consolidate + evict) · `flat` (unbounded + consolidate, measured in-run) ·
  `no_consol` (bounded F, consolidation DISABLED — the load-bearing control).

## Result: the hypothesis is FALSIFIED — retention TRACKS F, it does not decouple

Retention (frac of all N facts recalled from the neural slow readout), F=5, slow_hidden=100, 6-seed mean
(`teacher_loop_cls_two_store_6seed_means.json`):

| N | two_store (F=5) | flat (O(N) replay) | no_consol (F, no consol) |
|---|---|---|---|
| 10 | **0.800** | 0.933 | 0.600 |
| 20 | **0.517** | 0.950 | 0.450 |

- **two_store DEGRADES as N grows past F** (0.800 @ N=10 -> 0.517 @ N=20) while **flat HOLDS** (0.933 -> 0.950) on
  the SAME reservoir. Gap at N=20: two_store is **0.433 <!--derived--> below flat** (0.950 minus 0.517). two_store retains ~the working set F (the
  most-recent facts) plus a few durably-consolidated older facts (evicted-fact recall 0.378); the rest of the
  evicted set drifts.
- **Retention COUPLES to F (the decoupling test, inverted):** at N=20, two_store rises **0.517 (F=5) -> 0.717
  (F=10)** (`teacher_loop_cls_two_store_F10_AGG.json`, seeds 42-44). To retain more of N you must ENLARGE F — i.e.
  per-step replay cost stays coupled to how much you want to retain. This is the direct opposite of the decoupling
  claim.
- **Boundedness held (the mechanism worked as specified):** two_store's active fast-store size is F=5 at BOTH
  N=10 and N=20 (does NOT grow with N); flat's is N. Max replay set: 6 (two_store) vs 20 (flat). The slow
  reservoir is constant across the curriculum (all 6 seeds). So the bounded store is genuinely bounded — it just
  does not retain lifetime N.

## WHY consolidation fails to transfer the evicted set (the teeth)

The decisive contrast is `flat` vs `two_store`: SAME reservoir, SAME per-fact wake+replay strength — the ONLY
difference is that flat replays ALL facts every sleep while two_store replays only the F in the buffer. flat
retains 0.950; two_store loses the evicted set. Therefore **the failure is not capacity and not the e-prop
mechanism — it is the loss of ONGOING rehearsal for evicted facts.** Once a fact leaves the bounded buffer it has
no replay source, and the shared slow readout keeps being moved by every subsequent fact's wake+replay, so the
evicted fact's cortical trace drifts.

**It is a wall FUNDAMENTAL to THIS mechanism (a recency buffer with eviction), not an operating point — and NOT
a capability wall** (the capability, bounded-cost lifetime retention, is not abandoned; the surpass is named below).
`EXTERNAL-SEARCH-RAN:` the CLS continual-learning literature was read — the recency-buffer-with-eviction is a known
weak baseline; generative replay (van de Ven et al., 2020; see Sources) is the mechanism that
achieves bounded-cost all-class coverage. Tripling per-round consolidation (replay-epochs 12->40,
per-fact 8->16, so each fact is consolidated far more strongly while in the buffer) did NOT rescue the evicted set:
two_store @ N=20 stayed 0.35 and evicted-fact recall stayed 0.20 (= no_consol). Strengthening
consolidation-while-in-buffer cannot compensate for zero-rehearsal-after-eviction. Consolidation IS load-bearing
at N=10 (two_store - no_consol = **+0.20**, 0.800 vs 0.600) but the margin collapses to **+0.067** at N=20 (0.517
vs 0.450) — it durably transfers a FEW evicted facts, but not the working-set capacity the hypothesis needs.

## The biological surpass (identified, NOT deferred)

The missing companion process: in real CLS the bounded object is not a raw recent buffer that is discarded — it is
a GENERATIVE mechanism whose PARAMETERS are bounded but which keeps regenerating ALL consolidated classes. Van de
Ven/Siegelmann/Tolias (2020, brain-inspired generative replay) achieve BOUNDED memory (~92% on class-incremental)
precisely because a fixed-size generator replays every previously-learned class every step — coverage decouples
from N while cost stays bounded by the generator, NOT by a recency buffer that evicts. Our raw-engram-buffer +
eviction is the WRONG bounded mechanism: it bounds COVERAGE to F, so it can only ever retain F. **The next de-risk
is cortical/generative self-replay** — the slow store (or a fixed-size generator trained alongside it) regenerates
its OWN consolidated memories for interleaving, so nothing needs to persist in the hippocampus and per-step replay
cost is fixed while coverage stays all-N. This is the mechanism the CLS literature actually uses for the bounded
working set; the recency-buffer version tested here is the honest falsification that motivates it.

## Anti-cheats (all asserted True in every raw, 6/6 seeds)

- Fast store TRULY bounded: two_store |fast| = F=5 at N=10 AND N=20 (no growth with N); flat |fast| = N.
- Consolidation NEURAL: fast-store-GENERATED replay -> e-prop (`_self_replay_consolidate` has no `env`; calls
  `hippo.generate_replay` then `_teach_fact`->`net.train_batch`). No host weight/input copy.
- Slow store not the flat growing reservoir in disguise: `slow_reservoir_active_constant == True`.
- `cfg.seed` byte-identical substrate across two builds at one seed (NOT actual_seed_used); de-clamped
  `bdsp_wmax=1e9`; `git diff main -- sim/` empty; backend recorded (numpy).

## Sources (external)

<!--derived: the numbers below are citation identifiers (DOI, volume, pages, years), not measurements-->

- McClelland, McNaughton, O'Reilly (1995). "Why there are complementary learning systems in the hippocampus and
  neocortex." Psychological Review 102(3):419-457. (CLS: fast hippocampal + slow cortical, replay consolidation.)
- van de Ven, Siegelmann, Tolias (2020). "Brain-inspired replay for continual learning with artificial neural
  networks." Nature Communications 11:4069. doi:10.1038 <!--derived-->/s41467-020-17866-2. (Bounded-cost generative replay of ALL
  learned classes — the surpass this negative motivates.)

## Reproduce

```
SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
  python -m research.runners._teacher_loop_cls_two_store_derisk --seeds 42 43 44 45 46 47 \
    --n-max 20 --milestones 10 20 --capacity 5 --slow-hidden 100 \
    --epochs 20 --replay-epochs 12 --replay-per-fact 8 --n-draws 16 --settle-steps 20 --test-n 40 \
    --out research/findings/raw/teacher_loop_cls_two_store.json
```
