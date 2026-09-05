---
type: finding
status: go
claim_check: measured-result
date: 2026-09-04
mechanism: coincidence-binding (SlotBinderComposer -- Path B of the VSA-composer-retirement roadmap, rung L2)
lane: scaffold-retirement (VSA composer -> learned) + consumer-hardware-reference
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: fanout=32 (the recommended value) carries the full 6-seed standard. fanout=8/16 carry a 1-seed
  (42) spot-check -- they are STRICTLY more conservative than fanout=32 (fewer wired candidates per slot, an
  easier-or-equal discrimination problem) and already unambiguously pass FIT; fanout=64 carries a 1-seed (42)
  spot-check -- it is a VRAM-gate BOUNDARY case already excluded from the production recommendation by FIT
  alone (see §2), so full 6-seed composition rigor there has low marginal decision value. See §6.
artifacts:
  - research/findings/raw/_slotbinder_l2_sparse_derisk/formula_fit.json
  - research/findings/raw/_slotbinder_l2_sparse_derisk/compose_f8_s42.json
  - research/findings/raw/_slotbinder_l2_sparse_derisk/compose_f16_s42.json
  - research/findings/raw/_slotbinder_l2_sparse_derisk/compose_f32_s42.json
  - research/findings/raw/_slotbinder_l2_sparse_derisk/compose_f32_s43.json
  - research/findings/raw/_slotbinder_l2_sparse_derisk/compose_f32_s44.json
  - research/findings/raw/_slotbinder_l2_sparse_derisk/compose_f32_s100.json
  - research/findings/raw/_slotbinder_l2_sparse_derisk/compose_f32_s101.json
  - research/findings/raw/_slotbinder_l2_sparse_derisk/compose_f32_s102.json
  - research/findings/raw/_slotbinder_l2_sparse_derisk/compose_f64_s42.json
  - research/runners/_slotbinder_l2_sparse_derisk.py
  - research/runners/_keystone2_spiking_slot_binder_derisk.py
  - research/runners/slotbinder_composer.py
  - tools/gates/consumer_hardware_reference.py
  - bridges/developed/scale787/day_33/{brain.json,facts.json}
  - research/findings/2026-09-04-slotbinder-live-scale-derisk-NOGO-dense-pathway-blowup.md
  - research/findings/2026-09-04-vsa-composer-learned-retirement-ROADMAP.md
  - research/biology/dg-ca3-sparse-index.md
---

# SlotBinderComposer L2 sparse fan-out de-risk: GO — a fixed small per-slot fan-out (fanout=32) fits a single consumer RTX 3090 AND composes correctly, 6-seed, at the real 404-fact production scale

**Board/lane: rung L2 of `research/findings/2026-09-04-vsa-composer-learned-retirement-ROADMAP.md`**, spawned
directly by the L1 NO-GO
(`research/findings/2026-09-04-slotbinder-live-scale-derisk-NOGO-dense-pathway-blowup.md`): the DENSE
all-to-all slot->filler wiring needed ~316GB to build and ~36-463GB GPU-resident at the real 404-fact/788-vocab
production scale, ~1000x heavier than the incumbent FHRR `RFPhasorComposer` (334MB, correct). L1's own
recommendation #1 named, but did not build, the fix: replace the dense `K*KF` pathway with a fixed small
fan-out per slot. **This finding builds it and re-runs L1's own two questions (fit + composition) at the
identical live scale: it works.**

## TL;DR

**GO.** A per-slot sparse fan-out of **32** candidate filler pools (instead of wiring every one of the 1195)
brings `SlotBinderComposer` at the REAL production topology (404 facts, 788-word vocab, `K=2020` slot pools,
`KF=1195` filler pools, read from the live deployed bundle `bridges/developed/scale787/day_33`) to
**28,603,200 synapses (34x fewer than dense's 968.3M)**, **14.73 GiB** by the project's own
`consumer_hardware_reference` gate — comfortably under the single-consumer-RTX-3090 24 GiB reference — and
**composes correctly on ALL 6 seeds (42/43/44/100/101/102)**: 12/12 real-fact store->recall round-trips
correct, 6/6 moat abstentions on never-stored cues, 6/6 mismatched-role-cue rejections, every check against
the true `K=2020/KF=1195` substrate (`research/findings/raw/_slotbinder_l2_sparse_derisk/compose_f32_s*.json`).
Two smaller fan-outs (8, 16) were spot-checked (1 seed each) and show the identical clean pattern; a larger one
(64) sits in a genuine VRAM-gate BOUNDARY (fits by the exact-measured-bytes formula, blocks the project's own
conservative 8x-margin gate) and also composes correctly where tested (1 seed). **Recommend `fanout=32` as the
production value.** This banks the L1 NO-GO's dense-wiring METHOD for good and clears the roadmap's stated L2
gate ("recall matrix unregressed 6-seed at scale; RSS O(K) not O(K*KF)") — **FHRR retirement (Path B) is now
viable pending the L3 wire-in**, which is the next rung, not executed here (see §5, §6).

## 1. What was built

Two files changed, both additive (new optional kwargs, default value unchanged — the pre-existing dense path
is byte-identical when `fanout` is not passed; all 8 pre-existing `tests/test_slotbinder_composer.py` tests
still pass unmodified, confirmed by direct run):

1. **`research/runners/_keystone2_spiking_slot_binder_derisk.py`** — `build_binder_bridge()` gained `fanout`
   and `required_fillers` kwargs, plus a new `slot_filler_nnz_formula(K, KF, fanout)` helper shared between the
   wiring code and every FIT calculation in this finding (so the formula used to PREDICT memory and the formula
   the code actually WIRES cannot drift apart — the discipline L1 used for the dense case, now extended to the
   sparse one). When `fanout` is `None` or `>= KF` the code path is IDENTICAL to before (dense). When
   `fanout < KF`, each slot pool `w_k` gets a candidate filler set of size `fanout`: `required_fillers.get(k, ())`
   (fillers this specific slot is GUARANTEED to need) plus a seeded-random pad up to `fanout` from the
   remaining filler pools.
2. **`research/runners/slotbinder_composer.py`** — `SlotBinderComposer` gained `fanout` and `prewire_facts`
   kwargs. `prewire_facts` (an ordered list of the flat-SVO facts this composer will be `store()`-d with) lets
   `_ensure()` precompute the one true required filler for every (fact, role) slot before the substrate is
   built — a wiring-time PRE-REGISTRATION from an already-known corpus, not a per-query lookahead. `bind`
   (`store_pair`), `write` (the per-slot Hebbian gate), `recall` (`read_slot`/`_match`) and the moat (`_match`
   returning `None`) are byte-for-byte UNCHANGED — only which `RegionPathway` objects get created differs.

**Two wiring modes, deliberately, because they answer different questions:**
- **Oracle-inclusive** (`prewire_facts` given — the mode used for every GO result in this finding): the
  batch-consolidation scenario this task is about — migrating an ALREADY-KNOWN, already-collected static
  knowledge bundle (`bridges/developed/scale787/day_33`) from FHRR to SlotBinder. The wiring-time
  pre-registration is analogous to a replay/consolidation pass laying down structural connectivity for known
  material before Hebbian learning strengthens it (host code decides WHICH candidates are wired — this is
  HOST-SELECTED pre-wiring, not `self-organized` per `docs/TERMS.md`; the READOUT mechanism itself never
  consults the answer — only WHICH synapse exists is pre-registered, WHICH one wins is still decided by
  spike-driven Hebbian potentiation + firing-rate argmax).
- **Blind** (no `prewire_facts`): the online/incremental scenario — no foreknowledge of which filler a slot
  will need, a purely random fixed candidate set. Included (at small scale — see §3) to honestly quantify the
  coverage cost when the corpus is NOT known in advance, per the roadmap's own L2 gate note ("a scale lever,
  may bound max_facts").

**Why oracle-inclusion is not the DG-CA3 index's forbidden cheat.** `research/biology/dg-ca3-sparse-index.md`'s
own anti-cheat rule requires a sparse-routing key to be computed from the CUE, never the ground-truth answer —
because that mechanism must generalize to arbitrary NEW cues at QUERY time. `SlotBinderComposer`'s architecture
differs in a way that matters here: each slot is permanently assigned to exactly ONE (fact, role) for its
entire life (never asked to route a novel cue among many candidates), so a wiring-time pre-registration from
the known corpus does not let the retrieval mechanism "cheat" at the thing being measured — it tests "given
the correct synapse was pre-wired, does a k-candidate argmax readout still work as reliably as a KF-candidate
one", exactly the roadmap's own stated question ("does the correct filler still win the WTA competition when
only ~k of KF candidates are wired in?"). It does NOT establish that an arbitrary, not-known-in-advance fact
could be correctly stored under sparsification — that needs either corpus foreknowledge (true for this
batch-migration use case) or genuine activity-dependent synaptogenesis (not built here; §3's small-scale blind
measurement quantifies its absence honestly).

## 2. FIT — does the REAL K=2020/KF=1195 topology fit a single consumer RTX 3090?

<!--derived: ratios and GB/GiB conversions below are computed from the measured nnz; the nnz and RSS/build-time
figures themselves are direct measurements, not derived-->

All four fanout rows are REAL measurements — a fresh build at the true production topology, matching
`bridges/developed/scale787/day_33`'s 404 facts / 788-word vocab exactly (`K=5*404=2020` slot pools,
`KF=788+3+404=1195` filler pools, `n_neurons=64,324`, identical to L1's own derivation) — not extrapolations,
an improvement on L1, which had to extrapolate its dense point because the literal 968M-edge build was too
large to attempt safely on this machine. The sparse builds are 18-105x smaller and were run directly.
`slot_filler_nnz_formula` matched the actual measured `cp_connections.nnz` EXACTLY at all 4 points (and at 3
additional small-scale cross-check points spanning `fanout in {None, 8, 32}`) —
`research/findings/raw/_slotbinder_l2_sparse_derisk/formula_fit.json`.

| fanout k | synapses (nnz) | vs dense (968.3M) | build (s) | this-process peak RSS (GB) | CH-gate estimate | CH-gate verdict | exact-40B/synapse estimate | exact verdict |
|---|---|---|---|---|---|---|---|---|
| 8  | 9,211,200  | <!--derived-->105x fewer | 11.3 | 3.36  | 5.49 GiB  | PASS  | 0.34 GiB | PASS |
| 16 | 15,675,200 | <!--derived-->62x fewer  | 17.6 | 5.63  | 8.57 GiB  | PASS  | 0.58 GiB | PASS |
| **32** | **28,603,200** | <!--derived-->**34x fewer**  | 30.7 | 10.01 | **14.73 GiB** | **PASS**  | 1.07 GiB | PASS |
| 64 | 54,459,200 | <!--derived-->18x fewer  | 62.9 | 17.39 | 27.06 GiB | **BLOCK** | 2.03 GiB | PASS |
| dense (L1, cited) | 968,307,200 | — | ~941 (extrapolated) | 316 (extrapolated) | 462.85 GiB | BLOCK | 36.07 GiB | BLOCK |

Two DIFFERENT VRAM estimators are shown because they disagree at `fanout=64`, and the disagreement itself is
informative, not noise: `vram_ch_gate` is `tools/gates/consumer_hardware_reference.py`'s own `estimate_vram_bytes`
formula — the project's actual CI-enforced gate, worst-case-all-features-on with an 8x co-residency safety
margin; `exact_40bytes_per_synapse` is L1's own EXHAUSTIVELY-INTROSPECTED, no-margin figure (every `cp_*`
array whose length scales with `nnz`, measured directly on a built bridge — fanout does not change
per-synapse byte cost, only synapse COUNT, so that ratio transfers unchanged). Per CLAUDE.md's "the gates are
authoritative" doctrine, `k=64`'s CH-gate BLOCK is the operative verdict for a production decision even though
the more precise, measured-not-modeled estimate says it would fit with room to spare — a real, not cosmetic,
boundary case: k=64 sits inside the gate's deliberately conservative safety margin, not inside an actual
measured-memory blocker. **`k=8/16/32` pass BOTH estimators comfortably** — even the largest of these, k=32,
uses only 61% of the conservative-gate budget (14.73 of 24 GiB), leaving headroom for the rest of a deployed
brain if SlotBinder were later co-resident with other organs (an L3/one-brain-integration question, untouched
here).

## 3. COMPOSITION — does it still compose, 6-seed, at the same real scale?

**Scope** (why fanout=32 carries the full 6 seeds and fanout=8/16/64 carry 1 each): observed per-combo wall
time on this shared, contended dev machine ranged 275s-1159s depending on fanout and concurrent system load
(load average 7-11 from unrelated jobs on this box); running the full 24-combo grid was not a fully efficient
use of compute once fanout=32's own 6-seed result was unambiguous and consistent, so the sweep was
deliberately bounded: full rigor on **fanout=32 (the recommended value)**, single-seed spot-checks on fanout=8
and 16 (STRICTLY more conservative than 32 — fewer wired candidates per slot, an easier-or-equal
discrimination problem — and already unambiguous on FIT) and on fanout=64 (already excluded from the
production recommendation by §2's FIT result regardless of composition outcome). Every combo uses a REAL,
seed-dependent sample of 2 facts drawn without replacement from the true 404-fact corpus — different real
facts every seed, not a fixed pair (12 distinct facts total across fanout=32's 6 seeds, zero overlap).

| fanout | seed | facts sampled (corpus idx) | query_patient hit | moat pass | mismatch pass | coverage |
|---|---|---|---|---|---|---|
| 8  | 42  | [35, 312]  | 2/2 | 1/1 | 1/1 | 10/10 |
| 16 | 42  | [35, 312]  | 2/2 | 1/1 | 1/1 | 10/10 |
| **32** | **42**  | [35, 312]  | **2/2** | **1/1** | **1/1** | **10/10** |
| **32** | **43**  | [203, 263] | **2/2** | **1/1** | **1/1** | **10/10** |
| **32** | **44**  | [49, 268]  | **2/2** | **1/1** | **1/1** | **10/10** |
| **32** | **100** | [309, 337] | **2/2** | **1/1** | **1/1** | **10/10** |
| **32** | **101** | [125, 381] | **2/2** | **1/1** | **1/1** | **10/10** |
| **32** | **102** | [64, 178]  | **2/2** | **1/1** | **1/1** | **10/10** |
| 64 | 42  | [35, 312]  | 2/2 | 1/1 | 1/1 | 10/10 |

**fanout=32, full 6-seed (42/43/44/100/101/102): 12/12 query_patient hits (100%), 6/6 moat abstentions
(100%), 6/6 mismatch-rejections (100%), 60/60 coverage checks (100%, guaranteed by oracle construction) — a
genuine 6-seed GO on the roadmap's own stated criterion ("recall matrix unregressed 6-seed at scale").** Every
seed drew a different real `(agent, action, patient, polarity)` fact from the corpus and every one was stored
and correctly recalled through only 32 of 1195 candidate filler pools per slot — a 97.3% reduction in
per-slot candidate count with zero loss of correctness on every tested case.
(`research/findings/raw/_slotbinder_l2_sparse_derisk/compose_f32_s{42,43,44,100,101,102}.json`)

**Small-scale mechanism sanity check** (before the real-scale sweep, seconds not minutes — toy scale
`vocab=7, max_facts=6` -> `K=30, KF=16`): oracle-inclusive `fanout=8` reproduces the ORIGINAL dense
`SlotBinderComposer._selftest()` output byte-for-byte on every check (`query_patient`, `query_agent`,
`ask_yes_no`, the moat) — the mechanism is unperturbed by sparsification when the correct filler is guaranteed
present. **The SAME config WITHOUT `prewire_facts` (blind) FAILS immediately**:
`query_patient("dog","chase")` returns `"dog"` (wrong, expected `"cat"`); `query_agent("see","dog")` returns
`None` (wrong, expected `"bird"`) — an empirical, not hypothetical, confirmation of the coverage problem that
motivates the two-mode design. This small-scale blind failure is the evidence base for §6's honest-scope
caveat about online/incremental use; a real-production-scale blind measurement was scoped (`--mode blind` in
the runner) but not completed in this session (stopped alongside the rest of the sweep once fanout=32's 6-seed
GO was decisive — see the note in artifacts).

## 4. Comparison to FHRR (cited, not re-measured — identical scale, from the L1 finding)

`RFPhasorComposer` at the same 404-fact/788-vocab scale: 334 MB peak RSS, 0.9s/query mean, 3/3 correct queries,
20.6s to store all 404 facts. Even the recommended sparse SlotBinder configuration (k=32: 10.0 GB this-process
peak build RSS, 14.73 GiB by the conservative gate) is roughly two orders of magnitude heavier at BUILD time
alone than FHRR's total footprint, before counting the retrieval-latency gap L1's own S5 flagged: sparsifying
the wiring shrinks `nnz` per step (roughly linearly, per L1's measured slope) but SlotBinderComposer's
`read_slot`'s `O(KF)` python scan over ALL 1195 filler-pool firing rates every retrieval step is UNCHANGED by
fanout (it does not know or care how many of those pools are actually wired to the driven slot) — this rung
fixes the MEMORY blocker, not the (separate, untouched) latency one. **FHRR remains lighter and faster** even
after this fix; the case for SlotBinder is capability completeness (multi-slot storage sidesteps FHRR's
superposition cap) and having a genuinely learned, on-substrate write, not raw efficiency.

## 5. Verdict

**GO on rung L2** (per `docs/TERMS.md`: this is the gate's own positive verdict, not a metric lifted from an
ambiguous run — every number above is read directly from its artifact). `fanout=32` clears BOTH of the
roadmap's L2 gate criteria at the real production scale: **FIT** (14.73 GiB of 24 GiB by the project's own
conservative estimator, 1.07 GiB by the exact-measured one) and **composition** (6/6 seeds, 12/12 recall,
6/6 moat, 6/6 mismatch-rejection, all against the true K=2020/KF=1195 substrate). This banks the L1 NO-GO's
dense-all-to-all-wiring METHOD for good, per THE LAW (a wall is a verdict on a method, not the capability) —
the capability (SlotBinderComposer's competitive-slot-plus-Hebbian-write architecture) is now demonstrated to
fit consumer hardware. **Term-check:** this is `de-risked`/`GO at runner level`, NOT `wired`/`integrated`
(`docs/TERMS.md`) — `SlotBinderComposer` is not reachable from `/api/brain-chat` and the deployed bundle still
runs `composer_kind="rf"`; the L3 wire-in (production default flip + a 320-scale GPU re-verify + demoting FHRR
to a verify-only oracle, per the roadmap's own L3 gate) is the next rung, not executed here.

**Recommendation: `fanout=32`.** It is the largest sweep value that passes BOTH VRAM estimators (comfortable
margin even under the conservative gate), giving the most headroom for future corpus growth or co-residency
with other organs, while carrying full 6-seed composition evidence. `fanout=64` is NOT recommended: it BLOCKS
the project's own standard gate at 27.06 GiB (>24 GiB), even though its composition (1 seed tested) and exact
byte count both look fine — per CLAUDE.md's "gates are authoritative" doctrine, the conservative gate's verdict
governs a production decision, not the more optimistic measured-bytes figure.

**This unblocks FHRR retirement (Path B).** The single named blocker after L1 — "the dense pathway does not
fit" — is resolved. The path to actually retiring FHRR now runs through L3 (wire-in) and L4 (learned codes),
per the roadmap's own ladder; FHRR remains the correct, efficient, production default until that wire-in lands
and is itself re-verified (§4's latency gap is a real, separate, unresolved cost that a wire-in decision must
weigh, not a reason to distrust this rung's own GO).

## 6. Honest scope / what this does not establish

- **Oracle mode requires the corpus to be known in advance.** Fine for THIS use case (migrating an existing
  static bundle) but does not generalize to an online/incrementally-growing knowledge base without either (a)
  a periodic re-wiring/consolidation pass over newly accumulated facts, or (b) genuine activity-dependent
  synaptogenesis (a new, unbuilt mechanism) — named as the next rung if online use is ever required, not
  resolved here. §3's small-scale blind test confirms the failure mode is real, not hypothetical; a
  real-production-scale blind measurement was scoped but not run (the sweep was stopped once fanout=32's
  6-seed GO made it decisive for THIS finding's verdict — see below).
- **The retrieval-latency problem (L1 S5) is untouched.** This rung fixes the MEMORY blocker; the `O(KF)`-
  per-retrieval-step python readout loop and the `_match()` linear scan over stored facts are separate,
  independent costs that would need their own rung (e.g. the DG-CA3 sparse-index pattern already used
  elsewhere in this codebase for `OneBrainComposer`/`ShardedPhasorStore` — a genuinely different mechanism from
  this one, since IT routes at query time from the cue and is meant to generalize to novel cues) before
  SlotBinder would be interactively usable at 404-fact scale.
- **The composition sweep samples 2 real facts per seed (12 total query_patient trials for fanout=32 across 6
  seeds), not all 404.** Exhaustively teaching all 404 facts remains the O(nnz)-per-step latency problem named
  above (L1 extrapolated ~91 CPU-hours for the dense case; the sparse case is proportionally cheaper per step
  but still far outside a bounded session's budget once the O(KF) readout floor is accounted for). The
  SUBSTRATE tested is the full real K=2020/KF=1195 topology (unlike the sample size, this is NOT reduced); the
  workload exercised on it is a bounded, seed-varying sample, clearly reported as such.
- **The sweep was deliberately stopped short of its originally-planned scope** (fanout=64 at 3 seeds + 2 real-
  scale blind checks were planned; 1 seed of fanout=64 and 0 real-scale blind checks were completed) once
  fanout=32's full 6-seed GO, combined with the already-consistent fanout=8/16/64 spot-checks, made the
  verdict decisive — grinding the remaining combos would have cost roughly another 1-2 hours of wall-clock for
  no change to the recommended fanout or the GO verdict (fanout=64 was already excluded from the recommendation
  by FIT alone; the online/blind scenario is already honestly characterized at small scale in §3).
- **`max_clauses` was left at its production default (404, giving KF=1195).** L1's own §8.2 noted a free ~34%
  cut to KF is available (`max_clauses=0` or `1` for a corpus that never uses embedded clauses, true of every
  real fact in this bundle) — not applied here to keep this rung a single-variable change (sparsification
  only) comparable apples-to-apples against L1's own KF=1195 baseline. Combining it with L2 is a further,
  free-in-principle memory win for a future rung.
- **A citation to flag, not chased down here:** `2026-07-21-gap2-spiking-learned-binder-6seed-GO-emergence-bar-close.md`
  is cited by the L1 NO-GO finding and the retirement roadmap alongside SlotBinder-specific GOs, but reads (frontmatter
  `mechanism: learned-binder`, title "a LEARNED local binder READ on the SPIKING RF substrate") as the
  delta-rule/additive-Hebbian binder read on the RF/FHRR phasor substrate — a related but DISTINCT mechanism
  from `SlotBinderComposer`'s discrete-neuron-pool architecture. Not relied upon here (this finding's own §3
  small-scale check is the definitive "vs dense" comparison used); flagged as a separate follow-on doc-accuracy
  task, out of this finding's scope.

## Sources

Code: `research/runners/_slotbinder_l2_sparse_derisk.py` (new), `_keystone2_spiking_slot_binder_derisk.py`,
`slotbinder_composer.py` (both edited, additive), `tools/gates/consumer_hardware_reference.py`.
Data: `bridges/developed/scale787/day_33/{brain.json,facts.json}` (live deployed bundle, gitignored, read
directly, identical to L1). Findings:
`2026-09-04-slotbinder-live-scale-derisk-NOGO-dense-pathway-blowup.md` (L1, this finding's direct parent);
`2026-09-04-vsa-composer-learned-retirement-ROADMAP.md` (the L2 gate definition: "recall matrix unregressed
6-seed at scale; RSS O(K) not O(K*KF)"); `2026-07-17-keystone-2-spiking-slot-binder-STEP1-prereq-GO.md`,
`2026-07-22-gap2-attribute-slot-GO-FHRR-retirement-step1.md`,
`2026-07-22-gap2-pointer-clause-GO-FHRR-fully-retirable.md` (the SlotBinderComposer-specific small-scale GOs
cited by L1/the roadmap as the composition-capability baseline; see §6's citation note). Biology:
`research/biology/dg-ca3-sparse-index.md` (cited for the DG-expansion sparse fixed-fan-out connectivity motif
this mechanism borrows structurally, and for the query-time-vs-wiring-time distinction that makes
oracle-inclusion a fair test here — see §1). Tests: `tests/test_slotbinder_composer.py` (8/8 pass, unmodified,
confirming no regression to the dense default path).
