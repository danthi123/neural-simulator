---
type: finding
status: live
claim_check: measured-negative-result
date: 2026-09-04
mechanism: coincidence-binding (SlotBinderComposer -- Path B of the VSA-composer-retirement roadmap)
lane: scaffold-retirement (VSA composer -> learned, rung L1) + consumer-hardware-reference
seed-waiver: this measures a DETERMINISTIC architecture property (neuron count, exact synapse-count formula, bytes/synapse) fixed given (K,KF), not a stochastic accuracy metric -- seed=42 used throughout for reproducibility, not statistical replication. Composition CORRECTNESS (bind->unbind recovers the filler) is not re-measured here; it is cited from its own prior 6-seed GO findings (below) at small scale.
artifacts:
  - research/findings/raw/_slotbinder_live_scale_derisk_2026-09-04.json
  - research/findings/raw/_fhrr_rf_composer_live_scale_404facts.json
  - bridges/developed/scale787/day_33/brain.json
  - bridges/developed/scale787/day_33/facts.json
  - research/runners/slotbinder_composer.py
  - research/runners/_keystone2_spiking_slot_binder_derisk.py
  - research/runners/rf_phasor_composer.py
  - sim/regions.py
  - tools/gates/consumer_hardware_reference.py
  - research/findings/2026-09-04-vsa-composer-learned-retirement-ROADMAP.md
  - research/findings/2026-08-25-gap2-deltarule-binder-production-integration-NOT-WIRED.md
---

# SlotBinderComposer L1 live-scale de-risk: NO-GO — the dense slot→filler pathway needs ~316GB to build and ~36-463GB resident at 404 facts / 788-word vocab, ~1000x FHRR's measured footprint at the identical real scale

**Board/task: `task_5c54ca7f`** (spawned by `2026-08-25-gap2-deltarule-binder-production-integration-NOT-WIRED.md` L132-134,
restated as rung **L1** of `research/findings/2026-09-04-vsa-composer-learned-retirement-ROADMAP.md`). That roadmap named
this the single unclosed gate on the most mature FHRR-retirement path (Path B, `SlotBinderComposer`) and asked for exactly
this measurement before either wiring it in (if it fits) or sparsifying it (if it does not). It does not fit, as built.

## TL;DR

**NO-GO on the CURRENT dense (`density=1.0`) slot→filler wiring, measured (not guessed) at the real production scale
(404 facts, 788-word vocab, read directly from the live deployed bundle `bridges/developed/scale787/day_33`).** The
composer allocates `K=2020` slot pools (5 roles × 404 facts) and `KF=1195` filler pools (788 vocab + 3 internal + 404
clause-pointers), then wires **every** slot pool to **every** filler pool at density 1.0 — `K·KF·400 ≈ 966M` edges just
for that one pathway, ~968.3M total. Building that wiring plan (a Python double-loop, per `sim/regions.py`'s
`_build_pathway`) would need an extrapolated **~316 GB of host RAM and ~15.7 minutes single-threaded**, verified by
regression over 4 real builds spanning 53,600 to 26,562,560 synapses (the analytical edge-count formula matched the
measured synapse count **exactly** at all 4 points). That alone would OOM this 46GB-RAM dev machine, and essentially any
single consumer machine, before a GPU array is ever allocated. The post-build **GPU-resident steady state** — measured by
exhaustively itemizing every per-synapse CuPy array the bridge actually allocates (40.0 bytes/synapse, exact) — would need
**~36 GB**, already 1.5× a single RTX 3090's 24GB with zero room for the rest of the deployed brain; by the project's own
`gates/consumer_hardware_reference.py` formula (worst-case-features-on, 8× co-residency margin) it is **~463 GB**, 19× over.
`RFPhasorComposer` (the deployed FHRR composer, `composer_kind="rf"`), measured **for real** (not extrapolated) at the
identical 404-fact/788-vocab scale, costs **334 MB** RAM and **≈0.9s/query** — roughly **1,000× lighter** and (by
extrapolation) **5-6 orders of magnitude faster per query** than SlotBinder would be even if its memory problem were
solved. **The composition capability (bind→unbind, 6-seed GO) stands only at the small scales it was tested at; it is
UNREACHABLE at live scale with the current wiring because the substrate cannot even be constructed to test it there.**
**Recommendation: do NOT wire `SlotBinderComposer` as the production default as built. Rung L2 (sparsify the slot→filler
pathway) is a precondition, not an optional hardening step — a ~40-50× reduction in total synapse count is needed before
the CURRENT builder is even viable on this hardware, let alone a 24GB card.**

## 1. Methodology — why this is a real measurement, not a re-statement of the roadmap's guess

The roadmap flagged the scale question but explicitly had not measured it ("a back-of-envelope synapse count, not
measured" — `2026-08-25-...NOT-WIRED.md` L138-140). This session did four things the roadmap asked for:

1. **Read the exact wiring code** (`research/runners/slotbinder_composer.py` + `_keystone2_spiking_slot_binder_derisk.py`
   `build_binder_bridge`, + `sim/regions.py` `RegionManager._build_pathway`) to derive the EXACT edge-count formula, not
   an estimate: `nnz = K·KF·400 (dense slot→filler) + K·1360 (self-recurrent + slot↔fs)`. `sim/regions.py` confirms
   connectivity is a genuinely SPARSE edge list (`cp_connections`, a CSR matrix built from `pre_indices`/`post_indices`
   lists), not an `O(n²)` dense matrix — so the cost scales with actual synapse COUNT, which is what makes this
   formula the right thing to extrapolate.
2. **Built the real bridge** (`build_binder_bridge`, the exact function `SlotBinderComposer` imports and uses) at four
   scales — `(K,KF) ∈ {(10,10),(40,40),(160,160),(256,256)}`, i.e. `nnz ∈ {53600, 694400, 10457600, 26562560}` — each in
   a **fresh subprocess** (so peak RSS reflects only that point), on **CPU/numpy**. GPU was busy at measurement time
   (`tools/gpu_queue.sh status`: a `SIM_BACKEND=cupy` fluency sweep running, 96% util, 19585/24576 MiB used, 4513 MiB
   free) — per this task's own instruction, measured on CPU and extrapolated analytically for VRAM. `float32`/`int32`/
   `bool` array byte sizes are backend-independent, so a bytes-per-synapse ratio measured on CPU/numpy transfers
   directly to a GPU/cupy run of the identical code path.
3. **Exhaustively itemized** every `cp_*` array the bridge actually allocates whose length equals `nnz` or `1.5·nnz`
   (an intentional scan of ALL matching attributes, not a hand-picked subset — a first pass using a hand-picked list of
   5 array names under-counted by 40%, corrected by scanning every `cp_*` attribute on the built bridge object).
4. **Built `RFPhasorComposer` for real** at the identical 404-fact/788-vocab scale (not extrapolated — it is cheap
   enough to run directly) for an apples-to-apples comparison, on the same CPU/numpy backend.

All four (K,KF) build points' measured synapse counts matched the analytical formula **exactly** — 53600, 694400,
10457600, 26562560 — which is why the extrapolation below is a regression fit through exact points, not a guess chained
onto a guess.

## 2. The production scale, from the live deployed bundle (not assumed)

Read directly from `bridges/developed/scale787/day_33/brain.json` + `facts.json` (gitignored deployment data, present
on this machine, not part of the repo checkout):

| quantity | value | source |
|---|---|---|
| `n_facts` | 404 | `brain.json["n_facts"]`, `len(facts.json["facts"])` |
| vocab size | 788 | `brain.json["vocab"]` (== `n_grounded_codes`) |
| `D` (FHRR dim) | 128 | `brain.json["D"]` |
| `composer_kind` (live) | `"rf"` | `brain.json["composer_kind"]` — confirms the 2026-08-25 finding: still FHRR in production |
| roles populated/fact | 4 (agent, action, patient, polarity) | direct read of all 404 facts; none use `attribute`/`attribute2` |

`SlotBinderComposer` always allocates **5** role slots per fact (`_ROLES=5`, including `attribute`, unused today) and
`max_clauses = max_facts` filler-pointer pools by default (`slotbinder_composer.py` L42, L59-66, L84):

- **`K = 5 × 404 = 2020`** slot pools (20 neurons each).
- **`KF = 788 (vocab) + 3 (AFFIRM/NEGATE/NOATTR) + 404 (clause pointers) = 1195`** filler pools (20 neurons each).

This refines the 2026-08-25 finding's own loose guess ("KF≈300-800") — that guess counted only the vocabulary and
missed the `+3` internal fillers and, more importantly, the `+404` clause-pointer filler pools the constructor appends
by default regardless of whether embedded clauses are ever used (production's facts are 100% flat SVO). **A production
wire-in that never needs embedded clauses could set `max_clauses=0` or `1`, cutting `KF` to ~791 (a ~34% reduction in
`KF`, hence in the dominant `K·KF` term) — noted as a minor secondary lever in §6, not a fix for the underlying
`O(K·KF)` scaling.**

## 3. The build-cost measurement (CPU/numpy, 4 real points, regression, exact-formula cross-check)

| K | KF | K·KF | n_neurons | measured nnz | formula nnz | build (s) | peak RSS (GB) |
|---|---|---|---|---|---|---|---|
| 10 | 10 | 100 | 424 | 53,600 | 53,600 | 1.28 | 0.253 |
| 40 | 40 | 1,600 | 1,624 | 694,400 | 694,400 | 1.94 | 0.480 |
| 160 | 160 | 25,600 | 6,424 | 10,457,600 | 10,457,600 | 11.93 | 3.901 |
| 256 | 256 | 65,536 | 10,264 | 26,562,560 | 26,562,560 | 26.99 | 8.890 |

(Raw: `research/findings/raw/_slotbinder_live_scale_derisk_2026-09-04.json` → `empirical_build_measurements_cpu_numpy`.)

Every measured `nnz` matches `K·KF·400 + K·1360` **exactly** — this is a deterministic property of the code, not a
noisy fit, which is what licenses extrapolating past the tested range with a linear regression rather than treating it
as speculation. Least-squares fits (both check within <1% at the two largest, most representative points):

- **peak RSS (GB) ≈ 0.30 + 3.26×10⁻⁷ × nnz**
- **build time (s) ≈ 1.37 + 9.70×10⁻⁷ × nnz**

## 4. Extrapolation to production scale (`nnz_full = 968,307,200`)

`nnz_full = K·KF·400 + K·1360 = 2020·1195·400 + 2020·1360 = 965,560,000 + 2,747,200 = 968,307,200` (~968.3M synapses,
`n_neurons = 64,324`). Applying the fits above:

- **Host build peak RSS ≈ 316 GB.** This is the HOST-side Python wiring-plan construction (`sim/regions.py`
  `_build_pathway` + `sim/bridge.py` `inject_explicit_wiring`'s per-synapse Python lists — pre/post indices,
  weights, and several PER-SYNAPSE STRING fields for gate/receptor names) — it happens **before** any GPU array is
  created, identically regardless of `SIM_BACKEND`. This machine has 46GB RAM total (25GB available at measurement
  time, with 36/46GB swap already in use from other running jobs) — nowhere close to 316GB. This step would be
  OOM-killed (or, if swap absorbed it, would thrash for a very long time) on essentially any single consumer machine.
- **Host build wall time ≈ 941s ≈ 15.7 minutes**, single-threaded CPU, Python-loop-bound (`_build_pathway`'s nested
  loop runs ~968M inner iterations across ~2.42M `RegionPathway` objects) — independent of GPU availability, since
  this phase is pure Python/numpy regardless of backend.
- **GPU-resident steady state (if the host build somehow completed): ~36.07 GB**, from an EXACT (not estimated)
  bytes-per-synapse introspection of the actual bridge object: `cp_connections` CSR (8.0 bytes/synapse: float32 data +
  int32 indices) + 8 other per-synapse CuPy arrays this specific bridge config allocates regardless of whether their
  feature is active (`cp_eligibility_trace`, `cp_synapse_pulse_progress`, `cp_synapse_pulse_timers` at 1.5× capacity,
  `cp_plasticity_gain`, `cp_plasticity_rate_gain`, `cp_synapse_action_tag`, `cp_nmda_recurrent_synapse_mask`,
  `cp_synapse_plastic_mask`) = **40.0 bytes/synapse exactly**, measured at K=160/KF=160 and structurally scale-invariant
  (these are fixed dtype×multiplier allocations, not something that changes shape with K/KF). `968,307,200 × 40 bytes
  ≈ 38.73 GB ≈ 36.07 GiB` — already **1.5×** a single RTX 3090's 24GB, before counting the rest of the deployed brain
  (other organs, CUDA context, activation buffers).
- **Cross-check against the project's OWN standard estimator**
  (`tools/gates/consumer_hardware_reference.py`, the CH gate that already governs the production-default path):
  applying its worst-case-all-features-on formula (200 bytes/neuron + 64 bytes/synapse, ×8 co-residency safety
  multiplier, +1 GiB fixed overhead) to `n_neurons=64,324, n_synapses=968,307,200` gives **~462.85 GiB** — even more
  decisively over budget. The two estimates bracket a wide range (36-463 GB) depending on how conservatively you
  model co-residency and margin, but **both exceed the 24GB reference**, so the qualitative verdict does not hinge on
  which estimator is "right."

**Either number is decisive: this does not fit a single consumer RTX 3090, and the host-side build does not even fit
this development machine's RAM.** The MEMORY blocker, not a GPU-capacity nuance, is what makes this NO-GO.

## 5. Per-step / per-query latency — a second, compounding concern (not the primary blocker, but real)

Timed `b._run_one_simulation_step()` directly (30 steps, CPU/numpy) at three scales: `K=40` (0.00239s/step), `K=160`
(0.04571s/step), `K=256` (0.11318s/step) — approximately linear in `nnz` (marginal slope ≈4.19×10⁻⁹ s/synapse/step
between the two largest points). Extrapolated to `nnz_full`: **≈4.06 s/simulation-step** on CPU. Each `store_pair`/
`read_slot` call runs `teach_steps=40`/`retr_steps=40` such steps (≈162s per call); storing one fact costs 5 such calls
(≈812s); teaching the full 404-fact KB would cost **≈91 CPU-hours**. The `_match()` recall itself is a **linear scan**
over stored facts (up to `len(self.facts)` iterations, each 1-2 `read_slot` calls) — a worst-case query could reach
**~131,000 CPU-seconds**. This is measured on CPU only (GPU busy, per instruction); a GPU/cupy sparse implementation
would be meaningfully faster per step, but (a) it does not fix the memory blocker in §4, which dominates, and (b) even
an optimistic 100-1000× GPU speedup leaves per-query latency far outside any interactive/conversational-turn budget at
this synapse count. **This is reported as a second, independent reason a purely denser-is-fine mitigation (e.g. "just
get a bigger GPU") would not actually make SlotBinder production-viable at this scale even if VRAM were unlimited** —
the linear-scan recall architecture and the O(nnz)-per-step cost both need to shrink together with the wiring density.

## 6. FHRR comparison at the SAME real scale — measured, not extrapolated

`RFPhasorComposer` (`composer_kind="rf"`, the composer actually deployed today) was built and exercised for real at
`D=128`, the full 788-word vocab, all 404 real facts (`research/findings/raw/_fhrr_rf_composer_live_scale_404facts.json`):
build 0.0043s; storing all 404 facts 20.6s total (51ms/fact); three sampled queries (first/middle/last fact) each
**≈0.87-1.01s** (mean 0.93s) and **3/3 correct** (`boy/go→park`, `king/rule→part`, `tower/fall→crash`, each matching the
stored patient exactly); **peak RSS 334 MB** for the whole process. The architectural reason this is roughly **1,000×**
lighter than SlotBinder's extrapolated footprint: FHRR/RF reuses **one fixed, small (~D-scale) neuron population**,
bridge-cached by neuron count and time-multiplexed across every bind/unbind/read operation for every fact and every
word (`_build_rf_bridge`, `connections_per_neuron=0` — no pre-wired synapses at all; `rf_set_complex_weights` sets
per-op weights directly on the same reused population). It does **not** allocate a dedicated neuron population or a
dense synaptic pathway per `(fact, role, filler)` combination — the actual fact data lives host-side in a plain Python
list (`self.kb`, ~2KB/fact). **This is exactly the CLAUDE.md "wall reframe" question answered concretely: SlotBinder's
dense pathway replaces FHRR's temporal reuse/multiplexing (one substrate, many timesteps) with a spatial
one-time-allocated-forever mapping (one substrate PER fact×role×filler combination) — trading a cheap, reusable
computation for an expensive, permanently-allocated one.**

## 7. Verdict for the roadmap's L1 gate

**NO-GO**, per the roadmap's own stated criterion ("GO ⇒ proceed to L3 wire-in; NO-GO ⇒ L2 first: sparsify the
slot→filler pathway"). `SlotBinderComposer`, as currently built, **cannot be constructed at all** at the live 404-fact/
788-vocab production scale on this (or, by the numbers above, essentially any single consumer) machine — the host-side
wiring-plan build alone needs an order of magnitude more RAM than exists. This is a verdict on the CURRENT DENSE-WIRING
METHOD, not on the SlotBinder architecture's capability or on the composition mechanism it implements: the coincidence
bind + competitive-slot write + content-addressable scan design that earned its 6-seed GO
(`2026-07-21-gap2-spiking-learned-binder-6seed-GO-emergence-bar-close.md`,
`2026-07-22-gap2-attribute-slot-GO-FHRR-retirement-step1.md`, `2026-07-22-gap2-pointer-clause-GO-FHRR-fully-retirable.md`)
is untouched by this finding — those GOs were established at small scales (the module's own default vocabulary is 10
words, its self-test uses `max_facts=6`), 2-3 orders of magnitude below the live scale, and **that capability claim
simply cannot yet be extended to live scale, because the substrate needed to test it there cannot be built.** Per
CLAUDE.md's law, this banks the DENSE-ALL-TO-ALL WIRING method, not the capability — the capability (a learned,
on-bridge-written FHRR replacement) stays live, routed through rung L2.

## 8. Recommendation

**Do not wire `SlotBinderComposer` as the production default in its current form.** Concretely:

1. **L2 is a precondition, not a hardening step.** The dense `density=1.0` slot→filler pathway must be replaced with a
   SPARSE one — each slot pool connecting to a small, fixed-size subset of candidate filler pools rather than all
   `KF` of them (the CLAUDE.md wall-reframe's own suggested direction: "is the dense pathway standing in for a sparse,
   developmentally-wired connectivity (DG/F.12 expansion)?" — here, concretely yes). A back-of-envelope target: to
   bring the HOST BUILD (the more binding constraint, §4) to a comfortable ~8GB budget on this class of machine needs
   `nnz ≲ 24M` — a **~40×** reduction from the current 968M, i.e. a per-slot fan-out on the order of `KF × (1/40)` ≈ 30
   candidate filler pools instead of all 1,195. Whether store/recall accuracy SURVIVES that sparsification (does the
   correct filler still win the WTA competition when only ~30 of 1,195 candidates are wired in?) is an open, cheap,
   CPU-only experiment — the next concrete rung, not yet run here.
2. **Setting `max_clauses` to production's actual need (0 or 1, not `max_facts`) is a free ~34% cut to `KF`** — worth
   doing regardless of L2, but nowhere near sufficient alone (§2).
3. **The latency finding (§5) should inform L2's design target, not just its memory target** — a sparsification that
   fixes VRAM but leaves the linear-scan recall + O(nnz)-per-step cost intact would still not be interactively usable.
4. **Until L2 lands, `composer_kind="rf"` (FHRR) remains the only composer that is actually usable at live scale** —
   consistent with, and now with hard numbers behind, the 2026-08-25 finding's decision not to wire any alternative
   binder mechanism into production yet.

## 9. Honest scope / what this does not establish

- **The full-scale build was NOT actually attempted.** Deliberately — the extrapolation already gives a decisive,
  many-fold-over-budget answer, and attempting the literal 968M-synapse build on this shared, memory-constrained (25GB
  available, 36GB swap already in use) development machine risked hanging or OOM-crashing a box other queued work
  depends on. The regression is grounded in 4 REAL points spanning a ~500× range with an EXACT (not approximate)
  analytical formula match at every one of them; residual uncertainty is a scale factor (plausibly up to ~2×), not a
  qualitative reversal of the verdict.
- **GPU-side timing was not measured directly** (GPU busy running a queued fluency sweep, per instruction). The
  bytes-per-synapse figure is exact and backend-independent (dtype sizes do not change between numpy/cupy); the
  per-step LATENCY figures in §5 are CPU-measured and explicitly flagged as an upper bound a GPU would improve on,
  not a claim about GPU performance.
- **L2's sparsification is proposed, not built or de-risked here.** Whether a sparse fan-out preserves recall accuracy
  is an open question this finding does not answer.
- **This does not re-test composition correctness.** The 6-seed GO for bind/write/recall stands as previously
  reported, at the small scales those findings used; this finding only establishes that live-scale testing of that
  same claim is currently impossible to run.
- **`max_clauses` reduction (§2, §8.2) is arithmetic, not measured** — it follows directly from the `KF` formula and
  was not independently re-run at that specific configuration.

## Sources

Code: `research/runners/slotbinder_composer.py`, `_keystone2_spiking_slot_binder_derisk.py`, `rf_phasor_composer.py`;
`sim/regions.py` (`RegionManager._build_pathway`, `_build_region_internal` — confirms sparse CSR, not dense `O(n²)`);
`sim/bridge.py` (`inject_explicit_wiring`, per-synapse array allocation); `tools/gates/consumer_hardware_reference.py`.
Data: `bridges/developed/scale787/day_33/{brain.json,facts.json}` (live deployed bundle, gitignored, read directly).
Findings: `research/findings/2026-09-04-vsa-composer-learned-retirement-ROADMAP.md` (this finding's parent, rung L1);
`2026-08-25-gap2-deltarule-binder-production-integration-NOT-WIRED.md` (spawned `task_5c54ca7f`, the loose scale guess
this finding refines); `2026-07-21-gap2-spiking-learned-binder-6seed-GO-emergence-bar-close.md`,
`2026-07-22-gap2-attribute-slot-GO-FHRR-retirement-step1.md`, `2026-07-22-gap2-pointer-clause-GO-FHRR-fully-retirable.md`
(the composition-capability GOs this finding does not disturb, cited for their small-scale scope). Memory:
`project_consumer_hardware_reference_principle.md` (the 3090/24GB standard this finding measures against).
NO-EXTERNAL-NEEDED: this is a direct measurement of this project's own code against its own stated hardware reference,
not a capability-walled/fundamental-limit claim about binding or VSA composition in general — see §7's explicit
banking of the METHOD (dense all-to-all wiring), not the capability.
