---
type: preregistration
status: preregistered
date: 2026-09-23
lane: memory (knowledge scale; D6 fact store capacity)
mechanism: ca3-superposed-fact-attractor
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_ca3_superposed_fact_attractor/dev_seed7/uniform_nrel2000/sparse_dg_nrel2000_s7.json
  - research/findings/raw/_ca3_superposed_fact_attractor/dev_seed7/uniform_nrel2000/sparse_dg_c2_nrel2000_s7.json
  - research/findings/raw/_ca3_superposed_fact_attractor/dev_seed7/uniform_nrel2000/dense_nodg_nrel2000_s7.json
  - research/findings/raw/_ca3_superposed_fact_attractor/dev_seed7/hub_nrel128/*.json
note: the 54-line evaluation grid this prereg governs has NOT been run; its future aggregate summary (under
  the runner's --out grid directory, see "Staging" below) will be cited once it exists. The dev-seed-7
  artifacts above are cited for provenance of the disclosed numbers ONLY -- see dev_seed7/NON_EVIDENCE.md --
  they are design-viability data, never a result.
---

# PRE-REGISTRATION: capacity law of a superposed CA3 fact store (2026-09-23)

Committed on its own, BEFORE any evaluation-seed run. The code it governs is
`research/runners/ca3_superposed_fact_attractor.py` as committed in the parent of this commit; every constant
below is fixed there. Biology: `research/biology/ca3-superposed-fact-attractor.md`.

## The question

The owner asked on 2026-09-23: "Regarding fact learning, given you mentioned small scale, I'm curious if the
learning we're proving here scales to the levels needed to go head to head with even a tiny llm?"

The D6 store cannot answer this. It gives every fact its own block of synapses, so the storage shared between facts
is zero and recall cannot degrade with the number of facts. This runner writes every fact into SHARED synapses,
so capacity becomes a measurable law. The fast hippocampal store is expected to reach rat-hippocampus scale, not
LLM scale. The registered claims are about the law and its companions, not about parity with an LLM.

## What runs

A default-off research runner; nothing in `sim/` or `webapp/` imports it. Per arm and seed:

- One fixed network per seed: EC-in 3 x 1000 cells (k_ec = 20 per filler), DG 15000 cells (a_dg = 0.005), <!--derived-->
  CA3 10000 cells (a = 0.01 unless the arm says otherwise), EC-out patient layer 1000 cells.
- Fixed random topologies: EC->DG 200 per granule, mossy 46 per CA3 cell, recurrent c_rec, perforant c_pp,
  readout c_out.
- Facts (agent, relation, patient) are written in a random order, one presentation each, by the covariance rule
  with each cell's own running rate (Welford form).
- At each checkpoint P in {50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000} (plus 100000 for
  `sparse_dg_c2`), 200 stored facts are cued with agent + relation only.
- Read: T = 8 gamma cycles of recurrent drive plus the persisting perforant drive, each followed by k-WTA; then
  CA3 -> EC-out. The recent-100 facts and 100 never-stored keys are also cued.
- Scoring (instrument): correct iff the true patient's code overlap with the EC-out pattern strictly exceeds every
  other entity's. Chance is 1/2000.
- The lesions `rec_zero` (recurrent synapses zeroed) and `rec_shuffle` (each row's recurrent values re-attached to
  other presynaptic cells) change ONLY the recurrent edge; everything else is identical.

Arms (9):

| regime | arm | what differs |
|---|---|---|
| uniform (n_rel = 2000) | `sparse_dg` | DG sparse coding, c_rec 2000, c_pp 600, c_out 2000 |
| uniform | `sparse_dg_c2` | every fan-in doubled (4000 / 1200 / 4000), grid to 100000 |
| uniform | `sparse_dg_recx2` | c_rec ALONE doubled (4000); c_pp, c_out unchanged -- isolates the recurrent edge |
| uniform | `dense_nodg` | no DG; CA3 selected by a fixed EC->CA3 projection; a = 0.05 |
| hub (n_rel = 128) | `sparse_dg_hub` | as `sparse_dg` |
| hub | `dense_nodg_hub` | as `dense_nodg` |
| hub | `sparse_nodg_hub` | sparse code (a = 0.01) without the DG: the dissociation arm |
| hub | `sparse_dg_c2_hub` | fan-ins doubled |
| hub | `sparse_dg_bounded_hub` | integer synapses in [-1, 1], LTP +1, stochastic heterosynaptic LTD (palimpsest) |

`sparse_dg_recx2` was added 2026-09-24, in response to review: `sparse_dg_c2` doubles c_rec, c_pp AND c_out
together, so a capacity-law k fit off it cannot be attributed to the recurrent edge specifically -- the dev
`rec_zero` lesion shows the recurrent edge owns only ~25% of capacity linearly (P50 8119 -> 6047), so a k in the
Rolls recurrent-autoassociative range fit off `sparse_dg_c2` was "a coincidental match of a confounded
normalization". `sparse_dg_recx2` has **no dev-seed-7 run**: it is a real, untested prediction, not something
seen before registration (see `research/findings/raw/_ca3_superposed_fact_attractor/dev_seed7/NON_EVIDENCE.md`).

Commands (one pool line each, 54 lines):
`python -m research.runners.ca3_superposed_fact_attractor --arm <arm> --seed <seed> --out research/findings/raw/_ca3_superposed_fact_attractor/grid`,
then `--aggregate research/findings/raw/_ca3_superposed_fact_attractor/grid`.

## Seen before registration: dev seed 7

**NON-EVIDENCE (2026-09-24 review).** Dev seed 7 is not an evaluation seed. It ran on pool42 from the working
tree to check the design could answer its questions. Artifacts:
`research/findings/raw/_ca3_superposed_fact_attractor/dev_seed7/` (see that directory's `NON_EVIDENCE.md`). Those
JSONs were **hand-edited after the run** (backend/runner/provenance_note fields added manually) and have **no
`.prov.json` sidecars** -- `research/runners/__init__.py`'s provenance wrapper never ran on them. They inform the
four design choices disclosed below and nothing else: **they must never be cited as a result, a measurement, or
evidence for any gate**, and `aggregate()` cannot pick them up by construction (it only reads seeds in `SEEDS`).
Disclosed here because each choice below was made after seeing this non-evidence.

1. **Regime split.** With n_rel = 128 the capacity law failed: doubling every fan-in moved P50 only
   3240 -> 3660 (x1.13). The perforant path's initial CA3 overlap fell along the same curve in every arm, bounded
   or not, DG or not. Each cue half-matches ~P/128 stored facts, so capacity was limited by cue ambiguity, not by
   synapse count. With n_rel = 2000 the law held: 8119 -> 16650 (x2.05). The capacity law is therefore
   registered in the uniform regime, and the hub limit is registered as its own claim (G9).
2. **Companion regime.** DG sparse coding over the dense baseline gave x1.30 in the uniform regime and x3.6 in
   the hub regime. The DG has little to separate when cues are already dissimilar. G3 is registered in the hub
   regime.
3. **G5 threshold.** Zeroing the recurrent synapses lowered uniform P50 only from 8119 to ~6040 (x1.34). The
   readout still names the patient from the ~0.38-overlap pattern the perforant path gives. G5 is registered at
   x1.2, not the x1.5 first drafted. Most of this store's capacity is perforant -> CA3 -> readout
   heteroassociation, and the attractor adds a minority share. That is disclosed now, not discovered later.
4. **Operating constants** set before any evaluation seed: c_pp 300 -> 600, grid extended to 50000 (100000 for
   `sparse_dg_c2`), and the read made column-compressed (1.3 s -> ~30 ms per query).

Also seen: novel-cue settling does not separate stored from never-stored cues (0.88-0.95 for both at P >= 2000).
No abstention claim is registered; the convergence signal is not a familiarity monitor in this design.

## Gates, each with its realistic failing outcome written first

An effect gate passes on >= 5 of 6 seeds. The integrity gate G6 must pass on 6 of 6. An UNDEFINED seed (a
censored P50, a missing arm) counts as a FAIL, never a pass. P50 is the log-interpolated P at which recall first
falls below 0.5.

- **G1 learns (uniform `sparse_dg`): recall >= 0.9 at P = 50 and P = 200.**
  FAILS IF the perforant cue plus completion does not reinstate the patient even for a handful of facts, e.g.
  because the readout crosstalk or the k-WTA pick unrelated cells.
- **G2 cliff (all 8 unbounded arms): recall at the arm's largest P <= 0.2.**
  FAILS IF any arm still recalls above 0.2 at 50000 (100000 for c2). That would mean the curve cannot show
  capacity, as on the localist D6 store, and the instrument is VOID for that arm.
- **G3 companion (hub): P50(`sparse_dg_hub`) / P50(`dense_nodg_hub`) >= 1.5.**
  FAILS IF DG sparse coding does not raise capacity on correlated facts, e.g. because finite-size noise at
  aC = 20 cancels the sparseness gain, or because mossy-selected CA3 codes are themselves correlated.
- **G4 capacity law (uniform): P50(`sparse_dg_c2`) / P50(`sparse_dg`) in [1.4, 3.0].**
  FAILS IF capacity is not set by synapses per cell, e.g. the ratio stays near 1 because another bottleneck
  (readout, perforant cue ambiguity, k-WTA ties) caps both arms.
- **G5 recurrent completion load-bearing (uniform `sparse_dg`): P50(`rec_zero`) <= P50 / 1.2 AND
  P50(`rec_shuffle`) <= P50 / 1.2.**
  FAILS IF removing only the recurrent edge leaves capacity unchanged. The store would then be a perforant ->
  readout heteroassociator with a decorative attractor, a realistic outcome given dev x1.34.
  The reported side quantity is `attributable_linear_frac = (P50_intact - P50_rec_zero) / P50_intact`, the
  LINEAR fraction of P50 lost when only the recurrent edge is removed (dev: ~25%). A fraction over log(P50) is
  meaningless, since log(P50) has an arbitrary zero (one fact) -- 2026-09-24 review; this field is not a gate
  input.
- **G6 cost law, INTEGRITY SMOKE (uniform `sparse_dg`): single-query latency at the largest P / at P = 50 <= 2.0,
  and synapse bytes identical at every checkpoint.**
  FAILS IF retrieval scans stored facts or storage allocates per fact. This passes by construction of the
  fixed-topology store; it is labelled integrity and is not counted as evidence.
- **G7 palimpsest (hub): recent-100 recall at P = 50000 >= 0.5 for `sparse_dg_bounded_hub` AND <= 0.2 for
  `sparse_dg_hub`.**
  FAILS IF bounded ternary synapses lack the signal to recall even fresh facts, or LTD erases faster than 100
  writes, or the unbounded store does not collapse for recent facts.
- **G8 shared crosstalk (uniform `sparse_dg`): log-log slope of the clamped-pattern d' against P (P >= 200) in
  [-0.8, -0.2].**
  FAILS IF d' does not fall with P (slope near 0: no shared-synapse interference, i.e. storage is effectively
  localist) or falls far faster than the P^-1/2 crosstalk law.
- **G9 the hub limit is not synaptic (hub): P50(`sparse_dg_c2_hub`) / P50(`sparse_dg_hub`) < 1.4.**
  FAILS IF doubling synapses per cell does restore the capacity law with 128 relations, i.e. dev seed 7's x1.13
  was noise.
  **G9 is an absence-of-effect gate that shares its code path with G4** (both are a P50 ratio against a doubled
  arm): a broken `sparse_dg_c2_hub` arm would also pass G9. **Do not headline G9 in a finding unless G4 passes on
  the same seed set** (2026-09-24 review). The runner's `--aggregate` CLI prints a NOTE when this condition is
  violated.

GO for the superposed store's capacity law requires G1-G5, G7-G9 on >= 5/6 and G6 on 6/6. Any other outcome is
reported per gate and banked as measured.

## Predictions (theory; reported, not gated)

- Uniform `sparse_dg` P50 in 4000-15000. **The capacity-per-recurrent-synapse fit is the per-seed MARGINAL
  between `sparse_dg` and `sparse_dg_recx2` -- the CONTRAST that isolates the recurrent edge, not either arm's
  own ratio (AMENDMENT, see the AMENDMENT LOG below):**
  `k_rec = (P50_recx2 - P50_sparse_dg) * a * ln(1/a) / (c_rec_recx2 - c_rec_sparse_dg)`, computed per seed, then
  `k_fit` = the median of that per-seed marginal over seeds. `k_fit` is predicted in 0.1-0.3; Rolls 2013 gives
  0.2-0.3 asymptotically, and finite size lowers it. This is a genuinely untested prediction: dev seed 7 never
  ran `sparse_dg_recx2` (see the dev-seed-7 NON-EVIDENCE note above), so no dev number anchors it.
  **FAILS IF the median per-seed marginal is <= 0, i.e. doubling c_rec alone does not raise P50** (a realistic
  outcome: some other bottleneck -- the perforant cue, the readout, k-WTA ties -- could cap capacity regardless
  of recurrent fan-in). A non-positive marginal is reported AS MEASURED, never clipped to zero and never dropped
  from the seed-by-seed record; `aggregate()`'s `gpu_point_extrapolation` names this outcome explicitly and
  reports no extrapolation when it occurs. Each arm's own all-fan-in ratio (`P50 * a ln(1/a) / c_rec` for
  `sparse_dg` or `sparse_dg_recx2` alone) is also reported, per seed, but ONLY as a descriptive quantity: it is
  never fed into `k_fit` and never compared to the Rolls range, because that per-arm number is the SAME
  all-fan-in-confounded quantity dismissed for `sparse_dg_c2` below -- averaging it with a genuine marginal would
  reintroduce the confound by dilution rather than remove it.
- Extrapolation from the fitted k to one 3090-sized fast store (n_ca3 = 1e5, c_rec = 1e4, a = 0.005): ~5e4-1e5 <!--derived-->
  facts. The aggregate prints this as EXTRAPOLATION, not a measurement. The a-scaling from 0.01 (this runner's
  arms) to 0.005 (the GPU point) is ALSO untested: no arm varies a with the DG held fixed in the uniform regime <!--derived-->
  (`dense_nodg` changes both a and the DG together), so treat the extrapolation as order-of-magnitude only.
- **The arithmetic against a tiny LLM, stated explicitly and conservatively (2026-09-24 review; the prior draft
  understated the gap by roughly an order of magnitude):** Qwen2.5-0.5B has about 4.9e8 parameters; at
  Allen-Zhu & Li (2024)'s measured ~2 bits/parameter that is about 4.9e8 x 2 ~= 1e9 bits of storable knowledge.
  This runner's own scoring already charges log2(2000) ~= 11 bits per fact (chance is 1/2000 entities), so at
  that same per-fact cost 1e9 bits buys about 1e9 / 11 ~= 9e7, i.e. **of order 1e8 facts, not 1e6-1e7**. Against
  the extrapolated fast store's 5e4-1e5 facts, the honest gap is **about 3 orders of magnitude**, not 1-2. Say
  plainly: **the fast store alone is rat-hippocampus scale** (Buzsaki/Rolls: "tens of thousands" of items in rat
  CA3), and **reaching a tiny LLM's fact count needs the slow cortical store filled by consolidation** --
  interleaved replay into a superposed cortical associator (`cls-interleaved-consolidation`), which this runner
  does not build.
- `sparse_nodg_hub` vs `sparse_dg_hub` separates sparseness from DG pattern separation (dev: 1600 vs 3240).
- Per-query latency ~20-100 ms on one CPU core, flat in P. Write cost ~5-30 ms per fact, flat in P, so teaching
  P facts is O(P).

## Declared host shortcuts and abstractions

- (h1) Gamma-cycle binary discretization with k-WTA (host `argpartition`) as the idealization of E%-max feedback
  inhibition. It is not cross-checked against the full Izhikevich bridge; that is the named next rung.
- (h2) The EC-out patient layer is driven at encoding by the same cortical patient code (superficial -> deep EC
  relay abstracted as identity).
- (h3) Naming by overlap against every entity code is the instrument, not a brain decision.
- (h4) The facts, their order and the partial cue are the environment.
- (h5) Fixed random topologies are developmental wiring drawn from the seed, not self-organized.
- (h6) Subtractive row-centering (each cell's own mean weight) is applied identically in every arm.

## What this cannot show

It cannot show LLM parity. It says nothing about the slow cortical store. It cannot show that the k-WTA
abstraction matches spiking dynamics at this scale. It cannot show abstention: no familiarity signal is
registered. A GO here means the superposed store has a measurable capacity law with working companions; it does
not wire anything into the chat path.

## Staging

54 pool lines (9 arms x 6 seeds) run against an isolated revision on pool41 and pool42. Each line declares
`mem_gb=` as a NUMBER: 3 for `*_c2*` arms (measured peak RSS in the seed-42 smoke, rounded up), 2 for every other
arm that HAS been smoke-measured. **`sparse_dg_recx2` has never run (see the arm table above: it has no
dev-seed-7 number either), so its `mem_gb` is NOT a measurement** -- AMENDMENT (see the AMENDMENT LOG below):
it declares `mem_gb=3`, matching the `*_c2*` arms' bracket rather than the plain `sparse_dg` bracket, because its
doubled recurrent projection adds a second full `H`/`W` values array, a second `idx` topology array and a second
per-cell spike-count pair over `sparse_dg`'s allocation -- closer in size to a `_c2` arm's tripled fan-in state
than to `sparse_dg`'s single one. `mem_gb=3` for this arm is an ESTIMATE from the allocation accounting above,
not a measured peak RSS, and is declared as such wherever it is cited.

## AMENDMENT LOG

- **AMENDMENT (2026-09-24, filed after a second adversarial review of the `sparse_dg_recx2` fix-required round
  (this branch's `fix:distributed-store` result, `safe_to_merge: false`), BEFORE any grid job runs.** What had
  been seen when this amendment was written: the same dev-seed-7 artifacts as the original registration (none of
  them cover `sparse_dg_recx2`, which has no dev-seed-7 run by design -- see the arm table above); no evaluation
  seed of any arm has run; `research/queue/pool.queue`, `.claims` and `runs.jsonl` have zero `ca3_superposed`
  entries. Three corrections, none touching the arms, thresholds, fact/probe construction or seeds.

  **(a) The capacity-per-recurrent-synapse fit is corrected from a per-arm median to a per-seed marginal.** The
  first review round added `sparse_dg_recx2` (c_rec alone doubled) specifically so a k fit could be attributed to
  the recurrent edge, since `sparse_dg_c2` (all three fan-ins doubled) cannot support that attribution. The code
  that round shipped, and this prereg's original "Predictions" text, still fit k PER ARM as
  `k = P50 * a * ln(1/a) / c_rec` and took the MEDIAN over `(sparse_dg, sparse_dg_recx2)`. That does not attribute
  anything to the recurrent edge: the `sparse_dg` half of the median is the identical all-fan-in-confounded
  quantity the first review rejected off `sparse_dg_c2` ("P50 driven by all fan-ins, divided by c_rec"), and
  `sparse_dg_recx2` was never used as a MARGINAL (a difference) -- it only diluted the confounded number with a
  second, less-confounded one, while the code comment and this prereg both kept calling the result "attributed to
  the recurrent edge alone". Fixed: `k_fit` is now the median, over seeds, of the per-seed marginal
  `(P50_recx2 - P50_sparse_dg) * a * ln(1/a) / (c_rec_recx2 - c_rec_sparse_dg)`; the per-arm all-fan-in numbers
  are still reported per seed but labelled descriptive-only and excluded from the fit and from the Rolls
  comparison. See the corrected "Predictions" bullet above (edited in place by this amendment) and
  `research/runners/ca3_superposed_fact_attractor.py`'s `aggregate()`.
  **A worked example (the reviewer's own numbers): P50 8119 -> 10500 when c_rec doubles 2000 -> 4000. The old
  per-arm-median fit gave k_fit = median(0.187, 0.121) = 0.154 -- inside the registered 0.1-0.3 band. The
  corrected marginal gives k_fit = (10500-8119) x 0.01 x ln(100) / 2000 ~= 0.055 -- outside it.** This is exactly
  the kind of disagreement the fix must be able to surface, not paper over.
  **(b) A non-positive marginal is reported, never clipped or dropped.** A seed where doubling c_rec does not
  raise P50 (a live possibility if some other bottleneck already caps capacity) now reports its marginal as a
  negative or zero number in `k_marginal_per_seed`, sets `k_fit` from it like any other seed, and the
  `gpu_point_extrapolation` field states plainly that no extrapolation is defined rather than emitting a negative
  or nonsensical predicted fact count. See "FAILS IF" under the corrected "Predictions" bullet above.
  **(c) Two minor provenance/labelling fixes.** `sparse_dg_recx2`'s `mem_gb` in "Staging" above no longer claims a
  measurement it never had; it is now declared `mem_gb=3` as an explicit ESTIMATE from allocation accounting
  (matching the `*_c2*` bracket), not the `sparse_dg` bracket it was previously (incorrectly) grouped with. Pool
  lines re-staged after this amendment cite THIS amendment's own commit SHA in their `--checked` provenance text,
  not the branch's merge SHA (the prior round's lines cited the merge commit instead of the prereg's own commit).

  A unit test (`tests/test_ca3_superposed_fact_attractor.py::test_capacity_law_fit_is_the_marginal_not_the_per_arm_ratio`)
  pins the corrected fit on the reviewer's own worked numbers and is a MUTATION GUARD: it fails if `aggregate()`
  reverts to the per-arm-median fit (verified by running it against the pre-amendment code, where it fails with
  `0.1539... != 0.0548...`, before restoring the fix). A second test
  (`test_capacity_law_marginal_not_positive_is_reported_not_clipped`) pins (b).

  None of (a)-(c) changes G1-G9, the arms, the fact/probe construction, or the seeds. The grid (54 pool lines)
  had not been dispatched when this amendment was filed.
