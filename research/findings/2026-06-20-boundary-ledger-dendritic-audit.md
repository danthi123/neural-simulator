# Boundary ledger + dendritic-debt audit — are we quietly accumulating boundaries that will require the deferred dendritic substrate? (2026-06-20)

**Type:** READ-ONLY strategic audit (no code, no experiments). Single deliverable = this doc. Stayed on `main`.
**The owner's question, verbatim:** across the whole project, how many distinct hard boundaries have we hit, and for
each — was it SURPASSED on the point-neuron substrate, is it a GENUINE dendritic boundary (blocking vs deferred-by-
choice), or is it OPEN/unresolved? The worry: are we quietly accumulating dendritic debt?
**Method:** built the ledger from the project's own boundary/NEGATIVE findings (grepped + read in full the load-bearing
ones), cross-checked CLAUDE.md's documented arc + `AUTONOMOUS_STATE.md`, and verified each "surpassed" claim against the
actual finding text. Where the truth is toy-scale / window-bounded / single-seed, it is flagged.

---

## TOP-LINE ANSWER (the honest call)

**NO — we are NOT accumulating dendritic debt. The opposite is the documented pattern: the "needs dendrites" verdict has
been raised at least six times and OVERTURNED on the point-neuron substrate every time it actually gated a shipped
capability.** The count, across ~16 distinct hard boundaries:

| classification | count | what it means |
|---|---|---|
| **SURPASSED on point neurons** | **9** | a point-neuron reframe/mechanism resolved it (FHRR pivot, PPMI local-norm, population coding, NEF cleanup, A-CSC TD, spiking-WTA decision, biased-competition WTA, diagonal-gain S5, dense-readout) |
| **GENUINE DENDRITIC — BLOCKING a shipped goal** | **0** | nothing currently shipped or on the conversational/nav critical path is gated by a dendrite |
| **GENUINE DENDRITIC — DEFERRED-BY-CHOICE** | **1** | a faithful *spiking-from-real-experience* learned cortex that preserves *weak/diffuse* structure — reserved for the artificial-life goal, NOT required for the conversational product (which ships flat + PPMI) |
| **OPEN / not-yet-closed (point-neuron engineering, NOT dendritic)** | **4** | nav reward/value/sustained-control loop (the big one), nav #5 place-code δ, merged-TD-cue-shift consolidation anti-cheat, two-attribute (F=3) bind on correlated learned codes |
| **DENDRITIC CANDIDATES TESTED → NEGATIVE (ruled out)** | **2** | dendritic multiplicative binding (memorizes, doesn't generalize) + apical-basal credit assignment (nothing to route) — both cheap-first NEGATIVE, both *saved* a months-scale build |

**The decisive fact:** the project *built* the dendritic machinery (D2 Phase 1, the per-source divisive gain, byte-clean
in `sim/`) AND ran the two cheap-first toy de-risks of its two named jobs — **and both came back NEGATIVE on the current
walls** (`2026-06-19-dendritic-binding-toy-derisk.md`, `2026-06-19-dendrite-credit-assignment-toy-stage1.md`). So the
dendrite is not a looming unpaid debt; it is a tool that was reached for, tested, and shown *not* to be the unlocker for
any current blocker. The one place a dendrite is still genuinely implicated (the faithful spiking learned cortex on the
*weak/diffuse real* corpus) is a DEFERRED *artificial-life* goal, explicitly NOT blocking the delivered conversational
product. **The debt is not real for any shipped or critical-path capability. It is a deliberate, scoped reservation for
the deepest biology-translatable goal — and even there the evidence is mixed (PPMI overtook most of its rationale).**

The single most-overstated "surpassed" claim found, flagged below, is **the on-bridge PPMI learned cortex** — it is
genuinely surpassed at the *rate/numpy* level and the off-diagonal dendrite is provably unnecessary, but the *faithful
spiking* realization on the *real* corpus is a documented NEGATIVE (+0.06 vs host +0.44); the *flat curated* cortex is
what actually ships. The conversational product does not depend on the spiking-from-real-experience version.

---

## THE BOUNDARY LEDGER

Legend — classification: **SURP** = surpassed on point neurons · **DEND-DEF** = genuine dendritic, deferred-by-choice ·
**OPEN** = unresolved point-neuron engineering · **DEND-RULED-OUT** = dendritic candidate tested NEGATIVE.

| # | boundary (one-line) | class | resolving mechanism — OR why it needs dendrites | evidence (finding) | honest caveat |
|---|---|---|---|---|---|
| 1 | **Conversational decorrelation/whitening blocker** — composer demands decorrelated codes; the point neuron can't whiten | **SURP** | REFRAMED twice: (i) FHRR phase code has no common mode to remove; (ii) the codes the cortex needs are *correlated* (PPMI generalizes *because* of it) — off-diagonal decorrelation was a **red herring** | `2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md`; `2026-06-17-offdiagonal-dendritic-derisk-NEGATIVE-ship-flat-cortex.md` | the off-diagonal dendrite de-risk found the learned gains **INERT** (lesion == mechanism, +0.519); local PPMI-centering already reaches the ZCA ceiling — airtight |
| 2 | **Opponency / rate-coded SNR wall** — `onoff(bon−boff)` common-mode removal of a small correlated difference can't survive rate coding (3 independent mechanisms NEGATIVE) | **SURP** | **Structural pivot to spiking-phasor FHRR** (info in PHASE, unit magnitude → no common mode, no small signed difference → opponency does not exist). Now the production composer default | `2026-06-05-B-opponency-rate-coded-SNR-wall-CONFIRMED.md`; CLAUDE.md "OPPONENCY ESCAPED" | the *rate-coded* op is a genuine boundary (biology removes the common mode analog pre-spike); the pivot sidesteps it rather than solving it — but that IS the brain's solution |
| 3 | **NEF/composer cleanup** — the argmax-nearest-concept readout was a host `np.argmax` | **SURP** | **Spiking NEF thresholded cleanup** (Stewart-Tang-Eliasmith): input-norm FS pool + placed per-concept firing threshold → off-target emits ZERO spikes; == numpy 27/27 at D=2048 | CLAUDE.md "COMPOSER CLEANUP SHORTCUT CLEARED"; `2026-06-05-composer-cleanup-NEF-GO.md` | validated at production D=2048 multi-seed — solid |
| 4 | **Generalizing learned cortex (rate level)** — recover real semantic category structure from real co-occurrence, no curated concepts | **SURP** | **PPMI feedforward LOCAL normalization** (log + per-hub + per-concept mean-subtraction + ReLU — all local ops) reaches host (+0.518) AND generalizes (held-out 0.86); codes land in the binding sweet-spot | `2026-06-15-off-diagonal-red-herring-...md` (CYCLES 88-90) | **numpy/rate result.** The *spiking-from-real* realization is boundary #15 below — keep the two separate |
| 5 | **Rate-code wall on single-neuron readout** — a single-neuron firing-rate code loses graded PPMI structure (20% of host) | **SURP** | **Population coding** — 16 neurons/dim → 66%, 32 neurons/dim → 94% of host. The brain's standard for graded values | `2026-06-15-...ppmi...md` (CYCLE 91); `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` | the faithful read climbs to ~host with enough population; the on-bridge *compute* of PPMI from raw counts needs a log-domain circuit (sub-piece, not a wall) |
| 6 | **On-bridge co-occurrence learning rule** — STDP is the wrong rule (symmetric co-occurrence has no pre→post order; measured 656k events / 0 weight change) | **SURP** | **Rate-Hebbian co-occurrence** learning (`corr(M,C)` +0.686 6-seed); the cortex learns the corpus word-by-word from the stream | `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` | validated at 64 concepts; on-bridge absolute fidelity is window-budget-bounded (wall-clock, not substrate) — honest scope flagged in the finding |
| 7 | **Nav action-selection / motor read-out** — the action came from a host Python `argmax` over spike counts | **SURP** | **Spiking WTA accumulator** (Wang-2002 NMDA attractor + Lo-Wang commit burst); now the LIBRARY default at **1.16× host, 100% commit-burst** (zero argmax fallback) | `2026-06-19-spiking-decision-default-on-GO.md` | the ~16% residual is the irreducible commit-timing/finite-size floor — the honest brain-based-only deliverable, not a tuning miss |
| 8 | **TD cue-shift / temporal credit** — Rescorla-Wagner lacks the bootstrap; phasic DA must migrate US→CS (the prime "dendrite candidate") | **SURP** | **Point-neuron A-CSC** (tapped-delay state + B-2 slow GABA_B conductance-derivative `+dV/dt` + short eligibility tau). Migration r = −0.80/−0.77/−0.89, full Schultz signature, anti-cheats decisive | `2026-06-10-N9-TD-cue-shift-A-CSC-GO.md`; scoped `2026-06-18-TD-cueshift-dendrite-decision-scoping.md` | the prompt's "this is where the dendrite earns its keep" was **empirically falsified** on point neurons. The standalone GO is solid; the *merged-bridge* lift is OPEN (#13) |
| 9 | **Multi-referent pronoun disambiguation** — a bare pronoun among several held referents (recency NEGATIVE, salience-boost NEGATIVE) | **SURP** | **WTA biased competition** (Desimone-Duncan lateral inhibition between referent attractors + content-graded bias). GO 6/6, bias-lesion breaks it 6/6, moat 6/6 | `2026-06-19-multireferent-graded-bias-polish.md` (closes `2026-06-17-multireferent-disambiguation-NEGATIVE.md`) | two converging NEGATIVEs first *located* the exact missing mechanism, then it was built on point neurons — a model of the pattern |
| 10 | **S5 per-query cleanup normalization** — the integrated loop's one residual host op (read the match score to drive the word-line) | **SURP** (de-risked, build pending) | **Divisive GAIN on the DIAGONAL** (Carandini-Heeger; `input_divisive_norm` primitive already in `sim/bridge.py`; NEF input-norm FS pool already validated). The non-negative rectified score has no common mode → the *diagonal/gain* half, NOT the off-diagonal whitening | `2026-06-19-S5-on-bridge-normalization-deep-research.md` (commits `94ca9fb8`/`1270397b`) | deep-research call dated 2026-06-19; the falsification de-risk (peak-sweep + moat) is the next step, not yet run. High-confidence (the exact normalizer already ships at D=2048) but not yet executed |
| 11 | **Spiking learned cortex on SYNTHETIC/strong structure** — spike-count readout lost the category code (~1 spike/neuron) | **SURP** | **Dense-firing readout** (~15 spikes/neuron via stronger coupling + longer window) → spike-count code reaches +0.40; the wall was spike-count SPARSITY, a config issue | `2026-06-15-phaseB-spiking-cortex-WALL-rate-to-spike.md` (CYCLE 64) | a **honest whipsaw**: an "airtight WALL/needs-dendrites" was declared at CYCLE 62 then **cracked** — the lesson (don't claim a fundamental wall before exhausting readout/regime knobs) is logged in the finding. Holds only for *strong/concentrated* structure (see #15) |
| 12 | **Nav neural reward** — reward was a host distance/sign formula | **SURP** (qualified) | **Neural reward** (spiking SC proximity / goal-salience approach reward); QUALIFIED GO with two op-point caveats | `2026-06-18-merged-neural-reward-QUALIFIED-GO.md` | the reward *organ* works in isolation; the closed *loop* around it is OPEN (#12-open, the SC-deploy NO-GO) — this row is the reward signal, #12-open is the sustained loop |
| 13 | **Nav reward/value/sustained-control LOOP** — the fully-neural SC-orient + neural-reward + critic + SNc closed loop can't sustain navigation (~58× worse than host; the actor goes silent) | **OPEN** | NOT yet resolved and NOT yet deep-researched as a unit. The scramble control localizes it to the **reward→SNc→critic→actor-drive** loop (orienting is fine); the actor fires in warmup then drops to ~0 | `2026-06-19-nav-spiking-sc-deploy-NO-GO.md` | **the foremost open boundary.** Same family as #14 (the value-train δ). Whether it is point-neuron-engineering or finally a real substrate wall is the open question — the research gate should hit this next |
| 14 | **Nav merged value-train δ (spatial value grading)** — the afferent-driven δ=r−V is graded-but-WEAK (~1.3×) vs the direct-drive ceiling (4-19×) | **OPEN** | V *is* learned co-resident (20× weight growth, critic-grade flip, lesion-confirmed δ direction); but the magnitude is capped by the **position-blind non-plastic up-state floor** (a structural property of the A1+A2 critic, needed to fire the cold MSN-D1) | `2026-06-18-merged-navcritic-valuetrain-BOUNDARY.md` | honest BOUNDARY; the cap is structural, not tuning. Cheap follow-ons named (sharper up-state / windowed read / learnable up-state). Point-neuron engineering, not obviously dendritic |
| 15 | **Nav #5 place-code δ (self-org place fields)** — a self-org sparse place code can't grade the value δ past 1.3× | **OPEN** (dendritic *flavor*) | sparsification FIXES value-learning (1.01→1.91× weight) but the **FS-ping-open read regime is NOT location-selective** (a few dominant cells fire everywhere) + the all-or-none coincidence-plateau readout over-clamps the SNc. Host-Gaussian place stays the better-δ scaffold | `2026-06-19-place-code-sparsify-default-BOUNDARY.md` | the finding itself calls it "closest to a tuning/architecture limit with a **dendritic flavor**" — a genuinely sparse+selective place code *would plausibly* need per-cell nonlinear input integration. The *immediate* blocker is a readout-regime + all-or-none readout issue (point-neuron-fixable); the *deeper* cause is the Mikulasch-Priesemann selectivity limit. NOT on the conversational path |
| 16 | **Spiking learned cortex on REAL weak/diffuse structure** — even with PPMI input + dense readout + E/I signed projection + per-hub adaptation, the spiking hub→cortex transform loses the *weak/diffuse* real category code (+0.06–0.155 vs host +0.44) | **DEND-DEF** | This is the one place a dendrite is still genuinely implicated: the real whitened structure is a *low-magnitude signed differential* that rate coding can't carry, and faithful per-feature whitening on a point neuron is the documented Mikulasch-Priesemann limit. RESERVED for the artificial-life goal | `2026-06-15-phaseB-spiking-cortex-WALL-rate-to-spike.md` (CYCLE 65, real-corpus NEGATIVE); `2026-06-15-...ppmi...md` (whole spiking-cortex arc, CYCLES 59-78) | **DEFERRED-BY-CHOICE, NOT blocking:** the conversational product ships the **flat 2,048-concept curated cortex** + PPMI; the spiking-from-real version is the deep frontier. AND it may be cheaper than dendrites (the off-diagonal de-risk #1 + the retinal/E-I escape reached marginal-but-positive without dendrites; the SM-lateral joint dynamics is the precise unconverged piece, not "needs a dendrite") |
| 17 | **Learnable multi-attribute (two-attribute) BINDING** — the K=5-load two-attribute boundary; can a learned dendritic multiplication generalize? | **DEND-RULED-OUT** | A learned dendritic supralinear (sigma-pi/plateau) conjunction binder **memorizes** (train 0.422) but does **NOT generalize** (held-out 0.168, below the fixed FHRR primitive's 0.261). The dendrite's native op buys memorization, not generalization | `2026-06-19-dendritic-binding-toy-derisk.md` | the binding wall is NOT (only) the missing dendritic multiplication — it is a deeper capacity/representation problem (more codes, or the F=3 resonator). Production keeps the fixed ±1/FHRR primitive |
| 18 | **Apical-basal CREDIT ASSIGNMENT for navigation** — the dendrite's other named job | **DEND-RULED-OUT** | NEGATIVE — a single-layer actor has nothing to route; the D2 two-compartment is a *spatial* decorrelation/feedback-alignment machine, the wrong dendrite for any current problem | `2026-06-19-dendrite-credit-assignment-toy-stage1.md`; scoped `2026-06-18-TD-cueshift-dendrite-decision-scoping.md` | the two named dendrite jobs (binding #17 + credit-assignment #18) are BOTH cheap-first NEGATIVE → "the dendrite is thoroughly assessed and ruled out for current walls" (CLAUDE.md commit `cc6cfd58`) |
| 19 | **Merged-TD-cue-shift CONSOLIDATION** — lift the standalone A-CSC GO onto the merged "one brain" | **OPEN** (engineering) | r<−0.7 IS reachable on the merged bridge (best −0.719) but the cue-pathway **LESION anti-cheat does not discriminate** (a learning-independent ~66 Hz cue-onset transient survives the lesion) → cannot be honestly certified | `2026-06-19-merged-TD-cueshift-opsearch-BOUNDARY.md` | explicitly a *merge-engineering* boundary (the merged SNc's onset excitability), NOT a substrate/dendrite finding — the standalone GO disproves a dendrite need. Named bounded fix (SNc onset-recovery). "Dendrite question stays CLOSED-NEGATIVE" |
| 20 | **Two-attribute (F=3 resonator) decode on the LEARNED production codes** — holds on clean phasors, degrades on correlated learned codes | **OPEN** (boundary, recorded) | flat + one-attribute decode **HOLD** (100% on learned codes); two-attribute **DEGRADES** (100% clean → 29% learned) because semantic correlation defeats the 3-factor permutation tie-break | `2026-06-19-resonator-on-learned-codes-derisk.md` | PARTIAL; the named next moves are decorrelate-the-grounded-phases (the standing whitening problem), a stronger restart schedule, or distinct per-attribute role tags. NOT shipped as a reliable capability; not blocking the production who/what turn |

---

## THE GENUINE DENDRITIC LIST (blocking vs deferred — the owner's core question)

**BLOCKING a shipped or critical-path goal: ZERO.** There is no capability currently delivered, or on the conversational
or navigation critical path, that is gated by a dendritic mechanism. Every conversational capability (parse · store ·
recall · abstain · negate · generate · dialogue-plan · learn-from-conversation · multi-hop · multi-turn) ships on point
neurons; the merged nav+conv "one brain" + the fully-spiking nav decision ship on point neurons.

**DEFERRED-BY-CHOICE: exactly ONE genuine candidate (#16).**

- **A faithful spiking-from-real-experience learned cortex that preserves WEAK/DIFFUSE real structure** (the real corpus
  category code is +0.44 at host; the spiking substrate loses it to +0.06–0.155 because the whitened structure is a
  low-magnitude signed differential rate coding can't carry, and per-feature whitening on a point neuron is the
  Mikulasch-Priesemann limit). **WHY it's deferred-by-choice and not blocking:** the conversational product ships the
  **flat 2,048-concept curated cortex** + PPMI codes, which generalize and pass the full pipeline. The
  spiking-from-real-experience version is the *artificial-life / biology-translatable* goal, reserved deliberately.
  **WHY it may not even be a dendrite when built:** the off-diagonal de-risk (#1) found a dendrite *unnecessary* (PPMI
  reaches the ZCA ceiling locally); the retinal center-surround + E/I signed-projection + per-hub adaptation escape
  reached marginal-but-positive on point neurons; the precise unconverged piece is the **SM anti-Hebbian lateral's joint
  spiking dynamics** (a recurrent point-neuron motif, `graded_lateral` already in `sim/`), not a missing dendrite. So
  even this "deferred dendritic" candidate is, on the latest evidence, more likely a *point-neuron convergent-dynamics*
  problem than a true dendritic one.

**DENDRITIC CANDIDATES THAT WERE TESTED AND RULED OUT: two (#17, #18)** — multiplicative binding and apical-basal credit
assignment. Both cheap-first NEGATIVE; both *saved* a months-scale build. The D2 two-compartment dendrite (`sim/
dendritic_neuron.py`, Larkum BAC / Guerguiev-Lillicrap-Richards) is a *spatial* decorrelation/feedback-alignment machine
— the wrong dendrite for the *temporal* credit (#8/#19) and shown un-load-bearing for the cortex code (#16's D2 Phase 2
clean-readout control inverted "gain confirmed").

---

## THE OPEN BOUNDARIES (the research gate's next targets, in priority order)

These are real, unresolved boundaries with no resolution yet — but they are point-neuron-engineering / op-point /
consolidation problems, **not (yet) dendritic**. None blocks a *shipped* capability; they block *deeper* goals.

1. **#13 — the nav reward/value/sustained-control LOOP (foremost).** The fully-neural SC-orient + neural-reward + critic +
   SNc closed loop underperforms the host ~58× and the actor goes silent. The scramble control localizes it to the
   reward→SNc→critic→actor-drive half. This is the biggest open item and the one CLAUDE.md/MEMORY flags ("expect
   point-neuron walls where the deferred dendritic substrate may finally earn its keep"). **It has NOT had a deep-research
   gate as a unit** — that is the recommended next move (the standing deep-research-first directive).
2. **#14 — the merged value-train δ magnitude** (graded-but-weak ~1.3×, capped by the position-blind up-state floor). A
   cheap-follow-on family is named; point-neuron engineering.
3. **#19 — the merged-TD-cue-shift consolidation anti-cheat** (r<−0.7 reachable but the lesion doesn't discriminate; a
   learning-independent cue-onset transient). A named bounded fix (SNc onset-recovery); merge-engineering, not substrate.
4. **#15 — the nav #5 self-org place-code δ** (readout-regime non-selectivity + all-or-none over-clamp). The one OPEN item
   with a genuine *dendritic flavor* (a truly sparse+selective place code may need per-cell nonlinear integration), but
   the immediate blocker is a point-neuron-fixable readout problem.
5. **#20 — two-attribute (F=3) decode on correlated learned codes** (29% vs 100% clean). Recorded boundary; not on the
   production who/what critical path; the named fixes route back to the (deferred) whitening problem or to per-attribute
   role tags.

---

## "WHAT WOULD CHANGE THIS PICTURE" (the blunt version)

- **If the nav reward/value/sustained-control loop (#13) turns out to need a dendrite, the count shifts** — that is the
  one large open boundary not yet deep-researched, and it is exactly where the project's own notes say a substrate wall
  *might* finally appear. Today it reads as a point-neuron op-point/loop-stability problem (the *organs* — SC, reward,
  critic — each work in isolation; the *loop* doesn't sustain), the same class as the #14 value-train δ. But it is honest
  to say: **this is the boundary most likely to reclassify**, and it should get the research gate next. Until then it is
  OPEN, not dendritic.
- **If #16 (the spiking-from-real cortex) is ever prioritized AND the SM-lateral joint-dynamics route fails on point
  neurons,** the deferred dendritic reservation becomes a real (deferred) dendritic requirement for the artificial-life
  goal — but it would still not block the *conversational product* (which ships flat + PPMI).
- **What would NOT change the picture:** more conversational levers. The conversational arc is comprehensively complete on
  point neurons; the two-attribute (#17, #20) and multi-referent (#9, now closed) frontiers were the candidates, and the
  dendrite was tested for #17 and ruled out.
- **The pattern to trust:** six "needs dendrites" verdicts (whitening, opponency, rate-code single-neuron, spiking-cortex
  CYCLE-62, TD cue-shift, S5) were raised and overturned on point neurons. The discipline that produced this — cheap-first
  de-risk + thorough readout/regime sweeps + the honest-whipsaw self-corrections (CYCLE 62→64, the D2 "gain confirmed"
  inversion) — is the reason the dendritic debt has NOT accumulated. The risk is the *reverse* of the owner's worry: the
  project has, if anything, been *quick* to cry "dendrite" and then disproved itself — so the standing scepticism toward
  the dendrite verdict is well-calibrated.

---

## Sources (verified against the actual finding text)

- `research/findings/2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md` (PPMI; off-diagonal red
  herring; CYCLES 88-92)
- `research/findings/2026-06-17-offdiagonal-dendritic-derisk-NEGATIVE-ship-flat-cortex.md` (off-diagonal dendrite gains INERT)
- `research/findings/2026-06-15-phaseB-spiking-cortex-WALL-rate-to-spike.md` (the whole spiking-cortex arc: WALL → crack →
  real-corpus NEGATIVE; CYCLES 59-78; the one genuine deferred-dendritic candidate, with the SM-lateral localization)
- `research/findings/2026-06-05-B-opponency-rate-coded-SNR-wall-CONFIRMED.md` (opponency → FHRR pivot)
- `research/findings/2026-06-05-composer-cleanup-NEF-GO.md` (NEF cleanup) [via CLAUDE.md + S5 doc cross-refs]
- `research/findings/2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (rate-Hebbian + population code on the real substrate)
- `research/findings/2026-06-19-spiking-decision-default-on-GO.md` (#4 motor read-out default-on, 1.16× host)
- `research/findings/2026-06-10-N9-TD-cue-shift-A-CSC-GO.md` + `2026-06-18-TD-cueshift-dendrite-decision-scoping.md` (TD cue-shift on point neurons; the "wrong dendrite" analysis)
- `research/findings/2026-06-19-S5-on-bridge-normalization-deep-research.md` (S5 = diagonal-gain, NOT dendritic; commits `94ca9fb8`, `1270397b`)
- `research/findings/2026-06-19-nav-spiking-sc-deploy-NO-GO.md` (the open nav reward/value/control loop)
- `research/findings/2026-06-18-merged-navcritic-valuetrain-BOUNDARY.md` (the value-train δ boundary)
- `research/findings/2026-06-19-place-code-sparsify-default-BOUNDARY.md` (#5 place-code δ; "dendritic flavor")
- `research/findings/2026-06-19-merged-TD-cueshift-opsearch-BOUNDARY.md` (merged-TD consolidation anti-cheat)
- `research/findings/2026-06-19-dendritic-binding-toy-derisk.md` + `2026-06-19-dendrite-credit-assignment-toy-stage1.md` (both dendrite jobs NEGATIVE)
- `research/findings/2026-06-19-resonator-on-learned-codes-derisk.md` (F=3 two-attribute on correlated learned codes)
- `research/findings/2026-06-19-multireferent-graded-bias-polish.md` + `2026-06-17-multireferent-disambiguation-NEGATIVE.md` (multi-referent: NEGATIVE → closed on point neurons)
- `research/findings/2026-06-14-D2-phase1-DONE-phase2-frontier.md` (D2 dendritic Phase 1 built byte-clean; Phase 2 gain NOT load-bearing on the spiking substrate)
- `sim/dendritic_neuron.py`, `sim/dendritic_mlp.py`, `sim/dendritic_plasticity.py` (the built D2 machinery); `sim/config.py:233,386,421,440` (`enable_dendritic_divisive_gain`, `enable_graded_lateral`, `enable_input_mean_adapt`, `enable_input_divisive_norm` — all default-off, byte-identical when off)
- `research/findings/AUTONOMOUS_STATE.md` (CYCLE 205-291 arc); CLAUDE.md (the documented capability arc)

_Read-only audit deliverable. No code, no experiments. Load-bearing "surpassed" claims verified against the actual
finding text; toy-scale / window-bounded / single-seed flagged where that is the truth._
