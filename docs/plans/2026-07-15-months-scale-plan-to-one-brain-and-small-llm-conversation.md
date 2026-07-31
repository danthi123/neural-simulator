---
type: plan
status: live
date: 2026-07-15
---

# The Months-Scale Plan: One Brain, All Spiking → Small-LLM-Level Conversation

*Synthesized 2026-07-15 from a 4-lens strategic audit Workflow (`wf_7a411db0-3b1`) over ROADMAP.md §3/§8/§9, AUTONOMOUS_STATE.md, and the grounded findings cited inline. This is the forward-looking PLAN; `ROADMAP.md` remains the at-a-glance STATUS surface. Prompted by an owner strategic conversation (2026-07-15).*

---

## 1. Verdict — was the path already clearly documented?

**No. The *status* was excellently documented; the *plan* was not.** A strategic conversation was genuinely needed, and it surfaced a synthesis the docs held only diffusely.

- **Already clear (stays authoritative):** ROADMAP.md is a strong STATUS map — §3 (one-screen picture), §5 (13 developmental stages, badged + cited), §8 (6 stand-ins), §9 (7 frontier walls). §9.1 already carries the 2026-07-15 resolution (fluency = data/scale not brain design; systematicity = fixed/bilinear bind over decorrelated codes, validated numpy + spikes across 320 concepts).
- **NOT captured anywhere as a followable plan:** (1) the **two-gap framing** — (A) unification-of-separately-validated-pieces + (B) scale/data; (2) the shortcut burn-down as an *ordered* sequence with "blocks-what"; (3) the **decision gates** (invest-wallclock / convert-generator / spend-cloud), scattered across a raw findings doc + CLAUDE.md + an auto-memory; (4) a unification dependency graph with a single named next edge + longest pole.
- **Stale plan layer:** `docs/plans/2026-06-08-brain-fidelity-roadmap.md` (~5 wks stale, silent on scale), `2026-06-23-inventory-burndown-roadmap.md` (~3 wks stale, Phases 1-3 largely already committed), `2026-05-11-strategic-reevaluation.md` (~2 mo stale). This doc is their successor.

---

## 2. The four gap axes — biologize · integration · speed · data

| Axis | Done | Left |
|---|---|---|
| **Biologize** | nav decision spiking-by-default; spiking WTA cleanup default-on; discourse router routes-by-meaning (17-agent audited); one-backend-per-process RESOLVED (EMERGE-71 `SimulationBridge.xp`); spiking DA-RPE; A→W spelling on spikes; systematicity bind is on-spike | explicit **spiking value critic** (ROADMAP §5.4, highest-value); word-cortex normalization circuit (host log-domain → feedforward inhibition, scoped CYCLE 93b); conduction delays + multi-compartment neurons (protected additive `sim/`); fully-spiking novel-referent fast-weight |
| **Integration (A)** | conversational **core** co-executes fully-spiking in ONE cupy process (EMERGE-70/71); nav+conv merged; cross-region synaptic routes proven | the **learning** pieces are not yet ONE co-resident continuously-learning brain — stream cortex + deep-credit rule + selective-SSM long-range learner + self-organized producer are each validated as *separate* de-risks. **"one-brain-that-ANSWERS" is done; "one-brain-that-LEARNS" is the open crux of gap (A).** |
| **Speed** | the spiking-forward tax is largely removed: dense matvec ~3600-13000×/shape; on-GPU forward killed 114→1 H↔D copies/token; **KV-cache bit-exact, 12.5× prefill, 111 tok/s** (CYCLE 514) — the tax is launch-bound, not FLOP-bound | reservoir-LM scale path is **core-parallelism-bound, not GPU-bound** (§9.1) — levers here are CUDA-graph / sentence-batching; O(N²) `.todense()` per-bridge cap (~30-50K neurons ≈ ≤320 concepts) |
| **Data (B)** | **the wall is precisely diagnosed, and as of 2026-07-16 it is MEASURED on BOTH model classes.** At 5M-tok/V=300 on WikiText a full GPU **transformer** loses to an add-1 bigram at every depth, monotonically worse with depth (−0.059 → −0.379; on that corpus there is **no** vocab regime where it cleanly wins — V=300 barely trains, V≥2000 catastrophically overfits). **✅ And so does a full-backprop LSTM** — 3 seeds, **0/3 beat the bigram**, deep-margin mean **−1.415** (range −1.445…−1.392), the transformer's exact signature only stronger. *(⚠️ This LSTM half was ASSERTED in this cell for weeks with NO measurement behind it — the runner defaults V=2000/24M and the string "300" does not occur in it. The 2026-07-16 anchor-claim audit caught it; the settle then took ~4 min of GPU and CONFIRMED the claim. An assertion that happens to be true is still an assertion.)* The 2026-07-15 fair-baseline probe likewise shows **none of our levers beats an interpolated trigram at achievable scale, at any depth**. Long-range signal only real at the **~23.7M-word / 60M-token** regime — where the same LSTM runner **BEATS** the bigram at every depth with a margin that GROWS with context (+0.494→**+1.813**), i.e. recurrence demonstrably holds long-range **once the data is there**. Fair-baseline discipline established (never add-1). **⇒ the exculpation now holds for RECURRENCE too — the one class the spiking substrate actually is: at our data scale a full-backprop recurrent net is n-gram-bound exactly as a transformer is, so "our brain is only n-gram-level" is a property of the DATA REGIME, not a defect of the substrate.** *Caveats: **"5M-tok" is a MISNOMER** wherever it appears — `wikitext.txt` is 2,045,059 words, so both runs cap at ~2M (identically ⇒ the comparison holds). **Both models are OVER-PARAMETERIZED** for the corpus (2.6M params / ~2M words; the CEILING finding says this of itself) ⇒ this demonstrates **the data regime, not a recurrence-specific limit**; a right-sized recurrent model at 2M words is unmeasured. Findings: `2026-07-16-D5-SETTLED-*.md`, `2026-07-16-anchor-claim-audit-*.md`.* | the **decisive scale run has not been run** (deliberately — trigram-bound at achievable scale). No corpus/vocab-ladder/wall-clock-budget/go-no-go build plan yet. Largest remaining item, least planned. |

**The defensible interim (state it plainly):** a **minimized ANN generator phrases the brain's grounded answer behind the gate-first no-confab moat; the brain does all cognition, grounding, composition, and the answer decision.** The generator is invoked *only after* the brain decides answer-vs-abstain (moat holds by construction). Given the 2026-07-15 systematicity resolution — the one thing an n-gram *structurally cannot do* is already solved and in production — this interim is defensible **longer** than ROADMAP §8's stand-in framing implies. It is a scored, temporary gap-(B) bridge, not a permanent dependency.

---

## 3. The shortcut burn-down SEQUENCE (ordered by leverage; "unblocks" is the column §8 lacks)

| # | Shortcut / stand-in | Status | Retirement path | Effort · compute | Unblocks |
|---|---|---|---|---|---|
| 1 | **FHRR exact-inverse bind → learned bilinear** | de-risked (numpy 0.87 held-out 12-seed + transport-free 6/12; `2026-07-15-TEST-A` + `-learned-bilinear-...`) | ship the learned bilinear binder over decorrelated codes; on-substrate fixed-conjunction bind with LEARNED projections trained by the committed feedforward deep-credit rule **in its GO regime = e-prop-family credit + POPULATION CODING** (see the 2026-07-16 correction note below -- writing "BDSP" unqualified here was misleading: bare on-bridge BDSP is a 6-seed at-or-below-chance NEGATIVE) | **days-weeks**, GPU | systematicity engine off the "idealization" list; the shallow learned-representation build |
| 2 | **word-cortex normalization in host code** | de-risked-rate-level (CYCLE 93b) | per-concept feedforward inhibition + per-hub adaptation post-f-I | **days-weeks**, GPU | fully-brain-based read-out |
| 3 | **formula reward residue + missing value critic** | partial (spiking DA-RPE done; explicit critic NOT built) | build the explicit spiking value-critic population (§5.4) | **weeks**, GPU | RL self-taught policy (gates #7); value purity |
| 4 | **given vs stream-learned codes (64 vs 320)** | partial (stream cortex GO @64 on-bridge) | scale learn-from-listening to 320 — **bottleneck is a DATA gate**: a cleanly-extractable is-a/taxonomic corpus (triple-negative on WikiText/TinyStories/dictionary, CYCLE 1038) | **weeks** data-acquisition + GPU | fully-learned 320-concept vocabulary |
| 5 | **emergence content-addressable store = on-bridge WM** | de-risked-rate-level GO (6-seed) | unite the store + RUNG6d Hebbian STP binder on ONE bridge (a-1 caveat: the FS-WTA on-bridge binder is BANKED, RUNG6e/6f — do the tractable clean-barcode-key version) | **weeks**, GPU | activity-silent synaptic WM literal |
| 6 | **ANN generator (fluency crutch)** | spiking-forward validated to 88.6M; ANN for speed | **shrink-and-gate now; convert later** once the open-generation ladder scales (Gate B) | interim **days**; conversion **weeks** | fully-spiking speech |
| 7 | **pre-allocated vs self-organized growth** | partial | structural self-organization (dendritic frontier) | **months** | learn-new-concepts-live at scale |

---

## 4. The unification CRITICAL PATH ("one-brain-that-LEARNS")

The learning rule (feedforward deep-credit — **⛔ NOT-GO as of 2026-07-17: on valid seed-fixed data the trained net does NOT reliably beat a fixed random spiking reservoir (4/5 blind seeds negative; the prior "GO" was ~80% reservoir on an UNSEEDED substrate — bug fixed). Node Perturbation, the roadmap's "next bet", is off-brain-GO but REFUTED vs Kolen-Pollack (12-seed) and retired on-spike 2026-07-13. The blocker is a supervised spiking-READOUT wall shared across e-prop/D1/NP. ⇒ the supervised-deep-credit link of THIS critical path is PARKED; the working learning path is the UNSUPERVISED on-spike stream cortex. Map: `research/findings/2026-07-17-learning-rule-frontier-map-eprop-NOT-GO-NP-retired-the-real-blocker-is-the-shared-readout-wall.md`**) → the recurrent language cortex that develops structure from a stream → learned codes at full vocab (#4, DATA-gated) → the composer becoming a LEARNED binder (#1, rate-level GO) → the generator spiking (#6) → structural growth (#7). **★ Single highest-leverage next edge:** unite the content-addressable store (#5) + the RUNG6d STP binder on one bridge (activity-silent synaptic WM). **Longest pole:** co-training the learning pieces (stream cortex + deep-credit + long-range learner) *without cross-talk* at scale — the plasticity-isolation gates are validated but simultaneous stream-cortex co-training is unshown.

---

## 5. The DECISION GATES (updated 2026-07-15 to the owner's broadened compute policy)

**Gate A — invest significant wall-clock (day+ runs).** Fire when a learned lever (R3 learned-W_in / 2-stage read-out) **crosses a FAIR n-gram** (tuned add-k bigram + interpolated trigram, same split) at achievable scale — i.e. the thing we'd train is the BRAIN learning real structure, not a scaffold racing a bigram. **Per the owner (2026-07-15): day+ runs need NO approval — notify, then run — gated on MY judgment of sufficient justification.** Not a fishing expedition; a decisive experiment or a validated-mechanism scale run.

**Gate B — convert the generator to spiking (vs keep the ANN crutch).** Trigger (KV-cache speed lever): ⚠️ **CORRECTED 2026-07-16 — "111 tok/s" is the PREFILL rate, not decode.** `raw/_burndown_2A_full_build_o1_o3.json` `timing`: `production_o1_prefill_tok_per_sec: 111.3` (a 12.5× **prefill** speedup, from staying GPU-resident — the file's own note attributes it to killing per-linear H↔D, **not** to the KV cache), while `o3_cached_generation_tok_per_sec: **19.83**` vs `no_cache_full_recompute_tok_per_sec: 19.31` — i.e. the KV cache landed bit-exact but buys **~1.03×** at this tier (unsurprising: 8 generated tokens from a 24-token context, where an O(1)-vs-O(context) saving barely bites). Measured on the **494M co-resident Qwen**, *not* the ~21M generator this gate governs. **19.8 tok/s may well be real-time-viable, so the gate's RECOMMENDATION likely survives — but it is not measured at chat context length nor on the model in question.** (`:23` of this doc labels it correctly as "12.5× prefill, 111 tok/s"; this line contradicted it.) But converting does NOT buy fluency (fluency = scale/data). So the live decision is **shrink-and-gate the ANN now (interim, days)** vs **deploy the validated spiking-forward (weeks)**. **Recommended: shrink-and-gate now; defer conversion** until the brain's own open-generation ladder scales. (Update CLAUDE.md L88 + ROADMAP §8 #1 which still say "held/deferred.")

**Gate C — spend on cloud.** *(REVISED from the audit's VRAM-only framing per the owner's 2026-07-15 clarification.)* The owner has **no objection to cloud in principle**; the gate is **MY judgment that "we're sufficiently developed to justify the spend"** — i.e. the thing we'd run is the real unified brain (shortcuts biologized/unified enough that the compute trains the substrate, not scaffolds) AND the result is decision-decisive. Cloud is a ~3-5× turnaround accelerator (e.g. 22h→5h) for the far-tier ~23.7M-word confirmation once a learned lever passes Gate A. Note: the reservoir-LM is core-parallelism-bound, not GPU-bound, so cloud/GPU may not help until scale grows past a real VRAM wall — always MEASURE VRAM + throughput + ETA first, then it's my call.

---

## 6. Honest distance (keep the bounded milestone separate from the open-ended wall)

**(a) One-brain, all-spiking: ~2-4 months of focused work.** Nothing fundamentally blocked. The conversational core already co-executes fully-spiking in one cupy process (EMERGE-70/71); the remaining work is integration engineering, mostly no `sim/` edit — Edge 5 (weeks) → co-locate the learning pieces (weeks-months) → the value critic + normalization circuit (weeks). Key uncertainties: the RUNG6d + content-store merge preserving horizon on real barcode keys; co-training the learning pieces without cross-talk at scale.

**(b) Small-LLM-level conversation — two horizons, kept separate:**
- **Bounded milestone (~3-5 months):** scaled-vocab, grounded conversation where the brain does all cognition/grounding/composition and a minimized ANN phrases the answer behind the moat. Gated on: 320-concept stream cortex (#4, DATA-gated) + the learned bilinear binder (#1) + open-vocab spiking spelling (bounded per-bank engineering). **This is what "months" honestly means.**
- **Open-ended wall (unbounded):** fully model-free open-domain fluency from the brain's OWN spikes — a **field-wide unsolved problem**, not project-specific. Routes to the DATA/SCALE pole + possibly a deep-recurrent-credit rule (a possibly-multi-quarter open problem). **Do not promise this in months.** Manage with grounding + composition + abstention.
- **Key uncertainty resolving (b):** whether a learned long-range INPUT representation (R3) crosses a FAIR n-gram at the 23.7M-word regime. Crosses → the full-corpus run is warranted (Gate A→C). Asymptotes above zero → the wall is genuinely data/scale or needs the deep rule. **One test resolves this (§7 #2).**

---

## 7. Immediate next 2-3 concrete builds

1. **★ Edge 5 — unite the content-addressable store + RUNG6d Hebbian STP binder on ONE `SimulationBridge`** (clean barcode keys; a-1: the FS-WTA on-bridge binder is BANKED per RUNG6e/6f — do the tractable version). The single highest-leverage integration edge; reuse-by-import, no `sim/` edit; 6-seed anti-cheated. **Weeks, CuPy.**
2. **The fair-baseline scale-lever re-verify (resolves the (b) uncertainty + arms Gate A).** Re-run the best learned lever (R3 learned-W_in / 2-stage read-out) vs a **tuned + interpolated** n-gram on the same split at achievable scale. Decision-useful independent work that keeps the cores busy. Crosses → arms Gate A→C; asymptotes → confirms the DATA pole. **Weeks, core-parallel.**
3. **Ship the learned bilinear binder as the FHRR retire-step (#1) + re-run Gate B.** Land the bilinear binder over decorrelated codes as the production systematicity engine (numpy done; on-substrate learned-projection build via committed BDSP), AND re-evaluate Gate B (shrink-and-gate vs convert). **Days-weeks, GPU.**

**Parallel doc action (days, near-zero risk):** add **ROADMAP §10 "The months-scale plan"** pointing here; fix stale pointers (AUTONOMOUS_STATE §12→§9; the ROADMAP header sync-date; §9 item-1's "top lever" headline — the deep rule is OFF the open-generation critical path); mark the two stale plan docs superseded.

---

*The near-term critical path is integration (Edge 5) + settling the fluency question with a fair test; the deep learning-rule frontier is correctly parked; the ANN generator stays a scored interim behind the moat, not a permanent dependency. The science status was well-documented; this sequenced plan + the decision gates were the missing piece — now committed.*

---

## ⚠️ CORRECTION (2026-07-16) — "the committed feedforward BDSP rule (the deep-credit-GO regime)" was MISLEADING; the GO ingredient is POPULATION CODING

Caught while staging the longest pole's segment (b) (co-train the stream cortex WITH the deep-credit learner): §3 row 1
originally named the deep-credit-GO regime "the committed feedforward BDSP rule" unqualified. Read against the findings,
that phrasing points a reader at a rule this project has already exhausted:

- **Bare on-bridge BDSP (burst-multiplexed, point neurons, no population coding) = a decisive 6-seed NEGATIVE.**
  `2026-07-15-onsubstrate-bind-onbridge-bdsp-readout-RUNG3-BOUNDARY.md`: "BDSP is AT-or-BELOW chance on ALL 6 seeds --
  it NEVER gets above chance"; the moat does not hold (BDSP ~= lesion on 5/6); NEGATIVE across lr / P-bar / pool-k /
  task-difficulty / the dense-redundant BurstCCN regime. Crucially the obvious escape was already tried and closed: the
  apical-DECOUPLED confound was FOUND and FIXED (`--soma-g 120` makes B rise 0.000 -> ~0.24, directed credit genuinely
  delivered) and **the boundary still held on both task families** -> a genuine credit-QUALITY limit, not a wiring bug.
- **The GO is the e-prop-family rule PLUS population coding.** The 2026-07-15 deep-research gate: "feedforward spiking
  deep credit is ALREADY GO (e-prop + population coding)" -- transport-free forward eligibility x membrane-potential
  surrogate sigma'(v-theta) x fixed-random DFA, 6-seed GO, shuffle-DFA collapses, K=8 -> 0.877 ~= the LIF ceiling 0.89.

**These are NOT rival rules and the plan is not fabricating** -- `2026-07-11-eprop-recurrent-learning-...md` states
e-prop/RFLO IS "the rate analogue of the BDSP/Burstprop already in `sim/bridge.py` (`enable_bdsp`)", i.e. the SAME family,
and BDSP is the committed `sim/` kernel. The structure is this project's recurring **rate-vs-spike gap**:
rate e-prop/RFLO = REAL-WITH-SCOPE; the same family on SPIKING point neurons = NEGATIVE; **population coding is the
ingredient that closes it** (it graded-ifies the sparse binary spike signal the eligibility is built from -- the same
lever that later FAILED to rescue the RECURRENT off-diagonal, which is a separate, now-CLOSED arc).

**⇒ Operational consequence for the longest pole (line 46).** Segment (b) co-trains the stream cortex with the deep-credit
learner **in its GO regime (e-prop-family + population coding)**. Co-training bare on-bridge BDSP would be meaningless:
a learner that never exceeds chance cannot show whether co-residence costs it anything. Anyone reading §3 row 1 and
building on unqualified "BDSP" would burn a GPU cycle on a closed boundary.


---

## ⛔ CORRECTION (2026-07-16) — the critical path's FIRST link is not what it says. "The learning rule (feedforward deep-credit / BDSP, GO)" is wrong twice over.

§4's unification critical path opens with *"The learning rule (feedforward deep-credit / BDSP, GO)"*. Both halves of that are now corrected, and the second correction is the serious one.

**1. It is NOT BDSP.** Bare on-bridge BDSP is a decisive **6-seed at-or-below-chance NEGATIVE** (`2026-07-15-onsubstrate-bind-onbridge-bdsp-readout-RUNG3-BOUNDARY.md`) — and the obvious escape was already closed: the apical-DECOUPLED confound was FOUND and FIXED (`--soma-g 120` makes B rise 0.000→~0.24, directed credit genuinely delivered) and **the boundary still held on both task families** ⇒ a credit-QUALITY limit, not a wiring bug. The regime that works is **e-prop-family credit + POPULATION CODING**. (§3 row 1 corrected separately, same day.)

**2. The "GO" itself is ~80% a RESERVOIR — and it was lifted out of runs that reported HONEST NEGATIVE.** The banked headline (*"K=8 0.877 ≈ LIF ceiling 0.89, anti-cheat-clean"*) traces to `research/findings/raw/_epropport/k8_s4{2,3,4}.json`, whose `inherit` values **0.889/0.926/0.815 average to exactly 0.877** — and **all three report `SIGNAL=False` with `shuf_ok=False`** (shuffle-DFA 0.556/0.593/0.630 against a required ≤0.433), each printing *"HONEST NEGATIVE — the ported e-prop does NOT cleanly train the task on the bridge"*. **The instrument was not broken; it was OVERRIDDEN.**

Measured with the frozen-hidden RESERVOIR control that the gate never had (the `train_layers` isolation hook existed in-file, documented for exactly this, and had **never once been invoked**):

| seed | FULL | FROZEN (fixed random reservoir + linear readout) | deep-credit contribution |
|---|---|---|---|
| 42 | 0.852 | 0.667 | +0.185 |
| 43 | 0.926 | 0.889 | **+0.037** |
| **mean** | **0.889** | **0.778** | **+0.111** |

Above chance (0.333): **reservoir +0.444, deep credit +0.111 ⇒ the reservoir is ~80% of the margin.** Deep credit is REAL and positive but MINOR and seed-variable. Reproduction is exact (seed 43 = 0.926, matching `k8_s43` to 3dp, `SIGNAL=False`) ⇒ not a migration artifact.

**CONSEQUENCES FOR THIS PLAN:**
- **The critical path's first link is weaker than stated.** "The learning rule ... GO" should read: *e-prop + population coding reaches ~0.89 held-out inheritance on the bridge, but ~80% of that margin is a fixed random spiking reservoir + a linear readout; the deep-credit share is ~20% and seed-variable; the runner's own aggregate gate does not pass.*
- **The LONGEST POLE's segment (b) is GATED.** Co-training "the stream cortex + the deep-credit learner" would mostly test co-residence of a **reservoir**, not of a second learning **RULE** — which is the entire purpose of (b) (rule heterogeneity). Segment (a) is DONE (see the ROADMAP §10 entry: 93.2% retained, 5/6 GO, cost decomposed ~97% time-sharing / ~3% interference ⇒ co-training SCALES).
- **The 2026-07-15 gate's "feedforward is SOLVED / not a blocker — the frontier is RECURRENT off-diagonal" is wrong at its root**, and that conclusion REDIRECTED the programme: the off-diagonal arc was prioritized partly BECAUSE feedforward was believed solved, and it then closed as a decisive negative. The feedforward side never passed its own aggregate gate. **The frontier is wider than this plan says.**

**STATUS:** a **6-seed FULL-vs-FROZEN** is in flight (the above is n=2 = INDICATIVE, against the standing 6-seed rule). The gate now carries the reservoir control DEFAULT-ON + CI-guarded (`tests/test_plasticity_inertness.py`), so this cannot recur silently.

**STANDING RULE this produced:** *never average a metric out of a run whose own `SIGNAL` is False.* A runner printing `HONEST NEGATIVE` has already done the analysis.

Full audit: `research/findings/2026-07-16-deep-credit-GO-is-80pct-RESERVOIR-the-frozen-hidden-control-was-never-run.md`.
