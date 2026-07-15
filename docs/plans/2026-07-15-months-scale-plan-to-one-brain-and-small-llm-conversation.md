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
| **Data (B)** | the wall is **precisely diagnosed**: even a full transformer + full-backprop LSTM lose to a *tuned/interpolated* n-gram at 5M-tok/V=300 (the CEILING finding); long-range signal only real at the **~23.7M-word / 60M-token** regime; fair-baseline discipline established (never add-1) | the **decisive scale run has not been run** (deliberately — trigram-bound at achievable scale). No corpus/vocab-ladder/wall-clock-budget/go-no-go build plan yet. Largest remaining item, least planned. |

**The defensible interim (state it plainly):** a **minimized ANN generator phrases the brain's grounded answer behind the gate-first no-confab moat; the brain does all cognition, grounding, composition, and the answer decision.** The generator is invoked *only after* the brain decides answer-vs-abstain (moat holds by construction). Given the 2026-07-15 systematicity resolution — the one thing an n-gram *structurally cannot do* is already solved and in production — this interim is defensible **longer** than ROADMAP §8's stand-in framing implies. It is a scored, temporary gap-(B) bridge, not a permanent dependency.

---

## 3. The shortcut burn-down SEQUENCE (ordered by leverage; "unblocks" is the column §8 lacks)

| # | Shortcut / stand-in | Status | Retirement path | Effort · compute | Unblocks |
|---|---|---|---|---|---|
| 1 | **FHRR exact-inverse bind → learned bilinear** | de-risked (numpy 0.87 held-out 12-seed + transport-free 6/12; `2026-07-15-TEST-A` + `-learned-bilinear-...`) | ship the learned bilinear binder over decorrelated codes; on-substrate fixed-conjunction bind with LEARNED projections trained by the committed feedforward BDSP rule (the deep-credit-GO regime) | **days-weeks**, GPU | systematicity engine off the "idealization" list; the shallow learned-representation build |
| 2 | **word-cortex normalization in host code** | de-risked-rate-level (CYCLE 93b) | per-concept feedforward inhibition + per-hub adaptation post-f-I | **days-weeks**, GPU | fully-brain-based read-out |
| 3 | **formula reward residue + missing value critic** | partial (spiking DA-RPE done; explicit critic NOT built) | build the explicit spiking value-critic population (§5.4) | **weeks**, GPU | RL self-taught policy (gates #7); value purity |
| 4 | **given vs stream-learned codes (64 vs 320)** | partial (stream cortex GO @64 on-bridge) | scale learn-from-listening to 320 — **bottleneck is a DATA gate**: a cleanly-extractable is-a/taxonomic corpus (triple-negative on WikiText/TinyStories/dictionary, CYCLE 1038) | **weeks** data-acquisition + GPU | fully-learned 320-concept vocabulary |
| 5 | **emergence content-addressable store = on-bridge WM** | de-risked-rate-level GO (6-seed) | unite the store + RUNG6d Hebbian STP binder on ONE bridge (a-1 caveat: the FS-WTA on-bridge binder is BANKED, RUNG6e/6f — do the tractable clean-barcode-key version) | **weeks**, GPU | activity-silent synaptic WM literal |
| 6 | **ANN generator (fluency crutch)** | spiking-forward validated to 88.6M; ANN for speed | **shrink-and-gate now; convert later** once the open-generation ladder scales (Gate B) | interim **days**; conversion **weeks** | fully-spiking speech |
| 7 | **pre-allocated vs self-organized growth** | partial | structural self-organization (dendritic frontier) | **months** | learn-new-concepts-live at scale |

---

## 4. The unification CRITICAL PATH ("one-brain-that-LEARNS")

The learning rule (feedforward deep-credit / BDSP, GO) → the recurrent language cortex that develops structure from a stream → learned codes at full vocab (#4, DATA-gated) → the composer becoming a LEARNED binder (#1, rate-level GO) → the generator spiking (#6) → structural growth (#7). **★ Single highest-leverage next edge:** unite the content-addressable store (#5) + the RUNG6d STP binder on one bridge (activity-silent synaptic WM). **Longest pole:** co-training the learning pieces (stream cortex + deep-credit + long-range learner) *without cross-talk* at scale — the plasticity-isolation gates are validated but simultaneous stream-cortex co-training is unshown.

---

## 5. The DECISION GATES (updated 2026-07-15 to the owner's broadened compute policy)

**Gate A — invest significant wall-clock (day+ runs).** Fire when a learned lever (R3 learned-W_in / 2-stage read-out) **crosses a FAIR n-gram** (tuned add-k bigram + interpolated trigram, same split) at achievable scale — i.e. the thing we'd train is the BRAIN learning real structure, not a scaffold racing a bigram. **Per the owner (2026-07-15): day+ runs need NO approval — notify, then run — gated on MY judgment of sufficient justification.** Not a fishing expedition; a decisive experiment or a validated-mechanism scale run.

**Gate B — convert the generator to spiking (vs keep the ANN crutch).** Trigger (KV-cache speed lever) is MET (111 tok/s, real-time-viable). But converting does NOT buy fluency (fluency = scale/data). So the live decision is **shrink-and-gate the ANN now (interim, days)** vs **deploy the validated spiking-forward (weeks)**. **Recommended: shrink-and-gate now; defer conversion** until the brain's own open-generation ladder scales. (Update CLAUDE.md L88 + ROADMAP §8 #1 which still say "held/deferred.")

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
