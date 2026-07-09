# D3 — recurrent multi-hop composition: research gate for the cheap-first de-risk (does a LEARNED recurrent-credit path learn a bind→re-bind composition a feedforward net PROVABLY cannot?)

**Date:** 2026-07-09
**Type:** READ-ONLY research-gate scoping. NO build, NO GPU, NO `sim/` edit, NO multi-seed sweep. The one file written is this doc. The controller reads the load-bearing sources itself and builds.
**Trigger:** the feedforward deep-credit arc closed (`2026-07-08-deep-credit-feedforward-arc-COMPLETE-*.md`): FF deep-credit's depth-benefit is real but NARROW (needs a nonlinear conjunction over a POOLED rep — the XOR-over-pool positive control, deep-best 0.993 vs 1-layer 0.499); the "natural" tasks all shortcut. The NAMED frontier: **does a recurrent credit path (BPTT-SNN / e-prop / the fronto-striatal reservoir) LEARN a MULTI-HOP bind→re-bind that a feedforward net PROVABLY cannot** — the temporal analog of XOR-over-pool.

---

## THE HEADLINE (say it plainly, up front)

**D3 is cheaply de-riskable NOW**, by reuse-by-import + **ONE small new piece** (a ~40-line numpy vanilla-RNN-BPTT oracle — the recurrent analog of `DendriticMLP.oracle`). NO `sim/` edit for the rate de-risk.

**The one load-bearing correction to the prior gate.** The 2026-07-08 recurrent-multihop gate concluded "already done 3 ways — CONSOLIDATE, don't invent." That verdict is **over-broad and answers a different question.** On the controller's PRECISE question — *does a LEARNED recurrent-credit path solve a task a feedforward net PROVABLY cannot* — **none of that gate's three "DONE pillars" qualifies:**

| prior "pillar" | what it actually is | why it does NOT answer D3 |
|---|---|---|
| EMERGE-28 transitive chaining | UNSUPERVISED HTM sequence memory | transitive inference has a **scalar-monotone-score shortcut** — the deep-credit arc's OWN map lists "transitive inference (order) — NOT depth-required — any monotone scalar score is transitive." A feedforward net solves it. Not the FF-impossible task. |
| Reservoir comprehension (EMERGE-78..82) | **FIXED** random recurrent weights + **trained LINEAR read-out** | the recurrence is a *fixed dynamical system*; the only learning is a linear read-out. This is NOT a learned recurrent-credit path. And its non-locality is CONTINGENT (`that` is OOV, EMERGE-78's own honest note). |
| WM buffer (EMERGE-85/86) | **HAND-DESIGNED** mirror-pair coincidence feature (`WMBuffer`, `_emerge85:69-94`) + trained ridge | the stack structure is hand-installed, not learned. Not learned recurrent credit. |

And the prior gate's "do NOT re-run supervised recurrent credit (5×-confirmed dead-end)" **conflates two task regimes**: the EMERGE-6/6b "dead-end" was on **autonomous sequence GENERATION** (free-run trajectory recall, where the wall is *generation stability* / exposure bias). A bind→re-bind **CLASSIFICATION** (read a property of the final composite, input-driven throughout) is **teacher-forced-throughout** — the exact regime where EMERGE-6b's e-prop DID learn the map (`eprop_tf` one-step 0.695). **The D3 question is genuinely OPEN, and plausibly winnable in a regime that sidesteps the documented wall.** This is the SURPASS-sharpening move: the comfortable "already done" verdict is the START of the research, not the end.

---

## PART (a0) — MAP OF THE EXISTING RECURRENT-CREDIT MACHINERY (read, not guessed)

### The reservoir arc EMERGE-78/79/82/83/84/85 — FIXED reservoir + trained linear read-out (NOT learned recurrent credit)

`research/runners/_emerge78_reservoir_form_to_role_derisk.py`:
- `class Reservoir` (L155-170): `W_in` random fixed; `W_res` random, sparsified (`_RES_DENSITY=0.1`), rescaled to **spectral radius 0.95** (L159-163) — **set once from seed, NEVER trained.** `leak=0.3`, `tanh`. `final_state(U)` (L166-170) = leaky-integrate the sequence, return the FINAL hidden state.
- The only trained part is a **ridge read-out** on the final state (`_fit_slots` L197-204, `np.linalg.solve`). Echo-state computing: fixed nonlinear temporal expansion + linear read.
- **This is a fixed-dynamics feature extractor, not a recurrent credit path.** Whatever composition it does is whatever the fixed random dynamics happen to make linearly separable in the high-dim state (the RC "kernel").

`_emerge79...` (variable-distance, GO): the reservoir's fading memory holds a REAL discovered marker across ≈variable distance and beats every fixed window — but this is **bounded graded memory** (a held cue), not composition. Names the reservoir's fading-memory depth.

`_emerge84_reservoir_stack_recursion_derisk.py` (the boundary that matters): genuine stack-recursion (nested subject-verb pair-matching, **multiset-preserving swap so the count shortcut is at chance** — `_count_multiset_baseline_acc` L133-145, swap L92-100). The plain reservoir does depth-1 then **degrades to d\*=2** — "fading memory, NOT a push/pop stack." This is the reservoir's intrinsic recursion limit.

`_emerge85_wm_buffer_recursion_derisk.py` (the "surpass"): `class WMBuffer` (L69-94) — a **HAND-DESIGNED** ordered-slot multiplex that computes per-mirror-pair coincidence `f[k] = 1.0 if idx[k]==idx[N-1-k]` (L90-91). Trained ridge on that hand-feature reaches d\*=3. **The stack is hand-installed, not learned.** `slot_scramble` is the anti-cheat.

`_emerge82_onbridge_lsm_derisk.py`: `OnBridgeLSM` — the fixed reservoir realized as a recurrent Izhikevich `BrainRegion` on a real `SimulationBridge` (`internal_density=0.1` fixed-random recurrent synapses; read-out = spike-counts). This is the **spiking realization of the reservoir arm** (the arm-1 spiking port target, deferred behind a rate GO).

### The learned recurrent-credit machinery (the genuinely-relevant assets)

`research/runners/_emerge6_recurrent_microcircuit_seq_derisk.py`:
- `class RecurrentMicrocircuitRNN` (L90-174): a genuine **learned recurrent weight `W_rec` (NxN, L109)**, trained by LOCAL credit with TWO eligibility forms selected by `elig`:
  - `"forward"` (κ=0): memoryless map + low-pass input eligibility (the rung-3a baseline).
  - `"eprop"` (κ>0): a **proper e-prop first-order eligibility** (Bellec 2020) on a leaky-integrator unit — per-synapse `eps_ji = κ·eps_ji + pre_i`, gated by `phi'(u_j)`, credit `dW += (target - a)·phi'·eps` (L148-151). **Local, online, NO BPTT, NO weight transport** (`used_transpose` False).
- **This is the reusable bio-plausible learned-recurrent-credit arm.** Its EMERGE-6b verdict: on **autonomous trajectory GENERATION** it BOUNDARIES at *generation stability* (free-run destabilizes) — but the diagnostic `eprop_tf` shows **credit quality is fine teacher-forced (one-step 0.695).** A classification task is teacher-forced throughout → the good regime.

`sim/bptt_snn.py` + `sim/bptt_snn_gpu.py`: full BPTT through stacked LIF with the ATan surrogate. **CRITICAL (grep-confirmed): NEITHER has a recurrent weight matrix `W_rec`** — the "recurrent_dv/ds" (L145-161) is the membrane-leak-through-time + hard-reset chain, NOT a lateral recurrent weight. So a stacked-LIF-BPTT is feedforward-in-space (leaky-integrator state only) and cannot implement an arbitrary finite-state automaton (which group composition requires). **To get a learned recurrent-credit oracle you must ADD a `W_rec`** — cheapest as a ~40-line numpy vanilla RNN (the rate oracle); the spiking version adds `W_rec` to `bptt_snn` (GPU, deferred).

### The FF deep-credit harness (reuse as the FF-CEILING control, NOT the test)

`research/runners/_rolefiller_binding_deep_credit_derisk.py` — a **feedforward** harness (no recurrence):
- `stage0_depth_genuineness` (L316-353): the **self-correcting gate** — trains `DendriticMLP` oracles at 0/1/2/3 hidden layers, measures held-out accuracy + depth-gap + a linear probe; declares DEPTH-SEPARATING only if the deep oracle clears AND the shallow underfits AND the linear probe is at chance. **This exact discipline transfers to D3** (recurrence-separating gate).
- `make_task_xor_pool_positive_control` (L238-258): XOR-over-pool — the FF depth-separating positive control (proves the harness detects depth). **The D3 task must be its TEMPORAL analog.**
- `sim/dendritic_mlp.py` `DendriticMLP`: sigmoid MLP, `oracle` mode (L142-144) = hand-derived backprop (fenced measurement). **Reuse as the FF-ceiling arm** (flatten the sequence → best of 0/1/2/3 hidden → the ceiling that provably caps).

### The existing "multi-hop" (a FIXED pointer-chase, not learned)

`_phaseB_multihop_query_chain_derisk.py`: `query_chain` iterates the validated `query_patient` (match agent+action, read patient, abstain) — a **FIXED, hand-wired** relational pointer-chase over separately-stored facts. NOT learned multi-hop; NOT recurrence-required (it re-cues via a host loop). Its **anti-cheat discipline is reusable**: permuted-relation → chance, lesion (random re-cue between hops) → chance, spreading-baseline floor, moat abstains. (And its own header cites the RETRACTED-2026-05-14 "90% transitive" spreading-activation artifact — the exact trap the D3 anti-cheats must defeat.)

**Net of (a0): the project has the FF ceiling (`DendriticMLP`), the fixed-reservoir baseline (`Reservoir`/`OnBridgeLSM`), and a bio-plausible learned-recurrent-credit arm (`RecurrentMicrocircuitRNN` e-prop) — but has NEVER tested any of them on a bind→re-bind composition with an FF-impossibility control. That test is the gap.**

---

## PART (a) — THE EXTERNAL LITERATURE (the minimal task that PROVABLY separates recurrent from feedforward)

The field has already identified the theoretically-clean primitive, and it maps directly onto "XOR-over-pool but requiring TIME."

### The provable separation: streaming NON-ABELIAN GROUP-WORD composition = STATE TRACKING

- **Merrill, Petty & Sabharwal 2024, "The Illusion of State in State-Space Models," arXiv:2404.08819.** A **simple RNN solves permutation composition (the S5 word problem — "state tracking") with ONE layer**, while transformers AND diagonal/linear SSMs (S4, Mamba) are simulable by **TC⁰** (bounded-depth threshold circuits) and **provably CANNOT** solve it with a fixed number of layers as sequence length grows. The canonical hard case is the **word problem for A₅** (the smallest non-solvable group, |A₅|=60), which is **NC¹-complete** — outside TC⁰ unless TC⁰=NC¹ (believed false). This is the exact "recurrence-required, bounded-depth-provably-cannot" theorem, grounded in **Barrington 1989** (bounded-width branching programs / NC¹ / S₅) and **Merrill & Sabharwal 2023** (log-precision transformers ⊆ TC⁰).
  - **The task, precisely:** a sequence of group elements `g_1 g_2 … g_K` arrives one per timestep; the target is the running product `s_K = g_1·g_2·…·g_K` (or a property/coset of it). The intermediate state `s_t = s_{t-1}·g_t` must be **held and re-composed** over time = the literal "bind→re-bind." Non-abelian ⇒ **order matters ⇒ the count/multiset shortcut is provably at chance**. Non-solvable (A₅/S₅) ⇒ **provably outside bounded-depth (FF/transformer) reach**.
- **"A Held-Out Transition-Pair Falsifier for Long-Horizon Non-Abelian State Tracking," arXiv:2606.07254 (2026).** Directly gives the **anti-cheat**: hold out specific transition pairs / compositions the model NEVER sees in training; a model that memorizes / uses a lookup table / surface-matches FAILS the held-out compositions, while genuine composition-from-primitives generalizes. "No test composition should be directly observable in training data." Uses S₅/A₅. **This IS the systematic-generalization / held-out control the D3 de-risk needs, published for exactly this task.**
- The broader "state-tracking" wave confirms the separation is a live, load-bearing distinction (DeltaProduct 2502.10297; "Illusion of State" follow-ups 2605.07755, 2602.14814) — recurrence (or Householder-product state) is what lets a model track composed state; bounded-depth parallel architectures cannot.

### Why a FIXED reservoir will likely BOUNDARY (motivating a LEARNED recurrent path)

- **Dambre et al. 2012 (memory–nonlinearity tradeoff) + "Reservoir Computing Beyond Memory-Nonlinearity Trade-off," Sci Rep 2017.** A fixed reservoir + linear read-out **cannot simultaneously have high memory AND high nonlinearity**; **parity / delayed-XOR are the canonical HARD benchmarks** precisely because they need both, sustained over time. Group-word composition is a nonlinear state update that must persist exactly over the whole sequence → predicted reservoir BOUNDARY (consistent with EMERGE-84's d\*=2 fading-memory limit). **This is a PREDICTION to RUN, not assume** — but it means the fixed-reservoir arm cleanly tests "is fixed dynamics enough, or is LEARNED recurrence needed."

### The bio-plausible learned recurrent-credit rule

- **Bellec et al. 2020, "A solution to the learning dilemma for recurrent networks of spiking neurons," Nat Commun (e-prop; PMC7367848).** Local, online three-factor rule (eligibility trace × learning signal) that approximates BPTT in a recurrent SNN, no weight transport. Demonstrated on evidence-accumulation / store-recall / speech — **but NOT on compositional state tracking.** So "does e-prop learn non-abelian composition?" is a genuine, un-answered question — exactly the master-directive-relevant arm. The project already has a numpy e-prop (`RecurrentMicrocircuitRNN`).

### Why not SCAN / seq2seq compositional generalization

- **Lake & Baroni 2018 (SCAN)** is the famous compositional-generalization benchmark, but seq2seq RNNs/Transformers *also* fail its hard splits — so it is NOT a clean "recurrent wins" separation (everybody struggles; the failure is confounded with decoder/search issues). The **group-word / state-tracking task is strictly cleaner**: it is a *theorem* that recurrence can and bounded-depth cannot, and the anti-cheat (held-out transition pairs) is published. **Use the group task, not SCAN.**

---

## PART (c) — THE CHEAP-FIRST DE-RISK DESIGN (the controller builds + runs this)

### THE TASK — streaming running-composition ("bind → re-bind → read"), pool-noise encoded

A **new, small** task builder (~60 lines) that is the temporal analog of `make_task_xor_pool_positive_control`:

- **Group `G`.** Cheap-first: **S₃** (|G|=6, non-abelian, solvable) for the fast smoke; escalate to **A₅** (|G|=60, non-solvable, NC¹-complete) for the theoretically-airtight FF-impossibility. (A cheap intermediate: A₄, |G|=12.)
- **Encoding (defeats the linear/lookup shortcut, reusing XOR-over-pool's trick).** Each group element `g` → a **noisy pool code**: a fixed `+-1` prototype per element, corrupted per-observation (`base + noise·randn(pool)`), exactly `make_task_xor_pool_positive_control:238-258`. So a single dim is uninformative; the element identity is only recoverable nonlinearly. A fresh code per timestep prevents per-position memorization.
- **Sequence.** `g_1 … g_K` presented ONE element per timestep. State `s_0 = identity`; `s_t = s_{t-1}·g_t` via the multiplication table. **Target `y` = the final state `s_K`** (|G|-way) OR **a 2-way property** (is `s_K ∈ H` for a fixed subset `H`, the Barrington read-out — cheaper, keeps full composition). Cheap-first: the 2-way property.
- **Splits (the recurrence-AND-composition requirement):**
  - **LENGTH-generalization split** (the FF-ceiling lever): TRAIN on lengths `K ≤ L_train` (e.g. ≤3); TEST on held-out DEEPER `K ∈ {4,5,6}`. A fixed-depth FF net cannot extend its composition depth; a weight-shared recurrent net iterates. **This is the primary separation for small/solvable groups.**
  - **HELD-OUT-COMPOSITION split** (per arXiv:2606.07254): hold out specific `(prefix-state, g)` transition pairs / specific length-K products; test inference-by-composition. The Fodor-Pylyshyn / falsifier control.

### THE MECHANISM RANKING (cheapest-first, reuse-by-import)

| rank | arm | reuse | role | prediction |
|---|---|---|---|---|
| **1 (RUN FIRST)** | **Fixed reservoir + trained ridge** | `_emerge78.Reservoir` + `np.linalg.solve` (drop-in) | the cheapest "recurrence"; does fixed dynamics + a linear read already solve it? | likely BOUNDARIES on deeper K (Dambre memory-nonlinearity + EMERGE-84 d\*=2) — but MUST be run |
| **2 (the ORACLE)** | **BPTT vanilla-RNN** (~40 lines NEW numpy, W_rec + tanh + BPTT, or reuse the ATan-surrogate BPTT machinery of `bptt_snn.py` with a W_rec added) | the recurrent analog of `DendriticMLP.oracle` | **the decisive "is it learnable-with-recurrence at all"** — length-generalizes + held-out compositions | if it clears where FF+reservoir fail ⇒ **D3 is REAL and winnable** |
| **3 (BIO-PLAUSIBLE)** | **e-prop local recurrent credit** | `_emerge6.RecurrentMicrocircuitRNN(elig="eprop")` (adapt the trajectory-recall loop to an end-of-sequence classification read-out) | does LOCAL, no-weight-transport recurrent credit match the BPTT oracle? (teacher-forced throughout ⇒ the regime EMERGE-6b showed works) | the master-directive surpass question |
| **4 (SPIKING, deferred)** | on-bridge LSM (fixed) + BPTT-SNN/e-prop-on-substrate (learned) | `_emerge82.OnBridgeLSM`; add W_rec to `sim/bptt_snn` | the spiking realization | GATED behind a rate GO; GPU/decisive-run |

### THE ANTI-CHEATS (all mandatory; mirror the deep-credit Stage-0 gate + EMERGE-84 + the falsifier)

- **A. FF-CEILING control (the "provably caps" arm).** `DendriticMLP` (best of 0/1/2/3 hidden) on the **flattened** sequence, trained on `K ≤ L_train`, tested on held-out deeper `K`. Must CAP / fail length-generalization. **Positive control (harness-validity):** the SAME elements presented ALL-AT-ONCE (single timestep, flattened) → the FF-deep net CAN compute the fixed-length product → proves the FF arm is not blind and isolates the separation as specifically TEMPORAL (exactly the XOR-over-pool positive-control move).
- **B. COUNT / MULTISET baseline → chance.** Predict from the bag of elements (ignoring order). Non-abelian ⇒ provably at chance. Strengthen with **held-out sequences that are permutations of training sequences** (same multiset, different order → different product), EMERGE-84's exact discipline.
- **C. 1-HOP / MARKOV floor → fails.** Predict `s_K` from only the last token (or last-2 window). Fails (state depends on full history) = the "no fixed window" control.
- **D. HELD-OUT COMPOSITIONS (systematic generalization).** arXiv:2606.07254: compositions/transition-pairs never seen in training must generalize (recurrent) — not memorize (FF/lookup).
- **E. LESION (recurrence off → collapse).** Set `W_rec = 0` (or feed only the current token to the read-out) → collapses to the Markov floor. Proves recurrence is load-bearing.
- **F. PERMUTED-LABEL → chance.**
- **G. STAGE-0 LEARNABILITY-ORACLE gate (measured FIRST, self-correcting).** Before reading arm 3, require RECURRENCE-SEPARATING = (BPTT-RNN held-out-deeper ≥ 0.80) AND (FF-deep-oracle ≤ chance+margin on held-out-deeper) AND (count/Markov/linear-probe at chance). If NOT separating → honest boundary (redesign: bigger group / longer L / property read-out); do NOT read the bio arm on a non-separating task (the deep-credit harness's proven discipline).
- **H. Multi-seed 42/43/44** (cheap smoke), 6-seed for any GO claim.

### CHEAP-FIRST STAGING

1. **NUMPY, CPU, S₃, 2-way property, L_train≤3 → test K∈{4,5,6}.** ~half a day to build (task ~60 lines + reservoir/FF reuse + ~40-line BPTT-RNN), MINUTES to run. Decides the whole question.
2. If GO: **escalate to A₅** (non-solvable — the airtight FF-impossibility theorem) + the **e-prop arm** (bio-plausible).
3. If e-prop GO at rate: **spiking port** (BPTT-SNN with W_rec, or e-prop-on-substrate) — GPU, behind the rate GO. NO `sim/` edit until then.

---

## PART VERDICT — is D3 cheaply de-riskable now?

**YES — cheaply de-riskable now, by reuse + ONE small new piece; NO `sim/` edit for the rate de-risk.**

- **Reuse:** the FF-ceiling arm (`DendriticMLP`, its 0/1/2/3-hidden Stage-0 gate + positive-control pattern); the fixed-reservoir arm (`_emerge78.Reservoir` + ridge); the bio-plausible learned-recurrent arm (`_emerge6.RecurrentMicrocircuitRNN` e-prop); the anti-cheat discipline (EMERGE-84 multiset-swap, `_phaseB_multihop` permuted/lesion/moat, the deep-credit Stage-0 self-correcting gate).
- **New (small):** the **task builder** (~60 lines: group table + pool-noise encoding + length/held-out splits) and a **~40-line numpy vanilla-RNN-BPTT oracle** (the one genuinely-missing machinery — because both `bptt_snn.py` and `bptt_snn_gpu.py` lack a recurrent weight `W_rec`, grep-confirmed).

**Ranked cheapest-first path:** (1) fixed reservoir + ridge [reuse, run first — is fixed dynamics enough?] → (2) BPTT-RNN oracle [~40 lines — is it learnable-with-recurrence at all? the decisive arm] → (3) e-prop [reuse — does bio-plausible local recurrent credit match?] → (4) spiking port [GPU, gated behind rate GO].

### THE SINGLE CHEAPEST DECISIVE EXPERIMENT

> **On S₃ streaming running-composition (2-way property read-out, pool-noise encoding), train on length ≤3, test held-out length {4,5,6}, run THREE arms + anti-cheats B/C/E/F, seeds 42/43/44:**
> **(a) FF-deep-oracle** (`DendriticMLP`, best of 0/1/2/3 hidden, flattened) — the ceiling that should CAP; **(b) fixed reservoir + ridge** (`_emerge78.Reservoir`) — the cheapest recurrence; **(c) BPTT vanilla-RNN** (~40 lines) — the learned-recurrent oracle.
>
> **The decisive number: held-out-deeper accuracy of (c) BPTT-RNN vs (a) FF-deep-oracle.** If (c) ≫ (a) with count/Markov/permuted at chance and lesion collapsing → **a recurrent credit path learns what feedforward provably cannot; D3 is real + winnable → escalate to A₅ + e-prop.** Whether (b) the fixed reservoir already does it tells you if you even need LEARNED recurrence (predicted NO, per Dambre + EMERGE-84 — but run it). If (c) also caps → honest boundary (task not recurrence-separating at this size → escalate group/length before reading the bio arm).

Cost: ~half a day build, minutes compute, CPU/numpy, NO `sim/` edit. It answers "is D3 real, is it winnable, and is learned recurrence necessary" in one shot — the analog of the deep-credit harness's Stage-0 smoke that saved the whole FF arc from reading credit arms on shortcut-able tasks.

---

## LOAD-BEARING CLAIMS THE CONTROLLER MUST VERIFY ITSELF

1. **FF-impossibility is a THEOREM only for NON-SOLVABLE groups (A₅/S₅ — TC⁰⊊NC¹, Merrill 2404.08819 + Barrington 1989).** For small SOLVABLE groups (S₃/A₄) a bounded-depth circuit CAN compute the product at any length in principle — so the S₃ cheap-smoke's FF-ceiling is **empirical-via-length-generalization** (a fixed-depth net can't extend / can't ingest longer), NOT a theorem. Frame the S₃ result as length-generalization-empirical; use **A₅ for the airtight "provably cannot" claim.** Do not overclaim the theorem on S₃.
2. **The reservoir BOUNDARY is a PREDICTION (Dambre memory-nonlinearity + EMERGE-84 d\*=2), not a certainty — RUN arm (b), don't assume it.** The reservoir might solve short compositions; the point is the empirical arm.
3. **The "e-prop sidesteps the EMERGE-6b wall" reframe is a reasoning claim — sanity-check it.** EMERGE-6b's wall was *autonomous generation stability* (free-run). A classification read-out at end-of-sequence is teacher-forced-throughout / input-driven with NO autonomous rollout → the generation-stability wall does not apply, and it is the regime where `eprop_tf` already learned the map (0.695). Confirm the D3 task has no free-run component.
4. **`RecurrentMicrocircuitRNN` is built for trajectory RECALL, not classification — the adaptation (end-of-sequence read-out on the final recurrent state, cross-entropy) is modest but real.** Verify it before claiming reuse-by-import for arm 3.
5. **`bptt_snn.py`/`bptt_snn_gpu.py` have NO recurrent weight `W_rec`** (grep-confirmed; their "recurrent" is membrane-leak-through-time) → the learned-recurrent oracle genuinely needs a new W_rec (numpy RNN for rate; add to bptt_snn for spiking). Confirm before scoping the spiking arm.
6. **The prior 2026-07-08 gate's "consolidate, don't invent / do not re-run recurrent credit" is over-broad** (conflates generation with classification; its three pillars are unsupervised-HTM / fixed-reservoir / hand-designed-buffer, none a learned recurrent-credit path on an FF-impossible task). Re-read PART (a0) + the headline table and judge for yourself before deferring D3.

---

## Files read (with the load-bearing lines)

`research/runners/_emerge78_reservoir_form_to_role_derisk.py` (`Reservoir` L155-170 FIXED weights, ridge read-out L197-204) · `_emerge79_reservoir_variable_distance_derisk.py` (bounded graded memory GO) · `_emerge82_onbridge_lsm_derisk.py` (`OnBridgeLSM` spiking reservoir region) · `_emerge84_reservoir_stack_recursion_derisk.py` (d\*=2 fading-memory limit; `_count_multiset_baseline_acc` L133-145; swap L92-100) · `_emerge85_wm_buffer_recursion_derisk.py` (`WMBuffer` HAND-DESIGNED L69-94) · `_emerge6_recurrent_microcircuit_seq_derisk.py` (`RecurrentMicrocircuitRNN` L90-174, e-prop elig L148-151, W_rec L109) · `_rolefiller_binding_deep_credit_derisk.py` (FF harness; Stage-0 gate L316-353; XOR-over-pool positive control L238-258) · `sim/dendritic_mlp.py` (`DendriticMLP`, oracle mode L142-144) · `sim/bptt_snn.py` + `sim/bptt_snn_gpu.py` (stacked LIF BPTT, NO W_rec) · `_phaseB_multihop_query_chain_derisk.py` (FIXED pointer-chase; anti-cheat discipline) · `research/findings/2026-07-08-deep-credit-feedforward-arc-COMPLETE-*.md` · `2026-07-08-recurrent-multihop-composition-frontier-research-gate.md` (the prior gate this corrects) · `2026-07-08-imaginative-recombination-frontier-research-gate.md` · `2026-07-02-emerge6b-rung3a-eprop-*.md`.

## External citations (controller verifies the load-bearing ones)

Merrill, Petty & Sabharwal 2024, "The Illusion of State in State-Space Models," arXiv:2404.08819 (RNN solves S5 state tracking 1 layer; transformers+diagonal SSMs are TC⁰ and provably cannot; A5 NC¹-complete) · "A Held-Out Transition-Pair Falsifier for Long-Horizon Non-Abelian State Tracking," arXiv:2606.07254 (2026) (the held-out-composition anti-cheat, S5/A5) · Barrington 1989 (bounded-width branching programs, NC¹, S5) · Merrill & Sabharwal 2023 (log-precision transformers ⊆ TC⁰) · Bellec et al. 2020, "A solution to the learning dilemma for recurrent networks of spiking neurons," Nat Commun (e-prop; PMC7367848) · Dambre et al. 2012, "Information processing capacity of dynamical systems," Sci Rep; "Reservoir Computing Beyond Memory-Nonlinearity Trade-off," Sci Rep 2017 (memory-nonlinearity tradeoff; parity/delayed-XOR the hard RC benchmarks) · Lake & Baroni 2018 (SCAN — why NOT to use it: seq2seq also fails, confounded).
