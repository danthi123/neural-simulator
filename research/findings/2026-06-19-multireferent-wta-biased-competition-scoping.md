# Multi-referent disambiguation via WTA biased-competition — deep-research + scoping (2026-06-19)

> **READ-ONLY deep-research + scoping doc. No code edited, no jobs run; this doc is the only write.** Produced per
> the standing "deep research + catalog review FIRST at a new direction" directive (CLAUDE.md;
> `feedback_deep_research_at_roadblocks`). This scopes the owner-approved **conversational #2 lever**: when several
> discourse referents are held in working memory, which one does a bare pronoun ("it") bind to? Conversational #1
> (consolidate attributed-1-attr + multi-frame into the production agent) landed GO; this is the ranked #2 from
> `2026-06-19-conversational-scaling-next-lever-scoping.md` §3. Every load-bearing project fact below was verified
> against the repo (file/finding cited); the two decision-flipping NEGATIVEs and the WTA machinery were read in
> full, not trusted from a summary.

---

## 0. The one-paragraph answer

Multi-referent disambiguation fails on the plain spiking working-memory loop because the loop holds each referent
in an **independent** attractor with **no cross-referent coupling** — so neither recency (the read carries no
position signal) nor a uniform salience boost (it only *adds* activity to one independent attractor; it cannot
*suppress* a competitor) can pick the right one (the two documented converging NEGATIVEs). The fix is **winner-
take-all biased competition** (Desimone & Duncan 1995; Wong & Wang 2006): install **mutual inhibition between the
held referent attractors** (each referent's assembly recruits a dedicated inhibitory interneuron that suppresses
the *other* referents' assemblies — the validated `sel_X`/`sel_FS_X` Rutishauser selective-inhibition motif from
the navigation read-out, reused by import), and feed it a **content-based top-down BIAS** — a feed-forward
excitatory current into the referent whose **features are compatible with the pronoun + the query** (animacy/number
agreement: "it" → inanimate; plus the verb's selectional restriction: "what does it *eat*?" → the edible
candidate). The competitive layer then amplifies that small biased-input difference into a clean single winner that
**suppresses** the alternatives. The crux — and the thing that breaks the symmetry the two NEGATIVEs hit — is that
the BIAS is a **content signal** (feature/role compatibility), not a **position signal** (recency) or a
**magnitude signal** (a uniform gain boost): recency and a uniform boost are exactly the two signals already proven
not to work, *because the loop has no competition to convert any signal into suppression.* The cheapest-first
de-risk is a CPU/numpy probe on the existing `SpikingLoopContextBuffer` with two referents of opposing features,
asserting that the biased+competitive read picks the favored referent at ≥2 referents (multi-seed) where the
recency-only and salience-boost-only baselines (the two NEGATIVE harnesses, re-run verbatim) fail, that **lesioning
the bias → the WTA picks at chance/the wrong one** (proving the bias is load-bearing, not a relabelled boost), and
that the no-confab moat holds (empty/tied WM → abstain). **Reuse-by-import; no `sim/` edit anticipated** (the WTA
motif and the inhibitory pathway installation already exist).

---

## 1. THE MECHANISM MAP — why WTA biased-competition succeeds where recency + salience failed

### 1.1 Why the plain loop fails (root cause, verified in code)

The held referents live in `SpikingLoopContextBuffer` (`research/runners/content_selection_spiking.py`). Its design,
read in full, is the root cause of both NEGATIVEs:

- Each concept gets a **distinct, near-orthogonal** assembly (a random subset of `pattern_size` indices in the
  `cortex_ctx ↔ dlpfc_wm` loop), and its attractor is installed as **only** the within-concept outer-product loop
  weights (`set_pathway_weights("c2d"/"d2c", attractor_weight=50)`). The constructor deliberately sets
  `loop_weight=0` and `internal_density=0` *"so the installed concept attractors are the ONLY loop connections (no
  generic random reverberation to bleed driven patterns into undriven ones)"* and `enable_ou=False` for a quiet
  hold. **There is, by construction, NO synaptic path between one referent's assembly and another's.** The
  attractors are independent.
- Consequence (exactly what the NEGATIVE finding observed): *"whichever concept has the stronger **intrinsic**
  attractor (seed-dependent random pattern) wins regardless of recency OR drive ... a boost only adds activity; it
  does not suppress the competitor"* (`2026-06-17-multireferent-disambiguation-NEGATIVE.md`). The read
  (`buffer.read()`) is just the per-assembly firing sum; with independent attractors it reflects intrinsic
  attractor strength + total injected drive, not *which referent the discourse foregrounds*.

So the two failed mechanisms failed for two distinct, diagnosable reasons:

| Failed mechanism | What it supplied | Why it could not work |
|---|---|---|
| **Recency** (`_phaseB_multireferent_disambiguation_derisk.py`) | a *position* signal (write A then B) | the rate-attractor read carries no order; order-control did not flip the winner → no recency gradient exists in this representation |
| **Salience boost** (`_phaseB_salience_pointer_derisk.py`, up to 4×) | a *magnitude* signal (drive one harder) | a uniform gain boost on an **independent** attractor only adds its own activity; with no mutual inhibition it cannot pull the competitor down → the intrinsically-stronger attractor still wins |

The common missing ingredient is **competition itself** (a mechanism that turns *any* asymmetry into *suppression*
of the loser), plus a **content** asymmetry to feed it.

### 1.2 The fix: biased competition (Desimone-Duncan) realized as a Wong-Wang attractor WTA

**Desimone & Duncan 1995 (biased competition, the canonical attention model):** multiple stimulus representations
**compete** by **mutually suppressing** one another; a **top-down bias** based on the observer's goals/features tips
the competition so the task-relevant representation wins and *suppresses* the distractors (single-unit V4/IT
evidence: a neuron's response to two stimuli is intermediate — competition — and rises toward the attended
stimulus's level when attention is directed to it — biased competition resolving). The two ingredients are
**mutual inhibition** + a **top-down bias**.

**Wong & Wang 2006 (the spiking/reduced realization the project already uses):** NMDA-slow recurrent excitation +
feedback inhibition produce an **attractor WTA** that **amplifies the difference between conflicting inputs** into a
binary choice; a *small* input difference (the bias) is amplified by the recurrence into a decisive winner. This is
the exact two-pool competition the navigation read-out already deploys (`sel_X` accumulators + `sel_FS_X` selective
inhibition; see §2).

**Mapped onto referents in the WM loop:**

1. **The competitors** = the held referent assemblies in `SpikingLoopContextBuffer` (one assembly per referent;
   already there).
2. **The mutual inhibition (the NEW wiring)** = for each held referent X, a small **inhibitory interneuron pool**
   `ref_FS_X` driven **only** by X's assembly and projecting **only** onto the *other* referents' assemblies
   (`ref_Y ≠ X`). This is the **Rutishauser-Douglas-Slotine 2011 selective inhibition** the codebase already uses
   (NOT a symmetric blanket): *"sel_FS_X is driven only by sel_X and inhibits only sel_Y!=X"* (g11_bg_runner.py).
   Catalog substrate: **N.19** gamma binding-by-synchrony (FS-interneuron GABAa mutual inhibition co-groups
   one assembly per gamma cycle and segregates the others) + the **B-cluster MSN lateral inhibition** (catalog
   §"competitive selection within striatum … winner-take-all dynamics"). When X's assembly is even slightly ahead,
   `ref_FS_X` fires more → suppresses the others → the recurrence amplifies the gap → X wins and the losers go
   *quiet* (the suppression a boost alone could never produce).
3. **The top-down BIAS (the crux — a CONTENT signal)** = a feed-forward excitatory current injected into the
   referent whose **features are compatible** with the pronoun + the query, supplied by the parser/query, not by
   position or magnitude. Concretely, two content sources, both available from the existing parse:
   - **Agreement / feature compatibility of the pronoun.** A bare pronoun carries features (animacy, number,
     gender) that **filter** candidate antecedents — the psycholinguistic finding that "agreement features like
     gender and number actively filter candidate antecedents during real-time processing" (Frontiers 2014;
     coreference reviews). "it" → inanimate (excludes animate referents); "they"/"them" → plural; "she" →
     feminine-animate. In the project's small worlds this is a per-concept feature tag (animate {dog, cat, bird,
     fox} vs inanimate {ball, apple, river}); the bias current is added to referents whose tag matches the
     pronoun's.
   - **Selectional restriction of the query verb** (the structural bias). "what does it **eat**?" prefers a referent
     that is a plausible *eater* (animate); "where is **it**?" is neutral; a query whose action has a strong
     argument preference biases toward the compatible referent. This is a content-compatibility score between the
     held referent and the query frame, fed as a small bias current.

   Both are **content**, not position/magnitude — which is precisely **why this breaks the recency/salience
   symmetry**: the bias distinguishes referents by *what they are* and *what the pronoun/query selects for*, and the
   competition turns that small content asymmetry into a suppressive win. (Recency can *augment* the bias as a
   tie-breaker — and `MultiTurnAgentV2`'s order-encoded buffer already supplies order — but recency is **not** the
   load-bearing signal; the NEGATIVE proved a pure recency/order read does not resolve the binding.)

**The one-line "why it works":** the loop already holds the candidates but lacks both *competition* and a *content*
signal; biased competition adds **mutual inhibition** (turns asymmetry into suppression — what a boost lacked) and a
**feature/role-compatibility bias** (a content asymmetry — what recency lacked), and the Wong-Wang recurrence
amplifies the biased difference into a clean, suppressive winner.

---

## 2. REUSE-vs-NEW — what to import, the minimal new wiring

### 2.1 Reusable (by import — already validated)

| Component | Where | Role in the de-risk |
|---|---|---|
| **The held-referent buffer** | `SpikingLoopContextBuffer` (`content_selection_spiking.py`) — independent concept attractors via `set_pathway_weights("c2d"/"d2c", …)` | The competitors live here. **The de-risk subclasses / wraps it** to add the inhibitory cross-wiring; the buffer's `update`/`read` are reused verbatim. |
| **The order-encoded buffer (alt. holding substrate)** | `OrderedPositionWM` inside `MultiTurnAgentV2` (`multi_turn_agent_v2.py`) — a gamma-slot WM that already gives order; its docstring states the set-buffer "has no order, whose winner is fixed by intrinsic attractor strength" | Optional substrate to compose the WTA on top of (order as a *tie-break* feature, never the primary bias). Not required for the cheap-first probe. |
| **The selective-inhibition WTA motif** | `sel_X` (NMDA-slow Wong-Wang accumulator) + `sel_FS_X` (Rutishauser selective inhibition: driven only by sel_X, inhibits only sel_Y≠X) in `g11_bg_runner.py` (build_…, ~lines 2094-2186) | The **exact pattern** to replicate for referents: per-referent inhibitory pool driven only by its own assembly, inhibiting only the others. Reuse the wiring recipe (region + pathway construction), retargeted onto referent assemblies. |
| **The Rutishauser α<1 stability constraint** | documented in CLAUDE.md (line 667: *"violated the Rutishauser α>1 WTA-stability condition"*) + g11_bg comments (soft-WTA gain α<1) | The design guard: keep recurrent self-excitation gain **below** the self-ignition threshold so a referent ramps/holds **only under bias**, never self-ignites — the failure mode of the hand-WTA 0.13 result. |
| **The composer + its moat** | `RFPhasorComposer.query_patient`/`query_agent` → `return None` on no-match (the `is None` abstention, verified) | The downstream answer + the no-confab moat the probe must keep intact (empty/tied WM → no resolved referent → composer not called → abstain). |
| **The multi-turn anaphora harness** | `_phaseB_multiturn_anaphora_derisk.py` (resolve_referent + dialogue + reset/lesion/empty controls) | The **test scaffold** to extend: same `resolve_referent` read, same control structure, now with ≥2 held referents + the bias + the WTA. |
| **The two NEGATIVE harnesses (as baselines)** | `_phaseB_multireferent_disambiguation_derisk.py` (recency), `_phaseB_salience_pointer_derisk.py` (boost) | Re-run **verbatim** as the recency-only and salience-only baselines that must FAIL on the identical setup (the anti-cheat that proves the WTA+bias is the cause, not the setup). |

### 2.2 New wiring needed (minimal, additive, default-OFF)

1. A `BiasedCompetitionContextBuffer` (a new **runner** class wrapping/subclassing `SpikingLoopContextBuffer`) that,
   per held referent, **(a)** adds a small inhibitory FS pool (`ref_FS_X`) — a `BrainRegion(exc_fraction=0.0, …
   IZH2007_FS_CORTICAL_INTERNEURON)` — and **(b)** wires `ref_X(assembly) → ref_FS_X` (excitatory) +
   `ref_FS_X → ref_Y≠X(assembly)` (inhibitory), via the existing `set_pathway_weights`/region-framework machinery.
   Plus a `bias(referent, current_pA)` method that injects a feed-forward bias current into a referent's assembly
   during the competitive read window (using `cp_external_input_current` at the assembly indices — the same drive
   path `update` already uses).
2. A **host-side, content-only** `bias_for(pronoun, query_action, candidate)` helper that returns the bias magnitude
   from **feature-agreement** (animacy/number tag match) + **selectional restriction** (action-argument
   compatibility). *Provenance note (BRAIN-BASED-ONLY):* in the cheap-first probe this scoring is a host scaffold
   that **selects which assembly receives the bias current** — legitimate as a *teaching scaffold* (the
   innate-reflex-teaches-a-learned-circuit pattern), but flagged for conversion: the production version computes the
   feature-compatibility as a **learned synaptic** map (pronoun-feature population × candidate-feature population →
   bias current), so the bias itself is neural. The probe's job is to validate the *competition mechanism*; the
   bias-as-learned-synapse is the follow-on (analogous to how N5 reward / N1 reflex were host-shaped scaffolds for
   their spiking versions).

**No `sim/` edit anticipated:** the WTA is built entirely from the existing region-framework primitives (regions +
inhibitory pathways + `cp_external_input_current` bias injection), exactly as the navigation `sel_X`/`sel_FS_X` WTA
and the `SpikingSpreadingController`'s cross-assembly synapses already are. If a byte-review is ever needed it would
only be if a new neuromodulator/gate target type were required — not expected here.

---

## 3. THE CHEAPEST-FIRST DE-RISK — config, GO bar, baselines, lesion

**The single question:** *does WTA biased-competition bind a bare pronoun to the correct one of ≥2 held referents,
where recency and a salience boost (the two documented NEGATIVEs) fail?*

### 3.1 Config (CPU/numpy first — `SIM_BACKEND=numpy`, small ~600-neuron buffer, mirrors the existing harnesses)

- **Buffer:** `BiasedCompetitionContextBuffer(CONCEPTS, n=600, pattern_size=40, seed=S, enable_ou=False)` — the
  validated clean-WM config (the same `n`/`pattern_size`/OU-off the two NEGATIVE harnesses + the anaphora GO use, so
  the *only* change is the added competition + bias).
- **Referents with OPPOSING content features** (so the bias has a content handle and the test is not solvable by
  recency): hold **two** referents that differ in animacy, e.g. `{cat (animate), ball (inanimate)}`; the pronoun
  "it" in *"what does it eat?"* biases toward the animate eater (`cat`). Critically, run **both orders** (write cat
  then ball; write ball then cat) so a recency baseline cannot pass by accident, and run the **feature-flipped**
  query (a query whose compatible referent is the *other* one) so the win **tracks the content bias, not a fixed
  concept** (the order-control discipline both NEGATIVE harnesses already use).
- **WTA params (frozen, from the validated motif):** `ref_FS` driven-only/inhibit-others wiring as in `sel_FS_X`;
  recurrent self-excitation gain **α<1** (Rutishauser stability — ramp/hold under bias, never self-ignite); bias
  current a *small* feed-forward injection (start ~1× the per-assembly drive scale, NOT a large multiple — the
  whole point is the competition amplifies a *small* bias, unlike the failed 4× boost).
- **Read:** the existing `resolve_referent(wm)` (top-assembly firing / mean-of-rest > spec_threshold, else None) —
  unchanged, so the moat machinery is identical.
- **Answer + moat:** `RFPhasorComposer.query_patient(resolved, "eat")` → `cat`'s patient (e.g. `fish`); empty/tied
  WM → `resolved is None` → composer not called → abstain.
- **Scale:** the cheap-first gate is **2 referents**; a follow-on (same probe) extends to **3** held referents (a
  third, feature-incompatible distractor must be suppressed) to confirm it is not a 2-referent toy (see §4 risk).

### 3.2 Pre-registered GATE (FROZEN before data; multi-seed per `feedback_6seed_validation`)

- **GO** (the decisive condition — ALL of):
  1. **The bias-favored referent wins** at 2 referents — `resolved == cat` (the animacy/role-compatible one) — in
     **both write-orders** and across the feature-flipped query (so the win tracks the **content bias**, not order,
     not a fixed concept), on **≥5/6 seeds**.
  2. **The recency-only baseline FAILS** on the identical setup — re-run `_phaseB_multireferent_disambiguation_derisk`
     verbatim (no competition, no bias): no reliable order-driven winner (the documented 0/3 natural). This proves
     the setup is genuinely ambiguous *without* the new mechanism.
  3. **The salience-boost-only baseline FAILS** on the identical setup — re-run `_phaseB_salience_pointer_derisk`
     verbatim (a uniform boost, no competition): even 4× does not reliably win (the documented NEGATIVE). This
     isolates that it is the **competition + content bias**, not raw drive.
  4. **Lesion the BIAS → the WTA picks at chance / the wrong referent** — with the mutual inhibition present but the
     bias current zeroed, the winner reverts to the seed-dependent intrinsic attractor (no content control). This is
     the load-bearing control that **proves the bias is genuine, not a relabelled salience boost**: the competition
     alone, unbiased, does *not* resolve the pronoun correctly; it is the *content bias steering the competition*
     that does.
  5. **The no-confab moat is intact** — empty WM (no turn-1 referent) → `resolved is None` → abstain; a **tie**
     (two equally-biased, equally-intrinsic referents) → `resolved is None` → abstain (no confabulated antecedent).
     **0 moat breaches** (a moat breach voids the result, per `feedback_brain_based_only_standard` — the moat is
     traded only deliberately, never as a wiring side effect).
- **BOUNDARY:** the WTA resolves the **2**-referent case but degrades at **3** (the third distractor's intrinsic
  attractor occasionally wins despite incompatibility), or the win is seed-fragile (passes 3-4/6) — a real partial
  that localizes "competition strength vs N referents" as the next tuning sub-problem (raise inhibition gain / bias
  magnitude within the α<1 stability envelope), not a mechanism failure.
- **NEGATIVE:** even with mutual inhibition + a content bias, the intrinsic-attractor asymmetry dominates and the
  favored referent does not reliably win — an honest finding that the **point-neuron rate-attractor** substrate
  cannot host clean biased competition at this scale, re-scoping to whether a *spiking-phase* (gamma-cycle N.19)
  segregation is required rather than rate competition. This itself is the deliverable (it maps the substrate
  boundary).

### 3.3 The two baselines + the lesion, stated as the anti-cheat triad

The result is an artifact unless **all three** hold simultaneously on the identical setup: (i) **recency baseline
fails** (the setup is truly ambiguous without the mechanism), (ii) **salience-boost baseline fails** (drive alone
isn't it), (iii) **bias-lesion picks wrong/at-chance** (the competition *needs the content bias* — the bias is not a
relabelled boost). GO requires the favored referent to win *and* all three negative/lesion conditions to hold.

---

## 4. HONEST RISK + the clear cheap-first GO vs NEGATIVE

**The biggest way it could mislead — a bias that is secretly just a relabelled salience boost.** If the "bias" is a
large feed-forward current, it could win simply by out-driving (the failure mode of the salience NEGATIVE), and a
naive harness would call that GO while the mechanism is no different from the thing already proven not to
generalize. **Guards (mandatory):** (a) keep the bias **small** (~1× drive scale, the magnitude a uniform boost
*already failed* at) so any win must come from the *competition amplifying* a small content asymmetry, not from raw
magnitude; (b) the **bias-lesion control** (§3.2.4) is the decisive separator — *with the mutual inhibition present
but the bias removed*, the WTA must NOT resolve the pronoun correctly (if it does, the inhibition alone is picking a
fixed winner and the "bias" is cosmetic); (c) the **feature-flipped query** must flip the winner (a query selecting
the other referent must make the other referent win) — proving the win tracks **content**, not a fixed concept or
raw drive. Only a result where a *small content bias* steers the competition AND removing it breaks resolution AND
flipping it flips the winner is genuine biased competition.

**Second risk — a 2-referent toy that doesn't scale to real ambiguity.** Two opposing referents is the minimum;
real discourse holds several similar-feature referents (two animate candidates of the same number/gender, where
agreement does *not* disambiguate and only finer role/recency cues do). The cheap-first gate is 2 referents
(decisive for "does the mechanism exist"); the **3-referent extension** (one compatible + two incompatible, then the
harder *two compatible* case) is the in-probe scale check, and the genuinely-hard *all-compatible* case is the
honest follow-on (where agreement is silent and the bias must come from finer cues — recency-as-tie-break, salience,
or discourse role — composed *on top of* the validated competition). Stating this scope up front prevents
overclaiming "multi-referent disambiguation solved" from a 2-opposing-referent GO.

**Third risk — the BIAS provenance (BRAIN-BASED-ONLY).** The cheap-first bias score is a host scaffold that selects
which assembly gets the bias current. This is legitimate as a teaching scaffold but must be flagged (§2.2): the win
is brain-based (spiking competition + suppression), the *content scoring* is host in the probe and is the follow-on
to convert to a learned synaptic feature-compatibility map. An honest NEGATIVE on the *neural-bias* version
(host-bias GO but learned-bias fails) would itself be a deliverable (it maps what the substrate can compute about
feature agreement).

**The clear cheap-first call:** **GO** = the favored referent wins (both orders + feature-flipped), ≥5/6 seeds,
**with** recency-baseline-fail + salience-baseline-fail + bias-lesion-wrong + moat-intact — promote toward a
`MultiTurnAgent(enable_biased_competition=True)` build and the learned-bias follow-on. **BOUNDARY** = 2-referent GO
but 3-referent degrades / seed-fragile — localize competition-strength-vs-N. **NEGATIVE** = intrinsic attractors
dominate even with inhibition + content bias — the honest rate-attractor substrate boundary (re-scope to gamma-cycle
N.19 phase segregation). **Stop criterion:** report the three-state outcome after the CPU probe (+ the 3-referent
in-probe check); do NOT escalate a NEGATIVE into a config search — a clean substrate boundary IS the answer.

---

## 5. SUMMARY (the return)

- **Mechanism map (the crux, the BIAS signal):** the plain loop holds referents as **independent attractors with no
  cross-coupling**, so recency (no position signal in the read) and a salience boost (adds activity but can't
  *suppress* an independent competitor) both fail. The fix is **biased competition** — **mutual inhibition between
  referent assemblies** (the validated `sel_FS_X` Rutishauser selective-inhibition motif, α<1 stable) + a
  **content-based top-down BIAS** (a small feed-forward current into the referent whose **animacy/number agreement
  with the pronoun + selectional compatibility with the query verb** match) — where the Wong-Wang recurrence
  amplifies the *small content* asymmetry into a suppressive winner. The bias is a **content** signal, not the
  **position** (recency) or **magnitude** (boost) signals already proven not to work, which is exactly why it breaks
  the symmetry.
- **Reuse-vs-new:** REUSE by import — `SpikingLoopContextBuffer` (the held referents), the `sel_X`/`sel_FS_X`
  selective-inhibition WTA recipe from `g11_bg_runner.py`, the Rutishauser α<1 stability guard, the composer + its
  `is None` moat, and the `_phaseB_multiturn_anaphora` test scaffold + the two NEGATIVE harnesses as baselines.
  NEW (additive, default-OFF, **no `sim/` edit**) — a `BiasedCompetitionContextBuffer` (per-referent inhibitory FS
  pool driven-only/inhibit-others + a `bias()` current injector) + a host content-bias helper (flagged for
  conversion to a learned synaptic feature-compatibility map).
- **Cheapest-first de-risk + GO bar + baselines + lesion:** CPU/numpy, ~600-neuron buffer, **2 referents of opposing
  features** ({cat-animate, ball-inanimate}), pronoun "it"+query "eat" → favored = cat, in **both write-orders** +
  **feature-flipped** query. **GO** = favored referent wins ≥5/6 seeds AND **recency baseline fails**
  (`_phaseB_multireferent_disambiguation_derisk` verbatim) AND **salience-boost baseline fails**
  (`_phaseB_salience_pointer_derisk` verbatim) AND **bias-lesion → WTA picks at chance/wrong** (the decisive control
  proving genuine content-steered competition, not a relabelled boost) AND **moat intact** (empty/tied → abstain, 0
  breaches). In-probe **3-referent** scale check. **NEGATIVE** = intrinsic attractors dominate even with
  competition + content bias (honest rate-substrate boundary → gamma-cycle N.19 re-scope).
- **Sequencing:** this is conversational #2 (after #1 integration), the most precisely-specified open conversational
  mechanism (two converging NEGATIVEs did the scoping), point-neuron-feasible (lateral inhibition is a core project
  motif), and it extends multi-turn dialogue directly.

---

### Catalog entries cited
**N.19** (gamma binding-by-synchrony, ING/PING — the FS-interneuron GABAa mutual-inhibition substrate that
co-groups one assembly per gamma cycle and segregates the others: the biological substrate for biased competition
between referent attractors), **B-cluster** (MSN lateral inhibition — "competitive selection within striatum …
winner-take-all dynamics", the project's existing WTA precedent), **G.08** (PFC working memory / executive control —
where the held referents + their competition live), **H.24/H.25** (SC omnipause/burst commit — the navigation WTA
that supplies the reusable `sel`/`commit` recipe). Catalog:
`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`.

### Papers cited (links)
- **Desimone & Duncan 1995**, *Annu. Rev. Neurosci.* — biased-competition attention: multiple representations
  compete by mutual suppression; a top-down bias from the observer's goals/features tips the competition to the
  task-relevant one (single-unit V4/IT evidence). The #2 mechanism.
  https://www.cognitivepsychology.com/Biased_Competition_Model
- **Wong & Wang 2006**, *J. Neurosci.* 26(4):1314 — *A Recurrent Network Mechanism of Time Integration in
  Perceptual Decisions*: NMDA-slow recurrent excitation + feedback inhibition → attractor WTA that **amplifies the
  difference between conflicting inputs**; a small biased input is amplified to a decisive winner (the reduced model
  the project's `sel_X` accumulator realizes). https://www.jneurosci.org/content/26/4/1314
- **Rutishauser, Douglas & Slotine 2011**, *Neural Computation* — collective stability of networks of WTA circuits;
  the **α (recurrent gain) stability condition** (soft-WTA α<1 to ramp/hold under bias without self-ignition) the
  codebase already enforces.
- **Pronoun resolution / agreement features** — agreement features (gender, number, animacy) **actively filter
  candidate antecedents during real-time processing** (the content bias): *Immediate sensitivity to structural
  constraints in pronoun resolution*, Frontiers in Psychology 2014.
  https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2014.00630/full ; coreference/agreement
  review: https://arxiv.org/pdf/1910.09329

### Project files / findings reviewed (this pass, file-cited)
- **The two converging NEGATIVEs (the lever's basis):** `research/findings/2026-06-17-multireferent-disambiguation-NEGATIVE.md`
  (recency NEGATIVE 0/3 + salience-boost NEGATIVE; names WTA biased-competition as the fix),
  `research/runners/_phaseB_multireferent_disambiguation_derisk.py` (the recency harness — a baseline),
  `research/runners/_phaseB_salience_pointer_derisk.py` (the boost harness — a baseline).
- **The holding substrate + the GO it extends:** `research/runners/content_selection_spiking.py`
  (`SpikingLoopContextBuffer` — independent attractors, the root cause; `SpikingSpreadingController` — the
  cross-assembly synapse install pattern to reuse), `research/runners/multi_turn_agent.py` (`MultiTurnAgent` +
  `held_referent`/`_resolve` — where the WTA plugs in), `research/runners/multi_turn_agent_v2.py`
  (`MultiTurnAgentV2` order-encoded buffer — the alt. substrate, order-as-tie-break not primary bias),
  `research/runners/_phaseB_multiturn_anaphora_derisk.py` (the test scaffold to extend; `2026-06-17-multiturn-anaphora-derisk-GO.md`).
- **The reusable WTA machinery:** `research/runners/g11_bg_runner.py` (the `sel_X`/`sel_FS_X` Rutishauser selective-
  inhibition Wong-Wang WTA, ~lines 2094-2200; default-on spiking decision `2026-06-19-spiking-decision-default-on-GO.md`),
  `CLAUDE.md` line 667 (the Rutishauser α stability condition).
- **The moat:** `research/runners/rf_phasor_composer.py` (`query_patient`/`query_agent` → `return None` abstention).
- **The #2 ranking + framing:** `research/findings/2026-06-19-conversational-scaling-next-lever-scoping.md` §3 (#2 =
  multi-referent WTA biased-competition). `CLAUDE.md` (conversational sections, N.19/N.15 substrate). Catalog:
  `E:\Documents\Projects\sim-catalog\references\feature-catalog.md`.
