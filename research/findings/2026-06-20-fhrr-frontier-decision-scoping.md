# FHRR composer frontier — DECISION-PREP for the deepest shortcut (#12): close as honest-negative, or pursue the learned-cortical-binder conversion? (2026-06-20)

**Type:** READ-ONLY decision-prep scoping. Not a build, not a full research gate. Single deliverable = this doc.
Stayed on `main`. Every load-bearing claim verified against the actual finding text (read in full, not grepped).
**The item:** #12 on the shortcut burndown (`2026-06-20-shortcut-burndown-inventory.md`), classified DEEP-FRONTIER
(NEGATIVE-so-far) in the boundary ledger (`2026-06-20-boundary-ledger-dendritic-audit.md`, row #17).
**The owner's fork:** **(A)** close it as a documented honest-negative (keep the fixed exact-inverse FHRR algebra +
curated codes, acknowledged as the principled idealization; cost ≈ 0), **OR (B)** pursue the actual conversion to a
LEARNED cortical binder (multi-week-to-months, against the existing NEGATIVEs).

---

## TOP-LINE RECOMMENDATION (a decision FOR THE OWNER, not a commitment)

**Recommend (A) — close #12 as a documented honest-negative — with one cheap, already-specified caveat-probe the owner
may optionally green-light first.** The data says the fixed exact-inverse self-inverse bind is the *right engineering
choice*, not a stopgap: it is the one binding form that recovers a multi-attribute fact from a superposition, and that
property is a **theorem of the algebra** (bundling is additive superposition, which has no exact inverse; only a
*self-inverse multiplicative* bind recovers a bundled filler). A from-scratch learned bind has been tested and is
NEGATIVE *on the decisive axis* (multi-attribute bundling), and the dendritic-multiplication candidate that might have
unlocked it is also NEGATIVE (memorizes, does not generalize). Critically, the **strategic prize a learned binder was
supposed to buy — generalization across similar concepts — has ALREADY been delivered on a DIFFERENT axis** (the PPMI
stream cortex learns the *codes* from conversation and generalizes; the bind itself does not need to be learned for
that). So pursuing (B) would spend months to (at best) match a capability the fixed primitive already has, while not
adding the generalization that is already shipped elsewhere.

**The one honest qualifier:** there IS an identified *non*-from-scratch hybrid (fixed self-inverse role + LEARNED
filler codes) that bundles AND generalizes at the numpy level (0.603–0.639 held-out) and is on-bridge GO for the full
who/what conversational turn (0.969 / 1.000 multi-seed). It is **already the production composer's architecture in
spirit** ("learned codes through a fixed coincidence bind"). It lands ~0.39 below the fixed-algebra ceiling (0.993),
so it is not an upgrade — but if the owner wants to *retire the "curated codes" half* of the idealization (not the
exact-inverse-form half), that hybrid is the de-risked, weeks-scale (not months) path, and one afternoon-scale
caveat-probe (the capacity+cleanup sweep, already prescribed) decides whether it can be lifted to parity cheaply. That
is the ONLY thread under (B) with an identified, positive-signal path; everything else under (B) is NEGATIVE.

---

## WHAT IS DESIGNED-NOT-LEARNED vs WHAT IS ALREADY ON-SUBSTRATE / LEARNABLE

The shortcut #12 has **two separable designed pieces** — and the project record shows they have very different status.
Separating them is the crux of the whole decision.

| piece of the composer bind | status | evidence |
|---|---|---|
| **the bind/bundle/unbind OPERATIONS** (resonate-and-fire phasor neurons + complex synapses) | **ALREADY ON-SUBSTRATE SPIKING** — not a shortcut at all | the production `RFPhasorComposer` / `OneBrainComposer` realize bind/store/unbind through `NeuronModel.RESONATE_AND_FIRE` + complex-synapse matvec (CLAUDE.md "OPPONENCY ESCAPED"); the burndown explicitly excludes the ops, narrowing #12 to the *algebra + codes* |
| **the concept CODES** (what gets bound) | **PARTLY LEARNABLE, and the learned version SHIPS on a different axis** | the PPMI stream cortex learns 320 codes word-by-word from the conversation stream on the real spiking substrate (recall 1.00, moat clean, generalizes held-out 0.86) — `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`, ledger #4. The production flagship can run on these learned codes (`consolidated_320_conversation_demo`). The *curated* codes are the fast/oracle default, not a hard requirement. |
| **SINGLE-attribute role-filler binding** | **LEARNABLE + validated on real LIF spikes** | a learned additive bind generalizes single-attribute bindings: numpy 0.806 → real-LIF on-bridge **0.833 = 100% of numpy** — `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md` |
| **the exact-inverse FORM** of the bind (a clean, exactly-invertible algebra) | **DESIGNED, not learned — and this is the load-bearing residual** | the residual idealization (Eliasmith Spaun / Semantic Pointer). A real cortex would have a lossy, redundant, *learned* read-out. This is the "form" half of #12. |
| **MULTI-attribute BUNDLING** (a fact = superposition of 3 bindings; unbind one role) | **NOT LEARNABLE FROM SCRATCH on point neurons — the decisive NEGATIVE** | additive bind has no inverse (held-out **0.193**); a learned LINEAR inverse cannot be a reciprocal (**0.056**, breaks even single-attr); the FIXED ±1 self-inverse bundles **0.989** on the identical harness — `2026-06-16-...bundling-NEGATIVE.md` (verified verbatim) |

**So the residual that is genuinely "designed, not learned-by-the-brain" is narrow:** the **exact-inverse FORM** of the
bind (and, on the codes axis, the *curated* default, which is optional since the learned PPMI codes exist). The
operations are spiking; the codes are learnable; single-attribute binding is learnable. The one thing that provably
needs a fixed structure is the **multi-attribute bundle inverse**.

---

## THE TWO NEGATIVES, VERIFIED (what EXACTLY failed)

**Both findings read in full; numbers below are verbatim from the source.**

### (1) `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md` — the learned-binder map
The capability map, on the identical systematicity harness (stream codes, F=16, leakage-asserted held-out splits, 3 seeds):

| Bind | single-attribute | 3-way bundle (a fact) |
|---|---|---|
| **Fixed ±1 / FHRR algebra** (self-inverse role) | 1.000 | **0.989** |
| **Learned additive** (point-neuron) | 0.806 → real-LIF **0.833** | **0.193** |
| **Learned multiplicative + learned LINEAR inverse** | 0.083 (broken) | **0.056** (broken) |
| chance (1/F) | 0.062 | 0.062 |

- **What FAILED:** *learning the bind* for **multi-attribute bundling**. Additive bundling fails even on TRAIN combos
  (0.285) → it is superposition CROSSTALK, not a generalization gap. A learned *linear* inverse collapses even
  single-attribute (0.056) — a linear map provably cannot implement the role-dependent reciprocal `1/u_t` a
  superposition-unbind needs.
- **What PASSED:** single-attribute learned binding (0.833 on real spikes = 100% of numpy), AND the fixed ±1
  self-inverse bundles (0.989) — the positive control proving the harness genuinely detects working bundling, so the
  NEGATIVEs are real.
- **The finding's own conclusion (verbatim):** *"This is not a host shortcut — it is a biologically-grounded structural
  primitive. Real neural binding is built on coincidence detection and dendritic multiplication (structural
  mechanisms), not an operation learned from scratch."*

### (2) `2026-06-19-dendritic-binding-toy-derisk.md` — the dendritic-multiplication candidate
The natural "the missing piece is dendritic multiplication" hypothesis, tested cheap-first (the dendrite's other named
job). 3-seed mean, R=4/F=16, leakage-free splits:

| arm | single-attr held | two-attr bundle TRAIN | two-attr bundle HELD-OUT |
|---|---|---|---|
| **dendritic sigma-pi (test)** | 0.500 | 0.422 | **0.168** (train→held gap **+0.254**) |
| lesion = point/additive (must fail) | 0.250 | 0.085 | 0.032 |
| memorization-floor (must ≈ chance) | — | — | 0.000 |
| **FHRR fixed ±1 (production reference / ceiling)** | — | — | **0.261** |

- **What FAILED:** the learned dendritic multiplication **memorizes** (train 0.422) but does **NOT generalize**
  (held-out **0.168**, below the 0.40 GO bar, +0.254 train→held gap = the memorization signature). It is even **WORSE
  than the fixed FHRR primitive it would replace** (0.261) on the same two-attribute held-out test.
- **Controls valid:** the lesion (plateau→identity) collapses to the additive failure (0.032), so the supralinearity
  IS load-bearing for the train *fit*; the memorization-floor scores 0.000 → the held-out metric discriminates.
- **The finding's own conclusion (verbatim):** *"the binding wall is NOT (only) the missing dendritic multiplication …
  Production keeps the fixed ±1 / FHRR primitive."*

**⇒ Both named dendrite jobs (multiplicative binding #17, apical-basal credit assignment #18) are cheap-first
NEGATIVE.** Two afternoons of CPU/numpy tests saved a months-scale build on both premises. The dendrite is "thoroughly
assessed and ruled out for current walls" (ledger; CLAUDE.md commit cc6cfd58).

---

## WHAT'S ALREADY PARTIAL (the positive-signal thread under B)

Crucially, the picture is **not** all-NEGATIVE. There is one identified hybrid path that is GO — but it is decisive to
understand what it does and does NOT buy:

- **Fixed self-inverse role + LEARNED filler codes BUNDLES + generalizes** (`2026-06-17-fixed-role-learned-filler-bundling-derisk.md`,
  6-seed; `2026-06-18-learned-filler-fixed-bind-bundles-GO.md`, CYCLE 196): held-out-combo **0.603** (6-seed) / **0.639**
  (3-seed), single-binding held-out 0.806–0.833, 5/6 seeds ≥ 0.40, beats learned-additive (0.238) and learned-linear
  (0.069) at every seed; permuted-role and lesion both collapse to ~chance; the moat separation holds.
- **On-bridge, conversation-capable** (`2026-06-17-onbridge-learned-filler-binding-step1-GO.md` 0.969;
  `2026-06-17-onbridge-learned-composer-step2-GO.md` 1.000 6-seed who/what + moat) — the learned-filler + fixed-bind
  composer does real store / who-what Q&A / abstention on real LIF spikes.
- **BUT it lands ~0.39 below the fixed-algebra ceiling (0.993)**, and — the load-bearing point — **a fixed self-inverse
  role is EXACTLY what the production composer already does.** A GO here proves the *LEARNED-FILLER* version holds; it
  does NOT prove "a learned bind from scratch works." The bundle inverse is still the fixed ±1 structure.
- **The strategic prize is already delivered elsewhere.** The reason to want a learned binder was *generalization across
  similar/correlated concepts*. That is a **different axis** — carried by the PPMI stream cortex (learns the *codes*) +
  cross-modal-Hebbian generalization (ledger #4; CLAUDE.md generalization arc), already shipped on point neurons. The
  bundling A/B does not test, need, or add that.

**So under (B), the only thread with positive signal retires the *curated-codes* half of the idealization (learn the
fillers), NOT the *exact-inverse-form* half (the bundle inverse stays fixed) — and it does not improve accuracy.**

---

## LITERATURE (brief, confirms the theoretical backbone)

The fixed-primitive result is not an artifact of the harness — it follows from VSA theory:
- **Bundling = additive superposition, by design has no exact inverse.** It is a *similarity-preserving* operation
  (the bundle is similar to each bundled vector); recovering a clean element requires a separate *binding* inverse, not
  an inverse of the sum. (ACM Computing Surveys VSA survey Part I; emergentmind VSA topic.)
- **The self-inverse property is what makes a bundled filler recoverable.** For bipolar/±1 vectors, binding is its own
  inverse: `a⊗a=1`, so `(a⊗b)⊗a=b` (MAP / Multiply-Add-Permute). This is precisely the project's fixed ±1 self-inverse
  bind. A bind WITHOUT an exact (self-)inverse cannot cleanly unbind from a superposition.
- **Learned binding (elementwise product, circular convolution, tensor product) needs larger dimension and has
  "limited generalization."** Active research seeks to combine learned representations with fixed sparse-VSA binding —
  i.e. the literature's own resolution is the project's resolution: *learn the representations, keep a fixed binding
  primitive.* (RNNs-implicitly-implement-TPR arXiv:1812.08718; Variable Binding for SDR arXiv:2009.06734; Frady-Sommer
  resonator networks arXiv:2208.12880.)

This corroborates: the fixed self-inverse bind is the *correct* engineering primitive for invertible bundling, and a
learned-from-scratch bundle inverse is fighting the algebra, not a missing mechanism.

---

## THE FORK, WITH DATA

### Option A — close #12 as a documented honest-negative (RECOMMENDED)
- **What stays:** the fixed exact-inverse FHRR self-inverse bind + the curated (or PPMI-learned) codes, **explicitly
  labelled the principled idealization** (Eliasmith Spaun / Semantic Pointer — a serious cortical hypothesis, not a hack).
- **Cost:** ≈ 0 (already shipped, already the production default; no build).
- **The acknowledged standing limitation (state it plainly):** (i) the bind FORM is designed (exact-inverse), not
  learned by the brain; (ii) multi-attribute bundling is not learnable-from-scratch on point neurons (it requires the
  fixed self-inverse structure — which is a *structural* neural primitive, coincidence-detection / dendritic-product, so
  this is biology-grounded, not arbitrary). The single-attribute bind AND the codes ARE learnable + on-substrate; only
  the bundle inverse's exact form is fixed.
- **Why this is the disciplined call, not a punt:** every from-scratch learned-bundle path is NEGATIVE (additive 0.193,
  learned-linear 0.056, dendritic 0.168 < fixed 0.261); the prize (generalization) is already delivered on the codes
  axis; the fixed primitive's superiority for bundling is a theorem, not a tuning gap. "Closing as an acknowledged
  idealization" is the correct outcome the data supports.

### Option B — pursue the learned-cortical-binder conversion
Three sub-paths, with their actual status:

| sub-path | status | cost | what it would buy |
|---|---|---|---|
| **B1: learned-from-scratch bundle bind** (additive / learned-linear / dendritic-multiplication) | **NEGATIVE** (0.193 / 0.056 / 0.168) | months | nothing — provably can't invert a superposition; worse than the fixed primitive |
| **B2: fixed self-inverse role + LEARNED fillers** (the identified hybrid) | **GO at numpy 0.603, on-bridge who/what 1.000**, but ~0.39 below the 0.993 ceiling | weeks (additive guarded `sim/` wiring, D2-Phase-1-scale — NOT a new NeuronModel) | retires the *curated-codes* half (learn the fillers); keeps the fixed bundle-inverse FORM; **does NOT add generalization** (already shipped) and **does NOT improve accuracy** |
| **B3: D2 Phase 3 / true two-compartment NeuronModel** as a learning-rule (credit-assignment) unlock | **wrong lever** — credit assignment is a different problem from binding; prior sound-instrument VOID; D2 Phase 3's conversational gate already passed on point neurons | months, HIGH variance | a sample-efficiency unlock, **redundant for binding** |

- **Risk of B (overall):** the only positive-signal sub-path (B2) does not improve the capability and does not add the
  prize; B1 is NEGATIVE; B3 is the wrong lever and redundant. The honest read: **B has no identified path to a capability
  the fixed primitive lacks.** The one defensible B-action is B2 *only if the owner specifically wants to retire the
  "curated codes" label* (a brain-based-purity goal, not a capability goal) — and even then, the cheap caveat-probe
  below should gate it.
- **The one cheap caveat-probe that could change B2's verdict** (already prescribed,
  `2026-06-17-fixed-role-learned-filler-bundling-derisk.md` §recommended-next): an afternoon CPU/numpy capacity+cleanup
  sweep on the SAME harness — bind-space dim `D_h` 64→128→256 + a multiplicative (vs nearest-cosine) cleanup — to test
  whether learned-filler bundling lifts 0.603 → ~0.9. **If it reaches parity, B2's weeks-scale on-bridge build becomes
  justified** (route learned fillers through the already-built, guarded `fused_coincidence_plateau` self-inverse
  primitive). **If it plateaus, the fixed FHRR algebra stays load-bearing and the learned frontier is closed for
  bundling.** Either outcome is a clean, citable result, costs an afternoon, and touches no `sim/`.

---

## RECOMMENDATION (restated as the owner's decision)

1. **Default: take (A).** Close #12 as a documented honest-negative. The fixed exact-inverse self-inverse bind is the
   right engineering choice (a theorem-backed primitive, the biology-grounded coincidence/multiplicative structure),
   the learned-from-scratch conversion is NEGATIVE on the decisive axis, and the generalization prize is already
   delivered on a different axis. Update the standing limitation note; cost ≈ 0.
2. **Optional, if the owner wants to retire the *curated-codes* label specifically (a purity, not a capability, goal):**
   green-light ONLY the afternoon capacity+cleanup caveat-probe first. If it lifts learned-filler bundling to ~parity,
   the weeks-scale B2 wiring becomes justified (and is D2-Phase-1-scale, not a NeuronModel rewrite). If not, (A) stands,
   now with the parity question also closed on a measured signal.
3. **Do NOT pursue B1 (from-scratch learned bundle) or B3 (two-compartment NeuronModel for binding)** — both are
   NEGATIVE/wrong-lever for this op against the existing data; months-scale with no identified payoff over the fixed
   primitive.

**Honest bottom line:** for #12, "closing it as an acknowledged idealization" IS the correct outcome the data supports.
A learned bundle-inverse is provably hard (the algebra forbids it) and unnecessary (the prize is elsewhere); the fixed
self-inverse primitive binding learned codes is the disciplined resting point, and it is already what production runs.

---

## Sources (verified against the actual finding text this pass)
- `research/findings/2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md` (read in full — the
  capability map; single-attr GO 0.833; bundling NEGATIVE 0.193/0.056; fixed ±1 0.989)
- `research/findings/2026-06-19-dendritic-binding-toy-derisk.md` (read in full — dendritic multiplication memorizes
  0.422 / doesn't generalize 0.168 < fixed 0.261)
- `research/findings/2026-06-20-boundary-ledger-dendritic-audit.md` (rows #17 binding, #18 credit-assignment, #20 F=3
  resonator on learned codes; the DEND-RULED-OUT classification)
- `research/findings/2026-06-20-shortcut-burndown-inventory.md` (#12 = DEEP-FRONTIER; the ops are excluded, narrowing
  #12 to algebra + codes)
- `research/findings/2026-06-17-fixed-role-learned-filler-bundling-derisk.md` (6-seed hybrid GO 0.603; the caveat-probe
  prescription) + `2026-06-18-learned-filler-fixed-bind-bundles-GO.md` (CYCLE 196, 0.639)
- `research/findings/2026-06-17-onbridge-learned-filler-binding-step1-GO.md` (0.969) +
  `2026-06-17-onbridge-learned-composer-step2-GO.md` (1.000 6-seed who/what + moat)
- `research/findings/2026-06-18-step3-dendritic-learned-bind-frontier-scoping.md` (the OP-A/OP-B split; the generalizing
  cortex ships on point neurons; the B2/B3 framing)
- `research/findings/2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (PPMI stream cortex — learned
  codes ship + generalize, the prize on a different axis)
- Literature: ACM Computing Surveys VSA survey Part I (dl.acm.org/doi/10.1145/3538531); MAP self-inverse `a⊗a=1`;
  RNNs-implicitly-implement-TPR (arXiv:1812.08718); Variable Binding for SDR (arXiv:2009.06734); Frady-Sommer resonator
  networks (arXiv:2208.12880) — confirms bundling has no exact inverse + the self-inverse primitive recovers it +
  learned binding has limited generalization.

_Read-only decision-prep deliverable. No code, no experiments. Stayed on `main`. The fork is presented as a decision
for the owner; the recommendation is (A) with an optional afternoon caveat-probe gating any (B2) move._
