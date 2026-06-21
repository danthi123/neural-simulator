# Can the binding STRUCTURE self-organize developmentally on the spiking substrate? (FHRR-B, CYCLE 344)

**Type:** READ-ONLY deep-research + catalog/literature/code scoping. NO code written, NO experiments, NO GPU. ONE
findings doc. Stayed on `main`. Every load-bearing project claim is file:line- or finding-cited; literature is
paper-cited.

**The owner's challenge that opened this scope (CYCLE 344, commit `8aa75cc1`):** the bind OPERATION runs in spikes
(resonate-and-fire + complex synapses — not a shortcut), but the bind STRUCTURE — the specific complex-synapse
*weights* that make the operation a clean invertible role-filler bind — is **host-computed in Python and injected** via
`SimulationBridge.rf_set_complex_weights` (`sim/bridge.py:5604`). It does NOT self-organize on the substrate. This
breaks two things the owner cares about: (1) a true "spiking end-to-end / self-contained brain that develops its own
structure" claim; and (2) a clean port to specialized neuromorphic hardware (the chip would need a host to compute +
inject the bind weights). The prior 3-mechanism arc (CYCLE 343, commit `8d3a0cd9`) already showed the structure is NOT
*learnable from task data* (it is an exact multiplicative reciprocal). **The untested category this scope addresses:
can the binding connectivity self-organize DEVELOPMENTALLY — from local genetic/activity wiring rules, so the weights
EMERGE on-substrate — rather than being host-injected?**

**"Developmental self-organization," defined for this scope:** the connectivity/weights emerge from *local* rules that
run *on the substrate* — either (a) a genome-style wiring rule (a fixed local prescription executed at construction,
e.g. "draw this synapse's weight from this distribution with this seed") or (b) activity-dependent refinement (Hebbian
/ structural plasticity driven by spontaneous or sensory activity). It is explicitly **NOT** task-learning (gradient on
a loss) and **NOT** host-design (a Python formula computing each weight from an algebraic identity and writing it in).

---

## 0. TOP-LINE (the honest answer, then the path)

**The genuine host residual is far smaller than "the bind structure," and most of it is ALREADY developmentally-cheap
random structure — but ONE relationship is genuinely host-imposed: the conjugate-symmetry tie between the bind weight
and the unbind weight.** Decomposing what `rf_set_complex_weights` actually receives:

| sub-part of the bind STRUCTURE | what it is, in code | developmental status |
|---|---|---|
| the role CODES (`comp.roles[r]`) | `rng.uniform(0,1,D)` per role, deterministic per seed (`rf_phasor_composer.py:132`) | **developmentally-cheap RANDOM** — a genome-style "draw from a distribution" rule; biologically faithful (see §2.1) |
| the concept CODES (the fillers) | LEARNED from conversation (PPMI stream cortex, real spikes, 320) | **already on-substrate LEARNED** (`2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`) |
| the bind OPERATION (diagonal complex synapse, weight = the role phasor) | `_bind`: `conns=[(D+k,k,zr[k])]` (`rf_phasor_composer.py:156-162`) | **on-substrate SPIKING** (the resonate + matvec run on the bridge's own neurons) |
| the **unbind weight** = `conj(role)` | `_unbind_phases`: `zr_conj = np.conj(...)`, `conns=[(D+k,k,zr_conj[k])]` (`rf_phasor_composer.py:204-209`) | **THE GENUINE HOST RESIDUAL** — `conj` is computed host-side and the result injected; the substrate is never told "the unbind synapse is the inverse of the bind synapse" |
| the cleanup codebook (nearest-cosine over concept codes) | `np.conj(exp(2j*pi*concepts[w]))` stacked (`rf_phasor_composer.py:335`) | reducible to LEARNED (Option-1 GO; see §4 and `2026-06-20-FHRR-B-learned-binder-scoping.md` §2) |

**So the residual that is genuinely "host-designed, not self-organized" is NOT the whole structure — it is the single
algebraic relationship `unbind_weight = conj(bind_weight)`** (the exact-inverse FORM), plus the per-instance host
*computation* of the conjugate. Everything else is either already-learned (concept codes), already-spiking (the bind
op itself), or developmentally-cheap random (the role codes — see §2.1, which argues this is biologically faithful, not
a cheat).

**The feasibility verdict (stated up front, defended in §5):** on-substrate developmental self-organization of the
*role-code half* is **cheap and already essentially true** (a random draw IS a developmental rule — the project already
accepts this pattern in `sim/dendritic_neuron.py:25`, "FIXED RANDOM ... set once from seed, NEVER learned"). The
*conjugate-symmetry half* is the real question, and the honest finding is that there are **two genuinely-distinct
outcomes, and a single cheap probe decides which**:
- **(A)** the conjugate is derivable by a *local* rule that runs on the substrate (a per-synapse "install the
  phase-conjugate of my forward partner" wiring rule — biologically a reciprocal/transpose-like connection), in which
  case the structure self-organizes from a local prescription and the host computation is *eliminated*, not just
  relabelled — a real win, weeks-scale; OR
- **(B)** no purely-local rule supplies the conjugate (the inverse is non-local information), in which case
  host-specification of a *fixed developmental structure* is the honest, defensible end-state — and §2 shows WHY that
  is biologically faithful (the genome specifies the retina's center-surround and the cerebellum's random expansion;
  it does not learn them), not a shortcut.

**The hardware-port implication is the decisive lens (§6):** for a neuromorphic chip, the question is not "is it
learned?" but "is the weight pattern a *fixed* prescription the device can be configured with once, or does it require
a host in the loop *per operation*?" The role codes (random, fixed-per-seed) and the conjugate (a fixed function of
them) are BOTH one-time-configurable — they are computed ONCE at construction, not per bind. So even the current
host-design is already hardware-portable in the sense that matters (a one-time config load, like programming a
memristor crossbar), as long as the conjugate relationship can be expressed as a *wiring rule* the configuration step
applies, rather than a runtime host call. That reframes the residual from "breaks the hardware port" to "needs to be
expressed as a one-time wiring rule, not a runtime host op" — a much smaller, achievable target.

---

## 1. THE PRECISE TWO-PART SPLIT (operation = spiking-done; structure = mostly-random + one host relationship)

The owner's operation/structure split is exactly right, and the code confirms it precisely. Reading the production
path (`research/runners/one_brain_composer.py` `_compose_phases`, lines 253-273, and
`research/runners/rf_phasor_composer.py` `_bind`/`_unbind_phases`/`_cleanup`):

**The OPERATION** (multiply / coincidence — biologically dendritic multiplication, the σ-π / NMDA-plateau conjunction):
already runs in spikes. `_bind` installs a *diagonal* complex synapse (`conns=[(D+k,k,zr[k])]`,
`rf_phasor_composer.py:162`) and lets the resonate-and-fire neurons + the complex matvec
`u_i = Σ_j W_ij z_j` (`sim/bridge.py:5608-5609`) perform `filler_phasor × role_phasor = phase sum` ON the bridge's own
neurons. `rf_set_complex_weights` is a *pure writer* — it builds a sparse complex CSR from a `(post, pre, weight)`
connection list and replaces the prior weights (`sim/bridge.py:5604-5614`); it computes nothing. This is genuinely
on-substrate.

**The STRUCTURE** (the weight *values* that make the operation a clean invertible bind): decomposes into the four
sub-parts in §0's table. The crux: a phasor bind is `z_bound[k] = z_filler[k] · z_role[k]` (a per-component complex
product), and the *only* thing that makes it invertible is that the role is **unit-magnitude**, so its inverse is its
**complex conjugate**: `z_filler[k] = z_bound[k] · conj(z_role[k])`. In the code, the bind synapse weight is `zr[k]`
and the unbind synapse weight is `np.conj(...)[k]` (`rf_phasor_composer.py:208`). **Both are derived from the SAME
random role code drawn once at construction (`rf_phasor_composer.py:132`).** The forward (bind) weight is itself
developmentally-cheap (it IS the random role code, installed directly). The genuinely host-imposed thing is the
*relationship*: the unbind synapse must carry the *conjugate* of whatever the bind synapse carries — and that
conjugate is computed by a host `np.conj` call and injected, never derived by a rule the substrate runs.

**This is the same localization the CYCLE-343/344 arc reached, sharpened:** the residual is the **exact-inverse FORM**
(`2026-06-20-FHRR-B-learned-binder-scoping.md` §0, lines 35-42: "the exact-inverse FORM of the multi-attribute bundle
inverse"). What that scope did not separate, and this one does, is that the FORM has two physically-distinct pieces:
(i) the role codes (random — developmentally cheap, §2.1) and (ii) the conjugate-symmetry tie between bind and unbind
(the one genuinely-host relationship, the target of §2-§5).

---

## 2. HOW BIOLOGY DEVELOPS STRUCTURED / BINDING CONNECTIVITY (literature — NOT task-learning)

The decisive literature finding reframes the whole problem: **in VSAs the binding matrix can be RANDOM, and a random
matrix is exactly what a developmental rule produces for free.** From the neurobiologically-plausible-VSA literature
(Eliasmith/Stewart; the Frontiers VSA-for-neural-computation topic; the cs/0412059 "VSAs answer Jackendoff's
challenges" survey): *"matrix multiplication can be used as the binding operator for a VSA, and matrix elements can be
chosen at random. A consequence for living systems is that binding is mathematically possible without the need to
specify, in advance, precise neuron-to-neuron connection properties for large numbers of synapses."* That is the entire
crux: **the binding STRUCTURE does not need to be designed — it can be random — and a random projection is the
cheapest possible developmental product** (a genome that says "wire this layer with random weights from this
distribution," not "learn these specific weights").

### 2.1 The genome specifies RANDOM structure — and biology does this constantly (the role-code half)

The two canonical examples in the catalog are direct precedents for "a developmentally-fixed random expansion is
biologically faithful, not a shortcut":
- **Cerebellar granule expansion (codon recoding), catalog F.12** (`feature-catalog.md:1613`): mossy-fibre →
  granule-cell, a sparse combinatorial *expansion* recoding (Marr-Albus). The granule-layer wiring is a
  developmentally-set, essentially-random divergence onto a much larger population — the brain does NOT learn it; it is
  specified by development and used as a fixed random feature basis.
- **Dentate-gyrus expansion recoding, catalog D.18-class** (`feature-catalog.md:1227-1229`): EC layer II → a much
  larger sparse DG population; "Marr expansion recoding — divergence onto a larger sparse population orthogonalizes
  similar inputs." Again a developmentally-specified random/sparse projection, not a learned one.

Both establish the principle the role codes rely on: **a fixed random projection, set by development, is a legitimate
biological substrate for binding-relevant structure.** The project ALREADY accepts and ships this exact pattern:
`sim/dendritic_neuron.py:25-27` uses a "FIXED RANDOM apical feedback — feedback alignment. Never learned, never read
from W_basal (no weight transport)... set once from seed." That is feedback-alignment biology (Lillicrap 2016;
Guerguiev-Lillicrap-Richards 2017) — a fixed random projection that works *because* it is random and need not be
learned or transported. **By this standard the role codes are NOT a host-design cheat at all** — `rng.uniform(...,
seed)` IS a genome-style developmental wiring rule (draw from a distribution with a developmental seed). The honest
correction to "the bind structure is host-designed" is that the *role-code half* is developmentally-cheap random
structure the project already treats as faithful elsewhere.

### 2.2 How biology refines connectivity *with* activity (the general substrate we already have)

The catalog's Cluster L (Development & critical periods, 23 entries, `feature-catalog.md:4179`) is the relevant body:
- **L.06 Activity-dependent refinement is general (NMDAR-dependent)** (`feature-catalog.md:4233-4241`): "the *common
  substrate* of refinement is NMDAR-dependent Hebbian plasticity: coincident pre/post → strengthening; uncorrelated →
  weakening/pruning... **implemented** (J.08 + STDP)." The simulator's STDP IS the algorithmic content of biological
  developmental refinement.
- **L.05 Spontaneous-activity-driven refinement (retinal waves)** (`feature-catalog.md:4223-4231`): before sensory
  experience, the system generates *patterned* spontaneous activity that refines connections via NMDAR rules; "**the
  wave content matters — random noise wouldn't produce ocular dominance maps; coherent waves do.** The brain is
  *self-organizing* its sensory representations before experience arrives." Catalog status: missing as a mechanism
  (we have OU background noise but not *patterned* spontaneous activity); explicitly flagged as a likely-useful
  "developmental pretraining" lever.
- **L.02 Synapse elimination by activity competition** (`feature-catalog.md:4193-4201`): "use it or lose it" — synapses
  that fire coincidently with the strongest axon are stabilized, weakly-correlated ones eliminated; status **partial**
  (the structural-pruning option). This is the project's structural-plasticity analogue.
- **L.01 Target recognition (cell-adhesion molecular code)** (`feature-catalog.md:4183-4191`): the chemoaffinity /
  genome-specified path — "the wiring of which-axon-finds-which-target is *not* random in real brains" — combinatorial
  cadherin/neurexin-neuroligin/Eph-ephrin codes. Status: missing entirely (no molecular-recognition layer); flagged as
  "probably out of scope unless we want to model developmental wiring errors." **This is the biology that would specify
  a STRUCTURED (non-random) developmental wiring** — e.g. a reciprocal/transpose connection — but the catalog itself
  judges a full recognition-code mechanism out of scope.

### 2.3 The named neural binding substrates (the operation, for completeness)

For the *operation* (already on-substrate), the candidate binding mechanisms in the literature are: von der Malsburg
binding-by-synchrony (gamma; catalog N.19, `feature-catalog.md:1028`); Mel/Larkum dendritic multiplication
(the σ-π conjunction — the project's dendritic substrate); conjunctive / mixed-selectivity coding; and the
Eliasmith/Plate VSA-via-structured-connectivity (circular convolution / element-wise product). The FHRR phasor product
the composer uses is the VSA route, and its operation is already spiking. None of these change the structure question —
they are all "how to multiply," and the multiply is done.

---

## 3. THE KEY SUB-QUESTION — is the host-design a real cheat, or a faithful developmental stand-in?

The task asks to isolate WHICH part is genuinely non-developmental host-design and quantify it. The answer, from §1-§2:

| part | host-design? | verdict |
|---|---|---|
| role CODES (random draw) | NO | a genome-style random wiring rule; biologically faithful (F.12, D.18, `dendritic_neuron.py:25`); the project already ships this pattern |
| concept CODES (fillers) | NO | already learned on-substrate from conversation |
| bind OPERATION | NO | already spiking (resonate + complex matvec on the bridge's neurons) |
| cleanup codebook | reducible | Option-1 learned-iterative-cleanup is GO-able (§4); not the core residual |
| **the conjugate-symmetry tie `unbind = conj(bind)`** | **YES** | the single genuinely-host relationship: a `np.conj` computed host-side and injected; the substrate is never given a rule that derives the unbind synapse from the bind synapse |

**Quantification of the residual:** of the `2 · D` synapses per role that realize bind+unbind, the `D` *bind* synapses
are developmentally-cheap (they are the random role code, installed directly). The `D` *unbind* synapses are host-
computed (`conj` of the bind synapses) and injected. So the genuine host residual is **D conjugate weights per role
relationship** — and they are not free parameters; they are a *deterministic function* (`conj`) of weights that are
themselves developmentally cheap. **The residual is therefore not "a designed structure" but "a fixed local
transformation (`conj`) of a random structure, currently executed by the host instead of by a wiring rule."** That is a
much smaller and more tractable thing than the framing "the bind structure is host-designed" implies — and it is the
precise target for §4-§5.

---

## 4. RANKED CANDIDATE SELF-ORGANIZATION MECHANISMS (for the substrate)

Ranked by P(eliminates the host computation of the bind structure, yielding weights that emerge on-substrate) ×
cheapness. The bar is: the binding+unbinding weights are produced by a rule that RUNS on/at the substrate (a one-time
local wiring prescription or activity-driven refinement), the resulting bind still inverts (held-out recovery ~parity
with the host conjugate), the no-confab moat holds, and generalization is unaffected.

### Mechanism 1 (DO FIRST — afternoon CPU/numpy, NO `sim/` edit) — the conjugate as a one-time LOCAL WIRING RULE (reciprocal connection)

**The cheapest and highest-probability win.** The unbind weight is `conj(role)`; the bind weight is `role`. For a
*unit-magnitude phasor*, the conjugate is just the phase negation (`exp(-iθ)` from `exp(+iθ)`) — a *per-component,
purely-local* function of the forward synapse. The mechanism: instead of the host calling `np.conj` and injecting,
express the unbind synapse as a **reciprocal/transpose wiring rule** applied ONCE at construction — "for each forward
bind synapse `(post, pre, w)`, install the conjugate-reciprocal synapse `(pre, post, conj(w))`." Biologically this is a
*reciprocal connection* (ubiquitous in cortex and thalamocortical loops) whose weight is the phase-conjugate of its
partner. The point is that the conjugate becomes a property of the *wiring rule the construction step applies locally*,
not a runtime host computation — the structure then "emerges" from a local prescription, identically to how
`dendritic_neuron.py` installs its fixed random apical projection from a rule at construction.
- **What's reusable:** the entire RF complex-synapse path (`rf_set_complex_weights`, the diagonal bind/unbind synapses)
  + the FRLF / `cortex_learned_binder_systematicity_probe.py` harness + the 320 stream codes. CPU/numpy.
- **Cheap-first de-risk:** in numpy, build the bind+unbind synapse lists by the *local reciprocal rule* (a function
  over each forward synapse) rather than by a global `np.conj` over the role vector, and confirm held-out bundle
  recovery is byte-identical to the current host path (it must be — `conj` per component IS the per-synapse rule). Then
  the deliverable is: the conjugate is a *local wiring rule*, so the host `np.conj` call is eliminable, and the
  structure is self-organized in the developmental sense (a local prescription run at construction).
- **Honest nuance:** this does NOT make the conjugate *learned* — it makes it a *local developmental rule* instead of a
  *host computation*. Under the project's own standard (`dendritic_neuron.py:25`, F.12, D.18) a fixed local wiring rule
  IS developmental self-organization; it is exactly how the genome specifies the retina. So this CLOSES the owner's
  two concerns: (1) the structure emerges from a substrate-local rule (spiking-end-to-end), and (2) it is a one-time
  configuration a chip applies (hardware-portable), with no runtime host. **This is the likely honest end-state** and
  it is an afternoon's de-risk.

### Mechanism 2 (weeks, gated on M1 — activity-dependent REFINEMENT of the conjugate via reciprocal STDP)

**The genuinely-learned (activity-driven) version, if M1's "local rule" framing is judged insufficiently emergent.**
The biology of L.06 (NMDAR Hebbian refinement) + L.02 (competitive elimination) says reciprocal connections can be
*shaped by activity* so that a backward synapse becomes the functional inverse of its forward partner. The mechanism: a
reciprocal-STDP / anti-Hebbian rule on the unbind synapses that, driven by spontaneous *paired* activity (bind a known
filler, drive the unbind, reinforce when the recovered filler matches), refines the unbind weight toward `conj(role)`
WITHOUT the host ever computing it. This is the resonator/predictive-coding flavor: the substrate discovers the inverse
by minimizing a recovery error it can compute locally (the cleaned-up filler vs the driven filler).
- **What's reusable:** STDP (J.08, on-bridge), the structural-plasticity gates, the dendritic plasticity module
  (`sim/dendritic_plasticity.py` Urbanczik-Senn local rule) for a local error signal, and L.05 patterned-spontaneous-
  activity as the *developmental driver* (the project's flagged "retinal-wave pretraining" lever).
- **Cheap-first de-risk:** numpy — initialize the unbind synapses at random, run a reciprocal-STDP refinement under
  paired spontaneous activity, and test whether the refined unbind recovers held-out bundled fillers ≥ ~0.9
  (parity with the host conjugate). Anti-cheats: a permuted-role control must collapse; a lesion of the refinement must
  not match; the moat must hold.
- **Honest nuance:** higher variance — the prior arc showed a *task-gradient* learner cannot discover the multiplicative
  reciprocal (`2026-06-20-FHRR-B-learned-binder-scoping.md` §1.2). The bet here is different: an *activity-driven local
  reciprocal rule with a local recovery error* (not a task gradient) targets the *conjugate of a known forward
  synapse*, which is a far more constrained problem than "learn an inverse from scratch." Still, this is the weeks-scale,
  research-grade option, to be attempted only if M1's local-rule reduction is deemed not to satisfy "emergent."

### Mechanism 3 (de-risk-only — is the structure mostly random, so the residual is small?) — quantify the random fraction

**Not a build — a measurement that may close the question.** §3 already shows the bind half is random and the residual
is the `conj` of that random half. This mechanism makes that quantitative and decisive: confirm in numpy that (a) the
role codes carry no host-design beyond `rng.uniform(seed)` (a developmental rule), and (b) the *only* host-computed
quantity is the per-component conjugate, which is a fixed local function. If both hold (they do, from the code read),
then the verdict is: **the structure is ~50% developmentally-cheap random + a fixed local transform of it — there is no
designed structure to self-organize, only a host *call* to relocate into a wiring rule (M1).** This is the cheapest
"de-risk" and it largely pre-answers the feasibility question.

### Mechanism 4 (NOT recommended — full molecular recognition code, catalog L.01) — for completeness

A genome-specified *structured* (non-random) wiring via a cadherin/neurexin-neuroligin recognition code (L.01) could in
principle specify the reciprocal conjugate structure directly. But the catalog itself judges this "probably out of
scope" (`feature-catalog.md:4187`) — it is a whole new molecular-recognition subsystem, months-scale, and it buys
nothing over M1's local reciprocal rule (which already gives the structure from a local prescription). Listed only to
show it was considered and ranked last.

---

## 5. CHEAP-FIRST DE-RISK + FEASIBILITY/SCOPE VERDICT

**Recommended cheap-first de-risk (Mechanism 1 + Mechanism 3 together, one afternoon, CPU/numpy, NO `sim/` edit):**
1. **Quantify the residual (M3):** in a numpy reproduction of the composer's bind/unbind, confirm by direct inspection
   + assertion that the only host-computed quantity in the structure is the per-component `conj(role)`, that the role
   codes are `rng.uniform(seed)`, and that the concept codes are the learned stream codes. (Pre-answered by the code
   read; the de-risk makes it an explicit, committed measurement.)
2. **Reduce the conjugate to a local wiring rule (M1):** build the unbind synapse list by a *local reciprocal rule*
   (a function applied to each forward synapse: `(post,pre,w) → (pre,post,conj(w))`) and confirm held-out bundle
   recovery is identical to the current global-`np.conj` path on the FRLF harness + the 320 stream codes.
3. **Anti-cheats:** the bind must still invert (held-out recovery ~parity with the host path); a permuted-role control
   must collapse to chance; the no-confab moat must hold; generalization across similar concepts must be unaffected
   (it is a codes property, already delivered — `2026-06-15-...GO.md`).
- **GO** (recovery byte-identical, controls collapse) ⇒ the conjugate is a local wiring rule, the host `np.conj` is
  eliminable, and the structure self-organizes in the developmental sense. Wire it as the construction-time rule
  (small, additive, reuses `rf_set_complex_weights`); the residual shrinks to "the bind op is a fixed structural
  primitive realized by a local reciprocal-wiring rule" — which is a *structural neural primitive*
  (binding-by-coincidence / a reciprocal connection), not a host computation. **This is the expected outcome and the
  honest end-state.**
- **NEGATIVE** (the local rule does not reproduce the host conjugate — implausible given `conj` is per-component, but
  if so) ⇒ the conjugate carries non-local information, host-specification of a fixed developmental structure is the
  defensible end-state, and Mechanism 2 (activity-driven reciprocal STDP) is the weeks-scale research option.

**Feasibility / scope verdict:**
- **Eliminating the host *computation* of the bind structure is feasible and CHEAP** (M1, afternoon-scale, no `sim/`
  edit beyond a construction-time wiring rule). The conjugate is a per-component local function; expressing it as a
  reciprocal-wiring rule the construction step applies is a faithful developmental stand-in by the project's own
  standard (`dendritic_neuron.py:25`; catalog F.12 / D.18 / L.01) and dissolves the owner's "host-injected" concern.
- **Making the conjugate genuinely *activity-learned* is weeks-scale and research-grade** (M2), worth attempting only
  if the owner judges the local-wiring-rule reduction insufficiently "emergent." The prior arc's NEGATIVE was for a
  *task-gradient* inverse; an *activity-driven local reciprocal rule* is a different and more constrained bet, but
  unproven.
- **The role codes are NOT a residual to close** — a random draw from a developmental seed is the cheapest, most
  faithful developmental structure (F.12, D.18, feedback-alignment). Treat the role-code half as already self-organized.
- **Honest scope caveat:** none of this adds a *capability* — the strategic prize (generalization across similar
  concepts) is already delivered on the *codes* axis (`2026-06-15-...GO.md` per `2026-06-20-FHRR-B-learned-binder-
  scoping.md` §1.3). Closing this residual is a **brain-based-purity + hardware-portability** goal: retire the host
  computation of the bind structure so the brain develops its own structure and a chip needs no host in the loop.

---

## 6. THE HARDWARE-PORT IMPLICATION (the decisive lens)

The neuromorphic roadmap (`docs/plans/2026-06-18-hardware-acceleration-neuromorphic-roadmap.md`) frames the target:
Mead's silicon retina (§1, lines 58-60) reproduces a *developmentally-fixed* structure in device physics — it does not
learn the center-surround; the structure is built in. The right hardware question for the bind structure is therefore
**not** "is it learned?" but **"is the weight pattern a one-time configuration the device loads, or does it require a
host in the loop per operation?"**
- The role codes (random, fixed-per-seed) and the conjugate (a fixed function of them) are **computed ONCE at
  construction, not per bind**. So even today they are one-time-configurable — like programming a memristor crossbar or
  a Loihi-2 synapse table once. The current break is narrow: the conjugate is computed by a *runtime host call* in the
  Python composer, which a host-free chip cannot make.
- **Mechanism 1 closes exactly this gap:** if the conjugate is a *local wiring rule* the configuration step applies
  (install the phase-conjugate reciprocal of each forward synapse), then the chip's one-time configuration produces the
  full bind+unbind structure with NO host in the loop — a clean port to the digital-neuromorphic rung (SpiNNaker2 /
  Loihi 2, §8 of the roadmap) and, ultimately, to an analog crossbar where the conjugate reciprocal is a wiring choice.
- **⇒ the hardware-port implication of this scope:** the residual does NOT fundamentally break the port; it identifies
  the ONE thing that must be expressed as a one-time wiring rule rather than a runtime host op (the bind/unbind
  conjugate reciprocal). Mechanism 1 is precisely that expression, and it is cheap. The role codes + concept codes are
  already config-loadable (random + learned weight tables). After M1, the entire bind structure is a one-time device
  configuration — host-free at runtime, which is the property the port requires.

---

## 7. SUMMARY

- **Two-part split:** the bind OPERATION (complex-synapse multiply / coincidence) is on-substrate spiking; the bind
  STRUCTURE is mostly developmentally-cheap RANDOM (role codes) + already-learned (concept codes) + ONE genuinely-host
  relationship: the conjugate-symmetry tie `unbind_weight = conj(bind_weight)`, computed by a host `np.conj` and
  injected.
- **The genuine residual** is therefore narrow and quantified: `D` conjugate weights per role relationship, which are a
  *fixed local function* (`conj`) of developmentally-cheap random weights — not a designed structure, but a host *call*
  that can be relocated into a local wiring rule.
- **Ranked self-organization mechanisms:** (1) the conjugate as a one-time LOCAL reciprocal-wiring rule (afternoon, no
  `sim/` edit, expected GO — the honest end-state); (2) activity-driven reciprocal-STDP refinement of the conjugate
  (weeks, research-grade, only if the local rule is judged insufficiently emergent); (3) measure the random fraction to
  confirm there is no designed structure to self-organize (de-risk-only); (4) full molecular recognition code (out of
  scope, ranked last).
- **Cheap-first de-risk:** numpy — build the unbind synapses by a local reciprocal rule, confirm held-out recovery is
  byte-identical to the host-`conj` path on the FRLF + 320-stream harness, with permuted-role / lesion / moat anti-
  cheats.
- **Feasibility/scope verdict:** eliminating the host *computation* of the bind structure is feasible and CHEAP (M1);
  the role codes are already self-organized (a developmental random draw); a genuinely-*learned* conjugate is weeks-
  scale and unproven; host-specification of a fixed developmental structure is the honest, biologically-faithful
  fallback if even the local rule fails. No capability is added — this is a brain-based-purity + hardware-portability
  goal.
- **Hardware-port implication:** the residual identifies the ONE runtime host op (the conjugate) that must become a
  one-time wiring rule; M1 is that rule; after it, the whole bind structure is a host-free one-time device
  configuration — the property a neuromorphic port requires.

**Sources (literature):**
- [Vector Symbolic Architectures (overview)](https://www.emergentmind.com/topics/vector-symbolic-architectures-vsas)
- [A Neurobiologically Plausible Vector Symbolic Architecture (Eliasmith/Stewart)](https://www.researchgate.net/publication/268195855_A_Neurobiologically_Plausible_Vector_Symbolic_Architecture)
- [VSAs answer Jackendoff's challenges for cognitive neuroscience (Gayler, cs/0412059)](https://arxiv.org/pdf/cs/0412059)
- [Vector-Derived Transformation Binding — better-suited to spiking neurons (Gosmann & Eliasmith)](https://compneuro.uwaterloo.ca/files/publications/gosmann.2019b.pdf)
- [Holographic Reduced Representations (Plate)](https://redwood.berkeley.edu/wp-content/uploads/2020/08/Plate-HRR-IEEE-TransNN.pdf)
- [Self-Organized Structuring of Recurrent Neuronal Networks (plasticity interplay)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8301101/)

**Cited project files / findings:**
- `sim/bridge.py:5559-5614` (`rf_kick`, `rf_set_complex_weights` — the pure-writer injection point)
- `research/runners/rf_phasor_composer.py:132,156-165,204-211,335` (role codes random; bind/unbind/cleanup structure)
- `research/runners/one_brain_composer.py:253-273` (`_compose_phases` — the production bind path)
- `sim/dendritic_neuron.py:25-27` (the project's existing FIXED-RANDOM developmental-projection precedent)
- `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` L.01/L.02/L.05/L.06 (4183-4241), F.12 (1613),
  D.18-class (1227-1229) — developmental wiring biology
- `research/findings/2026-06-20-FHRR-B-learned-binder-scoping.md` (the task-learning axis; §0/§1.2/§1.3)
- `research/findings/2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (codes axis already learned)
- `docs/plans/2026-06-18-hardware-acceleration-neuromorphic-roadmap.md` §1 (the silicon-retina / fixed-structure lens)
- CYCLE 343/344 commits `8d3a0cd9` / `8aa75cc1` (the prior arc + the owner challenge)
