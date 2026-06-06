# Deep research: the on-bridge spiking DECORRELATION blocker — missed mechanisms, literature synthesis, ranked shortlist + honest verdict — 2026-06-06

**RESEARCH ONLY. No `sim/` edits. Deliverable = this findings document.**

## 0. The blocker, restated precisely

For vector-symbolic composition (binding attributes to nouns) to work, the grounded concept codes (V1-Gabor /
modality-block features) must be DECORRELATED. numpy **ZCA whitening** reaches all-pairs coherence ~0.003 and gives
100% composition. Three LOCAL-RULE spiking attempts on the project's point-neuron Izhikevich E/I bridge all tested
NEGATIVE on the WORST PAIR (max pairwise coherence stuck ~0.9; mean dropped 0.42→0.06 but one within-block pair stays
clustered):

1. Single global FS (PV) pool: WTA + Hebbian feed-forward + homeostatic thresholds → competitive learning that
   **CLUSTERS** correlated inputs.
2. Plastic anti-Hebbian FS lateral (Földiák 1990: co-active outputs strengthen mutual inhibition) → **UNSTABLE**
   (over-suppresses toward silence), never resolves the worst pair.
3. K=8 topographic LOCAL FS sub-pools (each a local WTA on its own output window) → each clusters the same correlated
   input in parallel; no improvement.

Prior project diagnosis (`2026-06-06-A-spiking-decorrelation-mean-GO-worstpair-BOUNDARY.md`): *the local competitive
rule fundamentally clusters correlated inputs; Földiák's pairwise inhibitory weight W_ik does not map onto a
single-interneuron-class point-neuron E/I spiking substrate via any spatial/connectivity arrangement tested.*

This document asks: **what known mechanism did those three attempts MISS, and is any of it realizable on the bridge?**

### 0.1 The single most important reframe found by this research (read this first)

**DECORRELATION ≠ WHITENING, and the project has been measuring/chasing the wrong target.** Whitening = decorrelation
**+ variance equalization** (covariance → identity). ZCA reaches 0.003 *because it is the symmetric whitening that
equalizes every projection's variance*; the worst within-block pair is flattened precisely by the variance-equalizing
(rescaling) step, not by decorrelation alone (PCA/ZCA whitening literature [13]).

The three failed attempts are all **competitive / sparse-coding** rules. A competitive (WTA + Hebbian + homeostasis)
or anti-Hebbian rule **decorrelates and SPARSIFIES; it does not whiten.** Sparsification is the opposite move from
variance-equalization: it concentrates energy onto few units, which (a) leaves the dominant shared-block mode intact
on the units that "win" for that block, and (b) is exactly why the worst within-block pair — two concepts that share
the block mode AND get assigned to overlapping winners — stays clustered. **Földiák/SAILnet/competitive learning is
the wrong tool for the worst-pair**; it is a redundancy-reduction-by-sparsity tool, and its own authors only ever
claim **mean/RMS** correlation reduction, never worst-pair → 0 (see §3, §6 below — and they all PRE-WHITEN the input).

The two mechanisms that genuinely target **covariance → identity (all pairs, including the worst)** with the right
math are:
- **Pehlevan–Chklovskii Hebbian/anti-Hebbian whitening** [7] — local rules, but the lateral weights must converge to
  a *specific saddle-point solution ≈ inverse correlation matrix*, NOT "more inhibition between co-active units."
- **Duong–Lipshutz–Heeger–Chklovskii–Simoncelli adaptive whitening via gain-modulating interneurons** [9][10] — full
  covariance→identity by adapting **multiplicative GAINS** of an **overcomplete** interneuron set, explicitly because
  synaptic-plasticity whitening is "too slow and insufficiently reversible."

Both are RATE/graded models. Whether either survives the project's rate-coded-spiking + point-neuron substrate is the
crux, and is assessed honestly per-mechanism below. The strong prior (from the project's own opponency wall) is that
the **variance-equalization / common-mode-cancellation step wants a graded/analog or multi-timescale-gain stage**, the
same constraint that killed the opponency.

---

## 1. Candidate: balanced spike-coding networks (Deneve–Machens / Boerlin) — the headline hypothesis

### (a) What it is + references
The "tight E/I balance → efficient code" framework: Boerlin, Machens & Denève 2013 [1] (predictive coding of dynamical
variables in balanced spiking LIF networks) and Denève & Machens 2016 [2] (*Efficient codes and balanced networks*,
Nat Neurosci). A population represents a signal `x` such that a **linear readout** `x̂ = Γ r` of the spike trains
tracks it; each neuron fires only when a spike **reduces the readout error**. The recurrent connections are
**not learned by a generic Hebbian rule** — they are set by the decoder geometry: the recurrent weight matrix is
**Ω = Γᵀ Γ** (the Gram/overlap matrix of the neurons' decoding vectors). Neurons with similar decoding vectors
strongly inhibit each other (fast recurrent inhibition), producing tight, millisecond E/I balance: inhibition tracks
excitation spike-by-spike, suppressing any firing that is *predictable from other neurons* — i.e. redundancy.

### (b) Does it achieve PAIRWISE (worst-pair) decorrelation?
**Conceptually yes, for the represented signal — but this is the strongest and most double-edged finding.** Tight
balance literally means "don't fire if another neuron's spike already explains this input component," which is
redundancy removal at the level of *every pair* (the inhibition Ω = ΓᵀΓ is exactly pairwise-specific to decoding
overlap). Denève–Machens [2] state the balance "prevents the neuron from firing at times predictable by other
neurons" — pairwise predictability, not just mean. **BUT:** the decorrelation in spike-coding networks is of the
*read-out residual*, and the recurrent weights are **derived analytically from the decoder Γ**, not learned by a local
co-activity rule. The decorrelating power comes from **knowing Γ and computing ΓᵀΓ**, i.e. from an externally-imposed,
pairwise-exact lateral weight matrix. This is the *whitening solution handed to the network as fixed wiring*, not a
local rule that discovers it.

The recent **biology-grounded learning** version is decisive for the project's question: Habashy/Akrout-style and
specifically the 2024/2025 PLOS Comp Biol paper **"Emergence of sparse coding, balance and decorrelation from a
biologically-grounded spiking V1 model"** [3] *learns* the balanced weights with symmetric STDP on a single
homogeneous inhibitory population, and reports decorrelation that is **mean-near-zero with a POSITIVE-SKEWED residual**
("highly correlated firing is more likely than highly uncorrelated firing") — and crucially **pre-whitens the LGN
input with a divisive-normalization center-surround filter before the spiking network sees it**, and **does not
benchmark against ZCA/PCA** [3]. That is the project's exact result reproduced in the literature: learned-balance
spiking nets reduce the MEAN, leave a skewed worst-pair tail, and lean on an upstream graded whitening stage.

### (c) Realizable on the project's bridge as described?
**Partially, and only the WEAK form.** The bridge can (i) wire `Ω = ΓᵀΓ` as fixed lateral inhibition IF Γ (the
concept codes' decoding directions) is known — but that is just installing ZCA-as-wiring, the numpy answer in spiking
clothes, not a learned biological decorrelation. (ii) The *learned* form (symmetric STDP balance) is essentially what
attempt #1+#2 already did, and the literature [3] confirms it bottoms out at the same mean-reduced/skewed-tail place.
The bridge has no mechanism to make fast recurrent inhibition track excitation at the sub-millisecond, spike-by-spike
precision the *tight*-balance theory requires (dt=0.5–1.0 ms, Izhikevich point neurons, conductance with finite decay)
— so the project gets *loose* balance (mean) not *tight* balance (pairwise). This matches [2]'s own timescale caveat.

### (d) Concrete on-bridge experiment (if pursued)
Install `Ω = ΓᵀΓ` as **fixed** lateral inhibition among IT neurons, where Γ = the raw grounded code matrix (rows =
concept codes). This tests the *ceiling* of the balanced-network idea (the analytic solution) without learning. If
even the analytic ΓᵀΓ wiring fails to drop the worst pair in spikes → confirms the rate-coded-spiking SNR wall (it
will, per the opponency precedent). If it succeeds → the gap is purely *learning* Ω locally, and Pehlevan–Chklovskii
(§7) is the rule to try next. **Honest expectation: the fixed-Ω test is the single most informative experiment in this
whole document** — it cleanly separates "can spikes hold a whitening solution at all" from "can a local rule find it."

> **Verdict on Deneve–Machens:** This is the theoretically-correct decorrelation mechanism and the headline hypothesis
> was right to flag it — its lateral weights ARE pairwise-exact (ΓᵀΓ). **But the pairwise exactness lives in the
> analytic weight matrix, handed in as wiring, not in a local learning rule; and the only *learned* spiking
> realization in the literature [3] reproduces the project's mean-reduced / skewed-worst-pair boundary AND pre-whitens
> upstream.** So it does NOT supply a missed *local* mechanism that resolves the worst pair — it supplies the
> *target* (ΓᵀΓ) and confirms that getting there in spikes wants either fixed wiring or an upstream graded stage.

---

## 2. Candidate: LOCAL DENDRITIC BALANCE — the precise mechanistic explanation of the project's failure

### (a) What it is + references
Mikulasch, Rudelt, Priesemann 2021, *Local dendritic balance enables learning of efficient representations in networks
of spiking neurons*, PNAS [4] (+ the 2022 follow-up *Dendritic predictive coding* [11], + Trends Neurosci 2022 review
*Where is the error?* [12]). Core claim, quoted from the project's reading of the paper: **"A point neuron with global
somatic integration cannot decorrelate inputs; lateral inhibition must be computed LOCALLY at dendrites to whiten the
input representation. Without dendritic compartmentalization, a single inhibitory pool produces uniform suppression
across all inputs, failing to capture the structure needed for efficient coding of correlated signals."** [4]

The mechanism: decorrelation requires **input-specific lateral weights** (the lateral inhibition between two units must
match THEIR specific shared correlation). In a point neuron, all inhibition arrives at one somatic compartment and is
averaged together with excitation → the input-specific structure is destroyed before plasticity can read it. Putting
inhibition on **dendritic compartments** lets each compartment balance E and I *locally*, so the synapse-specific
correlation is visible to a local (voltage-dependent) plasticity rule and the lateral weight can be learned per-pair.

### (b) Does it achieve PAIRWISE (worst-pair) decorrelation?
**Yes — this is the paper's central result, and it is explicitly the pairwise/whitening claim, not mean.** Per [4]:
"the resulting code exhibits pairwise decorrelation — the covariance structure is whitened, not just mean-adjusted.
The neurons develop mutually inhibitory connections proportional to input correlations." It draws the connection to
Deneve–Machens (it is the *learned, local, dendritic* route to the same ΓᵀΓ-style balance) and to voltage-dependent
plasticity.

### (c) Realizable on the project's bridge as described?
**NO — this is the load-bearing negative.** The paper's thesis IS the project's diagnosis, promoted to a theorem:
**point neurons with a single global inhibitory pool cannot whiten correlated inputs; dendritic compartmentalization is
the necessary ingredient.** The project's substrate is *point* Izhikevich neurons. There are no dendritic compartments,
no local dendritic E/I balance, no voltage-dependent compartmental plasticity. So the mechanism that the literature
identifies as the one that turns "mean-only decorrelation" into "pairwise whitening" is **exactly the capability the
substrate lacks**. This is not a tuning gap; it is an architectural prerequisite the point-neuron bridge does not meet.

### (d) Concrete on-bridge experiment
A faithful test would require adding dendritic compartments (multi-compartment neurons with separate E and I sites +
local plasticity) — a **multi-month `sim/` build** that changes the neuron model, explicitly out of scope here and a
strategic owner decision. A cheap *proxy* (NOT faithful): give each IT neuron a small set of "sub-dendrite" helper
units that each receive one feedforward cluster + matched inhibition, then pool — but this is just re-deriving a
2-layer network and will re-cluster (it is attempt #3 with extra steps). **No cheap on-bridge realization exists.**

> **Verdict on dendritic balance:** This is the **single most important explanatory finding** of the deep research.
> The peer-reviewed mechanism for converting mean-decorrelation into pairwise-whitening in spiking nets is **local
> dendritic balance**, and it is mathematically/biologically the reason point neurons fail. **It CONFIRMS the
> project's boundary at the level of theory: the worst-pair limit is a point-neuron limit, and the named escape
> (dendritic computation) is a substrate the bridge does not have.** Translatable insight: interneuron-TYPE diversity
> AND dendritic targeting are *load-bearing for whitening* — a single point-neuron interneuron class provably cannot.

---

## 3. Candidate: SPARSE-CODING dynamics with a SEPARATE inhibitory population (SAILnet, King–Zylberberg–DeWeese, LCA)

This is the most important *constructive* candidate because it directly addresses the project's two failures (the
clustering of attempt #1 and the **instability** of attempt #2), and it is the closest match to the bridge's actual
machinery (separate FS pool, Hebbian feedforward, homeostatic thresholds, spiking LIF).

### (a) What it is + references
- **SAILnet** (Zylberberg, Murphy, DeWeese 2011 [5]): spiking LIF network, **synaptically-local** rules, learns
  Gabor RFs + a sparse code. Three rules: feedforward (Oja `ΔW_ff ∝ y·x − y²·W_ff`), threshold homeostasis
  (`ΔΘ_i ∝ y_i − p`), and **lateral inhibition `ΔW_ij ∝ (c_ij − p²)`** where `c_ij` = pairwise co-firing and `p` = the
  target rate. The lateral rule is all-to-all pairwise and has a **fixed point at `c_ij = p²` for EVERY pair** — it is
  a *target-correlation* rule, not a maximize-suppression rule. It is described as "a biologically-inspired variation
  on a network originally due to **Földiák**" [5].
- **King, Zylberberg & DeWeese 2013** [6] (*Inhibitory Interneurons Decorrelate Excitatory Cells…*, J Neurosci):
  extends SAILnet to obey **Dale's law with a SEPARATE inhibitory interneuron population** (no direct E–E inhibition).
  Introduces the **"Correlation-Measuring (CM) rule"**: `ΔW_jk = α(x_j y_i − ⟨x_j⟩⟨y_i⟩)`, which **converges to weights
  ∝ covariance** of pre/post activity, constrained non-negative. Reports final **RMS pairwise correlation < 0.13** [6].
- **LCA** (Rozell et al. 2008): the rate/dynamical-systems sparse-coding solver; lateral inhibition strength ∝ the
  **similarity (overlap) of two neurons' dictionary elements**, forcing decorrelation [Rozell].

### (b) Does it achieve PAIRWISE (worst-pair) decorrelation? — **the decisive nuance**
**No — only mean/RMS, by the authors' own reporting; and BOTH pre-whiten the input.** This is the most important
correction to any optimism about this family:
- SAILnet [5]: the target is `⟨c_ij⟩ = p²` and "the decorrelation operates **on average** over many images, not
  enforcing an exact target for every pair instantaneously… individual pair correlations fluctuate, but their ensemble
  **mean** converges." Network parameters "bounce around the final target state."
- King–Zylberberg [6]: report **RMS** Pearson correlation **< 0.13** (good, ~7× better than the project's 0.9 worst-
  pair, but RMS, NOT max), and — critically — train on **"image patches drawn from WHITENED images of natural
  scenes."** The spiking network's job is to *maintain/refine* decorrelation on already-whitened input, not to whiten
  raw correlated input from scratch.

So this family **does not** claim worst-pair → 0 on raw correlated input. It is a redundancy-reduction-by-sparsity
mechanism layered on top of an upstream graded whitening stage — the same two-stage architecture as Deneve–Machens-
learned [3] and the retina (§5).

### (c) Realizable on the bridge? — **YES for the stable rule; this is the genuine actionable upgrade**
The CM rule [6] and SAILnet's `(c_ij − p²)` rule [5] are **the fix for the project's attempt-#2 instability.** The
project's anti-Hebbian FS lateral went unstable (over-suppressed to silence) precisely because it was a naive
"co-active → strengthen inhibition" rule with no fixed point. SAILnet/CM have an explicit fixed point (`c_ij = p²` /
`W ∝ covariance`), which is why they are stable and convergent, and they use a **separate inhibitory population**
(the bridge's FS pool) with **homeostatic thresholds** (the bridge has this). The project's own §A finding noted it
implemented "Földiák" but the **stable, fixed-point form (CM / `c_ij − p²`) is a different, better rule than what was
tested.** The bridge can express it: per-pair plastic IT→FS→IT with a plasticity rule targeting `c_ij − p²` and the
existing homeostasis. **This is the one mechanism in this document that is both missed AND cheaply realizable.**

### (d) Concrete on-bridge experiment
Re-run attempt #2 but replace the naive anti-Hebbian rule with the **SAILnet `Δw ∝ (c_ij − p²)` / King CM
`Δw ∝ cov(pre,post)` rule** on the IT↔FS lateral synapses, keep the homeostatic thresholds, **and on the SAME
already-mean-decorrelated codes** (don't expect raw→whitened). Measure RMS *and* max coherence. Honest hypothesis:
**RMS will drop toward ~0.1 (matching [6]); the MAX/worst pair will improve modestly but NOT reach 0.003**, because
the rule targets mean correlation, sparsifies rather than variance-equalizes, and the literature never claims worst-
pair → 0 without the upstream graded whitening. A positive result (RMS ≪ 0.13, worst-pair materially < 0.9, stable,
no silence) would still be a real GO for "mean decorrelation done with the *correct stable biological rule*," even if
the worst pair remains a residual.

> **Verdict on sparse-coding/SAILnet family:** The project tested the *unstable* Földiák form; the literature's
> *stable* form (SAILnet `c_ij−p²` / King CM `W∝cov`) is a genuine missed mechanism and the **most promising cheap
> on-bridge upgrade** — but it targets MEAN/RMS correlation and **pre-whitens its input**, so it is very unlikely to
> resolve the worst pair to ZCA's 0.003. Expect "stable mean-decorrelation," not "whitening."

---

## 4. Candidate: DIVISIVE NORMALIZATION (Carandini–Heeger)

### (a) What it is + references
Carandini & Heeger 2012 [8] (*Normalization as a canonical neural computation*, Nat Rev Neurosci): a neuron's response
is **divided by the summed activity of a normalization pool**, `R_i = D·x_i / (σ + Σ_j x_j)`. PV interneurons in V1
produce **divisive** gain changes [8]; the fly antennal lobe implements feedforward divisive normalization via
presynaptic local interneurons [8].

### (b) Does it achieve PAIRWISE decorrelation?
**No — divisive normalization is a GAIN/common-scale operation, not a pairwise-decorrelator.** It removes a *shared
multiplicative* common mode (contrast/ambient level) — this is variance/energy *normalization*, which is one HALF of
whitening (the variance-equalization half) but applied as a *scalar pool divisor*, not a per-pair covariance cancel.
It will *reduce* mean coherence by suppressing the dominant shared mode's amplitude, but two concepts that share a
*block* (a structured, non-scalar common component) are not separated by dividing both by the same pool sum — their
*pattern* overlap is untouched. Divisive normalization is the mechanism behind the project's **own prior "divisive-norm
cleanup 0.84" NEGATIVE** (CLAUDE.md composer-cleanup arc) — it caps below parity for exactly this reason.

### (c) Realizable on the bridge?
Partially — a shunting/divisive inhibition pool is expressible (FS pool whose inhibition scales with total IT
activity, or a conductance-based shunt). But per (b) it addresses the wrong half of the problem (scalar gain, not
pairwise covariance), and the project already saw it plateau. **Not recommended as the worst-pair fix.** It is,
however, a legitimate *component* of the adaptive-whitening gain mechanism (§9) — divisive normalization is the
biological substrate for the "gain modulation" that §9 needs.

> **Verdict on divisive normalization:** Real, canonical, partially on-bridge — but it is *scalar common-mode/gain*
> control, which is the variance-equalization half of whitening applied non-specifically. It reduces the mean and the
> dominant mode's amplitude; it does NOT do per-pair decorrelation. Useful only as the gain primitive inside §9.

---

## 5. Candidate: RETINAL / THALAMIC decorrelation (Atick–Redlich, Dan–Atick–Reid, center-surround)

### (a) What it is + references
Atick & Redlich 1992 [14] (*What does the retina know about natural scenes?*, Neural Comp): the retina **spatially
whitens** natural input — a center-surround filter whose sensitivity rises with spatial frequency to flatten (whiten)
the 1/f² natural power spectrum, **decorrelating** ganglion outputs and suppressing noise. Dan, Atick & Reid 1996
(J Neurosci) confirmed LGN responses to natural movies are decorrelated as predicted. Graham et al. + Pitkow–Meister
extended/qualified it (the whitening is SNR-dependent — sharp high-pass at high SNR, low-pass smoothing at low SNR).

### (b) Does it achieve PAIRWISE decorrelation? — and the constraint that matters most
**Yes for the represented signal — but it is GRADED / ANALOG, in the pre-spike stage, by design.** This is the
critical structural fact: retinal/thalamic whitening is done by **graded center-surround filtering in the outer/inner
plexiform layers BEFORE ganglion-cell spiking** (Atick–Redlich's filter is a linear graded transfer function). Every
downstream spiking model in this literature — SAILnet [5], King–Zylberberg [6], the spiking-V1 PLOS paper [3] —
**takes whitened input as a given** and applies a *graded* center-surround / divisive pre-filter at the LGN stage. The
bioRxiv paper **"Spatial whitening in the retina may be necessary for V1 to learn a sparse representation"** makes the
strong-form claim in its title: the cortical spiking sparse-coder needs the *upstream graded whitening* to work.

### (c) The decorrelation/opponency parallel — ASSESSED (this was an explicit ask)
**Decorrelation has the SAME constraint as the project's opponency wall.** The opponency finding
(`2026-06-05-B-opponency-rate-coded-SNR-wall-CONFIRMED.md`): common-mode removal of a small signed difference of
correlated channels CANNOT be done in spike rates (SNR amplification ~√(2/(1−ρ)) ≈ 4.3× at ρ=0.89); **biology removes
the common mode in the GRADED analog stage before action potentials** (Kandel Ch 22 p543). **Whitening's variance-
equalization / worst-pair-cancellation is structurally the same operation: subtracting/cancelling a large shared
component to expose a small residual.** The retina does it graded-and-pre-spike for exactly the same SNR reason. The
spiking V1 models confirm it by *importing* the graded-whitened input rather than whitening in spikes. **So the deep
research's strongest convergent conclusion is: the worst-pair/whitening step belongs in a GRADED/ANALOG (or pre-spike
divisive) stage, not in a rate-coded spiking competition — the identical boundary the project already proved for
opponency.**

### (d) Concrete on-bridge experiment
The project ALREADY does this graded-whitening role in numpy (ZCA on the codebook before they hit the bridge) — which,
read through this lens, is **biologically correct, not a cheat**: it is the retina/LGN graded-whitening stage that
biology also performs pre-spike. The honest on-bridge experiment is to *accept* a graded pre-spike whitening stage
(numpy ZCA, OR a graded divisive-normalization layer computed on the input features before they drive the bridge) and
make only the *downstream* (binding/cleanup) spiking — which the project has already validated at 100% with ZCA.

> **Verdict on retinal/thalamic decorrelation:** Strong PAIRWISE whitening — but **GRADED and PRE-SPIKE by biological
> design**, for the same SNR reason as the opponency wall. This is the deep research's **central convergent finding:
> decorrelation/whitening of strongly-correlated codes is, like opponency, an analog/graded-stage operation; rate-
> coded spiking competition cannot reach ZCA's all-pairs 0.003.** The numpy-ZCA pre-stage is the biologically-faithful
> analog whitening, not a shortcut to apologize for.

---

## 6. Candidate: INTERNEURON-TYPE DIVERSITY (PV + SST + VIP) — the project's "option 1"

### (a) What it is + references
The canonical cortical microcircuit has ≥3 GABAergic classes (>80% of interneurons): **PV** (fast, perisomatic,
divisive gain/precision), **SST** (slow, dendrite-targeting, lateral/surround), **VIP** (disinhibitory, inhibits SST).
Connectivity (Pfeffer et al. 2013 [15]): **PV → pyramidal + PV (incl. itself); SST → all other types EXCEPT itself;
VIP → mainly SST** (disinhibition). Pi et al. 2013 [16] (Nature): VIP cells mediate **disinhibitory gating/gain**,
recruited by reinforcement signals. Functional-role syntheses [17]: in a predictive-coding microcircuit, "SST encode
the mean of top-down predictions, VIP capture uncertainty, PV represent precision of feedforward drive" [17]; PV →
local processing/boundary detection, **SST → longer-range spatial competition among receptive fields** [17].

### (b) Does the diversity achieve PAIRWISE decorrelation that one class can't?
**Mixed / qualified — and a key counterintuitive finding.** The diversity is genuinely load-bearing for efficient
coding (different temporal kernels, dendrite-vs-soma targeting, disinhibitory gating), and **SST's dendrite-targeting
+ longer-range competition is the closest cellular analog to the "input-specific lateral weight" that whitening needs**
(it overlaps the dendritic-balance story §2: SST inhibition lands on dendrites). BUT: the developmental-decorrelation
literature reports that **"inhibiting SST interneurons ACCELERATES the decorrelation of cortical activity"** [17] —
i.e. in some regimes SST inhibition *correlates* (synchronizes) rather than decorrelates. So "add SST and whitening
appears" is NOT supported; the diversity's decorrelation role is regime- and target-specific, and intertwined with
dendritic targeting (which the point-neuron bridge cannot express).

### (c) Realizable on the bridge?
**The connectivity motif is expressible (the bridge has region framework + multiple neuron types + transmission/
plasticity gates) — but the COMPUTATIONAL power of the diversity is in the DENDRITIC targeting and distinct temporal
dynamics, which the point-neuron substrate strips away.** The project's K=8 topographic-FS test (attempt #3) already
showed that *spatial/connectivity* diversity of a single point-neuron inhibitory class does not help (each sub-pool
clusters in parallel). Adding PV+SST+VIP *as more point-neuron classes with different time constants* is the untested
residual, but §2 (dendritic balance) predicts it will still fail on the worst pair because the missing ingredient is
**dendritic compartmentalization**, not interneuron *count* or *temporal kernel* alone.

### (d) Concrete on-bridge experiment
Build the Pfeffer motif with distinct Izhikevich presets: PV (fast, perisomatic→IT soma, divisive), SST (slow,
→IT — but the bridge can only target the soma), VIP (→SST disinhibition gate). Honest expectation per §2/§3: PV gives
the §4 divisive gain (mean reduction), SST gives §3-style lateral (mean reduction), VIP gives gating (irrelevant to
worst-pair); **without dendritic targeting the worst pair persists.** This is a 1–2 week build whose most likely
outcome is a *cleaner mean-decorrelation with stable dynamics* (PV divisive + SST lateral + VIP gating is a principled
stable microcircuit) but **not** worst-pair resolution.

> **Verdict on interneuron diversity:** The diversity is load-bearing for efficient coding and SST-dendritic is the
> right *cellular* analog of the whitening lateral weight — **but its decorrelating power is inseparable from DENDRITIC
> targeting (§2), which point neurons lack**, and "inhibit-SST-accelerates-decorrelation" [17] shows it is not a simple
> add-on. Diversity-as-extra-point-classes is predicted to give cleaner stable *mean* decorrelation, not worst-pair
> whitening. The project's §A boundary statement ("interneuron-TYPE diversity is load-bearing for efficient coding")
> is CORRECT, and this research adds the mechanistic *why*: the type diversity matters because SST targets dendrites.

---

## 7. Candidate: Hebbian/anti-Hebbian WHITENING with the CORRECT objective (Pehlevan–Chklovskii)

### (a) What it is + references
Pehlevan & Chklovskii 2015 [7] (*Optimization theory of Hebbian/anti-Hebbian networks for PCA and whitening*) + the
2017 "why similarity-matching → Hebbian/anti-Hebbian" follow-up. A **similarity-matching objective with an added
decorrelating term**, optimized as a **min-max / saddle point**: feedforward weights MINIMIZE, lateral (inhibitory)
weights MAXIMIZE. The output of the correctly-solved network is **fully whitened (covariance → identity, all pairs
decorrelated AND variance-equalized)** with local rules. Rate/graded network.

### (b) Does it achieve PAIRWISE (worst-pair) whitening? — and WHY the project's anti-Hebbian failed
**Yes — full whitening — and this paper gives the EXACT reason the project's attempt #2 failed.** Quoted from the
extraction of [7]: *"A straightforward anti-Hebbian approach (strengthening inhibition between co-active neurons)
FAILS because the lateral weights must converge to the INVERSE of the input correlation matrix at the saddle point —
a mathematically specific solution, not merely increased inhibition."* The project's anti-Hebbian rule strengthened
inhibition with co-activity (Földiák-style) — which is the *wrong dynamics*: it does not solve the min-max, so it does
not converge to `M ≈ C⁻¹` (the whitening lateral weights). Instead it monotonically increases inhibition → the
over-suppression-to-silence the project observed. **This is the precise theoretical explanation of the instability.**

### (c) Realizable on the bridge?
**This is the most theoretically-promising LEARNED route — but with two serious caveats.** (1) It is a **rate/graded**
network; the saddle-point dynamics (lateral weights doing gradient ASCENT while feedforward does descent) are a
continuous-value optimization, and whether the saddle is reachable with **spiking, non-negative (Dale), point-neuron**
lateral inhibition is unproven — the same rate-vs-spike gap as everywhere else. (2) The lateral weights converging to
`C⁻¹` is a *dense, signed, precise* matrix; Dale's law (non-negative inhibition) + finite spiking precision are exactly
the constraints that the project's substrate imposes and that [7]'s graded derivation does not respect. King–Zylberberg
[6] is the *Dale-compliant, spiking* descendant of this idea and it only reached RMS 0.13 on pre-whitened input — which
is the empirical answer to "what happens when you force [7] onto a Dale spiking substrate": you get mean-decorrelation,
not full whitening.

### (d) Concrete on-bridge experiment
Implement the [7] min-max as closely as the bridge allows: feedforward Hebbian (minimize) + lateral inhibitory
(anti-Hebbian, but with the *similarity-matching* update `ΔM_ij ∝ y_i y_j − M_ij`, which has the saddle fixed point,
NOT the naive `ΔM_ij ∝ y_i y_j`). The single-character difference — the **`− M_ij` decay term** giving a fixed point at
`M_ij = ⟨y_i y_j⟩` — is plausibly the missing piece vs the project's unstable rule, and is the SAME fixed-point idea as
SAILnet's `c_ij − p²` (§3) and King's `W ∝ cov` (§3). **So §3 and §7 converge on ONE actionable change: give the
lateral rule a fixed point (a decay/target term), don't just accumulate co-activity.**

> **Verdict on Pehlevan–Chklovskii:** Theoretically achieves FULL whitening with local rules and **precisely diagnoses
> the project's instability** (lateral weights must reach `C⁻¹` at a saddle, not just grow with co-activity). The
> actionable distillation — *add a fixed-point/decay term to the lateral rule* — is the same fix §3 gives. **But the
> guarantee is for a graded, signed, non-Dale network; the Dale-compliant spiking descendant (King–Zylberberg) only
> reaches RMS 0.13 on pre-whitened input**, which is the realistic on-bridge ceiling.

---

## 8. Candidate: PREDICTIVE CODING (Rao–Ballard, Bastos, Keller–Mrsic-Flogel)

### (a) What it is + references
Rao & Ballard 1999; Bastos et al. 2012 [B] (canonical microcircuits for predictive coding): higher levels send
**inhibitory top-down predictions** that are **subtracted** from input; only the **prediction-error residual**
propagates forward. Subtracting a prediction of the redundant/correlated component decorrelates the residual (the
error neurons carry the un-predictable part).

### (b) Does it achieve PAIRWISE decorrelation?
**In principle yes (the residual is the decorrelated/whitened part) — but with the SAME graded/SNR caveat.** Predictive
coding decorrelates by **subtracting a prediction** — and (per §1's opponency synthesis Option C, Srinivasan–Laughlin–
Dubs 1982) the robust version subtracts a *smooth/denoised* prediction estimated by a well-averaged population, with the
gain SET BY SNR (Atick–Redlich). The error computation requires inhibitory feedback to subtract from excitation — and
the error neurons are rate-coded, so recovering a *small* error residual after subtracting a *large* correlated
prediction is **exactly the small-signed-difference-of-correlated-channels problem** the opponency wall identified.
Mikulasch's *dendritic predictive coding* [11] makes the subtraction DENDRITIC for precisely this robustness reason —
looping back to §2 (dendrites needed).

### (c) Realizable on the bridge?
The motif (top-down inhibitory prediction, feedforward error) is expressible, but (i) it needs a *learned predictor*
of the common mode (another training stage), (ii) the error-residual read is rate-coded small-signal (SNR wall), and
(iii) the robust form is dendritic [11] (substrate gap). Same boundary as §2/§5.

> **Verdict on predictive coding:** Decorrelates by subtracting a prediction = whitening-by-prediction, but the
> residual read is the small-signal-of-correlated-rates problem (opponency wall), and the robust realization is
> dendritic [11]. No cheap point-neuron rate-coded route to the worst pair.

---

## 9. Candidate: ADAPTIVE WHITENING via GAIN-MODULATING INTERNEURONS (Duong–Lipshutz–Heeger–Chklovskii–Simoncelli) — the most promising MISSED mechanism

### (a) What it is + references
Duong, Lipshutz, Heeger, Chklovskii, Simoncelli — ICML 2023 [9] (*Adaptive whitening in neural populations with
gain-modulating interneurons*) + NeurIPS 2023 [10] (*Adaptive whitening with fast gain modulation and slow synaptic
plasticity*). A **recurrent network with FIXED synaptic weights** plus an **overcomplete set of auxiliary
interneurons** whose **multiplicative GAINS** are adapted online to **whiten the primary-neuron outputs (covariance →
identity)**. The explicit motivation, quoted from [9]: *"existing neural circuit models of adaptive whitening operate
by modifying synaptic interactions; however, such modifications would seem both too slow and insufficiently
reversible"* — so they replace synaptic plasticity with **gain modulation** (fast, reversible, biologically grounded
in divisive normalization §4). The objective whitens by **adjusting the marginal variances of an overcomplete set of
projections** [9]; sign-constraining the gains (gains ≥ 0, biologically apt) improves robustness [9]. The 2023 NeurIPS
follow-up [10] unifies BOTH timescales: **fast gain modulation + slow synaptic plasticity** together.

### (b) Does it achieve PAIRWISE (worst-pair) whitening?
**Yes — this is the one mechanism whose stated GOAL is exactly covariance → identity (all pairs, including the worst,
AND variance-equalized), with a biologically-plausible interneuron circuit.** It is the gain-based route to the same
whitening target as Pehlevan–Chklovskii §7, and it explicitly fixes the "synaptic plasticity is too slow/irreversible"
failure mode — which maps onto the project's two synaptic-plasticity failures (attempts #1/#2). The
**overcompleteness** of the interneuron pool is the structural requirement: you need more interneuron projections than
input dimensions, so their gains can independently cancel each pairwise covariance.

### (c) Realizable on the bridge?
**Partially — and this is the most interesting "maybe."** The bridge HAS a per-neuron excitability/gain control
(homeostatic threshold adaptation, and the neuromodulator `excitability_drive` / `synaptic_gain` machinery). The
Duong mechanism needs: (i) an **overcomplete fixed-weight interneuron pool** (expressible: FS pool larger than input
dim, fixed random/structured weights), (ii) **per-interneuron multiplicative gain** adapted by the whitening rule
(the bridge's homeostasis adapts thresholds = a gain-like knob, but NOT the *specific* Duong gain-update rule, and
NOT a clean multiplicative gain on the interneuron output), (iii) the gains feed back divisively/subtractively to the
primary neurons. Caveats: it is a **rate/graded** algorithm (no spiking realization demonstrated in [9][10]); the gain
update is a specific online rule, not the bridge's homeostasis; and "marginal variance of a projection" is a graded
quantity that a rate-coded spike read estimates noisily (the recurring SNR concern). **But it is the only candidate
that (a) targets the worst pair by design, (b) explicitly diagnoses the project's synaptic-plasticity failure, and
(c) uses a mechanism (gain modulation) the bridge partially has.**

### (d) Concrete on-bridge experiment (the single best constructive test in this document)
Build an **overcomplete fixed-weight FS interneuron pool** (n_FS > n_input_dim, fixed random projections of the input
codes) feeding divisive/subtractive inhibition to the IT pool, and adapt **per-FS multiplicative gains** with the
Duong rule (gain of interneuron k ↑ when its projection's variance > target, ↓ otherwise — a *variance*-homeostasis,
distinct from the bridge's *rate*-homeostasis). Measure covariance → identity (all-pairs coherence incl. max). Cheap
numpy-first de-risk (mandatory before any spiking attempt, per project discipline): run the Duong rate algorithm on the
exact correlated codebook and confirm it reaches all-pairs ~ZCA; THEN test whether a spiking/Dale/point-neuron version
preserves it. **Honest prediction: the numpy Duong algorithm WILL whiten (it provably does); the open question is
whether the gain modulation survives rate-coded spiking — and the opponency/retina precedents (§5) suggest the
variance-cancellation step, like common-mode removal, wants a graded gain stage.** If the spiking version holds → a
genuine on-bridge whitening GO; if it degrades to mean-only → confirms the graded-stage boundary one more time, now
for the *gain* route specifically.

> **Verdict on gain-modulating adaptive whitening:** The **#1 ranked missed mechanism.** It is the only candidate
> that targets worst-pair whitening (covariance→identity) by design, explicitly diagnoses why the project's synaptic-
> plasticity attempts failed ("too slow, not reversible"), and uses a primitive (interneuron gain modulation /
> divisive normalization) the bridge partially possesses. **Caveat: rate/graded algorithm, no demonstrated spiking
> realization; the variance-equalization is graded and likely hits the same analog-stage boundary.** Worth a numpy-
> first de-risk before any build.

---

## 10. RANKED SHORTLIST — most promising MISSED mechanisms for closing the WORST PAIR

| # | Mechanism | Targets worst-pair? | On-bridge realizable? | Cheap to test? | Honest expectation |
|---|---|---|---|---|---|
| **1** | **Gain-modulating adaptive whitening** (Duong–Chklovskii–Simoncelli [9][10]) | **YES — covariance→identity by design** | Partial (bridge has gain/homeostasis primitives; overcomplete FS pool expressible) | **YES (numpy-first)** | numpy whitens provably; spiking-gain survival is the open question (graded-stage risk) |
| **2** | **Stable-fixed-point lateral rule** (SAILnet `c_ij−p²` [5] / King CM `W∝cov` [6] / Pehlevan `−M_ij` decay [7]) | NO (mean/RMS only) | **YES (FS pool + homeostasis + per-pair plastic; the bridge's actual machinery)** | **YES** | Fixes attempt-#2 instability → stable RMS~0.13; worst-pair improves modestly, NOT to 0.003 |
| **3** | **Fixed Ω = ΓᵀΓ balanced-net wiring** (Deneve–Machens [2]) | YES (analytic) but handed-in, not learned | YES (install fixed lateral inhibition) | **YES** | Cleanly tests "can spikes HOLD a whitening solution"; if it fails → confirms rate-spike wall decisively |
| 4 | **PV+SST+VIP microcircuit** (Pfeffer [15] / Pi [16]) | Partial (SST-dendritic is the right analog) | Motif yes; computational power needs dendrites (absent) | No (1–2 wk build) | Cleaner stable *mean* decorrelation; worst-pair persists (no dendrites) |
| 5 | **Local dendritic balance** (Mikulasch–Priesemann [4]) | **YES — the literature's actual worst-pair→whitening mechanism** | **NO — requires dendritic compartments the point-neuron bridge lacks** | No (multi-month sim/ rewrite) | The named escape; out of scope; confirms the boundary is a point-neuron limit |
| — | Divisive normalization (§4); predictive coding (§8) | No / graded-residual | Partial | — | Wrong half / SNR wall; components only |
| — | Retinal graded whitening (§5) | YES but **graded/pre-spike by design** | Already done as numpy-ZCA pre-stage (biologically correct) | — | The honest home of the operation: analog, not spiking |

---

## 11. HONEST VERDICT

**The deep research did NOT find a local-rule spiking mechanism that resolves the worst pair to ZCA's 0.003 on the
project's point-neuron substrate. It found three convergent things, all of which CORROBORATE the project's boundary
while sharpening it into translatable science:**

1. **The right TARGET is whitening (covariance→identity), not decorrelation — and the project's three failed attempts
   are all the wrong *family* (competitive/sparse-coding = decorrelate-by-sparsify), which by the authors' own
   reporting only ever reduces the MEAN/RMS, never the worst pair, and which ALWAYS pre-whitens its input.** SAILnet,
   King–Zylberberg, and the learned-balance spiking-V1 model all reduce mean correlation on **already-whitened** input
   and report skewed worst-pair tails [3][5][6]. This is the project's exact result, three times over in the
   literature.

2. **The peer-reviewed mechanism that DOES convert mean-decorrelation into pairwise-WHITENING in spiking nets is LOCAL
   DENDRITIC BALANCE [4], and its central theorem is the project's own diagnosis: a point neuron with a single global
   inhibitory pool CANNOT whiten correlated inputs; dendritic compartmentalization is the necessary ingredient.** The
   bridge is point neurons. So the worst-pair limit is, precisely and citably, a **point-neuron limit** — not a tuning
   miss, not a connectivity-arrangement miss (attempt #3 confirmed), not an interneuron-count miss. The escape
   (dendrites) is a multi-month substrate change and an owner decision.

3. **Whitening's variance-equalization / common-mode-cancellation step is, like the project's already-proven OPPONENCY
   wall, fundamentally a GRADED/ANALOG / pre-spike (or multi-timescale-gain) operation.** The retina whitens graded,
   pre-spike, for an SNR reason [14]; every spiking cortical model imports that graded-whitened input rather than
   whitening raw input in spikes [3][5][6]; the one mechanism that whitens with interneurons does it by **gain
   modulation** (divisive, graded), explicitly *because synaptic plasticity is too slow/irreversible* [9][10]. **The
   project's numpy-ZCA pre-stage is therefore biologically FAITHFUL — it is the retina/LGN graded-whitening stage —
   not a cheat to be apologized for.** The honest architecture is: **graded whitening pre-stage (analog/ZCA/divisive)
   → spiking binding/cleanup downstream** (which the project has validated at 100%).

**Does anything genuinely HELP (vs confirm the boundary)?** Two things are worth a cheap test before the boundary is
declared final, because the project's prior NEGATIVES tested *weaker* versions of them:

- **(a) The stable-fixed-point lateral rule (#2)** — the project tested the *unstable* naive Földiák anti-Hebbian; the
  *stable* form (add a `−M_ij` / `c_ij−p²` fixed-point term, separate Dale FS pool, homeostasis) is a real missed
  upgrade that will at least give *stable mean-decorrelation done with the correct biological rule* and may push the
  worst pair below 0.9. Cheap, on-bridge, low-risk. **Expect: better mean, residual worst-pair — a partial win, not a
  resolution.**
- **(b) Gain-modulating adaptive whitening (#1)** — the only mechanism that targets the worst pair by design and that
  diagnoses the project's synaptic-plasticity failure. **Mandatory numpy-first de-risk:** run the Duong rate algorithm
  on the exact codebook (it provably whitens), THEN test whether a spiking/Dale/point-neuron gain version preserves it.
  **Expect: numpy whitens; spiking-gain survival is genuinely open** — this is the one test whose negative would be
  newly informative (the gain route specifically hitting the graded-stage wall) and whose positive would be a genuine
  on-bridge whitening GO.

**Most likely outcome, stated plainly:** both cheap tests improve the *mean* and the *stability* but leave a residual
worst pair above ZCA's 0.003, confirming — now with the full mechanistic literature behind it — that **all-pairs
whitening of strongly-correlated codes is a graded/analog (or dendritic, or fixed-Ω-wiring) operation, and rate-coded
spiking competition on point neurons cannot reach it.** That is a clean, well-cited, biology-translatable boundary, and
per the project's top-level goal (honest negatives under strict biology ARE the deliverable), it is a *result*, not a
failure: **the cortex's dendritic computation and graded pre-spike whitening are load-bearing for VSA-grade
decorrelation; a point-neuron E/I spiking substrate with a single (or topographic, or even multi-type-but-somatic)
interneuron class provably cannot substitute for them.** The numpy-ZCA stage is the faithful analog whitening, and the
recommended architecture is graded-whitening-pre-stage → spiking-binding-downstream.

---

## 12. The single highest-value next action (if the owner pursues this)

Run the **fixed Ω = ΓᵀΓ test (shortlist #3)** FIRST and cheaply: install the analytic balanced-network whitening
solution as fixed lateral inhibition among IT neurons (Γ = raw grounded codes) and measure worst-pair coherence in
spikes. It is the cleanest possible separation of the two hypotheses:
- If even the **analytic** whitening matrix, handed in as wiring, cannot hold the worst pair down in spikes → the
  boundary is the **rate-coded spiking substrate itself** (decisive, matches opponency), and no local rule can ever
  win → accept numpy-ZCA graded pre-stage as final.
- If the fixed Ω **does** hold the worst pair down in spikes → the substrate CAN represent a whitening solution, and
  the gap is purely *learning it locally* → escalate to the numpy-first gain-modulation de-risk (#1) and the stable-
  fixed-point rule (#2).

This one test costs ~an afternoon, uses only the existing brain-region framework + fixed lateral inhibition (no
`sim/` edits, no plasticity), and routes every subsequent decision.

---

## References

[1] [Predictive Coding of Dynamical Variables in Balanced Spiking Networks](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003258) (Boerlin, Machens, Denève, 2013, PLOS Comp Biol).
[2] [Efficient codes and balanced networks](https://www.nature.com/articles/nn.4243) (Denève & Machens, 2016, Nature Neuroscience).
[3] [Emergence of sparse coding, balance and decorrelation from a biologically-grounded spiking neural network model of learning in V1](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013644) (2025, PLOS Comp Biol; bioRxiv 2024.12.05.627100).
[4] [Local dendritic balance enables learning of efficient representations in networks of spiking neurons](https://www.pnas.org/doi/10.1073/pnas.2021925118) (Mikulasch, Rudelt, Priesemann, 2021, PNAS; arXiv:2010.12395).
[5] [A Sparse Coding Model with Synaptically Local Plasticity and Spiking Neurons (SAILnet)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002250) (Zylberberg, Murphy, DeWeese, 2011, PLOS Comp Biol).
[6] [Inhibitory Interneurons Decorrelate Excitatory Cells to Drive Sparse Code Formation in a Spiking Model of V1](https://www.jneurosci.org/content/33/13/5475) (King, Zylberberg, DeWeese, 2013, J Neurosci; PMC6705060).
[7] [Optimization theory of Hebbian/anti-Hebbian networks for PCA and whitening](https://arxiv.org/abs/1511.09468) (Pehlevan & Chklovskii, 2015).
[8] [Normalization as a canonical neural computation](https://www.nature.com/articles/nrn3136) (Carandini & Heeger, 2012, Nature Reviews Neuroscience).
[9] [Adaptive Whitening in Neural Populations with Gain-modulating Interneurons](https://proceedings.mlr.press/v202/duong23a.html) (Duong, Lipshutz, Heeger, Chklovskii, Simoncelli, 2023, ICML; arXiv:2301.11955).
[10] [Adaptive whitening with fast gain modulation and slow synaptic plasticity](https://arxiv.org/abs/2308.13633) (Duong, Lipshutz, et al., 2023, NeurIPS).
[11] [Dendritic predictive coding: A theory of cortical computation with spiking neurons](https://arxiv.org/abs/2205.05303) (Mikulasch, Rudelt, Wibral, Priesemann, 2022/2023).
[12] [Where is the error? Hierarchical predictive coding through dendritic error computation](https://www.cell.com/trends/neurosciences/fulltext/S0166-2236(22)00186-2) (Mikulasch et al., 2022, Trends in Neurosciences).
[13] [Optimal Whitening and Decorrelation](https://arxiv.org/pdf/1512.00809) (Kessy, Lewin, Strimmer, 2016) + PCA/ZCA whitening references (whitening = decorrelation + variance equalization; ZCA = symmetric whitening closest to identity).
[14] [What Does the Retina Know about Natural Scenes? / retinal whitening theory](https://direct.mit.edu/neco/article/4/2/196/5632/What-Does-the-Retina-Know-about-Natural-Scenes) (Atick & Redlich, 1992, Neural Computation); [Can the theory of "whitening" explain center-surround RFs?](https://pmc.ncbi.nlm.nih.gov/articles/PMC1575921/) (Graham, Chandler, Field); [Spatial whitening in the retina may be necessary for V1 to learn a sparse representation](https://www.biorxiv.org/content/10.1101/776799v1.full.pdf) (bioRxiv).
[15] [Inhibition of inhibition in visual cortex: the logic of connections between molecularly distinct interneurons](https://pubmed.ncbi.nlm.nih.gov/23817549/) (Pfeffer, Xue, He, Huang, Scanziani, 2013, Nature Neuroscience).
[16] [Cortical interneurons that specialize in disinhibitory control](https://www.nature.com/articles/nature12676) (Pi, Hangya, Kvitsiani, Sanders, Huang, Kepecs, 2013, Nature).
[17] [Cortical networks with multiple interneuron types generate oscillatory patterns during predictive coding](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013469) (2025, PLOS Comp Biol) + [A Computational Analysis of the Function of Three Inhibitory Cell Types in Contextual Visual Processing](https://pmc.ncbi.nlm.nih.gov/articles/PMC5403882/) (2017, Front Comput Neurosci).
[B] [Canonical microcircuits for predictive coding](https://pubmed.ncbi.nlm.nih.gov/23177956/) (Bastos, Usrey, Adams, Mangun, Fries, Friston, 2012, Neuron); Rao & Ballard (1999, Nature Neuroscience).
[Rozell] [Sparse coding via thresholding and local competition in neural circuits (LCA)](https://pubmed.ncbi.nlm.nih.gov/18439138/) (Rozell, Johnson, Baraniuk, Olshausen, 2008, Neural Computation).

## Project cross-references (internal)
- `research/findings/2026-06-06-A-spiking-decorrelation-mean-GO-worstpair-BOUNDARY.md` (the three failed attempts)
- `research/findings/2026-06-05-B-opponency-rate-coded-SNR-wall-CONFIRMED.md` (the parallel graded-stage boundary)
- `research/findings/2026-06-05-spiking-opponency-literature-synthesis.md` (Kandel Ch 22 graded-pre-spike argument)
- `research/findings/2026-05-31-foldiak-learned-decorrelation-BOUNDARY-...md` (the numpy-Földiák over-sparsification)
- `research/findings/2026-06-04-spine-item2-spiking-cleanup-needs-decorrelation.md` (why the cortex must decorrelate)
- `research/findings/2026-06-04-v-multimodal-grounding-decorrelation-unifies.md` (ZCA → 100% composition baseline)

*Discipline: research-only, no `sim/` edits, no probes run. Every mechanism extracted from the primary source via
WebFetch/WebSearch with corroboration where the fetcher editorialized (the SAILnet/King rules cross-verified against
two sources each; the Pehlevan saddle-point reason and Duong gain-modulation motivation quoted from the papers). The
bio-research MCP tools (consensus/biorxiv) were not callable in this agent thread; WebSearch+WebFetch over the open
literature (PLOS, J Neurosci, PNAS, Nature, arXiv, ICML/NeurIPS proceedings) supplied equivalent primary-source
coverage.*
