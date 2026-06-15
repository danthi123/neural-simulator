# The W↔M coupling of similarity-matching on a spiking substrate — deep-research scope (CYCLE 79 crux)

**Date:** 2026-06-15. **Type:** READ-ONLY deep-research + reference-catalog review at a precisely-diagnosed
roadblock (the standing directive: deep research before committing the build). NO `sim/` edits, NO GPU jobs.
**Deliverable:** confirmed/refined diagnosis → ranked options (cheapest-first) → verdict + cheap-first numpy
de-risk + anti-cheat → reusable machinery. **This is the ONE open piece of an otherwise-resolved arc** (the
spiking whitening front-end is SHIPPED; the learned low-rank cortex is the escape; the joint STDP-W +
graded_lateral-M does **not converge on the bridge** — every attempt gave Pearson +0.025–0.131, eff-rank 2.4–4.8,
the W rank-1-collapses).

---

## 0. The question, answered in one paragraph

**How is the W↔M coupling of similarity-matching realized on a spiking substrate, such that the feedforward W
learns the LOW-RANK subspace (not rank-1 collapse) because the lateral M decorrelates the activity that drives
W's plasticity?** **Answer (triply confirmed — two SM theory papers + one published spiking realization + the
bridge code):** the coupling is **a TIMESCALE-SEPARATED loop through the settled, M-decorrelated output `y`**.
The feedforward weights must learn from the *equilibrated* activity `y = (I+M)⁻¹Wx` (AFTER the recurrent lateral
M has decorrelated it) — **and the lateral M must adapt on a FASTER timescale than W**, so the decorrelation is
already in place when W's Hebbian update fires. The bridge breaks BOTH halves of this coupling: (i) its STDP-W is
a **spike-timing** rule (`Δw = f(Δt)·(w_max−w)`) that is **not** the SM Hebbian outer-product `ΔW ∝ y·xᵀ` of the
decorrelated activity — STDP potentiates *co-active* hub→cortex pairs by a fixed `(w_max−w)` increment regardless
of the decorrelated magnitude, so the rank-1 collapse is driven by the rule's *form*, not just its input; and
(ii) there is **no timescale separation** — STDP-W and the analog-M run at the same per-step cadence with no
settle and no fast-M-first ordering. The fix is to make the feedforward plasticity an **activity/rate outer-
product driven by the post-lateral (decorrelated) cortex activity**, with the M learning **faster** than W.

---

## 1. DIAGNOSIS — confirmed and sharpened (with the exact update order)

### 1a. What the SM coupling IS (canonical Pehlevan-Chklovskii)

The online similarity-matching algorithm
([Pehlevan, Sengupta & Chklovskii 2018, *Neural Comput.* 30:84 / arXiv:1703.07914, "Why Do Similarity Matching
Objectives Lead to Hebbian/Anti-Hebbian Networks?"](https://arxiv.org/abs/1703.07914)) eliminates the activity
variables from the SM objective to get a **min-max (saddle) problem in W and M**, which the paper explicitly
frames as "the adversarial relationship of Hebbian and anti-Hebbian learning rules." Per presentation of input
`x`:

1. **Neural dynamics settle** to the fixed point `y = (I+M)⁻¹ W x` (in practice iterate `y ← y + η(Wx − My)` to
   convergence). `M` is the **recurrent anti-Hebbian lateral**; `Wx` is the feedforward drive; the settled `y` is
   the **decorrelated** output.
2. **Feedforward W (Hebbian) learns from the settled `y`:** `ΔW ∝ y·xᵀ` (the project's numpy uses the Oja
   self-normalizing variant `ΔW = lr·(y·xᵀ − y²·W)`).
3. **Lateral M (anti-Hebbian) also learns from the settled `y`:** `ΔM ∝ y·yᵀ − M` (diagonal held at 0).

**The coupling is `y`.** Both W and M read the **same settled, decorrelated `y`** — that is the entire mechanism.
This is *verified in the project's own validated numpy SM*: `learn_simmatch`
(`research/runners/learned_graded_cortex_fair_test.py:185-192`) and `simmatch_converged`
(`research/runners/_l1_simmatch_converges_check.py:54-58`) both do, per input:
```python
ff = W_ff @ x                                   # feedforward drive
for _ in range(settle_steps): y = 0.5*y + 0.5*(ff - M @ y)   # SETTLE to the M-decorrelated output
W_ff += lr_ff * (np.outer(y, x) - (y**2)[:,None]*W_ff)       # W learns from the DECORRELATED y  ← coupling
dM = np.outer(y, y) - M; np.fill_diagonal(dM,0); M += lr_m * dM   # M learns from the same y
```
The W update's `np.outer(y, x)` uses the settled, M-decorrelated `y` — **this is the line the bridge breaks.**

### 1b. The TIMESCALE-SEPARATION requirement (the load-bearing new finding)

The coupling is not merely "W reads `y`"; the SM convergence proof **requires a timescale hierarchy**.
[**Similarity Matching Networks: Hebbian Learning and Convergence Over Multiple Time Scales (2025),
arXiv:2506.06134**](https://arxiv.org/abs/2506.06134) states the network as three coupled ODEs (its eqs 7a–c):

| timescale | dynamics | rule |
|---|---|---|
| **fast (neural)** `ε₁ε₂ Ẏ = (4/T)(WX − MY)` | settle to `Y⋆(W,M)=M⁻¹WX` | (settles first) |
| **intermediate (lateral)** `ε₂ Ṁ = −2M + (2/T)YYᵀ` | **anti-Hebbian** | M learns from settled Y |
| **slow (feedforward)** `Ẇ = −4W + (4/T)YXᵀ` | **Hebbian** | W learns from settled Y |

with the explicit ordering **`0 < ε₁ε₂ ≪ ε₂ ≪ 1`** — *"neural activity settles fastest, then lateral weights,
then feedforward weights evolve slowest."* The paper's Theorems 1–3 chain **sequential convergence at each
level** (neural settles → M contracts to `M⋆(W)` → W converges), and it is explicit that **the Hebbian update
`Ẇ ∝ YXᵀ` employs the equilibrated Y after lateral inhibition via M has decorrelated the activity** — and that if
the separation is violated, *"learning from un-decorrelated activity would occur — corrupting the embedding."*
**This is precisely the bridge's failure: no settle, no M-faster-than-W ordering → W learns from un-decorrelated
spiking → rank-1 collapse.**

This is corroborated by the *spiking* realization (next section), which independently arrives at the same
"recurrent learning must be faster" principle.

### 1c. Why the bridge breaks it — airtight, code-localized

The diagnosis in AUTONOMOUS_STATE CYCLE 79 is **confirmed and made precise** by reading the bridge:

1. **`graded_lateral` DOES decorrelate the analog drive AND it DOES reach the spiking.** `_graded_lateral_*`
   (`sim/bridge.py:1827-1884`) reads `a = clip((v−baseline)/scale, 0, 1)` (the analog membrane), and the
   inhibition `-(M @ a)·gain` IS added to `total_input_current_pA[start:end]` **before the spike threshold**
   (`sim/bridge.py:5902-5907`). So the cortex *spikes* are driven by the decorrelated current `Wx − Ma`. The
   M-learning `ΔM ∝ ⟨aaᵀ⟩ − I − λM` (`:1855-1884`) is the correct anti-Hebbian rule. **This half is fine.**
2. **But the bridge STDP-W is the WRONG RULE CLASS — and it does NOT read `y`.** The feedforward W learning
   (`sim/bridge.py:6479-6529`) is a **pure spike-timing rule**: `delta_t = t_post − t_pre` (`:6501`) →
   `fused_stdp_weight_update(Δt, w, a_plus, a_minus, …, w_max)` (`:6520`), a soft-bound STDP
   `Δw_LTP ∝ A_plus·(w_max−w)·exp(−Δt/τ)`. The Hebbian path (`:6437-6450`) is the same shape: `Δw =
   hebbian_lr·(hebbian_max − w)` on co-firing pairs. **Neither is `ΔW ∝ y·xᵀ − y²·W`.** Both read **binary spike
   events** (`cp_last_spike_time`, `cp_prev_firing_states`, `fired_this_step`), NOT the decorrelated analog `a`.
   Consequence: even though the spikes are decorrelated-*driven*, the *weight update* potentiates **every
   co-active pre→post pair toward the same ceiling `w_max`** — a force that (a) is largely independent of the
   decorrelated *magnitude* the SM rule's `y·xᵀ` depends on, and (b) drives all cortex rows toward the dominant
   (top-PC) hub profile = the **rank-1 / Hebbian collapse** (eff-rank 1.5–4.8 observed). The lateral changes
   *which/when* neurons spike, but with a `(w_max−w)` timing rule the decorrelation does not become the
   *learning signal*. **This is the broken coupling, exactly.**
3. **And there is NO timescale separation.** The bridge runs the M update and the STDP-W update **in the same
   `_run_one_simulation_step`**, one membrane step per call, no recurrent settle (`graded_lateral` is the
   **one-step** `-(M@a)` feedforward-inhibition approximation of the settle, not the iterated `y=(I+M)⁻¹Wx`),
   and no "M converges faster than W" ordering. The 2025 convergence theorem says this ordering is **required**.

**⇒ Refined crux:** the bridge breaks the coupling in TWO places — (i) the feedforward rule is spike-timing-STDP,
not the SM activity-outer-product on the decorrelated `y`; (ii) no settle + no M-faster-than-W timescale
separation. **Fixing (i) is necessary and is the lead;** (ii) is the secondary tuning that the convergence proof
also requires.

---

## 2. RANKED OPTIONS (cheapest-first) to COUPLE W↔M on the bridge

### (a) **Drive the feedforward plasticity from the DECORRELATED post-lateral activity, as a rate/activity outer-product — NOT spike-timing STDP.** ★ RECOMMENDED LEAD.

- **Biology + citation.** This is exactly what every spiking SM realization does. The published, canonical
  integrate-and-fire SM ([**Pehlevan 2019, arXiv:1902.01429, "A Spiking Neural Network with Local Learning Rules
  Derived From Nonnegative Similarity Matching"**](https://arxiv.org/abs/1902.01429)) derives the **same membrane
  fixed point `Y⋆ = M⁻¹WX`** (lateral inhibition applied sub-threshold) and a **feedforward Hebbian rule on
  low-pass-filtered spike trains (rates)** — *not* a Δt-STDP rule. The **spike-coding** realization
  ([Brendel, Bourdoukan, Vertechi, Machens & Denève 2020, *PLOS Comput. Biol.* 16(3):e1007692, "Learning to
  represent signals spike by spike", PMC7135338](https://doi.org/10.1371/journal.pcbi.1007692)) is even more
  literal and directly relevant: its feedforward rule is **`ΔF_ij = α·x_j·o_i`** (eq 7) — a Hebbian **outer-
  product of pre-input × post-SPIKE**, and its recurrent rule **`ΔΩ_ik = −β·o_k·V_i − μ·δ_ik·o_i`** (eq 6) reads
  the **post-synaptic membrane voltage** (the coding error / decorrelated drive). Both spiking papers use a
  rate/spike outer-product feedforward rule on the *balanced* (post-inhibition) activity; **neither uses spike-
  timing STDP.** (According to PubMed, Brendel et al. 2020, [DOI](https://doi.org/10.1371/journal.pcbi.1007692).)
- **Bridge realization.** Replace the STDP-W on the hub→cortex pathway with an **activity-outer-product Hebbian
  update driven by the cortex's post-lateral analog activity `a`** (the very `a = clip((v−baseline)/scale,0,1)`
  that `graded_lateral` already computes and that already includes the `−(M@a)` inhibition in the membrane). I.e.
  `ΔW ∝ a_post · a_pre − (a_post²)·W` (Oja self-normalizing), computed from the *decorrelated* `a_post`. This is
  the literal `y·xᵀ` coupling: W learns from the M-decorrelated output, so M's decorrelation *shapes* W's
  learning → no rank-1 collapse. Note the project's CYCLE-58 `_l1_phaseA_end_to_end_spiking.py` already showed a
  **rate/activity-outer-product** Hebbian feedforward (its `bounded_Hebbian`, `W_ff += lr·outer(y_spk, x_in)`)
  reaches the ceiling in the *normalized* regime — the missing piece on the bridge is feeding that outer-product
  the **decorrelated `a`** (and the Oja/bound normalization), not the raw co-firing.
- **Edit class.** **A GUARDED `sim/` edit** (owner-gated, byte-level diff review) — a new default-off plasticity
  mode on a tagged pathway that, when active, computes `ΔW` from the per-region post-lateral analog activity
  instead of (or alongside) the spike-timing STDP. It is small and additive (it reuses the `graded_lateral`
  slice + `a`), mirroring the existing `input_mean_adapt` / `graded` guarded edits (default-off, guarded on a
  `is None` sentinel, true pre/post A/B byte-identity). **It is the smallest edit that fixes the rule-class
  mismatch**, which §1c shows is the primary breakage. (A config/runner-only version is NOT possible — the
  bridge has no rate/activity-outer-product feedforward plasticity primitive; grep confirms only STDP-Δt +
  `(w_max−w)` Hebbian on spikes exist.)
- **Risk.** (i) It is a new plasticity rule (guarded edit) — but small, additive, default-off, and de-riskable
  in numpy first (§3). (ii) Still needs the M-faster-than-W timescale separation (option b) to converge; treat
  (a)+(b) as one build. (iii) The bridge spiking realization adds the usual losses (the front-end already lost
  numpy +0.31 → bridge +0.155), so the result may be a *characterized marginal* rather than full host — but it
  is the validated, biologically-canonical direction and the honest result is the deliverable.

### (b) **Enforce the TIMESCALE SEPARATION: M faster than W + a multi-step settle before the W update.** ★ REQUIRED CO-FACTOR with (a).

- **Biology + citation.** Both the 2025 SM convergence paper (the explicit `ε₁ε₂ ≪ ε₂ ≪ 1` ordering;
  [arXiv:2506.06134](https://arxiv.org/abs/2506.06134)) and Brendel-Denève (verbatim: *"the recurrent weights
  remain plastic on a faster time scale … Since the recurrent plasticity rules are faster, they win this
  competition, and the network remains in a balanced state"* — i.e. the fast lateral keeps the greedy
  "unbalancing" feedforward Hebbian in check) require the lateral to adapt **faster** than the feedforward, and
  the neural activity to **settle** before the feedforward update.
- **Bridge realization.** Two cheap knobs + one slightly-deeper one: (1) set `graded_lateral_lr` ≫ the feedforward
  plasticity rate (M faster than W — config-only); (2) interleave the bridge's existing **multi-step settle**
  (`rf_resonate_steps` has a precedent for a fast inner loop; or simply hold the input current for K sub-steps so
  the `−(M@a)` inhibition equilibrates before reading `a` for the W update) — this upgrades the one-step
  `graded_lateral` toward the iterated `y=(I+M)⁻¹Wx` (the research's option-d settle, but as a *few-step* settle,
  not a full inversion). (3) Update the feedforward W only **after** the settle, from the settled `a`.
- **Edit class.** (1) config-only; (2)+(3) runner/streaming-protocol if a few-step settle suffices, or a small
  guarded `sim/` settle loop if it must be inside the step. **Try the config knob (M-faster) + a runner-level
  few-step current-hold first.**
- **Risk.** Low/medium. The full matrix-inverse settle is numerically finicky (CYCLE-79: anti-Hebbian M can
  diverge → singular `(I+M)`), so use the **damped iterative** few-step settle (`y ← 0.5y + 0.5(Wx−My)`, the
  validated numpy form) + the existing `−λM` bounded fixed point, NOT a literal inverse.

### (c) **The Brendel-Denève joint spike-coding rule (the most literal spiking realization).** Stronger, heavier alternative to (a)+(b).

- **Biology + citation.** Brendel et al. 2020 (PMC7135338, [DOI](https://doi.org/10.1371/journal.pcbi.1007692)):
  each neuron's **membrane = the representation/prediction error** `V_i = Σ_j F_ij(x_j − ŷ_j)` (eq 4); a spike
  fires only when it *reduces* that error; the feedforward rule `ΔF = α·x·o` (eq 7) and the recurrent rule
  `ΔΩ = −β·o·V − μ·diag` (eq 6) are learned **jointly**, with the recurrent learning **faster**. Degeneracy/
  collapse is prevented because *"the first [neuron to spike] immediately inhibits (resets) the others"* and
  neurons with overlapping feedforward weights **develop anti-correlated recurrent connections** → forced
  specialization. The optimal recurrent connectivity is `Ω = −Fᵀ D` (feedforward × decoder), *"lateral inhibition
  proportional to shared feedforward inputs weighted by their decoding importance."*
- **Why it's heavier than (a)+(b).** It is the *full* EI-balanced predictive autoencoder — it needs the
  membrane-as-error reset dynamics (the recurrent reconstruction fed back as inhibition every spike), reciprocal
  E↔I plasticity, and the `−β·o·V` voltage-dependent recurrent rule — a larger guarded `sim/` build than (a)'s
  single activity-outer-product feedforward rule. But it is the **most biologically literal** answer to the exact
  question ("how do feedforward + recurrent couple in spikes to avoid collapse"), and it is the **same coupling
  principle** (fast recurrent keeps greedy feedforward in check; both read the post-inhibition activity/voltage).
- **Edit class.** A larger guarded `sim/` edit (membrane-error reset + voltage-dependent recurrent plasticity).
  **Verdict: the principled fallback if (a)+(b) — the SM activity-outer-product + timescale separation — proves
  too weak; do not lead with it (heavier, and (a)+(b) is the same coupling more cheaply).**

### (d) **Stronger `graded_lateral` coupling into the spiking (gain ↑).** INSUFFICIENT alone — already falsified.

- Raising `graded_lateral_gain_pA` makes the inhibition bite harder into the membrane, but CYCLE-78 already
  swept this (gain 40 vs 300): low gain fixed over-suppression but eff-rank stayed 2.9 (no decorrelation of the
  *learned* code); high gain over-suppressed. **The gain is not the coupling** — the coupling is the *rule that
  reads the decorrelated activity*, which gain does not change. Keep `graded_lateral` (it is the verified M and
  the analog-domain decorrelation that sidesteps the Mikulasch-Priesemann rate-code wall), but the missing piece
  is option (a): make W *learn from* `a`, not just be *driven by* `−(M@a)`.

---

## 3. VERDICT + the cheap-first de-risk

**VERDICT.** The W↔M coupling is **unambiguous and triply confirmed**: the feedforward W must learn an **activity/
rate outer-product of the M-DECORRELATED (settled, post-lateral) output** (`ΔW ∝ y·xᵀ − y²·W`), with the **lateral
M adapting on a FASTER timescale** so the decorrelation precedes W's update. The bridge breaks it because its
feedforward plasticity is **spike-timing STDP with a `(w_max−w)` gate** (the wrong rule class — it reads spike
*timing*, not the decorrelated *activity*) **and** there is **no settle / no M-faster-than-W ordering**. The
recommended build is **(a) a guarded, default-off feedforward plasticity mode that computes `ΔW` from the cortex's
post-`graded_lateral` analog activity `a` (Oja-bounded outer-product) + (b) the timescale separation
(`graded_lateral_lr` ≫ W-rate + a damped few-step settle before the W update)**. `graded_lateral` stays as the
verified M; the new piece is *coupling W to the decorrelated `a` it produces*. Months-scale dendrites remain
**unnecessary** (the front-end + the SM lateral, not dendrites, are the path — re-confirmed).

**THE CHEAP-FIRST DE-RISK (numpy, no `sim/` edits, build once — the controller's next move BEFORE any guarded
`sim/` edit):** add ONE arm to the existing L1 numpy battery that mimics the bridge's *actual* mechanism — a
**one-step (then few-step) lateral** + a feedforward rule that learns from the **decorrelated** output — and
prove the COUPLING is what fixes it:

> On the **centered + E/I-projected** real input (`Xppmi_c`, the front-end the bridge already has), run the
> **bridge-faithful** SM where (i) `a = relu(Wx − M·a)` is the **one-step** (then K-step damped) post-lateral
> analog drive (matching `graded_lateral`, NOT the full inverse), (ii) **W learns from the decorrelated `a`:**
> `W += lr_W·(outer(a, x) − (a²)·W)`, (iii) **M learns faster:** `M += lr_M·(outer(a,a) − βI − λM)` with
> `lr_M ≫ lr_W`. The **load-bearing comparison** (this is the de-risk's whole point):
> - **COUPLED** (W learns from the decorrelated `a`) — must REACH the low-rank host (+0.44) and eff-rank ≫ 1.5
>   (rising toward k≈8), beating random (+0.31 numpy / +0.155 bridge) — **THE GATE**;
> - **BROKEN-COUPLING control** (W learns from the **RAW** `Wx` co-firing, M still on; everything else identical)
>   — must **COLLAPSE to rank-1** and trail (this reproduces the bridge's STDP-W failure in numpy, proving the
>   coupling — not the lateral's mere presence — is the fix);
> - **lateral-only on a random full-rank W** (the CYCLE-79 result, +0.323/eff-rank 44) — must over-whiten / NOT
>   reach low-rank (proving the lateral *alone* is insufficient; the LEARNED low-rank W is needed);
> - **timescale knob** — sweep `lr_M / lr_W ∈ {1, 5, 20}` and `settle K ∈ {1, 4, 10}`: confirm convergence
>   IMPROVES with M-faster + more settle (the 2025-paper prediction), and DEGRADES at `lr_M/lr_W = 1` (no
>   separation = the bridge's current regime).

A GO here (COUPLED reaches host + eff-rank rises; BROKEN-COUPLING collapses; separation helps) **de-risks the
guarded `sim/` edit and tells you the minimal form** (one-step vs few-step settle; the `lr_M/lr_W` ratio). A
NEGATIVE (even the coupled numpy one-step doesn't reach low-rank) says escalate the settle depth (toward option d)
BEFORE the `sim/` work. This is ~1 new function bolted onto `_l1_simmatch_converges_check.py` /
`learned_graded_cortex_fair_test.py` (reuse `build_real_corpus`, `ppmi_matrix`, `center_cols`, the E/I projection,
`_pearson_vs_Strue`, `heldout_generalization`, the eff-rank metric). CPU/numpy, minutes.

**Anti-cheat controls (reuse the battery already in the L1 runners):**
- **beats-random — THE gate.** The coupled learner must beat the frozen random projection (the bar all 5 prior
  bridge fixes failed). Non-negotiable.
- **learning-load-bearing — the coupling is the win, not just "adding inhibition."** The explicit BROKEN-COUPLING
  control (W-from-raw with M still on) MUST collapse to rank-1 while the COUPLED arm (W-from-decorrelated-`a`)
  reaches low-rank — this isolates that the *coupling through `y`* (not M's mere presence, which CYCLE-79 showed
  is insufficient) is load-bearing.
- **eff-rank ≫ 1.5.** The coupled code must have eff-rank rising toward the low-rank k (≈8), not the collapse
  signature (1.5–2.9). Rising eff-rank IS the mechanistic tell that the lateral is shaping W.
- **permuted-label** — `Pearson(cos, S_perm) ≈ 0` (structure not an artifact) — already in every L1 runner.
- **generalization (Fodor–Pylyshyn / held-out)** above chance — already in `heldout_generalization`.
- **no host shortcut (BRAIN-BASED-ONLY).** The decorrelated `a` must come from the on-substrate `−(M@a)` membrane
  inhibition (no host `XᵀX`, no host whitening, no host argmax in the neural path). The bridge primitive computes
  `a` and M from the membrane; the de-risk's numpy must mirror that one-step/few-step form (NOT a host matrix
  inverse). Per the standing standard, an honest NEGATIVE (the coupled spiking realization underperforms host) IS
  the deliverable — it maps what the point-neuron substrate can/can't do for the joint SM.

---

## 4. Reusable machinery (and the headline: `graded_lateral` is M; the missing piece is COUPLING W to its `a`)

| Need | Reusable machinery | Status |
|---|---|---|
| **Recurrent anti-Hebbian decorrelating lateral M** | `graded_lateral` (`sim/bridge.py:1827-1884`, `regions.py:188`, `config.py:381-398`): plastic K×K M, `ΔM ∝ ⟨aaᵀ⟩−I−λM`, `−(M@a)·gain` pre-threshold, **on analog membrane `a`** | **VERIFIED = the M; reuse directly.** |
| **The decorrelated activity `a` to drive W's learning** (the coupling signal) | `_graded_lateral_activity()` returns `a=clip((v−baseline)/scale,0,1)` (`sim/bridge.py:1827-1841`); already includes `−(M@a)` in the membrane (`:5902-5907`) | **reuse `a` as the `y` the new W rule reads** |
| Feedforward W plasticity that reads the decorrelated `a` (the SM `y·xᵀ−y²W`) | **none yet** — the bridge has only spike-timing STDP (`:6479-6529`) + `(w_max−w)` Hebbian on spikes (`:6437-6450`); **neither reads `a`** | **THE GUARDED `sim/` EDIT (option a)** — small, additive, default-off; numpy reference = `bounded_Hebbian`/`learn_simmatch` |
| Timescale separation (M faster than W) | `graded_lateral_lr` (config) ≫ the feedforward rate; the project's `−λM` bounded fixed point (`config.py:388`) | config-only knob; sweep in the numpy de-risk first |
| Damped few-step settle (one-step → iterated `y`) | the damped iteration `y←0.5y+0.5(Wx−My)` (validated numpy `learn_simmatch:187-188`); precedent fast inner loop `rf_resonate_steps` | runner/streaming few-step current-hold, or a small guarded settle loop |
| Per-feature mean-centering + E/I signed projection (the whitening front-end) | `enable_input_mean_adapt` (shipped byte-clean) + E/I balance (`_phaseB_input_mean_bridge.py --enable-ei`) | **shipped; reuse as the input to the coupled cortex** |
| Full Brendel-Denève spike-coding SM (option c, fallback) | none yet (membrane-as-error reset + `−β·o·V` recurrent rule = a larger guarded edit) | defer behind (a)+(b) |
| numpy SM ground truth (the validated coupling to mirror) | `learn_simmatch` (`learned_graded_cortex_fair_test.py:162`), `simmatch_converged` (`_l1_simmatch_converges_check.py:33`), the `_l1_phaseA_end_to_end_spiking.py` bounded-Hebbian arm | reuse verbatim for the de-risk |

**Catalog grounding (`sim-catalog/references/feature-catalog.md`):** **A.12** sparse decorrelated cortico-striatal
("decorrelation rule" — the closest learned-decorrelation kin); **E.05** lateral inhibition / center-surround
("decorrelates output", the retinal motif `graded_lateral` realizes); cluster **J** "sparse coding via
inhibition". **No catalog entry yet for a *learned anti-Hebbian SM lateral coupled to a feedforward rate-Hebbian
W*** — `graded_lateral` + the proposed coupled feedforward rule is the project's own realization of the
Pehlevan/Brendel-Denève motif (worth a future catalog entry once built).

---

### Citations (load-bearing)

- Pehlevan, Sengupta & Chklovskii 2018, *Neural Comput.* 30:84 / **arXiv:1703.07914** — SM ⇒ min-max in W (Hebbian)
  + M (anti-Hebbian); both learn from the settled `y=(I+M)⁻¹Wx`; the "adversarial" H/AH coupling.
  https://arxiv.org/abs/1703.07914
- **Similarity Matching Networks: Hebbian Learning and Convergence Over Multiple Time Scales, 2025,
  arXiv:2506.06134** — the THREE coupled timescales `0<ε₁ε₂≪ε₂≪1` (neural ≪ lateral ≪ feedforward); W's Hebbian
  uses the equilibrated Y; **violating separation → learning from un-decorrelated activity corrupts the
  embedding** (the bridge's exact failure). https://arxiv.org/abs/2506.06134
- **Pehlevan 2019, arXiv:1902.01429** — the canonical **integrate-and-fire** SM: lateral inhibition sub-threshold,
  membrane fixed point `Y⋆=M⁻¹WX`, feedforward Hebbian on **filtered (rate) spike trains** (NOT Δt-STDP).
  https://arxiv.org/abs/1902.01429
- **Brendel, Bourdoukan, Vertechi, Machens & Denève 2020, *PLOS Comput. Biol.* 16(3):e1007692, PMC7135338** — the
  literal **spiking joint-learning** SM: membrane=prediction error (eq 4); feedforward `ΔF=α·x·o` (eq 7, rate/spike
  outer-product); recurrent `ΔΩ=−β·o·V−μ·diag` (eq 6, reads membrane); **recurrent plasticity is FASTER and "wins"
  → keeps greedy feedforward in check**; collapse prevented by lateral reset + anti-correlated recurrent weights
  (forced specialization). (Per PubMed; [DOI](https://doi.org/10.1371/journal.pcbi.1007692).)
- Földiák 1990, *Biol. Cybern.* — local **anti-Hebbian feedback** decorrelates a code (the decorrelating lateral
  motif). https://link.springer.com/article/10.1007/BF02331346
- Project code (verified this pass): `sim/bridge.py:1827-1884` (`graded_lateral` = M on analog `a`), `:5902-5907`
  (M inhibition added pre-threshold), `:6479-6529` (STDP-W = spike-timing `f(Δt)·(w_max−w)`, reads NO `a` — the
  broken coupling), `:6437-6450` (Hebbian = `(w_max−w)` on spikes); `learned_graded_cortex_fair_test.py:185-192`
  + `_l1_simmatch_converges_check.py:54-58` (the numpy coupling: W & M both learn from the settled, decorrelated
  `y`); prior scope `2026-06-15-L1-SM-on-spiking-deep-research.md`; arc `2026-06-15-phaseB-spiking-cortex-WALL-
  rate-to-spike.md`; AUTONOMOUS_STATE CYCLE 79.
