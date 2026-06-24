# BURNDOWN Phase-3B — the DEEP dendritic learned binder: honest NEGATIVE (memorizes, does NOT generalize — even at the oracle capacity ceiling) (2026-06-24)

**The question (the ONE surviving untested hypothesis from the cortex/dendrite scoping):** does a **DEEP
(≥2 hidden-layer) learned binder with apical-basal CREDIT ASSIGNMENT** learn GENERALIZABLE
**multi-attribute** composition — the genuine FHRR-idealization residual (inventory **C-1 / H-3**)? The
cheap **single-layer** learned-dendritic-bind already came back NEGATIVE (`2026-06-19-dendritic-binding-toy-derisk.md`:
memorizes two-attribute 0.422 / generalizes **0.168** held-out, *below* the fixed FHRR primitive 0.261). The
single-layer had "nothing to credit-assign." The **DEEP regime — the one the literature says the apical-basal
dendrite is *designed* for** (Sacramento-Senn 2018; Payeur-Naud-Richards 2021: credit assignment needs HIDDEN
layers) — was **UNTESTED**. This is the cheapest decisive next de-risk, and it reuses the project's OWN deep
credit-assignment machine.

**Per the standing SURPASS directive:** the multi-attribute residual / FHRR-idealization boundary is accepted as
a characterized boundary ONLY after *this* de-risk survives the round. It survives → the boundary holds.

## Method (cheapest-first, STRICTLY CPU/numpy, NO `sim/` edit, reuse-by-import)

Extended the rigorous existing binding harness (`_phaseB_dendritic_bind_derisk.py` + the systematicity protocol
in `cortex_learned_binder_systematicity_probe.py`): leakage-free Fodor-Pylyshyn splits, memorization floor, the
fixed-FHRR positive control, R=4 / F=16, 3 leakage-free splits, 3 seeds — on the **production PPMI stream codes**
(correlated, between-cos mean 0.047 / max 0.618 = the real target regime). The ONE new arm: a **DEEP dendritic
binder** —

- **bind** = the MULTIPLICATIVE (sigma-pi / Hadamard) dendritic conjunction `(role@W_R) ⊗ (filler@W_F)` (the
  supralinear op a point neuron's single linear sum cannot form);
- **bundle** = `Σ_r bind(role_r, filler_r)` (the 3-way SVO superposition = the multi-attribute load);
- **unbind+cleanup** = the project's **`sim.dendritic_mlp.DendriticMLP`** (`[D_bind+R → 128 → 128 → F]`, ≥2
  HIDDEN layers) reading `[bundle ; onehot(query-role)] → softmax over F fillers`, trained by the committed
  **`urbanczik_senn_update`** local rule (`mode="local_correct"`) — **feedback alignment via the FIXED-RANDOM
  apical `B`, NO weight transport** (the brain-faithful path). 400 epochs × batch 64, one LR all seeds (no
  per-seed tuning).

Runner: `research/runners/_burndown_3B_deep_dendritic_binder_derisk.py`. Reuse-by-import: `DendriticMLP` /
`urbanczik_senn_update` (the deep credit-assignment machine), `DendriticSigmaPiBinder` (the single-layer prior
NEGATIVE, run same-run as a control), `MultFHRRBinder` (the learned-linear control), `fhrr_*` (the ceiling),
`make_systematicity_splits` / `MemorizationLookup` (the protocol).

## Result (3-seed mean; the DECISIVE metric is held-out **generalization**, not raw recall)

| arm | single-attr held | 3-attr bundle TRAIN | **3-attr bundle HELD-OUT** |
|---|---|---|---|
| **DEEP dendrite — feedback alignment (TEST)** | 0.000 | 0.345 | **0.007** (train→held gap **+0.338**) |
| **DEEP dendrite — ORACLE true-gradient (capacity ceiling, fenced)** | — | 0.98* | **0.007** |
| single-layer dendrite (prior NEGATIVE, same harness) | 0.50 | — | 0.168 |
| LESION = additive bind, SAME deep MLP (the dendrite anti-cheat) | 0.85 | 0.33 | 0.169 |
| permuted role↔filler (must collapse) | — | — | 0.007 |
| memorization-floor (must ≈ chance) | — | — | 0.000 |
| chance (1/F) | — | — | 0.062 |
| **FHRR fixed ±1 primitive (production reference / ceiling)** | — | — | **0.228** |

\* the oracle's bundle-train reaches **0.984** (single split, 600 ep) — i.e. it FITS the training combos almost
perfectly while held-out stays at **0.000**: the pure memorization signature.

**Verdict: NEGATIVE — comprehensively, and certified by the capacity ceiling.**

1. **The DEEP dendrite (feedback alignment) does NOT generalize:** held-out **0.007** (≈ chance 0.062 — actually
   below it), against train **0.345**, a +0.338 gap = the memorization signature. No better than the single-layer
   (0.168); if anything worse.
2. **The decisive certification — the ORACLE (hand-derived TRUE gradient, full capacity) ALSO fails:** held-out
   **0.007** with train fit up to **0.984**. So the NEGATIVE is **NOT** "the local Urbanczik-Senn rule can't
   credit-assign it" and **NOT** "under-trained / mis-scaled" — **even an unlimited-capacity, true-gradient deep
   network cannot generalize this task on these codes.** The failure is the **task/representation**, not the
   learning rule and not my architecture (a fairness probe with raw atoms + 256×256 + oracle also gave held-out
   **0.000**).
3. **The harness is sound** (the FHRR fixed-primitive positive control = **0.228** = the cited ~0.261 reference —
   the harness DOES detect working bundling), and the controls behave (permuted 0.007, mem-floor 0.000).

### Why a *learned* deep read-out generalizes *against* the held-out answer (the mechanism)

A softmax read-out conditioned on the query-role learns, from training, the **per-role allowed-filler
distribution**. On a held-out fact the correct filler for a role-slot is *precisely the one filler that role
NEVER saw paired in training* — so the role-conditioned classifier has learned to **down-weight exactly that
filler**. It generalizes in the wrong direction. The fixed ±1/FHRR algebra wins (0.228) **because it does not
learn** those priors: its inverse is structural and role-symmetric, so there is nothing to overfit. Depth does
not help — it makes the per-role prior *sharper* (held-out 0.007 < the single-layer 0.168 < the fixed FHRR 0.228).

### The lesion nuance (recorded honestly)

The additive LESION (0.169) is **above** the multiplicative deep arm (0.007), so `lesion_collapses` is technically
True in the JSON (0.169 < 0.007 is false, but 0.169 ≤ 0.25 is true). This is NOT the usual "the product is the
load-bearing thing" story — here the multiplicative deep net is the *worst* generalizer (its sharper per-role
prior hurts most). The product is load-bearing for the **train fit** (the single-layer lesion in the prior doc
collapsed train), but multiplication is irrelevant to **generalization**: neither additive (0.169) nor
multiplicative (0.007) generalizes — both far below the fixed FHRR (0.228). The lesion confirms the same
conclusion from the opposite side: **no learned variant generalizes; only the fixed primitive does.**

## Verdict for Phase 3B — an honest BOUNDARY, not a build

The **DEEP regime — the ONE surviving untested hypothesis the scoping isolated — is ALSO ruled out**, and now
the *capacity ceiling* certifies it (the prior single-layer NEGATIVE could be dismissed as "nothing to
credit-assign"; this one cannot — even true gradients fail). Combined with the two prior cheap-first NEGATIVEs
(learned multiplicative bind 0.056; single-layer dendritic sigma-pi 0.168), the dendrite is **comprehensively
ruled out for learnable, generalizable, multi-attribute composition.**

**⇒ The fixed ±1 / FHRR self-inverse primitive STAYS.** It is **not a host shortcut** — binding-by-coincidence /
the multiplicative self-inverse is a **STRUCTURAL neural primitive** (the production composer binds the LEARNED
PPMI codes *through* it; it bundles 0.989). The genuine residual — a *learned* generalizable multi-attribute bind
— is an honest, **characterized point-neuron BOUNDARY**, not a fixable one. **Phase 3B is an honest boundary, not
a months-scale `sim/` build.** This is the SURPASS-validated terminus: cheap-first (~90 s, 3 seeds, CPU), NO GPU,
NO `sim/` edit, NO months-scale commitment consumed.

What this does NOT touch (and is already BANKED on point neurons, per the scoping's reconciliation): the
**generalization** deliverable (PPMI local normalization — the dendrite would *hurt* it), **single-attribute**
binding (0.833 on real spikes), and the dendrite's REAL earned win, the **graded read-out** of a distributed code
(`enable_graded_dendritic_plateau`, the nav critic δ=1.33). The deep-dendrite NEGATIVE closes only the
**multi-attribute learned-bind** residual — the last open piece of the FHRR-idealization umbrella — as a boundary.

### Files
- Runner: `research/runners/_burndown_3B_deep_dendritic_binder_derisk.py` (461 lines, new; reuse-by-import only)
- Result JSON: `research/findings/raw/_burndown_3B_deep_dendritic_binder.json`
- Extends: `_phaseB_dendritic_bind_derisk.py` (single-layer NEGATIVE), `cortex_learned_binder_systematicity_probe.py`
  (the protocol), `_phaseB_multiplicative_bind_bundled_derisk.py` (the learned-linear control)
- Reuses (NO edit): `sim/dendritic_mlp.py`, `sim/dendritic_plasticity.py`, `sim/dendritic_neuron.py`
- Scoping this closes: `research/findings/2026-06-24-learned-cortex-dendrite-phase3-scoping.md` (§4 Stage 0, §6 SURPASS verdict)

_Cheap-first CPU/numpy de-risk. NO GPU, NO `sim/` edit. The deep dendritic binder — the last untested hypothesis
for a learned generalizable multi-attribute bind — fails to generalize even at the oracle capacity ceiling; the
fixed ±1/FHRR binding-by-coincidence primitive is the honest brain-grounded structural primitive, and the
multi-attribute residual is a characterized point-neuron boundary. Phase 3B = boundary, not build._
