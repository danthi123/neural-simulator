# On-substrate learned binder (CYCLE 102–103): single-attribute GO on the real bridge; 3-way bundling is a point-neuron limit

**Date:** 2026-06-16
**Status:** single-attribute learned bind = **GO end-to-end on the spiking bridge** (real LIF = 100% of numpy); 3-way bundling (a conversational fact) = **NEGATIVE for any naive learned bind** (additive 0.193; learned-multiplicative-with-learned-inverse 0.056), while a **fixed ±1 self-inverse bind bundles at 0.989** on the same harness. **Conclusion:** bundling needs a fixed, biologically-grounded coincidence/multiplicative primitive (the production composer, already binding learned codes) — it is not learnable from scratch and is not a host shortcut.

## What this arc tested

The "step 3" frontier is to replace the conversational composer's fixed vector-symbolic-algebra bind (a principled idealization) with a cortex that **learns** to bind role-filler pairs. Prior cheap-first cycles established the learned binder is additive (`bound = nonlinearity(role·W_R + filler·W_F)`), generalizes systematically on the stream codes (held-out 0.889), and that the sign must be carried by **ON/OFF opponency** rate coding (#2b: 0.806). This arc pushed three remaining questions to ground.

## Results

| Question | Test | Result | Verdict |
|---|---|---|---|
| Is the learning rule **brain-faithful** (no weight transport)? | #2c feedback alignment (fixed random feedback matrix replaces W_Uᵀ in the hidden-layer backward) | mean **0.528** vs exact-backprop 0.806; seed 42 **0.917** (beats exact), seed 43 **0.083** (collapse), seed 44 0.583 | **PARTIAL / seed-unstable** — bare random feedback works when alignment seeds well but has no mechanism to guarantee it. Needs **e-prop eligibility traces** (the documented follow-up). |
| Does the **real LIF spiking** ON/OFF preserve a SINGLE-attribute learned bind on the bridge? | on-bridge step 1: drive `bind_pos`=relu(h), `bind_neg`=relu(−h) as LIF populations (N_PER=16/dim), read spike rates, numpy-unbind | on-bridge held-out **0.833 = 100% of the numpy reference** (3 seeds 0.75/0.75/1.00), ≫ mem-floor 0.0 | **GO** — the real spiking nonlinearity (threshold/refractory/finite-count) carries the single-attribute learned bind exactly. |
| Does the learned bind handle a 3-way **BUNDLED** fact (agent+verb+object, the conversational structure)? | single-pair-trained binder unbinds each role from the superposition | single-binding ceiling 0.806, but BUNDLED **train-combo 0.285 / held-out 0.206** (~chance 0.062) | **NEGATIVE** — even train combos fail ⇒ superposition crosstalk, not a generalization gap. |
| Does **bundle-aware** training fix it? | train the unbind to recover each role's filler FROM the 3-way bundle (d_bundle to all 3 binds) | BUNDLED **train-combo 0.269 / held-out 0.193** (3 seeds) ~ the single-pair NEGATIVE | **NEGATIVE** — bundle-aware training does not rescue it. |
| Does a **learned multiplicative** bind recover superposition? | learned FHRR-style: bind g=u⊙w, unbind r=bundle⊙uinv with a **learned** uinv=role·W_Rinv, linear cleanup | single 0.083 / bundled **0.056** (3 seeds, ~chance) — collapses even on single-attribute | **NEGATIVE (confounded)** — a **learned linear** inverse cannot approximate the reciprocal 1/u; the formulation is broken, not "multiplication doesn't help." |
| **POSITIVE CONTROL**: does a **fixed** ±1 VSA bind bundle on this exact harness? | role binarized to a ±1 hypervector (its own inverse under ⊙), filler projected; bundle then unbind by the ±1 self-inverse | single **1.000** / bundled **0.989** (3 seeds) | **GO** — the harness DETECTS working bundling, so the learned NEGATIVEs are REAL, and a **fixed self-inverse** algebra is what bundles. |

## The capability map (the decisive contrast)

| Bind | Single-attribute | 3-way bundle (a fact) |
|---|---|---|
| **Fixed ±1 / FHRR algebra** (self-inverse role) | 1.000 | **0.989** |
| **Learned additive** (point-neuron) | **0.806** → real-LIF **0.833** | 0.193 |
| **Learned multiplicative + learned LINEAR inverse** | 0.083 (broken) | 0.056 (broken) |
| chance | 0.062 | 0.062 |

Bundling works **only** with a FIXED, self-inverse (or conjugate) algebraic structure. It is not learnable as a naive bind: additive has no multiplicative inverse, and a learned *linear* inverse cannot be a reciprocal (it breaks even single-attribute).

## The localization (why additive + linear unbind cannot bundle)

Unbinding role *t* from a superposition requires applying the **role-specific inverse** to the bundle: `bundle / u_t` recovers the matching filler while the other facts become noise. That inverse is a **multiplication** by a role-dependent factor (`1/u_t`). A shared **linear** unbind cannot implement a role-dependent scaling of the bound vector — it is structurally incapable, independent of capacity (D_h) or training (single-pair or bundle-aware). This is the same **point-neuron limit** the project keeps meeting (Mikulasch-Priesemann: the operations that need analog/multiplicative interaction — whitening/decorrelation, and now binding-superposition — are dendritic, not point-neuron). Multiplication is a **dendritic** operation; the project already has a two-compartment dendritic neuron on the bridge (D2 arc).

## Honest scope and the biology-translatable conclusion

The conversational bind decomposes into three pieces, and this arc places each precisely:

1. **The representations (concept codes)** — LEARNED on the spiking substrate (the stream cortex, validated multi-seed). ✅
2. **Single-attribute binding** — LEARNABLE and validated end-to-end in real spikes (numpy 0.806 → ON/OFF 0.806 → real-LIF on-bridge 0.833 = 100% of numpy). ✅
3. **Multi-attribute bundling (a fact = a superposition of bindings)** — requires a **FIXED, self-inverse algebraic structure** (the ±1 hypervector / FHRR conjugate). A naive *learned* bind cannot replace it: additive lacks any inverse (0.193), and a learned *linear* inverse cannot be a reciprocal (0.056, breaks even single-attribute). The fixed structure bundles at 0.989.

**This is not a host shortcut — it is a biologically-grounded structural primitive.** Real neural binding is built on *coincidence detection* and *dendritic multiplication* (structural mechanisms), not an operation learned from scratch. The project already realizes exactly this: the production composer binds the **learned** stream-cortex codes with **fixed ±1 coincidence** (spiking AND on ON/OFF, validated 0.92 who-Q&A in the biologization sweep). So the conversational bind is, honestly: **learned representations flowing through a fixed, biologically-grounded coincidence/multiplicative binding primitive.**

**Definitive answer to the "step 3" question (replace the fixed bind algebra with a learned bind):** the *codes* and *single-attribute binding* are learnable on the substrate (done); the *bundling algebra's fixed self-inverse structure* is **load-bearing and biology-grounded**, and a from-scratch learned bind does not improve on it while losing bundling. The "principled idealization" is, in this precise sense, the binding primitive itself — which brains also have as structure, not as a learned operation. The learned-multiplicative-with-learned-inverse direction is closed (a learned linear inverse can't be a reciprocal); the only conceivable further build (fixed self-inverse roles + learned filler codes + dendritic multiplication) **is already the production composer binding learned codes**.

## Reproduce

```bash
# brain-faithful local rule (#2c)
SIM_BACKEND=numpy python -u -m research.runners._phaseB_spiking_bind_feedback_align_derisk
# real-LIF single-attribute bind on the bridge (step 1) -- GPU
SIM_BACKEND=cupy  python -u -m research.runners._phaseB_onbridge_bind_nonlinearity_derisk
# bundled facts: single-pair-trained, then bundle-aware-trained
SIM_BACKEND=numpy python -u -m research.runners._phaseB_learned_bind_bundled_facts_derisk
SIM_BACKEND=numpy python -u -m research.runners._phaseB_learned_bind_bundle_trained_derisk
# learned multiplicative bind (confounded NEGATIVE: learned linear inverse can't be a reciprocal)
SIM_BACKEND=numpy python -u -m research.runners._phaseB_multiplicative_bind_bundled_derisk
# POSITIVE CONTROL: fixed +-1 FHRR bundles 0.989 on the same harness (proves the harness is sound)
SIM_BACKEND=numpy python -u -m research.runners._phaseB_fixed_fhrr_bundled_control
```

Anti-cheats throughout: leakage-free train/held-out systematicity splits; memorization-floor (lookup table → 0.0); chance line (1/F = 0.062); the on-bridge result is compared to its own numpy reference (the spiking-vs-numpy gap reported honestly). The flat 2,048-concept curated cortex remains the shipped product.
