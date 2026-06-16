# On-substrate learned binder (CYCLE 102–103): single-attribute GO on the real bridge; 3-way bundling is a point-neuron limit

**Date:** 2026-06-16
**Status:** single-attribute learned bind = **GO end-to-end on the spiking bridge**; 3-way bundling (the conversational fact structure) = **NEGATIVE for the additive point-neuron bind** (multi-seed) → localized to multiplicative/dendritic binding (de-risk in flight).

## What this arc tested

The "step 3" frontier is to replace the conversational composer's fixed vector-symbolic-algebra bind (a principled idealization) with a cortex that **learns** to bind role-filler pairs. Prior cheap-first cycles established the learned binder is additive (`bound = nonlinearity(role·W_R + filler·W_F)`), generalizes systematically on the stream codes (held-out 0.889), and that the sign must be carried by **ON/OFF opponency** rate coding (#2b: 0.806). This arc pushed three remaining questions to ground.

## Results

| Question | Test | Result | Verdict |
|---|---|---|---|
| Is the learning rule **brain-faithful** (no weight transport)? | #2c feedback alignment (fixed random feedback matrix replaces W_Uᵀ in the hidden-layer backward) | mean **0.528** vs exact-backprop 0.806; seed 42 **0.917** (beats exact), seed 43 **0.083** (collapse), seed 44 0.583 | **PARTIAL / seed-unstable** — bare random feedback works when alignment seeds well but has no mechanism to guarantee it. Needs **e-prop eligibility traces** (the documented follow-up). |
| Does the **real LIF spiking** ON/OFF preserve a SINGLE-attribute learned bind on the bridge? | on-bridge step 1: drive `bind_pos`=relu(h), `bind_neg`=relu(−h) as LIF populations (N_PER=16/dim), read spike rates, numpy-unbind | on-bridge held-out **0.833 = 100% of the numpy reference** (3 seeds 0.75/0.75/1.00), ≫ mem-floor 0.0 | **GO** — the real spiking nonlinearity (threshold/refractory/finite-count) carries the single-attribute learned bind exactly. |
| Does the learned bind handle a 3-way **BUNDLED** fact (agent+verb+object, the conversational structure)? | single-pair-trained binder unbinds each role from the superposition | single-binding ceiling 0.806, but BUNDLED **train-combo 0.285 / held-out 0.206** (~chance 0.062) | **NEGATIVE** — even train combos fail ⇒ superposition crosstalk, not a generalization gap. |
| Does **bundle-aware** training fix it? | train the unbind to recover each role's filler FROM the 3-way bundle (d_bundle to all 3 binds) | BUNDLED **train-combo 0.269 / held-out 0.193** (3 seeds) ~ the single-pair NEGATIVE | **NEGATIVE** — bundle-aware training does not rescue it. |
| Does a **multiplicative (dendritic)** bind recover superposition? | learned FHRR-style: bind g=u⊙w, unbind r=bundle⊙uinv (both Hadamard products), linear cleanup | _in flight_ | _CYCLE 103, `_phaseB_multiplicative_bind_bundled_derisk.py`_ |

## The localization (why additive + linear unbind cannot bundle)

Unbinding role *t* from a superposition requires applying the **role-specific inverse** to the bundle: `bundle / u_t` recovers the matching filler while the other facts become noise. That inverse is a **multiplication** by a role-dependent factor (`1/u_t`). A shared **linear** unbind cannot implement a role-dependent scaling of the bound vector — it is structurally incapable, independent of capacity (D_h) or training (single-pair or bundle-aware). This is the same **point-neuron limit** the project keeps meeting (Mikulasch-Priesemann: the operations that need analog/multiplicative interaction — whitening/decorrelation, and now binding-superposition — are dendritic, not point-neuron). Multiplication is a **dendritic** operation; the project already has a two-compartment dendritic neuron on the bridge (D2 arc).

## Honest scope and the division of labor

- **Single-attribute learned binding is validated end-to-end on the real spiking substrate** (numpy 0.806 → ON/OFF 0.806 → real LIF on-bridge 0.833 = 100% of numpy). This is genuine on-substrate "step 3" progress for the part that works.
- **Bundling (multi-attribute facts)** is, for the point-neuron additive bind, a genuine capacity limit. Two honest resolutions: (a) keep the **fixed FHRR / ±1 algebra** for bundling — already the production composer default, validated multi-seed at V=320; (b) a **learned multiplicative (dendritic)** bind — the in-flight de-risk; if GO, realize it on the two-compartment substrate.
- This sharpens the long-standing project boundary: the learned cortex provides the codes (stream-learned) + single-attribute learned binding; superposition of facts needs either the fixed algebra or the dendritic substrate.

## Reproduce

```bash
# brain-faithful local rule (#2c)
SIM_BACKEND=numpy python -u -m research.runners._phaseB_spiking_bind_feedback_align_derisk
# real-LIF single-attribute bind on the bridge (step 1) -- GPU
SIM_BACKEND=cupy  python -u -m research.runners._phaseB_onbridge_bind_nonlinearity_derisk
# bundled facts: single-pair-trained, then bundle-aware-trained
SIM_BACKEND=numpy python -u -m research.runners._phaseB_learned_bind_bundled_facts_derisk
SIM_BACKEND=numpy python -u -m research.runners._phaseB_learned_bind_bundle_trained_derisk
# multiplicative (dendritic) bind -- the localization
SIM_BACKEND=numpy python -u -m research.runners._phaseB_multiplicative_bind_bundled_derisk
```

Anti-cheats throughout: leakage-free train/held-out systematicity splits; memorization-floor (lookup table → 0.0); chance line (1/F = 0.062); the on-bridge result is compared to its own numpy reference (the spiking-vs-numpy gap reported honestly). The flat 2,048-concept curated cortex remains the shipped product.
