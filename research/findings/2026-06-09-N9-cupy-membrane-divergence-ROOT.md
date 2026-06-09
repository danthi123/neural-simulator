# N9 ROOT (refined): NOT homeostasis — a CuPy-vs-numpy divergence in the MSN-D1 critic's membrane response to excitatory current

**Date:** 2026-06-09
**Type:** controller-run dual-backend instrumentation (numpy + CuPy, same script/config).
**Owner directive (this session):** *"I want everything biologized before we move on — no banking any cheats."* → N9 is a **bug to FIX**, not a mechanism to bank. Standing rule for the whole nav arc.

## What was ruled OUT (cleanly)

- **Substrate/wiring/current path:** afferent→critic CSR byte-identical pre/post Gabor; same g_e on both backends.
- **The plastic-mask bug** (`06dce515`): real + fixed, but unrelated to the warm-up (in-bounds).
- **Per-region homeostasis (`89b8d909`):** **NOT broken on CuPy.** Instrumented the critic's firing threshold on both backends — it adapts down **identically**: numpy −41.5 mV, CuPy −42.7 mV (vs vpeak +40), mask correctly set 80/80 on both. The homeostasis edit works on GPU.

## The pinned divergence (clean apples-to-apples, 1× drive, same config)

| critic `striosome_value` COLD probe | numpy (CPU) | CuPy (GPU, production) |
|---|---|---|
| g_e (last-half) | 0.0852 | 0.0750 (~12% lower, ≈ same) |
| g_i | 0.0 | 0.0 |
| **membrane V (mean)** | **−69.06 mV** (11 mV above rest) | **−79.89 mV** (0.1 mV above rest) |
| max V reached | −66.2 | −79.8 |
| → fires? bootstraps? | yes → 0.20→3.31 | **no → 0.20→0.20 (frozen)** |
| firing threshold (homeostasis) | −41.5 (adapted) | −42.7 (adapted) — same |

**At a nearly-identical excitatory conductance, the membrane depolarizes 11 mV on numpy but 0.1 mV on
CuPy.** And at **25× drive** (g_e 1.87, a huge conductance) the CuPy membrane *still* only reached
−77 mV — so the excitatory current barely moves the critic's membrane on CuPy **regardless of
magnitude.** This is a real CuPy-vs-numpy divergence in the MSN-D1 critic's membrane / synaptic-current
integration, NOT a firing margin and NOT homeostasis.

## Why str_D1/D2 MSNs fire on CuPy but this critic doesn't

Same Izhikevich MSN-D1 cell type, but: the cascade MSNs receive **huge** cortico-striatal drive
(weight ~125) — enough that even a partial response crosses threshold. The value critic reads a
**weak** place afferent (weight 0.2) and so it sits in the regime where this divergence is decisive.
The MSN-D1 has an intrinsic up/down bistable barrier (the k(v−vr)(v−vt) term between rest and
threshold); on numpy the synaptic current pushes the critic over into the up-state (−69, fires), on
CuPy the same (and even 25×) current does not. The override `syn_reversal_potential_i_override=−60`
on the critic is a suspect (could the excitatory current be using the wrong reversal on CuPy?), as is
the fused Izhikevich/conductance kernel (`@fuse` = `cp.fuse` on CuPy, plain python on numpy).

## Open root-cause (the fix target — biologize, do not bank)

WHERE does the excitatory synaptic current fail to depolarize the critic membrane on CuPy? Candidate
ops (to instrument step-by-step on BOTH backends — the production backend is CuPy, the lesson of this
whole thread is "never conclude from numpy alone"):
1. the g_e → synaptic-current conversion (`fused_conductance_decay_and_current`): is `g_e×(E_e−V)`
   using `E_e=0` for the critic's excitatory input on CuPy, or is the `−60` inhibitory override
   leaking into it?
2. the assembly of `total_input_current` for the critic (is the synaptic current actually added?);
3. the fused Izhikevich MSN-D1 update (`fused_izhikevich2007_dynamics_update`) — does the fused CuPy
   kernel settle the membrane differently from the unfused numpy path for the MSN parameters?

## Status / next

- `sim/` byte-empty (only the committed plastic-mask fix `06dce515`). No new edits.
- Harness: `research/findings/raw/_n9_critic_current_diag.py` (dual-backend; `_homeo_state` probe;
  `CRITIC_MARGIN_MULT` env). Logs: `_n9_homeo_{numpy,cupy}.log`, `_n9_margin25_cupy.log`.
- **NEXT:** root-cause the membrane/current divergence (instrument the 3 candidate ops on both
  backends) → propose the minimal protected `sim/` fix (byte-review) → re-test N9 on CuPy (the critic
  must fire + bootstrap on the production backend) → then the 6-seed nav A/B. Per the owner directive:
  this gets FIXED (N9 biologized on GPU), not banked.
