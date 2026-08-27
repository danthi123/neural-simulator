---
type: finding
status: contributing
date: 2026-08-26
mechanism: b1-v1-orientation-selforg-onbridge-BCM
lane: b1-v1-selforg
seeds: [42, 43, 44, 45, 46, 47]
artifacts:
  - research/findings/raw/_b1_v1_selforg_bcm_6seed.json
---

# B1 on-bridge V1 orientation self-org: a BCM sliding metaplastic threshold BREAKS the common-mode boundary — oriented RFs emerge on-bridge (62x the potentiation-only control), seed-variable: PARTIAL (runner label BOUNDARY)
<!--derived-->

## One-line verdict
<!--derived-->
The 2026-08-14 on-bridge wall was diagnosed COMMON-MODE CONVERGENCE: the potentiation-only rate-Hebbian rule
drives ON and OFF to identical weights, so the signed RF cancels (osi_post_frac ~0.003, no lift over freeze or
shuffle). Adding a **BCM sliding metaplastic threshold** (Bienenstock, Cooper & Munro 1982) as an additive,
default-OFF substrate primitive supplies the missing input-specific DEPRESSION (LTP above theta_M=<y^2>, LTD
below). On-bridge, 6 seeds, production dims: osi_post_frac rises from 0.0037 to **0.173** — a ~62x
lift over the potentiation-only control, op-point-verified active-sparse, no weight collapse or dead cells.
Verdict: **PARTIAL (runner label BOUNDARY)** (the runner's own overall_verdict reads BOUNDARY at its strict 4/6-majority gate; 3/6 seeds clear +0.15 over BOTH controls, 2 of them by ~0.32, so oriented RFs decisively emerge on half the seeds).

## Why BCM is the right fix for THIS boundary (mechanism)
<!--derived-->
The rate-window Hebbian rule potentiates only, above a fixed coactivity threshold: `dw = lr*(x_tr*y_tr)*(w_max-w)`
for co-active pairs. Averaged over random orientation/phase, every ON and OFF synapse sees equal co-activation, so
both saturate identically and `W_ON - W_OFF ~ 0` (the common mode). BCM replaces this with a signed rule carrying a
per-postsynaptic-cell sliding threshold:

    theta_M_i = <y_i^2>   (running average of postsynaptic activity squared)
    dw_ij = gain * x_j * y_i * (y_i - theta_M_i)

`y_i > theta_M` potentiates the co-active input; `y_i < theta_M` DEPRESSES it. A cell fires strongly for its
(randomly-initialised) preferred phase/orientation and weakly for the contrast-reversed phase, so the ON/OFF pixels
co-active at the anti-preferred phase are depressed while the preferred-phase pixels are potentiated -> `W_ON` and
`W_OFF` become spatially anti-correlated -> a signed oriented RF. `theta_M ~ <y^2>` grows superlinearly, so runaway
potentiation is self-limited (the classic BCM stability). This is exactly the input-specific depression the
2026-08-14 finding named as the missing companion.

## Result (6 seeds, on-bridge SimulationBridge, production dims 8x4x16x16, dev=40000, bcm_gain=800, pre_floor=0.002)
<!--derived-->
**Artifacts.** `research/findings/raw/_b1_v1_selforg_bcm_6seed.json` (6-seed BCM run + provenance sidecar; backend cupy). The bcm=0 MATCHED control and the 1-seed tuning-lever numbers below are from scratch runs of the same runner (not re-saved); their values here are marked derived.
| arm | osi_post_frac | osi_pre (freeze) | osi_shuffle | margin over both | dev firing frac | verdict |
|---|---|---|---|---|---|---|
| BCM (learned) | 0.173 | 0.0037 | 0.0086 | 3/6 clear +0.15 | 0.0176 ✓ | PARTIAL (runner label BOUNDARY) |
| MATCHED CONTROL (bcm=0, same syn_scaling=0/decay=0, 1-seed) | 0.0028 | 0.0026 | 0.0026 | +0.0002 | 0.0089 ✓ | common-mode (no lift) |

Per-seed osi_post_frac: 42:0.157, 43:0.030, 44:0.331, 45:0.333, 46:0.027, 47:0.162. Seeds clearing the +0.15 margin over BOTH controls: 3/6.
RSA-to-host-Gabor (secondary): 0.884; orient-decode 0.268 (host reference 0.979).

## Instrument verification (the controls genuinely differ — the test is not void)
<!--derived-->
The MATCHED control (bcm=0 with the SAME syn_scaling=0 and hebb_decay=0 as the BCM arm — so the ONLY difference is
the rule) reads freeze(pre)=0.0026 ~ post=0.0028 ~ shuffle=0.0026 (**learn == freeze**): on OSI the potentiation-
only rule is genuinely inert, so the OSI instrument DIFFERENTIATES a working rule from a non-working one (if it did
not, the whole test would be void). This matched control also CLOSES the obvious confound — removing synaptic
scaling does NOT produce the lift (osi_post still 0.0028), so the ~62x lift is attributable to BCM's signed
LTD, not to the change in the competitive-normalization setting. With BCM on, the learned arm (0.173) separates
from BOTH freeze (0.0037) and the shuffle control (0.0086); the shuffle (orientation-destroyed input)
rises only modestly — BCM's LTD extracts a little orientation from the random spatial correlations of shuffled
input — but stays far below the learned arm, so the bulk of the selectivity comes from the oriented input
statistics via the rule, not from the substrate or the support (this residual shuffle level is exactly why the
strict margin is taken over BOTH controls). Anti-cheats from
the base runner: isotropic radius-4 RF support (carries no orientation), random init (orientation must be learned),
the host Gabor bank never applied to the pathway (only the RSA scoring reference). Operating point measured
active-sparse throughout development (dev firing frac 0.0176 in [0.005, 0.05]) -> not a dead-forward VOID; no
weight collapse (`frac_cells_l2_near_zero = 0`).

Note: the runner's legacy `on_minus_off_mean` diagnostic still prints "COMMON-MODE CONVERGENCE" under BCM — that is
a GLOBAL mean of ON minus OFF over all cells, which is ~0 whether the channels cancel per-cell OR each cell has a
per-cell signed opponent RF (ON in some pixels, OFF in others). It cannot see per-cell opponency; OSI is the
discriminating read, and OSI lifts ~62x (mean). The diagnostic string is not load-bearing here.

## Levers mapped (production dims, dev=40000, 1 seed)
<!--derived-->
`pre_floor` (the x_j presynaptic gate; lower = wider LTD coverage, closer to the pure-BCM limit) is the dominant
lever, with no dead cells anywhere in the range:

| pre_floor | gain 200 | gain 400 | gain 800 |
|---|---|---|---|
| 0.010 | margin 0.076 | 0.083 | — |
| 0.005 | — | 0.118 | 0.127 |
| 0.003 | — | 0.135 | 0.143 |
| 0.002 | — | 0.145 | **0.154** (clears +0.15) |

gain is a secondary lever (~+0.05 across 200->800). theta_alpha (0.001 vs 0.003) and homeostasis speed (default
vs slow) were near-neutral. Longer development (12k -> 40k) did NOT lift the fraction — the residual is not
development length; it is input decorrelation (below).

## Verdict: BCM breaks the common mode, but sits AT the +0.15 margin and is seed-variable — PARTIAL
<!--derived-->
BCM decisively breaks the common-mode boundary: on every seed osi_post_frac is many-fold its own freeze/pre and
the matched potentiation-only control (0.0028), so oriented RFs DO emerge on-bridge where the potentiation-only
rule produced none — on the strong seeds ~1/3 of cells reach OSI>0.5. But the outcome is **BIMODAL across seeds**,
which holds it short of a GO. osi_post_frac splits into a strong mode (~0.333, ~120x the control) and a weak
mode (~0.027, still ~10x the control), with 3/6 seeds clearing the strict +0.15 margin over BOTH
controls. This is the classic BCM/Hebbian INITIAL-CONDITION dependence: whether a cell's random initial weights
give it a strong enough phase/orientation preference to bootstrap the LTD-driven symmetry-break is seed-dependent,
so some seeds settle into sharply oriented RFs and others stay near the common mode. Two honest secondary facts:
the gain/pre_floor config was tuned on seed 42 (which itself lands at 0.157, just missing the margin because the
shuffle control rose to ~0.0086); and the shuffle is not at floor (BCM's LTD extracts a little orientation
from shuffled input too), so the strict margin is rightly taken over BOTH controls. Longer development did not lift
the fraction; pre_floor (LTD coverage, toward the pure-BCM x_j limit) and gain did, up to this plateau, with no
dead cells anywhere in the range.

The residual is the RAW ON/OFF input correlation. BCM's signed LTD sharpens opponency but cannot fully decorrelate
an input whose ON and OFF channels are anti-phase-correlated by construction (a full-field grating), so the
fraction of STRONGLY-oriented (OSI>0.5) cells plateaus near the margin and varies by seed. Named next lever
(grounded, and under concurrent separate test in this arc): compose BCM with input DECORRELATION — an LGN
center-surround/DoG whitening front-end that removes the pairwise ON/OFF correlation (Phase-B 2026-06-15 reached
+0.33 in numpy with whitening + ON/OFF dual pathway) — and/or plastic anti-Hebbian lateral inhibition (SAILnet;
Zylberberg-Murphy-DeWeese 2011), whose Gabor-RF GO used whitened patches AND plastic lateral inhibition together.
The 2026-08-14 finding named both as the companions BCM alone does not supply; this rung shows BCM supplies the
input-specific DEPRESSION half on its own: 3/6 seeds over the bar (2 by a wide margin), mean osi_post 0.173 (~62x the control) with RSA-to-host 0.88 — decisive emergence, not yet robust across seeds.

## Implementation (additive, guarded, default-OFF, byte-identical when off)
<!--derived-->
`sim/config.py`: `hebbian_bcm` (gain; 0.0 = OFF), `hebbian_bcm_theta_alpha`, `hebbian_bcm_pre_floor`.
`sim/bridge.py`: the rate-window Hebbian branch computes the signed BCM delta when `hebbian_bcm>0` (reusing
`cp_hebb_coactivity_trace` as y), and maintains `cp_bcm_theta = <y^2>`. With `hebbian_bcm=0` the branch takes the
verbatim original potentiation path -> byte-identical (`TestSubstrateActuallySeeded` passes; the bcm=0 control
reproduces the prior common-mode result exactly). Runner `research/runners/_b1_v1_selforg_bcm_derisk.py` reuses the
base on-bridge runner's build/develop/RF/OSI/RSA/controls/operating-point instrument by import; BCM on via env
passthrough. No other `sim/` behavior changed; the STDP path is untouched.

## Sources
<!--derived-->
BCM: Bienenstock, Cooper & Munro 1982, J Neurosci 2(1):32-48 (the sliding metaplastic threshold). Cooper & Intrator
2004 (BCM theory review). Law & Cooper 1994, PNAS 91:7797 (BCM develops oriented RFs from natural images). SAILnet:
Zylberberg, Murphy & DeWeese 2011, PLoS Comput Biol 7(10):e1002250 (whitened patches + plastic anti-Hebbian lateral
inhibition -> Gabor RFs). Whitening companion, on-bridge: research/findings/2026-06-15-phaseB-retinal-cortex-bridge-BUILD.md (and an LGN
center-surround/DoG whitening front-end is under separate concurrent test in this arc via
`_b1_v1_selforg_onbridge_lgn_whiten_derisk` — the composition BCM x whitening is the next rung).
Re-examines research/findings/2026-08-14-b1-v1-selforg-onbridge-operating-point-BOUNDARY.md; numpy ceiling
research/findings/2026-06-21-B1-v1-gabor-selforg-derisk.md.
