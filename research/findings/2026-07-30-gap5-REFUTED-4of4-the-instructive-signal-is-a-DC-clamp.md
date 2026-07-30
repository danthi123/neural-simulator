# gap#5: my plateau-terminator recommendation REFUTED 4/4 — the instructive signal is a saturated DC clamp

**Date:** 2026-07-30 · **Status:** adversarial round returned `survived: false`, 4 of 4 skeptics refuted.
**Corrects:** [`2026-07-30-gap5-plateau-termination-is-INTRINSIC-and-the-machinery-already-exists.md`](2026-07-30-gap5-plateau-termination-is-INTRINSIC-and-the-machinery-already-exists.md).
Its *biology* survives and is in fact vindicated; its *proposed fix* was premature and is withdrawn.

## What I got wrong

I read Kandel Fig 10-15 correctly (plateau termination is intrinsic, inhibition only modulates), found the Kir
precedent on `cp_v_apical`, and proposed adding an intrinsic terminator conductance as the next build. Every
skeptic refuted that, for the same reason from four angles: **I proposed a fix before measuring the operating
point of the thing I was fixing.** The arc has never once instrumented its own instructive signal — the only
readout is a single scalar `apical_max` over a whole run, which cannot distinguish "on once" from "on always".

## The actual diagnosis — the write is EXACTLY uniform, by arithmetic

From `sim/kernels.py:325-345` plus the apical ODE at `sim/bridge.py:7185-7198`, at the runner's exact operating
point (`dt=1.0, tau_d=80, tau_r=2, strength=80, gain=2, k_thresh=4.0, R=0.15`). `fused_coincidence_plateau` adds
`g_inc = 80/(1+exp(-2*(c_count-4)))` EVERY step, with no refractory, no adaptation and no termination term.
Solving the apical fixed point against coincident-input count:

| `c_count` | `v_apical` | `is_post` |
|---|---|---|
| 1 | −59.32 mV | 0.00 |
| 2 | −9.94 mV | 25.06 |
| 3 | −1.49 mV | 33.51 |
| 4 | −0.36 mV | 34.64 |
| ≥6 | −0.18 mV | 34.82 (flat to 3 s.f. thereafter) |

The runner drives a σ=5 bump over 60 place cells at **density=1.0**, so every reader sees ~11-15 coincident
inputs at every step. `c_count` is therefore ~12 at all times, for all 12 readers, at all positions, and
`is_post = max(v_apical − (−35), 0) = 34.82` is **pinned — always, everywhere**, roughly 500× past its own
switch point.

So `dw = eta·etilde_pre·is_post·(w_max−w)` collapses to `dw ∝ etilde_pre · const`. Integrated over a lap, credit
to place index *p* is proportional to `∫etilde_p(t)`, and every place cell is active for the same total duration
per lap ⇒ **the integrated write is exactly uniform.** The measured baseline width of 51/60 is not "insufficient
sharpening"; it is the mathematically necessary output. Whatever place-selectivity exists at all is manufactured
by PRE-side nonlinearities bolted on afterwards (`w_max` saturation, `btsp_hetero_dep`, the normalize-then-⁴
eligibility). **That is exactly why every pre-side lever moved the number and every post-side lever was inert** —
they were sharpening the only term carrying any structure.

Critically, `c_count` **cannot** vary with position under this wiring: at density=1.0 every reader connects to all
60 place cells and the bump's active-cell count is position-invariant. No threshold calibration can make the
plateau an event here — `k_thresh` only selects "always on" or "always off". **The DC-ness is a property of the
CONNECTIVITY, not of the plateau parameters.**

## The residual is much smaller than 33%, and it is kernel shape, not sharpening

`cp_btsp_pre_elig` is a causal one-sided exponential low-pass; biological BTSP's kernel is bidirectional.
Scoring the kernel shapes directly: the symmetric σ=5 oracle gives circ 0.8719; a one-sided exponential at
`tau = btsp_elig_tau_ms/dwell = 1000/30 = 33.3` place indices gives **0.6343**. Measured is **0.588** — i.e. we
are at **~93% of the ceiling of the kernel we actually implemented**. Only ~7% of the "33% residual" is
recoverable by any amount of sharpening; the other ~26% is kernel shape, and a one-sided exponential cannot
become a symmetric Gaussian however hard it is inhibited.

**R1 (dominant):** the instructive signal is a DC clamp, not a discrete self-terminating EVENT. Real CA1 plateaus
last 140.2 ± 10.2 ms and are actively terminated; the engine's runs the full 1800 ms lap, ~13× biology, because
nothing in the engine can END a plateau — the session's own log noted we have only ever built mechanisms that
START one.
**R2:** the eligibility kernel is causal-only, so even a perfect event-plateau yields a one-sided field capped
near circ 0.63 at the current tau.

**They share one fix.** An event-plateau that ignites, self-terminates, then decays over seconds supplies the
FORWARD half of the kernel (pre-activity arriving after the plateau still finds `is_post > 0`), while the existing
pre-eligibility supplies the backward half. Making the plateau an event is not one of two competing repairs — it
is the repair.

## Standing of the two named candidates

**(b) plateau-terminating apical inhibition** targets the right term but is **inert as-is**: at the clamped
operating point `g_eff ≈ 6237` gives an effective apical conductance ≈730, so an inhibitory conductance would have
to exceed ~700 to pull `v_apical` below `v_hold` — and it would be re-ignited the next step regardless. It needs
an intrinsic terminator first, which is the arrangement biology uses (SK supplies the near-balancing outward
current that makes the plateau terminable; apamin doubles plateau duration, 81.3 → 179.7 ms).

**(a) multi-subunit apical dendrites** does not address R1 or R2 at all — it converts one DC instructive signal
into K DC instructive signals. It also has a contrary primary source: a PLOS Comput Biol 2019 CA1 tuft study
finds apical TUFT branches are **not** independent functional subunits (a shared Ca-spike initiation zone couples
them); independence belongs to thin basal/oblique dendrites. Multi-subunit stays genuinely needed for catalog
G.02's "cluster on one branch ≫ scattered" and the roadmap's nonlinear per-cell read, but it is the wrong
instrument for THIS residual.

## The actual next step — OPT-0, measure before paying for code

No `sim/` edit, essentially no GPU, on the existing harness.
- **Arm A:** instrument `is_post` per reader per position inside a run that already exists. If it comes back
  position-tuned, the diagnosis above is wrong and three planned builds were aimed at the wrong term. If it comes
  back flat, the diagnosis is *measured* rather than argued and becomes the pre-registered baseline.
- **Arm B:** drop `density` below 1.0 so `c_count` can vary with position. This may be a complete surpass at zero
  edit cost — the DC-ness is a wiring property, and density=1.0 is what destroys it.

The refutation's second argument is the one I should have applied to myself: if we build the `sim/` edit first and
it appears to work, the effect is entangled with an operating point nobody has characterized, and we credit a code
change with what may be a config artifact. **That is precisely the 2026-07-25 retraction's failure mode** (an
uncharacterized apical constant carrying a conclusion). This arc's dominant pattern is that the needed mechanism
was already in the engine at an inert default — five times this session.

## Honest sourcing caveat

The plateau-termination quantities (140.2 ms; apamin 81.3 → 179.7 ms; OLM-Ndnf vs α2 termination 84.4% vs 42.6%
at 50 ms) come from a full text that WAS opened. **The bidirectional-BTSP-kernel claim underpinning R2 rests on
Bittner-Magee 2017 read SECONDHAND — not opened.** R2's magnitude is provisional until that paper is read; R2's
DIRECTION is independently established by the engine's own causal-only low-pass and the kernel arithmetic above.
