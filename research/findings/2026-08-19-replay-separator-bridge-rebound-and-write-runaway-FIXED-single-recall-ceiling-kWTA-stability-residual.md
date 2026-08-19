---
type: finding
status: qualified
date: 2026-08-19
runner: research/runners/_replay_dg_pattern_separation_bridge.py
artifacts:
  - research/findings/raw/replay_dg_sep/bridge_fixed_6seed.json
---

# Porting the DG replay-separator to the Izhikevich substrate: rebound + a Hebbian-write runaway FIXED, single-memory recall now at ceiling; the residual is k-WTA STABILITY, not separation

**Board #43** — port the controlled-LIF DG pattern-separation result
(`2026-08-19-replay-pattern-separation-DG-separator-keeps-similar-memories-discriminable-6seed-GO.md`)
onto the production `SimulationBridge` (Izhikevich + rate-window Hebbian).

## Verdict

**Qualified. The headline board #43 bar is NO-GO on the Izhikevich substrate (two SIMILAR memories both
discriminable: 0/6 seeds), but two biologically-grounded fixes advance the port decisively and RELOCATE the
residual.** Single-memory recall now reaches ceiling (selectivity +1.00, 6/6; scramble-teach inverts it to
−0.94…−1.00, 6/6) — the consolidation read no longer stalls at chance, which is exactly where the prior bridge
runner (`_replay_dg_pattern_separation_gate.py`) left it. The remaining blocker is precisely localized and
quantified: the Izhikevich k-WTA does not reliably hold a SPARSE code for BOTH memories at once. No `sim/` edit
— both fixes use existing public config fields. Deterministic (per-seed byte-identical on re-run).

## The prior root cause was partly wrong; the dominant blocker was a WRITE runaway the prior finding missed

The #43 finding attributed the chance read to (a) Izhikevich post-inhibitory REBOUND and (b) a razor-thin
k-WTA. Root-causing on the bridge shows a different picture:

1. **Rebound is real but FIXABLE.** With the committed gate config the DG already sparsified (Jaccard
   1.0→0.68, ~20 granules), so rebound was not the read blocker there.
2. **The dominant blocker was a Hebbian WRITE runaway the prior finding never named.** The bridge rate-window
   rule is SOFT-BOUND (`bridge.py:1141,9591`): `dw = lr·coact·(w_max − w)`. On an all-to-all dg→answer path
   with strong dg drive, ONLINE plasticity runs away — as any dg→answer weight grows, dg drives non-target
   answer cells, they fire, Hebbian potentiates them, and the whole matrix saturates. Measured: the dg→answer
   mean climbs 0.05→33–68 (max 90) and answer spikes balloon to ~1900/event during plastic replay, while a
   teacher-only measurement (plasticity off) shows the target assembly firing ALONE (a0=21, a1=0). A saturated
   matrix drives every answer equally → chance read. This, not the DG rebound, is why the prior read was flat.

## The two fixes (additive, no `sim/` edit, biologically grounded)

**FIX 1 — shunting inhibition (kills the rebound).** The DG granule (and answer) inhibitory reversal is set to
≈vr via the existing `BrainRegion.syn_reversal_potential_i_override` (the same field striatal MSNs use; dg
−63 mV, answer −60 mV, vs the default −75 mV). Inhibition then DIVIDES the excitatory drive toward rest
instead of hyperpolarizing BELOW vr, where the Izhikevich quadratic `k·(v−vr)·(v−vt)` turns regeneratively
depolarizing and produces the rebound burst. Biology: shunting (chloride, ECl≈Vrest) feedforward inhibition —
Carandini & Heeger divisive normalization; granule cells sit near ECl.

**FIX 2 — transmission-gated write (kills the runaway).** A TRANSMISSION gate on dg→answer, OFF during replay
(the WRITE: the answer fires from the reinstated-index teacher only → clean teacher-only coincidence → no
runaway) and ON during probe (the READ: dg drives answer via the learned weights). Plasticity is per-neuron
pre×post firing, independent of transmission, so the coincidence still writes. This is the bridge equivalent
of the LIF's OFFLINE write. Biology: encoding coincidence (SWR-time potentiation) is distinct from recall
transmission — the synapse is being written before it is read.

With both fixes the write stays teacher-clean (replay of m0 fires a0 and leaves a1 silent, 6/6) and
single-memory recall reaches ceiling (below).

## Results — 6 seeds (42/43/44/100/101/102), per-seed and pooled (`bridge_fixed_6seed.json`)

<!--derived-->
_Numbers rounded from the cited full-precision artifact._

| seed | single sel | scramble sel | sim-ON m0 | sim-ON m1 | DG Jaccard ON | DG sizes (m0,m1) | dense-collapse |
|---:|---:|---:|---:|---:|---:|:--:|:--:|
| 42  | +1.00 | −0.94 | −0.17 | +0.15 | 0.22 | (200, 44)  | yes |
| 43  | +1.00 | −1.00 | +0.55 | −0.59 | 0.45 | (39, 55)   | no  |
| 44  | +1.00 | −1.00 | −0.11 | +0.06 | 0.28 | (56, 198)  | yes |
| 100 | +1.00 | −1.00 | +0.09 | −0.13 | 0.30 | (199, 60)  | yes |
| 101 | +1.00 | −1.00 | +0.36 | −0.36 | 0.55 | (46, 39)   | no  |
| 102 | +1.00 | −1.00 | +0.44 | −0.44 | 0.34 | (199, 68)  | yes |

**Advance controls (PASS):** single-memory recall +1.00 (6/6, correct assembly wins, wrong ≈ 0);
scramble-teach — consolidate m0 with m1's answer, probe m0 against the TRUE pairing — inverts the selectivity
to ≈−1.00 (6/6), so the read rides the LEARNED mapping, not readout geometry; the write coincidence is
teacher-clean (6/6). These four advance checks pass on seeds 43/101/102 (`advances=True`).

**Headline board #43 bar (NO-GO):** two SIMILAR memories both discriminable — 0/6 seeds. The pooled sim-ON
mean selectivity is ≈0 (−0.013) with the separator OFF also ≈0, so there is no dissociation. The per-memory
signature is universal and ANTI-SYMMETRIC: whichever memory reads correct (+x), the other reads backward
(−x). It is not noise — |per-memory sel| reaches 0.55.

## The residual, precisely localized: k-WTA STABILITY, not separation quality

The anti-symmetry is NOT a pattern-separation-quality problem. Three measurements localize it to the k-WTA:

1. **Even DISSIMILAR memories (input Jaccard 0.04, DG Jaccard 0.18–0.35) fail both-win 0/6** with the same
   anti-symmetric signature — separation quality is adequate, yet both reads pick the SAME answer.
2. **Direct-readout of the learned mapping.** Driving each written engram DIRECTLY (bypassing input→dg) and
   reading the answer: BOTH engrams drive the same assembly (e.g. seed 43 dissimilar: m0-engram a0=89/a1=40,
   m1-engram a0=90/a1=45 — both pick a0).
3. **The weight matrix explains it.** For that seed the isolated engram sizes are m0=196 (DENSE), m1=69, with
   68 of m1's 69 cells INSIDE m0's engram. The k-WTA collapsed ONE memory to a near-dense code that SUBSUMES
   the other; the dense memory's answer then wins both probes. `dense-collapse` (max DG active fraction > 0.60)
   fires on 4/6 seeds. The two non-collapsed seeds (43, 101) still fail both-win because the residual overlap
   is nested rather than symmetric.

So the substrate residual is: **the Izhikevich DG k-WTA cannot reliably hold a stable, SYMMETRIC, sparse code
for two similar (or even dissimilar) inputs — it collapses to dense for one memory (seed-dependent), and the
dense engram dominates the read.** This is the 2026-05-31 separation-vs-reliability boundary, now traced to
k-WTA symmetric-stability on the Izhikevich substrate specifically, distinct from the DG rebound and the write
runaway that this runner closes.

## Levers tried against the k-WTA-stability residual (all measured, none robust)

Shunting reversal (−60…−70 mV); feedforward vs feedback-dominant basket inhibition (set-point); perforant
fan-in 6–20 (LIF-faithful ~15% sampling); expansion n_dg 200→400; integrator granule phenotypes
(IZH2007_HIPPO_PYRAMIDAL b=+5, IZH2007_THALAMIC_RELAY b=+15, IZH2007_GPE_PACEMAKER b=+1); homogeneous granules
(heterogeneity off); write w_max 3–90 and learning-rate 0.1–40. Every regime lands in one of two failure
modes: STRONG competition → sparse but ASYMMETRIC/nested (one memory dense) → the dense memory dominates; GENTLE
competition (shunting/feedback/integrator) → SYMMETRIC sizes but DENSE (Jaccard 0.6–0.97) → both memories drive
the same answer. Neither yields two symmetric, mutually-non-nested sparse engrams.

## The named next mechanisms (this is a wall on a METHOD, not the capability)

1. **A homeostatic sparsity set-point that actually caps activity.** The DG's ~1–2% activity is held by a
   competitive/homeostatic loop we replaced with fixed feedforward weights + a fixed threshold — the razor
   window. A slow per-granule adaptive threshold (intrinsic excitability homeostasis; Turrigiano) that drives
   every granule toward a target firing fraction, run alongside the fast basket inhibition, would cap the
   dense-collapse without the rebound that hyperpolarizing k-WTA incurs. Not expressible with the current
   fixed-weight config; needs a spiking homeostatic-threshold mechanism on the granule region.
2. **A developed (not fixed-random) perforant projection.** Fixed random fan-in makes the winner set
   drive-dominated by the shared inputs. A competitively-learned input→dg projection (the LIF result is on a
   fixed projection too, but the Izhikevich nonlinearity needs the extra margin) would place each granule on a
   distinct input conjunction.
3. **An AdEx or LIF-regime granule region.** `2026-07-25` recorded replay as "AdEx substrate-specific"; the
   ADEX_RS phenotype lacks the Izhikevich quadratic rebound and may hold the clean threshold-linear k-WTA the
   LIF demonstrates. Buildable via `izh/adex_neuron_type` on the dg region.

## Tracked scaffolds (host, not brain)

Host-defined input (sensory) patterns and answer assemblies; host reinstatement of each memory's input AND
answer during replay (the hippocampal index / SWR trigger); scheduled down-states; the WRITE/READ
transmission-gate phase (a host-scheduled sleep/wake gate, like the plasticity gate); a rate-window Hebbian
coactivity write (the stand-in the consolidation gates use); an argmax over answer spike counts for
MEASUREMENT only; a fixed random perforant projection and fixed FS anatomy (not developed).

## Reproduce

    OMP_NUM_THREADS=2 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_bridge \
        --seeds 42 43 44 100 101 102 \
        --out research/findings/raw/replay_dg_sep/bridge_fixed_6seed.json

## Sources

EXTERNAL-SEARCH-RAN: Izhikevich post-inhibitory rebound (quadratic spike current below vr); shunting
inhibition and divisive normalization for k-WTA; DG sparse-expansive pattern separation (logged to the
external-search record via the corpus check, 2026-08-19).

- Izhikevich, E.M. (2007). Dynamical Systems in Neuroscience. — the quadratic `k(v−vr)(v−vt)` spike current
  and post-inhibitory rebound; negative recovery slope `b` gives rebound (codebase enum confirms: STN_BURST
  "strong rebound burst … uses negative b").
- Carandini, M., Heeger, D.J. (2012). Normalization as a canonical neural computation. Nat Rev Neurosci
  13:51–62. — shunting/divisive inhibition.
- Turrigiano, G. (2011). Too many cooks? Intrinsic and synaptic homeostatic mechanisms. Annu Rev Neurosci
  34:89–103. — the intrinsic-excitability homeostat named as next mechanism 1.
- Marr 1971; O'Reilly & McClelland 1994; Leutgeb 2007; Bakker 2008 — DG sparse-expansive separation (as in the
  #43 LIF finding).

Internal biology bindings: `research/biology/swr-sequence-replay.md`, `research/biology/systems-consolidation.md`.
The #43 LIF GO remains the science result; this finding is the Izhikevich-substrate PORT, which closes the
rebound + write-runaway blockers and localizes the remaining k-WTA-stability residual.
