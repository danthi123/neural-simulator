---
type: finding
status: no-go
date: 2026-08-19
runner: research/runners/_replay_dg_pattern_separation_bcm.py
artifacts:
  - research/findings/raw/sep_readout/bcm_6seed.json
---

# A BCM SELECTIVITY-GATED dg→answer WRITE writes the discriminative private-granule signal into the weights and breaks the anti-symmetry (6/6) — but both_win is still NO-GO (0/6), because the READ reactivates the DOMINANT engram: the residual re-localizes OUT of the dg→answer write entirely and INTO post-consolidation DG reactivation

**Board #90** — the memory-separator residual, lane-H. This is the mechanism the prior
finding (`2026-08-19-memory-separator-readout-competitive-write-NOGO-relocalizes-to-private-granule-underwrite.md`,
#73/commit 98cd33bb) named as the top untried lever after banking the per-granule
OUTPUT-transform family as insufficient: a **selectivity / novelty-gated write (BCM-like)**
that AMPLIFIES memory-private granules toward their answer and SUPPRESSES shared granules.
This finding BUILDS + 6-seed tests exactly that, and reports a decisive NEGATIVE that
CORRECTS the prior finding's localization a second time.

## Verdict

**NO-GO (both_win 0/6), and the negative is decisive + re-localizing — OUT of the
dg→answer write.** The BCM selectivity gate does precisely what the prior finding said was
the missing lever: it RAISES `private_m1_to_correct` far off baseline (writes the
discriminative signal into the memory-private granule weights) and BREAKS the anti-symmetry
— **both memories' weight-space answer-vote margins go positive on 6/6 seeds** (LESION 0/6).
The load-bearing WEIGHT quantity the prior finding identified is fixed. Yet `both_win`
stays 0/6, and the gate is in fact a behavioral REGRESSION (dissimilar both_win 2/6 → 0/6).
The reason, measured directly: **after the #78 consolidation the subordinate memory's own
input reactivates the DOMINANT memory's engram** — driving m1's input reactivates `eng0`
exactly (Jaccard 1.00 to eng0, 0.47–0.60 to eng1) on all 6 seeds, so the read is performed
through the wrong engram and the written private granules never fire during the read. The
separated weights are unreachable. All pipeline controls pass (single-memory recall at
ceiling 6/6; scramble-teach inverts 6/6; the LESION arm is BYTE-IDENTICAL to the committed
#78/readout baseline — exact compare, margins + private/shared decomposition match to 3
decimals on every seed), so the verdict is DEFINED, not UNDEFINED. No `sim/` edit. The
residual re-localizes from the WRITE (banked here) to READ-TIME DG reactivation.

## What was built (the named mechanism, no `sim/` edit)

A **BCM sliding-threshold selectivity gate** (Bienenstock, Cooper, Munro 1982) on the
plastic dg→answer weights, stacked on the #78 population set-point (DG code symmetric +
non-dense), so the only manipulated variable vs the #78 baseline is the WRITE rule:

1. Run the #78 base consolidation (the on-substrate coincidence write) unchanged — shared
   granules get their (soon-suppressed) excess. When `bcm_gate=False` this is the ONLY
   step ⇒ byte-identical LESION.
2. Read per-granule per-memory ISOLATED reactivation firing on a FRESH twin bridge (same
   seed ⇒ identical fixed perforant wiring), where the memory-private granules still
   reactivate. This recovers the memory-selectivity of the fixed input→dg projection. (The
   selectivity is UNREADABLE off the post-consolidation main substrate — see below — which
   is why a twin is used.)
3. Per-granule sliding threshold θ_g = ⟨activity across memories⟩; selectivity
   s_g = 1 − (second-memory rate / top rate). A granule selective for memory m
   (s ≥ thresh, rate above θ) is POTENTIATED toward a_m; a non-selective (shared) granule
   is SUPPRESSED (base excess × s^γ, heterosynaptic LTD). Applied as a runner-side
   transform on the same dg→answer weight vector the #readout renorm used; the twin
   reactivation firing and the read stay on-substrate spiking.

Biology: BCM metaplastic selectivity (a postsynaptic-history sliding threshold makes a cell
SELECTIVE — potentiating the input that drives it above threshold, depressing the rest),
applied per-granule across the memory-reactivation ensemble; heterosynaptic LTD of
non-selective inputs (Chistiakova et al. 2014). The `bcm_gate=False` LESION is a no-op →
byte-identical to #78.

## Results — 6 seeds (42/43/44/100/101/102), BCM ON vs LESION (`bcm_6seed.json`)

<!--derived-->
_Numbers from the cited artifact (rounded); weight-space margins are the dg→answer
answer-vote margin (correct − wrong over each engram)._

| seed | dgJ | ON both_win | LESION both_win | ON margins (m0,m1) | LESION margins (m0,m1) | ON private_m1 (corr,wrong) | LESION private_m1 | read m1→(J eng0, eng1) |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 42  | 0.56 | no | no | (+0.20, +0.99) | (−0.27, +0.27) | (7207, 7)   | (7, 7)   | (1.00, 0.56) |
| 43  | 0.47 | no | no | (+0.64, +1.00) | (+0.44, −0.46) | (16817, 17) | (17, 17) | (1.00, 0.47) |
| 44  | 0.53 | no | no | (+0.39, +0.99) | (−0.13, +0.12) | (20020, 20) | (20, 20) | (1.00, 0.53) |
| 100 | 0.56 | no | no | (+0.49, +0.99) | (+0.03, −0.03) | (15215, 15) | (15, 15) | (1.00, 0.56) |
| 101 | 0.48 | no | no | (+0.49, +0.99) | (+0.17, −0.15) | (8008, 8)   | (8, 8)   | (1.00, 0.48) |
| 102 | 0.60 | no | no | (+0.44, +0.99) | (+0.22, −0.26) | (8008, 8)   | (8, 8)   | (1.00, 0.60) |

**Pooled:** both_win ON **0/6**, LESION **0/6**; weight-space both-margins-positive ON
**6/6**, LESION **0/6**; private_m1 written (corr > wrong) ON **6/6**, LESION **0/6**;
read both-memories-reactivate-own-engram **0/6**; single-recall ceiling **6/6**; scramble
inverts **6/6**; dissimilar both_win ON **0/6** vs LESION **2/6** (a REGRESSION — the write
weakens the read). LESION per-seed margins + private/shared decomposition match the
committed `orthowrite_6seed.json` to 3 decimals (byte-identical baseline, exact compare).

## The decisive evidence: the WRITE lands but the READ reactivates the WRONG engram

Two facts, both universal across seeds, dissociate a fixed WRITE from a broken READ:

1. **The BCM write writes the discriminative signal (weight-space is fixed).** private_m1
   goes from baseline (corr == wrong, e.g. (7,7)) to strongly written (corr ≫ wrong, e.g.
   (7207, 7)); the anti-symmetry m0+m1 goes from ≈0.00 to +1.2…+1.6; both margins positive
   6/6; shared mass washed (e.g. 19044/32932 → 28/28). The weight-matrix reshape indexing
   was verified against the physical synapse coords (row = sorted granule, col = sorted
   answer, exact), so `private_m1_to_correct = 7207` IS the physical priv_m1→a1 synapses.

2. **The read reactivates the dominant engram, not the memory's own.** Driving m1's OWN
   input after consolidation reactivates `eng0` (m0's engram) exactly — Jaccard 1.00 to
   eng0, 0.47–0.60 to eng1 — on all 6 seeds; m0's input correctly reactivates eng0. So the
   read is decided by eng0's granules for BOTH probes; the written private-m1 granules
   never fire during m1's read, so the fixed weights are never consulted. Both memories
   pick m0's answer ⇒ the same anti-symmetric behavioral signature ⇒ both_win 0/6. Forcing
   priv_m1 to fire DIRECTLY still reads weakly / a0-biased (9–21 private granules cannot
   drive the answer assembly through the spiking read), which is why washing shared and
   relying on the private core is a net regression (dissimilar 2/6 → 0/6).

## Where the reactivation collapse comes from (characterized, locus not fully isolated)

The collapse is a persistent post-consolidation state, NOT the thing a write can reach:
it occurs with **plasticity forced OFF** (same replay activity, zero weight change → still
J(eng0)=1.00), is **independent of the synaptic time-step** (a fresh bridge reactivates the
correct engram at `current_time_step` 0/1/1036/1037), and is **not** restored by clearing
the #78 pop-controller integrator, the residual refractory timers, or the runner's
down-state reset (`_reset_dynamics`). A snapshot of every per-neuron `cp_*` numeric array
before vs after consolidation+reset shows NO load-bearing difference (only 6 refractory
timers + cosmetic viz timers), yet the read behavior differs — so the persistence lives in
an uncaptured global/instrument state the down-state reset does not clear. **This is the
CLAUDE.md instrument warning in the flesh:** the read is measured through a substrate the
"down-state" does not actually return to rest, and the prior two arcs tuned the WRITE for
weeks against a defect that lives in the READ. Even m0-only consolidation induces it (one
memory's heavy replay makes its engram capture the other memory's input afterward).

## The re-localization (corrects #readout again; names the next mechanism class)

#readout localized the residual to the WEIGHTS (under-written private granules), curable by
a selectivity write. That is now REFUTED as the *behavioral* blocker: the selectivity write
FIXES the weights (private written, anti-symmetry broken, both margins positive 6/6) and
both_win does not move. The residual re-localizes precisely: **at the biological-overlap
operating point, the #78 replay consolidation drives the DG into a persistent DOMINANT-
ENGRAM state in which the subordinate memory's input reactivates the dominant engram, so
the READ is performed through the wrong granules and no dg→answer weight — however
perfectly written — can be consulted.** The dg→answer WRITE family is now EXHAUSTED for
both_win: per-granule OUTPUT transform (#readout, banked) AND selectivity/BCM WRITE (this
finding, banked) both LAND in weight-space and neither moves the behavioral read. The next
mechanism must operate at READ / reactivation time, not at write time. Candidates, banked
for a new arc: (1) a CA3-style autoassociator that pattern-completes the retrieved input
onto the memory's OWN private core before the answer read (#readout candidate 4); (2)
characterize + clear the post-consolidation reactivation persistence (the down-state reset
is incomplete — an instrument fix that may itself unblock the read); (3) a read-time
attractor / novelty gate that prevents the dominant engram from capturing the subordinate
memory's input. The per-granule dg→answer write family is banked as INSUFFICIENT.

## Levers tried this arc (all measured; the dg→answer WRITE family is exhausted)

BCM selectivity write with heterosynaptic-LTD suppression of shared granules + potentiation
of memory-private granules toward their answer (writes private_m1 6/6, both margins positive
6/6, both_win 0/6); the suppression alone (private potentiation withheld, an early build)
washed shared to ~0 but left private at baseline → margins ≈ 0, both_win 0/6; the
potentiation initially failed because the selectivity was read off the post-consolidation
substrate (which reactivates the wrong engram) — corrected by reading it on a fresh twin.
Every write-side lever leaves the READ reactivating the dominant engram, so the weight fix
is unreachable.

## Tracked scaffolds (host, not brain)

Inherited from the #71/#78/#readout runners: host-defined input patterns + answer
assemblies; host reinstatement of each memory's input AND answer during replay (hippocampal
index / SWR trigger); scheduled down-states; the WRITE/READ transmission-gate phase; a
rate-window Hebbian coactivity write; an argmax over answer spike counts for MEASUREMENT
only; the population set-point PI controller (host); fixed random perforant projection +
fixed FS anatomy (not developed). NEW this runner: the BCM selectivity gate is a runner-side
transform on the plastic dg→answer weights; the memory-selectivity signal is read from a
FRESH TWIN bridge's isolated reactivation (on-substrate spiking); the twin build + the
gate arithmetic are host.

## Reproduce

    OMP_NUM_THREADS=4 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_bcm \
        --seeds 42 43 44 100 101 102 --bcm-gate \
        --out research/findings/raw/sep_readout/bcm_6seed.json

`--bcm-gate` enables the write (default OFF = byte-identical #78 baseline). `--gain` sets
the private-potentiation weight; `--supp-gamma` the heterosynaptic-LTD exponent on shared
granules; `--sel-thresh` / `--theta-scale` the BCM selectivity thresholds. The runner always
runs the LESION (`bcm_gate=False`) arm, the weight-space analysis (LESION vs ON margins +
private/shared decomposition), and the read-time reactivation-collapse block. Deterministic
(`cfg.seed`); 6-seed wall-clock ≈ 32 s (numpy, CPU).

## Sources

EXTERNAL-SEARCH-RAN: BCM sliding-threshold selectivity; heterosynaptic LTD of non-selective
inputs; pattern-completion / attractor reactivation in DG–CA3 (logged to the corpus-check
record, 2026-08-19).

- Bienenstock, E.L., Cooper, L.N., Munro, P.W. (1982). Theory for the development of neuron
  selectivity: orientation specificity and binocular interaction in visual cortex.
  J. Neurosci. 2:32–48. — the sliding-threshold selectivity rule built here (it writes the
  private granules, but the READ cannot reach the write).
- Chistiakova, M., Bannon, N.M., Chen, J.-Y., Rioult-Pedotti, M., Volgushev, M. (2014).
  Heterosynaptic plasticity: multiple mechanisms and their functions. Neuroscientist
  20:483–498. — the LTD of non-selective (high-cumulative-activity) granules.
- Marr 1971; O'Reilly & McClelland 1994; Guzman et al. 2016 — CA3 recurrent autoassociation
  / pattern completion (the next mechanism class: fix the READ reactivation, not the write).

Internal: builds on and CORRECTS
`2026-08-19-memory-separator-readout-competitive-write-NOGO-relocalizes-to-private-granule-underwrite.md`
(#73 — its "under-written private granule, curable by a selectivity write" localization is
the behavioral blocker is refuted here; the selectivity write fixes the weights but the
residual re-localizes to read-time reactivation). Preconditioned on the #78 population
set-point (`_replay_dg_pattern_separation_popsetpoint.py`) and the #71 write fixes.
