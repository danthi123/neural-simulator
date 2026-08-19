---
type: finding
status: no-go
date: 2026-08-19
runner: research/runners/_replay_dg_pattern_separation_readout.py
artifacts:
  - research/findings/raw/sep_readout/orthowrite_6seed.json
---

# A COMPETITIVE (heterosynaptic) dg→answer WRITE does NOT close the memory-separator readout residual — the collision is a near-perfect ANTI-SYMMETRY whose discriminative (memory-private) granules are UNDER-WRITTEN, so no per-granule OUTPUT transform can separate it

**Board #73** — the memory-separator residual. #78
(`2026-08-19-dg-population-setpoint-NOGO-relocalizes-residual-to-readout.md`) RE-LOCALIZED the
`both_win` blocker OUT of DG competition (three DG-side mechanisms exhausted: k-WTA #71, per-cell
homeostat #73, population set-point #78) INTO the **dg→answer readout/write**, and named a
"competitive/normalised write" as the top untried lever, hypothesising that "the soft-bound
rate-window rule writes overlapping-engram granules to BOTH answers". This finding builds and tests
exactly that lever and reports a DECISIVE NEGATIVE that CORRECTS #78's mechanistic hypothesis.

## Verdict

**NO-GO (both_win 0/6), and the negative is decisive + re-localizing.** A presynaptic (granule-output)
heterosynaptic renormalization of the plastic dg→answer weights — interleaved between replay events,
built on top of the #78 population set-point so the DG code is already symmetric + non-dense — does
NOT make two similar memories both-discriminable. All PIPELINE controls pass (single-memory recall at
ceiling 6/6; scramble-teach inverts 6/6; LESION reproduces the #78 residual; the write demonstrably
alters the learned map), so the verdict is DEFINED, not UNDEFINED. A WEIGHT-SPACE analysis shows the
mechanism cannot work IN PRINCIPLE at this operating point: across heterosynaptic exponents
{0.5, 1.0, 3.0}, NO exponent makes both memories' answer-vote margins positive (both-positive 0/6),
because the readout collision is a near-perfect ANTI-SYMMETRY (m0-margin ≈ −m1-margin; the two engrams
share 47–60% of their granules and those SHARED granules carry the net answer bias) whose only cure —
a strong memory-PRIVATE write — is ABSENT: the granules private to one memory's engram carry pure
BASELINE weight (unwritten). No `sim/` edit; the renorm is a runner-side transform on the same
dg→answer weight vector the runner already reads via `_path_weights`; the selection + read stay
on-substrate spiking. Deterministic (`cfg.seed`); the `orthowrite=False` LESION arm is BYTE-IDENTICAL
to the committed #78 ON arm on all 6 seeds (per-memory selectivity matches to 1e-12).

## What was built (the named mechanism, no `sim/` edit)

A **presynaptic heterosynaptic renormalization** of the plastic dg→answer weights: after every replay
event (the offline down-state window), each granule conserves a FIXED total outgoing dg→answer budget,
redistributed across its answer targets as `w0 + budget · Eᵖ / ΣEᵖ` (E = learned excess above the
0.05 baseline; p the competition exponent). p=1 conserves budget (a granule that tried to write to two
answers has its budget split ~half each → it cannot out-vote the memory-selective private granules);
p>1 sharpens toward a granule-output WTA; p<1 compresses toward uniform (equalizes the two answers a
granule wrote to). Interleaved with the #71 soft-bound rate-window write and stacked on the #78
population set-point (engrams symmetric, non-dense — the #78 GO precondition), so the ONLY manipulated
variable vs the #78 ON arm is the WRITE rule. Biology: heterosynaptic plasticity / synaptic
competition as a conserved-total normalizer (Royer & Paré 2003; Chistiakova et al. 2014; von der
Malsburg 1973; Oja 1982). The `orthowrite=False` LESION is a no-op → byte-identical to #78.

## Results — 6 seeds (42/43/44/100/101/102), ON (p=1) vs LESION (#78) (`orthowrite_6seed.json`)

<!--derived-->
_Numbers from the cited artifact (rounded)._

| seed | dgJ | ON both_win | ON per-mem sel (m0,m1) | LESION both_win | single | scramble |
|---:|:--:|:--:|:--:|:--:|---:|---:|
| 42  | 0.56 | no | (−0.30, +0.27) | no | +0.97 | −0.93 |
| 43  | 0.47 | no | (+0.58, −0.58) | no | +1.00 | −1.00 |
| 44  | 0.53 | no | (−0.02, +0.05) | no | +1.00 | −1.00 |
| 100 | 0.56 | no | (+0.15, −0.11) | no | +1.00 | −1.00 |
| 101 | 0.48 | no | (+0.22, −0.23) | no | +0.88 | −0.92 |
| 102 | 0.60 | no | (+0.13, −0.12) | no | +1.00 | −0.92 |

**Pooled:** both_win ON **0/6**, LESION **0/6**; single-recall ceiling **6/6**; scramble inverts
**6/6**; the ON per-memory read stays ANTI-SYMMETRIC (whichever memory reads +x, the other −x). The
competitive write LANDS (it demonstrably alters the learned map — direct-readout selectivity changes
ON vs LESION) but does not move both_win. dissimilar both-win 1/6 (a separate, pre-existing weakness of
this bridge port, unrelated to the write rule).

## The decisive evidence: WEIGHT-SPACE, no exponent orthogonalizes the map (both-positive 0/6)

Reading the learned dg→answer matrix directly and computing each memory's answer-vote margin
(correct − wrong, over its engram), then applying the renorm at exponents {0.5, 1.0, 3.0}:

<!--derived-->
| seed | raw margin (m0,m1) | anti-sym m0+m1 | p0.5 both-pos | p1 both-pos | p3 both-pos | private-m1 (corr,wrong) | shared (→a0,→a1) |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 42  | (−0.27, +0.27) | +0.00 | no | no | no | (7, 7)   | (19044, 32932) |
| 43  | (+0.44, −0.46) | −0.01 | no | no | no | (17, 17) | (31313, 11694) |
| 44  | (−0.13, +0.12) | −0.01 | no | no | no | (20, 20) | (16215, 20650) |
| 100 | (+0.03, −0.03) | +0.00 | no | no | no | (15, 15) | (19576, 18350) |
| 101 | (+0.17, −0.15) | +0.02 | no | no | no | (8, 8)   | (26319, 19563) |
| 102 | (+0.22, −0.26) | −0.03 | no | no | no | (8, 8)   | (40169, 23693) |

Two structural facts kill the mechanism, both universal across seeds:

1. **The collision is a near-perfect ANTI-SYMMETRY** (m0-margin ≈ −m1-margin; the sum is 0.00±0.03).
   The two engrams share 47–60% of their granules, so their answer-vote vectors are nearly identical;
   whatever the SHARED granules lean toward, BOTH engrams read it → one memory correct, the other
   backward. A per-granule OUTPUT transform reshapes each granule's answer distribution but cannot
   change WHICH answer the shared mass points to → the anti-symmetry survives every exponent.

2. **The memory-PRIVATE granules are UNDER-WRITTEN.** The granules private to m1's engram carry EQUAL
   correct/wrong weight on EVERY seed (e.g. (7,7), (17,17), (20,20)) = pure 0.05 baseline: they were
   never written to their memory's answer during consolidation. The discriminative signal that could
   break the anti-symmetry is absent from the weights, so no re-weighting of the OUTPUT can recover it.

The p<1 (equalize) branch DOES neutralize the shared bias (margins → ~0) but is not a valid operating
point: sqrt-compression amplifies baseline crosstalk and DESTROYS single-memory recall (measured:
single selectivity collapses +1.00 → +0.07). p≥1 preserves single recall but leaves the anti-symmetry
intact. There is no exponent that both preserves the pipeline AND orthogonalizes the two answer maps.

## The re-localization (corrects #78's hypothesis; names the next mechanism)

#78 hypothesised the blocker is shared granules cross-writing to BOTH answers, curable by a competitive
write. That is REFUTED: the read-vs-write engram Jaccard is 1.00 (reactivation is consistent — read
engram = write engram), so this is not a reactivation-inconsistency problem either; and the shared
granules' cross-write is not the load-bearing defect — even a perfect equalization of the shared mass
(p<1) leaves both_win at 0 because the PRIVATE granules carry no write to tip the read. The residual
re-localizes precisely: **at the biological-overlap operating point (DG Jaccard ~0.5, engrams >50%
shared), the strongly + stably firing SHARED granules dominate the soft-bound write while the weakly /
marginally firing memory-PRIVATE granules — the only discriminative synapses — barely potentiate. The
read is therefore decided by the non-discriminative shared mass.** The next mechanism must AMPLIFY the
private write or SUPPRESS the shared one at write time using a signal that distinguishes them — i.e.
NOT a per-granule output normalization. Candidates, banked for a new arc: (1) a
selectivity / novelty-gated write (a granule active across MANY replay events — high cumulative
activity, hence non-specific — receives heterosynaptic LTD; the memory-selective granules, active in
one memory's events only, are spared — BCM-like sliding threshold on the granule's own activity);
(2) a per-granule INPUT-side write normalization so every ACTIVE granule writes a fixed amount
regardless of firing rate (equalizes the marginal private granules with the strong shared ones);
(3) push DG overlap BELOW the shared-dominance threshold so private > shared (a DG-side lever #78
banked as "exhausted for competition" but never run WITH a working readout — private-fraction, not
Jaccard, is the quantity that matters); (4) a CA3-style autoassociator to complete the retrieved
pattern onto the private core before the read. The per-granule OUTPUT-transform family is banked as
INSUFFICIENT (this finding).

## Levers tried this arc (all measured; the per-granule OUTPUT-transform family is exhausted)

Presynaptic budget-conservation write (p=1 — anti-symmetry intact, both_win 0/6); granule-output WTA
(p=3 — sharpens magnitude but keeps the sign, both_win 0/6); granule-output equalization (p=0.5 —
neutralizes the shared bias but destroys single-memory recall, invalid); renorm cadence per-event vs
per-pair vs end-only (all leave the anti-symmetry; weight-space margins move <0.05); AUTO vs fixed
per-granule budget (the margin SIGN is budget-independent by construction — only the private-vs-shared
dilution ratio matters, and it does not flip the anti-symmetry). Every OUTPUT-side lever leaves the
anti-symmetric readout intact because the discriminative private write is missing upstream.

## Tracked scaffolds (host, not brain)

Inherited from the #71/#78 runners: host-defined input patterns + answer assemblies; host reinstatement
of each memory's input AND answer during replay (hippocampal index / SWR trigger); scheduled
down-states; the WRITE/READ transmission-gate phase (host-scheduled sleep/wake); a rate-window Hebbian
coactivity write; an argmax over answer spike counts for MEASUREMENT only; the population set-point PI
controller (host); a fixed random perforant projection + fixed FS anatomy (not developed). NEW this
runner: the heterosynaptic renormalization is applied as a host-side transform on the plastic dg→answer
weights (the offline down-state renorm); the SELECTION (which granules survive the divisive basket) and
the READ stay on-substrate spiking.

## Reproduce

    OMP_NUM_THREADS=4 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_readout \
        --seeds 42 43 44 100 101 102 --exponent 1.0 \
        --out research/findings/raw/sep_readout/orthowrite_6seed.json

`--exponent {0.5,1.0,3.0}` sets the heterosynaptic competition exponent; `--every N` the renorm
cadence; `--budget` a fixed per-granule budget (default AUTO). The runner always runs the LESION
(`orthowrite=False`, byte-identical to #78) arm for the dissociation and the weight-space analysis
block (raw + exponent sweep, private/shared decomposition).

## Sources

EXTERNAL-SEARCH-RAN: heterosynaptic plasticity / synaptic competition as a conserved-total normalizer;
per-granule output vs input normalization; BCM sliding-threshold selectivity (logged to the
corpus-check record, 2026-08-19).

- Royer, S., Paré, D. (2003). Conservation of total synaptic weight through balanced synaptic
  depression and potentiation. Nature 422:518–522. — the heterosynaptic conserved-total renormalizer
  built here (it lands, but does not close both_win).
- Chistiakova, M., Bannon, N.M., Chen, J.-Y., Rioult-Pedotti, M., Volgushev, M. (2014). Heterosynaptic
  plasticity: multiple mechanisms and their functions. Neuroscientist 20:483–498. — activity-dependent
  heterosynaptic LTD (the NEXT mechanism: penalize non-selective, high-activity granules).
- von der Malsburg 1973; Oja 1982 — weight normalization / competitive learning (the output-transform
  family, banked insufficient here).
- Bienenstock, Cooper, Munro 1982 (BCM) — the selectivity sliding-threshold named for the next arc.

Internal: builds on and CORRECTS
`2026-08-19-dg-population-setpoint-NOGO-relocalizes-residual-to-readout.md` (#78 — its "shared granules
cross-write to both answers, curable by a competitive write" hypothesis is refuted here; the residual
re-localizes to the under-written memory-private granules). Preconditioned on the #78 population
set-point (`_replay_dg_pattern_separation_popsetpoint.py`) and the #71 write fixes
(`2026-08-19-replay-separator-bridge-rebound-and-write-runaway-FIXED-...md`).
