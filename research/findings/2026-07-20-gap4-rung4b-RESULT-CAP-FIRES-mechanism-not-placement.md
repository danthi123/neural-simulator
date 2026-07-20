# gap#4 RUNG 4b — RESULT: attempt 2 FAILS. **The pre-registered CAP FIRES.** No third derivation.

Pre-registered at `09811843` before the run, on untouched seeds 306-311, with an explicit two-attempt cap stated
first precisely so it could not be quietly extended.

## Result

| arm | c_adj | c_far | `map_ok` | dw |
|---|---|---|---|---|
| `P4_bandOFF` (control) | **1.213** | **2.609** | **1** (6/6) | 445.5 |
| `MAIN_bandON` | 1.000 | ~1.000 | **0 (0/6)** | 272-1359 |

| prediction | bar | result |
|---|---|---|
| **P0** stage 1 survives | >= 5/6 | **0/6 — FAILED** |
| P1 adjacent contrast >= 1.60x | >= 5/6 | unmeasurable (no map) |
| P2 far contrast >= 2.0x | >= 5/6 | unmeasurable |
| P3 trough moves to adjacent | >= 5/6 | unmeasurable |
| **P4** band-OFF reproduces | 6/6 | **6/6 — PASSED** |

## The cap fires, as filed

> *"If this second pre-registered band also fails, I do NOT derive a third. The verdict becomes: the adjacent-band
> depression MECHANISM (not merely a placement) is in question, and the next step is a research gate on the
> mechanism, not another set of thresholds."*

**Invoked.** No third band is derived. Two independent, principled placements — one anchored to the lag curve, one
anchored to the eligibility distribution's mass floor — both destroy field formation. That is now a statement about
the mechanism as implemented, not about parameter choice.

## Two diagnostics that sharpen what failed

1. **It is NOT silencing plasticity.** With the band ON, `dw` is 272-1359 — comparable to or LARGER than band-OFF's
   445.5. Weights move plenty; they simply do not organize into a field. So the failure is not "depression removed
   the drive" (attempt 1's mode, where `dw` was exactly 0) but **"depression disrupted the structure."**
2. **It introduces seed-dependence where there was none.** Band-OFF gives `l2_peak = 8` on all six seeds; band-ON
   gives 4, 14, 14, 14, 7, 12. A coherent field is replaced by a noise-determined argmax — the signature of no
   field at all rather than a differently-placed one.

⇒ Refined statement of the failure: **an eligibility-magnitude band cannot select "the adjacent lag" without also
selecting synapses essential to forming the peak.** Eligibility magnitude is not a clean proxy for lag, because at
any instant the same eligibility value is shared by synapses at very different lags with very different roles.

## What stands, and what is now open

**STANDS (unaffected):** the contrast asymmetry itself, reproduced 6/6 on fresh seeds by `P4_bandOFF` — adjacent
1.213x, far 2.609x. The blocker's localization is solid; only the first proposed fix is refuted.

**OPEN (for the research gate, NOT another threshold):** if eligibility magnitude cannot separate adjacent from
peak, what carries lag information that a local rule can read? Candidates already ranked by the prior gate and now
promoted by this failure:
- **Rank 2 — the zero-DC difference-of-exponentials kernel.** It does not attempt to SELECT a lag band at all; it
  cancels the pedestal algebraically via a signed two-trace difference, so the "cannot separate by magnitude"
  objection does not apply to it. This is now the leading candidate.
- **Rank 4 — mean-subtracted increment** (Miller-MacKay): `sum_j dw_ij = 0` by construction, also no lag selection.
- **Rank 5 — replacing traces** as a prerequisite: the current trace is dominated by spike COUNT, not timing, which
  is plausibly WHY eligibility magnitude fails to encode lag cleanly. **This failure elevates Rank 5 from hygiene to
  a possible root cause.**

## Process note

Two pre-registered attempts, two honest failures, no tuning. The temptation each time was to nudge two numbers until
a pass appeared; the cap existed to make that decision in advance rather than in the moment. The cost was one extra
run; the benefit is that "an eligibility-magnitude band cannot do this" is now a *result* rather than an excuse.

---

## ⛔ SELF-CORRECTION (same day, ~40 min later): my "refined failure statement" above is WRONG

The block above concluded:

> *"an eligibility-MAGNITUDE band cannot select 'the adjacent lag' ... because at any instant the same eligibility
> value is shared by synapses at very different lags with very different roles. Magnitude is not a clean proxy for
> lag."*

**That is FALSE, and I measured it rather than continuing to assert it.** A root-cause test (Rank 5's hypothesis:
is the trace dominated by spike COUNT rather than timing?) returns:

- **corr(eligibility, LAG) = −0.9445** — eligibility encodes lag almost perfectly;
- corr(eligibility, spike count) = undefined, because **every pool fires exactly 250 spikes** — count has zero
  variance in this task and cannot confound anything.

Eligibility is a clean monotonic function of lag: 0.001685 at lag 13 rising smoothly to 0.022681 at lag 0.
**Magnitude IS a clean proxy for lag here.** Both my failure statement and Rank 5's count-domination concern are
refuted *for this task* (Rank 5 may still matter where firing rates vary; here they do not).

## THE ACTUAL CAUSE — a geometric collision, not a proxy failure

Mapping the band onto lags directly:

| lag | eligibility | in band? | role |
|---|---|---|---|
| 0 | 0.022681 | | plateau bin |
| 1 | 0.018570 | | |
| 2 | 0.015204 | **IN** | |
| 3 | 0.012448 | **IN** | |
| **4** | **0.010191** | **IN** | **← where the field actually forms** |
| **5** | 0.008344 | | **← where the field actually forms** |
| **6** | 0.006831 | | **← where the field actually forms** |

The measured backward shift is **−4 to −6 bins**, so the CA1 field forms from synapses at lags 4-6. **The band
covers lags 2-4 — so it depresses the very synapses that build the field.** That is why `map_ok=0` while `dw`
stayed large: the rule was vigorously depressing the field's own inputs.

**And the structural reason no placement fixes it at this geometry:**

    field spacing (CELL_TARGETS)  = 4 bins
    measured backward shift       = 4-6 bins
    ⇒ "the ADJACENT field" and "where THIS field forms" are THE SAME LAG.

At spacing 4 the target of the depression and the source of the field are the same synapses. **No band in
eligibility space can separate them, because they are not separated in lag space either.** The cap's verdict
("mechanism, not placement") stands — but the reason is now precise and, importantly, *falsifiable*: it predicts
the band would work at a geometry where **field spacing > backward shift** (e.g. spacing 8, where the adjacent
field sits at lag 8 and the field forms at lag 4-6).

## Scope discipline on that prediction

That is a **new experiment with a different geometry**, not a third derivation of the band — the cap forbids
re-tuning the thresholds to rescue the result, and this does not do that (the band logic is unchanged; the task
geometry changes). It must still be pre-registered before running, and it is honestly a weaker claim: it would show
the mechanism can work *when the geometry permits*, not that it solves the adjacent-contrast problem at the
geometry where the problem was measured.

**Recorded as: the cap stands, the mechanism is not refuted, and the failure has a precise geometric explanation
that makes a falsifiable prediction.**
