# objrel DANN (Dale-legal signed read-out) — the EMERGENT-LEARNING BOUNDARY; but a Dale-legal spiking read that holds both is now SUGGESTED (analytic reference, verify pending)

**Date:** 2026-07-06
**Runner:** `research/runners/_rungB1c_objrel_dann_readout_derisk.py`
**Raw:** `research/findings/raw/_rungB1c_objrel_dann_readout.json`
**Verdict:** BOUNDARY (the *emergent* learning), + a strong existence-suggestion (the analytic reference).
**Builds on the diagnosis:** `2026-07-06-objrel-trained-readout-NOT-surpass-DALE-SHIFT-diagnosis.md` (the Dale-shift destroys the sign; a Dale-legal signed read is the residual).

## The mechanism + the result

Per the Dale-shift diagnosis, the objrel read needs to carry the NEGATIVE ridge rows without the Dale-shift destroying the
sign. The Dale-LEGAL way (biology + the DANN literature — biorxiv 2025.01.09.632231) is a signed read via a POPULATION of
inhibitory interneurons: reservoir feature ≥0 → [E path: excit → output LIF] + [I path: excit → 48 inhibitory-interneuron
LIF → inhibitory → output LIF], all weights sign-clipped (Dale-legal) + BPTT-trained; read = output-LIF spike-count argmax,
like-for-like vs the fixed spiking WTA. NO sim/ edit. Confound-proofed (genuinely spiking, Dale-legal asserted, no host
ridge argmax, no signed output weights — neither prior retraction repeated).

**6-seed-blind aggregate:**

| read | canon (mean) | objrel-slot0 (mean) |
|---|---|---|
| fixed spiking WTA (baseline) | 0.44 | 0.5 |
| **DANN (BPTT-trained from scratch)** | 0.83 | **0.00** |
| **ANALYTIC Dale reference** (ridge E/I split, graded op-point, NOT trained) | **1.00** | **1.00** |
| DANN + inhibition silenced | — | (analytic inh-silence **0.0** on all 6 → inhibition load-bearing) |
| 0-epoch random Dale-init | 0.33 | 0.167 (seed-42 init-lucky outlier; else 0) |

**VERDICT: BOUNDARY (the emergent learning).** `objrel_recovers_gate False`, `bptt_does_work_all False` — the DANN trained
from a random Dale-init recovers objrel on **0/6** seeds. Genuinely-spiking + Dale-legal are True on all 6 (the build is
clean); the failure is that **surrogate-gradient BPTT from scratch cannot REACH the signed-THEME solution.**

## Why (the precise, load-bearing residual)

slot0 carries a **7:1 canonical-AGENT : object-relative-THEME class imbalance** (most constructions put an AGENT at slot0;
only objrel puts a THEME there). From a random Dale-init, gradient descent converges to the MAJORITY (AGENT) read on the
shared slot0 and never finds the minority signed-THEME direction — canon holds, objrel stays 0. Warm-starting from the
ridge would be the retracted inert-BPTT confound, so it is correctly refused. **This is a training-REACHABILITY boundary
under class imbalance — NOT a substrate, representation, or Dale's-law wall.**

## What is now SUGGESTED (the analytic reference — a strong hint, verify pending)

The analytic Dale reference (the same ridge discriminant split by sign into an excitatory path + a genuine inhibitory-
interneuron population, deployed at a graded op-point) reads canon 1.0 AND objrel-slot0 1.0 on ALL 6 seeds, genuinely on
spikes, Dale-legal, with its own inhibition LOAD-BEARING on all 6 (silence → 0.0). ⇒ a Dale-legal, spike-native read that
holds both roles very likely EXISTS (the substrate + Dale's law are not the wall). **HONEST SCOPE / NOT YET A CLAIMED
SURPASS:** the analytic reference is (a) ridge-based (non-emergent — the read-out weights are host-computed by ridge, not
learned by the substrate), and (b) not yet adversarially verified against the ridge-re-expression critique that retracted
two prior GOs (though unlike those it IS Dale-legal with load-bearing inhibition, which is materially different). It is
recorded as a strong existence-SUGGESTION, not a verified surpass.

## The next mechanism (this launches it)

The residual is now precisely the EMERGENT learning of a minority signed direction under class imbalance — squarely the
master-directive core (emergent, from experience), and a known family. Research-gate + de-risk (cheap-first): (1)
class-balanced / minority-oversampled or minority-margin-weighted training; (2) a curriculum (teach the objrel construction
in isolation first); (3) a biologically-grounded plasticity rule (dopamine-gated three-factor / reward on the minority read)
instead of plain BPTT — the striatum learns exactly this way; (4) a basin-escaping init that is NOT the ridge (to avoid the
inert-BPTT confound). Also: adversarially verify the analytic reference (with controls designed for a FIXED read) to settle
whether the Dale-legal spike-native existence is genuine or a ridge re-expression.

## Files
- `research/runners/_rungB1c_objrel_dann_readout_derisk.py` — the DANN de-risk (Dale-legal, genuinely spiking; NO sim/ edit).
- `research/findings/raw/_rungB1c_objrel_dann_readout.json` — the 6-seed-blind record + the analytic reference + ablations.
