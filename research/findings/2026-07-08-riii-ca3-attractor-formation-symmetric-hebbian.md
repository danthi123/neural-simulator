---
type: finding
status: superseded
superseded_by:
  - research/findings/2026-07-09-riii-formation-rules-saturate-ensemble-dynamics-is-the-blocker.md
  - research/findings/2026-07-14-ca3-competitive-hebbian-formation-6seed-GO.md
date: 2026-07-08
mechanism: ca3-attractor-formation
---

# R-iii CA3 attractor FORMATION — the CYCLE-1066 residual root-caused + fixed: the bridge's Hebbian is CAUSAL-OFFSET (pre fired at t-1 AND post at t), but co-ensemble CA3 members driven by one pattern fire SYNCHRONOUSLY, so the offset is ~never satisfied → the recurrent LTP produces ~ZERO potentiation (byte-identical across a 100× learning-rate sweep × decay-off × drive). The fix is a guarded, default-off SYMMETRIC (offset-free) co-activity Hebbian (`hebbian_symmetric`), which potentiates within-ensemble preferentially (separation +0.01 → +0.87). The symmetric rule forms a SPECIFIC but WEAK attractor (within-ensemble ~1.14× member→silent; the per-step co-spike frequency is the bottleneck — members fire asynchronously). NO existing behavior changed (byte-identical default-off; `test_determinism.py` 7/7).

**Date:** 2026-07-08
**Runner:** `research/runners/_riii_ca3_attractor_diag.py` (within-ensemble vs member→silent weight separation) + `_riii_ca3_coincidence_completion_derisk.py`. `sim/` edit: `config.hebbian_symmetric` (default False) + a guarded branch in the bridge Hebbian block. GPU.
**Verdict:** mechanism GO for FORMATION (symmetric rule forms a specific attractor where the causal rule formed none); the attractor is WEAK and the completion payoff is [PENDING — see below].

## The blocker, root-caused (source-read + 3 byte-identical sweeps)
CYCLE 1066 found the ca3→ca3 rate-Hebbian did not write a specific within-ensemble attractor (held c_drive ≈ non). Reading the bridge Hebbian rule MYSELF (bridge.py:6940-6954, per the read-the-source discipline): the "active" synapse set is `where(prev_firing[row] & fired_this_step[col])` — a CAUSAL 1-step offset (pre at t-1 AND post at t). Co-ensemble CA3 members are driven by the SAME input pattern → they fire ~SYNCHRONOUSLY (same/overlapping steps, not a 1-step pre→post offset) → the offset coincidence is ~never satisfied. Confirmed decisively — the within-ensemble weight is BYTE-IDENTICAL across:
- learning-rate ∈ {0.0005, 0.005, 0.02, 0.05} (100× range) — within 4.87-4.88, separation +0.03 flat;
- decay ∈ {default, 0} — within 6.03 with decay off, separation +0.01 (weights just sit at init → ZERO potentiation);
- encoding drive ∈ {100, 200} pA — byte-identical (drive is not the sparsity/firing lever either).
⇒ the causal-offset rule produces ~ZERO recurrent potentiation on synchronously-firing CA3. This is the CYCLE-95/96 finding ("STDP/offset rules are WRONG for symmetric co-occurrence; Δt≈0 → 0 weight change") applied to the general connection Hebbian.

## The fix (biology-grounded, guarded, byte-safe)
A SYMMETRIC (offset-free) co-activity option: potentiate synapses where pre AND post fire in the SAME step (`fired_this_step` for both). Biology: CA3 recurrent (associational/commissural) LTP is associative/Hebbian — "fire together, wire together" (Kandel 6e Ch 54; Marr 1971 autoassociator); it does not require a causal pre→post offset. `sim/` edit: `CoreSimConfig.hebbian_symmetric: bool = False` (config.py) + a guarded branch in the bridge Hebbian block (bridge.py:6942) selecting `fired_this_step[row]` (symmetric) vs `prev_firing[row]` (causal, the default). **Default-off is byte-identical** (the False branch is the original line verbatim; `test_determinism.py` 7/7 pass).

## De-risk (seed 42) — the symmetric rule forms a SPECIFIC but WEAK attractor
```
rule / config                        within-ens   member->silent   separation
CAUSAL (default), lr 0.0005-0.05      4.87         4.84             +0.03   (no attractor)
CAUSAL, decay-off                     6.03         6.02             +0.01   (zero potentiation)
SYMMETRIC, lr 0.05, decay-off         6.79         6.02             +0.77   (specific, weak)
SYMMETRIC, lr 0.10, decay-off         6.89         6.02             +0.87   (specific, weak; lr saturating)
```
The symmetric rule potentiates within-ensemble ABOVE init while member→silent stays at init → a genuine SPECIFIC attractor (member↔member preferentially strengthened). But it is WEAK (~1.14× member→silent vs the hand-installed 10× of CYCLE 1068), and lr saturates (+0.77 → +0.87 from 2× lr) → the bottleneck is the NUMBER of co-spike events: the CA3 members fire ASYNCHRONOUSLY (sparse, not step-locked), so even the same-step symmetric coincidence fires rarely. The robust amplifier is a RATE-WINDOW co-activity (accumulate pre×post over the encoding window, not per-step) — the true CYCLE-95/96 rate-Hebbian — OR many more encoding events.

## Payoff (does the LEARNED attractor + dendritic completion complete?) — NEGATIVE (the attractor is too weak)
Train with the symmetric Hebbian, then the CYCLE-1068 dendritic dAP completion (two_comp, apical_R=50, k=3) on the LEARNED recurrents (seed 42):
```
COINC-ON held-out = 1.047   LINEAR-OFF = 0.018   NO-TRAIN = 0.976   non-stored = 0.765
c_drive[held = 52.5   non = 53.1]   <-- NO within-ensemble c_drive separation
```
The decisive number is **c_drive[held] ≈ c_drive[non] (52.5 ≈ 53.1)**: the +0.87 within-ensemble weight advantage is SWAMPED by the connectivity-driven c_drive variance (each held-out member and each non-member connects to ~the same number of cue partners; a ~14% per-synapse weight difference does not survive the sum). With no c_drive separation, a low k_thresh makes the plateau fire EVERYTHING cue-connected (non-stored 0.765; and NO-TRAIN 0.976 completes too — the trigger is the raw cue fan-in, not the learned attractor) — indiscriminate spread, NOT specific completion. This is the exact OPPOSITE of CYCLE 1068's hand-installed attractor (c_drive held 80 vs non 7.5 = 10× separation → clean specific completion). ⇒ the symmetric rule's WEAK attractor (~1.14×) is insufficient for the dendritic completion, which requires a strong within-vs-non c_drive separation.

## NEXT — STRENGTHEN the attractor formation (the honest gap)
The completion half (CYCLE 1068) works on a STRONG attractor; the formation half (this cycle) forms only a WEAK one. The bottleneck is the co-spike EVENT COUNT: the symmetric rule fires only when members co-spike in the SAME step, but the CA3 members fire ASYNCHRONOUSLY → rare per-step co-spikes → limited potentiation (and lr saturates: +0.77→+0.87 from 2× lr). Two ranked levers, cheap-first:
1. **MORE encoding events** (500-1000): each per-step co-spike adds `lr·(max−w)`, so with enough events the within-ensemble weight climbs toward `hebbian_max_weight`=30 while member→silent stays at init → a growing separation. Cheap to test (a train_events sweep on the diag), no new mechanism. [Launched: the strengthening test.]
2. **A RATE-WINDOW co-activity rule** (the robust CYCLE-95/96 rate-Hebbian): accumulate pre×post co-activity over the whole encoding window (not per-step), potentiate proportionally — fires far more events than per-step spike coincidence, so it reaches a strong separation regardless of the asynchronous firing. A further guarded `sim/` addition (per-synapse co-activity accumulator) if (1) saturates too low.
Then re-run the payoff (learned attractor + dendritic completion) → if the strong learned attractor gives a c_drive separation like CYCLE 1068's hand-installed one, the completion should fire → fully emergent CA3 pattern completion on-substrate → unblocks the SWR generative-replay loop.

## Files
`sim/config.py` (`hebbian_symmetric`), `sim/bridge.py:6942` (guarded branch), `research/runners/_riii_ca3_attractor_diag.py` (+ `--hebb-sym/--hebb-lr/--hebb-decay/--drive-pA`), `_riii_ca3_coincidence_completion_derisk.py` (threaded). Prior: `2026-07-08-riii-onsubstrate-coincidence-wired-but-blocked-by-missing-attractor.md` (1066), `-onsubstrate-dendritic-dAP-completion-SURPASS-6seed.md` (1068). Biology: Kandel 6e Ch 54 (CA3 associative LTP), Marr 1971; CLAUDE.md CYCLE 95-96 (rate-Hebbian for co-occurrence).
