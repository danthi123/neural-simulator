# gap#5 RANK 2 — the within-reactivation blocker is SOLVED (it was the per-assembly boundary `_silence_soma_apical + _zero_elig`), and the residual is a precisely-characterized within/chain co-existence tension

**2026-07-22, single-seed mechanism localization (CPU/GPU-auto, coexisting with the fluency training).** This SUPERSEDES
the retracted "deeply-elusive rest-phase / encode-ruled-out" verdict (2026-07-22-gap5-RANK2-verbatim-reuse-RULES-OUT-encode,
already corrected once for an n_mem=1 confound). A verify-not-assume single-variable ladder pinned the true cause.

## The corrected chain of evidence (each a single-variable step, seed 42, n_mem=2, same noise r=0.015 pA=1500 dur=10)

| Test | Config | Reactivation | Interpretation |
|------|--------|--------------|----------------|
| n_mem matrix | RANK 1 driver | n1 NO, **n2 YES, n3 YES** | reactivation needs n_mem≥2 (the earlier n_mem=1 comparison was confounded — RANK 1 also fails at n_mem=1) |
| A | RANK 2 disjoint, NO chain (w_within 15.2) | **NO** (events=0) | rules out chain-erosion as the *sole* blocker — pure within also fails |
| encode-only | RANK 2 disjoint +chain | w_within 15.2→**6.3** | the CHAIN phase ERODES the within-attractor 2.4× (transient-plateau + per-window theta silence) |
| B | RANK 2 **overlap-draw**, NO chain (w_within 22.9) | **NO** (events=0) | rules out disjoint-vs-overlap draw structure — the same draw reactivates on RANK 1's driver but not RANK 2's |
| B′ | RANK 2 overlap + 30 events + no chain (EXACT RANK 1 config) | **NO** (events=0) | ⇒ the divergence is a genuine CODE difference in `_prepare_sequence`, not a parameter |
| **C** | **+ `--rank1-encode`** (overlap, 30ev, no chain) | **YES — events=8, asm_active=[4,4]**, NO-NOISE=0 | **the per-assembly boundary `_silence_soma_apical + _zero_elig` is the blocker** |
| **D** | `--rank1-encode` **DISJOINT**, no chain | **YES — events=5, asm_active=[1,1]**, NO-NOISE=0 | the fix works for the order-preserving DISJOINT draw RANK 2 needs |

**The blocker (`_prepare_sequence` within-encode):** RANK 2 inserted a per-assembly `_silence_soma_apical(settle=0) +
_zero_elig` at each assembly boundary (to prevent spurious cross-links). RANK 1's proven `_prepare` uses a plain nested
loop that KEEPS eligibility across assemblies and never clears the soma/apical state at boundaries. That per-assembly
clear is what prevents the disjoint (and even overlapping) within-attractors from spontaneously reactivating under weak
noise. Removing it (`--rank1-encode`, additive default-off flag) restores reactivation for BOTH draws — the exact same
`_prepare`-reuse the pre-reboot board had queued, now isolated to the two specific boundary calls and validated.

## The residual — a precisely-characterized within/chain co-existence tension (the full A→B sequence)
RANK 2's *deliverable* is ordered sequence replay = within-reactivation (each assembly reactivates) AND a forward chain
(A→B ≫ B→A so replay runs forward). The two mechanisms interfere:
- **Test E** (`--rank1-encode` + forward chain, no refresh): the forward chain forms cleanly (**asym=+2.66**) BUT the chain
  erodes w_within to 6.4 → **NO reactivation**.
- **Config 1** (`--rank1-encode` + chain + rank1-style within-refresh 30): within-reactivation is STRONG (events=9,
  asm_active=[5,5], w_within=127) BUT the refresh's persistent-eligibility cross-links **overwhelm the forward chain**
  (asym flips to **−10.15 reverse**; replay direction near chance FWD 0.60 / REV 0.40).

⇒ the strong within-encode (persistent eligibility) that reactivation needs ADDS symmetric/reverse cross-links that swamp
the forward asymmetry. This was the genuine, well-scoped residual — a bounded tuning problem, per THE LAW (a method
verdict, not a capability wall), and lever (a) resolved it:

## ✅ RESOLVED — the full forward sequence replay is a SINGLE-SEED GO (all anti-cheats clean)
**Recipe: `--rank1-encode --within-events 30 --chain-fwd 24 --chain-rev 0 --within-refresh 8`** (a SMALL refresh restores
the within-attractor past the reactivation threshold with MINIMAL cross-linking, so the chain's forward asymmetry
survives). Seed 42, n_mem=2:
- **GO:** events=7, asm_active=[4,4] (strong within-reactivation) + **FWD=1.000 / REV=0.000, tau=+1.000** (perfect forward
  replay), encode **asym=+5.26** (w_fwd 29.46 > w_rev 24.20), w_within=143.
- **NO-NOISE acid = 0** (no self-sustaining artifact — the exact confound that RETRACTED the earlier RANK 1 attempt).
- **NO-ENCODE = 0** (learned weights load-bearing).
- **SCRAMBLE-BETWEEN → FWD=0.333** (shuffling the between-assembly edges BREAKS the forward direction: 1.000→0.333 ≈ the
  0.500 chance floor, reverse-biased) ⇒ the forward chain is the load-bearing structure, not an artifact.
- Runner VERDICT: **GO 1/1** — "a stored chain REPLAYS IN FORWARD ORDER under weak non-specific background: forward_frac
  1.000 vs reverse 0.000 vs SCRAMBLE 0.333 vs chance 0.500."
- Refresh=30 (too strong) flipped asym to −10.15 (reverse); refresh=8 is the sweet spot — restore, don't overwrite.

## Status
- **SOLVED:** the RANK 2 within-reactivation blocker (headline; corrects two prior over-framings on this exact question).
- **SINGLE-SEED GO:** the full forward sequence replay (within-reactivation + forward chain co-exist) — all anti-cheats
  clean. **Next: 6-seed (42/43/44/100/101/102) + n_mem=3 (A→B→C) + adversarial-verify → then RANK 2 CLOSES.**
- gap#5's solid rungs stand independent of this: the completion mechanism CLOSED (2026-07-18) + RANK 1 spontaneous
  reactivation 6-seed GO. RANK 2 (imagination-line rung 2) is a bonus; this advances it from "elusive blocker" to
  "within-reactivation solved, sequencing co-existence is the last tuning step."
- Infra added (additive, default-off, byte-identical when off): `--rank1-encode`, `--within-refresh N`, `--overlap-draw`
  diagnostic flags on `_gap5_sequence_replay_derisk.py`. Raw: `research/findings/raw/gap5_r4/rank2_nmem2_*.log`.
