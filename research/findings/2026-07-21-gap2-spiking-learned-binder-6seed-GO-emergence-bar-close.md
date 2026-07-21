# gap#2 — a LEARNED local binder READ on the SPIKING RF substrate reaches the 1.000 ceiling (6-seed GO): the emergence-bar residual is closed

**2026-07-21 · GO, 6-seed (42/43/44/100/101/102).** The gap-close research gate's Rank-1: replace the composer's FIXED
FHRR exact-inverse bind (the "principled idealization" the emergence bar rejects) with a J WRITTEN by a LOCAL
outer-product rule (no backprop, no transport) and READ via the committed RF resonate loop. Result: the LEARNED binder
matches the fixed-FHRR ceiling on the spiking substrate.

## Result (`_gap2_spiking_deltarule_binder_derisk.py`, 6-seed)

A per-fact fast-weight J (D×D complex), written by a LOCAL rule over (role-key, filler-value) phasor pairs — delta
`J += (v−Jk)kᴴ/D` or plain Hebbian `J += v kᴴ/D` — is installed as the FULL RF coupling (`out[k] ← in[j] : J[k,j]`);
the role key is kicked on the input block, the RF resonate loop runs (`rf_resonate_steps`), and the output-block phases
are read + cleaned up (nearest concept). Over the 788 correlated stream-cortex phasor codes:

| P (roles/fact) | DELTA | additive | permuted-role (anti-cheat) | decorrelated-ctrl |
|---|---|---|---|---|
| 1 | 1.000 | 1.000 | 0.000 | 1.000 |
| 2 | 1.000 | 1.000 | 0.000 | 1.000 |
| 3 | 1.000 | 1.000 | 0.000 | 1.000 |
| 4 | 1.000 | 1.000 | 0.000 | 1.000 |
| 5 | 1.000 | 1.000 | 0.000 | 1.000 |

All 6 seeds identical. **The LEARNED binder READ on the spiking RF loop = the fixed-FHRR 1.000 ceiling, P=1..5.**

## The silent-failure catch (rule 3: verify the instrument; the anti-cheat FLAGGED the bug)

The first pass FAILED its anti-cheat (permuted-role 1.000 at P=2, delta 0.000 while additive 1.000 — invalid,
confounded). Rather than report it, I debugged: the RF read is CORRECT (returns J@key at 0.0074 circular-phase error
for both diagonal AND full J). The bug was the DELTA RULE — it assumed unit-norm keys, but the keys are D=128-dim UNIT
PHASORS with `⟨k,k⟩=D=128`, so `J += (v−Jk)kᴴ` OVERSHOT by 128×. Fix = `/⟨k,k⟩=/D`. After the fix, permuted-role →
0.000 (role-addressed) and delta reaches the ceiling. **The anti-cheat failure was the instrument telling me the write
was wrong — exactly its job.**

## Read-out — the emergence-bar close + the honest scope

- **⇒ the emergence-bar residual is closed for the spiking READ:** a LEARNED local binder (a local Hebbian
  outer-product J, no backprop/transport) on the committed RF resonate loop retrieves multi-role facts at 1.000,
  matching the fixed-FHRR ceiling, 6-seed, anti-cheat clean. The composer's fixed exact-inverse algebra is NOT
  load-bearing — a learned local binder replaces it with no loss on spikes.
- **Honest scope (three, not buried):**
  1. **delta-vs-additive is NOT load-bearing at this scale** — both 1.000, because D=128 role phasors are
     near-orthogonal (`|⟨kᵢ,kⱼ⟩| ≈ √128 ≈ 11` vs signal 128, ~9% crosstalk that doesn't cause errors up to P=5). So a
     plain Hebbian outer-product suffices (even simpler, still a learned local rule). The record's additive-STP SHARED
     store collapse (P1 0.92→P2 0.11), where delta WOULD be load-bearing, is a MORE-COMPRESSED multi-FACT
     representation (the composer's separate-block store already works — a compression follow-on, not a capability gap).
  2. **The WRITE is a host-computed local outer-product** (each `J[k,j] = post_k · pre_j` is a local Hebbian
     coincidence — emergence-bar-compliant), installed as the RF coupling. The FULLY-on-bridge STP/BTSP write is the
     edge5 arc (refuted for the shared store; the composer's separate-block store is the deployed path). The READ is
     genuinely on the spiking substrate.
  3. Per-fact multi-ROLE bind (P roles in one fact), not the multi-FACT shared store.
- **This unifies with gaps #3/#5** per the gate: the same role-keyed spiking read is the multi-referent
  disambiguation read (phase-cluster/biased-competition) and the energy-descent completion — the Rank-4 follow-on.

## Also this cycle — gap#1 scale lever confirmed (measurable)

The WKV cortex scale sweep (`_emerge_wkv_lm_derisk`, d256/d512 × 100k/200k tokens): deep (10-99 token) WKV NLL drops
**3.276 → 3.191 → 3.168** (ppl 26.5 → 24.3 → 23.8) as DATA (100k→200k) then MODEL (d256→d512) grow — every config beats
a fair trigram at depth with memoryless-collapse +1.2-1.3 (genuinely uses long-range state). ⇒ gap#1 (open generation)
is MEASURABLY scale-progressing; the lever (more data + bigger model → lower ppl) works.

Runner: `_gap2_spiking_deltarule_binder_derisk.py` (`--seeds`, `--pmax`, `--n-facts`); ceiling
`_gap2_binder_resonator_ceiling.py`.
