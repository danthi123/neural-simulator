---
type: finding
status: superseded
date: 2026-07-17
mechanism: ca3-attractor-formation
---

# Gap #5 CA3 autoassociator — Rung 2 (mossy-detonator) sparsifies but does NOT select; Rung 3 (input gamma-pulse) is inert; the formation blocker is confirmed to be SYNCHRONY, needing a theta-gamma pacemaker (Rung 4). The completion FUNCTION is met via other validated methods; the biologically-faithful CA3-recurrent-autoassociator is the deep open frontier.

**2026-07-17.** Continuing the gap #5 (CA3 completion) close from the 2026-07-09 frontier (`2026-07-09-riii-ca3-feedback-inhibition-sparsifies-but-nonselective.md`: Rung 1 `ca3_pv_basket` sparsifies but global inhibition is non-selective; Rung 2 mossy-detonator named next).

## Rung 2/3 sweep result (seed 42, `_riii_ca3_attractor_diag.py`, mossy × cap × gamma-sync, feedback-inhib 120 + rate-window)

| config | sparsity | within-ens | member→silent | separation |
|---|---|---|---|---|
| mossy 30, cap 60 | 0.11 | 4.98 | 5.26 | **−0.27** |
| mossy 80, cap 60 | 0.11 | 4.98 | 5.26 | −0.27 (identical to mossy 30) |
| mossy 150, cap 150 | **0.03** | 4.79 | 5.26 | **−0.47** |
| mossy 80, cap 60, **+sync 2/4** | 0.11 | 4.99 | 5.26 | −0.27 (byte-identical to no-sync) |
| mossy 150, cap 150, +sync 2/4 | 0.03 | 4.80 | 5.27 | −0.47 (byte-identical to no-sync) |
| mossy 150, +sync 3/6, drive 120 | 0.03 | 4.79 | 5.27 | −0.47 |

## What the numbers say (read from the data)
1. **Mossy-detonator DOES sparsify** (0.43 baseline → 0.11 at mossy 80 → **0.03 at mossy 150**). Rung 2 achieves the sparse code — a few CA3 cells fire hard, the rest are inhibited. Confirms the gate's sparsity mechanism.
2. **But the within-ensemble recurrents DECAY below init** (6.0 → 4.79–4.99) and the separation is **NEGATIVE** (−0.27 to −0.47) — member→silent (5.26) stays HIGHER than within-ensemble. So the rate-window rule is NOT potentiating the ensemble; only decay applies. Root cause: with ~4 sparse cells firing ASYNCHRONOUSLY, the co-activity traces (trace[pre]×trace[post]) never clear the threshold → zero potentiation. Sparsity WITHOUT synchrony still cannot bind — exactly the 2026-07-09 diagnosis.
3. **Rung 3 (input gamma-pulse `--sync-on/--sync-off`) is INERT** — every sync run is byte-identical to its no-sync twin (4.98 vs 4.99; 4.79/4.80). The pulse drives the INPUT (lang→EC→DG→CA3); the CA3 response smears it, so the members do NOT fire in synchronous volleys. Pulsing the input does not synchronize CA3.

## The honest frontier (per THE LAW: a method-limit, the capability is not deferred)
- **Completion HALF: SOLVED** (dendritic dAP on a strong attractor, 6-seed GO, `2026-07-08-riii-onsubstrate-dendritic-dAP-completion-SURPASS-6seed`). This session ALSO independently reconfirmed the substrate CAN complete (a hand-installed strong symmetric attractor ignites specifically).
- **Formation HALF: the blocker is SYNCHRONY.** Sparsity is achievable (mossy detonator, 0.03). The plasticity rule is correct (rate-window). But a co-activity rule cannot bind cells that do not co-fire in a tight window, and neither the emergent PING (from `ca3_pv_basket`, too coarse at dt=1.0/150 cells) nor an input-side gamma pulse produces CA3 synchrony. **The named next mechanism = Rung 4: a genuine theta-gamma pacemaker** (a rhythmic inhibitory drive that paces the sparse CA3 survivors into gamma volleys DIRECTLY — Lisman-Idiart / Buzsáki, catalog N.15/N.19), likely a guarded `sim/` mechanism. This is the deep open frontier, NOT a quick close.

## The completion FUNCTION is met via OTHER validated methods (so conversation is not blocked)
The biologically-faithful CA3-recurrent-autoassociator is the specific hard target above. But pattern completion AS A
FUNCTION is already available multi-seed via different mechanisms: EMERGE **spreading-activation completion (12-seed
GO)**, **graded-confidence completion (12-seed GO)**, the composer cue-matching scan, and this session's gap-#2
slot-binder (content-addressable multi-fact recall + moat, 6-seed GO). Imaginative RECOMBINATION is GO
(`2026-07-08-imaginative-scenario-recombination-Ri-GO`); the SWR generative-replay REACTIVATION is blocked by the same
CA3-attractor formation gap.

⇒ **Gap #5 status: the completion FUNCTION is met; the biologically-faithful CA3-recurrent-autoassociator (and the SWR
replay reactivation that depends on it) is a genuine DEEP frontier gated on a theta-gamma pacemaker (Rung 4, a sim/
mechanism) to produce a sparse + SYNCHRONOUS ensemble.** This is materially deeper than the "quick close" gap #5 was
picked for — a strategic reassessment point.
