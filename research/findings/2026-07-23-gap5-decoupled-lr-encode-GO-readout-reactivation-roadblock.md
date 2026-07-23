# gap#5 forward-asymmetry: the DECOUPLED-lr encode GIVES a reactivatable + forward-biased store (weight GO); the spiking READOUT hits a discrete-reactivation roadblock (2026-07-23)

## Encode breakthrough — DECOUPLE the within-lr from the chain-lr (seed 42, encode-only, NO sim/ edit)
The single-btsp_lr sweet-spot sweep revealed a CLIFF: one btsp_lr can't give both a reactivatable within AND a
forward bias (lr0.05 -> within200/ratio1.67x; lr0.5 -> within40/ratio7.65x). But the within-attractor (within phases +
refresh) and the forward chain are SEPARATE encode phases sharing `btsp_lr`. Decoupling them (a new `chain_btsp_lr`
applied only during the chain phase; `btsp_learning_rate` is read live so it's set for the chain, restored for the
refresh) resolves the cliff:
```
within-lr 0.05 + chain-lr 0.3 + freeze:  within=205.5  adj_fwd=25.0 adj_rev=5.0  ratio=4.99x
within-lr 0.05 + chain-lr 0.5 + freeze:  within=206.1  adj_fwd=38.3 adj_rev=5.0  ratio=7.65x
within-lr 0.05 + chain-lr 1.0 + freeze:  within=199.3  adj_fwd=71.5 adj_rev=5.0  ratio=14.30x
```
BOTH within~200 (the symmetric store reactivated at 173, so this is reactivation-scale) AND a strong forward bias
(adj_rev pinned at init 5). ⇒ the forward-asymmetric weight store is now built at a reactivatable within magnitude.
Additive default-off edits (`chain_btsp_lr=None` -> byte-identical); NO sim/ edit.

## Readout ROADBLOCK — the decoupled store does NOT produce discrete reactivation events (4 attempts, ev=0)
The intrinsic-fatigue spiking readout on the decoupled store (within=206, ratio 7.65x), sweeping self_regen_read
{0.0, 0.08, 0.12}: INTRINSIC fwd=0.000, **ev=0, multi=0, act=[0,0,0]** every time (pop rose 0.055->0.104 vs the
within=40 store, so it IS more active, but NO discrete assembly bursts). Prior attempts also ev=0: within=40 store
(any sr), within=206 store (any sr).

**The load-bearing diagnosis (the real tension):** the symmetric store's earlier "reactivation" (ev=4, act=[3,3,3])
was DIFFUSE CO-FIRING — all 3 assemblies co-igniting, riding the STRONG symmetric between-links (~142) — which is
NOT clean forward replay (it's the co-ignition the whole arc fights). The decoupled store has WEAK between-links
(adj_fwd 38 / adj_rev 5), so it does NOT co-fire — but a single within=206 assembly then does not spontaneously
reactivate from Poisson noise EITHER. ⇒ on this substrate, spontaneous DISCRETE single-assembly reactivation is the
open problem: strong between-links -> diffuse co-fire; weak -> no reactivation. The forward-asymmetric weights are
necessary + now built, but a clean discrete reactivation+transition readout is the remaining piece.

## Next (the specified continuation — NOT blind sweeping)
1. **Test the decoupled store under the RANK-1 BISTABLE reactivation readout** (`_gap5_spontaneous_reactivation_derisk`,
   the one that reactivated the symmetric store), not the intrinsic-fatigue de-latch — to separate "the store can't
   reactivate" from "the intrinsic-fatigue de-latch (self_regen_read=0) prevents ignition." If it reactivates
   discretely under bistable, the readout is the fix (bistable-ignite -> then de-latch to transition).
2. If not, the discrete single-assembly reactivation on weak-between-link stores is a research-gate roadblock
   (candidate: a stronger/targeted ignition, a DG-detonator ignition per Kandel Ch 54, or a sharper within-attractor
   with feedback inhibition so a single assembly bursts discretely rather than smears).

## Status
Encode: forward-asymmetric + reactivation-scale within = GO (weight level). Readout: discrete forward replay on the
new store = OPEN (banked, precisely scoped). The static CA3 completion this rides on is CLOSED (2026-07-18); the
imaginative-replay capability stays OPEN per THE LAW. NO sim/ edit anywhere in this arc.

## UPDATE — encode is 6/6 GO (multi-seed confirmed, 2026-07-23)
The decoupled encode (within-lr 0.05 + chain-lr 0.5 + freeze) is now validated across all 6 seeds
(42,43,44,100,101,102): every seed lands at within ~196-207 (reactivatable) + adj_fwd 38 / adj_rev 5 / ratio
7.65-7.66x, remarkably consistent. Seeds 43/44/100/101/102 were run on the mini-PC pool, seed 42 + 100 locally
(byte-identical, numpy 2.4.6 pinned). ⇒ the FORWARD-ASYMMETRIC + REACTIVATION-SCALE encode is a robust 6-seed GO;
the spiking-readout discrete-reactivation on that store remains the scoped open piece.

## UPDATE — Next-step-1 ANSWERED: the RANK-1 BISTABLE readout is an HONEST NEGATIVE on the decoupled store (6-seed, 2026-07-23)
`_gap5_decoupled_store_bistable_readout_derisk.py --seeds 42 43 44 100 101 102` (numpy, fanned local):
- **decoupled store ignites discretely: 1/6** (only seed 43, an outlier).
- **readout works on the SYMMETRIC positive control: 4/6** (seeds 100/101/102 confirm the readout CAN ignite a
  strong-between store; seed 43 ignited the decoupled one; seeds 42/44 the readout didn't ignite even the symmetric
  control = INCONCLUSIVE, seed-fragile ignition).
⇒ On the 4 seeds where the readout demonstrably works (ignites the symmetric store), the DECOUPLED forward-asymmetric
store does NOT spontaneously ignite (0/4). This is exactly branch (B): **ev≈0 is a STORE property — the decoupled
store's WEAK between-links (adj_fwd 38 / adj_rev 5) cannot spontaneously ignite a single within~206 assembly from
Poisson noise.** The store is correct (forward-asymmetric, reactivation-scale) but SPONTANEOUS ignition is the wrong
readout for weak-between-link stores.

**⇒ Method verdict (per THE LAW — a verdict on the METHOD, not the capability):** spontaneous bistable ignition of the
decoupled store = NO. The next method is **TARGETED ignition — the DG-DETONATOR** (Kandel Ch 54 detonator synapse:
drive a single assembly's cells directly at rest, then let the forward-asymmetric between-links carry the transition),
built preemptively (`_gap5_dg_detonator_ignition_derisk.py`) and running 6-seed on the mini-PC pool. That branch-B
result decides whether targeted ignition + forward transition closes the imaginative-replay readout. NO `sim/` edit.
The imaginative-replay CAPABILITY stays OPEN per THE LAW; the failing METHOD (spontaneous ignition) is banked.
