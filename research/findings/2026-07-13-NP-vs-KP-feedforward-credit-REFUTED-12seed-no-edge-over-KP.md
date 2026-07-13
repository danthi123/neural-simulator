# NP-vs-KP feedforward W_in-credit: REFUTED (12-seed, 0/6 standard + 0/6 fresh) — node perturbation learns the credit (beats its own shuffle) but has NO EDGE over FA-partial Kolen-Pollack; the seed-42 GO was a seed artifact (the fresh-seed gate caught it AGAIN)

**Date:** 2026-07-13
**Runner:** `_np_feedforward_win_credit_derisk.py` (all-numpy on the R3 generalize instrument; oracle/fixed/fa/kp/np/shuffle/wrong arms). The design-gate (wf_82e0ddec-4a0) deliverable.
**Status:** ❌ REFUTED at 12 seeds. The fresh-seed pre-registration gate — the exact one that caught the recurrent-NP artifact — caught a SECOND seed artifact here.

## Result (headroom config sf=2/idn=30/id_pool=80/n=50; margins over chance)
- **seed 42 (the tuning seed): GO** — np +0.600 >> shuffle +0.017, np−kp +0.167 (NP beats KP). Looked decisive.
- **Full 12 seeds: 0/6 standard GO, 0/6 fresh GO.** Pooled **np−kp = −0.069 (standard), −0.100 (fresh)** — NP is on average SLIGHTLY WORSE than KP, not better. Per-seed np−kp: standard `[0.0,−0.28,−0.15,+0.02,0.0,0.0]`, fresh `[+0.10,−0.23,−0.30,−0.17,0.0,0.0]`. Only seed 42 (and marginally 100) had np>kp.
- **NP DOES ride genuine credit** (np−shuffle consistently positive, +0.07..+0.60; the anti-cheat holds) — so NP learns the W_in credit. It simply has NO EDGE over FA-partial KP. The design gate's hypothesis (NP's unbiasedness beats FA-partial KP on the ctx1 gap) is REFUTED on this instrument.

## ⇒ The honest, complete NP-arc bottom line (this reframes the whole session)
- **NP is a VALID, fresh-mechanism-class, robust FEEDFORWARD deep-credit rule** — it trains depth-2..6 to near-oracle on the emerge1 task (6/6 standard + 6/6 fresh, huge margins) where VANILLA feedback alignment fails to generalize, and it rides genuine credit everywhere (shuffle collapses). That headline STANDS.
- **BUT NP does NOT uniquely solve deep-credit-on-spikes, and does NOT beat the better biological rules.** Its three hoped-for unique advantages did NOT materialize: (1) recurrent W_in — RETRACTED (seed artifact); (2) beats-FA-partial-KP for the language input map — REFUTED here (0/12, NP≈KP); (3) works on-spike where the burst family's SNR failed — a small-net VARIANCE boundary (readout-noise-limited). Off-bridge, SEVERAL rules already work (Kolen-Pollack, burstprop, NP) and NP is not superior; the emerge1 edge is over VANILLA FA specifically, not over KP/burstprop (which also clear emerge1).
- **⇒ deep-credit-on-spikes, honest state:** OFF-bridge is solved by multiple biological rules (KP/burstprop/NP, none uniquely best); the mission-critical ON-SPIKE deep credit remains a genuine open BOUNDARY — a variance/SNR wall that bites EVERY rule on the small spiking substrate (burst-multiplexing SNR for the FA family; zeroth-order readout-noise variance for NP). The lever for the on-spike frontier is the noisy-spiking-readout problem (averaging/scale), NOT the choice of credit rule.
- **Methodology: the fresh-seed pre-registration gate is now 2-for-2** at catching seed-42-tuned artifacts (recurrent-NP + this). Adopt it as standard for any thin/tuned effect; single-seed "GO"s on a tuned config are not to be trusted.
