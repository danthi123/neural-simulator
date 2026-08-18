---
type: finding
status: negative
date: 2026-08-17
mechanism: wave1-banking
---
## emerge-stream-eprop-6 — wave-1 verdict: NEGATIVE (directional gain REFUTED by mandatory controls)

**Set.** EMERGENCE stream/reservoir language-cortex; recurrent weights W_rec trained by transport-free random-feedback e-prop (RFLO / Bellec 2020, rate analogue of the on-bridge BDSP). Runner `_emerge_reservoir_lm_eprop_recurrent_derisk`, WikiText, 6 seeds 42/43/44/100/101/102 (`_eprop/wiki_np300_s*`). Forward state identical across arms; only whether/how W_rec learns varies.

**Wave-1 result (naive).** Plastic e-prop lowers held-out next-token CE vs the SAME-SIZE fixed reservoir by mean **−0.036 nats, 6/6** (range −0.026..−0.046). In-run anti-cheats pass: `shuffle_elig` (same update magnitude, scrambled eligibility) ≈ fixed (+0.0002, ~60x smaller); `zero_signal` == fixed exactly. Reads as credit-structure-load-bearing. <!--derived--> (aggregate CE deltas over seeds; per-run values in the cited artifacts)

**Why it is NEGATIVE.** Pre-registered follow-up controls (`_eproplm_controls/controls_s{42,43,44}`) refute it on three axes: (1) **sign_flip** reproduces the deep margin while the true-gradient/symmetric direction HURTS → the effect is credit-DIRECTION-INDEPENDENT (a memory-timescale / operating-point nudge, not credit assignment); (2) **distal-prefix scramble** — margin survives randomizing tokens 0..t-4 → LOCAL, not long-range; (3) **strongest n-gram** — plastic deep CE 3.180 loses to an interpolated trigram 2.804 (−0.376, 3/3), and the fixed substrate (3.70) is below an add-1 bigram (3.27). The two clean anti-cheats missed exactly this operating-point confound. <!--derived--> (CE/n-gram baselines aggregated over the control seeds)

**Honest residual.** De-risks a mechanism DIRECTION only — a local transport-free rule does move recurrent weights and the plumbing runs; it is NOT a performance win, NOT genuine long-range deep credit. The 0.036-nat falling loss is the canonical "falling loss ≠ GO" trap. No `.prov.json` sidecars (runs predate provenance, Jul 11–14). Sibling files (`onbridge_*` BDSP, `hetero/horizon/alif3` variants, `_snnbptt/eprop_*` classification probe) are distinct experiments, not this set. The separate deep-credit-on-spikes CLASSIFICATION core is untouched by this refutation. <!--derived-->

Banked artifacts (this branch): plastic-vs-fixed reservoir runs `research/findings/raw/_eprop/wiki_np300_s*.json`; the decisive follow-up controls `research/findings/raw/_eproplm_controls/controls_s*.json`. (No .prov.json sidecars — runs predate provenance.)
