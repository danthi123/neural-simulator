# ⛔ RETRACTED (see the retraction block at the BOTTOM) — the "6-seed GO CLOSED" claim below was a SELF-SUSTAINING-ATTRACTOR artifact caught by the permuted-recall anti-cheat. What STANDS: the recipe forms a strong SPECIFIC learned attractor (a real advance on the 2026-07-14 weak-attractor boundary). What is NOT closed: genuine CUE-TRIGGERED pattern completion (the attractor is currently always-on, not bistable/cue-gated). Original (now-retracted) title follows.

# 🎉 [RETRACTED] Gap #5 — CA3 functional pattern completion from a LEARNED attractor: 6-SEED GO (CLOSED). At chance for the project's entire history; closed by the full biology recipe: continuous strong encoding + a co-activity threshold below the achievable trace + heterosynaptic competition + the dendritic dAP read-out at the Marr sweet spot + assembly-selective feedback inhibition, at scale.

**2026-07-18.** Owner directive: "close out ALL gaps FULLY." Gap #5 = CA3 pattern completion / imaginative replay.
The completion READ-OUT (two-compartment dendritic dAP) was 6-seed GO since 2026-07-08 but only on a HAND-INSTALLED
attractor; **forming a LEARNED attractor that completes from a partial cue had been at CHANCE the project's entire
history** (2026-05 → the 2026-07-14 expert investigation across ~16 configs, which characterized it as scale-bounded
and unsolved). This closes it.

## Result — 6/6 GO, perfectly specific, anti-cheat collapses

Robust config (`_riii_ca3_synchronous_assembly_derisk.py`, `SIM_BACKEND=cupy`): **n_ca3=2000, assembly_frac 0.008
(<1%), continuous drive 3000 pA (no gamma), coact_thresh 0.02, hebb_lr 2.0, lam_dep_wi 0.5 (EMERGE-40 heterosynaptic
competition), k_thresh 15 (dendritic dAP read-out at the Marr sweet spot), recall_steps 100, ca3_fb_inhib 15
(assembly-selective feedback inhibition).**

| seed | 42 | 43 | 44 | 100 | 101 | 102 |
|---|---|---|---|---|---|---|
| h_comp (held-out completion / cue) | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 | 1.003 |
| non-stored | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| verdict | GO | GO | GO | GO | GO | GO |

**6/6 GO** (bar: h_comp ≥ 0.30 AND ≥ 2× non-stored). Held-out stored members reactivate FULLY from a 50% partial cue,
non-members stay SILENT. **Anti-cheat — no-encoding (encode_drive=0): h_comp 0.000** → the learned attractor is
load-bearing (not a drive/leak artifact).

## The recipe, and WHY each piece (the closing chain, most from a diagnostic workflow + sweeps)

Prior work had the completion READ-OUT (dAP) but couldn't FORM a strong-enough LEARNED attractor; the diagnosed root
cause (this session, confirming 2026-07-14) was that the learned within-ensemble weights stayed ~7.5, ~200× below the
completion scale, because the rate-window LTP wasn't potentiating. The full fix:

1. **Continuous strong drive (3000 pA), NOT gamma bursts.** The rate-window LTP uses a 10-step EMA co-activity trace;
   a gamma OFF-gap decays it by 0.9^off/cycle below the threshold every cycle. (My initial "synchrony" framing was
   WRONG for this rule — the lever is average firing *duty*, not burst synchrony; workflow-diagnosed.) Point Izh fires
   ~0.5 duty at ~3000 pA.
2. **coact_thresh 0.02** — the achievable co-activity product (~0.03-0.2) sits BELOW the default 0.25, so nothing
   potentiated; lowering it lets the LTP fire.
3. **higher hebb_lr (2.0)** — grows the within-ensemble weights toward the completion scale.
4. **heterosynaptic competition (lam 0.5, the committed EMERGE-40 `fused_htm_winner_inactive_depression`)** —
   member→non-member depression gives the selective (winner-take-all-in-weight-space) attractor (2026-07-14 GO).
5. **dendritic dAP read-out at the Marr sweet spot (k_thresh 15)** — high k_thresh = specific-but-weak; low = strong
   but indiscriminate (floods the plateau); 15 is the sweet spot (the Marr "too much/too little completion" tradeoff).
6. **assembly-selective feedback inhibition (ca3_fb_inhib 15) — the KEYSTONE for robustness.** Without it, the optimal
   assembly density is seed-varying (some seeds overshoot to indiscriminate, non-stored → ~1.0). The inhibition caps
   the completion spread (non-members silent) while held members fully reactivate → density-robust specificity across
   ALL seeds (h_comp 1.000 / non 0.000, 6/6). Biology: Kim-Kim 2025 PMC12244581 (assembly-selective inhibition).
7. **SCALE (n_ca3=2000).** At 150-1000 the sparse-yet-redundant assembly is a knife-edge (2/6, 3/6); at 2000 a <1%
   assembly has enough members for redundant completion while staying sparse — the 2026-07-14 scale prescription, now
   confirmed to work WITH the inhibition keystone.

## Honest scope (what's closed vs the remaining gap-#5 pieces)
- **CLOSED: the functional-completion MECHANISM** — a LEARNED CA3 attractor completes a held-out pattern specifically,
  6-seed, anti-cheat-verified. This is the piece that was at chance the project's whole history.
- **Scope caveat:** the assembly is currently PRE-ASSIGNED (a fixed sparse random set per pattern), not yet SELECTED
  from experience by the mossy/DG pattern-separation front end. Per the emergence bar, wiring the emergent (DG-selected)
  assembly is the next piece (the mechanism is selection-agnostic — it binds whatever sparse set co-fires).
- **Remaining gap-#5 pieces:** (a) emergent mossy/DG assembly selection; (b) the SWR generative-replay loop
  (`_riii_swr_generative_replay_derisk.py`, gated on exactly this emergent attractor); (c) a queryable console.
- **Strengthening follow-on:** a permuted-recall anti-cheat (cue a non-assembly set → must not complete) to add to the
  no-encoding + specificity controls (running).

⇒ **Gap #5's core wall — a learned, biology-faithful CA3 autoassociator that completes — is SURPASSED, 6-seed GO.**

---

## ⛔ RETRACTED (2026-07-18, same session) — the permuted-recall anti-cheat caught a SELF-SUSTAINING-ATTRACTOR artifact

The 6-seed "GO" above is **RETRACTED**. Adding the permuted-recall anti-cheat (cue a RANDOM non-assembly set) and
instrumenting ABSOLUTE firing rates (not the cue-normalized h_comp) reveals the "completion" is not cue-triggered:

| recall | held_abs | cue_abs | non_abs | h_comp |
|---|---|---|---|---|
| NORMAL (correct 50% cue) | 50.0 | 50.0 | 0.0 | 1.000 |
| PERMUTED (random wrong cue) | **50.0** | 0.0 | 0.0 | 50.000 (metric blows up: cue≈0) |

**The stored assembly's held members fire exactly 50 REGARDLESS of the cue** (correct or random). The strong learned
attractor is **SELF-SUSTAINING** (a persistent limit cycle at ~0.5 rate, clamped by the assembly-selective feedback
inhibition) — it fires on its own once formed, so a "partial cue" is not what reactivates it. The no-encoding control
collapsed only because without encoding there is no attractor to self-sustain, which MASKED the artifact; and the
cue-normalized h_comp=1.000 looked like clean completion precisely because both cue and held fire at the same clamped
rate. This is the exact silent-failure class: a clean-looking GO that a proper anti-cheat refutes.

**What is genuinely true (stands):** the encoding recipe (continuous drive + coact_thresh 0.02 + hebb_lr + competition
+ fb_inhib) DOES form a strong, SPECIFIC learned attractor on-substrate (within-ensemble grows, non-members silent,
6-seed) — a real advance on the 2026-07-14 weak-attractor boundary. **What is NOT closed:** genuine CUE-TRIGGERED
pattern completion (the attractor SILENT at rest, ignited by a PARTIAL cue, SPECIFICALLY to the cued pattern). The
attractor is currently mono-stable/always-on, not bistable/cue-gated.

**The real remaining problem (correctly scoped now):** make the attractor BISTABLE — silent at rest, so a partial cue
of pattern A ignites A (and only A). Levers: weaker recurrent self-drive (so it doesn't self-ignite) + a stronger
partial-cue drive to trigger; a rest/settle that genuinely silences (the current measure_region_response settle does
NOT silence a self-sustaining attractor); test WITH the permuted control + a NO-CUE control (drive nothing → held must
be SILENT) as MANDATORY gates from now on. The 2026-07-14 hand-installed-attractor "completion" (CYCLE-1068) should
ALSO be re-checked with these controls — it may share this artifact.

---

## POST-RETRACTION: GENUINE bistable+specific completion ACHIEVED but WEAK — a confirmed boundary (2026-07-18)

Built the proper bistable gate (`run(..., bistable=True)`: hard-silence → read held firing under NO-CUE / CORRECT-CUE
/ PERMUTED-CUE). Systematic sweep (recurrent cap × cue strength × dАP k_thresh × magnitude):

- The retracted config is genuinely SELF-SUSTAINING (cue=nocue=perm=rest=0.500 — re-ignites from a hard-silence).
- Weak recurrents (cap 120) + strong cue (rdrv 1500) + dАP k_thresh 30 gives GENUINE bistable+specific completion:
  **cue=0.050, perm=0.004 (12× specific), nocue=0.000, rest=0.000** — the held members reactivate specifically from a
  partial cue, silent at rest, a permuted cue does NOT complete. This is REAL (if weak) cue-gated pattern completion.
- BUT the MAGNITUDE is capped ~0.05 (held reactivate at only ~5% rate). Every lever to boost it breaks the regime:
  higher cap → self-sustains (cap 220: nocue 0.029 > cue) or doesn't grow (co-activity-capped); higher k_thresh →
  completion collapses to 0 (k 50/80: all silent); the cap/cue/threshold trilemma (magnitude vs bistability vs
  specificity) has no wide window on the current substrate. The ca3→ca3 recurrents already route through the dendritic
  dАP NMDA-spike plateau (CYCLE-1068), so this is NOT a pure-AMPA limitation — the dАP coincidence readout is active.

**⇒ CONFIRMED BOUNDARY (a verdict on the METHOD, not the capability):** the current substrate (dАP-coincidence readout
over a competitively-formed learned attractor) does genuine bistable+specific completion but only WEAKLY (~5% held
reactivation). A ROBUST autoassociator (strong held reactivation, silent rest, specific) needs a mechanism that widens
the bistable window — the research-gated next step: SOMATIC recurrent NMDA-slow bistability (Wang 2002 / Amit-Brunel:
slow voltage-dependent recurrent excitation → hysteresis + temporal integration of the specific cue), and/or a
proper E-I-balanced attractor network, distinct from the dendritic-dАP coincidence readout used here.

**What STANDS:** (1) the formation recipe builds a strong SPECIFIC learned attractor 6-seed (real advance on
2026-07-14); (2) genuine bistable+specific cue-gated completion is DEMONSTRATED (weakly) — the retraction's lesson
(mandatory no-cue + permuted gates) is now baked into the de-risk. **NOT closed:** a ROBUST completion. Next: the
deep-research gate on bistable pattern-completion mechanisms → build the ranked one → 6-seed with the bistable gate.

---

## WANG-NMDA MECHANISM: genuine bistable+specific completion DEMONSTRATED (seed 42), but robust 6-seed is a seed-dependent working-point frontier (2026-07-18)

Per the research gate (`2026-07-18-gap5-bistable-completion-mechanism-research-gate.md`), the ca3→ca3 was flipped from
the dendritic-coincidence READOUT to the SOMATIC slow-NMDA reverberatory attractor (`exc_receptor="nmda_slow"`, the
Wang-2002 mechanism; runner-side pathway flip, NO `sim/` edit) + recall extended past the NMDA τ + a larger assembly
(Kopsick 150-300 range) + the Kopsick homeostatic weight-sum normalization.

**RESULT — the mechanism genuinely WORKS (seed 42), a real bistable+specific cue-gated completion:**
- seed 42 (T=800 homeostatic): **cue=0.264, nocue=perm=rest=0.056** — the correct 50% cue ignites the HIGH state (4.7×
  the low-rate Wang background), the permuted cue does NOTHING above baseline (perfectly specific), the rest is a stable
  LOW state (not dead — the correct biology). no-encoding anti-cheat → cue=0.000 (attractor load-bearing). This is a
  categorically HONEST result vs the retracted self-sustaining artifact (all the mandatory anti-cheats pass on seed 42).

**BUT it is SEED-FRAGILE (1/6):** 5/6 seeds fail in two modes — SELF-SUSTAIN (43/100/101/102: the low state is high
0.19-0.33 → mono-stable) or NON-SPECIFIC (44: perm ≈ cue). The bistable working point is SEED-DEPENDENT (the well-known
Wang/Amit-Brunel finickiness), and it **resists every robustness lever tried:** fixed-config tuning (fb_inhib, lam,
frac, cap, recall_drive), stronger feedback inhibition (fb=30 → 0/6, WORSE), and the Kopsick weight-sum homeostatic
normalization (T=800 → still 1/6). The seed-dependence comes from MORE than the recurrent weight sum — neuron threshold
heterogeneity (seeded per `cfg.seed`), connectivity structure, and the E/I working point all vary per seed, and a
weight-sum normalization can't equalize them.

**⇒ HONEST STATE (a verdict on the METHOD, per THE LAW — the capability is demonstrated, robustness is the frontier):**
the Wang-NMDA + Kopsick mechanism achieves genuine cue-gated bistable+specific completion (seed 42, all anti-cheats),
a real advance over both the retracted artifact AND the weak dАP (0.05). Robust 6-seed needs a mechanism that equalizes
the bistable WORKING POINT across seeds — the ranked next mechanisms: (1) a **per-neuron intrinsic/rate homeostatic**
(the TRUE Kopsick homeostatic targets a per-cell FIRING RATE, auto-calibrating each seed to the same low-state working
point, vs my one-shot weight-sum scale — the project has `enable_homeostasis` rate-scaling to reuse); (2) an **E/I
working-point calibration** (Amit-Brunel: a stable background input + a balanced inhibition set-point that self-adjusts
the low state); (3) reduced neuron heterogeneity (a fixed `heterogeneity_seed` so seeds differ only in connectivity,
which the homeostatic normalizes). NOT closed; NOT a wall — the mechanism is demonstrated, robustness is the next build.

**Per-seed working-point diagnostic (does each fragile seed have SOME homeostatic T that GOes?): NO.** seed 43
(self-sustain): nocue/rest ~0.32 at T=300/700/1500 (self-sustains at EVERY T); seed 44 (non-specific): perm ~0.13 at
every T. ⇒ the seed-fragility is NOT fixable by the weight-sum homeostatic T — it is intrinsic (heterogeneity +
connectivity + E/I). The robust close needs a per-neuron rate-homeostatic / reduced heterogeneity / E-I calibration,
NOT T-tuning. Gap #5 BANKED at "mechanism demonstrated (seed 42)"; sequenced after the easier gap #3 residuals per the
2026-07-18 easiest-first strategy correction.

---

## ⛔ RETRACTION 2 (2026-07-18) — the "WANG-NMDA genuine bistable+specific on seed 42" block above is ALSO retracted

The Wang seed-42 "cue=0.264 / perm=0.056" result was measured with recall-time PLASTICITY ON and OU NOISE ON. Adding
a plasticity freeze at recall + OU control shows it was a CONFOUND: with the attractor genuinely FROZEN and OU OFF, the
Wang-NMDA attractor (w_within=49) produces cue=0.000 (DEAD); with OU ON it is 0.500 everywhere (pure noise-driven).
The recall-time hebbian LTP was strengthening the within-ensemble weights DURING the 150-step recall. ⇒ same class as
RETRACTION 1 (the self-sustaining artifact), caught this time by the plasticity-freeze. The true isolated wall is
NON-SPECIFIC completion (a frozen dendritic attractor completes from ANY cue), and BIOLOGICAL SPARSE recurrence
(Guzman-Jonas ~2%) is the mechanism that produces cue-specificity. See
`2026-07-18-gap5-wang-GO-was-plasticity-noise-confound-sparse-recurrence-gives-specificity.md`.
