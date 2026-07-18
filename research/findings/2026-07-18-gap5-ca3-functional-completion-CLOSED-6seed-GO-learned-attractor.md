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
