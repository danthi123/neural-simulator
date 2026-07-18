# Gap #5 — the Wang seed-42 "genuine bistable+specific" completion was a PLASTICITY+NOISE CONFOUND; the real wall is NON-SPECIFIC completion, and BIOLOGICAL SPARSE recurrence is the mechanism that produces specificity

**2026-07-18.** Continuing the owner directive "close out ALL gaps FULLY," easiest→hardest, gap #5 now active (after
gap #3 fully closed). The prior state banked gap #5 at "Wang-NMDA mechanism gives genuine bistable+specific cue-gated
completion on seed 42, but seed-fragile (1/6); the ranked fix is a per-neuron rate-homeostatic." Building that fix
surfaced a deeper confound that **retracts the seed-42 claim** and **re-isolates the true wall**.

## The instrument fix that exposed the confound: FREEZE plasticity + control OU noise at recall

A pattern-completion test reads a FIXED stored attractor: the recall must NOT learn, and (to isolate the deterministic
bistability) the OU background noise must be controllable. Both were implicitly ON in the prior bistable runs. Added
(default-off / byte-identical): a **plasticity freeze** in the bistable recall (`enable_hebbian_learning=False`), and
an `enable_ou` thread-through (default True == `CoreSimConfig` default). Then re-measured the Wang seed-42 config.

## RESULT 1 — the Wang seed-42 "GO" does NOT survive a frozen + noise-free recall (RETRACTION)

| condition | w_within | cue | nocue | perm | rest | reading |
|---|---|---|---|---|---|---|
| Wang, plasticity ON, OU ON (as prior-claimed) | 49 | 0.264* | — | ~0.056* | ~0.056* | the prior "GO" |
| Wang, **frozen recall, OU OFF** | 49 | **0.000** | 0.000 | 0.000 | 0.000 | **DEAD** |
| Wang, frozen recall, OU **ON** | 49 | 0.500 | 0.500 | 0.500 | 0.500 | pure noise-driven (0.5 everywhere) |

(*prior-claimed numbers.) With the attractor genuinely FIXED (no recall-time LTP) and no OU noise, the Wang-NMDA
attractor produces **zero completion** (w_within=49 is ~30× below the ~1600 completion scale). The prior seed-42 "cue
0.264 / perm 0.056" was a **confound**: recall-time hebbian LTP was strengthening the within-ensemble weights DURING
the 150-step recall (the co-firing members potentiate each other), and OU noise was seeding firing. **⇒ the
2026-07-18 "Wang-NMDA genuine bistable+specific on seed 42" claim is RETRACTED** — same class of self-deception as the
earlier "6-seed GO" self-sustaining-attractor retraction, caught this time by the plasticity-freeze + OU control.
Lesson banked: **a completion test MUST freeze plasticity and control noise, or the recall dynamics fake the result.**

## RESULT 2 — the DENDRITIC formation attractor IS a genuine frozen bistable attractor, but is NON-SPECIFIC

The dendritic-dAP formation config (the retracted-closure recipe: n_ca3=2000, encode 3000pA continuous, coact_thresh
0.02, hebb_lr 2.0, heterosynaptic competition, dAP k_thresh 15, assembly-selective fb-inhib), run through the SAME
frozen + OU-off gate, grows a much stronger frozen attractor (**w_within=450**) that genuinely completes and is
bistable — but is **NOT cue-specific**:

| recall_drive | cue | nocue | perm | rest | reading |
|---|---|---|---|---|---|
| 400 | 0.223 | 0.096 | 0.237 | 0.066 | bistable (low rest) but perm ≈ cue |
| 700 | 0.258 | 0.069 | 0.276 | 0.048 | perm > cue |
| 2000 | 0.238 | 0.088 | 0.252 | 0.057 | perm ≈ cue |

Robust across recall_drive (400–2000) AND dAP k_thresh (15/25/45): **nocue/rest are genuinely LOW (0.05–0.10 = a real
bistable low state, not self-sustaining), the correct cue DOES complete (0.18–0.26) — but a PERMUTED cue (a random
non-assembly set of the same size) completes the held members just as much.** ⇒ the frozen learned attractor is a
strong SPECIFIC attractor in weight-space (within 450 ≫ silent) yet a **functionally NON-SPECIFIC** one: **any**
sufficient input ignites the stored pattern. This is the same non-specificity the permuted-recall anti-cheat caught in
the retracted closure, now cleanly isolated on a frozen attractor (so it is NOT a plasticity artifact — it is the
attractor's actual recall geometry).

**Root cause (reasoned, then tested):** the CA3 recurrent connectivity was `density=0.5` (each of 2000 cells wired to
~1000 others). At 50% recurrence, driving ANY 16 cells floods the network, so the held members receive enough generic
recurrent drive to complete regardless of whether the cue OVERLAPS the assembly. Real CA3 recurrent connectivity is
**~2% (Guzman-Jonas 2016)**, not 50%.

## RESULT 3 — BIOLOGICAL SPARSE recurrence PRODUCES cue-specificity (the mechanism direction, seed 42)

Threading `ca3_density` through and dropping it to the biological range, with a larger assembly (enough within-
connections for the correct cue, but a random cue mostly misses the held members):

| density | assembly | w_within | cue | nocue | perm | rest | cue/perm |
|---|---|---|---|---|---|---|---|
| 0.50 | 16 (0.008) | 450 | 0.238 | 0.088 | 0.252 | 0.057 | **0.94** (non-specific) |
| **0.05** | **150 (0.075)** | 27 | 0.199 | 0.075 | **0.145** | 0.068 | **1.37** (specificity emerges) |

For the first time the correct cue completes MORE than a permuted cue (1.37× vs ≤1.0 at dense recurrence) — **sparse
biological recurrence is the mechanism that makes CA3 completion cue-SPECIFIC** (a random cue's cells mostly don't
connect to the held members). Still below the GO bar (cue ≥ 0.20, cue ≥ 3× perm, nocue ≤ 0.10): the sparse recurrence
also weakens the within-ensemble weight concentration (w 450→27), so the correct completion is marginal. A grid sweep
over (density × assembly_frac × fb_inhib) is running to find the working point that is both specific AND strong before
6-seed validation.

## Honest state (a verdict on the METHOD, per THE LAW — the capability is NOT closed)

- **RETRACTED:** "Wang-NMDA genuine bistable+specific completion on seed 42" — it was a recall-plasticity + OU-noise
  confound; the frozen noise-free Wang attractor (w 49) is dead.
- **ISOLATED WALL:** a frozen learned CA3 attractor completes from ANY partial cue (non-specific) when recurrence is
  dense — this is the genuine, plasticity-independent barrier to functional pattern completion.
- **MECHANISM FOUND (direction):** biological SPARSE recurrence (Guzman-Jonas ~2%) makes completion cue-specific
  (cue > perm for the first time). Needs a working point that is both specific (cue ≥ 3× perm) and strong (cue ≥ 0.20,
  nocue ≤ 0.10) — the running sweep. If found → 6-seed with the mandatory frozen + no-cue + permuted anti-cheats.
- **Infrastructure banked (default-off / byte-identical):** plasticity-freeze at recall, `enable_ou` control,
  `nmda_ratio` + `ca3_density` thread-through, and a per-neuron intrinsic-excitability rate-homeostatic (`rate_homeo`,
  Turrigiano) — the last is for taming an over-strong (self-igniting) attractor, a separate lever from the specificity
  one isolated here.

Runners: `research/runners/_riii_ca3_synchronous_assembly_derisk.py` (`bistable=True, enable_ou=, ca3_density=,
rate_homeo=`), `research/findings/raw/_gap5_wang_rate_homeo_driver.py`, `_gap5_sparse_specificity_sweep.py`.
