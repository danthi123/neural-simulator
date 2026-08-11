---
type: finding
status: contributing
claim_check: synthesis
date: 2026-08-11
mechanism: ROLE-GATE x HIDDEN-LAYER + CHAINED-FA + sigma' — add a hidden sigmoid population (barcode + recurrent latch -> hidden -> the scalar load-logit) to the role-gate and propagate credit TRANSPORT-FREE by chained multi-hop Feedback Alignment + the surrogate-derivative sigma' (the 2026-08-01 load-bearing ingredients), via a two-pass forward-eligibility (e-prop) rule; feedback arms aligned (transport ceiling) / chained_fa (fixed-random, brain-based candidate) / chained_kp (co-adapting Kolen-Pollack at both hops)
lane: emergence engine / working memory x gap#4 / role-gate transport-free reliability
verdict: 6-SEED (42 43 44 100 101 102) real-slot at L=2/3/4 (GO distance = L4, chance 0.250, marker ceiling 1.000/all seeds/all L). HONEST NEGATIVE at the transport-free reliability bar; the CEILING clears. A hidden layer does NOT make the TRANSPORT-FREE chained-FA reach role RELIABLY (chained_fa L4 mean 0.422 [min 0.222], gap +0.23 [min −0.14] — FAILS GO: needs mean≥0.55, min≥0.60, gap_min≥0.30), but the aligned (weight-TRANSPORT) ceiling with the SAME hidden layer reaches role RELIABLY on all 6 seeds at all L (1.000 [min 1.000], gap +1.00 [min +1.00]) — so the 2-layer architecture + the chained-credit MECHANISM express and learn role perfectly when feedback is exact. THE RESIDUAL, isolated (the exact question — alignment / sigma' / depth / reliability): it is RELIABILITY, not the other three. NOT depth (aligned = 1.000/6 at every L). NOT sigma' (present + load-bearing: chained_fa 0.422 vs no-sigma' 0.267). NOT feedback alignment (the co-adapting chained_kp arm FULLY recovers alignment at BOTH hops, cos hopA +0.96 / hopB +0.99 — structural, not dimensional) — YET chained_kp still collapses on some seeds (L4 min 0.133). So alignment is necessary-not-sufficient; the seed-dependent collapse into a non-role (fire-everything) basin survives a hidden layer AND full feedback re-alignment. This advances LEVER-1's "variance intrinsic to the single-layer joint dynamics" to: the transport-free reliability wall is NOT single-layer expressivity and NOT a feedback-alignment problem — it is a memorise-phase basin instability (Refinetti 2021 align-then-memorise; alignment does not guarantee the fitting phase succeeds). Anti-cheats bite 6/6: sigma' load-bearing, permuted-reward collapses (gap +0.01), identity crux fails (gap +0.00), lesion-the-hold 0.046, HTM/n-gram/perm-pos at chance. NO sim/ edit; SIM_BACKEND=numpy.
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_var_bind_rolegate_hidden_chainedFA_derisk.py
artifacts:
  - research/findings/raw/_rolegate_hidden_chainedFA/rolegate_hidden_chainedFA_6seed.json
  - research/findings/raw/_rolegate_hidden_chainedFA/seed_42.json
  - research/findings/raw/_rolegate_hidden_chainedFA/seed_43.json
  - research/findings/raw/_rolegate_hidden_chainedFA/seed_44.json
  - research/findings/raw/_rolegate_hidden_chainedFA/seed_100.json
  - research/findings/raw/_rolegate_hidden_chainedFA/seed_101.json
  - research/findings/raw/_rolegate_hidden_chainedFA/seed_102.json
---

# Role-gate x hidden-layer + chained-FA + sigma' — the transport-free RELIABILITY residual survives a hidden layer AND full feedback re-alignment; the residual is isolated to the memorise-phase basin (not alignment, not sigma', not depth). The aligned ceiling clears 6/6. 6-seed HONEST NEGATIVE.

## One-line verdict

LEVER 2 (the structural lever) on the role-gate's transport-free reliability residual. Hypothesis: the
single-layer role-gate cannot express the gap#4 lane's PROVEN transport-free mechanism (chained multi-hop
Feedback Alignment + sigma', banked 2026-08-01), so a HIDDEN LAYER + that mechanism should reach RELIABLE
transport-free role induction where single-layer canonical KP could not. **Result: the hypothesis is
FALSIFIED at the reliability bar, and the negative is precisely isolated.** A hidden layer + chained-FA +
sigma' does NOT reach role reliably transport-free (6-seed L=4 mean 0.422, min 0.222) — but the SAME 2-layer
gate with an aligned (weight-transport) feedback ceiling reaches role on **every** seed at **every** distance
(1.000 [min 1.000]). The residual is not depth, not sigma', not feedback alignment — it is **RELIABILITY**
(the memorise-phase basin), the same wall LEVER 1 hit, now shown to survive both a hidden layer and full
feedback re-alignment.

## The 6-seed result (real spiking D3 slot; GO distance = L4; chance 0.250; marker ceiling 1.000)

Artifact: `research/findings/raw/_rolegate_hidden_chainedFA/rolegate_hidden_chainedFA_6seed.json` (the merged
6-seed aggregate + verdict; per-seed sources `seed_42.json` … `seed_102.json` in the same directory).

Per-arm held-out branch(verb) accuracy, mean [min over the 6 seeds], at the GO distance L=4 (dist 5, 20 736
distractor paths, novel held-out tuples):

| arm | mean | min | per-seed (42/43/44/100/101/102) | cos hopA→ / hopB→ | reads |
|---|---|---|---|---|---|
| **hidden_ALIGNED** (transport CEILING) | **1.000** | **1.000** | 1.0 / 1.0 / 1.0 / 1.0 / 1.0 / 1.0 | +1.00 / +1.00 | reliable — the mechanism + architecture are correct |
| **hidden_chained_FA** (transport-free, THE candidate) | 0.422 | 0.222 | 1.0 / 0.378 / 0.367 / 0.256 / 0.222 / 0.311 | −0.06 / +0.57 | high-variance; only seed 42 clears |
| hidden_chained_KP (transport-free, co-adapting) | 0.578 | 0.133 | 0.256 / 0.5 / 0.133 / 0.8 / 0.778 / 1.0 | **+0.96 / +0.99** | aligns fully yet STILL collapses on ≥1 seed |
| single_layer kp_canon (the ref to beat) | 0.326 | 0.089 | 0.733 / 0.111 / 0.8 / 0.089 / 0.1 / 0.122 | +0.99 (single hop) | the banked high-variance single-layer |
| REINFORCE (matched budget) | 0.752 | 0.222 | 1.0 / 0.289 / 0.222 / 1.0 / 1.0 / 1.0 | — | bimodal per seed (role basin or collapse) |
| marker (scaffold ceiling) | 1.000 | — | — | — | memory carries role when timing is given |

Reliability holds nowhere transport-free: at L=2 chained_fa min 0.167, L=3 min 0.067, L=4 min 0.222; chained_kp
L=2 min 0.111, L=3 min 0.667, L=4 min 0.133. The aligned ceiling is 1.000 [min 1.000] at all three L.

GO gate (transport-free chained_fa, at L=4): mean 0.422 < 0.55 ✗; min 0.222 < 0.60 ✗; gap_min −0.14 < 0.30 ✗
→ **NOT-GO.** `role_go=False`.

## The residual, isolated — alignment / sigma' / depth / reliability (the exact question)

1. **Depth / architecture — NOT the residual.** The 2-layer aligned (transport) ceiling reaches role
   1.000 [min 1.000] on all 6 seeds at L=2, 3 AND 4. The hidden-layer architecture and the chained two-pass
   e-prop credit MECHANISM express and learn the role solution perfectly when the feedback is exact. Adding
   depth was not the blocker.
2. **sigma' — load-bearing, and present in the candidate.** Dropping the hidden sigma' from the candidate
   collapses it: chained_fa 0.422 → chained_fa-no-sigma' 0.267 (L4; and 0.463 → 0.376 at L2). This reproduces
   the 2026-08-01 cube's headline (sigma' the largest, tightest main effect). sigma' is not the missing piece
   — it is already in the candidate and doing work.
3. **Feedback alignment — recoverable, STRUCTURAL, but NECESSARY-NOT-SUFFICIENT.** The fixed-random FA arm's
   feedback only partially aligns (hopA stuck at −0.06 — a fixed-random matrix has no co-adapt attractor;
   hopB +0.57). The co-adapting Kolen-Pollack arm (Akrout 2019: co-adapt forward + feedback + weight decay at
   BOTH hops) FULLY recovers alignment — **cos hopA +0.96, hopB +0.99** — confirming alignment is structural,
   not dimensional, now at 2-layer depth (mirroring the single-layer finding). KP's better mean (0.578 vs
   0.422) shows alignment helps. **But even with cos +0.99 at both hops, KP min stays 0.133** — a seed still
   fully collapses.
4. **RELIABILITY — THE residual.** The decisive isolation: full feedback re-alignment (KP cos +0.99 both hops)
   does NOT buy reliable role. The seed-dependent collapse into a non-role fixed point (the gate settling into
   fire-everything / wrong-position) survives (a) a hidden layer, (b) the proven chained-FA + sigma' mechanism,
   AND (c) full feedback alignment. So the reliability wall is NOT single-layer expressivity and NOT a
   feedback-alignment problem — it is a basin-of-attraction / operating-point instability in the FORWARD gate
   dynamics under transport-free credit.

The per-seed columns show the signature: every arm except the aligned ceiling is **bimodal** — each seed either
lands the role basin (≈1.0) or collapses (≈chance). Only exact weight transport steers every seed into the role
basin; transport-free credit (FA or fully-aligned KP) does not.

## What this advances beyond LEVER 1

LEVER 1 (readout-regularization, banked HONEST-NEGATIVE, commit 874f543b) concluded the variance is "intrinsic
to the SINGLE-LAYER joint R+B+gate dynamics." LEVER 2 sharpens that: the reliability variance is **not** a
single-layer limitation and **not** a feedback-alignment problem — it survives adding a hidden layer AND fully
re-aligning the transport-free feedback (KP cos +0.99). The remaining residual is the memorise-phase basin
instability, isolated away from alignment, sigma', and depth. Two levers (readout-reg, hidden-layer) are now
banked against the same defect; the next move is external-literature-guided (below), not a third quick lever.

## Anti-cheat teeth (all bite, 6/6)

- **sigma' load-bearing:** chained_fa 0.422 vs chained_fa-no-sigma' 0.267 (drop 0.155). The surrogate
  derivative is doing real work on the transport-free path.
- **permuted-reward → no learning:** shuffle the verb target per sentence → role SELECTIVITY collapses:
  token-identity gap +0.01 (vs the real arm's +0.23), acc 0.204 ≈ chance. The gap is the principled teeth
  here (acc is confounded upward by the shared early-firing homeostasis prior).
- **token-identity crux — the identity control FAILS as required:** the code-only identity gate has gap +0.00
  (gates on token class, cannot condition on position); the role arms have positive gaps. Selectivity is
  reported with this control, the permuted-reward control, and the raw fire rates (pos0 vs pos>0).
- **lesion-the-hold:** recur=0 slot (structural, holds by construction) → 0.046 (memory dies over the
  distractor span).
- **task validity:** marker ceiling 1.000; HTM 0.000; best n-gram held-out floor ≈0.27 ≈ chance;
  permuted-position ≈0.26 ≈ chance (the task is genuinely POSITIONAL). 7/7 earned verdict preconditions pass.
- **transport-free by construction:** the chained_fa/kp backward path reads only the feedback matrices B_out
  (hop A) and B1 (hop B) — never a forward weight's transpose (R⊤/W2⊤/W1⊤). KP's cos +0.99 is EARNED by
  co-adaptation, not copied; only the labelled `aligned` arm uses weight transport (B_out=R⊤, B1=W2).

## External anchor (the isolated residual, verified)

Refinetti, d'Ascoli, Ohana, Goldt, "Align, then memorise: the dynamics of learning with feedback alignment,"
ICML 2021 (arXiv:2011.12428): FA learning proceeds in two phases — an **alignment** phase (the random feedback
aligns with the true gradient) followed by a **memorise** phase (fitting the data) — and alignment is
necessary but does NOT guarantee the memorise phase succeeds (FA aligns yet notoriously fails to train
conv nets). This is exactly the isolated residual here: our KP arm reaches the alignment phase (cos +0.99 at
both hops) yet the memorise phase reliably fits the role solution on only some seeds. The sigma' anchor stands
from the 2026-08-01 finding (WF-Act-PC, arXiv:2607.13380 — FA collapses at depth precisely because it drops
sigma'). Next-candidate pointer surfaced by the same search: "Overcoming Rank Collapse in Feedback Alignment"
(arXiv:2606.11123) — a low-rank collapse is a candidate mechanism for the fire-everything basin.

## Honest scope + the next candidate

- **Scope:** a numpy/CPU rate-and-real-slot de-risk (F=4, N_noun=12, hidden H=32, 80-episode matched budget,
  seeds 42/43/44/100/101/102, L=2/3/4, held-out novel-distractor eval on the banked D3 spiking slot). The
  2-layer net + chained-FA credit are HOST math; their on-substrate spiking DA-gated realisation is the named
  next rung (unchanged from the single-layer arc). NO sim/ edit.
- **The next candidate follows from the isolation** (reliability = the memorise-phase basin, not alignment):
  do NOT attack alignment again (KP already recovers cos +0.99). Attack the FORWARD basin instability. The
  "what companion process did we replace with a constant?" reframe points hardest at a **competitive /
  normalising stabilizer** — the collapse is INTO fire-everything, exactly what lateral inhibition / a
  winner-take-all / a divisive-normalisation companion would forbid, and this de-risk proxied that with only a
  scalar firing-rate homeostatic nudge (homeo=0.10). Concretely, the ranked next levers: (a) a competitive
  write-gate population (k-WTA / lateral inhibition that structurally forbids the fire-everything fixed point)
  trained WITH this transport-free rule; (b) a curriculum / warm-start that seeds the role basin (short-L
  first) so the memorise phase starts inside it; (c) periodic feedback↔forward sync (Frozen-Backprop
  arXiv:2505.13741) to stabilise the memorise phase; (d) check the rank-collapse hypothesis (arXiv:2606.11123)
  on the collapsing seeds directly. Per the deep-research discipline, (a)–(d) start from the external record
  above, not a blind third lever.

## Disposition

- **Status: contributing.** A precise 6-seed HONEST NEGATIVE that (i) confirms the 2-layer architecture +
  chained credit MECHANISM reach role reliably at the transport ceiling, (ii) confirms sigma' is load-bearing
  and feedback alignment is fully recoverable transport-free (KP cos +0.99), and (iii) isolates the remaining
  residual to memorise-phase RELIABILITY (basin instability), redirecting the sub-arc from alignment (solved)
  to a competitive/normalising forward stabilizer.
- The banked sibling
  (`2026-08-11-rolegate-gap4-deep-credit-resolves-role-at-ceiling-transport-free-alignment-STRUCTURAL-not-dimensional-6seed`)
  stands; this finding extends its "residual is now RELIABILITY, not alignment" tail with the depth +
  full-alignment controls that rule out expressivity and alignment as the cause.
- Runner: `research/runners/_var_bind_rolegate_hidden_chainedFA_derisk.py` (reuse-by-import of the gap#4-credit
  runner's stream / SpikingSlot eval / crux / EpropCreditGate reference; the 6-seed sweep runs in PARALLEL and
  aggregates through the SAME verdict code via `--merge-from`, verified byte-equivalent to a native
  multi-seed run).
