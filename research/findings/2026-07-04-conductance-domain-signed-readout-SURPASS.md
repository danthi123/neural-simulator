# RUNG B-1c signed read-out — CONDUCTANCE-DOMAIN attempt + an ANTI-CHEAT CORRECTION (2026-07-04)

## ⚠️ CORRECTION (anti-cheat, appended after the body below was written) — the "signed" claim does NOT hold

The body below claims a **conductance-domain SIGNED** read-out surpasses the 2/3 boundary to a clean 3/3. **The anti-cheat
REFUTES the mechanism attribution.** Lesion results at the N_BIAS=6 clean-3/3 config (`step2e_anticheat.py`, seeds 42/43/44):

| lesion | 42 | 43 | 44 | reading |
|---|---|---|---|---|
| INTACT | 18/18 | 18/18 | 18/18 | the result |
| SYN-LESION (zero res→ens synapses) | 6/18 | 6/18 | 6/18 | read-out synapses ARE load-bearing (collapses to chance) ✓ |
| FOLLOW-LESION (zero the reservoir→res_inh follower) | 18/18 | 18/18 | 18/18 | the inhibitory follower (Wn / neg-info) is **NOT** load-bearing ✗ |
| BIAS-LESION (zero the bias units) | 18/18 | 18/18 | 18/18 | the bias is **NOT** load-bearing ✗ |
| **BARE (Wp exc rows only — follower AND bias off)** | **18/18** | **18/18** | **18/18** | **the POSITIVE `Wp=max(Ws,0)` rows alone carry it** |

**⇒ the SIGNED machinery (the inhibitory follower + the bias units) is DECORATIVE.** The 18/18 is carried by the POSITIVE
`Wp=max(Ws,0)` excitatory rows at a low floor. So this is NOT a "signed" surpass — it is at most a **clipped-positive read
at a low floor**, and the whole conductance/follower/bias narrative below is not what does the work.

**Two things remain HONESTLY UNRESOLVED** (the disambiguation tool `step2f_bare_investigate.py` was CONFOUNDED — it used a
fixed scale instead of the per-read scale sweep, so its SHIFT read gave 0/18 on seed 42 where the c2 harness gives 18/18,
i.e. its numbers are not trustworthy):
1. **clip vs floor:** whether `Wp=max(Ws,0)` (clipping the negatives) or the low floor (30 vs the c2's 150) is what resolves
   seed 44 — a prior clean test had the c2 `Ws_shifted` positive read at floor 30 = only 12/18 on seed 44, which hints the
   CLIPPING matters, but this is not cleanly established.
2. **generalization / build-dependence:** the 18/18 is only established on the 3 TUNED seeds (42/43/44) in the specific
   `step2d` build; a minimal Wp-only build (`step2f`, albeit confounded) did NOT reproduce it on seed 42, so the result may
   be **build/operating-point-fragile** and its generalization to unseen seeds (100/101/102) is UNVERIFIED.

**Status: the committed B-1c.2 close-out remains GO 2/3 (positive Dale-shifted read). This "3/3 surpass" is DOWNGRADED to an
unverified, possibly-fragile positive-read-at-low-floor result whose mechanism is NOT the claimed signed circuit.** The
anti-cheat did its job — it caught the overclaim before it stood.

### FINAL VERDICT (the fair 6-seed generalization test — `step2g_clip_vs_shift_fair.py`, c2 harness + per-seed scale sweep, floor 30)

| seed | CLIP `max(Ws,0)` | SHIFT `Ws−min` |
|---|---|---|
| 42 (tuned) | 18/18 | 18/18 |
| 43 (tuned) | 18/18 | 18/18 |
| 44 (tuned) | 16/18 | 12/18 |
| **100 (unseen)** | **0/18** | 7/18 |
| **101 (unseen)** | **6/18** | 10/18 |
| **102 (unseen)** | **5/18** | 10/18 |

**⇒ NO generalizing surpass. The clipped-positive read is 0–6/18 on the UNSEEN seeds 100/101/102 (worse than even Ws−shift
there); the "18/18 on 42/43/44" was OVERFIT to the three tuned seeds.** So the whole arc fails rigorous validation on BOTH
counts: the signed mechanism was **decorative** (anti-cheat) AND the positive read that carried it was **overfit** (6-seed
test). **The B-1c.2 read-out shortcut close-out honestly STANDS at GO 2/3; the degraded-seed-44 residual is NOT surpassed by
any approach tried here.** This is an honest NEGATIVE — the anti-cheat + 6-seed generalization did exactly their job, and per
the BRAIN-BASED-ONLY standard an honest negative that maps a substrate limit IS the deliverable. The body below is retained
verbatim as the (now-fully-corrected) record of the attempt; its headline claims are RETRACTED.

### FEATURE-conditioning also fails — the boundary is now MULTIPLY-CONFIRMED

The read-out attack surface being exhausted, the FEATURE surface (Design-D) was tested too: (a) reservoir conditioning
RES_N 300→600 = seed-44 8/18 (no lift over 11); (b) an on-substrate RESERVOIR COMMITTEE (M=3 independent reservoirs,
concatenated features, `step3_committee.py`) = seed-44 **10/18 (no lift)**. ⇒ neither more reservoir neurons nor draw-
diversity resolves seed 44. **This makes it a MULTIPLY-CONFIRMED boundary across BOTH attack surfaces** (read-out: signed/
positive/divnorm/latency all overfit-or-decorative; feature: RES_N + committee no help). Root cause: the Dale-offset makes
the winning margin ~1% of the ens drive (the B-1c.2 finding's own named "sub-1% post-offset margin"), and a point-neuron
spike-COUNT read cannot reliably resolve a sub-1% margin across variable draws under the non-monotone Izhikevich f-I. This
is the **graded-magnitude / rate-code-wall family** the project has repeatedly documented (the analog/sub-threshold read a
point-neuron substrate structurally lacks). ⇒ dispatched a focused research gate on this precise boundary (isolate residual
→ reframe via biology → rank cheap-first surpass → verdict) before accepting it; see the CYCLE-919 research finding.

### FINAL (CYCLE-919): the research verdict REFRAMED the boundary, its #1 fix ALSO fails to generalize, and the committed read is itself SEED-FRAGILE

The research gate's measurement-grounded verdict REFUTED the "sub-1%-margin / degraded-feature / dendritic-frontier"
framing: it fit Ws + measured margins and found **DRIVE-WRONG=0/18** (the encoding delivers the correct winner in the
pre-spike drive on every slot; the isolated ens f-I is monotone to ≥450 pA). The seed-44 residual is a **WTA
IGNITION-ORDER inversion**, not depol block: the E→E amplification + I→E inhibition make the first-igniting AGENT ensemble
suppress the higher-drive THEME ensemble. Its #1 fix — a **feedforward per-role read** (per-row Dale shift
`Ws − Ws.min(axis=1)` [argmax-exact, 3-4× wider margin] + REMOVE the I→E inhibition + fixed operating point) — was built +
6-seed tested (`step4_feedforward.py`): **42/43 = 18/18 but 100/101/102 = 0/18** — removing I→E BROKE the unseen seeds
(they NEED the competition; seed 44 needs LESS of it → opposite needs, whack-a-mole around the WTA).

DECISIVE check (`step2g` at the committed floor 150): the **COMMITTED positive read (global shift + WTA) is 18/18 on
seed 42 but 7 / 9 / 5 out of 18 on the unseen 100/101/102** — near chance (6/18). ⇒ **the read-out is broadly
SEED-FRAGILE; the committed "GO 2/3" is accurate for the 3 development seeds 42/43/44 but does NOT generalize to a wider
seed set.** That is why every surpass this session OVERFIT (chasing a 3/3 on the two lucky seeds). The anti-cheats + 6-seed
tests caught it every time.

**The precisely-mapped boundary (the honest deliverable):** the spiking read-out of a tight-margin `argmax(f·Ws)` is
UNRELIABLE across reservoir draws — the WTA competition that sharpens the read on some draws (100/101/102) CAUSES an
ignition-order inversion on others (44); the Dale-offset pedestal (needed for excitatory-only synapses) shrinks the margin
the competition then mis-amplifies. NOT a sub-1%-margin wall (refuted), NOT the graded/dendritic frontier (refuted). The
genuine missing mechanism is a **SEED-ADAPTIVE read** (per-draw competition/normalization that neither under- nor
over-sharpens) — i.e. a learned/adaptive read-out, a deeper frontier than a fixed circuit. Mechanisms exhaustively tried +
anti-cheated + 6-seed-tested this session: signed count-opponent, conductance-signed (decorative), divisive-norm (plateau),
gain-cal, latency, low-floor positive (overfit), per-row shift, feedforward (broke unseen), RES_N conditioning, reservoir
committee. All in `research/findings/raw/signed_conductance/`.

---

# (original body — mechanism attribution CORRECTED above) RUNG B-1c signed read-out — CONDUCTANCE-DOMAIN attempt

**One-line:** the committed B-1c.2 read-out shortcut close-out was GO **2/3** (the positive Dale-shifted read-out; the
degraded seed 44 under-resolved at 11/18 and was named an irreducible "degraded draw" boundary). A **conductance-domain
signed read-out** — the design-workflow's ranked #1 — **surpasses it to a robust CLEAN 3/3**: it delivers the full signed
read-out `(Wp−Wn)@[f;1]` in the LINEAR pre-spike current domain, resolving the negative-weight information seed 44 genuinely
needs. **A SINGLE fixed operating point gives all of seeds 42/43/44 = 18/18** (the degraded seed 44 lifted 11→18), robust
across the whole tested region (**27/27 configs 18/18**: ratio 1.2–1.7 × bgain 4–10, floor 30, N_BIAS=6, c90 — not a
knife-edge). Strictly CPU/numpy; the mechanism is runner-side wiring, **NO `sim/` edit**.

## Why the positive read was 2/3 and the spike-count opponent was 0/18 on 42/43

The read-out reproduces `argmax_r((f·Ws)[r])` on the substrate. The committed **positive** read (Dale-shifted `Ws_shifted =
Ws − Ws.min()`, purely excitatory, winner = neural argmax over ensemble firing) is 18/18 on 42/43 but only 11/18 on the
DEGRADED seed-44 draw — its patient slot genuinely needs the **negative-weight** information the positive read discards.

The obvious fix — a signed ON/OFF **spike-count** opponent (`argmax_r(Σens_pos − Σens_neg)`) — resolves seed 44 (18/18)
but is **0/18 on 42/43**: subtracting spike COUNTS is `f_nonlin(Wp@f) − f_nonlin(Wn@f)`, and `f(a)−f(b)` does NOT preserve
`sign(a−b)` because the Izhikevich f-I is non-monotone at high drive (depolarization block). A 4-agent research-gate
workflow + two independent diagnostic lenses confirmed the block (role-0 at drive 526 fires 1439 spikes < role-1 at drive
400 fires 5075 — MORE drive, FEWER spikes). Divisive-norm (9/18), gain-calibration (6/18), integration (7/18),
decorrelation (0/18 unchanged), rank-order latency, and reservoir-conditioning (RES_N 300→600, no lift) all failed to make
the spike-count subtraction faithful — because the defect is the subtraction being AFTER the nonlinearity.

## The mechanism (Design A): subtract in the CONDUCTANCE domain, before the spike nonlinearity

`I_syn = g_e·(E_e−v) + g_i·(E_i−v)` (`E_e=0`, `E_i=−75`) — excitatory and inhibitory conductances sum **linearly in the
pre-spike current**. Put `Wp=max(Ws,0)` on the **excitatory** conductance of ONE ensemble per role and `Wn=max(−Ws,0)` on
its **inhibitory** conductance; the net drive is a monotone-affine function of `(Wp−Wn)@f = Ws@f` — the true signed logit —
and a single monotone f-I of it preserves the argmax (exactly why the positive read is 18/18). **No spikes are ever
subtracted.** Realized on the substrate:

- **reservoir** (exc, RES_N) — the fixed-random spiking LSM (unchanged).
- **res_inh** (inh-trait, RES_N) — a **1:1 FOLLOWER** copy: `reservoir[i]→res_inh[i]` unweighted (a linear spike-RELABEL,
  NOT a weighted-sum threshold), so its firing == the reservoir's but routes through `g_i`. This is the fix to the prior
  B-1c relay's 0/18 failure — that relay was a *spiking interneuron* thresholding `Wn@f` (re-inserting the nonlinearity).
- `reservoir → ens[r]` weighted `Wp[:,r]·sc` (exc); `res_inh → ens[r]` weighted `Wn[:,r]·sci` (inh). `ratio = sci/sc ≈ 1.5`
  compensates the driving-force asymmetry `|E_e−v|/|E_i−v| · prop/inh_prop`.
- **BIAS UNITS** (a small exc+inh-follower population, constant-rate) deliver the `+1` intercept row `Ws[n_res,:]`
  SYNAPTICALLY (exc for +bias, inh for −bias) — additive/subtractive, NOT a silencing tonic. (A direct per-role tonic
  SILENCES a host role whose bias is negative: seed-43 slot2's host bias is −0.54 → tonic floor `30 + (−0.54)·280 = −121`
  → host role fires 0 → 12/18. Synaptic delivery makes it a small hyperpolarizing current instead → 18/18.)
- Winner = **neural argmax over ens summed firing**. NO host `f@Ws`, NO host argmax deciding the role.

**Critical operating point: LOW floor (~30 pA).** At a high floor the ensemble's `v` is depolarized, so `g_i·(E_i−v)` is a
strong DIVISIVE (shunting) current — the subtraction stops being linear and a strongly-inhibited role gets shunted below an
un-inhibited one (floor 150–250 → 0/18). At a LOW floor `v` sits near rest, `E_i−v ≈ −15` is small, and `g_i` is near-
SUBTRACTIVE → the conductance subtraction is faithful (floor 30 → 18/18).

## Results (seeds 42/43/44, CPU/numpy; host-agree over the 18 canonical content slots)

| read-out | seed 42 | seed 43 | seed 44 (degraded) |
|---|---|---|---|
| POSITIVE (committed B-1c.2, Ws_shifted) | 18/18 | 18/18 | **11/18** (the 2/3 boundary) |
| signed spike-COUNT opponent | 0/18 | 0/18 | 18/18 (but block-fails 42/43) |
| **CONDUCTANCE signed, single SHARED config (N_BIAS=6)** | **18/18** | **18/18** | **18/18** |

⇒ **the degraded-seed-44 boundary is SURPASSED to a robust CLEAN 3/3**: the negative-weight info, delivered in the linear
conductance domain, resolves what the positive read never could (11→18), and a SINGLE fixed operating point gives all three
seeds 18/18.

**How the single-shared-config residual was CLOSED — the bias-population noise level.** The bias delivery had a per-seed
conflict: seed 43's slot2 host role has a NEGATIVE bias (wants weaker/noisier delivery so it isn't over-suppressed), seed 44
(degraded draw) wants stronger/cleaner delivery. A single bias unit (N=1) is too noisy → 43:18/44:15; a large population
(N=16) is too smooth → 43:12/44:17. The **intermediate N_BIAS=6** is the sweet spot — enough averaging for seed 44's clean
bias, enough residual variance to not over-suppress seed 43's negative-bias slot — and it is WIDE: **27/27 configs 18/18**
across ratio 1.2–1.7 × bgain 4–10. (Since each bias unit carries `1/N_BIAS` of the intercept, N_BIAS is a pure noise/
smoothness knob at fixed total bias strength — a biologically-natural population-coding lever, not a magnitude hack.)

## Honest scope / next

- **3-seed de-risk** on the isolated read-out (host-agree), the same bar the B-1c.2 boundary was reported at. The
  confirmation step: promote to a durable runner, run the **6-seed** rule (+100/101/102), the **anti-cheats** (syn-readout
  lesion collapses; the res_inh follower + bias units are load-bearing; source-clean = no host `f@Ws`/argmax), and wire it
  into the actual `UnifiedBrainBridge` comprehend→select→bind close-out (route 12/12 == dict).
- The single-shared-config residual on seed 44 (the bias-magnitude conflict) is the one bounded item; a **self-calibrated**
  bias delivery (the bias unit's rate tracking the reservoir's per-seed firing scale, so bias/rows gain is seed-consistent)
  is the named next mechanism.
- Scratchpad de-risk chain: `step2_signed_conductance.py` (tonic bias, 42=18/44=18/43=12), `step2c_syn_bias.py` (synaptic
  bias, 42=18/43=18/44=15), `step2d_bias_pop.py` (bias population). Design: the workflow `signed-readout-full-closure-design`.

**NO `sim/` edit** anywhere (reservoir/follower/bias-unit wiring + the divisive/conductance operating point are all
runner-side `set_pathway_weights` + `cp_traits` + external-current, consistent with the whole B-1c arc).
