---
type: finding
status: design
claim_check: synthesis
date: 2026-09-03
mechanism: DESIGN — the SPIKE-NATIVE realization of the linattn mouth's content-weighted num/den normalization (the last non-spiking op in the confirmed open-fluency breakthrough) as divisive normalization by a shunting-conductance normalization pool, over the query's match-mass axis
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [42, 43, 44, 100, 101, 102]
verdict: DESIGN NOTE (no new measurement) — closes the named honest residual of the linattn milestone; specifies how a spiking substrate computes the num/den division, the reused on-substrate hooks, the two-tier de-risk + GO gate, and the honest brain-based-purity residual if genuine (non-subtractive) division is not fully achievable on spikes. No sim/ file and no runner is edited by this doc.
artifacts:
  - research/findings/2026-09-03-OPEN-FLUENCY-BREAKTHROUGH-linattn-deployable-spiking-mouth-beats-trigram-6of6.md
  - research/findings/2026-09-03-spiking-content-addressable-read-DESIGN.md
  - research/findings/2026-09-03-linattn-production-mouth-wiring-DESIGN.md
  - research/findings/raw/_linattn_smoke_normON.json
  - research/findings/raw/_linattn_smoke_normOFF.json
  - research/findings/raw/_emerge_wkv_lm_linattn_depth2_contiguous_6seed.json
---

# DESIGN — the spike-native num/den normalization for the linattn own-voice mouth

**This is a DESIGN NOTE, not a measured result.** It closes the ONE honest residual named by the open-fluency
milestone (`2026-09-03-OPEN-FLUENCY-BREAKTHROUGH-linattn-deployable-spiking-mouth-beats-trigram-6of6.md`, commit
`43c5b6b4`): the `linattn` mouth crossed the trigram bar 6/6 (mean **+0.0505**), and its read
`= φ(q_t)ᵀM_t / (φ(q_t)ᵀzden_t + ε)` is spike-native in state (graded fast-weight traces, same concession as
the shipped ssm mouth) and read-out (`FewSpikeWordRead`) — with exactly ONE graded host op left: the **num/den
DIVISION** (the content-weighted normalization). The milestone flags its "shunting-inhibition spike-native
realization" as a named later rung. This doc specifies that realization: how a spiking substrate computes a
content-weighted divisive normalization, grounded in a verified read of the divisive-normalization biophysics,
with the reused on-substrate hooks, the cheapest de-risk + GO gate, and the honest brain-based-purity residual if
genuine (non-subtractive) division is not fully reachable on spikes. **No `sim/` file and no runner is edited by
this doc.** It edits no GPU state and does not touch the live verification.

## 1. What the division does, and why it is load-bearing (pinned to code + measurement)

The read, per position `t`, in `LinAttnLayer.forward` (`research/runners/_emerge_wkv_lm_derisk.py`, line ~689,
and identically in the deployment `LinAttnReadout.advance_and_logits`, production-wiring DESIGN §3e):

```
num_t  = φ(q_t)ᵀ M_t         # D-vector: query-weighted retrieved values (the associative recall)
den_t  = φ(q_t)ᵀ zden_t      # SCALAR (per token, per sample): the query's total match-mass over past keys
read_t = num_t / (den_t + ε) # ε = 1e-6 — content-weighted AVERAGE (softmax-free normalized retrieval)
```

Three facts fix the shape of the spiking realization:

1. **`den_t` is a SINGLE SCALAR.** `den_t = Σ_d φ(q_t)_d · zden_d = Σ_s λ^{t-s} (φ(q_t)·φ(k_s))` — the total
   query-key overlap summed over past positions `s` (softmax's denominator). All `D` channels of `num_t` are
   divided by the SAME scalar. So this is a **common-mode gain control** — one divisor shared across the whole
   read pool — which is the simplest and most canonical form of divisive normalization (Carandini & Heeger's
   single shared normalization pool), NOT a per-channel operation.

2. **The divisor is over the MATCH-MASS axis, not the channel population.** This is the axis lesson the arc
   already paid for: `--dual-nonneg-divnorm` pooled over the `D` channels (`R_i = ap2_i / (σ + Σ_j ap2_j)`) and
   was a NO-GO at every σ (collapsed to −2.98), because it squashed every channel toward `1/D`
   (`2026-09-03-spiking-content-addressable-read-DESIGN.md` §4). The linattn denominator is a scalar summarizing
   `q · Σ_s φ(k_s)` — the query's total match against the accumulated keys — which requires the `φ(k)` write gain
   to be present so the divisor is informative. This is the correct axis, and it is why the spiking realization
   below is a normalization POOL driven by `den_t`, never a cross-channel pool.

3. **The division is LOAD-BEARING, measured.** CPU ablation (word-level TinyStories smoke, V=800, d96, 1-seed;
   `research/findings/raw/_linattn_smoke_normON.json` / `_normOFF.json`; recorded in the milestone and the
   production-wiring DESIGN §5), deepest bucket d10-99, `margin_vs_trigram`:

| arm | d10-99 margin_vs_trigram | reading |
|---|---|---|
| linattn norm ON (φ=elu) | **+0.456** | best; uses long-range content + order |
| linattn norm OFF (raw num, no divide) | **+0.190** | normalization dropped ⇒ margin more than halves |
| exact-wkv baseline (same config) | +0.429 | linattn ≥ the wkv upper bound it generalizes |

So a spike-native version MUST PRESERVE the division: dropping it (norm OFF) more than halves the margin (0.456 →
0.190). Any on-substrate realization that degrades the division toward subtractive behaviour (Holt & Koch, §2)
risks landing at the norm-OFF level — which is the precise failure the GO gate (§4) measures against.

## 2. The biophysics of a spiking divisive normalization (verified this session)

**Divisive normalization is a canonical cortical computation** (Carandini & Heeger 2012, "Normalization as a
canonical neural computation", *Nat Rev Neurosci* 13:51–62, doi:10.1038/nrn3136; original Carandini & Heeger
1994, *Science* 264:1333–1336): a neuron's response is its excitatory drive divided by a pooled signal,
`R_i = drive_i / (σ + pool)`. The pool is the summed activity of a normalization set; here that pool is a single
scalar `den_t` (the match-mass), so the read pool's target computation `num_i / (σ + den)` is EXACTLY the C&H
form with a one-dimensional pool.

**The biophysical substrate of division is a conductance in the denominator — shunting inhibition.** A neuron's
input-current → firing-rate gain scales inversely with its total membrane conductance `g_total = g_leak + g_inh`.
An inhibitory conductance with reversal near rest (`E_inh ≈ E_L`, "shunting") that grows with the pool activity
raises `g_total` and so DIVIDES the gain: `rate_i ∝ I_num_i / (g_leak + k·g_pool)`. Mapping `σ = g_leak`,
`g_pool ∝ den`, this is the read.

**The honest caveat (Holt & Koch 1997, *Neural Comput.* 9:1001–1013): pure somatic shunting is SUBTRACTIVE, not
divisive, on the mean firing rate.** For a point neuron driven by a steady superthreshold current, adding a
shunting conductance mostly shifts the f–I curve rightward (raises rheobase) rather than scaling its slope — the
effect on rate is a subtraction, not a division. A naive "shunt the soma" realization would therefore land near
the norm-OFF failure, not the norm-ON read. This is the crux the design must clear, and there are three verified
routes to GENUINE (slope-scaling) division:

- **(R-fluct) Fluctuation-driven / high-conductance regime with noisy balanced background.** When the read
  neuron is driven in the fluctuation-dominated regime (a barrage of balanced excitation+inhibition, the cortical
  "high-conductance state"), an increase in total conductance IS divisive on the f–I curve — the gain scales with
  the noise-set slope. Demonstrated in real cortical pyramidal neurons by **Chance, Abbott & Reyes 2002**
  ("Gain modulation from background synaptic input", *Neuron* 35:773–782 — verified) and in cerebellar granule
  cells by **Mitchell & Silver 2003** ("Shunting inhibition modulates neuronal gain during synaptic excitation",
  *Neuron* 38:433–445 — verified): tonic/shunting inhibition during *frequency-dependent (noisy) excitation*
  reduces the GAIN (slope), i.e. divides. Review: **Silver 2010**, "Neuronal arithmetic", *Nat Rev Neurosci*
  11:474–489 (already an in-repo anchor). The operating point — the presence of balanced noise — is the
  companion process that turns Holt & Koch's subtraction into division. This is the "what else does the real
  system run alongside this?" reframe applied directly: the divisor is not a static shunt, it is a shunt IN a
  fluctuation-driven regime, and the regime is part of the mechanism.

- **(R-dend) Dendritic / conductance-level shunting before somatic spike generation.** A shunting conductance on
  the same dendritic branch as the excitatory input divides the LOCAL dendritic depolarization by passive-cable
  interaction, before the soma integrates it — a genuine division that a point-neuron soma "provably cannot do"
  (the substrate's own `enable_dendritic_divisive_gain` config comment, `sim/config.py:813-818`, echoing Silver
  2010). This is the most faithful single-neuron route and reuses an existing hook (§3d).

- **(R-net) Emergent network normalization (SSN).** Divisive normalization emerges as the fixed point of a
  recurrent excitatory–inhibitory microcircuit whose units have a supralinear (power-law) f–I — the Stabilized
  Supralinear Network (**Rubin, Van Hooser & Miller 2015**, "The stabilized supralinear network: a unifying
  circuit motif underlying multi-input integration in sensory cortex", *Neuron* 85:402–417 — verified). Here the
  division is not imposed by any formula; it is the settled state of the read pool + normalization interneurons.
  Most faithful, most emergent, hardest to match to the exact rate-level read — the honest-negative candidate and
  the longest rung.

**Convergent lesson.** The scalar-`den` shape means the substrate does NOT need a per-channel divisive circuit —
it needs ONE normalization pool whose pooled activity is `den`, shunting a `D`-neuron read pool that carries
`num`, operated in a regime where the shunt is divisive (R-fluct as the deployable rung; R-dend/R-net as the
more-faithful rungs). This is the same architecture C&H, Chance-Abbott-Reyes, and SSN all describe.

## 3. THE DESIGN — a shunting normalization pool over the match-mass axis

The division is an INTERMEDIATE computation inside each linattn layer (the residual stream threads through it and
into the next layer / the head; only the FINAL head logits are read out by `FewSpikeWordRead`). So the division
must be done by neurons per layer. The circuit, per linattn layer:

```
  φ(q_t) rate code  ──┬─────────────────►  READ POOL (D neurons)   ── g_e ∝ num_i = φ(q)ᵀ M_{:,i}
   (D presyn rates)   │                         │  (synapses = M column i, the fast-weight KV trace)
                      │                         │  g_shunt ∝ den  (GABA_A, E_inh ≈ E_L)   ◄──────┐
                      └─────────────►  NORM NEURON (1, or small pool) ── rate = den = φ(q)ᵀ zden │
                                          (synapses = zden trace, the running key-match mass) ───┘
  read_i (settled rate of read neuron i)  ≈  num_i / (g_leak + k·den)  =  num_i / (σ + den)
```

### (a) How `zden` (the denominator trace) and `den` are represented + read as spiking/conductance signals

- **`zden_t` is a graded fast-weight vector** (`D` synaptic efficacies), exactly as in the shipped design: a
  short-term-synaptic-plasticity / calcium trace (Mongillo, Barak & Tsodyks 2008, *Science* 319:1543–1546 —
  in-repo anchor; Ba et al. 2016 fast weights). The state stays graded — the SAME concession the ssm mouth's
  `ap/an` already carries and the whole spiking-LM literature keeps; only I/O is spiked.
- **`den_t` is read as the FIRING RATE of a single NORMALIZATION INTERNEURON** (or a small redundant pool for
  robustness). Its `D` input synapses carry the `zden` trace as their efficacies; its presynaptic drive is the
  `φ(q_t)` rate code. Its output rate is `Σ_d φ(q_t)_d · zden_d = den_t` — a rate-coded synaptic sum, the
  Carandini-Heeger normalization pool realized as one interneuron. `den ≥ 0` always (φ non-negative,
  `zden = Σ λ^{t-s} φ(k_s) ≥ 0`), so the derived shunt conductance is always a valid non-negative conductance —
  no sign problem.
- **`num_t` is the excitatory drive of the READ POOL** (`D` neurons): read neuron `i` receives `g_e ∝ num_i =
  φ(q)ᵀ M_{:,i}`, a Hebbian-fast-weight-weighted synaptic matvec (presynaptic query-rate `φ(q_d)` × fast-weight
  synapse `M_{d,i}`). Signed values (`v` is signed) use ON/OFF rate sub-pools `[relu(num_i), relu(-num_i)]`
  differenced at `Wo_sp`, exactly as `dual-nonneg` already splits `v`; the shunt divides both sub-pools equally
  (common gain), so `(ON−OFF)/(σ+den) = num_i/(σ+den)`.

### (b) How the division is realized (the shunting conductance)

The norm neuron projects a **shunting GABA_A conductance** (`E_inh ≈ E_L`, near rest) onto every read neuron,
with `g_shunt = k · den`. In the fluctuation-driven regime (R-fluct — the substrate already provides OU
background noise via `ou_seed` and balanced E/I), the read neuron's steady output rate is
`rate_i ≈ num_i / (g_leak + k·den)`, which is the read `num_i / (σ + den)` with the identifications:

- **`σ = g_leak`** — the baseline leak conductance IS the ε. The `ε = 1e-6` in the exact read is not a fudge; it
  is the read neuron's resting leak, a genuine biophysical floor that also guarantees a finite gain when `den → 0`.
- **`k`** — a scale factor from the norm-neuron rate to shunt conductance (a fixed synaptic weight), calibrated
  once so the divisive gain matches the trained read over the operating range (k=1, g_leak=ε recovers the exact
  division at the rate level; the substrate's f–I then applies its own monotone squash — §c).

The division is the SETTLED read-pool rate, not an instantaneous divide: the GABA_A shunt has a decay time
constant (~5–10 ms) and the pool settles over a few ms per token. That settling IS the companion process the
static formula hides (§2, R-net makes it explicit) — the read is a fixed point, at a per-token time cost that is
in-scope (speed is secondary).

### (c) Does it preserve the +0.05 crossing? (the load-bearing test)

At the rate level the shunt read equals `num_i/(σ+den)` to first order (R-fluct), so the crossing is preserved up
to three effects, each a measurable de-risk knob, not a hidden loss:

1. **The σ = g_leak offset.** Choosing `g_leak` small (the ε role) keeps the divisor den-dominated. The arc's own
   "97% of a gap#5 effect was the clamp" lesson applies directly: the GO gate (§4) must confirm the divisor is
   genuinely `den`-driven, not `σ`-dominated (sweep `g_leak`; the read must track `den`, not sit at a fixed gain).
2. **The read-neuron f–I nonlinearity.** The read pool applies a monotone transfer on top of the divisive gain.
   This is not fatal: the downstream head + `FewSpikeWordRead` already apply their own nonlinearity, and the
   cleanest fix is to CALIBRATE / retrain the linattn checkpoint with the read-neuron f–I in the read path (a
   read-in-the-loop retrain, routed to the GPU queue, NOT this doc's to run) so the trained weights absorb the
   squash — the same way the ssm mouth was trained against its own read-out.
3. **Spike-rate quantization.** Finite spike counts per token add noise to `den` and `num`; a small redundant
   norm pool (a few neurons averaging to `den`) and adequate per-token integration time reduce it.

Net: the +0.05 crossing is preserved to the extent the divisive approximation holds — which is exactly the
quantity the GO gate measures. If R-fluct alone does not hold it, R-dend (§d) is the banked next rung.

### (d) Reused on-substrate hooks (grep-confirmed — the primitives already exist)

The substrate ALREADY carries three divisive/shunting primitives; this design reuses them and names the exact gap
each has for the match-mass axis (so a build agent extends, not rebuilds — before_you_build discipline):

| hook | file:line | what it gives | the gap for THIS use |
|---|---|---|---|
| `enable_input_divisive_norm` / `BrainRegion.input_divisive_norm` | `sim/config.py:1121-1125`, `sim/regions.py:278`, applied `sim/bridge.py:9035-9044` | the exact primitive `r_i = x_i / (σ + gain·mean_pool)` — a feedforward Carandini-Heeger divisive-gain circuit, per-neuron-masked, byte-identical when off | pools over the FLAGGED CHANNEL SET (`mean_j x_j`) — the WRONG axis (this is the `--dual-nonneg-divnorm` failure). Extension: let the divisor be an EXTERNAL scalar = the norm-neuron rate `den`, not the flagged-set mean. The `x/(σ+·)` machinery is exactly right; only the pool source changes. |
| `enable_dendritic_divisive_gain` / `dendritic_divisive_sigma` | `sim/config.py:813-824`, applied `sim/bridge.py:8909-8912` | `g = σ/(σ + a)` DENDRITIC divisive gain — the config comment names it "the Carandini-Heeger divisive-normalization the point-neuron soma provably cannot do" (the R-dend genuine-division route) | keyed on each presynaptic SOURCE's own firing EMA `a_i` (a per-source suppression), NOT the query-conditioned pooled `den`. Extension (R-dend rung): key the dendritic gain on the pooled `den` signal (the norm neuron's rate), giving genuine division that clears Holt & Koch by construction. |
| `ssm_k_leak` + `cp_ssm_shunt` (`lam_eff = clip(1 − k_leak·(1+shunt),0,1)`) | `sim/config.py:569-579` | a "graded integrator with shunting-modulated membrane time constant" — an input-driven shunt already modulating a graded state's leak | modulates the STATE leak (`lam`), not the READ gain. Precedent that an input-driven shunt is already on-bridge; the read-pool shunt is the same primitive applied to the read neuron's `g_total`. |

Plus: the OU background (`ou_seed`) + balanced E/I already in the substrate supply the fluctuation-driven regime
R-fluct needs — no new machinery, only the operating point.

### (e) Code-level slot + flags (design spec — NOT an edit; sim/ and runners untouched)

The division lives at exactly two rate-level sites, plus the on-bridge realization:

1. **Rate-level de-risk mode (Tier 1, §4)** — a read-side function, added to a PROBE (not the runner), that swaps
   the exact divide for the shunt-gain form. Sketch:

```python
def divisive_read(num, den, mode="exact", g_leak=1e-6, k=1.0, fI=None):
    """Spike-native num/den realization for the linattn read. mode:
       'exact'  -> num / (den + 1e-6)                      # today's graded host divide
       'shunt'  -> num / (g_leak + k*den)                  # C&H conductance-divisive gain (R-fluct rate model)
    An optional read-neuron f-I (fI) applies the substrate's monotone transfer on top (design (c) effect 2)."""
    g = num / (den + 1e-6) if mode == "exact" else num / (g_leak + k * den)
    return fI(g) if fI is not None else g
```

Slots in at `LinAttnLayer.forward` line ~689 (`read = num / (den + 1e-6)`) and identically in the deployment
`LinAttnReadout.advance_and_logits` (production-wiring DESIGN §3e, line ~309) behind a `--linattn-div
{exact,shunt}` flag (default `exact`, byte-identical when unset). The Tier-1 test needs NO retrain: with
`k=1, g_leak=ε, fI=None`, `shunt` is IDENTICAL to `exact`; the test is robustness to the f–I squash + rate
quantization as `fI` and `k` deviate.

2. **On-bridge realization (Tier 2, §4)** — a `CoreSimConfig` with `enable_input_divisive_norm` (extended to an
   external-scalar divisor) OR `enable_dendritic_divisive_gain` (extended to the pooled `den`), a `D`-neuron read
   region carrying `num` as `g_e`, and one norm neuron carrying `den` as a shunting GABA_A projection, run in the
   OU/balanced regime. The build extends the flagged hook to take the external `den` scalar; everything else is
   existing bridge machinery.

## 4. The cheapest de-risk + GO gate + anti-cheats

Two tiers, cheap-first, both routed to the GPU queue / CPU (NOT an agent; cost-routing). NEITHER is run by this
doc, and neither touches the live verification currently on the GPU.

**Tier 1 (rate model, near-free, CPU, no substrate).** On the ALREADY-TRAINED linattn checkpoints, swap the read
to `divisive_read(..., mode="shunt", ...)` and re-measure `margin_vs_trigram` at d10-99 vs the exact-division
linattn, on the SAME 6 seeds (42/43/44/100/101/102). Sweep `g_leak` (confirm den-dominated, not σ-dominated) and
apply a representative read-neuron f–I as `fI` (confirm the squash does not erase the margin). This is a read-side
swap — no retrain — so it is minutes of CPU.

- **GO gate (Tier 1):** the shunt read preserves `margin_vs_trigram ≥ +0.03` (keeps most of the +0.05 crossing),
  6/6 seeds, with the built-in anti-cheats STILL clean: `wkv_memoryless − wkv > 0.05` and `wkv_perm − wkv > 0.05`
  (the read still uses long-range, order-dependent content — the division swap must not manufacture a shortcut,
  nor collapse toward the norm-OFF +0.190 level).
- If Tier 1 needs the read-in-the-loop retrain to hold the margin (f–I squash costs too much raw), that retrain
  (linattn with the read-neuron f–I in the read path) is the banked next step — GPU queue, still not an agent.

**Tier 2 (on-bridge, the real spiking test).** Instantiate the §3e circuit on the bridge (a `D`-neuron read pool
+ 1 norm neuron, GABA_A shunt, OU/balanced regime), drive it with the checkpoint's `q/k/v` rate codes for a batch
of contexts, and compare the read pool's settled output rate to the exact `num/(den+ε)`. Trivial VRAM
(D≈192 + 1 neurons) — well within the single-3090 consumer reference — but scoped to the GPU queue, sequenced
AFTER the live verification frees the GPU.

- **GO gate (Tier 2):** the on-bridge read preserves `margin_vs_trigram ≥ +0.03` 6/6 vs the exact-division
  linattn, AND — the anti-cheat SPECIFIC to this design — the shunt is measurably DIVISIVE, not subtractive:
  measure the read neuron's f–I as a function of `den` and confirm the GAIN (slope) scales with `1/(σ+k·den)`
  rather than the OFFSET (rheobase) shifting. A slope-scaling result clears Holt & Koch on our substrate; an
  offset-shifting result IS the honest negative (§5) and triggers R-dend/R-net.

**Anti-cheats (both tiers), beyond the built-in memoryless/perm:**
- **Divisive-not-subtractive check** (Tier 2 GO gate above) — the load-bearing one; directly tests the Holt & Koch
  failure.
- **σ-domination check** — sweep `g_leak`; the read must track `den` (a den-driven divisor), not sit at a
  den-independent fixed gain (the "the clamp owned 97% of the effect" trap).
- **Like-for-like** — the shunt read is compared to the exact-division linattn at the IDENTICAL config/seeds/
  checkpoints, at the same deepest bucket, with the same trigram baseline.

## 5. The honest brain-based-purity residual (a first-class deliverable if genuine division isn't fully reached)

Per the BRAIN-BASED-ONLY standard, a documented limit is the deliverable — not a stopping point, a MAP of what the
substrate can/can't do, with the next rung named:

1. **If R-fluct (somatic shunt in the fluctuation-driven regime) holds** (Tier 2 slope-scaling passes): the
   division is spike-native — the LAST graded host op in the linattn mouth is closed, the read is neurons +
   synapses end to end (graded fast-weight state, as everywhere in the spiking-LM literature; genuine spiking
   read-out). This is the target.
2. **If R-fluct is only partially divisive** (some residual subtractive component — the likely Holt & Koch
   outcome for a plain point-neuron soma): the honest residual is QUANTIFIED — "the somatic shunt recovers X% of
   the +0.05 margin; the residual is subtractive (Holt & Koch 1997 confirmed on our substrate)". The banked next
   rung is **R-dend** (the dendritic divisive gain, `enable_dendritic_divisive_gain` re-keyed on the pooled `den`
   — genuine division by construction), then **R-net** (SSN, division as the settled state of a recurrent E/I read
   pool). No capability is abandoned; a method is banked and the next biological method taken (THE LAW).
3. **The recurrent WRITE learning rule and the state's graded-vs-spiking status** are the SAME residual the
   shipped ssm mouth already carries (BPTT-trained weights; graded fast-weight state) — not new to this design,
   and out of this doc's scope.

The residual this doc CLOSES is the num/den division's realization; the residual it may DISCLOSE (if only R-dend/
R-net reach genuine division) is which of the three routes the substrate actually needs — itself a mapped result.

## 6. Failure modes (and the banked response for each)

1. **Somatic shunt is subtractive (Holt & Koch)** — Tier 2 shows offset-shift, not slope-scale. Banked: R-dend
   (dendritic divisive gain on the pooled `den`), which divides before somatic spike generation by construction;
   then R-net (SSN). Named in §5(2).
2. **f–I squash erases the margin** (Tier 1: the read-neuron transfer costs too much). Banked: the read-in-the-
   loop retrain (linattn trained with the read-neuron f–I in the read path), GPU queue.
3. **σ dominates the divisor** (the "clamp owned the effect" trap). Caught by the σ-domination anti-cheat (§4);
   fix is a smaller `g_leak`, or normalize the operating range so `den` spans the divisor.
4. **Rate quantization noise on `den`/`num`.** Banked: a small redundant norm pool + adequate per-token
   integration time; a perf/faithfulness knob, not a wall (speed secondary).
5. **The division swap manufactures a shortcut** (memoryless/perm anti-cheats regress). Then the shunt read is
   not the same computation — reject and diagnose; the like-for-like comparison isolates it.

## 7. Provenance

Code read this session (2026-09-03): `research/runners/_emerge_wkv_lm_derisk.py` (`LinAttnLayer` ~L574-692, the
read `read = num / (den + 1e-6)` ~L689); `sim/bridge.py` (the input-divisive-norm application ~L9035-9044, the
dendritic-divisive-gain application ~L8909-8912); `sim/regions.py` (`input_divisive_norm` ~L278); `sim/config.py`
(`enable_input_divisive_norm`/`input_divisive_sigma`/`input_divisive_gain` ~L1121-1125,
`enable_dendritic_divisive_gain`/`dendritic_divisive_sigma` ~L813-824, `ssm_k_leak`+shunt ~L569-579). Findings
cited: the milestone (`2026-09-03-OPEN-FLUENCY-BREAKTHROUGH-linattn-deployable-spiking-mouth-beats-trigram-6of6.md`,
commit 43c5b6b4), the mechanism DESIGN (`2026-09-03-spiking-content-addressable-read-DESIGN.md`, §3/§4 the axis
fix + biology anchors), the production-wiring DESIGN (`2026-09-03-linattn-production-mouth-wiring-DESIGN.md`, §3e
`LinAttnReadout` + §5 the CPU ablation + §7 the disclosed residual). Measurements read directly: the linattn CPU
smokes `research/findings/raw/_linattn_smoke_{normON,normOFF}.json` (norm-ON +0.456 / norm-OFF +0.190) and the
6-seed milestone artifact `research/findings/raw/_emerge_wkv_lm_linattn_depth2_contiguous_6seed.json` (mean
+0.0505, 6/6). External biophysics verified via live web fetch this session (2026-09-03): Chance, Abbott & Reyes
2002 *Neuron* 35:773-782; Mitchell & Silver 2003 *Neuron* 38:433-445; Rubin, Van Hooser & Miller 2015 *Neuron*
85:402-417. In-repo anchors reused (verified in the two prior linattn DESIGN docs): Carandini & Heeger 2012
*Nat Rev Neurosci* 13:51-62 (doi:10.1038/nrn3136) + 1994 *Science* 264:1333-1336; Holt & Koch 1997 *Neural
Comput.* 9:1001-1013; Silver 2010 *Nat Rev Neurosci* 11:474-489; Mongillo, Barak & Tsodyks 2008 *Science*
319:1543-1546 (doi:10.1126/science.1150769); Ba et al. 2016 arXiv:1610.06258. This doc edits no `sim/` file, no
runner, and no GPU state.
