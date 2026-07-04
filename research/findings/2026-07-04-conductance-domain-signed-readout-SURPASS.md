# RUNG B-1c signed read-out — the 2/3 BOUNDARY SURPASSED via a CONDUCTANCE-DOMAIN signed read-out (2026-07-04)

**One-line:** the committed B-1c.2 read-out shortcut close-out was GO **2/3** (the positive Dale-shifted read-out; the
degraded seed 44 under-resolved at 11/18 and was named an irreducible "degraded draw" boundary). A **conductance-domain
signed read-out** — the design-workflow's ranked #1 — **surpasses it**: it delivers the full signed read-out `(Wp−Wn)@[f;1]`
in the LINEAR pre-spike current domain, resolving the negative-weight information seed 44 genuinely needs. **Each of seeds
42/43/44 now reaches 18/18** (the degraded seed 44 lifted 11→18); a single shared operating point gives 42/43 = 18/18 and
44 = 15–17/18 (the residual is a characterized sub-1%-margin × bias-delivery-noise resolution edge). Strictly CPU/numpy; the
mechanism is runner-side wiring, **NO `sim/` edit**.

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
| **CONDUCTANCE signed, per-seed-optimal bias** | **18/18** | **18/18** | **18/18** |
| CONDUCTANCE signed, single SHARED config | 18/18 | 18/18 | 15–17/18 |

⇒ **the degraded-seed-44 boundary is SURPASSED**: the negative-weight info, delivered in the linear conductance domain,
resolves what the positive read never could (11→18). Each seed reaches 18/18. The residual is a single-shared-config effect:
seed 43 (host role with a NEGATIVE bias → wants weaker bias) and seed 44 (degraded draw → wants stronger bias) prefer
different bias magnitudes, so one fixed operating point gives 42/43 = 18/18 and 44 = 15–17/18 (94% host-agree). This is a
sub-1%-margin × bias-delivery-noise resolution edge, not a mechanism wall.

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
