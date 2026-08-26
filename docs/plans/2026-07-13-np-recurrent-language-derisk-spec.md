---
type: plan
status: live
date: 2026-07-13
---

# NP-toward-recurrent-language de-risk SPEC (design + adversarial-verify workflow wf_f69e1f86-fb5, 2026-07-13)

The confound-checked cheap-first spec from an 8-agent research+design+adversarial-verification workflow. Key contribution: the adversarial phase caught that the naive "NP learns W_in on a reservoir with zero-input fillers" is a STATIC 2-layer problem in a recurrent costume (W_in enters at ONE step) -> uninterpretable; redesigned to an ORDER-GATED DELAYED next-class task with a mandatory CEILING-FIRST rung-0 gate. Implemented in research/runners/_reslm_np_learn_win_gated_derisk.py.

---

I've read all three fork-target source files and confirmed every load-bearing fact the critiques rest on:

- `build_stream:73` — `next_class = (c+1)%G` is a **memoryless function of the cue's class alone** (critique 1's core).
- `_fwd:102` — `x = codes[tok] if t==0 else zeros`; `b = np.zeros(n)` at `:100`; noise drawn from an advancing `trng` at `:110`; ridge `lam=1.0` at `:136` with **no state normalization** in `_reads` (critiques 2 and 3's cores).
- `train_nodepert:70` — antithetic-k estimator perturbs **node states** (pre-activations), so the spec's weight-perturbation-of-`W_in` is a genuine port, not a copy (matters for critique 2's variance argument).
- `_ssm..._run_seq:39` — `tanh(A@x+W_in@u)` (nonlinear ESN) vs `A*x+W_in@u` (linear) is a **one-line swap on the same `A`** → critique 3's clean linear-vs-nonlinear control; `_decode_acc:81` already z-scores states (the normalization critique 3 wants).

All three critiques converge on ONE task redesign plus three instrumentation fixes. Here is the final spec.

---

# FINAL DE-RISK SPEC — "NP learns `W_in` through genuine recurrent computation" (order-gated delayed classification)

## (1) VERDICT

**Yes — this is the right highest-leverage next rung, but ONLY after the task is redesigned as the critiques demand.** The original spec's instrument is **not cleanly interpretable**: with zero-input fillers `W_in` enters at exactly one timestep, so the learnable object factorizes into `read = Wout·F(W_in·code)` — a *static* 2-layer input-credit problem (already validated by XOR-inheritance) wearing a recurrent costume. A GO on that instrument certifies "NP learns a 1-step input rep behind a passive delay line," which is exactly the feedforward regime R3 already established and says nothing about the rung-3 language claim (credit for effects that must **combine across timesteps through nonlinear recurrence**). The three anti-cheats the original spec relied on (recurrence-lesion, derangement, persists-with-dist) cannot distinguish that static problem from the real one, and — worse — its `dist`-dependent NP criteria would manufacture a **false NO-GO** from an estimator-variance artifact.

The redesign makes the instrument interpretable by construction: an **order-sensitive, nonlinear, delayed** target forces both load-bearing axes (learned input representation AND recurrent nonlinear binding), and three empirical control arms (bag, linear-reservoir, oracle-anchored gap) *arbitrate* rather than *assume* that both axes carry the credit. It stays cheap (numpy, one forked runner, one-line reservoir swaps), single-variable (rule-on-identical-scaffold), and sidesteps the CEILING scale-confound entirely by deferring the scale-valid LM to rung 3. **This is a necessary-not-sufficient rung: a GO validates the NP *mechanism* on a synthetic 2-cue task; it does not yet validate language.**

## (2) THE EXACT TASK — "order-gated delayed next-class prediction"

Confound-free because BOTH axes are manufactured and each is proven load-bearing by a lesion arm (not argued).

**Codes (reuse `build_codes` verbatim).** `G` classes × `syn` synonyms → `V` tokens. Each token = its class's shared `sf`-block (the class feature) + `idn` identity-confound dims from a shared `id_pool` (the realistic overlapping confound). Held-out convention reused: the last synonym of each class (`held[c]=c*syn+(syn-1)`) is excluded from training and appears only at eval.

**Sequence (per example).** Two cues drawn from the **shared** token vocabulary, presented in a fixed order with input-bearing fillers between them:
```
pos 0        : token P     (role = GATE, by position)
pos 1..d     : distractor tokens (input-bearing, disjoint distractor vocab)
pos d+1      : token Q     (role = CONTENT, by position)
pos d+2      : blank read step  -> read h_read here
```
`d` = the delay (sweep `dist ∈ {0,3,8}`; `dist` ≡ `d`).

**Target (order-sensitive + nonlinear + delayed).**
```
gP     = class(P) mod 2                      # binary gate carried by the FIRST cue's shared block
target = class(Q)          if gP == 0
         (class(Q)+1) % G  if gP == 1        # the earlier gate conditionally shifts the later content
```
Read a `G`-way logit at the blank step.

**Why this is THE interpretable instrument (each critique's guard baked in):**
- **Order-sensitivity defeats the bag by construction (critique 1, hardened).** The bag control is order-invariant (a commutative sum). The gate is read from *position 0* and the content from *position d+1*; swapping P and Q generally changes `target`. So a bag — *regardless of how wide or well-learned its `W_in` is* (the loophole in the original "XOR" framing: a wide learned bag is a 1-hidden-layer MLP and *can* do unordered XOR) — is provably below the reservoir on an **order-probe set** (same two tokens, both orders, different targets). This is stronger than XOR-of-two-cues and closes the exact hole in critique 1's own proposed fix.
- **Nonlinear gate×content interaction defeats the linear reservoir (critique 3).** A linear reservoir makes `h_read` a time-weighted *linear* function of the input history; a linear readout of it cannot represent the conditional shift `gP ? shift(class(Q)) : class(Q)` (a multiplicative gate). Only a nonlinear reservoir can hold `gP` across the delay and gate `class(Q)` on arrival.
- **The delay makes recurrent RETENTION genuinely load-bearing, not a passive buffer.** `gP` must survive `d` fillers *and then be combined with* `Q` — the reservoir binds across time; it is not buffering a single vector the readout inverts.
- **learn-`W_in` headroom (the R3 lever)** comes from reading `gP` (parity of P's class) and `class(Q)` through the shared blocks despite the identity confound + held-out rare synonyms — the exact regime `_reslm_generalize_rate_check` already shows has headroom (learn 0.883 vs fixed 0.694 at the bottleneck).

**The forward (the fork of `_fwd`, deterministic — `noise=0.0` pre-registered so ±ξ NP evals are exactly common-random; see critique 2):**
```python
def _fwd(P, Q, fillers, dist, W_in, recur="tanh"):   # recur in {"tanh","linear"} = the critique-3 control swap
    h = np.zeros(n); T = dist + 3                     # 0:gate  1..dist:fillers  dist+1:content  dist+2:blank read
    for t in range(T):
        if   t == 0:        x = codes[P]
        elif t == dist + 1: x = codes[Q]
        elif t <= dist:     x = dcodes[fillers[t-1]]  # input-bearing fillers (disjoint distractor vocab)
        else:               x = np.zeros(m)           # blank read step only
        pre = W_rec @ h + W_in @ x + b
        act = pre if recur == "linear" else np.tanh(pre)   # ONE-LINE swap = the linear-reservoir control
        h   = (1 - alpha) * h + alpha * act
        if t == T - 1: h_read = h.copy()
    return h_read
```
**Readout normalization (critique 3, applied to EVERY arm uniformly → single-variable preserved).** Fit the ridge readout on **per-dim z-scored** read-states (reuse `_ssm..._decode_acc`'s `Xn=(Xtr-m)/s` pattern), NOT on the raw `[h_read,1]`. This removes the global magnitude shrink so the recurrence-lesion (`W_rec→0`, which with `b=0` only scales `h` by `(1-alpha)^dist`) cannot fake a "collapse" via the ridge floor — the lesion then collapses only for the RIGHT reason (a memoryless reservoir cannot bind `gP` to `Q`).

## (3) WHAT NP TRAINS · ARMS · ANTI-CHEATS · DECISIVE METRIC

**What NP trains: `W_in` ONLY.** `W_rec` FROZEN (R3: frozen-`W_rec` + learned-`W_in` beats full BPTT; training `W_rec` destabilizes). The `G`-way readout is re-fit by the SAME clean ridge/delta rule each outer epoch for EVERY arm, so the single variable is genuinely the `W_in` credit rule.

**NP update — WEIGHT-perturbation of `W_in` (T-independent perturbed dimension `N×m`), antithetic-k, CRN, running-mean baseline** (port of `train_nodepert`'s estimator to the sequential forward):
```python
for i in range(k):                                     # k = antithetic resamples (raise until dist-fidelity matched; see GO)
    xi = sigma * rng.standard_normal(W_in.shape)
    Lp = ce(readout(_fwd(P,Q,fill,dist, W_in+xi)), y)  # +xi and -xi share ONE deterministic forward => exact CRN
    Lm = ce(readout(_fwd(P,Q,fill,dist, W_in-xi)), y)
    dL = 0.5*((Lp-bl) - (Lm-bl))                        # bl = running-mean loss baseline (Miconi/Williams)
    g += (dL / sigma**2) * xi
W_in -= lr_in * g / k                                    # W_rec, Wout, b UNTOUCHED  (assert ||dW_rec||==0)
```

**Arms (single-variable swap: the `W_in` credit rule; same fixed `W_rec`, same normalized readout, same task, same seeds):**
| # | Arm | Reservoir | Role |
|---|---|---|---|
| 1 | **chance** | — | `1/G` floor |
| 2 | **frozen-`W_in`** | nonlinear | the R3 fixed-reservoir floor (`learn=False` / NP `hidden_frozen`) |
| 3 | **NP-`W_in`** | nonlinear | **mechanism under test** (no feedback matrix, no weight transport) |
| 4 | **KP/e-prop-`W_in`** | nonlinear | the FA plateau reference (`rate_ref_generalize` `learn=True`: input eligibility `e_in` + fixed-random `Bfb`) |
| 5 | **oracle-`W_in`** | nonlinear | ceiling = backprop through the FROZEN reservoir; **its analytic `g*` anchors all `dist` metrics** |
| 6 | **linear-reservoir {frozen, NP, oracle}** | linear | critique-3 control (`recur="linear"`, same `W_rec`) — the recurrent-COMPUTATION arbiter |
| 7 | **bag {frozen, NP, oracle}** | none | critique-1 control (sum per-step `W_in@x` → one tanh → normalized readout) + an **order-probe bag** |
| 8 | **node-perturb-STATES** (secondary, reported not gated) | nonlinear | perturbs `h_t` per step (dim `N×T`) → must degrade with `dist` while NP-`W_in` stays flat |

**Anti-cheats (all established patterns; each proves an axis, not assumes it):**
- **shuffle-dL** → collapses to ≈ frozen (credit rides the real global signal). *Required.*
- **wrong-sign dL** → anti-learns (< frozen). *Required.*
- **linear-reservoir control (PROMOTED to primary)** → the nonlinear arm must beat it at `dist>0`, growing with `dist` (recurrent *computation*, not fading memory).
- **bag control + order-probe (PROMOTED, no longer "N/A")** → the recurrent arm must beat the best bag by ≥+0.10, and the order-probe bag must sit at ≈ chance.
- **recurrence-lesion (`W_rec→memoryless`, DEMOTED, readout-normalized)** → collapses the delayed decode for the RIGHT reason; kept as a sanity, not the load-bearing recurrence proof.
- **shared-class derangement** (permute synonym↔class) → collapses learn's advantage over frozen (the STRUCTURE is the lever).
- **input-lesion→chance** and **label-scramble→chance** (the read is from real state).
- **`‖ΔW_rec‖==0` assertion** across NP/oracle (no weight transport, frozen-recurrence core).
- **held-out generalization is the metric** — train acc matched across arms; only `evl` separates them.

**Decisive metric (oracle-anchored, per critique 2):** held-out gated-classification accuracy on rare synonyms, reported as
1. `margin_over_frozen = NP − frozen`,
2. **`gap_to_oracle(dist) = oracle − NP` vs `dist`** (gate on the GAP, not NP's absolute dist-curve — the oracle curve absorbs the reservoir's own retention loss),
3. `margin_over_BAG` and `margin_over_LINEAR` (the two computation-arbiter margins),
4. **`cos(ĝ_NP, g*)` and `cos(ĝ_KP, g*)` vs `dist`** — the estimator-fidelity diagnostic that converts a `dist`-dependent NO-GO from "mechanism failure" into "raise k."

## (4) REUSE-BY-IMPORT ENTRY POINTS

New runner: **`research/runners/_reslm_np_learn_win_gated_derisk.py`** (numpy / off-bridge / NO `sim/` edit).
- Fork `_reslm_generalize_rate_check.py`: `build_codes:43` (verbatim), the held-out convention from `build_stream:73` (`held[c]=c*syn+(syn-1)`; **replace the stream body** with the two-cue order-gated generator), `rate_ref_generalize:93` as the template (its fixed `W_rec` sr-0.95, the `learn=True` input-eligibility+`Bfb` path = arm 4, `learn=False` = arm 2), `_fit_ridge:33`/`_decode_acc:39` (readout) — but **add per-dim z-scoring** to the readout (from `_ssm`).
- `_nodepert_deep_credit_derisk.py`: `train_nodepert:70` antithetic-k estimator + modes `np`/`shuffle_dl`/`wrong_sign`/`hidden_frozen` (port to the weight-perturbation-of-`W_in` update above); `_softmax`/`_ce`/`_acc`.
- `_ssm_fixed_structured_reservoir_derisk.py`: `_run_seq:39` for the `tanh`↔`identity` recurrence swap (arm 6); `_decode_acc:81` z-scoring pattern; `_build_A:50` `fastonly` for the recurrence-lesion.
- **Escalation only** (rung 3): `_emerge_reservoir_lm_derisk.py` (`Vocab`, `train_readout`, `_bag_cache`, `fit_bigram`), `_reslm_batched_reservoir_derisk.per_token_states_batch` (67× enabler), `_ssm_context_depth_derisk.py` (by-depth buckets).

## (5) GO GATE + MULTI-SEED PLAN (CEILING-FIRST)

**Rung 0 — mandatory, cheap, run BEFORE any NP** (oracle/frozen/linear/bag only). ALL must hold at the chosen `dist`, on dev seeds:
- `oracle(nonlinear) − frozen(nonlinear) ≥ +0.10` held-out AND `frozen ≤ 0.90` (learn-`W_in` headroom).
- `oracle(nonlinear) − oracle(linear) ≥ +0.10` at `dist>0`, **growing with `dist`** (nonlinear recurrent computation load-bearing — critique 3).
- `oracle(nonlinear,recurrent) − best-bag(oracle-`W_in`) ≥ +0.10` AND **order-probe bag ≈ chance** (temporal binding, not bag-fakeable — critique 1).
- recurrence-lesion collapses the oracle margin at `dist>0` with **normalized** readout.
- **If any sub-gate fails → STOP and re-tune the regime** (target nonlinearity/order-dependence, `sf`/`idn`/`id_pool`/`n`, `d`) on dev seeds. Do NOT run NP in a mis-specified regime (the CEILING lesson: an expensive arm in a null regime is uninterpretable). Seed the regime near the known-headroom bottleneck: `G=8, syn=6 (5 train + 1 held-out), sf=3, idn=10, id_pool≈40, n≈60 (n<m)`.

**Rung 1 — NP GO (oracle-anchored). ALL must hold:**
1. `NP − frozen ≥ +0.10` on **≥5/6 seeds**;
2. **`gap_to_oracle(dist)` does NOT widen with `dist`** beyond the k-fidelity tolerance, AND `cos(ĝ_NP,g*)` at max `dist` ≥ its `dist=0` value − small tol (fidelity is `dist`-matched by **raising k**, not by the mechanism failing — the fix that prevents critique 2's false NO-GO);
3. shuffle-dL collapses to ≈ frozen AND wrong-sign anti-learns;
4. `NP` beats BOTH best-bag AND linear-reservoir-NP by ≥+0.10 (temporal + nonlinear recurrent credit, not bag/linear-fakeable);
5. `NP ≥ KP` rate-reference plateau, **credited only when `cos(ĝ_NP,g*)` and `cos(ĝ_KP,g*)` are both ≥ a pre-registered floor** (a variance-matched mechanism comparison, not estimator-class vs estimator-class).

**Stretch GO (headline):** NP closes part of the oracle−frozen residual the FA family cannot — `NP margin > KP margin` on **≥4/6 seeds**, under the same cosine-floor fairness condition. This is the fresh-mechanism claim: a rule with **no backward channel** beating the feedback-alignment plateau that the whole FA/burst family (and spiking-BDSP linear-`W_in`) plateaued at.

**Secondary (reported, not gated):** node-perturb-STATES (dim `N×T`) held-out **degrades with `dist`** while NP-`W_in` (dim `N×m`) stays flat, with `cos(·,g*)` confirming it is the variance∝T effect (Züge) — direct evidence for "train `W_in`, freeze `W_rec`."

**Multi-seed / pre-registration:**
- **Dev = 42/43/44:** build the fork, tune the regime + NP hypers (`sigma∈[0.01,0.05]`, `lr_in`, `lr_out`, `epochs` few-hundred–2k, `n_seq` ~2–4k, `alpha=0.3`, `k` **raised until `cos(ĝ_NP,g*)` is `dist`-matched**), pass rung-0, **lock config + exact thresholds + the frozen `k`.**
- **FREEZE.** No config/threshold/`k` change after this.
- **Blind = 100/101/102:** run all arms + all anti-cheats untouched; report rung-1 + stretch. GO needs the ≥5/6 criteria across all six; report dev and blind separately.

## (6) RESIDUAL RISK (honest)

- **Thin nonlinear advantage at small numpy `n`.** If the nonlinear reservoir's edge over the linear control is marginal at achievable `n`, rung-0 fails → bigger `n` or stronger target nonlinearity, at more compute. If it can't be made to pass cheaply, that is itself an informative instrument NO-GO (documented, not forced).
- **k could blow up at large `dist`.** The "raise k until fidelity-matched" rule can demand large `k` at `dist=8`. If it does, that IS the honest "NP is high-variance at depth" property — reported as a finding; the oracle-gap framing keeps it interpretable rather than a false mechanism NO-GO. (Weight-perturbing `W_in` is T-independent in perturbed *dimension*, but the earliest cue — the gate — is still the most-delayed, so the *true-gradient magnitude* through it still shrinks; the oracle anchor is what keeps this from being read as a mechanism failure.)
- **Scope.** A rung-1 GO validates NP assigning `W_in` credit through genuine recurrent nonlinear computation on a SYNTHETIC 2-cue task — necessary, not sufficient, for the language claim. Rung 2 (deep nonlinear `W_in` encoder, NP's depth-non-degradation) and rung 3 (scale-valid LM, scored by **`margin_over_BAG` bucketed BY DEPTH, required to GROW with depth**, ceiling-first, never margin-over-bigram) remain gated behind it. Stated plainly so a synthetic GO is not over-read as "green-light the recurrent language cortex."
- **Order-gate target is a design choice, not sacred.** It is the cheapest construction that is provably order-sensitive (bag-proof) AND nonlinear-across-time (linear-reservoir-proof); the empirical rung-0 gate — not the hand-argument — is the arbiter, and the target's nonlinearity/order-dependence is the first dev-phase knob if rung-0 underperforms.