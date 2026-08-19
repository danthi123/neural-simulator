---
status: live
type: finding
lane: gap#4
date: 2026-08-19
claim_check: measured
---

# Fluid mouth read-out — substrate-forward e-prop at 40k (5×) coverage STAYS at the 8k plateau: coverage EXCLUDED, the few-spike read-SNR is the real credit limit — NO-GO (coverage confound-exclusion done)

**Date:** 2026-08-19 · **Type:** de-risk finding (research), confound-exclusion. · **Lane:** gap#4 (deep-credit-on-spikes) / A1 mouth read-out. · **Verdict:** **NO-GO** on the coverage hypothesis — raising the training coverage 5× does NOT lift substrate-forward recovery; the bottleneck is the substrate forward's few-spike read (multiplexing SNR), exactly as the gap#4 arc predicted. Terminology checked against `docs/TERMS.md` (this is a NO-GO / boundary-mapping finding, not a positive GO).
**Reads AFTER:** [`2026-08-17-mouth-readout-eprop-batched-substrate-forward-PARTIAL.md`](2026-08-17-mouth-readout-eprop-batched-substrate-forward-PARTIAL.md) (the 8k-coverage predecessor, `sub_learned_recov_mean` 0.371, 6-seed) and the plan-of-record [`docs/plans/2026-08-19-deep-credit-plan-of-record.md`](../../docs/plans/2026-08-19-deep-credit-plan-of-record.md) (commit 961fd7aa; §5 "NEW confirming evidence"). This is the named coverage confound-exclusion continuation of that arc, NOT a re-derivation — the corpus check (`tools/before_you_build.sh`, recorded) surfaced the 8k predecessor as its top hit.
**Runner:** [`research/runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py`](../runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py) (additive `--forward {substrate,host_proxy}` + `--eval-every-epochs`, committed 2f9b6e6b3).
**Artifacts (this finding):**
- `research/findings/raw/_wkv_readout_eprop_substrate_coverage40k_3seed.json` — the decisive substrate-forward run (`--forward substrate`, `host_matmul_on_forward_max` 0).
- `research/findings/raw/_wkv_readout_eprop_proxy_coverage40k_3seed_control.json` — the matched-coverage host-linear-proxy instrument control (`--forward host_proxy`, `host_matmul_on_forward_max` 24990 by design).
- `research/findings/raw/_wkv_readout_eprop_batched_substrate_marginclean_6seed.json` — the 8k-coverage baseline (already on main; `sub_learned_recov_mean` 0.371).

**Scope flags:** runner-only, additive, default-off, **NO `sim/` edit**. cfg.seed-controlled. cupy/GPU (RTX 3090). Substrate run: 3 seeds 42/43/44 (a NO-GO corroborating the 6-seed 8k baseline; see the honesty note on seed count in §4).

## Headline (honest)

<!--derived-->

At **40 000** held-out training positions (5× the 8 000 of the 8k baseline), the e-prop-through-the-actual-spiking-substrate FORWARD read-out recovers `sub_learned_recov_mean` **0.3403** of the perfect-argmax mass (go 0/3), essentially UNCHANGED from the 8k baseline's 0.371 — coverage bought nothing. The whole point is `forward_is_substrate_all` True with `host_matmul_on_forward_max` **0** (a falsifiable count: every learning-forward gradient step is a substrate read, not a host-linear proxy). Meanwhile the **matched-coverage** host-linear-proxy control (identical 40k positions, epochs, lr, decay) reaches `sub_learned_recov_mean` **0.8593** with host-linear recovery **0.8996** — so at the SAME coverage a host-linear forward learns the head and the substrate forward does not. Therefore **coverage is EXCLUDED** as the cause; the residual is the few-spike READ of the substrate forward (its multiplexing signal-to-noise), not the amount of data, not the feedback rule, and not the read of a good matrix (the copied head reads back at **0.9734**). This CLOSES the coverage arm of the mouth confound-exclusion (board #37 / #80).

## 1. The comparison that excludes coverage

Three arms at IDENTICAL coverage (40 000 train positions, 30 epochs, B=48, lr 0.3, weight-decay 8e-4, w_target 40, shuffle-frac 0.05, `eval-every-epochs` 3). The only variable is the FORWARD used to compute the per-output error that drives the local three-factor rule.

| arm | forward | `sub_learned_recov_mean` | `hostlinear_recov_mean` | `host_matmul_on_forward_max` | go |
|---|---|---|---|---|---|
| substrate (decisive) | ACTUAL spiking substrate read | **0.3403** | 0.3631 | **0** | 0/3 |
| host-linear proxy (control) | `W@h + head_b` host matmul | **0.8593** | 0.8996 | 24990 | 0/3 |
| 8k baseline (predecessor) | ACTUAL spiking substrate read | 0.371 (6-seed) | — | 0 | 0/6 |

<!--derived-->

The reading: the substrate arm and the proxy arm are the SAME experiment except for whether the forward margin is read off the spiking substrate (`cp_conductance_g_e/g_i`) or computed by a host matmul. The proxy learns the head to ~0.86–0.90; the substrate does not move off the ~0.34 plateau it already sat at with 5× less data. So the variable that separates success from failure is NOT coverage (held identical) — it is the read. And the substrate arm at 40k is statistically indistinguishable from the 8k baseline at 0.371, so the extra 32 000 positions changed nothing.

## 2. What the wall IS (few-spike read-SNR), and what it is NOT

**Is:** the substrate FORWARD margin is a finite-spike-count read of graded synaptic current over a short window; its per-position SNR is low, so the softmax error that drives the local rule is noisy and the learned `W_hat` converges to a WEAK approximation of the target head. The host-side read of that learned matrix confirms it is genuinely weak, not a read artifact: `weight_cosine_mean` to `head_w` is **0.1363** (vs the proxy control's 0.49), and the host-linear recovery of the same `W_hat` is only 0.3631 (vs the proxy's 0.8996). The matrix itself is a poor copy of the target — the substrate-noise forward could not teach it better. This is the same signature the 2026-08-11 gap#4 arc-summary located as learning-rate-invariant on the production Izhikevich bridge (fixed-DFA 0/6, KP 0/6, even a perfect Wᵀ oracle failed): the read regime, not the rule.

**Is NOT:**
- **NOT coverage** — 5× coverage (this finding) leaves it at the 8k plateau; the matched-coverage proxy reaches 0.86.
- **NOT the read window / integration time** — the mouth read-window lever (120→360) was tested this session and did NOT move it (recorded in the plan-of-record §5 DO-NOT-REPEAT); integration-time is excluded too.
- **NOT the feedback-alignment family** — DFA/KP-as-fixed/DRTP are exhausted-negative on this substrate (plan-of-record §2, 2026-07-12).
- **NOT the read of a good matrix** — the copied target head reads back at `sub_copied_recov_mean` **0.9734**; the substrate demo read is faithful for a good `W`. The ceiling on READING is ~0.97; the wall is on LEARNING a good `W` through the substrate forward.

## 3. Anti-cheats clean (the substrate arm)

`anticheats_collapse_count` 3/3. The shuffled-teacher `W` collapses to `sub_shuffle_recov_mean` **0.0018** (so the substrate metric is discriminative, not a frequency-tie-break confound), and the host-linear floor of a null/floor read is `hostlinear_floor_recov_max` **0.0537** (a floor read of a bad matrix recovers near zero). Each candidate `W` (learned / copied / shuffle / floor) is read on its OWN fresh reseeded bridge, so learned/copied/shuffle see identical OU noise (a fair A/B) and one large-`||W||` read cannot poison the next (the reused-bridge corruption fixed in the 2026-08-17 predecessor). `forward_is_substrate_all` True and `host_matmul_on_forward_max` 0 confirm the learning forward never fell back to a host matmul — the NO-GO is a genuine substrate result, not a mis-wired proxy.

## 4. Two honesty notes (process + the failed pre-check)

**(a) This verdict was mis-called TWICE mid-run from a shared interleaved log before the run's OWN file settled it.** `gpu_queue.log` interleaves stdout from concurrent jobs. Early in the run a `hostlinear` trajectory line was read as the substrate result; later the PROXY control's `sub_learned_recov_mean` 0.8593 line (which carries `host_matmul_on_forward_max` 24990 — a host-linear forward BY DESIGN) was read as if it were the substrate arm, briefly suggesting a GO. Both were wrong. The authoritative read is the substrate run's OWN output file (`_wkv_readout_eprop_substrate_coverage40k_3seed.json`, `host_matmul_on_forward_max` 0), which reads 0.3403. **Lesson (recorded here so it is not re-learned): read the run's own output file, keyed by its `host_matmul_on_forward` count, NOT a shared interleaved queue log.**

**(b) `verify_first_all_ok` is False.** One of the three seeds failed the in-run verify-first pre-check (8 substrate-forward updates on one held 48-position batch must drop its CE) — the same B=48 noise-limited pre-check that failed on seed-100 in the 6-seed margin-clean baseline. It does NOT rescue the arm: all three seeds land at ~0.34 regardless (`sub_recov_ratio_mean` **0.3496**, min 0.29), so the NO-GO holds whether or not the pre-check passed. Disclosed because a failed pre-check is exactly the kind of thing a headline should not bury. This is why the run is reported at 3 seeds as a NO-GO corroborating the 6-seed 0.371 baseline, not as a new positive generalization (which would demand 6 seeds).

## 5. Consequence — Option 2, per the plan-of-record

This is the last read-regime datapoint the plan-of-record ([`docs/plans/2026-08-19-deep-credit-plan-of-record.md`](../../docs/plans/2026-08-19-deep-credit-plan-of-record.md)) waited on. With coverage AND read-window both excluded and the read-SNR isolated as the wall, the owner-delegated decision is **Option 2: accept the scaffold-bridge** for the mouth (speak-with-own-neurons), keep the crux on the conversation frontier (drive-couplings, memory), and leave deep-credit a documented, mapped boundary with ONE recorded open lever (a read-SNR manipulation that is NOT integration-window: higher gain / ensemble read / dendritic multi-compartment read — the BurstCCN "two mechanisms our port lacks"). **No further deep-credit compute is queued.** The board #80 speak-with-own-neurons task is NOT done — the scaffold bridges it — but its coverage confound-exclusion sub-arm IS closed by this finding.

## Derived

- "5×" = 40 000 / 8 000 training positions (integers).
- "essentially unchanged / at the plateau": substrate 0.3403 at 40k vs 0.371 at 8k — a difference of ~0.03, within seed spread.
- "~0.86–0.90": the proxy control's `sub_learned_recov_mean` 0.8593 and `hostlinear_recov_mean` 0.8996.
- Substrate run cost: `elapsed_s` 46817 for 3 seeds ≈ 13.0 GPU-h (~4.3 h/seed), recorded with a `cost_projection` in the artifact.
