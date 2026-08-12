---
type: finding
status: contributing
date: 2026-08-12
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/_gap4_birdsong/tutor_teach_6seed_N2.json
  - research/findings/raw/_gap4_birdsong/tutor_teach_6seed_N3.json
  - research/findings/raw/_gap4_birdsong/of0.json
  - research/findings/raw/_gap4_birdsong/od2.json
  - research/findings/raw/_gap4_birdsong/od3.json
  - research/runners/_gap4_birdsong_tutor_teach_derisk.py
---

# gap#4 crux — the TWO-STAGE BIRDSONG tutor-decomposition (Teşileanu 2017) is an HONEST NEGATIVE on a DEEP (N=2/3) spiking LIF stack: the low-dim RL tutor + reward-independent local Hebbian-follow does NOT enter the learning regime (6-seed, both depths, tutor ≤ reservoir). The DECISIVE failing stage is STAGE B — local Hebbian following a fixed-random broadcast of the tutor's output target confers NO deep credit even given a PERFECT target and a strong readout, while surrogate-BPTT (true deep credit) reaches 0.74–0.75

<!--derived-->
**One-line verdict.** The birdsong two-stage decomposition (a LOW-DIM LMAN-analogue tutor learns a corrective teaching signal by reward-modulated node-perturbation + a reward baseline; the DEEP HVC→RA motor stack then trains by a REWARD-INDEPENDENT local Hebbian rule following the tutor's per-neuron target — so no top-down error is routed through the finite-spike σ′ read) was built and tested at 6 seeds (42/43/44/100/101/102) on depth-2 AND depth-3 LIF nets, on the validated compositional-inheritance instrument. It is a clean NEGATIVE: `tutor_teach` never beats the frozen reservoir (mean 0.284 vs 0.352 at N=2; 0.265 vs 0.333 at N=3; 0/6 and 1/6 seeds ≥+0.10; above-chance −0.05/−0.07 where GO needs +0.20). The isolation LOCATES the failure at STAGE B: even an ORACLE tutor (the perfect onehot target on train) followed by the deep Hebbian broadcast rule sits BELOW the frozen reservoir (mean 0.191/0.204), and turning the hidden updates ON makes held-out generalization WORSE than frozen — while surrogate-BPTT on the identical net trains to 0.74/0.75. No `sim/` edit (additive runner, reuse-by-import).

## Did I read the sources + how I differ from the prior 2026-05-16 songbird NEGATIVE

<!--derived-->
**Teşileanu, Ölveczky, Balasubramanian 2017, eLife 6:e20944** ("Rules and mechanisms for efficient two-stage learning in neural circuits") — READ. The two load-bearing equations were implemented faithfully: the STUDENT (RA) rule is **Eq 1, reward-INDEPENDENT**: `dW_ij = η · c̃_i(pre) · (g_j(tutor) − θ)` (Hebbian product of filtered presynaptic drive and the thresholded tutor teaching, NO σ′); the TUTOR (LMAN) rule is **Eq 6, reward-modulated node perturbation with a baseline**: `Δf_j = η(R − R̄) ξ_j`, R̄ the running mean reward. The paper's matched-timescale result (Eq 4) is honored by construction (student + tutor act on the SAME per-trial teaching). Faithfulness caveat (documented in the runner): the birdsong circuit is SHALLOW (one HVC→RA synapse; LMAN biases RA); the DEEP extension — broadcasting the low-dim tutor latent to every layer by a FIXED-RANDOM matrix — is mine, and it is exactly the part that fails (below).

<!--derived-->
**How this differs from `2026-05-16-generator-G1-songbird-NEGATIVE`** (which I read): that attempt trained a `SongHVC` argmax controller by SELF-COMPREHENSION of babbled productions over the G.20 recognition substrate; it failed because the order-readout JUDGE could not discriminate order, so reward was identically 0 and `SongHVC.W` never moved — a single-stage controller with a broken reward, no two-stage decomposition, no baseline, no low-dim-tutor confinement. This work is a different mechanism on a different substrate: (1) a WELL-DEFINED graded environmental reward (the negative auditory-template error, `−‖produced − onehot(y)‖²`, computed by the world), which is non-zero and informative; (2) the TWO-STAGE decomposition itself (Eq 1 + Eq 6); (3) a reward BASELINE + advantage normalization; (4) a DEEP spiking LIF student on today's validated compositional-inheritance instrument. So this is NOT a re-run of the 2026-05-16 gate — it tests the decomposition the 2026-05-16 work never had, and it fails for a NEW, cleanly-isolated reason (STAGE B, not a broken reward).

<!--derived-->
**Substrate reused (reuse-by-import, NO `sim/` edit):** the LIF surrogate-BPTT stack + task from `research/runners/_snn_bptt_forward_vs_learning_isolation_derisk.py` (`_build_layers`, `_forward_logits`, `_train_snn`) and `research/runners/_semantic_inheritance_deep_credit_derisk.py` (`make_task_semantic_inheritance`, `stage0_depth_genuineness`, `_train_oracle`) — the same instrument where DFA e-prop trains at N=2 (inherit ~0.895) but N≥3 local rules do not enter the regime (`2026-08-02-gap4-depth-rescue-untestable-*`). The songbird `SongHVC` (`sim/song_hvc.py`) is a host-computed argmax controller, NOT a spiking stack, so it is NOT the HVC→RA motor substrate; the LIF stack is.

## Result — 6 seeds (42/43/44/100/101/102), held-out-inheritance accuracy

<!--derived-->
Chance (= majority-class on the inheritance held-out subset) is 0.333. Artifacts: `research/findings/raw/_gap4_birdsong/tutor_teach_6seed_N2.json` and `..._N3.json` (numpy/CPU, 150 epochs, hidden 32, T 24). Reward-independent local rules (candidate + controls) are BRAIN-BASED; BPTT is a non-biological CEILING only.

| arm (held-out inherit, mean) | N=2 | N=3 | role |
|---|---|---|---|
| surrogate-BPTT (ceiling, non-bio) | **0.741** | **0.753** | true deep credit — the substrate CAN benefit |
| naive high-dim node-perturbation | 0.414 | 0.358 | genuine NP (perturbed forward + antithetic baseline) |
| frozen reservoir (floor) | 0.352 | 0.333 | the bar the candidate must beat by ≥0.10 |
| **tutor_teach (CANDIDATE)** | **0.284** | **0.265** | below chance, below reservoir |
| permuted_reward (anti-cheat) | 0.235 | 0.222 | tutor RL de-driven → collapse ✓ |
| oracle_tutor (STAGE-B ceiling, perfect target) | 0.191 | 0.204 | perfect target still ≤ reservoir |
| shuffle_tutor (anti-cheat) | 0.167 | 0.179 | teaching de-signaled → collapse ✓ |

<!--derived-->
**tutor_teach GO-gate read (all fail):** above-chance mean −0.049 (N=2) / −0.068 (N=3) — GO needs ≥+0.20; beats-reservoir mean −0.068 / −0.068 with **0/6** (N=2) and **1/6** (N=3) seeds ≥+0.10 — GO needs 6/6 by ≥0.10; enters-the-regime (leaves majority-class) **1/6** at each depth. Anti-cheats behave: shuffle_tutor (scramble the per-example teaching) and permuted_reward (permute the reward inside the tutor NP) both collapse BELOW chance, so the small signal tutor_teach does carry is genuinely reward-driven target-following — there is just not enough of it to clear the reservoir.

## The DECISIVE isolation — STAGE B fails even with a PERFECT target (this is the wall, not the tutor's RL)

<!--derived-->
The `oracle_tutor` arm hands the deep stack the PERFECT target — `onehot(y)` on train, broadcast to the hidden layers by the same fixed-random matrix — and skips the RL entirely. It is a within-mechanism CEILING for STAGE B. Even so it sits at 0.191/0.204 (6-seed), BELOW the frozen reservoir (0.352/0.333). Seed-42 isolation pins the mechanism directly (`of0.json`): with the hidden Hebbian FROZEN the readout-follow gives held-out 0.333 (chance); turning the hidden updates ON drops it to 0.222 — the deep credit is NEGATIVE (the fixed-random broadcast of the output target CORRUPTS the hidden representation). Strengthening the readout to a local error-correcting delta rule (`od2.json`/`od3.json`, seed 42, still reward-independent + local) does not rescue it: oracle_tutor+delta held-out 0.333 (N=2) / 0.370 (N=3) still ≤ the frozen reservoir (0.370 / 0.407), while BPTT on the same net is 0.82/0.85. So the STAGE-B negative is not a weak-readout or a bad-tutor artifact.

<!--derived-->
**Why (mechanistic, and it is the SAME reason as the located wall).** To build the compositional hidden representation (member→super→property), the hidden layers need INTERMEDIATE targets. A fixed-random broadcast of the low-dim OUTPUT target carries NO intermediate-target information — it is a fixed feedback of the target (not the error), which is known not to build deep features, and it is precisely the transport-free signal the located wall showed has no purchase on the finite-spike read (`2026-08-02-gap4-crux-wall-LOCATED-*`: even a perfect W⊤ oracle equals a label-shuffle on the spike read). The two-stage decomposition RELOCATES credit to a low-dim tutor that can only teach the OUTPUT; the deep hidden credit remains exactly the unsolved problem. Sidestepping the σ′ read does not help, because the σ′ read was never the whole wall — the missing INTERMEDIATE TARGET is.

## STAGE A also fails on this task (representational, not just NP variance)

<!--derived-->
A LOW-DIM linear tutor (`u = X·U`, perturbed in the k-dim output space — the variance-tractable regime) cannot even REPRESENT the target here: a supervised least-squares linear classifier on the 7-dim member code reaches only 0.356 train (the task is deliberately depth-required and not linearly separable), and the NP-trained tutor lands at tutor-train-acc ~0.08–0.33 (≈chance). So STAGE A cannot supply a useful target on a depth-required task with a low-dim tutor — but STAGE B is the DEEPER failure, since it fails even when STAGE A is replaced by the perfect oracle target.

## Honest secondary finding — the "refuted" naive high-dim NP OUT-performs the candidate here (scope-caveated)

<!--derived-->
`naive_np` (genuine node perturbation on the high-dim deep read-state: a perturbed forward pass propagates the hidden perturbation to the output, credit by `R_pert − R_clean`, the clean rendition as an antithetic baseline; output by delta) is the BEST local arm (0.414/0.358, above the reservoir), inverting the pre-registered expectation that it would collapse. This does NOT contradict the prior on-bridge refutation (`2026-07-13-NP-vs-KP-REFUTED`): that was the on-bridge Izhikevich / dense-redundant SPIKING regime with a stochastic readout-noise variance wall; THIS is the deterministic surrogate-LIF substrate (OU/noise off), where injected exploration + a clean antithetic baseline give NP a usable zeroth-order gradient (consistent with the off-bridge NP-on-rate GO, `2026-07-13-fresh-deep-credit-class-NODE-PERTURBATION-*`). The lesson is mechanistic and reinforces the wall: **deep credit on this stack requires MEASURING the hidden→output causal effect** (NP measures it by perturbation; BPTT by gradient). The tutor's decoupled Hebbian-follow of a broadcast target measures nothing about the deep layer's effect on the output — which is exactly why it confers no deep credit.

## Verdict + the named next direction (no defer)

<!--derived-->
EXTERNAL-SEARCH-RAN: read the mechanism's own source in full — Teşileanu, Ölveczky & Balasubramanian 2017, "Rules and mechanisms for efficient two-stage learning in neural circuits", eLife 6:e20944 (doi:10.7554/eLife.20944) — implementing its Eq 1 (student) + Eq 6 (tutor) + Eq 4 (matched timescale) faithfully; and the adjacent node-perturbation literature the project already digested (Fiete-Seung 2006 node perturbation; Nøkland 2016 direct feedback alignment; Lillicrap 2016 feedback alignment). This verdict is a METHOD-negative (the decoupled low-dim tutor + fixed-random broadcast-follow is insufficient for deep spiking credit), NOT a capability wall: the capability stays OPEN and the successor mechanism (measured-deep node-perturbation) is named below — nothing is banked as a fundamental limit.

<!--derived-->
**NEGATIVE — the two-stage birdsong tutor-decomposition does NOT get a deep spiking stack into the learning regime; the DECISIVE failing stage is STAGE B (local Hebbian following a fixed-random broadcast of the tutor's output target confers no deep credit even given a perfect target and a strong readout), with STAGE A also failing on the depth-required task (a low-dim linear tutor cannot represent the target).** This is a wall a METHOD hits, not a capability abandoned: it BANKS the "decoupled target-following sidesteps the σ′ read" method as insufficient, and it SHARPENS the located wall — the binding constraint is not the σ′ read per se but the missing INTERMEDIATE TARGET for the hidden layers, which no transport-free fixed broadcast can manufacture. **Named next mechanism (not deferred):** the only local arm that DID confer deep credit on this substrate is the one that MEASURES the hidden→output causal effect (naive_np, 0.41/0.36 > reservoir) — so the productive direction is a variance-reduced, baseline-subtracted node-perturbation MEASUREMENT of the deep spiking layers (the antithetic clean-rendition baseline already helps), i.e. put the RL where the DEEP credit is (measure the hidden units' effect) rather than confining it to a low-dim output tutor and hoping a fixed broadcast carries the hidden target. That is a measured-deep-NP de-risk, and it is the successor to this arc.

## Reproduce (6-seed, foreground, deterministic — verified seed-42 byte-identical across two independent runs)

```
SIM_BACKEND=numpy python -m research.runners._gap4_birdsong_tutor_teach_derisk \
    --seeds 42 43 44 100 101 102 --n-hidden-layers 2 --epochs 150 \
    --lr 0.2 --lr-out 0.2 --lr-tutor 0.05 --sigma 0.3 --beta 2 --antithetic-k 8 \
    --out research/findings/raw/_gap4_birdsong/tutor_teach_6seed_N2.json
# depth-3: same, with --n-hidden-layers 3 --out ..._N3.json
```

## Files

- Runner (additive, NO `sim/` edit): `research/runners/_gap4_birdsong_tutor_teach_derisk.py`
- 6-seed artifacts: `research/findings/raw/_gap4_birdsong/tutor_teach_6seed_{N2,N3}.json`
- STAGE-B isolation: `of0.json` (hidden-frozen vs on), `od2.json`/`od3.json` (perfect target + strong delta readout)
- Located wall this refines: `2026-08-02-gap4-crux-wall-LOCATED-at-the-spiking-read-regime-*.md`,
  `2026-08-02-gap4-depth-rescue-untestable-on-spikes-*.md`
- Prior songbird negative this differs from: `2026-05-16-generator-G1-songbird-NEGATIVE.md`
