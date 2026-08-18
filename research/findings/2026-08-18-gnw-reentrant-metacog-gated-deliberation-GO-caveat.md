---
status: qualified
type: finding
lane: T1-1
date: 2026-08-18
---
# THE KEYSTONE — a conflict/ignition-count-GATED re-entrant deliberation loop on the GNW bus (T1-1 rung d): the re-entrant cycle count EMERGES from the substrate's own spiking IGNITION/CONFLICT read, not a host-fixed counter — GO (with a verified attribution caveat)

**Date:** 2026-08-18 · **Runner:** [`research/runners/_gnw_reentrant_metacog_gated_deliberation_derisk.py`](runners/_gnw_reentrant_metacog_gated_deliberation_derisk.py) · **Artifact:** `research/findings/raw/_gnw_reentrant_metacog_gated/summary.json` (+ `.prov.json`) · **Scope:** additive default-off de-risk, reuse-by-import, `NO sim/ edit` (`git diff sim/` empty). FUNCTIONAL correlate only; NO phenomenal claim.

## Verdict: GO-caveat (6/6 seeds; core substrate-control claim survives 3 adversarial lenses; 2 narrowings)

The **core capability GOes**: the re-entrant deliberation cycle count is set by a read of the substrate's OWN SPIKES (`n_ignited`, off `cp_firing_states`), NOT a host-fixed counter — the first time a spiking conflict/ignition signal CONTROLS deliberation depth (T1-1 rung d: "an ACC conflict unit reads the WTA competition → gates an extra deliberation cycle"). Substrate control is carried by the no-host-orchestration guard + A3 (per-trial halt timing) + A4 (lesion dissociation).

**Two verify-go narrowings (commit-with-caveat), stated up front:**
1. **The decisive spiking read is `n_ignited` (the ignition/CONFLICT count), NOT the graded NMDA-balance `conf`.** On this fixture `conf` is a binary constant (0.94 at a resolved hop / 0.00 at the terminal) perfectly redundant with `n_ignited` (1 / 0). A θ_hi sweep on seed 42 reads `reent_acc = 1.000` for EVERY θ_hi in [0.00, 0.94] and 0.000 only for θ_hi ≥ 0.96 — so the graded confidence does NO independent work on the primary variable-depth task; any threshold in the wide gap is identical. Graded `conf`'s independent role is reserved for the moat (refuse a low-`conf` single winner) and the unbuilt Part-B tie-break. So "confidence-gated" over-attributes; the honest name is **ignition-count / conflict-gated**.
2. **The +0.75 "beats best fixed-k" is a variable-depth QUALITATIVE fact, not a substrate MAGNITUDE.** `build_var_chains(4)` puts exactly 4 chains at each depth L∈{2,3,4,5}, so `fixed_count_chase(k)` lands the leaf only at k==L → best single k ≡ 4/16 = 0.25 BY CONSTRUCTION, independent of the substrate; a terminal-reaching loop scores 1.0 → +0.75 is forced by the equal-depth design. Skewing the depth distribution changes the margin with the substrate unchanged. A2 in isolation proves "depth varies and the loop reaches the terminal," not substrate magnitude — the substrate-control burden is correctly carried by A3 + A4 + the guard.

## Why it is the keystone

The faculty audit's #1 ([`docs/plans/2026-08-12-faculty-map-gap-audit-and-roadmap.md`](../../docs/plans/2026-08-12-faculty-map-gap-audit-and-roadmap.md) T1-1): ACT on the conflict/confidence signals we only REPORT.
In every prior artifact the number of re-entrant cycles is a HOST CONSTANT: P1.2 `reentrant_chase` runs `n_hops = len(actions)`; the coincidence integrator runs `HOPS=2`; the metacog read (`nmda_norm_margin`) is wired to production but READ-ONLY ("nothing acts on it"). This de-risk closes that: the substrate's own spiking read decides whether to iterate another cycle or halt, so the brain works through a multi-step inference whose DEPTH it discovers itself, not a host counter.

## Mechanism (reuse-by-import; the ONLY structural change is host-count → substrate-read)

ONE persistent GNW workspace (the P1.2 `build_workspace_bridge`: K=4 dense self-recurrent assemblies + one shared inhibitory `workspace_fs` pool = single-content WTA). Per cycle:
- **PROPOSE** — `composer.query_patient(x, EAT)` (the declared modular-processor boundary, same as P1.2 / the coincidence integrator; `x` = the spiking read of the last committed winner; returns None at the leaf).
- **EVALUATE/COMMIT** — drive the candidate strong + distractors weak into the slots; mutual-inhibition WTA + ignition threshold sustain ONE winner (the EXACT `_deliberate_hop`/`norgan_hop` the production bus runs). If PROPOSE missed (leaf), drive NOTHING → the workspace stays quiescent.
- **READ (the control variable)** — off the SAME workspace: `n_ignited` (# slots over the ignition knee, off `cp_firing_states`) = the CONFLICT/ignition read, plus `conf = |g_nmda(win) − g_nmda(runnerup)| / (g_nmda(win) + g_nmda(runnerup) + eps)` (the production-default `metacog_production_organ.nmda_norm_margin`, off `cp_conductance_g_nmda`) = a corroborating confidence read.
- **ACC GATE** — `acc_conflict_gate(conf, n_ignited, cycles_on_hop, R_max, θ)` → {ADVANCE|RETRY|COMMIT|ABSTAIN}. `n_ignited==0 → COMMIT` (terminal), `conf≥θ_hi ∧ n_ignited==1 → ADVANCE`, `n_ignited≥2 → RETRY`, else ABSTAIN. `reentrant_chase`'s `for h in range(n_hops)` becomes `while gate != COMMIT/ABSTAIN`.

**The terminal HALT is a spiking read, upstream-caused by a declared boundary.** At the leaf PROPOSE misses → the loop drives all-zeros → the workspace reads `n_ignited==0` off `cp_firing_states` → the gate returns COMMIT. There is NO host `if target is None: break` — the STOP DECISION is the gate reading spikes. The composer miss (PROPOSE) is the declared modular-processor boundary that CAUSES the collapse; the substrate's independent contribution is the per-trial halt TIMING (A3 proves a rate-matched random halt fails) + the ADVANCE guard + the dissociation (A4).

**θ self-calibration (seed-invariant).** A synthetic SOLO/CONFLICT/NULL battery with NO task labels: SOLO → conf≈1.0; CONFLICT → conf≈0.0; NULL → conf≈0.0. `θ_hi = 0.5·(min_solo + max_conflict)` → **θ_hi=0.500, clean_gap=True on all 6 seeds**. (Per narrowing #1 the value is uncritical: any θ_hi ∈ [0, 0.94] gives the identical 6/6 GO.)

## GO task — variable-depth transitive chase, depth NEVER told

Chains of mixed depth L∈{2,3,4,5} under one relation (EAT); 16 chains, 72 concepts, chance ≈ 0.014. Cue = ch[0], answer = the terminal leaf. The loop keeps re-entering while a single slot ignites and HALTS when PROPOSE misses at the leaf → the workspace reads `n_ignited==0`. <!--derived-->

## RESULT — 6/6 seeds (rule ≥5/6)

| metric | gate | mean | per-seed 42/43/44/100/101/102 |
|---|---|---|---|
| reentrant_confgated_acc | (1) ≥0.90 | **1.000** | 1.00 all six |
| single-pass acc (k=1, the wired bus) | (2) ≤0.15 | **0.000** | 0.00 all six |
| BEATS best fixed-k (qualitative; see caveat 2) | (3) ≥0.20 | **+0.750** | +0.75 all six |
| spearman(halt_cycle, true depth) | (4) ≥0.9 | **1.000** | 1.00 all six; halt_at_H_cap=False all |
| θ_hi (self-calibrated split) | seed-invariant | **0.500** | 0.50 all six (clean_gap=True all) |
| conf-blind acc (A3) | →floor | 0.101 | 0.08/0.17/0.04/0.10/0.10/0.10 | <!--derived-->
| lesion acc / 1-hop reflex (A4) | ≤0.10 / ≥0.85 | 0.000 / 1.000 | 0.00 / 1.00 all six |
| re-cue lesion (A5) | ≤0.10 | 0.017 | 0.00/0.04/0.00/0.02/0.04/0.00 | <!--derived-->
| permuted-puredepth (A6a) / permuted-workspace (A6b) | ≈0 / collapse | 0.007 / 0.059 | pure ≤0.02 all; workspace 0.00–0.12 | <!--derived-->
| spreading floor (A7) | reent−floor ≥0.5 | 0.135 | 0.12–0.19 | <!--derived-->

## Verify-go (adversarial; 3 lenses + direct re-runs)

Ran control-integrity, novelty/same-quantity, and no-host-orchestration/seeding lenses; refined the claim to what survives.
- **Same-quantity (SURVIVES clean):** `reent_acc`, `singlepass_acc`, `fixed_count_acc[k]` are computed over the SAME 16 chains, SAME scoring (`== ch[-1]`), SAME `_deliberate_hop_conf` ignition + SAME per-hop distractor RNG; the ONLY difference is the stop rule (`for range(k)` vs `while gate`). Single-pass IS the one-hop workspace ignition (the wired-bus baseline). No wrong-quantity comparison.
- **A3 valid (SURVIVES):** `confidence_blind_chase` is genuinely DENIED the free terminal — a probe with p_stop=0 returns None for L=2..5 (it over-runs to the leaf and abstains, `if target is None: return None`); the ONLY non-None return is the Bernoulli halt. p_stop=1/4.5=0.222 (correct). Blind 0.101 < best-fixed-k 0.25: matching the average stop rate is insufficient; per-trial timing wins. <!--derived-->
- **Control-integrity (NARROW → caveat 1):** θ_hi sweep on seed 42 = 1.000 for θ_hi ∈ [0,0.94], 0.000 for θ_hi ≥ 0.96 → graded `conf` is a binary redundant with `n_ignited`; retitled to ignition-count/conflict-gated.
- **No-host-orchestration (SURVIVES):** `acc_conflict_gate` signature is exactly `(conf, n_ignited, cycles_on_hop, R_max, θ_hi, θ_lo)` — no `target`/`L`/`chain`/`len(`/depth; `confidence_gated_chase` never receives L/the chain; `assert_no_host_orchestration()` runtime-checks both and passes. Determinism: `cfg.seed` build-twice `cp_neuron_firing_thresholds` hash identical (same-seed) / different (diff-seed). `git diff sim/` empty.
- **Novelty (NARROW → caveat 2):** +0.75 is a variable-depth qualitative fact, not a substrate magnitude (best-k≡0.25 by the equal-depth fixture).

## Honest residuals (declared, not faked)

- **Graded confidence is not yet load-bearing.** On this fixture the NMDA balance is a binary copy of `n_ignited`. Making the GRADED confidence do independent work needs the Part-B conflict rung (an ambiguous hop where two candidates co-ignite and cross-retry NMDA accumulation breaks the near-tie) — a STRENGTHENING probe, NOT built here: under the per-hop-reset wash-out RETRY re-drives rather than accumulates. Part A (variable-depth termination via `n_ignited`) carries the GO.
- **Per-hop-reset form only.** The continuous no-reset train-of-thought is gated on the unbuilt Rung-2b async attractor.
- **PROPOSE is a declared modular-processor boundary** (`composer.query_patient`), same as P1.2 / the coincidence integrator. The terminal is upstream-caused by its miss; the substrate's independent work is the per-trial halt TIMING (A3) + the ADVANCE guard + the A4 dissociation. The novelty is the LOOP COUNT moving from host to a spiking read, NOT that PROPOSE moved.
- **rung-e STN→GPi STOP-veto effector is the next follow-on.**

## Backend / determinism (measured)

Decisive 6-seed on `SIM_BACKEND=numpy` (390-neuron workspace: numpy 24 ms/ignition vs cupy 68 ms — GPU launch overhead dominates a tiny net; determinism identical, `cfg.seed` seeds both; matches the P1.2 numpy-decisive precedent). `query_patient` (the composer's spiking RF resonator) is memoized per composer — a deterministic pure read, cached, changing no result. n_shuffles=3 (the A6 verdict rests on perm_puredepth≈0, robust to shuffle count). D=256; retrieval exact (D=128 also exact).

## Path to production (after this GO, not this session)

Wrap `webapp/gnw_bus_shadow.py::gate_via_bus` so on the covered class the bus runs the ignition-count-gated re-entrant loop instead of one `norgan_hop`; additive default-off (`BRAIN_GNW_REENTRANT`), byte-identical on 1-hop turns; `BRAIN_GNW_BUS_HOST=1` reverts. Closes rung (d) of the coincidence-integrator finding's honest path to wiring; the Part-B graded-confidence tie-break + rung (e) STN→GPi are the follow-ons.

Cites: P1.2 GO [`2026-07-24-P1.2-GNW-workspace-deliberation-6seed-GO-adversarially-verified.md`](2026-07-24-P1.2-GNW-workspace-deliberation-6seed-GO-adversarially-verified.md); coincidence integrator [`2026-08-12-gnw-coincidence-integrator-substrate-combines-two-organ-reads.md`](2026-08-12-gnw-coincidence-integrator-substrate-combines-two-organ-reads.md); metacog confidence [`2026-08-13-metacog-robust-confidence-GO.md`](2026-08-13-metacog-robust-confidence-GO.md); faculty audit T1-1.
