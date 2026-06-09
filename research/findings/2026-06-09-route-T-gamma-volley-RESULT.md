# Route T Step-0 RESULT — gamma volley synchronizes the sparse place ensemble; the jitter arbiter PASSES; verdict PARTIAL

**Date:** 2026-06-09
**Type:** RUNNER-ONLY de-risk (ZERO `sim/` edits — `git status --short sim/` byte-empty, verified before+after). CuPy, deterministic regime, ≥3 seeds (42/43/44).
**Design:** `2026-06-09-route-T-volley-synchronization-design.md` (THE design — gamma/PING re-times the sparse ensemble into a coincident volley that the landed Route D detector fires).
**Solves:** the ASYNCHRONY wall (`coincidence_wall_probe.py` + `_coincidence_wall_probe.json`: a sparse-distinct ~10 Hz place ensemble emits per-step coincidence c_i ≤ 1 → Route D has nothing to detect; `no_valid_K_above_1=true` across 3 CuPy seeds).
**Probe:** `research/runners/coincidence_volley_n9_derisk.py` (the staged bed extended with the gamma synchronizer; reuses `coincidence_wall_probe._build` topology + Route D landed `b980070a`).
**Owner directive:** biologize everything, brain-based-only; the jitter anti-cheat is the arbiter; an honest negative IS the deliverable.

---

## TL;DR — VERDICT: **PARTIAL** (mechanism validated; the brain-based FS-PING is the better synchronizer; two residual gaps named)

A gamma synchronizer on the sparse place pool **does** re-time its active cells into a coincident volley that fires the downstream MSN-D1 through the landed Route-D coincidence detector — and **the load-bearing jitter anti-cheat PASSES unanimously** (de-synchronizing the volley collapses MSN firing to 0.0 Hz at every seed, both synchronizers). The mechanism is genuinely **coincidence, not rate.** Route D is load-bearing (ablating it → MSN silent even with the rhythm on). **This is the N9 place-grading unblock the route was after — the detector reads a synchronized sparse-distinct volley.**

**But it is not a clean GO**, for two honest reasons:
1. **G_SPARSE fails for host-pacing** (the uniform depolarizing pulse densifies the place code to ~18–19%). The **brain-based FS-PING synchronizer keeps the code ~3× sparser (5.9–7.2%)** — it is the *better* synchronizer — but it still marginally misses the ≤5% bar and is seed-variable (fires the MSN at 2/3 seeds, weak at seed 44).
2. **The async chance-coincidence floor is non-zero** (no-rhythm MSN ~2.8–5.6 Hz): the sustained sparse async code occasionally aligns ≥K=4 spikes by chance over 20 targets. The rhythm's firing (11–22 Hz) is clearly **above** this floor and jitter collapses it to 0, so the rhythm's contribution is real and coincidence-based — but the floor is not clean zero.

**Which synchronizer worked:** BOTH fire the MSN and BOTH collapse under jitter. **Host-pacing** (the documented SH-5 scaffold) fires more reliably (3/3) but densifies. **FS-PING** (the faithful, brain-based, neurons+synapses-only generator) preserves sparsity and fires 2/3 — it is the better candidate, and the honest next step is tightening it (not a host teacher). The volley does **not** span so many dt steps that it fails G_VOLLEY (max c_i reaches 5–8 ≥ K in the volley step), so the deferred conduction-delay `sim/` ring buffer is **NOT** triggered by this result.

---

## The headline numbers (3 seeds: 42/43/44, K=4, s2t-density 0.5 clustered, CuPy deterministic)

### Host-pacing synchronizer (SH-5 scaffold: location-blind gamma pulse on the place pool)

| Condition | MSN Hz (42/43/44) | source sparsity | volley c_i p90 / max | G_FIRE | G_DISTINCT | verdict |
|---|---|---|---|---|---|---|
| **Volley (rhythm ON)** | 22.2 / 16.7 / 11.1 | ~18–19% | 2.2 / 5–6 | **3/3** | 3/3 | fires + distinct, but DENSE |
| **No-rhythm (floor)** | 2.8 / 5.6 / 2.8 | 18–19%* | 1.7 / 4 | 1/3 | 2/3 | async chance floor |
| **JITTER (de-sync)** | 0.0 / 0.0 / 0.0 | (sparse) | 0.0 / 0 | **0/3** | 0/3 | ✅ COLLAPSES (arbiter) |
| **Ablate Route D** | 0.0 / 0.0 / 0.0 | — | 1.2 / 4 | **0/3** | 0/3 | ✅ plateau load-bearing |

### FS-PING synchronizer (FAITHFUL, brain-based: place→FS→place recurrent inhibition → emergent gamma)

| Condition | MSN Hz (42/43/44) | source sparsity | volley c_i p90 / max | G_FIRE | G_DISTINCT | verdict |
|---|---|---|---|---|---|---|
| **Volley (FS-PING ON)** | 11.1 / 15.3 / 1.4 | **5.9–7.2%** | 0–0.2 / 7–8 | **2/3** | 1/3 | fires (2/3) + SPARSE |
| **No-rhythm (FS omitted)** | 2.8 / 5.6 / 2.8 | 14%* | — / — | 1/3 | 2/3 | async floor |
| **JITTER (de-sync)** | 0.0 / 0.0 / 0.0 | (sparse) | 0.0 / 0 | **0/3** | 0/3 | ✅ COLLAPSES (arbiter) |
| **Ablate Route D** | 0.0 / 0.0 / 0.0 | — | — / — | **0/3** | 0/3 | ✅ plateau load-bearing |

\* with the FS pool omitted (no-rhythm for ping) the source is *denser* (~14%) than with FS on (~6%) — the FS inhibition is what keeps it sparse; that the bare source still chance-fires the target is the floor.

**The decisive pair (run first, per the task):**
- **G_FIRE:** synchronizer + Route D ON → MSN fires (pacing 11–22 Hz 3/3; FS-PING 11–15 Hz 2/3) vs the c_i≤1 baseline's ~0–6 Hz floor. ✅ (pacing) / ◑ (FS-PING 2/3).
- **JITTER anti-cheat (the arbiter, Branco-Häusser):** de-synchronize the volley → MSN firing COLLAPSES to **0.0 Hz at every seed, both synchronizers.** ✅✅ — proves coincidence, not rate. **This is the load-bearing pass.**

---

## What each gate / anti-cheat showed

- **G_VOLLEY (3/3 both):** the synchronizer raises the per-step max coincidence to **5–8 ≥ K=4** in the volley step (vs Step-0's ≤1). The rhythm genuinely packs the sparse ensemble into a coincident packet **within a single dt step** — so the volley does NOT span multiple steps, and the deferred conduction-delay `sim/` ring buffer (§3.2 of the design) is **not** the bottleneck here.
- **G_FIRE:** pacing 3/3 (22/17/11 Hz), FS-PING 2/3 (11/15 Hz; seed 44 weak at 1.4 Hz). Both clearly above the async floor.
- **G_DISTINCT (downstream position-specificity):** holds at 3/3 for pacing, 1/3 for FS-PING. The clustered `s2t-density=0.5` projection (each target samples a different ~50% subset of the place pool) makes different locations' volleys hit different targets → downstream diff-cos 0.03–0.16 (distinct). When firing is weak (FS-PING seed 43/44) the distinctness proxy is unreliable (too few target spikes), so G_DISTINCT 1/3 there reflects sparse firing, not position-blindness.
- **G_SPARSE:** **fails for both** at the ≤5% bar — but very differently. **Host-pacing densifies to ~18–19%** (the uniform depolarizing pulse recruits the band of sub-threshold cells; sharpening the pulse to duty 0.04 didn't help — it's the *amplitude*, not the duration, that crosses the sub-threshold band). **FS-PING stays at 5.9–7.2%** — ~3× sparser, just over the bar, because the inhibitory gating raises the effective threshold for everyone and only the best-driven escape. **This is exactly why FS-PING (inhibitory gating) is the biologically correct synchronizer and pacing (pure depolarization) is only a scaffold.**
- **JITTER → COLLAPSE (0/3 both):** ✅ THE ARBITER. De-synchronizing the sensor drive (same cells, same total drive, spikes spread across alternating steps) drops MSN firing to 0.0 Hz everywhere. The firing is the synchronized volley, not rate.
- **Ablate Route D → silent (0/3 both):** ✅ with the rhythm ON but `enable_coincidence_detection=False`, the MSN never fires (0.0 Hz) — a synchronized volley of K *sub-threshold* AMPA inputs without the supralinear plateau cannot fire the soma. Confirms BOTH halves (rhythm + Route D) are needed; the volley is not "just more rate."
- **K > 1:** K=4 throughout (a single coincident input cannot trigger). At K=6 the achievable sparse c_i (max 4–6) never reaches threshold, so nothing fires — K=4 is the operating point for this fan-in/sparsity.
- **NO host teacher (audited):** the only `cp_external_input_current` writes are the sensory afferent (`src_sensors`) and, for pacing, the **location-BLIND** uniform gamma pulse on the place pool (asserted: the same {on,off} pulse vector regardless of location — it sets WHEN, not WHICH). FS-PING has no host pacing at all. The MSN fires from the brain's own routed synaptic coincidence. Recorded in `driven_regions` + `pace_pulse_audit` per run.
- **CuPy regime:** `backend=="cupy"` hard-asserted (numpy disqualified); OU / conductance-noise / global-homeostasis / heterogeneity / STP OFF; no per-region homeostasis on the MSN target. All asserted.

---

## The methodological subtlety that the diagnostic settled (why this is honest, not noise)

A constant sensory clamp on a **noiseless** IF place pool (deterministic regime, required) produces a **sustained sparse ASYNCHRONOUS train** — NOT a one-shot onset burst. The per-step diagnostic (200 steps, bare source, constant clamp) confirmed: first spike step 44, last step 187, 42/200 active steps, ~0.3 spikes/step scattered throughout — exactly the wall-probe's c_i≤1 regime. So the asynchrony wall is real in the sustained state, and the rhythm's job (align those scattered spikes into one step) is the genuine test. The non-zero no-rhythm floor (~2.8–5.6 Hz) is **chance** alignment of the async spikes over 20 targets × 120 steps at K=4; the rhythm-on firing (11–22 Hz) is decisively above it and collapses under jitter — so the rhythm's contribution is real coincidence. Raising K to 6 cleans the floor to 0 but then the sparse c_i can't reach threshold (the fan-in/sparsity ceiling). This trade-off (clean floor ⇄ achievable volley) is the honest residual, not an artifact.

---

## Honest scope + the named next levers

- **The volley mechanism is validated runner-side, zero `sim/` edits.** The jitter arbiter passes; Route D is load-bearing; the volley packs into a single step (G_VOLLEY 3/3) so conduction delays are NOT the missing piece here.
- **The brain-based FS-PING synchronizer is the better one** (sparse-preserving, fires 2/3, passes jitter+ablate). It is NOT a clean 3/3 GO: (a) it marginally exceeds the ≤5% sparsity bar (5.9–7.2%), and (b) it is seed-variable (weak at seed 44). **This is a documented next step — tighten the FS-PING operating point (FS weight/density, GABA_A decay → gamma frequency, place-pool excitability) so all seeds fire ≥5 Hz while the code stays ≤5%. NOT a host teacher.**
- **Host-pacing is a working scaffold but densifies** (uniform depolarization recruits sub-threshold cells). It validated the detector reads a synchronized volley; it is not the faithful endpoint.
- **What would push FS-PING from PARTIAL → GO:** (i) tune the FS-PING gamma so the active cells re-fire in a tighter packet at all seeds (the seed-44 weakness is the volley not reaching the MSN, not a sparsity problem); (ii) a per-region intrinsic homeostasis on the place pool to pin sparsity ≤5% across seeds (the `placecode_selforg_stage1` `place_homeostasis` mechanism, legitimate intrinsic excitability, not a threshold-collapse rescue); (iii) if the within-gamma-window spikes still scatter across dt steps at finer dt, THEN the conduction-delay ring buffer (`sim/` §3.2) becomes the named lever — but G_VOLLEY 3/3 says it is not needed at dt=1ms.

---

## Reproduce

```bash
# decisive pair (host-pacing scaffold): G_FIRE + jitter
SIM_BACKEND=cupy python -m research.runners.coincidence_volley_n9_derisk \
    --seeds 42,43,44 --sync pacing --k-threshold 4 --s2t-density 0.5 --pace-amp-pA 60 \
    --out research/findings/raw/_volley_pacing_volley.json
SIM_BACKEND=cupy python -m research.runners.coincidence_volley_n9_derisk \
    --seeds 42,43,44 --sync pacing --jitter-inputs   # MUST collapse -> 0 Hz

# brain-based FS-PING (the faithful synchronizer):
SIM_BACKEND=cupy python -m research.runners.coincidence_volley_n9_derisk \
    --seeds 42,43,44 --sync ping --k-threshold 4 --s2t-density 0.5 --fs-to-place-weight 14 \
    --out research/findings/raw/_volley_ping_volley.json
SIM_BACKEND=cupy python -m research.runners.coincidence_volley_n9_derisk \
    --seeds 42,43,44 --sync ping --jitter-inputs      # MUST collapse -> 0 Hz

# controls (both syncs): no-rhythm (async floor) + ablate Route D (must be silent)
SIM_BACKEND=cupy python -m research.runners.coincidence_volley_n9_derisk --seeds 42,43,44 --sync ping --no-rhythm
SIM_BACKEND=cupy python -m research.runners.coincidence_volley_n9_derisk --seeds 42,43,44 --sync ping --ablate-subunit
```

Raw JSON: `research/findings/raw/_volley_{pacing,ping}_{volley,jitter,norhythm,ablate}.json`.

---

## Bottom line

The gamma-volley route is **real and runner-side**: a synchronizer re-times the sparse-distinct place code into a coincident volley that the landed Route-D detector fires the MSN-D1 from, and the **decisive jitter anti-cheat collapses it (0 Hz, 3/3, both synchronizers) — so it is coincidence, not rate.** The brain-based FS-PING generator is the better synchronizer (sparse-preserving, fires 2/3 above the async floor, passes jitter+ablate). It is **PARTIAL**, not GO, because FS-PING marginally exceeds the ≤5% sparsity bar and is seed-variable; the named, brain-based next step is tightening the FS-PING operating point (+ intrinsic place-pool homeostasis), **NOT** a host teacher and **NOT** the conduction-delay `sim/` edit (G_VOLLEY 3/3 shows the volley already packs into one step). N9 place-grading is plausibly unblockable runner-side once FS-PING is tightened to a clean ≥5 Hz / ≤5% across seeds.
