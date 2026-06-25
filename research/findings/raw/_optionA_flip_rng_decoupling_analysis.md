# Option A flip — RNG-coupling mechanism + decoupling feasibility + same-N gate redesign

**Date:** 2026-06-24
**Mode:** READ-ONLY analysis (no edits, no runs, no webapp). All claims are file:line-traced + a small numpy/cuRAND sanity check.
**Context:** the onebrain co-resident composer build-fix is DONE + Probe-1 byte-identical (`_closure1_optionA_buildfix_gate3.json`). The FLIP (merged-bridge default `co_resident_composer_kind` `rf`→`onebrain`) is HELD because gate 3 PART 2 returned Δ=+0.6154 ≠ 0 at seed 42. This doc pins the EXACT coupler, says whether a runner-only decoupling exists, redesigns the gate, and ranks the paths.

---

## 0. TL;DR

- **The Δ≠0 is NOT a functional regression and NOT the OU per-step noise during the episode.** During the scored nav episode the OU process is **OFF** (`finalize_conv_for_nav_gate` sets `enable_ou_process=False`, `nav_conv_merged_bridge.py:1369`), and every other per-step RNG site in the episode loop is gated off (het/structural/conductance-noise/homeostasis all False). With all per-step RNG off, the nav trajectory is a deterministic function of the **fixed parameters + the v/u state at episode start**.
- **The actual coupler is the pre-episode PARSER-TRAIN PASS, which steps the WHOLE bridge with OU ON.** `finalize_conv_for_nav_gate` → `train_parser_on_slices` runs `n_epochs(30) × 6 × train_steps(120) = 21,600` `bridge._run_one_simulation_step()` calls with `enable_ou_process=True` (`nav_conv_merged_bridge.py:1358`, `:146-147`). Each step draws `cp.random.randn(n_neurons)` (size = total **N**, `sim/bridge.py:6581`) and steps **all** N neurons — including the nav slice. This leaves the nav neurons' `cp_membrane_potential_v`/`cp_recovery_variable_u` in an **N-dependent** configuration at the end of the parser train. The episode loop (`g11_bg_runner.py:6535`) starts from that state without resetting the nav slice → the N-dependent v/u offset compounds over the trajectory. Phase-0 finalQ matches (the strong reward/perception drive momentarily washes out the small offset); phases 1-3 diverge (it compounds) — exactly the observed `[0.46, 0.54, 1.54, 1.15]` vs `[0.46, 1.31, 0.62, 0.69]`.
- **Decoupling IS achievable runner-side (no `sim/` edit).** Reset the nav slice's `v`/`u` to its resting baseline (`cp_izh_vr`) at the END of `finalize_conv_for_nav_gate` (after the OU-on parser train, before the episode). The nav fixed parameters (traits, firing thresholds) are drawn at seed-time BEFORE any N-divergent consumption and **prefix-match across N** (verified, §1.3); nav weights are per-pathway draws consumed before any rf influence (rf has zero pathways) and are byte-identical. So once the v/u start-state is made N-independent, the OU-off episode is byte-identical → Δ=0.
- **The current gate is methodologically right in PREMISE (a disjoint out-edge-free slice is nav-inert) but the harness it reuses (`finalize_conv_for_nav_gate`) injects an N-dependent v/u start-state through the OU-on parser train.** The fix is either (a) the v/u reset above (makes the gate byte-identical AND fixes the deployed agent's reproducibility), or (b) a same-N functional-toggle gate (§3).

---

## 1. THE RNG-COUPLING MECHANISM (deliverable 1)

### 1.1 The nav neurons' fixed parameters: prefix-matched, N-independent

After the brain-region framework allocates contiguous slices in **list order** (`sim/regions.py:422-447`: `cursor` advances through `self._regions`), the merged bridge's region list is: **nav cascade FIRST** (`build_bg_brain_regions`, `g11_bg_runner.py:4314`), then `regions.extend(extra_regions)` (`g11_bg_runner.py:4669-4670`) appends parser → dlPFC → **rf LAST** (`conv_extra_regions_pathways`, `nav_conv_merged_bridge.py:1306`: `parser + dlpfc + rf + drive`). So:

- **Nav occupies indices `[0, N_nav)` — identical in both arms** (PART 1 confirmed every nav region byte-identical in name+size; `nav_regions_identical=true`).
- **rf occupies the highest indices** — `[N − N_rf, N)`. Changing the rf size changes only the **total N** and the indices of the *trailing* rf block; it does NOT move any nav index.

The global RNG is seeded once with `cfg.seed` at `sim/bridge.py:1105` (`_initialize_rng` → `cp.random.seed(seed)`, `:1053`). The init-time CuPy draw order for the Izhikevich merged config (external-input is `cp.zeros`, no draw, `:1128`) is:

1. `cp_traits = cp.random.randint(0, max(1,num_traits), (n,))` — size-N (`sim/bridge.py:1132`).
2. `cp_neuron_firing_thresholds = cp.random.uniform(thresh_lo, thresh_hi, n)` — size-N (`sim/bridge.py:1461`).

(`_apply_parameter_heterogeneity` is **NOT called** in the gate-3/episode path: `cfg.enable_parameter_heterogeneity=False` at `g11_bg_runner.py:4700` and there is no per-region heterogeneity mask in the standalone `build_bg_brain_regions` nav regions, so the gate at `sim/bridge.py:1677-1678` is False/None.)

Both of these are consumed from the freshly-seeded stream, and **cuRAND prefix-matches the leading elements** when only the array size differs. Verified directly on GPU:

```
cuRAND randn prefix-match (randn(27973)[:k] vs randn(8424)[:k]):                 True
cuRAND thresholds prefix-match after a preceding size-N randint (traits→thresh): True
```

⇒ the nav slice `[0:N_nav]` gets **identical** traits and firing thresholds in both arms. Nav fixed parameters are N-independent.

### 1.2 The nav weights: per-pathway draws, consumed before any rf influence, byte-identical

Connectivity weights are drawn **per-pathway**, sized by per-pathway nnz (not total N): `sim/connectivity.py` (`np.random.uniform(min_w, max_w, num_connections)` at `:452`, `cp.random.uniform(min_w, max_w, (n,k))` at `:181/602`, etc.). The framework path injects pathways in list order via `inject_explicit_wiring`. Nav pathways come first; **rf contributes ZERO pathways** (`co_resident_rf` appends an rf region with no edges, `nav_conv_merged_bridge.py:1292-1293`). So every nav weight draw is consumed from an identical stream state in both arms ⇒ **nav weights are byte-identical**. (The parser-train pass is plasticity-gain-masked to the parser slice — `finalize_conv_for_nav_gate` sets `gain[:]=0; gain[parser_mask]=1`, `:1353-1354` — so it cannot CHANGE the frozen nav weights either.)

### 1.3 The actual coupler: the OU-ON parser-train pass perturbs nav v/u N-dependently

`finalize_conv_for_nav_gate` (`nav_conv_merged_bridge.py:1310`), which BOTH gate-3 arms run as the `prebuilt_post_init_hook` (`_closure1_optionA_gate3.py:99-100, 110`), does:

```
1348  saved = (..., enable_ou_process, ...)
1358  cc.enable_ou_process = True            # OU ON for the parser train
1362  train_parser_on_slices(bridge, ..., n_epochs=30, train_steps=120)
1366  # restore for the episode: ... OU off (nav default).
1369  cc.enable_ou_process = False
```

`train_parser_on_slices` (`:132-148`):

```
139  for _ in range(n_epochs):          # 30
140    for k in range(6):               # 6 conjunctions
141      _step_reset(bridge)            # also steps the full bridge (OU on)
146      for _ in range(train_steps):   # 120
147        bridge._run_one_simulation_step()   # steps ALL N neurons, OU on
```

⇒ ~21,600 full-bridge steps with OU ON. The OU per-step draw is `noise_samples = cp.random.randn(n_neurons)` (`sim/bridge.py:6581`, gated by `enable_ou_process` at `:6575`) — **size = total N**. Two consequences, both N-dependent:

1. **Stream-state divergence.** One arm consumes `21600 × 27973` randn samples, the other `21600 × 8424` → the global stream is in a wildly different state by episode start. (Irrelevant during the OU-OFF episode, but it would matter for any *other* downstream draw.)
2. **Nav v/u divergence (the load-bearing one).** Each `_run_one_simulation_step` integrates the nav neurons' membrane dynamics under OU noise. After 21,600 steps the nav slice's `cp_membrane_potential_v` / `cp_recovery_variable_u` sit at an **N-dependent** configuration. Even though the first OU draw prefix-matches (so the very first step's nav noise is identical), step 2 onward diverges because the per-step draw consumes the whole size-N array and the stream state then differs:

```
step1 first-k elements equal (prefix match):                       True   # phase-0 finalQ matches
step2 first-k elements equal after consuming different N:          False  # phases 1-3 diverge
```

### 1.4 The episode inherits that v/u state (no nav reset) → trajectory diverges

The episode loop (`g11_bg_runner.py:6535`) does **not** reset the nav slice's v/u at the start — it begins from whatever `finalize_conv_for_nav_gate` left. Confirmed there is **no active per-step RNG consumer** during the scored episode (all 13 RNG sites in the loop body are gated): OU off (`enable_ou_process=False`), conductance-noise off (`enable_conductance_noise=False`, `:4699`), structural plasticity off (`enable_structural_plasticity=False`, `:4701`), homeostasis off (`enable_homeostasis=False`, `:4674`), heterogeneity off (`:4700`). So the episode is a **deterministic function of (fixed params, v/u start-state)**; fixed params are N-independent (§1.1-1.2), so the ONLY thing that differs is the v/u start-state from the OU-on parser train → the observed Δ.

This is the textbook **noise-driven trajectory-divergence** signature (phase-0 identical, later phases diverge), confirming the diagnosis's "NOT a functional change" conclusion — and pinning it to the v/u start-state, not the OU per-step draw during scoring.

### 1.5 Why 896 (step-2b) was byte-identical but 24,051 (onebrain) is not

The documented step-2b "nav-not-regressed = 2.0 byte-identical" (`2026-06-10-step2b-rf-composer-coresident-COMPLETE.md`, gate 3) compared **`co_resident_rf=True` (rf=896) vs `co_resident_rf=False` (NO rf region)** via the SAME `_nav_gate_merged_run.py` harness, which ALSO uses `finalize_conv_for_nav_gate` + `build_with_ou=True` (`_nav_gate_merged_run.py:143-151`). So the regime is the same — yet it scored byte-identical. The reconciliation:

- The score metric is **coarse** (sum of 4 per-phase `final_quarter_mean_distance` on a discrete grid). The v/u start-state offset from the OU-on parser train is **proportional to the trajectory-divergence amplitude, which grows with how different N is**. The 896-vs-0 N change (ΔN ≈ +896, i.e. 8424 → ~9320, ~+11%) produced a v/u perturbation small enough that the *quantized* phase-distance score didn't flip off `[0.4956, 0.5044, 0.4956, 0.5044]` = 2.0. The 24,051-vs-896 change (ΔN = +19,549, N 8424 → 27,973, **+232%**) produces a much larger stream-state + per-step divergence, enough to move the coarse score by 0.62.
- **Caveat on the precedent:** the step-2b "byte-identical" was therefore likely a *small-ΔN* coincidence at the score's quantization granularity, NOT a guarantee that "a disjoint slice never perturbs the v/u-seeded trajectory." The premise (a disjoint out-edge-free slice is *functionally/synaptically* inert) is TRUE; the score byte-identity is an artifact of small ΔN + a coarse metric. This is the methodological flaw the diagnosis flagged, now precisely located.

---

## 2. DECOUPLING FEASIBILITY — RUNNER-SIDE vs sim/ (deliverable 2)

**A runner-side decoupling exists and is the cleanest path. NO `sim/` edit required.**

The coupler is entirely the **v/u start-state** the OU-on parser train leaves on the nav slice (§1.3-1.4). Make that start-state N-independent and the OU-off episode is byte-identical. Two runner-only options, both editing only `nav_conv_merged_bridge.py`:

### Option 2A (preferred): reset the nav slice's v/u to its resting baseline after the parser train

At the END of `finalize_conv_for_nav_gate` (`nav_conv_merged_bridge.py:1366-1369`, after the OU-on train, before `return`), reset the **nav** neurons' v/u to the resting `vr` baseline — the exact pattern the episode setup already uses (`g11_bg_runner.py`: `bridge.cp_membrane_potential_v[idx] = bridge.cp_izh_vr[idx]; bridge.cp_recovery_variable_u[idx] = 0.0`). Concretely: build the nav-index set = all region indices MINUS the conv slices (`parse_conj`, `parse_role`, `cortex_ctx`, `dlpfc_wm`, `rf`, `drive_*`), then:

```python
# (illustrative — runner-level, nav_conv_merged_bridge.py inside finalize_conv_for_nav_gate, before return)
nav_idx = <all indices not in the conv/rf/drive slices>          # cupy int64
bridge.cp_membrane_potential_v[nav_idx] = bridge.cp_izh_vr[nav_idx]
bridge.cp_recovery_variable_u[nav_idx]  = bridge.cp_izh_b[nav_idx] * 0.0   # u = b*(v−vr) = 0 at rest
```

- **Why this gives Δ=0:** nav fixed params are N-independent (§1.1-1.2); after this reset the v/u start-state is `vr` (model-determined, N-independent); the episode runs OU-off with no per-step RNG (§1.4) ⇒ the trajectory is byte-identical between the two arms.
- **Why it's safe / not a "cheat":** resetting to the resting baseline before an episode is standard practice (the episode setup already does it for the agent/SC slices). The parser train's PURPOSE is to learn the parser weights (a plasticity effect, preserved); its *side-effect* on nav v/u is incidental membrane noise that the episode should not inherit. This is a methodological correction, not a result-altering hack.
- **Bonus:** it also makes the **deployed agent's** nav reproducibility N-independent (the agent's `build_merged_nav_conv_bridge` runs the same OU-on parser train, `build_merged_nav_conv_bridge` step 5). So onebrain-vs-rf nav becomes byte-identical for the agent too, not just the gate.

### Option 2B (alternative): run the parser train with OU OFF

Set `cc.enable_ou_process = False` for the `train_parser_on_slices` call inside `finalize_conv_for_nav_gate` (i.e. don't flip it on at `:1358`). Then no size-N per-step draw is consumed during training (no stream-state divergence), AND the nav v/u evolves identically (deterministic dynamics, identical fixed params). **Risk:** the parser-train comment (`:1351-1352`) says "OU=20 for the WTA role readout" — OU during training may be load-bearing for the PARSER's binding quality. Turning it off could degrade the parser (a conversational-capability regression), which would need its own re-validation. Option 2A is strictly safer because it preserves the parser train verbatim and only neutralizes the incidental nav side-effect.

### Why NOT a region-reorder or a per-step-draw change

- **Reordering rf** doesn't help: the OU draw is `randn(n_neurons)` over the FULL N regardless of where rf sits, and rf's presence changes N either way. The coupler is N, not rf's index position.
- **Seeding OU/het from an N-independent sub-stream** WOULD fix it generally, but that is a `sim/` edit (it changes `_initialize_ou_process_state` / the per-step draw to use a dedicated `cp.random.RandomState` of fixed sub-stream, or to draw per-slice). Not needed given Option 2A.

**Verdict for deliverable 2:** runner-side (Option 2A, a nav v/u reset in `finalize_conv_for_nav_gate`) is sufficient and clean; **no `sim/` edit is required**. A `sim/` RNG-substream change (§4c) is a more general but heavier alternative, not necessary here.

---

## 3. SAME-N FUNCTIONAL-NEUTRALITY GATE DESIGN (deliverable 3)

The corrected gate must isolate the composer's **functional** effect on nav from the **N-noise** confound. Two designs; the first is the cleanest.

### Gate design 3A (recommended): same-N functional toggle (composer OFF vs ON at FIXED N)

Build ONE bridge at the **onebrain N** and run the nav episode twice, toggling only whether the composer is functionally exercised — never changing N:

1. Build `MergedNavConvAgent(seed, co_resident_composer=True, co_resident_composer_kind="onebrain")` (or equivalently `build_merged_nav_conv_bridge(..., onebrain_rf_size=CoResidentOneBrainComposer.n_total_for(...))`). **N is fixed at the onebrain size for BOTH legs.**
2. **Leg A (composer functionally OFF):** run the validated gate-2a nav episode WITHOUT touching the rf slice mid-episode (the deployed reality — no composer op runs during a nav episode). Record the per-phase nav score.
3. **Leg B (composer functionally ON):** run the SAME episode but interleave/precede composer ops (`hear` / `query_*`) so the rf slice is actually kicked + read. Record the score.
4. **Assert Δ = 0** (byte-identical). Because N is identical, the v/u-start-state confound is gone; any Δ would be a REAL synaptic/state leak from the composer into nav (which the Task-1 anti-cheat says cannot happen — rf has 0 `cp_connections` out-edges into nav, and `_zero_rf_v_u()` resets the rf slice each op so an op can't even leave residue that the next nav step reads through shared arrays). This directly tests "does running the composer perturb nav," which is the deployed question.

This is the gate that the diagnosis's "needed_to_actually_flip (b)" describes, made concrete. It does NOT compare two different N's at all, so it is immune to the §1 confound. **It is the FUNCTIONAL-neutrality gate** (matches the owner memory `feedback_validate_signal_by_its_function`).

### Gate design 3B (the decoupled version of the existing gate): keep PART 2's two-arm structure but add the nav v/u reset

If a same-N toggle is awkward to wire, apply Option 2A (the nav v/u reset in `finalize_conv_for_nav_gate`) and re-run the EXISTING `_closure1_optionA_gate3.py` PART 2 unchanged. With the v/u start-state made N-independent, the two arms (rf-size vs onebrain-size, both composer-free) become byte-identical → Δ=0 GO. This certifies "the rf-region SIZE does not perturb nav" — a weaker but still-valid claim (it confirms the disjoint slice is inert once the harness's incidental v/u side-effect is removed). 3A is stronger (it tests the composer's actual functional presence, not just its size); 3B is a smaller change to the existing driver.

**Recommendation:** run **3A** as the load-bearing flip gate (functional neutrality), and optionally 3B as the corroborating size-inertness check. Both should be multi-seed (≥3; the byte-identity is mechanically seed-independent, but the standing 6-seed rule for any non-null effect argues for 3 byte-identical seeds as conclusive for a true-null gate, mirroring the step-2a 3/6 byte-identical precedent).

---

## 4. RANKED RECOMMENDATION (deliverable 4)

### (a) — RECOMMENDED — Byte-identical flip via the runner-side v/u decoupling (Option 2A) + the same-N functional gate (3A)

- **What:** add the nav v/u reset to resting `vr` at the end of `finalize_conv_for_nav_gate` (`nav_conv_merged_bridge.py`, runner-only). Re-run the redesigned gate 3A (same-N composer OFF-vs-ON, Δ=0) AND optionally 3B (the existing two-arm gate, now byte-identical). On GO, flip `co_resident_composer_kind` default `rf`→`onebrain` (`nav_conv_merged_bridge.py:1638`) + ride Closure 3 (`enable_da_encoding_gain` default→True, `:1641`).
- **Trade-offs:** preserves the gate-2a/2b **byte-identity guarantee** (it RESTORES it for the size-difference case AND for the deployed agent's nav reproducibility); no `sim/` edit; small, well-precedented runner change (the v/u-reset-to-vr pattern is already used in the episode setup). The only cost is one extra runner change + a re-run of the (cheap) gate.
- **Why best:** it converts the HELD flip into a genuinely byte-identical flip, fixes the *root* methodological flaw (incidental v/u side-effect), and improves the deployed agent (N-independent nav). This is the path that satisfies the literal Δ=0 gate without overriding it.

### (b) — Flip-with-documented-noise-caveat (functional neutrality proven; standalone benchmark unaffected)

- **What:** accept that (i) the rf slice is array-disjoint from nav (0 out-edges; never stepped mid-episode), (ii) gate-3 PART 2 instantiates NO composer (the Δ is purely the rf-region-SIZE → N → v/u-start-state noise), and (iii) the documented **standalone** nav benchmark (`--readout-source motor`, CLI default) is a SEPARATE path that never builds the conversational rf region at all → totally unaffected by the merged composer. Flip the default with a findings note documenting the N-noise caveat.
- **Trade-offs:** **sacrifices the literal gate-2a/2b byte-identity guarantee** for the merged nav score (the merged-bridge nav trajectory becomes a different — equally valid — trajectory under onebrain vs rf, because of the v/u-start-state N-noise). Whether that matters: functionally it does NOT (no synaptic path; the deployed standalone benchmark is unchanged; the conversational no-confab moat is untouched). But it leaves the merged nav score non-reproducible across the composer kind, which is exactly the kind of "soft boundary" the SURPASS directive says to fix rather than accept when a cheap fix exists. Since (a) IS that cheap fix, (b) is dominated by (a).

### (c) — Scoped `sim/` RNG-substream change

- **What:** in `sim/bridge.py`, draw the OU per-step noise (`:6581`) and/or the size-N init draws (`:1132`, `:1461`) from a **dedicated, fixed sub-stream** seeded independently of N (e.g. a per-slice draw, or a `cp.random.RandomState(seed)` reserved for OU that always draws the same per-neuron sequence regardless of total N). This would make the OU/het N-independent globally (not just for nav).
- **Trade-offs:** a protected `sim/` edit (byte-level review required per the owner directive), broader blast radius (changes RNG semantics for EVERY bridge → would need a default-off/byte-identical guard + re-validation of unrelated determinism tests). General + principled, but heavier than (a) and not necessary because the coupler is the *episode v/u start-state*, which (a) fixes runner-side. Reserve (c) only if a future need for truly N-independent OU during a scored episode arises.

### (d) — Keep onebrain opt-in (status quo)

- **What:** leave `co_resident_composer_kind` default `rf`; onebrain remains opt-in.
- **Trade-offs:** zero risk, but does NOT advance the owner's "TRUE one brain / move everything onto the shared spiking substrate" headline for the MERGED agent, and leaves the build-fix's value unrealized at the default. Acceptable only as a fallback if (a)'s gate unexpectedly fails (it should not, per §1-2).

**Ranked:** **(a) ≫ (b) > (d) > (c)** for this specific flip. (a) is a cheap runner-side change that yields a genuinely byte-identical flip + improves the deployed agent; (b) is viable on the functional-neutrality argument but needlessly forfeits byte-identity that (a) recovers; (c) is over-engineered for this coupler; (d) is the do-nothing fallback.

---

## Appendix — load-bearing file:line index

| Claim | Location |
|---|---|
| Global RNG seeded once with `cfg.seed` | `sim/bridge.py:1053` (`_initialize_rng`), called `:1105` |
| Izhikevich external input = zeros (no draw) | `sim/bridge.py:1128` |
| First size-N CuPy draw = traits (randint) | `sim/bridge.py:1132` |
| Second size-N CuPy draw = firing thresholds (uniform) | `sim/bridge.py:1461` |
| Heterogeneity NOT applied in gate/episode path | `g11_bg_runner.py:4700` (`enable_parameter_heterogeneity=False`); gate `sim/bridge.py:1677-1678` |
| OU per-step draw = `randn(n_neurons)`, size = total N | `sim/bridge.py:6581`, gated by `enable_ou_process` `:6575` |
| Regions allocated in list order (contiguous slices) | `sim/regions.py:422-447` |
| Nav regions FIRST, extras (parser/dlPFC/rf) appended | `g11_bg_runner.py:4314`, `:4669-4670`; `nav_conv_merged_bridge.py:1306` |
| rf region has ZERO pathways (no out-edges) | `nav_conv_merged_bridge.py:1292-1293` |
| Parser train sets OU ON | `nav_conv_merged_bridge.py:1358` |
| Parser train steps full bridge ×21,600 | `nav_conv_merged_bridge.py:139-147` (30×6×120) |
| Hook restores OU OFF for the episode | `nav_conv_merged_bridge.py:1369` |
| Parser train is plasticity-gain-masked to parser slice | `nav_conv_merged_bridge.py:1353-1354` |
| Episode loop (no nav v/u reset at start) | `g11_bg_runner.py:6535` |
| Episode per-step RNG all gated off | `enable_ou_process` False `:1369`; conductance-noise `:4699`; structural `:4701`; homeostasis `:4674` |
| v/u-reset-to-vr pattern (available runner-side) | `g11_bg_runner.py` episode setup (`cp_membrane_potential_v[idx]=cp_izh_vr[idx]; cp_recovery_variable_u[idx]=0`) |
| step-2b "nav-not-regressed=2.0 byte-identical" precedent | `2026-06-10-step2b-rf-composer-coresident-COMPLETE.md` (gate 3); harness `_nav_gate_merged_run.py:143-151` |
| FLIP default-fixture test is conversational-only (no nav score) | `tests/test_nav_conv_step2b_coresident.py` |
| `n_total_for` (onebrain rf size) | `nav_conv_merged_bridge.py:1595` |
| Agent passes `onebrain_rf_size` to the builder | `nav_conv_merged_bridge.py:1864-1870` |

**Sanity-check snippets (numpy + cuRAND, run read-only):** `randn(N_big)[:k] == randn(N_small)[:k]` prefix-matches on step 1 (True), diverges on step 2 after consuming different N (False); cuRAND confirms the same prefix-match for `randn` and for a `uniform` following a size-N `randint`.
