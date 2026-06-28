# Knowledge-scaling resume-frontier scoping — make the longitudinal develop loop scale to 100s of concepts with good recall (2026-06-27)

**Type:** SCOPING (read-only). NO code/`sim`/GPU edit. A live develop run (the owner's CORPUS-64 3-day run, CYCLE 687) is using the GPU; this doc does NOT launch anything.

**Goal (owner, CYCLE 686 TOP RESUME-FRONTIER):** thread a bigger composer D through `build_agent -> MultiTurnAgent -> RFPhasorComposer`, budget the CONVERSE-agent VRAM (sparser agent / sharding / fast `cp_connections`-resume), and reach a genuine 100s-of-concepts, high-recall, multi-day-continuous-learning develop run on a 24GB 3090.

**Inputs (do NOT re-run):** the 3 GPU de-risks of CYCLE 685-686 already established: (a) recall caps ~72 concepts (composer D fixed ~128 crowds cleanup -> moat abstains; 200-cap collapsed at vocab 96); (b) the per-day CONVERSE agent's VRAM is ~quadratic in vocab (47840 neurons / 171M synapses @ 320 -> cupy OOM at the 80% pool, peak ~20.7GB); (c) `build_agent` builds `MultiTurnAgent` without passing D (composer uses its own default).

---

## 1. ISOLATE — the exact D source + the exact VRAM driver (with line refs)

### 1a. Where the composer's D comes from — it is **HARDCODED at 128, never threaded**
The chain that should carry D, traced end-to-end:

- `develop_gpu(..., D=128, ...)` (`_longitudinal_develop_loop_gpu.py:288`) threads `D` **only into `StreamCortex`** (`:334`, `StreamCortex(..., D=D, ...)`). `StreamCortex.D` sets the grounded-phasor dimension (`:151` default `D=128`; `:187` the random complex projection `proj:(D, n_hub)`; `:261-262` `grounded[w] = angle(proj @ code_row)/2pi`, a length-D phasor).
- The CONVERSE agent is built by `build_agent(full_vocab, seed, ..., referent_nouns=...)` (`_longitudinal_develop_loop_gpu.py:390`) -> `build_agent` (`_longitudinal_develop_loop.py:256`). **`build_agent` has NO `D` parameter.** It constructs `MultiTurnAgent(referent_concepts=refs, concepts=concepts, seed=seed, wm_n=wm_n, wm_pattern_size=40, enable_neural_render=..., composer_kind="rf", ...)` (`:277-280`) — **no D passed.**
- `MultiTurnAgent.__init__` (`multi_turn_agent.py:47`) — **no `D` parameter.** It builds `BrainConversationalAgent(seed=..., concepts=..., grounded_codes=..., composer_kind=..., ...)` (`:67-72`) — no D.
- `BrainConversationalAgent.__init__` (`brain_conversational_agent.py:175`) — **no `D` parameter.** The rf branch (`:260-277`) hardcodes `RFPhasorComposer(seed=seed, D=128, vocab=vocab, period=200, ...)`. The onebrain branch (`:254`) hardcodes `OneBrainComposer(seed=seed, D=128, ...)`. **Both pin D=128 as a literal.**
- `RFPhasorComposer.__init__` (`rf_phasor_composer.py:62`) DOES accept `D=64` (default) — but it is always called with the literal `128` from the agent, so the knob is unreachable from the develop loop.

**The kicker (`_inject_grounded`, `_longitudinal_develop_loop_gpu.py:469-482`):** the grounded codes are injected into the composer's concept table **only if `v.shape[0] == D`** (`:481`, where `D = comp.D`). So even if one raised `develop_gpu(D=2048)`, the StreamCortex would emit 2048-dim codes that `_inject_grounded` would then **silently DROP** against the composer's `D=128` — the brain would converse on the composer's *own random* codes, not its learned ones. **D must be made consistent across BOTH `StreamCortex` AND the composer**, or the grounded-code injection no-ops. This is the load-bearing isolation: the cap is not "D=128 is too small" alone — it is that the single `D` knob does not reach the composer at all.

### 1b. What makes the CONVERSE agent's VRAM ~quadratic — the **WM loop bridge**, not the composer
`MultiTurnAgent` builds a persistent discourse working-memory loop eagerly in `__init__` (`multi_turn_agent.py:86-87` -> `_build_wm` `:109-112`):

```
SpikingLoopContextBuffer(self.referents, n=self._wm_n, pattern_size=40, seed=..., enable_ou=False)
```

Note `_build_wm` does **NOT** pass `internal_density`, so `SpikingLoopContextBuffer` uses its **default `internal_density=0.1`** (`content_selection_spiking.py:175`). That flows to `build_loop_wm_bridge(n=wm_n, density=0.1, loop_density=0.05, ...)` (`csp.py:185-186`), which builds **two regions** (`cortex_ctx`, `dlpfc_wm`) each `n=wm_n` neurons with **`internal_density=0.1` recurrence** (`csp.py:78`) PLUS two cross-region pathways at **`loop_density=0.05`** (`csp.py:86-90`).

`build_agent` sizes the loop **quadratically in the referent count** (`_longitudinal_develop_loop.py:276`):

```
wm_n = max(600, 2 * pattern_size * len(refs))   # = 80 * len(refs)  (pattern_size=40)
```

So the WM bridge synapse count is dominated by:
- region recurrence: `2 regions x internal_density x wm_n^2 = 2 x 0.1 x (80R)^2 = 1280 R^2`
- cross pathways:    `2 paths   x loop_density   x wm_n^2 = 2 x 0.05 x (80R)^2 =  640 R^2`
- **WM bridge total ~= 1920 R^2 synapses**, where `R = len(referent_nouns)` (`csp.py:78` region edges scale `internal_density x n^2`; the framework builds dense-within-region random connectivity).

The referents are `full_vocab minus actions` (`_longitudinal_develop_loop.py:193-198`), so for a TinyStories corpus (verbs a minority) `R ~= 0.7-0.8 x V`. Substituting `R ~= 0.75 V`: **WM bridge ~= 1920 x (0.75 V)^2 ~= 1080 V^2 synapses.** That is the quadratic and it is the OOM:
- V=320 -> R~=240 -> wm_n~=19200/region -> WM ~= 1920 x 240^2 ~= **111M synapses** (the bulk of the reported 171M total; the rest is the per-op composer bridges + the cortex), peak ~20.7GB > the 20.6GB 80% pool -> `OutOfMemoryError in set_pathway_weights tocsr` (CYCLE 685 — exactly the WM CSR build, `csp.py:221-224`).
- V=64  -> R~=48  -> wm_n~=3840/region -> WM ~= 1920 x 48^2 ~= **4.4M synapses** (comfortably fits — the running CORPUS-64).

**The composer is NOT the quadratic.** `RFPhasorComposer` builds per-op RF bridges sized `2D` (unbind, `:301`), `2*K*D` (a K-fact batched scan, `:407`), and `D+V` (the spiking cleanup, `:347-350`), all cached by neuron-count in `_bridge_cache`/`_izh_bank_cache`. These scale with **D and the fact-count K and V (linear)**, not V^2. The numpy `kb` fast path (`:395`, `:544`) holds composites as host arrays (no bridge). At D=128, V=320, K~=few-hundred facts these are <1-2GB. **`content_selection_spiking`'s `SpikingLoopContextBuffer` (the dlpfc_wm cortico-loop) IS the quadratic-VRAM driver, via `build_agent`'s `wm_n = 80*R` sizing into `internal_density=0.1` x 2 regions.**

(Aside: `enable_biased_competition` defaults **OFF** in `build_agent`'s call (`_longitudinal_develop_loop.py:280` passes `enable_biased_competition=False`), so the second `BiasedCompetitionContextBuffer` — which would DOUBLE the WM cost — is never built. Good; no change needed there.)

---

## 2. QUANTIFY — the (vocab, D, VRAM) relationship + feasibility on 24GB

### 2a. What D ~200/~320 concepts need for recall
The project validated **320 concepts at D=2048** for the flat-distinct composition tier (CLAUDE.md, 2026-06-02; between-code cos mean 0.045). The conversational composer's documented production scale is **D=128 for the onebrain/rf path up to V=320 stream-learned codes** (CLAUDE.md onebrain-320 GO) — BUT that GO used the *stream-learned PPMI-normalized* codes whose familiarity-gap the moat reads cleanly, and it was the **OneBrainComposer with `integrated_loop`/spiking-cleanup tuned at 320**. The develop loop's de-risk used the **rf composer at D=128** and saw recall collapse at vocab ~96 — i.e. **D=128 is under-provisioned for the develop loop's per-day fresh-code regime at 100s of concepts.**

The governing quantity is the cleanup/abstention margin: the matched-filter score separation between the true concept and the nearest distractor scales ~`sqrt(D)` for random phasors (the FHRR capacity law), while the number of distractors grows with V. To hold a fixed false-accept margin as V grows, **D must grow at least ~linearly in V** in the worst (uncorrelated) case; the stream-learned codes' structure relaxes this but the de-risk shows 128 is already too small at ~96. A safe target by analogy to the validated tiers:

| concepts (V) | D for clean recall (target) | basis |
|---|---|---|
| ~72  | 128 (current) | de-risk: holds to 72, collapses by 96 |
| ~128 | 256 | ~2x headroom over the 96 collapse |
| ~200 | 512 | linear-in-V extrapolation + margin |
| ~320 | 1024-2048 | the validated flat-distinct tier used 2048 |

### 2b. CONVERSE-agent VRAM at (V, D)
VRAM is the **sum of two independent terms**:
- **WM bridge (quadratic in V, INDEPENDENT of D):** `~1080 V^2` synapses x ~(CSR: 1 float32 weight + 1 int32 col-index + delay/state overhead ~ 16-24 bytes/synapse effective on this bridge) -> the empirical anchor is V=320 -> ~111M syn -> ~17-20GB-dominant. **This term does NOT shrink by changing D.** It is the binding constraint at 100s of concepts.
- **Composer (linear in D, K, V):** per-op RF bridges `2KD`/`D+V` complex weights + the numpy `kb`. At D=512, V=320, K~=300 facts: the scan bridge `2*300*512 ~= 307K` neurons' complex weights built fresh per op — note this is *per-op transient*, freed/replaced each op (`:177` "builds the sparse complex weights FRESH each op -> replaces"), bounded by the largest single op (the K-batched scan ~ `2KD` complex entries ~ a few hundred MB at D=512). **Raising D 128->2048 (16x) inflates the composer term ~16x but it starts small (~1-2GB) -> ~tens of GB only at D=2048 with large K;** at D=512 it stays a few GB.

**Putting it together (24GB budget, ~19GB usable at the 80% pool):**

| V | D | WM bridge (V^2) | composer (D,K) | total est. | fits 24GB? |
|---|---|---|---|---|---|
| 64  | 128 | ~4.4M  / <2GB | <1GB | **~6-8GB** | YES (the running run) |
| 128 | 256 | ~17.7M / ~4-5GB | ~1GB | **~6-8GB** | YES |
| 200 | 512 | ~43M / ~9-11GB | ~2-3GB | **~13-16GB** | YES (tight; CYCLE 685 saw 17.9GB at V=200/D=128) |
| 320 | 512 | ~111M / ~18-20GB | ~3GB | **~22-24GB** | **NO** (WM alone OOMs) |
| 320 | 1024 | ~111M / ~18-20GB | ~6GB | **>26GB** | NO |

**Verdict on feasibility:** 100s-of-concepts high-recall is feasible on 24GB **only if the WM-bridge quadratic is killed** (a sparser / smaller WM, OR a non-multi-turn agent for the develop probe). With the WM as-is, the ceiling is **~200 concepts** (matching the CYCLE-685 observation). **D-threading alone does NOT unlock 320** — D doesn't touch the WM quadratic; it only fixes the *recall* axis. Both fixes are needed together: D-threading for recall, WM-shrink for VRAM.

---

## 3. RANK — cheap-first options

**(a) Thread a bigger D through `build_agent -> MultiTurnAgent -> RFPhasorComposer` (+ match the StreamCortex D).** THE MINIMAL recall fix. Add a `D=128` kwarg to `build_agent` (`_longitudinal_develop_loop.py:256`), pass it to `MultiTurnAgent(D=...)`; add `D` to `MultiTurnAgent.__init__` (`multi_turn_agent.py:47`) and pass to `BrainConversationalAgent(D=...)`; add `D` to `BrainConversationalAgent.__init__` (`brain_conversational_agent.py:175`) and substitute it for the two literal `128`s (`:254`, `:273`). In `develop_gpu`, pass the SAME `D` to `build_agent` that is passed to `StreamCortex` (`:334`, `:390`) so `_inject_grounded`'s `v.shape[0]==D` guard (`:481`) passes. **Cost:** ~4 one-line signature threads + 2 literal substitutions, NO `sim/` edit, default `D=128` = byte-identical. **Unlocks:** clean recall at V up to ~200 (where VRAM still fits). **De-risk:** a CPU `SIM_BACKEND=numpy` run at tiny vocab confirming `_inject_grounded` now injects (codes present in `comp.concepts` post-build), then a single GPU day at V~=128/D=256 confirming recall rises above the 72-cap and moat stays 0-FA.

**(b) Kill the WM-bridge quadratic.** THE VRAM fix; required for >200 concepts. Three sub-options, cheapest first:
  - **(b1) Use a NON-multi-turn agent for the develop probe.** `build_agent` already supports `use_multiturn=False` -> a plain `BrainConversationalAgent` with **no WM loop at all** (`_longitudinal_develop_loop.py:281-283`). The develop loop's per-day battery is who/what recall + heldout + retention + chain + yes-no + moat — **none of these needs cross-turn anaphora** (the WM loop only resolves pronouns across turns). The multi-hop `chain` runs on the composer's `query_chain`, not the WM. **So the WM loop is dead weight for the develop probe.** Flipping `develop_gpu`'s `build_agent(..., use_multiturn=True)` (`:390`) to `use_multiturn=False` removes the ENTIRE `~1080 V^2` term. **Cost:** one kwarg flip (or a `develop_gpu(use_multiturn=False)` knob), NO `sim/` edit. **Unlocks:** V=320 at D=512 drops from ~22-24GB to ~3GB composer-only -> fits with huge headroom; the only loss is cross-turn pronoun resolution, which the per-day metrics never test. **This is the single highest-leverage cheap fix** — it removes the binding constraint at near-zero cost. **De-risk:** a GPU day at V=320/D=512 `use_multiturn=False` confirming the per-day battery (recall/heldout/retain/chain/moat) is unchanged vs multi-turn at a vocab where both fit (e.g. V=64), then confirming V=320 now fits + recall is clean.
  - **(b2) If multi-turn is wanted later: shrink the WM sizing.** Pass `internal_density=0.0` through `_build_wm` (`multi_turn_agent.py:111` — the VALIDATED clean-WM config per `csp.py:298,305` is `internal_density=0 + enable_ou=False`; the loop's attractors are INSTALLED outer-products `csp.py:221-224`, the random recurrence is NOT load-bearing and `csp.py:182` even says `internal_density=0 -> less cross-talk`). That removes the `1280 V^2` region-recurrence term (the bigger half), leaving only `640 V^2` cross-loop. Also reduce `wm_n` headroom from `2x` to `1.2x` (`_longitudinal_develop_loop.py:276`) — the loop capacity is `wm_n/pattern_size` patterns, `1.2x` still holds every referent. Combined ~`~250 V^2` (4x smaller). **Cost:** 2 one-line changes, NO `sim/` edit, and `internal_density=0` is the documented-VALIDATED config (a strict improvement, not a regression). **De-risk:** the existing `tests/test_multi_turn_agent.py` must pass verbatim with `internal_density=0`.
  - **(b3) Reuse ONE WM bridge across days** (the develop loop rebuilds the agent — and its WM — every day, `:390`). Lower priority; (b1) makes it moot for the probe.

**(c) Wire the fast `cp_connections`-resume.** Orthogonal to VRAM/D — it is a **wall-clock** fix for the resume RE-HEAR cost (`_longitudinal_develop_loop_gpu.py:342-348`, `:795`: resume currently re-hears the cumulative vocab to re-acquire codes). At 100s of concepts the cumulative re-hear is minutes-to-tens-of-minutes per resume (the CYCLE-685 "pause sparingly" caveat). Persisting `StreamCortex.bridge.cp_connections` (the learned hub->target weights) in the lineage `.h5` and reloading them would make resume O(load) not O(re-hear). **Cost:** a real build (a `cp_connections` save/load in the lineage payload, ~the `synapse_storage`/`lineage.export_shards` machinery already exists). **NOT on the recall/VRAM critical path** — defer until after (a)+(b) land; needed only if the owner pauses a long 100s-concept run often.

**(d) Shard the composer across vocab blocks.** Split the V concepts into B blocks of V/B, each with its own composer (D fixed), route a query to its block. **Cost:** a substantial build (block routing + cross-block facts). **Only needed if D must stay tiny for the composer-VRAM term AND V is very large (>500)** — the (b1) WM-kill already gives V=320 at D=512 with headroom, so sharding is **not needed for "100s of concepts"** on 24GB. Defer; it is the lever for the *thousands*-of-concepts horizon, not 100s.

**The combination that unlocks 100s-of-concepts high-recall on 24GB, cheapest-first: (b1) + (a).**
- (b1) removes the `V^2` WM term (the OOM) at ~zero cost — V=320 now fits in a few GB.
- (a) threads D so the composer has the dimension recall needs (D~=512 for ~320) and `_inject_grounded` actually injects the learned codes.
- Together: ~5 one-line signature threads + 2 literal substitutions + 1 kwarg flip, NO `sim/` edit, all default-preserving (default `D=128`, default `use_multiturn` unchanged for non-develop callers — only `develop_gpu` opts into `use_multiturn=False` + a bigger D).

---

## 4. VERDICT — recommended cheap-first build SEQUENCE, per-step de-risk, realistic 24GB ceiling

**Build sequence (cheapest-first; each step independently shippable + default-preserving):**

1. **Step 1 — WM-kill for the develop probe (b1).** Add `use_multiturn` (default True) to `develop_gpu`'s `build_agent` call site (`_longitudinal_develop_loop_gpu.py:390`) and set it `False` for the develop loop (the per-day battery needs no cross-turn anaphora). **De-risk:** GPU day at V=64 — confirm the per-day metrics (recall/heldout/retain/chain/moat) are identical to the current `use_multiturn=True` run (no regression); then a GPU day at V=320/D=128 — confirm it now FITS (no OOM) where the multi-turn agent OOM'd. **This alone lifts the VRAM ceiling from ~200 to V=320+.**

2. **Step 2 — thread D (a).** Add a `D` kwarg through `build_agent -> MultiTurnAgent -> BrainConversationalAgent`, substitute the two literal `128`s (`bca.py:254,273`), and pass `develop_gpu`'s `D` to BOTH `StreamCortex` AND `build_agent` so `_inject_grounded` injects. **De-risk:** CPU numpy tiny-vocab — assert `comp.concepts[w]` equals the injected grounded code (not the composer's random default) for a heard `w` (proves the D match + injection); then GPU day at V=128/D=256 and V=200/D=512 — confirm recall rises above the 72-cap and moat stays 0-FA. Default `D=128` keeps every other caller byte-identical.

3. **Step 3 (optional, only if multi-turn is later wanted at scale) — WM-shrink (b2).** `internal_density=0.0` through `_build_wm` + `1.2x` `wm_n` headroom. **De-risk:** `tests/test_multi_turn_agent.py` passes verbatim. Makes the multi-turn agent itself fit at 100s of concepts (if a future capstone needs cross-turn anaphora in the develop loop).

4. **Step 4 (defer; wall-clock only) — fast `cp_connections`-resume (c).** Persist/reload `StreamCortex.bridge.cp_connections` in the lineage instead of re-hearing. **De-risk:** a save->reload->one-query round-trip equals a re-hear->one-query (codes match). Only build if the owner pauses long runs often.

**Realistic concept ceiling on a 24GB 3090:**
- **With Step 1 + Step 2 (the recommended cheap combo):** the WM quadratic is gone, so VRAM is composer-only (linear in D, K). **High-recall ~300-400 concepts is feasible at D=512-1024** (composer ~3-6GB, huge headroom). The binding constraint becomes the **composer's recall margin at fixed D** — which D-threading directly addresses, and per the validated flat-distinct tier, **320 at D=512-1024 is well within reach.** The practical ceiling on 24GB is set by the per-op scan bridge `2KD` at large fact-count K and the wall-clock of fresh-code re-learning per day, NOT by VRAM — comfortably **300-500 concepts.**
- **Without Step 1 (multi-turn WM left on):** ceiling stays **~200 concepts** (the CYCLE-685 wall) regardless of D — the `~1080 V^2` WM term OOMs at 320.
- **Thousands of concepts:** needs (d) sharding OR a bigger GPU — out of scope for "100s", a separate horizon.

**Net:** the owner's framing ("thread a bigger D + budget VRAM") is right, and the budget lever is cheaper than expected — **the dominant VRAM cost (the multi-turn WM loop) is dead weight for the develop probe and can be dropped with one kwarg.** D-threading is then the only other change and is a handful of signature lines. No `sim/` edit, no sharding, no GPU run required for this scoping. The single decisive insight: **D-threading fixes RECALL but not VRAM; the WM-kill fixes VRAM but not recall; you need BOTH, and both are cheap.**

---

## Files / line refs (all absolute under `E:\Documents\Projects\sim`)
- `research/runners/_longitudinal_develop_loop.py:256` — `build_agent` (NO D param; `wm_n=80*R` at `:276`; `referent_nouns` at `:193`)
- `research/runners/_longitudinal_develop_loop_gpu.py:288` — `develop_gpu(D=128)` (threads D to StreamCortex `:334` only); `:390` `build_agent` call; `:469` `_inject_grounded` (`v.shape[0]==D` guard `:481`); `:342`/`:795` re-hear resume seam
- `research/runners/multi_turn_agent.py:47` — `MultiTurnAgent.__init__` (NO D param); `:111` `_build_wm` (no `internal_density` -> default 0.1)
- `research/runners/brain_conversational_agent.py:175` — `BrainConversationalAgent.__init__` (NO D param); `:254` `OneBrainComposer(D=128)`; `:273` `RFPhasorComposer(D=128)` — the two hardcoded literals
- `research/runners/rf_phasor_composer.py:62` — `RFPhasorComposer.__init__(D=64)` (the unreachable knob); `:301`/`:347`/`:407` per-op bridges (`2D`/`D+V`/`2KD`, NOT V^2)
- `research/runners/content_selection_spiking.py:66` — `build_loop_wm_bridge` (2 regions x `internal_density` x n^2 + 2 x `loop_density` x n^2); `:164` `SpikingLoopContextBuffer` (default `internal_density=0.1`; validated clean config is `0.0` per `:298,305`)
