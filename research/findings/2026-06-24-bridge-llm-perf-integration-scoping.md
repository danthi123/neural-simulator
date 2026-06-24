# Deep-research gate — bridge-co-resident LLM: PERF (Phase 2A) + FUNCTIONAL INTEGRATION (Phase 3D) scoping (2026-06-24)

> **Read-only research gate** (no edits, no GPU, no watcher loops). Scopes the two open frontiers for the
> bridge-co-resident grounded-language faculty: **(A) PERF — make it FAST** (roadmap Phase 2A; inventory
> O-1/O-2/O-3 + H-4) and **(B) FUNCTIONAL INTEGRATION — make the faculty INTERACT with the brain on ONE
> substrate** (roadmap Phase 3D; inventory I-2/I-3/H-1 + H-4 on-chip representation).
>
> **Builds on (does NOT re-derive):** bridge co-residence DEMONSTRATED (24-layer Qwen2.5-0.5B on the live RF
> substrate, bit-exact, ppl 7.041 == B-1, coherent, **14.05 GB resident, LOCAL** — `2026-06-23-bridge-coresidence-DEMONSTRATED.md`);
> the perf de-risk (`2026-06-23-bridge-coresidence-perf-dense-matvec-GO-WITH-CAVEAT.md`: dense matvec
> **bit-exact ~3600–13000×/shape**, but **97% of the wall-clock SHIFTS to the host nonlinearities/attention/H↔D**
> once the matvec is cheap → measured **8.8 tok/s**); the grounded-language faculty P1/P2/P3 COMPLETE
> (`_grounded_lang_integration_derisk.py`, off-bridge Qwen, the GATE→CONSTRAIN→VERIFY moat holds with a real LLM).

---

## TL;DR (the bottom line up front)

- **PERF is a known, mostly-engineering chain. The proven dense-matvec lever (O-2) is NECESSARY-NOT-SUFFICIENT
  — the real usability win is O-1 (keep the WHOLE forward on-GPU).** Build order = **O-1 (on-GPU forward, 97%
  of the wall, NO `sim/` edit) → O-3 (KV cache, the generation lever) → O-2-purity (the optional default-off
  `cfg.rf_dense_weights`, on-bridge-purity only)**. (The dense GEMM that O-2 "lands" is already host-computable
  with NO `sim/` edit — the host forward calls `A @ W_dense` directly; the `sim/` edit only buys on-bridge
  *purity*, not throughput, so it is LAST + optional.) **Projected: prefill ~7200 tok/s, generation
  ~330 tok/s** (matvec ceiling); the on-GPU-forward + KV-cache close most of the gap to that ceiling. **All
  LOCAL** (no VRAM wall; cloud NOT triggered).
- **INTEGRATION's headline: the GATE→CONSTRAIN→VERIFY loop ALREADY EXISTS as a working host pipeline
  (`brain_chat_tui.py:ChatBrain`), and the brain's own GATE/VERIFY OPS are ALREADY on the bridge** (the
  spiking recall `what_does` + the `BridgeParser.parse` re-parse). "Functional, on-one-substrate" therefore is
  NOT a new mechanism — it is **(1) running the faculty's forward on the SAME bridge** (co-residence
  DEMONSTRATED → it is the O-1/O-2 forward, computed via the bridge's own RF read with `rf_dense_weights`) and
  **(2) making the three loop DECISIONS spiking-on-the-bridge** instead of host `==`/orchestration (the moat
  `==`, the matched-filter routing, the abstain gate — all already have validated spiking replacements built
  for the conversational composer: `enable_spiking_cleanup` WTA, the Bogacz-Brown familiarity gate, the
  `integrated_loop` K-way sequencer). **⇒ the integration is a CONSOLIDATION + a default-flip arc, gated on the
  perf work (the faculty's forward must be on-bridge + fast first), NOT a from-scratch build.**
- **On-chip representation (H-4)** rides for free on O-2: the dense-fp16 weight (**0.99 GB**, vs the as-is
  float64 complex-CSR **11.86 GB** = 12× the fp16 ANN size) IS the natural neuromorphic form AND the O-2 lever.
  The cheap pre-step (drop the all-zero imaginary CSR → 5.93 GB; f32 → 3.95 GB) is roadmap item **1C** (Phase 1).

---

# PART A — PERF (Phase 2A): the concrete O-1 / O-2 / O-3 build sequence

## The diagnosis (from the two findings, not re-derived)

The full 24-layer forward on the RF bridge is **0.786 tok/s prefill / 161 s per generated token**. The profile
(`_bridge_cores_perf_derisk.py`) localizes the cost precisely:

1. **The per-row RF matvec is a cuSPARSE complex-CSR matvec over a DENSE 494M weight** — the WRONG storage
   (Qwen layers are 100% dense). The resonate megakernel already fused the per-step launches (0.42 ms/step +
   0.10 ms fixed = only 3% launch overhead), so the wall is the **CSR-on-dense gather**, not launch overhead.
2. The dense cuBLAS GEMM `a@W_dense` is the SAME math (`Re(Z)/nsteps = a@W`, verified max-err 1.3e-7) and
   **bit-exact + ~3600–13000×/shape** (lm_head ≈9000×). **But once the linears are cheap, the MEASURED
   end-to-end is only 8.8 tok/s** because **97% of the forward is now the host (numpy) graded nonlinearities
   (RMSNorm/SiLU/softmax) + attention + RoPE + ~216 per-linear device↔host copies/token.**
3. **Generation re-runs the full forward over the growing context** every token (161 s/tok) — a KV cache
   collapses that to O(1)/token.

⇒ three levers, with a hard ordering implication: **O-2 (dense matvec) is necessary but the dense GEMM is
already host-computable, so the REAL usability lever is O-1 (the whole forward on-GPU). O-3 (KV cache) is the
generation-specific lever.**

## The build sequence + per-step de-risk

### O-1 — the on-GPU LLM forward (FIRST; the 97% wall; NO `sim/` edit)

**What:** keep the entire host forward on the GPU between matvecs — port the graded RMSNorm/SiLU/softmax reads
to cupy, run attention + RoPE on-GPU, and **eliminate the per-linear H↔D round-trips** (only the final logits
are read to host). The dense GEMM (the O-2 math) is folded in here as `cupy A @ W_dense` for the dense linears.

- **Reuse-by-import surface:** the de-risk #3 / #2 `layer_forward` / `run_attention` / `graded_rmsnorm` are the
  numpy reference; the port is host-forward (the graded ops → cupy elementwise + the megakernel-style pool
  reads). The B-1 graded-nonlinearity calibration (SiLU/exp banks, pool budgets T) is already validated
  (`_grounded_lang_p1b_stepB1_forward_derisk` + `_bridge_cores_layer_derisk.build_host_banks`).
- **`sim/`-edit flag: NONE.** This is purely a host-forward change (the graded ops staying on-device). It is
  the actual usability work and it is *not* a bridge edit.
- **De-risk:** (1) **bit-faithful** — the on-GPU graded forward logits must match the de-risk #3 host-numpy
  graded forward to f32 precision (cos ≥ 0.9999, argmax-agree 1.0 on the held-out ppl slice; the graded ops
  are the SAME math, just on-device); (2) **speedup bar** — end-to-end ≥ **100 tok/s prefill** (the brief's
  "usable >10 tok/s" with headroom; the matvec-only ceiling is ~7200, so the residual host overhead must drop
  from 97% to a small fraction); (3) **ppl unchanged** vs de-risk #3 (7.041) — the on-GPU port must not
  perturb the graded read fidelity.
- **Cheap-first probe (no GPU needed to scope, GPU to run):** extend `_bridge_cores_perf_derisk.measure_end_to_end`
  to keep the activation resident (no per-linear `cp.asnumpy`) + cupy graded ops, and re-measure the
  rest-vs-linears split — the de-risk already isolates the 97%/3% breakdown, so the probe is "does on-GPU
  nonlinearity + no-D→H drop the 97% rest term."

### O-2 — the dense-matvec storage lever (the PROVEN math; purity `sim/` edit is OPTIONAL + LAST)

**What:** the dense weight GEMM. **The host-forward version needs NO `sim/` edit** (it is the `A @ W_dense` call
inside O-1, bypassing the RF matvec for the dense linears) — and that already achieves the throughput. The
**ON-BRIDGE-PURITY version** (so the bridge's OWN RF read is the fast path, for the "one brain" claim) is the
optional, default-off `sim/` edit.

- **The `sim/` edit (OPTIONAL, default-off, byte-review):** `cfg.rf_dense_weights` + a stored dense
  `cp_rf_w_dense` (fp16/fp32), read in `rf_resonate_steps` / `_rf_advance_one` / `_rf_resonate_steps_megakernel`
  via a GEMM (`Re(Z)/nsteps = a @ W_dense`) when the flag is set; **DEFAULT-OFF = the byte-identical CSR path**
  (the composer's sparse O(D) bind/unbind is unaffected — its weights are genuinely sparse). The edit surface is
  the three RF-advance paths in `sim/bridge.py:5710-5856`; the guard mirrors the existing `enable_rf_cudagraph`
  short-circuit pattern (a `getattr(cfg, "rf_dense_weights", False)` branch before the CSR matvec).
- **De-risk:** (1) **bit-faithful** — `a @ W_dense` (f64) == `a@W` to numerical roundoff (max-err <1e-9, already
  shown 7.5e-15); the f32 dense vs the f32 CSR-RF membrane read agree to 3.2e-6 (f32 precision); (2) **speedup
  bar** — ≥ 1000× per shape over the per-row CSR loop (already 3600–13000×); (3) **default-off byte-identity** —
  with `rf_dense_weights=False`, the existing RF tests (`test_rf_megakernel.py`, `test_one_brain_composer_agent.py`,
  the composer bind/unbind suite) pass VERBATIM (the composer must be untouched).
- **Why LAST + optional:** the runner-level dense GEMM (in O-1) already gets the throughput; the `sim/` edit
  only matters for the "the bridge's own RF read is fast" purity claim. It is precisely scoped + cheap, but it
  is NOT on the critical path to usability.

### O-3 — the KV cache (the generation lever; cheap-ish; NO `sim/` edit)

**What:** cache per-layer K/V across autoregressive steps so each generated token's forward is O(1) over the
new token, not O(context) over the whole sequence (161 s/tok → ~1/context of that). This is the single biggest
**generation**-throughput lever (prefill is already fast once O-1 lands).

- **Reuse-by-import surface:** the de-risk #3 generation loop (`rf_full_forward` per new token over the growing
  `cur` list) is where the recompute lives; the cache is a host-forward change to the attention path (store
  `k_proj`/`v_proj` outputs per position, append the new position, attend over the cache). HuggingFace's own
  `DynamicCache` is the reference contract; here it is the on-GPU graded forward's K/V tensors.
- **`sim/`-edit flag: NONE** (host-forward attention change).
- **De-risk:** (1) **bit-faithful** — cached-generation logits == full-recompute logits per step to f32 (the
  cache is an algebraic identity, not an approximation; argmax-agree 1.0 so the greedy generation is
  byte-identical to the de-risk #3 "Once upon a time..."); (2) **speedup bar** — generation ≥ **30 tok/s** (with
  O-1's on-GPU forward; the matvec-only gen ceiling is ~330, the cache removes the O(context) blow-up);
  (3) **coherence** — the generation text matches the no-cache reference (token-agree 1.0).

## H-4 (on-chip representation) rides on O-2

The dense-fp16 weight form is BOTH the O-2 lever AND the natural neuromorphic representation:

| storage | size (494M Qwen) | note |
|---|---|---|
| as-is: two complex CSRs, **float64** data + int32 idx | **11.86 GB** | 12× the fp16 ANN; the all-zero im CSR is pure waste |
| drop the all-zero imaginary CSR | 5.93 GB | **roadmap 1C (Phase 1, cheap `sim/` edit, byte-review)** |
| + f32 data | 3.95 GB | 1C stretch |
| **dense-fp16** | **0.99 GB** | **= the O-2 lever = the natural chip form** |

The cheap pre-step (1C) is a Phase-1 item (drop the im CSR + f32). The dense-fp16 is the O-2 `rf_dense_weights`
mode. ⇒ closing the storage waste advances the **VRAM**, **perf**, AND **neuromorphic-port** axes in one move,
exactly as the inventory's strategic synthesis root #1 says.

## PERF `sim/`-edit summary

| lever | `sim/` edit? | on critical path? |
|---|---|---|
| **O-1 on-GPU forward** | **NONE** (host-forward) | **YES — the 97% wall, the real usability win** |
| **O-2 dense matvec (host)** | **NONE** (`A @ W_dense` in the host forward) | YES (folds into O-1) |
| O-2 purity (`cfg.rf_dense_weights` + `cp_rf_w_dense`) | OPTIONAL default-off, byte-review | NO (on-bridge purity only) |
| **O-3 KV cache** | **NONE** (host-forward attention) | YES — the generation lever |
| H-4 1C (drop im CSR + f32) | cheap default-off→on, byte-review | Phase-1 storage (separate) |

---

# PART B — FUNCTIONAL INTEGRATION (Phase 3D): the faculty's fluency gated by the brain's grounding, on ONE substrate

## What "co-RESIDENT but not INTERACTING" concretely is, and what "functional on-one-substrate" means

**Co-residence DEMONSTRATED** = the 24-layer Qwen forward RUNS on the live RF bridge (its weights are RF
complex synapses; its matvec is the bridge's RF read). That is *spatial* co-residence — the faculty shares the
substrate, but the integration runs (`_grounded_lang_integration_derisk.py`) still use the **off-bridge PyTorch
Qwen** (I-2), and the GATE→CONSTRAIN→VERIFY loop is **host Python orchestration** (I-3).

**The decisive finding for this gate:** the GATE→CONSTRAIN→VERIFY loop **already exists as a working pipeline**
in `research/runners/brain_chat_tui.py` (`ChatBrain`), and **two of its three layers' OPS are already on the
bridge:**

- **GATE** (`ChatBrain.gate`): `self.router.match_fact(...)` is host (the cue-match scan, = inventory C-2),
  but the load-bearing decision — `recalled = self.inner.what_does(a, v); if recalled == p` — is the
  **brain's SPIKING recall** (`RFPhasorComposer`/`OneBrainComposer` on-bridge unbind+cleanup). The abstain is
  a host `==` on the spiking recall's output.
- **CONSTRAIN** (`ChatBrain.render` → `renderer.render_svo`): the faculty forward — **today off-bridge Qwen**;
  on-bridge = the co-residence forward (PART A).
- **VERIFY** (`ChatBrain._verify`): `_extract_svo_from_prose(...)` (host content extraction) **then
  `self.inner.parse(...)`** — the **brain's BridgeParser, ON the bridge** — re-assigns roles; the final
  `rsvo == list(gate_svo)` is a host `==`.

⇒ **"functional, on-one-substrate" is therefore NOT a new mechanism.** It is two things:

1. **Run the CONSTRAIN faculty forward on the SAME bridge** (PART A: co-residence is DEMONSTRATED; the work is
   making it fast via O-1/O-2, then routing the loop's render through the bridge forward instead of off-bridge
   PyTorch). **This is the I-2/H-1 closure and it is GATED ON PART A** (the off-bridge faculty stays the
   reference until the on-bridge forward is fast enough to render in the loop).
2. **Make the three loop DECISIONS spiking-on-the-bridge** instead of host glue (I-3) — each already has a
   VALIDATED spiking replacement built for the conversational composer:
   - the **GATE cue-match + abstain** (`match_fact` + the `== p` moat) → the **Bogacz-Brown familiarity gate**
     (`2026-06-11-familiarity-gate-v320-GO.md`: matches the host abstain decision at V=320 multi-seed, zero
     moat-breaches) + the `integrated_loop` spiking K-way sequencer (`one_brain_composer.py`, K=32 GO) for the
     "which fact" routing;
   - the **CONSTRAIN→VERIFY cleanup/selection** argmax → `enable_spiking_cleanup` (Izhikevich NEF WTA,
     ==argmax, ON in the flagship 320 demo);
   - the **VERIFY `rsvo == gate_svo` match** → a spiking match/compare over the re-parsed roles (the same
     biased-competition the multi-referent work specified; the cheapest first version is the familiarity-gate
     agreement read).

## The cheapest-first de-risk for the on-bridge gate/verify (catalog-grounded)

**Biology (catalog-first):** the dual-stream language network (catalog **G.11** Hickok-Poeppel; **G.12** Broca
sensorimotor mapping / grammatical processing; **G.13** Wernicke auditory→semantic). The CONSTRAIN (fluent
production) maps to the **dorsal stream** (Broca); the GATE/VERIFY (meaning grounding + comprehension match)
maps to the **ventral stream** (Wernicke semantic interface) + the brain's existing parser/composer. The
"render then re-comprehend and check" loop IS the arcuate-fasciculus production↔comprehension cycle (a
conduction-aphasia lesion is exactly the repetition/self-monitoring break) — so the architecture is
biology-faithful, and the integration is making that cycle's decisions neural rather than host `==`.

**De-risk ladder (cheap-first, reuse-by-import, anti-cheated):**

- **B-i (GATE-on-bridge, cheapest):** replace the host `match_fact` + `recalled == p` abstain with the
  **familiarity-gate + spiking-cleanup** decision already validated for the composer. De-risk: at V=64→320,
  the on-bridge GATE's answer-vs-abstain agrees with the host gate (agreement ≥ the V=320 GO bar; **zero
  moat-breaches** — the no-confab moat is the hard invariant); anti-cheat = an untaught cue MUST abstain
  (the familiarity gate's whole point), a lesion of the familiarity read collapses to chance.
- **B-ii (VERIFY-match-on-bridge):** replace the final `rsvo == gate_svo` with a spiking compare over the
  re-parsed roles (the parse is ALREADY on-bridge; only the `==` is host). De-risk: an adversarial DRIFT (the
  integration's test (c) — the faculty steered to a wrong patient) is REJECTED by the spiking match
  (drift-caught == the host loop's), the moat holds; anti-cheat = a true render passes, a role-inverted render
  fails.
- **B-iii (CONSTRAIN-on-bridge):** route `render_svo` through the **co-residence bridge forward** (PART A) —
  **gated on O-1/O-2** (the off-bridge Qwen stays the reference until the on-bridge forward renders fast
  enough). De-risk: the on-bridge-rendered sentence == the off-bridge faculty's (token-agree, already
  byte-identical at the forward level per de-risk #3); the end-to-end grounded/untaught/drift matrix
  (the integration runner's a/b/c) reproduces GO on the **one-substrate** path.
- **B-iv (host-glue residual):** the remaining `_extract_svo_from_prose` (content extraction handling
  determiners/inflection) is legitimate-ish host text-processing (the body's "render the heard string"), but
  the cleanest target is the brain's own comprehension reading the prose — a deeper follow-on (ties to C-6 the
  neural discourse-planner), NOT on the cheap path.

**Honest scope:** B-i and B-ii are **cheap default-flips on the conversational composer's already-validated
spiking ops** (the familiarity gate, the WTA cleanup, the K-way sequencer) — wired into `ChatBrain` so the
loop's DECISIONS are neural. B-iii is **gated on PART A** (the faculty forward must be on-bridge AND fast).
B-iv is deep. ⇒ the integration is "consolidate the existing pieces onto one substrate + flip the
already-built spiking decisions on," not a new build — exactly the inventory's strategic root #2
(built-but-default-OFF spiking replacements).

## On-chip representation (H-4) for the integration

The faculty's on-bridge weights are the float64 complex-CSR (12× the fp16 size). For the integration to be a
*clean* one-substrate / neuromorphic story, the faculty's weights want the **dense-fp16 form** (= the O-2
`rf_dense_weights` mode = H-4's natural form). So H-4 is shared between PART A (perf) and PART B (a clean
on-chip representation for the integrated faculty). The cheap pre-step (drop the im CSR + f32, roadmap 1C) is
Phase-1; the dense-fp16 is O-2.

---

# Recommended sequence + what parallelizes + honest far-vs-near

## Sequence

```
Phase 1 (separate, cheap):  1C  drop the all-zero im CSR + f32  (storage; advances VRAM/perf/hardware)
                                  │
PART A (perf, the gate):    O-1  on-GPU forward  ──folds in──  O-2(host)  dense GEMM      [NO sim/ edit]
                                  │                                  │
                            O-3  KV cache (generation lever)   O-2-purity  cfg.rf_dense_weights  [optional, default-off, LAST]
                                  │
PART B (integration):       B-i   GATE-on-bridge (familiarity gate + spiking cleanup)   ┐ cheap default-flips,
                            B-ii  VERIFY-match-on-bridge (spiking compare)              ┘ NOT gated on PART A
                                  │
                            B-iii CONSTRAIN-on-bridge (route render through the bridge forward)  ── GATED ON PART A (O-1/O-2)
                                  │
                            B-iv  host-glue residual (brain re-comprehends the prose)  ── DEEP, last
```

## What parallelizes

- **1C ∥ PART A research/build** — 1C is an independent Phase-1 storage edit.
- **PART A (perf chain: O-1 → O-3, with O-2-purity optional-last) ∥ PART B-i/B-ii (cheap GATE/VERIFY default-flips)** —
  B-i and B-ii operate on the conversational composer's already-validated spiking ops + the host `ChatBrain`
  loop; they DO NOT depend on the faculty forward, so they parallelize with the perf work.
- **Within PART A:** O-1 and O-3 are partly sequential (O-3's gen speedup is most visible after O-1's on-GPU
  forward), but the O-3 attention-cache scoping can run in parallel with the O-1 build.
- **Serial dependency:** **B-iii (CONSTRAIN-on-bridge) is GATED ON PART A** (O-1/O-2 must land first so the
  on-bridge forward renders fast enough to sit in the loop). B-iv is deep + last.

## Honest far-vs-near

- **NEAR (cheap, mostly engineering + default-flips, weeks-class):** O-1 (on-GPU forward, the 97% wall), O-3
  (KV cache), 1C (storage), B-i/B-ii (the GATE/VERIFY decisions become neural via the already-validated
  composer spiking ops). These are the inventory's `[cheap]` items; no new science. **O-1 is the single
  highest-leverage near item** (it converts the DEMONSTRATED-but-slow co-residence into usable-local, and it
  is the prerequisite for B-iii). The PERF de-risk already PROVED the dense matvec; O-1 is "do the proven
  thing across the whole forward."
- **MID (gated, but reuse-heavy):** B-iii (route the loop's render through the on-bridge forward) — gated on
  PART A, then a reproduce-the-a/b/c-matrix-on-one-substrate validation; the O-2-purity `sim/` edit (optional,
  default-off, for the "the bridge's own read is fast" claim).
- **FAR (deep, owner-flagged):** B-iv (the brain re-comprehends the faculty's prose instead of host content
  extraction; ties to **C-6** the neural discourse-planner); **H-2** the host-DESIGNED structure (the faculty's
  weights are host-computed+injected — a chip would need developmental self-organization, the deepest
  categorical blocker, owner-flagged `feedback_spiking_structure_must_self_organize`); **C-9/I-9** brain-OWNS-
  generation (replace the Qwen faculty with the BPTT-SNN generative loop — the owner-accepted decoupling keeps
  Qwen-as-fluency for now, so this is the genuinely-far frontier).

**The one honest caveat:** the faculty being a real 494M Qwen (C-9) is an **OWNER-TRADE** (the LLM-supplies-
fluency decoupling; the moat makes hallucination impossible by construction). Bridge co-residence + functional
integration make that faculty *on-substrate and brain-gated*, but it is still not the BRAIN generating — that
(brain-owns-generation) is the separate I-9/3E frontier and is explicitly out of this gate's scope.

---

## `sim/`-edit flags — the complete list for both parts

| item | `sim/` edit | default | review |
|---|---|---|---|
| O-1 on-GPU forward | **NONE** (host-forward) | — | — |
| O-2 dense matvec (host) | **NONE** (`A @ W_dense`) | — | — |
| O-2 purity: `cfg.rf_dense_weights` + `cp_rf_w_dense` (read in `_rf_advance_one`/`rf_resonate_steps`/megakernel) | OPTIONAL | **OFF** = byte-identical CSR | byte-level diff |
| O-3 KV cache | **NONE** (host-forward attention) | — | — |
| 1C drop all-zero im CSR + f32 (`sim/bridge.py:5707`/`5786`/`5826`) | YES (cheap) | default-off→on | byte-level diff |
| B-i/B-ii GATE/VERIFY-on-bridge | **NONE** (reuse the composer's spiking ops + wire into `ChatBrain`) | flip on | — |
| B-iii CONSTRAIN-on-bridge | **NONE** (route render through the co-residence forward) | — | — |

**Net:** the ONLY new `sim/` edits are (a) the OPTIONAL default-off `cfg.rf_dense_weights` purity mode (O-2,
last) and (b) the cheap default-off storage trim (1C, Phase-1). Everything load-bearing — the on-GPU forward,
the KV cache, the GATE/VERIFY neural decisions — is host-forward / reuse-by-import with NO `sim/` edit. This
matches both findings' verdicts ("NO `sim/` edit required for either lever — both are host-forward changes").

## Key file:line references

- Perf de-risk + the proven dense GEMM: `research/runners/_bridge_cores_perf_derisk.py` (profile, `bench_dense`,
  `extrapolate`, `measure_end_to_end`); finding `2026-06-23-bridge-coresidence-perf-dense-matvec-GO-WITH-CAVEAT.md`.
- Full forward + co-residence: `research/runners/_bridge_cores_fullfwd_derisk.py` (the `RFMatvecVec` CSR install
  + the streamed 24-layer forward); finding `2026-06-23-bridge-coresidence-DEMONSTRATED.md`.
- The RF read / matvec / megakernel (the O-2 `sim/`-edit surface): `sim/bridge.py:5646` (`rf_kick`), `5691`
  (`rf_set_complex_weights`), `5710` (`_rf_advance_one`), `5749` (`rf_resonate_steps`), `5782`/`5814`
  (`_RF_MEGASTEP_SRC` / `_rf_resonate_steps_megakernel`); the float64 complex-CSR storage at `5707`/`5826-5827`
  (= H-4).
- The faculty + GATE/CONSTRAIN/VERIFY: `research/runners/_grounded_lang_integration_derisk.py`
  (`SpikingQwenFaculty`, `grounded_reply`); the WORKING host loop = `research/runners/brain_chat_tui.py`
  (`ChatBrain.gate` / `render` / `_verify` — the integration target, with the brain's GATE/VERIFY ops already
  on-bridge).
- The on-bridge composer precedent (megakernel + cache + spiking decisions): `research/runners/one_brain_composer.py`
  (`build_coresident_bridge`, `enable_rf_cudagraph`, `enable_spiking_cleanup`, `local_reciprocal_unbind`,
  `integrated_loop`).
- The validated spiking GATE/VERIFY replacements: `2026-06-11-familiarity-gate-v320-GO.md` (Bogacz-Brown
  abstain gate, zero moat-breaches at V=320); `enable_spiking_cleanup` (NEF WTA ==argmax).
- Biology (catalog): G.11 dual-stream (Hickok-Poeppel), G.12 Broca, G.13 Wernicke — the production↔comprehension
  (dorsal↔ventral / arcuate) cycle that the GATE→CONSTRAIN→VERIFY loop instantiates.
