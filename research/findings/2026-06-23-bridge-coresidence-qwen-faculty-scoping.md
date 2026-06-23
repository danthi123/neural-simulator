# Bridge co-residence of the spiking Qwen2.5-0.5B faculty — SCOPING (2026-06-23)

> **Status:** READ-ONLY deep-research + code/findings scoping for the owner-chosen TRUE-ONE-BRAIN-for-language
> direction — consolidate the **494M fluent Qwen faculty (B-1, ppl 1.08× ANN, coherent generation)** ONTO the
> `SimulationBridge`'s RF (resonate-and-fire complex-synapse) substrate, co-resident with the existing
> conversational brain (parser / composer), so faculty + brain run on ONE spiking substrate. **NO `sim/` edits, NO
> build, NO GPU run beyond reading.** Single deliverable = this doc. Every load-bearing claim re-verified against the
> repo (file:line) and the VRAM math computed from the cached `config.json`. The controller should trust-but-verify
> the **[VERIFY]** items, then push + present before building. This is a SCOPING/DECISION doc, NOT a result and NOT a
> commitment to build.

---

## 0. One-paragraph answer

**Bridge co-residence is FEASIBLE and LOCAL — the C1 consolidation mechanism transfers directly to the LLaMA stack
with NO new mechanism for the matvecs, ONE small exactness gap to close for RMSNorm, and the real wall being
WALL-CLOCK (not VRAM).** The C1 RF complex-synapse path consolidates *any* learned linear EXACTLY
(`Re(Z)/nsteps = a@W`, measured `max|·| = 4.9e-7` on the full Gen-F block, `_genseq_loopstep3_fullblock_rf.json`),
and that exactness is **architecture-agnostic** — q/k/v/o/gate/up/down/lm_head are all just dense matvecs, so the
GQA + tied-embedding + RoPE differences from Gen-F's vanilla GPT do **not** touch the consolidation. The
parameter-free nonlinearities the LLaMA stack adds (RMSNorm, SiLU-in-SwiGLU, Softmax) are *already* validated as the
same calibrated graded read B-1 used in PyTorch (ppl 1.08× ANN at T=16) and as the *on-bridge* graded-read + divisive
circuits the fully-spiking-C1 arc shipped for Gen-F (LayerNorm 0.962 / GELU 0.991 / softmax 0.9998, all NO `sim/`
edit) — **SiLU == GELU's mechanism verbatim** (a rectified-basis read), **RoPE is a fixed bit-exact rotation (0
learned params, applied host-side on the read)**, and the ONE genuine residual is that **Qwen's RMSNorm needs exact
`√(mean x²)` whereas the shipped `enable_input_divisive_norm` is an L1/mean-abs approximation** (the LayerNorm GO's
+0.037 residual). **VRAM (the load-bearing number): ~11.9 GB worst-case** (all 494M learned weights resident as the
current re+im float64 complex CSR), **comfortably < the 24 GB RTX-3090 wall → LOCAL**, and trivially shrinkable to
**~1–3 GB** by dropping the all-zero imaginary CSR (real weights ⇒ `W_im=0`) and/or storing dense (these layers are
100% dense — a sparse CSR is the *wrong* storage and costs 12× the fp16 ANN size). The cheapest-first de-risk is **ONE
Qwen decoder layer's q_proj ported to the live RF bridge + a bit-exactness check vs the B-1 PyTorch forward** before
anything bigger. The genuine open question is **NOT feasibility but wall-clock**: the RF matvec is a sparse-CSR
gather-bound op replayed per token in a Python kick/resonate/read loop, ~10–50× slower than a dense GEMM for the same
FLOPs, so the de-risk must measure tokens/sec and decide between (a) accepting slow-but-correct local wall-clock as a
*consolidation demonstration* (the project's framing — C1 is a brain-purity milestone, not a real-time deploy), (b)
batching tokens, or (c) a `sim/` edit for a dense-on-bridge RF matvec. **Cloud is NOT triggered by VRAM**; it is
triggered only if the de-risked wall-clock for the full 24-layer forward is so slow that even a small validation
corpus can't run overnight on the 3090 (per the owner rule `feedback_long_local_runs_ok_confirm_cloud_cause`:
wall-clock alone is fine with an ETA; cloud needs a genuine >24 GB VRAM wall, which this does NOT have).

---

## 1. Diagnosis — is the C1 mechanism sufficient for the LLaMA stack + the 494M scale?

### 1a. What the C1 mechanism IS (verified to source)

The C1 consolidation, as shipped and de-risked for Gen-F, is two parts:

1. **Every learned-weight matvec → the RF complex-synapse path, EXACTLY.** `rf_set_complex_weights(connections)`
   installs a real weight `W` (D_in→D_out) as a complex synapse `(post=D_in+nn, pre=m, weight=W[m,nn]+0j)` on a
   bridge of `n = D_in+D_out` neurons (`sim/bridge.py:5691-5708`). Kicking `z_in = a_in` (real), then
   `rf_resonate_steps(nsteps)` with `lam=0, ω≈0` (period huge), makes the complex accumulator compute
   `Re(Z_out) = nsteps·(a_in @ W)` with **no clip, no g·(V−E), no refractory ceiling** (`_rf_advance_one`,
   `sim/bridge.py:5710-5747`; the matvec line `cp_rf_w_re @ _rf_re - cp_rf_w_im @ _rf_im`). Read
   `Re(Z)/nsteps = a@W`. **Measured exact:** the full Gen-F block-0 reported `max|Re(Z)/nsteps − h@W| = 4.9e-7` across
   all 786,432-param matvecs, full-block fidelity spearman/cosine = **1.000**
   (`research/findings/raw/_genseq_loopstep3_fullblock_rf.json`; finding
   `2026-06-22-genseq-loopstep3-fullblock-rf-integration-GO.md`). The C1 milestone stacked this across all 4 Gen-F
   blocks + the head and GENERATED byte-identical text (`_genseq_loopstep3_full_genf_generate_derisk.py`).
2. **Every parameter-free nonlinearity → a faithful read.** In C1 these were *host* reads (softmax/GELU/LayerNorm);
   the fully-spiking-C1 arc then moved each ONTO the bridge with NO `sim/` edit: LayerNorm via
   `enable_input_mean_adapt` (subtract μ) + `enable_input_divisive_norm` (scale) + affine-on-read (0.962,
   `2026-06-23-spiking-layernorm-GO.md`), GELU via a 25-knot rectified-basis read through the shipped `a_cont`
   graded transfer (`bridge.py:6144`, 0.991, `2026-06-23-spiking-gelu-GO.md`), softmax via a calibrated graded `exp`
   read + the divisive-norm sum (0.9998, `2026-06-23-spiking-softmax-GO.md`).

### 1b. Does it TRANSFER to the LLaMA / Qwen stack? — op by op

| Qwen op | Type | C1 transfer? | Evidence |
|---|---|---|---|
| **q/k/v/o_proj, gate/up/down_proj** (7 linears/layer × 24) | learned dense matvec | **YES, exact, unchanged** | the RF exact-matvec is architecture-agnostic — it consolidates *any* `a@W` at `max-err ~5e-7` (`_genseq_loopstep3_fullblock_rf.json`). GQA only changes the *shape* (k/v are 896→128), not the mechanism. |
| **lm_head** (tied to tok_emb) | learned dense matvec, 896→151936 | **YES, exact, unchanged** | same RF exact-matvec; the C1 Gen-F runner already put the output head (256→513) on RF. Tying means it is the *same tensor* as the embedding (one CSR, not two). |
| **token + position** | embedding lookup (RoPE has no learned pos) | **YES — a gather, no matvec** | Gen-F's learned-pos lookup was used faithfully (the rows ARE `x`); Qwen ties tok_emb to the head and uses RoPE for position, so there is *only* the token gather (even simpler). |
| **RMSNorm** (49 instances) | parameter-free `x/√(mean x²)` + learned affine | **YES via the divisive circuit — but with ONE residual** (see 1c) | B-1 did exact-RMS as a host read (graded-pool SEM on the divisor, ppl 1.08× ANN). On-bridge: `enable_input_divisive_norm` gives the divisive read; RMSNorm has **no mean-centering** so `enable_input_mean_adapt` is unused (simpler than LayerNorm). |
| **SiLU** (inside SwiGLU, 24 instances) | parameter-free `x·σ(x)` | **YES — identical to GELU's mechanism** | B-1 already realized SiLU via the SAME calibrated rectified-basis graded read used for GELU (`_grounded_lang_p1b_stepB1_forward_derisk.py:168-177`, `make_silu_bank`; measured SiLU-input range [−7.34, 5.41], 30 knots, fit-max-err 0.003). The on-bridge GELU GO (0.991) is the proof the bridge realizes this read class. SwiGLU's gating `SiLU(xW_gate)⊙(xW_up)` is an element-wise product of two exact RF matvec outputs (faithful, like the attention value-mix). |
| **Softmax** (24 attentions) | parameter-free `exp`/normalize | **YES — already on-bridge for Gen-F** | the on-bridge softmax GO (0.9998) used exactly the graded `exp` read + divisive-sum B-1 used. **HONEST regime note:** B-1 measured Qwen's post-max-subtract logit min at **−102.5** (`stepB1_forward.json:33`) — far wider than Gen-F's [−3.96, 0] — and B-1 *handled it* with a WIDE exp grid [−34, 0.5] (39 knots) + a larger softmax pool (4096 at T=16). On the bridge this is the same wide-grid `exp` read; the spiking-softmax GO's low-temperature caveat is the thing to re-measure at the bridge (1c). |
| **RoPE** | fixed trig rotation of Q/K, **0 learned params** | **YES — applied host-side on the read, bit-exact** | B-0 confirmed RoPE bit-exact (no convert). It is a deterministic rotation matrix applied to the q/k RF-matvec outputs before the QKᵀ score — a faithful host op (like the bias add), NOT a learned weight to install. NEXUS/Plug-and-Play also treat it as fixed. |

**Verdict on transfer:** the C1 mechanism is **sufficient for the matvecs and for SiLU/Softmax/RoPE with NO new
mechanism**. The architectural deltas from Gen-F (RMSNorm vs LayerNorm, SiLU vs GELU, RoPE vs learned-pos, GQA, tied
embedding) are all handled by existing, already-de-risked machinery. **No new mechanism is needed for the LLaMA
stack per se.**

### 1c. The ONE genuine residual — exact-RMS vs the shipped L1 divisive circuit

The LayerNorm GO recorded honestly that the shipped `enable_input_divisive_norm` is an **L1 / mean-abs** spread, not
exact RMS `√var`, and that exact-RMS "needs a square+sqrt circuit (heavier than the shipped L1 op)" with a +0.037
residual on the block (`2026-06-23-spiking-layernorm-GO.md`). **Qwen's RMSNorm is exactly `x/√(mean x²)`** — so the
shipped divisive circuit is an *approximation* of it (49 instances across the model; small per-token per-instance, but
it compounds the same way the LayerNorm one did, and RMSNorm is more divisor-sensitive than LayerNorm because there is
no mean-centering to absorb the error). Three resolutions, cheapest first: **(i)** accept the L1-approx on-bridge
RMSNorm and *measure* the ppl impact (the cheapest — it may stay within the 1.2× bar, as B-1's exact-RMS host read had
huge margin at ppl 1.08×); **(ii)** keep RMSNorm as a faithful HOST read (exactly what B-1 did — the matvecs are
on-bridge, the divisive norm is a host read, which is *still* the C1 "weights-on-bridge + nonlinearities-as-reads"
scope the fully-spiking arc started from); **(iii)** build the exact `√(mean x²)` divisive circuit on the bridge (a
square + sum + sqrt read; the heaviest, only if (i) shows the L1-approx degrades generation). **This is the ONLY piece
that is genuinely new vs the shipped Gen-F machinery, and even it has a zero-`sim/`-edit fallback (ii).**

### 1d. Does the SCALE (494M, 145× Gen-F's 3.4M) break anything?

- **Exactness:** no. The RF matvec is exact per-op regardless of size; stacking 24 layers + head accumulates no error
  (the C1 Gen-F result showed per-layer fidelity stays ~1.0 across the 4-block stack — "no error accumulation"). The
  same will hold at 24 layers (the only risk is the RMSNorm-approx compounding, 1c — measurable).
- **VRAM:** the load-bearing number — see §2. Fits local.
- **Wall-clock:** the genuine scaling wall — see §3 + §2c. The RF matvec is sparse-CSR + per-token Python loop; at
  494M-dense it is slow. This is the thing to de-risk, NOT feasibility.

**⇒ Diagnosis: the C1 mechanism is SUFFICIENT — no new mechanism for the LLaMA stack; ONE small exactness gap
(RMSNorm) with a zero-edit fallback; the only genuinely new challenge is wall-clock at scale, which is an
engineering/perf question, not a feasibility one.**

---

## 2. VRAM feasibility — 494M on the bridge (THE load-bearing number)

### 2a. Exact per-matvec dimensions (from the cached `config.json`)

Qwen2.5-0.5B-Instruct: `hidden_size=896`, `num_hidden_layers=24`, `intermediate_size=4864`,
`num_attention_heads=14`, `num_key_value_heads=2` (GQA, head_dim=64, kv_dim=128), `vocab_size=151936`,
**`tie_word_embeddings=true`** (lm_head = tok_emb). Install convention `a@W`, RF bridge `n = D_in+D_out`,
`nnz = D_in·D_out` (these layers are **100% dense** ⇒ nnz = the full matrix):

| matvec | D_in | D_out | n=D_in+D_out | nnz=params |
|---|---|---|---|---|
| q_proj | 896 | 896 | 1,792 | 802,816 |
| k_proj | 896 | 128 | 1,024 | 114,688 |
| v_proj | 896 | 128 | 1,024 | 114,688 |
| o_proj | 896 | 896 | 1,792 | 802,816 |
| gate_proj | 896 | 4,864 | 5,760 | 4,358,144 |
| up_proj | 896 | 4,864 | 5,760 | 4,358,144 |
| down_proj | 4,864 | 896 | 5,760 | 4,358,144 |
| **lm_head** (tied) | 896 | 151,936 | **152,832** | **136,134,656** |

Per-layer learned nnz (7 linears) = 14,909,440; ×24 = 357,826,560; + lm_head 136,134,656 =
**493,961,216 params (~494M)**. (The tied tok_emb 151936×896 = 136.1M is the *same tensor* as lm_head — one CSR,
not two.)

### 2b. THE VRAM ESTIMATE (assumptions stated)

The bridge stores each RF matvec as **two** complex CSR matrices (`cp_rf_w_re` + `cp_rf_w_im`,
`sim/bridge.py:5707-5708`), each with **float64 data + int32 indices + int32 indptr**. Per weight ≈
`2 CSR × (8 B data + 4 B index) = 24 B` (indptr is `(n+1)·4`, negligible). Bridge state arrays per unique shape
(v/u/prev_im f32 + fired bool + spike_step int64 = 21 B/neuron) are ~3 MB total. **The C1 `full_genf` pattern caches
EVERY unique matvec's CSR simultaneously** (`_WEIGHT_CSR_CACHE`, `_genseq_loopstep3_full_genf_generate_derisk.py:280`),
so the persistent weight VRAM is the sum over all 24×7+1 = 169 learned matvecs:

| Variant | VRAM | Notes |
|---|---|---|
| **AS-IS (current bridge): re+im CSR, f64 data, int32 idx** | **11.86 GB** | what `rf_set_complex_weights` builds today, all weights resident |
| Drop the im CSR (real weights ⇒ `W_im=0`): re-only, f64 | 5.93 GB | the matvec only reads `W_re`; the all-zero im CSR is pure waste |
| + float32 data (megakernel already casts; activations f32) | 3.95 GB | f32 data halves the data term |
| **Dense float16 weights (the NATURAL form — these layers are 100% dense)** | **0.99 GB** | = the ANN model size; a sparse CSR is the *wrong* storage |
| Dense float32 weights | 1.98 GB | |

- **Worst-case (do-nothing, as-is): ~11.9 GB.** Add the co-resident conversational brain (parser+composer
  ~54k-neuron one-brain at V=320 is a few hundred MB of complex synapses; the parser/dlPFC Izhikevich slices are tiny)
  + transients (the lm_head kick is a complex128 vector of n=152,832 = **2.4 MB**, freed per call; KV cache at a
  2048-ctx eval is ~25 MB) ⇒ **well under 24 GB.**
- **The lm_head alone** (n=152,832, nnz=136M) as the current re+im f64 CSR = **3.27 GB** — the single biggest object;
  it dominates because the vocab is 151,936. (At fp16-dense it is 272 MB.)
- **Blowup factor:** the as-is CSR is **12× the fp16 ANN size** (24 B/weight vs 2 B), because the RF complex-synapse
  path was designed for the composer's *sparse* O(D) diagonal bind/unbind, not a dense 494M matvec. **This is the
  structural mismatch to flag — see §3.**

**⇒ Per the owner cloud rule (`feedback_long_local_runs_ok_confirm_cloud_cause`: cloud needs a genuine >24 GB VRAM
wall): even the worst-case 11.9 GB is LOCAL. VRAM does NOT trigger cloud.** [VERIFY] the 11.9 GB number assumes the
`_WEIGHT_CSR_CACHE`-everything pattern; if weights are streamed layer-by-layer (build → use → free) the resident
peak is one layer (~360 MB) + lm_head (3.27 GB) ≈ **3.6 GB**, even safer.

### 2c. The transient/activation memory + the perf caveat (NOT a VRAM problem, a wall-clock problem)

The C1 `full_genf` runs the RF matvec **per token** in a Python loop (`_rf_matvec_rows`,
`_genseq_loopstep3_full_genf_generate_derisk.py:313` — `for r in range(rows.shape[0])`, each row = kick + resonate(8)
+ read). This is **VRAM-cheap but WALL-CLOCK-expensive**: each resonate step touches all nnz CSR edges; one token
through the whole model = `8 × 494M = 4.0B` edge-ops via a *sparse CSR matvec* (cuSPARSE, gather-bound) — roughly
**10–50× slower than the equivalent dense cuBLAS GEMM** for the same FLOPs, and the per-token Python kick/read loop
adds launch overhead the masked megakernel (`enable_rf_cudagraph`, `sim/bridge.py:5764`) only partly amortizes (it
fuses the ~15–20 kernels/step into 1, but the per-token loop and the CSR-on-dense remain). **This is the genuine
scaling wall, and it is wall-clock, not memory.** It is exactly what the §3 de-risk must measure.

---

## 3. Ranked cheapest-first de-risks

Each runs in PyTorch-free RF-on-bridge land (`SIM_BACKEND=cupy`, the B-1 PyTorch forward is the TEACHER). Ordered so
the cheapest signal fires first and a NEGATIVE stops the ladder before the expensive steps.

| # | De-risk | What it proves | Cost | Cloud? |
|---|---|---|---|---|
| **1** | **ONE Qwen layer's q_proj (896→896) ported to the live RF bridge → bit-exactness vs the B-1 PyTorch matmul.** Install the real `q_proj` weight via `rf_set_complex_weights`, kick the B-1 layer-12 input activation, read `Re(Z)/nsteps`, assert `max|· − a@W_q| < 1e-5`. | the RF exact-matvec holds on a REAL Qwen weight tensor (Gen-F proved it for 256×256; this confirms it at 896-wide on Qwen weights). The foundational feasibility check. | minutes, 1×3090, 1 matvec | **No** |
| **2** | **ONE full Qwen decoder layer on the bridge** — all 7 linears on RF + RMSNorm (host read, B-1 style) + SiLU (the GELU rectified-basis read) + softmax (the wide-grid exp read) + RoPE (host) — fidelity of the layer output vs the B-1 PyTorch layer (spearman/cosine ≥ ~0.99). | the LLaMA-stack BLOCK consolidates (the C1 full-block result, but for Qwen's RMSNorm/SwiGLU/RoPE/GQA). Measures whether the RMSNorm-approx (1c) bites at the block level. | ~10–30 min, 1×3090 | **No** |
| **3** | **The RMSNorm exactness probe** — compare on-bridge `enable_input_divisive_norm` (L1) vs exact `√(mean x²)` on real Qwen RMSNorm inputs, and measure the ppl impact of the approx over a few held-out windows. Decide (i) accept-L1 / (ii) host-read / (iii) build-exact-circuit. | resolves the one genuine residual; picks the resolution before stacking 24 layers. | ~10 min | **No** |
| **4** | **The full 24-layer Qwen forward on the bridge, FEW tokens, logit-fidelity + greedy-match vs the B-1 PyTorch forward** (the C1 `full_genf` pattern, scaled up; weights streamed layer-by-layer to cap resident VRAM at ~3.6 GB). **Plus the wall-clock measurement** — tokens/sec, and extrapolate the time for a small validation corpus. | the WHOLE faculty consolidates end-to-end (logit spearman ≥ 0.8, greedy argmax-agreement high) **and** the load-bearing wall-clock number that decides perf strategy. | hours, 1×3090 (slow per the perf caveat) | **No** unless §5 trips |
| **5** | **Co-residence smoke** — the consolidated Qwen faculty as a masked RF slice on the SAME bridge as the conversational brain (one-brain parser/composer), via the established `neuron_mask` + `inject_explicit_wiring` slicing; confirm a composer op and a faculty matvec on one bridge with byte-isolation (the faculty slice doesn't perturb the composer's no-confab moat, and vice-versa). | the actual "one spiking substrate" claim — faculty + brain co-resident, capability-preserved. | ~30 min | **No** |
| **6** | **(only if #4's wall-clock is prohibitive) Perf lever** — batch tokens through the RF matvec (one weight-set, many kicks) and/or scope a dense-on-bridge RF matvec `sim/` edit (a dense GEMM in `Re(Z) = a@W` instead of CSR). | turns the consolidation from "slow demonstration" into "usable," IF the demonstration (#4) already passed. | days (a `sim/` edit) | **No** (compute, not VRAM) |

**The cheapest-first principle:** #1 is minutes and de-risks the whole foundation; #2–#3 settle the LLaMA-stack +
RMSNorm question on ONE layer (the expensive 24-layer run #4 only happens after a single layer is proven). Do NOT run
#4 before #1–#3 are GREEN.

---

## 4. Reusable machinery (what already exists — reuse-by-import, minimize new code)

| Need | Existing machinery | Location |
|---|---|---|
| **RF exact matvec primitive** (install W, kick, resonate, read `a@W`) | `rf_linear_layer_signed` + `_build_rf_bridge` + the operating point `RF_PERIOD=1e5, RF_NSTEPS=8, RF_LAMBDA=0` | `research/runners/_genseq_loopstep3_rf_probe.py:116-138`; `_set_rf_weights` + `_WEIGHT_CSR_CACHE` (the build-CSR-once optimization) `_genseq_loopstep3_full_genf_generate_derisk.py:283-323` |
| **The bridge RF substrate** | `rf_set_complex_weights` / `rf_kick` / `rf_resonate_steps` / `rf_read_phases` / `_rf_advance_one` / the masked megakernel `enable_rf_cudagraph` | `sim/bridge.py:5646,5684,5691,5710,5749,5814`; `NeuronModel.RESONATE_AND_FIRE` |
| **The B-1 spiking ops** (RMSNorm-graded, SiLU-graded, wide-grid softmax-exp, RoPE-exact) | the calibrated `GradedRead` + `make_silu_bank` + `make_exp_bank` + `spiking_rmsnorm_forward` + the wide [−34,0.5] exp grid (Qwen logit-min −102.5 already handled) | `research/runners/_grounded_lang_p1b_stepB1_forward_derisk.py:114-306` (the install hooks `install_spiking_ops:312`) |
| **On-bridge parameter-free nonlinearity circuits** | `enable_input_divisive_norm` (`bridge.py:6190`, the softmax-sum + RMSNorm divisor), `enable_input_mean_adapt` (`bridge.py:6238`, unused for RMSNorm), the `a_cont` graded transfer (`bridge.py:6144`, SiLU/GELU/exp reads) | `sim/bridge.py`; the Gen-F GO runners `_genseq_spiking_{layernorm,gelu,softmax}_derisk.py` |
| **C1 consolidation harness** (per-matvec on RF, batched-per-shape, logit-fidelity + greedy-match + lesion + ppl) | the whole `rf_full_forward` / `_rf_block_forward` / `_greedy_continue` / `_heldout_nll_numpy` scaffold — adapt the block from `_Block` to a Qwen `Qwen2DecoderLayer` | `research/runners/_genseq_loopstep3_full_genf_generate_derisk.py:335-495` |
| **RF-distill** (the clip-aware fallback if a per-layer read ever compresses) | `distill_weights_rf_faithful` + `install_and_measure_rf` (train W' through the RF-faithful clip forward, install at unit scale) | `research/runners/_genseq_loopstep3_rf_distill_derisk.py:148-346` — **NOTE: Qwen's nonlinearities have no per-layer clip readout (the RF-distill's 0.556 wall was Gen-F-specific to a clipped LIF readout); Qwen reads are graded `a_cont`, not clipped, so this is a SAFETY-NET, not expected to be needed.** |
| **Co-residence slicing** (faculty slice + brain slice on one bridge, masked RF ops) | the `neuron_mask=` arg on `rf_kick`/`_rf_advance_one`/the megakernel + `inject_explicit_wiring` framework-slice wiring + the nav+conv merge pattern | `sim/bridge.py:5646,5710,5814`; `research/runners/nav_conv_merged_bridge.py`; `research/runners/one_brain_composer.py` (the masked co-resident RF composer) |
| **The teacher** (the fidelity reference) | the B-1 PyTorch full spiking forward (ppl 1.08× ANN, T=16) AND the exact-float ANN Qwen forward | `_grounded_lang_p1b_stepB1_forward_derisk.py`; the cached `Qwen/Qwen2.5-0.5B-Instruct` HF model (already downloaded) |

**Net new code:** a Qwen-decoder-layer adapter (extract the 7 linears + RMSNorm affines + RoPE freqs from the HF
`Qwen2DecoderLayer`, in the `a@W` install convention — exactly what `load_genf_full` does for Gen-F's `_Block`,
ported to the LLaMA stack) + the per-layer RMSNorm/SiLU/softmax/RoPE wiring (all reusing the B-1 ops + the on-bridge
circuits). No new *mechanism*.

---

## 5. Anti-cheat controls

The C1/B-0/B-1 anti-cheats carry over verbatim; the load-bearing ones for THIS scoping:

1. **Bit-exactness measured, not asserted** (#1, #2, #4): report `max|Re(Z)/nsteps − a@W|` per matvec (the Gen-F
   block reported 4.9e-7; a real consolidation must show the same ~1e-6 on Qwen weights). A *claimed* exactness without
   the measured max-err is a cheat.
2. **Logit/generation fidelity vs the B-1 teacher** (#4): spearman ≥ 0.8 over probed positions AND greedy argmax-
   agreement high AND the generated text reads coherent (the load-bearing READ — ppl alone is insufficient, per the
   B-1 / scoping [VERIFY]). Match the B-1 verbatim generations.
3. **Load-bearing lesion** (#2, #4): scramble (row-permute) the RF complex weights of every matvec → the logit
   fidelity must collapse to ~chance and the lesioned greedy decode must DIVERGE from the off-bridge decode (the C1
   `full_genf` lesion: real 1.0 → lesioned ~chance). Proves the RF matvecs carry the computation, not the host reads.
4. **RMSNorm sanity** (#3): the on-bridge L1-divisive RMSNorm vs exact `√(mean x²)` — report the per-token cosine and
   the ppl delta. A "RMSNorm works on bridge" claim that hides the L1-vs-exact gap is the exact cheat the LayerNorm GO
   avoided (it reported the +0.037 honestly).
5. **Co-residence isolation** (#5): the faculty slice must NOT perturb the conversational brain's no-confab moat
   (the composer abstains 0-false-accept with the faculty co-resident) and the brain must not corrupt a faculty
   matvec (byte-identity of the faculty read with/without the brain slice). The nav+conv merge's byte-isolation gate
   is the template.
6. **The honest-scope label** (every step): "weights-on-RF (exact) + RMSNorm/SiLU/softmax as graded reads
   (faithful) + RoPE host-exact" — distinct from "fully-spiking" (which would also require the exact-RMS circuit if
   #3 picks (iii), and is the Gen-F fully-spiking-C1 scope). Don't overclaim "fully spiking on one brain" if RMSNorm
   stays a host read (it's still the legitimate C1 "weights-on-bridge" scope, just labeled accurately).

---

## 6. The explicit cloud-trigger

Per the owner rule (`feedback_long_local_runs_ok_confirm_cloud_cause`: **cloud is for a genuine >24 GB VRAM wall;
wall-clock alone is fine with an ETA, run local**), and given §2's finding that the worst-case is ~11.9 GB (LOCAL):

- **VRAM does NOT trigger cloud.** 494M as RF complex CSR (worst-case 11.9 GB, layer-streamed 3.6 GB, dense-fp16
  ~1 GB) + the brain + transients all fit on the 24 GB 3090 with room.
- **The ONLY cloud-trigger is wall-clock-infeasibility, and only at one specific point:** if de-risk #4 (the full
  24-layer forward) measures a tokens/sec so low that even a *small validation corpus / a handful of generations*
  cannot complete overnight on the 3090 (the CSR-on-dense + per-token-loop wall, §2c), **AND** the perf lever #6
  (batch tokens / dense RF matvec) is deferred. Even then, the first move is the local perf lever, not cloud (the
  bottleneck is compute, which an H100 only makes ~3–5× faster — it does not lift a VRAM wall, because there is none).
- **Concretely:** measure tokens/sec in #4 first; report the ETA for the validation run; run it locally unless the
  ETA exceeds a reasonable overnight budget — and even then, fix the perf (#6) before reaching for cloud.

---

## 7. Trust-but-verify (load-bearing claims)

**Verified directly this pass (file:line / computed):**
- **RF exact matvec** `Re(Z)/nsteps = a@W`, `max-err 4.9e-7` on the full Gen-F block, fidelity 1.000:
  `research/findings/raw/_genseq_loopstep3_fullblock_rf.json` (read), `_genseq_loopstep3_rf_probe.py:116-138` (read).
- **RF substrate storage** = two complex CSR (`cp_rf_w_re`+`cp_rf_w_im`, f64 data + int32 idx), sparse `(n,n)`,
  `n=D_in+D_out`: `sim/bridge.py:5691-5708` (read in full), the matvec `_rf_advance_one:5710-5747`, the masked
  megakernel `:5814-5855`.
- **B-1 spiking Qwen forward** GO (ppl 7.08 = 1.08× ANN at T=16, coherent generation, RMSNorm/SiLU/softmax graded +
  RoPE exact, linears exact PyTorch matmul): `_grounded_lang_p1b_stepB1_forward_derisk.py` (read in full),
  `_grounded_lang_p1b_stepB1_forward.json` (read). Qwen logit-min −102.5 handled by the wide [−34,0.5] exp grid.
- **On-bridge parameter-free nonlinearities** GO for Gen-F (LayerNorm 0.962 / GELU 0.991 / softmax 0.9998, NO `sim/`
  edit; the L1-vs-exact-RMS +0.037 residual recorded): `2026-06-23-spiking-{layernorm,gelu,softmax}-GO.md` (read).
- **Qwen2.5-0.5B exact architecture** (d=896, L=24, I=4864, H=14, KV=2, V=151936, tied embedding): the cached
  `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/.../config.json` (read).
- **VRAM math** (494M learned nnz; as-is re+im f64 CSR = 11.86 GB; lm_head 3.27 GB; dense-fp16 0.99 GB; 12× blowup):
  computed from the config + the CSR storage model (this pass).
- **Co-residence** = `neuron_mask` on the RF ops + `inject_explicit_wiring` framework slices: `sim/bridge.py:5646,
  5710,5814`, `nav_conv_merged_bridge.py` (read), `one_brain_composer.py` (the masked co-resident composer pattern,
  per CLAUDE.md).
- **Scoping prior** (modern SLMs are LLaMA-family; Qwen2.5-0.5B the recommended faculty; the LLaMA-stack convert is
  SOLVED-class): `2026-06-22-grounded-language-faculty-scoping.md` (read).

**Flagged honestly (could NOT fully verify / are the de-risk's job):**
1. **[VERIFY — most load-bearing] the wall-clock of the full 24-layer RF forward at 494M.** §2c is a FLOP/edge-op +
   CSR-vs-dense order-of-magnitude estimate (~10–50× a dense GEMM, per-token Python loop), NOT a profiled run. De-risk
   #4 produces the real tokens/sec. This is the number that decides perf strategy and the (sole) cloud-trigger.
2. **[VERIFY] whether the L1-approx on-bridge RMSNorm (49 instances) keeps ppl within the 1.2× bar.** The LayerNorm
   GO's +0.037 was for ONE Gen-F LayerNorm; Qwen has 49 RMSNorms with no mean-centering. De-risk #3 measures it;
   resolution (ii) (host-read RMSNorm) is the zero-edit fallback if it bites.
3. **[VERIFY] that the wide-grid softmax exp read holds on the bridge at Qwen's −102.5 logit-min** (B-1 handled it in
   PyTorch; the on-bridge softmax GO was on Gen-F's narrow [−3.96,0]). De-risk #2 measures it; the spiking-softmax
   GO's low-temperature caveat is the precise thing to re-check.
4. **[VERIFY] the 11.9 GB assumes cache-everything;** layer-streaming caps resident at ~3.6 GB. Either way LOCAL, but
   the de-risk should pick the streaming pattern to keep headroom for the co-resident brain.
5. **The RF-distill safety-net (§4) is almost certainly unneeded** for Qwen (its 0.556 wall was a Gen-F clipped-LIF
   readout; Qwen reads are graded `a_cont`, not clipped) — but it exists if a per-layer read unexpectedly compresses.

---

## Sources

### Project record (re-verified this pass, file:line)
- `sim/bridge.py` — RF substrate (`rf_kick:5646`, `rf_read_phases:5684`, `rf_set_complex_weights:5691`,
  `_rf_advance_one:5710`, `rf_resonate_steps:5749`, megakernel `:5782,5814`); the nonlinearity circuits
  (`a_cont:6144`, `enable_input_divisive_norm:6190`, `enable_input_mean_adapt:6238`).
- `research/runners/_grounded_lang_p1b_stepB1_forward_derisk.py` — the B-1 spiking Qwen forward (the TEACHER + the
  reusable graded ops).
- `research/runners/_genseq_loopstep3_full_genf_generate_derisk.py` — the C1 milestone harness (per-matvec RF,
  `_WEIGHT_CSR_CACHE`, lesion/ppl/greedy).
- `research/runners/_genseq_loopstep3_rf_probe.py` — `rf_linear_layer_signed` + `_build_rf_bridge` + the RF operating
  point.
- `research/runners/_genseq_loopstep3_rf_distill_derisk.py` — the RF-distill clip-aware fallback (Gen-F-specific
  safety-net).
- `research/runners/nav_conv_merged_bridge.py`, `research/runners/one_brain_composer.py` — the co-residence slicing.
- `research/findings/raw/_genseq_loopstep3_fullblock_rf.json`, `_grounded_lang_p1b_stepB1_forward.json` — the measured
  exactness + B-1 GO.
- `research/findings/2026-06-22-genseq-step0-C1-consolidation-GO.md`,
  `2026-06-22-genseq-loopstep3-fullblock-rf-integration-GO.md`,
  `2026-06-23-spiking-{layernorm,gelu,softmax}-GO.md`, `2026-06-23-generative-loop-DEMONSTRATED.md`,
  `2026-06-22-grounded-language-faculty-scoping.md`.
- `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/.../config.json` — the exact Qwen2.5-0.5B architecture.

### Memory (owner directives applied)
- `feedback_long_local_runs_ok_confirm_cloud_cause` (cloud only for a genuine >24 GB VRAM wall; wall-clock fine with
  an ETA) — the §6 cloud-trigger.
- `project_generative_sequence_frontier` (the BPTT-SNN generative arc; C1 one-spiking-bridge + C2 no-forget gates) +
  `project_one_brain_integrated_pipeline_and_cleanup` (the TRUE one-brain integration goal).
- `feedback_brain_based_only_standard` / `feedback_spiking_structure_must_self_organize` — the honest-scope labeling
  (§5.6): "weights-on-RF + nonlinearities-as-reads" is NOT "fully spiking"; and the faculty's weights are
  host-DESIGNED (a converted ANN), a known residual vs self-organized structure — to be labeled, not hidden.
