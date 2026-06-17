# Optimizing GPU spiking-neural-network + vector-symbolic compute: how similar architectures are traditionally accelerated (literature review for the two profiled bottlenecks)

**Date:** 2026-06-17
**Type:** read-only deep-research / reference review (no code changed; this doc is the only deliverable)
**Lens:** the RTX-3090 profile (`research/findings/2026-06-17-scaling-profile-3090-latency-is-the-wall-not-vram.md`) found the conversational
wall is **per-op LATENCY**, with two distinct bottlenecks:
1. **The resonate-loop launch overhead** — one bind/unbind op runs a 208-timestep Python loop issuing ~3,000–4,000 tiny
   sequential GPU kernel launches → ~160 ms/op, GPU ~99% idle (launch-bound).
2. **The fact-store query = an O(K·B) linear scan** — answering who/what unbinds the cue against every stored fact one
   at a time; abstention scans them all.

Terms defined once, in plain language:
- **Vector Symbolic Architecture / hyperdimensional computing (VSA / HDC):** computing with very long vectors ("hypervectors")
  where structured facts are built by two reversible operations — **bind** (combine a role and a filler into one vector)
  and **bundle/superpose** (add several vectors into a set) — and read back by **unbind** + a **cleanup** step that snaps a
  noisy result to the nearest stored item.
- **Cleanup / item memory:** the dictionary of known concept vectors; "cleanup" = find the nearest one to a noisy query
  (a nearest-neighbor lookup). This project's **no-confab abstention** is exactly a cleanup with a similarity *threshold*:
  if nothing is close enough, answer "I don't know."
- **Resonate-and-fire (RF) neuron:** a spiking neuron whose sub-threshold state is a rotating complex number
  `z ← z·exp(λ+iω)`; a zero-crossing emits a spike. The composer realizes VSA bind/unbind through *complex synapses*
  on these neurons (Frady & Sommer's spiking-phasor scheme).
- **Kernel launch overhead:** the fixed CPU cost (~20–200 µs) to dispatch one GPU operation, regardless of how little work
  it does. Many tiny sequential ops → the GPU sits idle waiting for the CPU to feed it ("launch-bound").
- **CUDA graph:** record a fixed sequence of GPU ops once, then replay the whole sequence with a single CPU submission —
  the standard cure for launch overhead.
- **SpMV / matvec:** sparse-matrix × vector. The RF complex synapse is four of these per step (via cuSPARSE).

---

## 1. Executive summary

The two bottlenecks are textbook, and the fixes are well-established in two adjacent literatures (GPU SNN simulators; GPU
VSA/HDC libraries). Highest-leverage findings:

- **Bottleneck 1 (launch overhead) — the single highest-leverage technique overall is to stop looping in Python and run the
  whole resonate evolution as ONE launch.** Two independent, proven routes, both directly applicable, and they compound:
  (a) **CUDA-graph the 208-step loop** (the project already prototyped this for 11×); the established workaround for "CuPy
  can't capture cuSPARSE" is to replace the structured bind matvec with a **custom elementwise gather-scale kernel** (the
  bind weights are a permutation, so this is exact, not an approximation) — this is the *same pattern* GeNN uses to beat
  Brian2CUDA 2–3× on small networks via "merged kernels." (b) **Parallel-scan the timestep loop** — because the RF
  sub-threshold dynamics are *linear* (`z·exp(λ+iω)`), the 208 sequential steps can be computed in parallel over time with an
  associative scan (the **ParaLIF** technique, "up to 200× faster" for linear spiking neurons). Route (a) is the cheaper,
  lower-risk near-term win and is already de-risked; route (b) is the bigger ceiling but a deeper arc.
- **Bottleneck 2 (O(K·B) fact scan) — the highest-leverage technique is to BATCH the scan into one op, then add a threshold
  nearest-neighbor index.** Stacking all stored composites and unbinding in a single batched kernel turns K-many ops into 1
  (this is exactly what TorchHD and the "Sutra" VSA compiler do for VSA; ~16–100× reported). For the larger fact store, a
  nearest-neighbor index with **range/radius search** (FAISS `range_search`, or HNSW) replaces the linear scan with indexed
  lookup AND natively supports the abstention threshold ("return everything within radius r; empty ⇒ abstain").
- **The owner's graph-data-structure libraries (`scipy.sparse.csgraph` RCM, `networkx`/METIS) help a different thing:** they
  reorder the *connectivity matrix* for memory locality inside each matvec (modest, structure-dependent: literature reports
  average <5% on GPU SpMV, occasionally up to ~50%). They do **not** touch the launch overhead, which is 97.7% of the cost
  here. Place them as a minor, optional matvec tweak — not the fix.

**Single most-promising direction:** the CUDA-graph + custom-elementwise-kernel refactor of the resonate loop (bottleneck 1),
with batched-unbind for the fact scan (bottleneck 2) as the cheap companion. Both are squarely the "many tiny ops → one
launch" pattern that the entire GPU-SNN and GPU-VSA literature converges on.

---

## 2. Bottleneck 1 — the resonate-loop launch overhead (ranked techniques)

Ranked by (applicability × expected win ÷ effort) for THIS resonate loop.

### 2.1 [TOP] Collapse the loop into one launch — CUDA graph + custom-kernel matvec  ·  proven 11× already; ceiling much higher

**What it is.** A CUDA graph records the fixed op sequence once and replays it with a single CPU submission, so the ~3,000–4,000
per-op launches collapse toward ~1. PyTorch and JAX both use this specifically to kill small-op launch overhead; NVIDIA quotes
per-kernel launch overhead at **20–200 µs** and notes the benefit "is particularly visible … with very small batch sizes, where
CPU overheads are more pronounced" — i.e. exactly the launch-bound regime this profile is in. PyTorch's own example: a CPU-bound
backbone went **31 ms → 6 ms (≈5×)** purely from graphing, with "CPU maxed at 100% while GPU is idle most of the time" (same
signature as our 99%-idle 3090).

**The cuBLAS/cuSPARSE-capture obstacle and its established workaround.** The project already hit, and correctly diagnosed, the
wall: *CuPy* cannot capture a cuSPARSE/cuBLAS call inside a stream-captured graph (CuPy raises on synchronous/library calls
during capture; CuPy's docs confirm "during stream capture, synchronous device-host transfers are not allowed"). The standard
fix used across the field is **don't put the library call in the graph — replace it with a custom graph-safe kernel.** Here that
is *free of accuracy cost*, because the bind/unbind weights are a **near-diagonal permutation** (post-neuron `D+k` ← pre-neuron
`k` × a phase): the "matvec" is really a **gather-scale**, expressible as one elementwise CuPy/RawKernel op that captures
cleanly. This is exactly the prototype that already gave **107 ms → 9.8 ms (11×, measured)**. Two host-side requirements the
graph imposes (both already noted in the profile): the per-step host counter must become a **device scalar**, and all scratch
must be **pre-allocated** (graphs replay fixed addresses; PyTorch's BERT example shows the same static-shape discipline).

**How it maps.** Make `_rf_advance_one` graph-able (elementwise gather-scale synapse for the structured composer weights +
device-scalar step index + pre-allocated buffers), capture `rf_resonate_steps` once, replay per op. The 208 steps then run at
compute speed.

**Expected win / effort / risk.** ~11× demonstrated; with the elementwise fusion below, more. Effort: a contained protected
`sim/` edit (the prototype exists). Risk: low-moderate — the `tests/test_rf_*` suite pins bit-identical RF dynamics; keep a
default-preserving path for non-permutation/general weights.

> **Trust-but-verify flag (load-bearing):** there is a subtlety in "CUDA graphs can't capture cuBLAS." NVIDIA's own forums say
> cuBLAS *can* be captured "without restrictions in most situations," the exceptions being routines that write results into
> **host** buffers or use host-pointer scalar mode. So the true blocker is **CuPy's capture path / the device-host sync**, not
> CUDA graphs categorically. The project's conclusion (use a custom kernel) is still the right move and is the simplest path,
> but the framing should be "CuPy can't capture *our* cuSPARSE call as written," not "CUDA graphs fundamentally can't capture
> cuSPARSE." Worth a one-line correction in the profile doc.

### 2.2 [HIGH ceiling, deeper arc] Parallel-scan the timestep loop (ParaLIF / temporal parallelization)  ·  up to ~200×, but a rewrite

**What it is.** The sequential timestep loop exists only because each step depends on the last. But if the sub-threshold
dynamics are **linear**, the whole trajectory can be computed **in parallel across time** using an *associative (parallel/prefix)
scan* — the same trick that parallelizes RNNs/state-space models. Recent SNN work makes this explicit: the "charge–fire–reset"
sequential dependency is the barrier, but "sub-threshold dynamics are linear and can be parallelized using associative scans"
(Temporal Parallelization for GPU Acceleration of SNNs, ICLR-track; CUDA + JAX implementations). The **ParaLIF** neuron
("Accelerating SNNs with Parallelizable Leaky Integrate-and-Fire Neurons") reports **up to 200× faster** than sequential LIF at
similar accuracy/sparsity.

**Why it's a strong fit here.** The RF neuron's sub-threshold update is *exactly* a linear recurrence: `z_{t+1} = z_t·exp(λ+iω)`
plus the (structured, linear) synaptic input. A pure rotation/decay is the easy case for a scan — over a fixed 208-step window,
`z_t = z_0·exp((λ+iω)t) + Σ_{s<t} exp((λ+iω)(t-s))·input_s` is a parallel prefix over complex numbers. The only nonlinearity is
the spike (zero-crossing) read-out, which the literature handles by computing the linear trajectory in parallel, then detecting
crossings after.

**How it maps / expected win / effort.** Replace the 208-iteration Python loop with one batched complex associative scan over
the window (CuPy can express the closed-form geometric accumulation directly; or use a log-depth scan). Potentially collapses the
208 steps to O(log 208) parallel depth → comparable-or-better than the graph route, and it removes the loop entirely rather than
just cheapening its dispatch. Effort: higher (a genuine reformulation of the RF integrator + a careful re-derivation that keeps
spike timing bit-faithful), so this is the **deeper arc after** the graph quick-win. Risk: spike-timing fidelity must be proven
against the existing `test_rf_*` golden outputs.

### 2.3 [MEDIUM] Fuse the per-step element-wise ops into one/two kernels (megakernel-lite)  ·  partial win if the graph refactor is deferred

**What it is.** Independent of graphs, a single timestep currently issues ~15–20 separate kernels (rotate, decay, four matvecs,
zero-crossing, masked writes). Fusing the element-wise ones into one or two kernels (CuPy `@fuse`, or a hand-written `RawKernel`)
cuts launches per step several-fold. This is the GPU-SNN field's baseline lesson: the **"megakernel"** / "one fused kernel per
timestep for the whole population" pattern. The clearest published evidence is **Brian2CUDA vs Brian2GeNN**: Brian2CUDA emits
"three separate CUDA kernels" per population per step (integrate / detect-spike / reset) and is consequently **"2–3× slower for
smaller networks (N < 10⁴)" … "due to sequential execution from multiple small kernels … compared to fewer merged kernels in
Brian2GeNN."** GeNN's **"merged groups"** (and `mergePostsynapticModels`) generate **one kernel covering many
populations/pathways**, which is precisely why it wins the small-network regime — our exact regime. Norse shows the same effect
inside PyTorch: its *compiled* model wins "due to compiler techniques such as kernel fusion, where subsequent layers are fused
into single operations."

**How it maps / win / effort.** A `@fuse()`d `_rf_advance_one` element-wise core (rotate+decay+crossing+masked-write as one
kernel) is a small, low-risk edit that helps even before the full graph capture, and it makes the body *simpler to capture* once
graphs land. Win: a few-× on the per-step launch count; effort: low. Best used as the **stepping-stone** to 2.1.

### 2.4 [SITUATIONAL] Procedural / on-the-fly connectivity (Knight & Nowotny)  ·  a memory/scaling tool, not the latency fix here

**What it is.** GeNN's **procedural connectivity** generates synaptic connections and weights *on the fly* in the kernel (from a
seed) instead of storing/reading them from memory. Knight & Nowotny (*Nature Computational Science*, 2020) used it to simulate
**4.13×10⁶ neurons and 24.2×10⁹ synapses on a single GPU**, run a cortical column at **~0.5× real-time on one V100** (beating a
CPU cluster and SpiNNaker), at up to **14× lower energy** per synaptic event.

**Honest placement for us.** This solves *memory/bandwidth and capacity*, which the profile already says is **not** our
near-term wall (VRAM is fine at small-LLM scale). It is **not** the launch-overhead fix. Two caveats make it a poor near-term
fit: (a) the documented limitation is that **procedural connectivity is incompatible with synaptic plasticity** (you can't store
learned weights you regenerate) — and the composer's *binding* weights are a fixed permutation anyway, so there's nothing to
regenerate-vs-store. File this as relevant **only** if/when the fact store or cortex scales to the point VRAM bites; it's a
known escape hatch, not today's move.

---

## 3. Bottleneck 2 — the O(K·B) fact-store scan (ranked techniques)

The query "who/what" currently unbinds the cue against each stored fact in turn; abstention scans all. Two complementary fixes,
both must preserve the **thresholded, abstaining** read (the no-confab moat). Per the owner's standing guidance, the moat is a
"plus, not a hard gate" and may be traded where it buys scaling — but the techniques below *preserve* it for free, so there is no
trade needed.

### 3.1 [TOP, cheap] Batch the scan into one launch  ·  K ops → 1 op; ~16–100× precedent

**What it is.** Instead of looping over facts, **stack all K stored composites into one array and unbind/compare in a single
batched op** (block-diagonal weights, or one big complex matvec + one batched cleanup). This is the canonical "batch many small
problems into one launch" pattern, and it is exactly what the GPU-VSA libraries do:
- **TorchHD** (the standard HD/VSA library on PyTorch): implements bind/unbind/bundle with **batch processing**, and reports
  experiments "up to **100× faster**" than prior public code by leaning on batched GPU tensor ops.
- **"Sutra: Tensor-Op RNNs as a Compilation Target for VSA"** (arXiv 2026) compiles VSA programs to fused, batched tensor graphs:
  bundling "**stacks rotations into a (k,d,d) tensor, stacks fillers into (k,d), runs one batched einsum + sum + L2-normalize**,
  combining what would otherwise be separate operations into single kernel launches," for **~16×** over a per-sample Python loop
  *on CPU alone* (GPU would compound it). Notably Sutra's substrate is the **same structure as our composer** — frozen role
  rotations precomputed at compile time, `bind = matmul against a precomputed matrix`, `unbind = Rᵀv` — i.e. the academic
  blueprint for batching exactly our operations.

**How it maps / win / effort.** One batched unbind over all KB facts replaces the per-fact loop; the abstention scan becomes one
reduction over the batch. Win: ~K× at scale (and removes the per-fact launch overhead, which compounds with bottleneck 1).
Effort: low–moderate, pure runner/composer-side (no `sim/` edit needed if done at the composer/orchestration layer). **This is
the cheapest bottleneck-2 win and should ship with, or before, the graph work.**

### 3.2 [HIGH at scale] Nearest-neighbor index with thresholded/abstaining lookup  ·  O(K·B) → ~O(log) retrieval

**What it is.** Replace the linear scan with an **approximate-nearest-neighbor (ANN) index** over the stored hypervectors:
**FAISS** (Facebook AI Similarity Search) or **HNSW** (Hierarchical Navigable Small World graphs). These are the standard tools
for "find the closest stored vector(s) to a query" at scale, with GPU support in FAISS.

**Preserving the no-confab abstention — the key constraint, and it's natively supported.** The moat is a *similarity threshold*,
not top-1. FAISS provides **`range_search`** which "returns all vectors within a **radius** around the query … all vectors with
distance < radius" — i.e. **set the radius to the abstention threshold; an empty result set ⇒ abstain.** This maps the moat onto
the index exactly. Real caveat to verify before relying on it: **FAISS `range_search` is CPU-only** for the relevant index types
(IndexFlat, IndexIVFFlat, IndexScalarQuantizer, IndexIVFScalarQuantizer per the FAISS wiki) — the GPU indexes are k-NN-oriented,
so a thresholded GPU path may need k-NN-then-filter (retrieve top-k on GPU, then apply the radius on the host) rather than native
GPU range search. OpenSearch/Milvus document the same "radial/range search" pattern if a different backend is preferred.

**How it maps / win / effort.** Index the fact store by cue (the project profile already flags "index the fact store by cue
(agent+action)"); a query touches only candidates → ~O(1)–O(log K) instead of O(K). Win: turns the "abstention at KB=1000 is
minutes" worst case into sub-millisecond. Effort: moderate (add an index, keep it in sync with the store, route the thresholded
lookup). **Best after 3.1** — batching alone may suffice until the store is large; the index is the durable answer for a big
fact store.

### 3.3 [REFERENCE] HDC/VSA item-memory acceleration literature  ·  confirms the approach, hardware-flavored

The HDC hardware literature treats "associative search = nearest-neighbor over stored hypervectors" as *the* hot loop and
accelerates it directly. **HD-Core** (FPGA, exploiting computational reuse) reports **4.8× speedup and 4.4× energy** over an
optimized GPU baseline and 2.4× over prior FPGA, much of it from reuse in the **associative-search** stage; **FACH** reduces the
similarity-search complexity. Kleyko/Frady's "VSA as a computing framework for nanoscale hardware" frames cleanup as
content-addressable / in-memory search. Takeaway for us: these validate that the *associative search is the right thing to index/
batch*, and that **computational reuse across consecutive similar queries** is a known lever — relevant if conversational turns
query overlapping cues. (Software batching + ANN index capture most of this win without new hardware.)

---

## 4. The graph-data-structure libraries (`scipy.sparse.csgraph`, `networkx`/METIS) — where they DO and DON'T help

The owner's search surfaced reverse Cuthill-McKee (`scipy.sparse.csgraph.reverse_cuthill_mckee`) and graph partitioning
(METIS / `networkx`). Placed honestly:

- **What they actually do:** reorder the rows/columns of a sparse matrix to reduce its **bandwidth** (pull non-zeros toward the
  diagonal). This improves **memory locality** when you multiply that matrix by a vector (the reused `x` vector and matrix data
  hit cache better). Partitioning (METIS) clusters strongly-connected nodes to localize a SpMV or to split work across devices.
- **Where they help us:** *only inside the matvec*, i.e. the cuSPARSE complex synapse — and **only if** we keep a real sparse
  general matvec. Honest magnitude from the GPU-SpMV literature: RCM reordering gives **on average <5%** on GPU SpMV across test
  matrices, occasionally **up to ~50%** for favorably-structured matrices; reported 26–33% reductions in `x`-vector / matrix
  access time in some studies. It is **structure-dependent and modest**.
- **Where they DON'T help — the decisive point:** they do nothing about **kernel launch overhead**, which is **97.7%** of our op
  cost. Reordering a matrix doesn't reduce the *number* of launches; it makes each (already-tiny, already-idle-GPU) matvec a bit
  more cache-friendly. Worse, for *this* composer the bind weights are a **permutation/near-diagonal** — already maximal-locality
  and, in the recommended design (§2.1), replaced by an elementwise gather-scale that has **no matrix to reorder at all.** So RCM
  is essentially moot on the *bind* path.
- **The one place graph-structure analysis is genuinely useful here:** *detecting* that the bind weights are a permutation (so
  you can swap cuSPARSE → elementwise gather-scale) is a structure-recognition step — but that's a one-time observation already
  made, not a runtime reordering library. METIS/partitioning would only matter if the *learned cortex* (the separate, plastic,
  genuinely-sparse recurrent network) became the bottleneck and you wanted locality/multi-GPU there — not the composer.

**Bottom line:** correct tools, wrong bottleneck. Keep RCM in mind as an optional, low-priority tweak for any *general* sparse
matvec that survives the refactor; it is not the latency fix and should not be sequenced ahead of §2.1/§3.1.

---

## 5. Recommended sequence for the production fix (cheapest-highest-leverage first)

1. **(Quick win, bottleneck 2)** **Batch the fact-store scan** into one unbind+cleanup over all stored composites (§3.1). Pure
   composer/runner-layer, no `sim/` edit, ~K× at scale, removes the per-fact launch overhead. Ship first — it's the lowest-risk
   leverage and de-risks the query path before touching `sim/`.
2. **(Quick win, bottleneck 1)** **Fuse `_rf_advance_one`'s element-wise ops** with `@fuse()`/RawKernel (§2.3). Small, low-risk,
   cuts per-step launches, and shapes the body for graph capture.
3. **(Main near-term arc, bottleneck 1)** **CUDA-graph the resonate loop** with the **custom elementwise gather-scale** standing
   in for the permutation matvec + **device-scalar step counter** + **pre-allocated scratch** (§2.1). This is already prototyped
   at 11×; it's the single highest-leverage move. Protected `sim/` edit, byte-reviewed, default-preserving for general weights,
   `test_rf_*` as the golden gate.
4. **(Durable scaling, bottleneck 2)** **Add the cue-indexed nearest-neighbor lookup** with **thresholded/range search** for the
   abstaining read (§3.2) once the fact store is large enough that batching alone lags; mind the FAISS GPU-range-search caveat
   (k-NN-then-filter on GPU, or CPU `range_search`).
5. **(Deeper ceiling, bottleneck 1 — separate arc)** **Parallel-scan the RF timestep loop** (ParaLIF-style associative scan over
   the linear sub-threshold dynamics, §2.2). Bigger potential win than the graph (it deletes the loop), but a genuine
   reformulation that must prove spike-timing fidelity — schedule after the graph win lands and is validated.
6. **(Optional, low priority)** RCM/locality reordering only for any *general* sparse matvec that remains (§4); not on the bind
   path. Procedural connectivity (§2.4) only if/when VRAM becomes the wall.

Quick-wins (1–3) plausibly land most of the profile's projected "**~0.8 s/turn → ~10–25 ms = real-time**" with low risk; (4)–(5)
are the scaling/ceiling follow-ons.

---

## 6. Sources (URLs)

GPU SNN simulators / launch overhead / megakernel / procedural connectivity:
- Knight & Nowotny, *Larger GPU-accelerated brain simulations with procedural connectivity*, Nature Computational Science 2020 — https://www.nature.com/articles/s43588-020-00022-7 (preprint: https://www.biorxiv.org/content/10.1101/2020.04.27.063693v2)
- Brian2CUDA: *Flexible and Efficient Simulation of SNN Models on GPUs*, Frontiers in Neuroinformatics 2022 (the "multiple small kernels vs merged kernels; 2–3× slower for N<10⁴" evidence) — https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.883700/full
- GeNN — code-generation SNN simulator; "merged groups" / `mergePostsynapticModels` — https://genn-team.github.io/ ; release notes https://genn-team.github.io/genn/documentation/4/html/df/ddb/ReleaseNotes.html ; original paper https://www.nature.com/articles/srep18854
- *Spike: A GPU Optimised SNN Simulator* (task grouping + async spike generation to cut launch overhead) — https://www.biorxiv.org/content/10.1101/461160v2.full
- NEST-GPU / NeuronGPU throughput comparison — https://pmc.ncbi.nlm.nih.gov/articles/PMC7925400/
- Norse / BindsNET / snnTorch benchmarking (compiled/fused wins; per-timestep loop bottleneck) — https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/ ; BindsNET https://github.com/BindsNET/bindsnet ; Norse https://github.com/electronicvisions/norse

Temporal parallelization / parallel scan for linear spiking dynamics (bottleneck-1 ceiling):
- *Temporal Parallelization for GPU Acceleration of SNNs* (associative scan over linear sub-threshold dynamics; CUDA + JAX) — https://openreview.net/forum?id=SMZnJtkNX5
- *Accelerating SNNs with Parallelizable Leaky Integrate-and-Fire Neurons* (ParaLIF, "up to 200× faster") — https://www.techrxiv.org/doi/full/10.36227/techrxiv.170905886.62702188/v1
- *Bullet Trains: Parallelizing Training of Temporally Precise SNNs* — https://arxiv.org/html/2603.13283v2

CUDA graphs / launch overhead / capture limitations:
- *Accelerating PyTorch with CUDA Graphs* (20–200 µs/launch; 31→6 ms backbone; static-shape requirement; what can't be captured) — https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- NVIDIA *CUDA Graph Best Practice for PyTorch* — https://docs.nvidia.com/dl-cuda-graph/latest/
- CuPy CUDA-graph API (`Stream.begin_capture`/`end_capture`, `Graph.launch`; "synchronous device-host transfers not allowed during capture") — https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Graph.html ; https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html
- cuBLAS-in-CUDA-graph nuance (capturable except host-buffer / host-pointer-scalar cases) — https://forums.developer.nvidia.com/t/stream-capture-of-cublas-gemm/216148 ; https://forums.developer.nvidia.com/t/graph-capture-of-cublasddot-in-device-pointer-mode/287388
- PyGraph / torch.compile CUDA-graph integration (reference patterns) — https://arxiv.org/html/2503.19779v2

VSA/HDC + resonator acceleration / batching (bottleneck 2 + bind/unbind):
- *Resonator Networks 1 & 2*, Frady/Kent/Olshausen/Sommer, Neural Computation 2020 — https://direct.mit.edu/neco/article/32/12/2311/95651 ; https://direct.mit.edu/neco/article/32/12/2332/95653 ; PDF https://rctn.org/bruno/papers/resonator1.pdf
- *Neuromorphic visual scene understanding with resonator networks*, Nature Machine Intelligence 2024 — https://www.nature.com/articles/s42256-024-00848-0
- TorchHD (batched GPU VSA/HDC, "up to 100× faster") — https://github.com/hyperdimensional-computing/torchhd ; paper https://arxiv.org/abs/2205.09208
- *Sutra: Tensor-Op RNNs as a Compilation Target for VSA* (precomputed role rotations, batched einsum bundling, ~16×) — https://arxiv.org/html/2605.20919v1
- *VSA as a computing framework for nanoscale hardware* (Kleyko/Frady et al.; cleanup as content-addressable search) — https://par.nsf.gov/biblio/10486268-vector-symbolic-architectures-computing-framework-nanoscale-hardware
- HD-Core (FPGA, 4.8× over GPU via associative-search reuse) — https://par.nsf.gov/servlets/purl/10301134 ; FACH — https://acsweb.ucsd.edu/~sag076/papers/aspdac19_fach.pdf
- Holographic Reduced Representations / FHRR (binding = circular convolution, O(d log d) via FFT; "precompute FFTs of frequent vectors") — https://arxiv.org/pdf/2109.02157 ; https://www.neurips.cc/paper/2021/file/d71dd235287466052f1630f31bde7932-Paper.pdf

Nearest-neighbor index + thresholded/abstaining lookup (bottleneck 2 constraint):
- FAISS (GPU ANN) — https://github.com/facebookresearch/faiss ; docs https://faiss.ai/
- FAISS `range_search` (radius/threshold search; CPU-only for the listed index types) — https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes
- HNSW (Malkov & Yashunin, arXiv:1603.09320) — referenced via FAISS wiki https://github.com/facebookresearch/faiss/wiki
- OpenSearch radial/range vector search (alt backend for thresholded lookup) — https://opensearch.org/blog/vector-radial-search/

Sparse-matvec reordering (the graph-data-structure libraries, honest placement):
- *Is Sparse Matrix Reordering Effective for SpMV?* (RCM avg <5% on GPU, structure-dependent) — https://arxiv.org/html/2506.10356
- *Optimization of SpMV using reordering techniques on GPUs* (26–33% access-time reductions in some cases) — https://dl.acm.org/doi/10.1016/j.micpro.2011.05.005
- `scipy.sparse.csgraph.reverse_cuthill_mckee` — https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.reverse_cuthill_mckee.html

---

## 6b. Addendum — Renner/Frady/Sommer Loihi resonator paper (owner-supplied arXiv:2208.12880, read in full)

The owner supplied the open arXiv version of the otherwise-paywalled Nature MI resonator paper. Read in full; it
sharpens three of the recommendations above (synthesis only; no paper text reproduced):

- **Confirms the bottleneck-2 batching recommendation (§3.1) at the algorithm level.** The resonator does its
  combinatorial search by **superposition + a single codebook matched-filter** — each module's update is
  `f( C·Cᵀ · ( s ⊙ unbind-of-all-other-factors ) )`, where `C·Cᵀ` is one matmul against the *whole* codebook and
  `f(x)=x/|x|` is the phasor-normalize cleanup. There is **no per-candidate loop** — the parallel search lives in
  the superposition + one matched-filter matmul. This is exactly our "stack all stored composites, one batched
  unbind + one codebook cleanup" plan; the canonical VSA inference engine is built this way. Strong precedent.
- **The 208-step resonate window may be shortenable (a cheap extra latency lever).** Their spike-timing phase code
  on Loihi represents a phasor with a **T=16-timestep cycle**; our composer runs a **208-step** resonate window per
  op. Worth a cheap probe: does a much shorter `period` still give a faithful phase read on our resonate-and-fire
  substrate? If so, it multiplies directly with the graph/fusion wins (fewer steps to fuse/scan) — and it's a
  one-line knob to test against `test_rf_*` golden outputs.
- **Neuromorphic hardware is an ENERGY play, not a SPEED play — honest placement.** On Loihi the resonator is
  *slower* than a CPU but **orders of magnitude more energy-efficient** (their Fig. 6). So for our *latency* goal on
  the 3090, the graph + batch + (later) parallel-scan fixes are the right levers; neuromorphic silicon (Loihi/etc.)
  would only matter if energy/embedding became the objective, not wall-clock. This refines, not contradicts, §2.
- **Bonus (deeper arc):** their iterative factorization (each module unbinds all *other* factors, converges in a
  few iterations via the superposition→sharpening dynamic) is the efficient method **if/when** the composer needs
  genuine multi-factor factorization (the F≥2 attribute case that is currently a numpy reference) — the resonator
  network is the principled, parallel way to do it, and it's the same FHRR substrate we run.

## 7. Trust-but-verify — claims/numbers the controller should double-check

- **"CUDA graphs cannot capture cuSPARSE/cuBLAS" is too strong as stated.** NVIDIA forums indicate cuBLAS *is* capturable in
  most cases (exceptions: host-buffer output, host-pointer scalar mode). The real, confirmed blocker is **CuPy's** capture path +
  the device-host sync, not CUDA graphs categorically. Recommend softening the profile doc's wording. (The custom-kernel fix is
  unaffected and remains the simplest route.)
- **ParaLIF "up to 200×"** and **TorchHD "up to 100×"** and **Sutra "~16×"** are the authors' headline figures on *their*
  benchmarks/hardware (ParaLIF: neuromorphic classification tasks; TorchHD: vs prior public code; Sutra: CPU-only laptop, vs a
  Python loop). They establish the *technique's* magnitude, not a guaranteed transfer to our 208-step complex RF loop — treat as
  directional ceilings, not promises. The project's own **11× measured** CUDA-graph prototype is the reliable in-house number.
- **RCM "<5% average, up to ~50%"** is aggregated across heterogeneous test matrices in the SpMV literature; the exact figure for
  any specific matrix varies widely. Used here only to argue RCM is *not* the fix, which holds regardless.
- **FAISS `range_search` GPU support:** I confirmed it's documented CPU-only for the listed index types; whether a current FAISS
  build adds GPU range search should be verified against the installed version before designing around it (fallback: GPU top-k +
  host radius filter).
- **Knight & Nowotny capacity numbers** (4.13M neurons / 24.2B synapses / 0.5× real-time / 14× energy) are from the abstract and
  consistent across the Nature CS + bioRxiv versions; not independently re-derived. They're cited only to scope procedural
  connectivity as a *memory/scaling* tool, which doesn't bear on the latency fix.
