# Fluid conversation — Phase 0: a minimal (~21M) transformer generator is FLUENT (the transformer-minimization thesis)

**2026-07-01 (autonomous night; the owner's main-priority pivot).** Phase 0 of the fluid-conversation roadmap
(`2026-07-01-fluid-conversation-mechanisms-roadmap.md`): can a MINIMAL transformer — dramatically smaller than the
current external Qwen2.5-0.5B — supply fluency, so the transformer is *minimized* (the owner's directive) rather than
deleted (a genuine open-domain wall)? **Result: YES on the fluency half.** A **~21.3M-param** GPT (d_model 512, 6
layers, 8 heads, vocab 2049, block 512), trained locally on a **90M-token TinyStories** subset, is genuinely fluent.

## Training (local GPU, ~4.3h on the 3090)
- Corpus: a fetched TinyStories subset (443,800 stories, ~90M train tokens + ~2M held-out; `data/corpus/tinystories_*`).
- `tiny_transformer_train`, 25,000 steps, dropout 0.1, weight-decay 0.1, cosine LR, held-out probe every 1,500 steps.
- **init loss 7.78 → final 1.78.** Held-out ppl descended **monotonically 8.00 → 5.66** with the train↔held-out gap
  staying **tiny throughout** (train 6.05 ≈ held-out 6.11 at step 24k) → the model **generalizes, does not overfit**
  — the documented "overfit-not-size" generative-ceiling wall is *avoided* by matching the model to a constrained
  domain with enough data (the TinyStories regime, Eldan & Li 2023). Ckpt: `research/findings/raw/fluidconv/gen_tinystories_20M.ckpt.pt`.

## Fluency (free-generation samples, temp 0.8)
Coherent, grammatical, multi-sentence prose:
- *"Once upon a time, there was a dolphin who was very brave and loved to play with his friends. … They laughed and smiled together."*
- *"Lily saw a big flower. She had never seen a flower like that before. It was pink and green and had many petals. Lily wanted to touch the petals and smell them."*
- *"One day, a boy named Tom went to the park with his mom. Tom was very happy. He liked to play on the swings and slide."*
- *"Sara and her mom went to the park to play on the swings. Sara loved the swing and wanted to try it…"*

TinyStories-level coherence (occasional logical quirks, e.g. a dolphin "waving his arms") — expected at this scale/
domain, and exactly the regime the roadmap identified as small-model-fluent.

## What this establishes (+ what's next)
- **The transformer can be MINIMIZED, not just used:** a ~21M model — **15–25× smaller** than the external Qwen-0.5B
  — is fluent on a constrained domain. And it is small enough that the already-validated **88.6M spiking-forward path**
  makes bridge co-residence (a spiking-on-substrate generator) cheap — the transformer becomes *a spiking network on
  the one brain*, minimized + brain-integrated (the roadmap's honest sweet spot), rather than a bolted-on 0.5B box.
- **NEXT (the core Phase-0 drop-in, in progress):** drop this ~21M generator into the EXISTING grounded-lang
  gate→constrain→verify loop (`constrained_decode_gate._GroundedConstrainedLM`, parameterized for this arch) in place
  of Qwen — and confirm it renders GATED facts fluently (constrained non-vacuity), the generation stays GROUNDED
  (asserts the gated fact), and the no-confab MOAT holds (an unconstrained/wrong-fact generation is vetoed / falls
  back to the template — the transformer never asserts an unverified fact).
- **Then Phase 1** (recurrent/RWKV-style block for incremental on-substrate generation + multi-turn coherence) and
  **Phase 2** (growth via the develop loop), per the roadmap; **Phase 3** the transformer-free thalamocortical bet
  in parallel.

**NO `sim/` edit.** Corpus fetch (public TinyStories) + local training + reuse of the grounded-lang loop.
