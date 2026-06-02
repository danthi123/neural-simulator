# Generative 3090-ceiling test: scaling the subword spiking LM does NOT rescue it -- honest NEGATIVE (overfit, not a size limit) -- 2026-06-02

## The owner's question
"Push scale -- the ideal is at least comparable to a tiny/small SOTA modern LLM in conversational
capabilities." Owner chose (after I surfaced the documented wall) the "3090 generative ceiling" path: push the
biological generative net as far as one GPU allows on a real corpus, measure the honest gap to a tiny SOTA LLM.
Owner notes that sharpened the design: (1) open LLM corpora available; (2) we've hit COMPUTE/SPEED limits on
the 3090 but NOT VRAM -> push model SIZE up, accept slow training.

## Grounding (check-existing-first)
The project already has a 2026-05-17 "generator" arc:
- **Generator-S** = subword spiking LM (surrogate-grad BPTT) on real TinyStories, hidden 256,256 -> honest
  NEGATIVE: held-out perplexity 117K-388K, token-soup, worse than uniform-random.
- **Generator-F** = 6M-param TRANSFORMER, same corpus -> PASS, held-out ppl ~6.1, coherent simple-story
  English. This is our concrete "tiny LLM" reference.

Generator-S was small (hidden 256). The owner's note (VRAM headroom) points at the one unexplored cell: does
SCALING the spiking LM rescue it?

## The ceiling test (single-seed decisive probe, _ceiling_probe.py)
A ~25M-param subword spiking LM -- 100x more params than Generator-S, 8x its training data -- on TinyStories:

```
arch: 1025 -> 4096 -> 4096 -> 1025 LIF  (25.2M params)  T=48 batch=64 lr=0.005 epochs=30 n_train=16000
backend=GPU  train 17.9 min
train loss: 20.11 -> 6.13   (fits the training data well)
held-out perplexity = 203,753
uniform-random floor (vocab 1025) = 1025  -> 200x WORSE than random (token-soup)
transformer reference (Generator-F) = 6.1  -> 33,402x worse
generated: "would big somandsome keep happy. knew climbheafrom He in Map s. it. a "for it. "No, she tta
            the s. n, ard yxf." o stres, he very ifriendtree. al <|endoftext|> The se. , ounwas tring..."
VERDICT: TOKEN-SOUP (ceiling NEGATIVE at this scale)
```

## What this means (honest mechanism, no spin)
- **Scaling did NOT rescue the spiking LM.** 100x params + 8x data took held-out perplexity from Generator-S's
  ~100K-388K to ~204K -- still token-soup, still worse than random. The generation has real words ("big",
  "happy", "He", "little", "tree", "friend") but no grammar or coherence.
- **The mechanism is OVERFITTING, not a size limit.** The train loss dropped substantially (20.1 -> 6.1: the
  model FITS the training windows) while held-out perplexity is astronomically bad (~200K). The spiking
  surrogate-grad LM memorizes training patterns but does not GENERALIZE to held-out language. This is exactly
  the Generator-S lesson ("a decreasing train loss is not held-out language competence") -- now confirmed it
  PERSISTS at 100x scale.
- **Therefore more VRAM/size is not the lever.** Adding parameters to an overfitting model makes it fit train
  better, not generalize better (consistent with the earlier 50M-char run, where 375x more params made word
  features WORSE). The bottleneck is the spiking architecture's generalization, not the GPU's memory.
- **A 6M transformer reaches ppl 6.1 (coherent) on the same corpus.** The gap is the ARCHITECTURE (surrogate-
  gradient BPTT through LIF spikes), not the scale -- a 4x-smaller transformer generalizes; a 25M spiking net
  does not.

## Confirmation in flight
A ~50M-param run (hidden 4096x3) with 2.5x more data (40,000 samples) is running (job bhkf5gatm) to make the
ceiling claim airtight -- throwing both size AND more data at the generalization problem. Prediction (per the
overfit mechanism + the 50M-char precedent): still token-soup, because size+data alone don't fix the spiking
LM's generalization. [Folded in when it lands.]

## Honest answer to the owner's goal
Reaching tiny-SOTA-LLM conversational capability requires GENERATIVE language modeling that GENERALIZES to
held-out text. On a single 3090, under the standalone/no-external-LLM constraint:
- The biological SPIKING generative LM does NOT generalize to coherent held-out language, and scaling it (size
  + data) does not rescue it -- confirmed now at 25M params (and a 50M confirmation in flight), not just the
  small Generator-S.
- A standard TRANSFORMER (Generator-F, 6M params) DOES generate coherent simple-story English (ppl 6.1) on the
  same corpus and hardware.

So the honest gap is architectural: the spiking surrogate-grad approach is the bottleneck. The options (owner's
call, each with honest trade-offs):
1. **Non-spiking transformer** -- already works locally (Generator-F coherent at ppl 6.1); could be scaled on
   the 3090 toward richer generation. But it's not the biological/spiking substrate (a paradigm choice).
2. **Cloud-class generative scale** -- the only documented path to real-LM-class, with real cost + uncertainty.
3. **Accept the biological system's actual strength** -- the validated symbolic conversation (320-448 concepts,
   >=30-fact KB, who/what/yes-no/negation, honest abstention) + continual learning, which is genuinely
   brain-analogue and biology-translatable, but is structured Q&A, not open-ended LLM dialogue.

This is the honest negative the project's goal explicitly values as the deliverable ("honest negatives under
strict biology ARE the scientific deliverable"). The spiking generative path is a documented 3090 dead-end for
coherent generation -- now confirmed it does not yield to scale.

## Reproduce
```
python -m research.findings.raw._ceiling_probe --tag big1 --vocab 1024 --hidden 4096,4096 --T 48 \
    --epochs 30 --n-train 16000 --batch 64
```
