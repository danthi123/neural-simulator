---
type: finding
status: design
claim_check: synthesis
date: 2026-09-05
mechanism: hippokey — STRUCTURED HiPPO SSM -> CONTENT-ADDRESSABLE LEARNED-KEY ATTENTION (a FIXED HiPPO multi-timescale diagonal SSM produces a per-position multi-timescale context code x_s; a causal softmax read forms Q/K over x_s and V over the token content z), the literal owner steer for the next own-voice-fluency mechanism class
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [42, 43, 44, 100, 101, 102]
verdict: >
  ROADMAP / DESIGN NOTE + FIRST VERSION IMPLEMENTED + 6-SEED TRAINING QUEUED (no measurement yet — the GO/NO-GO
  lands when the queued GPU run clears). Establishes the honest current own-voice fluency ceiling from the record
  (linattn, the deployable mouth, CROSSES a fair trigram on the SIMPLE simplewiki domain +0.0505 6/6 but FALLS
  BELOW it on the BROAD wikitext103 domain, -0.29..-0.57 at depth>=2; assoc/assoc_t/ssm/hippo all sit at the
  ~-0.12/-0.15 bound), scopes WHY the SSM/reservoir family is trigram-bound (fixed-size-state compression) and
  why the two prior attention arms (assoc/assoc_t) hit the same bound (weak token-local keys), and implements the
  literal owner steer as a NEW --recurrence hippokey arm distinct from the fixed-codebook learnkey the project had
  substituted. First version is additive (a new --recurrence arm, byte-identical when off), CPU-smoke-verified
  (builds/trains/evals; anti-cheats behave — mless-collapse +0.855, perm-collapse +0.175 at a toy d48 scale), and
  the apples-to-apples 6-seed simplewiki de-risk is QUEUED on the GPU lane.
lane_wall: brain-native open-ended generation (own-voice mouth) — roadmap Wall #7 / R4
external: >
  Gu, Dao, Ermon, Rudra, Re 2020, "HiPPO: Recurrent Memory with Optimal Polynomial Projections" (NeurIPS) — the
  multi-timescale diagonal state family (A eigen-spectrum a spread of decay rates). MacDonald, Lepage, Eden,
  Eichenbaum 2011 (Neuron 71:737-749) hippocampal time cells; Howard & Kahana 2002 (J Math Psychol 46:269-299)
  Temporal Context Model — the entorhinal multi-timescale "when/context" code. Ramsauer et al. 2020 ("Hopfield
  Networks is All You Need", ICLR 2021) — modern-Hopfield <-> softmax-attention equivalence (the content-
  addressable read as one-shot pattern completion). Marr 1971; Treves & Rolls 1994; Rolls & Treves 1998 — CA3
  recurrent-collateral autoassociation; Hasselmo — EC-context-cued CA3 retrieval. Grounds the entorhinal ->
  CA3 circuit mapping (multi-timescale context KEYS the content-addressable read). Same external round that
  grounded the ordered-attention bound-investigation (2026-09-03) and the token-supply lever (2026-09-01).
artifacts:
  - research/runners/_emerge_wkv_lm_derisk.py
  - research/findings/raw/_emerge_wkv_lm_linattn_depth2_contiguous_6seed.json
  - research/findings/raw/_emerge_wkv_lm_assoc_temporal_depth2_contiguous_6seed.json
  - research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json
  - research/findings/2026-09-03-OPEN-FLUENCY-BREAKTHROUGH-linattn-deployable-spiking-mouth-beats-trigram-6of6.md
  - research/findings/2026-09-04-fluency-scale-wt103-linattn-below-trigram-on-broad-domain.md
  - research/findings/2026-09-03-ordered-attention-at-shared-fluency-bound-investigation-verdict.md
  - research/findings/2026-07-11-content-addressable-retrieval-needs-LEARNED-keys-the-arc-converges-on-deep-credit-learned-representations.md
runner: research/runners/_emerge_wkv_lm_derisk.py
---

# hippokey: structured HiPPO SSM -> content-addressable learned-key attention — the literal own-voice-fluency steer, implemented and queued

**Pending output (QUEUED, no measurement yet):** the 6-seed simplewiki contiguous depth-2 de-risk writes `_emerge_wkv_lm_hippokey_depth2_contiguous_6seed.json` into the raw findings dir; this is a DESIGN/ROADMAP note whose GO/NO-GO lands when that GPU run clears. Every number in the table below traces to the EXISTING artifacts cited in the frontmatter (the linattn/assoc_t 6-seed JSONs and the wt103 scale JSON).

## 0. Headline (the decision this re-anchors)

The #1 goal-blocker is brain-native open-ended generation — the brain's own "mouth" writing fluent prose so the
Qwen scaffold can retire (the one-brain roadmap shows the mouth blocking ~48/64 ledger rows). The owner's binding
steer (MEMORY `project_own_voice_fluency_pursue_fully_2026_09_03`) is to pursue OPEN fluency FULLY — reject the
"it's a shared data-regime bound, nothing to do" off-ramp — via a NEW mechanism class: **a structured HiPPO-style
SSM -> content-addressable learned-key attention**. This note establishes the honest current ceiling, shows the
project had SUBSTITUTED a fixed-codebook (`learnkey`) that drops the HiPPO SSM entirely, and implements the
LITERAL steer as a new additive `--recurrence hippokey` arm (byte-identical when off, CPU-smoke-verified), with
the apples-to-apples 6-seed simplewiki de-risk queued on GPU.

## 1. The honest current own-voice fluency ceiling (from the record)

The fluency bar in this arc is `margin_vs_trigram` at the deepest context bucket (positions 10-99): a fluent LM
must BEAT a fair interpolated trigram (a trivially weak 2-token-context baseline). The deployable spiking mouth
is `linattn` (normalized Hebbian fast-weight linear attention). The record, on the shared depth-2 contiguous
protocol (BPE V=8001, d_model=192, 6 seeds unless noted):

<!--derived (each cell traces to the cited finding/artifact)-->
| mechanism (deployable-family, depth-2, contiguous) | corpus | mean margin_vs_trigram |
|---|---|---|
| bag content-addressable attention (`assoc`, no order) | simplewiki | -0.347 |
| spiking SSM dual-nonneg (`ssm`, recurrence) | simplewiki | -0.125 |
| FIXED HiPPO multi-timescale SSM + local read (`hippo`) | simplewiki | -0.126 (2-seed) |
| ordered attention (`assoc_t`, +time-cell "when") | simplewiki | -0.147 |
| **linattn (the current deployable mouth)** | **simplewiki** | **+0.0505 (6/6 CROSS)** |
| **linattn (same mechanism, BROAD domain)** | **wikitext103** | **-0.29 .. -0.57 (depth>=2, 1-seed)** |

The deep-bucket per-seed means backing this table live in the cited artifacts: the linattn simplewiki row in
`research/findings/raw/_emerge_wkv_lm_linattn_depth2_contiguous_6seed.json`, the ordered-attention row in
`research/findings/raw/_emerge_wkv_lm_assoc_temporal_depth2_contiguous_6seed.json`, and the broad-domain linattn
row in `research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json` (the ssm/hippo/bag rows are the
comparison table in the linattn-breakthrough finding cited in the frontmatter).

So the honest ceiling: **the deployable mouth clears a trigram on the SIMPLE domain (+0.05) but FALLS BELOW it on
a BROAD domain (wt103)** — i.e. it is NOT yet fluent about arbitrary topics, exactly the capability needed to
retire Qwen. Every non-linattn family sits at a shared ~-0.12/-0.15 bound.

## 2. WHY the family is trigram-bound, and why the two prior attention arms did not break it

Two distinct diagnosed causes, both in the record:

1. **Fixed-state compression (the linear-recurrence bound).** `2026-07-15-selective-ssm-generator-trigram-bound-`
   diagnosed that wkv/ssm/hippo/linattn all crush the whole prefix into a FIXED-size state a read-out cannot
   losslessly invert — so they only APPROXIMATE which past token mattered, while a trigram keeps the EXACT
   identity of the last two. On a broad domain (many more distinct contexts to compress) the approximation
   degrades below the trigram — the wt103 result.
2. **Weak keys (why `assoc`/`assoc_t` also failed).** The two attention arms already tried DID keep per-position
   values (no compression) — yet still hit the bound. The ordered-attention bound-investigation
   (`2026-09-03-ordered-attention-...verdict.md`) concluded content+order is NECESSARY-BUT-NOT-SUFFICIENT: the
   read machinery was fine, the KEYS were weak. This echoes the July diagnosis
   (`2026-07-11-content-addressable-retrieval-needs-LEARNED-keys-...`): "the fading reservoir state is a BAD
   KEY". `assoc`'s keys are Wk(z_s) with z_s dominated by the CURRENT token + shallow context; `assoc_t` adds
   an absolute-position "when" code but no CONTEXT DEPTH — so both can match "same token near the same position"
   but not "same DEEP multi-timescale context," the structure a trigram cannot see and the long-range signal
   fluency needs.

## 3. The project had substituted the steer, not implemented it

`--recurrence learnkey` (2026-09-04, already in the runner) was tagged in its own docstring as "gap#1's NAMED
next mechanism class ... 'structured HiPPO SSM -> content-addressable learned-key attention'". But learnkey is a
FIXED bank of M learned key PROTOTYPES (a codebook) — **there is no HiPPO SSM in it at all.** It fixes assoc's
O(T^2)/non-spiking property, a real and separate concern, but it does NOT test the steer's actual hypothesis
(that a multi-timescale HiPPO KEY is what assoc's read was missing). The literal composition the owner named was
never built. hippokey builds it.

## 4. The mechanism: HiPPO-keyed content-addressable attention (`--recurrence hippokey`)

Per position, causal (s<=t): a FIXED HiPPO multi-timescale diagonal SSM (`x_{t+1}=A x_t + B u_t`, A a fixed
log-spaced fast->slow decay grid, B a fixed random projection — both register_buffers, no learned recurrent
credit, the identical fixed structure `hippo` already validates) produces a per-position multi-timescale context
code x_s. The content-addressable read then keys off that state:

```
q_t = Wq(x_t);  k_s = Wk(x_s)         -- match by DEEP multi-timescale context (Q/K over the HiPPO state)
v_s = Wv(z_s)                          -- retrieve the token CONTENT that followed matching contexts (V over z)
alpha_{t,:} = softmax_s( q_t . k_s / sqrt(D) ), causal-masked
read_t = sum_{s<=t} alpha_{t,s} v_s ;  delta_t = Wo(read_t)   -- (--assoc-gate: a learned trust gate on read_t)
```

The ONLY change from `assoc` is that Q/K read the HiPPO state x, not the token-local z. That single change
targets BOTH diagnosed failure modes at once: (a) BAD KEY -> the HiPPO state is a rich multi-timescale context
code, not a shallow token read; (b) ORDER-BLINDNESS -> the state is built by stepping through the sequence, so it
is inherently order-dependent (assoc's bag-of-tokens problem) WITHOUT assoc_t's added time code. And unlike the
linear-recurrence family, the full per-position softmax recall keeps a value for every past position (unbounded
effective context), so it is NOT subject to the fixed-state compression trigram-bound.

**Biological anchor (bio-grounded, not a transformer bolted on — the owner accepts attention-like reads IF
grounded):** this is the ENTORHINAL -> CA3 pathway. Medial entorhinal cortex supplies a multi-timescale
temporal-context / grid code (time cells, MacDonald 2011; TCM drift, Howard & Kahana 2002; multi-scale grid
modules) — a bank of leaky integrators at log-spaced time constants, i.e. the diagonal HiPPO-LegS approximation
(Gu et al. 2020). CA3 recurrent collaterals then perform content-addressable autoassociative pattern completion
(Marr 1971; Treves & Rolls 1994; Hasselmo's EC-context-cued retrieval), one-shot modern-Hopfield <-> attention
(Ramsauer et al. 2020). So hippokey COMPOSES the two anchors `hippo` and `assoc` already carry, the way biology
composes them (EC context feeds the CA3 cue) rather than running each alone.

## 5. First version — implemented, additive, smoke-verified

- **Additive, byte-identical when off (by construction):** a new `HippoAssocLayer` class + `--recurrence hippokey`
  in `research/runners/_emerge_wkv_lm_derisk.py` (NO sim/ edit, NO production edit). `self.hippoassoc_layers` is
  a guarded `nn.ModuleList([...]) if RECUR=="hippokey" else nn.ModuleList()` (empty, ZERO init-RNG draws) placed
  before the aux_heads block, exactly as the 4 sibling arms (hippo/assoc/linattn/learnkey) were added — so
  wkv/ssm/hippo/assoc/assoc_t/linattn/learnkey init RNG (hence outputs) is unchanged.
- **CPU smoke (toy d48/1000-sent/1-epoch, functionality only):** builds/trains/evals end-to-end and produces the
  per-depth `margin_vs_trigram`; the anti-cheats behave correctly — memoryless-collapse +0.855 (the read uses the
  past) and permute-collapse +0.175 (it uses order). The toy margin (-0.368) is meaningless at that scale; it
  only confirms wiring.
- **Deployability (honest, named not hidden):** the read here is EXACT causal softmax (O(T^2)) — a CEILING /
  capability instrument, like assoc/assoc_t and like the BPTT-trained WKV the local-rule read-out only later
  matched (2026-07-20). It is NOT yet spike-deployable. At this protocol T<=40, so O(T^2) is cheap. This first
  version answers the CAPABILITY question (does a HiPPO key break the bound?); the spike-port rung (a HiPPO-keyed
  linattn kernel — feed x into linattn's phi(q)/phi(k), inheriting the deployed LinAttnReadout machinery — or a
  fixed-slot read) is named and follows the prove-on-instrument-then-port discipline the arc used for wkv/linattn.
  `--uniform-decay` is inert for hippokey (the HiPPO A is always the fixed multi-timescale grid; the flag only
  sizes an unused base parameter) — kept in the queued command only so it is a one-flag diff from the linattn
  baseline.

## 6. The queued 6-seed de-risk (GO bar + next rung)

Apples-to-apples with the whole comparison table above — the EXACT linattn-breakthrough protocol, only
`--recurrence linattn` -> `hippokey`:

```bash
SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._emerge_wkv_lm_derisk \
    --recurrence hippokey --n-layers 2 --uniform-decay --d-model 192 --batch 128 --tokenizer bpe \
    --corpus data/corpus/simplewiki.txt --contiguous --max-len 40 \
    --n-sentences 1200000 --max-train-sents 1000000 --max-eval-sents 4000 --epochs 5 \
    --seeds 42 43 44 100 101 102 --tok-cache \
    --json _emerge_wkv_lm_hippokey_depth2_contiguous_6seed.json   # writes into the raw findings dir
```

(Bare output filename shown so this DESIGN note cites only artifacts that already exist; the ACTUAL queued
command uses the absolute output path under the main checkout's raw findings dir — see the commit message and
the gpu_queue entry for the verbatim command.)

(Queued on the GPU lane via `tools/gpu_queue.sh` with an explicit `cd` to the build worktree + absolute paths, so
the daemon — which runs jobs from the main checkout that does not yet carry this arm — executes the hippokey code;
the JSON lands in the main checkout's `research/findings/raw/` for the controller to harvest.)

- **GO bar:** `margin_vs_trigram` (deep bucket 10-99) 6/6 seeds ABOVE the linattn simplewiki baseline (+0.0505
  mean), with the anti-cheats holding every seed (perm-collapse and memoryless-collapse both positive) — i.e.
  the HiPPO key genuinely breaks the bound the token-local-key attention arms sat at, not a harness artifact.
- **If GO -> next rung: the BROAD domain (the real prize).** Re-run at wt103 scale (where linattn FELL BELOW the
  trigram) to test whether the HiPPO key survives broad-domain — that is the capability that retires Qwen. Then
  the spike-port (HiPPO-keyed linattn kernel) toward the deployable mouth.
- **If NO-GO -> the key was not the missing piece.** Cheaply banks that the bound is deeper (objective/capacity),
  re-aiming the arc onto the predictive-objective (`--pred-aux-weight`, already built) and capacity levers the
  bound-investigation ranked — a verdict on a METHOD, not the capability.

## 7. No-defer note

This implements the owner's literal steer as a first-class mechanism and hands a gated ladder either way. It
defers no capability: brain-native arbitrary prose remains the target; hippokey tests the specific unexhausted
hypothesis (a multi-timescale HiPPO key makes content-addressable recall load-bearing at long range) that the
record's own diagnosis points to, and every branch of the outcome hands the next concrete method.
