# Pre-registration: routing screen 3. (X) Keep permanent self episodes out of the S_all recall competition, fetch level, on arm S's own night snapshots; (J) a J-space (Jacobian-lens) key, rank ceiling on the KR snapshots

Written 2026-11-26 before any run. Capability: episodic recall (memory selection). DEV, 3 seen seeds: a lead at most.
This is the last routing method of the arc (HANDOFF timebox: KEEP to 11-27, then STOP).

## 1. Record brief
- **Arm S live (11-25, `results/2026-11-25-sall-live-dev-read-1.5b.md`): KILLED.**
  - Told recall: intact 125/223 against hippocampus-off 133.
  - Intact-frozen fetches: right 18/223, none 127.
  - On 65/223 intact told rows the fetch was the permanent self-name episode ("User: What's your name? / Assistant: My
    name is Nova.", permanent, strength 4-5). lifeKP did this 0 times.
- **Offline the same key took the self episode on 10 of 40 wrong told fetches** (screen 2 on the S0-shaped KR stores).
  Live it took 65, so the capture grows over the S life: a feedback loop the KR snapshots cannot show. Arm S kept
  `--night-snapshots`, so its own stores (nights 1-15, s319-321) exist and are used for part X.
- **Why the self episode wins.** The weighting multiplies similarity by `tanh(effective_strength)`.
  - Permanent episodes: tanh(4-5) is about 1.0.
  - Told episodes: the weight median is 0.36 on the 168 raw-first rows (screen 2 read).
  - So a permanent episode wins any row where its raw similarity is within about a factor of 3 of the right episode's.
- **Ceiling (11-25):** S_all_live ranks the right episode first on 168/217 KR told rows (S0 90; S12 160). It is rank only.
- **J-space (Anthropic 2026, "Verbalizable Representations Form a Global Workspace in Language Models",
  transformer-circuits.pub/2026/workspace).** The Jacobian lens maps a layer-l residual through the mean Jacobian
  J_l = E[dh_L,t' / dh_l,t] (t' >= t, over about 10-1000 pretraining-like prompts) and reads it with the unembedding. The
  directions it picks out carry concepts the model can verbalize, not the surface form, in roughly the 38-92% depth band.
  It was shown on transformers only, so on RWKV-7 it is untested. Hypothesis: a key built from it matches on topic
  ("sister", "cat") rather than question form ("What is my ...?"), which the arm S read named as untested.
- **Record check (11-26, `tools/before_you_build.sh`, corpus current at 26696c103):** no J-lens, Jacobian-lens or
  workspace-key work in brain/ or the findings. The old track's GNW workspace work is unrelated (a spiking organ).
- **Prerequisites.** Near-duplicate separation and a paraphrase-robust cue are "missing" (`brain/PREREQUISITES.md`).
  Expected-fail risks:
  - X: the self fetches become no-fetch rather than right (the told episodes' weight stays low), so X1 may fail.
    Identity rows lose the self episode by construction.
  - J: the J-lens may not carry over to a recurrent cortex at 1.5B. H (plain hidden state) may equal J (the Jacobian
    adds nothing). The store side re-reads episode text, which S0drift showed can differ from the stored keys.
- **Host decision.**
  - X: a new switch `Brain(recall_skip_permanent=False)`, default OFF. When on, `Brain.hear`'s recall passes the
    permanent episodes' indices as `exclude`, the existing mechanism the self-belief uses. It is declared in
    `brain/SCAFFOLDS.md` as a routing scaffold that stands in for a missing self/other separation in the store.
  - J: an offline measurement harness only. The J matrices are a fixed linear readout of the pretrained cortex, computed
    once and never fitted to any battery row.

## 2. Part X: skip permanent episodes, fetch level, on arm S's own snapshots
- **Population.** lifeS s319/s320/s321 night snapshots, nights 5-15, the banked intact-frozen battery rows. Each home is
  opened exactly as `routing_replay.open_snapshot` opens one, with deltas from the S life's final trunk, at the probe
  clock `life.day_start(night + 1, weekend)`, with the probe reseed. The life's `brain_kw` is `key S_all, key_bar
  slept`.
- **Arms**, both FREEZE on:
  - **K:** the S life's own competition (`KeyedHippocampus` S_all, slept bar 0.165 / 0.02, built by
    `Brain._make_hippocampus`). It must reproduce the banked fetch idx and recalled text on every row of a night, or
    the night is void. A seed with more than 2 void nights is void.
  - **KX:** K's similarities and weights with every permanent episode's index excluded (`-1e9`, as `recall(exclude=)`
    does), then the same `_compete`.
- **Engagement (must hold, else UNDEFINED):** KX fetches a permanent episode on 0 rows, and on at least 10 pooled told
  rows K fetched a permanent episode.
- **Bars (KILL first, KX vs K, told rows, pooled unless stated):**
  - X1: right fetches >= max(ceil(1.25 x K), K + 10).
  - X2: right rises on >= 2 of 3 seeds and falls by more than 1 on none.
  - X3: wrong non-permanent fetches at most K's + 5 per seed (the excluded fetches must not move to another wrong
    episode).
  - X4: fires on fire rows (open, greet, never-told, untold near-duplicate) at most K's + 1 per seed.
- **Reported, not barred:** identity rows (name / nova; they lose the self episode); the four H items; per night; and
  the counts of rows going from self to right, self to none, and self to wrong.
- **PASS** = no kill, engagement holds. It earns one live DEV life, arm SX: the arm S job line plus
  `--brain-kw-extra '{"recall_skip_permanent": true}'` on s319-321 (AWS g5, about $25), read on replies by the arm S
  prereg's bars unchanged (P1, P2s, every guard; G7 self-name is the expected risk). Launch 11-26, read 11-27. KILLED
  or UNDEFINED: no live life.

## 3. Part J: J-space key, rank ceiling on the KR snapshots
- **J matrices.**
  - Model and corpus: the pretrained 1.5B cortex (adapters off, fp32). The corpus is the 24 `life_heldout.HELDOUT`
    texts, the first 64 tokens of each, read from the zero state. These texts are used read-only; the language guard
    itself is not touched.
  - Definition: for each layer l, J_l = the mean over prompts and over position pairs t <= t' of
    dh_L,t' / dh_l,t. Here h_l is the residual after block l and h_L the final residual before `ln_out`.
  - Computation: vector-Jacobian products through `PlasticCortex`'s differentiable forward, run on AWS g5 (A10G, about
    1 h, about $1-2; no workstation compute). The matrices are saved with their code commit and sha256.
  - Sanity tests: J of the last layer onto itself is the identity, and on the stub a finite-difference check matches.
- **Population.** The routing ceiling's exactly: KR snapshots s319-321, its non-void nights, its 217 told rows with a
  right episode, opened as `routing_ceiling.ceiling_night` does. The night's cortex has its adapters on (the brain's
  own `LiveCortex.plastic`).
- **Keys** (layers 12 and 18, fixed before data; these are the ceiling's S12 / S18 layers, inside the workspace band).
  Each text is read from base: the store side as `e.text + "\n\n"`, the cue side as `"User: {cue}\n\n"`. Positions are
  every token of that read.
  - **H_l:** the mean of h_l over positions, centred cosine. This is the plain hidden state, the no-Jacobian control.
  - **L_l:** the logit lens. Per position, softmax(W_U . LN_out(h_l,t)); then the mean over positions, the top 25
    tokens kept (the rest 0), and the cosine. This is the no-Jacobian readout control.
  - **Jt_l:** the J-lens: L_l with J_l h_l,t in place of h_l,t.
  - **Jv_l:** the dense form: the centred cosine of J_l (mean h_l).
  - **Reference:** S_all_live and S12 / S18 right@1 on the same rows, from the banked ceiling JSONs. The harness also
    recomputes S0 on every row and must reproduce the ceiling's S0 rank, else that night is void for part J.
- **Bars (pooled right@1 over the 217 rows; 4 J keys, so a +10 margin):**
  - J1 (the Jacobian adds): some J key (Jt_l or Jv_l) scores >= max(H_l, L_l) + 10 at its own layer.
  - J2 (beats the current key): the same key scores >= S_all_live + 10 (>= 178), and >= S_all_live on >= 2 of 3 seeds.
- **Reported:**
  - rank-level self capture (the top is a permanent episode) per key;
  - right@1 on the 89 non-repeated rows;
  - the four H items;
  - the same keys' right@1 with permanent episodes excluded;
  - rank@5.
- **Decision.** LEAD = J1 and J2 for one key. A LEAD earns a fetch-level J key in `KeyedHippocampus`, with its own bar
  fit on the DEV bank and its own prereg, after the 12-04 test2 launch. Otherwise KILLED: this J-lens method on this
  cortex at this scale. If H or L alone clears J2, that is reported as a lead for that key, not for J.

## 4. What closes the arc
- If X is KILLED / UNDEFINED and J is not a LEAD, routing STOPS on 11-27. It is banked as "missing prerequisite
  (near-duplicate separation and a self/other tag in the store) at this scale", and test2 runs with the current memory.
- A negative is a verdict on these methods, not on episodic recall.

## 5. Cost and timebox
- Part X: about 10 min per seed on pool CPU, $0, 11-26 morning.
- Part J: J matrices on AWS about $2; ranks on pool CPU at about 1 h per seed, $0.
- Live SX only if X passes: about $25, run 11-26 to 27.
- Read by end of 11-27. No workstation compute.

## Review (two sonnet lenses, 11-26, before data): OK_WITH_FIXES x2, applied
Part X:
- **Baselines.** K's count is the harness's own replay of the banked S competition. Every X bar is relative to it, so
  no number is chosen after data.
- **Circularity, stated.** The self capture was found on these seeds, and KX can change only rows where a permanent
  episode leads the weighted competition. X1 alone would pass largely by construction. Three bars are added, on the
  told rows where K's weighted winner is permanent ("self rows"):
  - **X5:** self -> right >= 1/3 of the self rows, pooled.
  - **X6:** self -> wrong <= 1/4 of the self rows, pooled.
  - **X7:** told no-fetch at most K's + 5 per seed (the screen 2 K5 lesson; arm S had 127 no-fetch).
- **Self rows counted.** Engagement now counts self rows (the weighted winner is permanent), not only fetched-permanent
  rows: >= 10 pooled.
- **Snapshot opening.** Arm S stores keep `key = None` (S_all keeps no S0 key). The snapshot opener accepts `None ==
  None` and checks the stored delta layers instead. A test covers it.
- **Reproduction tolerance.** The banked fetches are g5 fp16, the replay CPU fp32. A row where K's idx differs from the
  banked one is dropped, not voided, when the banked fetch's `margin_w` is within 0.01 of 0.02 or K's best weighted
  score is within 0.01 of 0.165. Any other mismatch voids the night; more than 2 void nights void the seed. The dropped
  rows are reported.
- **Other read-outs are not barred.** `life.option_lp` and `life_ladder` read the hippocampus directly, not through
  `hear`, so the switch does not reach them. A live SX would be read only on the `hear` battery rows (count.py) and the
  arm S guards. Arm S has abstain off, so `knows()` and `familiarity` do not gate recall.

Part J:
- **Definition.**
  - J_l is the mean over source positions t of the sum over t' >= t of dh_L,t'/dh_l,t, so J_23 = I exactly. This is
    the pair mean times a constant per prompt length.
  - Layers are 0-based: h_l is the output of block l, and h_L is block 23's output, before `ln_out`. The readout is
    `layer_norm(., ln_out) @ head.weight`.
  - The J corpus is read from the zero state with adapters off. The keys are read from `base` with the night's adapters.
- **Checks.** `jlens compute` asserts on prompt 0 that its residual-path logits equal `PlasticCortex.forward` to 1e-4,
  and that TF32 is off (fp32 cortex). The first prompt is timed. If the batched VJP is slower than looped or fails, the
  looped chunked VJP is used; the choice is logged.
- **Centring.** L and Jt are centred on the stored mean like H and Jv. The shared prefix and generic next-token mass
  would otherwise dominate.
- **Controls added.**
  - **S_all_rk:** S_all_live's math on every episode re-keyed from its text through the night's cortex (all
    matrix-memory layers). This separates re-read drift from the key.
  - **Rv<l>:** the centred cosine of R (mean h_l), where R is a random matrix with J_l's singular values (random
    orthogonal U, V; draws 42 / 43 / 44). An orthogonal map alone leaves the centred cosine unchanged, so the spectrum
    is the control.
- **Bars, revised.** A J key passes only if all of these hold at its own layer:
  - **J1:** pooled right@1 >= max(H_l, L_l, max over the Rv_l draws) + 10, and >= that control on >= 2 of 3 seeds.
  - **J1n:** on the 89 non-H rows (items other than pixel / admin / marguerite / volvo), >= max(H_l, L_l) + 5.
  - **J2:** pooled >= max(S_all_live, S_all_rk) + 10, and >= S_all_live on >= 2 of 3 seeds.
  - **J2n:** on the non-H rows, >= S_all_live - 2 (S_all_live already scores 89/89 there).
  - **Scope of J2.** J2's +10 is set against the known 168. A key that passes only with permanent episodes excluded
    ("/np") is not a pass.
- **Void.** A night is void for part J if any row's recomputed S0 rank differs from the ceiling's (both CPU fp32). A
  seed is void if more than 2 nights are void here.
