# Sparse-distributed A→W SPEAK — the naive compose hits a crosstalk/read-out boundary (needs a `language_output` WTA); the multi-bridge dispatch is the proven scaling path

**2026-07-09. Conversation frontier: scale the speakable-on-spikes vocabulary past the 16-word grandmother cap.**

## Context

The talkable brain SPEAKS its answers ON SPIKES via an A→W (articulate→word) read-out, but that read-out caps at ~16 words (the concept-pool architecture: one dedicated ~500-neuron pool PER word, 4 kinds × 4 pools; `concept_speak_demo.py` + `concept_pool_demo.py`). Content beyond 16 is spoken as TEXT (a host scaffold). To make more of the conversation genuinely brain-produced (the owner's minimize-transformer thesis), the speakable-on-spikes vocab must scale to hundreds.

A read-only research gate (2026-07-09) ranked the **biology-preferred path (Rank 1)** as: replace the disjoint per-word pools with a SHARED sparse-distributed pool (Kanerva/Pulvermüller G.20) where each word is a sparse K-of-N pattern, decoded from `language_output` — capacity ~500-2000. `concept_pool_sparse_distributed.py` already builds this (shared_concept_pool + sparse patterns + a TRAINED `shared_concept_pool→language_output` read-out) but only evals RETRIEVAL. The gate's verdict: a COMPOSE of validated pieces; the missing piece is a small A→W SPEAK decode; the KEY UNCERTAINTY = "does driving ONE of many OVERLAPPING sparse codes yield a cleanly-decodable `language_output` pattern, or does crosstalk blur it?"

## The de-risk (`research/runners/_sparse_aw_speak_derisk.py`, reuse-by-import, NO `sim/` edit)

Built the missing A→W SPEAK decode: build + train the sparse-distributed bridge, then drive each concept's sparse pattern in `shared_concept_pool` → accumulate the `language_output` firing vector → cosine-match against each word's `language_output` reference band → decode. Anti-cheats: LESION `shared→language_output` (→ collapse), MOAT (a novel untrained pattern), per-word margins.

## Result — the crosstalk uncertainty resolves as a BOUNDARY for the naive compose

GPU, 16 concepts, `language_output`=8192, sparsity 0.03 (bands non-overlapping: n_active 246 < stride 512, so this is NOT reference crosstalk):

| config (16 concepts) | speak_acc | mean margin | langout_spikes | lesion_acc |
|---|---|---|---|---|
| pattern=100, 100 events | 0.188 (3× chance) | 0.0059 | — | **0.000** |
| pattern=100, 400 events | **0.062 (= chance)** | 0.0053 | 3977 | **0.000** |
| pattern=50 (less overlap), 100 events | **0.250 (4× chance)** | 0.0053 | 3102 | **0.000** |

**The overlap hypothesis is CONFIRMED but partial:** halving the pattern (overlap ~5 → ~1.25 shared neurons) lifts the decode 0.188 → 0.250, so pattern-overlap crosstalk IS a real factor — yet even the low-overlap decode is only 4× chance (margins ~0.005, broad `langout_spikes`), so less overlap alone does NOT reach a clean decode. More TRAINING makes it WORSE (divergence).

**A decode-side top-k WTA does NOT recover it, and a TRAIN-time `language_output` WTA makes it WORSE (both decisive):** keeping only the top-k strongest-firing `language_output` before the cosine gives TOPK **0.188 < raw 0.250** (the word's band is NOT among the strongest-firing units); and adding a `language_output_FS` WTA active DURING training (so only the word band should fire → clean binding) instead COLLAPSES it to **0.062 = chance** (the WTA over-suppresses `language_output`, langout_spikes 3102→1536, and the intact read-out becomes WORSE than the lesioned one — 0.062 < lesion 0.188 — i.e. the WTA-trained read-out actively mis-maps). So a competitive sparsifier — decode-side OR train-side — does NOT fix the specificity. ⇒ the read-out is **fundamentally NON-word-specific** (the sparse pattern does not fire ITS `language_output` band preferentially), so a sparsifier/WTA ALONE cannot fix it. The problem is read-out **specificity** (the shared `shared_pool→language_output` weights, trained across overlapping patterns, don't cleanly separate word bands), NOT just broad firing. This is DISTINCT from the CA3 completion boundary (which needs a selection-timing sparsifier): the A→W read-out needs a **word-specific read-out mechanism** (a weight-cap/homeostatic/decorrelated read-out, or per-word read-out sub-populations), a harder problem than the naive compose implied.

- **The decode is genuinely SPIKING** — lesioning `shared→language_output` collapses it to 0.000 both times (it rides those synapses, not a host lookup). The pipeline runs end-to-end, moat included.
- **But the decode does NOT discriminate** — margins are ~0.005 (near-random among words), and **MORE training makes it WORSE (0.188 → chance)**: the shared `shared_pool→language_output` read-out, trained for many overlapping patterns on one shared pool, DIVERGES to a broad / non-word-specific `language_output` activation. Driving one sparse pattern does not light up ITS word band cleanly.

**⇒ the research's key uncertainty is answered: aggregate crosstalk on the SHARED read-out blurs the SPEAK decode.** The grandmother architecture avoids this only because each word owns a DEDICATED pool (no shared read-out to diverge). This is the SAME sparsification/WTA family as the R-iii CA3 co-residence boundary (CYCLE 1091-1095): a point-neuron read-out fires broadly without a competitive sparsifier. The sparse build has a `shared_FS` WTA on the POOL but **none on `language_output`** — the read-out is un-sparsified.

## The next mechanism (named) + the proven fallback

- **To fix Rank 1 (biology-preferred) — a WORD-SPECIFIC read-out is the primary need (the top-k WTA showed a sparsifier ALONE does not recover the band):** the `shared_pool→language_output` weights must SEPARATE word bands despite overlapping input patterns. Candidate mechanisms (research-gate the choice): (1) a weight-cap / homeostatic read-out that stops the divergence-under-training; (2) a DECORRELATED read-out (each word's `language_output` teacher band chosen ~orthogonal AND the read-out competitively normalized so overlap-neurons don't smear across bands); (3) per-word read-out sub-populations in `language_output` (a middle ground between the shared read-out and the grandmother pools). A `language_output` WTA is a SECONDARY refinement (cleans firing once the read-out is specific), not the fix. This is a genuine read-out-learning sub-problem, harder than the naive compose implied — the honest deliverable of this de-risk.
- **Rank 2 — the PROVEN, LIVE pragmatic scaling path:** the EMERGE-75 multi-bridge A→W dispatch (`_realcorpus_multi_bridge_speaker.py` / `_realcorpus_productive_multi_speaker.py`) — each independent ~16-word grandmother bridge decodes cleanly; a word is dispatched to whichever bridge holds it; utterances stitched across bridges. Works today (all-word spell 1.00, function-word lesion-collapse, moat 0), bounded only by O(N) grandmother bridges (~20 for 320 words) + per-bridge training budget. So broadening the speakable vocab NOW = train + dispatch more 16-word bridges (linear, proven), while the elegant single-shared-pool A→W (Rank 1) awaits the read-out-WTA mechanism.

## UPDATE — the decorrelating read-out kernel WORKS (0.25 → 0.5), a validated partial surpass

A 3rd research gate + my own read pinned the fix: the ALREADY-COMMITTED `sim/kernels.py:432 fused_htm_winner_inactive_depression` (EMERGE-40; Cui-Ahmad-Hawkins 2017 / Diehl-Cook 2015 / Foldiak 1990) is EXACTLY the decorrelating rule — a winning `language_output` band DEPRESSES its synapses from shared-pool neurons NOT in the word's pattern, so each band reads ONLY its pattern. **Mikulasch-Priesemann does NOT apply** (this is learned WEIGHT-space decorrelation, not analog input-side whitening; point neurons proved it: EMERGE-39/40 lifted overlapping-category acc 0.20→0.96). Wired into `_sparse_aw_speak_derisk.py` (`--winner-inactive-ld`, per-word COO-gather of the `shared_pool→language_output` synapses, on-device kernel apply; reuse-by-import, NO `sim/` edit):

| config (16 concepts) | speak_acc | margin | moat (novel margin) | lesion |
|---|---|---|---|---|
| no decorrelation (ld=0) | 0.25 | 0.005 | — | 0.06 (chance) |
| **ld=0.03, pattern=100, 100 ev** | **0.50 (8× chance)** | 0.015 | 0.004 (< trained ✓) | 0.06 (chance ✓) |
| ld=0.1, pattern=100, 100 ev | 0.50 | 0.017 | 0.007 (< trained ✓) | 0.06 ✓ |
| ld=0.05, pattern=50, 200 ev | 0.44 | 0.010 | 0.015 (≈ trained) | 0.06 ✓ |
| ld=0.03 + synaptic-scaling (Turrigiano) | 0.50 | 0.019 | **0.0005** (≪ trained ✓✓) | 0.00 ✓ |
| ld=0.03 + read-out cap w_max=4 | 0.50 | 0.015 | 0.004 (< trained ✓) | 0.06 ✓ |

**The ~0.5 plateau is ROBUST across every lever** (ld 0.03-0.1 / pattern 50-100 / events 100-200 / synaptic-scaling / read-out-cap). The Turrigiano synaptic-scaling stabilizer does NOT lift the accuracy but SHARPENS THE MOAT decisively (a novel pattern's margin drops to 0.0005 ≪ the trained 0.019 → clean abstention). So the decorrelating kernel gives a ROBUST ~50%-word-specific read-out + (with synaptic-scaling) a clean moat, but not the >0.75 GO.

**The mechanism is VALIDATED** (doubles the accuracy, the moat separates, the decode is genuinely spiking — lesion → chance; `ld=0` collapses to the 0.25 boundary, so the term is load-bearing). **But it PLATEAUS at ~0.5** across the cheap tuning (ld / pattern / events) — the read-out is only weakly word-specific. Root cause of the residual: broad `shared_concept_pool` firing during decode (the 0.05 internal recurrence + `off_target_factor=0.1` lang_input leak fire NON-pattern shared neurons → they drive `language_output`), which the read-out-weight rule does not touch. `langout_spikes` stays ~3000 (broad).

## Honest status + the named next lever

**A validated PARTIAL surpass with a GENUINE ~50% CEILING (decisive):** the research-gated decorrelating read-out kernel doubles the A→W speak accuracy (0.25→0.50) via a committed kernel, NO `sim/` edit, and (with Turrigiano synaptic-scaling) gives a CLEAN MOAT (novel_margin 0.0005). But a **WHITENED decode** — subtracting the per-neuron common mode across all word decodes = the IDEAL/best-case output decorrelation (the effect a Foldiak anti-Hebbian `language_output` lateral would learn) — gives **WHITE = 0.50, identical to the raw 0.50.** ⇒ the remaining ~50% of words' specificity is NOT recoverable even by ideal output decorrelation; the info is IRREDUCIBLY LOST in the shared read-out (the overlap-neurons' contribution is ambiguous). So a Foldiak lateral would NOT reach the GO — the shared sparse read-out has a genuine ~50%-word-specificity CEILING at these params. **⇒ Rank-2 (the EMERGE-75 multi-bridge dispatch, dedicated per-word pools = word-specific by construction, live) is the answer for the full-fidelity speakable vocab.** The sparse-shared-read-out (Rank-1) is a validated HALF-solution (doubles the accuracy + a clean moat via the committed kernel) but caps at ~50%; it would suit a coarse/abstaining speaker, not full fidelity. **PROVEN pragmatic fallback:** the EMERGE-75 multi-bridge dispatch (dedicated per-word pools = word-specific by construction, live) broadens the speakable vocab today. Cross-cutting theme: the point-neuron substrate's spiking READ-OUTs are the recurring hard part — CA3 completion needs a selection sparsifier; A→W articulation needs read-out specificity (now half-solved by the committed decorrelating kernel) + clean pool firing.
