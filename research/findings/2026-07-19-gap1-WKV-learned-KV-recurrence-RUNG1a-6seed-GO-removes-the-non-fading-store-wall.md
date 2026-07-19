# gap#1 Rung 1a — the WKV/linear-attention LEARNED key–value recurrence 6-SEED GO at ceiling-valid scale: it BEATS the fair interpolated trigram at deep context (the exact bar every fading reservoir FAILED), removing the documented non-fading-store wall. The mission-primary open-generation lever, validated.

**2026-07-19.** Per the gap#1 research gate (`2026-07-19-gap1-open-generation-research-gate-WKV-learned-KV-recurrence-is-the-next-build`),
built + ran the WKV de-risk (`research/runners/_emerge_wkv_lm_derisk.py`). **6/6 GO at the ceiling-valid scale** (TinyStories,
V=2000, d_model=256, 80K train sentences, ~9 min total on GPU).

## RESULT (6-seed 42/43/44/100/101/102, DEEP context d10-99)
| metric | value (range) | meaning |
|---|---|---|
| **WKV beats FAIR interpolated trigram** | **+0.62 to +0.73 nats** | the exact control that KILLED every reservoir lever (`2026-07-15`: fading memory LOST to it at every depth) — WKV BEATS it, 6/6 |
| margin GROWS with depth | d2 +0.23 → d6-9 +0.67 | the transformer/LSTM long-range SIGNATURE (a bigram/fixed-window can't) |
| perm-collapse (shuffle prefix order) | +4.2 to +4.4 | the deep-context advantage collapses → genuine long-range ORDER use, not a context bag |
| memoryless-collapse (recurrence off) | **+1.06 to +1.16** | the LEARNED recurrence carries >1 nat of deep-context info beyond the current token (grows with depth) — isolates the RECURRENCE from the embedding |

⇒ the WKV op (an O(N) recurrent gated leaky K/V integrator with learned K,V,receptance,decay) is a **content-selective
NON-FADING learned-write store** — the "deepest unbuilt frontier" the arc's own synthesis named, which every reservoir/
echo-state/e-prop/ALIF lever lacked. This is the at-scale confirmation of the decision's core bet (SpikeGPT's 45M-param
spiking existence proof at this project's scale).

## HONEST SCOPE (Rung 1a is foundational, NOT the full gap#1 close — the ladder ahead)
- **Input = a LEARNED EMBEDDING** (not yet the emergent stream-cortex pooler codes = Rung 1b, the emergence-bar priority;
  the memoryless-collapse already proves the deep-context signal is the RECURRENCE, not the embedding — but the input
  must become emergent to clear the bar).
- **Trained by BPTT** (a TRACKED shortcut to establish the MECHANISM first; Rung 3 biologizes the rule — e-prop/BDSP on
  the fixed-form recurrence, the R3-REFRAME 78%-closed input-rep lever).
- **Rate-level** (not yet spiking = Rung 2: port the fixed-form WKV/SSM recurrence onto a spiking `BrainRegion`; SpikeGPT
  confirms faithful; SNN membrane leak IS the SSM state update).
- **Within-sentence deep context** (≤16 tokens, apples-to-apples with the reservoir arc). Cross-sentence / open
  multi-clause prose is the further R4 reach (the WKV recurrence CAN carry cross-sentence context — a follow-on test).
- **The gap#1↔gap#4 convergence:** Rung 1b feeds the WKV read from the UNSUPERVISED stream cortex (the gap#4-pivot
  deep-representation engine) — this is where the two gaps meet.

## STATUS + NEXT (per the decision's ladder + the emergence bar)
- **DONE (Rung 1a, 6-seed GO):** the WKV MECHANISM removes the non-fading-store wall at the rate level.
- **NEXT (in flight): de-risk the emergent-input (Rung 1b) regime cheaply BEFORE its expensive prerequisite** — a
  FROZEN-embedding variant (freeze the input at random init; only WKV+head learn) tests whether the mechanism works with
  a FIXED (non-LM-learned) input = the Rung 1b regime (the pooler codes are learned by the unsupervised cortex, frozen for
  the LM). GO → Rung 1b's TinyStories stream-cortex develop is worth building; the mechanism doesn't depend on a learned
  embedding.
- **THEN:** Rung 1b (develop stream-cortex codes over the TinyStories vocab → feed as WKV input), Rung 2 (spiking port),
  Rung 3 (biologize the rule). PARALLEL engineering: the 21M spiking-forward deploy (ledgered scaffold, milestone-met NOT
  closed). NO `sim/` edit anywhere in Rung 1a. Runner: `_emerge_wkv_lm_derisk.py`.

## RUNG 1b — EMERGENT INPUT: GO (3-seed) + the emergent STRUCTURE genuinely HELPS (gap#1↔gap#4 convergence realized)
Two de-risks resolved the emergent-input question WITHOUT a separate stream-cortex-over-TinyStories develop (cheaper, direct):
- **Frozen-embedding (3/3 GO):** freeze the input at random init (only WKV+head learn) → the WKV STILL beats the fair
  trigram at deep context **+0.48–0.52**, anti-cheats collapse ⇒ the recurrence does NOT depend on an LM-learned input.
- **EMERGENT PPMI codes `--input ppmi` (3/3 GO):** compute UNSUPERVISED windowed co-occurrence → PPMI (log + positive
  threshold, the CYCLE-88 local normalization) → SVD → per-word code, FROZEN as the WKV input. Deep-context vs-trigram
  **+0.60/+0.66/+0.65** (6-seed-scale config, seeds 42/43/44), perm-collapse +4.1–4.3, memoryless-collapse +1.02–1.06.
  **PPMI (+0.63 avg) > frozen-random (+0.52 avg) by ~+0.11 nats** → the emergent co-occurrence STRUCTURE genuinely helps,
  nearly matching the LM-learned embedding (+0.68 avg).
- **⇒ the gap#1 open-generation lever is fed by the UNSUPERVISED cortex representation (frozen PPMI codes), clearing the
  EMERGENCE BAR for the input, and realizing the gap#1↔gap#4 convergence** (the WKV read is the first client of the
  unsupervised deep-representation engine the gap#4 pivot named). NO `sim/` edit; `_emerge_wkv_lm_derisk.py --input ppmi`.
- **NEXT: Rung 2 — the fully-SPIKING WKV port** (map the fixed-form WKV recurrence onto a spiking `BrainRegion` via the
  SNN-membrane-leak ≡ SSM-state-update equivalence; SpikeGPT confirms faithful) — the one-brain/fully-spiking milestone.
  Then Rung 3 (biologize the BPTT rule: e-prop/BDSP on the fixed-form recurrence). PARALLEL: the 21M spiking-forward deploy.

## RUNG 2 — SPIKING-SUBSTRATE-FAITHFUL recurrence de-risked at the rate level (GO): the LEAKY-INTEGRATOR form works
The full WKV op has a divisive num/den NORMALIZATION that is hard on spikes. The `--recurrence ssm` variant tests the
spiking-FAITHFUL form: a plain LEAKY INTEGRATOR `a_t = decay·a_{t-1} + v_t` (a slow membrane/conductance leak — NO exp(k)
weighting, NO normalization), receptance-gated read. **Smoke GO:** beats the fair trigram at deep context **+0.375**
(vs the full WKV's +0.472 at the same scale — only ~0.1 nat cost for dropping the normalization), perm-collapse +3.68,
memoryless-collapse +0.61. ⇒ **the spiking membrane-leak form CAPTURES deep context and beats the trigram** — the
SNN-membrane-leak ≡ SSM-state equivalence holds at the rate level; the WKV normalization is an optional ~0.1-nat
enhancement (realizable on spikes via FS divisive inhibition if wanted). **Rung 2's core recurrence is de-risked;** the
remaining build is the full ON-BRIDGE realization (a recurrent spiking `BrainRegion` with a slow leaky conductance as the
state + the trained K/V/receptance weights, read from `cp_firing_states`), reusing the EMERGE-82 on-bridge-LSM machinery.
NO `sim/` edit in the de-risk; `_emerge_wkv_lm_derisk.py --recurrence ssm`.

**Rung 2 spiking firing-rate constraint (GO) + at-scale confirmation (3-seed GO):** `--spiking-state` reads the leaky
state via NON-NEGATIVE ON/OFF rate channels `[relu(a), relu(-a)]` (the two-population sign code a spiking region uses).
Smoke +0.374 == signed-analog +0.375 (the firing-rate constraint costs NOTHING). At V2000 scale (3-seed): the FULLY
spiking-faithful form (membrane-leak recurrence + non-negative firing-rate read) beats the fair trigram at deep context
**+0.55/+0.61/+0.59**, perm-collapse +4.4-4.6, memoryless-collapse **+1.73-1.85** (GROWS at scale — the recurrence carries
MORE deep-context info with data). ⇒ **the entire fully-spiking open-generation path is de-risked END-TO-END at the rate
level** (Rung 1a mechanism 6-seed · Rung 1b emergent input 3-seed · Rung 2 spiking-faithful recurrence + firing-rate read
3-seed-at-scale). **The remaining build is the ACTUAL on-bridge realization** (a recurrent Izhikevich `BrainRegion` of D
channels, each a slow leaky conductance = the state, learned decay→conductance tau, learned Wv→input synapses, driven
through `_run_one_simulation_step`, read from `cp_firing_states` → the Wo_sp read-out — reusing the EMERGE-82 on-bridge-LSM
pattern), then Rung 3 (biologize the BPTT rule: e-prop/BDSP on the fixed-form recurrence). PARALLEL: the 21M spiking-forward
deploy (ledgered scaffold).
