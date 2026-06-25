# Foundational-curriculum Step-1 generalization miss — DEEP scoping (2x-refuted; the prior comparisons were FLAWED)

**Date:** 2026-06-25
**Type:** RESEARCH-GATE deep-research scoping (READ-ONLY; NO edits/runs/webapp/GPU). Standing-practice deep-research at a 2x-REFUTED boundary BEFORE building/running a fix.
**Trigger (two refuted hypotheses, foregrounded):**
1. The **wrong-yardstick** hypothesis (`_curriculum_gen_miss_scoping.md`, "swap the gen reference → ≥0.80") **FAILED**: coherent-ref re-measure moved gen only 0.153 → 0.167 (Pearson +0.070 → +0.082).
2. The **content-vocab** hypothesis (`_curriculum_gen_miss_REAL_scoping.md`, "`--vocab-filter content` → ≥0.80, the validated recipe corpus-frequency-ranked") **FAILED**: content-filtered gen = **0.125** (per the prompt), *below* the 0.153 it was meant to fix.

**⇒ BOTH prior scopings predicted ≥0.80 confidently and BOTH were empirically refuted.** This scoping therefore does NOT predict a number — it DESIGNS the decisive isolation controls and pins WHICH of the three live causes (scale / pipeline-diff / corpus) is dominant.
**Result JSONs:** `_curriculum_step1_320_real_corpus_seed42.json` (the `--vocab-filter all` run: gen 0.153, corr(M,C) 0.756, recall 1.0, moat 0-FA). The content-filtered 0.125 + coherent-ref 0.167 are prompt-reported re-measures (no separate JSON read this session).
**Runner:** `research/runners/_curriculum_step1_320_real_corpus.py`

---

## TL;DR verdict — the prior "byte-identical pipeline" claim is FALSE; the dominant cause is most likely a PIPELINE/SCALE diff, NOT the vocab

The prior REAL scoping's load-bearing claim — *"SAME winning pipeline … on the SAME corpus … only the VOCAB differs (64 curated → 320 freq), and gen collapses 0.91 → 0.15"* — is **source-refuted**. The validated **0.91** and the Step-1 **0.15** are NOT the same pipeline. They differ on **at least three uncontrolled axes simultaneously**, any of which can cause 0.91→0.15:

| axis | validated 0.91 run (`_phaseB_online_stream_cortex_derisk.py`) | Step-1 320-run (`_curriculum_step1_320_real_corpus.py`) |
|---|---|---|
| **SUBSTRATE (the big one)** | **pure NUMPY host** — `M[t,h] += 1.0` is an **exact integer co-occurrence count** | **SPIKING BRIDGE** — `M` = population block-mean of **rate-Hebbian-learned synaptic weights**, a *noisy* estimate (corr(M,C)=**0.756**, NOT 1.0) |
| **n_per** | **1** (single, but the count is exact so noise-free) | **16** (population, but reading noisy spiking weights) |
| **n_hub** | **500** | **300** |
| **chance / #categories** | 8 categories → **chance 0.125** | 30 categories → **chance 0.033** |
| **#concepts** | **64** | **320** |
| vocab | 64 curated coherent (`TAXONOMY_8x8`) | 320 freq-selected |
| corpus | TinyStories | TinyStories (byte-identical loader + tokenizer — VERIFIED) |
| normalization | `double_center(log1p(M·100))` | `double_center(log1p(M·100))` (byte-identical — VERIFIED) |

**The prior scoping controlled normalization + corpus (correctly) but treated the substrate (numpy-exact-count vs spiking-noisy-population), n_hub, #concepts, and chance as if they were held — they are NOT.** The decisive, never-run control is: **the validated 0.91 has NEVER been reproduced on the bridge.** The ONLY on-bridge generalization number that exists for the 64-curated vocab is **gen 0.45** (`_phaseB_onbridge_stream_cortex_derisk.py`, 30K windows — the prior REAL scoping itself cites this at its line 13 but then ignores it). So the on-bridge pipeline already drops 0.91 → 0.45 *on the validated curated vocab* — **half the collapse is the substrate, before vocab or scale even enter.**

**Dominant-cause ranking (pinned, evidence-based, NOT a confident prediction):**
1. **PIPELINE: numpy-exact-count → spiking-noisy-population read-out (corr(M,C)=0.756, not 1.0) — LIKELY DOMINANT.** The validated 0.91 is a *noise-free integer count*; the on-bridge read is a noisy population estimate that already gave 0.45 at 64 words. The fine category-similarity structure generalization needs is the FIRST casualty of read-out noise (recall/moat survive because they only need codes *distinguishable*, not *similarity-faithful* — exactly what Step-1 shows: recall 1.0, moat 0-FA, gen 0.15).
2. **SCALE + the chance/metric artifact — LIKELY A REAL CONTRIBUTOR (under-weighted by both priors).** `heldout_generalization` is a nearest-**category**-centroid accuracy; at 30 categories (chance 0.033) the SAME code-similarity quality scores far lower in *absolute* gen than at 8 categories (chance 0.125), AND a 5× larger target vocab over a *smaller* hub pool (300 vs 500) gives each concept a thinner, more-overlapping context fingerprint. The validated curated-320 (CYCLE-96) **NEVER reported a clean generalization number** — its GO was recall+moat+familiarity-gap; the "(the gate's) gen" was never printed. So "curated-320 held its gen" is UNVERIFIED — the prior scoping's "scale RULED OUT" rests on a number that does not exist.
3. **VOCAB / corpus-flatness — a CONTRIBUTOR, but REFUTED as the sole/dominant cause.** The content-filter (drop the 48% flat adjective/function/emotion words, keep entities+verbs) moved gen 0.153 → **0.125** (the prompt) — it did NOT lift it and arguably *hurt* it (fewer concepts → fewer per-category members → the nearest-category metric is harsher; and the entity-only vocab over the same noisy substrate still can't recover fine structure). So flat-word dominance is NOT the binding constraint at 320 on this substrate.

**VERDICT: the gen miss is most plausibly the SPIKING-SUBSTRATE READ-OUT FIDELITY (numpy-exact-count → noisy-population) AT SCALE (320 cats, thin hub), NOT the vocab.** This is a **goal-blocker** (generalization = relate-to-anything richness) but it is **likely closeable** by lifting on-bridge read-out fidelity (more windows / bigger population n_per / bigger hub) — the documented rate-code-wall lever — OR it is the honest on-bridge fidelity boundary, which a clean control settles. The recall-1.000 + moat-0-FA WIN stands (reference-independent). **The mechanism (Hebbian co-occurrence + log-double-centre) is validated; what's unproven is that the SPIKING read-out preserves the FINE similarity structure at 320-concept scale.** The decisive controls below SPLIT pipeline-vs-scale-vs-vocab by varying ONE axis at a time from a reproduced-validated anchor.

---

## 1. DIAGNOSIS — the validated-0.91 EXACT config, and the PRECISE diffs vs the 320-run

### 1a. The validated 0.91 — EXACT config (source-verified, `_phaseB_online_stream_cortex_derisk.py`)

| field | value | source |
|---|---|---|
| runner | `_phaseB_online_stream_cortex_derisk.py` (CYCLE 94) | finding `2026-06-15-biology-faithful-online-stream-cortex-reaches-target.md` |
| **substrate** | **pure NUMPY** — NO bridge, NO spikes | `:81` `M = np.zeros((Nt, N_HUB))`; `:99` `M[tgt_row[w], hub_idx[u]] += 1.0` (exact integer count) |
| vocab | **`TAXONOMY_8x8` = 64 concepts, 8 categories** | `:122` `taxonomy_to_vocab_categories(TAXONOMY_8x8)`; `option_c_…:108-117` (animals/food/… 8×8) |
| **chance** | **1/8 = 0.125** | `heldout_generalization` returns `1/len(cats)` |
| n_hub | **500** | `:47` `N_HUB = 500` |
| n_per | **1** (single neuron — but the count is EXACT, so noise-free) | numpy host count, no population |
| corpus | TinyStories (`data/corpus/tinystories.txt`) | `:56-60` `re.findall(r"[a-z]+", …)` |
| co-occurrence | EXACT online integer count `M[a,b]+=1` in a WM window=2 | `:80-100` |
| normalization | `double_center(np.log1p(M * 100.0))` | `:103` |
| gen metric | `heldout_generalization(code, cat_ids)` (nearest-category centroid acc) | `:105` (imported from `dendritic_d1_…:187`) |
| **gen reference** | the `TAXONOMY_8x8` category blocks (`cat_ids`) | `:68` `S_true` from `cat_ids` |
| **RESULT** | **gen 0.91, Pearson(cos,S_true) +0.513**, 3 seeds | finding line 28-30 |

### 1b. The Step-1 320-run — EXACT config (source-verified, `_curriculum_step1_320_real_corpus.py`)

| field | value | source |
|---|---|---|
| runner | `_curriculum_step1_320_real_corpus.py` | this run |
| **substrate** | **SPIKING `SimulationBridge`** via `build_stream_bridge` (hub+target regions, rate-Hebbian hub→target) | `:80,:396`; `_phaseB_onbridge_stream_cortex_derisk.py:62-91` |
| **read-out of M** | population **block-mean of LEARNED synaptic weights** read off `cp_connections` — a NOISY estimate | `:460-463` `W=…todense(); blk=…mean(axis=(1,3)); M=blk.T` |
| **corr(M,C)** | **0.756** (NOT 1.0 — the spiking read is imperfect) | result JSON `corr_MC` |
| vocab | 320 freq-selected (`all`) / content (`content`) / curated (`curated`) | `derive_curriculum_from_corpus` `:238` |
| **chance** | **1/30 = 0.033** (30 categories present) | result JSON `chance 0.0333` |
| n_hub | **300** | `--n-hub` default 300 |
| n_per | **16** (population) | `--n-per` default 16 |
| corpus | TinyStories (byte-identical loader+tokenizer to 1a — VERIFIED `corpus_stream.py:21-22,48-50`) | — |
| normalization | `double_center(np.log1p(M * 100.0))` — byte-identical to 1a | `:464` |
| gen metric | `heldout_generalization` — byte-identical import | `:83` |
| gen reference | `sharding` (full g20) OR `coherent` (TAXONOMY_40x8 + coherent g20) | `measure_generalization` `:620` |
| **RESULT** | gen **0.153** (all/sharding) · **0.167** (all/coherent) · **0.125** (content) | JSON + prompt |

### 1c. The PRECISE diffs (which plausibly cause 0.91→0.12), ranked

**DIFF 1 — SUBSTRATE: numpy-exact-integer-count vs spiking-noisy-population-read (the axis BOTH priors missed).**
- Validated: `M[a,b]+=1` is the *true count*, corr(M,C)=**1.0** by construction.
- Step-1: `M` is read from learned spiking synaptic weights, corr(M,C)=**0.756**. The 24.4% the read-out loses is **exactly the fine off-diagonal similarity structure** generalization depends on (the soft-bound Hebbian `Δw=rate·(w_max−w)` saturates/compresses, and population block-mean averages spiking noise but does not eliminate it).
- **The smoking gun the priors cite-then-ignore:** the on-bridge 64-curated run (`_phaseB_onbridge_stream_cortex_derisk.py`, 30K windows) reports **gen 0.45** — NOT 0.91. So on the **same curated vocab**, just moving numpy-exact-count → spiking-population-read drops gen **0.91 → 0.45**. That is HALF the total collapse, attributable to the substrate alone, BEFORE 320-scale / thin-hub / chance enter.
- **This single fact refutes the prior "only the vocab differs" framing.** The substrate differs, and it demonstrably costs ~0.46 of gen on the validated vocab.

**DIFF 2 — SCALE + the chance/metric artifact (under-weighted by both priors).**
- `heldout_generalization` = nearest-**category** accuracy. At **30 categories (chance 0.033)** a given code-similarity quality maps to a much lower *absolute* accuracy than at **8 categories (chance 0.125)**: more categories = more ways for the nearest centroid to be a wrong neighbour. Comparing the *raw* 0.91 (8-cat) to 0.15 (30-cat) is partly a chance-baseline artifact — `ratio_vs_chance` is 7.3× (validated) vs 4.6× (Step-1), a smaller but real gap, NOT a 6× one.
- **n_hub 300 over 320 targets vs n_hub 500 over 64 targets:** the validated run had ~7.8 hubs/target of context budget; Step-1 has ~0.94 hub/target. Each concept's context fingerprint is far thinner + more overlapping at 320 → less separable codes.
- **The "curated-320 held its gen" claim is UNVERIFIED.** CYCLE-96 (`2026-06-15-on-bridge-hebbian-…:203-220`) reports the curated-320 GO as **recall 1.00 + moat 0-FA + familiarity-gap** only — it says "(the gate's) gen" but **never prints a generalization number**. `_phaseB_stdp_cooccurrence_derisk.py` (the runner behind that finding) uses `TAXONOMY_8x8` (**64**, not 320) for its gen table, and that table reports `corr(M,C)` + Pearson "normalized code", NOT held-out gen at 320. **So there is NO validated on-bridge 320-concept generalization data point anywhere.** The prior scoping's "(a) SCALE RULED OUT (curated-320 held)" rests on a number that does not exist in the findings.

**DIFF 3 — VOCAB flatness (the prior REAL hypothesis — now REFUTED as dominant).**
- The `all` freq-top-320 is ~48% flat adjective/function/emotion words (the prior scoping's correct observation).
- BUT the **content-filter that removes them gave gen 0.125** (prompt) — it did not help and likely hurt (dropping categories shrinks per-category membership; the harsher 30→fewer-cat metric + the same noisy substrate). So flat-word dominance is a *contributor to code homogenization* but is **NOT the binding constraint** — removing it does not recover gen. This is the second refutation, and it points the finger AWAY from vocab and TOWARD substrate+scale (DIFF 1+2).

**Honest note on what IS controlled (the priors got these right):** normalization is byte-identical (`double_center(log1p(M·100))`, line-confirmed in both); corpus + tokenizer are byte-identical (`corpus_stream.py` mirrors the validated `re.findall(r"[a-z]+")`); the gen metric import is identical. So the cause is NOT normalization, NOT tokenizer, NOT corpus. It is **substrate (read-out fidelity) × scale (categories+hub) × (residually) vocab.**

---

## 2. REFRAME — what the validated arc ACTUALLY proved on the bridge (and what it did NOT)

The CYCLE 88→96 arc proved, **in stages on different substrates**, and the priors conflated the stages:

| claim | substrate | vocab | gen reported | source |
|---|---|---|---|---|
| online stream cortex reaches target + generalizes | **numpy (exact count)** | 64 curated | **0.91** | CYCLE 94 finding line 28 |
| mechanism: spiking Hebbian learns M~C | bridge | 64 curated | — (corr(M,C) 0.705; gen NOT the deliverable) | CYCLE 95 GO finding |
| population lift to host fidelity | bridge | 64 curated | — (corr(M,C)→0.93; "normalized code" Pearson, NOT gen) | CYCLE 95 finding lift table |
| capstone: full stream cortex on bridge | **bridge (noisy read)** | 64 curated | **0.45** | CYCLE 95 finding line 138 |
| conversation on stream-learned codes | bridge | 64 curated | (recall 1.0, moat — gen not gated) | CYCLE 95 finding |
| 320-scale | bridge | 320 **curated** | **NONE PRINTED** (recall+moat+gap only) | CYCLE 96 finding lines 203-220 |

**The reframe (load-bearing): generalization 0.91 was ONLY ever achieved with a noise-free numpy integer count at 64 concepts / 8 categories. On the spiking bridge, the only generalization number ever measured is 0.45 (64 curated). NO on-bridge generalization was EVER measured at 320 concepts — neither curated nor frequency.** So "the validated run generalized at 320 and Step-1 broke it" is not a thing that happened: the validated arc never tested on-bridge 320 generalization at all. Step-1 is the FIRST on-bridge 320-concept generalization measurement in the project's history — and a FIRST data point of 0.15 against a never-established 0.91-on-bridge-320 baseline is not a "regression," it is an **unmeasured cell finally being measured.**

**The biology angle (real distributional semantics):** a distributional cortex's generalization lives in the *off-diagonal* code-similarity (co-occurring words → similar codes). That structure is exactly what a low-SNR spiking read-out blurs first (the diagonal "this concept fired" survives noise; the fine "how similar is concept A to B" does not — hence recall/moat hold while gen collapses). The population code (CYCLE 91) was the documented lift for the *single-neuron rate-code wall* — but it was validated for a **fixed host-PPMI drive** and for **corr(M,C)/normalized-code Pearson**, NOT for *held-out generalization at 320 categories*. Whether population averaging recovers the *fine off-diagonal similarity* well enough for nearest-category gen at 30 categories is **the precise open question** — never tested.

---

## 3. The decisive ISOLATION controls — the MINIMAL GPU set that SPLITS pipeline-vs-scale-vs-corpus

**Design principle: reproduce the validated EXACT config IN THIS RUNNER's substrate, then vary ONE axis at a time.** This is the control the two priors skipped (they re-scored / re-filtered the SAME 320-bridge run instead of building the apples-to-apples ladder). The anchor question: *does the spiking bridge reproduce 0.91 on the validated 64-curated vocab AT ALL?* If not, the substrate is the cause and no vocab fix can work.

All runs READ-ONLY-safe to design here; the controller runs them on GPU. Each prints `gen` on BOTH references + corr(M,C) + recall + moat + derangement + frozen-flat (the runner already emits these).

### Control ladder (cheapest decisive first)

**C0 — INSTANT host re-score (no GPU): reproduce the validated NUMPY run's 0.91, then drop n_hub 500→300 and 64→a-320-subset IN NUMPY.**
Run the *validated* `_phaseB_online_stream_cortex_derisk.py` verbatim (numpy, ~minutes CPU) → must print **gen ≈ 0.91** (confirms the validated baseline reproduces at all). Then a 3-line host variant: same numpy exact-count pipeline but (i) n_hub 300, (ii) the 320 frequency vocab + its 30-category cat_ids, (iii) chance 0.033.
- **Splits the chance/scale/hub artifact from the substrate** with ZERO GPU: if the numpy-exact-count pipeline ALSO collapses to ~0.15 at 320-cat/300-hub, then **scale+metric (DIFF 2) is the dominant cause and the spiking substrate is exonerated**. If numpy-exact-count STAYS high (~0.7+) at 320, then **the spiking read-out (DIFF 1) is the dominant cause.** This one cheap host run is the single most decisive measurement and should run FIRST.
```
SIM_BACKEND=numpy python -u -m research.runners._phaseB_online_stream_cortex_derisk
# then (controller adds a tiny host harness, NO sim/ edit): same numpy M[t,h]+=1 count on the
# 320 freq-vocab + its cat_ids + n_hub=300, score heldout_generalization → the numpy-320 gen.
```

**C1 — THE ANCHOR (GPU): reproduce the validated 64-curated vocab ON THE BRIDGE via this runner.**
`--vocab-filter curated` is the wrong control (it's TAXONOMY_40x8 = 320 words). The needed anchor is the **64-word `TAXONOMY_8x8` on the bridge**. The runner does not expose that directly, but `_phaseB_onbridge_stream_cortex_derisk.py` DOES (it IS the 64-curated on-bridge run). So:
```
# the on-bridge 64-curated anchor at the validated n_hub=500, longer windows:
SIM_BACKEND=cupy python -u -m research.runners._phaseB_onbridge_stream_cortex_derisk \
    --seeds 42 --n-per 16 --n-hub 500 --max-windows 150000
```
- **Expected from the finding:** gen ≈ 0.45 at 30K; this run uses 150K windows + the same population n_per=16 + n_hub=500 as the validated-numpy n_hub. **If gen stays ≪ 0.91 here, the substrate read-out is confirmed as the dominant cause** (same 64 vocab, same coherent reference, only numpy→bridge changed). If it climbs toward 0.91 with more windows, the lever is window budget.

**C2 — VARY n_per (population fidelity) on the 320 bridge run, ONE axis.**
```
SIM_BACKEND=cupy python -u -m research.runners._curriculum_step1_320_real_corpus \
    --seeds 42 --n-concepts 320 --vocab-filter content --gen-reference coherent \
    --n-per 32 --n-hub 500 --max-windows 300000 \
    --out research/findings/raw/_curriculum_step1_320_hifi.json
```
- Bumps n_per 16→32 (population lift), n_hub 300→500 (context budget), windows 150K→300K (read-out SNR). **If gen rises materially, the read-out-fidelity hypothesis (DIFF 1+2) is confirmed and the lever is fidelity.** If it stays flat at ~0.12, on-bridge 320-gen is a genuine boundary regardless of fidelity.

**C3 — the curated-320 control (the prior's #3), but now interpreted correctly.**
```
SIM_BACKEND=cupy python -u -m research.runners._curriculum_step1_320_real_corpus \
    --seeds 42 --n-concepts 320 --vocab-filter curated --gen-reference coherent \
    --n-per 32 --n-hub 500 --max-windows 300000 \
    --out research/findings/raw/_curriculum_step1_320_curated_hifi.json
```
- This is the FIRST-EVER on-bridge curated-320 generalization measurement. **If curated-320 on the bridge ALSO misses 0.80, then the prior scoping's "curated-320 held" was an unverified assumption and the cause is substrate+scale, NOT vocab — vocab fixes are futile.** If curated-320 clears 0.80 but content-320 does not, THEN vocab is the residual (and the prior REAL scoping is partly vindicated — but only after C0/C1/C2 isolate the substrate).

### What each outcome concludes (the decision table)

| C0 numpy-320 | C1 bridge-64 | C2/C3 bridge-320 hi-fi | ⇒ dominant cause |
|---|---|---|---|
| stays high (~0.7) | drops (~0.45) | low | **SUBSTRATE read-out (DIFF 1)** — lever = on-bridge fidelity (n_per/windows) or honest boundary |
| collapses (~0.15) | (n/a, exonerated) | low | **SCALE+metric (DIFF 2)** — 30-cat chance + thin hub; the metric/curriculum granularity, not the substrate |
| stays high | climbs with windows | rises | **fidelity-recoverable** — lever = more windows / bigger population |
| stays high | drops | curated-320 clears, content-320 misses | **VOCAB residual** (prior REAL partly right) — but only confirmed AFTER substrate ruled out |

**Minimal decisive set: C0 (instant, host) FIRST — it alone splits substrate-vs-scale with zero GPU. Then C1 (the bridge-64 anchor) + C2 (320 hi-fi), one GPU pass each (~75 min / ~2.5 hr).** C3 only if C0/C1 point at vocab. Do NOT run a vocab sweep before C0 — the priors did exactly that and got refuted twice.

---

## 4. ANTI-CHEATS + GO bars + VERDICT

### Anti-cheats (mandatory — this is where the prior over-confidence hid)

1. **The gen reference MUST stay INDEPENDENT / a-priori.** Score against `TAXONOMY_40x8` / `TAXONOMY_8x8` / coherent-g20 category blocks — NEVER corpus-derived `S_true` (the load-bearing correctness property, `option_c` design SS1). C0's numpy-320 must use the SAME a-priori cat_ids the bridge-320 uses (so the only changed axis is substrate, not the labels).
2. **Report gen on BOTH references AND `ratio_vs_chance` AND Pearson(cos,S_true).** The chance baseline differs across 8-cat (0.125) vs 30-cat (0.033); the raw 0.91-vs-0.15 comparison is partly a chance artifact (DIFF 2). The load-bearing tell is **Pearson(cos,S_true)** (chance-independent): validated +0.513 vs Step-1 **+0.07/+0.08**. A real fix must lift Pearson toward +0.40, not just the chance-sensitive accuracy.
3. **PROVENANCE — report ALL prior numbers alongside (the 2x-refutation must be visible):** all-vocab 0.153 (sharding) / 0.167 (coherent), content 0.125, AND the new numbers. Foreground that BOTH prior hypotheses (yardstick, content-vocab) were REFUTED. Do NOT hide the content-0.125 (it is the scientifically-correct refutation of the vocab hypothesis).
4. **No loosening of any other bar.** recall ≥ 0.95 and **moat 0 false-accepts** MUST hold on every arm (they are reference-independent — assert they stay ≥ the Step-1 1.000 / 0-FA). A gen fix that breaks the moat is a HARD STOP.
5. **Frozen-brain control** (`plasticity_on=False`) MUST stay competence-flat (corr(M,C)~0, recall<0.5, gen≤learned) on every arm — already PASS in Step-1; re-assert (it proves the lift is LEARNED, not the metric/curriculum inflating chance).
6. **Derangement control** (shuffle category labels → gen collapses to ~chance) on every arm — already PASS (0.009); re-assert (it proves the gen number reflects real structure, not a denser/sparser reference inflating the baseline). Report gen AND ratio_vs_chance since chance shifts with #categories.
7. **C0's numpy anchor must REPRODUCE 0.91 verbatim before any variant is trusted** — if the validated runner does not reprint ~0.91 today, the baseline itself is stale and the whole comparison is moot (verify the anchor first).

### GO bars

- **Primary (the fix is real, IF a fix exists):** generalization ≥ 0.80 on the independent a-priori coherent reference, 3 seeds, **AND Pearson(cos,S_true) ≥ +0.40** (the chance-independent tell, vs the current +0.07), with derangement collapse + frozen-flat + recall ≥ 0.95 + moat 0-FA all holding.
- **Diagnostic bars (these SETTLE the cause regardless of whether 0.80 is reached):**
  - C0: numpy-exact-count gen at 320-cat/300-hub — splits substrate vs scale.
  - C1: bridge-64 gen at n_hub=500/150K windows — quantifies the substrate cost on the validated vocab.
  - C2/C3 monotonicity in (n_per, windows, n_hub) — tells whether the miss is fidelity-recoverable or a boundary.
- **Honest-boundary acceptance bar (the SURPASS gate):** a boundary is accepted ONLY if C0 stays high (numpy fine) AND C1/C2/C3 show on-bridge 320-gen does NOT rise with fidelity (n_per↑, windows↑, n_hub↑ all flat) — i.e. the spiking read-out genuinely cannot carry fine 320-category similarity. THEN the residual is precisely quantified (the gap between numpy-320 gen and bridge-320 gen at max fidelity) and the next lever is named (richer/fact-denser corpus #4, OR a dendritic read-out for the off-diagonal — the documented deep frontier).

### VERDICT

**The boundary is REAL (gen 0.15 reproduces on both references and survives the content-filter), but the prior diagnoses are WRONG: both scopings claimed "byte-identical pipeline, only vocab differs" and both predicted ≥0.80 — and the substrate axis they missed (numpy-exact-count vs spiking-noisy-population, which alone drops the validated vocab 0.91→0.45 on the bridge) plus the never-verified "curated-320 held its gen" assumption mean the dominant cause is most likely the SPIKING READ-OUT FIDELITY AT 320-CONCEPT SCALE, not the vocab.** The single decisive control is **C0 (instant host re-score): reproduce the validated numpy 0.91, then run the SAME numpy exact-count pipeline at 320-cat/300-hub** — this splits substrate-vs-scale with ZERO GPU and ZERO new mechanism. If numpy-320 stays high, the spiking read-out is the cause (lever: on-bridge fidelity via n_per/windows/n_hub — C2 confirms); if numpy-320 collapses too, it's the scale/metric/curriculum granularity (the 30-category chance artifact + thin hub), addressable by curriculum granularity, NOT a substrate wall.

**This is NOT a fire-the-gate-to-build-a-new-mechanism** (the Hebbian-co-occurrence + log-double-centre mechanism is validated; the question is whether its SPIKING read-out preserves fine 320-category similarity — an empirical isolation, not a new build). **It is gated behind the standing anti-cheats** (independent a-priori reference, moat-0-FA-preserved, frozen-flat, derangement-collapse, Pearson ≥ +0.40 as the chance-independent tell). **The recall-1.000 + moat-0-FA WIN stands regardless** — only the FINE category-similarity structure (generalization) is in question, and the controls SETTLE whether it is recoverable on the spiking substrate or a precisely-quantified boundary.

**DO NOT predict a number** (both priors did and were refuted). Run C0 first; let it decide.

---

## Sources / artifacts (read-only, verified this session)

- `research/runners/_phaseB_online_stream_cortex_derisk.py` — the VALIDATED **0.91 / +0.513** run: **pure NUMPY** (`:81` `M=np.zeros`, `:99` `M[…]+=1.0` exact count), `TAXONOMY_8x8` (**64 concepts, 8 categories, chance 0.125**, `:122`), `N_HUB=500` (`:47`), `n_per=1` (host count), `double_center(log1p(M·100))` (`:103`), `heldout_generalization(code, cat_ids)` (`:105`). **This is NOT a bridge run** — the load-bearing diff the priors missed.
- `research/runners/_curriculum_step1_320_real_corpus.py` — the 320-run: **SPIKING bridge** (`build_stream_bridge`, `:80,:396`), `M` = population block-mean of LEARNED weights read off `cp_connections` (`:460-463`, corr(M,C)=0.756 NOT 1.0), n_hub 300 / n_per 16 / 320 concepts / chance 0.033, byte-identical `double_center(log1p(M·100))` (`:464`) + `heldout_generalization` import (`:83`). `--vocab-filter {content,all,curated}` (`:238,:803`) — the content fix that gave 0.125.
- `research/runners/_phaseB_onbridge_stream_cortex_derisk.py` — the ON-BRIDGE 64-curated capstone: `build_stream_bridge` (`:62-91`), `gen` printed (`:166,:174`); **reports gen 0.45 at 30K windows on the validated curated vocab** (finding line 138) — the cite-then-ignored proof that numpy→bridge alone costs ~0.46 of gen. This IS the C1 anchor runner (exposes `--n-hub`, `--max-windows`, `--n-per`).
- `research/runners/_phaseB_stdp_cooccurrence_derisk.py` — the CYCLE-95/96 mechanism+population-lift runner: uses `TAXONOMY_8x8` (**64**, `:200` via `taxonomy_to_vocab_categories`), reports `corr(M,C)` + Pearson "normalized code" (`:161-172`), gen printed but the GO gate is `corr(M,C)≥0.60` (`:209`), NOT gen. **Its lift table (corr(M,C)→0.93 at n_per=32) is NOT a generalization table** — the prior scoping read population-lift-of-corr as if it were gen.
- `research/runners/corpus_stream.py` — `load_token_stream`/`iter_stories`: tokenizer `re.findall(r"[a-z]+")` (`:34,:48-50`), byte-identical to the validated run's `:60` — corpus + tokenizer are NOT a diff (VERIFIED).
- `research/runners/dendritic_d1_learn_graded_structure_derisk.py` — `heldout_generalization` (`:187-202`, returns `correct/Nc, 1/len(cats)` — **chance = 1/#categories**, the 0.125-vs-0.033 metric artifact source), `_cos_sim` (`:128`), `_pearson_vs_Strue` (`:134`) — identical import in all runners.
- `research/runners/option_c_real_cooccurrence_derisk.py` — `TAXONOMY_8x8` = 8×8 = **64 concepts** (`:108-117`); the INDEPENDENT-`S_true` correctness property (design SS1); the `BOUNDARY_weak_graded` 64-curated real-corpus result (gen 0.308, host ceiling Pearson +0.126) — a LOSSIER pipeline data point showing TinyStories-64 co-occurrence gen is non-trivial even curated.
- `research/runners/stream_taxonomy_320.py` — `TAXONOMY_40x8` (40×8 = **320** balanced coherent content words, freq≥50) — the `--vocab-filter curated` vocab; docstring `:16-23` excludes abstract/function words.
- Result JSON `research/findings/raw/_curriculum_step1_320_real_corpus_seed42.json` — gen 0.153, Pearson +0.070, chance 0.0333, ratio 4.59×, corr(M,C) 0.756, recall 1.0 (48/48), moat 0-FA, derangement 0.009 (collapses), frozen-flat PASS. **Config has NO `vocab_filter` field → this is the `all` provenance run** (top25 = hey/day/big/very/happy…); the content-0.125 + coherent-0.167 are prompt-reported re-measures (not separate JSONs this session).
- Findings: `2026-06-15-biology-faithful-online-stream-cortex-reaches-target.md` (**gen 0.91 / Pearson +0.513 — pure NUMPY, 64 curated, line 28**); `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (**on-bridge gen 0.45 at 30K windows / 64 curated, line 138**; CYCLE-96 curated-320 GO lines 203-220 = **recall+moat+gap only, NO gen number printed**; lift table lines 104-114 = corr(M,C)+normalized-code, NOT gen).
- Prior scopings (BOTH REFUTED, corrected here): `_curriculum_gen_miss_scoping.md` (yardstick-swap → predicted ≥0.80; moved 0.153→0.167 — REFUTED); `_curriculum_gen_miss_REAL_scoping.md` (content-vocab → predicted ≥0.80 "the validated recipe corpus-frequency-ranked"; content-filter gave 0.125 — REFUTED; AND its "byte-identical pipeline, only vocab differs" + "curated-320 held / scale RULED OUT" claims are source-refuted here: the substrate differs (numpy vs bridge) and no on-bridge 320-gen baseline exists).
- Catalog: `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` (G.20 Pulvermüller distributed cortical word ensembles; A.11/A.12 convergence/decorrelation — the off-diagonal similarity a low-SNR spiking read-out blurs first).
