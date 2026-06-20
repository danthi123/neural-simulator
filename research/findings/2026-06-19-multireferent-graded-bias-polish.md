# Multi-referent disambiguation — content-graded bias polish: closes the seed-100 boundary (GO 6/6, 2026-06-19)

**Verdict: GO (6/6 on the strict GO-arm; all anti-cheat controls 6/6).** A **content-graded bias** closes the one
pre-registered miss of the validated biased-competition de-risk (seed 100, the extreme-intrinsic-asymmetry case)
**without breaking anything that already worked**: the bias-lesion still breaks resolution 6/6 (the graded bias
stays load-bearing — it is *not* a relabelled global gain), the no-confab moat still holds 6/6 (0 breaches), and the
two prior NEGATIVE baselines (recency, salience-4×) still fail 6/6 on the identical setup.

- Runner: `research/runners/_phaseB_biased_competition_graded_derisk.py`
- Raw: `research/findings/raw/_phaseB_biased_competition_graded.json`
- Closes the miss in: `research/findings/2026-06-19-multireferent-biased-competition-derisk.md` (§4, GO 5/6)
- Reuses-by-import the production `BiasedCompetitionContextBuffer` + `content_bias_target` + `resolve_referent`
  (`research/runners/biased_competition_buffer.py`) and the validated baseline battery (the de-risk runner) — **no
  `sim/` edit**. CPU/numpy (`SIM_BACKEND=numpy`).

---

## 1. The gap this closes

The biased-competition de-risk was GO 5/6, with **seed 100** the single pre-registered miss. There the
content-favored referent's intrinsic accumulator feed-forward is **essentially zero** relative to a strong rival:
on the feature-flip (`roll` favors **ball**) the unbiased sel read is **ball 0.000 vs cat 0.292** — the most extreme
intrinsic asymmetry across the 6 seeds. A **fixed-magnitude** content bias (≈1×, 2500 pA) could not lift ball past
cat, so the WTA either kept the wrong (cat) winner (`roll/cat-first`) or neither side reached the 1.3× margin
(`roll/ball-first` → **None**). The abstention is **moat-preserving** (it refused to bind the wrong referent, it did
*not* confabulate) — but the named fix is a **content-graded bias** that scales the steer to the difficulty of the
case.

## 2. The mechanism (additive; reuse-by-import; no `sim/` edit)

The bias is graded by **how badly the content-favored referent is intrinsically dominated by its rival**:

1. The host content helper (`content_bias_target`, unchanged) selects **which** held referent the pronoun+verb favors
   (animacy/number agreement + the verb's selectional restriction). Content-silent/ambiguous → abstain (moat).
2. A **cheap unbiased probe read** of the per-referent accumulator (sel) competition measures the favored referent's
   intrinsic sel `fav_sel` and its strongest rival's sel `rival_sel`. The probe is **non-destructive** — the held
   attractors persist across reads (verified directly: cat 0.2925 → 0.31 on a re-read; the read re-presents and
   reads, it does not rewrite weights).
3. The **content-graded magnitude** is set by the favored referent's competitive **deficit** and injected into **only
   the content-favored** sel pool, exactly as the fixed bias was:

   ```
   deficit = max(0, rival_sel - fav_sel)              # how much the favored referent is intrinsically dominated
   bias_pA = min(cap, base * (1 + gain * deficit/ref))    # base=2500, gain=1.0, ref=0.20, cap=8000
   ```
4. Resolve the WTA winner, gated by the no-confab moat (the winner must be a referent actually held in WM; ties /
   content-silent / empty WM → abstain).

**Why this is content-graded, not a global "turn it up" (the decisive distinction):** the magnitude is (a) applied
to **only** the content-favored referent (a global gain would lift every sel pool, including the rival), and (b)
scaled by **that referent's** content-vs-rival competitive deficit — so a deficit of 0 (an already-competitive
favored referent) leaves the bias at base, the easy cases unchanged (no over-steer). And critically, the
**bias-lesion removes the bias entirely** (`bias_pA=0`, no probe), so `graded(lesioned)=0` → the WTA reverts to the
intrinsic winner → the lesion **still breaks** resolution. A graded bias that survived the lesion would be a global
gain — the runner checks this and would not call it GO.

## 3. Result — 6 seeds (42, 43, 44, 100, 101, 102), base = 2500 pA, gain = 1.0, ref = 0.20, cap = 8000 pA

| Gate | Fixed-bias de-risk | Content-graded (this) |
|---|---|---|
| **GO-arm** — favored wins both write-orders **and** the feature-flip flips the winner | 5/6 | **6/6** ✅ |
| **every roll case resolves** (the seed-100 close) | 4/6 | **6/6** ✅ |
| **bias-LESION breaks** — unbiased WTA reverts to intrinsic winner ⇒ bias load-bearing | 6/6 | **6/6** ✅ |
| **no-confab MOAT intact** — empty WM → abstain; content-silent → abstain; 0 breaches | 6/6 | **6/6** ✅ |
| **recency baseline FAILS** on the identical {cat, ball} setup | 6/6 | **6/6** ✅ |
| **salience-4× baseline FAILS** on the identical setup | 6/6 | **6/6** ✅ |
| **3-referent scale** (one compatible + two incompatible) | 6/6 | **6/6** ✅ |

**Win table (per write-order × feature-flip), all 6 seeds** — `eat` selects animate (favored = cat), `roll` selects
inanimate (favored = ball). The previously-failing seed-100 `roll` cells are now ✅:

| Seed | eat / cat-1st | eat / ball-1st | roll / cat-1st | roll / ball-1st | go_arm |
|---|---|---|---|---|---|
| 42  | cat ✅ | cat ✅ | ball ✅ | ball ✅ | GO |
| 43  | cat ✅ | cat ✅ | ball ✅ | ball ✅ | GO |
| 44  | cat ✅ | cat ✅ | ball ✅ | ball ✅ | GO |
| 100 | cat ✅ | cat ✅ | **ball ✅** | **ball ✅** | **GO (was miss)** |
| 101 | cat ✅ | cat ✅ | ball ✅ | ball ✅ | GO |
| 102 | cat ✅ | cat ✅ | ball ✅ | ball ✅ | GO |

**The seed-100 close, in spiking detail** (`roll` favors ball; the bias graded up to ~5.5–6.2k pA from the deficit):
- `roll/cat-first`: deficit 0.292 → bias **6156 pA** → sel **ball 0.4875** vs cat 0.155 (held both ~0.329) → resolves
  **ball** (3.1× margin). Was: cat (wrong) at fixed 2500.
- `roll/ball-first`: deficit 0.237 → bias **5469 pA** → sel **ball 0.4675** vs cat 0.1175 → resolves **ball**. Was:
  None (abstain) at fixed 2500.
- **lesion** (`roll`, bias removed → graded(0) = **0 pA**): sel **cat 0.2925** vs ball 0.000 → reverts to **cat by
  intrinsic** ⇒ wrong for the ball-favoring verb ⇒ the graded bias is the load-bearing signal, not a global gain.

**The grading is targeted, not a blanket increase** (per-seed `roll`-case bias magnitudes):

| Seed | roll/cat-1st | roll/ball-1st | note |
|---|---|---|---|
| 42  | 2500 | 2500 | favored ball already competitive → stays at base |
| 43  | 2906 | 2500 | small deficit → small bump |
| 44  | 3250 | 2500 | small deficit → small bump |
| 100 | **6156** | **5469** | extreme deficit → strong steer (the close) |
| 101 | 2500 | 2500 | already competitive → base |
| 102 | 5688 | 2500 | one extreme, one competitive |

Seed-100's **`eat`** cases (favored cat, already intrinsically dominant, deficit 0) stay at base **2500 pA** — the
grading does not over-steer the cases that never needed help.

## 4. Why it doesn't regress the moat

The graded bias raises **magnitude only for the content-favored referent**, and resolution is still gated by the same
`resolve_referent` moat (the winner must clear the 1.3× margin **and** be held in WM above the floor; ties /
content-silent / empty WM → None). The two moat probes confirm this is intact at every seed:
- **empty WM** → sel reads exist but held = 0 → held-floor gate **abstains** (the bias alone cannot confabulate an
  antecedent), 6/6.
- **content-silent** (`see`, no selectional restriction) → favored None → **abstains** (refuses to pick by intrinsic
  strength), 6/6.

A graded bias that pushed a genuinely-**tied** case to resolve would be a moat regression — it does not: the
all-incompatible / content-silent / empty cases still abstain.

## 5. Honest scope

- This closes the **seed-100 extreme-intrinsic-asymmetry** boundary the GO de-risk pre-registered. The honest
  follow-on remains the **all-compatible** case (two same-animacy/number candidates where agreement is silent and only
  finer role/recency cues decide) — there the bias must come from finer cues composed *on top of* the validated
  competition; the content helper currently abstains on that tie.
- The **content scoring is still a host scaffold** (`content_bias_target`), FLAGGED for conversion to a learned
  synaptic feature-compatibility map per BRAIN-BASED-ONLY. The graded layer adds a **host probe-read + deficit
  scaling**; the brain-based pieces (the spiking accumulator competition + the selective FS suppression + the
  recurrence amplifying the steered evidence) are unchanged. The BRAIN-BASED conversion target is unchanged: both
  *which* referent to bias and *how strongly* would be computed by neurons (a feature-compatibility map + a
  competition-deficit signal the substrate could plausibly read from the accumulator pools themselves).
- The probe-read costs one extra unbiased read window per query (cheap; it reuses the same `read()` path).

## 6. Recommendation

**GO** → update the production `MultiTurnAgent` biased-competition to the **content-graded bias** (a small follow-on
to the already-recommended `enable_biased_competition` integration): on a pronoun query, run the unbiased probe read,
grade the bias by the content-favored referent's competitive deficit, then resolve the WTA under the unchanged moat.
This lifts the strict GO-arm from 5/6 to 6/6 while keeping the lesion-load-bearing and moat-intact guarantees that
made the original a GO.

---

### Files
- `research/runners/_phaseB_biased_competition_graded_derisk.py` — the graded-bias polish (the probe-read + deficit
  scaling + the full 6-seed GO/lesion/moat/baseline harness, reusing the production buffer + helpers verbatim).
- `research/findings/raw/_phaseB_biased_competition_graded.json` — 6-seed raw.

### Cited
- Desimone & Duncan 1995 (biased competition); Wong & Wang 2006 (attractor WTA amplifying a biased input);
  Rutishauser-Douglas-Slotine 2011 (the α<1 WTA-stability condition the codebase enforces). The deficit-graded steer
  is the homeostatic/normalized-bias variant the GO de-risk (§4, §7) pre-registered for the extreme-intrinsic-asymmetry
  boundary. Catalog: N.19 (FS mutual inhibition), B-cluster (MSN lateral-inhibition WTA), H.24/H.25 (the `sel`/`commit`
  recipe reused).
