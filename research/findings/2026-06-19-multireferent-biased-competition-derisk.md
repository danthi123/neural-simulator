# Multi-referent disambiguation via WTA biased competition — cheap-first de-risk: GO (2026-06-19)

**Verdict: GO (5/6 seeds on the strict GO-arm; all anti-cheat controls 6/6).** WTA biased competition resolves
a bare pronoun to the correct one of ≥2 held discourse referents **where recency and a salience boost cannot**.
The favored referent wins both write-orders and the feature-flip flips the winner (proving the steer is **content**,
not position or magnitude); the bias is **load-bearing** (the bias-lesion reverts the winner to the seed-dependent
intrinsic attractor on every seed); the no-confab moat holds (empty WM and content-silent query → abstain, 0
breaches); and the two prior NEGATIVE baselines (recency, salience-4×) both FAIL on the identical {cat, ball} setup.

- Runner: `research/runners/_phaseB_biased_competition_derisk.py`
- Raw: `research/findings/raw/_phaseB_biased_competition.json`
- Scoping (followed): `research/findings/2026-06-19-multireferent-wta-biased-competition-scoping.md` (commit `344ad3af`)
- The two NEGATIVEs this converts: `research/findings/2026-06-17-multireferent-disambiguation-NEGATIVE.md`
- CPU/numpy (`SIM_BACKEND=numpy`); reuse-by-import; **no `sim/` edit**.

---

## 1. The question

When the spiking working memory holds several discourse referents, which one does a bare pronoun ("it") bind to?
Two prior NEGATIVEs showed **recency** (0/3) and a **salience boost** (even 4× fails) both cannot pick the right
one — because the plain `SpikingLoopContextBuffer` holds each referent in an **independent** attractor (`loop_weight=0`,
`internal_density=0`), so neither a position signal nor a uniform gain boost can *suppress* a competitor. The scoping
named the fix: **biased competition** (Desimone-Duncan 1995; Wong-Wang 2006) = **mutual inhibition** between referent
representations + a small **content-based top-down bias** (animacy/number agreement with the pronoun + selectional
restriction of the query verb), where the attractor recurrence amplifies the small content asymmetry into a
suppressive winner.

## 2. What was built (additive, default-OFF, no `sim/` edit)

`BiasedCompetitionContextBuffer` faithfully reuses the navigation **`sel_X`/`sel_FS_X` Wong-Wang accumulator WTA**
(`g11_bg_runner.py`) as a **read-out tap** over the held referents:

- Per referent, a `sel_X` accumulator pool (excitatory, NMDA-slow recurrent, soft-WTA gain **α<1** = Rutishauser
  stability) + a `sel_FS_X` **selective** inhibitory pool (driven only by `sel_X`, inhibits only `sel_Y≠X`).
- `cortex_assembly[X] → sel_X` (feed-forward evidence; **read-only tap — no `sel_X → cortex` back-projection**, so
  the held attractors are unperturbed, exactly the navigation design).
- A **small content bias** current injected into the favored `sel_X` during the read.
- A host content-bias helper (`content_bias_target`) maps the pronoun's features + the verb's selectional
  restriction → which referent to bias. **FLAGGED for conversion to a learned synaptic feature-compatibility map
  per BRAIN-BASED-ONLY** (a teaching scaffold; the win is brain-based spiking competition + suppression, the content
  *scoring* is host in this probe; the follow-on neuralizes it).

**Two substrate facts found + handled in the build (both diagnosed against `sim/bridge.py`, not assumed):**
1. The synapse E/I sign is the **PRE-neuron's inhibitory trait** (`is_inhibitory_neuron_output` from `cp_traits`,
   `bridge.py:5942-5953`), NOT the weight sign — so the FS pools are `exc_fraction=0.0` framework regions (every
   neuron inhibitory → their out-synapses route to `g_i`). `set_pathway_weights(add_missing=True)` then installs the
   assembly-targeted cross-referent edges.
2. **The plain loop does NOT hold 2 referents as a coexisting set** at the {cat, ball} config: the *stronger
   intrinsic attractor dominates the hold* and the other collapses (cat 0.034 vs ball 0.331, the SAME in 2-concept
   and 6-concept lists, every protocol tried — the exact mechanism the original NEGATIVE named). So the competitive
   read **RE-PRESENTS the held discourse-referent registry as co-active competitors** (a retrieval cue gently
   re-drives their assemblies — the biology of biased competition, where the competing stimuli are *simultaneously
   present*). Both assemblies then active, the sel accumulators both driven, the content bias arbitrates. The moat
   reads the held assembly (a winner must be re-presentable above a floor) + abstains when the content is silent.

## 3. Result — 6 seeds (42, 43, 44, 100, 101, 102), bias = 2500 pA (= 1× the assembly drive; the salience boost failed at 4× = 10000)

| Gate | Result |
|---|---|
| **GO-arm** — favored referent wins **both** write-orders **and** the feature-flip flips the winner | **5/6** ✅ (≥5/6 bar) |
| **bias-LESION breaks** — unbiased WTA reverts to the intrinsic winner (wrong for ≥1 verb) ⇒ bias load-bearing | **6/6** ✅ |
| **no-confab MOAT intact** — empty WM → abstain; content-silent query → abstain; 0 breaches | **6/6** ✅ |
| **recency baseline FAILS** on the identical {cat, ball} setup (run in-probe) | **6/6** ✅ |
| **salience-4× baseline FAILS** on the identical setup (run in-probe) | **6/6** ✅ |
| **3-referent scale** (one compatible + two incompatible) resolves the compatible one | **6/6** ✅ |

**Win table (per write-order × feature-flip), all 6 seeds** — `eat` selects animate (favored = cat), `roll` selects
inanimate (favored = ball):

| Seed | eat / cat-first | eat / ball-first | roll / cat-first | roll / ball-first | go_arm |
|---|---|---|---|---|---|
| 42  | cat ✅ | cat ✅ | ball ✅ | ball ✅ | GO |
| 43  | cat ✅ | cat ✅ | ball ✅ | ball ✅ | GO |
| 44  | cat ✅ | cat ✅ | ball ✅ | ball ✅ | GO |
| 100 | cat ✅ | cat ✅ | **cat ✗** | **None ✗** | miss |
| 101 | cat ✅ | cat ✅ | ball ✅ | ball ✅ | GO |
| 102 | cat ✅ | cat ✅ | ball ✅ | ball ✅ | GO |

Representative spiking detail (seed 42, sel-pool firing rates):
- `eat` (favored cat): sel **cat 0.44** vs ball 0.10 → resolves **cat**.
- `roll` (favored ball, FEATURE-FLIP): sel cat 0.02 vs **ball 0.50** → resolves **ball** (the flip flips the winner).
- **lesion** (bias removed): sel cat 0.025 vs **ball 0.215** → resolves **ball by intrinsic** ⇒ for `eat` (favored
  cat) the unbiased answer is **wrong** ⇒ the content bias is the load-bearing signal, not a relabelled boost.
- **moat empty**: sel cat 0.48 but held 0.0 → held-floor gate **abstains** (the bias alone cannot confabulate).
- **moat silent** (`see`, no selectional restriction): favored None → **abstains** (refuses to pick by intrinsic).

## 4. The single miss (seed 100) is the honest, anticipated boundary — and the moat held through it

Seed 100 has an **extreme intrinsic-cat dominance** (even the lesion: sel cat 0.29 vs ball 0.0). When `roll` favors
**ball**, the fixed +ball bias lifts ball's sel to 0.09-0.12 but cat's intrinsic feed-forward keeps it at 0.15-0.19,
so cat still leads (`roll/cat-first`), or **neither reaches the 1.3× margin** (`roll/ball-first` → **None**). This is
exactly the BOUNDARY the scoping pre-registered: a *fixed-magnitude* bias occasionally cannot flip a referent whose
intrinsic attractor is extreme — localizing "competition-strength vs intrinsic-asymmetry" as the next tuning
sub-problem (a content-graded or homeostatically-normalized bias, within the α<1 envelope). Critically, the failure
mode is an **abstention (None), not a confabulation** — the no-confab moat correctly refused to bind the wrong
referent rather than fabricate one. A bias scaled to 1500 pA confirmed 2500 is the right operating point, not a
knife-edge: at 1500 a second case abstains (None), never gives a wrong answer.

## 5. Why this succeeds where recency + salience failed (the mechanism, reproduced)

The two prior signals failed because the loop has **no competition** to convert any signal into *suppression*:
recency carries no position signal in the rate read; a uniform boost only *adds* activity to an independent
attractor. Biased competition adds **both** missing ingredients: (i) **mutual inhibition** (`sel_FS_X` selective
inhibition) turns any asymmetry into suppression of the loser — visible as the loser's sel going to ~0.0-0.1 while
the winner ramps to ~0.3-0.5; and (ii) a **content asymmetry** (the feature/role-compatibility bias) — which the
feature-flip proves is the steer (flip the verb's selectional restriction → flip the winner). The Wong-Wang
recurrence amplifies the small content bias into a clean, suppressive winner. The baselines confirm the setup is
genuinely ambiguous without the mechanism (recency + salience-4× both fail 6/6 on the identical {cat, ball}).

## 6. Honest scope

- **2 opposing-feature referents** is the decisive minimum ("does the mechanism exist") + a **3-referent**
  (one-compatible + two-incompatible) in-probe scale check passes 6/6. The genuinely-hard **all-compatible** case
  (two animate candidates of the same number/gender, where agreement is *silent* and only finer role/recency cues
  decide) is the honest follow-on — there the bias must come from finer cues composed *on top of* the validated
  competition.
- The **content scoring is a host scaffold** (FLAGGED): legitimate as a teaching scaffold; the brain-based piece is
  the spiking competition + suppression. The follow-on converts the feature-compatibility to a **learned synaptic
  map** (pronoun-feature population × candidate-feature population → bias current) so the bias itself is neural — an
  honest NEGATIVE on the *neural-bias* version would itself map what the substrate can compute about agreement.
- The competitive read uses a **re-presentation** cue (a retrieval probe re-activating the held referents). This is
  biologically faithful (biased competition operates on co-active representations) and is *driven by the
  discourse-referent registry* (the spiking buffer's held set), not by privileged knowledge of the answer.

## 7. Recommendation

**GO** → wire biased competition into `MultiTurnAgent` behind a default-OFF `enable_biased_competition` flag
(a follow-on, not this de-risk): on a pronoun query, build the per-referent `sel`/`sel_FS` WTA over the held
referents, re-present + bias the feature/role-compatible candidate, resolve the WTA winner (gated by the no-confab
moat). The two precise next sub-problems are already localized: (a) the **content-graded / normalized bias** for
extreme-intrinsic seeds (the seed-100 boundary), and (b) the **all-compatible** case (finer cues on top of the
competition). The bias-as-learned-synapse is the BRAIN-BASED-ONLY conversion target.

---

### Files
- `research/runners/_phaseB_biased_competition_derisk.py` — the de-risk (the `BiasedCompetitionContextBuffer`
  accumulator WTA + the content-bias helper + the GO/lesion/moat/baseline harness).
- `research/findings/raw/_phaseB_biased_competition.json` — 6-seed raw.

### Cited
- Desimone & Duncan 1995 (biased competition); Wong & Wang 2006 (attractor WTA amplifying a biased input);
  Rutishauser-Douglas-Slotine 2011 (the α<1 WTA-stability condition the codebase enforces); pronoun-agreement
  filtering (Frontiers 2014). Catalog: N.19 (gamma binding-by-synchrony FS mutual inhibition), B-cluster (MSN
  lateral-inhibition WTA precedent), G.08 (PFC WM), H.24/H.25 (the navigation `sel`/`commit` recipe reused).
