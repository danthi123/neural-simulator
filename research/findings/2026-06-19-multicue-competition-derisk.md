# Multi-cue competition parser — degraded-input robustness DE-RISK = GO (6/6 seeds)

**Date:** 2026-06-19
**Type:** Phase-1 cheapest-first MECHANISM de-risk (CPU/numpy), per the scoping
`2026-06-19-multicue-competition-parser-scoping.md` (commit `0ecdb628`).
**Runner:** `research/runners/_phaseB_multicue_competition_derisk.py`
**Raw:** `research/findings/raw/_phaseB_multicue_competition.json`
**Verdict:** **GO — 6/6 seeds.** Adding competing cues (animacy + verb-selectional-fit) to the word-position cue,
integrated by a reliability-weighted competition with **LEARNED cue-validity weights**, makes thematic-role
assignment **robust to degraded English (scrambled / object-fronted order) where a position-ONLY parser
collapses.** All decisive anti-cheat controls pass; the no-confab moat holds (0/240 breaches). Recommend the
spiking realization as the production build (see §6).

---

## 1. What it decides

Does the Bates-MacWhinney **Competition Model** — multiple thematic-role cues competing with **learned cue
validities**, realized as biased competition over role assemblies + a reliability-weighted accumulator (catalog
G.18 LIP "integrates ANY evidence weighted by reliability, additively"; G.12 Broca "semantic cues carry
comprehension when order is hard") — make role assignment robust to imperfect input **where the current
position-only parser** (`BridgeParser`, `(position × voice) → role`) **fails**? And — the load-bearing
question — are the cue WEIGHTS genuinely **LEARNED** (cue validities), not hand-set, and are the added cues
genuinely **LOAD-BEARING**?

This is the CPU/numpy MECHANISM de-risk. It is the FUNCTIONAL stand-in for the spiking biased-competition WTA;
the production build is on the substrate (§6). **NO `sim/` edit; reuse-by-import** (the `ANIMACY` lexicon is
imported from `research/runners/biased_competition_buffer.py`, with a drift assertion).

Task: assign **agent / patient** to the two nouns of a transitive sentence (**chance = 0.500**). Cues:
**position** (1st→agent, last→patient, structural), **animacy** (animate→agent), **verb-selectional-fit** (does
the noun fit the verb's agent vs patient slot), plus a chance-validity **distractor** cue (`lexbias`, validity
0.5, sign-correlated with position) the learner must zero out.

---

## 2. Result (reproducible, 6 seeds, held-out fillers + verbs)

Metric = the **position-DEGRADING** battery (scramble + object-front), where the position cue is genuinely
unreliable. `drop-verb` is reported separately because it removes the verb (hence verb-fit) but does **not**
degrade position, so it is excluded from the position-only-collapse gate.

```
 seed | MULTICUE | POS-ONLY | NO-LEARN |  LESION | PERMUTE | moat_br | GO
   42 |    0.950 |    0.263 |    0.625 |   0.263 |   0.263 |       0 | GO
   43 |    0.950 |    0.275 |    0.637 |   0.275 |   0.275 |       0 | GO
   44 |    0.950 |    0.237 |    0.600 |   0.237 |   0.237 |       0 | GO
   45 |    0.950 |    0.225 |    0.575 |   0.275 |   0.225 |       0 | GO
   46 |    0.950 |    0.287 |    0.625 |   0.287 |   0.287 |       0 | GO
   47 |    0.950 |    0.263 |    0.600 |   0.263 |   0.263 |       0 | GO
 mean |    0.950 |    0.258 |    0.610 |   0.267 |   0.258 |       0 |
```
(`MULTICUE`/`POS-ONLY`/etc. are the position-degrading-battery role accuracies; `moat_br` = decisive commits on
the genuinely-ambiguous set, must be 0. The JSON artifact + this table are byte-reproducible run-to-run — see §4
reproducibility.)

**Per-degradation (mean across seeds): MULTICUE vs POSITION-ONLY**

| condition | MULTICUE | POSITION-ONLY | note |
|---|---|---|---|
| scramble | **1.000** | 0.517 | same words, randomized order → position misleading |
| object-front (OSV) | **0.900** | **0.000** | fronted patient → position table maps it agent (NEMO weakness) |
| drop-verb | 1.000 | 1.000 | verb gone (no verb-fit); position NOT degraded → both fine |

**clean canonical (no-regression):** multi-cue **1.000** vs position-only **1.000** (multi-cue does not hurt the
native canonical case).

**Learned cue weights (mean across seeds):** `position ≈ 0.34, animacy ≈ 0.76, verbfit ≈ 0.72, lexbias ≈ 0.03`.
Frozen (no-learning) weights stay uniform `0.50 / 0.50 / 0.50 / 0.50`.

---

## 3. Why this is a real result, not hand-tuned cues (the decisive controls)

The PRIMARY mislead the scoping flagged — *hand-tuned cues masquerading as a learned model* — is directly tested
and excluded. The runner was deliberately designed so that **learning is necessary**: cues are individually
**non-decisive** (per-cue label noise at rate 1−validity), training is **naturalistic** (canonical-majority +
~40% non-canonical, with gold, so position's *empirical* validity is high-but-imperfect), and a **chance-validity
distractor** is present. With these, ANY non-negative weights do **not** solve the task — only the *learned*
weighting does.

| Control | Result | Gate | What it proves |
|---|---|---|---|
| **POSITION-ONLY baseline** (drop animacy+verbfit) | **0.258** (object-front 0.000) | ≤0.45 collapse | THE LOAD-BEARING control: the battery genuinely degrades position; the win is the *added cues carrying degraded input*, not a generically better parser. |
| **NO-LEARNING** (cue weights FROZEN at uniform init) | **0.610** | ≤ MULTICUE−0.15 | The validities are **LEARNED, not hand-set**: a uniform parser over-trusts position+distractor and is misled (−34pp vs the learned 0.95). |
| **CUE-LESION** (zero animacy+verbfit, keep position) | **0.258** | ≈ position-only | The semantic cues are **load-bearing**: removing them collapses robustness back to position-only. |
| **PERMUTED-CUE** (train on scrambled animacy/verb-fit tags) | **0.258** | ≤0.60 | The cues carry **real** role information, not a relabelled position signal / leak. |
| **HELD-OUT FILLERS + VERBS** (test pools disjoint from training) | GO (held-out 0.950; train==test 1.000) | — | **Not memorizing** examples; role correctness is vocab-agnostic. |
| **no-confab MOAT** (two animate nouns + symmetric verb, scrambled → no decisive cue) | **0/240 breaches**, abstain 1.00 | 0 | The moat is **not weakened**: when the cues genuinely tie, the parser ABSTAINS, it does not confabulate a role. |

**The mechanistic signature** is the clincher: the learner ends every seed with **`w_position` driven BELOW the
semantic weights** and **the distractor `w_lexbias` driven to ~0** (e.g. seed 45: position 0.05, animacy 0.69,
verbfit 0.87, lexbias 0.11). That is cue-validity learning — down-weight the cue that is unreliable on the input
distribution, up-weight the reliable ones, discard the chance cue. A frozen-uniform parser cannot do this and is
correctly misled on degraded input. This is exactly the Competition Model's central claim (English/German/Italian
cue-weight dissociation), and the spiking build realizes the same weights as **Hebbian co-firing of a cue's vote
with the correct role assembly** (`enable_hebbian_learning=True`, the parser's own v16 rule).

---

## 4. Honesty notes (scope + what is/ isn't shown)

- **This is a numpy MECHANISM de-risk, not the brain build.** The COMPETITION + the reliability-weighted
  ACCUMULATION + the WINNER are the validated spiking computation; here they are a functional stand-in (a delta
  rule for the weights, a softmax/argmax for the settle). The production build runs on the `SimulationBridge`
  (§6). The feature LEXICONS (animacy, verb-selectional-fit) are HOST scaffolds — reused verbatim from
  `biased_competition_buffer.py`, already flagged there for conversion to a learned lexical-feature map. They
  supply each cue's VALUE; they do NOT supply the role decision (that is the learned-weight competition — the
  PERMUTED-CUE + NO-LEARNING controls guard against the lexicon doing the discrimination).
- **The non-canonical-minority training is a feature, not a fudge.** The no-learning control can only be made to
  collapse if the learner has something to learn — i.e. the training distribution must expose a cue-validity
  difference. A purely-canonical training set gives position 100% in-distribution validity and "learning" is
  vacuous (the first smoke confirmed this: NO-LEARN matched MULTICUE at 1.000). Naturalistic input (order is
  reliable but not perfect) is exactly the Competition Model's premise and what makes cue validity *learnable*.
- **Reproducibility bug found + fixed mid-arc.** The per-cue label noise was initially keyed on Python's builtin
  `str` `hash()`, which is **per-process salted** → not reproducible across runs. Re-keyed on a stable integer
  mix; verified `run1 == run2` byte-for-byte. (A real multi-seed-validity bug; caught before the headline.)
- **2-role (agent/patient), V≈16, single transitive clause.** The de-risk scope. The position-degrading battery
  is the gated metric (drop-verb does not degrade position and is reported separately, honestly).
- **What this does NOT show:** generalization across *similar* concepts (treating "dog"/"cat" as related cue-
  bearers) — that is the separate, already-mapped generalization arc, not this robustness goal. And the point-
  neuron risk for the cue competition itself is LOW (it is rate-coded reliability-weighted accumulation, NOT the
  analog/dendritic decorrelation that walled before; scoping §7.2).

---

## 5. Verdict

**GO (6/6 seeds).** Multi-cue competition with **learned cue validities** makes English comprehension robust to
degraded word order (scrambled, object-fronted) at **0.95 role accuracy where a position-only parser collapses to
~0.26 (object-fronting → 0.00)**, with the cues genuinely **load-bearing** (lesion collapses) and the weights
genuinely **learned** (no-learning collapses; distractor zeroed; position down-weighted below semantics), no
clean-canonical regression, and the **no-confab moat intact (0/240 breaches)**. Every decisive control in the
scoping's GO bar (§6.3/§6.4) passes on every seed.

---

## 6. Recommended next step — the spiking production build

Promote into the spiking substrate, reuse-by-import, default-OFF (byte-identical when off, like
`enable_multiframe` / `enable_attributed`):

1. **Re-point `research/runners/biased_competition_buffer.py`'s `sel_X` / `sel_FS_X` from referent-indexed to
   ROLE-indexed** (agent / patient role assemblies in mutual inhibition — the Rutishauser selective-inhibition
   WTA the navigation read-out + the buffer already run). The competition that today picks a *referent* picks a
   *role*.
2. **Add plastic cue→role projection families** (`position_cue → {agent,patient}`, `animacy_cue → {…}`,
   `verbfit_cue → {…}`), trained by the parser's existing Hebbian co-firing loop on the naturalistic distribution
   → the synaptic weights ARE the learned cue validities (this numpy de-risk's `position≈0.34 < animacy≈0.76`
   becomes the trained synaptic strengths).
3. **A cue driver** that lights each cue population per word (position = the existing conjunction unit; animacy /
   verb-fit = the lexical-feature lookups, the flagged host scaffold to neuralize as a learned lexical-feature
   map).
4. **Validate** the full who/what + moat pipeline on the degraded battery at production scale, multi-seed GPU,
   then add the **case/agreement cue** (Phase 2, true cross-language) as "just another competing cue."

The honest-residual conversion target stays the host feature lexicons → a learned synaptic lexical-feature map
(the buffer's documented conversion). The COMPETITION + the LEARNED cue validities — the load-bearing claim — are
validated here.

## 7. Provenance
- Scoping (the plan, GO bar, controls): `research/findings/2026-06-19-multicue-competition-parser-scoping.md`.
- Competition Model: Bates & MacWhinney 1982/1989; MacWhinney-Bates-Kliegl 1984 (English=order, German/Italian=
  agreement). Good-enough / NVN: Ferreira 2003. Neural cue-integration: catalog G.18 (LIP reliability-weighted
  additive accumulation), G.12 (Broca; semantic cues carry non-canonical comprehension), Desimone-Duncan 1995
  (biased competition), Wong-Wang 2006 (WTA). Biologically-plausible parser precedent + the single-cue gap:
  NEMO / Assembly Calculus (object-initial weakness; no animacy/case integration).
- Reuse substrate: `research/runners/biased_competition_buffer.py` (`ANIMACY`/`VERB_SELECTS` lexicons + the
  `sel_X`/`sel_FS_X` WTA), `research/runners/brain_conversational_agent.py:28` (`BridgeParser` position cue).
