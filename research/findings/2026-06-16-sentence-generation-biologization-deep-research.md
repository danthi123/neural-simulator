# Sentence-generation / formatting biologization — deep-research scoping doc (2026-06-16)

Read-only deep research. No code edited. Produced before any build, per the standing
"deep research + catalog review FIRST" directive. Controller to trust-but-verify the
load-bearing claims (flagged inline), then push + present before building.

**Scope of this doc:** the LAST host-code shortcut in the conversational OUTPUT path —
the step that, once the brain has SELECTED what to say, FORMATS the selected concepts
into a short English utterance. Binding is settled and is NOT the topic (single-attribute
learned bind validated on LIF today; multi-attribute bundling needs the fixed coincidence
primitive). Content-selection ("what to say") is also settled (the dlPFC spiking Control,
validated). This doc is ONLY about converting the host f-string formatter into a neural
sequence-generation mechanism at the achievable scale.

---

## 1. DIAGNOSIS

### 1a. What the host templating currently does (verified against the files)

The conversational agent recalls a role-filler SVO fact from spiking memory, then a host
function ORDERS and SPELLS it. Concretely:

- `research/runners/brain_conversational_agent.py:216` `describe(agent)` → `self.composer.render_fact(agent)`.
  `render_fact` takes the recalled `{agent, action, patient}` and emits the string in
  agent→action→patient order (e.g. `'dog go north'`). The **ordering of the three role
  slots and the emission of each slot as a word is done by host Python**, not by neurons.
- `research/runners/dialogue_agent.py:51-60` etc. — the Q&A surface (`_answer_yes_no`,
  `describe`, link/yes-no replies) builds replies with Python f-strings, e.g.
  `f"Yes -- {x} and {y} are associated (strength {s:.1f})."`. The *content* (x, y, the
  association) comes from the brain; the *surface form / word order* is a host template.

Under the project's **BRAIN-BASED-ONLY** standard (CLAUDE.md), host code is legitimate
ONLY for (1) the environment and (2) the body acting on motor output. A templated f-string
that orders/spells the output sits between "content chosen" and "utterance produced" — it
is cognitive, so it is a **documented host SHORTCUT** whose spiking/synaptic replacement is
the target. (Owner directive 2026-06-08; the composer's exact-inverse algebra carries the
identical "principled idealization, convert when feasible" status — CLAUDE.md "composer-as-idealization".)

The *minimal* thing the template does, stated precisely: given an UNORDERED set of
already-selected concept pools (≤3 role-fillers for an SVO answer; ≤4–6 for a short reply),
**(i) impose a serial order on them** (agent before action before patient; or a who/what
answer's single filler; or "X and Y are associated") and **(ii) read each ordered slot out
as its word**. That is a *serial-order production + read-out* problem — the canonical
neuroscience of sequence generation — NOT open-ended language modelling.

### 1b. The minimal brain-based target (in scope)

> A **neural sequence-generation mechanism** that, given a small set of already-selected
> concept-pool codes (the brain's own G.20 sparse codes) and a frame/intention, **emits
> them one at a time in a learned serial order**, and reads each emitted slot out as a
> word — replacing the f-string ordering+spelling for: (a) express one stored SVO fact
> ("dog go north"); (b) answer a who/what question (one filler, or abstain); (c) a fixed
> small inventory of short reply frames ("X and Y are associated"). Achievable scale: a few
> thousand neurons, ≤320 concepts, 2–6-word utterances, a handful of frames.

**Explicitly OUT of scope** (state plainly, this is the honest scale boundary):

- **Fluent, open-ended language generation.** The project already has a recorded, pre-data
  anchor: a from-scratch surrogate-gradient BPTT spiking LM (4-layer, ~134K params, Tiny
  Shakespeare/TinyStories) reached toy scale and was judged **~4 orders of magnitude too
  small** for real generation vs the ~1B-param "Project Nord" reference
  (`2026-05-07-Phase-2.3a-NEGATIVE-next-char-features.md`; the whole Generator S/D/E/F/G/H
  arc, `2026-05-17-generator-*`). Do not re-attempt fluent generation; it is a known wall.
- **Novel proposition CONTENT.** Selecting which fact/concept to express is the dlPFC Control
  (already validated, `content_selection_spiking.py`). This doc takes the selected content
  as given and only orders+verbalizes it.
- **Multi-attribute binding inside a slot.** A slot's filler is a single concept code (or a
  recursively-rendered clause, which the composer already handles). Binding is not re-opened.

### 1c. The decisive prior-work nuance (this re-scopes the whole effort)

There is a **fully-built, fully-anti-cheated, recorded NEGATIVE** for an HVC-style sequence
generator on this exact substrate — the "Increment G1 / G1.5 / Generator-P" arc. It MUST be
read correctly or the new work will repeat it. **[Controller: verify — this is the most
load-bearing claim in the doc.]**

- `sim/song_hvc.py` `SongHVC` is a real HVC synfire-chain controller (one state active per
  step, Hahnloser-2002; babble + DA-reinforce, Fee-Goldberg 2011). It provably emits an
  ordered concept sequence in isolation. `sim/predictive_coding.py` `PredictiveCoder`
  provably LEARNS order from prefix→next prediction error in isolation (FD-verified gradients;
  reviewed SOUND).
- They were wired onto the real 320-concept G.20 substrate (`song_g1_ignite.py` — write-only
  drive + `self_comprehend` read), trained self-supervised (`song_g1_train.py`), and judged
  by a **pre-registered held-out anti-cheat gate** (`song_g1_gate.py`: held-out-only,
  permuted-ORDER controls, control-calibrated frozen abstention floor, abs-floor 0.5, ≥10%
  margin). **All three FAILED** (`song_g1_gate.json`/`song_g1_traj_gate.json`/`song_g1_pc_gate.json`,
  all GATE: FAIL, mean_true 0.000).
- **Root cause (precise, and the key to re-scoping):** the failure was NOT that the generator
  couldn't order concepts. It was that they made the generator **self-supervise against a
  JUDGE that reads order back out of the substrate's residual pool state**, and that judge
  cannot discriminate order — Step-0 calibration recorded encoded-vs-control **AUC 0.775
  (G1 final), 0.40 (G1.5 trajectory), 0.475 (P)**. A judge that can't tell intended order
  from scrambled order gives the generator no gradient → the generator never learns
  (`mean_reward = 0.0000` every epoch for G1). The G1 doc states this verbatim: *"The
  bottleneck is the order-readout / order-discriminability of the integrated single-concept
  self-comprehension judge on the existing recognition-only substrate — not the songbird
  controller idea per se."* (`2026-05-16-generator-G1-songbird-NEGATIVE.md`).

**Implication for THIS scope (the re-frame):** the recognition-only G.20 substrate cannot
serve as its OWN order-judge, so a *closed-loop self-supervised* generator on it is dead.
But the present task does not need that loop. The output formatter has an **external,
unambiguous teacher**: the stored fact ITSELF specifies the target order (agent, action,
patient) and the target words. The new target is **supervised serial-order production with a
read-out to words**, teacher = the structured fact, judged by **does the emitted word
sequence match the fact's order** (an exact string/index comparison, not a fuzzy
self-comprehension residual). This sidesteps the exact mechanism that killed G1/G1.5/P. The
honest cost: this is a *more constrained, less self-contained* claim than G1's "the sim
generates novel propositions judged by its own comprehension" — and an honest NEGATIVE here
(neural order-production underperforming the f-string) is itself a valid deliverable
(maps what the substrate can serialize on its own).

---

## 2. RANKED biologically-grounded OPTIONS

Bars used below: a candidate is preferred if it (i) has direct catalog + literature grounding,
(ii) realizes at ≤ a few-thousand-neuron scale, (iii) reuses existing project machinery, and
(iv) avoids the G1 closed-loop-self-judge trap (uses the fact as an external teacher).

### Option 1 (RECOMMENDED) — Competitive-queuing serial-order generator over the concept/role pools, read out word-by-word

- **Biology:** Competitive Queuing (CQ) — a parallel *planning* layer holding all
  to-be-produced items with a **primacy activation gradient**, feeding a *competitive choice*
  layer (WTA: each item self-excites + laterally inhibits competitors; the winner is emitted
  then **self-inhibited / removed** so the next-strongest wins). Grossberg 1978; Houghton 1990;
  **Bullock & Rhodes 2003** ("Competitive queuing for planning and serial performance");
  Bullock 2004 (TICS). Maps onto catalog **G.07** (pre-SMA/SMA "internally generated
  sequences", Kandel 6e Ch 34 pp 822–828) and **H.19** (premotor/SMA sequential & rule-based
  action). The competitive-choice WTA is the project's *already-built* striatal/cortical
  WTA + FS lateral-inhibition motif (catalog **A.04** BG output WTA, **B.04** MSN lateral
  inhibition, **B.06** PV-FSI feedforward inhibition); the "emit then suppress" step is
  spike-frequency adaptation / inhibition-of-return (the project's `SaidTrace` made spiking).
- **Concrete spiking mechanism:** one *planning* pool per role-slot/concept, pre-loaded with a
  primacy gradient (agent strongest, then action, then patient — set by the frame); a
  *choice* layer of the corresponding concept pools with lateral inhibition (reuse
  `SpikingLoopContextBuffer` + the validated `internal_density=0`, `enable_ou=False` clean-WM
  config from `content_selection_spiking.py`); at each "syllable" step the most-active concept
  pool wins, drives its G.20 sparse pattern → word read-out (reuse `lang_output` cosine /
  `concept_speak_demo` A→W, validated 100% multi-seed), then is adaptation-suppressed so step
  t+1 emits the next role. The **order is produced by the WTA dynamics over a primacy gradient**,
  not by a host loop.
- **Why it fits the scale:** CQ is the *smallest* serial-order model in the literature — a
  planning layer + a choice WTA, both of which the project already builds at ~few-hundred-neuron
  pool scale. ≤6 items, ≤320 concepts is squarely in range. No BPTT, no ~1B params.
- **Main risk:** the primacy gradient must be *installable and stable* (CQ's classic failure
  mode is order errors when two items are near-equal in the planning layer — the project saw
  exactly this as "equidistant neighbours tie" at dt=1.0, CLAUDE.md one-bridge step 3). Mitigation:
  small ordered frames (≤3 SVO roles) where the gradient is large; the *external teacher* (the
  fact) lets a thin Hebbian/STDP step LEARN role→primacy from examples rather than hand-set it.
- **Project machinery already present:** YES — WTA + FS lateral inhibition, the clean-WM loop
  buffer, `SaidTrace` inhibition-of-return, the A→W word read-out, the G.20 sparse codes. This
  is mostly *assembly of validated parts*, which is why it is ranked first.

### Option 2 — HVC-like synfire-chain / learned-serial-order generator (reuse `SongHVC`), with a SUPERVISED word-match judge (NOT self-comprehension)

- **Biology:** HVC→RA ultra-sparse synfire-chain sequence generation — one HVC ensemble bursts
  at one precise time, the chain is the clock, RA reads it out to motor. **Hahnloser, Kozhevnikov
  & Fee 2002 (Nature, "ultra-sparse code")**; **Fee, Kozhevnikov & Andalman 2004**; Long & Fee
  2008; Jin 2009 (branching chains for variable sequences). Catalog: NO dedicated HVC/birdsong
  entry exists (verified — only an incidental Fiala-Grossberg-Bullock cerebellar-timing cite at
  line 1777); birdsong-HVC is **project-internal** (the SongHVC asset), grounded by the
  synfire/polychrony note at catalog **J.04** ("run a polychrony / synfire-chain experiment").
- **Concrete spiking mechanism:** keep `SongHVC`'s chain emitting the ordered concept sequence
  (state→concept map learned by babble+DA), keep the write-only `ignite_sequence` drive into
  G.20 + the A→W word read-out — but **replace the order-JUDGE**: instead of `self_comprehend`
  reading order out of the residual pool (the AUC-0.775 wall), judge order by **comparing the
  emitted word-index sequence to the fact's word-index sequence directly** (the fact is the
  teacher). DA reward = exact ordered-match of words, not fuzzy residual cosine.
- **Why it fits the scale:** the chain is `SongHVC(8, 64)` — trivially small; the substrate
  load (5 sparse bridges) is the only heavy part and is already built/loaded.
- **Main risk:** this is **exactly the arc that failed** — the failure mode must be the judge,
  not the chain, for this to now pass. The G1 docs argue (and Step-0 AUC corroborates) the
  generator was starved by the judge; swapping in a word-match teacher is the minimal change
  that tests that hypothesis. If it still fails, the failure was deeper (the A→W read-out
  itself can't preserve order through the substrate) — a genuinely informative negative.
  Risk that the controller re-treads a forbidden "config-crank the same negative" path: avoided
  because the JUDGE change is a *mechanism* change, not a tuning knob (and is pre-registered).
- **Project machinery already present:** YES, almost entirely — `sim/song_hvc.py`,
  `song_g1_ignite.py` (write-only drive + read), `song_g1_core.py` (`score_order` + the
  permuted-ORDER anti-cheat + `g1_verdict`), the checkpoint/sidecar/gate harness. Only the
  judge wiring is new.

### Option 3 — A small inventory of LEARNED syntactic FRAMES as neural sequences, selected by the dlPFC

- **Biology:** Pulvermüller **sequence detectors / discrete combinatorial neuronal assemblies
  (DCNAs)** — neural units that fire to AB but not BA, learned by Hebbian binding, generalizing
  across a syntactic category (Pulvermüller & Knoblauch 2009, *Neural Networks*; Pulvermüller
  2010, *Brain & Language*). Plus the population "geometry of sequences": **near-orthogonal
  rank subspaces** superimposed in one population (Xie et al. 2022, *Science*, macaque PFC;
  Dehaene et al. 2015, *Neuron*, transition-probabilities→algebraic-patterns→linguistic-trees).
  The frame = a fixed ordered template (S-V-O, "X and Y assoc"); the brain learns one DCNA-like
  sequence per frame and the dlPFC selects the frame.
- **Concrete spiking mechanism:** each frame is a short learned synfire/CQ sub-sequence of
  *role slots* (not words); the dlPFC Control (existing) picks the frame; the slots are then
  filled by the selected concepts and read out. The role-order lives in learned sequence
  detectors, not host strings.
- **Why it fits the scale:** a handful of frames × ≤6 slots is tiny; matches the few-frames
  reality of the current agent (SVO answer, who/what answer, yes-no, assoc-statement).
- **Main risk:** with only a handful of frames, a host `if frame == SVO: ...` selector is
  itself a shortcut — the frame SELECTION must be neural (dlPFC) AND the within-frame ordering
  neural, or it just relocates the template. Higher integration cost than Options 1–2; better
  as a *layer on top of* Option 1's serial-order engine than as a standalone first step.
- **Project machinery already present:** PARTIAL — dlPFC Control exists; sequence-detector
  assemblies do not (would be new, though buildable as Hebbian-bound AB-order pools).

### Option 4 (lowest priority) — Generative replay of stored utterance sequences (theta-gamma slot multiplexing)

- **Biology:** theta-gamma multiplex as an **ordinal-position buffer** — each item-assembly
  in its own gamma sub-cycle, ordinal position = theta phase; Lisman & Idiart 1995; **Lisman &
  Jensen 2013** (*Neuron*, "The θ-γ neural code"); **Heusser et al. 2016** (*Nat Neuro*,
  episodic sequence memory IS a theta-gamma phase code). Catalog **N.15** (theta-gamma
  multiplexed cell-assembly buffer, Bz Cycle 12 / Lisman-Idiart) + **D.18 supplemental** (theta
  sequences compress real-time order into one ~120 ms theta cycle = the STDP window) + **D.24**
  (theta-paced sequence compression). Replay/compression = catalog **D.19/N.07** (SWR).
- **Concrete spiking mechanism:** store an utterance as a theta-sequence (slots at successive
  theta phases), replay it compressed to emit the ordered concepts; read each out to a word.
  The order is carried by *phase*, which is biologically the substrate for "slots within a frame".
- **Why it fits the scale:** the oscillators are cheap (NM-framework sinusoidal
  `excitability_drive` at theta + a phase-locked faster modulator, per N.15 "Sim status:
  straightforward"); ≤7±2 slots is the native capacity.
- **Main risk:** the project has **NO theta or gamma generator in this path** (N.15/D.18 both
  "Sim status: missing") — this is the most *new substrate* of the four. Worse, the project's
  own dt-bound finding (CLAUDE.md one-bridge step 3: "rank-order/latency coding RESOLUTION is
  dt-bound; at dt=1.0 equidistant neighbours tie") flags that fine phase-ordinal coding is
  fragile at the merged-bridge dt. Highest biological faithfulness for *word-order-as-slots*,
  but highest build cost and dt-risk → defer behind Options 1/2.

**Ranking rationale:** Option 1 (CQ) is the most-faithful-ACHIEVABLE because it is the
canonical serial-order-PRODUCTION model (vs theta-gamma, which is canonical serial-order-
*memory*), realizes from parts the project has already validated, and uses the fact as an
external teacher (dodging the G1 trap). Option 2 (SongHVC + word-match judge) is the cheapest
*single experiment* because the asset is fully built — it is the natural cheap-first de-risk.
Option 4 (theta-gamma) is the most biologically-faithful account of "word order = slots" but
is out of cheap-reach (new oscillator substrate + dt-fragility) — recommend it as the
follow-on if Options 1/2 hit a flat order-production wall, not as the opener.

---

## 3. REUSABLE PROJECT MACHINERY (file paths + what each contributes)

Sequence-generation cores:
- `sim/song_hvc.py` — `SongHVC`: HVC synfire-chain controller; `rollout`/`babble`/`reinforce`/
  `set_intention_bias`. Emits an ordered concept-index sequence; learnable state→concept map.
- `sim/predictive_coding.py` — `PredictiveCoder`: prefix→next-concept top-down predictor;
  `learn`/`select_next`/`rollout`. FD-verified order learning (the only mechanism in the arc
  that learned order in isolation).
- `sim/bptt_snn.py` / `sim/bptt_snn_gpu.py` / `sim/surrogate_grad.py` — surrogate-gradient
  BPTT through LIF (ATan/fast-sigmoid). The from-scratch-LM path; **the scale anchor that says
  fluent generation is out of reach** — reuse only if a *tiny* supervised seq2word net is
  wanted, not for open generation.

Substrate drive + read-out (write-only, no-harm-proven):
- `research/runners/song_g1_ignite.py` — `load_members` (5 sparse 320-bridges), `ignite_sequence`
  (WRITE-ONLY concept drive into G.20), `self_comprehend` / `ignite_and_trajectory_decode`
  (read-out; NOTE the order-readout is the part that failed as a JUDGE). `ignite_prediction`
  (single-concept top-down drive).
- `research/runners/concept_speak_demo.py` (+ `chat_speak_*`) — A→W word read-out (drive a
  concept pool → decode the spoken word via `lang_output` cosine). **Validated 100% A→W
  multi-seed** — this is the "read each emitted slot out as a word" primitive, already working.
- `sim/bridge.py` engram API (`stimulate_tag`, `commit_engram_tag`) — alternative read-out /
  ensemble drive if pool-pattern drive is insufficient.

WTA / serial-order / inhibition-of-return parts (for Option 1 CQ):
- `research/runners/content_selection_spiking.py` — `SpikingLoopContextBuffer` (clean multi-
  concept WM, validated `internal_density=0`+`enable_ou=False`), `SpikingSpreadingController`
  (spiking relevance), and `SaidTrace`-based inhibition-of-return (the "emit-then-suppress"
  step CQ needs). `relevance_by_latency` (rank-order/latency coding) is a ready order signal.
- `research/runners/content_selection.py` — `ContentSelectionController`, `select_candidate`,
  `SaidTrace`: the validated "what to say" Control + said-trace.
- WTA biology in the engine: MSN lateral inhibition / FS-PV feedforward inhibition motifs
  (g11_bg_runner flags `--enable-msn-lateral-inhibition` etc.) — the competitive-choice layer.

Anti-cheat / gate harness (reuse verbatim — this is a major asset):
- `research/runners/song_g1_core.py` — `score_order` (ordered-match, penalizes trailing
  confabulation, strips clean-stop sentinels), `permuted_order_controls` (the load-bearing
  same-multiset order-scramble control), `compose_reward`, `g1_verdict` (FIXED bars:
  margin ≥10%, abs-floor 0.5). **Directly reusable to grade word-order production.**
- `research/runners/song_g1_gate.py` / `song_g1_train.py` — the pre-registered held-out gate +
  kill-safe resumable trainer + sidecar-frozen-floor + cross-mode-refusal discipline. The
  no-confab abstention plumbing (`abstention_gate`, frozen floor) is reusable for "abstain
  rather than emit a garbled utterance".
- `sim/train_checkpoint.py` — atomic kill-safe checkpoint (resume-on-rerun).

Conversational integration target (where the f-string lives today):
- `research/runners/brain_conversational_agent.py` (`describe`→`render_fact`, `:216`;
  `elaborate`→dlPFC, `:242`) and `research/runners/dialogue_agent.py` (reply f-strings) — the
  host formatters the neural mechanism replaces.

---

## 4. RECOMMENDED CHEAP-FIRST DE-RISK

**The single smallest experiment, and it falsifies-or-supports the top decision before any
GPU build.** It targets the question that gates BOTH Option 1 and Option 2 and that the G1 arc
left genuinely open: *with an EXTERNAL teacher (the fact's own order) instead of the
self-comprehension judge, can a small spiking serial-order generator emit ≤3 concept pools in
the correct order and have that order survive read-out, beating the permuted-order control?*

Two-phase, both CPU/numpy, both runnable in minutes (no 5-bridge load, no GPU):

**Setup (phase A — pure-core, seconds):** drive Option 1's CQ engine in pure numpy — a
planning layer of K≤3 role slots with a primacy gradient set from a frame, a choice WTA
(self-excite + lateral inhibit), emit-then-suppress (`SaidTrace`-style). Teacher = a small
fixed set of facts (e.g. 8 SVO triples drawn deterministically, same idiom as
`_build_frozen_propositions`). A thin Hebbian step learns role→primacy from the *train* facts.
Metric = `score_order(emitted_role_order, fact_role_order)` (REUSE `song_g1_core.score_order`)
on **held-out** facts (NEVER trained), vs the **best permuted-order control**
(`permuted_order_controls`) and the **host-template baseline** (the f-string order = 1.0).

**Setup (phase B — one small spiking bridge, minutes):** put the CQ choice layer on ONE
`SpikingLoopContextBuffer` bridge (the validated clean-WM config), drive ≤3 concept pools,
emit in WTA order, read each out to a word via the A→W cosine primitive (CPU/numpy backend,
`SIM_BACKEND=numpy`). Same `score_order` metric on held-out, same permuted control.

**GATE (pre-registered, FIXED — reuse `g1_verdict`'s bars exactly):** let `T` = mean held-out
ordered-word-match score, `P` = best permuted-order control score.
- **GO:** `T ≥ 0.5` (abs-floor: majority of the proposition correctly ordered) **AND** `T ≥
  1.10·P` (≥10% over the order-scramble control) **AND** the held-out facts cleared the
  abstention floor (no confabulated/garbled emission rewarded), on ≥6 seeds.
- **PARTIAL:** `T ≥ 1.10·P` but `T < 0.5` (real order signal, sub-majority) OR GO on phase A
  but not phase B (order produced in the core but lost through substrate read-out — localizes
  the wall to the read-out, itself decision-useful).
- **NEGATIVE:** `T < 1.10·P` (no order-learning above the same-multiset scramble) — i.e. the
  CQ engine with an external teacher still can't beat chance order, which would mean the wall
  is deeper than the G1 self-judge and re-confirm the substrate's order limit. An honest
  negative here is a valid deliverable.

Why this is the right cheap-first: it is the *minimal mechanism change* from the recorded
G1/G1.5/P negative (swap the self-comprehension judge for the fact-as-teacher), it reuses the
exact pre-registered scoring + permuted control that made those negatives trustworthy, it runs
on CPU in minutes, and its three outcomes each cleanly route the next move (GO → build Option 1
on GPU at 320-scale; PARTIAL-readout → fix the A→W order-preservation; NEGATIVE → record the
substrate serial-order boundary and stop, exactly as the project's discipline requires).

---

## 5. ANTI-CHEAT CONTROLS (the de-risk needs all of these or a "success" is an artifact)

1. **Permuted-ORDER control is load-bearing (must drop to chance).** Reuse
   `song_g1_core.permuted_order_controls`: the control has the **same concept multiset, order
   scrambled**. A mechanism that merely ignites the right concepts (no learned order) scores
   the true order ≈ permuted; only genuine order-learning beats it by ≥10%. This is THE control
   that exposed the G1 failure and must be the primary gate. (Without it, "the right words came
   out" passes trivially — the words are selected upstream.)

2. **Held-out facts only (leakage-free split).** Train role→primacy (or the chain map) on a
   FROZEN train set; grade ONLY on a disjoint held-out set never trained (reuse the
   `_build_frozen_propositions` train/held-out split idiom). The whole G1/G1.5/P arc's
   credibility rests on held-out-only grading; an in-sample "success" is memorization, not
   serialization (the explicit Inc-3 lesson, `2026-05-16-generator-increment3-*`).

3. **Host-template baseline to beat-or-honestly-match.** The f-string gives order = 1.0 by
   construction. Report `T` against it explicitly. The neural mechanism does NOT have to beat
   1.0 to be the deliverable (an honest underperformance IS the scientific result per the
   BRAIN-BASED-ONLY standard), but it MUST be reported so "neural sentence generation works" is
   never claimed when it is materially below the template it replaces.

4. **No-confab abstention check (the moat must not be weakened).** If the selected content is
   absent/garbled, the mechanism must **abstain** (emit nothing / clean stop), not confabulate
   an order. Reuse the frozen abstention floor + `score_order`'s trailing-confabulation penalty
   + a `describe()`-style `is None` assertion on an unknown agent. Verify the moat empirically
   (the project's standing no-confab probe pattern) — a generator that fills order by guessing
   when it shouldn't is a regression even if held-out order improves.

5. **Degenerate-tie-break guard (carry-forward G1 Minor #4).** The candidate ordering fed to
   any WTA/argmax MUST be the fixed canonical (vocab-index / range(n)) order, NEVER target-
   ordered — else a degenerate "always pick the first" rollout correlates with the target and
   scores above chance without learning (the exact anti-cheat in `_canonical_candidates`).

6. **Bars frozen, protocol run to completion, no config-cranking.** `g1_verdict`'s
   `_G1_MARGIN=0.10` / `_G1_ABS_FLOOR=0.5` are module constants — never tuned per run. Pre-
   register the gate before seeing held-out data; run all seeds (≥6) to completion; do not spin
   variants to chase a pass (the garden-of-forking-paths the project's discipline forbids).
   This experiment is a NEW mechanism (fact-as-teacher CQ), not a re-tune of the recorded G1
   negative — keep that distinction explicit so it is not mistaken for re-running a dead path.

---

## Files reviewed (provenance)

Project: `sim/song_hvc.py`, `sim/predictive_coding.py`, `sim/bptt_snn.py`,
`sim/surrogate_grad.py`, `sim/compose_temporal_bind.py`,
`research/runners/song_g1_{core,ignite,noharm_probe,train,gate}.py`,
`research/runners/content_selection_spiking.py`, `research/runners/dialogue_agent.py`,
`research/runners/brain_conversational_agent.py`,
`research/findings/2026-05-16-generator-G1-songbird-NEGATIVE.md`,
`research/findings/2026-05-16-generator-P-predictive-coding-NEGATIVE.md`,
`research/findings/raw/g11_bg/song_g1_{gate,pc_gate}.json` (+ `.meta.json` sidecars).
Catalog: `sim-catalog/references/feature-catalog.md` entries **G.07, H.19, N.15, D.18
(+supplemental), D.24, J.04, A.04, B.04, B.06, H.17, E.03** (and verified NO HVC/birdsong/CQ
entry exists). Glossary: `references/glossary.md` (theta-gamma / sequence-learning rows).

## Literature cited

- Hahnloser, Kozhevnikov & Fee 2002, *Nature* 419:65 — ultra-sparse HVC synfire code for song.
- Fee, Kozhevnikov & Andalman 2004 — neural mechanisms of vocal sequence generation; Long & Fee
  2008; Jin 2009 (branching chains for variable sequences).
- Grossberg 1978; Houghton 1990; **Bullock & Rhodes 2003** (competitive queuing for planning &
  serial performance); Bullock 2004, *TICS* — competitive-queuing serial-order production.
- Lisman & Idiart 1995; **Lisman & Jensen 2013**, *Neuron* (the θ-γ neural code); **Heusser et
  al. 2016**, *Nat. Neuro.* — theta-gamma phase code for ordinal position / sequence memory.
- **Dehaene, Meyniel, Wacongne, Wang & Pallier 2015**, *Neuron* — neural representation of
  sequences (transition probabilities → algebraic patterns → linguistic trees).
- Xie et al. 2022, *Science* — geometry of sequence working memory (near-orthogonal rank
  subspaces) in macaque PFC.
- Botvinick & Plaut 2006, *Psychol. Review* — recurrent NN model of serial-order STM.
- **Pulvermüller & Knoblauch 2009**, *Neural Networks*; Pulvermüller 2010, *Brain & Language*
  — sequence detectors / discrete combinatorial neuronal assemblies for syntax / word order.
- Rao & Ballard 1999; Friston (active inference); Bastos et al. 2012 — predictive coding (the
  PredictiveCoder basis; recorded as a terminal negative for the closed-loop variant).
- Kandel 6e Ch 34 (pre-SMA/SMA, pp 822–835); Buzsáki 2006 Cycles 11–12 (theta sequences,
  theta-gamma multiplex).
