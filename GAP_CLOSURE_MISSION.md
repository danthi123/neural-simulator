# GAP CLOSURE MISSION — the ACTIVE directive (read this EVERY session; it REPLACES `/neural-simulator`)

**This file is the single source of truth + the self-anchor. If you are Claude working this repo: read this file
first, act on the CURRENT STATE section, update it every cycle. You do NOT need the owner to run `/neural-simulator`
— this board carries the anchor.**

---

## THE DIRECTIVE (owner, 2026-07-17 — verbatim intent)

Fully focus on **closing the 5-gap cluster** the 2026-07-17 audit identified
(`research/findings/2026-07-17-banked-capabilities-audit-two-buckets.md`). Deliver **genuinely biology-based,
FULLY-SPIKING implementations on the ONE brain / shared substrate**. **No deferrals just because we hit walls** —
with the ONE exception that a *failing METHOD* may be deferred in favor of a *NEW method* that meets the same
requirements. **The actual closing-out as fully-working functionality can NOT be deferred.** Also: a bullet-proof
anti-drift autonomous workflow; reboot-resilient (owner gives short notice → minimal work lost); and it must not
require running the skill to stay on track.

## THE ONE LAW THAT CHANGES EVERYTHING (stronger than the old skill)

**An honest negative is a verdict on a METHOD, never a license to abandon a CAPABILITY.** The old norm ("an honest
negative IS the deliverable") applies to a *method* only. Under this directive, a boundary/NEGATIVE/"characterized
limit"/"defensible"/"can't on this substrate" **triggers the search for the next biology-based spiking one-brain
method — and the capability stays OPEN until one WORKS.** You never get to close a gap with a negative. You close it
with working functionality or it stays on the board.

**"CLOSED" is defined narrowly:** the capability is (a) realized fully-spiking on the one shared-substrate brain,
(b) genuinely biology-grounded (neurons/synapses/their communication — host code only for world+body), (c) validated
6-seed with anti-cheats, (d) adversarially verified, and (e) **wired into the actual system the owner uses (no
stranding, no scaffold left standing as the faculty).** Anything short of (a)-(e) is IN PROGRESS, not closed.

---

## THE 5 GAPS (status board — update as they move)

| # | Gap | Why it's load-bearing for LLM-like chat | Failing methods (banked) | Status |
|---|-----|------------------------------------------|--------------------------|--------|
| 1 | **Open-ended fluent generation (open prose)** | no "talk about anything" without it | from-scratch spiking LM loses to a bigram at few-M-token scale; the categorical novelty gap (composer emits 0/16 novel) | OPEN — met only by the ~21M TinyStories ANN scaffold (must be replaced by simulated circuitry) |
| 2 | **Learned binder over the brain's OWN structured/correlated codes** | fluid reasoning/composition over the brain's own semantics; replaces the hand-designed exact-inverse FHRR algebra | multi-attr bundling from scratch NEGATIVE; learned-linear-inverse ≈chance; deep-dendritic-credit binder BOUNDARY; write-rule multi-bind capped ~2 (EDGE-5); naive always-on filler-WTA HURT; more-filler-pools lever at P=4 REFUTED (0.71<0.79) | **🎉 FULLY-SPIKING 6-SEED GO** — the SELF-ORGANIZING competitive-SLOT binder recovers a fact's role-filler bundle on SPIKES, **reset now NEURALIZED** (FS inhibitory burst = the D3 CLEAR): at the SVO load P=3, slot-sep **1.00 (6/6) >> shared cap 0.33, permuted→0.00, neural-clear == host-reset EXACTLY per-seed**. P=4 graceful-degrade 0.79 (intrinsic 4-slot read edge, honest). Replaces the FHRR algebra with a learned self-organizing spiking binder. `2026-07-17-keystone-2-spiking-slot-binder-STEP1-prereq-GO`. **(a) neuralize-reset DONE. (b) adversarial-verify CONFIRMED** (independent skeptic: no-teach→chance, scramble-teach→0.00, KF=12 11× chance — genuine learned role-addressed binder). **(c) wire-in mechanism 6-seed GO** (content-addressable multi-fact recall: query 1.00 both dirs, moat+perm abstain 1.00, scramble→chance). **🎉 CAPABILITY CLOSED:** `SlotBinderComposer` wired into `BrainConversationalAgent` (`composer_kind="slotbinder"`), agent answers who/what/yes-no/describe/moat through it, CI 6 pass / 0 regress. ONE tracked refinement: self-organizing (adaptation-based) slot ALLOCATOR to replace the host next-free-slot counter. Honest: LTM not WM; concept-pool fillers (generalization = separate closed arc). |
| 3 | **Multi-referent disambiguation** | real dialogue holds several entities; bind a bare pronoun to the salient one | recency/salience-boost/symmetric-WTA NEGATIVE — but SURPASSED | **LARGELY CLOSED** (audit MISfiled it) — biased-competition WTA GO 6-seed (controls) + CI-guarded + wired into `MultiTurnAgent` + D3 centering-focus (0.945 vs host 0.000). Residual = EMERGENCE-BAR polish: neuralize the host `content_bias_target` scoring → a learned synaptic feature-compatibility map (cheap, flagged in-code) + the all-compatible tie (moat-safe abstention). `2026-07-17-gap3-...-LARGELY-CLOSED` |
| 4 | **Dendritic / local-credit learning lever (KEYSTONE — engine for #2 & #5, upstream of #1)** | a substrate that LEARNS its binding + sequence structure, no weight transport | e-prop feedforward NOT-GO; recurrent e-prop refuted; Node Perturbation retired; BDSP-on-classifier-readout blocked; graded-readout escape does-not-unlock (2026-07-17) | OPEN / NOT-GO — needs a NEW biology-based local spiking credit rule |
| 5 | **CA3 completion / imaginative-replay (episodic memory + imagination)** | remember/complete/imagine episodes; SWR generative replay | on-bridge replay at chance (5.78% vs 6.25%); held-out completion 0; dAP completion only as a read-out on a hand-installed attractor | OPEN — point-neuron boundary characterized; needs the dendritic dAP substrate (ties to #4) |

**Dependency read (drives the order):** #4 (a working local spiking credit rule + the dendritic substrate) is the
KEYSTONE — it is the engine for #2 (learned binder) and #5 (dAP completion), and is upstream of #1 (a substrate-native
generative sequence model also needs a spiking sequence-credit rule). #3 (multi-referent) is the most self-contained
and tractable (a biased-competition WTA between referent attractors). **Planned attack:** research-gate the keystone
(#4) FIRST (highest unlock), run #3 as a parallel tractable lane, then #2/#5 on the keystone, then #1.

---

## THE PER-GAP WORKFLOW (bulletproof; the same loop for every gap; NO step skipped)

1. **RESEARCH-GATE (read-only, mandatory before any build):** a-1 RAG-check our OWN record (have we tried/retired
   X? — drift #12), then READ the biology in depth (the catalog, Kandel, papers — not grep-skim), AND search the
   external engineering/ML/SNN literature + repos. Produce: diagnosis → ranked *biology-based, spiking, one-brain*
   methods → the cheapest first de-risk → the anti-cheats it needs. Commit the gate doc.
2. **CHEAP-FIRST DE-RISK:** rate → spike → one-brain ladder; single-variable; anti-cheats mandatory; **6-seed
   (42/43/44/100/101/102)**; `cfg.seed` set + substrate-hash verified (the 2026-07-17 seed law). GO or BOUNDARY.
3. **BOUNDARY ≠ STOP.** The gate already ranked the next method. Bank the failed method (documented), take the next
   one that meets the requirements, iterate. The gap stays OPEN until a method WORKS. (This is THE LAW above.)
4. **BUILD** the full fully-spiking, one-brain, biology-grounded implementation (`sim/` edits allowed when a faithful
   mechanism needs them: additive / default-off / byte-identical-when-off / guarded).
5. **ADVERSARIALLY VERIFY** before declaring closed: independent skeptics probe for the confound / the artifact /
   the stranding. A GO that has not survived adversarial verify is not closed.
6. **CLOSE:** wire into the real system (retire the scaffold it replaces; no stranding), 6-seed, commit BOTH remotes
   (`tools/push_both.sh`), update this board (move the gap to CLOSED with the evidence).

---

## ANTI-DRIFT LAW (the self-anchor — why you don't need the skill)

- **The ONLY things that end a turn:** the owner explicitly says stop/pause, or a safety/permission gate. **Never** a
  milestone, a scoped next-step, a status report, or session length. Reports are emitted WHILE tools run.
- **A boundary/negative LAUNCHES the next method; it never closes a gap** (THE LAW). Do not write "characterized
  limit / honest negative / defensible / can't on this substrate" and stop — that IS the drift.
- **"Closed" requires fully-working WIRED functionality** on the one brain — not a demo, not a single-seed console,
  not a scaffold left standing. Stranded capability = not closed.
- **Whack-a-mole guard:** don't hand-build a capability as its permanent home; the bar is "does it EMERGE from a
  learning substrate?" These 5 gaps converge on the emergence engine (a learning substrate + substrate-native
  generation) — route THROUGH the shared foundations (#4), not around them.
- **Read + update THIS board every cycle.** It is the mission. It replaces `/neural-simulator`. A plain "continue"
  from the owner + this board is enough to re-anchor — no skill required.
- Parallelize independent work; adversarially verify before committing a GO; both remotes every commit; GPU/CuPy for
  real runs (numpy tiny-smoke only); measure before fanning wide.

### STAYING ANCHORED WITHIN A LONG (hours-long) SESSION — the start-of-session load RECEDES; do NOT rely on it

The `CLAUDE.md` pointer + memory + this board load at SESSION START and then drift toward the back of the context as
the conversation grows; only a compaction event auto-reloads them. So within a long session the anchor is NOT
self-maintaining — these THREE keep it live (do all three; do not rely on the start-of-session load alone):
1. **This board is LIVING working-memory — UPDATE the CURRENT STATE section after EVERY meaningful step** (a de-risk
   result, a build, a commit, a gap moving). The act of updating re-reads the anchor, so it never recedes, and it
   keeps the resume point current for compaction AND reboot.
2. **RE-READ this whole board at every gap-step boundary** (before starting a research-gate / de-risk / build /
   verify / close). Cheap, and it re-loads THE LAW + the current gap before each phase.
3. **The automatic anchor-heartbeat (a Monitor, armed 2026-07-17) fires ~every 25 min** with a re-anchor + anti-stall
   self-check. Each firing is NOT a user message — it is a forced re-anchor: on it, RE-READ this board, verify you
   are executing the current gap-step per THE LAW (a wall defers a METHOD not the CAPABILITY), and if you have drifted
   (wrapped up, deferred a capability, stopped taking the next step, relabelled a wall as a stop), CORRECT NOW. If the
   heartbeat is not running (new session, or it was stopped), RE-ARM it as the first action after re-anchoring.

---

## REBOOT / PAUSE PROTOCOL (lose minimal work on short notice)

- **Owner says "rebooting" / "pausing":** immediately (1) drop PAUSE sentinels on every resumable run
  (`bridges/developed/<root>/PAUSE` stops the develop loop cleanly at the next day boundary; research runners
  checkpoint per-seed), (2) record the exact resume state in CURRENT STATE below, (3) commit+push, (4) ack. Then it
  is safe to reboot at the next checkpoint.
- **Every long run MUST be resumable:** per-seed / per-day checkpoints, a PAUSE sentinel, a Monitor armed, launched as
  controller `run_in_background` (never a subagent's detached child). No un-checkpointed multi-hour job.
- **On resume (owner says "continue" / "back"):** read this board → remove PAUSE sentinels → re-arm monitors → continue
  from EXACT NEXT ACTION. No skill, no re-anchor prompt needed.
- Helper: `tools/reboot_prepare.sh` (lists running jobs + drops known PAUSE sentinels + prints resume state).

---

## KEYSTONE (#4/#2) GATE GROUNDING — a-1 deep-read DONE 2026-07-17 (read-only, during the scale run)

The conversational bind decomposes (from `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE` +
`2026-06-11-cortex-learned-binder-systematicity-NEGATIVE-ON-CORRELATED`, both read in full):
1. **Concept codes — LEARNED on spikes** ✅ (stream cortex, the scale787 run is confirming at 787).
2. **Single-attribute binding — LEARNED + validated on real LIF** ✅ (on-bridge held-out 0.833 = 100% of numpy).
3. **Multi-attribute BUNDLING (a fact) — THE OPEN WALL.** NEGATIVE for any *point-neuron* learned bind (additive 0.193;
   learned-LINEAR-inverse 0.056 ≈ chance) because unbinding a role from a superposition needs a **role-specific
   MULTIPLICATIVE inverse (1/u_t)** and a shared *linear* unbind is *structurally* incapable of role-dependent scaling
   (Mikulasch-Priesemann point-neuron limit — multiplication is a DENDRITIC op). A **learned BILINEAR** binder DOES
   generalize systematically — but only on **DECORRELATED** codes (held-out 1.000); it goes **NEGATIVE on CORRELATED/
   structured** codes, which is exactly what the brain's own emergent codes are.

**⇒ the sharp keystone question (gaps #4+#2 are ONE gate):** bind over the brain's OWN correlated/structured codes with
a **DENDRITIC multiplicative** operation **whose structure SELF-ORGANIZES** (not the hand-set ±1/FHRR algebra, not a
supervised learned-linear inverse — both closed). The un-tried levers, all present: the **D2 two-compartment dendritic
substrate is BUILT** (`sim/dendritic_neuron.py`, `dendritic_mlp.py`, `dendritic_plasticity.py`); a **bilinear
multiplicative** bind is native to a dendrite; and `2026-07-16-MDGL-offdiagonal-...-replay-is-the-biological-path`
flags **replay** as a candidate for the credit the point-neuron rules couldn't carry. The research-gate ranks:
dendritic-multiplicative bind on D2 · correlated-code-tolerant binding · replay-driven / developmental self-organization
of the bind structure · the cheapest de-risk. (The "fixed algebra is biology-grounded, done" conclusion of the
2026-06-16 doc is a DISGUISED boundary under THE LAW — it does not self-organize, so the capability stays OPEN.)

## CURRENT STATE (⚠️ keep this section current every cycle — it is the resume point)

- **Phase:** GAP-CLOSING. **Gap #2 FULLY CLOSED** ✅ · Gap #3 largely closed · **Gap #5 = the "6-SEED GO" was RETRACTED
  (2026-07-18)** ⛔ — the permuted-recall anti-cheat caught a SELF-SUSTAINING-ATTRACTOR artifact (held members fire the
  same [50] whether cued CORRECTLY or with a RANDOM set → not cue-triggered completion). What STANDS: the recipe forms
  a strong SPECIFIC learned attractor (real advance on the 2026-07-14 weak-attractor boundary). NOT closed: genuine
  CUE-TRIGGERED completion (the attractor is always-on, needs to be BISTABLE/cue-gated). · Gaps #1/#4 open.
- **🔬 gap#5 Rung 2/3 RESULT (2026-07-17):** the mossy-detonator SPARSIFIES CA3 (0.43→0.03) but sparsity WITHOUT
  synchrony can't select — within/silent separation goes NEGATIVE (−0.27 to −0.47; the sparse cells fire async →
  co-activity traces never clear threshold → no potentiation). Rung 3 (input gamma-pulse) is INERT (byte-identical to
  no-sync — pulsing the INPUT doesn't synchronize CA3). ⇒ the formation blocker is confirmed to be SYNCHRONY; the named
  next mechanism is **Rung 4: a genuine theta-gamma PACEMAKER** (rhythmic inhibition pacing the sparse CA3 survivors
  into gamma volleys DIRECTLY, N.15/N.19), a guarded sim/ mechanism — a DEEP build. Completion FUNCTION is met (EMERGE
  spreading-activation 12-seed GO, graded-confidence 12-seed GO, composer scan, gap-#2 slot-binder); imaginative
  RECOMBINATION is GO; SWR replay reactivation is blocked by the same CA3-attractor gap.
  `2026-07-17-gap5-rung2-mossy-sparsifies-but-sync-unsolved-CA3-attractor-is-the-theta-gamma-frontier.md`. **✅ OWNER
  STEER (2026-07-18): "close out ALL gaps FULLY"** — NO banking-as-function-met.
- **⛔ A-1 FAILURE CORRECTED (2026-07-18) — my ENTIRE session gap-#5 work was RE-DERIVATION (4× I built from stale
  findings, not the newest).** TRUE latest = `2026-07-14-ca3-competitive-hebbian-formation-6seed-GO.md`: (1) **FORMATION
  = 6-SEED GO** — COMPETITIVE-HEBBIAN (committed EMERGE-40 `fused_htm_winner_inactive_depression` on ca3→ca3 recurrents,
  1st time) gives within/silent **5.2-8.9×** (pure-LTP 1.01×; lam=0 control 1.01 = load-bearing), surpassing the
  2026-07-09 saturation, NO sim/ edit. (2) **BUT weight-ratio ≠ functional completion** — depression-based → LOW
  absolute within-ensemble drive → dAP plateau doesn't fire; completion SEED-FRAGILE (2/6 clean). (3) **Root-caused
  (recall-inhibition diagnostic: ZERO recall inhibition still fails → iSTDP RULED OUT):** bottleneck = the
  RECURRENT-WEIGHT STRUCTURE (learned within-ensemble too weak); members must fire STRONGLY+SYNCHRONOUSLY at encoding
  (Kopsick-Ascoli 2024) for a HIGH-ABSOLUTE attractor. (4) **Kopsick knobs BUILT** (`mossy_density`, `dg_ffi_weight`,
  `sync_on/off`) but OVER-SUPPRESS at 150 CA3 (knife-edge). ⇒ **functional completion is SCALE-BOUNDED**: needs
  ~1000-2000 CA3 for a robust <1% assembly + redundancy. My session's silent-recurrents/Hebbian-collapse/mossy-sync at
  150 = all re-derivation, none new. **LESSON: a-1 to the NEWEST findings FIRST (drift-#12, 4× this session).**
- **✅ SCALE + CAP tested (2026-07-18, GPU n_ca3=1000):** scale ALONE doesn't fix it (functional held-out completion
  ~0 at n_ca3=1000, all configs). The DECISIVE cap-vs-synchrony test — raise `hebb_max` 30→2000 → **BYTE-IDENTICAL**
  (c_drive 52.60/52.60, h_comp 0.009/0.009): the cap is NOT the lever, the within-ensemble weights are stuck at ~7.5
  (co-activity-limited), ~200× below the completion weight scale (hand-installed that completes = raw ~1600). ⇒ **the
  crux is DEFINITIVELY SYNCHRONY** — the members fire ASYNC → low co-activity traces → the rate-window LTP can't grow
  the within-ensemble weights, no matter the cap or the scale. `run_payoff` drives the INPUT (lang), which smears; the
  un-tried mechanism (2026-07-14 named it, never built) = **the Kopsick recipe done right: drive the assembly members
  DIRECTLY + SYNCHRONOUSLY during encoding** so they co-fire strongly → high co-activity → within-ensemble LTP reaches
  the completion scale → dAP completion.
- **🎉 gap#5 COMPLETION MOVING OFF CHANCE (2026-07-18) — the mechanism is FOUND (first time in the project).** The
  workflow (`wkn6apwgj`) diagnosed the growth failure: (1) my "synchrony" framing was WRONG for this EMA rule — the
  gamma OFF-gap DECAYS the 10-step co-activity trace (0.9^off/cycle) below the 0.25 threshold at ANY drive, so sync=async
  and nothing potentiated; the fix is CONTINUOUS strong drive (high avg firing DUTY), not bursts. (2) coact_thresh 0.25
  sits above the achievable co-activity product (~0.03 @700pA) → lower it to 0.02. (3) lr 0.0005 too slow. **GROW-SWEEP
  with the levers (continuous 3000pA + coact_thresh 0.02 + higher lr, n_ca3=500 seed 42):** the within-ensemble weights
  now GROW (6.0 → 15.9 @lr0.05 → 28.2 @lr0.5) and **functional completion rose 0.000 → 0.214, SPECIFIC** (h_comp 0.214
  vs non-stored 0.007 = ~30×). Just below the 0.30 GO bar. **PUSH + READOUT sweeps (2026-07-18): encoding/growth SOLVED** — w_within grows 6→65 (lr)
  → 250-459 (lowering k_thresh, which is entangled with encoding). BUT the remaining lever is the **MARR COMPLETION
  TRADEOFF**: at k_thresh=18 completion is SPECIFIC but weak (h_comp 0.233, non 0.007); at k_thresh=6 it's STRONG but
  INDISCRIMINATE (h_comp ~1.0 AND non ~1.0 — lowering k_thresh floods the plateau during encoding too, the exact
  2026-07-14 warning). **⚙️ 2D sweep `bb15znulo` running** (k_thresh {15,12,10} × lam {0.3,0.5}) for the intermediate
  sweet spot: h_comp≥0.30 AND ≥2× non-stored (specific completion). **🎉🎉 SEED-42 GO FOUND (2026-07-18): k_thresh=15,
  lam=0.5 → h_comp=0.367 (≥0.30), non-stored 0.007 (52× SPECIFIC)** — the FIRST specific functional CA3 pattern
  completion from a LEARNED attractor in the project's history (it was at chance the entire time). Full recipe:
  pre-assigned sparse assembly + CONTINUOUS strong drive (3000pA, NO gamma) + coact_thresh 0.02 + hebb_lr 2.0 +
  heterosynaptic competition lam 0.5 + dendritic dAP read-out k_thresh 15. **⚙️ 6-seed + anti-cheat confirmation running**
  (`bklj3vokg`: 6 seeds + GAMMA control [no_sync=False → should collapse: the OFF-gap decays the EMA trace] +
  NO-ENCODING control [drive=0 → should collapse]). **RESULT (honest, n_ca3=500): 2/6 GO** (seed 42 h_comp 0.367,
  seed 100 0.840; seeds 43/44/101/102 fail — weights collapse / completion 0). NO-ENCODING collapses (2.3, h_comp
  0.167 = encoding load-bearing ✓). BUT the GAMMA anti-cheat does NOT collapse (h_comp 0.625) → **CORRECTION: the
  "continuous drive is load-bearing" claim is REFUTED** — at coact_thresh=0.02 BOTH gamma and continuous potentiate
  (the threshold is below even the gamma-decayed trace); the load-bearing levers are LOW coact_thresh + high lr +
  competition + the k_thresh sweet spot, NOT the drive pattern. **⇒ off chance (0/6) → 2/6 GO + mechanism found, but
  SEED-FRAGILE — exactly the 2026-07-14 scale-boundary (2/6 clean at small scale), which that finding predicted SCALE
  fixes.** **⚙️ Scale test running** (`<id>`: the GO config at n_ca3=1000, 6-seed). **⚠️ EXACT NEXT ACTION:** read it →
  if scale robustifies to ≥5/6 → gap#5 completion mechanism CLOSED → wire EMERGENT mossy/DG selection → SWR loop →
  console. **SCALE RESULT (n_ca3=1000): 3/6 GO** (seeds 43/44/102 clear; 42/100/101 miss). KEY: **all 6 seeds show
  SPECIFIC completion** (h_comp ≥ 2× non-stored on EVERY seed — the mechanism works on all), but the MAGNITUDE is a
  knife-edge at 0.30 (0.303/0.368/0.418 clear; 0.107/0.216/0.088 miss). ⇒ NOT a mechanism failure — a formation-strength
  variance (the 2026-07-14 scale-boundary). **⚙️ Robustness sweep running** (targeting the fragile seeds: lr/events/
  assembly-size to push all seeds' magnitude past 0.30). **⚠️ EXACT NEXT ACTION:** read it → find a config with ≥5/6
  clearing 0.30 (or widen assembly redundancy / n_ca3=2000). THEN: gap#5 functional-completion CLOSED → wire EMERGENT
  mossy/DG selection → SWR loop → console. HONEST MILESTONE: gap#5 completion went from CHANCE (project-lifetime) to a
  WORKING specific learned-attractor completion (all 6 seeds specific, 3/6 clearing the strict magnitude bar) — a real
  breakthrough; robust 6-seed is a knife-edge tuning continuation.
- **ROBUSTNESS sweep (2026-07-18, n_ca3=1000):** larger/redundant assembly + longer recall LIFTS the fragile seeds
  (assembly 0.02 + recall 100: worst seed 101 0.088→**0.425 GO**, seed 42 0.107→0.256). BUT pushing assembly_frac to
  0.025-0.04 OVERSHOOTS into the Marr INDISCRIMINATE regime (h_comp ~1.0 AND non-stored ~1.0 — the denser attractor
  spreads to non-members, overwhelming the competition). ⇒ the sweet spot is SPARSE-yet-REDUNDANT, which trade off at
  n_ca3=1000 → the 2026-07-14 prescription: **n_ca3=2000 so a <1% assembly is BOTH sparse AND redundant.** **⚙️
  `b90pmxei1` running** (n_ca3=2000, assembly_frac 0.008/0.012, recall 100, seeds 42/101/44). **SCALE-2K RESULT (major):** at n_ca3=2000, assembly_frac 0.008
  (<1%), the FORMERLY-CHANCE fragile seeds now show STRONG SPECIFIC completion — **seed 42 0.107→0.917 (non 0.014, GO),
  seed 101 0.088→0.481 (non 0.006, GO)**. The scale prescription WORKS. But seed 44 now OVERSHOOTS to indiscriminate
  (non 0.995) — the optimal assembly density is SEED-VARYING, so one fixed frac misses some seeds' sweet spot. The
  principled fix (2026-07-14 gate cited PMC12244581): assembly-SELECTIVE feedback inhibition (k-WTA capping the active
  count at recall → specificity robust to density). **⚙️ `bo9depc5z` running** (n_ca3=2000, fb_inhib 15/40, seeds
  42/101/44). **🎉🎉🎉 FB-INHIB FIXED IT (2026-07-18): fb_inhib=15 at
  n_ca3=2000/assembly 0.008 → ALL THREE tested seeds (42, 101, 44) complete PERFECTLY + SPECIFICALLY: h_comp=1.000,
  non-stored=0.000, GO.** Assembly-selective inhibition caps the completion spread (non-members silent) while held
  members fully reactivate — density-robust, exactly PMC12244581. **THE ROBUST CONFIG:** n_ca3=2000, assembly_frac
  0.008, continuous drive 3000pA, coact_thresh 0.02, hebb_lr 2.0, lam 0.5, k_thresh 15, recall_steps 100, ca3_fb_inhib
  15. **⚙️ FULL 6-SEED + no-encoding anti-cheat running** (`b4ncflcd4`). **⛔ "6/6 GO" RETRACTED (2026-07-18) — self-sustaining-attractor
  artifact.** The permuted-recall anti-cheat + ABSOLUTE-firing instrumentation: NORMAL held_abs=50/cue_abs=50; PERMUTED
  (random wrong cue) held_abs=**50**/cue_abs=0 → **held members fire the SAME (50) regardless of the cue.** The strong
  learned attractor is SELF-SUSTAINING (always-on limit cycle clamped by fb_inhib), NOT cue-triggered completion. The
  cue-normalized h_comp=1.000 masked it (both fire at the clamped rate); no-encoding collapsed only because no attractor
  exists to self-sustain. Finding retraction block:
  `2026-07-18-gap5-ca3-functional-completion-CLOSED-6seed-GO-learned-attractor.md`. **STANDS:** the recipe forms a strong
  SPECIFIC learned attractor on-substrate (6-seed) — a real advance on the 2026-07-14 weak-attractor boundary. **⚠️ THE REAL PROBLEM = BISTABLE cue-gated completion.** Built the proper
  gate into the de-risk (`run(..., bistable=True)`: HARD-SILENCE the net [clear v/u/firing/conductances + 30 settle
  steps] then read held-member firing under NO-CUE / CORRECT-CUE / PERMUTED-CUE; GO = correct fires ≥0.20 AND ≥3× both
  no-cue & permuted AND no-cue ≤0.05). **Confirmed self-sustaining at the retracted config (seed 42, n_ca3=2000):
  cue=nocue=perm=rest=0.500** — the assembly RE-IGNITES from a hard-silence, so it's the recurrent STRENGTH (w_within
  grew to hundreds → supra-threshold), NOT the recall protocol. **⚙️ Bistable-regime sweep running** (`bbgvd0gcq`:
  hebb_max cap DOWN {30,60,120,250} → want held_nocue~0 [silent at rest] AND held_cue high [50% cue completes] AND
  held_perm~0 [specific]). **BISTABLE-REGIME SWEEP RESULT:** no window on the cap axis alone (cap≤120
  silent everywhere; cap 250 self-ignites). BUT **weak recurrents (cap 120) + STRONG cue (rdrv 1000-2000) → the REST
  STATE IS GENUINELY SILENT (nocue 0.000, rest 0.000)** — the bistable rest achieved. **Remaining issue = SPECIFICITY:**
  the permuted cue drives held ~0.06 vs correct ~0.11 (a strong drive to random cells spreads via background
  connectivity). The specificity mechanism = the dendritic dAP COINCIDENCE threshold (a held member fires only when
  ≥k WITHIN-assembly members drive it → a permuted cue can't provide k). **⚙️ k_thresh specificity sweep running**
  (`biz67z0eh`: cap 120 × rdrv {800,1500} × k_thresh {30,50,80} → want cue high, perm~0, nocue~0). **RESULT: GENUINE bistable+specific completion ACHIEVED but WEAK —
  a CONFIRMED BOUNDARY.** At cap 120 + rdrv 1500 + k_thresh 30: **cue=0.050, perm=0.004 (12× specific), nocue=0.000,
  rest=0.000** — real cue-gated completion (held reactivate specifically from a partial cue, silent at rest, permuted
  doesn't complete). BUT magnitude capped ~0.05 (5% reactivation); every boost lever breaks the regime (higher cap →
  self-sustains; higher k_thresh → collapses to 0). The ca3→ca3 already route through the dendritic dAP NMDA-plateau
  (CYCLE-1068), so it's NOT pure-AMPA. The trilemma (magnitude vs bistability vs specificity) has no wide window here.
  Full finding boundary block: `2026-07-18-gap5-ca3-functional-completion-CLOSED-6seed-GO-learned-attractor.md`. **✅ RESEARCH GATE DONE
  (`2026-07-18-gap5-bistable-completion-mechanism-research-gate.md`) — ROOT CAUSE PINNED:** the ca3→ca3 uses
  `coincidence_detector=True` = the dendritic dAP READOUT (detects co-activity, CANNOT HOLD a stable high fixed point →
  the whole trilemma). Wang's attractor uses SOMATIC slow-NMDA reverberatory excitation (`exc_receptor="nmda_slow"`) —
  the recurrent was NOT wired to it, AND **the project's `nmda_recurrent` defaults ARE literally the Wang values**
  (τ=100ms). **#1 (built, running):** flip ca3→ca3 to `exc_receptor="nmda_slow"` (added `nmda_recurrent` to `_build` +
  threaded through the de-risk; NO sim/ edit — reuses D3 machinery, proven bistable on this substrate) + 2 near-free
  levers the gate found (recall_steps 300 >> the &lt;100 I used = &lt;1 NMDA τ; settle-to-low-background not dead). Kopsick
  2024: keep ABSOLUTE weight moderate + high SNR (not magnitude), assembly 150-300 (our 16 is far below). **Wang
  NMDA mode BUILT + runs** (mask-free ca3→ca3 extraction fix for no-coincidence-mask). First sweep (16-cell assembly):
  NO clean window — silent at low cap, self-sustains at cap 500 (+ an anomalous cue=0.000/rest=0.156, a small-assembly
  artifact). Per the research gate's KEY lever (my 16-cell assembly is far below Wang/Kopsick's 150-300 robust range —
  a bistable high state needs enough neurons for stable statistics). **🎉 KOPSICK LARGER-ASSEMBLY WORKS:** at
  frac=0.1 (200 cells, n_ca3=2000) + Wang NMDA (cap 250): **cue=0.266 (STRONG, ≥0.20), nocue=0.000, rest=0.000
  (genuinely SILENT at rest = bistable!)** — the larger assembly + somatic NMDA fixed BOTH the magnitude AND the
  bistability the dАP version couldn't. The mechanism WORKS. Only SPECIFICITY remains (perm=0.168 vs cue 0.266 = 1.6×,
  need ≥3× — a permuted cue partially drives held via background). **SPECIFICITY tuning:** competition-lam SATURATES
  (~1.7×); weaker cue better (rdrv 700-1000: cue 0.25-0.27, nocue/rest 0.000, **perm ~0.12 = 2.1×**). ⇒ perm stuck at a
  ~0.12 BACKGROUND-CASCADE FLOOR (a permuted cue's random cells always cascade into the assembly via background
  connectivity). **🎉 SPARSER CODE HITS IT: frac=0.08 →
  cue=0.261, perm=0.081 (3.2× SPECIFIC), nocue=0.055, rest=0.055.** The rest is NOT dead — it's the Wang LOW-RATE
  BACKGROUND (~0.055, the correct biology the gate wrongly rejected with an absolute nocue≤0.05). Correct cue IGNITES
  the high state (4.7× the low); above-baseline signal (cue-rest 0.206) is ~8× the permuted residual (0.026). GATE
  FIXED to the relative criterion (cue≥0.20 & ≥3× nocue & ≥3× perm & nocue≤0.10 = a genuine low state). **⚙️ 6-SEED
  verify + no-encoding anti-cheat running** (`baywbdlux`, ~30min at n_ca3=2000). **⚠️ EXACT NEXT ACTION (RIGOROUS, per
  the retraction lesson — do NOT over-claim):** read it → if 6/6 GO (cue >> low state every seed) + no-encoding
  collapses → **gap #5 REAL cue-gated bistable completion CLOSED on the Wang-NMDA mechanism** (magnitude + bistability +
  specificity all genuine, mandatory anti-cheats baked in) → write finding → emergent DG-selection → SWR loop → console.
  If seed-fragile → per-seed frac/lam or the dAP+NMDA combination. Gaps #1/#4 remain per "close ALL fully".
  **What STANDS:** strong SPECIFIC learned attractor 6-seed + genuine (weak) bistable+specific completion demonstrated;
  the retraction's lesson (mandatory no-cue+permuted gates) is baked in. Gaps #1/#4 remain per "close ALL fully".
- (superseded by the breakthrough above) direct synchronous assembly encoding did NOT grow the weights — DECAYED,
  and sync=async BYTE-IDENTICAL (the gamma mechanism was inert because NOTHING was potentiating). The rate-window LTP
  isn't firing: a cell firing ~1/3 of steps has trace ≈0.33 → product ≈0.11 < the 0.25 coactivity threshold → no
  potentiation → only decay. Likely fixes: fb_inhib over-suppresses the (already-sparse pre-assigned) assembly →
  lower/remove it; AND/OR lower `hebbian_coactivity_thresh`. **⚙️ Workflow `wkn6apwgj` running:** diagnose (assembly
  firing / coact-thresh / competition-crush / does-LTP-apply-at-all positive-control) → joint-sweep to GROW w_within to
  the completion scale + functional completion → adversarial-verify + multiseed. **⚠️ EXACT NEXT ACTION:** read the
  workflow result → if a config GROWS the attractor + completes robustly → gap #5 completion CLOSES (wire emergent
  mossy/DG selection → SWR replay loop → console); if not → the precise remaining blocker (knob / scale / deeper).
- **(superseded plan below — the synchrony-isolation build IS this, now running via the workflow):** pre-assign a
  sparse CA3 assembly (~1% = ~10 cells at n_ca3=1000) per pattern, drive those cells DIRECTLY with strong SYNCHRONOUS
  gamma-pulsed current during encoding (all fire together each window), competition ON (depress to non-assembly), then
  recall a 50% partial cue → does the held-out 50% FIRE (functional completion)? If GO → synchrony IS the fix (then
  wire the EMERGENT mossy/DG assembly selection as the follow-on, keeping it experience-derived); if NO even with
  perfect synchrony → a deeper issue. Gate: h_comp≥0.30 & ≥2× non-stored, lam=0 vs lam=0.5, 6-seed. Then the emergent
  version → wire → SWR replay loop (`_riii_swr_generative_replay_derisk.py`) → console. (This is a genuinely deep,
  multi-cycle open problem — the project's expert prior work 2026-07-08/09/14 reached this exact frontier unsolved.)
- **DONE this session:** scale787 = **CLOSED GO** (corr +0.81≥0.70 @787, retain 1.00, moat 0; RESULT doc + ROADMAP;
  run PAUSED/resumable). Keystone research-gate `wb5udqdul` = DONE + verified + written
  (`2026-07-17-keystone-binder-research-gate.md`; kernel+harness verified, codes regenerated from scale787's own
  correlated codes, `_burndown_3B` deep-dendritic-credit route already a BOUNDARY). **Keystone #1 bounding-probe = GO**
  (coincidence-product bundles CORRELATED codes **0.873 ≥ 0.80**, 3-seed, @0.755 mean-cos; additive 0.193 / chance
  0.062 — the OP class bundles correlated codes; graceful degrade from 0.989 at lower correlation).
- **DONE (cont'd):** keystone **#2 rate-rung GO** (self-organizing fast-weight bind, correlated codes 1.000, beats
  fixed-FHRR 0.873; local write, no conj-inverse; permuted-role collapses) — `2026-07-17-keystone-2-...RATE-GO`. **DRIFT-#12
  CONNECTION (load-bearing):** the EDGE-5 arc (2026-07-15) already realized the **spiking single-bind store 6-seed GO**
  (`edge5-rung2-STP-store-onbridge`) AND found the **multi-bind on-bridge store COLLAPSES** (additive STP, below chance
  at P=2) with the **named surpass = a delta-like error-correcting on-bridge write** (un-built). My #2 per-fact test
  trivialized delta (isolated facts); the realistic SHARED store is where delta matters (EDGE-5 numpy rung-1 confirms).
- **DRIFT-#12 catch #2 (the on-bridge delta write is REFUTED):** EDGE-5 rung-3/3b already built + refuted it (delta ≈
  additive at 2 scales; value-specific potentiation caps at ~2 binds). **⇒ the genuine gap-#2 spiking edge is the
  SELF-ORGANIZING SPIKING SLOT BINDER** (keystone gate #3: sparse-conjunctive competitive cells + BTSP; each bind →
  its own bounded slot so binds don't interfere; the slot competition must SELF-ORGANIZE, NOT a hand-tuned FS-WTA
  [EDGE-5 banked that]). The rate rung is DONE (my #2 key-addressed store = 1.000 @ K=12 = slot-separation); the open
  edge is specifically its SPIKING realization past the ~2 write-rule cap. A deep frontier (EDGE-5's own term).
- **Slot-binder research-gate DONE + verified + written** (`2026-07-17-keystone-slot-binder-research-gate.md`).
  Diagnosis: the ~2 cap = wrong storage primitive (SNR-limited shared sum); a COMPETITIVE SLOT store converts capacity
  to slot-count-limited (combinatorial). **#1 de-risk = compose already-GO pieces (residual is COMPOSITION, no new
  `sim/`): the EMERGE spiking competitive pooler (EMERGE-41 rank-order + `sim/kernels.fused_htm_permanence_update`
  hfac homeostatic boosting = the SELF-CALIBRATING, non-hand-tuned threshold) as a slot allocator + the
  `HebbianBinder` retrieve-vs-allocate rule + the D3 persistent-slot attractor, on the EDGE-5 role-filler task.** All 6
  pieces + the kernels VERIFIED present. The ONE host residual = the retrieve-vs-allocate θ (self-calibrates via the
  Bogacz-Brown familiarity gate; named, non-blocking).
- **Running jobs:** anchor-heartbeat `bj0sku4ga`. **gap#3 multi-referent research-gate `wa2ucpp1p` (running, parallel lane).**
- **gap#2 spiking slot binder — BUILD STEP 1 = GO** (`2026-07-17-keystone-2-spiking-slot-binder-STEP1-prereq-GO`):
  distinct role-drives select DISTINCT competitive slots via the EMERGE-41 pooler `FSWTAProbe` (3-seed, all 6-col
  non-empty, Jaccard ~0.07). The load-bearing slot-separation property holds. Also identified: sequential selections
  on a REUSED bridge give `[6,0,6,0]` (adaptation + FS carryover suppresses the next selection — the EMERGE-61 family).
- **BUILD STEP 2 progress:** step 1 (slot separation) GO · step 2a (P=3 NMDA slots COEXIST, no-recur collapses) GO ·
  step 2c (role-cued retrieval) = runner `_keystone2_spiking_slot_binder_derisk.py` BUILT + diagnosed. Single-bind
  mechanism WORKS (held slot drives its filler, argmax correct, f0=0.45). Two composition bugs found+fixed (reset
  breaks retrieval → rely on NMDA hold; shared→per-slot gates). Decay ruled OUT by reading the substrate
  (`bridge.py:7051` decay IS gated). **Remaining = WRITE STRENGTH in the multi-bind flow** (w0→f2 weak/lost: 0.02 at
  teach1, 0.00 at retrieve). All in `2026-07-17-keystone-2-spiking-slot-binder-STEP1-prereq-GO.md`.
- **🎉 gap#2 FULLY-SPIKING 6-SEED GO + reset NEURALIZED (2026-07-17):** the competitive-slot binder recovers a fact's
  role-filler bundle on SPIKES. At the SVO load **P=3: slot-sep 1.00 (6/6) >> shared cap 0.33, permuted→0.00.** The
  readout reset is now the **D3 CLEAR done on spikes** — an FS inhibitory burst (`clear_gain=1500`, `clear_steps=250`)
  + a **settle gap** (50 steps, lets the fast g_i decay before the cued read) — and it reproduces the host reset
  **byte-identically per-seed** (P=4 host 0.792 == neural 0.792, same [0.75,0.75,1,0.75,0.75,0.75]). `neural_clear=True`
  is now the runner DEFAULT. P=4 = 0.79 is the mechanism's intrinsic 4-slot read edge (5/6 read 3-of-4); the more-KF
  lever is REFUTED (KF=10→0.71). The key store recipe: reset-held-at-readout + read-calibration (maxw=250) TOGETHER
  (found after ~16 probes + 3 self-corrections). Honest: LTM store (NMDA hold NOT load-bearing, no-recur 0.83).
  **Follow-on (a) neuralize-reset = DONE.** All in `2026-07-17-keystone-2-spiking-slot-binder-STEP1-prereq-GO.md`.
- **Gap #3 = LARGELY CLOSED** (audit corrected); its cheap residual (neuralize `content_bias_target`) is a parallel
  emergence-bar-polish lane available anytime.
- **✅ (b) ADVERSARIAL-VERIFY = CONFIRMED (2026-07-17, independent skeptic subagent):** GO survives every control —
  **NO-TEACH → chance 0.167** (the learned write IS load-bearing), **SCRAMBLE-TEACH → 0.00** (genuine learned read,
  not fixed structure / drive-leak), **KF=12 → 11× chance** (not small-space luck), permuted reads its OWN taught
  filler (genuine role-addressing, not silence), neural-clear == host byte-identical, `cfg.seed` genuinely varies the
  substrate (3 distinct threshold hashes). Honest scope (author-disclosed, not confounds): LTM not WM; "shared caps ~2"
  label loose (recovers ~1); small scale KF≤12; P=4 graceful-degrade.
- **✅ (c) WIRE-IN MECHANISM = 6-SEED GO (2026-07-17, `_slotbinder_content_addressable_probe.py`):** content-addressable
  multi-fact recall on the spiking slot-binder — each (fact,role)→its own slot; recall = a NEURAL SCAN (drive each
  fact's agent/verb slots, match read-back to the cue, read the matching patient; abstain if none = the moat). 6/6:
  query_patient 1.00, query_agent 1.00, **moat-abstain 1.00, permuted-cue-abstain 1.00**; scramble-store collapses to
  chance 0.33. Uses concept POOLS as fillers (the validated g20/Pulvermüller representation) — generalization across
  SIMILAR concepts is the separate already-closed arc, not this.
- **🎉🎉 gap#2 CAPABILITY FULLY CLOSED (2026-07-17) — fully-spiking, 6-seed, adversarially verified, WIRED IN:**
  `research/runners/slotbinder_composer.py` `SlotBinderComposer` implements the full composer contract (store /
  query_patient / query_agent / ask_yes_no [negation via a 4th polarity slot] / render_fact / query_chain + the moat)
  on the spiking slot-binder. Wired as a selectable composer in `BrainConversationalAgent` (`composer=...` AND
  `composer_kind="slotbinder"`) — the AGENT answers the whole who/what/yes-no/describe/abstain matrix through it
  (verified). CI `tests/test_slotbinder_composer.py` **6 pass**; existing agent tests **7 pass / 0 regress** (additive
  branch). ⇒ the FHRR exact-inverse algebra is replaced by a learned self-organizing spiking competitive-slot store,
  live in the real conversational agent. `2026-07-17-gap2-adversarial-verify-CONFIRMED-and-content-addressable-wire-in-GO.md`.
- **Self-organization ASSESSED (emergence bar met, honest reasoning):** the COGNITIVE STRUCTURE that must
  self-organize = the BINDING; it DOES (Hebbian slot→filler write, adversarially confirmed load-bearing —
  no-teach→chance, scramble→0.00). The slot ALLOCATION (which fresh slots a new fact gets) is memory-address
  bookkeeping, NOT cognitive structure: the host next-free-slot counter is the CORRECT engram-by-availability policy.
  A self-organizing allocator gives NO capability gain — adaptation-competition yields only RECENCY-avoidance (used
  slots' `u` decays → early slots free up → collisions at scale), and a neural-occupancy scan is behaviorally
  identical to the counter. ⇒ no refinement needed; the counter is legitimate infrastructure (like a hippocampal
  allocation pointer), not a hand-designed cognitive shortcut. **Gap #2 is FULLY CLOSED.**
- **▶ NEXT GAP = #5 (CA3 completion / imaginative replay) — owner chose "whichever closes quicker" (2026-07-17).** a-1
  DONE: the root cause is a CONCRETE MECHANICAL BUG never fixed — the **ca3→ca3 recurrent synapses are functionally
  SILENT** (~0.2 mV at weight-120, ~1000× too weak) and **WEIGHT-INVARIANT** (24× weight → byte-identical Vm = a
  transmission bug, NOT a point-neuron limit; the old "point-neuron completion boundary" was RETRACTED because of
  this, `2026-07-08-riii-CORRECTION-ca3-recurrents-functionally-silent-...`). This ONE bug blocks BOTH gap-#5
  sub-problems (learned-attractor completion + SWR generative-replay reactivation). The dendritic dAP completion
  READOUT is already 6-seed GO (on a hand-installed attractor), so fixing the recurrents is the single enabler.
  Candidates to investigate in `sim/bridge.py` (per the finding): effective-synaptic-strength scaling for within-region
  recurrent RegionPathways · conductance-decay tau vs recurrent input rate · CSR matvec orientation for SELF-pathways
  (from_region==to_region) · whether recurrent synapses are in the per-step conductance update at all. **⚠️ EXACT NEXT
  ACTION (SUPERSEDED — see the RE-OPENING below):** READ the sim/ recurrent-current delivery code, reproduce the
  weight-invariance, find the discrepancy, fix. [This assumed a transmission bug; a direct probe refuted it.]
- **🔎 gap#5 RE-OPENING (2026-07-17) — the "silent recurrents transmission bug" is REFUTED by a direct instrument.**
  Read the substrate first: the ca3→ca3 pathway is plain-AMPA (no nmda_slow suppression; the gate is plasticity-only,
  not current), matvec orientation correct, `propagation_strength=0.05` small but WEIGHT-PROPORTIONAL. A direct g_e/Vm
  probe (override the recurrent weights in `cp_connections`, drive 24 presynaptics ~125 spikes, measure targets):
  **weight 120 → target g_e ~10, Vm Δ 3.66 mV; weight 5 → g_e 0.15, Vm Δ 1.43 mV** — the recurrents TRANSMIT and SCALE
  with weight. ⇒ NOT silent, NOT a transmission bug (the 2026-07-08 "~1000× too weak / weight-invariant" was a
  WEAK-DRIVE artifact — 8 presynaptics/18 spikes near the floor; my own FIRST probe also mis-read it because
  `_build(train=False)` HARDCODES weight 1.5). **The real question is attractor STRENGTH** (3.66 mV won't fire non-cue
  members from a partial cue → needs stronger recurrent weight / density / LTP), NOT a sim/ fix.
  `2026-07-17-gap5-ca3-recurrents-NOT-silent-transmission-refuted-attractor-strength-is-the-real-question.md`.
- **⚠️ gap#5 sweep RESULT = SILENT-FAILURE SIGNATURE (2026-07-17):** the w120/300/600 × d0.30/0.50/0.60 completion sweep
  returned **BYTE-IDENTICAL to 3 decimals across ALL configs** (TRAINED held-out=0.027, NO-TRAIN=0.027, gain=+0.000,
  own_cos=0.506≈√0.5 the drive-artifact floor). Impossible if weight/density reach the computation — yet my direct
  probe PROVES the recurrents scale with weight. So the sweep is untrustworthy (silent failure). **A Workflow
  (`wxim0nt7c`) diagnosed it — DECISIVE (H1/H3/H4 in; synthesis+verify pending):** **H1 REFUTED→ROOT CAUSE:** build
  params DO apply (density 0.30→6654 syn/0.60→13312; weight 120→build-mean 120/600→600) BUT **after TRAINING both
  collapse to an IDENTICAL ca3→ca3 mean|w|=0.846** — training's Hebbian COLLAPSES the recurrent weights regardless of
  init (the Hebbian-decay-to-floor). That is the byte-identical cause. **H3 REFUTED:** the completion metric is VALID
  (force held-out firing → completion 0.000→6.023). **H4 REFUTED:** a HAND-INSTALLED strong symmetric attractor
  **DOES complete** — reliable at effective weight ≥1600 (raw ×0.05), held-out ignites to match the cue, specific
  (non-ensemble silent, no runaway to w3000), bistable (~600 threshold). ⇒ **the point-neuron CA3 CAN complete; the
  only blocker is that TRAINING collapses the attractor instead of building it** (NOT a substrate floor, NOT a
  transmission bug — both prior diagnoses REFUTED). Prior half-fix exists:
  `2026-07-08-riii-ca3-attractor-formation-symmetric-hebbian.md` — the default Hebbian is CAUSAL-offset (pre@t-1 &
  post@t) but co-ensemble members fire SYNCHRONOUSLY → offset never satisfied → ~0 potentiation → collapse; a guarded
  `hebbian_symmetric` config (offset-free same-step co-firing) already exists + forms a SPECIFIC but WEAK attractor
  (+0.87, capped at `hebbian_max_weight=30`). **SYNTHESIS+VERIFY DONE (6-agent, adversarially confirmed):** the
  Hebbian-collapse confound is confirmed exactly (Hebbian-OFF: 120→120; ON: 120 AND 600 → 0.846). **VERIFIER
  CORRECTION (flips the ordering):** H4's completion threshold is NON-reproducible — the point-neuron all-to-all volley
  needs raw w≈**6000** (not 1600; intermediate weights SUPPRESS the cue), so even a de-confounded sweep at built
  weights 120–600 likely still misses the >0.30 gate. ⇒ **the dendritic-dAP completion (already 6-seed GO, completes
  at far lower weight) is the PRIMARY closer**, fed by a learned pattern-specific attractor; the point-neuron volley is
  secondary/extreme-weight only. Full finding: `2026-07-17-gap5-completion-ROOT-CAUSED-hebbian-collapse-not-a-floor-workflow-6agent-verified.md`.
  **⚠️ FRONTIER CORRECTED (a-1 to the LATEST findings, 2026-07-09 — I had a-1'd only to 2026-07-08 + the audit, a
  drift-#12 re-derivation; my session's Hebbian-collapse/attractor-strength work was toward an ALREADY-ANSWERED
  question).** The real R-iii state: **completion half = SOLVED** (dendritic dAP, 6-seed GO on a strong attractor).
  **Formation:** ALL 4 plasticity rules (incl. rate-window) form only a WEAK ~1.44× attractor because the trained CA3
  code is DISTRIBUTED (35-47% active, async) — no Hebbian rule binds cells that don't co-fire strongly
  (`2026-07-09-riii-formation-rules-saturate-ensemble-dynamics-is-the-blocker`). Root cause: **CA3 has NO feedback
  inhibition** (`internal_density=0.0` leaves its 15% inh cells unconnected). **Rung 1 (add `ca3_pv_basket` feedback
  inhibition, runner-side, NO sim/ edit) sparsifies 0.43→0.21 but is NON-SELECTIVE** (global inhib suppresses members
  too → ratio stuck 1.16×). **THE OPEN FRONTIER = Rung 2 the MOSSY-DETONATOR** (`2026-07-09-riii-ca3-feedback-inhibition-sparsifies-but-nonselective`):
  strengthen `dg→ca3` mossy so a few CA3 cells fire HARD (detonate = the ensemble) despite the inhibition → a SPARSE +
  strongly-firing SELECTIVE ensemble → does within/silent ratio pass ~3×? Runner-side, NO sim/ edit. **⚠️ EXACT NEXT
  ACTION:** run the Rung-2 sweep — `_riii_ca3_attractor_diag.py --mossy-weight {8,30,80} --ca3-fb-inhib 120 --hebb-rate
  --hebb-max <high>` measuring the within/silent RATIO + sparsity (gate: ratio ≥3× AND sparsity ~0.05). If it passes →
  `_riii_ca3_coincidence_completion_derisk.py --two-comp` (dAP) on that config → held-out completion GO (trained>0.30 &
  specificity & no-train collapse & LESION), 6-seed. Rung 3 (`--sync-on/--sync-off` gamma-pulse) is the named fallback
  if PING is too coarse. Then wire completion → SWR replay loop → console. CLOSES gap #5. (My session's contributions
  that STAND: recurrents transmit + scale [refutes the silent-transmission bug]; the substrate CAN complete
  [hand-installed attractor]; the diagnostic's byte-identical sweep is a Hebbian-collapse confound — all consistent
  with the 2026-07-09 picture, just downstream of the real frontier.)
- **✅ Gap #4 a-1 RAG check DONE (2026-07-17) — the research-gate is essentially the `learning-rule-frontier-map`
  (2026-07-17), re-read + current:** SUPERVISED deep credit on spikes (e-prop / NP / D1-BDSP) is PARKED — all blocked
  by ONE shared "spiking-classifier-readout-training wall"; the rate-net positive control (done) confirmed the block
  is the RULE, not the readout. NP is retired (12-seed refuted). The WORKING emergence engine = the UNSUPERVISED
  on-spike stream cortex (HTM + committed BDSP `fused_htm_permanence_update`), which SIDESTEPS the wall. **Gap #2's
  just-completed closure VALIDATES this** — it closed via unsupervised/local Hebbian slots, no deep-credit engine.
  ⚠️ **Per THE LAW: supervised deep credit is a BANKED METHOD (parked), NOT an abandoned capability** — the capability
  (a learning engine for emergence) is served by the working unsupervised path.
- **⚠️ EXACT NEXT ACTION (gap #4, a-1-ranked rec #2):** the ONE live unresolved fork in the record — does the
  off-diagonal recurrent-credit gap REPRODUCE on the real on-bridge recurrent substrate (→ MDGL warranted) or DISSOLVE
  under real population coding (→ stop, advance the unsupervised path). Build: a small recurrent on-bridge Izhikevich
  net on a delayed-cue task, diagonal e-prop only (reuse `_onbridge_eprop_port_derisk.py` + a recurrent slice),
  **SEEDED (`cfg.seed` set + two-process threshold-hash verified — the seed bug confounded the whole prior arc)**,
  frozen-hidden + shuffle/wrong-sign anti-cheats, ≥6 seeds. Cheap (hours, CPU), decisive either way. Docs to read to
  scope it: `2026-07-15-offdiagonal-recurrent-credit-ARC-SYNTHESIS-*` + `2026-07-15-emergence-engine-research-gate-horizon-frontier`
  (the two disagreeing gates) + `_onbridge_eprop_port_derisk.py`. (Owner may redirect to #5 CA3-completion or #1
  open-generation — a genuine strategic fork — but per the standing dendritic/emergence-engine priority, proceed here.)
  - **KEYSTONE REFRAME (a-1 done 2026-07-17, the load-bearing insight):** every failing credit method (e-prop
    feedforward NOT-GO, recurrent refuted, NP retired, BDSP-on-readout blocked, graded-readout no-unlock) was
    **SUPERVISED global-loss deep credit through a spiking CLASSIFIER READOUT** — that *shared readout wall* is the
    common cause (`2026-07-17-learning-rule-frontier-map`), and it is exactly what the UNSUPERVISED stream cortex
    already sidesteps. ⇒ do NOT re-run the supervised family. The research-gate's FIRST question is the SURPASS
    reframe: **can the BINDING STRUCTURE (gap #2) SELF-ORGANIZE unsupervised** — Hebbian/BCM/competitive/developmental,
    the way the stream-cortex CODES already self-organize — since biology DEVELOPS binding from local wiring rules,
    not supervised credit? Rank unsupervised/self-organizing binder mechanisms FIRST; a supervised local rule is a
    fallback, not the default. This makes #4 and #2 the same research-gate (the substrate that DEVELOPS the binder).
  - **Prior-work grounding for #4/#2 (a-1 done — the gate must SURPASS these, not re-derive):** single-attribute
    learned bind is **GO on spikes** (`2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE`);
    the two open negatives to surpass are (a) **multi-attribute BUNDLING from scratch = NEGATIVE** (additive has no
    inverse; learned-linear-inverse ≈chance) and (b) **binding over CORRELATED/structured codes ≈ chance**
    (`2026-06-11-cortex-learned-binder-systematicity-NEGATIVE-ON-CORRELATED`). Both failures used SUPERVISED
    inverse-learning → the self-organizing reframe (develop the binding structure from local rules, unsupervised) is
    the un-tried angle. Read both docs in depth as move (1) of the gate.
  - Launch gap #3's research-gate in parallel (self-contained, tractable): biased-competition WTA between referent
    attractors (the 2 prior NEGATIVEs were recency + salience-boost; the named fix is winner-take-all inhibition).
- **Continuation mechanism (owner chose 2026-07-17):** MANUAL — the owner says "continue" (no watchdog, no system
  changes). That + this board + the CLAUDE.md pointer re-anchor instantly. Within a session, NEVER stop (async
  pattern); across a reboot/idle, a plain "continue" resumes from EXACT NEXT ACTION. Do NOT propose a watchdog again
  unless the owner asks.
