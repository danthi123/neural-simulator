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
| 2 | **Learned binder over the brain's OWN structured/correlated codes** | fluid reasoning/composition over the brain's own semantics; today comprehension/composition ride a hand-designed exact-inverse FHRR algebra (fails the emergence bar) | multi-attr bundling from scratch NEGATIVE (additive 0.193 / learned-linear-inverse 0.056 ≈ chance); learned binder ≈chance on correlated codes | OPEN — single-attr learned bind is on-spike GO; bundling + structured-code binding are not |
| 3 | **Multi-referent disambiguation** | real dialogue holds several entities; bind a bare pronoun to the salient one | recency NEGATIVE; salience-boost NEGATIVE (2 converging negatives) | OPEN — named fix (biased-competition WTA / attention-salience pointer) specified but UNBUILT |
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

- **Phase:** finishing the pre-pivot confirmation run + read-only keystone research-gate prep (the anchor-heartbeat
  correctly caught idle-waiting → doing the read-only gate prep, which is NOT the build-pivot the owner sequenced).
- **Running jobs:**
  - `scale787` develop loop (787-concept "given enough training" test) — GPU, monitor `bois5ocpf` (gate-anomaly +
    heartbeat + done/crash). Root `bridges/developed/scale787`, log `scale787_full.log`. Day 24/33, 600 concepts,
    still GO (corr +0.83 ≥ 0.70 gate, retain 1.00, moat 0). PAUSE sentinel: `bridges/developed/scale787/PAUSE`.
  - anchor-heartbeat `bj0sku4ga` (~25min self-check) · (audit workflow DONE).
  - **keystone research-gate workflow `wb5udqdul`** (read-only: our-record · biology · external-lit · correlated-code
    angle → ranked self-organizing dendritic-bind de-risk). Running.
- **⚠️ EXACT NEXT ACTION (updated):** (1) when `wb5udqdul` completes → review its ranked de-risk, TRUST-BUT-VERIFY the
  load-bearing claims (esp. "the D2 dendritic neuron can multiply" + "method X tolerates correlated codes"), write the
  keystone gate doc `research/findings/2026-07-17-keystone-binder-research-gate.md`. (2) when `scale787` fires TERMINAL
  → `_scale787_analyze` vs frozen gate → RESULT doc + reconcile ROADMAP §5.10. (3) THEN start the #1 cheap-first
  de-risk from the gate (the build-pivot the owner sequenced after the run) — rate→spike→one-brain ladder, anti-cheats,
  6-seed, `cfg.seed` set. Launch gap #3's research-gate (multi-referent WTA) in parallel.
- **Gap order (planned):** #4 keystone research-gate FIRST (unlocks #2/#5/#1) · #3 multi-referent parallel tractable
  lane · then #2/#5 on the keystone · then #1.
- **⚠️ EXACT NEXT ACTION:** when `scale787` fires TERMINAL → run `_scale787_analyze` vs the FROZEN gate → write the
  RESULT doc + reconcile ROADMAP §5.10 → **then START gap #4's research-gate.**
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
