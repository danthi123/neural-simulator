# Roadmap from here to the primary goal

*2026-09-29, Claude, at the owner's request. Phase structure follows the master roadmap's six developmental stages
([2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md](2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md)); current phase plan:
[2026-09-25-prove-who-owns-the-computation-PLAN.md](2026-09-25-prove-who-owns-the-computation-PLAN.md); operating
model: [2026-09-29-local-model-transition-PLAN.md](2026-09-29-local-model-transition-PLAN.md). Calendar windows are
low-confidence estimates (section 4).*

## 1. The goal and how we will know

A simulated brain that genuinely converses: it reasons to its own conclusions, feels in context, notices and seeks
what it does not know, and reports honestly on its own state, all from simulated neurons and synapses, not from
looking facts up and not from a language model producing plausible text. Qwen2.5-0.5B stays as the mouth that words
what the brain decides. The bet: consciousness emerges from a complete and faithful emulation.

Reached when all three hold:
1. **Behaviour bar** (master roadmap): Stage 4 exit (held-out multi-turn dialogue: own insights, seeks what it
   lacks, contextual affect, honest self-report) and Stage 5 exit (keeps learning from real people with the AI
   teacher retired).
2. **Ownership bar** (owner 2026-09-25): in that dialogue the brain is shown causally to have produced the content:
   high reply-level load-bearing fraction, low host-decision share on content and commitment by intervention.
3. **Honesty boundary:** every measurable functional correlate of consciousness measured; self-reports are honest
   functional read-outs; phenomenal experience is never claimed.

## 2. Where we are (first commit 2025-11-29)

- Ledger: 69 faculties tracked, 30 spiking and on by default in live chat, 5 host scaffolds fully retired;
  ~3,000 findings. Many Stage 1-3 abilities pass in isolation.
- No developmental stage-exit gate has been passed by the integrated brain. Stage 4 not met; Stage 5 not started.
- #1 metric at two levels (B2b): decision fields 0.843 (core 28); reply level 0.535 (core 17). Part of the gap is
  instrument blind spots (no Qwen mouth in the battery; probes that never reach the reply).
- Host still decides: which stored fact answers (first-match loop), speak/silent (salience), workspace winner
  (argmax), parts of the no-confabulation check and templates; part of the "store" is a host list.
- Open: prioritized memory; learning over time; learned (not retrieved) knowledge; deep credit; host relays between
  organs.
- Measurement debt found 2026-09-29: three organs ignore the seed; a replay noise setting never applied; a crashing
  lesion arm; the metric overstated by counting decision fields.

## 3. The path

### Phase A: prove who owns the computation (now to ~late Oct 2026)
Host-share baselines (6 seams: decoding, substitution, intervention); the first live seam retired (speak/silent +
workspace argmax, default-off flags); delayed credit on the production basal-ganglia organ; sleep transitive
inference inside the prioritized-memory prereg; B2d reply-level battery at HEAD with the Qwen mouth arm; instrument
fixes (probe remaps, seeding). **Exit:** six-seed host-share baseline for content and commitment; one live seam
retired with host share before/after; reply-level fraction at HEAD with the mouth.

### Phase B: hand the decisions to the brain (~Nov 2026 to Jan 2027)
Seam by seam in order of host share: content selection first (spiking competition in the store replaces the host
first-match), then commitment, routing, candidate generation (the host never pre-filters candidates). Each behind a
default-off flag, flipped after six-seed answer-preserving + load-bearing checks. Consolidate remaining host relays
that carry decisions. **Exit (to preregister):** host-decision share on content+commitment below ~0.2 by
intervention; reply-level load-bearing fraction at or above ~0.8 with the mouth; no battery regressions.

### Phase C: memory that matters, learning over time (overlaps B; ~Nov 2026 to Mar 2027)
Affect/DA/surprise-weighted encoding; replay that favours what matters and protects premises; knowledge store
learned from dialogue and the teacher with lossy capacity; the teacher develop loop at the learning edge; learned
value from Phase A's delayed credit. **Exit:** Stage 1-2 exit gates in the integrated brain (lived-fact recall,
valence inheritance r >= 0.55, curiosity ask-rate >= 2x with learning progress, held-out inference) and the
"remember what matters" gate with a replay-lesion control.

### Phase D: reasoning to its own conclusions, self-model (~Q1-Q2 2027)
Workspace deliberation in live chat (3-hop never-told conclusions), self-report tracking ground-truth confidence
(meta-d'), false-belief ToM, causal forward model, internalized deep credit. **Exit:** Stage 3 exit gates in the
integrated brain, low host share on those seams.

### Phase E: conversant, the Stage 4 bar (~mid to late 2027)
Flagship console; affect shaping speech through state; recursive ToM and affective empathy; autobiographical
continuity; counterfactuals; the consciousness-correlate dashboard. **Exit:** the held-out multi-turn dialogue,
preregistered, brain-owned by intervention.

### Phase F: human-ready, the Stage 5 bar (2028+)
Teacher retired; grows from real human conversation; lifelong learning; remaining scaffolds biologized. **Exit:**
Stage 5 exit.

## 4. Timeline summary

| Phase | Window (estimate) | Confidence | Main dependency |
|---|---|---|---|
| A Ownership measured | now to late Oct 2026 | Medium-high | Pool throughput; local-model execution |
| B Decisions to the brain | Nov 2026 to Jan 2027 | Medium | Content-seam design; Claude inspection budget |
| C Memory and learning | Nov 2026 to Mar 2027 | Medium-low | Prioritized replay; capacity on one 3090 |
| D Reasoning and self-model | Q1 to Q2 2027 | Low | Deep credit (historical wall); integration |
| E Conversant (Stage 4) | mid to late 2027 | Low | Everything composing in one brain |
| F Human-ready (Stage 5) | 2028+ | Very low | Real-human learning at scale |

Basis: 10 months to reach mostly de-risked, not-yet-integrated faculties; design work slows an estimated 2-3x once
the local model drives and Claude inspects weekly; execution continues around the clock on the pool.

## 5. Cross-cutting workstreams
Measurement integrity (the instrument is part of the emulation); necessity by six-seed ablation; ownership tracking
(host share and reply-level fraction reported together monthly); the honesty boundary; infrastructure (containerized
harness, pool, digest, gates for the local model's failure modes).

## 6. Operating model
Daily: the local model in the server container executes packets, scores with registered scorers, syncs board/docs,
drafts builds, raises NEEDS-CLAUDE. Weekly: one Claude Pro inspection reads the digest, rules, merges, adds gates for
new failure modes, writes the next 2-3 packets. Monthly: owner milestone review (the two ownership numbers +
stage-exit checklist; the next seam order).

## 7. Risks
1. Local-model mistakes -> guard hooks, sandboxed jobs, provisional verdicts, weekly inspection.
2. Claude budget on Pro -> a packet backlog built now; digest-driven inspections.
3. One 3090 (24 GB) -> numpy on the pool; reference-hardware rule; AWS within the $50/day cap when justified.
4. Measurement debt -> Phase A; every number at both levels.
5. Emergence may not appear -> falsifiable bet; failed gates are findings; no-defer means the next method.
6. Permanent-mouth concession (open-prose confabulation) -> brain-owned content; generator-agnostic honesty checks.

## 8. Next four weeks

| Week | Research (pool) | Transition / Claude |
|---|---|---|
| Sep 29 | Packets land; host-share, selection-seam, credit sweep, sleep gate start | Serving switch; container secured and started; runbook; Friday dry run |
| Oct 6 (Pro) | Results arrive; local model scores them | First weekly inspection; B2d mouth arm; content-seam packet |
| Oct 13 | B2d at HEAD; credit gate with frozen settings | Rule on steps 3/4/5; memory-method packet |
| Oct 20 | Phase A exit check | Phase B kickoff: content selection into the brain |
