# Pivot 2026-11-02: a functional digital brain that speaks for itself (DECISION + strategy memo)

**Owner, 2026-11-02 (verbatim):** "I want a simulated brain I can fluently speak with, not just have short back and forths with template based responses picking which facts to recollect. It must be open ended like an LLM or an actual person, even if it's a very young person. And it must be able to learn and grow. We're not looking to recreating the classic LLM + RAG combo just out of spiking neurons, we're looking to create a functional, digital brain."

**Owner rulings (2026-11-02):**

1. **Words.** The brain's own language cortex produces the words, fed by the brain's state through real connections. Qwen becomes teacher and fallback. This reverses the mouth decision of `docs/plans/2026-09-19-roadmap-with-a-permanent-llm-mouth.md` (its load-bearing metric stays).
2. **Realism trade (declared, per the 09-18 per-mechanism rule).** Gradient pretraining of the language cortex stands in for evolution and childhood; everything learned after "birth" is learned in conversation with local, brain-style rules. Disclosed in the honesty read-outs; reversible if a local-rule method reaches the same quality.
3. **Consolidation.** A new, small core is built alongside the old code (substrate, regions, learning rules, language cortex, chat loop), with an architecture document short enough to read in full every session. The old code is frozen as a library of proven mechanisms and a baseline; a mechanism is ported only when it earns its place. First gate: an end-to-end conversation with the native cortex and memory that survives a restart, within about two weeks.
4. **In flight.** The Stage-2 retake finishes and is scored (due 11-21). All other old-track work is frozen (RULING `pivot-1102` in the runbook); the C2 drain-wta lines were withdrawn.

**What this retires:** seam-by-seam retirement on separate hand-wired organs as the unit of progress; Qwen as the permanent mouth; new faculties on their own bridges; the fact-triple-to-Qwen reply path as the target design; per-line runbook bookkeeping on main (to be moved off main with the new core).

**Not retired:** six seeds before a generalization claim, the honesty boundary, the gates (re-focused on the new core), the Stage calendar.

The strategy memo below (Fable synthesis of four audits, revised after an adversarial critique) is the evidence for this decision; its section 5 decisions are answered by the rulings above. The new core's design (architecture doc and two-week plan) follows in its own document.

---

# Strategy memo — 2026-11-02

Legend: [M] measured in repo with citation · [I] my inference · [E] external, checked.

## 1. Where the project honestly stands

**Single shared substrate: not there.** A production chat runs ~31 spiking organ types on 26-47 physically separate bridge instances per conversation, plus 78,857 separate 129-neuron LTM bridges (~5.4 GiB) for the imported wikidata store [M: research/findings/2026-10-26-membrane-census-dev7.md:16; research/FAILURE_LOG.md:412]. "One brain" today means a 7,002-neuron pool holding 11 frozen circuits beside a 54,320-neuron composer bridge (45,856 of them RF phasors). Organs exchange values through host Python, not synapses. Host-decision share by intervention: 0.958 on measured seams, 0.844 on call-graph seams [M: 2026-10-02-host-share-baselines-RESULT.md]. The July "composer + WKV on one bridge" capstone passed with a cross-interaction of exactly 0.0, i.e. co-residence, not integration [M: 2026-07-20-single-shared-substrate-CAPSTONE…:35-40].

**Grows and learns over time: not there.** Through the real /api/brain-chat, the shared pool and the affect bridge show max|dw| = 0.0 on every turn; 278,364 synapses are gain-0 frozen by design [M: 2026-09-24-chat-time-plasticity-audit §3a, :183]. Weights move only in ~6 small stores, and the biggest (the fact store) is placed by host code with a host `.kb` list beside it that measurably decides chained answers (C2 scramble, 1/1; spiking read lesion changes nothing 7/7) [M: one_brain_composer.py:735-753, 769-771; 2026-10-26-c2-kb-scramble-host-ownership.md:134-147]. D5 BTSP write is deferred on numpy (~510 s/topic) [M: webapp/server.py:3622-3633]. BRAIN_PERSIST_LEARNING and BRAIN_ONE_BRAIN are both built and default OFF, so nothing survives a restart and every conversation gets its own brain [M: cross_session_persistence.py:68; webapp/one_brain.py:49]. LBF reply-level at HEAD: 22/41 = 0.537 (stub mouth), with d5-consolidate, sleep-replay and da-gated-encoding at 0/6 [M: 2026-11-02-lbf-at-head-gate-seeds-RESULT.md §1-2].

**Converse at small-LLM level: met by the system, not by the brain.** Every English word comes from Qwen-0.5B (or a template stub without CUDA) wording a brain-chosen (a, v, p) triple [M: brain_chat_tui.py:887-910; _grounded_lang_integration_derisk.py:182-186]. The best deployable brain-native LM beats a trigram by +0.05 nats at deep context on SimpleWiki and loses to it at depth 2 (-0.49) and on WT103 (-0.29 to -0.57) [M: 2026-09-03-OPEN-FLUENCY…; 2026-09-04-fluency-scale-wt103…]. The 267M BPTT-trained WKV (val ppl 45.66, exact phasor-forward port, 6 seeds) is near SpikeGPT territory [M: 2026-08-02-gap1-wkv-width-ladder; E: SpikeGPT-216M WT103 ppl 39.75] but is quarantined as a scaffold and never wired into a turn. The 09-19 roadmap marks a spiking mouth "do not re-attempt" and no lane pursues it. **This conflicts with your 11-02 goal and needs your ruling (Decision 1).**

## 2. Why progress is slow (ranked by evidence)

1. **The path was not aimed at the goal.** In 8 weeks sim/ grew +1,612 lines, all default OFF, versus +110,841 in research/runners/; three language arcs (moat, word-role, cs-race) consumed ~490 commits and produced zero flips [M: velocity audit]. Seam-by-seam retirement on separate hand-wired organs does not converge on one plastic substrate, and nothing in the velocity data touches brain-native language.
2. **Coordination overhead.** 35% of non-merge commits are runbook/queue-only (958 "runbook:" commits); since the 10-01 handover coordination outnumbers code 2-9x weekly; ~60 of 72 NEEDS STOPs were sync/provisioner/gate blocks, not verdicts [M].
3. **Instrument defects reach six-seed runs.** 111 VOID/UNDEFINED/CONFOUND and 229 amendment commits; FAILURE_LOG sample: ~89% ops/instrument, ~11% brain. The BG grid was voided on 10-31 for a dropped `--commit` [M: 04fa24f6f].
4. **Sync/provisioning stalls.** The iw1 re-flip has been parked 24+ days behind a stale-sha refusal and then sync conflicts [M: runbook RULING iw1-reflip-1028].
5. **One GPU, shared with gaming.** BG cells killed twice by reboots; a 5 h queue deadlock; gate seeds starved behind dev smokes [M: FAILURE_LOG 11-02]. The 34-day BG arc had code on 7 days.

## 3. Recommended plan: "one learning sheet + a native mouth", revised

Three strands, all under the non-negotiables except one declared, logged trade.

**Standing hygiene (week 1, free; from Plan C).** Driver clears sync/provisioner/gate refusals itself, never files them as NEEDS. WORKLOG lines go to a runbook branch squashed weekly. Mandatory dev-seed-7 smoke exercising every bar reader before a six-seed line is queued. Reserved gate lane with priority on gate seeds. GPU allocation written down: production chat stays on the server on numpy; the 3090 takes gate waves and the readout test; any pretraining/fine-tuning runs on AWS spot within $50/day (answers the critique's single-GPU objection). cs-race STOPs (f1j verdict 11-02).

**Strand A — one learning sheet (CPU pool + 3090 gate waves).**
- Wk 1-2: flip gates for BRAIN_PERSIST_LEARNING, BRAIN_ONE_BRAIN and the iw1 re-flip, each with flag-off six-seed arms; bar is the module's own teach→restart→recall proof (81.12→90.34 survives restart) plus an accuracy floor. Confirm the two 10-13 persistence defects (composer AttributeError; two words one code after reload, FAILURE_LOG:402-403) are fixed first. Run the 09-24 plasticity audit at HEAD and publish a per-bridge table (max|dw|, synapses, reply-dependency) as the port-order oracle.
- Wk 3-8: port the composer fact write to a three-factor (pre × post × DA) write into sheet synapses; the spiking `query_patient` read becomes the only authority; host `.kb` read-only. Bar: `.kb` scramble no longer moves the C2 conclusion on ≥5/6 seeds **and** C2 accuracy ≥ today's on ≥5/6 (closes the "delete the reads" loophole).
- Every "same bridge" bar is replaced by an **interaction bar**: lesioning region X changes region Y's output through synapses (keeps the July costume from passing again). Any organ un-frozen on the sheet gets a defined learning target, a named third factor (DA/surprise), and a six-seed forgetting/interference bar (battery regression ≤ 3 rows after N new facts). "max|dw| > 0" alone is never a bar.

**Strand B — native mouth with a declared trade (AWS + 3090 nights).**
- Realism trade, logged under the 09-18 rule: surrogate-gradient/BPTT pretraining of the language cortex stands in for evolution + development (no local-rule existence proof anywhere [E]); everything learned after "birth" (words, facts, episodes, affect associations) is local-rule only. Honesty boundary gains one line: "an ANN emulated in phasors, pretrained; what I know, I learned." No new from-scratch pretrain: fine-tune the existing run3 (83M) / run4 (267M) on dialogue (TinyStories → teacher transcripts → SimpleWiki). The May Qwen-distillation NEGATIVE (ppl 804) stays banked; this is fine-tuning, not distillation.
- Wk 1 decisive test (re-specced per critique): freeze run3's reservoir, detached; train a **multi-layer** readout with KP/FA so it actually differs from the gradient; six seeds; report gap to a BPTT-trained twin readout. Bar: gap ≤ 0.3 nats (prior from 07-20 is ~0.6 on TinyStories). Larger gap pre-declares a widened trade (readout also pretrained). Same week: bits-per-byte, BLiMP and a LAMBADA-style test for run3/run4, Qwen-0.5B and a trigram on one held-out dialogue set, so "distance to a small LLM" becomes one tracked number.
- Wk 2-3 **wiring de-risk before anything else**: affect/WM organ output enters run3's hidden state through synapses on the bridge; lesion changes the wording on dev seed 7, then six seeds. The 10-06 and 09-23 mouth arms were mouth-silent/UNDEFINED [M], so this is the likeliest failure and is tested first. Measure tokens/s on the 3090 with organs loaded.

**Milestones.**
- Wk 4: persistence + one-brain + iw1 ON by default; readout-gap verdict recorded; wiring de-risk verdict on six seeds; comparable-instrument numbers on the board; coordination share < 15%, infra NEEDS = 0. Stage-2 retake run by 11-21 on the current build.
- Wk 8: composer write is a sheet plasticity write, `.kb` read-only, C2 bars met; fine-tuned run3 reaches ≤ 1.3× Qwen bits-per-byte on the dialogue held-out set and BLiMP ≥ 70% [E: BabyLM baselines ~60-70%]; native mouth behind a flag words ≥ 30% of in-vocabulary turns with VERIFY unchanged and no battery regression; LBF reply-level measured **with** each mouth (Qwen and native).
- Wk 12: one pooled rider learning on the sheet with its forgetting bar met; native mouth DEFAULT for in-vocabulary grounded topics, Qwen fallback ≤ 50%; ≥ 2 of 4 affect rows change wording (today 0); ≥ 20 words taught in conversation reproduced in prose after restart 6/6, pretrained weights byte-identical; **Stage 3 attempted on 12-19 as ruled**, expected fail disclosed.

**Kill signals.** Wk 1: readout gap > 0.3 nats → widen the trade, bank the method. Wk 3: wiring lesion changes wording 0/6 → the native mouth is a costume; Strand B drops to measurement-only (LBF with Qwen) and GPU returns to Strand A. Wk 6: sheet port regresses > 20% of battery rows or plasticity moves dw without a reply change 6/6 → keep sheet frozen, route to sparser masks/heterosynaptic normalization. Wk 6: hygiene landed but net flips still ≤ 1.3/week → the stall was never throughput.

**Cost.** 3090: gate waves + ~2 GPU-days for readout/wiring tests. AWS: $400-900 total for fine-tuning seeds within the daily cap. Claude: ~2 days/week (trade memo, preregs, port and wiring designs, verify-go); local driver runs queues, scoring, board sync.

## 4. What it retires

Phase-B "seam by seam on separate organs" as the unit of progress (replaced by organ-onto-sheet with interaction bars). "default_on_spiking_faculties = 30" and census label counts as success metrics (replaced by bridge instances per turn incl. LTM bridges, chat-time dw with reply change, restart survival). `_apply_gain0_freeze` build-then-freeze for ported riders. The host `.kb` as deciding authority. Per-line runbook commits on main; infra NEEDS STOPs. cs-race (stopped), moat and word-role (parked; their job is absorbed if the cortex owns words). The small-vocab research mouths (V=1000-8001) and per-token NLL at V=2000 as a language metric. The 09-19 "do not re-attempt" row, re-marked "re-opened under the 09-18 trade: pretrained prior + local post-birth learning" — only if Decision 1 is yes. Standalone BG STN/GPe cells (continue only as a sheet region). Not retired: the Stage calendar, six-seed rule, the honesty boundary.

## 5. Decisions only you can make

1. **Does "small-LLM-level conversation" belong to the brain or to the system?** Recommended: the brain, at TinyStories/BabyLM/GPT-2-small class on simple English + dialogue, not Qwen-0.5B breadth. This reverses 09-19 §2/§7; Qwen becomes fallback, not permanent mouth.
2. **Ratify the realism trade in writing:** backprop pretraining of the language cortex = developmental prior; local plasticity only after birth; phasor emulation disclosed. Recommended: yes, as a per-mechanism entry under your 09-18 rule, reversible if a local-rule method ever appears.
3. **Moratorium: no new faculty may be born on its own bridge;** every new organ is a sheet region with an interaction bar. Recommended: yes.
4. **GPU and budget allocation:** 3090 = gate waves + short tests; AWS ≤ $50/day for fine-tuning (~$1k over the quarter); production chat stays numpy on the server. Recommended: yes. Stage 3 stays 12-19 per your 09-29 ruling.

## 6. Runner-up

If Decision 1 is "the system": run Strand A alone with the hygiene rules (persistence, one-brain, iw1, `.kb` demotion, one rider learning on the sheet with forgetting bars, LBF measured through the Qwen mouth), keep the Stage calendar, and write into ROADMAP.md that the brain-native language half of the goal is deferred to 2027 by that ruling, so the gap sits on the board rather than silent. It is cheaper, fully within the non-negotiables, and moves the "grows and learns" half; it leaves Qwen as 100% of the words.