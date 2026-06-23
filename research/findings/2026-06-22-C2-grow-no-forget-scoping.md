# Generative-sequence frontier (Spine A) — C2 (GROW + NO-CATASTROPHIC-FORGETTING) scoping (2026-06-22)

> **Status:** READ-ONLY deep-research + code/findings/literature scoping for **C2** of the generative-sequence
> frontier — *demonstrate the GROW → CONFIRM-NO-FORGET back half of the loop* on the consolidated spiking generator
> (C1 just achieved). **NO `sim/` edits, NO experiments, NO GPU.** Single deliverable = this doc. Every load-bearing
> project claim re-verified against the repo (file:line). Continual-learning literature bounded by a fresh June-2026
> pass verified to primary sources (EWC, generative replay, parameter-isolation, CLS, LLM-CL). Builds on — does not
> re-derive — C1 (`2026-06-22-genseq-loopstep3-rf-distill-GO-cheap-ladder-WINS.md`,
> `2026-06-22-genseq-loopstep3-consolidation-scoping.md`, the full-generate C1 raw
> `_genseq_loopstep3_full_genf_generate.json`). **This is a SCOPING/DECISION doc, NOT a brain-based result and NOT a
> commitment to build.** The controller should trust-but-verify the **[VERIFY]** items, push, and present before
> building.

---

## 0. One-paragraph answer (the rest is the evidence)

**C2 is a CLEAN COMPOSE, *once the tension is resolved by naming the right no-forget mechanism* — and that mechanism is
NOT the Phase-1.4 gate-freeze.** The tension the prompt names is real: C1's consolidation is **offline-distill-then-
install** (a PyTorch Adam trainer learns clip-aware weights through the RF-faithful forward `clip(a@W',0,1)`, then
installs them as RF complex synapses — `_genseq_loopstep3_rf_distill_derisk.py:148-233` trainer + `:239-257` install),
whereas the Phase-1.4 gate-freeze (`cp_plasticity_rate_gain=0`, `bridge.py:3153-3177`) is an **on-bridge** mechanism
that freezes on-bridge weight *updates* (STDP/Hebbian). **They don't compose because the C1 path never updates weights
on-bridge — there is no on-bridge STDP to freeze.** The resolution is to recognize C1's pipeline is already pure CLS:
the **offline distill trainer = slow "cortical" learning**, and the no-forget half is **generative self-replay = the
"hippocampal replay" the CLS theory mandates** (McClelland 1995; Shin et al. NeurIPS 2017) — and it is *uniquely free
here because the model being protected IS a generator*: it samples its OWN old distribution (frozen Gen-F generates
old TinyStories text) into the distill corpus alongside the new data, so the next distill installs weights that fit
BOTH. The cheapest decisive C2 demo: consolidated generator on original TinyStories → GROW on a NEW small distribution
(distill on new-data + replayed-old-data) → install on RF → measure learns-new (new-dist ppl drops) WHILE retains-old
(original held-out ppl stays ~7.13) → **anti-cheat: a NO-replay control = catastrophic forgetting (old ppl spikes).**
**Verdict: a clean compose of two literature-standard, project-precedented pieces — the existing distill+install
pipeline (reuse) + a replay-corpus mixer (small new harness) — NOT a genuine new problem, *for the easy corner the
project is in* (ONE new small distribution onto a frozen generator; the literature's effectively-solved case). The
on-bridge Phase-1.4 gate-freeze is the WRONG tool and should NOT be forced onto the offline path; the CLS-correct tool
is replay (or, as the zero-forgetting-by-construction alternative, parameter-isolation — freeze old RF synapses, add a
small new RF capacity).**

---

## 1. The tension, stated precisely — and why the gate-freeze does NOT apply

**C1's consolidation mechanism (verified):** `_genseq_loopstep3_rf_distill_derisk.py`
- the trainer `distill_weights_rf_faithful` (`:148-233`) is **PyTorch Adam, offline** (`:164-169`,
  `torch.optim.Adam`), training `W'` through the differentiable forward `torch.clamp(a @ Wt[L], 0.0, 1.0)`
  (`:172-176`) — i.e. `clip(a@W',0,1)`, the RF-faithful forward.
- the install `rf_stack_forward_install` (`:239-257`) sets those trained `W'` as RF complex synapses
  (`rf_set_complex_weights`, `bridge.py:5691`) and reads `Re(Z)/nsteps` (`bridge.py:5646` `rf_kick` →
  `:5749` `rf_resonate_steps` → `:5684` `rf_read_phases`). The RF path has **no `g·(V−E)`** — that was the load-bearing
  reason the install HOLDS at 0.872 (vs the graded install's 0.444; finding §"the decisive contrast").
- the full-generator C1 (`_genseq_loopstep3_full_genf_generate.json`) installs ALL 3.4M learned matvec params
  (`learned_matvec_params_total: 3408384`) on the RF path EXACT (`a_logit_fidelity_vs_teacher.spearman: 1.0`,
  `teacher_forced_argmax_agreement: 1.0`), generating byte-identical to off-bridge.

**The Phase-1.4 gate-freeze mechanism (verified):** `bridge.set_plasticity_gate(name, 0.0)` (`bridge.py:3153-3177`)
sets `cp_plasticity_rate_gain=0` on a tagged synapse slice; `cp_plasticity_rate_gain` multiplies the STDP / eligibility
/ Hebbian / scaling **weight-UPDATE** deltas (`bridge.py:6862-6876` and the masked-clip paths `:6673/:6990/:7253`). It
is validated on the **conversational** bindings (Phase-1.4 BRANCH A: synonym vocab, 5/6 ≥80% retention —
`continual_forgetting_eval.py` header + CLAUDE.md). It freezes *on-bridge plasticity*.

**Why they don't compose:** the C1 generator's weights are **installed, frozen, and never updated on-bridge** — the
generator does not learn via STDP on the bridge; it learns *offline in the distill trainer* and is *re-installed*. So
there is **no on-bridge weight update for `cp_plasticity_rate_gain` to gate.** Freezing the gate on the generator slice
is a no-op for forgetting, because forgetting (if it happens) happens in the **offline re-distill**, not on the bridge.
**The gate-freeze protects the wrong stage.** (It WOULD be the right tool if C2's grow were done by on-bridge STDP on
the RF path — see §2 option B — but that is the more expensive, less-faithful grow route.)

**The reframe that resolves the tension:** C1's pipeline is *already* a Complementary-Learning-Systems split. Map it:

| CLS component (McClelland 1995; Kumaran-Hassabis 2016) | C1/C2 realization |
|---|---|
| **slow neocortex** (gradual structured learning) | the **offline distill trainer** (`distill_weights_rf_faithful`) — backprop = "a long period of initial learning/growth" per the owner's reframe |
| **fast hippocampus + REPLAY** (interleave old with new to prevent overwrite) | **generative self-replay** — the frozen generator samples its OWN old distribution into the distill corpus |
| **consolidated cortical store** (the stable substrate) | the **RF complex-synapse install** on the one bridge |

So the no-forget half is the CLS theory's *replay*, not a freeze. This is exactly what the project's own Phase-1.3
already does for the conversational vocab — `run_swr_replay_phase` / `run_concept_replay_phase`
(`consolidation_trainer.py:154` / `:43`) interleave hippocampal replay so cortex retains. C2 is the **generative**
form of the same mechanism: the generator IS its own replay source.

---

## 2. Q2 — HOW DOES THE GROW HAPPEN (given the offline-distill consolidation)?

**Three routes, ranked cheapest+most-faithful-first:**

| Rank | Grow route | Mechanism | Cost | Faithful to C1? |
|---|---|---|---|---|
| **#1** | **Re-distill on (new + replayed-old) data, re-install** | Run the SAME trainer (`distill_weights_rf_faithful`) on a corpus = NEW-distribution data **mixed with frozen-generator-sampled OLD data** (the replay); install the new `W'` on the RF path (same `rf_set_complex_weights`). The grow IS another C1 pass on a grown corpus. | **lowest** (reuse the entire C1 pipeline; only the corpus changes) | **EXACT** — the grow step is byte-identically the C1 step, just on more data. Zero new substrate machinery. |
| **#2** | **On-bridge STDP/Hebbian finetune on the RF path** (the gate-freeze's natural home) | Let the RF slice learn the new distribution via on-bridge plasticity, freeze the old-important synapses with `set_plasticity_gate`. | **high** — the RF complex synapses are `cp_rf_w_re/im`, *array-disjoint from `cp_connections`* (CLAUDE.md, the 5b finding), so on-bridge STDP does NOT currently touch them; a `sim/` edit would be needed to make RF weights plastic on-bridge. Plus on-bridge credit assignment for a transformer is the documented hard wall. | **LOW** — re-introduces the very `g·(V−E)` / on-bridge-dynamics gap C1 escaped by distilling offline. |
| **#3** | **Parameter-isolation: freeze old RF synapses, ADD new RF capacity** | Keep the installed old `W'` frozen; install an ADDITIONAL small RF sub-block trained (offline) only on the new distribution; route/sum at read. | **medium** (a small new distill + a routing read) | **HIGH** — additive, the old install is byte-untouched (zero forgetting *by construction*). The cost is capacity growth + a routing decision at read. |

**The honest call:** **#1 is the cheapest and most C1-faithful grow route** — it is literally "run C1 again on a grown
corpus." It composes with the offline-distill consolidation *trivially* because it IS the offline-distill consolidation.
The no-forget burden then lives entirely in **what's in the corpus** (the replay, §3), which is the CLS-correct place
for it. #3 (parameter-isolation) is the **zero-forgetting-by-construction fallback** if #1's replay can't hold retention
to bar. #2 (on-bridge STDP, the gate-freeze's home) is the *least* attractive — it needs a `sim/` edit to make RF
weights plastic AND re-opens the dynamics gap C1 just closed; it is NOT recommended unless the owner specifically wants
the fully-on-bridge-learning version (a separate, deeper arc).

> **[VERIFY]** That the RF install is genuinely re-runnable on a new corpus with no hidden state carryover — i.e. a
> second `distill_weights_rf_faithful` + `rf_set_complex_weights` call cleanly overwrites the first. The trainer
> re-inits `Wt` from the sliced verbatim weights each call (`:168-169`), and `rf_set_complex_weights` sets the full
> complex CSR — so a fresh grow pass is independent. Confirm no residual RF state pins the old weights (the cache in
> `rf_stack_forward_install` is per-`n`, not per-weight — `:250-253` — so it's a bridge *topology* cache, weights are
> re-set each block; looks clean).

---

## 3. Q3 — HOW DOES NO-FORGET HAPPEN (for a consolidated GENERATOR vs conversational bindings)?

**The literature is decisive for this exact case (frozen pretrained GENERATIVE model + ONE new small distribution):**

- **Generative self-replay is the natural fit because the protected model IS a generator** (Shin et al., *Deep
  Generative Replay*, NeurIPS 2017, arXiv:1705.08690 — generative replay is "the only method capable of performing well
  without storing data"; the LLM form **Self-Synthesized Rehearsal**, Huang et al., ACL 2024, arXiv:2403.01244 — frozen
  checkpoint generates synthetic old-task data, mixed with new, "superior-or-comparable to real-data rehearsal, more
  data-efficient"). The frozen Gen-F samples old TinyStories → those samples go into the grow corpus → the re-distill
  fits both. **No stored old data, single unified model, no routing.**
- **The CL-of-LLMs consensus (2023-2026):** a **small replay fraction (≈1–5%, often ≤2%) of old data + LR re-warm
  matches full retraining** and "significantly reduces forgetting" (Ibrahim et al., *Simple and Scalable Strategies to
  Continually Pre-train LLMs*, arXiv:2403.08763; survey Wu et al., CSUR 2025, arXiv:2404.16789). Rehearsal-free
  regularization (EWC) is **not** the scaled winner.
- **EWC / Synaptic-Intelligence** (Kirkpatrick PNAS 2017 arXiv:1611.00796; Zenke ICML 2017 arXiv:1703.04200) — the
  Fisher-importance penalty — **degrades on long sequences and at large/transformer scale** (Fisher saturation; only
  2-task no-forgetting shown). **Poor fit; use only as a cheap add-on, never alone.**
- **Parameter-isolation** (Progressive Networks, Rusu 2016 arXiv:1606.04671; PackNet; LoRA/adapters) — **zero forgetting
  by construction** (freeze old, add new capacity) — **most reliable** but needs distribution-routing at inference.

**Conversational-bindings no-forget vs generator no-forget — the key difference:** Phase-1.4 BRANCH A retains the
conversational primaries because **synonym training REINFORCES the shared motor pools** (`continual_forgetting_eval.py`
header: "synonym training preserves (often improves) primary bindings via shared motor pool reinforcement") — a
*structural-overlap* effect specific to the shared-pool architecture, PLUS the gate-freeze as belt-and-suspenders. **A
generator has no such automatic shared-pool reinforcement** — learning a NEW text distribution does NOT automatically
rehearse the old one (that is exactly catastrophic forgetting). So the generator **must be given** the rehearsal that
the conversational shared-pools got for free — i.e. **explicit generative self-replay**. This is the precise reason the
gate-freeze (which worked for conversation partly via shared-pool reinforcement) is insufficient for the generator: the
generator needs the *replay*, not (only) a freeze.

**⇒ C2 no-forget = generative self-replay (CLS hippocampal-replay analogue), realized as the replay fraction in the
re-distill corpus (#1 grow route).** Fallback if replay underperforms: parameter-isolation (#3, freeze-old + add-new RF
capacity = zero forgetting by construction). EWC is not recommended for this substrate/scale.

---

## 4. Q4 — REUSABLE vs NEW (file:line)

**REUSABLE (the bulk of C2 is reuse — this is why it's a clean compose):**
- **The whole grow step** = the C1 distill+install pipeline: `distill_weights_rf_faithful`
  (`_genseq_loopstep3_rf_distill_derisk.py:148-233`) + `rf_stack_forward_install` (`:239-257`) +
  `install_and_measure_rf` (`:260-346`). The grow IS a second call on a grown corpus.
- **The RF substrate ops** (the install target): `rf_set_complex_weights` (`bridge.py:5691`), `rf_kick` (`:5646`,
  with `neuron_mask` for co-residence `:5656`), `rf_resonate_steps` (`:5749`), `rf_read_phases` (`:5684`),
  `_rf_advance_one` (`:5710`) — all EXACT, array-disjoint from `cp_connections` (immune to nav/conv plasticity per the
  5b finding). The masked-megakernel fast path (`cfg.enable_rf_cudagraph`) for speed.
- **The frozen generator as the replay source** = Gen-F itself (`sim/tiny_transformer.py` `TinyGPT`, ckpt
  `g11_bg/generator_f_gate.ckpt.s42.real.pt`, `_genseq_loopstep3_full_genf_generate.json:genf_checkpoint`) — sample old
  TinyStories from it (`TinyGPT.generate`, the convert-GO already generates from it).
- **The CLS replay PRECEDENT** = `run_swr_replay_phase` (`consolidation_trainer.py:154`) + `run_concept_replay_phase`
  (`:43`) + the awake/sleep gate alternation (`run_consolidation_training:206`) — the project's own
  interleave-replay-to-retain machinery (for conversation); C2 is its generative analogue.
- **The retention-eval skeleton** = `continual_forgetting_eval.py` (the train-A → eval-A → train-B → eval-{A,B}
  retention curve; Phase-1.4) — the structure ports directly (replace "vocab words" with "held-out ppl on each
  distribution").
- **The no-forget GATE-FREEZE** (`set_plasticity_gate`, `bridge.py:3153-3177`) — reusable ONLY if C2 takes grow-route
  #2 (on-bridge STDP); NOT used in the recommended #1 path. Listed for completeness.
- **Persistence across grow steps** = `BridgeLineage.save` (`sim/lineage.py:190`, atomic `.new`+rename + history
  snapshot) + `export_shards` (`:392`) — to checkpoint the consolidated generator between grow rounds.
- **The off-bridge teacher / ppl harness** = `bptt_snn_gpu` `forward_unroll` (`sim/bptt_snn_gpu.py`) and the convert-GO
  generation/ppl path — for the retain/learn ppl measurements.

**NEW (small — the C2-specific glue, no `sim/` edit on the #1 path):**
1. **A replay-corpus mixer** — a small runner that builds the grow corpus = `new_distribution_data` ∪
   `frozen_genf.generate(N)` (the replayed old). ~tens of lines.
2. **A two-distribution ppl harness** — measure held-out ppl on BOTH the original TinyStories and the new distribution,
   before and after the grow. (Extends the convert-GO ppl read to two corpora.)
3. **The new small distribution itself** — a tiny, distinct held-out text corpus the original generator does NOT model
   well (e.g. a small different-domain text, or a synthetic pattern). Must be *measurably distinct* (high pre-grow ppl)
   so "learns-new" is detectable.
4. **The C2 de-risk runner** (`_genseq_C2_grow_no_forget_derisk.py`) — wires 1+2+3 + the reused pipeline + the
   anti-cheats. **No `sim/` edit** (grow-route #1 is pure reuse-by-import).

---

## 5. Q5 — THE CHEAPEST CHEAP-FIRST C2 DE-RISK + GO/NO-GO bar + ANTI-CHEATS

> **Principle:** cheapest-first; CuPy for decisive runs, numpy for tiny smoke (`feedback_gpu_not_numpy`); ≥6 seeds for
> the variable claim (the retention/learning numbers — `feedback_6seed_validation`); the no-confab moat asserted intact
> throughout (the conversational retrieval layer still abstains; C2 touches only the generator slice). The
> **NO-protection forgetting control is the essential anti-cheat** — without it, "old ppl stayed flat" could just mean
> the new distribution was too similar / the finetune too weak to perturb anything.

| # | Step | Scale / cost | What it PROVES | `sim/` edit? | GO / NO-GO |
|---|---|---|---|---|---|
| **0** | **Pre-flight: pick a measurably-distinct new distribution** | minutes (CPU) | the new corpus has high pre-grow ppl under Gen-F (so "learns-new" is detectable) AND the original held-out ppl baseline is pinned (~7.13 / the convert-GO value). | NONE | proceed if pre-grow new-ppl ≫ original-ppl (the distributions are distinct); else pick a more distinct corpus. |
| **1** | **The decisive C2 demo (grow-route #1 + generative self-replay)** | **hours, 1×3090** | (a) GROW: offline-distill on (new-data + frozen-Gen-F-replayed-old) → install on RF → **new-dist ppl DROPS** (learns the new). (b) NO-FORGET: **original held-out ppl STAYS ≈ baseline** (retains the old). | **NONE** (reuse the C1 distill+install on a grown corpus). | **GO** if new-dist ppl drops by a clear margin (≥~20% relative) AND original ppl retains ≥~90% (degrades <~10–15% relative — the CL-LLM replay tolerance). **NO-GO** → escalate to grow-route #3 (parameter-isolation, zero-forget-by-construction) and re-measure. |
| **2** | **THE ESSENTIAL ANTI-CHEAT: the NO-replay (no-protection) control** | runs alongside #1 | with replay REMOVED (distill on new-data ONLY, no frozen-Gen-F samples), the **original ppl SPIKES** (catastrophic forgetting) while new-dist ppl drops. | NONE | the result is only valid if **no-replay forgets (old ppl ↑ markedly) AND with-replay does NOT** — this is the load-bearing contrast that proves the replay is *causal* for retention (not that the task was trivial). |
| **3** | **(if #1 GO) Multi-round + lineage persistence** | day, 1×3090 | the loop iterates: grow→retain→grow again (3 rounds, each a new tiny distribution), persisting via `BridgeLineage.save` between rounds; retention holds across rounds (no slow drift). | NONE | **GO** = the full loop (train→generate→grow→no-forget) demonstrated end-to-end across ≥3 grow rounds with retention held → **C2 DONE**, the owner's loop is closed at a demonstrable scale. |

**Additional anti-cheats (beyond the essential no-replay control):**
- **Replay-fraction sweep** — confirm retention scales with replay fraction (more replay → better retention), the
  CL-LLM dose-response (Ibrahim 2024); a flat curve would mean retention isn't coming from the replay.
- **New-distribution specificity** — the generated text after grow actually reflects the NEW distribution on new-domain
  prompts AND the OLD distribution on old-domain prompts (not a collapse to one).
- **Moat byte-intact** — assert `test_nav_conv_step2b_coresident` / the conversational `is None` abstentions unchanged
  (C2 touches only the generator slice; the no-confab moat must not move). The moat is a plus not a hard gate
  (`feedback_moat_not_hard_lossy_memory_ok`), but a *change* in it would signal cross-talk.
- **Held-out (not train) ppl** for both distributions — never measure retention on trained text (memorization
  confound).

**GO/NO-GO bar (the single decisive line):** **C2 is GO when, with generative self-replay, the new-distribution
held-out ppl drops by a clear margin WHILE the original held-out ppl retains ≥~90% of baseline — AND the no-replay
control catastrophically forgets (original ppl spikes).** The widened tolerance (~10–15% retention loss) follows the
CL-LLM replay literature (≤2% loss at scale with proper replay; widened here for the small-scale + RF-install path).

---

## 6. Q6 — HONEST VERDICT: clean compose, or a genuine new problem?

**CLEAN COMPOSE — for the corner the project is in.** The honest two-part answer:

- **Continual learning on transformers is NOT solved in general** (the literature consensus — Wu et al. CSUR 2025: the
  stability-plasticity "no free lunch" persists, forgetting mechanisms in LLMs are "poorly understood,"
  loss-of-plasticity is a separate open axis). So C2 is *not* a trivially-solved problem in the abstract.
- **BUT the specific case here is the literature's EASY, effectively-solved corner:** ONE new small distribution onto a
  FROZEN pretrained generator, where the generator can replay its own old distribution. For this corner, "small replay
  fraction (1–5%) + re-warm" reliably works (Ibrahim 2024), and generative self-replay is the *natural* mechanism
  because the protected model IS a generator (Shin 2017; Huang 2024). The project ALSO has the exact CLS-replay
  precedent already built for conversation (Phase-1.3 `run_swr_replay_phase`) and the retention-eval skeleton
  (Phase-1.4 `continual_forgetting_eval.py`).

**Why the tension dissolves rather than blocks:** the apparent "offline-trainer vs on-bridge-gate-freeze don't compose"
is resolved by recognizing the gate-freeze is the **wrong** no-forget tool for the offline-distill path (it freezes
on-bridge updates that the C1 path doesn't make). The **right** tool — generative self-replay — composes *trivially*
with the offline distill (it's just *what's in the corpus*), and the grow step is *byte-identically the C1 step on a
grown corpus*. So C2 = (the C1 pipeline, reused) + (a replay-corpus mixer + two-distribution ppl harness, ~small new
glue) + (the essential no-replay anti-cheat). **No `sim/` edit on the recommended #1 path.**

**The one genuine residual risk (honest):** the RF-install fidelity is itself lossy at the *full-generator multi-layer*
scale beyond the C1-validated narrow-512/3-block slice + per-layer-analog-Spearman metric (C1's own honest scope). C2's
grow re-runs that same install, so C2 inherits — does not worsen — C1's fidelity scope. If the full-width end-task ppl
(C1's named follow-on) hasn't been pinned, C2's "ppl drops/retains" is measured through the same install lossiness;
that is a C1-scope caveat to carry forward, not a C2-specific new problem.

**Recommendation to the owner:** proceed with the **#1 grow route (re-distill on new+replayed-old) + generative
self-replay no-forget**, cheapest-first per §5, with the **no-replay forgetting control as the load-bearing
anti-cheat**. Hold **#3 (parameter-isolation, zero-forget-by-construction)** as the NO-GO fallback. Do **NOT** force the
Phase-1.4 gate-freeze onto the offline path; reserve grow-route #2 (on-bridge STDP + gate-freeze) only if the owner
later wants the fully-on-bridge-*learning* generator (a separate deeper arc needing a `sim/` edit to make RF weights
plastic).

---

## 7. Trust-but-verify (load-bearing claims; verified vs flagged)

**Verified directly this pass (file:line / source read):**
- **C1 = offline-distill-then-install** — PyTorch Adam trainer (`_genseq_loopstep3_rf_distill_derisk.py:164` Adam,
  `:172-176` `clip(a@W',0,1)` forward) + RF install (`:239-257` `rf_set_complex_weights`/kick/resonate/read). Full-gen
  C1 installs 3.4M params EXACT, logit Spearman 1.0 (`_genseq_loopstep3_full_genf_generate.json`).
- **The gate-freeze is on-bridge weight-UPDATE gating** — `set_plasticity_gate` (`bridge.py:3153-3177`) sets
  `cp_plasticity_rate_gain` which multiplies STDP/Hebbian deltas (`bridge.py:6862-6876`), validated on conversational
  vocab (`continual_forgetting_eval.py` header). **It cannot gate a stage that makes no on-bridge updates** — the C1
  generator install does no on-bridge STDP. (The reframe's load-bearing claim.)
- **Phase-1.4 retains partly via SHARED-POOL REINFORCEMENT** (`continual_forgetting_eval.py` header: "synonym training
  preserves … via shared motor pool reinforcement") — the architecture-specific effect a generator lacks, ⇒ a generator
  needs explicit replay.
- **The project HAS CLS replay precedent** — `run_swr_replay_phase` (`consolidation_trainer.py:154`),
  `run_concept_replay_phase` (`:43`), awake/sleep gate alternation (`:206`).
- **RF synapses array-disjoint from `cp_connections`** (the 5b finding / CLAUDE.md) — so on-bridge STDP does NOT touch
  RF weights (⇒ grow-route #2 needs a `sim/` edit; #1 avoids the issue entirely).
- **Lineage persistence** — `BridgeLineage.save` atomic (`sim/lineage.py:190-219`), `export_shards` (`:392`).
- **The frozen generator can self-sample** — Gen-F `TinyGPT.generate` (the convert-GO already generates novel
  TinyStories from `g11_bg/generator_f_gate.ckpt.s42.real.pt`).
- **Literature (June-2026 pass, primary sources):** generative replay is the data-free SOTA and natural when the model
  is a generator (Shin NeurIPS 2017 arXiv:1705.08690; Self-Synthesized Rehearsal, Huang ACL 2024 arXiv:2403.01244);
  CL-LLM consensus = small replay fraction (1–5%) + re-warm matches full retrain (Ibrahim arXiv:2403.08763; survey Wu
  CSUR 2025 arXiv:2404.16789); EWC degrades at transformer scale/long sequences (Kirkpatrick PNAS 2017 arXiv:1611.00796;
  Zenke ICML 2017 arXiv:1703.04200); parameter-isolation = zero forgetting by construction (Progressive Nets, Rusu
  arXiv:1606.04671; PackNet); CL-on-transformers NOT solved in general but the one-new-distribution-onto-frozen-gen
  corner is effectively solved.

**Could NOT fully verify (flagged honestly):**
1. **[VERIFY — most load-bearing]** That a second `distill_weights_rf_faithful` + `rf_set_complex_weights` call cleanly
   overwrites the first with no hidden RF-state carryover pinning the old weights (the trainer re-inits `Wt` each call
   `:168-169`; `rf_set_complex_weights` sets the full complex CSR; the `rf_stack_forward_install` cache is per-topology
   not per-weight `:250-253` — *looks* clean; confirm in a tiny smoke).
2. **[VERIFY — the retention number]** That generative self-replay actually holds original ppl ≥~90% at THIS scale +
   the RF-install lossiness (the literature's ≤2% loss is at LLM scale with real corpora; the small-scale + lossy-install
   case is unmeasured — ladder step #1+#2 measure it; the no-replay control proves causality regardless of the absolute
   number).
3. **[VERIFY — full-gen ppl through the install]** That the end-task next-token ppl (not just per-layer analog Spearman)
   is the right C2 metric AND is measurable through the install at full width (C1's own named follow-on — "full-width +
   the end-task next-token head"). C2 inherits this C1-scope caveat.
4. **[VERIFY — the new distribution]** That a *measurably distinct* tiny held-out corpus exists for the demo (high
   pre-grow ppl under Gen-F) — picked in ladder step #0; trivial but must be confirmed distinct, else "learns-new" is
   undetectable.

---

## Sources

### Project record (re-verified this pass, file:line cited)
- `research/findings/2026-06-22-genseq-loopstep3-rf-distill-GO-cheap-ladder-WINS.md` (C1 consolidation read = GO 0.872; offline-distill-then-install; read in full).
- `research/findings/raw/_genseq_loopstep3_full_genf_generate.json` (full-gen C1: 3.4M params on RF, logit Spearman 1.0, argmax-agreement 1.0; read).
- `research/findings/2026-06-22-genseq-loopstep3-consolidation-scoping.md` (the consolidation-step scoping this builds on; §0,§6 read).
- `research/findings/2026-06-22-genseq-step0-C1-consolidation-GO.md` (the entry-point C1 de-risk + its caveats; read in full).
- `research/runners/_genseq_loopstep3_rf_distill_derisk.py` (the offline trainer `:148-233` + RF install `:239-257` + measure `:260-346`; read in full).
- `sim/bridge.py`: `set_plasticity_gate`/`cp_plasticity_rate_gain` (`:3153-3177`, `:6862-6876`); the RF ops (`rf_kick:5646` incl. `neuron_mask:5656`, `rf_read_phases:5684`, `rf_set_complex_weights:5691`, `_rf_advance_one:5710`, `rf_resonate_steps:5749`).
- `research/runners/continual_forgetting_eval.py` (Phase-1.4 retention-eval skeleton + the shared-pool-reinforcement note; header read).
- `research/runners/consolidation_trainer.py` (Phase-1.3 CLS replay: `run_concept_replay_phase:43`, `run_swr_replay_phase:154`, `run_consolidation_training:206`; read).
- `sim/lineage.py` (`BridgeLineage.save:190`, `export_shards:392`).
- `sim/bptt_snn_gpu.py` (the off-bridge spiking forward/teacher; header read). `sim/tiny_transformer.py` (`TinyGPT`, the frozen replay source).

### Current literature (June 2026 pass, primary sources verified)
- **Generative replay** — Shin et al., *Deep Generative Replay*, NeurIPS 2017, arXiv:1705.08690 (data-free SOTA; natural when the model is a generator). Robins, *Catastrophic forgetting + pseudo-rehearsal*, 1995.
- **Self-synthesized rehearsal (LLM)** — Huang et al., *Mitigating Catastrophic Forgetting in LLMs with Self-Synthesized Rehearsal*, ACL 2024, arXiv:2403.01244 (frozen checkpoint generates old-task data; superior/comparable to real-data rehearsal, more data-efficient).
- **CL-LLM continual pretraining** — Ibrahim et al., *Simple and Scalable Strategies to Continually Pre-train LLMs*, arXiv:2403.08763 (small replay fraction + LR re-warm matches full retrain). Survey: Wu et al., *Continual Learning of LLMs: A Survey*, CSUR 2025, arXiv:2404.16789 (CL-on-transformers NOT solved in general).
- **Regularization** — Kirkpatrick et al., *Overcoming catastrophic forgetting* (EWC), PNAS 2017, arXiv:1611.00796 (permuted-MNIST 96.8% vs SGD 63.0%; degrades on long sequences). Zenke et al., *Continual Learning Through Synaptic Intelligence*, ICML 2017, arXiv:1703.04200. Aljundi et al., *MAS*, ECCV 2018.
- **Parameter-isolation** — Rusu et al., *Progressive Neural Networks*, 2016, arXiv:1606.04671 (zero forgetting by construction). Mallya & Lazebnik, *PackNet*, CVPR 2018. LoRA/adapter CL (PEFT; 2025).
- **CLS theory** — McClelland, McNaughton & O'Reilly, *Why there are complementary learning systems*, Psych Review 1995. Kumaran, Hassabis & McClelland, *What learning systems do intelligent agents need?*, Trends Cog Sci 2016.
