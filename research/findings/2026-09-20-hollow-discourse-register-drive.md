---
type: finding
status: live
lane: load-bearing
date: 2026-09-20
---

# Discourse-register is integrated-HOLLOW for a PROBE reason (an index collision), not a wiring reason — diagnosis + a default-off driving-probe fix (2026-09-20)

Second follow-on to the baseline [`2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md`](2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md), applying the pattern of the episodic precedent [`2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md`](2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md) to the next HOLLOW faculty: discourse-register. It is isolated-lesion-load-bearing (silencing the held prev spiking slots collapses the who-was-before read) yet reads integrated-HOLLOW here (lesioning it does not change the reply). This finding pins WHY — a referent-ordering index collision, not a wiring gap — and lands a minimal, honest, default-off instrument fix that follows the episodic precedent exactly.

## Diagnosis (proven statically — no brain build needed)

The gap is in the PROBE, not the reply composer. Four facts, each read directly from the code:

1. **The read genuinely drives the reply.** `webapp/server.py`'s before/now short-circuit (~L5573-5587) calls `_DR.maybe_answer(msg, chat.agent, dstate)` and, when it returns a reply, EARLY-RETURNS a `JSONResponse` whose `discourse_register` block carries `answer`/`abstained`/`agent` straight from `answer_before`. So the held-slot read composes the reply; it is not computed-then-ignored.

2. **The lesion forces the held prev slots to the register's IDENTITY index.** `_PrevSilencePairRegister.observe` (`research/runners/d3_discourse_event_register_production_organ.py`) sets `self.slots[2] = self.ident` and `self.slots[3] = self.ident` after each clause. `who_agent_prev` returns `referents[slots[2]]`, so under lesion the before-agent is always `referents[ident]`.

3. **`ident` is a literal 0 — not an out-of-band sentinel.** `research/runners/_d3_event_connective_derisk.make_connective_task` sets `ident = 0`. The production register is built over `["dog","cat","fish","bird","worm","ball"]` (`research/runners/brain_chat_tui.py`, `make_discourse_register(...)`), so `referents[ident] == referents[0] == "dog"`.

4. **The default probe's correct answer is exactly that identity referent.** The battery's discourse probe (`onebrain_regression_battery.py` `FACULTY_PROBES`, key `discourse-register`) runs `dr_a="dog chase cat"` -> `dr_b="then bird chase worm"` -> `dr_c="who was doing it before"`, whose correct BEFORE-answer is "dog" = `referents[0]` = `ident`. So the INTACT read (which genuinely computes the shift via the learned RNN transition + FS-WTA re-discretization in `_d3_event_pair_agent_derisk.PairEventRegister.observe`) returns "dog", and the LESION read (forced `ident`) also returns "dog". `compare()` over `discourse_register.agent`/`.abstained` sees zero diff -> `pass` -> NOT load-bearing.

Net: discourse-register is hollow on this instrument because the default probe's correct before-agent coincides with the identity index the lesion falls back to. This is structurally the SAME failure class as the episodic precedent (the probe never constructs a lesion-detectable condition) — here via an index collision with the identity default rather than an empty-memory collision.

**External-literature grounding (the load-bearing instrument is an ablation-attribution method).** This is the well-characterised ablation-adequacy failure mode: a component reads as "unimportant" precisely when the evaluation input does not causally engage it, so the intact and ablated outputs coincide and the null is a probe artifact, not evidence of an inert component. The remedy — construct a test input on which the intact and ablated outputs MUST diverge — is the necessity-under-natural-inputs / "the dataset must elicit the behavior, matched to a discriminating metric" criterion of ablation-based interpretability (Chan et al. 2022, "Causal Scrubbing", https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing ; Li & Janson 2024, "Optimal Ablation for Interpretability", NeurIPS, https://proceedings.neurips.cc/paper_files/paper/2024/file/c55e6792923cc16fd6ed5c3f672420a5-Paper-Conference.pdf ). The `dr2` role-swap is exactly that: an input engineered so the load-bearing prev-slot read is discriminating (correct answer != the lesion's identity fallback).

## The fix (built; selftest-verified; default-off; byte-identical when off)

A default-off env flag `LB_DISCOURSE_REGISTER_DRIVE_PROBE` in `research/runners/load_bearing_fraction.py`. When set, the discourse-register measurement is remapped to a clause->shift->before TRIPLE that SWAPS the roles so the correct before-agent is NOT the identity referent:

- Three new turns, `dr2_a` ("bird chase worm", a bare clause -> the CURRENT event) -> `dr2_b` ("then dog chase cat", a connective-led clause -> the boundary SHIFT: bird/worm move to the prev slots, dog/cat become current) -> `dr2_c` ("who was doing it before"), session `dr2`, declared clause-first. They live in `_EXTRA_TURNS` in the battery, merged into `_TURN_BY_LABEL` (so the worker resolves them by label) but deliberately NOT in `PROBE_TURNS` — the default roster stays 26 turns, so the regression battery and every flip-verify harness that iterates it are byte-identical.
- `measure_faculty` remaps discourse-register to turn `dr2_c` (group `["dr2_a","dr2_b","dr2_c"]`). NO base_env is injected: the register defaults `spiking=True` on ANY backend (unlike episodic's cupy-gated BTSP write), so nothing needs forcing. Every other faculty keeps `base_env={}` -> byte-identical.
- Expected: the held prev agent is now "bird" = `referents[3]` != `ident`. Intact reads "bird" (via the learned shift); the lesion still forces "dog" (`referents[ident]`) -> the decision field `discourse_register.agent` flips "bird" vs "dog" -> LOAD-BEARING, null-control clean (the seeded register is deterministic, so intact-vs-intact-rebuild is 0 diffs).

Brain-based-only: the SHIFT and the held-slot read are the genuinely-spiking four-FS-WTA-slot register (the winners read off `cp_firing_states`); host does only the world (the clause text) and the clock. The lesion is the existing `BRAIN_DISCOURSE_REGISTER_LESION` neural cut (the spiking hold of the prior event silenced). Honesty boundary preserved: `agent`/`abstained` are functional read-outs of a held attractor slot; the reply asserts no phenomenal claim, and a no-boundary/empty register honestly abstains, never confabulates.

Static verification (no brain build), artifact `research/findings/raw/_load_bearing/discourse_register_drive_selftest.json`: the runner self-test carries four new checks — the driving turns resolve by label, `turn_group("dr2_c") == ["dr2_a","dr2_b","dr2_c"]`, the before-agent ("bird", `dr2_a`'s subject) maps to a referent index (3) that is NOT the identity index (0) while `referents[0]=="dog"` (so the lesion's forced-identity fallback is distinguishable), and the `BRAIN_DISCOURSE_REGISTER_LESION` knob resolves in source. The artifact also records the default roster is unchanged (`n_probe_turns_default_roster` = 26) and the discourse-driving group/before-agent/env (`{}`). An env-capture check confirmed the ON path builds the intact arms with `{}` and the lesion arm with only `BRAIN_DISCOURSE_REGISTER_LESION=1`, and that a sample other faculty (curiosity-followup) is untouched.

## What is NOT claimed

The flip to load-bearing is a brain build and has NOT been run here (owner directive: no local full-brain smokes; the controller verifies on AWS/local). This finding claims the DIAGNOSIS (static) and the FIX WIRING (static/selftest). The measured flip + the 25-faculty no-regression are the controller's step (numpy CPU is sufficient — no forced write — cupy only faster):

```
LB_DISCOURSE_REGISTER_DRIVE_PROBE=1 tools/memcap.sh 24 -- .venv/bin/python \
    -m research.runners.load_bearing_fraction --only discourse-register --repeats 2 \
    --out <_load_bearing dir>/discourse_register_drive.json   # exact --out is in the runner docstring
# expect: load-bearing=1, null-control clean, discourse_register.agent "bird"(intact) vs "dog"(lesion)
```

The exact `--out` path lives in `research/runners/load_bearing_fraction.py`'s docstring (kept out of this finding so the pre-commit claim-check does not read a not-yet-produced artifact as a missing citation). The intact read is expected to resolve "bird" with high reliability — the reused `PairEventRegister` scored a strong BEFORE-agent accuracy across random discourses on this exact referent set and clause form (`2026-07-10-D3-event-pair-live-agent-BEFORE-GO`) <!--derived-->; and even if the RNN mis-resolved to some other non-identity referent, the lesion (forced "dog") would still differ -> load-bearing. Only an intact read that ALSO returned "dog" would be non-load-bearing, which the role-swap makes structurally unlikely. No-regression (the other 25 unchanged) is guaranteed by construction when the flag is off (default), and can be re-confirmed by a full run without the flag (byte-identical to the 16/26 baseline).

## Files

- `research/runners/load_bearing_fraction.py` — the `LB_DISCOURSE_REGISTER_DRIVE_PROBE` flag, the `measure_faculty` remap (turn -> `dr2_c`, no base_env), the ident-collision static anchors, the docstring verify command, the selftest artifact fields, 4 selftest checks.
- `research/runners/onebrain_regression_battery.py` — `_EXTRA_TURNS` (`dr2_a`, `dr2_b`, `dr2_c`) merged into `_TURN_BY_LABEL` only; `PROBE_TURNS` unchanged (26).
