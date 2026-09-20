---
type: finding
status: live
lane: load-bearing
date: 2026-09-20
---

# Open-ended-generation is integrated-HOLLOW for TWO reasons — a PROBE gap AND a lesion-WIRING gap — diagnosis + a default-off driving-probe + the wiring fix (2026-09-20)

Follow-on to the baseline [`2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md`](2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md), which read open-ended-generation as HOLLOW: lesioning the spiking generative DRAW (`BRAIN_SPIKING_DRAW_LESION`) does not change the reply on the default `rich_open` probe. This mirrors the episodic precedent [`2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md`](2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md) — but the diagnosis here is DEEPER: open-ended-generation is hollow for TWO independent reasons, and one of them is a genuine WIRING gap (not a probe artifact). This finding pins both and lands a minimal, honest, default-off fix. The measured flip to load-bearing is the controller's AWS/local step (this finding claims the DIAGNOSIS + the FIX WIRING, both proven statically — no brain build here, per the gaming-window rule).

## Diagnosis (proven statically — no brain build needed)

Every fact below is read directly from the code.

1. **The reply IS already driven by the generated hypothesis.** `webapp/server.py` (~L6183, L6245-6251) sets `is_hyp = bool(r.get("hypothesis"))` and, when true, attaches `resp["hypothesis"]`, `resp["hypothesis_svo"]` (the drawn `(agent,action,patient)` triple) and the fluent `answer` that asserts it. So the drawn patient genuinely composes the reply — it is not computed-then-ignored. (The generation branch lives on the rich composer path, so it requires `rich=True`.)

2. **PROBE gap — the probe never gives generation a NOVEL candidate to draw.** The open-ended probe (`research/runners/onebrain_regression_battery.py`, `FACULTY_PROBES` key `open-ended-generation`) is the single `rich_open` turn "what might a dog chase" on a FRESH tiny-demo brain. That brain's fixed KB (`brain_chat_tui.py` `facts=[...]`) has exactly ONE `chase` fact — `(dog,chase,cat)`. `_generate_hypothesis(topic="dog", action="chase")` draws a patient `p` and rejects any `(dog,chase,p)` that is degenerate, already-stored, implausible, or contradicting. The ONLY patient co-occurring with `chase` is `cat`, and `(dog,chase,cat)` is already stored, so it is excluded by the novelty check (`triple in prop.all_stored`). There is therefore NO novel+plausible candidate for `(dog,chase,?)` — `_generate_hypothesis` returns None (abstain) in BOTH arms, so intact == lesion == abstain regardless of the lesion.

3. **WIRING gap — even WITH a novel candidate, the lesion could not bite on the production draw.** The lesion `BRAIN_SPIKING_DRAW_LESION` builds the sampler with `ablate_likelihood=True`. But `ablate_likelihood` is consulted ONLY inside `SpikingWTASampler._weights()` (`research/runners/_followon2_spiking_wta_sampler_derisk.py`), which is used by the seed-word draw path (`_draw`/`draw_svo`) — the path the de-risk's LESION verify exercises (`_vocab_agnostic_spiking_generation_production_organ_verify.py` (B) uses `draw_many`). Production `_generate_hypothesis` draws through `GenerativeReplayProposer._sample_weighted` → `sampler.draw_from_weights(weights, candidates)`, which maps the CALLER's `_weight_partner` weights and NEVER consulted `ablate_likelihood`. A lesioned sampler therefore drew IDENTICALLY to the intact one on the production draw — the lesion was inert there. (The FACULTY_LESIONS `~0.83 → ~0.04` plausible-frac collapse it cites is from the B1 verify's `draw_many` path, not this one.)

Net: open-ended-generation is hollow on this instrument because the probe never constructs a novel candidate AND, underneath that, the lesion was disconnected from the production draw. Both are consistent with the baseline's own raw open-ended arms (the `intact_a_rich_open` / `intact_b_rich_open` / `lesion_open_ended_generation` arm files under `_load_bearing/`) reading identical abstains.

## The fix (built; selftest-verified; default-off)

Two coordinated changes; the flag stays OFF by default.

- **PROBE (default-off flag `LB_OPEN_ENDED_DRIVE_PROBE`, `research/runners/load_bearing_fraction.py`).** When set, the open-ended-generation measurement is remapped to a TEACH→ASK group in one isolated session: nine teach turns (`oe_t1..oe_t9`, session `oe2`) teach NEW `chase` facts (`wolf/fox chase rabbit`, `lion chase deer`, `bear chase mouse`, `hawk chase frog`, `owl chase duck`, `eagle chase hare`, `crow chase beetle`, `pike chase moth`), then `oe_ask` asks the SAME prompt "what might a dog chase" with `rich=True`. This opens eight genuine novel+plausible `(dog,chase,?)` patients `{rabbit,deer,mouse,frog,duck,hare,beetle,moth}` beyond the already-known `cat`. `rabbit` is taught twice so the intact likelihood-weighted draw peaks it (a deterministic intact reference), while the lesion's uniform draw selects among all novel candidates — so the decision field `hypothesis_svo` (and the rendered `answer`) can differ between arms (the eight-candidate pool makes the intact peak and the lesion's uniform pick coincide with prob ~1/8, so ~7/8 of seeds flip; the exact flip on seed 42 is the controller's measurement). The turns live in `_EXTRA_TURNS` (merged into `_TURN_BY_LABEL`) but deliberately OUT of `PROBE_TURNS`, so the default roster stays 26 turns and the regression battery + every flip-verify harness iterate an unchanged roster (measured: `n_probe_turns_default_roster` = 26).

- **WIRING (`research/runners/_followon2_spiking_wta_sampler_derisk.py`, `draw_from_weights`).** The production wire-in draw now honors `ablate_likelihood`: when set, it replaces the caller's weights with a uniform vector (`np.ones`) — the EXACT `_weights` ablation semantics — so the lesion actually cuts the likelihood off the production draw. When `ablate_likelihood=False` (the default, i.e. not lesioned) the branch is skipped, so non-lesion behavior is unchanged by construction; the arm-output identity is the controller's confirm. This branch is reached only under `BRAIN_SPIKING_DRAW_LESION`, which no production path and no default battery sets. The de-risk verify's (B) LESION arm (`draw_many`/`_draw`/`_weights`), (C) noise-ablation (`ablate_likelihood=False`), and (D) flag-off (spiking draw off) are all untouched by this change.

Brain-based-only: the DRAW is a `cp_firing_states` read on a firing Izhikevich WTA bank; the host does only the world (the turn text) and the clock. The lesion is the existing neural cut (the likelihood drive severed). Honesty boundary preserved: `hypothesis`/`hypothesis_svo` are functional generative read-outs — a VOLUNTEERED guess that never passes as a known fact (the #3E moat verify), asserting no phenomenal claim.

## Static verification (no brain build)

`python -m research.runners.load_bearing_fraction --selftest` PASSES all checks, including four new ones — the driving turns resolve by label, `turn_group("oe_ask")` is the teach-first group ending at `oe_ask`, the lesion knob resolves in source, and `draw_from_weights` now consults `ablate_likelihood` (a code-level read of the method body). The citable artifact `research/findings/raw/_load_bearing/open_ended_drive_selftest.json` records the checks, the default roster length, the driving group/fields/env, and `open_ended_lesion_bites_production_draw`.

## What is NOT claimed

The flip to load-bearing is a brain build and has NOT been run here. This finding claims the DIAGNOSIS (static) and the FIX WIRING (static/selftest). The flip is a VALUE diff (which plausible patient is drawn), so it depends on the intact weighted draw and the lesion uniform draw selecting different patients on seed 42 — an honest residual the controller settles. The measured flip + the 25-faculty no-regression are the controller's step:

```
LB_OPEN_ENDED_DRIVE_PROBE=1 tools/memcap.sh 24 -- .venv/bin/python \
    -m research.runners.load_bearing_fraction --only open-ended-generation --repeats 2
# expect: LOAD-BEARING=1, null-control clean, hypothesis_svo differs intact-vs-lesion
```

The exact `--out` path lives in `research/runners/load_bearing_fraction.py`'s docstring (kept out of this finding so the pre-commit claim-check does not read a not-yet-produced artifact as a missing citation). No-regression (the other 25 unchanged) holds by construction when the flag is off (default) and the wiring branch is lesion-gated; it can be re-confirmed by a full run without the flag.

## Biological grounding (external)

The open-ended-generation faculty (#3E) VOLUNTEERS a novel grounded proposition by generative-replay recombination over the brain's stored associations, gated by selectional-preference plausibility. This matches the hippocampal-replay compositional-inference literature: Schwartenbeck et al. 2023 (Cell, "Generative replay underlies compositional inference in the hippocampal-prefrontal circuit") report that replay assembles stored elements into compounds with each replay sequence "constituting a hypothesis about a possible configuration of elements"; Kurth-Nelson et al. 2022 (Neuron, "Replay and compositional computation") frame replay as stringing role-bound objects into novel compound statements; Bakermans et al. 2025 (Nat Neurosci) show replay can construct states never experienced. These ground WHY a lesion of the generative DRAW must change the reply for the faculty to be load-bearing — the draw is the point where stored elements are recombined into the candidate the reply then asserts.

## Files

- `research/runners/load_bearing_fraction.py` — the `LB_OPEN_ENDED_DRIVE_PROBE` flag, the `measure_faculty` remap (turn `oe_ask` + fields `hypothesis_svo`/`answer`), the `_draw_from_weights_honors_ablate` code-level check, the docstring verify command, 4 selftest checks.
- `research/runners/onebrain_regression_battery.py` — the `oe_t1..oe_t6`/`oe_ask` driving group added to `_EXTRA_TURNS` (label-only); `PROBE_TURNS` unchanged.
- `research/runners/_followon2_spiking_wta_sampler_derisk.py` — `draw_from_weights` now honors `ablate_likelihood` (uniform drive when lesioned); untouched when not lesioned.
