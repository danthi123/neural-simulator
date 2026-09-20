---
type: finding
status: live
lane: load-bearing
date: 2026-09-20
---

# DA-gated encoding is integrated-HOLLOW for a PROBE reason (a wiring-presence constant + a deferred effect), not a wiring reason — diagnosis + a default-off driving-probe fix (2026-09-20)

Sibling to the episodic fix [`2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md`](2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md) and follow-on to the baseline [`2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md`](2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md).
`da-gated-encoding` read HOLLOW on the load-bearing baseline: lesioning the DA->encoding-gain link (`BRAIN_DA_ENCODING_LESION`) does NOT change the reply on the default `well` probe.
This finding pins WHY (two static causes) and lands a minimal, honest, default-off instrument fix.
The measured flip is a brain build and is NOT run here (no local brain builds; owner gaming window) — it is the controller's AWS step.

## Diagnosis (proven statically — no brain build needed)

The gap is in the PROBE, not the write-side coupling. Three facts, each read directly from the code:

1. **The checked field `da_encoding.on` is a WIRING-PRESENCE CONSTANT.** `webapp/server.py` builds `da_encoding = {"on": True, ...}` whenever `da_encoding_enabled()` is true — a literal `True`, in BOTH the intact and the lesion arm. The lesion knob the load-bearing harness wires up (`BRAIN_DA_ENCODING_LESION`, `da_encoding_drives_chat.da_encoding_lesioned()`) pins the gain `g=1.0` but never touches `on`. So `da_encoding.on` is `True==True` across the two arms -> `compare()` verdict `pass` -> NOT load-bearing. The lesion has nothing to flip in that field.

2. **The mechanically-correct field `da_encoding.g` is unconditionally EXCLUDED from comparison.** Its leaf `g` is in the battery's `_NOISE_FIELDS`, and `compare()` skips any field whose leaf is a noise field ("never compare a continuous measurement"). So even remapping to `da_encoding.g` would be silently dropped — a field-path fix there is dead on arrival.

3. **The effect is inherently DEFERRED.** The encoding gain scales a STORED trace's MAGNITUDE at write time; a CLEAN RF read is a phase read, magnitude-INVARIANT (the flip-gate soak `_da_encoding_leansoak`'s own result: sigma=0 -> zero regression). So the gain can only surface on a STRESS-tested LATER recall — never in the store-only teach turn's own reply fields. No field-path fix on that turn could work even in principle.

Net: `da-gated-encoding` is hollow on this instrument because (a) the probe field is a constant, (b) the correct field is noise-excluded, and (c) the real effect is a deferred, stress-gated recall consequence the lone `well` teach turn never constructs.
The COUPLING is not a wiring gap: `_da_encoding_wired_verify` is an existing GO proving `g_high>g_low` intact and `g_high==g_low==1.0` under the lesion, on the magnitude-carrying `OneBrainComposer.store_conns` — the SAME default composer the load-bearing worker builds (it never overrides `BRAIN_COMPOSER_KIND`).

## The fix (built; selftest-verified; default-off; byte-identical when off)

A default-off env flag `LB_DA_ENCODING_DRIVE_PROBE` in `research/runners/load_bearing_fraction.py`.
When set, `da-gated-encoding` is remapped to a STORE->RECALL pair that exercises the deferred consequence, and the checked field is remapped to a genuinely categorical recall-turn field:

- Two new turns `dae_store` ("the bird chase the worm") -> `dae_recall` ("what does the bird chase"), session `dae2`, declared store-first. They live in `_EXTRA_TURNS` in the battery, merged into `_TURN_BY_LABEL` but deliberately NOT in `PROBE_TURNS` — the default roster stays 26 turns, so the regression battery and every flip-verify harness are byte-identical. `bird`/`chase`/`worm` are all tiny-demo vocab and `(bird,chase)` is not a build-time cue (build KB: dog-chase-cat, cat-eat-fish, brain-{use,learn,store}-*), so the recall depends ONLY on this turn-stored trace.
- The store turn is taught under HIGH induced DA (`BRAIN_DA_DRIVES_INDUCE`, reused from `_da_encoding_wired_verify`): the intact arm's write gain rides it (`g>1`, a stronger stored `|w|`); the lesion arm pins `g=1.0`. `base_env` also quiets the between-turn idle-tick consolidation (`BRAIN_CONTINUOUS=0`) so the boost is not homeostatically regulated toward unit before the recall.
- The recall is read under the VALIDATED I-7-b READ DAMAGE — a new default-off composer knob `BRAIN_ONEBRAIN_RETRIEVE_DAMAGE_SIGMA` that reuses `_damage_store_conns` VERBATIM (the I-7-b +6/12-within-fact-lift GO operator), applied as a temporary `store_conns` perturbation for the duration of a recall (== the I-7-b `_query_under_damage`, moved in-class as a default-off wrapper over the four public recall methods). The DA-boosted intact trace has higher per-neuron SNR -> survives the RF read floor -> recalls (`recalled_svo=[bird,chase,worm]`); the unit lesion trace degrades below it -> abstains (`recalled_svo=None`) -> the categorical field FLIPS -> LOAD-BEARING. The moat holds under damage (I-7-b: an unstored cue abstains, never confabulates).
- The checked field is remapped from the hollow `da_encoding.on` to `recalled_svo` (categorical, outside `_NOISE_FIELDS`). The read damage is swept ASCENDING across the I-7-b knee (~0.75..4.0): the differential exists only in the knee window (below it both traces recall; above it both degrade), so the first sigma with a clean differential AND a clean null control is the knee. The knee sigma is reported. Every OTHER faculty keeps `base_env={}`, `sigma_grid=None` -> byte-identical.

Brain-based-only: the store gain rides the brain's OWN self-produced tonic DA (the spiking SNc read); the recall is the on-substrate RF read; host does only the world (the turn text) and the clock.
The read damage is the biologically-faithful trace degradation (Kandel D.16 — dopamine gates entry into DURABLE LTM; a weakly-encoded trace decays/interferes below recall while a salient one survives), a documented host-proxy at the SAME composer layer as the DA gate it companions (exactly as the DA gate itself is a host proxy for DA-gated synaptic potentiation).
External grounding (verified 2026-09-20): the coupling's own Lisman & Grace 2005 (Neuron 46(5):703-713, the hippocampal-VTA loop), and — for the specific "boost buys PERSISTENCE, not initial encoding" property this probe exploits — Bethus, Tse & Morris 2010 (J Neurosci 30(5):1610-1618, https://www.jneurosci.org/content/30/5/1610, PMID 20130171): intrahippocampal D1/D5 antagonism modulated the PERSISTENCE of new paired-associate memories OVER TIME (their durability against forgetting), NOT their initial encoding — exactly the intact-survives / lesion-degrades-under-stress differential the probe reads.
The lesion is the existing `BRAIN_DA_ENCODING_LESION` neural-cut (the DA->encoding-gain link; distinct from `BRAIN_DA_DRIVES_LESION`, which collapses the DA LEVEL).
Honesty boundary preserved: `recalled_svo` is a functional recall read-out; an abstain is an honest not-recalled, never a confabulation; no phenomenal claim.

Static verification (no brain build), artifact `research/findings/raw/_load_bearing/da_encoding_drive_selftest.json`: `python -m research.runners.load_bearing_fraction --selftest` PASSES, including the new checks — the driving turns resolve by label, `turn_group("dae_recall") == ["dae_store","dae_recall"]`, the read-damage knob resolves in the composer source, the damage reuses the validated I-7-b `_damage_store_conns` operator, the remapped field `recalled_svo` is outside `_NOISE_FIELDS` (while the hollow `da_encoding.on` is a wiring constant and `da_encoding.g` is noise-excluded), the high-DA store env is armed, and the default roster is unchanged.

## What is NOT claimed

The flip to load-bearing is a brain build and has NOT been run here.
This finding claims the DIAGNOSIS (static) and the FIX WIRING (static/selftest).
The measured flip + the no-regression of the other faculties are the controller's AWS step (cupy strongly preferred so the `OneBrainComposer` builds are ~seconds).
The exact `--out` path + command live in `research/runners/load_bearing_fraction.py`'s docstring (kept out of this finding so the pre-commit claim-check does not read a not-yet-produced artifact as a missing citation).
If the sweep finds NO knee (both arms recall, or both abstain, at every swept sigma), that is an honest not-load-bearing / characterized result reported with the per-sigma outcomes — NOT a fake pass; the grid is env-overridable (`LB_DA_ENCODING_DRIVE_SIGMAS`) to widen the search.

## Files

- `research/runners/load_bearing_fraction.py` — the `LB_DA_ENCODING_DRIVE_PROBE` flag, the `measure_faculty` remap + high-DA `base_env`, the read-damage knee sweep, the docstring verify command, the new selftest checks.
- `research/runners/onebrain_regression_battery.py` — `_EXTRA_TURNS` (`dae_store`, `dae_recall`) merged into `_TURN_BY_LABEL` only; `PROBE_TURNS` unchanged (roster stays 26).
- `research/runners/one_brain_composer.py` — the default-off `BRAIN_ONEBRAIN_RETRIEVE_DAMAGE_SIGMA` read-damage knob (`_maybe_read_damage`, reuse of the I-7-b `_damage_store_conns`), wrapping the four public recall methods; byte-identical when off.
