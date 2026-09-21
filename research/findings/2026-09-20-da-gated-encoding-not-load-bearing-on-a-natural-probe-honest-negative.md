---
type: finding
status: live
lane: load-bearing
date: 2026-09-20
---

# DA-gated encoding is NOT load-bearing on a NATURAL conversational probe — a brain-build-verified honest-negative (2026-09-20)

Follow-on to the baseline [`2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md`](2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md) and the REJECTED first fix on branch `research/hollow-da-gated-encoding-drive` ([`2026-09-20-hollow-da-gated-encoding-diagnosis-and-probe-fix.md`](2026-09-20-hollow-da-gated-encoding-diagnosis-and-probe-fix.md)).

## The question, and why the first fix was rejected

`da-gated-encoding` read HOLLOW on the load-bearing baseline: lesioning the DA->encoding-gain link (`BRAIN_DA_ENCODING_LESION`) did not change the reply on the default probe. The first fix made it read "load-bearing" ONLY under a SWEPT read-damage operating point (`BRAIN_ONEBRAIN_RETRIEVE_DAMAGE_SIGMA` ascending across a knee) plus induced arousal — a TUNED condition, which the controller rejected as metric-tuning. This finding asks the honest question the rejection implies, and answers it with a real brain build: is there ANY natural conversational condition — no tuned damage knob, no induce-then-sweep DA — where DA-gated encoding changes the produced reply?

## The mechanism (read directly from `research/runners/one_brain_composer.py`)

DA-gated encoding scales the STORED MAGNITUDE of a trace at write time: `_write_block` (line ~767) writes `complex(g)*zc[k]` for every k in the block — a COMMON scale `g` over the whole D-run of that fact's readout synapses. A CLEAN recall on the production-default magnitude-carrying `OneBrainComposer` (`_COMPOSER_KIND_DEFAULT="onebrain"`) is magnitude-INVARIANT at EVERY decision point:

- `_read_block` (line ~982) kicks ONE block's trigger, resonates, unbinds, cleans up -> membrane `scores` that all scale by ~`g` (a common factor).
- `_select` (line ~968) = argmax over scores (invariant to a common positive scale); or `_spiking_select` (line ~936), whose WTA drive is `(scores/peak)*_cleanup_drive_pA` — PEAK-NORMALIZED, so the winner is invariant to `g`.
- `_margin` (line ~929) = `(peak - runner_up)/(peak+eps)` — a RATIO, so the `confidence_gate` decision is invariant to `g`.
- `query_role`/`query_patient`/`query_agent` return the FIRST cue-matching block (POSITIONAL first-match), never a magnitude-weighted competition among matching facts.
- Blocks are per-block-independent (`store_base + i*block`; each fact = its own trigger + D readout neurons), so adding facts does NOT degrade a given fact's read SNR — there is no natural interference-based degradation.

The ONLY magnitude-sensitive element is the ABSOLUTE RF read floor (`sim/bridge.py:5589`): a block whose readout magnitude falls BELOW the floor decodes to noise -> abstain. A clean read at `g>=1` never approaches it, and the DEFAULT-ON homeostatic floor (`webapp/da_encoding_drives_chat`: `g_floor=1.0`) keeps every write at or above unit. Crossing the floor requires DAMAGING the read (the rejected tuned knob) or writing BELOW unit (which default homeostasis prevents).

## The wall reframe: what the real system runs alongside this

The CLAUDE.md wall question — "what else does the real system run alongside this, that we replaced with a constant?" — answers itself here, and the answer confirms the negative rather than defeating it. The biology runs DA-gated potentiation ALONGSIDE Turrigiano homeostatic synaptic scaling, and in THIS substrate that companion is DEFAULT-ON (`apply_homeostatic_scaling`, the consolidation pass fired on the idle tick): it multiplicatively rescales engrams toward the unit set-point, PRESERVING RELATIVE ORDER while removing the absolute magnitude advantage. So even the write-side boost is regulated toward unit between turns — and the two remaining routes by which magnitude could ever bite a clean read (interference-driven SNR loss; magnitude-arbitrated competition) do not exist in this substrate (per-block-independent reads; positional first-match). The natural conversational path therefore never constructs the stress-gated recall the effect needs.

## The brain build (SIM_BACKEND=numpy, production-default onebrain store, under memcap)

`research/runners/_da_encoding_natural_probe_honest_negative.py`, artifact `research/findings/raw/_da_encoding_natural_probe/verdict.json`. Run: `tools/memcap.sh 12 -- .venv/bin/python -u -m research.runners._da_encoding_natural_probe_honest_negative`.

- **TEST-1 (store-strength -> clean-recall INVARIANCE).** Store the same fact-set at gains spanning the WHOLE natural gain-map range `g in {1.0(tonic/unengaged == the lesion pin), 1.5, 2.0, 3.0(g_max, a maximally salient turn)}`, then a CLEAN recall (query_patient/query_agent/query_role — no damage knob, no confidence-gate change). The recalled SVO is BYTE-IDENTICAL across ALL four gains: `recall_hash = fcb401495e484700` for every `g`. The STORED trace genuinely differs (mean `|w|` = 1.000 / 1.500 / 2.000 / 3.000 — exactly `g`; ratio g_max/unit = 3.000). This is an INVARIANCE demonstration (the opposite of sweeping until it flips): even the CEILING boost g=3.0 does not change the clean recall, so no natural DA level can. The g=1.0 arm IS the `BRAIN_DA_ENCODING_LESION` state, so this is precisely the intact-vs-lesion comparison at the layer that determines the reply.
- **TEST-2 (multi-fact competition is POSITIONAL, not magnitude).** Two facts share the (agent,action) cue, differ in patient, one SALIENT (g=3.0) one UNIT; query the shared cue. Winner when salient stored first = `cat` (the salient fact); winner when unit stored first = `bird` (the unit fact). The winner FOLLOWS STORE ORDER — a magnitude-arbitrated recall would return the salient `cat` in BOTH. So a salient memory does not win a natural competition on this substrate.
- **TEST-3 (end-to-end handler reply).** Opt-in (`--handler`); onebrain handler recall is ~minutes/turn on numpy (cupy preferred). Not required for the verdict: TEST-1's g=1(lesion-pin)-vs-g>1(salient) clean recall on the production `OneBrainComposer` IS the reply-determining-layer comparison, and handler-level recall byte-identity is already the GO [`2026-08-21-da-gated-encoding-wired-into-chat-GO.md`](2026-08-21-da-gated-encoding-wired-into-chat-GO.md).

## Verdict: HONEST-NEGATIVE

DA-gated encoding is NOT load-bearing on a natural conversational probe. It is an encoding-MAGNITUDE effect that is invisible to a clean read: its behavioral bite requires a degraded / stress-gated recall (a read-damage knee) that the natural conversational path never constructs. This is a genuine honest-negative deliverable — it maps what the substrate does on its own — not a stopping point on the CAPABILITY: the write-side coupling remains a real, validated, on-store mechanism (the same GO 2026-08-21), and the durable behavioral pathway for it is memory PERSISTENCE over time (durability against forgetting), not single-turn clean recall.

External grounding (verified 2026-09-20): the coupling's Lisman & Grace 2005 (Neuron 46(5):703-713, the hippocampal-VTA loop) and, for the specific property this negative turns on, Bethus, Tse & Morris 2010 (J Neurosci 30(5):1610-1618, https://www.jneurosci.org/content/30/5/1610, PMID 20130171): intrahippocampal D1/D5 antagonism modulated the PERSISTENCE of new paired-associate memories OVER TIME (their durability against forgetting), NOT their initial encoding or immediate retrieval. A DA-gated encoding effect is biologically expected to be a durability/persistence effect — exactly the read a single clean conversational recall does NOT probe. The honest-negative is thus the biologically faithful result, and it names the RIGHT probe for a future load-bearing test: a natural forgetting/persistence interval (biological, homeostasis-driven decay of the un-boosted trace across turns), NOT a synthetic read-damage sweep.

Brain-based-only + honesty boundary: the write gain rides the brain's OWN self-produced tonic DA (the spiking SNc read); the recall is the on-substrate RF read; host does only the world (the turn text) and the clock. `recalled_svo` is a functional recall read-out; an abstain is an honest not-recalled; no phenomenal claim.

## What is NOT claimed

This does NOT retract the write-side coupling or its wired-into-chat GO — it characterizes the READ-side behavioral consequence on a natural probe. It does NOT claim the capability is dead: it names the durable pathway (persistence-over-time) as the next probe. The end-to-end handler arm is opt-in and unrun here (cupy step); the composer-layer TEST-1 is the decisive intact-vs-lesion measurement. No read-damage knob, no DA sweep, no operating-point tuning is used anywhere in this probe.

## Files

- `research/runners/_da_encoding_natural_probe_honest_negative.py` — the natural-probe runner (TEST-1 invariance across the natural gain range, TEST-2 positional-vs-magnitude competition, opt-in TEST-3 handler reply); selftest-free, prints the verdict + writes the JSON artifact.
- `research/findings/raw/_da_encoding_natural_probe/verdict.json` — the brain-build artifact (HONEST-NEGATIVE; recall_hash identical across gains; write mag ratio 3.0; competition positional) + its provenance sidecar.
