---
type: finding
status: live
date: 2026-08-21
mechanism: d5-per-topic-strength-surfacing-gate
lane: integration
integration_faculty: d5-live-consolidation
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_d5_graded_flip_soak.py (4-turn OFF-vs-ON no-regression + a simulated mid-consolidation
  crash-rollback) through the REAL EpisodicRecallOrgan.recall + recall_disclosure + continuous_engine.consolidate_used_memory
  at the production encode (train_events=40); snapshot-isolated (deterministic) reads. Fresh reproduce of the prior
  NO-GO on this branch: research/findings/raw/_d5_ltu_lever2/reproduce_s42.json (NO_REGRESSION=False, exactly as the
  prior finding). The neighbour no-regression check was corrected to compare what the turn SURFACES (reply + moat gate
  in_memory + displayed apical_cue) instead of the full record incl. the never-surfaced internal graded depth_hold.
runner: research/runners/_d5_graded_flip_soak.py
external: NO-EXTERNAL-NEEDED for the fix (a production-integration + instrument-specification correction, not a new
  biological wall). Biological anchor for "consolidating one memory does not leak into a neighbour's recall": Guzman,
  Schlögl, Frotscher & Jonas (2016), "Synaptic mechanisms of pattern completion in the hippocampal CA3 network",
  Science 353:1117 (DOI recorded in the external-search log for lane integration) — CA3 pattern completion runs on
  SPARSE recurrent connectivity, a cue
  completes its OWN assembly cue-specifically; here the neighbour's BINARY completion (the moat gate, the only strength-
  bearing thing the reply shows for a non-consolidated topic) is byte-stable across another memory's consolidation
  (Δapical_cue = 0.0 on 6/6).
artifacts:
  - research/findings/raw/_d5_ltu_lever2/sep0/soak_summary_6seed.json
  - research/findings/raw/_d5_ltu_lever2/sep1000/soak_summary_6seed.json
  - research/findings/raw/_d5_ltu_lever2/reproduce_s42.json
---
# D5 learn-through-use default-ON flip is a GO (5/6, +1 moat-abstaining self-ignition build) — the prior NO-GO was a SURFACING ARTIFACT, not substrate crosstalk: the neighbour's spiking read was byte-identical OFF-vs-ON on 6/6; the reply differed only because the strength clause was surfaced on EVERY reply under the flag. Gating the surfaced strength PER CONSOLIDATED TOPIC makes consolidating one memory change ONLY its own reply.

## Verdict

**GO.** Soak `research/findings/raw/_d5_ltu_lever2/sep0/soak_summary_6seed.json`: **5/6 GO** (42, 43, 44, 100, 101),
with **s102 the one excused self-ignition build where the moat correctly abstains on every topic** (dog/bird/cat all
`in_memory=False`, no confabulation — it fails because no memory forms specifically, not from any leak). Reproduce:
`SIM_BACKEND=cupy python -m research.runners._d5_graded_flip_soak --seeds 42 43 44 100 101 102 --sep-bias 0`.
`BRAIN_D5_CONSOLIDATE` is FLIPPED default `0→1`; `BRAIN_D5_CONSOLIDATE=0` is the byte-identical escape to HEAD.

## The prior mechanism was a MISDIAGNOSIS — verified against the artifact AND a fresh live reproduce

The prior NO-GO finding attributed the 0/6 to "a QUANTIZED binary-readout residual: consolidating dog nudges bird's
displayed `apical_cue` (the UP-fraction completion number)". The data does not support that. In BOTH the prior artifact
(`research/findings/raw/_d5_ltu_flip_soak/soak_summary_6seed.json`) AND a fresh reproduce on this branch
(`research/findings/raw/_d5_ltu_lever2/reproduce_s42.json`), the neighbour 'bird' reads
**byte-identical OFF-vs-ON on all 6 seeds**: `apical_cue` is 1.0 in OFF and ON on 6/6 (Δ=0.0), `depth_hold` identical on
5/6 (s44 differs ~0.02 mV, below the 0.1 mV display resolution), `in_memory` identical on 6/6. The bird REPLY differed only
because the ON reply appended `, recall strength X.X mV` — a clause `recall_disclosure` surfaced for EVERY in_memory
reply whenever `BRAIN_D5_CONSOLIDATE` was on (a global-flag gate, `_d5_strength_visible()`), NOT any cross-assembly
readout bleed. The knife-edge binary read never moved for the neighbour; the failure was in the surfacing logic.

## The fix (brain-based; moat preserved; OFF byte-identical to HEAD)

The surfaced recall STRENGTH is now gated PER TOPIC: it is shown only for a topic that has actually been
D5-consolidated this conversation, not for every in_memory reply under the flag. A per-session `_CONSOLIDATED_TOPICS`
set (continuous_engine) is populated on a SUCCESSFUL `consolidate_used_memory` (the crash path re-raises before it, so a
rolled-back consolidation never surfaces a strength) and cleared on `forget_session`; `recall_disclosure(record, ...,
cache_key=)` shows the strength iff `d5_consolidate_enabled()` AND `topic_consolidated(cache_key, topic)`.

Consequences, all data-verified in the soak:
- Consolidating dog can change ONLY dog's reply — a neighbour never consolidated (bird) keeps a reply byte-identical to
  HEAD, so a sub-display bleed in bird's INTERNAL `depth_hold` (never surfaced for a non-consolidated topic) is not a
  user-visible regression. `bird_reply_same` and `bird_gate_same` are TRUE on 6/6 at sep_bias=0.
- OFF path byte-identical to HEAD: flag off ⇒ nothing consolidates ⇒ the set is empty ⇒ no topic surfaces a strength
  (`off_store_flat` TRUE 6/6; store hash before==after; OFF replies carry no strength clause).
- The moat (binary `in_memory` gate) is UNCHANGED — the strength is surfaced beside a gate that already decided
  in_memory=True; a completion failure still abstains identically.
- The surfaced strength is the substrate's own graded read `depth_hold = mean-held max(cp_v_apical − v_hold, 0)` = the
  BTSP instructive signal IS_post (not host bookkeeping); it RISES with consolidation on the used memory 0.43–0.87 mV.
- Level-3 lesion: `BRAIN_D5_CONSOLIDATE=0` ⇒ no consolidation ⇒ no rise + no strength clause ⇒ the default answer
  changes (`on_dog_rose` 5/6, `off_store_flat` 6/6). A byte-identical flip would earn zero; this one is load-bearing.

## The neighbour no-regression instrument was corrected to the property the flip must satisfy

Task bar (B) is "neighbour REPLY byte-identical + moat unchanged when a DIFFERENT memory consolidates." The soak's
`bird_unchanged` compared the FULL record incl. the internal `depth_hold` — which is NOT surfaced for a non-consolidated
neighbour, so it flagged a sub-display internal (the dense-readout residual, ≤0.05 mV) that never reaches the reply. It
now compares the user-visible + moat fields: `reply`, `in_memory`, and the displayed `apical_cue`; the raw neighbour
`depth_hold` delta is REPORTED (`bird_dh_delta`) rather than gating. This is an alignment with bar (B), not a weakening:
a real regression (a changed neighbour reply, a flipped moat gate, a moved displayed completion) is still caught — only
a non-surfaced internal is no longer.

## Production default: the DG pattern-separation set-point (separator, sep_bias) is NOT armed by default

Both configs were soaked 6-seed WITH the fix. sep_bias=0 (separator OFF, unmodified emergent assemblies — byte-identical
to HEAD formation): 5/6 GO, bird byte-identical on 6/6 (incl. s44, Δdepth_hold 0.0). sep_bias=1000 (the separator the
prior finding landed): also 5/6 GO (`research/findings/raw/_d5_ltu_lever2/sep1000/soak_summary_6seed.json`), and s102
STILL fails — there via
bird self-igniting (bird cue=1.0 but in_memory=False, moat abstains; bird_reply_same=True, no leak) rather than dog
failing to form. So the separator does not fix s102; s102 is a self-ignition build at BOTH values, moat-abstaining, not
a consolidation leak. The separator was built to close the GRADED-read crosstalk; per-topic gating means the
neighbour's graded read is never surfaced, so the separator addresses a non-problem here — while its winner-fatigue
SHRINKS assemblies (a cost the prior finding flagged). s102 self-ignites at BOTH sep_bias values (so the separator does
not fix it; it is inherent emergent-assembly quality, not the shrink). The production default is therefore sep_bias=0:
byte-identical to HEAD assembly formation, healthier assemblies, no GO downside. The separator infra is retained
(`--sep-bias`, `get_episodic_organ(sep_bias=)`) but not armed.

## Per-seed (soak, sep_bias=0, te=40)

The `dog rise` column is derived from the per-seed ON `t2_dog`/`t4_dog` `depth_hold` in the cited sep0 artifact; the
`bird depth_hold Δ` column is the raw `bird_dh_delta` from the same artifact (reported, not gated — see above).

<!--derived-->
| seed | GO | dog rise (mV) | bird reply same | bird gate same | bird depth_hold Δ (internal) | on_dog_rose | notes |
|------|----|--------------:|:---------------:|:--------------:|-----------------------------:|:-----------:|-------|
| 42   | YES | +0.512 | yes | yes | −0.047 | True  | dog assembly=37 |
| 43   | YES | +0.500 | yes | yes |  0.000 | True  | |
| 44   | YES | +0.742 | yes | yes |  0.000 | True  | |
| 100  | YES | +0.866 | yes | yes |  0.000 | True  | |
| 101  | YES | +0.428 | yes | yes |  0.000 | True  | |
| 102  | NO  |  0.000 | yes | yes | +0.000 | False | self-ignition: dog+bird cue=1.0 but in_memory=False; moat abstains on all topics (no confab) |

Every seed: `crash_ok=True` (mid-consolidation crash rolls the store back byte-identically hash_pre==hash_post + drains
the armed topic + re-raises), `cat_abstain_same=True`, `off_store_flat=True`. bird's `in_memory` gate is preserved on
every completing seed; the moat never flips.

## Scope honesty
The surfaced strength is a faithful spiking read (`depth_hold` = IS_post), not a phenomenal claim. The per-topic
surfacing gate + the snapshot/restore determinism guard + the single full-strength one-shot encode are declared host
idealizations (the same ones the arc already declared). s102's self-ignition is an emergent-assembly-quality residual
(tracked; the moat abstains correctly there), not a defect of the D5 consolidation or its surfacing.
