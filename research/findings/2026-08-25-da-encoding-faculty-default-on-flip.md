---
type: finding
status: contributing
date: 2026-08-25
mechanism: da-gated-encoding
lane: integration
integration_faculty: da-gated-encoding
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO
instrument: the flip is verified at three levels — (1) the 6-seed magnitude-store no-regression soak
  (research/runners/_da_encoding_leansoak.py --substrate-scaling, the flip GATE, unchanged by this diff); (2) the two
  handler-level no-regression verifiers (research/runners/_da_encoding_wired_verify.py + _wave4_composed_flip_noregression.py)
  re-run through the REAL brain_chat handler WITH the default flipped ON, their OFF arms now pinned to explicit
  BRAIN_DA_ENCODING=0; (3) two flip-specific verifiers proving the DEFAULT (unset) path itself drives + the
  consolidation trigger fires (research/runners/_da_encoding_flip_verify.py + _da_encoding_homeo_trigger.py). LOAD-BEARING
  is proven three ways (g_high>g_low off the live handler; a measurably stronger trace on a magnitude store; a real
  store-synapse change from the consolidation pass) and each is SEVERED by BRAIN_DA_ENCODING_LESION=1 / the =0 escape.
runner: research/runners/_da_encoding_flip_verify.py + research/runners/_da_encoding_homeo_trigger.py
artifacts:
  - research/findings/raw/_da_encoding_leansoak/soak_substrate.json
  - research/findings/raw/_da_encoding_wired/verify.json
  - research/findings/raw/_da_encoding_wired/flip_verify.json
  - research/findings/raw/_da_encoding_wired/homeo_trigger.json
  - research/findings/raw/_wave_flip_soak/composed_noregression.json
external: Turrigiano 2008 "The Self-Tuning Neuron" (Cell 135(3):422-435) -- homeostatic synaptic scaling; DA-gate
  anchor: Lisman-Grace hippocampal-VTA loop / Kandel D.16 (dopamine gates entry into long-term memory).
supersedes: none -- COMPLETES the flip that 2026-08-25-da-encoding-substrate-turrigiano-scaling-FLIP unblocked (its two
  named open rungs are delivered here) and moves the production-integration ledger row da-gated-encoding to
  on_by_default:YES.
---
# DA-gated-encoding FACULTY flipped to production default-ON (Gap-4 write-side; the two prep rungs + flip verify)

## Verdict

**GO (coordinated production default-ON flip).** `da_encoding_enabled()` now defaults ON (`_DA_ENCODING_DEFAULT_ON =
True`), so every taught fact's WRITE MAGNITUDE rides the brain's own self-produced tonic dopamine by default (a salient /
engaged utterance is encoded stronger; Lisman-Grace / Kandel D.16). `BRAIN_DA_ENCODING=0` is the byte-identical escape.
The substrate GO that unblocked this (`2026-08-25-da-encoding-substrate-turrigiano-scaling-FLIP`, the on-substrate
Turrigiano synaptic-scaling homeostat) is unchanged; this finding delivers the two prep rungs that flip named, then
verifies the flip does not regress a turn and IS load-bearing.

## Prep rung 1 -- the OFF-arm verifiers pinned to explicit BRAIN_DA_ENCODING=0

Flipping the default moves the byte-identical baseline from UNSET to `=0`. The two no-regression verifiers had OFF arms
that relied on `unset==off`; an unset OFF arm post-flip silently ARMS the coupling and the comparison becomes ON-vs-ON.
Fixed: `_da_encoding_wired_verify._teach_turn` now exports `BRAIN_DA_ENCODING=0` on its OFF arm (was: pop the key), and
`_wave4_composed_flip_noregression` (which already set each flag to "0") carries a load-bearing comment so the pin is not
regressed. Both re-run GREEN through the REAL handler with the default flipped ON (below), so the pin holds the baseline.

## Prep rung 2 -- the substrate-homeostasis consolidation TRIGGER wired into the idle tick

Turrigiano scaling is biologically SLOW/OFFLINE, so the on-substrate homeostat (`apply_substrate_homeostasis` ->
`OneBrainComposer.apply_homeostatic_scaling`) is a CONSOLIDATION-time pass, not a per-write call. It is now wired into the
between-turn idle tick (`webapp/continuous_engine.consolidate_substrate_homeostasis`, called from
`tick_idle_sessions` alongside the D5 learn-through-use pass; `webapp/server.py` supplies the session chat via
`_get_chat_existing`). A NEW-WRITES-SINCE-LAST-PASS trigger (the session's `len(kb)` must have grown) prevents
re-running the pass on an already-scaled store -- `apply_homeostatic_scaling` is NOT idempotent on repeat calls (it would
keep pulling strong engrams toward unit and erase the DA-salience order). The per-write path keeps its recall-safe floor
(g>=set-point), so the live store is safe between passes. Verified (`_da_encoding_homeo_trigger`, artifact
research/findings/raw/_da_encoding_wired/homeo_trigger.json): the pass FIRES on store-growth -- a real store-synapse
change, mean |w| 1.333333333333333 -> 1.30940917847764 over 3 DA-gated engrams; is a NO-OP with no new writes (returns
None); RE-FIRES after a 4th fact is taught (n_engrams=4); and is disarmed by both BRAIN_DA_ENCODING_LESION=1 and
BRAIN_DA_ENCODING=0.

## The flip verify (this reshapes every turn's response, so it is checked thoroughly)

**(a) No-regression -- 6-seed + both handler verifiers.** The 6-seed magnitude-store soak
(research/findings/raw/_da_encoding_leansoak/soak_substrate.json, seeds 42/43/44/100/101/102, cupy) is GO:
`moat_introduced_total=0`, genuine (target-block-attributed) `stress_net_genuine_violations=0`, derivation cross-check
byte-equal. It constructs its ON/OFF arms directly and does NOT read the flip default, so it is invariant to this diff
(neither the soak runner nor the composer rule is touched). Through the REAL brain_chat handler with the default flipped
ON: the wire-in verifier is GO (research/findings/raw/_da_encoding_wired/verify.json -- OFF `=0` byte-identical: no
da_encoding key, encoding_gain_fn None, recall 'grass'); the composed no-regression over the out-of-scope panel is GO
(research/findings/raw/_wave_flip_soak/composed_noregression.json -- 0 of 8 turns diverge, all-four-ON vs all-four-OFF).
The moat holds -- no new confabulation (the soak's genuine stress-net is 0; the wire-in recall CONTENT is unchanged vs
OFF).

**(b) Load-bearing (drive, not decoration).** A taught fact is recorded MORE STRONGLY when the brain is engaged (higher
DA), and severing the coupling changes the outcome, proven on the DEFAULT (unset) path
(research/findings/raw/_da_encoding_wired/flip_verify.json): teaching the SAME fact under a HIGH-DA turn vs a LOW-DA turn
writes g_high 2.4773555339723594 (DA 1.2386777669861797) > g_low 1.0 (DA 0.04616293556102311) off the live handler; that
same g writes a measurably stronger trace on a magnitude store (stored |w| ratio 2.4773555339723594 == g_high/g_low,
research/findings/raw/_da_encoding_wired/verify.json); and the idle consolidation pass makes a real store-synapse change
(prep rung 2 above). LESIONING (BRAIN_DA_ENCODING_LESION=1) pins g=1.0 on both arms -> the differential VANISHES
(attribution to the live DA read = 1.0, verify.json) -> the memory-strength difference is gone. Drive, not decoration.

**(c) OFF byte-identical to pre-flip.** `BRAIN_DA_ENCODING=0` takes the identical code path pre- and post-flip: the flip
only changes the UNSET default (a one-line named-constant flip), and with `=0` `da_encoding_enabled()` returns False on
both old and new code, so the server skips the install entirely -> `encoding_gain_fn` stays None, no da_encoding key, and
the new idle-tick consolidation is a pure no-op (flip_verify.json: `apply_substrate_homeostasis` and
`consolidate_substrate_homeostasis` both return None under `=0`). The store, the reply, and the between-turn
consolidation are byte-identical to HEAD.

## What flipped

- `webapp/da_encoding_drives_chat.py`: `_DA_ENCODING_DEFAULT_ON = True`; `da_encoding_enabled()` returns it when
  `BRAIN_DA_ENCODING` is unset (the named constant is the ledger anchor).
- `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` row `da-gated-encoding`: `on_by_default: YES` + a `default_anchor` on
  `_DA_ENCODING_DEFAULT_ON` (sub-check A now blocks a source/ledger disagreement).
- `webapp/continuous_engine.py` + `webapp/server.py`: the substrate-homeostasis consolidation trigger on the idle tick
  (additive; no-op under `=0`).

## Next mechanism (no-defer)

The remaining named residuals on this faculty are unchanged by the flip and stay tracked on the ledger row
(scaffold_retired:NO): the message->engagement->SNc-afferent scalar (the same host sensory boundary the #79 DA-mode read
names), the rf numpy fast-path's magnitude-invariant recall (the write-magnitude effect lives on the onebrain magnitude
default), and the host-tuned gain-map constants (k_DA, g_min/g_max == the I-7-b operating point).
