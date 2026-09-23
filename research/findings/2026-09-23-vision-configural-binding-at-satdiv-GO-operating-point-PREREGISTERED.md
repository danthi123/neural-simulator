---
type: finding
status: partial
claim_check: synthesis
date: 2026-09-23
mechanism: conjunctive S2.5 binding (fixed-random pairwise AND competitive-selection) re-tested AT the now-confirmed satdiv capability-GO operating point (--s2-norm satdiv --s2-satdiv-sigma 8 --s2-satdiv-scale 760 --ridge 1.0 --n-glimpses 6 --heldout-position --scramble-null) in _vision_lindiscrim_readout_derisk.py, instead of the earlier default op-point (n_glimpses=2, ridge=0.5, no satdiv) every prior binding-vs-flat-pool comparison used
lane: vision (D-perception configural binding / position-invariant readout)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTERED gate below; STAGED on the mini-PC pool, not yet landed (do not report a result until the pool artifacts exist and the sidecar provenance is read).
artifacts:
  - research/findings/raw/lanes/perception/satdiv_sig8_sc760_r1p0_nglim6_heldoutpos_scramblenull_6seed.json
  - research/findings/raw/lanes/perception/satdiv_sig8_sc760_r1p0_nglim6_heldoutpos_scramblenull_seeds200-205.json
external: NO-EXTERNAL-NEEDED -- re-runs two ALREADY-BUILT flags (--conj-bind, --conj-select) of an existing runner at a new operating point; no new mechanism or claim.
builds_on:
  - research/findings/2026-09-17-vision-satdiv-divisive-norm-is-the-decisive-lever-for-position-invariant-readout-binding-not-load-bearing-GO.md
  - research/findings/2026-09-09-vision-configural-binding-competitive-selection-NEXT-MECHANISM-PREREGISTERED.md
---

# Does conjunctive binding help once the front end is already at its capability-GO ceiling? (staged, not yet run)

## Why this check, not a re-derivation

The 09-17 GO finding closed two questions at once but at TWO DIFFERENT operating points, leaving one live
confound. (A) `satdiv` ON at `n_glimpses=6, ridge=1.0, sigma=8, scale=760` (flat pool, `--conj-bind none`)
is `capability_go` 5/6 on seeds 42/43/44/100/101/102 (`task_go_5of6_beat_and_lb: True`,
`satdiv_sig8_sc760_r1p0_nglim6_heldoutpos_scramblenull_6seed.json`), independently reconfirmed 6/6 on a
second seed batch 200-205 (`..._seeds200-205.json`). (B) every conjunctive-binding-vs-flat-pool comparison
in the SAME finding (`conjbind_widthctrl_...`, `conjbind_competitive_...c2basistopo_...`, and every sibling
in `research/findings/raw/lanes/perception/`) ran at `n_glimpses=2, ridge=0.5, satdiv OFF (z-norm)` — the
BORDERLINE op-point from before the satdiv/glimpses lever was found, never the confirmed-GO cell. `bash
tools/before_you_build.sh` (this session) confirms no existing finding or artifact combines `--conj-bind` /
`--conj-select` with `--s2-norm satdiv` (grepped every `conjbind_*` and `vlin_*` provenance sidecar: 0
hits). So "binding is not the lever" (09-17, finding B) is proven at the OLD op-point, not the one the
project now treats as GO — a genuinely untested cell, not a re-sweep.

## The two arms (both additive, byte-identical-off flags already built; no runner/sim edit)

1. **Fixed-random pairwise** (the lane's original binding baseline): `--conj-bind fixed --conj-select
   fixed --conj-n 1152 --conj-offset-max 4`.
2. **Competitive-selection** (the lane's best-performing binding mechanism, `beat4/6-lb6/6` at the OLD
   op-point): `--conj-bind fixed --conj-select competitive --conj-select-overcomplete 4
   --conj-select-kwta-frac 0.1 --conj-n 1152 --conj-offset-max 4`.

Both stacked on the confirmed GO cell: `--s2-norm satdiv --s2-satdiv-sigma 8 --s2-satdiv-scale 760
--s2-satdiv-n 2.0 --ridge 1.0 --n-glimpses 6 --n-s2 96 --heldout-position --scramble-null --seeds 42 43 44
100 101 102`.

## Pre-registered GO gate (fixed BEFORE the run; read the landed artifacts against this, do not re-derive bands after seeing numbers)

Control (already banked, no re-run needed):
`research/findings/raw/lanes/perception/satdiv_sig8_sc760_r1p0_nglim6_heldoutpos_scramblenull_6seed.json`
-- `capability_go` 5/6, `RATE_lin_ceiling_held` 0.5504, `task_go_5of6_beat_and_lb` True (<!--derived--> read
off `by_code.count.summary` in that file); independently reconfirmed 6/6 on
`research/findings/raw/lanes/perception/satdiv_sig8_sc760_r1p0_nglim6_heldoutpos_scramblenull_seeds200-205.json`.

- **NEUTRAL / extends 09-17-B** (most likely, per the prior finding's own diagnosis that this task's
  signal is "fine distributed cosine modulation" a hard conjunction bank is the wrong tool for): either arm
  lands at `capability_go` 5/6 or 6/6 with `RATE_lin_ceiling_held` within +-0.02 of the control's 0.5504 --
  binding remains genuinely not load-bearing even at the ceiling op-point. Report as a strengthened,
  op-point-matched version of 09-17-B (closes the confound honestly), not a new claim.
- **POSITIVE** (a real finding, would need adversarial re-verify before any claim): either arm reaches
  `capability_go` 6/6 on THIS seed set (recovers seed 102, the control's only miss) AND
  `RATE_lin_ceiling_held` rises measurably (> control + 0.02) -- binding is a fine-grained lever precisely
  at the ceiling, not before it. Would motivate a follow-up production wire-in candidate.
- **NEGATIVE** (also informative, strengthens the roadmap's own named fallback): either arm's
  `capability_go` drops below the control's 5/6 or `RATE_lin_ceiling_held` falls -- binding is actively
  harmful even at the GO cell, reinforcing `docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md` §2.1's own
  named fallback ("retire STDP V2/IT and standardize on the validated V1->pooler codon") with a second,
  independent op-point.
- **Anti-cheats**: `--heldout-position --scramble-null` carried on every arm (unchanged from the control);
  `scramble_null_pass` must stay 1.0 on every seed for any arm's result to be trusted.

## Compute + status

Both arms are CPU/numpy, ~70-150s per 6-seed run at this scale (matches every prior `conjbind_*_6seed`
artifact's own `elapsed_seconds`) -- staged on the mini-PC pool (`tools/pool_queue.sh`), not run locally,
not waited on. This file records the pre-registration; a follow-up finding reads the landed artifacts
against the bands above once `tools/pool_health` shows them done.
