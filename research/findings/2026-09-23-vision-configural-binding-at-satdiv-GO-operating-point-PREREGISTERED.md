---
type: finding
status: partial
claim_check: synthesis
date: 2026-09-23
mechanism: conjunctive S2.5 binding (fixed-random pairwise AND competitive-selection) re-tested AT the now-confirmed satdiv capability-GO operating point (--s2-norm satdiv --s2-satdiv-sigma 8 --s2-satdiv-scale 760 --ridge 1.0 --n-glimpses 6 --heldout-position --scramble-null) in _vision_lindiscrim_readout_derisk.py, instead of the earlier default op-point (n_glimpses=2, ridge=0.5, no satdiv) every prior binding-vs-flat-pool comparison used. AMENDED 2026-09-23 (fix round -- see AMENDMENT LOG below): added a width-matched flat null (n_s2=1152, conj-bind none) AT this op-point, a clean-provenance re-run of the flat control at the current revision, and rewrote the gate as paired per-seed differences in LEARNED_spkwta_held.
lane: vision (D-perception configural binding / position-invariant readout)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTERED gate below (v2, amended); STAGED on the mini-PC pool, not yet landed (do not report a result until the pool artifacts exist and the sidecar provenance is read).
artifacts:
  - research/findings/raw/lanes/perception/conjbind_none_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_6seed.json (clean-provenance control, supersedes the banked satdiv_..._6seed.json as THE control for this gate -- see AMENDMENT LOG)
  - research/findings/raw/lanes/perception/conjbind_widthctrl_n1152_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_6seed.json (width-matched null)
  - research/findings/raw/lanes/perception/satdiv_sig8_sc760_r1p0_nglim6_heldoutpos_scramblenull_6seed.json (original banked control -- misattributed provenance, kept only as a cross-check, not the gate's control)
  - research/findings/raw/lanes/perception/satdiv_sig8_sc760_r1p0_nglim6_heldoutpos_scramblenull_seeds200-205.json
external: NO-EXTERNAL-NEEDED -- re-runs ALREADY-BUILT flags (--conj-bind, --conj-select, --n-s2) of an existing runner at a new operating point; no new mechanism or claim.
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

## The four arms (all additive, byte-identical-off flags already built; no runner/sim edit)

1. **Control** (flat pool at the base width): `--n-s2 96 --conj-bind none`. Re-run at the current revision
   for clean provenance (see AMENDMENT LOG) -- the banked `satdiv_..._6seed.json` used the identical flags
   but its `.prov.json` is misattributed to a different runner's `--help` invocation on dirty code, and the
   real runner has changed twice since (`6c90aef16`, `9fae02391`).
2. **Width-matched null** (flat pool at the binding arms' width, isolates capacity from binding): `--n-s2
   1152 --conj-bind none`. Genuinely new at this op-point -- the only prior width-matched null
   (`conjbind_widthctrl_n1152_heldoutpos_scramblenull_6seed.json`) ran at the OLD op-point (`n_glimpses=2,
   ridge=0.5`, satdiv OFF).
3. **Fixed-random pairwise binding** (the lane's original binding baseline): `--n-s2 96 --conj-bind fixed
   --conj-select fixed --conj-n 1152 --conj-offset-max 4`.
4. **Competitive-selection binding** (the lane's best-performing binding mechanism, `beat4/6-lb6/6` at the
   OLD op-point): `--n-s2 96 --conj-bind fixed --conj-select competitive --conj-select-overcomplete 4
   --conj-select-kwta-frac 0.1 --conj-n 1152 --conj-offset-max 4`.

All four stacked on the confirmed GO cell: `--s2-norm satdiv --s2-satdiv-sigma 8 --s2-satdiv-scale 760
--s2-satdiv-n 2.0 --ridge 1.0 --n-glimpses 6 --heldout-position --scramble-null --seeds 42 43 44 100 101
102`. Arms 3 and 4 have `n_s2=96` base templates but `conj_n=1152` conjunctive readout units -- the SAME
1152-unit readout width as arm 2 -- so arm-2 is the correct isolation of "does binding help over-and-above
mere width", not arm-1.

## AMENDMENT LOG (2026-09-23, fix round -- read this before the gate below)

At amendment time, `ls research/findings/raw/lanes/perception/ | grep -i AT_satdiv` returned **NOTHING**: no
`conjbind_none_AT_satdiv_GO_*`, `conjbind_widthctrl_*_AT_satdiv_GO_*`, `conjbind_fixed_AT_satdiv_GO_*`, or
`conjbind_competitive_AT_satdiv_GO_*` artifact of any kind existed anywhere in the repo, on the pool queue's
`.done`, or (checked via `ssh pool41/pool42 find ... -iname '*AT_satdiv*'`) on either pool node's
`research/findings/raw/lanes/perception/` at revision `f10e13f33`. The two arms the build agent originally
staged (fixed, competitive) had themselves been silently DROPPED from `pool.queue`/`.running`/`.claims`/`.done`
between the build session and this fix round (confirmed absent everywhere; re-queued below) -- so this
amendment happens before ANY seed of ANY arm of this gate has been observed. Nothing here is tuned to a
result. (Arms 3/4 were subsequently re-queued and, as of writing, ARE running on pool41 per `ssh pool41 ps
aux`; arms 1/2 are queued and awaiting dispatch to pool42 -- still zero results landed.)

What changed from the v1 gate a review caught before any artifact landed:
1. Added arm 2 (width-matched null) -- v1 had no way to attribute a binding-arm effect to binding rather
   than to the arm's larger conjunctive-unit count vs the control's 96 base templates.
2. Made arm 1 a fresh run at the current revision instead of reusing the misattributed banked artifact.
3. Replaced the single-seed "recovers seed 102" / unmagnituded "ceiling falls" bands with a paired
   per-seed-difference test on `LEARNED_spkwta_held` (the metric `capability_go` is actually built on,
   per `_vision_lindiscrim_readout_derisk.py:1799-1818`), with an explicit SE-derived threshold, exhaustive
   and mutually exclusive per arm.

## Pre-registered GO gate v2 (fixed BEFORE any result; read the landed artifacts against this, do not re-derive after seeing numbers)

**Primary effect size**: for each binding arm X in {fixed, competitive}, the PAIRED per-seed difference (same
seed, same revision) in `LEARNED_spkwta_held` against BOTH:
  - `d_ctrl(X)[seed] = X.LEARNED_spkwta_held[seed] - control.LEARNED_spkwta_held[seed]` (control = arm 1)
  - `d_width(X)[seed] = X.LEARNED_spkwta_held[seed] - widthctrl.LEARNED_spkwta_held[seed]` (widthctrl = arm 2)
  averaged over the 6 seeds to `mean_d_ctrl(X)` and `mean_d_width(X)`. `d_width` is the DECISIVE quantity --
  it is the one that isolates binding from feature-count, per the arms note above.

**Noise-aware threshold**: from the original banked control's own per-seed `LEARNED_spkwta_held`
(`[0.4792, 0.5208, 0.4896, 0.4792, 0.5104, 0.3854]`), sample SD = 0.048, SE of the 6-seed mean = 0.0197 (~=
the "SD ~0.05" scale named for this gate). The held-accuracy quantum is 1/96 image = 0.0104 (RATE/LEARNED
read out over 96 held positions), so 2 x SE ~= 0.039 ~= 3.75 quantum steps -- well above single-image
rounding noise, used as the pre-registered magnitude tau = **0.04** (rounded up from 2 x SE, fixed before any
arm result is seen; will be re-derived from the CLEAN-PROVENANCE control's own actual per-seed SD once arm 1
lands, and any change to tau from that re-derivation gets its own dated AMENDMENT LOG entry BEFORE arms 3/4
are read against it).

**Bands, applied independently per arm X, exhaustive and mutually exclusive on `mean_d_width(X)`:**
- **BINDING-POSITIVE(X)**: `mean_d_width(X) > +tau` AND `mean_d_ctrl(X) > +tau` (binding beats BOTH the
  width-matched null and the flat control -- not just riding the width increase) AND per-seed
  `capability_go` for X does not regress below the control's count AND `scramble_null_pass == 1.0` on every
  seed of X. A real finding; needs adversarial re-verify before any claim, per the rules. Motivates a
  follow-up production wire-in candidate.
- **BINDING-NEUTRAL(X)**: `-tau <= mean_d_width(X) <= +tau` -- binding's marginal contribution over an
  equal-width flat pool is within noise, REGARDLESS of `mean_d_ctrl(X)`'s sign (if the arm beats the
  96-unit control but not the 1152-unit width-matched null, that lift is a width effect, not binding).
  Report as a strengthened, op-point-matched version of 09-17-B (closes the confound honestly), not a new
  claim. Most likely per 09-17's own diagnosis that this task's signal is "fine distributed cosine
  modulation" a hard conjunction bank is the wrong tool for.
- **BINDING-NEGATIVE(X)**: `mean_d_width(X) < -tau` (binding actively hurts relative to an equal-width flat
  pool) OR (per-seed `capability_go` for X drops below the control's count AND `mean_d_width(X) <= 0`).
  Reinforces `docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md` §2.1's own named fallback ("retire STDP
  V2/IT and standardize on the validated V1->pooler codon") with a second, independent op-point.
- A `POSITIVE`-range `mean_d_width(X)` that fails the secondary `capability_go`/`scramble_null_pass`/
  `mean_d_ctrl` checks is reported as `BINDING-POSITIVE-UNCONFIRMED` (treated as NEUTRAL for the roadmap
  fallback question pending the required adversarial re-verify) -- not silently rounded to NEUTRAL or
  POSITIVE.

**Secondary diagnostic (reported, not gating)**: the same paired-difference construction on
`RATE_lin_ceiling_held` (the host ridge-on-rates ceiling; original v1's sole metric) is still reported
alongside, since it is the quantity the roadmap's headroom bookkeeping already tracks -- but it no longer
decides the verdict, because it is not the capability (`capability_go`) is built on.

Also reported descriptively (not gated): `widthctrl.LEARNED_spkwta_held - control.LEARNED_spkwta_held`,
the pure width effect alone, for context on how much of any control-vs-binding-arm gap width explains by
itself.

**Anti-cheats**: `--heldout-position --scramble-null` carried on every arm (unchanged from the control);
`scramble_null_pass` must stay 1.0 on every seed for any arm's result to be trusted, per the bands above.

## Compute + status

All four arms are CPU/numpy, ~70-150s per 6-seed run at this scale (matches every prior `conjbind_*_6seed`
artifact's own `elapsed_seconds`) -- staged on the mini-PC pool (`tools/pool_queue.sh`), not run locally, not
waited on. Arms 3 (fixed) and 4 (competitive) are confirmed RUNNING on pool41 as of this fix round (`ssh
pool41 ps aux` shows both PIDs, started 09:25/09:27); arms 1 (control) and 2 (widthctrl) are queued and
awaiting a free pool node. This file records the pre-registration; a follow-up finding reads the landed
artifacts against the v2 bands above once `tools/pool_health` shows all four done -- and must re-derive tau
from the clean control's actual per-seed SD (dated amendment) before reading arms 3/4 against it if that
changes tau materially from the 0.04 placeholder above.
