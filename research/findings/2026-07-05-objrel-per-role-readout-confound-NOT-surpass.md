# objrel per-role read-out — NOT a surpass (adversarial-verify CAUGHT a deployment-path confound); the SPIKING boundary STANDS

**Date:** 2026-07-05/06
**Runner:** `research/runners/_rungB1c_objrel_per_role_readout_derisk.py`
**Raw:** `research/findings/raw/_rungB1c_objrel_per_role_readout.json`
**Research gate:** `2026-07-05-objrel-second-research-gate-per-role-readout.md` (RANK 1 = per-role loci + ridge committee).
**Verdict:** the "6-seed-blind GO" is **RETRACTED as a surpass** — an adversarial-verify workflow (ultracode) caught that
the GO's read path is a HOST ridge argmax (= the already-known linear-argmax result), confounded against a SPIKING-WTA
baseline. Kept as an honest correction + a sharper frontier.

## What was claimed, and what the adversarial-verify found

The de-risk replaced the single 3-way winner-take-all (WTA) with per-role independent ridge-committee detectors and reported
canon 1.0 + objrel-slot0 1.0 on all 6 seeds (incl. 3 blind), vs a single-WTA baseline at canon 0.44 / objrel 0.5 — and
attributed the gap to **per-role independence + ridge + committee**. An adversarial-verify (6 skeptic lenses + a
synthesizer that did its own controls) **REFUTED the central attribution**, confirmed by controller inspection:

1. **The comparison is a DEPLOYMENT-PATH confound, not two architectures.** The "per-role main" numbers (`_score_per_role`
   → `PerRoleReadout.predict` → `np.argmax(f @ w)`) are a **HOST argmax on ridge scores** over the spiking reservoir feature
   — NO spiking synaptic read (the runner's own comment at line 210 says "a host ridge argmax"). The "single-WTA baseline"
   (`_c2_single_wta_baseline` → `res._drive_and_read` → argmax over ensemble firing) is the **SPIKING synaptic WTA**. So the
   headline compared host-argmax-per-role (1.0) against spiking-WTA-single (0.44/0.5) — different deployment paths.
2. **Per-role independence is NOT what resolves objrel.** The synthesizer verified that a **SINGLE** multi-output ridge with
   a host argmax ALSO reads objrel-slot0 = 1.0 on the same feature. The host ridge decode was already the established result
   ("a LINEAR argmax gets objrel ~100%"); per-role vs shared makes no difference on the host path.
3. **Committee is NOT load-bearing.** K=1 (no committee), λ=1e-3 → objrel-slot0 1.0. The only load-bearing host-path
   ingredient is a small ridge λ (a generic ridge property, applies to the single read-out too).
4. **The genuinely-SPIKING per-role pool** (`_score_per_role_spiking` — each role a dedicated pool read by its own firing)
   still UNDER-resolves: objrel-slot0 mean **0.333**. That is the real spiking read, and it does NOT surpass.

## What genuinely HOLDS (verified, kept)

- The objrel test is genuinely NON-LOCAL (skeptic HOLDS + controller check): slot0 = THEME (role ≠ position); slot0 sits at
  token-position 1 in ALL constructions so position carries zero discriminative signal; the disambiguator (position 3, verb
  vs determiner) is to the RIGHT of the head → the read requires rightward/whole-sequence integration; a context truncated
  before it reads at chance (0.5), whole-sequence at 1.0. No positional or lexical shortcut. NOT the Mikulasch wall.
- No fact-level leakage (0/12 exact test facts in train; train rng ≠ test rng). Within-construction (objrel in `_TRAIN_KINDS`).
- The lesion + scramble anti-cheats are valid (on the host path).

## The honest verdict + the sharpened frontier

**NOT a surpass.** The objrel role is HOST-decodable from the spiking reservoir feature (single or per-role ridge argmax =
1.0) — which was already known. The **SPIKING read-out** of that feature is the genuine boundary, and it STANDS: the single
spiking WTA gets objrel 0.5, the per-role spiking pool gets 0.333 — neither resolves the sub-1% margin. The per-role
ARCHITECTURE does not help the SPIKING read (the see-saw was a property of the spiking WTA deployment, and a per-role
spiking pool doesn't fix it either).

**The sharpened problem:** the spiking reservoir feature linearly/host-encodes the non-local role, but no spiking read-out
tested (rate-WTA, divided, timing, per-role pool) resolves the sub-1% margin through spikes. The RANK-2 mechanism from the
research gate — a per-role **opponent ACCUMULATOR** that integrates the signed difference over the sentence (√T × √N gain,
drift-diffusion / LIP) — is the remaining ranked spiking mechanism explicitly aimed at resolving a sub-threshold margin by
temporal integration, and is the honest next build. If it too fails, the boundary is that a point-neuron *instantaneous*
spiking read cannot resolve a sub-1% margin a host linear read trivially can — a characterized spiking-read-resolution limit.

## Process note (the discipline worked)

The would-be "surpass" was caught BEFORE commit by the mandated adversarial-verify — six skeptics + a synthesizer that ran
its own single-ridge / K=1 / deployment-path controls. The controller had already flagged the within-construction training
scope; the adversarial pass found the deeper deployment-path confound. This is the honest-negative-as-deliverable standard:
a confounded GO caught and corrected is worth more than a committed overclaim.

## Files
- `research/runners/_rungB1c_objrel_per_role_readout_derisk.py` — the de-risk (host per-role read + a spiking per-role pool variant; NO sim/ edit).
- `research/findings/raw/_rungB1c_objrel_per_role_readout.json` — the 6-seed record (host GO; spiking-pool 0.333).
