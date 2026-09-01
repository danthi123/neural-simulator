---
type: finding
status: live
date: 2026-09-01
lane: laneC
board: 129
mechanism: source-provenance-honesty (#129) production-default flip, BRAIN_SOURCE_PROVENANCE_HONESTY
runner: research/runners/_source_provenance_honesty_flip_verify.py
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO (20/20 gated preconditions), through the real webapp.server.brain_chat handler
artifacts:
  - research/findings/raw/_source_provenance_honesty_flip/flip_verify.json
---

# Board #129 flip: `BRAIN_SOURCE_PROVENANCE_HONESTY` default-ON — the #140 rung's own hedge wording is now load-bearing on the live GENERATED chain-route reply, byte-identical off, no recall regression

<!--derived-->
**Verdict: GO**, per the cited artifact
`research/findings/raw/_source_provenance_honesty_flip/flip_verify.json` (20/20 gated preconditions).
`research/runners/source_provenance_production_organ.py`'s `source_provenance_enabled()` is
flipped from default-OFF to default-ON (`_DEFAULT_ON = True`): `BRAIN_SOURCE_PROVENANCE_HONESTY` unset now builds
the #129 opponent-comparator provenance organ; `BRAIN_SOURCE_PROVENANCE_HONESTY=0` remains the byte-identical
reversible escape. Independently re-verified through the real `webapp.server.brain_chat` handler (in-process,
`SIM_BACKEND=numpy`), 6-seed at the mechanism level plus a through-the-handler demonstration at the production
singleton's own seed (42) — the same two-layer method `_source_monitoring_honesty_flip_verify.py` established for
the adjacent #140 rung earlier this session.

## Why this flip matters now (it did not, three hours ago)

This organ was already `wired: YES` / `de_risked: YES` in `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` since
2026-08-25 (`2026-08-25-129-source-provenance-honesty-production-wirein-GO.md`), but shipped `on_by_default: NO`
with a documented residual: the GENERATED half of the framing had **no live HTTP exposure** — a chain-route
(composed multi-hop) answer always used the host-generic `frame_derived_answer` wording, never the organ's own
judged-label hedge. Board #140 closed that residual earlier this session
(`webapp/source_monitoring_honesty_chat.py`, `BRAIN_SOURCE_MONITORING_FRAMES_HONESTY`, flipped default-ON in
commit `bedb9ad6e`) — but that rung's entire branch in `webapp/server.py` lives nested **inside**
`if _SP.source_provenance_enabled():`, which was still default-OFF. So the #140 flip was itself hollow in
production until this one lands: with `BRAIN_SOURCE_PROVENANCE_HONESTY` unset, the organ was never built, the
#140 branch never ran, and neither flag changed a single byte of any real reply. This finding is the flip that
makes both rungs actually reach production traffic.

## What was verified, and how

**(1) LOAD-BEARING.** With both env vars fully unset (the new production default), teaching a 2-hop chain
("the wolf hunts the deer" / "the deer eats the worm") and asking the possessive-chain question ("what does the
wolf's prey eat?") swaps the reply from the host-generic `frame_derived_answer` wording to the organ's own hedge:
*"I believe the deer eats the worm, but I reasoned that myself rather than being told it directly."* — driven by
the live readback (`provenance.label == "generated"`, `agrees_with_encoded: True`), not the caller's
`_is_chain_route` claim. Adding `BRAIN_SOURCE_PROVENANCE_HONESTY_LESION=1` (the #129 de-risk's own verified
failing-direction anti-cheat — the Hebbian plasticity gate held shut at encode) collapses the swap back to the
pre-existing wording, and the mechanism-level 6-seed sweep shows the same pattern directly against
`SourceProvenanceHonestyMonitor`: unlesioned accuracy/vary_frac = 1.00 on all 6 seeds, lesioned accuracy collapses
to 0.35-0.50 and vary_frac to 0.30-0.50. Not hollow.

**(2) BYTE-IDENTICAL-OFF.** `BRAIN_SOURCE_PROVENANCE_HONESTY=0` set explicitly produces text identical, on both
the direct-recall and the chain-route turn, to a pre-flip emulation (this module's own `_DEFAULT_ON` monkey-patched
back to `False` with the env fully unset — literally today's pre-flip code path) — asserted by exact string
compare on the captured JSON, not inferred from reading the code, and confirmed the organ is never built in
either arm (no `provenance` key on the response). The env was always set to an explicit `"0"` for every OFF arm,
never `os.environ.pop`-ed (the unset-then-flip staleness trap in `research/FAILURE_LOG.md`).

**(3) MOAT-SAFE + NO-REGRESSION.** The tiny-demo brain's own pre-taught facts (`brain use spikes`, `dog chase
cat`, `cat eat fish` — built into the brain at construction, never taught via this session's chat) recall
correctly (non-abstained, `verified: True`, correct object) under the new default, with answer text **identical**
to the explicit-off baseline for all three — the PERCEIVED framing path is a genuine no-op on real recall. A
never-taught question ("what does the bird eat?") still abstains under both the new default and the off arm — no
fabrication. Across every condition, `derived=True`/`recalled_svo=None` on the chain-route turn and the stated
terminal fact ("worm") are unchanged — this flip only ever swaps the HEDGE WORDING around an already-produced,
already-verified answer, never which fact is stated.

## A verification-design correction made mid-run, disclosed

The runner's first check (1f) originally asserted the direct-recall (PERCEIVED) reply text stays byte-identical
across **all four** handler conditions, including the diagnostic `BRAIN_SOURCE_PROVENANCE_HONESTY_LESION=1` arm.
That arm failed it: with the plasticity gate held shut, a freshly-encoded PERCEIVED fact's judged label ties at
`d == 0.0` (the deterministic `lesion_d_zero` signature the #129 ledger's own `lesion_note` already documents:
*"both prov pools read exactly silent... any residual accuracy is a noisy host tie-break on a fixed RNG stream,
reported not gated"*) and the tie-break happened to read `generated` for this content pattern, spuriously hedging
a directly-recalled fact. This is pre-existing, already-documented, already-accepted noise in the #129 mechanism's
own diagnostic lesion arm — not a property of the default flip (the lesion flag itself defaults OFF in production
and is never set by real traffic), and not one of the three conditions this task actually names. The check was
narrowed to compare only the two arms real traffic can ever be in (`BRAIN_SOURCE_PROVENANCE_HONESTY` unset vs
explicit `=0`), which is the production-relevant moat-safety property and holds exactly (`A == C == D`); the
lesion arm's tie-break is now reported as an observation, not gated, matching the precedent this codebase already
set for this exact mechanism. `research/runners/_source_provenance_honesty_flip_verify.py` carries the corrected
logic; the verdict in the cited artifact was recomputed from the same captured turn data under that correction
(no re-run of the ~24-minute handler/moat turn generation — same measured data, corrected gating scope, disclosed
in the artifact's own `_recompute_note`).

## Scope, per docs/TERMS.md

This flip earns `wired` (already true since 2026-08-25) + `on-by-default` (the new production default, no opt-in
flag now required — LESION-tested per docs/TERMS.md's Level-3 bar). It does **not** earn `scaffold_retired` or
`integrated`: this is an additive honesty-framing wrapper with no pre-existing host provenance pipeline to retire,
as the 2026-08-25 wire-in already documented. `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s row for
`source-provenance-honesty` is left as its own future sync (not touched by this commit, matching the immediately
preceding #140 flip's own scope) — its `on_by_default: NO` line is now stale and should be corrected on the next
ledger sync pass.

## Files

- `research/runners/source_provenance_production_organ.py` — `_DEFAULT_ON = True`; `source_provenance_enabled()`
  now reads ON unless `BRAIN_SOURCE_PROVENANCE_HONESTY` is an explicit off (`0`/`false`/`no`/`off`/`""`).
- `research/runners/_source_provenance_honesty_flip_verify.py` (new) — the two-layer re-verification: a 6-seed
  mechanism sweep against `SourceProvenanceHonestyMonitor` + `provenance_framed_text`, and a through-the-real-
  handler demonstration (`webapp.server.brain_chat`, in-process) across four env conditions plus a known-fact /
  never-taught moat battery.
