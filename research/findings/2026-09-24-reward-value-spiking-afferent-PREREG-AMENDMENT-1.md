---
type: finding
status: partial
lane: load-bearing
date: 2026-09-24
mechanism: A10 (midnight plan S15c) PREREGISTRATION AMENDMENT 1 -- design changes after the adversarial review of 58c400ff6, filed BEFORE the rerun they govern. The GO criteria of the original pre-registration are NOT changed.
seeds: [7]
verdict: AMENDMENT only. No result is claimed here. Filed after the first seed-7 run (58c400ff6, s7.json) was seen; it changes the design and the instrument, not the criteria.
runner: research/runners/_reward_value_afferent_derisk.py
artifacts:
  - research/findings/raw/_reward_value_afferent_derisk/s7.json
---

# A10 pre-registration, amendment 1 (filed before the rerun)

Amends `research/findings/2026-09-24-reward-value-spiking-afferent-PREREGISTRATION.md` (commit 57f0ebfd0).
Filed on `research/reward-value-afferent` in the fix round after the adversarial review of 58c400ff6. The first
seed-7 run (`research/findings/raw/_reward_value_afferent_derisk/s7.json`) HAS been seen: CONFIRM normalized
0.0646 vs CONTRADICT 0.9583 live, 0.8614 vs 0.9690 under the lesion, attribution 0.8795. For that reason this
amendment does NOT touch any GO threshold. It changes the mechanism's routing (the review found it bypassed existing
spiking stages), removes an untested path, and fixes the instrument. Seed 7 stays a dev/calibration seed.

## What stays exactly as pre-registered

- (A) OFF identity: no `reward_value` key on either OFF turn; `da_drives` (afferent_pA, mode) on the `confirm` and
  `contra` turns identical to a pre-patch main build with the same env.
- (B) ON: `reward_value.source == "surprise"` on both turns; contra strictly greater than confirm; and
  `attributable_to(normalized differential, control = lesion) >= 0.9`.
- (C) LESION: |normalized(contra) - normalized(confirm)| < 1e-6.
- GO = A and B and C, on seed 7 only, as a de-risk (never a gate verdict). The 6-seed gate stays in B2b (S28).

## Design changes (the mechanism)

1. **Routing.** The spiking read no longer replaces the SNc current through `afferent_override`. That bypassed
   the default-ON spiking novelty organ, the shared spiking salience afferent, the engagement EMA and the
   content-free HOLD. It now enters `DaModeDrivesWorkspace.observe(..., turn_signal_override=normalized)`. There it
   replaces only the per-turn `engagement_of()` mix. The novelty organ still runs, and the salience afferent, EMA
   and EMA->pA map are unchanged.
2. **The affect-valence fallback is removed.** It fired on nearly every non-assertion turn once affect-drives had
   run, and it was never tested.
3. **The module's own `pa = normalized * 1400` map is removed.** The pA that reaches the SNc is the existing
   pipeline's `da_drives.afferent_pA`. Wherever the pre-registration says "`pa`", this amendment reads
   `da_drives.afferent_pA`.
4. **Errors never drive.** A failed read returns `drives=False` with no `normalized`, and the caller runs the
   pre-existing afferent. The first version drove 0 pA on an error.
5. **Naming (honesty).** The signal is an UNSIGNED prediction-error salience, not a signed reward value. The flag
   names stay as the plan wrote them. The LBF row key becomes `surprise-salience-snc-afferent`.

## Instrument changes (how the criteria are measured)

1. **(A) is measured as pre-registered.** The same battery worker (`onebrain_regression_battery --worker`, turns
   `confirm` and `contra`) runs with the same env at the pre-patch main revision (the branch's merge parent) and at
   the head. `da_drives` and `answer` must be exactly equal, and the whole response (timing keys dropped, listed in
   the artifact) is compared too. An OFF null control (two head OFF builds) must be clean. A module-level half
   (`_reward_value_afferent_offidentity.py`) hashes `observe_turn` pre-patch vs post-patch over fake sessions, with
   a determinism control and a sensitivity negative control.
2. **The verdict can read NO-GO.** Only real preconditions are `Verdict.require`d: arms built, turns measured, the
   override reached the workspace, the lesion cut holds at read time, the null controls are clean. The criteria go
   to `decide(go=...)`. The first runner required every criterion, so it could only print GO or UNDEFINED.
3. **Lesion check at read time.** Each lesioned read records the twin's patient_expected<->surprise weight sum
   (must be 0.0) and the intact organ's sum for reference (`tools.lab.void_if` at read time, `Verdict.reaches` and
   `tools.lab.lever` at scoring).
4. **Reply-level fields are recorded** for every arm and turn: `answer`, `da_drives.mode`, `.lead`,
   `.afferent_pA`, `.da_level`. The midnight plan's S15 success check (the reply changes under the lesion, clean
   null, byte-identical off) is reported in its own block. It is not part of this pre-registration's GO.
5. **Block-asymmetry control (added, reported, not a GO criterion).** The lesioned twin has one route into the
   surprise pool, patient_asserted->surprise. Its patient_expected->surprise edges are zeroed, and it has no other
   pathway. So the within-lesion confirm-vs-contra residual may be a surprise-block-s vs block-t difference, not a
   prediction effect. On a standalone copy of the twin (same seed and build path as the organ's `_ensure_les`),
   the runner replicates the two lesioned reads. It then reads block s and block t with no cue at all. The
   first-run finding's homeostat explanation is withdrawn as unverified: the twin never runs the homeostat.
6. **Composer.** The run uses the production default composer (onebrain, spiking recall) unless `--composer rf`
   is passed, and each read records the composer class. The first run forced `rf`, the host closed-form RF
   composer.
7. **Provenance.** Arms run on the pool at the committed SHA, never on uncommitted code and never as a local brain
   build.

## Statements in the pre-registration that this amendment withdraws

- "a CONFIRM and a CONTRADICT turn read the SAME (elevated) rate -- the differentiation ... COLLAPSES": a
  prediction, refuted by the first run (0.8614 vs 0.9690 under the lesion).
- The baseline "host `engagement_of()` novelty+richness scalar": understated. Its novelty term is spiking, and it
  reaches the EMA through the shared spiking salience afferent. The host parts are the richness count, the mix,
  the EMA and the pA map.
- The self-trace's "`chat.inner.what_does` (the brain's own spiking recall)": true only under the onebrain
  composer. The first run forced `rf`, the host closed-form composer.
- The LBF row on the `contra` turn with `normalized` among its fields, and "`--extra-env`" as the opt-in
  mechanism. `contra` never reaches the lesion (finding 2026-09-20-hollow-surprise-monitor-confirm-probe), and a
  float field reads exact-compare jitter as a change. `--extra-env` exists on `tools/lb_shard.py jobs`, not on
  `load_bearing_fraction.py`, and nothing made the row opt-in. The row now probes `confirm`, compares decision
  fields only, and reads `thin` unless the master flag is set in the harness process.
- The affect-valence fallback (removed, see above).

## Declared, not changed

- `extract_assertion` (regex tokenizer, function-word and WH-word lists, '?' test, exact-3-token rule) is a
  keyword/regex gate on the path. It decides whether the surprise read drives the SNc on a turn. It is registered
  in `docs/SCAFFOLD-LEDGER.md`. S15(c)'s "no keyword or regex classifier" trace cannot pass without an owner waiver.
- The lesion is the organ's own twin, not a substrate-matched cut. The twin is a standalone bridge with no
  homeostat, while the intact read runs on the merged pool slice after the homeostat.
- `hz -> normalized` (`clip(hz / (2*threshold), 0, 1)`) is a host rescale. So are the EMA and the EMA->pA map
  downstream.
