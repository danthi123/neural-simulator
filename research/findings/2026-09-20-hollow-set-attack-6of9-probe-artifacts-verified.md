---
type: finding
status: live
lane: load-bearing
date: 2026-09-20
---

# Hollow-set attack: 6 of 9 "hollow" faculties were probe artifacts (verified load-bearing); 2 fixes failed, 1 rejected as tuned (2026-09-20)

Follow-on to the 6-seed baseline (`2026-09-20-load-bearing-fraction-6seed-0.59-robust-core-14.md`), which left 9
faculties HOLLOW (episodic already corrected separately). A triage+fix workflow classified all 9 as PROBE_ARTIFACT
and built flag-gated (default-off) probe fixes for each. **Brain-build verification (AWS numpy CPU, seed 42) then
separated genuine from claimed — it is NOT 9/9.**

## Verified result (artifact: research/findings/raw/_load_bearing/_hollow_verify/verify_results.txt)

**6 genuinely flip to load-bearing** (intact-vs-lesion decision diff, null-control clean):
- surprise-monitor (treat 1) · discourse-register (treat 1) · common-ground-drives (treat 1) ·
  noncontradiction-gate (treat 3) · affect-coloring (treat 2) · bg-action-selection (treat 1).

**3 do NOT** (correctly excluded):
- prospective-memory — the probe fix did NOT flip it on a real build (treat 0, verdict pass) despite the agent's
  confident "episodic twin" static claim.
- open-ended-generation — same (treat 0, verdict pass).
- da-gated-encoding — REJECTED before verify: its "load-bearing" only appears under a swept read-damage operating
  point (`BRAIN_ONEBRAIN_RETRIEVE_DAMAGE_SIGMA` ascending across a knee) + induced arousal — a tuned condition, not
  a natural conversational drive (53 tuning red-flags in its diff vs 0 for the other 8). An honest characterized
  partial, not a GO.

## Honest reframe of the #1 metric

With a probe that actually exercises each faculty's driving condition, load-bearing is **~23/26 = 0.88 <!--derived-->**
(14 robust-core + 2 borderline + episodic + these 6), up from the **0.59** default-probe floor. The 0.59 was
UNDER-MEASURING 7 faculties (episodic + these 6) whose single-turn / wrong-trial-type / KB-degenerate probes never
triggered a two-phase or conditional driving role — "the instrument is part of the emulation" made concrete.

**Load-bearing caveats (do not overclaim):**
- The 7 probe-fix flips are SINGLE-SEED (seed 42) with IMPROVED, DEFAULT-OFF probes. The production default battery
  still reads ~0.59 (the flags are off); 0.88 is "what the brain drives WHEN probed adequately", not the shipped
  default. 6-seed on the improved probes is the next rung.
- The 2 borderline (affect-marker-spiking-wta, source-provenance-honesty) remain seed-dependent.
- **3 faculties are genuinely NOT load-bearing** even after a fix attempt (prospective-memory, open-ended-generation,
  da-gated-encoding) — the honest actionable remainder.

## Methodology (the real win)

The triage+fix workflow (9 faculties, 18 agents) reported ALL 9 as "tractable, built, selftest PASS, byte-identical".
Static claims did NOT predict the brain: only 6 flip. Trusting them would have produced a false ~0.96. The
discipline that held: brain-build verify per fix + a diff-scan for tuned operating points (caught da-gated-encoding)
+ per-probe legitimacy read. Count none on faith.

## Provenance
- Verify summary (backend-stamped, per-faculty treat/verdict): research/findings/raw/_load_bearing/_hollow_verify/hollow_verify_summary.json
- Raw per-faculty verify: research/findings/raw/_load_bearing/_hollow_verify/verify_results.txt + per-faculty JSONs.
- Fixes on branches research/hollow-<faculty>-drive (all pushed). The 6 verified merge to main; the 2 failed +
  da-gated-encoding stay on branches (not merged).

## Sources (external literature — DR gate for the load-bearing lane)
The probe-adequacy insight is grounded in the neuroscience of these faculties' multi-phase structure. For the
FAILED prospective-memory fix specifically: PM retrieval is a dissociable multi-phase process (maintenance vs
cue-triggered retrieval), and cue-driven spontaneous retrieval fires on an EXACT cue match — so a PM load-bearing
probe must exercise the cue-retrieval phase with an exact-match cue held in a heightened state; our formation->cue
probe read treat=0, implying the tiny-demo cue-detection did not trigger spontaneous retrieval on that turn.
- McDaniel & Einstein 2000, *Appl. Cogn. Psychol.* — multiprocess framework (strategic monitoring vs spontaneous retrieval). https://consensus.app/papers/details/e650e5e30f7b5a8fa424375d482de774/
- Einstein et al. 2005, *JEP:General* — direct test of spontaneous retrieval; exact-cue sensitivity (Mullet et al. 2013, *Psych. & Aging*).
- Cona et al., *Neurosci. Biobehav. Rev.* — AtoDI meta-analysis: dissociable neural bases for the maintenance vs retrieval phases.

## Next
1. Merge the 6 verified probe-fix branches to main; update ledger load_bearing_integrated.
2. 6-seed the improved-probe fraction.
3. The genuine remainder: prospective-memory + open-ended-generation (fixes failed — diagnose why the probe didn't
   flip them) + da-gated-encoding (needs a natural, not tuned, driving condition).
