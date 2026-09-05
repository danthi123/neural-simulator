---
type: finding
status: live
date: 2026-09-05
mechanism: shared-spiking-salience-novelty-afferent
lane: integration
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: PART A (default-change correctness + load-bearing-not-hollow at the new default, through the 3 real
  consumer entry points) is the 6-seed gate (numpy-CPU, subprocess-per-seed for the process-shared organ singleton).
  PART B (the REAL trained striosome_value critic) and PART C (the REAL brain_chat handler) run ONCE at seed 42 --
  the identical seed-waiver scoping the de-risk (2026-09-05-shared-spiking-salience-afferent-wired-GO.md) and its
  cited siblings use: the heavy value-train critic and the tiny-demo brain build are pre-existing,
  already-6-seed-GO'd mechanisms this flip does NOT modify, and production runs ONE process at ONE seed.
verdict: GO
runner: research/runners/_shared_salience_afferent_prodflip_verify.py
artifacts:
  - research/findings/raw/_shared_salience_prodflip/verify_AB.json
  - research/findings/raw/_shared_salience_prodflip/verify_C.json
  - research/findings/raw/_shared_salience_prodflip/derisk_postflip_6seed.json
external: NO-EXTERNAL-NEEDED -- this is a production-default FLIP of an already-6-seed-GO'd, already-wired coupling
  (2026-09-05-shared-spiking-salience-afferent-wired-GO.md); the biology (one shared salience/novelty afferent
  feeding several consumers; DA/arousal vigor) is grounded in that de-risk and the Aston-Jones-Cohen LC-NE
  adaptive-gain / tonic-DA account it cites. This runner adds the flip-specific gates (default-change correctness,
  integrated no-regression), not a new mechanism.
---

# Shared spiking salience/novelty afferent flipped to production-default-ON (BRAIN_SHARED_SALIENCE) — GO

**Recommendation: GO — safe + genuinely load-bearing to ship default-ON.** All three flip requirements hold: 6-seed
default-change correctness + load-bearing-not-hollow (PART A), the REAL trained critic's commit flips shoe->cat under
lesion (PART B), and the REAL brain_chat handler shows NO regression (PART C: no crash, substantive answer content
byte-identical baseline-vs-default, every other faculty live) with the coupling load-bearing end-to-end. The de-risk
6-seed gate also re-passes 6/6 post-flip. **The flip is NOT merged to main (owner-gated); it is ready on branch
`research/shared-salience-prodflip-verify`.** This flips
`research/runners/shared_salience_afferent.py::shared_salience_enabled()` from default-OFF to **default-ON** (the
env var UNSET now arms the ONE shared spiking ASK-pool afferent at all three consumer sites; `BRAIN_SHARED_SALIENCE=0`
is the byte-identical escape). The default-OFF de-risk is 6-seed GO
([2026-09-05-shared-spiking-salience-afferent-wired-GO.md](2026-09-05-shared-spiking-salience-afferent-wired-GO.md));
this finding is the **production-flip gate** it explicitly deferred: *"A default-ON flip needs its own no-regression
soak on the live production default ... which this de-risk does not attempt."*

## The flip (one boolean, the sole channel)

`shared_salience_enabled()` is the ONLY path by which `BRAIN_SHARED_SALIENCE` reaches any consumer -- all three
consumer sites gate solely on it (`webapp/da_mode_drives_chat.py::observe()`,
`research/runners/bg_action_selection_production_organ.py::salience()`,
`research/runners/value_choice_production_organ.py::default_context_fn()`). The flip changes exactly one literal (the
default from a truthy-only read to a DEFAULT-ON anchor: unset -> True, `{0,false,no,off,''}` -> False). So **ON-by-
default (unset) and ON-by-`=1` take the byte-identical code branch by construction** -- the de-risk's validated ON
behavior transfers verbatim to the new default. The de-risk runner's OFF arm was also made flip-aware (it now writes
`BRAIN_SHARED_SALIENCE=0` explicitly instead of popping -- the 2026-08-27 flip-off-arm-staleness precedent, so the
de-risk 6-seed gate stays valid and re-runnable post-flip; `gates/flip_offarm_staleness` does not fire because both
runners write an explicit `=0` baseline).

## What a GO requires, and how each is tested (runner: `_shared_salience_afferent_prodflip_verify.py`)

The three arms post-flip: **BASELINE** (`=0`, the pre-flip oracle), **DEFAULT** (unset, the new production default),
**EXPLICIT_ON** (`=1`, the validated de-risk arm), **DEF_LESION** (unset + `BRAIN_SHARED_SALIENCE_LESION=1`).

### (1) DEFAULT-CHANGE CORRECTNESS -- PART A, 6-seed, through the 3 real consumers
For each seed, at the FLAG level: `shared_salience_enabled()` is True when unset AND when `=1`, False when `=0`; each
consumer's DEFAULT output matches EXPLICIT_ON (same mode band + within the organ's OU read-tolerance); and the `=0`
BASELINE reproduces the pre-wiring host formula EXACTLY (da_mode: no `shared_salience` key; bg: `(min(1,n/2),
max(0,1-n))`; value: `[0,0.5,1.0]`, each computed independently from source, not a hand-picked literal).

### (2) LOAD-BEARING, NOT HOLLOW (the anti-hollow crux) -- PART A (6-seed) + PART B (real critic) + PART C (handler)
At the NEW DEFAULT (var unset), VARY the salience -> the live decision genuinely CHANGES; LESION the afferent -> that
variation VANISHES toward a shared floor. A hollow coupling (identical decision whether salience varies or not, no
lesion effect) is an explicit NO-GO.

### (3) NO REGRESSION (integrated) -- PART C, the REAL brain_chat handler (stub renderer, all faculties at production defaults)
With the flag ON by default, a battery of real messages through `webapp.server.brain_chat`: no crash; the substantive
answer content (abstained/recalled_svo/verified, computed BEFORE any DA decoration) BYTE-IDENTICAL to the `=0`
baseline; every other faculty still live (the default-ON da_drives trace present on both arms). The DA-mode engagement
suffix may change where the shared afferent moves the self-produced DA across a mode band -- the load-bearing effect,
characterized, not a regression.

## Results

### PART A -- default-change correctness + load-bearing-not-hollow (6 seeds 42/43/44/100/101/102, ALL PASS)
`flag` on every seed: unset->ON, `=1`->ON, `=0`->OFF, lesion active at default -- **flip correct**; `all_gates_pass`
on all 6. Per-seed (rounded from artifact `verify_AB.json`):

<!--derived-->
| seed | da_mode default/lesion mode | da_lesion | da vary intact/lesion | value spread def/lesion (attrib) | bg default/lesion |
|---|---|---|---|---|---|
| 42  | focus / **rest** | 0.0462 | 0.748 / **0.000** | 1.060 / 0.0086 (**99.2%**) | 1.060 / 0.000 |
| 43  | focus / **rest** | 0.0462 | 0.744 / **0.000** | 1.010 / 0.0064 (**99.4%**) | 1.030 / 0.005 |
| 44  | focus / **rest** | 0.0462 | 0.750 / **0.000** | 1.034 / 0.0197 (**98.1%**) | 1.053 / 0.005 |
| 100 | focus / **rest** | 0.0462 | 0.752 / **0.000** | 1.051 / 0.0062 (**99.4%**) | 1.047 / 0.000 |
| 101 | focus / **rest** | 0.0462 | 0.741 / **0.000** | 1.020 / 0.0029 (**99.7%**) | 1.045 / 0.001 |
| 102 | focus / **rest** | 0.0462 | 0.745 / **0.000** | 1.041 / 0.0098 (**99.1%**) | 1.022 / 0.000 |

Every seed: BASELINE (`=0`) is the pre-wiring oracle (da_mode no `shared_salience` key; bg `(1.0,0.0)`==host; value
`[0,.5,1]`); DEFAULT matches EXPLICIT_ON (same mode band, within OU tol); DEFAULT is load-bearing (in the path,
nonzero vs baseline); and every consumer's cross-input variation VANISHES under the lesion -- da_mode da_level varies
by ~0.75 with message salience intact and by **exactly 0.000** under lesion (collapses focus->rest, suffix gone);
value cross-candidate spread collapses to ~1-2% (98-99.7% attributable); bg collapses to ~0. **Not hollow.**

### PART B -- the REAL trained striosome_value critic at the new default (seed 42, `verify_AB.json`)
<!--derived-->
Three candidates (`cat`,`ball`,`shoe`) at recency `[0,.5,1]`, the value-train critic built at
`value_train_trials=40` (490s on the contended CPU). Engagement fed to the critic: DEFAULT `[0.0, 0.306, 1.049]`,
DEF_LESION `[0.001, 0.0, 0.0]`.

| arm | fed spread | commit |
|---|---|---|
| BASELINE (`=0`) | 1.000 | shoe |
| DEFAULT (unset) | 1.049 | shoe |
| DEF_LESION | **0.0014** | **cat** |

Lesioning the shared afferent collapses the fed gradient by **99.86%** (1.049 -> 0.0014, attributable_to) and **FLIPS
the real critic's commit shoe -> cat** -- the de-risk's headline result, reproduced at the production default (the
decision the coupling feeds genuinely changes when the afferent is severed; not cosmetic).

### PART C -- the REAL brain_chat handler, no-regression + load-bearing (seed 42, `verify_C.json`)
<!--derived-->
Through `webapp.server.brain_chat` (stub renderer, brain=`tiny-demo`, ALL faculties at production defaults, a fresh
session per turn), a battery of 3 real messages, BASELINE(`=0`) vs DEFAULT(unset):

| check | result |
|---|---|
| build-determinism (content fields, two fresh baseline sessions) | **True** (clean instrument) |
| no crash on any turn, either arm | **True** |
| substantive answer content (abstained/recalled_svo/verified) byte-identical baseline vs default | **True** |
| every other faculty live (da_drives trace present on both arms) | **True** |
| answers that differ ON vs OFF (the load-bearing suffix effect, NOT content) | 1 / 3 |

**No regression.** The content the moat/recall path produces is byte-identical baseline-vs-default; the flip only
changes the DA-mode engagement suffix where the shared afferent moves the self-produced DA across a mode band (1 of 3
battery messages), exactly the load-bearing effect this coupling is for.

**Load-bearing end-to-end (anti-hollow at the handler).** On the novel/rich message: DEFAULT `da_level=0.791`,
mode=**focus**, lead=`" — worth going further here."`; DEFAULT+LESION `da_level=0.0462`, mode=**rest**, lead=`""` --
the engagement suffix **vanishes** when the shared afferent is severed. da_level tracks message salience intact
(novel 0.791 vs low-content 0.046, spread **0.745**) and is pinned to the floor under lesion (spread **0.000**).

## Overall verdict: GO (all three requirements met)
- **Req 1 NO REGRESSION** -- GO (PART C: no crash, content byte-identical, faculties live, clean instrument).
- **Req 2 LOAD-BEARING NOT HOLLOW** -- GO (PART A 6/6 + PART B real critic commit flip + PART C handler lesion-vanish).
- **Req 3 DEFAULT-CHANGE CORRECTNESS** -- GO (PART A 6/6 unset==explicit-ON / `=0`==oracle; de-risk 6-seed re-run 6/6).

## Anti-cheats / honest scope
- **The anti-hollow test is explicit and per-consumer.** Every consumer's DEFAULT output differs from BASELINE
  (afferent genuinely in the path) AND its cross-input variation VANISHES under the lesion (`attributable_to`, the
  gap#5 discipline), at the actual production default -- not merely at an explicit `=1`.
- **da_mode load-bearing is "afferent-in-path + lesion-collapses", not a large OFF/ON margin** on a single novel
  message: by the coupling's design ON~=OFF `da_level` on one novel message (both focus). The load-bearing proof is
  (a) the nonzero DEFAULT-vs-BASELINE difference, (b) da_level tracking message salience across messages, and (c) the
  lesion collapsing focus->rest (suffix vanishes) -- the #76/#79 signature.
- **bg honest floor-case (unchanged from the de-risk).** The only live-reachable STAY-SILENT entry-gate anchor feeds
  raw=0 (a floor both intact and lesioned); load-bearing + lesion-collapse are proven on `salience()`'s general range.
- **Observed but ORTHOGONAL (not a flip regression): a METACOG #184 warning** fired on the tiny-demo handler battery
  ("an answer was produced by a trace-capable composer but the confidence read came back empty ... the honesty hedge
  is silently disabled"). Its own text names the prior cause (a `TieredFactStore.__setattr__` trace-plumbing bug); it
  concerns the metacog activity-trace, which `BRAIN_SHARED_SALIENCE` does not touch, and the substantive content is
  byte-identical baseline-vs-default (so the flip neither caused nor worsened it). Flagged here honestly as a
  pre-existing plumbing issue worth a separate look, NOT a blocker for this flip.
- **PART C ran with the corpus symlinked into the worktree** (`data/corpus/tinystories.txt`, a gitignored regenerable
  cache absent from a fresh worktree; without it the onebrain build degrades to standalone organs). numpy-CPU here, so
  the onebrain build is slow (~10x); a GPU re-run would be faster but is not needed -- the content-identity + lesion-
  vanish results are backend-independent.
- **NOT "closed" / NOT "scaffold-retired".** This flips the production default of ONE coupling; the raw-scalar
  computations (message novelty/richness, content-token count, fact-recency) remain host sensory/memory-provenance
  boundaries. FUNCTIONAL correlate, NOT phenomenal. The organ is still its own co-resident bridge (rides one-brain
  consolidation, like the sibling curiosity/affect/surprise organs).

## Files
`research/runners/shared_salience_afferent.py` (the flip), `research/runners/_shared_salience_afferent_derisk.py`
(flip-aware OFF arm), `research/runners/_shared_salience_afferent_prodflip_verify.py` (new flip gate). Artifacts:
`research/findings/raw/_shared_salience_prodflip/verify_AB.json` (PART A 6-seed + PART B critic),
`research/findings/raw/_shared_salience_prodflip/verify_C.json` (PART C handler),
`research/findings/raw/_shared_salience_prodflip/derisk_postflip_6seed.json` (de-risk 6-seed re-run).

## Citations
- The de-risk this flips: [2026-09-05-shared-spiking-salience-afferent-wired-GO.md](2026-09-05-shared-spiking-salience-afferent-wired-GO.md) + [2026-09-05-value-choice-real-critic-neural-salience-context-6seed-GO.md](2026-09-05-value-choice-real-critic-neural-salience-context-6seed-GO.md).
- The flip precedent (OFF-arm staleness): research/findings/2026-08-27-flip-soak-off-arm-staleness-audit.md; `gates/flip_offarm_staleness`.
- Sibling default-ON flips (the soak-then-flip pattern): 2026-08-21-da-gated-encoding-wired-into-chat-GO.md, 2026-08-26-bg-action-selection-speak-vs-silent-production-wirein-GO.md.
- Attribution discipline: `tools/lab.py::attributable_to` (the gap#5 lesson).
