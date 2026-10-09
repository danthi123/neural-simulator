# WORKLOG — marker-wta-s2-bisect (self-directed pick, §5 step 1 / RULING iw2-stage1-credit owed (a))

## 2026-10-09 10:05 EDT — choice and why

Picked: **bisect the affect-marker-spiking-wta S2-row loss on main**, range `a308f1e09..1fb805e9e`
(B2b..flip F), iw2 flags off, dev seed 7 first, starting candidate = the `BRAIN_AFFECT_MARKER_SURFACE`
default change (candidate, not a finding). Deliverable = one NEEDS line naming the culprit commit.

Why this one (§5 step 1 ordering):
- GAP_CLOSURE_MISSION CURRENT STATE's Owed list is all Claude's (T7 cells, amended C4 control).
- Roadmap §8 2026-10-05 "Owed: bisect the affect-marker-spiking-wta S2 loss on main" is the first open
  item not reserved to §2, and RULING iw2-stage1-credit (2026-10-05) names it Hermes's, as a §4 line.
- It serves integration-first: the faculty is default-on; a lost battery row on main is a silent
  reply-level regression, and a culprit commit before the Stage-1 attempt (10-24) restores the row's
  meaning.

Non-reserved: local git + short dev-seed-7 probes only; no merge, no gate, no prereg, no scorer, no
default flip. Per-run probes must stay under ~10 min (RULING compute-placement); measured before
bisecting.
## 2026-10-09 (this wake) — record researched; first bisect step queued next, not run yet

- Endpoints REPLICATED with the committed `score_s2` definition (imported from `tools/b2d_score.py`, no new scorer):
  `b2b0924-base` = S2 regressed 4/6 (s44/s100/s101/s102) ; `iw2-alloff-1fb805e` = 0/6 ; `iw2-flip-1fb805e` = 0/6.
  Matches the published findings, so the harness is right and the loss is real between a308f1e09 and 1fb805e9e.
- Static narrowing: only 4 commits in the 718-commit range touch the marker path — 969894855 (candidate,
  default OFF), 5b5ea1b74 (A2 gate), 6ce76fdfd (provenance), 42ba63177 (server marker read). Parent of the
  candidate = 0c86f45b. The adjacent parent-vs-candidate pair therefore decides the culprit with one run each.
- First local cell attempt stopped: one cell = 4 arms x ~3 min = ~12 min > the ~10-min server rule
  (RULING compute-placement), so it will not run on this server. `gpu_remote.sh add --rev` REFUSED both pins
  (no `tools/local_llm/gpu_add.sh` at those revisions). Route chosen next wake: `gpu_remote.sh provision <sha>`
  (both pins have pool_provision.sh) then `tools/pool_queue.sh` with the registered line shape
  `<epoch>\tcd ~/derisk-pool/revisions/<sha> && ...` (dev seed 7 first, then seed 44, dev-first honored).
- Scratch probe: /opt/data/cache/scratch/vm_s2_rescore.py (read-only; imports the committed scorer).
