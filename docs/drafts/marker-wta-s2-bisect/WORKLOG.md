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

## 2026-10-09 11:17 EDT — four controlled cells dispatched (queue)
Old pins 969894855/0c86f45b refused as stale (provision.json, 15:06Z) so a literal
per-commit bisect needs an owner-approved one-off (flip-F precedent) = not driver-callable.
Intervention instead, all on one provisionable revision (research/marker-s2-bisect =
3d29c10b3, state=ok on pool40/41/42/wspool): control (marker default OFF) vs
BRAIN_AFFECT_MARKER_SURFACE=1 (code's own docstring: restores pre-2026-09-25 behavior
byte-identically), dev seed 7 first then LB-known seed 44. Four pool lines dispatched
11:11 to pool40 (s7 x2) and wspool (s44 x2); shards bisect-head-off /
bisect-head-surface1. Static rule-out holds: the other three in-range path commits are
5ce76fdfd (provenance only), 5b5ea1b74 (congruence gate, default off),
42ba63177 (gated answer, default off).

## 2026-10-09 11:29 EDT — attempt 1 died: absolute cd; four lines re-queued
All four pool jobs ended rc!=0 within minutes. Cause: mkline expanded `~` under the
driver shell, embedding /opt/data/home/derisk-pool/revisions/<sha>; node homes are
/home/node (pool40/41/42) and /srv/simpool (wspool), so the cd failed instantly.
All 3030 historical pool lines use the literal `cd ~/derisk-pool/...` form; mine were
the only 4 absolute ones ever. Guard passes locally (rc=0). Corpus lives at
~/derisk-pool/corpus_extra/fineweb_edu.txt (4,326,798,376 B) on all 4 nodes, outside
revision dirs, so each line now also creates corpus_extra symlink if absent.
Re-queued all four (same tags/seeds/env, mem_gb=8, max_age_h=48) at 11:26; dispatched
11:27:57-11:28:18 (pool40 x2 = s7 pair, wspool x2 = s44 pair); pool depth 34 at add.
Node stderr (~/derisk-pool/sim/autodispatch.out) is not readable from this host (no pool
ssh config here; dispatcher runs remotely as user sim) - diagnosis is from path evidence.

## 2026-10-09 11:50 EDT — attempt 2 rc cause found; attempt 3 dev pair running; gate-seed canary launched

- **Attempt 2 (11:27) died at argparse, not at corpus.** Node stderr on pool40/wspool:
  `--seed: invalid int value: 's7'` (rc=2 = the fast-fail classifier's "bad command").
  The tag name (`s7`) was passed where the runner wants an int. The corpus `ln -s` also
  failed (`Read-only file system`: the pool sandbox mounts the revision read-only,
  writable only `research/findings/raw`) — but that was harmless noise: marker cells read
  `data/corpus/*` (simplewiki.txt + run3 frames, both provisioned in the revision's data/,
  same files the reference cells ran with); `corpus_extra/fineweb_edu.txt` belongs to the
  affect-vocab runner only and is not needed here (today's sandbox bind covers that one).
- **Attempt 3 = numeric seeds, no ln.** Both dev-seed-7 lines queued and dispatched:
  11:42:16 pool40 (control), 11:42:24 wspool (marker-surface=1); pool40 stderr shows
  `RNG initialized with seed: 7` + brain build → the line runs past the gate that killed 1-2.
- **Gate-seed pair (s44) blocked by the pool canary rule** (`pool_queue.sh add` refuses a
  42/43/44/100/101/102 line until a canary of the same runner+revision passed ON A POOL
  NODE). Launched `pool_canary.sh run auto '<s44 control line>' --wait-s 1500`
  (job `proc_eb33c86f20d7`); canary output is redirected to `_pool_canary/` by design.
  The s44 lines cite the covering prereg `2026-09-24-production-default-battery-B2b-PREREGISTRATION.md`
  — the same prereg historical marker-cell lines cite (provenance check: claims ledger).
