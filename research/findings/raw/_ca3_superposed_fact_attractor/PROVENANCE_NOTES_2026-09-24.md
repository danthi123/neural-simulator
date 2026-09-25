# Provenance evidence for the 54-cell CA3 superposed-fact-attractor grid (2026-09-24)

Captured while scoring `research/findings/2026-09-24-ca3-superposed-fact-attractor-capacity-GO-6seed-reruns-provenance-fixed.md`.
`research/queue/dispatch.log` and `research/queue/pool_sync.log` are live, un-tracked queue-state logs (not
committed to git), so this file snapshots the lines this finding's provenance claim rests on, in case either log
later rotates or is truncated.

## `dispatch.log`: every CA3 dispatch line (54 total, all at revision `ab2adcf51bb452f322fabe94b5668335c7d687a7`)

Extracted with `grep -i "ca3_superposed" research/queue/dispatch.log`, then reduced to timestamp / pool / revision
/ arm / seed / out-dir. 36 lines target `.../grid` (seeds 42/43/44/100, the original dispatch window,
09:50:36-11:10:56 on 2026-09-24); 18 lines target `.../rerun_s101_s102` (seeds 101/102, the rerun window,
22:55:18-23:44:17 the same day, split across pool1 and pool42). No other CA3 dispatch lines exist in the log.

```
09:50:36 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg seed=42 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
09:50:43 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg seed=43 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
09:50:50 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg seed=44 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
09:50:56 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg seed=100 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
09:51:03 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_c2 seed=42 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
09:54:34 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_c2 seed=43 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
09:55:49 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_c2 seed=44 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
09:55:56 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_c2 seed=100 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
09:56:02 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_recx2 seed=42 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:01:49 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_recx2 seed=43 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:03:05 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_recx2 seed=44 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:04:20 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_recx2 seed=100 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:07:52 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=dense_nodg seed=42 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:09:07 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=dense_nodg seed=43 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:09:14 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=dense_nodg seed=44 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:09:21 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=dense_nodg seed=100 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:10:36 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_hub seed=42 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:16:22 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_hub seed=43 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:18:46 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_hub seed=44 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:20:01 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_hub seed=100 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:20:08 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=dense_nodg_hub seed=42 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:21:23 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=dense_nodg_hub seed=43 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:22:39 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=dense_nodg_hub seed=44 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:22:45 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=dense_nodg_hub seed=100 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:36:42 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_nodg_hub seed=42 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:59:35 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_nodg_hub seed=43 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:59:41 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_nodg_hub seed=44 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
10:59:48 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_nodg_hub seed=100 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
11:02:10 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_c2_hub seed=42 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
11:02:17 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_c2_hub seed=43 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
11:03:32 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_c2_hub seed=44 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
11:04:48 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_c2_hub seed=100 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
11:04:55 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_bounded_hub seed=42 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
11:08:25 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_bounded_hub seed=43 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
11:09:40 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_bounded_hub seed=44 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
11:10:56 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_bounded_hub seed=100 out=research/findings/raw/_ca3_superposed_fact_attractor/grid
22:55:18 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=dense_nodg seed=101 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
22:58:53 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=dense_nodg_hub seed=101 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
23:22:06 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg seed=101 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
23:22:13 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_bounded_hub seed=101 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
23:23:27 pool42 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_c2 seed=101 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
23:23:33 pool42 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_c2_hub seed=101 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
23:23:39 pool42 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_hub seed=101 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
23:23:46 pool42 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_recx2 seed=101 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
23:26:17 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_nodg_hub seed=101 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
23:27:36 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=dense_nodg seed=102 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
23:31:16 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=dense_nodg_hub seed=102 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
23:33:45 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg seed=102 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
23:37:25 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_bounded_hub seed=102 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
23:37:32 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_c2 seed=102 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
23:37:39 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_c2_hub seed=102 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
23:39:04 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_hub seed=102 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
23:42:52 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_dg_recx2 seed=102 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
23:44:17 pool1 rev=ab2adcf51bb452f322fabe94b5668335c7d687a7 arm=sparse_nodg_hub seed=102 out=research/findings/raw/_ca3_superposed_fact_attractor/rerun_s101_s102
```

Cross-checked: `research/queue/pool.queue`, `pool.queue.unchecked` and `gpu.queue` hold zero CA3 lines (nothing
pending); `pool.queue.done` / `pool.queue.running` hold zero CA3 lines either (nothing mid-flight, unresolved).

## `pool_sync.log`: files pulled from the pool nodes at the same revision, during the same two windows

Extracted with `grep -n "ab2adcf51" research/queue/pool_sync.log`. This is an independent log (written by the
harvester that pulls completed files back from the pool nodes into this checkout's `research/findings/raw/`), so
it corroborates the dispatch log rather than repeating it: files were actually observed landing FROM the pinned
revision on pool1/pool42, not merely requested.

```
8288:  pool1:ab2adcf51bb452f322fabe94b5668335c7d687a7/research/findings/raw: 9 file(s)
8296:  pool1:ab2adcf51bb452f322fabe94b5668335c7d687a7/research/findings/raw: 13 file(s)
8304:  pool1:ab2adcf51bb452f322fabe94b5668335c7d687a7/research/findings/raw: 11 file(s)
8312:  pool1:ab2adcf51bb452f322fabe94b5668335c7d687a7/research/findings/raw: 5 file(s)
8320:  pool1:ab2adcf51bb452f322fabe94b5668335c7d687a7/research/findings/raw: 3 file(s)
8330:  pool1:ab2adcf51bb452f322fabe94b5668335c7d687a7/research/findings/raw: 11 file(s)
8338:  pool1:ab2adcf51bb452f322fabe94b5668335c7d687a7/research/findings/raw: 2 file(s)
8897:  pool1:ab2adcf51bb452f322fabe94b5668335c7d687a7/research/findings/raw: 2 file(s)
8912:  pool1:ab2adcf51bb452f322fabe94b5668335c7d687a7/research/findings/raw: 3 file(s)
8920:  pool42:ab2adcf51bb452f322fabe94b5668335c7d687a7/research/findings/raw: 4 file(s)
8925:  pool1:ab2adcf51bb452f322fabe94b5668335c7d687a7/research/findings/raw: 10 file(s)
8934:  pool42:ab2adcf51bb452f322fabe94b5668335c7d687a7/research/findings/raw: 3 file(s)
8938:  pool1:ab2adcf51bb452f322fabe94b5668335c7d687a7/research/findings/raw: 7 file(s)
8949:  pool42:ab2adcf51bb452f322fabe94b5668335c7d687a7/research/findings/raw: 1 file(s)
```

Lines 8288-8338 fall in the original (36-cell) dispatch window; lines 8897-8949 fall in the rerun (18-cell,
seeds 101/102) window, and are the first `pool42:ab2adcf51...` lines in the whole log -- consistent with pool42
only joining the CA3 job at 23:23 for the rerun's 4 arms (dispatch lines above). Sync counts do not need to sum
to exactly 36 or 18 (the harvester reports incremental deltas, including partial/retry states), but the presence
of `ab2adcf51...` traffic from both pool1 and pool42 in exactly these two windows, and nowhere else in the file,
corroborates the dispatch log.

## mtime cross-check (the original FAILURE_LOG signal)

The 18 files this finding EXCLUDES (`grid/{arm}_s101.json`, `grid/{arm}_s102.json` in the primary checkout) share
one identical mtime, `1790255323` (2026-09-24 09:08:43 EDT) -- about 42 minutes BEFORE the earliest verified
`grid/` dispatch line above (09:50:36) and 13h47m before the rerun dispatch window (22:55:18) even opened. The 36
included `grid/` files and the 18 included `rerun_s101_s102/` files each have distinct, staggered mtimes
consistent with real per-job runtimes (minutes, matching the dispatch-to-dispatch gaps above). Re-verified in
this session with `stat -c '%Y %n' .../grid/*.json | sort` on the primary checkout.

## Sanity check on the excluded files (informational only, not provenance)

For every one of the 18 excluded `grid/*_s101.json` / `*_s102.json` files, the `summary.recall` /
`recall_recent` / `recall_rec_zero` / `recall_rec_shuffle` arrays are IDENTICAL to the corresponding (included)
`rerun_s101_s102/` file for the same arm/seed, and `summary.dprime` agrees to 5-6 significant figures (small
float differences consistent with normal run-to-run FP accumulation order on the numpy backend, not a different
computation). This suggests the excluded files were very likely genuine runs of the correct code at the correct
seed -- but genuine-looking is not provenance, and they remain excluded from this finding's scoring because
neither log above can place them.
