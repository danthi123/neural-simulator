# bg-tonic-stn-1019 — STN pacemaking drive sweep: PROVISIONAL RESULT

**Status:** PROVISIONAL (agent-filed, needs Claude verification). Per ruling bg-tonic-stn-1019,
the downward sweep as scoped CANNOT satisfy the pick rule; STOP per ruling — variance RESULT.

## Mechanical verdict

No current in {40, 30, 25, 20, 15, 10} pA has all 8 seeds (s9–s16) in the 10–30 Hz STN band.

- **40 pA:** 6/8 seed means in band; s12 = 7.3 Hz (BELOW), s14 = 32.7 Hz (ABOVE). Strict
  all-events-in-band fails more seeds.
- **30 / 25 / 20 / 15 pA:** all 8 seeds silent (stn30 means 0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0,
  0.2 Hz; stn25/20/15 all 0.0), 8/8 cells complete each.
- **10 pA:** 6/8 cells complete, all 0.0 Hz (s9–s14); s15 at 6/7 jsons, s16 at 4/7 in flight.
  These 2 unfinished cells cannot change any current's status.

## Variance characterization

At 40 pA the seed means span 7.3–32.7 Hz: the seed-variance problem is UNCHANGED by lowering
drive. Lowering drive does not taper the rate — it silences STN entirely by 30 pA (0.0 Hz, GPe
unaffected at 27–38 Hz), i.e. a cliff between 30 and 40 pA. Cross-check: GPe keeps firing in
every zero-STN cell (e.g. stn{30,25,20,15,10}/s9 = 36.46/37.66 Hz, /s13 = 27.45/28.49, /s16 =
36.15/33.59, identical per seed across currents with GPe held at 100 pA), so the cells RAN; STN
silence is genuine, not a dead measurement. Dev/persist reference (default config, seeds 10-16):
STN ranged 14–42.83 Hz (6/8 above band) — the variance that started this sweep.

## Open question (not a claim)

Is the 30–40 pA cliff biological (rheobase), or an artifact of how the sweep sets the drive? Not
asserted here.

## Continuation

The 32 queued points at 32/34/36/38 pA × seeds 9–16 (already on the pool, revision
8177576ae, no bars) straddle the cliff and decide whether any single current lands all 8 seeds
in band. Re-picking / re-designing remains Claude's call (already flagged on NEEDS).

## Preconditions

```
ruling: bg-tonic-stn-1019
revision: 8177576ae802f52be5bfa477b134fbe3b67871b6
seeds: 9-16
band_hz: 10-30
gpe: held at 100 pA
bars_read: none
data_dir: research/findings/raw/_bg_live_seam/stn_sweep/8177576/
extraction: 2026-10-20 13:19 EDT
cells_complete: 46/48 (stn10 s15/s16 in flight)
```
