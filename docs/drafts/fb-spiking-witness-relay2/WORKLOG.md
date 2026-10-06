# fb-spiking-witness-relay2 — WORKLOG

All runs on dev seeds only (DIAGNOSIS header), local CPU numpy; no organ, scorer, bar or default edited — the scan
scripts' own contract (DIAGNOSIS line 3).

## 2026-10-06 12:35-12:45 EDT — DIAGNOSIS "Next measurement" items 1 and 2 (driver)

Instrument: new `scans/measure_hold.py` extends the per-step measurement through HOLD_STEPS (belief / relay spike
counts per slot, mean membrane voltage mV, g_i/g_e, phases pres_read/write/hold labelled from the `_run` call
sequence, final query). `scans/and_scan.py` gained a `wb` argument (default 48) plus a printed
`wb_applied={g['W_RELAY_BELIEF']}` proof. Raw per-step rows: `scans/out/*.jsonl` (19 files); summaries:
`BATTERY-seed7plus8.log`, `BATTERY-wbtraces.log`, `BATTERY-andscan.log`.

### Item 1 — slot trajectory write → hold → query, seeds 1-8 (wb 48, defect-1 OP)

| seed | relay write spk | relay hold spk (per-10-step) | relay query rate | readout write spk | readout hold spk | readout query rate |
|---|---|---|---|---|---|---|
| 1 | 191 | 55 [0,30,25] | 0.0 | 1903 | 0 | 0.0587 |
| 2 | 187 | 61 [1,32,28] | 0.0 | 1864 | 1 | 0.0614 |
| 3 | 196 | 67 [2,32,33] | 0.0 | 1989 | 0 | 0.0689 |
| 4 | 184 | 49 [1,26,22] | 0.0 | 1877 | 0 | 0.0519 |
| 5 | 184 | 48 [1,22,25] | 0.0 | 1906 | 1 | 0.0580 |
| 6 | 183 | 49 [1,19,29] | 0.0 | 1867 | 0 | 0.0674 |
| 7 | 188 | 44 [1,22,21] | 0.0 | 1887 | 0 | 0.0580 |
| 8 | 189 | 42 [1,17,24] | 0.0 | 1930 | 0 | 0.0621 |

Read as: the relay-mode slot ignites in write (183-196 spikes, target relay 94 in write / 78 in hold — the
registered OP relay numbers), ramps LATE in hold (first 10 steps 0-2 spikes, last 20 steps 17-35), then reads 0.0
at query in 8/8 seeds — it never reaches a self-sustaining state in the free window. The readout-mode slot takes
the 5000 pA helper in write (1864-1989 spikes) and collapses at hold onset (0-1 hold spikes; seed-7 voltage
15.51 -> -0.78 mV across the 25 hold steps), leaving a faint 0.052-0.069 residue at query in 8/8 seeds. No
mid-write dip appears in either mode (no mid-write kill signature in these traces); the eviction pop itself is not
exported in `idx` (keys: belief/self/world/relay/presence/evs only), so eviction spikes cannot be recorded without
an organ edit — noted as an instrument limitation, not worked around.

### Item 2 — the write strengthened (wb 48 / 100 / 200 / 400, seed 7)

| wb | belief write spk | relay write spk | belief hold spk (per-10) | relay hold spk | query rate |
|---|---|---|---|---|---|
| 48 | 188 | 94 | 44 [1,22,21] | 78 | 0.0 |
| 100 | 139 | 94 | 12 [2,4,6] | 78 | 0.0 |
| 200 | 54 | 94 | 21 [7,6,8] | 78 | 0.0 |
| 400 | 16 | 94 | 14 [6,6,2] | 78 | 0.0 |

`and_scan` re-run at each wb: all 16 bias/wp/ww rows are identical across the four wb values (verified
`wb_applied=400.0` printed on all 16 rows of the wb-400 run) — the AND discriminator lives entirely upstream of
W_RELAY_BELIEF (no belief -> relay feedback in the wiring), so the registered AND rows hold unchanged while the
belief-side write spikes fall 188 -> 16 as wb rises and the query reads 0.0 at every wb. Strengthening the write
therefore does not move the slot toward persistence in this organ.

Next (DIAGNOSIS item 3): Claude picks the mechanism at the OP; reported via NEEDS-CLAUDE [fb-relay2].