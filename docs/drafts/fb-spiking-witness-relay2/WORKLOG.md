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
Landing note (Claude review 2026-10-06): the three summaries are committed as `logs/BATTERY-*.txt` (`*.log` is
gitignored); the content is unchanged.

## 2026-10-06 13:2x — Claude review of items 1-2: the slot is KILLED at query time, by an instrument defect

The raw rows (`scans/out/relay_nmda_wb48_seed*_loc0.jsonl`, query phase, which the summary tables do not print)
contradict "never reaches a self-sustaining state". On all 8 seeds the written slot fires through the hold and the
first ~30 query steps (101-113 spikes in query steps 0-29). Then g_i on the slot jumps from ~2 to 326-395 nS at query
step 32-37. Slot mean V falls to a minimum of -40 to -44 mV (late read window mean -24 to -30 mV), and the slot stays silent through the late read window. In the same
window the OFF-target relays 1-3 fire 138-169 spikes each (hold: 0).

Cause, in code: `_run` adds `RELAY_BIAS_PA` to every relay cell on every step ("a cell property, the same every
step"), but `_query` zeroes the input in its own loop and never adds it. At query time the relays lose the -250 pA
bias and fire on presence alone. Each off-target relay drives its eviction pool, and `evict_j` inhibits every
belief slot except j, so slot 0 is evicted by relays 1-3. This explains the g_i surge with eviction itself never
recorded. Registered relay ops 0-3 (bias -100 to -300) share the defect; readout ops 4-7 do not (no relays).

Check (`scans/query_bias_check.py`, bias applied inside the query loop the way `_run` applies it; output
`logs/query_bias_check.txt`). AS-IS, the slot reads 0.0 on 8/8 seeds. With BIAS-IN-QUERY, slot 0 reads
0.024-0.039 with every other slot at 0 on 8/8 seeds, and the off-target relays fire 0. Battery on seed 7: 1/10
AS-IS, 4/10 with the bias (true_belief, return_then_move, two_moves_one_seen, present_after_wait pass).
false_belief, two_unseen_moves, absent_after_wait, return_after_wait, lesion_false_belief and silence_true_belief
still fail.

How to read it: the cases that now pass keep presence on through the query, so target relay 0 keeps firing
(221-257 query spikes) and keeps re-driving the slot. That is a live AND, not a held memory. The cases that need the
belief held with the relay silent (the agent gone) still fail, so defect 2 (attractor persistence without drive) is
still open. It is now isolated from the query artifact. In item 2, write spikes fall as wb rises. Over the second half of the write (seed 7), slot
mean V rises from -1.9 to +5.6, +11.3 and +10.6 mV, and g_e rises from 92 to 123, 180 and 333 nS, while spikes fall from 188 to 139, 54
and 16. More drive, more depolarized, fewer spikes: that pattern is the signature of depolarization block (over-drive), not of
a weak write. It is a reading of the traces and has not been tested.

Organ change: `QUERY_RELAY_BIAS` (default False, the registered behavior) applies the bias in `_query` for relay
modes; `tests/test_fb_relay_query_bias.py`. Item 3 waits for the re-measurement in the runbook ruling.
