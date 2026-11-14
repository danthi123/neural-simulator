# Pre-registration: the lived episode in the live 1.5B brain, unslept day (six seeds, pool)

Written 2026-11-14 before the run. Capability: episodic recall (answering from the listener's perspective).

**Record brief.** Oracle (`results/2026-11-14-lived-episode-1.5b.md`): storing the telling with the brain's own reply
halved echoes (13 -> 6) and raised held-out told 7 -> 10, but missed the dev bar (+7 < 10, CI -4..18): NOT A LEAD.
The oracle uses greedy replies and the right episode; live replies are sampled and routed by the S0 key, and in the
unslept live day S0 named the fact in 36 told replies and was credited 29 (`2026-11-14-key-default-1.5b.md`, A). This
is a separate test of the same format in the live path, not a re-scoring of the oracle.

**Prerequisites.** Met: 1.5B cortex, gain 0.5 default, the S0 default key, switch `Brain(episode="lived")` (default
OFF, `brain/tests/test_episode_switch.py`). Host decision: none (the stored reply is the brain's own live reply).
Untested here: sleep on lived episodes (a night replays the stored text).

**Method.** `brain/experiments/unslept_day_probe.py --arms c,s --episode tell|lived`, seeds 42/43/44/100/101/102, pool
CPU fp32 (both formats of a seed on the same node). Arm s = S0, gain 0.5 (the 1.5B default). Table
`brain/experiments/lived_live_table.py` (tested).

**Criteria (six-seed means, arm s).**
- P0 instrument: tell runs store no "Assistant:", lived runs store the exchange (recorded `stored`); and the tell run
  reproduces part A's arm s told count on every seed (3/2/6/6/6/6, same code path and seeds). Else VOID.
- P1 told (of 12): lived - tell >= 1, lived >= tell on >= 5/6 seeds.
- P2 false (never-told, of 12): lived <= tell + 0.5.
- P3 self (of 3): lived >= tell - 0.5.
GO = P0-P3 -> a mixed-day AWS run with sleep on lived episodes before any default flip.

**Expected.** P1 is uncertain (oracle +3 held-out told, dev +7 of 108). The routing key (S0, the stored text's layer-0
change) is computed from the longer lived text, so routing may move as well; that is part of the live effect.
