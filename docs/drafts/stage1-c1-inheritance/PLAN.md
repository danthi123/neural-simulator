# Stage-1 C1 (category held-out inheritance): the next method, a plan for a Claude build session (2026-10-04)

Stage-1 exit attempt 1 failed C1 with the capability absent on 6 of 6 seeds, on both routes, exactly as its prereg
predicted (section 8: "no inheritance route in the live handler"). The prereg's registered next step is: "wire
rung-3c's learned member->context pathway (default-off) into the strict route's recall miss; re-run this group at
dev seeds". This file makes that concrete. The build touches production code (`research/runners/one_brain_composer.py`
and/or `webapp/gnw_bus_shadow.py`), so it is Claude's. Hermes runs only the dev and gate lines it ends with. Stage-1
exit is due 10-24.

## What exists (read before building)

- `research/findings/2026-07-07-GNW-rung3c-learned-category-inheritance-6of6-heldout-5of6-identity.md` (6/6 GO on
  identity, held-out inheritance, permuted collapse and no-learning abstain; runner
  `research/runners/_gnw_rung3c_learned_inheritance_derisk.py`). Members and a `context` region sit on one bridge,
  and the only plastic pathway is member -> context (rate-Hebbian). During streaming both the member and its
  category's context are force-driven (`MEMBER_DRIVE_PA` 2500, `CONTEXT_DRIVE_PA` 2200). Igniting the held-out
  member drives its learned context, and the context drives the property the category shares.
- In production the strict route's recall for a never-told (agent, action) misses in organ A
  (`gnw_bus_shadow._bus_combine_inner`: `abstain_reason = "primary_recall_miss"`). The generative route's plausibility
  gate needs a co-occurrence that was never given.

## The build (default-off flag `BRAIN_CATEGORY_INHERIT`; byte-identical when unset)

1. **Learning during the teach turns, not a separate stream.** Each acquired fact (`ChatBrain._maybe_acquire` ->
   the composer's store) also drives, for a fixed window, the agent's member assembly together with the context code
   of that fact's (action, patient) pair. A plastic agent -> context pathway learns by the same on-bridge
   rate-Hebbian as rung-3c. The fact store stays as it is. The context region is shared by every fact, so members
   that share contexts converge (robin, finch and canary all share `fly-sky`).
2. **Use only on a primary recall miss, then return to the bus.** When organ A misses on (agent, action) and the
   flag is on, ignite the agent's member assembly. Read the context region it drives through the learned pathway,
   then read from that context the patient most strongly co-bound with the queried action (`eat` -> `worm` for the
   bird context). Hand that candidate to the same consensus ignition as any recall, so the bus commits or abstains.
   The reply is flagged as an inference ("the canary probably eats the worm"), never as a stored fact (the honesty
   boundary).
3. **No host selection.** The context read and the patient read are spiking reads (a soft-WTA over the context and
   patient populations), not a host argmax over the fact list. Any host step that remains is named in the finding's
   host-share table.
4. **Engagement counters** in the reply's `activity`: member -> context mean weight before and after the teach
   turns, the context rates at the miss, and whether the inheritance read ran.

## Controls (fixed before the dev run)

- **Identity:** taught members still answer their own facts. The flag must not change any answer that does not miss.
- **Permuted exposure (the gate's own arm):** the canary streamed with the fish context names `bug` or abstains,
  never `worm`.
- **lr = 0 arm:** with member -> context plasticity off, the canary abstains (the context was learned, not wired).
- **Lesion:** cutting the agent -> context pathway (a patch with a call count) removes the inherited answer.
- **Flag off:** byte-identical replies on the whole Stage-1 group battery (a test).

## Then

Dev seeds 1-8 with the gate's `cat_disc` and `cat_told` groups and the flag on, via the gate runner's shard mode at
the build's pin. Claude reads it. If it passes on at least 7 of 8, write Stage-1 Amendment A2 (C1 at the new pin,
flag on, all C1 arms), and Hermes queues the 60 C1 lines.
