# Pre-registration: speaker-contrast reinstatement (first-person echo), 1.5B oracle

Written 2026-11-14 before the run, after the research round the lever count required (3 echo levers failed their
bars, `results/2026-11-14-lived-*.md`). Capability: episodic recall from the listener's perspective.

**Record + research brief.** Local record: no speaker/perspective method in the new core beyond the lived formats;
old-track source monitoring (authorship tag GO). External: hippocampal neurons code a heard word in a blend of a
speaker-general and a speaker-specific subspace, with self vs other the most separated and names the most
differentiated (Chavez et al., 2025, bioRxiv); source monitoring framework (Johnson, Hashtroudi & Lindsay, 1993).
The stored episode (the change from reading "User: <s>") holds the fact and only a weak "the other said it" part;
reinstated, the cortex reads "my cat" as its own.

**Method.** Format contrast_mu: the reinstated change is d_user + mu (d_user - d_self), d_self = the change from
reading "Assistant: <s>" (the same sentence as the brain's own words) from the same base. The fact is kept (d_user);
only the speaker difference is amplified. mu in {0.5, 1} (`contrast50`, `contrast100`). Host decision: none (both
reads are the cortex's own; no text is added to the reply). `episode_format_probe.py`, gains 0.25/0.5/0.75/1, dev =
DEV_FACTS36, test = 12 FACTS, oracle routing, CPU fp32. dev tell is reused from `results/2026-11-14-lived-episode-1.5b/`
(deterministic; E0 checks it on test). pool40 dev contrast50, pool41 dev contrast100, pool42 test tell + both.

**Criteria.** Variant = the mu whose dev right at its own G* is higher (tie -> 0.5). Then E0-E3 exactly as
`2026-11-14-lived-episode-1.5b.md` with that variant in place of lived (`episode_format_table <dir> <variant>`).
LEAD -> the contrast in Brain.reinstate behind a switch (default OFF), live six-seed test.

**Expected.** Uncertain; mu = 1 may also push the cortex away from the fact. Echoes should fall if the speaker part is
what the cortex misreads.
