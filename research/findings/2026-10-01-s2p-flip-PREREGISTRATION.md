---
type: preregistration
status: preregistration
date: 2026-10-01
packet: s2p-flip
tool: tools/s2p_flip.py
scorers: tools/iw2_score.py (step B), tools/s2p_score.py (step F1, --at-default)
---
# s2p-flip: the spiking paths become the iw2 faculties' defaults (PRE-REGISTRATION, 2026-10-01)

Written by the cloud Claude session 2 before any run. HANDOFF: `docs/drafts/s2p-flip/HANDOFF.md`. This is the step
that turns RULING `iw2-stage1-credit`'s PARTIAL into FULL for each packet that passes.

**Inputs (start conditions).** The iw2 flips are on main: steps B and F passed and Claude merged them (RULING
`iw2-flip-hold`). FLIP is the packets among fb-spiking-witness, transitive-spiking-reach and word-role-parse that
Claude confirmed with O PASS + R PASS. Each enters with the operating point its own `selection.json` chose.

**The flip (one scripted commit, G).** `python tools/s2p_flip.py apply <packet>=<op> ...` sets each packet's DEFAULT_ON
anchor to True and its DEFAULT_OP to <op>, and lifts its rows' REQUIRED_ENV. Nothing else changes.

**Step B (combined no-regression).** At G, run the full LB battery: one line per registry faculty and seed, made with
`research/packets/integration-wave2/make_jobs.sh flip <G> false-belief-chat,transitive-chat,neural-selfid` (the iw2
anchors are on at G). Score it with `tools/iw2_score.py B` against the iw2 flip battery at F as the base (tag
`iw2-flip-<F7>`, pin F). Its F1 reads re-check the three iw2 capability rows at G: each spiking path honors the iw2
row's lesion flag (false-belief: BRAIN_FALSE_BELIEF_LESION; transitive: BRAIN_TRANSITIVE_LESION; self-ID:
BRAIN_SUBSTRATE_PARSE_LESION, checked before the word-role read). **Bar: B PASS** under iw2's registered rules, the
iw2 HANDOFF section 7 attribution table included.

**Step F1 (the new rows at default).** For each packet in FLIP: `tools/s2p_score.py R --packet <p> --tag
iw2-flip-<G7> --pin <G> --selection <its selection.json> --at-default`, where the sidecars carry no packet flag.
**Bar: PASS on 6/6 with the control 6/6**, every step-R rule unchanged.

**Decision table.** B PASS and every F1 PASS: the flips stand, and the finding uses section 6 of each packet's prereg
plus "now the default path of <faculty>". An F1 FAIL for packet p: revert p (`s2p_flip.py revert p`), commit, and redo
B at the new pin. A B FAIL attributed to one packet: revert it and redo. A B FAIL not attributable: revert all, then
NEEDS-CLAUDE. UNDEFINED: one re-run per cell, then NEEDS-CLAUDE. Seeds 42/43/44/100/101/102 throughout.

**Not measured here (declared).** No identity arms (D/S): the sub-flags change only turns their faculty already owns.
Their scope was checked by each packet's step R control and by the iw2 step S, so they add no new turns. The
ledger rows' host-decision notes are Claude's at merge.
