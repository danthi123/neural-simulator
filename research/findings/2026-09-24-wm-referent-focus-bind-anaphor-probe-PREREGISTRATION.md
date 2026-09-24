---
type: finding
status: live
date: 2026-09-24
lane: load-bearing
mechanism: PRE-REGISTRATION of BRAIN_MULTIREF_FOCUS_BIND (default OFF) -- a spiking referent->focus binding for the D6 multi-referent working-memory organ, in which the organ's held state persists across turns and an anaphor retrieves one held register by a lateral-inhibition competition -- and of its measurement LB_WMB_FOCUS_PROBE, which asks whether an ORDINARY reply follows WHICH referent the organ holds
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. No evaluation seed has been built. No result is claimed here.
runner: research/runners/load_bearing_fraction.py
artifacts:
  - research/findings/raw/_wm_focus_bind/calib_dev_seeds.json
---

# wm-binding: a referent->focus binding and an anaphor probe that can fail, PRE-REGISTRATION (2026-09-24)

Branch `research/wm-referent-focus-bind`, cut from main `ee73ecd4f` (the pinned pre-change SHA for the flags-off
check). Mechanism `f4117ebd6`; gate + dev calibration `385c94192`; one build per session `732aed378`; the 2/3-organ
GNW buses honor an organ-resolved anaphor `09f8e6009`; the gate decides on reply content `df2bd4f6a`.

## Why

The ordinary-content probe read 6 clean negatives on 6 seeds
(`research/findings/2026-09-24-wm-binding-ordinary-content-probe-6seed-NOGO-held-state-does-not-reach-an-ordinary-reply.md`).
Its prereg (`research/findings/2026-09-24-wm-binding-ordinary-content-probe-PREREGISTRATION.md`, AMENDMENT B)
gives the reason: the WM focus handed to comprehension is the positional `CAND_POOLS[0]`, set on the intro whatever
referent is held. No field that carries WHICH referent is held is read on the path to an ordinary reply. Across turns,
the held referents live in the host dict `_slot_of_ref`, and every load resets the buffer. The prereg names the next
mechanism: a spiking referent->focus binding, with a spiking pronoun-resolution read-out, so that an ordinary answer
depends on which referent the buffer holds. This document registers that mechanism and a probe for it.

## The mechanism (`BRAIN_MULTIREF_FOCUS_BIND=1`; unset -> byte-identical)

Code: `research/runners/d6_multiref_wm_production_organ.py` (block "SPIKING REFERENT -> FOCUS BINDING"), wired in
`webapp/server.py` (D6 block), `research/runners/brain_chat_tui.py` (`ChatBrain._resolve_anaphora`) and
`webapp/gnw_bus_shadow.py`. Biology: `research/biology/wm-referent-focus-retrieval.md`.

1. **The hold persists across turns.** After a >=2-referent load, the organ keeps this session's slice state: the
   per-neuron membrane, recovery, conductances (including the slow-NMDA recurrence), firing and refractory arrays.
   On a later turn it writes that state back and runs a 100 ms zero-input span. A live bump keeps firing; a dead one
   stays dead. What is held is read off that state. The host `_slot_of_ref` is no longer read to decide WHICH
   referents are held across turns. It stays only as the binder's codebook for the next write.
2. **An anaphor retrieves one register by competition.** The ChatBrain's own spiking CA3 anaphor detector finds the
   pronoun. The organ reads each register's held firing over a 40 ms zero-input window. Each register's rate drives
   its own assembly of a 5-way lateral-inhibition WTA: each assembly has its own fast-spiking sub-pool that inhibits
   the others (`_affect_marker_wta_derisk._build_bridge`, reused, as the spiking question-route selector reuses
   it). The anaphor adds a common 150 pA drive, which is sub-threshold alone. The assemblies race for 150 ms. The
   winning assembly names the register; the register's live local slot names the referent.
3. **The resolved referent reaches the ordinary reply.** `ChatBrain._resolve_anaphora` substitutes it for the pronoun
   in this turn's question. The ordinary recall then runs on the substituted question. A substrate miss on an
   organ-resolved referent abstains. It does not take the host keyword router's rescue, which would answer about a
   different agent. That is the rank-13 neural-abstain behaviour, applied only when the organ resolved the pronoun,
   in every gate combiner that has the rescue (`ChatBrain.gate`, the substrate bus, the 2- and 3-organ buses).
4. **The focus becomes the winning register.** Comprehension's `wm_focus` becomes the winning pool, but only if the
   d6->comprehension cross-edge spans it (`w0..w2`); otherwise it is None. When no bump is live it is None. It is no
   longer `CAND_POOLS[0]` whatever is held.
5. **The hold-query reads the live state.** With the flag on, "who are we talking about" reads the registers off
   the resumed state. It no longer re-loads the referents from `_slot_of_ref`.

## Declared residual shortcuts

- The register-to-assembly projection crosses two bridges as a host rate relay (rate x 20000 pA -> current).
- The final read of which WTA assembly won is an argmax over its settled rates, with a 0.05 rate floor and a 0.05
  dead margin. This is the same read-out as the question-route and affect-marker WTAs. A tie resolves nothing.
- The winning register's local slot is the organ's existing `read()`-class argmax over its bank.
- The slot is named by the host RUNG6c binder codebook (`_ref_of_slot`). The binder is content-agnostic: the first
  referent mentioned gets local slot 0 in register 0, the second local slot 1 in register 1. So two sessions that
  hold the same two referents in swapped order have the SAME spiking state. Only the codebook names differ. The
  organ's spiking contribution is which slot is live and which register wins; the word comes from the codebook.
- Referent extraction is the host lexicon. The pronoun-for-referent string substitution is host.
- The inter-turn interval is compressed to 100 ms.
- The per-session state stash exists because one shared slice serves every session. The stash is the substrate
  state and carries no referent label.
- Which of two equally-matched held referents wins is set by the pools' intrinsic excitability. Discourse salience
  (subjecthood, recency, topic) is not represented. A Centering preference is the next rung. This matches the
  interference case of cue-based retrieval (Lewis & Vasishth 2005, doi:10.1207/s15516709cog0000_25 <!--derived-->).
- The cross-edge onto comprehension spans only `w0..w2`, so a referent held in any other pool gives no focus.
- With the flag on, the hold-query path does not apply the curiosity->d6 semantic-drop drive (default OFF).

## The probe (`LB_WMB_FOCUS_PROBE=1`; label-only turns, not in the default roster)

Each pair is two sessions. Both introduce the same two lexicon referents, in swapped order. Then both ask the same
ordinary anaphor question.

| pair | session | intro | ask |
|---|---|---|---|
| A | `wmfa1` | the dog and the cat walked in | what does it chase |
| A | `wmfa2` | the cat and the dog walked in | what does it chase |
| B | `wmfb1` | the cat and the bird walked in | what does it eat |
| B | `wmfb2` | the bird and the cat walked in | what does it eat |

The tiny-demo brain knows `dog chase cat` and `cat eat fish` from build. Within a pair the words are the same and
only their order differs, so anything that ignores word order cannot tell the sessions apart. The mention order
decides which referent sits in the register that wins the retrieval. An organ whose held content reaches the reply
must answer the two sessions differently; a positional route cannot. Order-sensitive host paths can also tell the
sessions apart (each intro is itself acquired as a 3-word parse). The lesion arms control for them: a difference
that survives the hold lesion is not the organ's.

Arm kinds per seed: ON = {intact `a`, intact rebuild `b`, lesion, lesion rebuild} with `BRAIN_MULTIREF_FOCUS_BIND=1`;
OFF = the same four with `BRAIN_MULTIREF_FOCUS_BIND=0`. Every arm sets `BRAIN_MULTIREF_LESION_SCOPE=recur`. The lesion
arms add `BRAIN_MULTIREF_LESION=1`: the confined lesion that zeroes only the organ's `w_k->w_k` slow-NMDA synapses on
the shared slice. Each arm kind is built once per SESSION (intro then ask, in a fresh process), so the two sessions of
a pair share no process state and differ only in the intro's mention order. That makes 8 x 4 = 32 builds per seed.
(A 4-session build was tried first and passed 15 GB on numpy: every session builds its own ChatBrain and composer.)
The runs use the numpy backend. The seed is set by `--seed`, which sets `BRAIN_CHAT_SEED`.

## Pre-registered decision rule (per seed; `_wmf_gate` + `_wmf_headline`, selftested in both directions)

The ON arms are evaluated per pair p in {A, B}. Conditions are checked in this order. The first one that fails names
an UNDEFINED verdict, which is never a pass and never a negative.

1. **build**: every arm built, with no per-turn `_error`. Otherwise `arm-build-failed`.
2. **R1**: both intros are in scope on every arm (`multiref.kind == "maintain"`, `n_referents == 2`). The lesion arms
   must also carry `lesion_scope == "recur"`. Otherwise `probe-inadequate:route`.
3. **R2**: both asks are ordinary on every arm: no `inner_state_readout`, no multiref hold-query, and an `answer`.
   Otherwise `probe-inadequate:not-ordinary`.
4. **RES**: on both intact arms, each ask was resolved by the organ (`multiref.kind == "resolve"`) to one of its own
   session's referents. Otherwise `probe-inadequate:no-resolution`. The two sessions of the pair must resolve
   DIFFERENT referents. Otherwise `probe-inadequate:same-referent`.
5. **L**: on both lesion arms, the intros read `hold_alive_min == 0.0`, and the lesion still holds at the ask: the
   ask's own retrieval record reads 0.0 on every register. Otherwise `lesion-not-effective`.
6. **N**: intact `a` == intact `b` on every ask's reply content. Otherwise `noisy-null-control`.
7. **C**: each intact reply follows its resolution: `recalled_svo` is None (an abstain), or its agent is the resolved
   referent. Otherwise `probe-inadequate:content`.
8. **R**: lesion == lesion rebuild on every ask's reply content. Otherwise `noisy`.

**T_p** = the pair's two intact replies differ. **X_p** = the pair's two LESION replies are identical.

"Reply" here is the reply's CONTENT: `recalled_svo` (the fact it states) and `abstained`. It is not the surface
string. Each intro is itself acquired as a junk 3-word parse ("the dog cats the walked" vs "the cat dogs the
walked"), and that changes how the same fact is rendered. On dev seed 7 with the flag OFF, the two sessions of pair A
replied "the dog chases the cat" and "The dog chases cat.": the same fact, different strings. On the surface string
the positional route would read as differing, and the probe would go UNDEFINED on a rendering artifact that the
organ never touches. The surface difference is recorded, report-only (`T_answer_surface_differs`,
`X_answer_surface_same`).

| outcome | load_bearing | verdict |
|---|---|---|
| every pair T and X | true | regressed (the reply follows the held referent; killing the hold removes it) |
| no pair T | false | pass (a real negative) |
| some pair T but not X | null | off-organ-route (the difference survives the hold lesion) |
| otherwise | null | content-dependent-effect |

**The failing direction.** The OFF arms go through the same gate without RES and C, which are properties of the ON
route. If the OFF gate reads `load_bearing = true`, a positional route passes this probe and the seed's verdict is
`probe-inadequate:positional-passes` (null), whatever the ON gate says. The record carries both gates and
`failing_direction_ok`.

The decision fields are `recalled_svo` and `abstained` only. Report-only per arm and turn (`wmf_mechanism`):
`answer`, `multiref.kind/resolved/resolved_register/resolved_pool/margin/register_rates/wta_rates/hold_alive_min`,
`inner_state_readout`.

## Headline rule (6 seeds: 42 43 44 100 101 102)

- **GO**: `load_bearing = true` on 6 of 6 seeds. It means only this: with `BRAIN_MULTIREF_FOCUS_BIND=1`, the organ's
  held state decides which referent an ordinary anaphor reply is about; killing the hold removes that; and the
  positional route does not pass. It does not mean the faculty is on by default, that referent identity is in the
  substrate (the codebook names the slot), that the pronoun preference is linguistically right, or that the path is
  fully spiking.
- **PARTIAL**: 4-5 of 6 true. Reported per seed; not a GO.
- **NO-GO**: 3 or fewer true. If 4 or more seeds read `pass`, the verdict is: *with the focus binding on, the ordinary
  reply still does not follow the held referent.*
- UNDEFINED on 3 or more seeds means the probe is inadequate. The reasons are reported, and no claim is made either
  way. `probe-inadequate:no-resolution` on a seed means the retrieval tied there (see "Seen before").

Counting: the record uses its own key, `wm-binding-referent-focus`, with kind `neural-lesion-opt-in`. It measures a
default-OFF mechanism, so `run()` leaves it out of the production load-bearing fraction. A GO here does not change
wm-binding-advanced's production row. Flipping the flag on needs its own verification.

## Seen before this was written (declared)

prereg-same-commit: the dev-seed-7 full-probe-shaped calibration artifacts under `research/findings/raw/_load_bearing/wmb_focus/dev_s7/` land in this same commit; they are declared PRE-existing inputs this document reports on, not output of the run this prereg governs (that run has not started; seed 7 is dev-only and excluded from the 6 evaluation seeds). (The Compute section's earlier reference to timing/RSS figures from this same calibration run was removed in the AMENDMENT LOG's documentation fix round -- those figures were never written to a committed artifact.)

- Dev-seed calibration, private organ, seeds 7/11/13/17/19/23/29/31:
  `research/findings/raw/_wm_focus_bind/calib_dev_seeds.json`. At the shipped operating point (gain 20000 pA, cue
  150 pA, 150-step race), a live 2-referent buffer resolved on 8/8, with a minimum WTA margin of 0.087778. The same
  register won in both mention orders on 8/8. Both referents were still held after the retrieval on 8/8. An empty
  buffer and a recur=0 buffer resolved on 0/8. Four other settings were tried; they resolved on 6/8 or 7/8.
- Before the WTA: a design that used the D6 bank's own shared FS as the competition was measured on dev seed 7 and
  dropped. FS drive of 300 or 600 pA scaled both held bumps down together, keeping the ratio, because the bank is
  built to hold several bumps without cross-talk (scratch, not committed).
- Exploratory flag-OFF runs on main's code (scratch, not committed; the session was killed before it finished):
  "the dog and the cat walked in" -> "what does it chase" -> "the dog chases the cat";
  "what does dog chase" -> "As for it — the dog chases the cat";
  "what does cat chase" -> "Setting the held thread aside — On cat, then — I don't know about that."

## Byte-identity (flags OFF)

Asserted in data before this filing: `research/findings/raw/_wm_focus_bind/offflag_byte_identity_s7.json` reports
`byte_identical_off: true`. The check (`research.runners._wmf_offflag_byte_identity`) runs the real `brain_chat` in
an extracted tree at the pinned main `ee73ecd4f` and in one at the branch `09f8e6009`, with every new flag unset,
`BRAIN_CHAT_SEED=7`, numpy, one session per process. It covers the D6 maintain load plus the held transitive that
reads the xedge focus (`bi_hold`), a 2-referent intro followed by an anaphor through `_resolve_anaphora` and the GNW
buses (`bi_bc`, the probe's turn shape), and the hold-query read-out (`bi_wmb`). The sha256 of every response was
identical, 6 of 6. `PROBE_TURNS`, `FACULTY_PROBES` and every pre-existing turn label hash identically; 8 labels were
added. `git diff 09f8e6009..HEAD` over `webapp/`, `sim/`, the d6 organ, `brain_chat_tui.py` and the battery is empty.
Raw per-tree responses: `research/findings/raw/_wm_focus_bind/byte_identity_s7/`.

## Compute

One pool line per seed, from a revision provisioned at the pushed branch head. Each line runs 32 builds in sequence,
one session each (numpy, `OMP_NUM_THREADS=1`). An earlier draft of this section stated specific peak-RSS and
per-build-duration figures for the dev-seed-7 calibration run; those were an unlogged live `ps`/`top` observation,
not a number written to any committed artifact, and traced to no artifact when checked -- see the AMENDMENT LOG.
No RSS or duration figure is claimed here. `mem_gb=7` is a conservative operational budget carried over from that
unlogged observation, not a measured/citable one; neither `load_bearing_fraction.py` nor its provenance sidecar
instruments per-build RSS or wall time, so no committed artifact retroactively supports a specific figure either --
a citable measurement would need a separate, purpose-built instrumentation pass, not attempted here.

```
bash tools/pool_queue.sh add 'cd ~/derisk-pool/revisions/<pushed head> && SIM_BACKEND=numpy OMP_NUM_THREADS=1 LB_WMB_FOCUS_PROBE=1 .venv/bin/python -u -m research.runners.load_bearing_fraction --only wm-binding-advanced --repeats 2 --seed <s> --out research/findings/raw/_load_bearing/wmb_focus/s<s>/lbf.json' --checked 'wm referent->focus bind prereg research/findings/2026-09-24-wm-referent-focus-bind-anaphor-probe-PREREGISTRATION.md; the ordinary-content probe read 6/6 clean negatives and names this mechanism; flags-off byte-identity in data (offflag_byte_identity_s7.json); mem_gb=7'
```

for `<s>` in 42 43 44 100 101 102.

## Honesty

Functional read-outs only. "Holds", "retrieves" and "resolves" name the organ's measured firing and the chat
output. No felt state is asserted.

## AMENDMENT LOG

(empty at filing)

- **2026-09-24, documentation fix round, before any evaluation seed was built (still zero -- 6-seed battery queued
  on the pool but not pulled).** An independent review of this filing found three document defects, none touching
  the pre-registered mechanism, probe, decision rule or headline rule:
  1. The `Biology:` line cited `research/biology/wm-referent-focus-retrieval.md`, which did not exist. It now
     does, binding HOLD to persistent prefrontal delay-period firing (Kandel PNS-6e) and RETRIEVE to the project's
     own established N-way lateral-inhibition WTA primitive (`affective-marker-lateral-inhibition-wta.md`,
     already reused once by `question-route-selection-wta.md`), and declaring the discourse-salience residual
     (Lewis & Vasishth 2005) as NOT yet implemented. `tools/biology_check.py` passes on it (4/4 sources resolve).
  2. The Compute section stated specific peak-RSS/per-build-duration figures that traced to no committed
     artifact. Corrected above: no specific figure is claimed; `mem_gb=7` is disclosed as an unlogged operational
     choice, not a measurement.
  3. A leftover `DEV_SMOKE_PLACEHOLDER` line in "Seen before this was written" (empty template content) is
     removed.
  Nothing above changes the mechanism code, the probe's turn shapes, the pre-registered decision rule (`_wmf_gate`
  section) or the headline rule -- `git diff 4da72fd23..HEAD -- research/runners/d6_multiref_wm_production_organ.py
  webapp/server.py research/runners/brain_chat_tui.py webapp/gnw_bus_shadow.py webapp/gnw_two_organ_bus.py
  webapp/gnw_three_organ_bus.py research/runners/load_bearing_fraction.py` is empty, so the 6-seed pool jobs already
  queued and pinned to `4da72fd23` still measure the correct, unchanged code and were left in place rather than
  requeued.
