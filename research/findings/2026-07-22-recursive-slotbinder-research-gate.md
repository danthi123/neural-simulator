# RESEARCH-GATE — Recursive slot-binding: fully retiring the FHRR exact-inverse algebra (2026-07-22)

Read-only deep-research gate (the standing "before-a-new-mechanism-class" step). Decides whether/how to give the
`SlotBinderComposer` recursive fillers — embedded clauses + attributed patients — so the FHRR/VSA exact-inverse algebra
can be fully retired (the project's #1 documented idealization shortcut, flagged by the 2026-07-22 field-novelty
assessment). Controller-reviewed; the load-bearing claims (the slot-binder `store()` flat-only guard, the FHRR
single-attribute + depth-1 deployed scope, the parser depth-1 abstain) were verified against the code.

## VERDICT UP FRONT
- **Attributed patients (single-attribute) = SURPASSABLE-AND-CHEAP, and it is NOT recursion.** One extra slot (a 5th
  `attribute` role). ~1-day de-risk, near-certain GO. Do it FIRST. (2-attribute stays the FHRR's own ~29% boundary.)
- **Embedded clauses (depth-1) = SURPASSABLE via INDIRECTION** (a pointer to a second slot-group), NOT
  copy-a-composite-into-a-filler. Biology says *point, don't copy*; the repo already has every piece (the slot-binder
  for the inner clause, the D3 factored register + `_d3_persistent_slot` attractor for the pointer hold, the neural scan
  for both reads). Depth-1 is the entire requirement (parser abstains beyond; depth-2 = human center-embedding limit).
- **Together #1 (attribute) + #2 (pointer clause) close the ENTIRE capability set the FHRR delivers in production**
  (flat SVO + polarity + multi-hop are already in the slot-binder; the FHRR's 2-attribute-F3 ≈29% and depth-2 are
  boundaries the FHRR itself does not cross). ⇒ these two CPU de-risks fully retire the FHRR exact-inverse algebra.
- **All CPU/numpy-tractable; NOT GPU-gated** (the slot-binder de-risks run on the numpy substrate). NO `sim/` edit
  (all reuse-by-import; the only change is the composer-contract in `slotbinder_composer.py`: route a `Clause`/tuple
  patient instead of the current `return False`).

## MOVE 1 — the genuine residual (isolated)
`SlotBinderComposer` stores each fact's `(agent,verb,patient,polarity)` into its OWN four spiking competitive slots
(`_ROLES=4`), recalls by a neural scan (`_match`→`_read_slot`) with an intrinsic no-confab moat (returns `None` on no
match). 6-seed GO + adversarially verified. Its capacity is slot-count-limited (the gap-#2 win over the FHRR ~2 cap).
**`store()` HARD-REJECTS non-flat fillers** (`if not (isinstance(patient,str) and ...): return False`) — that guard is
the exact hook. The FHRR provides exactly two things the slot-binder lacks: (a) EMBEDDED CLAUSES (a `Clause` patient,
stored by recursively `_encode`-ing the inner clause into a composite phasor and binding THAT as the patient filler —
a 2-level unbind to read); (b) ATTRIBUTED PATIENTS (adj+noun, a separate `attribute` role in the SAME flat bundle —
NOT recursion; single-attribute only, 2-attribute is the FHRR's own ~29% boundary). Depth needed: DEPTH-1 only — the
`EmbeddedClauseParser` is depth-1 (subject/object relatives) and abstains on depth-2; the agent surface only builds a
depth-1 `Clause` patient + single adjectives.

## MOVE 2 — biology reframe: POINT, don't COPY
The FHRR does recursion by copying the inner clause's content into a composite and binding it as a filler — which
re-imports the superposition problem one level down (Smolensky 1990 exponential-dimensionality; VSA "sacrifices
capacity/fidelity via noisy compression"). Biology does not copy — it points:
- **Neural Blackboard Architecture** (van der Velde & de Kamps 2006 *BBS* 29:37-108; 2017 *Front. Psychol.* 8:1297):
  word assemblies stay IN SITU; structure = temporary gated connection paths to structure assemblies (NP/VP with
  role subassemblies), held by memory/delay circuits. Recursion = binding one STRUCTURE ASSEMBLY to another via the
  same gated path. No architectural depth limit — the limit is working-memory capacity.
- **Assembly projections + multilevel pointers** (Müller, Papadimitriou, Maass & Legenstein 2020 *eNeuro*
  7(3):ENEURO.0533-19): fillers bind to roles via projections into distinct structural SUBSPACES (one per role) via
  disinhibition+STDP; nesting via multilevel POINTERS implemented as transient excitability + a MERGE op.
- **Factored data registers** (Frankland & Greene 2015 *PNAS*; 2020 *Annu.Rev.Psychol.*): adjacent lmSTC subregions
  hold agent & patient as separate abstract variables ("data registers") — the grounding the project's own D3 EVENT
  REGISTER already stands on (a factored two-slot register, discrete attractors, push/pop without erasing, 6-seed GO).
- **Binding is rate-enhancement, not obligatory synchrony** (Roelfsema/VU 2023 *Neuron*): supports the slot-binder's
  rate-based competitive-slot substrate; the recursion extension need NOT be theta-gamma-timed to be faithful.

## MOVE 3 — ranked cheap-first mechanisms
**#2-difficulty but DO-FIRST — Attribute slot (the easy win, separable, near-certain):** bump `_ROLES` 4→5, add an
`attribute` role; "big apple" writes the attribute filler; queries read+join it. Reuse `SlotBinderComposer` + mirror
`one_brain_composer._resolve_patient` for the `(adjs,noun)` split (first adjective). De-risk (6-seed): recover BOTH
patient AND attribute ≥0.90 with ≥3 attributes in the set. Anti-cheats: (1) permuted-attribute → attribute read →
chance; (2) moat unchanged (un-attributed fact → no confabulated adjective); (3) do NOT claim 2-attribute /
per-noun-attribution (the latter is a mini-pointer problem, out of scope). Risk ~zero.

**#1 RECOMMENDED recursion — Pointer/indirection slot:** store the inner clause `(cat,chase,bird)` as its OWN fact at
group `j` (existing GO mechanism, its own near-orthogonal slots); store the matrix fact `(dog,saw,PTR_j)` where the
patient slot binds a POINTER filler = one of a small set of dedicated `CLAUSE0,CLAUSE1,…` pools (appended exactly like
the `AFFIRM/NEGATE` polarity pools). Read = read matrix patient → recover `PTR_j` → FOLLOW → read group `j` with the
same scan. No copying, no clause-level superposition (gap-#2 property preserved at depth-1). Reuse (all in-repo, NO
`sim/` edit): the slot-binder for the inner clause; `_d3_persistent_slot_derisk.py` (recurrent NMDA attractor holding a
slot with zero input = the pointer's biological hold) + the D3 two-gate register push/pop; the `_match`/`_read_slot`
scan for both reads. De-risk (6-seed, single-variable) on the parser's held-out depth-1 subject+object relatives:
embedded-clause roles ≥0.90 AND matrix roles ≥0.90 on ≥5/6, flat SVO un-regressed, moat holds. Anti-cheats:
(1) permuted-pointer → embedded read → chance; (2) lesion-the-second-hop → returns the pointer code not clause content;
(3) wrong-clause distractor (≥2 clauses stored, must select the RIGHT group); (4) moat (pointer names no stored group →
abstain). Risk: the #-of-pointer-pools is a scale lever (like `max_facts`), not a wall; the pointer read is within the
validated scan regime. Depth-1 only.

**#3 fallback — theta-gamma ordered-WM sub-clause store (`OrderedPositionWM`, EMERGE-85/86):** store the inner clause
as an ordered sequence in the 6-seed-GO spiking Lisman-Idiart buffer + reference it. Honest flag: `OrderedPositionWM`
SUBCLASSES `RFPhasorComposer` — it leans on the FHRR RF substrate, so it delivers working depth-1 recursion but does
NOT purely close the slot-binder's own recursion. Use only if #1's pointer-read stalls, or for the parse-side WM latch.

**#4 DOMINATED — copy-the-clause-composite-into-a-slot-filler (the FHRR's own method ported to slots):** re-introduces
the ~2 superposition cap at the clause level. Recorded only to note pointer(#1) strictly dominates copy(#4). Do NOT build.

## MOVE 4 — recommended sequencing
1. **FIRST — attribute slot** (`_ROLES` 4→5). Cheapest, near-certain, separable; also de-risks the add-a-role plumbing #1 needs.
2. **THEN — pointer-clause** (#1): route a `Clause` patient through store-inner-as-fact → bind `CLAUSE_j` pointer →
   follow-the-pointer read; reuse the D3 persistent-slot attractor. 6-seed on held-out depth-1; the four anti-cheats.
3. **Fallback** if the pointer-read stalls: #3 (`OrderedPositionWM`), with the honest RF-substrate caveat.

## Sources (actually read)
van der Velde & de Kamps 2006 BBS 29:37-108 + 2017 Front.Psychol. 8:1297 (NBA in-situ binding, recursion by
structure-assembly binding, depth = WM not architecture); Müller/Papadimitriou/Maass/Legenstein 2020 eNeuro
7(3):ENEURO.0533-19 (assembly projections + multilevel pointers); Frankland & Greene 2015 PNAS + 2020 Annu.Rev.Psychol.
(lmSTC factored registers); Roelfsema/VU 2023 Neuron (binding = rate-enhancement); Smolensky 1990 AI 46:159-216 +
arXiv 2606.11391 (TPR/VSA recursion dimensionality/capacity). In-repo: `slotbinder_composer.py`, `rf_phasor_composer.py`,
`one_brain_composer.py`, `brain_conversational_agent.py`, `_phaseB_embedded_clause_parse_derisk.py`,
`ordered_position_wm.py`, `_d3_persistent_slot_derisk.py`; findings `2026-07-17-keystone-2-*`, `2026-07-03-emerge8{5,6}-*`,
`2026-07-09-D3-event-composition-*`.
