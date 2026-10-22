Census v2 (`research/runners/_deixis_encoding_census.py` on `research/deixis-census`, extended): the frozen probe set (sha
`e2918e3f...`) and oracle v1 (sha `3f83d6b3...`) unchanged; `--deixis off|on|lesion|lesion-both`, `--fixture
en|mm|en-ho`; a v2 sidecar with per-item outcomes for every arm; flag-off `census_pool.json` byte-identical to the census
re-run at the branch's merge-base with main (both re-run; `f84e868a...` stays the stage-0 record only). New measures: m4
split into silent / held (with class); m5 RENAMED for what it counts, "W1 declines whose first token is SPEAKER,
ADDRESSEE or NONE" (any decline, not B3 contradiction rejections: refuter minor 14), split into decline / hold; m1 also
scored on oracle v2; a real cross-speaker B3 measure (one shared store under `BRAIN_ONE_BRAIN=1`, A then B, calling
`check`), its stage-0 baseline taken at the merge-base before any flag-on arm, its S1-hold bar "no increase", the stage
that moves it S1-bind; the server-path arm (`webapp/brain_reply.reply_over_chat` on a tiny-demo chat with every default
wrapper installed, so the attribute path and W4's skip are measured where production runs them); production-shaped W3
(closed-list register, labelled fallback-only) and W4 (`OneBrainComposer`, retiring the rf-composer KeyError artefact);
two new writers with stage-0 baselines taken at the merge-base (completeness major 2): W5, the false-belief scene writer
(the probe set's statements plus a frozen scene set, 'you put the marble in the basket', 'he leaves the room', 'kofi puts
the ball in the box': count of scenarios started or advanced with a pointer form as the tracked agent), and W6, the
affective-ToM person split (count of first-person exclusions and my -> your renderings decided by its English list).
Oracle v2 (filed beside v1, never replacing it): p009 'latter' and p010 'that' relabelled ANTECEDENT(missing) (v1 labels
them NONE, so v1's m1 cannot count them; completeness blocking 0), new single-turn items 'the former likes jazz', 'this
one likes jazz' and two-turn former/latter items ('kofi and dalia study at dunmere' then 'the former likes jazz' / 'the
latter married fenna': turn 2 is stored today), and ADDRESSEE = brain in a chat addressed to the brain (for S1-bind).

Projection (`round2/scratch/fix/project_s1hold_v2.py`, pure arithmetic on the recorded W1 outcomes plus the six stores the
census m3 list omits, read from `out_b3on_onebrain_single.txt`; exposure = the R1-R3 table of 4.3 plus `brain`; the hold
replaces a store only): 59 of the 126 items move from stored to HELD (the judged design's projection was 27). These bars
are fixture-recovery checks (the organ reproduces the teacher's usage, and R2/R3 were chosen after the census was seen);
the LEARNING claim rests on the nonce arms (8.3).

| Measure | W1 baseline | S1-hold bar (flag on, EN fixture) | Residue, counted (method, stage, date in section 11) |
|---|---|---|---|
| 1 pointer stored as a name (oracle v1) | 28 | <= 3 | `moi` x3 (p007, p061, p062): produced by no exposure rule |
| 1v2 the same on oracle v2 (adds latter, that) | 30 | <= 3 | as above |
| 2 role shifts | 4 | 0 (p020-p023 held) | |
| 3 cross-speaker collisions | 60 | <= 3 | `moi` x3 |
| 4 silent drops | 60 | 60 silent (unchanged by design) + 57 held with a class; 0 new silent drops | the 60 are B3's own strip deciding the candidate (first person: S1-bind; told "you": stage 2 M3b; third person: stage 2 resolution) |
| 5 W1 declines (renamed) | 25 | 25 non-hold declines (unchanged) + the holds reported separately; 0 holds on non-deictic items | the 25 move with the 60 silent drops (S1-bind, stage 2) |
| 6 non-deictic stored item lost | 0 | 0: every stored probe item with no exposed form (p016, p035, p036, p109) and a frozen non-pointer set B3 stores ('the dog chases a cat', 'kofi chases the cat') | |
| 6b homographs | 0 | reported with a preregistered cost, ruling requested before the gate: 'a dog has a brain' (the self's name as a common noun), 'kofi owns a mine', 'kofi and fenna were friends' (R3's we're), each stored flag-off today and held under the flag | #260 (role context) |
| 7 correct facts lost (declared recall cost, accepted by ruling BEFORE the gate) | 0 | 5 coordinations with a true partial triple (p011, p027, p028, p113, p114); 16 over-general possessive stores (p015, p024, p025, p026, p039, p040, p043, p044, p047, p048, p051, p052, p055, p056, p059, p060); 1 over-general demonstrative (p010, stored as `(guy, like, jazz)`) | S1-bind and stage 2 store the bound version |
| former / latter / demonstrative arm (single turn and two-turn, oracle v2 items) | stored | HELD 100% on 6/6, and its made-up-marker twin held >= 29/32 per seed | |
| W2 legacy (36 / 0 / 52 / 68) | | the same holds (the read runs before the legacy split) | |
| W3 echo | every person-subject item | 0 echoes for exposed forms (the organ reads every turn the fallback would echo) | `moi` |
| W4 extract | 28 | <= 3 (exposed forms never reach `extract_assertion` on a held turn); 0 raises on the production-shaped arm | `moi` |
| W5 false-belief pointer agents | measured at the merge-base | 0 scenarios started or advanced with an exposed form as the tracked agent | Stage-3 C3 (the tracked agent bound by the organ) |
| W6 affective-ToM English split | measured at the merge-base | unchanged (reported) | S1-bind |
| cross-speaker B3 (BRAIN_ONE_BRAIN=1) | measured at the merge-base | no increase | S1-bind |
| question side | i/my/me -> brain; "we" literal | unchanged (S1-hold touches no question) | S1-bind (exposed forms), stage 2 (it/its) |
| slug scan | 17 tokens, 0 conversation-only | unchanged | |
| held-out spelling arm (EN-HO: the R3 u/ur/urself family withheld) | | those items stored exactly as flag-off (reported: generalization to an unheard spelling is not claimed) | stage 3 |

Per-miss attribution: for every exposed-form item that is NOT held, the record names which English list decided it (B3's
`_FUNCTION_WORDS`, surprise's copy, or `DEFAULT_SELF_ALIASES`): the "retires" claim is "shadowed where the organ fires".
