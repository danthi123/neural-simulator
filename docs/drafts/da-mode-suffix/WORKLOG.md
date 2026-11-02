# da-mode-suffix — draft WORKLOG (section 5 self-directed, 2026-11-01)

2026-11-01 21:52 — CHOICE + WHY. Section 4 is exhausted tonight: T0 done; T1/T8/T19/T21/T23 open-NEEDS or ruling
blocked (s2p-window-1018 holds T21); T10 Start 0/150; T13's Start log rotated away (no DONE(rc= line for
test_bounded even in a 3000-line window); T14 INFO posted 10-27 (t14-info-1018, 0 new s7 files since); T15/T17 done
on their branches (0cbe8a401 + phantom-cfg/AFFECTED.md exists); T18/T20 complete; T16 no cff-reverify; T22/T24
RESULTed. Per section 1.5 -> section 5.

PICKED: board #277, the DA-mode engagement suffix dangling after a bare abstain (opened 11-01 by the C3 flip-gate
finding, research/findings/2026-11-01-c3-flip-gate-RESULT.md lines 48-51). WHY: it is the only CURRENT STATE open
item that (a) serves the 11-21 Stage-2 retake (the suffix is live production surface on every focus/arousal reply),
(b) is integration-first — the defect is a reply change, so any fix's lesion changes the reply, (c) is NOT
Claude-reserved to investigate (no merge/prereg/scorer/flip needed to measure it), and (d) is blocked by no open
NEEDS line. The other current items are Claude-gated (f1j engine edit, C2 match-unit packet, identity canary
queueing, BG re-run under research/bg-tonic-loop) or already landed tonight (bg step-1 score, [c2-ff-grade]).

THE DEFECT: the engagement suffix (" — worth going further here." / " — there's plenty more to dig into here!",
webapp/da_mode_drives_chat.py) appends to every focus/arousal reply. With C3 topic-stop ON, the follow-up question
is dropped on a repeat topic, so the suffix rides a bare "I don't know about that." — punctuation that promises
going further on a reply that declines to. OFF-arm already appends after the question ("... about alone? — worth
going further here."). Property of the suffix placement, not of the flip; filed as its own defect (#277).

RECORD (byb recorded 21:55; sources read): 2026-08-19-da-mode-drives-chat-load-bearing-GO.md — suffix-iff-engaged,
rest turns withhold it, content byte-identical, mode→suffix string = declared host conditioned-articulation
scaffold (owner-sanctioned), never-crashes guard; 2026-09-17-...-da-mode-engagement-regression-HELD-default-off.md —
the opposite failure (suffix DROPPED when engagement shifted low via a curiosity calibration mismatch) is a
different, root-caused defect, onebrain pool flip still default-OFF. Question for the PLAN: which lever governs
suffix placement — L1 host reply-shape suppression on abstain, L2 suffix-vs-follow-up ordering, or L3 the
topic-stop faculty's own signal gates the suffix (integration-first: one neural signal decides both)?

NEXT (section 5 steps 3-5): PLAN.md via one MiMo Pro delegate, refuted by a second; build in
$WT/draft-da-mode-suffix on research/draft-da-mode-suffix; dev seeds only; PROVISIONAL finding; no GO.

2026-11-01 22:05 — REFUTER ROUND 1 (MiMo Pro): verdict REVISE, 4 blocking issues — (1) L1's
'SNc-only / 8-19 lesion story unchanged' claim false once question presence flows through the
topic-stop organ + host ask threshold (declare indirect vs direct, all levers amend the
invariant); (2) L1's S6 precedent miscited — S6 is the NAMED HOST SHORTCUT the congruence gate
replaced; cite the congruence-gate Declared-host-step pattern and name the question-presence
signal (ask-decision boolean, never a reply-text parse) as boundary debt; (3) lesion cells in
both S1 and S3 = double-count — one locus; (4) content-free cells contradict Table A5 (induced
focus/arousal content-free turns WOULD carry the suffix per 8-19). 11 edits applied, including
the refuter's factual suggestions (two append sites server.py ~7601/~7927, pre-map label, no
focus-band afferent in the record, 16-cell batch bound, L2-attach abstain-only scope).
ROUND 2: re-approval delegate UNAVAILABLE (delegation budget 2/2 for this run) -> inline
consistency review instead (leftover-string greps all 0; 84 S1 cells / 28 per seed / S3-only
lesion agree across sections 5/6/7/8). PLAN.md carries this status honestly. A later wake may
run the re-approval delegate before step 4 (build in $WT/draft-da-mode-suffix).
MILESTONE: stage-2 dev preview reached 105/105 at 22:05 (s1/s2/s3 35 each) -> RESULT posted.
