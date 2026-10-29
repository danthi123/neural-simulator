# [f1j-w1] RESULT — cs-race f1j step-0 IDENTITY score (2026-10-28 18:2x EDT)

Verbatim stdout of `python -m research.runners.cs_race_f1j_score --mode identity --pin 2592c6461ec38477b041bc053639e0e177f6f30a` (RC=0), run after f1j wave 1 landed 90/90 (34 cells `ccb8c39c4` + 56 cells `18e99f032`), per RULING f1j-w1-1028.

```
cs-race f1j step-0 IDENTITY mode, pin 2592c6461
f1i reference counts (re-derived): {"early": 45, "late": 6, "pass": 21, "corner_defined": 23} ok=True
Identity (i): {"41": {"equal": true, "problems": []}, "45": {"equal": true, "problems": []}}
Identity (iii): 72/72 APV records equal to f1i; not equal: {}
Invalid artifacts: 0; STOP-ID read-back problems (S1-S3, S7, S9-S13): 0; consistency: 0
f1j records: host summary: 90 file(s): pool=66, workstation=24
f1i references: host summary: 155 file(s): pool=138, workstation=17
VERDICT: IDENTITY PASS: queue wave 2 (952 lines), no ruling needed
```