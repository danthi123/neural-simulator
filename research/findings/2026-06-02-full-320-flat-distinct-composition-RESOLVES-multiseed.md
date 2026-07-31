---
type: finding
status: live
date: 2026-06-02
---

# Full-320 flat-distinct biological composition RESOLVES multi-seed -- the honest completion -- 2026-06-02

## What this closes
The deferred full-320 completion of the flat-distinct cross-bridge biological composition. The 192-concept
result (3 bridges, 2026-06-02-flat-distinct-RESOLVES-...) was the validated honest recovery after the
hierarchical-320 shortcut was retracted (it scored a CATASTROPHIC 0.000 on structured facts at seed 42 -- the
documented NESTING/multi-hop SNR wall from stacking a 2nd binding level). This extends the honest flat path
to the full documented 320 "age-5" target.

## The honest path (recap)
Hierarchical-320 made the 5 bridges' duplicate seed-42 codes distinct by binding each with a bridge-role
vector -- but that 2nd binding level interferes catastrophically for STRUCTURED (bridge-systematic) fillers.
The fix is NOT a cleverer 2nd level; it is to remove the need for one: retrain each bridge with a DISTINCT
seed so its sparse patterns -- and therefore its captured flat codes -- differ at a SINGLE binding level, like
the within-bridge 64-concept composition that was always robust. So: 5 bridges, seeds 42-46
(noun@42 existing / verb@43 / adj@44 / spatial@45 / functional@46).

## Result (job bh4o2reg3, GPU/CuPy, seeds 42/43/44)
- 320 FLAT codes captured from all 5 distinct-seed bridges.
- 320-wide between-concept cosine: **mean 0.045, max 0.604 (DISTINCT)** -- max < 0.9, the VOID-duplicate guard
  did not trigger. (Max stayed at the 192-result's 0.604 because the most-overlapping pair lives within the
  unchanged A/B/C banks; adding D/E lowered the MEAN 0.108 -> 0.045 by diluting with more-distinct codes --
  internally consistent.)
- **STRUCTURED SVO composition (agent=noun / action=verb / patient=adj), full-3-slot QA, cleanup over ALL 320:
  1.000 / 1.000 / 1.000 (mean 1.000).**  [hierarchical reference on the SAME harness: 0.000 / 0.950 / 1.000]

## Scrutiny of the PASS (a PASS is scrutinised harder than a FAIL)
The 320 hierarchical shortcut scored 0.000 at seed 42 on this exact test, so a 320 PASS must survive hard
checks before being claimed:
1. **Codes truly distinct** -- max between-cos 0.604 < 0.9; mean 0.045; no duplicate-code artifact (the bug
   that produced a false "160-pool boundary" earlier). Guard explicit and not triggered.
2. **Realistic distribution, not random fillers** -- fillers are bank-systematic (agent=noun/action=verb/
   patient=adj), i.e. the STRUCTURED distribution that EXPOSED the hierarchical overclaim, not the random-
   filler distribution that masked it. (The lesson from the retraction: validate the realistic input
   distribution, not random samples.)
3. **Cleanup over all 320** -- the unbind cleans up against all 320 words, including the 128 spatial/functional
   DISTRACTORS that the 192 test did not have. Harder than 192, still 60/60 correct. Per-fact chance ~ (1/320)^3
   -> 60/60 is decisively not luck.
4. **The harness can fail** -- the identical harness scored hierarchical 0.000 at seed 42, so the 1.000 is the
   flat-distinct codes genuinely composing, not a rigged/trivial test.
5. **Multi-seed, not a single lucky seed** -- all three seeds 1.000, including seed 42 where hierarchical was
   catastrophic.

## Honest scope (what this is and is NOT)
- IS: robust cross-bridge biological composition over structured SVO facts at the full 320-concept scale, in
  the spiking substrate, via the validated coincidence bind/unbind + cosine cleanup -- the brain-analogue
  mechanism the owner's goal asks conversation to be built on (NOT static engram-tag retrieval/ranking).
- IS NOT a claim that the 320 CONCEPT CODES are learned end-to-end: per the cheating-audit, large-V sparse
  codes are GIVEN by the orthogonal/Kanerva sparse encoding; the COMPOSITION on top is the genuine, now-
  robust-at-320 result. Recognition (word -> code) is validated at small vocab; the 320 substrate uses the
  given sparse codes.
- Cost honesty: each new distinct-seed bridge is ~73-75 min to train (64 concepts x 400 events x 8192 lang =
  ~25,600 events x ~0.17s); the documented "~17 min/bridge" was wrong for the 64-concept tier (bridgeD 73 min,
  bridgeE 75 min on a verified-clean GPU -- not fragmentation). Incremental/resumable training (--resume-from,
  shipped today) lets such retrains be chunked across breaks if needed.

## Any-bank escalation (strictly harder than structured) -- 6 composition-seeds
agent/action/patient each drawn from ALL 320 (any concept, ANY role), cleanup over all 320, seeds 42-47
(owner's generalization-claim standard):
- **[1.000, 1.000, 0.950, 1.000, 1.000, 1.000], mean 0.992** -- 5/6 seeds perfect, 119/120 facts fully correct.
- The single miss (seed 44) is honestly localised: miss-by-bank = {spatial: 1} -- the spatial bridge (D@45,
  one of the two freshly-retrained distinct-seed bridges) cleans up marginally harder. A real, characterised
  residual, not hidden, not a flaw. VERDICT: RESOLVES (min 0.950 >= 0.80, 6-seed).
- So all 320 concepts are usable as fillers in ANY role, not just bank-structured SVO -> the strongest 320
  compositional claim, hardened multi-seed. (Codes distinct throughout: between-cos mean 0.045, max 0.604.)

## Conversational-KB demo (the tangible artifact; compose_flatdist320_conversation_demo.py)
Single-seed illustration (the multi-seed any-bank above is the evidence; the demo shows it conversationally):
stored 3 cross-bank SVO facts (bear/close/clean, ant/break/sweet, bed/pull/rich); answered **6/6** role +
relational queries correctly; the absent-cue control ("apple", never stored -> "(no fact found)") correctly
ABSTAINED -- the decisive anti-artifact check (a drive-echo/trivial store cannot abstain). Transcript:
```
   stored: agent=bear  action=close  patient=clean   (+ ant/break/sweet, bed/pull/rich)
   Q: who is the agent of fact 0?         A: bear   (OK)
   Q: what is 'bear' close?               A: clean  (OK)        ... 6/6 ...
   Q: what is 'apple' (never stored)?     A: (no fact found)    (OK, clean abstention)
```

## Reproduce
```
# 5 distinct-seed bridges (each ~73-75 min; bridgeA noun@42 pre-exists):
bash research/findings/raw/_run_flatdist_DE.sh          # spatial@45 + functional@46 (or _run_flatdist_E_only.sh)
# 320 structured composition test (~15 min):
python -m research.findings.raw._insubstrate_flatdistinct320_test
# any-bank escalation + demo (~10 min, loads the cached codes):
python -m research.findings.raw._insubstrate_flatdist320_anybank_test
python -m research.runners.compose_flatdist320_conversation_demo
```
