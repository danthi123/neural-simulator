> **RETRACTED 2026-06-02 (the "RESOLVES multi-seed" claim):** the 1.000/0.98 was on RANDOM fillers. On STRUCTURED facts (noun/verb/adjective -- the realistic conversational case) the same composition is wildly seed-dependent: full-3-slot QA 0.000 / 0.950 / 1.000 at seeds 42/43/44. At seed 42 (where random fillers scored 1.000) structured composition COMPLETELY FAILS. The hierarchical bridge-role bind makes the 320 codes DISTINCT (recognition fine) but stacks a 2nd binding level (composition-role x bridge-role x code) that hits the documented NESTING / multi-hop SNR wall -- at some seeds the extra level interferes catastrophically with the composition roles. The integration demo (seed 42, structured) caught it: 0/6. So full-320 biological COMPOSITION via the hierarchical shortcut is NOT robust. What stands: within-bridge 64-concept composition (FLAT codes, no nesting) is robust multi-seed (0.95); recognition over 320 hierarchical codes is fine. The honest path to robust full-320 biological composition is DISTINCT FLAT codes (retrain the 5 bridges with distinct seeds 42-46 -> no extra nesting level), NOT the hierarchical shortcut. See 2026-06-02-hierarchical-320-NESTING-WALL-honest-retraction.md.

# FULL 320-concept biological composition works IN SPIKING -- the brain-analogue substrate scales to 320 -- 2026-06-02

## The result
Biological spiking relational composition + wh-QA + abstention over ALL 320 concepts of the real deployed
G.20 substrate, in ONE cleanup space, computed by actual spiking neurons:

- 320 real codes captured (5 bridges x 64, temporal-integration readout stim=300)
- hierarchical bridge-role bind applied -> 320-wide between-concept cos: mean 0.048, MAX 0.537 (DISTINCT;
  no duplicates -- the bind separated the shared-pattern bridges on the REAL codes, not just synthetic)
- **SPIKING 320-concept relational QA (who) = 1.000  |  abstention = 1.000**  (chance 0.0031), seed 42, n=20

VERDICT: RESOLVES. The brain-analogue conversational substrate scales to 320 concepts.

MULTI-SEED CONFIRMED (composition seeds 42/43/44 on the cached 320 codes): spiking 320-way QA 1.000/1.000/0.950 (mean 0.983), abstention 1.000 EVERY seed. Clean multi-seed PASS -- not seed-luck.

## How the duplicate-code blocker was solved (no retrain)
The 320-tier's 5 bridges were all trained with seed 42, so they share sparse patterns -> bridgeA-i and
bridgeB-i have near-identical FLAT codes (max-cos 1.000) -> a global 320-way spiking cleanup is ambiguous.
FIX: bind each concept with its bridge's near-orthogonal ROLE vector (concept_code = bridge_role (Hadamard)
within_code). Because the bridge-roles are near-orthogonal, (roleA*p_i).(roleB*p_i) ~ 0, so cross-bridge
same-index concepts become DISTINCT. This is a clean VSA hierarchical bind -- biologically the same
coincidence/gain-field operation the project already validated -- and it needed NO retraining.

Cheap-first (numpy algebra) gated it: FLAT max-cos 1.000 -> HIERARCHICAL max-cos 0.323, algebra 320-way QA
1.000. The spiking run then confirmed it on the REAL captured codes: max-cos 0.537 (real codes noisier, still
distinct), spiking QA 1.000, abstention 1.000.

## Why this is genuine (scrutinized)
- Abstention = 1.000: the decisive anti-artifact control. A drive-echo / code-distinctness artifact cannot
  correctly abstain on unstored facts (it would clean up to some concept and answer wrongly). Perfect
  abstention over 320 establishes genuine relational composition.
- between-cos max 0.537 << 0.9: no near-duplicate codes survived -> the 320-way cleanup is unambiguous.
- The composition is the SAME validated spiking bind/unbind (multi-seed at 64: 0.95); here it carries 320
  distinct hierarchical codes.

## Trajectory (this session)
spiking bind validated -> within-bridge 64-concept composition ROBUST multi-seed (0.95) -> hierarchical
bridge-role bind makes 320 codes distinct (cheap-first 1.000) -> FULL 320-concept biological composition
RESOLVES in spiking (1.000 + abstention). The brain-analogue mechanism (not static retrieval) now does
structured relational reasoning across the full 320-concept deployed substrate.

## Honest scope
- Seed 42 (n=20); multi-seed confirmation running (cached codes -> fast; varies the composition RNG).
- The concept CODES are still given by the sparse encoding (cheating-audit scope); the COMPOSITION on top is
  the genuine biological win, now at 320 concepts.
- This is structured fact-memory + wh-QA (store SVO, query by role, abstain), not open-ended dialogue. The
  next layers (negation, learned parser, generation -- validated at small scale) stack onto this 320-concept
  biological substrate.
