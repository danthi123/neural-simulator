# RUNG B-1c.2 surpass — substrate probes (seeds 42/43/44, SIM_BACKEND=numpy)

These probes located the residual boundary precisely and drove the two fixes (P=80/T=30 resolution + dropping the
lesion-immune per-role bias). All on the on-bridge SPIKING reservoir + the c2 P=80 WTA read-out.

## 1. Read-out RESOLUTION (P=80 / T=30) — the sub-1% margin
- Baseline (P=20/T=12, replay-3): seed 42 route 10/12; the crux found P=20/T=12 INVERTS the top-2.
- P=80/T=30: the scale-sweep host-agree band widens to 18/18 across scales 2.3–7.4 (seed 42) and 2.0–6.4 (seed 43);
  route 12/12 == host-dict on seeds 42/43. Seed 44: host-agree MAXES at 11/18 (scales 5.6/8.0) — a degraded draw.

## 2. Why the reservoir-lesion did NOT collapse — the lesion-immune per-role bias
`probe_c2bias` / `probe_nobias` (c2 spiking read-out, ens_sum argmax, canonical facts, per seed):

| condition                              | 42     | 43     | 44    |
|----------------------------------------|--------|--------|-------|
| bias-ON  intact                        | 18/18  | 18/18  | 6/18  |
| bias-OFF intact (reservoir rows alone) | 18/18  | 18/18  | 6/18  |
| bias-OFF reservoir SILENCED (W_in=0)   | 6/18   | 6/18   | 6/18  |
| bias-OFF ENCODER closed-class lesion   | 18/18  | 18/18  | 3/18  |

Reading: (a) at P=80/T=30 the reservoir ROWS ALONE resolve the canonical argmax (bias-OFF intact 18/18 on 42/43) — the
per-role bias tonic is NOT needed for the intact route. (b) The bias tonic is LESION-IMMUNE: with it ON, SILENCING the
reservoir still gives 18/18 (the bias carries the canonical prior) — so the reservoir-lesion cannot bite. (c) DROP the
bias (WS_BIAS_SCALE_C2=0) and SILENCE becomes load-bearing (18/18 -> 6/18 collapse). (d) The ENCODER closed-class
lesion is NOT load-bearing on the POSITIVE read-out (stays 18/18) — canonical role == content-word POSITION, which the
closed-class lesion preserves. => c2's reservoir-lesion = SILENCE the reservoir (W_in=0) with the bias dropped.

## 3. Why the closed-class lesion / recurrence-lesion are NOT load-bearing on canonical facts
- REC-LESION (zero reservoir_rec) on the c2 positive read-out: canonical-argmax 18/18 (NO collapse). The LSM's
  recurrent integration is not load-bearing for canonical role assignment — position (in the feedforward W_in map)
  suffices.
- So on the CANONICAL task, thematic role is over-determined by content-word position; only a NON-CANONICAL construction
  (position != role) exercises the reservoir's computation.

## 4. The non-canonical (objrel) construction — why it does NOT rescue res-lesion here
`["the", PAT, "that", "the", AGT, V3]` : slot0=THEME, slot1=AGENT, slot2=PREDICATE (position != role).
- HOST SIGNED argmax on the spiking feature: objrel slot0==THEME 12/12 INTACT (all seeds) — the reservoir DOES read it.
- But objrel slot0==THEME also 12/12 UNDER the closed-class lesion — "that" is NOT a discovered closed-class word (it is
  open-class), so it SURVIVES the lesion => objrel does not collapse under res-lesion.
- And the c2 POSITIVE spiking read-out reads objrel slot0 by POSITION (=AGENT), 0/6 correct on-substrate — the Dale
  OFFSET loses the structural THEME the SIGNED read-out captures. A SIGNED (host-current stand-in) read-out recovers
  slot0=THEME (but mis-scales slots 1/2 without per-slot tuning).
=> objrel is dropped; it is the POINTER to the residual mechanism (the signed +/- read-out).

## 5. Final 3-seed c2 (P=80/T=30, bias dropped, res-lesion=SILENCE)
- seed 42 GO: route 12/12 (dict 12), route-lesion 0<12, res-lesion 0<12, syn-readout-lesion 0<12, ws-scramble 0<12,
  moat 0.00, source-clean, neural-select. scale 2.287.
- seed 43 GO: route 12/12 (dict 12), res-lesion 0<12, all anti-cheats. scale 1.967.
- seed 44 NO-GO: route 0/12 (dict 12) — the patient slot latches AGENT; the degraded feature under-resolves the sub-1%
  margin (sweep host-agree max 11/18). scale 5.573.
- VERDICT: PARTIAL 2/3 (was 1/3); mean route recall 0.667.

## Residual mechanism the substrate needs (named, not forced)
A SIGNED ON/OFF (+/-) read-out: the negative Ws rows delivered through an INHIBITORY relay population (Dale-legal),
replacing the argmax-preserving Dale OFFSET. The offset preserves the LINEAR argmax but the SPIKING read-out of the
offset-positive drive loses the small non-canonical/borderline margins (seed 44's patient slot; the objrel structural
read). That, a larger reservoir, or a better-conditioned draw would resolve seed 44 at high recall.
