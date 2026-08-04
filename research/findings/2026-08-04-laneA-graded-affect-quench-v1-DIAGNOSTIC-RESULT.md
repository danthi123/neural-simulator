---
type: finding
status: complete
date: 2026-08-04
verdict: DIAGNOSTIC_NO_SELECTION
mechanism: laneA-graded-affect-quench-v1
artifacts:
  - research/specs/lanea_graded_affect_quench_v1.json
  - research/findings/raw/affect/graded_quench_v1/diagnostic/seed-6158765.json
  - research/findings/raw/affect/graded_quench_v1/diagnostic/seed-6158765.receipt.json
  - research/findings/raw/affect/graded_quench_v1/diagnostic/seed-7695139.json
  - research/findings/raw/affect/graded_quench_v1/diagnostic/seed-7695139.receipt.json
  - research/findings/raw/affect/graded_quench_v1/diagnostic/aggregate.json
  - research/findings/raw/affect/graded_quench_v1/diagnostic/aggregate.receipt.json
---

# Graded affect diagnostic retains clearing but selects no operating point

**Verdict: DIAGNOSTIC_NO_SELECTION.** Both locked diagnostic seeds completed,
but none of the seven recurrent weights passed every required gate on both
seeds. The aggregate therefore contains no eligible candidate, selects no
recurrent weight, and records `NO_SELECTION_FORMAL_REMAINS_SEALED`. This
diagnostic has no promotion value.

## Evidence integrity

The authoritative decision artifact is
`research/findings/raw/affect/graded_quench_v1/diagnostic/aggregate.json`.
The seed artifacts and aggregate are bound to archived source revision
`3cb87cf411d35093286d6dec8eff1096be8d9f31` and locked spec SHA-256
`611790d76c43e8ccb1e00614d1bb39627c8cb4185d6395bf6b8380b5ee55e5a0`.
The copied source manifest has SHA-256
`c54bac1db8bdd380f2f3264f140bf4fbb4397cf0b78b1e704bb19c64c3c91753`
and binds the ancestry attestation. The attestation exactly matches the
commits reachable from the archived revision and contains the locked source
anchor.

A read-only audit recomputed `6472` checks. <!--derived--> All `2098`
manifest entries matched the selected committed archive bytes, including all
`47` direct simulator and Lane A source files recorded by each artifact. The
two seed artifacts and aggregate carry the same complete source identity.
Their receipts match the present artifact hashes and sizes, canonical runner
commands, NumPy environment, `cpu:numpy` device, successful exit state, and
the same archive revision and manifest.

There is one portability boundary in the historical receipts. Each receipt
records its source manifest as `.source_manifest.sha256`, relative to the
original pool execution root. Running the repository-root receipt verifier on
these relocated files therefore fails because that historical relative path
does not exist at the repository root. The audit did not rewrite that path or
copy evidence into it. Instead, it recomputed each receipt's artifact hash and
size and validated the receipt's source fields against the read-only copied
`source-manifest.sha256`, `source-revision.txt`, and `source-ancestry.json` in
the diagnostic directory. This is a location portability limitation, not a
source-identity or digest mismatch.

The provisioning record reports one dirty checkout path excluded from the
archive. This does not make the executed source dirty: provisioning used
`git archive` bytes from the named commit, every deployed source byte is
manifest-bound, and the execution artifacts record clean, read-only archive
source. The provenance sidecars also contain unrelated but fresh shared-pool
corpus-check queries; those strings are not inputs to the locked protocol,
candidate scores, aggregate decision, or execution receipts.

## What the two seeds establish

Across all seven weights on both seeds, the opponent circuit retained both
positive and negative driven states after input removal. The matched NMDA-off
control lost persistence. State polarity was correct, and the existing
spiking `quench_fs` pathway actively cleared the held state and allowed it to
restart with the opposite sign. Closing only the pathway's output gate
prevented clearing while leaving the clear drive and FS activity present.
The FS population was driven during clearing, quiet at the scored read, and
had no residual host-issued clear drive. These controls passed for all `14`
seed-by-weight candidates.

In plain terms, the previously established memory-and-clear mechanism still
works after lowering recurrent gain. It can hold one side, clear it through
the intended spiking inhibitory pathway, and start again. The diagnostic does
not show that this held state behaves like a smoothly varying emotional
quantity.

## Why no recurrent weight was eligible

Every candidate failed the same four load-bearing graded-state requirements:

- The magnitude span never reached the locked `0.020` minimum. The largest
  observed span was only `0.006650`.
- Sign accuracy was `0.533333` for every candidate, below the locked `0.75`
  requirement.
- Every candidate recorded zero sign crossings, below the required two.
- State around zero remained too far from neutral. Even the lowest observed
  zero-band fraction was `0.7988333831887526`, above the `0.5` maximum.

Nine of the `14` candidates also missed magnitude correlation, and five had
an excessively large latch-like step. <!--derived--> A trend correlation by
itself sometimes passed, but the state did not actually cross sign under the
locked down/up/down schedule. Because eligibility required every gate on both
seeds, each universal failure was independently sufficient to exclude all
seven weights.

## Boundary and next decision

No formal artifact exists, and none of the six reserved formal seeds appears
in the runtime provenance or corpus-check records. Formal remains sealed as
required by the locked selection rule. The result does not reject persistent
affect or active clearing; it rejects this recurrent-weight ladder as a way to
obtain graded, neutral-crossing valence from the current circuit.

A follow-up must change the mechanism or protocol under a fresh
preregistration. It must not tune against, rerun, or promote these diagnostic
seeds, and it must not release the formal seeds without a newly earned
diagnostic selection.
