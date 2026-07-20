# gap#1 M5 — INVALID harness + RETRACTION: my standalone write-fidelity 0.906 measured a NON-DEPLOYED quantity

Two honest corrections, both instances of the day's core lesson (validate on DEPLOYED inputs; a control that does
not reproduce its known value invalidates the run) — this time biting my own gap#1 work.

## 1. The M5 deep-NLL harness is INVALID — the NEF CONTROL does not reproduce

I wired a `--input-mode tokensdr` into the M2 runner and ran the pre-registered deep-NLL test. **Before reading any
tokensdr verdict, I ran the NEF control on the same checkpoint** — and it FAILS to reproduce M2's known result:

| mode | verify corr(state, ref) | deep d10-99 vs-trigram |
|---|---|---|
| **NEF (M2's own mode, KNOWN ~-0.030)** | **0.663** | **-3.069** |
| tokensdr (M5) | 0.388 | -3.062 |

**NEF gives -3.069, not its known -0.030.** A control that does not reproduce its established value invalidates the
whole run: the checkpoint (regenerated today at V=1000/d=128) and/or the eval harness is misconfigured relative to
what M2's finding used. **No mechanism verdict is readable from M5** — the -3.06 for tokensdr is meaningless while
the control is broken. (Also caught en route: the FIRST M5 runs used the M2 runner's `--n-sentences` DEFAULT 40000
while the checkpoint was trained at 80000 — the EXACT documented vocab-mismatch silent failure I wrote about earlier
today. Fixed by passing 80000; the NEF control still fails, so that was not the only issue.)

## 2. ⛔ RETRACTION: my standalone write-fidelity 0.906 measured a NON-DEPLOYED quantity

The standalone write-fidelity de-risk reported token-SDR **0.906 > M2's 0.786** ("selection beats regression"). The
deployed M5 state correlation REVERSES this: **tokensdr 0.388 < nef 0.663.** The token-SDR is deployed-WORSE than
NEF, the opposite of what my standalone metric claimed.

**Why the standalone was optimistic — it measured a different quantity than the deployment uses:**
- it RESET the membrane per token (`drive_token_window` set v=-65 each call); the deployed encode does NOT (membrane
  carries over). Adding a per-token reset to M5 moved the state corr 0.546 -> 0.697 — real but partial.
- it read `ge_pos - ge_neg` (a D-dim SUBTRACTED value) and correlated it against `v_true`; the deployed state is the
  2D UN-subtracted `[relu(+v); relu(-v)]` accumulated over the sequence through the slow integrator. Per-token
  instantaneous v-fidelity is NOT the accumulated-state fidelity the read-out consumes.

⇒ **The "token-SDR beats M2 (0.906 vs 0.786)" claim is WITHDRAWN.** It was measured on a per-token, membrane-reset,
subtracted quantity that the deployment does not use. The honest deployed number is the accumulated-state corr
(0.39), which is WORSE than NEF's 0.66.

## The lesson, a third time — and the specific process failure

The day's rule is "verify the claimed property on the DEPLOYED inputs before pre-registering." I built a standalone
write-fidelity probe, got 0.906, and pre-registered a deep-NLL test on it — WITHOUT first confirming the probe
measured what the deployment consumes. It did not. The deep-NLL integration (with its NEF control) is what exposed
it. **A standalone metric that is cheaper than the deployment is worth exactly nothing if it measures a different
quantity** — and the cheapest way to know is to run the deployment's OWN control first.

## Honest status of gap#1 after this

- **UNCHANGED / STANDS:** M1's graded-state result (host-inject) beats the trigram; the reservoir arc's slow-state
  legitimacy; the gate's ENCODE-is-the-wall reframe (that correction to my reconciliation stands).
- **RETRACTED:** "token-SDR selection beats M2 at write-fidelity 0.906." The deployed number is 0.39, worse than NEF.
- **INVALID (no verdict):** the M5 deep-NLL, because the NEF control gives -3.069 vs its known -0.030.
- **REQUIRED NEXT (before any gap#1 spiking-input verdict):** (a) make the NEF control reproduce its ~-0.030 on a
  freshly validated checkpoint+harness (diagnose the state-corr-0.66-but-deep-NLL--3.07 inconsistency, likely a
  read-out scaling or vocab-provenance issue); (b) ONLY THEN measure tokensdr against a working NEF baseline, with
  write-fidelity computed on the DEPLOYED accumulated state, not a standalone proxy.

No mechanism conclusion about the token-SDR encode is claimed. The honest deliverable here is the retraction + the
invalid-harness diagnosis.

---

## DECISIVE DIAGNOSTIC — the root cause is VOCAB PROVENANCE, not the encode (M1 reference ALSO fails)

Ran M1 on-bridge (host-inject, the EXACT-input reference that should give +0.486 at V=1000/d=128) on the SAME
regenerated checkpoint:

| reference | verify corr | deep d10-99 vs-trigram | expected |
|---|---|---|---|
| **M1 host-inject** | 0.751 | **-3.062** | **+0.486** |

**M1 — the host-inject reference with a KNOWN +0.486 — also gives -3.062.** Since the exact-input path fails
identically to NEF and tokensdr, the encode is NOT the cause: **the regenerated checkpoint + the on-bridge vocab
rebuild are mismatched.** onbridge NLL 6.7 ~= ln(1000)=6.9 (near-uniform), so the read-out is producing garbage
logits even with perfect input — the emb/Wv/head indices do not correspond to the re-tokenized stream.

**Root cause (pinned):** the on-bridge runners REBUILD the vocab with `Vocab.build(tr, V)` rather than loading the
checkpoint's SAVED `words`, and the rebuild does not reproduce the training token->id mapping even with n-sentences
matched — the LM trainer applies a `max_train_sents` truncation to `tr` BEFORE `Vocab.build` that the on-bridge
runner does not, so `tr` (hence the frequency-ordered vocab) differs. The checkpoint stores `words`; the on-bridge
eval ignores it. This is exactly the "vocab-provenance issue" the retraction above predicted as the required-next.

## What this means for the whole gap#1 deep-NLL push today

**EVERY on-bridge deep-NLL number produced today on this regenerated checkpoint is INVALID** — M1, M2 (NEF), and M5
(tokensdr) all -3.06, because they share the broken vocab rebuild. This does NOT retroactively invalidate the
ORIGINAL M1 +0.126/+0.486 (those used a checkpoint+eval that were vocab-consistent, per the M1 finding's own matched
n-sentences fix); it invalidates only TODAY's regenerated-checkpoint runs.

## The precise, decisive REQUIRED-NEXT

The on-bridge runners must use the checkpoint's SAVED vocab (`words`) instead of rebuilding it — a small, additive
fix (load `W["words"]` and build the token->id map from it, bypassing `Vocab.build`). ONLY after M1 host-inject
reproduces ~+0.486 on the checkpoint is ANY encode comparison (NEF vs tokensdr) meaningful. Until then, no gap#1
spiking-input verdict is possible, and the token-SDR mechanism is neither confirmed nor refuted — it was never
validly tested.

## The honest tally for gap#1 today

Three self-corrections in one thread: (1) the reconciliation's "conductance-drive escapes M2" was a false dichotomy
(gate); (2) the standalone write-fidelity 0.906 measured a non-deployed quantity (deployed 0.39 < NEF 0.66,
retracted); (3) the deep-NLL harness is invalid via a vocab-provenance mismatch (M1 reference fails too). The
constant across all three: I trusted a cheaper proxy (a reconciliation framing, a standalone metric, a regenerated
checkpoint) without running the deployment's OWN control first. Running the M1/NEF control FIRST would have caught
every one before a single mechanism claim.

---

## FINAL: I could NOT reproduce M1's +0.486 on a regenerated checkpoint — the whole gap#1 deep-NLL push was unvalidatable

Chased the M1 reference through every config, applying the vocab fix (use saved `words`) and the correct M1 mode:

| M1 config | verify corr | deep vs-trigram | expected |
|---|---|---|---|
| firing-rate readout, rebuilt vocab | 0.66 | -3.06 | — |
| firing-rate readout, **saved vocab** | 0.756 | -3.026 | — |
| **`--ssm-state`** (graded), saved vocab | **1.000** | -3.011 | — |
| **`--ssm-state --use-ssm-readout`** (CANONICAL +0.486 config), saved vocab | **1.000** | **-3.013** | **+0.486** |

**Even the canonical M1 config — byte-exact state (corr 1.000) + the SSM's OWN trained read-out + the saved vocab —
gives -3.013, not +0.486.** With the state provably exact and the model's own read-out, the only remaining cause is
that **the regenerated checkpoint is not on-bridge-compatible** (a state-layout / read-out-input convention, or the
eval-stream reconstruction, differs from whatever produced the ORIGINAL M1 checkpoint). The off-bridge WKV training
itself was GO (+0.512), so the model is fine; the on-bridge REALIZATION of THIS checkpoint is not.

## The honest, decisive close of the gap#1 push today

- **I never established a working M1 reference on the regenerated checkpoint**, so NO on-bridge encode comparison
  (NEF, tokensdr) was ever valid. **The token-SDR mechanism is neither confirmed nor refuted — it was never validly
  tested**, and the standalone 0.906 remains retracted (non-deployed quantity).
- **The regenerated checkpoint (`bridges/wkv_ckpt/wkv_v1000_d128_seed42.npz`) is UNUSABLE for the on-bridge path** as
  produced — a fresh `_emerge_wkv_lm_derisk --save-ssm` run does not yield an on-bridge-reproducing artifact under
  the configs tried. Reproducing M1 requires either the ORIGINAL checkpoint (lost post-migration; `.npz` gitignored)
  or diagnosing the exact state-layout/read-out/eval-stream convention the original M1 realization used.
- **REQUIRED-NEXT is now a HARNESS task, not a mechanism task:** produce a checkpoint on which `--ssm-state
  --use-ssm-readout` reproduces ~+0.486 (the M1 control MUST pass before any encode is tested). Only then is the
  token-SDR question answerable.

## The meta-lesson, stated plainly

Four consecutive self-corrections in this one thread (false-dichotomy reconciliation -> non-deployed metric ->
vocab-provenance -> checkpoint-incompatibility), each surfaced only by running a control that failed. The through-
line: **I kept building forward (a probe, a pre-registration, an encode, a deep-NLL) on an unvalidated foundation,
when the FIRST action should have been "make the M1 control reproduce its known +0.486 on this checkpoint."** Every
error was downstream of skipping that. The discipline that WORKED today (gap#4: pre-register, verify on deployed
inputs, run the control first, cap the tuning) is exactly what I failed to front-load here. The gap#1 push produced
no mechanism result — its honest deliverable is this negative + the precise harness blocker.
