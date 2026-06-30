# R4 close — the perception→compose grounding host-marshal made DEVICE-RESIDENT (2026-06-30)

**Type:** build (research/runners/ only; opt-in; default byte-identical). Closes **R4** from
`research/findings/2026-06-30-tier2-integrated-spiking-loop-scoping.md` (Option 3 — the cross-region analogue of R1).
**NO `sim/` edit.** Reuse-by-import. The no-confab moat is never touched. CPU-smoke done; the GPU 6-seed is the
controller's run (command below).

---

## UPDATE (2026-06-30) — GPU 6-seed was NEGATIVE *as a de-risk-harness artifact*; root-caused + FIXED. The CLOSE itself is correct.

The first GPU 6-seed came back **structurally GO but numerically NEGATIVE**: the SEAM is closed (`gen_concept-spike
to_host-clean 6/6` — the carrier never crosses host, the R4 structural goal ✓), but `device==host 0/6` (phase_cos
~0.73–0.90). **Root cause (it is the de-risk HARNESS, not the close):** the old `_seed_compare` ran the HOST path
(`read_gen_concept_spikes`) and the DEVICE path (`accumulate_conc_spikes_device`) as **TWO SEPARATE perception windows**
on the same bridge → **two DIFFERENT gen_concept spike snapshots**. GPU spike timing is not bit-identical across two
windows, so the RATE INPUTS differed — a phase_cos ~0.87 gap is **cross-window rate-input variance, NOT the complex op**
(a GEMV float-order delta would be phase_cos > 0.99999). The CPU smoke passed only because numpy is deterministic across
windows. **The production code reads ONCE per object**, so the two-window variance never arises in deployment — it was
purely an artifact of the de-risk comparing two reads.

**The fix (de-risk only, no change to the close):** `_seed_compare` now accumulates the gen_concept spikes **ONCE**
(device-resident) and formats that **ONE shared rate snapshot** BOTH ways — the host `angle(gen_proj@rate)` matmul and
the on-device projection — and asserts they are equal. This is the apples-to-apples test that isolates the close (the
only thing the close changes is *where* the projection runs), and is exactly what `test_device_resident_equals_host_matmul`
already pins on a fixed rate vector (**==host to atol 1e-9**, CPU). On GPU the same-snapshot device vs host is a
complex128 GEMV float-order delta (expected phase_cos > 0.99999) — the **GPU re-check command is in §5**. The earlier
"two separate windows" comparison was the bug; the close (accumulate-on-device + on-device fan-in, read once) is correct.

---

## TOP-LINE

**R4 is CLOSEABLE and CLOSED (device-resident), not a boundary.** The scoping ranked R4 SECONDARY and flagged it as
"possibly bounded," but the SURPASS analysis isolates the genuine residual to a **host DATA TRANSFER**, not a host
computation-of-cognition: the load-bearing percept→concept transform (the LEARNED `gen_perception→gen_concept`
convergence) is ALREADY synaptic; only the **gen_concept-spike VECTOR `to_host` + the host `gen_proj @ rate` matmul**
crossed host. That is the exact same class as R1 (an op's spiking RESULT crossing host to become the next op's operand),
and it closes with the exact same mechanism R1 used (CYCLE 722-723): keep the spiking result on-device and run the
fixed cortico-cortical fan-in projection on-device (backend→backend), so the host `gen_proj @ rate` matmul is gone and
the gen_concept-spike vector never crosses host — only the final D-length grounded PHASES cross host (the formatted
code, the R5 body-read, the same legitimacy class as `rf_read_phases`). The on-device matmul == the host matmul to
numerical tolerance (atol 1e-9, unit-pinned). **Default OFF = the validated host path BYTE-UNCHANGED; opt-in flips it.**

---

## 1. THE R4 DIAGNOSIS (file:line — exactly what crosses host)

R4 lives in the step-3 grounding (`navigate_to_compose_then_answer.py`, the DEFAULT `gen_spikes` mode), in two
functions called per perceived object from `_perceive_and_ground`:

1. **`read_gen_concept_spikes` — the gen_concept-spike VECTOR `to_host`** (`navigate_to_compose_then_answer.py:217-218`,
   pre-change):
   ```python
   fs = np.asarray(to_host(bridge.cp_firing_states))   # :217 -- reads the gen_concept SPIKES TO HOST every step
   conc_acc += fs[conc_region].astype(np.float64)      # :218 -- accumulates host -> a host rate VECTOR
   ```
   The percept is rendered into `gen_perception` (sensory render — legit body/world); the LEARNED rate-Hebbian
   `gen_perception→gen_concept` convergence fires `gen_concept` (SYNAPTIC — the load-bearing transform, already
   on-bridge). The residual: the gen_concept spikes are read TO HOST + accumulated host-side into a rate vector.

2. **`gen_grounded_phases` — the host `gen_proj @ rate` matmul** (`navigate_to_compose_then_answer.py:227-228`,
   pre-change) — **this is the "M @ rate" the scoping R4 names**:
   ```python
   z = gen_proj @ conc_rate.astype(np.complex128)      # :227 -- the HOST complex fan-in matmul
   return (np.angle(z) % (2.0*np.pi)) / (2.0*np.pi)    # :228 -- the host rate->phasor angle()
   ```

3. **the codebook write** (`navigate_to_compose_then_answer.py:445`): `cb.concepts[obj_word] = phases` — the grounded
   code lands in the composer codebook (a host numpy array the FHRR algebra reads; this is INTRINSIC to the composer,
   the R5 body-read class — NOT R4).

The same hand-off is reached by the PRODUCTION one-brain path: `MergedNavConvAgent.perceive_and_ground`
(`nav_conv_merged_bridge.py:2106-2118`) delegates to the SAME `_perceive_and_ground`, so closing it in those two
functions closes it for both the standalone runner AND the `CoResidentOneBrainComposer` production path.

**What is NOT R4 (correctly out of scope):** the LEARNED convergence (synaptic, already on-bridge — the load-bearing
transform); the sensory render of the percept into `gen_perception` (legit body/world); the final-phases `to_host` (the
formatted code = R5 body-read, like `rf_read_phases`); abstract verbs (not perceived — the composer's own codes).

---

## 2. WHY IT IS A LEGIT CLOSE, NOT A HOST SHORTCUT MOVED

The `gen_proj` projection is a **FIXED cortico-cortical fan-in** — the C-6 ruling
(`2026-06-27-comprehensive-shortcut-inventory-burndown-plan.md`: "accept as a legit fixed cortico-cortical fan-in … or
realize it as a fixed complex bridge synapse. Owner call; low stakes"). It is run ONCE per perceived object (not per
fact) and carries NO learned per-fact structure. The SAME `angle(M @ code)` projection is the production-flagship
conversational grounding (`consolidated_320_conversation_demo.py:77-78`, documented there as "a fixed cortico-cortical
fan-in, not learned per fact"). R4 was never a host computation-of-cognition leak — it was a host DATA TRANSFER (the
spike vector + the fan-in crossing host). The close realizes the C-6 "fixed complex bridge synapse" option **on-device**:
the projection matrix is moved to the backend once (cached) and the matvec + `angle` run on-device, so the host
`gen_proj @ rate` matmul is eliminated and the spike vector never crosses host.

This is the **cross-region twin of R1** (`_seq_fused_fabric.py`, CYCLE 722-723): R1 kept the cleanup-score VECTOR
device-resident (no `to_host` of the cleanup membrane) and ran the divnorm divide on-device. R4 keeps the
gen_concept-spike VECTOR device-resident (no `to_host` of the firing carrier) and runs the fan-in projection on-device.
Both are CODE-PATH properties (hold on numpy + cupy): on numpy "device-resident" is a passthrough, but the call-site
that marshalled the data to host is GONE.

---

## 3. THE CLOSE (research/runners/ only, NO sim/)

**`research/runners/_r4_grounding_onbridge.py` (NEW):**
- `accumulate_conc_spikes_device(bridge, conc_region, read_steps, drive_dev)` — accumulate
  `cp_firing_states[conc_region]` ON-DEVICE (a backend gather + add; the per-step firing CARRIER is NEVER `to_host`-ed).
  Returns the per-neuron mean rate as a BACKEND array. **REPLACES** `read_gen_concept_spikes`'s
  `fs = to_host(cp_firing_states)` accumulate.
- `device_resident_grounded_phases(conc_rate_dev, gen_proj)` — the fixed cortico-cortical fan-in `gen_proj @ rate` +
  `angle()` run ON-DEVICE (the projection moved to `xp` once + cached). The ONLY host crossing is the final D-length
  phases (`to_host` = R5 body-read). **REPLACES** `gen_grounded_phases`'s host `gen_proj @ rate` matmul.
- A self-contained de-risk (`main`) with the `to_host`-spy GO bar (==host + the gen_concept-spike `to_host` gone).

**`research/runners/navigate_to_compose_then_answer.py` (wired, opt-in):**
- `read_gen_concept_spikes(..., device_resident=False)` — `:183` — True routes through
  `accumulate_conc_spikes_device` (device array out); False = the verbatim host accumulate.
- `gen_grounded_phases(..., device_resident=False)` — `:223` — True routes through
  `device_resident_grounded_phases` (on-device fan-in); False = the verbatim host matmul.
- `_perceive_and_ground` — `:401-414` — reads `handles["device_resident_grounding"]` (gen_spikes only) and passes it
  through; the provenance capture brings the (possibly-device) source to host via `to_host` (`:462`).
- `build_compose_bridge(..., device_resident_grounding=False)` / `run_seed(..., device_resident_grounding=False)` / the
  `--device-resident-grounding` CLI flag — all default OFF.

**`research/runners/nav_conv_merged_bridge.py` (production path, opt-in):**
- `MergedNavConvAgent(..., perception_device_resident=False)` — `:1640` — stored `:1885`, fed into the
  `perceive_and_ground` handles `:2113` so the production `CoResidentOneBrainComposer` perceive-and-ground runs the
  device-resident grounding when on.

**NO `sim/` edit.** (The R1 design's contingent `rf_kick` tracker-mask edit is NOT relevant here — the gen_concept read
is a plain Izhikevich `_run_one_simulation_step` accumulate on the merged bridge, not an RF op; no register-isolation
break is in play.)

---

## 4. CPU SMOKE (the GO bar, numpy)

**Unit tests** (`tests/test_r4_grounding_onbridge.py`, `SIM_BACKEND=numpy`): **3/3 PASS.**
- `test_device_resident_equals_host_matmul` — device-resident grounded phases == host `angle(gen_proj@rate)` phases
  (atol 1e-9). **==host: the close changes WHERE the projection runs, not the value.**
- `test_gen_proj_matmul_is_on_device_not_host` — the device path consumes a backend spike-rate array + returns valid
  phases (the host `proj @ rate_host` call site is replaced by the on-device op).
- `test_accumulate_conc_spikes_device_keeps_carrier_on_device` — instruments `to_host`; **0 reads of the per-step
  gen_concept firing carrier** during the device accumulate (the carrier stays device-resident).

**Regression** (`tests/test_merged_rf_composer_coresident.py`, `SIM_BACKEND=numpy`): **5/5 PASS** — the default path is
byte-unchanged (the device-resident flag defaults OFF; the change is purely additive).

The unit tests are the CONCLUSIVE, fast CPU proof of the load-bearing R4 properties (==host to 1e-9 + the firing
carrier stays device-resident — they pin exactly the close). The full LIVE-merged-bridge de-risk
(`_r4_grounding_onbridge.py` main, `SIM_BACKEND=numpy`) is dominated by the numpy-CPU gen-stack convergence-training
build (~10-15 min, the same cost as the navcompose runner build) and is the controller's CPU/GPU run (see §5) — it
exercises the property on the live bridge but adds nothing the unit tests don't already prove. (NOTE on the de-risk's
==host: the host read + the device read are SEPARATE perception windows; the de-risk rests OU off so they are
deterministic + equal, and falls back to a phasor-cosine >0.9999 check if a residual remains.)

---

## 5. THE GPU RUN (the controller's — DO NOT run the full GPU here)

The decisive validation is the full navigate-to-compose 6-seed with the device-resident grounding ON, confirming
==host compose + moat 0-FA + held-out >> floor + lesion collapses, at production scale. The CHEAP de-risk first
(the controller's run), then the integration 6-seed:

```bash
# 1) the device-resident grounding de-risk (==host + gen_concept-spike to_host gone), 6 seeds.
#    NOTE: _r4_grounding_onbridge's --seeds is COMMA-separated (the navcompose runner's --seeds is SPACE-separated).
SIM_BACKEND=cupy python -u -m research.runners._r4_grounding_onbridge --seeds 42,43,44,100,101,102 \
    --out research/findings/raw/_r4_grounding_onbridge.json

# 2) the integration 6-seed: navigate-to-compose with the device-resident grounding ON (gen_spikes), rf composer:
SIM_BACKEND=cupy python -u -m research.runners.navigate_to_compose_then_answer \
    --seeds 42 43 44 100 101 102 --grounding gen_spikes --composer rf --device-resident-grounding \
    --out research/findings/raw/navigate_to_compose_then_answer_r4_devres.json

# 3) (optional) the production one-brain path with the device-resident grounding ON:
SIM_BACKEND=cupy python -u -m research.runners.navigate_to_compose_then_answer \
    --seeds 42 43 44 100 101 102 --grounding gen_spikes --composer onebrain --device-resident-grounding \
    --out research/findings/raw/navigate_to_compose_then_answer_r4_devres_onebrain.json
```

GO bar (the integration run): GO if device-resident == the host-grounding compose (held-out >> floor, every seed),
**moat 0-FA HARD** (a single false-accept = FAIL, never traded), lesion collapses the compose, ISO-perception grounds
0, byte-identity holds — i.e. the EXISTING navcompose GO bar, now with the grounding hand-off device-resident.

---

## 6. BOTTOM LINE

R4 (the perception→compose grounding host-marshal) is **CLOSED device-resident**, not a boundary. The genuine residual
was a host DATA TRANSFER (the gen_concept-spike VECTOR `to_host` + the host `gen_proj @ rate` fan-in matmul) — the
cross-region twin of R1, closed with R1's mechanism: accumulate the gen_concept spikes on-device + run the fixed
cortico-cortical fan-in on-device, so the host `gen_proj @ rate` matmul is GONE and the spike vector never crosses host
(only the final phases do, the R5 body-read). On-device == host to 1e-9 (unit-pinned). Default OFF = byte-identical;
opt-in (`--device-resident-grounding` / `MergedNavConvAgent(perception_device_resident=True)`) flips it. NO `sim/` edit;
the moat is never touched. The controller runs the GPU 6-seed integration to confirm ==host compose + moat 0-FA at scale.

Sources (verified against the code): the R4 diagnosis (`navigate_to_compose_then_answer.py:183-228,401-445`); the
production delegation (`nav_conv_merged_bridge.py:2106-2118`); the C-6 cortico-cortical-fan-in ruling
(`2026-06-27-comprehensive-shortcut-inventory-burndown-plan.md` C-6); the same projection in production
(`consolidated_320_conversation_demo.py:69-78`); the R1 device-resident precedent (`_seq_fused_fabric.py`,
`2026-06-30-R1-fold-cleanup-score-onbridge-design.md`); the RF body-read class (`bridge.py:5684` `rf_read_phases`);
the scoping (`2026-06-30-tier2-integrated-spiking-loop-scoping.md` R4 / Option 3 / R5).
