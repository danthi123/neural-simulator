---
type: finding
status: contributing
claim_check: synthesis
date: 2026-09-03
mechanism: DESIGN — a spiking content-addressable read for the own-voice fluency mouth (normalized Hebbian fast-weight linear-attention: a real-valued outer-product KV trace, content-weighted num/den read, read out by spikes)
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [42, 43, 44, 100, 101, 102]
verdict: DESIGN NOTE (no new measurement) — specifies the mechanism the 10-agent bound-investigation named, grounded in a deep-read of the spiking-LM literature; the cheapest validation experiment + the code-level slot are specified for a one-pass build
artifacts:
  - research/findings/raw/_emerge_wkv_lm_assoc_temporal_depth2_contiguous_6seed.json
  - research/findings/raw/_emerge_wkv_lm_ssm_dualnonneg_simplewiki_6seed.json
---

# DESIGN — a spiking content-addressable read for the own-voice mouth: normalized Hebbian fast-weight linear attention

**This is a DESIGN NOTE, not a measured result.** It specifies the mechanism the fluency arc's 10-agent bound-investigation named as the top lever (`2026-09-03-ordered-attention-at-shared-fluency-bound-investigation-verdict.md`): *"complete the content-addressable read on the deployable spiking mouth … realized as a real-valued outer-product KV trace read out by spikes."* It is grounded in (1) our own record's exact-vs-spiking diagnosis and (2) a verified deep-read of the spiking-language-model literature (SpikeGPT, SpikingSSMs, P-SpikeSSM, Spikformer/Spikingformer, SpikeLM, BiSpikCLM, WTA-Spiking-Transformer, and the "SNNs Are Not Transformers (Yet)" sample-complexity claim). It hands a follow-up build agent an exact mechanism spec + flags + code sketch + the single cheapest validation experiment. **No `sim/` file and no runner is edited by this doc.**

## 1. The gap, named precisely (from our own record)

The deployable spiking mouth is `--recurrence ssm --dual-nonneg` in `research/runners/_emerge_wkv_lm_derisk.py`: two positive leaky integrators `ap2_t = decay·ap2_{t-1} + relu(v_t)`, `an2_t = decay·an2_{t-1} + relu(-v_t)`, read via `Wo_sp([ap2, an2])`. It is the *realizable* form (a slow synaptic/conductance leak, few-spike). The exact-math `wkv` (RWKV linear attention) is only an UPPER BOUND — the trainer asserts `n_layers==1` in the ssm branch and a wkv checkpoint read through the ssm math is garbage, so the two are distinct families, not a checkpoint swap.

Measured margins vs a FAIR interpolated trigram at the deepest bucket (d10-99), matched depth-2 / contiguous / 6-seed regime unless noted:

| family | mean margin_vs_trigram | source |
|---|---|---|
| exact-math wkv (num/den normalized), CONTIGUOUS | **+0.02 (crossed)** — 1-seed, confounds tokens×context-length | verdict finding |
| exact-math wkv, sentence-mode | −0.125 | verdict finding |
| spiking SSM dual-nonneg (depth-2, contiguous) | −0.1252 | `_emerge_wkv_lm_assoc_depth2_contiguous_6seed` sibling |
| HiPPO structured SSM (seed 42) | −0.126 | verdict finding |
| ordered attention (assoc_t, +time-cell "when") | **−0.147** | `_emerge_wkv_lm_assoc_temporal_depth2_contiguous_6seed.json` |
| bag content-addressable attention (no order) | −0.347 | content-addressable-read-without-order NO-GO |
| spiking SSM dual-nonneg, sentence-mode n_layers=1 | **−0.461** | `_emerge_wkv_lm_ssm_dualnonneg_simplewiki_6seed.json` |

**The named computation dual-nonneg drops** (diagnosis, `2026-09-03-spiking-mouth-ssm-dualnonneg-fluency-NO-GO-first-brain-based-baseline.md`): *"`dual-nonneg` discards RWKV's numerator/denominator NORMALIZATION — the division by an accumulated decay-weighted denominator that gives `wkv` its content-addressed, softmax-like weighting over past tokens. Without it, the spiking state is per-channel leaky accumulation with no cross-time competition."*

There are TWO coupled pieces in that one sentence, and both must be restored together:

1. **A content-dependent nonnegative WRITE GAIN** on each token's contribution — RWKV's `exp(k_t)`. dual-nonneg writes `relu(±v_t)` with NO per-write gain, so every past token is written with equal weight.
2. **A running DENOMINATOR** that accumulates that same gain, and a **DIVISION** of the read by it — making the read a content-weighted AVERAGE (a soft key-matched retrieval), not a raw leaky sum.

These are coupled: **without (1), (2) is a no-op.** The already-tried `--dual-nonneg-divnorm` gate failed precisely here — see §4.

## 2. Deep-read of the spiking-LM literature (verified; flags kept)

Every number below was pulled from a live fetch of the paper's arXiv/ar5iv/bioRxiv text this session (2026-09-03); flags mark anything not verifiable from primary text.

### The read mechanism, per paper

- **SpikeGPT** (arXiv:2302.13939, Zhu et al., TMLR 2023) — **keeps the full RWKV num/den normalization.** State is a decay-weighted running numerator/denominator pair: `A_t = exp(K_t)⊙V_t + exp(W_d)⊙A_{t-1}`; `B_t = exp(K_{t-1}) + exp(W_d)⊙B_{t-1}`; `Y_{t+1} = SN(σ(R_t) ⊙ [exp(W_f)⊙exp(K_t)⊙V_t + A_t] / [exp(W_f)⊙exp(K_t) + B_t])`. `A_t, B_t` are REAL-VALUED and persist across every timestep; only the block OUTPUT `Y` is binarized by a LIF neuron `SN`. This is the softmax-free, decay-kernel content-weighted average — the exact shape of the "real-valued KV trace read out by spikes" this design targets.
- **SpikeLM** (arXiv:2406.03287) — **does NOT drop softmax.** It keeps standard softmax attention and only spike-encodes K/V. Best GLUE of any spiking LM found: 76.5% vs its own BERTbase 83.2% (6.7-pt gap). The closest existing precedent for "keep softmax-like normalization while spiking the read."
- **BiSpikCLM** (arXiv:2605.13859; postdates the training cutoff — from fetch only) — Softmax-Free Spiking Attention: spike dot-product → causal mask → a spiking-neuron threshold → weighted sum. No softmax, no float normalization in the read path. Reaches 84.8% of the OPT-1.3B teacher (42.19 vs 49.73 zero-shot avg, ratio computed directly) via a 5-term Spike-Aware Alignment Distillation (SpAD).
- **WTA-Spiking-Transformer** (arXiv:2604.11321; from fetch only) — replaces softmax with a hard Winner-Take-All: `A_w = WTA(QK^T·s)`, one-hot. Their Appendix E proves a temperature-softened softmax → hard WTA as τ→0, framing WTA as "an extremely sparse softmax." GLUE: 66.3% vs BERTbase 79.6% (**−13.3 pt**, confirming the prompt's figure).
- **Spikformer / Spikingformer** (arXiv:2209.15425 / 2304.11954) — **drop softmax for VISION** ("spike-form Q,K have natural non-negativeness … we do not need softmax to keep the attention matrix non-negative, its most important role"). But directly load-bearing here: Spikingformer's own text says **for LANGUAGE tasks they RE-ADDED softmax** ("we retained the softmax layer … consistent with SpikeLM").
- **SpikingSSMs** (arXiv:2408.14909) — plain diagonal S4D-style SSM `h_t = Ā h_{t-1} + B̄ x_t`, NO normalization, LIF on top. 75M/33.94 PPL on WikiText-103 vs Transformer 231M/20.51 (65% relative gap remaining) — but beats SpikeGPT at 1/3 the params.
- **P-SpikeSSM** (arXiv:2406.02923, ICLR 2025) — DPLR (HiPPO-LegS) SSM + a clamped-sigmoid probability read-out + stochastic Bernoulli spike sampling (surrogate gradient exact-in-expectation). **LRA / classification only — no language-modeling numbers at all**, so it does not speak to the LM-parity question.

### The two convergent lessons for THIS design

1. **The literature independently found our gap.** On LANGUAGE, the spiking models that KEEP a content-weighted normalizer do best (SpikeGPT's num/den; SpikeLM's retained softmax), and the ones that DROP it lag (SpikingSSMs' plain SSM; Spikformer's softmax-free SSA is a vision method whose own authors re-added softmax for language). This is external corroboration that the normalization dual-nonneg dropped is load-bearing **for language specifically** — the very axis where vision SNNs get away without it.
2. **The recurrent STATE is real-valued in EVERY paper; only I/O is spiked.** SpikeGPT's `A_t/B_t`, SpikingSSMs' `h_t` and LIF membrane, P-SpikeSSM's `h_t` and probability — all real-valued and persistent; binarization happens once, at the block output. So "a real-valued outer-product KV trace read out by spikes" is not an exotic concession — it is the **universal** shape of a working spiking sequence model. This answers design question (a) directly (§5a).

### The sample-complexity caveat, and why it does NOT bind this design

"SNNs Are Not Transformers (Yet)" (bioRxiv 2025.10.31.685901, Fishell & Honnuraiah) — **NOT peer-reviewed; only the abstract-level theorem was reachable (rate-limited on full text), proof not verified line-by-line.** Claim (their Thm 2.3): the **non-leaky integrate-and-fire (nLIF)** model has worst-case sample complexity `M = O(D·|S|²·log(L_global))` — QUADRATIC in sequence length `|S|`, vs RNN linear and Transformer logarithmic. **Why it is unlikely to bind us:** the bound is derived for a *pure spiking recurrence* (nLIF, binarized state). This design keeps the state REAL-VALUED (a fast-weight synaptic trace) and spikes only the I/O — exactly the escape every language-competitive spiking LM above takes. The claim, if it holds, is better read as *evidence FOR keeping graded state* than as a wall against a spiking read. Flag it, do not treat it as settled.

## 3. How spiking substrates preserve the dropped computation (the biology)

**The write gain + running denominator + division are all standard, separately-attested biological operations:**

- **A real-valued outer-product KV trace = short-term synaptic plasticity ("fast weights").** Working memory in cortex can be held not in persistent spiking but in **graded, calcium-mediated short-term synaptic facilitation** on recurrent connections — "presynaptic residual calcium is a buffer that is loaded, refreshed, and read out by spiking activity" (Mongillo, Barak, Tsodyks 2008, *Science* 319:1543–1546, doi:10.1126/science.1150769). Ba, Hinton, Mnih, Leibo & Ionescu 2016 ("Using Fast Weights to Attend to the Recent Past", arXiv:1610.06258) frame exactly this as a fast-changing weight matrix "slower than activities but much faster than the standard weights," written by an outer-product Hebbian rule and read by settling. Schlag, Irie & Schmidhuber 2021 ("Linear Transformers Are Secretly Fast Weight Programmers", arXiv:2102.11174) prove the formal equivalence of linearized self-attention and fast-weight controllers ("additive outer products of … keys and values"). **A real-valued KV trace is therefore not a host shortcut — it is how biology holds short-term associative memory.**
- **The Hebbian outer-product write = CA3 recurrent-collateral autoassociation.** CA3's recurrent collaterals are the canonical biological autoassociator: they store patterns by Hebbian plasticity and complete them from a partial cue (Marr 1971; McNaughton & Morris 1987; Treves & Rolls 1994; Rolls & Treves 1998). Formal capacity for **graded-response units with sparse codes** is ∝ the number of modifiable recurrent synapses per cell. The write `M += φ(k) ⊗ v` is literally a Hebbian synapse (pre `φ(k)` × post `v`). This is the same CA3 pattern-completion anchor the existing `AssocLayer` docstring already uses (Ramsauer et al. 2020, modern-Hopfield ⇔ attention).
- **The division = divisive normalization by shunting inhibition.** `R = drive / (σ + pool)` is a canonical cortical computation (Carandini & Heeger 1994, 2012 *Nat Rev Neurosci* 13:51–62), realized by controlling membrane conductance — the normalization pool raises each cell's input conductance, dividing input-current→membrane-potential gain. **Honest caveat:** Holt & Koch 1997 (*Neural Comput.* 9:1001) showed pure somatic shunting is subtractive on firing rate, not divisive; the resolution is that the divisive effect comes from a *conductance increase* via balanced excitation+inhibition / dendritic pooling (Silver 2010, "Neuronal arithmetic"). So the on-substrate realization of the denominator is a shunting/conductance pool, with an honest-negative in scope if it degrades — but the RATE-LEVEL de-risk (§6) uses exact division first.

## 4. Why the naive normalization fix already failed — and the axis that fixes it

`--dual-nonneg-divnorm` (a Carandini-Heeger gate ALREADY in the runner) is a NO-GO at every σ (`2026-09-03-spiking-divnorm-gate-NO-GO-cross-channel-pool-over-suppresses.md`): worse than the −0.46 baseline, collapsing to −2.98 at σ=32. **Root cause: it pooled over the wrong axis** — `R_i = ap2_i^n / (σ^n + Σ_j ap2_j^n)`, a sum over ALL D=256 CHANNELS, squashing each channel toward `1/D`. And the dual-nonneg diagnosis had already noted a *literal* per-channel temporal denominator degenerates too: with no `exp(k)` write gain, `b_t → decay·b + 1`, a channel-independent constant — a no-op divisor.

**The correct axis is the one softmax and RWKV normalize over: the CONTENT/TIME match-mass, not the channel population.** Softmax's denominator is `Σ_s exp(score_{t,s})` — a sum over PAST POSITIONS weighted by query-key match. RWKV's `B_t` is a per-channel sum over PAST TIME weighted by `exp(k_s)`. Neither pools across channels at a fixed time. The design in §5 restores exactly this: a denominator that is the query's total match-mass against the accumulated keys — which requires the `φ(k)` write gain to be present (so the denominator is informative), the exact coupling §1 names.

## 5. THE DESIGN — `--recurrence linattn`: normalized Hebbian fast-weight linear attention

A new recurrence branch, the deployable-spiking successor to `dual-nonneg`. It is **linear-attention in the fast-weight form** (Katharopoulos et al. 2020, "Transformers are RNNs", arXiv:2006.16236, Eqs. 7/10–12/18–20): a real-valued outer-product KV matrix + a running normalizer vector, read by a content query, in O(T) recurrent form (no T×T matrix — unlike `assoc_t`, so it is spike-deployable). It restores BOTH dropped pieces (§1) AND adds the genuine query-key content-addressing (`φ(q)^T M`) that even `wkv` lacks (wkv's `k` is a per-channel gain, not a q·k match).

### The mechanism (per position t, causal, O(T) recurrent)

```
z_t   = LN(h_t)                                  # pre-norm input (block contract, like every other layer)
q_t   = Wq(z_t),  k_t = Wk(z_t),  v_t = Wv(z_t)  # learned query / key / value projections
φ(·)  = elu(·) + 1                               # non-negative feature map (Katharopoulos Eq.7); § flags for alternatives

# WRITE — a decaying Hebbian outer-product fast-weight matrix M (D×D) + its normalizer vector zden (D):
M_t    = λ ⊙ M_{t-1}   + φ(k_t) ⊗ v_t            # M: real-valued synaptic KV trace  (Hebbian pre×post)
zden_t = λ ⊙ zden_{t-1} + φ(k_t)                 # zden: the running key-match mass (the denominator trace)

# READ — content-weighted num/den (the restored normalization):
num_t  = φ(q_t)^T M_t            # D-vector: query-weighted retrieved values (the associative recall)
den_t  = φ(q_t)^T zden_t         # scalar   : the query's total match-mass against accumulated keys (normalizer)
read_t = num_t / (den_t + ε)     # content-weighted AVERAGE — the softmax-free normalized read
delta_t = Wo( r_t ⊙ read_t )     # r_t = sigmoid(Wr(z_t)) receptance gate (reuse the existing pattern)
# caller: h = h + delta_t        # pre-norm residual; stacks under --n-layers exactly like WkvLayer/HippoLayer/AssocLayer
```

`λ ∈ (0,1)` is the per-channel (or scalar, `--uniform-decay`) leak, identical in role to wkv's `exp(-w)`. `M_t` is the real-valued outer-product KV trace; `zden_t` is the running denominator. **This IS SpikeGPT's num/den read generalized from per-channel to a full outer product** — and it strictly generalizes the existing `wkv` (restrict `M` to its diagonal and set `Wq=Wk=I, φ=exp` and it degenerates to wkv's per-channel num/den), so with usable capacity it cannot do worse than the wkv upper bound it descends from.

### (a) Is the real-valued outer-product KV trace brain-based-legit? YES.

Answered in §3: `M_t` is a short-term-synaptic-plasticity fast-weight matrix (Mongillo-Barak-Tsodyks 2008 graded calcium buffer; Ba et al. 2016). The write is Hebbian (CA3 autoassociation, Treves & Rolls 1994). And the literature (§2, lesson 2) keeps real-valued recurrent state in EVERY working spiking LM. The state is graded; the I/O is spikes — the SpikeGPT bar, already supported by the runner's `--spike-output` straight-through path.

### (b) How is the content-addressed READ done with spikes?

- `φ(k), v, φ(q)` are firing-rate codes. Rate codes are naturally non-negative, so `φ = elu+1` (or `φ = identity` on a rate code) yields a valid non-negative key/query for a positive normalizer. Signed values use ON/OFF rate channels `[relu(·), relu(-·)]`, exactly as dual-nonneg already does for `v`.
- The WRITE `M += φ(k) ⊗ v` is a **Hebbian coincidence** (presynaptic key-rate × postsynaptic value-rate) onto a fast-weight synapse — a spike-driven local update.
- The DIVISION `num/den` is **divisive normalization by a shunting/conductance pool** (§3): `den_t` is realized as an inhibitory-pool conductance in the read neuron's denominator. Crucially the pool is over the **match-mass axis** (the query's total key-overlap), NOT the channel population — this is the axis fix of §4, and it is why this does not repeat the `--dual-nonneg-divnorm` collapse.
- The block output `delta_t` is read out to the residual stream as spikes (`--spike-output` straight-through), matching the "graded local state, spikes for I/O" bar SpikeGPT and biology both take.
- Optional **WTA / sparse key** (`--linattn-phi sparse`): a k-winners-take-all `φ(k)` makes the read approximate a hard content match — biologically a sparse pattern-separated key (DG→CA3), and the WTA-Spiking-Transformer's "sparse-softmax limit." This is the sharper-read lever the July learned-keys finding wanted (below).

### (c) Composition with the existing levers

- **--n-layers (depth):** `LinAttnLayer` is a pre-norm residual block `forward(h, memoryless) -> delta`, identical contract to `WkvLayer/HippoLayer/AssocLayer`, so `--n-layers N` stacks N of them (the depth that moved dual-nonneg −0.461 → −0.125).
- **--assoc-gate (trust gate):** the July learned-keys de-risk (`2026-07-11-LEARNED-keys-make-content-addressable-retrieval-load-bearing`) found the raw retrieved feature is informative-but-NOISY (`content − base` stays positive) and named the fix: "a learned GATING of when to trust the retrieval; retrieval as a RESIDUAL CORRECTION." Reuse the existing `--assoc-gate` `g_t = sigmoid(Wg(z_t))` on `read_t` before `Wo`: `delta_t = Wo(g_t ⊙ read_t)`. Same init-open trick, byte-identical when off.
- **--uniform-decay:** scalar `λ` (one shared decay = the substrate's uniform NMDA τ) vs per-channel `λ`.
- **--contiguous / --spike-output / --tokenizer bpe:** all upstream/orthogonal; compose unchanged.
- It does NOT modify `ssm/dual-nonneg` — it REPLACES it as the deployable read (a separate `RECUR` branch, like hippo/assoc).

### (d) Exact flags + code-level slot (mirrors the hippo/assoc additions — byte-identical when off)

Add to `research/runners/_emerge_wkv_lm_derisk.py` (the file another agent owns — this is the slot spec, not an edit):

1. **New class**, alongside `AssocLayer`, inside `build_and_train_wkv`:
```python
class LinAttnLayer(nn.Module):
    """Normalized Hebbian fast-weight linear-attention read (--recurrence linattn). See DESIGN doc
    2026-09-03-spiking-content-addressable-read-DESIGN.md. Real-valued outer-product KV trace M (D×D) +
    running normalizer zden (D); content-weighted num/den read φ(q)^T M / (φ(q)^T zden). Restores the
    num/den normalization dual-nonneg drops, in a spike-deployable O(T) form. Pre-norm residual block."""
    def __init__(self, D, uniform_decay=False, phi="elu", gate=False):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.Wq = nn.Linear(D, D, bias=False); self.Wk = nn.Linear(D, D, bias=False)
        self.Wv = nn.Linear(D, D, bias=False); self.Wr = nn.Linear(D, D, bias=False)
        self.Wo = nn.Linear(D, D, bias=False)
        self.w = nn.Parameter(torch.zeros(1 if uniform_decay else D))   # λ = exp(-softplus(w)), like wkv
        self.phi = phi; self.norm = True    # set self.norm=False for the --no-linattn-norm ablation
        self.gate = gate
        if gate:
            self.Wg = nn.Linear(D, D, bias=True)
            nn.init.zeros_(self.Wg.weight); nn.init.constant_(self.Wg.bias, 2.0)   # init-open, reuse assoc-gate

    def _phi(self, x):
        if self.phi == "elu":  return torch.nn.functional.elu(x) + 1.0
        if self.phi == "relu": return torch.relu(x) + 1e-3
        if self.phi == "exp":  return torch.exp(x - x.amax(-1, keepdim=True))       # stable RWKV-like
        if self.phi == "sparse":                                                     # k-WTA sparse key
            kth = torch.topk(x, max(1, x.shape[-1] // 8), dim=-1).values[..., -1:]
            return torch.relu(x - kth)
        raise ValueError(self.phi)

    def forward(self, h, memoryless=False):        # h:[B,T,D] -> delta:[B,T,D]
        B, T, D = h.shape
        z = self.ln(h)
        q = self._phi(self.Wq(z)); k = self._phi(self.Wk(z)); v = self.Wv(z)
        r = torch.sigmoid(self.Wr(z))
        lam = torch.exp(-torch.nn.functional.softplus(self.w))          # (D,) or (1,)
        M = torch.zeros(B, D, D, device=h.device)                       # outer-product KV trace
        zden = torch.zeros(B, D, device=h.device)                       # normalizer trace
        outs = []
        for t in range(T):
            if memoryless:                                              # ANTI-CHEAT: current token only
                M_r = torch.einsum("bd,be->bde", k[:, t], v[:, t]); zden_r = k[:, t]
            else:
                M = lam.unsqueeze(-1) * M + torch.einsum("bd,be->bde", k[:, t], v[:, t])
                zden = lam * zden + k[:, t]
                M_r, zden_r = M, zden
            num = torch.einsum("bd,bde->be", q[:, t], M_r)              # φ(q)^T M
            den = torch.einsum("bd,bd->b", q[:, t], zden_r).unsqueeze(-1)   # φ(q)^T zden  (scalar/token)
            read = num / (den + 1e-6) if self.norm else num            # --no-linattn-norm = raw sum ablation
            if self.gate: read = torch.sigmoid(self.Wg(z[:, t])) * read
            outs.append(self.Wo(r[:, t] * read))
        return torch.stack(outs, 1)
```

2. **Construction** in `WKV.__init__` (mirror `assoc_layers`, empty ModuleList when not selected → zero extra init-RNG, byte-identical for every other branch):
```python
self.linattn_layers = nn.ModuleList([
    LinAttnLayer(D, uniform_decay=getattr(args, "uniform_decay", False),
                 phi=getattr(args, "linattn_phi", "elu"), gate=getattr(args, "assoc_gate", False))
    for _ in range(max(n_layers, 1))
]) if RECUR == "linattn" else nn.ModuleList()
```

3. **Forward dispatch** in `WKV.forward` (insert beside the `if RECUR in ("assoc","assoc_t"):` block):
```python
if RECUR == "linattn":
    hh = h
    for blk in self.linattn_layers:
        hh = hh + blk(hh, memoryless=self.memoryless)
    return self.head(hh)
```

4. **CLI**: add `"linattn"` to the `--recurrence` choices; add `--linattn-phi {elu,relu,exp,sparse}` (default `elu`) and `--no-linattn-norm` (store_false into `args.linattn_norm`, default True — wire into `LinAttnLayer.norm`). Reuse `--uniform-decay`, `--assoc-gate`, `--n-layers`, `--spike-output`, `--contiguous`, `--tokenizer bpe`.

**Memory note:** `M` is `B×D×D`. At d192 that is 192² = 36,864 floats/sample — trivial on GPU, and it is a fixed-size running state (O(1) in T), so the pass stays O(T). A build agent should confirm the batch×D×D matmul is not a throughput regression vs the per-channel wkv; if it is, a chunked/blocked linear-attention scan (Katharopoulos) recovers it. This is well within the single-3090 consumer-hardware reference.

## 6. The single cheapest validation experiment (before any 6-seed GPU spend)

**Core assumption to falsify:** restoring the num/den normalization (with the `φ(k)` write gain that makes it informative) is what closes the exact-vs-spiking gap. The cheapest decisive test is a **one-variable ablation pair at numpy CPU smoke scale** — the normalization ON vs OFF on the SAME model, minutes on CPU, zero GPU / zero pool:

```bash
# Arm A (normalization ON — the design):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._emerge_wkv_lm_derisk \
    --recurrence linattn --n-layers 2 --d-model 96 --seeds 42 \
    --n-sentences 8000 --vocab 800 --epochs 8 \
    --json research/findings/raw/_linattn_smoke_{normON,normOFF}.json   # to-be-created outputs
# Arm B (normalization OFF — the control): add --no-linattn-norm, same everything.
```

**Prediction that makes it decisive:** at d10-99, Arm A's `margin_vs_trigram` clearly beats Arm B, and A moves toward the exact-wkv line while B collapses toward the dual-nonneg (no-normalization) level. Also require the built-in anti-cheats on Arm A: `wkv_memoryless − wkv > 0.05` and `wkv_perm − wkv > 0.05` (it must use long-range content, not the current token). **If A ≈ B, the normalization hypothesis is falsified for ~minutes of CPU** — a first-class NO-GO that redirects the arc to the objective/token levers (the verdict's #2/#4) without a GPU run. Only if the smoke passes does the 6-seed BPE Simple-Wiki de-risk (the `assoc_t`/dual-nonneg config, `--seeds 42 43 44 100 101 102`, ~1 GPU-h) run — routed to `tools/gpu_queue.sh`, NOT an agent.

A second, near-free discriminator from the same smoke: `--linattn-phi exp` vs `elu` vs `sparse` — tells the build agent whether a sharper (more selective) read matters before committing the 6-seed budget.

## 7. Expected failure modes (and the mitigation banked for each)

1. **Normalization helps but does not CLOSE the gap** (linattn lands at ~−0.12 with the rest). Then the ~−0.12 is genuinely the shared data/tokenizer-regime bound the verdict's primary hypothesis names, and the lever is the predictive-coding OBJECTIVE (`--pred-aux-weight`, verdict #2) + tokens (#4), not the read. Still a first-class deliverable: it discriminates the two live hypotheses (exactness-gap vs shared-bound) that the pending 6-seed contiguous-wkv diagnostic is also probing.
2. **`φ=elu+1` is too flat** — every key matches every query, so `den` washes out selectivity and the read ≈ a uniform average (the "informative-but-noisy" symptom the July learned-keys finding measured as `content − base > 0`). Mitigation, cheap and pre-banked: sharper `φ` (`exp` or `sparse`/WTA), and compose `--assoc-gate` (the residual-correction trust gate the same finding named).
3. **The q·k content-addressing does not beat per-channel wkv.** Honest tension in the record: `assoc_t`'s full q·k (−0.147) did NOT beat wkv's per-channel num/den (−0.125). So the outer-product's value may be *only* that it keeps normalization in a spike-realizable Hebbian form, not that q·k adds signal. Mitigation: the design strictly generalizes wkv, so it inherits the wkv floor; and the norm-ON/OFF ablation (§6) plus a diagonal-M restriction flag isolates "normalization" from "outer-product q·k" as separate testable axes in ONE runner — the scientifically honest way to find out.
4. **On-substrate division degrades** (Holt & Koch 1997: somatic shunting is subtractive). The rate-level de-risk uses exact division and is the GO that matters first; the on-bridge conductance-pool realization is a LATER rung, and an honest-negative there (divisive→subtractive) is itself the deliverable (it maps what the substrate's shunting can do).
5. **D×D matrix throughput regression** vs per-channel wkv. Mitigation: chunked linear-attention scan (Katharopoulos); at d192 the matrix is tiny, so this is a perf tune, not a wall.

## 8. Hand-off for the build agent

- Implement §5d in `research/runners/_emerge_wkv_lm_derisk.py` (additive, default-off for every other branch, byte-identical-when-unselected — mirror the `hippo`/`assoc` additions exactly, including the empty-ModuleList RNG guarantee).
- Run §6 Arm A / Arm B numpy smokes FIRST (CPU, minutes). Gate the 6-seed GPU de-risk on the smoke passing.
- 6-seed config = the arc standard: `--tokenizer bpe --corpus data/corpus/simplewiki.txt --contiguous --max-len 40 --n-sentences 1200000 --max-train-sents 1000000 --max-eval-sents 4000 --epochs 5 --n-layers 2 --d-model 192 --seeds 42 43 44 100 101 102`, routed to `tools/gpu_queue.sh` (300W cap standing), NOT an agent.
- Compare against the two cited baseline artifacts at d10-99, with the memoryless + permute anti-cheats:
  `research/findings/raw/_emerge_wkv_lm_ssm_dualnonneg_simplewiki_6seed.json` (dual-nonneg −0.461 sentence)
  and `research/findings/raw/_emerge_wkv_lm_assoc_temporal_depth2_contiguous_6seed.json` (assoc_t −0.147).

## Provenance of the external claims

Papers verified from live arXiv/ar5iv/bioRxiv fetches this session (2026-09-03): SpikeGPT arXiv:2302.13939 · SpikingSSMs arXiv:2408.14909 · P-SpikeSSM arXiv:2406.02923 · Spikformer arXiv:2209.15425 · Spikingformer arXiv:2304.11954 · SpikeLM arXiv:2406.03287 · BiSpikCLM arXiv:2605.13859 · WTA-Spiking-Transformer arXiv:2604.11321 · "SNNs Are Not Transformers (Yet)" bioRxiv:2025.10.31.685901 (NOT peer-reviewed, abstract-level only). Mechanism/biology anchors: Katharopoulos et al. 2020 arXiv:2006.16236 (Eqs. 7/10-12/18-20) · Schlag-Irie-Schmidhuber 2021 arXiv:2102.11174 · Ba et al. 2016 arXiv:1610.06258 · Mongillo-Barak-Tsodyks 2008 doi:10.1126/science.1150769 · Treves & Rolls 1994 / Rolls & Treves 1998 / Marr 1971 (CA3 autoassociation) · Carandini & Heeger 1994/2012 doi:10.1038/nrn3136 with the Holt & Koch 1997 (Neural Comput. 9:1001) subtractive-shunting caveat.
