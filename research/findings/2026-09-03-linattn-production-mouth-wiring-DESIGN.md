---
type: finding
status: design
claim_check: synthesis
date: 2026-09-03
mechanism: DESIGN — wire the brain's own `--recurrence linattn` spiking mouth (normalized Hebbian fast-weight linear attention) into the live `/api/brain-chat` open-ended channel as the default FORM generator, brain-state-driven, and retire/minimize the Qwen2.5-0.5B scaffold to a non-default oracle
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [42, 43, 44, 100, 101, 102]
verdict: DESIGN NOTE (no new measurement of the deployable-scale mouth) — maps the shipped production mouth path exactly, specifies the linattn read-side + wiring + flags + verification gate + honest residual. CONDITIONAL on the linattn BPE simplewiki 6-seed holding (seed-42 crossing is +0.049 GO; the full 6-seed was still running when this was written). No `sim/` file and no runner is edited by this doc.
artifacts:
  - research/findings/2026-09-03-spiking-content-addressable-read-DESIGN.md
  - research/runners/_emerge_wkv_lm_derisk.py
  - research/runners/_wkv_fewspike_read_derisk.py
  - webapp/wkv_mouth_generator.py
  - webapp/open_ended_chat.py
  - webapp/server.py
  - research/findings/raw/_linattn_smoke_normON.json
  - research/findings/raw/_linattn_smoke_normOFF.json
  - research/findings/raw/_linattn_smoke_wkv_baseline.json
  - research/findings/2026-08-28-wkv-mouth-into-open-ended-WIRED-GO.md
  - research/findings/2026-08-12-INTEGRATION-default-chat-turn-is-fluent-multi-sentence-mouth-is-external-qwen-cupy.md
---

# DESIGN — make the linattn spiking mouth the production FORM generator, brain-driven, and retire Qwen to an oracle

**This is a DESIGN NOTE, not a measured deployment result.** It is FORWARD PREP, explicitly CONDITIONAL on the
own-voice fluency arc's `--recurrence linattn` mechanism holding at deployable scale. The seed-42 crossing is
`margin_vs_trigram = +0.049` (GO) where every prior mouth mechanism lost; the full BPE Simple-Wiki 6-seed
(`--seeds 42 43 44 100 101 102`) was still running when this was written. The read MECHANISM (normalized Hebbian
fast-weight linear attention) and its single load-bearing lever (the restored num/den normalization) are already
CPU-smoke-confirmed — see §5 — but the deployable-scale verdict is pending. **Nothing here is a claim that linattn
IS the production mouth; it is the wiring spec for when the 6-seed lands.** It edits no `sim/` file and no runner;
it hands a build agent an exact integration map. Companion mechanism doc:
`research/findings/2026-09-03-spiking-content-addressable-read-DESIGN.md` (the read's derivation, the spiking-LM
literature deep-read, and the biology anchors — all NOT re-argued here).

## 1. The current production mouth path — what Qwen does vs what the brain already drives

A live `/api/brain-chat` turn has TWO reply surfaces, and Qwen sits in a DIFFERENT place in each. This map is read
straight from the shipped code (`webapp/server.py::brain_chat`, `webapp/open_ended_chat.py::answer_turn`,
`webapp/wkv_mouth_generator.py`), not inferred.

### Surface A — the DEFAULT strict/rich turn (production default today)

`BRAIN_OPEN_ENDED` is default-OFF, so the live default turn is the strict/rich path
(`webapp/server.py`, `use_rich = req.rich if req.rich is not None else _brain_rich_default()`,
`BRAIN_RICH` default ON; finding
`2026-08-12-INTEGRATION-default-chat-turn-is-fluent-multi-sentence-mouth-is-external-qwen-cupy.md`). Here:

- **The BRAIN drives** the CONTENT end-to-end: the genuinely-spiking onebrain composer
  (`BRAIN_COMPOSER_KIND=onebrain`) recalls the facts, the neural dlPFC planner
  (`neural_planner=True`) orders the multi-sentence discourse, each sentence is re-parsed and VERIFY-moat-gated
  (a gate-sourced fact per sentence; the Paris firewall holds — the Qwen mouth knows Paris, the brain does not
  leak it).
- **A spiking Broca mouth** already words the RECALL/RICH surface for the bounded transitive-SVO frame inventory
  (`2026-08-26-spiking-broca-mouth-recall-surface-production-wirein-GO.md`, 6-seed GO, wired default).
- **Qwen** is the FALLBACK for the open-prose the bounded frames cannot cover (irregular-verb / copula / arbitrary
  prose) — "Touchpoint A" in the crutch-burndown scope
  (`2026-08-28-mouth-crutch-burndown-scope.md`), the LARGER live share of actual Qwen render calls.

### Surface B — the BRAIN_OPEN_ENDED free-generation channel (the mouth arc's target)

When `BRAIN_OPEN_ENDED` is on, `answer_turn` REPLACES the strict/rich reply with a free, first-person,
multi-sentence reply behind the no-confab post-filter moat. Per turn (`webapp/open_ended_chat.py::answer_turn`):

1. `extract_topic(msg)` — host comprehension of the world input (a DECLARED scaffold boundary, same as the SVO
   parser).
2. `retrieve(build_index(...), topic)` — the grounded (agent, action, patient) facts the live store holds
   (`facts.json` `by_agent`); empty ⇒ the genuine abstain / moat. **BRAIN-driven content.**
3. `StateContext(valence, arousal, familiarity, novelty, curiosity, self_model, ...)` — valence/arousal read off
   the real spiking affect organ's differential (`_valence_from_differential`); familiarity/novelty/curiosity
   grounded in whether the store knows the topic. **BRAIN-driven affect + epistemic state.**
4. `build_prompt(state)` → (system, user) — assembles that state into the generator's conditioning context.
5. The FORM generator, in this priority order:
   - `BRAIN_OPEN_ENDED_WKV_MOUTH` (default-ON under the channel) + `in_vocab_scope(msg)` ⇒ the from-scratch WKV/SSM
     spiking cortex (`wkv_ssmU6_v1000_d128`, V=1000 TinyStories, genuine few-spike Izhikevich soft-WTA decode).
   - else `BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK` (default-ON) + a known topic ⇒ `render_fact_sentence` — a
     brain-based fact→clause render via the 6-seed-GO `SpikingClauseProducer` (moat-safe by construction).
   - else `BRAIN_OPEN_ENDED_GEN_TIME_HONESTY` (default-OFF) + a live organ-wired `chat` + a known topic ⇒ Qwen,
     stepped one sentence at a time through the LTM-exempt organ-B/C spiking CONSENSUS VETO.
   - else the one-shot `OpenEndedGenerator.generate` — **Qwen (SpikingQwenFaculty), the sole open-prose FORM
     mouth.**
6. `post_filter(raw, topic, known, facts)` — the VERIFY moat: persona-strip; unknown-topic hedge/abstain;
   known-topic per-sentence contradiction filter (`sentence_contradicts` + `_clause_filter_sentence`); optional
   spiking NP-entailment gate (`BRAIN_OPEN_ENDED_NP_ENTAILMENT`). Applied UNCONDITIONALLY, regardless of which
   generator wrote `raw` — a safety net, never bypassed.

**The precise division of labor.** Qwen contributes exactly ONE thing: fluent surface WORDING of open prose on a
free-generation turn (Surface B, one-shot path) and the open-prose fallback (Surface A, Touchpoint A). Everything
that makes the reply TRUE and the reply the BRAIN'S — the retrieved content, the affect coloring, the
familiarity/novelty/curiosity, the honesty veto (post-filter + generation-time consensus), the fact→clause render
— is already brain-driven and load-bearing (vary the state and the reply changes; lesion the organ and the
affective/empathic lead vanishes — the anti-hollow property the existing organs already carry, memories
"faculties must DRIVE not observe"). **This design changes only WHO does the wording**, and keeps every
brain-driven stage exactly where it is.

| stage | who drives it today | after this design |
|---|---|---|
| topic comprehension | host scaffold (`extract_topic`) | unchanged (declared scaffold) |
| content / facts | BRAIN (store retrieval + no-confab abstain) | unchanged |
| affect (valence/arousal) | BRAIN (spiking affect organ) | unchanged (still conditions the prompt/state) |
| familiarity/novelty/curiosity | BRAIN (store-grounded) | unchanged |
| honesty (post-hoc) | BRAIN-adjacent (`post_filter` VERIFY moat) | unchanged (still runs on linattn output) |
| honesty (generation-time) | BRAIN (organ-B/C spiking consensus veto) | unchanged (composes with linattn) |
| fact→clause render | BRAIN (`SpikingClauseProducer`) | unchanged (still first for covered relations) |
| **surface WORDING (open prose)** | **Qwen scaffold** | **the linattn spiking mouth (this design)** |

## 2. The seam already exists — linattn is a read-family swap in the WKV mouth, not a new pipeline

The crucial fact that makes this a WIRING design and not a rebuild: `webapp/wkv_mouth_generator.py` is ALREADY the
from-scratch spiking-mouth generator slot in `answer_turn`, already default-ON under the channel, already a genuine
non-Qwen spiking cortex with a genuine few-spike read-out, already default-OFF-safe (finding
`2026-08-28-wkv-mouth-into-open-ended-WIRED-GO.md`). It also ALREADY carries the two forward hooks a linattn
checkpoint needs:

- **`BRAIN_WKV_MOUTH_CKPT`** — the checkpoint path (env-overridable, `_CKPT_TEMPLATE`).
- **`BRAIN_WKV_MOUTH_TOKENIZER=bpe`** — a BPE decode/encode mode (`tokenizer_mode()` / `_get_bpe_tokenizer`),
  built 2026-09-03 specifically for the own-voice retrain's future `--save-ssm` BPE checkpoint, reusing
  `sim.bpe_tokenizer.BPETokenizer.decode/encode` verbatim.

What it does NOT yet carry is a linattn READ path. Today the mouth's read is `WKVReadout`
(`research/runners/_wkv_fewspike_read_derisk.py`), which realizes exactly ONE recurrence family: the
ssm/dual-nonneg per-channel state (`advance`/`logits`: `ap = decay·ap + relu(v)`, `an = decay·an + relu(-v)`,
`logits = head_w @ (r_h · (Wo_sp @ [ap;an])) + head_b`). Its own comment block (that file, lines 116-160) is
explicit that the two recurrence families must NEVER share a read path or fall back silently into each other — a
checkpoint trained in the wrong family and read through this math produced near-uniform garbage (NLL ≈ ln V) that
loaded WITHOUT error (finding `2026-07-20-gap1-ROOT-CAUSE-wrong-recurrence-mode-retrain-fixes-catastrophe`).
The class DID gain an additive multi-layer WKV path (`step_wkv`) but that is a HOST decode/verification utility,
explicitly NOT spiking-deployed. **So the single new code artifact this design needs is a `LinAttnReadout` — a
faithful numpy transcription of `LinAttnLayer.forward` in autoregressive O(1)-state form — read out by the SAME
`FewSpikeWordRead` spiking WTA the WKV mouth already uses.** Nothing else in the pipeline moves.

## 3. The wiring design

### (a) Where linattn's generation hooks into the pipeline

`answer_turn`'s generator slot is UNCHANGED. The linattn checkpoint enters through the existing WKV-mouth branch:
`wkv_mouth_enabled()` → `_WKV.in_vocab_scope(msg)` → `_WKV.generate(msg, ..., sentence_facts=..., facts=...)`.
Inside `wkv_mouth_generator.generate`, the only new decision is WHICH readout to build:

```
recurrence_mode()  # "ssm" (default) | "linattn"  — mirrors tokenizer_mode() exactly
  "ssm"     -> WKVReadout(ckpt)          # today's advance()/logits() per-channel spiking read (UNCHANGED)
  "linattn" -> LinAttnReadout(ckpt)      # new: outer-product KV trace read, § below
```

Both expose the SAME tiny interface the driving loop (`_free_gen`) needs: `advance(state, tid) -> state`,
`logits(state, prev_tid) -> [V]`, `.words`, `.D`, `.unk_idx`. `_free_gen`'s body — top-K cut, the repetition
guard, the fact-boost, `reader.read(p)` (the genuine few-spike Izhikevich soft-WTA), the self-NLL bookkeeping,
the BPE/word detokenize — is UNTOUCHED. Only the `state` object and the two calls that produce logits differ, so
the entire spiking read-out and every decode control compose with linattn for free.

### (b) How the brain's content / affect / honesty stay load-bearing

By reuse, not re-implementation — this is the whole point of hooking into the existing slot:

- **Content / grounding.** `retrieve()` + `known` + the `facts` triples are computed BEFORE the generator runs and
  fed to it: `sentence_facts=facts` (the fact→clause render, tried FIRST for a covered relation — so a known-topic
  reply is still a guaranteed-correct brain fact, NOT free-gen) and `facts=facts` (the decode-time fact-grounding
  logit boost, `fact_grounding_ids`/`_apply_fact_boost`). Both are decode-CONTROL over which candidates reach the
  spiking read — legitimately host territory, the read mechanism itself untouched — and both work IDENTICALLY for
  a linattn readout because they operate on the vocab-logits, not the recurrence.
- **Affect + epistemic state.** `valence/arousal/familiarity/novelty/curiosity` are assembled into `StateContext`
  → `build_prompt` → the prompt the mouth conditions on. With a general-vocabulary BPE linattn checkpoint (unlike
  the closed V=1000 TinyStories one), that state prompt is FINALLY expressible in-vocab, so it can actually steer
  the wording — closing the coverage ceiling the TinyStories mouth hit
  (`2026-09-01-wkv-mouth-fact-grounding-lever.md`: ~26% fact coverage). See §6 verification (ii).
- **Honesty.** `post_filter` runs AFTER, unconditionally, on whatever linattn emits — the VERIFY moat is
  generator-agnostic by construction. The generation-time consensus veto
  (`BRAIN_OPEN_ENDED_GEN_TIME_HONESTY`) currently gates only the Qwen one-shot path; §3e notes the one-line
  extension to also step the linattn mouth sentence-by-sentence through the organ-B/C spiking consensus (it drives
  `gen.generate` today; a linattn generator exposing the same `(system,user,seed)->text` shape drops in).

**Load-bearing test built in:** because content/affect/honesty are computed upstream and merely CONSUMED by the
mouth, the anti-hollow property is directly checkable — vary the retrieved facts or the affect differential and
the linattn reply must change; lesion the organ and the affective lead must vanish (§6).

### (c) What of Qwen is retired vs kept as an oracle

Staged, by touchpoint, and NEVER a hard delete (Qwen stays a loadable oracle — memories "keep rf/numpy as oracle",
"retire the transformer to oracle"):

- **Retire first: Surface B, one-shot open-prose path.** Once the 6-seed holds and the routing gate (§3e) admits
  general-vocabulary prompts, the linattn mouth becomes the default open-ended FORM generator. `BRAIN_OPEN_ENDED`
  itself stays default-OFF until an owner-gated channel flip; within the channel, `BRAIN_WKV_MOUTH_RECURRENCE`
  flips to `linattn` at the same owner gate.
- **Retire later: Surface A, Touchpoint A (the bounded-frame recall surface's open-prose fallback).** The larger
  live share; a separate, larger wiring task, out of this design's scope but named as the end-state.
- **Keep Qwen as:** (i) a loadable fallback when the linattn readout raises or a prompt is out of the checkpoint's
  scope (the existing try/except degrade in `answer_turn` already does this — a WKV-path exception falls straight
  to Qwen), and (ii) the A/B QUALITY ORACLE for the verification gate (§6) and future regression checks. "Retire"
  here means "no longer the default wording engine", not "deleted".

### (d) Is the linattn read genuinely spike-native? The honest brain-based-purity status

Stated plainly, per the BRAIN-BASED-ONLY standard (a graded host co-process is a documented shortcut, and the
honest negative is the deliverable):

- **The recurrent STATE is real-valued (graded), by design and by the whole spiking-LM literature.** `M_t` (the
  D×D outer-product KV trace) and `zden_t` (the normalizer) are real-valued fast weights — short-term synaptic
  plasticity (Mongillo-Barak-Tsodyks 2008 graded calcium buffer; Ba et al. 2016 fast weights), the SAME graded
  state SpikeGPT/SpikingSSMs/P-SpikeSSM all keep; only I/O is spiked (companion DESIGN §2 lesson 2, §3). This is
  NOT a new concession — today's deployed ssm mouth ALSO carries real-valued `ap/an` state; linattn changes the
  state's SHAPE (a matrix, not two vectors), not its graded-vs-spiking status.
- **The read-OUT is genuinely spiking, unchanged.** The next-word winner is read by `FewSpikeWordRead` — an
  Izhikevich few-spike soft-WTA over the top-K candidate pools off `cp_firing_states`, NOT a host argmax/softmax
  draw (the GO-verified read the WKV mouth already uses; `research.runners._wkv_fewspike_read_derisk`). linattn
  changes only the logits FED to it.
- **The WRITE is Hebbian, the DIVISION is the residual.** `M += φ(k)⊗v` is a pre×post Hebbian outer product (CA3
  autoassociation). The num/den DIVISION (`read = φ(q)ᵀM / (φ(q)ᵀzden + ε)`) is, at the rate level, host graded
  arithmetic — its on-substrate realization is divisive normalization by a shunting/conductance pool over the
  query's match-mass axis (companion DESIGN §3, §5b). **That on-substrate division is a LATER rung and an explicit
  honest-negative candidate** (Holt & Koch 1997: pure somatic shunting is subtractive, not divisive; the divisive
  effect needs a conductance increase via balanced E/I or dendritic pooling). The deployable rung uses exact
  rate-level division; if the shunting realization degrades, that IS the mapped deliverable.

**Net:** linattn-as-mouth is exactly as spike-native as the shipped ssm mouth — graded recurrent state, genuine
spiking read-out — with one additional graded host op (the normalizer division) whose spiking realization is a
named, in-scope next rung, not a hidden shortcut. The purity residual is DISCLOSED, not resolved (§7).

### (e) Exact integration points + code sketch + flags

Two build artifacts (a runner + a webapp module edit; NEITHER is a `sim/` edit), plus a checkpoint produce step.
All flags mirror the existing `BRAIN_OPEN_ENDED_*` / `BRAIN_WKV_MOUTH_*` family and are DEFAULT-OFF (or
default-preserving) until an owner-gated flip.

**P0 — produce the checkpoint (no code; the trainer already supports it).** Run the deployable-scale linattn
config with `--save-ssm` so the state_dict persists:

```bash
# routed to tools/gpu_queue.sh, NOT an agent (cost-routing); ~1 GPU-h/seed at the arc-standard config
.venv/bin/python -m research.runners._emerge_wkv_lm_derisk \
    --recurrence linattn --n-layers 2 --d-model 192 --linattn-phi elu \
    --tokenizer bpe --corpus data/corpus/simplewiki.txt --contiguous \
    --max-len 40 --n-sentences 1200000 --max-train-sents 1000000 --max-eval-sents 4000 --epochs 5 \
    --seeds 42 43 44 100 101 102 \
    --save-ssm bridges/wkv_ckpt/wkv_linattn_bpe_d192   # -> wkv_linattn_bpe_d192_seed{seed}.npz
```

`--save-ssm` writes `np.savez(..., V, d_model, words, **state_dict)` (trainer lines ~1464-1469). For
`--recurrence linattn` the state_dict keys are: `emb.weight`, `ln.weight`, `ln.bias`, `head.weight`, `head.bias`,
and per layer `i` in `0..n_layers-1`: `linattn_layers.{i}.ln.weight`, `.ln.bias`, `.Wq.weight`, `.Wk.weight`,
`.Wv.weight`, `.Wr.weight`, `.Wo.weight`, `.w` (and `.Wg.weight`/`.Wg.bias` iff `--assoc-gate`). The base
`Wk/Wv/Wr/Wo/Wo_sp/w/u` are also present but UNUSED on the linattn branch (constructed for parity, no gradient —
trainer `forward`, `if RECUR == "linattn"` returns `self._out(hh, aux)` off `linattn_layers` only, never touching
them). A `LinAttnReadout` must therefore read the `linattn_layers.*` namespace and IGNORE the base names — the
same disjoint-namespace discipline `_init_wkv_multilayer` uses for `extra.*`.

**P1 — the read side: `LinAttnReadout` (new class in `research/runners/_wkv_fewspike_read_derisk.py`, additive).**
An O(1)-state autoregressive transcription of `LinAttnLayer.forward`, verified against the torch forward to
float precision by a companion test (mirror `tests/test_wkv_readout_multilayer.py`). State = per-layer
`(M[D,D], zden[D])`; the residual stream `h` threads through the layers exactly as the trainer's
`hh = hh + blk(hh)`:

```python
class LinAttnReadout:
    """Autoregressive O(1)-state read of a --recurrence linattn checkpoint (normalized Hebbian fast-weight linear
    attention). Numpy transcription of LinAttnLayer.forward (research.runners._emerge_wkv_lm_derisk), verified
    against the torch forward. Reads ONLY the linattn_layers.* namespace; the base Wk/Wv/Wr/Wo/Wo_sp/w/u are
    present but unused on this recurrence family and are never touched here (disjoint-namespace, like extra.*)."""
    def __init__(self, ckpt_path):
        W = np.load(ckpt_path, allow_pickle=True)
        self.V = int(W["V"]); self.D = int(W["d_model"]); self.words = list(W["words"])
        self.unk_idx = (len(self.words) - 1) if (self.words and self.words[-1] == "<unk>") else -1
        self.emb = W["emb.weight"].astype(np.float64)
        self.ln_w = W["ln.weight"].astype(np.float64); self.ln_b = W["ln.bias"].astype(np.float64)
        self.head_w = W["head.weight"].astype(np.float64); self.head_b = W["head.bias"].astype(np.float64)
        self.layers = []                                   # one dict per linattn block, index-ordered
        j = 0
        while f"linattn_layers.{j}.Wq.weight" in W.files:
            p = f"linattn_layers.{j}."
            lam = np.exp(-np.log1p(np.exp(W[p + "w"].astype(np.float64))))   # exp(-softplus(w)) in (0,1)
            self.layers.append({
                "ln_w": W[p+"ln.weight"], "ln_b": W[p+"ln.bias"],
                "Wq": W[p+"Wq.weight"], "Wk": W[p+"Wk.weight"], "Wv": W[p+"Wv.weight"],
                "Wr": W[p+"Wr.weight"], "Wo": W[p+"Wo.weight"], "lam": lam,
                "Wg": (W[p+"Wg.weight"] if (p+"Wg.weight") in W.files else None),
                "Wg_b": (W[p+"Wg.bias"] if (p+"Wg.bias") in W.files else None),
                "phi": str(W["linattn_phi"]) if "linattn_phi" in W.files else "elu",
                "norm": bool(W["linattn_norm"]) if "linattn_norm" in W.files else True,
            })
            j += 1
        if not self.layers:
            raise RuntimeError("not a linattn checkpoint (no linattn_layers.* keys) — use WKVReadout for ssm/wkv")

    def init_state(self):
        return [{"M": np.zeros((self.D, self.D)), "zden": np.zeros(self.D)} for _ in self.layers]

    @staticmethod
    def _phi(x, kind):
        if kind == "relu":   return np.maximum(x, 0.0) + 1e-3
        if kind == "exp":    z = x - x.max(); return np.exp(z)
        if kind == "sparse":
            kth = np.sort(x)[::-1][max(1, x.shape[-1] // 8) - 1]; return np.maximum(x - kth, 0.0)
        return np.maximum(x, 0.0) - np.maximum(-x, 0.0) * 0.0 + np.where(x > 0, x, np.exp(x) - 1.0) + 1.0  # elu+1

    def _ln(self, v, w, b):
        return (v - v.mean()) / (v.std() + 1e-5) * w + b

    def advance_and_logits(self, state, tid):     # one autoregressive step: update state with tid, THEN read
        h = self.emb[tid].copy()                  # layer 0 input = token embedding (no base residual, like trainer)
        for lyr, st in zip(self.layers, state):
            z = self._ln(h, lyr["ln_w"], lyr["ln_b"])
            q = self._phi(lyr["Wq"] @ z, lyr["phi"]); k = self._phi(lyr["Wk"] @ z, lyr["phi"]); v = lyr["Wv"] @ z
            r = 1.0 / (1.0 + np.exp(-(lyr["Wr"] @ z)))
            st["M"]    = lyr["lam"][:, None] * st["M"] + np.outer(k, v)   # M += phi(k) (x) v  (Hebbian pre x post)
            st["zden"] = lyr["lam"] * st["zden"] + k                      # running normalizer trace
            num = q @ st["M"]; den = float(q @ st["zden"])
            read = num / (den + 1e-6) if lyr["norm"] else num            # the restored num/den normalization
            if lyr["Wg"] is not None:
                read = (1.0 / (1.0 + np.exp(-(lyr["Wg"] @ z + lyr["Wg_b"])))) * read   # --assoc-gate trust gate
            h = h + lyr["Wo"] @ (r * read)                               # pre-norm residual: h = h + delta
        return state, self.head_w @ self._ln(h, self.ln_w, self.ln_b) + self.head_b
```

(The final `self.ln` before `head` matches the trainer's `_out(hidden) = head(hidden)` where `hidden` is the
post-block residual stream; a build agent confirms whether the trainer applies a final LayerNorm before `head` and
mirrors it exactly — the ONE ambiguity to pin against the torch forward.) An adapter (`advance`/`logits` split, or
a thin `_free_gen` that calls `advance_and_logits`) matches the existing driving-loop contract so
`FewSpikeWordRead`, the repetition guard, and the fact-boost all apply unchanged.

**P2 — the routing side: `webapp/wkv_mouth_generator.py` (additive, default-preserving).**

```python
_RECUR_ENV = "BRAIN_WKV_MOUTH_RECURRENCE"          # "ssm" (default) | "linattn" — mirrors _TOKENIZER_ENV exactly
def recurrence_mode() -> str:
    v = os.environ.get(_RECUR_ENV, "ssm").strip().lower()
    return "linattn" if v == "linattn" else "ssm"

# in _get_readout(seed): pick the readout family (byte-identical when unset -> WKVReadout, today's path)
def _build_readout(path):
    if recurrence_mode() == "linattn":
        from research.runners._wkv_fewspike_read_derisk import LinAttnReadout
        return LinAttnReadout(path)
    return WKVReadout(path)                          # unchanged default
```

**P3 — the scope/routing gate.** The TinyStories `in_vocab_scope` (function-word/content-word overlap over a
closed V=1000) is NOT meaningful over a general-vocabulary BPE checkpoint (already flagged as BPE residual #3 in
`wkv_mouth_generator.py`). Add a scope mode mirroring the tokenizer/recurrence pattern:

```python
_SCOPE_ENV = "BRAIN_WKV_MOUTH_SCOPE"               # "vocab" (default, today's in_vocab_scope) | "broad"
def scope_mode() -> str: ...
# "broad": a BPE checkpoint tokenizes ANY input (no OOV), so scope becomes a COVERAGE/CONFIDENCE decision, not a
# hard vocab gate — e.g. admit when the BPE subword-coverage fraction >= a threshold, else fall to Qwen. The exact
# threshold is a de-risk knob, set from the 6-seed's own held-out coverage, NOT guessed here.
```

**P4 — flags summary (all default-OFF / default-preserving; owner-gated flip only):**

| flag | default | effect |
|---|---|---|
| `BRAIN_OPEN_ENDED` | OFF | the whole open-ended channel (unchanged) |
| `BRAIN_WKV_MOUTH_RECURRENCE` | `ssm` | `linattn` ⇒ build `LinAttnReadout` instead of `WKVReadout` |
| `BRAIN_WKV_MOUTH_CKPT` | ssm ckpt | point at `wkv_linattn_bpe_d192_seed{seed}.npz` |
| `BRAIN_WKV_MOUTH_TOKENIZER` | `word` | `bpe` ⇒ decode/encode via `BPETokenizer` (already shipped) |
| `BRAIN_WKV_MOUTH_SCOPE` | `vocab` | `broad` ⇒ BPE-coverage routing instead of TinyStories in_vocab_scope |

The production-default flip is: `BRAIN_OPEN_ENDED=1` (channel on) + `BRAIN_WKV_MOUTH_RECURRENCE=linattn` +
`BRAIN_WKV_MOUTH_CKPT=<linattn bpe ckpt>` + `BRAIN_WKV_MOUTH_TOKENIZER=bpe` + `BRAIN_WKV_MOUTH_SCOPE=broad`. Every
one is default-off/default-preserving, so with the flip un-set the live turn is BYTE-IDENTICAL to today.

## 4. The linattn mechanism is already built in the trainer

`LinAttnLayer` (the read this design deploys) is ALREADY implemented in
`research/runners/_emerge_wkv_lm_derisk.py` (lines ~574-692), with the `--recurrence linattn` forward dispatch,
`--linattn-phi {elu,relu,exp,sparse}`, `--no-linattn-norm`, `--assoc-gate`, `--uniform-decay`, `--n-layers`, and
the memoryless/permute anti-cheats — exactly the §5d spec of the companion DESIGN doc. So the mouth's read
mathematics exist and are tested at NLL scale; this doc supplies only the DEPLOYMENT read-back
(`LinAttnReadout`) + the webapp routing.

## 5. What is already de-risked (the CPU smoke), and what the 6-seed must still show

The companion DESIGN's §6 cheapest experiment (normalization ON vs OFF, one variable, minutes on CPU) has ALREADY
RUN (word-level TinyStories smoke, V=800, d96, 1-seed). At the deepest bucket d10-99, `margin_vs_trigram`:

| arm | d10-99 margin_vs_trigram | anti-cheats (memoryless / perm NLL) | reading |
|---|---|---|---|
| linattn norm ON (φ=elu) | **+0.456** | 5.178 / 6.884 vs wkv 3.505 | best; uses long-range content + order |
| linattn norm OFF | **+0.190** | 4.857 / 7.183 vs wkv 3.770 | normalization dropped ⇒ margin halves |
| exact-wkv baseline (same config) | +0.429 | 4.021 / 7.354 vs wkv 3.531 | linattn ≥ the wkv upper bound it generalizes |
| linattn φ=exp | +0.435 | — | ≈ elu |
| linattn φ=sparse (k-WTA) | +0.358 | — | sharper read slightly worse here |

Sources: `research/findings/raw/_linattn_smoke_{normON,normOFF,phi_exp,phi_sparse,wkv_baseline}.json`.

**This decisively confirms the mechanism's core lever on CPU:** the num/den normalization is load-bearing
(0.456 → 0.190 when removed), linattn matches/beats the exact-wkv bound it strictly generalizes (0.456 ≥ 0.429),
and the built-in anti-cheats pass (memoryless and permuted are both much worse — the mouth genuinely uses
long-range, order-dependent content, not the current token). **What the smoke does NOT establish, and the
running 6-seed must:** that this carries to DEPLOYABLE scale — BPE tokenizer, Simple-Wiki (not TinyStories),
d192, n_layers=2, and all 6 non-negotiable seeds (42/43/44/100/101/102), where the seed-42 crossing (+0.049) is a
much thinner margin against a FAIR interpolated trigram. **This design is CONDITIONAL on that 6-seed holding**
(memories/board 6-seed rule; `feedback_6seed_validation`). A NO-GO there redirects to the objective/token levers
the verdict names (`2026-09-03-ordered-attention-...-verdict.md`), and this wiring waits.

## 6. The verification gate — "is it real" for linattn-as-mouth

Three properties, each a concrete measurement, gating the production-default flip. (The build/verify is routed to
`tools/gpu_queue.sh` + a verify runner, NOT an agent — cost-routing.)

**(i) FLUENT — the trigram-beating carries to real turns.** GATE: the 6-seed deployable
`margin_vs_trigram > 0` at the deep bucket holds (the arc's own crux), AND a live-turn read: sample N real
`/api/brain-chat` open-ended turns through the linattn mouth and confirm coherent, multi-sentence,
non-degenerate prose (no loop/repeat collapse — the repetition guard already ships). A/B against the Qwen oracle
on the SAME turns for a human-read quality reference (Qwen kept exactly for this).

**(ii) BRAIN-GROUNDED — the anti-hollow test (vary → changes; lesion → vanishes).** The property the owner
requires ("faculties must DRIVE not observe"). Two measurements, both already expressible because content/affect
are computed upstream and consumed by the mouth:
- VARY: hold the prompt, change the retrieved `facts` (or the affect differential) → the linattn reply's
  wording/content must change. If the reply is invariant to the brain state, the mouth is hollow (a free-runner
  ignoring its conditioning) — a NO-GO.
- LESION: cut the affect organ's `affect_out` (the existing `BRAIN_AFFECTIVE_TOM_LESION`-style lever) → the
  affective/empathic coloring must collapse to neutral and the affective lead must vanish. The fact→clause and
  fact-boost paths give a second lesion: remove `facts` → the guaranteed-correct clause disappears and the reply
  falls to free-gen (which the moat then must still keep honest — property iii).

**(iii) HONEST — the no-confab moat still holds.** The VERIFY moat is generator-agnostic (`post_filter` runs on
whatever `raw` is), so the gate is a re-run of the existing moat-safety soak
(`_open_ended_bundle_moat_safety_soak.py` family) with `BRAIN_WKV_MOUTH_RECURRENCE=linattn`: on a
brain-UNKNOWN / Qwen-known topic the reply must NOT leak the unknown fact (fabrication stays ≈0), and on a known
topic wrong-supplement rate must not regress vs the shipped mouth. A BPE general-vocabulary mouth is a NEW
fabrication surface (it CAN now word arbitrary claims the TinyStories mouth structurally could not), so this is
the load-bearing new risk, not a formality — the fact→clause-FIRST routing (a guaranteed-correct brain fact
replaces free-gen for covered relations) is the primary mitigation, and the gen-time consensus veto is the
second.

### Failure modes (and the banked response for each)

1. **6-seed does not hold at deployable scale** (BPE/Simple-Wiki/d192 erases the seed-42 crossing). Then linattn
   is not yet the mouth; redirect to the objective (`--pred-aux-weight`) + token levers the verdict names. This
   design waits, unspent — no wiring lands.
2. **Fluent NLL-win, but hollow** (ii fails: reply invariant to brain state). The mouth is a free language model
   ignoring its conditioning. Response: strengthen the conditioning that is load-bearing — the fact→clause and
   fact-boost paths (which are grounded BY CONSTRUCTION) carry more of the turn, and the free-gen share shrinks,
   until the vary-test passes. An honest negative here (the free-gen surface cannot be made state-dependent
   enough) is itself the deliverable — it maps what the mouth can/can't do grounded.
3. **New fabrication surface** (iii regresses: the general-vocab mouth confabulates where the closed-vocab one
   couldn't). Response: the fact→clause-first routing already answers covered known topics with a guaranteed
   brain fact (no free token can appear); tighten the `BRAIN_WKV_MOUTH_SCOPE=broad` coverage threshold so
   low-confidence prompts fall to the (moat-checked) Qwen path rather than a free linattn paragraph.
4. **`LinAttnReadout` ≠ the torch forward** (a transcription bug reads garbage that loads without error — the
   2026-07-20 silent-wrong-recurrence class). Response: the companion test gates deployment — assert the numpy
   readout matches the torch `LinAttnLayer.forward` logits to float precision on a tiny synthetic checkpoint
   BEFORE any live wiring, exactly as `test_wkv_readout_multilayer.py` does for `step_wkv`.
5. **D×D state throughput regression** vs the per-channel ssm read. At d192 the matrix is 192²≈37k floats — trivial
   on the single-3090 consumer reference; a perf tune (chunked scan), not a wall.

## 7. The honest brain-based-purity residual (what stays host)

Disclosed, not resolved — the standing standard makes the honest negative the deliverable:

1. **The num/den DIVISION is rate-level host arithmetic at the deployable rung.** Its spiking realization
   (divisive normalization by a shunting/conductance pool over the match-mass axis) is a LATER rung and an
   explicit honest-negative candidate (Holt & Koch 1997). The deployable mouth is spike-native in state (graded
   M/zden, same as today's ssm mouth) and read-out (FewSpikeWordRead), with this ONE graded op named.
2. **The recurrent WRITE learning rule.** As with the shipped WKV checkpoint, whether the linattn weights are
   produced by a local rule vs host-BPTT is a training-method question the mouth's INFERENCE-time spiking status
   does not settle (the same residual `wkv_mouth_generator.py` already carries). The trainer trains by BPTT
   today; a local-rule linattn train is a separate arc.
3. **Topic comprehension + the state→prompt assembly stay host scaffolds** — the declared boundary
   (`extract_topic`, `build_prompt`), unchanged by this design and identical to the SVO parser's boundary.
4. **The decode controls (top-K, repetition guard, fact-boost, scope routing) are host** — legitimately, the same
   category as every existing decode knob; the read mechanism (`reader.read(p)`) is never touched.

None of these is NEW to linattn — items 1-2 are the same class the shipped ssm mouth carries — and each is a
named next rung, not a hidden shortcut.

## 8. Hand-off (staged; NONE of it is this doc's to run)

- **P0 (GPU queue, not an agent):** the `--save-ssm` linattn BPE 6-seed run (§3e P0). Gate on the 6-seed
  `margin_vs_trigram > 0` verdict FIRST — if it does not hold, STOP (failure mode 1).
- **P1 (one build agent, worktree-isolated, sonnet):** `LinAttnReadout` + its torch-parity test (§3e P1) — a
  mechanical transcription against a spec, verified by an exact-match test.
- **P2 (same or a second build agent):** the `webapp/wkv_mouth_generator.py` routing (`recurrence_mode`,
  `_build_readout`, `scope_mode`) — additive, default-preserving, byte-identical when unset (§3e P2/P3), with a
  wiring-verify test in the `tests/test_wkv_mouth_*` family.
- **P3 (verify runner, GPU queue):** the three-property gate (§6) — fluent + brain-grounded + honest — on the real
  `answer_turn`, A/B'd against the Qwen oracle.
- **Owner-gated flip:** only after all three pass, set the §3e P4 flip flags as the production default; keep Qwen
  as the loadable oracle/fallback (Surface A Touchpoint A remains a later, separate wiring task).

## Provenance

Shipped code read this session (2026-09-03): `webapp/open_ended_chat.py` (the full `answer_turn` pipeline +
flag family), `webapp/wkv_mouth_generator.py` (the WKV mouth + `in_vocab_scope` + BPE mode + fact-clause render),
`webapp/server.py` (the `answer_turn` call site ~L4697 + the strict/rich default), `research/runners/
_wkv_fewspike_read_derisk.py` (`WKVReadout` + `FewSpikeWordRead` + the multi-layer read), `research/runners/
_emerge_wkv_lm_derisk.py` (`LinAttnLayer` ~L574-692, the `RECUR=="linattn"` forward dispatch ~L865, the
`--save-ssm` state_dict save ~L1464). Findings cited: the companion mechanism DESIGN
(`2026-09-03-spiking-content-addressable-read-DESIGN.md`), the fluency verdict
(`2026-09-03-ordered-attention-at-shared-fluency-bound-investigation-verdict.md`), the dual-nonneg NO-GO
(`2026-09-03-spiking-mouth-ssm-dualnonneg-fluency-NO-GO-first-brain-based-baseline.md`), the WKV-mouth wire-in
(`2026-08-28-wkv-mouth-into-open-ended-WIRED-GO.md`), the crutch-burndown scope
(`2026-08-28-mouth-crutch-burndown-scope.md`), the default-turn integration
(`2026-08-12-INTEGRATION-default-chat-turn-is-fluent-multi-sentence-mouth-is-external-qwen-cupy.md`), the
spiking-Broca recall surface (`2026-08-26-spiking-broca-mouth-recall-surface-production-wirein-GO.md`), and the
fact-grounding coverage ceiling (`2026-09-01-wkv-mouth-fact-grounding-lever.md`). Measurements: the linattn CPU
smokes `research/findings/raw/_linattn_smoke_*.json` (read directly). This doc edits no `sim/` file and no runner.
