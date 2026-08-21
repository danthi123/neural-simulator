"""THE GROUNDED-LANGUAGE ARC CAPSTONE -- the END-TO-END integration de-risk.

The REAL spiking Qwen faculty (P1) renders the BRAIN's retrieved facts (P2 store) into fluent prose,
GATED + VERIFIED (P3's gate->constrain->verify loop), so the no-confab moat the whole conversational arc
built is PRESERVED *even with a real generative LLM in the loop*. This REPLACES the P3 template-stub
faculty with the converted-Qwen spiking forward of `_grounded_lang_p1b_stepB1_forward_derisk` (T=16 GO,
ppl 1.08x ANN, coherent generation; or T=8 ppl 1.21x if T=16 is too slow per-generation).

The three layers, per query (the P3 architecture, now with a REAL generative faculty):
  (i)  GATE      -- the brain's composer exact-match recall returns the stored SVO fact OR ABSTAINS
                    (what_does/who_does -> None ; is_it_true -> 'unknown'). On abstain the faculty is given
                    NOTHING -> no generation (the MOAT). The GATE is the numpy-CPU brain (parser+composer).
  (ii) CONSTRAIN -- the REAL spiking Qwen faculty is PROMPTED to render the gated SVO as one short sentence
                    keeping the same verb (the variant-D constrain prompt: "Turn the triple (a, v, p) into
                    one grammatical sentence. Keep the same verb 'v'. Reply with only the sentence."). The
                    faculty's freedom is grammar/phrasing; the prompt+grounded content pin the meaning.
  (iii) VERIFY   -- the faculty's GENERATED PROSE is re-parsed back into an SVO by the BRAIN: a content
                    extractor recovers the 3 curriculum content tokens (handling determiners + verb
                    inflection -- eat->eats/ate, chase->chased, make->makes, live->lives in/lived) and the
                    SAME BridgeParser (agent.parse) re-assigns roles; the re-parsed {agent, action, patient}
                    must MATCH the gated fact, else the output is REJECTED. A real LLM can DRIFT (a synonym
                    'consume' for 'eat', an added object, a swapped role) -- VERIFY catches exactly that.

  Layers (i)+(iii) are the moat-preservers; (ii) is the fluency producer. The DIFFERENCE from P3: the
  faculty is now a REAL hallucination-capable generative model, NOT a content-locked template. The VERIFY
  re-parse is therefore GENUINELY load-bearing -- it reads the model's actual prose, not a handed-back SVO.

THE KEY RESULT (test (c)): show the moat HOLDS even with the real generative LLM. Two paths:
  - UNTAUGHT cue -> the GATE abstains -> the faculty is never invoked -> no sentence (no chance to confab).
  - DRIFT/CONFAB -> an adversarial prompt steers the faculty toward a WRONG fact (a wrong patient injected
    into the prompt). The faculty emits a fluent-but-FALSE sentence; VERIFY re-parses it, the content
    mismatches the gated fact -> REJECTED. The false assertion never reaches the user.

Metrics (a small set -- the spiking faculty, while fast at T=16 ~0.3-1.2s/gen, is run FOREGROUND):
  (a) GROUNDED   -- ~4 grounded queries -> a fluent sentence whose re-parsed SVO matches the taught fact.
  (b) UNTAUGHT   -- ~2 untaught cues -> the GATE abstains -> NO sentence (the MOAT).
  (c) DRIFT      -- ~1 adversarial steered-to-wrong-fact -> VERIFY re-parse REJECTS it (content mismatch).
  Plus a regenerate-on-reject demonstration (drift caught -> re-prompt tighter -> verified or hard-abstain).

VERDICT: GO = the real spiking faculty renders grounded facts FLUENTLY (re-parse matches) AND untaught->
abstain AND any drift->caught-by-verify -> the END-TO-END grounded-language capability works (a spiking
fluent faculty + brain-grounded content + the no-confab moat, COMBINED). Or HONEST: where it leaks (the
faculty too verbose / drops/adds content / the render doesn't re-parse cleanly) + what it needs.

FOREGROUND/blocking by design. GPU (RTX 3090). The spiking faculty forward is PyTorch OFF the bridge; the
brain (parser/composer) is the numpy-CPU pipeline (SIM_BACKEND defaults to numpy for the brain half). NO
`sim/` edit (the faculty machinery is reused-by-import from the P1b runner; the brain half from P2/P3).

Usage:
  python -m research.runners._grounded_lang_integration_derisk           # T=16 (default), full small set
  python -m research.runners._grounded_lang_integration_derisk --T 8     # faster faculty, ppl 1.21x
  python -m research.runners._grounded_lang_integration_derisk --max-new-tokens 20
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

# the brain half is a numpy-CPU pipeline; pin it to numpy so the (parser/composer) build is portable +
# does not contend with the PyTorch CUDA faculty for the GPU. (The faculty forward is its own torch device.)
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# --- the BRAIN half (reused VERBATIM from P2/P3; the parser+composer pipeline is already GO) ---
from research.runners.brain_conversational_agent import BrainConversationalAgent
from research.runners._grounded_lang_p2_derisk import _collect_vocab, _teach, CURRICULUM

OUT = _REPO / "research" / "findings" / "raw" / "_grounded_lang_integration_derisk.json"


# =================================================================================================
# VERIFY content extraction: recover the canonical SVO from a real LLM's GENERATED PROSE.
# The brain's BridgeParser is position-only over 3 content tokens; a real sentence has determiners +
# inflection, so we first map the prose back to the 3 curriculum content tokens (the genuine re-parse
# of the faculty's output -- NOT a handed-back SVO). Inflection is mapped by a curriculum-derived table.
# =================================================================================================
def _build_inflection_map(verbs):
    """Map every surface verb form a faculty might emit -> the curriculum base verb. Covers 3rd-person-sg
    present (eats/chases/makes/gives/likes/lives), simple past (ate/chased/made/gave/liked/lived), the
    PROGRESSIVE (eating/chasing/making -- a real generative LLM commonly uses 'is eating'), and the 'live in'
    phrasal the curriculum's 'live' fact implies. These are SURFACE-fluency inflections the faculty is free to
    choose; the VERIFY extractor normalizes them back so a true fact stated in any aspect still re-parses."""
    irregular_past = {"eat": "ate", "give": "gave", "make": "made"}   # the curriculum's irregular pasts
    m = {}
    for v in verbs:
        m[v] = v
        # 3rd-person-sg present
        if v.endswith(("s", "sh", "ch", "x", "z")):
            m[v + "es"] = v
        else:
            m[v + "s"] = v
        # regular past (-ed / -d)
        if v.endswith("e"):
            m[v + "d"] = v
        else:
            m[v + "ed"] = v
        # progressive (-ing): drop a trailing 'e' (make->making, chase->chasing, give->giving, like->liking)
        stem = v[:-1] if v.endswith("e") else v
        m[stem + "ing"] = v
        # irregular past
        if v in irregular_past:
            m[irregular_past[v]] = v
    # progressive/copular auxiliaries the faculty may insert ('is eating', 'are chasing') are NOT content -- they
    # are dropped by the extractor naturally (they are not in agents/actions/patients), so the -ing main verb above
    # is what the extractor keys on.
    return m


def _extract_svo_from_prose(prose, agents, actions, patients, inflect):
    """Recover (agent, action, patient) content tokens from a faculty's generated PROSE, in surface order.

    Returns the 3 canonical tokens [a, v, p] (curriculum base forms) if all three roles are found, else
    None (the prose did not yield a clean SVO -> VERIFY cannot confirm -> reject). This is the real re-parse
    of the model's output: it strips function words (determiners, 'in'/'the'/'a'), normalizes verb
    inflection back to the base form, and reads the content words in the order they appear -- exactly what a
    downstream comprehension stage does to a heard sentence."""
    # tokenize: lowercase words only (drop punctuation)
    toks = re.findall(r"[a-z]+", prose.lower())
    found_agent = None
    found_action = None
    found_patient = None
    a_idx = v_idx = p_idx = None
    for i, t in enumerate(toks):
        base_v = inflect.get(t)                      # is this token a (possibly-inflected) curriculum verb?
        if found_action is None and base_v in actions:
            found_action, v_idx = base_v, i
            continue
        # a noun: could be the agent (before the verb) or the patient (after)
        if t in agents and found_agent is None and (v_idx is None or i < v_idx):
            found_agent, a_idx = t, i
        elif t in patients and (v_idx is not None) and (found_patient is None) and i > v_idx:
            found_patient, p_idx = t, i
    # a salvage pass: if the agent slot is still empty but a known agent-noun appears anywhere before the verb
    if found_agent is None and v_idx is not None:
        for i, t in enumerate(toks[:v_idx]):
            if t in agents:
                found_agent, a_idx = t, i
                break
    # the patient: first known patient-noun AFTER the verb (already captured); salvage if a known patient
    # appears after the verb but wasn't in the `patients` order set captured above
    if found_patient is None and v_idx is not None:
        for i in range(v_idx + 1, len(toks)):
            if toks[i] in patients:
                found_patient, p_idx = toks[i], i
                break
    if found_agent and found_action and found_patient:
        return [found_agent, found_action, found_patient]
    return None


# =================================================================================================
# The REAL spiking Qwen faculty (P1b stepB1): load the model, install the calibrated graded-read spiking
# ops, and render a gated SVO into fluent prose. Reused-by-import from the P1b runner (NO duplication of the
# convert machinery; NO sim/ edit). The faculty's ONLY job is surface form -- the brain supplies + verifies
# the content.
# =================================================================================================
class SpikingQwenFaculty:
    """The converted Qwen2.5-0.5B-Instruct spiking forward as the fluent renderer. Builds the model once,
    runs the P1b calibration pass to size the SiLU/exp banks, installs the spiking ops at pool-budget T."""

    # the CONSTRAIN prompt (variant D from the integration smoke -- best in-vocab + shortest of the tested
    # set; keeps the verb verbatim so the VERIFY re-parse recovers the action). The faculty's freedom is
    # determiners + inflection; the prompt + grounded content pin the meaning.
    CONSTRAIN_TEMPLATE = ("Turn the triple ({a}, {v}, {p}) into one short grammatical sentence. "
                          "Keep the same verb '{v}'. Reply with only the sentence.")
    # a TIGHTER regenerate prompt used after a VERIFY reject (more explicit, forces the exact 3 words).
    REGEN_TEMPLATE = ("Write exactly one short sentence that means '{a} {v} {p}'. "
                      "Use the words {a}, {v}, and {p}. Reply with only the sentence, nothing else.")

    def __init__(self, T=16, max_new_tokens=24, seed=42, device="cuda"):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from research.runners import _grounded_lang_p1b_stepB1_forward_derisk as B1
        self._torch = torch
        self._B1 = B1
        self.T = int(T)
        self.max_new_tokens = int(max_new_tokens)
        self.seed = int(seed)

        t0 = time.time()
        # DTYPE by device: float16 on CUDA (the GPU path, byte-identical to before); float32 on CPU. Half-precision
        # matmul/attention kernels are largely UNIMPLEMENTED on CPU in PyTorch ("addmm"/"baddbmm" for Half raise
        # NotImplementedError), so a CPU load MUST be float32 or generation crashes on the first forward. float32 is
        # slower but correct — the intended CPU trade (2026-08-21: enable a real Qwen render on GPU-less hosts). The
        # spiking-op banks are already float32 and upcast internally, so they compose with either model dtype.
        _is_cuda = str(device).startswith("cuda")
        _dtype = torch.float16 if _is_cuda else torch.float32
        self.tok = AutoTokenizer.from_pretrained(B1.MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            B1.MODEL_ID, dtype=_dtype, attn_implementation="eager").to(device).eval()
        self.device = next(self.model.parameters()).device
        self.load_seconds = round(time.time() - t0, 2)

        B1.SPK.gen = torch.Generator(device=self.device)
        B1.SPK.eps = float(self.model.config.rms_norm_eps)

        # --- P1b CALIBRATION PASS (SPK disabled): size the SiLU-input + softmax-logit banks on real acts ---
        held, _corpus = B1.held_out_text()
        B1.SPK.enabled = False
        cal_layer = self.model.model.layers[12]
        gate_outs = []

        def gate_hook(_m, _a, output):
            gate_outs.append((output.detach().float().min().item(), output.detach().float().max().item()))

        h1 = cal_layer.mlp.gate_proj.register_forward_hook(gate_hook)
        logit_min = {"v": math.inf}
        real_softmax = F.softmax

        def cal_softmax(inp, *a, **k):
            if inp.dim() == 4:
                x = inp.detach().float()
                m = x.max(dim=-1, keepdim=True).values
                sh = x - m
                valid = sh > -1e4
                if valid.any():
                    logit_min["v"] = min(logit_min["v"], float(sh[valid].min().item()))
            return real_softmax(inp, *a, **k)

        cal_ids = self.tok(held[:3000], return_tensors="pt").to(self.device)
        F.softmax = cal_softmax
        with torch.no_grad():
            self.model(**{k: v[:, :512] for k, v in cal_ids.items()})
        F.softmax = real_softmax
        h1.remove()
        silu_range = (min(g[0] for g in gate_outs), max(g[1] for g in gate_outs))
        self.measured_ranges = {"silu_min": silu_range[0], "silu_max": silu_range[1],
                                "logit_shift_min": logit_min["v"]}

        # --- BUILD BANKS + INSTALL the spiking ops at pool-budget T ---
        B1.SPK.silu_bank, _ = B1.make_silu_bank(silu_range, self.device)
        B1.SPK.exp_bank, _ = B1.make_exp_bank(self.device)
        self.install_info = B1.install_spiking_ops(self.model)
        B1.SPK.set_T(self.T)
        B1.SPK.enabled = True
        self.pools = {"pool_silu": B1.SPK.pool_silu, "pool_div": B1.SPK.pool_div,
                      "pool_softmax": B1.SPK.pool_softmax}

    def _generate(self, user_msg, seed=None):
        """One greedy (deterministic) spiking generation from a chat prompt. Returns (first_line, seconds)."""
        torch = self._torch
        B1 = self._B1
        sd = self.seed if seed is None else int(seed)
        msgs = [{"role": "user", "content": user_msg}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        torch.manual_seed(sd)
        if B1.SPK.gen is not None:
            B1.SPK.gen.manual_seed(1000 + sd)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens,
                                       do_sample=False, pad_token_id=self.tok.eos_token_id)
        new = out[0, ids.input_ids.shape[1]:]
        txt = self.tok.decode(new, skip_special_tokens=True)
        first_line = txt.strip().split("\n")[0].strip()
        return first_line, txt.strip(), round(time.time() - t0, 2)

    def render_svo(self, a, v, p, seed=None):
        """CONSTRAIN: prompt the spiking faculty to render the gated SVO as one short sentence (verb kept)."""
        return self._generate(self.CONSTRAIN_TEMPLATE.format(a=a, v=v, p=p), seed=seed)

    def render_svo_regen(self, a, v, p, seed=None):
        """A TIGHTER re-prompt after a VERIFY reject (forces the exact 3 content words)."""
        return self._generate(self.REGEN_TEMPLATE.format(a=a, v=v, p=p), seed=seed)

    def render_svo_adversarial(self, a, v, wrong_p, seed=None):
        """An ADVERSARIAL render: steer the faculty toward a WRONG patient (the drift/confab the moat must
        catch). The prompt itself injects the false content -> the faculty emits a fluent-but-FALSE sentence;
        VERIFY re-parses the prose, the content mismatches the GATED (true) fact -> reject."""
        return self._generate(self.CONSTRAIN_TEMPLATE.format(a=a, v=v, p=wrong_p), seed=seed)


# =================================================================================================
# The GATE -> CONSTRAIN(spiking render) -> VERIFY loop (one query), with the REAL faculty.
# =================================================================================================
def grounded_reply(agent, faculty, q, vocab_sets, faculty_mode="constrain", allow_regen=True):
    """Run one query through the three-layer grounding loop with the REAL spiking Qwen faculty.

    `faculty_mode`: 'constrain' (the normal grounded render) or 'adversarial' (steer to a wrong patient).
    `allow_regen`: on a VERIFY reject in 'constrain' mode, RE-PROMPT the faculty tighter (the REGEN_TEMPLATE,
    which forces the 3 content words + subject-first order) ONCE and re-verify -- the realistic production
    recovery path (the P3 spec's reject/regenerate). The 0.5B faculty occasionally object-fronts under the
    loose constrain prompt ('Rabbit chased fox' for fox/chase); a tighter prompt recovers a faithful render.
    Disabled for the adversarial path (the drift MUST stay caught -- we never 'recover' a steered-wrong fact).
    Returns a structured record (gate_svo, surface prose, reparsed SVO, verified, emitted, abstained)."""
    agents, actions, patients, inflect = vocab_sets
    qtype = q["type"]
    cue = q["cue"]
    truth = None

    # (i) GATE -- exact-match recall over the spiking store; abstains (None / 'unknown') when no fact matches
    if qtype == "patient":
        content = agent.what_does(cue[0], cue[1])
        gate_svo = [cue[0], cue[1], content] if content is not None else None
    elif qtype == "agent":
        content = agent.who_does(cue[0], cue[1])
        gate_svo = [content, cue[0], cue[1]] if content is not None else None
    elif qtype == "yesno":
        truth = agent.is_it_true(cue[0], cue[1], cue[2])
        gate_svo = [cue[0], cue[1], cue[2]] if truth != "unknown" else None
    else:
        raise ValueError(f"unknown query type {qtype!r}")

    rec = {"cue": cue, "type": qtype, "gate_svo": gate_svo, "gate_truth": truth}

    # gate abstained -> the faculty is given NOTHING -> no generation (the MOAT)
    if gate_svo is None:
        rec.update({"surface": None, "surface_full": None, "reparse_svo": None,
                    "verified": None, "emitted": False, "abstained": True, "gen_seconds": 0.0})
        return rec

    # (ii) CONSTRAIN -- the REAL spiking faculty renders the gated content into fluent prose
    a, v, p = gate_svo
    if faculty_mode == "adversarial":
        wrong_p = q["wrong_patient"]
        surface, surface_full, gen_s = faculty.render_svo_adversarial(a, v, wrong_p)
    else:
        surface, surface_full, gen_s = faculty.render_svo(a, v, p)

    # (iii) VERIFY -- re-parse the faculty's GENERATED PROSE back into an SVO (content extraction + the brain's
    # BridgeParser role assignment); the re-parsed SVO must match the GATED fact, else REJECT (drift caught).
    def _verify(prose):
        """Return (reparse_svo_or_None, verified_bool, reason). reparse_svo is None when the prose did not
        yield a clean 3-token SVO (verb drifted out of vocab / a role missing -- a real generative-faculty
        leak signature, e.g. 'consume' for 'eat', or a role inversion 'Rabbit chased fox')."""
        csvo = _extract_svo_from_prose(prose, agents, actions, patients, inflect)
        if csvo is None:
            return None, False, "prose did not re-parse to a clean SVO"
        parsed_ = agent.parse(csvo, voice="active")            # the brain's comprehension of the recovered SVO
        rsvo = [parsed_.get("agent"), parsed_.get("action"), parsed_.get("patient")]
        return rsvo, (rsvo == gate_svo), (None if rsvo == gate_svo else "re-parsed SVO mismatches the gated fact")

    reparse_svo, verified, reason = _verify(surface)
    regen_used = False
    if (not verified) and allow_regen and faculty_mode != "adversarial":
        # the loose constrain prompt produced an unverifiable render (drift/role-inversion). RE-PROMPT tighter
        # (forces the 3 content words + subject order) ONCE, then re-verify -- the production recovery path.
        regen_used = True
        surface2, surface_full2, gen_s2 = faculty.render_svo_regen(a, v, p)
        reparse2, verified2, reason2 = _verify(surface2)
        rec["constrain_surface"] = surface                     # keep the first (rejected) render for the trail
        rec["constrain_reparse_svo"] = reparse_svo
        surface, surface_full, reparse_svo, verified, reason = surface2, surface_full2, reparse2, verified2, reason2
        gen_s += gen_s2

    rec.update({"surface": surface, "surface_full": surface_full, "reparse_svo": reparse_svo,
                "verified": bool(verified), "emitted": bool(verified), "abstained": False,
                "regen_used": regen_used, "reject_reason": reason, "gen_seconds": round(gen_s, 2)})
    return rec


def run(cur, vocab, seed, faculty, max_new_tokens):
    """Teach the curriculum, then run the small grounded/untaught/drift set end-to-end with the real faculty."""
    agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf")
    taught = _teach(agent, cur)

    # content-token sets for the VERIFY re-parse (agents = subjects, patients = objects, actions = verbs)
    agents_set = {f[0] for f in cur.get("facts", [])}
    patients_set = {f[2] for f in cur.get("facts", [])}
    actions_set = {f[1] for f in cur.get("facts", [])}
    inflect = _build_inflection_map(sorted(actions_set))
    vocab_sets = (agents_set, actions_set, patients_set, inflect)

    # --- (a) GROUNDED: ~4 grounded patient/agent queries -> a fluent sentence whose re-parse matches the fact ---
    grounded_queries = [q for q in cur.get("queries_recall", []) if q["type"] in ("patient", "agent")][:4]
    grounded = []
    for q in grounded_queries:
        rec = grounded_reply(agent, faculty, q, vocab_sets, faculty_mode="constrain")
        rec["ok"] = bool(rec["emitted"] and rec["verified"])
        grounded.append(rec)

    # --- (b) UNTAUGHT: ~2 untaught cues -> the GATE abstains -> NO sentence (the MOAT) ---
    untaught_queries = [q for q in cur.get("queries_moat", []) if q["type"] in ("patient", "agent")][:2]
    untaught = []
    for q in untaught_queries:
        rec = grounded_reply(agent, faculty, q, vocab_sets, faculty_mode="constrain")
        # the moat HOLDS iff the loop emitted NO sentence (the gate gave the faculty nothing)
        rec["held"] = (rec["abstained"] is True) and (rec["emitted"] is False)
        rec["note"] = q.get("note", "")
        untaught.append(rec)

    # --- (c) DRIFT/CONFAB: ~1 adversarial steered-to-wrong-fact -> VERIFY re-parse REJECTS it ---
    # take a grounded fact, gate the TRUE fact, but steer the faculty to a DIFFERENT (wrong) patient.
    drift = []
    all_patients = sorted(patients_set)
    base = grounded_queries[0]                                  # (dog, eat) -> true patient 'meat'
    true_p = agent.what_does(base["cue"][0], base["cue"][1])
    wrong_p = next((x for x in all_patients if x != true_p), (true_p or "thing") + "_X")
    adv_q = {"type": "patient", "cue": base["cue"], "wrong_patient": wrong_p}
    rec = grounded_reply(agent, faculty, adv_q, vocab_sets, faculty_mode="adversarial")
    # caught iff the loop refused to emit the drifted assertion (gate had the true fact; verify rejected)
    rec["true_patient"] = true_p
    rec["confab_patient"] = wrong_p
    rec["caught"] = (rec["gate_svo"] is not None) and (rec["emitted"] is False)
    drift.append(rec)

    # --- (c') REGENERATE-ON-REJECT demonstration: after a drift reject, re-prompt tighter -> verified-or-abstain.
    # (Shows the loop's recovery path: a rejected render does not silently leak; the brain re-prompts and
    # re-verifies; if STILL unverified it hard-abstains. Here we regenerate the SAME true fact tighter.)
    regen = None
    if drift and drift[0]["caught"]:
        a, v, p = agent.what_does(base["cue"][0], base["cue"][1]) and \
            [base["cue"][0], base["cue"][1], agent.what_does(base["cue"][0], base["cue"][1])]
        surface, surface_full, gen_s = faculty.render_svo_regen(a, v, p)
        content_svo = _extract_svo_from_prose(surface, agents_set, actions_set, patients_set, inflect)
        reparse = agent.parse(content_svo, voice="active") if content_svo else None
        reparse_svo = [reparse.get("agent"), reparse.get("action"), reparse.get("patient")] if reparse else None
        verified = (reparse_svo == [a, v, p])
        regen = {"gate_svo": [a, v, p], "surface": surface, "surface_full": surface_full,
                 "reparse_svo": reparse_svo, "verified": bool(verified), "emitted": bool(verified),
                 "gen_seconds": gen_s}

    n_grounded_ok = sum(r["ok"] for r in grounded)
    n_untaught_held = sum(r["held"] for r in untaught)
    n_drift_caught = sum(r["caught"] for r in drift)
    return {
        "seed": seed,
        "taught": taught,
        "grounded_correct": n_grounded_ok,
        "grounded_total": len(grounded),
        "untaught_held": n_untaught_held,
        "untaught_total": len(untaught),
        "drift_caught": n_drift_caught,
        "drift_total": len(drift),
        "grounded_detail": grounded,
        "untaught_detail": untaught,
        "drift_detail": drift,
        "regen_after_reject": regen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=16, help="rate-code pool budget for the spiking faculty (16=GO,1.08x ANN; 8=1.21x)")
    ap.add_argument("--max-new-tokens", type=int, default=24, help="faculty surface-form length cap (keep small)")
    ap.add_argument("--seed", type=int, default=42, help="brain seed (the faculty is deterministic-greedy)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t_start = time.time()
    with open(os.path.abspath(CURRICULUM), "r", encoding="utf-8") as fh:
        cur = json.load(fh)
    vocab = _collect_vocab(cur)
    # the adversarial path may fall back to '<token>_X'; add encodable fall-backs to the vocab (parity with P3)
    vocab = sorted(set(vocab) | {p + "_X" for p in {f[2] for f in cur.get("facts", [])}})

    print(f"[integ] brain backend={os.environ.get('SIM_BACKEND')} vocab={len(vocab)} words; "
          f"loading the REAL spiking Qwen faculty at T={args.T} ...", flush=True)

    err = None
    try:
        import torch
        if not torch.cuda.is_available():
            print("[integ] WARNING: CUDA not available -- the spiking faculty is a GPU runner; will be slow.", flush=True)
        faculty = SpikingQwenFaculty(T=args.T, max_new_tokens=args.max_new_tokens, seed=args.seed,
                                     device=("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[integ] faculty loaded in {faculty.load_seconds}s; pools={faculty.pools}; "
              f"ranges={faculty.measured_ranges}", flush=True)
        result = run(cur, vocab, args.seed, faculty, args.max_new_tokens)
    except Exception as e:
        err = repr(e)
        traceback.print_exc()
        result = {"seed": args.seed, "error": err, "traceback": traceback.format_exc()}

    # --- VERDICT ---
    if err is None:
        g_ok = result["grounded_correct"] == result["grounded_total"] and result["grounded_total"] > 0
        u_ok = result["untaught_held"] == result["untaught_total"] and result["untaught_total"] > 0
        d_ok = result["drift_caught"] == result["drift_total"] and result["drift_total"] > 0
        go = g_ok and u_ok and d_ok
        if go:
            verdict = (
                f"GO -- the REAL spiking Qwen faculty (T={args.T}) renders the brain's grounded facts FLUENTLY "
                f"(grounded {result['grounded_correct']}/{result['grounded_total']}, each re-parses to the taught "
                f"fact) AND untaught cues ABSTAIN (moat held {result['untaught_held']}/{result['untaught_total']}) "
                f"AND the adversarial DRIFT is caught-by-VERIFY ({result['drift_caught']}/{result['drift_total']}). "
                "The END-TO-END grounded-language capability works: a spiking fluent faculty + brain-grounded "
                "content + the no-confab moat, COMBINED -- the moat holds EVEN WITH a real generative LLM in the loop."
            )
        else:
            leaks = []
            if not g_ok:
                misses = [r for r in result["grounded_detail"] if not r["ok"]]
                leaks.append(f"GROUNDED leak {result['grounded_correct']}/{result['grounded_total']} -- "
                             + "; ".join(f"({'/'.join(map(str, r['cue']))}) reason={r.get('reject_reason') or 'not emitted'} "
                                         f"surface={r.get('surface')!r}" for r in misses))
            if not u_ok:
                breaches = [r for r in result["untaught_detail"] if not r["held"]]
                leaks.append(f"MOAT leak {result['untaught_held']}/{result['untaught_total']} -- an untaught cue "
                             "produced a sentence: " + "; ".join(f"({'/'.join(map(str, r['cue']))}) surface={r.get('surface')!r}" for r in breaches))
            if not d_ok:
                leaks.append(f"VERIFY leak {result['drift_caught']}/{result['drift_total']} -- an adversarial drift "
                             "was NOT caught (a false assertion reached the user): "
                             + "; ".join(f"surface={r.get('surface')!r} reparse={r.get('reparse_svo')}" for r in result["drift_detail"] if not r["caught"]))
            verdict = "HONEST/PARTIAL -- " + " || ".join(leaks)
    else:
        go = False
        verdict = f"ERROR -- {err}"

    summary = {
        "probe": "grounded_lang_integration_end_to_end_real_spiking_qwen_faculty",
        "resolves": "the grounded-language arc CAPSTONE -- the END-TO-END demo: the REAL spiking Qwen faculty "
                    "renders the brain's retrieved facts, GATED + VERIFIED, preserving the no-confab moat EVEN "
                    "WITH a real generative LLM in the loop.",
        "architecture": "GATE (brain composer exact-match recall / abstain; numpy-CPU) -> CONSTRAIN (the REAL "
                        "converted-Qwen2.5-0.5B SPIKING forward, T-pool graded-read RMSNorm/SiLU/Softmax, renders "
                        "the gated SVO into fluent prose with the verb kept; PyTorch on GPU) -> VERIFY (the brain "
                        "re-parses the faculty's GENERATED PROSE back into an SVO -- content extraction handling "
                        "determiners+inflection + the BridgeParser role assignment -- and rejects on mismatch with "
                        "the gated fact). The difference from P3: the faculty is a REAL hallucination-capable "
                        "generative model, so VERIFY reads its actual prose, not a handed-back SVO.",
        "faculty": f"REAL spiking Qwen2.5-0.5B-Instruct forward (P1b stepB1: T={args.T} GO, ppl "
                   f"{'1.08' if args.T == 16 else ('1.21' if args.T == 8 else '?')}x ANN), reused-by-import; "
                   "constrain prompt = variant-D ('Turn the triple (a,v,p) ... Keep the same verb ... only the "
                   "sentence'); greedy/deterministic; max_new_tokens=" + str(args.max_new_tokens),
        "curriculum": os.path.relpath(os.path.abspath(CURRICULUM), str(_REPO)),
        "brain_backend": os.environ.get("SIM_BACKEND"),
        "T": args.T,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "GO": go,
        "verdict": verdict,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "result": result,
    }

    out_path = os.path.abspath(args.out)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False,
                  default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else o)

    print("\n" + "=" * 100, flush=True)
    print(f"[integ] VERDICT: {verdict}", flush=True)
    print("=" * 100, flush=True)
    print(f"[integ] wrote {out_path}", flush=True)
    return summary


if __name__ == "__main__":
    main()
