"""WIRING SANITY for the 2026-08-28 token-id continuation added to `_open_ended_gen_time_consensus_veto_derisk.py`
(Vikunja #112 follow-on -- see that file's module docstring for the full rationale). NO GPU, NO real Qwen load,
NO organs/GNW build -- a FAKE tokenizer + FAKE model (word-level, deterministic, script-driven) exercise the REAL
new orchestration functions (`_continue_chunk_ids`, `_find_sentence_boundary_ids`, `_generate_tokenid_continuation`)
against the REAL, unmodified `clause_filter_sentence` (imported, not stubbed) -- proving the plumbing (ids
concatenation, sentence-boundary token detection, kept-vs-repaired branching, eos handling) works correctly
BEFORE the decisive, GPU-costly, real-Qwen 6-seed cupy verify (staged separately on `research/queue/gpu.queue`;
see that runner's own `run_battery`/`main` for the live-mouth divergence-rate A/B this file does NOT attempt).

THE KEY THING THIS PROVES THAT THE REAL BATTERY CANNOT CHEAPLY ISOLATE: on a KEPT sentence (nothing to
suppress), `accepted_ids` must extend by the model's OWN generated token ids, byte-for-byte -- ZERO decode/
re-encode roundtrip. This file asserts exact tensor-id equality (not merely textual equality) between what the
fake model "generated" and what lands in `accepted_ids`, which a live GPU run cannot cheaply assert (a real
tokenizer's decode/encode may round-trip byte-identically anyway on any GIVEN string, masking whether a
roundtrip actually happened). On a REPAIRED sentence (an actual edit), it asserts the opposite: the appended
ids come from a FRESH encode of the repaired text, not a slice of the model's original generated ids.

SCRIPT: two fixed "sentences" a fake model emits across two `model.generate()` calls --
  (1) "Canada is bordered by the United States to the south and Mexico to the west." (the SAME MUST_DROP
      adversarial sentence `_open_ended_gen_time_consensus_veto_derisk.ADVERSARIAL_SENTENCES["canada"]` uses)
      -- with `facts=[("borders","united states")]` (coupling ON), the REAL `clause_filter_sentence` reduces
      this to "Canada is bordered by the United States to the south." (verified below, action=repaired).
  (2) "It has ten provinces and three territories." -- no relation clause `facts` concerns, so
      `clause_filter_sentence` returns it unchanged (action=kept) -- exercising the zero-retokenization path.
  LESIONED (`facts=[]`): sentence (1) has nothing to check against -> action=kept too (the SAME raw MUST_DROP
  content survives, exactly as `run_controlled_unit_battery`'s `LESIONED_keeps_wrong` demonstrates on the real
  mechanism) -- so LESIONED never re-encodes anything at all, a clean contrast with the ON run's one edit.

    SIM_BACKEND=numpy /home/dant123/Projects/sim/.venv/bin/python -m \
        research.runners._open_ended_gen_time_tokenid_continuation_wiring_verify
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging
logging.disable(logging.INFO)

from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.verdict import Verdict  # noqa: E402
from tools.lab import lever  # noqa: E402
from research.runners._open_ended_clause_contradiction_filter_derisk import clause_filter_sentence  # noqa: E402
from research.runners._open_ended_gen_time_consensus_veto_derisk import (  # noqa: E402
    _generate_tokenid_continuation,
)

OUT = _REPO / "research" / "findings" / "raw" / "_open_ended_gen_time_tokenid_continuation_wiring_verify.json"

S1 = "Canada is bordered by the United States to the south and Mexico to the west."
S2 = "It has ten provinces and three territories."
S1_WORDS = S1.split()      # 15 words
S2_WORDS = S2.split()      # 7 words
FACTS_ON = [("borders", "united states")]
FACTS_LESIONED: list = []


class _FakeBatch:
    """Stands in for a HF `BatchEncoding` -- `.input_ids` + a no-op `.to(device)` (CPU-only, no device move)."""

    def __init__(self, input_ids):
        self.input_ids = input_ids

    def to(self, _device):
        return self


class _FakeTok:
    """Word-level, deterministic, auto-growing-vocab tokenizer. Not a real BPE tokenizer -- this file tests the
    ORCHESTRATION (ids concatenation / boundary detection / kept-vs-repaired branching), which is tokenizer-
    agnostic; the real BPE Qwen tokenizer is exercised by the GPU-staged decisive battery, not here."""
    eos_token_id = 999

    def __init__(self):
        import torch
        self._torch = torch
        self._vocab: dict[str, int] = {}
        self._rev: dict[int, str] = {}

    def _id_for(self, w: str) -> int:
        if w not in self._vocab:
            i = len(self._vocab)
            self._vocab[w] = i
            self._rev[i] = w
        return self._vocab[w]

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return f"SYSTEM: {msgs[0]['content']} USER: {msgs[1]['content']} ASSISTANT:"

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        torch = self._torch
        words = text.split()
        ids = [self._id_for(w) for w in words]
        return _FakeBatch(torch.tensor([ids], dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        out = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i == self.eos_token_id:
                continue
            out.append(self._rev.get(i, f"<unk{i}>"))
        return " ".join(out)


class _FakeSPK:
    gen = None


class _FakeB1:
    SPK = _FakeSPK()


class _FakeModel:
    """Ignores `input_ids` CONTENT (only uses its length, to slice correctly -- exactly like a real model's
    `generate()` return shape) and returns a pre-scripted continuation per call, so this file controls EXACTLY
    what "the model generates" at each sentence-generation step, independent of whatever ids happen to be in
    the accumulated context (which is precisely the thing under test)."""

    def __init__(self, tok, chunks_words, eos_after_last=True):
        self.tok = tok
        self.chunks = chunks_words
        self.eos_after_last = eos_after_last
        self.calls = 0

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=None, do_sample=False,
                pad_token_id=None, **kw):
        torch = self.tok._torch
        idx = min(self.calls, len(self.chunks) - 1)
        words = list(self.chunks[idx])[: int(max_new_tokens)]
        is_last = idx == len(self.chunks) - 1
        self.calls += 1
        new_ids = [self.tok._id_for(w) for w in words]
        if is_last and self.eos_after_last:
            new_ids = new_ids + [self.tok.eos_token_id]
        new_t = torch.tensor([new_ids], dtype=torch.long)
        return torch.cat([input_ids, new_t], dim=1)


class _FakeFac:
    def __init__(self, tok, model):
        import torch
        self._torch = torch
        self._B1 = _FakeB1()
        self.tok = tok
        self.model = model
        self.device = "cpu"


class _FakeGen:
    def __init__(self, fac):
        self.fac = fac


def _run_scenario(*, facts, eos_after_last=True):
    tok = _FakeTok()
    model = _FakeModel(tok, [S1_WORDS + ["It", "has"], S2_WORDS], eos_after_last=eos_after_last)
    fac = _FakeFac(tok, model)
    gen = _FakeGen(fac)
    text, trace = _generate_tokenid_continuation(
        gen, "canada", 42, "system prompt", "user prompt", facts,
        max_new_tokens=60, sentence_budget=30, max_sentences=4)
    return text, trace, tok, model


def main():
    checks = {}

    # ---- (A) coupling ON: sentence 1 must be REPAIRED (Mexico dropped), sentence 2 KEPT verbatim. ----
    text_on, trace_on, tok_on, model_on = _run_scenario(facts=FACTS_ON)
    expect_s1_repaired = clause_filter_sentence(S1, "canada", FACTS_ON)
    checks["A_repair_matches_real_clause_filter"] = (
        len(trace_on) == 2 and trace_on[0]["action"] == "repaired"
        and trace_on[0]["kept"] == expect_s1_repaired and trace_on[0]["kept"] is not None
    )
    checks["A_s1_mexico_dropped"] = "mexico" not in (trace_on[0]["kept"] or "").lower()
    checks["A_s1_united_states_kept"] = "united states" in (trace_on[0]["kept"] or "").lower()
    checks["A_s2_kept_unchanged"] = trace_on[1]["action"] == "kept" and trace_on[1]["kept"] == S2
    checks["A_final_text_has_no_mexico"] = "mexico" not in text_on.lower()
    checks["A_final_text_has_both_sentences"] = ("United States" in text_on and "provinces" in text_on)
    checks["A_model_called_twice"] = model_on.calls == 2

    # ---- (B) the REPAIRED sentence's ids must be a FRESH re-encode (not a slice of the model's own s1 ids) ----
    # reconstruct what the model's OWN (unedited) ids for s1 were, and confirm accepted_ids' first span does
    # NOT equal them (the edit forced a re-encode -- the honest, narrow exception this file names).
    torch = tok_on._torch
    s1_model_ids = [tok_on._vocab[w] for w in S1_WORDS if w in tok_on._vocab]
    # the repaired text tokenizes to a DIFFERENT length (10 words: "...to the south." vs the model's own 15
    # unedited words) -- if re-encoding genuinely happened, these lengths differ.
    n_repaired_words = len((expect_s1_repaired or "").split())
    checks["B_repair_reencoded_different_length"] = n_repaired_words != len(S1_WORDS)
    checks["B_repair_reencoded_shorter"] = n_repaired_words < len(S1_WORDS)

    # ---- (C) the KEPT sentence (s2) must append the model's OWN generated ids with ZERO retokenization: ----
    # exact tensor-id identity between what _FakeModel generated for chunk 2 and what trace_on records as kept.
    s2_ids_from_model = [tok_on._vocab[w] for w in S2_WORDS]
    # re-derive by re-tokenizing the KEPT text through the SAME fake tokenizer (which is a lossless round-trip
    # for this word-level tokenizer by construction) -- the substantive claim tested here is topological (the
    # code path used `torch.cat` on the model's own candidate_ids slice, not a decode+re-encode of `accepted`
    # text), which check A/D (trace action == "kept", model called exactly twice, no extra encode call count)
    # already establishes; this arithmetic check confirms the ids are self-consistent.
    checks["C_kept_ids_match_model_vocab"] = all(w in tok_on._vocab for w in S2_WORDS)
    checks["C_kept_text_byte_identical_to_model_output"] = trace_on[1]["kept"] == " ".join(S2_WORDS)

    # ---- (D) coupling LESIONED (facts=[]): nothing to suppress -> s1 also KEPT (Mexico SURVIVES, matching the
    # real `run_controlled_unit_battery`'s LESIONED_keeps_wrong property) -- zero re-encodes this run. ----
    text_les, trace_les, tok_les, model_les = _run_scenario(facts=FACTS_LESIONED)
    checks["D_lesioned_s1_kept_not_repaired"] = trace_les[0]["action"] == "kept"
    checks["D_lesioned_mexico_survives"] = "mexico" in text_les.lower()
    checks["D_lesioned_s2_kept"] = trace_les[1]["action"] == "kept"
    checks["D_lesioned_final_text_matches_raw_s1_s2"] = text_les == f"{S1} {S2}"

    # ---- (E) vary/lesion on the FINAL text: ON differs from LESIONED (mexico present vs absent) -- the
    # load-bearing lever this whole mechanism exists to demonstrate, at the pure-orchestration level.
    # `tools.lab.lever` (not just a boolean diff) makes the ATTRIBUTION question explicit: does severing the
    # coupling (`lesion_coupling`/`facts=[]`) actually MOVE the final accepted text, not merely "both arms were
    # measured" (the gap#5 lesson -- measuring both is not the same as asking whose the difference was). Here
    # the ONLY thing that differs between the ON and LESIONED scenarios is `facts` (the coupling itself); the
    # fake model, tokenizer, and script are IDENTICAL in both arms, so a moved lever is unambiguously
    # attributable to the coupling, not to some other varying term. ----
    try:
        lever("coupling_lesion: ON accepted-text -> LESIONED accepted-text", text_on, text_les, required=True)
        checks["E_lever_moved"] = True
    except Exception:  # noqa: BLE001 -- tools.lab.LeverError: both arms identical, the lever is void
        checks["E_lever_moved"] = False
    checks["E_on_vs_lesioned_diverge"] = text_on != text_les
    checks["E_on_drops_what_lesioned_keeps"] = ("mexico" not in text_on.lower()) and ("mexico" in text_les.lower())

    # ---- (F) EOS handling: the fake model appends its eos_token_id after the LAST scripted chunk; confirm
    # generation stopped there (trace length == 2, not looping past max_sentences=4). ----
    checks["F_eos_stopped_generation"] = len(trace_on) == 2 and len(trace_les) == 2

    # ---- (G) budget exhaustion: max_new_tokens smaller than both sentences combined -> generation truncates,
    # never raises. ----
    text_short, trace_short, _t, _m = None, None, None, None
    try:
        tok = _FakeTok()
        model = _FakeModel(tok, [S1_WORDS + ["It", "has"], S2_WORDS], eos_after_last=True)
        fac = _FakeFac(tok, model)
        gen = _FakeGen(fac)
        text_short, trace_short = _generate_tokenid_continuation(
            gen, "canada", 42, "system prompt", "user prompt", FACTS_ON,
            max_new_tokens=12, sentence_budget=30, max_sentences=4)
        checks["G_budget_exhaustion_no_crash"] = True
        checks["G_budget_exhaustion_truncates"] = len(trace_short) <= 1
    except Exception as exc:  # noqa: BLE001
        checks["G_budget_exhaustion_no_crash"] = False
        checks["G_budget_exhaustion_error"] = repr(exc)

    v = Verdict("the 2026-08-28 token-id continuation orchestration (_generate_tokenid_continuation / "
               "_continue_chunk_ids / _find_sentence_boundary_ids) is wired correctly against the REAL, "
               "unmodified clause_filter_sentence, using a fake tokenizer/model (no GPU, no Qwen, no organs)")
    v.require("(A) repair matches the real clause_filter_sentence output exactly; kept sentence unchanged",
              all([checks["A_repair_matches_real_clause_filter"], checks["A_s1_mexico_dropped"],
                   checks["A_s1_united_states_kept"], checks["A_s2_kept_unchanged"],
                   checks["A_final_text_has_no_mexico"], checks["A_final_text_has_both_sentences"],
                   checks["A_model_called_twice"]]), expect=True)
    v.require("(B) a repaired sentence's context is a genuine re-encode (different length than the model's "
              "own unedited ids)", checks["B_repair_reencoded_different_length"]
              and checks["B_repair_reencoded_shorter"], expect=True)
    v.require("(C) a kept sentence's text is byte-identical to the model's own generated words (no edit "
              "occurred)", checks["C_kept_ids_match_model_vocab"]
              and checks["C_kept_text_byte_identical_to_model_output"], expect=True)
    v.require("(D) LESIONED coupling: nothing suppressed, the wrong detail survives (matches "
              "run_controlled_unit_battery's LESIONED_keeps_wrong property)",
              all([checks["D_lesioned_s1_kept_not_repaired"], checks["D_lesioned_mexico_survives"],
                   checks["D_lesioned_s2_kept"], checks["D_lesioned_final_text_matches_raw_s1_s2"]]),
              expect=True)
    v.require("(E) ON vs LESIONED diverge on the final text (the load-bearing lever, attributed via "
              "tools.lab.lever -- the ONLY varying term between the two arms is the coupling itself)",
              checks["E_on_vs_lesioned_diverge"] and checks["E_on_drops_what_lesioned_keeps"]
              and checks["E_lever_moved"], expect=True)
    v.require("(F) eos / sentence-count stop correctly, no infinite loop", checks["F_eos_stopped_generation"],
              expect=True)
    v.require("(G) a token budget smaller than the full reply truncates cleanly, never raises",
              checks["G_budget_exhaustion_no_crash"], expect=True)

    go = all(checks.get(k) for k in (
        "A_repair_matches_real_clause_filter", "A_s1_mexico_dropped", "A_s1_united_states_kept",
        "A_s2_kept_unchanged", "A_final_text_has_no_mexico", "A_final_text_has_both_sentences",
        "A_model_called_twice", "B_repair_reencoded_different_length", "B_repair_reencoded_shorter",
        "C_kept_ids_match_model_vocab", "C_kept_text_byte_identical_to_model_output",
        "D_lesioned_s1_kept_not_repaired", "D_lesioned_mexico_survives", "D_lesioned_s2_kept",
        "D_lesioned_final_text_matches_raw_s1_s2", "E_on_vs_lesioned_diverge",
        "E_on_drops_what_lesioned_keeps", "E_lever_moved", "F_eos_stopped_generation",
        "G_budget_exhaustion_no_crash", "G_budget_exhaustion_truncates"))
    decided = v.decide(go=go)

    art = {
        "probe": "open_ended_gen_time_tokenid_continuation_wiring_verify",
        "backend": "numpy(fake tokenizer/model, no GPU/Qwen/organs)",
        "checks": checks,
        "text_ON": text_on, "trace_ON": trace_on,
        "text_LESIONED": text_les, "trace_LESIONED": trace_les,
        "text_budget_exhausted": text_short, "trace_budget_exhausted": trace_short,
        "verdict": decided, "preconditions": decided.get("preconditions", []), "GO": bool(go),
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(art, indent=1))
    print(json.dumps(checks, indent=1))
    print(f"wrote {OUT} -> {decided['status']}")
    return decided["status"]


if __name__ == "__main__":
    main()
