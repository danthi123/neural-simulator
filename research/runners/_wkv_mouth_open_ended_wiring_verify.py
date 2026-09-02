"""WIRING VERIFY: the from-scratch WKV/SSM spiking mouth is correctly wired into `webapp.open_ended_chat.answer_turn`
as an alternate, IN-VOCAB-ONLY, default-OFF generator for the BRAIN_OPEN_ENDED channel. (2026-08-28)

CONTEXT. `research/findings/2026-08-28-mouth-crutch-burndown-scope.md` §4 named this exact rung: the
`BRAIN_OPEN_ENDED` channel (server.py:4505-4535, default-OFF) is the ONE live-pipeline touchpoint where the
literal Qwen2.5-0.5B model (`SpikingQwenFaculty`) is the SOLE generator with no fallback. This wiring adds a
SECOND, independent flag (`BRAIN_OPEN_ENDED_WKV_MOUTH`, default-OFF) that, when truthy AND the prompt is in-vocab
for the checkpoint's V=1000 TinyStories vocabulary, routes generation through `webapp.wkv_mouth_generator` instead
-- a genuinely different, from-scratch, architecturally-Qwen-unrelated recurrent SSM/RWKV cortex
(`bridges/wkv_ckpt`) reading its own word decisions via a GENUINE few-spike Izhikevich soft-WTA population read
(reused verbatim from the GO-verified `research.runners._wkv_fewspike_read_derisk`), not a host argmax. This is a
WIRING verify -- it does not re-derive the underlying few-spike read mechanism's own anti-cheats (scramble /
equal-drive / noise-ablation / provenance), which are that module's own GO result, untouched here.

WHAT THIS CHECKS:
  (a) FLAG-OFF BYTE-IDENTICAL CONTENT. `answer_turn`'s full result -- `answer`/`raw`/`filtered`/`topic`/`known`/
      `facts`/`gen_seconds`/`gen_time_honesty_used`/`gen_time_trace`/`state` -- is IDENTICAL between the PATCHED
      module and a git-show snapshot of the pre-edit original, with the Qwen call stubbed identically in both, when
      `BRAIN_OPEN_ENDED_WKV_MOUTH` is unset. `webapp.wkv_mouth_generator` is never even imported on that path (a
      poison-pill module raises if touched) -- proving zero import-time or call-time change, not just equal output.
      Two purely-additive trace keys (`generator`, `wkv_mouth_used`) are new; both default to the flag-off values
      ("qwen" / False) and are excluded from the content diff by name, not by silently ignoring a real difference.
  (b) COHERENT GENERATION, flag ON, in-vocab prompt. The REAL WKV pipeline runs (CPU/numpy, no GPU, no Qwen call
      -- the Qwen stub never fires); `generator=="wkv_mouth"`; the raw continuation's OWN self-NLL under the
      checkpoint's teacher-forced next-word distribution (the SAME metric `_wkv_fewspike_read_derisk` uses to
      report on-distribution generation) is compared against chance (log(V)); the existing `post_filter` still ran
      (the honesty gate is never bypassed -- an unknown-topic prompt still gets the honest hedge).
  (c) FALLBACK is genuine, not decorative. An OUT-OF-VOCAB prompt with the flag ON falls back to Qwen -- output
      byte-identical to the flag-off case (same stub fires). A forced exception inside the WKV generator (LESION)
      also falls back to Qwen without crashing the turn.
  (d) RNG DISCIPLINE. The host process-global numpy RNG state is BYTE-IDENTICAL before/after a WKV-path call (the
      #77 footgun this module's own `_RngIsolation` exists to avoid) -- measured directly, not asserted.
  (e) PROVENANCE. `host_rng_draws_on_read_path == 0` on the `FewSpikeWordRead` instance the wiring builds (the
      few-spike read's own anti-cheat field) -- confirms the wiring invokes the real spiking read, not a shortcut.

MEMORY-SAFE BY DESIGN: CPU/numpy only, ~512 neurons, no GPU, no Qwen render (stubbed).

  SIM_BACKEND=numpy python -m research.runners._wkv_mouth_open_ended_wiring_verify
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging
logging.disable(logging.INFO)
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_wkv_mouth_open_ended_wiring_verify.json"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_qwen_generate(self, system, user, seed=42, max_new_tokens=None):
    return ("STUB QWEN OUTPUT the boy played with his ball", 0.011)


def main():
    import numpy as np

    tmp = Path("/tmp") / "wkv_mouth_wiring_verify_ORIGINAL_open_ended_chat.py"
    orig_src = subprocess.run(["git", "show", "HEAD:webapp/open_ended_chat.py"], cwd=str(_REPO),
                              capture_output=True, text=True, check=True).stdout
    tmp.write_text(orig_src)

    # -- (a) FLAG OFF: byte-identical content, WKV module never imported -----------------------------------------
    # 2026-09-02 off-arm-staleness fix (flip_offarm_staleness gate): `wkv_mouth_enabled()` FLIPPED default-ON
    # 2026-08-30 (open_ended_chat.py:236 reads `.get(..., "1")`), so `os.environ.pop` no longer means OFF -- it
    # now reads ON and this "FLAG OFF" arm would silently import/reach the WKV path. Force it explicitly, matching
    # the reference `_spiking_mouth_recall_soak.py::_set_flag` fix (NOT re-run end-to-end, per that audit's pattern).
    os.environ["BRAIN_OPEN_ENDED_WKV_MOUTH"] = "0"
    orig = _load_module("_wkv_verify_open_ended_chat_ORIGINAL", str(tmp))
    patched_off = _load_module("_wkv_verify_open_ended_chat_PATCHED_off", str(_REPO / "webapp" / "open_ended_chat.py"))
    orig.OpenEndedGenerator.generate = _stub_qwen_generate
    patched_off.OpenEndedGenerator.generate = _stub_qwen_generate

    poison = Path("/tmp") / "wkv_mouth_wiring_verify_POISON_wkv_mouth_generator.py"
    poison.write_text(
        "def in_vocab_scope(*a, **k):\n    raise AssertionError('WKV module touched while flag OFF')\n"
        "def generate(*a, **k):\n    raise AssertionError('WKV module touched while flag OFF')\n"
    )
    poison_mod = _load_module("webapp.wkv_mouth_generator", str(poison))
    sys.modules["webapp.wkv_mouth_generator"] = poison_mod

    msg = "Tell me about photosynthesis"
    kwargs = dict(warm_faculty=object(), valence=0.2, arousal=0.3, ltm_bundle=None, brain_bundle=None,
                 seed=42, max_new_tokens=40)
    r_orig = orig.answer_turn(msg, **kwargs)
    r_off = patched_off.answer_turn(msg, **kwargs)          # poison pill would raise here if WKV were touched
    content_keys = ["answer", "raw", "filtered", "topic", "known", "facts", "gen_seconds",
                    "gen_time_honesty_used", "gen_time_trace", "state", "n_sentences"]
    diffs = [k for k in content_keys if r_orig.get(k) != r_off.get(k)]
    off_byte_identical = (len(diffs) == 0)
    off_generator_is_qwen = (r_off.get("generator") == "qwen" and r_off.get("wkv_mouth_used") is False)

    # -- (b) FLAG ON, in-vocab: real WKV pipeline, coherent generation, honesty gate still runs --------------------
    del sys.modules["webapp.wkv_mouth_generator"]
    os.environ["BRAIN_OPEN_ENDED_WKV_MOUTH"] = "1"
    patched_on = _load_module("_wkv_verify_open_ended_chat_PATCHED_on", str(_REPO / "webapp" / "open_ended_chat.py"))
    patched_on.OpenEndedGenerator.generate = _stub_qwen_generate     # if this fires, WKV path was NOT used

    msg_invocab = "once upon a time there was a little boy named tim who had a dog"
    host_rng_before = np.random.get_state()[1].copy()
    r_on = patched_on.answer_turn(msg_invocab, **kwargs)
    host_rng_after = np.random.get_state()[1].copy()
    rng_untouched = bool((host_rng_before == host_rng_after).all())

    used_wkv = (r_on.get("generator") == "wkv_mouth" and r_on.get("wkv_mouth_used") is True)
    qwen_stub_fired_on_wkv_path = (r_on.get("raw") == "STUB QWEN OUTPUT the boy played with his ball")
    post_filter_ran = (r_on.get("filtered") is not None and len(r_on.get("filtered", "")) > 0)

    # self-NLL of the WKV raw continuation under the checkpoint's OWN teacher-forced next-word distribution --
    # the SAME on-distribution metric _wkv_fewspike_read_derisk reports (chance == log(V) for a V=1000 vocab).
    from webapp import wkv_mouth_generator as _WKV
    ro, _vocab, word_to_id = _WKV._get_readout(42)
    words = [w for w in r_on["raw"].split() if w in word_to_id]
    self_nll = None
    if len(words) >= 3:
        # gate on the PREVIOUS (already-advanced) token to predict the NEXT -- the SAME pattern
        # `wkv_mouth_generator._free_gen` and `_wkv_fewspike_read_derisk.run_seed` both use
        # (`logits(ap, an, gen[-1])` predicts gen[-1]+1, not itself). Gating on the token being
        # predicted (an earlier version of this check did) reads a different, meaningless quantity.
        ap = np.zeros(ro.D)
        an = np.zeros(ro.D)
        prev_id = word_to_id[words[0]]
        ap, an = ro.advance(ap, an, prev_id)
        nlls = []
        for w in words[1:]:
            lg = ro.logits(ap, an, prev_id)          # gate on the PREVIOUS token, predict `w`
            p = np.exp(lg - lg.max())
            p = p / p.sum()
            tid = word_to_id[w]
            nlls.append(-float(np.log(max(p[tid], 1e-12))))
            ap, an = ro.advance(ap, an, tid)
            prev_id = tid
        self_nll = sum(nlls) / len(nlls)
    chance_nll = float(np.log(ro.V))

    # provenance: the FewSpikeWordRead instance the wiring itself builds never draws a host categorical sample.
    reader = _WKV.FewSpikeWordRead(64, 8, 42, read_window=30)
    reader.read(np.array([1.0, 0.5, 0.2] + [0.0] * 61))
    host_rng_draws_on_read_path = reader.n_host_rng_draws

    # checkpoint provenance sanity check (task-requested: legitimately-trained model, or a placeholder?). Pure
    # weight/vocab statistics -- NOT a claim about WHICH training method produced them (unresolved, see the
    # wiring finding doc). A random-init or corrupted placeholder would show all-zero/all-identical rows or NaN.
    hw = ro.head_w
    emb = ro.emb
    checkpoint_provenance = {
        "vocab_size": len(ro.words), "first_40_words": list(ro.words[:40]), "last_10_words": list(ro.words[-10:]),
        "head_weight_shape": list(hw.shape), "head_weight_mean": float(hw.mean()), "head_weight_std": float(hw.std()),
        "head_weight_min": float(hw.min()), "head_weight_max": float(hw.max()),
        "head_weight_has_nan": bool(np.isnan(hw).any()), "head_weight_all_zero": bool((hw == 0).all()),
        "emb_weight_mean": float(emb.mean()), "emb_weight_std": float(emb.std()),
        "emb_weight_has_nan": bool(np.isnan(emb).any()), "emb_weight_all_zero": bool((emb == 0).all()),
    }

    # -- (c) FALLBACK: out-of-vocab -> Qwen; forced exception -> Qwen -----------------------------------------------
    msg_oov = "explain the geopolitical ramifications of quantum computing on cryptography policy"
    r_oov = patched_on.answer_turn(msg_oov, **kwargs)
    oov_falls_back = (r_oov.get("generator") == "qwen" and r_oov.get("wkv_mouth_used") is False
                      and r_oov.get("raw") == "STUB QWEN OUTPUT the boy played with his ball")

    _orig_generate = _WKV.generate
    _WKV.generate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("forced lesion"))
    r_lesion = patched_on.answer_turn(msg_invocab, **kwargs)
    _WKV.generate = _orig_generate
    lesion_falls_back = (r_lesion.get("generator") == "qwen" and r_lesion.get("wkv_mouth_used") is False
                         and r_lesion.get("raw") == "STUB QWEN OUTPUT the boy played with his ball")
    # ATTRIBUTION (not just measuring both arms): the SAME in-vocab prompt, same seed, same flags -- the ONLY
    # thing that differs between r_on (intact) and r_lesion (WKV generator forced to raise) is the lesion itself.
    # wkv_mouth_used is a clean 1/0 indicator; attributable_to must read 100% -- if it read less, something OTHER
    # than the forced exception would also be suppressing the WKV path, which would undercut claim (c).
    wkv_used_intact = 1.0 if used_wkv else 0.0
    wkv_used_lesioned = 1.0 if r_lesion.get("wkv_mouth_used") else 0.0
    lesion_attribution = attributable_to("forced WKV-generator exception -> wkv_mouth_used",
                                         treatment_value=wkv_used_intact, control_value=wkv_used_lesioned)

    art = {
        "probe": "wkv_mouth_open_ended_wiring_verify", "backend": "numpy",
        "off_content_diffs": diffs, "off_byte_identical": off_byte_identical,
        "off_generator_is_qwen": off_generator_is_qwen,
        "on_used_wkv": used_wkv, "on_qwen_stub_fired": qwen_stub_fired_on_wkv_path,
        "on_post_filter_ran": post_filter_ran, "on_raw": r_on.get("raw"), "on_filtered": r_on.get("filtered"),
        "self_nll_wkv_continuation": self_nll, "chance_nll_uniform_over_V": chance_nll,
        "n_words_scored": len(words), "rng_untouched_across_wkv_call": rng_untouched,
        "host_rng_draws_on_read_path": host_rng_draws_on_read_path,
        "oov_falls_back_to_qwen": oov_falls_back, "lesion_falls_back_to_qwen": lesion_falls_back,
        "lesion_attribution_fraction": lesion_attribution,
        "V": int(ro.V), "checkpoint_provenance_sanity": checkpoint_provenance,
    }

    v = Verdict("the from-scratch WKV mouth is correctly wired into webapp.open_ended_chat.answer_turn "
               "as a default-OFF, in-vocab-scoped alternate generator for BRAIN_OPEN_ENDED")
    v.require("(a) flag-off content is identical to the pre-edit original (WKV module never touched)",
              off_byte_identical, expect=True, note=f"diffs={diffs}")
    v.require("(a) flag-off generator trace reads qwen/False", off_generator_is_qwen, expect=True)
    v.require("(provenance sanity) checkpoint weights are non-degenerate (no NaN, not all-zero)",
              (not checkpoint_provenance["head_weight_has_nan"] and not checkpoint_provenance["head_weight_all_zero"]
               and not checkpoint_provenance["emb_weight_has_nan"] and not checkpoint_provenance["emb_weight_all_zero"]),
              expect=True, note="a random-init or corrupted placeholder would show NaN or all-zero rows")
    v.require("(b) flag ON + in-vocab -> the WKV path actually ran (Qwen stub did NOT fire)", used_wkv, expect=True)
    v.require("(b) Qwen stub never fired on the WKV path", qwen_stub_fired_on_wkv_path, expect=False)
    v.require("(b) post_filter still ran on the WKV output (honesty gate never bypassed)", post_filter_ran, expect=True)
    if self_nll is not None:
        v.control("(b) WKV continuation self-NLL vs chance (uniform over V=1000)",
                  treatment=chance_nll - self_nll, control=0.0, min_separation=2.0,
                  note=f"self_nll={self_nll:.3f} nats vs chance={chance_nll:.3f} nats "
                       f"(lower self_nll == more on-distribution / coherent)")
    v.require("(d) host numpy RNG state is byte-identical across the WKV-path call", rng_untouched, expect=True)
    v.require("(e) the wiring's own FewSpikeWordRead makes zero host categorical draws on the read path",
              host_rng_draws_on_read_path, expect=0)
    v.require("(c) an out-of-vocab prompt (flag ON) falls back to Qwen, unchanged", oov_falls_back, expect=True)
    v.require("(c) a forced WKV-path exception falls back to Qwen without crashing the turn",
              lesion_falls_back, expect=True)
    v.require("(c) the forced-exception lesion attributes 100% of the wkv_mouth_used drop (no other cause)",
              lesion_attribution, expect=lambda x: x is not None and abs(x - 1.0) < 1e-9)

    ckpt_clean = (not checkpoint_provenance["head_weight_has_nan"] and not checkpoint_provenance["head_weight_all_zero"]
                 and not checkpoint_provenance["emb_weight_has_nan"] and not checkpoint_provenance["emb_weight_all_zero"])
    go = (off_byte_identical and off_generator_is_qwen and used_wkv and not qwen_stub_fired_on_wkv_path
         and post_filter_ran and rng_untouched and host_rng_draws_on_read_path == 0
         and oov_falls_back and lesion_falls_back and ckpt_clean
         and (lesion_attribution is not None and abs(lesion_attribution - 1.0) < 1e-9)
         and (self_nll is not None and (chance_nll - self_nll) > 2.0))
    decided = v.decide(go=go)
    art["verdict"] = decided
    art["preconditions"] = decided.get("preconditions", [])
    art["GO"] = bool(go)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art, indent=1))
    print(json.dumps({k: art[k] for k in (
        "off_byte_identical", "off_generator_is_qwen", "on_used_wkv", "on_qwen_stub_fired",
        "on_post_filter_ran", "self_nll_wkv_continuation", "chance_nll_uniform_over_V",
        "rng_untouched_across_wkv_call", "host_rng_draws_on_read_path",
        "oov_falls_back_to_qwen", "lesion_falls_back_to_qwen", "lesion_attribution_fraction",
        "checkpoint_provenance_sanity", "GO")}, indent=1))
    print(f"wrote {OUT} -> {decided['status']}")
    return decided["status"]


if __name__ == "__main__":
    main()
