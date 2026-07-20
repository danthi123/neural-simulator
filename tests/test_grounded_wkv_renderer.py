"""CI guard for the north-star grounded-fluent-conversation WKV renderer (De-risk 0-5, 2026-07-20).

The spiking WKV cortex renders grounded fluent answers in the fluid console, replacing the ~21M ANN scaffold, behind
the gate-first no-confab moat. This pins the capability so it can't silently regress: (1) WKVFaculty renders a grounded
fact cleanly + is RA-faithful (follows the prompt fact); (2) FluidChat(renderer="wkv") answers grounded Q&A and
ABSTAINS on untaught with the faculty invoked 0x (gate-first moat). CPU (numpy); skips gracefully if the fine-tuned WKV
checkpoint is absent (it is committed at ~9.8MB, so this normally RUNS).
"""
from __future__ import annotations
import os
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")   # must precede the sim.backend import; CPU-portable
from research.runners._grounded_lang_p2_derisk import CURRICULUM  # noqa: E402

WKV_CKPT = "bridges/wkv_ckpt/wkv_ssmU_v4000_d256_grounded_ft.npz"
pytestmark = pytest.mark.skipif(
    not (os.path.exists(WKV_CKPT) and os.path.exists(os.path.abspath(CURRICULUM))),
    reason="grounded-fine-tuned WKV checkpoint or curriculum absent")


def test_wkv_faculty_renders_grounded_fact():
    from research.runners._wkv_faculty import WKVFaculty
    f = WKVFaculty(ckpt=WKV_CKPT, max_new=8)
    assert f.has_markers, "the fine-tuned checkpoint must carry the <ans>/<eos> format markers"
    ans = f.answer("the dog eats meat .", "what does the dog eat ?")
    assert "meat" in ans.split(), f"grounded fact not rendered: {ans!r}"
    assert "<eos>" not in ans and "<ans>" not in ans, f"markers leaked into the answer: {ans!r}"


def test_wkv_faculty_ra_faithful_follows_prompt():
    # the copy skill must follow the PROMPT fact (grounded on retrieval), not a memorized/bias one
    from research.runners._wkv_faculty import WKVFaculty
    f = WKVFaculty(ckpt=WKV_CKPT, max_new=8)
    ans = f.answer("the dog eats cake .", "what does the dog eat ?")   # cake != the taught meat
    assert "cake" in ans.split() and "meat" not in ans.split(), f"not RA-faithful: {ans!r}"


def test_wkv_unk_index_by_name_not_position():
    # regression: <unk> must be indexed BY NAME (a format fine-tune appends <ans>/<eos> after <unk>, so V-1 != <unk>)
    from research.runners._wkv_faculty import WKVFaculty
    f = WKVFaculty(ckpt=WKV_CKPT)
    assert f.words[f.unk] == "<unk>", f"unk index points at {f.words[f.unk]!r}, not <unk> (the eos-suppression bug)"


def test_fluidchat_wkv_grounded_and_gatefirst_moat():
    from research.runners._fluidconv_chat_repl import FluidChat
    chat = FluidChat(seed=42, renderer="wkv")
    assert type(chat.faculty).__name__ == "WKVFaculty"
    # grounded answer via the spiking WKV
    chat.faculty.n_invocations = 0
    r = chat.turn("what does the dog eat?")
    assert "meat" in r.lower(), f"grounded answer missing: {r!r}"
    assert chat.faculty.n_invocations >= 1, "faculty should be invoked on a grounded answer"
    # GATE-FIRST MOAT: an untaught cue abstains WITHOUT invoking the faculty (no chance to confab)
    chat.faculty.n_invocations = 0
    r = chat.turn("what does the lion eat?")
    assert "don't know" in r.lower(), f"moat should abstain on untaught: {r!r}"
    assert chat.faculty.n_invocations == 0, "MOAT BREACH: faculty invoked on an abstain (not gate-first)"
