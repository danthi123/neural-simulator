# Burndown C-1 — first-chat console default → spiking onebrain (GPU) — GO (2026-06-27)

The flagship chat's `--composer` default is flipped from `rf` to **`auto`**, which resolves to the spiking `OneBrainComposer` on a GPU (cupy) backend and to the numpy `RFPhasorComposer` ORACLE on a CPU/numpy backend (the `consolidated_320` pattern). ⇒ the chat the owner talks to runs **fully-spiking-on-one-brain by default on GPU**, while the numpy test-oracle + CPU-portable path (and the `--rubric`, which runs on numpy) stay rf, byte-unchanged. Reuse-by-import / default flip; **NO `sim/` edit** — 3 additive edits to `research/runners/first_chat_console.py`: the `--composer` choices+default (`auto`), the help, and the `auto`→onebrain-on-cupy/rf-on-numpy resolution before the `_onebrain` derivation.

## Verified (both backends)
- **numpy → rf:** `--rubric` = **10/10, moat leaks 0, VERDICT PASS** — byte-unchanged (the `auto` default resolves to rf on numpy; the regression is intact, no error).
- **cupy → onebrain:** `--demo` confirms `[console] C3: composer=OneBrainComposer (... on ONE persistent spiking bridge)` — the `auto` default resolved to the spiking onebrain; the demo runs clean (who/what recall "the curry describes pine" / "the plate touches autumn", grounded PPMI hedges, unknown-word abstain), **moat leaks 0 (CLEAN)**.

## Why it's the highest-value flip
The flagship chat was still running the numpy *reference* composer by default even though the spiking onebrain path was built + validated answer-identical (C3: 24/24, moat 0-FA, rubric 10/10). One default flip makes the live chat fully-spiking-on-one-brain (GPU), with the rf oracle retained for CPU/test/portability. Built by the controller (not a subagent) to avoid the GPU-verify background-wait stall that B1 and R1-a hit.
