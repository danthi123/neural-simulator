"""The named home for MULTI-FRAME comprehension (richer-syntax #2): a `FrameParser` that comprehends a sentence in
an AUTO-SELECTED word-order frame (SVO / VSO / OSV) end-to-end, on the spiking substrate. It composes the two
validated GO pieces:

  - `FrameSelector` (verb-position cue -> frame ensemble, Hebbian; 2026-06-18-frame-selection-GO.md): the structural
    cue is the VERB'S POSITION (verb-at-0 -> VSO, verb-at-1 -> SVO, verb-at-2 -> OSV); a NEURAL co-firing map selects
    the frame. Only "which word is the verb" is a host lexical lookup (the morphology/POS front end).
  - `MultiFrameParser` (position x frame -> role, Hebbian; 2026-06-18-multiframe-comprehension-GO.md, GO 6/6): given
    the frame, a NEURAL parser assigns each position's role.

Both are reused-by-import from their de-risk runners (the validated implementations); this module only assembles
them into the agent-facing `parse(words, verbs)` -> {agent, action, patient}. So the role assignment is over
POSITION x FRAME (productive across learned frames), not memorized word templates.

Provided to the production agent behind a default-OFF flag (`BrainConversationalAgent(enable_multiframe=True)` /
`OneBrainComposer(enable_multiframe=True)`); default OFF = byte-identical (the flat-SVO BridgeParser path is
unchanged). GPU for real (Hebbian training on the bridge); numpy is a tiny smoke. NO sim/ edit (reuse-by-import).
"""
from __future__ import annotations

from research.runners._phaseB_multiframe_comprehension_derisk import (
    MultiFrameParser, FRAMES, FRAME_KEYS, N_POS)
from research.runners._phaseB_frame_selection_derisk import FrameSelector, VERBPOS_TO_FRAME


class FrameParser:
    """Comprehend a 3-word sentence in an auto-selected frame: detect the verb position (host lexical lookup against
    the known verbs) -> NEURAL frame selection -> NEURAL per-position role assignment. The native SVO frame still
    comprehends (no regression); a non-native frame (VSO/OSV) comprehends productively. Mirrors the BridgeParser API
    enough that the agent can route comprehension through it (it returns {role: word})."""

    def __init__(self, seed=42):
        self.seed = int(seed)
        self.selector = FrameSelector(seed)         # verb-position -> frame (neural)
        self.parser = MultiFrameParser(seed)        # position x frame -> role (neural)

    def _verb_position(self, words, verbs):
        """The position of the (first) verb in the sentence, found by the known-verb lexical lookup. Falls back to
        position 1 (the native SVO verb slot) when no word is a known verb -> the parser then comprehends as SVO."""
        for pos, w in enumerate(words):
            if w in verbs:
                return pos
        return FRAMES["SVO"].index("action")

    def select_frame(self, words, verbs):
        """The frame the NEURAL selector picks for this sentence (from the verb position)."""
        return self.selector.select(self._verb_position(words, verbs))

    def parse(self, words, verbs):
        """{role: word} for a 3-word sentence, the frame auto-selected from the verb position and the roles read out
        neurally per position. `verbs` is the agent's known-verb set (the lexical front end)."""
        words = list(words)
        assert len(words) == N_POS, "this multi-frame parser handles 3-word sentences (S/V/O in some order)"
        frame = self.select_frame(words, verbs)
        fi = FRAME_KEYS.index(frame)
        roles = [self.parser.role_of(pos, fi)[0] for pos in range(N_POS)]   # role_of -> (role, margin)
        return {role: words[pos] for pos, role in enumerate(roles)}
