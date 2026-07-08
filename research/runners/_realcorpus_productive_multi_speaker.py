"""A PRODUCTIVE multi-bridge A->W speaker: dispatches whole-word spelling across BRIDGE-1/2/3/4 AND composes
PRODUCTIVE inflections (a 3sg form NOT stored as a lexeme is spelled as spell(stem)+spell(affix), reusing the
CYCLE-1024 affix bridge). So "gives" = spell("give")+spell("s"), "runs" = spell("run")+spell("s") -- on spikes,
never a stored 3sg lexeme. Exposes the ConceptFrameSpeaker `spell(word)` contract so the EMERGE-72/74 registry
producer renders richer constructions (C_PPGOAL/C_PPLOC/C_DITRANS) with productive morphology. NO `sim/` edit.
"""
from __future__ import annotations
from research.runners._realcorpus_multi_bridge_speaker import MultiBridgeFrameSpeaker, DEFAULT_BRIDGES
from research.runners._realcorpus_full_frame_speech_derisk import ConceptFrameSpeaker
from research.runners._realcorpus_train_breadth_aw4 import VOCAB as V4, WORD_TO_POOL as P4
from research.runners._realcorpus_train_affix_pool import VOCAB as VA, WORD_TO_POOL as PA

BRIDGE4 = "bridges/breadth_aw4/seed42.simstate.h5"
AFFIX = "bridges/affix_aw/seed42.simstate.h5"


class ProductiveMultiSpeaker:
    """Multi-bridge whole-word spelling (BRIDGE-1/2/3/4) + productive stem+affix inflection (the affix bridge)."""

    def __init__(self, seed=42, aw_seed=42, reset_steps=150):
        bridges = list(DEFAULT_BRIDGES) + [(BRIDGE4, V4, P4)]
        self.multi = MultiBridgeFrameSpeaker(bridges=bridges, seed=aw_seed)
        self.affix = ConceptFrameSpeaker(AFFIX, seed=aw_seed, vocab=VA, word_to_pool=PA)
        self.vocab = self.multi.vocab
        # a LARGER pre-spell settling window fully decays the Izhikevich adaptation carried from the prior spell
        # (the EMERGE-75b A->W read-path state-carryover surpass; 50 steps was insufficient on deep render history).
        self.reset_steps = reset_steps

    def _spell_word(self, word):                                 # underlying whole-word spell with extra settling
        return self.multi._of[word].spell(word, reset_steps=self.reset_steps)

    def spell(self, word):
        """Spell a word ON SPIKES: whole-word if stored, else productive stem+affix (3sg), else None (moat)."""
        if word in self.multi._of:
            return self._spell_word(word)
        for affix in ("ies", "es", "s"):                         # productive 3sg: longest affix first
            if word.endswith(affix) and len(word) > len(affix):
                stem = word[:-len(affix)]
                if affix == "ies":
                    stem = stem + "y"                            # flies -> fly (the -ies allomorphy inverse)
                if stem in self.multi._of:
                    st = self._spell_word(stem)
                    af = self.affix.spell(affix, reset_steps=self.reset_steps)
                    if st is not None and af is not None:
                        return f"{st}{af}"
        return None
