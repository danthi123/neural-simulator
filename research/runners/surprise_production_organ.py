"""EXPECTATION-VIOLATION / SURPRISE wired into the PRODUCTION conversational turn (Gate-B, D2, 2026-08-12).

The owner's "understanding of consequences / expectation": when an incoming assertion (agent, action, patient)
VIOLATES the brain's stored expectation — the user says "the dog eats grass" while the brain holds
"(dog,eats)->meat" — a genuinely-SPIKING surprise signal fires, so the brain can honestly NOTICE
("that surprises me — I'd learned <stored>") instead of silently overwriting what it knew. This is a SEMANTIC
CONTENT mismatch read from spikes, NOT a host `recalled_patient == asserted_patient` string compare.

It REUSES (does not reinvent) the adversarially-verified D2 faculty
(`research/runners/_spiking_expectation_rpe_derisk.py`, 6/6 GO at the robust operating point, lesion-decisive):
a spiking predictive-coding MISMATCH unit. cue (agent,action) --Hebbian topographic--> patient_expected (an FS /
PV-like interneuron that delivers GABA_A SUBTRACTIVE inhibition = the recalled prediction); patient_asserted
--excitation (the sensory drive of the asserted patient)--> surprise (RS pyramidal). CONFIRM (assert==expected):
excitation AND matching inhibition -> cancel -> ~0 Hz. CONTRADICT / NOVEL: the asserted concept's surprise block
is excited but NOT inhibited (the prediction inhibits the EXPECTED block) -> FIRES. The surprise pool's windowed
firing rate off `cp_firing_states[surprise]` IS the signal.

BRAIN-BASED: the surprise = a `cp_firing_states[surprise]` READ; `current_reward_signal == 0`; no host subtraction
of the asserted vs expected codes produces the signal. The DECISION to surface the notice is a threshold on that
spiking rate (calibrated at build). The only host boundary is the SENSORY encoding — the asserted-patient token
delivered as drive to its concept block, and the STORED patient RECALLED (via the brain's own spiking recall) to
pick which block the cue predicts (exactly the de-risk's legitimate teacher/environment boundary).

MOAT-SAFE + ADDITIVE: surprise NEVER manufactures a fact, flips an abstain, or enters the certainty band. It runs
ONLY when the brain ALREADY HOLDS a stored (agent,action)->patient (a genuine expectation to violate), and it only
PREPENDS an honest functional NOTICE to the turn's normal answer. Default-ON; `BRAIN_SURPRISE=0` -> the
byte-identical oracle (fully skipped).

LESION-LOAD-BEARING: zeroing the patient_expected->surprise prediction edges (`BRAIN_SURPRISE_LESION=1`) removes
the subtractive inhibition, so CONFIRM fires as high as CONTRADICT -> the separation collapses (the de-risk's
22.8x -> ~1.0x). The surprise on a CONFIRMED assertion is exactly the part the spiking prediction cancelled, so
the discrimination is caused by the learned spiking prediction, not a fixed input-driven artifact.

HONEST RESIDUALS (declared, ride existing burn-down items):
  * CO-RESIDENT: the mismatch unit runs on ITS OWN circuit bridge, ALONGSIDE the recall composer, not merged onto
    the ONE recall bridge — rides on the one-brain merge (burn-down #1), exactly as the affect organ does.
  * PRECISION BOUNDARY (the de-risk's mapped residual): at LOW prediction gain the GO drops to 3/6 (the
    divisive-normalization / gain-match companion process is proxied by a fixed weight). We wire at the ROBUST
    operating point (cue_to_expected_weight=0.8), where the separation is 6/6-GO with headroom.
  * TOPOGRAPHIC prior: the which-patient mapping is a topographic prior with Hebbian-learned STRENGTH; a
    fully-learned all-to-all CA3 recall + homeostatic gain precision are the named next rungs.
  * INFLECTION: the (agent,action) recall + patient-block mapping key on surface tokens (light base-form
    tolerance); a fully inflection-robust lookup rides on the same lemmatization work as D4.

NO `sim/` edit; reuse-by-import; process backend (cupy in production, numpy in tests).
"""
from __future__ import annotations

import os
import re

from research.runners._spiking_expectation_rpe_derisk import (
    build_expectation_circuit,
    train_expectation,
    measure_conditions,
    _drive_read,
    _hard_reset,
    _idx,
    _install_block_diagonal,
)

# Function words stripped to expose an (agent action patient) assertion. Minimal so a declarative SVO resolves
# while a WH-question (patient is the query) reduces to <3 content tokens (-> not an assertion; no surprise read).
_FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "and", "or", "that",
    "this", "these", "those", "it", "its", "they", "them", "he", "she", "his", "her",
    "my", "your", "our", "i", "you", "we", "me", "us", "him", "on", "in", "at", "by", "with",
    "for", "as", "so", "then", "now", "just", "please", "does", "do", "did",
}
_WH = {"what", "who", "whom", "whose", "where", "when", "why", "how", "which"}
_WORD_RE = re.compile(r"[a-zA-Z']+")


def surprise_enabled() -> bool:
    """Default-ON. `BRAIN_SURPRISE` in {0,false,no,off} -> the byte-identical oracle (fully disabled)."""
    v = os.environ.get("BRAIN_SURPRISE")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def surprise_lesioned() -> bool:
    """`BRAIN_SURPRISE_LESION` in {1,true,yes,on} -> zero the prediction->surprise edges (load-bearing lesion)."""
    v = os.environ.get("BRAIN_SURPRISE_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def extract_assertion(text: str):
    """Return (agent, action, patient) surface tokens when `text` is a 3-content-token declarative SVO assertion,
    else None (a WH-question / non-assertion has the patient as the query -> not an expectation-bearing assertion)."""
    toks = [w.lower() for w in _WORD_RE.findall(text or "")]
    if any(t in _WH for t in toks) or "?" in (text or ""):
        return None
    content = [t for t in toks if t not in _FUNCTION_WORDS]
    if len(content) != 3:
        return None
    a, v, p = content
    if a == v or v == p:
        return None
    return a, v, p


class SurpriseProductionOrgan:
    """A process-shared spiking expectation-violation organ. Built ONCE (lazily): the predictive-coding mismatch
    circuit at the robust operating point, TRAINED (Hebbian topographic cue->expected), learning then FROZEN, with
    a build-time calibration of the confirm-vs-contradict firing threshold. Patient CONCEPTS are mapped to circuit
    blocks on demand (stored patients -> the cue-addressable blocks; novel asserted patients -> the spare blocks).
    Each read drives the prediction (cue) then the assertion (cue + asserted patient) and reads surprise firing."""

    def __init__(self, seed: int = 42, cue_to_expected_weight: float = 0.8, n_reps: int = 22):
        self.seed = int(seed)
        self.cue_w = float(cue_to_expected_weight)     # 0.8 = the robust 6/6-GO operating point (de-risk)
        self.n_reps = int(n_reps)
        self._built = False
        self.bridge = self.cfg = self.meta = self.xp = self.idx_map = None
        self.les = None                                # lazily-built lesioned twin (edges zeroed)
        self.threshold = None
        self.calib = None
        # concept-word -> block index. Stored patients occupy [0, n_trained) (cue-addressable); novel asserted
        # patients occupy [n_trained, n_concepts). Simple round-robin/LRU within each range.
        self._block = {}
        self._cue_next = 0
        self._novel_next = None

    def _build_one(self, lesion=False):
        from sim.backend import get_backend
        xp, _ = get_backend()
        bridge, cfg, meta = build_expectation_circuit(
            self.seed, n_trained=8, n_novel=4, blk=24, cue_blk=24,
            cue_to_expected_weight=self.cue_w)
        bridge._blk = meta["blk"]
        regions = ("cue", "patient_expected", "patient_asserted", "surprise")
        idx_map = {n: xp.asarray(_idx(bridge, n)) for n in regions}
        # LEARN the topographic cue->expected association (strength), then FREEZE (per-turn reads never learn).
        train_expectation(bridge, cfg, idx_map, meta, xp, n_reps=self.n_reps)
        cfg.enable_hebbian_learning = False
        if lesion:
            _install_block_diagonal(bridge, "patient_expected", "surprise", meta["blk"], 0.0)  # remove prediction
        return bridge, cfg, meta, xp, idx_map

    def ensure_built(self):
        if self._built:
            return
        self.bridge, self.cfg, self.meta, self.xp, self.idx_map = self._build_one(lesion=False)
        self._novel_next = self.meta["n_trained"]
        # calibrate the confirm-vs-contradict threshold from the trained concepts (the de-risk's measure).
        res = measure_conditions(self.bridge, self.cfg, self.idx_map, self.meta, self.xp)
        conf, contra, nov = res["confirm_hz"], res["contradict_hz"], res["novel_hz"]
        self.threshold = 0.5 * (conf + min(contra, nov))   # midpoint of confirm and the weaker violation
        self.calib = {"confirm_hz": float(conf), "contradict_hz": float(contra), "novel_hz": float(nov),
                      "cue_to_expected_weight": self.cue_w}
        self._built = True

    def _ensure_les(self):
        if self.les is None:
            b, c, m, xp, idx = self._build_one(lesion=True)
            self.les = {"bridge": b, "cfg": c, "meta": m, "xp": xp, "idx_map": idx}
        return self.les

    def _block_for(self, word: str, cue_addressable: bool):
        """Assign a stable circuit block to a patient CONCEPT word. cue_addressable stored patients live in
        [0, n_trained) so the topographic cue can predict them; novel asserted patients live in the spare range."""
        w = str(word).lower()
        if w in self._block:
            return self._block[w]
        n_tr = self.meta["n_trained"]
        n_cc = self.meta["n_concepts"]
        if cue_addressable:
            b = self._cue_next % n_tr
            self._cue_next += 1
        else:
            b = n_tr + (self._novel_next - n_tr) % max(1, n_cc - n_tr)
            self._novel_next += 1
        self._block[w] = b
        return b

    def read_surprise(self, p_stored: str, p_asserted: str, lesion: bool = False) -> float:
        """The SPIKING surprise rate (Hz) for asserting `p_asserted` when the brain expected `p_stored`. Drives a
        PREDICTION phase (cue recalls the expected block) then the ASSERTION phase (cue + asserted patient block)
        and reads `cp_firing_states[surprise]`. Same word -> same block -> CONFIRM (cancel ~0 Hz); different ->
        the asserted block is un-inhibited -> FIRES. `lesion` uses the prediction-removed twin."""
        self.ensure_built()
        s = self._block_for(p_stored, cue_addressable=True)
        # the asserted patient: if it is the SAME concept as the stored one, it MUST share the block (confirm);
        # otherwise a distinct block (a stored one if known, else a novel spare).
        if str(p_asserted).lower() == str(p_stored).lower():
            t = s
        else:
            t = self._block_for(p_asserted, cue_addressable=False)
            if t == s:                                  # avoid an accidental collision masking a contradiction
                t = self._block_for("__" + str(p_asserted).lower(), cue_addressable=False)
        st = self._ensure_les() if lesion else None
        bridge = st["bridge"] if lesion else self.bridge
        idx_map = st["idx_map"] if lesion else self.idx_map
        xp = st["xp"] if lesion else self.xp
        _hard_reset(bridge)
        r = _drive_read(bridge, idx_map,
                        {"cue": (s, 600.0), "patient_asserted": (t, 600.0)},
                        60, xp, ["surprise"], pre_drives={"cue": (s, 600.0)}, pre_steps=60)
        return float(r["surprise"])

    def judge(self, agent: str, action: str, p_stored: str, p_asserted: str, lesion: bool = False) -> dict:
        """Read whether asserting (agent, action, p_asserted) SURPRISES the brain that holds (agent,action)->
        p_stored. Returns the spiking surprise rate, the threshold, and `surprised` (rate >= threshold)."""
        self.ensure_built()
        hz = self.read_surprise(p_stored, p_asserted, lesion=lesion)
        return {
            "on": True, "lesioned": bool(lesion),
            "agent": agent, "action": action, "stored_patient": p_stored, "asserted_patient": p_asserted,
            "surprise_hz": float(hz), "threshold": float(self.threshold),
            "surprised": bool(hz >= self.threshold), "calib": self.calib,
        }


_ORGAN: SurpriseProductionOrgan | None = None


def get_organ(seed: int = 42) -> SurpriseProductionOrgan:
    """The process-shared surprise organ (built once on first use)."""
    global _ORGAN
    if _ORGAN is None:
        _ORGAN = SurpriseProductionOrgan(seed=seed)
    return _ORGAN


def surprise_notice(agent: str, action: str, stored_patient: str) -> str:
    """The honest functional NOTICE surfaced when the mismatch unit fires on an expectation-violating assertion.
    A FUNCTIONAL read of the spiking surprise signal — never a phenomenal claim."""
    return (f"That surprises me — my mismatch monitor fired: I'd learned that {agent} {action} {stored_patient}. ")
