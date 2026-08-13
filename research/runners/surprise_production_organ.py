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
  * PRECISION COMPANION (WIRED 2026-08-13, default-ON): the per-block HOMEOSTATIC PREDICTION-GAIN equalizer
    (`_homeostat`, de-risk GO 6/6, `2026-08-13-surprise-organ-homeostat-GO.md`) now runs in `ensure_built`, lifting
    the single-read confirm precision (`het_vote_rate` 0.9375 -> 1.0: every FAMILIAR edge reads reliably below
    threshold) while surprise specificity holds BY CONSTRUCTION (the topographic block-diagonal prediction inhibits
    only its own confirm block, so contradict/novel are untouched). `BRAIN_SURPRISE_HOMEOSTAT=0` reverts to the
    uniform-0.8-gain circuit (byte-identical to the pre-wiring organ). RESIDUAL: the equalizer is a BUILD-TIME
    host-orchestrated calibration loop (like the organ's existing threshold + Hebbian `train_expectation` loops); an
    ONLINE spiking inhibitory/homeostatic-plasticity rule (Vogels 2011) is the further step. The which-patient
    MAPPING is still a topographic prior (a fully-learned all-to-all CA3 recall is the unchanged separate rung).
  * TOPOGRAPHIC prior: the which-patient mapping is a topographic prior with Hebbian-learned STRENGTH; a
    fully-learned all-to-all CA3 recall + homeostatic gain precision are the named next rungs.
  * INFLECTION: the (agent,action) recall + patient-block mapping key on surface tokens (light base-form
    tolerance); a fully inflection-robust lookup rides on the same lemmatization work as D4.

NO `sim/` edit; reuse-by-import; process backend (cupy in production, numpy in tests).
"""
from __future__ import annotations

import os
import re

import numpy as np

from research.runners._spiking_expectation_rpe_derisk import (
    build_expectation_circuit,
    train_expectation,
    measure_conditions,
    _drive_read,
    _hard_reset,
    _host,
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


def surprise_homeostat_enabled() -> bool:
    """Default-ON. The per-block HOMEOSTATIC PREDICTION-GAIN equalizer (the precision companion, de-risk GO 6/6,
    `2026-08-13-surprise-organ-homeostat-GO.md`) runs at build. `BRAIN_SURPRISE_HOMEOSTAT` in {0,false,no,off} ->
    the PRE-HOMEOSTAT circuit (uniform 0.8 prediction gain), byte-identical to the pre-wiring surprise organ."""
    v = os.environ.get("BRAIN_SURPRISE_HOMEOSTAT")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def _install_block_gains(bridge, meta, src, dst, gains):
    """Set the TOPOGRAPHIC (block-diagonal) src->dst weights to a PER-BLOCK gain vector `gains[block]` (concept c of
    src -> concept c of dst), zeroing cross-concept edges. Operates on the CSR weight matrix (orientation-robust);
    the per-block generalization of `_spiking_expectation_rpe_derisk._install_block_diagonal`. This is the equalizer's
    write path — the de-risk (`_surprise_organ_homeostat_derisk`) validated the identical routine 6/6."""
    import scipy.sparse as sp
    src_idx = set(int(i) for i in _idx(bridge, src))
    dst_idx = set(int(i) for i in _idx(bridge, dst))
    src_base = min(src_idx); dst_base = min(dst_idx)
    blk = meta["blk"]
    M = bridge.cp_connections.tocsr()
    indptr = np.asarray(_host(M.indptr)); indices = np.asarray(_host(M.indices))
    data = np.asarray(_host(M.data)).astype(np.float32)
    # orientation: is a CSR row the post (dst) or the pre (src)? (same probe as _install_block_diagonal)
    row_is_dst = row_is_src = 0
    for r in range(M.shape[0]):
        r_in_dst = r in dst_idx; r_in_src = r in src_idx
        if not (r_in_dst or r_in_src):
            continue
        for off in range(int(indptr[r]), int(indptr[r + 1])):
            c = int(indices[off])
            if r_in_dst and c in src_idx:
                row_is_dst += 1
            if r_in_src and c in dst_idx:
                row_is_src += 1
    row_is_post = row_is_dst >= row_is_src
    for r in range(M.shape[0]):
        for off in range(int(indptr[r]), int(indptr[r + 1])):
            c = int(indices[off])
            post, pre = (r, c) if row_is_post else (c, r)
            if pre in src_idx and post in dst_idx:
                sc = (pre - src_base) // blk
                dc = (post - dst_base) // blk
                data[off] = float(gains[dc]) if sc == dc else 0.0
    bridge.cp_connections = sp.csr_matrix((data, indices, indptr), shape=M.shape)


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

    def __init__(self, seed: int = 42, cue_to_expected_weight: float = 0.8, n_reps: int = 22,
                 hz_target: float = 0.5, gain_eta: float = 0.18, gain_max: float = 3.0, homeo_reps: int = 12):
        self.seed = int(seed)
        self.cue_w = float(cue_to_expected_weight)     # 0.8 = the robust 6/6-GO operating point (de-risk)
        self.n_reps = int(n_reps)
        # ── the per-block HOMEOSTATIC PREDICTION-GAIN equalizer (the precision companion; de-risk GO 6/6) ──
        self.hz_target = float(hz_target)      # per-block confirm set-point (well below any contradict/novel)
        self.gain_eta = float(gain_eta)        # homeostatic step: weight per Hz of confirm error over target
        self.gain_max = float(gain_max)        # cap on the per-block prediction gain (no runaway)
        self.homeo_reps = int(homeo_reps)
        self.pred_gains = None                 # per-trained-block cue->expected gain (the equalized precision)
        self.homeo_trace = []                  # per-rep max confirm error (convergence record)
        self.confirm_before = None             # per-block confirm at base gain (the residual, for transparency)
        self.confirm_after = None              # per-block confirm after equalization
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

    def _confirm_per_block(self):
        """Per-stored-block CONFIRM surprise rate (Hz): drive cue i (prediction phase) then cue i + asserted i, read
        the surprise pool — the organ's OWN read path (identical to `measure_conditions`' confirm branch)."""
        nt = self.meta["n_trained"]
        rates = []
        for i in range(nt):
            _hard_reset(self.bridge)
            r = _drive_read(self.bridge, self.idx_map,
                            {"cue": (i, 600.0), "patient_asserted": (i, 600.0)},
                            60, self.xp, ["surprise"], pre_drives={"cue": (i, 600.0)}, pre_steps=60)
            rates.append(r["surprise"])
        return np.asarray(rates)

    def _homeostat(self):
        """THE COMPANION PROCESS — an iterative per-block prediction-gain equalizer (precision-weighted predictive
        coding: the gain that cancels an expected input is set by a homeostatic / divisive-normalization control;
        Vogels-Sprekeler-Zenke-Ganguli-Gerstner 2011; Feldman & Friston 2010). Where a stored block's CONFIRM error
        (the surprise pool's firing on a FAMILIAR assertion) exceeds the target, strengthen THAT block's top-down
        prediction gain (cue->patient_expected); iterate until every familiar read is at target (or the cap is hit).
        Specificity is preserved BY CONSTRUCTION: the topographic block-diagonal prediction inhibits ONLY its own
        confirm block, so contradict/novel reads (a DIFFERENT block) are untouched. Validated 6/6 by the de-risk."""
        nt = self.meta["n_trained"]
        base = self.cue_w
        gains = np.full(nt, base, dtype=np.float64)
        self.confirm_before = self._confirm_per_block()
        conf = self.confirm_before
        for _ in range(self.homeo_reps):
            over = np.maximum(0.0, conf - self.hz_target)
            self.homeo_trace.append(float(over.max()))
            if over.max() <= 0.0:
                break
            gains = np.clip(gains + self.gain_eta * over, base, self.gain_max)  # only strengthen (E/I balance)
            _install_block_gains(self.bridge, self.meta, "cue", "patient_expected", gains)
            conf = self._confirm_per_block()
        self.confirm_after = conf
        self.pred_gains = gains
        return gains

    def ensure_built(self):
        if self._built:
            return
        self.bridge, self.cfg, self.meta, self.xp, self.idx_map = self._build_one(lesion=False)
        self._novel_next = self.meta["n_trained"]
        # THE PRECISION COMPANION (default-ON): equalize the per-block prediction gain BEFORE calibrating the
        # threshold, so every familiar (confirm) edge reads reliably below threshold (het_vote_rate -> 1.0) while
        # surprise specificity holds by construction. `BRAIN_SURPRISE_HOMEOSTAT=0` skips it (byte-identical oracle).
        homeostat_on = surprise_homeostat_enabled()
        if homeostat_on:
            self._homeostat()
        # calibrate the confirm-vs-contradict threshold on the (homeostatted) circuit (the de-risk's measure).
        res = measure_conditions(self.bridge, self.cfg, self.idx_map, self.meta, self.xp)
        conf, contra, nov = res["confirm_hz"], res["contradict_hz"], res["novel_hz"]
        self.threshold = 0.5 * (conf + min(contra, nov))   # midpoint of confirm and the weaker violation
        self.calib = {"confirm_hz": float(conf), "contradict_hz": float(contra), "novel_hz": float(nov),
                      "cue_to_expected_weight": self.cue_w}
        if homeostat_on:
            self.calib.update({
                "homeostat": True,
                "pred_gain_min": float(self.pred_gains.min()), "pred_gain_max": float(self.pred_gains.max()),
                "confirm_before_max": float(self.confirm_before.max()),
                "confirm_after_max": float(self.confirm_after.max()),
            })
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
