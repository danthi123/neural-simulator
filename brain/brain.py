"""The brain: cortex + hippocampus + self + sleep, one continuous life saved to a directory.

The cortex the brain speaks with is the pretrained network plus the slow synapses sleep changed (LiveCortex). Every
restart is a night: a brain shut down or started with episodes stored since its last sleep replays them into those
synapses once (brain/sleep.py, unchanged), and the synapses are saved next to the hippocampus, so learning survives.
With no consolidated synapses the live cortex IS the pretrained one (same code path, byte-equal output).
"""
from __future__ import annotations

import fcntl
import json
import os

import torch
import torch.nn.functional as F

from brain import clock
from brain.cortex import Cortex, model_size, resolve_model
from brain.hippocampus import Episode, Hippocampus
from brain.hippocampus_keys import KeyedHippocampus
from brain.plastic_cortex import PlasticCortex
from brain.sleep import sleep as replay_into_cortex

# Scaffold (see SCAFFOLDS.md): a neutral "birth" context that sets the conversational register. Names and facts are
# NOT here; they live in memory.
BIRTH = "System: A friendly conversation between a young digital brain and a person.\n\n"


# Scaffold (SCAFFOLDS.md): which turns are experiences worth storing. Questions and requests are not stored.
_REQUEST = ("tell me", "can you", "could you", "please", "let's", "what", "how", "why", "who", "where", "when", "do you")


def _is_telling(text: str) -> bool:
    t = text.strip().lower()
    return not t.endswith("?") and not t.startswith(_REQUEST)


# Scaffold (SCAFFOLDS.md, sleep row): ordinary experience interleaved into every replay batch and held to the
# pretrained cortex by the KL guard. Same texts as brain/experiments/sleep_probe.py GENERIC (the six-seed setting).
ORDINARY = [
    "User: Hi, how are you?\n\nAssistant: I'm good, thanks! How about you?\n\nUser: Pretty good. It rained all day."
    "\n\nAssistant: Rain can be cozy.\n\n",
    "User: What's a good way to spend a rainy afternoon?\n\nAssistant: Reading a book with tea is nice.\n\n",
    "User: Can you count to five?\n\nAssistant: One, two, three, four, five.\n\n"]
ADAPTERS = "cortex_adapters.pt"
EPISODES = "episodes"            # one file per episode (tensors), written once
EPISODE_INDEX = "episodes.json"  # text + strength/uses/times per episode, rewritten after a turn (small)
INQUIRY_FILE = "inquiry_episode.pt"   # the inquiry lesson (outside the episode list), written once
LOCK = ".lock"

# Sleep settings per cortex size (adopt-1.5b plan step 8), keyed by Cortex.model_name's size tag. No silent fallback:
# a cortex of another size must be given its sleep settings explicitly.
#   0.1b: the integration default (240 steps / lr 5e-4 / KL 1), unchanged.
#   0.4b: six seeds, brain/experiments/results/2026-11-03-sleep-overnight-0.4b-kl3-st480.md.
#   1.5b: PROVISIONAL (3 seeds, results/2026-11-03-cortex-size); not the default cortex until the six-seed bar holds.
SLEEP_DEFAULTS = {
    "0.1b": {"steps": 240, "lr": 5e-4, "kl_weight": 1.0, "rank": 8},
    "0.4b": {"steps": 480, "lr": 1e-3, "kl_weight": 3.0, "rank": 8},
    "1.5b": {"steps": 480, "lr": 5e-4, "kl_weight": 3.0, "rank": 8},
}


def sleep_defaults(cortex_model: str | None) -> dict:
    """The sleep settings for a cortex (by name or path). Raises KeyError for a size with no row in SLEEP_DEFAULTS."""
    size = model_size(cortex_model)
    if size not in SLEEP_DEFAULTS:
        raise KeyError(f"no sleep settings for cortex {cortex_model!r} (size {size}); pass sleep_steps, sleep_lr, "
                       f"sleep_kl_weight and adapter_rank explicitly or add a row to SLEEP_DEFAULTS")
    return dict(SLEEP_DEFAULTS[size])


class CortexMismatch(RuntimeError):
    """A brain home was made with another cortex: its deltas and adapters do not transfer (plan step 10: migrate)."""



MONITOR_OVERLAP = 0.75   # declared scaffold (reply_monitor): share of a draft's words found in one earlier text


def _words(t: str) -> list[str]:
    return [w for w in "".join(c.lower() if c.isalnum() or c in "'" else " " for c in t).split() if w]


def _is_copy(reply: str, earlier: list[str]) -> bool:
    """True if >= MONITOR_OVERLAP of the draft's words (>= 3 of them) occur in one earlier text."""
    w = _words(reply)
    if len(w) < 3:
        return False
    for e in earlier:
        ew = set(_words(e))
        if ew and sum(x in ew for x in w) / len(w) >= MONITOR_OVERLAP:
            return True
    return False


class BrainBusy(RuntimeError):
    """Another process (chat or night) holds this brain home."""


class LiveCortex:
    """The cortex as the brain has it today: the pretrained network plus the slow synapses sleep consolidated.

    While no synapse has changed (or the consolidated lesion is on) every call goes to the pretrained cortex unchanged.
    Otherwise the network runs through PlasticCortex's forward (adapters on), without gradients.
    """

    def __init__(self, cortex: Cortex):
        self.raw = cortex
        self.plastic: PlasticCortex | None = None
        self.consolidated_on = True   # lesion switch: False = speak with the pretrained synapses only
        self._changed = False         # any consolidated synapse non-zero

    def __getattr__(self, name):      # pipe, model, n_layers, matrix_delta, key_of, add_delta
        return getattr(self.raw, name)

    def refresh(self):
        p = self.plastic
        self._changed = p is not None and any(bool(b.detach().abs().sum() > 0) for d in (p.B_k, p.B_v)
                                               for b in d.values())

    @property
    def active(self) -> bool:
        return self._changed and self.consolidated_on

    def forward(self, tokens, state):
        if not self.active:
            return self.raw.forward(tokens, state)
        p = self.plastic
        dt = p.z["emb.weight"].dtype
        was = p.enabled
        p.enabled = True
        with torch.no_grad():
            st = None if state is None else [s.to(dt) for s in state]
            logits, new = p(tokens, st)
        p.enabled = was
        if state is not None:
            new = [n.to(s.dtype) for n, s in zip(new, state)]
        return logits[-1], new

    def read(self, text, state=None):
        return Cortex.read(self, text, state) if self.active else self.raw.read(text, state)

    def speak(self, logits, state, **kw):
        return Cortex.speak(self, logits, state, **kw) if self.active else self.raw.speak(logits, state, **kw)

    def read_nll(self, text, state=None):
        """Read with the per-token prediction error (Brain(arousal_gain)). Declared scope: the pretrained cortex only;
        once sleep has consolidated slow synapses the full-output pass is not wired through the adapters."""
        if self.active:
            raise RuntimeError("Brain(arousal_gain) is declared for brains that have not slept (consolidated cortex "
                               "active); lesion it (consolidated_on=False) or leave the switch off")
        return self.raw.read_nll(text, state)

    def cue_response(self, text, base) -> float:
        """Cortical (perirhinal-like) familiarity of a cue: how strongly the synapses sleep changed respond to it. The
        largest per-token log-prob gain of the cue's own tokens, consolidated synapses on minus off, read from the birth
        base (brain/experiments/confidence_probe.py cue_gain_max). 0.0 while nothing is consolidated."""
        if not self.active:
            return 0.0
        p = self.plastic
        ids = self.pipe.encode(text)
        dt = p.z["emb.weight"].dtype
        st = [s.to(dt) for s in base]
        was = p.enabled
        with torch.no_grad():
            p.enabled = False
            off = F.log_softmax(p(ids, st)[0].float(), -1)
            p.enabled = True
            on = F.log_softmax(p(ids, st)[0].float(), -1)
        p.enabled = was
        tgt = torch.tensor(ids[1:], device=on.device)[:, None]
        return float((on[:-1].gather(1, tgt) - off[:-1].gather(1, tgt)).max())


class Brain:
    # Feeling-of-knowing bars, fitted on the 36-question fitting set only (brain/experiments/abstain_in_brain.py;
    # SCAFFOLDS.md). Hippocampus on: its familiarity score; off: the cortex's cue response in the consolidated synapses.
    KNOW_BAR_HIPPOCAMPUS = 0.22
    KNOW_BAR_CORTEX = 1.0

    def __init__(self, home: str, name: str = "Nova", friend: str = "Dan", device: str = "cpu", cortex: Cortex = None,
                 sleep_on_restart: bool = True, sleep_steps: int | None = None, sleep_lr: float | None = None,
                 sleep_kl_weight: float | None = None, adapter_rank: int | None = None, sleep_log=None,
                 sleep_seed: int = 0, abstain: bool = False, cortex_model: str | None = None, lock: bool = False,
                 key: str | None = None, key_bar: str = "default", gain: str = "auto",
                 episode: str = "tell", speaker_contrast: float = 0.0,
                 near_miss: int = 0, near_miss_frame: str = "N1", near_miss_exclude=None, teacher_context: bool = False,
                 self_completion: bool = True, self_key: str = "exchange", sleep_rehearse: bool | str = False,
                 habituation: int = 0, reply_monitor: int = 0,
                 own_gain: float | None = None, own_gain_copies: bool = False,
                 carry_penalty: bool = False, fake_think: bool = False, inquiry: float | str | None = "auto",
                 arousal_gain: float | None = None, arousal_pin: bool = False,
                 arousal_calibration: tuple | None = None, use_gate: str | None = None,
                 sleep_downscale: float | None = None, rank: str | None = None,
                 downscale_factors: str = "B", downscale_when: str = "before",
                 replay_priority: str | None = None):
        """inquiry: novelty gate for "it asks about what is new to it". Default ON since 2026-11-19 = "auto": the size bar
        (INQUIRY_BAR_BY_SIZE; a size with no row, e.g. 0.1B, stays off) and, in a new home, the lesson lived at birth.
        Six-seed PASS NARROWED + weekly-with-sleep guard (results/2026-11-19-inquiry-{test,weekly}-1.5b.md). None = off
        (the lesion). A float is the bar in
        nats: when a heard telling holds a word whose surprisal, read from the current state (brain/novelty.py), exceeds
        it and nothing was recalled, the brain reinstates its lived inquiry episode (live_inquiry, a declared teacher
        scaffold) before replying. Lesion = None. brain/research/2026-11-18-novelty-inquiry-record.md. "auto" = the
        size-fitted bar (INQUIRY_BAR_BY_SIZE; off for a size with no row) and, in a new home, INQUIRY_LESSON lived at
        birth. Not saved in the home: pass it each time.
        arousal_gain: (switch, default OFF = None: hear() and commit_exchange() are byte-identical to before) arousal as
        encoding gain (mood iteration 6, brain/research/2026-11-03-mood-state-design.md section 19). A float is SPAN.
        The heard turn is read split ("User: ...\n\n" then the reply prompt) with the cortex's own per-token
        prediction error; the mean of its two highest values drives a leaky arousal (Affect.drive_arousal, normalised by
        MU/SD); the turn's matrix-memory write is scaled by g = min(1 + SPAN * arousal, 2), for this reply and in the
        committed state (commit_exchange); the reply's own write keeps the current rule. No direction, no words.
        0.0 = the split read with no gain (the control path). Declared scope: a cortex that has not slept (raises if the
        consolidated cortex is active). Not saved.
        arousal_pin: (default False) the lesion: g held at 1, arousal still computed and read out (info["arousal"]).
        arousal_calibration: (MU, SD) of the drive; None = affect.AROUSAL_CALIBRATION for this cortex size (an error if
        it has no row). The harness may set `arousal_gains` (a list consumed one per heard turn) to replay another
        conversation's g (SHUFFLED) or a constant (CONSTANT), and `arousal.TAU_AROUSAL = 1.0` for the CURRENT arm.
        use_gate: (switch, default OFF = None: the hippocampus is byte-identical to before) M-use, content-confirmed use
        (brain/research/2026-11-19-life-muse-mscale-prereg.md): "raw_win" = a fetch raises its episode's strength only if
        the raw similarity names the same winner by Hippocampus.MARGIN_USE. None = what the home stored. A non-default
        choice is saved in meta.json (like key).
        sleep_downscale: (switch, default OFF = None) M-scale, the night budget homeostat: K (1.0 pre-registered). The
        first slept night sets budget = K * adapter norm (meta adapter_budget); each later night, before replay, B is
        scaled by budget / norm if the norm exceeds it (meta norm_pre_night, downscale_g). Not saved: pass it each time.
        downscale_factors: "B" (the pre-registered M-scale) or "AB" (the M-scale-3 variant: A and B each scaled by
        sqrt(g); same norm cut). Only read when sleep_downscale is set. Not saved: pass it each time.
        downscale_when: "before" (the pre-registered cut, at the start of the night, before replay) or "end" (M-scale-E:
        the same cut after replay, so the waking cortex runs at the budget; SHY renormalization by morning). Only
        read when sleep_downscale is set. Not saved: pass it each time.
        replay_priority: (switch, default OFF = None: Brain.sleep byte-identical) D2-PR, loss-prioritised replay
        (brain/research/2026-11-21-replay-priority-prereg.md): at the start of a night, before replay, each replayed
        episode gets L_e = the cortex's mean per-token NLL of its replay text under the current adapters (no gradient,
        no RNG). "strength_x_loss" = replay weight effective_strength * L_e; "loss" = L_e. The night's weights are in
        last_replay_weights. A non-default choice is saved in meta.json (like use_gate).
        rank: (switch, default OFF = None: recall competition verbatim) R-raw: "raw" = the winner is the highest raw
        similarity, strength only gates firing (Hippocampus.RANK_ON_RAW; brain/research/2026-11-20-hub-rank-raw-prereg.md).
        Not saved: pass it each time.
        cortex_model: checkpoint path or size tag ("0.1b", "1.5b") used when no `cortex` is given; None = the
        default cortex (0.1B). Sleep settings left None come from SLEEP_DEFAULTS for the cortex's size.
        lock: hold <home>/.lock for this object's life (chat and brain.night set it), so two processes never
        live one brain at once; raises BrainBusy if another holds it.
        key: pattern-completion key (switch, default OFF = "S0": the default Hippocampus, store unchanged). "S_all":
        the meaning key read from every layer's stored matrix-memory delta (brain/hippocampus_keys.py) with its own
        dev-fit bar (2026-11-03-meaning-selection-realpath). None = what the home stored, else "S0". A non-S0 choice
        is saved in meta.json; an S0 home is written without a key entry.
        key_bar: which selection bar a non-S0 key uses (switch, default OFF = "default": KEY_BARS, fit at 0.1B).
        "size": the bar refit for this key at this cortex size (KEY_BARS_BY_SIZE; brain/research/
        2026-11-03-hippo-routing-1.5b.md); a size with no row is an error. "slept": the bar refit on slept live reads
        (KEY_BARS_SLEPT_BY_SIZE; brain/research/2026-11-15-sall-slept-bar-1.5b.md). Not saved in the home: pass it each time.
        gain: reinstatement gain. Default ON since 2026-11-14 = "auto": GAIN_BY_SIZE for this cortex size if it has a row
        (1.5B: 0.5; results/2026-11-13-gain-refit-1.5b.md oracle, 2026-11-13-live-gain-1.5b.md live six-seed GO), else
        Hippocampus.GAIN (4.0; 0.1B unchanged). "fixed": always Hippocampus.GAIN (the pre-11-14 behaviour, the flag-off
        arm). "size": GAIN_BY_SIZE, a size with no row is an error. Not saved in the home: pass it each time.
        episode: what a telling stores (switch, default OFF = "tell": the user's sentence). "lived": the exchange, the
        user's sentence and the brain's own reply to it (brain/research/2026-11-14-lived-episode-live-1.5b.md: NO-GO, the
        S0 key read from the exchange routes to the self episodes). "lived_key": the same content, keyed by the telling
        alone (S0 only; brain/research/2026-11-14-lived-key-1.5b.md). Sleep
        replays the stored text, so this also changes what a night replays. Not saved in the home.
        speaker_contrast: (switch, default OFF = 0.0) a telling's stored content becomes d + mu (d - d_self), d_self =
        the change from reading the telling as the brain's own words (oracle LEAD at mu 3, gain 0.5:
        results/2026-11-14-speaker-contrast-2-1.5b.md). S0, or S_all (routing then reads a kept plain copy of the
        change, so routing is unchanged; twice the episode's bytes). Not saved in the home.
        self_completion: default ON since 2026-11-15: the permanent self episodes grown at birth store only their
        completion (what the answer adds after the question; Hippocampus.store completion_of), routing unchanged
        (results/2026-11-15-own-name-day1-probe-1.5b.md; registered weekly test NARROWED, replication on seeds 300-305
        GO: results/2026-11-15-self-completion-replication-1.5b.md). False = the pre-11-15 behaviour, the flag-off arm.
        Acts at birth only: a home keeps what it was born with.
        self_key: what the permanent self episodes grown at birth are keyed by (switch, default OFF = "exchange": the
        whole question-and-answer text). "question": the question alone (Hippocampus.store key_text); content unchanged.
        S0 only (brain/research/2026-11-15-self-key-question-1.5b.md). Acts at birth only.
        sleep_rehearse: (switch, default OFF) the night also holds, by the second KL term, the pretrained cortex's own
        reply to every user turn heard since birth (near_miss.guard_texts on the told episodes), so replay cannot
        teach the cortex to answer with stored user sentences (results/2026-11-15-open-conv-baseline-1.5b.md: day 2
        collapses into one repeated sentence on 4/6 seeds). "reply": the KL covers only the reply tokens, so the heard
        turn itself stays free to be consolidated (prereg addendum 3: the whole-text guard fought the replay of the
        fact sentence). Not saved in the home.
        habituation: (switch, default OFF = 0) the mouth may not repeat a 4-token sequence from its own last
        `habituation` replies or from the user's current turn (no-repeat n-gram across turns; a declared scaffold,
        brain/SCAFFOLDS.md): the cortex's in-context loops (echoing every turn, one line x14) cannot form. Not saved.
        reply_monitor: (switch, default OFF = 0) before a reply enters the conversation state it is compared with the
        turn it answers and with the brain's own earlier replies since waking; a copy (>= MONITOR_OVERLAP of its words
        in one of them, >= 3 words) is withheld and the cortex speaks again from the same state, up to reply_monitor
        times (the last draft is kept if every one is a copy). A declared scaffold (brain/SCAFFOLDS.md): the
        sampled onset of an in-context echo or loop, which the conversation state then repeats on every later turn
        (results/2026-11-16-reply-rehearse-1.5b.md, echo-onset probe), never enters the state. Not saved.
        own_gain: (switch, default OFF = None: the exchange is read into the conversation state as one text) the
        brain's own reply is written into the conversation state's matrix memory at gain own_gain (the heard turn
        at full gain): speaking-induced suppression of self-produced speech, so a sampled copy cannot reinforce itself
        turn after turn (research round 11-16, brain/research/2026-11-16-own-gain-1.5b.md). 1.0 = the split read at
        full gain (the control). Not saved.
        own_gain_copies: (switch, default OFF) own_gain applies only to a reply the reply monitor's comparison calls a
        copy (_is_copy against the heard turn and the brain's own replies since waking); every other reply is written
        at full gain, so the turn structure stays (a flat gain <= 0.25 lost it: results/2026-11-16-own-gain-probe.md).
        The copy is still said; it is not reinforced. Same declared scaffold as reply_monitor. Not saved.
        carry_penalty: (switch, default OFF) the mouth's presence/frequency penalty counts carry across the turns since
        waking, decaying 0.996 per generated token, as the model family's own chat tooling runs it (reference arm;
        results/2026-11-16-onset-baseline.md). Off: the counts reset every turn. Not saved.
        fake_think: (switch, default OFF) each reply is prompted with the model makers' recommended chat prefix,
        "Assistant: <think></think" (the cortex continues ">" + reply; the ">" is not spoken), and the exchange enters the
        conversation state in that form (rwkv7-g1 model card: "Fake think prompt (great result, highly recommended)").
        A format choice of the body's interface, not a brain process; brain/research/2026-11-16-fake-think-1.5b.md.
        Not saved."""
        self.home = home
        self._key_arg = key
        if use_gate not in (None, "raw_win"):
            raise ValueError(f"use_gate must be None or 'raw_win', not {use_gate!r}")
        self._use_gate = self._use_gate_arg = use_gate
        if rank not in (None, "raw"):
            raise ValueError(f"rank must be None or 'raw', not {rank!r}")
        self.rank = rank
        if replay_priority not in (None, "strength_x_loss", "loss"):
            raise ValueError(f"replay_priority must be None, 'strength_x_loss' or 'loss', not {replay_priority!r}")
        self._replay_priority = self._replay_priority_arg = replay_priority
        self.last_replay_weights = None   # the last night's per-episode weights (only with replay_priority on)
        self.sleep_downscale = None if sleep_downscale is None else float(sleep_downscale)
        if downscale_factors not in ("B", "AB"):
            raise ValueError(f"downscale_factors must be 'B' or 'AB', not {downscale_factors!r}")
        self.downscale_factors = downscale_factors
        if downscale_when not in ("before", "end"):
            raise ValueError(f"downscale_when must be 'before' or 'end', not {downscale_when!r}")
        self.downscale_when = downscale_when
        self.key_bar = key_bar
        self.gain_mode = gain
        if episode not in ("tell", "lived", "lived_key"):
            raise ValueError(f"episode must be 'tell', 'lived' or 'lived_key', not {episode!r}")
        self.episode_format = episode
        self.speaker_contrast = float(speaker_contrast)
        self.self_completion = bool(self_completion)
        if self_key not in ("exchange", "question"):
            raise ValueError(f"self_key must be 'exchange' or 'question', not {self_key!r}")
        self.self_key = self_key
        if sleep_rehearse not in (False, True, "reply"):
            raise ValueError(f"sleep_rehearse must be False, True or 'reply', not {sleep_rehearse!r}")
        self.sleep_rehearse = sleep_rehearse
        self.habituation = int(habituation)
        self.arousal_gain = None if arousal_gain is None else float(arousal_gain)
        self.arousal_pin = bool(arousal_pin)
        self.arousal = None          # Affect (arousal fields only) while the switch is on
        self.arousal_gains = None    # harness hook: gains to apply in place of the computed ones, consumed per turn
        self._turn_gain = 1.0
        if self.arousal_gain is not None:
            from brain.affect import AROUSAL_CALIBRATION, Affect
            cm = cortex_model or getattr(cortex, "model_name", None)
            size = model_size(cm) if cm else None
            if arousal_calibration is None and size not in AROUSAL_CALIBRATION:
                raise ValueError(f"arousal_gain: no (MU, SD) for cortex size {size!r}; pass arousal_calibration "
                                 f"(brain.affect.calibrate_surprise) or add a row to AROUSAL_CALIBRATION")
            mu, sd = arousal_calibration or AROUSAL_CALIBRATION[size]
            self.arousal = Affect(enabled=False, surprise_mu=mu, surprise_sd=sd)
            self.arousal.gain_on = not self.arousal_pin
        self.reply_monitor = int(reply_monitor)
        self.own_gain = None if own_gain is None else float(own_gain)
        self.own_gain_copies = bool(own_gain_copies)
        self.carry_penalty = bool(carry_penalty)
        self.fake_think = bool(fake_think)
        self._asst = "Assistant: <think></think" if self.fake_think else "Assistant:"
        self._occ: dict = {}
        self.monitor_log: list[int] = []   # per turn: drafts withheld
        self._recent_replies: list[str] = []
        self._inquiry_ep = None   # the inquiry lesson: kept OUTSIDE the hippocampus list (only the novelty gate reaches it)
        # Near-miss guard (switch, default OFF = 0; brain/near_miss.py, brain/research/2026-11-14-near-miss-guard-1.5b.md):
        # before each night the pretrained cortex poses near_miss questions per episode about something it was not
        # told, and sleep holds its answers to them by a second KL term. near_miss_exclude: test hygiene only (a probe
        # drops cues that name its evaluation topics); None = keep every cue.
        self.near_miss, self.near_miss_frame, self.near_miss_exclude = int(near_miss), near_miss_frame, near_miss_exclude
        # Teacher night (switch, default OFF; brain/research/2026-11-17-teacher-night-context-1.5b.md): the near-miss
        # guard's replies and KL target come from the pretrained cortex WITH the day's tellings in its context (the
        # stored episodes replayed in order), not from the bare birth state. The bare cortex invents on 79-86% of
        # never-told questions; with the tellings in context it declines most (results/2026-11-17-incontext-abstain.md).
        self.teacher_context = bool(teacher_context)
        self.near_miss_log = []
        os.makedirs(home, exist_ok=True)
        self._lock_fh = None
        if lock:
            self._acquire_lock()
        try:
            self.abstain = abstain   # switch (default OFF): below the feeling-of-knowing bar, answer from "not known"
            self.inquiry = inquiry   # default "auto" (size bar + birth lesson; on at 1.5B); None = off: novelty bar (nats)
            self.inquiry_birth = False   # "auto": the lesson is lived at birth (set below once the size is known)
            self.know_bar = {"hippocampus": self.KNOW_BAR_HIPPOCAMPUS, "cortex": self.KNOW_BAR_CORTEX}
            # Which read-out decides "known" (abstention v2, brain/experiments/abstain_v2.py): "fok" = v1 (hippocampal
            # familiarity, or the cortical cue response with the hippocampus off); "F" = familiarity only, "R" =
            # recollection only, "FR" = both must pass (bars know_bar["fam"], know_bar["rec"]). rec_variant: "max"/"mean".
            self.know_rule = "fok"
            self.rec_variant = "max"
            self.recollect_tokens = 12
            # Below the bar, reinstate the taught "when I have no memory of something" episode (meta["self_belief_unknown"],
            # stored by teach_self_belief) instead of nothing. Default OFF.
            self.reinstate_self_belief = False
            self.sleep_extra = []    # (text, strength) replayed every night beside the episodes; [] = none (method 2)
            self.raw_cortex = cortex or Cortex(model_path=resolve_model(cortex_model), device=device)
            self.cortex_model = getattr(self.raw_cortex, "model_name", None)   # None for a test mock
            self.cortex = LiveCortex(self.raw_cortex)
            if self.inquiry == "auto":   # the size-fitted bar, with the lesson lived at birth; no row = off
                self.inquiry = self.INQUIRY_BAR_BY_SIZE.get(model_size(self.cortex_model) if self.cortex_model else None)
                self.inquiry_birth = self.inquiry is not None
            self.hippocampus = self._make_hippocampus(key or "S0")
            self.sleep_on_restart = sleep_on_restart
            given = {"steps": sleep_steps, "lr": sleep_lr, "kl_weight": sleep_kl_weight, "rank": adapter_rank}
            if any(v is None for v in given.values()):
                given = {k: (sleep_defaults(self.cortex_model)[k] if v is None else v) for k, v in given.items()}
            self.sleep_cfg = {k: given[k] for k in ("steps", "lr", "kl_weight")}
            self.adapter_rank = given["rank"]
            self.sleep_log = sleep_log
            self.sleep_seed = sleep_seed  # night n replays with seed sleep_seed + n; adapters init from sleep_seed
            # Birth context is read by the pretrained cortex: stored episodes are deltas from this base, so it must not
            # move when sleep changes the slow synapses.
            _, self.base = self.raw_cortex.read(BIRTH, None)
            self.state = [x.clone() for x in self.base]
            self.meta = {"name": name, "friend": friend, "born": clock.now(), "turns": 0, "sleeps": 0, "last_sleep": 0.0,
                         "cortex_model": self.cortex_model}
            if not self.load():
                self._grow_self()
                if self.inquiry_birth:
                    self._live_inquiry_at_birth()
                self.meta["last_sleep"] = clock.now()  # birth is not a lived day: the first night comes at a restart
                self.save()
            elif self.sleep_on_restart and self.pending():
                self.sleep()  # restart = a night: consolidate what was lived since the last sleep, once
                self.save()
        except BaseException:
            self.release()   # a failed start must not keep the home locked
            raise

    # Dev-fit selection bars (FAMILIARITY, MARGIN) per non-default key; S0 keeps the Hippocampus class constants.
    KEY_BARS = {"S_all": (0.185, 0.05)}
    # Per-size refits, used only with key_bar="size": (key, size tag) -> (FAMILIARITY, MARGIN). S_all at 1.5B: fit on the
    # DEV bank only (3 seeds, unslept reads; brain/experiments/hippo_route_probe.py + hippo_route_table.py --fit).
    KEY_BARS_BY_SIZE = {("S_all", "1.5b"): (0.22, 0.01)}
    # Refits on SLEPT live reads (key_bar="slept"): the DEV bank told live, one night at the size defaults, read through
    # the slept cortex (brain/research/2026-11-15-sall-slept-bar-1.5b.md; sall_live_bar_fit.py score --size-sleep).
    # S_all 1.5B: seeds 42/43/44, fit J 0.407 (results/2026-11-18-sall-slept-fit-1.5b.md).
    KEY_BARS_SLEPT_BY_SIZE = {("S_all", "1.5b"): (0.165, 0.02)}

    GAIN_BY_SIZE = {"1.5b": 0.5}
    # inquiry="auto": the novelty bar per size (fit on the dev bank, brain/research/2026-11-19-curiosity-test-prereg.md)
    # and the lesson lived at birth (declared teacher scaffold, SCAFFOLDS.md). A size with no row = off.
    INQUIRY_BAR_BY_SIZE = {"1.5b": 12.4}
    INQUIRY_LESSON = ("My friend keeps a zorbit in her kitchen.", "What's a zorbit? I've never heard of that word.")

    def _make_hippocampus(self, key: str) -> Hippocampus:
        h = self._hippocampus_for(key)
        if self.gain_mode not in ("auto", "fixed", "size"):
            raise ValueError(f"gain must be 'auto', 'fixed' or 'size', not {self.gain_mode!r}")
        if self.gain_mode != "fixed":
            size = model_size(self.cortex_model) if self.cortex_model else None
            if size in self.GAIN_BY_SIZE:
                h.GAIN = self.GAIN_BY_SIZE[size]
            elif self.gain_mode == "size":
                raise ValueError(f"no size-fitted reinstatement gain at cortex size {size!r}")
        if self._use_gate is not None:
            h.USE_GATE = self._use_gate
        if self.rank == "raw":
            h.RANK_ON_RAW = True
        return h

    def _hippocampus_for(self, key: str) -> Hippocampus:
        if key == "S0":
            return Hippocampus()
        if self.key_bar in ("size", "slept"):
            size = model_size(self.cortex_model)
            table = self.KEY_BARS_BY_SIZE if self.key_bar == "size" else self.KEY_BARS_SLEPT_BY_SIZE
            if (key, size) not in table:
                raise ValueError(f"no {self.key_bar}-fitted bar for key {key!r} at cortex size {size!r}")
            fam, margin = table[(key, size)]
        elif self.key_bar != "default":
            raise ValueError(f"key_bar must be 'default', 'size' or 'slept', not {self.key_bar!r}")
        elif key not in self.KEY_BARS:
            raise ValueError(f"no fitted bar for key {key!r}")
        else:
            fam, margin = self.KEY_BARS[key]
        return KeyedHippocampus(key=key, familiarity=fam, margin=margin)

    @property
    def key(self) -> str:
        return getattr(self.hippocampus, "key_name", "S0")

    # ---- self ----
    def _grow_self(self):
        n, f = self.meta["name"], self.meta["friend"]
        # Self-knowledge is stored as lived exchanges (question and own answer), so the matching question cues it.
        for q, ans in ((f"What's your name?", f"My name is {n}."), (f"Who are you?", f"I'm {n}, a young digital brain."),
                       (f"Do you remember my name?", f"Of course, you're {f}, my friend.")):
            self.hippocampus.store(self.cortex, self.base, f"{q}\n\nAssistant: {ans}", speaker="User", salience=3.0,
                                   permanent=True, completion_of=q if self.self_completion else None,
                                   **({"key_text": q} if self.self_key == "question" else {}))

    # ---- one turn ----
    def hear(self, text: str):
        """Process one user turn; returns (reply, info). Memory returns as cortical state, never as text."""
        text = text.strip()
        info = {"recalled": None, "similarity": None, "stored": False}
        known = True
        if self.abstain and not _is_telling(text):
            known, sig = self.knows(text)
            info.update(sig, known=known)
        gated = self.hippocampus.USE_GATE is not None or self.hippocampus.FREEZE_USE or self.hippocampus.RANK_ON_RAW
        if gated:
            self.hippocampus.last_fetch = None
        if known:
            ep, sim = self.hippocampus.recall(self.cortex, self.base, text, question=text.endswith("?"))
        else:  # nothing familiar enough: no pattern completion, no reinstatement
            ep, sim = None, (info["fok"] if info.get("fok_source") == "hippocampus" else info.get("fam", 0.0))
        info["similarity"] = round(sim, 3)
        if gated:
            info["fetch"] = self.hippocampus.last_fetch
        # Recall is a transient reactivation for THIS reply; it does not accumulate in the ongoing state.
        active = self.state
        if ep is not None:
            active = self.hippocampus.reinstate(self.cortex, self.state, ep)
            info["recalled"] = ep.text
        elif self.inquiry is not None and _is_telling(text) and self.inquiry_episode() is not None and \
                self._novelty(text) > self.inquiry:
            # a word in what was heard is new to the cortex: its own surprise brings back the lived episode in which it
            # met an unknown word (an ordinary episode, reinstated as state; no reply text exists in the host)
            ep_q = self.inquiry_episode()
            active = self.hippocampus.reinstate(self.cortex, self.state, ep_q)
            info["recalled"], info["inquiry"] = ep_q.text, True
        elif not known and self.reinstate_self_belief and self.self_belief() is not None:
            # the low-familiarity state brings back what the teacher once said about not knowing (an ordinary episode)
            belief = self.self_belief()
            active = self.hippocampus.reinstate(self.cortex, self.state, belief)
            info["recalled"], info["self_belief"] = belief.text, True
        # Below the bar no retrieval is attempted: no reinstatement, and this reply is spoken without the synapses sleep
        # consolidated (they hold what was lived). What it then says is the cortex's own continuation; no reply text
        # exists in the host (a fixed "not known" line read before answering was tried on the fitting set and dropped).
        retrieval = self.cortex.consolidated_on
        self.cortex.consolidated_on = retrieval and known
        try:
            if self.arousal_gain is None:
                logits, st = self.cortex.read(f"User: {text}\n\n{self._asst}", active)
            else:   # heard turn read split, with the cortex's own prediction error; its write at the arousal gain
                from brain.affect import top2_mean
                _, su, nll = self.cortex.read_nll(f"User: {text}\n\n", active)
                self.arousal.drive_arousal(top2_mean(nll[1:]))
                g = self._next_gain()
                logits, st = self.cortex.read(self._asst, self._write_gain(active, su, g))
                self._turn_gain = g
            blk = self.raw_cortex.ngram_block(self._recent_replies[-self.habituation:] + [text]) \
                if self.habituation else None
            kw = {**({"block": blk} if blk is not None else {}), **({"carry": self._occ} if self.carry_penalty else {})}
            reply = self._spoken(self.cortex.speak(logits, st, **kw)[0])
            withheld = 0
            while withheld < self.reply_monitor and _is_copy(reply, [text] + self._recent_replies):
                withheld += 1
                reply = self._spoken(self.cortex.speak(logits, st, **kw)[0])
            if self.reply_monitor:
                self.monitor_log.append(withheld)
                info["withheld"] = withheld
        finally:
            self.cortex.consolidated_on = retrieval
        self.state = self.commit_exchange(text, reply, self.state)
        self._recent_replies.append(reply)
        if self.hippocampus.enabled and _is_telling(text):
            lived = self.episode_format in ("lived", "lived_key") and reply.strip()
            self.hippocampus.store(self.cortex, self.base, f"{text}\n\nAssistant: {reply.strip()}" if lived else text,
                                   speaker="User", key_text=text if lived and self.episode_format == "lived_key" else None,
                                   contrast=self.speaker_contrast)
            info["stored"] = True
        self.meta["turns"] += 1
        if self.arousal is not None:
            info["arousal"] = self.arousal.readout()
        return reply, info

    def _next_gain(self) -> float:
        """The gain of this heard turn: the harness's replayed one if set, else min(1 + SPAN * arousal, 2) (1.0 pinned)."""
        g = self.arousal.encode_gain(self.arousal_gain, self.arousal_pin)
        if self.arousal_gains:
            g = float(self.arousal_gains.pop(0))
            self.arousal.last_gain = g
        return g

    @staticmethod
    def _write_gain(before, after, g: float):
        """`after` with the heard turn's matrix-memory write (after - before) scaled by g (token-shift slots as read).
        g == 1.0 returns `after` unchanged (bit for bit)."""
        if g == 1.0:
            return after
        return [before[i] + g * (after[i] - before[i]) if i % 3 == 1 else after[i] for i in range(len(after))]

    def _spoken(self, raw: str) -> str:
        """What is said of a generated turn: under fake_think the ">" that closes the empty think tag is not spoken."""
        return raw.lstrip(">").strip() if self.fake_think else raw

    def commit_exchange(self, text: str, reply: str, state):
        """The conversation state after hearing `text` and saying `reply`. own_gain None: one read of the exchange.
        Else the heard turn is read at full gain and the reply's change to the matrix memory is scaled by own_gain
        (the token-shift slots hold the last token, the same for any reply, and are taken as read)."""
        head, sep = ("Assistant: <think></think>", "") if self.fake_think else ("Assistant:", " ")
        heard = text
        if self.arousal_gain is not None:   # heard turn at its arousal gain, then the reply by the rules below
            g, self._turn_gain = self._turn_gain, 1.0
            state = self._write_gain(state, self.cortex.read(f"User: {text}\n\n", state)[1], g)
            text = None
        if self.own_gain is None:
            return self.cortex.read(f"{head}{sep}{reply}\n\n" if text is None else
                                    f"User: {text}\n\n{head}{sep}{reply}\n\n", state)[1]
        if text is None:
            _, su = self.cortex.read(head, state)
            _, sp = self.cortex.read(f"{sep}{reply}\n\n", su)
            g = self.own_gain
            if self.own_gain_copies and not _is_copy(reply, [heard] + self._recent_replies):
                g = 1.0
            return [su[i] + g * (sp[i] - su[i]) if i % 3 == 1 else sp[i] for i in range(len(sp))]
        _, su = self.cortex.read(f"User: {text}\n\n{head}", state)
        _, sp = self.cortex.read(f"{sep}{reply}\n\n", su)
        g = self.own_gain
        if self.own_gain_copies and not _is_copy(reply, [text] + self._recent_replies):
            g = 1.0
        return [su[i] + g * (sp[i] - su[i]) if i % 3 == 1 else sp[i] for i in range(len(sp))]

    def knows(self, text: str):
        """The gate's decision for a request turn: (known, read-outs) under know_rule."""
        if self.know_rule == "fok":
            fok, src = self.feeling_of_knowing(text)
            return fok >= self.know_bar[src], {"fok": round(fok, 4), "fok_source": src}
        sig = {}
        if "F" in self.know_rule:
            sig["fam"] = self.hippocampus.familiarity(self.cortex, self.base, text, exclude=self._belief_idx())
        if "R" in self.know_rule:
            sig["rec"] = self.recollection(text)[self.rec_variant]
        return all(sig[k] >= self.know_bar[k] for k in sig), {k: round(v, 4) for k, v in sig.items()}

    def _belief_idx(self):
        i = self.meta.get("self_belief_unknown")
        return () if i is None else (i,)

    def self_belief(self):
        i = self.meta.get("self_belief_unknown")
        return None if i is None or i >= len(self.hippocampus.episodes) else self.hippocampus.episodes[i]

    def teach_self_belief(self, text: str):
        """Declared teacher scaffold (SCAFFOLDS.md): the teacher says once, in conversation, what one does when one has
        no memory of something. It is heard and stored like any telling; the brain only notes which episode it was."""
        reply, info = self.hear(text)
        assert info["stored"], "the lesson must be a telling (stored as an ordinary episode)"
        self.meta["self_belief_unknown"] = len(self.hippocampus.episodes) - 1
        return reply, info

    def live_self_belief(self, question: str, guided_reply: str, trace: str = "exchange"):
        """Confabulation M2 (brain/research/2026-11-03-confabulation-next-methods.md). Declared teacher scaffold
        (SCAFFOLDS.md): the teacher asks one question the brain was never told about; the brain hears it as any question
        (its spontaneous reply is returned, not stored); the teacher then guides the reply once, so the brain's own turn
        is `guided_reply`. The exchange is read into the working state with that reply as the brain's OWN (Assistant)
        turn and stored as one episode tagged source="self" (the brain's own words), noted as the self-belief episode
        that the gate may reinstate. No reply text is kept in the host: later replies are the cortex's continuation.
        trace="self_turn" (M2 v2, results/2026-11-03-confab-m2-v2.md): the episode is still indexed by the whole lived
        exchange (same key), but what it reinstates is only the brain's own turn, read from the birth state with no
        question before it, so the question's topic is not carried back. (Read after the question instead, the own-turn
        change holds the decay of the question trace: cosine -0.48 with it at 0.1B, an inverted topic carrier.)"""
        assert trace in ("exchange", "self_turn")
        question = question.strip()
        assert not _is_telling(question), "the lesson must be a question the brain is asked"
        before = [x.clone() for x in self.state]
        spontaneous, _ = self.hear(question)
        _, self.state = self.cortex.read(f"User: {question}\n\nAssistant: {guided_reply}\n\n", before)
        ep = self.hippocampus.store(self.cortex, self.base, f"{question}\n\nAssistant: {guided_reply}", speaker="User")
        ep.source = "self"
        if trace == "self_turn":
            _, st = self.cortex.read(f"Assistant: {guided_reply}\n\n", self.base)
            ep.delta = [d.half().cpu() if d is not None else None for d in self.cortex.matrix_delta(st, self.base)]
        self.meta["self_belief_unknown"] = len(self.hippocampus.episodes) - 1
        return spontaneous

    def _novelty(self, text: str) -> float:
        from brain.novelty import novelty
        v = novelty(self.cortex, self.state, text)
        self._last_novelty = v
        return v

    def inquiry_episode(self):
        """The lived inquiry lesson, or None. It is not in hippocampus.episodes, so ordinary recall, familiarity, key
        centring, sleep replay and the memory listing never see it; only the novelty gate in hear() reinstates it."""
        return self._inquiry_ep

    def live_inquiry(self, telling: str, guided_reply: str):
        """Declared teacher scaffold (SCAFFOLDS.md), the curiosity twin of live_self_belief(trace="self_turn"): the
        teacher says something with a word the brain does not know; the brain hears it (its spontaneous reply is
        returned, not stored as the lesson); the teacher guides its reply once, so the brain's own turn is
        `guided_reply` (asking what the word means). The exchange is read into the working state and stored as one
        episode tagged source="self"; what it reinstates is only the brain's own turn read from the birth state, so the
        lesson's word is not carried back. Noted as the inquiry episode that the novelty gate may reinstate."""
        telling = telling.strip()
        assert _is_telling(telling), "the lesson is a telling with an unknown word"
        before = [x.clone() for x in self.state]
        inq, self.inquiry = self.inquiry, None          # the lesson itself is heard without the gate
        n_before = len(self.hippocampus.episodes)
        # hearing the lesson must leave the stored memories as they were (a recall would count as a use)
        was = [(e.strength, e.uses, e.last_used) for e in self.hippocampus.episodes]
        try:
            spontaneous, _ = self.hear(telling)
        finally:
            self.inquiry = inq
        del self.hippocampus.episodes[n_before:]
        for e, (st_, us, lu) in zip(self.hippocampus.episodes, was):
            e.strength, e.uses, e.last_used = st_, us, lu
        _, self.state = self.cortex.read(f"User: {telling}\n\nAssistant: {guided_reply}\n\n", before)
        ep = self.hippocampus.store(self.cortex, self.base, f"{telling}\n\nAssistant: {guided_reply}", speaker="User")
        self.hippocampus.episodes.pop()                  # it lives outside the ordinary list (inquiry_episode)
        ep.source = "self"
        _, st = self.cortex.read(f"Assistant: {guided_reply}\n\n", self.base)
        ep.delta = [d.half().cpu() if d is not None else None for d in self.cortex.matrix_delta(st, self.base)]
        self._inquiry_ep = ep
        self._inquiry_dirty = True
        return spontaneous

    def _live_inquiry_at_birth(self):
        """The inquiry lesson as an earlier conversation before the first day: the episode stays, the working state,
        the reply history and the turn count return to birth (the six-seed test lived it the same way), and the random
        number streams are restored, so a turn the gate does not fire on is the same as without the lesson."""
        import random

        import numpy as np
        turns = self.meta["turns"]
        import copy
        py, nps = random.getstate(), np.random.get_state()   # the lesson draws no numbers from the life that follows
        arousal, n_mon = copy.deepcopy(self.arousal), len(self.monitor_log)
        devs = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devs):
            self.live_inquiry(*self.INQUIRY_LESSON)
        random.setstate(py)
        np.random.set_state(nps)
        self.arousal = arousal
        del self.monitor_log[n_mon:]
        self._last_novelty = None
        self.state = [x.clone() for x in self.base]
        self._recent_replies = []
        self.meta["turns"] = turns

    @torch.no_grad()
    def recollection(self, text: str) -> dict:
        """Recollection test (dual process): reinstate the best-matching episode (forced top-1 over topic episodes) and
        ask whether it makes the cortex say something specific it would not say from the same state without it. The
        reinstated cortex answers greedily for recollect_tokens tokens; along that answer, gain_t = log p(token |
        reinstated) - log p(token | not reinstated). Returns {"max": max gain, "mean": mean gain}; 0 with no episode."""
        ep, _ = self.hippocampus.best(self.cortex, self.base, text, exclude=self._belief_idx())
        if ep is None:
            return {"max": 0.0, "mean": 0.0}
        prompt = f"User: {text}\n\nAssistant:"
        lr, sr = self.cortex.read(prompt, self.hippocampus.reinstate(self.cortex, self.state, ep))
        lb, sb = self.cortex.read(prompt, self.state)
        gains, toks = [], []
        for _ in range(self.recollect_tokens):
            tok = int(torch.argmax(lr))
            if tok == 0:
                break
            gains.append(float(F.log_softmax(lr.float(), -1)[tok] - F.log_softmax(lb.float(), -1)[tok]))
            toks.append(tok)
            if "\n\n" in self.cortex.pipe.decode(toks):
                break
            lr, sr = self.cortex.forward([tok], sr)
            lb, sb = self.cortex.forward([tok], sb)
        if not gains:
            return {"max": 0.0, "mean": 0.0}
        return {"max": max(gains), "mean": sum(gains) / len(gains)}

    def feeling_of_knowing(self, text: str):
        """Have I been told about this? Returns (signal, source). With the hippocampus on: its familiarity of the cue
        against every stored episode. With it off: the cortex's response to the cue in the synapses sleep changed."""
        if self.hippocampus.enabled:
            return self.hippocampus.familiarity(self.cortex, self.base, text), "hippocampus"
        return self.cortex.cue_response(f"User: {text}\n\n", [s.float() for s in self.base]), "cortex"

    # ---- lesions (weekly test) ----
    def set_lesions(self, hippocampus_off: bool = False, consolidated_off: bool = False):
        """hippocampus_off: no recall, no storing. consolidated_off: speak with the pretrained synapses only."""
        self.hippocampus.enabled = not hippocampus_off
        self.cortex.consolidated_on = not consolidated_off

    # ---- sleep ----
    def pending(self) -> list:
        """Episodes stored since the last sleep."""
        return [e for e in self.hippocampus.episodes if e.created > self.meta.get("last_sleep", 0.0)]

    def _plastic(self) -> PlasticCortex:
        if self.cortex.plastic is None:
            with torch.random.fork_rng(devices=[]):  # fixed adapter init; leaves the speaking RNG untouched
                torch.manual_seed(self.sleep_seed)
                self.cortex.plastic = PlasticCortex(self.raw_cortex, rank=self.adapter_rank)
        return self.cortex.plastic

    def sleep(self, steps: int | None = None):
        """Replay every stored episode, in proportion to its current strength, into the slow synapses
        (brain/sleep.py: interleaved with ordinary text under the KL guard). Returns the per-step losses."""
        if not self.hippocampus.episodes:
            return []
        plastic = self._plastic()
        if self.sleep_downscale is not None and self.downscale_when == "before":   # M-scale: the cut before replay (no RNG)
            self._night_budget(plastic)
        now = clock.now()
        # replay what the hippocampus encoded (speaker line as read at storage time)
        night_eps = self._night_episodes()
        if self._replay_priority is None:
            eps = [(f"{e.text}\n\n", self.hippocampus.effective_strength(e, now)) for e in night_eps]
        else:
            eps = self._prioritised_replay(plastic, night_eps, now)
        eps += list(self.sleep_extra)   # declared scaffold when non-empty (method 2); [] by default
        base =[s.float() for s in self.base]
        guard = self._near_miss_guard() if self.near_miss else []
        if self.sleep_rehearse:   # the cortex's own replies to what it heard, held to the pretrained cortex
            from brain.near_miss import guard_texts
            heard = [e.text.removeprefix("User: ").strip() for e in self._night_episodes()
                     if not e.permanent and e.text.startswith("User: ")]
            guard = guard + guard_texts(self.raw_cortex, self.base, heard, split=self.sleep_rehearse == "reply")
        with torch.random.fork_rng(devices=[]):
            losses = replay_into_cortex(plastic, base, eps, ORDINARY, steps=steps or self.sleep_cfg["steps"],
                                        lr=self.sleep_cfg["lr"], kl_weight=self.sleep_cfg["kl_weight"],
                                        seed=self.sleep_seed + self.meta.get("sleeps", 0),
                                        log=self.sleep_log, guard=guard)
        for prm in plastic.parameters():
            prm.grad = None
        if self.sleep_downscale is not None and self.meta.get("adapter_budget") is None:
            self.meta["adapter_budget"] = self.sleep_downscale * plastic.adapter_norm()   # first slept night sets it
        if self.sleep_downscale is not None and self.downscale_when == "end":   # M-scale-E: the cut after replay (no RNG)
            self._night_budget(plastic)
        self.meta["sleeps"] = self.meta.get("sleeps", 0) + 1
        self.meta["last_sleep"] = clock.now()
        self._adapters_dirty = True
        self.cortex.refresh()
        return losses

    def _prioritised_replay(self, plastic, night_eps, now) -> list:
        """D2-PR: [(replay text, weight)] with weight = effective_strength * L_e or L_e, L_e the mean per-token NLL of the
        replay text under the current adapters (no gradient, base state, no RNG draw: the torch RNG is forked and the
        replay order seed is untouched). Records self.last_replay_weights."""
        base = [s.float() for s in self.base]
        was = plastic.enabled
        rows = []
        try:
            plastic.enabled = True
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                for e in night_eps:
                    text = f"{e.text}\n\n"
                    ids = self.cortex.pipe.encode(text)
                    tgt = torch.tensor(ids[1:])
                    lp = torch.log_softmax(plastic(ids, base)[0].float(), -1)
                    tgt = tgt.to(lp.device)
                    loss = float(torch.nn.functional.nll_loss(lp[:-1], tgt))
                    st = float(self.hippocampus.effective_strength(e, now))
                    w = loss if self._replay_priority == "loss" else st * loss
                    rows.append({"text": e.text, "permanent": bool(e.permanent), "strength": st, "loss": loss, "weight": w})
        finally:
            plastic.enabled = was
        tot = sum(max(r["weight"], 1e-3) for r in rows) or 1.0
        for r in rows:
            r["share"] = max(r["weight"], 1e-3) / tot
        self.last_replay_weights = rows
        return [(f"{e.text}\n\n", r["weight"]) for e, r in zip(night_eps, rows)]

    def _night_budget(self, plastic):
        """M-scale at the start of a night: n = adapter norm; with a budget b and n > b, B *= b / n. Records
        norm_pre_night and downscale_g in meta. A home that has adapters but no budget takes the current norm as its
        budget and is not scaled that night (logged)."""
        n = plastic.adapter_norm()
        b = self.meta.get("adapter_budget")
        g = 1.0
        if b is None and n > 0.0:
            b = self.meta["adapter_budget"] = n
            if self.sleep_log:
                self.sleep_log(f"M-scale: home has adapters but no budget; budget = current norm {n:.4f}, no downscale tonight")
        elif b is not None and n > b:
            g = b / n
            plastic.downscale(g, self.downscale_factors)
        self.meta["norm_pre_night"] = n
        self.meta["downscale_g"] = g
        if self.sleep_log:
            self.sleep_log(f"M-scale: norm {n:.4f} budget {b} g {g:.4f}")

    def _night_episodes(self) -> list:
        """What a night replays: the stored episodes. The inquiry lesson is never among them (it lives outside the
        list): replayed, the slow synapses learned its made-up word (weekly s102 "the lost city of Zorbit",
        results/2026-11-19-inquiry-weekly-1.5b.md)."""
        return list(self.hippocampus.episodes)

    def _near_miss_guard(self):
        """The near-miss guard texts for tonight, all generated by the pretrained cortex (adapters off, nothing
        reinstated, birth state): per episode its own question, near_miss rewrites, and its reply to each."""
        from brain.near_miss import frame_cues, guard_texts, near_miss_cues
        texts = []
        if self.near_miss_frame in ("F", "FT"):   # the frame guard: the cortex's own "my X" questions, once per night
            cues = frame_cues(self.raw_cortex, self.base, k=self.near_miss, seed=self.sleep_seed + self.meta.get("sleeps", 0))
            kept = [q for q in cues if not (self.near_miss_exclude and self.near_miss_exclude(q))]
            if self.near_miss_frame == "FT":   # M4b: drop cues about a told topic (brain-side, from its own episodes)
                from brain.near_miss import told_topic
                told = [e.text for e in self._night_episodes()]
                kept = [q for q in kept if not told_topic(q, told)]
            t = guard_texts(self.raw_cortex, self.base, kept, context=self._teacher_context())
            self.near_miss_log.append({"episode": None, "own": None, "cues": cues, "kept": kept, "texts": t})
            return t
        # Paired night ("P:<rewrite frame>", 2026-11-17, needs teacher_context): per TOLD episode the cortex's own
        # question about it AND its near-miss rewrites, all answered by the teacher that has read the day's tellings,
        # so the night holds "told detail -> the name" next to "untold detail of the same topic -> no name".
        paired = self.near_miss_frame.startswith("P:")
        frame = self.near_miss_frame[2:] if paired else self.near_miss_frame
        assert not paired or self.teacher_context, "the paired night needs teacher_context=True"
        with torch.random.fork_rng(devices=[]):
            for i, e in enumerate(self.hippocampus.episodes):
                if paired and (e.permanent or not e.text.startswith("User: ")):
                    continue
                own, cues = near_miss_cues(self.raw_cortex, self.base, e.text, k=self.near_miss,
                                           frame=frame, seed=self.sleep_seed * 100 + i)
                if paired:
                    cues = [own] + cues
                kept = [q for q in cues if not (self.near_miss_exclude and self.near_miss_exclude(q))]
                t = guard_texts(self.raw_cortex, self.base, kept, context=self._teacher_context())
                self.near_miss_log.append({"episode": e.text, "own": own, "cues": cues, "kept": kept, "texts": t})
                texts += t
        return texts

    TEACHER_OPEN, TEACHER_ACK = "A few things about me.", "Thanks for telling me. I'll keep that in mind."

    def _teacher_context(self):
        """The teacher night's context: what the user told (every non-permanent told episode's user words, in storage
        order) in ONE user turn and one acknowledgement. Layout measured in results/2026-11-17-incontext-abstain.md:
        one turn names 22/24 told rewordings and invents on 19/48 never-told, the episodes replayed turn by turn 17/24
        and 22/48 with persona leaks. Declared scaffold (brain/SCAFFOLDS.md): the opening and acknowledgement words,
        read by the teacher only and never a training target. None when the switch is off (guard_texts unchanged)."""
        if not self.teacher_context:
            return None
        said = [e.text.removeprefix("User: ").split("\n\nAssistant:")[0].strip()
                for e in self._night_episodes() if not e.permanent and e.text.startswith("User: ")]
        if not said:
            return None
        return f"User: {self.TEACHER_OPEN} {' '.join(said)}\n\nAssistant: {self.TEACHER_ACK}\n\n"

    def close(self):
        """Shut down: sleep once if anything was lived since the last sleep (and sleep_on_restart), then save and
        release the home lock."""
        if self.sleep_on_restart and self.pending():
            self.sleep()
        self.save()
        self.release()

    # ---- one process per home ----
    def _acquire_lock(self):
        fh = open(os.path.join(self.home, LOCK), "a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            raise BrainBusy(f"{self.home} is in use by another process (chat or night)") from None
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        self._lock_fh = fh

    def release(self):
        if self._lock_fh is not None:
            fcntl.flock(self._lock_fh, fcntl.LOCK_UN)
            self._lock_fh.close()
            self._lock_fh = None

    # ---- memory accounting ----
    def memory_report(self) -> dict:
        """Bytes held by the hippocampus (by device) and by the consolidated synapses, plus what is on disk."""
        eps = self.hippocampus.episodes
        by_dev = self.hippocampus.nbytes()
        total = sum(by_dev.values())
        p = self.cortex.plastic
        ad = 0 if p is None else sum(v.numel() * v.element_size() for v in p.parameters())
        disk = 0
        for root, _, files in os.walk(self.home):
            disk += sum(os.path.getsize(os.path.join(root, f)) for f in files)
        return {"cortex_model": self.cortex_model, "episodes": len(eps), "episode_bytes": by_dev,
                "bytes_per_episode": total // len(eps) if eps else 0, "adapter_bytes": ad, "home_disk_bytes": disk}

    # ---- persistence ----
    # Layout (adopt-1.5b plan step 5): <home>/episodes/<n>.pt holds one episode's tensors and is written once (again
    # only if its key object was replaced); <home>/episodes.json holds every episode's text and the scalars that change
    # with use; adapters are written only after a night changed them. Homes in the older single-file layout
    # (memory.pt) load unchanged and are written in the new layout at their next save.
    _SCALARS = ("text", "strength", "created", "last_used", "uses", "permanent", "source")
    _legacy_key = None

    def save(self):
        d = os.path.join(self.home, EPISODES)
        os.makedirs(d, exist_ok=True)
        index = []
        # The counter lives in meta.json, written after episodes.json: a save cut between the two leaves it behind
        # the index, so never hand out a name a loaded episode or a file on disk already has.
        used = {getattr(e, "_file", None) for e in self.hippocampus.episodes} | set(os.listdir(d))
        for e in self.hippocampus.episodes:
            f = getattr(e, "_file", None)
            if f is None:
                n = self.meta.get("episode_files", 0)
                while f"{n:07d}.pt" in used:
                    n += 1
                f = e._file = f"{n:07d}.pt"
                used.add(f)
                self.meta["episode_files"] = n + 1
            if getattr(e, "_saved_key", None) is not e.key or not os.path.exists(os.path.join(d, f)):
                blob = {"delta": e.delta, "key": e.key}
                if getattr(e, "activity", None) is not None:
                    blob["activity"] = e.activity
                if getattr(e, "route_delta", None) is not None:
                    blob["route_delta"] = e.route_delta
                torch.save(blob, os.path.join(d, f + ".tmp"))
                os.replace(os.path.join(d, f + ".tmp"), os.path.join(d, f))
                e._saved_key = e.key   # the object written (identity, so a replaced key is rewritten)
            index.append({"file": f, **{k: getattr(e, k) for k in self._SCALARS}})
        self._write_json(EPISODE_INDEX, index)
        if self._inquiry_ep is not None and (getattr(self, "_inquiry_dirty", False)
                                             or not os.path.exists(os.path.join(self.home, INQUIRY_FILE))):
            e = self._inquiry_ep
            tmp = os.path.join(self.home, INQUIRY_FILE + ".tmp")
            torch.save({"delta": e.delta, "key": e.key, **{k: getattr(e, k) for k in self._SCALARS}}, tmp)
            os.replace(tmp, os.path.join(self.home, INQUIRY_FILE))
            self._inquiry_dirty = False
        meta = dict(self.meta)
        if self.key != "S0":  # an S0 home is written without a key entry
            meta["key"] = self.key
        else:
            meta.pop("key", None)
        if self._use_gate is not None:
            meta["use_gate"] = self._use_gate
        else:
            meta.pop("use_gate", None)
        if self._replay_priority is not None:
            meta["replay_priority"] = self._replay_priority
        else:
            meta.pop("replay_priority", None)
        self._write_json("meta.json", meta)
        p = self.cortex.plastic
        if p is not None and (self._adapters_dirty or not os.path.exists(os.path.join(self.home, ADAPTERS))):
            tmp = os.path.join(self.home, ADAPTERS + ".tmp")
            torch.save({"rank": self.adapter_rank, "layers": p.layers,
                        "adapters": {k: v.detach().cpu() for k, v in p.state_dict().items()}}, tmp)
            os.replace(tmp, os.path.join(self.home, ADAPTERS))
            self._adapters_dirty = False

    _adapters_dirty = False

    def _write_json(self, name, obj):
        tmp = os.path.join(self.home, name + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=1)
        os.replace(tmp, os.path.join(self.home, name))

    def _load_episodes(self):
        idx = os.path.join(self.home, EPISODE_INDEX)
        if os.path.exists(idx):
            with open(idx) as fh:
                index = json.load(fh)
            eps = []
            for row in index:
                blob = torch.load(os.path.join(self.home, EPISODES, row["file"]), map_location="cpu",
                                  weights_only=False)
                e = Episode(delta=blob["delta"], key=blob["key"], **{k: row[k] for k in self._SCALARS if k in row})  # "source" absent in older homes -> default
                if "activity" in blob:
                    e.activity = blob["activity"]
                if "route_delta" in blob:
                    e.route_delta = blob["route_delta"]
                e._file, e._saved_key = row["file"], e.key
                eps.append(e)
            return eps
        p = os.path.join(self.home, "memory.pt")
        if os.path.exists(p):  # older single-file layout
            blob = torch.load(p, map_location="cpu", weights_only=False)
            self._legacy_key = blob.get("key")   # a non-S0 key the old home stored (now kept in meta.json)
            return [Episode(**e) for e in blob["episodes"]]
        return None

    def _check_cortex(self, saved: str | None):
        """Refuse a home made with another cortex: by recorded name, and (homes older than the record) by the shape
        of the stored traces against this cortex's state."""
        if saved is not None and self.cortex_model is not None and saved != self.cortex_model:
            raise CortexMismatch(f"{self.home} was made with cortex {saved}, not {self.cortex_model}; its episodes "
                                 f"and adapters do not transfer (migrate it: adopt-1.5b plan step 10)")
        for e in self.hippocampus.episodes:
            shapes = [None if x is None else tuple(x.shape) for x in e.delta]
            want = [tuple(b.shape) if i % 3 == 1 else None for i, b in enumerate(self.base)]
            if shapes != want:
                raise CortexMismatch(f"{self.home}: stored traces have shapes of another cortex than "
                                     f"{self.cortex_model or 'this one'}")

    def load(self) -> bool:
        if not os.path.exists(os.path.join(self.home, "meta.json")):
            return False
        eps = self._load_episodes()
        if eps is None:
            return False
        with open(os.path.join(self.home, "meta.json")) as fh:
            meta = json.load(fh)
        key = self._key_arg or meta.pop("key", None) or self._legacy_key or "S0"
        meta.pop("key", None)
        ug = self._use_gate_arg or meta.pop("use_gate", None)
        meta.pop("use_gate", None)
        if ug not in (None, "raw_win"):
            raise ValueError(f"use_gate in {self.home}/meta.json must be 'raw_win', not {ug!r}")
        self._use_gate = ug
        rp = self._replay_priority_arg or meta.pop("replay_priority", None)
        meta.pop("replay_priority", None)
        if rp not in (None, "strength_x_loss", "loss"):
            raise ValueError(f"replay_priority in {self.home}/meta.json must be 'strength_x_loss' or 'loss', not {rp!r}")
        self._replay_priority = rp
        if key != self.key:
            enabled = self.hippocampus.enabled
            self.hippocampus = self._make_hippocampus(key)
            self.hippocampus.enabled = enabled
        self.hippocampus.episodes = eps
        if ug is not None:
            self.hippocampus.USE_GATE = ug
        q = os.path.join(self.home, INQUIRY_FILE)
        if os.path.exists(q):
            blob = torch.load(q, weights_only=False)
            self._inquiry_ep = Episode(delta=blob["delta"], key=blob["key"],
                                       **{k: blob[k] for k in self._SCALARS if k in blob})
        if key == "S0":  # a store written under a delta-read key holds no S0 key: rebuild it from the stored delta
            for e in eps:
                if e.key is None:
                    e.key = self.raw_cortex.key_of([None if d is None else d.float() for d in e.delta])
        else:  # a delta-read key keeps no separate key per episode
            for e in eps:
                e.key = None
        try:
            self._check_cortex(meta.get("cortex_model"))
        except CortexMismatch:
            self.hippocampus.episodes = []
            self.release()
            raise
        # A restart is a new day: working state starts from the base (plus, later, what sleep consolidated); the
        # previous day's live state is NOT restored (that would be working memory surviving sleep).
        self.meta = meta
        if self.meta.get("cortex_model") is None:
            self.meta["cortex_model"] = self.cortex_model
        a = os.path.join(self.home, ADAPTERS)
        if os.path.exists(a):  # what earlier nights consolidated
            blob = torch.load(a, weights_only=False)
            self.adapter_rank = blob["rank"]
            p = self._plastic()
            dev = p.z["emb.weight"].device
            p.load_state_dict({k: v.to(dev) for k, v in blob["adapters"].items()})
            self.cortex.refresh()
        return True
