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
import time

import torch
import torch.nn.functional as F

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
                 key: str | None = None, key_bar: str = "default", gain: str = "default"):
        """cortex_model: checkpoint path or size tag ("0.1b", "1.5b") used when no `cortex` is given; None = the
        default cortex (0.1B). Sleep settings left None come from SLEEP_DEFAULTS for the cortex's size.
        lock: hold <home>/.lock for this object's life (chat and brain.night set it), so two processes never
        live one brain at once; raises BrainBusy if another holds it.
        key: pattern-completion key (switch, default OFF = "S0": the default Hippocampus, store unchanged). "S_all":
        the meaning key read from every layer's stored matrix-memory delta (brain/hippocampus_keys.py) with its own
        dev-fit bar (2026-11-03-meaning-selection-realpath). None = what the home stored, else "S0". A non-S0 choice
        is saved in meta.json; an S0 home is written without a key entry.
        key_bar: which selection bar a non-S0 key uses (switch, default OFF = "default": KEY_BARS, fit at 0.1B).
        "size": the bar refit for this key at this cortex size (KEY_BARS_BY_SIZE; brain/research/
        2026-11-03-hippo-routing-1.5b.md); a size with no row is an error. Not saved in the home: pass it each time.
        gain: reinstatement gain (switch, default OFF = "default": Hippocampus.GAIN, 4.0, fit at 0.1B with the old scorer).
        "size": GAIN_BY_SIZE for this cortex size (1.5B: 0.5, results/2026-11-13-gain-refit-1.5b.md, oracle routing);
        a size with no row is an error. Not saved in the home: pass it each time."""
        self.home = home
        self._key_arg = key
        self.key_bar = key_bar
        self.gain_mode = gain
        os.makedirs(home, exist_ok=True)
        self._lock_fh = None
        if lock:
            self._acquire_lock()
        try:
            self.abstain = abstain   # switch (default OFF): below the feeling-of-knowing bar, answer from "not known"
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
            self.meta = {"name": name, "friend": friend, "born": time.time(), "turns": 0, "sleeps": 0, "last_sleep": 0.0,
                         "cortex_model": self.cortex_model}
            if not self.load():
                self._grow_self()
                self.meta["last_sleep"] = time.time()  # birth is not a lived day: the first night comes at a restart
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

    GAIN_BY_SIZE = {"1.5b": 0.5}

    def _make_hippocampus(self, key: str) -> Hippocampus:
        h = self._hippocampus_for(key)
        if self.gain_mode == "size":
            size = model_size(self.cortex_model)
            if size not in self.GAIN_BY_SIZE:
                raise ValueError(f"no size-fitted reinstatement gain at cortex size {size!r}")
            h.GAIN = self.GAIN_BY_SIZE[size]
        elif self.gain_mode != "default":
            raise ValueError(f"gain must be 'default' or 'size', not {self.gain_mode!r}")
        return h

    def _hippocampus_for(self, key: str) -> Hippocampus:
        if key == "S0":
            return Hippocampus()
        if self.key_bar == "size":
            size = model_size(self.cortex_model)
            if (key, size) not in self.KEY_BARS_BY_SIZE:
                raise ValueError(f"no size-fitted bar for key {key!r} at cortex size {size!r}")
            fam, margin = self.KEY_BARS_BY_SIZE[(key, size)]
        elif self.key_bar != "default":
            raise ValueError(f"key_bar must be 'default' or 'size', not {self.key_bar!r}")
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
                                   permanent=True)

    # ---- one turn ----
    def hear(self, text: str):
        """Process one user turn; returns (reply, info). Memory returns as cortical state, never as text."""
        text = text.strip()
        info = {"recalled": None, "similarity": None, "stored": False}
        known = True
        if self.abstain and not _is_telling(text):
            known, sig = self.knows(text)
            info.update(sig, known=known)
        if known:
            ep, sim = self.hippocampus.recall(self.cortex, self.base, text, question=text.endswith("?"))
        else:  # nothing familiar enough: no pattern completion, no reinstatement
            ep, sim = None, (info["fok"] if info.get("fok_source") == "hippocampus" else info.get("fam", 0.0))
        info["similarity"] = round(sim, 3)
        # Recall is a transient reactivation for THIS reply; it does not accumulate in the ongoing state.
        active = self.state
        if ep is not None:
            active = self.hippocampus.reinstate(self.cortex, self.state, ep)
            info["recalled"] = ep.text
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
            logits, st = self.cortex.read(f"User: {text}\n\nAssistant:", active)
            reply, _ = self.cortex.speak(logits, st)
        finally:
            self.cortex.consolidated_on = retrieval
        _, self.state = self.cortex.read(f"User: {text}\n\nAssistant: {reply}\n\n", self.state)
        if self.hippocampus.enabled and _is_telling(text):
            self.hippocampus.store(self.cortex, self.base, text, speaker="User")
            info["stored"] = True
        self.meta["turns"] += 1
        return reply, info

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
        now = time.time()
        # replay what the hippocampus encoded (speaker line as read at storage time)
        eps = [(f"{e.text}\n\n", self.hippocampus.effective_strength(e, now)) for e in self.hippocampus.episodes]
        eps += list(self.sleep_extra)   # declared scaffold when non-empty (method 2); [] by default
        base =[s.float() for s in self.base]
        with torch.random.fork_rng(devices=[]):
            losses = replay_into_cortex(plastic, base, eps, ORDINARY, steps=steps or self.sleep_cfg["steps"],
                                        lr=self.sleep_cfg["lr"], kl_weight=self.sleep_cfg["kl_weight"],
                                        seed=self.sleep_seed + self.meta.get("sleeps", 0),
                                        log=self.sleep_log)
        for prm in plastic.parameters():
            prm.grad = None
        self.meta["sleeps"] = self.meta.get("sleeps", 0) + 1
        self.meta["last_sleep"] = time.time()
        self._adapters_dirty = True
        self.cortex.refresh()
        return losses

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
                torch.save(blob, os.path.join(d, f + ".tmp"))
                os.replace(os.path.join(d, f + ".tmp"), os.path.join(d, f))
                e._saved_key = e.key   # the object written (identity, so a replaced key is rewritten)
            index.append({"file": f, **{k: getattr(e, k) for k in self._SCALARS}})
        self._write_json(EPISODE_INDEX, index)
        meta = dict(self.meta)
        if self.key != "S0":  # an S0 home is written without a key entry
            meta["key"] = self.key
        else:
            meta.pop("key", None)
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
        if key != self.key:
            enabled = self.hippocampus.enabled
            self.hippocampus = self._make_hippocampus(key)
            self.hippocampus.enabled = enabled
        self.hippocampus.episodes = eps
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
