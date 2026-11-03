"""The brain: cortex + hippocampus + self + sleep, one continuous life saved to a directory.

The cortex the brain speaks with is the pretrained network plus the slow synapses sleep changed (LiveCortex). Every
restart is a night: a brain shut down or started with episodes stored since its last sleep replays them into those
synapses once (brain/sleep.py, unchanged), and the synapses are saved next to the hippocampus, so learning survives.
With no consolidated synapses the live cortex IS the pretrained one (same code path, byte-equal output).
"""
from __future__ import annotations

import json
import os
import time

import torch
import torch.nn.functional as F

from brain.cortex import Cortex
from brain.hippocampus import Episode, Hippocampus
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
                 sleep_on_restart: bool = True, sleep_steps: int = 240, sleep_lr: float = 5e-4,
                 sleep_kl_weight: float = 1.0, adapter_rank: int = 8, sleep_log=None, sleep_seed: int = 0,
                 abstain: bool = False):
        self.home = home
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
        os.makedirs(home, exist_ok=True)
        self.raw_cortex = cortex or Cortex(device=device)
        self.cortex = LiveCortex(self.raw_cortex)
        self.hippocampus = Hippocampus()
        self.sleep_on_restart = sleep_on_restart
        self.sleep_cfg = {"steps": sleep_steps, "lr": sleep_lr, "kl_weight": sleep_kl_weight}
        self.adapter_rank = adapter_rank
        self.sleep_log = sleep_log
        self.sleep_seed = sleep_seed  # night n replays with seed sleep_seed + n; adapters init from sleep_seed
        # Birth context is read by the pretrained cortex: stored episodes are deltas from this base, so it must not
        # move when sleep changes the slow synapses.
        _, self.base = self.raw_cortex.read(BIRTH, None)
        self.state = [x.clone() for x in self.base]
        self.meta = {"name": name, "friend": friend, "born": time.time(), "turns": 0, "sleeps": 0, "last_sleep": 0.0}
        if not self.load():
            self._grow_self()
            self.meta["last_sleep"] = time.time()  # birth is not a lived day: the first night comes at a restart
            self.save()
        elif self.sleep_on_restart and self.pending():
            self.sleep()  # restart = a night: consolidate what was lived since the last sleep, once
            self.save()

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
        self.cortex.refresh()
        return losses

    def close(self):
        """Shut down: sleep once if anything was lived since the last sleep, then save."""
        if self.sleep_on_restart and self.pending():
            self.sleep()
        self.save()

    # ---- persistence ----
    def save(self):
        eps = [{k: getattr(e, k) for k in ("text", "delta", "key", "strength", "created", "last_used", "uses",
                                             "permanent")} for e in self.hippocampus.episodes]
        torch.save({"episodes": eps, "state": self.state}, os.path.join(self.home, "memory.pt"))
        with open(os.path.join(self.home, "meta.json"), "w") as fh:
            json.dump(self.meta, fh, indent=1)
        p = self.cortex.plastic
        if p is not None:
            tmp = os.path.join(self.home, ADAPTERS + ".tmp")
            torch.save({"rank": self.adapter_rank, "layers": p.layers,
                        "adapters": {k: v.detach().cpu() for k, v in p.state_dict().items()}}, tmp)
            os.replace(tmp, os.path.join(self.home, ADAPTERS))

    def load(self) -> bool:
        p = os.path.join(self.home, "memory.pt")
        if not os.path.exists(p):
            return False
        blob = torch.load(p, weights_only=False)
        self.hippocampus.episodes = [Episode(**e) for e in blob["episodes"]]
        # A restart is a new day: working state starts from the base (plus, later, what sleep consolidated); the
        # previous day's live state is NOT restored (that would be working memory surviving sleep).
        with open(os.path.join(self.home, "meta.json")) as fh:
            self.meta = json.load(fh)
        a = os.path.join(self.home, ADAPTERS)
        if os.path.exists(a):  # what earlier nights consolidated
            blob = torch.load(a, weights_only=False)
            self.adapter_rank = blob["rank"]
            p = self._plastic()
            dev = p.z["emb.weight"].device
            p.load_state_dict({k: v.to(dev) for k, v in blob["adapters"].items()})
            self.cortex.refresh()
        return True
