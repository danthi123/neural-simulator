"""Pure, CPU-testable atomic checkpoint/resume core for training runs.

WHY THIS EXISTS
---------------
A long training run on the GPU must be killable at any instant (the user
frees the GPU to game) and resumable simply by re-running the script. The
mechanism is a per-epoch checkpoint that is written *atomically* so that a
kill mid-write can never leave a corrupt or partial file behind.

KILL-SAFETY GUARANTEE
---------------------
``save_checkpoint`` writes to ``path + ".tmp"`` and then calls
``os.replace(path + ".tmp", path)``. ``os.replace`` is atomic on the same
filesystem: the destination either still points at the previous complete
checkpoint or at the new complete checkpoint -- never a half-written one.
If the process is killed while ``np.savez`` is still writing the ``.tmp``
file, the real checkpoint path is untouched, so the last good epoch is
preserved and resume works. The stale ``.tmp`` is harmless (overwritten
on the next save).

PURITY
------
numpy-only. No torch, no cupy imported here. Callers that hold device
arrays (e.g. cupy) must transfer them to host/numpy before calling
``save_checkpoint`` -- ``np.asarray`` here only normalizes already-host
array-likes.
"""

import json
import os

import numpy as np

__all__ = ["save_checkpoint", "load_checkpoint", "resume_epoch"]


def save_checkpoint(path, epoch, weights, rng_state, loss_history):
    """Atomically persist a training checkpoint.

    Parameters
    ----------
    path : str
        Destination ``.npz`` path.
    epoch : int
        The epoch that was just completed.
    weights : list of array-like
        Per-layer weights. Each is converted with ``np.asarray`` (host
        arrays only -- do NOT pass cupy arrays; convert to host first).
    rng_state : dict
        A numpy bit-generator state dict (``rng.bit_generator.state``).
        JSON-serialized into the archive because ``np.savez`` cannot
        store a Python dict directly. numpy's default_rng (PCG64) state
        is JSON-safe (nested dict/str/int only), so the round-trip
        compares ``==`` equal.
    loss_history : sequence of float
        Loss values so far.

    The write is atomic: data goes to ``path + ".tmp"`` first, then
    ``os.replace`` swaps it into place. A kill mid-write therefore
    cannot corrupt the existing checkpoint (see module docstring).
    """
    arrs = [np.asarray(w) for w in weights]

    payload = {
        "epoch": np.asarray(int(epoch)),
        "n_weights": np.asarray(len(arrs)),
        # rng state dict -> JSON string (np.savez can't store a dict).
        "rng_state_json": np.asarray(json.dumps(rng_state)),
        "loss_history": np.asarray(loss_history, dtype=np.float64),
    }
    for i, w in enumerate(arrs):
        payload["w%d" % i] = w

    tmp = path + ".tmp"
    # Write the full archive to a temp file first...
    with open(tmp, "wb") as fh:
        np.savez(fh, **payload)
    # ...then atomically swap it into place. os.replace is atomic on the
    # same filesystem, so the real path is never a partial file.
    os.replace(tmp, path)


def load_checkpoint(path):
    """Load a checkpoint, or return ``None`` if it does not exist.

    Returns a dict with plain-Python ``epoch`` (int), ``weights``
    (list of numpy arrays), ``rng_state`` (dict, JSON round-tripped),
    and ``loss_history`` (list of Python floats).
    """
    if not os.path.exists(path):
        return None

    # allow_pickle=False keeps loading restricted to plain arrays only
    # (we never store Python objects -- the rng state is a JSON string).
    with np.load(path, allow_pickle=False) as data:
        epoch = int(data["epoch"])
        n_weights = int(data["n_weights"])
        weights = [data["w%d" % i] for i in range(n_weights)]
        # JSON string array -> str -> dict. numpy may store the scalar
        # string as a 0-d array; str() recovers the text.
        rng_state = json.loads(str(data["rng_state_json"]))
        loss_history = [float(x) for x in data["loss_history"]]

    return {
        "epoch": epoch,
        "weights": weights,
        "rng_state": rng_state,
        "loss_history": loss_history,
    }


def resume_epoch(ckpt):
    """Epoch to start training from.

    ``0`` when there is no checkpoint (fresh run), otherwise one past
    the last completed epoch.
    """
    return 0 if ckpt is None else int(ckpt["epoch"]) + 1
