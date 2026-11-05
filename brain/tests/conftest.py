"""Run brain/tests without the model stack.

`brain.cortex` imports `rwkv` at module scope, but these unit tests need no weights (mock `Cortex`). Where the real
package is absent (the lane's CPU environment), stub exactly the two names the cortex imports; where it is present
(W, the model checkout) the real package is used. Test files stay unchanged either way.
"""
import importlib.util
import sys
import types

if importlib.util.find_spec("rwkv") is None:
    _rwkv = types.ModuleType("rwkv")
    _model = types.ModuleType("rwkv.model")
    _utils = types.ModuleType("rwkv.utils")

    class RWKV:  # import stub only: constructing it must never happen in a weight-free test
        def __init__(self, *a, **k):
            raise RuntimeError("rwkv stub: brain/tests run without model weights")

    class PIPELINE:  # ditto
        def __init__(self, *a, **k):
            raise RuntimeError("rwkv stub: brain/tests run without model weights")

    _model.RWKV = RWKV
    _utils.PIPELINE = PIPELINE
    _rwkv.model = _model
    _rwkv.utils = _utils
    sys.modules.update({"rwkv": _rwkv, "rwkv.model": _model, "rwkv.utils": _utils})