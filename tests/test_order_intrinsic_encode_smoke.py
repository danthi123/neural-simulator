"""Import / signature smoke for the order-intrinsic encode module.

Per the implementation plan (Task 5): integration is validated by the
Task 6 no-harm check + the Task 7 pre-registered multi-seed gate, NOT
by a contrived orchestration unit test here. This smoke only pins that
the module imports cheaply and exposes the 3 expected callables.
"""


def test_encode_module_exposes_expected_api():
    import research.runners.order_intrinsic_encode as oe

    for fn in ("build_order_intrinsic_bridge", "encode_proposition",
               "readback_sweep"):
        assert hasattr(oe, fn), fn
