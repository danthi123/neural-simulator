def test_ignite_module_exposes_expected_api():
    import research.runners.song_g1_ignite as ig
    for fn in ("load_members", "ignite_sequence", "self_comprehend"):
        assert hasattr(ig, fn), fn


def test_ignite_module_exposes_trajectory_decode():
    # G1.5 Task 1: write-only ordered trajectory readout (additive).
    import research.runners.song_g1_ignite as ig
    assert hasattr(ig, "ignite_and_trajectory_decode"), (
        "ignite_and_trajectory_decode")
    assert callable(ig.ignite_and_trajectory_decode)


def test_ignite_module_exposes_prediction_ignition():
    # Generator-P Task 7a: write-only top-down-prediction ignition
    # alias (additive; the ONLY P write into concept pools).
    import research.runners.song_g1_ignite as ig
    assert hasattr(ig, "ignite_prediction"), "ignite_prediction"
    assert callable(ig.ignite_prediction)
