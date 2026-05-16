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
