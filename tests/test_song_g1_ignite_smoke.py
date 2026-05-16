def test_ignite_module_exposes_expected_api():
    import research.runners.song_g1_ignite as ig
    for fn in ("load_members", "ignite_sequence", "self_comprehend"):
        assert hasattr(ig, fn), fn
