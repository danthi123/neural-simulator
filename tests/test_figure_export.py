"""Tests for publication figure export."""
import pytest
import os
import numpy as np

try:
    import matplotlib
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
class TestFigureExport:
    def test_sweep_figure(self, tmp_path):
        from ui.figure_export import export_sweep_figure
        results = [
            {"param_value": 0.005, "delta_hz": 3.0, "t_statistic": 2.5, "p_significant": True},
            {"param_value": 0.010, "delta_hz": 6.0, "t_statistic": 5.0, "p_significant": True},
            {"param_value": 0.020, "delta_hz": 10.0, "t_statistic": 8.0, "p_significant": True},
        ]
        path = str(tmp_path / "test_sweep.png")
        result = export_sweep_figure(results, "stdp_a_plus", filepath=path)
        assert result == path
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000  # Not empty

    def test_sweep_figure_with_errors(self, tmp_path):
        from ui.figure_export import export_sweep_figure
        results = [
            {"param_value": 0.005, "delta_hz": 3.0, "p_significant": False},
            {"param_value": 0.010, "error": "init failed"},
            {"param_value": 0.020, "delta_hz": 10.0, "p_significant": True},
        ]
        path = str(tmp_path / "test_sweep_err.png")
        result = export_sweep_figure(results, "stdp_a_plus", filepath=path)
        assert result == path
        assert os.path.exists(path)

    def test_sweep_figure_t_statistic_metric(self, tmp_path):
        from ui.figure_export import export_sweep_figure
        results = [
            {"param_value": 0.005, "t_statistic": 2.5, "p_significant": True},
            {"param_value": 0.010, "t_statistic": 5.0, "p_significant": True},
        ]
        path = str(tmp_path / "test_sweep_tstat.png")
        result = export_sweep_figure(results, "stdp_a_plus", metric="t_statistic", filepath=path)
        assert result == path
        assert os.path.exists(path)

    def test_comparison_figure(self, tmp_path):
        from ui.figure_export import export_experiment_comparison
        pre = [5.0, 6.0, 4.5, 5.5, 6.5]
        post = [12.0, 14.0, 11.0, 13.0, 15.0]
        path = str(tmp_path / "test_compare.png")
        result = export_experiment_comparison(pre, post, filepath=path)
        assert result == path
        assert os.path.exists(path)

    def test_comparison_figure_no_significance(self, tmp_path):
        from ui.figure_export import export_experiment_comparison
        pre = [5.0, 6.0, 4.5, 5.5, 6.5]
        post = [5.1, 5.9, 4.6, 5.4, 6.6]  # Very similar
        path = str(tmp_path / "test_compare_nosig.png")
        result = export_experiment_comparison(pre, post, filepath=path)
        assert result == path
        assert os.path.exists(path)

    def test_freq_response_figure(self, tmp_path):
        from ui.figure_export import export_frequency_response
        data = [{"freq_hz": f, "net_delta": np.sin(f/20)*2}
                for f in [1, 5, 10, 20, 40, 80]]
        path = str(tmp_path / "test_freq.png")
        result = export_frequency_response(data, filepath=path)
        assert result == path
        assert os.path.exists(path)
