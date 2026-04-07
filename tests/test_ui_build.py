"""
Test suite for validating DearPyGUI UI structure of the neural simulator.

This test file validates the UI build by parsing the source code as text,
avoiding the need to import CuPy or CUDA modules. It checks:
- Tooltip completeness
- Configuration read/write wiring
- UI control counts in specific sections
- Critical compatibility fixes
- VBO cleanup code
"""

import pytest
import re
from pathlib import Path


# Paths to all source files that may contain UI code after the refactor
_PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_PATHS = [
    _PROJECT_ROOT / "neural-simulator.py",
    _PROJECT_ROOT / "ui" / "layout.py",
    _PROJECT_ROOT / "ui" / "callbacks.py",
    _PROJECT_ROOT / "viz" / "renderer.py",
    _PROJECT_ROOT / "sim" / "bridge.py",
    _PROJECT_ROOT / "sim" / "config.py",
]

# Keep for backward-compat reference in test_simulator_file_exists
SIMULATOR_PATH = _PROJECT_ROOT / "neural-simulator.py"


def _read_all_sources() -> str:
    """Concatenate text of all source files that exist."""
    content = ""
    for p in SOURCE_PATHS:
        if p.exists():
            content += p.read_text(encoding='utf-8') + "\n"
    return content


class TestUIBuild:
    """Test suite for DearPyGUI UI structure validation."""

    @pytest.fixture(scope="class")
    def source_code(self):
        """Load and cache all simulator source code."""
        return _read_all_sources()

    def test_simulator_file_exists(self):
        """Verify at least the main neural-simulator.py file exists."""
        assert SIMULATOR_PATH.exists(), f"Neural simulator not found at {SIMULATOR_PATH}"

    def test_tooltip_count_is_88(self, source_code):
        """Verify exactly 88 tooltip= occurrences in the source code."""
        tooltip_count = len(re.findall(r'tooltip=', source_code))
        assert tooltip_count == 88, (
            f"Expected 88 tooltip= occurrences, found {tooltip_count}. "
            "This likely indicates UI controls were added/removed without updating tooltips."
        )

    def test_add_parameter_table_row_calls_have_tooltips(self, source_code):
        """Verify add_parameter_table_row calls include tooltip= arguments where appropriate."""
        # Verify tooltip= appears 82 times (already tested separately)
        # This test verifies the pattern is used broadly
        tooltip_count = len(re.findall(r'tooltip=', source_code))

        # Most UI controls should have tooltips
        assert tooltip_count >= 70, (
            f"Found only {tooltip_count} tooltip= occurrences. "
            f"Expected most UI controls (>70) to have helpful tooltips."
        )

    def test_tooltip_dictionaries_exist(self, source_code):
        """Verify tooltip dictionaries for all neuron models exist."""
        required_dicts = [
            '_izh_tooltips',
            '_hh_tooltips',
            '_adex_tooltips'
        ]

        for dict_name in required_dicts:
            assert dict_name in source_code, (
                f"Required tooltip dictionary '{dict_name}' not found in source. "
                f"This is needed to define help text for neuron model parameters."
            )

            # Verify it's defined as an assignment (not just referenced)
            pattern = rf'{dict_name}\s*=\s*\{{'
            assert re.search(pattern, source_code), (
                f"Tooltip dictionary '{dict_name}' found but not properly defined. "
                f"Expected pattern: '{dict_name} = {{...}}'"
            )

    def test_nmda_controls_exist_and_wired(self, source_code):
        """Verify NMDA section has exactly 5 controls with proper read/write wiring."""
        nmda_controls = [
            'cfg_enable_nmda',
            'cfg_nmda_ratio',
            'cfg_nmda_tau_decay',
            'cfg_nmda_tau_rise',
            'cfg_nmda_mg_conc'
        ]

        # Verify all controls exist in the UI (add_parameter_table_row calls)
        for control in nmda_controls:
            assert control in source_code, (
                f"NMDA control '{control}' not found in source. "
                f"Expected NMDA section to have: {nmda_controls}"
            )

        # Verify read wiring in _update_sim_config_from_ui (cfg_dict_from_ui reads)
        for control in nmda_controls:
            pattern = rf'dpg\.get_value\("{control}"\)'
            assert re.search(pattern, source_code), (
                f"NMDA control '{control}' is missing read wiring. "
                f"Expected dpg.get_value(\"{control}\") in config read function."
            )

        # Verify write wiring in _populate_ui_from_config_dict (dpg.set_value calls)
        for control in nmda_controls:
            pattern = rf'dpg\.set_value\("{control}"'
            assert re.search(pattern, source_code), (
                f"NMDA control '{control}' is missing write wiring. "
                f"Expected dpg.set_value(\"{control}\", ...) in config write function."
            )

    def test_synaptic_scaling_controls_exist_and_wired(self, source_code):
        """Verify Synaptic Scaling section has exactly 2 controls with proper wiring."""
        scaling_controls = [
            'cfg_enable_synaptic_scaling',
            'cfg_synaptic_scaling_rate'
        ]

        # Verify all controls exist
        for control in scaling_controls:
            assert control in source_code, (
                f"Synaptic Scaling control '{control}' not found in source. "
                f"Expected Synaptic Scaling section to have: {scaling_controls}"
            )

        # Verify read wiring
        for control in scaling_controls:
            pattern = rf'dpg\.get_value\("{control}"\)'
            assert re.search(pattern, source_code), (
                f"Synaptic Scaling control '{control}' is missing read wiring."
            )

        # Verify write wiring
        for control in scaling_controls:
            pattern = rf'dpg\.set_value\("{control}"'
            assert re.search(pattern, source_code), (
                f"Synaptic Scaling control '{control}' is missing write wiring."
            )

    def test_vbo_cleanup_code_exists(self, source_code):
        """Verify VBO cleanup code with glDeleteBuffers exists."""
        # Should have import
        assert 'from OpenGL.GL import glDeleteBuffers' in source_code, (
            "glDeleteBuffers import not found. "
            "This is needed for proper cleanup of OpenGL vertex buffer objects."
        )

        # Should have usage
        assert 'glDeleteBuffers' in source_code, (
            "glDeleteBuffers function call not found. "
            "VBO cleanup code is missing."
        )

        # Verify it's called with proper arguments
        pattern = r'glDeleteBuffers\s*\(\s*len\s*\(\s*valid_vbos\s*\)\s*,\s*valid_vbos\s*\)'
        assert re.search(pattern, source_code), (
            "glDeleteBuffers is not being called with correct arguments. "
            "Expected: glDeleteBuffers(len(valid_vbos), valid_vbos)"
        )

    def test_cupy_sparse_import_compatibility_fix(self, source_code):
        """Verify CuPy sparse import compatibility fix exists."""
        # Should have try/except wrapper
        pattern = (
            r'try:\s*'
            r'import\s+cupy\.sparse\s+as\s+csp\s*'
            r'except\s+\(\s*ImportError\s*,\s*ModuleNotFoundError\s*\):\s*'
            r'import\s+cupyx\.scipy\.sparse\s+as\s+csp'
        )

        assert re.search(pattern, source_code, re.DOTALL), (
            "CuPy sparse import compatibility fix not found. "
            "Expected try/except to handle both cupy.sparse and cupyx.scipy.sparse imports."
        )

    def test_config_read_function_exists(self, source_code):
        """Verify the configuration read function exists and is properly structured."""
        # Look for _update_sim_config_from_ui function
        assert 'def _update_sim_config_from_ui' in source_code, (
            "_update_sim_config_from_ui function not found."
        )

        # It should have the config dict building pattern
        assert 'cfg_dict_from_ui' in source_code, (
            "Configuration dictionary reading pattern 'cfg_dict_from_ui' not found."
        )

    def test_config_write_function_exists(self, source_code):
        """Verify the configuration write function exists and is properly structured."""
        # Look for _populate_ui_from_config_dict or _set_ui_from_config
        assert '_populate_ui_from_config_dict' in source_code, (
            "Configuration write function '_populate_ui_from_config_dict' not found."
        )

        # Should use dpg.set_value for writes
        assert 'dpg.set_value' in source_code, (
            "UI value setting pattern 'dpg.set_value' not found."
        )

    def test_cfg_tags_have_read_wiring(self, source_code):
        """Verify critical cfg_* input tags are wired for reading."""
        # Test a sample of critical controls rather than all tags
        critical_read_tags = [
            'cfg_num_neurons',
            'cfg_dt_ms',
            'cfg_seed',
            'cfg_neuron_model_type',
            'cfg_enable_nmda',
            'cfg_enable_synaptic_scaling',
        ]

        # For each critical tag, verify it has read wiring (dpg.get_value)
        missing_reads = []
        for tag in critical_read_tags:
            pattern = rf'dpg\.get_value\("{tag}"\)'
            if not re.search(pattern, source_code):
                missing_reads.append(tag)

        assert len(missing_reads) == 0, (
            f"Critical cfg_* tags are missing read wiring: {missing_reads}. "
            f"Every input control needs dpg.get_value() in the config read function."
        )

    def test_cfg_tags_have_write_wiring(self, source_code):
        """Verify critical cfg_* input tags are wired for writing."""
        # Test a sample of critical controls rather than all tags
        critical_write_tags = [
            'cfg_num_neurons',
            'cfg_dt_ms',
            'cfg_seed',
            'cfg_neuron_model_type',
            'cfg_enable_nmda',
            'cfg_enable_synaptic_scaling',
        ]

        # For each critical tag, verify it has write wiring (dpg.set_value)
        missing_writes = []
        for tag in critical_write_tags:
            pattern = rf'dpg\.set_value\("{tag}"'
            if not re.search(pattern, source_code):
                missing_writes.append(tag)

        assert len(missing_writes) == 0, (
            f"Critical cfg_* tags are missing write wiring: {missing_writes}. "
            f"Every input control needs dpg.set_value() in the config write function."
        )

    def test_ui_section_is_valid_dearpygui(self, source_code):
        """Verify the UI section uses valid DearPyGUI constructs."""
        required_patterns = [
            r'import\s+dearpygui\.dearpygui\s+as\s+dpg',  # Import statement
            r'dpg\.create_context\(\)',  # Context creation
            r'with\s+dpg\.window\(',  # Window creation
            r'dpg\.show_viewport\(\)',  # Viewport showing
            r'dpg\.set_primary_window\(',  # Primary window setting
        ]

        for pattern in required_patterns:
            assert re.search(pattern, source_code), (
                f"Required DearPyGUI pattern not found: {pattern}. "
                f"The UI section may not be properly initialized."
            )

    def test_no_hardcoded_magic_numbers_in_ui_tooltips(self, source_code):
        """Verify tooltip text is properly formatted and readable."""
        # Find all tooltip strings
        tooltip_matches = re.findall(r'tooltip="([^"]+)"', source_code)

        assert len(tooltip_matches) > 0, (
            "No tooltip strings found. Expected to find tooltip= assignments."
        )

        # Check that tooltips are not empty
        empty_tooltips = [t for t in tooltip_matches if not t.strip()]
        assert len(empty_tooltips) == 0, (
            f"Found {len(empty_tooltips)} empty tooltip strings. "
            f"All tooltips should have meaningful help text."
        )

        # Check that tooltips have reasonable length (not too short)
        short_tooltips = [t for t in tooltip_matches if len(t) < 10]
        assert len(short_tooltips) == 0, (
            f"Found {len(short_tooltips)} tooltips that are too short (< 10 chars). "
            f"Tooltips should provide meaningful help."
        )

    def test_dearpygui_callbacks_are_defined(self, source_code):
        """Verify that critical DearPyGUI callbacks are defined."""
        # Check for critical callbacks that should exist
        critical_callbacks = [
            'handle_start_simulation_event',
            'handle_pause_simulation_event',
            'handle_stop_simulation_event',
            '_update_sim_config_from_ui_and_signal_reset_needed',
        ]

        # Check that each callback is defined as a function
        undefined_callbacks = []
        for callback in critical_callbacks:
            if not re.search(rf'def\s+{callback}\s*\(', source_code):
                undefined_callbacks.append(callback)

        assert len(undefined_callbacks) == 0, (
            f"Found {len(undefined_callbacks)} undefined critical callbacks: {undefined_callbacks}. "
            f"All referenced callbacks must be defined as functions."
        )

    def test_table_row_helper_function_exists(self, source_code):
        """Verify the add_parameter_table_row helper function exists."""
        assert 'def add_parameter_table_row' in source_code, (
            "add_parameter_table_row helper function not found. "
            "This function should simplify adding rows to parameter tables."
        )

    def test_no_duplicate_cfg_tags(self, source_code):
        """Verify that cfg_* tags are not duplicated in the UI."""
        # Find all cfg_* tag definitions (not references)
        tag_patterns = re.findall(r'tag="(cfg_[a-z_0-9]+)"', source_code)

        seen = set()
        duplicates = []
        for tag in tag_patterns:
            if tag in seen:
                duplicates.append(tag)
            seen.add(tag)

        assert len(duplicates) == 0, (
            f"Found duplicate cfg_* tags: {duplicates}. "
            f"Each control must have a unique tag to avoid UI conflicts."
        )

    def test_critical_ui_sections_present(self, source_code):
        """Verify all critical UI sections are present."""
        critical_sections = [
            'Core Simulation Parameters',
            'Izhikevich 2007 Model Parameters',
            'Hodgkin-Huxley Model Parameters',
            '--- AdEx Model Parameters ---',
            'Enable NMDA',
            'Synaptic Plasticity',
        ]

        for section in critical_sections:
            assert section in source_code, (
                f"Critical UI section '{section}' not found. "
                f"This section is essential for the simulator's functionality."
            )


class TestUIDocumentation:
    """Test that UI elements are properly documented."""

    @pytest.fixture(scope="class")
    def source_code(self):
        """Load all simulator source code."""
        return _read_all_sources()

    def test_tooltip_references_literature(self, source_code):
        """Verify tooltips include scientific literature references where appropriate."""
        # Find tooltip content
        tooltips = re.findall(r'tooltip="([^"]*)"', source_code)

        # At least some should have references (year format like "2007", "1990", etc.)
        with_refs = [t for t in tooltips if re.search(r'\b(19|20)\d{2}\b', t)]

        assert len(with_refs) > 0, (
            "No tooltips with year references found. "
            "Biophysical parameters should cite literature sources."
        )

    def test_tooltip_units_specified(self, source_code):
        """Verify tooltips include units or helpful descriptions."""
        # Look for units in tooltip text at all
        tooltips = re.findall(r'tooltip="([^"]+)"', source_code)

        # Check for common unit/description patterns
        with_units = [t for t in tooltips
                     if re.search(r'(ms|mV|pA|nS|pF|mM|Hz|cycles|range|value|time)', t, re.IGNORECASE)]

        # Should have many tooltips with descriptive content
        assert len(with_units) >= 50, (
            f"Only {len(with_units)} tooltips contain descriptive unit/value information. "
            f"Parameters should have helpful descriptions."
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
