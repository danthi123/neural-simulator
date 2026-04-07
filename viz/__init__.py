# viz/ package - OpenGL visualization components
# Re-exports key functions for convenient access

from viz.renderer import (
    init_gl, update_gl_data, render_scene_gl, fast_vbo_update,
    get_color_for_trait, apply_neuron_filters_to_indices,
    apply_synapse_filters_to_indices, trigger_filter_update_signal,
    get_current_filter_settings_from_gui,
)
from viz.camera import (
    mouse_button_func_gl, mouse_motion_func_gl, keyboard_func_gl,
    reshape_gl_window,
)
from viz.overlays import render_text_gl

__all__ = [
    'init_gl', 'update_gl_data', 'render_scene_gl', 'fast_vbo_update',
    'get_color_for_trait', 'apply_neuron_filters_to_indices',
    'apply_synapse_filters_to_indices', 'trigger_filter_update_signal',
    'get_current_filter_settings_from_gui',
    'mouse_button_func_gl', 'mouse_motion_func_gl', 'keyboard_func_gl',
    'reshape_gl_window',
    'render_text_gl',
]
