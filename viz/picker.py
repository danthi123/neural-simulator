"""GPU color-based neuron picking for the 3D view.

Uses a separate rendering pass where each neuron is drawn with a unique
color encoding its index. Reading back the pixel at the click position
decodes which neuron was hit.
"""
import math
import numpy as np

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False


def index_to_color(index):
    """Encode a neuron index as RGB bytes (supports up to 16M neurons)."""
    r = index & 0xFF
    g = (index >> 8) & 0xFF
    b = (index >> 16) & 0xFF
    return r, g, b


def color_to_index(r, g, b):
    """Decode RGB bytes back to a neuron index."""
    return int(r) | (int(g) << 8) | (int(b) << 16)


def pick_neuron_at(x, y, neuron_positions, num_neurons, viz_cfg):
    """Pick the neuron at screen position (x, y).

    Renders a single off-screen frame with index-encoded colors,
    reads the pixel at the click position, and returns the neuron index.

    Args:
        x, y: Screen coordinates from GLUT (y=0 is top)
        neuron_positions: numpy array (n, 3) of neuron world positions
        num_neurons: total neuron count
        viz_cfg: VisualizationConfig object with camera state

    Returns:
        int: neuron index (0-based), or -1 if no neuron was hit
    """
    if not OPENGL_AVAILABLE or neuron_positions is None or num_neurons == 0:
        return -1

    try:
        viewport = glGetIntegerv(GL_VIEWPORT)
        width = int(viewport[2])
        height = int(viewport[3])
        if width <= 0 or height <= 0:
            return -1

        # Save all GL state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        # Set up clean render state for picking
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glDisable(GL_DITHER)
        glDisable(GL_FOG)
        glDisable(GL_POINT_SMOOTH)

        # Clear with background color that maps to "no neuron"
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set up same projection as main renderer
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        fov = getattr(viz_cfg, 'camera_fov', 60.0)
        near = getattr(viz_cfg, 'camera_near_clip', 0.1)
        far = getattr(viz_cfg, 'camera_far_clip', 10000.0)
        gluPerspective(fov, width / height, near, far)

        # Set up same modelview as main renderer
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        azimuth = viz_cfg.camera_azimuth_angle
        elevation = viz_cfg.camera_elevation_angle
        radius = viz_cfg.camera_radius
        cx = viz_cfg.camera_center_x
        cy = viz_cfg.camera_center_y
        cz = viz_cfg.camera_center_z

        eye_x = cx + radius * math.cos(elevation) * math.sin(azimuth)
        eye_y = cy + radius * math.sin(elevation)
        eye_z = cz + radius * math.cos(elevation) * math.cos(azimuth)

        up_x = getattr(viz_cfg, 'camera_up_x', 0.0)
        up_y = getattr(viz_cfg, 'camera_up_y', 1.0)
        up_z = getattr(viz_cfg, 'camera_up_z', 0.0)

        gluLookAt(eye_x, eye_y, eye_z, cx, cy, cz, up_x, up_y, up_z)

        # Draw neurons as large points with index-encoded colors.
        # Use larger point size for easier picking.
        glPointSize(12.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)

        # Cap to avoid extremely slow immediate-mode rendering
        draw_count = min(num_neurons, len(neuron_positions), 50000)

        # Build color + vertex arrays for batch rendering (much faster than glBegin/glEnd)
        colors = np.zeros((draw_count, 3), dtype=np.uint8)
        for i in range(draw_count):
            r, g, b = index_to_color(i)
            colors[i] = [r, g, b]

        positions_f32 = neuron_positions[:draw_count].astype(np.float32)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, positions_f32)
        glColorPointer(3, GL_UNSIGNED_BYTE, 0, colors)
        glDrawArrays(GL_POINTS, 0, draw_count)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        glFinish()  # Ensure drawing is complete before reading

        # Read pixel at click position (GLUT y=0 is top, GL y=0 is bottom)
        gl_y = height - y - 1
        pixel = glReadPixels(x, gl_y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE)

        # Restore GL state
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glPopAttrib()

        # Decode pixel
        if pixel is None:
            return -1

        # Handle different pixel format returns
        try:
            if hasattr(pixel, 'shape'):
                r_val = int(pixel[0, 0, 0])
                g_val = int(pixel[0, 0, 1])
                b_val = int(pixel[0, 0, 2])
            else:
                r_val = int(pixel[0][0][0])
                g_val = int(pixel[0][0][1])
                b_val = int(pixel[0][0][2])
        except (IndexError, TypeError):
            return -1

        index = color_to_index(r_val, g_val, b_val)

        # White (255, 255, 255) = background = no hit
        if r_val == 255 and g_val == 255 and b_val == 255:
            return -1

        if index >= num_neurons:
            return -1

        return index

    except Exception as e:
        print(f"[Picker] Error: {e}")
        # Try to restore state on error
        try:
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()
            glPopAttrib()
        except Exception:
            pass
        return -1
