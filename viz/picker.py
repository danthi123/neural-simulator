"""GPU color-based neuron picking for the 3D view."""
import math
import numpy as np

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False


def index_to_color(index):
    """Encode a neuron index as an RGB color (up to 16M neurons)."""
    r = (index & 0xFF) / 255.0
    g = ((index >> 8) & 0xFF) / 255.0
    b = ((index >> 16) & 0xFF) / 255.0
    return (r, g, b)


def color_to_index(r, g, b):
    """Decode an RGB color back to a neuron index."""
    return int(r) | (int(g) << 8) | (int(b) << 16)


def pick_neuron_at(x, y, neuron_positions, num_neurons, viz_cfg):
    """Pick the neuron at screen position (x, y).

    Renders a single off-screen frame with index-encoded colors,
    reads the pixel, and returns the neuron index or -1 if no hit.

    Args:
        x, y: Screen coordinates (y is from top in GLUT window systems)
        neuron_positions: numpy array (n, 3) of neuron positions
        num_neurons: number of neurons
        viz_cfg: VisualizationConfig with camera parameters
            (camera_azimuth_angle, camera_elevation_angle, camera_radius,
             camera_center_x/y/z, camera_fov, camera_near_clip, camera_far_clip)

    Returns:
        int: neuron index, or -1 if background was clicked
    """
    if not OPENGL_AVAILABLE or neuron_positions is None or num_neurons == 0:
        return -1

    # Save current state
    glPushAttrib(GL_ALL_ATTRIB_BITS)

    try:
        # Clear to white (index 0xFFFFFF = 16777215 = "no neuron")
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Disable lighting, textures, blending for clean color IDs
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glDisable(GL_DITHER)
        glDisable(GL_FOG)

        # Use same camera as main render
        _setup_pick_camera(viz_cfg)

        # Draw neurons as points with index-encoded colors
        glPointSize(8.0)  # Larger than normal for easier picking
        glBegin(GL_POINTS)
        count = min(num_neurons, len(neuron_positions))
        for i in range(count):
            r, g, b = index_to_color(i)
            glColor3f(r, g, b)
            pos = neuron_positions[i]
            glVertex3f(float(pos[0]), float(pos[1]), float(pos[2]))
        glEnd()
        glFlush()

        # Read pixel at click position
        # OpenGL y is from bottom, screen y is from top
        viewport = glGetIntegerv(GL_VIEWPORT)
        gl_y = viewport[3] - y - 1

        pixel = glReadPixels(x, gl_y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE)
        r_val, g_val, b_val = pixel[0][0]

        index = color_to_index(r_val, g_val, b_val)

        # White background = no hit
        if index >= num_neurons or (r_val == 255 and g_val == 255 and b_val == 255):
            return -1

        return index

    except Exception:
        return -1
    finally:
        glPopAttrib()


def _setup_pick_camera(viz_cfg):
    """Set up the same camera transformation as the main render."""
    viewport = glGetIntegerv(GL_VIEWPORT)
    width = viewport[2]
    height = max(viewport[3], 1)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    fov = getattr(viz_cfg, 'camera_fov', 60.0)
    near_clip = getattr(viz_cfg, 'camera_near_clip', 0.1)
    far_clip = getattr(viz_cfg, 'camera_far_clip', 10000.0)
    gluPerspective(fov, width / height, near_clip, far_clip)

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

    gluLookAt(eye_x, eye_y, eye_z,
              cx, cy, cz,
              0.0, 1.0, 0.0)
