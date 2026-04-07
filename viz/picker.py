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

    Returns neuron index (0-based), or -1 if no neuron was hit.
    """
    if not OPENGL_AVAILABLE or neuron_positions is None or num_neurons == 0:
        return -1

    try:
        viewport = glGetIntegerv(GL_VIEWPORT)
        width = int(viewport[2])
        height = int(viewport[3])
        if width <= 0 or height <= 0:
            print(f"[Picker] Invalid viewport: {width}x{height}")
            return -1

        # Render to the BACK buffer (double-buffered GLUT)
        glDrawBuffer(GL_BACK)
        glReadBuffer(GL_BACK)

        # Save state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        # Clean render state
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glDisable(GL_DITHER)
        glDisable(GL_FOG)
        glDisable(GL_POINT_SMOOTH)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        # White background = no hit
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Same projection as main renderer
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        fov = getattr(viz_cfg, 'camera_fov', 60.0)
        near = getattr(viz_cfg, 'camera_near_clip', 0.1)
        far = getattr(viz_cfg, 'camera_far_clip', 10000.0)
        gluPerspective(fov, width / height, near, far)

        # Same modelview as main renderer
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

        # Draw neurons with index-encoded colors using vertex arrays
        glPointSize(12.0)

        draw_count = min(num_neurons, len(neuron_positions), 50000)

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

        glFinish()

        # Read pixel — GL y=0 is bottom, GLUT y=0 is top
        gl_y = height - y - 1

        # Read a small area around the click for tolerance
        read_size = 5  # 5x5 pixel area
        half = read_size // 2
        rx = max(0, x - half)
        ry = max(0, gl_y - half)
        rw = min(read_size, width - rx)
        rh = min(read_size, height - ry)

        pixels = glReadPixels(rx, ry, rw, rh, GL_RGB, GL_UNSIGNED_BYTE)

        # Restore state
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glPopAttrib()

        # Search the read area for a non-white pixel (nearest to center first)
        if pixels is None:
            print("[Picker] glReadPixels returned None")
            return -1

        # Convert to numpy for easier indexing
        try:
            pixel_arr = np.frombuffer(pixels, dtype=np.uint8).reshape(rh, rw, 3)
        except (ValueError, TypeError):
            # Try alternative pixel format
            pixel_arr = np.array(pixels, dtype=np.uint8).reshape(rh, rw, 3)

        # Search from center outward
        center_ry = min(half, rh - 1)
        center_rx = min(half, rw - 1)

        best_idx = -1
        best_dist = 999

        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                py = center_ry + dy
                px = center_rx + dx
                if 0 <= py < rh and 0 <= px < rw:
                    r_val = int(pixel_arr[py, px, 0])
                    g_val = int(pixel_arr[py, px, 1])
                    b_val = int(pixel_arr[py, px, 2])

                    # Skip white (background)
                    if r_val == 255 and g_val == 255 and b_val == 255:
                        continue

                    idx = color_to_index(r_val, g_val, b_val)
                    if 0 <= idx < num_neurons:
                        dist = abs(dx) + abs(dy)
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = idx

        if best_idx >= 0:
            print(f"[Picker] Hit neuron #{best_idx} at pixel offset {best_dist}")
        else:
            print(f"[Picker] No hit at ({x},{y}) — center pixel RGB=({int(pixel_arr[center_ry, center_rx, 0])},{int(pixel_arr[center_ry, center_rx, 1])},{int(pixel_arr[center_ry, center_rx, 2])})")

        return best_idx

    except Exception as e:
        print(f"[Picker] Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()
            glPopAttrib()
        except Exception:
            pass
        return -1
