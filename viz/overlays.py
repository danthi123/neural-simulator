"""OpenGL text overlay rendering (HUD footer).

Provides render_text_gl() for drawing bitmap text in the OpenGL window.
"""

try:
    from OpenGL.GL import *
    import OpenGL.GLUT as glut
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

# Reference to opengl_viz_config set from renderer module
opengl_viz_config = None


def set_viz_config(cfg):
    global opengl_viz_config
    opengl_viz_config = cfg


def render_text_gl(x, y, text, font=None):
    """Renders text on the OpenGL screen. Called by the main thread."""
    if not OPENGL_AVAILABLE:
        return
    if font is None:
        font = glut.GLUT_BITMAP_9_BY_15 if hasattr(glut, "GLUT_BITMAP_9_BY_15") else None
    if font is None:
        print("Warning: GLUT font not available for render_text_gl.")
        return

    try:
        current_win = glut.glutGetWindow()
        if current_win == 0:
            return

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        _cfg = opengl_viz_config or {}
        win_w = _cfg.get('WINDOW_WIDTH', 800)
        win_h = _cfg.get('WINDOW_HEIGHT', 600)
        gluOrtho2D(0, win_w, 0, win_h)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glColor3f(0.9, 0.9, 0.9)
        glDisable(GL_DEPTH_TEST)

        glRasterPos2i(int(x), int(y))
        for character in text:
            glut.glutBitmapCharacter(font, ord(character))

        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    except Exception as e:
        print(f"[ERROR] OpenGL render_text_gl: {e}")
