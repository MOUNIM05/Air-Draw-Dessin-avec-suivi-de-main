# Global configuration values for the air drawing app.

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Drawing
BRUSH_SIZE_MIN = 5
BRUSH_SIZE_MAX = 40
ERASER_SIZE = 40
SMOOTHING_WINDOW = 5  # number of frames for smoothed finger states
STROKE_SMOOTHING_WINDOW = 5  # moving average over index tip positions while drawing

# Colors (BGR for OpenCV)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_ERASE = (0, 0, 0)
DEFAULT_COLOR = COLOR_RED
BACKGROUND_BLUR = True

# HUD
HUD_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE = 0.7
HUD_COLOR = (255, 255, 255)
HUD_THICKNESS = 2

# Undo stack
UNDO_STACK_SIZE = 10

# Brush Y range mapping (pixels)
BRUSH_Y_MIN = 100  # higher in frame -> thinner brush
BRUSH_Y_MAX = 600  # lower in frame -> thicker brush
