from collections import deque
from typing import Tuple

import cv2
import numpy as np

import config


class StrokeManager:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.undo_stack = deque(maxlen=config.UNDO_STACK_SIZE)
        self.last_point = None

    def resize_if_needed(self, width: int, height: int):
        if width == self.width and height == self.height:
            return
        self.width, self.height = width, height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.undo_stack.clear()
        self.last_point = None

    def push_state(self):
        self.undo_stack.append(self.canvas.copy())

    def undo(self):
        if self.undo_stack:
            self.canvas = self.undo_stack.pop()
            self.last_point = None

    def clear(self):
        self.push_state()
        self.canvas[:] = 0
        self.last_point = None

    def reset_stroke(self):
        self.last_point = None

    def draw(self, point: Tuple[int, int], color: Tuple[int, int, int], brush_size: int):
        if self.last_point is None:
            self.last_point = point
            return
        cv2.line(self.canvas, self.last_point, point, color, brush_size, cv2.LINE_AA)
        self.last_point = point

    def erase(self, point: Tuple[int, int], brush_size: int):
        if self.last_point is None:
            self.last_point = point
            return
        cv2.line(self.canvas, self.last_point, point, (0, 0, 0), brush_size, cv2.LINE_AA)
        self.last_point = point

    def merge(self, frame: np.ndarray) -> np.ndarray:
        mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        return cv2.add(frame_bg, canvas_fg)


def draw_hud(frame: np.ndarray, color: Tuple[int, int, int], brush_size: int, fps: float):
    hud = frame
    cv2.rectangle(hud, (10, 10), (200, 90), (0, 0, 0), thickness=-1)
    cv2.addWeighted(frame[10:90, 10:200], 0.4, hud[10:90, 10:200], 0.6, 0, hud[10:90, 10:200])

    cv2.putText(hud, f"Color", (20, 35), config.HUD_FONT, config.HUD_SCALE, config.HUD_COLOR, config.HUD_THICKNESS, cv2.LINE_AA)
    cv2.rectangle(hud, (90, 18), (170, 42), color, thickness=-1)

    cv2.putText(hud, f"Brush {brush_size}px", (20, 65), config.HUD_FONT, config.HUD_SCALE, config.HUD_COLOR, config.HUD_THICKNESS, cv2.LINE_AA)
    cv2.putText(hud, f"FPS {fps:.1f}", (20, 85), config.HUD_FONT, config.HUD_SCALE, config.HUD_COLOR, config.HUD_THICKNESS, cv2.LINE_AA)
