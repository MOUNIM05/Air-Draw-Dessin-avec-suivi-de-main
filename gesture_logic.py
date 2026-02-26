from typing import List, Optional, Tuple

import numpy as np

import config


def _bool_state(smoothed_value: float, thresh: float = 0.5) -> bool:
    return smoothed_value >= thresh


def is_drawing(smoothed_states: List[float]) -> bool:
    thumb, index, middle, ring, pinky = [_bool_state(v) for v in smoothed_states]
    return index and not (middle or ring or pinky)


def detect_color(smoothed_states: List[float]) -> Optional[Tuple[int, int, int]]:
    thumb, index, middle, ring, pinky = [_bool_state(v) for v in smoothed_states]

    if not thumb and not index and not middle and not ring and not pinky:
        return "erase"
    if index and middle and ring and not pinky:
        return config.COLOR_GREEN
    if index and middle and not ring and not pinky:
        return config.COLOR_RED
    if index and pinky and not middle and not ring:
        return config.COLOR_BLUE
    return None


def brush_size_from_y(index_y: int) -> int:
    return int(np.interp(
        index_y,
        [config.BRUSH_Y_MIN, config.BRUSH_Y_MAX],
        [config.BRUSH_SIZE_MIN, config.BRUSH_SIZE_MAX]
    ))
