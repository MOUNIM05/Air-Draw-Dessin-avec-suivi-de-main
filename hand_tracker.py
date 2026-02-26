import collections
from typing import List, Tuple

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import config

mp_hands = mp.solutions.hands


def _landmarks_to_points(landmarks, width: int, height: int) -> List[Tuple[int, int]]:
    return [(
        int(lmk.x * width),
        int(lmk.y * height)
    ) for lmk in landmarks.landmark]


def _calc_finger_states(points: List[Tuple[int, int]], handedness: str) -> List[bool]:
    # tip ids: thumb 4, index 8, middle 12, ring 16, pinky 20
    # pip ids: thumb 3, index 6, middle 10, ring 14, pinky 18
    thumb_tip, thumb_ip = points[4], points[3]
    index_tip, index_pip = points[8], points[6]
    middle_tip, middle_pip = points[12], points[10]
    ring_tip, ring_pip = points[16], points[14]
    pinky_tip, pinky_pip = points[20], points[18]

    is_right = handedness.lower() == "right"
    thumb_up = thumb_tip[0] > thumb_ip[0] if is_right else thumb_tip[0] < thumb_ip[0]

    index_up = index_tip[1] < index_pip[1]
    middle_up = middle_tip[1] < middle_pip[1]
    ring_up = ring_tip[1] < ring_pip[1]
    pinky_up = pinky_tip[1] < pinky_pip[1]

    return [thumb_up, index_up, middle_up, ring_up, pinky_up]


class HandTracker:
    def __init__(self,
                 max_num_hands: int = 1,
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.6):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.drawing_utils = mp.solutions.drawing_utils
        self._state_window = collections.deque(maxlen=config.SMOOTHING_WINDOW)

    def process(self, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.hands.process(frame_rgb)

    def detect(self, frame_bgr: np.ndarray):
        results = self.process(frame_bgr)
        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label
        h, w, _ = frame_bgr.shape
        points = _landmarks_to_points(hand_landmarks, w, h)
        bbox = self._calc_bbox(points, w, h)
        finger_states = _calc_finger_states(points, handedness)
        smoothed_states = self._smooth_states(finger_states)

        index_tip = points[8]  # (x, y)
        return {
            "points": points,
            "bbox": bbox,
            "states": finger_states,
            "smoothed_states": smoothed_states,
            "index_tip": index_tip,
        }

    @staticmethod
    def _calc_bbox(points: List[Tuple[int, int]], width: int, height: int):
        xs, ys = zip(*points)
        x_min, x_max = max(min(xs), 0), min(max(xs), width)
        y_min, y_max = max(min(ys), 0), min(max(ys), height)
        return x_min, y_min, x_max, y_max

    def _smooth_states(self, states: List[bool]) -> List[float]:
        self._state_window.append(states)
        if not self._state_window:
            return [0.0] * 5
        arr = np.array(self._state_window, dtype=np.float32)
        return np.mean(arr, axis=0).tolist()

    def draw_landmarks(self, frame_bgr: np.ndarray, points: List[Tuple[int, int]]):
        hand_landmarks = landmark_pb2.NormalizedLandmarkList()
        h, w, _ = frame_bgr.shape
        for (x, y) in points:
            nl = hand_landmarks.landmark.add()
            nl.x = x / w
            nl.y = y / h
        self.drawing_utils.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            self.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
            self.drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=2),
        )

    def close(self):
        self.hands.close()
