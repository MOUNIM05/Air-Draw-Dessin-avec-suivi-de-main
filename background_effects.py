from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np

import config


class BackgroundEffects:
    def __init__(self):
        self.segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def apply(self, frame: np.ndarray, hand_bbox: Tuple[int, int, int, int] | None = None) -> np.ndarray:
        if not config.BACKGROUND_BLUR:
            return frame

        h, w, _ = frame.shape
        blurred = cv2.GaussianBlur(frame, (35, 35), 0)

        # Try segmentation first
        results = self.segmentation.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.segmentation_mask is not None:
            mask = results.segmentation_mask
            mask = cv2.resize(mask, (w, h))
            mask = np.clip(mask, 0, 1)
            mask3 = np.dstack([mask] * 3)
            return (frame * mask3 + blurred * (1 - mask3)).astype("uint8")

        # Fallback: unblur hand bbox only
        if hand_bbox:
            x1, y1, x2, y2 = hand_bbox
            result = blurred.copy()
            region = frame[y1:y2, x1:x2]
            result[y1:y2, x1:x2] = region
            return result

        return blurred

    def close(self):
        self.segmentation.close()
