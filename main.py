import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import background_effects
import config
import drawing_utils
import gesture_logic
import hand_tracker


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: cannot open webcam. Check camera index or permissions.")
        return

    tracker = hand_tracker.HandTracker()
    bg = background_effects.BackgroundEffects()

    ret, frame = cap.read()
    if not ret:
        print("Error: failed to read initial frame.")
        return
    height, width, _ = frame.shape
    strokes = drawing_utils.StrokeManager(width, height)

    current_color = config.DEFAULT_COLOR
    brush_size = config.BRUSH_SIZE_MIN
    fps = 0.0
    prev_time = time.time()
    point_smoother = deque(maxlen=config.STROKE_SMOOTHING_WINDOW)

    save_dir = Path("drawings")
    save_dir.mkdir(exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: frame grab failed; exiting loop.")
            break

        height, width, _ = frame.shape
        strokes.resize_if_needed(width, height)

        detect_result = tracker.detect(frame)
        hand_bbox = None

        if detect_result:
            smoothed = detect_result["smoothed_states"]
            index_tip = detect_result["index_tip"]
            hand_bbox = detect_result["bbox"]

            drawing_on = gesture_logic.is_drawing(smoothed)
            color_action = gesture_logic.detect_color(smoothed)
            brush_size = gesture_logic.brush_size_from_y(index_tip[1])

            erasing = color_action == "erase"
            if isinstance(color_action, tuple):
                current_color = color_action

            active_draw = drawing_on or erasing
            draw_color = config.COLOR_ERASE if erasing else current_color
            draw_size = config.ERASER_SIZE if erasing else brush_size

            if active_draw:
                point_smoother.append(index_tip)
                avg_x = int(np.mean([p[0] for p in point_smoother]))
                avg_y = int(np.mean([p[1] for p in point_smoother]))
                if erasing:
                    strokes.erase((avg_x, avg_y), draw_size)
                else:
                    strokes.draw((avg_x, avg_y), draw_color, draw_size)
            else:
                strokes.reset_stroke()
                point_smoother.clear()

            # Optional landmarks for debugging
            tracker.draw_landmarks(frame, detect_result["points"])
        else:
            strokes.reset_stroke()
            point_smoother.clear()

        blurred_frame = bg.apply(frame, hand_bbox)
        output = strokes.merge(blurred_frame)

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-6))
        prev_time = curr_time

        drawing_utils.draw_hud(output, current_color, brush_size, fps)

        cv2.imshow("Air Draw", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            strokes.clear()
        if key == ord('u'):
            strokes.undo()
        if key == ord('s'):
            filename = save_dir / f"drawing_{int(time.time())}.png"
            cv2.imwrite(str(filename), strokes.merge(np.zeros_like(frame)))
            print(f"Saved drawing to {filename}")

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    bg.close()


if __name__ == "__main__":
    main()
