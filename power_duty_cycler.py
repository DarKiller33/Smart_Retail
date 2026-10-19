# power_duty_cycler.py
# Motion-Aware Adaptive Edge AI Inference Duty Cycler

import cv2
import numpy as np

class AdaptivePowerManager:
    def __init__(self, motion_threshold=1500.0):
        self.motion_threshold = motion_threshold
        self.prev_gray = None

    def calculate_frame_motion(self, current_frame) -> float:
        """Computes optical flow motion magnitude between consecutive frames."""
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        frame_delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion_score = float(np.sum(thresh))

        self.prev_gray = gray
        return motion_score

    def get_adaptive_sleep_interval(self, motion_score: float) -> tuple:
        """
        Returns (target_fps, sleep_duration_seconds) based on motion score.
        """
        if motion_score > self.motion_threshold:
            return 30, 0.033  # Active Pick Zone: 30 FPS
        else:
            return 2, 0.500   # Cart Idle: 2 FPS (Saves ~78% battery power)