"""
Two estimators:
- SpeedEstimatorMPP: uses meters_per_pixel (naive)
- SpeedEstimatorHomography: maps image centroid -> world XY using homography and computes speeds in m/s -> km/h
Both expect history: list of (timestamp, (x,y)) in image pixels.
"""

import math
import numpy as np
import cv2
from collections import deque

class SpeedEstimatorMPP:
    def __init__(self, meters_per_pixel=0.02, smoothing=3):
        self.mpp = meters_per_pixel
        self.smoothing = smoothing
        self.speeds = {}  # id -> deque of recent m/s

    def estimate_speed(self, obj_id, history_positions):
        """
        history_positions: list of (timestamp, (x,y)) in chronological order
        returns speed_kmph or None
        """
        if len(history_positions) < 2:
            return None
        (t1,p1), (t2,p2) = history_positions[-2], history_positions[-1]
        dt = t2 - t1
        if dt <= 1e-4:
            return None
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist_pixels = math.hypot(dx, dy)
        dist_meters = dist_pixels * self.mpp
        speed_mps = dist_meters / dt
        q = self.speeds.setdefault(obj_id, deque(maxlen=self.smoothing))
        q.append(speed_mps)
        avg_mps = sum(q)/len(q)
        return avg_mps * 3.6  # km/h

class SpeedEstimatorHomography:
    def __init__(self, H_path='calib/H.npy', smoothing=3):
        self.H = np.load(H_path)
        self.smoothing = smoothing
        self.speeds = {}  # id -> deque

    def image_to_world(self, centroid):
        pt = np.array([[[centroid[0], centroid[1]]]], dtype='float32')  # shape 1x1x2
        wp = cv2.perspectiveTransform(pt, self.H)[0][0]  # X, Y
        return float(wp[0]), float(wp[1])

    def estimate_speed(self, obj_id, history_positions):
        """
        history_positions: list of (timestamp, (x,y)) in chronological order
        We'll convert last two points to world coords and compute speed.
        Returns speed_kmph or None
        """
        if len(history_positions) < 2:
            return None
        (t1,p1), (t2,p2) = history_positions[-2], history_positions[-1]
        dt = t2 - t1
        if dt <= 1e-4:
            return None
        x1,y1 = self.image_to_world(p1)
        x2,y2 = self.image_to_world(p2)
        dist_m = math.hypot(x2-x1, y2-y1)
        speed_mps = dist_m / dt
        q = self.speeds.setdefault(obj_id, deque(maxlen=self.smoothing))
        q.append(speed_mps)
        avg_mps = sum(q)/len(q)
        return avg_mps * 3.6  # km/h
