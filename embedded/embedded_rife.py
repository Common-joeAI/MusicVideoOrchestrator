import os
import cv2
import numpy as np
from typing import List


class EmbeddedRIFE:
    """Class for frame interpolation using OpenCV"""

    def __init__(self, models_dir: str):
        self.models_dir = os.path.abspath(models_dir)
        self.initialized = False

    def initialize(self):
        """Initialize interpolation system"""
        if self.initialized:
            return True

        try:
            # Nothing to initialize for OpenCV interpolation
            self.initialized = True
            print("Initialized OpenCV-based frame interpolation")
            return True
        except Exception as e:
            print(f"Error initializing interpolation system: {str(e)}")
            return False

    def interpolate_frames(self, frame1: np.ndarray, frame2: np.ndarray, num_frames: int) -> List[np.ndarray]:
        """Interpolate between two frames using OpenCV"""
        if not self.initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize interpolation system")

        # Use OpenCV's linear interpolation
        interpolated_frames = []
        for i in range(1, num_frames + 1):
            alpha = i / (num_frames + 1)
            interpolated = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            interpolated_frames.append(interpolated)

        return interpolated_frames

    def cleanup(self):
        """Clean up resources"""
        # Nothing to clean up for OpenCV interpolation
        pass
