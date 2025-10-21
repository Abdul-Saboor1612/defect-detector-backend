import cv2
import numpy as np
from app.config import CONFIDENCE_THRESHOLD, MIN_DEFECT_AREA, DEBUG

class DefectDetector:
    def __init__(self):
        if DEBUG:
            print("[INFO] DefectDetector initialized")

    def detect_defects(self, image):
        """Simple heuristic defect detection using edge + contour analysis."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_DEFECT_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append((x, y, x + w, y + h))

        confidence = min(1.0, len(boxes) * 0.1)
        defect_present = confidence > CONFIDENCE_THRESHOLD

        if DEBUG:
            print(f"[DEBUG] Detected {len(boxes)} potential defects with confidence {confidence:.2f}")

        return {
            "defect": defect_present,
            "confidence": round(confidence, 2),
            "boxes": boxes
        }
