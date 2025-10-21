import cv2
import numpy as np
from typing import List, Tuple

def load_image_from_bytes(image_bytes: bytes):
    """Load an image from raw bytes using OpenCV."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def draw_boxes(image, boxes: List[Tuple[int, int, int, int]]):
    """Draw bounding boxes on the image for visualization (optional)."""
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image
