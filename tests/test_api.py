import io
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_detect_endpoint():
    # Create a fake blank image for quick test
    import numpy as np
    import cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    img_bytes = io.BytesIO(buffer.tobytes())

    response = client.post(
        "/detect",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "defect" in data
    assert "confidence" in data
    assert "boxes" in data
