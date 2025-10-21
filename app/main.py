# main.py (replace the /detect route with this version)

from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import io
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Defect Detection Backend Running!"}


@app.post("/detect")
async def detect_defect(file: UploadFile = File(...)):
    # Read the uploaded file as bytes
    image_bytes = await file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges using Canny
    edges = cv2.Canny(blur, 50, 150)

    # Find contours (potential defect regions)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the total area of contours relative to image area
    defect_area = sum(cv2.contourArea(c) for c in contours)
    total_area = img.shape[0] * img.shape[1]
    defect_ratio = defect_area / total_area

    # Classify defect based on area ratio
    if defect_ratio > 0.01:
        status = "Defect detected (possible scratch or crack)"
    elif 0.002 < defect_ratio <= 0.01:
        status = "Minor defect detected"
    else:
        status = "No visible defect"

    return JSONResponse({
        "status": status,
        "defect_ratio": round(defect_ratio, 4),
        "contours_detected": len(contours)
    })
