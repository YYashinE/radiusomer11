"""
Веб-сервис измерения диаметра режущей части фрезы по фото.

Запуск (dev):
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Для доступа с телефона в локальной сети браузеру нужен HTTPS
(getUserMedia требует secure context, кроме localhost).
Проще всего для теста — открыть с того же компьютера (localhost),
либо прокинуть туннель (ngrok/cloudflared) для доступа с телефона.
См. README.md.
"""
import json
import io

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from measure import detect_marker, measure_diameter

app = FastAPI(title="Mill Radius Estimator")

# CORS открыт полностью — это прототип для локального использования.
# Перед публичным деплоем сузьте allow_origins до реального домена фронтенда.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_image(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")
    return img


@app.post("/api/detect-marker")
async def api_detect_marker(image: UploadFile = File(...)):
    """Проверяет, видна ли калибровочная метка на кадре, возвращает её
    углы в пикселях — фронтенд рисует поверх видео зелёную рамку,
    чтобы пользователь понимал, что метку видно."""
    img = _read_image(image)
    result = detect_marker(img)
    return JSONResponse({
        "found": result.found,
        "corners_px": result.corners_px,
        "image_size": [img.shape[1], img.shape[0]],
    })


@app.post("/api/measure")
async def api_measure(
    image: UploadFile = File(...),
    marker_size_mm: float = Form(...),
    points: str = Form(...),  # JSON: "[[x1,y1],[x2,y2]]"
):
    img = _read_image(image)

    try:
        pts = json.loads(points)
        assert len(pts) == 2
        pt1, pt2 = pts
    except Exception:
        raise HTTPException(status_code=400, detail="points должен быть JSON вида [[x1,y1],[x2,y2]]")

    try:
        result = measure_diameter(img, marker_size_mm, pt1, pt2)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return JSONResponse({
        "diameter_mm": round(result.diameter_mm, 3),
        "radius_mm": round(result.radius_mm, 3),
        "error_mm": round(result.error_mm, 3),
        "mm_per_px": round(result.mm_per_px, 5),
        "reprojection_error_mm": round(result.reprojection_error_mm, 3),
        "warning": result.warning,
    })


@app.get("/api/health")
async def health():
    return {"status": "ok"}
