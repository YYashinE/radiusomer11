"""
Ядро измерения: детекция ArUco-метки, вычисление гомографии
(перевод "пиксели фото" -> "миллиметры в плоскости метки") и
измерение диаметра режущей части фрезы по двум точкам, отмеченным
пользователем на изображении.

Важное допущение (см. README): фреза должна лежать в ТОЙ ЖЕ плоскости,
что и калибровочная метка. Если ось фрезы наклонена к плоскости метки,
результат будет систематически искажён — это ограничение гомографии
от плоского маркера, а не программная ошибка.
"""
from dataclasses import dataclass
import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_5X5_50
MARKER_ID = 0


@dataclass
class MarkerResult:
    found: bool
    corners_px: list  # [[x,y], ...] 4 угла в порядке, возвращаемом OpenCV
    reprojection_error_mm: float = 0.0


@dataclass
class MeasureResult:
    diameter_mm: float
    radius_mm: float
    error_mm: float
    mm_per_px: float
    reprojection_error_mm: float
    warning: str = ""


def _get_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    # Более чувствительные пороги адаптивной бинаризации — метка на фото
    # телефона часто занимает малую площадь кадра и может быть смазана.
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 33
    params.adaptiveThreshWinSizeStep = 4
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def detect_marker(image: np.ndarray) -> MarkerResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    detector = _get_detector()
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return MarkerResult(found=False, corners_px=[])

    idx = None
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id == MARKER_ID:
            idx = i
            break
    if idx is None:
        idx = 0  # fallback: используем первую найденную метку

    pts = corners[idx][0]  # shape (4,2), порядок: top-left, top-right, bottom-right, bottom-left
    return MarkerResult(found=True, corners_px=pts.tolist())


def _build_homography(corners_px: list, marker_size_mm: float):
    image_pts = np.array(corners_px, dtype=np.float32)
    # Мировые координаты угла метки (Z=0 плоскость), мм.
    # Порядок должен соответствовать порядку corners_px из detectMarkers:
    # top-left, top-right, bottom-right, bottom-left.
    s = marker_size_mm
    world_pts = np.array([
        [0, 0],
        [s, 0],
        [s, s],
        [0, s],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(image_pts, world_pts, method=0)

    # Оценка качества: репроецируем углы метки через H и считаем ошибку в мм
    ones = np.ones((4, 1), dtype=np.float32)
    img_h = np.hstack([image_pts, ones])
    proj = (H @ img_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    reproj_err = float(np.mean(np.linalg.norm(proj - world_pts, axis=1)))

    return H, reproj_err


def image_point_to_mm(H: np.ndarray, pt_px):
    v = np.array([pt_px[0], pt_px[1], 1.0], dtype=np.float64)
    w = H @ v
    w = w[:2] / w[2]
    return w


def measure_diameter(image: np.ndarray, marker_size_mm: float, pt1_px, pt2_px) -> MeasureResult:
    marker = detect_marker(image)
    if not marker.found:
        raise ValueError("Калибровочная метка не найдена на изображении")

    H, reproj_err_mm = _build_homography(marker.corners_px, marker_size_mm)

    p1_mm = image_point_to_mm(H, pt1_px)
    p2_mm = image_point_to_mm(H, pt2_px)
    diameter_mm = float(np.linalg.norm(p1_mm - p2_mm))

    # Локальный масштаб мм/px рядом с точками клика (для оценки погрешности
    # клика в мм — вблизи метки масштаб примерно линеен, поэтому берём
    # среднее расстояние между двумя близкими точками возле каждого клика)
    eps_px = 2.0
    p1_mm_dx = image_point_to_mm(H, (pt1_px[0] + eps_px, pt1_px[1]))
    local_mm_per_px = float(np.linalg.norm(p1_mm_dx - p1_mm)) / eps_px

    # Оценка погрешности: неопределённость клика (~2px на каждую точку)
    # + ошибка репроекции метки, сложенные геометрически.
    click_uncertainty_mm = local_mm_per_px * 2.0 * np.sqrt(2)  # 2 точки, по 2px каждая
    total_error_mm = float(np.sqrt(click_uncertainty_mm**2 + reproj_err_mm**2))

    warning = ""
    if reproj_err_mm > 0.5:
        warning = (
            "Метка распознана нечётко (большая ошибка репроекции). "
            "Переснимите фото при лучшем освещении и меньшем угле наклона камеры."
        )

    return MeasureResult(
        diameter_mm=diameter_mm,
        radius_mm=diameter_mm / 2,
        error_mm=total_error_mm,
        mm_per_px=local_mm_per_px,
        reprojection_error_mm=reproj_err_mm,
        warning=warning,
    )
