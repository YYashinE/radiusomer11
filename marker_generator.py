"""
Генератор печатной калибровочной метки (ArUco) для сервиса измерения фрез.

Метка нужна для перевода "пиксели -> миллиметры" на фото. Пользователь
распечатывает её БЕЗ масштабирования ("Actual size / 100%", не "Fit to page"),
кладёт рядом с фрезой в той же плоскости и фотографирует.

Запуск:
    python marker_generator.py --size-mm 40 --out marker_40mm.png

Метка использует словарь DICT_5X5_50, id=0 — выбран из соображений
устойчивости распознавания на смартфонных камерах при разных углах/освещении.
"""
import argparse
import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_5X5_50
MARKER_ID = 0

# Печатаем метку на листе A4 с полями и подписью реального размера,
# чтобы пользователь мог проверить линейкой, что печать не масштабирована.
DPI = 300


def mm_to_px(mm: float, dpi: int = DPI) -> int:
    return int(round(mm / 25.4 * dpi))


def generate(marker_size_mm: float, out_path: str):
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    marker_px = mm_to_px(marker_size_mm)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, MARKER_ID, marker_px)

    # Холст A4: 210x297 мм
    page_w = mm_to_px(210)
    page_h = mm_to_px(297)
    canvas = np.full((page_h, page_w), 255, dtype=np.uint8)

    # Метка по центру верхней трети листа
    x0 = (page_w - marker_px) // 2
    y0 = mm_to_px(30)
    canvas[y0:y0 + marker_px, x0:x0 + marker_px] = marker_img

    # Белая рамка-quiet zone вокруг метки (важно для устойчивого распознавания)
    border = mm_to_px(5)
    cv2.rectangle(
        canvas,
        (x0 - border, y0 - border),
        (x0 + marker_px + border, y0 + marker_px + border),
        128, 1
    )

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # Подписи (важно: реальный размер метки в мм, для проверки после печати)
    text_y = y0 + marker_px + mm_to_px(15)
    cv2.putText(canvas_bgr, f"Marker size: {marker_size_mm:.0f} x {marker_size_mm:.0f} mm",
                (x0, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas_bgr, "Print at 100% / Actual size (NOT 'Fit to page')",
                (x0, text_y + mm_to_px(8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas_bgr, f"ArUco dict: DICT_5X5_50, id={MARKER_ID}",
                (x0, text_y + mm_to_px(16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)

    # Контрольная линейка 50мм для проверки печати
    ruler_y = text_y + mm_to_px(28)
    ruler_len = mm_to_px(50)
    cv2.line(canvas_bgr, (x0, ruler_y), (x0 + ruler_len, ruler_y), (0, 0, 0), 2)
    for mm in range(0, 51, 10):
        px = x0 + mm_to_px(mm)
        cv2.line(canvas_bgr, (px, ruler_y - 8), (px, ruler_y + 8), (0, 0, 0), 2)
        cv2.putText(canvas_bgr, f"{mm}", (px - 8, ruler_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas_bgr, "Check ruler with a real ruler after printing",
                (x0, ruler_y + mm_to_px(12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas_bgr)
    print(f"Saved: {out_path} ({marker_size_mm}mm marker, {DPI} DPI, print at 100%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mm", type=float, default=40.0)
    parser.add_argument("--out", type=str, default="marker_40mm.png")
    args = parser.parse_args()
    generate(args.size_mm, args.out)
