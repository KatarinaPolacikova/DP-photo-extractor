import os
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "../../runs/segment/trained_models/photo_segmentation_model_yolo11s/weights/best.pt"
OBJECT_MODEL_PATH = "../../yoloModels/yolov8n.pt"
OUTPUT_DIR = "../../extracted_photos"
CONF_THRESHOLD = 0.25

CLASS_PERSON = [0]
# Zoznam objektov (autá, lode, stoly, poháre...)
CLASSES_OTHER = [1, 2, 3, 5, 6, 7, 8, 11, 15, 16, 24, 39, 41, 56, 58, 63, 67, 73, 74, 76]


def order_points(pts):
    """Zoradi 4 body v poradí: TL, TR, BR, BL"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect


def get_distance(p1, p2):
    """Vypočíta euklidovskú vzdialenosť medzi dvoma bodmi"""
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def get_physics_score(img):
    """
    Kombinované fyzikálne skóre pre krajinky a architektúru.
    Sleduje horizont, hladkosť oblohy a rozloženie jasu.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Analýza línií (Horizontálne vs Vertikálne)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=w // 4, maxLineGap=20)
    h_val = 0
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # Vodorovné línie sú pre krajinky plus
            if angle < 10 or angle > 170:
                h_val += length
            # Zvislé línie (kostoly, budovy) sú plus ak je fotka správne otočená
            elif 80 < angle < 100:
                h_val += length * 0.5

    # 2. Obloha (Hladkosť textúry - Sky detection)
    top_std = np.std(gray[:h // 4, :])
    bot_std = np.std(gray[3 * h // 4:, :])
    sky_val = 500 if top_std < bot_std else 0

    # 3. Jas (Obloha býva svetlejšia)
    top_mean = np.mean(gray[:h // 4, :])
    bot_mean = np.mean(gray[3 * h // 4:, :])
    sky_val += 300 if top_mean > bot_mean else 0

    return h_val + sky_val


def refine_mask_with_convex_hull(yolo_mask, w, h):
    """
    Post-processing YOLOv8/YOLO11 masky s Convex Hull

    Pipeline:
    1. Resize masky na plné rozlíšenie
    2. Nižší threshold - zachová viac obsahu pre person detection
    3. Jemná morfológia - neoreže detaily
    4. Convex Hull (bezpečné ohraničenie)
    5. Douglas-Peucker aproximácia

    Returns:
        4-bodový polygón (numpy array)
    """
    # Resize na plné rozlíšenie
    mask_resized = cv2.resize(yolo_mask, (w, h))

    # Threshold (0.2) - zachová viac detailov
    mask_binary = (mask_resized > 0.2).astype(np.uint8) * 255

    # Jemnejšia morfológia - kernel 5x5, len 2 iterácie
    kernel = np.ones((5, 5), np.uint8)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Nájde kontúry
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Najväčšia kontúra
    largest_contour = max(contours, key=cv2.contourArea)

    # CONVEX HULL
    hull = cv2.convexHull(largest_contour)

    # Douglas-Peucker aproximácia
    perimeter = cv2.arcLength(hull, True)
    epsilon = 0.005 * perimeter
    approx = cv2.approxPolyDP(hull, epsilon, True)

    max_iterations = 50
    iteration = 0
    while len(approx) > 4 and iteration < max_iterations:
        epsilon *= 1.03
        approx = cv2.approxPolyDP(hull, epsilon, True)
        iteration += 1

    # Fallback
    if len(approx) != 4:
        rect = cv2.minAreaRect(hull)
        approx = cv2.boxPoints(rect).astype(int).reshape(-1, 1, 2)

    return approx.squeeze()


def rotate_photo(cropped, obj_model):
    """
        Vykoná analýzu 4 uhlov rotácie (0°, 90°, 180°, 270°) a vyberie ten najvhodnejší
        na základe detekcie osôb, iných objektov a fyzikálneho skóre (horizont/obloha).
        """
    candidates = []

    for angle in [0, 90, 180, 270]:
        temp = cropped.copy()
        if angle == 90:
            temp = cv2.rotate(temp, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            temp = cv2.rotate(temp, cv2.ROTATE_180)
        elif angle == 270:
            temp = cv2.rotate(temp, cv2.ROTATE_90_COUNTERCLOCKWISE)

        res_p = obj_model.predict(source=temp, conf=0.45, classes=CLASS_PERSON, verbose=False)[0]
        res_o = obj_model.predict(source=temp, conf=0.45, classes=CLASSES_OTHER, verbose=False)[0]

        p_boxes = res_p.boxes.conf.cpu().numpy()
        p_count = len(p_boxes)
        p_sum = p_boxes.sum() if p_count > 0 else 0.0

        o_boxes = res_o.boxes.conf.cpu().numpy()
        o_count = len(o_boxes)
        o_sum = o_boxes.sum() if o_count > 0 else 0.0

        phys = get_physics_score(temp)

        candidates.append({
            'angle': angle,
            'p_count': p_count,
            'p_sum': p_sum,
            'o_count': o_count,
            'o_sum': o_sum,
            'phys': phys,
            'img': temp
        })

    # --- ROZHODOVACIA LOGIKA ---
    # 1. Priorita: Osoby
    best_p = max(candidates, key=lambda x: (x['p_count'], x['p_sum']))
    if best_p['p_count'] > 0:
        return best_p['img'], "Osoby", best_p['angle']

    # 2. Priorita: Objekty (autá, zvieratá, predmety)
    best_o = max(candidates, key=lambda x: (x['o_count'], x['o_sum']))
    if best_o['o_count'] > 0:
        return best_o['img'], "Objekty", best_o['angle']

    # 3. Priorita: Fyzika (Krajinky/Budovy bez osôb)
    best_phys = max(candidates, key=lambda x: x['phys'])
    return best_phys['img'], "Budovy/Krajinky", best_phys['angle']


def crop_photos(input_image_path):
    """
    Extrahuje fotografie zo skenu.

    Pipeline:
    1. YOLOv11 segmentácia
    2. Convex Hull refinement
    3. Perspektívna transformácia
    4. Multi-level orientation detection
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    seg_model = YOLO(MODEL_PATH)
    obj_model = YOLO(OBJECT_MODEL_PATH)

    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Chyba: Nepodarilo sa načítať {input_image_path}")
        return

    h, w = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]

    print(f"\nSpracovávam: {base_name} ({w}x{h})")

    # YOLOv11 Segmentácia
    results = seg_model.predict(
        source=input_image_path,
        conf=CONF_THRESHOLD,
        imgsz=1024,
        retina_masks=True,
        verbose=False
    )[0]

    if results.masks is None:
        print("Nenašli sa žiadne fotky.")
        return

    print(f"Detekované: {len(results.masks.data)} fotiek\n")

    photo_count = 0
    for i in range(len(results.masks.data)):
        print(f"Foto {i + 1}/{len(results.masks.data)}:")

        yolo_mask = results.masks.data[i].cpu().numpy()

        # Convex Hull Refinement
        precise_polygon = refine_mask_with_convex_hull(yolo_mask, w, h)

        if precise_polygon is None or len(precise_polygon) != 4:
            continue

        # Zoradí body
        ordered_polygon = order_points(precise_polygon.reshape(4, 2).astype(np.float32))

        # Vypočíta rozmery pre perspektívnu transformáciu
        w_real = get_distance(ordered_polygon[0], ordered_polygon[1])
        h_real = get_distance(ordered_polygon[0], ordered_polygon[3])
        new_w, new_h = int(w_real), int(h_real)

        print(f"  Polygón: {new_w}x{new_h}px")

        # Perspektívna transformácia
        dst_pts = np.array([
            [0, 0],
            [new_w - 1, 0],
            [new_w - 1, new_h - 1],
            [0, new_h - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(ordered_polygon, dst_pts)
        warped = cv2.warpPerspective(img, M, (new_w, new_h), flags=cv2.INTER_LANCZOS4)

        # Safety inset (jemné orezanie okrajov pre čistotu)
        inset_w, inset_h = max(1, int(new_w * 0.015)), max(1, int(new_h * 0.015))
        if new_w > 2 * inset_w and new_h > 2 * inset_h:
            cropped = warped[inset_h:-inset_h, inset_w:-inset_w]
        else:
            cropped = warped

        # Rotácie
        final_img, method, ang = rotate_photo(cropped, obj_model)

        photo_count += 1
        save_path = os.path.join(OUTPUT_DIR, f"{base_name}_{photo_count}.jpg")
        cv2.imwrite(save_path, final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(
            f"Foto {photo_count} - Rotácia: {ang}° Metóda: {method}")
        print(f"  Uložené: {os.path.basename(save_path)} ({final_img.shape[1]}x{final_img.shape[0]})\n")

    print(f"{'=' * 60}")
    print(f"HOTOVO! Extrahovaných: {photo_count} fotiek")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    crop_photos("../../photo_dataset/images/test/img_0000127.jpg")
