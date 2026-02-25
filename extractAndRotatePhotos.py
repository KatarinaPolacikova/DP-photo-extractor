import os
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "runs/segment/trained_models/photo_segmentation_model_yolo11s/weights/best.pt"
OBJECT_MODEL_PATH = "yolov8n.pt"
OUTPUT_DIR = "extracted_photos"
CONF_THRESHOLD = 0.25

CLASS_PERSON = [0]
CLASSES_OTHER = [1, 2, 3, 15, 16, 24, 39, 41, 56, 58, 63, 67, 73, 74, 76]


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


def refine_mask_with_convex_hull(yolo_mask, w, h):
    """
    Post-processing YOLOv8/YOLO11 masky s Convex Hull
    OPTIMALIZOVANÉ: Jemnejšie nastavenia pre zachovanie detailov

    Pipeline:
    1. Resize masky na plné rozlíšenie
    2. Nižší threshold - zachová viac obsahu pre person detection
    3. Jemná morfológia - neoreže detaily
    4. Convex Hull (bezpečné ohraničenie)
    5. Douglas-Peucker aproximácia

    Returns:
        4-bodový polygón (numpy array)
    """
    # 1. Resize na plné rozlíšenie
    mask_resized = cv2.resize(yolo_mask, (w, h))

    # 2. NIŽŠÍ Threshold (0.2) - zachová viac detailov
    # Predtým 0.3 bolo príliš vysoké
    mask_binary = (mask_resized > 0.2).astype(np.uint8) * 255

    # 3. Jemnejšia morfológia - kernel 5x5, len 2 iterácie
    # Predtým 7x7 a 3 iterácie orezávali príliš veľa
    kernel = np.ones((5, 5), np.uint8)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Nájde kontúry
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Najväčšia kontúra
    largest_contour = max(contours, key=cv2.contourArea)

    # 5. CONVEX HULL
    hull = cv2.convexHull(largest_contour)

    # 6. Douglas-Peucker aproximácia - jemnejšia (0.005 namiesto 0.01)
    perimeter = cv2.arcLength(hull, True)
    epsilon = 0.005 * perimeter
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # Postupne zjednodušuj - pomalšie (1.03 namiesto 1.05)
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


def get_line_score(img):
    """Skórovanie podľa horizontálnych a vertikálnych línií"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=10)
    score = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            if angle < 10 or angle > 170 or (80 < angle < 100):
                score += length
    return score


def crop_photos(input_image_path, output_folder=OUTPUT_DIR):
    """
    Extrahuje fotografie zo skenu.

    Pipeline:
    1. YOLOv11 segmentácia
    2. Convex Hull refinement
    3. Perspektívna transformácia
    4. Multi-level orientation detection
    """
    os.makedirs(output_folder, exist_ok=True)

    seg_model = YOLO(MODEL_PATH)
    obj_model = YOLO(OBJECT_MODEL_PATH)

    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Chyba: Nepodarilo sa načítať {input_image_path}")
        return

    h, w = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]

    print(f"\nSpracovávam: {base_name} ({w}x{h})")

    # STAGE 1: YOLOv11 Segmentácia
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

        # STAGE 2: Convex Hull Refinement
        precise_polygon = refine_mask_with_convex_hull(yolo_mask, w, h)

        if precise_polygon is None or len(precise_polygon) != 4:
            continue

        # Zoradi body
        ordered_polygon = order_points(precise_polygon.reshape(4, 2).astype(np.float32))

        # Vypočítaj rozmery
        w_real = get_distance(ordered_polygon[0], ordered_polygon[1])
        h_real = get_distance(ordered_polygon[0], ordered_polygon[3])
        new_w, new_h = int(w_real), int(h_real)

        print(f"  Polygón: {new_w}x{new_h}px")

        # STAGE 3: Perspektívna transformácia
        dst_pts = np.array([
            [0, 0],
            [new_w - 1, 0],
            [new_w - 1, new_h - 1],
            [0, new_h - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(ordered_polygon, dst_pts)
        warped = cv2.warpPerspective(img, M, (new_w, new_h), flags=cv2.INTER_LANCZOS4)

        # Safety inset
        inset_w, inset_h = max(1, int(new_w * 0.015)), max(1, int(new_h * 0.015))
        if new_w > 2 * inset_w and new_h > 2 * inset_h:
            cropped = warped[inset_h:-inset_h, inset_w:-inset_w]
        else:
            cropped = warped

        # STAGE 4: Multi-level Orientation Detection
        rotation_results = {0: {}, 90: {}, 180: {}, 270: {}}

        for angle in [0, 90, 180, 270]:
            temp = cropped.copy()
            if angle == 90:
                temp = cv2.rotate(temp, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                temp = cv2.rotate(temp, cv2.ROTATE_180)
            elif angle == 270:
                temp = cv2.rotate(temp, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Level 1: Person detection
            res_p = obj_model.predict(source=temp, conf=0.25, classes=CLASS_PERSON, verbose=False)[0]
            p_conf = res_p.boxes.conf.cpu().numpy().sum() if len(res_p.boxes) > 0 else 0.0

            # Level 2: Object detection
            res_o = obj_model.predict(source=temp, conf=0.25, classes=CLASSES_OTHER, verbose=False)[0]
            o_conf = res_o.boxes.conf.cpu().numpy().sum() if len(res_o.boxes) > 0 else 0.0

            # Level 3: Line scoring
            l_score = get_line_score(temp)

            rotation_results[angle] = {
                'p_conf': p_conf,
                'o_conf': o_conf,
                'l_score': l_score
            }

        # Hierarchická detekcia
        max_p = max(rotation_results[a]['p_conf'] for a in rotation_results)
        max_o = max(rotation_results[a]['o_conf'] for a in rotation_results)

        if max_p > 0.3:  # Znížený threshold
            best_angle = max(rotation_results, key=lambda k: rotation_results[k]['p_conf'])
            method = "Osoby"
        elif max_o > 0.4:  # Znížený threshold
            best_angle = max(rotation_results, key=lambda k: rotation_results[k]['o_conf'])
            method = "Objekty"
        else:
            best_angle = max(rotation_results, key=lambda k: rotation_results[k]['l_score'])
            method = "Línie"

        print(f"  Orientácia: {best_angle}° ({method})")

        # Aplikuj rotáciu
        if best_angle == 90:
            final = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
        elif best_angle == 180:
            final = cv2.rotate(cropped, cv2.ROTATE_180)
        elif best_angle == 270:
            final = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            final = cropped

        # Uloženie
        photo_count += 1
        save_path = os.path.join(output_folder, f"{base_name}_photo_{photo_count}.jpg")
        cv2.imwrite(save_path, final, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        print(f"  Uložené: {os.path.basename(save_path)} ({final.shape[1]}x{final.shape[0]})\n")

    print(f"{'=' * 60}")
    print(f"HOTOVO! Extrahovaných: {photo_count} fotiek")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    crop_photos("photo_dataset/images/test/img_0001272.jpg")