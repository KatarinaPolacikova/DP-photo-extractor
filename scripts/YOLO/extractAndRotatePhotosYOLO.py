import os
import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# KONFIGURÁCIA SYSTÉMU A HYPERPARAMETRE
# ==============================================================================
# YOLO11 na segmentáciu (hľadanie fotiek na skene) a YOLOv8n na detekciu objektov (orientácia)
MODEL_PATH = "../../trained_models/trained_yolo11_model.pt"
OBJECT_MODEL_PATH = "../../yolo_models/yolov8n.pt"
OUTPUT_DIR = "../../extracted_photos"
CONF_THRESHOLD = 0.25

# Definícia cieľových tried z COCO datasetu pre určenie rotácie
CLASS_PERSON = [0]
CLASSES_OTHER = [1, 2, 3, 5, 6, 7, 8, 11, 15, 16, 24, 39, 41, 56, 58, 63, 67, 73, 74, 76]


# ==============================================================================
# GEOMETRICKÉ A MATEMATICKÉ POMOCNÉ FUNKCIE
# ==============================================================================
def order_points(pts):
    """
        Zabezpečuje konzistentné poradie rohov polygónu:
        [vľavo-hore, vpravo-hore, vpravo-dole, vľavo-dole].
        Toto poradie je kritické pre správnu perspektívnu transformáciu.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Minimálny súčet súradníc (vľavo-hore)
    rect[2] = pts[np.argmax(s)]  # Maximálny súčet súradníc (vpravo-dole)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Minimálny rozdiel (vpravo-hore)
    rect[3] = pts[np.argmax(diff)]  # Maximálny rozdiel (vľavo-dole)
    return rect


def get_distance(p1, p2):
    """Výpočet Euklidovskej vzdialenosti medzi dvoma bodmi v 2D priestore."""
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def get_physics_score(img):
    """
        Heuristický algoritmus na určenie orientácie obrazu založený na nízkoúrovňových príznakoch.
        Predpoklad: Obloha (vrch) je svetlejšia a menej textúrovaná ako zem (spodok).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Detekcia línií (hľadanie horizontu alebo stien budov)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=w // 4, maxLineGap=20)
    h_val = 0
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if angle < 10 or angle > 170:  # Vodorovné línie
                h_val += length
            elif 80 < angle < 100:  # Zvislé línie
                h_val += length * 0.5

    # 2. Analýza oblohy (Hladkosť: Standard Deviation)
    # Predpoklad: horná štvrtina fotky by mala mať nižšiu smerodajnú odchýlku než spodok (tráva/zem)
    top_std = np.std(gray[:h // 4, :])
    bot_std = np.std(gray[3 * h // 4:, :])
    sky_val = 500 if top_std < bot_std else 0

    # 3. Analýza jasu (Mean)
    top_mean = np.mean(gray[:h // 4, :])
    bot_mean = np.mean(gray[3 * h // 4:, :])
    sky_val += 300 if top_mean > bot_mean else 0

    return h_val + sky_val


def refine_mask_with_convex_hull(yolo_mask, w, h):
    """
        Geometrické vylepšenie YOLO masky.
        YOLO masky bývajú 'zubaté'. Táto funkcia ich obalí do Convex Hull (konvexný obal)
        a pomocou Douglas-Peuckerovho algoritmu ich zjednoduší na presný 4-uholník.
    """
    mask_resized = cv2.resize(yolo_mask, (w, h))
    mask_binary = (mask_resized > 0.2).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)

    # Douglas-Peucker aproximácia
    # Iteratívne hľadanie 4 bodov (rohov fotografie)
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
    # Ak algoritmus zlyhá, použijeme minimálny ohraničujúci obdĺžnik
    if len(approx) != 4:
        rect = cv2.minAreaRect(hull)
        approx = cv2.boxPoints(rect).astype(int).reshape(-1, 1, 2)

    return approx.squeeze()


def rotate_photo(cropped, obj_model):
    """
        Viacúrovňový rozhodovací proces pre automatickú korekciu rotácie.
        Metóda Brute-force testuje 4 základné rotácie (0°, 90°, 180°, 270°) a
        vyberá optimálnu orientáciu na základe hierarchie sémantických detekcií.
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

        # Detekcia objektov pre každý uhol
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

    # ROZHODOVACIA LOGIKA
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
        Hlavný procesný kanál (Pipeline) pre extrakciu fotografií.
        Zahŕňa: preprocessing, sémantickú segmentáciu, perspektívnu transformáciu a korekciu orientácie.
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

        # 1. Získanie presného 4-uholníka
        yolo_mask = results.masks.data[i].cpu().numpy()
        # Convex Hull Refinement
        precise_polygon = refine_mask_with_convex_hull(yolo_mask, w, h)

        if precise_polygon is None or len(precise_polygon) != 4:
            continue

        ordered_polygon = order_points(precise_polygon.reshape(4, 2).astype(np.float32))

        # 2. Perspektívna transformácia (narovnanie nakrivo položenej fotky)
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

        # 3. Jemný 'inset' (odstránenie milimetrových bielych okrajov zo skenu)
        inset_w, inset_h = max(1, int(new_w * 0.015)), max(1, int(new_h * 0.015))
        if new_w > 2 * inset_w and new_h > 2 * inset_h:
            cropped = warped[inset_h:-inset_h, inset_w:-inset_w]
        else:
            cropped = warped

        # 4. Inteligentná rotácia
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
    crop_photos("../../photo_dataset/images/test/testsnimka.png")
