import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from ultralytics import YOLO

# ==============================================================================
# KONFIGURÁCIA SYSTÉMU A HYPERPARAMETRE
# ==============================================================================
UNET_MODEL_PATH = "../../trained_models/trained_unet_model.pt"
OBJECT_MODEL_PATH = "../../yolo_models/yolov8n.pt"   # YOLOv8 pre sémantickú analýzu obsahu
OUTPUT_DIR = "../../extracted_photos_UNET"
CONF_THRESHOLD = 0.5  # Klasifikačný prah pre binárnu segmentáciu

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Rozdelenie obrazu na vrchnú (obloha) a spodnú (zem) polovicu
    top_half = gray[:h // 2, :]
    bot_half = gray[h // 2:, :]

    # 1. Analýza distribúcie jasu (Luminance Analysis)
    top_mean = np.mean(top_half)
    bot_mean = np.mean(bot_half)
    # Normalizácia rozdielu (čím viac do plusu, tým lepšie)
    brightness_score = (top_mean - bot_mean) / 255.0

    # 2. Analýza hustoty hrán (Edge Density) - detekcia detailov a textúry pomocou Cannyho detektora
    edges = cv2.Canny(gray, 50, 150)
    top_edges = np.count_nonzero(edges[:h // 2, :])
    bot_edges = np.count_nonzero(edges[h // 2:, :])

    total_edges = max(top_edges + bot_edges, 1)
    # Normalizovaný rozdiel (kladné číslo znamená viac hrán dole)
    edge_score = (bot_edges - top_edges) / total_edges

    # 3. Houghova transformácia na detekciu horizontálnych línií (horizont)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=max(w, h) // 4, maxLineGap=20)
    horiz_len = 0
    vert_len = 0
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if angle < 20 or angle > 160:
                horiz_len += length
            elif 70 < angle < 110:
                vert_len += length

    total_len = max(horiz_len + vert_len, 1)
    line_score = (horiz_len - vert_len) / total_len

    # Vážený súčet príznakov pre finálne rozhodnutie
    final_score = (brightness_score * 2.0) + (edge_score * 1.5) + (line_score * 0.5)
    return final_score


def refine_mask_with_convex_hull(mask_resized, w, h):
    """
        Post-processing segmentačnej masky.
        Transformuje pravdepodobnostnú mapu z U-Netu na geometricky presný konvexný štvoruholník.
        Využíva morfologické operácie a Douglas-Peuckerov algoritmus na aproximáciu polygónu.
    """
    mask_binary = (mask_resized > 0.2).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    # Odstránenie šumu a spojenie nespojitých oblastí v maske
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Identifikácia najväčšieho kontúru a výpočet konvexnej obálky (Convex Hull)
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)

    # Iteratívna aproximácia na získanie presne 4 vrcholov (geometrická definícia fotografie)
    perimeter = cv2.arcLength(hull, True)
    epsilon = 0.005 * perimeter
    approx = cv2.approxPolyDP(hull, epsilon, True)
    iteration = 0

    while len(approx) > 4 and iteration < 50:
        epsilon *= 1.03
        approx = cv2.approxPolyDP(hull, epsilon, True)
        iteration += 1

    # Záložné riešenie: Ak aproximácia zlyhá, použije sa bounding box s minimálnou plochou
    if len(approx) != 4:
        rect = cv2.minAreaRect(hull)
        approx = cv2.boxPoints(rect).astype(int).reshape(-1, 1, 2)
    return approx.squeeze()


# ==============================================================================
# NAČÍTANIE MODELOV A INFERENCIA
# ==============================================================================
def load_unet_model(path):
    """Inicializácia architektúry U-Net s enkóderom ResNet34 a načítanie váh."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation='sigmoid'
    )
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


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

        # Inferencia YOLO na detekciu objektov v aktuálnej rotácii
        res_p = obj_model.predict(source=temp, conf=0.25, classes=CLASS_PERSON, verbose=False)[0]
        res_o = obj_model.predict(source=temp, conf=0.25, classes=CLASSES_OTHER, verbose=False)[0]

        # 1. Metrika: Prítomnosť osôb
        if len(res_p.boxes) > 0:
            p_scores = res_p.boxes.conf.cpu().numpy()
            p_sum = float(p_scores.sum())
            p_count = len(p_scores)
        else:
            p_sum = 0.0
            p_count = 0

        # 2. Metrika: Prítomnosť objektov
        if len(res_o.boxes) > 0:
            o_scores = res_o.boxes.conf.cpu().numpy()
            o_sum = float(o_scores.sum())
        else:
            o_sum = 0.0

        # 3. Metrika: Fyzikálne príznaky (horizont a jas)
        phys = get_physics_score(temp)

        # Penalizácia 180° rotácie na zníženie falošne pozitívnych nálezov v "hore nohami" stave
        penalty = 0.9 if angle == 180 else 1.0

        candidates.append({
            'angle': angle,
            'img': temp,
            'p_sum': p_sum * penalty,
            'p_count': p_count,
            'o_sum': o_sum,
            'phys': phys
        })

    # ROZHODOVACIA LOGIKA
    # 1. Priorita: Osoby
    best_p = max(candidates, key=lambda x: (x['p_count'], x['p_sum']))

    if best_p['p_count'] > 0:
        return best_p['img'], "Osoby", best_p['angle']

    # 2. Priorita: Objekty (autá, zvieratá, predmety)
    best_o = max(candidates, key=lambda x: x['o_sum'])

    if best_o['o_sum'] > 0.5:
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

    # Inicializácia neurónových sietí
    unet_model = load_unet_model(UNET_MODEL_PATH)
    obj_model = YOLO(OBJECT_MODEL_PATH)

    img = cv2.imread(input_image_path)

    if img is None:
        return

    h_orig, w_orig = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]

    # 1. PREPROCESSING: Normalizácia a príprava tenzora pre U-Net (ImageNet štatistiky)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img_rgb, (512, 512))
    img_input = img_input.astype(np.float32) / 255.0
    img_input = (img_input - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(DEVICE).float()

    # 2. SÉMANTICKÁ SEGMENTÁCIA (Inferencia)
    with torch.no_grad():
        output_mask = unet_model(img_tensor).cpu().squeeze().numpy()

    # Rescaling masky na pôvodné rozlíšenie vstupného obrazu
    full_mask = cv2.resize(output_mask, (w_orig, h_orig))
    binary_mask = (full_mask > CONF_THRESHOLD).astype(np.uint8) * 255

    # 3. EXTRAKCIA ELEMENTOV (Detekcia kontúr)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrácia kontúr na základe relatívnej plochy (eliminácia artefaktov)
    valid_contours = [c for c in contours if cv2.contourArea(c) > (w_orig * h_orig * 0.005)]
    print(f"Detekované: {len(valid_contours)} fotiek\n")

    photo_count = 0
    for i, cnt in enumerate(valid_contours):
        print(f"Foto {i + 1}/{len(valid_contours)}:")

        single_obj_mask = np.zeros((h_orig, w_orig), dtype=np.float32)
        cv2.drawContours(single_obj_mask, [cnt], -1, 1.0, -1)

        # Convex Hull Refinement masky na štvoruholník
        precise_polygon = refine_mask_with_convex_hull(single_obj_mask, w_orig, h_orig)

        if precise_polygon is None or len(precise_polygon) != 4:
            continue

        # 4. PERSPEKTÍVNA TRANSFORMÁCIA (Rectification)
        # Odstránenie perspektívneho skreslenia a narovnanie objektu
        ordered_polygon = order_points(precise_polygon.reshape(4, 2).astype(np.float32))
        w_real = get_distance(ordered_polygon[0], ordered_polygon[1])
        h_real = get_distance(ordered_polygon[0], ordered_polygon[3])
        new_w, new_h = int(w_real), int(h_real)

        dst_pts = np.array([[0, 0], [new_w - 1, 0], [new_w - 1, new_h - 1], [0, new_h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered_polygon, dst_pts)
        warped = cv2.warpPerspective(img, M, (new_w, new_h), flags=cv2.INTER_LANCZOS4)

        # Orezanie okrajov (Inset) na elimináciu zvyškov pozadia/masky
        inset_w, inset_h = max(1, int(new_w * 0.015)), max(1, int(new_h * 0.015))
        cropped = warped[inset_h:-inset_h, inset_w:-inset_w] if new_w > 2 * inset_w else warped

        # 5. KOREKCIA ORIENTÁCIE (Sémantická rotácia)
        final_img, method, ang = rotate_photo(cropped, obj_model)

        photo_count += 1
        save_path = os.path.join(OUTPUT_DIR, f"{base_name}_{photo_count}.jpg")
        cv2.imwrite(save_path, final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"  Foto {photo_count} - Rotácia: {ang}° Metóda: {method} - Uložené")

    print(f"{'=' * 60}")
    print(f"\nHOTOVO! Extrahovaných: {photo_count} fotiek")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    crop_photos("../../photo_dataset/images/test/testsnimka.png")

