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
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_distance(p1, p2):
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def get_line_score(img):
    """Mriežkové zarovnanie (ideálne pre budovy)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=10)
    score = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            # Body za horizontálne aj vertikálne línie (mriežka)
            if angle < 10 or angle > 170 or (80 < angle < 100):
                score += length
    return score


def crop_photos(input_image_path, output_folder=OUTPUT_DIR):
    os.makedirs(output_folder, exist_ok=True)
    seg_model = YOLO(MODEL_PATH)
    obj_model = YOLO(OBJECT_MODEL_PATH)

    img = cv2.imread(input_image_path)
    if img is None: return

    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    results = \
    seg_model.predict(source=input_image_path, conf=CONF_THRESHOLD, imgsz=1024, retina_masks=True, verbose=False)[0]

    if results.masks is None: return

    photo_count = 0
    for i in range(len(results.masks.xy)):
        points = np.array(results.masks.xy[i], dtype=np.float32)
        rect = cv2.minAreaRect(points)
        box_pts = order_points(cv2.boxPoints(rect))

        w_real = get_distance(box_pts[0], box_pts[1])
        h_real = get_distance(box_pts[0], box_pts[3])
        new_w, new_h = int(w_real), int(h_real)

        dst_pts = np.array([[0, 0], [new_w - 1, 0], [new_w - 1, new_h - 1], [0, new_h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(box_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (new_w, new_h), flags=cv2.INTER_LANCZOS4)

        inset_w, inset_h = max(1, int(new_w * 0.015)), max(1, int(new_h * 0.015))
        cropped = warped[inset_h:-inset_h, inset_w:-inset_w] if new_w > 2 * inset_w else warped

        # ROTÁCIA
        rotation_results = {0: {}, 90: {}, 180: {}, 270: {}}
        for angle in [0, 90, 180, 270]:
            temp = cropped.copy()
            if angle == 90:
                temp = cv2.rotate(temp, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                temp = cv2.rotate(temp, cv2.ROTATE_180)
            elif angle == 270:
                temp = cv2.rotate(temp, cv2.ROTATE_90_COUNTERCLOCKWISE)

            res_p = obj_model.predict(source=temp, conf=0.35, classes=CLASS_PERSON, verbose=False)[0]
            res_o = obj_model.predict(source=temp, conf=0.35, classes=CLASSES_OTHER, verbose=False)[0]

            rotation_results[angle]['p_conf'] = res_p.boxes.conf.cpu().numpy().sum() if len(res_p.boxes) > 0 else 0.0
            rotation_results[angle]['o_conf'] = res_o.boxes.conf.cpu().numpy().sum() if len(res_o.boxes) > 0 else 0.0
            rotation_results[angle]['l_score'] = get_line_score(temp)

        max_p = max(rotation_results[a]['p_conf'] for a in rotation_results)
        max_o = max(rotation_results[a]['o_conf'] for a in rotation_results)

        if max_p > 0.4:
            best_angle = max(rotation_results, key=lambda k: rotation_results[k]['p_conf'])
            method = "Osoby"
        elif max_o > 0.5:
            best_angle = max(rotation_results, key=lambda k: rotation_results[k]['o_conf'])
            method = "Objekty"
        else:
            # Ak sú línie vyrovnané, preferuje pôvodný výrez (0°) pred rotáciou
            best_angle = max(rotation_results, key=lambda k: rotation_results[k]['l_score'])
            method = "Línie"

        if best_angle == 90:
            final = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
        elif best_angle == 180:
            final = cv2.rotate(cropped, cv2.ROTATE_180)
        elif best_angle == 270:
            final = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            final = cropped

        photo_count += 1
        save_path = os.path.join(output_folder, f"{base_name}_photo_{photo_count}.jpg")
        cv2.imwrite(save_path, final, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"Uložené: {save_path} ({method}, {best_angle}°)")


if __name__ == "__main__":
    crop_photos("photo_dataset/images/test/img_0000234.jpg")
