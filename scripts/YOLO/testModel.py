import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import random
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.patches as mpatches
from tqdm import tqdm
matplotlib.use('Agg')


class ModelTester:
    """Testovanie natrénovaného modelu"""

    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model neexistuje: {model_path}")

        print(f"Načítavam model: {model_path}")
        self.model = YOLO(model_path)
        self.model_path = model_path
        print(f"Model načítaný!\n")

    def process_mask(self, yolo_mask, w, h):
        """ Zobrazí presnejšiu masku s miernym rozšírením (dilatáciou) aby pokryla celé okraje fotky."""
        # Resize na plné rozlíšenie
        mask_resized = cv2.resize(yolo_mask, (w, h))

        # Thresholding
        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

        # Dilatácia (Nafúknutie)
        # Tento kernel pridá pár pixelov na každú stranu, čím vykompenzuje neistotu modelu na hranách.
        kernel = np.ones((5, 5), np.uint8)
        mask_binary = cv2.dilate(mask_binary, kernel, iterations=2)

        # Morfológia na vyhladenie
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)

        # Nájde kontúry
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return (mask_binary > 127).astype(np.uint8)

        # Najväčšia kontúra
        largest_contour = max(contours, key=cv2.contourArea)

        # Použijeme aproximáciu s malou odchýlkou aby sme zachovali presný tvar, ale vyhladili šum
        epsilon = 0.002 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        smooth_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(smooth_mask, [approx], 1)

        return smooth_mask

    def load_ground_truth(self, label_path, img_width, img_height):
        """Načíta ground truth anotácie z YOLO formátu"""
        if not os.path.exists(label_path):
            return []

        gt_masks = []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                coords = [float(x) for x in parts[1:]]

                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * img_width)
                    y = int(coords[i + 1] * img_height)
                    points.append([x, y])

                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(points)], 1)
                gt_masks.append(mask)

        return gt_masks

    def compute_iou(self, mask1, mask2):
        """Vypočíta Intersection over Union"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0

    def compute_dice(self, mask1, mask2):
        """Vypočíta Dice coefficient"""
        intersection = np.logical_and(mask1, mask2).sum()
        total = mask1.sum() + mask2.sum()
        return 2 * intersection / total if total > 0 else 0.0

    def match_predictions_to_gt(self, pred_masks, gt_masks):
        """Spáruje predikcie s ground truth (greedy matching)"""
        if len(gt_masks) == 0 or len(pred_masks) == 0:
            return []

        iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
        dice_matrix = np.zeros((len(pred_masks), len(gt_masks)))

        for i, pred_mask in enumerate(pred_masks):
            for j, gt_mask in enumerate(gt_masks):
                iou_matrix[i, j] = self.compute_iou(pred_mask, gt_mask)
                dice_matrix[i, j] = self.compute_dice(pred_mask, gt_mask)

        matches = []
        used_pred = set()
        used_gt = set()

        while True:
            max_iou = 0
            max_pos = None

            for i in range(len(pred_masks)):
                if i in used_pred:
                    continue
                for j in range(len(gt_masks)):
                    if j in used_gt:
                        continue
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        max_pos = (i, j)

            if max_pos is None or max_iou < 0.5:
                break

            i, j = max_pos
            matches.append((i, j, iou_matrix[i, j], dice_matrix[i, j]))
            used_pred.add(i)
            used_gt.add(j)

        return matches

    def _evaluate_single_image(self, img_path, test_labels_dir, conf_threshold):
        """Pomocná funkcia na vyhodnotenie jedného obrázka bez vizualizácie"""
        img = cv2.imread(img_path)
        if img is None:
            return None

        h, w = img.shape[:2]
        results = self.model.predict(source=img_path, conf=conf_threshold, verbose=False)[0]

        # Load Ground Truth
        label_path = Path(test_labels_dir) / f"{Path(img_path).stem}.txt"
        gt_masks = self.load_ground_truth(str(label_path), w, h)

        # Process Predictions
        pred_masks = []
        if results.masks is not None:
            for mask in results.masks.data.cpu().numpy():
                clean_mask = self.process_mask(mask, w, h)
                pred_masks.append(clean_mask)

        # Match and calculate metrics
        matches = self.match_predictions_to_gt(pred_masks, gt_masks)

        avg_iou = np.mean([m[2] for m in matches]) if matches else 0.0
        avg_dice = np.mean([m[3] for m in matches]) if matches else 0.0

        return {
            'image': Path(img_path).name,
            'num_gt': len(gt_masks),
            'num_pred': len(pred_masks),
            'num_matched': len(matches),
            'avg_iou': avg_iou,
            'avg_dice': avg_dice
        }

    def test_all_images(self,
                       test_images_dir='photo_dataset/images/test',
                       test_labels_dir='photo_dataset/labels/test',
                       conf_threshold=0.25,
                       save_dir='test_results'):
        os.makedirs(save_dir, exist_ok=True)

        print("="*70)
        print("TESTOVANIE NA VŠETKÝCH OBRÁZKOCH")
        print("="*70)

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        all_images = []
        for ext in image_extensions:
            all_images.extend(Path(test_images_dir).glob(f"*{ext}"))
            all_images.extend(Path(test_images_dir).glob(f"*{ext.upper()}"))

        if len(all_images) == 0:
            print(f"Žiadne obrázky v: {test_images_dir}")
            return None

        print(f"\nPočet testovacích obrázkov: {len(all_images)}\n")

        all_metrics = []

        for img_path in tqdm(all_images, desc="Testujem"):
            metrics = self._evaluate_single_image(
                str(img_path),
                test_labels_dir,
                conf_threshold
            )
            if metrics:
                all_metrics.append(metrics)

        if not all_metrics:
            print("Žiadne výsledky!")
            return None

        total_gt = sum(m['num_gt'] for m in all_metrics)
        total_pred = sum(m['num_pred'] for m in all_metrics)
        total_matched = sum(m['num_matched'] for m in all_metrics)

        overall_precision = total_matched / total_pred if total_pred > 0 else 0
        overall_recall = total_matched / total_gt if total_gt > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
                     if (overall_precision + overall_recall) > 0 else 0

        all_ious = [m['avg_iou'] for m in all_metrics if m['avg_iou'] > 0]
        all_dices = [m['avg_dice'] for m in all_metrics if m['avg_dice'] > 0]

        avg_iou = np.mean(all_ious) if all_ious else 0
        avg_dice = np.mean(all_dices) if all_dices else 0

        print("\n" + "="*70)
        print("CELKOVÉ VÝSLEDKY")
        print("="*70)
        print(f"\nŠTATISTIKA:")
        print(f"   Počet testovaných obrázkov:  {len(all_metrics)}")
        print(f"   GT fotografie celkom:  {total_gt}")
        print(f"   Detekované celkom:     {total_pred}")
        print(f"   Správne detekcie:      {total_matched}")

        print(f"\nDETEKČNÉ METRIKY:")
        print(f"   Precision:  {overall_precision:.4f}")
        print(f"   Recall:     {overall_recall:.4f}")
        print(f"   F1-Score:   {overall_f1:.4f}")

        print(f"\nSEGMENTAČNÉ METRIKY:")
        print(f"   Priemerné IoU:   {avg_iou:.4f}")
        print(f"   Priemerný Dice:  {avg_dice:.4f}")

        print("="*70)

        summary = {
            'total_images': len(all_metrics),
            'total_gt_photos': total_gt,
            'total_pred_photos': total_pred,
            'total_matched': total_matched,
            'overall_precision': float(overall_precision),
            'overall_recall': float(overall_recall),
            'overall_f1': float(overall_f1),
            'average_iou': float(avg_iou),
            'average_dice': float(avg_dice),
            'per_image_results': all_metrics
        }

        summary_file = os.path.join(save_dir, 'overall_metrics_unet.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nSúhrnné metriky: {summary_file}\n")

        return summary

    def test_random_image(self,
                         test_images_dir='photo_dataset/images/test',
                         test_labels_dir='photo_dataset/labels/test',
                         conf_threshold=0.25,
                         save_dir='test_results'):
        """Testovanie na JEDNOM náhodnom obrázku s vizualizáciou"""
        os.makedirs(save_dir, exist_ok=True)

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        all_images = []
        for ext in image_extensions:
            all_images.extend(Path(test_images_dir).glob(f"*{ext}"))
            all_images.extend(Path(test_images_dir).glob(f"*{ext.upper()}"))

        if len(all_images) == 0:
            print(f"Žiadne obrázky v: {test_images_dir}")
            return None

        random_image = random.choice(all_images)

        print("="*70)
        print("TESTOVANIE NA NÁHODNOM OBRÁZKU")
        print("="*70)
        print(f"\nVybraný: {random_image.name}")

        img = cv2.imread(str(random_image))
        if img is None:
            print(f"Chyba pri načítaní!")
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        print(f"   Rozlíšenie: {w}x{h}")

        print(f"\nPredikcia (confidence >= {conf_threshold})...")
        results = self.model.predict(
            source=str(random_image),
            conf=conf_threshold,
            iou=0.7,
            verbose=False
        )[0]

        label_path = Path(test_labels_dir) / f"{random_image.stem}.txt"
        gt_masks = self.load_ground_truth(str(label_path), w, h)

        print(f"   Ground Truth: {len(gt_masks)} fotiek")

        pred_masks = []
        pred_confidences = []
        pred_boxes = []

        if results.masks is not None:
            print(f"   Detekované:   {len(results.masks.data)} fotiek")

            for i, mask in enumerate(results.masks.data.cpu().numpy()):
                clean_mask = self.process_mask(mask, w, h)
                pred_masks.append(clean_mask)

                conf = results.boxes.conf[i].cpu().numpy()
                bbox = results.boxes.xyxy[i].cpu().numpy()
                pred_confidences.append(float(conf))
                pred_boxes.append(bbox)
        else:
            print(f"   Detekovaných 0 fotiek!")

        matches = self.match_predictions_to_gt(pred_masks, gt_masks)

        precision = len(matches) / len(pred_masks) if len(pred_masks) > 0 else 0
        recall = len(matches) / len(gt_masks) if len(gt_masks) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nVÝSLEDKY:")
        print(f"   Správne:        {len(matches)}/{len(gt_masks)}")
        print(f"   False Positive: {len(pred_masks) - len(matches)}")
        print(f"   False Negative: {len(gt_masks) - len(matches)}")
        print(f"\n   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")

        if matches:
            avg_iou = np.mean([m[2] for m in matches])
            avg_dice = np.mean([m[3] for m in matches])
            print(f"\n   Priemerné IoU:  {avg_iou:.4f}")
            print(f"   Priemerný Dice: {avg_dice:.4f}")

            print(f"\nDETAILY:")
            for pred_idx, gt_idx, iou, dice in matches:
                conf = pred_confidences[pred_idx]
                print(f"   Foto {pred_idx+1}: Conf={conf:.3f}, IoU={iou:.3f}, Dice={dice:.3f}")

        self.visualize_results(
            img_rgb, gt_masks, pred_masks, pred_confidences,
            pred_boxes, matches, random_image.name, save_dir
        )

        metrics_data = {
            'image': random_image.name,
            'num_gt': len(gt_masks),
            'num_pred': len(pred_masks),
            'num_matched': len(matches),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'avg_iou': float(avg_iou) if matches else 0.0,
            'avg_dice': float(avg_dice) if matches else 0.0,
        }

        metrics_file = os.path.join(save_dir, f"{random_image.stem}_metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)

        print(f"\nMetriky: {metrics_file}")
        print("="*70 + "\n")

        return metrics_data

    def visualize_results(self, img, gt_masks, pred_masks, confidences,
                          boxes, matches, filename, save_dir):

        fig, axes = plt.subplots(1, 3, figsize=(22, 8))

        # 1. Originál
        axes[0].imshow(img)
        axes[0].set_title('Originálny obrázok', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # 2. Ground Truth
        axes[1].imshow(img)
        overlay_gt = np.zeros_like(img)
        for i, mask in enumerate(gt_masks):
            color = matplotlib.colormaps['tab10'](i % 10)[:3]
            overlay_gt[mask > 0] = [int(c * 255) for c in color]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                polygon = contour.reshape(-1, 2)
                axes[1].add_patch(MplPolygon(polygon, fill=False, edgecolor=color, linewidth=2))

        axes[1].imshow(overlay_gt, alpha=0.3)
        axes[1].set_title(f'Ground Truth ({len(gt_masks)} fotiek)', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # 3. Predikcie
        axes[2].imshow(img)

        MASK_COLOR = (0, 1.0, 0)
        BBOX_COLOR = (0, 0.4, 1.0)

        overlay_pred = np.zeros_like(img)

        for i, mask in enumerate(pred_masks):
            # Zelená maska
            overlay_pred[mask > 0] = [0, 255, 0]

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                polygon = contour.reshape(-1, 2)
                axes[2].add_patch(MplPolygon(polygon, fill=False, edgecolor=MASK_COLOR, linewidth=1.5, alpha=0.8))

            # Bounding Box (modrý)
            x1, y1, x2, y2 = boxes[i]
            rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      fill=False, edgecolor=BBOX_COLOR, linewidth=2, linestyle='-')
            axes[2].add_patch(rect)

            # Label
            conf = confidences[i]
            match_info = ""
            if any(m[0] == i for m in matches):
                match = next(m for m in matches if m[0] == i)
                match_info = f"\nIoU: {match[2]:.2f}"

            label_text = f"Conf: {conf:.2f}{match_info}"
            axes[2].text(x1, y1 - 10, label_text,
                         color='white', fontsize=8, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=BBOX_COLOR, alpha=0.8, edgecolor='none'))

        axes[2].imshow(overlay_pred, alpha=0.3)
        axes[2].set_title(f'Predikcie ({len(pred_masks)} fotiek, {len(matches)} správne)',
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')

        mask_patch = mpatches.Patch(color=MASK_COLOR, label='Maska')
        bbox_patch = mpatches.Patch(color=BBOX_COLOR, label='Bounding Box')
        axes[2].legend(handles=[mask_patch, bbox_patch], loc='lower right', fontsize=10)

        plt.tight_layout()

        output_path = os.path.join(save_dir, f"{Path(filename).stem}_visualization.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Vizualizácia uložená: {output_path}")
        plt.close()


def main():
    MODEL_PATH = '../../runs/segment/trained_models/photo_segmentation_model_yolo8s/weights/best.pt'
    TEST_IMAGES_DIR = '../../photo_dataset/images/test'
    TEST_LABELS_DIR = '../../photo_dataset/labels/test'
    CONF_THRESHOLD = 0.25
    SAVE_DIR = '../../test_results'
    MODE = 'all'

    print(f"\nKONFIGURÁCIA:")
    print(f"   Model:       {MODEL_PATH}")
    print(f"   Test images: {TEST_IMAGES_DIR}")
    print(f"   Confidence:  {CONF_THRESHOLD}")
    print(f"   Režim:       {MODE.upper()}")
    print("="*70 + "\n")

    try:
        tester = ModelTester(MODEL_PATH)
    except FileNotFoundError as e:
        print(f"{e}")
        return

    if MODE == 'all':
        tester.test_all_images(
            test_images_dir=TEST_IMAGES_DIR,
            test_labels_dir=TEST_LABELS_DIR,
            conf_threshold=CONF_THRESHOLD,
            save_dir=SAVE_DIR
        )
    elif MODE == 'random':
        tester.test_random_image(
            test_images_dir=TEST_IMAGES_DIR,
            test_labels_dir=TEST_LABELS_DIR,
            conf_threshold=CONF_THRESHOLD,
            save_dir=SAVE_DIR
        )
    else:
        print(f"Neznámy režim: {MODE}")

    print("HOTOVO!")


if __name__ == "__main__":
    main()
