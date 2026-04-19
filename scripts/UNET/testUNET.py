import os
import cv2
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp


class UNetTester:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inicializácia architektúry (rovnaká ako pri trénovaní)
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )

        # Načítanie váh
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        print(f"U-Net model načítaný z: {model_path}")

    def preprocess(self, img_path, target_size=(512, 512)):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]

        img_resized = cv2.resize(img, target_size)

        # Normalizácia
        img_tensor = img_resized.astype(np.float32) / 255.0
        img_tensor = (img_tensor - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device).float()

        return img_tensor, (h_orig, w_orig), img

    def get_individual_masks(self, full_mask):
        """Rozdelí spojenú masku na zoznam samostatných masiek (jedna pre každú fotku)"""
        # Binarizácia
        binary = (full_mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        individual_masks = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:  # Ignoruj malý šum
                continue
            m = np.zeros_like(binary)
            cv2.drawContours(m, [cnt], -1, 1, -1)
            individual_masks.append(m)
        return individual_masks

    def load_gt_masks(self, label_path, w, h):
        """Načíta YOLO polygóny a vytvorí zoznam masiek"""
        if not os.path.exists(label_path): return []
        gt_masks = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                coords = [float(x) for x in parts[1:]]
                pts = np.array([[int(coords[i] * w), int(coords[i + 1] * h)] for i in range(0, len(coords), 2)])
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 1)
                gt_masks.append(mask)
        return gt_masks

    def compute_metrics(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        iou = intersection / union if union > 0 else 0
        dice = (2 * intersection) / (mask1.sum() + mask2.sum()) if (mask1.sum() + mask2.sum()) > 0 else 0
        return iou, dice

    def test_all(self, img_dir, lbl_dir, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        all_metrics = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        for ext in extensions:
            image_paths.extend(list(Path(img_dir).glob(ext)))

        print(f"\nTestujem {len(image_paths)} obrázkov...")

        for img_p in tqdm(image_paths):
            input_tensor, orig_dim, _ = self.preprocess(str(img_p))
            with torch.no_grad():
                pred = self.model(input_tensor)
                pred = pred.cpu().squeeze().numpy()

            pred_full = cv2.resize(pred, (orig_dim[1], orig_dim[0]))
            pred_individual = self.get_individual_masks(pred_full)

            gt_p = Path(lbl_dir) / (img_p.stem + ".txt")
            gt_individual = self.load_gt_masks(str(gt_p), orig_dim[1], orig_dim[0])

            matched_ious = []
            matched_dices = []
            used_gt = set()

            for p_m in pred_individual:
                best_iou = 0
                best_dice = 0
                best_gt_idx = -1
                for idx, g_m in enumerate(gt_individual):
                    if idx in used_gt: continue
                    iou, dice = self.compute_metrics(p_m, g_m)
                    if iou > best_iou:
                        best_iou, best_dice, best_gt_idx = iou, dice, idx

                if best_iou > 0.5:
                    matched_ious.append(best_iou)
                    matched_dices.append(best_dice)
                    used_gt.add(best_gt_idx)

            # Uloženie dát pre každý obrázok
            all_metrics.append({
                'image': img_p.name,
                'num_gt': len(gt_individual),
                'num_pred': len(pred_individual),
                'num_matched': len(matched_ious),
                'avg_iou': np.mean(matched_ious) if matched_ious else 0.0,
                'avg_dice': np.mean(matched_dices) if matched_dices else 0.0
            })

        # --- VÝPOČET CELKOVÝCH ŠTATISTÍK ---
        total_gt = sum(m['num_gt'] for m in all_metrics)
        total_pred = sum(m['num_pred'] for m in all_metrics)
        total_matched = sum(m['num_matched'] for m in all_metrics)

        overall_precision = total_matched / total_pred if total_pred > 0 else 0
        overall_recall = total_matched / total_gt if total_gt > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
            if (overall_precision + overall_recall) > 0 else 0

        all_ious = [m['avg_iou'] for m in all_metrics if m['avg_iou'] > 0]
        all_dices = [m['avg_dice'] for m in all_metrics if m['avg_dice'] > 0]

        summary = {
            "total_images": len(all_metrics),
            "total_gt_photos": total_gt,
            "total_pred_photos": total_pred,
            "total_matched": total_matched,
            "overall_precision": float(overall_precision),
            "overall_recall": float(overall_recall),
            "overall_f1": float(overall_f1),
            "average_iou": float(np.mean(all_ious)) if all_ious else 0.0,
            "average_dice": float(np.mean(all_dices)) if all_dices else 0.0,
            "per_image_results": all_metrics
        }

        summary_file = os.path.join(save_dir, 'overall_metrics_unet.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nSúbor uložený: {summary_file}")


if __name__ == "__main__":
    tester = UNetTester("../../runs/unet/final_unet_model.pt")
    tester.test_all(
        img_dir="../../photo_dataset/images/test",
        lbl_dir="../../photo_dataset/labels/test",
        save_dir="../../test_results_unet"
    )
