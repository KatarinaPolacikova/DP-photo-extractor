import os
import json
import torch
from ultralytics import YOLO
from datetime import datetime


class PhotoSegmentationTrainer:
    def __init__(self, data_yaml='photo_dataset/dataset.yaml'):
        self.data_yaml = data_yaml
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.project_name = 'trained_models'
        self.run_name = 'photo_segmentation_model_yolo11s_new'

        print("\n" + "=" * 60)
        print("DEVICE INFO")
        print("=" * 60)
        print(f"Device: {self.device.upper()}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 60 + "\n")

    def train(
        self,
        model_size='s',
        epochs=60,
        img_size=960,
        batch_size=2,
        patience=20,
    ):
        print("\n" + "=" * 60)
        print("ZAČÍNAME TRÉNOVANIE...")
        print("=" * 60)

        model_name = f'yolo11{model_size}-seg.pt'
        self.model = YOLO(model_name)

        self.model.train(
            data=os.path.abspath(self.data_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=self.device,
            optimizer='AdamW',
            patience=patience,
            rect=False,

            overlap_mask=True,  # Pomáha, ak sa fotky na skene prekrývajú
            mask_ratio=2,  # Vyššie rozlíšenie masiek (pomer k imgsz)

            amp=True,
            cos_lr=True,
            val=True,
            save=True,
            project=self.project_name,
            name=self.run_name,
            exist_ok=True,
            seed=42,
            plots=True,
            workers=0,

            # GEOMETRICKÉ AUGMENTÁCIE
            degrees=15.0,
            shear=5.0,
            perspective=0.001,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0
        )

        print("\nTRÉNOVANIE DOKONČENÉ")
        print("=" * 60)

    def evaluate(self):
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        base_path = os.path.join('../../runs', 'segment', self.project_name, self.run_name)
        weights_path = os.path.join(base_path, 'weights', 'best.pt')

        if not os.path.exists(weights_path):
            base_path = os.path.join(self.project_name, self.run_name)
            weights_path = os.path.join(base_path, 'weights', 'best.pt')

        if not os.path.exists(weights_path):
            print(f"ERROR: Model weights not found at: {weights_path}")
            return

        model = YOLO(weights_path)


        print("\nRunning Validation...")
        val_metrics = model.val(split='val')

        print("\nRunning Test...")
        test_metrics = model.val(split='test', rect=True, imgsz=1024)

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model_source": weights_path,
            "validation": {
                "mAP50": float(val_metrics.seg.map50),
                "mAP50-95": float(val_metrics.seg.map),
                "precision": float(val_metrics.seg.mp),
                "recall": float(val_metrics.seg.mr),
            },
            "test": {
                "mAP50": float(test_metrics.seg.map50),
                "mAP50-95": float(test_metrics.seg.map),
                "precision": float(test_metrics.seg.mp),
                "recall": float(test_metrics.seg.mr),
            }
        }

        metrics_file = os.path.join(base_path, 'evaluation_metrics.json')

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        print(f"\nMetriky uložené do: {metrics_file}")
        print(f"Model uložený do: {weights_path}")
        print("=" * 60)


def main():
    trainer = PhotoSegmentationTrainer(
        data_yaml='photo_dataset/dataset.yaml'
    )

    trainer.train(
        model_size='s',
        epochs=60,
        img_size=1024,
        batch_size=4,
        patience=20
    )

    trainer.evaluate()

    print("\nHOTOVO!")


if __name__ == "__main__":
    main()

