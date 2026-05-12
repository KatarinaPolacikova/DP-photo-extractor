import os
import json
import torch
from ultralytics import YOLO
from datetime import datetime


class PhotoSegmentationTrainer:
    """
        Trieda zabezpečujúca kompletný životný cyklus modelu: konfiguráciu,
        trénovanie a následnú evaluáciu na segmentáciu fotografií v skenoch.
    """
    def __init__(self, data_yaml='../../photo_dataset/dataset.yaml'):
        self.data_yaml = data_yaml
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.project_name = '../../runs/segment/trained_models'
        self.run_name = 'photo_segmentation_model_yolo11s_new'

        print("\n" + "=" * 60)
        print("HARDVÉROVÁ KONFIGURÁCIA")
        print("=" * 60)
        print(f"Zariadenie: {self.device.upper()}")
        if self.device == 'cuda':
            print(f"Detekované GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 60 + "\n")

    def train(
        self,
        model_size='s',
        epochs=60,
        img_size=960,
        batch_size=2,
        patience=20,
    ):

        """
            Spúšťa trénovací proces s definovanými hyperparametrami.
            Využíva architektúru YOLOv11 s podporou segmentačných masiek.
        """
        print("\n" + "=" * 60)
        print("INICIALIZÁCIA TRÉNOVACIEHO PROCESU...")
        print("=" * 60)

        # Načítanie predtrénovaného modelu YOLO11 (Transfer Learning)
        model_name = f'yolo11{model_size}-seg.pt'
        self.model = YOLO(model_name)

        # Spustenie tréningu s optimalizáciou pre segmentáciu dokumentov/fotiek
        self.model.train(
            data=os.path.abspath(self.data_yaml),
            epochs=epochs,  # Maximálny počet trénovacích epoch
            imgsz=img_size,  # Rozlíšenie vstupných obrázkov
            batch=batch_size,  # Veľkosť trénovacej dávky
            device=self.device,
            optimizer='AdamW',  # Moderný optimalizátor s adaptívnym učením a váhovou penalizáciou
            patience=patience,  # Early Stopping: zastavenie ak sa model nezlepšuje (prevencia overfittingu)
            rect=False,  # Zakázanie obdĺžnikového tréningu pre lepšiu generalizáciu pri rotáciách

            # PARAMETRE PRE SEGMENTÁCIU
            overlap_mask=True,  # Povolenie prekrývajúcich sa masiek (kritické pre tesne susediace fotky)
            mask_ratio=2,  # Zvýšenie granularity segmentačnej masky voči rozlíšeniu obrazu

            amp=True,  # Automatická zmiešaná presnosť (rýchlejšie trénovanie na GPU)
            cos_lr=True,  # Cosine Learning Rate Scheduler pre plynulejšiu konvergenciu
            val=True,  # Validácia po každej epoche
            save=True,  # Ukladanie kontrolných bodov (checkpoints)
            project=self.project_name,
            name=self.run_name,
            exist_ok=True,
            seed=42,  # Reprodukovateľnosť výsledkov
            plots=True,  # Generovanie grafov (strata, mAP, matica zámen)
            workers=0,  # Počet paralelných procesov pre načítanie dát

            # GEOMETRICKÉ AUGMENTÁCIE (Robustnosť voči rôznym polohám skenovania)
            degrees=15.0,  # Náhodná rotácia (±15 stupňov)
            shear=5.0,  # Skosenie obrazu
            perspective=0.001,  # Simulácia perspektívneho skreslenia
            flipud=0.5,  # Vertikálne preklopenie
            fliplr=0.5,  # Horizontálne preklopenie
            mosaic=1.0  # Kombinovanie viacerých obrázkov do jedného pre lepší kontext
        )

        print("\nTRÉNOVANIE DOKONČENÉ")
        print("=" * 60)

    def evaluate(self):
        """
            Vykonáva finálnu evaluáciu naučeného modelu na validačnej a testovacej množine.
            Výsledky ukladá vo formáte JSON pre ďalšiu analýzu.
        """
        print("\n" + "=" * 60)
        print("EVALUÁCIA MODELU")
        print("=" * 60)

        # Lokalizácia najlepších natrénovaných váh
        base_path = os.path.join('../../runs', 'segment', self.project_name, self.run_name)
        weights_path = os.path.join(base_path, 'weights', 'trained_yolov8_model.pt')

        if not os.path.exists(weights_path):
            base_path = os.path.join(self.project_name, self.run_name)
            weights_path = os.path.join(base_path, 'weights', 'trained_yolov8_model.pt')

        if not os.path.exists(weights_path):
            print(f"CHYBA: Váhy modelu neboli nájdené na: {weights_path}")
            return

        # Načítanie natrénovaného modelu pre inferenciu
        model = YOLO(weights_path)


        print("\nSpúšťam validáciu...")
        val_metrics = model.val(split='val')

        print("\nSpúšťam testovanie...")
        test_metrics = model.val(split='test', rect=True, imgsz=1024)

        # Agregácia kľúčových metrík pre segmentáciu (mAP - mean Average Precision)
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model_source": weights_path,
            "validation": {
                "mAP50": float(val_metrics.seg.map50),     # Presnosť pri 50% prekrytí (IoU)
                "mAP50-95": float(val_metrics.seg.map),    # Priemerná presnosť naprieč rôznymi prahmi IoU
                "precision": float(val_metrics.seg.mp),    # Schopnosť modelu neoznačiť pozadie za fotku
                "recall": float(val_metrics.seg.mr),       # Schopnosť modelu nájsť všetky fotky
            },
            "test": {
                "mAP50": float(test_metrics.seg.map50),
                "mAP50-95": float(test_metrics.seg.map),
                "precision": float(test_metrics.seg.mp),
                "recall": float(test_metrics.seg.mr),
            }
        }

        # Export metrík
        metrics_file = os.path.join(base_path, 'evaluation_metrics.json')

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        print(f"\nMetriky uložené do: {metrics_file}")
        print(f"Model uložený do: {weights_path}")
        print("=" * 60)


def main():
    trainer = PhotoSegmentationTrainer(
        data_yaml='../../photo_dataset/dataset.yaml'
    )

    # Spustenie procesu učenia s definovanými parametrami rozlíšenia a výkonu
    trainer.train(
        model_size='s',  # 's' predstavuje Small verziu - balans medzi rýchlosťou a presnosťou
        epochs=60,
        img_size=1024,
        batch_size=4,
        patience=20
    )

    # Vyhodnotenie kvality modelu
    trainer.evaluate()

    print("\nHOTOVO! PROCES ÚSPEŠNE UKONČENÝ.")


if __name__ == "__main__":
    main()

