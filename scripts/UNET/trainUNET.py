import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import IoU, Precision, Recall, Fscore
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# DEFINÍCIA DÁTOVEJ SADY (DATASET CLASS)
# =============================================================================
class PhotoDataset(Dataset):
    """
        Vlastná trieda Datasetu, ktorá zabezpečuje párovanie vstupných skenov
        s ich príslušnými binárnymi maskami.
    """
    def __init__(self, root, subset, transforms=None):
        self.img_dir = os.path.join(root, 'images', subset)
        self.mask_dir = os.path.join(root, 'masks', subset)
        # Zoradenie zabezpečí, že image[i] bude vždy zodpovedať mask[i]
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Načítanie obrazu a masky
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, os.path.splitext(self.images[idx])[0] + '.png')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Maska sa načítava v odtieňoch sivej a normalizuje sa do rozsahu [0, 1]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32) / 255.0

        # Aplikácia augmentácií (napr. rotácia, zmena jasu, normalizácia)
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.unsqueeze(0)


# =============================================================================
# VIZUALIZÁCIA PREDPOVEDÍ (INFERENCIA)
# =============================================================================
def visualize_results(model, dataset, device, n=3):
    """
        Pomocná funkcia na vizuálnu kontrolu úspešnosti modelu po trénovaní.
        Porovnáva originál, skutočnú masku (Ground Truth) a odhad modelu.
    """
    model.eval()
    plt.figure(figsize=(12, n * 4))
    for i in range(n):
        img, mask = dataset[i]
        img_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)
            # Thresholding: pravdepodobnosť nad 0.5 sa považuje za objekt (fotku)
            pred = (pred > 0.5).float().cpu().squeeze().numpy()

        # Spätná denormalizácia obrazu pre správne zobrazenie v matplotlib
        img_show = img.permute(1, 2, 0).cpu().numpy()
        img_show = (img_show * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        img_show = np.clip(img_show, 0, 1)

        plt.subplot(n, 3, i * 3 + 1)
        plt.imshow(img_show)
        plt.title("Originálny sken")
        plt.axis('off')

        plt.subplot(n, 3, i * 3 + 2)
        plt.imshow(mask.squeeze(), cmap='gray')
        plt.title("Skutočná maska (GT)")
        plt.axis('off')

        plt.subplot(n, 3, i * 3 + 3)
        plt.imshow(pred, cmap='gray')
        plt.title("Predikcia modelu")
        plt.axis('off')

    plt.tight_layout()
    os.makedirs("../../runs/unet", exist_ok=True)
    plt.savefig("../../runs/unet/prediction_samples.png")
    plt.close()


# =============================================================================
# TRÉNOVACÍ PROCES
# =============================================================================
def train_model():
    # Nastavenia hyperparametrov
    ROOT_PATH = "../../photo_dataset"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 30
    BATCH_SIZE = 8
    LR = 1e-4

    # AUGMENTÁCIA DÁT
    # Pomáha modelu generalizovať a predchádzať overfittingu (preučeniu)
    train_transform = A.Compose([
        A.Resize(512, 512),
        A.Rotate(limit=45, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Príprava Loaderov (zabezpečujú paralelný prísun dát do GPU)
    train_ds = PhotoDataset(ROOT_PATH, "train", transforms=train_transform)
    val_ds = PhotoDataset(ROOT_PATH, "val", transforms=val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ARCHITEKTÚRA MODELU
    # Použitie U-Netu s predtrénovaným enkodérom ResNet34 (Transfer Learning)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation='sigmoid' # Sigmoid pre binárnu segmentáciu (0-1)
    ).to(DEVICE)

    # OPTIMALIZÁTOR A STRATOVÁ FUNKCIA
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # DiceLoss je odolnejšia voči nevyváženosti tried (keď fotka zaberá malú plochu skenu)
    loss_fn = smp.losses.DiceLoss(mode='binary')

    # Definícia metrík pre priebežné sledovanie kvality počas trénovania
    metrics = {
        "IoU": smp.utils.metrics.IoU(threshold=0.5),
        "Dice": smp.utils.metrics.Fscore(threshold=0.5),
        "Precision": smp.utils.metrics.Precision(threshold=0.5),
        "Recall": smp.utils.metrics.Recall(threshold=0.5)
    }

    history = {
        "train_loss": [], "val_loss": [],
        "val_iou": [], "val_dice": [],
        "val_precision": [], "val_recall": []
    }

    print(f"Začiatok trénovania na: {DEVICE}")

    # TRÉNOVACÍ CYKLUS
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()  # Vynulovanie gradientov z predchádzajúceho kroku
            output = model(images)
            loss = loss_fn(output, masks)
            loss.backward()  # Spätná propagácia chyby
            optimizer.step()  # Aktualizácia váh modelu
            train_loss += loss.item()

        # VALIDÁCIA
        model.eval()
        val_loss = 0
        scores = {k: 0 for k in metrics.keys()}

        with torch.no_grad(): # Vypnutie výpočtu gradientov pre šetrenie pamäte
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                output = model(images)
                val_loss += loss_fn(output, masks).item()

                for name, fn in metrics.items():
                    scores[name] += fn(output, masks).item()

        # Priemery za epochu
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        for k in scores:
            scores[k] /= len(val_loader)

        # Ukladanie do histórie pre neskoršie grafy
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(scores["IoU"])
        history["val_dice"].append(scores["Dice"])
        history["val_precision"].append(scores["Precision"])
        history["val_recall"].append(scores["Recall"])

        print(f"Epoch {epoch + 1:02d} | Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"IoU: {scores['IoU']:.4f} | Dice: {scores['Dice']:.4f} | "
              f"Prec: {scores['Precision']:.4f} | Rec: {scores['Recall']:.4f}")

    # FINÁLNE OPERÁCIE
    print("\n" + "=" * 30)
    print("FINÁLNE VÝSLEDKY NA VALIDAČNEJ MNOŽINE")
    print("=" * 30)
    for k, v in scores.items():
        print(f"{k:10}: {v:.4f}")

    # Uloženie natrénovaných váh modelu
    os.makedirs("../../runs/unet", exist_ok=True)
    torch.save(model.state_dict(), "../../trained_models/trained_unet_model.pt")

    # GENERÁVANIE TRÉNOVACÍCH GRAFOV
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title("Dice Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["val_iou"], label="IoU")
    plt.plot(history["val_dice"], label="Dice")
    plt.title("Accuracy metrics")
    plt.legend()

    plt.tight_layout()
    plt.savefig("../../runs/unet/final_metrics_graph.png")
    plt.close()

    print("\nHOTOVO. Trénovanie dokončené.")
    visualize_results(model, val_ds, DEVICE)


if __name__ == "__main__":
    train_model()
