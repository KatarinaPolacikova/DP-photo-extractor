
import os
import glob
import cv2
import numpy as np


# =============================================================================
# GENERÁTOR BINÁRNYCH MASIEK ZO SÚRADNÍC YOLO POLYGÓNOV
# =============================================================================
def yolo_polygon_to_mask(label_path, h, w):
    """
        Funkcia transformuje anotačný textový súbor vo formáte YOLO (normalizované polygóny)
        do podoby 2D binárnej masky (rastrovaného obrázka), kde pixely patriace objektu majú
        hodnotu 255 (biela farba) a pixely pozadia majú hodnotu 0 (čierna farba).

        Parametre:
        label_path (str): Cesta k anotačnému .txt súboru pre daný obrázok.
        h (int): Výška (height) pôvodného obrazového súboru v pixeloch.
        w (int): Šírka (width) pôvodného obrazového súboru v pixeloch.

        Návratová hodnota:
        numpy.ndarray: Vygenerovaná binárna matica typu uint8 s rozmermi (h, w).
    """
    # Inicializácia prázdnej matice (čierne pozadie) o rozmeroch vstupného obrazu
    mask = np.zeros((h, w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return mask
    with open(label_path) as f:
        for line in f:
            # Rozdelenie riadku; prvý prvok je ID triedy, nasledujú normalizované súradnice bodov polygónu
            parts = line.strip().split()
            # Konverzia reťazcov na čísla s plávajúcou desatinnou čiarkou (vynechávame ID triedy pomocou parts[1:])
            coords = list(map(float, parts[1:]))
            # Spätná transformácia normalizovaných súradníc (interval 0.0 - 1.0) na absolútne pixelové hodnoty.
            # Súradnice osi 'x' (párne indexy) sa násobia šírkou obrazu (w) a osi 'y' (nepárne indexy) jeho výškou (h).
            pts = np.array([[int(coords[i]*w), int(coords[i+1]*h)]
                            for i in range(0, len(coords), 2)], dtype=np.int32)
            # Vykreslenie plného polygónu bielou farbou (hodnota 255) do vopred pripravenej masky
            cv2.fillPoly(mask, [pts], 255)
    return mask


def process_subset(root, subset):
    """
        Funkcia slúžiaca na dávkové spracovanie (batch processing) konkrétnej podmnožiny datasetu
        (napr. 'train', 'val' alebo 'test'). Zabezpečuje iteráciu cez všetky snímky a ukladanie masiek.

        Parametre:
        root (str): Hlavná cesta ku koreňovému adresáru datasetu.
        subset (str): Názov spracovávanej podmnožiny, ktorý definuje vnorený adresár.
    """
    img_dir = os.path.join(root, 'images', subset)
    lbl_dir = os.path.join(root, 'labels', subset)
    out_dir = os.path.join(root, 'masks', subset)
    os.makedirs(out_dir, exist_ok=True)

    # Identifikácia všetkých obrazových súborov v danom podadresári
    images = glob.glob(os.path.join(img_dir, '*'))
    print(f"{subset}: {len(images)} obrázkov.")

    # Iteratívne generovanie rastrovej masky pre každý jeden obrázok v množine
    for img_path in images:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        name = os.path.splitext(os.path.basename(img_path))[0]

        lbl_path = os.path.join(lbl_dir, name + '.txt')
        mask = yolo_polygon_to_mask(lbl_path, h, w)

        # Fyzické uloženie vygenerovanej rastrovej masky vo formáte .png.
        out_path = os.path.join(out_dir, name + '.png')
        cv2.imwrite(out_path, mask)


if __name__ == "__main__":
    root = "../photo_dataset"
    for s in ["train", "val", "test"]:
        process_subset(root, s)

    print(f"Hotovo. Masky vygenerované a uložené v priečinku: {root}/masks.")
