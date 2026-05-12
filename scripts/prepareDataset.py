import os
import xml.etree.ElementTree as ET
import shutil
import random
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

# =============================================================================
# KONFIGURÁCIA ZÁKLADNÝCH CIEST A PARAMETROV
# =============================================================================
XML_PATH = '../all_annotated_photos/annotations.xml'
IMAGES_SRC = '../all_annotated_photos'
OUTPUT_DIR = '../photo_dataset'

# Definícia pomerov pre rozdelenie datasetu: 70 % trénovacia, 15 % validačná, 15 % testovacia množina
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def prepare_dataset_convert_xml_to_yolo():
    """
        Funkcia na transformáciu datasetu z XML (CVAT) do formátu YOLO.
        Zabezpečuje:
        1. Vytvorenie priečinkovej štruktúry.
        2. Normalizáciu súradníc polygonov (0.0 až 1.0).
        3. Rozdelenie dát na Train, Val a Test množiny.
        4. Vygenerovanie konfiguračného súboru dataset.yaml.
        """
    # Príprava štruktúry priečinkov
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    subfolders = [
        'images/train', 'images/val', 'images/test',
        'labels/train', 'labels/val', 'labels/test'
    ]
    for sub in subfolders:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

    # Načítanie a analýza XML
    if not os.path.exists(XML_PATH):
        print(f"Chyba: Súbor {XML_PATH} neexistuje!")
        return

    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    # Vytvorenie slovníka anotácií z XML: { 'meno_obrazku.jpg': ['yolo_format_anotacia', ...] }
    xml_annotations = {}
    print("Prebieha načítavanie XML anotácií a ich konverzia do YOLO formátu...")
    for img in root.findall('image'):
        filename = img.get('name')
        # Získanie rozmerov obrázka pre následnú normalizáciu súradníc
        w, h = float(img.get('width')), float(img.get('height'))

        polygons = []
        for poly in img.findall('polygon'):
            # Extrakcia bodov polygónu (uložené ako "x1,y1;x2,y2;...") a ich transformácia na list čísel
            points_str = poly.get('points').replace(';', ',').split(',')
            points = [float(p) for p in points_str]

            # Normalizácia súradníc pre YOLO (súradnice v intervale <0.0, 1.0>)
            # Párne indexy zoznamu reprezentujú os 'x' (delíme šírkou), nepárne os 'y' (delíme výškou)
            normalized_points = [str(points[i] / w if i % 2 == 0 else points[i] / h)
                                 for i in range(len(points))]
            # Formát YOLO pre segmentáciu: "<class_id> <x1> <y1> <x2> <y2> ..."
            # Jedna trieda s ID 0
            polygons.append("0 " + " ".join(normalized_points))

        xml_annotations[filename] = polygons

    # Detekcia a spracovanie všetkých obrazových dát
    # Nájdenie všetkých relevantných súborov v zdrojovom priečinku
    all_images = [f for f in os.listdir(IMAGES_SRC) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Nájdených {len(all_images)} súborov v priečinku.")

    images_data = []

    # Priradenie anotácií k obrázkom
    for filename in all_images:
        if filename in xml_annotations:
            # Obrázok má anotácie (fotky)
            images_data.append({'filename': filename, 'annotations': xml_annotations[filename]})
        else:
            # Obrázok je prázdny sken (vytvorí sa preň prázdny label súbor pre YOLO)
            images_data.append({'filename': filename, 'annotations': []})

    # Rozdelenie datasetu (Train / Val / Test)
    random.seed(42)
    random.shuffle(images_data)

    total = len(images_data)
    train_idx = int(total * TRAIN_RATIO)
    val_idx = train_idx + int(total * VAL_RATIO)

    train_set = images_data[:train_idx]
    val_set = images_data[train_idx:val_idx]
    test_set = images_data[val_idx:]

    def save_subset(dataset, subset_name):
        """
        Pomocná vnorená funkcia slúžiaca na fyzické uloženie podmnožiny datasetu.
        Vykonáva kopírovanie obrazových súborov a generovanie príslušných anotačných textových súborov.

        Parametre:
        dataset (list): Zoznam mapovacích slovníkov (súbor - anotácia) pre danú množinu.
        subset_name (str): Označenie cieľového adresára ('train', 'val' alebo 'test').
        """
        for item in tqdm(dataset, desc=f"Ukladám {subset_name}"):
            src_img = os.path.join(IMAGES_SRC, item['filename'])
            if os.path.exists(src_img):
                shutil.copy(src_img, os.path.join(OUTPUT_DIR, 'images', subset_name, item['filename']))
                # Zmena prípony na .txt a zápis anotácií
                label_fn = os.path.splitext(item['filename'])[0] + '.txt'
                with open(os.path.join(OUTPUT_DIR, 'labels', subset_name, label_fn), 'w') as f:
                    f.write("\n".join(item['annotations']))

    print(f"\nRealizujem rozdelenie {total} súborov v pomere 70/15/15:")
    save_subset(train_set, 'train')
    save_subset(val_set, 'val')
    save_subset(test_set, 'test')

    # Generovanie konfiguračného YAML súboru
    yaml_content = f"""path: .
                   train: images/train
                   val: images/val
                   test: images/test
                    
                   names:
                      0: photo
                   """
    with open(os.path.join(OUTPUT_DIR, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

    print(f"\nHotovo! Vygenerovaný dataset je pripravený v priečinku: {OUTPUT_DIR}")


if __name__ == "__main__":
    prepare_dataset_convert_xml_to_yolo()
