import os
import xml.etree.ElementTree as ET
import shutil
import random
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')


# NASTAVENIA
XML_PATH = '../all_annotated_photos/annotations.xml'
IMAGES_SRC = '../all_annotated_photos'
OUTPUT_DIR = '../photo_dataset'

# Rozdelenie dát: 70% tréning, 15% validácia, 15% test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def prepare_dataset_convert_xml_to_yolo():
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

    # Vytvorenie slovníka anotácií z XML
    xml_annotations = {}
    print("Načítavam XML anotácie a konvertujem do YOLO formátu...")
    for img in root.findall('image'):
        filename = img.get('name')
        w, h = float(img.get('width')), float(img.get('height'))

        polygons = []
        for poly in img.findall('polygon'):
            points_str = poly.get('points').replace(';', ',').split(',')
            points = [float(p) for p in points_str]

            # YOLO normalizácia
            normalized_points = [str(points[i] / w if i % 2 == 0 else points[i] / h)
                                 for i in range(len(points))]
            polygons.append("0 " + " ".join(normalized_points))

        xml_annotations[filename] = polygons

    # Načítanie všetkých súborov z priečinka
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

    # Rozdelenie dát (Train / Val / Test)
    random.seed(42)
    random.shuffle(images_data)

    total = len(images_data)
    train_idx = int(total * TRAIN_RATIO)
    val_idx = train_idx + int(total * VAL_RATIO)

    train_set = images_data[:train_idx]
    val_set = images_data[train_idx:val_idx]
    test_set = images_data[val_idx:]

    def save_subset(dataset, subset_name):
        for item in tqdm(dataset, desc=f"Ukladám {subset_name}"):
            src_img = os.path.join(IMAGES_SRC, item['filename'])
            if os.path.exists(src_img):
                shutil.copy(src_img, os.path.join(OUTPUT_DIR, 'images', subset_name, item['filename']))
                label_fn = os.path.splitext(item['filename'])[0] + '.txt'
                with open(os.path.join(OUTPUT_DIR, 'labels', subset_name, label_fn), 'w') as f:
                    f.write("\n".join(item['annotations']))

    print(f"\nRozdeľujem {total} súborov v pomere 70/15/15:")
    save_subset(train_set, 'train')
    save_subset(val_set, 'val')
    save_subset(test_set, 'test')

    # Generovanie YAML
    yaml_content = f"""path: .
train: images/train
val: images/val
test: images/test

names:
  0: photo
"""
    with open(os.path.join(OUTPUT_DIR, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

    print(f"\nHotovo! Dataset pripravený v priečinku: {OUTPUT_DIR}")


if __name__ == "__main__":
    prepare_dataset_convert_xml_to_yolo()
