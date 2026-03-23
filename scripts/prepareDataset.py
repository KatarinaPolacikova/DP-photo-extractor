import os
import xml.etree.ElementTree as ET
import shutil
import random
from tqdm import tqdm
from collections import Counter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


# --- NASTAVENIA ---
XML_PATH = '../all_annotated_photos/annotations.xml'
IMAGES_SRC = '../all_annotated_photos'
OUTPUT_DIR = '../photo_dataset'

# Rozdelenie dát: 70% tréning, 15% validácia, 15% test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def polygon_area(points):
    """Vypočíta plochu polygónu pomocou Shoelace formula"""
    x = points[::2]
    y = points[1::2]
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1)))


def get_polygon_bounds(points):
    """Vráti bounding box polygónu"""
    x_coords = points[::2]
    y_coords = points[1::2]
    return {
        'x_min': min(x_coords),
        'x_max': max(x_coords),
        'y_min': min(y_coords),
        'y_max': max(y_coords),
        'width': max(x_coords) - min(x_coords),
        'height': max(y_coords) - min(y_coords)
    }


def convert_xml_to_yolo():
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

    images_data = []
    stats = {
        'total_images_in_xml': 0,
        'total_photos_found': 0,
        'photos_per_image': [],
        'image_resolutions': [],
        'polygon_complexities': [],
        'aspect_ratios': [],
        'polygon_areas_normalized': [],  #Veľkosť fotiek vzhľadom na sken
        'edge_photos': 0,
        'small_photos': 0,
        'orientation_distribution': {'portrait': 0, 'landscape': 0, 'square': 0}
    }

    print("Analyzujem XML anotácie a pripravujem štatistiky...")
    for img in root.findall('image'):
        stats['total_images_in_xml'] += 1
        filename = img.get('name')
        w, h = float(img.get('width')), float(img.get('height'))
        stats['image_resolutions'].append((w, h))

        polygons = []
        for poly in img.findall('polygon'):
            points_str = poly.get('points').replace(';', ',').split(',')
            points = [float(p) for p in points_str]
            stats['polygon_complexities'].append(len(points) // 2)

            # ANALÝZA: Aspect ratio a orientácia
            bounds = get_polygon_bounds(points)
            aspect_ratio = bounds['width'] / bounds['height'] if bounds['height'] > 0 else 1
            stats['aspect_ratios'].append(aspect_ratio)

            if aspect_ratio > 1.1:
                stats['orientation_distribution']['landscape'] += 1
            elif aspect_ratio < 0.9:
                stats['orientation_distribution']['portrait'] += 1
            else:
                stats['orientation_distribution']['square'] += 1

            # ANALÝZA: Normalizovaná plocha
            area = polygon_area(points)
            normalized_area = area / (w * h)  # Plocha fotky / plocha skenu
            stats['polygon_areas_normalized'].append(normalized_area)

            if normalized_area < 0.01:  # Menej ako 1% skenu
                stats['small_photos'] += 1

            # ANALÝZA: Detekcia fotiek pri okraji
            margin = 5  # 5px tolerancia
            if (bounds['x_min'] < margin or bounds['y_min'] < margin or
                    bounds['x_max'] > w - margin or bounds['y_max'] > h - margin):
                stats['edge_photos'] += 1

            # YOLO normalizácia
            normalized_points = [str(points[i] / w if i % 2 == 0 else points[i] / h)
                                 for i in range(len(points))]
            polygons.append("0 " + " ".join(normalized_points))

        stats['total_photos_found'] += len(polygons)
        stats['photos_per_image'].append(len(polygons))
        images_data.append({'filename': filename, 'annotations': polygons})

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

    print(f"\nRozdeľujem {total} obrázkov v pomere 70/15/15:")
    save_subset(train_set, 'train')
    save_subset(val_set, 'val')
    save_subset(test_set, 'test')

    # VÝPIS ANALÝZY DATASETU
    print("\n" + "=" * 50)
    print("      ANALÝZA DATASETU")
    print("=" * 50)
    print(f"Celkový počet skenov:             {stats['total_images_in_xml']}")
    print(f"Celkový počet fotiek (anotácií):  {stats['total_photos_found']}")
    print("-" * 50)
    print(f"Priemerný počet fotiek na sken:   {np.mean(stats['photos_per_image']):.2f}")
    print(f"Najviac fotiek na jednom skene:   {np.max(stats['photos_per_image'])}")
    print("-" * 50)

    counts = Counter(stats['photos_per_image'])
    print("DISTRIBÚCIA (počet fotiek : počet skenov):")
    for num_p in sorted(counts.keys()):
        c_img = counts[num_p]
        perc = (c_img / stats['total_images_in_xml']) * 100
        print(f"  {num_p} fotiek: {c_img:4} skenov ({perc:5.1f}%)")

    print("-" * 50)
    res_counts = Counter(stats['image_resolutions'])
    common_res = res_counts.most_common(3)
    print("NAJČASTEJŠIE ROZLÍŠENIA:")
    for res, c in common_res:
        print(f"  {int(res[0])}x{int(res[1])} px: {c}x")

    print("-" * 50)
    print("ORIENTÁCIA FOTOGRAFIÍ:")
    total_photos = stats['total_photos_found']
    for orient, count in stats['orientation_distribution'].items():
        perc = (count / total_photos) * 100
        print(f"  {orient.capitalize():10}: {count:4} ({perc:5.1f}%)")

    print("-" * 50)
    print(f"Priemerný aspect ratio:           {np.mean(stats['aspect_ratios']):.2f}")
    print(f"Priemerná plocha fotky (vs sken): {np.mean(stats['polygon_areas_normalized']) * 100:.1f}%")
    print(f"Priemerný počet bodov polygónu:   {np.mean(stats['polygon_complexities']):.1f}")

    print("-" * 50)
    print("POTENCIÁLNE VÝZVY PRE MODEL:")
    edge_perc = (stats['edge_photos'] / total_photos) * 100
    small_perc = (stats['small_photos'] / total_photos) * 100
    print(f"  Fotky pri okraji skenu:         {stats['edge_photos']:4} ({edge_perc:5.1f}%)")
    print(f"  Veľmi malé fotky (<1% skenu):   {stats['small_photos']:4} ({small_perc:5.1f}%)")

    print("=" * 50)

    # 5. Generovanie YAML
    yaml_content = f"""path: .
train: images/train
val: images/val
test: images/test

names:
  0: photo
"""
    with open(os.path.join(OUTPUT_DIR, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

    # VIZUALIZÁCIA
    fig = plt.figure(figsize=(18, 10))

    # Plot 1: Histogram počtu fotiek
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(stats['photos_per_image'], bins=range(max(stats['photos_per_image']) + 2),
             color='#29c5ab', edgecolor='black', align='left')
    ax1.set_title('Počet fotografií na jeden sken')
    ax1.set_xlabel('Počet fotiek')
    ax1.set_ylabel('Počet skenov')
    ax1.grid(alpha=0.3)

    # Plot 2: Rozlíšenia skenov
    ax2 = plt.subplot(2, 3, 2)
    widths = [r[0] for r in stats['image_resolutions']]
    heights = [r[1] for r in stats['image_resolutions']]
    ax2.scatter(widths, heights, alpha=0.4, color='purple')
    ax2.set_title('Rozlíšenie zdrojových skenov')
    ax2.set_xlabel('Šírka (px)')
    ax2.set_ylabel('Výška (px)')
    ax2.grid(alpha=0.3)

    # Plot 3: Aspect ratio distribúcia
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(stats['aspect_ratios'], bins=30, color='orange', edgecolor='black', alpha=0.7)
    ax3.axvline(1.0, color='red', linestyle='--', label='Štvorcová (1:1)')
    ax3.set_title('Distribúcia aspect ratio fotografií')
    ax3.set_xlabel('Aspect ratio (šírka/výška)')
    ax3.set_ylabel('Počet fotiek')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Plot 4: Orientácia
    ax4 = plt.subplot(2, 3, 4)
    orientations = list(stats['orientation_distribution'].keys())
    counts_orient = [stats['orientation_distribution'][o] for o in orientations]
    colors_orient = ['#3498db', '#e74c3c', '#95a5a6']
    ax4.bar(orientations, counts_orient, color=colors_orient, edgecolor='black')
    ax4.set_title('Distribúcia orientácie fotografií')
    ax4.set_ylabel('Počet fotiek')
    ax4.grid(alpha=0.3, axis='y')

    # Plot 5: Veľkosť fotiek
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist([a * 100 for a in stats['polygon_areas_normalized']], bins=30,
             color='green', edgecolor='black', alpha=0.7)
    ax5.set_title('Veľkosť fotiek vzhľadom na sken')
    ax5.set_xlabel('% plochy skenu')
    ax5.set_ylabel('Počet fotiek')
    ax5.grid(alpha=0.3)

    # Plot 6: Komplexnosť polygónov
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(stats['polygon_complexities'], bins=range(min(stats['polygon_complexities']),
                                                       max(stats['polygon_complexities']) + 2), color='brown',
             edgecolor='black', alpha=0.7)
    ax6.set_title('Komplexnosť anotácií (počet bodov)')
    ax6.set_xlabel('Počet bodov v polygóne')
    ax6.set_ylabel('Počet anotácií')
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('eda_analyza.png', dpi=150)

    print(f"\nGRAF uložený ako 'eda_analyza.png'")
    print(f"YAML súbor pripravený v '{OUTPUT_DIR}/dataset.yaml'")

    # Export štatistík do textového súboru
    with open('../graphsAndStatistics/dataset_statistics.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("ŠTATISTIKY DATASETU\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Celkový počet skenov: {stats['total_images_in_xml']}\n")
        f.write(f"Celkový počet fotografií: {stats['total_photos_found']}\n")
        f.write(f"Priemerný počet fotografií na sken: {np.mean(stats['photos_per_image']):.2f}\n")
        f.write(f"Priemerný pomer strán: {np.mean(stats['aspect_ratios']):.2f}\n")
        f.write(f"Fotografie orientované na šírku: {stats['orientation_distribution']['landscape']}\n")
        f.write(f"Fotografie orientované na výšku: {stats['orientation_distribution']['portrait']}\n")
        f.write(f"Okrajové prípady: {stats['edge_photos']} ({edge_perc:.1f}%)\n")
        f.write(f"Malé fotografie: {stats['small_photos']} ({small_perc:.1f}%)\n")

    print("Štatistiky exportované do 'dataset_statistics.txt'")


if __name__ == "__main__":
    convert_xml_to_yolo()
