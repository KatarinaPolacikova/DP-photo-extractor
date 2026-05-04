import os
import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

try:
    from ultralytics import YOLO
    yolo_model = YOLO('../yoloModels/yolov8n.pt')
    USE_YOLO = True
except ImportError:
    print("Knižnica 'ultralytics' nie je nainštalovaná. Pre detekciu osôb spusti: pip install ultralytics")
    USE_YOLO = False

matplotlib.use('Agg')

# NASTAVENIA
XML_PATH = '../all_annotated_photos/annotations.xml'
IMAGES_SRC = '../all_annotated_photos'  # Priečinok so samotnými skenmi
OUTPUT_DIR_STATS = '../graphsAndStatistics'

os.makedirs(OUTPUT_DIR_STATS, exist_ok=True)


def polygon_area(points):
    x = points[::2]
    y = points[1::2]
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1)))


def get_polygon_bounds(points):
    x_coords = points[::2]
    y_coords = points[1::2]
    return {
        'x_min': min(x_coords), 'x_max': max(x_coords),
        'y_min': min(y_coords), 'y_max': max(y_coords),
        'width': max(x_coords) - min(x_coords),
        'height': max(y_coords) - min(y_coords)
    }


def calculate_rotation_angle(points):
    """Vypočíta uhol natočenia fotky v stupňoch"""
    pts = np.array(points).reshape(-1, 2).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    return abs(angle)


def analyze_dataset():
    if not os.path.exists(XML_PATH):
        print(f"Chyba: Súbor {XML_PATH} neexistuje!")
        return

    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    stats = {
        'total_scans_processed': 0,
        'total_photos_found': 0,
        'photos_per_image': [],
        'image_resolutions': [],
        'polygon_complexities': [],
        'aspect_ratios': [],
        'polygon_areas_normalized': [],
        'edge_photos': 0,
        'small_photos': 0,
        'orientation_distribution': {'portrait': 0, 'landscape': 0, 'square': 0},
        'angles': [],
        'content_tags': Counter(),
        'tilted_photos': 0
    }

    # Vytvorenie slovníka anotácií z XML
    xml_annotations = {}
    for img in root.findall('image'):
        filename = img.get('name')
        w, h = float(img.get('width')), float(img.get('height'))
        polygons = []
        for poly in img.findall('polygon'):
            points_str = poly.get('points').replace(';', ',').split(',')
            polygons.append([float(p) for p in points_str])
        xml_annotations[filename] = {'width': w, 'height': h, 'polygons': polygons}

    # Načítanie všetkých súborov z priečinka
    all_images = [f for f in os.listdir(IMAGES_SRC) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Nájdených {len(all_images)} súborov v priečinku.")
    print("Spúšťam analýzu datasetu ...")

    # Analýza všetkých súborov
    for img_name in tqdm(all_images, desc="Spracovanie skenov"):
        img_path = os.path.join(IMAGES_SRC, img_name)
        stats['total_scans_processed'] += 1

        if img_name in xml_annotations:
            # Sken obsahuje anotácie (fotky)
            w = xml_annotations[img_name]['width']
            h = xml_annotations[img_name]['height']
            polygons = xml_annotations[img_name]['polygons']
            cv_img = cv2.imread(img_path) if USE_YOLO else None
        else:
            # Prázdny sken (Background)
            cv_img = cv2.imread(img_path)
            if cv_img is None:
                stats['total_scans_processed'] -= 1
                continue
            h, w = cv_img.shape[:2]
            polygons = []
            stats['content_tags']['Prázdne skeny (Background)'] += 1

        stats['image_resolutions'].append((w, h))

        poly_count = 0
        for points in polygons:
            stats['polygon_complexities'].append(len(points) // 2)

            bounds = get_polygon_bounds(points)
            aspect_ratio = bounds['width'] / bounds['height'] if bounds['height'] > 0 else 1
            stats['aspect_ratios'].append(aspect_ratio)

            if aspect_ratio > 1.1:
                stats['orientation_distribution']['landscape'] += 1
            elif aspect_ratio < 0.9:
                stats['orientation_distribution']['portrait'] += 1
            else:
                stats['orientation_distribution']['square'] += 1

            angle = calculate_rotation_angle(points)
            stats['angles'].append(angle)
            if angle > 2.0: stats['tilted_photos'] += 1

            area = polygon_area(points)
            norm_area = area / (w * h)
            stats['polygon_areas_normalized'].append(norm_area)
            if norm_area < 0.01: stats['small_photos'] += 1

            margin = 5
            if (bounds['x_min'] < margin or bounds['y_min'] < margin or
                    bounds['x_max'] > w - margin or bounds['y_max'] > h - margin):
                stats['edge_photos'] += 1

            # YOLO DETEKCIA OSÔB
            if cv_img is not None and USE_YOLO:
                # Orezanie fotky zo skenu
                y1, y2 = max(0, int(bounds['y_min'])), min(cv_img.shape[0], int(bounds['y_max']))
                x1, x2 = max(0, int(bounds['x_min'])), min(cv_img.shape[1], int(bounds['x_max']))

                if y2 > y1 and x2 > x1:
                    crop = cv_img[y1:y2, x1:x2]
                    # Spustenie YOLO inferencie
                    results = yolo_model(crop, verbose=False)
                    # Získanie zoznamu detegovaných tried (0 = person v COCO datasete)
                    detected_classes = results[0].boxes.cls.cpu().numpy()

                    if 0 in detected_classes:
                        stats['content_tags']['Osoby'] += 1
                    else:
                        stats['content_tags']['Iné / Bez osôb'] += 1
                else:
                    stats['content_tags']['Chyba orezu'] += 1
            elif not USE_YOLO:
                stats['content_tags']['Neznáme (YOLO neaktívne)'] += 1

            poly_count += 1

        stats['total_photos_found'] += poly_count
        stats['photos_per_image'].append(poly_count)

    # GENEROVANIE TEXTOVÉHO REPORTU
    txt_path = os.path.join(OUTPUT_DIR_STATS, 'dataset_statistics.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        def write_and_print(text):
            print(text)
            f.write(text + "\n")

        write_and_print("\n" + "=" * 50)
        write_and_print("      ANALÝZA DATASETU")
        write_and_print("=" * 50)
        write_and_print(f"Celkový počet skenov:             {stats['total_scans_processed']}")
        write_and_print(f"Celkový počet fotiek (anotácií):  {stats['total_photos_found']}")
        write_and_print("-" * 50)
        write_and_print(f"Priemerný počet fotiek na sken:   {np.mean(stats['photos_per_image']):.2f}")
        write_and_print(f"Najviac fotiek na jednom skene:   {np.max(stats['photos_per_image'])}")

        write_and_print("-" * 50)
        counts = Counter(stats['photos_per_image'])
        write_and_print("DISTRIBÚCIA (počet fotiek : počet skenov):")
        for num_p in sorted(counts.keys()):
            c_img = counts[num_p]
            perc = (c_img / stats['total_scans_processed']) * 100 if stats['total_scans_processed'] > 0 else 0
            write_and_print(f"  {num_p} fotiek: {c_img:4} skenov ({perc:5.1f}%)")

        write_and_print("-" * 50)
        write_and_print("GEOMETRIA A ZAKRIVENIE (ROTÁCIA):")
        if stats['angles']:
            write_and_print(f"Priemerný uhol natočenia:         {np.mean(stats['angles']):.2f}°")
        write_and_print(
            f"Počet výrazne rotovaných (>2°):   {stats['tilted_photos']} ({(stats['tilted_photos'] / max(1, stats['total_photos_found'])) * 100:.1f}%)")
        if stats['aspect_ratios']:
            write_and_print(f"Priemerný aspect ratio:           {np.mean(stats['aspect_ratios']):.2f}")

        write_and_print("-" * 50)
        write_and_print("ANALÝZA OBSAHU (YOLO KLASIFIKÁCIA & BACKGROUND):")

        # Pre percentá oddelíme fotky a prázdne skeny
        for tag, count in stats['content_tags'].items():
            if tag == 'Prázdne skeny (Background)':
                perc = (count / stats['total_scans_processed']) * 100 if stats['total_scans_processed'] > 0 else 0
                write_and_print(f"  {tag:26}: {count:4}x ({perc:.1f}% zo všetkých skenov)")
            else:
                perc = (count / stats['total_photos_found']) * 100 if stats['total_photos_found'] > 0 else 0
                write_and_print(f"  {tag:26}: {count:4}x ({perc:.1f}% zo všetkých fotiek)")

        write_and_print("-" * 50)
        write_and_print("POTENCIÁLNE VÝZVY PRE MODEL:")
        write_and_print(
            f"  Fotky pri okraji skenu:         {stats['edge_photos']} ({(stats['edge_photos'] / max(1, stats['total_photos_found'])) * 100:.1f}%)")
        write_and_print(
            f"  Veľmi malé fotky (<1% skenu):   {stats['small_photos']} ({(stats['small_photos'] / max(1, stats['total_photos_found'])) * 100:.1f}%)")
        write_and_print("=" * 50)

    # VIZUALIZÁCIA
        # Plot 1: Histogram počtu fotiek
        plt.figure(figsize=(8, 6))
        plt.hist(stats['photos_per_image'], bins=range(max(stats['photos_per_image'] + [0]) + 2),
                 color='#A3FFFF', edgecolor='black', align='left')
        plt.title('Distribúcia počtu extrahovaných objektov vo vstupných obrazoch')
        plt.xlabel('Počet detegovaných objektov (n)')
        plt.ylabel('Absolútna početnosť (vstupné obrazy)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_STATS, '01_photos_per_scan.png'), dpi=150)
        plt.close()

        # Plot 2: Rozlíšenia skenov
        plt.figure(figsize=(8, 6))
        widths = [r[0] for r in stats['image_resolutions']]
        heights = [r[1] for r in stats['image_resolutions']]
        plt.scatter(widths, heights, alpha=0.6, color='#D0B3FF', edgecolor='black', linewidth=0.5)
        plt.title('Priestorové rozlíšenie vstupných obrazových dát')
        plt.xlabel('Šírka (px)')
        plt.ylabel('Výška (px)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_STATS, '02_scan_resolutions.png'), dpi=150)
        plt.close()

        # Plot 3: Aspect ratio distribúcia
        if stats['aspect_ratios']:
            plt.figure(figsize=(8, 6))
            plt.hist(stats['aspect_ratios'], bins=30, color='#FFD1A3', edgecolor='black', alpha=0.9)
            plt.axvline(1.0, color='#8A2BE2', linestyle='--', linewidth=1.5)
            plt.title('Distribúcia pomeru strán extrahovaných objektov')
            plt.xlabel('Pomer strán (šírka/výška)')
            plt.ylabel('Absolútna početnosť (objekty)')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR_STATS, '03_aspect_ratios.png'), dpi=150)
            plt.close()

        # Plot 4: Orientácia
        plt.figure(figsize=(8, 6))
        orient_map = {'portrait': 'Orientácia na výšku', 'landscape': 'Orientácia na šírku', 'square': 'Štvorec'}
        x_labels = [orient_map.get(k, k) for k in stats['orientation_distribution'].keys()]

        plt.bar(x_labels, list(stats['orientation_distribution'].values()),
                color=['#A3C2FF', '#FFB3FF', '#B4FFB4'], edgecolor='black')
        plt.title('Kategorizácia objektov podľa priestorovej orientácie')
        plt.ylabel('Absolútna početnosť (objekty)')
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_STATS, '04_orientation.png'), dpi=150)
        plt.close()

        # Plot 5: Veľkosť fotiek
        if stats['polygon_areas_normalized']:
            plt.figure(figsize=(8, 6))
            plt.hist([a * 100 for a in stats['polygon_areas_normalized']], bins=30,
                     color='#B4FFB4', edgecolor='black', alpha=0.9)
            plt.title('Relatívna plošná veľkosť objektov voči vstupnému obrazu')
            plt.xlabel('Pomer plochy objektu k celkovému obrazu (%)')
            plt.ylabel('Absolútna početnosť (objekty)')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR_STATS, '05_photo_sizes.png'), dpi=150)
            plt.close()

        # Plot 6: Komplexnosť polygónov
        if stats['polygon_complexities']:
            plt.figure(figsize=(8, 6))
            plt.hist(stats['polygon_complexities'],
                     bins=range(min(stats['polygon_complexities']), max(stats['polygon_complexities']) + 2),
                     color='#FFB3FF', edgecolor='black', align='left', alpha=0.9)
            plt.title('Geometrická komplexnosť anotačných polygónov')
            plt.xlabel('Počet vrcholov polygónu (n)')
            plt.ylabel('Absolútna početnosť (anotácie)')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR_STATS, '06_polygon_complexity.png'), dpi=150)
            plt.close()

        # Plot 7: Uhly rotácie
        if stats['angles']:
            plt.figure(figsize=(8, 6))
            plt.hist(stats['angles'], bins=30, color='#FFFFB3', edgecolor='black', alpha=0.9)
            plt.title('Distribúcia uhla natočenia extrahovaných objektov')
            plt.xlabel('Uhol natočenia (°)')
            plt.ylabel('Absolútna početnosť (objekty)')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR_STATS, '07_rotation_angles.png'), dpi=150)
            plt.close()

        # Plot 8: Pie chart obsahu
        labels = list(stats['content_tags'].keys())
        sizes = list(stats['content_tags'].values())
        if sum(sizes) > 0:
            plt.figure(figsize=(8, 8))
            pastel_pie_colors = ['#D0B3FF', '#A3C2FF', '#B4FFB4', '#FFB3FF', '#FFD1A3']
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,
                    colors=pastel_pie_colors[:len(labels)], wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
            plt.title('Kategorizácia sémantického obsahu datasetu')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR_STATS, '08_content_pie_chart.png'), dpi=150)
            plt.close()

        print(f"\nJednotlivé grafy boli uložené do priečinka: {OUTPUT_DIR_STATS}")
        print("Analýza úspešne dokončená.")


if __name__ == "__main__":
    analyze_dataset()