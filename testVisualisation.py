import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib
import os
from matplotlib.colors import ListedColormap

matplotlib.use('Agg')


def plot_comprehensive_comparison(v8_path, v11_path, save_dir='test_results'):
    os.makedirs(save_dir, exist_ok=True)

    with open(v8_path, 'r') as f:
        v8 = json.load(f)
    with open(v11_path, 'r') as f:
        v11 = json.load(f)

    # --- 1. DETEKČNÉ MATICE (Vylepšené farby) ---
    def get_stats(data):
        tp = data['total_matched']
        fp = data['total_pred_photos'] - tp
        fn = data['total_gt_photos'] - tp
        return tp, fp, fn

    stats_v8 = get_stats(v8)
    stats_v11 = get_stats(v11)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Definujeme vlastnú farebnú mapu pre kvadranty:
    # TP (Zelená), FP (Červená), FN (Orandžová), TN (Sivá)
    custom_colors = np.array([
        [0.7, 0.9, 0.7, 1],  # Svetlo zelená (TP)
        [1.0, 0.8, 0.8, 1],  # Svetlo červená (FP)
        [1.0, 0.9, 0.7, 1],  # Oranžová (FN)
        [0.9, 0.9, 0.9, 1]  # Sivá (TN)
    ])

    for ax, stats, title in zip(axes, [stats_v8, stats_v11], ['YOLOv8s', 'YOLO11s']):
        tp, fp, fn = stats

        # Vizualizujeme maticu farieb (každý kvadrant má index 0-3)
        color_indices = np.array([[0, 1], [2, 3]])
        ax.imshow(color_indices, cmap=ListedColormap(custom_colors), interpolation='nearest')

        ax.set_title(f'Konfúzna matica: {title}', fontsize=16, fontweight='bold', pad=20)

        # Texty a hodnoty
        cell_data = [
            [(f"TRUE POSITIVE\n(Správne)", tp), (f"FALSE POSITIVE\n(Navyše)", fp)],
            [(f"FALSE NEGATIVE\n(Chýba)", fn), ("TRUE NEGATIVE\n(Pozadie)", "N/A")]
        ]

        for i in range(2):
            for j in range(2):
                label, val = cell_data[i][j]
                # Hlavný text
                ax.text(j, i, f"{label}\n\n{val}", ha="center", va="center",
                        color="#2c3e50", fontsize=13, fontweight='bold')

        # Estetika osí
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predikcia: FOTO', 'Predikcia: NIČ'], fontsize=12)
        ax.set_yticklabels(['Realita: FOTO', 'Realita: NIČ'], fontsize=12, rotation=90, va='center')

        # Odstránenie mriežky, ktorá by rušila farby
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'konfuzne_matice.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Graf matíc uložený do: {save_dir}/konfuzne_matice.png")


# Spustenie
plot_comprehensive_comparison(
    'test_results/overall_metrics_yolo8s.json',
    'test_results/overall_metrics_yolo11s.json'
)