import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib
import os
from matplotlib.colors import ListedColormap

matplotlib.use('Agg')


def plot_comprehensive_comparison(v8_path, v11_path, unet_path, save_dir='test_results'):
    os.makedirs(save_dir, exist_ok=True)

    def load_json(path):
        if not os.path.exists(path):
            print(f"Súbor {path} nebol nájdený!")
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    v8 = load_json(v8_path)
    v11 = load_json(v11_path)
    unet = load_json(unet_path)

    models_data = [d for d in [v8, v11, unet] if d is not None]
    model_names = [n for d, n in zip([v8, v11, unet], ['YOLOv8s', 'YOLO11s', 'U-Net']) if d is not None]

    if not models_data:
        print("Žiadne dáta na porovnanie!")
        return

    # --- 1. KONFÚZNE MATICE ---
    fig, axes = plt.subplots(
        1,
        len(models_data),
        figsize=(6 * len(models_data), 6),
        constrained_layout=True
    )

    if len(models_data) == 1:
        axes = [axes]

    # Farby: TP, FP, FN, TN
    custom_colors = np.array([
        [0.7, 0.9, 0.7, 1],  # TP - zelená
        [1.0, 0.8, 0.8, 1],  # FP - červená
        [1.0, 0.9, 0.7, 1],  # FN - oranžová
        [0.9, 0.9, 0.9, 1]  # TN - sivá
    ])

    for ax, data, title in zip(axes, models_data, model_names):
        tp = data['total_matched']
        fp = data['total_pred_photos'] - tp
        fn = data['total_gt_photos'] - tp

        # Farebná matica
        ax.imshow(
            np.array([[0, 1], [2, 3]]),
            cmap=ListedColormap(custom_colors)
        )

        # Titulok
        ax.set_title(
            f'Konfúzna matica: {title}',
            fontsize=15,
            fontweight='bold',
            pad=20
        )

        # Text v bunkách
        cell_data = [
            [(f"TP\n{tp}", 0, 0), (f"FP\n{fp}", 0, 1)],
            [(f"FN\n{fn}", 1, 0), (f"TN\nN/A", 1, 1)]
        ]

        for row in cell_data:
            for text, i, j in row:
                ax.text(
                    j, i, text,
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight='bold'
                )

        # Osy
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(
            ['Pred: FOTO', 'Pred: NIČ'],
            fontsize=12,
            fontweight='bold'
        )

        ax.set_yticklabels(
            ['Real: FOTO', 'Real: NIČ'],
            fontsize=12,
            fontweight='bold'
        )

        ax.grid(False)

    # Uloženie
    plt.savefig(
        os.path.join(save_dir, 'confusion_matrixes.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # --- 2. STĹPCOVÝ GRAF METRÍK ---
    metrics = ['average_iou', 'average_dice', 'overall_precision', 'overall_recall', 'overall_f1']
    labels = ['IoU', 'Dice', 'Precision', 'Recall', 'F1-Score']

    x = np.arange(len(labels))
    width = 0.25  # Šírka stĺpca

    fig, ax = plt.subplots(figsize=(12, 7))

    # Farby pre jednotlivé modely
    colors = ['#A3FFFF', '#FFB3FF', '#B4FFB4']

    for i, (data, name) in enumerate(zip(models_data, model_names)):
        values = [data.get(m, 0) for m in metrics]
        offset = (i - (len(models_data) - 1) / 2) * width
        rects = ax.bar(x + offset, values, width, label=name, color=colors[i], edgecolor='black', alpha=0.85)

        # Pridanie hodnôt nad stĺpce
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Skóre', fontsize=12, fontweight='bold')
    ax.set_title('Porovnanie výkonnosti modelov', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.15)  # Trochu miesta pre text nad stĺpcami
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    metrics_path = os.path.join(save_dir, 'models_comparison_bars.png')
    plt.savefig(metrics_path, dpi=200)
    plt.close()

    print(f"Grafy uložené v '{save_dir}':\n   - confusion_matrixes.png\n   - models_comparison_bars.png")


# Spustenie
plot_comprehensive_comparison(
    '../test_results/overall_metrics_yolo8s.json',
    '../test_results/overall_metrics_yolo11s.json',
    '../test_results/overall_metrics_unet.json'
)
