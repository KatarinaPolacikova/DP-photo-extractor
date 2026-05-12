import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib
import os
from matplotlib.colors import ListedColormap

matplotlib.use('Agg')


# =============================================================================
# FUNKCIA PRE KOMPLEXNÉ POROVNANIE VÝKONNOSTI MODELOV
# =============================================================================
def plot_comprehensive_comparison(v8_path, v11_path, unet_path, save_dir='test_results'):
    """
        Funkcia slúži na porovnanie výsledkov z rôznych architektúr (YOLOv8, YOLO11, U-Net)
        a ich následnú vizualizáciu vo forme konfúznych matíc a porovnávacieho stĺpcového grafu.

        Parametre:
        v8_path, v11_path, unet_path (str): Cesty k JSON súborom s vypočítanými metrikami.
        save_dir (str): Adresár pre uloženie vygenerovaných grafických výstupov.
    """

    os.makedirs(save_dir, exist_ok=True)

    def load_json(path):
        """Pomocná funkcia pre bezpečné načítanie údajov z JSON súborov."""
        if not os.path.exists(path):
            print(f"Súbor {path} nebol nájdený!")
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Načítanie dát pre jednotlivé modely
    v8 = load_json(v8_path)
    v11 = load_json(v11_path)
    unet = load_json(unet_path)

    # Filtrácia modelov, ktorých dáta sa podarilo úspešne načítať
    models_data = [d for d in [v8, v11, unet] if d is not None]
    model_names = [n for d, n in zip([v8, v11, unet], ['YOLOv8s', 'YOLO11s', 'U-Net']) if d is not None]

    if not models_data:
        print("Chyba: Neboli nájdené žiadne relevantné dáta na porovnanie!")
        return

    # KONFÚZNE MATICE (CONFUSION MATRICES)
    # Konfúzna matica vizualizuje presnosť klasifikácie a chyby typu I (FP) a typu II (FN).
    fig, axes = plt.subplots(
        1,
        len(models_data),
        figsize=(5 * len(models_data), 5),
        constrained_layout=True
    )

    if len(models_data) == 1:
        axes = [axes]

    # Definícia palety farieb
    # TP (zelená), FP (červená), FN (oranžová), TN (svetlosivá)
    custom_colors = np.array([
        [0.8, 0.95, 0.8, 1],  # TP
        [1.0, 0.85, 0.85, 1],  # FP
        [1.0, 0.9, 0.75, 1],  # FN
        [0.95, 0.95, 0.95, 1]  # TN
    ])

    for ax, data, title in zip(axes, models_data, model_names):
        # Extrakcia základných prvkov matice z údajov
        tp = data['total_matched']
        fp = data['total_pred_photos'] - tp
        fn = data['total_gt_photos'] - tp

        # Vykreslenie matice
        ax.imshow(
            np.array([[0, 1], [2, 3]]),
            cmap=ListedColormap(custom_colors)
        )

        ax.set_title(f'Konfúzna matica: {title}', fontsize=14, fontweight='bold', pad=15)

        # Mapovanie textových informácií priamo do buniek matice
        # TP, FP, FN sú kľúčové metriky pre pochopenie správania sa segmentačného modelu.
        cell_info = [
            ("TP", tp, 0, 0, "Správna detekcia"),
            ("FP", fp, 0, 1, "Falošná detekcia"),
            ("FN", fn, 1, 0, "Nezachytené"),
            ("TN", "-", 1, 1, "Pozadie")
        ]

        for label, val, i, j, full_name in cell_info:
            color = 'black'
            ax.text(j, i, f"{label}\n{val}\n({full_name})",
                    ha="center", va="center", fontsize=11, fontweight='bold', color=color)

        # Nastavenie popisov osí v slovenčine
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Fotografia', 'Nič'], fontsize=11)
        ax.set_yticklabels(['Fotografia', 'Nič'], fontsize=11, rotation=90, va='center')

        ax.set_xlabel('Predikcia modelu', fontsize=12, labelpad=10)
        ax.set_ylabel('Skutočnosť (Ground Truth)', fontsize=12, labelpad=10)

        # Odstránenie mriežky a zbytočných čiar
        ax.tick_params(which="both", bottom=False, left=False)

    plt.savefig(os.path.join(save_dir, 'confussion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # STĹPCOVÝ GRAF METRÍK
    # Tento graf poskytuje priame porovnanie presnosti (Precision), úplnosti (Recall), F1-skóre a priestorového prekryvu (IoU, Dice).
    metrics = ['average_iou', 'average_dice', 'overall_precision', 'overall_recall', 'overall_f1']
    labels = ['IoU', 'Dice', 'Precision', 'Recall', 'F1-Score']

    x = np.arange(len(labels))
    width = 0.25  # Šírka jednotlivých stĺpcov pre skupiny

    fig, ax = plt.subplots(figsize=(12, 7))

    # Farebné kódovanie modelov pre ľahkú orientáciu v grafoch
    colors = ['#A3FFFF', '#FFB3FF', '#B4FFB4']

    for i, (data, name) in enumerate(zip(models_data, model_names)):
        values = [data.get(m, 0) for m in metrics]
        offset = (i - (len(models_data) - 1) / 2) * width
        rects = ax.bar(x + offset, values, width, label=name, color=colors[i], edgecolor='black', alpha=0.85)

        # Numerické označenie výšky stĺpca priamo v grafe (kvôli prehľadnosti)
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

    print(f"Vizualizácia úspešne dokončená. Výstupy uložené v '{save_dir}':")
    print(f"   - confussion_matrices.png")
    print(f"   - models_comparison_bars.png")


# =============================================================================
# SPUSTENIE ANALÝZY
# =============================================================================
if __name__ == "__main__":
    plot_comprehensive_comparison(
        '../test_results/overall_metrics_yolo8s.json',
        '../test_results/overall_metrics_yolo11s.json',
        '../test_results/overall_metrics_unet.json'
    )
