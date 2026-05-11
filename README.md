# DP-photo-extractor: Automatizované predspracovanie digitalizovaných fotografií

Tento softvérový systém slúži na plne automatizované predspracovanie digitalizovaných archívnych fotografií. Navrhnuté riešenie integruje pokročilé metódy počítačového videnia a hlbokého učenia do všetkých fáz spracovania — od detekcie a extrakcie snímok zo skenu, cez geometrickú korekciu (narovnanie), až po určenie správnej rotácie. 

Vďaka plnej automatizácii systém minimalizuje potrebu manuálneho zásahu a je schopný efektívne spracovávať aj rozsiahle archívne súbory.

---
## Technické požiadavky
- **Operačný systém:** Windows 10/11, Linux alebo macOS.
- **Prostredie:** Python 3.9 až 3.11.
- **Hardvér:** Odporúčaná NVIDIA GPU (min. 4GB VRAM) pre akceleráciu CUDA, alebo CPU.
---
## Inštalácia a príprava

1. **Klonovanie repozitára:**
   ```bash
   git clone https://github.com/KatarinaPolacikova/DP-photo-extractor.git
   cd DP-photo-extractor
   
2. **Inštalácia závislostí:**

   ```bash
   pip install -r requirements.txt
---   
## Správa modelov (Weights)
Pre správnu funkciu systému sú nevyhnutné súbory s váhami modelov vo formáte .pt.

- **Segmentačné modely:** Súbory pre natrénované modely YOLO11s, YOLOv8s a U-Net sa nachádzajú v priečinku trained_models/.

- **Detekčný model (YOLOv8n):** Nachádza sa v priečinku yolo_models/. Slúži na sémantickú analýzu obrazu pre potreby rotácie. Je možné ho stiahnuť aj príkazom:

    ```bash
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
---
## Opis a spustenie hlavných skriptov

Systém po spustení automaticky spracuje vstupný obrázok a výsledné jednotlivé snímky uloží do priečinka `extracted_photos/`. Skripty sa nachádzajú v príslušných podadresároch podľa zvolenej architektúry.

### 1. Implementácia YOLO (`scripts/YOLO/`)
Tento prístup je vhodný pre inštančnú segmentáciu a separáciu viacerých (aj čiastočne prekrývajúcich sa) fotografií na jednom skene.

```python
from scripts.YOLO.extractAndRotatePhotosYOLO import crop_photos
    
# Spustenie automatického spracovania
crop_photos("cesta/k/vasmu/obrazku.jpg")
```

### 2. Implementácia U-Net (scripts/UNET/)
Tento prístup je vhodný pre sémantickú segmentáciu s vysokým dôrazom na presnosť orezu hrán a detailné spracovanie kontúr.

```python
from scripts.UNET.extractAndRotatePhotosUNET import crop_photos
    
# Spustenie automatického spracovania
crop_photos("cesta/k/vasmu/obrazku.jpg")
```
---
## Algoritmus rotácie a extrakcie fotografií

Systém vykonáva proces v troch nadväzujúcich krokoch:

1. **Segmentácia:** Vygenerovanie masky (binary mask) pre každú identifikovanú fotografiu na skene.
2. **Geometrická korekcia:** Výpočet rohových bodov z nájdenej masky a následná aplikácia perspektívnej transformácie na narovnanie snímky do pravouhlého formátu.
3. **Rotácia:** Kaskádový rozhodovací proces, ktorý analyzuje:
    - Prítomnosť a orientáciu osôb/tvárí.
    - Polohu sémantických objektov.
    - Distribúciu jasu (rozlíšenie oblohy a zeme) pre určenie finálneho natočenia.

---

## Doplnkové moduly a skripty

Repozitár obsahuje aj pomocné moduly využité vo výskumnej časti práce pre potreby trénovania modelov a objektívneho hodnotenia úspešnosti systému.

### Príprava dát a analýza
- `scripts/prepareDataset.py`: Automatizovaná príprava a čistenie dát pred procesom trénovania.
- `scripts/generateMasks.py`: Skript na generovanie binárnych masiek potrebných pre trénovanie architektúry U-Net.
- `scripts/analyzeDataset.py`: Nástroj na štatistickú analýzu vlastností a distribúcie dát v datasete.

### Trénovanie modelov
- `scripts/YOLO/trainYOLO.py`: Skript určený na trénovanie a ladenie modelov architektúry YOLO.
- `scripts/UNET/trainUNET.py`: Skript určený na trénovanie architektúry U-Net.

### Testovanie a evaluácia
- `scripts/YOLO/testYOLO.py` & `scripts/UNET/testUNET.py`: Nástroje na komplexnú evaluáciu úspešnosti modelov pomocou metrík (IoU, F1-score, Dice).
- `scripts/compareModels.py`: Skript slúžiaci na vzájomné porovnanie kvalitatívnych výsledkov jednotlivých implementovaných modelov.

---
## Ukážky výsledkov a záznamy experimentov

Repozitár obsahuje predgenerované ukážky a výsledky experimentov popísaných v práci, ktoré demonštrujú úspešnosť algoritmov:

- **`extracted_photos/` & `extracted_photos_UNET/`**: Obsahujú príklady finálne spracovaných fotografií (výstupy programu) vo forme separovaných, narovnaných a správne zorientovaných snímok.
- **`graphs_and_statistics/`**: Priečinok s výstupmi analýzy datasetu. Obsahuje vizualizácie distribúcie dát a grafické porovnanie úspešnosti jednotlivých modelov.
- **`runs/`**: Kompletné záznamy z procesov trénovania modelov YOLO a U-Net, vrátane priebežných metrík, strát (loss functions) a vygenerovaných grafov učenia.
- **`test_results/`**: Detailné výsledky testovania úspešnosti. Nachádza sa tu vyhodnotenie pre jednotlivé testovacie snímky (podpriečinok **`single_image/`**), ako aj súhrnné štatistiky pre celú testovanú množinu dát.
---

## Poznámka

Tento softvér bol vyvinutý ako praktický výstup diplomovej práce. Podrobný teoretický opis algoritmov, priebeh experimentov a hĺbková analýza výsledkov sú uvedené v hlavnom texte práce. Podrobné technické inštrukcie a špecifikácie prostredia sú spracované v **Prílohe A – Používateľská príručka**.

**Autor:** Katarína Poláčiková  
**Rok:** 2026