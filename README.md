# Ciro v3 · Sistema de Clasificación y Detección de Residuos Sólidos (NTP 900.058:2019)

**Universidad Peruana Unión (UPeU) · Facultad de Ingeniería y Arquitectura · EP Ingeniería de Sistemas**

Proyecto integral de visión artificial e inteligencia artificial para la clasificación y localización multiobjeto en tiempo real de residuos sólidos conforme al **código de 7 colores de la Norma Técnica Peruana NTP 900.058:2019**, desplegado de forma nativa en navegador mediante **ONNX Runtime Web (WebAssembly)**.

---

## 1. Descripción del Proyecto y Arquitectura Ciro v3

El sistema **Ciro v3** resuelve el problema de segregación automatizada de residuos municipales, institucionales y hospitalarios operando bajo tres modos complementarios en tiempo real:

1. **Clasificador Calibrado (`models/ciro_cls_ntp_v3.onnx` · `YOLO11s-cls` · `320×320`):**
   - Red neuronal de clasificación holística de escena con cabezal lineal optimizado mediante regularización proximal $L_2$ ($\lambda=0.20$).
   - Alcanza **67.8% de exactitud balanceada** en los 9 grupos de evaluación externa, **81.5% en residuos orgánicos**, **33.3% de recall en residuos biocontaminados (Rojo)** y **0.0% de fallos por falta de detección**.
2. **Detector Multiobjeto Calibrado (`models/yolo26s_ntp_7c_v3.onnx` · `YOLO26s` · `640×640`):**
   - Detector de cajas delimitadoras (*bounding boxes*) con calibración multiescala (`P3, P4, P5`) en los cabezales `cv3` y `one2one_cv3`.
   - En el conjunto de prueba independiente (**2,003 imágenes, 3,065 instancias**) alcanza **mAP@0.5 = 0.6664** global y **mAP@0.5 = 0.9323 (F1 = 0.9014)** en la clase crítica **`peligroso` (Rojo)**.
3. **Modo Híbrido Ciro v3 (Fusión Ponderada + Regla de Prioridad Sanitaria):**
   - Combina en paralelo ambos modelos ($S = 0.74 \cdot P_{\text{cls}} + 0.26 \cdot P_{\text{det}}$) y activa una regla de prioridad sanitaria para residuos peligrosos/biocontaminados, alcanzando **85.2% en orgánicos** e **invariancia positiva a escala de grises ($\Delta +4.7\text{ pp}$)**.

---

## 2. Taxonomía Oficial de 7 Colores (NTP 900.058:2019)

| ID | Clase Canónica | Color de Contenedor NTP | Tipos de Residuos Sólidos Incluidos |
| :-: | :--- | :--- | :--- |
| `0` | `plastico` | **BLANCO** | Botellas PET transparentes y de color, envases rígidos HDPE/PP, bolsas y tapas plásticas limpias. |
| `1` | `papel_carton` | **AZUL** | Cajas de cartón corrugado, papel de oficina, periódicos, revistas y empaques de cartulina secos. |
| `2` | `metal` | **AMARILLO** | Latas de bebidas de aluminio, conservas de hojalata, tapas y envases metálicos. |
| `3` | `vidrio` | **GRIS / PLOMO** | Botellas, frascos y envases de vidrio transparente, verde o ámbar. |
| `4` | `organico` | **MARRÓN** | Restos de alimentos, cáscaras de frutas y verduras, residuos vegetales y compostables. |
| `5` | `otros_residuos` | **NEGRO** | Envases de tecnopor (poliestireno expandido), envolturas multicapa, servilletas sucias y residuos sanitarios. |
| `6` | `peligroso` | **ROJO** | EPP clínico (mascarillas quirúrgicas/N95, guantes de nitrilo/látex, gasas), jeringas, pilas/baterías, aerosoles y fármacos. |

---

## 3. Dataset Completo Entrenado (`dataset_ntp900058_splits.zip`)

El repositorio incluye el archivo **[`dataset_ntp900058_splits.zip`](dataset_ntp900058_splits.zip)** (`46.1 MB`), el cual contiene el corpus completo de **14,189 imágenes y 14,189 etiquetas en formato YOLO** dividido exactamente en las particiones utilizadas durante el entrenamiento (**Partición Estratificada Iterativa Consciente de Grupos 70% / 15% / 15%** con **0.0% de fuga de datos** verificada mediante `SHA-256` y `dHash` de 64 bits):

```text
dataset_ntp900058_splits.zip
└── dataset_ntp900058/
    ├── dataset.yaml                 # Configuración oficial de splits y 7 clases para Ultralytics YOLO
    ├── taxonomy_ntp900058.yaml      # Mapeo taxonómico canónico NTP 900.058:2019
    ├── dataset_manifest.json        # Manifiesto de auditoría criptográfica (SHA-256: ed951fc07f63519c)
    ├── class_distribution.csv       # Conteo exacto de cajas por clase en train, val y test
    ├── images/
    │   ├── train/                   # 9,914 imágenes de entrenamiento (28% en escala de grises 3-ch)
    │   ├── val/                     # 2,102 imágenes de validación
    │   └── test/                    # 2,173 imágenes de prueba independiente
    └── labels/
        ├── train/                   # 9,914 archivos .txt con coordenadas YOLO normalizadas [cls xc yc w h]
        ├── val/                     # 2,102 archivos .txt de validación
        └── test/                    # 2,173 archivos .txt de prueba independiente
```

---

## 4. Estructura del Repositorio

```text
.
├── index.html                       # Estación interactiva de demostración en vivo (ONNX Runtime Web)
├── app.js                           # Motor client-side WebAssembly (Clasificador, Detector NMS e Híbrido)
├── styles.css                       # Interfaz dark-mode con simulador visual de los 7 contenedores NTP
├── dataset_ntp900058_splits.zip     # Dataset completo (14,189 imágenes + etiquetas en splits train/val/test)
├── models/
│   ├── ciro_cls_ntp_v3.onnx         # Clasificador calibrado ONNX (21.3 MB · metadatos NTP embebidos)
│   └── yolo26s_ntp_7c_v3.onnx       # Detector multiobjeto ONNX (38.1 MB · metadatos NTP embebidos)
├── samples/                         # 14 imágenes reales de prueba rápida (2 por contenedor NTP)
└── assets/
    └── logo_upeu.png                # Emblema institucional Universidad Peruana Unión
```