# Experimento: CLAHE

Ecualización adaptativa de histograma con contraste limitado sobre imágenes de hojas de maíz.
**Aislado del pipeline principal**: no toca `raw/`, `clean/` ni `outputs/`, no lee `DATASET_ROOT`
y no depende de `src.*`. Es un banco de pruebas visual, no un paso de preprocesamiento.

## Uso

```bash
# 1. Copia las imágenes a evaluar
cp mi_imagen.jpg experiments/clahe/input/

# 2. Ejecuta
python experiments/clahe/apply_clahe.py

# Variantes
python experiments/clahe/apply_clahe.py --limit 10          # solo las primeras 10
python experiments/clahe/apply_clahe.py --clip-limit 3.0    # más contraste (más ruido)
python experiments/clahe/apply_clahe.py --tile-grid 16      # tiles más finos
python experiments/clahe/apply_clahe.py --no-comparisons    # omite los paneles
python experiments/clahe/apply_clahe.py --input-dir <otro>  # otra carpeta de entrada
```

## Estructura

```
experiments/clahe/
├── apply_clahe.py
├── input/          # imágenes de entrada (git-ignoradas)
├── output/         # <nombre>_clahe.png       (git-ignorado)
└── comparisons/    # <nombre>_compare.png     (git-ignorado)
```

El panel comparativo muestra original vs. CLAHE con sus histogramas de luminancia.

## Nota de implementación

CLAHE se aplica **solo al canal L de LAB**, no a los tres canales RGB por separado: ecualizar
RGB desplaza el tono, y el color es señal diagnóstica en las clases de deficiencia nutricional
(`nitrogen_deficiency`, `phosphorus_deficiency`, `potassium_deficiency`), donde la clorosis
amarillenta es justamente lo que distingue la clase.

## Parámetros

| Flag | Default | Efecto |
|---|---|---|
| `--clip-limit` | `2.0` | Umbral de recorte. Más alto = más contraste local y más amplificación de ruido. |
| `--tile-grid` | `8` | Grilla NxN de tiles. Más alto = adaptación más local. |

## Dependencias

Requiere `opencv-python-headless` (incluida en la instalación base) y `matplotlib`
(`pip install -e ".[experiments]"`).
