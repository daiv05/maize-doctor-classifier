---
title: Auditoría Forense de Píxeles y Duplicados Exactos
description: Verificación determinista de 33 437 imágenes del corpus canónico, exclusiones de conflictos y refutación de fuga en test.
---

# Auditoría Forense de Píxeles y Duplicados Exactos

> **Preguntas experimentales:** ¿Existen imágenes idénticas etiquetadas con enfermedades contradictorias que penalicen al modelo? ¿Hay imágenes duplicadas compartidas entre el conjunto de entrenamiento y el conjunto de prueba que inflen artificialmente la exactitud (*data leakage*)?

---

## 1. Metodología de la Representación Canónica de Píxeles

Dos archivos pueden tener bytes diferentes (`file_sha256` distinto) debido a metadatos EXIF, compresión JPEG o encabezados, pero decodificar exactamente la misma imagen visual. Para detectar duplicados y contradicciones reales:

1. Cada archivo se decodifica con Pillow y se aplica la orientación EXIF (`exif_transpose`).
2. Se convierte al modo canónico RGB de tres canales descartando perfiles ICC o canales alfa.
3. Se extrae la secuencia cruda de bytes RGB en orden de filas (*row-major*).
4. Se calcula un hash criptográfico determinista `pixel_sha256`:

   ```text
   pixel_sha256 = SHA256(
       UTF8("doctormaiz-pixel-sha256-v1\0RGB\0")
       || uint32be(width)
       || uint32be(height)
       || RGB_row_major_bytes
   )
   ```

---

## 2. Cobertura del Censo Canónico

La auditoría analizó el **100 % del volumen canónico** (33 437 archivos descubiertos):

| Población de Archivos | Cantidad | Duplicados Exactos Internos | Duplicados Entre Splits (*Cross-Split*) | Conflictos de Etiqueta (*Cross-Label*) |
|---|---:|:---:|:---:|:---:|
| **Muestras Elegibles (`seed_42`)** | **33 429** | **0** | **0** | **0** |
| **Archivos Excluidos Contractualmente** | **8** | 4 pares | *No aplica (excluidos)* | **4 pares (8 imágenes)** |
| **Total de Archivos Auditados** | **33 437** | — | — | — |

---

## 3. Resolución de los Dos Puntos Críticos

### A. Los Conflictos de Etiquetas Contradictorias (Punto 3)
* **Sospecha histórica:** Se temía la existencia de unos 15 grupos de imágenes idénticas con etiquetas contradictorias.
* **Hallazgo real y exhaustivo:** El número de 15 pertenecía a versiones preliminares no congeladas y **no se reprodujo** en el corpus actual. La auditoría localizó de forma unívoca exactamente **4 grupos (8 archivos)** con píxeles idénticos pero clases opuestas (pertenecientes a `maize-beans-tomatoes-africa`).
* **Blindaje aplicado:** Los 8 archivos fueron catalogados y excluidos de forma permanente en `config/dataset_exclusions.csv` vinculados a sus hashes SHA-256. El generador de splits (`scripts/pipeline/create_splits.py`) aborta automáticamente si detecta cualquier conflicto *cross-label* no declarado.

### B. El Mito del 0.68 % de Gemelos en Test (Punto 4)
* **Sospecha histórica:** Se afirmaba que el 0.68 % del conjunto de prueba (~34 imágenes) tenía un duplicado exacto en el conjunto de entrenamiento.
* **Hallazgo en el split vigente (`seed_42`):**
  * Duplicados exactos Train / Test: **0 muestras (0.0000 %)**.
  * Duplicados exactos Val / Test: **0 muestras (0.0000 %)**.
  * Duplicados exactos Train / Val: **0 muestras (0.0000 %)**.
  * Imágenes de prueba afectadas: **0 de 5 015 imágenes (0.00 %)**.
* **Conclusión:** En la materialización canónica vigente, el conjunto de prueba es **estrictamente independiente** a nivel de bytes y de píxeles decodificados.

---

## 4. Límite Metodológico Declarado con Transparencia

> [!WARNING] Alcance: Duplicados Exactos vs. Similitud Perceptual
> La auditoría garantiza matemáticamente la **ausencia total de duplicados exactos** (por bytes y por raster RGB). Sin embargo, la materialización actual no ejecutó deduplicación perceptual (`deduplicate_perceptual=false`), por lo que no prueba la ausencia de imágenes casi duplicadas por rotación, recorte menor o encuadres consecutivos de ráfaga (PHash). Este límite se encuentra debidamente documentado en la metodología del proyecto.

---

## 5. Artefactos y Evidencia Versionada

Los datos crudos de la auditoría completa están disponibles para inspección directa:
* Informe formal de la auditoría: [`PIXEL_DUPLICATE_AUDIT.md`](../reproducibilidad/evidencia/pixel_duplicate_audit/PIXEL_DUPLICATE_AUDIT.md)
* Inventario de hashes de las 33 437 imágenes: [`corpus_preflight.csv`](../reproducibilidad/evidencia/pixel_duplicate_audit/corpus_preflight.csv) (**11.6 MB**)
* Inventario canónico RGB post-EXIF: [`pixel_hash_inventory.csv`](../reproducibilidad/evidencia/pixel_duplicate_audit/pixel_hash_inventory.csv) (**11.3 MB**)
* Registro contractual de los 4 conflictos excluidos: `config/dataset_exclusions.csv` y [`pixel_label_conflicts.csv`](../reproducibilidad/evidencia/pixel_duplicate_audit/pixel_label_conflicts.csv)
