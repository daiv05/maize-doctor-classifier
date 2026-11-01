"""
Herramienta interactiva para encontrar y eliminar imágenes duplicadas
usando imagededup (PHash - rápido, escala a 20k+ imágenes).

Analiza cada enfermedad completa (lab/ + real/ juntos).

Opciones:
  1. Encontrar duplicados  - guarda CSV en <OUTPUT_ROOT>/cleanup/duplicates/
  2. Encontrar y eliminar  - guarda CSV y elimina duplicados
  3. Eliminar duplicados   - elimina basándose en CSV ya existente
"""

import csv
import json
from datetime import datetime
from pathlib import Path

from src.config import get_dataset_root, get_output_root

ANALYSIS_DIR = get_output_root() / "cleanup" / "duplicates"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Umbral de similitud para PHash (0 = idénticas, 10 = muy similares).
# 0 solo detecta copias exactas; sube a 5-8 para near-duplicates.
PHASH_THRESHOLD = 0


#  helpers


def get_disease_dirs() -> list[Path]:
    clean_dir = get_dataset_root() / "clean"
    if not clean_dir.exists():
        raise SystemExit(
            f"No se encontró la carpeta clean en {clean_dir.parent}. Verifica DATASET_ROOT en .env"
        )
    return sorted([d for d in clean_dir.iterdir() if d.is_dir()])


def count_images(disease_dir: Path) -> int:
    return sum(1 for f in disease_dir.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


def choose_disease() -> Path:
    dirs = get_disease_dirs()
    if not dirs:
        raise SystemExit(f"No se encontraron carpetas en {get_dataset_root() / 'clean'}.")
    print("\nEnfermedades disponibles:\n")
    for i, d in enumerate(dirs, 1):
        total = count_images(d)
        subdirs = [s.name for s in sorted(d.iterdir()) if s.is_dir()]
        sub_info = f"  [{', '.join(subdirs)}]" if subdirs else ""
        print(f"  [{i}] {d.name}{sub_info}  ({total} imágenes en total)")
    print()

    while True:
        raw = input("Selecciona el número de enfermedad: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(dirs):
            selected = dirs[int(raw) - 1]
            print(f"\nSeleccionada: {selected.name}\n")
            return selected
        print("  Opción inválida, intenta de nuevo.")


def choose_action() -> int:
    print("¿Qué deseas hacer?\n")
    print("  [1] Encontrar duplicados (guarda CSV)")
    print("  [2] Encontrar y eliminar duplicados")
    print("  [3] Eliminar duplicados desde CSV existente\n")
    while True:
        raw = input("Selecciona una opción (1/2/3): ").strip()
        if raw in ("1", "2", "3"):
            return int(raw)
        print("  Opción inválida, intenta de nuevo.")


def csv_path_for(disease_dir: Path) -> Path:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ANALYSIS_DIR / f"duplicates_{disease_dir.name}_{ts}.csv"


def latest_csv_for(disease_dir: Path) -> Path | None:
    matches = sorted(ANALYSIS_DIR.glob(f"duplicates_{disease_dir.name}_*.csv"), reverse=True)
    return matches[0] if matches else None


#  core


def collect_images(disease_dir: Path) -> dict[str, Path]:
    """
    Retorna {ruta_absoluta_str: Path} para todas las imágenes bajo disease_dir.
    imagededup acepta encoding_map con claves = rutas absolutas.
    """
    return {
        str(f.resolve()): f.resolve()
        for f in disease_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    }


def deduplicate_groups(raw: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    imagededup reporta cada par de forma simétrica: si A duplica B,
    también reporta B duplica A.  Esto colapsa esos grupos en uno solo
    eligiendo como 'original' el representante con menor ruta (reproducible)
    y listando el resto como duplicados a eliminar.

    Garantía: cada imagen aparece en exactamente un grupo - como original
    o como duplicado - nunca en ambos.
    """
    visited: set[str] = set()
    result: dict[str, list[str]] = {}

    for node, neighbors in raw.items():
        if node in visited or not neighbors:
            continue
        # BFS para reunir todos los miembros del componente conectado
        component: set[str] = set()
        queue = [node]
        while queue:
            current = queue.pop()
            if current in component:
                continue
            component.add(current)
            for nb in raw.get(current, []):
                if nb not in component:
                    queue.append(nb)

        visited |= component
        members = sorted(component)  # orden determinista
        result[members[0]] = members[1:]

    return result


def find_duplicates(disease_dir: Path) -> tuple[dict[str, list[str]], dict[str, Path]]:
    """
    Usa PHash con rutas absolutas directamente - sin directorio temporal.
    Retorna (duplicados, mapa_clave-Path_real).
    """
    from imagededup.methods import PHash

    image_map = collect_images(disease_dir)
    total = len(image_map)
    print(f"  {total} imágenes encontradas en '{disease_dir.name}' (lab + real)")

    if total == 0:
        return {}, {}

    print("Calculando PHash (rápido, escala a grandes volúmenes)...")
    hasher = PHash()
    encodings = _encode_all(hasher, image_map)

    raw: dict[str, list[str]] = hasher.find_duplicates(
        encoding_map=encodings,
        max_distance_threshold=PHASH_THRESHOLD,
        scores=False,
    )

    duplicates = deduplicate_groups({k: v for k, v in raw.items() if v})
    return duplicates, image_map


def _encode_all(hasher, image_map: dict[str, Path]) -> dict[str, str]:
    """
    Encoda todas las imágenes de image_map una a una usando PHash.
    Evita crear directorios temporales y soporta archivos en múltiples carpetas.
    """
    import numpy as np
    from PIL import Image

    encodings: dict[str, str] = {}
    total = len(image_map)

    for i, (key, path) in enumerate(image_map.items(), 1):
        if i % 500 == 0 or i == total:
            print(f"  Procesadas {i}/{total} imágenes...", end="\r")
        try:
            img = Image.open(path).convert("L").resize((32, 32), Image.LANCZOS)
            encodings[key] = hasher.encode_image(image_file=None, image_array=np.array(img))
        except Exception as e:
            print(f"\n  Omitida {path.name}: {e}")

    print()  # nueva línea tras el progreso
    return encodings


def report(duplicates: dict[str, list[str]]) -> None:
    total_dupes = sum(len(v) for v in duplicates.values())
    print(f"\nGrupos con duplicados: {len(duplicates)}")
    print(f"Imágenes que se conservan (una por grupo): {len(duplicates)}")
    print(f"Imágenes que se eliminarán: {total_dupes}\n")
    for original, dupes in duplicates.items():
        print(f"  conservar: {Path(original).name}  ({Path(original).parent.name})")
        for d in dupes:
            print(f"    └ eliminar: {Path(d).name}  ({Path(d).parent.name})")


def save_csv(
    duplicates: dict[str, list[str]],
    disease_dir: Path,
) -> Path:
    path = csv_path_for(disease_dir)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original_path", "duplicate_paths"])
        for original, dupes in duplicates.items():
            writer.writerow([original, json.dumps(dupes)])
    print(f"CSV guardado en: {path}")
    return path


def delete_paths(paths: list[Path]) -> None:
    unique = list({str(p): p for p in paths}.values())
    print(f"\nSe eliminarán {len(unique)} archivos duplicados.")
    confirm = input("¿Confirmas? (s/N): ").strip().lower()
    if confirm != "s":
        print("Operación cancelada.")
        return

    deleted = 0
    for path in unique:
        if path.exists():
            path.unlink()
            deleted += 1
        else:
            print(f"  No encontrado, omitido: {path.name}")

    print(f"\nEliminados: {deleted} archivos.")


def delete_duplicates_from_csv(disease_dir: Path) -> None:
    csv_file = latest_csv_for(disease_dir)
    if csv_file is None:
        print(f"No se encontró ningún CSV para: {disease_dir.name}")
        print(f"Buscado en: {ANALYSIS_DIR}")
        return

    print(f"CSV encontrado: {csv_file.name}")
    raw: dict[str, list[str]] = {}

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths = json.loads(row["duplicate_paths"])
            if paths:
                raw[row["original_path"]] = paths

    if not raw:
        print("El CSV no contiene duplicados registrados.")
        return

    # Re-colapsa los pares simétricos que imagededup reporta por duplicado
    duplicates = deduplicate_groups(raw)
    dupe_paths = [Path(p) for dupes in duplicates.values() for p in dupes]

    print(f"\nGrupos en CSV: {len(duplicates)}")
    print(f"Imágenes que se conservan: {len(duplicates)}")
    print(f"Imágenes que se eliminarán: {len(dupe_paths)}")
    for original, dupes in duplicates.items():
        print(f"  conservar: {Path(original).name}  ({Path(original).parent.name})")
        for d in dupes:
            print(f"    └ eliminar: {Path(d).name}  ({Path(d).parent.name})")

    delete_paths(dupe_paths)


#  main


def main() -> None:
    print("=" * 60)
    print("  Detector de imágenes duplicadas - Corn Leaf Disease")
    print("=" * 60)

    disease_dir = choose_disease()
    action = choose_action()

    if action == 1:
        duplicates, _ = find_duplicates(disease_dir)
        report(duplicates)
        if duplicates:
            save_csv(duplicates, disease_dir)
        else:
            print("No se encontraron duplicados.")

    elif action == 2:
        duplicates, _ = find_duplicates(disease_dir)
        report(duplicates)
        if not duplicates:
            print("No se encontraron duplicados.")
            return
        save_csv(duplicates, disease_dir)
        dupe_paths = [Path(d) for dupes in duplicates.values() for d in dupes]
        delete_paths(dupe_paths)

    elif action == 3:
        delete_duplicates_from_csv(disease_dir)


if __name__ == "__main__":
    main()
