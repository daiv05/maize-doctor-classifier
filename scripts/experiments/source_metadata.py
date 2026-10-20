"""Resume tamaños de imagen ya inventariados en la auditoría de píxeles."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.inventory, dtype={"sample_id": str})
    if len(frame) != 33429 or frame["sample_id"].duplicated().any():
        raise ValueError("El inventario no corresponde a las 33,429 muestras elegibles")
    required = {"source_id", "width", "height", "file_extension", "error"}
    if not required.issubset(frame):
        raise ValueError(f"Faltan columnas: {sorted(required - set(frame))}")
    if frame["error"].notna().any() or frame[["width", "height"]].isna().any().any():
        raise ValueError("El inventario contiene errores o dimensiones desconocidas")
    frame["megapixels"] = frame["width"] * frame["height"] / 1_000_000
    frame["resolution"] = frame["width"].astype(str) + "x" + frame["height"].astype(str)
    rows = []
    for source, group in frame.groupby("source_id"):
        common = group["resolution"].value_counts()
        rows.append(
            {
                "source_id": source,
                "samples": len(group),
                "median_width": group["width"].median(),
                "median_height": group["height"].median(),
                "median_megapixels": group["megapixels"].median(),
                "distinct_resolutions": group["resolution"].nunique(),
                "most_common_resolution": common.index[0],
                "most_common_resolution_n": int(common.iloc[0]),
                "pct_most_common_resolution": 100 * common.iloc[0] / len(group),
            }
        )
    output = pd.DataFrame(rows).sort_values("samples", ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "source_image_metadata.csv", index=False)


if __name__ == "__main__":
    main()
