#!/usr/bin/env bash
set -euo pipefail

if ! command -v rar >/dev/null 2>&1; then
  echo "Falta 'rar'. No se creó ningún archivo .rar." >&2
  exit 1
fi

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
archive_dir="$(dirname "$package_dir")/doctor_maiz_model_archives"
mkdir -p "$archive_dir"
cd "$package_dir"

rar a -r -m3 "$archive_dir/DoctorMaiz_Baseline_Lite0.rar" baseline_lite0
rar a -r -m3 "$archive_dir/DoctorMaiz_HPO_Lite0.rar" hpo_lite0
rar a -r -m3 "$archive_dir/DoctorMaiz_LOSO_MaizeDiseases.rar" loso_maize_diseases
rar a -r -m3 "$archive_dir/DoctorMaiz_LOSO_Multicrop_Lite0.rar" loso_multicrop/lite0
rar a -r -m3 "$archive_dir/DoctorMaiz_LOSO_Multicrop_B0.rar" loso_multicrop/b0
rar a -r -m3 "$archive_dir/DoctorMaiz_LOSO_Multicrop_ShuffleNet.rar" loso_multicrop/shufflenet
rar a -r -m3 "$archive_dir/DoctorMaiz_Historicos_B0_ShuffleNet.rar" historical_standard
rar a -r -m3 "$archive_dir/DoctorMaiz_Modelos_Metadata_Resultados.rar" \
  README.md RESULTS_SUMMARY.md MODEL_INDEX.csv MODEL_COMPARISON_RESULTS.csv \
  MODEL_CHECKSUMS.sha256 CREATE_RARS.sh CREATE_TARS.sh metadata

for archive in "$archive_dir"/*.rar; do
  rar t "$archive"
done
(
  cd "$archive_dir"
  sha256sum ./*.rar > RAR_CHECKSUMS.sha256
)
echo "RAR verificados en: $archive_dir"
