#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
archive_dir="$(dirname "$package_dir")/doctor_maiz_model_archives"
mkdir -p "$archive_dir"

make_tar() {
  local name="$1"
  shift
  local target="$archive_dir/$name.tar.gz"
  if [[ -e "$target" ]]; then
    echo "Archivo previo: $target; no se sobrescribe" >&2
    exit 1
  fi
  tar -C "$package_dir" -czf "$target" "$@"
  tar -tzf "$target" >/dev/null
}

make_tar DoctorMaiz_Baseline_Lite0 baseline_lite0
make_tar DoctorMaiz_HPO_Lite0 hpo_lite0
make_tar DoctorMaiz_LOSO_MaizeDiseases loso_maize_diseases
make_tar DoctorMaiz_LOSO_Multicrop_Lite0 loso_multicrop/lite0
make_tar DoctorMaiz_LOSO_Multicrop_B0 loso_multicrop/b0
make_tar DoctorMaiz_LOSO_Multicrop_ShuffleNet loso_multicrop/shufflenet
make_tar DoctorMaiz_Historicos_B0_ShuffleNet historical_standard
make_tar DoctorMaiz_Modelos_Metadata_Resultados \
  README.md RESULTS_SUMMARY.md MODEL_INDEX.csv MODEL_COMPARISON_RESULTS.csv \
  MODEL_CHECKSUMS.sha256 CREATE_RARS.sh CREATE_TARS.sh metadata

(
  cd "$archive_dir"
  sha256sum ./*.tar.gz > TAR_CHECKSUMS.sha256
)
echo "Ocho TAR.GZ verificados en: $archive_dir"
