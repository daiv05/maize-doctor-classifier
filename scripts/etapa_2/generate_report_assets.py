"""Genera todas las figuras y tablas LaTeX de ETAPA 2 desde CSV/JSON."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.loader import load_and_normalize_image
from src.provenance import sha256_file

CLASS_LABELS = {
    "common_rust": "Roya común",
    "fall_armyworm": "Gusano cogollero",
    "gray_leaf_spot": "Mancha gris",
    "healthy": "Sana",
    "lethal_necrosis": "Necrosis letal",
    "nitrogen_deficiency": "Def. nitrógeno",
    "northern_corn_leaf_blight": "Tizón norteño",
    "phosphorus_deficiency": "Def. fósforo",
    "potassium_deficiency": "Def. potasio",
}
MODEL_LABELS = {
    "efficientnet_b0": "EfficientNet-B0",
    "shufflenet_v2_x1_0": "ShuffleNetV2-x1.0",
    "mobilenet_v3_small": "MobileNetV3 Small",
    "fastvit_t8": "FastViT-T8",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def write_table(path: Path, headers: list[str], rows: list[list[object]], align: str) -> None:
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule"]
    lines.append(" & ".join(r"\textbf{" + tex_escape(header) + "}" for header in headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(tex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def metric(value: float) -> str:
    return f"{value:.4f}"


def generate(output_dir: Path, report_dir: Path, dataset_root: Path) -> None:
    figures = report_dir / "figures"
    tables = report_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    manifest = pd.read_csv(output_dir / "master_manifest.csv")
    dataset_summary = load_json(output_dir / "dataset_summary.json")
    tuning_summary = load_json(output_dir / "tuning" / "summary.json")
    models = pd.read_csv(output_dir / "model_comparison" / "models.csv")
    methods = pd.read_csv(output_dir / "ensemble" / "methods.csv")
    selection = load_json(output_dir / "ensemble" / "selection.json")
    cv = pd.read_csv(output_dir / "cross_validation" / "folds.csv")
    cv_summary = load_json(output_dir / "cross_validation" / "summary.json")
    final_metrics = load_json(output_dir / "final" / "metrics.json")
    report = pd.read_csv(output_dir / "final" / "classification_report.csv", index_col=0)
    matrix = pd.read_csv(output_dir / "final" / "confusion_matrix.csv", index_col=0)
    fairness_groups = pd.read_csv(output_dir / "fairness" / "by_group.csv")
    fairness_gaps = pd.read_csv(output_dir / "fairness" / "gaps.csv")
    fairness_environment_class = pd.read_csv(
        output_dir / "fairness" / "environment_within_class.csv"
    )

    # Dataset: barras apiladas lab/campo.
    distribution = manifest.groupby(["label", "environment"]).size().unstack(fill_value=0)
    distribution = distribution.loc[distribution.sum(axis=1).sort_values().index]
    ax = distribution[[column for column in ["lab", "real"] if column in distribution]].plot(
        kind="barh", stacked=True, figsize=(9, 5.5), color=["#4c78a8", "#59a14f"]
    )
    ax.set_xlabel("Imágenes únicas")
    ax.set_ylabel("")
    ax.set_yticklabels([CLASS_LABELS.get(value, value) for value in distribution.index])
    ax.legend(["Laboratorio", "Campo"], title="Entorno")
    ax.set_title("Corpus congelado para ETAPA 2")
    plt.tight_layout()
    plt.savefig(figures / "dataset_distribution.pdf")
    plt.savefig(figures / "dataset_distribution.png", dpi=180)
    plt.close()

    trials = pd.read_csv(output_dir / "tuning" / "trials.csv")
    complete = trials[trials["state"].eq("COMPLETE")].copy()
    complete["best_so_far"] = complete["value"].cummax()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.scatter(complete["number"] + 1, complete["value"], color="#4c78a8", label="Trial completo")
    ax.plot(
        complete["number"] + 1, complete["best_so_far"], color="#e45756", label="Mejor acumulado"
    )
    ax.axhline(
        tuning_summary["baseline_metrics"]["macro_f1"],
        color="#777777",
        linestyle="--",
        label="Baseline",
    )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Macro-F1 de validación")
    ax.set_title("Historial de optimización con Optuna")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures / "optuna_history.pdf")
    plt.close()

    importance = pd.read_csv(output_dir / "tuning" / "parameter_importance.csv")
    if not importance.empty:
        importance = importance.sort_values("importance")
        fig, ax = plt.subplots(figsize=(8.2, 4.5))
        ax.barh(importance["parameter"], importance["importance"], color="#f28e2b")
        ax.set_xlabel("Importancia relativa")
        ax.set_title("Importancia de hiperparámetros")
        plt.tight_layout()
        plt.savefig(figures / "optuna_importance.pdf")
        plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        models["fp32_parameter_size_mb"],
        models["macro_f1"],
        s=80,
        c=models["cpu_latency_ms_median"],
        cmap="viridis_r",
    )
    for row in models.itertuples():
        ax.annotate(
            MODEL_LABELS.get(row.model, row.model),
            (row.fp32_parameter_size_mb, row.macro_f1),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Tamaño FP32 estimado por parámetros (MB)")
    ax.set_ylabel("Macro-F1 de validación")
    ax.set_title("Calidad y costo de los backbones evaluados")
    plt.tight_layout()
    plt.savefig(figures / "model_comparison.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    method_labels = [
        MODEL_LABELS.get(value, value.replace("_", " ")) for value in methods["method"]
    ]
    colors = ["#4c78a8" if value == "best_individual" else "#f28e2b" for value in methods["type"]]
    ax.bar(method_labels, methods["macro_f1"], color=colors)
    ax.set_ylabel("Macro-F1 de validación")
    ax.set_title("Modelo individual frente a soft voting")
    ax.tick_params(axis="x", rotation=12)
    ax.set_ylim(max(0, methods["macro_f1"].min() - 0.08), min(1, methods["macro_f1"].max() + 0.03))
    plt.tight_layout()
    plt.savefig(figures / "ensemble_comparison.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cv["fold"], cv["macro_f1"], marker="o", color="#4c78a8", linewidth=2)
    mean = cv_summary["metrics"]["macro_f1"]["mean"]
    ax.axhline(mean, linestyle="--", color="#e45756", label=f"Media = {mean:.4f}")
    ax.set_xticks(cv["fold"])
    ax.set_xlabel("Fold")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Variación en validación cruzada")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures / "cross_validation.pdf")
    plt.close()

    labels_short = [CLASS_LABELS.get(value, value) for value in matrix.index]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    sns.heatmap(matrix, cmap="Blues", annot=True, fmt="d", cbar=False, ax=axes[0])
    axes[0].set_title("Conteos")
    normalized = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
    sns.heatmap(normalized, cmap="Blues", annot=True, fmt=".2f", cbar=False, ax=axes[1])
    axes[1].set_title("Normalizada por clase real")
    for axis in axes:
        axis.set_xticklabels(labels_short, rotation=55, ha="right", fontsize=7)
        axis.set_yticklabels(labels_short, rotation=0, fontsize=7)
        axis.set_xlabel("Predicción")
        axis.set_ylabel("Clase real")
    plt.tight_layout()
    plt.savefig(figures / "final_confusion_matrices.pdf")
    plt.close()

    class_report = report.loc[list(CLASS_LABELS)].copy()
    class_report.index = [CLASS_LABELS[value] for value in class_report.index]
    class_report[["precision", "recall", "f1-score"]].plot(
        kind="bar", figsize=(10, 5), color=["#4c78a8", "#f28e2b", "#59a14f"]
    )
    plt.ylabel("Métrica")
    plt.xlabel("")
    plt.ylim(0, 1.03)
    plt.title("Rendimiento final por clase")
    plt.xticks(rotation=35, ha="right")
    plt.legend(["Precision", "Recall", "F1"])
    plt.tight_layout()
    plt.savefig(figures / "final_per_class.pdf")
    plt.close()

    env = fairness_groups[fairness_groups["dimension"].eq("environment")].copy()
    if not env.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        env.set_index("group")[["precision", "recall", "f1"]].plot(
            kind="bar", ax=axes[0], color=["#4c78a8", "#f28e2b", "#59a14f"]
        )
        axes[0].set_ylabel("Macro métrica dentro del grupo")
        axes[0].set_xlabel("Entorno")
        axes[0].set_ylim(0, 1.03)
        axes[0].set_title("Resumen global (composición distinta)")
        axes[0].tick_params(axis="x", rotation=0)
        if not fairness_environment_class.empty:
            controlled = fairness_environment_class.pivot(
                index="class", columns="environment", values="recall"
            )
            controlled.index = [CLASS_LABELS.get(value, value) for value in controlled.index]
            controlled.plot(kind="bar", ax=axes[1], color=["#4c78a8", "#59a14f"])
            axes[1].set_ylabel("Recall dentro de clase")
            axes[1].set_xlabel("")
            axes[1].set_ylim(0, 1.03)
            axes[1].set_title("Comparación controlada por clase")
            axes[1].tick_params(axis="x", rotation=20)
        fig.suptitle("Laboratorio frente a campo: lectura descriptiva")
        plt.tight_layout()
        plt.savefig(figures / "fairness_environment.pdf")
        plt.close()

    # Captura reproducible del prototipo: foto real y salida JSON generada por inferencia.
    demo = load_json(output_dir / "prototype" / "demo_result.json")
    # Rebind only the documented clean-relative path, not an arbitrary missing filename.
    historical_image = Path(demo["image"])
    parts = historical_image.parts
    if "clean" not in parts:
        raise ValueError("Prototype image lacks a clean-relative source path")
    image_path = dataset_root.joinpath(*parts[parts.index("clean") :])
    source_row = manifest.loc[
        manifest.image_path.eq(image_path.relative_to(dataset_root).as_posix())
    ]
    if len(source_row) != 1 or sha256_file(image_path) != source_row.iloc[0]["sha256"]:
        raise ValueError("Prototype image differs from the archived manifest")
    image = load_and_normalize_image(image_path)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), gridspec_kw={"width_ratios": [1.1, 1.4]})
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Entrada")
    axes[1].axis("off")
    lines = [
        "DOCTOR MAÍZ — INFERENCIA LOCAL",
        "",
        f"Modelo: {demo['model']}",
        f"Latencia CPU: {demo['latency_ms']:.1f} ms",
        "",
        "Top-3:",
    ]
    lines.extend(f"  {item['class']}: {item['probability']:.1%}" for item in demo["top_k"])
    lines.extend(
        [
            "",
            "Advertencia de baja confianza: " + ("sí" if demo["low_confidence_warning"] else "no"),
            "",
            demo["disclaimer"],
        ]
    )
    axes[1].text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10, wrap=True)
    axes[1].set_title("Salida real del prototipo")
    plt.tight_layout()
    plt.savefig(figures / "prototype_capture.png", dpi=180)
    plt.close()

    # Las explicaciones se generan con los modelos finales; aquí solamente se
    # incorporan al árbol autocontenido del informe.
    interpretability_dir = output_dir / "interpretability"
    for name in ("final_gradcam_cases.png", "final_lime_case.png"):
        source = interpretability_dir / name
        if source.exists():
            shutil.copy2(source, figures / name)

    # Tablas derivadas de los artefactos.
    e1_counts = {
        "common_rust": 2256,
        "fall_armyworm": 4857,
        "gray_leaf_spot": 1119,
        "healthy": 8744,
        "lethal_necrosis": 6415,
        "nitrogen_deficiency": 523,
        "northern_corn_leaf_blight": 6830,
        "phosphorus_deficiency": 612,
        "potassium_deficiency": 266,
    }
    current_counts = manifest["label"].value_counts().to_dict()
    dataset_rows = []
    for class_name in CLASS_LABELS:
        dataset_rows.append(
            [
                CLASS_LABELS[class_name],
                f"{e1_counts[class_name]:,}".replace(",", "~"),
                f"{current_counts[class_name]:,}".replace(",", "~"),
                f"{current_counts[class_name] - e1_counts[class_name]:+d}",
            ]
        )
    dataset_rows.append(
        ["Total", "31~622", f"{len(manifest):,}".replace(",", "~"), f"{len(manifest) - 31622:+d}"]
    )
    write_table(
        tables / "dataset_etapa1_vs_etapa2.tex",
        ["Clase", "Etapa 1", "Etapa 2", "Cambio"],
        dataset_rows,
        "lrrr",
    )

    baseline_params = load_json(output_dir / "tuning" / "baseline_params.json")
    best_params = load_json(output_dir / "tuning" / "best_params.json")
    param_rows = []
    for key in [
        "alpha",
        "eta0",
        "penalty",
        "l1_ratio",
        "batch_size",
        "balance_power",
        "average",
        "feature_norm",
        "epochs",
    ]:
        param_rows.append([key, baseline_params.get(key, "--"), best_params.get(key, "--")])
    write_table(
        tables / "tuning_configuration.tex",
        ["Hiperparámetro", "Baseline", "Mejor valor"],
        param_rows,
        "lcc",
    )

    top_trials = complete.nlargest(5, "value")
    trial_rows = [
        [int(row.number) + 1, metric(row.value), row.state] for row in top_trials.itertuples()
    ]
    write_table(tables / "top_trials.tex", ["Trial", "Macro-F1 val.", "Estado"], trial_rows, "rrl")

    baseline_vs = pd.read_csv(output_dir / "tuning" / "baseline_vs_tuned.csv")
    write_table(
        tables / "baseline_vs_tuned.tex",
        ["Configuración", "Accuracy", "Macro-F1", "F1 ponderado"],
        [
            [
                row.configuration.replace("_", " "),
                metric(row.accuracy),
                metric(row.macro_f1),
                metric(row.weighted_f1),
            ]
            for row in baseline_vs.itertuples()
        ],
        "lrrr",
    )

    model_rows = []
    for row in models.itertuples():
        model_rows.append(
            [
                MODEL_LABELS.get(row.model, row.model),
                f"{row.parameters / 1e6:.2f} M",
                f"{row.fp32_parameter_size_mb:.1f}",
                f"{row.cpu_latency_ms_median:.1f}",
                metric(row.accuracy),
                metric(row.macro_precision),
                metric(row.macro_recall),
                metric(row.macro_f1),
            ]
        )
    write_table(
        tables / "model_comparison.tex",
        ["Modelo", "Parám.", "MB", "ms", "Acc.", "P-macro", "R-macro", "F1-macro"],
        model_rows,
        "lrrrrrrr",
    )

    write_table(
        tables / "ensemble.tex",
        ["Método", "Accuracy", "P-macro", "R-macro", "F1-macro", "F1 pond."],
        [
            [
                MODEL_LABELS.get(str(row.method), str(row.method).replace("_", " ")),
                metric(row.accuracy),
                metric(row.macro_precision),
                metric(row.macro_recall),
                metric(row.macro_f1),
                metric(row.weighted_f1),
            ]
            for row in methods.itertuples()
        ],
        "lrrrrr",
    )

    cv_rows = [
        [
            int(row.fold),
            metric(row.accuracy),
            metric(row.macro_precision),
            metric(row.macro_recall),
            metric(row.macro_f1),
            metric(row.weighted_f1),
        ]
        for row in cv.itertuples()
    ]
    cv_rows.append(
        [
            "Media",
            *[
                metric(cv_summary["metrics"][name]["mean"])
                for name in [
                    "accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "weighted_f1",
                ]
            ],
        ]
    )
    cv_rows.append(
        [
            "Desv.",
            *[
                metric(cv_summary["metrics"][name]["std"])
                for name in [
                    "accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "weighted_f1",
                ]
            ],
        ]
    )
    write_table(
        tables / "cross_validation.tex",
        ["Fold", "Accuracy", "P-macro", "R-macro", "F1-macro", "F1 pond."],
        cv_rows,
        "lrrrrr",
    )

    write_table(
        tables / "final_metrics.tex",
        ["Accuracy", "P-macro", "R-macro", "F1-macro", "F1 pond.", "N"],
        [
            [
                metric(final_metrics["accuracy"]),
                metric(final_metrics["macro_precision"]),
                metric(final_metrics["macro_recall"]),
                metric(final_metrics["macro_f1"]),
                metric(final_metrics["weighted_f1"]),
                final_metrics["n"],
            ]
        ],
        "rrrrrr",
    )

    per_class_rows = []
    for class_name in CLASS_LABELS:
        row = report.loc[class_name]
        per_class_rows.append(
            [
                CLASS_LABELS[class_name],
                int(row["support"]),
                metric(row["precision"]),
                metric(row["recall"]),
                metric(row["f1-score"]),
            ]
        )
    write_table(
        tables / "final_per_class.tex",
        ["Clase", "Soporte", "Precision", "Recall", "F1"],
        per_class_rows,
        "lrrrr",
    )

    gap_rows = [
        [row.dimension, metric(row.precision_gap), metric(row.recall_gap), metric(row.f1_gap)]
        for row in fairness_gaps.itertuples()
    ]
    write_table(
        tables / "fairness_gaps.tex",
        ["Dimensión", "Gap precision", "Gap recall", "Gap F1"],
        gap_rows,
        "lrrr",
    )

    potassium = report.loc["potassium_deficiency"]
    stage_rows = [
        ["Accuracy", "0.9521", metric(final_metrics["accuracy"]), "Corpus y protocolo distintos"],
        ["Macro-F1", "0.9146", metric(final_metrics["macro_f1"]), "Corpus y protocolo distintos"],
        ["Recall potasio", "--", metric(potassium["recall"]), "No comparable"],
        ["F1 potasio", "0.62", metric(potassium["f1-score"]), "Referencia B0 histórica"],
        [
            "Imágenes potasio",
            "266",
            str(current_counts["potassium_deficiency"]),
            f"+{current_counts['potassium_deficiency'] - 266}",
        ],
    ]
    write_table(
        tables / "etapa1_vs_etapa2.tex",
        ["Métrica", "Etapa 1", "Etapa 2", "Lectura"],
        stage_rows,
        "lrrl",
    )

    rubric_path = report_dir.parent / "rubrica_final.json"
    if rubric_path.exists():
        rubric = load_json(rubric_path)
        lines = [
            r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X r l "
            r">{\raggedright\arraybackslash}X}",
            r"\toprule",
        ]
        lines.append(
            r"\textbf{Criterio} & \textbf{Puntos} & \textbf{Estado} & \textbf{Evidencia} \\"
        )
        lines.append(r"\midrule")
        for row in rubric["criteria"]:
            lines.append(
                " & ".join(
                    [
                        tex_escape(row["criterion"]),
                        tex_escape(f"{row['awarded']}/{row['maximum']}"),
                        tex_escape(row["status"]),
                        tex_escape(row["evidence_short"]),
                    ]
                )
                + r" \\"
            )
        lines.extend(
            [
                r"\midrule",
                " & ".join(
                    [
                        r"\textbf{TOTAL}",
                        rf"\textbf{{{rubric['awarded_total']}/{rubric['maximum_total']}}}",
                        tex_escape(rubric["overall_status"]),
                        tex_escape(rubric["overall_note"]),
                    ]
                )
                + r" \\",
                r"\bottomrule",
                r"\end{tabularx}",
            ]
        )
        (tables / "rubric.tex").write_text("\n".join(lines) + "\n")

    manifest_payload = {
        "schema_version": 2,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset_fingerprint": dataset_summary["fingerprint_sha256"],
        "selected_model": selection["selected_name"],
        "figures": sorted(path.name for path in figures.iterdir() if path.is_file()),
        "tables": sorted(path.name for path in tables.iterdir() if path.is_file()),
        "generator_sha256": sha256_file(Path(__file__)),
        "asset_sha256": {
            path.relative_to(report_dir).as_posix(): sha256_file(path)
            for directory in (figures, tables)
            for path in directory.iterdir()
            if path.is_file()
        },
        "input_sha256": {
            path.relative_to(output_dir).as_posix(): sha256_file(path)
            for path in output_dir.rglob("*")
            if path.is_file() and path.suffix in {".csv", ".json"}
        },
    }
    (report_dir / "MANIFEST.generated.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/etapa_2"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/etapa_2/informe"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output_dir.resolve(), args.report_dir.resolve(), args.dataset_root.resolve())


if __name__ == "__main__":
    main()
