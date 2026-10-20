"""CLI validation-only: plan local, ejecución explícita y reporte sin GPU."""

from __future__ import annotations

import argparse
import json
import statistics
import zipfile
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, get_output_root
from src.data.preparation import atomic_write_json, sha256_file, sha256_json
from src.training.multiseed import (
    CONFIGURATIONS,
    EXPERIMENT,
    METRICS,
    SEEDS,
    audit_splits,
    canonical_protocol,
    completed_result,
    execute,
    freeze_protocol,
    read_json,
    verify_sources,
)


def statistics_for(values):
    """SD muestral; IC t de Student exploratorio entre seeds, no entre dominios."""
    from scipy.stats import t

    n = len(values)
    if not n:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "ci95_t_exploratory": None,
        }
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else None
    margin = float(t.ppf(0.975, n - 1)) * sd / n**0.5 if n > 1 else None
    return {
        "n": n,
        "mean": mean,
        "median": statistics.median(values),
        "std": sd,
        "min": min(values),
        "max": max(values),
        "ci95_t_exploratory": [mean - margin, mean + margin] if margin is not None else None,
    }


def aggregate(rows):
    completed = [r for r in rows if r["status"] == "COMPLETE"]
    configurations = {
        name: {
            metric: statistics_for([r[metric] for r in completed if r["configuration"] == name])
            for metric in METRICS
        }
        for name in CONFIGURATIONS
    }
    indexed = {(r["configuration"], r["seed"]): r for r in completed}
    pairs = []
    for seed in SEEDS:
        baseline, hpo = indexed.get(("baseline", seed)), indexed.get(("hpo_trial_0", seed))
        if baseline and hpo:
            delta = hpo["macro_f1"] - baseline["macro_f1"]
            pairs.append(
                {
                    "seed": seed,
                    "baseline": baseline["macro_f1"],
                    "hpo_trial_0": hpo["macro_f1"],
                    "delta": delta,
                }
            )
    deltas = [p["delta"] for p in pairs]
    return {
        "complete": len(completed) == 10,
        "completed_runs": len(completed),
        "configurations": configurations,
        "pairs": pairs,
        "paired_delta": statistics_for(deltas),
        "wins_hpo": sum(d > 0 for d in deltas),
        "wins_baseline": sum(d < 0 for d in deltas),
        "ties": sum(d == 0 for d in deltas),
    }


def plots(directory, rows, pairs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    completed = [r for r in rows if r["status"] == "COMPLETE"]
    if not completed:
        return []
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    saved = []
    for metric in ("macro_f1", "ece"):
        fig, ax = plt.subplots()
        for name in CONFIGURATIONS:
            values = [r for r in completed if r["configuration"] == name]
            ax.plot([str(r["seed"]) for r in values], [r[metric] for r in values], "o-", label=name)
        ax.set(xlabel="Seed (split fijo)", ylabel=f"Validation {metric}")
        ax.legend()
        path = figures / f"{metric}_by_seed.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        saved.append(path.relative_to(directory).as_posix())
    fig, ax = plt.subplots()
    for index, name in enumerate(CONFIGURATIONS):
        values = [r["macro_f1"] for r in completed if r["configuration"] == name]
        if values:
            ax.boxplot(values, positions=[index])
            ax.scatter([index] * len(values), values)
    ax.set_xticks([0, 1], CONFIGURATIONS)
    ax.set_ylabel("Validation Macro-F1")
    fig.savefig(figures / "macro_f1_distribution.png", bbox_inches="tight")
    plt.close(fig)
    saved.append("figures/macro_f1_distribution.png")
    fig, ax = plt.subplots()
    if pairs:
        ax.bar([str(p["seed"]) for p in pairs], [p["delta"] for p in pairs])
    ax.axhline(0, color="black")
    ax.set(xlabel="Seed", ylabel="HPO − baseline (Macro-F1)")
    fig.savefig(figures / "paired_delta.png", bbox_inches="tight")
    plt.close(fig)
    saved.append("figures/paired_delta.png")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, metric in zip(axes, ("nitrogen_f1", "phosphorus_f1", "potassium_f1")):
        for index, name in enumerate(CONFIGURATIONS):
            values = [r[metric] for r in completed if r["configuration"] == name]
            ax.scatter([index] * len(values), values)
        ax.set_xticks([0, 1], CONFIGURATIONS, rotation=20)
        ax.set_title(metric)
    fig.savefig(figures / "npk_by_configuration.png", bbox_inches="tight")
    plt.close(fig)
    saved.append("figures/npk_by_configuration.png")
    return saved


def report(directory):
    directory = Path(directory)
    protocol = read_json(directory / "PROTOCOL.json")
    rows = []
    for seed in SEEDS:
        for name in CONFIGURATIONS:
            slot = directory / name / f"seed_{seed}"
            row = {
                "configuration": name,
                "seed": seed,
                "status": "PENDING",
                **{m: None for m in METRICS},
            }
            if (slot / "COMPLETE.json").exists():
                receipt = completed_result(directory, protocol, name, seed)
                row.update({k: v for k, v in receipt.items() if k != "artifact_hashes"})
            elif (slot / "FAILED.json").exists():
                row["status"] = "FAILED"
            elif (slot / "STARTED.json").exists():
                row["status"] = "STARTED_WITHOUT_COMPLETION"
            rows.append(row)
    summary = aggregate(rows)
    summary.update(
        {
            "experiment_id": EXPERIMENT,
            "protocol_sha256": sha256_json(protocol),
            "test_used": False,
            "results": rows,
            "figures": plots(directory, rows, summary["pairs"]),
        }
    )
    atomic_write_json(directory / "MULTISEED_SUMMARY.json", summary)
    pd.DataFrame(rows).to_csv(directory / "MULTISEED_RESULTS.csv", index=False)
    lines = [
        "# Comparación multi-seed — validation exclusivamente",
        "",
        f"Completadas: {summary['completed_runs']}/10. Test NO evaluado.",
        "",
        "| Configuración | Seed | Estado | Macro-F1 | Accuracy | N F1 | P F1 | K F1 | ECE |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = ["—" if row[m] is None else f"{row[m]:.6f}" for m in METRICS]
        lines.append(
            f"| {row['configuration']} | {row['seed']} | {row['status']} | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "## Estadísticos descriptivos (SD muestral)",
        "",
        "| Configuración | Métrica | n | Media | SD | Mediana | Min | Max | IC95% t exploratorio |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    def fmt(value):
        return "—" if value is None else f"{value:.6f}"

    for name, metrics in summary["configurations"].items():
        for metric, stats in metrics.items():
            fields = [fmt(stats[k]) for k in ("mean", "std", "median", "min", "max")]
            ci = stats["ci95_t_exploratory"]
            lines.append(
                f"| {name} | {metric} | {stats['n']} | "
                + " | ".join(fields)
                + f" | {str(ci) if ci else '—'} |"
            )
    lines += [
        "",
        "## Comparación pareada",
        "",
        "| Seed | Baseline | HPO | Delta HPO − baseline |",
        "|---:|---:|---:|---:|",
    ]
    for pair in summary["pairs"]:
        lines.append(
            f"| {pair['seed']} | {pair['baseline']:.6f} | {pair['hpo_trial_0']:.6f}"
            f" | {pair['delta']:.6f} |"
        )
    lines += [
        "",
        f"Delta medio: {fmt(summary['paired_delta']['mean'])}; "
        f"SD: {fmt(summary['paired_delta']['std'])}. HPO gana {summary['wins_hpo']}; "
        f"baseline gana {summary['wins_baseline']}; empates {summary['ties']}.",
        "",
        "## Interpretación y límites",
        "",
        (
            "Cohorte completa; valorar conjuntamente media, variabilidad, deltas, N/P/K "
            "(especialmente potasio) y ECE. No se promociona una configuración automáticamente."
            if summary["complete"]
            else "Experimento incompleto: no hay evidencia suficiente para preferir "
            "una configuración. "
            "Los pendientes no son ceros y no se sustituyen por las runs históricas."
        ),
        "",
        "Cinco semillas sobre un split fijo no equivalen a cross-validation, LOSO ni "
        "generalización entre dominios. IC t exploratorio bajo independencia aproximada "
        "entre seeds y normalidad; con n=5 es frágil. No se realizan pruebas de significancia.",
        "La configuración HPO fue seleccionada sobre esta misma validation: persiste "
        "sesgo de selección. Cambiar seed no crea un conjunto de validación independiente.",
        "Siguiente paso: completar y revisar esta cohorte; ninguna fase posterior se ejecuta.",
        "",
        f"Protocolo SHA-256: `{sha256_json(protocol)}`.",
    ]
    (directory / "MULTISEED_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_write_json(
        directory / "REPORT_HASHES.json",
        {
            p.name: sha256_file(p)
            for p in (
                directory / "MULTISEED_SUMMARY.json",
                directory / "MULTISEED_RESULTS.csv",
                directory / "MULTISEED_REPORT.md",
                directory / "PROTOCOL.json",
                directory / "PREFLIGHT.json",
                directory / "source_snapshot.zip",
            )
            if p.is_file()
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "report"))
    parser.add_argument(
        "--output-dir", type=Path, default=get_output_root() / "multiseed" / EXPERIMENT
    )
    parser.add_argument("--splits-dir", type=Path, default=get_output_root() / "splits/seed_42")
    parser.add_argument("--max-new-runs", type=int, default=10)
    args = parser.parse_args()
    if args.action == "report":
        print(json.dumps({"completed": report(args.output_dir)["completed_runs"]}))
        return
    protocol_path = args.output_dir / "PROTOCOL.json"
    protocol = read_json(protocol_path) if protocol_path.exists() else canonical_protocol()
    if args.action == "plan":
        verify_sources(protocol)
        audit = audit_splits(args.splits_dir, protocol["split_hashes"])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        freeze_protocol(args.output_dir, protocol)
        atomic_write_json(args.output_dir / "PREFLIGHT.json", audit)
        archive = args.output_dir / "source_snapshot.zip"
        if not archive.exists():
            with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
                for relative in sorted(
                    set(protocol["source_files"]) | set(protocol["canonical_files"])
                ):
                    bundle.write(PROJECT_ROOT / relative, relative)
                bundle.write(PROJECT_ROOT / "config/dataset.yaml", "config/dataset.yaml")
        report(args.output_dir)
        print("Plan: 2 configuraciones × 5 semillas = 10 entrenamientos. No se ha iniciado GPU.")
    else:
        try:
            execute(args.output_dir, args.splits_dir, protocol, max_new_runs=args.max_new_runs)
        finally:
            if protocol_path.exists():
                report(args.output_dir)


if __name__ == "__main__":
    main()
