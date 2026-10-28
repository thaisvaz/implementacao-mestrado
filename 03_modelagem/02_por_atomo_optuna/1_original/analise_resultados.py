# ============================================================
# ANALISE DE RESULTADOS MULTI-BASES DMN
# ============================================================

from pathlib import Path
import yaml
import pandas as pd

from plotnine import (
    ggplot,
    aes,
    geom_point,
    labs,
    theme_minimal,
    theme,
    element_text,
    element_rect,
    position_jitter,
    scale_shape_manual
)


# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "output_root": Path(__file__).resolve().parent / "output_modelagem",
    "plots_dir": Path(__file__).resolve().parent / "output_modelagem" / "plots",
    "yaml_filename": "model_metrics.yaml",
    "figure_width": 14,
    "figure_height": 6,
    "dpi": 150,
    "x_jitter_width": 0.10,
    "point_size": 2.4,
}


# ============================================================
# IO
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml_file(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_yaml_files(output_root: Path, yaml_filename: str) -> list[Path]:
    files = sorted(output_root.rglob(yaml_filename))

    # evita pegar YAML dentro de plots, se um dia existir
    files = [f for f in files if "plots" not in f.parts]

    print(f"Encontrados {len(files)} YAML(s) em {output_root}")
    for file in files:
        print(" -", file)

    return files


# ============================================================
# NORMALIZAÇÃO DOS YAMLs
# ============================================================
def extract_feature_set(dataset_name: str) -> str:
    return dataset_name


def normalize_yaml_to_records(yaml_path: Path) -> list[dict]:
    data = load_yaml_file(yaml_path)

    if not isinstance(data, list):
        print(f"Aviso: YAML sem estrutura de lista -> {yaml_path}")
        return []

    records = []

    for item in data:
        if not isinstance(item, dict):
            continue

        dataset_name = item.get("dataset", yaml_path.parent.name)
        feature_set = extract_feature_set(dataset_name)
        model_name = item.get("model_name", "unknown_model")

        for split in ["train", "test"]:
            split_data = item.get(split, {})
            if not isinstance(split_data, dict):
                continue

            records.append({
                "dataset": dataset_name,
                "feature_set": feature_set,
                "model_name": model_name,
                "split": split,
                "auc": split_data.get("auc"),
                "average_precision": split_data.get("average_precision"),
            })

    return records


def build_metrics_dataframe(yaml_files: list[Path]) -> pd.DataFrame:
    all_records = []

    for yaml_path in yaml_files:
        records = normalize_yaml_to_records(yaml_path)
        if not records:
            print(f"Aviso: YAML sem registros válidos -> {yaml_path}")
            continue
        all_records.extend(records)

    if not all_records:
        raise ValueError("Nenhum registro válido foi encontrado nos YAMLs.")

    df = pd.DataFrame(all_records)

    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df["average_precision"] = pd.to_numeric(df["average_precision"], errors="coerce")

    df = df.dropna(subset=["model_name", "split", "feature_set"], how="any").copy()

    feature_order = [
        "rtr", "rts", "rtx", "rty",
        "str", "sts", "stx", "sty",
        "xtr", "xts", "xtx", "xty",
        "ytr", "yts", "ytx", "yty",
    ]

    existing_feature_sets = df["feature_set"].dropna().unique().tolist()
    ordered_present = [x for x in feature_order if x in existing_feature_sets]
    unordered_rest = sorted([x for x in existing_feature_sets if x not in ordered_present])

    df["feature_set"] = pd.Categorical(
        df["feature_set"],
        categories=ordered_present + unordered_rest,
        ordered=True
    )

    df["split"] = pd.Categorical(
        df["split"],
        categories=["test", "train"],
        ordered=True
    )

    print("\nDataFrame consolidado:")
    print(df.head())
    print("\nShape:", df.shape)

    return df


# ============================================================
# PLOTS
# ============================================================
def make_metric_plot(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    y_label: str,
    output_path: Path
) -> None:
    df_plot = df.dropna(subset=[metric_col]).copy()

    p = (
        ggplot(
            df_plot,
            aes(
                x="feature_set",
                y=metric_col,
                color="model_name",
                shape="split"
            )
        )
        + geom_point(
            size=CONFIG["point_size"],
            position=position_jitter(
                width=CONFIG["x_jitter_width"],
                height=0
            )
        )
        + scale_shape_manual(
            values={
                "train": "x",
                "test": "o",
            }
        )
        + labs(
            title=title,
            x="Feature set",
            y=y_label,
            color="Modelo",
            shape="Split"
        )
        + theme_minimal()
        + theme(
            figure_size=(CONFIG["figure_width"], CONFIG["figure_height"]),
            axis_text_x=element_text(rotation=35, ha="right"),
            legend_position="right",
            panel_border=element_rect(fill=None, color="black"),
        )
    )

    p.save(
        filename=str(output_path),
        width=CONFIG["figure_width"],
        height=CONFIG["figure_height"],
        dpi=CONFIG["dpi"]
    )

    print(f"Gráfico salvo em: {output_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    output_root = CONFIG["output_root"]
    plots_dir = CONFIG["plots_dir"]

    ensure_dir(plots_dir)

    yaml_files = find_yaml_files(
        output_root=output_root,
        yaml_filename=CONFIG["yaml_filename"]
    )

    if not yaml_files:
        raise FileNotFoundError(
            f"Nenhum {CONFIG['yaml_filename']} encontrado em {output_root}"
        )

    df_metrics = build_metrics_dataframe(yaml_files)

    consolidated_csv = plots_dir / "metrics_consolidated.csv"
    df_metrics.to_csv(consolidated_csv, index=False)
    print(f"CSV consolidado salvo em: {consolidated_csv}")

    make_metric_plot(
        df=df_metrics,
        metric_col="average_precision",
        title="Average Precision (Treino vs Teste) por Feature Set",
        y_label="Average Precision",
        output_path=plots_dir / "average_precision_por_feature_set.png"
    )

    make_metric_plot(
        df=df_metrics,
        metric_col="auc",
        title="AUC (Treino vs Teste) por Feature Set",
        y_label="AUC",
        output_path=plots_dir / "auc_por_feature_set.png"
    )

    print("✔ Análise finalizada com sucesso")


if __name__ == "__main__":
    main()