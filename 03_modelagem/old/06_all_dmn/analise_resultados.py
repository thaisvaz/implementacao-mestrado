# ============================================================
# BIBLIOTECAS
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
    "output_dir": Path("output_modelagem"),
    "plots_dir": Path("output_modelagem/plots"),
    "yaml_pattern": "*.yaml",
    "figure_width": 6,
    "figure_height": 8,
    "dpi": 150,
    "x_jitter_width": 0.08,
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


def find_yaml_files(output_dir: Path, pattern: str = "*.yaml") -> list[Path]:
    files = sorted(output_dir.rglob(pattern))
    print(f"Encontrados {len(files)} YAML(s) em {output_dir}")
    for file in files:
        print(" -", file)
    return files


# ============================================================
# NORMALIZAÇÃO DOS YAMLs
# ============================================================
def extract_records_from_model_metrics_list(data, source_name: str) -> list[dict]:
    records = []

    if not isinstance(data, list):
        return records

    for item in data:
        if not isinstance(item, dict):
            continue

        model_name = item.get("model_name", "unknown_model")

        for split in ["train", "test"]:
            split_data = item.get(split, {})
            if not isinstance(split_data, dict):
                continue

            records.append({
                "source": source_name,
                "model_name": model_name,
                "split": split,
                "auc": split_data.get("auc"),
                "average_precision": split_data.get("average_precision"),
            })

    return records


def extract_records_from_single_dict(data, source_name: str) -> list[dict]:
    records = []

    if not isinstance(data, dict):
        return records

    if "train" not in data and "test" not in data:
        return records

    model_name = data.get("model_name", source_name)

    for split in ["train", "test"]:
        split_data = data.get(split, {})
        if not isinstance(split_data, dict):
            continue

        records.append({
            "source": source_name,
            "model_name": model_name,
            "split": split,
            "auc": split_data.get("auc"),
            "average_precision": split_data.get("average_precision"),
        })

    return records


def normalize_yaml_to_records(yaml_path: Path) -> list[dict]:
    data = load_yaml_file(yaml_path)
    source_name = yaml_path.stem

    records = []
    records.extend(extract_records_from_model_metrics_list(data, source_name))
    records.extend(extract_records_from_single_dict(data, source_name))

    return records


def build_metrics_dataframe(yaml_files: list[Path]) -> pd.DataFrame:
    all_records = []

    for yaml_path in yaml_files:
        records = normalize_yaml_to_records(yaml_path)
        if not records:
            print(f"Aviso: YAML sem estrutura reconhecida -> {yaml_path}")
            continue
        all_records.extend(records)

    if not all_records:
        raise ValueError("Nenhum registro válido foi encontrado nos YAMLs.")

    df = pd.DataFrame(all_records)

    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df["average_precision"] = pd.to_numeric(df["average_precision"], errors="coerce")

    df = df.dropna(subset=["model_name", "split"], how="any").copy()

    df["split"] = pd.Categorical(
        df["split"],
        categories=["test", "train"],
        ordered=True
    )

    print("\nDataFrame wide final:")
    print(df.head())
    print("\nShape:", df.shape)

    return df


def reshape_to_long_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df_long = df.melt(
        id_vars=["source", "model_name", "split"],
        value_vars=["average_precision", "auc"],
        var_name="metric",
        value_name="value"
    ).copy()

    metric_map = {
        "average_precision": "Average Precision",
        "auc": "AUC"
    }
    df_long["metric"] = df_long["metric"].map(metric_map)

    df_long = df_long.dropna(subset=["value"]).copy()

    df_long["metric"] = pd.Categorical(
        df_long["metric"],
        categories=["Average Precision", "AUC"],
        ordered=True
    )

    print("\nDataFrame long final:")
    print(df_long.head())
    print("\nShape:", df_long.shape)

    return df_long


# ============================================================
# PLOT
# ============================================================
def make_metrics_plot(df_long: pd.DataFrame, output_path: Path) -> None:
    p = (
        ggplot(
            df_long,
            aes(
                x="metric",
                y="value",
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
            title="AUC e Average Precision por modelo",
            x="Métrica",
            y="Valor",
            color="Modelo",
            shape="Split"
        )
        + theme_minimal()
        + theme(
            figure_size=(CONFIG["figure_width"], CONFIG["figure_height"]),
            axis_text_x=element_text(rotation=0, ha="center"),
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
    output_dir = CONFIG["output_dir"]
    plots_dir = CONFIG["plots_dir"]

    ensure_dir(plots_dir)

    yaml_files = find_yaml_files(
        output_dir=output_dir,
        pattern=CONFIG["yaml_pattern"]
    )

    if not yaml_files:
        raise FileNotFoundError(f"Nenhum YAML encontrado em {output_dir}")

    df_metrics = build_metrics_dataframe(yaml_files)

    consolidated_csv = plots_dir / "metrics_consolidated.csv"
    df_metrics.to_csv(consolidated_csv, index=False)
    print(f"CSV consolidado salvo em: {consolidated_csv}")

    df_long = reshape_to_long_metrics(df_metrics)

    long_csv = plots_dir / "metrics_long.csv"
    df_long.to_csv(long_csv, index=False)
    print(f"CSV long salvo em: {long_csv}")

    make_metrics_plot(
        df_long=df_long,
        output_path=plots_dir / "metrics_plot.png"
    )

    print("✔ Plot finalizado com sucesso")


if __name__ == "__main__":
    main()