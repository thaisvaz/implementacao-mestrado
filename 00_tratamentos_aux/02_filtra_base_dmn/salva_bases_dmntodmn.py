# ============================================================
# CONFIG GLOBAL
# ============================================================
import os
import time
from pathlib import Path
import csv
import re

import numpy as np
import pandas as pd
import polars as pl

from nilearn.datasets import fetch_abide_pcp

# seeds
import random
random.seed(19)
np.random.seed(19)

# paths
BASE_DIR = Path.cwd()
RESULTADOS_PATH = Path("_bases/resultados.csv")
FLAGS_PATH = Path("_bases/cc200_dmn_overlap_por_roi.csv")
ABIDE_CACHE_DIR = BASE_DIR / "abide_cache"

# outputs
PARQUET_NO_DIAG = BASE_DIR / "step1_no_diagonal.parquet"
PARQUET_FLAG1 = BASE_DIR / "step2_flag1.parquet"

# ENV CONFIG
# default sem amostragem
TEST = os.getenv("TEST", "0") == "1"
NROWS_TEST = int(os.getenv("NROWS_TEST", "50"))
# default nao levar para o pandas porque jah salva o parquet em paz
TO_PANDAS = os.getenv("TO_PANDAS", "0") == "1"

PAIR_PATTERN = re.compile(r"^(\d+)_(\d+)(?:-(.+))?$")


# ============================================================
# 1. ABIDE TARGET
# ============================================================
def load_abide_target():
    print("\n[STEP] carregar ABIDE target")

    ABIDE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    abide = fetch_abide_pcp(
        pipeline='cpac',
        derivatives=['rois_cc200'],
        band_pass_filtering=False,
        global_signal_regression=False,
        quality_checked=True,
        data_dir=str(ABIDE_CACHE_DIR)
    )

    abide_pheno = pd.DataFrame(abide.phenotypic)

    y = abide_pheno[['SUB_ID', 'DX_GROUP']].copy()
    y['DX_GROUP'] = y['DX_GROUP'].map({1: 1, 2: 0})

    if TEST:
        y = y.iloc[:NROWS_TEST, :]

    print("y shape:", y.shape)
    print("y.valuecounts():\n", y['DX_GROUP'].value_counts())

    return y


# ============================================================
# HELPERS
# ============================================================
def is_diagonal(col):
    m = PAIR_PATTERN.match(col)
    if not m:
        return False
    return int(m.group(1)) == int(m.group(2))


def read_flags(flag_column: str = "flag_dmn_07"):
    """
    Lê arquivo de ROI → DMN e retorna set de ROIs com flag=1

    Ex:
    roi,n_voxels_roi,...,flag_dmn_07
    1,...,1
    2,...,0
    """

    print(f"\n[STEP] lendo flags DMN ({flag_column})")

    flags_file = FLAGS_PATH  # 🔥 ajuste nome aqui

    df_flags = pd.read_csv(flags_file)

    if flag_column not in df_flags.columns:
        raise ValueError(f"Coluna {flag_column} não encontrada no arquivo de flags")

    # pega ROIs com flag=1
    flag_set = set(
        df_flags.loc[df_flags[flag_column] == 1, "roi"].astype(int)
    )

    print(f"[INFO] total ROIs com {flag_column}=1:", len(flag_set))
    print(f"[INFO] exemplos:", sorted(list(flag_set))[:10])

    return flag_set


# ============================================================
# PIPELINE X
# ============================================================
def read_resultados_fast(path: Path):
    print("\n[STEP] pipeline X")

    start = time.time()

    # --------------------------
    # HEADER
    # --------------------------
    with path.open("r", encoding="utf-8") as f:
        header = next(csv.reader(f))

    columns = [c.strip() for c in header if c]
    print("[INFO] total columns:", len(columns))

    # --------------------------
    # REMOVE DIAGONAL
    # --------------------------
    cols_no_diag = [
        c for c in columns
        if c.lower() == "subid" or not is_diagonal(c)
    ]

    print("[INFO] columns sem diagonal:", len(cols_no_diag))

    # --------------------------
    # LAZY LOAD BASE
    # --------------------------
    lf = pl.scan_csv(
        str(path),
        has_header=True,
        infer_schema_length=200,
        low_memory=True,
        rechunk=False,
    )

    if TEST:
        lf = lf.limit(NROWS_TEST)

    # --------------------------
    # STEP 1 → SAVE SEM DIAGONAL
    # --------------------------
    print("\n[STEP] salvando parquet SEM diagonal")

    lf_no_diag = lf.select(cols_no_diag).with_columns(
        pl.all().exclude("subid").cast(pl.Float32)
    )

    lf_no_diag.sink_parquet(
        str(PARQUET_NO_DIAG),
        compression="zstd"
    )

    print("[SAVED]", PARQUET_NO_DIAG)

    # --------------------------
    # READ FLAGS
    # --------------------------
    flag1 = read_flags()
    print("[INFO] flag=1:", sorted(flag1))

    # --------------------------
    # FILTER FLAG
    # --------------------------
    def keep(col):
        if col.lower() == "subid":
            return True

        m = PAIR_PATTERN.match(col)
        if not m:
            return False

        i = int(m.group(1))
        j = int(m.group(2))

        return (i in flag1) and (j in flag1)

    cols_flag = [c for c in cols_no_diag if keep(c)]

    print("[INFO] columns após flag:", len(cols_flag))

    # --------------------------
    # STEP 2 → SAVE FLAG
    # --------------------------
    print("\n[STEP] salvando parquet FLAG=1")

    lf_flag = lf.select(cols_flag).with_columns(
        pl.all().exclude("subid").cast(pl.Float32)
    )

    lf_flag.sink_parquet(
        str(PARQUET_FLAG1),
        compression="zstd"
    )

    print("[SAVED]", PARQUET_FLAG1)

    # --------------------------
    # COLLECT FINAL (flag)
    # --------------------------
    df = lf_flag.collect()

    print("shape:", df.shape)
    print("mem (MB):", round(df.estimated_size("mb"), 2))
    print("tempo total:", round((time.time() - start)/60, 2), "min")

    return df


# ============================================================
# ALIGN
# ============================================================
def align_X_y(df, y):
    print("\n[STEP] align")

    y = y.copy()
    y.set_index("SUB_ID", inplace=True)

    if TO_PANDAS:
        df = df.to_pandas()
        df.set_index("subid", inplace=True)
        aligned = df.index.equals(y.index)
    else:
        aligned = (
            df.select("subid").to_series().to_list()
            == y.index.tolist()
        )

    print("aligned?", aligned)

    return df, y

def save_final_dataset(df, y):
    print("\n[STEP] salvar dataset final alinhado")

    OUTPUT_FINAL = BASE_DIR / "final_dataset.parquet"

    if TO_PANDAS:
        print("[INFO] salvando via pandas")

        df_final = df.copy()
        df_final["target"] = y["DX_GROUP"]

        df_final.to_parquet(OUTPUT_FINAL, index=True)

    else:
        print("[INFO] salvando via polars")

        # transformar y em polars
        y_reset = y.reset_index()
        y_pl = pl.from_pandas(y_reset)

        # join via subid
        df_final = df.join(
            y_pl,
            left_on="subid",
            right_on="SUB_ID",
            how="inner"
        )

        df_final = df_final.rename({
            "DX_GROUP": "target"
        })

        df_final.write_parquet(
            OUTPUT_FINAL,
            compression="zstd"
        )

    print("[SAVED]", OUTPUT_FINAL)

    print("shape final:", df_final.shape)

    return df_final


# ============================================================
# SPLIT FINAL DATASET BY FEATURE FAMILY
# ============================================================
def extract_feature_name(col: str) -> str | None:
    """
    Ex:
    1_7-rtr   -> rtr
    166_200-sts -> sts
    target -> None
    subid -> None
    """
    if col in {"target", "subid"}:
        return None

    if "-" not in col:
        return None

    return col.rsplit("-", 1)[-1].strip()


def get_feature_families_from_columns(columns):
    families = sorted({
        feat for feat in (extract_feature_name(col) for col in columns)
        if feat is not None
    })
    return families


def save_parquets_by_feature_family(df_final, output_dir: Path | None = None):
    """
    Salva um parquet por característica/sufixo:
    rtr, rtx, rty, rts, xtr, xtx, xty, xts, ytr, ytx, yty, yts, str, stx, sty, sts

    Mantém target em todos os parquets.
    Mantém subid se existir.
    """
    print("\n[STEP] salvar um parquet por característica")

    if output_dir is None:
        output_dir = Path("_bases")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # descobrir colunas e famílias
    # --------------------------------------------------
    if isinstance(df_final, pd.DataFrame):
        columns = df_final.columns.tolist()
        is_pandas = True
    else:
        columns = df_final.columns
        is_pandas = False

    feature_families = get_feature_families_from_columns(columns)

    print("[INFO] total de características únicas:", len(feature_families))
    print("[INFO] características encontradas:", feature_families)

    summary_rows = []

    for feat in feature_families:
        feat_cols = [
            col for col in columns
            if extract_feature_name(col) == feat
        ]

        selected_cols = []

        if "subid" in columns:
            selected_cols.append("subid")

        selected_cols.extend(feat_cols)

        if "target" in columns:
            selected_cols.append("target")

        out_path = output_dir / f"{feat}_dmntodmn.parquet"

        print(f"[WRITE] {out_path.name} | n_cols={len(selected_cols)}")
        

        if is_pandas:
            df_feat = df_final[selected_cols].copy()
            df_feat.to_parquet(out_path, index=False)
            n_rows, n_cols = df_feat.shape
        else:
            df_feat = df_final.select(selected_cols)
            df_feat.write_parquet(out_path, compression="zstd")
            n_rows, n_cols = df_feat.shape

        summary_rows.append({
            "feature_family": feat,
            "n_columns_total": n_cols,
            "n_feature_columns_only": len(feat_cols),
            "output_file": out_path.name,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary_feature_families.csv"
    summary_df.to_csv(summary_path, index=False)

    print("[SAVED]", summary_path)

    return feature_families

print("\n========== START ==========")

y = load_abide_target()
df = read_resultados_fast(RESULTADOS_PATH)
df, y = align_X_y(df, y)
df_final = save_final_dataset(df, y)
save_parquets_by_feature_family(df_final)

print("\n========== END ==========")
