# ============================================================
# MATERIALIZA BASES POR CARACTERÍSTICA
# (STORAGE, COPY, TRANSFER, ERASURE, DOWNWARD, UPWARD)
# ============================================================
#
# Lê a base original "resultados_no_diagonal.parquet" (todos os pares de
# ROI, todos os 16 códigos de matriz, sem filtro de DMN) e monta, por
# característica, um parquet novo combinando as colunas dos códigos que
# compõem aquela característica. Roda uma única vez; o modelagem.py só
# lê os parquets já materializados.

import time
from pathlib import Path

import polars as pl

# ============================================================
# CONFIG
# ============================================================
BASE_PATH = Path("~/_bases").expanduser()
SOURCE_PATH = BASE_PATH / "resultados_no_diagonal.parquet"

DATASET_GROUPS = {
    "storage": ["rtr", "xtx", "yty", "sts"],
    "copy": ["xtr", "ytr"],
    "transfer": ["xty", "yty"],
    "erasure": ["rtx", "rty"],
    "downward": ["str", "stx", "sty"],
    "upward": ["xts", "yts", "rts"],
}


# ============================================================
# HELPERS
# ============================================================
def get_feature_code(col: str) -> str | None:
    if col == "subid":
        return None
    if "-" not in col:
        return None
    return col.rsplit("-", 1)[-1].strip()


def build_dataset(
    name: str,
    codes: list[str],
    schema_names: list[str],
    lazy_source: pl.LazyFrame,
    base_path: Path
) -> Path:
    print(f"\n[STEP] montando dataset '{name}' a partir de: {codes}")
    start = time.time()

    code_set = set(codes)
    feature_cols = [
        col for col in schema_names
        if get_feature_code(col) in code_set
    ]

    if not feature_cols:
        raise ValueError(f"Nenhuma coluna encontrada para os códigos {codes}")

    print(f"[INFO] total de colunas de features: {len(feature_cols)}")

    df = lazy_source.select(["subid"] + feature_cols).collect()

    output_path = base_path / f"{name}.parquet"
    df.write_parquet(output_path, compression="zstd")

    print(f"[SAVED] {output_path} | shape={df.shape}")
    print(f"[TIME] '{name}' levou {(time.time() - start) / 60:.2f} minutos")

    return output_path


# ============================================================
# MAIN
# ============================================================
def main():
    print("========== START ==========")
    print("[INFO] fonte:", SOURCE_PATH)

    lazy_source = pl.scan_parquet(SOURCE_PATH)
    schema_names = lazy_source.collect_schema().names()

    print("[INFO] total de colunas na fonte:", len(schema_names))

    for name, codes in DATASET_GROUPS.items():
        build_dataset(name, codes, schema_names, lazy_source, BASE_PATH)

    print("\n========== END ==========")


if __name__ == "__main__":
    main()
