# ============================================================
# BIBLIOTECAS
# ============================================================

import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from nan_corr import convert_numeric_float32, pairwise_corr_nancorrmp


# ============================================================
# CONFIG
# ============================================================

BASE = os.path.expanduser("~/_bases")
OUTPUT = os.path.join(BASE, "output_corr")

os.makedirs(OUTPUT, exist_ok=True)

SUFIXOS = [
    "rtr","rtx","rty","rts",
    "xtr","xtx","xty","xts",
    "ytr","ytx","yty","yts",
    "str","stx","sty","sts"
]

CHUNKS = 1000


# ============================================================
# LEITURA BASE DE IDS DE TRAIN E TEST
# ============================================================

train_ids = pd.read_parquet(f"{BASE}/train_subids.parquet")
test_ids = pd.read_parquet(f"{BASE}/test_subids.parquet")

train_ids = set(train_ids["SUB_ID"])
test_ids = set(test_ids["SUB_ID"])


# ============================================================
# PROCESSAR UMA BASE
# ============================================================

def process_base(sufixo):

    start = time.time()

    path = f"{BASE}/{sufixo}.parquet"

    print(f"\n..lendo {path}")

    df = pd.read_parquet(path)

    print("..filtrando treino")

    df = df[df["subid"].isin(train_ids)].copy()

    print("..shape treino:", df.shape)

    # remover subid das features
    df_features = df.drop(columns=["subid"])

    # converter float32
    df_features = convert_numeric_float32(df_features)

    print("..n features:", df_features.shape[1])

    corr = pairwise_corr_nancorrmp(
        df=df_features,
        features=list(df_features.columns),
        n_jobs=1,
        chunks=CHUNKS
    )

    out = f"{OUTPUT}/{sufixo}_correlations.csv"

    corr.to_csv(out, index=False)

    elapsed = (time.time() - start) / 60

    return f"{sufixo} terminado | {elapsed:.2f} min"


# ============================================================
# MAIN
# ============================================================

def main():

    print("..iniciando pipeline")

    workers = min(len(SUFIXOS), mp.cpu_count()-1)

    print("..workers:", workers)

    with ProcessPoolExecutor(max_workers=workers) as executor:

        futures = [executor.submit(process_base, s) for s in SUFIXOS]

        for f in as_completed(futures):
            print(f.result())


if __name__ == "__main__":
    main()