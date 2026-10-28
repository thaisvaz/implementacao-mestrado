# ============================================================
# BIBLIOTECAS
# ============================================================

# Biblioteca padrão
import os
import random
import time
import concurrent.futures

# Manipulação de dados
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configurações de reprodutibilidade
os.environ['PYTHONHASHSEED'] = '19'
random.seed(19)
np.random.seed(19)

# ============================================================
# LEITURA VAR RESPOSTA (BASE NILEARN)
# ============================================================

from nilearn.datasets import fetch_abide_pcp

print("..ler base var resposta")
abide = fetch_abide_pcp(
    pipeline="cpac",
    derivatives=["rois_cc200"],
    band_pass_filtering=False,
    global_signal_regression=False,
    quality_checked=True
)
abide_pheno = pd.DataFrame(abide.phenotypic)

y = abide_pheno[["SUB_ID", "DX_GROUP"]].copy()
y["DX_GROUP"] = y["DX_GROUP"].map({1: 1, 2: 0})
print("y.valuecounts()", y["DX_GROUP"].value_counts())

# ============================================================
# SPLIT TRAIN E TEST
# ============================================================

train_subids, test_subids = train_test_split(
    y,
    test_size=0.1,
    stratify=y["DX_GROUP"],
    random_state=19,
    shuffle=True,
)

print("\nDistribuicao no treino:")
print(train_subids["DX_GROUP"].value_counts(normalize=False))
print(train_subids["DX_GROUP"].value_counts(normalize=True))

print("\nDistribuicao no teste:")
print(test_subids["DX_GROUP"].value_counts(normalize=False))
print(test_subids["DX_GROUP"].value_counts(normalize=True))

# Salvar bases com subids de treino e teste
BASE_DIR = os.path.expanduser("~/_bases")
os.makedirs(BASE_DIR, exist_ok=True)

train_subids.to_parquet(os.path.join(BASE_DIR, "train_subids.parquet"), index=False)
test_subids.to_parquet(os.path.join(BASE_DIR, "test_subids.parquet"), index=False)

# # ============================================================
# # LEITURA DATASET resultados com a teoria da complexidade
# # ============================================================

# print("..ler features de complexidade")
# start = time.time()
# df = pd.read_csv("~/_bases/resultados.csv")
# # rodar so para 50 linhas
# # df = pd.read_csv("~/_bases/resultados.csv", nrows=5)
# print(df.head())
# print("Ler a base levou: ", (time.time() - start)/60, "minutos!")

# # ============================================================
# # CRIAR BASES POR SUFIXO
# # ============================================================
# print("..salvar bases paralelizadas por sufixo")

# sufixos = {
#     "rtr", "rtx", "rty", "rts",
#     "xtr", "xtx", "xty", "xts",
#     "ytr", "ytx", "yty", "yts",
#     "str", "stx", "sty", "sts",
# }

# for s in sufixos:
#     cols = [c for c in df.columns if c.endswith(f"-{s}")]
#     cols and df[["subid"] + cols].to_parquet(f"~/_bases/{s}.parquet", index=False)
