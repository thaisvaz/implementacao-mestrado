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

# Configurações de reprodutibilidade
os.environ['PYTHONHASHSEED'] = '19'
random.seed(19)
np.random.seed(19)

# ============================================================
# LEITURA DATASET resultados com a teoria da complexidade
# ============================================================

print("..ler features de complexidade")
start = time.time()
df = pd.read_csv("~/_bases/resultados.csv")
# rodar so para 50 linhas
# df = pd.read_csv("~/_bases/resultados.csv", nrows=5)
print(df.head())
print("Ler a base levou: ", (time.time() - start)/60, "minutos!")

# ============================================================
# CRIAR BASES POR SUFIXO
# ============================================================
print("..salvar bases paralelizadas por sufixo")

sufixos = {
    "rtr", "rtx", "rty", "rts",
    "xtr", "xtx", "xty", "xts",
    "ytr", "ytx", "yty", "yts",
    "str", "stx", "sty", "sts",
}

for s in sufixos:
    cols = [c for c in df.columns if c.endswith(f"-{s}")]
    cols and df[["subid"] + cols].to_parquet(f"~/_bases/{s}.parquet", index=False)
