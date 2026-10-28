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
df = pd.read_csv(os.path.expanduser("~/_bases/resultados.csv"))
# rodar so para 50 linhas
# df = pd.read_csv(os.path.expanduser("~/_bases/resultados.csv"), nrows=5)
print(df.head())
print("Ler a base levou: ", (time.time() - start)/60, "minutos!")

# ============================================================
# PREPARAÇÃO BÁSICA (sem subid, tudo numérico)
# ============================================================

# remove identificador das análises
df = df.drop(columns=["subid"], errors="ignore")

# garante que tudo é numérico uma única vez
df_num = df.apply(pd.to_numeric, errors="coerce")

# constantes
SUFIXOS = [
    "rtr","rtx","rty","rts",
    "xtr","xtx","xty","xts",
    "ytr","ytx","yty","yts",
    "str","stx","sty","sts",
]

out_dir = os.path.expanduser("~/_analises/_plots")
os.makedirs(out_dir, exist_ok=True)

import re
import matplotlib.pyplot as plt

# ============================================================
# como estão as distribuições de cada característica? boxplot por característica
# ============================================================

data = []
counts = []

for suf in SUFIXOS:
    cols = df_num.filter(regex=rf"-{re.escape(suf)}$").columns
    if len(cols) == 0:
        data.append(np.array([]))
        counts.append(0)
        continue

    vals = df_num[cols].to_numpy().ravel()
    vals = vals[np.isfinite(vals)]
    data.append(vals)
    counts.append(vals.size)

print("N por característica:", dict(zip(SUFIXOS, counts)))

plt.figure(figsize=(18, 6))
plt.boxplot(data, labels=SUFIXOS, showfliers=False)
plt.title("Distribuição dos valores por característica (16 sufixos)")
plt.xlabel("Característica (sufixo)")
plt.ylabel("Valor")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

png_path = os.path.join(out_dir, "boxplots_16_caracteristicas.png")
plt.savefig(png_path, dpi=160)
plt.show()

print("Salvo em:", png_path)

# ============================================================
# como estão os valores máximos e mínimo de cada característica?
#  min, max, mean, median, p25, p50, p75
# ============================================================

rows = []

for suf in SUFIXOS:
    cols = df_num.filter(regex=rf"-{re.escape(suf)}$").columns
    if len(cols) == 0:
        continue

    vals = df_num[cols].to_numpy().ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        continue

    p25, p50, p75 = np.percentile(vals, [25, 50, 75])

    rows.append({
        "caracteristica": suf,
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "N": int(vals.size),
    })

summary_df = (
    pd.DataFrame(rows)
    .sort_values("caracteristica")
    .reset_index(drop=True)
)

print(summary_df)

out_path = os.path.expanduser("~/_analises/tabela_resumo_caracteristicas.csv")
summary_df.to_csv(out_path, index=False)
print("Tabela salva em:", out_path)

# ============================================================
# quais os resultados de cada característica por roixroi igual? (preciso retirar )
# ============================================================

pat = re.compile(r"^(?P<i>\d+)_(?P<j>\d+)-(?P<suf>[a-z]+)$")

# descobrir ROIs disponíveis na diagonal
rois = sorted({
    int(m.group("i"))
    for c in df_num.columns
    for m in [pat.match(c)]
    if m and m.group("i") == m.group("j") and m.group("suf") in SUFIXOS
})

# mapear colunas da diagonal existentes
diag_cols = []
diag_pairs = []

for roi in rois:
    for suf in SUFIXOS:
        col = f"{roi}_{roi}-{suf}"
        if col in df_num.columns:
            diag_cols.append(col)
            diag_pairs.append((roi, suf))

if len(diag_cols) == 0:
    print("Nenhuma coluna ROIxROI (diagonal) encontrada.")
else:
    df_diag = df_num[diag_cols].copy()
    df_diag.columns = pd.MultiIndex.from_tuples(
        diag_pairs, names=["roi_diag", "sufixo"]
    )

    def build_table(stat):
        """
        stat ∈ {'min', 'max', 'mean'}
        Retorna tabela: linhas=ROI (i=i), colunas=sufixos
        """
        s = df_diag.agg(stat, axis=0)
        tbl = (
            s.unstack("sufixo")
             .reindex(index=rois, columns=SUFIXOS)
        )
        tbl.index.name = "roi_diag"
        return tbl

    def plot_heatmap(table, stat):
        M = table.to_numpy(dtype=float)

        plt.figure(figsize=(12, max(6, len(table) * 0.06)))
        im = plt.imshow(M, aspect="auto", interpolation="nearest")
        plt.title(f"Heatmap diagonal ROIxROI (i=i) — {stat}")
        plt.xlabel("Característica (sufixo)")
        plt.ylabel("ROI (i=i)")

        plt.xticks(
            np.arange(len(table.columns)),
            table.columns.tolist(),
            rotation=45,
            ha="right"
        )

        idx = table.index.to_list()
        if len(idx) <= 60:
            plt.yticks(np.arange(len(idx)), [str(r) for r in idx])
        else:
            step = max(1, len(idx)//20)
            yt = np.arange(0, len(idx), step)
            plt.yticks(yt, [str(idx[i]) for i in yt])

        plt.colorbar(im, label=stat)
        plt.tight_layout()

        png_path = os.path.join(out_dir, f"heatmap_diagonal_{stat}.png")
        plt.savefig(png_path, dpi=180)
        plt.show()

        print("Salvo:", png_path)

    for stat in ["min", "max", "mean"]:
        tbl = build_table(stat)

        csv_path = os.path.join(out_dir, f"tabela_diagonal_{stat}.csv")
        tbl.to_csv(csv_path)
        print("Tabela salva:", csv_path)

        plot_heatmap(tbl, stat)
