# ============================================================
# BIBLIOTECAS
# ============================================================

import os
import random
import time

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pickle
import yaml

import matplotlib.pyplot as plt


# ============================================================
# CONFIGS
# ============================================================

os.environ["PYTHONHASHSEED"] = "19"
random.seed(19)
np.random.seed(19)

DEBUG_50_ROWS = False   # <<<<<<<<<<<<<<<<<<<  MODO TESTE

BASE_DIR = os.path.expanduser("~/_bases")

# pasta fixa pedida
OUT_DIR = os.path.join("output", "rts_reg_log_l2")
os.makedirs(OUT_DIR, exist_ok=True)

# timestamp fixo pedido
timestamp = "2026_02_24"


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

y_aligned = y.copy()
y_aligned.set_index("SUB_ID", inplace=True)


# ============================================================
# FUNCOES DE AVALIACAO
# ============================================================

def calcula_metricas(y_true, y_pred, model_name="modelo"):
    acc  = accuracy_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "recall": float(rec),
        "precision": float(prec),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def score_model(model, X, y_true):
    # score contínuo (logit/margem), e predição binária
    try:
        y_score = model.decision_function(X)
    except AttributeError:
        y_score = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    auc = roc_auc_score(y_true, y_score)
    ap  = average_precision_score(y_true, y_score)
    mets = calcula_metricas(y_true, y_pred, model_name="tmp")

    return {
        "auc": float(auc),
        "average_precision": float(ap),
        "accuracy": float(mets["accuracy"]),
        "recall": float(mets["recall"]),
        "precision": float(mets["precision"]),
        "f1": float(mets["f1"]),
        "tn": int(mets["tn"]),
        "fp": int(mets["fp"]),
        "fn": int(mets["fn"]),
        "tp": int(mets["tp"]),
    }


# ============================================================
# 1 - TREINO REGRESSAO LOGISTICA L2 + FEATURE IMPORTANCE
# ============================================================

print("..ler parquet rts.parquet")
pf = "rts.parquet"
df = pd.read_parquet(os.path.join(BASE_DIR, pf))
df = df.copy()
df.set_index("subid", inplace=True)

# alinhar X e y
common = df.index.intersection(y_aligned.index)
if DEBUG_50_ROWS:
    common = common[:50]

df = df.loc[common]
y_use = y_aligned.loc[common, "DX_GROUP"]

print("X shape:", df.shape, "| y shape:", y_use.shape)

# split (fixo) -> vamos REUSAR no item 2
X_train, X_test, y_train, y_test = train_test_split(
    df, y_use, test_size=0.1, stratify=y_use, random_state=19
)
train_idx = X_train.index.copy()
test_idx  = X_test.index.copy()

print("split fixo | X_train:", X_train.shape, "| X_test:", X_test.shape)

# pipeline com scaler (para coeficientes fazerem sentido)
model_name = "Logistic Regression L2"
run_prefix = "reg_log_l2"

pipe = Pipeline(steps=[
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("clf", LogisticRegression(
        penalty="l2",
        C=0.1,
        solver="saga",
        max_iter=5000,
        random_state=19
    ))
])

print(f"..treino modelo {model_name} | (com StandardScaler) | run={run_prefix}")
start_time = time.time()
pipe.fit(X_train, y_train)
exec_time = time.time() - start_time
print(f"treinar o {model_name} levou: {exec_time/60:.2f} minutos!")

# salvar pickle
model_filename = os.path.join(OUT_DIR, f"{run_prefix}_{timestamp}_model.pkl")
with open(model_filename, "wb") as f:
    pickle.dump(pipe, f)
print(f"Modelo salvo em {model_filename}")

# métricas train/test
train_scores = score_model(pipe, X_train, y_train)
test_scores  = score_model(pipe, X_test,  y_test)

base_results = [{
    "feature_set": "rts",
    "model_name": f"rts__{model_name}",
    "exec_time_minutes": float(exec_time / 60),
    "train": train_scores,
    "test": test_scores
}]

# YAML com resultados
yaml_filename = os.path.join(OUT_DIR, f"{run_prefix}_metrics_{timestamp}.yaml")
with open(yaml_filename, "w") as f:
    yaml.dump(base_results, f, sort_keys=False)
print(f"Métricas salvas em {yaml_filename}")

# feature importance (abs(coef)) — em dados padronizados
clf = pipe.named_steps["clf"]
coef = clf.coef_.ravel()
feat_names = list(X_train.columns)

imp_df = pd.DataFrame({
    "feature": feat_names,
    "coef": coef,
    "abs_coef": np.abs(coef),
})
imp_df = imp_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
imp_df["rank"] = np.arange(1, len(imp_df) + 1)

# salvar parquet com importâncias
imp_parquet = os.path.join(OUT_DIR, f"{run_prefix}_feature_importance_{timestamp}.parquet")
imp_df.to_parquet(imp_parquet, index=False)
print(f"Feature importance salva em {imp_parquet}")

# salvar png (top N)
TOP_N = min(30, imp_df.shape[0])
plot_df = imp_df.head(TOP_N).iloc[::-1]  # inverter p/ ficar “maior” em cima no barh

plt.figure(figsize=(12, max(6, 0.28 * TOP_N)))
plt.barh(plot_df["feature"], plot_df["abs_coef"])
plt.xlabel("|coef| (dados padronizados)")
plt.title(f"Top {TOP_N} Features — {model_name} (L2) — rts")
plt.tight_layout()

imp_png = os.path.join(OUT_DIR, f"{run_prefix}_feature_importance_top{TOP_N}_{timestamp}.png")
plt.savefig(imp_png, dpi=160)
plt.close()
print(f"Gráfico salvo em {imp_png}")


# ============================================================
# 2 - TREINO INCREMENTAL (ADD 1 FEATURE POR VEZ) (MESMO SPLIT)
# ============================================================

print("\n==============================")
print("TREINO INCREMENTAL (MESMO SPLIT)")
print("==============================\n")

ordered_features = imp_df["feature"].tolist()

# vamos guardar uma linha por incremento
rows = []

# referência p/ ganho
ap_ref = None

for k in range(1, len(ordered_features) + 1):
    added_feature = ordered_features[k - 1]
    feats_k = ordered_features[:k]

    # manter o MESMO split (índices)
    Xtr_k = df.loc[train_idx, feats_k]
    Xte_k = df.loc[test_idx,  feats_k]
    ytr_k = y_use.loc[train_idx]
    yte_k = y_use.loc[test_idx]

    # pipeline com scaler (refit a cada k)
    pipe_k = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            penalty="l2",
            C=0.1,
            solver="saga",
            max_iter=5000,
            random_state=19
        ))
    ])

    print(f"..treinando agora com k={k} | adicionou={added_feature}")
    print("   X_train shape:", Xtr_k.shape, "| X_test shape:", Xte_k.shape)

    pipe_k.fit(Xtr_k, ytr_k)

    tr = score_model(pipe_k, Xtr_k, ytr_k)
    te = score_model(pipe_k, Xte_k, yte_k)

    if ap_ref is None:
        ap_ref = te["average_precision"]

    gain_ap = te["average_precision"] - ap_ref

    # prints pedidos
    print(f"   TRAIN | ap={tr['average_precision']:.6f} | auc={tr['auc']:.6f} | tp={tr['tp']} fp={tr['fp']} tn={tr['tn']} fn={tr['fn']}")
    print(f"   TEST  | ap={te['average_precision']:.6f} | auc={te['auc']:.6f} | tp={te['tp']} fp={te['fp']} tn={te['tn']} fn={te['fn']}\n")

    rows.append({
        "feature_added": added_feature,
        "avg_precision": float(te["average_precision"]),
        "auc": float(te["auc"]),
        "n_features": int(k),
        "gain_avg_precision_vs_first": float(gain_ap),

        # extras úteis (mantive como colunas adicionais)
        "train_avg_precision": float(tr["average_precision"]),
        "train_auc": float(tr["auc"]),
        "train_tp": int(tr["tp"]),
        "train_fp": int(tr["fp"]),
        "train_tn": int(tr["tn"]),
        "train_fn": int(tr["fn"]),
        "test_tp": int(te["tp"]),
        "test_fp": int(te["fp"]),
        "test_tn": int(te["tn"]),
        "test_fn": int(te["fn"]),
    })

inc_df = pd.DataFrame(rows)

# salvar tabela incremental
inc_parquet = os.path.join(OUT_DIR, f"{run_prefix}_incremental_{timestamp}.parquet")
inc_df.to_parquet(inc_parquet, index=False)
print(f"Tabela incremental salva em {inc_parquet}")

# salvar YAML incremental
inc_yaml = os.path.join(OUT_DIR, f"{run_prefix}_incremental_{timestamp}.yaml")
with open(inc_yaml, "w") as f:
    yaml.dump(rows, f, sort_keys=False)
print(f"YAML incremental salvo em {inc_yaml}")

# ============================================================
# GRAFICOS INCREMENTAIS (AP e AUC) (train=x, test=o)
# ============================================================

# eixo X como a feature adicionada (categorico)
x_labels = inc_df["feature_added"].tolist()
x = np.arange(len(x_labels))

# --- Average Precision ---
plt.figure(figsize=(max(12, 0.35 * len(x_labels)), 6))
plt.scatter(x, inc_df["train_avg_precision"].values, marker="x", s=40, label="train")
plt.scatter(x, inc_df["avg_precision"].values,       marker="o", s=30, label="test")
plt.xticks(x, x_labels, rotation=35, ha="right")
plt.ylabel("Average Precision")
plt.title("Average Precision (Treino vs Teste) — incremental por feature adicionada")
plt.legend()
plt.tight_layout()

ap_png = os.path.join(OUT_DIR, f"{run_prefix}_incremental_avg_precision_{timestamp}.png")
plt.savefig(ap_png, dpi=160)
plt.close()
print(f"Gráfico AP salvo em {ap_png}")

# --- AUC ---
plt.figure(figsize=(max(12, 0.35 * len(x_labels)), 6))
plt.scatter(x, inc_df["train_auc"].values, marker="x", s=40, label="train")
plt.scatter(x, inc_df["auc"].values,       marker="o", s=30, label="test")
plt.xticks(x, x_labels, rotation=35, ha="right")
plt.ylabel("AUC")
plt.title("AUC (Treino vs Teste) — incremental por feature adicionada")
plt.legend()
plt.tight_layout()

auc_png = os.path.join(OUT_DIR, f"{run_prefix}_incremental_auc_{timestamp}.png")
plt.savefig(auc_png, dpi=160)
plt.close()
print(f"Gráfico AUC salvo em {auc_png}")

print("\nOK! Tudo salvo em:", OUT_DIR)