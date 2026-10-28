A ideia dessa etapa é ir adicionando as features mais importantes encontradas no modelo l2 de regressão logística, que teve melhor resultado dentre os demais tipos de modelos. E fazer um modelo incrementando essas features de forma recorrente.
Ir plotando o resultado. Eixo x colocar a feature que foi adicionada. Eixo y valor de average precision. Em treino e teste. O mesmo gráfico utilizando anteriormente.

```
# ============================================================
# BIBLIOTECAS
# ============================================================

import os
import random
import time
import concurrent.futures

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
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

import lightgbm as lgb
from sklearn.svm import LinearSVC

import pickle
import yaml

# ============================================================
# CONFIGS
# ============================================================

os.environ["PYTHONHASHSEED"] = "19"
random.seed(19)
np.random.seed(19)

DEBUG_50_ROWS = False   # <<<<<<<<<<<<<<<<<<<  MODO TESTE

BASE_DIR = os.path.expanduser("~/_bases")

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


def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, feature_set):
    print(f"..treino modelo {model_name} | feature_set={feature_set}")
    start_time = time.time()

    os.makedirs("_output", exist_ok=True)

    # id da rodada
    timestamp = 21012026  # ou int(time.time())

    model_slug = model_name.lower().replace(" ", "_")
    run_prefix = f"{feature_set}__{model_slug}"

    if hasattr(model, "random_state"):
        model.random_state = 19

    model.fit(X_train, y_train)

    # salvar pickle
    model_filename = f"_output/{run_prefix}_{timestamp}_model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo salvo em {model_filename}")

    exec_time = time.time() - start_time
    print(f"treinar o {model_name} ({feature_set}) levou: {exec_time/60:.2f} minutos!")

    # train
    try:
        y_score_train = model.decision_function(X_train)
    except AttributeError:
        y_score_train = model.predict_proba(X_train)[:, 1]
    y_pred_train = model.predict(X_train)

    # test
    try:
        y_score_test = model.decision_function(X_test)
    except AttributeError:
        y_score_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = model.predict(X_test)

    # avaliar
    auc_train = roc_auc_score(y_train, y_score_train)
    ap_train = average_precision_score(y_train, y_score_train)
    metrics_train = calcula_metricas(y_train, y_pred_train, f"{run_prefix}_train")

    auc_test = roc_auc_score(y_test, y_score_test)
    ap_test = average_precision_score(y_test, y_score_test)
    metrics_test = calcula_metricas(y_test, y_pred_test, f"{run_prefix}_test")

    model_results = [{
        "feature_set": feature_set,
        "model_name": f"{feature_set}__{model_name}",
        "exec_time_minutes": exec_time / 60,
        "train": {
            "auc": float(auc_train),
            "average_precision": float(ap_train),
            **metrics_train
        },
        "test": {
            "auc": float(auc_test),
            "average_precision": float(ap_test),
            **metrics_test
        }
    }]

    # scorados
    score_col = f"{run_prefix}_score"

    train_scorado = pd.DataFrame({
        "subid": X_train.index,
        "y_true": y_train,
        score_col: y_score_train
    })
    train_scorado_filename = f"_output/{run_prefix}_train_scorado_{timestamp}.parquet"
    train_scorado.to_parquet(train_scorado_filename)

    test_scorado = pd.DataFrame({
        "subid": X_test.index,
        "y_true": y_test,
        score_col: y_score_test
    })
    test_scorado_filename = f"_output/{run_prefix}_test_scorado_{timestamp}.parquet"
    test_scorado.to_parquet(test_scorado_filename)

    # YAML
    yaml_filename = f"_output/{run_prefix}_metrics_{timestamp}.yaml"
    with open(yaml_filename, "w") as f:
        yaml.dump(model_results, f, sort_keys=False)

    print(f"Métricas salvas em {yaml_filename}")


# ============================================================
# MODELOS
# ============================================================

models = [
    ("LinearSVC L2", LinearSVC(penalty="l2", C=0.1, dual=True, tol=0.0001, max_iter=10000, random_state=19)),
    ("Ridge Classifier", RidgeClassifier(alpha=10.0)),
    ("Logistic Regression L2", LogisticRegression(penalty="l2", C=0.1, solver="saga", max_iter=5000, random_state=19)),
    ("Logistic Regression ElasticNet", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.1, solver="saga", max_iter=5000, random_state=19)),
    ("SGD Classifier", SGDClassifier(loss="hinge", alpha=1e-4, max_iter=2000, tol=1e-3, random_state=19)),
    ("LightGBM", lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        num_leaves=10,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        random_state=19
    )),
]

# ============================================================
# RODAR POR FEATURE_SET (16 PARQUETS)
# ============================================================

parquet_files = sorted([
    f for f in os.listdir(BASE_DIR)
    if f.endswith(".parquet") and f != "resultados.parquet"
])

print("Parquets encontrados:", parquet_files)

y_aligned = y.copy()
y_aligned.set_index("SUB_ID", inplace=True)

for pf in parquet_files:
    feature_set = pf.replace(".parquet", "")
    print(f"\n==============================")
    print(f"RODANDO FEATURE_SET = {feature_set}")
    print(f"==============================\n")

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

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        df, y_use, test_size=0.1, stratify=y_use, random_state=19
    )

    # treinar em paralelo (por feature_set)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                train_and_evaluate_model,
                model, name,
                X_train, X_test, y_train, y_test,
                feature_set
            )
            for name, model in models
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Erro em modelo ({feature_set}): {e}")

print("Todos os modelos foram treinados e avaliados (por feature_set).")

```
com base nesse codigo, tentando alterar o minimo. Deixe o mesmo estilo de comentario de sessao, quero manter o que escrevi



1 - Treino da regressão logística com l2. E pego as features mais importantes. Em uma pasta output/reg_log_l2:
a) colocar a data de hoje 2026_02_24 no timestamp fixo.
b) aplicar o standard scaler, para a analise dos coeficientes fazerem sentido.
O codigo tem que fazer:
a) Salvar pickle do modelo.
b) yaml com resultados de treino e teste do modelo.
c) salvar um gráfico com features mais importantes em png.
d) salvar tabela com features mais importantes em parquet


2 - Treino de um novo modelo, incrementando cada feature mais importante, uma a uma. ou seja, primeiro faz um modelo com a feature mais importante 1, depois faz um modelo com a feature mais importante 1 e 2, depois 1,2 e 3. E assim por diante, até usar todas as features. Mantido o split de treino e teste do modelo inicial, para conservar a explicabilidade.
a) fazer uma tabela com o nome da feature adicional na primeira coluna, avg precision segunda coluna, auc terceira coluna, quantidade de features na quarta coluna, ganho de avg precision em relacao a primeira linha na quinta coluna.
b) fazer um gráfico com eixo x sendo a nova feature adicionada. e o eixo y o resultado da average precision. Com o resultado em treino e em teste. O treino sendo x e o treino sendo bolinha.
c) fazer um gráfico com eixo x sendo a nova feature adicionada. e o eixo y o resultado da auc. Com o resultado em treino e em teste. O treino sendo x e o treino sendo bolinha.
d) colocar no código para que ele va printando qual a variavel que ele esta usando para treinar agora. E printar o resultado da average precision e auc, tp, fp, tn, fn, shape de treino, shape de treino e test.

como no codigo:
```
from plotnine import *

def pick_col(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"Nenhuma dessas colunas existe no df: {cands}")

col_fs    = pick_col(df, "feature_set", "feature_set_")
col_model = pick_col(df, "model_name", "model_name_")
col_tr_ap = pick_col(df, "train__average_precision", "train_average_precision", "train_average_precision".replace("__","_"))
col_te_ap = pick_col(df, "test__average_precision",  "test_average_precision",  "test_average_precision".replace("__","_"))

# --- long format: 1 linha por (feature_set, model, split) ---
plot_df = df[[col_fs, col_model, col_tr_ap, col_te_ap]].copy()
plot_df[col_tr_ap] = pd.to_numeric(plot_df[col_tr_ap], errors="coerce")
plot_df[col_te_ap] = pd.to_numeric(plot_df[col_te_ap], errors="coerce")

long_df = plot_df.melt(
    id_vars=[col_fs, col_model],
    value_vars=[col_tr_ap, col_te_ap],
    var_name="split",
    value_name="avg_precision"
)

# normaliza split -> train/test
long_df["split"] = long_df["split"].map({
    col_tr_ap: "train",
    col_te_ap: "test"
})

# remove NaN
long_df = long_df.dropna(subset=["avg_precision", col_fs, col_model])

# garante ordem do eixo X (opcional)
feature_order = sorted(long_df[col_fs].unique())
long_df[col_fs] = pd.Categorical(long_df[col_fs], categories=feature_order, ordered=True)

p = (
    ggplot(long_df, aes(x=col_fs, y="avg_precision", color=col_model, shape="split"))
    + geom_point(
        size=2.2,
        alpha=0.9,
        position=position_jitter(width=0.18, height=0.0)
    )
    + scale_shape_manual(values={"train": "x", "test": "o"})
    + scale_y_continuous(
        limits=(0, None),
        breaks=np.arange(0, 1.01, 0.1)   # 👈 ticks de 0.1 em 0.1
    )
    + labs(
        x="Feature set",
        y="Average Precision",
        title="Average Precision (Treino vs Teste) por Feature Set — cor = modelo, shape = split"
    )
    + theme(
        figure_size=(16, 6),
        axis_text_x=element_text(rotation=35, ha="right"),
        legend_position="right",
        legend_title=element_text(size=10),
        legend_text=element_text(size=9),
    )
    + guides(
        color=guide_legend(title="Modelo"),
        shape=guide_legend(title="Split")
    )
)
p
```

3 - Aplicado isso para:
a) modelo com todas as features - output/reg_log_l2_all
b) modelo com rtr - output/reg_log_l2_rtr
c) modelo com rts - output/reg_log_l2_rts


e) aplicar o shape values com as forças de cada variável. (para essa parte, usar os dados puros, sem standarscaler.) e salvar um plot em png