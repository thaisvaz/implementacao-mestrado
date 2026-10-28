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

# Machine Learning
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

from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.svm import LinearSVC

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Utilitários
from tqdm import tqdm
import pickle
import yaml

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
    pipeline='cpac', 
    derivatives=['rois_cc200'], 
    band_pass_filtering=False, 
    global_signal_regression=False, 
    quality_checked=True
)
abide_pheno = pd.DataFrame(abide.phenotypic)

print("abide_pheno", abide_pheno.head())

y = abide_pheno[['SUB_ID','DX_GROUP']].copy()
# 1=TEA 2=controle
y['DX_GROUP'] = y['DX_GROUP'].map({1:1, 2:0})
# rodar so para 50 linhas
# y= y.iloc[0:50,:]


## pequeno EDA
print("y.valuecounts()", y['DX_GROUP'].value_counts())
# y.to_csv("df_exemplo_target.csv", index=False)

# ============================================================
# LEITURA DATASET resultados com a teoria da complexidade
# ============================================================

print("..ler features de complexidade")
start = time.time()
df = pd.read_csv("~/_bases/resultados.csv")
# rodar so para 50 linhas
# df = pd.read_csv("~/_bases/resultados.csv", nrows=50)
print(df.head())
print("Ler a base levou: ", (time.time() - start)/60, "minutos!")

y.set_index('SUB_ID', inplace=True)
df.set_index('subid', inplace=True)
indices_iguais = df.index.equals(y.index)
print("X e y estão perfeitamente alinhados?", indices_iguais)

## SPLIT TRAIN TEST
X_train, X_test, y_train, y_test = train_test_split(df, y['DX_GROUP'], test_size=0.1, stratify=y['DX_GROUP'], random_state=19)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# FUNCOES DE AVALIACAO DE MODELOS
# ============================================================

def calcula_metricas(y_true, y_pred, model_name="modelo"):
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

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
        "tp": int(tp)
    }


def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    print(f"..treino modelo {model_name}")
    start_time = time.time()

    # criar pasta _output se não existir
    os.makedirs("_output", exist_ok=True)

    # garantir reprodutibilidade
    if hasattr(model, 'random_state'):
        model.random_state = 19

    model.fit(X_train, y_train)

    # salvar pickle imediatamente após o treino
    model_filename = f"_output/{model_name.lower().replace(' ', '_')}_{int(time.time())}_model.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modelo salvo em {model_filename}")

    exec_time = time.time() - start_time
    print(f"treinar o {model_name} levou: {exec_time/60:.2f} minutos!")

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

    # avaliar resultados train
    auc_train = roc_auc_score(y_train, y_score_train)
    ap_train = average_precision_score(y_train, y_score_train)
    metrics_train = calcula_metricas(y_train, y_pred_train, f"{model_name.lower().replace(' ', '_')}_train")
    
    # avaliar resultados test
    auc_test = roc_auc_score(y_test, y_score_test)
    ap_test = average_precision_score(y_test, y_score_test)
    metrics_test = calcula_metricas(y_test, y_pred_test, f"{model_name.lower().replace(' ', '_')}_test")
    
    print(f"""
Métricas Treino – {metrics_train['model_name']}
----------------------
Average Precision train: {ap_train:.4f}
AUC train: {auc_train:.4f}
Acurácia : {metrics_train['accuracy']:.4f}
Recall   : {metrics_train['recall']:.4f}
Precision: {metrics_train['precision']:.4f}
F1       : {metrics_train['f1']:.4f}

Matriz de Confusão
-----------------
TN (Controle → Controle): {metrics_train['tn']}
FP (Controle → TEA)     : {metrics_train['fp']}
FN (TEA → Controle)    : {metrics_train['fn']}
TP (TEA → TEA)         : {metrics_train['tp']}
""")

    print(f"""
Métricas Teste – {metrics_test['model_name']}
----------------------
Average Precision test: {ap_test:.4f}
AUC test: {auc_test:.4f}                    
Acurácia : {metrics_test['accuracy']:.4f}
Recall   : {metrics_test['recall']:.4f}
Precision: {metrics_test['precision']:.4f}
F1       : {metrics_test['f1']:.4f}

Matriz de Confusão
-----------------
TN (Controle → Controle): {metrics_test['tn']}
FP (Controle → TEA)     : {metrics_test['fp']}          
FN (TEA → Controle)    : {metrics_test['fn']}
TP (TEA → TEA)         : {metrics_test['tp']}
""")

    model_results = [{
        "model_name": model_name,
        "exec_time_minutes": exec_time / 60,
        "train": {
            "auc": auc_train,
            "average_precision": ap_train,
            **metrics_train
        },
        "test": {
            "auc": auc_test,
            "average_precision": ap_test,
            **metrics_test
        }
    }]

    # salvar bases scoradas (subid + y_true + score) em parquet
    timestamp = 11012026
    # timestamp = int(time.time())
    score_col = f"{model_name.lower().replace(' ', '_')}_score"
    
    train_scorado = pd.DataFrame({
        'subid': X_train.index,
        'y_true': y_train,
        score_col: y_score_train
    })
    train_scorado_filename = f"_output/{model_name.lower().replace(' ', '_')}_train_scorado_{timestamp}.parquet"
    train_scorado.to_parquet(train_scorado_filename)
    print(f"Base de treino scorada salva em {train_scorado_filename}")

    test_scorado = pd.DataFrame({
        'subid': X_test.index,
        'y_true': y_test,
        score_col: y_score_test
    })
    test_scorado_filename = f"_output/{model_name.lower().replace(' ', '_')}_test_scorado_{timestamp}.parquet"
    test_scorado.to_parquet(test_scorado_filename)
    print(f"Base de teste scorada salva em {test_scorado_filename}")

    # salvar métricas em YAML
    yaml_filename = f"_output/{model_name.lower().replace(' ', '_')}_metrics_{timestamp}.yaml"
    with open(yaml_filename, 'w') as f:
        yaml.dump(model_results, f)
    print(f"Métricas salvas em {yaml_filename}")


# Lista de modelos para executar em paralelo
models = [
    ("LinearSVC L2", LinearSVC(penalty='l2', C=0.1, dual=True, tol=0.0001, max_iter=10000, random_state=19)),
    ("Ridge Classifier", RidgeClassifier(alpha=10.0)),
    ("Logistic Regression L2", LogisticRegression(penalty="l2", C=0.1, solver="saga", max_iter=5000, random_state=19)),
    ("Logistic Regression ElasticNet", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.1, solver="saga", max_iter=5000, random_state=19)),
    ("SGD Classifier", SGDClassifier(loss="hinge", alpha=1e-4, max_iter=2000, tol=1e-3, random_state=19)),
    ("LightGBM", lgb.LGBMClassifier(objective='binary', metric='auc', boosting_type='gbdt', num_leaves=10, learning_rate=0.05, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, random_state=19))
]

# Executar modelos em paralelo
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = [executor.submit(train_and_evaluate_model, model, name, X_train, X_test, y_train, y_test) for name, model in models]
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()  # Aguarda conclusão e captura exceções
        except Exception as e:
            print(f"Erro em modelo: {e}")

print("Todos os modelos foram treinados e avaliados.")