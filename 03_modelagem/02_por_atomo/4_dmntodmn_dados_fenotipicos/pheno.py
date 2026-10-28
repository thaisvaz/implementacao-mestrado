# ============================================================
# MODELAGEM COM FEATURES DE CONECTIVIDADE + DADOS PHENOTÍPICOS ABIDE
# ============================================================

import time
import random
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import yaml

from nilearn.datasets import fetch_abide_pcp

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "seed": 19,

    "test_mode": False,
    "test_nrows": 50,

    "base_path": Path("~/_bases").expanduser(),

    "output_root": Path(__file__).resolve().parent / "output_pheno",
}


# ============================================================
# SETUP
# ============================================================
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_nrows(config: dict):
    return config["test_nrows"] if config["test_mode"] else None


# ============================================================
# DADOS PHENOTÍPICOS
# ============================================================

# Colunas clínicas excluídas: ADOS_*, ADI_R_*, SRS_*, SCQ_*, AQ_*
# Essas escalas são aplicadas quase exclusivamente em sujeitos ASD,
# o que causaria vazamento de dados (data leakage) no modelo.
PHENO_CONTINUOUS = [
    "AGE_AT_SCAN",
    "FIQ",
    "VIQ",
    "PIQ",
    "func_mean_fd",
    "func_perc_fd",
    "func_fwhm",
    "func_dvars",
]

PHENO_CATEGORICAL = [
    "SITE_ID",          # local de coleta — one-hot encoding
    "SEX",              # 1=Masculino, 2=Feminino → binário
    "EYE_STATUS_AT_SCAN",  # 1=Abertos, 2=Fechados → binário
    "HANDEDNESS_CATEGORY",  # R/L/Mixed/Ambi → left-handed flag
]


def _encode_sex(series: pd.Series) -> pd.Series:
    # 1=Male→0, 2=Female→1
    return series.map({1: 0, 2: 1}).astype(float)


def _encode_eye_status(series: pd.Series) -> pd.Series:
    # 1=Open→0, 2=Closed→1
    return series.map({1: 0, 2: 1}).astype(float)


def _encode_handedness(series: pd.Series) -> pd.Series:
    # Cria flag binária: 1 se canhoto (L), 0 caso contrário
    # Valores -9999 e NaN são imputados com 0 (direito = maioria)
    left = series.map(lambda v: 1 if v == "L" else (np.nan if pd.isna(v) or v == "-9999" else 0))
    return left.fillna(0).astype(float)


def _encode_site(series: pd.Series) -> pd.DataFrame:
    dummies = pd.get_dummies(series, prefix="SITE", drop_first=True, dtype=float)
    return dummies


def load_phenotypic_features(nrows=None) -> pd.DataFrame:
    """
    Carrega e preprocessa os dados phenotípicos do ABIDE.
    Retorna DataFrame indexado por SUB_ID com features prontas para modelagem.
    """
    print("..ler dados phenotípicos do ABIDE")

    abide = fetch_abide_pcp(
        pipeline="cpac",
        derivatives=["rois_cc200"],
        band_pass_filtering=False,
        global_signal_regression=False,
        quality_checked=True
    )

    pheno = pd.DataFrame(abide.phenotypic)

    if nrows is not None:
        pheno = pheno.iloc[:nrows, :]

    pheno = pheno.set_index("SUB_ID")

    # ---- Variável resposta ----
    y = pheno[["DX_GROUP"]].copy()
    y["DX_GROUP"] = y["DX_GROUP"].map({1: 1, 2: 0})

    print("y.value_counts():")
    print(y["DX_GROUP"].value_counts())

    # ---- Features contínuas ----
    X_cont = pheno[PHENO_CONTINUOUS].copy()

    # Imputa FIQ/VIQ/PIQ com a mediana (calculada no dataset completo, sem separar treino/teste,
    # mas como é um valor global e não supervisionado, o impacto de leakage é mínimo)
    for col in ["FIQ", "VIQ", "PIQ"]:
        median_val = X_cont[col].median()
        n_missing = X_cont[col].isna().sum()
        if n_missing > 0:
            print(f"  Imputando {col}: {n_missing} NaN → mediana={median_val:.1f}")
        X_cont[col] = X_cont[col].fillna(median_val)

    # ---- Features categóricas ----
    sex_enc = _encode_sex(pheno["SEX"]).rename("SEX_female")
    eye_enc = _encode_eye_status(pheno["EYE_STATUS_AT_SCAN"]).rename("EYE_closed")
    hand_enc = _encode_handedness(pheno["HANDEDNESS_CATEGORY"]).rename("HANDEDNESS_left")
    site_enc = _encode_site(pheno["SITE_ID"])

    # ---- Junção ----
    X_pheno = pd.concat([X_cont, sex_enc, eye_enc, hand_enc, site_enc], axis=1)
    X_pheno.index.name = "subid"

    print(f"Features phenotípicas: {X_pheno.shape[1]} colunas")
    print(f"  Contínuas : {len(PHENO_CONTINUOUS)}")
    print(f"  SEX       : 1")
    print(f"  EYE       : 1")
    print(f"  HANDEDNESS: 1")
    print(f"  SITE (OHE): {site_enc.shape[1]}")

    return X_pheno, y


# ============================================================
# LEITURA DE FEATURES DE CONECTIVIDADE
# ============================================================
def load_connectivity_features(features_path: Path, nrows=None) -> pd.DataFrame:
    print(f"..ler features de conectividade: {features_path.name}")
    start = time.time()

    lazy_df = pl.scan_parquet(features_path)
    if nrows is not None:
        lazy_df = lazy_df.limit(nrows)

    df = lazy_df.collect().to_pandas()

    print(df.head(2))
    print("Leitura levou:", round((time.time() - start) / 60, 4), "minutos")

    if "subid" not in df.columns:
        raise ValueError(f"Coluna 'subid' não encontrada em {features_path}")

    df = df.set_index("subid")
    df = df.drop(columns=["target"], errors="ignore")
    return df


def join_features(X_conn: pd.DataFrame, X_pheno: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join entre features de conectividade e features phenotípicas.
    Usa inner join para garantir que apenas sujeitos com ambos os dados sejam mantidos.
    """
    X = X_conn.join(X_pheno, how="inner")
    print(f"Shape após join (conectividade + phenotípicos): {X.shape}")
    diff = len(X_conn) - len(X)
    if diff > 0:
        print(f"  {diff} sujeitos removidos por não terem dados phenotípicos")
    return X


def validate_alignment(X: pd.DataFrame, y: pd.DataFrame) -> None:
    aligned = X.index.equals(y.index)
    print("X e y alinhados?", aligned)
    if not aligned:
        shared = X.index.intersection(y.index)
        print(f"  Sujeitos em comum: {len(shared)}")


def load_subids(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    col = "SUB_ID" if "SUB_ID" in df.columns else "subid"
    return df[col]


def split_train_test(
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_ids: pd.Series,
    test_ids: pd.Series
):
    X_train = X[X.index.isin(train_ids)].copy()
    X_test  = X[X.index.isin(test_ids)].copy()
    y_train = y[y.index.isin(train_ids)].copy()
    y_test  = y[y.index.isin(test_ids)].copy()

    X_train = X_train.sort_index()
    X_test  = X_test.sort_index()
    y_train = y_train.sort_index()
    y_test  = y_test.sort_index()

    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}  y_test: {y_test.shape}")
    print("Train alinhado?", X_train.index.equals(y_train.index))
    print("Test  alinhado?", X_test.index.equals(y_test.index))

    return X_train, X_test, y_train, y_test


# ============================================================
# MÉTRICAS E OUTPUTS
# ============================================================
def calculate_classification_metrics(y_true, y_pred, model_name) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "model_name": model_name,
        "accuracy":  float(acc),
        "recall":    float(rec),
        "precision": float(prec),
        "f1":        float(f1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def print_metrics(metrics: dict, title: str) -> None:
    print(f"""
Métricas – {title}
----------------------
Acurácia : {metrics['accuracy']:.4f}
Recall   : {metrics['recall']:.4f}
Precision: {metrics['precision']:.4f}
F1       : {metrics['f1']:.4f}

Matriz de Confusão
-----------------
TN (Controle → Controle): {metrics['tn']}
FP (Controle → TEA)     : {metrics['fp']}
FN (TEA → Controle)     : {metrics['fn']}
TP (TEA → TEA)          : {metrics['tp']}
""")


def save_pickle(obj, output_path: Path) -> None:
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)


def save_yaml(data, output_path: Path) -> None:
    with open(output_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


def get_estimator(model):
    if isinstance(model, Pipeline):
        return model.named_steps["model"]
    return model


def save_feature_importance(model, model_name: str, feature_names: list, output_dir: Path):
    estimator = get_estimator(model)
    fi_df = None

    if hasattr(estimator, "coef_"):
        coef = np.ravel(estimator.coef_)
        fi_df = pd.DataFrame({
            "feature":        feature_names,
            "importance":     coef,
            "importance_abs": np.abs(coef),
            "model_name":     model_name,
            "importance_type": "coef"
        }).sort_values("importance_abs", ascending=False)

    elif hasattr(estimator, "feature_importance"):
        gain  = estimator.feature_importance(importance_type="gain")
        split = estimator.feature_importance(importance_type="split")
        fi_df = pd.DataFrame({
            "feature":          feature_names,
            "importance_gain":  gain,
            "importance_split": split,
            "model_name":       model_name,
            "importance_type":  "lightgbm"
        }).sort_values("importance_gain", ascending=False)

    if fi_df is not None:
        fname = f"{model_name.lower().replace(' ', '_')}_feature_importance.csv"
        fi_df.to_csv(output_dir / fname, index=False)
        print(f"Feature importance salva: {output_dir / fname}")
        return fi_df

    print(f"Modelo {model_name} não possui feature importance disponível.")
    return None


def save_decision_tree_plot(model, feature_names: list, output_dir: Path, dataset_name: str) -> None:
    estimator = get_estimator(model)
    if not isinstance(estimator, DecisionTreeClassifier):
        return

    fig, ax = plt.subplots(figsize=(40, 16))
    plot_tree(
        estimator,
        feature_names=feature_names,
        class_names=["Controle", "TEA"],
        filled=True, rounded=True, fontsize=8, ax=ax
    )
    ax.set_title(f"Decision Tree — {dataset_name}", fontsize=12)
    out_path = output_dir / "decision_tree.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Árvore salva: {out_path}")


# ============================================================
# MODELOS
# ============================================================
def build_linear_models(seed: int):
    return [
        (
            "SVC L2",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearSVC(penalty="l2", C=0.1, dual=True, tol=0.0001,
                                    max_iter=10000, random_state=seed))
            ]),
            False, True, "svc_l2_model.pkl"
        ),
        (
            "SVC",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="linear", probability=False,
                              class_weight="balanced", random_state=42))
            ]),
            False, True, "svc_model.pkl"
        ),
        (
            "Logistic Regression L2",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(penalty="l2", C=0.1, solver="saga",
                                             max_iter=5000, random_state=42))
            ]),
            True, False, "logreg_l2_model.pkl"
        ),
        (
            "Logistic Regression ElasticNet",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.1,
                                             solver="saga", max_iter=5000, random_state=42))
            ]),
            True, False, "logreg_elasticnet_model.pkl"
        ),
        (
            "Ridge Classifier",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", RidgeClassifier(alpha=10.0))
            ]),
            False, True, "ridge_model.pkl"
        ),
        (
            "SGD Classifier",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SGDClassifier(loss="hinge", alpha=1e-4, max_iter=2000,
                                        tol=1e-3, random_state=42))
            ]),
            False, True, "sgd_model.pkl"
        ),
        (
            "Decision Tree",
            DecisionTreeClassifier(max_depth=5, random_state=seed),
            True, False, "decision_tree_model.pkl"
        ),
    ]


def evaluate_sklearn_model(
    model, model_name, X_train, X_test, y_train, y_test,
    feature_names, output_dir, dataset_name, features_file,
    model_results, all_feature_importances,
    use_proba=False, use_decision_function=True, pickle_name=None
) -> None:
    print(f"..treino {model_name}")
    start_time = time.time()
    model.fit(X_train, y_train)
    exec_time = time.time() - start_time
    print(f"treinar {model_name} levou: {exec_time / 60:.4f} min")

    if use_proba:
        y_score_train = model.predict_proba(X_train)[:, 1]
        y_score_test  = model.predict_proba(X_test)[:, 1]
    elif use_decision_function:
        y_score_train = model.decision_function(X_train)
        y_score_test  = model.decision_function(X_test)
    else:
        raise ValueError(f"Modelo {model_name} sem estratégia de score.")

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    auc_train = roc_auc_score(y_train, y_score_train)
    ap_train  = average_precision_score(y_train, y_score_train)
    auc_test  = roc_auc_score(y_test, y_score_test)
    ap_test   = average_precision_score(y_test, y_score_test)

    print(f"AP train={ap_train:.4f}  AUC train={auc_train:.4f}")
    print(f"AP test ={ap_test:.4f}  AUC test ={auc_test:.4f}")

    metrics_train = calculate_classification_metrics(y_train, y_pred_train, f"{model_name}_train")
    metrics_test  = calculate_classification_metrics(y_test, y_pred_test,  f"{model_name}_test")
    print_metrics(metrics_train, f"{model_name}_train")
    print_metrics(metrics_test,  f"{model_name}_test")

    model_results.append({
        "dataset":           dataset_name,
        "features_file":     features_file,
        "model_name":        model_name,
        "exec_time_minutes": float(exec_time / 60),
        "train": {"auc": float(auc_train), "average_precision": float(ap_train), **metrics_train},
        "test":  {"auc": float(auc_test),  "average_precision": float(ap_test),  **metrics_test},
    })

    if pickle_name is not None:
        save_pickle(model, output_dir / pickle_name)

    fi_df = save_feature_importance(model, model_name, feature_names, output_dir)
    if fi_df is not None:
        all_feature_importances.append(fi_df)

    save_decision_tree_plot(model, feature_names, output_dir, dataset_name)


def train_lightgbm(
    X_train, X_test, y_train, y_test,
    feature_names, output_dir, dataset_name, features_file,
    model_results, all_feature_importances, seed
) -> None:
    print("..treino LightGBM")
    start_time = time.time()

    lgb_train = lgb.Dataset(X_train, label=y_train)

    params = {
        "objective":       "binary",
        "metric":          "average_precision",
        "boosting_type":   "gbdt",
        "num_leaves":      10,
        "learning_rate":   0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq":    1,
        "force_col_wise":  True,
        "seed":            seed,
        "verbose":         -1,
    }

    gbm = lgb.train(
        params, lgb_train, num_boost_round=100,
    )

    exec_time = time.time() - start_time
    print(f"treinar LightGBM levou: {exec_time / 60:.4f} min")

    y_score_train = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    y_score_test  = gbm.predict(X_test,  num_iteration=gbm.best_iteration)
    y_pred_train  = (y_score_train >= 0.5).astype(int)
    y_pred_test   = (y_score_test  >= 0.5).astype(int)

    auc_train = roc_auc_score(y_train, y_score_train)
    ap_train  = average_precision_score(y_train, y_score_train)
    auc_test  = roc_auc_score(y_test, y_score_test)
    ap_test   = average_precision_score(y_test, y_score_test)

    print(f"LightGBM AP train={ap_train:.4f}  AUC train={auc_train:.4f}")
    print(f"LightGBM AP test ={ap_test:.4f}  AUC test ={auc_test:.4f}")

    metrics_train = calculate_classification_metrics(y_train, y_pred_train, "lgbm_train")
    metrics_test  = calculate_classification_metrics(y_test,  y_pred_test,  "lgbm_test")
    print_metrics(metrics_train, "lgbm_train")
    print_metrics(metrics_test,  "lgbm_test")

    model_results.append({
        "dataset":           dataset_name,
        "features_file":     features_file,
        "model_name":        "LightGBM",
        "exec_time_minutes": float(exec_time / 60),
        "train": {"auc": float(auc_train), "average_precision": float(ap_train), **metrics_train},
        "test":  {"auc": float(auc_test),  "average_precision": float(ap_test),  **metrics_test},
    })

    save_pickle(gbm, output_dir / "lightgbm_model.pkl")

    fi_df = save_feature_importance(gbm, "LightGBM", feature_names, output_dir)
    if fi_df is not None:
        all_feature_importances.append(fi_df)


# ============================================================
# MULTI-BASES
# ============================================================
def list_dmn_feature_files(base_path: Path) -> list:
    existentes = sorted(p.name for p in base_path.glob("*dmntodmn.parquet"))

    print("Bases DMN→DMN encontradas:")
    for f in existentes:
        print(" -", f)

    if not existentes:
        raise FileNotFoundError(f"Nenhuma base *dmntodmn.parquet em {base_path}")

    return existentes


def run_one_dataset(
    features_file: str,
    X_pheno: pd.DataFrame,
    y: pd.DataFrame,
    train_ids: pd.Series,
    test_ids: pd.Series,
    config: dict
) -> dict:
    base_path   = config["base_path"]
    output_root = config["output_root"]
    nrows       = get_nrows(config)

    dataset_name = Path(features_file).stem + "_pheno"
    output_dir   = output_root / dataset_name
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n======================================")
    print(f"Rodando: {dataset_name}")
    print("======================================")

    features_path = base_path / features_file
    X_conn = load_connectivity_features(features_path, nrows=nrows)

    # Junta conectividade + phenotípicos
    X = join_features(X_conn, X_pheno)

    # Alinha y com X (inner join pode reduzir sujeitos)
    y_aligned = y.loc[y.index.isin(X.index)].copy()
    X = X.loc[X.index.isin(y_aligned.index)].copy()
    X = X.sort_index()
    y_aligned = y_aligned.sort_index()

    validate_alignment(X, y_aligned)

    X_train, X_test, y_train_df, y_test_df = split_train_test(
        X, y_aligned, train_ids, test_ids
    )

    y_train = y_train_df["DX_GROUP"].astype(int).values
    y_test  = y_test_df["DX_GROUP"].astype(int).values

    feature_names = X_train.columns.tolist()
    print(f"Total de features (conectividade + phenotípicos): {len(feature_names)}")

    model_results          = []
    all_feature_importances = []

    for model_name, model, use_proba, use_df, pickle_name in build_linear_models(config["seed"]):
        evaluate_sklearn_model(
            model=model, model_name=model_name,
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            feature_names=feature_names,
            output_dir=output_dir,
            dataset_name=dataset_name,
            features_file=features_file,
            model_results=model_results,
            all_feature_importances=all_feature_importances,
            use_proba=use_proba,
            use_decision_function=use_df,
            pickle_name=pickle_name
        )

    train_lightgbm(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        output_dir=output_dir,
        dataset_name=dataset_name,
        features_file=features_file,
        model_results=model_results,
        all_feature_importances=all_feature_importances,
        seed=config["seed"]
    )

    save_yaml(model_results, output_dir / "model_metrics.yaml")
    print("Métricas salvas:", output_dir / "model_metrics.yaml")

    if all_feature_importances:
        fi_concat = pd.concat(all_feature_importances, axis=0, ignore_index=True)
        fi_concat.to_csv(output_dir / "all_models_feature_importance.csv", index=False)
        print("Feature importance consolidada salva:", output_dir / "all_models_feature_importance.csv")

    best_ap  = max(m["test"]["average_precision"] for m in model_results)
    best_auc = max(m["test"]["auc"] for m in model_results)

    print(f"Melhor test AP : {best_ap:.4f}")
    print(f"Melhor test AUC: {best_auc:.4f}")

    return {
        "dataset":                     dataset_name,
        "features_file":               features_file,
        "n_conn_features":             X_conn.shape[1],
        "n_pheno_features":            X_pheno.shape[1],
        "n_total_features":            len(feature_names),
        "n_train":                     len(y_train),
        "n_test":                      len(y_test),
        "best_test_average_precision": float(best_ap),
        "best_test_auc":               float(best_auc),
        "output_dir":                  str(output_dir),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    set_random_seed(CONFIG["seed"])

    output_root = CONFIG["output_root"]
    output_root.mkdir(exist_ok=True, parents=True)

    print("======================================")
    print("CONFIGURAÇÃO")
    print("test_mode :", CONFIG["test_mode"])
    print("base_path :", CONFIG["base_path"])
    print("output_root:", output_root)
    print("======================================")

    nrows = get_nrows(CONFIG)

    # Carrega phenotípicos e target uma única vez
    X_pheno, y = load_phenotypic_features(nrows=nrows)

    train_ids = load_subids(CONFIG["base_path"] / "train_subids.parquet")
    test_ids  = load_subids(CONFIG["base_path"] / "test_subids.parquet")

    feature_files = list_dmn_feature_files(CONFIG["base_path"])

    run_summaries = []
    total_start   = time.time()

    for features_file in feature_files:
        summary = run_one_dataset(
            features_file=features_file,
            X_pheno=X_pheno,
            y=y,
            train_ids=train_ids,
            test_ids=test_ids,
            config=CONFIG
        )
        run_summaries.append(summary)

    summary_df = pd.DataFrame(run_summaries)
    summary_df.to_csv(output_root / "run_summary.csv", index=False)
    print("\nResumo salvo:", output_root / "run_summary.csv")
    print(summary_df.to_string())

    print(f"\nTempo total: {(time.time() - total_start) / 60:.2f} minutos")
    print("Pipeline com dados phenotípicos finalizado com sucesso")


if __name__ == "__main__":
    main()
