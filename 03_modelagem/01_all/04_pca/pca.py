# ============================================================
# MODELAGEM COM PCA — resultados_no_diagonal
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

from sklearn.decomposition import PCA
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

    "features_path": Path("~/_bases/resultados_no_diagonal.parquet").expanduser(),
    "base_path": Path("~/_bases").expanduser(),

    "output_root": Path(__file__).resolve().parent / "output_pca",

    # variância a ser explicada pelo PCA (também testa valores fixos)
    "pca_variance_thresholds": [0.95, 0.99],
    "pca_fixed_components": [50, 100, 200],
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
# LEITURA DOS DADOS
# ============================================================
def load_target_abide(nrows=None) -> pd.DataFrame:
    print("..ler var resposta")
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

    if nrows is not None:
        y = y.iloc[:nrows, :]

    print("y.value_counts():")
    print(y["DX_GROUP"].value_counts())
    y = y.set_index("SUB_ID")
    return y


def load_features(features_path: Path, nrows=None) -> pd.DataFrame:
    print(f"..ler features: {features_path.name}")
    start = time.time()

    lazy_df = pl.scan_parquet(features_path)
    if nrows is not None:
        lazy_df = lazy_df.limit(nrows)

    df = lazy_df.collect().to_pandas()
    print(f"Shape lido: {df.shape}")
    print("Ler a base levou:", round((time.time() - start) / 60, 4), "minutos!")

    if "subid" not in df.columns:
        raise ValueError(f"Coluna 'subid' não encontrada em {features_path}")

    df = df.set_index("subid")
    df = df.drop(columns=["target"], errors="ignore")
    return df


def load_subids(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    col = "SUB_ID" if "SUB_ID" in df.columns else "subid"
    return df[col]


def split_train_test(X, y, train_ids, test_ids):
    X_train = X[X.index.isin(train_ids)].sort_index()
    X_test  = X[X.index.isin(test_ids)].sort_index()
    y_train = y[y.index.isin(train_ids)].sort_index()
    y_test  = y[y.index.isin(test_ids)].sort_index()

    print("X_train:", X_train.shape, " | X_test:", X_test.shape)
    print("Train alinhado?", X_train.index.equals(y_train.index))
    print("Test  alinhado?", X_test.index.equals(y_test.index))
    return X_train, X_test, y_train, y_test


# ============================================================
# PCA
# ============================================================
def apply_pca(X_train: pd.DataFrame, X_test: pd.DataFrame, n_components, seed: int):
    """
    Fit PCA on train, transform train and test.
    n_components pode ser int (fixo) ou float 0<n<1 (variância explicada).
    """
    print(f"\n..PCA n_components={n_components}")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca  = pca.transform(X_test_sc)

    n_comp_actual = pca.n_components_
    var_explained = pca.explained_variance_ratio_.sum()

    print(f"  Componentes retidos  : {n_comp_actual}")
    print(f"  Variância explicada  : {var_explained:.4f}")
    print(f"  X_train_pca shape   : {X_train_pca.shape}")
    print(f"  X_test_pca  shape   : {X_test_pca.shape}")

    feature_names = [f"PC{i+1}" for i in range(n_comp_actual)]

    return (
        pd.DataFrame(X_train_pca, index=X_train.index, columns=feature_names),
        pd.DataFrame(X_test_pca,  index=X_test.index,  columns=feature_names),
        pca,
        scaler,
        n_comp_actual,
        float(var_explained),
    )


def save_pca_variance_plot(pca: PCA, output_dir: Path, label: str) -> None:
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, marker=".", markersize=3)
    ax.axhline(0.95, color="r", linestyle="--", label="95%")
    ax.axhline(0.99, color="g", linestyle="--", label="99%")
    ax.set_xlabel("Número de componentes")
    ax.set_ylabel("Variância acumulada")
    ax.set_title(f"PCA — variância acumulada [{label}]")
    ax.legend()
    out_path = output_dir / f"pca_variance_{label}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico de variância salvo em: {out_path}")


# ============================================================
# MÉTRICAS
# ============================================================
def calculate_classification_metrics(y_true, y_pred, model_name) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "model_name": model_name,
        "accuracy":   float(accuracy_score(y_true, y_pred)),
        "recall":     float(recall_score(y_true, y_pred, zero_division=0)),
        "precision":  float(precision_score(y_true, y_pred, zero_division=0)),
        "f1":         float(f1_score(y_true, y_pred, zero_division=0)),
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
Confusão — TN:{metrics['tn']} FP:{metrics['fp']} FN:{metrics['fn']} TP:{metrics['tp']}
""")


def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_yaml(data, path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


def get_estimator(model):
    if isinstance(model, Pipeline):
        return model.named_steps["model"]
    return model


def save_feature_importance(model, model_name, feature_names, output_dir):
    estimator = get_estimator(model)
    fi_df = None

    if hasattr(estimator, "coef_"):
        coef = np.ravel(estimator.coef_)
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": coef,
            "importance_abs": np.abs(coef),
            "model_name": model_name,
            "importance_type": "coef"
        }).sort_values("importance_abs", ascending=False)

    elif hasattr(estimator, "feature_importance"):
        gain  = estimator.feature_importance(importance_type="gain")
        split = estimator.feature_importance(importance_type="split")
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance_gain": gain,
            "importance_split": split,
            "model_name": model_name,
            "importance_type": "lightgbm"
        }).sort_values("importance_gain", ascending=False)

    if fi_df is not None:
        fname = f"{model_name.lower().replace(' ', '_')}_feature_importance.csv"
        fi_df.to_csv(output_dir / fname, index=False)
        print(f"Feature importance salva: {output_dir / fname}")
        return fi_df

    print(f"Modelo {model_name} sem feature importance disponível.")
    return None


# ============================================================
# MODELOS
# ============================================================
def build_linear_models(seed: int):
    return [
        (
            "SVC L2",
            Pipeline([
                ("model", LinearSVC(penalty="l2", C=0.1, dual=True, tol=1e-4,
                                    max_iter=10000, random_state=seed))
            ]),
            False, True, "svc_l2_model.pkl"
        ),
        (
            "SVC",
            Pipeline([
                ("model", SVC(kernel="linear", probability=False,
                              class_weight="balanced", random_state=42))
            ]),
            False, True, "svc_model.pkl"
        ),
        (
            "Logistic Regression L2",
            Pipeline([
                ("model", LogisticRegression(penalty="l2", C=0.1, solver="saga",
                                             max_iter=5000, random_state=42))
            ]),
            True, False, "logreg_l2_model.pkl"
        ),
        (
            "Logistic Regression ElasticNet",
            Pipeline([
                ("model", LogisticRegression(penalty="elasticnet", l1_ratio=0.5,
                                             C=0.1, solver="saga", max_iter=5000,
                                             random_state=42))
            ]),
            True, False, "logreg_elasticnet_model.pkl"
        ),
        (
            "Ridge Classifier",
            Pipeline([("model", RidgeClassifier(alpha=10.0))]),
            False, True, "ridge_model.pkl"
        ),
        (
            "SGD Classifier",
            Pipeline([
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
    feature_names, output_dir, pca_label, model_results,
    all_feature_importances, use_proba, use_decision_function, pickle_name
):
    print(f"..treino {model_name}")
    start = time.time()
    model.fit(X_train, y_train)
    exec_time = time.time() - start

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
    print(f"AP test ={ap_test:.4f}   AUC test ={auc_test:.4f}")

    metrics_train = calculate_classification_metrics(y_train, y_pred_train, f"{model_name}_train")
    metrics_test  = calculate_classification_metrics(y_test,  y_pred_test,  f"{model_name}_test")
    print_metrics(metrics_train, f"{model_name}_train")
    print_metrics(metrics_test,  f"{model_name}_test")

    model_results.append({
        "pca_label": pca_label,
        "model_name": model_name,
        "exec_time_minutes": float(exec_time / 60),
        "train": {"auc": float(auc_train), "average_precision": float(ap_train), **metrics_train},
        "test":  {"auc": float(auc_test),  "average_precision": float(ap_test),  **metrics_test},
    })

    if pickle_name:
        save_pickle(model, output_dir / pickle_name)

    fi_df = save_feature_importance(model, model_name, feature_names, output_dir)
    if fi_df is not None:
        all_feature_importances.append(fi_df)


def train_lightgbm(
    X_train, X_test, y_train, y_test, feature_names,
    output_dir, pca_label, model_results, all_feature_importances, seed
):
    print("..treino LightGBM")
    start = time.time()

    lgb_train = lgb.Dataset(X_train, label=y_train)

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "num_leaves": 10,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "force_col_wise": True,
        "seed": seed,
        "verbose": -1,
    }

    gbm = lgb.train(
        params, lgb_train, num_boost_round=100,
    )
    exec_time = time.time() - start

    y_score_train = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    y_score_test  = gbm.predict(X_test,  num_iteration=gbm.best_iteration)
    y_pred_train  = (y_score_train >= 0.5).astype(int)
    y_pred_test   = (y_score_test  >= 0.5).astype(int)

    auc_train = roc_auc_score(y_train, y_score_train)
    ap_train  = average_precision_score(y_train, y_score_train)
    auc_test  = roc_auc_score(y_test, y_score_test)
    ap_test   = average_precision_score(y_test, y_score_test)

    print(f"LGBM AP train={ap_train:.4f}  AUC train={auc_train:.4f}")
    print(f"LGBM AP test ={ap_test:.4f}   AUC test ={auc_test:.4f}")

    metrics_train = calculate_classification_metrics(y_train, y_pred_train, "lgbm_train")
    metrics_test  = calculate_classification_metrics(y_test,  y_pred_test,  "lgbm_test")
    print_metrics(metrics_train, "lgbm_train")
    print_metrics(metrics_test,  "lgbm_test")

    model_results.append({
        "pca_label": pca_label,
        "model_name": "LightGBM",
        "exec_time_minutes": float(exec_time / 60),
        "train": {"auc": float(auc_train), "average_precision": float(ap_train), **metrics_train},
        "test":  {"auc": float(auc_test),  "average_precision": float(ap_test),  **metrics_test},
    })

    save_pickle(gbm, output_dir / "lightgbm_model.pkl")
    fi_df = save_feature_importance(gbm, "LightGBM", feature_names, output_dir)
    if fi_df is not None:
        all_feature_importances.append(fi_df)


# ============================================================
# RUN UMA CONFIGURAÇÃO DE PCA
# ============================================================
def run_one_pca(
    n_components,
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_root: Path,
    config: dict,
    all_summaries: list,
):
    label = str(n_components).replace(".", "")
    output_dir = output_root / f"pca_{label}"
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*60}")
    print(f"PCA n_components = {n_components}  →  {label}")
    print(f"{'='*60}")

    X_train_pca, X_test_pca, pca, scaler, n_comp_actual, var_explained = apply_pca(
        X_train_raw, X_test_raw, n_components, config["seed"]
    )

    save_pca_variance_plot(pca, output_dir, label)
    save_pickle(pca,    output_dir / "pca.pkl")
    save_pickle(scaler, output_dir / "scaler.pkl")

    feature_names = X_train_pca.columns.tolist()
    model_results = []
    all_feature_importances = []

    for model_name, model, use_proba, use_df, pkl_name in build_linear_models(config["seed"]):
        evaluate_sklearn_model(
            model=model,
            model_name=model_name,
            X_train=X_train_pca,
            X_test=X_test_pca,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            output_dir=output_dir,
            pca_label=label,
            model_results=model_results,
            all_feature_importances=all_feature_importances,
            use_proba=use_proba,
            use_decision_function=use_df,
            pickle_name=pkl_name,
        )

    train_lightgbm(
        X_train=X_train_pca, X_test=X_test_pca,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        output_dir=output_dir,
        pca_label=label,
        model_results=model_results,
        all_feature_importances=all_feature_importances,
        seed=config["seed"],
    )

    save_yaml(model_results, output_dir / "model_metrics.yaml")

    if all_feature_importances:
        fi_concat = pd.concat(all_feature_importances, axis=0, ignore_index=True)
        fi_concat.to_csv(output_dir / "all_models_feature_importance.csv", index=False)

    best_ap  = max(m["test"]["average_precision"] for m in model_results)
    best_auc = max(m["test"]["auc"] for m in model_results)
    best_model = max(model_results, key=lambda m: m["test"]["average_precision"])["model_name"]

    print(f"\n[PCA {label}] Melhor test AP : {best_ap:.4f}  ({best_model})")
    print(f"[PCA {label}] Melhor test AUC: {best_auc:.4f}")

    all_summaries.append({
        "pca_label": label,
        "n_components_requested": str(n_components),
        "n_components_actual": n_comp_actual,
        "variance_explained": var_explained,
        "n_train": len(y_train),
        "n_test":  len(y_test),
        "best_test_average_precision": float(best_ap),
        "best_test_auc": float(best_auc),
        "best_model": best_model,
        "output_dir": str(output_dir),
    })


# ============================================================
# MAIN
# ============================================================
def main():
    set_random_seed(CONFIG["seed"])

    output_root = CONFIG["output_root"]
    output_root.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("PIPELINE PCA — resultados_no_diagonal")
    print("test_mode:", CONFIG["test_mode"])
    print("features :", CONFIG["features_path"])
    print("output   :", output_root)
    print("=" * 60)

    nrows = get_nrows(CONFIG)

    y        = load_target_abide(nrows=nrows)
    train_ids = load_subids(CONFIG["base_path"] / "train_subids.parquet")
    test_ids  = load_subids(CONFIG["base_path"] / "test_subids.parquet")

    X = load_features(CONFIG["features_path"], nrows=nrows)

    X_train_raw, X_test_raw, y_train_df, y_test_df = split_train_test(X, y, train_ids, test_ids)
    y_train = y_train_df["DX_GROUP"].astype(int).values
    y_test  = y_test_df["DX_GROUP"].astype(int).values

    all_summaries = []
    total_start = time.time()

    all_n_components = CONFIG["pca_variance_thresholds"] + CONFIG["pca_fixed_components"]

    for n_components in all_n_components:
        run_one_pca(
            n_components=n_components,
            X_train_raw=X_train_raw,
            X_test_raw=X_test_raw,
            y_train=y_train,
            y_test=y_test,
            output_root=output_root,
            config=CONFIG,
            all_summaries=all_summaries,
        )

    summary_df = pd.DataFrame(all_summaries)
    summary_path = output_root / "pca_run_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("RESUMO FINAL")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\nResumo salvo em {summary_path}")
    print(f"Tempo total: {(time.time() - total_start) / 60:.2f} minutos!")
    print("Pipeline PCA finalizado com sucesso.")


if __name__ == "__main__":
    main()
