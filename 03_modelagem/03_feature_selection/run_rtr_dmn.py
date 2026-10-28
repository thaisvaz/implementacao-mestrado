"""
run_rtr_dmn.py
==============
Pipeline completo: split → feature selection só no treino → modelo → avaliação.

871 amostras × 3429 features → seleciona até 200 via torneio eliminatório.
"""

import sys
import time
from pathlib import Path

import polars as pl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selection import FeatureSelector, TreeConfig, TournamentConfig

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_PATH  = Path("~/_bases").expanduser()
PARQUET    = BASE_PATH / "rtr_dmn.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "output_rtr_dmn"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Carrega dados e split train/test pelos subids oficiais
# ---------------------------------------------------------------------------
print("Carregando parquet...")
t0 = time.time()
df = pl.scan_parquet(PARQUET).collect().to_pandas().set_index("subid")
print(f"  shape: {df.shape}  ({time.time()-t0:.1f}s)")

train_ids = set(pl.read_parquet(BASE_PATH / "train_subids.parquet")["SUB_ID"].to_list())
test_ids  = set(pl.read_parquet(BASE_PATH / "test_subids.parquet")["SUB_ID"].to_list())

feature_cols = [c for c in df.columns if c != "target"]

df_train = df[df.index.isin(train_ids)].sort_index()
df_test  = df[df.index.isin(test_ids)].sort_index()

X_train = df_train[feature_cols]
y_train = df_train["target"].astype(int).values
X_test  = df_test[feature_cols]
y_test  = df_test["target"].astype(int).values

print(f"  train: {X_train.shape}  test: {X_test.shape}")
print(f"  target train — 0:{(y_train==0).sum()}  1:{(y_train==1).sum()}")
print(f"  target test  — 0:{(y_test==0).sum()}  1:{(y_test==1).sum()}")

# ---------------------------------------------------------------------------
# Feature selection — FIT APENAS NO TREINO
# ---------------------------------------------------------------------------
tree_cfg = TreeConfig(
    n_estimators=50,
    max_depth=4,
    min_samples_leaf=15,        # ~2% de 783 amostras de treino
    min_samples_split=30,
    max_features="sqrt",
    n_estimators_final=100,
    task="classification",
    random_state=42,
)

tour_cfg = TournamentConfig(
    chunk_size=800,
    top_k_pct=0.08,
    final_top_n=20,
    n_tournament_seeds=3,
    seed_consensus="majority",
    variance_threshold=None,    # escala e-05, threshold quebraria
    verbose=True,
)

selector = FeatureSelector(target_col="target", tree_cfg=tree_cfg, tour_cfg=tour_cfg)

# monta df_train com target para o fit
df_train_with_target = X_train.copy()
df_train_with_target["target"] = y_train

print("\nIniciando feature selection (somente treino)...")
t0 = time.time()
result = selector.fit(df_train_with_target, feature_cols=feature_cols)
print(f"Feature selection concluída em {time.time()-t0:.1f}s")
print(f"Features selecionadas: {len(result.selected_features)}")

# Salva ranking
summary = result.summary()
summary.to_csv(OUTPUT_DIR / "selected_features.csv", index=False)
print(f"Ranking salvo em: {OUTPUT_DIR / 'selected_features.csv'}")
print("\nTop 20:")
print(summary.head(20).to_string(index=False))

# Aplica seleção nos dois conjuntos
X_train_sel = X_train[result.selected_features].values
X_test_sel  = X_test[result.selected_features].values

# ---------------------------------------------------------------------------
# Modelo: Logistic Regression L2
# ---------------------------------------------------------------------------
print("\nTreinando Logistic Regression...")
t0 = time.time()
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty="l2", C=0.01, solver="lbfgs", max_iter=1000, random_state=42)),
])
model.fit(X_train_sel, y_train)
print(f"  treino: {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------
def calc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_score),
        "ap":  average_precision_score(y_true, y_score),
        "acc": accuracy_score(y_true, y_pred),
    }

score_train = model.predict_proba(X_train_sel)[:, 1]
score_test  = model.predict_proba(X_test_sel)[:, 1]

m_train = calc_metrics(y_train, score_train)
m_test  = calc_metrics(y_test,  score_test)

for split, m in [("TREINO", m_train), ("TESTE", m_test)]:
    print(f"\n=== {split} ===")
    print(f"  AUC              : {m['auc']:.4f}")
    print(f"  Average Precision: {m['ap']:.4f}")
    print(f"  Accuracy         : {m['acc']:.4f}")

# ---------------------------------------------------------------------------
# Gráfico
# ---------------------------------------------------------------------------
metrics   = ["AUC", "Average Precision"]
v_train   = [m_train["auc"], m_train["ap"]]
v_test    = [m_test["auc"],  m_test["ap"]]

x     = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
bars_tr = ax.bar(x - width/2, v_train, width, label="Treino", color="#4C72B0")
bars_te = ax.bar(x + width/2, v_test,  width, label="Teste",  color="#DD8452")

ax.set_ylim(0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylabel("Score", fontsize=11)
ax.legend(fontsize=11)
ax.set_title(
    f"Logistic Regression — {len(feature_cols)} features → {len(result.selected_features)} selecionadas",
    fontsize=11, pad=12,
)

for bar in list(bars_tr) + list(bars_te):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{bar.get_height():.3f}",
        ha="center", va="bottom", fontsize=10,
    )

fig.tight_layout()
plot_path = OUTPUT_DIR / "metrics_bar.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"\nGráfico salvo em: {plot_path}")

# ---------------------------------------------------------------------------
# Feature importance (top 30)
# ---------------------------------------------------------------------------
TOP_N = 30
fi = summary.head(TOP_N).iloc[::-1]  # inverte para barra horizontal crescente

fig2, ax2 = plt.subplots(figsize=(9, TOP_N * 0.35 + 1.5))
ax2.barh(fi["feature"], fi["importance"], color="#4C72B0")
ax2.set_xlabel("Importância (ExtraTrees)", fontsize=11)
ax2.set_title(
    f"Top {TOP_N} features — {len(feature_cols)} originais → {len(result.selected_features)} selecionadas",
    fontsize=11, pad=12,
)
fig2.tight_layout()
fi_path = OUTPUT_DIR / "feature_importance.png"
fig2.savefig(fi_path, dpi=150)
plt.close(fig2)
print(f"Feature importance salvo em: {fi_path}")

# ---------------------------------------------------------------------------
# Modelo 2: Decision Tree
# ---------------------------------------------------------------------------
print("\nTreinando Decision Tree...")
t0 = time.time()
dt = DecisionTreeClassifier(max_depth=2, min_samples_leaf=50, min_samples_split=100, random_state=42)
dt.fit(X_train_sel, y_train)
print(f"  treino: {time.time()-t0:.1f}s")

score_train_dt = dt.predict_proba(X_train_sel)[:, 1]
score_test_dt  = dt.predict_proba(X_test_sel)[:, 1]

m_train_dt = calc_metrics(y_train, score_train_dt)
m_test_dt  = calc_metrics(y_test,  score_test_dt)

for split, m in [("TREINO", m_train_dt), ("TESTE", m_test_dt)]:
    print(f"\n=== Decision Tree {split} ===")
    print(f"  AUC              : {m['auc']:.4f}")
    print(f"  Average Precision: {m['ap']:.4f}")
    print(f"  Accuracy         : {m['acc']:.4f}")

# Gráfico Decision Tree
v_train_dt = [m_train_dt["auc"], m_train_dt["ap"]]
v_test_dt  = [m_test_dt["auc"],  m_test_dt["ap"]]

fig3, ax3 = plt.subplots(figsize=(7, 5))
bars_tr3 = ax3.bar(x - width/2, v_train_dt, width, label="Treino", color="#4C72B0")
bars_te3 = ax3.bar(x + width/2, v_test_dt,  width, label="Teste",  color="#DD8452")

ax3.set_ylim(0, 1.05)
ax3.set_xticks(x)
ax3.set_xticklabels(metrics, fontsize=12)
ax3.set_ylabel("Score", fontsize=11)
ax3.legend(fontsize=11)
ax3.set_title(
    f"Decision Tree (max_depth=2, min_leaf=50) — {len(feature_cols)} features → {len(result.selected_features)} selecionadas",
    fontsize=11, pad=12,
)

for bar in list(bars_tr3) + list(bars_te3):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{bar.get_height():.3f}",
        ha="center", va="bottom", fontsize=10,
    )

fig3.tight_layout()
dt_plot_path = OUTPUT_DIR / "metrics_bar_decision_tree.png"
fig3.savefig(dt_plot_path, dpi=150)
plt.close(fig3)
print(f"Gráfico Decision Tree salvo em: {dt_plot_path}")

# Visualização da árvore
fig4, ax4 = plt.subplots(figsize=(60, 20))
plot_tree(
    dt,
    feature_names=result.selected_features,
    class_names=["Controle", "TEA"],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax4,
)
ax4.set_title(
    f"Decision Tree (max_depth=2, min_leaf=50) — {len(feature_cols)} features → {len(result.selected_features)} selecionadas",
    fontsize=14, pad=16,
)
fig4.tight_layout()
tree_viz_path = OUTPUT_DIR / "decision_tree.png"
fig4.savefig(tree_viz_path, dpi=120, bbox_inches="tight")
plt.close(fig4)
print(f"Árvore salva em: {tree_viz_path}")

# ---------------------------------------------------------------------------
# Histogramas por feature — top 10 separadas por target
# ---------------------------------------------------------------------------
HIST_DIR = OUTPUT_DIR / "histogramas_features"
HIST_DIR.mkdir(exist_ok=True)

top10 = result.selected_features[:10]
df_full = X_train.copy()
df_full["target"] = y_train

for feat in top10:
    vals_0 = df_full.loc[df_full["target"] == 0, feat]
    vals_1 = df_full.loc[df_full["target"] == 1, feat]

    fig_h, ax_h = plt.subplots(figsize=(7, 4))
    ax_h.hist(vals_0, bins=40, alpha=0.55, color="#4C72B0", label="Controle (0)", density=True)
    ax_h.hist(vals_1, bins=40, alpha=0.55, color="#DD8452", label="TEA (1)",      density=True)
    ax_h.set_xlabel(feat, fontsize=10)
    ax_h.set_ylabel("Densidade", fontsize=10)
    ax_h.legend(fontsize=10)
    ax_h.set_title(f"{feat}", fontsize=11, pad=10)
    fig_h.tight_layout()
    safe_name = feat.replace("/", "_").replace(" ", "_")
    fig_h.savefig(HIST_DIR / f"{safe_name}.png", dpi=130)
    plt.close(fig_h)

print(f"\nHistogramas salvos em: {HIST_DIR}")
