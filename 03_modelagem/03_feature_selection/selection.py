"""
feature_selection.py
====================
Seleção de features via torneio eliminatório em chunks.
Estratégia para cenários p >> n (muito mais features que amostras).

Uso rápido:
    from feature_selection import FeatureSelector

    selector = FeatureSelector(target_col="label")
    result = selector.fit(df, feature_cols=["f1", "f2", ...])
    df_selected = result.transform(df)
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold

# ---------------------------------------------------------------------------
# Controle de paralelismo — altere aqui ou via variável de ambiente
# ---------------------------------------------------------------------------

USE_PARALLEL: int = 0
TOTAL_THREADS: int = 40
THREADS_PER_TREE: int = 4
MAX_PARALLEL_JOBS: int = max(1, TOTAL_THREADS // THREADS_PER_TREE)  # 10 jobs simultâneos

# ---------------------------------------------------------------------------
# Configuração dos hiperparâmetros da árvore — edite aqui
# ---------------------------------------------------------------------------

@dataclass
class TreeConfig:
    """
    Hiperparâmetros do ExtraTrees usado em cada etapa do torneio.

    Atributos
    ----------
    n_estimators : int
        Número de árvores na floresta.
    max_depth : int
        Profundidade máxima de cada árvore.
        Valores baixos (3-5) evitam overfit com poucas amostras.
    min_samples_leaf : int
        Mínimo de amostras por folha.
        Com ~1.000 linhas, use 15-30 (1.5% a 3% dos dados).
    min_samples_split : int
        Mínimo de amostras para dividir um nó.
    max_features : str | float
        Fração ou estratégia de features por split.
        "sqrt" é conservador; 0.5 é mais agressivo.
    n_estimators_final : int
        Número de árvores no round final (pode ser maior para mais estabilidade).
    task : str
        "classification" ou "regression".
    random_state : int
        Semente para reprodutibilidade.
    """
    n_estimators: int = 50
    max_depth: int = 4
    min_samples_leaf: int = 20
    min_samples_split: int = 40
    max_features: str | float = "sqrt"
    n_estimators_final: int = 100
    task: str = "classification"          # "classification" | "regression"
    random_state: int = 42


# ---------------------------------------------------------------------------
# Configuração da estratégia de torneio — edite aqui
# ---------------------------------------------------------------------------

@dataclass
class TournamentConfig:
    """
    Controla o comportamento do torneio eliminatório.

    Atributos
    ----------
    chunk_size : int
        Quantas features competem em cada bloco.
        Regra: chunk_size >> n_samples para não overfit.
    top_k_pct : float
        Fração de features que sobrevivem de cada chunk (ex: 0.08 = 8%).
    final_top_n : int
        Número máximo de features no output final.
    n_tournament_seeds : int
        Quantas vezes o torneio é repetido com seeds diferentes.
        Aumenta estabilidade. Features que aparecem em todas as rodadas
        são as mais confiáveis.
    seed_consensus : str
        "union"       → qualquer seed salva a feature (conservador, mais features)
        "majority"    → mais da metade das seeds (balanceado)
        "intersection"→ todas as seeds precisam eleger (restritivo, menos features)
    variance_threshold : float | None
        Remove features com variância abaixo desse valor antes do torneio.
        None desativa esse pré-filtro.
    verbose : bool
        Exibe logs de progresso.
    """
    chunk_size: int = 2000
    top_k_pct: float = 0.08
    final_top_n: int = 200
    n_tournament_seeds: int = 3
    seed_consensus: str = "majority"      # "union" | "majority" | "intersection"
    variance_threshold: Optional[float] = 0.01
    verbose: bool = True


# ---------------------------------------------------------------------------
# Resultado do fit
# ---------------------------------------------------------------------------

@dataclass
class SelectionResult:
    """
    Resultado retornado pelo FeatureSelector.fit().

    Atributos
    ----------
    selected_features : List[str]
        Nomes das features selecionadas, em ordem decrescente de importância.
    importances : np.ndarray
        Importâncias médias das features selecionadas (alinhadas com selected_features).
    all_rounds_info : List[dict]
        Log de cada seed/rodada para auditoria.
    dropped_by_variance : List[str]
        Features eliminadas pelo pré-filtro de variância.
    """
    selected_features: List[str]
    importances: np.ndarray
    all_rounds_info: List[dict] = field(default_factory=list)
    dropped_by_variance: List[str] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """Retorna DataFrame com features e importâncias."""
        return pd.DataFrame({
            "feature": self.selected_features,
            "importance": self.importances,
        })

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra o DataFrame mantendo apenas as features selecionadas."""
        return df[self.selected_features].copy()


# ---------------------------------------------------------------------------
# Core: uma rodada do torneio
# ---------------------------------------------------------------------------

def _tree_n_jobs() -> int:
    """Threads por árvore: 4 no paralelo, todos os cores no sequencial."""
    return THREADS_PER_TREE if USE_PARALLEL else -1


def _build_model(tree_cfg: TreeConfig, seed: int):
    """Instancia o modelo de acordo com a task."""
    common = dict(
        n_estimators=tree_cfg.n_estimators,
        max_depth=tree_cfg.max_depth,
        min_samples_leaf=tree_cfg.min_samples_leaf,
        min_samples_split=tree_cfg.min_samples_split,
        max_features=tree_cfg.max_features,
        random_state=seed,
        n_jobs=_tree_n_jobs(),
    )
    if tree_cfg.task == "classification":
        return ExtraTreesClassifier(**common)
    else:
        return ExtraTreesRegressor(**common)


def _build_final_model(tree_cfg: TreeConfig, seed: int):
    """Modelo final com mais estimators para maior estabilidade."""
    common = dict(
        n_estimators=tree_cfg.n_estimators_final,
        max_depth=tree_cfg.max_depth,
        min_samples_leaf=max(1, tree_cfg.min_samples_leaf - 5),  # leve afrouxamento
        min_samples_split=tree_cfg.min_samples_split,
        max_features=tree_cfg.max_features,
        random_state=seed,
        n_jobs=_tree_n_jobs(),
    )
    if tree_cfg.task == "classification":
        return ExtraTreesClassifier(**common)
    else:
        return ExtraTreesRegressor(**common)


def _run_single_tournament(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: np.ndarray,
    tree_cfg: TreeConfig,
    tour_cfg: TournamentConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Executa um torneio completo com um único seed.

    Retorna
    -------
    selected_names : np.ndarray de str
        Nomes das features selecionadas.
    importances : np.ndarray de float
        Importâncias correspondentes.
    """
    rng = np.random.RandomState(seed)
    n_features = X.shape[1]
    current_indices = np.arange(n_features)

    round_num = 1
    while True:
        # Embaralha pra evitar bias posicional
        rng.shuffle(current_indices)

        chunks = [
            current_indices[i : i + tour_cfg.chunk_size]
            for i in range(0, len(current_indices), tour_cfg.chunk_size)
        ]

        if tour_cfg.verbose:
            print(
                f"    [seed={seed}] Round {round_num}: "
                f"{len(current_indices)} features → {len(chunks)} chunk(s) de ~{tour_cfg.chunk_size}"
            )

        next_survivors: List[int] = []

        for chunk_idx in chunks:
            X_chunk = X[:, chunk_idx]
            model = _build_model(tree_cfg, seed + round_num)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_chunk, y)

            importances = model.feature_importances_
            top_k = max(1, int(len(chunk_idx) * tour_cfg.top_k_pct))
            top_local = np.argsort(importances)[-top_k:]
            next_survivors.extend(chunk_idx[top_local].tolist())

        current_indices = np.array(next_survivors)

        if tour_cfg.verbose:
            print(f"           → {len(current_indices)} sobreviventes")

        # Critério de parada: pool pequeno o suficiente para round final
        if len(current_indices) <= tour_cfg.final_top_n * 2:
            break

        round_num += 1

    # ---- Round final: modelo único no pool restante ----
    if tour_cfg.verbose:
        print(f"    [seed={seed}] Round Final: modelo único em {len(current_indices)} features")

    X_final = X[:, current_indices]
    final_model = _build_final_model(tree_cfg, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(X_final, y)

    final_importances = final_model.feature_importances_
    top_n = min(tour_cfg.final_top_n, len(current_indices))
    top_final = np.argsort(final_importances)[-top_n:]

    selected_global_idx = current_indices[top_final]
    selected_importances = final_importances[top_final]

    # Ordena decrescente
    order = np.argsort(selected_importances)[::-1]
    return feature_names[selected_global_idx[order]], selected_importances[order]


# ---------------------------------------------------------------------------
# Worker para execução paralela de seeds
# ---------------------------------------------------------------------------

def _run_tournament_worker(
    args: tuple,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """
    Wrapper picklável para ProcessPoolExecutor.
    Suprime verbose nos workers para evitar output embaralhado.
    Retorna (seed_index, seed, names, importances).
    """
    idx, X, y, feature_names, tree_cfg, tour_cfg, seed = args
    quiet_cfg = dataclasses.replace(tour_cfg, verbose=False)
    names, imps = _run_single_tournament(X, y, feature_names, tree_cfg, quiet_cfg, seed)
    return idx, seed, names, imps


# ---------------------------------------------------------------------------
# Consenso entre seeds
# ---------------------------------------------------------------------------

def _apply_consensus(
    all_selected: List[np.ndarray],
    all_importances: List[np.ndarray],
    consensus: str,
    n_seeds: int,
    final_top_n: int,
) -> tuple[List[str], np.ndarray]:
    """
    Combina resultados de múltiplos seeds usando a estratégia de consenso.

    Retorna features únicas com importância média, ordenadas.
    """
    # Conta em quantas seeds cada feature apareceu
    from collections import defaultdict

    vote_count: dict[str, int] = defaultdict(int)
    importance_sum: dict[str, float] = defaultdict(float)

    for names, imps in zip(all_selected, all_importances):
        for name, imp in zip(names, imps):
            vote_count[name] += 1
            importance_sum[name] += imp

    # Filtra por consenso
    if consensus == "intersection":
        threshold = n_seeds
    elif consensus == "majority":
        threshold = max(1, (n_seeds // 2) + 1)
    else:  # union
        threshold = 1

    qualified = {
        name: importance_sum[name] / vote_count[name]
        for name, count in vote_count.items()
        if count >= threshold
    }

    if not qualified:
        # Fallback: usa union se o consenso não encontrar nada
        warnings.warn(
            f"Consenso '{consensus}' não encontrou features. Usando union como fallback.",
            UserWarning,
        )
        qualified = {
            name: importance_sum[name] / vote_count[name]
            for name in vote_count
        }

    # Ordena por importância média e limita ao final_top_n
    sorted_features = sorted(qualified.items(), key=lambda x: x[1], reverse=True)
    sorted_features = sorted_features[:final_top_n]

    names_out = [f for f, _ in sorted_features]
    imps_out = np.array([i for _, i in sorted_features])
    return names_out, imps_out


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class FeatureSelector:
    """
    Seleção de features via torneio eliminatório em chunks com ExtraTrees.

    Parâmetros
    ----------
    target_col : str
        Nome da coluna target no DataFrame.
    tree_cfg : TreeConfig, opcional
        Hiperparâmetros da árvore. Se None, usa defaults.
    tour_cfg : TournamentConfig, opcional
        Configuração do torneio. Se None, usa defaults.

    Exemplo
    -------
    >>> selector = FeatureSelector(
    ...     target_col="survived",
    ...     tree_cfg=TreeConfig(max_depth=3, min_samples_leaf=25),
    ...     tour_cfg=TournamentConfig(chunk_size=1000, final_top_n=100),
    ... )
    >>> result = selector.fit(df, feature_cols=feature_list)
    >>> df_selected = result.transform(df)
    >>> print(result.summary())
    """

    def __init__(
        self,
        target_col: str,
        tree_cfg: Optional[TreeConfig] = None,
        tour_cfg: Optional[TournamentConfig] = None,
    ):
        self.target_col = target_col
        self.tree_cfg = tree_cfg or TreeConfig()
        self.tour_cfg = tour_cfg or TournamentConfig()
        self._result: Optional[SelectionResult] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> SelectionResult:
        """
        Executa o torneio de seleção.

        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame completo com features e target.
        feature_cols : list[str] | None
            Lista de colunas a considerar como features.
            Se None, usa todas as colunas exceto target_col.

        Retorna
        -------
        SelectionResult
        """
        # --- Prepara features e target ---
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != self.target_col]

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas não encontradas no DataFrame: {missing}")

        if self.target_col not in df.columns:
            raise ValueError(f"Target '{self.target_col}' não encontrado no DataFrame.")

        X_df = df[feature_cols].copy()
        y = df[self.target_col].values
        feature_names = np.array(feature_cols)

        if self.tour_cfg.verbose:
            print(f"[FeatureSelector] Início — {X_df.shape[1]} features, {X_df.shape[0]} amostras")

        # --- Pré-filtro: variância ---
        dropped_by_variance: List[str] = []
        if self.tour_cfg.variance_threshold is not None:
            vt = VarianceThreshold(threshold=self.tour_cfg.variance_threshold)
            X_var = vt.fit_transform(X_df.values)
            mask = vt.get_support()
            dropped_by_variance = feature_names[~mask].tolist()
            feature_names = feature_names[mask]
            X_values = X_var.astype(np.float32)
            if self.tour_cfg.verbose:
                print(
                    f"[VarianceThreshold] {len(dropped_by_variance)} features removidas "
                    f"→ {len(feature_names)} restantes"
                )
        else:
            X_values = X_df.values.astype(np.float32)

        # --- Torneio multi-seed ---
        all_selected: List[np.ndarray] = []
        all_importances: List[np.ndarray] = []
        rounds_info: List[dict] = []

        seeds = [
            self.tree_cfg.random_state + i * 1000
            for i in range(self.tour_cfg.n_tournament_seeds)
        ]

        if USE_PARALLEL and len(seeds) > 1:
            if self.tour_cfg.verbose:
                print(
                    f"[Paralelo] {len(seeds)} seeds × {THREADS_PER_TREE} threads/árvore "
                    f"— até {MAX_PARALLEL_JOBS} jobs simultâneos (total ≤ {TOTAL_THREADS} threads)"
                )
            worker_args = [
                (i, X_values, y, feature_names, self.tree_cfg, self.tour_cfg, seed)
                for i, seed in enumerate(seeds)
            ]
            ordered: dict[int, tuple] = {}
            with ProcessPoolExecutor(max_workers=min(MAX_PARALLEL_JOBS, len(seeds))) as executor:
                futures = {executor.submit(_run_tournament_worker, args): args[0] for args in worker_args}
                for future in as_completed(futures):
                    idx, seed, names, imps = future.result()
                    ordered[idx] = (seed, names, imps)
                    if self.tour_cfg.verbose:
                        print(f"  ✓ Torneio seed={seed} concluído — {len(names)} features")

            for i in sorted(ordered):
                seed, names, imps = ordered[i]
                all_selected.append(names)
                all_importances.append(imps)
                rounds_info.append({"seed": seed, "n_selected": len(names), "features": names.tolist()})
        else:
            for i, seed in enumerate(seeds):
                if self.tour_cfg.verbose:
                    print(f"\n[Torneio {i+1}/{len(seeds)}] seed={seed}")

                names, imps = _run_single_tournament(
                    X=X_values,
                    y=y,
                    feature_names=feature_names,
                    tree_cfg=self.tree_cfg,
                    tour_cfg=self.tour_cfg,
                    seed=seed,
                )
                all_selected.append(names)
                all_importances.append(imps)
                rounds_info.append({"seed": seed, "n_selected": len(names), "features": names.tolist()})

        # --- Consenso ---
        if self.tour_cfg.verbose:
            print(f"\n[Consenso] Estratégia: '{self.tour_cfg.seed_consensus}'")

        final_features, final_importances = _apply_consensus(
            all_selected=all_selected,
            all_importances=all_importances,
            consensus=self.tour_cfg.seed_consensus,
            n_seeds=len(seeds),
            final_top_n=self.tour_cfg.final_top_n,
        )

        if self.tour_cfg.verbose:
            print(f"[FeatureSelector] Concluído — {len(final_features)} features selecionadas")

        self._result = SelectionResult(
            selected_features=final_features,
            importances=final_importances,
            all_rounds_info=rounds_info,
            dropped_by_variance=dropped_by_variance,
        )
        return self._result

    # ------------------------------------------------------------------
    # Conveniência
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Executa fit e já retorna o DataFrame filtrado (sem a coluna target)."""
        result = self.fit(df, feature_cols)
        return result.transform(df)

    @property
    def result(self) -> Optional[SelectionResult]:
        """Retorna o último SelectionResult após fit()."""
        return self._result


# ---------------------------------------------------------------------------
# Exemplo de uso — executar direto: python feature_selection.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Exemplo: 20.000 features, 1.000 amostras, classificação binária")
    print("=" * 60)

    rng = np.random.default_rng(0)
    N_SAMPLES, N_FEATURES = 1_000, 20_000
    X_demo = rng.standard_normal((N_SAMPLES, N_FEATURES)).astype(np.float32)
    y_demo = rng.integers(0, 2, N_SAMPLES)

    # 50 features com sinal real
    signal_idx = rng.choice(N_FEATURES, 50, replace=False)
    for idx in signal_idx:
        X_demo[:, idx] += y_demo * 2.0

    feature_names_demo = [f"feat_{i:05d}" for i in range(N_FEATURES)]
    df_demo = pd.DataFrame(X_demo, columns=feature_names_demo)
    df_demo["target"] = y_demo

    # --- Configure aqui ---
    tree_config = TreeConfig(
        n_estimators=50,
        max_depth=4,
        min_samples_leaf=20,
        min_samples_split=40,
        max_features="sqrt",
        n_estimators_final=100,
        task="classification",
        random_state=42,
    )

    tournament_config = TournamentConfig(
        chunk_size=2000,
        top_k_pct=0.08,
        final_top_n=200,
        n_tournament_seeds=3,
        seed_consensus="majority",
        variance_threshold=0.01,
        verbose=True,
    )

    selector = FeatureSelector(
        target_col="target",
        tree_cfg=tree_config,
        tour_cfg=tournament_config,
    )

    t0 = time.time()
    result = selector.fit(df_demo, feature_cols=feature_names_demo)
    elapsed = time.time() - t0

    print(f"\nTempo total: {elapsed:.1f}s")
    print(f"Features selecionadas: {len(result.selected_features)}")
    print(f"Features com sinal real recuperadas: "
          f"{sum(f'feat_{i:05d}' in result.selected_features for i in signal_idx)}/50")
    print("\nTop 20:")
    print(result.summary().head(20).to_string(index=False))