import ctypes
import logging
import multiprocessing as mp
import time
from multiprocessing.sharedctypes import RawArray
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


# =========================================================
# CONFIG
# =========================================================
CHUNKS = 1000
N_JOBS = -1  # use all logical CPUs; set manually if needed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =========================================================
# TYPES / SHARED STATE
# =========================================================
ArrayLike = Union[pd.DataFrame, np.ndarray]
shared_variables_dictionary = {}


# =========================================================
# HELPERS
# =========================================================
def convert_numeric_float32(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to numeric float32 when possible.
    Non-numeric columns are dropped in-place.

    IMPORTANT:
    This keeps numeric columns as float32 and does NOT convert back to float64.
    """
    logger.info("Converting columns to numeric float32 when possible")

    drop_cols = []

    for col in df.columns:
        try:
            # Convert to numeric, then force float32
            df[col] = pd.to_numeric(df[col], errors="raise").astype(np.float32)
        except Exception:
            drop_cols.append(col)

    if drop_cols:
        logger.info("Dropping %d non-numeric columns", len(drop_cols))
        df.drop(columns=drop_cols, inplace=True)

    logger.info("Remaining numeric columns: %d", len(df.columns))
    return df


# =========================================================
# NaNCorrMp using SPEARMAN and float32 end-to-end
# =========================================================
class NaNCorrMp:
    """
    IMPORTANT:
    This implementation uses SPEARMAN correlation, not Pearson.

    Also important:
    - Input data is kept as float32
    - Shared raw arrays use float32
    - Correlation matrix uses float32
    - No forced conversion back to float64
    """

    @staticmethod
    def _init_worker(
        X: RawArray,
        X_finite_mask: RawArray,
        X_corr: RawArray,
        X_shape: Tuple[int, int],
        X_corr_shape: Tuple[int, int],
        X_p_value: RawArray = None,
    ) -> None:
        shared_variables_dictionary["X"] = X
        shared_variables_dictionary["X_finite_mask"] = X_finite_mask
        shared_variables_dictionary["X_corr"] = X_corr
        shared_variables_dictionary["X_shape"] = X_shape
        shared_variables_dictionary["X_corr_shape"] = X_corr_shape
        shared_variables_dictionary["X_p_value"] = X_p_value
        shared_variables_dictionary["X_p_value_shape"] = X_corr_shape

    @staticmethod
    def calculate(X: ArrayLike, n_jobs: int = -1, chunks: int = 500) -> ArrayLike:
        return NaNCorrMp._calculate(
            X=X,
            n_jobs=n_jobs,
            chunks=chunks,
            add_p_values=False,
        )

    @staticmethod
    def calculate_with_p_value(
        X: ArrayLike,
        n_jobs: int = -1,
        chunks: int = 500,
    ) -> Tuple[ArrayLike, ArrayLike]:
        return NaNCorrMp._calculate(
            X=X,
            n_jobs=n_jobs,
            chunks=chunks,
            add_p_values=True,
        )

    @staticmethod
    def _calculate(
        X: ArrayLike,
        n_jobs: int,
        chunks: int,
        add_p_values: bool,
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
        logger.info("Preparing shared arrays for SPEARMAN correlation (float32 only)")

        if isinstance(X, pd.DataFrame):
            # Keep float32, avoid float64
            X_array = X.to_numpy(dtype=np.float32, copy=False).transpose()
            col_names = list(X.columns)
        else:
            X_array = np.asarray(X, dtype=np.float32)
            col_names = None

        logger.info("Input array shape after transpose to feature-first: %s", X_array.shape)
        logger.info("Input dtype: %s", X_array.dtype)

        # -------------------------------------------------
        # Shared raw data: float32
        # -------------------------------------------------
        X_raw = RawArray(ctypes.c_float, X_array.shape[0] * X_array.shape[1])
        X_np = np.frombuffer(X_raw, dtype=np.float32).reshape(X_array.shape)
        np.copyto(X_np, X_array)

        # -------------------------------------------------
        # Shared finite mask: bool
        # -------------------------------------------------
        finite_mask_data = np.isfinite(X_array)
        finite_mask_raw = RawArray(ctypes.c_bool, X_array.shape[0] * X_array.shape[1])
        finite_mask_np = np.frombuffer(finite_mask_raw, dtype=bool).reshape(X_array.shape)
        np.copyto(finite_mask_np, finite_mask_data)

        # -------------------------------------------------
        # Shared correlation matrix: float32
        # -------------------------------------------------
        n_features = X_array.shape[0]

        X_corr_raw = RawArray(ctypes.c_float, n_features * n_features)
        X_corr_np = np.frombuffer(X_corr_raw, dtype=np.float32).reshape((n_features, n_features))

        # -------------------------------------------------
        # Optional p-value matrix: float32
        # -------------------------------------------------
        if add_p_values:
            X_p_value_raw = RawArray(ctypes.c_float, n_features * n_features)
            X_p_value_np = np.frombuffer(X_p_value_raw, dtype=np.float32).reshape((n_features, n_features))
        else:
            X_p_value_raw = None
            X_p_value_np = None

        total_pairs = n_features * (n_features - 1) // 2

        logger.info("Number of features: %d", n_features)
        logger.info("Total feature pairs to compute: %d", total_pairs)
        logger.info("Chunksize: %d", chunks)

        arguments = ((j, i) for i in range(n_features) for j in range(i))
        processes = n_jobs if n_jobs > 0 else None

        worker_function = (
            NaNCorrMp._set_spearman_correlation_with_p_value
            if add_p_values
            else NaNCorrMp._set_spearman_correlation
        )

        logger.info("Starting multiprocessing pool for SPEARMAN")
        with mp.Pool(
            processes=processes,
            initializer=NaNCorrMp._init_worker,
            initargs=(
                X_raw,
                finite_mask_raw,
                X_corr_raw,
                X_np.shape,
                X_corr_np.shape,
                X_p_value_raw,
            ),
        ) as pool:
            list(pool.imap_unordered(worker_function, arguments, chunks))

        logger.info("Finished multiprocessing pool for SPEARMAN")

        for i in range(n_features):
            X_corr_np[i, i] = np.float32(1.0)
            if add_p_values:
                X_p_value_np[i, i] = np.float32(0.0)

        if add_p_values:
            if isinstance(X, pd.DataFrame):
                return (
                    pd.DataFrame(X_corr_np, columns=col_names, index=col_names),
                    pd.DataFrame(X_p_value_np, columns=col_names, index=col_names),
                )
            return X_corr_np, X_p_value_np

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_corr_np, columns=col_names, index=col_names)

        return X_corr_np

    @staticmethod
    def _get_global_variables(get_p_value: bool = False):
        X_np = np.frombuffer(
            shared_variables_dictionary["X"], dtype=np.float32
        ).reshape(shared_variables_dictionary["X_shape"])

        X_corr_np = np.frombuffer(
            shared_variables_dictionary["X_corr"], dtype=np.float32
        ).reshape(shared_variables_dictionary["X_corr_shape"])

        finite_mask = np.frombuffer(
            shared_variables_dictionary["X_finite_mask"], dtype=bool
        ).reshape(shared_variables_dictionary["X_shape"])

        if not get_p_value:
            return X_np, X_corr_np, finite_mask

        X_p_value_np = np.frombuffer(
            shared_variables_dictionary["X_p_value"], dtype=np.float32
        ).reshape(shared_variables_dictionary["X_p_value_shape"])

        return X_np, X_corr_np, finite_mask, X_p_value_np

    @staticmethod
    def _pearson_corr_on_arrays(x: np.ndarray, y: np.ndarray) -> np.float32:
        """
        Pearson formula on arrays.
        In this codebase it is applied to ranks, therefore final measure = SPEARMAN.
        Operates in float32.
        """
        if len(x) < 2:
            return np.float32(np.nan)

        x = x.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)

        mx = x.mean(dtype=np.float32)
        my = y.mean(dtype=np.float32)

        xm = x - mx
        ym = y - my

        r_num = np.add.reduce(xm * ym, dtype=np.float32)
        r_den = np.sqrt(
            np.sum(xm * xm, dtype=np.float32) * np.sum(ym * ym, dtype=np.float32),
            dtype=np.float32,
        )

        if r_den == 0:
            return np.float32(np.nan)

        r = np.float32(r_num / r_den)

        if r > 1:
            r = np.float32(1.0)
        elif r < -1:
            r = np.float32(-1.0)

        return r

    @staticmethod
    def _spearman_corr(x: np.ndarray, y: np.ndarray) -> np.float32:
        """
        SPEARMAN correlation:
        rank x, rank y, then Pearson on ranks.

        rankdata returns float64 by default, so we explicitly cast ranks to float32.
        """
        if len(x) < 2:
            return np.float32(np.nan)

        rx = stats.rankdata(x).astype(np.float32, copy=False)
        ry = stats.rankdata(y).astype(np.float32, copy=False)

        return NaNCorrMp._pearson_corr_on_arrays(rx, ry)

    @staticmethod
    def _set_spearman_correlation(arguments: Tuple[int, int]) -> None:
        j, i = arguments

        X_np, X_corr_np, finite_mask = NaNCorrMp._get_global_variables()

        finites = finite_mask[i] & finite_mask[j]
        x = X_np[i][finites]
        y = X_np[j][finites]

        corr = NaNCorrMp._spearman_corr(x, y)

        X_corr_np[i, j] = corr
        X_corr_np[j, i] = corr

    @staticmethod
    def _set_spearman_correlation_with_p_value(arguments: Tuple[int, int]) -> None:
        j, i = arguments

        X_np, X_corr_np, finite_mask, X_p_value_np = NaNCorrMp._get_global_variables(
            get_p_value=True
        )

        finites = finite_mask[i] & finite_mask[j]
        x = X_np[i][finites]
        y = X_np[j][finites]

        corr = NaNCorrMp._spearman_corr(x, y)

        X_corr_np[i, j] = corr
        X_corr_np[j, i] = corr

        p_value = NaNCorrMp._p_value(float(corr), len(x))

        X_p_value_np[i, j] = np.float32(p_value)
        X_p_value_np[j, i] = np.float32(p_value)

    @staticmethod
    def _p_value(corr: float, observation_length: int) -> float:
        """
        Approximate p-value from correlation and sample size.
        Scalar math here may use Python/scipy float internally, which is fine.
        Stored output goes back to float32.
        """
        if observation_length < 3 or np.isnan(corr) or abs(corr) >= 1:
            if abs(corr) == 1:
                return 0.0
            return 1.0

        t_stat = corr * np.sqrt((observation_length - 2) / (1 - corr**2))
        p_value = 2 * stats.t.sf(np.abs(t_stat), df=observation_length - 2)
        return float(p_value)


# =========================================================
# Pairwise output helper
# =========================================================
def pairwise_corr_nancorrmp(
    df: pd.DataFrame,
    features: list[str],
    n_jobs: int = -1,
    chunks: int = 500,
) -> pd.DataFrame:
    """
    Computes pairwise SPEARMAN correlations using NaNCorrMp
    and returns a long dataframe without duplicate pairs.

    IMPORTANT:
    - This returns SPEARMAN, not Pearson
    - Internal data/correlation matrix kept in float32
    """
    logger.info("Building dataframe subset for SPEARMAN correlation")
    X = df[features]

    logger.info("Calling NaNCorrMp.calculate (SPEARMAN, float32)")
    corr_matrix = NaNCorrMp.calculate(
        X,
        n_jobs=n_jobs,
        chunks=chunks,
    )

    logger.info("Converting SPEARMAN matrix to long format")
    corr_long = corr_matrix.stack(dropna=False).reset_index()
    corr_long.columns = ["feature_1", "feature_2", "correlation"]

    logger.info("Removing diagonal and duplicate pairs")
    corr_long = corr_long[corr_long["feature_1"] < corr_long["feature_2"]].reset_index(
        drop=True
    )

    return corr_long


# =========================================================
# MAIN
# =========================================================
def main():
    logger.info("Starting correlation pipeline")
    logger.info("Correlation method: SPEARMAN")
    logger.info("Internal dtype target: float32 only")

    input_path = Path("input.parquet")
    output_path = Path("correlations.csv")

    logger.info("Reading parquet dataset from: %s", input_path.resolve())
    df = pd.read_parquet(input_path)

    logger.info("Dataset loaded successfully")
    logger.info("Original dataframe shape: %s", df.shape)

    logger.info("Converting dataframe columns to numeric float32 when possible")
    before_cols = len(df.columns)
    df = convert_numeric_float32(df)
    after_cols = len(df.columns)

    logger.info("Numeric conversion finished")
    logger.info("Columns before conversion: %d", before_cols)
    logger.info("Columns after conversion: %d", after_cols)
    logger.info("Dataframe shape after conversion: %s", df.shape)

    features = list(df.columns)
    logger.info("Number of features used for SPEARMAN correlation: %d", len(features))

    logger.info("Starting pairwise SPEARMAN correlation computation")
    start_time = time.time()

    corr_pairs = pairwise_corr_nancorrmp(
        df=df,
        features=features,
        n_jobs=N_JOBS,
        chunks=CHUNKS,
    )

    elapsed = time.time() - start_time

    logger.info("Pairwise SPEARMAN correlation computation finished")
    logger.info("Elapsed time: %.2f seconds", elapsed)
    logger.info("Total correlation pairs: %d", len(corr_pairs))

    logger.info("Sample of resulting SPEARMAN correlations:")
    logger.info("\n%s", corr_pairs.head())

    logger.info("Saving results to CSV: %s", output_path.resolve())
    corr_pairs.to_csv(output_path, index=False)

    logger.info("File saved successfully")
    logger.info("Pipeline finished successfully")


if __name__ == "__main__":
    main()