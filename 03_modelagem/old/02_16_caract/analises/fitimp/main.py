from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import config
from feature_names import load_feature_names_from_nomescolunas, try_get_model_feature_names
from io_utils import (
    find_all_prefixed_model_pkls,
    load_pickle,
    model_basename,
    parse_feature_and_model_from_filename,
)
from plot_utils import plot_top_cum_percent
from models.registry import resolve_importance_handler, normalize_model_key_for_folder


def save_debug_yaml(path: Path, payload: Dict[str, Any]) -> None:
    lines = [f"{k}: {v}" for k, v in payload.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_one_model(pkl_path: Path, all_feature_names: list[str]) -> None:
    parsed = parse_feature_and_model_from_filename(pkl_path.name)
    if parsed is None:
        print(f"[WARN] Skipping (unexpected name): {pkl_path.name}")
        return

    feature_set, model_key = parsed

    if config.ALLOWED_FEATURE_SETS is not None and feature_set not in config.ALLOWED_FEATURE_SETS:
        print(f"[INFO] Skipping feature_set={feature_set} not in allowed list.")
        return

    base = model_basename(pkl_path)
    print(f"\n[INFO] Loading: {pkl_path.name} | feature_set={feature_set} | model_key={model_key}")

    model = load_pickle(pkl_path)

    # feature names priority:
    # 1) model-provided names
    # 2) nomescolunas.txt
    model_names = try_get_model_feature_names(model)
    if model_names:
        feature_names = model_names
        names_source = "model"
    else:
        feature_names = all_feature_names
        names_source = "nomescolunas.txt"

    handler = resolve_importance_handler(model_key=model_key, model=model)
    if handler is None:
        print(f"[WARN] No importance handler for model_key={model_key} | type={type(model)}. Skipping.")
        return

    df, imp_meta = handler.compute_importance(model=model, feature_names=feature_names)

    # output dir: out_mod/<feature_set>/<model_key>/
    model_folder = normalize_model_key_for_folder(model_key)
    out_dir = config.OUT_DIR / feature_set / model_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"{base}__importance.csv"
    out_png = out_dir / f"{base}__cum{config.CUM_CUTOFF_PCT_TO_PLOT:.1f}pct.png"
    out_yaml = out_dir / f"{base}__debug.yaml"

    df_out = df[["variable", "abs_importance", "pct_importance", "cum_pct_importance"]].copy()
    df_out.to_csv(out_csv, index=False)
    print(f"[OK] CSV saved: {out_csv.resolve()}")

    n_plotted = plot_top_cum_percent(
        df=df,
        out_png=out_png,
        cutoff_pct=config.CUM_CUTOFF_PCT_TO_PLOT,
        title=f"{feature_set} | {model_key} | {handler.name} | {imp_meta} | cum {config.CUM_CUTOFF_PCT_TO_PLOT:.1f}%",
        max_features=config.MAX_FEATURES_IN_PLOT,
    )

    save_debug_yaml(
        out_yaml,
        {
            "pickle": pkl_path.name,
            "basename": base,
            "feature_set": feature_set,
            "model_key_from_filename": model_key,
            "model_python_type": type(model).__name__,
            "handler": handler.name,
            "importance_meta": imp_meta,
            "names_source": names_source,
            "n_features_names": len(feature_names),
            "n_rows_df": len(df),
            "cum_cutoff_pct_plotted": config.CUM_CUTOFF_PCT_TO_PLOT,
            "n_features_plotted": n_plotted,
        },
    )
    print(f"[OK] Debug YAML saved: {out_yaml.resolve()}")


def main() -> None:
    print(f"[INFO] Reading feature names: {config.NOMES_COLUNAS_PATH}")
    all_feature_names = load_feature_names_from_nomescolunas(config.NOMES_COLUNAS_PATH)

    print(f"[INFO] Scanning pkls: {config.PKLS_DIR} (only names containing '__')")
    pkls = find_all_prefixed_model_pkls(config.PKLS_DIR)
    if not pkls:
        raise RuntimeError(f"No prefixed pkls found in: {config.PKLS_DIR}")

    print(f"[INFO] Found {len(pkls)} prefixed model pkls.")
    for p in pkls[:30]:
        print("  -", p.name)
    if len(pkls) > 30:
        print(f"  ... +{len(pkls)-30} more")

    for p in pkls:
        process_one_model(p, all_feature_names)


if __name__ == "__main__":
    main()
