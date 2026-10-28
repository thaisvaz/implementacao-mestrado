# -*- coding: utf-8 -*-
"""
TITULO:
Extracao do atlas Yeo7 + extracao automatica do CC200 a partir de scorr05_mean_all
+ calculo do percentual de cada ROI do CC200 dentro da DMN
+ geracao de base final por ROI + graficos diagnosticos

O que este script faz:
1) Baixa/prepara o atlas Yeo 7 networks
2) Marca corretamente a DMN (label 7)
3) Lê o arquivo local scorr05_mean_all.nii.gz do Craddock
4) Escolhe automaticamente o volume mais próximo de 200 ROIs
5) Extrai o CC200 em 3D
6) Reamostra o Yeo7 para o espaço do CC200
7) Calcula, para cada ROI, o percentual de voxels pertencentes à DMN
8) Gera uma base final por ROI com flags para vários thresholds
9) Salva gráficos para inspeção

Requisitos:
    python -m pip install numpy pandas nibabel nilearn matplotlib

Como usar:
    1) Coloque este .py na mesma pasta do arquivo:
           scorr05_mean_all.nii.gz
    2) Rode:
           python nome_do_arquivo.py

Saídas:
    pasta atlases_extraidos/
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilearn.datasets import fetch_atlas_yeo_2011
from nilearn.image import load_img, resample_to_img


# ============================================================
# CONFIG
# ============================================================

CRADDOCK_LOCAL = "scorr05_mean_all.nii.gz"

OUT_DIR = Path("atlases_extraidos")
OUT_DIR.mkdir(exist_ok=True)

YEO7_FIXED_PATH = OUT_DIR / "yeo7_3d_fixed.nii.gz"
YEO7_LUT_PATH = OUT_DIR / "yeo7_lut_annotated.csv"
YEO7_DMN_LABEL_PATH = OUT_DIR / "yeo7_dmn_label.csv"
YEO7_LABEL_COUNTS_PATH = OUT_DIR / "yeo7_label_counts.csv"

CC200_PATH = OUT_DIR / "cc200_3d.nii.gz"
CC200_VOLUME_INFO_PATH = OUT_DIR / "craddock_volume_roi_counts.csv"

FINAL_ROI_OVERLAP_PATH = OUT_DIR / "cc200_dmn_overlap_por_roi.csv"

PLOT_HIST_PATH = OUT_DIR / "grafico_distribuicao_pct_dmn_por_roi.png"
PLOT_CURVE_PATH = OUT_DIR / "grafico_curva_threshold_vs_numero_de_rois_dmn.png"

DMN_LABEL = 7
THRESHOLDS_FOR_FLAGS = [0.50, 0.60, 0.70, 0.80]


# ============================================================
# HELPERS
# ============================================================

def save_nifti_like(data: np.ndarray, ref_img, out_path: Path):
    out_img = nib.Nifti1Image(data, ref_img.affine, ref_img.header)
    nib.save(out_img, str(out_path))


def make_yeo7_ready():
    print("\n" + "=" * 80)
    print("PASSO 1 - BAIXAR E PREPARAR YEO7")
    print("=" * 80)

    yeo = fetch_atlas_yeo_2011(n_networks=7, thickness="thick")

    maps_obj = yeo["maps"]
    if isinstance(maps_obj, str):
        img = nib.load(maps_obj)
    else:
        img = maps_obj

    data = img.get_fdata()
    print("Yeo7 shape original:", data.shape)

    # transformar de (X, Y, Z, 1) para (X, Y, Z)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]

    data = np.rint(data).astype(np.int16)

    save_nifti_like(data, img, YEO7_FIXED_PATH)

    print("Yeo7 shape final:", data.shape)
    print("Yeo7 labels únicos:", np.unique(data))
    print("Yeo7 salvo em:", YEO7_FIXED_PATH)

    lut = yeo["lut"].copy()

    canonical_map = {
        0: "Background",
        1: "Visual",
        2: "Somatomotor",
        3: "Dorsal Attention",
        4: "Ventral Attention / Salience",
        5: "Limbic",
        6: "Frontoparietal / Control",
        7: "Default Mode Network",
    }

    lut["canonical_name"] = lut["index"].map(canonical_map)
    lut["is_dmn"] = (lut["index"] == DMN_LABEL).astype(int)
    lut.to_csv(YEO7_LUT_PATH, index=False)

    dmn_row = lut.loc[lut["index"] == DMN_LABEL, ["index", "name", "canonical_name", "is_dmn"]].copy()
    dmn_row.to_csv(YEO7_DMN_LABEL_PATH, index=False)

    labels, counts = np.unique(data, return_counts=True)
    df_counts = pd.DataFrame({
        "label": labels,
        "n_voxels": counts,
    })
    df_counts = df_counts.merge(
        lut[["index", "name", "canonical_name", "is_dmn"]],
        left_on="label",
        right_on="index",
        how="left",
    ).drop(columns="index")
    df_counts.to_csv(YEO7_LABEL_COUNTS_PATH, index=False)

    print("\nLUT anotado salvo em:", YEO7_LUT_PATH)
    print("Arquivo DMN salvo em:", YEO7_DMN_LABEL_PATH)
    print("Contagem de labels salva em:", YEO7_LABEL_COUNTS_PATH)
    print("\nResumo Yeo7:")
    print(df_counts)

    return data


def inspect_and_extract_cc200():
    print("\n" + "=" * 80)
    print("PASSO 2 - LER CRADDOCK LOCAL E ESCOLHER AUTOMATICAMENTE O VOLUME ~200 ROIS")
    print("=" * 80)

    img = nib.load(CRADDOCK_LOCAL)
    data = img.get_fdata()

    print("Craddock shape:", data.shape)
    print("Craddock ndim:", data.ndim)
    print("Craddock min/max:", np.nanmin(data), np.nanmax(data))

    if data.ndim != 4:
        raise ValueError(f"Esperava arquivo 4D, mas veio {data.ndim}D")

    roi_counts = []
    for i in range(data.shape[3]):
        vol = data[..., i]
        n_rois = len(np.unique(vol[vol > 0]))
        roi_counts.append((i, n_rois))

    df_roi_counts = pd.DataFrame(roi_counts, columns=["volume_idx", "n_rois"])
    df_roi_counts["dist_to_200"] = (df_roi_counts["n_rois"] - 200).abs()
    df_roi_counts = df_roi_counts.sort_values(["dist_to_200", "volume_idx"]).reset_index(drop=True)
    df_roi_counts.to_csv(CC200_VOLUME_INFO_PATH, index=False)

    print("\nVolumes mais próximos de 200 ROIs:")
    print(df_roi_counts.head(10))

    best_vol = int(df_roi_counts.iloc[0]["volume_idx"])
    best_n_rois = int(df_roi_counts.iloc[0]["n_rois"])

    print(f"\nVolume escolhido automaticamente: {best_vol}")
    print(f"Número de ROIs desse volume: {best_n_rois}")

    cc200 = data[..., best_vol]
    cc200 = np.rint(cc200).astype(np.int16)

    save_nifti_like(cc200, img, CC200_PATH)

    print("\nCC200 salvo em:", CC200_PATH)
    print("CC200 shape:", cc200.shape)
    print("CC200 n labels > 0:", len(np.unique(cc200[cc200 > 0])))

    return best_vol, best_n_rois, cc200


def compute_overlap_and_save():
    print("\n" + "=" * 80)
    print("PASSO 3 - REAMOSTRAR YEO7 PARA O ESPACO DO CC200 E CALCULAR PERCENTUAL DMN POR ROI")
    print("=" * 80)

    cc_img = load_img(str(CC200_PATH))
    yeo_img = load_img(str(YEO7_FIXED_PATH))

    print("CC200 shape original:", cc_img.shape)
    print("Yeo7 shape original:", yeo_img.shape)

    yeo_resampled = resample_to_img(
        source_img=yeo_img,
        target_img=cc_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )

    cc = np.asanyarray(cc_img.dataobj)
    yeo = np.asanyarray(yeo_resampled.dataobj)

    cc = np.rint(cc).astype(np.int16)
    yeo = np.rint(yeo).astype(np.int16)

    print("CC200 shape final:", cc.shape)
    print("Yeo7 reamostrado shape:", yeo.shape)

    if cc.shape != yeo.shape:
        raise ValueError(f"Shapes diferentes: cc={cc.shape}, yeo={yeo.shape}")

    roi_ids = np.unique(cc)
    roi_ids = roi_ids[np.isfinite(roi_ids)]
    roi_ids = roi_ids[roi_ids > 0].astype(int)

    rows = []
    for roi in roi_ids:
        mask = (cc == roi)
        n_voxels_roi = int(mask.sum())

        if n_voxels_roi == 0:
            n_voxels_dmn = 0
            pct_dmn = 0.0
        else:
            n_voxels_dmn = int((yeo[mask] == DMN_LABEL).sum())
            pct_dmn = n_voxels_dmn / n_voxels_roi

        row = {
            "roi": int(roi),
            "n_voxels_roi": n_voxels_roi,
            "n_voxels_dmn": n_voxels_dmn,
            "pct_dmn": float(pct_dmn),
        }

        for thr in THRESHOLDS_FOR_FLAGS:
            col = f"flag_dmn_{str(thr).replace('.', '')}"
            row[col] = int(pct_dmn >= thr)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("roi").reset_index(drop=True)
    df.to_csv(FINAL_ROI_OVERLAP_PATH, index=False)

    print("\nBase final por ROI salva em:", FINAL_ROI_OVERLAP_PATH)
    print("\nPrimeiras linhas:")
    print(df.head(20))

    print("\nResumo:")
    print(df.describe(include="all"))

    for thr in THRESHOLDS_FOR_FLAGS:
        col = f"flag_dmn_{str(thr).replace('.', '')}"
        print(f"ROIs com {col} = 1:", int(df[col].sum()))

    return df


def make_plots(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("PASSO 4 - GERAR GRAFICOS")
    print("=" * 80)

    # --------------------------------------------------------
    # Histograma da distribuicao percentual
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(df["pct_dmn"], bins=30)
    plt.axvline(0.50, linestyle="--", label="threshold 0.50")
    plt.axvline(0.60, linestyle="--", label="threshold 0.60")
    plt.axvline(0.70, linestyle="--", label="threshold 0.70")
    plt.axvline(0.80, linestyle="--", label="threshold 0.80")
    plt.xlabel("Percentual de voxels da ROI dentro da DMN")
    plt.ylabel("Número de ROIs")
    plt.title("Distribuição do percentual de voxels na DMN por ROI do CC200")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_HIST_PATH, dpi=200)
    plt.close()

    print("Histograma salvo em:", PLOT_HIST_PATH)

    # --------------------------------------------------------
    # Curva threshold vs numero de ROIs
    # --------------------------------------------------------
    thresholds = np.linspace(0, 1, 101)
    counts = [(df["pct_dmn"] >= t).sum() for t in thresholds]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, counts)
    plt.axvline(0.50, linestyle="--", label="0.50")
    plt.axvline(0.60, linestyle="--", label="0.60")
    plt.axvline(0.70, linestyle="--", label="0.70")
    plt.axvline(0.80, linestyle="--", label="0.80")
    plt.xlabel("Threshold de percentual de voxels na DMN")
    plt.ylabel("Número de ROIs com pct_dmn >= threshold")
    plt.title("Curva threshold versus número de ROIs classificadas como DMN no CC200")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_CURVE_PATH, dpi=200)
    plt.close()

    print("Curva threshold salva em:", PLOT_CURVE_PATH)


def main():
    print("\nINICIO DO PIPELINE COMPLETO DE EXTRACAO YEO7 + EXTRACAO CC200 + OVERLAP DMN\n")

    make_yeo7_ready()
    best_vol, best_n_rois, _ = inspect_and_extract_cc200()
    df = compute_overlap_and_save()
    make_plots(df)

    print("\n" + "=" * 80)
    print("FINALIZADO")
    print("=" * 80)
    print(f"Volume escolhido do Craddock: {best_vol}")
    print(f"Número de ROIs nesse volume: {best_n_rois}")
    print(f"DMN label usada no Yeo7: {DMN_LABEL}")
    print("\nArquivos principais gerados:")
    print(" -", YEO7_FIXED_PATH)
    print(" -", YEO7_LUT_PATH)
    print(" -", YEO7_DMN_LABEL_PATH)
    print(" -", YEO7_LABEL_COUNTS_PATH)
    print(" -", CC200_PATH)
    print(" -", CC200_VOLUME_INFO_PATH)
    print(" -", FINAL_ROI_OVERLAP_PATH)
    print(" -", PLOT_HIST_PATH)
    print(" -", PLOT_CURVE_PATH)
    print("\nLeitura rápida do resultado final:")
    print(df.sort_values("pct_dmn", ascending=False).head(20))


if __name__ == "__main__":
    main()