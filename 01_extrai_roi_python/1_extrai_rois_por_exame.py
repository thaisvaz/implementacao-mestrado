import pandas as pd
from nilearn.datasets import fetch_abide_pcp

#legenda dos dados phenotipicos https://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_LEGEND_V1.02.pdf
# n_subjects
abide = fetch_abide_pcp(pipeline='cpac', derivatives=['rois_cc200'], band_pass_filtering=False, global_signal_regression=False, quality_checked=True)

from google.colab import drive
drive.mount('/content/drive')

for exame in range(871):
  pd.DataFrame(abide.rois_cc200[exame]).to_csv(f'drive/MyDrive/01.Mestrado/00.ProjetoPesquisa/implementacoes/rois_por_exame/{exame}.csv',index=False, header=False)


