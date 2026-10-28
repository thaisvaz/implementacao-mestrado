#legenda dos dados phenotipicos https://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_LEGEND_V1.02.pdf

# 0 = control; 1= Autism; 2= Asperger's Syndrome or PDD-NOS


# Importar Bibliotecas

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

#%matplotlib inline

import random
random.seed(19)

# Leitura das Bases

from nilearn.datasets import fetch_abide_pcp

# n_subjects

abide = fetch_abide_pcp(pipeline='cpac', derivatives=['rois_cc200'], band_pass_filtering=False, global_signal_regression=False, quality_checked=True)

abide_pheno = pd.DataFrame(abide.phenotypic)


print(abide.rois_cc200[0].shape)

from nilearn.connectome import ConnectivityMeasure

conn_est2 = ConnectivityMeasure(kind='correlation', vectorize=True, discard_diagonal=True)
conn_matrice2 = conn_est2.fit_transform(abide.rois_cc200)

X=conn_matrice2
X.shape

X.shape

abide_pheno['DX_GROUP']=np.where(abide_pheno['DX_GROUP']==1,1,0)

# 1=autism
# 2=control

y=abide_pheno['DX_GROUP']

# Separar dado para validação

from sklearn.model_selection import train_test_split

# Separa 10% para teste
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,stratify=y, random_state=19)

# FUNÇÃO AVALIA MODELO

def avalia(model_name, y_test, y_pred):
    df = pd.DataFrame()
    from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
    
    df.loc['Acurácia', model_name]=accuracy_score(y_test, y_pred)
    df.loc['Recall', model_name]=recall_score(y_test, y_pred)
    df.loc['Precision', model_name]=precision_score(y_test, y_pred)  
    df.loc['F1', model_name]=f1_score(y_test, y_pred)

    df=round(df,4)
    return df


# FUNÇÃO AVALIA ACC

df_metrics_acc = pd.DataFrame()

def avalia_acc(df, model_name, y_test, y_pred):

    from sklearn.metrics import  recall_score, precision_score, f1_score, accuracy_score
    
    df.loc['Acurácia', model_name]=accuracy_score(y_test, y_pred)
    df.loc['Recall', model_name]=recall_score(y_test, y_pred)
    df.loc['Precision', model_name]=precision_score(y_test, y_pred)  
    df.loc['F1', model_name]=f1_score(y_test, y_pred)

    df=round(df,4)
    return df

import seaborn as sns
from sklearn.metrics import confusion_matrix
class_names = ['TEA', 'Controle']

def plot_confusion(cf_matrix, model_name):

    plt.figure(figsize=(9,5))

    group_names = ['VN','FP','FP','VP']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    ax=sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues',annot_kws={"fontsize":15})
    classes=['Controle', 'TEA']
    ax.set_xticklabels(classes, fontsize=12, fontname="Special Elite")
    ax.set_yticklabels(classes, fontsize=12, fontname="Special Elite")

    plt.tight_layout()
    plt.ylabel('Classe Real',fontsize=12,fontname="Special Elite")
    plt.xlabel('Classe Predita',fontsize=12,fontname="Special Elite")

    plt.savefig('drive/My Drive/00.TCC/00_Entrega/matrizes/'+model_name+'.png')

# Leitura das Bases - CC200

from nilearn.datasets import fetch_abide_pcp

abide = fetch_abide_pcp(pipeline='cpac', derivatives=['rois_cc200'], band_pass_filtering=False,global_signal_regression=False, quality_checked=True)

abide_pheno = pd.DataFrame(abide.phenotypic)

abide.rois_cc200[0].shape

# Matriz de Conectividade

from nilearn.connectome import ConnectivityMeasure

conn_est2 = ConnectivityMeasure(kind='correlation', vectorize=True, discard_diagonal=True)
conn_matrice2 = conn_est2.fit_transform(abide.rois_cc200)
X=conn_matrice2

X.shape

# 1=autism
# 2=control --->0
abide_pheno['DX_GROUP']=np.where(abide_pheno['DX_GROUP']==1,1,0)
y=abide_pheno['DX_GROUP']

y.shape

# Split

# Separar dado para teste
from sklearn.model_selection import train_test_split

# Separa 10% para teste
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.1, stratify=y, random_state=19)

abide.rois_cc200[0].shape

abide_pheno.shape

# 1. SVC CC200

from sklearn.svm import LinearSVC

clf_svc200 = LinearSVC(penalty='l2', C=0.1, dual=True, tol=0.0001, max_iter=10000, random_state=19)

# Cross validated

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

scores = cross_validate(clf_svc200, X_train, y_train, cv=5, scoring=['accuracy','recall','precision','f1'])

resultados = pd.DataFrame(scores)

data = {
  "Média" : [resultados['test_accuracy'].mean(),resultados['test_recall'].mean(),resultados['test_precision'].mean(),resultados['test_f1'].mean()],
  "Desvio Padrão" : [resultados['test_accuracy'].std(),resultados['test_recall'].std(),resultados['test_precision'].std(),resultados['test_f1'].std()]
}

df_cv_svc200 = pd.DataFrame(data, index = ['Acurácia','Recall','Precision','F1'])
df_cv_svc200.round(4)

# Modelo Final

clf_svc200.fit(X_train, y_train)

y_pred_train_svc200 = clf_svc200.predict(X_train)
y_pred_test_svc200 = clf_svc200.predict(X_test)

from sklearn.metrics import roc_auc_score, average_precision_score

# score contínuo (margem do SVC)
y_score_test_svc200 = clf_svc200.decision_function(X_test)

auc_svc200 = roc_auc_score(y_test, y_score_test_svc200)
ap_svc200  = average_precision_score(y_test, y_score_test_svc200)

print("ROC-AUC (SVC_200):", round(auc_svc200, 4))
print("Average Precision (SVC_200):", round(ap_svc200, 4))

y_score_train_svc200 = clf_svc200.decision_function(X_train)

print("ROC-AUC train:", round(roc_auc_score(y_train, y_score_train_svc200), 4))
print("AP train:", round(average_precision_score(y_train, y_score_train_svc200), 4))




df_svc200=avalia('SVC_200', y_test, y_pred_test_svc200)
df_svc200

avalia_acc(df_metrics_acc, 'SVC_200', y_test, y_pred_test_svc200)
print(df_metrics_acc.head())


# Matriz de confusão

import matplotlib.pyplot as plt

cnf_matrix1 = confusion_matrix(y_test, y_pred_test_svc200)

print(cnf_matrix1)

# plot_confusion(cnf_matrix1, model_name='SVC_200')



