import os
import os.path as op
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pypet
from collections import OrderedDict
from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit,
                                     cross_val_score)
from pypet.features import compute_regional_features
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (roc_auc_score, precision_score, recall_score)
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

meta_fname = op.join(db_path, 'extra', 'SUV_database10172017_2.xlsx')

df = compute_regional_features(db_path, meta_fname)
markers = [x for x in df.columns if 'aal' in x]
df_train = df.query('QC_PASS == True and ML_VALIDATION == False')
df_val = df.query('QC_PASS == True and ML_VALIDATION == True')

X_train = df_train[markers].values
y_train = 1 - (
    df_train['Final diagnosis (behav)'] == 'VS').values.astype(np.int)

X_val = df_val[markers].values
y_val = 1 - (
    df_val['Final diagnosis (behav)'] == 'VS').values.astype(np.int)

sizes = [np.sum(y_val == 0), np.sum(y_val == 1)]

clf = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectPercentile(f_classif, 10.)),
    ])

clf.fit(X_train, y_train)
p_values  = clf.named_steps['select'].pvalues_
val_shorted = np.argsort(p_values)
for ind in val_shorted:
    f, axes = plt.subplots(2, 1, sharey=True)
    plot = sns.distplot(X_train[y_train == 0, ind], ax=axes[0])
    plot = sns.distplot(X_train[y_train == 1, ind], ax=axes[0])
    plot = sns.distplot(X_val[y_val == 0, ind], ax=axes[1])
    plot = sns.distplot(X_val[y_val == 1, ind], ax=axes[1])
    plt.savefig("test.pdf")
    plt.savefig(plot.fig)
plt.close(fig)
