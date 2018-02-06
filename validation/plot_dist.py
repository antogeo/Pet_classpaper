import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import f_classif, SelectPercentile
from pypet.features import compute_regional_features


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
p_values = clf.named_steps['select'].pvalues_
val_shorted = np.argsort(p_values)
for rank, ind in enumerate(val_shorted):
    fig, axes = plt.subplots(2, 1, sharex=True)
    plot = sns.distplot(X_train[y_train == 0, ind], ax=axes[0])
    plot = sns.distplot(X_train[y_train == 1, ind], ax=axes[0])
    plot = sns.distplot(X_val[y_val == 0, ind], ax=axes[1])
    plot = sns.distplot(X_val[y_val == 1, ind], ax=axes[1])
    plt.savefig(rank + "_" + df_train[markers].columns[ind] + ".pdf")
    plt.close(fig)
