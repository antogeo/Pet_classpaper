import os
import os.path as op
import numpy as np
import pandas as pd
import pypet
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold
from pypet.features import compute_regional_features
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

meta_fname = op.join(db_path, 'extra', 'SUV_database10172017_2.xlsx')
df = compute_regional_features(db_path, meta_fname)
df = df.query('QC_PASS == True and ML_VALIDATION == False')


fcols = [x for x in df.columns if 'aal' in x]
markers = sorted(fcols, key=lambda x: int(x.replace('aal_atlas_', '')))

X = df.query('QC_PASS == True and ML_VALIDATION == False')
df = df.reset_index()
X = df[markers].values
y = 1 - (df['Final diagnosis (behav)'] == 'VS').values.astype(np.int)

# TODO: select classifiers

classifiers = OrderedDict()
classifiers['SVC'] = Pipeline([
        ('scaler', RobustScaler()),
        ('clf', SVC(kernel="linear", C=1))
    ])

iter = 2
results = np.zeros((len(df), len(markers), iter))

for t_iter in range(iter):
    skf = StratifiedKFold(n_splits=5, random_state=t_iter)
    for train, test in skf.split(X, y):
        for f_num, feature in enumerate(markers):
            X_train, X_test = [X[train, f_num]], [X[test, f_num]]
            y_train, y_test = y[train], y[test]
            y_score = []
            for clf_name, clf in classifiers.items():
                # Fit the model on the training set
                clf.fit(X_train, y_train)
                y_pred_class = clf.predict(X_test)
                y_score.append(y_pred_class)
                # TODO: Assign y_pred_class to the X[test] (n_subj values)

            results[f_num, n_subj, t_iter] = np.round(np.mean(y_score))


df = pd.DataFrame(results)
df.to_csv('data/models_eval_90pcfeat.csv')
