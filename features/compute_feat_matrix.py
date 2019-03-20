#!/usr/bin/env      python
import os
import os.path as op
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from scipy.spatial.distance import pdist, squareform

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

meta_fname = pd.read_csv(op.join(db_path, 'Liege', 'group_results_SUV',
                         'Liege' + '_db_GM_masks_atlas.csv'))
#  df = compute_regional_features(db_path, meta_fname)

df = meta_fname.query('QC_PASS == True and ML_VALIDATION == False')

markers = [x for x in df.columns if 'aal' in x]

df = df.reset_index()
X = df[markers].values
y = 1 - (df['Final diagnosis (behav)'] == 'VS').values.astype(np.int)

classifiers = OrderedDict()
classifiers['lr'] = Pipeline([
        ('scaler', RobustScaler()),
        ('clf', LogisticRegression())
    ])

results = dict()
results['clf'] = []
results['feature'] = []
results['subj'] = []
results['score'] = []
results['proba'] = []
results['lbl'] = []
loo = LeaveOneOut()
for train, test in loo.split(X):
    for f_num, feature in enumerate(markers):
        X_train, X_test = X[train, f_num], X[test, f_num]
        y_train, y_test = y[train], y[test]
        y_score = []
        for clf_name, clf in classifiers.items():
            # Fit the model on the training set
            clf.fit(X_train.reshape(-1, 1), y_train)
            y_proba = clf.predict_proba(X_test.reshape(-1, 1))
            y_score = clf.predict(X_test.reshape(-1, 1))
            # TODO: Assign y_pred_class to the X[test] (n_subj values)
            results['clf'].append(clf_name)
            results['feature'].append(feature)
            results['subj'].append(test.astype(int))
            results['score'].append(y_score)
            results['proba'].append(y_proba)
            results['lbl'].append(y[test])


sim_mat = pdist(results['score'], metric='minkowski', p=1)
DF_euclid = pd.DataFrame(squareform(sim_mat), columns=markers, index=markers)
df_res = pd.DataFrame(results)
mat = df_res[['feature', 'score', 'subj']].pivot(
    values='score', index='feature', columns='subj')
df.to_csv('../group_results_SUV/feat_similarity_mat.csv')
