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
import seaborn as sns
import re


def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    From
    https://arcpy.wordpress.com/2012/05/11/
        sorting-alphanumeric-strings-in-python/
    and http://stackoverflow.com/questions/
        2669059/how-to-sort-alpha-numeric-set-in-python
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


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
ord_markers = sorted_nicely(markers)

df = df.reset_index()
X = df[ord_markers].values
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
    for f_num, feature in enumerate(ord_markers):
        X_train, X_test = X[train, f_num], X[test, f_num]
        y_train, y_test = y[train], y[test]
        y_score = []
        for clf_name, clf in classifiers.items():
            # Fit the model on the training set
            clf.fit(X_train.reshape(-1, 1), y_train)
            y_proba = clf.predict_proba(X_test.reshape(-1, 1))[:, 1]
            y_score = clf.predict(X_test.reshape(-1, 1))
            # TODO: Assign y_pred_class to the X[test] (n_subj values)
            results['clf'].append(clf_name)
            results['feature'].append(feature)
            results['subj'].append(test[0])
            results['score'].append(y_score[0])
            results['proba'].append(y_proba[0])
            results['lbl'].append(y[test][0])

df_res = pd.DataFrame(results)
mat = df_res.pivot(
    values='score', index='feature', columns='subj')
dis_mat = pdist(mat, metric='minkowski', p=1)
sq_sim = pd.DataFrame((max(dis_mat) - squareform(dis_mat)),
                      columns=ord_markers, index=ord_markers)

sns.heatmap(sq_sim)
df.to_csv('../group_results_SUV/feat_similarity_mat.csv')
