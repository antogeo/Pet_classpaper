#!/usr/bin/env      python
import os
import os.path as op
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
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

# results = dict()
# results['repet'] = []
# # results['clf'] = []
# results['feature'] = []
# results['subj'] = []
# results['score'] = []
# results['fold'] = []
# # results['proba'] = []
# results['lbl'] = []
iters = 50
subj_feat = np.zeros((iters, X.shape[1], X.shape[0])) + 2

for iter in range(iters):
    folds = StratifiedKFold(n_splits=5, random_state=iter)
    for train, test in folds.split(X, y):
        for f_num, feature in enumerate(ord_markers):
            X_train, X_test = X[train, f_num], X[test, f_num]
            y_train, y_test = y[train], y[test]
            for clf_name, clf in classifiers.items():
                # Fit the model on the training set
                clf.fit(X_train.reshape(-1, 1), y_train)
                # y_proba = clf.predict_proba(X_test.reshape(-1, 1))[:, 1]
                y_score = clf.predict(X_test.reshape(-1, 1))
                for test_num, test_ind in enumerate(test):
                    # print(test_num)
                    subj_feat[iter, f_num, test_ind] = y_score[test_num]
                # # results['clf'].append(clf_name)
                # results['feature'].append(feature)
                # results['subj'].append(test)
                # results['score'].append(y_score)
                # # results['proba'].append(y_proba[0])
                # results['lbl'].append(y[test][0])
subj_feat2d = np.sum(subj_feat, axis=0)
normed_feat = (subj_feat2d - subj_feat2d.min()) / (
               subj_feat2d.max() - subj_feat2d.min())

df_res = pd.DataFrame(subj_feat2d, index=ord_markers,
                      columns=np.array(range(1, X.shape[0] + 1)))

sim_mat = np.zeros((X.shape[1], X.shape[1]))

for i in range(X.shape[1]):
    for j in range(i, X.shape[1]):
        # print(i, j)
        for x in range(X.shape[0]):
            sim_mat[i, j] = sim_mat[i, j] + np.abs(
                normed_feat[i, x] - normed_feat[j, x])
        sim_mat[j, i] = sim_mat[i, j]

df_sim = pd.DataFrame(sim_mat, index=ord_markers, columns=ord_markers)
sns.heatmap(df_sim, cbar_kws={
    'label': 'pairwise discrepancy in subjects lassification'})
df_sim.to_csv('../group_results_SUV/feat_similarity_mat.csv')
