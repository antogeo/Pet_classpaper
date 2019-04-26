import os
import os.path as op
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, precision_score, recall_score)
from sklearn.feature_selection import f_classif, SelectKBest

import seaborn as sns
import matplotlib.pyplot as plt
if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/dox/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

logging.basicConfig(level='INFO')

meta_fname = pd.read_csv(op.join(db_path, 'Liege', 'group_results_SUV',
                         'Liege' + '_db_GM_masks_3_atlases_nAAL.csv'))
#  df = compute_regional_features(db_path, meta_fname)

df = meta_fname.query('QC_PASS == True and ML_VALIDATION == False')
markers = [x for x in df.columns if 'aal' in x]
X = df[markers].values
y = (df['Final diagnosis (behav)'] == 'VS').values.astype(np.int)

results = dict()
results['Iteration'] = []
results['feat_num'] = []
results['AUC'] = []
results['Precision'] = []
results['Recall'] = []
for feat_num in range(1, X.shape[1]):
    print(feat_num)
    sss = StratifiedShuffleSplit(
        n_splits=100, test_size=0.3, random_state=feat_num)
    for t_iter, (train, test) in enumerate(sss.split(X, y)):
        # print(t_iter)
        clf = Pipeline([
            ('scaler', RobustScaler()),
            ('select', SelectKBest(f_classif, feat_num)),
            ('clf', SVC(kernel="linear", C=1,  probability=True))
            ])
        # print('Iteration {}'.format(t_iter))
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        clf.fit(X_train, y_train)
        # Predict the test set
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred_class = clf.predict(X_test)

        auc_score = roc_auc_score(y_test, y_pred_proba)
        prec_score = precision_score(y_test, y_pred_class)
        rec_score = recall_score(y_test, y_pred_class)

        results['Iteration'].append(t_iter)
        results['feat_num'].append(feat_num)
        results['AUC'].append(auc_score)
        results['Precision'].append(prec_score)
        results['Recall'].append(rec_score)

results_df = pd.DataFrame(results)

# metrics = ['AUC', 'Precision', 'Recall']
# colors = ['red', 'green', 'blue']
fig, ax = plt.subplots(1, 1)

sns.lineplot(x="feat_num", y='AUC', data=results_df, color='red')

results_df.to_csv(op.join(db_path, 'Liege', 'group_results_SUV',
                          'Liege' + 'feature_eval_KbestSVC_nAALspace.csv'))
