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
from sklearn.feature_selection import f_classif, SelectPercentile
import seaborn as sns
import matplotlib.pyplot as plt
if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

logging.basicConfig(level='INFO')

meta_fname = pd.read_csv(op.join(db_path, 'Liege', 'group_results_SUV',
                         'Liege' + '_db_GM_masks_atlas.csv'))
#  df = compute_regional_features(db_path, meta_fname)

df = meta_fname.query('QC_PASS == True and ML_VALIDATION == False')

markers = [x for x in df.columns if 'aal' in x]
X = df[markers].values
y = (df['Final diagnosis (behav)'] == 'VS').values.astype(np.int)

results = dict()
results['Iteration'] = []
results['perc'] = []
results['AUC'] = []
results['Precision'] = []
results['Recall'] = []
percs = np.arange(.05, 1.05, .05)
for perc in percs:
    print(perc)
    sss = StratifiedShuffleSplit(
        n_splits=100, test_size=0.3, random_state=42)
    for t_iter, (train, test) in enumerate(sss.split(X, y)):
        print(t_iter)
        clf = Pipeline([
            ('scaler', RobustScaler()),
            ('select', SelectPercentile(f_classif, perc)),
            ('clf', SVC(kernel="linear", C=1,  probability=True))
            ])
        print('Iteration {}'.format(t_iter))
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
        results['perc'].append(perc)
        results['AUC'].append(auc_score)
        results['Precision'].append(prec_score)
        results['Recall'].append(rec_score)

results['perc'] = np.round(results['perc'], decimals=2)
results_df = pd.DataFrame(results)

metrics = ['AUC', 'Precision', 'Recall']
colors = ['red', 'green', 'blue']
fig, ax = plt.subplots(1, 1)

for metric, color in zip(metrics, colors):
    sns.pointplot(x="perc", y=metric, data=results_df,  estimator=np.mean,
                  color=color)

results_df.to_csv(op.join(db_path, 'Liege', 'group_results_SUV',
                          'Liege' + 'feature_eval_01SVC.csv'))
