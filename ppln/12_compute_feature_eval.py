import os
import os.path as op
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.feature_selection import f_classif, SelectPercentile


if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

logging.basicConfig(level='INFO')

group = 'Liege'

df = pd.read_csv(op.join(db_path, group, 'group_results_SUV',
                 group + '_db_GM_masks_atlas.csv'))

df = df.query('QC_PASS == True and ML_VALIDATION == False')

markers = [x for x in df.columns if 'aal' in x]
X = df[markers].values
y = (df['Final diagnosis (behav)'] == 'VS').values.astype(np.int)

results = dict()
results['Iteration'] = []
results['perc'] = []
results['AUC'] = []
results['Precision'] = []
results['Recall'] = []

for perc in range(1, 11, 1):
    sss = StratifiedShuffleSplit(
        n_splits=100, test_size=0.3, random_state=42)
    for t_iter, (train, test) in enumerate(sss.split(X, y)):
        print(perc/10)
        clf = Pipeline([
            ('scaler', RobustScaler()),
            ('select', SelectPercentile(f_classif, perc/10)),
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
        results['perc'].append(perc/10)
        results['AUC'].append(auc_score)
        results['Precision'].append(prec_score)
        results['Recall'].append(rec_score)

        # http://scikit-learn.org/stable/auto_examples/model_selection/
        # plot_precision_recall.html
        prec_curve, rec_curve, thres = precision_recall_curve(
            y_test, y_pred_class)

results_df = pd.DataFrame(results)
results_df.to_csv(op.join(db_path, group, 'group_results_SUV',
                  group + 'feature_eval_SVC.csv'))
