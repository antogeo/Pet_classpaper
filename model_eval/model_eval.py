import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypet
from collections import OrderedDict
from sklearn.model_selection import StratifiedShuffleSplit
from pypet.features import compute_regional_features
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, precision_score, recall_score)
from sklearn.feature_selection import f_classif, SelectPercentile

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

meta_fname = op.join(db_path, 'extra', 'SUV_database10172017_2.xlsx')
df = compute_regional_features(db_path, meta_fname)
df = df.query('QC_PASS == True and ML_VALIDATION == False')

classifiers = OrderedDict()

markers = [x for x in df.columns if 'aal' in x]
X = df[markers].values
y = 1 - (df['Final diagnosis (behav)'] == 'VS').values.astype(np.int)

results = dict()
results['Iteration'] = []
results['Classifier'] = []
results['AUC'] = []
results['Precision'] = []
results['Recall'] = []

sss = StratifiedShuffleSplit(
    n_splits=500, test_size=0.3, random_state=42)

classifiers['SVC_fs_W40_10'] = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectPercentile(f_classif, 90.)),
        ('clf', SVC(kernel="linear", C=1, probability=True,
                    class_weight={0: 1, 1: .25}))
    ])
classifiers['SVC_fs_W10_26'] = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectPercentile(f_classif, 90.)),
        ('clf', SVC(kernel="linear", C=1,  probability=True,
                    class_weight={0: 1, 1: 2.6}))
    ])
classifiers['RF_w'] = Pipeline([
    ('scaler', RobustScaler()),
    ('clf', RandomForestClassifier(
        max_depth=5, n_estimators=2000, max_features='auto',
        class_weight={0: 1, 1: .7}))
])
classifiers['Dummy'] = Pipeline([
        ('clf', DummyClassifier(strategy="stratified", random_state=42))
    ])

for t_iter, (train, test) in enumerate(sss.split(X, y)):
    print('Iteration {}'.format(t_iter))
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    for clf_name, clf in classifiers.items():
        # Fit the model on the training set
        clf.fit(X_train, y_train)

        # Predict the test set
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred_class = clf.predict(X_test)

        auc_score = roc_auc_score(y_test, y_pred_proba)
        prec_score = precision_score(y_test, y_pred_class)
        rec_score = recall_score(y_test, y_pred_class)

        results['Iteration'].append(t_iter)
        results['Classifier'].append(clf_name)
        results['AUC'].append(auc_score)
        results['Precision'].append(prec_score)
        results['Recall'].append(rec_score)


df = pd.DataFrame(results)
df.to_csv('data/models_eval_90pcfeat.csv')

fig_mean, axes = pypet.viz.plot_values(
    df,
    values=['Recall', 'Precision', 'AUC'],
    target='Classifier',
    classes=['SVC_fs_W40_10', 'SVC_fs_W10_26', 'RF_w', 'Dummy'])
plt.show()
