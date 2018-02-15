import numpy as np
import pandas as pd
import os
import os.path as op
import seaborn as sns
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import f_classif, SelectPercentile
import matplotlib.pyplot as plt
from collections import OrderedDict

from pypet.features import compute_regional_features

# load dataset
if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

meta_fname = op.join(db_path, 'extra', 'SUV_database10172017_2.xlsx')
df = compute_regional_features(db_path, meta_fname)
df_train = df.query('QC_PASS == True and ML_VALIDATION == False')
df_val = df.query('QC_PASS == True and ML_VALIDATION == True')

classifiers = OrderedDict()

classifiers['SVC_fs_W40_10'] = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectPercentile(f_classif, 10.)),
        ('clf', SVC(kernel="linear", C=1, probability=True,
                    class_weight={0: 1, 1: .25}))
    ])
classifiers['SVC_fs_W10_26'] = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectPercentile(f_classif, 10.)),
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

markers = [x for x in df.columns if 'aal' in x]
X_train = df_train[markers].values
y_train = 1 - (df_train['Final diagnosis (behav)'] == 'VS').values.astype(np.int)

y_val = 1 - (df_val['Final diagnosis (behav)'] == 'VS').values.astype(np.int)
sizes = [np.sum(y_val == 0), np.sum(y_val == 1)]

# run bootstrap

results = dict()
results['Patient'] = []
results['SVC_fs_W40_10'] = []
results['SVC_fs_W10_26'] = []
results['RF_w'] = []
results['Dummy'] = []

for clf_name, clf in classifiers.items():
    # Fit the model on the training set\
    print(clf_name, clf)
    clf.fit(X_train, y_train)
    clf.predict(X_test)
