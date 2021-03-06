import numpy as np
import pandas as pd
import os
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import f_classif, SelectPercentile
from collections import OrderedDict

# load dataset
if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

df = pd.read_csv('group_results_SUV/Liege_db_GM_masks_atlas.csv')
df_train = df.query('QC_PASS == True and ML_VALIDATION == False')
df_val = df.query('QC_PASS == True and ML_VALIDATION == True')

classifiers = OrderedDict()

classifiers['SVC_rec'] = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectPercentile(f_classif, 10.)),
        ('clf', SVC(kernel="linear", C=1, probability=True,
                    class_weight={0: 1, 1: 3.6}))
    ])
classifiers['SVC_prec'] = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectPercentile(f_classif,10.)),
        ('clf', SVC(kernel="linear", C=1,  probability=True,
                    class_weight={0: 1, 1: .46}))
    ])
classifiers['RF'] = Pipeline([
    ('scaler', RobustScaler()),
    ('clf', RandomForestClassifier(
        max_depth=10, n_estimators=2000, max_features='auto',
        class_weight={0: 1, 1: .95}))
])
classifiers['Dummy'] = Pipeline([
        ('clf', DummyClassifier(strategy="stratified"))
    ])

markers = [x for x in df.columns if 'aal' in x]
X_train = df_train[markers].values
y_train = 1 - (
    df_train['Final diagnosis (behav)'] == 'VS').values.astype(np.int)

# configure bootstrap
t_iter = 1000

y_val = 1 - (df_val['Final diagnosis (behav)'] == 'VS').values.astype(np.int)
sizes = [np.sum(y_val == 0), np.sum(y_val == 1)]

# run bootstrap

results = dict()
results['Iteration'] = []
results['Classifier'] = []
results['AUC'] = []
results['Precision'] = []
results['Recall'] = []

predictions = dict()
predictions['Patient'] = []
predictions['Classifier'] = []
predictions['Label'] = []
#
# for clf_name, clf in classifiers.items():
#     # Fit the model on the training set\
#     print(clf_name, clf)
#     clf.fit(X_train, y_train)
#     probas = clf.predict_proba(X_train)[:, 1]
#     auc_score = roc_auc_score(y_train, probas)
#     print('In sample AUC: {}'.format(auc_score))


for i in range(t_iter):
    # prepare train and test sets
    df_test_vs = resample(
        df_val[y_val == 0], n_samples=sizes[0], random_state=i)
    df_test_mcs = resample(
        df_val[y_val == 1], n_samples=sizes[1], random_state=i)

    df_test = df_test_vs.append(df_test_mcs)

    X_test = df_test[markers].values
    y_test = 1 - (
        df_test['Final diagnosis (behav)'] == 'VS').values.astype(np.int)
    # fit model

    for clf_name, clf in classifiers.items():
        # Predict the test set
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred_class = clf.predict(X_test)

        auc_score = roc_auc_score(y_test, y_pred_proba)
        prec_score = precision_score(y_test, y_pred_class)
        rec_score = recall_score(y_test, y_pred_class)

        results['Iteration'].append(i)
        results['Classifier'].append(clf_name)
        results['AUC'].append(auc_score)
        results['Precision'].append(prec_score)
        results['Recall'].append(rec_score)
        print('Iter {} Clf = {} AUC = {} Prec = {} Rec = {}'.format(
            i, clf_name, auc_score, prec_score, rec_score))


df = pd.DataFrame(results)
df.to_csv('group_results_SUV/svc10_rf_boot1000_0328.csv')
