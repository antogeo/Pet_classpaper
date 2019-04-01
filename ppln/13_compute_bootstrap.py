import numpy as np
import pandas as pd
import os
import os.path as op
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
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

group = 'Liege'

df = pd.read_csv(op.join(db_path, group, 'group_results_SUV',
                 group + '_db_GM_masks_atlas.csv'))
df = df.query('QC_PASS == True and ML_VALIDATION == False')

classifiers = OrderedDict()

classifiers['SVC_rec'] = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectPercentile(f_classif, 10.)),
        ('clf', SVC(kernel="linear", C=1, probability=True,
                    class_weight={0: 1, 1: 3.6}))
    ])
classifiers['SVC_prec'] = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectPercentile(f_classif, 10.)),
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
X = df[markers].values
y = 1 - (df['Final diagnosis (behav)'] == 'VS').values.astype(np.int)

results = dict()
results['Iteration'] = []
results['Recall'] = []
results['Precision'] = []
results['AUC'] = []
results['Classifier'] = []
feat_sel = []

sss = StratifiedShuffleSplit(
   n_splits=5, test_size=0.3, random_state=30)
for iter, (train_ind, test_ind) in enumerate(sss.split(X, y)):
    for clf_name, clf in classifiers.items():
        clf.fit(X[train_ind], y[train_ind])
        y_pred_class = clf.predict(X[test_ind])
        y_pred_proba = clf.predict_proba(X[test_ind])[:, 0]
        auc_score = roc_auc_score(y[test_ind], y_pred_proba)
        prec_score = precision_score(y[test_ind], y_pred_class)
        rec_score = recall_score(y[test_ind], y_pred_class)

        results['Iteration'].append(iter)
        results['Classifier'].append(clf_name)
        results['AUC'].append(auc_score)
        results['Precision'].append(prec_score)
        results['Recall'].append(rec_score)
        if 'select' in clf.named_steps:
            print('saving features for {} iter {}'.format(clf_name, iter))
            feat_sel.append(
                clf.named_steps['select'].get_support(indices=True))

        # results[clf_name] = cross_validate(
        #   clf, X, y, cv=sss, scoring=scores_f,
        #   return_train_score=True, n_jobs=4)

unique, counts = np.unique(feat_sel, return_counts=True)
dict(zip(unique, counts))

df = pd.DataFrame(results)
