import numpy as np
import pandas as pd
import os
import os.path as op
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.metrics import (make_scorer, roc_auc_score, precision_score,
                             recall_score)
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

scoring = {'AUC': 'roc_auc',
           'rec': 'recall',
           'prec': 'precision'}

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

scores_f = {'auc': make_scorer(roc_auc_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score)}
sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.3, random_state=23)
scores = dict()
for clf_name, clf in classifiers.items():
    scores = cross_validate(clf, X, y, cv=sss, scoring=scores_f,
                            return_train_score=True)

df = pd.DataFrame(scores)
