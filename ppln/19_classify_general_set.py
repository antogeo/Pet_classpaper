import numpy as np
import pandas as pd
import os
import os.path as op
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import f_classif, SelectKBest
from collections import OrderedDict


# load dataset
if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/dox/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

group = 'Paris'

df = pd.read_csv(op.join(db_path, 'Liege', 'group_results_SUV',
                 'Liege' + '_db_GM_AAL.csv'))
gen_df = pd.read_csv(op.join(db_path, group, 'group_results_SUV',
                 group + '_db_GM_AAL_nocereb.csv'))
df_train = df.query('QC_PASS == True and ML_VALIDATION == False')
df_test = gen_df.query('QC_PASS == True and ML_gener == True')
classifiers = OrderedDict()

classifiers['SVC_rec'] = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectKBest(f_classif, 10)),
        ('clf', SVC(kernel="linear", C=1, probability=True,
                    class_weight={0: 1, 1: 2.4}))
    ])
classifiers['SVC_prec'] = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectKBest(f_classif, 10)),
        ('clf', SVC(kernel="linear", C=1,  probability=True,
                    class_weight={0: 1, 1: .55}))
    ])
classifiers['RF'] = Pipeline([
    ('scaler', RobustScaler()),
    ('clf', RandomForestClassifier(
        max_depth=5, n_estimators=2000, max_features='auto',
        class_weight={0: 1, 1: 1}))
])

markers = [x for x in df_train.columns if 'aal' in x]
X_train = df_train[markers].values
y_train = 1 - (df_train[
    'Final diagnosis (behav)'] == 'VS').values.astype(np.int)

X_test = df_test[markers].values
y_test = 1 - (df_test[
    'Final diagnosis (behav)'] == 'VS').values.astype(np.int)

results = dict()
results['Classifier'] = []
results['Subject'] = []
results['Proba'] = []
results['Score'] = []
results['Label'] = []
len(df_test)
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    for i in range(len(df_test)):
        y_pred_class = clf.predict(
            df_test.iloc[i][markers].values.reshape(1, -1))[0]
        y_pred_proba = clf.predict_proba(
            df_test.iloc[i][markers].values.reshape(1, -1))[:, 1]
        results['Classifier'].append(clf_name)
        results['Subject'].append(df_test.iloc[i]['Code'])
        results['Proba'].append(y_pred_proba)
        results['Score'].append(y_pred_class)
        results['Label'].append(1 - (df_test.iloc[i][
            'Final diagnosis (behav)'] == 'VS'))

df_res = pd.DataFrame(results)
df = df_res.pivot(columns='Classifier', index='Subject')
df.to_csv(op.join(db_path, group, 'group_results_SUV',
                  group + 'gener_res_f10_AAL90_noctrl.csv'))
