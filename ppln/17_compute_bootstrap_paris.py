import numpy as np
import pandas as pd
import os
import os.path as op
from sklearn.utils import resample
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
    db_path = '/home/coma_meth/dox/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

group = 'Paris'

df = pd.read_csv(op.join(db_path, 'Liege', 'group_results_SUV',
                 'Liege' + '_db_GM_AAL_nocereb.csv'))
gen_df = pd.read_csv(op.join(db_path, group, 'group_results_SUV',
                 group + '_db_GM_AAL_nocereb.csv'))
df_train = df.query('QC_PASS == True and ML_VALIDATION == False')
df_gen = gen_df.query('QC_PASS == True and ML_gener == True')
classifiers = OrderedDict()

classifiers['SVC_rec'] = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectPercentile(f_classif, 10.)),
        ('clf', SVC(kernel="linear", C=1, probability=True,
                    class_weight={0: 1, 1: 2.4}))
    ])
classifiers['SVC_prec'] = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectPercentile(f_classif, 10.)),
        ('clf', SVC(kernel="linear", C=1,  probability=True,
                    class_weight={0: 1, 1: .55}))
    ])
classifiers['RF'] = Pipeline([
    ('scaler', RobustScaler()),
    ('clf', RandomForestClassifier(
        max_depth=5, n_estimators=2000, max_features='auto',
        class_weight={0: 1, 1: 1}))
])
classifiers['Dummy'] = Pipeline([
        ('clf', DummyClassifier(strategy="stratified"))
    ])

markers = [x for x in df_train.columns if 'aal' in x]
X_train = df_train[markers].values
y_train = 1 - (df_train[
    'Final diagnosis (behav)'] == 'VS').values.astype(np.int)

y_val = 1 - (df_gen[
    'Label'] == 'VS').values.astype(np.int)
sizes = [np.sum(y_val == 0), np.sum(y_val == 1)]

results = dict()
results['Iteration'] = []
results['Recall'] = []
results['Precision'] = []
results['AUC'] = []
results['Classifier'] = []
t_iter = 1000

for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    for i in range(t_iter):
        # prepare train and val sets
        print(i, clf_name)
        df_test_vs = resample(
            df_gen[y_val == 0], n_samples=sizes[0], random_state=i)
        df_test_mcs = resample(
            df_gen[y_val == 1], n_samples=sizes[1], random_state=i)
        df_test = df_test_vs.append(df_test_mcs)
        X_test = df_test[markers].values
        y_test = 1 - (df_test[
            'Final diagnosis (behav)'] == 'VS').values.astype(np.int)
        y_pred_class = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        prec_score = precision_score(y_test, y_pred_class)
        rec_score = recall_score(y_test, y_pred_class)

        results['Iteration'].append(iter)
        results['Classifier'].append(clf_name)
        results['AUC'].append(auc_score)
        results['Precision'].append(prec_score)
        results['Recall'].append(rec_score)
    # if 'select' in clf.named_steps:
    #     print('saving features for {} iter {}'.format(clf_name, iter))
    #     feat_sel.append(
    #         clf.named_steps['select'].get_support(indices=True))
    #     feat_rank.append(dict(zip(markers, -np.log10(
    #         clf.named_steps['select'].pvalues_))))
        # results[clf_name] = cross_validate(
        #   clf, X, y, cv=sss, scoring=scores_f,
        #   return_train_score=True, n_jobs=4)

# unique, counts = np.unique(feat_sel, return_counts=True)
# feats = dict(zip(unique, counts))
# df_feats = pd.DataFrame(feats)
# df_feat = pd.DataFrame(feat_rank)
df_res = pd.DataFrame(results)
# df_feat.to_csv('./group_results_SUV/feat_rank.csv')
df_res.to_csv(op.join(db_path, group, 'group_results_SUV',
    'gen_perf_estim_bs_f10_AAL90.csv'))
# df_feats.to_csv('./group_results_SUV/selected_features.csv')
