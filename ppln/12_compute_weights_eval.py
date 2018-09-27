import os
import os.path as op
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pypet
from collections import OrderedDict
from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit,
                                     cross_val_score)
from pypet.features import compute_regional_features
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                             classification_report, precision_score,
                             recall_score)
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile

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

weight_val = np.arange(1, 10, .2)
weight_val = sorted(np.concatenate((1 / weight_val, weight_val)))

classifiers = OrderedDict()

markers = [x for x in df.columns if 'aal' in x]
X = df[markers].values
y = 1 - (df['Final diagnosis (behav)'] == 'VS').values.astype(np.int)

results = dict()
results['Iteration'] = []
results['Weight Val'] = []
results['Classifier'] = []
results['AUC'] = []
results['Precision'] = []
results['Recall'] = []

sss = StratifiedShuffleSplit(
    n_splits=50, test_size=0.3, random_state=42)

for t_iter, (train, test) in enumerate(sss.split(X, y)):
    for val in weight_val:
        classifiers['SVC_fs20p'] = Pipeline([
                ('scaler', RobustScaler()),
                ('select', SelectPercentile(f_classif, 20.)),
                ('clf', SVC(kernel="linear", C=1,  probability=True,
                            class_weight={0: 1, 1: val}))
            ])
        classifiers['SVC_fs10p'] = Pipeline([
                ('scaler', RobustScaler()),
                ('select', SelectPercentile(f_classif, 10.)),
                ('clf', SVC(kernel="linear", C=1,  probability=True,
                            class_weight={0: 1, 1: val}))
            ])
        classifiers['RF_w'] = Pipeline([
                ('scaler', RobustScaler()),
                ('clf', RandomForestClassifier(
                        max_depth=5, n_estimators=2000, max_features='auto',
                        class_weight={0: 1, 1: val}))
            ])
        classifiers['Dummy'] = Pipeline([
                ('clf', DummyClassifier(
                 strategy="most_frequent", random_state=42))
            ])
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

            results['Weight Val'].append(val)
            results['Iteration'].append(t_iter)
            results['Classifier'].append(clf_name)
            results['AUC'].append(auc_score)
            results['Precision'].append(prec_score)
            results['Recall'].append(rec_score)

df = pd.DataFrame(results)
df.to_csv(
    '../PET_class/scratch/weights_eval.csv')
cw_mcs = np.sum(y)/y.shape[0]
float(np.sum(y)) / y.shape[0]
float(y.shape[0] - np.sum(y)) / y.shape[0]

classif = ['SVC_fs20p', 'SVC_fs10p', 'RF_w']

fig, ax = plt.subplots(1, 1)
paper_rc = {'lines.linewidth': .6, 'lines.markersize': 6}
sns.set_context("paper", rc=paper_rc)
df['Weight Val'] = df['Weight Val'].round(4)
sns.pointplot(x="Weight Val", y="Recall", hue="Classifier", data=df, ax=ax,
                hue_order=classif)
sns.pointplot(x="Weight Val", y="Precision", hue="Classifier", data=df, ax=ax,
                hue_order=classif)
sns.pointplot(x="Weight Val", y="AUC", hue="Classifier", data=df, ax=ax,
                hue_order=classif)

ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
ax.tick_params(axis='x', direction='out', length=3, width=1, grid_color='r',
               labelrotation=90, grid_alpha=0.5)
# # ax.set_aspect('equal', axis='x', adjustable='datalim')
# # ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
# ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%2.2f'))