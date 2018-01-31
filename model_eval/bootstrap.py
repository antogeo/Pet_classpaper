import numpy as np
import pandas as pd
import os
import os.path as op
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import f_classif, SelectPercentile
from matplotlib import pyplot
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
X_train = df[markers].values
y_train = 1 - (df['Final diagnosis (behav)'] == 'VS').values.astype(np.int)

# configure bootstrap
t_iter = 1000
n_size = int(len(df_val))
# run bootstrap

results = dict()
results['Iteration'] = []
results['Classifier'] = []
results['AUC'] = []
results['Precision'] = []
results['Recall'] = []

for i in range(t_iter):
    # prepare train and test sets
    df_test = resample(df_val, n_samples=n_size)

    markers = [x for x in df.columns if 'aal' in x]
    X_test = df_test[markers].values
    y_test = 1 - (
        df_test['Final diagnosis (behav)'] == 'VS').values.astype(np.int)
    # fit model
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
        print('Iter {} Clf = {} AUC = {} Prec = {} Rec = {}'.format(
            i, clf_name, auc_score, prec_score, rec_score))


df = pd.DataFrame(results)
df.to_csv('models_eval.csv')

# plot scores
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('boot_1000.csv')

plt.figure(1)
plt.subplot(221)
SVM_AUC = df.loc[df['Classifier'] == 'SVC_fs_W40_10', 'AUC']
plt.hist(SVM_AUC, histtype='stepfilled', align='mid')
SVM_rec = df.loc[df['Classifier'] == 'SVC_fs_W40_10', 'Recall']
plt.hist(SVM_rec, histtype='stepfilled', align='mid')
SVM_prec = df.loc[df['Classifier'] == 'SVC_fs_W40_10', 'Precision']
plt.hist(SVM_prec, histtype='stepfilled', align='mid')
plt.show()

plt.subplot(222)
SVM2_AUC = df.loc[df['Classifier'] == 'SVC_fs_W10_26', 'AUC']
plt.hist(SVM2_AUC, histtype='stepfilled', align='mid')
SVM2_rec = df.loc[df['Classifier'] == 'SVC_fs_W10_26', 'Recall']
plt.hist(SVM2_rec, histtype='stepfilled', align='mid')
SVM2_prec = df.loc[df['Classifier'] == 'SVC_fs_W10_26', 'Precision']
plt.hist(SVM2_prec, histtype='stepfilled', align='mid')
plt.show()

plt.subplot(223)
RF_AUC = df.loc[df['Classifier'] == 'RF_w', 'AUC']
plt.hist(RF_AUC, histtype='stepfilled', align='mid')
RF_rec = df.loc[df['Classifier'] == 'RF_w', 'Recall']
plt.hist(RF_rec, histtype='stepfilled', align='mid')
RF_prec = df.loc[df['Classifier'] == 'RF_w', 'Precision']
plt.hist(RF_prec, histtype='stepfilled', align='mid')
plt.show()

plt.subplot(224)
Dummy_AUC = df.loc[df['Classifier'] == 'Dummy', 'AUC']
plt.hist(Dummy_AUC, histtype='stepfilled', align='mid', color='blue')
Dummy_rec = df.loc[df['Classifier'] == 'Dummy', 'Recall']
plt.hist(Dummy_rec, histtype='stepfilled', align='mid', color='green')
Dummy_prec = df.loc[df['Classifier'] == 'Dummy', 'Precision']
plt.hist(Dummy_prec, histtype='barstacked', align='mid', color='red')
plt.show()

# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, numpy.percentile(results, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, numpy.percentile(results, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
