# plot scores
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypet

df = pd.read_csv('corrected_boot_1000.csv')
fig_mean, axes = pypet.viz.plot_values(
    df,
    values=['Recall', 'Precision', 'AUC'],
    target='Classifier',
    classes=['SVC_fs_W40_10', 'SVC_fs_W10_26', 'RF_w', 'Dummy'])
plt.show()

SVM_AUC = df.loc[df['Classifier'] == 'SVC_fs_W40_10', 'AUC']
SVM_rec = df.loc[df['Classifier'] == 'SVC_fs_W40_10', 'Recall']
SVM_prec = df.loc[df['Classifier'] == 'SVC_fs_W40_10', 'Precision']
SVM2_AUC = df.loc[df['Classifier'] == 'SVC_fs_W10_26', 'AUC']
SVM2_rec = df.loc[df['Classifier'] == 'SVC_fs_W10_26', 'Recall']
SVM2_prec = df.loc[df['Classifier'] == 'SVC_fs_W10_26', 'Precision']
RF_AUC = df.loc[df['Classifier'] == 'RF_w', 'AUC']
RF_rec = df.loc[df['Classifier'] == 'RF_w', 'Recall']
RF_prec = df.loc[df['Classifier'] == 'RF_w', 'Precision']
Dummy_AUC = df.loc[df['Classifier'] == 'Dummy', 'AUC']
Dummy_rec = df.loc[df['Classifier'] == 'Dummy', 'Recall']
Dummy_prec = df.loc[df['Classifier'] == 'Dummy', 'Precision']

plt.figure(1)
plt.subplot(221)
plt.hist(SVM_AUC, histtype='stepfilled', align='mid', color='blue', bins=50)
plt.hist(SVM_rec, histtype='stepfilled', align='mid', color='green', bins=50)
plt.hist(SVM_prec, histtype='stepfilled', align='mid', color='red', bins=50)
plt.show()


plt.subplot(222)
plt.hist(SVM2_AUC, histtype='stepfilled', align='mid', color='blue', bins=50)
plt.hist(SVM2_rec, histtype='stepfilled', align='mid', color='green', bins=50)
plt.hist(SVM2_prec, histtype='stepfilled', align='mid', color='red', bins=50)
plt.show()


plt.subplot(223)
plt.hist(RF_AUC, histtype='stepfilled', align='mid', color='blue', bins=50)
plt.hist(RF_rec, histtype='stepfilled', align='mid', color='green', bins=50)
plt.hist(RF_prec, histtype='stepfilled', align='mid', color='red', bins=50)
plt.show()

plt.subplot(224)
plt.hist(Dummy_AUC, histtype='stepfilled', align='mid', color='blue', bins=50)
plt.hist(Dummy_rec, histtype='stepfilled', align='mid', color='green', bins=50)
plt.hist(Dummy_prec, histtype='barstacked', align='mid', color='red', bins=50)
plt.show()

results = dict()
results['Classifier'] = []
results['Mean AUC'] = []
results['Mean Precision'] = []
results['Mean Recall'] = []
for clf in np.unique(df['Classifier']):
    results['Classifier'].append(clf)
    results['Mean AUC'].append(
        df.loc[df['Classifier'] == clf, 'AUC'].mean())
    results['Mean Recall'].append(
        df.loc[df['Classifier'] == clf, 'Recall'].mean())
    results['Mean Precision'].append(
        df.loc[df['Classifier'] == clf, 'Precision'].mean())
    alpha = 0.95
    p_up = ((1.0-alpha)/2.0) * 100
    p_low = (alpha+((1.0-alpha)/2.0)) * 100

    lower_AUC = max(0.0, np.percentile(
        df.loc[df['Classifier'] == clf, 'AUC'], p_up))
    upper_AUC = min(1.0, np.percentile(
        df.loc[df['Classifier'] == clf, 'AUC'], p_low))
    print('%.1f confidence interval of AUC for %s:  %.1f%% and %.1f%%' % (
        alpha*100, clf, lower_AUC*100, upper_AUC*100))

    lower_Rec = max(0.0, np.percentile(
        df.loc[df['Classifier'] == clf, 'Recall'], p_up))
    upper_Rec = min(1.0, np.percentile(
        df.loc[df['Classifier'] == clf, 'Recall'], p_low))
    print('%.1f confidence interval of Recall for %s:  %.1f%% and %.1f%%' % (
        alpha*100, clf, lower_Rec*100, upper_Rec*100))

    lower_Prec = max(0.0, np.percentile(
        df.loc[df['Classifier'] == clf, 'Precision'], p_up))
    upper_Prec = min(1.0, np.percentile(
        df.loc[df['Classifier'] == clf, 'Precision'], p_low))
    print('%.1f confidence interval of Precision for %s:  %.1f%% and %.1f%%' % (
        alpha*100, clf, lower_Prec*100, upper_Prec*100))

print('AUC mean of Dummy:  %.1f%%' % (
    results['Mean AUC']['Classifier'== 'Dummy']))
print('Recall mean of Dummy:  %.1f%%' % (
    results['Mean Recall']['Classifier'== 'Dummy']))
print('Prec mean of Dummy:  %.1f%%' % (
     results['Mean Precision']['Classifier'== 'Dummy']))
