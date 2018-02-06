# plot scores
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypet

df = pd.read_csv('boot_1000.csv')

fig_mean, axes = pypet.viz.plot_values(
    df,
    values=['Recall', 'Precision', 'AUC'],
    target='Classifier',
    classes=['SVC_fs_W40_10', 'SVC_fs_W10_26', 'RF_w', 'Dummy'])
plt.show()

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
lower = max(0.0, np.percentile(results, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(results, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (
    alpha*100, lower*100, upper*100))
