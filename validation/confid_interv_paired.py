# plot scores
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypet
import seaborn as sns
import matplotlib.gridspec as gridspec

df = pd.read_csv('../corrected_boot_1000.csv')
Diff = dict()
Diff['Contrast'] = []
Diff['AUC'] = []
Diff['Prec'] = []
Diff['Rec'] = []

for clf1 in np.unique(df['Classifier']):
    for clf2 in np.unique(df['Classifier']):
        if clf1 != clf2 and clf1 != 'Dummy':
            df1 = df[df['Classifier'] == clf1]
            df2 = df[df['Classifier'] == clf2]
            contr = clf1 + '-' + clf2
            Diff['AUC'].extend(df1['AUC'].values - df2['AUC'].values)
            Diff['Prec'].extend(df1['Precision'].values - df2['Precision'].values)
            Diff['Rec'].extend(df1['Recall'].values - df2['Recall'].values)
            contrast = [contr] * len(df1)
            Diff['Contrast'].extend(contrast)

res = pd.DataFrame(Diff)

# res.to_csv('DummyVSclfs.csv')

plt.figure(1)
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

ax1 = plt.subplot(gs[0])

pypet.viz.plot_values(
    df,
    values = 'AUC',
    target = 'Classifier',
    axes = ax1,
    classes = ['SVC_fs_W40_10', 'SVC_fs_W10_26', 'RF_w', 'Dummy'],
    style='violinplot')

ax2 = plt.subplot(gs[1])
sns.boxplot(data=res,
    x = 'AUC',
    y = 'Contrast',
    ax=ax2)

plt.figure(2)
gs2 = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

ax21 = plt.subplot(gs2[0])

pypet.viz.plot_values(
    df,
    values = 'Recall',
    target = 'Classifier',
    axes = ax21,
    classes = ['SVC_fs_W40_10', 'SVC_fs_W10_26', 'RF_w', 'Dummy'],
    style='violinplot')

ax22 = plt.subplot(gs2[1])
sns.boxplot(data=res,
    x = 'Rec',
    y = 'Contrast',
    ax=ax22)

plt.figure(3)
gs3 = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

ax31 = plt.subplot(gs3[0])

pypet.viz.plot_values(
    df,
    values = 'Precision',
    target = 'Classifier',
    axes = ax31,
    classes = ['SVC_fs_W40_10', 'SVC_fs_W10_26', 'RF_w', 'Dummy'],
    style='violinplot')

ax32 = plt.subplot(gs3[1])
sns.boxplot(data=res,
    x = 'Prec',
    y = 'Contrast',
    ax=ax32)

plt.show()
