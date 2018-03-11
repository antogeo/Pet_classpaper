# plot scores
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
            Diff['Prec'].extend(
                df1['Precision'].values - df2['Precision'].values)
            Diff['Rec'].extend(df1['Recall'].values - df2['Recall'].values)
            contrast = [contr] * len(df1)
            Diff['Contrast'].extend(contrast)

res = pd.DataFrame(Diff)

# res.to_csv('DummyVSclfs.csv')

gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

ax1 = plt.subplot(gs[0])
pypet.viz.plot_values(
    df, values='AUC', target='Classifier', axes=ax1,
    classes=['SVC_fs_W40_10', 'SVC_fs_W10_26', 'RF_w', 'Dummy'])

ax2 = plt.subplot(gs[1])
sns.boxplot(data=res, x='AUC', y='Contrast', ax=ax2)
#
#
# figure(2)
# Rec_plot = sns.boxplot(data=res,
#     x = 'Rec',
#     y = 'Contrast')
# figure(3)
# Prec_plot = sns.boxplot(data=res,
#     x = 'Prec',
#     y = 'Contrast')
