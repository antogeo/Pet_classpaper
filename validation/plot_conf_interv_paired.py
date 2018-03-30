# plot scores
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypet
import seaborn as sns
import matplotlib.gridspec as gridspec

df = pd.read_csv('validation/data/corrected_boot_1000.csv',
                 index_col=['Iteration', 'Classifier'])[
                    ['AUC', 'Precision', 'Recall']]

df = df.reset_index()

contr_df = pd.read_csv('validation/data/Clfs_contrasts.csv')
diff_df = contr_df.loc[~contr_df['Contrast'].str.contains("Dummy-")]

plt.figure(1)
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

ax1 = plt.subplot(gs[0])

pypet.viz.plot_values(
    df,
    values='AUC',
    target='Classifier',
    axes=ax1,
    classes=['SVC_fs_W40_10', 'SVC_fs_W10_26', 'RF_w', 'Dummy'],
    style='violinplot')

ax2 = plt.subplot(gs[1])
sns.boxplot(data=diff_df,
            x='AUC',
            y='Contrast',
            ax=ax2)

plt.figure(2)
gs2 = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

ax21 = plt.subplot(gs2[0])

pypet.viz.plot_values(
    df,
    values='Recall',
    target='Classifier',
    axes=ax21,
    classes=['SVC_fs_W40_10', 'SVC_fs_W10_26', 'RF_w', 'Dummy'],
    style='violinplot')

ax22 = plt.subplot(gs2[1])
sns.boxplot(data=diff_df,
            x='Recall',
            y='Contrast',
            ax=ax22)

plt.figure(3)
gs3 = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

ax31 = plt.subplot(gs3[0])

pypet.viz.plot_values(
    df,
    values='Precision',
    target='Classifier',
    axes=ax31,
    classes=['SVC_fs_W40_10', 'SVC_fs_W10_26', 'RF_w', 'Dummy'],
    style='violinplot')

ax32 = plt.subplot(gs3[1])
sns.boxplot(data=diff_df,
            x='Precision',
            y='Contrast',
            ax=ax32)

plt.show()
