import pandas as pd
import matplotlib.pyplot as plt
import pypet
import seaborn as sns
import matplotlib.gridspec as gridspec

df = pd.read_csv('group_results_SUV/performance_estimate_1000iter_nAAL.csv',
                 index_col=['Iteration', 'Classifier'])[
                    ['AUC', 'Precision', 'Recall']]

df = df.reset_index()

contr_df = pd.read_csv('group_results_SUV/Clfs_contrast_nAAL.csv')
diff_df = contr_df.loc[contr_df['Contrast'].str.contains("-Dummy")]

plt.figure(1)
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

ax1 = plt.subplot(gs[0])

pypet.viz.plot_values(df, values='AUC',
                      target='Classifier', axes=ax1,
                      classes=['RF', 'SVC_prec', 'SVC_rec', 'Dummy'],
                      style='violinplot')
sum((diff_df[diff_df['Contrast'] == 'RF-Dummy']['AUC'] < 0) * 1)

ax2 = plt.subplot(gs[1])

sns.boxplot(data=diff_df, x='AUC', y='Contrast', ax=ax2)
print('{0} times that SVC_rec-Dummy AUC was below 0'.format(
    sum((diff_df[diff_df['Contrast'] == 'SVC_rec-Dummy']['AUC'] < 0) * 1)))
print('{0} times that SVC_prec-Dummy AUC was below 0'.format(
    sum((diff_df[diff_df['Contrast'] == 'SVC_prec-Dummy']['AUC'] < 0) * 1)))
print('{0} times that RF-Dummy AUC was below 0 '.format(
    sum((diff_df[diff_df['Contrast'] == 'RF-Dummy']['AUC'] < 0) * 1)))
plt.figure(2)
gs2 = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

ax21 = plt.subplot(gs2[0])

pypet.viz.plot_values(df, values='Recall',
                      target='Classifier', axes=ax21,
                      classes=['RF', 'SVC_prec', 'SVC_rec', 'Dummy'],
                      style='violinplot')

ax22 = plt.subplot(gs2[1])
sns.boxplot(data=diff_df, x='Recall', y='Contrast', ax=ax22)

print('{0} times that SVC_rec-Dummy Recall was below 0'.format(
    sum((diff_df[diff_df['Contrast'] == 'SVC_rec-Dummy']['Recall'] < 0) * 1)))
print('{0} times that SVC_prec-Dummy Recall was below 0'.format(
    sum((diff_df[diff_df['Contrast'] == 'SVC_prec-Dummy']['Recall'] < 0) * 1)))
print('{0} times that RF-Dummy Recall was below 0 '.format(
    sum((diff_df[diff_df['Contrast'] == 'RF-Dummy']['Recall'] < 0) * 1)))

plt.figure(3)
gs3 = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

ax31 = plt.subplot(gs3[0])

pypet.viz.plot_values(df, values='Precision',
                      target='Classifier', axes=ax31,
                      classes=['RF', 'SVC_prec', 'SVC_rec', 'Dummy'],
                      style='violinplot')

ax32 = plt.subplot(gs3[1])
sns.boxplot(data=diff_df, x='Precision', y='Contrast', ax=ax32)

print('{0} times that SVC_rec-Dummy Precision was below 0'.format(
    sum((diff_df[diff_df['Contrast'] == 'SVC_rec-Dummy']['Precision'] < 0) * 1)))
print('{0} times that SVC_prec-Dummy Precision was below 0'.format(
    sum((diff_df[diff_df['Contrast'] == 'SVC_prec-Dummy']['Precision'] < 0) * 1)))
print('{0} times that RF-Dummy Precision was below 0 '.format(
    sum((diff_df[diff_df['Contrast'] == 'RF-Dummy']['Precision'] < 0) * 1)))
plt.show()
