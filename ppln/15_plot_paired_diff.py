import pandas as pd
import matplotlib.pyplot as plt
import pypet
import seaborn as sns
import matplotlib.gridspec as gridspec

df = pd.read_csv('group_results_SUV/perf_estim_1000iter_f10_AAL90.csv',
                 index_col=['Iteration', 'Classifier'])[
                    ['AUC', 'Precision', 'Recall']]

df = df.reset_index()

contr_df = pd.read_csv('group_results_SUV/Clfs_contrast_f10_AAL90.csv')
diff_df = contr_df.loc[contr_df['Contrast'].str.contains("-Dummy")]

rf_auc = df[df['Classifier'] == 'RF']['AUC'].mean()
rf_rec = df[df['Classifier'] == 'RF']['Recall'].mean()
rf_prec = df[df['Classifier'] == 'RF']['Precision'].mean()
svc_rec_auc = df[df['Classifier'] == 'SVC_rec']['AUC'].mean()
svc_rec_rec = df[df['Classifier'] == 'SVC_rec']['Recall'].mean()
svc_rec_prec = df[df['Classifier'] == 'SVC_rec']['Precision'].mean()
svc_prec_auc = df[df['Classifier'] == 'SVC_prec']['AUC'].mean()
svc_prec_rec = df[df['Classifier'] == 'SVC_prec']['Recall'].mean()
svc_prec_prec = df[df['Classifier'] == 'SVC_prec']['Precision'].mean()
Dum_auc = df[df['Classifier'] == 'Dummy']['AUC'].mean()
Dum_rec = df[df['Classifier'] == 'Dummy']['Recall'].mean()
Dum_prec = df[df['Classifier'] == 'Dummy']['Precision'].mean()


plt.figure(1)
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

ax1 = plt.subplot(gs[0])

pypet.viz.plot_values(df, values='AUC',
                      target='Classifier', axes=ax1,
                      classes=['RF', 'SVC_prec', 'SVC_rec', 'Dummy'],
                      style='violinplot')
sum((diff_df[diff_df['Contrast'] == 'RF-Dummy']['AUC'] < 0) * 1)

yposlist = df.groupby(['Classifier']).mean()
xposlist = range(len(yposlist))
stringlist = [rf_auc, svc_prec_auc, svc_rec_auc, Dum_auc]
for i in range(len(stringlist)):
    ax1.text(xposlist[i] + .1,
             stringlist[i] + .1, format(round(stringlist[i], 2)))

ax1.axhline(.5, linestyle='--', color='red', alpha=0.4)
ax2 = plt.subplot(gs[1])

sns.boxplot(data=diff_df, x='AUC', y='Contrast', ax=ax2)
print('{0} times that SVC_rec-Dummy AUC was below 0'.format(
    sum((diff_df[diff_df['Contrast'] == 'SVC_rec-Dummy'][
        'AUC'] < 0) * 1)))
print('{0} times that SVC_prec-Dummy AUC was below 0'.format(
    sum((diff_df[diff_df['Contrast'] == 'SVC_prec-Dummy'][
        'AUC'] < 0) * 1)))
print('{0} times that RF-Dummy AUC was below 0 '.format(
    sum((diff_df[diff_df['Contrast'] == 'RF-Dummy'][
        'AUC'] < 0) * 1)))

ax2.axvline(0, linestyle='--', color='red', alpha=0.4)
# ==================================================================
plt.figure(2)
gs2 = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

ax21 = plt.subplot(gs2[0])

pypet.viz.plot_values(df, values='Recall',
                      target='Classifier', axes=ax21,
                      classes=['RF', 'SVC_prec', 'SVC_rec', 'Dummy'],
                      style='violinplot')

yposlist = df.groupby(['Classifier']).mean()
xposlist = range(len(yposlist))
stringlist = [rf_rec, svc_prec_rec, svc_rec_rec, Dum_rec]
for i in range(len(stringlist)):
    ax21.text(xposlist[i] + .1,
              stringlist[i]+ .1, format(round(stringlist[i], 2)))
ax21.axhline(0.5, linestyle='--', color='red', alpha=0.4)
ax22 = plt.subplot(gs2[1])
sns.boxplot(data=diff_df, x='Recall', y='Contrast', ax=ax22)
ax22.axvline(0, linestyle='--', color='red', alpha=0.4)
print('{0} times that SVC_rec-Dummy Recall was below 0'.format(
    sum((diff_df[diff_df['Contrast'] == 'SVC_rec-Dummy'][
        'Recall'] < 0) * 1)))
print('{0} times that SVC_prec-Dummy Recall was below 0'.format(
    sum((diff_df[diff_df['Contrast'] == 'SVC_prec-Dummy'][
        'Recall'] < 0) * 1)))
print('{0} times that RF-Dummy Recall was below 0 '.format(
    sum((diff_df[diff_df['Contrast'] == 'RF-Dummy'][
        'Recall'] < 0) * 1)))
# ===================================================================
plt.figure(3)
gs3 = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

ax31 = plt.subplot(gs3[0])
ax31.axhline(0.5, linestyle='--', color='red', alpha=0.4)
pypet.viz.plot_values(df, values='Precision',
                      target='Classifier', axes=ax31,
                      classes=['RF', 'SVC_prec', 'SVC_rec', 'Dummy'],
                      style='violinplot')
yposlist = df.groupby(['Classifier']).mean()
xposlist = range(len(yposlist))
stringlist = [rf_prec, svc_prec_prec, svc_rec_prec, Dum_prec]
for i in range(len(stringlist)):
    ax31.text(xposlist[i] + .1,
              stringlist[i]+ .1, format(round(stringlist[i], 2)))
ax32 = plt.subplot(gs3[1])
sns.boxplot(data=diff_df, x='Precision', y='Contrast', ax=ax32)
ax32.axvline(0, linestyle='--', color='red', alpha=0.4)

print('{0} times that SVC_rec-Dummy Precision was below 0'.format(
    sum((diff_df[diff_df['Contrast'] == 'SVC_rec-Dummy'][
        'Precision'] < 0) * 1)))
print('{0} times that SVC_prec-Dummy Precision was below 0'.format(
    sum((diff_df[diff_df['Contrast'] == 'SVC_prec-Dummy'][
        'Precision'] < 0) * 1)))
print('{0} times that RF-Dummy Precision was below 0 '.format(
    sum((diff_df[diff_df['Contrast'] == 'RF-Dummy'][
        'Precision'] < 0) * 1)))
plt.show()
