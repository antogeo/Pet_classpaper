import pandas as pd
import matplotlib.pyplot as plt
import pypet
import os
import os.path as op
import seaborn as sns
import matplotlib.gridspec as gridspec

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/dox/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/dox/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'
group = 'Liege'

df = pd.read_csv(op.join(db_path, group, 'group_results_SUV',
                 'perf_estim_1000iter_f10_AAL90_Xrf5.csv'),
                 index_col=['Iteration', 'Classifier'])[
                    ['AUC', 'Precision', 'Recall']]

df = df.reset_index()

contr_df = pd.read_csv(op.join(db_path, group, 'group_results_SUV',
                       'Clfs_contrast_f10_AAL90_Xrf5.csv'))
diff_df = contr_df.loc[contr_df['Contrast'].str.contains("-Dummy")]

xrf_auc = df[df['Classifier'] == 'XRF']['AUC'].mean()
xrf_rec = df[df['Classifier'] == 'XRF']['Recall'].mean()
xrf_prec = df[df['Classifier'] == 'XRF']['Precision'].mean()
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

ax11 = plt.subplot(gs[0])

pypet.viz.plot_values(df, values='AUC',
                      target='Classifier', axes=ax11,
                      classes=['XRF', 'SVC_prec', 'SVC_rec', 'Dummy'],
                      style='violinplot')

yposlist = df.groupby(['Classifier']).mean()
xposlist = range(len(yposlist))
stringlist = [xrf_auc, svc_prec_auc, svc_rec_auc, Dum_auc]
for i in range(len(stringlist)):
    ax11.text(xposlist[i] + .1,
              stringlist[i] + .1, format(round(stringlist[i], 2)))

ax11.axhline(.5, linestyle='--', color='red', alpha=0.4)
ax12 = plt.subplot(gs[1])

sns.boxplot(data=diff_df, x='AUC', y='Contrast', ax=ax12)
print('{0} times that SVC_rec-Dummy AUC was below 0'.format(
    sum((diff_df[diff_df['Contrast'] == 'SVC_rec-Dummy'][
        'AUC'] < 0) * 1)))
print('{0} times that SVC_prec-Dummy AUC was below 0'.format(
    sum((diff_df[diff_df['Contrast'] == 'SVC_prec-Dummy'][
        'AUC'] < 0) * 1)))
print('{0} times that RF-Dummy AUC was below 0 '.format(
    sum((diff_df[diff_df['Contrast'] == 'XRF-Dummy'][
        'AUC'] < 0) * 1)))

ax12.axvline(0, linestyle='--', color='red', alpha=0.4)
# ==================================================================
plt.figure(2)
gs2 = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

ax21 = plt.subplot(gs2[0])

pypet.viz.plot_values(df, values='Recall',
                      target='Classifier', axes=ax21,
                      classes=['XRF', 'SVC_prec', 'SVC_rec', 'Dummy'],
                      style='violinplot')

yposlist = df.groupby(['Classifier']).mean()
xposlist = range(len(yposlist))
stringlist = [xrf_rec, svc_prec_rec, svc_rec_rec, Dum_rec]
for i in range(len(stringlist)):
    ax21.text(xposlist[i] + .1,
              stringlist[i] + .1, format(round(stringlist[i], 2)))
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
    sum((diff_df[diff_df['Contrast'] == 'XRF-Dummy'][
        'Recall'] < 0) * 1)))
# ===================================================================
plt.figure(3)
gs3 = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

ax31 = plt.subplot(gs3[0])
ax31.axhline(0.5, linestyle='--', color='red', alpha=0.4)
pypet.viz.plot_values(df, values='Precision',
                      target='Classifier', axes=ax31,
                      classes=['XRF', 'SVC_prec', 'SVC_rec', 'Dummy'],
                      style='violinplot')
yposlist = df.groupby(['Classifier']).mean()
xposlist = range(len(yposlist))
stringlist = [xrf_prec, svc_prec_prec, svc_rec_prec, Dum_prec]
for i in range(len(stringlist)):
    ax31.text(xposlist[i] + .1,
              stringlist[i] + .1, format(round(stringlist[i], 2)))
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
    sum((diff_df[diff_df['Contrast'] == 'XRF-Dummy'][
        'Precision'] < 0) * 1)))
plt.show()

ax11.set_ylim(.3, 1.1)
ax12.set_xlim(-.5, .7)
ax21.set_ylim(.3, 1.1)
ax22.set_xlim(-.5, .7)
ax31.set_ylim(.3, 1.1)
ax32.set_xlim(-.5, .7)
