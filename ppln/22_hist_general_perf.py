import os
import os.path as op
import seaborn as sns
import pandas as pd

# load dataset
if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/dox/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

metrics = ['AUC', 'Recall', 'Precision']
x_lowlim = [0.6, .3, .6]
df_val = pd.read_csv(op.join(db_path, 'Liege', 'group_results_SUV',
                 'perf_estim_1000iter_f10_AAL90_ctrlout.csv'))
df_gen = pd.read_csv(op.join(db_path, 'Paris', 'group_results_SUV',
                 'gen_perf_estim_bs_f10_AAL90.csv'))
for count, metric in enumerate(metrics):
    fig, axes = plt.subplots(3, 1, figsize=(4,6))
    sns.distplot(bins=100, a=df_val[df_val['Classifier'] == 'RF']
                 [metric], label='Liege AUC',
                 rug=True, hist=True, color = 'b', ax=axes[0])
    sns.distplot(bins=100, a=df_gen[df_gen['Classifier'] == 'RF']
                 [metric], label='Paris AUC',
                 rug=True, hist=True, color = 'r', ax=axes[0])
    # axes[0].axvline(df_val[df_val['Classifier'] == 'RF']
    #                [metric].mean(), color='b')
    # axes[0].axvline(df_gen[df_gen['Classifier'] == 'RF']
    #                [metric].mean(), color='r')
    axes[0].set_xlabel('RF ' + metric + ' for Validation and Paris datasets')
    axes[0].set_xlim(x_lowlim[count], 1.05)

    sns.distplot(bins=100, a=df_val[df_val['Classifier'] == 'SVC_rec']
                 [metric], label='Liege AUC',
                 rug=True, hist=True, color = 'b', ax=axes[1])
    sns.distplot(bins=100, a=df_gen[df_gen['Classifier'] == 'SVC_rec']
                 [metric], label='Paris AUC',
                            rug=True, hist=True, color = 'r', ax=axes[1])
    # axes[1].axvline(df_val[df_val['Classifier'] == 'SVC_rec']
    #                [metric].mean(), color='b')
    # axes[1].axvline(df_gen[df_gen['Classifier'] == 'SVC_rec']
    #                [metric].mean(), color='r')
    axes[1].set_xlabel('SVMrec ' + metric + ' for Validation and Paris datasets')
    axes[1].set_xlim(x_lowlim[count], 1.05)

    sns.distplot(bins=100, a=df_val[df_val['Classifier'] == 'SVC_prec']
                 [metric], label='Liege AUC',
                 rug=True, hist=True, color = 'b', ax=axes[2])
    sns.distplot(bins=100, a=df_gen[df_gen['Classifier'] == 'SVC_prec']
                 [metric], label='Paris AUC',
                 rug=True, hist=True, color = 'r', ax=axes[2])
    # axes[2].axvline(df_val[df_val['Classifier'] == 'SVC_prec']
    #                [metric].mean(), color='b')
    # axes[2].axvline(df_gen[df_gen['Classifier'] == 'SVC_prec']
    #                [metric].mean(), color='r')
    axes[2].set_xlabel('SVMprec ' + metric + ' for Validation and Paris datasets')
    axes[2].set_xlim(x_lowlim[count], 1.05)
