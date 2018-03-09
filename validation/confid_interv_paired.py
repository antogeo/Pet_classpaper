# plot scores
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypet
import seaborn as sns
import matplotlib.gridspec as gridspec

%matplotlib


df = pd.read_csv('corrected_boot_1000.csv')

Diff = dict()
Diff['iter'] = []
Diff['RF_AUC'] = []
Diff['RF_Rec'] = []
Diff['RF_Prec'] = []
Diff['SVM40_10_AUC'] = []
Diff['SVM40_10_Rec'] = []
Diff['SVM40_10_Prec'] = []
Diff['SVM10_26_AUC']= []
Diff['SVM10_26_Rec'] = []
Diff['SVM10_26_Prec'] = []

for iter in np.unique(df['Iteration']):
    Diff['iter'].append(iter)
    ###################### RF 10 - .7  #############################
    Diff['RF_AUC'].extend(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'RF_w'), 'AUC'].values - \
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'AUC'].values)
    Diff['RF_Rec'].extend(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'RF_w'), 'Recall'].values - \
        df.loc[(df['Iteration'] == iter) & (
        df['Classifier'] == 'Dummy'), 'Recall'].values)
    Diff['RF_Prec'].extend(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'RF_w'), 'Precision'].values - \
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'Precision'].values)
    ###################### SVM 40 - 10  #############################
    Diff['SVM40_10_AUC'].extend(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'SVC_fs_W40_10'), 'AUC'].values - \
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'AUC'].values)
    Diff['SVM40_10_Rec'].extend(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'SVC_fs_W40_10'), 'Recall'].values - \
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'Recall'].values)
    Diff['SVM40_10_Prec'].extend(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'SVC_fs_W40_10'), 'Precision'].values - \
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'Precision'].values)
    ###################### SVM 10 - 26  #############################
    Diff['SVM10_26_AUC'].extend(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'SVC_fs_W10_26'), 'AUC'].values - \
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'AUC'].values)
    Diff['SVM10_26_Rec'].extend(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'SVC_fs_W10_26'), 'Recall'].values - \
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'Recall'].values)
    Diff['SVM10_26_Prec'].extend(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'SVC_fs_W10_26'), 'Precision'].values - \
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'Precision'].values)

res = pd.DataFrame(Diff)
# res.to_csv('DummyVSclfs.csv')

gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[4, 1])

test = sns.boxplot(x=target, y=t_value, data=res, showcaps=False,
            boxprops={'facecolor':'None'},
            showfliers=False,whiskerprops={'linewidth':0},
            ax=t_ax, order=classes)

class_res = ['RF_AUC', 'RF_Rec', 'RF_Prec']

test = sns.violinplot(data=res, x=['RF_AUC', 'RF_Rec', 'RF_Prec'] ,y=['RF_AUC', 'RF_Rec', 'RF_Prec'])

fig_mean, axes = pypet.viz.plot_values(
    res,
    values=['iter'],
    target='Classifier',
    classes=['SVC_fs_W40_10', 'SVC_fs_W10_26', 'RF_w'])
plt.show()
