# plot scores
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypet

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
    Diff['iter'] = iter

    Diff['RF_AUC'].append(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'AUC'].values - \
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'RF_w'), 'AUC'].values)

    Diff['RF_Rec'].append(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'Recall'].values - \
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'RF_w'), 'Recall'].values)

    Diff['RF_Prec'].append(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'Precision'].values - \
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'RF_w'), 'Precision'].values)

    Diff['SVM40_10_AUC'].append(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'AUC'].values - \
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'SVC_fs_W40_10'), 'AUC'].values)

    Diff['SVM40_10_Rec'].append(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'Recall'].values - \
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'SVC_fs_W40_10'), 'Recall'].values)

    Diff['SVM40_10_Prec'].append(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'Precision'].values - \
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'SVC_fs_W40_10'), 'Precision'].values)

    Diff['SVM10_26_AUC'].append(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'AUC'].values - \
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'SVC_fs_W10_26'), 'AUC'].values)

    Diff['SVM10_26_Rec'].append(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'Recall'].values - \
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'SVC_fs_W10_26'), 'Recall'].values)

    Diff['SVM10_26_Prec'].append(
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'Precision'].values - \
        df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'SVC_fs_W10_26'), 'Precision'].values)
    res = pd.DataFrame(Diff)
    res.to_csv('DummyVSclfs.csv')
