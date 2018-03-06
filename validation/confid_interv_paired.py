# plot scores
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypet

df = pd.read_csv('corrected_boot_1000.csv')

for iter in np.unique(df['Iteration']):
        SVM1_AUC = df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'AUC'].values
        SVM2_Rec = df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'Recall'].values
        RF_Prec = df.loc[(df['Iteration'] == iter) & (
            df['Classifier'] == 'Dummy'), 'Precision'].values
