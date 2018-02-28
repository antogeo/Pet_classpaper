import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypet

df = pd.read_csv('predictions.csv')
color_lbl = df.loc[df['Classifier'] == 'RF_w', 'Label']

plt.figure()
plt.scatter(
    df.loc[df['Classifier'] == 'RF_w', 'Probability'],
    df.loc[df['Classifier'] == 'RF_w', 'Unnamed: 0'],
    s=80, c=color_lbl, marker="v")
plt.scatter(
    df.loc[df['Classifier'] == 'RF_w', 'Probability'],
    df.loc[df['Classifier'] == 'RF_w', 'Unnamed: 0'],
    s=40, c='yellow', marker="+")
plt.plot([.5, .5], [100, 160], 'k-', lw=1)
plt.show()
