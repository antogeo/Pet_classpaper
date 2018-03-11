import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypet

df = pd.read_csv('validation/predictions.csv')

color = ['y' if x == 1 else 'b' for x in df.loc[
    df['Classifier'] == 'RF_w', 'Label']]
shape = ['o' if x == 1 else 's' for x in df.loc[
    df['Classifier'] == 'SVC_fs_W10_26', 'Prediction']]
fill = ['x' if x == 1 else '+' for x in df.loc[
    df['Classifier'] == 'SVC_fs_W10_26', 'Prediction']]

fig, ax = plt.subplots()
for i, prob in enumerate(df.loc[df['Classifier'] == 'RF_w', 'Probability']):
    ax.scatter(prob, i, s=120, c=color[i], marker=shape[i])

for i, prob in enumerate(df.loc[df['Classifier'] == 'RF_w', 'Probability']):
    ax.scatter(prob, i, s=120, c='r', marker=fill[i])
ax.plot([.5, .5], [0, 50], 'k-', lw=1)
ax.legend()
ax.grid(True)
fig.show()
df.columns

fig2, ax2 = plt.subplots()
for ind, name in enumerate(np.unique(df['Code'])):
    color = ['b' if df.loc[df['Code'] == name, 'Label'].mean() == 1 else 'y']
    ax2.scatter(df.loc[df['Code'] == name, 'Probability'].mean(), ind,
                s=120, c=color, marker='o')
ax2.plot([.5, .5], [0, 50], 'k-', lw=1)
