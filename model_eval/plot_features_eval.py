import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pypet

df = pd.read_csv(
    '/home/antogeo/codes/PET_class/result_dfs/feature_eval_SVC.csv')

g = sns.PairGrid(df, y_vars=["AUC", "Precision", "Recall"],
                 x_vars="perc",
                 size=5, aspect=.5)

g.map(sns.pointplot, color=sns.xkcd_rgb["plum"])
g.set(ylim=(0, 1))
sns.despine(fig=g.fig, left=True)
