import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pypet

df = pd.read_csv(
    '/home/antogeo/codes/PET_class/result_dfs/feature_eval_SVC.csv')

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1})
sns.set_style("ticks")

g = sns.PairGrid(df, y_vars=["AUC", "Precision", "Recall"],
                 x_vars="perc",
                 size=8, aspect=1)

g.map(sns.pointplot, color=sns.xkcd_rgb["olive green"])
g.set(ylim=(0, 1))
sns.despine(fig=g.fig, left=True)
plt.show()
fig1 = plt.gcf()
fig1.savefig('feature_eval_plot.pdf')
