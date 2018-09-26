import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/home/antogeo/codes/Pet_classpaper/group_results_SUV/ \
    Liege_weight_eval.csv')


classif = ['SVC_fs20p', 'SVC_fs10p', 'RF_w']

fig, ax = plt.subplots(1, 1)
paper_rc = {'lines.linewidth': .6, 'lines.markersize': 6}
sns.set_context("paper", rc=paper_rc)
df['Weight Val'] = df['Weight Val'].round(4)
sns.pointplot(x="Weight Val", y="Recall", hue="Classifier", data=df, ax=ax,
                hue_order=classif)
sns.pointplot(x="Weight Val", y="Precision", hue="Classifier", data=df, ax=ax,
                hue_order=classif)
sns.pointplot(x="Weight Val", y="AUC", hue="Classifier", data=df, ax=ax,
                hue_order=classif)

ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
ax.tick_params(axis='x', direction='out', length=3, width=1, grid_color='r',
               labelrotation=90, grid_alpha=0.5)
# # ax.set_aspect('equal', axis='x', adjustable='datalim')
# # ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
# ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%2.2f'))
