import os
import os.path as op
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/codes/Pet_classpaper/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/antogeo/git_codes/Pet_classpaper/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = 'Add git repo'

df = pd.read_csv(op.join(db_path, 'group_results_SUV/Liege_weight_eval.csv'))

classif = ['SVC_fs20p', 'RF_w']

fig, ax = plt.subplots(3, 1)
paper_rc = {'lines.linewidth': .6, 'lines.markersize': 6}
sns.set_context("paper", rc=paper_rc)
colors = sns.color_palette("Paired")
df['Weight Val'] = df['Weight Val'].round(4)
sns.pointplot(x="Weight Val", y="Recall", hue="Classifier", data=df, ax=ax[0],
                hue_order=classif, palette=colors[0:2])
sns.pointplot(x="Weight Val", y="Precision", hue="Classifier", data=df, ax=ax[1],
                hue_order=classif, palette=colors[2:4])
sns.pointplot(x="Weight Val", y="AUC", hue="Classifier", data=df, ax=ax[2],
                hue_order=classif, palette=colors[4:6])

ax[0].set(xticklabels=[])
ax[1].set(xticklabels=[])
ax[2].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
ax[2].tick_params(axis='x', direction='out', length=3, width=1, grid_color='r',
                  labelrotation=90, grid_alpha=0.5)
# # ax.set_aspect('equal', axis='x', adjustable='datalim')
# # ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
# ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%2.2f'))
plt.show()
