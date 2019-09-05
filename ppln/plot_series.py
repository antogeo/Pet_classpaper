import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import os
import os.path as op

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/dox/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/dox/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

group = 'Liege'

df = pd.read_csv(op.join(db_path, group,
                 'group_results_SUV', 'weights_eval_AAL_XRF_kfold.csv'))
df = df[df['Classifier'] != 'SVC_fs20']
df = df[df['Classifier'] != 'XRF']
res = dict()
res['weights'] = []
res['mean rec'] = []
res['mean pre'] = []
for weight in np.unique(df['Weight Val']):
    res['weights'].append(weight)
    res['mean rec'].append(np.mean(df.loc[(df['Classifier'] == 'SVC_fs10') & (
        df['Weight Val'] == weight)]['Recall']))
    res['mean pre'].append(np.mean(df.loc[(df['Classifier'] == 'SVC_fs10') & (
        df['Weight Val'] == weight)]['Precision']))
results = pd.DataFrame(res)

line1 = results.loc[np.abs(results['mean rec'] - .95).idxmin(), 'weights']
line2 = results.loc[np.abs(results['mean pre'] - .9).idxmin(), 'weights']

fig, axes = plt.subplots(3, 1, figsize=(12, 6))
sns.set(font_scale=1.02)


sns.lineplot(
    x="Weight Val", y="Recall", hue="Classifier", data=df, ax=axes[0])
plt.xlabel('MCS/UWS (Weights Ratio)')

axes[0].axvline(line1, color='red')

sns.lineplot(
    x="Weight Val", y="Precision", hue="Classifier", data=df, ax=axes[1])

axes[1].axvline(line2, color='red')

sns.lineplot(
    x="Weight Val", y="AUC", hue="Classifier", data=df, ax=axes[2])
for i in axes:
    i.set_xlabel('MCS/UWS (Weights Ratio)')
    i.set_xlim(0.1, 10)
    i.set_ylim(0.4, 1.05)
    i.xaxis.set_major_locator(ticker.MultipleLocator(.4))
    i.set_xscale('log')
    i.set_xticks([0.1, line2, 1, line1, 10])
    i.tick_params(axis='x', direction='out', length=3, width=1, grid_color='r',
                  labelrotation=90, grid_alpha=0.5)
    # i.xticks(x, labels, rotation='vertical')
    # i.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter('%.1f'))
    i.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    i.legend(('SVM', 'Dummy'), title="Classifier",
             bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    i.axvline(1, linestyle='dashed', color='red')
# plt.savefig('./group_results_SUV/weights_nAAL_compar_box.pdf')
plt.show()
