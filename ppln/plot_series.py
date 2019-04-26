import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

df = pd.read_csv('./group_results_SUV/weights_eval_nAAL20.csv')

res = dict()
res['weights'] = []
res['mean rec'] = []
res['mean pre'] = []
for weight in np.unique(df['Weight Val']):
    res['weights'].append(weight)
    res['mean rec'].append(np.mean(df.loc[(df['Classifier'] == 'SVC_fs20') & (
        df['Weight Val'] == weight)]['Recall']))
    res['mean pre'].append(np.mean(df.loc[(df['Classifier'] == 'SVC_fs20') & (
        df['Weight Val'] == weight)]['Precision']))
results = pd.DataFrame(res)

line1 = results.loc[np.abs(results['mean rec'] - .95).idxmin(), 'weights']
line2 = results.loc[np.abs(results['mean pre'] - .9).idxmin(), 'weights']

fig, axes = plt.subplots(3, 1, figsize=(12, 6))

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
    i.set_xlim(0, 10)
    i.xaxis.set_major_locator(ticker.MultipleLocator(.2))
    i.legend(('SVC', 'Random Forest', 'Dummy'), title="Classifier")

# plt.savefig('./group_results_SUV/weights_nAAL_compar_box.pdf')
plt.show()
