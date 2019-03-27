import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

df = pd.read_csv('./group_results_SUV/weights_eval_20190326.csv')

# fig, axes = plt.subplots(3, 1, figsize=(12, 6))
# sns.swarmplot(
#     x="Classifier", y="Recall", hue="Classifier", data=df, ax=axes[0])
# sns.swarmplot(
#     x="Classifier", y="Precision", hue="Classifier", data=df, ax=axes[1])
# sns.swarmplot(
#     x="Classifier", y="AUC", hue="Classifier", data=df, ax=axes[2])
# # plt.savefig('figures/model_compar.pdf')
# plt.show()

res = dict()
res['weights'] = []
res['mean rec'] = []
res['mean pre'] = []
for weight in np.unique(df['Weight Val']):
    res['weights'].append(weight)
    res['mean rec'].append(np.mean(df.loc[(df['Classifier'] == 'SVC_fs10p') & (
        df['Weight Val'] == weight)]['Recall']))
    res['mean pre'].append(np.mean(df.loc[(df['Classifier'] == 'SVC_fs10p') & (
        df['Weight Val'] == weight)]['Precision']))
results = pd.DataFrame(res)

line1 = results.loc[np.abs(results['mean rec'] - .95).idxmin(), 'weights']
line2 = results.loc[np.abs(results['mean pre'] - .75).idxmin(), 'weights']

fig, axes = plt.subplots(3, 1, figsize=(12, 6))

sns.lineplot(
    x="Weight Val", y="Recall", hue="Classifier", data=df[df[
        'Classifier'] != 'SVC_fs20p'], ax=axes[0])
plt.xlabel('MCS/UWS (Weights Ratio)')

axes[0].axvline(line1, color='red')

sns.lineplot(
    x="Weight Val", y="Precision", hue="Classifier", data=df[df[
        'Classifier'] != 'SVC_fs20p'], ax=axes[1])

axes[1].axvline(line2, color='red')

sns.lineplot(
    x="Weight Val", y="AUC", hue="Classifier", data=df[df[
        'Classifier'] != 'SVC_fs20p'], ax=axes[2])

axes[0].set_xlabel('MCS/UWS (Weights Ratio)')
axes[1].set_xlabel('MCS/UWS (Weights Ratio)')
axes[2].set_xlabel('MCS/UWS (Weights Ratio)')

axes[0].xaxis.set_major_locator(ticker.MultipleLocator(.2))
axes[1].xaxis.set_major_locator(ticker.MultipleLocator(.2))
axes[2].xaxis.set_major_locator(ticker.MultipleLocator(.2))
# plt.savefig('figures/model_compar_box.pdf')
plt.show()
