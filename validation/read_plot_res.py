import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('group_results_SUV/50pc_10dp_bootstrap_1000.csv')

# fig, axes = plt.subplots(1, 3, figsize=(12, 6))
# sns.swarmplot(
#     x="Classifier", y="Recall", hue="Classifier", data=df, ax=axes[0])
# sns.swarmplot(
#     x="Classifier", y="Precision", hue="Classifier", data=df, ax=axes[1])
# sns.swarmplot(
#     x="Classifier", y="AUC", hue="Classifier", data=df, ax=axes[2])
# # plt.savefig('figures/model_compar.pdf')
# plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 6))
sns.violinplot(
    x="Classifier", y="Recall", hue="Classifier", data=df, ax=axes[0])
sns.violinplot(
    x="Classifier", y="Precision", hue="Classifier", data=df, ax=axes[1])
sns.violinplot(
    x="Classifier", y="AUC", hue="Classifier", data=df, ax=axes[2])
# plt.savefig('figures/model_compar_box.pdf')
plt.show()
