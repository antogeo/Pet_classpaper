import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('models_eval_90pcfeat.csv')

fig, axes = plt.subplots(1, 3, figsize=(12, 6))
sns.violinplot(
    x="Classifier", y="Recall", hue="Classifier", data=df, ax=axes[0])
sns.violinplot(
    x="Classifier", y="Precision", hue="Classifier", data=df, ax=axes[1])
sns.violinplot(
    x="Classifier", y="AUC", hue="Classifier", data=df, ax=axes[2])
plt.savefig('model_eval/figures/model_compar90pcfeatViolin.pdf')
plt.show()

fig2, axes2 = plt.subplots(1, 3, figsize=(12, 6))
sns.swarmplot(
    x="Classifier", y="Recall", hue="Classifier", data=df, ax=axes2[0])
sns.swarmplot(
    x="Classifier", y="Precision", hue="Classifier", data=df, ax=axes2[1])
sns.swarmplot(
    x="Classifier", y="AUC", hue="Classifier", data=df, ax=axes2[2])
plt.savefig('model_eval/figures/model_compar90pcfeat.pdf')
plt.show()
