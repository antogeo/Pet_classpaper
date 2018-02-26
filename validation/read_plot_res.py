import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('validation/boot_1000.csv')

fig, axes = plt.subplots(1, 3, figsize=(12, 6))
sns.swarmplot(
    x="Classifier", y="Recall", hue="Classifier", data=df, ax=axes[0])
sns.swarmplot(
    x="Classifier", y="Precision", hue="Classifier", data=df, ax=axes[1])
sns.swarmplot(
    x="Classifier", y="AUC", hue="Classifier", data=df, ax=axes[2])
# plt.savefig('figures/model_compar.pdf')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 6))
<<<<<<< HEAD
sns.boxplot(
    x="Classifier", y="Recall", hue="Classifier", data=df, ax=axes[0])
sns.boxplot(
    x="Classifier", y="Precision", hue="Classifier", data=df, ax=axes[1])
sns.boxplot(
    x="Classifier", y="AUC", hue="Classifier", data=df, ax=axes[2])
# plt.savefig('/model_comparbox.pdf')
=======
sns.boxplot(x="Classifier", y="Recall", hue="Classifier", data=df, ax=axes[0])
sns.boxplot(x="Classifier", y="Precision", hue="Classifier", data=df, ax=axes[1])
sns.boxplot(x="Classifier", y="AUC", hue="Classifier", data=df, ax=axes[2])
plt.savefig('figures/model_compar_box.pdf')
>>>>>>> 132734a5bd4429f9dd7baf3cd373f70dcd851aea
plt.show()
