import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../PET_class/scratch/weights_eval.csv')

fig, ax = plt.subplots(1, 1)
sns.pointplot(x="Weight Val", y="Recall", hue="Classifier", data=df, ax=ax)
sns.pointplot(x="Weight Val", y="Precision", hue="Classifier", data=df, ax=ax)
sns.pointplot(x="Weight Val", y="AUC", hue="Classifier", data=df, ax=ax)
plt.show()
