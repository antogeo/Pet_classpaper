import numpy as np
import pandas as pd
import os
import os.path as op

import itertools

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, classes, labels,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel(labels[0])
    plt.xlabel(labels[1])


df = pd.read_csv('predictions.csv',
                 index_col=['Code', 'Label'])[['Classifier', 'Prediction']]

df = df.pivot(columns='Classifier')
df.columns = [x[1] for x in df.columns]
df = df.reset_index().set_index('Code')

class_names = ['UWS', 'MCS']
to_plot = [
    ('Label', 'SVC_fs_W10_26'),
    ('Label', 'SVC_fs_W40_10'),
    ('Label', 'RF_w'),
    ('SVC_fs_W10_26', 'SVC_fs_W40_10'),
    ('SVC_fs_W10_26', 'RF_w'),
    ('SVC_fs_W40_10', 'RF_w'),
]
for c1, c2 in to_plot:
    y1 = df[c1].values
    y2 = df[c2].values
    cnf_matrix = confusion_matrix(y1, y2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, labels=[c1, c2],
        title='Confusion matrix, without normalization')

plt.show()

t_df = df.query('SVC_fs_W10_26 == 1 and SVC_fs_W40_10 == 0')
y1 = t_df['Label'].values
y2 = t_df['RF_w'].values
cnf_matrix = confusion_matrix(y1, y2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, labels=['Label', 'RF_w'],
    title='Middle Zone')

plt.show()
