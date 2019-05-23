import numpy as np
import pandas as pd
import os
import os.path as op
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import f_classif, SelectKBest
from collections import OrderedDict
from sklearn.preprocessing import minmax_scale
import nibabel as nib
import nilearn.plotting as plotting
import seaborn as sns
import matplotlib.pyplot as plt
import re

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

# load dataset
if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/dox/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

group = 'Paris'

df = pd.read_csv(op.join(db_path, 'Liege', 'group_results_SUV',
                 'Liege' + '_db_GM_AAL.csv'))
gen_df = pd.read_csv(op.join(db_path, group, 'group_results_SUV',
                 group + '_db_GM_AAL_nocereb.csv'))
df_train = df.query('QC_PASS == True and ML_VALIDATION == False')
df_gen = gen_df.query('QC_PASS == True and ML_gener == True')

markers = [x for x in df_train.columns if 'aal' in x]
markers = sorted_nicely(markers)
X_train = df_train[markers].values
y_train = 1 - (df_train[
    'Final diagnosis (behav)'] == 'VS').values.astype(np.int)

X_test = df_gen[markers].values
y_test = 1 - (df_gen[
    'Final diagnosis (behav)'] == 'VS').values.astype(np.int)

roi_names = pd.read_csv(op.join(db_path, 'Liege', 'extra',
                        'aal_rois.csv'))

select = SelectKBest(f_classif, 10)
select.fit(X_train, y_train)
feat_sel = dict()
for i in range(10):
    name = 'reg_' + str(i)
    feat_sel[name] = select.transform(X_train)[:, i]
feat_sel['label'] = y_train
feat_sel['site'] = ['Liege'] * len(y_train)

for i in range(10):
    name = 'reg_' + str(i)
    feat_sel[name] = np.concatenate(
            (feat_sel[name], select.transform(X_test)[:, i]))
feat_sel['label'] = np.concatenate((feat_sel['label'], y_test))
feat_sel['site'] = np.concatenate((feat_sel['site'], ['Paris'] * len(y_test)))

df = pd.DataFrame(feat_sel)
df['label'] = df['label'].replace(0, 'UWS')
df['label'] = df['label'].replace(1, 'MCS')


selected_liege_feat = select.transform(X_train)
selected_paris_feat = select.transform(X_test)
rois = roi_names[:95]
selected_regions = rois[select.get_support()]
for reg in range(10):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    liege_mcs = selected_liege_feat[numpy.where(y_train == 1)[0]]
    liege_uws = selected_liege_feat[numpy.where(y_train == 0)[0]]
    paris_mcs = selected_paris_feat[numpy.where(y_test == 1)[0]]
    paris_uws = selected_paris_feat[numpy.where(y_test == 0)[0]]
    sns.distplot(bins=100, a=liege_mcs[:, reg], ax=axes[0], label='Liege MCS',
                 rug=True, hist=False, color = 'b')
    sns.distplot(bins=100, a=liege_uws[:, reg], ax=axes[0], label='Liege UWS',
                 rug=True, hist=False, color = 'r')
    axes[0].axvline(liege_mcs[:, reg].mean(), color='b')
    axes[0].axvline(liege_uws[:, reg].mean(), color='r')

    sns.distplot(bins=100, a=paris_mcs[:, reg], ax=axes[1], label='Paris MCS',
                 rug=True, hist=False, color = 'g')
    sns.distplot(bins=100, a=paris_uws[:, reg], ax=axes[1], label='Paris UWS',
                 rug=True, hist=False, color = 'y')
    axes[1].axvline(paris_mcs[:, reg].mean(), color='g')
    axes[1].axvline(paris_uws[:, reg].mean(), color='y')
    axes[0].set_xlim(0, 10)
    axes[1].set_xlim(0, 10)
    plt.xlabel('mean metabolic activity of ' + selected_regions.iloc[reg]['roi_name'])
    plt.show()


for reg in range(10):
    region = 'reg_' + str(reg)
    liege_mcs = selected_liege_feat[numpy.where(y_train == 1)[0]]
    liege_uws = selected_liege_feat[numpy.where(y_train == 0)[0]]
    paris_mcs = selected_paris_feat[numpy.where(y_test == 1)[0]]
    paris_uws = selected_paris_feat[numpy.where(y_test == 0)[0]]
    plt.figure()
    gs = sns.swarmplot(x="label", y=region, hue="site",
                  data=df, dodge=True)
    gs.axhline(y = liege_mcs[:, reg].mean(), xmin=0.08, xmax=0.22, color='r')
    gs.axhline(y = paris_mcs[:, reg].mean(), xmin=.28, xmax=.42, color='r')
    gs.axhline(y = liege_uws[:, reg].mean(), xmin=.58, xmax=.72, color='r')
    gs.axhline(y = paris_uws[:, reg].mean(), xmin=.78, xmax=.92, color='r')
    gs.set_ylim(0, 10)
    plt.title('mean metabolic activity of ' + selected_regions.iloc[reg]['roi_name'])
    plt.tight_layout()
    plt.show()
