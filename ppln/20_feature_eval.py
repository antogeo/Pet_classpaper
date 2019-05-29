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
    db_path = '/home/antogeo/dox/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/dox/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

group = 'Liege'

df = pd.read_csv(op.join(db_path, group, 'group_results_SUV',
                 group + '_db_GM_AAL.csv'))
df_train = df.query('QC_PASS == True and ML_VALIDATION == False')

classifiers = OrderedDict()

classifiers['SVC'] = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectKBest(f_classif, 10)),
        ('clf', SVC(kernel="linear", C=1, probability=True,
                    class_weight={0: 1, 1: 2.4}))
    ])
classifiers['RF'] = Pipeline([
    ('scaler', RobustScaler()),
    ('clf', RandomForestClassifier(
        max_depth=5, n_estimators=2000, max_features='auto',
        class_weight={0: 1, 1: 1}))
])
classifiers['XRF'] = Pipeline([
    ('scaler', RobustScaler()),
    ('clf', ExtraTreesClassifier(
        max_depth=5, n_estimators=2000, max_features='auto',
        class_weight={0: 1, 1: 1}))
])

markers = [x for x in df_train.columns if 'aal' in x]
markers = sorted_nicely(markers)

X_train = df_train[markers].values
y_train = 1 - (df_train[
    'Final diagnosis (behav)'] == 'VS').values.astype(np.int)


roi_names = pd.read_csv(op.join(db_path, group, 'extra',
                        'aal_rois.csv'))

classifiers['SVC'].fit(X_train, y_train)
classifiers['RF'].fit(X_train, y_train)
classifiers['XRF'].fit(X_train, y_train)


roi_names['roi_name'][0:95].values[classifiers['SVC'].named_steps['select'].get_support()]

results = dict()
results['rois'] = roi_names['roi_name'][0:95].values
results['fscore'] = minmax_scale(-np.log10(
    classifiers['SVC'].named_steps['select'].pvalues_),  feature_range=(0, 1))
results['RFimportances'] = minmax_scale(classifiers[
    'RF'].named_steps['clf'].feature_importances_,  feature_range=(0, 1))
results['XRFimportances'] = minmax_scale(classifiers[
    'XRF'].named_steps['clf'].feature_importances_,  feature_range=(0, 1))


df_f = pd.DataFrame(results)

df = pd.melt(df_f, id_vars="rois", var_name="classifiers",
             value_name="importances")
axes = sns.factorplot(x='rois', y='importances', hue='classifiers',
                      data=df, kind='bar')

axes.set_xticklabels(results['rois'], rotation=90)

roi_names['roi_name'][0:95]


# FOR SVM
svm_reg_name = roi_names['roi_name'][0:95].values[selector.get_support()]
svm_reg_values = selector.pvalues_[selector.get_support()]
plt.figure()
plt.ylim(0, 1.05)
plt.bar(roi_names['roi_name'][0:95].values, minmax_scale(-np.log10(
    selector.pvalues_), feature_range=(0, 1)), 0.35, label='10 best rois')
plt.xticks(rotation='vertical')


atlas_nii = nib.load(op.join(db_path, group, 'extra',
                     'AAL_noCereb.nii'))
vol = atlas_nii.get_data()
new_vol = np.zeros_like(vol)
for val in np.unique(vol):
    print(int(val.round()))
    print(results['RFimportances'][int(val.round())])
    if classifiers['SVC'].named_steps['select'].get_support()[
        int(val.round())]==True:
        new_vol[vol == val] = results['fscore'][int(val.round())]

new_header = atlas_nii.header.copy()
new_image = nib.Nifti1Image(new_vol, atlas_nii.affine, header=new_header)
new_image.to_filename(op.join(
    db_path, group, 'group_results_SUV', 'f_scores.nii'))
plotting.plot_anat(new_image, title='test')
