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


# load dataset
if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/dox/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/dox/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

group = 'Liege'

df = pd.read_csv(op.join(db_path, group, 'group_results_SUV',
                 group + '_db_GM_AAL_nocereb.csv'))
df_train = df.query('QC_PASS == True and ML_VALIDATION == False')

classifiers = OrderedDict()

classifiers['SVC'] = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectKBest(f_classif, 94)),
        ('clf', SVC(kernel="linear", C=1, probability=True,
                    class_weight={0: 1, 1: 1}))
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
X_train = df_train[markers].values
y_train = 1 - (df_train[
    'Final diagnosis (behav)'] == 'VS').values.astype(np.int)


roi_names = pd.read_csv(op.join(db_path, group, 'extra',
                        'aal_rois.csv'))
classifiers['SVC'].fit(X_train, y_train)
classifiers['SVC'].named_steps['select'].get_support(indices=True)
classifiers['RF'].fit(X_train, y_train)
classifiers['XRF'].fit(X_train, y_train)


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

atlas_nii = nib.load(op.join(db_path, group, 'extra',
                     'AAL_noCereb.nii'))
vol = atlas_nii.get_data()

for val in np.unique(vol):
    print(int(val.round()))
    print(results['importances'][int(val.round())])
    vol[vol == val] = results['importances'][int(val.round())]

new_header = atlas_nii.header.copy()
new_image = nib.Nifti1Image(vol, atlas_nii.affine, header=new_header)
new_image.to_filename(op.join(
    db_path, group, 'group_results_SUV', 'feat_importance.nii'))
plotting.plot_anat(new_image, title='test')
