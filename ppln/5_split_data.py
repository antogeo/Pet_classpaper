import os
import os.path as op
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from pypet.constants import CLASSES

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

groups = ['Liege']
for group in groups:
    df = pd.read_csv(op.join(db_path, group, 'group_results_SUV',
                             group + '_db_GM_masks_full_atlas.csv'),
                     index_col=False)
    good_patients = df.query('QC_PASS == 1')

    good_patients = good_patients.replace(
            {'Final diagnosis (behav)': {'MCS*': 'UWS'}})

    y = np.zeros(len(good_patients))
    for i_class, klass in enumerate(CLASSES):
        mask = good_patients['Final diagnosis (behav)'].values == klass
        print('{} ({}) = {}'.format(klass, i_class, np.sum(mask)))
        y[mask] = i_class + 1

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)

X = np.zeros_like(y)

train, test = list(sss.split(X, y))[0]

validation_subjects = good_patients['Code'].values[test]
validation_subjects = [x for x in validation_subjects]
df['ML_VALIDATION'] = df['Code'].isin(validation_subjects)

df = df.set_index('Code')

old_df = pd.read_excel(meta_fname)
old_df = old_df.set_index('Code')

df = df['ML_VALIDATION']

old_df = old_df.join(df)

old_df.to_csv(meta_fname.replace('.xlsx', '_2.csv'))
