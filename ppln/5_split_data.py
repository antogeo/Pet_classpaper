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
    meta_fname = op.join(db_path, group, 'extra', group + '_meta.xls')
    df = pd.read_excel(meta_fname, index_col=False)
    # patients = df.query('Final diagnosis (behav) != CTRL')
    patients = df.replace(
            {'Final diagnosis (behav)': {'MCS*': 'VS'}})

    y = np.zeros(len(patients))
    for i_class, klass in enumerate(CLASSES):
        mask = patients['Final diagnosis (behav)'].values == klass
        print('{} ({}) = {}'.format(klass, i_class, np.sum(mask)))
        y[mask] = i_class + 1

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)

    X = np.zeros_like(y)

    train, test = list(sss.split(X, y))[0]

    validation_subjects = patients['Code'].values[test]
    validation_subjects = [x for x in validation_subjects]
    df['ML_VALIDATION'] = df['Code'].isin(validation_subjects)
    df = df.set_index('Code')
    df = df['ML_VALIDATION']

    old_df = pd.read_excel(meta_fname)
    old_df = old_df.set_index('Code')

    old_df = old_df.join(df)

    old_df.to_csv(op.join(
                  db_path, group, 'extra', group + '_meta.csv'))
