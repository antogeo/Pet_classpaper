import os
import os.path as op
import pandas as pd
from pypet.features import compute_regional_features

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/dox/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

results = 'group_results_SUV'
folders = ['Liege']
all_df = []
for folder in folders:
    meta_fname = op.join(db_path, folder, 'extra', folder + '_meta.csv')
    s_path = op.join(db_path, folder)
    all_df = compute_regional_features(s_path, meta_fname, ftype='s8SUV')
    print('{} {}'.format(s_path, meta_fname))
    dframe = pd.DataFrame(all_df)
    dframe = dframe.reset_index(drop=True)
    dframe.to_csv(op.join(
        db_path, folder, results, folder + '_db_GM_AAL.csv'))
